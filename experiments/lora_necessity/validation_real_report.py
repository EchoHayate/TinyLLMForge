"""Bind D01 observations from a trusted controller, not an execution attestation."""

import copy
import hashlib
import json
import re

from pilot_protocol import FLAGS, fields, require
from validation_contract import build_schedule, digest, encoded, parse_json, worker_requests
from validation_report import parse_outcome


STATUSES = ("completed", "program_error", "timeout", "output_limit", "launch_failed")
TRANSPORT_STDOUT_LIMIT = 8388608
TRANSPORT_STDERR_LIMIT = 131072


def _hex_bytes(value, limit):
    require(type(value) is str and len(value) <= 2 * limit and len(value) % 2 == 0,
            "output hex byte budget")
    require(re.search(r"[^0-9a-fA-F]", value) is None, "invalid output hex")
    return bytes.fromhex(value)


def _process_result(result, *, outer=False):
    fields(result, ("status", "returncode", "stdout_hex", "stderr_hex", "elapsed_ns",
                    "process_reaped", "stdin_complete"))
    require(type(result["status"]) is str and result["status"] in STATUSES,
            "process status")
    require(result["returncode"] is None or type(result["returncode"]) is int,
            "returncode type")
    require(type(result["elapsed_ns"]) is int and result["elapsed_ns"] >= 0,
            "elapsed time")
    require(type(result["process_reaped"]) is bool
            and type(result["stdin_complete"]) is bool, "process flags")
    completed = result["status"] == "completed"
    overflow = 0 if completed else 1
    stdout_limit = TRANSPORT_STDOUT_LIMIT if outer else 65536
    stderr_limit = TRANSPORT_STDERR_LIMIT if outer else 4096
    stdout = _hex_bytes(result["stdout_hex"], stdout_limit + overflow)
    stderr = _hex_bytes(result["stderr_hex"], stderr_limit + overflow)
    if completed:
        require(result["returncode"] == 0 and result["process_reaped"]
                and result["stdin_complete"], "inconsistent completed process")
        require(outer or not stderr, "completed worker stderr")
    return stdout


def _events(raw, *, timed_out):
    lines = raw.split(b"\n")
    tail = lines.pop()
    events = [parse_json(line, limit=TRANSPORT_STDOUT_LIMIT + 1) for line in lines]
    truncated = False
    if tail:
        try:
            tail.decode("utf-8")
        except UnicodeDecodeError:
            require(timed_out, "invalid final JSONL encoding")
            truncated = True
        if not truncated:
            try:
                events.append(parse_json(tail, limit=TRANSPORT_STDOUT_LIMIT + 1))
            except json.JSONDecodeError:
                require(timed_out, "invalid final JSONL record")
                truncated = True
    return events, truncated


def _bundle(bundle_raw, source_raw, manifest_raw):
    bundle = parse_json(bundle_raw)
    fields(bundle, ("kind", "mode", "run_id", "worker_source", "requests"))
    require(bundle["kind"] == "LORA_REMOTE_VALIDATION_BUNDLE"
            and bundle["mode"] == "d01", "D01 bundle required")
    require(type(bundle["run_id"]) is str
            and re.fullmatch(r"[0-9a-f]{32}", bundle["run_id"]), "bundle run ID")
    require(type(bundle["worker_source"]) is str and bundle["worker_source"],
            "worker source text")
    require(type(bundle["requests"]) is list
            and encoded(bundle["requests"]) == encoded(
                worker_requests(source_raw, manifest_raw)[:32]),
            "frozen D01 request projection")
    try:
        worker_raw = bundle["worker_source"].encode("utf-8")
    except UnicodeError as error:
        raise ValueError("worker source encoding") from error
    return bundle, hashlib.sha256(worker_raw).hexdigest()


def _observations(events, planned_rows, run_id, bundle_sha256, worker_sha256):
    runtime = None
    done = None
    observed = []
    stop_ordinal = None
    for event in events:
        require(type(event) is dict, "event object")
        require(done is None, "event after done")
        require(event.get("run_id") == run_id, "event run binding")
        kind = event.get("kind")
        if kind == "ready":
            fields(event, ("kind", "run_id", "bundle_sha256", "worker_sha256", "runtime"))
            require(runtime is None and not observed, "ready must be first and unique")
            require(event["bundle_sha256"] == bundle_sha256
                    and event["worker_sha256"] == worker_sha256, "ready binding")
            require(type(event["runtime"]) is dict and event["runtime"],
                    "runtime metadata")
            runtime = event["runtime"]
        elif kind == "row":
            fields(event, ("kind", "run_id", "ordinal", "request_sha256", "result"))
            require(runtime is not None, "row before ready")
            require(stop_ordinal is None, "row after infrastructure stop")
            ordinal = event["ordinal"]
            require(type(ordinal) is int and ordinal == len(observed)
                    and 0 <= ordinal < 32, "row order")
            require(event["request_sha256"] == planned_rows[ordinal]["request_sha256"],
                    "request binding")
            raw = _process_result(event["result"])
            outcome = None
            if event["result"]["status"] == "completed":
                outcome = parse_outcome(raw)
            else:
                stop_ordinal = ordinal
            observed.append({"result": event["result"], "outcome": outcome})
        elif kind == "done":
            fields(event, ("kind", "run_id", "received", "elapsed_ns", "stopped"))
            require(runtime is not None, "done before ready")
            require(type(event["received"]) is int
                    and event["received"] == len(observed), "done count")
            require(type(event["elapsed_ns"]) is int and event["elapsed_ns"] >= 0,
                    "done elapsed time")
            require(type(event["stopped"]) is bool, "done stopped type")
            require(stop_ordinal is None or event["stopped"], "done ignored stop")
            done = event
        else:
            raise ValueError("unknown event kind")
    return runtime, done, observed, stop_ordinal


def build_d01_report(source_raw, manifest_raw, bundle_raw, transport):
    """Compare local frozen answers with receipts; never execute candidate source."""
    plan = build_schedule(source_raw, manifest_raw)
    planned_rows = plan["rows"][:32]
    require(len(planned_rows) == 32 and all(row["task_id"] == "D01"
                                          for row in planned_rows), "D01 schedule")
    bundle, worker_sha256 = _bundle(bundle_raw, source_raw, manifest_raw)
    bundle_sha256 = hashlib.sha256(bundle_raw).hexdigest()
    stdout = _process_result(transport, outer=True)
    events, truncated = _events(stdout, timed_out=transport["status"] == "timeout")
    runtime, done, observed, stop_ordinal = _observations(
        events, planned_rows, bundle["run_id"], bundle_sha256, worker_sha256)
    rows = []
    for planned in planned_rows:
        observation = observed[planned["ordinal"]] if planned["ordinal"] < len(
            observed) else None
        result = None if observation is None else observation["result"]
        outcome = None if observation is None else observation["outcome"]
        prediction = planned["witness_prediction"]
        rows.append({
            **{name: planned[name] for name in (
                "ordinal", "task_id", "role", "group", "case_id", "task_sha256",
                "source_sha256", "input_sha256", "expected_sha256", "request_sha256")},
            "status": "missing" if result is None else result["status"],
            "result": copy.deepcopy(result), "outcome": copy.deepcopy(outcome),
            "expected_match": None if outcome is None else
            encoded(outcome) == encoded(planned["expected"]),
            "prediction_match": None if outcome is None or prediction is None else
            encoded(outcome) == encoded(prediction),
        })
    references = [row for row in rows if row["role"] == "reference"]
    witnesses = [row for row in rows if
                 planned_rows[row["ordinal"]]["witness_prediction"] is not None]
    reference_ok = len(references) == 8 and all(
        row["expected_match"] is True for row in references)
    initial_scoring_rejected = any(
        row["role"] == "initial" and row["group"] == "scoring"
        and row["expected_match"] is False for row in rows)
    witnesses_ok = len(witnesses) == 3 and all(
        row["prediction_match"] is True and row["expected_match"] is False
        for row in witnesses)
    complete = (len(observed) == 32 and all(row["status"] == "completed" for row in rows)
                and done is not None and not done["stopped"]
                and transport["status"] == "completed" and not truncated)
    return {
        "kind": "LORA_D01_EXECUTION_REPORT", "version": 1,
        "run_id": bundle["run_id"], "source_sha256": plan["source_sha256"],
        "manifest_sha256": plan["manifest_sha256"], "plan_sha256": digest(plan),
        "bundle_sha256": bundle_sha256, "worker_sha256": worker_sha256,
        "runtime": copy.deepcopy(runtime),
        "runtime_sha256": None if runtime is None else digest(runtime),
        "transport": copy.deepcopy(transport), "transport_sha256": digest(transport),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "done": copy.deepcopy(done), "truncated_final_line": truncated,
        "full_scheduled": 1416, "batch_scheduled": 32, "received": len(observed),
        "remaining_full": 1416 - len(observed), "stop_ordinal": stop_ordinal,
        "rows": rows, "reference_ok": reference_ok,
        "initial_scoring_rejected": initial_scoring_rejected, "witnesses_ok": witnesses_ok,
        "batch_complete": complete,
        "batch_checks_satisfied": complete and reference_ok
        and initial_scoring_rejected and witnesses_ok,
        "full_dynamic_validation_complete": False,
        "generated_candidate_backend_ready": False,
        **{flag: False for flag in FLAGS},
    }
