"""Compare the frozen matrix with controller receipts without executing source."""

import copy
import hashlib
import re

from pilot_protocol import FLAGS, fields, require
from validation_contract import build_schedule, digest, encoded, parse_json
from validation_real_report import (
    _events,
    _observations,
    _process_result,
    build_d01_report,
)


def batch_ranges():
    """Return the ordered remaining slices, excluding the accepted D01 batch."""
    return [(start, min(start + 32, 1416)) for start in range(32, 1416, 32)]


def _expected_runtime(runtime):
    require(type(runtime) is dict and runtime, "approved runtime required")
    return encoded(runtime)


def _bundle(bundle_raw, planned_rows):
    bundle = parse_json(bundle_raw)
    fields(bundle, ("kind", "mode", "run_id", "worker_source", "requests"))
    require(bundle["kind"] == "LORA_REMOTE_VALIDATION_BUNDLE"
            and bundle["mode"] == "matrix", "matrix bundle required")
    require(type(bundle["run_id"]) is str
            and re.fullmatch(r"[0-9a-f]{32}", bundle["run_id"]), "bundle run ID")
    require(type(bundle["worker_source"]) is str and bundle["worker_source"],
            "worker source text")
    requests = [
        {"request_id": f"request-{planned['ordinal']:04d}",
         "source": planned["source"], "entry_point": planned["entry_point"],
         "args": planned["args"]}
        for planned in planned_rows
    ]
    require(type(bundle["requests"]) is list
            and encoded(bundle["requests"]) == encoded(requests),
            "frozen matrix request projection")
    try:
        worker_raw = bundle["worker_source"].encode("utf-8")
    except UnicodeError as error:
        raise ValueError("worker source encoding") from error
    return bundle, hashlib.sha256(worker_raw).hexdigest()


def _rows(planned_rows, observed):
    rows = []
    for position, planned in enumerate(planned_rows):
        observation = observed[position] if position < len(observed) else None
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
    return rows


def _build_batch_report(plan, bundle_raw, transport, *, start, expected_runtime):
    approved = _expected_runtime(expected_runtime)
    require(type(start) is int and start in range(32, 1416, 32),
            "scheduled matrix start required")
    stop = min(start + 32, 1416)
    planned_rows = plan["rows"][start:stop]
    bundle, worker_sha256 = _bundle(bundle_raw, planned_rows)
    bundle_sha256 = hashlib.sha256(bundle_raw).hexdigest()
    stdout = _process_result(transport, outer=True)
    events, truncated = _events(stdout, timed_out=transport["status"] == "timeout")
    for event in events:
        if type(event) is dict and event.get("kind") == "row":
            require(type(event.get("ordinal")) is int
                    and 0 <= event["ordinal"] < len(planned_rows), "batch row bounds")
        if type(event) is dict and event.get("kind") == "ready":
            require(encoded(event.get("runtime")) == approved,
                    "runtime differs from approved matrix runtime")
    runtime, done, observed, local_stop = _observations(
        events, planned_rows, bundle["run_id"], bundle_sha256, worker_sha256)
    if done is not None:
        require(done["received"] == len(planned_rows) or done["stopped"],
                "short batch must stop")
    rows = _rows(planned_rows, observed)
    complete = (
        len(observed) == len(planned_rows)
        and all(row["status"] == "completed" for row in rows)
        and done is not None and not done["stopped"]
        and transport["status"] == "completed" and not truncated
    )
    return {
        "kind": "LORA_MATRIX_BATCH_REPORT", "version": 1,
        "run_id": bundle["run_id"], "start": start, "stop": stop,
        "source_sha256": plan["source_sha256"],
        "manifest_sha256": plan["manifest_sha256"],
        "bundle_sha256": bundle_sha256, "worker_sha256": worker_sha256,
        "runtime": copy.deepcopy(runtime),
        "runtime_sha256": None if runtime is None else digest(runtime),
        "expected_runtime_sha256": digest(expected_runtime),
        "transport": copy.deepcopy(transport), "transport_sha256": digest(transport),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "done": copy.deepcopy(done), "truncated_final_line": truncated,
        "full_scheduled": 1416, "batch_scheduled": len(planned_rows),
        "received": len(observed),
        "stop_ordinal": None if local_stop is None else start + local_stop,
        "rows": rows, "batch_complete": complete,
        "full_dynamic_validation_complete": False,
        "generated_candidate_backend_ready": False,
        **{flag: False for flag in FLAGS},
    }


def build_batch_report(source_raw, manifest_raw, bundle_raw, transport, *,
                       start, expected_runtime):
    """Validate one exact frozen slice and retain its missing global rows."""
    plan = build_schedule(source_raw, manifest_raw)
    return _build_batch_report(plan, bundle_raw, transport, start=start,
                               expected_runtime=expected_runtime)


def _task_checks(plan, rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["task_id"], []).append(row)
    tasks = []
    for task_id, selected in sorted(grouped.items()):
        references = [row for row in selected if row["role"] == "reference"]
        witnesses = [row for row in selected if
                     plan["rows"][row["ordinal"]]["witness_prediction"] is not None]
        tasks.append({
            "task_id": task_id,
            "reference_ok": all(row["expected_match"] is True for row in references),
            "initial_scoring_rejected": any(
                row["role"] == "initial" and row["group"] == "scoring"
                and row["expected_match"] is False for row in selected),
            "witnesses_ok": len(witnesses) == 3 and all(
                row["prediction_match"] is True and row["expected_match"] is False
                for row in witnesses),
        })
    return tasks


def aggregate_matrix(source_raw, manifest_raw, d01_bundle_raw, d01_transport,
                     batch_records, *, expected_runtime, d01_expected_runtime):
    """Revalidate raw D01 and ordered new receipts over the fixed denominator."""
    _expected_runtime(expected_runtime)
    approved_d01 = _expected_runtime(d01_expected_runtime)
    require(type(batch_records) is list and len(batch_records) <= 44,
            "ordered matrix batch records required")
    plan = build_schedule(source_raw, manifest_raw)
    d01 = build_d01_report(source_raw, manifest_raw, d01_bundle_raw, d01_transport)
    require(encoded(d01["runtime"]) == approved_d01, "D01 runtime differs from approval")
    require(d01["batch_complete"] and d01["batch_checks_satisfied"],
            "D01 complete and checks satisfied required")
    rows = _rows(plan["rows"], [])
    rows[:32] = d01["rows"]
    batches = []
    run_ids = {d01["run_id"]}
    for position, record in enumerate(batch_records):
        fields(record, ("start", "bundle_raw", "transport"))
        require(type(record["start"]) is int
                and record["start"] == batch_ranges()[position][0],
                "matrix batch gap, duplicate or reordering")
        require(not batches or batches[-1]["batch_complete"],
                "record after incomplete batch")
        batch = _build_batch_report(
            plan, record["bundle_raw"], record["transport"],
            start=record["start"], expected_runtime=expected_runtime)
        require(batch["run_id"] not in run_ids, "duplicate run ID")
        run_ids.add(batch["run_id"])
        rows[batch["start"]:batch["stop"]] = batch["rows"]
        batches.append(batch)
    tasks = _task_checks(plan, rows)
    received = d01["received"] + sum(batch["received"] for batch in batches)
    complete = (len(batches) == 44 and all(batch["batch_complete"] for batch in batches)
                and received == 1416 and all(row["status"] == "completed" for row in rows))
    checks = all(task["reference_ok"] and task["initial_scoring_rejected"]
                 and task["witnesses_ok"] for task in tasks)
    references = [row for row in rows
                  if row["role"] == "reference" and row["status"] == "completed"]
    witnesses = [row for row in rows if row["status"] == "completed"
                 and plan["rows"][row["ordinal"]]["witness_prediction"] is not None]
    return {
        "kind": "LORA_MATRIX_EXECUTION_REPORT", "version": 1,
        "source_sha256": plan["source_sha256"],
        "manifest_sha256": plan["manifest_sha256"], "plan_sha256": digest(plan),
        "expected_runtime": copy.deepcopy(expected_runtime),
        "expected_runtime_sha256": digest(expected_runtime),
        "d01_expected_runtime": copy.deepcopy(d01_expected_runtime),
        "d01_expected_runtime_sha256": digest(d01_expected_runtime),
        "d01": d01, "batches": batches, "scheduled": 1416,
        "received": received, "remaining": 1416 - received,
        "reference_count": len(references), "witness_count": len(witnesses),
        "initial_witness_count": sum(row["role"] == "initial" for row in witnesses),
        "mutant_witness_count": sum(row["role"] != "initial" for row in witnesses),
        "tasks": tasks, "rows": rows, "matrix_complete": complete,
        "data_checks_satisfied": checks,
        "full_dynamic_validation_complete": complete and checks,
        "generated_candidate_backend_ready": False,
        **{flag: False for flag in FLAGS},
    }
