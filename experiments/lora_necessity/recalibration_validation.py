import copy
import hashlib
import re

from pilot_protocol import FLAGS, fields, require
from recalibration_data_contract import FILE_PINS, TASK_IDS, _plain, load_dataset
from validation_contract import digest, encoded
from validation_real_report import _events, _observations, _process_result


WORKER_SHA256 = "d2c8095b4d1e854a2f497b084226a92b47aa8b083d59c8ab272b088544fedca4"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _request(row):
    return {
        "request_id": "request-%04d" % row["ordinal"], "source": row["source"],
        "entry_point": row["entry_point"], "args": copy.deepcopy(row["args"]),
    }


def build_schedule(raw_files):
    dataset = load_dataset(raw_files)
    rows = []
    for prompt, oracle, metadata in zip(
            dataset["prompts"], dataset["oracles"], dataset["structure"]["tasks"]):
        task_sha256 = digest({"prompt": prompt, "oracle": oracle, "metadata": metadata})
        for program in oracle["references"] + oracle["incorrect_programs"]:
            for case in oracle["cases"]:
                witness = program.get("witness")
                prediction = (witness["predicted"] if witness is not None
                              and witness["case_id"] == case["id"] else None)
                row = {
                    "ordinal": len(rows), "task_id": prompt["id"],
                    "role": program["id"], "case_id": case["id"],
                    "entry_point": prompt["entry_point"], "file_sha256": dict(FILE_PINS),
                    "task_sha256": task_sha256, "source": program["source"],
                    "source_sha256": _sha(program["source"].encode("utf-8")),
                    "args": copy.deepcopy(case["args"]), "input_sha256": digest(case["args"]),
                    "expected": copy.deepcopy(case["expected"]),
                    "expected_sha256": digest(case["expected"]),
                    "witness_prediction": copy.deepcopy(prediction),
                    "witness_prediction_sha256": None if prediction is None else digest(prediction),
                }
                row["request_sha256"] = digest(_request(row))
                row["row_sha256"] = digest(row)
                rows.append(row)
    require(len(rows) == 480, "recalibration schedule count")
    return {
        "kind": "LORA_RECALIBRATION_VALIDATION_PLAN", "version": 1,
        "data_version": "recalibration_v2_r1", "file_sha256": dict(FILE_PINS), "rows": rows,
    }


def _packet(rows, worker_raw, run_id):
    require(type(worker_raw) is bytes and len(worker_raw) <= 16384, "worker byte limit")
    require(len(rows) == 32, "batch count")
    requests = [_request(row) for row in rows]
    for request in requests:
        require(len(request["source"].encode("utf-8")) <= 4096, "source byte limit")
        require(len(encoded(request)) <= 8192, "request byte limit")
    bundle_raw = encoded({
        "kind": "LORA_REMOTE_VALIDATION_BUNDLE", "mode": "matrix", "run_id": run_id,
        "worker_source": worker_raw.decode("utf-8"), "requests": requests,
    })
    require(len(bundle_raw) <= 131072, "bundle byte limit")
    return bundle_raw


def _batches(plan, worker_raw, run_ids):
    require(type(worker_raw) is bytes and len(worker_raw) <= 16384
            and _sha(worker_raw) == WORKER_SHA256, "fixed worker identity")
    require(type(run_ids) is list and len(run_ids) == 15
            and all(type(value) is str and re.fullmatch(r"[0-9a-f]{32}", value)
                    for value in run_ids) and len(set(run_ids)) == 15, "batch run IDs")
    plan_sha256 = digest(plan)
    batches = []
    for batch_index, run_id in enumerate(run_ids):
        start = batch_index * 32
        stop = start + 32
        raw = _packet(plan["rows"][start:stop], worker_raw, run_id)
        batches.append({
            "batch_index": batch_index, "start": start, "stop": stop, "run_id": run_id,
            "plan_sha256": plan_sha256, "bundle_sha256": _sha(raw),
            "worker_sha256": WORKER_SHA256, "bundle_raw": raw,
        })
    return batches


def build_batches(raw_files, worker_raw, run_ids):
    return _batches(build_schedule(raw_files), worker_raw, run_ids)


def _rows(planned_rows, observed, dispatch_state):
    rows = []
    for position, planned in enumerate(planned_rows):
        observation = observed[position] if position < len(observed) else None
        result = None if observation is None else observation["result"]
        outcome = None if observation is None else observation["outcome"]
        prediction = planned["witness_prediction"]
        rows.append({
            **{name: planned[name] for name in (
                "ordinal", "task_id", "role", "case_id", "row_sha256",
                "task_sha256", "source_sha256", "input_sha256", "expected_sha256",
                "request_sha256", "witness_prediction_sha256")},
            "dispatch_state": dispatch_state if result is None else "observed",
            "status": dispatch_state if result is None else result["status"],
            "result": copy.deepcopy(result), "outcome": copy.deepcopy(outcome),
            "expected_match": None if outcome is None else
            encoded(outcome) == encoded(planned["expected"]),
            "prediction_match": None if outcome is None or prediction is None else
            encoded(outcome) == encoded(prediction),
        })
    return rows


def _receipt(planned_rows, batch, record, expected_runtime):
    transport = record["transport"]
    observed = []
    runtime = None
    done = None
    truncated = False
    local_stop = None
    reason = None
    detail = None
    stdout_sha256 = None
    try:
        require(record["bundle_raw"] == batch["bundle_raw"], "exact planned bundle")
        if transport is None:
            reason = "missing_transport"
        else:
            stdout = _process_result(transport, outer=True)
            stdout_sha256 = _sha(stdout)
            events, truncated = _events(stdout, timed_out=transport["status"] == "timeout")
            for event in events:
                if type(event) is dict and event.get("kind") == "ready":
                    require(encoded(event.get("runtime")) == encoded(expected_runtime),
                            "runtime differs from expected runtime")
            runtime, done, observed, local_stop = _observations(
                events, planned_rows, batch["run_id"], batch["bundle_sha256"], WORKER_SHA256)
            if done is not None:
                require(done["received"] == 32 or done["stopped"], "short batch must stop")
            if local_stop is not None:
                reason = "worker_" + observed[local_stop]["result"]["status"]
            elif transport["status"] != "completed":
                reason = "transport_" + transport["status"]
            elif runtime is None:
                reason = "missing_ready"
            elif done is None:
                reason = "missing_done"
            elif done["stopped"]:
                reason = "backend_stopped"
            elif truncated or len(observed) != 32:
                reason = "incomplete_receipt"
    except ValueError as error:
        observed = []
        runtime = None
        done = None
        local_stop = None
        reason = "evidence_error"
        detail = str(error)
    return {
        **{name: value for name, value in batch.items() if name != "bundle_raw"},
        "actual_bundle_sha256": _sha(record["bundle_raw"]),
        "transport": copy.deepcopy(transport), "transport_sha256": digest(transport),
        "stdout_sha256": stdout_sha256, "runtime": copy.deepcopy(runtime),
        "runtime_sha256": None if runtime is None else digest(runtime),
        "done": copy.deepcopy(done), "truncated_final_line": truncated,
        "rows": _rows(planned_rows, observed, "unknown_after_dispatch"),
        "received": len(observed), "batch_complete": reason is None,
        "stop_reason": reason, "evidence_error": detail,
        "stop_ordinal": None if local_stop is None else batch["start"] + local_stop,
    }


def _checks(rows):
    references = [row for row in rows if row["role"] in ("reference_a", "reference_b")]
    witnesses = [row for row in rows if row["witness_prediction_sha256"] is not None]
    return {
        "reference_counts": {
            "planned": len(references),
            "observed": sum(row["outcome"] is not None for row in references),
            "matched": sum(row["expected_match"] is True for row in references),
        },
        "witness_counts": {
            "planned": len(witnesses),
            "observed": sum(row["outcome"] is not None for row in witnesses),
            "matched": sum(row["prediction_match"] is True and row["expected_match"] is False
                           for row in witnesses),
        },
    }


def build_report(raw_files, worker_raw, run_ids, records, *, expected_runtime):
    plan = build_schedule(raw_files)
    planned_batches = _batches(plan, worker_raw, run_ids)
    _plain(expected_runtime)
    require(type(expected_runtime) is dict and bool(expected_runtime), "expected runtime required")
    require(type(records) is list and len(records) <= 15, "batch records")
    rows = _rows(plan["rows"], [], "not_dispatched")
    batches = []
    for position, record in enumerate(records):
        fields(record, ("batch_index", "bundle_raw", "transport"))
        require(all(type(name) is str for name in record), "record key type")
        require(type(record["batch_index"]) is int and record["batch_index"] == position,
                "batch gap, duplicate or ordering")
        require(not batches or batches[-1]["batch_complete"], "record after incomplete batch")
        require(type(record["bundle_raw"]) is bytes
                and len(record["bundle_raw"]) <= 131072, "record bundle bytes")
        _plain(record["transport"])
        require(record["transport"] is None or type(record["transport"]) is dict,
                "transport object or absent")
        batch = planned_batches[position]
        report = _receipt(plan["rows"][batch["start"]:batch["stop"]],
                          batch, record, expected_runtime)
        rows[batch["start"]:batch["stop"]] = report["rows"]
        batches.append(report)
    checks = _checks(rows)
    completed = sum(row["status"] == "completed" for row in rows)
    complete = len(batches) == 15 and all(batch["batch_complete"] for batch in batches)
    complete = complete and completed == 480
    first_stop = None
    if batches and not batches[-1]["batch_complete"]:
        batch = batches[-1]
        first_stop = {"batch_index": batch["batch_index"], "ordinal": batch["stop_ordinal"],
                      "reason": batch["stop_reason"]}
    elif not complete:
        first_stop = {"batch_index": len(batches), "ordinal": len(batches) * 32,
                      "reason": "records_exhausted"}
    tasks = []
    for task_id in TASK_IDS:
        selected = [row for row in rows if row["task_id"] == task_id]
        task_checks = _checks(selected)
        tasks.append({
            "task_id": task_id, "scheduled": 40,
            "completed": sum(row["status"] == "completed" for row in selected),
            "unknown_after_dispatch": sum(row["dispatch_state"] == "unknown_after_dispatch"
                                          for row in selected),
            "not_dispatched": sum(row["dispatch_state"] == "not_dispatched" for row in selected),
            **task_checks,
            "reference_ok": task_checks["reference_counts"]["matched"] == 16,
            "witnesses_ok": task_checks["witness_counts"]["matched"] == 3,
        })
    evidence = [{"batch_index": batch["batch_index"],
                 "bundle_sha256": batch["actual_bundle_sha256"],
                 "transport_sha256": batch["transport_sha256"]} for batch in batches]
    return {
        "kind": "LORA_RECALIBRATION_VALIDATION_REPORT", "version": 1,
        "data_version": "recalibration_v2_r1", "file_sha256": dict(FILE_PINS),
        "plan_sha256": digest(plan), "worker_sha256": WORKER_SHA256,
        "run_ids": list(run_ids), "expected_runtime": copy.deepcopy(expected_runtime),
        "expected_runtime_sha256": digest(expected_runtime),
        "evidence_sha256": digest(evidence), "batches": batches,
        "scheduled": 480, "completed": completed,
        "observed": sum(row["dispatch_state"] == "observed" for row in rows),
        "unknown_after_dispatch": sum(row["dispatch_state"] == "unknown_after_dispatch"
                                      for row in rows),
        "not_dispatched": sum(row["dispatch_state"] == "not_dispatched" for row in rows),
        "rows": rows, "tasks": tasks, **checks, "first_stop": first_stop,
        "matrix_complete": complete,
        "data_checks_satisfied": complete and checks["reference_counts"]["matched"] == 192
        and checks["witness_counts"]["matched"] == 36,
        "execution_provenance_verified": False, "run_authorized": False,
        "full_dynamic_validation_complete": False,
        **{flag: False for flag in FLAGS},
    }
