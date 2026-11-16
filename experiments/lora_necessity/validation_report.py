from pilot_protocol import FLAGS, fields, require
from validation_contract import build_schedule, digest, encoded, parse_json


STATUSES = (
    "completed", "program_error", "timeout", "output_limit",
    "resource_limit", "protocol_error", "launch_failed",
)


def _value(value, depth=0):
    require(depth <= 16, "outcome nesting")
    if value is None or type(value) in (bool, int):
        return
    if type(value) is str:
        require(value.isascii(), "outcome ASCII string")
        return
    if type(value) is list:
        require(len(value) <= 4096, "outcome list size")
        for item in value:
            _value(item, depth + 1)
        return
    if type(value) is dict:
        require(len(value) <= 64 and all(type(name) is str and name.isascii()
                                       for name in value), "outcome keys")
        for item in value.values():
            _value(item, depth + 1)
        return
    raise ValueError("unsupported outcome type")


def parse_outcome(raw):
    result = parse_json(raw, limit=65536)
    require(type(result) is dict, "outcome object")
    if result.get("kind") == "return":
        fields(result, ("kind", "value"))
        _value(result["value"])
    else:
        fields(result, ("kind", "type"))
        require(result["kind"] == "raises" and result["type"] == "ValueError",
                "exception outcome")
    return result


def simulate_validation(source_raw, manifest_raw, observations_raw):
    plan = build_schedule(source_raw, manifest_raw)
    observations = parse_json(observations_raw, limit=8388608)
    require(type(observations) is list and len(observations) <= 1416, "observation count")
    indexed = {}
    previous = -1
    stop_ordinal = None
    for observation in observations:
        fields(observation, ("kind", "ordinal", "request_sha256", "status",
                             "payload", "cleanup_complete"))
        require(stop_ordinal is None, "observation after infrastructure stop")
        require(observation["kind"] == "synthetic", "synthetic observations only")
        ordinal = observation["ordinal"]
        require(type(ordinal) is int and previous < ordinal < 1416, "observation order")
        previous = ordinal
        planned = plan["rows"][ordinal]
        require(observation["request_sha256"] == planned["request_sha256"],
                "request binding")
        require(type(observation["cleanup_complete"]) is bool, "cleanup type")
        status = observation["status"]
        require(type(status) is str and status in STATUSES, "observation status")
        outcome = None
        if status == "completed":
            require(type(observation["payload"]) is str, "completed payload")
            try:
                outcome = parse_outcome(observation["payload"].encode("utf-8"))
            except (ValueError, UnicodeError):
                status = "protocol_error"
        else:
            require(observation["payload"] is None, "unexpected payload")
        if observation["status"] == "launch_failed" or not observation["cleanup_complete"]:
            stop_ordinal = ordinal
        indexed[ordinal] = {
            "status": status, "outcome": outcome,
            "cleanup_complete": observation["cleanup_complete"],
            "payload_sha256": digest(observation["payload"]),
        }
    rows = []
    for planned in plan["rows"]:
        observed = indexed.get(planned["ordinal"])
        outcome = None if observed is None else observed["outcome"]
        prediction = planned["witness_prediction"]
        rows.append({
            **{name: planned[name] for name in
               ("ordinal", "task_id", "role", "group", "case_id", "request_sha256")},
            "status": "missing" if observed is None else observed["status"],
            "outcome": outcome,
            "cleanup_complete": None if observed is None else observed["cleanup_complete"],
            "payload_sha256": None if observed is None else observed["payload_sha256"],
            "expected_match": None if outcome is None else
            encoded(outcome) == encoded(planned["expected"]),
            "prediction_match": None if outcome is None or prediction is None else
            encoded(outcome) == encoded(prediction),
        })
    tasks = []
    for task_id in sorted({row["task_id"] for row in rows}):
        selected = [row for row in rows if row["task_id"] == task_id]
        references = [row for row in selected if row["role"] == "reference"]
        initial = [row for row in selected if row["role"] == "initial"]
        witnesses = [row for row in selected if
                     plan["rows"][row["ordinal"]]["witness_prediction"] is not None]
        tasks.append({
            "task_id": task_id,
            "reference_ok": all(row["expected_match"] is True for row in references),
            "initial_scoring_rejected": any(row["group"] == "scoring"
                                           and row["expected_match"] is False
                                           for row in initial),
            "witnesses_ok": len(witnesses) == 3 and all(
                row["prediction_match"] is True and row["expected_match"] is False
                for row in witnesses),
        })
    complete = len(indexed) == 1416 and all(
        row["status"] == "completed" and row["cleanup_complete"] is True for row in rows)
    checks = all(task["reference_ok"] and task["initial_scoring_rejected"]
                 and task["witnesses_ok"] for task in tasks)
    return {
        "kind": "LORA_VALIDATION_SIMULATION", "version": 1,
        "plan_sha256": digest(plan), "observations_sha256": digest(observations),
        "scheduled": 1416, "received": len(indexed), "stop_ordinal": stop_ordinal,
        "simulated_decision": "INCOMPLETE" if not complete else
        ("CHECKS_SATISFIED" if checks else "CHECKS_FAILED"),
        "tasks": tasks, "rows": rows, **{flag: False for flag in FLAGS},
    }
