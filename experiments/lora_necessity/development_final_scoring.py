import copy
import hashlib
import re

from candidate_feedback import EXECUTION_ERRORS, _outcome, _valid
from candidate_execution_protocol import _parse_result
from development_final_protocol import prepare_request
from pilot_protocol import FLAGS, fields, require
from validation_contract import PINS, _tasks, digest, encoded, parse_json


def _identity(value, width):
    require(type(value) is str
            and re.fullmatch("[0-9a-f]{" + str(width) + "}", value), "identity")


def _cases(task):
    cases = []
    for group, name in (("public", "public_cases"), ("scoring", "scoring_cases")):
        items = task.get(name)
        require(type(items) is list and bool(items), "empty case group")
        seen = set()
        for case in items:
            fields(case, ("id", "args", "expected", "reason"))
            case_id = case["id"]
            require(type(case_id) is str
                    and re.fullmatch("[A-Za-z][A-Za-z0-9_-]{0,31}", case_id),
                    "case identity")
            require(case_id not in seen, "duplicate case")
            require((case_id in ("P01", "P02")) == (group == "public"),
                    "case group identity")
            require(type(case["args"]) is list and type(case["reason"]) is str,
                    "case data")
            _outcome(_parse_result(encoded(case["expected"])))
            seen.add(case_id)
        if group == "public":
            require(seen == {"P01", "P02"}, "public case coverage")
        for case in sorted(items, key=lambda item: item["id"]):
            cases.append({"case_group": group, "case_id": case["id"],
                          "args": copy.deepcopy(case["args"]),
                          "expected": copy.deepcopy(case["expected"])})
    return cases


def build_final_plan(source_raw, manifest_raw, *, run_id, task_id, arm,
                     generation_call_id, generation_raw, expected_generation_sha256):
    _identity(run_id, 32)
    _identity(generation_call_id, 32)
    _identity(expected_generation_sha256, 64)
    require(type(task_id) is str and re.fullmatch("D0[1-4]", task_id),
            "development task")
    require(type(arm) is str and arm in ("base", "text"), "development arm")
    generation = parse_json(generation_raw)
    require(hashlib.sha256(generation_raw).hexdigest() == expected_generation_sha256,
            "generation bytes changed")
    require(type(generation) is dict
            and type(generation.get("kind")) is str
            and generation["kind"] == "LORA_GREEDY_CANDIDATE_UNATTESTED",
            "generation kind")
    candidate = generation.get("candidate")
    require(type(candidate) is str, "candidate text")
    try:
        candidate_bytes = candidate.encode("utf-8")
    except UnicodeError as error:
        raise ValueError("candidate encoding") from error
    require(len(candidate_bytes) <= 65536, "candidate byte cap")
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    require(type(generation.get("candidate_sha256")) is str
            and generation["candidate_sha256"] == candidate_sha256,
            "candidate bytes changed")
    matches = [task for task in _tasks(source_raw, manifest_raw)
               if task["id"] == task_id]
    require(len(matches) == 1 and matches[0]["split"] == "development",
            "frozen development task")
    task = matches[0]
    cases = _cases(task)
    case_set = {
        "kind": "LORA_DEVELOPMENT_FINAL_CASE_SET", "version": 1,
        "source_sha256": PINS["task_sources_v1.json"],
        "manifest_sha256": PINS["manifest_v1.json"],
        "task_id": task_id, "task_sha256": digest(task),
        "entry_point": task["entry_point"], "cases": cases,
    }
    case_set_sha256 = digest(case_set)
    binding = {
        "run_id": run_id, "task_id": task_id, "arm": arm,
        "generation_call_id": generation_call_id,
        "generation_sha256": expected_generation_sha256,
        "case_set_sha256": case_set_sha256,
    }
    valid = _valid(task, candidate)
    requests = []
    if valid:
        for case in cases:
            case_identity = {name: case[name] for name in ("case_group", "case_id")}
            attempt = {"kind": "LORA_DEVELOPMENT_FINAL_CASE_ATTEMPT",
                       "version": 1, **binding, **case_identity}
            request = {
                "kind": "LORA_DEVELOPMENT_FINAL_REQUEST", "version": 1, "round": 2,
                **binding, **case_identity, "attempt_id": digest(attempt)[:32],
                "entry_point": task["entry_point"], "source": candidate,
                "args": copy.deepcopy(case["args"]),
            }
            prepare_request(request)
            requests.append(request)
        require(len({request["attempt_id"] for request in requests}) == len(requests),
                "duplicate attempt")
    return {
        "kind": "LORA_DEVELOPMENT_FINAL_PLAN", "version": 1, **binding,
        "candidate_sha256": candidate_sha256, "task_sha256": digest(task),
        "source_sha256": case_set["source_sha256"],
        "manifest_sha256": case_set["manifest_sha256"],
        "candidate_valid": valid, "scheduled_cases": len(cases),
        "requests": requests,
        "expected_outcomes": [
            {name: copy.deepcopy(case[name]) for name in
             ("case_group", "case_id", "expected")} for case in cases
        ],
    }


def score_final(source_raw, manifest_raw, receipts_raw, **identity):
    plan = build_final_plan(source_raw, manifest_raw, **identity)
    receipts = parse_json(receipts_raw)
    require(type(receipts) is list and len(receipts) <= len(plan["requests"]),
            "receipt coverage")
    rows = []
    for ordinal, expected in enumerate(plan["expected_outcomes"]):
        request = plan["requests"][ordinal] if plan["candidate_valid"] else None
        request_sha256 = None if request is None else digest(request)
        status = "missing" if plan["candidate_valid"] else "invalid_source"
        if ordinal < len(receipts):
            receipt = receipts[ordinal]
            require(type(receipt) is dict, "receipt object")
            status = receipt.get("status")
            require(type(status) is str and status in ("completed",) + EXECUTION_ERRORS,
                    "execution status")
            names = ("request_sha256", "status")
            fields(receipt, names + ("outcome",) if status == "completed" else names)
            require(type(receipt["request_sha256"]) is str
                    and receipt["request_sha256"] == request_sha256, "request binding")
            if status == "completed":
                actual = encoded(_outcome(_parse_result(encoded(receipt["outcome"]))))
                status = "pass" if actual == encoded(expected["expected"]) else "mismatch"
        rows.append({"case_group": expected["case_group"], "case_id": expected["case_id"],
                     "request_sha256": request_sha256, "status": status})
    passed = sum(row["status"] == "pass" for row in rows)
    if not plan["candidate_valid"]:
        status = "invalid_source"
    elif len(receipts) < plan["scheduled_cases"]:
        status = "incomplete"
    else:
        status = "pass" if passed == plan["scheduled_cases"] else "fail"
    binding_names = (
        "run_id", "task_id", "arm", "generation_call_id", "generation_sha256",
        "candidate_sha256", "case_set_sha256", "candidate_valid", "scheduled_cases",
    )
    return {
        "kind": "LORA_DEVELOPMENT_FINAL_SCORE", "version": 1,
        **{name: plan[name] for name in binding_names},
        "plan_sha256": digest(plan), "status": status,
        "received_cases": len(receipts), "passed_cases": passed, "cases": rows,
        **{flag: False for flag in FLAGS},
    }
