import ast
import copy
import re

from pilot_protocol import ARMS, fields, require
from validation_contract import digest, encoded, load_frozen, parse_json


EXECUTION_ERRORS = ("program_error", "timeout", "output_limit")


def _task(prompts_raw, task_id, arm):
    require(type(task_id) is str and type(arm) is str, "episode identity")
    tasks = load_frozen("prompts_v1.json", prompts_raw)
    matches = [task for task in tasks if task["id"] == task_id]
    require(len(matches) == 1, "unknown task")
    allowed = ARMS[:2] if task_id.startswith("D") else ARMS
    require(arm in allowed, "unknown episode")
    return matches[0]


def _valid(task, candidate):
    require(type(candidate) is str, "candidate text")
    try:
        require(len(candidate.encode("utf-8")) <= 65536, "candidate text byte cap")
    except UnicodeError as error:
        raise ValueError("candidate encoding") from error
    try:
        tree = ast.parse(candidate, mode="exec")
        if len(tree.body) != 1 or type(tree.body[0]) is not ast.FunctionDef:
            return False
        function = tree.body[0]
        initial = ast.parse(task["initial_source"]).body[0]
        if (function.name != task["entry_point"] or function.decorator_list
                or function.returns is not None or getattr(function, "type_params", [])
                or ast.dump(function.args) != ast.dump(initial.args)):
            return False
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                return False
            if isinstance(node, ast.FunctionDef) and node is not function:
                return False
        compile(tree, "<candidate-syntax-only>", "exec", dont_inherit=True)
    except (SyntaxError, ValueError, RecursionError):
        return False
    return True


def candidate_valid(prompts_raw, task_id, arm, candidate):
    return _valid(_task(prompts_raw, task_id, arm), candidate)


def visible_requests(prompts_raw, task_id, arm, candidate, *, attempt_id):
    require(type(attempt_id) is str and re.fullmatch("[0-9a-f]{32}", attempt_id),
            "attempt identity")
    task = _task(prompts_raw, task_id, arm)
    if not _valid(task, candidate):
        return []
    return [
        {"version": 1, "round": 1, "attempt_id": attempt_id, "task_id": task_id,
         "arm": arm, "case_id": case["id"], "entry_point": task["entry_point"],
         "source": candidate, "args": copy.deepcopy(case["args"])}
        for case in task["public_cases"]
    ]


def _outcome(value):
    require(type(value) is dict, "outcome object")
    if value.get("kind") == "return":
        fields(value, ("kind", "value"))
    else:
        fields(value, ("kind", "type"))
        require(value["kind"] == "raises" and value["type"] == "ValueError",
                "unsupported outcome")
    return value


def visible_feedback(prompts_raw, task_id, arm, candidate, receipts_raw, *, attempt_id):
    requests = visible_requests(prompts_raw, task_id, arm, candidate,
                                attempt_id=attempt_id)
    receipts = parse_json(receipts_raw, limit=65536)
    require(type(receipts) is list and len(receipts) == len(requests),
            "public receipt coverage")
    if not requests:
        return {"kind": "invalid_source"}
    task = _task(prompts_raw, task_id, arm)
    cases = []
    for request, receipt, case in zip(requests, receipts, task["public_cases"]):
        require(type(receipt) is dict, "receipt object")
        status = receipt.get("status")
        require(type(status) is str and status in ("completed",) + EXECUTION_ERRORS,
                "execution status")
        names = ("request_sha256", "status")
        fields(receipt, names + ("outcome",) if status == "completed" else names)
        require(type(receipt["request_sha256"]) is str
                and receipt["request_sha256"] == digest(request), "request binding")
        if status == "completed":
            actual = encoded(_outcome(receipt["outcome"]))
            status = "pass" if actual == encoded(case["expected"]) else "mismatch"
        cases.append({"id": case["id"], "status": status})
    return {"kind": "visible_tests", "cases": cases}
