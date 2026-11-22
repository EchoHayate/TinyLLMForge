import ast
import re

from pilot_protocol import ARMS, fields, require
from validation_contract import digest, encoded, parse_json


__all__ = ("READY", "prepare_request", "InfrastructureError", "make_receipt")

READY = b"TLF_CANDIDATE_READY_V1\n"


class InfrastructureError(RuntimeError):
    pass


def _json_budget(value, remaining):
    require(remaining > 0, "JSON byte budget")
    kind = type(value)
    if kind in (list, dict):
        remaining -= 2 + max(0, len(value) - 1)
        require(remaining >= 0, "JSON byte budget")
        if kind is list:
            for item in value:
                remaining = _json_budget(item, remaining)
        else:
            for name, item in value.items():
                require(type(name) is str, "JSON object key")
                remaining = _json_budget(name, remaining) - 1
                remaining = _json_budget(item, remaining)
    else:
        require(kind in (type(None), bool, int, float, str), "JSON value type")
        if kind is str:
            require(len(value) <= remaining, "JSON byte budget")
        remaining -= len(encoded(value))
    require(remaining >= 0, "JSON byte budget")
    return remaining


def _source(source, entry_point):
    require(type(source) is str and len(source) <= 65536, "candidate text")
    require(len(source.encode("utf-8")) <= 65536, "candidate text byte cap")
    require(type(entry_point) is str and entry_point.isidentifier(), "entry point")
    tree = ast.parse(source, mode="exec")
    require(len(tree.body) == 1 and type(tree.body[0]) is ast.FunctionDef,
            "single function required")
    function = tree.body[0]
    require(function.name == entry_point and not function.decorator_list
            and function.returns is None and not getattr(function, "type_params", []),
            "function entry syntax")
    require(not function.args.defaults
            and all(default is None for default in function.args.kw_defaults),
            "function argument defaults")
    for node in ast.walk(tree):
        require(not isinstance(node, (ast.Import, ast.ImportFrom,
                                      ast.AsyncFunctionDef, ast.ClassDef)),
                "unsupported function syntax")
        require(not isinstance(node, ast.FunctionDef) or node is function,
                "nested function")
        require(not isinstance(node, ast.arg) or node.annotation is None,
                "function argument annotation")
    compile(tree, "<candidate-syntax-only>", "exec", dont_inherit=True)


def prepare_request(request):
    fields(request, ("version", "round", "attempt_id", "task_id", "arm",
                     "case_id", "entry_point", "source", "args"))
    for name in ("version", "round"):
        require(type(request[name]) is int and request[name] == 1, name)
    require(type(request["attempt_id"]) is str
            and re.fullmatch("[0-9a-f]{32}", request["attempt_id"]), "attempt identity")
    task_id = request["task_id"]
    require(type(task_id) is str
            and re.fullmatch(r"(D0[1-4]|[TR](0[1-9]|1[0-6]))", task_id),
            "task identity")
    allowed = ARMS[:2] if task_id.startswith("D") else ARMS
    require(type(request["arm"]) is str and request["arm"] in allowed, "arm identity")
    require(type(request["case_id"]) is str and request["case_id"] in ("P01", "P02"),
            "case identity")
    require(type(request["args"]) is list, "request args")
    try:
        _source(request["source"], request["entry_point"])
        payload = {name: request[name] for name in ("source", "entry_point", "args")}
        _json_budget(payload, 1048575)
        return digest(request), encoded(payload) + b"\n"
    except (SyntaxError, UnicodeError, RecursionError) as error:
        raise ValueError("invalid request encoding, depth or syntax") from error


def _parse_result(raw):
    require(len(raw) <= 16384, "result byte budget")
    depth = 0
    in_string = False
    escaped = False
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                in_string = False
        elif byte == 34:
            in_string = True
        elif byte in (91, 123):
            depth += 1
            require(depth <= 128, "result JSON depth")
        elif byte in (93, 125):
            depth -= 1
            require(depth >= 0, "result JSON nesting")
    return parse_json(raw, limit=16384)


def make_receipt(request_sha256, *, returncode, ready, input_complete,
                 reaped, eof, termination, result_raw):
    try:
        require(type(request_sha256) is str
                and re.fullmatch("[0-9a-f]{64}", request_sha256), "request digest")
        require(all(type(flag) is bool for flag in (ready, input_complete, reaped, eof)),
                "parent fact type")
        require(type(returncode) is int, "process return code")
        require(termination is None or type(termination) is str
                and termination in ("timeout", "output_limit"), "termination reason")
        require(type(result_raw) is bytes, "result bytes")
        require(ready and input_complete and reaped, "incomplete execution lifecycle")
        require(eof or termination is not None, "incomplete stream collection")
    except ValueError as error:
        raise InfrastructureError(str(error)) from error
    receipt = {"request_sha256": request_sha256, "status": "program_error"}
    if termination is not None:
        receipt["status"] = termination
        return receipt
    if returncode != 0:
        return receipt
    try:
        outcome = _parse_result(result_raw)
        require(type(outcome) is dict, "outcome object")
        if outcome.get("kind") == "return":
            fields(outcome, ("kind", "value"))
        else:
            fields(outcome, ("kind", "type"))
            require(outcome["kind"] == "raises" and outcome["type"] == "ValueError",
                    "unsupported outcome")
    except (ValueError, RecursionError):
        return receipt
    return {**receipt, "status": "completed", "outcome": outcome}
