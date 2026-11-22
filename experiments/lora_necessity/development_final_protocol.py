import re

from candidate_execution_protocol import _json_budget, _source
from pilot_protocol import fields, require
from validation_contract import digest, encoded


def _plain_json(value, remaining=1052672):
    require(remaining > 0, "JSON traversal budget")
    remaining -= 1
    kind = type(value)
    if kind is list:
        require(len(value) <= remaining, "JSON traversal budget")
        for item in value:
            remaining = _plain_json(item, remaining)
    elif kind is dict:
        require(len(value) <= remaining, "JSON traversal budget")
        for name, item in value.items():
            require(type(name) is str, "JSON object key")
            remaining = _plain_json(item, remaining)
    else:
        require(any(kind is primitive for primitive in
                    (type(None), bool, int, float, str)), "JSON value type")
    return remaining


def prepare_request(request):
    try:
        _plain_json(request)
        _json_budget(request, 1052672)
        fields(request, (
            "kind", "version", "round", "run_id", "attempt_id", "generation_call_id",
            "generation_sha256", "case_set_sha256", "task_id", "arm",
            "case_group", "case_id", "entry_point", "source", "args",
        ))
        require(type(request["kind"]) is str
                and request["kind"] == "LORA_DEVELOPMENT_FINAL_REQUEST",
                "final request kind")
        require(type(request["version"]) is int and request["version"] == 1, "version")
        require(type(request["round"]) is int and request["round"] == 2, "final round")
        for name, width in (
            ("run_id", 32), ("attempt_id", 32), ("generation_call_id", 32),
            ("generation_sha256", 64), ("case_set_sha256", 64),
        ):
            require(type(request[name]) is str
                    and re.fullmatch("[0-9a-f]{" + str(width) + "}", request[name]),
                    "final request identity")
        require(type(request["task_id"]) is str
                and re.fullmatch("D0[1-4]", request["task_id"]), "development task")
        require(type(request["arm"]) is str and request["arm"] in ("base", "text"),
                "development arm")
        require(type(request["case_group"]) is str
                and request["case_group"] in ("public", "scoring"), "case group")
        case_id = request["case_id"]
        require(type(case_id) is str
                and re.fullmatch("[A-Za-z][A-Za-z0-9_-]{0,31}", case_id), "case identity")
        require((case_id in ("P01", "P02")) == (request["case_group"] == "public"),
                "case group identity")
        require(type(request["args"]) is list, "request args")
        _source(request["source"], request["entry_point"])
        payload = {name: request[name] for name in ("source", "entry_point", "args")}
        _json_budget(payload, 1048575)
        return digest(request), encoded(payload) + b"\n"
    except (SyntaxError, UnicodeError, RecursionError) as error:
        raise ValueError("invalid final request encoding, depth or syntax") from error
