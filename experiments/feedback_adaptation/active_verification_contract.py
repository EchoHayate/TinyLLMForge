import copy
import hashlib
import json
import re


METHODS = ("keep", "direct", "rethink", "always_verify", "active_verify")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def fields(value, expected):
    require(type(value) is dict and set(value) == set(expected), "unexpected fields")


def text(value):
    require(type(value) is str and bool(value.strip()), "nonempty text required")
    value.encode("utf-8")


def natural(value):
    require(type(value) is int and 0 <= value <= 2**63 - 1, "invalid integer")


def identity(value):
    require(
        type(value) is str and re.fullmatch(r"[0-9a-f]{32}", value) is not None,
        "invalid opaque identity",
    )


def validate_view(view):
    fields(view, ("specification", "initial_source", "candidates", "check_inputs", "feedback"))
    text(view["specification"])
    text(view["initial_source"])
    fields(view["candidates"], ("A", "B", "C"))
    for source in view["candidates"].values():
        text(source)
    require(len(set(view["candidates"].values())) == 3, "duplicate candidate")
    require(view["initial_source"] in view["candidates"].values(), "initial candidate absent")
    checks = view["check_inputs"]
    require(type(checks) is list and bool(checks), "missing checks")
    identifiers = set()
    for check in checks:
        fields(check, ("id", "input"))
        text(check["id"])
        text(check["input"])
        require(check["id"] not in identifiers, "duplicate check")
        identifiers.add(check["id"])
    feedback = view["feedback"]
    require(type(feedback) is list and len(feedback) <= 1, "feedback shape")
    for claim in feedback:
        text(claim)
    canonical(view)


def public_view(envelope):
    fields(envelope, ("evidence_kind", "public", "private"))
    require(envelope["evidence_kind"] == "synthetic", "synthetic only")
    require(type(envelope["private"]) is dict, "private sentinel container")
    validate_view(envelope["public"])
    return copy.deepcopy(envelope["public"])


def validate_limits(limits):
    fields(limits, ("max_input_tokens", "max_output_tokens"))
    for value in limits.values():
        natural(value)
        require(value > 0, "positive token limit required")


def model_action(event, allow_verify, limits):
    validate_limits(limits)
    require(type(allow_verify) is bool, "verify permission")
    fields(event, ("kind", "status", "text", "terminated", "input_ids", "output_ids", "wall_ns"))
    require(event["kind"] == "model", "expected model event")
    require(event["status"] in ("completed", "error", "timeout"), "model status")
    require(type(event["text"]) is str and type(event["terminated"]) is bool, "model output")
    if event["wall_ns"] is not None:
        natural(event["wall_ns"])
    for name in ("input_ids", "output_ids"):
        tokens = event[name]
        if tokens is None:
            require(event["status"] != "completed", "completed tokens missing")
        else:
            require(type(tokens) is list, "token array")
            for token in tokens:
                natural(token)
    canonical(event)
    if event["status"] != "completed":
        return event["status"]
    require(bool(event["input_ids"]), "completed input tokens missing")
    require(
        bool(event["output_ids"]) or (not event["terminated"] and not event["text"]),
        "completed output tokens inconsistent",
    )
    if (
        len(event["input_ids"]) > limits["max_input_tokens"]
        or len(event["output_ids"]) > limits["max_output_tokens"]
    ):
        return "budget_exceeded"
    if not event["terminated"]:
        return "truncated"
    output = event["text"].strip()
    if not output:
        return "empty"
    if output in ("A", "B", "C") or (allow_verify and output == "VERIFY"):
        return output
    return "invalid"


def probe_request(view, run_id, episode_id, method):
    validate_view(view)
    identity(run_id)
    identity(episode_id)
    require(method in ("always_verify", "active_verify"), "method has no tool")
    body = {
        "evidence_kind": "synthetic", "run_id": run_id, "episode_id": episode_id,
        "method": method,
        "public_sha256": hashlib.sha256(canonical(view)).hexdigest(),
        "initial_source": view["initial_source"],
        "check_inputs": copy.deepcopy(view["check_inputs"]),
    }
    return {**body, "request_id": hashlib.sha256(canonical(body)).hexdigest()}


def tool_view(event, request):
    fields(event, ("kind", "request_id", "status", "results", "executed_cases", "wall_ns"))
    require(event["kind"] == "tool", "expected tool event")
    require(event["request_id"] == request["request_id"], "tool request mismatch")
    require(event["status"] in ("completed", "error", "timeout", "unavailable"), "tool status")
    natural(event["executed_cases"])
    if event["wall_ns"] is not None:
        natural(event["wall_ns"])
    results = event["results"]
    checks = request["check_inputs"]
    require(type(results) is list and len(results) <= len(checks), "tool result count")
    require(event["executed_cases"] == len(results), "attempted cases mismatch")
    if event["status"] == "completed":
        require(len(results) == len(checks), "incomplete successful tool")
    for result, check in zip(results, checks):
        fields(result, ("id", "input", "expected", "actual", "status"))
        require(result["id"] == check["id"] and result["input"] == check["input"], "check mismatch")
        require(type(result["expected"]) is str, "expected value encoding")
        require(result["status"] in ("completed", "error", "timeout"), "case status")
        if result["status"] == "completed":
            require(type(result["actual"]) is str, "actual value encoding")
        else:
            require(result["actual"] is None, "invented failure output")
    canonical(event)
    return copy.deepcopy({
        name: event[name] for name in ("status", "results", "executed_cases")
    })
