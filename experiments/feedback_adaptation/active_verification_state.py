import copy

from active_verification_contract import (
    METHODS, canonical, identity, model_action, probe_request,
    require, tool_view, validate_limits, validate_view,
)


def known_sum(values):
    return None if any(value is None for value in values) else sum(values)


def replay(view, method, run_id, episode_id, events, limits):
    validate_view(view)
    validate_limits(limits)
    identity(run_id)
    identity(episode_id)
    require(method in METHODS, "unknown method")
    require(type(events) is list, "event list")
    initial = next(
        letter for letter, source in view["candidates"].items()
        if source == view["initial_source"]
    )
    pending = "done" if method == "keep" else (
        "tool" if method == "always_verify" else "model"
    )
    stage = "initial"
    selected = initial if method == "keep" else None
    status = "kept" if method == "keep" else "pending"
    history = []
    model_calls = 0
    tool_calls = 0
    executed_cases = 0
    request = (
        probe_request(view, run_id, episode_id, method)
        if method in ("always_verify", "active_verify") else None
    )
    for event in events:
        require(pending != "done", "event after terminal decision")
        require(type(event) is dict and event.get("kind") == pending, "unexpected event order")
        if pending == "tool":
            require(tool_calls == 0, "duplicate tool event")
            result = tool_view(event, request)
            tool_calls += 1
            executed_cases += event["executed_cases"]
            history.append({"role": "tool", "content": canonical(result).decode("utf-8")})
            pending, stage = "model", "final"
            continue
        require(model_calls < 2, "model call limit")
        allow_verify = method == "active_verify" and stage == "initial"
        action = model_action(event, allow_verify, limits)
        model_calls += 1
        history.append({"role": "assistant", "content": event["text"]})
        if action == "VERIFY":
            pending = "tool"
        elif action in ("A", "B", "C"):
            if method == "rethink" and stage == "initial":
                stage = "final"
            else:
                selected, status, pending = action, "valid", "done"
        else:
            status, pending = action, "done"
    next_action = None
    if pending == "tool":
        next_action = {"kind": "tool", "request": request}
    elif pending == "model":
        allowed = ["A", "B", "C"]
        if method == "active_verify" and stage == "initial":
            allowed.append("VERIFY")
        next_action = {
            "kind": "model", "round": model_calls + 1,
            "stage": "tentative" if method == "rethink" and stage == "initial" else "final",
            "allowed": allowed, "view": view, "history": history, "token_limits": limits,
        }
    model_events = [event for event in events if event["kind"] == "model"]
    wall_values = [event["wall_ns"] for event in events]
    cost = {
        "model_calls": model_calls, "tool_calls": tool_calls,
        "executed_cases": executed_cases,
        "input_tokens": known_sum([
            None if event["input_ids"] is None else len(event["input_ids"])
            for event in model_events
        ]),
        "output_tokens": known_sum([
            None if event["output_ids"] is None else len(event["output_ids"])
            for event in model_events
        ]),
        "component_wall_ns": known_sum(wall_values),
        "observed_component_wall_ns": sum(value for value in wall_values if value is not None),
        "episode_wall_ns": None,
    }
    return copy.deepcopy({
        "evidence_kind": "SYNTHETIC_ONLY",
        "real_execution_eligible": False, "benchmark_eligible": False,
        "method": method, "complete": pending == "done", "status": status,
        "selected": selected,
        "deployed": (selected or initial) if pending == "done" else None,
        "next_action": next_action, "events": events, "cost": cost,
    })
