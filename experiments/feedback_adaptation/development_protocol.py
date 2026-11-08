import copy
import hashlib
import json
from collections import Counter

from active_verification_contract import METHODS, canonical, fields, identity, model_action, natural, require
from development_tasks import catalog


LIMITS = {"max_input_tokens": 4096, "max_output_tokens": 32}
MODEL_FIELDS = ("kind", "status", "text", "terminated", "input_ids", "output_ids", "wall_ns")
PERMUTATIONS = (
    ("defect", "correct", "other"), ("other", "defect", "correct"),
    ("correct", "other", "defect"), ("defect", "other", "correct"),
)


def checksum(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def build_manifest(run_id):
    identity(run_id)
    cells = []
    for task, permutation in zip(catalog(), PERMUTATIONS):
        for initial in ("correct", "defect"):
            for condition in ("none", "correct", "incorrect"):
                position = len(cells)
                episode_id = checksum([run_id, position])[:32]
                checks = [
                    {"id": f"probe-{index}", "input": row["input"]}
                    for index, row in enumerate(task["check_inputs"])
                ]
                view = {
                    "specification": task["specification"],
                    "initial_source": task["sources"][initial],
                    "candidates": dict(zip("ABC", (task["sources"][role] for role in permutation))),
                    "check_inputs": checks,
                    "feedback": [] if condition == "none" else [
                        task["private"]["feedback_by_initial"][initial][condition]
                    ],
                }
                offset = position % len(METHODS)
                cells.append({
                    "episode_id": episode_id, "task_id": task["task_id"],
                    "family_id": task["family_id"], "condition": condition, "initial": initial,
                    "verification_information": task["private"]["verification_information"],
                    "permutation": list(permutation), "view": view,
                    "methods": list(METHODS[offset:] + METHODS[:offset]),
                })
    return copy.deepcopy({
        "kind": "FIXED_ONLINE_DEVELOPMENT", "version": 1, "run_id": run_id,
        "limits": dict(LIMITS),
        "budget": {"model_calls": 144, "tool_calls": 48, "probe_cases": 96, "wall_seconds": 600},
        "cells": cells,
        "schedule": [{"episode_id": cell["episode_id"], "method": method}
                     for cell in cells for method in cell["methods"]],
        "benchmark_eligible": False, "formal_evaluation_ready": False,
    })


def manifest_digest(manifest):
    require(type(manifest) is dict, "invalid manifest")
    expected = build_manifest(manifest.get("run_id"))
    require(canonical(manifest) == canonical(expected), "manifest changed")
    return checksum(manifest)


def tool_request(cell, method, run_id):
    identity(run_id)
    identity(cell["episode_id"])
    require(method in ("active_verify", "always_verify"), "no tool permitted")
    body = {
        "run_id": run_id, "episode_id": cell["episode_id"], "method": method,
        "view_sha256": checksum(cell["view"]),
    }
    return {**body, "request_id": checksum(body)}


def validate_tool(cell, method, event, run_id):
    fields(event, ("kind", "request_id", "status", "results", "executed_cases", "wall_ns"))
    require(event["request_id"] == tool_request(cell, method, run_id)["request_id"], "tool binding")
    require(event["status"] in ("completed", "error", "timeout", "unavailable"), "tool status")
    if event["wall_ns"] is not None:
        natural(event["wall_ns"])
    count = event["executed_cases"]
    if count is not None:
        natural(count)
        require(count <= 2, "tool case budget")
    checks = cell["view"]["check_inputs"]
    rows = event["results"]
    require(type(rows) is list and len(rows) <= 2, "tool rows")
    if event["status"] == "completed":
        require(count == 2 and len(rows) == 2, "successful tool incomplete")
    else:
        require(rows == [], "failed tool must not invent rows")
    for row, check in zip(rows, checks):
        fields(row, ("id", "input", "expected", "actual", "status"))
        require(row["id"] == check["id"] and canonical(row["input"]) == canonical(check["input"]),
                "tool input binding")
        require(row["status"] == "ok", "tool row status")
        require(type(row["expected"]) is int and type(row["actual"]) is int, "tool values")


def next_step(cell, method, events, run_id):
    identity(run_id)
    require(method in METHODS and type(events) is list, "invalid method/events")
    view = cell["view"]
    initial = next(letter for letter, source in view["candidates"].items()
                   if source == view["initial_source"])
    pending = "done" if method == "keep" else "tool" if method == "always_verify" else "model"
    selected = initial if method == "keep" else None
    status = "kept" if method == "keep" else "pending"
    model_calls = tool_calls = 0
    executed_cases = 0
    final = False
    for position, event in enumerate(events):
        require(pending != "done", "event after terminal")
        require(type(event) is dict and event.get("kind") == pending, "event order")
        if pending == "tool":
            require(tool_calls == 0, "second tool")
            validate_tool(cell, method, event, run_id)
            tool_calls += 1
            executed_cases = event["executed_cases"]
            pending, final = "model", True
            continue
        require(model_calls < 2, "too many model calls")
        fields(event, (*MODEL_FIELDS, "request_id", "request_sha256"))
        expected_request = _render_request(cell, method, events[:position], run_id)
        require(event["request_id"] == expected_request["id"]
                and event["request_sha256"] == checksum(expected_request), "model event binding")
        action = model_action(
            {name: event[name] for name in MODEL_FIELDS}, method == "active_verify" and not final, LIMITS
        )
        model_calls += 1
        if action == "VERIFY":
            pending = "tool"
        elif action in ("A", "B", "C"):
            if method == "rethink" and not final:
                final = True
            else:
                selected, status, pending = action, "valid", "done"
        else:
            status, pending = action, "done"
    return {
        "next": pending, "complete": pending == "done", "status": status,
        "selected": selected, "deployed": (selected or initial) if pending == "done" else None,
        "model_calls": model_calls, "tool_calls": tool_calls, "executed_cases": executed_cases,
    }


def model_request(cell, method, events, run_id):
    state = next_step(cell, method, events, run_id)
    require(state["next"] == "model", "no model request pending")
    return _render_request(cell, method, events, run_id)


def _render_request(cell, method, events, run_id):
    request_id = checksum([
        run_id, cell["episode_id"], method, sum(event["kind"] == "model" for event in events)
    ])[:32]
    instruction = (
        "Choose the candidate implementing the specification. Feedback can be wrong. "
        "A fixed check is not proof of whole-program correctness. "
    )
    if method == "active_verify" and not events:
        instruction += (
            "Return exactly A, B, C, or VERIFY. VERIFY checks only the initial program "
            "on the listed inputs once, then you must choose A, B, or C."
        )
    elif method == "rethink" and not events:
        instruction += "Give a tentative choice. Return exactly A, B, or C."
    elif method == "rethink":
        instruction += "Reconsider your tentative choice with no new evidence. Return exactly A, B, or C."
    else:
        instruction += "Make your final choice. Return exactly A, B, or C. No further check is available."
    history = []
    for event in events:
        if event["kind"] == "model":
            history.append({"previous_response": event["text"]})
        else:
            history.append({"fixed_check": {
                name: event[name] for name in ("status", "results", "executed_cases")
            }})
    content = instruction + "\n\n" + json.dumps(
        {"task": cell["view"], "history": history}, ensure_ascii=False, sort_keys=True, allow_nan=False
    )
    require(len(content) <= 32000, "request text limit")
    return {"id": request_id, "messages": [{"role": "user", "content": content}]}


def _cost(events, complete, episode_wall):
    models = [event for event in events if event["kind"] == "model"]
    tools = [event for event in events if event["kind"] == "tool"]
    values = {
        "model_calls": [len(models)], "tool_calls": [len(tools)],
        "executed_cases": [event["executed_cases"] for event in tools],
        "input_tokens": [None if event["input_ids"] is None else len(event["input_ids"])
                         for event in models],
        "output_tokens": [None if event["output_ids"] is None else len(event["output_ids"])
                          for event in models],
        "component_wall_ns": [event["wall_ns"] for event in events],
        "episode_wall_ns": [episode_wall],
    }
    return {
        name: {
            "total": sum(parts) if complete and all(part is not None for part in parts) else None,
            "observed": sum(part for part in parts if part is not None),
            "unknown_episodes": int(not complete or any(part is None for part in parts)),
        } for name, parts in values.items()
    }


def _summary(rows):
    counts = ("complete", "valid", "selection_pass_count", "deployment_pass_count",
              "deployment_unknown", "repair_count", "regression_count", "kept", "changed",
              "case_repairs", "case_regressions", "verify_requested", "tool_completed", "tool_failed")
    result = {name: sum(row[name] for row in rows) for name in counts}
    statuses = Counter(row["status"] for row in rows)
    scheduled = len(rows)
    result.update({
        "scheduled": scheduled, "status_counts": dict(statuses),
        "missing": statuses["missing"], "incomplete": statuses["incomplete"],
        "repair_denominator": sum(not row["initial_pass"] for row in rows),
        "regression_denominator": sum(row["initial_pass"] for row in rows),
        "selection_pass_rate": result["selection_pass_count"] / scheduled,
        "deployment_pass_rate_lower_bound": result["deployment_pass_count"] / scheduled,
        "cost": {},
    })
    for rate, count, denominator in (
        ("repair_rate", "repair_count", "repair_denominator"),
        ("regression_rate", "regression_count", "regression_denominator"),
    ):
        result[rate] = result[count] / result[denominator] if result[denominator] else None
    for name in rows[0]["cost"]:
        parts = [row["cost"][name] for row in rows]
        result["cost"][name] = {
            "total": (sum(part["total"] for part in parts)
                      if all(part["total"] is not None for part in parts) else None),
            "observed": sum(part["observed"] for part in parts),
            "unknown_episodes": sum(part["unknown_episodes"] for part in parts),
        }
    return result


def score(manifest, frozen_digest, records, audit):
    from development_cpu import compare

    require(manifest_digest(manifest) == frozen_digest, "manifest digest mismatch")
    fields(audit, ("manifest", "observation"))
    require(audit["manifest"]["mode"] == "audit", "full audit required")
    comparison = compare(audit["manifest"], audit["observation"])
    require(comparison["admitted"], "development audit not admitted")
    tasks = {task["task_id"]: task for task in audit["manifest"]["tasks"]}
    actual = {
        (row["task_id"], row["program"], row["case_id"]): row["value"]
        for row in audit["observation"]["records"]
    }
    cells = {cell["episode_id"]: cell for cell in manifest["cells"]}
    bound = {}
    require(type(records) is list, "record list required")
    for record in records:
        fields(record, ("episode_id", "method", "manifest_sha256", "events", "episode_wall_ns"))
        key = (record["episode_id"], record["method"])
        require(record["episode_id"] in cells and record["method"] in METHODS, "unknown episode/method")
        require(key not in bound and record["manifest_sha256"] == frozen_digest, "record binding")
        if record["episode_wall_ns"] is not None:
            natural(record["episode_wall_ns"])
        state = next_step(cells[key[0]], key[1], record["events"], manifest["run_id"])
        bound[key] = (record, state)
    rows = []
    for scheduled in manifest["schedule"]:
        cell = cells[scheduled["episode_id"]]
        method = scheduled["method"]
        pair = bound.get((cell["episode_id"], method))
        record, state = pair if pair is not None else (None, None)
        events = record["events"] if record is not None else []
        task = tasks[cell["task_id"]]
        cases = {case["id"]: case for case in task["private"]["cases"]}
        pass_by_letter = {
            letter: [
                actual[(task["task_id"], role, case_id)] == cases[case_id]["expected"]
                for case_id in task["private"]["score_ids"]
            ] for letter, role in zip("ABC", cell["permutation"])
        }
        for event in events:
            if event["kind"] != "tool" or event["status"] != "completed":
                continue
            for result, probe in zip(event["results"], task["check_inputs"]):
                require(result["expected"] == cases[probe["id"]]["expected"], "tool reference mismatch")
                require(result["actual"] == actual[(task["task_id"], cell["initial"], probe["id"])],
                        "tool actual mismatch")
        initial_letter = "ABC"[cell["permutation"].index(cell["initial"])]
        before = pass_by_letter[initial_letter]
        selected = state["selected"] if state else None
        deployed = state["deployed"] if state else None
        after = pass_by_letter[selected] if selected else None
        success = after is not None and all(after)
        complete = state is not None and state["complete"]
        rows.append({
            **{name: cell[name] for name in (
                "episode_id", "task_id", "family_id", "initial", "condition", "verification_information"
            )},
            "method": method, "selected": selected, "deployed": deployed,
            "status": "missing" if state is None else state["status"] if complete else "incomplete",
            "complete": complete, "valid": selected is not None, "initial_pass": all(before),
            "selection_pass_count": int(success),
            "deployment_pass_count": int(deployed is not None and all(pass_by_letter[deployed])),
            "deployment_unknown": int(deployed is None),
            "repair_count": int(not all(before) and success),
            "regression_count": int(all(before) and after is not None and not success),
            "kept": int(selected == initial_letter), "changed": int(selected is not None and selected != initial_letter),
            "case_repairs": sum(not old and new for old, new in zip(before, after)) if after else 0,
            "case_regressions": sum(old and not new for old, new in zip(before, after)) if after else 0,
            "verify_requested": int(method == "active_verify" and bool(events)
                                    and model_action({name: events[0][name] for name in MODEL_FIELDS},
                                                     True, LIMITS) == "VERIFY"),
            "tool_completed": sum(event["kind"] == "tool" and event["status"] == "completed" for event in events),
            "tool_failed": sum(event["kind"] == "tool" and event["status"] != "completed" for event in events),
            "cost": _cost(events, complete, record["episode_wall_ns"] if record else None),
        })
    by_method = {method: _summary([row for row in rows if row["method"] == method]) for method in METHODS}
    comparisons = {}
    active = by_method["active_verify"]
    for method in METHODS[:-1]:
        other = by_method[method]
        comparisons[method] = {
            "direction": "active_verify minus " + method, "scheduled_pairs": len(cells),
            "complete_pairs": sum(
                all(bound.get((cell_id, strategy), ({}, {"complete": False}))[1]["complete"]
                    for strategy in ("active_verify", method)) for cell_id in cells
            ),
            **{name + "_delta": active[name] - other[name] for name in (
                "selection_pass_rate", "repair_rate", "regression_rate"
            )},
            "cost_delta": {
                name: None if active["cost"][name]["total"] is None or other["cost"][name]["total"] is None
                else active["cost"][name]["total"] - other["cost"][name]["total"]
                for name in active["cost"]
            },
        }
    return {
        "kind": "FIXED_ONLINE_DEVELOPMENT_SCORE", "manifest_sha256": frozen_digest,
        "scheduled_records": len(rows), "complete_records": sum(row["complete"] for row in rows),
        "by_method": by_method, "comparisons": comparisons, "rows": rows,
        "strata": {
            dimension: {
                value: {method: _summary([
                    row for row in rows if row[dimension] == value and row["method"] == method
                ]) for method in METHODS}
                for value in sorted({row[dimension] for row in rows})
            } for dimension in ("family_id", "condition", "initial", "verification_information")
        },
        "benchmark_eligible": False, "formal_evaluation_ready": False,
        "net_benefit_established": False, "isolated": False,
    }
