import copy
from collections import Counter

from active_verification_contract import METHODS, fields, identity, model_action, require
from active_verification_manifest import manifest_digest
from active_verification_state import known_sum, replay


COST_FIELDS = (
    "model_calls", "tool_calls", "executed_cases", "input_tokens",
    "output_tokens", "component_wall_ns", "episode_wall_ns",
)
DIMENSIONS = (
    "condition", "initial", "family_id", "domain", "task_id", "verification_information",
)
COUNT_FIELDS = (
    "observed", "complete", "valid", "selection_pass_count", "deployment_pass_count",
    "deployment_unknown", "repair_count", "regression_count", "kept", "changed",
    "selection_case_repairs", "selection_case_regressions",
    "verify_requested", "tool_completed", "tool_failed",
)


def bind_transcript(manifest, episode_id, method, events):
    digest = manifest_digest(manifest)
    identity(episode_id)
    require(type(method) is str and method in METHODS, "unknown method")
    cells = {cell["episode_id"]: cell for cell in manifest["cells"]}
    require(episode_id in cells, "unknown episode")
    replay(cells[episode_id]["view"], method, manifest["run_id"],
           episode_id, events, manifest["limits"])
    return copy.deepcopy({
        "evidence_kind": "synthetic", "manifest_sha256": digest,
        "run_id": manifest["run_id"], "episode_id": episode_id,
        "method": method, "events": events,
    })


def episode_cost(state, events):
    models = [event for event in events if event["kind"] == "model"]
    tools = [event for event in events if event["kind"] == "tool"]
    observed = {
        "model_calls": len(models), "tool_calls": len(tools),
        "executed_cases": sum(event["executed_cases"] for event in tools),
        "input_tokens": sum(len(event["input_ids"]) for event in models if event["input_ids"] is not None),
        "output_tokens": sum(len(event["output_ids"]) for event in models if event["output_ids"] is not None),
        "component_wall_ns": sum(event["wall_ns"] for event in events if event["wall_ns"] is not None),
        "episode_wall_ns": 0,
    }
    result = {}
    for name in COST_FIELDS:
        total = state["cost"][name] if state is not None and state["complete"] else None
        result[name] = {
            "total": total, "observed": observed[name],
            "unknown_episodes": int(total is None),
        }
    return result


def scored_row(cell, method, state, limits):
    events = [] if state is None else state["events"]
    selected = None if state is None else state["selected"]
    deployed = None if state is None else state["deployed"]
    initial_letter = next(
        letter for letter, source in cell["view"]["candidates"].items()
        if source == cell["view"]["initial_source"]
    )
    initial_cases = cell["pass_by_letter"][initial_letter]
    selected_cases = None if selected is None else cell["pass_by_letter"][selected]
    initial_pass = all(initial_cases)
    selection_pass = selected_cases is not None and all(selected_cases)
    deployed_pass = None if deployed is None else all(cell["pass_by_letter"][deployed])
    status = "missing" if state is None else (
        state["status"] if state["complete"] else "incomplete"
    )
    return {
        **{name: cell[name] for name in DIMENSIONS},
        "episode_id": cell["episode_id"], "method": method,
        "status": status, "observed": state is not None,
        "complete": state is not None and state["complete"], "valid": selected is not None,
        "selected": selected, "deployed": deployed, "initial_pass": initial_pass,
        "selection_pass_count": int(selection_pass),
        "deployment_pass_count": int(deployed_pass is True),
        "deployment_unknown": int(deployed_pass is None),
        "repair_count": int(not initial_pass and selection_pass),
        "regression_count": int(initial_pass and selected is not None and not selection_pass),
        "kept": int(selected == initial_letter),
        "changed": int(selected is not None and selected != initial_letter),
        "selection_case_repairs": 0 if selected_cases is None else sum(
            not before and after for before, after in zip(initial_cases, selected_cases)
        ),
        "selection_case_regressions": 0 if selected_cases is None else sum(
            before and not after for before, after in zip(initial_cases, selected_cases)
        ),
        "verify_requested": int(
            method == "active_verify" and bool(events)
            and model_action(events[0], True, limits) == "VERIFY"
        ),
        "tool_completed": sum(event["kind"] == "tool" and event["status"] == "completed" for event in events),
        "tool_failed": sum(event["kind"] == "tool" and event["status"] != "completed" for event in events),
        "events": events, "cost": episode_cost(state, events),
    }


def summarize_rows(rows):
    counts = {name: sum(row[name] for row in rows) for name in COUNT_FIELDS}
    statuses = dict(sorted(Counter(row["status"] for row in rows).items()))
    repair_denominator = sum(not row["initial_pass"] for row in rows)
    regression_denominator = sum(row["initial_pass"] for row in rows)
    return {
        **counts, "scheduled": len(rows),
        "task_count": len({row["task_id"] for row in rows}),
        "family_count": len({row["family_id"] for row in rows}),
        "missing": statuses.get("missing", 0), "incomplete": statuses.get("incomplete", 0),
        "status_counts": statuses,
        "repair_denominator": repair_denominator,
        "regression_denominator": regression_denominator,
        "repair_rate": counts["repair_count"] / repair_denominator if repair_denominator else None,
        "regression_rate": counts["regression_count"] / regression_denominator if regression_denominator else None,
        "selection_pass_rate": counts["selection_pass_count"] / len(rows) if rows else None,
        "protocol_success_rate": counts["valid"] / len(rows) if rows else None,
        "completion_rate": counts["complete"] / len(rows) if rows else None,
        "deployment_pass_rate_lower_bound": counts["deployment_pass_count"] / len(rows) if rows else None,
        "verify_request_rate": counts["verify_requested"] / len(rows) if rows else None,
        "tool_completion_rate": (
            counts["tool_completed"] / (counts["tool_completed"] + counts["tool_failed"])
            if counts["tool_completed"] + counts["tool_failed"] else None
        ),
        "cost": {
            name: {
                "total": known_sum([row["cost"][name]["total"] for row in rows]),
                "observed": sum(row["cost"][name]["observed"] for row in rows),
                "unknown_episodes": sum(row["cost"][name]["unknown_episodes"] for row in rows),
            } for name in COST_FIELDS
        },
    }


def difference(left, right):
    return None if left is None or right is None else left - right


def paired_comparison(rows, comparator):
    active_rows = {row["episode_id"]: row for row in rows if row["method"] == "active_verify"}
    other_rows = {row["episode_id"]: row for row in rows if row["method"] == comparator}
    active = summarize_rows(list(active_rows.values()))
    other = summarize_rows(list(other_rows.values()))
    scheduled = len(active_rows)
    return {
        "comparator": comparator, "direction": "active_verify minus comparator",
        "scheduled_pairs": scheduled,
        "complete_pairs": sum(
            row["complete"] and other_rows[episode_id]["complete"]
            for episode_id, row in active_rows.items()
        ),
        "selection_pass_rate_delta": (
            (active["selection_pass_count"] - other["selection_pass_count"]) / scheduled
            if scheduled else None
        ),
        "repair_rate_delta": difference(active["repair_rate"], other["repair_rate"]),
        "regression_rate_delta": difference(active["regression_rate"], other["regression_rate"]),
        "deployment_pass_rate_delta": (
            (active["deployment_pass_count"] - other["deployment_pass_count"]) / scheduled
            if scheduled and not active["deployment_unknown"] and not other["deployment_unknown"]
            else None
        ),
        "cost_delta": {
            name: difference(active["cost"][name]["total"], other["cost"][name]["total"])
            for name in COST_FIELDS
        },
        "descriptive_only": True, "net_benefit_established": False,
    }


def score_batch(manifest, expected_sha256, records):
    digest = manifest_digest(manifest)
    require(type(expected_sha256) is str and expected_sha256 == digest, "frozen digest mismatch")
    require(type(records) is list, "records must be a list")
    cells = {cell["episode_id"]: cell for cell in manifest["cells"]}
    states = {}
    for record in records:
        fields(record, (
            "evidence_kind", "manifest_sha256", "run_id", "episode_id", "method", "events",
        ))
        require(record["evidence_kind"] == "synthetic", "synthetic records only")
        require(record["manifest_sha256"] == digest, "record manifest mismatch")
        require(record["run_id"] == manifest["run_id"], "record run mismatch")
        identity(record["episode_id"])
        require(record["episode_id"] in cells, "unknown episode")
        require(type(record["method"]) is str and record["method"] in METHODS, "unknown method")
        key = (record["episode_id"], record["method"])
        require(key not in states, "duplicate transcript")
        states[key] = replay(
            cells[key[0]]["view"], key[1], manifest["run_id"], key[0],
            record["events"], manifest["limits"],
        )
    rows = [
        scored_row(cell, method, states.get((cell["episode_id"], method)), manifest["limits"])
        for cell in manifest["cells"] for method in METHODS
    ]
    by_method = {
        method: summarize_rows([row for row in rows if row["method"] == method])
        for method in METHODS
    }
    strata = {
        dimension: {
            value: {
                method: summarize_rows([
                    row for row in rows if row[dimension] == value and row["method"] == method
                ]) for method in METHODS
            } for value in sorted({row[dimension] for row in rows})
        } for dimension in DIMENSIONS
    }
    overall = summarize_rows(rows)
    overruns = [
        name for name in ("model_calls", "tool_calls", "executed_cases", "input_tokens", "output_tokens")
        if overall["cost"][name]["observed"] > manifest["budgets"][name]
    ]
    comparisons = {
        comparator: {
            **paired_comparison(rows, comparator),
            "per_task": {
                task_id: paired_comparison(
                    [row for row in rows if row["task_id"] == task_id], comparator,
                ) for task_id in sorted({row["task_id"] for row in rows})
            },
        } for comparator in ("rethink", "always_verify", "direct", "keep")
    }
    return copy.deepcopy({
        "evidence_kind": "SYNTHETIC_ONLY", "real_execution_eligible": False,
        "benchmark_eligible": False, "formal_evaluation_ready": False,
        "manifest_sha256": digest, "run_id": manifest["run_id"], "split": manifest["split"],
        "scheduled_cells": len(cells), "scheduled_records": len(rows),
        "task_count": overall["task_count"], "family_count": overall["family_count"],
        "complete": all(row["complete"] for row in rows),
        "all_records_validated": True, "budget_overruns": overruns,
        "budgets": manifest["budgets"], "overall": overall,
        "by_method": by_method, "strata": strata, "comparisons": comparisons, "rows": rows,
    })
