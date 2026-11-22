import copy
import re

from development_final_scoring import _cases
from pilot_protocol import require
from prompt_rendering import PUBLIC_FIELDS, episode_jobs
from validation_contract import PINS, _tasks, digest, encoded, load_frozen


def _public(task):
    result = {name: copy.deepcopy(task[name]) for name in PUBLIC_FIELDS}
    result["public_cases"] = [
        {name: copy.deepcopy(case[name]) for name in ("id", "args", "expected")}
        for case in task["public_cases"]
    ]
    return result


def _inventory(task):
    return [{
        "case_group": case["case_group"], "case_id": case["case_id"],
        "input_sha256": digest(case["args"]), "expected_sha256": digest(case["expected"]),
    } for case in _cases(task)]


def build_schedule(supervision_raw, prompts_raw, source_raw, manifest_raw, *, run_id):
    require(type(run_id) is str and re.fullmatch("[0-9a-f]{32}", run_id),
            "run identity")
    tasks = {task["id"]: task for task in _tasks(source_raw, manifest_raw)}
    prompts = {task["id"]: task for task in load_frozen("prompts_v1.json", prompts_raw)}
    jobs = episode_jobs(supervision_raw, prompts_raw)
    selected = [job for job in jobs if job["task_id"].startswith("D")]
    expected = {(f"D0{ordinal}", arm) for ordinal in range(1, 5)
                for arm in ("base", "text")}
    require(len(selected) == 8
            and {(job["task_id"], job["arm"]) for job in selected} == expected,
            "development episode coverage")
    indexed = {(job["task_id"], job["arm"]): job for job in selected}
    rows = []
    identifiers = set()
    for ordinal in range(1, 5):
        task_id = f"D0{ordinal}"
        require(task_id in tasks and task_id in prompts, "missing development task")
        task = tasks[task_id]
        require(task["split"] == "development", "development split")
        inventory = _inventory(task)
        require(encoded(_public(task)) == encoded({
            name: prompts[task_id][name] for name in PUBLIC_FIELDS
        }), "public and private task views differ")
        arms = ("base", "text") if ordinal % 2 else ("text", "base")
        for arm in arms:
            identity = {"run_id": run_id, "task_id": task_id, "arm": arm}
            calls = [
                digest({"kind": "LORA_DEVELOPMENT_GENERATION_ID", "version": 1,
                        **identity, "round": round_number})[:32]
                for round_number in (1, 2)
            ]
            attempt = digest({"kind": "LORA_DEVELOPMENT_PUBLIC_ATTEMPT",
                              "version": 1, **identity, "round": 1})[:32]
            for identifier in (*calls, attempt):
                require(identifier not in identifiers, "duplicate logical identity")
                identifiers.add(identifier)
            job = copy.deepcopy(indexed[(task_id, arm)])
            rows.append({
                "ordinal": len(rows), **job, "task_sha256": digest(task),
                "messages_sha256": digest(job["messages"]),
                "first_call_id": calls[0], "second_call_id": calls[1],
                "public_attempt_id": attempt, "scheduled_final_cases": len(inventory),
                "final_cases": copy.deepcopy(inventory),
            })
    return {
        "kind": "LORA_DEVELOPMENT_SCHEDULE", "version": 1, "run_id": run_id,
        "data_sha256": dict(PINS), "scheduled_episodes": 8,
        "scheduled_generation_calls": 16, "episodes": rows,
    }
