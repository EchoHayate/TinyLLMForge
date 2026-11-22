# Development Schedule Implementation Plan

> **For agentic workers:** Use subagent-driven-development with independent review.

**Goal:** Freeze eight development episodes and their full final-case inventory.

**Architecture:** A pure scheduling module combines the original public renderer
with pinned private task metadata on the trusted side. It emits public messages
and hash-only final-case inventory; it never generates or executes code.

**Tech Stack:** Existing Python 3.12, unittest and frozen-data helpers.

## Global Constraints

Implements the fixed-order portion of
`../specs/2026-11-17-lora-development-coordinator-design.md`.
Only create `development_schedule.py` and `test_development_schedule.py`.
No sealed edits, real private-data reads, model/remote/worker execution, training,
installs, telemetry, commits or pushes. Use apply_patch and no inline code comments.
Only D01–D04/base/text; alternating order; exactly eight episodes/sixteen call IDs.
Returned messages must be byte-equivalent under canonical encoding to the
original public renderer. No new example selection, prompt changes or score data.
This module is a pure plan, not an execution grant or an observed experiment.

## Task 1: Frozen Schedule

**Create:** `experiments/lora_necessity/development_schedule.py`

**Test:** `experiments/lora_necessity/test_development_schedule.py`

**Consumes:** `episode_jobs`, `PUBLIC_FIELDS`, `PINS`, `_tasks`, `load_frozen`,
`digest`, `encoded`, `require`, accepted `development_final_scoring._cases`.

**Produces:** `build_schedule(supervision_raw, prompts_raw, source_raw,
manifest_raw, *, run_id)` returning the schema in the implementation below.

- [ ] Write synthetic fixtures and behavioral RED assertions before implementation.
Use `import test_development_final_scoring as scoring_fixtures` only after that
component is accepted. Reuse its synthetic_tasks/frozen_bytes functions through
the module reference, not its TestCase class. Add `spec="Synthetic identity task"`
and a synthetic initial_witness to copies of all task dicts; replace supervision
reference strings with a literal toy function string. Build the four views:

```python
def public(task):
    return {
        "id": task["id"], "entry_point": task["entry_point"],
        "initial_source": task["initial_source"], "spec": task["spec"],
        "public_cases": [
            {name: copy.deepcopy(case[name]) for name in ("id", "args", "expected")}
            for case in task["public_cases"]
        ],
    }

prompts = [public(task) for task in tasks if task["split"] != "supervision"]
supervision = [
    {**public(task), "reference_source": task["reference_source"],
     "initial_witness": {"public_case_id": "P01",
                         "predicted_initial": {"kind": "return", "value": 99}}}
    for task in tasks if task["split"] == "supervision"
]
```

Encode all views and patch the four existing PINS entries only within each test.
Do not read real data. Exercise actual `_tasks` and `episode_jobs`, not mocked
returned lists except a specifically named malformed-renderer negative test.

- [ ] Retain the missing-component RED with an explicit setUp file assertion:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error \
  -m unittest discover -s experiments/lora_necessity -p test_development_schedule.py -v
```

- [ ] Implement the pure module:

```python
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
```

- [ ] Tests require the exact eight-item alternating order and ordinal0..7,
sixteen generation IDs and eight public attempt IDs, their independently
constructed digests, and changes under a different run_id.
Compare every message/examples digest against the original renderer.
Check base has no examples and text retains all eight original examples.
Hash-only final inventory must retain public/scoring union and sorted order.
Private scoring values/reasons/reference strings must not appear in returned
messages or final inventory. Supervision references in text are allowed and
must not be mistaken for leaked private D/T/R references.

- [ ] Test mismatched public/private projection (entry/spec/source/public args/
expected), any raw hash drift, invalid run IDs including subclasses, duplicate
or missing cases, missing development task, malformed renderer coverage,
returned-object mutation including cross-arm aliasing, stable repeated planning,
and cold imports with no Torch/Transformers. Freeze all test pin patches with
cleanup so the next suite sees unchanged actual PINS.

- [ ] Run GREEN and focused final protocol/scoring/schedule regression:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error \
  -m unittest discover -s experiments/lora_necessity -p test_development_schedule.py -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lora_necessity \
  /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error -m unittest \
  test_development_final_protocol test_development_final_scoring \
  test_development_schedule -v
```

- [ ] Save commands, logs, hashes, scope and concerns in
`/tmp/tlf-candidate-executor-20261117-gb1gZX/development-schedule-report.md`.
Independent review is required before use by the coordinator.

## Task 2: Handoff

- [ ] Record accepted schedule identities in the existing private stage ledger.
- [ ] The next coordinator consumes `episodes` in its exact order, verifies
final-case inventory against the scoring plan and retains missing denominators.
Do not run generators or candidates as part of schedule qualification.
