# Development Final Scoring Implementation Plan

> **For agentic workers:** Use subagent-driven-development with independent review.

**Goal:** Bind the final candidate to the frozen full case set and reduce receipts.

**Architecture:** A pure planner loads the pinned source/manifest using the existing
contract. A pure reducer rebuilds the plan instead of trusting a caller-edited one.
Neither function runs source, loads a model, writes artifacts or grants execution.

**Tech Stack:** Existing Python 3.12, unittest, standard library, sealed pure helpers.

## Global Constraints

Follow `../specs/2026-11-17-lora-development-final-scoring-design.md`.
Only create `development_final_scoring.py` and `test_development_final_scoring.py`
under `experiments/lora_necessity`. Do not alter sealed files or accepted components.
No actual private data access, candidate execution, remote/model operations,
installs, telemetry, training, commits or pushes.
Tests supply synthetic source/manifest bytes with synthetic pins patched only in
test scope; preserve the production fixed-pin loader and its actual validation.
Do not replace `_tasks` with a fake task list in all tests: cover the real loader
path on synthetic 44-task metadata and require a pin mismatch to fail.
Only D01–D04/base/text; public-plus-scoring union; exact final candidate, no repair.
Functional pass is not runtime attestation, calibration or LoRA benefit.

## Task 1: Pure Planner and Reducer

**Create:** `experiments/lora_necessity/development_final_scoring.py`

**Test:** `experiments/lora_necessity/test_development_final_scoring.py`

**Consumes:** `validation_contract._tasks/PINS/parse_json/encoded/digest`,
`candidate_feedback._valid/_outcome/EXECUTION_ERRORS`,
`candidate_execution_protocol._parse_result`,
`development_final_protocol.prepare_request`, `pilot_protocol.fields/require/FLAGS`.

**Produces:**
`build_final_plan(source_raw, manifest_raw, *, run_id, task_id, arm,
generation_call_id, generation_raw, expected_generation_sha256)`
and `score_final(source_raw, manifest_raw, receipts_raw, **identity)`.
The second function accepts exactly the same keyword identity arguments as the
first, explicitly or through a private forwarding wrapper; unknown kwargs fail.

- [ ] Create RED tests with a wholly synthetic task:

```python
task = {
    "id": "D01", "split": "development", "family": "family0",
    "template_id": "syntheticD01", "entry_point": "toy",
    "initial_source": "def toy(value):\n    return value\n",
    "public_cases": [
        {"id": "P02", "args": [2], "expected": {"kind": "return", "value": 2},
         "reason": "synthetic second public case"},
        {"id": "P01", "args": [1], "expected": {"kind": "return", "value": 1},
         "reason": "synthetic first public case"},
    ],
    "scoring_cases": [
        {"id": "H01", "args": [3], "expected": {"kind": "return", "value": 3},
         "reason": "synthetic scoring case"},
    ],
}
generation = {
    "kind": "LORA_GREEDY_CANDIDATE_UNATTESTED",
    "candidate": "def toy(value):\n    return value\n",
    "candidate_sha256": hashlib.sha256(
        b"def toy(value):\n    return value\n").hexdigest(),
}
```

Construct the other 43 synthetic tasks by unique IDs/templates, family-balanced
according to `SPLITS`. S/D/T share the four family names; R uses four disjoint
families. Build manifest metadata with `build_manifest` and `digest(task)`.
Create source `{"tasks": tasks}` and manifest `{"cpu_manifest": cpu_manifest}`,
encode both, then patch `validation_contract.PINS` entries to their byte hashes
only within each test. This exercises actual `_tasks` integrity/metadata binding
without opening any real data file. Give each copied task unique metadata.

Check initial successful plan requests are P01/P02/H01 in that order. Independently
construct the specified case-set object and attempt objects in tests and compare
digests. Build completed receipts from synthetic expected outcomes and assert pass;
change exactly one value and assert fail. Tests must also check that results do
not contain `reason`, `reference_source` or `expected_outcomes`.

- [ ] Run missing-component RED:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error \
  -m unittest discover -s experiments/lora_necessity \
  -p test_development_final_scoring.py -v
```

- [ ] Implement small pure helpers and planner. This sample gives the complete
flow; preserve the spec and add explicit validation where a focused test exposes
an actual gap rather than broadening the API:

```python
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
        items = task[name]
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
            and generation.get("kind") == "LORA_GREEDY_CANDIDATE_UNATTESTED",
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
```

- [ ] Add the reducer. Private expectations exist in the internal plan only.
The score retains case identities even for invalid/missing observations:

```python
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
```

- [ ] Extend synthetic tests: bad generation byte/candidate hash, unknown identity,
extra old-domain/LoRA/T/R requests, duplicate/missing public or scoring cases,
permuted input ordering, changed task/args/expected changing bound hashes,
candidate-signature mismatch, Markdown/empty/loop/raising source, UTF-8/surrogate
boundaries, no source invocation, and canonical value type differences.
All generation fixtures are synthetic JSON; no Torch import or model inference.

- [ ] Exercise every receipt status, valid ValueError, bool versus int versus float,
all-pass/all-fail/mixed outcomes, strict prefixes, reordering/duplicates/extra rows,
missing-middle with later receipt, malformed JSON/duplicate keys/NaN/Infinity,
and unrecognized/extra receipt fields. Wrong digest raises rather than becoming
`mismatch`. Missing observations stay `missing`, never pass.
Check outcome canonical byte/depth limits through the real sealed result parser.
Mutating returned plans/results must not affect future planning/scoring. Test an
edited plan cannot be supplied to `score_final` to bypass the frozen reconstruction.

- [ ] Run focused GREEN and pure final protocol/scoring aggregate:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error \
  -m unittest discover -s experiments/lora_necessity \
  -p test_development_final_scoring.py -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lora_necessity \
  /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error \
  -m unittest test_development_final_protocol test_development_final_scoring -v
```

- [ ] Save commands, RED/GREEN logs, hashes, test scope and limits to
`/tmp/tlf-candidate-executor-20261117-gb1gZX/development-final-scoring-report.md`.
Main performs an independent spec/quality review before any real use.
No real case count, development score, launch readiness or effectiveness claim.

## Task 2: Handoff

- [ ] Record accepted source/review identities in the stage progress ledger.
- [ ] Subsequent coordinator must authenticate generation and execution receipts,
keep expectations out of generation, freeze order and classify calibration.
No real private material is loaded or experiment launched by this plan.
