# LoRA Validation Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a frozen 1416-item audit schedule and a strictly simulation-only report without executing any task source.

**Architecture:** Pure functions accept pinned file bytes, derive private controller rows and separately projected worker requests, and compare synthetic observations externally. No executor, runtime callback, filesystem loader or real-receipt promotion exists.

**Tech Stack:** Python 3.12 standard library; colocated `unittest`; existing `pilot_protocol.py`.

**Local implementation:** both tasks completed with independent spec/quality approval
on 2026-11-11. Main combined suite: 48 tests passed; independent whole-component
review also passed. Evidence and closeout status:
`../../../.superpowers/sdd/lora-validation-20261111-progress.md`.
This is not real-execution or benchmark admission.

## Global Constraints

- Approved spec: `docs/superpowers/specs/2026-11-11-lora-validation-design.md`.
- Work in `/Users/bytedance/dev/TinyLLMForge`, preserving the existing experimental branch and unrelated untracked work. No checkout, commit, push, installation or telemetry.
- Use `apply_patch`; no inline code comments or one-letter identifiers.
- 8 supervision / 4 development / 16 transfer / 16 retention tasks; 354 cases; four source roles; 1416 scheduled observations.
- Never change the dataset, hashes, source strings, original CPU protocol, screening thresholds or main runtime.
- No task source execution, tokenizer/model loading, subprocess, SSH, network or GPU operations.
- Keep `benchmark_eligible`, `formal_evaluation_ready`, `net_benefit_established`, `isolated` false.
- Missing observations stay in the denominator; runtime errors do not count as semantic witnesses.
- This plan implements spec sections 3–5 and the controller portion of section 8. Prompt/token work is a separate plan.
- Parser byte limits below bound local parsing only; they are not backend CPU, memory or output enforcement.
- The implementation snippets are planned code, not executed or reviewed implementation evidence. Preserve TDD order; do not extract all files before RED.

## File Map And Dependencies

1. `experiments/lora_necessity/validation_contract.py`: pinned bytes, schedule, request projection, answer-only audit projection.
2. `experiments/lora_necessity/test_validation_contract.py`: identity, counts, isolation of fields, copy and malformed-input checks.
3. `experiments/lora_necessity/validation_report.py`: bounded parsing and simulation-only comparison.
4. `experiments/lora_necessity/test_validation_report.py`: synthetic full matrix, witnesses, incomplete and hostile observations.

Task 2 depends on Task 1. The separate prompt plan imports only the pure helpers and
public-file pins from Task 1; importing this module must never read private data.
Main owns plan status and final README updates. Workers own only their named source/test pair.

## Task 1: Frozen Schedule And Separate Projections

**Files:** create the first two files in the map.

**Interfaces:**
- `encoded(value) -> bytes`, `digest(value) -> str`: compact sorted ASCII JSON and SHA256.
- `parse_json(raw: bytes, limit: int = 1048576) -> object`: duplicate keys/nonfinite values/invalid UTF-8 rejected as `ValueError`.
- `load_frozen(name: str, raw: bytes) -> object`: accepts only the four named pinned files below, no file I/O.
- `build_schedule(source_raw: bytes, manifest_raw: bytes) -> dict`: private schedule with 1416 rows.
- `worker_requests(source_raw: bytes, manifest_raw: bytes) -> list[dict]`: only opaque ID, entry, source and args.
- `reference_jobs(source_raw: bytes, manifest_raw: bytes) -> list[dict]`: 44 ID/source pairs, private tokenizer audit only.

- [x] **1. Add the following tests first.**

```python
import copy
import importlib
import json
from pathlib import Path
import unittest


class ValidationContractTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("validation_contract.py").is_file(),
                        "validation contract not implemented")
        self.module = importlib.import_module("validation_contract")
        self.data = Path(__file__).with_name("data")
        self.source = (self.data / "task_sources_v1.json").read_bytes()
        self.manifest = (self.data / "manifest_v1.json").read_bytes()

    def test_fixed_schedule_and_order(self):
        plan = self.module.build_schedule(self.source, self.manifest)
        self.assertEqual(len(plan["rows"]), 1416)
        self.assertEqual(plan["rows"][0]["task_id"], "D01")
        self.assertEqual(plan["rows"][0]["role"], "reference")
        self.assertEqual([row["ordinal"] for row in plan["rows"]], list(range(1416)))
        self.assertEqual(sum(row["role"] == "reference" for row in plan["rows"]), 354)
        self.assertEqual(sum(row["witness_prediction"] is not None
                             for row in plan["rows"]), 132)

    def test_pin_rejects_tampering_and_reformatted_bytes(self):
        for raw in (self.source + b"\n", self.source.replace(b"count_before", b"count_after")):
            with self.subTest(raw_length=len(raw)), self.assertRaises(ValueError):
                self.module.build_schedule(raw, self.manifest)
        changed = json.loads(self.manifest)
        changed["isolated"] = True
        with self.assertRaises(ValueError):
            self.module.build_schedule(self.source, self.module.encoded(changed))

    def test_work_requests_have_no_private_fields_or_aliases(self):
        requests = self.module.worker_requests(self.source, self.manifest)
        for request in requests:
            self.assertEqual(set(request), {"request_id", "entry_point", "source", "args"})
        original = copy.deepcopy(requests)
        requests[0]["args"].append("sentinel")
        self.assertEqual(self.module.worker_requests(self.source, self.manifest), original)

    def test_reference_jobs_are_private_answer_only_projection(self):
        jobs = self.module.reference_jobs(self.source, self.manifest)
        self.assertEqual(len(jobs), 44)
        self.assertTrue(all(set(job) == {"id", "reference_source"} for job in jobs))
        self.assertEqual(self.module.digest(jobs),
                         "2333003d187a1f12618b4505ca3522b7c2a20e3295c48ae835f3e879ed601cf3")

    def test_strict_json(self):
        for raw in (b'{"a":1,"a":2}', b"NaN", b"Infinity", b"1e999", b"{}\n{}", b"\xff"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.module.parse_json(raw)
        with self.assertRaises(ValueError):
            self.module.parse_json(b"[]", limit=1)
        with self.assertRaises(ValueError):
            self.module.load_frozen("unknown.json", b"{}")
        self.assertNotEqual(self.module.digest(True), self.module.digest(1))
```

- [x] **2. Run RED.**

```sh
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 \
  -m unittest discover -s experiments/lora_necessity -p 'test_validation_contract.py' -v
```

Expected: assertion failures saying `validation contract not implemented`, not task execution.

- [x] **3. Add the complete minimal implementation.**

```python
import copy
import hashlib
import json
import math

from pilot_protocol import build_manifest, canonical, require


PINS = {
    "task_sources_v1.json": "4711c72508b78f5deeafa5ed8de13ec4045d38c8ffbe5f463cb16651c29c693c",
    "manifest_v1.json": "cdac4ae1218701c585f8a38a27cd86583b9d4f71dcc60a3826f9dcfd2a2e7320",
    "supervision_v1.json": "4dafb0ac759697e9b80c5bf097b363082a17a34e6b78fb3d3b91d37a76beb78a",
    "prompts_v1.json": "3668d996bb0e05d600859fd12780b9f582a65aa2ad6b1881fd0bcce9832b5a05",
}


def encoded(value):
    return canonical(value).encode("ascii")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def parse_json(raw, limit=1048576):
    require(type(raw) is bytes and len(raw) <= limit, "JSON byte budget")

    def pairs(entries):
        result = {}
        for name, value in entries:
            require(name not in result, "duplicate JSON key")
            result[name] = value
        return result

    def constant(value):
        raise ValueError("nonfinite JSON value")

    def number(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite JSON number")
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=constant, parse_float=number)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("invalid JSON encoding or depth") from error


def load_frozen(name, raw):
    require(type(name) is str and name in PINS, "unknown frozen file")
    require(type(raw) is bytes and len(raw) <= 1048576, "frozen file bytes")
    require(hashlib.sha256(raw).hexdigest() == PINS[name], "frozen bytes changed")
    return parse_json(raw)


def _tasks(source_raw, manifest_raw):
    source = load_frozen("task_sources_v1.json", source_raw)
    manifest = load_frozen("manifest_v1.json", manifest_raw)
    metadata = [
        {**{name: task[name] for name in ("id", "split", "family", "template_id")},
         "content_sha256": digest(task)}
        for task in source["tasks"]
    ]
    require(encoded(build_manifest(metadata)) == encoded(manifest["cpu_manifest"]),
            "task metadata binding")
    return sorted(source["tasks"], key=lambda task: task["id"])


def _request(row):
    return {
        "request_id": f"request-{row['ordinal']:04d}",
        "entry_point": row["entry_point"], "source": row["source"],
        "args": copy.deepcopy(row["args"]),
    }


def build_schedule(source_raw, manifest_raw):
    rows = []
    for task in _tasks(source_raw, manifest_raw):
        versions = [
            ("reference", task["reference_source"], None, None, None),
            ("initial", task["initial_source"], "public",
             task["initial_witness"]["public_case_id"],
             task["initial_witness"]["predicted_initial"]),
        ]
        versions.extend(
            (mutant["id"], mutant["source"], "scoring", mutant["scoring_case_id"],
             mutant["predicted_mutant"])
            for mutant in sorted(task["mutants"], key=lambda mutant: mutant["id"])
        )
        for role, source, witness_group, witness_id, prediction in versions:
            for group, name in (("public", "public_cases"), ("scoring", "scoring_cases")):
                for case in sorted(task[name], key=lambda case: case["id"]):
                    witness = group == witness_group and case["id"] == witness_id
                    row = {
                        "ordinal": len(rows), "task_id": task["id"], "role": role,
                        "group": group, "case_id": case["id"],
                        "task_sha256": digest(task), "source_sha256":
                        hashlib.sha256(source.encode("utf-8")).hexdigest(),
                        "input_sha256": digest(case["args"]),
                        "expected_sha256": digest(case["expected"]),
                        "entry_point": task["entry_point"], "source": source,
                        "args": copy.deepcopy(case["args"]),
                        "expected": copy.deepcopy(case["expected"]),
                        "witness_prediction": copy.deepcopy(prediction) if witness else None,
                    }
                    if witness:
                        require(encoded(prediction) != encoded(case["expected"]),
                                "witness must disagree")
                    row["request_sha256"] = digest(_request(row))
                    rows.append(row)
    require(len(rows) == 1416, "frozen schedule size")
    return {
        "kind": "LORA_VALIDATION_PLAN", "version": 1,
        "source_sha256": PINS["task_sources_v1.json"],
        "manifest_sha256": PINS["manifest_v1.json"], "rows": rows,
    }


def worker_requests(source_raw, manifest_raw):
    return [_request(row) for row in build_schedule(source_raw, manifest_raw)["rows"]]


def reference_jobs(source_raw, manifest_raw):
    return [{"id": task["id"], "reference_source": task["reference_source"]}
            for task in _tasks(source_raw, manifest_raw)]
```

- [x] **4. Run GREEN with the same command; then run the original CPU suite.**

```sh
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 \
  -m unittest discover -s experiments/lora_necessity -p 'test_pilot_protocol.py' -q
```

Expected: five new tests and the original 26 pass. Record actual results; do not assume counts.

- [x] **5. Review before Task 2.**

Check all imports are inert and requests contain no expectation hashes or other private
fields. Data pins make this deliberately a single frozen-dataset implementation, not a
general source validator. Changes to any dataset version require explicit re-review;
do not add an option to skip hash verification for tests. Record review evidence without committing.

## Task 2: Bounded Observation Parsing And Simulation Report

**Files:** create `validation_report.py` and `test_validation_report.py`.

**Interfaces:**
- Consumes `build_schedule`, `digest`, `encoded`, `parse_json` from Task 1.
- `parse_outcome(raw: bytes) -> dict`: exactly `{"kind":"return","value":...}` or `{"kind":"raises","type":"ValueError"}`.
- `simulate_validation(source_raw: bytes, manifest_raw: bytes, observations_raw: bytes) -> dict`.
- Observation fields: `kind="synthetic"`, integer `ordinal`, `request_sha256`,
  `status`, `payload` string or null, strict bool `cleanup_complete`.
- Statuses: `completed`, `program_error`, `timeout`, `output_limit`, `resource_limit`,
  `protocol_error`, `launch_failed`. Only completed status has a payload.
- Rows must be in ascending order; gaps stay missing. No later row is accepted after
  launch or cleanup failure. Malformed candidate payload becomes `protocol_error`;
  malformed controller metadata raises `ValueError`.

- [x] **1. Add the following tests before the report module.**

```python
import copy
import importlib
from pathlib import Path
import unittest

from validation_contract import build_schedule, encoded


def synthetic_matrix(plan):
    result = []
    for row in plan["rows"]:
        outcome = copy.deepcopy(row["expected"])
        if row["role"] == "initial" and row["group"] == "scoring":
            outcome = {"kind": "return", "value": "__synthetic_wrong__"}
        if row["witness_prediction"] is not None:
            outcome = copy.deepcopy(row["witness_prediction"])
        result.append({
            "kind": "synthetic", "ordinal": row["ordinal"],
            "request_sha256": row["request_sha256"], "status": "completed",
            "payload": encoded(outcome).decode("ascii"), "cleanup_complete": True,
        })
    return result


class ValidationReportTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("validation_report.py").is_file(),
                        "validation report not implemented")
        self.module = importlib.import_module("validation_report")
        data = Path(__file__).with_name("data")
        self.source = (data / "task_sources_v1.json").read_bytes()
        self.manifest = (data / "manifest_v1.json").read_bytes()
        self.plan = build_schedule(self.source, self.manifest)
        self.rows = synthetic_matrix(self.plan)

    def report(self, rows):
        return self.module.simulate_validation(self.source, self.manifest, encoded(rows))

    def test_full_simulation_never_becomes_real(self):
        report = self.report(self.rows)
        self.assertEqual(report["kind"], "LORA_VALIDATION_SIMULATION")
        self.assertEqual(report["simulated_decision"], "CHECKS_SATISFIED")
        self.assertEqual(report["scheduled"], 1416)
        self.assertEqual(len(report["tasks"]), 44)
        for flag in ("benchmark_eligible", "formal_evaluation_ready",
                     "net_benefit_established", "isolated"):
            self.assertIs(report[flag], False)

    def test_missing_does_not_shrink_denominator(self):
        report = self.report(self.rows[1:])
        self.assertEqual(report["scheduled"], 1416)
        self.assertEqual(report["received"], 1415)
        self.assertEqual(report["rows"][0]["status"], "missing")
        self.assertEqual(report["simulated_decision"], "INCOMPLETE")

    def test_reference_and_witness_failures(self):
        self.rows[0]["payload"] = '{"kind":"return","value":"wrong"}'
        self.assertEqual(self.report(self.rows)["simulated_decision"], "CHECKS_FAILED")
        self.rows = synthetic_matrix(self.plan)
        witness = next(row for row in self.plan["rows"] if row["role"] == "M01"
                       and row["witness_prediction"] is not None)
        self.rows[witness["ordinal"]]["payload"] = encoded(witness["expected"]).decode()
        self.assertEqual(self.report(self.rows)["simulated_decision"], "CHECKS_FAILED")
        self.rows[witness["ordinal"]].update(status="timeout", payload=None)
        report = self.report(self.rows)
        self.assertIsNone(report["rows"][witness["ordinal"]]["expected_match"])
        self.assertEqual(report["simulated_decision"], "INCOMPLETE")

    def test_stop_after_infrastructure_failure(self):
        self.rows[0]["cleanup_complete"] = False
        with self.assertRaises(ValueError):
            self.report(self.rows)
        report = self.report(self.rows[:1])
        self.assertEqual(report["stop_ordinal"], 0)
        self.assertEqual(report["simulated_decision"], "INCOMPLETE")

    def test_metadata_rejects_spoofing_duplicate_and_wrong_order(self):
        for name, value in (("kind", "real"), ("ordinal", True),
                            ("request_sha256", "0" * 64), ("cleanup_complete", 1)):
            changed = copy.deepcopy(self.rows[:1])
            changed[0][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.report(changed)
        for rows in ([self.rows[0], self.rows[0]], self.rows[:2][::-1]):
            with self.assertRaises(ValueError):
                self.report(rows)

    def test_outcomes_are_typed_and_bounded(self):
        for raw in (b'{"kind":"return","value":NaN}', b'{"kind":"return","value":1.0}',
                    b'{"kind":"return","value":0,"passed":true}',
                    b'{"kind":"return","value":0,"value":1}',
                    b'{"kind":"raises","type":"RuntimeError"}', b" " * 65537):
            with self.subTest(raw=raw[:80]), self.assertRaises(ValueError):
                self.module.parse_outcome(raw)
        self.assertNotEqual(encoded(self.module.parse_outcome(
            b'{"kind":"return","value":true}')), encoded(self.module.parse_outcome(
                b'{"kind":"return","value":1}')))
        self.rows[0]["payload"] = '{"passed":true}'
        report = self.report(self.rows)
        self.assertEqual(report["rows"][0]["status"], "protocol_error")
        self.assertIsNone(report["rows"][0]["expected_match"])
```

- [x] **2. Run RED with the explicit missing-report assertions.**

```sh
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 \
  -m unittest discover -s experiments/lora_necessity -p 'test_validation_report.py' -v
```

- [x] **3. Add this complete minimal implementation.**

```python
from pilot_protocol import FLAGS, fields, require
from validation_contract import build_schedule, digest, encoded, parse_json


STATUSES = (
    "completed", "program_error", "timeout", "output_limit",
    "resource_limit", "protocol_error", "launch_failed",
)


def _value(value, depth=0):
    require(depth <= 16, "outcome nesting")
    if value is None or type(value) in (bool, int):
        return
    if type(value) is str:
        require(value.isascii(), "outcome ASCII string")
        return
    if type(value) is list:
        require(len(value) <= 4096, "outcome list size")
        for item in value:
            _value(item, depth + 1)
        return
    if type(value) is dict:
        require(len(value) <= 64 and all(type(name) is str and name.isascii()
                                       for name in value), "outcome keys")
        for item in value.values():
            _value(item, depth + 1)
        return
    raise ValueError("unsupported outcome type")


def parse_outcome(raw):
    result = parse_json(raw, limit=65536)
    require(type(result) is dict, "outcome object")
    if result.get("kind") == "return":
        fields(result, ("kind", "value"))
        _value(result["value"])
    else:
        fields(result, ("kind", "type"))
        require(result["kind"] == "raises" and result["type"] == "ValueError",
                "exception outcome")
    return result


def simulate_validation(source_raw, manifest_raw, observations_raw):
    plan = build_schedule(source_raw, manifest_raw)
    observations = parse_json(observations_raw, limit=8388608)
    require(type(observations) is list and len(observations) <= 1416, "observation count")
    indexed = {}
    previous = -1
    stop_ordinal = None
    for observation in observations:
        fields(observation, ("kind", "ordinal", "request_sha256", "status",
                             "payload", "cleanup_complete"))
        require(stop_ordinal is None, "observation after infrastructure stop")
        require(observation["kind"] == "synthetic", "synthetic observations only")
        ordinal = observation["ordinal"]
        require(type(ordinal) is int and previous < ordinal < 1416, "observation order")
        previous = ordinal
        planned = plan["rows"][ordinal]
        require(observation["request_sha256"] == planned["request_sha256"],
                "request binding")
        require(type(observation["cleanup_complete"]) is bool, "cleanup type")
        status = observation["status"]
        require(type(status) is str and status in STATUSES, "observation status")
        outcome = None
        if status == "completed":
            require(type(observation["payload"]) is str, "completed payload")
            try:
                outcome = parse_outcome(observation["payload"].encode("utf-8"))
            except (ValueError, UnicodeError):
                status = "protocol_error"
        else:
            require(observation["payload"] is None, "unexpected payload")
        if observation["status"] == "launch_failed" or not observation["cleanup_complete"]:
            stop_ordinal = ordinal
        indexed[ordinal] = {
            "status": status, "outcome": outcome,
            "cleanup_complete": observation["cleanup_complete"],
            "payload_sha256": digest(observation["payload"]),
        }
    rows = []
    for planned in plan["rows"]:
        observed = indexed.get(planned["ordinal"])
        outcome = None if observed is None else observed["outcome"]
        prediction = planned["witness_prediction"]
        rows.append({
            **{name: planned[name] for name in
               ("ordinal", "task_id", "role", "group", "case_id", "request_sha256")},
            "status": "missing" if observed is None else observed["status"],
            "outcome": outcome,
            "cleanup_complete": None if observed is None else observed["cleanup_complete"],
            "payload_sha256": None if observed is None else observed["payload_sha256"],
            "expected_match": None if outcome is None else
            encoded(outcome) == encoded(planned["expected"]),
            "prediction_match": None if outcome is None or prediction is None else
            encoded(outcome) == encoded(prediction),
        })
    tasks = []
    for task_id in sorted({row["task_id"] for row in rows}):
        selected = [row for row in rows if row["task_id"] == task_id]
        references = [row for row in selected if row["role"] == "reference"]
        initial = [row for row in selected if row["role"] == "initial"]
        witnesses = [row for row in selected if
                     plan["rows"][row["ordinal"]]["witness_prediction"] is not None]
        tasks.append({
            "task_id": task_id,
            "reference_ok": all(row["expected_match"] is True for row in references),
            "initial_scoring_rejected": any(row["group"] == "scoring"
                                           and row["expected_match"] is False
                                           for row in initial),
            "witnesses_ok": len(witnesses) == 3 and all(
                row["prediction_match"] is True and row["expected_match"] is False
                for row in witnesses),
        })
    complete = len(indexed) == 1416 and all(
        row["status"] == "completed" and row["cleanup_complete"] is True for row in rows)
    checks = all(task["reference_ok"] and task["initial_scoring_rejected"]
                 and task["witnesses_ok"] for task in tasks)
    return {
        "kind": "LORA_VALIDATION_SIMULATION", "version": 1,
        "plan_sha256": digest(plan), "observations_sha256": digest(observations),
        "scheduled": 1416, "received": len(indexed), "stop_ordinal": stop_ordinal,
        "simulated_decision": "INCOMPLETE" if not complete else
        ("CHECKS_SATISFIED" if checks else "CHECKS_FAILED"),
        "tasks": tasks, "rows": rows, **{flag: False for flag in FLAGS},
    }
```

- [x] **4. Run GREEN, then all local pilot tests.**

```sh
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 \
  -m unittest discover -s experiments/lora_necessity -p 'test_validation_report.py' -v
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 \
  -m unittest discover -s experiments/lora_necessity -p 'test_*.py' -q
```

Expected: six new report tests pass; actual all-suite count depends on whether the separate
prompt plan has run. Report the observed count, never substitute a planned count.

- [x] **5. Independent review and closeout.**

Review complete new files, not empty tracked diffs. Verify mismatches from expected
`ValueError` versus normal returns are semantic results, but timeout/invalid output are not.
Check every reference and all three witnesses per task, not only aggregate counts.
Verify no implicit I/O or execution callback exists. Run approved tests only; do not call
candidate source strings. Correct review findings with an explicit RED/GREEN test.

## Documentation And Handoff

- [x] After both tasks are reviewed, main appends this exact paragraph to
  `experiments/lora_necessity/README.md` using `apply_patch`, only if it remains accurate:

```markdown
## 动态核验控制逻辑

`validation_contract.py` 绑定当前冻结题库，形成 1416 项源码×用例清单，
并分离工作请求与控制端答案。`validation_report.py` 只处理明确的模拟观察。
控制逻辑测试不执行题目；模拟报告不能证明参考实现正确、隔离有效或模型获益。
真实执行 worker、隔离后端及其启动预算尚未交付。
```

- [x] Record source/test hashes, RED/GREEN logs and independent review in a fresh,
  exclusively created local evidence directory; do not overwrite old roots or helper defaults.
- [x] Recheck frozen data hashes, existing protocol/test hashes and clean tracked/index diff.
  New files are unpublished additions. No automatic commit or push.
- [x] Continue with `2026-11-11-lora-prompt-token-budget.md` after Task 1's interface is stable.

## Plan Self-Review

All spec controller requirements map to Task 1 (identity and projections) or Task 2
(typed results, witnesses, ordering, missing denominator and simulation boundaries).
Real execution, resource enforcement and receipts remain explicitly outside this plan.
Reference jobs are private audit inputs, not a route for passing answers into model prompts.
The source pins are intentional scope, not an extensible ingestion interface.
