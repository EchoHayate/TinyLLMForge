# LoRA Necessity CPU Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement only the approved pilot's CPU metadata contract and synthetic paired-decision arithmetic, without loading a model or executing candidate code.

**Architecture:** A standalone standard-library module freezes configuration and validates task metadata, then scores explicitly synthetic final-outcome records. Its result always says `SYNTHETIC_ONLY`; simulated screening never becomes empirical acceptance.

**Tech Stack:** Python 3.12 standard library, colocated `unittest`.

## Global Constraints

- Repository: `/Users/bytedance/dev/TinyLLMForge`, existing `experiments/feedback-active-verification` branch; no checkout, worktree creation, commit, push, or dependency installation.
- New code only under `experiments/lora_necessity/`; never change previous experiments or `tinyvllm`.
- Main owns this plan, the approved design, the stage ledger and README. The implementer owns exactly `pilot_protocol.py` and `test_pilot_protocol.py`.
- No model/tokenizer load, GPU/remote operation, candidate execution, network calls, telemetry, or recovered-session execution.
- Use `apply_patch`; no inline code comments or one-letter identifiers.
- Design: `docs/superpowers/specs/2026-11-10-lora-necessity-pilot-design.md`.
- 8 supervision items in four families, 4 development items, 16 transfer items and 16 retention items.
- Four arms: `base`, `text`, `lora`, `lora_text`; development only uses base/text.
- Two model calls per completed episode; maximum 272 calls; 16 optimizer steps; 3600-second workflow cap; 4096 input/256 output tokens per call.
- Screening: lora_text minus text transfer passes at least 2; zero text-correct/lora_text-wrong retention cases.
- Keep `benchmark_eligible`, `formal_evaluation_ready`, `net_benefit_established`, `isolated` false.
- Missing rows stay in the denominator. Missing costs remain unknown. Invalid schema raises `ValueError`, never gets silently counted as an ordinary model miss.
- New files are untracked additions; review their full content, not only `git diff HEAD`.

## Scope Decomposition

This is stage one, not implementation of the entire pilot. Other independently reviewable deliverables remain: real data and oracle audit, training worker, isolated candidate evaluator, runtime receipts and launch admission.

The CPU stage contains no actual examples, model prompt builder, loss implementation or model execution. A task digest and a unique template label verify metadata consistency only, not semantic novelty. Synthetic outcomes and booleans are never trusted live-runtime evidence.

Later live-result integration must introduce separately reviewed binding to actual prompts, model/adapter identities, raw generations, tools, optimizer trace and cleanup. Do not extend this synthetic input schema to accept real receipts.

## Task 1: Fixed Contract and Synthetic Paired Scoring

**Files:**
- Create: `experiments/lora_necessity/pilot_protocol.py`
- Test: `experiments/lora_necessity/test_pilot_protocol.py`

**Interfaces:**
- `build_manifest(tasks: list[dict]) -> dict`: canonical fixed-config metadata manifest; deep-copy input.
- `validate_manifest(manifest: dict) -> str`: strict validation; return SHA256 of canonical JSON.
- `score_synthetic(manifest: dict, records: list[dict], checks: dict) -> dict`: synthetic-only report.
- `configuration() -> dict`: return independent fixed-configuration dictionaries.
- No file I/O or executable callback in these APIs.

### Step 1: Write Test File Before Production File

- [x] Create the following initial tests with `apply_patch`. The missing module is detected by an explicit assertion so RED is an expected test failure, not an accidental import error.

```python
import copy
import hashlib
import importlib
from pathlib import Path
import unittest


def task_fixture():
    tasks = []
    for split, count in (("supervision", 8), ("development", 4),
                         ("transfer", 16), ("retention", 16)):
        for index in range(count):
            task_id = f"{split}-{index:02d}"
            tasks.append({
                "id": task_id, "split": split,
                "family": f"{'retain' if split == 'retention' else 'adapt'}-{index % 4}",
                "template_id": f"template-{task_id}",
                "content_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
            })
    return tasks


def record_fixture(manifest):
    records = []
    for task in manifest["tasks"]:
        if task["split"] == "supervision":
            continue
        arms = ("base", "text") if task["split"] == "development" else (
            "base", "text", "lora", "lora_text")
        for arm in arms:
            ordinal = int(task["id"].rsplit("-", 1)[1])
            passed = ordinal % 2 == 0
            if task["split"] == "transfer" and arm == "lora_text" and ordinal in (1, 3):
                passed = True
            records.append({
                "kind": "synthetic", "task_id": task["id"], "arm": arm,
                "status": "pass" if passed else "fail",
                "model_calls": 2, "input_tokens": 20, "output_tokens": 10,
                "episode_ns": 100,
            })
    return records


class PilotProtocolTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("pilot_protocol.py").is_file(),
                        "pilot protocol is not implemented")
        self.protocol = importlib.import_module("pilot_protocol")
        self.manifest = self.protocol.build_manifest(task_fixture())
        self.records = record_fixture(self.manifest)
        self.checks = {
            "protocol_valid": True, "parameters_valid": True,
            "isolation_valid": True, "cleanup_complete": True,
        }

    def test_configuration_copies_and_binding(self):
        original = copy.deepcopy(self.manifest)
        config = self.protocol.configuration()
        self.assertEqual(config["training"]["steps"], 16)
        config["training"]["steps"] = 1
        self.assertEqual(self.protocol.configuration()["training"]["steps"], 16)
        self.assertEqual(self.manifest, original)
        changed = copy.deepcopy(self.manifest)
        changed["config"]["training"]["steps"] = True
        with self.assertRaises(ValueError):
            self.protocol.validate_manifest(changed)

    def test_synthetic_success_cannot_admit_real_experiment(self):
        original = copy.deepcopy(self.records)
        report = self.protocol.score_synthetic(self.manifest, self.records, self.checks)
        self.assertEqual(report["status"], "SYNTHETIC_ONLY")
        self.assertEqual(report["simulated_decision"], "PROMISING_FOR_REPLICATION")
        self.assertEqual(report["pairs"]["transfer"]["net"], 2)
        self.assertEqual(report["pairs"]["retention"]["right_to_wrong"], 0)
        self.assertEqual(report["model_calls"], 272)
        for flag in ("benchmark_eligible", "formal_evaluation_ready",
                     "net_benefit_established", "isolated"):
            self.assertIs(report[flag], False)
        self.assertEqual(self.records, original)

    def test_missing_and_unknown_cost_do_not_pass(self):
        report = self.protocol.score_synthetic(self.manifest, [], self.checks)
        self.assertEqual(report["simulated_decision"], "INCONCLUSIVE_INCOMPLETE")
        self.assertEqual(report["by_split"]["transfer"]["text"]["scheduled"], 16)
        self.assertEqual(report["by_split"]["transfer"]["text"]["passed"], 0)
        self.assertIsNone(report["by_split"]["transfer"]["text"]["episode_ns"])
        changed = copy.deepcopy(self.records)
        changed[-1]["episode_ns"] = None
        report = self.protocol.score_synthetic(self.manifest, changed, self.checks)
        self.assertEqual(report["simulated_decision"], "INCONCLUSIVE_INCOMPLETE")

    def test_retention_regression_blocks_positive_transfer(self):
        changed = copy.deepcopy(self.records)
        for record in changed:
            if record["task_id"] == "retention-00" and record["arm"] == "lora_text":
                record["status"] = "fail"
        report = self.protocol.score_synthetic(self.manifest, changed, self.checks)
        self.assertEqual(report["simulated_decision"], "OBSERVED_REGRESSION")

    def test_duplicate_and_live_records_rejected(self):
        with self.assertRaises(ValueError):
            self.protocol.score_synthetic(self.manifest, self.records + self.records[:1], self.checks)
        changed = copy.deepcopy(self.records)
        changed[0]["kind"] = "live"
        with self.assertRaises(ValueError):
            self.protocol.score_synthetic(self.manifest, changed, self.checks)

    def test_invalid_metadata_and_scalars_rejected(self):
        changed = task_fixture()
        changed[-1]["template_id"] = changed[0]["template_id"]
        with self.assertRaises(ValueError):
            self.protocol.build_manifest(changed)
        for invalid in (True, -1, float("nan"), float("inf"), "10"):
            changed_records = copy.deepcopy(self.records)
            changed_records[0]["episode_ns"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.protocol.score_synthetic(self.manifest, changed_records, self.checks)
```

### Step 2: Observe RED

- [x] Run `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -m unittest discover -s experiments/lora_necessity -p 'test_pilot_protocol.py' -v`.
- [x] Record the expected failures saying `pilot protocol is not implemented`. Do not create production code before this observation.

### Step 3: Minimal Production Implementation

- [x] Implement the following module. Expand only where targeted tests reveal a concrete gap against this contract; preserve API names and scope. Raise questions if a correction changes a research decision.

```python
import copy
import hashlib
import json
import re
from collections import Counter


ARMS = ("base", "text", "lora", "lora_text")
SPLITS = {"supervision": 8, "development": 4, "transfer": 16, "retention": 16}
FLAGS = ("benchmark_eligible", "formal_evaluation_ready", "net_benefit_established", "isolated")
CHECKS = ("protocol_valid", "parameters_valid", "isolation_valid", "cleanup_complete")
COSTS = ("input_tokens", "output_tokens", "episode_ns")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(value, names):
    require(type(value) is dict and set(value) == set(names), "unexpected fields")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def configuration():
    return {
        "arms": list(ARMS), "splits": dict(SPLITS),
        "training": {
            "rank": 8, "alpha": 16, "dropout": 0, "seed": 42,
            "targets": ["q_proj", "v_proj"], "base_dtype": "bfloat16",
            "adapter_dtype": "float32", "optimizer": "AdamW",
            "learning_rate": 0.00005, "weight_decay": 0,
            "gradient_clip": 1, "batch_size": 1, "accumulation": 1,
            "epochs": 2, "steps": 16, "answer_only_loss": True,
        },
        "limits": {
            "input_tokens": 4096, "output_tokens": 256, "episode_calls": 2,
            "total_calls": 272, "workflow_seconds": 3600, "gpu_count": 1,
        },
        "screen": {"transfer_net_min": 2, "retention_regression_max": 0},
    }


def validate_manifest(manifest):
    fields(manifest, ("kind", "version", "config", "tasks"))
    require(manifest["kind"] == "LORA_NECESSITY_CPU_MANIFEST", "manifest kind")
    require(type(manifest["version"]) is int and manifest["version"] == 1, "version")
    require(canonical(manifest["config"]) == canonical(configuration()), "configuration changed")
    tasks = manifest["tasks"]
    require(type(tasks) is list and len(tasks) == 44, "44 task metadata rows required")
    identifiers, templates, digests = set(), set(), set()
    groups = {split: [] for split in SPLITS}
    for task in tasks:
        fields(task, ("id", "split", "family", "template_id", "content_sha256"))
        for name in task:
            require(type(task[name]) is str and re.fullmatch(r"[A-Za-z0-9_-]+", task[name]),
                    "invalid task metadata")
        require(task["split"] in SPLITS, "unknown split")
        require(re.fullmatch(r"[0-9a-f]{64}", task["content_sha256"]), "invalid digest")
        for value, seen in ((task["id"], identifiers), (task["template_id"], templates),
                            (task["content_sha256"], digests)):
            require(value not in seen, "duplicate task, template or content")
            seen.add(value)
        groups[task["split"]].append(task["family"])
    families = {}
    for split, expected in SPLITS.items():
        require(len(groups[split]) == expected, "split count")
        counts = Counter(groups[split])
        require(len(counts) == 4 and set(counts.values()) == {expected // 4}, "family balance")
        families[split] = set(counts)
    require(families["supervision"] == families["transfer"], "transfer families")
    require(families["retention"].isdisjoint(families["supervision"]), "retention overlap")
    return hashlib.sha256(canonical(manifest).encode()).hexdigest()


def build_manifest(tasks):
    manifest = {
        "kind": "LORA_NECESSITY_CPU_MANIFEST", "version": 1,
        "config": configuration(), "tasks": copy.deepcopy(tasks),
    }
    validate_manifest(manifest)
    return manifest


def score_synthetic(manifest, records, checks):
    digest = validate_manifest(manifest)
    fields(checks, CHECKS)
    require(all(value is None or type(value) is bool for value in checks.values()), "checks")
    require(type(records) is list, "record list")
    schedule = {}
    for task in manifest["tasks"]:
        if task["split"] != "supervision":
            arms = ARMS[:2] if task["split"] == "development" else ARMS
            for arm in arms:
                schedule[(task["id"], arm)] = task["split"]
    indexed = {}
    for record in records:
        fields(record, ("kind", "task_id", "arm", "status", "model_calls", *COSTS))
        require(record["kind"] == "synthetic", "synthetic records only")
        require(type(record["task_id"]) is str and type(record["arm"]) is str, "record identity")
        key = record["task_id"], record["arm"]
        require(key in schedule and key not in indexed, "unknown or duplicate episode")
        require(record["status"] in ("pass", "fail", "invalid", "timeout"), "record status")
        require(type(record["model_calls"]) is int and 0 <= record["model_calls"] <= 2,
                "model call count")
        require(record["status"] == "timeout" or record["model_calls"] == 2,
                "completed episode needs two model calls")
        for name in COSTS:
            value = record[name]
            require(value is None or type(value) is int and value >= 0, "invalid cost")
        for name, cap in (("input_tokens", 4096), ("output_tokens", 256)):
            require(record[name] is None or record[name] <= cap * record["model_calls"],
                    "token budget")
        indexed[key] = record
    by_split = {}
    for split in ("development", "transfer", "retention"):
        by_split[split] = {}
        for arm in (ARMS[:2] if split == "development" else ARMS):
            rows = [indexed.get(key) for key, stage in schedule.items()
                    if stage == split and key[1] == arm]
            summary = {
                "scheduled": len(rows), "received": sum(row is not None for row in rows),
                "passed": sum(row is not None and row["status"] == "pass" for row in rows),
                "statuses": dict(Counter("missing" if row is None else row["status"] for row in rows)),
            }
            for name in COSTS:
                values = [row[name] for row in rows if row is not None and row[name] is not None]
                summary[name] = sum(values) if len(values) == len(rows) else None
                summary[f"{name}_observed"] = sum(values)
            by_split[split][arm] = summary
    pairs = {}
    for split in ("transfer", "retention"):
        counts = Counter()
        for task in manifest["tasks"]:
            if task["split"] != split:
                continue
            before = indexed.get((task["id"], "text"))
            after = indexed.get((task["id"], "lora_text"))
            if before is None or after is None:
                counts["missing"] += 1
            else:
                pattern = (before["status"] == "pass", after["status"] == "pass")
                counts[{(True, True): "both_right", (False, False): "both_wrong",
                        (True, False): "right_to_wrong", (False, True): "wrong_to_right"}[pattern]] += 1
        pairs[split] = {name: counts[name] for name in
                       ("both_right", "both_wrong", "right_to_wrong", "wrong_to_right", "missing")}
        pairs[split]["net"] = counts["wrong_to_right"] - counts["right_to_wrong"]
    incomplete = (len(indexed) != len(schedule)
                  or any(row["status"] == "timeout" for row in indexed.values())
                  or any(row[name] is None for row in indexed.values() for name in COSTS)
                  or any(value is None for value in checks.values()))
    development = by_split["development"]
    calibration_bad = (all(development[arm]["passed"] == 4 for arm in ARMS[:2])
                       or all(development[arm]["passed"] == 0 for arm in ARMS[:2])
                       or any(development[arm]["statuses"].get("invalid", 0) for arm in ARMS[:2]))
    if any(value is False for value in checks.values()):
        decision = "INVALID_PROTOCOL"
    elif incomplete:
        decision = "INCONCLUSIVE_INCOMPLETE"
    elif calibration_bad:
        decision = "INCONCLUSIVE_CALIBRATION"
    elif pairs["retention"]["right_to_wrong"]:
        decision = "OBSERVED_REGRESSION"
    elif pairs["transfer"]["net"] >= 2:
        decision = "PROMISING_FOR_REPLICATION"
    else:
        decision = "NO_GAIN_IN_THIS_PILOT"
    return {
        "status": "SYNTHETIC_ONLY", "manifest_sha256": digest,
        "simulated_decision": decision, "by_split": by_split, "pairs": pairs,
        "model_calls": sum(row["model_calls"] for row in indexed.values()),
        "scheduled_model_calls": 272, "checks": copy.deepcopy(checks),
        **{flag: False for flag in FLAGS},
    }
```

### Step 4: GREEN and Boundary Coverage

- [x] Run the focused command from Step 2. Record actual results.
- [x] Add focused RED/GREEN tests for wrong split/family counts, duplicate digest/ID, altered config, unknown supervision/arm records, integer/boolean confusion in model_calls, token cap overflow, timeout versus ordinary invalid final code, no gain, calibration floor/ceiling, explicit failed/unknown checks, order-independent records, manifest/task input immutability and equal missing denominators.
- [x] Extend output with per-family summaries if absent: `by_family[split][family][arm]`, with the same fixed denominators/status/cost fields as `by_split`. Tests must verify four families and no dropped missing rows.
- [x] Return expected schema errors as ValueError rather than incidental TypeError for malformed JSON-like inputs. Test scalar/list/dict substitutions at public boundaries.
- [x] These are testable CPU contracts only. No synthetic training, isolation or cleanup check is evidence of the corresponding real operation.
- [x] The stage-one scorer does not establish complete-run cost: no sum of per-arm episode costs is called whole-workflow time, and shared training cost remains absent rather than fabricated.

### Step 5: Review Handoff

- [x] Run the entire new directory's `test_*.py` suite once.
- [x] Include source/test hashes and RED/GREEN output in the task report.
- [x] Do not commit. Return source ownership to main for independent spec/quality review.

## Main-Owned Documentation and Completion

- [x] Update the design status to record user approval, without converting it to GPU permission.
- [x] Create `experiments/lora_necessity/README.md`: identify synthetic-only APIs, exact test command, missing actual-data/training/evaluator components, immutable old experiments and four false admission flags.
- [x] Keep progress in `.superpowers/sdd/lora-necessity-20261110-progress.md`, not another experiment's ledger.
- [x] Review full new-file diffs, not empty tracked diffs. Independently run the new suite after fixes.
- [x] Preserve all pre-existing tracked and untracked files.
- [x] Stop stage one with an honest local-contract result; do not call CPU success LoRA benefit or complete pilot implementation.

## Stage-One Closeout

Completed locally on 2026-11-10. Implementer RED/GREEN history is retained in
`/tmp/tlf-lora-protocol-20261110-mofnsg/task-1-report.md`; main independently
reran all 26 tests after the documentation correction: PASS, 0.101 seconds.
Independent task-spec, implementation-quality and documentation-consistency
reviews passed; F1 was closed by an append-only follow-up in `final-review.md`
under the same evidence root. Only the regression-label wording changed;
production/tests and numerical thresholds did not.

These checked items complete the CPU stage only. Actual tasks/oracles,
training, isolated evaluation, live evidence binding and bounded runtime
admission remain separate deliverables. No commit, push or real run occurred.
