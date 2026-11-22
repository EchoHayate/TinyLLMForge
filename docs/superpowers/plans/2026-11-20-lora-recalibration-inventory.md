# LoRA Recalibration Synthetic Inventory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construct and revalidate the exact 120-row local synthetic length inventory from pinned data and reconstructed raw receipts.

**Architecture:** Public rendering is the completed dependency; this module owns ordering, private reference answers, feedback extraction and source/evidence digests. Only an explicitly named synthetic builder is implemented. Real collection provenance and executable-tokenizer admission are separate future work, not a caller-selectable boolean here.

**Tech Stack:** Python standard library, existing pure renderer/data/receipt components, unittest.

## Global Constraints

- Parent specification: `docs/superpowers/specs/2026-11-19-lora-recalibration-validation-adapter-design.md`, section 5 and relevant section 7 cases.
- Existing checkout, all old files/data preserved; no network/SSH, stored-source execution, tokenizer/model load, generation/training, install, telemetry, branch/worktree, commit/push or cleanup.
- Renderer source pin must match the independently approved actual source before task dispatch.
- Only two new Python files; controller owns plans, ledger and results. Apply_patch edits, no inline code comments, no single-letter variable names.

### Task 1: Synthetic Inventory And Reconstruction Validator

**Binding constraints:** Work in `/Users/bytedance/dev/TinyLLMForge`, preserve all
existing files and frozen flags. Local JSON/AST and synthetic receipt tests only.
No network/SSH, telemetry/bootstrap/flush, installs, task-source execution,
tokenizer/model/remote operations, branches/worktrees, commits/push or cleanup.
Only two named new files and private evidence. No subagents or old-session recovery.

**Files:**
- Create `experiments/lora_necessity/recalibration_tokenizer_inventory.py`.
- Create `experiments/lora_necessity/test_recalibration_tokenizer_inventory.py`.

**Interfaces:**
- `build_synthetic_inventory(raw_files, worker_raw, run_ids, records, rendering_raw, *, expected_runtime) -> dict`.
- `validate_synthetic_inventory(inventory, raw_files, worker_raw, run_ids, records, rendering_raw, *, expected_runtime) -> None`.
- `raw_files`, worker/run IDs/records/runtime use the completed static data and
  receipt interfaces; caller-supplied prebuilt PASS/report is never accepted.
- `rendering_raw` has exact builtin-string keys `recalibration_rendering.py` and
  `prompt_rendering.py`, values bytes checked against the fixed implementation pins
  in the candidate below. These bytes are hashed, never imported/executed.
- Uses `first_round(task, arm, supervision_raw)` and
  `second_round(task, arm, supervision_raw, branch, feedback=None)` from the
  completed renderer. Public helpers alone do not authenticate feedback.
- Rebuild the full report using `recalibration_validation.build_report`.
  Only complete report AND satisfied data checks permit any constructed row.
  Otherwise return `status=DEPENDENCY_UNSATISFIED`, empty rows and zero constructed
  counts, retaining report/evidence identity and `planned=120`.
- Complete synthetic records yield `status=SYNTHETIC_ONLY` and 120 rows.
  `synthetic_only=True`, `evidence_kind=synthetic`, `run_authorized=False`,
  `tokenizer_dispatch_authorized=False`, `execution_provenance_verified=False`,
  `real_dynamic_validation_complete=False`, `tokenizer_length_validated=False`,
  `all_possible_second_rounds_validated=False`, and all FLAGS false, unconditionally.
- No option to upgrade provenance, start a tokenizer, supply token counts, or
  claim live readiness. This subplan is not the real executable inventory path.
- Kind `LORA_RECALIBRATION_TOKENIZER_INVENTORY`, version=1,
  data_version=`recalibration_v2_r1`. Full old 324-row inventory or old report rejected.
- Orders: categories first/references/sampled_second, counts 24/48/48;
  C01–C12 each category; odd task base/text, even text/base.
  Each references arm reference_a then reference_b. For task zero-based index
  mod4 in 0/1, base feedback/retry and text retry/feedback; mod4 in 2/3 reverses.
- Row identity digest over `[ordinal,category,task_id,arm,reference_id,branch]`
  for all 120 rows must equal independently derived
  `18dffaeabdda4ffa3633d9c3f2e1cb1026c3dd7e2d82cd2de929f9e8ccc769df`.
- Each row binds messages canonical SHA, optional answer UTF-8 SHA, initial_source
  UTF-8 SHA, examples SHA for text only, public feedback SHA for feedback only,
  and canonical row SHA. Entire inventory binds frozen files, examples, renderer
  bytes, report/plan/raw-evidence/runtime identities and canonical inventory SHA.
- Reference source is an independent `answer` string only on reference rows.
  No EOS bytes appended here: later trusted tokenizer counts unique EOS unchanged.
- Visible feedback uses only completed initial P01/P02 report rows, with strict
  expected_match boolean mapped to pass/mismatch. No outcome values, predictions
  or hidden cases enter that projection. Retry gets no feedback object.
- Limits recorded unchanged: input 4096, answer including EOS 256,
  enable_thinking=False. Recording these limits is not measuring compliance.
- Validation recomputes the inventory from original raw inputs and compares
  canonical bytes after strict builtin JSON-type validation; hashes alone cannot
  conceal tampering. No dependency mocks or permanent trust bypasses.

- [x] **Step 1: Add real synthetic tests before the production file.**

```python
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest

from pilot_protocol import FLAGS
from recalibration_data_contract import FILE_PINS
from recalibration_validation import build_batches, build_report, build_schedule
from validation_contract import digest, encoded, parse_json

if importlib.util.find_spec("recalibration_tokenizer_inventory") is not None:
    import recalibration_tokenizer_inventory as inventory_module
else:
    inventory_module = None


ROOT = Path(__file__).resolve().parent


def process(raw):
    return {"status": "completed", "returncode": 0, "stdout_hex": raw.hex(),
            "stderr_hex": "", "elapsed_ns": 1, "process_reaped": True,
            "stdin_complete": True}


class RecalibrationInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = {name: (ROOT / "data" / (
            name if name == "supervision_v1.json" else "recalibration_v2_r1/" + name
        )).read_bytes() for name in FILE_PINS}
        cls.worker = (ROOT / "validation_task_worker.py").read_bytes()
        cls.renderer = {name: (ROOT / name).read_bytes()
                        for name in ("recalibration_rendering.py", "prompt_rendering.py")}
        cls.run_ids = ["%032x" % position for position in range(15)]
        cls.runtime = {"host": "synthetic-inventory-only", "uid": 123}
        cls.plan = build_schedule(cls.raw)
        cls.batches = build_batches(cls.raw, cls.worker, cls.run_ids)
        cls.records = []
        for batch in cls.batches:
            events = [{
                "kind": "ready", "run_id": batch["run_id"],
                "bundle_sha256": batch["bundle_sha256"], "worker_sha256": batch["worker_sha256"],
                "runtime": cls.runtime,
            }]
            for local, row in enumerate(cls.plan["rows"][batch["start"]:batch["stop"]]):
                events.append({
                    "kind": "row", "run_id": batch["run_id"], "ordinal": local,
                    "request_sha256": row["request_sha256"],
                    "result": process(encoded(row["witness_prediction"] or row["expected"])),
                })
            events.append({"kind": "done", "run_id": batch["run_id"], "received": 32,
                           "elapsed_ns": 1, "stopped": False})
            cls.records.append({
                "batch_index": batch["batch_index"], "bundle_raw": batch["bundle_raw"],
                "transport": process(b"".join(encoded(event) + b"\n" for event in events)),
            })

    def setUp(self):
        self.assertIsNotNone(inventory_module, "recalibration inventory not implemented")

    def build(self, records=None, renderer=None):
        return inventory_module.build_synthetic_inventory(
            self.raw, self.worker, self.run_ids,
            self.records if records is None else records,
            self.renderer if renderer is None else renderer, expected_runtime=self.runtime)

    def validate(self, inventory):
        return inventory_module.validate_synthetic_inventory(
            inventory, self.raw, self.worker, self.run_ids, self.records,
            self.renderer, expected_runtime=self.runtime)

    def changed_outcome(self, ordinal, outcome):
        records = copy.deepcopy(self.records)
        batch_index, local = divmod(ordinal, 32)
        events = [parse_json(line) for line in bytes.fromhex(
            records[batch_index]["transport"]["stdout_hex"]).splitlines()]
        events[local + 1]["result"] = process(encoded(outcome))
        records[batch_index]["transport"] = process(
            b"".join(encoded(event) + b"\n" for event in events))
        return records

    def test_exact_order_counts_and_determinism(self):
        actual = self.build()
        self.assertEqual(actual, self.build())
        self.assertEqual(actual["status"], "SYNTHETIC_ONLY")
        self.assertEqual(actual["kind"], "LORA_RECALIBRATION_TOKENIZER_INVENTORY")
        self.assertEqual(actual["constructed_counts"], {"first": 24, "references": 48,
                                                       "sampled_second": 48})
        self.assertEqual(len(actual["rows"]), 120)
        identities = [[row[name] for name in (
            "ordinal", "category", "task_id", "arm", "reference_id", "branch")]
                      for row in actual["rows"]]
        self.assertEqual(digest(identities),
                         "18dffaeabdda4ffa3633d9c3f2e1cb1026c3dd7e2d82cd2de929f9e8ccc769df")
        self.assertEqual(actual["limits"],
                         {"input_tokens": 4096, "answer_tokens_including_eos": 256,
                          "enable_thinking": False})

    def test_reference_answer_and_public_information_boundaries(self):
        actual = self.build()
        tasks = {task["id"]: task for task in parse_json(self.raw["prompts.json"])}
        oracles = {task["id"]: task for task in parse_json(self.raw["oracles.json"])}
        for row in actual["rows"]:
            first = row["messages"][0]["content"]
            public = parse_json(first.split("\n\nCurrent task:\n")[1].encode("ascii"))
            self.assertEqual(public, tasks[row["task_id"]])
            if row["category"] == "references":
                expected = next(reference["source"] for reference in
                                oracles[row["task_id"]]["references"]
                                if reference["id"] == row["reference_id"])
                self.assertEqual(row["answer"], expected)
                self.assertEqual(len(row["messages"]), 1)
            else:
                self.assertIsNone(row["answer"])
            if row["category"] == "sampled_second":
                self.assertEqual(row["messages"][1]["content"],
                                 tasks[row["task_id"]]["initial_source"])
                payload = parse_json(row["messages"][-1]["content"].split("\n", 1)[1].encode("ascii"))
                if row["branch"] == "retry":
                    self.assertEqual(payload, {"kind": "results_withheld"})
                    self.assertIsNone(row["public_feedback_sha256"])
                else:
                    self.assertEqual(set(payload), {"kind", "cases"})
                    self.assertEqual([case["id"] for case in payload["cases"]], ["P01", "P02"])
                    self.assertTrue(all(set(case) == {"id", "status"} for case in payload["cases"]))
                    self.assertEqual(row["public_feedback_sha256"], digest(payload))

    def test_feedback_is_rebuilt_from_observed_public_rows(self):
        original = self.build()
        target = next(row for row in self.plan["rows"]
                      if (row["task_id"], row["role"], row["case_id"]) == ("C01", "initial", "P02"))
        self.assertIsNone(target["witness_prediction"])
        changed = self.build(self.changed_outcome(target["ordinal"],
                                                 {"kind": "return", "value": "wrong"}))
        self.assertEqual(changed["status"], "SYNTHETIC_ONLY")
        before = [row for row in original["rows"]
                  if row["task_id"] == "C01" and row["category"] == "sampled_second"]
        after = [row for row in changed["rows"]
                 if row["task_id"] == "C01" and row["category"] == "sampled_second"]
        for previous, current in zip(before, after):
            if current["branch"] == "feedback":
                self.assertNotEqual(previous["messages"], current["messages"])
                self.assertIn('"id":"P02","status":"mismatch"', current["messages"][-1]["content"])
            else:
                self.assertEqual(previous["messages"], current["messages"])
        self.assertNotEqual(original["dynamic_evidence_sha256"], changed["dynamic_evidence_sha256"])

    def test_incomplete_and_failed_data_dependency_construct_no_rows(self):
        for records in ([], self.records[:1], self.changed_outcome(
                0, {"kind": "return", "value": "wrong"})):
            with self.subTest(batches=len(records)):
                actual = self.build(records)
                self.assertEqual(actual["status"], "DEPENDENCY_UNSATISFIED")
                self.assertEqual(actual["planned"], 120)
                self.assertEqual(actual["rows"], [])
                self.assertEqual(sum(actual["constructed_counts"].values()), 0)
                self.assertFalse(actual["synthetic_data_checks_satisfied"])

    def test_identity_hashes_and_input_detachment(self):
        actual = self.build()
        report = build_report(self.raw, self.worker, self.run_ids, self.records,
                              expected_runtime=self.runtime)
        self.assertEqual(actual["validation_report_sha256"], digest(report))
        self.assertEqual(actual["dynamic_evidence_sha256"], report["evidence_sha256"])
        self.assertEqual(actual["rendering_sha256"], {
            name: hashlib.sha256(raw).hexdigest() for name, raw in self.renderer.items()})
        for row in actual["rows"]:
            self.assertEqual(row["row_sha256"], digest({
                name: value for name, value in row.items() if name != "row_sha256"}))
            self.assertEqual(row["messages_sha256"], digest(row["messages"]))
            self.assertEqual(row["answer_sha256"], None if row["answer"] is None
                             else hashlib.sha256(row["answer"].encode("utf-8")).hexdigest())
        self.assertEqual(actual["inventory_sha256"], digest({
            name: value for name, value in actual.items() if name != "inventory_sha256"}))
        original = copy.deepcopy(actual["rows"][0]["messages"])
        actual["rows"][24]["messages"][0]["content"] = "changed"
        self.assertEqual(actual["rows"][0]["messages"], original)
        self.assertEqual(self.build()["rows"][24]["messages"], original)

    def test_synthetic_flags_and_tampering_rejected_after_rehash(self):
        actual = self.build()
        self.assertTrue(actual["synthetic_only"])
        self.assertEqual(actual["evidence_kind"], "synthetic")
        flags = (*FLAGS, "run_authorized", "tokenizer_dispatch_authorized",
                 "execution_provenance_verified", "real_dynamic_validation_complete",
                 "tokenizer_length_validated", "all_possible_second_rounds_validated")
        for name in flags:
            self.assertIs(actual[name], False)
            changed = copy.deepcopy(actual)
            changed[name] = True
            changed["inventory_sha256"] = digest({
                key: value for key, value in changed.items() if key != "inventory_sha256"})
            with self.subTest(flag=name), self.assertRaises(ValueError):
                self.validate(changed)
        self.assertIsNone(self.validate(actual))

    def test_old_inventories_reordering_and_prebuilt_pass_rejected(self):
        actual = self.build()
        mutations = [
            {**actual, "kind": "LORA_TOKENIZER_INVENTORY"},
            {**actual, "rows": actual["rows"] * 3},
            {**actual, "rows": list(reversed(actual["rows"]))},
            {**actual, "version": True},
            {**actual, "evidence_kind": "real"},
            {**actual, "extra": 1},
        ]
        for changed in mutations:
            with self.subTest(kind=changed["kind"]), self.assertRaises(ValueError):
                self.validate(changed)
        with self.assertRaises(ValueError):
            self.build({"kind": "LORA_RECALIBRATION_VALIDATION_REPORT",
                        "matrix_complete": True, "data_checks_satisfied": True})

    def test_renderer_bytes_and_strict_local_types(self):
        changed = dict(self.renderer)
        changed["recalibration_rendering.py"] += b"\n"
        with self.assertRaises(ValueError):
            self.build(renderer=changed)
        class StringSubclass(str):
            pass
        with self.assertRaises(ValueError):
            self.build(renderer={StringSubclass(name): value
                                 for name, value in self.renderer.items()})
        changed = self.build()
        changed["limits"]["input_tokens"] = float("nan")
        with self.assertRaises(ValueError):
            self.validate(changed)

    def test_no_execution_or_tokenizer_imports(self):
        tree = ast.parse((ROOT / "recalibration_tokenizer_inventory.py").read_text())
        allowed = {"copy", "hashlib", "pilot_protocol", "recalibration_data_contract",
                   "recalibration_rendering", "recalibration_validation", "validation_contract"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name in allowed for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"exec", "eval", "compile", "open", "__import__"})


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Capture intended missing-implementation RED.**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_tokenizer_inventory -v
```

Expected 9 assertion failures `recalibration inventory not implemented`, not import
errors. Preserve raw logs and immediate exits in the task evidence root.

- [x] **Step 3: Implement the candidate; fix demonstrated defects test-first.**

```python
import copy
import hashlib

from pilot_protocol import FLAGS, fields, require
from recalibration_data_contract import FILE_PINS, _plain, load_dataset
from recalibration_rendering import EXAMPLES_SHA256, first_round, second_round
from recalibration_validation import build_report
from validation_contract import digest, encoded


RENDERING_PINS = {
    "recalibration_rendering.py": "ac8286fe3ceb567e23d3a04b32e69d5447d443000c707b3af334cdec94b4baf0",
    "prompt_rendering.py": "2016c3eb0255d1d4a40f6072c7407b099fa3e4e569345c8d1ac778d5bd7bb18f",
}
PLANNED_COUNTS = {"first": 24, "references": 48, "sampled_second": 48}


def _source_identity(rendering_raw):
    fields(rendering_raw, RENDERING_PINS)
    require(all(type(name) is str for name in rendering_raw), "rendering key type")
    for name, expected in RENDERING_PINS.items():
        raw = rendering_raw[name]
        require(type(raw) is bytes and len(raw) <= 1048576
                and hashlib.sha256(raw).hexdigest() == expected, "renderer identity")
    return dict(RENDERING_PINS)


def _text_sha(value):
    return None if value is None else hashlib.sha256(value.encode("utf-8")).hexdigest()


def _visible_feedback(report, task_id):
    rows = [row for row in report["rows"] if row["task_id"] == task_id
            and row["role"] == "initial" and row["case_id"] in ("P01", "P02")]
    require([row["case_id"] for row in rows] == ["P01", "P02"], "public observation identity")
    require(all(row["status"] == "completed" and type(row["expected_match"]) is bool
                for row in rows), "completed public observations")
    return {"kind": "visible_tests", "cases": [
        {"id": row["case_id"], "status": "pass" if row["expected_match"] else "mismatch"}
        for row in rows]}


def _append(rows, category, task, arm, messages, answer=None,
            reference_id=None, branch=None, feedback=None):
    row = {
        "ordinal": len(rows), "category": category, "task_id": task["id"], "arm": arm,
        "reference_id": reference_id, "branch": branch,
        "messages": copy.deepcopy(messages), "messages_sha256": digest(messages),
        "answer": answer, "answer_sha256": _text_sha(answer),
        "initial_source_sha256": _text_sha(task["initial_source"]),
        "examples_sha256": EXAMPLES_SHA256 if arm == "text" else None,
        "public_feedback_sha256": None if feedback is None else digest(feedback),
    }
    rows.append({**row, "row_sha256": digest(row)})


def build_synthetic_inventory(raw_files, worker_raw, run_ids, records, rendering_raw, *,
                              expected_runtime):
    rendering_sha256 = _source_identity(rendering_raw)
    dataset = load_dataset(raw_files)
    report = build_report(raw_files, worker_raw, run_ids, records, expected_runtime=expected_runtime)
    satisfied = report["matrix_complete"] is True and report["data_checks_satisfied"] is True
    rows = []
    if satisfied:
        for category in PLANNED_COUNTS:
            for task_index, (task, oracle) in enumerate(zip(dataset["prompts"], dataset["oracles"])):
                arms = ("base", "text") if task_index % 2 == 0 else ("text", "base")
                for arm in arms:
                    first = first_round(task, arm, raw_files["supervision_v1.json"])
                    if category == "first":
                        _append(rows, category, task, arm, first)
                    elif category == "references":
                        for reference in oracle["references"]:
                            _append(rows, category, task, arm, first,
                                    answer=reference["source"], reference_id=reference["id"])
                    else:
                        feedback_first = (task_index % 4 < 2) == (arm == "base")
                        branches = ("feedback", "retry") if feedback_first else ("retry", "feedback")
                        for branch in branches:
                            feedback = _visible_feedback(report, task["id"]) if branch == "feedback" else None
                            messages = second_round(task, arm, raw_files["supervision_v1.json"],
                                                    branch, feedback)
                            _append(rows, category, task, arm, messages,
                                    branch=branch, feedback=feedback)
    counts = {category: sum(row["category"] == category for row in rows)
              for category in PLANNED_COUNTS}
    require(counts == (PLANNED_COUNTS if satisfied else {
        category: 0 for category in PLANNED_COUNTS}), "inventory row counts")
    inventory = {
        "kind": "LORA_RECALIBRATION_TOKENIZER_INVENTORY", "version": 1,
        "data_version": "recalibration_v2_r1", "file_sha256": dict(FILE_PINS),
        "status": "SYNTHETIC_ONLY" if satisfied else "DEPENDENCY_UNSATISFIED",
        "synthetic_only": True, "evidence_kind": "synthetic",
        "planned": 120, "planned_counts": dict(PLANNED_COUNTS),
        "constructed_counts": counts, "rows": rows,
        "examples_sha256": EXAMPLES_SHA256, "rendering_sha256": rendering_sha256,
        "validation_report_sha256": digest(report),
        "validation_plan_sha256": report["plan_sha256"],
        "dynamic_evidence_sha256": report["evidence_sha256"],
        "expected_runtime_sha256": report["expected_runtime_sha256"],
        "worker_sha256": report["worker_sha256"], "run_ids": list(run_ids),
        "synthetic_data_checks_satisfied": satisfied,
        "dependency_first_stop": copy.deepcopy(report["first_stop"]),
        "limits": {"input_tokens": 4096, "answer_tokens_including_eos": 256,
                   "enable_thinking": False},
        "run_authorized": False, "tokenizer_dispatch_authorized": False,
        "execution_provenance_verified": False, "real_dynamic_validation_complete": False,
        "tokenizer_length_validated": False, "all_possible_second_rounds_validated": False,
        **{flag: False for flag in FLAGS},
    }
    return {**inventory, "inventory_sha256": digest(inventory)}


def validate_synthetic_inventory(inventory, raw_files, worker_raw, run_ids,
                                 records, rendering_raw, *, expected_runtime):
    _plain(inventory)
    expected = build_synthetic_inventory(raw_files, worker_raw, run_ids, records,
                                         rendering_raw, expected_runtime=expected_runtime)
    require(encoded(inventory) == encoded(expected), "synthetic inventory reconstruction differs")
```

- [x] **Step 4: Focused GREEN and integration regression with retained raw evidence.**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_tokenizer_inventory -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_prompt_rendering test_tokenizer_inventory test_recalibration_data_contract test_recalibration_validation test_recalibration_rendering test_recalibration_tokenizer_inventory -v
```

Expect no failures/skips. No tokenizer is imported; the old inventory tests are
pure construction checks. New helper tests may be added for concrete uncovered
requirements, with actual failure evidence before implementation repair.

- [x] **Step 5: Independent review, preservation and final evidence; no commit.**

Fresh reviewer receives this brief, actual two-file addition diff and raw test
report, checking spec and quality. Controller checks renderer/source hashes and
all original 173 files, independently matches all 120 identity tuples and frozen
example bytes, and closes the rendering+synthetic inventory subproject.
No entire-branch/whole-adapter readiness is claimed. The later tokenizer
report/coordinator and real provenance/launch admission remain unimplemented.

## Self-Review

The public helper is explicitly not evidence admission. The builder reconstructs
raw receipts before producing even first/reference rows, and exposes only a
synthetic path. Complete-order digest derives from a separate explicit four-task
branch table. Mutation tests must preserve the distinction between legitimate
non-witness semantic mismatch and reference/witness data-check failure.
The fixed 4096/256 settings are metadata, not asserted token measurements.
