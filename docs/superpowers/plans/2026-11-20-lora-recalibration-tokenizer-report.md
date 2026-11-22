# LoRA Recalibration Synthetic Tokenizer Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct a 120-row synthetic length report with strict receipt, stop, and finalization checks.

**Architecture:** Rebuild the completed synthetic inventory from original inputs, validate the new rows without the old 324-row validator, and reuse only schema-independent finalization/scalar checks. The report is explicitly synthetic; real collection and dispatch remain separate work.

**Tech Stack:** Existing Python standard library, pure data/report modules and unittest.

## Global Constraints

- Approved parent: `docs/superpowers/specs/2026-11-19-lora-recalibration-validation-adapter-design.md`, sections 6–7.
- Work in the existing checkout; preserve old files and frozen data. No network, SSH, task-source execution, tokenizer/model load, generation, training, installation, telemetry/bootstrap/flush, branch/worktree, commit/push or cleanup.
- Fixed input 4096, answer including unique EOS 256, enable_thinking false.
- Local synthetic receipts only. No real provenance, length or admission pass, no cleanup claim.
- Apply_patch edits, no inline code comments or single-letter variables. No subagents within delegated tasks.
- Parent owns plans, ledger and evidence closeout. Only the two new files named below belong to the implementer.

### Task 1: Pure Synthetic Report

**Files:**
- Create `experiments/lora_necessity/recalibration_tokenizer_report.py`.
- Create `experiments/lora_necessity/test_recalibration_tokenizer_report.py`.

**Interfaces and binding requirements:**

- `prepare_synthetic_inputs(validation_inputs, profile_raw, approved_profile_sha256)` returns `(inventory, profile)`.
- `validation_inputs` has exactly `raw_files`, `worker_raw`, `run_ids`, `records`, `rendering_raw`, `expected_runtime`, the argument names of the completed inventory builder. It never contains a prebuilt PASS or caller-approved inventory.
- `build_synthetic_report(validation_inputs, profile_raw, approved_profile_sha256, records, finalization, *, failure=None, abort_failed=False)` returns `{"public": ..., "private": ...}`.
- `_validate_record` is an internal coordinator helper for a row already reconstructed by `prepare_synthetic_inputs`; it does not authenticate arbitrary caller-selected rows.
- Reconstruct the pinned synthetic inventory first. Reject legacy inventory/report objects as inputs. Dependency failure accepts no tokenizer records, finalization, failure or abort evidence; returns zero constructed public rows, planned/not_checked 120 and exact planned category totals.
- Satisfied dependency yields all 120 public row identities, even when measurements stop early. No deduplication, no old `_row`, `_ORDER`, inventory validator or old report builder.
- Verify run/profile/ordinal/message/answer bindings, builtin types, token lists/counts, 4096/4097 and 256/257 boundaries, input prefix/target/tail partition, identity/state continuity and every chain link. Reject records after the first over-limit row.
- Reuse `tokenizer_audit_report._finalization` only because it is denominator-independent: it validates exact finalization, identity/EOS, runtime and snapshot profiles, final chain/count and stop reason. Existing dependencies remain byte-identical.
- Require complete 120 measurements or a normal over-limit stop before successful finish; reject missing finalization, invalid finalization and premature finish.
- Failure phases OPEN/MEASURE/RECEIPT/FINALIZE, bounded public reason only. OPEN has no records or abort failure; row failures name exactly next unaccepted ordinal; FINALIZE requires complete/normal-stop prefix. `abort_failed` is exact bool. Exceptions must never leak private text.
- Public has only identity/digests, counts/statuses and bounded failures, no messages/answers/token IDs/profile private values. Private holds reconstructed inventory, checked records and finalization; returned values detached.
- `synthetic_only=True`, `evidence_kind=synthetic`. `synthetic_all_rows_within_budget` only true with full valid finalization and no stop/failure. All four FLAGS plus `all_possible_second_rounds_validated`, `tokenizer_length_validated`, `execution_provenance_verified`, `real_dynamic_validation_complete`, `run_authorized`, `tokenizer_dispatch_authorized` stay false.
- Profile validation reuses unchanged pins/version inventory; this is validating supplied metadata, not inspecting/installing runtime. No real tokenizer imports or session operations.

- [x] **Step 1: Write the synthetic fixture and failing tests.**

Use the following actual tests as the starting point. Expand named requirements before GREEN where needed; do not mock validators. Existing helper functions here only construct JSON, never load or execute stored task programs.

```python
import ast
import copy
import importlib.util
from pathlib import Path
import unittest

from pilot_protocol import FLAGS
from test_recalibration_tokenizer_inventory import RecalibrationInventoryTests
from test_tokenizer_audit_report import synthetic_profile, synthetic_transcript, rechain
from validation_contract import digest, encoded

if importlib.util.find_spec("recalibration_tokenizer_report") is not None:
    import recalibration_tokenizer_report as report_module
else:
    report_module = None


def fixture_inputs():
    fixture = RecalibrationInventoryTests
    fixture.setUpClass()
    return {
        "raw_files": fixture.raw, "worker_raw": fixture.worker,
        "run_ids": fixture.run_ids, "records": fixture.records,
        "rendering_raw": fixture.renderer, "expected_runtime": fixture.runtime,
    }


class RecalibrationTokenizerReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from recalibration_tokenizer_inventory import build_synthetic_inventory
        cls.inputs = fixture_inputs()
        cls.inventory = build_synthetic_inventory(**cls.inputs)
        cls.profile = synthetic_profile()
        cls.profile_raw = encoded(cls.profile)
        cls.profile_sha = digest(cls.profile)
        cls.records, cls.finalization = synthetic_transcript(cls.inventory, cls.profile, count=120)

    def setUp(self):
        self.assertIsNotNone(report_module, "recalibration tokenizer report not implemented")

    def report(self, records=None, finalization=None, **options):
        return report_module.build_synthetic_report(
            self.inputs, self.profile_raw, self.profile_sha,
            self.records if records is None else records,
            self.finalization if finalization is None else finalization, **options)

    def test_complete_synthetic_report_and_public_boundary(self):
        actual = self.report()
        public = actual["public"]
        self.assertEqual(public["kind"], "LORA_RECALIBRATION_TOKENIZER_REPORT")
        self.assertEqual(public["status"], "SYNTHETIC_COMPLETE")
        self.assertEqual((public["planned"], public["checked"], public["not_checked"]), (120, 120, 0))
        self.assertTrue(public["synthetic_all_rows_within_budget"])
        self.assertEqual({name: count["scheduled"] for name, count in public["by_category"].items()},
                         {"first": 24, "references": 48, "sampled_second": 48})
        self.assertEqual(len(public["rows"]), 120)
        self.assertNotIn("private synthetic", encoded(public).decode())
        for flag in (*FLAGS, "all_possible_second_rounds_validated", "tokenizer_length_validated",
                     "execution_provenance_verified", "real_dynamic_validation_complete",
                     "run_authorized", "tokenizer_dispatch_authorized"):
            self.assertIs(public[flag], False)
        actual["private"]["records"][0]["measurement"]["input"]["input_ids"][0] = 99
        self.assertNotEqual(actual, self.report())

    def test_dependency_failure_and_no_prebuilt_inventory(self):
        inputs = copy.deepcopy(self.inputs)
        inputs["records"] = []
        actual = report_module.build_synthetic_report(
            inputs, self.profile_raw, self.profile_sha, [], None)
        self.assertEqual(actual["public"]["status"], "DEPENDENCY_UNSATISFIED")
        self.assertEqual(actual["public"]["not_checked"], 120)
        self.assertEqual(actual["public"]["rows"], [])
        with self.assertRaises(ValueError):
            report_module.build_synthetic_report(
                self.inventory, self.profile_raw, self.profile_sha, [], None)
        with self.assertRaises(ValueError):
            report_module.build_synthetic_report(
                inputs, self.profile_raw, self.profile_sha, self.records, self.finalization)

    def test_input_and_answer_over_limit_stop(self):
        for count, stop in ((1, "INPUT_LIMIT"), (25, "OUTPUT_LIMIT")):
            records, final = synthetic_transcript(self.inventory, self.profile, count=count, stop=stop)
            public = self.report(records, final)["public"]
            self.assertEqual(public["stop_reason"], stop)
            self.assertEqual(public["checked"], count)
            self.assertEqual(public["not_checked"], 120 - count)
            self.assertFalse(public["synthetic_all_rows_within_budget"])
            with self.assertRaises(ValueError):
                self.report(records + [self.records[count]], final)

    def test_exact_budget_boundaries(self):
        records, final = copy.deepcopy((self.records, self.finalization))
        observed = records[0]["measurement"]["input"]
        observed.update(input_ids=[11] * 4096, input_tokens=4096, remaining_input_tokens=0)
        target = records[24]["measurement"]["target"]
        target["target_ids"] = [12] * 255 + [2]
        target["target_tokens"] = 256
        target["full_sequence_ids"] = records[24]["measurement"]["input"]["input_ids"] + target["target_ids"] + target["template_tail_ids"]
        target["full_sequence_tokens"] = len(target["full_sequence_ids"])
        rechain(records, final)
        self.assertTrue(self.report(records, final)["public"]["synthetic_all_rows_within_budget"])

    def test_binding_and_chain_tampering(self):
        for path, replacement in (
            (("run_id",), "b" * 32), (("ordinal",), True),
            (("profile_sha256",), "0" * 64), (("identity_sha256",), "0" * 64),
            (("measurement", "state_sha256"), "0" * 64),
            (("measurement", "input", "messages_sha256"), "0" * 64),
            (("measurement", "input", "input_tokens"), True),
            (("measurement", "status"), "INPUT_LIMIT"),
        ):
            records, final = copy.deepcopy((self.records, self.finalization))
            changed = records[1]
            for name in path[:-1]:
                changed = changed[name]
            changed[path[-1]] = replacement
            rechain(records, final)
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.report(records, final)
        records = copy.deepcopy(self.records)
        records[0]["chain_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.report(records)

    def test_finalization_eos_missing_and_failure(self):
        for records, final in ((self.records, None), (self.records[:1], self.finalization)):
            with self.assertRaises(ValueError):
                report_module.build_synthetic_report(
                    self.inputs, self.profile_raw, self.profile_sha, records, final)
        records, final = copy.deepcopy((self.records, self.finalization))
        target = records[24]["measurement"]["target"]
        target["target_ids"][-1] = 12
        target["full_sequence_ids"] = [11] * 3 + target["target_ids"] + [13]
        rechain(records, final)
        with self.assertRaises(ValueError):
            self.report(records, final)
        actual = report_module.build_synthetic_report(
            self.inputs, self.profile_raw, self.profile_sha, self.records, None,
            failure={"phase": "FINALIZE", "ordinal": None}, abort_failed=True)
        self.assertEqual(actual["public"]["status"], "SYNTHETIC_FAILED")
        self.assertFalse(actual["public"]["synthetic_all_rows_within_budget"])
        self.assertTrue(actual["public"]["abort_failed"])

    def test_no_execution_imports(self):
        tree = ast.parse(Path(report_module.__file__).read_text())
        forbidden = {"tokenizer_session", "transformers", "torch", "subprocess", "socket"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name in forbidden for alias in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module, forbidden)
                self.assertFalse(any(alias.name in ("_row", "_ORDER", "validate_record")
                                     for alias in node.names) if node.module == "tokenizer_audit_report" else False)


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Capture actual missing-feature RED.**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_tokenizer_report -v
```

Expected seven assertion failures, not import errors. Save raw log and immediate exit.

- [x] **Step 3: Implement this candidate and repair demonstrated defects test-first.**

```python
import copy

from pilot_protocol import FLAGS, fields, require
from recalibration_data_contract import _plain
from recalibration_tokenizer_inventory import PLANNED_COUNTS, build_synthetic_inventory
from tokenizer_audit_report import _finalization, _hex, _ids, _integer, _public_validation, _text_hash
from tokenizer_runtime_profile import validate_profile
from validation_contract import digest


REAL_FLAGS = (*FLAGS, "all_possible_second_rounds_validated", "tokenizer_length_validated",
              "execution_provenance_verified", "real_dynamic_validation_complete",
              "run_authorized", "tokenizer_dispatch_authorized")
INPUT_FIELDS = ("raw_files", "worker_raw", "run_ids", "records", "rendering_raw", "expected_runtime")
REASONS = {"OPEN": "OPEN_FAILED", "MEASURE": "MEASUREMENT_FAILED",
           "RECEIPT": "RECEIPT_INVALID", "FINALIZE": "FINALIZATION_FAILED"}


@_public_validation
def prepare_synthetic_inputs(validation_inputs, profile_raw, approved_profile_sha256):
    fields(validation_inputs, INPUT_FIELDS)
    require(all(type(name) is str for name in validation_inputs), "input keys")
    inventory = build_synthetic_inventory(**validation_inputs)
    profile = validate_profile(profile_raw, approved_profile_sha256)
    return inventory, profile


def _measurement(row, measurement):
    fields(measurement, ("kind", "state_sha256", "status", "input", "target"))
    require(measurement["kind"] == "TOKENIZER_MEASUREMENT_UNATTESTED", "measurement kind")
    _hex(measurement["state_sha256"])
    observed = measurement["input"]
    fields(observed, ("messages_sha256", "prompt", "prompt_sha256", "input_ids",
                      "input_tokens", "remaining_input_tokens"))
    require(observed["messages_sha256"] == digest(row["messages"]), "message binding")
    require(observed["prompt_sha256"] == _text_hash(observed["prompt"]), "prompt binding")
    _ids(observed["input_ids"])
    _integer(observed["input_tokens"], minimum=1)
    require(observed["input_tokens"] == len(observed["input_ids"]), "input count")
    require(type(observed["remaining_input_tokens"]) is int and
            observed["remaining_input_tokens"] == 4096 - observed["input_tokens"], "remaining input")
    target = measurement["target"]
    status = "INPUT_LIMIT" if observed["input_tokens"] > 4096 else "OK"
    if status == "INPUT_LIMIT" or row["answer"] is None:
        require(target is None, "unexpected target")
    else:
        fields(target, ("answer_sha256", "full_prompt_sha256", "full_sequence_ids",
                        "full_sequence_tokens", "target_start", "target_ids",
                        "target_tokens", "template_tail_ids"))
        require(target["answer_sha256"] == _text_hash(row["answer"]), "answer binding")
        _hex(target["full_prompt_sha256"])
        for name in ("full_sequence_ids", "target_ids", "template_tail_ids"):
            _ids(target[name], nonempty=name != "template_tail_ids")
        for name in ("full_sequence_tokens", "target_start", "target_tokens"):
            _integer(target[name])
        require(target["target_start"] == observed["input_tokens"], "target boundary")
        require(target["full_sequence_tokens"] == len(target["full_sequence_ids"]) and
                target["target_tokens"] == len(target["target_ids"]), "target counts")
        require(target["full_sequence_ids"] == observed["input_ids"] + target["target_ids"] +
                target["template_tail_ids"], "sequence partition")
        if target["target_tokens"] > 256:
            status = "OUTPUT_LIMIT"
    require(measurement["status"] == status, "measurement status")


@_public_validation
def _validate_record(row, record, run_id, profile_sha256, previous_chain,
                     identity_sha256=None, state_sha256=None):
    _plain(row)
    _plain(record)
    _hex(run_id, 32)
    for value in (profile_sha256, previous_chain):
        _hex(value)
    fields(record, ("kind", "run_id", "profile_sha256", "identity_sha256",
                    "ordinal", "measurement", "chain_sha256"))
    require(record["kind"] == "TOKENIZER_SESSION_MEASUREMENT_PROVISIONAL", "record kind")
    require(record["run_id"] == run_id and record["profile_sha256"] == profile_sha256,
            "session binding")
    _integer(record["ordinal"])
    require(record["ordinal"] == row["ordinal"], "record position")
    _hex(record["identity_sha256"])
    require(identity_sha256 is None or record["identity_sha256"] == identity_sha256,
            "identity continuity")
    _measurement(row, record["measurement"])
    require(state_sha256 is None or record["measurement"]["state_sha256"] == state_sha256,
            "state continuity")
    unsigned = {name: value for name, value in record.items() if name != "chain_sha256"}
    require(record["chain_sha256"] == digest({"previous": previous_chain, "record": unsigned}),
            "record chain")
    return copy.deepcopy(record)


def _failure(failure, abort_failed, count, finalization, stop):
    require(type(abort_failed) is bool, "abort flag")
    if failure is None:
        require(not abort_failed and finalization is not None, "missing finalization")
        require(count == 120 or stop is not None, "incomplete measurements")
        return None
    _plain(failure)
    fields(failure, ("phase", "ordinal"))
    phase, ordinal = failure["phase"], failure["ordinal"]
    require(type(phase) is str and phase in REASONS and finalization is None, "failure phase")
    if phase == "OPEN":
        require(ordinal is None and count == 0 and not abort_failed, "open failure")
    elif phase == "FINALIZE":
        require(ordinal is None and (count == 120 or stop is not None), "finish failure")
    else:
        require(type(ordinal) is int and ordinal == count < 120 and stop is None, "row failure")
    return {"phase": phase, "ordinal": ordinal, "reason": REASONS[phase]}


@_public_validation
def build_synthetic_report(validation_inputs, profile_raw, approved_profile_sha256,
                           records, finalization, *, failure=None, abort_failed=False):
    inventory, profile = prepare_synthetic_inputs(validation_inputs, profile_raw, approved_profile_sha256)
    _plain(records)
    _plain(finalization)
    require(type(records) is list and len(records) <= 120 and type(abort_failed) is bool,
            "record denominator")
    ready = inventory["status"] == "SYNTHETIC_ONLY"
    checked, chain, identity, state, stop, stop_ordinal = [], digest([]), None, None, None, None
    public_failure = None
    if not ready:
        require(not records and finalization is None and failure is None and not abort_failed,
                "unsatisfied dependency evidence")
    else:
        for row, record in zip(inventory["rows"], records):
            require(stop is None, "measurement after stop")
            record = _validate_record(row, record, profile["run_id"], approved_profile_sha256,
                                      chain, identity, state)
            chain, identity = record["chain_sha256"], record["identity_sha256"]
            state = record["measurement"]["state_sha256"]
            checked.append(record)
            if record["measurement"]["status"] != "OK":
                stop, stop_ordinal = record["measurement"]["status"], row["ordinal"]
        public_failure = _failure(failure, abort_failed, len(checked), finalization, stop)
        if public_failure is None:
            _finalization(finalization, profile, approved_profile_sha256, checked, chain, stop)
        elif stop is None:
            stop, stop_ordinal = public_failure["reason"], public_failure["ordinal"]
    counts = {name: {"scheduled": total, "checked": 0, "not_checked": total, "over_limit": 0}
              for name, total in PLANNED_COUNTS.items()}
    rows = []
    for row in inventory["rows"]:
        ordinal = row["ordinal"]
        measurement = checked[ordinal]["measurement"] if ordinal < len(checked) else None
        status = "NOT_RUN_AFTER_STOP"
        if measurement is not None:
            status = measurement["status"]
        elif public_failure is not None and public_failure["ordinal"] == ordinal:
            status = public_failure["reason"]
        target = None if measurement is None else measurement["target"]
        rows.append({
            **{name: row[name] for name in ("ordinal", "category", "task_id", "arm",
                                          "reference_id", "branch", "row_sha256")},
            "status": status,
            "input_tokens": None if measurement is None else measurement["input"]["input_tokens"],
            "target_tokens": None if target is None else target["target_tokens"],
            "full_sequence_tokens": None if target is None else target["full_sequence_tokens"],
        })
        if measurement is not None:
            counts[row["category"]]["checked"] += 1
            counts[row["category"]]["not_checked"] -= 1
        counts[row["category"]]["over_limit"] += int(status in ("INPUT_LIMIT", "OUTPUT_LIMIT"))
    status = "DEPENDENCY_UNSATISFIED"
    if ready:
        status = "SYNTHETIC_FAILED" if public_failure else (
            "SYNTHETIC_STOPPED" if stop else "SYNTHETIC_COMPLETE")
    public = {
        "kind": "LORA_RECALIBRATION_TOKENIZER_REPORT", "version": 1,
        "data_version": "recalibration_v2_r1", "status": status,
        "synthetic_only": True, "evidence_kind": "synthetic",
        "run_id": profile["run_id"], "profile_sha256": approved_profile_sha256,
        "inventory_sha256": inventory["inventory_sha256"],
        "validation_evidence_sha256": inventory["dynamic_evidence_sha256"],
        "planned": 120, "checked": len(checked), "not_checked": 120 - len(checked),
        "over_limit": sum(count["over_limit"] for count in counts.values()),
        "rows": rows, "by_category": counts, "stop_reason": stop, "stop_ordinal": stop_ordinal,
        "failure": public_failure, "abort_failed": abort_failed,
        "session_finalized": ready and public_failure is None,
        "synthetic_all_rows_within_budget": ready and public_failure is None and
                                           len(checked) == 120 and stop is None,
        **{flag: False for flag in REAL_FLAGS},
    }
    return {"public": public, "private": {
        "inventory": inventory, "records": checked, "finalization": copy.deepcopy(finalization)}}
```

- [x] **Step 4: Focused GREEN and narrow regression.**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_tokenizer_report -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_tokenizer_audit_report test_tokenizer_runtime_profile test_recalibration_tokenizer_inventory test_recalibration_tokenizer_report -v
```

Retain raw logs and immediate exits. Additional test cases must explicitly cover invalid profile, runtime/snapshot/identity/EOS finalization, duplicate/reordered records, strict nested types and every legal/illegal failure phase. Never execute stored sources or use real tokenizer/runtime setup.

- [x] **Step 5: Independent review and preservation checkpoint, no commit.**

Controller gives a fresh reviewer this brief, actual additions diff and implementation report. Compare all baseline file hashes. The later local fake-session coordinator is a separate subplan; no real source or length results are claimed.

## Self-Review

- This is the synthetic subset of parent sections 6–7, not a real tokenizer admission path.
- Inventory reconstruction authenticates new row structure from frozen inputs. The internal record helper is not a standalone inventory validator.
- Schema-independent finalization retains unique EOS and fixed runtime/snapshot constraints without invoking any old fixed-denominator entry point.
- Finalization failure cannot produce even synthetic complete/budget success.
- Actual tokenizer equivalence, answer-prefix behavior and runtime provenance require the unchanged trusted session and future authorized collection; synthetic token lists cannot prove them.
