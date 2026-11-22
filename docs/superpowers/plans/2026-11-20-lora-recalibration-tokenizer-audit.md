# LoRA Recalibration Synthetic Session Coordinator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive the new 120-row report contract through an explicitly supplied local fake session and verify lifecycle failure behavior.

**Architecture:** Freeze validated inputs, open one caller-supplied synthetic session, measure in order, check each receipt, finish on completion/normal limit, and abort on exceptions. No default opener, root, real session import, download, or launch path is added. The supplied factory is an execution capability owned by the caller, not authenticated as synthetic by Python; this API never authorizes real invocation.

**Tech Stack:** Python standard library, completed pure report/inventory modules and unittest fake sessions.

## Global Constraints

- Parent design: `docs/superpowers/specs/2026-11-19-lora-recalibration-validation-adapter-design.md`, local synthetic subset of sections 6–7.
- Existing checkout and all old files/data preserved. No network/SSH, stored task-source execution/import, real tokenizer/model loading, generation/training, installs, telemetry/bootstrap/flush, branch/worktree, commit/push or cleanup.
- Synthetic test sessions only; all real flags false. No claim of real tokenizer equivalence, length validation, runtime provenance or release.
- Use apply_patch, no inline code comments or one-letter variables; no child subagents.
- Parent owns documentation/ledger. Implementer owns only the two new files below and task-2-* evidence.

### Task 1: Local Synthetic Coordinator

**Files:**
- Create `experiments/lora_necessity/recalibration_tokenizer_audit.py`.
- Create `experiments/lora_necessity/test_recalibration_tokenizer_audit.py`.

**Interfaces and requirements:**

- `run_synthetic_audit(validation_inputs, profile_raw, approved_profile_sha256, *, session_factory)`.
- Consumes `prepare_synthetic_inputs`, `_validate_record`, `build_synthetic_report` from the completed `recalibration_tokenizer_report.py`. Exact argument contracts are in the candidate below; only verified reconstructed rows go to `_validate_record`.
- Factory must be explicitly supplied, receives `(profile_raw, approved_profile_sha256)` once, returns an object supporting `measure(messages, answer)`, `finish()`, `abort()`. This matches existing session methods without importing or calling the real opener. Caller must supply only a local synthetic factory in this stage. No callback provenance/security guarantee is claimed.
- Invalid inputs fail before factory call. Valid but unsatisfied data dependency returns dependency report with no session calls.
- Freeze the original validated validation inputs before callbacks; callback mutation of original inputs cannot change the inventory/report under audit. Pass detached messages and reject mutation of them, even if the receipt claims the original hash.
- One sequential session, no retry/reopen/deduplication. First limit uses finish, not abort. First exception/invalid receipt stops; no later measure calls.
- Validate chain and identity/state immediately on each receipt, before append. Freeze accepted records so later mutation of a prior returned object cannot alter accepted history.
- Finish returns finalization to the pure report validator; invalid/missing finalization is FINALIZE failure and triggers abort. Never label all rows successful on finalization failure.
- Ordinary exceptions become bounded public phase/ordinal failure with abort_failed; do not expose exception messages. On BaseException cancellation attempt abort then propagate cancellation, never convert it into normal report. If abort raises BaseException while original was ordinary Exception, propagate abort cancellation. Otherwise preserve original cancellation; report an ordinary abort error via abort_failed.
- No real opener, runtime probe, tokenizer library, subprocess, SSH or filesystem access in production module. Public flags and privacy follow report contract; no cleanup status invented.

- [x] **Step 1: Add real fake-session tests before production exists.**

```python
import ast
import copy
import importlib.util
from pathlib import Path
import unittest

from recalibration_tokenizer_report import build_synthetic_report, prepare_synthetic_inputs
from test_recalibration_tokenizer_report import fixture_inputs
from test_tokenizer_audit_report import synthetic_profile, synthetic_transcript, rechain
from validation_contract import digest, encoded

if importlib.util.find_spec("recalibration_tokenizer_audit") is not None:
    import recalibration_tokenizer_audit as audit_module
else:
    audit_module = None


class FakeSession:
    def __init__(self, records, finalization):
        self.records, self.finalization = copy.deepcopy((records, finalization))
        self.calls = []
        self.measure_error = None
        self.finish_error = None
        self.abort_error = None
        self.hook = None

    def measure(self, messages, answer):
        ordinal = sum(call[0] == "measure" for call in self.calls)
        self.calls.append(("measure", copy.deepcopy(messages), answer))
        if self.measure_error is not None:
            raise self.measure_error
        if self.hook is not None:
            self.hook(ordinal, messages)
        return self.records[ordinal]

    def finish(self):
        self.calls.append(("finish",))
        if self.finish_error is not None:
            raise self.finish_error
        return self.finalization

    def abort(self):
        self.calls.append(("abort",))
        if self.abort_error is not None:
            raise self.abort_error


class RecalibrationTokenizerAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = fixture_inputs()
        cls.profile = synthetic_profile()
        cls.profile_raw, cls.approved = encoded(cls.profile), digest(cls.profile)
        cls.inventory, _ = prepare_synthetic_inputs(cls.inputs, cls.profile_raw, cls.approved)
        cls.records, cls.final = synthetic_transcript(cls.inventory, cls.profile, count=120)

    def setUp(self):
        self.assertIsNotNone(audit_module, "recalibration tokenizer audit not implemented")

    def run_session(self, session, inputs=None):
        calls = []
        def factory(raw, approved):
            calls.append((raw, approved))
            return session
        result = audit_module.run_synthetic_audit(
            self.inputs if inputs is None else inputs, self.profile_raw,
            self.approved, session_factory=factory)
        self.assertEqual(calls, [(self.profile_raw, self.approved)])
        return result

    def test_complete_session_matches_pure_reconstruction(self):
        session = FakeSession(self.records, self.final)
        actual = self.run_session(session)
        expected = build_synthetic_report(
            self.inputs, self.profile_raw, self.approved, self.records, self.final)
        self.assertEqual(actual, expected)
        self.assertEqual(session.calls, [
            ("measure", row["messages"], row["answer"]) for row in self.inventory["rows"]
        ] + [("finish",)])

    def test_dependency_and_invalid_inputs_never_open(self):
        calls = []
        def factory(raw, approved):
            calls.append((raw, approved))
            raise AssertionError("must not open")
        inputs = copy.deepcopy(self.inputs)
        inputs["records"] = []
        result = audit_module.run_synthetic_audit(
            inputs, self.profile_raw, self.approved, session_factory=factory)
        self.assertEqual(result["public"]["status"], "DEPENDENCY_UNSATISFIED")
        with self.assertRaises(ValueError):
            audit_module.run_synthetic_audit(
                {}, self.profile_raw, self.approved, session_factory=factory)
        self.assertEqual(calls, [])

    def test_normal_limits_finish_without_abort(self):
        for count, stop in ((1, "INPUT_LIMIT"), (25, "OUTPUT_LIMIT")):
            session = FakeSession(*synthetic_transcript(
                self.inventory, self.profile, count=count, stop=stop))
            result = self.run_session(session)
            self.assertEqual(result["public"]["stop_reason"], stop)
            self.assertEqual(len(session.calls), count + 1)
            self.assertEqual(session.calls[-1], ("finish",))

    def test_open_measure_finish_and_abort_failures(self):
        def factory(raw, approved):
            raise RuntimeError("PRIVATE FAILURE")
        result = audit_module.run_synthetic_audit(
            self.inputs, self.profile_raw, self.approved, session_factory=factory)
        self.assertEqual(result["public"]["stop_reason"], "OPEN_FAILED")
        for attribute, phase in (("measure_error", "MEASURE"), ("finish_error", "FINALIZE")):
            session = FakeSession(self.records, self.final)
            setattr(session, attribute, RuntimeError("PRIVATE FAILURE"))
            session.abort_error = RuntimeError("PRIVATE RELEASE")
            result = self.run_session(session)
            self.assertEqual(result["public"]["failure"]["phase"], phase)
            self.assertTrue(result["public"]["abort_failed"])
            self.assertFalse(result["public"]["synthetic_all_rows_within_budget"])
            self.assertNotIn("PRIVATE", encoded(result).decode())
            self.assertEqual(session.calls[-1], ("abort",))

    def test_bad_receipt_and_finalization_stop(self):
        session = FakeSession(self.records, self.final)
        session.records[1]["ordinal"] = 4
        rechain(session.records, session.finalization)
        result = self.run_session(session)
        self.assertEqual(result["public"]["failure"], {
            "phase": "RECEIPT", "ordinal": 1, "reason": "RECEIPT_INVALID"})
        self.assertEqual([call[0] for call in session.calls], ["measure", "measure", "abort"])
        session = FakeSession(self.records, None)
        result = self.run_session(session)
        self.assertEqual(result["public"]["failure"]["phase"], "FINALIZE")
        self.assertFalse(result["public"]["session_finalized"])

    def test_mutable_inputs_messages_and_previous_receipts(self):
        inputs = copy.deepcopy(self.inputs)
        session = FakeSession(self.records, self.final)
        def external_mutation(ordinal, messages):
            inputs["records"].clear()
            if ordinal == 1:
                session.records[0]["measurement"]["input"]["input_ids"][0] = 99
        session.hook = external_mutation
        result = self.run_session(session, inputs)
        self.assertEqual(result["public"]["status"], "SYNTHETIC_COMPLETE")
        self.assertEqual(result["private"]["records"][0]["measurement"]["input"]["input_ids"][0], 11)
        session = FakeSession(self.records, self.final)
        def message_mutation(ordinal, messages):
            messages[0]["content"] = "changed"
        session.hook = message_mutation
        result = self.run_session(session)
        self.assertEqual(result["public"]["stop_reason"], "RECEIPT_INVALID")
        self.assertEqual([call[0] for call in session.calls], ["measure", "abort"])

    def test_interrupt_propagates_after_abort(self):
        for attribute in ("measure_error", "finish_error"):
            session = FakeSession(self.records, self.final)
            setattr(session, attribute, KeyboardInterrupt())
            with self.assertRaises(KeyboardInterrupt):
                self.run_session(session)
            self.assertEqual(session.calls[-1], ("abort",))
        session = FakeSession(self.records, self.final)
        session.measure_error = ValueError("PRIVATE")
        session.abort_error = SystemExit(7)
        with self.assertRaises(SystemExit):
            self.run_session(session)

    def test_no_real_opener_or_execution_import(self):
        tree = ast.parse(Path(audit_module.__file__).read_text())
        allowed = {"copy", "pilot_protocol", "recalibration_tokenizer_report",
                   "tokenizer_audit_report", "validation_contract"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name in allowed for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"open", "exec", "eval", "compile", "__import__"})


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Capture missing-feature RED.**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_tokenizer_audit -v
```

Expected eight assertion failures, no import errors. Preserve log and immediate exit.

- [x] **Step 3: Implement this candidate; repair demonstrated defects test-first.**

```python
import copy

from pilot_protocol import require
from recalibration_tokenizer_report import (
    _validate_record, build_synthetic_report, prepare_synthetic_inputs,
)
from tokenizer_audit_report import _public_validation
from validation_contract import digest


@_public_validation
def run_synthetic_audit(validation_inputs, profile_raw, approved_profile_sha256, *, session_factory):
    inventory, profile = prepare_synthetic_inputs(
        validation_inputs, profile_raw, approved_profile_sha256)
    require(callable(session_factory), "explicit synthetic session factory")
    frozen_inputs = copy.deepcopy(validation_inputs)
    if inventory["status"] == "DEPENDENCY_UNSATISFIED":
        return build_synthetic_report(
            frozen_inputs, profile_raw, approved_profile_sha256, [], None)
    session, phase, ordinal = None, "OPEN", None
    records, chain, identity, state = [], digest([]), None, None
    try:
        session = session_factory(profile_raw, approved_profile_sha256)
        for row in inventory["rows"]:
            phase, ordinal = "MEASURE", row["ordinal"]
            messages = copy.deepcopy(row["messages"])
            record = session.measure(messages, row["answer"])
            phase = "RECEIPT"
            require(digest(messages) == digest(row["messages"]), "messages mutated")
            record = _validate_record(
                row, record, profile["run_id"], approved_profile_sha256, chain, identity, state)
            chain, identity = record["chain_sha256"], record["identity_sha256"]
            state = record["measurement"]["state_sha256"]
            records.append(record)
            if record["measurement"]["status"] != "OK":
                break
        phase, ordinal = "FINALIZE", None
        finalization = session.finish()
        return build_synthetic_report(
            frozen_inputs, profile_raw, approved_profile_sha256, records, finalization)
    except BaseException as error:
        abort_failed = False
        if session is not None:
            try:
                session.abort()
            except BaseException as abort_error:
                if isinstance(error, Exception) and not isinstance(abort_error, Exception):
                    raise
                abort_failed = True
        if not isinstance(error, Exception):
            raise
        return build_synthetic_report(
            frozen_inputs, profile_raw, approved_profile_sha256, records, None,
            failure={"phase": phase, "ordinal": ordinal}, abort_failed=abort_failed)
```

- [x] **Step 4: Focused GREEN and final narrow integration regression.**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_tokenizer_audit -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_tokenizer_audit_report test_tokenizer_runtime_profile test_recalibration_data_contract test_recalibration_validation test_recalibration_rendering test_recalibration_tokenizer_inventory test_recalibration_tokenizer_report test_recalibration_tokenizer_audit -v
```

Keep raw logs and immediate exits. Expand before GREEN to cover identity/state drift stopping immediately, ordinary abort success/failure, finalization rejection after normal limit, OPEN cancellation, callback argument identity and missing factory. No existing data or source edits.

- [x] **Step 5: Independent task plus final two-component integration review.**

Controller checks actual four-file additions, source/test/report hashes and 178 preserved files, closes ledger and documents synthetic limits. No commit. Real session binding/launch/provenance remain separate work.

## Self-Review

- No implicit real callback or root; this step exercises lifecycle logic with fake sessions only.
- Normal over-limit stop and exceptional failure use distinct terminal paths.
- Accepted receipt snapshots are detached; input snapshots precede callbacks.
- Finalization validation occurs inside the guarded region, so malformed finish evidence triggers abort and cannot grant synthetic success.
- Any true callback execution remains the caller's responsibility. Labeling this coordinator synthetic is not a sandbox or a way to authenticate arbitrary callbacks.
