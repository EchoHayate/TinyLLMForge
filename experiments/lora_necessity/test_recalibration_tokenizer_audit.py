import ast
import copy
import importlib.util
from pathlib import Path
import unittest

from recalibration_tokenizer_report import (
    REAL_FLAGS, build_synthetic_report, prepare_synthetic_inputs,
)
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
        cls.inventory = prepare_synthetic_inputs(cls.inputs, cls.profile_raw, cls.approved)[0]
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
            with self.subTest(stop=stop):
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
            with self.subTest(phase=phase):
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
            with self.subTest(attribute=attribute):
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

    def test_factory_receives_original_arguments_once(self):
        session = FakeSession(self.records, self.final)
        calls = []

        def factory(raw, approved):
            self.assertIs(raw, self.profile_raw)
            self.assertIs(approved, self.approved)
            calls.append((raw, approved))
            return session

        result = audit_module.run_synthetic_audit(
            self.inputs, self.profile_raw, self.approved, session_factory=factory)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["public"]["checked"], 120)
        self.assertEqual([call[0] for call in session.calls], ["measure"] * 120 + ["finish"])

    def test_missing_and_noncallable_factory_rejected(self):
        with self.assertRaisesRegex(ValueError, "^tokenizer audit evidence invalid$"):
            audit_module.run_synthetic_audit(self.inputs, self.profile_raw, self.approved)
        for factory in (None, False, {}, "not a factory"):
            with self.subTest(factory=factory):
                with self.assertRaisesRegex(ValueError, "^tokenizer audit evidence invalid$"):
                    audit_module.run_synthetic_audit(
                        self.inputs, self.profile_raw, self.approved, session_factory=factory)

    def test_invalid_input_and_profile_fail_before_factory(self):
        calls = []

        def factory(raw, approved):
            calls.append((raw, approved))
            raise AssertionError("factory must not run")

        for inputs, raw, approved in (
            ({}, self.profile_raw, self.approved),
            (self.inputs, b"PRIVATE invalid profile", self.approved),
            (self.inputs, self.profile_raw, "0" * 64),
        ):
            with self.subTest(raw=raw[:10], approved=approved):
                with self.assertRaisesRegex(ValueError, "^tokenizer audit evidence invalid$"):
                    audit_module.run_synthetic_audit(
                        inputs, raw, approved, session_factory=factory)
        self.assertEqual(calls, [])

    def test_factory_mutation_cannot_change_validated_inputs(self):
        inputs = copy.deepcopy(self.inputs)
        session = FakeSession(self.records, self.final)

        def factory(raw, approved):
            inputs.clear()
            return session

        result = audit_module.run_synthetic_audit(
            inputs, self.profile_raw, self.approved, session_factory=factory)
        expected = build_synthetic_report(
            self.inputs, self.profile_raw, self.approved, self.records, self.final)
        self.assertEqual(result, expected)
        self.assertEqual(inputs, {})

    def test_input_mutation_before_failure_keeps_frozen_report(self):
        inputs = copy.deepcopy(self.inputs)
        session = FakeSession(self.records, self.final)

        def mutate_then_fail(ordinal, messages):
            inputs.clear()
            raise RuntimeError("PRIVATE mutated input")

        session.hook = mutate_then_fail
        result = self.run_session(session, inputs)
        expected = build_synthetic_report(
            self.inputs, self.profile_raw, self.approved, [], None,
            failure={"phase": "MEASURE", "ordinal": 0})
        self.assertEqual(result, expected)
        self.assertEqual([call[0] for call in session.calls], ["measure", "abort"])

    def test_identity_and_state_drift_stop_before_next_measure(self):
        for field in ("identity_sha256", "state_sha256"):
            with self.subTest(field=field):
                session = FakeSession(self.records, self.final)
                if field == "state_sha256":
                    session.records[1]["measurement"][field] = "0" * 64
                else:
                    session.records[1][field] = "0" * 64
                rechain(session.records, session.finalization)
                result = self.run_session(session)
                expected = build_synthetic_report(
                    self.inputs, self.profile_raw, self.approved, self.records[:1], None,
                    failure={"phase": "RECEIPT", "ordinal": 1})
                self.assertEqual(result, expected)
                self.assertEqual([call[0] for call in session.calls],
                                 ["measure", "measure", "abort"])

    def test_invalid_receipts_are_not_appended_or_retried(self):
        for field, replacement in (
            ("chain_sha256", "0" * 64),
            ("run_id", "0" * 32),
            ("profile_sha256", "0" * 64),
            ("ordinal", True),
        ):
            with self.subTest(field=field):
                session = FakeSession(self.records, self.final)
                session.records[2][field] = replacement
                if field != "chain_sha256":
                    rechain(session.records, session.finalization)
                result = self.run_session(session)
                self.assertEqual(result["public"]["failure"], {
                    "phase": "RECEIPT", "ordinal": 2, "reason": "RECEIPT_INVALID"})
                self.assertEqual(result["private"]["records"], self.records[:2])
                self.assertEqual(result["public"]["not_checked"], 118)
                self.assertEqual([call[0] for call in session.calls],
                                 ["measure"] * 3 + ["abort"])

    def test_measure_exception_keeps_only_accepted_prefix(self):
        session = FakeSession(self.records, self.final)

        def fail_later(ordinal, messages):
            if ordinal == 25:
                raise RuntimeError("PRIVATE later failure")

        session.hook = fail_later
        result = self.run_session(session)
        expected = build_synthetic_report(
            self.inputs, self.profile_raw, self.approved, self.records[:25], None,
            failure={"phase": "MEASURE", "ordinal": 25})
        self.assertEqual(result, expected)
        self.assertEqual([call[0] for call in session.calls], ["measure"] * 26 + ["abort"])
        self.assertEqual(result["public"]["rows"][25]["status"], "MEASUREMENT_FAILED")
        self.assertTrue(all(row["status"] == "NOT_RUN_AFTER_STOP"
                            for row in result["public"]["rows"][26:]))

    def test_ordinary_abort_success_and_failure_are_bounded(self):
        for phase in ("MEASURE", "RECEIPT", "FINALIZE"):
            for abort_fails in (False, True):
                with self.subTest(phase=phase, abort_fails=abort_fails):
                    session = FakeSession(self.records, self.final)
                    if phase == "MEASURE":
                        session.measure_error = RuntimeError("PRIVATE measurement")
                    elif phase == "RECEIPT":
                        session.records[0] = None
                    else:
                        session.finish_error = RuntimeError("PRIVATE finalization")
                    if abort_fails:
                        session.abort_error = RuntimeError("PRIVATE abort")
                    result = self.run_session(session)
                    expected = build_synthetic_report(
                        self.inputs, self.profile_raw, self.approved,
                        self.records if phase == "FINALIZE" else [], None,
                        failure={"phase": phase, "ordinal": None if phase == "FINALIZE" else 0},
                        abort_failed=abort_fails)
                    self.assertEqual(result, expected)
                    self.assertEqual([call[0] for call in session.calls],
                                     ["measure"] * 120 + ["finish", "abort"]
                                     if phase == "FINALIZE" else ["measure", "abort"])
                    self.assertNotIn("PRIVATE", encoded(result).decode())

    def test_finalization_rejection_after_complete_or_normal_limit_aborts(self):
        for count, stop in ((120, None), (1, "INPUT_LIMIT"), (25, "OUTPUT_LIMIT")):
            for defect in ("missing", "count", "chain", "identity", "real_flag"):
                with self.subTest(stop=stop, defect=defect):
                    records, finalization = synthetic_transcript(
                        self.inventory, self.profile, count=count, stop=stop)
                    session = FakeSession(records, finalization)
                    if defect == "missing":
                        session.finalization = None
                    elif defect == "count":
                        session.finalization["measurements"] += 1
                    elif defect == "chain":
                        session.finalization["chain_sha256"] = "0" * 64
                    elif defect == "identity":
                        session.finalization["identity"]["class"] = "different.Tokenizer"
                    else:
                        session.finalization["benchmark_eligible"] = True
                    result = self.run_session(session)
                    expected = build_synthetic_report(
                        self.inputs, self.profile_raw, self.approved, records, None,
                        failure={"phase": "FINALIZE", "ordinal": None})
                    self.assertEqual(result, expected)
                    self.assertEqual(result["public"]["status"], "SYNTHETIC_FAILED")
                    self.assertFalse(result["public"]["session_finalized"])
                    self.assertFalse(result["public"]["synthetic_all_rows_within_budget"])
                    self.assertEqual([call[0] for call in session.calls],
                                     ["measure"] * count + ["finish", "abort"])

    def test_open_cancellation_propagates_unchanged(self):
        for cancellation in (KeyboardInterrupt("PRIVATE"), SystemExit(7), BaseException("PRIVATE")):
            with self.subTest(cancellation=type(cancellation).__name__):
                calls = []

                def factory(raw, approved):
                    calls.append((raw, approved))
                    raise cancellation

                with self.assertRaises(type(cancellation)) as raised:
                    audit_module.run_synthetic_audit(
                        self.inputs, self.profile_raw, self.approved, session_factory=factory)
                self.assertIs(raised.exception, cancellation)
                self.assertEqual(calls, [(self.profile_raw, self.approved)])

    def test_original_cancellation_survives_abort_errors(self):
        for attribute in ("measure_error", "finish_error"):
            for abort_error in (None, RuntimeError("PRIVATE abort"), SystemExit(19)):
                with self.subTest(attribute=attribute, abort_error=type(abort_error).__name__):
                    session = FakeSession(*synthetic_transcript(
                        self.inventory, self.profile, count=1, stop="INPUT_LIMIT"))
                    cancellation = KeyboardInterrupt("PRIVATE cancellation")
                    setattr(session, attribute, cancellation)
                    session.abort_error = abort_error
                    with self.assertRaises(KeyboardInterrupt) as raised:
                        self.run_session(session)
                    self.assertIs(raised.exception, cancellation)
                    self.assertEqual([call[0] for call in session.calls],
                                     ["measure", "abort"] if attribute == "measure_error"
                                     else ["measure", "finish", "abort"])

    def test_abort_cancellation_overrides_ordinary_error(self):
        for phase in ("MEASURE", "RECEIPT", "FINALIZE"):
            with self.subTest(phase=phase):
                session = FakeSession(*synthetic_transcript(
                    self.inventory, self.profile, count=1, stop="INPUT_LIMIT"))
                if phase == "MEASURE":
                    session.measure_error = RuntimeError("PRIVATE measurement")
                elif phase == "RECEIPT":
                    session.records[0] = None
                else:
                    session.finish_error = RuntimeError("PRIVATE finalization")
                cancellation = SystemExit(7)
                session.abort_error = cancellation
                with self.assertRaises(SystemExit) as raised:
                    self.run_session(session)
                self.assertIs(raised.exception, cancellation)
                self.assertEqual([call[0] for call in session.calls],
                                 ["measure", "finish", "abort"] if phase == "FINALIZE"
                                 else ["measure", "abort"])

    def test_later_message_mutation_cannot_change_inventory(self):
        session = FakeSession(self.records, self.final)
        received = []

        def mutate_previous(ordinal, messages):
            if received:
                received[-1][0]["content"] = "PRIVATE late mutation"
            received.append(messages)

        session.hook = mutate_previous
        result = self.run_session(session)
        expected = build_synthetic_report(
            self.inputs, self.profile_raw, self.approved, self.records, self.final)
        self.assertEqual(result, expected)
        self.assertEqual(len({id(messages) for messages in received}), 120)

    def test_result_and_session_receipts_are_detached_both_directions(self):
        session = FakeSession(self.records, self.final)
        result = self.run_session(session)
        frozen_result = copy.deepcopy(result)
        session.records[0]["measurement"]["input"]["input_ids"][0] = 99
        session.finalization["identity"]["settings"].clear()
        self.assertEqual(result, frozen_result)
        frozen_session = copy.deepcopy((session.records, session.finalization))
        result["private"]["records"][1]["measurement"]["input"]["input_ids"][0] = 88
        result["private"]["finalization"]["snapshot"]["file_bytes"].clear()
        self.assertEqual((session.records, session.finalization), frozen_session)

    def test_public_flags_and_privacy_remain_synthetic(self):
        session = FakeSession(self.records, self.final)
        public = self.run_session(session)["public"]
        self.assertTrue(public["synthetic_only"])
        self.assertEqual(public["evidence_kind"], "synthetic")
        for flag in REAL_FLAGS:
            self.assertIs(public[flag], False)
        serialized = encoded(public).decode()
        for private_text in ("private synthetic", "input_ids", "target_ids", "messages",
                             "chat_template", "cleanup"):
            self.assertNotIn(private_text, serialized)

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
                self.assertNotIn(node.func.id, {
                    "open", "exec", "eval", "compile", "__import__", "getattr", "globals",
                })
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, {
                    "open_session", "read_text", "read_bytes", "write_text", "write_bytes",
                    "load", "loads", "from_pretrained", "generate", "train", "system", "popen",
                })


if __name__ == "__main__":
    unittest.main()
