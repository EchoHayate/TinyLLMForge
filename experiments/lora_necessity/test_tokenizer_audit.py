import builtins
import copy
import hashlib
import importlib
import inspect
import io
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pilot_protocol import FLAGS
import test_tokenizer_audit_report as fixtures
import test_tokenizer_session as session_fixtures
import tokenizer_audit_report
import tokenizer_measurement
import tokenizer_runtime_guard
import tokenizer_runtime_profile
import tokenizer_session
import tokenizer_snapshot
from validation_contract import digest, encoded


class SyntheticSession:
    def __init__(self, records, finalization):
        self.records = copy.deepcopy(records)
        self.finalization = copy.deepcopy(finalization)
        self.calls = []
        self.messages = []
        self.measure_error = None
        self.error_ordinal = 0
        self.finish_error = None
        self.abort_error = None
        self.on_measure = None

    def measure(self, messages, answer=None):
        ordinal = len(self.messages)
        self.calls.append(("measure", ordinal))
        self.messages.append((copy.deepcopy(messages), answer))
        if self.measure_error is not None and ordinal == self.error_ordinal:
            raise self.measure_error
        if self.on_measure is not None:
            self.on_measure(ordinal, messages)
        return self.records[ordinal]

    def finish(self):
        self.calls.append(("finish", None))
        if self.finish_error is not None:
            raise self.finish_error
        return self.finalization

    def abort(self):
        self.calls.append(("abort", None))
        if self.abort_error is not None:
            raise self.abort_error


class TokenizerAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = fixtures.frozen_inventory()
        cls.inventory_raw = encoded(cls.inventory)
        cls.inventory_sha256 = digest(cls.inventory)
        cls.profile = fixtures.synthetic_profile()
        cls.profile_raw = encoded(cls.profile)
        cls.profile_sha256 = digest(cls.profile)
        cls.root = Path("/synthetic-only/tokenizer-audit")

    def setUp(self):
        self.assertTrue(Path(__file__).with_name("tokenizer_audit.py").is_file(),
                        "tokenizer audit controller not implemented")
        self.module = importlib.import_module("tokenizer_audit")

    def run_audit(self, **changes):
        arguments = {
            "root": self.root, "profile_raw": self.profile_raw,
            "approved_profile_sha256": self.profile_sha256,
            "inventory_raw": self.inventory_raw,
            "approved_inventory_sha256": self.inventory_sha256,
        }
        arguments.update(changes)
        return self.module.run_audit(**arguments)

    def session(self, count=324, stop=None):
        return SyntheticSession(*fixtures.synthetic_transcript(
            self.inventory, self.profile, count=count, stop=stop))

    def expected(self, records, finalization, **options):
        return tokenizer_audit_report.build_report(
            self.inventory_raw, self.inventory_sha256, self.profile_raw,
            self.profile_sha256, records, finalization, **options)

    def execute(self, session):
        with patch.object(tokenizer_session, "open_session", return_value=session) as opened:
            result = self.run_audit()
        opened.assert_called_once_with(self.root, self.profile_raw, self.profile_sha256)
        return result

    def assert_calls(self, session, count, terminal):
        self.assertEqual(session.calls, [("measure", ordinal) for ordinal in range(count)] +
                         [(operation, None) for operation in terminal])
        self.assertEqual(session.messages, [
            (row["messages"], row["answer"]) for row in self.inventory["rows"][:count]])

    def assert_pending(self, result):
        public = result["public"]
        self.assertEqual(set(result), {"public", "private"})
        self.assertEqual(len(public["rows"]), 324)
        self.assertEqual(public["kind"], "LORA_TOKENIZER_AUDIT_PENDING_CLEANUP")
        self.assertEqual(public["cleanup_status"], "CLEANUP_UNVERIFIED")
        self.assertIs(public["checked_inventory_within_budget"], False)
        for name in (*FLAGS, "all_possible_second_rounds_validated"):
            self.assertIs(public[name], False)
        self.assertEqual(
            {category: counts["scheduled"] for category, counts in public["by_category"].items()},
            {"training": 8, "first": 136, "references": 44, "sampled_second": 136})
        for row in public["rows"]:
            self.assertEqual(set(row), {
                "ordinal", "category", "task_id", "arm", "row_sha256", "status",
                "input_tokens", "target_tokens", "full_sequence_tokens"})
        self.assertNotIn("private synthetic", encoded(public).decode())
        self.assertNotIn("PRIVATE FAILURE", encoded(result).decode())

    def test_controller_exists_and_has_only_owned_interface(self):
        self.assertTrue(callable(self.module.run_audit))
        self.assertEqual(list(inspect.signature(self.module.run_audit).parameters), [
            "root", "profile_raw", "approved_profile_sha256",
            "inventory_raw", "approved_inventory_sha256"])

    def test_open_diagnostic_survives_controller_with_no_private_error_text(self):
        error = tokenizer_session.SessionError("PRIVATE FAILURE")
        error.open_diagnostic = {
            "stage": "TOKENIZER_LOAD", "error_type": "ImportError", "guard_failed": True}
        with patch.object(tokenizer_session, "open_session", side_effect=error):
            result = self.run_audit()
        self.assertEqual(result["public"].get("open_diagnostic"), error.open_diagnostic)
        self.assertEqual(result["public"]["stop_reason"], "OPEN_FAILED")
        self.assert_pending(result)
        self.assertEqual(result["private"]["records"], [])

    def test_full_frozen_success_preserves_every_message_and_answer(self):
        session = self.session()
        result = self.execute(session)
        self.assertEqual(result, self.expected(session.records, session.finalization))
        self.assert_calls(session, 324, ["finish"])
        self.assert_pending(result)
        self.assertIs(result["public"]["all_rows_within_budget"], True)
        self.assertIs(result["public"]["session_finalized"], True)
        self.assertEqual(result["private"]["inventory"], self.inventory)

    def test_invalid_inputs_fail_before_opening_or_runtime_discovery(self):
        for changes in (
            {"approved_inventory_sha256": "0" * 64},
            {"inventory_raw": b"PRIVATE FAILURE"},
            {"approved_profile_sha256": "0" * 64},
            {"profile_raw": b"PRIVATE FAILURE"},
            {"profile_raw": encoded({}), "approved_profile_sha256": digest({})},
        ):
            with self.subTest(changes=tuple(changes)), ExitStack() as stack:
                opened = stack.enter_context(patch.object(tokenizer_session, "open_session"))
                discovered = stack.enter_context(patch.object(
                    tokenizer_runtime_profile, "runtime_inventory",
                    side_effect=AssertionError("discovery forbidden")))
                with self.assertRaisesRegex(ValueError, "^tokenizer audit evidence invalid$"):
                    self.run_audit(**changes)
                opened.assert_not_called()
                discovered.assert_not_called()

    def test_every_category_first_and_last_input_limit(self):
        for ordinal in (0, 7, 8, 143, 144, 187, 188, 323):
            with self.subTest(ordinal=ordinal):
                session = self.session(ordinal + 1, "INPUT_LIMIT")
                result = self.execute(session)
                self.assertEqual(result, self.expected(session.records, session.finalization))
                self.assert_calls(session, ordinal + 1, ["finish"])
                self.assert_pending(result)
                self.assertEqual(result["public"]["stop_ordinal"], ordinal)
                self.assertEqual(result["public"]["stop_reason"], "INPUT_LIMIT")
                self.assertIs(result["public"]["all_rows_within_budget"], False)

    def test_answer_category_first_and_last_output_limit(self):
        for ordinal in (0, 7, 144, 187):
            with self.subTest(ordinal=ordinal):
                session = self.session(ordinal + 1, "OUTPUT_LIMIT")
                result = self.execute(session)
                self.assertEqual(result, self.expected(session.records, session.finalization))
                self.assert_calls(session, ordinal + 1, ["finish"])
                self.assert_pending(result)
                self.assertEqual(result["public"]["stop_reason"], "OUTPUT_LIMIT")
                self.assertEqual(result["public"]["rows"][ordinal]["target_tokens"], 257)

    def test_exact_budgets_are_not_stop_conditions(self):
        session = self.session()
        for ordinal in (0, 7, 8, 143, 144, 187, 188, 323):
            measurement = session.records[ordinal]["measurement"]
            measurement["input"].update(
                input_ids=[11] * 4096, input_tokens=4096, remaining_input_tokens=0)
            target = measurement["target"]
            if target is not None:
                target.update(target_ids=[12] * 255 + [2], target_tokens=256,
                              target_start=4096, full_sequence_tokens=4353,
                              full_sequence_ids=[11] * 4096 + [12] * 255 + [2, 13])
        fixtures.rechain(session.records, session.finalization)
        result = self.execute(session)
        self.assertEqual(result, self.expected(session.records, session.finalization))
        self.assert_calls(session, 324, ["finish"])
        self.assertIs(result["public"]["all_rows_within_budget"], True)

    def test_open_failure_is_private_and_not_retried(self):
        output = io.StringIO()
        with patch.object(tokenizer_session, "open_session",
                          side_effect=RuntimeError("PRIVATE FAILURE")) as opened:
            with redirect_stdout(output), redirect_stderr(output):
                result = self.run_audit()
        opened.assert_called_once_with(self.root, self.profile_raw, self.profile_sha256)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(result, self.expected(
            [], None, failure={"phase": "OPEN", "ordinal": None}))
        self.assert_pending(result)

    def test_measurement_failure_stops_immediately_and_aborts_once(self):
        for ordinal in (0, 8, 144, 188, 323):
            for abort_failed in (False, True):
                with self.subTest(ordinal=ordinal, abort_failed=abort_failed):
                    session = self.session()
                    session.measure_error = RuntimeError("PRIVATE FAILURE")
                    session.error_ordinal = ordinal
                    if abort_failed:
                        session.abort_error = RuntimeError("PRIVATE FAILURE abort")
                    result = self.execute(session)
                    self.assertEqual(result, self.expected(
                        session.records[:ordinal], None,
                        failure={"phase": "MEASURE", "ordinal": ordinal},
                        abort_failed=abort_failed))
                    self.assert_calls(session, ordinal + 1, ["abort"])
                    self.assert_pending(result)

    def test_invalid_receipt_is_never_appended_or_followed_by_measurement(self):
        for path, value in (
            (("kind",), "LORA_REAL_TOKENIZER_AUDIT"),
            (("ordinal",), 1), (("ordinal",), True), (("run_id",), "b" * 32),
            (("profile_sha256",), "0" * 64), (("identity_sha256",), "0" * 64),
            (("measurement", "input", "messages_sha256"), "0" * 64),
            (("measurement", "input", "input_tokens"), True),
            (("measurement", "input", "input_ids"), [-1]),
            (("measurement", "input", "prompt"), "PRIVATE FAILURE"),
            (("measurement", "target", "answer_sha256"), "0" * 64),
            (("measurement", "target", "target_start"), 0),
            (("measurement", "status"), "INPUT_LIMIT"),
            (("measurement", "state_sha256"), "invalid"),
            (("chain_sha256",), "0" * 64),
        ):
            with self.subTest(path=path):
                session = self.session()
                fixtures.replace(session.records[2], path, value)
                if path != ("chain_sha256",):
                    fixtures.rechain(session.records, session.finalization)
                result = self.execute(session)
                self.assertEqual(result, self.expected(
                    session.records[:2], None, failure={"phase": "RECEIPT", "ordinal": 2}))
                self.assert_calls(session, 3, ["abort"])
                self.assert_pending(result)

    def test_changed_neutral_state_stops_before_append_and_next_measure(self):
        for ordinal in (1, 8, 144, 188, 323):
            with self.subTest(ordinal=ordinal):
                session = self.session()
                session.records[ordinal]["measurement"]["state_sha256"] = "0" * 64
                fixtures.rechain(session.records, session.finalization)
                result = self.execute(session)
                self.assertEqual(result, self.expected(
                    session.records[:ordinal], None,
                    failure={"phase": "RECEIPT", "ordinal": ordinal}))
                self.assert_calls(session, ordinal + 1, ["abort"])
                self.assert_pending(result)

    def test_finish_failure_keeps_counts_and_earlier_limit(self):
        for count, stop in ((324, None), (9, "INPUT_LIMIT"), (145, "OUTPUT_LIMIT")):
            for abort_failed in (False, True):
                with self.subTest(count=count, abort_failed=abort_failed):
                    session = self.session(count, stop)
                    session.finish_error = RuntimeError("PRIVATE FAILURE")
                    if abort_failed:
                        session.abort_error = RuntimeError("PRIVATE FAILURE abort")
                    result = self.execute(session)
                    self.assertEqual(result, self.expected(
                        session.records, None, failure={"phase": "FINALIZE", "ordinal": None},
                        abort_failed=abort_failed))
                    self.assert_calls(session, count, ["finish", "abort"])
                    self.assert_pending(result)
                    self.assertIs(result["public"]["all_rows_within_budget"], False)
                    self.assertEqual(result["public"]["stop_reason"], stop or "FINALIZATION_FAILED")

    def test_invalid_finalization_uses_finalize_failure_not_raw_value_error(self):
        for path, value in (
            (("measurements",), 0), (("chain_sha256",), "0" * 64),
            (("run_id",), "b" * 32), (("stop_reason",), "INPUT_LIMIT"),
            (("identity", "measurement_state_sha256"), "0" * 64),
            (("runtime", "versions", "jinja2"), "PRIVATE FAILURE"),
            (("snapshot", "file_sha256", "config.json"), "0" * 64),
            (("benchmark_eligible",), True),
        ):
            with self.subTest(path=path):
                session = self.session()
                fixtures.replace(session.finalization, path, value)
                result = self.execute(session)
                self.assertEqual(result, self.expected(
                    session.records, None, failure={"phase": "FINALIZE", "ordinal": None}))
                self.assert_calls(session, 324, ["finish", "abort"])
                self.assert_pending(result)

    def test_missing_finalization_and_wrong_limit_stop_are_finalize_failures(self):
        for finalization in (None, {}, "PRIVATE FAILURE"):
            with self.subTest(finalization=finalization):
                session = self.session(1, "INPUT_LIMIT")
                session.finalization = finalization
                result = self.execute(session)
                self.assertEqual(result, self.expected(
                    session.records, None, failure={"phase": "FINALIZE", "ordinal": None}))
                self.assert_calls(session, 1, ["finish", "abort"])
        session = self.session(1, "INPUT_LIMIT")
        session.finalization["stop_reason"] = None
        result = self.execute(session)
        self.assertEqual(result, self.expected(
            session.records, None, failure={"phase": "FINALIZE", "ordinal": None}))
        self.assert_calls(session, 1, ["finish", "abort"])

    def test_passed_messages_and_retained_receipts_are_detached(self):
        session = self.session()
        expected = self.expected(session.records, session.finalization)

        def change(ordinal, messages):
            self.assertEqual(messages, self.inventory["rows"][ordinal]["messages"])
            messages[0]["content"] = "PRIVATE FAILURE mutation"
            if ordinal:
                session.records[ordinal - 1]["measurement"]["input"]["input_ids"].clear()

        session.on_measure = change
        result = self.execute(session)
        self.assertEqual(result, expected)
        self.assert_calls(session, 324, ["finish"])
        session.finalization["identity"]["settings"].clear()
        session.records[-1]["measurement"]["input"]["input_ids"].clear()
        self.assertEqual(result, expected)
        result["private"]["inventory"]["rows"][0]["task_id"] = "changed"
        self.assertEqual(result["public"], expected["public"])

    def test_mutated_message_receipt_is_checked_against_original_row(self):
        session = self.session()

        def change(ordinal, messages):
            messages[0]["content"] = "PRIVATE FAILURE mutation"
            session.records[ordinal]["measurement"]["input"]["messages_sha256"] = digest(messages)
            fixtures.rechain(session.records, session.finalization)

        session.on_measure = change
        result = self.execute(session)
        self.assertEqual(result, self.expected(
            [], None, failure={"phase": "RECEIPT", "ordinal": 0}))
        self.assert_calls(session, 1, ["abort"])
        self.assert_pending(result)

    def test_cancellation_preserves_original_exception_and_aborts_at_most_once(self):
        for cancellation in (KeyboardInterrupt, SystemExit, GeneratorExit):
            for phase in ("OPEN", "MEASURE", "RECEIPT", "FINISH", "REPORT"):
                for cleanup_error in (None, RuntimeError("cleanup"), SystemExit("cleanup")):
                    with self.subTest(cancellation=cancellation, phase=phase,
                                      cleanup_error=type(cleanup_error)), ExitStack() as stack:
                        session = self.session(1, "INPUT_LIMIT")
                        failure = cancellation("PRIVATE FAILURE cancellation")
                        session.abort_error = cleanup_error
                        opened = stack.enter_context(patch.object(
                            tokenizer_session, "open_session", return_value=session))
                        if phase == "OPEN":
                            opened.side_effect = failure
                        elif phase == "MEASURE":
                            session.measure_error = failure
                        elif phase == "RECEIPT":
                            stack.enter_context(patch.object(
                                self.module, "validate_record", side_effect=failure))
                        elif phase == "FINISH":
                            session.finish_error = failure
                        else:
                            stack.enter_context(patch.object(
                                self.module, "build_report", side_effect=failure))
                        with self.assertRaises(cancellation) as caught:
                            self.run_audit()
                        self.assertIs(caught.exception, failure)
                        opened.assert_called_once()
                        if phase == "OPEN":
                            self.assertEqual(session.calls, [])
                        else:
                            terminal = ["finish", "abort"] if phase in (
                                "FINISH", "REPORT") else ["abort"]
                            self.assert_calls(session, 1, terminal)

    def test_cancellation_during_abort_is_not_converted_to_a_report(self):
        session = self.session()
        session.measure_error = RuntimeError("PRIVATE FAILURE")
        session.abort_error = KeyboardInterrupt("cancel cleanup")
        with patch.object(tokenizer_session, "open_session", return_value=session):
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.run_audit()
        self.assertIs(caught.exception, session.abort_error)
        self.assert_calls(session, 1, ["abort"])

    def test_import_is_inert_and_does_not_consume_owned_attempt(self):
        used = tokenizer_runtime_guard._USED
        environment = dict(tokenizer_runtime_guard.os.environ)
        with ExitStack() as stack:
            for owner, name in (
                (tokenizer_session, "open_session"),
                (tokenizer_runtime_profile, "runtime_inventory"),
                (tokenizer_snapshot, "verify_snapshot"), (builtins, "open"),
            ):
                stack.enter_context(patch.object(
                    owner, name, side_effect=AssertionError("import side effect forbidden")))
            importlib.reload(self.module)
        self.assertEqual(tokenizer_runtime_guard._USED, used)
        self.assertEqual(dict(tokenizer_runtime_guard.os.environ), environment)

    def test_owned_session_integration_with_synthetic_import_metadata_and_files(self):
        with tempfile.TemporaryDirectory(prefix="task-2-owned-") as temporary, ExitStack() as stack:
            directory = Path(temporary).resolve()
            source = directory / "source"
            source.mkdir()
            payloads = {name: name.encode("ascii") for name in tokenizer_runtime_profile.FILE_PINS}
            pins = {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()}
            for name, raw in payloads.items():
                (source / name).write_bytes(raw)
            profile = fixtures.synthetic_profile()
            profile["file_sha256"] = pins
            raw, approved = encoded(profile), digest(profile)
            root = directory / profile["run_id"]
            stack.enter_context(patch.object(tokenizer_runtime_profile, "FILE_PINS", pins))
            metadata = stack.enter_context(patch.object(
                tokenizer_runtime_profile, "runtime_inventory", return_value=profile["versions"]))
            stack.enter_context(patch.object(tokenizer_runtime_guard, "_USED", False))
            stack.enter_context(patch.object(tokenizer_session, "_OWNER", None))
            stack.enter_context(patch.object(sys, "meta_path", list(sys.meta_path)))
            environment = {}
            stack.enter_context(patch.object(
                tokenizer_runtime_guard, "os", SimpleNamespace(environ=environment)))
            snapshot = tokenizer_snapshot.create_snapshot(source, directory, raw, approved)
            stack.callback((root / "snapshot").chmod, 0o700)
            tokenizer = session_fixtures.SessionFixtureTokenizer()
            loader = Mock(return_value=tokenizer)
            library = SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=loader),
                PreTrainedTokenizerFast=session_fixtures.SessionFixtureTokenizer)
            importer = Mock(return_value=library)
            stack.enter_context(patch.object(
                tokenizer_session, "importlib", SimpleNamespace(import_module=importer)))
            identity = tokenizer_session._observe(tokenizer)
            records = []
            stop = None
            for row in self.inventory["rows"]:
                measured = tokenizer_measurement.measure(
                    tokenizer, row["messages"], row["answer"])
                records.append({
                    "kind": "TOKENIZER_SESSION_MEASUREMENT_PROVISIONAL",
                    "run_id": profile["run_id"], "profile_sha256": approved,
                    "identity_sha256": digest(identity), "ordinal": row["ordinal"],
                    "measurement": measured,
                })
                if measured["status"] != "OK":
                    stop = measured["status"]
                    break
            fixtures.rechain(records)
            finalization = {
                "kind": "TOKENIZER_SESSION_FINALIZATION", "run_id": profile["run_id"],
                "profile_sha256": approved, "identity": identity, "snapshot": snapshot,
                "runtime": tokenizer_runtime_profile.inspect_runtime(raw, approved),
                "measurements": len(records), "chain_sha256": records[-1]["chain_sha256"],
                "stop_reason": stop, "tokenizer_loaded": True,
                **{name: False for name in (*FLAGS, "all_possible_second_rounds_validated")},
            }
            expected = tokenizer_audit_report.build_report(
                self.inventory_raw, self.inventory_sha256, raw, approved, records, finalization)
            calls = []
            original_measure = tokenizer_session._Session.measure
            original_finish = tokenizer_session._Session.finish

            def measured_owned(owner, messages, answer=None):
                calls.append(("measure", copy.deepcopy(messages), answer))
                return original_measure(owner, messages, answer)

            def finished_owned(owner):
                calls.append(("finish", None, None))
                return original_finish(owner)

            stack.enter_context(patch.object(tokenizer_session._Session, "measure", measured_owned))
            stack.enter_context(patch.object(tokenizer_session._Session, "finish", finished_owned))
            result = self.run_audit(root=root, profile_raw=raw, approved_profile_sha256=approved)
            self.assertEqual(result, expected)
            self.assert_pending(result)
            self.assertEqual(calls, [
                ("measure", row["messages"], row["answer"])
                for row in self.inventory["rows"][:len(records)]] + [("finish", None, None)])
            importer.assert_called_once_with("transformers")
            loader.assert_called_once_with(
                str(root / "snapshot"), use_fast=True, local_files_only=True,
                trust_remote_code=False)
            self.assertGreater(metadata.call_count, 1)
            self.assertTrue(tokenizer_runtime_guard._USED)
            self.assertEqual(environment, tokenizer_runtime_guard.ENVIRONMENT)
            self.assertFalse(any(isinstance(finder, tokenizer_runtime_guard.RuntimeGuard)
                                 for finder in sys.meta_path))
            with self.assertRaises(tokenizer_session.SessionError):
                tokenizer_session.open_session(root, raw, approved)
            importer.assert_called_once()
