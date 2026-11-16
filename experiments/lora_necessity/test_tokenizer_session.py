import builtins
import copy
import gc
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from test_tokenizer_measurement import MappingTokenizer
from validation_contract import digest, encoded


class SessionFixtureTokenizer(MappingTokenizer):
    def __init__(self):
        self.is_fast = True
        self.eos_token = "[EOS]"
        self.pad_token = "[PAD]"
        self.special_tokens_map = {"eos_token": "[EOS]", "pad_token": "[PAD]"}
        self.all_special_tokens = ["[EOS]", "[PAD]"]
        self.all_special_ids = [2, 0]
        self.backend_payload = '{"synthetic":1}'
        self.backend_tokenizer = SimpleNamespace(to_str=lambda: self.backend_payload)
        self.padding_side = "right"
        self.truncation_side = "right"
        self.model_max_length = 4096
        self.split_special_tokens = False


class TokenizerSessionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("tokenizer_session.py").is_file())
        self.module = importlib.import_module("tokenizer_session")
        self.profile_module = importlib.import_module("tokenizer_runtime_profile")
        self.guard_module = importlib.import_module("tokenizer_runtime_guard")
        self.snapshot_module = importlib.import_module("tokenizer_snapshot")
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name).resolve()
        self.addCleanup(self.cleanup_files)
        source = self.directory / "source"
        source.mkdir()
        payloads = {name: name.encode("ascii") for name in self.profile_module.FILE_PINS}
        pins = {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()}
        for name, raw in payloads.items():
            (source / name).write_bytes(raw)
        self.versions = {"python": "3.12.13", "transformers": "4.57.1",
                         "tokenizers": "fixture", "huggingface-hub": "fixture", "jinja2": "fixture"}
        profile = {
            "kind": "TOKENIZER_RUNTIME_PROFILE", "version": 1, "run_id": "a" * 32,
            "revision": self.profile_module.REVISION, "file_sha256": pins,
            "versions": self.versions,
        }
        self.raw = encoded(profile)
        self.approved = hashlib.sha256(self.raw).hexdigest()
        self.root = self.directory / profile["run_id"]
        for context in (
            patch.object(self.profile_module, "FILE_PINS", pins),
            patch.object(self.profile_module, "runtime_inventory", return_value=self.versions),
            patch.object(self.guard_module, "_USED", False),
            patch.object(sys, "meta_path", list(sys.meta_path)),
            patch.dict(self.guard_module.os.environ),
        ):
            context.start()
            self.addCleanup(context.stop)
        self.snapshot_module.create_snapshot(source, self.directory, self.raw, self.approved)
        self.tokenizer = SessionFixtureTokenizer()
        self.load_calls = []

        def from_pretrained(*arguments, **keywords):
            self.load_calls.append((arguments, keywords))
            return self.tokenizer

        self.library = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained),
            PreTrainedTokenizerFast=SessionFixtureTokenizer)
        context = patch.object(self.module.importlib, "import_module", return_value=self.library)
        self.importer = context.start()
        self.addCleanup(context.stop)
        self.messages = [{"role": "user", "content": "question"}]

    def cleanup_files(self):
        for path in self.directory.rglob("*"):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o700)
        self.temporary.cleanup()

    def open(self):
        session = self.module.open_session(self.root, self.raw, self.approved)
        self.addCleanup(session.abort)
        return session

    def test_session_reuses_vocabulary_digest_without_skipping_observations(self):
        vocabulary = self.tokenizer.get_vocab()
        measurement = sys.modules["tokenizer_measurement"]
        with patch.object(self.tokenizer, "get_vocab", wraps=self.tokenizer.get_vocab) as reads:
            with patch.object(self.module, "digest", wraps=digest) as session_hashes:
                with patch.object(measurement, "digest", wraps=digest) as measurement_hashes:
                    session = self.open()
                    session.measure(self.messages, "answer")
                    session.finish()
        self.assertEqual(reads.call_count, 17)
        self.assertEqual(sum(call.args[0] == vocabulary for call in
                             session_hashes.call_args_list + measurement_hashes.call_args_list), 2)
        self.assertIsNone(session._vocabulary_digest)

    def test_property_mutation_after_vocabulary_validation_invalidates_finish(self):
        class GetterMutation(SessionFixtureTokenizer):
            def __init__(self, value):
                super().__init__()
                self.vocabulary = {**super().get_vocab(), "unused": 1}
                self.reads = 0
                self.value = value

            def get_vocab(self):
                self.reads += 1
                return self.vocabulary

            @property
            def padding_side(self):
                if self.reads == 5:
                    self.vocabulary["unused"] = self.value
                return self._padding_side

            @padding_side.setter
            def padding_side(self, value):
                self._padding_side = value

        for value in (True, 1.0):
            with self.subTest(value=value), patch.object(self.guard_module, "_USED", False):
                self.tokenizer = GetterMutation(value)
                session = self.open()
                with self.assertRaises(self.module.SessionError):
                    session.finish()
                self.assertEqual(session._state, "FAILED")
                self.assertIsNone(session._vocabulary_digest)

    def test_owned_loading_measurement_chain_and_finalization(self):
        session = self.open()
        self.importer.assert_called_once_with("transformers")
        self.assertEqual(self.load_calls, [((str(self.root / "snapshot"),), {
            "use_fast": True, "local_files_only": True, "trust_remote_code": False,
        })])
        record = session.measure(self.messages, "answer")
        chain = record.pop("chain_sha256")
        self.assertEqual(chain, digest({"previous": digest([]), "record": record}))
        self.assertEqual(record["kind"], "TOKENIZER_SESSION_MEASUREMENT_PROVISIONAL")
        self.assertEqual(record["measurement"]["kind"], "TOKENIZER_MEASUREMENT_UNATTESTED")
        self.assertEqual(record["measurement"]["target"]["target_tokens"], 7)
        record["measurement"]["input"]["input_ids"].clear()
        final = session.finish()
        self.assertEqual(final["chain_sha256"], chain)
        self.assertEqual(final["measurements"], 1)
        self.assertIsNone(final["stop_reason"])
        self.assertTrue(final["tokenizer_loaded"])
        for name in (*self.module.FLAGS, "all_possible_second_rounds_validated"):
            self.assertIs(final[name], False)
        with self.assertRaises(self.module.SessionError):
            session.measure(self.messages)
        with self.assertRaises(self.module.SessionError):
            session.finish()

    def test_bad_approval_never_loads_and_consumes_attempt(self):
        with self.assertRaises(self.module.SessionError):
            self.module.open_session(self.root, self.raw, "0" * 64)
        self.importer.assert_not_called()
        with self.assertRaises(self.module.SessionError):
            self.open()
        self.importer.assert_not_called()

    def test_load_failure_is_sanitized_and_no_retry(self):
        self.importer.side_effect = RuntimeError("private fixture text")
        with self.assertRaisesRegex(self.module.SessionError, "^tokenizer session failed$"):
            self.open()
        with self.assertRaises(self.module.SessionError):
            self.open()
        self.assertEqual(self.importer.call_count, 1)

    def test_open_failure_retains_only_fixed_diagnostic_categories(self):
        self.importer.side_effect = RuntimeError("private answer and token ids")
        with self.assertRaises(self.module.SessionError) as caught:
            self.open()
        self.assertEqual(getattr(caught.exception, "open_diagnostic", None), {
            "stage": "TOKENIZER_LOAD", "error_type": "RuntimeError", "guard_failed": False})
        self.assertEqual(str(caught.exception), "tokenizer session failed")
        self.assertIsNone(caught.exception.__cause__)
        self.assertTrue(caught.exception.__suppress_context__)

    def test_custom_exception_type_and_message_are_not_diagnostic_text(self):
        private_error = type("PRIVATE_ANSWER_TYPE", (RuntimeError,), {})
        self.importer.side_effect = private_error("PRIVATE_ANSWER_MESSAGE")
        with self.assertRaises(self.module.SessionError) as caught:
            self.open()
        diagnostic = getattr(caught.exception, "open_diagnostic", None)
        self.assertEqual(diagnostic, {
            "stage": "TOKENIZER_LOAD", "error_type": "OTHER", "guard_failed": False})
        self.assertNotIn("PRIVATE", repr(diagnostic))

    def test_metadata_and_observation_failures_have_distinct_stages(self):
        with self.assertRaises(self.module.SessionError) as caught:
            self.module.open_session(self.root, self.raw, "0" * 64)
        self.assertEqual(getattr(caught.exception, "open_diagnostic", None), {
            "stage": "RUNTIME_METADATA", "error_type": "ValueError", "guard_failed": False})
        with patch.object(self.guard_module, "_USED", False):
            self.tokenizer.is_fast = False
            with self.assertRaises(self.module.SessionError) as caught:
                self.open()
            self.assertEqual(getattr(caught.exception, "open_diagnostic", None), {
                "stage": "OBSERVE", "error_type": "ValueError", "guard_failed": False})

    def test_guard_snapshot_and_final_check_failure_stages(self):
        for stage, operation in (
            ("GUARD_START", patch.object(self.guard_module, "_USED", True)),
            ("SNAPSHOT_VERIFY", patch.object(
                self.module, "verify_snapshot", side_effect=ValueError("PRIVATE"))),
            ("FINAL_CHECK", patch.object(
                self.module._Session, "_check", side_effect=ValueError("PRIVATE"))),
        ):
            with self.subTest(stage=stage), patch.object(self.guard_module, "_USED", False):
                with operation, self.assertRaises(self.module.SessionError) as caught:
                    self.open()
                self.assertEqual(caught.exception.open_diagnostic, {
                    "stage": stage, "error_type": "ValueError", "guard_failed": False})

    def test_environment_failure_is_visible_in_open_diagnostic(self):
        def load_with_environment_drift(name):
            guard = self.module._OWNER[0]()
            self.guard_module.os.environ["HF_HUB_OFFLINE"] = "0"
            guard.check()

        self.importer.side_effect = load_with_environment_drift
        with self.assertRaises(self.module.SessionError) as caught:
            self.open()
        self.assertEqual(caught.exception.open_diagnostic, {
            "stage": "TOKENIZER_LOAD", "error_type": "ValueError", "guard_failed": True})

    def test_loader_uses_normal_imports_and_only_the_local_tokenizer_api(self):
        before_finders = list(sys.meta_path)
        before_discovery = importlib.util.find_spec
        original = self.library.AutoTokenizer.from_pretrained

        def loader(*arguments, **keywords):
            self.assertEqual(sys.meta_path, before_finders)
            self.assertIs(importlib.util.find_spec, before_discovery)
            self.assertEqual(self.guard_module.os.environ["CUDA_VISIBLE_DEVICES"], "")
            return original(*arguments, **keywords)

        self.library.AutoTokenizer.from_pretrained = loader
        try:
            session = self.open()
        except self.module.SessionError as error:
            self.fail(f"normal tokenizer loading failed: {error.open_diagnostic}")
        self.assertEqual(self.load_calls, [((str(self.root / "snapshot"),), {
            "use_fast": True, "local_files_only": True, "trust_remote_code": False,
        })])
        self.importer.assert_called_once_with("transformers")
        session.measure(self.messages)
        self.assertTrue(session.finish()["tokenizer_loaded"])

    def test_slow_mode_and_wrong_special_mapping_rejected(self):
        self.tokenizer.is_fast = False
        with self.assertRaises(ValueError):
            self.module._observe(self.tokenizer)
        self.tokenizer.is_fast = True
        self.tokenizer.eos_token = "[PAD]"
        with self.assertRaises(ValueError):
            self.module._observe(self.tokenizer)
        with self.assertRaises(self.module.SessionError):
            self.open()

    def test_observed_state_drift_invalidates_session(self):
        session = self.open()
        self.tokenizer.backend_payload = '{"synthetic":2}'
        with self.assertRaises(self.module.SessionError):
            session.measure(self.messages)
        self.tokenizer.backend_payload = '{"synthetic":1}'
        with self.assertRaises(self.module.SessionError):
            session.measure(self.messages)
        with self.assertRaises(self.module.SessionError):
            session.finish()

    def test_metadata_drift_prevents_finalization(self):
        session = self.open()
        session.measure(self.messages)
        self.versions["jinja2"] = "changed"
        with self.assertRaises(self.module.SessionError):
            session.finish()
        self.assertIsNone(session._tokenizer)

    def test_snapshot_drift_prevents_finalization(self):
        session = self.open()
        session.measure(self.messages)
        target = self.root / "snapshot" / "config.json"
        target.chmod(0o600)
        target.write_bytes(b"changed")
        target.chmod(0o400)
        with self.assertRaises(self.module.SessionError):
            session.finish()
        self.assertIsNone(session._tokenizer)

    def test_input_limit_stops_measurement_but_finalizes_stop(self):
        session = self.open()
        record = session.measure([{"role": "user", "content": "x" * 4097}])
        self.assertEqual(record["measurement"]["status"], "INPUT_LIMIT")
        final = session.finish()
        self.assertEqual(final["stop_reason"], "INPUT_LIMIT")
        self.assertEqual(final["measurements"], 1)

    def test_output_limit_rejects_next_measurement(self):
        session = self.open()
        record = session.measure(self.messages, "x" * 256)
        self.assertEqual(record["measurement"]["status"], "OUTPUT_LIMIT")
        with self.assertRaises(self.module.SessionError):
            session.measure(self.messages)
        with self.assertRaises(self.module.SessionError):
            session.finish()

    def test_callback_message_mutation_and_reentry_fail_closed(self):
        session = self.open()
        original = SessionFixtureTokenizer.__call__

        def changed(tokenizer, text, **keywords):
            self.messages[0]["content"] = "changed"
            try:
                session.measure([{"role": "user", "content": "nested"}])
            except self.module.SessionError:
                pass
            return original(tokenizer, text, **keywords)

        with patch.object(SessionFixtureTokenizer, "__call__", changed):
            with self.assertRaises(self.module.SessionError):
                session.measure(self.messages)
        with self.assertRaises(self.module.SessionError):
            session.finish()

    def test_abort_and_environment_drift_do_not_finalize(self):
        session = self.open()
        self.guard_module.os.environ["USE_TORCH"] = "1"
        with self.assertRaises(self.module.SessionError):
            session.measure(self.messages)
        session.abort()
        self.assertIsNone(session._tokenizer)
        with self.assertRaises(self.module.SessionError):
            session.finish()

    def test_message_mutation_without_reentry_rejected(self):
        session = self.open()
        original = SessionFixtureTokenizer.__call__

        def changed(tokenizer, text, **keywords):
            self.messages[0]["content"] = "changed"
            return original(tokenizer, text, **keywords)

        with patch.object(SessionFixtureTokenizer, "__call__", changed):
            with self.assertRaises(self.module.SessionError):
                session.measure(self.messages)
        with self.assertRaises(self.module.SessionError):
            session.finish()

    def test_backend_drift_during_encoding_rejected(self):
        session = self.open()
        original = SessionFixtureTokenizer.__call__

        def changed(tokenizer, text, **keywords):
            tokenizer.backend_payload = '{"synthetic":2}'
            return original(tokenizer, text, **keywords)

        with patch.object(SessionFixtureTokenizer, "__call__", changed):
            with self.assertRaises(self.module.SessionError):
                session.measure(self.messages)
        with self.assertRaises(self.module.SessionError):
            session.finish()

    def test_active_abort_releases_instance_without_finalization(self):
        session = self.open()
        session.abort()
        session.abort()
        self.assertIsNone(session._tokenizer)
        with self.assertRaises(self.module.SessionError):
            session.finish()

    def test_caught_guard_failure_in_loader_rejected(self):
        def loader(*arguments, **keywords):
            guard = self.module._OWNER[0]()
            self.guard_module.os.environ["HF_HUB_OFFLINE"] = "0"
            try:
                guard.check()
            except ValueError:
                pass
            self.guard_module.os.environ["HF_HUB_OFFLINE"] = "1"
            return self.tokenizer

        self.library.AutoTokenizer.from_pretrained = loader
        with self.assertRaises(self.module.SessionError):
            self.open()

    def test_wrong_exported_fast_class_rejected(self):
        self.library.PreTrainedTokenizerFast = str
        with self.assertRaises(self.module.SessionError):
            self.open()

    def test_runtime_drift_during_load_rejected(self):
        def loader(*arguments, **keywords):
            self.versions["jinja2"] = "changed"
            return self.tokenizer

        self.library.AutoTokenizer.from_pretrained = loader
        with self.assertRaises(self.module.SessionError):
            self.open()

    def assert_failed(self, session):
        self.assertEqual(session._state, "FAILED")
        self.assertIsNone(session._tokenizer)
        self.assertFalse(session._guard.active)
        self.assertNotIn(session._guard, sys.meta_path)
        with self.assertRaises(self.module.SessionError):
            session.finish()

    def test_caught_reentry_without_mutation_invalidates_measurement(self):
        session = self.open()
        original = SessionFixtureTokenizer.__call__
        caught = []

        def reenter(tokenizer, text, **keywords):
            try:
                session.measure(self.messages)
            except self.module.SessionError:
                caught.append(True)
            return original(tokenizer, text, **keywords)

        with patch.object(SessionFixtureTokenizer, "__call__", reenter):
            with self.assertRaisesRegex(self.module.SessionError, "^tokenizer session failed$"):
                session.measure(self.messages)
        self.assertTrue(caught)
        self.assertEqual(self.messages, [{"role": "user", "content": "question"}])
        self.assert_failed(session)

    def test_caught_finish_reentry_during_observation_invalidates_finalization(self):
        session = self.open()
        caught = []

        def serialize():
            try:
                session.finish()
            except self.module.SessionError:
                caught.append(True)
            return self.tokenizer.backend_payload

        self.tokenizer.backend_tokenizer.to_str = serialize
        with self.assertRaises(self.module.SessionError):
            session.finish()
        self.assertEqual(caught, [True])
        self.assert_failed(session)

    def test_caught_abort_reentry_is_an_error_and_invalidates_measurement(self):
        session = self.open()
        original = SessionFixtureTokenizer.__call__
        caught = []

        def reenter(tokenizer, text, **keywords):
            try:
                session.abort()
            except self.module.SessionError:
                caught.append(True)
            return original(tokenizer, text, **keywords)

        with patch.object(SessionFixtureTokenizer, "__call__", reenter):
            with self.assertRaises(self.module.SessionError):
                session.measure(self.messages)
        self.assertTrue(caught)
        self.assert_failed(session)

    def test_cancellation_during_measurement_invalidates_before_reraise(self):
        for cancellation in (KeyboardInterrupt, SystemExit, GeneratorExit):
            with self.subTest(cancellation=cancellation.__name__):
                with patch.object(self.guard_module, "_USED", False):
                    session = self.open()
                failure = cancellation("synthetic cancellation")
                with patch.object(SessionFixtureTokenizer, "__call__", side_effect=failure):
                    with self.assertRaises(cancellation) as caught:
                        session.measure(self.messages)
                self.assertIs(caught.exception, failure)
                self.assert_failed(session)

    def test_cancellation_during_loading_closes_guard_and_consumes_attempt(self):
        for cancellation in (KeyboardInterrupt, SystemExit, GeneratorExit):
            with self.subTest(cancellation=cancellation.__name__):
                failure = cancellation("synthetic cancellation")
                guards = []

                def cancelled(name):
                    guards.append(self.module._OWNER[0]())
                    raise failure

                self.importer.side_effect = cancelled
                with patch.object(self.guard_module, "_USED", False):
                    with self.assertRaises(cancellation) as caught:
                        self.open()
                    self.assertIs(caught.exception, failure)
                    self.assertTrue(self.guard_module._USED)
                    self.assertEqual(len(guards), 1)
                    self.assertFalse(guards[0].active)
                    with self.assertRaises(self.module.SessionError):
                        self.open()

    def test_cancellation_during_finalization_invalidates_before_reraise(self):
        for cancellation in (KeyboardInterrupt, SystemExit, GeneratorExit):
            with self.subTest(cancellation=cancellation.__name__):
                with patch.object(self.guard_module, "_USED", False):
                    session = self.open()
                failure = cancellation("synthetic cancellation")
                with patch.object(self.profile_module, "runtime_inventory", side_effect=failure):
                    with self.assertRaises(cancellation) as caught:
                        session.finish()
                self.assertIs(caught.exception, failure)
                self.assert_failed(session)

    def test_main_thread_prerequisite_does_not_consume_attempt(self):
        failures = []

        def open_in_thread():
            try:
                self.module.open_session(self.root, self.raw, self.approved)
            except self.module.SessionError as error:
                failures.append(str(error))

        worker = threading.Thread(target=open_in_thread)
        worker.start()
        worker.join()
        self.assertEqual(failures, ["tokenizer session failed"])
        self.assertFalse(self.guard_module._USED)
        self.importer.assert_not_called()
        self.open().finish()

    def test_live_non_main_thread_measurement_invalidates_session(self):
        session = self.open()
        failures = []

        def measure_in_thread():
            try:
                session.measure(self.messages)
            except self.module.SessionError as error:
                failures.append(str(error))

        worker = threading.Thread(target=measure_in_thread)
        worker.start()
        worker.join()
        self.assertEqual(failures, ["tokenizer session failed"])
        self.assert_failed(session)

    def test_multi_record_chain_and_detached_identity(self):
        session = self.open()
        previous = digest([])
        for ordinal in range(3):
            record = session.measure(self.messages, "answer")
            chain = record.pop("chain_sha256")
            self.assertEqual(record["ordinal"], ordinal)
            self.assertEqual(record["run_id"], "a" * 32)
            self.assertEqual(record["profile_sha256"], self.approved)
            self.assertEqual(record["identity_sha256"], digest(self.module._observe(self.tokenizer)))
            self.assertEqual(chain, digest({"previous": previous, "record": record}))
            previous = chain
            record["measurement"]["input"]["input_ids"].clear()
        final = session.finish()
        self.assertEqual(final["kind"], "TOKENIZER_SESSION_FINALIZATION")
        self.assertEqual(final["measurements"], 3)
        self.assertEqual(final["chain_sha256"], previous)
        expected = copy.deepcopy(final["identity"])
        self.tokenizer.special_tokens_map["pad_token"] = "changed"
        self.assertEqual(final["identity"], expected)
        self.assertIsNone(session._tokenizer)
        self.assertNotIn(session._guard, sys.meta_path)

    def test_zero_measurement_finalization_is_not_inventory_acceptance(self):
        session = self.open()
        final = session.finish()
        self.assertEqual(final["measurements"], 0)
        self.assertEqual(final["chain_sha256"], digest([]))
        self.assertNotIn("checked_inventory_within_budget", final)
        self.assertIsNone(final["stop_reason"])
        with self.assertRaises(self.module.SessionError):
            self.open()
        self.assertEqual(self.importer.call_count, 1)

    def test_output_limit_can_finalize_without_another_measurement(self):
        session = self.open()
        record = session.measure(self.messages, "x" * 256)
        final = session.finish()
        self.assertEqual(final["stop_reason"], "OUTPUT_LIMIT")
        self.assertEqual(final["measurements"], 1)
        self.assertEqual(final["chain_sha256"], record["chain_sha256"])

    def test_snapshot_drift_during_load_rejected(self):
        guards = []

        def loader(*arguments, **keywords):
            guards.append(self.module._OWNER[0]())
            target = self.root / "snapshot" / "config.json"
            target.chmod(0o600)
            target.write_bytes(b"changed")
            target.chmod(0o400)
            return self.tokenizer

        self.library.AutoTokenizer.from_pretrained = loader
        with self.assertRaises(self.module.SessionError):
            self.open()
        self.assertEqual(len(guards), 1)
        self.assertFalse(guards[0].active)

    def test_observation_validates_each_identity_field(self):
        changes = (
            ("chat_template", ""), ("get_vocab", lambda: {"[EOS]": True}),
            ("special_tokens_map", {"eos_token": "[EOS]"}),
            ("all_special_tokens", ["[EOS]", "[EOS]"]),
            ("all_special_ids", [True, 0]), ("eos_token_id", True),
            ("pad_token_id", 2), ("padding_side", "middle"),
            ("truncation_side", "middle"), ("model_max_length", True),
            ("split_special_tokens", 0), ("backend_payload", ""),
        )
        for attribute, value in changes:
            with self.subTest(attribute=attribute):
                tokenizer = SessionFixtureTokenizer()
                setattr(tokenizer, attribute, value)
                with self.assertRaises(ValueError):
                    self.module._observe(tokenizer)

    def test_measurement_state_binding_rejects_unmatched_count(self):
        session = self.open()
        original = self.module.measure

        def changed(tokenizer, messages, answer=None):
            result = original(tokenizer, messages, answer)
            result["state_sha256"] = "0" * 64
            return result

        with patch.object(self.module, "measure", side_effect=changed):
            with self.assertRaises(self.module.SessionError):
                session.measure(self.messages)
        self.assert_failed(session)

    def test_import_remains_pure(self):
        with patch.object(self.profile_module, "runtime_inventory",
                          side_effect=AssertionError("metadata discovery forbidden")):
            with patch.object(self.guard_module.RuntimeGuard, "start",
                              side_effect=AssertionError("attempt forbidden")):
                with patch.object(builtins, "open", side_effect=AssertionError("file I/O forbidden")):
                    importlib.reload(self.module)
        self.importer.assert_not_called()
        self.assertFalse(self.guard_module._USED)

    def test_caught_reentry_during_instance_release_prevents_finalization(self):
        session = self.open()
        caught = []

        def on_release():
            try:
                session.finish()
            except self.module.SessionError:
                caught.append(True)

        self.tokenizer.backend_tokenizer.to_str = lambda: '{"synthetic":1}'
        finalizer = weakref.finalize(self.tokenizer, on_release)
        self.tokenizer = None
        with self.assertRaises(self.module.SessionError):
            session.finish()
        self.assertFalse(finalizer.alive)
        self.assertEqual(caught, [True])
        self.assert_failed(session)

    def test_metadata_drift_during_final_observation_prevents_finalization(self):
        session = self.open()

        def serialize():
            self.versions["jinja2"] = "changed"
            return self.tokenizer.backend_payload

        self.tokenizer.backend_tokenizer.to_str = serialize
        with self.assertRaises(self.module.SessionError):
            session.finish()
        self.assert_failed(session)

    def test_template_drift_during_final_observation_prevents_finalization(self):
        session = self.open()
        record = session.measure(self.messages)
        baseline = copy.deepcopy(session._identity)
        callbacks = []

        def serialize():
            callbacks.append(True)
            self.tokenizer.chat_template = baseline["chat_template"] + " changed"
            return self.tokenizer.backend_payload

        self.tokenizer.backend_tokenizer.to_str = serialize
        with self.assertRaisesRegex(self.module.SessionError, "^tokenizer session failed$"):
            session.finish()
        self.assertEqual(callbacks, [True, True])
        self.assertNotEqual(self.tokenizer.chat_template, baseline["chat_template"])
        self.assertEqual(self.tokenizer.padding_side, baseline["settings"]["padding_side"])
        self.assertEqual(hashlib.sha256(self.tokenizer.backend_payload.encode()).hexdigest(),
                         baseline["backend_sha256"])
        self.assertEqual(session._identity, baseline)
        self.assertEqual(session._count, 1)
        self.assertEqual(session._chain, record["chain_sha256"])
        self.assert_failed(session)

    def test_encoding_setting_drift_during_final_observation_prevents_finalization(self):
        session = self.open()
        record = session.measure(self.messages)
        baseline = copy.deepcopy(session._identity)
        callbacks = []

        def serialize():
            callbacks.append(True)
            self.tokenizer.padding_side = "left"
            return self.tokenizer.backend_payload

        self.tokenizer.backend_tokenizer.to_str = serialize
        with self.assertRaisesRegex(self.module.SessionError, "^tokenizer session failed$"):
            session.finish()
        self.assertEqual(callbacks, [True, True])
        self.assertNotEqual(self.tokenizer.padding_side, baseline["settings"]["padding_side"])
        self.assertEqual(self.tokenizer.chat_template, baseline["chat_template"])
        self.assertEqual(hashlib.sha256(self.tokenizer.backend_payload.encode()).hexdigest(),
                         baseline["backend_sha256"])
        self.assertEqual(session._identity, baseline)
        self.assertEqual(session._count, 1)
        self.assertEqual(session._chain, record["chain_sha256"])
        self.assert_failed(session)

    def test_confirmation_observation_caught_guard_failure_prevents_finalization(self):
        session = self.open()
        callbacks = []
        caught = []

        def serialize():
            callbacks.append(True)
            if len(callbacks) == 2:
                self.guard_module.os.environ["HF_HUB_OFFLINE"] = "0"
                try:
                    session._guard.check()
                except ValueError:
                    caught.append(True)
                self.guard_module.os.environ["HF_HUB_OFFLINE"] = "1"
            return self.tokenizer.backend_payload

        self.tokenizer.backend_tokenizer.to_str = serialize
        with self.assertRaises(self.module.SessionError):
            session.finish()
        self.assertEqual(callbacks, [True, True])
        self.assertEqual(caught, [True])
        self.assert_failed(session)

    def test_confirmation_observation_cancellation_invalidates_before_reraise(self):
        for cancellation in (KeyboardInterrupt, SystemExit, GeneratorExit):
            with self.subTest(cancellation=cancellation.__name__):
                self.tokenizer.backend_tokenizer.to_str = lambda: self.tokenizer.backend_payload
                with patch.object(self.guard_module, "_USED", False):
                    session = self.open()
                callbacks = []
                failure = cancellation("synthetic cancellation")

                def serialize():
                    callbacks.append(True)
                    if len(callbacks) == 2:
                        raise failure
                    return self.tokenizer.backend_payload

                self.tokenizer.backend_tokenizer.to_str = serialize
                with self.assertRaises(cancellation) as caught:
                    session.finish()
                self.assertIs(caught.exception, failure)
                self.assertEqual(callbacks, [True, True])
                self.assert_failed(session)

    def test_confirmation_observation_caught_nested_open_prevents_finalization(self):
        session = self.open()
        callbacks = []
        caught = []

        def serialize():
            callbacks.append(True)
            if len(callbacks) == 2:
                self.assert_nested_open_rejected(caught)
            return self.tokenizer.backend_payload

        self.tokenizer.backend_tokenizer.to_str = serialize
        with self.assertRaises(self.module.SessionError):
            session.finish()
        self.assertEqual(callbacks, [True, True])
        self.assertEqual(caught, [True])
        self.assert_failed(session)
        self.assert_attempt_remains_consumed()

    def test_snapshot_drift_during_final_observation_prevents_finalization(self):
        session = self.open()

        def serialize():
            target = self.root / "snapshot" / "config.json"
            target.chmod(0o600)
            target.write_bytes(b"changed")
            target.chmod(0o400)
            return self.tokenizer.backend_payload

        self.tokenizer.backend_tokenizer.to_str = serialize
        with self.assertRaises(self.module.SessionError):
            session.finish()
        self.assert_failed(session)

    def test_metadata_drift_during_instance_release_prevents_finalization(self):
        session = self.open()

        def on_release():
            self.versions["jinja2"] = "changed"

        self.tokenizer.backend_tokenizer.to_str = lambda: '{"synthetic":1}'
        finalizer = weakref.finalize(self.tokenizer, on_release)
        self.tokenizer = None
        with self.assertRaises(self.module.SessionError):
            session.finish()
        self.assertFalse(finalizer.alive)
        self.assert_failed(session)

    def test_message_mutation_during_initial_observation_rejected(self):
        session = self.open()

        def serialize():
            self.messages[0]["content"] = "changed"
            return self.tokenizer.backend_payload

        self.tokenizer.backend_tokenizer.to_str = serialize
        with self.assertRaises(self.module.SessionError):
            session.measure(self.messages)
        self.assert_failed(session)

    def test_post_load_failure_discards_unreturned_session(self):
        sessions = []
        original = self.module._Session

        class RetainedSession(original):
            def __init__(instance, *arguments, **keywords):
                super().__init__(*arguments, **keywords)
                sessions.append(instance)
                self.versions["jinja2"] = "changed"

        with patch.object(self.module, "_Session", RetainedSession):
            with self.assertRaises(self.module.SessionError):
                self.open()
        self.assertEqual(len(sessions), 1)
        self.assert_failed(sessions[0])

    def assert_nested_open_rejected(self, caught):
        with self.assertRaisesRegex(self.module.SessionError, "^tokenizer session failed$"):
            self.open()
        caught.append(True)

    def assert_attempt_remains_consumed(self):
        with self.assertRaises(self.module.SessionError):
            self.open()
        self.importer.assert_called_once_with("transformers")
        self.assertEqual(len(self.load_calls), 1)
        self.assertTrue(self.guard_module._USED)

    def test_caught_nested_open_during_loading_invalidates_outer(self):
        caught = []
        guards = []
        references = []
        original = self.library.AutoTokenizer.from_pretrained

        def loader(*arguments, **keywords):
            guards.append(self.module._OWNER[0]())
            tokenizer = original(*arguments, **keywords)
            tokenizer.backend_tokenizer.to_str = lambda: '{"synthetic":1}'
            references.append(weakref.ref(tokenizer))
            self.tokenizer = None
            self.assert_nested_open_rejected(caught)
            return tokenizer

        self.library.AutoTokenizer.from_pretrained = loader
        with self.assertRaisesRegex(self.module.SessionError, "^tokenizer session failed$"):
            self.open()
        self.assertEqual(caught, [True])
        self.assertFalse(guards[0].active)
        self.assertNotIn(guards[0], sys.meta_path)
        gc.collect()
        self.assertIsNone(references[0]())
        self.assert_attempt_remains_consumed()

    def test_caught_nested_open_during_initial_observation_invalidates_outer(self):
        caught = []
        sessions = []
        original = self.module._Session

        class RetainedSession(original):
            def __init__(instance, *arguments, **keywords):
                super().__init__(*arguments, **keywords)
                sessions.append(instance)

        def serialize():
            self.assert_nested_open_rejected(caught)
            return self.tokenizer.backend_payload

        self.tokenizer.backend_tokenizer.to_str = serialize
        with patch.object(self.module, "_Session", RetainedSession):
            with self.assertRaises(self.module.SessionError):
                self.open()
        self.assertEqual(caught, [True])
        self.assertEqual(len(sessions), 1)
        self.assert_failed(sessions[0])
        self.assert_attempt_remains_consumed()

    def test_caught_nested_open_during_measurement_invalidates_outer(self):
        session = self.open()
        session.measure(self.messages)
        original = SessionFixtureTokenizer.__call__
        caught = []

        def reenter(tokenizer, text, **keywords):
            self.assert_nested_open_rejected(caught)
            return original(tokenizer, text, **keywords)

        with patch.object(SessionFixtureTokenizer, "__call__", reenter):
            with self.assertRaisesRegex(self.module.SessionError, "^tokenizer session failed$"):
                session.measure(self.messages)
        self.assertTrue(caught)
        self.assertEqual(session._count, 1)
        self.assertEqual(self.messages, [{"role": "user", "content": "question"}])
        self.assert_failed(session)
        self.assert_attempt_remains_consumed()

    def test_caught_nested_open_during_final_observation_invalidates_outer(self):
        session = self.open()
        session.measure(self.messages)
        caught = []

        def serialize():
            self.assert_nested_open_rejected(caught)
            return self.tokenizer.backend_payload

        self.tokenizer.backend_tokenizer.to_str = serialize
        with self.assertRaisesRegex(self.module.SessionError, "^tokenizer session failed$"):
            session.finish()
        self.assertEqual(caught, [True])
        self.assert_failed(session)
        self.assert_attempt_remains_consumed()

    def test_caught_nested_open_during_final_release_invalidates_outer(self):
        session = self.open()
        caught = []

        def on_release():
            self.assert_nested_open_rejected(caught)

        self.tokenizer.backend_tokenizer.to_str = lambda: '{"synthetic":1}'
        finalizer = weakref.finalize(self.tokenizer, on_release)
        self.tokenizer = None
        with self.assertRaisesRegex(self.module.SessionError, "^tokenizer session failed$"):
            session.finish()
        self.assertFalse(finalizer.alive)
        self.assertEqual(caught, [True])
        self.assert_failed(session)
        self.assert_attempt_remains_consumed()

    def test_caught_nested_open_between_operations_invalidates_owner(self):
        session = self.open()
        caught = []
        self.assert_nested_open_rejected(caught)
        self.assertEqual(caught, [True])
        self.assert_failed(session)
        self.assert_attempt_remains_consumed()

    def test_caught_nested_open_after_budget_stop_invalidates_owner(self):
        session = self.open()
        session.measure([{"role": "user", "content": "x" * 4097}])
        caught = []
        self.assert_nested_open_rejected(caught)
        self.assertEqual(caught, [True])
        self.assert_failed(session)
        self.assert_attempt_remains_consumed()

    def test_nested_open_owner_does_not_retain_session_or_tokenizer(self):
        for terminal in ("finish", "abort", "failure", "drop"):
            with self.subTest(terminal=terminal):
                self.tokenizer = SessionFixtureTokenizer()
                self.tokenizer.backend_tokenizer.to_str = lambda: '{"synthetic":1}'
                with patch.object(self.guard_module, "_USED", False):
                    session = self.module.open_session(self.root, self.raw, self.approved)
                    guard = session._guard
                    session_reference = weakref.ref(session)
                    tokenizer_reference = weakref.ref(self.tokenizer)
                    self.tokenizer = None
                    try:
                        if terminal == "failure":
                            with self.assertRaises(self.module.SessionError):
                                session.measure([])
                        elif terminal != "drop":
                            getattr(session, terminal)()
                        del session
                        gc.collect()
                        self.assertIsNone(session_reference())
                        self.assertIsNone(tokenizer_reference())
                        with self.assertRaises(self.module.SessionError):
                            self.module.open_session(self.root, self.raw, self.approved)
                    finally:
                        guard.close()
