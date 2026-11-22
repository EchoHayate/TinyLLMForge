import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import generation_core
from pilot_protocol import FLAGS
from validation_contract import digest


class HookHandle:
    def __init__(self, hooks, key):
        self.hooks = hooks
        self.key = key
        self.remove_error = None

    def remove(self):
        self.hooks.pop(self.key, None)
        if self.remove_error is not None:
            raise self.remove_error


class HookModel:
    def __init__(self):
        self.pre_hooks = {}
        self.post_hooks = {}
        self.handles = []
        self.entered = 0
        self.failure = None
        self.fail_post_registration = None
        self.child = None
        self.output = object()

    def register_forward_pre_hook(self, hook):
        return self._register(self.pre_hooks, hook)

    def register_forward_hook(self, hook, *, prepend=False):
        if self.fail_post_registration is not None:
            raise self.fail_post_registration
        return self._register(self.post_hooks, hook, prepend=prepend)

    def _register(self, hooks, hook, *, prepend=False):
        key = len(self.handles)
        prior = dict(hooks)
        if prepend:
            hooks.clear()
        hooks[key] = hook
        if prepend:
            hooks.update(prior)
        handle = HookHandle(hooks, key)
        self.handles.append(handle)
        return handle

    def __call__(self, *arguments):
        for hook in list(self.pre_hooks.values()):
            replacement = hook(self, arguments)
            if replacement is not None:
                raise AssertionError("observer changed forward inputs")
        self.entered += 1
        if self.child is not None:
            self.child()
        if self.failure is not None:
            raise self.failure
        for hook in list(self.post_hooks.values()):
            replacement = hook(self, arguments, self.output)
            if replacement is not None:
                raise AssertionError("observer changed forward output")
        return self.output


class ObservedGenerationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("observed_generation.py").is_file(),
                        "observed generation module absent")
        self.module = importlib.import_module("observed_generation")
        self.model = HookModel()
        self.tokenizer = object()
        self.messages = [{"role": "user", "content": "synthetic question"}]
        self.expected_sha = "a" * 64
        self.call_id = "b" * 32
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.generated = None

    def run_observed(self, **changes):
        arguments = {
            "model": self.model, "tokenizer": self.tokenizer, "messages": self.messages,
            "expected_tokenizer_sha256": self.expected_sha, "call_id": self.call_id,
            "artifact_dir": self.root,
        }
        arguments.update(changes)
        return self.module.generate_observed(**arguments)

    def run_in_subprocess(self):
        import subprocess
        import sys

        if __name__ == "__main__":
            return False
        torch_modules = {name for name in sys.modules if name.split(".")[0] == "torch"}
        selected = f"{type(self).__name__}.{self._testMethodName}"
        result = subprocess.run(
            [sys.executable, "-B", "-W", "error", str(Path(__file__).resolve()), selected],
            capture_output=True, text=True, timeout=90,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            {name for name in sys.modules if name.split(".")[0] == "torch"},
            torch_modules,
        )
        return True

    def generation(self, model, tokenizer, messages, *, expected_tokenizer_sha256):
        self.assertIs(model, self.model)
        self.assertIs(tokenizer, self.tokenizer)
        self.assertEqual(expected_tokenizer_sha256, self.expected_sha)
        for ordinal in range(3):
            self.assertIs(model(ordinal), self.model.output)
        self.generated = {
            "kind": "LORA_GREEDY_CANDIDATE_UNATTESTED", "device_type": "cpu",
            "tokenizer_state_sha256": self.expected_sha,
            "messages_sha256": digest(messages), "prompt_sha256": "c" * 64,
            "input_tokens": 20, "output_tokens": 3, "output_ids": [107, 108, 2],
            "candidate": "ab", "candidate_sha256": hashlib.sha256(b"ab").hexdigest(),
            "stop_reason": "eos", "model_calls": 1, "forward_calls": 3,
            "timings_ns": {"preparation": 1, "generation": 2, "decoding": 1, "wall": 4},
            "peak_allocated_bytes": None, "peak_reserved_bytes": None,
            **{flag: False for flag in FLAGS},
        }
        return self.generated

    def read_record(self, name):
        return json.loads((self.root / name).read_bytes())

    def assert_hooks_removed(self):
        self.assertEqual(self.model.pre_hooks, {})
        self.assertEqual(self.model.post_hooks, {})

    def assert_failed(self, started, completed):
        observation = self.read_record("observation.json")
        self.assertEqual(observation["kind"], "LORA_GENERATION_OBSERVATION")
        self.assertEqual(observation["status"], "failed")
        self.assertEqual(observation["call_id"], self.call_id)
        self.assertEqual(observation["forward_started"], started)
        self.assertEqual(observation["forward_completed"], completed)
        self.assertIs(type(observation["elapsed_ns"]), int)
        self.assertGreaterEqual(observation["elapsed_ns"], 0)
        self.assertNotIn("output_tokens", observation)
        self.assertNotIn("model_calls", observation)
        self.assertIs(type(observation["failure"]["type"]), str)
        self.assertIs(type(observation["failure"]["message"]), str)
        self.assert_hooks_removed()
        return observation

    def test_completed_records_preserve_full_generation_and_claim(self):
        with mock.patch.object(generation_core, "generate_candidate",
                               side_effect=self.generation) as generate:
            result = self.run_observed()
        generate.assert_called_once()
        self.assertEqual(set(result), {"generation", "observation", "record_cleanup"})
        self.assertEqual(result["record_cleanup"], {"failed_count": 0, "failures": []})
        self.assertIs(result["generation"], self.generated)
        self.assertEqual(self.read_record("generation.json"), self.generated)
        observation = self.read_record("observation.json")
        self.assertEqual(result["observation"], observation)
        self.assertEqual(set(observation), {
            "kind", "call_id", "status", "messages_sha256", "forward_started",
            "forward_completed", "elapsed_ns", "failure",
        })
        self.assertEqual(observation["kind"], "LORA_GENERATION_OBSERVATION")
        self.assertEqual(observation["status"], "completed")
        self.assertEqual(observation["forward_started"], 3)
        self.assertEqual(observation["forward_completed"], 3)
        self.assertIsNone(observation["failure"])
        self.assertEqual(self.read_record("claim.json"), {
            "kind": "LORA_GENERATION_CLAIM", "call_id": self.call_id,
            "messages_sha256": digest(self.messages),
            "expected_tokenizer_sha256": self.expected_sha,
        })
        for path in self.root.iterdir():
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(sorted(path.name for path in self.root.iterdir()),
                         ["claim.json", "generation.json", "observation.json"])
        self.assert_hooks_removed()

    def test_claim_is_complete_before_hooks_or_generation(self):
        original = self.model.register_forward_pre_hook

        def install(hook):
            self.assertEqual(self.read_record("claim.json")["call_id"], self.call_id)
            return original(hook)

        with mock.patch.object(self.model, "register_forward_pre_hook", side_effect=install):
            with mock.patch.object(generation_core, "generate_candidate",
                                   side_effect=self.generation):
                self.run_observed()

    def test_root_only_counts_and_existing_hooks_are_preserved(self):
        self.model.child = HookModel()
        events = []
        self.model.register_forward_pre_hook(
            lambda module, arguments: events.append(("pre", arguments)))
        self.model.register_forward_hook(
            lambda module, arguments, output: events.append(("post", output)))
        before_pre = dict(self.model.pre_hooks)
        before_post = dict(self.model.post_hooks)
        with mock.patch.object(generation_core, "generate_candidate",
                               side_effect=self.generation):
            result = self.run_observed()
        self.assertEqual(self.model.pre_hooks, before_pre)
        self.assertEqual(self.model.post_hooks, before_post)
        self.assertEqual(len(events), 6)
        self.assertEqual(self.model.child.entered, 3)
        self.assertEqual(self.model.child.handles, [])
        self.assertEqual(result["observation"]["forward_started"], 3)

    def test_forward_failure_counts_entry_without_completion(self):
        failure = RuntimeError("synthetic root failure")

        def fail(model, tokenizer, messages, **options):
            model()
            model.failure = failure
            model()

        with mock.patch.object(generation_core, "generate_candidate", side_effect=fail) as generate:
            with self.assertRaises(self.module.GenerationAttemptError) as raised:
                self.run_observed()
        generate.assert_called_once()
        self.assertIs(raised.exception.__cause__, failure)
        observation = self.assert_failed(2, 1)
        self.assertEqual(observation["failure"],
                         {"type": "RuntimeError", "message": "synthetic root failure"})
        self.assertFalse((self.root / "generation.json").exists())

    def test_existing_post_hook_failure_preserves_normal_forward_completion(self):
        failure = RuntimeError("existing post hook failed")

        def prior_hook(module, arguments, output):
            raise failure

        self.model.register_forward_hook(prior_hook)
        prior = dict(self.model.post_hooks)
        with mock.patch.object(generation_core, "generate_candidate",
                               side_effect=self.generation) as generate:
            with self.assertRaises(self.module.GenerationAttemptError) as raised:
                self.run_observed()
        generate.assert_called_once()
        self.assertIs(raised.exception.__cause__, failure)
        self.assertEqual(self.model.pre_hooks, {})
        self.assertEqual(self.model.post_hooks, prior)
        observation = self.read_record("observation.json")
        self.assertEqual(observation["status"], "failed")
        self.assertEqual(observation["forward_started"], 1)
        self.assertEqual(observation["forward_completed"], 1)

    def test_existing_pre_hook_failure_precedes_root_entry_observation(self):
        failure = RuntimeError("existing pre hook failed")

        def prior_hook(module, arguments):
            raise failure

        self.model.register_forward_pre_hook(prior_hook)
        prior = dict(self.model.pre_hooks)
        with mock.patch.object(generation_core, "generate_candidate",
                               side_effect=self.generation) as generate:
            with self.assertRaises(self.module.GenerationAttemptError) as raised:
                self.run_observed()
        generate.assert_called_once()
        self.assertIs(raised.exception.__cause__, failure)
        self.assertEqual(self.model.entered, 0)
        self.assertEqual(self.model.pre_hooks, prior)
        self.assertEqual(self.model.post_hooks, {})
        observation = self.read_record("observation.json")
        self.assertEqual(observation["status"], "failed")
        self.assertEqual(observation["forward_started"], 0)
        self.assertEqual(observation["forward_completed"], 0)
        self.assertNotIn("output_tokens", observation)
        self.assertNotIn("model_calls", observation)
        self.assertFalse((self.root / "generation.json").exists())

    def test_post_forward_decode_failure_retains_counts_without_token_claim(self):
        failure = ValueError("synthetic decode failure")

        def fail(model, tokenizer, messages, **options):
            model()
            model()
            raise failure

        with mock.patch.object(generation_core, "generate_candidate", side_effect=fail) as generate:
            with self.assertRaises(self.module.GenerationAttemptError) as raised:
                self.run_observed()
        generate.assert_called_once()
        self.assertIs(raised.exception.__cause__, failure)
        self.assert_failed(2, 2)

    def test_second_hook_install_failure_removes_first_without_generation(self):
        failure = RuntimeError("post hook installation failed")
        self.model.fail_post_registration = failure
        with mock.patch.object(generation_core, "generate_candidate") as generate:
            with self.assertRaises(self.module.GenerationAttemptError) as raised:
                self.run_observed()
        generate.assert_not_called()
        self.assertIs(raised.exception.__cause__, failure)
        self.assertEqual(len(self.model.handles), 1)
        self.assert_failed(0, 0)

    def test_first_hook_install_failure_prevents_generation(self):
        failure = RuntimeError("pre hook installation failed")
        with mock.patch.object(self.model, "register_forward_pre_hook", side_effect=failure):
            with mock.patch.object(generation_core, "generate_candidate") as generate:
                with self.assertRaises(self.module.GenerationAttemptError) as raised:
                    self.run_observed()
        generate.assert_not_called()
        self.assertIs(raised.exception.__cause__, failure)
        self.assert_failed(0, 0)

    def test_cleanup_failure_aborts_and_still_removes_other_hook(self):
        failure = RuntimeError("hook removal failed")

        def generate(*arguments, **options):
            result = self.generation(*arguments, **options)
            self.model.handles[-1].remove_error = failure
            return result

        with mock.patch.object(generation_core, "generate_candidate", side_effect=generate):
            with self.assertRaises(self.module.GenerationAttemptError) as raised:
                self.run_observed()
        self.assertIs(raised.exception.__cause__, failure)
        self.assert_failed(3, 3)

    def test_original_core_error_is_retained_if_cleanup_also_fails(self):
        original = RuntimeError("core failed")
        cleanup = OSError("cleanup failed")

        def fail(*arguments, **options):
            self.model.handles[-1].remove_error = cleanup
            raise original

        with mock.patch.object(generation_core, "generate_candidate", side_effect=fail):
            with self.assertRaises(self.module.GenerationAttemptError) as raised:
                self.run_observed()
        self.assertIs(raised.exception.__cause__, original)
        observation = self.assert_failed(0, 0)
        self.assertEqual(observation["failure"]["cleanup_errors"],
                         [{"type": "OSError", "message": "cleanup failed"}])

    def test_message_snapshot_is_independent_and_original_mutation_is_isolated(self):
        before = copy.deepcopy(self.messages)

        def generate(model, tokenizer, snapshot, **options):
            self.assertIsNot(snapshot, self.messages)
            self.assertIsNot(snapshot[0], self.messages[0])
            self.messages[0]["content"] = "caller changed its copy"
            self.assertEqual(snapshot, before)
            return self.generation(model, tokenizer, snapshot, **options)

        with mock.patch.object(generation_core, "generate_candidate", side_effect=generate):
            result = self.run_observed()
        self.assertEqual(result["observation"]["messages_sha256"], digest(before))

    def test_core_snapshot_mutation_is_binding_failure(self):
        def generate(model, tokenizer, snapshot, **options):
            snapshot[0]["content"] = "core changed its input"
            return self.generation(model, tokenizer, snapshot, **options)

        with mock.patch.object(generation_core, "generate_candidate", side_effect=generate):
            with self.assertRaises(self.module.GenerationAttemptError):
                self.run_observed()
        self.assert_failed(3, 3)
        self.assertEqual(self.messages[0]["content"], "synthetic question")
        self.assertFalse((self.root / "generation.json").exists())

    def test_result_identity_and_exact_integer_counts_are_verified(self):
        changes = (
            ("messages_sha256", "0" * 64), ("tokenizer_state_sha256", "0" * 64),
            ("candidate_sha256", "0" * 64), ("candidate", "altered"),
            ("model_calls", 2), ("model_calls", True), ("forward_calls", 2),
            ("forward_calls", 3.0), ("output_tokens", 2), ("output_tokens", 3.0),
            ("kind", "OTHER"), ("output_ids", [107, 2]),
        )
        for field, value in changes:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as folder:
                self.root = Path(folder)

                def changed(*arguments, **options):
                    result = self.generation(*arguments, **options)
                    return {**result, field: value}

                with mock.patch.object(generation_core, "generate_candidate",
                                       side_effect=changed) as generate:
                    with self.assertRaises(self.module.GenerationAttemptError):
                        self.run_observed()
                generate.assert_called_once()
                self.assert_failed(3, 3)
                self.assertFalse((self.root / "generation.json").exists())

    def test_generator_bypassing_root_forward_cannot_report_success(self):
        with mock.patch.object(generation_core, "generate_candidate",
                               side_effect=self.generation):
            result = self.run_observed()
        with tempfile.TemporaryDirectory() as folder:
            self.root = Path(folder)
            with mock.patch.object(generation_core, "generate_candidate",
                                   return_value=result["generation"]) as generate:
                with self.assertRaises(self.module.GenerationAttemptError):
                    self.run_observed()
            generate.assert_called_once()
            self.assert_failed(0, 0)

    def test_duplicate_claim_does_not_retry_or_replace_records(self):
        with mock.patch.object(generation_core, "generate_candidate",
                               side_effect=self.generation) as generate:
            self.run_observed()
            before = {path.name: path.read_bytes() for path in self.root.iterdir()}
            with self.assertRaises(self.module.GenerationAttemptError):
                self.run_observed(call_id="c" * 32)
        generate.assert_called_once()
        self.assertEqual({path.name: path.read_bytes() for path in self.root.iterdir()}, before)
        self.assertEqual(len(self.model.handles), 2)

    def test_claim_only_incomplete_attempt_is_not_retried(self):
        raw = b'{"kind":"LORA_GENERATION_CLAIM","call_id":"' + self.call_id.encode() + b'"}'
        (self.root / "claim.json").write_bytes(raw)
        with mock.patch.object(generation_core, "generate_candidate") as generate:
            with self.assertRaises(self.module.GenerationAttemptError):
                self.run_observed()
        generate.assert_not_called()
        self.assertEqual((self.root / "claim.json").read_bytes(), raw)
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["claim.json"])
        self.assertEqual(self.model.handles, [])

    def test_invalid_call_id_or_tokenizer_sha_prevents_claim_and_call(self):
        for field, values in (
            ("call_id", (None, True, "", "a" * 31, "a" * 33, "A" * 32, "z" * 32)),
            ("expected_tokenizer_sha256", (None, True, "", "a" * 63, "A" * 64, "z" * 64)),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    with mock.patch.object(generation_core, "generate_candidate") as generate:
                        with self.assertRaises(self.module.GenerationAttemptError):
                            self.run_observed(**{field: value})
                    generate.assert_not_called()
                    self.assertEqual(list(self.root.iterdir()), [])
                    self.assertEqual(self.model.handles, [])

    def test_nonexistent_or_nondirectory_artifact_path_is_not_created(self):
        file_path = self.root / "file"
        file_path.write_text("existing")
        for path in (self.root / "absent", file_path):
            with self.subTest(path=path):
                with mock.patch.object(generation_core, "generate_candidate") as generate:
                    with self.assertRaises(self.module.GenerationAttemptError):
                        self.run_observed(artifact_dir=path)
                generate.assert_not_called()
        self.assertEqual(file_path.read_text(), "existing")
        self.assertFalse((self.root / "absent").exists())

    def test_existing_generation_or_observation_is_never_overwritten(self):
        for name in ("generation.json", "observation.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                self.root = Path(folder)
                (self.root / name).write_bytes(b"existing evidence")
                with mock.patch.object(generation_core, "generate_candidate",
                                       side_effect=self.generation) as generate:
                    with self.assertRaises(self.module.GenerationAttemptError) as raised:
                        self.run_observed()
                self.assertLessEqual(generate.call_count, 1)
                self.assertEqual((self.root / name).read_bytes(), b"existing evidence")
                if name == "observation.json":
                    self.assertIn("observation", str(raised.exception).lower())
                self.assert_hooks_removed()

    def test_record_write_short_write_and_close_failures_preserve_failed_observation(self):
        original_fdopen = os.fdopen
        for position in (1, 2, 3):
            for phase in ("write", "short_write", "close"):
                with self.subTest(position=position, phase=phase):
                    with tempfile.TemporaryDirectory() as folder:
                        self.root = Path(folder)
                        opened_count = 0
                        streams = []
                        failure = OSError("synthetic record failure")

                        def opened(descriptor, mode):
                            nonlocal opened_count
                            opened_count += 1
                            stream = original_fdopen(descriptor, mode)
                            streams.append(stream)
                            if opened_count != position:
                                return stream
                            output = mock.MagicMock(spec=type(stream), wraps=stream)
                            output.__enter__.return_value = output
                            if phase == "write":
                                output.write.side_effect = failure
                            elif phase == "short_write":
                                output.write.return_value = 0

                            def closed(*arguments):
                                stream.close()
                                if phase == "close":
                                    raise failure
                                return False

                            output.__exit__.side_effect = closed
                            return output

                        with mock.patch.object(self.module.os, "fdopen", side_effect=opened):
                            with mock.patch.object(generation_core, "generate_candidate",
                                                   side_effect=self.generation) as generate:
                                with self.assertRaises(self.module.GenerationAttemptError):
                                    self.run_observed()
                        expected = 0 if position == 1 else 3
                        self.assertEqual(generate.call_count, 0 if position == 1 else 1)
                        self.assert_failed(expected, expected)
                        self.assertTrue(all(stream.closed for stream in streams))

    def test_failed_observation_write_is_explicit_and_keeps_original_cause(self):
        failure = RuntimeError("generation failed")
        write_failure = OSError("evidence storage failed")
        original_save = self.module._save

        def save(root, name, record, **options):
            if name == "observation.json":
                raise write_failure
            return original_save(root, name, record, **options)

        with mock.patch.object(self.module, "_save", side_effect=save):
            with mock.patch.object(generation_core, "generate_candidate",
                                   side_effect=failure) as generate:
                with self.assertRaises(self.module.GenerationAttemptError) as raised:
                    self.run_observed()
        generate.assert_called_once()
        self.assertIs(raised.exception.__cause__, failure)
        self.assertIn("failed observation", str(raised.exception).lower())
        self.assertIs(raised.exception.observation_error, write_failure)
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["claim.json"])
        self.assert_hooks_removed()

    def test_postpublication_cleanup_failures_do_not_reverse_completed_attempt(self):
        original_unlink = os.unlink
        for names in (("observation.json",), ("generation.json",),
                      ("generation.json", "observation.json")):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as folder:
                self.root = Path(folder)
                residuals = []

                def unlink(path, *arguments, **options):
                    for name in names:
                        if (Path(path).name.startswith(f".{name}.")
                                and name not in [item[0] for item in residuals]):
                            self.assertTrue((self.root / name).is_file())
                            residuals.append((name, Path(path)))
                            raise OSError("temporary cleanup failed")
                    return original_unlink(path, *arguments, **options)

                with mock.patch.object(self.module.os, "unlink", side_effect=unlink):
                    with mock.patch.object(self.module.os, "link", wraps=os.link) as link:
                        with mock.patch.object(generation_core, "generate_candidate",
                                               side_effect=self.generation) as generate:
                            try:
                                result = self.run_observed()
                            except self.module.GenerationAttemptError as error:
                                self.fail(f"published record cleanup reversed success: {error}")
                generate.assert_called_once()
                self.assertEqual(link.call_count, 2)
                self.assertEqual(result["record_cleanup"], {
                    "failed_count": len(names),
                    "failures": [
                        {"record": name, "temporary": path.name, "published": True,
                         "error": {"type": "OSError", "message": "temporary cleanup failed"}}
                        for name, path in residuals
                    ],
                })
                self.assertEqual(len(residuals), len(names))
                self.assertEqual(result["generation"], self.read_record("generation.json"))
                self.assertEqual(result["observation"], self.read_record("observation.json"))
                self.assertEqual(result["observation"]["status"], "completed")
                self.assertIsNone(result["observation"]["failure"])
                for name, path in residuals:
                    self.assertEqual(path.read_bytes(), (self.root / name).read_bytes())
                    self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                self.assert_hooks_removed()
                before = {path.name: path.read_bytes() for path in self.root.iterdir()}
                with mock.patch.object(generation_core, "generate_candidate") as retry:
                    with self.assertRaises(self.module.GenerationAttemptError):
                        self.run_observed()
                retry.assert_not_called()
                self.assertEqual(before, {
                    path.name: path.read_bytes() for path in self.root.iterdir()})

    def test_failed_observation_cleanup_does_not_report_evidence_missing(self):
        failure = RuntimeError("generation failed")
        original_unlink = os.unlink
        residuals = []

        def unlink(path, *arguments, **options):
            if Path(path).name.startswith(".observation.json."):
                self.assertEqual(self.read_record("observation.json")["status"], "failed")
                residuals.append(Path(path))
                raise OSError("failed evidence cleanup failed")
            return original_unlink(path, *arguments, **options)

        with mock.patch.object(self.module.os, "unlink", side_effect=unlink):
            with mock.patch.object(generation_core, "generate_candidate",
                                   side_effect=failure) as generate:
                with self.assertRaises(self.module.GenerationAttemptError) as raised:
                    self.run_observed()
        generate.assert_called_once()
        self.assertIs(raised.exception.__cause__, failure)
        self.assertIsNone(raised.exception.observation_error)
        self.assertIn("failed observation saved", str(raised.exception))
        self.assertEqual(len(residuals), 1)
        self.assertEqual(raised.exception.record_cleanup, {
            "failed_count": 1,
            "failures": [{
                "record": "observation.json", "temporary": residuals[0].name,
                "published": True,
                "error": {"type": "OSError", "message": "failed evidence cleanup failed"},
            }],
        })
        observation = self.assert_failed(0, 0)
        self.assertEqual(observation["failure"], {
            "type": "RuntimeError", "message": "generation failed"})
        self.assertEqual(json.loads(residuals[0].read_bytes()), observation)
        self.assertFalse((self.root / "generation.json").exists())

    def test_prepublication_link_failure_is_fatal_with_original_cause(self):
        original_link = os.link
        for name in ("generation.json", "observation.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                self.root = Path(folder)
                failure = OSError("record link failed")
                failed = False

                def link(source, destination):
                    nonlocal failed
                    if Path(destination).name == name and not failed:
                        failed = True
                        raise failure
                    return original_link(source, destination)

                with mock.patch.object(self.module.os, "link", side_effect=link):
                    with mock.patch.object(generation_core, "generate_candidate",
                                           side_effect=self.generation) as generate:
                        with self.assertRaises(self.module.GenerationAttemptError) as raised:
                            self.run_observed()
                generate.assert_called_once()
                self.assertTrue(failed)
                self.assertIs(raised.exception.__cause__, failure)
                self.assertIsNone(raised.exception.observation_error)
                self.assertEqual(self.assert_failed(3, 3)["failure"], {
                    "type": "OSError", "message": "record link failed"})
                self.assertEqual(list(self.root.glob(".*")), [])
                self.assertEqual((self.root / "generation.json").exists(),
                                 name == "observation.json")

    def test_primary_write_or_link_failure_survives_temporary_cleanup_failure(self):
        original_write = self.module._write
        original_link = os.link
        original_unlink = os.unlink
        for phase in ("write", "link"):
            for name in ("generation.json", "observation.json"):
                with self.subTest(phase=phase, name=name):
                    with tempfile.TemporaryDirectory() as folder:
                        self.root = Path(folder)
                        failure = OSError(f"primary {phase} failed")
                        failed = False
                        residuals = []

                        def write(descriptor, record):
                            nonlocal failed
                            selected = (record["kind"] == "LORA_GREEDY_CANDIDATE_UNATTESTED"
                                        if name == "generation.json"
                                        else record.get("status") == "completed")
                            if phase == "write" and selected and not failed:
                                failed = True
                                os.close(descriptor)
                                raise failure
                            return original_write(descriptor, record)

                        def link(source, destination):
                            nonlocal failed
                            if phase == "link" and Path(destination).name == name and not failed:
                                failed = True
                                raise failure
                            return original_link(source, destination)

                        def unlink(path, *arguments, **options):
                            if Path(path).name.startswith(f".{name}.") and not residuals:
                                residuals.append(Path(path))
                                raise OSError("secondary unlink failed")
                            return original_unlink(path, *arguments, **options)

                        with mock.patch.object(self.module, "_write", side_effect=write):
                            with mock.patch.object(self.module.os, "link", side_effect=link):
                                with mock.patch.object(self.module.os, "unlink", side_effect=unlink):
                                    with mock.patch.object(generation_core, "generate_candidate",
                                                           side_effect=self.generation) as generate:
                                        with self.assertRaises(self.module.GenerationAttemptError) as raised:
                                            self.run_observed()
                        generate.assert_called_once()
                        self.assertTrue(failed)
                        self.assertIs(raised.exception.__cause__, failure)
                        self.assertIsNone(raised.exception.observation_error)
                        self.assertEqual(self.assert_failed(3, 3)["failure"], {
                            "type": "OSError", "message": f"primary {phase} failed"})
                        self.assertEqual(len(residuals), 1)
                        self.assertTrue(residuals[0].exists())
                        self.assertEqual(raised.exception.record_cleanup, {
                            "failed_count": 1,
                            "failures": [{
                                "record": name, "temporary": residuals[0].name,
                                "published": False,
                                "error": {"type": "OSError", "message": "secondary unlink failed"},
                            }],
                        })

    def test_failed_observation_link_and_cleanup_errors_keep_both_primary_causes(self):
        failure = RuntimeError("generation failed")
        link_failure = OSError("failed observation link failed")
        original_unlink = os.unlink
        residuals = []

        def unlink(path, *arguments, **options):
            if Path(path).name.startswith(".observation.json."):
                residuals.append(Path(path))
                raise OSError("secondary unlink failed")
            return original_unlink(path, *arguments, **options)

        with mock.patch.object(self.module.os, "link", side_effect=link_failure):
            with mock.patch.object(self.module.os, "unlink", side_effect=unlink):
                with mock.patch.object(generation_core, "generate_candidate",
                                       side_effect=failure) as generate:
                    with self.assertRaises(self.module.GenerationAttemptError) as raised:
                        self.run_observed()
        generate.assert_called_once()
        self.assertIs(raised.exception.__cause__, failure)
        self.assertIs(raised.exception.observation_error, link_failure)
        self.assertIn("could not be persisted", str(raised.exception))
        self.assertFalse((self.root / "observation.json").exists())
        self.assertEqual(len(residuals), 1)
        self.assertEqual(raised.exception.record_cleanup, {
            "failed_count": 1,
            "failures": [{
                "record": "observation.json", "temporary": residuals[0].name,
                "published": False,
                "error": {"type": "OSError", "message": "secondary unlink failed"},
            }],
        })
        self.assert_hooks_removed()

    def test_generation_publication_with_cleanup_failure_is_not_attempt_commit(self):
        failure = OSError("completed observation link failed")
        original_link = os.link
        original_unlink = os.unlink
        failed = False
        residuals = []

        def link(source, destination):
            nonlocal failed
            if Path(destination).name == "observation.json" and not failed:
                failed = True
                raise failure
            return original_link(source, destination)

        def unlink(path, *arguments, **options):
            if Path(path).name.startswith(".generation.json."):
                residuals.append(Path(path))
                raise OSError("generation temporary cleanup failed")
            return original_unlink(path, *arguments, **options)

        with mock.patch.object(self.module.os, "link", side_effect=link):
            with mock.patch.object(self.module.os, "unlink", side_effect=unlink):
                with mock.patch.object(generation_core, "generate_candidate",
                                       side_effect=self.generation) as generate:
                    with self.assertRaises(self.module.GenerationAttemptError) as raised:
                        self.run_observed()
        generate.assert_called_once()
        self.assertTrue(failed)
        self.assertIs(raised.exception.__cause__, failure)
        self.assertIsNone(raised.exception.observation_error)
        self.assertEqual(self.assert_failed(3, 3)["failure"], {
            "type": "OSError", "message": "completed observation link failed"})
        self.assertEqual(self.read_record("generation.json"), self.generated)
        self.assertEqual(len(residuals), 1)
        self.assertEqual(raised.exception.record_cleanup, {
            "failed_count": 1,
            "failures": [{
                "record": "generation.json", "temporary": residuals[0].name,
                "published": True,
                "error": {"type": "OSError", "message": "generation temporary cleanup failed"},
            }],
        })

    def test_failure_messages_are_bounded_and_broken_stringification_is_safe(self):
        class BrokenMessage(RuntimeError):
            def __str__(self):
                raise ValueError("cannot format exception")

        long_error = type("E" * 200, (RuntimeError,), {})("message" * 1000)
        for failure in (long_error, BrokenMessage()):
            with tempfile.TemporaryDirectory() as folder:
                self.root = Path(folder)
                with mock.patch.object(generation_core, "generate_candidate",
                                       side_effect=failure):
                    with self.assertRaises(self.module.GenerationAttemptError) as raised:
                        self.run_observed()
                self.assertIs(raised.exception.__cause__, failure)
                observation = self.assert_failed(0, 0)
                self.assertLessEqual(len(observation["failure"]["type"]), 128)
                self.assertLessEqual(len(observation["failure"]["message"]), 1024)

    def test_interrupt_and_system_exit_record_failure_and_remove_hooks(self):
        for failure in (KeyboardInterrupt(), SystemExit(7)):
            with tempfile.TemporaryDirectory() as folder:
                self.root = Path(folder)
                with mock.patch.object(generation_core, "generate_candidate",
                                       side_effect=failure) as generate:
                    with self.assertRaises(self.module.GenerationAttemptError) as raised:
                        self.run_observed()
                generate.assert_called_once()
                self.assertIs(raised.exception.__cause__, failure)
                self.assert_failed(0, 0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires existing PyTorch")
    def test_global_post_hook_failure_can_follow_body_return_without_completion_observation(self):
        if self.run_in_subprocess():
            return

        torch = importlib.import_module("torch")
        failure = RuntimeError("global post hook failed")
        returned = []
        global_calls = []

        class ReturningModel(torch.nn.Module):
            def forward(self):
                returned.append(True)
                return "synthetic output"

        model = ReturningModel()

        def global_hook(module, arguments, output):
            if module is model:
                global_calls.append(output)
                raise failure

        def generate(model, tokenizer, messages, **options):
            model()

        handle = torch.nn.modules.module.register_module_forward_hook(global_hook)
        try:
            with mock.patch.object(generation_core, "generate_candidate",
                                   side_effect=generate) as core:
                with self.assertRaises(self.module.GenerationAttemptError) as raised:
                    self.run_observed(model=model)
            core.assert_called_once()
            self.assertIs(raised.exception.__cause__, failure)
            self.assertEqual(returned, [True])
            self.assertEqual(global_calls, ["synthetic output"])
            self.assert_failed(1, 0)
            self.assertEqual(model._forward_pre_hooks, {})
            self.assertEqual(model._forward_hooks, {})
            self.assertFalse((self.root / "generation.json").exists())
            self.assertIs(torch.nn.modules.module._global_forward_hooks[handle.id], global_hook)
        finally:
            handle.remove()

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires existing PyTorch")
    def test_existing_small_synthetic_model_runs_unchanged_generation_core(self):
        if self.run_in_subprocess():
            return

        from test_generation_core import GenerationTokenizer, scripted_model
        from tokenizer_measurement import measure
        torch = importlib.import_module("torch")
        tokenizer = GenerationTokenizer()
        measured = measure(tokenizer, self.messages)
        model = scripted_model(torch, measured["input"]["input_tokens"], [107, 108, 2])
        parameters = {name: value.clone() for name, value in model.state_dict().items()}
        with mock.patch.object(generation_core, "generate_candidate",
                               wraps=generation_core.generate_candidate) as generate:
            result = self.run_observed(
                model=model, tokenizer=tokenizer,
                expected_tokenizer_sha256=measured["state_sha256"])
        generate.assert_called_once()
        self.assertEqual(result["generation"]["candidate"], "ab")
        self.assertEqual(result["generation"]["output_tokens"], 3)
        self.assertEqual(result["observation"]["forward_started"], 3)
        self.assertEqual(result["observation"]["forward_completed"], 3)
        self.assertEqual(model._forward_pre_hooks, {})
        self.assertEqual(model._forward_hooks, {})
        for name, parameter in model.state_dict().items():
            self.assertTrue(torch.equal(parameter, parameters[name]))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires existing PyTorch")
    def test_existing_synthetic_core_forward_and_decode_failures(self):
        if self.run_in_subprocess():
            return

        from test_generation_core import GenerationTokenizer, scripted_model
        from tokenizer_measurement import measure
        torch = importlib.import_module("torch")
        for stage in ("forward", "decode"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as folder:
                self.root = Path(folder)
                tokenizer = GenerationTokenizer()
                measured = measure(tokenizer, self.messages)
                model = scripted_model(
                    torch, measured["input"]["input_tokens"], [107, 108, 2])
                if stage == "forward":
                    model.mode = "exception"
                    failure = model.error
                else:
                    failure = ValueError("synthetic tokenizer decode failed")

                    def fail_decode():
                        raise failure

                    tokenizer.on_decode = fail_decode
                with mock.patch.object(generation_core, "generate_candidate",
                                       wraps=generation_core.generate_candidate) as generate:
                    with self.assertRaises(self.module.GenerationAttemptError) as raised:
                        self.run_observed(
                            model=model, tokenizer=tokenizer,
                            expected_tokenizer_sha256=measured["state_sha256"])
                generate.assert_called_once()
                self.assertIs(raised.exception.__cause__, failure)
                self.assert_failed(1 if stage == "forward" else 3,
                                   0 if stage == "forward" else 3)
                self.assertEqual(model._forward_pre_hooks, {})
                self.assertEqual(model._forward_hooks, {})


if __name__ == "__main__":
    unittest.main()
