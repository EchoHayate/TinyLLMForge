import copy
import hashlib
import importlib
import importlib.util
import io
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pilot_protocol import FLAGS
from test_training_worker import synthetic_dataset, tiny_model


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires existing PyTorch")
class AdapterArtifactTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("adapter_artifact.py").is_file(),
                        "adapter artifact module/API absent")
        self.artifact = importlib.import_module("adapter_artifact")
        for name in ("save_adapter", "load_adapter", "mounted_adapter"):
            self.assertTrue(callable(getattr(self.artifact, name, None)),
                            f"adapter artifact API absent: {name}")
        self.torch = importlib.import_module("torch")
        self.worker = importlib.import_module("training_worker")
        self.model = tiny_model(self.torch)
        self.model.register_buffer("rotary", self.torch.ones(2), persistent=False)
        self.result = self.worker.train_adapter(self.model, synthetic_dataset())
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "adapter.pt"
        tokens = self.torch.tensor([[3, 4, 5, 2]])
        self.inputs = {
            "input_ids": tokens, "attention_mask": self.torch.ones_like(tokens),
            "use_cache": False,
        }

    def snapshot(self):
        return {
            "state": {name: value.clone() for name, value in self.worker._state(
                self.model).items()},
            "modules": dict(self.model.named_modules()),
            "flags": {name: value.requires_grad
                      for name, value in self.model.named_parameters()},
            "modes": {name: value.training for name, value in self.model.named_modules()},
            "rng": self.torch.random.get_rng_state().clone(),
            "grad_enabled": self.torch.is_grad_enabled(),
        }

    def assert_restored(self, before, check_bytes=True):
        self.assertEqual(before["modules"], dict(self.model.named_modules()))
        self.assertEqual(before["flags"], {
            name: value.requires_grad for name, value in self.model.named_parameters()})
        self.assertEqual(before["modes"], {
            name: value.training for name, value in self.model.named_modules()})
        self.assertTrue(self.torch.equal(before["rng"], self.torch.random.get_rng_state()))
        self.assertEqual(before["grad_enabled"], self.torch.is_grad_enabled())
        self.assertTrue(all(value.grad is None for value in self.model.parameters()))
        if check_bytes:
            current = self.worker._state(self.model)
            self.assertEqual(set(before["state"]), set(current))
            for name, value in before["state"].items():
                self.assertTrue(self.torch.equal(value, current[name]), name)

    def write_envelope(self, result=None, **changes):
        envelope = {
            "kind": "LORA_ADAPTER_ARTIFACT_UNATTESTED", "version": 1,
            "result": self.result if result is None else result,
        }
        envelope.update(changes)
        stream = io.BytesIO()
        self.torch.save(envelope, stream)
        raw = stream.getvalue()
        self.path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def rebind(self, result):
        for state in ("initial_adapter", "adapter"):
            result["receipt"][f"{state}_sha256"] = self.worker._tensor_digest(result[state])
        result["receipt"]["adapter_bytes"] = sum(
            value.numel() * value.element_size() for value in result["adapter"].values())

    def assert_invalid_result(self, result):
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.artifact.save_adapter(self.path, result)
        self.assertFalse(self.path.exists())
        expected = self.write_envelope(result)
        with self.torch.sparse.check_sparse_tensor_invariants(), self.assertRaises(ValueError):
            self.artifact.load_adapter(self.path, expected_sha256=expected)
        self.path.unlink()
        with self.assertRaises(ValueError):
            with self.artifact.mounted_adapter(self.model, result):
                self.fail("invalid result mounted")
        self.assert_restored(before)

    def test_real_training_round_trip_and_caller_restoration(self):
        self.model.model.eval()
        self.model.embedding.weight.requires_grad_(False)
        before = self.snapshot()
        original = copy.deepcopy(self.result)
        with self.torch.no_grad():
            base = self.model(**self.inputs).logits.clone()
        with self.artifact.mounted_adapter(self.model, self.result) as active:
            in_memory = active(**self.inputs).logits.clone()
            self.assertIs(active, self.model)
            self.assertFalse(self.torch.is_grad_enabled())
            self.assertTrue(all(not module.training for module in active.modules()))
            self.assertTrue(all(not value.requires_grad for value in active.parameters()))
            self.torch.rand(5)
        self.assert_restored(before)
        saved = self.artifact.save_adapter(self.path, self.result)
        self.assertEqual(set(saved), {"sha256", "file_bytes", "save_ns"})
        self.assertEqual(saved["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(saved["file_bytes"], self.path.stat().st_size)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertGreater(saved["save_ns"], 0)
        loaded = self.artifact.load_adapter(self.path, expected_sha256=saved["sha256"])
        self.assertEqual(set(loaded), {"result", "load_ns"})
        self.assertGreater(loaded["load_ns"], 0)
        self.assertEqual(loaded["result"]["receipt"], original["receipt"])
        self.assertEqual(len(loaded["result"]["receipt"]["steps"]), 16)
        for flag in FLAGS:
            self.assertIs(loaded["result"]["receipt"][flag], False)
        for state in ("initial_adapter", "adapter"):
            with self.artifact.mounted_adapter(self.model, loaded["result"], state=state) as active:
                for path, wrapper in ((path, active.get_submodule(path))
                                      for path in original["receipt"]["target_paths"]):
                    self.assertIs(type(wrapper), self.worker.LoRAProjection)
                    for suffix in ("adapter_a", "adapter_b"):
                        self.assertTrue(self.torch.equal(
                            getattr(wrapper, suffix), original[state][f"{path}.{suffix}"]))
                actual = active(**self.inputs).logits
                self.assertTrue(self.torch.equal(
                    actual, base if state == "initial_adapter" else in_memory))
            self.assert_restored(before)
            for name, value in original[state].items():
                self.assertTrue(self.torch.equal(self.result[state][name], value))
                self.assertTrue(self.torch.equal(loaded["result"][state][name], value))
        with self.torch.no_grad():
            self.assertTrue(self.torch.equal(self.model(**self.inputs).logits, base))

    def test_exclusive_save_refuses_file_and_symlink(self):
        self.path.write_bytes(b"existing evidence")
        with self.assertRaises(FileExistsError):
            self.artifact.save_adapter(self.path, self.result)
        self.assertEqual(self.path.read_bytes(), b"existing evidence")
        link = self.path.with_name("link.pt")
        link.symlink_to(self.path)
        with self.assertRaises(FileExistsError):
            self.artifact.save_adapter(link, self.result)
        self.assertEqual(self.path.read_bytes(), b"existing evidence")

    def test_partial_write_remains_and_is_not_retried(self):
        original_write = os.write

        def short_write(descriptor, raw):
            return original_write(descriptor, raw[:17])

        with patch.object(self.artifact.os, "write", side_effect=short_write) as writer:
            with self.assertRaises(OSError):
                self.artifact.save_adapter(self.path, self.result)
        self.assertEqual(writer.call_count, 1)
        self.assertEqual(self.path.stat().st_size, 17)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        with self.assertRaises(FileExistsError):
            self.artifact.save_adapter(self.path, self.result)

    def test_write_and_sync_errors_preserve_evidence_without_retry(self):
        for operation in ("write", "fsync"):
            with self.subTest(operation=operation):
                error = OSError("synthetic I/O failure")
                with patch.object(self.artifact.os, operation, side_effect=error) as failing:
                    with self.assertRaises(OSError) as caught:
                        self.artifact.save_adapter(self.path, self.result)
                    self.assertIs(caught.exception, error)
                    self.assertEqual(failing.call_count, 1)
                self.assertTrue(self.path.exists())
                if operation == "write":
                    self.assertEqual(self.path.stat().st_size, 0)
                else:
                    self.assertGreater(self.path.stat().st_size, 0)
                with self.assertRaises(FileExistsError):
                    self.artifact.save_adapter(self.path, self.result)
                self.path.unlink()

    def test_save_serializes_clones_without_changing_receipt(self):
        original_save = self.torch.save

        def inspect_copy(envelope, stream):
            self.assertIsNot(envelope["result"]["receipt"], self.result["receipt"])
            for state in ("initial_adapter", "adapter"):
                for name, value in self.result[state].items():
                    self.assertNotEqual(
                        value.data_ptr(), envelope["result"][state][name].data_ptr())
            return original_save(envelope, stream)

        with patch.object(self.torch, "save", side_effect=inspect_copy):
            self.artifact.save_adapter(self.path, self.result)
        self.assertIsNone(self.result["receipt"]["adapter_save_ns"])

    def test_digest_required_checked_before_deserialization(self):
        saved = self.artifact.save_adapter(self.path, self.result)
        with patch.object(self.torch, "load") as loader:
            with self.assertRaises(TypeError):
                self.artifact.load_adapter(self.path)
            for expected in (None, "", "A" * 64, "g" * 64, "0" * 64, 123):
                with self.subTest(expected=expected), self.assertRaises(ValueError):
                    self.artifact.load_adapter(self.path, expected_sha256=expected)
            self.path.write_bytes(self.path.read_bytes() + b"tampered")
            with self.assertRaises(ValueError):
                self.artifact.load_adapter(self.path, expected_sha256=saved["sha256"])
            loader.assert_not_called()

    def test_load_uses_checked_bytes_cpu_weights_only_without_fallback(self):
        saved = self.artifact.save_adapter(self.path, self.result)
        original_load = self.torch.load
        raw = self.path.read_bytes()

        def inspect_load(stream, **kwargs):
            self.assertIsInstance(stream, io.BytesIO)
            self.assertEqual(stream.getvalue(), raw)
            self.assertEqual(kwargs, {"map_location": "cpu", "weights_only": True})
            self.path.write_bytes(b"changed after read")
            return original_load(stream, **kwargs)

        with patch.object(self.torch, "load", side_effect=inspect_load):
            self.artifact.load_adapter(self.path, expected_sha256=saved["sha256"])
        self.path.write_bytes(raw)
        error = RuntimeError("weights-only rejected")
        with patch.object(self.torch, "load", side_effect=error) as loader:
            with self.assertRaises(RuntimeError) as caught:
                self.artifact.load_adapter(self.path, expected_sha256=saved["sha256"])
            self.assertIs(caught.exception, error)
            self.assertEqual(loader.call_count, 1)

    def test_exact_envelope(self):
        for changes in ({"kind": "other"}, {"version": True}, {"version": 2}, {"extra": 1}):
            with self.subTest(changes=changes):
                expected = self.write_envelope(**changes)
                with self.assertRaises(ValueError):
                    self.artifact.load_adapter(self.path, expected_sha256=expected)

    def test_result_fields_and_receipt_bindings(self):
        for change in ("extra", "missing", "kind", "config", "base", "dataset",
                       "tokenizer", "initial_digest", "final_digest", "bytes", "bool_bytes"):
            with self.subTest(change=change):
                result = copy.deepcopy(self.result)
                receipt = result["receipt"]
                if change == "extra":
                    result["extra"] = None
                elif change == "missing":
                    del result["initial_adapter"]
                elif change == "kind":
                    receipt["kind"] = "attested"
                elif change == "config":
                    receipt["configuration_sha256"] = "0" * 64
                elif change in ("base", "dataset"):
                    receipt[f"{change}_sha256"] = "invalid"
                elif change == "tokenizer":
                    receipt["tokenizer_state_sha256"] = "invalid"
                elif change in ("initial_digest", "final_digest"):
                    receipt["initial_adapter_sha256" if change == "initial_digest"
                            else "adapter_sha256"] = "0" * 64
                else:
                    receipt["adapter_bytes"] = True if change == "bool_bytes" else 4
                self.assert_invalid_result(result)

    def test_receipt_flags_and_json_native_finite_metadata(self):
        for flag in FLAGS:
            for invalid in (True, 0, None):
                with self.subTest(flag=flag, invalid=invalid):
                    result = copy.deepcopy(self.result)
                    result["receipt"][flag] = invalid
                    self.assert_invalid_result(result)
        for invalid in (float("nan"), float("inf"), (1, 2), {1: "value"}, self.torch.ones(1)):
            with self.subTest(metadata=repr(invalid)):
                result = copy.deepcopy(self.result)
                result["receipt"]["extra_metadata"] = invalid
                self.assert_invalid_result(result)

    def test_consecutive_complete_target_paths(self):
        paths = self.result["receipt"]["target_paths"]
        for invalid in ([], paths[:1], paths[::-1], paths + paths,
                        [name.replace(".0.", ".1.") for name in paths]):
            with self.subTest(paths=invalid):
                result = copy.deepcopy(self.result)
                result["receipt"]["target_paths"] = invalid
                self.assert_invalid_result(result)

    def test_tensor_keys_shapes_dtypes_and_finiteness(self):
        for change in ("missing", "extra", "rank", "zero_dimension", "dtype", "device",
                       "nan", "inf", "sparse", "requires_grad", "gradient"):
            with self.subTest(change=change):
                result = copy.deepcopy(self.result)
                name = next(iter(result["adapter"]))
                value = result["adapter"][name]
                if change == "missing":
                    del result["adapter"][name]
                elif change == "extra":
                    result["adapter"]["unexpected"] = value.clone()
                elif change == "rank":
                    result["adapter"][name] = value[:7].clone()
                elif change == "zero_dimension":
                    result["adapter"][name] = value[:, :0].clone()
                elif change == "dtype":
                    result["adapter"][name] = value.bfloat16()
                elif change == "device":
                    result["adapter"][name] = value.to("meta")
                elif change in ("nan", "inf"):
                    value[0, 0] = float(change)
                elif change == "sparse":
                    result["adapter"][name] = value.to_sparse()
                elif change == "requires_grad":
                    value.requires_grad_(True)
                else:
                    value.grad = self.torch.ones_like(value)
                    with self.assertRaises(ValueError):
                        self.artifact.save_adapter(self.path, result)
                    with self.assertRaises(ValueError):
                        with self.artifact.mounted_adapter(self.model, result):
                            self.fail("gradient-bearing tensor mounted")
                    continue
                self.assert_invalid_result(result)

    def test_rehashed_invalid_shapes_initial_b_and_unchanged_final(self):
        for change in ("rank", "mismatch", "initial_b", "unchanged"):
            with self.subTest(change=change):
                result = copy.deepcopy(self.result)
                name = next(iter(result["adapter"]))
                if change == "rank":
                    for state in ("initial_adapter", "adapter"):
                        result[state][name] = result[state][name][:7].clone()
                elif change == "mismatch":
                    result["adapter"][name] = self.torch.zeros(8, 5)
                elif change == "initial_b":
                    name = name.replace("adapter_a", "adapter_b")
                    result["initial_adapter"][name][0, 0] = 1
                else:
                    result["adapter"] = copy.deepcopy(result["initial_adapter"])
                self.rebind(result)
                self.assert_invalid_result(result)

    def test_mount_rejects_validly_rehashed_wrong_projection_dimensions(self):
        result = copy.deepcopy(self.result)
        name = next(iter(result["adapter"]))
        for state in ("initial_adapter", "adapter"):
            result[state][name] = self.torch.zeros(8, 5)
        self.rebind(result)
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "shape|dimension"):
            with self.artifact.mounted_adapter(self.model, result):
                self.fail("wrong projection dimensions mounted")
        self.assert_restored(before)

    def test_invalid_selection_and_base_identity(self):
        before = self.snapshot()
        for state in ("base", "", None):
            with self.subTest(state=state), self.assertRaises(ValueError):
                with self.artifact.mounted_adapter(self.model, self.result, state=state):
                    self.fail("invalid state mounted")
        result = copy.deepcopy(self.result)
        result["receipt"]["base_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "base"):
            with self.artifact.mounted_adapter(self.model, result):
                self.fail("wrong base mounted")
        self.assert_restored(before)

    def test_mount_rejects_invalid_base_before_changing_modules(self):
        for change in ("float32", "gradient", "nan", "buffer_dtype", "shared", "layers"):
            with self.subTest(change=change):
                self.model = tiny_model(self.torch)
                if change == "float32":
                    self.model.float()
                elif change == "gradient":
                    self.model.embedding.weight.grad = self.torch.ones_like(
                        self.model.embedding.weight)
                elif change == "nan":
                    with self.torch.no_grad():
                        self.model.embedding.weight[0, 0] = float("nan")
                elif change == "buffer_dtype":
                    self.model.register_buffer("bad", self.torch.ones(2).double())
                elif change == "shared":
                    attention = self.model.model.layers[0].self_attn
                    attention.v_proj = attention.q_proj
                else:
                    self.model.config.num_hidden_layers = 2
                before = self.snapshot()
                result = copy.deepcopy(self.result)
                result["receipt"]["base_sha256"] = self.worker._tensor_digest(
                    self.worker._state(self.model))
                with self.assertRaises(ValueError):
                    with self.artifact.mounted_adapter(self.model, result):
                        self.fail("invalid base mounted")
                self.assertEqual(before["modules"], dict(self.model.named_modules()))
                self.assertEqual(before["flags"], {
                    name: value.requires_grad for name, value in self.model.named_parameters()})
                if change == "gradient":
                    self.assertTrue(self.torch.equal(
                        self.model.embedding.weight.grad,
                        self.torch.ones_like(self.model.embedding.weight)))
                self.assertEqual(self.model.calls, [])

    def test_autocast_and_nested_mount_rejected(self):
        before = self.snapshot()
        with self.torch.autocast("cpu", dtype=self.torch.bfloat16):
            with self.assertRaisesRegex(ValueError, "autocast"):
                with self.artifact.mounted_adapter(self.model, self.result):
                    self.fail("autocast mounted")
        with self.artifact.mounted_adapter(self.model, self.result):
            with self.assertRaises(ValueError):
                with self.artifact.mounted_adapter(self.model, self.result):
                    self.fail("nested mount accepted")
        self.assert_restored(before)

    def test_body_exception_identity_and_caller_context_restored(self):
        self.model.model.eval()
        self.model.embedding.weight.requires_grad_(False)
        for context in (self.torch.enable_grad, self.torch.no_grad, self.torch.inference_mode):
            with self.subTest(context=context.__name__), context():
                before = self.snapshot()
                error = RuntimeError("body failure")
                with self.assertRaises(RuntimeError) as caught:
                    with self.artifact.mounted_adapter(self.model, self.result):
                        self.torch.rand(3)
                        raise error
                self.assertIs(caught.exception, error)
                self.assert_restored(before)

    def test_partial_install_failure_restores_modules_flags_modes_rng(self):
        original = self.artifact.LoRAProjection
        calls = []
        error = RuntimeError("second projection failed")

        def fail_second(base):
            calls.append(base)
            wrapper = original(base)
            if len(calls) == 2:
                raise error
            return wrapper

        before = self.snapshot()
        with patch.object(self.artifact, "LoRAProjection", side_effect=fail_second):
            with self.assertRaises(RuntimeError) as caught:
                with self.artifact.mounted_adapter(self.model, self.result):
                    self.fail("partial installation yielded")
        self.assertIs(caught.exception, error)
        self.assertEqual(len(calls), 2)
        self.assert_restored(before)

    def test_inference_base_state_rejected_before_mutation(self):
        source = self.model
        for target in ("parameters", "buffer"):
            for context in (self.torch.no_grad, self.torch.inference_mode):
                for fail in (False, True):
                    with self.subTest(target=target, context=context.__name__, fail=fail):
                        with self.torch.inference_mode(target == "parameters"):
                            self.model = tiny_model(self.torch)
                            self.model.load_state_dict(source.state_dict())
                        with self.torch.inference_mode(target == "buffer"):
                            self.model.register_buffer(
                                "rotary", source.rotary.clone(), persistent=False)
                        self.model.model.eval()
                        before = self.snapshot()
                        entered = False
                        with patch.object(self.model, "requires_grad_",
                                          wraps=self.model.requires_grad_) as freeze, \
                                patch.object(self.model, "eval", wraps=self.model.eval) as evaluate, \
                                patch.object(self.artifact, "LoRAProjection",
                                             wraps=self.artifact.LoRAProjection) as wrap:
                            with context(), self.assertRaises(Exception) as caught:
                                with self.artifact.mounted_adapter(self.model, self.result):
                                    entered = True
                                    if fail:
                                        raise RuntimeError("body failure")
                            self.assertIsInstance(caught.exception, ValueError)
                            self.assertIn("inference", str(caught.exception))
                            freeze.assert_not_called()
                            evaluate.assert_not_called()
                            wrap.assert_not_called()
                        self.assertFalse(entered)
                        self.assert_restored(before)

    def test_ordinary_base_under_inference_mode_restores_after_both_body_outcomes(self):
        self.model.model.eval()
        self.model.lm_head.bias.requires_grad_(False)
        self.assertTrue(all(not value.is_inference()
                            for value in self.worker._state(self.model).values()))
        for fail in (False, True):
            with self.subTest(fail=fail):
                before = self.snapshot()
                error = RuntimeError("body failure")
                caught = None
                try:
                    with self.torch.inference_mode(), self.artifact.mounted_adapter(
                            self.model, self.result) as active:
                        self.assertTrue(self.torch.is_inference_mode_enabled())
                        self.assertFalse(self.torch.is_grad_enabled())
                        self.assertTrue(all(not module.training for module in active.modules()))
                        self.assertTrue(all(not value.requires_grad for value in active.parameters()))
                        active(**self.inputs)
                        self.torch.rand(3)
                        if fail:
                            raise error
                except BaseException as observed:
                    caught = observed
                self.assertIs(caught, error if fail else None)
                self.assert_restored(before)

    def test_dtype_mutation_restores_unaffected_state_and_chains_discard(self):
        source = self.model
        original_state = self.artifact._state
        for fail in (False, True):
            for unreadable in (False, True):
                with self.subTest(fail=fail, unreadable=unreadable):
                    self.model = copy.deepcopy(source)
                    self.model.model.eval()
                    self.model.lm_head.bias.requires_grad_(False)
                    before = self.snapshot()
                    error = RuntimeError("body failure")
                    entered = False

                    def read_state(model):
                        if entered and unreadable:
                            raise RuntimeError("integrity read failed")
                        return original_state(model)

                    with patch.object(self.artifact, "_state", side_effect=read_state) as reader:
                        with self.assertRaises(Exception) as caught:
                            with self.torch.no_grad(), self.artifact.mounted_adapter(
                                    self.model, self.result):
                                entered = True
                                self.model.embedding.weight.data = (
                                    self.model.embedding.weight.to(self.torch.int32))
                                self.torch.rand(3)
                                if fail:
                                    raise error
                    self.assertIsInstance(caught.exception, ValueError)
                    self.assertIn("discard model", str(caught.exception))
                    self.assertTrue(entered)
                    self.assertEqual(reader.call_count, 2)
                    self.assertIn("only Tensors of floating point dtype", str(caught.exception))
                    if fail:
                        self.assertIs(caught.exception.__cause__, error)
                    else:
                        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
                        self.assertIn("only Tensors of floating point dtype",
                                      str(caught.exception.__cause__))
                    self.assertIn("unverifiable" if unreadable else "changed",
                                  str(caught.exception))
                    if unreadable:
                        self.assertIn("integrity read failed", str(caught.exception))
                    self.assertEqual(self.model.embedding.weight.dtype, self.torch.int32)
                    self.assertTrue(self.torch.equal(
                        self.model.embedding.weight,
                        before["state"]["parameter:embedding.weight"].to(self.torch.int32)))
                    self.assertFalse(self.model.embedding.weight.requires_grad)
                    before["flags"]["embedding.weight"] = False
                    self.assert_restored(before, check_bytes=False)
                    for name, value in self.worker._state(self.model).items():
                        if name != "parameter:embedding.weight":
                            self.assertTrue(self.torch.equal(value, before["state"][name]), name)

    def test_restoration_collects_module_and_mode_errors_and_continues(self):
        source = self.model
        original_setattr = setattr
        for fail in (False, True):
            with self.subTest(fail=fail):
                self.model = copy.deepcopy(source)
                self.model.model.eval()
                before = self.snapshot()
                attention = self.model.model.layers[0].self_attn
                original_projection = attention.v_proj
                module_error = RuntimeError("projection restoration failed")
                mode_error = RuntimeError("root mode restoration failed")
                body_error = RuntimeError("body failure")
                entered = False

                def fail_restoration(owner, attribute, value):
                    if entered:
                        if owner is attention and attribute == "v_proj" and value is original_projection:
                            raise module_error
                        if owner is self.model and attribute == "training":
                            raise mode_error
                    original_setattr(owner, attribute, value)

                with patch.object(self.artifact, "setattr", side_effect=fail_restoration, create=True), \
                        patch.object(self.artifact, "_state", wraps=self.artifact._state) as reader:
                    with self.assertRaises(Exception) as caught:
                        with self.artifact.mounted_adapter(self.model, self.result):
                            entered = True
                            self.torch.rand(3)
                            if fail:
                                raise body_error
                self.assertIsInstance(caught.exception, ValueError)
                self.assertIn("discard model", str(caught.exception))
                self.assertIs(caught.exception.__cause__, body_error if fail else module_error)
                self.assertIn(str(module_error), str(caught.exception))
                self.assertIn(str(mode_error), str(caught.exception))
                self.assertEqual(reader.call_count, 2)
                self.assertIs(attention.q_proj, before["modules"]["model.layers.0.self_attn.q_proj"])
                self.assertIsNot(attention.v_proj, original_projection)
                for name, module in before["modules"].items():
                    for parameter_name, parameter in module.named_parameters(recurse=False):
                        self.assertEqual(parameter.requires_grad,
                                         before["flags"][f"{name}.{parameter_name}"])
                    if module is not self.model:
                        self.assertEqual(module.training, before["modes"][name])
                self.assertTrue(self.torch.equal(before["rng"], self.torch.random.get_rng_state()))

    def test_adapter_mutation_detected_without_mutating_result(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                before = self.snapshot()
                original = copy.deepcopy(self.result)
                error = RuntimeError("body failure")
                with self.assertRaisesRegex(ValueError, "adapter.*changed") as caught:
                    with self.artifact.mounted_adapter(self.model, self.result):
                        self.model.model.layers[0].self_attn.q_proj.adapter_b.add_(1)
                        if fail:
                            raise error
                if fail:
                    self.assertIs(caught.exception.__cause__, error)
                self.assert_restored(before)
                for state in ("initial_adapter", "adapter"):
                    for name, value in original[state].items():
                        self.assertTrue(self.torch.equal(self.result[state][name], value))

    def test_adapter_structure_or_gradient_mutation_restores_base(self):
        for change in ("replace", "base", "gradient", "requires_grad", "nonfinite"):
            with self.subTest(change=change):
                before = self.snapshot()
                with self.assertRaisesRegex(ValueError, "adapter.*changed"):
                    with self.artifact.mounted_adapter(self.model, self.result):
                        attention = self.model.model.layers[0].self_attn
                        wrapper = attention.q_proj
                        if change == "replace":
                            attention.q_proj = copy.deepcopy(wrapper)
                        elif change == "base":
                            wrapper.base = copy.deepcopy(wrapper.base)
                        elif change == "gradient":
                            wrapper.adapter_a.grad = self.torch.ones_like(wrapper.adapter_a)
                        elif change == "requires_grad":
                            wrapper.adapter_a.requires_grad_(True)
                        else:
                            wrapper.adapter_a[0, 0] = float("nan")
                self.assert_restored(before)

    def test_base_gradient_mutation_is_detected_and_cleared(self):
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "base.*changed.*discard"):
            with self.artifact.mounted_adapter(self.model, self.result):
                self.model.embedding.weight.grad = self.torch.ones_like(
                    self.model.embedding.weight)
        self.assert_restored(before)

    def test_base_mutation_requires_discard_and_preserves_body_cause(self):
        for target in ("parameter", "buffer"):
            for fail in (False, True):
                with self.subTest(target=target, fail=fail):
                    before = self.snapshot()
                    error = RuntimeError("body failure")
                    with self.assertRaisesRegex(ValueError, "base.*changed.*discard") as caught:
                        with self.artifact.mounted_adapter(self.model, self.result):
                            if target == "parameter":
                                self.model.model.layers[0].self_attn.q_proj.base.weight.add_(1)
                            else:
                                self.model.rotary.add_(1)
                            if fail:
                                raise error
                    if fail:
                        self.assertIs(caught.exception.__cause__, error)
                    self.assert_restored(before, check_bytes=False)
                    self.model = tiny_model(self.torch)
                    self.model.register_buffer("rotary", self.torch.ones(2), persistent=False)
                    self.result = self.worker.train_adapter(self.model, synthetic_dataset())

    def test_unverifiable_base_restores_state_and_requires_discard(self):
        original = self.artifact._state
        for fail in (False, True):
            with self.subTest(fail=fail):
                before = self.snapshot()
                entered = False
                error = RuntimeError("body failure")
                integrity_error = RuntimeError("integrity read failed")

                def unreadable(model):
                    if entered:
                        self.torch.rand(1)
                        raise integrity_error
                    return original(model)

                with patch.object(self.artifact, "_state", side_effect=unreadable):
                    with self.assertRaisesRegex(
                            ValueError, "base.*unverifiable.*discard") as caught:
                        with self.artifact.mounted_adapter(self.model, self.result):
                            entered = True
                            if fail:
                                raise error
                self.assertIs(caught.exception.__cause__, error if fail else integrity_error)
                self.assert_restored(before)


def load_tests(loader, tests, pattern):
    if __name__ == "__main__" or importlib.util.find_spec("torch") is None:
        return tests

    class IsolatedAdapterCase(unittest.TestCase):
        def __init__(self, case_id):
            super().__init__()
            self.case_id = case_id

        def id(self):
            return self.case_id

        def __str__(self):
            return self.case_id

        def runTest(self):
            selected = ".".join(self.case_id.split(".")[-2:])
            result = subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve()), selected],
                capture_output=True, text=True, timeout=90,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def isolated_cases(suite):
        for case in suite:
            if isinstance(case, unittest.TestSuite):
                yield from isolated_cases(case)
            else:
                yield IsolatedAdapterCase(case.id())

    return unittest.TestSuite(isolated_cases(tests))


if __name__ == "__main__":
    unittest.main()
