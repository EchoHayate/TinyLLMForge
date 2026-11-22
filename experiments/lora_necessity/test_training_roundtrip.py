import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pilot_protocol import FLAGS
from test_training_worker import synthetic_dataset, tiny_model
from validation_contract import digest


class RoundtripImportTests(unittest.TestCase):
    def test_imports_do_not_load_torch(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c",
             "import sys; import training_roundtrip; import test_training_roundtrip; "
             "assert 'torch' not in sys.modules; "
             "assert 'training_worker' not in sys.modules; "
             "assert 'adapter_artifact' not in sys.modules"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires existing PyTorch")
class TrainingRoundtripTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("training_roundtrip.py").is_file(),
                        "training roundtrip module absent")
        self.roundtrip = importlib.import_module("training_roundtrip")
        self.torch = importlib.import_module("torch")
        self.worker = importlib.import_module("training_worker")
        self.artifact = importlib.import_module("adapter_artifact")
        self.new_case()

    def new_case(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "adapter.pt"
        self.model = tiny_model(self.torch).eval().requires_grad_(False)
        self.model.register_buffer("rotary", self.torch.ones(2), persistent=False)
        self.dataset = synthetic_dataset()
        for index, row in enumerate(self.dataset["rows"]):
            row["input_ids"][1] = 4 + index
        self.dataset["dataset_sha256"] = digest({
            name: value for name, value in self.dataset.items()
            if name != "dataset_sha256"})

    def snapshot(self):
        return {
            "state": {name: value.clone() for name, value in
                      self.worker._state(self.model).items()},
            "objects": self.worker._state(self.model),
            "modules": dict(self.model.named_modules()),
            "modes": {name: module.training for name, module in self.model.named_modules()},
            "flags": {name: value.requires_grad
                      for name, value in self.model.named_parameters()},
            "rng": self.torch.random.get_rng_state().clone(),
            "grad_enabled": self.torch.is_grad_enabled(),
            "dataset": copy.deepcopy(self.dataset),
            "config": copy.deepcopy(vars(self.model.config)),
        }

    def assert_restored(self, before):
        current = self.worker._state(self.model)
        self.assertEqual(set(current), set(before["state"]))
        for name, value in current.items():
            self.assertIs(value, before["objects"][name])
            self.assertTrue(self.torch.equal(value, before["state"][name]), name)
        self.assertEqual(dict(self.model.named_modules()), before["modules"])
        self.assertEqual({name: module.training for name, module in
                          self.model.named_modules()}, before["modes"])
        self.assertEqual({name: value.requires_grad for name, value in
                          self.model.named_parameters()}, before["flags"])
        self.assertTrue(all(value.grad is None for value in self.model.parameters()))
        self.assertTrue(self.torch.equal(self.torch.random.get_rng_state(), before["rng"]))
        self.assertEqual(self.torch.is_grad_enabled(), before["grad_enabled"])
        self.assertEqual(self.dataset, before["dataset"])
        self.assertEqual(vars(self.model.config), before["config"])

    def test_ordered_real_training_probes_receipt_costs_and_emission(self):
        before = self.snapshot()
        original_forward = self.model.forward
        original_train = self.worker.train_adapter
        captured = {}
        forwards = []
        messages = []

        def observe_forward(**kwargs):
            forwards.append({
                "inputs": kwargs["input_ids"].tolist(),
                "mask": kwargs["attention_mask"].tolist(),
                "cache": kwargs["use_cache"],
                "return_dict": kwargs["return_dict"],
                "grad": self.torch.is_grad_enabled(),
                "frozen": all(not value.requires_grad for value in self.model.parameters()),
            })
            return original_forward(**kwargs)

        def observe_train(model, dataset):
            result = original_train(model, dataset)
            captured["receipt"] = result["receipt"]
            captured["snapshot"] = copy.deepcopy(result["receipt"])
            return result

        def emit(message):
            self.assertIs(type(message), dict)
            self.assertEqual(json.loads(json.dumps(message, allow_nan=False)), message)
            self.assertLessEqual(set(message), {"phase", "event", "status"})
            messages.append((message, len(self.model.calls)))

        with patch.object(self.model, "forward", side_effect=observe_forward), \
                patch.object(self.worker, "train_adapter", side_effect=observe_train), \
                patch.object(self.torch, "equal", wraps=self.torch.equal) as equals:
            result = self.roundtrip.run_roundtrip(
                self.model, self.dataset, self.path, emit=emit)
        comparisons = equals.call_args_list
        self.assertEqual(len(comparisons), 3)
        for comparison in comparisons:
            for logits in comparison.args:
                self.assertEqual(logits.device.type, "cpu")
                self.assertEqual(tuple(logits.shape), (1, 16))
                self.assertFalse(logits.requires_grad)
                self.assertIsNone(logits._base)
        self.assertEqual(set(result), {
            "training_receipt", "checks", "artifact", "timings_ns",
            "peak_allocated_bytes", "peak_reserved_bytes", *FLAGS})
        self.assertIs(result["training_receipt"], captured["receipt"])
        self.assertEqual(result["training_receipt"], captured["snapshot"])
        self.assertEqual(result["checks"], {
            "initial_equals_base": True, "reload_equals_final": True,
            "restored_equals_base": True})
        self.assertTrue(all(result[flag] is False for flag in FLAGS))
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        self.assertIsNone(result["peak_allocated_bytes"])
        self.assertIsNone(result["peak_reserved_bytes"])
        self.assertEqual(result["artifact"]["sha256"],
                         hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(result["artifact"]["file_bytes"], self.path.stat().st_size)
        self.assertEqual(result["training_receipt"]["steps"][-1]["step"], 16)
        self.assertEqual([step["task_id"] for step in result["training_receipt"]["steps"]],
                         [row["task_id"] for row in self.dataset["rows"]] * 2)
        expected_inputs = ([[[3, 4]]] +
                           [[row["input_ids"]] for row in self.dataset["rows"]] * 2 +
                           [[[3, 4]]] * 4)
        self.assertEqual([call["inputs"] for call in forwards], expected_inputs)
        self.assertEqual([call["grad"] for call in forwards], [False] + [True] * 16 + [False] * 4)
        self.assertEqual([call["frozen"] for call in forwards], [True] + [False] * 16 + [True] * 4)
        for call in forwards:
            self.assertIs(call["cache"], False)
            self.assertIs(call["return_dict"], True)
            self.assertEqual(call["mask"], [[1] * len(call["inputs"][0])])
        phases = [
            "base_probe", "training", "initial_adapter_mount", "initial_adapter_probe",
            "final_adapter_mount", "final_adapter_probe", "save_adapter", "load_adapter",
            "reloaded_adapter_mount", "reloaded_adapter_probe", "restored_probe",
        ]
        self.assertEqual([message["phase"] for message, _ in messages
                          if message["event"] == "start"], phases)
        stack = []
        for message, _ in messages:
            if message["event"] == "start":
                stack.append(message["phase"])
            else:
                self.assertEqual(message["phase"], stack.pop())
                self.assertEqual(message["status"], "ok")
        self.assertEqual(stack, [])
        self.assertEqual([(message["event"], count) for message, count in messages
                          if message["phase"] == "training"], [("start", 1), ("end", 17)])
        timings = result["timings_ns"]
        self.assertEqual(set(timings), {
            "base_probe", "training", "initial_adapter_mount", "final_adapter_mount",
            "save_adapter", "load_adapter", "reloaded_adapter_mount", "restored_probe",
            "roundtrip_wall"})
        self.assertTrue(all(type(value) is int and value > 0 for value in timings.values()))
        self.assertLessEqual(sum(value for name, value in timings.items()
                                 if name != "roundtrip_wall"), timings["roundtrip_wall"])
        self.assertGreaterEqual(timings["training"], result["training_receipt"]["core_wall_ns"])
        self.assert_restored(before)

    def test_existing_artifact_and_dangling_symlink_refused_before_forward(self):
        for symlink in (False, True):
            with self.subTest(symlink=symlink):
                self.new_case()
                if symlink:
                    self.path.symlink_to(self.path.parent / "missing.pt")
                else:
                    self.path.write_bytes(b"preserve existing artifact")
                before = self.snapshot()
                with self.assertRaises(FileExistsError):
                    self.roundtrip.run_roundtrip(self.model, self.dataset, self.path)
                self.assertEqual(self.model.calls, [])
                if symlink:
                    self.assertTrue(self.path.is_symlink())
                else:
                    self.assertEqual(self.path.read_bytes(), b"preserve existing artifact")
                self.assert_restored(before)

    def test_malformed_dataset_rejected_before_any_forward(self):
        self.dataset["rows"][-1]["labels"][0] = 3
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.roundtrip.run_roundtrip(self.model, self.dataset, self.path)
        self.assertEqual(self.model.calls, [])
        self.assertFalse(self.path.exists())
        self.assert_restored(before)

    def test_caller_must_be_eval_and_frozen_without_mutation(self):
        for invalid in ("root_training", "child_training", "trainable"):
            with self.subTest(invalid=invalid):
                self.new_case()
                if invalid == "root_training":
                    self.model.train()
                elif invalid == "child_training":
                    self.model.embedding.train()
                else:
                    self.model.embedding.weight.requires_grad_(True)
                before = self.snapshot()
                with self.assertRaisesRegex(ValueError, "eval|frozen"):
                    self.roundtrip.run_roundtrip(self.model, self.dataset, self.path)
                self.assertEqual(self.model.calls, [])
                self.assertFalse(self.path.exists())
                self.assert_restored(before)

    def test_inference_no_grad_and_autocast_rejected_before_forward(self):
        contexts = (
            self.torch.inference_mode, self.torch.no_grad,
            lambda: self.torch.autocast("cpu", dtype=self.torch.bfloat16),
        )
        for context in contexts:
            with self.subTest(context=context):
                self.new_case()
                before = self.snapshot()
                with context(), self.assertRaisesRegex(ValueError, "autograd|inference|autocast"):
                    self.roundtrip.run_roundtrip(self.model, self.dataset, self.path)
                self.assertEqual(self.model.calls, [])
                self.assertFalse(self.path.exists())
                self.assert_restored(before)

    def test_forward_failures_propagate_and_restore_at_every_stage(self):
        for fail_at in (1, 2, 17, 18, 19, 20, 21):
            with self.subTest(fail_at=fail_at):
                self.new_case()
                self.model.fail_at = fail_at
                before = self.snapshot()
                messages = []
                with self.assertRaises(RuntimeError) as caught:
                    self.roundtrip.run_roundtrip(
                        self.model, self.dataset, self.path, emit=messages.append)
                self.assertIs(caught.exception, self.model.forward_error)
                self.assertEqual(len(self.model.calls), fail_at)
                self.assertEqual(self.path.exists(), fail_at >= 20)
                self.assertEqual(messages[-1]["event"], "end")
                self.assertEqual(messages[-1]["status"], "error")
                self.assert_restored(before)

    def test_nonfinite_logits_at_each_probe_fail_closed(self):
        for fail_at in (1, 18, 19, 20, 21):
            with self.subTest(fail_at=fail_at):
                self.new_case()
                original = self.model.forward
                before = self.snapshot()

                def nonfinite(**kwargs):
                    output = original(**kwargs)
                    if len(self.model.calls) == fail_at:
                        output.logits[:, -1, 0] = float("nan")
                    return output

                with patch.object(self.model, "forward", side_effect=nonfinite):
                    with self.assertRaisesRegex(ValueError, "nonfinite"):
                        self.roundtrip.run_roundtrip(self.model, self.dataset, self.path)
                self.assertEqual(len(self.model.calls), fail_at)
                self.assertEqual(self.path.exists(), fail_at >= 20)
                self.assert_restored(before)

    def test_each_equality_check_is_fail_closed(self):
        for fail_at, check in ((18, "initial_equals_base"), (20, "reload_equals_final"),
                               (21, "restored_equals_base")):
            with self.subTest(check=check):
                self.new_case()
                original = self.model.forward
                before = self.snapshot()

                def different(**kwargs):
                    output = original(**kwargs)
                    if len(self.model.calls) == fail_at:
                        output.logits[:, -1, :] += 1
                    return output

                with patch.object(self.model, "forward", side_effect=different):
                    with self.assertRaisesRegex(ValueError, check):
                        self.roundtrip.run_roundtrip(self.model, self.dataset, self.path)
                self.assertEqual(len(self.model.calls), fail_at)
                self.assertEqual(self.path.exists(), fail_at >= 20)
                self.assert_restored(before)

    def test_corrupted_saved_artifact_is_preserved_and_rejected(self):
        original = self.artifact.save_adapter
        before = self.snapshot()

        def corrupt(path, result):
            saved = original(path, result)
            raw = path.read_bytes()
            path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            return saved

        with patch.object(self.artifact, "save_adapter", side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, "file digest mismatch"):
                self.roundtrip.run_roundtrip(self.model, self.dataset, self.path)
        self.assertEqual(len(self.model.calls), 19)
        self.assertTrue(self.path.is_file())
        self.assert_restored(before)

    def test_save_failure_propagates_without_removing_evidence(self):
        before = self.snapshot()
        error = OSError("synthetic incomplete save")

        def fail_save(path, result):
            path.write_bytes(b"partial artifact evidence")
            raise error

        with patch.object(self.artifact, "save_adapter", side_effect=fail_save):
            with self.assertRaises(OSError) as caught:
                self.roundtrip.run_roundtrip(self.model, self.dataset, self.path)
        self.assertIs(caught.exception, error)
        self.assertEqual(self.path.read_bytes(), b"partial artifact evidence")
        self.assertEqual(len(self.model.calls), 19)
        self.assert_restored(before)

    def test_emit_failure_inside_mount_restores_model(self):
        before = self.snapshot()
        error = RuntimeError("synthetic emission failure")

        def emit(message):
            if message == {"phase": "initial_adapter_probe", "event": "start"}:
                raise error

        with self.assertRaises(RuntimeError) as caught:
            self.roundtrip.run_roundtrip(self.model, self.dataset, self.path, emit=emit)
        self.assertIs(caught.exception, error)
        self.assertEqual(len(self.model.calls), 17)
        self.assertFalse(self.path.exists())
        self.assert_restored(before)


def load_tests(loader, tests, pattern):
    if __name__ == "__main__" or importlib.util.find_spec("torch") is None:
        return tests

    class IsolatedRoundtripCase(unittest.TestCase):
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
                yield IsolatedRoundtripCase(case.id())

    return unittest.TestSuite(isolated_cases(tests))


if __name__ == "__main__":
    unittest.main()
