import importlib
import importlib.util
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from validation_contract import PINS, digest


def synthetic_dataset():
    rows = [{
        "task_id": f"S{index:02}", "input_ids": [3, 4, 5, 2],
        "attention_mask": [1, 1, 1, 1], "labels": [-100, -100, 5, 2],
        "prompt_tokens": 2, "target_tokens": 2,
        "messages_sha256": "a" * 64, "answer_sha256": "b" * 64,
    } for index in range(1, 9)]
    dataset = {
        "kind": "LORA_TRAINING_INPUTS_UNATTESTED", "version": 1,
        "supervision_sha256": PINS["supervision_v1.json"],
        "tokenizer_state_sha256": "c" * 64, "eos_token_id": 2,
        "pad_token_id": 0, "rows": rows,
    }
    return {**dataset, "dataset_sha256": digest(dataset)}


def tiny_model(torch):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="qwen3", num_hidden_layers=1, return_dict=True)
            self.embedding = torch.nn.Embedding(16, 4)
            self.model = torch.nn.Module()
            layer = torch.nn.Module()
            layer.self_attn = torch.nn.Module()
            layer.self_attn.q_proj = torch.nn.Linear(4, 4)
            layer.self_attn.v_proj = torch.nn.Linear(4, 4)
            self.model.layers = torch.nn.ModuleList([layer])
            self.lm_head = torch.nn.Linear(4, 16)
            self.calls = []
            self.fail_at = None
            self.forward_error = RuntimeError("synthetic forward failure")

        def forward(self, input_ids, attention_mask, use_cache, return_dict=None):
            self.calls.append((input_ids.detach().cpu().tolist(), use_cache))
            if self.fail_at == len(self.calls):
                raise self.forward_error
            hidden = self.embedding(input_ids)
            attention = self.model.layers[0].self_attn
            hidden = torch.tanh(attention.q_proj(hidden) + attention.v_proj(hidden))
            logits = self.lm_head(hidden)
            if return_dict is None:
                return_dict = self.config.return_dict
            return SimpleNamespace(logits=logits) if return_dict else (logits,)

    return TinyModel().to(dtype=torch.bfloat16)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires existing PyTorch")
class TrainingPrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("training_worker.py").is_file(),
                        "training core not implemented")
        self.worker = importlib.import_module("training_worker")
        self.torch = importlib.import_module("torch")

    def test_zero_delta_and_first_step_gradient_path(self):
        torch = self.torch
        base = torch.nn.Linear(4, 3).to(dtype=torch.bfloat16)
        projection = self.worker.LoRAProjection(base)
        inputs = torch.ones(2, 4, dtype=torch.bfloat16)
        self.assertTrue(torch.equal(projection(inputs), base(inputs)))
        self.assertFalse(base.weight.requires_grad)
        self.assertFalse(base.bias.requires_grad)
        self.assertEqual(projection.adapter_a.dtype, torch.float32)
        self.assertEqual(projection.adapter_b.dtype, torch.float32)
        self.assertEqual(tuple(projection.adapter_a.shape), (8, 4))
        self.assertEqual(tuple(projection.adapter_b.shape), (3, 8))
        projection(inputs).float().sum().backward()
        self.assertIsNone(base.weight.grad)
        self.assertTrue(torch.equal(projection.adapter_a.grad,
                                    torch.zeros_like(projection.adapter_a)))
        self.assertGreater(projection.adapter_b.grad.abs().sum().item(), 0)

    def test_causal_answer_loss_ignores_prompt_and_final_logit(self):
        torch = self.torch
        logits = torch.tensor([[[3., 1., 2.], [1., 5., 0.],
                                [0., 1., 4.], [9., -2., 0.]]],
                              requires_grad=True)
        labels = torch.tensor([[-100, -100, 1, 2]])
        loss = self.worker.answer_loss(logits, labels)
        expected = (-torch.log_softmax(logits[0, 1], dim=-1)[1]
                    - torch.log_softmax(logits[0, 2], dim=-1)[2]) / 2
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertEqual(logits.grad[0, 0].abs().sum().item(), 0)
        self.assertEqual(logits.grad[0, 3].abs().sum().item(), 0)
        self.assertGreater(logits.grad[0, 1:3].abs().sum().item(), 0)

    def test_loss_rejects_missing_or_invalid_targets(self):
        torch = self.torch
        logits = torch.zeros(1, 4, 3)
        for labels in (torch.full((1, 4), -100),
                       torch.tensor([[-100, -100, 1, 3]]),
                       torch.tensor([[-100, -100, 1, -1]])):
            with self.subTest(labels=labels.tolist()), self.assertRaises(ValueError):
                self.worker.answer_loss(logits, labels)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires existing PyTorch")
class TrainingCoreTests(unittest.TestCase):
    def setUp(self):
        self.worker = importlib.import_module("training_worker")
        self.assertTrue(callable(getattr(self.worker, "train_adapter", None)),
                        "training entrypoint not implemented")
        self.torch = importlib.import_module("torch")
        self.model = tiny_model(self.torch)
        self.dataset = synthetic_dataset()

    def snapshot(self):
        return (
            {name: value.clone() for name, value in self.model.state_dict().items()},
            {name: parameter.requires_grad for name, parameter in self.model.named_parameters()},
            {name: module.training for name, module in self.model.named_modules()},
            self.torch.random.get_rng_state().clone(),
            dict(self.model.named_modules()),
        )

    def assert_restored(self, before):
        weights, flags, modes, rng, modules = before
        current = self.model.state_dict()
        self.assertEqual(set(current), set(weights))
        for name, value in weights.items():
            self.assertTrue(self.torch.equal(value, current[name]), name)
        self.assertEqual(flags, {name: value.requires_grad
                                 for name, value in self.model.named_parameters()})
        self.assertEqual(modes, {name: value.training
                                 for name, value in self.model.named_modules()})
        self.assertTrue(self.torch.equal(rng, self.torch.random.get_rng_state()))
        self.assertEqual(modules, dict(self.model.named_modules()))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_sixteen_steps_leave_base_and_caller_state_unchanged(self):
        self.model.model.eval()
        self.model.embedding.weight.requires_grad_(False)
        before = self.snapshot()
        original = copy.deepcopy(self.dataset)
        result = self.worker.train_adapter(self.model, self.dataset)
        self.assert_restored(before)
        self.assertEqual(self.dataset, original)
        self.assertEqual(set(result), {"receipt", "initial_adapter", "adapter"})
        receipt = result["receipt"]
        self.assertEqual(receipt["kind"], "LORA_TRAINING_CORE_RESULT_UNATTESTED")
        self.assertEqual(len(receipt["steps"]), 16)
        self.assertEqual([step["task_id"] for step in receipt["steps"]],
                         [f"S{index:02}" for index in range(1, 9)] * 2)
        self.assertEqual(len(self.model.calls), 16)
        self.assertTrue(all(cache is False for _, cache in self.model.calls))
        self.assertEqual(receipt["dataset_sha256"], self.dataset["dataset_sha256"])
        self.assertEqual(receipt["supervised_tokens"], 32)
        self.assertNotEqual(receipt["initial_adapter_sha256"], receipt["adapter_sha256"])
        for flag in ("benchmark_eligible", "formal_evaluation_ready",
                     "net_benefit_established", "isolated"):
            self.assertIs(receipt[flag], False)
        self.assertIsNone(receipt["end_to_end_ns"])
        self.assertGreater(receipt["adapter_bytes"], 0)
        self.assertGreater(receipt["optimizer_tensor_bytes"], 0)
        self.assertTrue(all(value.device.type == "cpu" and not value.requires_grad
                            for value in result["adapter"].values()))
        self.assertTrue(any(not self.torch.equal(value, result["adapter"][name])
                            for name, value in result["initial_adapter"].items()))

    def test_repeated_calls_are_deterministic_without_global_rng_changes(self):
        before = self.snapshot()
        first = self.worker.train_adapter(self.model, self.dataset)
        second = self.worker.train_adapter(self.model, self.dataset)
        self.assert_restored(before)
        for name in first["adapter"]:
            self.assertTrue(self.torch.equal(first["adapter"][name],
                                            second["adapter"][name]))

    def test_forward_failure_restores_state_and_does_not_retry(self):
        self.model.fail_at = 3
        before = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, "synthetic forward failure") as caught:
            self.worker.train_adapter(self.model, self.dataset)
        self.assertIs(caught.exception, self.model.forward_error)
        self.assert_restored(before)
        self.assertEqual(len(self.model.calls), 3)

    def test_mutation_and_forward_failure_requires_discard_with_original_cause(self):
        self.model.register_buffer("rotary", self.torch.ones(2), persistent=False)
        self.model.model.eval()
        self.model.embedding.weight.requires_grad_(False)
        self.model.fail_at = 1
        before = self.snapshot()
        original = self.model.forward

        def mutate_then_fail(**kwargs):
            self.model.rotary.add_(1)
            return original(**kwargs)

        with patch.object(self.model, "forward", side_effect=mutate_then_fail):
            with self.assertRaisesRegex(ValueError, "base state changed; discard model") as caught:
                self.worker.train_adapter(self.model, self.dataset)
        self.assertIs(caught.exception.__cause__, self.model.forward_error)
        self.assert_restored(before)
        self.assertEqual(self.model.rotary.tolist(), [2, 2])
        self.assertEqual(len(self.model.calls), 1)

    def test_unverifiable_base_requires_discard_on_success_and_failure(self):
        for fail_at in (None, 1):
            with self.subTest(fail_at=fail_at):
                self.model = tiny_model(self.torch)
                self.model.model.eval()
                self.model.embedding.weight.requires_grad_(False)
                self.model.fail_at = fail_at
                before = self.snapshot()
                original = self.worker._state
                integrity_error = RuntimeError("synthetic integrity read failure")

                def unreadable_after_forward(model):
                    if model.calls:
                        self.assertEqual(before[4], dict(model.named_modules()))
                        self.assertEqual(before[1], {
                            name: value.requires_grad for name, value in model.named_parameters()})
                        self.assertEqual(before[2], {
                            name: value.training for name, value in model.named_modules()})
                        self.torch.rand(1)
                        raise integrity_error
                    return original(model)

                with patch.object(self.worker, "_state", side_effect=unreadable_after_forward):
                    with self.assertRaisesRegex(
                            ValueError, "base state unverifiable; discard model") as caught:
                        self.worker.train_adapter(self.model, self.dataset)
                expected_cause = self.model.forward_error if fail_at else integrity_error
                self.assertIs(caught.exception.__cause__, expected_cause)
                self.assert_restored(before)
                self.assertEqual(len(self.model.calls), fail_at or 16)

    def test_tuple_return_config_is_overridden_without_config_mutation(self):
        self.model.config.return_dict = False
        inputs = self.torch.tensor([[3, 4, 5, 2]])
        output = self.model(input_ids=inputs, attention_mask=self.torch.ones_like(inputs),
                            use_cache=False)
        self.assertIsInstance(output, tuple)
        self.model.calls.clear()
        config = self.model.config
        config_before = copy.deepcopy(vars(config))
        before = self.snapshot()
        result = self.worker.train_adapter(self.model, self.dataset)
        self.assertEqual(len(result["receipt"]["steps"]), 16)
        self.assertEqual(len(self.model.calls), 16)
        self.assertIs(self.model.config, config)
        self.assertEqual(vars(config), config_before)
        self.assert_restored(before)

    def test_nonfinite_base_rejected_before_forward(self):
        with self.torch.no_grad():
            self.model.embedding.weight[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            self.worker.train_adapter(self.model, self.dataset)
        self.assertEqual(self.model.calls, [])

    def test_nonpersistent_float32_buffer_is_preserved_and_mutation_detected(self):
        self.model.register_buffer("rotary", self.torch.ones(2), persistent=False)
        result = self.worker.train_adapter(self.model, self.dataset)
        self.assertEqual(self.model.rotary.dtype, self.torch.float32)
        self.assertEqual(self.model.rotary.tolist(), [1, 1])
        self.assertEqual(len(result["receipt"]["steps"]), 16)
        original = self.model.forward

        def mutate_buffer(**kwargs):
            self.model.rotary.add_(1)
            return original(**kwargs)

        with patch.object(self.model, "forward", side_effect=mutate_buffer):
            with self.assertRaisesRegex(ValueError, "base state changed"):
                self.worker.train_adapter(self.model, self.dataset)
        self.assertIs(type(self.model.model.layers[0].self_attn.q_proj),
                      self.torch.nn.Linear)

    def test_invalid_topology_or_precision_rejected_without_mutation(self):
        for change in ("shared", "float32", "extra", "layers"):
            with self.subTest(change=change):
                self.model = tiny_model(self.torch)
                attention = self.model.model.layers[0].self_attn
                if change == "shared":
                    attention.v_proj = attention.q_proj
                elif change == "float32":
                    self.model.float()
                elif change == "extra":
                    self.model.q_proj = self.torch.nn.Linear(4, 4).bfloat16()
                else:
                    self.model.config.num_hidden_layers = 2
                before = self.snapshot()
                with self.assertRaises(ValueError):
                    self.worker.train_adapter(self.model, self.dataset)
                self.assert_restored(before)
                self.assertEqual(self.model.calls, [])

    def test_disabled_autograd_rejected(self):
        with self.torch.no_grad(), self.assertRaises(ValueError):
            self.worker.train_adapter(self.model, self.dataset)
        with self.torch.inference_mode(), self.assertRaises(ValueError):
            self.worker.train_adapter(self.model, self.dataset)
        self.assertEqual(self.model.calls, [])

    def test_autocast_rejected_before_forward(self):
        before = self.snapshot()
        with self.torch.autocast("cpu", dtype=self.torch.bfloat16):
            with self.assertRaisesRegex(ValueError, "autocast"):
                self.worker.train_adapter(self.model, self.dataset)
        self.assert_restored(before)
        self.assertEqual(self.model.calls, [])

    def test_nonfinite_output_restores_state_and_stops(self):
        original = self.model.forward

        def nonfinite_output(**kwargs):
            output = original(**kwargs)
            return SimpleNamespace(logits=output.logits * float("nan"))

        before = self.snapshot()
        with patch.object(self.model, "forward", side_effect=nonfinite_output):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                self.worker.train_adapter(self.model, self.dataset)
        self.assert_restored(before)
        self.assertEqual(len(self.model.calls), 1)

    def test_receipt_costs_and_tensor_payloads_are_consistent(self):
        result = self.worker.train_adapter(self.model, self.dataset)
        receipt = result["receipt"]
        self.assertEqual(receipt["adapter_bytes"], sum(
            value.numel() * value.element_size() for value in result["adapter"].values()))
        self.assertEqual(receipt["optimizer_tensor_bytes"],
                         2 * receipt["adapter_bytes"] + 4 * len(result["adapter"]))
        self.assertLessEqual(sum(step["elapsed_ns"] for step in receipt["steps"]),
                             receipt["training_ns"])
        self.assertLessEqual(sum(receipt[name] for name in
                                 ("preparation_ns", "training_ns", "verification_ns")),
                             receipt["core_wall_ns"])
        for name in ("model_load_ns", "adapter_save_ns", "launch_ns",
                     "peak_allocated_bytes", "peak_reserved_bytes", "end_to_end_ns"):
            self.assertIsNone(receipt[name])

    def test_optimizer_cannot_include_base_parameters(self):
        original = self.torch.optim.AdamW
        base_parameter = self.model.embedding.weight

        def wrong_optimizer(parameters, **kwargs):
            optimizer = original(parameters, **kwargs)
            optimizer.param_groups[0]["params"].append(base_parameter)
            return optimizer

        before = self.snapshot()
        with patch.object(self.torch.optim, "AdamW", side_effect=wrong_optimizer):
            with self.assertRaises(ValueError):
                self.worker.train_adapter(self.model, self.dataset)
        self.assert_restored(before)
        self.assertEqual(self.model.calls, [])


if __name__ == "__main__":
    unittest.main()
