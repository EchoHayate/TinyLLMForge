import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pilot_protocol import FLAGS
from test_tokenizer_measurement import MappingTokenizer
from tokenizer_measurement import measure
from validation_contract import digest


class GenerationTokenizer(MappingTokenizer):
    def __init__(self):
        self.render_calls = []
        self.decode_calls = []
        self.on_decode = None
        self.decoded = None

    def apply_chat_template(self, messages, **kwargs):
        self.render_calls.append((copy.deepcopy(messages), dict(kwargs)))
        return super().apply_chat_template(messages, **kwargs)

    def decode(self, token_ids, **kwargs):
        self.decode_calls.append((list(token_ids), dict(kwargs)))
        if self.on_decode is not None:
            self.on_decode()
        if self.decoded is not None:
            return self.decoded
        return "".join("[PAD]" if token == 0 else chr(token - 10)
                       for token in token_ids)


def scripted_model(torch, prompt_length, script):
    class ScriptedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.child = torch.nn.Identity()
            self.register_buffer("positions", torch.arange(2))
            self.prompt_length = prompt_length
            self.script = list(script)
            self.calls = []
            self.mode = None
            self.error = RuntimeError("synthetic forward failure")
            self.on_forward = None

        def forward(self, **kwargs):
            inputs = kwargs["input_ids"]
            self.calls.append({
                "keys": set(kwargs), "inputs": inputs.tolist(),
                "mask": kwargs["attention_mask"].tolist(),
                "input_dtype": inputs.dtype,
                "mask_dtype": kwargs["attention_mask"].dtype,
                "device": inputs.device,
                "grad": torch.is_grad_enabled(),
                "cache": kwargs["use_cache"], "return_dict": kwargs["return_dict"],
            })
            if self.on_forward is not None:
                self.on_forward()
            if self.mode == "exception":
                raise self.error
            position = inputs.shape[1] - self.prompt_length
            chosen = self.script[min(position, len(self.script) - 1)]
            logits = torch.zeros((1, inputs.shape[1], 140), device=inputs.device)
            logits[:, -1, chosen] = 5
            if self.mode == "tie":
                logits[:, -1, chosen + 1] = 5
            elif self.mode in ("nan", "inf", "-inf"):
                logits[:, -1, 139] = float(self.mode)
            elif self.mode == "earlier_nan":
                logits[:, 0, :] = float("nan")
            elif self.mode == "integer":
                logits = logits.long()
            elif self.mode == "complex":
                logits = logits.to(torch.complex64)
            elif self.mode == "list":
                logits = logits.tolist()
            elif self.mode == "rank":
                logits = logits[0]
            elif self.mode == "batch":
                logits = logits.repeat(2, 1, 1)
            elif self.mode == "sequence":
                logits = logits[:, -1:, :]
            elif self.mode == "empty_vocab":
                logits = logits[:, :, :0]
            elif self.mode == "missing":
                return SimpleNamespace()
            return SimpleNamespace(logits=logits)

    return ScriptedModel().eval()


class GenerationImportTests(unittest.TestCase):
    def test_cold_import_does_not_load_torch_or_transformers(self):
        self.assertTrue(Path(__file__).with_name("generation_core.py").is_file(),
                        "generation core module absent")
        result = subprocess.run(
            [sys.executable, "-B", "-c",
             "import sys; import generation_core; import test_generation_core; "
             "assert 'torch' not in sys.modules; "
             "assert 'transformers' not in sys.modules; "
             "assert 'training_worker' not in sys.modules"],
            cwd=Path(__file__).resolve().parent, capture_output=True, text=True,
            timeout=30, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires existing PyTorch")
class GenerationCoreTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("generation_core.py").is_file(),
                        "generation core module absent")
        self.core = importlib.import_module("generation_core")
        self.torch = importlib.import_module("torch")
        self.new_case()

    def new_case(self, messages=None, script=None):
        self.tokenizer = GenerationTokenizer()
        self.messages = messages if messages is not None else [
            {"role": "user", "content": "synthetic question"}]
        self.measured = measure(self.tokenizer, self.messages, answer=None)
        self.expected_sha = self.measured["state_sha256"]
        self.tokenizer.render_calls.clear()
        self.model = scripted_model(
            self.torch, self.measured["input"]["input_tokens"],
            [107, 108, 2] if script is None else script)

    def generate(self):
        return self.core.generate_candidate(
            self.model, self.tokenizer, self.messages,
            expected_tokenizer_sha256=self.expected_sha)

    def test_exact_receipt_greedy_prefix_growth_and_call_counts(self):
        before = copy.deepcopy(self.messages)
        parameters = list(self.model.parameters())
        state = {name: value.clone() for name, value in self.model.state_dict().items()}
        rng = self.torch.random.get_rng_state().clone()
        self.model.mode = "tie"
        with patch.object(self.torch.cuda, "reset_peak_memory_stats") as reset:
            result = self.generate()
        reset.assert_not_called()
        self.assertEqual(set(result), {
            "kind", "device_type", "tokenizer_state_sha256", "messages_sha256",
            "prompt_sha256", "input_tokens", "output_tokens", "output_ids",
            "candidate", "candidate_sha256", "stop_reason", "model_calls",
            "forward_calls", "timings_ns", "peak_allocated_bytes",
            "peak_reserved_bytes", *FLAGS})
        self.assertEqual(result["kind"], "LORA_GREEDY_CANDIDATE_UNATTESTED")
        self.assertEqual(result["device_type"], "cpu")
        self.assertEqual(result["tokenizer_state_sha256"], self.expected_sha)
        self.assertEqual(result["messages_sha256"], digest(before))
        self.assertEqual(result["prompt_sha256"], self.measured["input"]["prompt_sha256"])
        self.assertEqual(result["input_tokens"], self.measured["input"]["input_tokens"])
        self.assertEqual(result["output_ids"], [107, 108, 2])
        self.assertEqual(result["candidate"], "ab")
        self.assertEqual(result["candidate_sha256"], hashlib.sha256(b"ab").hexdigest())
        self.assertEqual(result["output_tokens"], 3)
        self.assertEqual(result["forward_calls"], 3)
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["stop_reason"], "eos")
        self.assertTrue(all(result[flag] is False for flag in FLAGS))
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        for position, call in enumerate(self.model.calls):
            prefix = self.measured["input"]["input_ids"] + [107, 108][:position]
            self.assertEqual(call["keys"], {
                "input_ids", "attention_mask", "use_cache", "return_dict"})
            self.assertEqual(call["inputs"], [prefix])
            self.assertEqual(call["mask"], [[1] * len(prefix)])
            self.assertEqual(call["input_dtype"], self.torch.long)
            self.assertEqual(call["mask_dtype"], self.torch.long)
            self.assertEqual(call["device"], self.model.anchor.device)
            self.assertIs(call["grad"], False)
            self.assertIs(call["cache"], False)
            self.assertIs(call["return_dict"], True)
        self.assertEqual(self.messages, before)
        self.assertTrue(self.torch.is_grad_enabled())
        self.assertTrue(self.torch.equal(rng, self.torch.random.get_rng_state()))
        self.assertTrue(all(current is prior for current, prior in
                            zip(self.model.parameters(), parameters)))
        for name, value in self.model.state_dict().items():
            self.assertTrue(self.torch.equal(value, state[name]))
        self.assertTrue(all(not module.training for module in self.model.modules()))
        self.assertTrue(all(not value.requires_grad and value.grad is None
                            for value in self.model.parameters()))

    def test_eos_immediately_returns_empty_candidate(self):
        self.model.script = [2, 107]
        result = self.generate()
        self.assertEqual(result["output_ids"], [2])
        self.assertEqual(result["output_tokens"], 1)
        self.assertEqual(result["forward_calls"], 1)
        self.assertEqual(result["candidate"], "")
        self.assertEqual(result["candidate_sha256"], hashlib.sha256(b"").hexdigest())
        self.assertEqual(result["stop_reason"], "eos")
        self.assertEqual(self.tokenizer.decode_calls[0][0], [])

    def test_exact_256_cap_and_eos_at_last_position(self):
        for script, reason in (([107], "length"), ([107] * 255 + [2], "eos")):
            with self.subTest(reason=reason):
                self.new_case(script=script)
                result = self.generate()
                expected = [107] * 256 if reason == "length" else script
                self.assertEqual(result["output_ids"], expected)
                self.assertEqual(result["output_tokens"], 256)
                self.assertEqual(result["forward_calls"], 256)
                self.assertEqual(len(self.model.calls), 256)
                self.assertEqual(result["stop_reason"], reason)
                self.assertEqual(result["candidate"], "a" * (256 if reason == "length" else 255))
                self.assertEqual(len(self.model.calls[-1]["inputs"][0]),
                                 result["input_tokens"] + 255)

    def test_input_4096_accepted_and_4097_rejected_before_forward(self):
        overhead = measure(self.tokenizer, [{"role": "user", "content": ""}])[
            "input"]["input_tokens"]
        self.new_case([{"role": "user", "content": "x" * (4096 - overhead)}], [2])
        self.assertEqual(self.generate()["input_tokens"], 4096)
        self.new_case([{"role": "user", "content": "x" * (4097 - overhead)}], [2])
        with self.assertRaisesRegex(ValueError, "input|INPUT_LIMIT"):
            self.generate()
        self.assertEqual(self.model.calls, [])

    def test_three_role_second_round_uses_complete_fixed_prompt(self):
        self.new_case(self.messages + [
            {"role": "assistant", "content": "previous output"},
            {"role": "user", "content": "synthetic feedback"}], [2])
        result = self.generate()
        self.assertEqual(self.model.calls[0]["inputs"],
                         [self.measured["input"]["input_ids"]])
        self.assertEqual(result["messages_sha256"], digest(self.messages))
        self.assertEqual(len(self.tokenizer.render_calls), 2)
        for messages, kwargs in self.tokenizer.render_calls:
            self.assertEqual(messages, self.messages)
            self.assertIs(kwargs["enable_thinking"], False)
            self.assertIs(kwargs["add_generation_prompt"], True)
            self.assertFalse(kwargs.get("truncation", False))

    def test_repeated_calls_reset_prefix_and_do_not_retain_output_list(self):
        first = self.generate()
        second = self.generate()
        self.assertEqual(first["output_ids"], second["output_ids"])
        self.assertIsNot(first["output_ids"], second["output_ids"])
        self.assertEqual(self.model.calls[0]["inputs"], self.model.calls[3]["inputs"])
        self.assertEqual(second["model_calls"], 1)
        self.assertEqual(second["forward_calls"], 3)

    def test_invalid_sha_and_tokenizer_mismatch_prevent_forward(self):
        for bad in (None, True, "", "a" * 63, "A" * 64, "g" * 64, "0" * 64):
            with self.subTest(sha=bad):
                self.expected_sha = bad
                with self.assertRaisesRegex(ValueError, "SHA256|sha256|digest|tokenizer"):
                    self.generate()
                self.assertEqual(self.model.calls, [])

    def test_tokenizer_mutation_at_decode_rejected_without_rerender(self):
        for attribute, value in (("chat_template", "changed"),
                                 ("eos_token_id", 0), ("pad_token_id", 2),
                                 ("get_vocab", lambda: {"a": 107, "eos": 2, "pad": 0})):
            with self.subTest(attribute=attribute):
                self.new_case()
                self.tokenizer.on_decode = lambda: setattr(self.tokenizer, attribute, value)
                with self.assertRaisesRegex(ValueError, "tokenizer"):
                    self.generate()
                self.assertEqual(len(self.model.calls), 3)
                self.assertEqual(len(self.tokenizer.render_calls), 2)

    def test_tokenizer_mutation_during_forward_rejected(self):
        self.model.on_forward = lambda: setattr(self.tokenizer, "chat_template", "changed")
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            self.generate()
        self.assertEqual(len(self.model.calls), 3)

    def test_message_mutation_at_decode_rejected(self):
        self.tokenizer.on_decode = lambda: self.messages[0].update(content="changed")
        with self.assertRaisesRegex(ValueError, "messages"):
            self.generate()
        self.assertEqual(len(self.model.calls), 3)

    def test_bad_roles_and_render_mutation_fail_before_model(self):
        self.messages = [{"role": "system", "content": "not permitted"}]
        with self.assertRaisesRegex(ValueError, "roles"):
            self.generate()
        self.assertEqual(self.model.calls, [])

        class MutatingTokenizer(GenerationTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                messages[0]["content"] = "changed"
                return super().apply_chat_template(messages, **kwargs)

        self.messages = [{"role": "user", "content": "synthetic question"}]
        self.tokenizer = MutatingTokenizer()
        before = copy.deepcopy(self.messages)
        with self.assertRaisesRegex(ValueError, "messages"):
            self.generate()
        self.assertEqual(self.messages, before)
        self.assertEqual(self.model.calls, [])

    def test_out_of_vocab_argmax_is_not_masked_or_retried(self):
        for invalid in (1, 138, 139):
            with self.subTest(token=invalid):
                self.new_case(script=[invalid])
                with self.assertRaisesRegex(ValueError, "vocabulary"):
                    self.generate()
                self.assertEqual(len(self.model.calls), 1)
                self.assertEqual(self.tokenizer.decode_calls, [])

    def test_nonfinite_last_position_logits_fail_closed(self):
        for mode in ("nan", "inf", "-inf"):
            with self.subTest(mode=mode):
                self.new_case()
                self.model.mode = mode
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    self.generate()
                self.assertEqual(len(self.model.calls), 1)

    def test_only_last_position_logits_need_be_finite(self):
        self.model.mode = "earlier_nan"
        self.assertEqual(self.generate()["candidate"], "ab")

    def test_wrong_type_and_shape_logits_rejected(self):
        for mode in ("integer", "complex", "list", "rank", "batch", "sequence",
                     "empty_vocab", "missing"):
            with self.subTest(mode=mode):
                self.new_case()
                self.model.mode = mode
                with self.assertRaisesRegex(ValueError, "logits"):
                    self.generate()
                self.assertEqual(len(self.model.calls), 1)

    def test_rejected_model_modes_and_state_are_not_repaired(self):
        for invalid in ("training", "child_training", "trainable", "grad", "empty",
                        "inference_parameter", "inference_buffer", "sparse_parameter",
                        "sparse_buffer", "mixed_device", "meta", "float64", "float16",
                        "complex_buffer", "float64_buffer", "grad_buffer", "not_module"):
            with self.subTest(invalid=invalid):
                self.new_case()
                model = self.model
                if invalid == "training":
                    model.train()
                elif invalid == "child_training":
                    model.child.train()
                elif invalid == "trainable":
                    model.anchor.requires_grad_(True)
                elif invalid == "grad":
                    model.anchor.grad = self.torch.ones_like(model.anchor)
                elif invalid == "empty":
                    del model.anchor
                    del model.positions
                elif invalid.startswith("inference"):
                    with self.torch.inference_mode():
                        tensor = self.torch.zeros(1)
                    if invalid == "inference_parameter":
                        model.anchor = self.torch.nn.Parameter(tensor, requires_grad=False)
                    else:
                        model.positions = tensor
                elif invalid.startswith("sparse"):
                    tensor = self.torch.ones(2).to_sparse()
                    if invalid == "sparse_parameter":
                        model.anchor = self.torch.nn.Parameter(tensor, requires_grad=False)
                    else:
                        model.positions = tensor
                elif invalid == "mixed_device":
                    model.positions = self.torch.empty(2, device="meta")
                elif invalid == "meta":
                    model.to("meta")
                elif invalid in ("float64", "float16"):
                    model.to(dtype=getattr(self.torch, invalid))
                elif invalid == "complex_buffer":
                    model.positions = self.torch.ones(2, dtype=self.torch.complex64)
                elif invalid == "float64_buffer":
                    model.positions = self.torch.ones(2, dtype=self.torch.float64)
                elif invalid == "grad_buffer":
                    model.positions = self.torch.ones(2, requires_grad=True)
                else:
                    self.model = object()
                parameters = [(value, value.requires_grad, value.grad)
                              for value in model.parameters()]
                modes = [module.training for module in model.modules()]
                with self.assertRaises(ValueError):
                    self.generate()
                self.assertEqual(model.calls, [])
                self.assertEqual([module.training for module in model.modules()], modes)
                for value, flag, grad in parameters:
                    self.assertEqual(value.requires_grad, flag)
                    self.assertIs(value.grad, grad)

    def test_bf16_fp32_and_integer_buffers_are_accepted(self):
        for dtype in (self.torch.float32, self.torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.new_case(script=[2])
                self.model.to(dtype=dtype)
                self.model.register_buffer("float_buffer", self.torch.ones(1, dtype=dtype))
                self.assertEqual(self.generate()["output_ids"], [2])

    def test_autocast_rejected_but_outer_no_grad_supported(self):
        with self.torch.autocast("cpu", dtype=self.torch.bfloat16):
            with self.assertRaisesRegex(ValueError, "autocast"):
                self.generate()
        self.assertEqual(self.model.calls, [])
        with self.torch.no_grad():
            self.assertEqual(self.generate()["candidate"], "ab")
            self.assertFalse(self.torch.is_grad_enabled())

    def test_decode_preserves_whitespace_markdown_and_special_tokens(self):
        raw = " \n```python\nunparsed synthetic text\n```\t "
        self.model.script = [0, 107, 2]
        self.tokenizer.decoded = raw
        result = self.generate()
        self.assertEqual(result["candidate"], raw)
        self.assertEqual(result["candidate_sha256"],
                         hashlib.sha256(raw.encode("utf-8")).hexdigest())
        self.assertEqual(self.tokenizer.decode_calls, [
            ([0, 107], {"skip_special_tokens": False,
                        "clean_up_tokenization_spaces": False})])

    def test_invalid_decode_type_utf8_and_byte_limit_rejected(self):
        for invalid in (17, b"text", ["text"], "\ud800", "a" * 65537, "é" * 32769):
            with self.subTest(kind=type(invalid).__name__, size=len(str(invalid))):
                self.new_case(script=[2])
                self.tokenizer.decoded = invalid
                with self.assertRaises((ValueError, UnicodeError)):
                    self.generate()
                self.assertEqual(len(self.model.calls), 1)
                self.assertEqual(len(self.tokenizer.decode_calls), 1)

    def test_exact_utf8_byte_limit_accepted(self):
        self.tokenizer.decoded = "é" * 32768
        result = self.generate()
        self.assertEqual(len(result["candidate"].encode("utf-8")), 65536)

    def test_model_exception_propagates_without_retry_and_restores_grad_context(self):
        self.model.mode = "exception"
        with self.assertRaises(RuntimeError) as caught:
            self.generate()
        self.assertIs(caught.exception, self.model.error)
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual(self.tokenizer.decode_calls, [])
        self.assertTrue(self.torch.is_grad_enabled())

    def test_cpu_memory_unavailable_and_timing_phases_nonoverlapping(self):
        result = self.generate()
        self.assertIsNone(result["peak_allocated_bytes"])
        self.assertIsNone(result["peak_reserved_bytes"])
        timings = result["timings_ns"]
        self.assertEqual(set(timings), {"preparation", "generation", "decoding", "wall"})
        self.assertTrue(all(type(value) is int and value > 0 for value in timings.values()))
        self.assertLessEqual(sum(timings[name] for name in
                                 ("preparation", "generation", "decoding")), timings["wall"])


def load_tests(loader, tests, pattern):
    if __name__ == "__main__" or importlib.util.find_spec("torch") is None:
        return tests

    class IsolatedGenerationCase(unittest.TestCase):
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
                yield IsolatedGenerationCase(case.id())

    return unittest.TestSuite(isolated_cases(tests))


if __name__ == "__main__":
    unittest.main()
