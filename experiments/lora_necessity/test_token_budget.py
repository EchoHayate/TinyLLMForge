import copy
import importlib
from pathlib import Path
import unittest

from validation_contract import reference_jobs


class SyntheticTokenizer:
    chat_template = "synthetic-template-v1"
    eos_token_id = 2
    pad_token_id = 0

    def get_vocab(self):
        return {**{chr(index): index + 10 for index in range(128)},
                "[EOS]": 2, "[PAD]": 0}

    def __call__(self, text, add_special_tokens=False, truncation=False):
        if add_special_tokens or truncation:
            raise AssertionError("unexpected tokenization mode")
        pieces = text.split("[EOS]")
        tokens = []
        for ordinal, piece in enumerate(pieces):
            if ordinal:
                tokens.append(2)
            tokens.extend(ord(character) + 10 for character in piece)
        return {"input_ids": tokens}

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False):
        if enable_thinking:
            raise AssertionError("thinking must be disabled")
        text = ""
        for message in messages:
            ending = "[EOS]\n" if message["role"] == "assistant" else "<end>\n"
            text += "<" + message["role"] + ">" + message["content"] + ending
        if add_generation_prompt:
            text += "<assistant>"
        return self(text)["input_ids"] if tokenize else text


class TokenBudgetTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("token_budget.py").is_file(),
                        "token budget not implemented")
        self.module = importlib.import_module("token_budget")
        self.tokenizer = SyntheticTokenizer()
        self.identity = self.module.simulation_identity(self.tokenizer)
        data = Path(__file__).with_name("data")
        self.supervision = (data / "supervision_v1.json").read_bytes()
        self.prompts = (data / "prompts_v1.json").read_bytes()
        self.answers = reference_jobs((data / "task_sources_v1.json").read_bytes(),
                                      (data / "manifest_v1.json").read_bytes())

    def test_exact_input_boundary_and_no_truncation(self):
        messages = [{"role": "user", "content": ""}]
        overhead = self.module.inspect_call(self.tokenizer, self.identity, messages)["input_tokens"]
        messages[0]["content"] = "x" * (4096 - overhead)
        row = self.module.check_call(self.tokenizer, self.identity, messages)
        self.assertEqual(row["input_tokens"], 4096)
        messages[0]["content"] += "x"
        with self.assertRaises(ValueError):
            self.module.check_call(self.tokenizer, self.identity, messages)
        self.assertEqual(self.module.inspect_call(
            self.tokenizer, self.identity, messages)["input_tokens"], 4097)

    def test_output_boundary_includes_eos_not_template_whitespace(self):
        messages = [{"role": "user", "content": "question"}]
        row = self.module.inspect_target(self.tokenizer, self.identity, messages, "x" * 255)
        self.assertEqual(row["target_tokens"], 256)
        self.assertTrue(row["target_within_limit"])
        self.assertEqual(row["template_tail_ids"], [ord("\n") + 10])
        row = self.module.inspect_target(self.tokenizer, self.identity, messages, "x" * 256)
        self.assertEqual(row["target_tokens"], 257)
        self.assertFalse(row["target_within_limit"])

    def test_identity_template_and_id_corruption_rejected(self):
        self.tokenizer.chat_template = "changed"
        with self.assertRaises(ValueError):
            self.module.inspect_call(self.tokenizer, self.identity,
                                     [{"role": "user", "content": "question"}])
        self.tokenizer = SyntheticTokenizer()
        changed = copy.deepcopy(self.identity)
        changed["kind"] = "real"
        with self.assertRaises(ValueError):
            self.module.inspect_call(self.tokenizer, changed,
                                     [{"role": "user", "content": "question"}])

        class CorruptTokenizer(SyntheticTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                result = super().apply_chat_template(messages, **kwargs)
                return [True] if kwargs.get("tokenize") else result

        corrupt = CorruptTokenizer()
        with self.assertRaises(ValueError):
            self.module.inspect_call(corrupt, self.module.simulation_identity(corrupt),
                                     [{"role": "user", "content": "question"}])

    def test_changed_full_prefix_and_duplicate_eos_rejected(self):
        class PrefixTokenizer(SyntheticTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                if not kwargs.get("add_generation_prompt", True):
                    messages = copy.deepcopy(messages)
                    messages[0]["content"] += "changed"
                return super().apply_chat_template(messages, **kwargs)

        changed = PrefixTokenizer()
        with self.assertRaises(ValueError):
            self.module.inspect_target(changed, self.module.simulation_identity(changed),
                                       [{"role": "user", "content": "question"}], "answer")
        with self.assertRaises(ValueError):
            self.module.inspect_target(self.tokenizer, self.identity,
                                       [{"role": "user", "content": "question"}], "[EOS]")

    def test_full_second_message_is_counted_not_candidate_length_estimate(self):
        first = [{"role": "user", "content": "x" * 3700}]
        self.module.check_call(self.tokenizer, self.identity, first)
        second = first + [
            {"role": "assistant", "content": "x" * 512},
            {"role": "user", "content": "feedback"},
        ]
        with self.assertRaises(ValueError):
            self.module.check_call(self.tokenizer, self.identity, second)

    def test_audit_is_complete_but_only_simulated_and_sampled(self):
        report = self.module.audit_simulated(
            self.supervision, self.prompts, self.answers, self.tokenizer, self.identity)
        self.assertEqual(report["kind"], "LORA_TOKEN_BUDGET_SIMULATION")
        self.assertEqual(len(report["training"]), 8)
        self.assertEqual(len(report["first"]), 136)
        self.assertEqual(len(report["references"]), 44)
        self.assertEqual(len(report["sampled_second"]), 136)
        self.assertFalse(report["all_possible_second_rounds_validated"])
        for flag in ("benchmark_eligible", "formal_evaluation_ready",
                     "net_benefit_established", "isolated"):
            self.assertIs(report[flag], False)
        self.answers[0]["reference_source"] += "\n"
        with self.assertRaises(ValueError):
            self.module.audit_simulated(self.supervision, self.prompts, self.answers,
                                        self.tokenizer, self.identity)

    def test_no_runtime_dependencies_or_import_side_effects(self):
        import ast
        import builtins
        from unittest.mock import patch
        for name in ("validation_contract", "validation_report", "prompt_rendering", "token_budget"):
            path = Path(__file__).with_name(name + ".py")
            if not path.is_file():
                self.fail("finish both controller tasks before combined purity check")
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertTrue(all(alias.name in ("copy", "hashlib", "json", "math")
                                        for alias in node.names))
                if isinstance(node, ast.ImportFrom):
                    self.assertIn(node.module, ("pilot_protocol", "validation_contract",
                                                "prompt_rendering"))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, ("exec", "eval", "compile", "open"))
            module = importlib.import_module(name)
            with patch.object(builtins, "open", side_effect=AssertionError("unexpected I/O")):
                importlib.reload(module)
