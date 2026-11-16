import ast
from collections import UserDict
import importlib
from pathlib import Path
import unittest
from unittest.mock import patch

from test_token_budget import SyntheticTokenizer


class MappingTokenizer(SyntheticTokenizer):
    def __call__(self, text, add_special_tokens=False, truncation=False):
        return UserDict(super().__call__(text, add_special_tokens, truncation))

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False,
                            truncation=False):
        if truncation:
            raise AssertionError("truncation forbidden")
        return super().apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )


class TokenizerMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("tokenizer_measurement.py").is_file(),
                        "tokenizer measurement not implemented")
        self.module = importlib.import_module("tokenizer_measurement")
        self.tokenizer = MappingTokenizer()
        self.messages = [{"role": "user", "content": "question"}]

    def test_mapping_normalization_and_unattested_output(self):
        result = self.module.measure(self.tokenizer, self.messages)
        self.assertEqual(result["kind"], "TOKENIZER_MEASUREMENT_UNATTESTED")
        self.assertEqual(result["status"], "OK")
        self.assertEqual(type(result["input"]["input_ids"]), list)
        self.assertIsNone(result["target"])
        self.assertEqual(set(result), {"kind", "state_sha256", "status", "input", "target"})
        self.assertEqual(result["input"]["input_tokens"],
                         len(result["input"]["input_ids"]))

    def test_unchanged_vocabulary_hashed_once_without_skipping_reads(self):
        vocabulary = self.tokenizer.get_vocab()
        for answer, expected_reads in ((None, 4), ("answer", 8)):
            with self.subTest(answer=answer):
                with patch.object(self.tokenizer, "get_vocab", wraps=self.tokenizer.get_vocab) as reads:
                    with patch.object(self.module, "digest", wraps=self.module.digest) as hashes:
                        self.module.measure(self.tokenizer, self.messages, answer)
                self.assertEqual(reads.call_count, expected_reads)
                self.assertEqual(sum(call.args[0] == vocabulary for call in hashes.call_args_list), 1)

    def test_mutated_shared_vocabulary_rejected_at_each_checkpoint(self):
        original = self.tokenizer.get_vocab()
        ordinary = next(name for name in original if name not in ("[EOS]", "[PAD]"))
        for checkpoint in range(2, 9):
            for value in (999, True, 10.0, -1):
                with self.subTest(checkpoint=checkpoint, value=value):
                    vocabulary = dict(original)
                    reads = 0

                    def get_vocab():
                        nonlocal reads
                        reads += 1
                        if reads == checkpoint:
                            vocabulary[ordinary] = value
                        return vocabulary

                    with patch.object(self.tokenizer, "get_vocab", side_effect=get_vocab):
                        with self.assertRaises(ValueError):
                            self.module.measure(self.tokenizer, self.messages, "answer")
                    self.assertEqual(reads, checkpoint)

    def test_property_mutation_after_vocabulary_validation_rejected(self):
        class GetterMutation(MappingTokenizer):
            def __init__(self, checkpoint, value):
                self.vocabulary = {**super().get_vocab(), "unused": 1}
                self.reads = 0
                self.checkpoint = checkpoint
                self.value = value

            def get_vocab(self):
                self.reads += 1
                return self.vocabulary

            @property
            def eos_token_id(self):
                if self.reads == self.checkpoint:
                    self.vocabulary["unused"] = self.value
                return 2

        for answer, checkpoint in ((None, 4), ("answer", 8)):
            for value in (True, 1.0):
                with self.subTest(answer=answer, value=value):
                    tokenizer = GetterMutation(checkpoint, value)
                    with self.assertRaises(ValueError):
                        self.module.measure(tokenizer, self.messages, answer)

    def test_4096_boundary_and_stop_before_target_encoding(self):
        messages = [{"role": "user", "content": ""}]
        overhead = self.module.measure(self.tokenizer, messages)["input"]["input_tokens"]
        messages[0]["content"] = "x" * (4096 - overhead)
        self.assertEqual(self.module.measure(self.tokenizer, messages)["status"], "OK")
        messages[0]["content"] += "x"
        result = self.module.measure(self.tokenizer, messages, "answer")
        self.assertEqual(result["status"], "INPUT_LIMIT")
        self.assertEqual(result["input"]["input_tokens"], 4097)
        self.assertIsNone(result["target"])

    def test_256_boundary_with_eos_and_template_tail(self):
        result = self.module.measure(self.tokenizer, self.messages, "x" * 255)
        self.assertEqual(result["status"], "OK")
        target = result["target"]
        self.assertEqual(target["target_tokens"], 256)
        self.assertEqual(target["target_ids"][-1], self.tokenizer.eos_token_id)
        self.assertEqual(target["template_tail_ids"], [ord("\n") + 10])
        self.assertEqual(target["full_sequence_tokens"],
                         result["input"]["input_tokens"] + 256 + 1)
        self.assertEqual(self.module.measure(
            self.tokenizer, self.messages, "x" * 256)["status"], "OUTPUT_LIMIT")

    def test_bool_ids_and_template_disagreement_rejected(self):
        class BadIds(MappingTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                result = super().apply_chat_template(messages, **kwargs)
                return [True] if kwargs.get("tokenize") else result

        class DifferentIds(MappingTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                result = super().apply_chat_template(messages, **kwargs)
                return result + [2] if kwargs.get("tokenize") else result

        for tokenizer in (BadIds(), DifferentIds()):
            with self.subTest(kind=type(tokenizer).__name__), self.assertRaises(ValueError):
                self.module.measure(tokenizer, self.messages)

    def test_prefix_duplicate_eos_and_message_mutation_rejected(self):
        class PrefixDrift(MappingTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                result = super().apply_chat_template(messages, **kwargs)
                if not kwargs.get("add_generation_prompt"):
                    return [42] + result if kwargs.get("tokenize") else "!" + result
                return result

        class MessageDrift(MappingTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                messages[0]["content"] += "changed"
                return super().apply_chat_template(messages, **kwargs)

        with self.assertRaises(ValueError):
            self.module.measure(PrefixDrift(), self.messages, "answer")
        with self.assertRaises(ValueError):
            self.module.measure(self.tokenizer, self.messages, "[EOS]")
        with self.assertRaises(ValueError):
            self.module.measure(MessageDrift(), self.messages)
        self.assertEqual(self.messages, [{"role": "user", "content": "question"}])

    def test_standalone_encoding_message_mutation_rejected(self):
        class LateMessageMutation(MappingTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                self.retained_messages = messages
                self.inside_template = True
                try:
                    return super().apply_chat_template(messages, **kwargs)
                finally:
                    self.inside_template = False

            def __call__(self, text, add_special_tokens=False, truncation=False):
                result = super().__call__(text, add_special_tokens, truncation)
                if not self.inside_template:
                    self.retained_messages[0]["content"] += "changed"
                return result

        with self.assertRaisesRegex(ValueError, "messages changed"):
            self.module.measure(LateMessageMutation(), self.messages)
        self.assertEqual(self.messages, [{"role": "user", "content": "question"}])

    def test_answer_eos_without_terminal_eos_rejected(self):
        class MissingTerminalEOS(MappingTokenizer):
            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False,
                                    truncation=False):
                text = super().apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                    enable_thinking=enable_thinking, truncation=truncation)
                if not add_generation_prompt:
                    text = text.removesuffix("[EOS]\n") + "\n"
                return self(text)["input_ids"] if tokenize else text

        for answer in ("prefix[EOS]" + "x" * 300, "answer[EOS]"):
            with self.subTest(answer=answer), self.assertRaisesRegex(
                    ValueError, "termination token must follow complete answer"):
                self.module.measure(MissingTerminalEOS(), self.messages, answer)

    def test_answer_boundary_uses_complete_context_and_fixed_modes(self):
        answer = "answer"
        for messages in (self.messages, self.messages + [
                {"role": "assistant", "content": "previous"},
                {"role": "user", "content": "feedback"}]):
            with self.subTest(messages=messages):
                tokenizer = MappingTokenizer()
                with patch.object(MappingTokenizer, "__call__",
                                  autospec=True,
                                  side_effect=MappingTokenizer.__call__) as encoding:
                    result = self.module.measure(tokenizer, messages, answer)
                boundary = result["input"]["prompt"] + answer
                calls = [call for call in encoding.call_args_list
                         if call.args[1] == boundary]
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0].kwargs,
                                 {"add_special_tokens": False, "truncation": False})
                self.assertEqual(result["target"]["target_tokens"], 7)
                self.assertEqual(set(result["target"]), {
                    "answer_sha256", "full_prompt_sha256", "full_sequence_ids",
                    "full_sequence_tokens", "target_start", "target_ids",
                    "target_tokens", "template_tail_ids",
                })

    def test_unprovable_answer_boundary_alignment_rejected(self):
        class MergedBoundary(MappingTokenizer):
            def __call__(self, text, add_special_tokens=False, truncation=False):
                result = super().__call__(text, add_special_tokens, truncation)
                if text.endswith("<assistant>answer[EOS]\n"):
                    result["input_ids"][-3] = ord("!") + 10
                return result

        with self.assertRaisesRegex(ValueError, "answer boundary token prefix mismatch"):
            self.module.measure(MergedBoundary(), self.messages, "answer")

    def test_boundary_encoding_message_mutation_rejected(self):
        class BoundaryMessageMutation(MappingTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                self.retained_messages = messages
                return super().apply_chat_template(messages, **kwargs)

            def __call__(self, text, add_special_tokens=False, truncation=False):
                result = super().__call__(text, add_special_tokens, truncation)
                if text.endswith("<assistant>answer"):
                    self.retained_messages[-1]["content"] += "changed"
                return result

        with self.assertRaisesRegex(ValueError, "messages changed"):
            self.module.measure(BoundaryMessageMutation(), self.messages, "answer")
        self.assertEqual(self.messages, [{"role": "user", "content": "question"}])

    def test_boundary_encoding_state_drift_rejected(self):
        class BoundaryStateDrift(MappingTokenizer):
            def __call__(self, text, add_special_tokens=False, truncation=False):
                result = super().__call__(text, add_special_tokens, truncation)
                if text.endswith("<assistant>answer"):
                    setattr(self, self.changed_attribute, self.changed_value)
                return result

        for attribute, value in (("chat_template", "changed"),
                                 ("eos_token_id", 0), ("pad_token_id", 2),
                                 ("get_vocab", lambda: {"replacement": 10, "eos": 2,
                                                       "pad": 0})):
            with self.subTest(attribute=attribute):
                tokenizer = BoundaryStateDrift()
                tokenizer.changed_attribute = attribute
                tokenizer.changed_value = value
                with self.assertRaisesRegex(ValueError, "tokenizer state changed"):
                    self.module.measure(tokenizer, self.messages, "answer")

    def test_boundary_final_state_observation_message_mutation_rejected(self):
        class BoundaryStateMessageMutation(MappingTokenizer):
            boundary_encoded = False

            def apply_chat_template(self, messages, **kwargs):
                self.retained_messages = messages
                return super().apply_chat_template(messages, **kwargs)

            def __call__(self, text, add_special_tokens=False, truncation=False):
                result = super().__call__(text, add_special_tokens, truncation)
                if text.endswith("<assistant>answer"):
                    self.boundary_encoded = True
                return result

            def get_vocab(self):
                if self.boundary_encoded:
                    self.retained_messages[0]["content"] += "changed"
                return super().get_vocab()

        with self.assertRaisesRegex(ValueError, "messages changed"):
            self.module.measure(BoundaryStateMessageMutation(), self.messages, "answer")
        self.assertEqual(self.messages, [{"role": "user", "content": "question"}])

    def test_boundary_encoding_invalid_mapping_and_ids_rejected(self):
        class InvalidBoundary(MappingTokenizer):
            def __call__(self, text, add_special_tokens=False, truncation=False):
                if text.endswith("<assistant>answer"):
                    return self.boundary_result
                return super().__call__(text, add_special_tokens, truncation)

        invalid_results = [
            (None, "tokenizer mapping"), ({}, "tokenizer mapping"),
            (UserDict({"input_ids": []}), "nonempty token ID list"),
            (UserDict({"input_ids": (10,)}), "nonempty token ID list"),
            (UserDict({"input_ids": [True]}), "declared integer token IDs"),
            (UserDict({"input_ids": [-1]}), "declared integer token IDs"),
            (UserDict({"input_ids": [999]}), "declared integer token IDs"),
        ]
        for result, error in invalid_results:
            with self.subTest(result=result), self.assertRaisesRegex(ValueError, error):
                tokenizer = InvalidBoundary()
                tokenizer.boundary_result = result
                self.module.measure(tokenizer, self.messages, "answer")

    def test_state_drift_rejected(self):
        class StateDrift(MappingTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                result = super().apply_chat_template(messages, **kwargs)
                self.chat_template = "different-template"
                return result

        with self.assertRaises(ValueError):
            self.module.measure(StateDrift(), self.messages)

    def test_full_second_round_count_and_legacy_agreement(self):
        first = [{"role": "user", "content": "x" * 3700}]
        self.assertEqual(self.module.measure(self.tokenizer, first)["status"], "OK")
        second = first + [
            {"role": "assistant", "content": "x" * 512},
            {"role": "user", "content": "feedback"},
        ]
        self.assertEqual(self.module.measure(self.tokenizer, second)["status"], "INPUT_LIMIT")
        legacy = importlib.import_module("token_budget")
        plain = SyntheticTokenizer()
        expected = legacy.inspect_target(
            plain, legacy.simulation_identity(plain), self.messages, "answer")
        actual = self.module.measure(self.tokenizer, self.messages, "answer")["target"]
        for name in ("target_ids", "target_tokens", "full_sequence_ids",
                     "full_sequence_tokens", "target_start", "template_tail_ids"):
            self.assertEqual(actual[name], expected[name])

    def test_new_modules_remain_import_pure(self):
        for name in ("tokenizer_inventory", "tokenizer_measurement"):
            path = Path(__file__).with_name(name + ".py")
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertTrue(all(alias.name in ("copy", "hashlib")
                                        for alias in node.names))
                if isinstance(node, ast.ImportFrom):
                    self.assertIn(node.module, ("collections.abc", "pilot_protocol",
                                                "prompt_rendering", "validation_contract"))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, ("exec", "eval", "compile", "open"))
            module = importlib.import_module(name)
            with patch("builtins.open", side_effect=AssertionError("unexpected file I/O")):
                importlib.reload(module)
