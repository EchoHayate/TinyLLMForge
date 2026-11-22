import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from prompt_rendering import training_jobs
from tokenizer_measurement import measure
from validation_contract import PINS, digest


class TrainingTokenizer:
    chat_template = "synthetic-training-template"
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self, prompt_tokens=12, target_tokens=5, tail_tokens=1):
        self.prompt_tokens = prompt_tokens
        self.target_tokens = target_tokens
        self.tail_tokens = tail_tokens
        self.encoded_lists = []
        self.rendered_messages = []

    def get_vocab(self):
        return {"pad": 0, "eos": 2, "prompt": 10, "answer": 11, "tail": 12}

    def __call__(self, text, add_special_tokens=False, truncation=False):
        if add_special_tokens or truncation:
            raise AssertionError("special tokens and truncation forbidden")
        prompt, separator, suffix = text.partition("<assistant>")
        if not separator or not prompt:
            raise AssertionError("missing synthetic prompt boundary")
        tokens = [10] * self.prompt_tokens
        if suffix:
            tokens += [11] * (self.target_tokens - 1)
            if suffix.endswith("[EOS]\n"):
                tokens += [self.eos_token_id] + [12] * self.tail_tokens
        self.encoded_lists.append(tokens)
        return {"input_ids": tokens}

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False,
                            truncation=False):
        if enable_thinking or truncation:
            raise AssertionError("thinking and truncation forbidden")
        self.rendered_messages.append(copy.deepcopy(messages))
        text = "<user>" + messages[0]["content"] + "<assistant>"
        if not add_generation_prompt:
            text += messages[-1]["content"] + "[EOS]\n"
        return self(text)["input_ids"] if tokenize else text


def bind_dataset(dataset):
    dataset["dataset_sha256"] = digest({
        name: value for name, value in dataset.items() if name != "dataset_sha256"
    })
    return dataset


def synthetic_dataset(prompt_tokens=3, target_tokens=2):
    rows = []
    for ordinal in range(1, 9):
        target_ids = [11] * (target_tokens - 1) + [2]
        rows.append({
            "task_id": f"S{ordinal:02d}",
            "input_ids": [10] * prompt_tokens + target_ids,
            "attention_mask": [1] * (prompt_tokens + target_tokens),
            "labels": [-100] * prompt_tokens + target_ids,
            "prompt_tokens": prompt_tokens,
            "target_tokens": target_tokens,
            "messages_sha256": "a" * 64,
            "answer_sha256": "b" * 64,
        })
    return bind_dataset({
        "kind": "LORA_TRAINING_INPUTS_UNATTESTED",
        "version": 1,
        "supervision_sha256": PINS["supervision_v1.json"],
        "tokenizer_state_sha256": "c" * 64,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "rows": rows,
    })


class TrainingInputsTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("training_inputs.py").is_file(),
                        "training inputs not implemented")
        self.module = importlib.import_module("training_inputs")
        self.supervision = (
            Path(__file__).with_name("data") / "supervision_v1.json"
        ).read_bytes()

    def test_prepare_exact_schema_and_measured_rows(self):
        tokenizer = TrainingTokenizer()
        dataset = self.module.prepare_training(tokenizer, self.supervision)
        self.assertEqual(set(dataset), {
            "kind", "version", "supervision_sha256", "tokenizer_state_sha256",
            "eos_token_id", "pad_token_id", "rows", "dataset_sha256",
        })
        self.assertEqual(dataset["kind"], "LORA_TRAINING_INPUTS_UNATTESTED")
        self.assertEqual(dataset["version"], 1)
        self.assertEqual(dataset["supervision_sha256"], PINS["supervision_v1.json"])
        self.assertEqual(dataset["eos_token_id"], 2)
        self.assertEqual(dataset["pad_token_id"], 0)
        self.assertEqual([row["task_id"] for row in dataset["rows"]],
                         [f"S{ordinal:02d}" for ordinal in range(1, 9)])
        for row, job in zip(dataset["rows"], training_jobs(self.supervision)):
            expected = measure(TrainingTokenizer(), job["messages"], job["answer"])
            self.assertEqual(set(row), {
                "task_id", "input_ids", "attention_mask", "labels", "prompt_tokens",
                "target_tokens", "messages_sha256", "answer_sha256",
            })
            self.assertEqual(row["input_ids"],
                             expected["input"]["input_ids"] + expected["target"]["target_ids"])
            self.assertEqual(row["prompt_tokens"], expected["input"]["input_tokens"])
            self.assertEqual(row["target_tokens"], expected["target"]["target_tokens"])
            self.assertEqual(row["messages_sha256"], digest(job["messages"]))
            self.assertEqual(row["answer_sha256"],
                             hashlib.sha256(job["answer"].encode("utf-8")).hexdigest())
            self.assertEqual(dataset["tokenizer_state_sha256"], expected["state_sha256"])
        self.assertEqual(dataset, json.loads(json.dumps(dataset)))
        self.assertEqual(self.module.validate_training(dataset), dataset["dataset_sha256"])

    def test_prepare_answer_only_unshifted_labels_and_tail_exclusion(self):
        dataset = self.module.prepare_training(
            TrainingTokenizer(tail_tokens=100), self.supervision)
        for row in dataset["rows"]:
            self.assertEqual(row["input_ids"], [10] * 12 + [11] * 4 + [2])
            self.assertEqual(row["attention_mask"], [1] * 17)
            self.assertEqual(row["labels"], [-100] * 12 + [11] * 4 + [2])

    def test_prepare_allows_exact_limits_excluding_template_tail(self):
        try:
            dataset = self.module.prepare_training(
                TrainingTokenizer(prompt_tokens=4096, target_tokens=256, tail_tokens=20),
                self.supervision)
        except ValueError as error:
            self.fail(f"independent prompt/target budgets rejected: {error}")
        self.assertTrue(all(len(row["input_ids"]) == 4352 for row in dataset["rows"]))
        self.assertTrue(all(row["prompt_tokens"] == 4096 for row in dataset["rows"]))
        self.assertTrue(all(row["target_tokens"] == 256 for row in dataset["rows"]))

    def test_prepare_rejects_independent_prompt_and_target_overflow(self):
        for prompt_tokens, target_tokens in ((4097, 256), (4096, 257)):
            with self.subTest(prompt=prompt_tokens, target=target_tokens):
                with self.assertRaises(ValueError):
                    self.module.prepare_training(
                        TrainingTokenizer(prompt_tokens, target_tokens), self.supervision)

    def test_prepare_rejects_corrupted_source_before_tokenization(self):
        tokenizer = TrainingTokenizer()
        for raw in (self.supervision + b"\n", b"[]", None,
                    self.supervision.decode("utf-8"), bytearray(self.supervision)):
            with self.subTest(raw_type=type(raw)), self.assertRaises(ValueError):
                self.module.prepare_training(tokenizer, raw)
        self.assertEqual(tokenizer.rendered_messages, [])

    def test_prepare_rejects_tokenizer_state_drift_between_jobs(self):
        class BetweenJobsDrift(TrainingTokenizer):
            reads = 0

            def get_vocab(self):
                self.reads += 1
                return {**super().get_vocab(), "unused": 30 + (self.reads - 1) // 8}

        with self.assertRaisesRegex(ValueError, "state"):
            self.module.prepare_training(BetweenJobsDrift(), self.supervision)

    def test_prepare_rejects_tokenizer_state_drift_within_job(self):
        class WithinJobDrift(TrainingTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                result = super().apply_chat_template(messages, **kwargs)
                self.chat_template = "changed-template"
                return result

        with self.assertRaisesRegex(ValueError, "state"):
            self.module.prepare_training(WithinJobDrift(), self.supervision)

    def test_prepare_copies_tokenizer_lists_and_does_not_alias_rows_or_labels(self):
        tokenizer = TrainingTokenizer()
        dataset = self.module.prepare_training(tokenizer, self.supervision)
        original = copy.deepcopy(dataset)
        for tokens in tokenizer.encoded_lists:
            tokens[:] = [999]
        self.assertEqual(dataset, original)
        dataset["rows"][0]["input_ids"][-1] = 999
        self.assertEqual(dataset["rows"][0]["labels"], original["rows"][0]["labels"])
        self.assertEqual(dataset["rows"][1:], original["rows"][1:])
        self.assertEqual(self.module.prepare_training(TrainingTokenizer(), self.supervision),
                         original)

    def test_validate_independent_dataset_returns_digest_without_mutation(self):
        dataset = synthetic_dataset()
        original = copy.deepcopy(dataset)
        with patch.object(self.module, "measure", side_effect=AssertionError("no tokenizer")):
            with patch.object(self.module, "training_jobs",
                              side_effect=AssertionError("no source read")):
                self.assertEqual(self.module.validate_training(dataset),
                                 dataset["dataset_sha256"])
        self.assertEqual(dataset, original)

    def test_validate_accepts_minimum_and_maximum_boundaries(self):
        for prompt_tokens, target_tokens in (
                (1, 1), (3840, 256), (3841, 256), (4096, 1), (4096, 256)):
            with self.subTest(prompt=prompt_tokens, target=target_tokens):
                dataset = synthetic_dataset(prompt_tokens, target_tokens)
                try:
                    actual = self.module.validate_training(dataset)
                except ValueError as error:
                    self.fail(f"independent prompt/target budgets rejected: {error}")
                self.assertEqual(actual, dataset["dataset_sha256"])

    def test_validate_allows_prompt_eos_and_pad_equal_to_eos(self):
        dataset = synthetic_dataset()
        dataset["pad_token_id"] = dataset["eos_token_id"]
        dataset["rows"][0]["input_ids"][0] = dataset["eos_token_id"]
        bind_dataset(dataset)
        self.assertEqual(self.module.validate_training(dataset), dataset["dataset_sha256"])

    def test_validate_rejects_missing_or_extra_fields(self):
        for level in ("dataset", "row"):
            original = synthetic_dataset()
            fields = original if level == "dataset" else original["rows"][0]
            for name in list(fields) + ["extra"]:
                with self.subTest(level=level, name=name):
                    dataset = copy.deepcopy(original)
                    changed = dataset if level == "dataset" else dataset["rows"][0]
                    if name == "extra":
                        changed[name] = False
                    else:
                        del changed[name]
                    if name != "dataset_sha256" or level != "dataset":
                        bind_dataset(dataset)
                    with self.assertRaises(ValueError):
                        self.module.validate_training(dataset)

    def test_validate_rejects_wrong_kind_version_source_and_special_ids(self):
        cases = {
            "kind": ("LORA_TRAINING_INPUTS", None, 1),
            "version": (0, 2, True, 1.0, "1"),
            "supervision_sha256": ("0" * 64, None),
            "eos_token_id": (-1, True, 2.0, "2", None),
            "pad_token_id": (-1, False, 0.0, "0", None),
        }
        for name, values in cases.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    dataset = synthetic_dataset()
                    dataset[name] = value
                    bind_dataset(dataset)
                    with self.assertRaises(ValueError):
                        self.module.validate_training(dataset)

    def test_validate_rejects_invalid_hashes(self):
        for name in ("tokenizer_state_sha256", "messages_sha256", "answer_sha256",
                     "dataset_sha256"):
            for value in (None, 123, "", "a" * 63, "a" * 65, "g" * 64, "A" * 64):
                with self.subTest(name=name, value=value):
                    dataset = synthetic_dataset()
                    owner = dataset["rows"][0] if name in (
                        "messages_sha256", "answer_sha256") else dataset
                    owner[name] = value
                    if name != "dataset_sha256":
                        bind_dataset(dataset)
                    with self.assertRaises(ValueError):
                        self.module.validate_training(dataset)

    def test_validate_rejects_missing_reordered_duplicate_or_nonsupervised_rows(self):
        for mode in ("missing", "extra", "reordered", "duplicate", "development",
                     "nonstring"):
            with self.subTest(mode=mode):
                dataset = synthetic_dataset()
                rows = dataset["rows"]
                if mode == "missing":
                    rows.pop()
                elif mode == "extra":
                    rows.append(copy.deepcopy(rows[-1]))
                elif mode == "reordered":
                    rows.reverse()
                elif mode == "duplicate":
                    rows[1] = copy.deepcopy(rows[0])
                else:
                    rows[0]["task_id"] = "D01" if mode == "development" else 1
                bind_dataset(dataset)
                with self.assertRaises(ValueError):
                    self.module.validate_training(dataset)

    def test_validate_rejects_nonpositive_noninteger_and_oversized_counts(self):
        for name in ("prompt_tokens", "target_tokens"):
            for value in (0, -1, True, 3.0, "3", None, 4097):
                with self.subTest(name=name, value=value):
                    dataset = synthetic_dataset()
                    dataset["rows"][0][name] = value
                    bind_dataset(dataset)
                    with self.assertRaises(ValueError):
                        self.module.validate_training(dataset)
        for prompt_tokens, target_tokens in ((4097, 256), (4096, 257)):
            with self.subTest(prompt=prompt_tokens, target=target_tokens):
                with self.assertRaises(ValueError):
                    self.module.validate_training(synthetic_dataset(prompt_tokens, target_tokens))

    def test_validate_rejects_length_mismatches(self):
        for name in ("input_ids", "attention_mask", "labels"):
            for delta in (-1, 1):
                with self.subTest(name=name, delta=delta):
                    dataset = synthetic_dataset()
                    values = dataset["rows"][0][name]
                    if delta < 0:
                        values.pop()
                    else:
                        values.append(values[-1])
                    bind_dataset(dataset)
                    with self.assertRaises(ValueError):
                        self.module.validate_training(dataset)

    def test_validate_rejects_noninteger_or_negative_input_ids(self):
        for index in (0, 3):
            for value in (True, 10.0, "10", -1, -100, None):
                with self.subTest(index=index, value=value):
                    dataset = synthetic_dataset()
                    dataset["rows"][0]["input_ids"][index] = value
                    if index == 3:
                        dataset["rows"][0]["labels"][index] = value
                    bind_dataset(dataset)
                    with self.assertRaises(ValueError):
                        self.module.validate_training(dataset)

    def test_validate_rejects_nonunit_or_noninteger_attention(self):
        for value in (0, 2, -100, True, 1.0, "1", None):
            with self.subTest(value=value):
                dataset = synthetic_dataset()
                dataset["rows"][0]["attention_mask"][0] = value
                bind_dataset(dataset)
                with self.assertRaises(ValueError):
                    self.module.validate_training(dataset)

    def test_validate_rejects_unmasked_prompt_and_changed_shifted_or_masked_answer(self):
        for index, value in ((0, 10), (0, -100.0), (3, -100), (3, 2), (3, 11.0),
                             (4, -100), (4, True)):
            with self.subTest(index=index, value=value):
                dataset = synthetic_dataset()
                dataset["rows"][0]["labels"][index] = value
                bind_dataset(dataset)
                with self.assertRaises(ValueError):
                    self.module.validate_training(dataset)
        dataset = synthetic_dataset()
        dataset["rows"][0]["labels"] = [-100, -100, 11, 2, -100]
        bind_dataset(dataset)
        with self.assertRaises(ValueError):
            self.module.validate_training(dataset)

    def test_validate_rejects_missing_duplicate_and_nonterminal_target_eos(self):
        for target_ids in ([11, 11], [2, 2], [2, 11]):
            with self.subTest(target_ids=target_ids):
                dataset = synthetic_dataset()
                dataset["rows"][0]["input_ids"][3:] = target_ids
                dataset["rows"][0]["labels"][3:] = target_ids
                bind_dataset(dataset)
                with self.assertRaises(ValueError):
                    self.module.validate_training(dataset)

    def test_validate_rejects_digest_mismatch_even_for_semantically_valid_change(self):
        dataset = synthetic_dataset()
        dataset["rows"][0]["input_ids"][0] = 12
        with self.assertRaisesRegex(ValueError, "digest"):
            self.module.validate_training(dataset)

    def test_validate_rejects_non_json_native_containers(self):
        for value in (None, [], (), "dataset"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.module.validate_training(value)
        for level in ("rows", "row", "input_ids", "attention_mask", "labels"):
            dataset = synthetic_dataset()
            if level == "rows":
                dataset["rows"] = tuple(dataset["rows"])
            elif level == "row":
                dataset["rows"][0] = list(dataset["rows"][0].items())
            else:
                dataset["rows"][0][level] = tuple(dataset["rows"][0][level])
            bind_dataset(dataset)
            with self.subTest(level=level), self.assertRaises(ValueError):
                self.module.validate_training(dataset)

    def test_validate_rejects_non_native_string_field_names(self):
        class StringAlias(str):
            pass

        for level in ("dataset", "row"):
            with self.subTest(level=level):
                dataset = synthetic_dataset()
                owner = dataset if level == "dataset" else dataset["rows"][0]
                name = "version" if level == "dataset" else "task_id"
                owner[StringAlias(name)] = owner.pop(name)
                bind_dataset(dataset)
                with self.assertRaises(ValueError):
                    self.module.validate_training(dataset)

    def test_prepare_and_validate_do_not_read_files_or_execute_tasks(self):
        with patch("builtins.open", side_effect=AssertionError("unexpected file I/O")):
            with patch.object(Path, "open", side_effect=AssertionError("unexpected path I/O")):
                dataset = self.module.prepare_training(TrainingTokenizer(), self.supervision)
                self.module.validate_training(dataset)
        tree = ast.parse(Path(self.module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name in ("copy", "hashlib", "re")
                                    for alias in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, (
                    "pilot_protocol", "prompt_rendering", "tokenizer_measurement",
                    "validation_contract",
                ))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, ("open", "eval", "exec", "compile"))
