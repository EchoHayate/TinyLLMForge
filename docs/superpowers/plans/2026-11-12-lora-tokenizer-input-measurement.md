# LoRA Tokenizer Input And Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the frozen 324-row tokenizer audit inventory and a runtime-independent token measurement component, without loading a tokenizer or admitting a real run.

**Architecture:** The host inventory builder reuses the accepted prompt and private-reference projections. A separate measurement function operates on an injected tokenizer and returns uninterpreted, private measurement evidence rather than a real-run receipt. A future trusted loader and controller must bind those measurements to actual runtime identity, stop rules and cleanup.

**Tech Stack:** Python 3.12 standard library, colocated `unittest`, existing pure rendering and validation helpers.

**Implementation review correction:** T2-001 found that the initial Task 2 snippet
omitted message validation after standalone encoding. The final check below
implements the existing spec's mutation-rejection requirement. Add the regression
in the correction section; the expected final suite count becomes 62, not 61.
Original RED/GREEN logs and the initial failed review remain historical evidence.

**Final review correction:** F-001 independently found that uniqueness of EOS
does not prove it follows the complete answer. The original suffix-only algorithm
can accept an EOS inside the answer and classify remaining answer tokens as
template tail. The complete-answer requirement below governs the correction;
neither prior task approval nor the 62-test suite closes this defect. Preserve
`final-review.md` and `final-findings.json` as the failed whole-component review.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-11-12-lora-tokenizer-admission-design.md`.
- Work in `/Users/bytedance/dev/TinyLLMForge` on `experiments/feedback-active-verification`; entry HEAD is `93b4e8e0dc5174db0426a5da9e29cc360ed331a7`.
- No branch/worktree creation, commit, push, installs, network, telemetry, SSH, tokenizer/model loading, GPU use or task-source execution.
- Use `apply_patch`; no inline code comments or one-letter identifiers. Preserve unrelated untracked files.
- Leave the eight accepted source/test files, the original CPU protocol, frozen data and the five previously sealed documents unchanged.
- Preserve 8 training, 136 first, 44 references and 136 sampled-second rows, in that order. No deduplication.
- Inputs are limited to 4096 tokens; targets to 256 including EOS. No truncation, demonstration removal, mode switching or retry.
- Second-round samples remain `INITIAL_SOURCE_SYNTHETIC_FEEDBACK`, not real feedback or universal bounds.
- Main comparison remains `lora_text` versus `text` with identical complete examples.
- `benchmark_eligible`, `formal_evaluation_ready`, `net_benefit_established`, `isolated` and `all_possible_second_rounds_validated` stay false.
- Planned snippets are not implementation evidence. Write each task's tests first and observe RED before adding its production source.
- The existing keep-local boundary overrides generic skill commit/worktree/bootstrap instructions. No additional test plugin installation or reporting.

---

## Scope And File Map

This is the first independently testable part of the tokenizer-admission spec, not
the entire real-run implementation. Separate the remaining runtime-dependent work
instead of treating a Python object or dictionary as a trusted loaded tokenizer.

| File to create | Responsibility |
| --- | --- |
| `experiments/lora_necessity/tokenizer_inventory.py` | Pinned host-only inventory, minimal private reference projection, deterministic row identities |
| `experiments/lora_necessity/test_tokenizer_inventory.py` | Counts, exact source bindings, privacy, order and mutation tests |
| `experiments/lora_necessity/tokenizer_measurement.py` | Neutral strict input/target accounting, Mapping normalization, state-drift checks |
| `experiments/lora_necessity/test_tokenizer_measurement.py` | Tokenizer doubles, boundaries, corruption, compatibility and import purity |

The two tasks do not modify one another's sources. Execute sequentially, with a
fresh task implementer and spec/quality review before moving to the next task.
Do not resume previously completed agents. Main owns this plan and the new ledger:
`.superpowers/sdd/lora-tokenizer-admission-20261112-progress.md`.

## Entry Verification

- [x] Record branch/HEAD, tracked/index state and hashes of the current sources/data.
  Compare the eight sources and five documents against
  `/tmp/tlf-lora-validation-20261111-DvglhT/final-verification.json`; do not rewrite
  that receipt or its documents.
- [x] Run the existing local baseline:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 \
  -m unittest discover -s experiments/lora_necessity -p 'test_*.py' -v
```

Expected before either task: 48 passing tests. If the workspace has changed, inspect
the difference; do not assume an unexpected count is a new tokenizer failure.
Put new logs in a fresh owned evidence directory, never in the old sealed root.

## Task 1: Frozen Tokenizer Inventory

**Files:** create `tokenizer_inventory.py` and `test_tokenizer_inventory.py` from the map.

**Consumes:**
- `reference_jobs(source_raw: bytes, manifest_raw: bytes) -> list[dict]`.
- `training_jobs(supervision_raw: bytes) -> list[dict]`.
- `episode_jobs(supervision_raw: bytes, prompts_raw: bytes) -> list[dict]`.
- `second_round(supervision_raw, prompts_raw, task_id, arm, candidate, feedback) -> list[dict]`.
- `load_frozen(name, raw)`, `digest(value)`, `encoded(value)` and `PINS` from `validation_contract`.

**Produces:**
- `build_inventory(source_raw: bytes, manifest_raw: bytes, supervision_raw: bytes, prompts_raw: bytes) -> dict`.
- The result has exact top-level fields `kind`, `version`, `data_sha256`,
  `references_sha256`, `rows`, plus the five false flags.
- Every row has `ordinal`, `category`, `task_id`, `arm`, `messages`, `answer`,
  `sample`, `row_sha256`. `arm` is null for training/references. `answer` is null
  for first/sampled-second. `sample` is non-null only for sampled-second rows.
- `row_sha256` hashes the entire row before that field is added. The consumer
  binds a complete inventory using `digest(inventory)` and its frozen raw inputs.
- This function is host-only: private source bytes are used to derive the reference
  projection, not uploaded or handed to a public prompt builder.

- [x] **1. Write the following failing tests in `test_tokenizer_inventory.py`.**

```python
import copy
import importlib
from pathlib import Path
import unittest

from prompt_rendering import episode_jobs, second_round, training_jobs
from validation_contract import digest, encoded, load_frozen, reference_jobs


class TokenizerInventoryTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("tokenizer_inventory.py").is_file(),
                        "tokenizer inventory not implemented")
        self.module = importlib.import_module("tokenizer_inventory")
        directory = Path(__file__).with_name("data")
        self.inputs = tuple((directory / name).read_bytes() for name in (
            "task_sources_v1.json", "manifest_v1.json",
            "supervision_v1.json", "prompts_v1.json",
        ))

    def test_complete_order_and_content_hashes(self):
        inventory = self.module.build_inventory(*self.inputs)
        rows = inventory["rows"]
        self.assertEqual(len(rows), 324)
        self.assertEqual([row["ordinal"] for row in rows], list(range(324)))
        self.assertEqual([row["category"] for row in rows],
                         ["training"] * 8 + ["first"] * 136 +
                         ["references"] * 44 + ["sampled_second"] * 136)
        for row in rows:
            self.assertEqual(set(row), {
                "ordinal", "category", "task_id", "arm", "messages",
                "answer", "sample", "row_sha256",
            })
            self.assertEqual(row["row_sha256"],
                             digest({name: value for name, value in row.items()
                                     if name != "row_sha256"}))
        for name in ("benchmark_eligible", "formal_evaluation_ready",
                     "net_benefit_established", "isolated",
                     "all_possible_second_rounds_validated"):
            self.assertIs(inventory[name], False)

    def test_exact_training_first_and_private_reference_projection(self):
        rows = self.module.build_inventory(*self.inputs)["rows"]
        training = training_jobs(self.inputs[2])
        first = episode_jobs(self.inputs[2], self.inputs[3])
        prompts = {}
        for actual, expected in zip(rows[:8], training):
            self.assertEqual(actual["messages"], expected["messages"])
            self.assertEqual(actual["answer"], expected["answer"])
            self.assertIsNone(actual["arm"])
            prompts[expected["task_id"]] = expected["messages"]
        for actual, expected in zip(rows[8:144], first):
            self.assertEqual(actual["messages"], expected["messages"])
            self.assertEqual(actual["arm"], expected["arm"])
            self.assertIsNone(actual["answer"])
            if expected["arm"] == "base":
                prompts[expected["task_id"]] = expected["messages"]
        for actual, expected in zip(rows[144:188], reference_jobs(*self.inputs[:2])):
            self.assertEqual(actual["task_id"], expected["id"])
            self.assertEqual(actual["answer"], expected["reference_source"])
            self.assertEqual(actual["messages"], prompts[expected["id"]])
            self.assertIsNone(actual["arm"])

    def test_fixed_second_round_and_equal_information(self):
        rows = self.module.build_inventory(*self.inputs)["rows"]
        public = {task["id"]: task for task in
                  load_frozen("prompts_v1.json", self.inputs[3])}
        for row in rows[188:]:
            task = public[row["task_id"]]
            feedback = {"kind": "visible_tests", "cases": [
                {"id": case["id"], "status": "mismatch"}
                for case in task["public_cases"]
            ]}
            expected = second_round(self.inputs[2], self.inputs[3], row["task_id"],
                                    row["arm"], task["initial_source"], feedback)
            self.assertEqual(row["messages"], expected)
            self.assertEqual(row["sample"]["kind"], "INITIAL_SOURCE_SYNTHETIC_FEEDBACK")
            self.assertEqual(row["sample"]["feedback_sha256"], digest(feedback))
            self.assertIsNone(row["answer"])
        indexed = {(row["task_id"], row["arm"]): row for row in rows[8:144]}
        for (task_id, arm), row in indexed.items():
            if arm == "lora_text":
                self.assertEqual(row["messages"], indexed[(task_id, "text")]["messages"])

    def test_frozen_byte_tampering_rejected(self):
        for index in range(4):
            changed = list(self.inputs)
            changed[index] += b"\n"
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.module.build_inventory(*changed)

    def test_minimal_rows_and_no_shared_mutable_payloads(self):
        inventory = self.module.build_inventory(*self.inputs)
        original = copy.deepcopy(inventory)
        self.assertNotIn(b"scoring_cases", encoded(inventory))
        self.assertNotIn(b"witness_prediction", encoded(inventory))
        self.assertNotIn(b"mutants", encoded(inventory))
        inventory["rows"][0]["messages"][0]["content"] = "changed"
        self.assertEqual(inventory["rows"][144:188], original["rows"][144:188])
        self.assertEqual(self.module.build_inventory(*self.inputs), original)
```

- [x] **2. Run RED and retain the failure log.**

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 \
  -m unittest discover -s experiments/lora_necessity -p 'test_tokenizer_inventory.py' -v
```

Expected: five failures with `tokenizer inventory not implemented`, not a dependency
installation or dataset parsing error.

- [x] **3. Add the following implementation in `tokenizer_inventory.py`.**

```python
import copy
import hashlib

from pilot_protocol import FLAGS, require
from prompt_rendering import episode_jobs, second_round, training_jobs
from validation_contract import PINS, digest, load_frozen, reference_jobs


REFERENCE_SHA256 = "2333003d187a1f12618b4505ca3522b7c2a20e3295c48ae835f3e879ed601cf3"


def build_inventory(source_raw, manifest_raw, supervision_raw, prompts_raw):
    references = reference_jobs(source_raw, manifest_raw)
    require(digest(references) == REFERENCE_SHA256, "reference projection changed")
    training = training_jobs(supervision_raw)
    episodes = episode_jobs(supervision_raw, prompts_raw)
    public = {task["id"]: task for task in load_frozen("prompts_v1.json", prompts_raw)}
    rows = []
    base_messages = {}

    def append(category, task_id, arm, messages, answer=None, sample=None):
        row = {
            "ordinal": len(rows), "category": category, "task_id": task_id,
            "arm": arm, "messages": copy.deepcopy(messages), "answer": answer,
            "sample": copy.deepcopy(sample),
        }
        rows.append({**row, "row_sha256": digest(row)})

    for job in training:
        append("training", job["task_id"], None, job["messages"], job["answer"])
        base_messages[job["task_id"]] = job["messages"]
    for job in episodes:
        append("first", job["task_id"], job["arm"], job["messages"])
        if job["arm"] == "base":
            base_messages[job["task_id"]] = job["messages"]
    for job in references:
        append("references", job["id"], None,
               base_messages[job["id"]], job["reference_source"])
    for job in episodes:
        task = public[job["task_id"]]
        candidate = task["initial_source"]
        feedback = {"kind": "visible_tests", "cases": [
            {"id": case["id"], "status": "mismatch"} for case in task["public_cases"]
        ]}
        messages = second_round(supervision_raw, prompts_raw, job["task_id"],
                                job["arm"], candidate, feedback)
        sample = {
            "kind": "INITIAL_SOURCE_SYNTHETIC_FEEDBACK",
            "candidate_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
            "feedback_sha256": digest(feedback),
        }
        append("sampled_second", job["task_id"], job["arm"], messages, sample=sample)
    require(len(training) == 8 and len(episodes) == 136 and
            len(references) == 44 and len(rows) == 324, "inventory size")
    return {
        "kind": "LORA_TOKENIZER_INVENTORY", "version": 1,
        "data_sha256": dict(PINS), "references_sha256": REFERENCE_SHA256,
        "rows": rows, "all_possible_second_rounds_validated": False,
        **{flag: False for flag in FLAGS},
    }
```

- [x] **4. Rerun the targeted command; expect five passing tests.**
- [x] **5. Run the full local suite; expect 53 tests if only this task was added.**
- [x] **6. Review the full added source/test files, not an empty tracked diff.**
  Spec review checks all 324 rows, private projection and equality of text/lora_text.
  Quality review checks aliases, exact pin use and no I/O on import. Preserve source
  and test-log hashes in the new evidence root. Do not commit.

## Task 2: Neutral Token Measurement

**Files:** create `tokenizer_measurement.py` and `test_tokenizer_measurement.py` from the map.

**Consumes:** an injected tokenizer with `chat_template`, `get_vocab()`,
`eos_token_id`, `pad_token_id`, `apply_chat_template(...)` and `__call__(...)`.
No loader, factory, remote client or runtime identity dictionary is accepted.
Consumes exact messages and optional answer from a Task 1 row.

**Produces:**
- `measure(tokenizer, messages: list[dict], answer: str | None = None) -> dict`.
- Exact top-level fields: `kind`, `state_sha256`, `status`, `input`, `target`.
- `kind` is always `TOKENIZER_MEASUREMENT_UNATTESTED`.
- Status is `OK`, `INPUT_LIMIT` or `OUTPUT_LIMIT`; malformed state/template/IDs
  raise `ValueError`. Input overflow returns before target encoding.
- The state hash binds observed template/vocabulary/EOS/PAD, not model revision,
  loaded files, runtime versions, cleanup or provenance.
- The complete result is private evidence. It has no real-run or eligibility flags
  and must never be promoted by simply replacing `kind`.

- [x] **1. Write the following tests in `test_tokenizer_measurement.py`.**

```python
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
```

- [x] **2. Run RED and retain the failure log.**

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 \
  -m unittest discover -s experiments/lora_necessity -p 'test_tokenizer_measurement.py' -v
```

Expected: eight failures with `tokenizer measurement not implemented`. The existing
synthetic tokenizer is a test fixture only; importing it does not load a model.

- [x] **3. Add the following implementation in `tokenizer_measurement.py`.**

```python
import copy
import hashlib
from collections.abc import Mapping

from pilot_protocol import fields, require
from validation_contract import digest, encoded


def _text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _state(tokenizer):
    template = tokenizer.chat_template
    vocabulary = tokenizer.get_vocab()
    require(type(template) is str and bool(template), "chat template")
    require(type(vocabulary) is dict and bool(vocabulary), "token vocabulary")
    require(all(type(name) is str and type(value) is int and value >= 0
                for name, value in vocabulary.items()), "vocabulary entries")
    for value in (tokenizer.eos_token_id, tokenizer.pad_token_id):
        require(type(value) is int and value in vocabulary.values(), "special token ID")
    return {
        "template_sha256": _text_hash(template), "vocabulary_sha256": digest(vocabulary),
        "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
    }, set(vocabulary.values())


def _ids(values, vocabulary):
    require(type(values) is list and bool(values), "nonempty token ID list")
    require(all(type(value) is int and value >= 0 and value in vocabulary
                for value in values), "declared integer token IDs")
    return list(values)


def _render(tokenizer, messages, generation, state, vocabulary, answer_prefix=None):
    require(type(messages) is list, "message list")
    roles = []
    for message in messages:
        fields(message, ("role", "content"))
        require(type(message["role"]) is str and type(message["content"]) is str,
                "message strings")
        roles.append(message["role"])
    allowed = (["user"], ["user", "assistant", "user"]) if generation else (
        ["user", "assistant"], ["user", "assistant", "user", "assistant"])
    require(roles in allowed, "message roles")
    copied = copy.deepcopy(messages)
    original = encoded(copied)
    arguments = {"add_generation_prompt": generation, "enable_thinking": False}
    text = tokenizer.apply_chat_template(copied, tokenize=False, **arguments)
    require(type(text) is str, "rendered text")
    require(encoded(copied) == original, "messages changed")
    require(_state(tokenizer)[0] == state, "tokenizer state changed")
    direct = _ids(tokenizer.apply_chat_template(
        copied, tokenize=True, truncation=False, **arguments), vocabulary)
    require(encoded(copied) == original, "messages changed")
    require(_state(tokenizer)[0] == state, "tokenizer state changed")
    actual = tokenizer(text, add_special_tokens=False, truncation=False)
    require(isinstance(actual, Mapping) and "input_ids" in actual, "tokenizer mapping")
    tokens = _ids(actual["input_ids"], vocabulary)
    require(tokens == direct, "template/tokenization disagreement")
    require(_state(tokenizer)[0] == state, "tokenizer state changed")
    require(encoded(copied) == original, "messages changed")
    if answer_prefix is not None:
        boundary = tokenizer(answer_prefix, add_special_tokens=False, truncation=False)
        require(isinstance(boundary, Mapping) and "input_ids" in boundary, "tokenizer mapping")
        boundary_ids = _ids(boundary["input_ids"], vocabulary)
        require(tokens[:len(boundary_ids)] == boundary_ids,
                "answer boundary token prefix mismatch")
        require(state["eos_token_id"] in tokens[len(boundary_ids):],
                "termination token must follow complete answer")
        require(_state(tokenizer)[0] == state, "tokenizer state changed")
        require(encoded(copied) == original, "messages changed")
    return text, tokens


def measure(tokenizer, messages, answer=None):
    require(answer is None or type(answer) is str and bool(answer), "answer text")
    state, vocabulary = _state(tokenizer)
    prompt, prompt_ids = _render(tokenizer, messages, True, state, vocabulary)
    input_record = {
        "messages_sha256": digest(messages), "prompt": prompt,
        "prompt_sha256": _text_hash(prompt), "input_ids": prompt_ids,
        "input_tokens": len(prompt_ids), "remaining_input_tokens": 4096 - len(prompt_ids),
    }
    result = {
        "kind": "TOKENIZER_MEASUREMENT_UNATTESTED", "state_sha256": digest(state),
        "status": "OK", "input": input_record, "target": None,
    }
    if len(prompt_ids) > 4096:
        return {**result, "status": "INPUT_LIMIT"}
    if answer is None:
        return result
    completed = copy.deepcopy(messages) + [{"role": "assistant", "content": answer}]
    full_text, full_ids = _render(
        tokenizer, completed, False, state, vocabulary, answer_prefix=prompt + answer)
    require(full_text.startswith(prompt), "answer text prefix mismatch")
    require(full_text[len(prompt):].startswith(answer), "answer text changed")
    require(full_ids[:len(prompt_ids)] == prompt_ids, "answer token prefix mismatch")
    suffix = full_ids[len(prompt_ids):]
    endings = [index for index, value in enumerate(suffix)
               if value == state["eos_token_id"]]
    require(len(endings) == 1, "unique answer termination token required")
    endpoint = endings[0] + 1
    target_ids = suffix[:endpoint]
    result["target"] = {
        "answer_sha256": _text_hash(answer), "full_prompt_sha256": _text_hash(full_text),
        "full_sequence_ids": full_ids, "full_sequence_tokens": len(full_ids),
        "target_start": len(prompt_ids), "target_ids": target_ids,
        "target_tokens": len(target_ids), "template_tail_ids": suffix[endpoint:],
    }
    if len(target_ids) > 256:
        result["status"] = "OUTPUT_LIMIT"
    return result
```

- [x] **4. Rerun the targeted command; expect eight passing tests.**
- [x] **5. Run the full local suite; expect 61 tests after both tasks.**
- [x] **6. Review the complete additions and their integration with Task 1.**
  Verify both direct and rendered-text encoding, real Mapping normalization, unchanged
  simulation APIs, state drift and no public output of private target IDs. A passing
  fake measurement cannot produce `LORA_REAL_TOKENIZER_AUDIT`.

### Task 2 Review Correction: Late Message Mutation

- [x] Add this regression to `TokenizerMeasurementTests` before changing the source:

```python
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
```

- [x] Run the new regression against the pre-fix source; expect failure because
  no `ValueError` is raised. Use the Task 2 focused command; expect one failure and
  the eight original tests still passing.
- [x] Add the final `require(encoded(copied) == original, "messages changed")`
  immediately before `_render` returns, as included in the corrected source above.
- [x] Rerun the focused suite (nine tests) and full suite (62 tests), then obtain
  re-review against the correction and original Task 2 requirements.

## Final Review Correction: F-001

The target boundary must be proven in the actual full context. Encode the exact
prompt plus complete original answer with special-token insertion and truncation
disabled; require exact prefix agreement with the full rendered token sequence.
The unique EOS must occur strictly after that prefix. If the boundary is not
preserved by tokenization, reject rather than infer it from bare-answer length.
Keep the public result schema unchanged. The extra tokenizer call must remain
inside the state/retained-message validation lifetime, including a final check
after that call. No new tokenizer API, remote execution or dependency is needed.

- [x] Add regressions for an in-answer EOS and an answer-ending EOS when the
  template omits its own terminal EOS; observe RED before modifying production.
- [x] Implement verified full-context boundary alignment and fail-closed rejection.
- [x] Cover boundary-merge disagreement and message/state changes during the new
  boundary call, retaining T2-001 and all existing budget/schema tests.
- [x] Run focused GREEN and the full suite; preserve fresh logs and hashes.
- [x] Update the earlier source snippet to the corrected implementation.
- [x] Obtain independent whole-component rereview over all four current additions.
  Keep all original failed reviews, logs and source packages unchanged.

F-001 implementation evidence: seven additional test methods (18 expected RED
assertion/subtest failures), then 16 focused and 69 full-suite tests passed.
Main independently reran 69 tests in 5.916s, exit 0. The current implementation
adds a guarded contextual-prefix encoding, not decoding or tokenizer loading.
The original eight-test snippet above is historical initial-task scope; the
T2-001 regression and seven F-001 methods are additive requirements and retained
in `test_tokenizer_measurement.py`. Independent whole-component rereview returned
local spec PASS / quality APPROVED, with F-001 and T2-001 resolved and no remaining
findings (`final-rereview.md`, `final-rereview-findings.json`).

## Combined Acceptance

- [x] Run the complete local suite with the entry command and preserve fresh output.
- [x] Recheck all sealed source/document hashes and frozen data; inspect any drift
  before making a completion claim. Preserve the untracked additions in review scope.
- [x] Inspect ASTs for forbidden imports/calls, inline comments and one-letter names.
- [x] Obtain an independent whole-addition spec/quality review after the task reviews.
  If a reviewer provider fails, retain that infrastructure failure; do not restart
  already accepted implementation or invent approval.
- [x] Main updates only this plan's completed checkboxes and the new tokenizer
  ledger with source hashes, tests, reviewer results and outstanding runtime gates.
  Leave the old sealed README and closeout documents unchanged.

Local acceptance is complete for these two components, after the retained failed
review and focused correction. Main's 69-test run and the final four-file snapshot
are bound in `/tmp/tlf-tokenizer-components-20261112-HhFKy7`; all 35 protected files
remain unchanged. The current spec's status is synchronized without expanding
scope. No source is committed or published. The final receipt is
`final-verification.json`; the original `stage_tools.py closeout` applies only
to the earlier 62-test snapshot and must not be used for this corrected closeout.
Use `closeout_after_f001.py`, which checks the new evidence while retaining the
original failed reviews and old source/test snapshots.

## Spec Coverage And Remaining Delivery Boundaries

| Spec requirement | Delivery |
| --- | --- |
| Frozen 324-row inventory, exact reference projection, equal examples | Task 1 |
| Mapping normalization, input/target limits, EOS/tail, complete second-round messages | Task 2 |
| Import purity, no mutation of old simulation, local regression | Both tasks and combined acceptance |
| File snapshot, exact dependency versions, actual class/mode and trusted instance identity | Separate runtime-identity plan, dependent on Task 2 |
| Stop at first failed row, retain 324 denominators, verify returned rows, private/public report split | Separate audit-controller plan, dependent on Tasks 1–2 and runtime identity |
| `LORA_REAL_TOKENIZER_AUDIT`, cleanup-qualified acceptance and rejection of counterfeit receipts | Audit controller plus launch wrapper, not this neutral measurement |
| One connection/run, 600-second deadline with work cutoff at 540, 60-second cleanup allowance | Separate launch-wrapper plan; no current connection |
| 4 GiB aggregate memory, 8 process/thread tasks, 128 MiB root, 64 MiB result, 1 MiB per stream | Launch-wrapper enforcement and admission probes, not Python collection limits |
| Private transfer authorization, 0700/0600 paths, no credentials in audit process | Launch-wrapper scope and access controls |
| Real remote dependency discovery, loading and all actual counts | Only after local implementation review and explicit concrete launch scope |

These are independent deliverables, not optional omissions. Completing this plan
does not complete the full spec, permit private transfer or make any of the five
admission/budget flags true. The next plan must consume the exact inventory and
measurement interfaces above; it must not duplicate their message-building logic.

## Execution Handoff

Use the established sequential fresh-implementer/reviewer workflow in this checkout.
Do not ask the user to choose a different workspace or resume an old session.
Both tasks and the two review corrections are complete; do not repeat their
implementation or rerun consumed evidence helpers. Start any subsequent work from
the new tokenizer ledger and the separate delivery boundaries above.
Keep the final status `LOCAL_INPUT_MEASUREMENT_ONLY_NOT_REAL_TOKENIZER_RUN` unless
and until a separate trusted runtime audit actually occurs.
