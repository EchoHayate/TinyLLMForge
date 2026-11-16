# LoRA Prompt And Token Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze information-equivalent prompts and test exact per-call token accounting using injected tokenizer doubles, without loading a real tokenizer or model.

**Architecture:** A pure renderer consumes only pinned supervision/public bytes; a separate counting module checks rendered text against tokenizer-produced IDs. Private reference answers enter only the audit function, never the inference-message builder, and all receipts remain explicitly simulated.

**Tech Stack:** Python 3.12 standard library, `unittest`, injected tokenizer protocol; no new dependencies.

**Local implementation:** both tasks completed with independent spec/quality approval
on 2026-11-11. Main combined suite: 48 tests passed; independent whole-component
review also passed. Evidence and closeout status:
`../../../.superpowers/sdd/lora-validation-20261111-progress.md`.
This is not real-tokenizer or runtime admission.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-11-11-lora-validation-design.md`.
- Requires reviewed Task 1 of `2026-11-11-lora-validation-controller.md`; no dependency on execution-report results.
- Work in `/Users/bytedance/dev/TinyLLMForge`; preserve the experimental branch and unrelated work. No branch/worktree creation, commit, push, installation or telemetry.
- Use `apply_patch`; no inline code comments or one-letter identifiers.
- No source execution, actual tokenizer/model load, torch import, subprocess, network, SSH, GPU or training.
- Four arms; development only base/text; 136 episodes, 272 calls maximum.
- 4096 input tokens, 256 output tokens including terminal token; eight whole examples in text/lora_text, identical bytes.
- No dropped task, shortened source, silent truncation, extra retry or threshold change.
- `benchmark_eligible`, `formal_evaluation_ready`, `net_benefit_established`, `isolated` stay false.
- This plan provides simulated token receipts and a fail-closed counting interface, not verified tokenizer identity or actual launch admission.
- Snippets are proposed implementation content. Do not execute or extract them as production files before the corresponding RED step.

## File Map And Interfaces

- Create `experiments/lora_necessity/prompt_rendering.py` and `test_prompt_rendering.py`.
- Create `experiments/lora_necessity/token_budget.py` and `test_token_budget.py`.
- Main appends the exact documentation paragraph at the end after actual verification.
- Shared existing helpers: `pilot_protocol.{ARMS,FLAGS,fields,require}` and
  `validation_contract.{PINS,load_frozen,encoded,digest}`.
- `reference_jobs` from the controller supplies 44 private ID/source pairs to the token audit.
  The renderer must not import or call that function or accept those pairs.

## Task 1: Deterministic Public/Training Message Construction

**Interfaces:**
- `episode_jobs(supervision_raw: bytes, prompts_raw: bytes) -> list[dict]`: 136 records
  with `task_id`, `arm`, `messages`, `examples_sha256`.
- `training_jobs(supervision_raw: bytes) -> list[dict]`: eight records with `task_id`,
  `messages`, `answer`. Training uses the same case fields and structured initial failure.
- `second_round(supervision_raw, prompts_raw, task_id, arm, candidate: str, feedback: dict) -> list[dict]`.
  It reconstructs the first round from pinned inputs, not caller-supplied mutable messages.
- Empty candidate text is allowed for the invalid-source path. A 64 KiB local text
  parsing cap is separate from the 256-token generation limit; no truncation occurs.
- Feedback is either `{"kind":"invalid_source"}` or `{"kind":"visible_tests","cases":[...]}`;
  cases must exactly match the two public case IDs in order, each with `id` and `status`.
- No free-form reason, stdout, expected hidden result, source hash or tool receipt is rendered.

- [x] **1. Add these tests before the source module.**

```python
import importlib
from pathlib import Path
import unittest

from validation_contract import encoded


class PromptRenderingTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("prompt_rendering.py").is_file(),
                        "prompt rendering not implemented")
        self.module = importlib.import_module("prompt_rendering")
        data = Path(__file__).with_name("data")
        self.supervision = (data / "supervision_v1.json").read_bytes()
        self.prompts = (data / "prompts_v1.json").read_bytes()

    def test_counts_and_information_equivalence(self):
        jobs = self.module.episode_jobs(self.supervision, self.prompts)
        self.assertEqual(len(jobs), 136)
        indexed = {(job["task_id"], job["arm"]): job for job in jobs}
        for task_id in ("T01", "R16"):
            self.assertEqual(indexed[task_id, "text"]["messages"],
                             indexed[task_id, "lora_text"]["messages"])
            self.assertEqual(indexed[task_id, "base"]["messages"],
                             indexed[task_id, "lora"]["messages"])
        self.assertNotIn(("D01", "lora"), indexed)
        text = indexed["T01", "text"]["messages"][0]["content"]
        self.assertIn('"id":"S01"', text)
        self.assertIn('"id":"S08"', text)
        for forbidden in ("scoring_cases", "mutants", "structure_notes", "initial_scoring_case"):
            self.assertNotIn(forbidden, text)

    def test_training_retains_own_failure_but_not_other_examples(self):
        jobs = self.module.training_jobs(self.supervision)
        self.assertEqual(len(jobs), 8)
        self.assertIn('"predicted_initial"', jobs[0]["messages"][0]["content"])
        self.assertNotIn('"id":"S02"', jobs[0]["messages"][0]["content"])
        self.assertTrue(jobs[0]["answer"].endswith("\n"))

    def test_second_round_reconstructs_first_and_constrains_feedback(self):
        feedback = {"kind": "visible_tests", "cases": [
            {"id": "P01", "status": "mismatch"}, {"id": "P02", "status": "pass"},
        ]}
        messages = self.module.second_round(
            self.supervision, self.prompts, "T01", "text", "candidate\n", feedback)
        first = next(job for job in self.module.episode_jobs(self.supervision, self.prompts)
                     if job["task_id"] == "T01" and job["arm"] == "text")
        self.assertEqual(messages[0], first["messages"][0])
        self.assertEqual(messages[1], {"role": "assistant", "content": "candidate\n"})
        self.assertIn(encoded(feedback).decode(), messages[2]["content"])
        feedback["cases"][0]["status"] = "launch_failed"
        with self.assertRaises(ValueError):
            self.module.second_round(self.supervision, self.prompts,
                                     "T01", "text", "candidate", feedback)

    def test_invalid_source_feedback_and_malformed_inputs(self):
        messages = self.module.second_round(self.supervision, self.prompts,
                                            "D01", "base", "", {"kind": "invalid_source"})
        self.assertEqual(messages[1]["content"], "")
        for feedback in ({"kind": "invalid_source", "stdout": "secret"},
                         {"kind": "visible_tests", "cases": []},
                         {"kind": "visible_tests", "cases": [
                             {"id": "H01", "status": "pass"},
                             {"id": "P02", "status": "pass"}]}):
            with self.subTest(feedback=feedback), self.assertRaises(ValueError):
                self.module.second_round(self.supervision, self.prompts,
                                         "D01", "base", "", feedback)
        with self.assertRaises(ValueError):
            self.module.episode_jobs(self.supervision + b"\n", self.prompts)
        with self.assertRaises(ValueError):
            self.module.second_round(self.supervision, self.prompts,
                                     "D01", "lora", "", {"kind": "invalid_source"})
```

- [x] **2. Run RED.**

```sh
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 \
  -m unittest discover -s experiments/lora_necessity -p 'test_prompt_rendering.py' -v
```

Expected: explicit `prompt rendering not implemented` failures. Missing prerequisite
`validation_contract` is not valid RED; finish its reviewed Task 1 first.

- [x] **3. Add this complete implementation, freezing the exact instruction strings.**

```python
import copy

from pilot_protocol import ARMS, fields, require
from validation_contract import digest, encoded, load_frozen


INSTRUCTION = (
    "Repair the function to satisfy its specification and public tests. "
    "Return only one complete Python function with the same name and parameters. "
    "Do not include Markdown, explanations, imports, or test code."
)
REVISION = (
    "Public-test feedback follows. Return the final complete function, "
    "even if the first candidate passed."
)
PUBLIC_FIELDS = ("entry_point", "id", "initial_source", "public_cases", "spec")
FEEDBACK_STATUSES = ("pass", "mismatch", "program_error", "timeout", "output_limit")


def _text(value):
    return encoded(value).decode("ascii")


def _public(task):
    return {name: copy.deepcopy(task[name]) for name in PUBLIC_FIELDS}


def _witness(task):
    witness = task["initial_witness"]
    return {name: copy.deepcopy(witness[name]) for name in
            ("public_case_id", "predicted_initial")}


def _examples(supervision):
    return [
        {**_public(task), "initial_witness": _witness(task),
         "reference_source": task["reference_source"]}
        for task in supervision
    ]


def _message(task, examples=None, witness=None):
    content = INSTRUCTION
    if examples is not None:
        content += "\n\nSupervised examples:\n" + _text(examples)
    content += "\n\nCurrent task:\n" + _text(_public(task))
    if witness is not None:
        content += "\n\nKnown initial failure:\n" + _text(witness)
    return [{"role": "user", "content": content}]


def episode_jobs(supervision_raw, prompts_raw):
    supervision = load_frozen("supervision_v1.json", supervision_raw)
    prompts = load_frozen("prompts_v1.json", prompts_raw)
    examples = _examples(sorted(supervision, key=lambda task: task["id"]))
    jobs = []
    for task in sorted(prompts, key=lambda task: task["id"]):
        arms = ARMS[:2] if task["id"].startswith("D") else ARMS
        for arm in arms:
            included = examples if arm in ("text", "lora_text") else None
            jobs.append({
                "task_id": task["id"], "arm": arm,
                "messages": _message(task, included),
                "examples_sha256": None if included is None else digest(included),
            })
    require(len(jobs) == 136, "episode count")
    return jobs


def training_jobs(supervision_raw):
    supervision = load_frozen("supervision_v1.json", supervision_raw)
    return [
        {"task_id": task["id"], "messages": _message(task, witness=_witness(task)),
         "answer": task["reference_source"]}
        for task in sorted(supervision, key=lambda task: task["id"])
    ]


def second_round(supervision_raw, prompts_raw, task_id, arm, candidate, feedback):
    require(type(task_id) is str and type(arm) is str, "episode identity")
    require(type(candidate) is str, "candidate text")
    try:
        require(len(candidate.encode("utf-8")) <= 65536, "candidate text byte cap")
    except UnicodeError as error:
        raise ValueError("candidate encoding") from error
    jobs = episode_jobs(supervision_raw, prompts_raw)
    matches = [job for job in jobs if job["task_id"] == task_id and job["arm"] == arm]
    require(len(matches) == 1, "unknown episode")
    require(type(feedback) is dict, "feedback object")
    if feedback.get("kind") == "invalid_source":
        fields(feedback, ("kind",))
    else:
        fields(feedback, ("kind", "cases"))
        require(feedback["kind"] == "visible_tests", "feedback kind")
        task = next(task for task in load_frozen("prompts_v1.json", prompts_raw)
                    if task["id"] == task_id)
        cases = feedback["cases"]
        require(type(cases) is list and len(cases) == len(task["public_cases"]),
                "public feedback coverage")
        for actual, expected in zip(cases, task["public_cases"]):
            fields(actual, ("id", "status"))
            require(actual["id"] == expected["id"] and
                    type(actual["status"]) is str and
                    actual["status"] in FEEDBACK_STATUSES, "public feedback case")
    return copy.deepcopy(matches[0]["messages"]) + [
        {"role": "assistant", "content": candidate},
        {"role": "user", "content": REVISION + "\n" + _text(feedback)},
    ]
```

- [x] **4. Run GREEN, then review information equivalence.**

Use the RED command again; expected four tests pass. Review complete output messages
as local strings only. Eight demonstrations are not selectively shortened, and private
`reason` fields never appear. Training also retains its structured failure prediction;
this makes the supervision content consistent with the text examples.

## Task 2: Tokenizer-Double Accounting And Fail-Closed Checks

**Interfaces:**
- Tokenizer double: `chat_template: str`, `eos_token_id: int`, `pad_token_id: int`,
  `get_vocab() -> dict[str,int]`, `apply_chat_template(messages, tokenize=...,
  add_generation_prompt=..., enable_thinking=False)`, and
  `__call__(text, add_special_tokens=False, truncation=False) -> {"input_ids":list[int]}`.
- `simulation_identity(tokenizer) -> dict`: captures template/vocab/special IDs only,
  with a synthetic tag. It does not authenticate files or the real Qwen checkpoint.
- `inspect_call(tokenizer, identity, messages) -> dict`: exact render/tokenize comparison,
  input IDs, count, remaining budget and fit flag; no truncation.
- `check_call(...) -> dict`: same inspection, but raises `ValueError` above 4096.
- `inspect_target(tokenizer, identity, messages, answer) -> dict`: full-template prefix
  matching, target through the unique EOS, and separately preserved post-EOS template IDs.
- `audit_simulated(supervision_raw, prompts_raw, answers, tokenizer, identity) -> dict`:
  complete 8 training / 136 first / 44 reference / 136 sampled-second inventory.
- `answers` is the exact, hash-bound output of controller `reference_jobs`; private
  audit only. No `answers` argument exists on inference rendering APIs.

The complete template may have whitespace after EOS. Count output through EOS; record
the remaining template suffix separately, not as generated output. Keep full serialized
sequence length for training-capacity review. This module does not create loss masks.
Missing EOS, multiple EOS, changed prefix or tokenization disagreement fails closed.

- [x] **1. Add this full test file before `token_budget.py`.**

```python
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
```

- [x] **2. Run RED.**

```sh
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 \
  -m unittest discover -s experiments/lora_necessity -p 'test_token_budget.py' -v
```

Expected: explicit missing token-budget assertions, no actual tokenizer import.

- [x] **3. Add the complete accounting implementation.**

```python
import copy
import hashlib

from pilot_protocol import FLAGS, fields, require
from prompt_rendering import episode_jobs, second_round, training_jobs
from validation_contract import PINS, digest, encoded, load_frozen


REFERENCE_JOBS_SHA256 = "2333003d187a1f12618b4505ca3522b7c2a20e3295c48ae835f3e879ed601cf3"


def _ids(values):
    require(type(values) is list and all(type(value) is int and value >= 0
                                       for value in values), "token IDs")
    return list(values)


def simulation_identity(tokenizer):
    template = tokenizer.chat_template
    require(type(template) is str and bool(template), "chat template")
    vocabulary = tokenizer.get_vocab()
    require(type(vocabulary) is dict and bool(vocabulary), "token vocabulary")
    require(all(type(name) is str and type(value) is int and value >= 0
                for name, value in vocabulary.items()), "vocabulary entries")
    for value in (tokenizer.eos_token_id, tokenizer.pad_token_id):
        require(type(value) is int and value in vocabulary.values(), "special token ID")
    return {
        "kind": "SYNTHETIC_TOKENIZER_IDENTITY", "version": 1,
        "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "vocabulary_sha256": digest(vocabulary),
        "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
    }


def _identity(tokenizer, identity):
    require(encoded(identity) == encoded(simulation_identity(tokenizer)),
            "synthetic tokenizer identity changed")


def _messages(messages, completed=False):
    require(type(messages) is list, "message list")
    roles = [message.get("role") if type(message) is dict else None for message in messages]
    allowed = (["user", "assistant"], ["user", "assistant", "user", "assistant"]) \
        if completed else (["user"], ["user", "assistant", "user"])
    require(roles in allowed, "message role sequence")
    for message in messages:
        fields(message, ("role", "content"))
        require(type(message["content"]) is str, "message content")


def _tokenize(tokenizer, messages, generation):
    _messages(messages, completed=not generation)
    copied = copy.deepcopy(messages)
    original = encoded(copied)
    arguments = {"add_generation_prompt": generation, "enable_thinking": False}
    text = tokenizer.apply_chat_template(copied, tokenize=False, **arguments)
    require(type(text) is str, "rendered text")
    direct = _ids(tokenizer.apply_chat_template(copied, tokenize=True, **arguments))
    actual = tokenizer(text, add_special_tokens=False, truncation=False)
    require(type(actual) is dict and "input_ids" in actual, "tokenizer output")
    tokens = _ids(actual["input_ids"])
    require(encoded(copied) == original, "tokenizer mutated messages")
    require(tokens == direct and bool(tokens), "template/tokenization disagreement")
    require(set(tokens).issubset(set(tokenizer.get_vocab().values())),
            "token outside declared vocabulary")
    return text, tokens


def inspect_call(tokenizer, identity, messages):
    _identity(tokenizer, identity)
    text, tokens = _tokenize(tokenizer, messages, True)
    _identity(tokenizer, identity)
    return {
        "kind": "SIMULATED_TOKEN_INPUT", "identity_sha256": digest(identity),
        "messages_sha256": digest(messages),
        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "prompt": text, "input_ids": tokens, "input_tokens": len(tokens),
        "remaining_input_tokens": 4096 - len(tokens), "within_limit": len(tokens) <= 4096,
    }


def check_call(tokenizer, identity, messages):
    row = inspect_call(tokenizer, identity, messages)
    require(row["within_limit"], "input exceeds 4096 tokens")
    return row


def inspect_target(tokenizer, identity, messages, answer):
    require(type(answer) is str and bool(answer), "answer text")
    prefix = inspect_call(tokenizer, identity, messages)
    completed = copy.deepcopy(messages) + [{"role": "assistant", "content": answer}]
    full_text, full_ids = _tokenize(tokenizer, completed, False)
    _identity(tokenizer, identity)
    prefix_ids = prefix["input_ids"]
    require(full_text.startswith(prefix["prompt"]), "answer text prefix mismatch")
    require(full_text[len(prefix["prompt"]):].startswith(answer), "answer text changed")
    require(full_ids[:len(prefix_ids)] == prefix_ids, "answer token prefix mismatch")
    suffix = full_ids[len(prefix_ids):]
    endings = [index for index, value in enumerate(suffix) if value == identity["eos_token_id"]]
    require(len(endings) == 1, "unique answer termination token required")
    endpoint = endings[0] + 1
    target = suffix[:endpoint]
    return {
        "kind": "SIMULATED_TOKEN_TARGET", "identity_sha256": digest(identity),
        "messages_sha256": digest(messages),
        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "full_prompt_sha256": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
        "prompt_tokens": len(prefix_ids), "prompt_within_limit": prefix["within_limit"],
        "full_sequence_tokens": len(full_ids), "full_sequence_ids": full_ids,
        "target_start": len(prefix_ids), "target_ids": target, "target_tokens": len(target),
        "target_within_limit": len(target) <= 256, "template_tail_ids": suffix[endpoint:],
    }


def audit_simulated(supervision_raw, prompts_raw, answers, tokenizer, identity):
    require(digest(answers) == REFERENCE_JOBS_SHA256, "frozen reference projection")
    _identity(tokenizer, identity)
    episodes = episode_jobs(supervision_raw, prompts_raw)
    training = training_jobs(supervision_raw)
    public = {task["id"]: task for task in load_frozen("prompts_v1.json", prompts_raw)}
    first_rows = []
    sample_rows = []
    base_messages = {}
    for job in episodes:
        key = {"task_id": job["task_id"], "arm": job["arm"]}
        first_rows.append({**key, **inspect_call(tokenizer, identity, job["messages"])})
        if job["arm"] == "base":
            base_messages[job["task_id"]] = job["messages"]
        task = public[job["task_id"]]
        feedback = {"kind": "visible_tests", "cases": [
            {"id": case["id"], "status": "mismatch"} for case in task["public_cases"]
        ]}
        messages = second_round(supervision_raw, prompts_raw, job["task_id"], job["arm"],
                                task["initial_source"], feedback)
        sample_rows.append({
            **key, "sample_kind": "INITIAL_SOURCE_SYNTHETIC_FEEDBACK",
            "candidate_sha256": hashlib.sha256(task["initial_source"].encode()).hexdigest(),
            "feedback_sha256": digest(feedback), **inspect_call(tokenizer, identity, messages),
        })
    training_rows = []
    for job in training:
        base_messages[job["task_id"]] = job["messages"]
        training_rows.append({
            "task_id": job["task_id"],
            **inspect_target(tokenizer, identity, job["messages"], job["answer"]),
        })
    reference_rows = [
        {"task_id": job["id"], **inspect_target(
            tokenizer, identity, base_messages[job["id"]], job["reference_source"])}
        for job in answers
    ]
    return {
        "kind": "LORA_TOKEN_BUDGET_SIMULATION", "version": 1,
        "identity": copy.deepcopy(identity),
        "supervision_sha256": PINS["supervision_v1.json"],
        "prompts_sha256": PINS["prompts_v1.json"],
        "references_sha256": REFERENCE_JOBS_SHA256,
        "training": training_rows, "first": first_rows,
        "references": reference_rows, "sampled_second": sample_rows,
        "all_possible_second_rounds_validated": False,
        **{flag: False for flag in FLAGS},
    }
```

- [x] **4. Run GREEN and the full local pilot suite.**

```sh
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 \
  -m unittest discover -s experiments/lora_necessity -p 'test_token_budget.py' -v
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 \
  -m unittest discover -s experiments/lora_necessity -p 'test_*.py' -q
```

Expected: six token tests pass. Character-based fake counts may show many prompts or
answers over budget; that is expected and says nothing about Qwen3-8B feasibility.
Record actual output; never rewrite samples to make a fake tokenizer report all-fit.

- [x] **5. Inspect and test import purity.**

In the existing trusted unittest process, add this complete method to `TokenBudgetTests`
and run it plus the full suite. This executes module imports, not task source strings.

```python
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
```

This final check depends on both controller tasks; Task 1 alone remains enough for the
renderer and token APIs. The static checks are maintenance guards, not a security sandbox.

- [x] **6. Independent review before claiming the component delivered.**

Review template equality, prefix alignment, EOS/tail treatment, exact integer IDs,
no truncation, all 136 episodes and all 44 private reference rows. Verify reference
source never enters inference `messages` except the eight intentionally shared
supervision examples. Mutation of an external tokenizer is rejected by identity checks;
the identity remains self-reported simulation, never authenticated provenance.

The reviewer must distinguish API capability from runtime integration:
`check_call` can reject an overlong actual message, but no generation worker invokes it
yet. Therefore runtime per-call enforcement, real tokenizer correctness, model context
capacity and training loss masks remain unestablished.

## Documentation And Handoff

- [x] After real local tests and review pass, main appends this paragraph to
  `experiments/lora_necessity/README.md` using `apply_patch`:

```markdown
## 提示与 token 预算控制逻辑

`prompt_rendering.py` 从冻结公开资料构造两轮输入及八份训练消息，
保留 text/lora_text 的完整相同示例。`token_budget.py` 用注入的 tokenizer
核对模板、token ID、终止 token 与逐次输入上限；当前只有替身测试和模拟报告。
44 份参考答案仅用于独立长度审计，不进入推理提示。真实 Qwen3-8B 分词、
生成 worker 调用前的预算守卫、训练容量和隔离执行仍需分别验收。
```

- [x] Preserve actual test/review evidence in a new exclusive local directory;
  keep plans unchecked until their actions really run.
- [x] Recheck frozen bytes and original CPU protocol/test hashes; no dataset rewrites.
- [x] Do not produce a real tokenizer receipt or launch scripts as a convenience extension.
- [x] Choose implementation mode at handoff: fresh workers with review per task
  (recommended) or inline TDD in this session. Neither mode resumes old agents or sessions.

## Plan Self-Review

Spec sections 6–7 map to renderer Task 1 and accounting Task 2. Section 8 maps to
boundary/purity tests and independent review. Private references use the controller's
answer-only projection with a checked digest; actual prompt APIs never accept it.
Terminal output includes EOS but not trailing template whitespace; full template
sequence is retained separately. Sampled second-round prompts are explicitly not an
upper bound. Real tokenizer authentication and execution admission remain outside scope.
