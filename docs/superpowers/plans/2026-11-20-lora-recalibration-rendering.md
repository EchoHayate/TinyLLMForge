# LoRA Recalibration Public Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render the approved public base/text messages and fixed feedback/retry continuation without accepting private oracle objects.

**Architecture:** Add one pure public renderer and its tests. Reuse the old instruction and exact frozen eight-example projection, not the old D/T/R episode enumeration. A following inventory subplan will bind these messages to raw reconstructed receipts and the actual completed renderer source hash; this module does not authenticate feedback or authorize execution.

**Tech Stack:** Existing Python standard library, canonical JSON, unittest.

## Global Constraints

- Approved parent: `docs/superpowers/specs/2026-11-19-lora-recalibration-validation-adapter-design.md`, section 5.1 and relevant section 7 cases.
- Only local JSON/AST, static projection and synthetic tests; no stored-source execution, network, SSH, tokenizer/model load, generation or training.
- Preserve every existing source/data file and all frozen pending/false fields.
- No install, telemetry/bootstrap/flush, branches/worktrees, commit/push or cleanup.
- Work in the existing `/Users/bytedance/dev/TinyLLMForge` checkout; no old session or agent recovery.
- Only create the two named Python files; controller owns plans, ledger and results. Apply_patch edits; no inline code comments or single-letter variables.

### Task 1: Pure Public Renderer

**Binding scope:** Local-only, no network or telemetry, no source execution,
tokenizer/model/remote operations, installs, worktree/branch/commit/push or cleanup.
Preserve all previous files. Only author the two new files named below and private
evidence/report files. Candidate code is guidance subordinate to requirements;
any discovered defect must be corrected with failure evidence.

**Files:**
- Create `experiments/lora_necessity/recalibration_rendering.py`.
- Create `experiments/lora_necessity/test_recalibration_rendering.py`.

**Interfaces:**
- `examples_projection(supervision_raw: bytes) -> list[dict]`: pinned S01–S08 projection.
- `first_round(task: dict, arm: str, supervision_raw: bytes) -> list[dict]`.
- `second_round(task: dict, arm: str, supervision_raw: bytes, branch: str, feedback=None) -> list[dict]`.
- Task is exactly public fields `entry_point/id/initial_source/public_cases/spec`;
  no oracle, metadata, reference or hidden cases. This helper validates public
  shape, not frozen task origin; the inventory builder must supply pinned tasks.
- Feedback is only a public projection: kind visible_tests, P01/P02 cases and
  pass/mismatch status. Rendering it does not prove it came from execution.
- Retry accepts `feedback=None` only and emits `{"kind":"results_withheld"}`.
- Both continuation branches append the exact task initial_source as assistant
  and the identical revision instruction followed by newline and canonical JSON.
- Frozen eight-example projection: 7514 bytes, SHA256
  `e1634ad790ed551484a91fe55db36fc2f74b9615d2e5f273989552ea4f6cb290`.
- Revision exactly:
  `Return the final complete function, even if you believe the first candidate is correct.`
- No answer rewriting, token counting, arbitrary generated-candidate input,
  admission, provenance assertion or executable inventory is added here.

- [x] **Step 1: Write tests before the module exists.**

```python
import ast
import copy
import importlib.util
from pathlib import Path
import unittest

from prompt_rendering import INSTRUCTION, _examples
from validation_contract import digest, encoded, load_frozen, parse_json

if importlib.util.find_spec("recalibration_rendering") is not None:
    import recalibration_rendering as rendering
else:
    rendering = None


ROOT = Path(__file__).resolve().parent
REVISION = "Return the final complete function, even if you believe the first candidate is correct."


class RecalibrationRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.supervision = (ROOT / "data/supervision_v1.json").read_bytes()
        cls.tasks = parse_json((ROOT / "data/recalibration_v2_r1/prompts.json").read_bytes())

    def setUp(self):
        self.assertIsNotNone(rendering, "recalibration rendering not implemented")
        self.task = copy.deepcopy(self.tasks[0])
        self.feedback = {"kind": "visible_tests", "cases": [
            {"id": "P01", "status": "mismatch"}, {"id": "P02", "status": "pass"}]}

    def test_exact_eight_example_projection(self):
        actual = rendering.examples_projection(self.supervision)
        expected = _examples(sorted(load_frozen("supervision_v1.json", self.supervision),
                                    key=lambda task: task["id"]))
        self.assertEqual(encoded(actual), encoded(expected))
        self.assertEqual(len(encoded(actual)), 7514)
        self.assertEqual(digest(actual), "e1634ad790ed551484a91fe55db36fc2f74b9615d2e5f273989552ea4f6cb290")
        self.assertEqual([task["id"] for task in actual],
                         ["S%02d" % position for position in range(1, 9)])

    def test_first_round_exact_bytes_and_information_boundary(self):
        examples = rendering.examples_projection(self.supervision)
        for task in self.tasks:
            for arm in ("base", "text"):
                with self.subTest(task=task["id"], arm=arm):
                    content = INSTRUCTION
                    if arm == "text":
                        content += "\n\nSupervised examples:\n" + encoded(examples).decode("ascii")
                    content += "\n\nCurrent task:\n" + encoded(task).decode("ascii")
                    messages = rendering.first_round(task, arm, self.supervision)
                    self.assertEqual(messages, [{"role": "user", "content": content}])
                    self.assertNotIn("Known initial failure:", content)
                    if arm == "base":
                        self.assertNotIn("Supervised examples:", content)
                    current = content.split("\n\nCurrent task:\n")[1]
                    self.assertEqual(parse_json(current.encode("ascii")), task)

    def test_common_continuation_and_retry_withholds_results(self):
        for arm in ("base", "text"):
            first = rendering.first_round(self.task, arm, self.supervision)
            for branch in ("feedback", "retry"):
                payload = self.feedback if branch == "feedback" else None
                messages = rendering.second_round(
                    self.task, arm, self.supervision, branch, payload)
                expected = self.feedback if branch == "feedback" else {"kind": "results_withheld"}
                self.assertEqual(messages, first + [
                    {"role": "assistant", "content": self.task["initial_source"]},
                    {"role": "user", "content": REVISION + "\n" + encoded(expected).decode("ascii")}])
                self.assertNotIn("Public-test feedback follows", messages[-1]["content"])

    def test_invalid_feedback_is_not_silently_projected(self):
        invalid = [
            {"kind": "visible_tests", "cases": []},
            {**self.feedback, "outcome": "private"},
            {"kind": "visible_tests", "cases": list(reversed(self.feedback["cases"]))},
            {"kind": "visible_tests", "cases": [
                {"id": "H01", "status": "pass"}, self.feedback["cases"][1]]},
            {"kind": "visible_tests", "cases": [
                {"id": "P01", "status": "timeout"}, self.feedback["cases"][1]]},
            {"kind": "visible_tests", "cases": [
                {"id": "P01", "status": "pass", "value": 42}, self.feedback["cases"][1]]},
            {"kind": "results_withheld"}, None,
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                rendering.second_round(self.task, "base", self.supervision, "feedback", payload)
        with self.assertRaises(ValueError):
            rendering.second_round(self.task, "base", self.supervision, "retry", self.feedback)

    def test_reject_unknown_identity_and_private_task_fields(self):
        for name, value in (("id", "D01"), ("id", True), ("entry_point", ""),
                            ("initial_source", "x" * 4097), ("spec", "")):
            changed = {**self.task, name: value}
            with self.subTest(name=name), self.assertRaises(ValueError):
                rendering.first_round(changed, "base", self.supervision)
        for name in ("references", "incorrect_programs", "cases", "metadata"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                rendering.first_round({**self.task, name: []}, "text", self.supervision)
        for arm in ("lora", "lora_text", True):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                rendering.first_round(self.task, arm, self.supervision)
        with self.assertRaises(ValueError):
            rendering.second_round(self.task, "base", self.supervision, "unknown")

    def test_frozen_examples_and_detached_outputs(self):
        with self.assertRaises(ValueError):
            rendering.first_round(self.task, "base", self.supervision + b"\n")
        first = rendering.first_round(self.task, "text", self.supervision)
        second = rendering.second_round(self.task, "text", self.supervision,
                                        "feedback", self.feedback)
        preserved = copy.deepcopy(second)
        first[0]["content"] = "changed"
        self.task["initial_source"] = "changed"
        self.feedback["cases"][0]["status"] = "pass"
        self.assertEqual(second, preserved)
        examples = rendering.examples_projection(self.supervision)
        examples[0]["id"] = "changed"
        self.assertEqual(rendering.examples_projection(self.supervision)[0]["id"], "S01")

    def test_strict_json_types_and_public_case_schema(self):
        class StringSubclass(str):
            pass
        changed = {StringSubclass(name): value for name, value in self.task.items()}
        with self.assertRaises(ValueError):
            rendering.first_round(changed, "base", self.supervision)
        for change in ("hidden", "extra", "wrong_args", "nonfinite", "duplicate"):
            task = copy.deepcopy(self.task)
            if change == "hidden":
                task["public_cases"][0]["id"] = "H01"
            elif change == "extra":
                task["public_cases"][0]["derivation"] = "secret"
            elif change == "wrong_args":
                task["public_cases"][0]["args"] = ()
            elif change == "nonfinite":
                task["public_cases"][0]["args"] = [float("nan")]
            else:
                task["public_cases"][1] = copy.deepcopy(task["public_cases"][0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                rendering.first_round(task, "base", self.supervision)

    def test_no_execution_dependencies(self):
        tree = ast.parse((ROOT / "recalibration_rendering.py").read_text())
        allowed = {"copy", "pilot_protocol", "prompt_rendering",
                   "recalibration_data_contract", "validation_contract"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name in allowed for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"open", "exec", "eval", "compile", "__import__"})


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Capture expected RED.**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_rendering -v
```

Expected: 8 real assertion failures `recalibration rendering not implemented`,
not import errors. Preserve raw output and immediate process exit in the private
evidence directory. Do not run telemetry/bootstrap/flush: user local-only scope
overrides that test skill workflow; existing unittest is the approved fallback.

- [x] **Step 3: Implement the following candidate and repair any demonstrated defect.**

```python
import copy

from pilot_protocol import fields, require
from prompt_rendering import INSTRUCTION, PUBLIC_FIELDS, _examples
from recalibration_data_contract import CASE_IDS, TASK_IDS, _outcome, _plain
from validation_contract import digest, encoded, load_frozen


EXAMPLES_SHA256 = "e1634ad790ed551484a91fe55db36fc2f74b9615d2e5f273989552ea4f6cb290"
REVISION = "Return the final complete function, even if you believe the first candidate is correct."


def examples_projection(supervision_raw):
    supervision = load_frozen("supervision_v1.json", supervision_raw)
    examples = _examples(sorted(supervision, key=lambda task: task["id"]))
    require(len(encoded(examples)) == 7514 and digest(examples) == EXAMPLES_SHA256,
            "frozen examples projection")
    return examples


def _public_task(task):
    _plain(task)
    fields(task, PUBLIC_FIELDS)
    require(type(task["id"]) is str and task["id"] in TASK_IDS, "public task identity")
    for name in ("entry_point", "spec", "initial_source"):
        require(type(task[name]) is str and bool(task[name]), "public task text")
    require(task["entry_point"].isidentifier(), "public entry point")
    require(len(task["initial_source"].encode("utf-8")) <= 4096, "initial source byte limit")
    cases = task["public_cases"]
    require(type(cases) is list and len(cases) == 2, "public case coverage")
    for case, identifier in zip(cases, CASE_IDS[:2]):
        fields(case, ("id", "args", "expected"))
        require(case["id"] == identifier and type(case["args"]) is list, "public case identity")
        _outcome(case["expected"])
    return copy.deepcopy(task)


def first_round(task, arm, supervision_raw):
    public = _public_task(task)
    require(type(arm) is str and arm in ("base", "text"), "public arm")
    examples = examples_projection(supervision_raw)
    content = INSTRUCTION
    if arm == "text":
        content += "\n\nSupervised examples:\n" + encoded(examples).decode("ascii")
    content += "\n\nCurrent task:\n" + encoded(public).decode("ascii")
    return [{"role": "user", "content": content}]


def _feedback(branch, feedback):
    require(type(branch) is str and branch in ("feedback", "retry"), "continuation branch")
    if branch == "retry":
        require(feedback is None, "retry withholds feedback")
        return {"kind": "results_withheld"}
    _plain(feedback)
    fields(feedback, ("kind", "cases"))
    require(feedback["kind"] == "visible_tests", "public feedback kind")
    cases = feedback["cases"]
    require(type(cases) is list and len(cases) == 2, "public feedback coverage")
    for case, identifier in zip(cases, CASE_IDS[:2]):
        fields(case, ("id", "status"))
        require(case["id"] == identifier and type(case["status"]) is str
                and case["status"] in ("pass", "mismatch"), "public feedback identity")
    return copy.deepcopy(feedback)


def second_round(task, arm, supervision_raw, branch, feedback=None):
    messages = first_round(task, arm, supervision_raw)
    payload = _feedback(branch, feedback)
    return messages + [
        {"role": "assistant", "content": task["initial_source"]},
        {"role": "user", "content": REVISION + "\n" + encoded(payload).decode("ascii")},
    ]
```

- [x] **Step 4: Capture GREEN and narrow regression.**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_rendering -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_prompt_rendering test_recalibration_data_contract test_recalibration_validation test_recalibration_rendering -v
```

Expected all pass, zero skips. Read source as text only; no task function execution.
Retain final source/test SHA256 and raw outputs/exits in `task-1-report.md`.
The regression includes focused tests; do not double-count them.

- [x] **Step 5: Independent review and protection check, no commit.**

Fresh reviewer receives requirements, report and actual two-file addition diff.
Review both spec and quality. Do not repeat same-hash suites without a concrete
new concern. Controller compares the final frozen example bytes independently,
checks all baseline paths, source/test hashes, original branch/index/HEAD and
updates the dedicated adapter ledger. Future inventory pins the resulting
renderer implementation, not the untested candidate in this plan.

## Self-Review

The requested public rendering behavior is covered by first/second-round exact
message tests and strict input rejection; the eight-example identity has both
legacy-projection and fixed-size/hash checks. Renderer intentionally does not
accept or validate private execution evidence; that is the next inventory
subplan. No token count, dispatch or LoRA-effect claim follows from rendering.
