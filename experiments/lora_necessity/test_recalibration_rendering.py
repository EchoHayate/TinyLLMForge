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
