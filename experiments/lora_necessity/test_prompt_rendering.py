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
