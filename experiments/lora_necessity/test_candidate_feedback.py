import copy
import importlib
from pathlib import Path
import unittest

from prompt_rendering import episode_jobs, second_round
from validation_contract import digest, encoded, load_frozen


class CandidateFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("candidate_feedback.py").is_file(),
                        "candidate feedback not implemented")
        self.module = importlib.import_module("candidate_feedback")
        data = Path(__file__).with_name("data")
        self.prompts = (data / "prompts_v1.json").read_bytes()
        self.supervision = (data / "supervision_v1.json").read_bytes()
        self.source = "def integer_sign(number):\n    return 0\n"
        self.attempt = "a" * 32

    def requests(self, source=None, **changes):
        arguments = dict(prompts_raw=self.prompts, task_id="D01", arm="base",
                         candidate=self.source if source is None else source,
                         attempt_id=self.attempt)
        arguments.update(changes)
        return self.module.visible_requests(**arguments)

    def feedback(self, receipts, source=None, **changes):
        arguments = dict(prompts_raw=self.prompts, task_id="D01", arm="base",
                         candidate=self.source if source is None else source,
                         receipts_raw=encoded(receipts), attempt_id=self.attempt)
        arguments.update(changes)
        return self.module.visible_feedback(**arguments)

    def receipts(self):
        return [{"request_sha256": digest(request), "status": "completed",
                 "outcome": {"kind": "return", "value": value}}
                for request, value in zip(self.requests(), (0, -1))]

    def test_valid_source_is_not_executed(self):
        for source in (self.source,
                       "def integer_sign(number):\n    raise RuntimeError('never run')\n",
                       "def integer_sign(number):\n    while True:\n        pass\n",
                       "def integer_sign(number):\n    return __import__('os').getpid()\n"):
            with self.subTest(source=source):
                self.assertTrue(self.module.candidate_valid(
                    self.prompts, "D01", "base", source))

    def test_no_repairs_or_extra_source(self):
        invalid = ("", "```python\n" + self.source + "```", "Explanation\n" + self.source,
                   self.source + "\nprint('not allowed')", "import os\n" + self.source,
                   self.source.replace("return 0", "import os\n    return 0"),
                   self.source + "\ndef helper():\n    return 0\n",
                   self.source.replace("return 0", "def helper():\n        return 0"),
                   self.source.replace("return 0", "class Helper:\n        pass"),
                   self.source.replace("return 0", "break"),
                   self.source.replace("return 0", "await other()"),
                   "@decorator\n" + self.source,
                   self.source.replace("def ", "async def "),
                   self.source.replace("integer_sign", "other"),
                   self.source.replace("number", "value"),
                   self.source.replace("number", "number=0"),
                   self.source.replace("number", "number: int"),
                   self.source.replace("number", "number, /"),
                   self.source.replace("number", "*number"),
                   self.source.replace("):", ") -> int:"),
                   "def integer_sign(number):\n    return (\n")
        for source in invalid:
            with self.subTest(source=source):
                self.assertEqual(self.requests(source), [])
                self.assertEqual(self.feedback([], source), {"kind": "invalid_source"})

    def test_source_size_and_encoding(self):
        padded = self.source + "#" + "x" * (65536 - len(self.source.encode()) - 1)
        self.assertEqual(len(self.requests(padded)), 2)
        for source in (padded + "x", self.source + "\ud800", b"source", None, 3):
            with self.subTest(source=repr(source)[:80]), self.assertRaises(ValueError):
                self.requests(candidate=source)
        self.assertEqual(self.requests(self.source + "\0"), [])
        self.assertEqual(self.requests(self.source + "(" * 1000), [])

    def test_identity_and_frozen_file_validation(self):
        for changes in ({"prompts_raw": self.prompts + b"\n"}, {"task_id": "S01"},
                        {"task_id": "unknown"}, {"arm": "lora"}, {"arm": "other"},
                        {"task_id": None}, {"arm": []}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.requests(**changes)

    def test_attempt_identity(self):
        for attempt in ("", "a" * 31, "a" * 33, "A" * 32, "z" * 32, None, True):
            with self.subTest(attempt=attempt), self.assertRaises(ValueError):
                self.requests(attempt_id=attempt)

    def test_requests_are_answer_free_and_fully_bound(self):
        requests = self.requests()
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0], {
            "version": 1, "round": 1, "attempt_id": self.attempt, "task_id": "D01",
            "arm": "base", "case_id": "P01", "entry_point": "integer_sign",
            "source": self.source, "args": [0],
        })
        self.assertEqual(requests[1]["case_id"], "P02")
        self.assertEqual(requests[1]["args"], [-4])
        self.assertEqual(self.requests(), requests)
        requests[0]["args"].append("changed")
        self.assertEqual(self.requests()[0]["args"], [0])

    def test_all_public_episode_signatures_without_execution(self):
        tasks = {task["id"]: task for task in load_frozen("prompts_v1.json", self.prompts)}
        jobs = episode_jobs(self.supervision, self.prompts)
        self.assertEqual(len(jobs), 136)
        for job in jobs:
            task = tasks[job["task_id"]]
            with self.subTest(task=job["task_id"], arm=job["arm"]):
                requests = self.requests(task["initial_source"],
                                         task_id=job["task_id"], arm=job["arm"])
                self.assertEqual(len(requests), len(task["public_cases"]))
                self.assertEqual([request["args"] for request in requests],
                                 [case["args"] for case in task["public_cases"]])

    def test_pass_and_second_round_integration(self):
        receipts = self.receipts()
        before = copy.deepcopy(receipts)
        feedback = self.feedback(receipts)
        self.assertEqual(feedback, {"kind": "visible_tests", "cases": [
            {"id": "P01", "status": "pass"}, {"id": "P02", "status": "pass"}]})
        messages = second_round(self.supervision, self.prompts, "D01", "base",
                                self.source, feedback)
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1]["content"], self.source)
        self.assertIn(encoded(feedback).decode(), messages[2]["content"])
        self.assertEqual(receipts, before)

    def test_mismatch_and_bool_are_not_numeric_success(self):
        for value in (True, False, 0.0, "0", None, [], {}, 7):
            receipts = self.receipts()
            receipts[0]["outcome"]["value"] = value
            with self.subTest(value=value):
                self.assertEqual(self.feedback(receipts)["cases"][0]["status"],
                                 "mismatch")

    def test_all_execution_error_statuses(self):
        for status in ("program_error", "timeout", "output_limit"):
            receipts = self.receipts()
            receipts[0] = {"request_sha256": receipts[0]["request_sha256"],
                           "status": status}
            with self.subTest(status=status):
                self.assertEqual(self.feedback(receipts)["cases"][0]["status"], status)

    def test_expected_value_error_is_an_outcome_not_worker_failure(self):
        tasks = load_frozen("prompts_v1.json", self.prompts)
        task = next(task for task in tasks if any(
            case["expected"]["kind"] == "raises" for case in task["public_cases"]))
        requests = self.requests(task["initial_source"], task_id=task["id"])
        receipts = [{"request_sha256": digest(request), "status": "completed",
                     "outcome": case["expected"]}
                    for request, case in zip(requests, task["public_cases"])]
        result = self.feedback(receipts, task["initial_source"], task_id=task["id"])
        self.assertTrue(all(case["status"] == "pass" for case in result["cases"]))

    def test_invalid_source_cannot_claim_tool_execution(self):
        with self.assertRaises(ValueError):
            self.feedback(self.receipts(), "")
        with self.assertRaises(ValueError):
            self.feedback({}, "")

    def test_missing_duplicate_reordered_receipts_rejected(self):
        receipts = self.receipts()
        for changed in ([], receipts[:1], receipts + receipts[:1],
                        receipts[::-1], [receipts[0], receipts[0]], {}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.feedback(changed)

    def test_cross_candidate_attempt_task_arm_binding_rejected(self):
        receipts = self.receipts()
        for changes in ({"candidate": self.source + "\n"},
                        {"attempt_id": "b" * 32}, {"arm": "text"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.feedback(receipts, **changes)
        for field, value in (("task_id", "D02"), ("case_id", "P02"),
                             ("args", [7]), ("entry_point", "other"),
                             ("round", 2), ("version", True)):
            changed = copy.deepcopy(receipts)
            request = self.requests()[0]
            request[field] = value
            changed[0]["request_sha256"] = digest(request)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.feedback(changed)

    def test_untrusted_status_and_extra_fields_rejected(self):
        for status in ("pass", "mismatch", "launch_failed", "disconnected", True, None):
            receipts = self.receipts()
            receipts[0]["status"] = status
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.feedback(receipts)
        for field in ("stdout", "stderr", "expected", "scoring_cases"):
            receipts = self.receipts()
            receipts[0][field] = "not model feedback"
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.feedback(receipts)
        receipts = self.receipts()
        receipts[0]["status"] = "timeout"
        with self.assertRaises(ValueError):
            self.feedback(receipts)

    def test_malformed_outcomes_rejected(self):
        for outcome in ({}, [], {"kind": "return"}, {"kind": "return", "value": 0,
                        "extra": True}, {"kind": "raises", "type": "RuntimeError"},
                        {"kind": "raises", "type": "ValueError", "message": "secret"}):
            receipts = self.receipts()
            receipts[0]["outcome"] = outcome
            with self.subTest(outcome=outcome), self.assertRaises(ValueError):
                self.feedback(receipts)

    def test_raw_receipts_strict_json(self):
        for raw in (b"", b"\xff", b"null", b"[NaN]", b"[1e999]",
                    b'[{"status":"completed","status":"timeout"}]',
                    b" " * 65537, "[]", b"[" * 2000):
            with self.subTest(raw=repr(raw)[:80]), self.assertRaises(ValueError):
                self.feedback([], receipts_raw=raw)


if __name__ == "__main__":
    unittest.main()
