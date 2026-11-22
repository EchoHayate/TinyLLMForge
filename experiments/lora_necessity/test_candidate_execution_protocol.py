import copy
import importlib
from pathlib import Path
import unittest

from candidate_feedback import visible_feedback, visible_requests
from validation_contract import digest, encoded, parse_json


class CandidateExecutionProtocolTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(
            Path(__file__).with_name("candidate_execution_protocol.py").is_file(),
            "candidate execution protocol not implemented",
        )
        self.module = importlib.import_module("candidate_execution_protocol")
        self.source = "def example(value):\n    return value\n"
        self.request = {
            "version": 1, "round": 1, "attempt_id": "a" * 32, "task_id": "D01",
            "arm": "base", "case_id": "P01", "entry_point": "example",
            "source": self.source, "args": [0],
        }
        self.request_sha256 = digest(self.request)

    def prepare(self, **changes):
        return self.module.prepare_request({**self.request, **changes})

    def receipt(self, **changes):
        facts = {
            "returncode": 0, "ready": True, "input_complete": True,
            "reaped": True, "eof": True, "termination": None,
            "result_raw": b'{"kind":"return","value":0}\n',
        }
        facts.update(changes)
        return self.module.make_receipt(self.request_sha256, **facts)

    def test_exports_and_ready(self):
        self.assertEqual(set(self.module.__all__), {
            "READY", "prepare_request", "InfrastructureError", "make_receipt",
        })
        self.assertEqual(self.module.READY, b"TLF_CANDIDATE_READY_V1\n")
        self.assertTrue(issubclass(self.module.InfrastructureError, RuntimeError))

    def test_prepare_canonical_answer_free_payload_without_mutation(self):
        before = copy.deepcopy(self.request)
        request_sha256, payload = self.prepare()
        self.assertEqual(request_sha256, digest(self.request))
        self.assertEqual(payload, encoded({
            "source": self.source, "entry_point": "example", "args": [0],
        }) + b"\n")
        self.assertEqual(self.request, before)
        self.assertEqual(set(parse_json(payload)), {"source", "entry_point", "args"})

    def test_request_exact_fields_and_builtin_object(self):
        for request in (None, [], {}, dict(self.request, expected=0)):
            with self.subTest(request=request), self.assertRaises(ValueError):
                self.module.prepare_request(request)
        for name in self.request:
            changed = dict(self.request)
            del changed[name]
            with self.subTest(missing=name), self.assertRaises(ValueError):
                self.module.prepare_request(changed)

    def test_version_round_strict_integer_one(self):
        for name in ("version", "round"):
            for value in (True, False, 1.0, "1", None, 0, 2):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.prepare(**{name: value})

    def test_identity_ranges_and_arms(self):
        for prefix, count in (("D", 4), ("T", 16), ("R", 16)):
            for ordinal in range(1, count + 1):
                for arm in ("base", "text", "lora", "lora_text"):
                    with self.subTest(prefix=prefix, ordinal=ordinal, arm=arm):
                        if prefix == "D" and arm.startswith("lora"):
                            with self.assertRaises(ValueError):
                                self.prepare(task_id=f"{prefix}{ordinal:02d}", arm=arm)
                        else:
                            self.prepare(task_id=f"{prefix}{ordinal:02d}", arm=arm)
        for task_id in ("D00", "D05", "D1", "T00", "T17", "R00", "R17",
                        "S01", "D01\n", "", None, True):
            with self.subTest(task_id=task_id), self.assertRaises(ValueError):
                self.prepare(task_id=task_id)
        for arm in ("other", "", None, [], True):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                self.prepare(arm=arm)

    def test_case_and_attempt_identity(self):
        self.prepare(case_id="P02", attempt_id="0123456789abcdef" * 2)
        for case_id in ("P00", "P03", "S01", "P1", "P01\n", None, True):
            with self.subTest(case_id=case_id), self.assertRaises(ValueError):
                self.prepare(case_id=case_id)
        for attempt_id in ("a" * 31, "a" * 33, "A" * 32, "z" * 32,
                           "a" * 32 + "\n", None, True):
            with self.subTest(attempt_id=attempt_id), self.assertRaises(ValueError):
                self.prepare(attempt_id=attempt_id)

    def test_every_request_field_is_digest_bound(self):
        for changes in ({"arm": "text"}, {"case_id": "P02"}, {"task_id": "D02"},
                        {"attempt_id": "b" * 32}, {"source": self.source + "\n"},
                        {"args": [1]},
                        {"entry_point": "other",
                         "source": self.source.replace("example", "other")}):
            with self.subTest(changes=changes):
                self.assertNotEqual(self.prepare(**changes)[0], self.request_sha256)

    def test_source_validation_never_runs_function(self):
        for source in (
            "def example(value):\n    raise RuntimeError('not executed')\n",
            "def example(value):\n    while True:\n        pass\n",
            "def example(*values, **options):\n    return values\n",
        ):
            with self.subTest(source=source):
                self.prepare(source=source)

    def test_reject_positional_defaults_without_evaluation(self):
        for signature in ("value=0", "value=None", "value=1 / 0", "value=0, /"):
            with self.subTest(signature=signature), self.assertRaises(ValueError):
                self.prepare(source=f"def example({signature}):\n    return value\n")

    def test_reject_keyword_only_defaults_without_evaluation(self):
        for signature in ("*, value=0", "*, value=None", "*, value=1 / 0",
                          "*, required, value=0"):
            with self.subTest(signature=signature), self.assertRaises(ValueError):
                self.prepare(source=f"def example({signature}):\n    return value\n")

    def test_reject_all_argument_annotations_without_evaluation(self):
        for annotation in ("int", "None", "1 / 0"):
            for signature in (f"value: {annotation}", f"value: {annotation}, /",
                              f"*, value: {annotation}", f"*values: {annotation}",
                              f"**options: {annotation}"):
                with self.subTest(signature=signature), self.assertRaises(ValueError):
                    self.prepare(source=f"def example({signature}):\n    return 0\n")

    def test_unannotated_required_arguments_remain_valid(self):
        for signature in ("value, /", "*, value", "value, *, required",
                          "value, /, other, *values, required, **options"):
            source = f"def example({signature}):\n    return 0\n"
            with self.subTest(signature=signature):
                self.assertEqual(parse_json(self.prepare(source=source)[1])["source"],
                                 source)

    def test_reject_invalid_or_multiple_function_syntax(self):
        for source in (
            "", "```python\n" + self.source + "```", "import os\n" + self.source,
            self.source + "\nexample(0)\n",
            self.source + "\ndef other():\n    pass\n",
            self.source.replace("return value", "import os\n    return value"),
            self.source.replace("return value", "def nested():\n        pass"),
            self.source.replace("return value", "class Nested:\n        pass"),
            self.source.replace("return value", "break"),
            self.source.replace("return value", "await other()"),
            self.source.replace("def ", "async def "),
            self.source.replace("example", "other"),
            "@decorator\n" + self.source,
            self.source.replace("):", ") -> int:"),
            self.source + "\0", self.source + "(" * 1000,
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.prepare(source=source)
        for entry_point in ("", "bad.name", "class", None, 1):
            with self.subTest(entry_point=entry_point), self.assertRaises(ValueError):
                self.prepare(entry_point=entry_point)

    def test_source_utf8_byte_boundary(self):
        prefix = self.source + "#"
        source = prefix + "é" * ((65536 - len(prefix.encode())) // 2)
        source += "x" * (65536 - len(source.encode()))
        self.prepare(source=source)
        for source in (source + "x", self.source + "\ud800", b"source", None, 3):
            with self.subTest(source=repr(source)[:60]), self.assertRaises(ValueError):
                self.prepare(source=source)

    def test_args_are_strict_builtin_json_without_coercion(self):
        values = [None, True, False, 1, 1.0, "é", [1], {"nested": [False]}]
        self.assertEqual(parse_json(self.prepare(args=values)[1])["args"], values)
        for args in (None, {}, (), "args", [()], [set()], [{1: "value"}],
                     [float("nan")], [float("inf")], [float("-inf")], [object()]):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.prepare(args=args)

    def test_payload_exact_byte_boundary_includes_newline(self):
        overhead = len(self.prepare(args=[""])[1])
        args = ["x" * (1048576 - overhead)]
        self.assertEqual(len(self.prepare(args=args)[1]), 1048576)
        with self.assertRaises(ValueError):
            self.prepare(args=[args[0] + "x"])
        with self.assertRaises(ValueError):
            self.prepare(args=["é" * 200000])

    def test_cyclic_deep_and_shared_expansion_requests_rejected(self):
        cycle = []
        cycle.append(cycle)
        deep = []
        for ordinal in range(2000):
            deep = [deep]
        shared = ["x" * 65536]
        for ordinal in range(10):
            shared = [shared, shared]
        for args in (cycle, deep, shared):
            with self.assertRaises(ValueError):
                self.prepare(args=args)

    def test_completed_outcome_and_type_distinctions(self):
        for value in (None, True, False, 0, 0.0, "0", [], {},
                      {"status": "timeout", "request_sha256": "spoofed"}):
            outcome = {"kind": "return", "value": value}
            with self.subTest(value=value):
                receipt = self.receipt(result_raw=encoded(outcome))
                self.assertEqual(receipt, {
                    "request_sha256": self.request_sha256, "status": "completed",
                    "outcome": outcome,
                })
                self.assertIs(type(receipt["outcome"]["value"]), type(value))

    def test_value_error_is_completed_not_program_error(self):
        outcome = {"kind": "raises", "type": "ValueError"}
        self.assertEqual(self.receipt(result_raw=encoded(outcome))["outcome"], outcome)

    def test_abnormal_exit_is_program_error_not_inferred_timeout(self):
        for returncode in (1, 137, -9, -24):
            with self.subTest(returncode=returncode):
                self.assertEqual(self.receipt(returncode=returncode), {
                    "request_sha256": self.request_sha256, "status": "program_error",
                })

    def test_parent_termination_wins_over_late_or_missing_results_without_eof(self):
        for termination in ("timeout", "output_limit"):
            for eof in (True, False):
                for result_raw in (b"", b"malformed", b'{"kind":"return","value":0}',
                                   b"x" * 16385):
                    with self.subTest(termination=termination, eof=eof):
                        self.assertEqual(self.receipt(
                            termination=termination, returncode=-9,
                            eof=eof, result_raw=result_raw,
                        ), {"request_sha256": self.request_sha256,
                            "status": termination})

    def test_initialization_input_reaping_and_eof_failures_are_infrastructure(self):
        for termination in (None, "timeout", "output_limit"):
            for flag in ("ready", "input_complete", "reaped"):
                with self.subTest(termination=termination, flag=flag):
                    with self.assertRaises(self.module.InfrastructureError):
                        self.receipt(termination=termination, **{flag: False})
        for returncode in (0, -9):
            with self.assertRaises(self.module.InfrastructureError):
                self.receipt(eof=False, returncode=returncode)

    def test_parent_fact_types_and_termination_are_validated(self):
        for flag in ("ready", "input_complete", "reaped", "eof"):
            for value in (0, 1, None, "true", []):
                with self.subTest(flag=flag, value=value):
                    with self.assertRaises(self.module.InfrastructureError):
                        self.receipt(**{flag: value})
        for returncode in (None, True, False, 0.0, "0"):
            with self.assertRaises(self.module.InfrastructureError):
                self.receipt(returncode=returncode)
        for termination in ("completed", "program_error", "startup_timeout", True, []):
            with self.assertRaises(self.module.InfrastructureError):
                self.receipt(termination=termination)
        for result_raw in (None, "", bytearray(), {}):
            with self.assertRaises(self.module.InfrastructureError):
                self.receipt(result_raw=result_raw)

    def test_parent_request_digest_is_validated(self):
        for request_sha256 in (None, True, "", "a" * 63, "a" * 65,
                               "A" * 64, "g" * 64, "a" * 64 + "\n"):
            self.request_sha256 = request_sha256
            with self.subTest(digest=request_sha256):
                with self.assertRaises(self.module.InfrastructureError):
                    self.receipt()

    def test_malformed_duplicate_nonfinite_or_multiple_results_are_program_error(self):
        for raw in (
            b"", b"\xff", b"null", b"[]", b"{}", b'{"kind":"return"}',
            b'{"kind":"return","value":NaN}', b'{"kind":"return","value":1e999}',
            b'{"kind":"return","value":0,"value":1}',
            b'{"kind":"return","value":{"nested":0,"nested":1}}',
            b'{"kind":"return","value":0}\n{"kind":"return","value":1}',
            b'{"kind":"return","value":0} trailing',
            b'{"kind":"raises","type":"RuntimeError"}',
            b'{"kind":"raises","type":"ValueError","message":"secret"}',
            b'{"kind":"return","value":0,"status":"completed"}',
            b'{"kind":"return","value":0,"request_sha256":"spoofed"}',
            b'{"status":"completed","outcome":{"kind":"return","value":0}}',
            b'{"kind":"return","value":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}",
        ):
            with self.subTest(raw=repr(raw)[:100]):
                receipt = self.receipt(result_raw=raw)
                self.assertEqual(receipt["status"], "program_error")
                self.assertEqual(receipt, {
                    "request_sha256": self.request_sha256, "status": "program_error",
                })

    def test_result_depth_is_bounded_and_string_brackets_do_not_count(self):
        for depth, status in ((127, "completed"), (128, "program_error")):
            raw = (b'{"kind":"return","value":' + b"[" * depth + b"0"
                   + b"]" * depth + b"}")
            self.assertEqual(self.receipt(result_raw=raw)["status"], status)
        value = ('[{"\\' * 1000) + (']"}' * 1000)
        outcome = {"kind": "return", "value": value}
        self.assertEqual(self.receipt(result_raw=encoded(outcome))["outcome"], outcome)

    def test_ready_prefix_is_not_part_of_result_raw(self):
        self.assertEqual(self.receipt(
            result_raw=self.module.READY + b'{"kind":"return","value":0}',
        )["status"], "program_error")

    def test_result_exact_byte_boundary(self):
        raw = encoded({"kind": "return", "value": ""})
        raw = encoded({"kind": "return", "value": "x" * (16384 - len(raw))})
        self.assertEqual(len(raw), 16384)
        self.assertEqual(self.receipt(result_raw=raw)["status"], "completed")
        self.assertEqual(self.receipt(result_raw=raw + b"\n")["status"], "program_error")

    def test_existing_feedback_integration_uses_synthetic_outcomes_only(self):
        prompts = Path(__file__).with_name("data").joinpath("prompts_v1.json").read_bytes()
        source = "def integer_sign(number):\n    return 0\n"
        requests = visible_requests(prompts, "D01", "base", source,
                                    attempt_id="a" * 32)
        receipts = []
        for request, value in zip(requests, (0, -1)):
            self.request_sha256 = self.module.prepare_request(request)[0]
            receipts.append(self.receipt(result_raw=encoded(
                {"kind": "return", "value": value})))
        feedback = visible_feedback(prompts, "D01", "base", source, encoded(receipts),
                                    attempt_id="a" * 32)
        self.assertEqual(feedback, {"kind": "visible_tests", "cases": [
            {"id": "P01", "status": "pass"}, {"id": "P02", "status": "pass"},
        ]})
        receipts[0]["outcome"]["value"] = False
        feedback = visible_feedback(prompts, "D01", "base", source, encoded(receipts),
                                    attempt_id="a" * 32)
        self.assertEqual(feedback["cases"][0]["status"], "mismatch")


if __name__ == "__main__":
    unittest.main()
