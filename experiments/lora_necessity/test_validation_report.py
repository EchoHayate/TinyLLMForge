import copy
import importlib
from pathlib import Path
import unittest

from validation_contract import build_schedule, encoded


def synthetic_matrix(plan):
    result = []
    for row in plan["rows"]:
        outcome = copy.deepcopy(row["expected"])
        if row["role"] == "initial" and row["group"] == "scoring":
            outcome = {"kind": "return", "value": "__synthetic_wrong__"}
        if row["witness_prediction"] is not None:
            outcome = copy.deepcopy(row["witness_prediction"])
        result.append({
            "kind": "synthetic", "ordinal": row["ordinal"],
            "request_sha256": row["request_sha256"], "status": "completed",
            "payload": encoded(outcome).decode("ascii"), "cleanup_complete": True,
        })
    return result


class ValidationReportTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("validation_report.py").is_file(),
                        "validation report not implemented")
        self.module = importlib.import_module("validation_report")
        data = Path(__file__).with_name("data")
        self.source = (data / "task_sources_v1.json").read_bytes()
        self.manifest = (data / "manifest_v1.json").read_bytes()
        self.plan = build_schedule(self.source, self.manifest)
        self.rows = synthetic_matrix(self.plan)

    def report(self, rows):
        return self.module.simulate_validation(self.source, self.manifest, encoded(rows))

    def test_full_simulation_never_becomes_real(self):
        report = self.report(self.rows)
        self.assertEqual(report["kind"], "LORA_VALIDATION_SIMULATION")
        self.assertEqual(report["simulated_decision"], "CHECKS_SATISFIED")
        self.assertEqual(report["scheduled"], 1416)
        self.assertEqual(len(report["tasks"]), 44)
        for flag in ("benchmark_eligible", "formal_evaluation_ready",
                     "net_benefit_established", "isolated"):
            self.assertIs(report[flag], False)

    def test_missing_does_not_shrink_denominator(self):
        report = self.report(self.rows[1:])
        self.assertEqual(report["scheduled"], 1416)
        self.assertEqual(report["received"], 1415)
        self.assertEqual(report["rows"][0]["status"], "missing")
        self.assertEqual(report["simulated_decision"], "INCOMPLETE")

    def test_reference_and_witness_failures(self):
        self.rows[0]["payload"] = '{"kind":"return","value":"wrong"}'
        self.assertEqual(self.report(self.rows)["simulated_decision"], "CHECKS_FAILED")
        self.rows = synthetic_matrix(self.plan)
        witness = next(row for row in self.plan["rows"] if row["role"] == "M01"
                       and row["witness_prediction"] is not None)
        self.rows[witness["ordinal"]]["payload"] = encoded(witness["expected"]).decode()
        self.assertEqual(self.report(self.rows)["simulated_decision"], "CHECKS_FAILED")
        self.rows[witness["ordinal"]].update(status="timeout", payload=None)
        report = self.report(self.rows)
        self.assertIsNone(report["rows"][witness["ordinal"]]["expected_match"])
        self.assertEqual(report["simulated_decision"], "INCOMPLETE")

    def test_stop_after_infrastructure_failure(self):
        self.rows[0]["cleanup_complete"] = False
        with self.assertRaises(ValueError):
            self.report(self.rows)
        report = self.report(self.rows[:1])
        self.assertEqual(report["stop_ordinal"], 0)
        self.assertEqual(report["simulated_decision"], "INCOMPLETE")

    def test_metadata_rejects_spoofing_duplicate_and_wrong_order(self):
        for name, value in (("kind", "real"), ("ordinal", True),
                            ("request_sha256", "0" * 64), ("cleanup_complete", 1)):
            changed = copy.deepcopy(self.rows[:1])
            changed[0][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.report(changed)
        for rows in ([self.rows[0], self.rows[0]], self.rows[:2][::-1]):
            with self.assertRaises(ValueError):
                self.report(rows)

    def test_outcomes_are_typed_and_bounded(self):
        for raw in (b'{"kind":"return","value":NaN}', b'{"kind":"return","value":1.0}',
                    b'{"kind":"return","value":0,"passed":true}',
                    b'{"kind":"return","value":0,"value":1}',
                    b'{"kind":"raises","type":"RuntimeError"}', b" " * 65537):
            with self.subTest(raw=raw[:80]), self.assertRaises(ValueError):
                self.module.parse_outcome(raw)
        self.assertNotEqual(encoded(self.module.parse_outcome(
            b'{"kind":"return","value":true}')), encoded(self.module.parse_outcome(
                b'{"kind":"return","value":1}')))
        self.rows[0]["payload"] = '{"passed":true}'
        report = self.report(self.rows)
        self.assertEqual(report["rows"][0]["status"], "protocol_error")
        self.assertIsNone(report["rows"][0]["expected_match"])
