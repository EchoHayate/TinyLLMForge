import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from demo_active_verification import diagnostic


class DiagnosticTests(unittest.TestCase):
    def test_mixed_hand_counted_outcomes_and_costs(self):
        result = diagnostic("mixed")
        report = result["report"]
        self.assertEqual((report["task_count"], report["family_count"]), (2, 2))
        self.assertEqual((report["scheduled_cells"], report["scheduled_records"]), (72, 360))
        self.assertTrue(report["complete"])
        self.assertEqual(report["budget_overruns"], [])
        expected = {
            "keep": (36, 0, 0, 0, 0),
            "direct": (36, 12, 12, 72, 0),
            "rethink": (48, 24, 6, 144, 0),
            "always_verify": (72, 36, 0, 72, 72),
            "active_verify": (48, 18, 6, 120, 48),
        }
        for method, counts in expected.items():
            group = report["by_method"][method]
            observed = (
                group["selection_pass_count"], group["repair_count"], group["regression_count"],
                group["cost"]["model_calls"]["total"], group["cost"]["tool_calls"]["total"],
            )
            self.assertEqual(observed, counts, method)
        self.assertEqual(report["by_method"]["rethink"]["status_counts"]["error"], 12)
        self.assertEqual(report["by_method"]["active_verify"]["tool_failed"], 12)
        self.assertEqual(report["by_method"]["always_verify"]["tool_failed"], 12)
        self.assertEqual(report["overall"]["cost"]["executed_cases"]["total"], 204)
        self.assertIsNone(report["overall"]["cost"]["input_tokens"]["total"])

    def test_ceiling_does_not_claim_incremental_benefit(self):
        result = diagnostic("ceiling")
        report = result["report"]
        for method in ("direct", "rethink", "always_verify", "active_verify"):
            self.assertEqual(report["by_method"][method]["selection_pass_count"], 72)
            self.assertEqual(report["by_method"][method]["regression_count"], 0)
        self.assertEqual(report["by_method"]["keep"]["selection_pass_count"], 36)
        self.assertTrue(result["fixture_controlled_choices"])
        self.assertFalse(report["formal_evaluation_ready"])

    def test_incomplete_preserves_schedule_and_unknown_cost(self):
        result = diagnostic("incomplete")
        report = result["report"]
        self.assertEqual(len(result["records"]), 359)
        self.assertEqual(report["scheduled_records"], 360)
        self.assertEqual(report["overall"]["complete"], 358)
        self.assertEqual(report["overall"]["missing"], 1)
        self.assertEqual(report["overall"]["incomplete"], 1)
        self.assertEqual(report["overall"]["cost"]["model_calls"]["observed"], 406)
        self.assertIsNone(report["overall"]["cost"]["model_calls"]["total"])
        self.assertFalse(report["complete"])

    def test_deterministic_synthetic_only_and_invalid_scenario(self):
        self.assertEqual(diagnostic("mixed"), diagnostic("mixed"))
        result = diagnostic("mixed")
        self.assertEqual(result["report"]["evidence_kind"], "SYNTHETIC_ONLY")
        self.assertFalse(result["report"]["benchmark_eligible"])
        self.assertFalse(result["report"]["real_execution_eligible"])
        with self.assertRaises(ValueError):
            diagnostic("real")

    def test_cli_emits_json_and_rejects_unknown_scenario(self):
        script = Path(__file__).with_name("demo_active_verification.py")
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run(
            [sys.executable, str(script), "--scenario", "ceiling"],
            capture_output=True, text=True, env=environment, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["report"]["scheduled_records"], 360)
        invalid = subprocess.run(
            [sys.executable, str(script), "--scenario", "real"],
            capture_output=True, text=True, env=environment, timeout=20,
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertEqual(invalid.stdout, "")


if __name__ == "__main__":
    unittest.main()
