"""Tests for the high-concurrency batch sweep analysis.

The sweep exists to answer one question that GATE A left open: does decode
throughput keep rising as concurrency rises? These tests pin the arithmetic and,
more importantly, pin the failure modes we already know about, so a sweep cannot
quietly report an encouraging answer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kvcapacity_batch_sweep_analysis as sweep

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "kvcapacity_batch_sweep_analysis.py")


def make_row(context, batch, step_ms, stdev_ms=0.5, count=24, drift=1.0):
    return {
        "context_length": context,
        "batch": batch,
        "measured": True,
        "prefill_tokens_match": True,
        "step": {"median_ms": step_ms, "stdev_ms": stdev_ms, "count": count},
        "drift": {"ratio": drift},
    }


def make_payload(rows, preregistered=False, grid_spec="2048:1,2,4"):
    return {
        "payload_sha256": "0" * 64,
        "grid_spec": grid_spec,
        "grid_is_preregistered": preregistered,
        "rows": rows,
    }


class ThroughputArithmeticTests(unittest.TestCase):
    def test_throughput_is_sequences_per_second(self):
        rows = sweep.throughput_rows(
            [
                {"context_length": 2048, "batch": 1, "step_ms": 40.0},
                {"context_length": 2048, "batch": 2, "step_ms": 40.0},
            ]
        )
        self.assertAlmostEqual(rows[0]["throughput_seq_per_s"], 25.0)
        self.assertAlmostEqual(rows[1]["throughput_seq_per_s"], 50.0)

    def test_marginal_cost_is_per_added_sequence_not_per_row(self):
        rows = sweep.throughput_rows(
            [
                {"context_length": 2048, "batch": 2, "step_ms": 40.0},
                {"context_length": 2048, "batch": 10, "step_ms": 48.0},
            ]
        )
        # eight sequences added eight milliseconds, so one millisecond each,
        # not the eight a per-row difference would report
        self.assertAlmostEqual(rows[1]["marginal_ms_per_seq"], 1.0)

    def test_first_row_has_no_marginal_entry(self):
        rows = sweep.throughput_rows(
            [{"context_length": 2048, "batch": 1, "step_ms": 40.0}]
        )
        self.assertIsNone(rows[0]["marginal_ms_per_seq"])
        self.assertIsNone(rows[0]["marginal_throughput_per_seq"])


class SaturationTests(unittest.TestCase):
    def _rows(self, pairs):
        return sweep.throughput_rows(
            [
                {"context_length": 2048, "batch": batch, "step_ms": step}
                for batch, step in pairs
            ]
        )

    def test_a_constant_step_is_perfect_scaling_and_not_saturating(self):
        rows = self._rows([(1, 40.0), (2, 40.0), (4, 40.0), (8, 40.0)])
        result = sweep.saturation(rows)
        self.assertFalse(result["saturating"])
        self.assertTrue(result["throughput_still_rising"])
        self.assertAlmostEqual(result["scaling_efficiency"], 1.0)

    def test_a_step_proportional_to_batch_gives_no_throughput_gain(self):
        # step doubles whenever batch doubles: throughput is flat, so holding
        # more sequences is worthless even though nothing got slower
        rows = self._rows([(1, 10.0), (2, 20.0), (4, 40.0), (8, 80.0)])
        result = sweep.saturation(rows)
        self.assertFalse(result["throughput_still_rising"])
        self.assertAlmostEqual(result["scaling_efficiency"], 0.125, places=6)

    def test_throughput_that_turns_over_is_reported_as_not_rising(self):
        rows = self._rows([(1, 40.0), (2, 42.0), (4, 200.0)])
        result = sweep.saturation(rows)
        self.assertFalse(result["throughput_still_rising"])

    def test_peak_before_the_last_batch_is_flagged(self):
        rows = self._rows([(1, 40.0), (2, 41.0), (4, 300.0)])
        result = sweep.saturation(rows)
        self.assertFalse(result["peak_is_largest_batch"])
        self.assertEqual(result["peak_batch"], 2)

    def test_two_points_are_not_enough_to_judge_saturation(self):
        rows = self._rows([(1, 40.0), (2, 41.0)])
        self.assertIsNone(sweep.saturation(rows))


class CapacityReadingTests(unittest.TestCase):
    def _contexts(self, pairs):
        rows = sweep.throughput_rows(
            [
                {"context_length": 2048, "batch": batch, "step_ms": step}
                for batch, step in pairs
            ]
        )
        return {2048: rows}

    def test_flat_step_keeps_the_capacity_argument_open(self):
        reading = sweep.capacity_reading(
            self._contexts([(1, 40.0), (2, 40.5), (4, 41.0), (8, 42.0)])
        )
        self.assertEqual(reading["reading"], "CAPACITY OPEN")

    def test_proportional_step_kills_the_capacity_argument(self):
        # flat throughput is as fatal as falling throughput: the extra
        # sequences are paid for and return nothing
        reading = sweep.capacity_reading(
            self._contexts([(1, 10.0), (2, 20.0), (4, 40.0), (8, 80.0)])
        )
        self.assertEqual(reading["reading"], "CAPACITY DEAD")

    def test_a_step_that_explodes_after_a_gain_is_bounded_at_the_peak(self):
        reading = sweep.capacity_reading(
            self._contexts([(1, 40.0), (2, 45.0), (4, 400.0)])
        )
        self.assertEqual(reading["reading"], "CAPACITY BOUNDED")
        self.assertIn("B=2", reading["detail"])

    def test_a_sweep_too_short_to_judge_is_inconclusive_not_open(self):
        reading = sweep.capacity_reading(
            self._contexts([(1, 40.0), (2, 40.5)])
        )
        self.assertEqual(reading["reading"], "INCONCLUSIVE")


class BatchFitTests(unittest.TestCase):
    def _rows(self, pairs):
        return sweep.throughput_rows(
            [
                {"context_length": 2048, "batch": batch, "step_ms": step}
                for batch, step in pairs
            ]
        )

    def test_a_linear_sweep_recovers_the_per_sequence_cost(self):
        rows = self._rows([(1, 41.0), (2, 42.0), (4, 44.0), (8, 48.0)])
        fit = sweep.fit_batch_terms(rows)
        self.assertAlmostEqual(fit["linear"]["a_ms_per_seq"], 1.0, places=6)
        self.assertAlmostEqual(fit["linear"]["c0_ms"], 40.0, places=6)
        self.assertGreater(fit["linear"]["r_squared"], 0.999)

    def test_a_bending_sweep_is_caught_by_the_quadratic_term(self):
        rows = self._rows(
            [(1, 40.1), (2, 40.4), (4, 41.6), (8, 46.4), (16, 65.6)]
        )
        fit = sweep.fit_batch_terms(rows)
        self.assertGreater(fit["quadratic"]["share_at_max_batch"], 0.3)
        self.assertGreater(
            fit["quadratic"]["r_squared"], fit["linear"]["r_squared"]
        )

    def test_fewer_than_three_batches_cannot_be_fitted(self):
        self.assertIsNone(sweep.fit_batch_terms(self._rows([(1, 40.0), (2, 41.0)])))


class ReportTests(unittest.TestCase):
    def test_report_drops_unmeasured_cells_and_names_the_reason(self):
        rows = [make_row(2048, 1, 40.0), make_row(2048, 2, 41.0)]
        rows.append(
            {
                "context_length": 2048,
                "batch": 256,
                "measured": False,
                "skipped_reason": "resident KV exceeds the device budget",
            }
        )
        report = sweep.build_report(make_payload(rows))
        self.assertEqual(len(report["contexts"][2048]), 2)
        self.assertTrue(
            any(
                "budget" in (cell.get("reason") or "")
                for cell in report["rejected_cells"]
            )
        )

    def test_report_refuses_to_pose_as_gate_a(self):
        report = sweep.build_report(
            make_payload([make_row(2048, 1, 40.0), make_row(2048, 2, 41.0)])
        )
        self.assertIn("not GATE A", report["note"])
        text = sweep.render(report)
        self.assertIn("not GATE A", text)

    def test_multiple_payloads_merge_across_context_processes(self):
        first = make_payload([make_row(2048, 1, 40.0), make_row(2048, 2, 41.0)])
        second = make_payload([make_row(8192, 1, 42.0), make_row(8192, 2, 44.0)])
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index, payload in enumerate((first, second)):
                path = os.path.join(tmp, f"payload{index}.json")
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle)
                paths.append(path)
            merged = sweep.load_payload(paths)
        report = sweep.build_report(merged)
        self.assertEqual(sorted(report["contexts"]), [2048, 8192])


class CommandLineTests(unittest.TestCase):
    def test_tool_runs_without_torch_installed(self):
        rows = [
            make_row(2048, batch, 40.0 + batch * 0.2)
            for batch in (1, 2, 4, 8, 16, 32)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            payload_path = os.path.join(tmp, "payload.json")
            with open(payload_path, "w", encoding="utf-8") as handle:
                json.dump(make_payload(rows), handle)
            json_out = os.path.join(tmp, "report.json")
            result = subprocess.run(
                [
                    sys.executable,
                    TOOL,
                    "--payload",
                    payload_path,
                    "--json-out",
                    json_out,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("batch sweep", result.stdout)
            with open(json_out, "r", encoding="utf-8") as handle:
                report = json.load(handle)
        self.assertEqual(report["kind"], "kvcapacity_batch_sweep_analysis")

    def test_tool_does_not_import_torch(self):
        source = open(TOOL, "r", encoding="utf-8").read()
        self.assertNotIn("import torch", source)


if __name__ == "__main__":
    unittest.main()


class ContaminationTests(unittest.TestCase):
    """A busy neighbour must not be reported as a property of the model."""

    def _payload(self, foreign_share, *, pinned=None):
        rows = [
            {
                "context_length": 8192,
                "batch": batch,
                "measured": True,
                "step": {
                    "median_ms": step,
                    "stdev_ms": 0.4,
                    "drift_ratio": 1.0,
                    "dispersion_ratio": 0.01,
                    "observed_batch": batch,
                },
            }
            for batch, step in ((16, 34.6), (32, 62.3))
        ]
        return {
            "rows": rows,
            "engines": [
                {
                    "context_length": 8192,
                    "identity": {
                        "device_foreign_share": foreign_share,
                        "kv_capacity_tokens": 89856,
                        "kv_blocks_requested": pinned,
                    },
                }
            ],
        }

    def test_a_shared_card_gets_its_own_reading(self):
        report = sweep.build_report(self._payload(0.4479))
        self.assertEqual(report["capacity_reading"]["reading"], "CONTAMINATED")
        self.assertIn("89856", report["capacity_reading"]["detail"])
        self.assertEqual(len(report["contaminated_engines"]), 1)

    def test_the_suppressed_reading_is_kept_for_inspection(self):
        """Hiding the number entirely would make the run unauditable."""
        report = sweep.build_report(self._payload(0.4479))
        self.assertIn("suppressed_reading", report["capacity_reading"])

    def test_an_idle_card_reads_normally(self):
        report = sweep.build_report(self._payload(0.0052))
        self.assertNotEqual(report["capacity_reading"]["reading"], "CONTAMINATED")
        self.assertEqual(report["contaminated_engines"], [])

    def test_unknown_foreign_share_is_not_given_the_benefit_of_the_doubt(self):
        report = sweep.build_report(self._payload(None))
        self.assertEqual(report["capacity_reading"]["reading"], "CONTAMINATED")

    def test_a_pinned_pool_is_immune_to_the_neighbour(self):
        """Pinned pools are served in full or fail loudly, so sharing is moot."""
        report = sweep.build_report(self._payload(0.4479, pinned=1100))
        self.assertNotEqual(report["capacity_reading"]["reading"], "CONTAMINATED")
