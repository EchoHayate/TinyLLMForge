"""Regression tests for the two contaminants the first sweep run actually hit.

The first sweep produced a CAPACITY WEAK reading on the default execution path
purely because batch 1 ran the CUDA graph fast path at 12.98 ms against 43.14 ms
at batch 2, which made the first marginal throughput negative and every later
one look like a collapse. It also carried one cell, L=2048 B=96, whose spread was
111.6 ms against a median of 58.4 ms. Neither is a fact about concurrency, so
both must be excluded by construction rather than by the reader noticing.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kvcapacity_batch_sweep_analysis as sweep
from test_kvcapacity_batch_sweep_analysis import make_payload, make_row


class RegimeBoundaryTests(unittest.TestCase):
    def _priced(self, pairs):
        return sweep.throughput_rows(
            [
                {"context_length": 2048, "batch": batch, "step_ms": step}
                for batch, step in pairs
            ]
        )

    def test_the_graph_fast_path_at_batch_one_is_detected(self):
        # the numbers the run actually produced
        rows = self._priced([(1, 12.980), (2, 43.142), (4, 44.843)])
        boundary = sweep.regime_boundary(rows)
        self.assertIsNotNone(boundary)
        self.assertGreater(boundary["ratio"], 3.0)

    def test_a_uniform_eager_sweep_has_no_regime_boundary(self):
        rows = self._priced([(1, 39.111), (2, 40.120), (4, 40.537)])
        self.assertIsNone(sweep.regime_boundary(rows))

    def test_batch_one_is_dropped_only_when_it_is_a_different_path(self):
        graph = self._priced([(1, 12.980), (2, 43.142), (4, 44.843)])
        kept, dropped, _ = sweep.analysable_rows(graph)
        self.assertEqual([row["batch"] for row in kept], [2, 4])
        self.assertIn("fast path", dropped[0]["reason"])

        eager = self._priced([(1, 39.111), (2, 40.120), (4, 40.537)])
        kept, dropped, _ = sweep.analysable_rows(eager)
        self.assertEqual([row["batch"] for row in kept], [1, 2, 4])
        self.assertEqual(dropped, [])

    def test_the_graph_path_reading_is_not_a_fake_collapse(self):
        # Same shape as the real graph-path run: flat from B=2 upwards, plus the
        # batch-1 fast path. Read from B=2 the concurrency story is healthy, so
        # the reading must not be driven by the regime boundary.
        rows = [
            make_row(2048, 1, 12.980),
            make_row(2048, 2, 43.142),
            make_row(2048, 4, 44.843),
            make_row(2048, 8, 45.217),
            make_row(2048, 16, 44.136),
            make_row(2048, 32, 46.427),
            make_row(2048, 64, 52.858),
            make_row(2048, 128, 65.929),
        ]
        report = sweep.build_report(make_payload(rows))
        self.assertNotIn(
            1, [row["batch"] for row in report["contexts"][2048]]
        )
        self.assertNotEqual(report["capacity_reading"]["reading"], "CAPACITY DEAD")
        self.assertIsNotNone(report["regime_boundary"][2048])


class DispersedCellTests(unittest.TestCase):
    def test_the_contaminated_cell_is_kept_out_of_the_fit(self):
        rows = [
            make_row(2048, 1, 39.111),
            make_row(2048, 2, 40.120),
            make_row(2048, 4, 40.537),
            # the real contaminated cell: stdev 111.623 against median 58.402
            make_row(2048, 96, 58.402, stdev_ms=111.623),
            make_row(2048, 128, 67.074, stdev_ms=2.097),
        ]
        report = sweep.build_report(make_payload(rows))
        batches = [row["batch"] for row in report["contexts"][2048]]
        self.assertNotIn(96, batches)
        self.assertIn(128, batches)
        reasons = [
            item["reason"] for item in report["excluded_cells"][2048]
        ]
        self.assertTrue(any("steady" in reason for reason in reasons))

    def test_a_tidy_cell_is_never_excluded(self):
        rows = [make_row(2048, batch, 40.0 + batch * 0.2) for batch in (1, 2, 4)]
        report = sweep.build_report(make_payload(rows))
        self.assertEqual(report["excluded_cells"][2048], [])

    def test_exclusions_are_visible_in_the_rendered_text(self):
        rows = [
            make_row(2048, 1, 39.111),
            make_row(2048, 2, 40.120),
            make_row(2048, 96, 58.402, stdev_ms=111.623),
        ]
        text = sweep.render(sweep.build_report(make_payload(rows)))
        self.assertIn("excluded B=96", text)


if __name__ == "__main__":
    unittest.main()
