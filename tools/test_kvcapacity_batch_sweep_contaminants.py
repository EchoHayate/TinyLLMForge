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


class InteriorPeakTests(unittest.TestCase):
    """The wall sweep found an interior throughput optimum at L=2048.

    Throughput rose to 2015 seq/s at B=128 and fell to 1950 seq/s at B=144, while
    the engine refused to co-run B>=160 at all. The first reading called this
    CAPACITY OPEN because another context length was still rising, which hides the
    single most decision-relevant shape a sweep can produce.
    """

    def _contexts(self, pairs, context=2048):
        return {
            context: sweep.throughput_rows(
                [
                    {"context_length": context, "batch": batch, "step_ms": step}
                    for batch, step in pairs
                ]
            )
        }

    def test_an_interior_peak_is_reported_as_bounded_not_open(self):
        # the real numbers: 32/64/96/128/144 on the eager path
        reading = sweep.capacity_reading(
            self._contexts(
                [
                    (32, 48.650),
                    (64, 51.303),
                    (96, 58.329),
                    (128, 63.511),
                    (144, 73.863),
                ]
            )
        )
        self.assertEqual(reading["reading"], "CAPACITY BOUNDED")
        self.assertIn("B=128", reading["detail"])

    def test_an_interior_peak_at_one_context_is_not_hidden_by_another(self):
        contexts = self._contexts(
            [(32, 48.650), (64, 51.303), (96, 58.329), (128, 63.511), (144, 73.863)]
        )
        contexts.update(
            self._contexts(
                [(16, 49.385), (32, 63.530), (40, 71.198)], context=8192
            )
        )
        reading = sweep.capacity_reading(contexts)
        self.assertEqual(reading["reading"], "CAPACITY BOUNDED")

    def test_a_sweep_that_peaks_at_the_wall_stays_open(self):
        reading = sweep.capacity_reading(
            self._contexts([(16, 49.385), (32, 63.530), (40, 71.198)], context=8192)
        )
        self.assertEqual(reading["reading"], "CAPACITY OPEN")

    def test_a_step_that_explodes_after_a_real_gain_is_bounded(self):
        # throughput nearly doubled before collapsing, so the honest reading is
        # that the payoff exists and ends at B=2, not that there is no payoff
        reading = sweep.capacity_reading(
            self._contexts([(1, 40.0), (2, 45.0), (4, 400.0)])
        )
        self.assertEqual(reading["reading"], "CAPACITY BOUNDED")

    def test_a_sweep_that_never_gains_is_dead(self):
        reading = sweep.capacity_reading(
            self._contexts([(1, 10.0), (2, 20.0), (4, 40.0), (8, 80.0)])
        )
        self.assertEqual(reading["reading"], "CAPACITY DEAD")


class EngineIdentityProvenanceTests(unittest.TestCase):
    """The wall sweep wrote every engine identity field as null.

    An artifact that looks complete but records nothing about what the engine
    actually allocated cannot be audited, so the config lookup must survive the
    wrapper layer instead of giving up at the first AttributeError.
    """

    def setUp(self):
        import kvcapacity_step_scaling_worker as worker

        self.worker = worker

    def test_config_is_found_directly_on_the_engine(self):
        class Engine:
            config = object()

        engine = Engine()
        self.assertIs(self.worker.resolve_config_holder(engine), engine)

    def test_config_is_found_behind_the_wrapper(self):
        class Inner:
            config = object()

        class Engine:
            def __init__(self):
                self.llm_engine = Inner()

        engine = Engine()
        self.assertIsInstance(
            self.worker.resolve_config_holder(engine), Inner
        )

    def test_identity_is_populated_through_the_wrapper(self):
        class HFConfig:
            vocab_size = 151936
            num_hidden_layers = 36
            num_key_value_heads = 8
            head_dim = 128

        class Config:
            num_kvcache_blocks = 1024
            kvcache_block_size = 256
            max_model_len = 40960
            max_num_seqs = 148
            max_num_batched_tokens = 40960
            gpu_memory_utilization = 0.85
            enforce_eager = True
            multi_sequence_cuda_graphs = False
            kv_quant_bits = None
            cpu_offload = None
            hf_config = HFConfig()

        class Inner:
            config = Config()

        class Engine:
            def __init__(self):
                self.llm_engine = Inner()

        identity = self.worker._engine_identity(Engine())
        self.assertEqual(identity["max_num_seqs"], 148)
        self.assertEqual(identity["max_model_len"], 40960)
        self.assertEqual(identity["kv_capacity_tokens"], 1024 * 256)
        self.assertIsNotNone(identity["kv_capacity_bytes"])

    def test_a_missing_config_still_does_not_raise(self):
        class Engine:
            pass

        identity = self.worker._engine_identity(Engine())
        self.assertIsNone(identity["max_num_seqs"])


if __name__ == "__main__":
    unittest.main()
