"""Tests for the latent-axis pricing model.

These pin the readings the direction decision rests on, so a later edit cannot
quietly make the token-count axis look better or worse than the measurements say.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import latent_axis_pricing as pricing


class BlockAccountingTests(unittest.TestCase):
    def test_a_2048_token_sequence_pins_nine_blocks_not_eight(self):
        """The decode window pushes it past a block boundary."""
        self.assertEqual(pricing.blocks_per_sequence(2048), 9)

    def test_an_8192_token_sequence_pins_thirty_three_blocks(self):
        self.assertEqual(pricing.blocks_per_sequence(8192), 33)

    def test_wall_batch_matches_the_measured_walls(self):
        """B=70 at L=2048 and B=19 at L=8192 are what the engine actually ran."""
        self.assertEqual(pricing.wall_batch(2048), 70)
        self.assertEqual(pricing.wall_batch(8192), 19)

    def test_removing_bytes_lets_the_same_budget_hold_more_sequences(self):
        self.assertEqual(pricing.wall_batch(8192, byte_fraction=0.5), 38)


class MeasuredFitTests(unittest.TestCase):
    def test_the_model_reproduces_the_measured_l8192_wall_step(self):
        """Measured 39.358 ms at L=8192 B=19; this is the anchored context."""
        got = pricing.step_ms(8192, 19, a_ms_per_seq=0.0417)
        self.assertAlmostEqual(got, 39.36, delta=0.60)

    def test_the_l2048_extrapolation_is_optimistic_and_stays_bounded(self):
        """Measured 40.148 ms; the L=8192-anchored c0 under-predicts by ~3%.

        Pinned deliberately: the bias favours the token-count axis, so it must not
        be allowed to grow unnoticed while that axis is being argued for.
        """
        got = pricing.step_ms(2048, 70, a_ms_per_seq=0.0417)
        self.assertLess(got, 40.148)
        self.assertGreater(got / 40.148, 0.96)

    def test_effective_kv_bandwidth_is_well_short_of_peak(self):
        """43% of peak: byte savings do not convert to time one-for-one."""
        bw = pricing.effective_kv_bandwidth_gb_s()
        self.assertAlmostEqual(bw, 885.0, delta=10.0)
        self.assertLess(bw / pricing.A100_HBM_PEAK_GB_S, 0.5)


class AxisPricingTests(unittest.TestCase):
    def setUp(self):
        self.report = pricing.price(
            pricing.DEFAULT_PROPOSALS, a_ms_per_seq=0.0417
        )
        self.by_label = {
            item["label"]: item for item in self.report["proposals"]
        }

    def test_the_token_axis_beats_the_byte_axis(self):
        """This is the whole argument for latent over quantisation."""
        bytes_only = self.by_label["2x fewer bytes/token, same L"]
        tokens_only = self.by_label["4x fewer tokens (L->2048), same bytes"]
        self.assertGreater(
            tokens_only["gain_over_baseline"], bytes_only["gain_over_baseline"]
        )
        self.assertAlmostEqual(bytes_only["gain_over_baseline"], 1.99, delta=0.05)
        self.assertAlmostEqual(tokens_only["gain_over_baseline"], 3.73, delta=0.08)

    def test_the_two_axes_compound_rather_than_add(self):
        both = self.by_label["4x fewer tokens AND 2x fewer bytes"]
        bytes_only = self.by_label["2x fewer bytes/token, same L"]
        tokens_only = self.by_label["4x fewer tokens (L->2048), same bytes"]
        self.assertAlmostEqual(both["gain_over_baseline"], 6.95, delta=0.15)
        self.assertGreater(
            both["gain_over_baseline"],
            bytes_only["gain_over_baseline"] + tokens_only["gain_over_baseline"] - 1.0,
        )


class CeilingSensitivityTests(unittest.TestCase):
    """`a` is unresolved, and the spread is the decision."""

    def test_the_optimistic_estimate_gives_a_fifty_fold_ceiling(self):
        report = pricing.price(pricing.DEFAULT_PROPOSALS, a_ms_per_seq=0.0417)
        self.assertAlmostEqual(report["ceiling_gain_over_baseline"], 49.7, delta=1.0)

    def test_the_pessimistic_estimate_gives_a_nine_fold_ceiling(self):
        report = pricing.price(pricing.DEFAULT_PROPOSALS, a_ms_per_seq=0.238)
        self.assertAlmostEqual(report["ceiling_gain_over_baseline"], 8.7, delta=0.3)

    def test_gains_are_measured_against_the_measured_wall_not_the_model(self):
        """Otherwise re-estimating `a` would silently move every published ratio."""
        optimistic = pricing.price(pricing.DEFAULT_PROPOSALS, a_ms_per_seq=0.0417)
        pessimistic = pricing.price(pricing.DEFAULT_PROPOSALS, a_ms_per_seq=0.238)
        self.assertEqual(
            optimistic["baseline_throughput_seq_per_s"],
            pessimistic["baseline_throughput_seq_per_s"],
        )

    def test_both_candidates_are_reported_when_a_is_not_supplied(self):
        """Defaulting to one of them would hide the open question."""
        self.assertEqual(len(pricing.A_CANDIDATES), 2)
        self.assertNotIn("default", pricing.A_CANDIDATES)


if __name__ == "__main__":
    unittest.main()
