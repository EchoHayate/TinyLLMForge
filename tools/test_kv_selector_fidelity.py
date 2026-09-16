"""Tests for the selector fidelity gate.

These tests do not claim anything about real models. They check that the estimator
computes what it says it computes, and - the part that matters - that the gate is
capable of returning NON_DISCRIMINATIVE, because a gate that always passes is exactly
the failure mode this tool was written to replace.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.kv_selector_fidelity import (  # noqa: E402
    evaluate_layer,
    quest_unit_scores,
    unit_bounds,
    verdict_for_cell,
)
from tools.needle_haystack_variants import build_variant  # noqa: E402


N_Q_HEADS = 4
N_KV_HEADS = 2
DIM = 8
SEQ_LEN = 64
GRAN = 8


def _planted(hot_unit: int, seed: int = 0):
    """Keys are noise except inside `hot_unit`, whose keys point along the group's q.

    The q heads sharing a kv head are built around one base direction, because GQA
    selection is per kv head: if group members disagreed, the "hot" unit would only be
    hot for one of them and the planted ground truth would be ambiguous.
    """
    rng = np.random.default_rng(seed)
    group = N_Q_HEADS // N_KV_HEADS
    q = np.empty((N_Q_HEADS, DIM), dtype=np.float32)
    directions = []
    for g in range(N_KV_HEADS):
        base = rng.normal(size=DIM)
        base /= np.linalg.norm(base)
        directions.append(base)
        for j in range(group):
            q[g * group + j] = (base + 0.05 * rng.normal(size=DIM)).astype(np.float32)
    k = (0.05 * rng.normal(size=(SEQ_LEN, N_KV_HEADS, DIM))).astype(np.float32)
    start = hot_unit * GRAN
    for g in range(N_KV_HEADS):
        k[start:start + GRAN, g, :] = (20.0 * directions[g]).astype(np.float32)
    return q, k


def test_unit_bounds_tile_the_sequence_without_gaps():
    bounds = unit_bounds(70, 32)
    assert bounds.tolist() == [[0, 32], [32, 64], [64, 70]]
    covered = sum(e - s for s, e in bounds)
    assert covered == 70


def test_quest_upper_bound_is_never_below_the_true_score():
    rng = np.random.default_rng(3)
    q = rng.normal(size=DIM)
    keys = rng.normal(size=(GRAN, DIM))
    kmin = keys.min(axis=0)[None, :]
    kmax = keys.max(axis=0)[None, :]
    bound = quest_unit_scores(q, kmin, kmax)[0]
    true_max = (keys @ q).max()
    assert bound >= true_max - 1e-9


def test_oracle_finds_the_planted_mass_and_recency_does_not():
    q, k = _planted(hot_unit=3)
    cell = evaluate_layer(q, k, granularity=GRAN, k_frac=3 / 8, needle_positions=[], seed=1)
    arms = cell["arms"]
    assert arms["oracle_kvhead"]["mass_mean"] > 0.99
    assert arms["quest_per_head"]["mass_mean"] > 0.99
    # Unit 3 is neither the sink nor the tail, so a recency-only arm must miss it.
    assert arms["recency"]["mass_mean"] < 0.05


def test_gate_calls_a_planted_case_discriminative():
    q, k = _planted(hot_unit=3)
    cell = evaluate_layer(q, k, granularity=GRAN, k_frac=3 / 8, needle_positions=[], seed=1)
    verdict = verdict_for_cell(cell)
    assert verdict["discriminative"], verdict["failures"]
    assert cell["arms"]["quest_per_head"]["mass_mean"] > 0.99
    # At this toy scale there are only two kv heads, so the random arm lands on the hot
    # unit for one of them about a sixth of the time and takes half the mass with it.
    # The margin therefore should not be asserted near 1.0 - that would be a test that
    # only passes because of a lucky seed.
    assert verdict["margin_vs_best_bad_arm"] > 0.4


def test_gate_reports_non_discriminative_when_every_arm_is_equivalent():
    """Identical keys make attention uniform, so selection quality cannot matter.

    This is the shape of the failure we hit in production: the measurement looks fine
    per-arm, but it carries no information. The gate must say so instead of passing.
    """
    q = np.ones((N_Q_HEADS, DIM), dtype=np.float32)
    k = np.ones((SEQ_LEN, N_KV_HEADS, DIM), dtype=np.float32)
    cell = evaluate_layer(q, k, granularity=GRAN, k_frac=0.5, needle_positions=[], seed=1)
    verdict = verdict_for_cell(cell)
    assert not verdict["discriminative"]
    assert any("margin" in f for f in verdict["failures"]), verdict["failures"]


def test_gate_reports_non_discriminative_when_a_bad_arm_saturates():
    """Plant the mass in the final unit: recency-only then wins, so k tells us nothing."""
    q, k = _planted(hot_unit=SEQ_LEN // GRAN - 1)
    cell = evaluate_layer(q, k, granularity=GRAN, k_frac=3 / 8, needle_positions=[], seed=1)
    verdict = verdict_for_cell(cell)
    assert not verdict["discriminative"]
    assert any(f.startswith("saturated_bad_arm") for f in verdict["failures"]), verdict["failures"]


def test_oracle_is_never_below_the_estimator():
    for seed in range(4):
        q, k = _planted(hot_unit=2 + (seed % 3), seed=seed)
        cell = evaluate_layer(q, k, granularity=GRAN, k_frac=0.25, needle_positions=[], seed=seed)
        assert cell["arms"]["oracle_kvhead"]["mass_mean"] >= cell["arms"]["quest_per_head"]["mass_mean"] - 1e-9


def test_mass_is_monotone_in_k():
    q, k = _planted(hot_unit=3)
    masses = [
        evaluate_layer(q, k, granularity=GRAN, k_frac=f, needle_positions=[], seed=1)["arms"]["uniform_stride"]["mass_mean"]
        for f in (0.25, 0.5, 1.0)
    ]
    assert masses[0] <= masses[1] + 1e-9 <= masses[2] + 1e-9
    assert masses[-1] == pytest.approx(1.0, abs=1e-6)


def test_needle_coverage_sees_a_missed_needle():
    q, k = _planted(hot_unit=3)
    # Needle tokens live in unit 5, which carries no attention mass, so a selector
    # that only chases mass will drop them - and the metric must show that.
    cell = evaluate_layer(q, k, granularity=GRAN, k_frac=3 / 8,
                          needle_positions=list(range(40, 48)), seed=1)
    assert cell["arms"]["quest_per_head"]["needle_coverage_mean"] == pytest.approx(0.0)
    cell_full = evaluate_layer(q, k, granularity=GRAN, k_frac=1.0,
                               needle_positions=list(range(40, 48)), seed=1)
    assert cell_full["arms"]["quest_per_head"]["needle_coverage_mean"] == pytest.approx(1.0)


def test_selected_token_fraction_matches_the_requested_budget():
    q, k = _planted(hot_unit=3)
    cell = evaluate_layer(q, k, granularity=GRAN, k_frac=0.5, needle_positions=[], seed=1)
    assert cell["k_units"] == 4
    assert cell["arms"]["quest_per_head"]["selected_token_frac"] == pytest.approx(0.5)


class _WordTokenizer:
    """Deterministic word-level tokenizer, enough to test prompt construction."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = True):
        ids = []
        for word in text.split():
            ids.append(self.vocab.setdefault(word, len(self.vocab) + 10))
        return ids

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv[i] for i in ids)


def test_distractor_variant_positions_point_at_the_target_needle():
    tok = _WordTokenizer()
    spec = build_variant(tok, "distractor", 4096, depth=0.5, num_decoys=4, seed=7)
    ids = spec["ids"]
    positions = spec["answer_needle_positions"]
    recovered = tok.decode([ids[p] for p in positions])
    assert spec["answer"] in recovered
    assert "harbor" in recovered
    assert spec["num_decoys"] == 4
    # Decoys must not overlap the answer, otherwise coverage would be meaningless.
    assert not (set(positions) & set(spec["decoy_positions"]))
    assert len(spec["decoy_positions"]) > 0


def test_repetitive_variant_has_a_single_needle_and_natural_variant_differs():
    tok = _WordTokenizer()
    rep = build_variant(tok, "repetitive", 2048, seed=1)
    nat = build_variant(tok, "natural", 2048, seed=1)
    assert rep["num_decoys"] == 0 and nat["num_decoys"] == 0
    # The control is meant to be degenerate: its filler has almost no distinct tokens.
    rep_body = set(rep["ids"][:1000])
    nat_body = set(nat["ids"][:1000])
    assert len(rep_body) < len(nat_body) / 3


def test_answer_positions_are_inside_the_prompt():
    tok = _WordTokenizer()
    for variant in ("repetitive", "natural", "distractor"):
        spec = build_variant(tok, variant, 3000, seed=2)
        assert spec["answer_needle_positions"]
        assert max(spec["answer_needle_positions"]) < spec["seq_len"]
        assert min(spec["answer_needle_positions"]) >= 0


def test_verdict_uses_coverage_when_needles_are_present():
    """Coverage is the primary criterion, and it must be able to pass and to fail.

    The mass criterion is deliberately not used here: on real 8192-token dumps a
    sink+recency arm recovers 0.89-0.94 of the mass while keeping none of the answer,
    which is exactly the blindness this criterion exists to remove.
    """
    q, k = _planted(hot_unit=3)
    needle = list(range(3 * GRAN, 3 * GRAN + 4))     # answer sits in the hot unit
    cell = evaluate_layer(q, k, granularity=GRAN, k_frac=3 / 8,
                          needle_positions=needle, seed=1)
    verdict = verdict_for_cell(cell)
    assert verdict["criterion"] == "coverage"
    assert verdict["coverage_quest_per_head"] == pytest.approx(1.0)
    # Two kv heads only, so the random arm lands on the hot unit for one of them and
    # takes coverage 0.5 with it; asserting a larger margin here would be asserting a
    # lucky seed rather than the property under test.
    assert verdict["coverage_margin"] >= 0.5
    assert verdict["discriminative"], verdict["failures"]


def test_verdict_fails_when_a_bad_arm_also_keeps_the_answer():
    """Answer in the recency tail: recency-only keeps it, so k proves nothing."""
    q, k = _planted(hot_unit=3)
    n_units = SEQ_LEN // GRAN
    needle = list(range((n_units - 1) * GRAN, SEQ_LEN))
    cell = evaluate_layer(q, k, granularity=GRAN, k_frac=3 / 8,
                          needle_positions=needle, seed=1)
    verdict = verdict_for_cell(cell)
    assert verdict["criterion"] == "coverage"
    assert not verdict["discriminative"]
    assert any("saturated_bad_arm_coverage" in f or "coverage_margin" in f
               for f in verdict["failures"]), verdict["failures"]


def test_mass_criterion_is_still_reported_as_a_diagnostic():
    q, k = _planted(hot_unit=3)
    needle = list(range(3 * GRAN, 3 * GRAN + 4))
    cell = evaluate_layer(q, k, granularity=GRAN, k_frac=3 / 8,
                          needle_positions=needle, seed=1)
    verdict = verdict_for_cell(cell)
    assert "mass_is_discriminative" in verdict
    assert "margin_vs_best_bad_arm" in verdict
