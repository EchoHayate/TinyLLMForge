"""Tests for the end-to-end sparse decode gate.

Two things are checked, and they are different things:

1. the selection math is what it claims to be (engine parity, exactness when nothing is
   dropped, correct handling of the short last unit);
2. the *gate* can fail. A verdict function that always says DISCRIMINATIVE would happily
   bless a broken selector, which is exactly the failure mode the previous needle gate had.

No model is required: the attention path is exercised with a stub module, so these run on
a laptop CPU and cannot silently depend on the GPU box being reachable.
"""

from __future__ import annotations

import math
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.e2e_sparse_attention import (  # noqa: E402
    BAD_ARMS,
    PRIMARY_ARM,
    SparseAttentionController,
    build_verdict,
    quest_scores_per_head,
    quest_scores_shared,
    query_repr,
    select_units,
    token_agreement,
    unit_minmax,
    units_to_token_index,
)


def stub_module(layer_idx: int, groups: int):
    return types.SimpleNamespace(layer_idx=layer_idx, num_key_value_groups=groups,
                                 training=False)


def dense_reference(query, key, value, groups, scaling):
    k = key.repeat_interleave(groups, dim=1)
    v = value.repeat_interleave(groups, dim=1)
    w = torch.softmax((query @ k.transpose(2, 3)) * scaling, dim=-1)
    return (w @ v).transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# selection math
# ---------------------------------------------------------------------------

def test_unit_minmax_matches_naive_loop_with_partial_last_unit():
    torch.manual_seed(0)
    key = torch.randn(1, 3, 70, 8)
    gran = 32
    kmin, kmax = unit_minmax(key, gran)
    assert kmin.shape == (1, 3, 3, 8) and kmax.shape == (1, 3, 3, 8)
    for u, (s, e) in enumerate([(0, 32), (32, 64), (64, 70)]):
        chunk = key[0, :, s:e, :]
        assert torch.allclose(kmin[0, u], chunk.amin(dim=1))
        assert torch.allclose(kmax[0, u], chunk.amax(dim=1))


def test_unit_minmax_padding_does_not_widen_the_short_unit():
    # If padding leaked into the reduction the last unit's bound would span [-inf, +inf]
    # and the selector would think it is infinitely promising.
    key = torch.full((1, 1, 33, 4), 0.5)
    kmin, kmax = unit_minmax(key, 32)
    assert torch.isfinite(kmin).all() and torch.isfinite(kmax).all()
    assert torch.allclose(kmin[0, 1], torch.full((1, 4), 0.5))
    assert torch.allclose(kmax[0, 1], torch.full((1, 4), 0.5))


def test_query_repr_is_group_wise_amax():
    q = torch.tensor([[[[1.0, -2.0]], [[3.0, -1.0]], [[0.0, 5.0]], [[-4.0, 0.0]]]])
    out = query_repr(q, n_kv_heads=2)
    assert out.shape == (1, 2, 2)
    assert torch.allclose(out[0, 0], torch.tensor([3.0, -1.0]))
    assert torch.allclose(out[0, 1], torch.tensor([0.0, 5.0]))


def test_query_repr_rejects_prefill_shapes():
    with pytest.raises(ValueError):
        query_repr(torch.randn(1, 4, 5, 2), n_kv_heads=2)


def test_quest_shared_equals_engine_formula():
    torch.manual_seed(1)
    q_repr = torch.randn(1, 4, 16)
    kmin = torch.randn(1, 7, 4, 16)
    kmax = kmin + torch.rand(1, 7, 4, 16)
    got = quest_scores_shared(q_repr, kmin, kmax)
    # `quest_score_kernel`: score = sum_{h,d} max(q*kmin, q*kmax)
    ref = torch.zeros(1, 7)
    for u in range(7):
        acc = 0.0
        for h in range(4):
            for d in range(16):
                qv = q_repr[0, h, d]
                acc += max(qv * kmin[0, u, h, d], qv * kmax[0, u, h, d])
        ref[0, u] = acc
    assert torch.allclose(got, ref, atol=1e-4)


def test_quest_per_head_sums_only_over_channels():
    q_repr = torch.randn(1, 2, 5)
    kmin = torch.randn(1, 3, 2, 5)
    kmax = kmin + 1.0
    per_head = quest_scores_per_head(q_repr, kmin, kmax)
    shared = quest_scores_shared(q_repr, kmin, kmax)
    assert per_head.shape == (1, 3, 2)
    assert torch.allclose(per_head.sum(dim=2), shared, atol=1e-4)


def test_quest_bound_is_an_upper_bound_on_the_real_score():
    torch.manual_seed(2)
    key = torch.randn(1, 2, 64, 8)
    q = torch.randn(1, 4, 1, 8)
    kmin, kmax = unit_minmax(key, 32)
    q_repr = query_repr(q, 2)
    bound = quest_scores_per_head(q_repr, kmin, kmax)[0]      # [U, KVH]
    for h in range(2):
        for u, (s, e) in enumerate([(0, 32), (32, 64)]):
            real = (q[0, 2 * h : 2 * h + 2, 0, :] @ key[0, h, s:e, :].T).max()
            assert bound[u, h] + 1e-4 >= real


def test_every_arm_keeps_the_last_unit():
    n_units, k_units = 20, 5
    scores = torch.arange(n_units, dtype=torch.float32)
    gen = torch.Generator().manual_seed(0)
    for arm in (PRIMARY_ARM,) + BAD_ARMS:
        picked = select_units(arm, scores_shared=scores, scores_per_head=None,
                              n_units=n_units, k_units=k_units, generator=gen,
                              device=torch.device("cpu"))
        assert (n_units - 1) in set(picked.tolist()), arm


def test_quest_shared_forces_the_sink_unit_even_when_scored_last():
    scores = torch.tensor([-1e9, 5.0, 4.0, 3.0, 2.0])
    picked = select_units(PRIMARY_ARM, scores_shared=scores, scores_per_head=None,
                          n_units=5, k_units=3, generator=None,
                          device=torch.device("cpu")).tolist()
    assert 0 in picked and 4 in picked


def test_per_head_arm_returns_one_index_set_per_head():
    # head 0 wants unit 1, head 1 wants unit 2: if the arm collapsed the heads into one
    # ranking this test would fail, which is the whole point of having the arm.
    per_head = torch.tensor([[1.0, 1.0], [9.0, 0.1], [0.5, 9.0], [0.1, 0.1]])
    picked = select_units("quest_per_head", scores_shared=None, scores_per_head=per_head,
                          n_units=4, k_units=3, generator=None, device=torch.device("cpu"))
    assert isinstance(picked, list) and len(picked) == 2
    assert set(picked[0].tolist()) == {0, 1, 3}
    assert set(picked[1].tolist()) == {0, 2, 3}


def test_units_to_token_index_drops_positions_past_the_end():
    idx = units_to_token_index(torch.tensor([0, 2]), granularity=32, seq_len=70)
    assert idx.tolist() == list(range(0, 32)) + list(range(64, 70))


# ---------------------------------------------------------------------------
# attention path
# ---------------------------------------------------------------------------

def test_full_budget_sparse_equals_dense():
    torch.manual_seed(3)
    b, kvh, groups, s, d = 1, 2, 2, 64, 8
    key = torch.randn(b, kvh, s, d)
    value = torch.randn(b, kvh, s, d)
    query = torch.randn(b, kvh * groups, 1, d)
    scaling = 1.0 / math.sqrt(d)
    module = stub_module(layer_idx=1, groups=groups)

    ctl = SparseAttentionController(PRIMARY_ARM, 32, 1.0, dense_layers=())
    ctl.reset(torch.device("cpu"))
    got, _ = ctl(module, query, key, value, None, scaling)
    want = dense_reference(query, key, value, groups, scaling)
    assert torch.allclose(got, want, atol=1e-5)
    assert ctl.touched_frac == pytest.approx(1.0)


def test_dense_layers_and_prefill_bypass_selection():
    torch.manual_seed(4)
    key = torch.randn(1, 2, 64, 8)
    value = torch.randn(1, 2, 64, 8)
    scaling = 1.0 / math.sqrt(8)
    ctl = SparseAttentionController(PRIMARY_ARM, 32, 0.1, dense_layers=(0,))
    ctl.reset(torch.device("cpu"))

    ctl(stub_module(0, 2), torch.randn(1, 4, 1, 8), key, value, None, scaling)
    assert (ctl.sparse_calls, ctl.dense_calls) == (0, 1)

    prefill_q = torch.randn(1, 4, 64, 8)
    mask = torch.zeros(1, 1, 64, 64)
    ctl(stub_module(5, 2), prefill_q, key, value, mask, scaling)
    assert (ctl.sparse_calls, ctl.dense_calls) == (0, 2)

    ctl(stub_module(5, 2), torch.randn(1, 4, 1, 8), key, value, None, scaling)
    assert (ctl.sparse_calls, ctl.dense_calls) == (1, 2)


def test_touched_frac_is_at_least_the_nominal_budget():
    torch.manual_seed(5)
    key = torch.randn(1, 2, 200, 8)
    value = torch.randn(1, 2, 200, 8)
    query = torch.randn(1, 4, 1, 8)
    ctl = SparseAttentionController(PRIMARY_ARM, 32, 0.05, dense_layers=())
    ctl.reset(torch.device("cpu"))
    ctl(stub_module(1, 2), query, key, value, None, 1.0 / math.sqrt(8))
    # 7 units, ceil(0.05*7)=1 unit, but unit 0 and the last unit are both forced in,
    # so the honest cost is larger than the nominal 5%.
    assert ctl.touched_frac > 0.05
    assert ctl.touched_frac < 1.0


def test_sparse_output_tracks_the_planted_unit_and_recency_misses_it():
    """A planted high-similarity key early in the sequence.

    quest_shared_heads must select the unit holding it, and its output must look like the
    dense output. `recency` must not, or the arm is not actually bad and the gate is not
    actually measuring anything.
    """
    torch.manual_seed(6)
    b, kvh, groups, s, d = 1, 1, 1, 256, 8
    gran = 32
    key = torch.randn(b, kvh, s, d) * 0.05
    value = torch.randn(b, kvh, s, d) * 0.05
    direction = torch.zeros(d)
    direction[0] = 1.0
    query = direction.view(1, 1, 1, d) * 8.0
    planted_pos = 40                       # unit 1, far from the tail
    key[0, 0, planted_pos] = direction * 8.0
    value[0, 0, planted_pos] = torch.ones(d)
    scaling = 1.0 / math.sqrt(d)
    module = stub_module(layer_idx=1, groups=groups)
    want = dense_reference(query, key, value, groups, scaling)

    # 8 units at gran=32; the budget must leave room beyond the two forced units, or the
    # selector is not being asked to choose anything and the test proves nothing.
    good = SparseAttentionController(PRIMARY_ARM, gran, 0.5, dense_layers=())
    good.reset(torch.device("cpu"))
    got_good, _ = good(module, query, key, value, None, scaling)

    bad = SparseAttentionController("recency", gran, 0.5, dense_layers=())
    bad.reset(torch.device("cpu"))
    got_bad, _ = bad(module, query, key, value, None, scaling)

    err_good = (got_good - want).abs().max().item()
    err_bad = (got_bad - want).abs().max().item()
    assert err_good < 1e-3
    assert err_bad > 10 * max(err_good, 1e-6)


def test_per_head_arm_runs_and_returns_dense_shape():
    torch.manual_seed(7)
    b, kvh, groups, s, d = 1, 2, 2, 128, 8
    key = torch.randn(b, kvh, s, d)
    value = torch.randn(b, kvh, s, d)
    query = torch.randn(b, kvh * groups, 1, d)
    ctl = SparseAttentionController("quest_per_head", 32, 0.5, dense_layers=())
    ctl.reset(torch.device("cpu"))
    out, _ = ctl(stub_module(2, groups), query, key, value, None, 1.0 / math.sqrt(d))
    assert out.shape == (b, 1, kvh * groups, d)
    assert 0.0 < ctl.touched_frac <= 1.0


# ---------------------------------------------------------------------------
# gate behaviour
# ---------------------------------------------------------------------------

def test_token_agreement_reports_first_divergence():
    agree, first = token_agreement([1, 2, 3, 4], [1, 2, 9, 4])
    assert agree == pytest.approx(0.75)
    assert first == 2
    agree, first = token_agreement([1, 2], [1, 2])
    assert agree == 1.0 and first == -1
    agree, first = token_agreement([1, 2, 3], [1, 2])
    assert first == 2


def _row(arm, gran, kf, correct, touched=0.1, agree=1.0):
    return {"arm": arm, "granularity": gran, "k_frac": kf, "answer_correct": correct,
            "touched_frac": touched, "token_agreement": agree}


def test_verdict_is_discriminative_when_bad_arms_fail():
    rows = [
        _row(PRIMARY_ARM, 32, 0.051, True, touched=0.06),
        _row("recency", 32, 0.051, False),
        _row("sink_recency", 32, 0.051, False),
        _row("random", 32, 0.051, False),
        _row(PRIMARY_ARM, 256, 0.25, True, touched=0.26),
        _row("recency", 256, 0.25, False),
    ]
    v = build_verdict(rows, dense_correct=True)
    assert v["verdict"] == "DISCRIMINATIVE"
    assert v["cheapest_passing"]["granularity"] == 32
    assert v["cheapest_passing"]["primary_touched_frac"] == pytest.approx(0.06)


def test_verdict_is_non_discriminative_when_bad_arms_also_pass():
    rows = [
        _row(PRIMARY_ARM, 32, 0.5, True),
        _row("recency", 32, 0.5, True),
        _row("sink_recency", 32, 0.5, True),
        _row("random", 32, 0.5, True),
    ]
    v = build_verdict(rows, dense_correct=True)
    assert v["verdict"] == "NON_DISCRIMINATIVE"
    assert "bad arm" in v["reason"]


def test_verdict_is_non_discriminative_when_the_real_selector_fails_too():
    rows = [
        _row(PRIMARY_ARM, 32, 0.02, False),
        _row("recency", 32, 0.02, False),
    ]
    v = build_verdict(rows, dense_correct=True)
    assert v["verdict"] == "NON_DISCRIMINATIVE"
    assert v["discriminative_cells"] == []


def test_verdict_rejects_a_prompt_the_dense_model_cannot_answer():
    rows = [_row(PRIMARY_ARM, 32, 0.051, True), _row("recency", 32, 0.051, False)]
    v = build_verdict(rows, dense_correct=False)
    assert v["verdict"] == "INVALID_PROMPT"
def test_top2_margin_is_the_gap_between_the_best_two_logits():
    from tools.e2e_sparse_attention import top2_margin
    logits = torch.tensor([0.5, 3.0, 1.0, -2.0])
    assert top2_margin(logits) == pytest.approx(2.0)
    flat = torch.zeros(5)
    assert top2_margin(flat) == pytest.approx(0.0)
