"""Fast KV Compaction via Attention Matching 核心算法单测。

跑法：python3 tools/test_attention_matching.py
"""

import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tinyvllm.engine.attention_matching import (  # noqa: E402
    AttentionMatchingDecodeCache,
    build_attention_matching_prefill_cache,
    attention_matching_compact_keys,
    attention_matching_highest_keys,
    attention_matching_decode,
    attention_output,
    fit_attention_bias,
    fit_compacted_values,
    highest_attention_key_indices,
    omp_attention_key_indices,
)
from tinyvllm.utils.context import Context, am_compact_enabled_layers, am_compact_layer_enabled  # noqa: E402


def test_highest_attention_key_indices_selects_dominant_key_by_rms_attention():
    queries = torch.tensor([[5.0, 0.0], [4.0, 0.0], [6.0, 0.0]], dtype=torch.float32)
    keys = torch.tensor([
        [0.0, 1.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ], dtype=torch.float32)

    indices = highest_attention_key_indices(keys, queries, budget=2, score_method="rms")

    assert indices.tolist()[0] == 1
    assert len(indices.tolist()) == 2


def test_omp_attention_key_indices_returns_budget_unique_indices():
    torch.manual_seed(10)
    queries = torch.randn(5, 4)
    keys = torch.randn(9, 4)
    values = torch.randn(9, 4)

    indices = omp_attention_key_indices(
        keys,
        values,
        queries,
        budget=4,
        ridge_lambda=1e-6,
        candidate_pool_size=6,
    )

    assert indices.shape == (4,)
    assert indices.dtype == torch.long
    assert len(set(indices.tolist())) == 4
    assert torch.all(indices >= 0)
    assert torch.all(indices < keys.shape[0])


def test_omp_selector_reduces_error_vs_highest_keys_on_synthetic_case():
    queries = torch.tensor([[6.0, 0.0], [0.0, 6.0]], dtype=torch.float32)
    keys = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, 0.8],
        [-1.0, 0.0],
    ], dtype=torch.float32)
    values = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
        [-1.0, 0.0],
    ], dtype=torch.float32)
    target = attention_output(queries, keys, values)

    highest = attention_matching_compact_keys(keys, values, queries, budget=2, selector="highest")
    omp = attention_matching_compact_keys(keys, values, queries, budget=2, selector="omp")

    highest_out = attention_output(queries, highest.keys, highest.values, highest.beta)
    omp_out = attention_output(queries, omp.keys, omp.values, omp.beta)

    highest_mse = torch.mean((highest_out - target) ** 2)
    omp_mse = torch.mean((omp_out - target) ** 2)

    assert omp_mse <= highest_mse + 1e-6


def test_fit_attention_bias_preserves_attention_mass_for_selected_keys():
    torch.manual_seed(0)
    keys = torch.randn(6, 4)
    queries = torch.randn(5, 4)
    selected = torch.tensor([0, 2, 4, 5])

    beta = fit_attention_bias(keys, queries, selected, beta_bound=3.0)

    full_mass = torch.exp((queries @ keys.T) / (keys.shape[1] ** 0.5)).sum(dim=1)
    compact_mass = torch.exp((queries @ keys[selected].T) / (keys.shape[1] ** 0.5) + beta).sum(dim=1)
    compact_mass_without_beta = torch.exp((queries @ keys[selected].T) / (keys.shape[1] ** 0.5)).sum(dim=1)

    err_with_beta = torch.mean((compact_mass - full_mass).abs())
    err_without_beta = torch.mean((compact_mass_without_beta - full_mass).abs())

    assert err_with_beta < err_without_beta
    assert torch.all(beta <= 3.0)
    assert torch.all(beta >= -3.0)


def test_fit_compacted_values_reduces_attention_output_error_vs_direct_values():
    torch.manual_seed(1)
    keys = torch.randn(8, 4)
    values = torch.randn(8, 4)
    queries = torch.randn(12, 4)
    selected = torch.tensor([0, 2, 5, 7])
    beta = fit_attention_bias(keys, queries, selected, beta_bound=3.0)

    target = attention_output(queries, keys, values)
    direct = attention_output(queries, keys[selected], values[selected], beta)
    compact_values = fit_compacted_values(keys, values, queries, selected, beta, ridge_lambda=1e-6)
    fitted = attention_output(queries, keys[selected], compact_values, beta)

    direct_mse = torch.mean((direct - target) ** 2)
    fitted_mse = torch.mean((fitted - target) ** 2)

    assert fitted_mse < direct_mse


def test_attention_matching_highest_keys_returns_compacted_cache_and_indices():
    torch.manual_seed(2)
    keys = torch.randn(10, 6)
    values = torch.randn(10, 6)
    queries = torch.randn(16, 6)

    compact = attention_matching_highest_keys(keys, values, queries, budget=4)

    assert compact.keys.shape == (4, 6)
    assert compact.values.shape == (4, 6)
    assert compact.beta.shape == (4,)
    assert compact.indices.shape == (4,)
    assert compact.keys.dtype == keys.dtype
    assert compact.values.dtype == values.dtype


def test_attention_matching_decode_supports_gqa_and_compact_output_shape():
    torch.manual_seed(3)
    q = torch.randn(1, 4, 6)
    keys = torch.randn(1, 10, 2, 6)
    values = torch.randn(1, 10, 2, 6)
    context_lens = torch.tensor([10], dtype=torch.int32)

    out = attention_matching_decode(q, keys, values, context_lens, budget=4)

    assert out.shape == (1, 4, 6)
    assert out.dtype == q.dtype


def test_attention_matching_decode_accepts_omp_selector():
    torch.manual_seed(11)
    q = torch.randn(1, 4, 6)
    keys = torch.randn(1, 12, 2, 6)
    values = torch.randn(1, 12, 2, 6)
    context_lens = torch.tensor([12], dtype=torch.int32)

    out = attention_matching_decode(q, keys, values, context_lens, budget=4, selector="omp")

    assert out.shape == (1, 4, 6)
    assert out.dtype == q.dtype


def test_attention_matching_decode_reuses_compact_cache_within_refresh_interval():
    torch.manual_seed(12)
    cache = AttentionMatchingDecodeCache()
    q1 = torch.randn(1, 2, 4)
    q2 = torch.randn(1, 2, 4)
    keys = torch.randn(1, 8, 1, 4)
    values = torch.randn(1, 8, 1, 4)
    context_lens = torch.tensor([8], dtype=torch.int32)

    out1 = attention_matching_decode(
        q1,
        keys,
        values,
        context_lens,
        budget=3,
        selector="omp",
        cache=cache,
        cache_refresh_interval=8,
        cache_signatures=("seq-a",),
    )
    out2 = attention_matching_decode(
        q2,
        keys,
        values,
        context_lens,
        budget=3,
        selector="omp",
        cache=cache,
        cache_refresh_interval=8,
        cache_signatures=("seq-a",),
    )

    assert out1.shape == q1.shape
    assert out2.shape == q2.shape
    assert cache.misses == 1
    assert cache.hits == 1
    assert len(cache.entries) == 1


def test_attention_matching_decode_refreshes_compact_cache_after_interval():
    torch.manual_seed(13)
    cache = AttentionMatchingDecodeCache()
    q1 = torch.randn(1, 2, 4)
    q2 = torch.randn(1, 2, 4)
    keys = torch.randn(1, 8, 1, 4)
    values = torch.randn(1, 8, 1, 4)
    context_lens = torch.tensor([8], dtype=torch.int32)

    attention_matching_decode(
        q1,
        keys,
        values,
        context_lens,
        budget=3,
        selector="omp",
        cache=cache,
        cache_refresh_interval=1,
        cache_signatures=("seq-a",),
    )
    attention_matching_decode(
        q2,
        keys,
        values,
        context_lens,
        budget=3,
        selector="omp",
        cache=cache,
        cache_refresh_interval=1,
        cache_signatures=("seq-a",),
    )

    assert cache.misses == 2
    assert cache.hits == 0
    assert len(cache.entries) == 1


def test_build_attention_matching_prefill_cache_populates_decode_cache():
    torch.manual_seed(14)
    cache = AttentionMatchingDecodeCache()
    queries = torch.randn(1, 6, 2, 4)
    keys = torch.randn(1, 6, 1, 4)
    values = torch.randn(1, 6, 1, 4)
    context_lens = torch.tensor([6], dtype=torch.int32)

    build_attention_matching_prefill_cache(
        cache,
        queries,
        keys,
        values,
        context_lens,
        budget=3,
        selector="omp",
        cache_signatures=("seq-a",),
        ref_query_stride=2,
    )

    assert cache.prefill_builds == 1
    assert len(cache.entries) == 1

    out = attention_matching_decode(
        torch.randn(1, 2, 4),
        keys,
        values,
        context_lens,
        budget=3,
        selector="omp",
        cache=cache,
        cache_refresh_interval=1024,
        cache_signatures=("seq-a",),
    )

    assert out.shape == (1, 2, 4)
    assert cache.hits == 1
    assert cache.misses == 0


def test_prefill_cache_builds_multi_cluster_bank_and_routes_decode_queries():
    torch.manual_seed(15)
    cache = AttentionMatchingDecodeCache()
    queries = torch.zeros(1, 8, 1, 4)
    queries[0, :4, 0, 0] = 4.0
    queries[0, 4:, 0, 1] = 4.0
    keys = torch.randn(1, 8, 1, 4)
    values = torch.randn(1, 8, 1, 4)
    context_lens = torch.tensor([8], dtype=torch.int32)

    build_attention_matching_prefill_cache(
        cache,
        queries,
        keys,
        values,
        context_lens,
        budget=3,
        selector="highest",
        cache_signatures=("seq-a",),
        ref_query_stride=1,
        num_clusters=2,
    )

    assert cache.prefill_builds == 1
    assert len(cache.entries) == 1
    entry = next(iter(cache.entries.values()))
    assert len(entry.compacts) == 2
    assert entry.centroids.shape == (2, 4)

    q_first = torch.tensor([[[4.0, 0.0, 0.0, 0.0]]])
    q_second = torch.tensor([[[0.0, 4.0, 0.0, 0.0]]])
    out_first = attention_matching_decode(
        q_first,
        keys,
        values,
        context_lens,
        budget=3,
        selector="highest",
        cache=cache,
        cache_refresh_interval=1024,
        cache_signatures=("seq-a",),
        num_clusters=2,
    )
    out_second = attention_matching_decode(
        q_second,
        keys,
        values,
        context_lens,
        budget=3,
        selector="highest",
        cache=cache,
        cache_refresh_interval=1024,
        cache_signatures=("seq-a",),
        num_clusters=2,
    )

    assert out_first.shape == q_first.shape
    assert out_second.shape == q_second.shape
    assert cache.hits == 2
    assert cache.misses == 0
    assert cache.last_cluster_indices == [0, 1]


def test_multi_cluster_route_top_k_ensembles_multiple_compacts():
    torch.manual_seed(16)
    cache = AttentionMatchingDecodeCache()
    queries = torch.zeros(1, 8, 1, 4)
    queries[0, :4, 0, 0] = 4.0
    queries[0, 4:, 0, 1] = 4.0
    keys = torch.randn(1, 8, 1, 4)
    values = torch.randn(1, 8, 1, 4)
    context_lens = torch.tensor([8], dtype=torch.int32)

    build_attention_matching_prefill_cache(
        cache,
        queries,
        keys,
        values,
        context_lens,
        budget=3,
        selector="highest",
        cache_signatures=("seq-a",),
        num_clusters=2,
    )

    out = attention_matching_decode(
        torch.tensor([[[3.0, 1.0, 0.0, 0.0]]]),
        keys,
        values,
        context_lens,
        budget=3,
        selector="highest",
        cache=cache,
        cache_refresh_interval=1024,
        cache_signatures=("seq-a",),
        num_clusters=2,
        cluster_route_top_k=2,
    )

    assert out.shape == (1, 1, 4)
    assert cache.hits == 1
    assert cache.misses == 0
    assert cache.last_cluster_indices == [[0, 1]]


def test_prefill_cache_builds_span_local_bank_covering_each_key_span():
    torch.manual_seed(17)
    cache = AttentionMatchingDecodeCache()
    queries = torch.randn(1, 8, 1, 4)
    keys = torch.randn(1, 8, 1, 4)
    values = torch.randn(1, 8, 1, 4)
    context_lens = torch.tensor([8], dtype=torch.int32)

    build_attention_matching_prefill_cache(
        cache,
        queries,
        keys,
        values,
        context_lens,
        budget=2,
        selector="highest",
        cache_signatures=("seq-a",),
        num_key_spans=2,
    )

    assert cache.prefill_builds == 1
    entry = next(iter(cache.entries.values()))
    assert len(entry.compacts) == 2
    assert entry.centroids.shape == (2, 4)
    assert torch.all(entry.compacts[0].indices < 4)
    assert torch.all(entry.compacts[1].indices >= 4)

    out = attention_matching_decode(
        torch.randn(1, 1, 4),
        keys,
        values,
        context_lens,
        budget=2,
        selector="highest",
        cache=cache,
        cache_refresh_interval=1024,
        cache_signatures=("seq-a",),
        num_key_spans=2,
        cluster_route_top_k=2,
    )

    assert out.shape == (1, 1, 4)
    assert cache.hits == 1
    assert cache.misses == 0
    assert sorted(cache.last_cluster_indices[-1]) == [0, 1]


def test_decode_refit_recomputes_cached_values_for_current_query():
    torch.manual_seed(18)
    cache = AttentionMatchingDecodeCache()
    queries = torch.randn(1, 8, 1, 4)
    keys = torch.randn(1, 8, 1, 4)
    values = torch.randn(1, 8, 1, 4)
    context_lens = torch.tensor([8], dtype=torch.int32)

    build_attention_matching_prefill_cache(
        cache,
        queries,
        keys,
        values,
        context_lens,
        budget=3,
        selector="highest",
        cache_signatures=("seq-a",),
    )
    entry = next(iter(cache.entries.values()))
    selected = entry.compact.indices
    q_decode = torch.randn(1, 1, 4)

    out = attention_matching_decode(
        q_decode,
        keys,
        values,
        context_lens,
        budget=3,
        selector="highest",
        cache=cache,
        cache_refresh_interval=1024,
        cache_signatures=("seq-a",),
        decode_refit=True,
    )

    beta = fit_attention_bias(keys[0, :, 0], q_decode[0], selected, beta_bound=3.0)
    compact_values = fit_compacted_values(
        keys[0, :, 0],
        values[0, :, 0],
        q_decode[0],
        selected,
        beta,
        ridge_lambda=1e-6,
    )
    expected = attention_output(q_decode[0], keys[0, selected, 0], compact_values, beta).view_as(out)

    assert torch.allclose(out, expected.to(out.dtype), atol=1e-5, rtol=1e-5)
    assert cache.hits == 1
    assert cache.misses == 0


def test_decode_refit_direct_mode_uses_original_selected_values():
    torch.manual_seed(19)
    cache = AttentionMatchingDecodeCache()
    queries = torch.randn(1, 8, 1, 4)
    keys = torch.randn(1, 8, 1, 4)
    values = torch.randn(1, 8, 1, 4)
    context_lens = torch.tensor([8], dtype=torch.int32)

    build_attention_matching_prefill_cache(
        cache,
        queries,
        keys,
        values,
        context_lens,
        budget=3,
        selector="highest",
        cache_signatures=("seq-a",),
    )
    entry = next(iter(cache.entries.values()))
    selected = entry.compact.indices
    q_decode = torch.randn(1, 1, 4)

    out = attention_matching_decode(
        q_decode,
        keys,
        values,
        context_lens,
        budget=3,
        selector="highest",
        cache=cache,
        cache_refresh_interval=1024,
        cache_signatures=("seq-a",),
        decode_refit=True,
        decode_refit_mode="direct",
    )

    beta = fit_attention_bias(keys[0, :, 0], q_decode[0], selected, beta_bound=3.0)
    expected = attention_output(q_decode[0], keys[0, selected, 0], values[0, selected, 0], beta).view_as(out)

    assert torch.allclose(out, expected.to(out.dtype), atol=1e-5, rtol=1e-5)
    assert cache.hits == 1
    assert cache.misses == 0


def test_decode_refit_anchor_mode_uses_anchor_target_subset():
    torch.manual_seed(21)
    cache = AttentionMatchingDecodeCache()
    queries = torch.randn(1, 8, 1, 4)
    keys = torch.randn(1, 8, 1, 4)
    values = torch.randn(1, 8, 1, 4)
    context_lens = torch.tensor([8], dtype=torch.int32)

    build_attention_matching_prefill_cache(
        cache,
        queries,
        keys,
        values,
        context_lens,
        budget=3,
        selector="highest",
        cache_signatures=("seq-a",),
    )
    entry = next(iter(cache.entries.values()))
    selected = entry.compact.indices
    q_decode = torch.randn(1, 1, 4)

    out = attention_matching_decode(
        q_decode,
        keys,
        values,
        context_lens,
        budget=3,
        selector="highest",
        cache=cache,
        cache_refresh_interval=1024,
        cache_signatures=("seq-a",),
        decode_refit=True,
        decode_refit_mode="anchor",
    )

    anchors = torch.arange(keys.shape[1], dtype=torch.long)
    fit_selected = selected
    beta = fit_attention_bias(keys[0, anchors, 0], q_decode[0], fit_selected, beta_bound=3.0)
    compact_values = fit_compacted_values(
        keys[0, anchors, 0],
        values[0, anchors, 0],
        q_decode[0],
        fit_selected,
        beta,
        ridge_lambda=1e-6,
    )
    expected = attention_output(q_decode[0], keys[0, selected, 0], compact_values, beta).view_as(out)

    assert torch.allclose(out, expected.to(out.dtype), atol=1e-5, rtol=1e-5)
    assert cache.hits == 1
    assert cache.misses == 0


def test_decode_refit_interval_reuses_refitted_compact_values():
    torch.manual_seed(20)
    cache = AttentionMatchingDecodeCache()
    queries = torch.randn(1, 8, 1, 4)
    keys = torch.randn(1, 8, 1, 4)
    values = torch.randn(1, 8, 1, 4)
    context_lens = torch.tensor([8], dtype=torch.int32)

    build_attention_matching_prefill_cache(
        cache,
        queries,
        keys,
        values,
        context_lens,
        budget=3,
        selector="highest",
        cache_signatures=("seq-a",),
    )

    q1 = torch.randn(1, 1, 4)
    q2 = torch.randn(1, 1, 4)
    attention_matching_decode(
        q1,
        keys,
        values,
        context_lens,
        budget=3,
        selector="highest",
        cache=cache,
        cache_refresh_interval=1024,
        cache_signatures=("seq-a",),
        decode_refit=True,
        decode_refit_interval=8,
    )
    refit_entry = next(iter(cache.refit_entries.values()))

    out2 = attention_matching_decode(
        q2,
        keys,
        values,
        context_lens,
        budget=3,
        selector="highest",
        cache=cache,
        cache_refresh_interval=1024,
        cache_signatures=("seq-a",),
        decode_refit=True,
        decode_refit_interval=8,
    )
    expected = attention_output(
        q2[0],
        refit_entry.compact.keys,
        refit_entry.compact.values,
        refit_entry.compact.beta,
    ).view_as(out2)

    assert torch.allclose(out2, expected.to(out2.dtype), atol=1e-5, rtol=1e-5)
    assert cache.refit_misses == 1
    assert cache.refit_hits == 1


def test_decode_refit_batches_multiple_rows_and_kv_heads():
    torch.manual_seed(22)
    cache = AttentionMatchingDecodeCache()
    queries = torch.randn(2, 9, 4, 4)
    keys = torch.randn(2, 9, 2, 4)
    values = torch.randn(2, 9, 2, 4)
    context_lens = torch.tensor([9, 9], dtype=torch.int32)

    build_attention_matching_prefill_cache(
        cache,
        queries,
        keys,
        values,
        context_lens,
        budget=3,
        selector="highest",
        cache_signatures=("seq-a", "seq-b"),
    )

    q_decode = torch.randn(2, 4, 4)
    out = attention_matching_decode(
        q_decode,
        keys,
        values,
        context_lens,
        budget=3,
        selector="highest",
        cache=cache,
        cache_refresh_interval=1024,
        cache_signatures=("seq-a", "seq-b"),
        decode_refit=True,
    )

    expected = torch.empty_like(out)
    group_size = 2
    for b, signature in enumerate(("seq-a", "seq-b")):
        for kv_h in range(2):
            q_start = kv_h * group_size
            q_end = q_start + group_size
            q_group = q_decode[b, q_start:q_end]
            k_seq = keys[b, :, kv_h]
            v_seq = values[b, :, kv_h]
            entry = next(
                entry for key, entry in cache.entries.items()
                if key[0] == signature and key[1] == kv_h
            )
            selected = entry.compact.indices
            beta = fit_attention_bias(k_seq, q_group, selected, beta_bound=3.0)
            compact_values = fit_compacted_values(
                k_seq,
                v_seq,
                q_group,
                selected,
                beta,
                ridge_lambda=1e-6,
            )
            expected[b, q_start:q_end] = attention_output(q_group, k_seq[selected], compact_values, beta).to(out.dtype)

    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)
    assert cache.hits == 4
    assert cache.misses == 0


def test_attention_matching_decode_matches_full_attention_when_budget_covers_cache():
    torch.manual_seed(4)
    q = torch.randn(1, 2, 4)
    keys = torch.randn(1, 5, 1, 4)
    values = torch.randn(1, 5, 1, 4)
    context_lens = torch.tensor([5], dtype=torch.int32)

    out = attention_matching_decode(q, keys, values, context_lens, budget=8)
    expected = attention_output(q[0], keys[0, :, 0], values[0, :, 0]).view(1, 2, 4)

    assert torch.allclose(out, expected.to(out.dtype), atol=1e-5, rtol=1e-5)


def test_am_compact_layer_enabled_respects_skip_stride_and_explicit_layers():
    disabled = Context(am_compact_blocks=0)
    assert not am_compact_layer_enabled(disabled, layer_idx=4, num_hidden_layers=32)

    skipped = Context(
        am_compact_blocks=4,
        am_compact_skip_first_layers=4,
        am_compact_skip_last_layers=4,
        am_compact_layer_stride=2,
    )
    assert not am_compact_layer_enabled(skipped, layer_idx=3, num_hidden_layers=32)
    assert am_compact_layer_enabled(skipped, layer_idx=4, num_hidden_layers=32)
    assert not am_compact_layer_enabled(skipped, layer_idx=5, num_hidden_layers=32)
    assert not am_compact_layer_enabled(skipped, layer_idx=28, num_hidden_layers=32)

    explicit = Context(am_compact_blocks=4, am_compact_enable_layers=(1, 7, 9))
    assert am_compact_layer_enabled(explicit, layer_idx=7, num_hidden_layers=32)
    assert not am_compact_layer_enabled(explicit, layer_idx=8, num_hidden_layers=32)
    assert am_compact_enabled_layers(skipped, num_hidden_layers=10) == (4,)
    assert am_compact_enabled_layers(explicit, num_hidden_layers=10) == (1, 7, 9)


def main():
    test_highest_attention_key_indices_selects_dominant_key_by_rms_attention()
    test_omp_attention_key_indices_returns_budget_unique_indices()
    test_omp_selector_reduces_error_vs_highest_keys_on_synthetic_case()
    test_fit_attention_bias_preserves_attention_mass_for_selected_keys()
    test_fit_compacted_values_reduces_attention_output_error_vs_direct_values()
    test_attention_matching_highest_keys_returns_compacted_cache_and_indices()
    test_attention_matching_decode_supports_gqa_and_compact_output_shape()
    test_attention_matching_decode_accepts_omp_selector()
    test_attention_matching_decode_reuses_compact_cache_within_refresh_interval()
    test_attention_matching_decode_refreshes_compact_cache_after_interval()
    test_build_attention_matching_prefill_cache_populates_decode_cache()
    test_prefill_cache_builds_multi_cluster_bank_and_routes_decode_queries()
    test_multi_cluster_route_top_k_ensembles_multiple_compacts()
    test_prefill_cache_builds_span_local_bank_covering_each_key_span()
    test_decode_refit_recomputes_cached_values_for_current_query()
    test_decode_refit_direct_mode_uses_original_selected_values()
    test_decode_refit_interval_reuses_refitted_compact_values()
    test_decode_refit_batches_multiple_rows_and_kv_heads()
    test_attention_matching_decode_matches_full_attention_when_budget_covers_cache()
    test_am_compact_layer_enabled_respects_skip_stride_and_explicit_layers()
    print("attention matching tests passed")


if __name__ == "__main__":
    main()
