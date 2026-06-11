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
    attention_matching_compact_keys,
    attention_matching_highest_keys,
    attention_matching_decode,
    attention_output,
    fit_attention_bias,
    fit_compacted_values,
    highest_attention_key_indices,
    omp_attention_key_indices,
)


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


def test_attention_matching_decode_matches_full_attention_when_budget_covers_cache():
    torch.manual_seed(4)
    q = torch.randn(1, 2, 4)
    keys = torch.randn(1, 5, 1, 4)
    values = torch.randn(1, 5, 1, 4)
    context_lens = torch.tensor([5], dtype=torch.int32)

    out = attention_matching_decode(q, keys, values, context_lens, budget=8)
    expected = attention_output(q[0], keys[0, :, 0], values[0, :, 0]).view(1, 2, 4)

    assert torch.allclose(out, expected.to(out.dtype), atol=1e-5, rtol=1e-5)


def main():
    test_highest_attention_key_indices_selects_dominant_key_by_rms_attention()
    test_omp_attention_key_indices_returns_budget_unique_indices()
    test_omp_selector_reduces_error_vs_highest_keys_on_synthetic_case()
    test_fit_attention_bias_preserves_attention_mass_for_selected_keys()
    test_fit_compacted_values_reduces_attention_output_error_vs_direct_values()
    test_attention_matching_highest_keys_returns_compacted_cache_and_indices()
    test_attention_matching_decode_supports_gqa_and_compact_output_shape()
    test_attention_matching_decode_accepts_omp_selector()
    test_attention_matching_decode_matches_full_attention_when_budget_covers_cache()
    print("attention matching tests passed")


if __name__ == "__main__":
    main()
