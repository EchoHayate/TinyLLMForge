import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import types
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


for package_name in ("tinyvllm", "tinyvllm.layers"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules[package_name] = package

_load_module(
    "tinyvllm.layers.qwen35_primitives",
    "tinyvllm/layers/qwen35_primitives.py",
)
full_attention = _load_module(
    "tinyvllm.layers.qwen35_full_attention",
    "tinyvllm/layers/qwen35_full_attention.py",
)


def _official_suffix_eager(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    query_heads = query.shape[1]
    head_dim = query.shape[2]
    repeats = query_heads // key.shape[1]
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    query = query.transpose(0, 1).unsqueeze(0)
    key = key.transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)
    scores = torch.matmul(query, key.transpose(2, 3))
    scores = scores * (head_dim ** -0.5)
    query_length = query.shape[2]
    key_length = key.shape[2]
    prefix_length = key_length - query_length
    query_positions = (
        torch.arange(query_length, device=query.device) + prefix_length
    )
    key_positions = torch.arange(key_length, device=query.device)
    mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
    scores = scores.masked_fill(
        mask.view(1, 1, query_length, key_length),
        float("-inf"),
    )
    probabilities = torch.softmax(
        scores,
        dim=-1,
        dtype=torch.float32,
    ).to(query.dtype)
    return torch.matmul(
        probabilities,
        value,
    ).transpose(1, 2).squeeze(0)


def _official_prefill_eager(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    query_heads = query.shape[1]
    head_dim = query.shape[2]
    repeats = query_heads // key.shape[1]
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    query = query.transpose(0, 1).unsqueeze(0)
    key = key.transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)
    scores = torch.matmul(query, key.transpose(2, 3))
    scores = scores * (head_dim ** -0.5)
    token_count = query.shape[2]
    positions = torch.arange(token_count, device=query.device)
    causal = positions.unsqueeze(0) > positions.unsqueeze(1)
    mask = torch.zeros_like(scores).masked_fill(
        causal.view(1, 1, token_count, token_count),
        torch.finfo(scores.dtype).min,
    )
    probabilities = torch.softmax(
        scores + mask,
        dim=-1,
        dtype=torch.float32,
    ).to(query.dtype)
    return torch.matmul(
        probabilities,
        value,
    ).transpose(1, 2).squeeze(0)


def test_prefill_eager_matches_official_additive_mask_bit_exact() -> None:
    query = torch.tensor(
        [
            [[1, 2], [2, 1]],
            [[3, 1], [1, 3]],
            [[2, 2], [4, 1]],
        ],
        dtype=torch.bfloat16,
    )
    key = torch.tensor(
        [
            [[1, 0]],
            [[0, 1]],
            [[1, 1]],
        ],
        dtype=torch.bfloat16,
    )
    value = torch.tensor(
        [
            [[1, 2]],
            [[3, 4]],
            [[5, 6]],
        ],
        dtype=torch.bfloat16,
    )
    context = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
    )

    actual = full_attention.qwen35_prefill_eager_attention(
        query,
        key,
        value,
        context,
        num_heads=2,
        head_dim=2,
        scale=2 ** -0.5,
    )
    expected = _official_prefill_eager(
        query,
        key,
        value,
    ).reshape(3, 4)

    assert actual.dtype == torch.bfloat16
    assert actual.shape == (3, 4)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_prefill_eager_populates_available_kv_cache() -> None:
    query = torch.tensor(
        [
            [[1, 2], [2, 1]],
            [[3, 1], [1, 3]],
        ],
        dtype=torch.bfloat16,
    )
    key = torch.tensor(
        [
            [[1, 0]],
            [[0, 1]],
        ],
        dtype=torch.bfloat16,
    )
    value = torch.tensor(
        [
            [[1, 2]],
            [[3, 4]],
        ],
        dtype=torch.bfloat16,
    )
    key_cache = torch.zeros(2, 2, 1, 2, dtype=torch.bfloat16)
    value_cache = torch.zeros_like(key_cache)
    context = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        slot_mapping=torch.tensor([3, 0], dtype=torch.int32),
    )

    full_attention.qwen35_prefill_eager_attention(
        query,
        key,
        value,
        context,
        num_heads=2,
        head_dim=2,
        scale=2 ** -0.5,
        key_cache=key_cache,
        value_cache=value_cache,
    )

    torch.testing.assert_close(key_cache[1, 1], key[0], atol=0, rtol=0)
    torch.testing.assert_close(key_cache[0, 0], key[1], atol=0, rtol=0)
    torch.testing.assert_close(value_cache[1, 1], value[0], atol=0, rtol=0)
    torch.testing.assert_close(value_cache[0, 0], value[1], atol=0, rtol=0)


def test_prefill_eager_uses_reference_length_without_writing_padding() -> None:
    query = torch.tensor(
        [
            [[1, 2], [2, 1]],
            [[3, 1], [1, 3]],
        ],
        dtype=torch.bfloat16,
    )
    key = torch.tensor(
        [
            [[1, 0]],
            [[0, 1]],
        ],
        dtype=torch.bfloat16,
    )
    value = torch.tensor(
        [
            [[1, 2]],
            [[3, 4]],
        ],
        dtype=torch.bfloat16,
    )
    key_cache = torch.zeros(3, 2, 1, 2, dtype=torch.bfloat16)
    value_cache = torch.zeros_like(key_cache)
    context = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        slot_mapping=torch.tensor([3, 0], dtype=torch.int32),
        prefill_attention_reference_lens=(4,),
    )
    matmul_shapes = []
    original_matmul = torch.matmul

    def record_matmul(left, right):
        matmul_shapes.append((tuple(left.shape), tuple(right.shape)))
        return original_matmul(left, right)

    padded_query = torch.cat(
        (query, torch.zeros_like(query)),
        dim=0,
    )
    padded_key = torch.cat(
        (key, torch.zeros_like(key)),
        dim=0,
    )
    padded_value = torch.cat(
        (value, torch.zeros_like(value)),
        dim=0,
    )
    expected = _official_prefill_eager(
        padded_query,
        padded_key,
        padded_value,
    )[:2].reshape(2, 4)

    with patch.object(full_attention.torch, "matmul", record_matmul):
        actual = full_attention.qwen35_prefill_eager_attention(
            query,
            key,
            value,
            context,
            num_heads=2,
            head_dim=2,
            scale=2 ** -0.5,
            key_cache=key_cache,
            value_cache=value_cache,
        )

    assert matmul_shapes == [
        ((1, 2, 4, 2), (1, 2, 2, 4)),
        ((1, 2, 4, 4), (1, 2, 4, 2)),
    ]
    assert actual.shape == (2, 4)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(key_cache[1, 1], key[0], atol=0, rtol=0)
    torch.testing.assert_close(key_cache[0, 0], key[1], atol=0, rtol=0)
    assert torch.count_nonzero(key_cache[2]).item() == 0
    assert torch.count_nonzero(value_cache[2]).item() == 0


def test_cached_prefill_eager_matches_official_suffix_bit_exact() -> None:
    query = torch.tensor(
        [
            [[1, 2], [2, 1]],
            [[3, 1], [1, 3]],
        ],
        dtype=torch.bfloat16,
    )
    dense_key = torch.tensor(
        [
            [[1, 0]],
            [[0, 1]],
            [[1, 1]],
        ],
        dtype=torch.bfloat16,
    )
    dense_value = torch.tensor(
        [
            [[1, 2]],
            [[3, 4]],
            [[5, 6]],
        ],
        dtype=torch.bfloat16,
    )
    key_cache = torch.zeros(2, 2, 1, 2, dtype=torch.bfloat16)
    value_cache = torch.zeros_like(key_cache)
    key_cache[1, 0] = dense_key[0]
    value_cache[1, 0] = dense_value[0]
    context = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 3], dtype=torch.int32),
        block_tables=torch.tensor([[1, 0]], dtype=torch.int32),
        slot_mapping=torch.tensor([3, 0], dtype=torch.int32),
    )

    actual = full_attention.qwen35_cached_prefill_eager_attention(
        query,
        dense_key[1:],
        dense_value[1:],
        key_cache,
        value_cache,
        context,
        num_heads=2,
        head_dim=2,
        scale=2 ** -0.5,
    )
    expected = _official_suffix_eager(
        query,
        dense_key,
        dense_value,
    ).reshape(2, 4)

    assert actual.dtype == torch.bfloat16
    assert actual.shape == (2, 4)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(key_cache[1, 1], dense_key[1], atol=0, rtol=0)
    torch.testing.assert_close(key_cache[0, 0], dense_key[2], atol=0, rtol=0)


def test_cached_prefill_eager_matches_two_variable_suffixes_bit_exact() -> None:
    query = torch.tensor(
        [
            [[1, 2], [2, 1]],
            [[3, 1], [1, 3]],
            [[2, 2], [4, 1]],
        ],
        dtype=torch.bfloat16,
    )
    dense_keys = (
        torch.tensor(
            [
                [[1, 0]],
                [[0, 1]],
                [[1, 1]],
            ],
            dtype=torch.bfloat16,
        ),
        torch.tensor(
            [
                [[2, 0]],
                [[0, 2]],
                [[2, 2]],
                [[1, 2]],
            ],
            dtype=torch.bfloat16,
        ),
    )
    dense_values = (
        torch.tensor(
            [
                [[1, 2]],
                [[3, 4]],
                [[5, 6]],
            ],
            dtype=torch.bfloat16,
        ),
        torch.tensor(
            [
                [[2, 1]],
                [[4, 3]],
                [[6, 5]],
                [[8, 7]],
            ],
            dtype=torch.bfloat16,
        ),
    )
    key_cache = torch.zeros(4, 2, 1, 2, dtype=torch.bfloat16)
    value_cache = torch.zeros_like(key_cache)
    key_cache[2, 0] = dense_keys[0][0]
    value_cache[2, 0] = dense_values[0][0]
    key_cache[3] = dense_keys[1][:2]
    value_cache[3] = dense_values[1][:2]
    key_cache[1, 0] = dense_keys[1][2]
    value_cache[1, 0] = dense_values[1][2]
    context = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 2, 3], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 3, 7], dtype=torch.int32),
        block_tables=torch.tensor(
            [
                [2, 0],
                [3, 1],
            ],
            dtype=torch.int32,
        ),
        slot_mapping=torch.tensor([5, 0, 3], dtype=torch.int32),
    )

    actual = full_attention.qwen35_cached_prefill_eager_attention(
        query,
        torch.cat((dense_keys[0][1:], dense_keys[1][3:]), dim=0),
        torch.cat((dense_values[0][1:], dense_values[1][3:]), dim=0),
        key_cache,
        value_cache,
        context,
        num_heads=2,
        head_dim=2,
        scale=2 ** -0.5,
    )
    expected = torch.cat(
        (
            _official_suffix_eager(query[:2], dense_keys[0], dense_values[0]),
            _official_suffix_eager(query[2:], dense_keys[1], dense_values[1]),
        ),
        dim=0,
    ).reshape(3, 4)

    assert actual.dtype == torch.bfloat16
    assert actual.shape == (3, 4)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(key_cache[2, 1], dense_keys[0][1], atol=0, rtol=0)
    torch.testing.assert_close(key_cache[0, 0], dense_keys[0][2], atol=0, rtol=0)
    torch.testing.assert_close(key_cache[1, 1], dense_keys[1][3], atol=0, rtol=0)


def test_cached_prefill_eager_matches_official_gqa_bit_exact() -> None:
    query = torch.tensor(
        [
            [[1, 0], [0, 1], [1, 1], [2, 1]],
            [[1, 2], [2, 1], [0, 2], [2, 0]],
        ],
        dtype=torch.bfloat16,
    )
    dense_key = torch.tensor(
        [
            [[1, 0], [0, 1]],
            [[1, 1], [2, 1]],
            [[2, 0], [1, 2]],
        ],
        dtype=torch.bfloat16,
    )
    dense_value = torch.tensor(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
        ],
        dtype=torch.bfloat16,
    )
    key_cache = torch.zeros(2, 2, 2, 2, dtype=torch.bfloat16)
    value_cache = torch.zeros_like(key_cache)
    key_cache[1, 0] = dense_key[0]
    value_cache[1, 0] = dense_value[0]
    context = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 3], dtype=torch.int32),
        block_tables=torch.tensor([[1, 0]], dtype=torch.int32),
        slot_mapping=torch.tensor([3, 0], dtype=torch.int32),
    )

    actual = full_attention.qwen35_cached_prefill_eager_attention(
        query,
        dense_key[1:],
        dense_value[1:],
        key_cache,
        value_cache,
        context,
        num_heads=4,
        head_dim=2,
        scale=2 ** -0.5,
    )
    expected = _official_suffix_eager(
        query,
        dense_key,
        dense_value,
    ).reshape(2, 8)

    assert actual.shape == (2, 8)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_cached_prefill_eager_uses_full_context_query_rows() -> None:
    query = torch.tensor(
        [
            [[1, 2], [2, 1]],
            [[3, 1], [1, 3]],
        ],
        dtype=torch.bfloat16,
    )
    dense_key = torch.tensor(
        [
            [[1, 0]],
            [[0, 1]],
            [[1, 1]],
        ],
        dtype=torch.bfloat16,
    )
    dense_value = torch.tensor(
        [
            [[1, 2]],
            [[3, 4]],
            [[5, 6]],
        ],
        dtype=torch.bfloat16,
    )
    key_cache = torch.zeros(2, 2, 1, 2, dtype=torch.bfloat16)
    value_cache = torch.zeros_like(key_cache)
    key_cache[1, 0] = dense_key[0]
    value_cache[1, 0] = dense_value[0]
    context = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 3], dtype=torch.int32),
        block_tables=torch.tensor([[1, 0]], dtype=torch.int32),
        slot_mapping=torch.tensor([3, 0], dtype=torch.int32),
    )
    matmul_shapes = []
    original_matmul = torch.matmul

    def record_matmul(left, right):
        matmul_shapes.append((tuple(left.shape), tuple(right.shape)))
        return original_matmul(left, right)

    with patch.object(full_attention.torch, "matmul", record_matmul):
        full_attention.qwen35_cached_prefill_eager_attention(
            query,
            dense_key[1:],
            dense_value[1:],
            key_cache,
            value_cache,
            context,
            num_heads=2,
            head_dim=2,
            scale=2 ** -0.5,
        )

    assert matmul_shapes[0] == ((1, 2, 3, 2), (1, 2, 2, 3))


def test_cached_decode_eager_matches_official_bfloat16_bit_exact() -> None:
    query = torch.tensor(
        [
            [[1, 2], [2, 1], [3, 1], [1, 3]],
            [[2, 2], [4, 1], [1, 4], [3, 2]],
        ],
        dtype=torch.bfloat16,
    )
    dense_key = torch.tensor(
        [
            [[1, 0], [0, 1]],
            [[1, 1], [2, 1]],
            [[2, 0], [1, 2]],
            [[2, 2], [3, 1]],
            [[1, 3], [2, 3]],
        ],
        dtype=torch.bfloat16,
    )
    dense_value = torch.tensor(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
            [[13, 14], [15, 16]],
            [[17, 18], [19, 20]],
        ],
        dtype=torch.bfloat16,
    )
    key_cache = torch.zeros(4, 2, 2, 2, dtype=torch.bfloat16)
    value_cache = torch.zeros_like(key_cache)
    key_cache[2] = dense_key[:2]
    value_cache[2] = dense_value[:2]
    key_cache[3, 0] = dense_key[3]
    value_cache[3, 0] = dense_value[3]
    context = SimpleNamespace(
        block_tables=torch.tensor(
            [
                [2, 0],
                [3, 0],
            ],
            dtype=torch.int32,
        ),
        context_lens=torch.tensor([3, 2], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 7], dtype=torch.int32),
    )

    actual = full_attention.qwen35_cached_decode_eager_attention(
        query,
        torch.stack((dense_key[2], dense_key[4])),
        torch.stack((dense_value[2], dense_value[4])),
        key_cache,
        value_cache,
        context,
        num_heads=4,
        head_dim=2,
        scale=2 ** -0.5,
    )
    expected = torch.cat(
        (
            _official_suffix_eager(
                query[:1],
                dense_key[:3],
                dense_value[:3],
            ),
            _official_suffix_eager(
                query[1:],
                dense_key[3:],
                dense_value[3:],
            ),
        ),
        dim=0,
    ).reshape(2, 8)

    assert actual.dtype == torch.bfloat16
    assert actual.shape == (2, 8)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(key_cache[0, 0], dense_key[2], atol=0, rtol=0)
    torch.testing.assert_close(value_cache[0, 0], dense_value[2], atol=0, rtol=0)
    torch.testing.assert_close(key_cache[3, 1], dense_key[4], atol=0, rtol=0)
    torch.testing.assert_close(value_cache[3, 1], dense_value[4], atol=0, rtol=0)


def test_cached_decode_graph_matches_eager_for_paged_tp4_batch() -> None:
    block_size = 256
    query_heads = 2
    kv_heads = 1
    head_dim = 2
    query = torch.tensor(
        [
            [[1, 2], [2, 1]],
            [[2, 3], [3, 2]],
        ],
        dtype=torch.bfloat16,
    )
    current_key = torch.tensor(
        [
            [[3, 1]],
            [[4, 2]],
        ],
        dtype=torch.bfloat16,
    )
    current_value = torch.tensor(
        [
            [[5, 2]],
            [[6, 3]],
        ],
        dtype=torch.bfloat16,
    )
    key_cache = torch.zeros(
        4,
        block_size,
        kv_heads,
        head_dim,
        dtype=torch.bfloat16,
    )
    value_cache = torch.zeros_like(key_cache)
    key_cache[2, :2] = torch.tensor(
        [[[1, 0]], [[0, 1]]],
        dtype=torch.bfloat16,
    )
    value_cache[2, :2] = torch.tensor(
        [[[1, 2]], [[3, 4]]],
        dtype=torch.bfloat16,
    )
    long_key = (
        torch.arange(
            257 * kv_heads * head_dim,
            dtype=torch.float32,
        ).reshape(257, kv_heads, head_dim)
        .remainder(17)
        .to(torch.bfloat16)
    )
    long_value = (
        torch.arange(
            257 * kv_heads * head_dim,
            dtype=torch.float32,
        ).reshape(257, kv_heads, head_dim)
        .remainder(23)
        .to(torch.bfloat16)
    )
    key_cache[3] = long_key[:block_size]
    value_cache[3] = long_value[:block_size]
    key_cache[1, 0] = long_key[block_size]
    value_cache[1, 0] = long_value[block_size]
    context = SimpleNamespace(
        block_tables=torch.tensor(
            [
                [2, 0],
                [3, 1],
            ],
            dtype=torch.int32,
        ),
        context_lens=torch.tensor([3, 258], dtype=torch.int32),
        slot_mapping=torch.tensor(
            [
                2 * block_size + 2,
                1 * block_size + 1,
            ],
            dtype=torch.int32,
        ),
    )
    eager_key_cache = key_cache.clone()
    eager_value_cache = value_cache.clone()
    graph_key_cache = key_cache.clone()
    graph_value_cache = value_cache.clone()

    distributed = full_attention.torch.distributed
    with patch.object(
        distributed,
        "is_initialized",
        return_value=True,
    ), patch.object(
        distributed,
        "get_world_size",
        return_value=4,
    ), patch.object(
        distributed,
        "get_rank",
        return_value=2,
    ):
        expected = full_attention.qwen35_cached_decode_eager_attention(
            query,
            current_key,
            current_value,
            eager_key_cache,
            eager_value_cache,
            context,
            num_heads=query_heads,
            head_dim=head_dim,
            scale=head_dim ** -0.5,
        )
        actual = full_attention.qwen35_cached_decode_graph_attention(
            query,
            current_key,
            current_value,
            graph_key_cache,
            graph_value_cache,
            context,
            num_heads=query_heads,
            head_dim=head_dim,
            scale=head_dim ** -0.5,
        )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        graph_key_cache,
        eager_key_cache,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        graph_value_cache,
        eager_value_cache,
        atol=0,
        rtol=0,
    )


def test_cached_decode_graph_captures_paged_cache_on_cuda() -> None:
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    block_size = 256
    torch.manual_seed(31)
    query = torch.randn(
        2,
        2,
        4,
        dtype=torch.bfloat16,
        device=device,
    )
    current_key = torch.randn(
        2,
        1,
        4,
        dtype=torch.bfloat16,
        device=device,
    )
    current_value = torch.randn_like(current_key)
    key_cache = torch.randn(
        4,
        block_size,
        1,
        4,
        dtype=torch.bfloat16,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    context = SimpleNamespace(
        block_tables=torch.tensor(
            [
                [2, 0],
                [3, 1],
            ],
            dtype=torch.int32,
            device=device,
        ),
        context_lens=torch.tensor(
            [3, 258],
            dtype=torch.int32,
            device=device,
        ),
        slot_mapping=torch.tensor(
            [
                2 * block_size + 2,
                1 * block_size + 1,
            ],
            dtype=torch.int32,
            device=device,
        ),
    )
    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        full_attention.qwen35_cached_decode_graph_attention(
            query,
            current_key,
            current_value,
            key_cache.clone(),
            value_cache.clone(),
            context,
            num_heads=2,
            head_dim=4,
            scale=4 ** -0.5,
        )
    torch.cuda.current_stream().wait_stream(warm_stream)
    graph_key_cache = key_cache.clone()
    graph_value_cache = value_cache.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = full_attention.qwen35_cached_decode_graph_attention(
            query,
            current_key,
            current_value,
            graph_key_cache,
            graph_value_cache,
            context,
            num_heads=2,
            head_dim=4,
            scale=4 ** -0.5,
        )
    next_query = torch.randn_like(query)
    next_key = torch.randn_like(current_key)
    next_value = torch.randn_like(current_value)
    context.context_lens.add_(1)
    context.slot_mapping.add_(1)
    expected = full_attention.qwen35_cached_decode_eager_attention(
        next_query,
        next_key,
        next_value,
        graph_key_cache.clone(),
        graph_value_cache.clone(),
        context,
        num_heads=2,
        head_dim=4,
        scale=4 ** -0.5,
    )
    query.copy_(next_query)
    current_key.copy_(next_key)
    current_value.copy_(next_value)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual,
        expected,
        atol=2e-2,
        rtol=2e-2,
    )


def test_cached_decode_eager_matches_official_expand_gqa_cuda_reduction() -> None:
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    fixture = torch.load(
        ROOT / "tools/fixtures/qwen35_layer7_attention_reduction_r404.pt",
        map_location=device,
        weights_only=True,
    )
    context_length = 1096
    block_size = 256
    head_dim = 256
    query_heads = 2
    kv_heads = 1
    block_count = (context_length + block_size - 1) // block_size
    query = torch.zeros(
        1,
        query_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    dense_key = torch.zeros(
        context_length,
        kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    dense_value = torch.zeros_like(dense_key)
    dense_value[:, 0, fixture["column"]] = fixture["value_column"]
    key_cache = torch.zeros(
        block_count,
        block_size,
        kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    value_cache = torch.zeros_like(key_cache)
    key_cache.reshape(-1, kv_heads, head_dim)[:context_length - 1] = (
        dense_key[:-1]
    )
    value_cache.reshape(-1, kv_heads, head_dim)[:context_length - 1] = (
        dense_value[:-1]
    )
    context = SimpleNamespace(
        block_tables=torch.arange(
            block_count,
            dtype=torch.int32,
            device=device,
        ).unsqueeze(0),
        context_lens=torch.tensor(
            [context_length],
            dtype=torch.int32,
            device=device,
        ),
        slot_mapping=torch.tensor(
            [context_length - 1],
            dtype=torch.int32,
            device=device,
        ),
    )

    with patch.object(
        full_attention.torch,
        "softmax",
        return_value=fixture["probabilities"].repeat(1, 4, 1, 1),
    ), patch.object(
        full_attention.torch.distributed,
        "is_initialized",
        return_value=True,
    ), patch.object(
        full_attention.torch.distributed,
        "get_world_size",
        return_value=4,
    ), patch.object(
        full_attention.torch.distributed,
        "get_rank",
        return_value=0,
    ):
        actual = full_attention.qwen35_cached_decode_eager_attention(
            query,
            dense_key[-1:],
            dense_value[-1:],
            key_cache,
            value_cache,
            context,
            num_heads=query_heads,
            head_dim=head_dim,
            scale=head_dim ** -0.5,
        )
    target = actual[0].reshape(query_heads, head_dim)[
        fixture["head"],
        fixture["column"],
    ]

    assert torch.equal(target, fixture["expected_scalar"])


def test_cached_decode_eager_matches_official_full_batch_cuda_reduction() -> None:
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    fixture = torch.load(
        ROOT
        / (
            "tools/fixtures/"
            "qwen35_layer7_step19_full_softmax_reduction_r436.pt"
        ),
        map_location=device,
        weights_only=True,
    )
    context_length = fixture["context_length"]
    head_dim = fixture["head_dim"]
    query_heads = fixture["local_query_heads"]
    kv_heads = 1
    block_size = 256
    block_count = (context_length + block_size - 1) // block_size
    global_head = fixture["global_head"]
    local_scores = fixture["full_scores"][
        :, global_head:global_head + query_heads
    ].to(torch.bfloat16)
    dense_value = (
        fixture["full_value"][:, global_head:global_head + 1]
        .to(torch.bfloat16).squeeze(0)
        .transpose(0, 1)
        .contiguous()
    )
    query = torch.zeros(
        1,
        query_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    dense_key = torch.zeros_like(dense_value)
    key_cache = torch.zeros(
        block_count,
        block_size,
        kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    value_cache = torch.zeros_like(key_cache)
    key_cache.reshape(-1, kv_heads, head_dim)[:context_length - 1] = (
        dense_key[:-1]
    )
    value_cache.reshape(-1, kv_heads, head_dim)[:context_length - 1] = (
        dense_value[:-1]
    )
    context = SimpleNamespace(
        block_tables=torch.arange(
            block_count,
            dtype=torch.int32,
            device=device,
        ).unsqueeze(0),
        context_lens=torch.tensor(
            [context_length],
            dtype=torch.int32,
            device=device,
        ),
        slot_mapping=torch.tensor(
            [context_length - 1],
            dtype=torch.int32,
            device=device,
        ),
    )

    original_matmul = full_attention.torch.matmul
    matmul_calls = 0

    def matmul_with_fixture(left, right):
        nonlocal matmul_calls
        matmul_calls += 1
        if matmul_calls == 1:
            return local_scores / (head_dim ** -0.5)
        return original_matmul(left, right)

    with patch.object(
        full_attention.torch,
        "matmul",
        side_effect=matmul_with_fixture,
    ), patch.object(
        full_attention.torch.distributed,
        "is_initialized",
        return_value=True,
    ), patch.object(
        full_attention.torch.distributed,
        "get_world_size",
        return_value=4,
    ), patch.object(
        full_attention.torch.distributed,
        "get_rank",
        return_value=1,
    ):
        actual = full_attention.qwen35_cached_decode_eager_attention(
            query,
            dense_key[-1:],
            dense_value[-1:],
            key_cache,
            value_cache,
            context,
            num_heads=query_heads,
            head_dim=head_dim,
            scale=head_dim ** -0.5,
        )

    expected = fixture["expected_output"].to(torch.bfloat16)
    torch.testing.assert_close(
        actual.reshape(query_heads, head_dim)[0],
        expected,
        atol=0,
        rtol=0,
    )
    assert matmul_calls == 2


def test_prefill_blockwise_matches_dense_gqa_and_bounds_tiles() -> None:
    torch.manual_seed(17)
    query = torch.randn(7, 4, 3, dtype=torch.float32)
    key = torch.randn(7, 2, 3, dtype=torch.float32)
    value = torch.randn(7, 2, 3, dtype=torch.float32)
    expected = _official_prefill_eager(
        query,
        key,
        value,
    ).reshape(7, 12)
    score_tiles = []
    original_matmul = full_attention.torch.matmul

    def record_matmul(left, right):
        if (
            left.ndim == 3
            and right.ndim == 3
            and left.shape[-1] == 3
            and right.shape[-2] == 3
        ):
            score_tiles.append(
                (int(left.shape[-2]), int(right.shape[-1]))
            )
        return original_matmul(left, right)

    with patch.object(
        full_attention.torch,
        "matmul",
        side_effect=record_matmul,
    ):
        actual = (
            full_attention.qwen35_prefill_blockwise_attention(
                query,
                key,
                value,
                num_heads=4,
                head_dim=3,
                scale=3 ** -0.5,
                block_tokens=3,
            )
        )

    assert actual.shape == (7, 12)
    assert torch.isfinite(actual).all()
    assert score_tiles
    assert all(
        query_tokens <= 3 and key_tokens <= 3
        for query_tokens, key_tokens in score_tiles
    )
    torch.testing.assert_close(
        actual,
        expected,
        atol=2e-6,
        rtol=2e-6,
    )


def test_prefill_public_blockwise_preserves_request_isolation() -> None:
    torch.manual_seed(23)
    query = torch.randn(7, 4, 3, dtype=torch.float32)
    key = torch.randn(7, 2, 3, dtype=torch.float32)
    value = torch.randn(7, 2, 3, dtype=torch.float32)
    context = SimpleNamespace(
        cu_seqlens_q=torch.tensor(
            [0, 3, 7],
            dtype=torch.int32,
        ),
        prefill_attention_reference_lens=(3, 4),
        qwen35_prefill_blockwise_threshold=0,
        qwen35_prefill_block_tokens=2,
    )
    score_tiles = []
    original_matmul = full_attention.torch.matmul

    def record_matmul(left, right):
        if (
            left.ndim == 3
            and right.ndim == 3
            and left.shape[-1] == 3
            and right.shape[-2] == 3
        ):
            score_tiles.append(
                (int(left.shape[-2]), int(right.shape[-1]))
            )
        return original_matmul(left, right)

    with patch.object(
        full_attention.torch,
        "matmul",
        side_effect=record_matmul,
    ):
        actual = full_attention.qwen35_prefill_eager_attention(
            query,
            key,
            value,
            context,
            num_heads=4,
            head_dim=3,
            scale=3 ** -0.5,
        )
    expected = torch.cat(
        (
            _official_prefill_eager(
                query[:3],
                key[:3],
                value[:3],
            ).reshape(3, 12),
            _official_prefill_eager(
                query[3:],
                key[3:],
                value[3:],
            ).reshape(4, 12),
        ),
        dim=0,
    )

    torch.testing.assert_close(
        actual,
        expected,
        atol=2e-6,
        rtol=2e-6,
    )
    assert score_tiles
    assert all(
        query_tokens <= 2 and key_tokens <= 2
        for query_tokens, key_tokens in score_tiles
    )


def test_prefill_public_blockwise_ignores_future_padding_for_kv() -> None:
    torch.manual_seed(29)
    query = torch.randn(3, 4, 3, dtype=torch.float32)
    key = torch.randn(3, 2, 3, dtype=torch.float32)
    value = torch.randn(3, 2, 3, dtype=torch.float32)
    key_cache = torch.zeros(4, 2, 2, 3, dtype=torch.float32)
    value_cache = torch.zeros_like(key_cache)
    context = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
        slot_mapping=torch.tensor([1, 4, 7], dtype=torch.int32),
        prefill_attention_reference_lens=(6,),
        qwen35_prefill_blockwise_threshold=0,
        qwen35_prefill_block_tokens=2,
    )
    score_tiles = []
    original_matmul = full_attention.torch.matmul

    def record_matmul(left, right):
        if (
            left.ndim == 3
            and right.ndim == 3
            and left.shape[-1] == 3
            and right.shape[-2] == 3
        ):
            score_tiles.append(
                (int(left.shape[-2]), int(right.shape[-1]))
            )
        return original_matmul(left, right)

    with patch.object(
        full_attention.torch,
        "matmul",
        side_effect=record_matmul,
    ):
        actual = full_attention.qwen35_prefill_eager_attention(
            query,
            key,
            value,
            context,
            num_heads=4,
            head_dim=3,
            scale=3 ** -0.5,
            key_cache=key_cache,
            value_cache=value_cache,
        )
    expected = _official_prefill_eager(
        query,
        key,
        value,
    ).reshape(3, 12)

    torch.testing.assert_close(
        actual,
        expected,
        atol=2e-6,
        rtol=2e-6,
    )
    assert score_tiles
    assert all(
        query_tokens <= 2 and key_tokens <= 2
        for query_tokens, key_tokens in score_tiles
    )
    for slot, expected_key, expected_value in zip(
        context.slot_mapping.tolist(),
        key,
        value,
    ):
        block = slot // key_cache.shape[1]
        offset = slot % key_cache.shape[1]
        torch.testing.assert_close(
            key_cache[block, offset],
            expected_key,
        )
        torch.testing.assert_close(
            value_cache[block, offset],
            expected_value,
        )
    assert torch.count_nonzero(key_cache[1]).item() == 0
    assert torch.count_nonzero(value_cache[1]).item() == 0
