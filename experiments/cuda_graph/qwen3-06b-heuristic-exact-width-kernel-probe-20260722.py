import json
import math

import torch
from flash_attn import flash_attn_with_kvcache


def ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


def heuristic_split(batch_size, width):
    num_query_heads = 16
    num_kv_heads = 8
    head_dim = 128
    page_block_size = 256
    multi_processor_count = 108
    effective_heads = num_kv_heads
    effective_seqlen_q = num_query_heads // num_kv_heads
    block_n = 128
    num_n_blocks = ceildiv(width * page_block_size, block_n)
    num_m_blocks = ceildiv(effective_seqlen_q, 64)
    work = batch_size * effective_heads * num_m_blocks
    num_sms = multi_processor_count * 2
    if work >= 0.8 * num_sms:
        return 1
    max_splits = min(128, num_sms, num_n_blocks)
    candidates = []
    best = 0.0
    for num_splits in range(1, max_splits + 1):
        if (
            num_splits > 1
            and ceildiv(num_n_blocks, num_splits)
            == ceildiv(num_n_blocks, num_splits - 1)
        ):
            continue
        waves = work * num_splits / num_sms
        efficiency = waves / math.ceil(waves)
        candidates.append((num_splits, efficiency))
        best = max(best, efficiency)
    return next(
        num_splits
        for num_splits, efficiency in candidates
        if efficiency >= 0.85 * best
    )


def kernel_inputs(batch_size, width):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260722 + batch_size * 10 + width)
    num_blocks = batch_size * width
    q = torch.randn(
        batch_size,
        1,
        16,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    k_cache = torch.randn(
        num_blocks,
        256,
        8,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    v_cache = torch.randn(
        num_blocks,
        256,
        8,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    block_table = torch.arange(
        num_blocks,
        device="cuda",
        dtype=torch.int32,
    ).reshape(batch_size, width)
    cache_seqlens = torch.tensor(
        [
            max(1, width * 256 - 1 - 7 * row)
            for row in range(batch_size)
        ],
        device="cuda",
        dtype=torch.int32,
    )
    return q, k_cache, v_cache, cache_seqlens, block_table


def run_attention(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    block_table,
    num_splits,
):
    return flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
        num_splits=num_splits,
    )


def difference(left, right):
    absolute = (left.float() - right.float()).abs()
    return {
        "bitwise_equal": bool(torch.equal(left, right)),
        "different_elements": int(torch.count_nonzero(left != right).item()),
        "max_abs_error": float(absolute.max().item()),
    }


def graph_replay(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    block_table,
    num_splits,
):
    for _ in range(3):
        output = run_attention(
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            block_table,
            num_splits,
        )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run_attention(
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            block_table,
            num_splits,
        )
    graph.replay()
    torch.cuda.synchronize()
    return output.clone()


results = {
    "environment": {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
    },
    "auto_vs_explicit": [],
}
for batch_size, width, expected_split in (
    (2, 1, 2),
    (5, 1, 2),
    (8, 2, 2),
    (9, 2, 2),
    (16, 3, 3),
):
    inputs = kernel_inputs(batch_size, width)
    derived_split = heuristic_split(batch_size, width)
    auto = run_attention(*inputs, num_splits=0)
    explicit = run_attention(*inputs, num_splits=derived_split)
    results["auto_vs_explicit"].append(
        {
            "batch_size": batch_size,
            "page_table_width": width,
            "expected_split": expected_split,
            "derived_split": derived_split,
            **difference(auto, explicit),
        }
    )

batch_size = 5
runtime_width = 1
derived_split = heuristic_split(batch_size, runtime_width)
inputs = kernel_inputs(batch_size, runtime_width)
eager_auto = run_attention(*inputs, num_splits=0)
exact_graph = graph_replay(*inputs, num_splits=derived_split)
results["exact_width_graph"] = {
    "batch_size": batch_size,
    "runtime_width": runtime_width,
    "capture_width": runtime_width,
    "derived_split": derived_split,
    **difference(eager_auto, exact_graph),
}

q, k_cache, v_cache, cache_seqlens, block_table = inputs
padded_cache = torch.cat(
    [
        k_cache,
        torch.zeros(
            batch_size * 3,
            256,
            8,
            128,
            device="cuda",
            dtype=torch.bfloat16,
        ),
    ],
    dim=0,
)
padded_values = torch.cat(
    [
        v_cache,
        torch.zeros(
            batch_size * 3,
            256,
            8,
            128,
            device="cuda",
            dtype=torch.bfloat16,
        ),
    ],
    dim=0,
)
padded_table = torch.cat(
    [
        block_table,
        torch.zeros(
            batch_size,
            3,
            device="cuda",
            dtype=torch.int32,
        ),
    ],
    dim=1,
)
padded_graph = graph_replay(
    q,
    padded_cache,
    padded_values,
    cache_seqlens,
    padded_table,
    derived_split,
)
results["padded_width_negative_control"] = {
    "batch_size": batch_size,
    "runtime_width": runtime_width,
    "capture_width": 4,
    "derived_split": derived_split,
    **difference(eager_auto, padded_graph),
}
results["passed"] = (
    all(
        row["derived_split"] == row["expected_split"]
        and row["bitwise_equal"]
        for row in results["auto_vs_explicit"]
    )
    and results["exact_width_graph"]["bitwise_equal"]
    and not results["padded_width_negative_control"]["bitwise_equal"]
)
print(json.dumps(results, sort_keys=True))
raise SystemExit(0 if results["passed"] else 1)
