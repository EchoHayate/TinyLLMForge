"""KV-Cartridge v0 helpers.

v0 是无训练、秒级可用的 read-side KV 压缩：不改写 KV cache 内容，只把 decode
attention 看到的历史 block table 压缩成少量代表性 block。这样可以快速扫 needle
曲线，先验证“高压缩比上下文记忆”是否有质量空间。
"""

from __future__ import annotations


def select_uniform_cartridge_indices(num_blocks: int, budget: int) -> list[int]:
    """Return sorted logical block indices for a uniform KV cartridge.

    Always preserves the first block (attention sink) and last block (recency / current
    partial block). Middle blocks are evenly spaced. If the budget covers the full
    sequence, the full range is returned.
    """
    if num_blocks <= 0:
        return []
    if budget <= 0 or budget >= num_blocks:
        return list(range(num_blocks))
    if budget == 1:
        return [num_blocks - 1]

    step = (num_blocks - 1) / (budget - 1)
    indices = [round(i * step) for i in range(budget)]
    indices[0] = 0
    indices[-1] = num_blocks - 1

    # round() can collide for tiny budgets; fill any gaps deterministically while
    # preserving order and the mandatory first/last blocks.
    unique = []
    seen = set()
    for idx in indices:
        idx = min(max(idx, 0), num_blocks - 1)
        if idx not in seen:
            unique.append(idx)
            seen.add(idx)
    if len(unique) < budget:
        for idx in range(num_blocks):
            if idx not in seen:
                unique.append(idx)
                seen.add(idx)
                if len(unique) == budget:
                    break
    return sorted(unique)


def should_use_kv_cartridge(
    seq_lens: list[int],
    num_blocks: list[int],
    budget: int,
    min_seq_len: int,
) -> bool:
    """Return whether a whole decode batch should use KV-Cartridge compression.

    The current engine uses one block table shape per batch. v0 therefore enables the
    cartridge only when every row is long enough and every row has more blocks than the
    requested budget; otherwise the batch falls back to full attention.
    """
    if budget <= 0 or not seq_lens or not num_blocks:
        return False
    return min(seq_lens) >= min_seq_len and min(num_blocks) > budget


def compress_decode_block_table_rows(
    block_table_rows: list[list[int]],
    context_lens: list[int],
    block_size: int,
    budget: int,
) -> tuple[list[list[int]], list[int]]:
    """Compress per-row physical block ids and return compact cache lengths.

    `block_table_rows` contains physical KV block ids in original order. The returned
    rows keep a subset of those physical ids. `context_lens` are rewritten to the
    compact logical length that FlashAttention should see for each compressed row.
    """
    compressed_rows: list[list[int]] = []
    compressed_lens: list[int] = []
    for row, context_len in zip(block_table_rows, context_lens):
        num_blocks = (context_len + block_size - 1) // block_size
        active_row = row[:num_blocks]
        indices = select_uniform_cartridge_indices(num_blocks, budget)
        compressed = [active_row[i] for i in indices]
        last_partial = ((context_len - 1) % block_size) + 1 if context_len > 0 else 0
        if not compressed:
            compact_len = 0
        elif indices[-1] == num_blocks - 1:
            compact_len = (len(compressed) - 1) * block_size + last_partial
        else:
            compact_len = len(compressed) * block_size
        compressed_rows.append(compressed)
        compressed_lens.append(compact_len)
    return compressed_rows, compressed_lens
