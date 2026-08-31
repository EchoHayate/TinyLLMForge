from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass


FLASH_ATTN_VERSION = "2.6.3"


def _ceildiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class FlashAttentionSplitInputs:
    batch_size: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    page_block_size: int
    page_table_width: int
    max_seqlen_q: int
    multi_processor_count: int

    def validate(self) -> None:
        positive_fields = (
            "batch_size",
            "num_query_heads",
            "num_kv_heads",
            "head_dim",
            "page_block_size",
            "page_table_width",
            "max_seqlen_q",
            "multi_processor_count",
        )
        for field in positive_fields:
            if int(getattr(self, field)) <= 0:
                raise ValueError(f"{field} must be positive")
        if self.head_dim > 256:
            raise ValueError("head_dim must be at most 256")
        if self.page_block_size % 256 != 0:
            raise ValueError("page_block_size must be divisible by 256")
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_query_heads must be divisible by num_kv_heads"
            )
        if self.max_seqlen_q != 1:
            raise ValueError(
                "FlashAttention 2.6.3 decode policy requires max_seqlen_q=1"
            )


@dataclass(frozen=True)
class FlashAttentionGraphIdentity:
    graph_batch_size: int
    active_batch_size: int
    page_table_width: int
    effective_num_splits: int
    flash_attn_version: str
    multi_processor_count: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    page_block_size: int
    max_seqlen_q: int
    execution_protocol: str = "forward_v1"
    state_schema_sha256: str = ""
    lease_seal: str = ""

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json_bytes(asdict(self))
        ).hexdigest()


def flash_attn_263_decode_num_splits(
    inputs: FlashAttentionSplitInputs,
) -> int:
    inputs.validate()
    swapped = (
        inputs.max_seqlen_q == 1
        and inputs.num_query_heads > inputs.num_kv_heads
    )
    effective_heads = (
        inputs.num_kv_heads if swapped else inputs.num_query_heads
    )
    effective_seqlen_q = (
        inputs.num_query_heads // inputs.num_kv_heads
        if swapped
        else inputs.max_seqlen_q
    )
    block_n = (
        256
        if inputs.head_dim <= 64
        else 128
        if inputs.head_dim <= 128
        else 64
    )
    seqlen_k = inputs.page_table_width * inputs.page_block_size
    num_n_blocks = _ceildiv(seqlen_k, block_n)
    num_m_blocks = _ceildiv(effective_seqlen_q, 64)
    batch_nheads_mblocks = (
        inputs.batch_size * effective_heads * num_m_blocks
    )
    num_sms = inputs.multi_processor_count * 2
    if batch_nheads_mblocks >= 0.8 * num_sms:
        return 1

    max_splits = min(128, num_sms, num_n_blocks)
    candidates = []
    max_efficiency = 0.0
    for num_splits in range(1, max_splits + 1):
        eligible = (
            num_splits == 1
            or _ceildiv(num_n_blocks, num_splits)
            != _ceildiv(num_n_blocks, num_splits - 1)
        )
        if not eligible:
            continue
        waves = batch_nheads_mblocks * num_splits / num_sms
        efficiency = waves / math.ceil(waves)
        candidates.append((num_splits, efficiency))
        max_efficiency = max(max_efficiency, efficiency)

    for num_splits, efficiency in candidates:
        if efficiency >= 0.85 * max_efficiency:
            return num_splits
    return 1


def build_flash_attn_263_graph_identity(
    *,
    graph_batch_size: int,
    inputs: FlashAttentionSplitInputs,
    flash_attn_version: str,
    require_exact_batch: bool = False,
    execution_protocol: str = "forward_v1",
    state_schema_sha256: str = "",
    lease_seal: str = "",
) -> FlashAttentionGraphIdentity:
    if flash_attn_version != FLASH_ATTN_VERSION:
        raise ValueError(
            "heuristic graph identity requires FlashAttention 2.6.3"
        )
    if graph_batch_size < inputs.batch_size:
        raise ValueError(
            "graph_batch_size must be at least the active batch size"
        )
    if require_exact_batch and graph_batch_size != inputs.batch_size:
        raise ValueError(
            "production graph_batch_size must equal active batch size"
        )
    return FlashAttentionGraphIdentity(
        graph_batch_size=int(graph_batch_size),
        active_batch_size=int(inputs.batch_size),
        page_table_width=int(inputs.page_table_width),
        effective_num_splits=flash_attn_263_decode_num_splits(inputs),
        flash_attn_version=flash_attn_version,
        multi_processor_count=int(inputs.multi_processor_count),
        num_query_heads=int(inputs.num_query_heads),
        num_kv_heads=int(inputs.num_kv_heads),
        head_dim=int(inputs.head_dim),
        page_block_size=int(inputs.page_block_size),
        max_seqlen_q=int(inputs.max_seqlen_q),
        execution_protocol=execution_protocol,
        state_schema_sha256=state_schema_sha256,
        lease_seal=lease_seal,
    )
