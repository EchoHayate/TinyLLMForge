from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


AttentionMode = Literal["prefill", "decode", "spec_verify"]
SPEC_VERIFY_FLASH_ATTN_NUM_SPLITS = 16


@dataclass(frozen=True)
class SpecVerifyPlan:
    input_tokens: tuple[int, ...]
    positions: tuple[int, ...]
    logical_slots: tuple[int, ...]
    context_len: int
    visible_block_count: int

    @property
    def query_len(self) -> int:
        return len(self.input_tokens)


@dataclass(frozen=True)
class SpecVerifyMetadata:
    query_len: int
    input_tokens: tuple[int, ...]
    positions: tuple[int, ...]
    logical_slots: tuple[int, ...]
    physical_slots: tuple[int, ...]
    context_len: int
    block_table: tuple[int, ...]


def build_spec_verify_plan(
    history_len: int,
    draft_tokens: list[int],
    block_size: int,
) -> SpecVerifyPlan:
    history_len = int(history_len)
    block_size = int(block_size)
    if history_len < 1:
        raise ValueError("spec_verify requires history_len >= 1")
    if not draft_tokens:
        raise ValueError("spec_verify requires at least one draft token")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    input_tokens = tuple(int(token_id) for token_id in draft_tokens[:-1])
    query_len = len(input_tokens)
    positions = tuple(
        range(history_len + 1, history_len + 1 + query_len)
    )
    logical_slots = tuple(range(history_len, history_len + query_len))
    context_len = history_len + query_len
    visible_block_count = (
        (context_len + block_size - 1) // block_size
        if context_len > 0
        else 0
    )
    return SpecVerifyPlan(
        input_tokens=input_tokens,
        positions=positions,
        logical_slots=logical_slots,
        context_len=context_len,
        visible_block_count=visible_block_count,
    )


def validate_spec_verify_slots(
    plan: SpecVerifyPlan,
    proxy_block_table: list[int],
    block_size: int,
) -> tuple[int, ...]:
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if len(proxy_block_table) < plan.visible_block_count:
        raise ValueError("proxy block table does not cover verifier context")

    physical_slots = []
    for logical_slot in plan.logical_slots:
        block_index = logical_slot // block_size
        if block_index >= len(proxy_block_table):
            raise ValueError("logical verifier slot is out of range")
        block_id = int(proxy_block_table[block_index])
        if block_id < 0:
            raise ValueError("verifier block table contains an invalid block")
        physical_slots.append(
            block_id * block_size + logical_slot % block_size
        )
    return tuple(physical_slots)


def spec_verify_metadata_to_dict(
    metadata: SpecVerifyMetadata,
) -> dict[str, object]:
    payload = asdict(metadata)
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in payload.items()
    }
