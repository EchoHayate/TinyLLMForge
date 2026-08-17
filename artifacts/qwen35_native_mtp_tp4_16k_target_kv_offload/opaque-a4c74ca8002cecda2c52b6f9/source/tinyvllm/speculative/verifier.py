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


def _validate_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _validate_non_negative_integer(
    value: object,
    name: str,
) -> int:
    normalized = _validate_integer(value, name)
    if normalized < 0:
        raise ValueError(f"{name} must be >= 0")
    return normalized


def _validate_positive_integer(
    value: object,
    name: str,
) -> int:
    normalized = _validate_integer(value, name)
    if normalized <= 0:
        raise ValueError(f"{name} must be > 0")
    return normalized


def _validate_integer_tuple(
    value: object,
    name: str,
    *,
    non_negative: bool = False,
) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple")
    for item in value:
        normalized = _validate_integer(item, f"{name} item")
        if non_negative and normalized < 0:
            raise ValueError(
                f"{name} items must be >= 0"
            )
    return value


@dataclass(frozen=True)
class SpecVerifyBatchRowMetadata:
    sequence_id: int
    batch_index: int
    query_offset: int
    query_len: int
    input_tokens: tuple[int, ...]
    positions: tuple[int, ...]
    logical_slots: tuple[int, ...]
    physical_slots: tuple[int, ...]
    context_len: int
    block_table: tuple[int, ...]

    def __post_init__(self):
        _validate_non_negative_integer(
            self.sequence_id,
            "sequence_id",
        )
        _validate_non_negative_integer(
            self.batch_index,
            "batch_index",
        )
        _validate_non_negative_integer(
            self.query_offset,
            "query_offset",
        )
        query_len = _validate_positive_integer(
            self.query_len,
            "query_len",
        )
        fields = (
            ("input_tokens", self.input_tokens, False),
            ("positions", self.positions, True),
            ("logical_slots", self.logical_slots, True),
            ("physical_slots", self.physical_slots, True),
        )
        for name, value, non_negative in fields:
            normalized = _validate_integer_tuple(
                value,
                name,
                non_negative=non_negative,
            )
            if len(normalized) != query_len:
                raise ValueError(
                    f"{name} length must match query_len"
                )
        _validate_positive_integer(
            self.context_len,
            "context_len",
        )
        block_table = _validate_integer_tuple(
            self.block_table,
            "block_table",
            non_negative=True,
        )
        if not block_table:
            raise ValueError(
                "block_table must not be empty"
            )


@dataclass(frozen=True)
class SpecVerifyBatchMetadata:
    rows: tuple[SpecVerifyBatchRowMetadata, ...]
    query_len: int
    total_query_tokens: int
    block_table_width: int

    def __post_init__(self):
        if not isinstance(self.rows, tuple) or not self.rows:
            raise ValueError(
                "rows must be a non-empty tuple"
            )
        query_len = _validate_positive_integer(
            self.query_len,
            "query_len",
        )
        total_query_tokens = (
            _validate_positive_integer(
                self.total_query_tokens,
                "total_query_tokens",
            )
        )
        block_table_width = _validate_positive_integer(
            self.block_table_width,
            "block_table_width",
        )
        sequence_ids = []
        expected_offset = 0
        for batch_index, row in enumerate(self.rows):
            if not isinstance(
                row,
                SpecVerifyBatchRowMetadata,
            ):
                raise ValueError(
                    "rows must contain batch row metadata"
                )
            if row.batch_index != batch_index:
                raise ValueError(
                    "batch row index mismatch"
                )
            if row.query_offset != expected_offset:
                raise ValueError(
                    "batch row query offset mismatch"
                )
            if row.query_len != query_len:
                raise ValueError(
                    "batch row query length mismatch"
                )
            if len(row.block_table) > block_table_width:
                raise ValueError(
                    "batch row block table exceeds width"
                )
            sequence_ids.append(row.sequence_id)
            expected_offset += query_len
        if len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError(
                "batch row sequence IDs must be unique"
            )
        if total_query_tokens != expected_offset:
            raise ValueError(
                "total query token count mismatch"
            )


@dataclass(frozen=True)
class SpecVerifyBatchResultRow:
    sequence_id: int
    target_tokens: tuple[int, ...]

    def __post_init__(self):
        _validate_non_negative_integer(
            self.sequence_id,
            "sequence_id",
        )
        target_tokens = _validate_integer_tuple(
            self.target_tokens,
            "target_tokens",
        )
        if not target_tokens:
            raise ValueError(
                "target_tokens must not be empty"
            )


def split_spec_verify_batch_target_tokens(
    metadata: SpecVerifyBatchMetadata,
    flat_target_tokens: tuple[int, ...],
) -> tuple[SpecVerifyBatchResultRow, ...]:
    if not isinstance(metadata, SpecVerifyBatchMetadata):
        raise ValueError(
            "metadata must be SpecVerifyBatchMetadata"
        )
    normalized_tokens = _validate_integer_tuple(
        flat_target_tokens,
        "flat target tokens",
    )
    if (
        len(normalized_tokens)
        != metadata.total_query_tokens
    ):
        raise ValueError(
            "flat target token count mismatch"
        )
    return tuple(
        SpecVerifyBatchResultRow(
            sequence_id=row.sequence_id,
            target_tokens=normalized_tokens[
                row.query_offset:
                row.query_offset + row.query_len
            ],
        )
        for row in metadata.rows
    )


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
        range(history_len, history_len + query_len)
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
