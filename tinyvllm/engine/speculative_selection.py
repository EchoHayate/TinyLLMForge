from __future__ import annotations

from dataclasses import dataclass
import math


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


@dataclass(frozen=True)
class SpeculativeSelectionConfig:
    enabled: bool
    max_proposal_tokens: int

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        max_proposal_tokens = _validate_non_negative_integer(
            self.max_proposal_tokens,
            "max_proposal_tokens",
        )
        if self.enabled and max_proposal_tokens < 2:
            raise ValueError(
                "enabled selection requires "
                "max_proposal_tokens >= 2"
            )
        if not self.enabled and max_proposal_tokens != 0:
            raise ValueError(
                "disabled selection requires "
                "max_proposal_tokens == 0"
            )


@dataclass(frozen=True)
class SpeculativeSelectionRow:
    sequence_id: int
    batch_index: int
    token_count_snapshot: int
    completion_token_count_snapshot: int
    temperature_snapshot: float
    remaining_output_tokens: int
    selected: bool
    max_proposal_tokens: int
    suppression_reason: str | None


@dataclass(frozen=True)
class SpeculativeSelectionRecord:
    schedule_generation: int
    policy_branch: str
    is_prefill: bool
    do_sample: bool
    batch_kind: str | None
    scheduled_sequence_ids: tuple[int, ...]
    rows: tuple[SpeculativeSelectionRow, ...]

    @property
    def selected_rows(
        self,
    ) -> tuple[SpeculativeSelectionRow, ...]:
        return tuple(row for row in self.rows if row.selected)

    @property
    def selected_sequence_ids(self) -> tuple[int, ...]:
        return tuple(
            row.sequence_id for row in self.rows if row.selected
        )


def _sequence_values(
    seq: object,
) -> tuple[int, int, int, int, float]:
    sequence_id = _validate_integer(
        getattr(seq, "seq_id", None),
        "sequence_id",
    )
    num_tokens = _validate_non_negative_integer(
        getattr(seq, "num_tokens", None),
        "num_tokens",
    )
    completion_tokens = _validate_non_negative_integer(
        getattr(seq, "num_completion_tokens", None),
        "num_completion_tokens",
    )
    max_tokens = _validate_non_negative_integer(
        getattr(seq, "max_tokens", None),
        "max_tokens",
    )
    temperature_value = getattr(
        seq,
        "temperature",
        None,
    )
    if (
        isinstance(temperature_value, bool)
        or not isinstance(
            temperature_value,
            (int, float),
        )
    ):
        raise ValueError(
            "temperature must be a finite number"
        )
    temperature = float(temperature_value)
    if not math.isfinite(temperature):
        raise ValueError(
            "temperature must be a finite number"
        )
    return (
        sequence_id,
        num_tokens,
        completion_tokens,
        max_tokens,
        temperature,
    )


def build_speculative_selection_record(
    *,
    seqs: tuple[object, ...],
    is_prefill: bool,
    do_sample: bool,
    batch_kind: str | None,
    policy_branch: str,
    schedule_generation: int,
    config: SpeculativeSelectionConfig,
) -> SpeculativeSelectionRecord:
    if not isinstance(seqs, tuple) or not seqs:
        raise ValueError("scheduled sequences must be a non-empty tuple")
    if not isinstance(is_prefill, bool):
        raise ValueError("is_prefill must be a bool")
    if not isinstance(do_sample, bool):
        raise ValueError("do_sample must be a bool")
    if batch_kind is not None and not isinstance(batch_kind, str):
        raise ValueError("batch_kind must be a string or None")
    if not isinstance(policy_branch, str) or not policy_branch:
        raise ValueError("policy_branch must be non-empty")
    generation = _validate_integer(
        schedule_generation,
        "schedule_generation",
    )
    if generation <= 0:
        raise ValueError("schedule_generation must be > 0")
    if not isinstance(config, SpeculativeSelectionConfig):
        raise ValueError(
            "config must be SpeculativeSelectionConfig"
        )

    sequence_ids = []
    rows = []
    for batch_index, seq in enumerate(seqs):
        (
            sequence_id,
            num_tokens,
            completion_tokens,
            max_tokens,
            temperature,
        ) = _sequence_values(seq)
        sequence_ids.append(sequence_id)
        remaining_output_tokens = max(
            0,
            max_tokens - completion_tokens,
        )
        if batch_kind == "mixed":
            step_is_decode = getattr(
                seq,
                "step_is_decode",
                None,
            )
            step_do_sample = getattr(
                seq,
                "step_do_sample",
                None,
            )
            if not isinstance(step_is_decode, bool):
                raise ValueError(
                    "mixed step_is_decode must be a bool"
                )
            if not isinstance(step_do_sample, bool):
                raise ValueError(
                    "mixed step_do_sample must be a bool"
                )
            row_is_prefill = not step_is_decode
            row_do_sample = step_do_sample
        else:
            row_is_prefill = is_prefill
            row_do_sample = do_sample

        if not config.enabled:
            suppression_reason = "disabled"
        elif row_is_prefill:
            suppression_reason = "prefill"
        elif not row_do_sample:
            suppression_reason = "not_sampling"
        elif temperature != 0.0:
            suppression_reason = "non_greedy"
        elif remaining_output_tokens < 2:
            suppression_reason = (
                "insufficient_output_budget"
            )
        else:
            suppression_reason = None

        selected = suppression_reason is None
        max_proposal_tokens = (
            min(
                config.max_proposal_tokens,
                remaining_output_tokens,
            )
            if selected
            else 0
        )
        rows.append(
            SpeculativeSelectionRow(
                sequence_id=sequence_id,
                batch_index=batch_index,
                token_count_snapshot=num_tokens,
                completion_token_count_snapshot=(
                    completion_tokens
                ),
                temperature_snapshot=temperature,
                remaining_output_tokens=(
                    remaining_output_tokens
                ),
                selected=selected,
                max_proposal_tokens=max_proposal_tokens,
                suppression_reason=suppression_reason,
            )
        )

    if len(set(sequence_ids)) != len(sequence_ids):
        raise ValueError(
            "scheduled sequence IDs must be unique"
        )
    return SpeculativeSelectionRecord(
        schedule_generation=generation,
        policy_branch=policy_branch,
        is_prefill=is_prefill,
        do_sample=do_sample,
        batch_kind=batch_kind,
        scheduled_sequence_ids=tuple(sequence_ids),
        rows=tuple(rows),
    )


def validate_speculative_selection_record(
    record: SpeculativeSelectionRecord,
    seqs: tuple[object, ...],
    *,
    expected_schedule_generation: int,
) -> tuple[object, ...]:
    if not isinstance(record, SpeculativeSelectionRecord):
        raise ValueError(
            "record must be SpeculativeSelectionRecord"
        )
    generation = _validate_integer(
        expected_schedule_generation,
        "expected_schedule_generation",
    )
    if record.schedule_generation != generation:
        raise ValueError(
            "speculative selection generation mismatch"
        )
    if not isinstance(seqs, tuple):
        raise ValueError("scheduled sequences must be a tuple")
    current_ids = tuple(
        _validate_integer(
            getattr(seq, "seq_id", None),
            "sequence_id",
        )
        for seq in seqs
    )
    if current_ids != record.scheduled_sequence_ids:
        raise ValueError(
            "speculative selection sequence order mismatch"
        )
    if len(record.rows) != len(seqs):
        raise ValueError(
            "speculative selection row count mismatch"
        )

    selected = []
    for batch_index, (row, seq) in enumerate(
        zip(record.rows, seqs)
    ):
        (
            sequence_id,
            num_tokens,
            completion_tokens,
            max_tokens,
            temperature,
        ) = _sequence_values(seq)
        if (
            row.batch_index != batch_index
            or row.sequence_id != sequence_id
        ):
            raise ValueError(
                "speculative selection row order mismatch"
            )
        if row.token_count_snapshot != num_tokens:
            raise ValueError(
                "speculative selection token count is stale"
            )
        if (
            row.completion_token_count_snapshot
            != completion_tokens
        ):
            raise ValueError(
                "speculative selection completion count is stale"
            )
        if row.temperature_snapshot != temperature:
            raise ValueError(
                "speculative selection temperature is stale"
            )
        remaining_output_tokens = max(
            0,
            max_tokens - completion_tokens,
        )
        if row.selected and temperature != 0.0:
            raise ValueError(
                "speculative selection requires greedy temperature"
            )
        if row.selected and (
            remaining_output_tokens == 0
            or row.max_proposal_tokens
            > remaining_output_tokens
        ):
            raise ValueError(
                "speculative selection output budget is stale"
            )
        if row.selected:
            selected.append(seq)
    return tuple(selected)
