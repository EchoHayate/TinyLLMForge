from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import os
import sys
import types

import pytest


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

tinyvllm_package = types.ModuleType("tinyvllm")
tinyvllm_package.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm")]
engine_package = types.ModuleType("tinyvllm.engine")
engine_package.__path__ = [
    os.path.join(_REPO_ROOT, "tinyvllm", "engine")
]
sys.modules.setdefault("tinyvllm", tinyvllm_package)
sys.modules.setdefault("tinyvllm.engine", engine_package)

from tinyvllm.engine.speculative_selection import (
    SpeculativeSelectionConfig,
    build_speculative_selection_record,
    validate_speculative_selection_record,
)


class _Sequence:
    def __init__(
        self,
        sequence_id,
        *,
        num_tokens=8,
        completion_tokens=1,
        max_tokens=8,
        step_is_decode=False,
        step_do_sample=True,
        temperature=0.0,
    ):
        self.seq_id = sequence_id
        self.num_tokens = num_tokens
        self.num_completion_tokens = completion_tokens
        self.max_tokens = max_tokens
        self.step_is_decode = step_is_decode
        self.step_do_sample = step_do_sample
        self.temperature = temperature


def _record(
    seqs,
    *,
    config=None,
    is_prefill=False,
    do_sample=True,
    batch_kind=None,
    generation=3,
):
    return build_speculative_selection_record(
        seqs=tuple(seqs),
        is_prefill=is_prefill,
        do_sample=do_sample,
        batch_kind=batch_kind,
        policy_branch="fixture",
        schedule_generation=generation,
        config=(
            SpeculativeSelectionConfig(
                enabled=True,
                max_proposal_tokens=4,
            )
            if config is None
            else config
        ),
    )


@pytest.mark.parametrize(
    "enabled,max_proposal_tokens",
    [
        (False, 1),
        (True, 0),
        (True, 1),
        (True, True),
        (1, 4),
    ],
)
def test_selection_config_rejects_invalid_values(
    enabled,
    max_proposal_tokens,
):
    with pytest.raises(ValueError):
        SpeculativeSelectionConfig(
            enabled=enabled,
            max_proposal_tokens=max_proposal_tokens,
        )


def test_disabled_config_requires_zero_and_is_immutable():
    config = SpeculativeSelectionConfig(
        enabled=False,
        max_proposal_tokens=0,
    )

    with pytest.raises(FrozenInstanceError):
        config.enabled = True


def test_ordinary_decode_selects_rows_and_caps_by_output_budget():
    seqs = (
        _Sequence(8, completion_tokens=1, max_tokens=8),
        _Sequence(4, completion_tokens=2, max_tokens=5),
    )

    record = _record(seqs)

    assert record.scheduled_sequence_ids == (8, 4)
    assert tuple(row.sequence_id for row in record.rows) == (8, 4)
    assert tuple(row.max_proposal_tokens for row in record.rows) == (
        4,
        3,
    )
    assert tuple(row.remaining_output_tokens for row in record.rows) == (
        7,
        3,
    )
    assert record.selected_sequence_ids == (8, 4)
    assert record.selected_rows == record.rows
    assert tuple(
        row.temperature_snapshot for row in record.rows
    ) == (0.0, 0.0)


def test_ordinary_prefill_suppresses_every_row():
    record = _record(
        (_Sequence(1), _Sequence(2)),
        is_prefill=True,
    )

    assert record.selected_rows == ()
    assert {
        row.suppression_reason for row in record.rows
    } == {"prefill"}


def test_disabled_config_has_stable_precedence():
    record = _record(
        (_Sequence(1, completion_tokens=8, max_tokens=8),),
        config=SpeculativeSelectionConfig(
            enabled=False,
            max_proposal_tokens=0,
        ),
        is_prefill=True,
        do_sample=False,
    )

    assert record.rows[0].suppression_reason == "disabled"


def test_mixed_batch_selects_only_sampling_decode_rows():
    record = _record(
        (
            _Sequence(1, step_is_decode=False),
            _Sequence(2, step_is_decode=True),
            _Sequence(
                3,
                step_is_decode=True,
                step_do_sample=False,
            ),
        ),
        is_prefill=True,
        batch_kind="mixed",
    )

    assert tuple(row.selected for row in record.rows) == (
        False,
        True,
        False,
    )
    assert tuple(
        row.suppression_reason for row in record.rows
    ) == ("prefill", None, "not_sampling")
    assert record.selected_sequence_ids == (2,)


def test_non_greedy_decode_is_suppressed_before_runtime():
    record = _record(
        (
            _Sequence(1, temperature=0.0),
            _Sequence(2, temperature=0.7),
        )
    )

    assert record.selected_sequence_ids == (1,)
    assert tuple(row.selected for row in record.rows) == (
        True,
        False,
    )
    assert tuple(
        row.suppression_reason for row in record.rows
    ) == (None, "non_greedy")
    assert tuple(
        row.max_proposal_tokens for row in record.rows
    ) == (4, 0)
    assert tuple(
        row.temperature_snapshot for row in record.rows
    ) == (0.0, 0.7)


def test_mixed_batch_selects_only_greedy_sampling_decode_rows():
    record = _record(
        (
            _Sequence(
                1,
                step_is_decode=True,
                temperature=0.0,
            ),
            _Sequence(
                2,
                step_is_decode=True,
                temperature=0.5,
            ),
        ),
        is_prefill=True,
        batch_kind="mixed",
    )

    assert record.selected_sequence_ids == (1,)
    assert tuple(
        row.suppression_reason for row in record.rows
    ) == (None, "non_greedy")


@pytest.mark.parametrize(
    "temperature",
    [
        True,
        "0",
        float("nan"),
        float("inf"),
    ],
)
def test_builder_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        _record((_Sequence(1, temperature=temperature),))


def test_suppresses_non_sampling_decode():
    record = _record(
        (_Sequence(1),),
        do_sample=False,
    )

    assert record.rows[0].selected is False
    assert record.rows[0].max_proposal_tokens == 0
    assert record.rows[0].suppression_reason == "not_sampling"


def test_greedy_decode_suppresses_single_remaining_output_token():
    record = _record(
        (
            _Sequence(
                1,
                completion_tokens=3,
                max_tokens=4,
            ),
        )
    )

    assert record.rows[0].selected is False
    assert record.rows[0].max_proposal_tokens == 0
    assert (
        record.rows[0].suppression_reason
        == "insufficient_output_budget"
    )


def test_builder_rejects_duplicate_or_boolean_sequence_ids():
    with pytest.raises(ValueError, match="unique"):
        _record((_Sequence(1), _Sequence(1)))
    with pytest.raises(ValueError, match="sequence"):
        _record((_Sequence(True),))


def test_record_and_rows_are_immutable():
    record = _record((_Sequence(1),))

    with pytest.raises(FrozenInstanceError):
        record.schedule_generation = 4
    with pytest.raises(FrozenInstanceError):
        record.rows[0].selected = False


def test_validation_returns_selected_sequences_in_record_order():
    seqs = (
        _Sequence(8),
        _Sequence(4, completion_tokens=7, max_tokens=8),
        _Sequence(2),
    )
    record = _record(seqs, generation=9)

    selected = validate_speculative_selection_record(
        record,
        seqs,
        expected_schedule_generation=9,
    )

    assert selected == (seqs[0], seqs[2])


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("generation", "generation"),
        ("reorder", "order"),
        ("tokens", "token"),
        ("completion", "completion"),
        ("budget", "budget"),
    ],
)
def test_validation_rejects_stale_or_mismatched_record(
    mutation,
    match,
):
    seqs = (_Sequence(1), _Sequence(2))
    record = _record(seqs, generation=7)
    expected_generation = 7
    current = seqs
    if mutation == "generation":
        expected_generation = 8
    elif mutation == "reorder":
        current = (seqs[1], seqs[0])
    elif mutation == "tokens":
        seqs[0].num_tokens += 1
    elif mutation == "completion":
        seqs[0].num_completion_tokens += 1
    elif mutation == "budget":
        seqs[0].max_tokens = 2

    with pytest.raises(ValueError, match=match):
        validate_speculative_selection_record(
            record,
            current,
            expected_schedule_generation=expected_generation,
        )


def test_validation_rejects_stale_temperature():
    seqs = (_Sequence(1, temperature=0.0),)
    record = _record(seqs, generation=7)
    seqs[0].temperature = 0.5

    with pytest.raises(ValueError, match="temperature"):
        validate_speculative_selection_record(
            record,
            seqs,
            expected_schedule_generation=7,
        )


def test_validation_rejects_selected_non_greedy_record():
    seqs = (_Sequence(1, temperature=0.5),)
    record = _record(seqs, generation=7)
    row = replace(
        record.rows[0],
        selected=True,
        max_proposal_tokens=4,
        suppression_reason=None,
    )
    malformed = replace(record, rows=(row,))

    with pytest.raises(ValueError, match="greedy temperature"):
        validate_speculative_selection_record(
            malformed,
            seqs,
            expected_schedule_generation=7,
        )
