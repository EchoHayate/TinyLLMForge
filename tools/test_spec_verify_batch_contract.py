from __future__ import annotations

import os
import sys
import types

import pytest


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

tinyvllm_package = types.ModuleType("tinyvllm")
tinyvllm_package.__path__ = [
    os.path.join(_REPO_ROOT, "tinyvllm")
]
speculative_package = types.ModuleType("tinyvllm.speculative")
speculative_package.__path__ = [
    os.path.join(_REPO_ROOT, "tinyvllm", "speculative")
]
sys.modules.setdefault("tinyvllm", tinyvllm_package)
sys.modules.setdefault("tinyvllm.speculative", speculative_package)

from tinyvllm.speculative.verifier import (
    SpecVerifyBatchMetadata,
    SpecVerifyBatchResultRow,
    SpecVerifyBatchRowMetadata,
    split_spec_verify_batch_target_tokens,
)


def _row(
    sequence_id,
    batch_index,
    query_offset,
    *,
    query_len=2,
    block_table=(5, 6),
):
    base = batch_index * 10
    return SpecVerifyBatchRowMetadata(
        sequence_id=sequence_id,
        batch_index=batch_index,
        query_offset=query_offset,
        query_len=query_len,
        input_tokens=tuple(
            base + 10 + offset for offset in range(query_len)
        ),
        positions=tuple(
            base + 5 + offset for offset in range(query_len)
        ),
        logical_slots=tuple(
            base + 4 + offset for offset in range(query_len)
        ),
        physical_slots=tuple(
            base + 20 + offset for offset in range(query_len)
        ),
        context_len=base + 6,
        block_table=block_table,
    )


def _metadata():
    return SpecVerifyBatchMetadata(
        rows=(
            _row(8, 0, 0, block_table=(5, 6)),
            _row(4, 1, 2, block_table=(10, 11, 12)),
        ),
        query_len=2,
        total_query_tokens=4,
        block_table_width=3,
    )


def test_splits_flat_target_tokens_in_metadata_order():
    rows = split_spec_verify_batch_target_tokens(
        _metadata(),
        (101, 102, 201, 202),
    )

    assert rows == (
        SpecVerifyBatchResultRow(
            sequence_id=8,
            target_tokens=(101, 102),
        ),
        SpecVerifyBatchResultRow(
            sequence_id=4,
            target_tokens=(201, 202),
        ),
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {
                "rows": [_row(8, 0, 0)],
                "query_len": 2,
                "total_query_tokens": 2,
                "block_table_width": 2,
            },
            "rows",
        ),
        (
            {
                "rows": (
                    _row(8, 0, 0),
                    _row(8, 1, 2),
                ),
                "query_len": 2,
                "total_query_tokens": 4,
                "block_table_width": 2,
            },
            "unique",
        ),
        (
            {
                "rows": (
                    _row(8, 0, 0),
                    _row(4, 1, 3),
                ),
                "query_len": 2,
                "total_query_tokens": 4,
                "block_table_width": 2,
            },
            "offset",
        ),
        (
            {
                "rows": (
                    _row(8, 0, 0),
                    _row(4, 1, 2, query_len=1),
                ),
                "query_len": 2,
                "total_query_tokens": 3,
                "block_table_width": 2,
            },
            "query",
        ),
        (
            {
                "rows": (_row(8, 0, 0),),
                "query_len": 2,
                "total_query_tokens": 3,
                "block_table_width": 2,
            },
            "total",
        ),
        (
            {
                "rows": (
                    _row(
                        8,
                        0,
                        0,
                        block_table=(5, 6, 7),
                    ),
                ),
                "query_len": 2,
                "total_query_tokens": 2,
                "block_table_width": 2,
            },
            "width",
        ),
    ],
)
def test_batch_metadata_rejects_invalid_layout(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SpecVerifyBatchMetadata(**kwargs)


def test_row_metadata_rejects_inconsistent_tuple_lengths():
    with pytest.raises(ValueError, match="positions"):
        SpecVerifyBatchRowMetadata(
            sequence_id=1,
            batch_index=0,
            query_offset=0,
            query_len=2,
            input_tokens=(10, 11),
            positions=(5,),
            logical_slots=(4, 5),
            physical_slots=(20, 21),
            context_len=6,
            block_table=(5, 6),
        )


@pytest.mark.parametrize(
    "tokens,match",
    [
        ([101, 102, 201, 202], "tuple"),
        ((101, 102, 201), "count"),
        ((101, 102, 201, True), "integer"),
    ],
)
def test_splitter_rejects_invalid_flat_targets(tokens, match):
    with pytest.raises(ValueError, match=match):
        split_spec_verify_batch_target_tokens(
            _metadata(),
            tokens,
        )
