from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tinyvllm"
    / "engine"
    / "spec_verify_trace.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "tinyvllm.engine.spec_verify_trace",
    _MODULE_PATH,
)
spec_verify_trace = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = spec_verify_trace
_SPEC.loader.exec_module(spec_verify_trace)

TRACE_SCHEMA = spec_verify_trace.TRACE_SCHEMA
compact_topk_logits = spec_verify_trace.compact_topk_logits
logical_block_coverage = spec_verify_trace.logical_block_coverage


def test_compact_topk_logits_breaks_ties_by_token_id():
    rows = compact_topk_logits(
        torch.tensor([[
            1.0,
            3.0,
            3.0,
            2.0,
            0.5,
            -1.0,
        ]]),
        top_k=5,
    )

    assert rows == (
        {
            "top_tokens": (1, 2, 3, 0, 4),
            "top_logits": (3.0, 3.0, 2.0, 1.0, 0.5),
            "top1_margin": 0.0,
            "argmax_token": 1,
        },
    )


def test_compact_topk_logits_rejects_less_than_five():
    with pytest.raises(
        ValueError,
        match="trace top_k must be at least five",
    ):
        compact_topk_logits(
            torch.zeros((1, 8)),
            top_k=4,
        )


def test_logical_block_coverage_uses_context_length_not_residency():
    assert logical_block_coverage(513, 256) == (
        (0, 0, 256),
        (1, 256, 512),
        (2, 512, 513),
    )


def test_trace_schema_is_frozen():
    assert TRACE_SCHEMA == (
        "qwen35.native-mtp-tp4-32k-paired-verify-trace.v1"
    )


def _context():
    return spec_verify_trace.TargetForwardTraceContext(
        policy="baseline",
        batch_size=1,
        engine_step=3,
    )


def _logits():
    return torch.tensor([[
        1.0,
        2.0,
        3.0,
        0.0,
        -1.0,
    ]])


def test_recorder_is_default_off_and_drain_is_tensor_free():
    recorder = spec_verify_trace.SpecVerifyTraceRecorder(
        rank=0,
        block_size=256,
    )
    recorder.set_context(_context())
    recorder.record_rows(
        stage="ordinary_decode",
        execution_mode="decode",
        sequence_ids=(7,),
        query_offset=0,
        query_len=1,
        input_tokens=(15,),
        positions=(32770,),
        prediction_indices=(3,),
        logical_block_identities=(
            tuple((index, 1) for index in range(129)),
        ),
        logits=_logits(),
    )
    assert recorder.drain() == ()

    recorder.enable(True)
    recorder.set_context(_context())
    recorder.record_rows(
        stage="ordinary_decode",
        execution_mode="decode",
        sequence_ids=(7,),
        query_offset=0,
        query_len=1,
        input_tokens=(15,),
        positions=(32770,),
        prediction_indices=(3,),
        logical_block_identities=(
            tuple((index, 1) for index in range(129)),
        ),
        logits=_logits(),
    )
    rows = recorder.drain()

    assert len(rows) == 1
    assert set(rows[0]) == {
        "schema",
        "policy",
        "batch_size",
        "engine_step",
        "target_forward_ordinal",
        "stage",
        "execution_mode",
        "sequence_id",
        "query_offset",
        "query_len",
        "row_index",
        "prediction_index",
        "input_token_id",
        "position",
        "context_length",
        "logical_block_identities",
        "logical_block_coverage",
        "top_tokens",
        "top_logits",
        "top1_margin",
        "argmax_token",
    }
    assert rows[0]["context_length"] == 32771
    assert rows[0]["prediction_index"] == 3
    assert not any(
        isinstance(value, torch.Tensor)
        for value in rows[0].values()
    )
    assert recorder.drain() == ()


def test_worker_rank_never_copies_or_stores_logits():
    recorder = spec_verify_trace.SpecVerifyTraceRecorder(
        rank=1,
        block_size=256,
    )
    recorder.enable(True)
    recorder.set_context(_context())
    recorder.record_rows(
        stage="verify_tail",
        execution_mode="spec_verify",
        sequence_ids=(7,),
        query_offset=0,
        query_len=1,
        input_tokens=(15,),
        positions=(5,),
        prediction_indices=(3,),
        logical_block_identities=(((0, 1),),),
        logits=_logits(),
    )
    assert recorder.drain() == ()


def test_disable_clears_undrained_rows_and_context():
    recorder = spec_verify_trace.SpecVerifyTraceRecorder(
        rank=0,
        block_size=256,
    )
    recorder.enable(True)
    recorder.set_context(_context())
    recorder.record_rows(
        stage="ordinary_decode",
        execution_mode="decode",
        sequence_ids=(7,),
        query_offset=0,
        query_len=1,
        input_tokens=(15,),
        positions=(5,),
        prediction_indices=(0,),
        logical_block_identities=(((0, 1),),),
        logits=_logits(),
    )
    recorder.enable(False)
    assert recorder.drain() == ()


def test_recorder_rejects_incomplete_block_identity_coverage():
    recorder = spec_verify_trace.SpecVerifyTraceRecorder(
        rank=0,
        block_size=256,
    )
    recorder.enable(True)
    recorder.set_context(_context())
    with pytest.raises(
        ValueError,
        match="trace block identity coverage is incomplete",
    ):
        recorder.record_rows(
            stage="ordinary_decode",
            execution_mode="decode",
            sequence_ids=(7,),
            query_offset=0,
            query_len=1,
            input_tokens=(15,),
            positions=(256,),
            prediction_indices=(0,),
            logical_block_identities=(((0, 1),),),
            logits=_logits(),
        )
