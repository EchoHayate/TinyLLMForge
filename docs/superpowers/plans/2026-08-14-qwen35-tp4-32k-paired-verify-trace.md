# Qwen3.5 TP4 32K Paired Verify Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task in the
> current session. Do not use subagents.

**Goal:** Add a default-disabled, source-bound paired target-forward trace that
locates the first aligned baseline/native Qwen3.5 TP4/32K logit, KV-lineage, or
side-state divergence without changing verifier, sampling, KV transaction, or
authority behavior.

**Architecture:** Put tensor compaction and immutable target-forward trace
contracts in a focused engine helper, and put Qwen3.5 checkpoint fingerprinting
in a separate focused helper. Wire both helpers into the existing
`ModelRunner` and `Qwen35SpeculativeStateOwner` behind explicit default-off
lifecycle methods. Extend only the 32K worker overlay to enrich, pair,
validate, and serialize trace rows; the frozen 16K worker and ordinary 32K
authority path remain behaviorally unchanged.

**Tech Stack:** Python 3, dataclasses, hashlib/SHA-256, JSON, PyTorch,
pytest, TinyLLMForge `ModelRunner`, Qwen3.5 speculative side-state owner, and
the existing 32K source-manifest contract.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, push, switch branches/worktrees, stash, reset, or
  clean.
- Do not use subagents.
- Do not terminate unrelated GPU processes.
- Do not run a remote workload in this plan.
- Exact greedy parity remains mandatory.
- Keep `MAX_PROPOSAL_TOKENS=4`.
- Keep target KV at 68 GPU blocks, 640 logical blocks, and block size 256.
- Keep proposal KV GPU-resident.
- Diagnostics are disabled by default and must issue no extra model forward.
- Do not change verifier token selection, fallback indexing, accepted-prefix
  calculation, target-KV transaction semantics, side-state selection
  semantics, or offload counters.
- Do not claim 32K authority, performance improvement, KV8/KV4, a second
  learned structure, production readiness, or Phase 1 completion.
- The plan produces local diagnostic infrastructure only. A later remote run
  requires separate explicit authorization.
- Every task ends with local verification instead of a git commit.

## File Structure

- Create `tinyvllm/engine/spec_verify_trace.py`
  - Immutable target-forward row contract, deterministic top-k compaction,
    logical block coverage, and a default-disabled rank-zero recorder.
- Create `tinyvllm/engine/qwen35_speculative_trace.py`
  - Exact byte-level candidate fingerprinting and a default-disabled
    Qwen3.5 checkpoint-lineage recorder.
- Modify `tinyvllm/engine/model_runner.py`
  - Own the target-forward recorder and call it at ordinary decode,
    first-target decode, and verify-tail boundaries.
- Modify `tinyvllm/engine/qwen35_speculative_state.py`
  - Own the side-state recorder and observe checkpoint creation/selection
    without mutating candidates.
- Modify `tools/qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py`
  - Explicit diagnostic lifecycle, semantic pairing, lineage assembly,
    first-divergence selection, and JSON artifact writing.
- Modify `tools/qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py`
  - Add the two new production helpers to the source-bound inventory without
    changing the authority schema or validator.
- Create `tools/test_spec_verify_trace.py`
  - Focused unit tests for the generic trace helper.
- Modify `tools/test_model_runner_spec_verify.py`
  - ModelRunner integration tests.
- Modify `tools/test_qwen35_speculative_state.py`
  - Side-state fingerprint and non-mutation tests.
- Modify
  `tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py`
  - Worker pairing, artifact, source-binding, and default-off tests.

---

### Task 1: Add the Immutable Target-Forward Trace Core

**Files:**
- Create: `tinyvllm/engine/spec_verify_trace.py`
- Create: `tools/test_spec_verify_trace.py`

**Interfaces:**
- Produces:
  - `TRACE_SCHEMA: str`
  - `TargetForwardTraceContext`
  - `TargetForwardTraceRow`
  - `SpecVerifyTraceRecorder`
  - `compact_topk_logits(logits, top_k=5)`
  - `logical_block_coverage(context_length, block_size)`
- Consumed later by `ModelRunner` and the 32K worker.

- [ ] **Step 1: Write RED tests for deterministic compaction and coverage**

Create `tools/test_spec_verify_trace.py` with imports through the ordinary
package path and these tests:

```python
from __future__ import annotations

import pytest
import torch

from tinyvllm.engine.spec_verify_trace import (
    TRACE_SCHEMA,
    SpecVerifyTraceRecorder,
    TargetForwardTraceContext,
    compact_topk_logits,
    logical_block_coverage,
)


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
```

- [ ] **Step 2: Run the focused tests RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_spec_verify_trace.py::test_compact_topk_logits_breaks_ties_by_token_id \
  tools/test_spec_verify_trace.py::test_compact_topk_logits_rejects_less_than_five \
  tools/test_spec_verify_trace.py::test_logical_block_coverage_uses_context_length_not_residency \
  tools/test_spec_verify_trace.py::test_trace_schema_is_frozen
```

Expected: import failure because `tinyvllm.engine.spec_verify_trace` does not
exist.

- [x] **Step 3: Implement deterministic compaction and coverage**

Create `tinyvllm/engine/spec_verify_trace.py` with:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


TRACE_SCHEMA = (
    "qwen35.native-mtp-tp4-32k-paired-verify-trace.v1"
)


def _positive_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a non-negative integer"
        )
    return value


def compact_topk_logits(
    logits: torch.Tensor,
    *,
    top_k: int = 5,
) -> tuple[dict, ...]:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise ValueError("trace logits must be a rank-two tensor")
    top_k = _positive_integer(top_k, "trace top_k")
    if top_k < 5:
        raise ValueError(
            "trace top_k must be at least five"
        )
    if top_k > logits.shape[1]:
        raise ValueError("trace top_k exceeds vocabulary size")
    cpu_rows = logits.detach().float().cpu().tolist()
    compact = []
    for values in cpu_rows:
        ranked = sorted(
            enumerate(values),
            key=lambda item: (-float(item[1]), item[0]),
        )[:top_k]
        top_tokens = tuple(
            int(token_id) for token_id, _ in ranked
        )
        top_logits = tuple(
            float(value) for _, value in ranked
        )
        compact.append({
            "top_tokens": top_tokens,
            "top_logits": top_logits,
            "top1_margin": (
                float(top_logits[0] - top_logits[1])
                if len(top_logits) > 1
                else None
            ),
            "argmax_token": top_tokens[0],
        })
    return tuple(compact)


def logical_block_coverage(
    context_length: int,
    block_size: int,
) -> tuple[tuple[int, int, int], ...]:
    context_length = _positive_integer(
        context_length,
        "trace context_length",
    )
    block_size = _positive_integer(
        block_size,
        "trace block_size",
    )
    return tuple(
        (
            block_ordinal,
            block_ordinal * block_size,
            min(
                context_length,
                (block_ordinal + 1) * block_size,
            ),
        )
        for block_ordinal in range(
            (context_length + block_size - 1) // block_size
        )
    )
```

- [ ] **Step 4: Add RED tests for recorder lifecycle and exact row fields**

Append:

```python
def _context():
    return TargetForwardTraceContext(
        policy="baseline",
        batch_size=1,
        engine_step=3,
    )


def test_recorder_is_default_off_and_drain_is_tensor_free():
    recorder = SpecVerifyTraceRecorder(
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
        logits=torch.tensor([[
            1.0,
            2.0,
            3.0,
            0.0,
            -1.0,
        ]]),
    )
    assert recorder.drain() == ()

    recorder.enable(True)
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
        logits=torch.tensor([[
            1.0,
            2.0,
            3.0,
            0.0,
            -1.0,
        ]]),
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
    assert recorder.drain() == ()


def test_worker_rank_never_copies_or_stores_logits():
    recorder = SpecVerifyTraceRecorder(
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
        positions=(32770,),
        prediction_indices=(3,),
        logical_block_identities=(((0, 1),),),
        logits=torch.tensor([[
            1.0,
            2.0,
            3.0,
            0.0,
            -1.0,
        ]]),
    )
    assert recorder.drain() == ()


def test_disable_clears_undrained_rows_and_context():
    recorder = SpecVerifyTraceRecorder(
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
        logits=torch.tensor([[
            1.0,
            2.0,
            3.0,
            0.0,
            -1.0,
        ]]),
    )
    recorder.enable(False)
    assert recorder.drain() == ()
```

- [x] **Step 5: Implement immutable context, row, and recorder**

Append to the helper:

```python
@dataclass(frozen=True)
class TargetForwardTraceContext:
    policy: str
    batch_size: int
    engine_step: int

    def __post_init__(self):
        if self.policy not in ("baseline", "native_mtp"):
            raise ValueError("trace policy is invalid")
        _positive_integer(self.batch_size, "trace batch_size")
        _non_negative_integer(
            self.engine_step,
            "trace engine_step",
        )


@dataclass(frozen=True)
class TargetForwardTraceRow:
    schema: str
    policy: str
    batch_size: int
    engine_step: int
    target_forward_ordinal: int
    stage: str
    execution_mode: str
    sequence_id: int
    query_offset: int
    query_len: int
    row_index: int
    prediction_index: int
    input_token_id: int
    position: int
    context_length: int
    logical_block_identities: tuple[tuple[int, int], ...]
    logical_block_coverage: tuple[tuple[int, int, int], ...]
    top_tokens: tuple[int, ...]
    top_logits: tuple[float, ...]
    top1_margin: float | None
    argmax_token: int

    def as_dict(self) -> dict:
        return asdict(self)


class SpecVerifyTraceRecorder:
    def __init__(self, *, rank: int, block_size: int):
        self.rank = _non_negative_integer(rank, "trace rank")
        self.block_size = _positive_integer(
            block_size,
            "trace block_size",
        )
        self._enabled = False
        self._context = None
        self._rows = []
        self._target_forward_ordinal = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self, enabled: bool) -> dict:
        if not isinstance(enabled, bool):
            raise ValueError("trace enabled must be a boolean")
        self._enabled = enabled
        self._rows.clear()
        self._context = None
        self._target_forward_ordinal = 0
        return {"rank": self.rank, "enabled": enabled}

    def set_context(
        self,
        context: TargetForwardTraceContext,
    ) -> None:
        if not self._enabled:
            raise RuntimeError("trace recording is disabled")
        if type(context) is not TargetForwardTraceContext:
            raise ValueError("trace context type mismatch")
        self._context = context

    def record_rows(
        self,
        *,
        stage: str,
        execution_mode: str,
        sequence_ids: tuple[int, ...],
        query_offset: int,
        query_len: int,
        input_tokens: tuple[int, ...],
        positions: tuple[int, ...],
        prediction_indices: tuple[int, ...],
        logical_block_identities: tuple[
            tuple[tuple[int, int], ...],
            ...,
        ],
        logits: torch.Tensor,
    ) -> None:
        if not self._enabled or self.rank != 0:
            return
        if self._context is None:
            raise RuntimeError("trace context is missing")
        if stage not in (
            "ordinary_decode",
            "first_target",
            "verify_tail",
        ):
            raise ValueError("trace stage is invalid")
        query_len = _positive_integer(
            query_len,
            "trace query_len",
        )
        row_count = len(input_tokens)
        if (
            row_count != len(positions)
            or row_count != len(prediction_indices)
            or row_count != logits.shape[0]
            or len(sequence_ids)
            != len(logical_block_identities)
            or row_count != len(sequence_ids) * query_len
        ):
            raise ValueError("trace row inventory mismatch")
        compact_rows = compact_topk_logits(logits, top_k=5)
        forward_ordinal = self._target_forward_ordinal
        self._target_forward_ordinal += 1
        for flat_index, compact in enumerate(compact_rows):
            sequence_index = flat_index // query_len
            position = int(positions[flat_index])
            identities = logical_block_identities[
                sequence_index
            ]
            coverage = logical_block_coverage(
                position + 1,
                self.block_size,
            )
            if len(identities) < len(coverage):
                raise ValueError(
                    "trace block identity coverage is incomplete"
                )
            if any(
                not isinstance(identity, tuple)
                or len(identity) != 2
                or isinstance(identity[0], bool)
                or not isinstance(identity[0], int)
                or identity[0] < 0
                or isinstance(identity[1], bool)
                or not isinstance(identity[1], int)
                or identity[1] < 0
                for identity in identities
            ):
                raise ValueError(
                    "trace block identity is invalid"
                )
            self._rows.append(TargetForwardTraceRow(
                schema=TRACE_SCHEMA,
                policy=self._context.policy,
                batch_size=self._context.batch_size,
                engine_step=self._context.engine_step,
                target_forward_ordinal=forward_ordinal,
                stage=stage,
                execution_mode=execution_mode,
                sequence_id=int(
                    sequence_ids[sequence_index]
                ),
                query_offset=(
                    query_offset
                    + sequence_index * query_len
                ),
                query_len=query_len,
                row_index=flat_index,
                prediction_index=int(
                    prediction_indices[flat_index]
                ),
                input_token_id=int(input_tokens[flat_index]),
                position=position,
                context_length=position + 1,
                logical_block_identities=identities,
                logical_block_coverage=coverage,
                **compact,
            ))

    def drain(self) -> tuple[dict, ...]:
        rows = tuple(row.as_dict() for row in self._rows)
        self._rows.clear()
        return rows
```

- [x] **Step 6: Run the complete helper test GREEN**

Run:

```bash
python3 -m pytest -q tools/test_spec_verify_trace.py
```

Expected: all new helper tests pass.

- [x] **Step 7: Run syntax and diff checks**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/spec_verify_trace.py \
  tools/test_spec_verify_trace.py

git diff --check -- \
  tinyvllm/engine/spec_verify_trace.py \
  tools/test_spec_verify_trace.py
```

Expected: both commands succeed with no output from `git diff --check`.

### Task 2: Wire Target Trace Capture into ModelRunner

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**
- Consumes `SpecVerifyTraceRecorder` and `TargetForwardTraceContext`.
- Produces ModelRunner methods:
  - `enable_spec_verify_trace_recording(enabled: bool) -> dict`
  - `set_spec_verify_trace_context(policy, batch_size, engine_step) -> dict`
  - `drain_spec_verify_trace_rows() -> tuple[dict, ...]`
- Captures no rows unless explicitly enabled.

- [ ] **Step 1: Write RED lifecycle tests**

Add a helper to `tools/test_model_runner_spec_verify.py`:

```python
def _trace_ready_runner(rank=0):
    runner = make_runner()
    runner.rank = rank
    runner.block_size = 256
    runner._spec_verify_trace = SpecVerifyTraceRecorder(
        rank=rank,
        block_size=256,
    )
    return runner
```

Add tests:

```python
def test_spec_verify_trace_is_default_off():
    runner = _trace_ready_runner()
    assert runner.drain_spec_verify_trace_rows() == ()


def test_spec_verify_trace_lifecycle_is_explicit():
    runner = _trace_ready_runner()

    assert runner.enable_spec_verify_trace_recording(True) == {
        "rank": 0,
        "enabled": True,
    }
    assert runner.set_spec_verify_trace_context(
        "native_mtp",
        1,
        4,
    ) == {
        "rank": 0,
        "policy": "native_mtp",
        "batch_size": 1,
        "engine_step": 4,
    }
    assert runner.enable_spec_verify_trace_recording(False) == {
        "rank": 0,
        "enabled": False,
    }
```

Import the helper types at the test module's existing ModelRunner import
boundary.

- [ ] **Step 2: Run lifecycle tests RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_model_runner_spec_verify.py::test_spec_verify_trace_is_default_off \
  tools/test_model_runner_spec_verify.py::test_spec_verify_trace_lifecycle_is_explicit
```

Expected: missing ModelRunner methods.

- [x] **Step 3: Initialize and expose the recorder**

In `ModelRunner.__init__`, immediately after `_last_step_logits_cpu`, add:

```python
self._spec_verify_trace = SpecVerifyTraceRecorder(
    rank=rank,
    block_size=self.block_size,
)
```

Add imports:

```python
from tinyvllm.engine.spec_verify_trace import (
    SpecVerifyTraceRecorder,
    TargetForwardTraceContext,
)
```

Add methods beside `enable_step_logits_recording()`:

```python
def enable_spec_verify_trace_recording(
    self,
    enabled: bool,
) -> dict:
    return self._spec_verify_trace.enable(enabled)


def set_spec_verify_trace_context(
    self,
    policy: str,
    batch_size: int,
    engine_step: int,
) -> dict:
    context = TargetForwardTraceContext(
        policy=policy,
        batch_size=batch_size,
        engine_step=engine_step,
    )
    self._spec_verify_trace.set_context(context)
    return {
        "rank": self.rank,
        "policy": policy,
        "batch_size": batch_size,
        "engine_step": engine_step,
    }


def drain_spec_verify_trace_rows(
    self,
) -> tuple[dict, ...]:
    return self._spec_verify_trace.drain()
```

- [x] **Step 4: Add a private policy-local identity resolver**

Write RED tests that construct a fake `kv_offload` with the same list-backed
shape as `KVOffloadMVP0.bound_generations`, for example
`bound_generations=[None, None, None, 7, 8, None]`, and require:

```python
assert runner._trace_block_identities((3, 4)) == (
    (3, 7),
    (4, 8),
)
```

Also require a missing generation to raise:

```python
with pytest.raises(
    RuntimeError,
    match="trace block generation is missing",
):
    runner._trace_block_identities((3, 5))
```

Implement:

```python
def _trace_block_identities(
    self,
    block_table,
) -> tuple[tuple[int, int], ...]:
    if not self._spec_verify_trace.enabled:
        return ()
    if self.kv_offload is None:
        raise RuntimeError(
            "trace block identities require kv_offload_mvp0"
        )
    identities = []
    for block_id in block_table:
        block_id = int(block_id)
        if (
            block_id < 0
            or block_id
            >= len(self.kv_offload.bound_generations)
        ):
            raise RuntimeError(
                "trace logical block id is out of range"
            )
        generation = (
            self.kv_offload.bound_generations[block_id]
        )
        if generation is None:
            raise RuntimeError(
                "trace block generation is missing"
            )
        identities.append((block_id, int(generation)))
    return tuple(identities)
```

- [ ] **Step 5: Write RED tests for both first-target entry points**

Use the existing first-target fake runner patterns for both
`_run_spec_first_target_batch()` and the native runtime's actual fused
`run_spec_first_target_and_proposal_batch()` path. Enable trace and set
context, then require one row from each entry point with:

```python
assert rows[0]["stage"] == "first_target"
assert rows[0]["execution_mode"] == "decode"
assert rows[0]["sequence_id"] == 7
assert rows[0]["prediction_index"] == 0
assert rows[0]["input_token_id"] == 11
assert rows[0]["position"] == 32767
assert rows[0]["context_length"] == 32768
assert rows[0]["logical_block_identities"][-1] == (127, 1)
```

The direct test must assert the returned `FirstTargetResult.target_token` is
unchanged from the pre-trace expectation. The fused test must assert both the
returned first target token and proposal payload are unchanged.

- [x] **Step 6: Record first-target rows through one shared helper**

Add a private helper so the direct and fused paths cannot drift:

```python
def _record_spec_first_target_trace(
    self,
    *,
    seqs,
    input_ids,
    positions,
    logits,
) -> None:
    if not self._spec_verify_trace.enabled:
        return
    self._spec_verify_trace.record_rows(
        stage="first_target",
        execution_mode="decode",
        sequence_ids=tuple(
            int(seq.seq_id) for seq in seqs
        ),
        query_offset=0,
        query_len=1,
        input_tokens=tuple(
            int(value)
            for value in input_ids.detach().cpu().tolist()
        ),
        positions=tuple(
            int(value)
            for value in positions.detach().cpu().tolist()
        ),
        prediction_indices=tuple(
            int(seq.num_completion_tokens)
            for seq in seqs
        ),
        logical_block_identities=tuple(
            self._trace_block_identities(seq.block_table)
            for seq in seqs
        ),
        logits=logits,
    )
```

Call this helper in both `_run_spec_first_target_batch()` and
`run_spec_first_target_and_proposal_batch()` immediately after `logits` is
resolved and before either path performs rank-zero return handling or target
token selection. This covers the native TP4 runtime's fused first-target path
without issuing another forward.

Use `num_completion_tokens`, which is already the stable pre-commit output
index exposed by `Sequence`.

- [ ] **Step 7: Write a RED verify-tail capture test**

Extend the existing `_run_spec_verify_batch()` fake setup with two metadata
rows and `query_len=3`. Require:

```python
assert [row["stage"] for row in rows] == [
    "verify_tail",
] * 6
assert [row["prediction_index"] for row in rows[:3]] == [
    1,
    2,
    3,
]
assert [row["input_token_id"] for row in rows[:3]] == [
    15,
    15,
    2658,
]
assert [row["position"] for row in rows[:3]] == [
    32768,
    32769,
    32770,
]
assert rows[0]["query_offset"] == 0
assert rows[3]["query_offset"] == 3
```

Also assert the returned target-token rows are exactly unchanged.

- [x] **Step 8: Record verify rows before argmax splitting**

In `_run_spec_verify_batch()`, immediately after `logits` is resolved and
before `rank != 0`, compute the pre-existing completion base from `items`:

```python
if self._spec_verify_trace.enabled:
    items_by_sequence = {
        int(item.sequence.seq_id): item
        for item in items
    }
    prediction_indices = []
    logical_block_identities = []
    for row in metadata.rows:
        item = items_by_sequence[row.sequence_id]
        prediction_base = (
            int(item.sequence.num_completion_tokens) + 1
        )
        prediction_indices.extend(
            prediction_base + offset
            for offset in range(row.query_len)
        )
        logical_block_identities.append(
            self._trace_block_identities(row.block_table)
        )
    self._spec_verify_trace.record_rows(
        stage="verify_tail",
        execution_mode="spec_verify",
        sequence_ids=tuple(
            row.sequence_id for row in metadata.rows
        ),
        query_offset=0,
        query_len=metadata.query_len,
        input_tokens=tuple(
            token
            for row in metadata.rows
            for token in row.input_tokens
        ),
        positions=tuple(
            position
            for row in metadata.rows
            for position in row.positions
        ),
        prediction_indices=tuple(prediction_indices),
        logical_block_identities=tuple(
            logical_block_identities
        ),
        logits=logits,
    )
```

- [x] **Step 9: Write and implement ordinary decode capture**

Add a RED test around `_run_model_step()` that:

- uses one decode sequence;
- enables trace;
- sets context;
- asserts one `ordinary_decode` row;
- asserts `prediction_index == seq.num_completion_tokens`; and
- asserts sampled token IDs remain unchanged.

In `_run_model_step()`, after `_select_sample_rows()` and before sampling, add
only for non-prefill, non-mixed decode:

```python
if (
    self._spec_verify_trace.enabled
    and not is_prefill
    and batch_kind != "mixed"
):
    self._spec_verify_trace.record_rows(
        stage="ordinary_decode",
        execution_mode="decode",
        sequence_ids=tuple(
            int(seq.seq_id) for seq in sample_seqs
        ),
        query_offset=0,
        query_len=1,
        input_tokens=tuple(
            int(seq.last_token)
            for seq in sample_seqs
        ),
        positions=tuple(
            int(seq.num_tokens - 1)
            for seq in sample_seqs
        ),
        prediction_indices=tuple(
            int(seq.num_completion_tokens)
            for seq in sample_seqs
        ),
        logical_block_identities=tuple(
            self._trace_block_identities(seq.block_table)
            for seq in sample_seqs
        ),
        logits=logits,
    )
```

Use the actual `Sequence` token/length property names already exercised by
the local implementation: `last_token`, `num_tokens`, and
`num_completion_tokens`. Do not add a compatibility fallback.

- [x] **Step 10: Prove `_last_step_logits_cpu` remains independent**

Add a regression test that enables both legacy step-logit recording and the
new trace recorder, drains trace rows, then asserts:

```python
assert runner.last_step_logits().tolist() == expected_logits
assert runner.drain_spec_verify_trace_rows()
assert runner.last_step_logits().tolist() == expected_logits
```

Disabling trace must not disable legacy step-logit recording.

Add a second regression test whose fake `run_model()` raises after trace has
been enabled. The worker-level `finally` cleanup remains authoritative, but
the ModelRunner test must prove no partial target row is appended before a
completed forward and `enable_spec_verify_trace_recording(False)` clears the
context and buffer.

- [x] **Step 11: Run focused ModelRunner tests GREEN**

Run the exact new tests plus existing first-target and verify tests:

```bash
python3 -m pytest -q \
  tools/test_model_runner_spec_verify.py \
  -k 'spec_verify_trace or first_target or run_spec_verify_batch'
```

Expected: all selected tests pass.

- [x] **Step 12: Run syntax and scoped diff checks**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/model_runner.py \
  tools/test_model_runner_spec_verify.py

git diff --check -- \
  tinyvllm/engine/model_runner.py \
  tools/test_model_runner_spec_verify.py
```

Expected: PASS.

### Task 3: Add Read-Only Qwen3.5 Side-State Lineage Fingerprints

**Files:**
- Create: `tinyvllm/engine/qwen35_speculative_trace.py`
- Modify: `tinyvllm/engine/qwen35_speculative_state.py`
- Modify: `tools/test_qwen35_speculative_state.py`

**Interfaces:**
- Produces:
  - `fingerprint_candidate_inventory(candidates) -> str`
  - `Qwen35SpeculativeTraceRecorder`
- Adds owner methods:
  - `enable_trace_recording(enabled: bool) -> dict`
  - `drain_trace_rows() -> tuple[dict, ...]`
- The owner emits only raw checkpoint events:
  `sequence_id`, `event`, `checkpoint_index`,
  `committed_input_count`, and `fingerprint`.
- The 32K worker binds each drained raw event to the active `engine_step` and
  enriches it into the exact final side-state lineage contract.
- Does not expose or retain raw tensors.

- [ ] **Step 1: Write RED fingerprint stability tests**

Add:

```python
from tinyvllm.engine.qwen35_speculative_trace import (
    Qwen35SpeculativeTraceRecorder,
    fingerprint_candidate_inventory,
)


def test_candidate_fingerprint_is_clone_stable_and_value_sensitive():
    candidates = (
        (
            torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
            torch.tensor([[3.0, 4.0]], dtype=torch.float32),
        ),
    )
    cloned = tuple(
        (convolution.clone(), recurrent.clone())
        for convolution, recurrent in candidates
    )
    changed = (
        (
            candidates[0][0].clone(),
            candidates[0][1].clone(),
        ),
    )
    changed[0][1][0, 0] = 9.0

    assert (
        fingerprint_candidate_inventory(candidates)
        == fingerprint_candidate_inventory(cloned)
    )
    assert (
        fingerprint_candidate_inventory(candidates)
        != fingerprint_candidate_inventory(changed)
    )
```

- [ ] **Step 2: Run fingerprint test RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_qwen35_speculative_state.py::test_candidate_fingerprint_is_clone_stable_and_value_sensitive
```

Expected: missing helper module.

- [x] **Step 3: Implement exact byte-level fingerprinting**

Create `tinyvllm/engine/qwen35_speculative_trace.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import torch


def fingerprint_candidate_inventory(candidates) -> str:
    if not isinstance(candidates, tuple) or not candidates:
        raise ValueError(
            "trace candidates must be a non-empty tuple"
        )
    digest = hashlib.sha256()
    for layer_index, pair in enumerate(candidates):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(
                "trace candidate must contain a state pair"
            )
        for state_name, tensor in zip(
            ("convolution", "recurrent"),
            pair,
        ):
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(
                    "trace candidate state must be a tensor"
                )
            contiguous = tensor.detach().contiguous()
            header = {
                "layer_index": layer_index,
                "state_name": state_name,
                "dtype": str(contiguous.dtype),
                "shape": list(contiguous.shape),
            }
            digest.update(
                json.dumps(
                    header,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(
                contiguous.view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            )
    return digest.hexdigest()


@dataclass(frozen=True)
class Qwen35SideStateTraceRow:
    sequence_id: int
    event: str
    checkpoint_index: int
    committed_input_count: int | None
    fingerprint: str

    def as_dict(self) -> dict:
        return asdict(self)


class Qwen35SpeculativeTraceRecorder:
    def __init__(self):
        self._enabled = False
        self._rows = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self, enabled: bool) -> dict:
        if not isinstance(enabled, bool):
            raise ValueError("trace enabled must be a boolean")
        self._enabled = enabled
        self._rows.clear()
        return {"enabled": enabled}

    def record_checkpoint(
        self,
        *,
        sequence_id: int,
        event: str,
        checkpoint_index: int,
        candidates,
    ) -> None:
        if not self._enabled:
            return
        if event not in (
            "first_target_checkpoint",
            "tail_checkpoint",
        ):
            raise ValueError(
                "trace checkpoint event is invalid"
            )
        if event == "first_target_checkpoint":
            if checkpoint_index != 1:
                raise ValueError(
                    "first-target checkpoint index must be one"
                )
        elif checkpoint_index < 2:
            raise ValueError(
                "tail checkpoint index must start at two"
            )
        self._rows.append(Qwen35SideStateTraceRow(
            sequence_id=int(sequence_id),
            event=event,
            checkpoint_index=int(checkpoint_index),
            committed_input_count=None,
            fingerprint=fingerprint_candidate_inventory(
                candidates
            ),
        ))

    def record_selection(
        self,
        *,
        sequence_id: int,
        committed_input_count: int,
        candidates,
    ) -> None:
        if not self._enabled:
            return
        if committed_input_count <= 0:
            raise ValueError(
                "selected checkpoint index must be positive"
            )
        self._rows.append(Qwen35SideStateTraceRow(
            sequence_id=int(sequence_id),
            event="selected_checkpoint",
            checkpoint_index=int(committed_input_count),
            committed_input_count=int(
                committed_input_count
            ),
            fingerprint=fingerprint_candidate_inventory(
                candidates
            ),
        ))

    def drain(self) -> tuple[dict, ...]:
        rows = tuple(row.as_dict() for row in self._rows)
        self._rows.clear()
        return rows
```

- [ ] **Step 4: Write RED owner lifecycle and ordering tests**

Using the existing owner fixture, add:

```python
def test_side_state_trace_is_default_off():
    owner, _, _, _ = _owner_fixture()
    assert owner.drain_trace_rows() == ()


def test_side_state_trace_records_first_tail_and_selection():
    owner, sequences, leases, prepared_steps = _owner_fixture()
    owner.enable_trace_recording(True)
    handle = owner.prepare(sequences, leases)
    owner.record_first_target(prepared_steps.first_target)
    owner.record_tail(
        prepared_steps.tail,
        tuple(sequence.seq_id for sequence in sequences),
    )
    owner.select(
        handle,
        (
            SpeculativeSideStateSelectionRow(
                sequence_id=sequences[0].seq_id,
                proposal_token_count=4,
                accepted_draft_count=2,
                verify_input_count=3,
                committed_tail_input_count=2,
                committed_input_count=3,
            ),
        ),
    )
    rows = owner.drain_trace_rows()

    assert rows[0]["event"] == "first_target_checkpoint"
    assert rows[0]["checkpoint_index"] == 1
    assert [
        row["checkpoint_index"]
        for row in rows
        if row["event"] == "tail_checkpoint"
    ] == [2, 3, 4]
    assert rows[-1]["event"] == "selected_checkpoint"
    assert rows[-1]["checkpoint_index"] == 3
    assert rows[-1]["committed_input_count"] == 3
```

Adapt `_owner_fixture()` to the existing fixture naming rather than creating a
duplicate state-transaction setup.

- [x] **Step 5: Wire recorder calls without changing candidate ownership**

In `Qwen35SpeculativeStateOwner.__init__`:

```python
self._trace = Qwen35SpeculativeTraceRecorder()
```

Add:

```python
def enable_trace_recording(self, enabled: bool) -> dict:
    return self._trace.enable(enabled)


def drain_trace_rows(self) -> tuple[dict, ...]:
    return self._trace.drain()
```

After assigning checkpoint 1 in `record_first_target()`:

```python
self._trace.record_checkpoint(
    sequence_id=sequence_id,
    event="first_target_checkpoint",
    checkpoint_index=1,
    candidates=batch.checkpoints[sequence_id][1],
)
```

After assigning each tail checkpoint in `record_tail()`:

```python
self._trace.record_checkpoint(
    sequence_id=sequence_id,
    event="tail_checkpoint",
    checkpoint_index=prefix_index,
    candidates=batch.checkpoints[
        sequence_id
    ][prefix_index],
)
```

After resolving `checkpoint` in `select()` and before storing it in
`selected`:

```python
self._trace.record_selection(
    sequence_id=expected_sequence_id,
    committed_input_count=checkpoint_index,
    candidates=checkpoint,
)
```

- [x] **Step 6: Prove fingerprinting does not mutate candidates**

Add a test that clones every checkpoint before draining and asserts exact
`torch.equal()` against every checkpoint after draining. Also assert:

```python
assert all(
    not any(
        isinstance(value, torch.Tensor)
        for value in row.values()
    )
    for row in rows
)
```

Also assert every raw row has exactly:

```python
{
    "sequence_id",
    "event",
    "checkpoint_index",
    "committed_input_count",
    "fingerprint",
}
```

The test must state that `schema`, `policy`, `batch_size`, `engine_step`,
proposal/accepted tokens, verify count, and fallback are worker-owned
enrichment fields and therefore do not belong in the owner recorder.

- [x] **Step 7: Run side-state tests GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_qwen35_speculative_state.py
```

Expected: all existing and new tests pass.

- [x] **Step 8: Run syntax and diff checks**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/qwen35_speculative_trace.py \
  tinyvllm/engine/qwen35_speculative_state.py \
  tools/test_qwen35_speculative_state.py

git diff --check -- \
  tinyvllm/engine/qwen35_speculative_trace.py \
  tinyvllm/engine/qwen35_speculative_state.py \
  tools/test_qwen35_speculative_state.py
```

Expected: PASS.

### Task 4: Assemble and Pair the 32K Diagnostic Trace

**Files:**
- Modify:
  `tools/qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py`
- Modify:
  `tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py`

**Interfaces:**
- Produces:
  - `run_generation_with_paired_trace(*, engine, prompt_rows, sampling_params, synchronize, policy, batch_size, trace_capture, target_forward_capture=None) -> tuple[list[dict], list[dict]]`
  - `run_paired_trace_cell(*, model_path, gpu_indices, policy, batch_size, dist_port, master_port, engine_factory, sampling_params_type, runtime_type, synchronize) -> dict`
  - `run_paired_trace_diagnostic(*, output_path, repo_root, cell_kwargs_by_key, run_cell_fn=run_paired_trace_cell) -> dict`
  - `pair_target_forward_rows(baseline_rows: list[dict], native_rows: list[dict]) -> list[dict]`
  - `assemble_side_state_lineage(*, policy: str, batch_size: int, trace_rows: list[dict], observations: list[dict], sequence_to_prompt: dict[int, int]) -> list[dict]`
  - `build_paired_trace_artifact(*, cells: dict[str, dict], source_manifest_sha256: str, target_manifest_sha256: str, mtp_manifest_sha256: str) -> dict`
  - `write_paired_trace_artifact(path, artifact)`
- Reuses frozen prompt generation and output-row validation.
- `run_generation_with_paired_trace()` preserves the inherited generation
  callback's exact two-value return contract:
  `(output_rows, observations)`.
- Diagnostic target rows, raw side rows, sequence-to-prompt mapping, and
  cleanup metadata move through the explicit mutable `trace_capture` sink.
- `run_paired_trace_cell()` passes that callback into the inherited
  `run_policy_cell()` so model/checkpoint validation, rank snapshots, KV
  movement accounting, and process cleanup remain unchanged.
- Does not replace the ordinary worker `main()` or authority path.

- [ ] **Step 1: Write RED tests for semantic keys and duplicate rejection**

Add a `WORKER_PATH` constant and load the 32K worker as a private module.
Create compact row fixtures with different runtime `sequence_id` values but
the same `prompt_index`, prediction identity, and logical coverage.

Add:

```python
def test_pair_trace_rows_uses_prompt_index_not_sequence_id():
    baseline = [_trace_row(
        policy="baseline",
        sequence_id=3,
        prompt_index=0,
    )]
    native = [_trace_row(
        policy="native_mtp",
        sequence_id=91,
        prompt_index=0,
    )]

    paired = worker.pair_target_forward_rows(
        baseline,
        native,
    )

    assert len(paired) == 1
    assert paired[0]["prompt_index"] == 0
    assert paired[0]["baseline_argmax_token"] == 15
    assert paired[0]["native_argmax_token"] == 15


def test_pair_trace_rows_rejects_missing_or_duplicate_matches():
    baseline = [_trace_row(policy="baseline")]
    native = [_trace_row(policy="native_mtp")]

    with pytest.raises(ValueError, match="duplicate native"):
        worker.pair_target_forward_rows(
            baseline,
            native + deepcopy(native),
        )

    with pytest.raises(ValueError, match="missing native"):
        worker.pair_target_forward_rows(baseline, [])
```

Also mutate one fixture by removing a required field and another by adding an
unknown field; both must fail exact-field validation before pairing.

- [x] **Step 2: Implement exact schema validation and semantic pairing**

In the 32K worker, add:

```python
TRACE_SCHEMA = (
    "qwen35.native-mtp-tp4-32k-paired-verify-trace.v1"
)
TRACE_LIMITATIONS = (
    "diagnostic_only",
    "full_logits_not_captured",
    "target_kv_shadow_not_established",
    "root_cause_not_established",
    "phase1_not_promotable",
    "performance_not_established",
)

_ENGINE_TRACE_ROW_FIELDS = frozenset({
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
})
_ENRICHED_TRACE_ROW_FIELDS = (
    _ENGINE_TRACE_ROW_FIELDS | {"prompt_index"}
)


def _validate_exact_fields(
    row: dict,
    expected: frozenset[str],
    name: str,
) -> None:
    if not isinstance(row, dict) or set(row) != expected:
        raise ValueError(f"{name} fields mismatch")


def _semantic_trace_key(row: dict) -> tuple:
    return (
        row["batch_size"],
        row["prompt_index"],
        row["prediction_index"],
        row["input_token_id"],
        row["position"],
        row["context_length"],
        tuple(
            tuple(value)
            for value in row["logical_block_coverage"]
        ),
    )


def pair_target_forward_rows(
    baseline_rows: list[dict],
    native_rows: list[dict],
) -> list[dict]:
    native_by_key = {}
    for row in native_rows:
        _validate_exact_fields(
            row,
            _ENRICHED_TRACE_ROW_FIELDS,
            "native trace row",
        )
        key = _semantic_trace_key(row)
        if key in native_by_key:
            raise ValueError("duplicate native trace match")
        native_by_key[key] = row
    paired = []
    for baseline in baseline_rows:
        _validate_exact_fields(
            baseline,
            _ENRICHED_TRACE_ROW_FIELDS,
            "baseline trace row",
        )
        key = _semantic_trace_key(baseline)
        native = native_by_key.pop(key, None)
        if native is None:
            raise ValueError("missing native trace match")
        shared = sorted(
            set(baseline["top_tokens"]).intersection(
                native["top_tokens"]
            )
        )
        baseline_logits = dict(zip(
            baseline["top_tokens"],
            baseline["top_logits"],
        ))
        native_logits = dict(zip(
            native["top_tokens"],
            native["top_logits"],
        ))
        paired.append({
            "batch_size": baseline["batch_size"],
            "prompt_index": baseline["prompt_index"],
            "prediction_index": baseline["prediction_index"],
            "input_token_id": baseline["input_token_id"],
            "position": baseline["position"],
            "context_length": baseline["context_length"],
            "baseline_stage": baseline["stage"],
            "native_stage": native["stage"],
            "baseline_query_len": baseline["query_len"],
            "native_query_len": native["query_len"],
            "baseline_top_tokens": baseline["top_tokens"],
            "native_top_tokens": native["top_tokens"],
            "baseline_top_logits": baseline["top_logits"],
            "native_top_logits": native["top_logits"],
            "baseline_argmax_token": baseline["argmax_token"],
            "native_argmax_token": native["argmax_token"],
            "argmax_equal": (
                baseline["argmax_token"]
                == native["argmax_token"]
            ),
            "baseline_logical_block_identities": baseline[
                "logical_block_identities"
            ],
            "native_logical_block_identities": native[
                "logical_block_identities"
            ],
            "logical_block_coverage_equal": (
                baseline["logical_block_coverage"]
                == native["logical_block_coverage"]
            ),
            "shared_token_logit_deltas": {
                str(token_id): (
                    float(native_logits[token_id])
                    - float(baseline_logits[token_id])
                )
                for token_id in shared
            },
            "first_topk_disagreement": (
                baseline["top_tokens"]
                != native["top_tokens"]
                or any(
                    native_logits[token_id]
                    != baseline_logits[token_id]
                    for token_id in shared
                )
            ),
            "baseline_target_forward_ordinal": baseline[
                "target_forward_ordinal"
            ],
            "native_target_forward_ordinal": native[
                "target_forward_ordinal"
            ],
        })
    if native_by_key:
        raise ValueError("unpaired native trace rows remain")
    return sorted(
        paired,
        key=lambda row: (
            row["prompt_index"],
            row["prediction_index"],
            row["native_target_forward_ordinal"],
        ),
    )
```

Before adding `prompt_index`, validate every drained ModelRunner row against
`_ENGINE_TRACE_ROW_FIELDS`. After enrichment, validate against
`_ENRICHED_TRACE_ROW_FIELDS`. Require each row's identity inventory to cover
every block ordinal in `logical_block_coverage`; generation presence was
already checked against
`KVOffloadMVP0.bound_generations[logical_block]` at capture time. Never use
raw logical block IDs as a cross-policy join key.

- [ ] **Step 3: Write RED lineage invariant tests**

Create side-state raw rows and one matching engine observation. Require:

```python
lineage = worker.assemble_side_state_lineage(
    policy="native_mtp",
    batch_size=1,
    trace_rows=side_rows,
    observations=[observation],
    sequence_to_prompt={7: 0},
)

selected = lineage[-1]
assert selected["committed_input_count"] == 3
assert selected["checkpoint_index"] == 3
assert selected["proposal_token_ids"] == [
    15,
    15,
    2658,
    8381,
]
assert selected["accepted_token_ids"] == [15, 15]
assert selected["verify_input_count"] == 3
assert selected["fallback_target_token"] == 220
```

Add mutations that reject:

- selected checkpoint index different from committed input count;
- accepted tokens not an exact proposal prefix; and
- partial fallback different from
  `new_completion_tokens[len(accepted_tokens)]`.

- [x] **Step 4: Implement lineage enrichment from existing observations**

Use these existing observation keys:

```text
speculative_proposal_token_ids_by_seq
speculative_accepted_draft_token_ids_by_seq
new_completion_tokens_by_seq
```

For each selected event:

```python
proposal = tuple(
    observation["speculative_proposal_token_ids_by_seq"][
        sequence_id
    ]
)
accepted = tuple(
    observation[
        "speculative_accepted_draft_token_ids_by_seq"
    ][sequence_id]
)
verify_input_count = max(0, len(proposal) - 1)
expected_committed = 1 + min(
    len(accepted),
    verify_input_count,
)
new_tokens = tuple(
    observation["new_completion_tokens_by_seq"][
        sequence_id
    ]
)
fallback = (
    new_tokens[len(accepted)]
    if len(new_tokens) > len(accepted)
    else None
)
```

Require `accepted == proposal[:len(accepted)]` and selected checkpoint equals
`expected_committed`. Preserve first-target and tail rows with empty
proposal/accepted fields and `fallback_target_token=None`.

Every finalized lineage row must have exactly:

```python
{
    "schema",
    "policy",
    "batch_size",
    "engine_step",
    "sequence_id",
    "event",
    "checkpoint_index",
    "committed_input_count",
    "proposal_token_ids",
    "accepted_token_ids",
    "verify_input_count",
    "fallback_target_token",
    "fingerprint",
}
```

Bind `engine_step` from the worker drain that captured the raw owner event.
Set `committed_input_count=None` on first-target and tail events. Reject any
missing or extra final field.

- [ ] **Step 5: Write RED diagnostic lifecycle tests with a fake engine**

Create a fake rank-zero model runner and side-state owner exposing:

```text
enable_spec_verify_trace_recording
set_spec_verify_trace_context
drain_spec_verify_trace_rows
enable_trace_recording
drain_trace_rows
```

Add tests proving:

- trace is explicitly enabled before the first step;
- context is set once per engine step;
- rows are drained only after `synchronize()`;
- both recorders are disabled in `finally`;
- a generation exception still disables and clears both recorders; and
- output rows, target-forward call count, KV movement counters, and cleanup
  inventory exactly match the same fake engine run with diagnostics disabled.

- [x] **Step 6: Implement the generation callback and cell wrapper**

Copy only the required control flow from the frozen generation loop while
preserving request insertion, output validation, observation capture, and the
exact two-value callback return shape. Replace legacy
`enable_step_logits_recording()` with:

```python
def run_generation_with_paired_trace(
    *,
    engine,
    prompt_rows,
    sampling_params,
    synchronize,
    policy,
    batch_size,
    trace_capture,
    target_forward_capture=None,
) -> tuple[list[dict], list[dict]]:
```

Keep the optional `target_forward_capture` keyword so the callback remains
drop-in compatible with the inherited native `run_policy_cell()` call. Use it
only for the same pre/post target-forward accounting already performed by the
frozen helper; do not alter its counters.

```python
runner = engine.model_runner
owner = getattr(
    runner,
    "qwen35_speculative_state_owner",
    None,
)
runner.enable_spec_verify_trace_recording(True)
if owner is not None:
    owner.enable_trace_recording(True)
target_rows = []
side_rows = []
observations = []
outputs_by_id = {}
try:
    engine_step = 0
    while not engine.is_finished():
        runner.set_spec_verify_trace_context(
            policy,
            batch_size,
            engine_step,
        )
        step_outputs, _ = engine.step()
        synchronize()
        target_rows.extend(
            runner.drain_spec_verify_trace_rows()
        )
        if owner is not None:
            side_rows.extend(
                {
                    **row,
                    "engine_step": engine_step,
                }
                for row in owner.drain_trace_rows()
            )
        observations.append(
            dict(engine.last_step_observation)
        )
        for sequence_id, token_ids in step_outputs:
            outputs_by_id[int(sequence_id)] = [
                int(token_id) for token_id in token_ids
            ]
        engine_step += 1
finally:
    runner.enable_spec_verify_trace_recording(False)
    if owner is not None:
        owner.enable_trace_recording(False)
```

Build the same `output_rows` contract as the inherited helper. Map
`sequence_id` to stable `prompt_index` from sorted final output IDs. Validate
raw target rows before enrichment, add `prompt_index`, then validate enriched
rows. Publish diagnostic side data through the explicit sink:

```python
trace_capture.clear()
trace_capture.update({
    "target_forward_trace_rows": target_rows,
    "raw_side_state_rows": side_rows,
    "sequence_to_prompt": sequence_to_prompt,
    "step_observations": observations,
})

return output_rows, observations
```

Add `run_paired_trace_cell()` that creates `trace_capture`, supplies a closure
around `run_generation_with_paired_trace()` as `run_generation_fn` to the
inherited `run_policy_cell()`, and receives the fully cleaned authority-shaped
cell. Build one diagnostic cell containing:

```python
diagnostic_cell = {
    "policy": cell["policy"],
    "batch_size": cell["batch_size"],
    "output_rows": cell["output_rows"],
    "target_forward_trace_rows": trace_capture[
        "target_forward_trace_rows"
    ],
    "side_state_lineage_rows": finalized_lineage,
    "step_observations": trace_capture[
        "step_observations"
    ],
    "rank_cleanup_summary": cell["cleanup"],
}
diagnostic_cell["cell_digest_sha256"] = gate._json_sha256(
    diagnostic_cell
)
```

The wrapper must not mutate the inherited cell, authority validator, ordinary
`run_policy_cell()`, or `main()`.

Do not change the ordinary
`run_generation_with_target_logit_diagnostics()` export or `main()`.

- [ ] **Step 7: Write RED artifact and first-divergence tests**

Build four synthetic cells and require:

```python
artifact = worker.build_paired_trace_artifact(
    cells=cells,
    source_manifest_sha256="a" * 64,
    target_manifest_sha256=gate.TARGET_MODEL_MANIFEST_SHA256,
    mtp_manifest_sha256=gate.MTP_CHECKPOINT_MANIFEST_SHA256,
)

assert artifact["schema"] == worker.TRACE_SCHEMA
assert artifact["first_divergence"]["prompt_index"] == 0
assert artifact["first_divergence"]["prediction_index"] == 3
assert artifact["limitations"] == list(
    worker.TRACE_LIMITATIONS
)
```

Also require a tensor anywhere in the artifact to raise
`ValueError("trace artifact contains a tensor")`.

Require the cell keys to be exactly:

```python
{
    "baseline:b1",
    "native_mtp:b1",
    "baseline:b4",
    "native_mtp:b4",
}
```

and verify each `cell_digest_sha256` against the canonicalized cell payload
with the digest field removed before pairing.

- [x] **Step 8: Implement artifact validation, divergence selection, and write**

Use a recursive tensor rejection helper:

```python
def _reject_tensors(value) -> None:
    if isinstance(value, torch.Tensor):
        raise ValueError("trace artifact contains a tensor")
    if isinstance(value, dict):
        for child in value.values():
            _reject_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_tensors(child)
```

Select the first divergence from paired rows where any of:

```python
not row["logical_block_coverage_equal"]
or not row["argmax_equal"]
or row["first_topk_disagreement"]
```

Build the frozen contract with:

```python
{
    "prompt_tokens": gate.PROMPT_TOKENS,
    "output_tokens": gate.MAX_OUTPUT_TOKENS,
    "world_size": gate.WORLD_SIZE,
    "batch_sizes": list(gate.BATCH_SIZES),
    "max_proposal_tokens": gate.MAX_PROPOSAL_TOKENS,
    "max_model_len": gate.MAX_MODEL_LEN,
    "max_num_batched_tokens": (
        gate.MAX_NUM_BATCHED_TOKENS
    ),
    "max_num_prefill_tokens_per_step": (
        gate.MAX_NUM_PREFILL_TOKENS_PER_STEP
    ),
    "kv_offload_gpu_blocks": gate.KV_OFFLOAD_GPU_BLOCKS,
    "kv_offload_logical_blocks": (
        gate.KV_OFFLOAD_LOGICAL_BLOCKS
    ),
    "block_size": gate.BLOCK_SIZE,
}
```

Build the top-level artifact with no extra fields:

```python
paired_rows = []
for batch_size in gate.BATCH_SIZES:
    paired_rows.extend(pair_target_forward_rows(
        cells[f"baseline:b{batch_size}"][
            "target_forward_trace_rows"
        ],
        cells[f"native_mtp:b{batch_size}"][
            "target_forward_trace_rows"
        ],
    ))
divergences = [
    row
    for row in paired_rows
    if (
        not row["logical_block_coverage_equal"]
        or not row["argmax_equal"]
        or row["first_topk_disagreement"]
    )
]
first_divergence = (
    min(
        divergences,
        key=lambda row: (
            row["prompt_index"],
            row["prediction_index"],
            row["native_target_forward_ordinal"],
        ),
    )
    if divergences
    else None
)
artifact = {
    "schema": TRACE_SCHEMA,
    "created_at_utc": (
        datetime.now(timezone.utc).isoformat()
    ),
    "source_manifest_sha256": source_manifest_sha256,
    "target_manifest_sha256": target_manifest_sha256,
    "mtp_manifest_sha256": mtp_manifest_sha256,
    "frozen_contract": frozen_contract,
    "cells": cells,
    "first_divergence": first_divergence,
    "limitations": list(TRACE_LIMITATIONS),
}
_reject_tensors(artifact)
return artifact
```

Import `datetime` and `timezone` from `datetime`. Tests must inject or
monkeypatch the clock boundary if they compare full artifacts; ordering and
digest tests must not depend on wall-clock time.

Write canonical JSON:

```python
def write_paired_trace_artifact(path, artifact) -> None:
    _reject_tensors(artifact)
    path = Path(path)
    if path.exists():
        raise ValueError(
            "paired trace artifact path already exists"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
```

Add `run_paired_trace_diagnostic()` as the only activation entry point. It
must:

1. compute `source_manifest_sha256` with
   `gate.source_tree_sha256(repo_root, gate.DEFAULT_SOURCE_FILES)`;
2. call `run_paired_trace_cell()` for exactly
   `baseline:b1`, `native_mtp:b1`, `baseline:b4`, and `native_mtp:b4`;
3. build the artifact with the frozen target/MTP manifest digests;
4. write it only under the caller-provided diagnostic directory; and
5. return the in-memory artifact for tests.

Do not call this function from module import, `main()`, the authority runner,
or the existing remote shell script. A later separately authorized run may
select it explicitly.

- [x] **Step 9: Run 32K worker/gate tests GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py \
  -k 'trace or pair or lineage or artifact or contract'
```

Expected: all selected tests pass.

- [x] **Step 10: Run syntax and scoped diff checks**

Run:

```bash
python3 -m py_compile \
  tools/qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py \
  tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py

git diff --check -- \
  tools/qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py \
  tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py
```

Expected: PASS.

### Task 5: Bind the New Diagnostic Sources and Prove Non-Invasiveness

**Files:**
- Modify:
  `tools/qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py`
- Modify:
  `tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py`
- Verify:
  `tinyvllm/engine/model_runner.py`
  `tinyvllm/engine/qwen35_speculative_state.py`
  `tinyvllm/engine/spec_verify_trace.py`
  `tinyvllm/engine/qwen35_speculative_trace.py`

**Interfaces:**
- Extends only `DEFAULT_SOURCE_FILES`.
- Leaves `SCHEMA_VERSION`, `CLASSIFICATION`, `validate_result()`, and the
  ordinary authority artifact unchanged.

- [ ] **Step 1: Write RED source-inventory isolation tests**

Extend `test_contract_constants_are_frozen()`:

```python
for source in (
    "tinyvllm/engine/spec_verify_trace.py",
    "tinyvllm/engine/qwen35_speculative_trace.py",
):
    assert source in gate.DEFAULT_SOURCE_FILES
```

Add:

```python
def test_trace_sources_do_not_change_authority_schema():
    assert gate.SCHEMA_VERSION == (
        "qwen35.native-mtp-tp4-32k-target-kv-offload.v1"
    )
    assert "paired-verify-trace" not in gate.SCHEMA_VERSION
    assert gate.validate_result(_valid_result()) == _valid_result()
```

- [x] **Step 2: Add only the helper files to the source inventory**

In the 32K gate's derived `DEFAULT_SOURCE_FILES`, append:

```python
"tinyvllm/engine/spec_verify_trace.py",
"tinyvllm/engine/qwen35_speculative_trace.py",
```

Do not modify frozen 16K source files or its source inventory.

- [x] **Step 3: Add a default-off source contract**

Add an AST/text source test that requires:

```text
enable_spec_verify_trace_recording(True)
```

to appear only in the explicit 32K diagnostic helper, not in:

```text
tinyvllm/engine/llm_engine.py
tinyvllm/engine/scheduler.py
tools/qwen35_native_mtp_tp4_16k_target_kv_offload_worker.py
```

Also require no call to `run_model()` or `engine.step()` was added to the
fingerprint helper.

- [x] **Step 4: Run the complete local focused suite**

Run:

```bash
python3 -m pytest -q \
  tools/test_spec_verify_trace.py \
  tools/test_qwen35_speculative_state.py \
  tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py \
  tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  tools/test_qwen35_generic_speculative_tp4_32k_gate.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_model_runner_spec_verify.py
```

Expected: all tests pass. Any unrelated pre-existing failure must be reported
without changing unrelated files.

- [x] **Step 5: Run compilation and shell syntax checks**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/spec_verify_trace.py \
  tinyvllm/engine/qwen35_speculative_trace.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/qwen35_speculative_state.py \
  tools/qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py \
  tools/test_spec_verify_trace.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_qwen35_speculative_state.py \
  tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py

bash -n \
  tools/run_qwen35_native_mtp_tp4_32k_target_kv_offload_remote.sh
```

Expected: PASS.

- [x] **Step 6: Run the complete scoped diff check**

Run:

```bash
git diff --check -- \
  tinyvllm/engine/spec_verify_trace.py \
  tinyvllm/engine/qwen35_speculative_trace.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/qwen35_speculative_state.py \
  tools/qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py \
  tools/qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py \
  tools/test_spec_verify_trace.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_qwen35_speculative_state.py \
  tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py \
  docs/superpowers/specs/2026-08-14-qwen35-tp4-32k-paired-verify-trace-design.md \
  docs/superpowers/plans/2026-08-14-qwen35-tp4-32k-paired-verify-trace.md
```

Expected: no whitespace errors.

- [x] **Step 7: Inspect the final change boundary**

Run:

```bash
git status --short

git diff --stat -- \
  tinyvllm/engine/spec_verify_trace.py \
  tinyvllm/engine/qwen35_speculative_trace.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/qwen35_speculative_state.py \
  tools/qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py \
  tools/qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py \
  tools/test_spec_verify_trace.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_qwen35_speculative_state.py \
  tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py
```

Confirm:

- no unrelated file was edited by this plan;
- no authority artifact was published;
- no remote command was run;
- no parity, KV budget, or proposal-length constraint changed; and
- the result remains `NOT_PROMOTABLE`.

## Completion Evidence

This implementation plan is complete only when local evidence proves:

1. diagnostics are default-off;
2. enabling diagnostics adds no target forward;
3. ordinary, first-target, and verify-tail rows are captured with exact
   semantic identity;
4. policy-local block generations are bound and logical coverage is explicit;
5. Qwen3.5 checkpoint fingerprints are stable and non-mutating;
6. accepted-prefix, fallback, committed-input, and selected-checkpoint
   invariants are enforced;
7. pairing uses prompt/prediction semantics rather than runtime sequence ID;
8. artifacts reject tensors and are source/checkpoint bound;
9. legacy step-logit recording and authority validation are unchanged; and
10. the complete focused local regression suite passes.

This evidence does not establish the root cause, 32K exact parity, remote
authority, performance, or Phase 1 promotion. Those require a later
explicitly authorized diagnostic run and a separate evidence-grounded fix
design.

## Plan Self-Review Record

- Spec activation and default-off lifecycle: Tasks 1, 2, 4, and 5.
- Exact target-forward row contract and deterministic top-five compaction:
  Tasks 1 and 2.
- Ordinary decode, direct first-target, fused native first-target, and
  verify-tail capture: Task 2.
- Policy-local target-KV generation binding and logical coverage: Tasks 1,
  2, and 4.
- First-target/tail/selected checkpoint fingerprints and non-mutation:
  Task 3.
- Exact final lineage fields, accepted-prefix, fallback, and
  committed-input invariants: Task 4.
- Prompt-index semantic pairing, duplicate/missing rejection, and
  deterministic first divergence: Task 4.
- Four required cells, canonical cell digests, source/checkpoint binding,
  tensor rejection, and separate diagnostic output: Task 4.
- Authority schema preservation, frozen 16K isolation, no extra forward,
  unchanged outputs/KV counters/cleanup, and complete local regression:
  Task 5.
- Verified existing names before execution:
  `Sequence.last_token`, `Sequence.num_tokens`,
  `Sequence.num_completion_tokens`,
  `KVOffloadMVP0.bound_generations` as a list,
  `_run_spec_first_target_batch()`,
  `run_spec_first_target_and_proposal_batch()`,
  `_run_spec_verify_batch()`, `record_first_target()`, `record_tail()`, and
  `select()`.

## 2026-08-14 Fresh Local Completion Evidence

The implementation was present in the dirty worktree and was re-audited
against the approved design and this plan. Historical RED steps were not
retroactively marked complete because this verification session did not
observe their original failing runs.

Fresh focused verification used the repository's existing offline uv cache:

```text
uv run --offline --python 3.12 --with pytest --with torch pytest -q \
  tools/test_spec_verify_trace.py \
  tools/test_qwen35_speculative_state.py \
  tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py

50 passed in 6.59s
```

Fresh complete local regression:

```text
uv run --offline --python 3.12 --with pytest --with torch pytest -q \
  tools/test_spec_verify_trace.py \
  tools/test_qwen35_speculative_state.py \
  tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py \
  tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  tools/test_qwen35_generic_speculative_tp4_32k_gate.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_model_runner_spec_verify.py

294 passed in 9.53s
```

Static validation:

```text
py_compile: PASS
bash -n tools/run_qwen35_native_mtp_tp4_32k_target_kv_offload_remote.sh: PASS
scoped git diff --check: PASS
```

Source binding was inspected through the actual manifest construction path.
The 16K frozen gate recursively binds every `tinyvllm/**/*.py` file, including
`spec_verify_trace.py` and `qwen35_speculative_trace.py`; the 32K overlay adds
its gate, worker, and verifier files. The ordinary authority schema remains
unchanged and trace activation remains explicit in the diagnostic worker.

This establishes:

```text
TP4_32K_PAIRED_VERIFY_TRACE_LOCAL_IMPLEMENTATION=ESTABLISHED
TP4_32K_PAIRED_VERIFY_TRACE_DEFAULT_OFF_CONTRACT=ESTABLISHED
TP4_32K_PAIRED_VERIFY_TRACE_SOURCE_BOUNDARY=ESTABLISHED
```

It does not establish:

```text
TP4_32K_FIRST_DIVERGENCE_ARTIFACT=NOT_ESTABLISHED
TP4_32K_ROOT_CAUSE=NOT_ESTABLISHED
TP4_32K_ENGINE_PARITY=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

No remote, GPU, or NCCL workload was launched during this local completion
verification.

## 2026-08-15 Fresh Checkbox Synchronization Evidence

The existing implementation was verified again before synchronizing plan
checkboxes. Historical steps explicitly labeled `RED` remain unchecked:
this session did not revert the implementation and therefore did not observe
the required failing state. All checked steps are supported by current source
inspection plus fresh GREEN or static evidence.

Fresh focused verification:

```text
tools/test_spec_verify_trace.py
tools/test_qwen35_speculative_state.py
tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py

50 passed in 7.73s
```

Fresh complete local regression:

```text
tools/test_spec_verify_trace.py
tools/test_qwen35_speculative_state.py
tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py
tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py
tools/test_qwen35_generic_speculative_tp4_32k_gate.py
tools/test_engine_speculative_runtime.py
tools/test_model_runner_spec_verify.py

294 passed in 9.62s
```

Fresh source-boundary subset:

```text
contract constants and helper inventory
authority schema preservation
explicit trace activation and no helper-added forward
frozen 16K gate source isolation

4 passed, 31 deselected in 1.33s
```

Fresh static validation:

```text
paired-trace py_compile set: PASS
remote-runner bash syntax: PASS
scoped git diff --check: PASS
final dirty-worktree boundary inspected: PASS
```

Checkbox state after synchronization:

```text
completed current-source/GREEN/static steps: 29
unchecked historical RED steps: 15
```

No remote, GPU, NCCL, loaded-checkpoint, performance, or authority workload
was launched. The diagnostic remains default-off and separate from the
ordinary authority artifact.

Strict boundary:

```text
TP4_32K_PAIRED_VERIFY_TRACE_LOCAL_IMPLEMENTATION=ESTABLISHED
TP4_32K_PAIRED_VERIFY_TRACE_DEFAULT_OFF_CONTRACT=ESTABLISHED
TP4_32K_PAIRED_VERIFY_TRACE_SOURCE_BOUNDARY=ESTABLISHED
TP4_32K_FIRST_DIVERGENCE_ARTIFACT=NOT_ESTABLISHED
TP4_32K_ROOT_CAUSE=NOT_ESTABLISHED
TP4_32K_ENGINE_PARITY=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```
