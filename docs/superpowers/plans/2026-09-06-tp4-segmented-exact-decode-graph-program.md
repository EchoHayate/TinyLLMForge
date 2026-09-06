# TP4 Segmented Exact Decode Graph Program Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Qwen3.8-27B BF16 TP4 pool-index decode graph satisfy the frozen per-capture limit by replacing one monolithic graph with a source-bound composite of contiguous graph segments, without weakening correctness or performance gates.

**Architecture:** Add a pure segment-plan/composite-graph contract, range-bounded Qwen3.8 layer execution with graph-owned hidden and state-candidate buffers, and a strict-clean capture-cost census before production integration. Only a census plan with at least 10% capture-budget headroom may enter the existing exact graph cache and TP4 evidence pipeline.

**Tech Stack:** Python 3.12, PyTorch CUDA Graphs, torch.distributed/NCCL, pytest, Qwen3.8 hybrid-state runtime, JSON/JSONL evidence, SSH/Kerberos remote controller.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge`.
- Do not update `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not create a worktree or dispatch subagents; execute this plan inline.
- Push only to `origin/feat/kv-sparse-attention`.
- Use strict RED -> minimal implementation -> GREEN for every runtime change.
- Stage exact paths only; never use `git add -A`, `git reset`, `git clean`, or broad formatting.
- Commit with `git -c core.hooksPath=/dev/null commit`.
- Every commit must contain exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Preserve all unrelated tracked and untracked files.
- Keep `multi_sequence_cuda_graph_segmented_capture` default-disabled.
- Require `multi_sequence_cuda_graphs` and `multi_sequence_cuda_graph_dynamic_pool_indices` when segmented capture is enabled.
- Preserve ordered `slot_id + generation + request_id` validation before every replay.
- Preserve the current warmup/measured graph-cache reset boundary.
- Keep the frozen production limits unchanged:
  - maximum individual capture `2_000_000_000 ns`;
  - maximum complete capture per rank `5_000_000_000 ns`;
  - replay coverage at least `0.80`;
  - exact token and text equality;
  - existing memory, throughput, TTFT, TPOT, and P99 E2E gates.
- Use stricter Stage 0 selection limits:
  - maximum individual capture `1_800_000_000 ns`;
  - maximum complete capture lifecycle `4_500_000_000 ns`.
- Do not move setup, gather, commit, final norm, or LM head outside measured accounting merely to pass a gate.
- Formal evidence requires `strict_clean`; `shared_capacity` remains diagnostic only.
- Do not execute `kinit` or `krenew`.
- Never terminate, suspend, adopt, or clean foreign GPU processes.
- Keep every remote source, cache, log, artifact, and temporary file below `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`.
- Do not copy large remote artifacts to the Mac.
- Prior run tags and evidence directories are immutable.
- Do not use the prohibited unused tag `20260906-qwen38-tp4-decode-replay-r55-pool-index-full`.
- A Stage 0 census is not a performance claim.

---

## File and Responsibility Map

- `tinyvllm/config.py`
  - owns the default-false segmented-capture flag and dependency validation.
- `tinyvllm/engine/segmented_exact_cuda_graph.py`
  - new dependency-light segment-plan, canonical hash, composite graph, and
    capture-accounting contracts.
- `tinyvllm/engine/flash_attn_split_policy.py`
  - binds the selected segment-plan hash into the stable program identity.
- `tinyvllm/engine/exact_cuda_graph_cache.py`
  - applies max-segment and complete-program capture budgets while preserving
    legacy one-graph behavior.
- `tinyvllm/layers/qwen35_packed_layer_stack.py`
  - executes exact contiguous layer ranges and writes graph-owned candidates.
- `tinyvllm/models/qwen35_packed.py`
  - exposes segmented pool-index prepare, range, finalize, and commit hooks.
- `tinyvllm/engine/model_runner.py`
  - captures, synchronizes, commits, replays, and resets one composite graph
    entry.
- `tinyvllm/engine/exact_cuda_graph_capture_receipt.py`
  - records segment-aware capture phases without changing legacy receipts.
- `tools/tp4_segmented_capture_census_worker.py`
  - new remote TP4 worker for bounded two/three/four-segment census plans.
- `tools/run_tp4_segmented_capture_census.py`
  - new strict-clean controller, source freezer, lifecycle owner, and artifact
    collector for Stage 0.
- `tools/verify_tp4_segmented_capture_census.py`
  - new independent census verifier and deterministic classifier.
- `tools/tp4_decode_replay_worker.py`
  - enables the selected segmented plan only for graph arms after Stage 0 GO.
- `tools/assemble_tp4_decode_replay.py`
  - emits segment-plan and segment-duration evidence.
- `tools/verify_tp4_decode_replay.py`
  - independently reconstructs segmented capture limits and TP agreement.
- Focused test files under `tools/test_*.py`
  - provide RED/GREEN coverage adjacent to each contract.
- `docs/superpowers/audits/2026-08-31-tp4-collective-stable-decode-replay-audit.md`
  - receives the immutable census, smoke, and conditional full-gate result.
- `AGENT_HANDOFF_STATE.md`
  - receives the final resumable state.

---

### Task 1: Add the segment-plan and composite-graph contracts

**Files:**
- Create: `tinyvllm/engine/segmented_exact_cuda_graph.py`
- Create: `tools/test_segmented_exact_cuda_graph.py`

**Interfaces:**
- Produces: `ExactGraphSegment(start_layer: int, end_layer: int, include_embedding: bool, include_final: bool, include_commit: bool)`
- Produces: `ExactGraphSegmentPlan(layer_count: int, segments: tuple[ExactGraphSegment, ...])`
- Produces: `ExactGraphSegmentPlan.sha256: str`
- Produces: `CompositeExactCudaGraph(graphs: tuple[object, ...], shared_pool: object)`
- Produces: `CompositeCaptureAccounting(segment_capture_durations_ns: tuple[int, ...], lifecycle_duration_ns: int)`

- [ ] **Step 1: Write failing pure-contract tests**

Create `tools/test_segmented_exact_cuda_graph.py` with:

```python
from dataclasses import replace

import pytest

from tinyvllm.engine.segmented_exact_cuda_graph import (
    CompositeCaptureAccounting,
    CompositeExactCudaGraph,
    ExactGraphSegment,
    ExactGraphSegmentPlan,
)


def plan(*ranges):
    segments = []
    for ordinal, (start, end) in enumerate(ranges):
        segments.append(ExactGraphSegment(
            start_layer=start,
            end_layer=end,
            include_embedding=ordinal == 0,
            include_final=ordinal == len(ranges) - 1,
            include_commit=ordinal == len(ranges) - 1,
        ))
    return ExactGraphSegmentPlan(
        layer_count=ranges[-1][1],
        segments=tuple(segments),
    )


def test_plan_requires_exact_contiguous_coverage():
    assert plan((0, 22), (22, 43), (43, 64)).layer_count == 64
    with pytest.raises(ValueError, match="contiguous"):
        plan((0, 22), (23, 64))
    with pytest.raises(ValueError, match="cover"):
        ExactGraphSegmentPlan(
            layer_count=64,
            segments=(ExactGraphSegment(1, 64, True, True, True),),
        )


def test_plan_hash_binds_boundaries_and_stage_owners():
    first = plan((0, 22), (22, 43), (43, 64))
    moved = plan((0, 21), (21, 43), (43, 64))
    four_segments = plan((0, 16), (16, 32), (32, 48), (48, 64))
    assert first.sha256 != moved.sha256
    assert first.sha256 != four_segments.sha256


def test_plan_rejects_invalid_stage_owners():
    canonical = plan((0, 22), (22, 43), (43, 64))
    with pytest.raises(ValueError, match="embedding"):
        replace(
            canonical,
            segments=(
                replace(
                    canonical.segments[0],
                    include_embedding=False,
                ),
                canonical.segments[1],
                canonical.segments[2],
            ),
        )


def test_composite_replay_and_reset_are_ordered():
    events = []

    class Graph:
        def __init__(self, ordinal):
            self.ordinal = ordinal

        def replay(self):
            events.append(("replay", self.ordinal))

        def reset(self):
            events.append(("reset", self.ordinal))

    composite = CompositeExactCudaGraph(
        graphs=(Graph(0), Graph(1), Graph(2)),
        shared_pool="pool",
    )
    composite.replay()
    composite.reset()
    assert events == [
        ("replay", 0), ("replay", 1), ("replay", 2),
        ("reset", 2), ("reset", 1), ("reset", 0),
    ]
    assert composite.pool() == "pool"


def test_capture_accounting_exposes_single_and_total_views():
    accounting = CompositeCaptureAccounting(
        segment_capture_durations_ns=(1_100, 1_300, 900),
        lifecycle_duration_ns=3_700,
    )
    assert accounting.max_segment_capture_duration_ns == 1_300
    assert accounting.total_capture_duration_ns == 3_700
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONPATH=. /usr/bin/python3 -m pytest \
  tools/test_segmented_exact_cuda_graph.py -q
```

Expected: collection fails because
`tinyvllm.engine.segmented_exact_cuda_graph` does not exist.

- [ ] **Step 3: Implement the minimal pure contracts**

Create `tinyvllm/engine/segmented_exact_cuda_graph.py` with:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


def _sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ExactGraphSegment:
    start_layer: int
    end_layer: int
    include_embedding: bool
    include_final: bool
    include_commit: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.start_layer, bool)
            or isinstance(self.end_layer, bool)
            or self.start_layer < 0
            or self.end_layer <= self.start_layer
        ):
            raise ValueError("segment layer range is invalid")
        for name in (
            "include_embedding",
            "include_final",
            "include_commit",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")


@dataclass(frozen=True)
class ExactGraphSegmentPlan:
    layer_count: int
    segments: tuple[ExactGraphSegment, ...]
    schema_version: str = "tinyllmforge.segmented-exact-graph-plan.v1"

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_count, bool)
            or self.layer_count <= 0
            or not self.segments
        ):
            raise ValueError("segment plan must contain positive layers")
        cursor = 0
        for ordinal, segment in enumerate(self.segments):
            if segment.start_layer != cursor:
                raise ValueError("segment ranges must be contiguous")
            if segment.include_embedding != (ordinal == 0):
                raise ValueError("embedding must belong to the first segment")
            if segment.include_final != (
                ordinal == len(self.segments) - 1
            ):
                raise ValueError("final stage must belong to the last segment")
            cursor = segment.end_layer
        if cursor != self.layer_count:
            raise ValueError("segment ranges must cover every model layer")
        if sum(segment.include_commit for segment in self.segments) != 1:
            raise ValueError("exactly one segment must own state commit")
        if not self.segments[-1].include_commit:
            raise ValueError("state commit must belong to the last segment")

    @property
    def sha256(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class CompositeCaptureAccounting:
    segment_capture_durations_ns: tuple[int, ...]
    lifecycle_duration_ns: int

    def __post_init__(self) -> None:
        if not self.segment_capture_durations_ns:
            raise ValueError("segment capture durations must be non-empty")
        if any(
            isinstance(value, bool) or value < 0
            for value in self.segment_capture_durations_ns
        ):
            raise ValueError("segment capture durations must be non-negative")
        if (
            isinstance(self.lifecycle_duration_ns, bool)
            or self.lifecycle_duration_ns < 0
        ):
            raise ValueError("lifecycle duration must be non-negative")
        if self.lifecycle_duration_ns < sum(
            self.segment_capture_durations_ns
        ):
            raise ValueError(
                "lifecycle duration cannot be below segment total"
            )

    @property
    def max_segment_capture_duration_ns(self) -> int:
        return max(self.segment_capture_durations_ns)

    @property
    def total_capture_duration_ns(self) -> int:
        return self.lifecycle_duration_ns


class CompositeExactCudaGraph:
    def __init__(self, *, graphs: tuple[object, ...], shared_pool):
        if not graphs:
            raise ValueError("composite graph requires at least one graph")
        if any(
            not callable(getattr(graph, "replay", None))
            or not callable(getattr(graph, "reset", None))
            for graph in graphs
        ):
            raise ValueError("every segment graph must support replay/reset")
        self.graphs = graphs
        self.shared_pool = shared_pool
        self._reset = False

    def replay(self) -> None:
        if self._reset:
            raise RuntimeError("composite graph was reset")
        for graph in self.graphs:
            graph.replay()

    def reset(self) -> None:
        if self._reset:
            return
        for graph in reversed(self.graphs):
            graph.reset()
        self._reset = True

    def pool(self):
        return self.shared_pool
```

- [ ] **Step 4: Run focused GREEN verification**

Run:

```bash
PYTHONPATH=. /usr/bin/python3 -m pytest \
  tools/test_segmented_exact_cuda_graph.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add -- \
  tinyvllm/engine/segmented_exact_cuda_graph.py \
  tools/test_segmented_exact_cuda_graph.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): add segmented graph contracts" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

Expected: one commit containing exactly the two named paths.

---

### Task 2: Add exact range-bounded Qwen3.8 execution

**Files:**
- Modify: `tinyvllm/layers/qwen35_packed_layer_stack.py`
- Modify: `tinyvllm/models/qwen35_packed.py`
- Modify: `tools/test_qwen35_prepared_model_step.py`

**Interfaces:**
- Produces: `Qwen35SegmentCandidates(layer_indices, values)`
- Produces: `Qwen35PackedHeterogeneousLayerStack.prepare_pool_index_range(...)`
- Produces: `Qwen35PackedForCausalLM.embed_exact_graph_inputs(...)`
- Produces: `Qwen35PackedForCausalLM.run_exact_cuda_graph_layer_range(...)`
- Produces: `Qwen35PackedForCausalLM.finalize_exact_cuda_graph_hidden(...)`
- Produces: `Qwen35PackedForCausalLM.commit_exact_cuda_graph_candidates(...)`

- [ ] **Step 1: Write failing range and state-isolation tests**

Add focused tests that use the existing fake packed layers and state adapters:

```python
def test_pool_index_ranges_equal_full_stack_and_preserve_layer_order():
    model, fixture = make_segmented_qwen35_fixture(layer_count=6)
    slot_ids = fixture.slot_ids
    hidden = model.embed_exact_graph_inputs(fixture.input_ids)
    first = model.run_exact_cuda_graph_layer_range(
        state_slot_ids=slot_ids,
        token_counts=fixture.token_counts,
        position_ids=fixture.position_ids,
        hidden_states=hidden,
        start_layer=0,
        end_layer=2,
    )
    second = model.run_exact_cuda_graph_layer_range(
        state_slot_ids=slot_ids,
        token_counts=fixture.token_counts,
        position_ids=fixture.position_ids,
        hidden_states=first.hidden_states,
        start_layer=2,
        end_layer=6,
    )
    logits = model.finalize_exact_cuda_graph_hidden(
        second.hidden_states
    )
    model.commit_exact_cuda_graph_candidates(
        slot_ids,
        first.candidates + second.candidates,
    )
    assert torch.equal(logits, fixture.eager_logits)
    assert fixture.selected_state_equals_eager()
    assert fixture.unselected_state_is_unchanged()


def test_range_candidates_are_keyed_by_model_layer_index():
    model, fixture = make_segmented_qwen35_fixture(
        block_types=(
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ),
    )
    result = model.run_exact_cuda_graph_layer_range(
        state_slot_ids=fixture.slot_ids,
        token_counts=fixture.token_counts,
        position_ids=fixture.position_ids,
        hidden_states=model.embed_exact_graph_inputs(
            fixture.input_ids
        ),
        start_layer=1,
        end_layer=4,
    )
    assert result.layer_indices == (2,)
```

Also add validation tests for an empty range, a range outside
`[0, len(layers)]`, duplicate candidates, missing stateful-layer candidates,
and candidate order drift.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=. /usr/bin/python3 -m pytest \
  tools/test_qwen35_prepared_model_step.py \
  -k 'pool_index_range or segment_candidates' -q
```

Expected: failures for missing segmented model and layer-stack methods.

- [ ] **Step 3: Implement the range result and layer-stack method**

In `tinyvllm/layers/qwen35_packed_layer_stack.py`, add:

```python
@dataclass(frozen=True)
class Qwen35SegmentCandidates:
    layer_indices: tuple[int, ...]
    values: tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass(frozen=True)
class Qwen35PreparedLayerRange:
    hidden_states: torch.Tensor
    candidates: Qwen35SegmentCandidates
```

Implement `prepare_pool_index_range(...)` by validating the half-open range
and mapping each linear layer through:

```python
adapter_by_layer = {
    adapter.layer_index: adapter
    for adapter in self.state_transaction.adapters
}
```

Use the following method body, retaining the existing helper calls:

```python
def prepare_pool_index_range(
    self,
    *,
    state_slot_ids: torch.Tensor,
    token_counts: tuple[int, ...],
    position_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    start_layer: int,
    end_layer: int,
) -> Qwen35PreparedLayerRange:
    self.state_transaction.adapters[0]._validate_slot_tensor(
        state_slot_ids
    )
    if (
        isinstance(start_layer, bool)
        or isinstance(end_layer, bool)
        or start_layer < 0
        or end_layer <= start_layer
        or end_layer > len(self.layers)
    ):
        raise ValueError("layer range is invalid")
    if len(token_counts) != state_slot_ids.shape[0]:
        raise ValueError(
            "slot_ids and token_counts batch size must match"
        )
    if sum(token_counts) != hidden_states.shape[0]:
        raise ValueError(
            "token_counts sum must match hidden_states token count"
        )
    adapter_by_layer = {
        adapter.layer_index: adapter
        for adapter in self.state_transaction.adapters
    }
    candidate_indices = []
    candidate_values = []
    for layer_index in range(start_layer, end_layer):
        layer = self.layers[layer_index]
        with profile_layer(layer_index, layer.block_type):
            if layer.block_type == "full_attention":
                hidden_states = self._run_full_layer(
                    layer,
                    token_counts,
                    position_ids,
                    hidden_states,
                )
                continue
            adapter = adapter_by_layer[layer_index]
            convolution_states, recurrent_states = (
                adapter.gather_batch_by_slot_tensor(state_slot_ids)
            )
            hidden_states, candidate, _ = self._run_linear_layer(
                layer,
                adapter,
                token_counts,
                hidden_states,
                convolution_states,
                recurrent_states,
                capture_prefix_states=False,
            )
            candidate_indices.append(layer_index)
            candidate_values.append(candidate)
    return Qwen35PreparedLayerRange(
        hidden_states=hidden_states,
        candidates=Qwen35SegmentCandidates(
            layer_indices=tuple(candidate_indices),
            values=tuple(candidate_values),
        ),
    )
```

Return hidden states plus candidates keyed by exact model-layer index. Do not
commit any state in this method.

- [ ] **Step 4: Implement the model-level segmented hooks**

In `tinyvllm/models/qwen35_packed.py`, add methods with these signatures:

```python
def embed_exact_graph_inputs(
    self,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.ndim != 1
        or input_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("input_ids must be a rank-one integer tensor")
    with profile_layer(len(self.layer_stack.layers), "embedding"):
        hidden_states = self.embed_tokens(input_ids)
    self._validate_hidden_output(
        hidden_states,
        token_count=input_ids.shape[0],
        name="embed_tokens",
    )
    return hidden_states

def run_exact_cuda_graph_layer_range(
    self,
    *,
    state_slot_ids: torch.Tensor,
    token_counts: tuple[int, ...],
    position_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    start_layer: int,
    end_layer: int,
) -> Qwen35PreparedLayerRange:
    return self.layer_stack.prepare_pool_index_range(
        state_slot_ids=state_slot_ids,
        token_counts=token_counts,
        position_ids=position_ids,
        hidden_states=hidden_states,
        start_layer=start_layer,
        end_layer=end_layer,
    )

def finalize_exact_cuda_graph_hidden(
    self,
    hidden_states: torch.Tensor,
) -> torch.Tensor | None:
    normalized = self.final_norm(hidden_states)
    self._validate_hidden_output(
        normalized,
        token_count=hidden_states.shape[0],
        name="final_norm",
        reference=hidden_states,
    )
    logits = self.lm_head(normalized)
    self._validate_logits(logits, normalized)
    return logits

def commit_exact_cuda_graph_candidates(
    self,
    state_slot_ids: torch.Tensor,
    candidates: tuple[Qwen35SegmentCandidates, ...],
) -> None:
    adapter_by_layer = {
        adapter.layer_index: adapter
        for adapter in self.layer_stack.state_transaction.adapters
    }
    flattened = {}
    for segment in candidates:
        for layer_index, value in zip(
            segment.layer_indices,
            segment.values,
            strict=True,
        ):
            if layer_index in flattened:
                raise ValueError("duplicate state candidate layer")
            flattened[layer_index] = value
    if tuple(sorted(flattened)) != self.layer_stack.linear_indices:
        raise ValueError("state candidate layer inventory mismatch")
    for layer_index in self.layer_stack.linear_indices:
        convolution, recurrent = flattened[layer_index]
        adapter_by_layer[layer_index].commit_batch_by_slot_tensor(
            state_slot_ids,
            convolution,
            recurrent,
        )
```

The commit method must canonicalize candidates by model-layer index, require
exactly the model's `linear_indices`, then call each matching adapter's
`commit_batch_by_slot_tensor()`. It must not infer adapter position from the
number of preceding full-attention layers.

- [ ] **Step 5: Run focused and adjacent GREEN verification**

Run:

```bash
PYTHONPATH=. /usr/bin/python3 -m pytest \
  tools/test_qwen35_prepared_model_step.py \
  tools/test_qwen35_packed_layer_stack.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add -- \
  tinyvllm/layers/qwen35_packed_layer_stack.py \
  tinyvllm/models/qwen35_packed.py \
  tools/test_qwen35_prepared_model_step.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(qwen35): add segmented graph execution" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

Expected: one commit containing only the named implementation and test paths.

---

### Task 3: Build the isolated TP4 segmented-capture census

**Files:**
- Create: `tools/tp4_segmented_capture_census_worker.py`
- Create: `tools/run_tp4_segmented_capture_census.py`
- Create: `tools/verify_tp4_segmented_capture_census.py`
- Create: `tools/test_tp4_segmented_capture_census_worker.py`
- Create: `tools/test_run_tp4_segmented_capture_census.py`
- Create: `tools/test_verify_tp4_segmented_capture_census.py`

**Interfaces:**
- Produces schema: `tinyllmforge.tp4-segmented-capture-census.v1`
- Produces classifications:
  - `GO_SEGMENT_PLAN_SELECTED`
  - `NO_GO_SEGMENTED_CAPTURE_CEILING`
  - `NO_GO_CORRECTNESS_OR_LIFECYCLE`
  - `INCOMPLETE`
- Produces selected plan only when all ranks agree and every Stage 0 gate passes.

- [ ] **Step 1: Write failing schema, classifier, and lifecycle tests**

Test the pure verifier with synthetic complete rows:

```python
def test_verifier_selects_smallest_plan_with_headroom():
    bundle = make_bundle(
        plans={
            "p2": durations((1_950_000_000, 1_700_000_000), 4_100_000_000),
            "p3": durations(
                (1_300_000_000, 1_250_000_000, 1_200_000_000),
                4_200_000_000,
            ),
            "p4": durations(
                (980_000_000,) * 4,
                4_400_000_000,
            ),
        },
        exact=True,
        state_exact=True,
        cleanup="CLEAN",
    )
    result = verify_bundle(bundle)
    assert result["classification"] == "GO_SEGMENT_PLAN_SELECTED"
    assert result["selected_plan_id"] == "p3"


def test_verifier_rejects_hidden_total_capture_cost():
    bundle = make_bundle(
        plans={
            "p3": durations(
                (1_200_000_000,) * 3,
                4_600_000_000,
            ),
        },
        exact=True,
        state_exact=True,
        cleanup="CLEAN",
    )
    assert verify_bundle(bundle)["classification"] == (
        "NO_GO_SEGMENTED_CAPTURE_CEILING"
    )
```

Add tests for missing ranks, plan-hash disagreement, duplicate row IDs,
segment-range disagreement, output mismatch, selected-state mismatch,
unselected-state mutation, incomplete cleanup, non-strict admission, source
identity mismatch, and exact-tag residue.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
PYTHONPATH=.:tools /usr/bin/python3 -m pytest \
  tools/test_tp4_segmented_capture_census_worker.py \
  tools/test_run_tp4_segmented_capture_census.py \
  tools/test_verify_tp4_segmented_capture_census.py -q
```

Expected: collection fails because the three census modules do not exist.

- [ ] **Step 3: Implement the worker**

`tp4_segmented_capture_census_worker.py` must:

1. construct Q1's exact four-request shape;
2. create a fresh engine for each candidate plan;
3. snapshot selected and unselected hybrid-state slots plus scratch KV;
4. allocate stable input, hidden-boundary, candidate, and logits tensors;
5. capture every compute segment and optional commit graph using one pool;
6. time capture body, post-capture synchronize, complete segment, and complete
   lifecycle separately;
7. restore state and scratch KV after capture;
8. replay the composite exactly once;
9. compare logits and selected post-state against eager;
10. compare all unselected slots byte-for-byte;
11. reset every graph and emit a process receipt.

Use deterministic plans:

```python
CANDIDATE_PLANS = {
    "p2": ((0, 32), (32, 64)),
    "p3": ((0, 22), (22, 43), (43, 64)),
    "p4": ((0, 16), (16, 32), (32, 48), (48, 64)),
}
```

Every row ID must include plan, segment, and rank:

```python
row_id = (
    f"{plan_id}:segment-{segment_ordinal}:rank-{rank}"
)
```

- [ ] **Step 4: Implement the controller**

Reuse the safety helpers from `tools/run_tp4_decode_replay.py`:

- Kerberos TTL preflight;
- SSH retry and ControlMaster handling;
- remote storage preflight;
- strict-clean four-GPU admission;
- exact source archive and source-tree SHA;
- exact-tag process ownership;
- bounded timeout;
- exact-tag-only cleanup;
- remote artifact streaming without downloading model/cache data.

The controller CLI is:

```text
python3 tools/run_tp4_segmented_capture_census.py monitor-and-run
  --run-tag TAG
  --admission-mode strict_clean
  --ssh-target sitian@10.232.195.203
  --remote-python /data00/home/sitian/tllm/env/bin/python
```

Reject all remote roots outside:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
```

- [ ] **Step 5: Implement the independent verifier**

The verifier must recompute:

```python
plan_pass = (
    max(tp_wide_segment_durations_ns) <= 1_800_000_000
    and tp_wide_lifecycle_duration_ns <= 4_500_000_000
    and exact_output
    and selected_state_exact
    and unselected_state_unchanged
    and cleanup == "CLEAN"
)
```

Choose the passing plan with the fewest compute segments; break ties by lower
TP-wide lifecycle duration and then lexical plan ID. Emit no selected plan for
any non-GO classification.

- [ ] **Step 6: Run GREEN verification**

Run:

```bash
PYTHONPATH=.:tools /usr/bin/python3 -m pytest \
  tools/test_segmented_exact_cuda_graph.py \
  tools/test_qwen35_prepared_model_step.py \
  tools/test_tp4_segmented_capture_census_worker.py \
  tools/test_run_tp4_segmented_capture_census.py \
  tools/test_verify_tp4_segmented_capture_census.py -q
```

Expected: all tests pass.

Run:

```bash
/usr/bin/python3 -m py_compile \
  tinyvllm/engine/segmented_exact_cuda_graph.py \
  tools/tp4_segmented_capture_census_worker.py \
  tools/run_tp4_segmented_capture_census.py \
  tools/verify_tp4_segmented_capture_census.py
```

Expected: exit code 0 and no output.

- [ ] **Step 7: Commit Task 3**

Run:

```bash
git add -- \
  tools/tp4_segmented_capture_census_worker.py \
  tools/run_tp4_segmented_capture_census.py \
  tools/verify_tp4_segmented_capture_census.py \
  tools/test_tp4_segmented_capture_census_worker.py \
  tools/test_run_tp4_segmented_capture_census.py \
  tools/test_verify_tp4_segmented_capture_census.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): add segmented capture census" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

Expected: one commit containing only the six census paths.

---

### Task 4: Run Stage 0 and apply the stop rule

**Files:**
- Create remotely and stream locally:
  `artifacts/tp4_segmented_capture_census/20260906-qwen38-tp4-segmented-capture-r56/`

**Interfaces:**
- Consumes: committed source SHA and exact source-tree SHA
- Produces: immutable selected plan or terminal `NO_GO`

- [ ] **Step 1: Verify the run tag is unused**

Run local and remote read-only checks:

```bash
test ! -e \
  artifacts/tp4_segmented_capture_census/20260906-qwen38-tp4-segmented-capture-r56
ssh sitian@10.232.195.203 \
  'test ! -e /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/tp4-segmented-capture-census/20260906-qwen38-tp4-segmented-capture-r56'
```

Expected: both commands exit 0 with no output.

- [ ] **Step 2: Launch the strict-clean census**

Run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
PYTHONPATH=.:tools \
/usr/bin/python3 tools/run_tp4_segmented_capture_census.py \
  monitor-and-run \
  --run-tag 20260906-qwen38-tp4-segmented-capture-r56 \
  --admission-mode strict_clean \
  --ssh-target sitian@10.232.195.203 \
  --remote-python /data00/home/sitian/tllm/env/bin/python
```

Expected: the controller waits for four strict-clean GPUs, runs one exact-tag
owner, streams the bounded evidence, verifies it independently, and exits with
a terminal classification.

- [ ] **Step 3: Enforce the Stage 0 decision**

Require all of:

```text
classification == GO_SEGMENT_PLAN_SELECTED
selected_plan_id is present
max TP-wide segment duration <= 1,800,000,000 ns
TP-wide lifecycle duration <= 4,500,000,000 ns
exact output == true
selected state exact == true
unselected state unchanged == true
cleanup == CLEAN
no exact-tag process remains
```

If any condition fails:

1. classify the design `NO_GO_SEGMENTED_CAPTURE_CEILING` or
   `NO_GO_CORRECTNESS_OR_LIFECYCLE`;
2. do not execute Tasks 5–8;
3. continue directly to Task 9 to record the negative result.

- [ ] **Step 4: Record the selected source-controlled plan**

If Stage 0 passes, add the exact selected ranges and plan SHA to
`tinyvllm/engine/segmented_exact_cuda_graph.py` as the Qwen3.8-27B TP4
profile. Do not add runtime autotuning.

Run the Task 1 and Task 3 focused tests again and commit only the profile and
its tests:

```bash
git add -- \
  tinyvllm/engine/segmented_exact_cuda_graph.py \
  tools/test_segmented_exact_cuda_graph.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): select segmented capture plan" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 5: Integrate composite capture into the exact graph cache

**Precondition:** Task 4 produced `GO_SEGMENT_PLAN_SELECTED`.

**Files:**
- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/flash_attn_split_policy.py`
- Modify: `tinyvllm/engine/exact_cuda_graph_cache.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/exact_cuda_graph_capture_receipt.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`
- Modify: `tools/test_model_runner_spec_verify.py`
- Modify: `tools/test_model_runner_spec_verify_cuda_graph.py`

**Interfaces:**
- Produces: `Config.multi_sequence_cuda_graph_segmented_capture: bool`
- Produces protocol: `lease_pool_index_segmented_v1`
- Extends: `ExactCudaGraphEntry.segment_capture_durations_ns`
- Extends: `ExactCudaGraphEntry.total_capture_duration_ns`
- Preserves: legacy entry construction and one-graph budget behavior

- [ ] **Step 1: Write failing configuration and identity tests**

Add:

```python
def test_segmented_capture_defaults_false_and_requires_pool_indices():
    assert Config().multi_sequence_cuda_graph_segmented_capture is False
    with pytest.raises(ValueError, match="requires dynamic pool indices"):
        Config(
            multi_sequence_cuda_graphs=True,
            multi_sequence_cuda_graph_dynamic_pool_indices=False,
            multi_sequence_cuda_graph_segmented_capture=True,
        )


def test_segment_plan_hash_changes_stable_program_key():
    first = make_identity(
        execution_protocol="lease_pool_index_segmented_v1",
        segment_plan_sha256="1" * 64,
    )
    second = replace(first, segment_plan_sha256="2" * 64)
    assert first.cache_key_sha256 != second.cache_key_sha256
```

- [ ] **Step 2: Write failing cache-accounting tests**

Add:

```python
def test_composite_entry_uses_max_segment_for_single_budget():
    entry = make_entry(
        capture_duration_ns=1_900_000_000,
        segment_capture_durations_ns=(
            1_200_000_000,
            1_900_000_000,
            1_100_000_000,
        ),
        total_capture_duration_ns=4_400_000_000,
    )
    cache.commit_capture(entry)
    assert entry.cache_key_sha256 in cache.ready_entries


def test_composite_entry_uses_lifecycle_for_total_budget():
    entry = make_entry(
        capture_duration_ns=1_500_000_000,
        segment_capture_durations_ns=(1_500_000_000,) * 3,
        total_capture_duration_ns=5_000_000_001,
    )
    cache.commit_capture(entry)
    assert cache.rejected[entry.cache_key_sha256] == (
        "total_capture_budget"
    )
```

Also prove that a legacy entry with no segment tuple retains its current
single-duration semantics.

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```bash
PYTHONPATH=. /usr/bin/python3 -m pytest \
  tools/test_multi_sequence_cuda_graph_gate.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_model_runner_spec_verify_cuda_graph.py \
  -k 'segmented or composite_entry or segment_plan' -q
```

Expected: failures for the missing flag, identity field, protocol, and entry
accounting.

- [ ] **Step 4: Implement configuration, identity, and cache accounting**

Add:

```python
multi_sequence_cuda_graph_segmented_capture: bool = False
```

Validate it as a strict boolean and require both existing graph flags.

Extend `FlashAttentionGraphIdentity` with:

```python
segment_plan_sha256: str = ""
```

For `lease_pool_index_segmented_v1`, derive the stable key exactly like
`lease_pool_index_v1`: include the segment-plan hash and exclude only the
concrete lease seal.

Extend `ExactCudaGraphEntry` with backward-compatible defaults:

```python
segment_capture_durations_ns: tuple[int, ...] = ()
total_capture_duration_ns: int | None = None
segment_plan_sha256: str = ""
```

Normalize a legacy entry to one segment. Apply:

```python
single_capture_ns = max(entry.segment_capture_durations_ns)
complete_capture_ns = (
    entry.capture_duration_ns
    if entry.total_capture_duration_ns is None
    else entry.total_capture_duration_ns
)
```

Use `single_capture_ns` for `max_single_capture_ns` and
`complete_capture_ns` for process-wide total accounting.

- [ ] **Step 5: Implement segmented capture and replay**

In `ModelRunner`:

1. select `lease_pool_index_segmented_v1` only when the new flag is true;
2. bind the source-controlled segment plan into graph identity;
3. allocate stable hidden boundaries and candidate storage;
4. capture each segment with a shared pool;
5. capture or fold commit exactly as selected by Stage 0;
6. create `CompositeExactCudaGraph`;
7. TP-all-reduce the complete duration vector with element-wise MAX;
8. return one atomic cache entry;
9. reuse the existing replay call through the wrapper.

The capture structure must be:

```python
graphs = []
segment_durations_ns = []
shared_pool = self._exact_multi_sequence_capture_pool()
for segment in plan.segments:
    graph = torch.cuda.CUDAGraph()
    started_ns = time.perf_counter_ns()
    with torch.cuda.graph(graph, pool=shared_pool):
        run_segment(segment)
    torch.cuda.synchronize()
    segment_durations_ns.append(
        time.perf_counter_ns() - started_ns
    )
    if shared_pool is None:
        shared_pool = graph.pool()
    graphs.append(graph)
```

Do not synchronize inside `CompositeExactCudaGraph.replay()`.

- [ ] **Step 6: Extend capture receipts**

Keep legacy receipt fields. Add deterministic segment rows containing:

```text
segment_ordinal
start_layer
end_layer
include_embedding
include_final
include_commit
capture_begin_ns
capture_body_completed_ns
capture_end_synchronize_completed_ns
capture_duration_ns
```

The parent receipt must include the plan hash, compute-segment count, captured
graph count, maximum segment duration, and complete lifecycle duration.

- [ ] **Step 7: Run focused and adjacent GREEN verification**

Run:

```bash
PYTHONPATH=. /usr/bin/python3 -m pytest \
  tools/test_segmented_exact_cuda_graph.py \
  tools/test_qwen35_prepared_model_step.py \
  tools/test_multi_sequence_cuda_graph_gate.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_model_runner_spec_verify_cuda_graph.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 5**

Run:

```bash
git add -- \
  tinyvllm/config.py \
  tinyvllm/engine/flash_attn_split_policy.py \
  tinyvllm/engine/exact_cuda_graph_cache.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/exact_cuda_graph_capture_receipt.py \
  tools/test_multi_sequence_cuda_graph_gate.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_model_runner_spec_verify_cuda_graph.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): integrate segmented decode graphs" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

Expected: one commit containing exactly the listed runtime and test files.

---

### Task 6: Extend the schema-v2 evidence path

**Precondition:** Task 4 produced `GO_SEGMENT_PLAN_SELECTED`.

**Files:**
- Modify: `tools/tp4_decode_replay_worker.py`
- Modify: `tools/assemble_tp4_decode_replay.py`
- Modify: `tools/verify_tp4_decode_replay.py`
- Modify: `tools/test_tp4_decode_replay_worker.py`
- Modify: `tools/test_assemble_tp4_decode_replay.py`
- Modify: `tools/test_verify_tp4_decode_replay.py`

**Interfaces:**
- Produces per-segment capture-cost rows with globally unique row IDs.
- Produces TP-wide max-segment and complete-program capture values.
- Preserves all existing schema-v2 correctness and lifecycle requirements.

- [ ] **Step 1: Write failing evidence tests**

Add tests proving:

```python
assert graph_config[
    "multi_sequence_cuda_graph_segmented_capture"
] is True
assert eager_config[
    "multi_sequence_cuda_graph_segmented_capture"
] is False
```

Require capture row IDs:

```text
CASE:capture-cost:step-N:segment-M:rank-R
```

Add assembler and verifier fixtures where:

- every segment is below `2 s` and total is below `5 s` -> accepted;
- one segment is above `2 s` -> `single_capture_budget`;
- every segment is below `2 s` but total is above `5 s` ->
  `total_capture_budget`;
- ranks disagree on plan hash or segment inventory -> incomplete/negative;
- duplicate segment row IDs -> invalid evidence.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
PYTHONPATH=.:tools /usr/bin/python3 -m pytest \
  tools/test_tp4_decode_replay_worker.py \
  tools/test_assemble_tp4_decode_replay.py \
  tools/test_verify_tp4_decode_replay.py \
  -k 'segment or capture_cost or graph_config' -q
```

Expected: failures for missing segmented configuration and evidence fields.

- [ ] **Step 3: Implement worker, assembler, and verifier changes**

The worker must enable segmented capture only for graph arms and emit every
segment row. The assembler and verifier must independently calculate:

```python
tp_wide_segment_ns[ordinal] = max(
    row["capture_duration_ns"]
    for row in rank_rows_for_segment
)
max_segment_capture_ns = max(tp_wide_segment_ns.values())
total_capture_ns = max(
    row["total_capture_duration_ns"]
    for row in complete_rank_program_rows
)
```

Do not infer total cost by multiplying the maximum segment. Require direct
lifecycle evidence from every rank.

- [ ] **Step 4: Run GREEN verification**

Run:

```bash
PYTHONPATH=.:tools /usr/bin/python3 -m pytest \
  tools/test_tp4_decode_replay_worker.py \
  tools/test_assemble_tp4_decode_replay.py \
  tools/test_verify_tp4_decode_replay.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 6**

Run:

```bash
git add -- \
  tools/tp4_decode_replay_worker.py \
  tools/assemble_tp4_decode_replay.py \
  tools/verify_tp4_decode_replay.py \
  tools/test_tp4_decode_replay_worker.py \
  tools/test_assemble_tp4_decode_replay.py \
  tools/test_verify_tp4_decode_replay.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): verify segmented graph evidence" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 7: Complete local verification and source review

**Precondition:** Task 4 produced `GO_SEGMENT_PLAN_SELECTED`.

**Files:**
- Inspect all files changed by Tasks 1–6.
- Create: `artifacts/code-review/tp4-segmented-exact-decode-graph/`

- [ ] **Step 1: Run focused suites**

Run:

```bash
PYTHONPATH=.:tools /usr/bin/python3 -m pytest \
  tools/test_segmented_exact_cuda_graph.py \
  tools/test_qwen35_prepared_model_step.py \
  tools/test_qwen35_packed_layer_stack.py \
  tools/test_multi_sequence_cuda_graph_gate.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_model_runner_spec_verify_cuda_graph.py \
  tools/test_tp4_segmented_capture_census_worker.py \
  tools/test_run_tp4_segmented_capture_census.py \
  tools/test_verify_tp4_segmented_capture_census.py \
  tools/test_tp4_decode_replay_worker.py \
  tools/test_assemble_tp4_decode_replay.py \
  tools/test_verify_tp4_decode_replay.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run the established Python 3.12 regression suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:/Users/bytedance/dev/TinyLLMForge \
/opt/homebrew/bin/python3.12 -m pytest -q \
  tools/test_segmented_exact_cuda_graph.py \
  tools/test_qwen35_prepared_model_step.py \
  tools/test_qwen35_packed_layer_stack.py \
  tools/test_multi_sequence_cuda_graph_gate.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_model_runner_spec_verify_cuda_graph.py \
  tools/test_tp4_segmented_capture_census_worker.py \
  tools/test_run_tp4_segmented_capture_census.py \
  tools/test_verify_tp4_segmented_capture_census.py \
  tools/test_tp4_decode_replay_contract.py \
  tools/test_tp4_decode_replay_worker.py \
  tools/test_assemble_tp4_decode_replay.py \
  tools/test_verify_tp4_decode_replay.py \
  tools/test_run_tp4_decode_replay.py
```

Expected baseline: at least the prior `458 passed, 1 skipped`, plus all newly
added tests, with zero failures.

- [ ] **Step 3: Run syntax and diff checks**

Run:

```bash
/usr/bin/python3 -m py_compile \
  tinyvllm/engine/segmented_exact_cuda_graph.py \
  tinyvllm/layers/qwen35_packed_layer_stack.py \
  tinyvllm/models/qwen35_packed.py \
  tinyvllm/engine/model_runner.py \
  tools/tp4_segmented_capture_census_worker.py \
  tools/run_tp4_segmented_capture_census.py \
  tools/verify_tp4_segmented_capture_census.py \
  tools/tp4_decode_replay_worker.py \
  tools/assemble_tp4_decode_replay.py \
  tools/verify_tp4_decode_replay.py
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 4: Review the exact source diff**

Use `bits-code-guard` on the commits introduced by this plan. Resolve every
high-confidence correctness, lifecycle, concurrency, memory-accounting, or
evidence-integrity issue using RED -> GREEN. Do not modify unrelated files.

- [ ] **Step 5: Commit any review fixes separately**

Stage only exact reviewed paths and commit:

```bash
git -c core.hooksPath=/dev/null commit \
  -m "fix(tp4): harden segmented graph lifecycle" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

Skip this commit when review requires no source change.

---

### Task 8: Run the strict-clean smoke and conditional full gate

**Precondition:** Tasks 4–7 passed.

**Files:**
- Create:
  `artifacts/tp4_decode_replay/20260906-qwen38-tp4-decode-replay-r57-segmented-q1-smoke/`
- Conditionally create:
  `artifacts/tp4_decode_replay/20260906-qwen38-tp4-decode-replay-r58-segmented-full/`

- [ ] **Step 1: Verify both tags are unused**

Check local and remote exact paths before launch. Any existing path makes that
tag unusable; choose the next unused monotonically increasing tag and record
the substitution in the audit.

- [ ] **Step 2: Launch the Q1 smoke**

Run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
PYTHONPATH=.:tools \
/usr/bin/python3 tools/run_tp4_decode_replay.py monitor-and-run \
  --run-tag 20260906-qwen38-tp4-decode-replay-r57-segmented-q1-smoke \
  --admission-mode strict_clean \
  --ssh-target sitian@10.232.195.203 \
  --remote-python /data00/home/sitian/tllm/env/bin/python \
  --case-id Q1__r0__eager \
  --case-id Q1__r0__graph
```

Require exactly `SMOKE_PASS`. Also inspect raw rows and receipts for:

```text
four-rank segment-plan agreement
every segment <= 2,000,000,000 ns
complete capture <= 5,000,000,000 ns
cross-lease replay on every rank
replay coverage >= 0.80
exact output and state equality
unselected state unchanged
memory within frozen limits
cleanup == CLEAN
no exact-tag process remains
```

If any condition fails, classify r57 terminally and skip r58.

- [ ] **Step 3: Launch the full gate only after SMOKE_PASS**

Run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
PYTHONPATH=.:tools \
/usr/bin/python3 tools/run_tp4_decode_replay.py monitor-and-run \
  --run-tag 20260906-qwen38-tp4-decode-replay-r58-segmented-full \
  --admission-mode strict_clean \
  --ssh-target sitian@10.232.195.203 \
  --remote-python /data00/home/sitian/tllm/env/bin/python
```

Require exactly 30 complete cases, 15 complete eager/graph pairs, producer
classification, remote independent verification, local frozen-source
verification, immutable manifest, post-verification hashes, and `CLEAN`
cleanup.

- [ ] **Step 4: Apply the final classification**

Only matching producer and both independent verifier results of
`GO_STAGE1_JUSTIFIED` authorize a positive claim. Otherwise preserve the exact
terminal negative or incomplete classification and its failed gates.

---

### Task 9: Reconcile audit, handoff, commit, and push

**Files:**
- Modify:
  `docs/superpowers/audits/2026-08-31-tp4-collective-stable-decode-replay-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [ ] **Step 1: Append the prompt-to-artifact checklist**

Map every design requirement to concrete evidence:

```text
source commit and source-tree SHA
model repository and immutable revision
strict-clean GPU admission and UUIDs
Kerberos TTL and remote storage preflight
selected segment plan, plan hash, and rank agreement
per-segment and complete capture durations
exact output and selected-state equality
unselected-state immutability
cross-lease replay and replay coverage
throughput, TTFT, TPOT, P99 E2E, and memory
producer and both independent verifier classifications
manifest and post-verification hashes
owned-process and process-group cleanup
local/tracking/GitHub SHA equality
```

- [ ] **Step 2: State benefit and cost without overclaiming**

Report:

- whether segmented capture crossed the frozen capture ceilings;
- whether it produced actual replay;
- steady-state benefit or regression for every frozen performance metric;
- extra launches, retained memory, and first-capture latency;
- whether evidence is census-only, smoke-only, diagnostic, incomplete, or
  full formal evidence.

- [ ] **Step 3: Run final documentation checks**

Run:

```bash
git diff --check -- \
  docs/superpowers/audits/2026-08-31-tp4-collective-stable-decode-replay-audit.md \
  AGENT_HANDOFF_STATE.md
```

Expected: exit code 0.

- [ ] **Step 4: Commit exact documentation paths**

Run:

```bash
git add -- \
  docs/superpowers/audits/2026-08-31-tp4-collective-stable-decode-replay-audit.md \
  AGENT_HANDOFF_STATE.md
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "docs(tp4): record segmented graph result" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

- [ ] **Step 5: Push and verify SHA equality**

Run:

```bash
git push origin feat/kv-sparse-attention
LOCAL_SHA=$(git rev-parse HEAD)
TRACKING_SHA=$(git rev-parse origin/feat/kv-sparse-attention)
REMOTE_SHA=$(git ls-remote origin refs/heads/feat/kv-sparse-attention | awk '{print $1}')
test "$LOCAL_SHA" = "$TRACKING_SHA"
test "$LOCAL_SHA" = "$REMOTE_SHA"
```

Expected: all three SHAs are identical.

- [ ] **Step 6: Perform the final completion audit**

Do not close the task until the prompt-to-artifact checklist covers:

- the design and implementation commits;
- RED/GREEN evidence;
- Stage 0 stop-rule outcome;
- smoke and conditional full-gate outcome;
- both verifiers and manifest integrity;
- benefit and cost;
- cleanup;
- exact branch and remote SHA.

Any uncovered or uncertain item remains incomplete.
