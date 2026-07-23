# Decode Residency-Aware Read-Window Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This repository's approved execution mode is inline; do not dispatch subagents.

**Goal:** Add a decode-only cross-layer residency-aware read-window planner to blockwise KV offload, then independently determine from source-bound remote evidence whether it reduces real H2D reloads or evictions without correctness, copy, memory, or latency regressions.

**Architecture:** Replace the unkeyed decode plan list with an immutable context-local `BlockwiseDecodePlan` containing an exact cache identity plus forward and reverse `BlockwiseDecodeWindow` records. Each record separates required blocks, same-layer future hints, and spare-capacity-bounded next-layer reuse hints; only required blocks load or wait, while both hint sets enter the existing `KVOffloadMVP0` eviction score as soft future information. A dedicated contract, remote runner, and independent verifier freeze the four staging shapes, single- and multi-prompt workloads, five measured repetitions, source identity, correctness rules, movement thresholds, and `GO | NO_GO | INVALID` classification.

**Tech Stack:** Python 3, dataclasses, PyTorch, TinyLLMForge blockwise online-softmax attention, `KVOffloadMVP0`, JSON/JSONL, SHA-256, dependency-light script tests, Qwen3-0.6B BF16 on remote A100 GPU 0, SSH ControlMaster.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`; never modify `/Users/bytedance/dev/TinyLLMForge`.
- Execute inline in the current session; do not dispatch subagents.
- Preserve unrelated untracked `experiments/` directories; stage exact paths only and never use `git add -A`.
- The approved design is `docs/superpowers/specs/2026-07-22-decode-residency-aware-read-window-planner-design.md`.
- Limit runtime behavior changes to the already default-off `kv_offload_blockwise_decode` path.
- Do not change prefill planning, prefill attention, scheduler behavior, CUDA Graphs, speculative decoding, quantization, Light Doc Cache, Gist KV sharing, token sparsity, low rank, Attention Matching, public defaults, or read-window size.
- Preserve even-layer forward traversal, odd-layer reverse traversal, and `layer_idx < 0` forward compatibility behavior.
- Preserve exact online-softmax equations, window membership, masks, GQA math, token order, and sampling.
- Load and wait for only the current window's required blocks.
- Cross-layer reuse blocks are soft future hints only: never load them, protect them, wait for them, add them to pending waits, or increment prefetch counters for them.
- Bound cross-layer hints by `gpu_blocks - unique(required_blocks union write_blocks)`; zero spare capacity must match current behavior.
- Do not add proactive next-window or next-layer H2D prefetch.
- Reuse the existing `ensure_resident()` eviction scoring and H2D/D2H batching; do not add a second residency manager or copy coalescer.
- Do not catch or downgrade unreadable-block, capacity, dirty-writeback, copy, CUDA, or attention correctness failures.
- GPU/model work runs only on `sitian@10.232.195.203` as user `sitian`.
- Use SSH ControlMaster `/tmp/ssh-sitian-10.232.195.203`.
- Use remote Python `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`.
- Use model `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`.
- Set `CUDA_VISIBLE_DEVICES=0`.
- Every model process receives fresh, mutually distinct dynamic `TINYVLLM_DIST_PORT` and `MASTER_PORT`.
- Do not modify the remote checkout, use `rsync`, kill unrelated processes, switch GPUs, or clean shared `/tmp`.
- Retry only `EADDRINUSE`, with a fresh port pair.
- Canonical baseline and candidate use immutable staged source snapshots and identical model/runtime/workload configuration.
- Run at least five measured repetitions per policy, workload, and staging shape after one excluded warmup.
- Exact token equality and strict per-step logits comparison are mandatory.
- `GO` requires H2D copies or evictions improve by at least `5%`; the other may worsen by at most `1%`.
- `copy_waits`, `prefetch_plans`, D2H copies/bytes, dirty evictions/writeback, peak staging blocks, and peak CUDA memory must not worsen.
- Median decode latency may worsen by at most `2%`.
- At least one low-capacity shape and the multi-prompt thrash workload must satisfy the `>=5%` movement improvement.
- Planner-counter improvements alone cannot produce `GO`.
- Do not update `README.md` or claim performance improvement before independent `GO`.

---

## File Map

- Modify `tinyvllm/layers/attention.py`: define immutable plan identity/window/plan dataclasses, build both directional records together, validate context-local cache identity, pass cross-layer hints only through `future_logical_blocks`, and maintain planner diagnostics.
- Modify `tinyvllm/engine/model_runner.py`: add planner diagnostic counters to `KVOffloadMVP0.stats`/`summary()` without changing residency semantics.
- Modify `tools/test_blockwise_attention_planning.py`: TDD for directional hints, spare-capacity bounds, zero-capacity equivalence, cache invalidation, staging isolation, resident fast path, and simulated multi-layer movement.
- Modify `tools/test_kv_offload.py`: regression tests proving future-only blocks affect victim score but are not loaded, protected, pending, or waited.
- Create `tools/kv_decode_residency_planner_contract.py`: frozen shapes, workloads, repetitions, thresholds, artifact names, case IDs, canonical JSON, and classification helpers.
- Create `tools/run_kv_decode_residency_planner_gate_remote.py`: immutable source staging, dynamic remote ports, local worker mode, paired baseline/candidate orchestration, artifact download, and verifier invocation.
- Create `tools/verify_kv_decode_residency_planner_gate.py`: independent provenance, domain, correctness, counter, latency, memory, and classification reconstruction.
- Create `tools/test_kv_decode_residency_planner_gate.py`: dependency-light contract, command, artifact, threshold, and tamper tests.
- Modify `tools/profile_ngram_commit.py`: optional per-step logits capture, decode-only timing aggregation, peak memory/staging evidence, and planner diagnostics required by the gate.
- Modify `tools/smoke_blockwise_prefill_remote.sh`: include focused KV tests in preflight and keep default behavior unchanged.
- Modify `AGENT_HANDOFF_STATE.md`: record implementation, commands, raw canonical artifact, exact outcome, negative branches, and limitations after evidence exists.
- Modify `README.md` only if the independent canonical classification is `GO`.
- Create raw run directories under `experiments/kv_offload/$RUN_TAG/`; never stage raw experiment directories.
- Create `docs/kv_offload_evidence_registry.json`: tracked closed-schema index for this gate's spec, plan, canonical artifact, verifier, and classification.

## Shared Interfaces

Use these exact immutable planner types in `tinyvllm/layers/attention.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class BlockwiseDecodePlanIdentity:
    block_rows: tuple[tuple[int, ...], ...]
    context_lens: tuple[int, ...]
    max_blocks: int
    block_size: int
    window_blocks: int
    write_blocks: tuple[int, ...]
    gpu_blocks: int


@dataclass(frozen=True)
class BlockwiseDecodeWindow:
    window_rows: tuple[tuple[int, ...], ...]
    window_lens: tuple[int, ...]
    required_blocks: tuple[int, ...]
    intra_layer_future_blocks: tuple[int, ...]
    cross_layer_reuse_blocks: tuple[int, ...]
    max_window_tokens: int


@dataclass(frozen=True)
class BlockwiseDecodePlan:
    identity: BlockwiseDecodePlanIdentity
    forward_windows: tuple[BlockwiseDecodeWindow, ...]
    reverse_windows: tuple[BlockwiseDecodeWindow, ...]
```

Use these exact planner helpers:

```python
def _build_blockwise_decode_plan_identity(
    block_rows: list[list[int]],
    context_lens: list[int],
    max_blocks: int,
    block_size: int,
    window_blocks: int,
    write_blocks: set[int],
    gpu_blocks: int,
) -> BlockwiseDecodePlanIdentity


def _ordered_unique_excluding(
    blocks,
    excluded: set[int],
) -> tuple[int, ...]


def _bounded_cross_layer_reuse_blocks(
    *,
    candidate_blocks,
    required_blocks: tuple[int, ...],
    write_blocks: set[int],
    gpu_blocks: int,
) -> tuple[int, ...]


def _build_blockwise_decode_window_plan(
    block_rows: list[list[int]],
    context_lens: list[int],
    max_blocks: int,
    block_size: int,
    window_blocks: int,
    write_blocks: set[int],
    gpu_blocks: int,
) -> BlockwiseDecodePlan


def _build_residency_aware_blockwise_decode_window_plan(
    block_rows: list[list[int]],
    context_lens: list[int],
    max_blocks: int,
    block_size: int,
    window_blocks: int,
    write_blocks: set[int],
    gpu_blocks: int,
) -> BlockwiseDecodePlan
```

Task 1 implements
`_build_residency_aware_blockwise_decode_window_plan()` side by side with the
legacy runtime builder so every commit remains green. Task 2 replaces the
legacy builder body with the residency-aware implementation, updates callers,
and removes the temporary helper name.

Use these exact new planner counters:

```text
decode_plan_builds
decode_plan_cache_hits
decode_plan_identity_invalidations
decode_windows_with_spare_capacity
decode_cross_layer_hint_blocks
decode_cross_layer_hint_resident
decode_cross_layer_hint_retained
```

Use this exact contract domain:

```python
STAGING_SHAPES = (
    (2, 1),
    (3, 2),
    (4, 1),
    (4, 2),
)

WORKLOADS = (
    "single_long_context",
    "multi_prompt_thrash",
)

POLICIES = ("baseline", "candidate")
WARMUP_REPETITIONS = 1
CORRECTNESS_REPETITIONS = 1
MEASURED_REPETITIONS = 5
LOGIT_RTOL = 1e-3
LOGIT_ATOL = 1e-2

THRESHOLDS = {
    "movement_improvement": 0.05,
    "other_movement_max_regression": 0.01,
    "decode_latency_max_regression": 0.02,
}
```

Every raw case row uses these exact keys:

```python
CASE_ROW_FIELDS = (
    "row_id",
    "case_id",
    "policy",
    "workload",
    "gpu_blocks",
    "blockwise_blocks",
    "repetition",
    "phase",
    "warmup",
    "source_sha256",
    "worker_pid",
    "tinyvllm_dist_port",
    "master_port",
    "cuda_visible_devices",
    "model_path",
    "python_path",
    "prompt_sha256",
    "decoded_token_ids",
    "decode_logits_path",
    "decode_logits_sha256",
    "decode_logits_shape",
    "decode_step_ms",
    "peak_cuda_allocated_bytes",
    "peak_cuda_reserved_bytes",
    "peak_resident_blocks",
    "kv_offload",
    "planner",
    "complete",
)
```

The remote runner CLI is:

```text
python tools/run_kv_decode_residency_planner_gate_remote.py
  preflight|smoke|canonical|local-worker|download-only|verify-only
  --run-tag RUN_TAG
```

The independent verifier CLI is:

```text
python tools/verify_kv_decode_residency_planner_gate.py
  --run-dir RUN_DIR
  --write-report
```

---

### Task 1: Immutable Directional Planner and Cache Identity

**Files:**
- Modify: `tinyvllm/layers/attention.py:1-280`
- Test: `tools/test_blockwise_attention_planning.py:15-380`

**Interfaces:**
- Consumes: existing `_normalize_logical_block_rows()`, `_unique_blocks_in_order()`, forward/reverse future-hint helpers, and context-local `kv_offload_decode_window_plan_cache`.
- Produces: `BlockwiseDecodePlanIdentity`, `BlockwiseDecodeWindow`, `BlockwiseDecodePlan`, `_build_blockwise_decode_plan_identity()`, `_ordered_unique_excluding()`, `_bounded_cross_layer_reuse_blocks()`, and side-by-side `_build_residency_aware_blockwise_decode_window_plan()` used by Task 2. The legacy runtime builder remains unchanged throughout Task 1.

- [ ] **Step 1: Write failing tests for directional cross-layer candidates and exact spare capacity**

Add imports and tests:

```python
from tinyvllm.layers.attention import (
    BlockwiseDecodePlan,
    _bounded_cross_layer_reuse_blocks,
    _build_residency_aware_blockwise_decode_window_plan,
)


def test_decode_plan_builds_forward_and_reverse_cross_layer_frontiers():
    plan = _build_residency_aware_blockwise_decode_window_plan(
        block_rows=[[0, 1, 2, 3, 4]],
        context_lens=[5],
        max_blocks=5,
        block_size=1,
        window_blocks=1,
        write_blocks=set(),
        gpu_blocks=3,
    )

    assert isinstance(plan, BlockwiseDecodePlan)
    assert [window.required_blocks for window in plan.forward_windows] == [
        (0,), (1,), (2,), (3,), (4,),
    ]
    assert [window.required_blocks for window in plan.reverse_windows] == [
        (4,), (3,), (2,), (1,), (0,),
    ]
    assert plan.forward_windows[-1].cross_layer_reuse_blocks == (3, 2)
    assert plan.reverse_windows[-1].cross_layer_reuse_blocks == (1, 2)


def test_cross_layer_reuse_is_stable_deduplicated_and_spare_bounded():
    assert _bounded_cross_layer_reuse_blocks(
        candidate_blocks=[3, 3, 2, 1, 0],
        required_blocks=(4,),
        write_blocks={0},
        gpu_blocks=4,
    ) == (3, 2)


def test_cross_layer_reuse_is_empty_without_spare_capacity():
    assert _bounded_cross_layer_reuse_blocks(
        candidate_blocks=[1, 2, 3],
        required_blocks=(0, 4),
        write_blocks={5},
        gpu_blocks=3,
    ) == ()
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD python3 - <<'PY'
from tools import test_blockwise_attention_planning as tests
tests.test_decode_plan_builds_forward_and_reverse_cross_layer_frontiers()
tests.test_cross_layer_reuse_is_stable_deduplicated_and_spare_bounded()
tests.test_cross_layer_reuse_is_empty_without_spare_capacity()
PY
```

Expected: import failure for `BlockwiseDecodePlan` or assertion failure because the current builder returns a mutable list without cross-layer records.

- [ ] **Step 3: Add immutable planner types and stable bounded helpers**

Add near the existing blockwise helpers:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class BlockwiseDecodePlanIdentity:
    block_rows: tuple[tuple[int, ...], ...]
    context_lens: tuple[int, ...]
    max_blocks: int
    block_size: int
    window_blocks: int
    write_blocks: tuple[int, ...]
    gpu_blocks: int


@dataclass(frozen=True)
class BlockwiseDecodeWindow:
    window_rows: tuple[tuple[int, ...], ...]
    window_lens: tuple[int, ...]
    required_blocks: tuple[int, ...]
    intra_layer_future_blocks: tuple[int, ...]
    cross_layer_reuse_blocks: tuple[int, ...]
    max_window_tokens: int


@dataclass(frozen=True)
class BlockwiseDecodePlan:
    identity: BlockwiseDecodePlanIdentity
    forward_windows: tuple[BlockwiseDecodeWindow, ...]
    reverse_windows: tuple[BlockwiseDecodeWindow, ...]


def _build_blockwise_decode_plan_identity(
    block_rows,
    context_lens,
    max_blocks,
    block_size,
    window_blocks,
    write_blocks,
    gpu_blocks,
):
    return BlockwiseDecodePlanIdentity(
        block_rows=tuple(tuple(int(block) for block in row) for row in block_rows),
        context_lens=tuple(int(length) for length in context_lens),
        max_blocks=int(max_blocks),
        block_size=int(block_size),
        window_blocks=int(window_blocks),
        write_blocks=tuple(sorted(int(block) for block in write_blocks)),
        gpu_blocks=int(gpu_blocks),
    )


def _ordered_unique_excluding(blocks, excluded):
    ordered = []
    seen = set(int(block) for block in excluded)
    for block in blocks:
        block = int(block)
        if block < 0 or block in seen:
            continue
        ordered.append(block)
        seen.add(block)
    return tuple(ordered)


def _bounded_cross_layer_reuse_blocks(
    *,
    candidate_blocks,
    required_blocks,
    write_blocks,
    gpu_blocks,
):
    hard_blocks = set(int(block) for block in required_blocks)
    hard_blocks.update(int(block) for block in write_blocks)
    spare_capacity = max(0, int(gpu_blocks) - len(hard_blocks))
    if spare_capacity == 0:
        return ()
    return _ordered_unique_excluding(
        candidate_blocks,
        hard_blocks,
    )[:spare_capacity]
```

- [ ] **Step 4: Add one bidirectional plan builder beside the legacy builder**

Implement `_build_residency_aware_blockwise_decode_window_plan()` so it:

1. builds raw windows once, including the exact current
   `future_hint_blocks` and `reverse_future_hint_blocks` sets produced by
   `_blockwise_read_window_future_hint_blocks()` and
   `_blockwise_read_window_reverse_future_hint_blocks()`;
2. creates forward records in ascending order;
3. creates reverse records from the same raw windows in descending order;
4. copies the current direction's existing same-layer future set without
   changing its membership;
5. derives cross-layer candidates from the opposite direction after the shared boundary record;
6. bounds only cross-layer candidates by exact spare capacity.

Use this complete directional-record helper:

```python
def _flatten_required_blocks(windows):
    return [
        block
        for window in windows
        for block in window["required_blocks"]
    ]


def _materialize_decode_direction(
    raw_windows,
    *,
    opposite_raw_windows,
    future_key,
    write_blocks,
    gpu_blocks,
):
    records = []
    for index, raw_window in enumerate(raw_windows):
        required_blocks = raw_window["required_blocks"]
        hard_blocks = set(required_blocks) | set(write_blocks)
        opposite_index = next(
            candidate_index
            for candidate_index, candidate in enumerate(opposite_raw_windows)
            if candidate["start_block"] == raw_window["start_block"]
        )
        cross_candidates = _flatten_required_blocks(
            opposite_raw_windows[opposite_index + 1:]
        )
        intra_layer_future_blocks = tuple(sorted(
            set(raw_window[future_key]) - hard_blocks
        ))
        cross_layer_reuse_blocks = _bounded_cross_layer_reuse_blocks(
            candidate_blocks=cross_candidates,
            required_blocks=required_blocks,
            write_blocks=write_blocks,
            gpu_blocks=gpu_blocks,
        )
        records.append(BlockwiseDecodeWindow(
            window_rows=raw_window["window_rows"],
            window_lens=raw_window["window_lens"],
            required_blocks=required_blocks,
            intra_layer_future_blocks=intra_layer_future_blocks,
            cross_layer_reuse_blocks=cross_layer_reuse_blocks,
            max_window_tokens=raw_window["max_window_tokens"],
        ))
    return tuple(records)
```

The returned plan must be:

```python
identity = _build_blockwise_decode_plan_identity(
    block_rows,
    context_lens,
    max_blocks,
    block_size,
    window_blocks,
    write_blocks,
    gpu_blocks,
)
forward_raw = tuple(raw_windows)
reverse_raw = tuple(reversed(raw_windows))
return BlockwiseDecodePlan(
    identity=identity,
    forward_windows=_materialize_decode_direction(
        forward_raw,
        opposite_raw_windows=reverse_raw,
        future_key="future_hint_blocks",
        write_blocks=write_blocks,
        gpu_blocks=gpu_blocks,
    ),
    reverse_windows=_materialize_decode_direction(
        reverse_raw,
        opposite_raw_windows=forward_raw,
        future_key="reverse_future_hint_blocks",
        write_blocks=write_blocks,
        gpu_blocks=gpu_blocks,
    ),
)
```

Before materialization, each `raw_window` must contain:

```python
{
    "start_block": int(start_block),
    "window_rows": tuple(tuple(row) for row in window_rows),
    "window_lens": tuple(int(length) for length in window_lens),
    "required_blocks": tuple(_unique_blocks_in_order(needed_blocks)),
    "future_hint_blocks": frozenset(future_hint_blocks),
    "reverse_future_hint_blocks": frozenset(
        reverse_future_hint_blocks
    ),
    "max_window_tokens": int(max(window_lens)),
}
```

The two frozen future sets must be computed with the current helper calls
before any cross-layer logic. This prevents the candidate from changing
same-layer eviction behavior under the label of a cross-layer experiment.

- [ ] **Step 5: Run focused planner tests and existing planning suite**

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
PYTHONPATH=$PWD python3 tools/test_blockwise_attention_planning.py
```

Expected: the new direct planner tests and every existing planning test pass.
The runtime still calls the unchanged legacy builder, so Task 1 must not create
a red intermediate commit.

- [ ] **Step 6: Commit the planner data model**

```bash
git add tinyvllm/layers/attention.py tools/test_blockwise_attention_planning.py
git diff --cached --check
git commit -m "refactor: model bidirectional decode window plans"
```

---

### Task 2: Decode Integration, Cache Invalidation, and Hint Isolation

**Files:**
- Modify: `tinyvllm/layers/attention.py:120-500`
- Modify: `tinyvllm/engine/model_runner.py:170-590`
- Test: `tools/test_blockwise_attention_planning.py:90-540`

**Interfaces:**
- Consumes: Task 1's immutable planner types and side-by-side residency-aware builder.
- Produces: exact cache identity validation, forward/reverse record selection, soft future-hint union, and planner diagnostic counters exposed by `KVOffloadMVP0.summary()`.

- [ ] **Step 1: Write failing cache identity tests**

Add:

```python
def test_decode_plan_exact_identity_reuses_cache():
    manager, context, q, k_cache, v_cache = _decode_fixture(
        block_rows=[[0, 1, 2]],
        context_lens=[3],
        gpu_blocks=3,
        window_blocks=1,
    )
    _blockwise_online_decode_attention(
        q, k_cache, v_cache, context, 1, 1, 1.0, layer_idx=0,
    )
    cached_plan = context.kv_offload_decode_window_plan_cache
    _blockwise_online_decode_attention(
        q, k_cache, v_cache, context, 1, 1, 1.0, layer_idx=1,
    )
    assert context.kv_offload_decode_window_plan_cache is cached_plan
    assert manager.stats["decode_plan_builds"] == 1
    assert manager.stats["decode_plan_cache_hits"] == 1


def test_decode_plan_identity_change_rebuilds_cache():
    manager, context, q, k_cache, v_cache = _decode_fixture(
        block_rows=[[0, 1, 2]],
        context_lens=[3],
        gpu_blocks=3,
        window_blocks=1,
    )
    _blockwise_online_decode_attention(
        q, k_cache, v_cache, context, 1, 1, 1.0, layer_idx=0,
    )
    first_plan = context.kv_offload_decode_window_plan_cache
    context.kv_offload_context_lens = [2]
    _blockwise_online_decode_attention(
        q, k_cache, v_cache, context, 1, 1, 1.0, layer_idx=1,
    )
    assert context.kv_offload_decode_window_plan_cache is not first_plan
    assert manager.stats["decode_plan_builds"] == 2
    assert manager.stats["decode_plan_identity_invalidations"] == 1
```

Implement `_decode_fixture()` once in the test file with a `_PlanOnlyManager`
whose `stats` includes all seven planner counters initialized to zero. For
staging-isolation tests, initialize `pending_wait_blocks=set()` so membership
can only arise from the tested call.

- [ ] **Step 2: Write failing staging-isolation and zero-spare tests**

Add:

```python
def test_cross_layer_hints_are_future_only():
    manager, context, q, k_cache, v_cache = _decode_fixture(
        block_rows=[[0, 1, 2, 3]],
        context_lens=[4],
        gpu_blocks=3,
        window_blocks=1,
        write_blocks=[7],
    )
    _blockwise_online_decode_attention(
        q, k_cache, v_cache, context, 1, 1, 1.0, layer_idx=0,
    )
    assert manager.ensure_calls[-1] == [3]
    assert 2 in manager.future_calls[-1]
    assert 2 not in manager.protected_calls[-1]
    assert manager.wait_calls[-1] == ([3], True)
    assert 2 not in manager.pending_wait_blocks


def test_zero_spare_capacity_matches_existing_alternating_future_sets():
    manager, context, q, k_cache, v_cache = _decode_fixture(
        block_rows=[[0, 1, 2]],
        context_lens=[3],
        gpu_blocks=1,
        window_blocks=1,
    )
    _blockwise_online_decode_attention(
        q, k_cache, v_cache, context, 1, 1, 1.0, layer_idx=1,
    )
    assert manager.ensure_calls == [[2], [1], [0]]
    assert manager.future_calls == [{2}, {1}, {0}]
    assert manager.stats["decode_cross_layer_hint_blocks"] == 0
```

- [ ] **Step 3: Run the new tests and verify RED**

Run only the four new tests through a small Python importer. Expected failures:

- unkeyed cache reuse after identity mutation;
- missing planner counters;
- dataclass records not consumed;
- cross-layer hints absent from the future set.

- [ ] **Step 4: Initialize planner counters in the real manager**

Add to `KVOffloadMVP0.stats`:

```python
"decode_plan_builds": 0,
"decode_plan_cache_hits": 0,
"decode_plan_identity_invalidations": 0,
"decode_windows_with_spare_capacity": 0,
"decode_cross_layer_hint_blocks": 0,
"decode_cross_layer_hint_resident": 0,
"decode_cross_layer_hint_retained": 0,
```

Keep `summary()`'s existing `dict(self.stats)` behavior so these counters are
automatically included. Do not change eviction or copy logic in this task.

- [ ] **Step 5: Validate cache identity before reuse**

First replace the legacy `_build_blockwise_decode_window_plan()` body with
Task 1's residency-aware implementation and remove the temporary
`_build_residency_aware_blockwise_decode_window_plan()` name. Then replace the
current `if plan_cache is None` block with:

```python
plan_identity = _build_blockwise_decode_plan_identity(
    block_rows,
    context_lens,
    max_blocks,
    block_size,
    window_blocks,
    write_blocks,
    manager.gpu_blocks,
)
plan_cache = getattr(
    context,
    "kv_offload_decode_window_plan_cache",
    None,
)
if plan_cache is None or plan_cache.identity != plan_identity:
    if plan_cache is not None:
        manager.stats["decode_plan_identity_invalidations"] += 1
    plan_cache = _build_blockwise_decode_window_plan(
        block_rows,
        context_lens,
        max_blocks,
        block_size,
        window_blocks,
        write_blocks,
        manager.gpu_blocks,
    )
    context.kv_offload_decode_window_plan_cache = plan_cache
    manager.stats["decode_plan_builds"] += 1
else:
    manager.stats["decode_plan_cache_hits"] += 1
```

- [ ] **Step 6: Consume directional records and pass soft hints only**

Use:

```python
reverse_windows = int(layer_idx) >= 0 and int(layer_idx) % 2 == 1
window_plans = (
    plan_cache.reverse_windows
    if reverse_windows
    else plan_cache.forward_windows
)
for window_plan in window_plans:
    required_blocks = window_plan.required_blocks
    future_hint_blocks = set(window_plan.intra_layer_future_blocks)
    future_hint_blocks.update(window_plan.cross_layer_reuse_blocks)
    future_hint_blocks.update(write_blocks)
    resident_before = set(manager.logical_to_slot)
    retained_candidates = (
        set(window_plan.cross_layer_reuse_blocks) & resident_before
    )
    if window_plan.cross_layer_reuse_blocks:
        manager.stats["decode_windows_with_spare_capacity"] += 1
        manager.stats["decode_cross_layer_hint_blocks"] += len(
            window_plan.cross_layer_reuse_blocks
        )
        manager.stats["decode_cross_layer_hint_resident"] += len(
            retained_candidates
        )
    _stage_blockwise_read_window(
        manager,
        required_blocks,
        future_extra_blocks=future_hint_blocks,
        protected_extra_blocks=write_blocks,
        capacity_extra_blocks=write_blocks,
        capacity_error_prefix="blockwise decode staging capacity exceeded",
    )
    manager.stats["decode_cross_layer_hint_retained"] += len(
        retained_candidates & set(manager.logical_to_slot)
    )
```

Continue using `window_plan.window_rows`, `window_plan.window_lens`, and
`window_plan.max_window_tokens` for the existing gather/mask/math path.

- [ ] **Step 7: Preserve resident fast-path isolation**

Add a test where required block `0` is resident, hinted-only block `2` is also
resident, and the window takes `_stage_blockwise_read_window()`'s fast path.
Assert only required block `0` receives `_touch()`. Do not change the fast path
to touch hinted-only blocks.

- [ ] **Step 8: Run focused and full local planning tests**

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
PYTHONPATH=$PWD python3 tools/test_blockwise_attention_planning.py

PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
python3 -m py_compile \
  tinyvllm/layers/attention.py \
  tinyvllm/engine/model_runner.py \
  tools/test_blockwise_attention_planning.py

git diff --check
```

Expected: all planning tests pass and compilation/diff checks exit zero.

- [ ] **Step 9: Commit decode integration**

```bash
git add \
  tinyvllm/layers/attention.py \
  tinyvllm/engine/model_runner.py \
  tools/test_blockwise_attention_planning.py
git diff --cached --check
git commit -m "feat: add decode cross-layer residency hints"
```

---

### Task 3: Manager Semantics and Simulated Movement Regression

**Files:**
- Modify: `tools/test_kv_offload.py:1-470`
- Modify: `tools/test_blockwise_attention_planning.py:450-960`
- Modify only if a test proves a defect: `tinyvllm/engine/model_runner.py:320-490`

**Interfaces:**
- Consumes: Task 2's use of `future_logical_blocks`.
- Produces: proof that hints affect only victim scoring and a deterministic multi-layer simulation showing required-window equivalence plus non-worsening movement.

- [ ] **Step 1: Add a manager test proving a future-only resident block is favored but not protected**

```python
def test_future_only_block_biases_victim_score_without_becoming_protected():
    manager = _RecordingKVOffload()
    manager.evict_policy = "lru_cost"
    manager.slot_last_used = [0, 1, 2, 3]
    manager.cpu_valid[4] = True

    mapping = manager.ensure_resident(
        [4],
        require_valid=True,
        future_logical_blocks={0},
        protected_logical_blocks=set(),
        wait=False,
    )

    assert mapping == {4: 1}
    assert 0 in manager.logical_to_slot
    assert 4 in manager.logical_to_slot
    assert manager.h2d_pairs == [[(4, 1)]]
    assert manager.pending_wait_blocks == set()
```

The explicit recency values make block `0` the oldest. Its future penalty must
preserve it, causing unhinted block `1` in slot `1` to be evicted.

- [ ] **Step 2: Add a test proving future-only missing blocks are not loaded or waited**

```python
def test_future_only_missing_blocks_are_not_loaded_pending_or_waited():
    manager = _RecordingKVOffload()
    manager.cpu_valid[4] = True
    manager.cpu_valid[5] = True

    mapping = manager.ensure_resident(
        [4],
        require_valid=True,
        future_logical_blocks={5},
        protected_logical_blocks=set(),
        wait=False,
    )

    assert mapping == {4: manager.logical_to_slot[4]}
    assert 5 not in manager.logical_to_slot
    assert all(logical != 5 for batch in manager.h2d_pairs for logical, _ in batch)
    assert 5 not in manager.pending_wait_blocks
```

- [ ] **Step 3: Add a deterministic planner/manager simulation**

In `tools/test_blockwise_attention_planning.py`, add a CPU-only
`_SimulatedResidencyManager` implementing:

```python
class _SimulatedResidencyManager:
    def __init__(self, gpu_blocks):
        self.gpu_blocks = int(gpu_blocks)
        self.logical_to_slot = {}
        self.slot_to_logical = [None] * self.gpu_blocks
        self.slot_last_used = [0] * self.gpu_blocks
        self.pending_wait_blocks = set()
        self.clock = 0
        self.required_trace = []
        self.stats = {
            "h2d_copies": 0,
            "evictions": 0,
            "prefetch_plans": 0,
            "prefetch_read_blocks": 0,
            "prefetch_write_blocks": 0,
            "decode_plan_builds": 0,
            "decode_plan_cache_hits": 0,
            "decode_plan_identity_invalidations": 0,
            "decode_windows_with_spare_capacity": 0,
            "decode_cross_layer_hint_blocks": 0,
            "decode_cross_layer_hint_resident": 0,
            "decode_cross_layer_hint_retained": 0,
        }

    def mark_dirty(self, blocks):
        return None

    def _touch(self, slot):
        self.clock += 1
        self.slot_last_used[int(slot)] = self.clock

    def ensure_resident(
        self,
        logical_blocks,
        require_valid,
        future_logical_blocks=None,
        protected_logical_blocks=None,
    ):
        required = _unique_blocks_in_order(logical_blocks)
        self.required_trace.append(tuple(required))
        protected = set(required) | set(protected_logical_blocks or ())
        future = set(future_logical_blocks or ())
        for logical in required:
            if logical in self.logical_to_slot:
                self._touch(self.logical_to_slot[logical])
                continue
            candidates = [
                slot
                for slot, resident in enumerate(self.slot_to_logical)
                if resident not in protected
            ]
            free = next(
                (slot for slot, resident in enumerate(self.slot_to_logical)
                 if resident is None),
                None,
            )
            slot = free
            if slot is None:
                slot = min(
                    candidates,
                    key=lambda item: (
                        self.slot_to_logical[item] in future,
                        self.slot_last_used[item],
                    ),
                )
                old = self.slot_to_logical[slot]
                del self.logical_to_slot[old]
                self.stats["evictions"] += 1
            self.logical_to_slot[logical] = slot
            self.slot_to_logical[slot] = logical
            self.stats["h2d_copies"] += 1
            self._touch(slot)
        return {
            logical: self.logical_to_slot[logical]
            for logical in required
        }

    def wait_for_blocks(self, logical_blocks, clear_pending=False):
        return None
```

Compare the candidate planner with a helper that strips
`cross_layer_reuse_blocks` from the same immutable plan. Run at least four
layers over `[0, 1, 2, 3, 4, 5]`, with `gpu_blocks=2` and
`window_blocks=1`. Assert:

```python
assert candidate.required_trace == baseline.required_trace
assert candidate.stats["h2d_copies"] <= baseline.stats["h2d_copies"]
assert candidate.stats["evictions"] <= baseline.stats["evictions"]
```

- [ ] **Step 4: Run tests and verify RED or expose no-op behavior**

Run both full scripts. If the deterministic simulation shows identical
movement rather than improvement, keep the non-worsening assertion and record
that CPU simulation only proves semantics; do not tune the algorithm solely to
make the synthetic test improve.

- [ ] **Step 5: Fix only manager defects demonstrated by tests**

Expected implementation outcome is no manager code change. If a future-only
block is accidentally loaded, protected, or added to pending waits, fix the
caller in `attention.py`, not `ensure_resident()`. Change `model_runner.py`
only if its existing public contract violates a direct manager unit test.

- [ ] **Step 6: Run the complete focused regression set**

```bash
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
PYTHONPATH=$PWD python3 tools/test_kv_offload.py

PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
PYTHONPATH=$PWD python3 tools/test_blockwise_attention_planning.py

PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
PYTHONPATH=$PWD python3 tools/test_chunked_prefill.py

PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
PYTHONPATH=$PWD python3 tools/test_ngram_speculative.py
```

Expected: all four scripts print their success messages and exit zero.

- [ ] **Step 7: Commit manager and simulation coverage**

```bash
git add tools/test_kv_offload.py tools/test_blockwise_attention_planning.py
git add tinyvllm/engine/model_runner.py  # only if Step 5 required a real fix
git diff --cached --check
git commit -m "test: cover decode residency hint semantics"
```

---

### Task 4: Frozen Gate Contract and Independent Verifier

**Files:**
- Create: `tools/kv_decode_residency_planner_contract.py`
- Create: `tools/verify_kv_decode_residency_planner_gate.py`
- Create: `tools/test_kv_decode_residency_planner_gate.py`

**Interfaces:**
- Consumes: the shared contract constants and raw case schema defined above.
- Produces: deterministic case matrix, canonical hashes, exact required files, ratio calculations, tamper rejection, and independent `GO | NO_GO | INVALID`.

- [ ] **Step 1: Write failing contract-domain tests**

Create `tools/test_kv_decode_residency_planner_gate.py` with:

```python
def test_canonical_matrix_is_closed_and_complete():
    matrix = contract.build_case_matrix()
    assert len(matrix) == (
        len(contract.STAGING_SHAPES)
        * len(contract.WORKLOADS)
        * len(contract.POLICIES)
        * (
            contract.WARMUP_REPETITIONS
            + contract.CORRECTNESS_REPETITIONS
            + contract.MEASURED_REPETITIONS
        )
    )
    assert len({case.case_id for case in matrix}) == len(matrix)
    assert {
        (case.gpu_blocks, case.blockwise_blocks)
        for case in matrix
    } == set(contract.STAGING_SHAPES)


def test_classification_requires_real_movement_improvement():
    ratios = _passing_ratio_fixture()
    ratios["h2d_improvement"] = 0.0
    ratios["eviction_improvement"] = 0.0
    assert contract.classify_ratios(ratios) == "NO_GO"


def test_classification_rejects_other_metric_regression():
    ratios = _passing_ratio_fixture()
    ratios["h2d_improvement"] = 0.06
    ratios["eviction_regression"] = 0.02
    assert contract.classify_ratios(ratios) == "NO_GO"
```

- [ ] **Step 2: Implement the frozen contract**

Define:

```python
@dataclass(frozen=True)
class GateCase:
    workload: str
    policy: str
    gpu_blocks: int
    blockwise_blocks: int
    repetition: int
    phase: str
    warmup: bool

    @property
    def pair_id(self):
        return (
            f"{self.workload}__g{self.gpu_blocks}"
            f"__w{self.blockwise_blocks}"
            f"__{self.phase}__r{self.repetition}"
        )

    @property
    def case_id(self):
        return f"{self.pair_id}__{self.policy}"
```

`build_case_matrix()` must use:

```text
warmup phase:      repetition 0
correctness phase: repetition 0
measured phase:    repetitions 0..4
```

The phase is part of every `pair_id`, so equal repetition numbers cannot
collide. `warmup` is true only for phase `warmup`.

```python
REQUIRED_FILES = (
    "manifest.json",
    "environment.json",
    "source_manifest.json",
    "worker_logs_manifest.json",
    "case_rows.jsonl",
    "summary.json",
    "report.md",
    "independent_verification.json",
)
```

Implement canonical JSON bytes using sorted keys and compact separators.
Implement `classify_ratios()` with explicit boolean checks for every design
gate; return `INVALID` only when passed an explicit `valid=False`.

Freeze the movement aggregation rules:

```text
aggregate H2D or aggregate evictions improves >= 5%
at least one gpu_blocks=2 case improves H2D or evictions >= 5%
aggregate multi_prompt_thrash H2D or evictions improves >= 5%
every workload/shape pair keeps the non-winning H2D/eviction metric
  within <= 1% regression
```

For every measured workload/shape pair, also require:

```text
candidate copy_waits <= baseline copy_waits
candidate prefetch_plans <= baseline prefetch_plans
candidate d2h_copies <= baseline d2h_copies
candidate d2h_bytes <= baseline d2h_bytes
candidate evict_dirty <= baseline evict_dirty
candidate peak_resident_blocks <= baseline peak_resident_blocks
candidate peak_cuda_allocated_bytes <= baseline peak_cuda_allocated_bytes
candidate peak_cuda_reserved_bytes <= baseline peak_cuda_reserved_bytes
candidate median_decode_step_ms <= baseline * 1.02
```

If both baseline and candidate movement counts are zero, improvement is `0.0`.
If baseline is zero and candidate is positive, regression is infinite and the
gate fails.

- [ ] **Step 3: Write verifier tamper and coverage tests**

Generate a temporary complete run directory, then assert verifier failures for:

- one missing case ID;
- one duplicate row ID;
- one unexpected extra row;
- source SHA mismatch;
- equal baseline/candidate port values;
- non-`0` GPU;
- missing decoded token;
- token mismatch;
- missing or hash-mismatched correctness logits tensor;
- logits outside `rtol=1e-3`, `atol=1e-2`;
- missing KV counter;
- non-finite decode latency;
- summary/raw disagreement;
- multi-prompt case without `>=5%` H2D or eviction improvement;
- low-capacity domain without `>=5%` improvement.

Use direct function calls, not subprocess-only assertions, so failures name the
rejected invariant.

- [ ] **Step 4: Implement independent verification**

The verifier must:

1. load only the contract module and raw artifact files;
2. reconstruct the exact expected case domain;
3. index unique `row_id` and `case_id`;
4. reject missing, duplicate, or extra cases;
5. verify immutable source and environment fields;
6. compare token lists exactly;
7. require logits artifacts only for phase `correctness`; load each CPU tensor
   with `torch.load(path, map_location="cpu", weights_only=True)`, verify its
   SHA-256 and exact recorded shape, and compare every logits element with:

```python
abs(candidate - baseline) <= (
    contract.LOGIT_ATOL
    + contract.LOGIT_RTOL * abs(baseline)
)
```

8. require no logits artifact for warmup or measured phases;
9. compare exact token lists in correctness and all five measured repetitions;
10. sum integer counters over identical decoded-token counts from measured
    repetitions only;
11. compute latency medians from the five measured repetitions only;
12. compute peak memory and resident-block maxima from measured repetitions;
13. evaluate every staging shape independently;
14. require one low-capacity shape and `multi_prompt_thrash` movement wins;
15. write `independent_verification.json` atomically;
16. write `report.md` from recomputed values;
17. verify every stdout/stderr log named by `worker_logs_manifest.json`
    exists and matches its recorded SHA-256;
18. exit nonzero on `INVALID`, but allow zero exit for valid `NO_GO`.

- [ ] **Step 5: Run contract and verifier tests**

```bash
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
PYTHONPATH=$PWD python3 tools/test_kv_decode_residency_planner_gate.py

python3 -m py_compile \
  tools/kv_decode_residency_planner_contract.py \
  tools/verify_kv_decode_residency_planner_gate.py \
  tools/test_kv_decode_residency_planner_gate.py

git diff --check
```

Expected: all tamper cases are rejected and the valid passing/no-go fixtures
receive the expected classifications.

- [ ] **Step 6: Commit the frozen evidence contract**

```bash
git add \
  tools/kv_decode_residency_planner_contract.py \
  tools/verify_kv_decode_residency_planner_gate.py \
  tools/test_kv_decode_residency_planner_gate.py
git diff --cached --check
git commit -m "test: define decode residency planner gate"
```

---

### Task 5: Profiler Evidence and Remote Orchestration

**Files:**
- Modify: `tools/profile_ngram_commit.py:230-285,1320-1410,2030-2120`
- Modify: `tools/smoke_blockwise_prefill_remote.sh:80-125`
- Create: `tools/run_kv_decode_residency_planner_gate_remote.py`
- Modify: `tools/test_kv_decode_residency_planner_gate.py`

**Interfaces:**
- Consumes: Task 4's matrix/schema/verifier and existing `profile_ngram_commit.py --mode baseline-only`.
- Produces: complete source-bound raw rows for paired baseline/candidate snapshots, strict logits records, decode-only timing, memory peaks, planner counters, and dynamic remote commands.

- [ ] **Step 1: Add failing profiler-schema tests**

Use a fake LLM/runner fixture to assert `run_baseline_only_profile()` emits:

```text
per_prompt[*].token_ids
decode_logits_path when --record-decode-logits
decode_logits_sha256 when --record-decode-logits
decode_logits_shape when --record-decode-logits
summary.decode_step_ms
summary.peak_cuda_allocated_bytes
summary.peak_cuda_reserved_bytes
summary.peak_resident_blocks
planner
```

Add CLI flag:

```text
--record-decode-logits
```

defaulting to `False`, so existing profiler output and cost remain unchanged.

- [ ] **Step 2: Capture decode evidence without changing model results**

Add these exact `ModelRunner` interfaces:

```python
def enable_step_logits_recording(self, enabled: bool) -> None:
    self._record_step_logits = bool(enabled)
    self._last_step_logits_cpu = None


def last_step_logits(self) -> torch.Tensor | None:
    if self._last_step_logits_cpu is None:
        return None
    return self._last_step_logits_cpu.clone()
```

Initialize both private fields in `ModelRunner.__init__`. In
`ModelRunner.run()`, after `_select_sample_rows()` and before calling the
sampler:

```python
if self._record_step_logits:
    self._last_step_logits_cpu = logits.detach().float().cpu()
else:
    self._last_step_logits_cpu = None
```

For non-sampling or non-rank-zero paths, set `_last_step_logits_cpu=None`.
Recording is default-off and is enabled only by `profile_ngram_commit.py` when
`--record-decode-logits` is present. The CPU transfer is intentionally inside
the measured model step and is identical in baseline and candidate snapshots.
Do not store a live GPU logits tensor across steps.

In the baseline-only loop:

- reset CUDA peak memory stats immediately before measured execution;
- call `llm.model_runner.enable_step_logits_recording(
  args.record_decode_logits)` before measured execution;
- after each decode step, read
  `llm.model_runner.last_step_logits()`;
- append one CPU logits tensor per decode step in stable sequence order;
- at process end, concatenate to one CPU `float32` tensor and save atomically
  with `torch.save()` to `Path(args.decode_logits_out + ".partial")`, then
  rename it to `Path(args.decode_logits_out)`;
- record only path, SHA-256, and shape in the case row;
- do not serialize logits into JSON;
- compute `decode_step_ms` from records where `num_tokens < 0`;
- derive peak resident blocks from `kv_offload_summary()["resident_blocks"]`
  sampled after every step;
- expose `planner` as the seven counters selected from the final KV summary.

The accessor must return a clone of the CPU tensor and must not mutate sampling
or runtime state.

The runner passes `--record-decode-logits` and `--decode-logits-out` only for
phase `correctness`. Warmup and five measured performance repetitions keep
recording disabled, so logits CPU transfer and serialization cannot affect the
latency gate.

- [ ] **Step 3: Extend smoke preflight only**

Add to `run_preflight()`:

```bash
"${PYTHON_BIN}" tools/test_blockwise_attention_planning.py
"${PYTHON_BIN}" tools/test_kv_offload.py
```

Keep the script's default GPU/model values untouched; canonical execution will
override them explicitly.

- [ ] **Step 4: Write failing remote command-discipline tests**

Assert `build_worker_command()`:

- uses `sitian@10.232.195.203`;
- passes `CUDA_VISIBLE_DEVICES=0`;
- uses the approved Python/model paths;
- rejects equal ports;
- includes distinct dynamic port values;
- includes exact `gpu_blocks` and `blockwise_blocks`;
- includes `--record-decode-logits` and `--decode-logits-out` only for
  correctness-phase workers;
- never contains `rsync`, `pkill`, `kill`, another GPU, or remote checkout
  mutation;
- retries only stderr containing `EADDRINUSE`.

- [ ] **Step 5: Implement the remote runner using the existing exact-CUDA runner patterns**

Reuse the proven mechanisms, without importing its feature contract:

```python
SSH_TARGET = "sitian@10.232.195.203"
SSH_CONTROL_PATH = "/tmp/ssh-sitian-10.232.195.203"
REMOTE_PYTHON = "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
REMOTE_MODEL = (
    "/data00/home/sitian/sitian-workspace01/.ms_cache/"
    "Qwen/Qwen3-0___6B"
)
CUDA_VISIBLE_DEVICES = "0"
OUTPUT_ROOT = ROOT / "experiments" / "kv_offload"
```

The runner must:

1. reject tracked-tree dirtiness outside ignored raw experiment directories;
2. compute a deterministic tracked-source tree SHA;
3. create `staging/source` using a tar stream over an explicit owned-source
   manifest, never `rsync`;
4. upload that immutable snapshot to a run-specific remote directory;
5. allocate two distinct local ephemeral ports per worker;
6. execute one process per case row;
7. retry at most three times and only for `EADDRINUSE`;
8. alternate baseline/candidate order by repetition;
9. use separate baseline and candidate staged source directories;
10. store command, PID, ports, source SHA, environment, and raw profiler JSON;
11. download with tar into a `.partial` directory and atomically rename;
12. invoke the local independent verifier after download.

Build both runtime snapshots from the candidate tracked tree so profiler,
manager diagnostics, runner support, and evidence schema are byte-identical.
For the baseline snapshot only, replace
`tinyvllm/layers/attention.py` with that file from approved baseline commit
`94056ba`. Candidate keeps the current implementation file. Record both the
synthetic staged-tree SHA and the originating git commit.

The runner must prove that the two staged runtime snapshots differ only in:

```text
tinyvllm/layers/attention.py
```

All tools and every other runtime file, including
`tinyvllm/engine/model_runner.py`, are staged identically for both policies.
This preserves current `94056ba` planner behavior as the baseline while giving
both policies identical read-only logits, planner-counter, memory, and artifact
instrumentation. Reject execution if:

```bash
git show 94056ba:tinyvllm/layers/attention.py
```

fails, or if any second file differs.

- [ ] **Step 6: Freeze workload commands**

Use:

```text
single_long_context:
  prompt_count=1
  max_num_seqs=1
  prompt_repeat=64
  max_model_len=4096
  max_output_len=16

multi_prompt_thrash:
  prompt_count=2
  max_num_seqs=2
  prompt_repeat=40
  max_model_len=2048
  max_output_len=16
```

Both use:

```text
--mode baseline-only
--temperature 0.0
--gpu-memory-utilization 0.7
--max-num-prefill-tokens-per-step 256
--kv-offload-mvp0
--kv-offload-logical-blocks 8
--kv-offload-blockwise-prefill
--kv-offload-blockwise-decode
```

The case shape supplies `--kv-offload-gpu-blocks` and
`--kv-offload-blockwise-blocks`. Correctness-phase workers additionally use:

```text
--record-decode-logits
--decode-logits-out "$CASE_OUTPUT_DIR/decode_logits.pt"
```

`CASE_OUTPUT_DIR` is the runner's unique directory for the current `case_id`.

- [ ] **Step 7: Run local contract/preflight tests**

```bash
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
PYTHONPATH=$PWD python3 tools/test_kv_decode_residency_planner_gate.py

PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
PYTHONPATH=$PWD python3 tools/test_blockwise_attention_planning.py

PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
PYTHONPATH=$PWD python3 tools/test_kv_offload.py

python3 -m py_compile \
  tools/profile_ngram_commit.py \
  tools/run_kv_decode_residency_planner_gate_remote.py

git diff --check
```

- [ ] **Step 8: Commit profiler and runner**

```bash
git add \
  tools/profile_ngram_commit.py \
  tools/smoke_blockwise_prefill_remote.sh \
  tools/run_kv_decode_residency_planner_gate_remote.py \
  tools/test_kv_decode_residency_planner_gate.py
git diff --cached --check
git commit -m "feat: add decode residency planner remote gate"
```

---

### Task 6: Remote Smoke, Canonical Gate, and Independent Classification

**Files:**
- Raw output only: `experiments/kv_offload/$RUN_TAG/`
- Modify only for proven defects: files owned by Tasks 1-5

**Interfaces:**
- Consumes: committed implementation, frozen contract, remote runner, and verifier.
- Produces: source-bound smoke and canonical evidence with an independent final classification.

- [ ] **Step 1: Verify local source state before remote work**

Run:

```bash
git status --short --branch
git log -6 --oneline --decorate
git diff --check
```

Expected: only pre-existing untracked experiment directories are present; all
planner/gate work is committed.

- [ ] **Step 2: Run remote preflight**

```bash
PYTHONPATH=$PWD python3 \
  tools/run_kv_decode_residency_planner_gate_remote.py \
  preflight \
  --run-tag qwen3-06b-kv-residency-preflight-$(date +%Y%m%d-%H%M%S)
```

Required remote tests:

```text
tools/test_blockwise_attention_planning.py
tools/test_kv_offload.py
tools/test_chunked_prefill.py
tools/test_ngram_speculative.py
```

Expected: all exit zero under approved remote Python and GPU environment.

- [ ] **Step 3: Run a reduced paired smoke**

```bash
PYTHONPATH=$PWD python3 \
  tools/run_kv_decode_residency_planner_gate_remote.py \
  smoke \
  --run-tag qwen3-06b-kv-residency-smoke-$(date +%Y%m%d-%H%M%S)
```

Smoke includes:

```text
workloads: single_long_context, multi_prompt_thrash
shapes: (2,1), (4,2)
warmup: 1
correctness repetitions: 1
measured repetitions: 1
policies: baseline, candidate
```

Expected:

- exact tokens;
- logits within tolerance;
- all required counters present;
- candidate planner hints exercised in at least one case;
- no source, command, port, or artifact-domain failure.

Do not interpret smoke latency as a performance conclusion.

- [ ] **Step 4: Diagnose only evidence-backed failures**

If smoke fails:

- use `systematic-debugging`;
- classify failure as implementation correctness, evidence schema, remote
  environment, or `EADDRINUSE`;
- retry automatically only for `EADDRINUSE`;
- write a failing regression test before code changes;
- rerun local focused tests and the reduced smoke;
- commit each proven fix separately.

Do not relax thresholds, remove shapes, shorten correctness output, or change
workloads based on observed candidate performance.

- [ ] **Step 5: Run the full canonical matrix**

```bash
CANONICAL_RUN_TAG="qwen3-06b-kv-residency-canonical-$(date +%Y%m%d-%H%M%S)"
printf '%s\n' "$CANONICAL_RUN_TAG" \
  > /private/tmp/tinyllmforge_kv_residency_canonical_run_tag
PYTHONPATH=$PWD python3 \
  tools/run_kv_decode_residency_planner_gate_remote.py \
  canonical \
  --run-tag "$CANONICAL_RUN_TAG"
```

Expected domain:

```text
2 workloads
× 4 staging shapes
× 2 policies
× (1 warmup + 1 correctness + 5 measured)
= 112 process rows
```

Every row must use a unique PID-scoped command record and distinct
`TINYVLLM_DIST_PORT`/`MASTER_PORT`.

- [ ] **Step 6: Rerun independent verification from disk**

```bash
CANONICAL_RUN_TAG="$(
  cat /private/tmp/tinyllmforge_kv_residency_canonical_run_tag
)"
PYTHONPATH=$PWD python3 \
  tools/verify_kv_decode_residency_planner_gate.py \
  --run-dir "experiments/kv_offload/$CANONICAL_RUN_TAG" \
  --write-report
```

Inspect:

```text
independent_verification.json
report.md
manifest.json
source_manifest.json
case_rows.jsonl
```

Confirm the verifier independently recomputed all ratios and did not trust a
runner-provided classification.

- [ ] **Step 7: Apply the classification without overclaiming**

- `GO`: keep implementation; a performance claim may quote only independently
  recomputed canonical metrics.
- `NO_GO`: correctness/evidence are valid but the strict movement/regression
  gate failed. Report the negative result. Keep the implementation only after
  explicit user approval; otherwise revert implementation commits while
  preserving design, plan, tests/tooling useful for evidence, and the raw
  artifact.
- `INVALID`: fix evidence/correctness coverage and rerun; make no performance
  conclusion.

No classification permits mixing in another optimization before this gate is
closed.

---

### Task 7: Handoff, Registry, and Claim Discipline

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify if `GO`: `README.md`
- Create: `docs/kv_offload_evidence_registry.json`

**Interfaces:**
- Consumes: canonical raw artifact and independent classification.
- Produces: durable implementation/evidence handoff and, only after `GO`, a narrowly scoped user-facing claim.

- [ ] **Step 1: Write the handoff entry from raw evidence**

Append a dated section containing:

```text
design spec path and commit
implementation plan path and commit
implementation commits
baseline source SHA and candidate source SHA
canonical run directory
approved remote host/user/GPU/Python/model
workload and staging-shape matrix
correctness result
H2D, eviction, wait, prefetch, D2H, dirty-writeback, memory, and latency ratios
planner diagnostic counters
independent classification
what the result proves
what it does not prove
negative branches and any reverted attempts
best next step
```

State explicitly that KV offload remains default-off and that this planner does
not change prefill or proactively prefetch blocks.

- [ ] **Step 2: Create the test/evidence registry**

Create `docs/kv_offload_evidence_registry.json` with exactly one top-level
object and one entry:

```json
{
  "schema_version": 1,
  "entries": [
    {
      "id": "decode_residency_aware_read_window_planner",
      "spec": "docs/superpowers/specs/2026-07-22-decode-residency-aware-read-window-planner-design.md",
      "plan": "docs/superpowers/plans/2026-07-22-decode-residency-aware-read-window-planner-implementation.md",
      "canonical_run": "experiments/kv_offload/$CANONICAL_RUN_TAG",
      "verification": "experiments/kv_offload/$CANONICAL_RUN_TAG/independent_verification.json",
      "classification": "GO"
    }
  ]
}
```

Replace `$CANONICAL_RUN_TAG` and `GO` with the actual run-tag string and actual
`GO | NO_GO | INVALID` classification before writing JSON. Validate the file by
loading it with Python's `json` module and asserting the exact key sets shown
above.

- [ ] **Step 3: Update README only on independent GO**

For `GO`, add one concise experimental bullet with:

- exact model/GPU/workload scope;
- H2D or eviction improvement;
- latency bound/result;
- default-off status;
- canonical artifact path.

For `NO_GO` or `INVALID`, do not modify README.

- [ ] **Step 4: Run final local verification**

```bash
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
PYTHONPATH=$PWD python3 tools/test_blockwise_attention_planning.py

PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
PYTHONPATH=$PWD python3 tools/test_kv_offload.py

PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
PYTHONPATH=$PWD python3 tools/test_chunked_prefill.py

PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
PYTHONPATH=$PWD python3 tools/test_ngram_speculative.py

PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
PYTHONPATH=$PWD python3 tools/test_kv_decode_residency_planner_gate.py

python3 -m py_compile \
  tinyvllm/layers/attention.py \
  tinyvllm/engine/model_runner.py \
  tools/profile_ngram_commit.py \
  tools/kv_decode_residency_planner_contract.py \
  tools/run_kv_decode_residency_planner_gate_remote.py \
  tools/verify_kv_decode_residency_planner_gate.py

git diff --check
git status --short --branch
```

Expected: all tests pass; only intended documentation/registry files are
tracked changes; raw `experiments/` remain untracked.

- [ ] **Step 5: Commit durable conclusions with exact staging**

For all classifications:

```bash
git add AGENT_HANDOFF_STATE.md
git add docs/kv_offload_evidence_registry.json
git diff --cached --check
git commit -m "docs: record decode residency planner gate"
```

For `GO` only, include `README.md` in that exact staged set.

- [ ] **Step 6: Perform the completion audit**

Build a prompt-to-artifact checklist covering:

- decode-only scope;
- immutable planner identity;
- both traversal directions;
- spare-capacity bounds;
- no hint load/protection/wait/pending behavior;
- zero-capacity equivalence;
- local regression coverage;
- approved remote host/user/GPU/Python/model;
- dynamic distinct ports;
- four staging shapes;
- single- and multi-prompt workloads;
- five measured repetitions;
- exact tokens and logits tolerance;
- every movement/copy/memory/latency threshold;
- source-bound raw evidence;
- independent classification;
- handoff/registry landing;
- README claim discipline.

Inspect the actual files and raw outputs for every item. Treat uncertainty as
incomplete. Do not claim completion from test success or a summary label alone.

---

## Execution Checkpoints

Pause for review after:

1. Task 3, when planner semantics and all focused local regressions pass;
2. Task 5, when the immutable remote gate is committed but before GPU runs;
3. Task 6 smoke, before launching the 112-row canonical matrix;
4. Task 6 classification, before deciding whether a `NO_GO` implementation is
   kept or reverted.

The user has already selected Inline Execution. After this plan is approved,
invoke `superpowers:executing-plans`, execute in the current session, and do
not spawn subagents.
