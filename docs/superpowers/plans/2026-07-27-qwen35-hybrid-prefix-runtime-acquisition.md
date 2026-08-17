# Qwen3.5 Hybrid Prefix Runtime Acquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make exact KV prefix reservation and Qwen3.5 hybrid-state snapshot restoration one failure-atomic CPU acquisition.

**Architecture:** Keep `BlockManager` as the sole KV owner, add content-lifetime block generations and temporary exact-prefix reservations, then coordinate those reservations with real hybrid-state leases and the existing snapshot cache. Commit request-visible KV and state metadata only after state restore succeeds.

**Tech Stack:** Python 3.9, PyTorch CPU, dataclasses, existing BlockManager, HybridStateSlotAllocator, HybridStateTensorPool, and Qwen35HybridPrefixSnapshotCache.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not stage, commit, merge, or delete untracked experiment evidence.
- Keep `Scheduler._allocate_request_storage()` fail-closed for hybrid prefix hits.
- Do not connect ModelRunner, GPU KV tensors, or a checkpoint in this phase.
- Do not modify Qwen3 RMSNorm, RoPE, attention, or checkpoint semantics.
- Increment block generation only when physical storage is assigned to new content.
- Exact token comparison is mandatory in addition to chained hashes.
- Reserve all matching prefix blocks before allocating state.
- Do not expose request KV or hybrid-state metadata before successful restore.
- Release every newly acquired resource on miss or exception.
- A successful acquisition attaches only the reusable prefix, not uncached suffix blocks.
- Preserve the immutable Qwen3.5 schema-v2 canonical `NO_GO`.
- Do not claim latency, throughput, cache, compression, quality, or physical-memory improvement.

---

### Task 1: RED Block Generation and Reservation Contract

**Files:**
- Modify: `tools/test_chunked_prefill.py`
- Modify after RED: `tinyvllm/engine/block_manager.py`

**Interfaces:**
- Produces: `Block.generation`.
- Produces: `PrefixBlockReservation`.
- Produces: `BlockManager.reserve_exact_prefix()`.
- Produces: `BlockManager.attach_prefix_reservation()`.
- Produces: `BlockManager.release_prefix_reservation()`.

- [x] Add a test proving first new-content allocation changes generation from
  `0` to `1`, deallocation preserves it, and idle exact reuse preserves it.
- [x] Add a test proving reuse for different tokens increments generation and
  invalidates the old exact identity.
- [x] Add exact one-block and multi-block reservation tests for live and idle
  blocks with `owner_count > 1`.
- [x] Add a partial-chain miss test proving free lists, used sets, refcounts,
  generations, and request metadata are unchanged.
- [x] Add release tests proving every temporary reference is removed exactly
  once.
- [x] Add one- and multi-destination attachment tests proving ownership
  transfer does not change refcounts.
- [x] Add validation tests for non-tuple/non-aligned tokens, invalid owner
  count, duplicate destinations, dirty destinations, and double
  release/attach.
- [x] Run the focused script and confirm RED because generation and reservation
  APIs do not exist.

### Task 2: GREEN Block Generation and Reservation Ownership

**Files:**
- Modify: `tinyvllm/engine/block_manager.py`
- Modify: `tools/test_chunked_prefill.py`

**Interfaces:**

```python
@dataclass
class PrefixBlockReservation:
    block_ids: tuple[int, ...]
    block_identities: tuple[tuple[int, int, int], ...]
    token_count: int
    owner_count: int
    state: str = "reserved"
```

```python
def reserve_exact_prefix(
    self,
    token_ids: tuple[int, ...],
    *,
    owner_count: int = 1,
) -> Optional[PrefixBlockReservation]
```

```python
def attach_prefix_reservation(
    self,
    reservation: PrefixBlockReservation,
    sequences: tuple[Sequence, ...],
) -> None
```

```python
def release_prefix_reservation(
    self,
    reservation: PrefixBlockReservation,
) -> None
```

- [x] Add `generation = 0` to `Block` and increment it only in the
  new-content reset path.
- [x] Split idle exact-cache activation from new-content allocation so exact
  reactivation preserves hash, tokens, and generation.
- [x] Validate an entire exact chained-token prefix before mutating ownership.
- [x] Acquire `owner_count` references per block and record identities after
  all blocks are held.
- [x] Roll back already held references if reservation acquisition raises.
- [x] Prevalidate every destination before attachment.
- [x] Transfer reservation ownership to destinations without changing
  refcounts.
- [x] Release only still-reserved ownership and reject repeated terminal
  operations.
- [x] Run the focused tests and confirm GREEN.
- [x] Run all existing `tools/test_chunked_prefill.py` tests and confirm no
  prefix-cache lifecycle regression.

### Task 3: RED Atomic Hybrid Prefix Coordinator

**Files:**
- Create: `tools/test_qwen35_hybrid_prefix_acquisition.py`
- Create after RED: `tinyvllm/engine/qwen35_hybrid_prefix_acquisition.py`

**Interfaces:**
- Consumes: Task 2 reservation APIs.
- Consumes: `HybridStateSlotAllocator`, `HybridStateTensorPool`, and
  `Qwen35HybridPrefixSnapshotCache`.
- Produces: fixtures covering real block, allocator, pool, adapter,
  transaction, and snapshot-cache interactions.

- [x] Build a real two-layer CPU fixture and publish one exact source
  snapshot.
- [x] Test one-destination exact acquisition attaches the prefix, writes lease
  metadata, and restores every state component.
- [x] Test one source snapshot broadcasts to multiple destination requests
  while block refcounts equal the destination count.
- [x] Test absent KV returns `False` before state allocation.
- [x] Test stale generation and ordinary snapshot miss release every KV and
  state resource.
- [x] Test allocator exhaustion after KV reservation restores original KV
  ownership.
- [x] Inject a later pool activation failure and prove prior activations,
  leases, and KV reservations are released.
- [x] Inject a later snapshot restore failure and prove all request metadata
  remains pristine.
- [x] Test constructor, duplicate request, dirty request, key token-count, and
  block-size validation before mutation.
- [x] Test wrong-prefix and too-short destination prompts fail before
  mutation.
- [x] Run the focused script and confirm RED because the coordinator module
  does not exist.

### Task 4: GREEN Atomic Hybrid Prefix Coordinator

**Files:**
- Create: `tinyvllm/engine/qwen35_hybrid_prefix_acquisition.py`
- Modify: `tools/test_qwen35_hybrid_prefix_acquisition.py`

**Interfaces:**

```python
class Qwen35HybridPrefixAcquireCoordinator:
    def __init__(
        self,
        block_manager: BlockManager,
        state_allocator: HybridStateSlotAllocator,
        state_pool: HybridStateTensorPool,
        snapshot_cache: Qwen35HybridPrefixSnapshotCache,
    )

    def acquire(
        self,
        sequences: tuple[Sequence, ...],
        key: Qwen35HybridPrefixKey,
        token_ids: tuple[int, ...],
    ) -> bool
```

- [x] Validate constructor ownership and request/key invariants before
  resource mutation.
- [x] Reserve one exact KV prefix for every destination owner.
- [x] Allocate and activate one state lease per request.
- [x] Restore all destination rows using reservation block identities.
- [x] On normal miss, release pool bindings, allocator leases, and KV
  reservation and return `False`.
- [x] On exception, perform the same cleanup and re-raise the original error.
- [x] After successful restore, attach the KV reservation and lease metadata
  in one final commit section.
- [x] Prove existing `BlockManager.deallocate()` releases successfully
  transferred request ownership.
- [x] Run focused tests and confirm GREEN.

### Task 5: Regression and Completion Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

- [x] Run:

```text
tools/test_qwen35_hybrid_prefix_acquisition.py
tools/test_qwen35_hybrid_prefix_cache.py
tools/test_qwen35_cross_layer_state_transaction.py
tools/test_qwen35_layer_state_adapter.py
tools/test_hybrid_state.py
tools/test_hybrid_state_sequence.py
tools/test_hybrid_state_scheduler.py
tools/test_hybrid_state_runtime_bridge.py
tools/test_chunked_prefill.py
```

- [x] Run the established broader Qwen3.5 packed-layer CPU regression.
- [x] Run Python 3.9 and Python 3.12 `py_compile` with a temporary pycache
  prefix.
- [x] Run `git diff --check`.
- [x] Confirm `git diff --cached --name-only` is empty.
- [x] Build a prompt-to-artifact checklist covering generation lifetime,
  exact reservation, multi-owner refcounts, failure rollback, state restore,
  request-visible atomicity, scheduler fail-closed behavior, CPU-only scope,
  and performance-claim boundaries.
- [x] Record fresh commands, results, what they prove, and remaining
  scheduler/ModelRunner/GPU gates in `AGENT_HANDOFF_STATE.md`.
- [x] Mark checkboxes complete only from fresh evidence.
