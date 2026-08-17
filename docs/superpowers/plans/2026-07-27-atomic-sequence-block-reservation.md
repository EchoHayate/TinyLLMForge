# Atomic Sequence Block Reservation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Privately reserve and atomically attach a complete request KV block table containing an exact reusable prefix plus newly allocated suffix blocks.

**Architecture:** Extend `BlockManager` with a one-request reservation that acquires prefix references and suffix blocks without changing `Sequence`. Validate the complete reservation before transferring ownership to request metadata; release all held references on miss, failure, or explicit rollback.

**Tech Stack:** Python 3.9, dataclasses, existing BlockManager and Sequence, dependency-light test harness.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not stage, commit, merge, or delete untracked experiment evidence.
- Do not enable scheduler hybrid prefix reuse in this phase.
- Do not move HybridStateTensorPool or snapshot tensors into Scheduler.
- Do not modify ModelRunner, Qwen3 math, attention, RoPE, RMSNorm, or checkpoint semantics.
- Preserve existing `BlockManager.allocate()` behavior.
- Apply the existing sampleable-token prefix cap.
- Exact token comparison is mandatory in addition to chained hashes.
- Do not mutate request metadata before complete reservation attachment.
- Roll back every prefix and suffix reference on failure.
- Preserve the immutable Qwen3.5 schema-v2 canonical `NO_GO`.
- Do not claim latency, throughput, hit rate, compression, quality, or physical-memory improvement.

---

### Task 1: RED Complete Sequence Reservation

**Files:**
- Modify: `tools/test_chunked_prefill.py`
- Modify after RED: `tinyvllm/engine/block_manager.py`

**Interfaces:**
- Produces: `SequenceBlockReservation`.
- Produces: `BlockManager.reserve_sequence_blocks()`.
- Produces: `BlockManager.attach_sequence_reservation()`.
- Produces: `BlockManager.release_sequence_reservation()`.

- [x] Add cold one-block and multi-block tests proving reservation does not
  mutate sequence metadata.
- [x] Add warm live/idle prefix tests comparing reserved block table and
  cached token count with existing `allocate()`.
- [x] Add exact block-aligned and sampleable-token-cap tests.
- [x] Add collision and first-prefix-miss tests.
- [x] Add insufficient-capacity read-only failure test.
- [x] Inject a later suffix allocation failure and prove all prefix/suffix
  ownership is restored.
- [x] Add attachment and release ownership-transfer tests.
- [x] Add stale identity, dirty request, malformed reservation, and repeated
  terminal-operation tests.
- [x] Run focused tests and confirm RED because the complete reservation APIs
  do not exist.

### Task 2: GREEN Reservation and Rollback

**Files:**
- Modify: `tinyvllm/engine/block_manager.py`
- Modify: `tools/test_chunked_prefill.py`

**Interfaces:**

```python
@dataclass
class SequenceBlockReservation:
    block_ids: tuple[int, ...]
    block_identities: tuple[tuple[int, int, int], ...]
    cached_tokens: int
    prefix_block_count: int
    new_block_count: int
    state: str = "reserved"
```

```python
def reserve_sequence_blocks(
    self,
    seq: Sequence,
    *,
    max_cached_tokens: Optional[int] = None,
) -> SequenceBlockReservation
```

```python
def attach_sequence_reservation(
    self,
    reservation: SequenceBlockReservation,
    seq: Sequence,
) -> None
```

```python
def release_sequence_reservation(
    self,
    reservation: SequenceBlockReservation,
) -> None
```

- [x] Validate pristine request state and cache cap before mutation.
- [x] Discover the exact reusable prefix chain without ownership changes.
- [x] Fail before mutation when total free-block capacity is insufficient.
- [x] Acquire one reference for each reusable prefix block.
- [x] Allocate every suffix block through the existing new-content path.
- [x] Roll back partial prefix/suffix ownership on any exception.
- [x] Record complete ordered table and exact prefix identities.
- [x] Validate identities and counts before attachment.
- [x] Transfer ownership without refcount changes.
- [x] Release exactly one reference per reserved block and reject repeated
  terminal operations.
- [x] Run focused tests and confirm GREEN.

### Task 3: Compatibility and Completion Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

- [x] Run all sequence-reservation focused tests.
- [x] Run the full Python 3.12 zero-argument
  `tools/test_chunked_prefill.py` matrix with only the documented Config AST
  skip.
- [x] Run Qwen3.5 hybrid prefix acquisition/cache and hybrid-state regression.
- [x] Run Python 3.9 and Python 3.12 `py_compile`.
- [x] Run `git diff --check`.
- [x] Confirm `git diff --cached --name-only` is empty.
- [x] Build a prompt-to-artifact checklist covering cold/warm reservation,
  exact identity, sampleable cap, complete-table atomicity, failure rollback,
  compatibility, CPU-only scope, and no performance overclaim.
- [x] Record the scheduler/ModelRunner cross-process restore ticket as the next
  gate in `AGENT_HANDOFF_STATE.md`.
- [x] Mark checkboxes complete only from fresh evidence.
