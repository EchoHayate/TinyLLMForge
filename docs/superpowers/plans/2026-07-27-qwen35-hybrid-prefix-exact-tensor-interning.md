# Qwen3.5 Hybrid Prefix Exact Tensor Interning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deduplicate byte-identical immutable Qwen3.5 hybrid prefix snapshot tensors while preserving exact restore semantics.

**Architecture:** Add a cache-local SHA-256 content index whose buckets are guarded by exact contiguous logical-byte checks. Snapshot entries reference canonical detached contiguous clones, lifecycle operations maintain reference counts, and byte limits charge only unique canonical storage while separately reporting logical referenced bytes.

**Tech Stack:** Python 3.9, PyTorch CPU tests, `hashlib.sha256`, dataclasses, `OrderedDict`, existing `Qwen35CrossLayerStateTransaction`.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; do not use subagents.
- Do not stage, commit, merge, or delete untracked experiment evidence.
- Do not modify scheduler, Engine, ModelRunner, publication wiring, block ownership, model math, or checkpoint loading.
- Share only tensors with identical dtype, shape, device, digest, and logical bytes.
- Preserve source clone isolation and transaction rollback.
- Base `max_bytes` and `current_bytes` on unique physical snapshot storage.
- Preserve logical referenced byte accounting separately.
- Do not claim production memory, CUDA allocator, latency, throughput, or quality improvement.
- Preserve the canonical Qwen3.5 schema-v2 `NO_GO`.

---

### Task 1: RED Exact Sharing and Accounting

**Files:**
- Modify: `tools/test_qwen35_hybrid_prefix_cache.py`

**Interfaces:**
- Consumes: existing `_fixture()`, `publish()`, `acquire()`, `invalidate_blocks()`, `clear()`, and `observation_snapshot()`.
- Produces: failing behavioral tests for canonical sharing and unique-byte accounting.

- [x] Add a helper that returns ordered cached snapshots and a test publishing
  two different prefix identities from the same lease.
- [x] Assert both entries survive, corresponding tensors are the same objects
  with equal `data_ptr()`, `current_bytes` equals one 112-byte FP32 snapshot,
  `current_logical_bytes` equals 224, and `deduplicated_bytes` equals 112.
- [x] Mutate the live source after both publications and assert cached values
  remain unchanged.
- [x] Invalidate one entry and assert physical bytes remain 112; invalidate the
  final entry and assert all current physical/logical/ref counters are zero.
- [x] Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_qwen35_hybrid_prefix_cache.py
```

Expected RED: the second publication owns separate tensors and reports 224
physical bytes or the new observation keys are absent.

### Task 2: GREEN Intern Table and Reference Lifecycle

**Files:**
- Modify: `tinyvllm/engine/qwen35_hybrid_prefix_cache.py`

**Interfaces:**
- Produces private `_TensorInternKey`, `_InternedTensor`,
  `_tensor_digest()`, candidate grouping, intern acquisition, and release.
- Preserves the public cache method signatures.

- [x] Add SHA-256 over a detached contiguous `uint8` CPU view and bind dtype,
  shape, and device metadata in `_TensorInternKey`.
- [x] Group exact-equal candidate clones before publication.
- [x] Resolve each group against an intern bucket with exact contiguous
  `uint8` byte-view equality.
- [x] Count one new canonical allocation as a miss and every reused occurrence
  as a hit.
- [x] Store canonical references in the snapshot and increment one ref per
  tensor occurrence.
- [x] Change `_remove_entry()` and `clear()` to release refs and free canonical
  storage only after the final ref.
- [x] Track physical bytes, logical bytes, intern tensor count, total refs,
  deduplicated bytes, and peak logical bytes.
- [x] Run the focused suite and confirm GREEN.

### Task 3: RED/GREEN Partial Sharing, Replacement, and Collision Safety

**Files:**
- Modify: `tools/test_qwen35_hybrid_prefix_cache.py`
- Modify: `tinyvllm/engine/qwen35_hybrid_prefix_cache.py`

**Interfaces:**
- Consumes: private module `_tensor_digest` only for an injected collision test.
- Produces: exact collision handling and replacement rollback/refcount proof.

- [x] Add a partial-sharing test where one recurrent tensor differs and assert
  only that tensor adds physical bytes.
- [x] Add a replacement test that changes one tensor, then replaces it with the
  original content and checks physical/logical bytes and refs after each step.
- [x] Monkeypatch `_tensor_digest` to a constant, publish unequal snapshots,
  and assert unequal tensors do not share while `intern_collisions` increases.
- [x] Force the same digest for `+0.0` and `-0.0` tensors and assert their
  distinct bit patterns do not share storage.
- [x] Verify acquire from both collision entries exactly restores their own
  values.
- [x] Inject failure on the second intern acquisition and assert all acquired
  refs are rolled back while the previous entry remains restorable.
- [x] Count digest calls and assert each candidate tensor is hashed once per
  publication.
- [x] Run each new test before implementation and observe the expected failure.
- [x] Implement collision buckets and atomic intern rollback, then confirm the
  focused suite passes.

### Task 4: RED/GREEN Unique-Byte LRU Semantics

**Files:**
- Modify: `tools/test_qwen35_hybrid_prefix_cache.py`
- Modify: `tinyvllm/engine/qwen35_hybrid_prefix_cache.py`

**Interfaces:**
- Preserves `max_entries`, `max_bytes`, deterministic LRU, and publish return
  semantics.

- [x] Change the byte-budget regression so two identical entries fit in a
  one-snapshot physical budget.
- [x] Publish genuinely different state in the same budget and assert the
  oldest entry is evicted.
- [x] Add a candidate whose standalone unique footprint exceeds `max_bytes`
  and assert rejection preserves the previous same-key entry.
- [x] Observe RED, implement standalone unique-footprint checking and
  unique-byte limit enforcement, then confirm GREEN.

### Task 5: Regression, Documentation, and Objective Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

**Interfaces:**
- Produces fresh proof and explicit production claim boundaries.

- [x] Run the focused cache suite.
- [x] Run dependent suites:

```text
tools/test_qwen35_hybrid_prefix_acquisition.py
tools/test_qwen35_hybrid_prefix_restore_ticket.py
tools/test_qwen35_cross_layer_state_transaction.py
tools/test_qwen35_layer_state_adapter.py
tools/test_hybrid_state.py
```

- [x] Compile changed Python files with
  `PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge-pycache`.
- [x] Run `git diff --check`.
- [x] Confirm `git diff --cached --name-only` is empty.
- [x] Record exact FP32/BF16 logical bytes, unique bytes, deduplicated bytes,
  ref lifecycle, collision safety, and test commands in the handoff.
- [x] Build a prompt-to-artifact audit for no-loss semantics, cache reduction,
  production integration, speed, and remaining gates.
- [x] Mark checkboxes complete only from fresh command output.
