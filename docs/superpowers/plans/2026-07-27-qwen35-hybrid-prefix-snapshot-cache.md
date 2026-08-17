# Qwen3.5 Hybrid Prefix Snapshot Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded, atomic CPU cache that binds exact full-KV block identities to Qwen3.5 cross-layer convolution/recurrent state snapshots.

**Architecture:** Keep `BlockManager` as the KV owner. Add a separate exact-token, model/layout-bound LRU snapshot index that gathers state only after successful prefix completion and restores all linear layers through the existing cross-layer transaction. Stale KV block generations, identity mismatches, malformed state, and partial restore failures all fail closed.

**Tech Stack:** Python 3.9, PyTorch CPU, dataclasses, OrderedDict, existing HybridStateTensorPool and Qwen35CrossLayerStateTransaction.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not stage, commit, merge, or delete untracked experiment evidence.
- Do not connect scheduler, ModelRunner, GPU KV tensors, or a checkpoint in this phase.
- Do not modify Qwen3 RMSNorm, RoPE, attention, or checkpoint semantics.
- Cache only positive full-block token boundaries.
- Exact token comparison is mandatory in addition to hashes.
- Model, layout, TP, dtype, block size, and block generation must match exactly.
- Publication and restoration must be all-or-nothing.
- Enforce both `max_entries` and `max_bytes` with deterministic LRU.
- Do not claim latency, throughput, hit-rate, compression, quality, or physical-memory improvement.
- Preserve the immutable Qwen3.5 schema-v2 canonical `NO_GO`.

---

### Task 1: RED Cache Identity and Publication Tests

**Files:**
- Create: `tools/test_qwen35_hybrid_prefix_cache.py`
- Create after RED: `tinyvllm/engine/qwen35_hybrid_prefix_cache.py`

**Interfaces:**
- Uses: `HybridStateTensorPool`, `Qwen35LayerStateAdapter`, and `Qwen35CrossLayerStateTransaction`.
- Produces fixtures reused by Tasks 2 and 3.

- [x] Build a two-linear-layer, four-slot real-pool fixture with one source and
  three distinct destination rows.
- [x] Define valid keys for two exact block-aligned prefixes and explicit block identities.
- [x] Test successful publication records exact tokens, ordered cloned state, and exact storage bytes.
- [x] Mutate gathered/source tensors and prove the published entry remains unchanged.
- [x] Test invalid token count, block alignment, hash, fingerprint, TP, dtype, and block identity values fail before publication.
- [x] Test a gather/validation failure leaves a previously published same-key entry intact.
- [x] Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_qwen35_hybrid_prefix_cache.py
```

Expected RED:

```text
FileNotFoundError:
tinyvllm/engine/qwen35_hybrid_prefix_cache.py
```

### Task 2: GREEN Identity, Publication, and Exact Acquisition

**Files:**
- Create: `tinyvllm/engine/qwen35_hybrid_prefix_cache.py`
- Modify: `tools/test_qwen35_hybrid_prefix_cache.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class Qwen35HybridPrefixKey:
    token_hash: int
    token_count: int
    terminal_block_hash: int
    block_size: int
    model_fingerprint: str
    layout_fingerprint: str
    tensor_parallel_size: int
    dtype: torch.dtype
```

```python
class Qwen35HybridPrefixSnapshotCache:
    def publish(
        self,
        key: Qwen35HybridPrefixKey,
        token_ids: tuple[int, ...],
        block_identities: tuple[tuple[int, int, int], ...],
        lease: HybridStateLease,
    ) -> bool

    def acquire(
        self,
        key: Qwen35HybridPrefixKey,
        token_ids: tuple[int, ...],
        block_identities: tuple[tuple[int, int, int], ...],
        leases: tuple[HybridStateLease, ...],
    ) -> bool
```

- [x] Validate constructor limits and key scalar/type invariants.
- [x] Validate exact token length and full-block boundary.
- [x] Validate block identities are non-empty, ordered, unique by block ID, and contain non-negative generation plus valid hash.
- [x] Gather one source row from every adapter and clone every tensor before
  index mutation.
- [x] Compute storage bytes from owned tensor storage.
- [x] Publish or atomically replace only after complete validation.
- [x] Add exact acquire tests that broadcast one cached row into three
  out-of-slot destination leases across all layers.
- [x] Add miss tests for token collision, model/layout/TP/dtype/block-size mismatch, and stale block generation/hash.
- [x] Restore through one cross-layer transaction commit and refresh LRU only after success.
- [x] Confirm focused GREEN.

### Task 3: Failure Atomicity and Bounded LRU

**Files:**
- Modify: `tinyvllm/engine/qwen35_hybrid_prefix_cache.py`
- Modify: `tools/test_qwen35_hybrid_prefix_cache.py`

**Interfaces:**

```python
def invalidate_blocks(
    self,
    block_identities: tuple[tuple[int, int, int], ...],
) -> int

def clear(self) -> int

def observation_snapshot(self) -> dict[str, int]
```

- [x] Corrupt a stored later-layer candidate and prove validation fails before destination writes.
- [x] Inject a later cross-layer copy failure and prove every layer/request destination row is restored.
- [x] Prove failed acquire does not refresh LRU recency.
- [x] Implement deterministic `OrderedDict` LRU.
- [x] Test same-key replacement updates bytes without double counting.
- [x] Test entry-count eviction.
- [x] Test byte-budget eviction.
- [x] Test an individually oversize entry is not retained.
- [x] Test explicit block invalidation removes every dependent entry.
- [x] Test lazy stale-identity removal during acquire.
- [x] Test clear resets current entries/bytes but keeps cumulative event counters.
- [x] Test FP32 and BF16 storage and restore.
- [x] Verify every observation counter against direct events.
- [x] Confirm focused GREEN.

### Task 4: Regression and Completion Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

- [x] Run the focused cache test.
- [x] Run:

```text
tools/test_qwen35_cross_layer_state_transaction.py
tools/test_qwen35_packed_full_decoder_layer.py
tools/test_qwen35_packed_stateful_linear_decoder_layer.py
tools/test_qwen35_packed_layer_stack.py
tools/test_qwen35_layer_state_adapter.py
tools/test_qwen35_stateful_linear_decoder_layer.py
tools/test_qwen35_hybrid_state_layout.py
tools/test_hybrid_state.py
tools/test_hybrid_state_sequence.py
tools/test_hybrid_state_scheduler.py
tools/test_hybrid_state_runtime_bridge.py
tools/test_model_runner_spec_verify.py
```

- [x] Run Python 3.9 `py_compile` with:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge-pycache
```

- [x] Run Python 3.12 TP/loader compatibility tests and compile.
- [x] Run `git diff --check`.
- [x] Confirm `git diff --cached --name-only` is empty.
- [x] Build a prompt-to-artifact checklist covering exact identity, clone isolation, atomic publish, atomic restore, stale-block rejection, LRU budgets, byte accounting, CPU-only execution, and no performance overclaim.
- [x] Record the proof and the scheduler/runtime/GPU gates that remain in `AGENT_HANDOFF_STATE.md`.
- [x] Mark all plan checkboxes complete only from fresh evidence.
