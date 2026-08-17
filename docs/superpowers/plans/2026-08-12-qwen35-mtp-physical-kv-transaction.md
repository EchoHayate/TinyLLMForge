# Qwen3.5 MTP Physical KV Transaction Implementation Plan

> **Execution mode:** Inline execution only. The active workspace constraints
> forbid subagents, staging, commits, branch changes, and worktrees.

**Goal:** Replace the Qwen3.5 MTP metadata-only proposal-slot store with real
device-resident K/V tensors, execute MTP attention against committed plus
staged slots, and make the real-checkpoint gate verify accepted-slot identity,
rejected-suffix release, and rollback-safe cached continuation.

**Architecture:** Keep the generic `ProposalKVCache` lifecycle source-neutral.
The Qwen3.5 adapter owns a block-size-one physical K/V store so arbitrary
accepted prefixes remain in their original slots without copy or replay. The
executor installs an MTP-local attention context around each bootstrap or
proposal forward; no MTP tensors or slot IDs cross the ModelRunner boundary.

**Tech Stack:** Python, PyTorch, pytest, existing TinyLLMForge attention
context and Qwen3.5 eager cached-attention helpers.

## Global Constraints

- TP1 only.
- KV offload disabled.
- One Qwen3.5 MTP layer.
- Greedy proposals only.
- Shared target embedding and LM head.
- Distinct exact-Q families; no Q padding, rounding, or merging.
- Accepted MTP K/V remains in-place; rejected suffix slots are released.
- No replay, copy, or per-token rematerialization of accepted MTP K/V.
- Generic Engine, Scheduler, verifier, target-KV, and residency code remains
  source-neutral.
- Overall classification remains `NOT_PROMOTABLE`.

---

### Task 1: Device-Resident Qwen3.5 MTP Slot Store

**Files:**
- Modify: `tinyvllm/engine/qwen35_mtp_registration.py`
- Modify: `tinyvllm/engine/proposal_kv_cache.py`
- Create: `tools/test_qwen35_mtp_physical_kv.py`

**Interfaces:**
- Produces:
  - `Qwen35MTPPhysicalSlotStore(..., num_kv_heads, head_dim, dtype, device)`
  - `bind_attention_backend(backend) -> None`
  - `slot_identity(slot_id) -> tuple[int, int]`
  - `is_allocated(slot_id) -> bool`
  - `ProposalKVCache.physical_store`
- Preserves:
  - `reserve_slots(count) -> tuple[int, ...]`
  - `release_slots(slot_ids) -> None`

- [x] **Step 1: Write failing physical-tensor tests**

Add tests that require:

```python
store = Qwen35MTPPhysicalSlotStore(
    capacity=8,
    num_kv_heads=2,
    head_dim=4,
    dtype=torch.float32,
    device="cpu",
)
store.bind_attention_backend(backend)
slots = store.reserve_slots(3)
identity = store.slot_identity(slots[0])
backend.k_cache[slots[0], 0].fill_(7)
store.release_slots(slots[1:])

assert backend.k_cache is store.key_cache
assert backend.v_cache is store.value_cache
assert store.slot_identity(slots[0]) == identity
assert store.is_allocated(slots[0]) is True
assert store.is_allocated(slots[1]) is False
```

Also assert release clears rejected K/V storage and that a committed slot's
data pointer and contents do not change across `ProposalKVCache.commit_finalize`.

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_qwen35_mtp_physical_kv.py -q
```

Expected: collection or assertion failure because the store has no tensors,
binding API, identity API, or cache property.

- [x] **Step 3: Implement the minimal physical store**

Allocate:

```python
self.key_cache = torch.zeros(
    capacity, 1, num_kv_heads, head_dim,
    dtype=dtype, device=device,
)
self.value_cache = torch.zeros_like(self.key_cache)
```

Use block size one so every logical token maps directly to one physical block
ID. `release_slots()` must clear only released rows before returning them to
the free list. `slot_identity()` must use the real K/V row data pointers.

- [x] **Step 4: Run focused GREEN tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_qwen35_mtp_physical_kv.py \
  tools/test_proposal_kv_cache.py -q
```

Expected: all tests pass.

---

### Task 2: MTP Attention Context Uses Committed and Staged Slots

**Files:**
- Modify: `tinyvllm/engine/qwen35_mtp_executor.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_qwen35_mtp_executor.py`
- Modify: `tools/test_qwen35_mtp_model_runner_integration.py`

**Interfaces:**
- Consumes:
  - `ProposalKVCache.physical_store`
  - transaction `staged_slot_ids`
  - `ProposalKVCache.committed_slot_ids(sequence_id)`
- Produces:
  - bootstrap prefill context with exact staged slot mapping;
  - decode context with committed slots plus the staged prefix through the
    current step;
  - `runner.qwen35_mtp_physical_store` for gate inspection.

- [x] **Step 1: Write failing context tests**

Use a context-recording fake MTP module. For bootstrap require:

```python
context.mode == "prefill"
context.slot_mapping.tolist() == list(transaction.staged_slot_ids)
context.cu_seqlens_q.tolist() == [0, prompt_length]
```

For proposal step `i`, require:

```python
context.mode == "decode"
context.slot_mapping.tolist() == [transaction.staged_slot_ids[i]]
context.block_tables[0].tolist() == (
    list(committed_slots)
    + list(transaction.staged_slot_ids[:i + 1])
)
context.context_lens.tolist() == [
    len(committed_slots) + i + 1
]
```

After accepting `k`, start another proposal and assert its first block table
contains the original committed prefix plus exactly `max(k - 1, 0)` accepted
proposal slots, with no rejected slot IDs.

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_qwen35_mtp_executor.py \
  tools/test_qwen35_mtp_model_runner_integration.py -q
```

Expected: context assertions fail because executor forwards currently inherit
the unrelated global target context and registration does not retain or bind
the physical store.

- [x] **Step 3: Implement bootstrap and decode context helpers**

Add executor-local helpers using `temporary_context`:

```python
with temporary_context(
    mode="prefill",
    is_prefill=True,
    slot_mapping=slot_mapping,
    cu_seqlens_q=offsets,
    cu_seqlens_k=offsets,
    max_seqlen_q=token_count,
    max_seqlen_k=token_count,
    block_tables=None,
):
    return self.module.forward_step(...)
```

and:

```python
visible_slots = committed_slots + staged_slots[:step + 1]
with temporary_context(
    mode="decode",
    is_prefill=False,
    slot_mapping=current_slot,
    block_tables=visible_slots_tensor.unsqueeze(0),
    context_lens=torch.tensor([len(visible_slots)], ...),
):
    return self.module.forward_step(...)
```

All context tensors must be on the same device as the hidden state. Context
must restore even when the MTP forward raises.

- [x] **Step 4: Bind store to the real MTP attention backend**

After CPU checkpoint binding and target-device movement:

```python
physical_store = dependencies.build_physical_store(config, module)
physical_store.bind_attention_backend(
    module.layer.decoder_layer.full_attention.attention_backend
)
self.qwen35_mtp_physical_store = physical_store
```

Derive capacity from the already-resolved target visible KV token capacity:

```python
capacity = config.num_kvcache_blocks * config.kvcache_block_size
```

- [x] **Step 5: Run focused GREEN tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_qwen35_mtp_physical_kv.py \
  tools/test_proposal_kv_cache.py \
  tools/test_qwen35_mtp_executor.py \
  tools/test_qwen35_mtp_model_runner_integration.py -q
```

Expected: all tests pass.

---

### Task 3: Real-Checkpoint Physical Transaction Probe

**Files:**
- Modify: `tools/qwen35_mtp_real_checkpoint_gate.py`
- Modify: `tools/test_qwen35_mtp_real_checkpoint_gate.py`
- Modify: `tools/run_qwen35_mtp_real_checkpoint_gate_remote.sh`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md`

**Interfaces:**
- Consumes:
  - loaded real `Qwen35MTPProposalExecutor`;
  - real bound `Qwen35MTPPhysicalSlotStore`;
  - synthetic deterministic token IDs and target hidden tensors on the real
    CUDA device.
- Produces:
  - `transaction_probe(q, batch_size, accepted)`;
  - all 28 transaction cases with physical identity and rollback continuation
    evidence.

- [x] **Step 1: Write failing gate tests**

Require `_load_real_runtime()` to expose a callable `transaction_probe` when
registration succeeds. The probe must:

1. bootstrap fresh sequence IDs;
2. run the real MTP module for exact Q;
3. snapshot accepted slot IDs, K/V data pointers, and values;
4. commit `max(accepted - 1, 0)` entries;
5. verify accepted rows retain IDs, pointers, and values;
6. verify rejected rows are unowned and cleared;
7. run and roll back an identical continuation twice and require equal token
   output while committed K/V remains unchanged.

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_qwen35_mtp_real_checkpoint_gate.py -q
```

Expected: failure because the runtime still installs a fixed transaction
blocker and no probe.

- [x] **Step 3: Implement the transaction probe**

The probe may use synthetic target hidden rows, but all MTP projections,
attention writes, and K/V tensors must be the real loaded checkpoint module on
CUDA. It must not simulate K/V copies or claim KV-offload evidence.

Remove only the `transaction` blocker when the probe is installed. Preserve
the eager/reference and graph/eager blockers, so overall status remains
`FAIL / NOT_PROMOTABLE`.

- [x] **Step 4: Run local and remote gates**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_qwen35_mtp_physical_kv.py \
  tools/test_proposal_kv_cache.py \
  tools/test_qwen35_mtp_executor.py \
  tools/test_qwen35_mtp_model_runner_integration.py \
  tools/test_qwen35_mtp_real_checkpoint_gate.py -q

KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
bash tools/run_qwen35_mtp_real_checkpoint_gate_remote.sh
```

Expected remote report:

```text
accepted_slot_identity_preserved=true
rejected_slots_released=true
post_rollback_continuation_equal=true
status=FAIL
promotion_classification=NOT_PROMOTABLE
```

- [x] **Step 5: Record exact evidence and limitations**

## Final Evidence

Local focused regression was run in three isolated Python processes to avoid
test-module stub pollution:

```text
46 passed in 2.31s
48 passed in 3.55s
27 passed in 0.11s
total: 121 passed
```

The authoritative real-checkpoint transaction artifact is:

```text
remote run root:
  /data00/home/sitian/sitian-workspace01/tllm/qwen35-mtp-runs/
    qwen35-mtp-20260813-042915-14310

local artifact:
  artifacts/qwen35-mtp-runs/
    qwen35-mtp-20260813-042915-14310/
      qwen35_mtp_real_checkpoint_gate.json
```

The `20260813` directory token is copied verbatim from the remote run even
though the current session date is 2026-08-12. It is treated as an opaque run
identifier, not as chronology evidence.

It records all 28 combinations across `Q=(1,2,3,4)`, batch sizes `(1,4)`,
and accepted counts `0..Q`:

```text
accepted_slot_identity_preserved: true
rejected_slots_released:           true
post_rollback_continuation_equal:  true
status:                            FAIL
promotion_classification:          NOT_PROMOTABLE
```

The transaction blocker is closed. The two remaining blockers are the absent
independent eager/reference comparison backend and the absent exact-Q MTP CUDA
graph capture backend. This evidence is limited to TP1, KV offload disabled,
one Qwen3.5 architecture, no long-context coverage, and no performance claim.

Record the local test count, isolated remote test count, unique remote run
root, local artifact path, device/runtime, transaction booleans, remaining
blockers, and the explicit boundary that this is TP1/no-offload/single-model
correctness evidence rather than performance or promotion evidence.

