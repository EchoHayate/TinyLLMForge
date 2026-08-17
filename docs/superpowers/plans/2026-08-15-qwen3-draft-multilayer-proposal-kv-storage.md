# Qwen3 Draft Multi-Layer Proposal-KV Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable multi-layer Proposal-KV storage adapter for the
independent Qwen3 drafter while keeping runtime registration direct-only.

**Architecture:** Extract storage ownership from the Qwen3 backend module into
one class that owns every layer's GPU rows and optional CPU logical backing.
Retain the existing direct physical-store constructor as a compatibility
subclass, and leave ModelRunner registration on `DirectProposalKVAllocator`.

**Tech Stack:** Python, PyTorch tensors, pytest, TinyLLMForge Proposal-KV
allocator and residency protocols.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Preserve exact greedy selection and `MAX_PROPOSAL_TOKENS=4`.
- Default registration remains direct-only.
- Default registration allocates no CPU backing and creates no copy stream.
- Add no learned-drafter offload configuration.
- Run no GPU, remote, NCCL, or authority workload.
- Do not stage, commit, push, stash, reset, or clean.

---

### Task 1: Multi-Layer Storage Contract

**Files:**
- Create: `tinyvllm/engine/qwen3_draft_proposal_kv.py`
- Create: `tools/test_qwen3_draft_proposal_kv_storage.py`

**Interfaces:**
- Produces: `Qwen3DraftProposalKVStorage(model, *, logical_capacity,
  gpu_capacity, dtype, device, allocate_cpu_backing,
  allocate_pinned_cpu=True)`
- Produces: `entry_nbytes()`, `copy_gpu_to_cpu(rows)`,
  `copy_cpu_to_gpu(rows)`, `bind_attention_backends()`

- [x] **Step 1: Write failing geometry and byte-accounting tests**

Create fake Qwen3 layers with two local layers, two KV heads, and head
dimension four. Assert GPU shape `[2, 2, 1, 2, 4]`, CPU shape
`[2, 4, 1, 2, 4]`, and:

```python
assert storage.entry_nbytes() == (
    2 * 2 * 1 * 2 * 4 * torch.float32.itemsize
)
```

- [x] **Step 2: Verify RED**

Run:

```bash
python -m pytest -q tools/test_qwen3_draft_proposal_kv_storage.py
```

Expected: collection fails because
`tinyvllm.engine.qwen3_draft_proposal_kv` does not exist.

- [x] **Step 3: Implement the minimal storage owner**

Implement validated model geometry, GPU allocation, optional CPU allocation,
all-layer attention binding, and exact entry-byte accounting.

- [x] **Step 4: Verify GREEN**

Run the focused test file and expect all geometry tests to pass.

### Task 2: Atomic Multi-Layer Copy Semantics

**Files:**
- Modify: `tinyvllm/engine/qwen3_draft_proposal_kv.py`
- Modify: `tools/test_qwen3_draft_proposal_kv_storage.py`

**Interfaces:**
- Consumes: storage tensors from Task 1
- Produces: validated full-entry D2H/H2D copy behavior

- [x] **Step 1: Write failing copy tests**

Fill every layer of two physical slots with distinct values, copy them to
logical rows, clear GPU rows, then restore them to different physical slots.
Assert every layer's K and V values survive. Add malformed, duplicate, and
out-of-range row cases that assert destination tensors remain unchanged.

- [x] **Step 2: Verify RED**

Run the focused file and expect failures because copy methods are absent.

- [x] **Step 3: Implement minimal validated copies**

Normalize rows to a tuple, validate all rows and uniqueness before mutation,
require CPU backing, and use `Tensor.copy_(..., non_blocking=True)` on full
layer slices.

- [x] **Step 4: Verify GREEN**

Run the focused file and expect all copy tests to pass.

### Task 3: Direct Compatibility Extraction

**Files:**
- Modify: `tinyvllm/engine/qwen3_draft_proposal_kv.py`
- Modify: `tinyvllm/engine/qwen3_draft_backend.py`
- Modify: `tinyvllm/engine/autoregressive_draft_registration.py`
- Modify: `tools/test_qwen3_draft_proposal_kv_storage.py`
- Modify: `tools/test_qwen3_draft_backend.py`

**Interfaces:**
- Produces: unchanged
  `Qwen3DraftPhysicalSlotStore(model, *, capacity, dtype, device)`
- Preserves: existing backend import surface and direct allocator behavior

- [x] **Step 1: Write failing compatibility tests**

Assert the direct store is a `Qwen3DraftProposalKVStorage`, has no CPU
backing, binds every layer, reserves/releases slots, zeroes every layer, and
retains the existing authority schema. Assert generic storage reports
payload/backing metadata without exposing competing occupancy authority.

- [x] **Step 2: Verify RED**

Run the two focused files and expect failures before the class is extracted.

- [x] **Step 3: Move direct storage ownership**

Implement the direct subclass in the new module. Replace the old class body
in `qwen3_draft_backend.py` with an import. Update registration to import the
direct store from the storage module.

- [x] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest -q \
  tools/test_qwen3_draft_proposal_kv_storage.py \
  tools/test_qwen3_draft_backend.py \
  tools/test_autoregressive_draft_registration.py \
  tools/test_autoregressive_draft_model_runner_integration.py
```

Expected: all pass.

### Task 4: Regression and Documentation Gate

**Files:**
- Modify: `docs/superpowers/audits/2026-08-15-phase1-prompt-to-artifact-coverage.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Produces: scoped evidence and non-promotion classification

- [x] **Step 1: Run focused Proposal-KV and learned-drafter regressions**

Run the new storage tests, learned-drafter TP1/TP4 local tests, Proposal-KV
allocator/residency/cache/lifecycle tests, and generic speculative runtime
tests in short-lived processes.

- [x] **Step 2: Run static gates**

Run `py_compile`, a production symbol scan confirming registration remains
direct-only, and scoped `git diff --check`.

- [x] **Step 3: Update audit and handoff**

Record exact commands and counts. Preserve:

```text
AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_OFFLOAD=NOT_ENABLED
REAL_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Plan Self-Review

- Spec coverage: storage shape, entry bytes, copies, validation, direct
  compatibility, default-off runtime, regression, and classification are
  covered.
- Placeholder scan: no implementation placeholder remains.
- Type consistency: the direct subclass and generic storage constructor names
  match the design.
