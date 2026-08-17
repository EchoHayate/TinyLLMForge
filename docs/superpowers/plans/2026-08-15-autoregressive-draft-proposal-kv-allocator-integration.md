# Autoregressive Draft Proposal-KV Allocator Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. This
> workspace explicitly forbids subagents, new worktrees, staging, commits,
> pushes, stashes, resets, and cleans.

**Goal:** Make the independent Qwen3 autoregressive drafter reuse the generic
generation-aware Proposal-KV allocator and lease contract in default-direct
mode.

**Architecture:** Registration wraps the existing multi-layer physical store
in `DirectProposalKVAllocator`. The executor owns readable/writable leases and
passes temporary physical mappings to backend rows; the backend no longer
looks up physical slots from durable cache state.

**Tech Stack:** Python dataclasses and protocols, PyTorch model contexts,
existing Proposal-KV allocator/cache/lifecycle modules, pytest.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Keep `MAX_PROPOSAL_TOKENS=4` and exact greedy behavior unchanged.
- Do not add physical-slot compatibility aliases to Proposal-KV durable state.
- Do not enable proposal-KV offload or create GPU/remote authority artifacts.
- Do not stage, commit, push, stash, reset, clean, or use subagents.

---

### Task 1: Registration Uses the Allocator Boundary

**Files:**
- Modify: `tools/test_autoregressive_draft_model_runner_integration.py`
- Modify: `tools/test_autoregressive_draft_registration.py`
- Modify: `tinyvllm/engine/autoregressive_draft_registration.py`
- Modify: `tinyvllm/engine/model_runner.py`

**Interfaces:**
- Consumes: `DirectProposalKVAllocator(physical_store)`.
- Produces: an autoregressive draft `ProposalKVCache` constructed with an
  allocator and a registration candidate that retains the physical store only
  for diagnostics.

- [x] Add failing tests that require the construction order
  `physical_store -> entry_allocator -> ProposalKVCache`.
- [x] Run the focused registration and ModelRunner tests and verify the
  failure is caused by the missing allocator construction dependency.
- [x] Add `build_proposal_kv_allocator` to
  `AutoregressiveDraftRegistrationDependencies`.
- [x] Build `DirectProposalKVAllocator` by default and pass it to
  `build_proposal_kv_cache`.
- [x] Re-run the focused tests and verify GREEN.

### Task 2: Executor Acquires and Completes Leases

**Files:**
- Modify: `tools/test_autoregressive_draft_executor.py`
- Modify: `tinyvllm/engine/autoregressive_draft_executor.py`

**Interfaces:**
- Consumes:
  `ProposalKVEntryAllocator.ensure_writable`,
  `ensure_readable`, `record_write_complete`, and `record_read_complete`.
- Produces:
  `AutoregressiveDraftPrefillRow.physical_slot_ids`,
  `AutoregressiveDraftDecodeRow.writable_physical_slot_id`, and
  `AutoregressiveDraftDecodeRow.visible_physical_slot_ids`.

- [x] Add failing bootstrap tests that require writable-lease slot mappings
  and one completion record per transaction.
- [x] Run the focused executor tests and verify RED on the removed physical
  slot API.
- [x] Implement executor-owned bootstrap leases and replace all
  `staged_slot_ids`/`committed_slot_ids` length checks with logical identity
  counts.
- [x] Add failing decode tests for exact committed-plus-staged physical
  ordering and read/write completion on success and backend failure.
- [x] Implement executor-owned decode leases and completion recording.
- [x] Re-run the focused executor tests and verify GREEN.

### Task 3: Backend Consumes Only Ephemeral Physical Mappings

**Files:**
- Modify: `tools/test_qwen3_draft_backend.py`
- Modify: `tinyvllm/engine/qwen3_draft_backend.py`

**Interfaces:**
- Consumes: the row mapping fields produced by Task 2.
- Produces: unchanged prefill/decode model contexts without reading removed
  cache physical-slot APIs.

- [x] Add failing tests that construct rows with explicit physical mappings
  and reject wrong counts, wrong final writable slot, duplicates, and
  non-integer slots.
- [x] Run the focused backend tests and verify RED.
- [x] Resolve the physical store through
  `proposal_kv_cache.entry_allocator.physical_store`.
- [x] Build prefill `slot_mapping` and decode `block_tables` exclusively from
  row fields.
- [x] Remove all production references to `ProposalKVCache.physical_store`,
  `staged_slot_ids`, and `committed_slot_ids`.
- [x] Re-run the backend and executor tests and verify GREEN.

### Task 4: Local Reuse Gate and Handoff

**Files:**
- Create: `tools/test_autoregressive_draft_proposal_kv_allocator_contract.py`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: a dependency-light terminal classification gate and an updated
  continuation record.

- [x] Add a source-and-contract gate that requires allocator wrapping,
  lease acquisition/completion calls, zero stale physical-slot API references,
  default-direct mode, and unchanged `MAX_PROPOSAL_TOKENS=4`.
- [x] Run the focused learned-drafter tests.
- [x] Run Proposal-KV Tasks 1-7 regression tests and the generic speculative
  runtime regression slice.
- [x] Run `py_compile` for changed Python files and scoped
  `git diff --check`.
- [x] Update `AGENT_HANDOFF_STATE.md` with exact commands, pass/fail counts,
  dependency limitations, and the completion boundary from the design.

## 2026-08-15 Status Reconciliation

The plan status was reconciled against the recorded Task 1-3 RED/GREEN
history, current production source, and fresh local verification:

```text
allocator/runtime learned-drafter matrix: 305 passed in 15.22s
registration-focused subset:              99 passed
focused py_compile:                       PASS
stale production symbol scan:             PASS
scoped diff check:                        PASS
```

This establishes allocator reuse and ephemeral lease-to-row physical mappings
as local contracts. It does not establish loaded-checkpoint parity, real
proposal-KV movement, performance improvement, Phase 1 completion, or
promotion.

