# Qwen3 Draft Blockwise Proposal-KV Offload Implementation Plan

> **For agentic workers:** Execute inline in this checkout. Do not dispatch
> subagents, create a branch/worktree, stage, commit, push, stash, reset, or
> clean. Every production change requires a demonstrated RED test first.

**Goal:** Run the real Qwen3 independent-draft TP1 gate with Proposal-KV GPU
capacity eight, exact batch 1/4 greedy parity, and real bidirectional KV
movement.

**Architecture:** Adapt `ProposalKVResidencyManager` to the existing blockwise
online-softmax attention contract. Residency-backed prompt bootstrap becomes
incremental, and proposal decode stages logical history window by window while
protecting current write entries. Direct allocation and lifecycle semantics are
unchanged.

**Tech Stack:** Python 3.11, PyTorch 2.7, FlashAttention 2.8.3.post1, pytest,
CUDA 12.6 runtime on NVIDIA A100.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Keep `MAX_PROPOSAL_TOKENS=4`.
- Preserve exact greedy target parity and accepted-prefix commit/rollback.
- Do not modify existing Task 8 TP4 authority producer/worker/verifier/runner.
- Do not count synthetic KV copies as authority.
- Do not commit; the usual plan commit steps are intentionally omitted.

---

### Task 1: Proposal-KV Blockwise Staging Adapter

**Files:**
- Modify: `tools/test_proposal_kv_residency.py`
- Modify: `tinyvllm/engine/proposal_kv_residency.py`

**Interfaces:**
- Produces: `ProposalKVResidencyManager.blockwise_attention_adapter`
- Produces adapter methods used by `tinyvllm.layers.attention`

- [ ] Add a test that commits more logical entries than physical slots, marks a
  dirty committed victim, stages a protected logical window, and asserts the
  adapter preserves required/write slots while recording real D2H and H2D
  counters.
- [ ] Run:

```bash
python -m pytest -q \
  tools/test_proposal_kv_residency.py::test_blockwise_adapter_stages_window_without_evicting_protected_writes
```

Expected RED: `ProposalKVResidencyManager` has no blockwise adapter.

- [ ] Implement a dedicated adapter with generation-safe logical-ID lookup,
  protected victim selection, staging, wait, touch, and tensor-free diagnostic
  stats.
- [ ] Re-run the focused test and all Proposal-KV residency tests:

```bash
python -m pytest -q tools/test_proposal_kv_residency.py
```

Expected GREEN.

### Task 2: Qwen3 Backend Blockwise Decode Context

**Files:**
- Modify: `tools/test_qwen3_draft_backend.py`
- Modify: `tinyvllm/engine/autoregressive_draft_executor.py`
- Modify: `tinyvllm/engine/qwen3_draft_backend.py`

**Interfaces:**
- Extend `AutoregressiveDraftDecodeRow` with logical visible-entry IDs and a
  blockwise-offload flag.
- `Qwen3DraftBackend.decode_step_batch()` consumes logical rows and configures
  `temporary_context()` for blockwise decode.

- [ ] Add a backend test with four rows, four protected writable slots, logical
  histories longer than physical capacity, and a recording model that asserts
  `kv_offload_blockwise_decode=True`, window size one, and logical block tables
  are passed without constructing a full physical block table.
- [ ] Run the focused test and verify RED because the row/backend lacks the
  logical blockwise contract.
- [ ] Implement the minimal row fields and backend context branch. Keep the
  existing dense physical-slot branch byte-for-byte equivalent for direct
  allocators.
- [ ] Run:

```bash
python -m pytest -q tools/test_qwen3_draft_backend.py
```

Expected GREEN.

### Task 3: Incremental Residency Bootstrap and Proposal Decode

**Files:**
- Modify: `tools/test_autoregressive_draft_executor.py`
- Modify: `tinyvllm/engine/autoregressive_draft_executor.py`

**Interfaces:**
- Residency bootstrap creates and commits one real prompt-token transaction per
  active sequence per round.
- Residency proposal decode acquires only write leases and passes logical
  history to the backend.

- [ ] Add a test with logical capacity 32, GPU capacity two, and a five-token
  prompt. Assert bootstrap succeeds through five real decode forwards, commits
  five logical prompt entries, and never asks for more than one write slot per
  sequence.
- [ ] Add a batch-four proposal test with GPU capacity eight. Assert decode rows
  are emitted as one backend batch, full-history `ensure_readable()` is not
  called before the backend, and finalize/rollback behavior matches the direct
  path.
- [ ] Run both tests and verify RED on the current whole-prompt bootstrap and
  full-history read leases.
- [ ] Implement incremental residency bootstrap with reverse-order cleanup and
  the blockwise proposal branch.
- [ ] Run:

```bash
python -m pytest -q \
  tools/test_autoregressive_draft_executor.py \
  tools/test_qwen3_draft_backend.py \
  tools/test_proposal_kv_residency.py
```

Expected GREEN.

### Task 4: Local Regression and Remote TP1 Authority

**Files:**
- Modify after evidence: `AGENT_HANDOFF_STATE.md`
- Modify after evidence:
  `docs/superpowers/audits/2026-08-15-phase1-prompt-to-artifact-coverage.md`

**Interfaces:**
- Remote output:
  `/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815/run/tp1-qwen3-loaded-offload-gpu4.json`

- [ ] Run the focused local regression:

```bash
python -m pytest -q \
  tools/test_autoregressive_draft_tp1_engine_gate.py \
  tools/test_autoregressive_draft_registration.py \
  tools/test_autoregressive_draft_model_runner_integration.py \
  tools/test_autoregressive_draft_executor.py \
  tools/test_qwen3_draft_backend.py \
  tools/test_proposal_kv_residency.py \
  tools/test_qwen3_draft_proposal_kv_storage.py
```

Expected GREEN.

- [ ] Synchronize only changed source/test files to the existing remote tmpfs
  source snapshot and refresh `source_sha256.txt`.
- [ ] Run GPU 4 TP1 with target Qwen3-1.7B, draft Qwen3-0.6B, batch 1/4,
  `max_output_tokens=8`, offload enabled, and GPU capacity eight.
- [ ] Validate the artifact requires:
  `gate_pass=true`, exact output-token parity for both cases, nonempty
  acceptance rows, real draft forwards, zero extra target forwards, distinct
  proposal/target storage, zero accepted-entry copy/replay/rematerialization,
  zero live slots after release, and positive H2D/D2H operations and bytes.
- [ ] Download a compact authority bundle containing the result, logs,
  FlashAttention smoke, wheel/build provenance, source hashes, checkpoint
  identities, tokenizer contract, and verifier receipt into the local
  repository.
- [ ] Update the handoff and Phase 1 audit with what the TP1 result proves, what
  it does not prove, and the remaining TP4/performance/long-context work.
