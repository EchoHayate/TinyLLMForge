# Qwen3.5 Generic Speculative TP4 32K Transactional Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: execute inline with strict test-driven development. Subagents, worktrees, commits, staging, pushes, stashes, resets, and cleans are forbidden.

**Goal:** Build and run an independent source-bound Qwen3.5 TP4/32K transactional correctness authority without modifying the established 4K or 16K authorities.

**Architecture:** Add a 32K overlay over the frozen 4K authority validators and orchestration, explicitly injecting a 32K worker, complete source inventory, and 32K verifier. Reuse the established worker behavior through an Engine-factory wrapper that forces the approved long-context configuration, while requiring positive candidate H2D evidence for both batch sizes.

**Tech Stack:** Python 3, pytest, TinyLLMForge Engine, PyTorch distributed TP4, Bash, SSH/rsync, JSON and SHA-256 source binding.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not modify existing 4K or 16K authority files or artifacts.
- Freeze schema `qwen35.generic-speculative-tp4-32k-transactional-correctness.v1`.
- Freeze classification `SECOND_MODEL_TP4_32K_ESTABLISHED`.
- Freeze scope `second_model_tp4_32k_only`.
- Freeze context `32768`, batch `(1,4)`, output `8`, n-gram `3`, proposal `4`.
- Freeze `max_model_len=33024`, batched tokens `132096`, prefill step `1024`.
- Freeze GPU/logical/blockwise blocks `68/640/8`.
- Require positive candidate H2D copies and bytes in batch 1 and batch 4.
- Require exact parity and zero accepted-prefix replay.
- Use only `sitian@10.232.195.203`, the approved Kerberos cache, and non-persistent SSH.
- Keep Phase 1 `NOT_PROMOTABLE`.

---

### Task 1: Freeze the 32K gate contract

**Files:**
- Create: `tools/test_qwen35_generic_speculative_tp4_32k_gate.py`
- Create: `tools/qwen35_generic_speculative_tp4_32k_gate.py`

**Interfaces:**
- Consumes: validators and campaign orchestration from `tools/qwen35_generic_speculative_tp4_gate.py`.
- Produces: 32K constants, result validation, hashing helpers, and campaign CLI.

- [ ] Write a failing import/constants test asserting the exact schema,
  classification, scope, context, Engine capacities, limitations, and complete
  source inventory.
- [ ] Run the focused test and observe `FileNotFoundError` for the missing gate.
- [ ] Implement the minimal private frozen-gate overlay.
- [ ] Run the focused test and observe PASS.

### Task 2: Require real H2D for both candidate cells

**Files:**
- Modify: `tools/test_qwen35_generic_speculative_tp4_32k_gate.py`
- Modify: `tools/qwen35_generic_speculative_tp4_32k_gate.py`

**Interfaces:**
- Consumes: normalized `kv_rank_deltas`.
- Produces: `validate_result(value) -> dict` that rejects zero H2D copies or
  bytes in `ngram:b1` and `ngram:b4`.

- [ ] Add one passing positive-H2D test and four failing zero copies/bytes tests.
- [ ] Run `pytest ... -k h2d` and observe the four missing failures.
- [ ] Add the minimal post-validation loop over batch sizes `(1,4)`.
- [ ] Re-run and observe all H2D tests pass.

### Task 3: Add the 32K worker

**Files:**
- Modify: `tools/test_qwen35_generic_speculative_tp4_32k_gate.py`
- Create: `tools/qwen35_generic_speculative_tp4_32k_worker.py`

**Interfaces:**
- Consumes: frozen worker receipts, profiling, cleanup, and generation helpers.
- Produces: `run_policy_cell(...) -> dict` and CLI using 32K prompts and the
  frozen long-context Engine envelope.

- [ ] Add a failing injected-factory test for `32768` prompt tokens and exact
  Engine kwargs `33024/132096/1024/68/640/8`.
- [ ] Observe missing-worker RED.
- [ ] Implement the private worker overlay and Engine-factory wrapper.
- [ ] Observe focused GREEN.

### Task 4: Fix campaign dispatch at the authority boundary

**Files:**
- Modify: `tools/test_qwen35_generic_speculative_tp4_32k_gate.py`
- Modify: `tools/qwen35_generic_speculative_tp4_32k_gate.py`

**Interfaces:**
- Consumes: frozen `run_campaign`.
- Produces: an adapter that defaults to the 32K worker, 32K source inventory,
  and independent 32K verifier.

- [ ] Add a delegated-capture test that asserts all three defaults.
- [ ] Observe RED before the adapter exists.
- [ ] Implement `run_campaign` with explicit 32K defaults while preserving
  caller overrides.
- [ ] Observe GREEN.

### Task 5: Add the independent verifier

**Files:**
- Modify: `tools/test_qwen35_generic_speculative_tp4_32k_gate.py`
- Create: `tools/verify_qwen35_generic_speculative_tp4_32k_gate.py`

**Interfaces:**
- Consumes: the 32K gate result and hash validators.
- Produces: `verify_run(run_dir, source_root=None) -> dict` and CLI status.

- [ ] Add valid-source PASS, zero-H2D FAIL, result-hash FAIL, and bound-source
  tamper FAIL tests.
- [ ] Observe missing-verifier RED.
- [ ] Implement the private frozen-verifier overlay bound to the 32K gate.
- [ ] Observe verifier GREEN.

### Task 6: Add the bounded remote runner

**Files:**
- Modify: `tools/test_qwen35_generic_speculative_tp4_32k_gate.py`
- Create: `tools/run_qwen35_generic_speculative_tp4_32k_gate_remote.sh`

**Interfaces:**
- Consumes: the frozen runner protocol and all bound 32K/frozen source files.
- Produces: a non-replayable local/remote campaign and terminal authority state.

- [ ] Add a failing source-contract test for remote identity, Kerberos,
  non-persistent SSH, four-GPU selection, finite retries, campaign markers,
  32K authority filenames, and frozen dependencies.
- [ ] Observe missing-runner RED.
- [ ] Implement a checked runtime derivation of the frozen runner into the 32K
  namespace and add all frozen dependencies to the source archive.
- [ ] Run the source-contract test and `bash -n`.

### Task 7: Local regression gate

**Files:**
- Modify only new 32K files if a focused RED exposes a defect.

- [ ] Run all 32K tests.
- [ ] Run frozen 4K and 16K authority tests.
- [ ] Run `py_compile` on all new Python files.
- [ ] Run runner `bash -n`.
- [ ] Run `git diff --check` on all related files.

Expected: all available commands exit zero. If a torch-dependent local test
cannot collect because the host lacks torch, record it as an environment
failure rather than pass/fail.

### Task 8: Real TP4/32K campaign

**Files:**
- Generate: `artifacts/qwen35_generic_speculative_tp4_32k/opaque-*/`

- [ ] Launch one fresh run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
  bash tools/run_qwen35_generic_speculative_tp4_32k_gate_remote.sh
```

- [ ] Reuse the same unified-exec session for bounded polling.
- [ ] Preserve any failed authority, diagnose with a focused RED, and use a
  fresh run ID after the fix.
- [ ] Require verifier PASS, exact parity, acceptance and rejection, zero
  replay, positive candidate H2D in both batch sizes, all-rank transaction
  agreement, and complete cleanup.

### Task 8A: Remove the Qwen3.5 dense-prefill 32K OOM

**Files:**
- Modify: `tools/test_qwen35_cached_prefill_eager_attention.py`
- Modify: `tinyvllm/layers/qwen35_full_attention.py`

**Interfaces:**
- Consumes: flattened request-local Q/K/V and `cu_seqlens_q`.
- Produces: bounded-tile exact causal prefill attention for reference lengths
  greater than 16,384.

- [ ] Add forced-small blockwise numerical, GQA, request-isolation, bounded
  matmul-shape, finite-output, and reference-padding tests.
- [ ] Copy the changed test/source to the approved remote Python environment
  and invoke the focused tests directly; observe RED because the blockwise
  helper/path does not exist.
- [ ] Implement query/key tiled FP32 online-softmax with 512-token production
  tiles and a 16,384-token production threshold.
- [ ] Re-run the focused remote tests and observe GREEN.
- [ ] Run the existing Qwen3.5 cached-prefill/full-attention regressions.
- [ ] Launch a fresh, non-replayed TP4/32K campaign; preserve the OOM campaign
  as `authority.failed`.

### Task 9: Audit and handoff

**Files:**
- Modify: `docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [ ] Record authority path, GPU indices, result/source/model hashes, proposal
  totals, H2D/D2H totals, parity, replay, cleanup, verifier, and failed runs.
- [ ] Mark only Qwen3.5 TP4/32K correctness established.
- [ ] Keep TP4 performance, learned drafter/native MTP, KV8/KV4, and Phase 1
  promotion open.
- [ ] Run final fresh tests, compile, shell syntax, verifier, and diff check
  before any completion claim.
