# Qwen3.5 TP4 Real Prerequisite Authority Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert complete verified TP4 correctness authority runs and their
remote receipts into the existing benchmark prerequisite bundle contract.

**Architecture:** Add one read-only adapter that invokes production verifiers,
validates cached/Engine receipt chains, derives canonical bundle documents,
and returns existing `AuthorityInput` rows. Keep the existing prerequisite
builder as the final bundle publication boundary.

**Tech Stack:** Python standard library, existing TP4 authority verifiers,
remote plan/receipt modules, and direct executable test files.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, merge, create a branch, or open a PR.
- Do not execute SSH, `scp`, `nvidia-smi`, Torch, Transformers, CUDA, model
  loading, Engine construction, or GPU workloads.
- Treat root-logit remote receipt support as absent and explicit.
- Do not claim performance or accuracy benefit from CPU-only authority tests.

---

### Task 1: Complete-Directory Adapter Contract

**Files:**
- Create: `tools/qwen35_tp4_real_prerequisite_authority_adapter.py`
- Create: `tools/test_qwen35_tp4_real_prerequisite_authority_adapter.py`

**Interfaces:**
- Consumes: `RealAuthorityRun`, complete authority directories, and optional
  plan/authorization/receipt paths.
- Produces:
  `adapt_real_authorities(*, runs, verification_output_dir) -> tuple[AuthorityInput, ...]`.

- [x] Write tests proving naked summary JSON and incomplete authority
  directories are rejected.
- [x] Run the new test file and observe the expected missing-module or
  missing-interface RED.
- [x] Implement strict input dataclasses, safe regular-file/directory checks,
  and lazy production-verifier loading.
- [x] Run the focused tests and require GREEN.

### Task 2: Cached and Engine Receipt Binding

**Files:**
- Modify: `tools/qwen35_tp4_real_prerequisite_authority_adapter.py`
- Modify: `tools/test_qwen35_tp4_real_prerequisite_authority_adapter.py`

**Interfaces:**
- Consumes: verified remote plan, consumed authorization, execution receipt,
  and exact plan-local downloaded authority directory.
- Produces: receipt-bound canonical authority inputs.

- [x] Add RED tests for missing receipt chain, unconsumed authorization,
  source/model/run-tag drift, and wrong downloaded directory.
- [x] Reuse each authority's production plan and receipt verifier rather than
  reimplementing receipt semantics.
- [x] Bind receipt summary identities to independent verifier output.
- [x] Run focused tests and require GREEN.

### Task 3: Bundle Integration and Source Contract

**Files:**
- Modify: `tools/build_qwen35_tp4_performance_prerequisites.py`
- Modify: `tools/test_build_qwen35_tp4_performance_prerequisites.py`
- Modify: `tools/run_qwen35_tp4_hybrid_prefix_benchmark_remote.py`
- Modify: `tools/test_run_qwen35_tp4_hybrid_prefix_benchmark_remote.py`

**Interfaces:**
- Consumes: adapter-produced `AuthorityInput` rows.
- Produces: provenance-bearing v2 `correctness_prerequisites.json` bundle;
  the adapter remains intentionally outside the remote source inventory.

- [x] Add an end-to-end RED test proving adapter output builds a valid bundle.
- [x] Keep the local-only adapter out of the benchmark remote source
  inventory and add a regression assertion for that boundary.
- [x] Upgrade the prerequisite schema to v2 while preserving benchmark
  authorization SHA binding.
- [x] Run focused tests and require GREEN.

### Task 4: Expanded Authority Gate and Handoff

**Files:**
- Modify:
  `docs/superpowers/plans/2026-07-29-qwen35-tp4-hybrid-prefix-performance-cache-authority.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] Run the adapter, prerequisite builder, benchmark runner, source
  contract, and selected authority suites.
- [x] Run focused `py_compile`, embedded-script compilation where applicable,
  and `git diff --check`.
- [x] Confirm staged file count remains zero.
- [x] Record exact counts and the explicit root-logit receipt gap.
- [x] Preserve the boundary that no real correctness or performance evidence
  was produced.

## 2026-07-29 Completion Update

All four implementation tasks are complete under the CPU-only boundary.

The adapter now accepts only complete authority directories. It invokes the
production independent verifier for each authority and, for cached
continuation and Engine authority, additionally verifies the exact remote
execution plan, consumed authorization, execution receipt, plan-local
downloaded directory, run tag, source identity, and model identity.

The prerequisite bundle contract is now:

```text
schema:
  qwen35.tp4-performance-prerequisites.v2
provenance:
  qwen35.tp4-performance-prerequisite-provenance.v1
root-logit binding:
  complete_directory_only
cached/Engine binding:
  remote_execution_receipt
```

Every authority row carries a provenance path and SHA. Cached and Engine
provenance is self-contained: the builder copies the verified execution plan,
consumed authorization, and execution receipt into the bundle, and the
benchmark contract independently rehashes all three files. Root-logit
provenance explicitly records `root_logit_receipt_gap=true`; no receipt is
invented for that authority.

The adapter remains local-only and is intentionally excluded from the remote
benchmark source tar. Its output is frozen by the v2 prerequisite bundle and
the benchmark authorization's prerequisite SHA.

Fresh CPU-safe evidence:

```text
real prerequisite adapter:             6 passed
prerequisite builder:                   6 passed
performance prerequisite contract:     16 passed
benchmark worker:                       27 passed
benchmark assembler:                     5 passed
benchmark verifier:                     11 passed
focused affected gate:                 116 passed across 12 files
expanded selected authority gate:      288 passed across 41 files
focused py_compile:                    passed
embedded command-script compile:       passed
git diff --check:                      passed
staged files:                          0
```

No SSH, `scp`, `nvidia-smi`, Torch, Transformers, CUDA, model load, Engine
construction, remote directory creation, or GPU workload was executed.

Strict boundary:

```text
complete-directory adapter:
  implemented and CPU-verified
self-contained v2 prerequisite provenance:
  implemented and CPU-verified
real root-logit receipt:
  absent by explicit provenance declaration
real root-logit/cached/Engine bundle:
  not produced
canonical performance/cache benchmark:
  not run
latency/throughput/cache/GPU-memory/compression/quality/accuracy benefit:
  unmeasured and not claimable
```
