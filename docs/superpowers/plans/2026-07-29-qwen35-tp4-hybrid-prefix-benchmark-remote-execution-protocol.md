# Qwen3.5 TP4 Hybrid-Prefix Benchmark Remote Execution Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the existing data-only 70-case benchmark launch plan into a
single-use, receipt-bound, independently reverified remote execution protocol
without executing SSH or GPU work during implementation.

**Architecture:** Keep the current benchmark preflight and canonical launch
plan as immutable input data. Add a benchmark-specific single-use
authorization, execution receipt, and dependency-injected offline executor;
the only future subprocess owner remains an isolated adapter. Downloaded
canonical artifacts must be safely extracted and reverified locally with the
source-bound independent verifier before a PASS receipt can be published.

**Tech Stack:** Python standard library, canonical JSON/SHA256, atomic rename,
file-local test entrypoints, existing hybrid-prefix benchmark contract,
assembler, verifier, and Engine/cached remote protocol patterns.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, merge, create a branch, or create a PR.
- Do not execute SSH, SCP, `nvidia-smi`, Torch, Transformers, CUDA, model load,
  Engine construction, or a GPU workload.
- Future remote execution target is exactly `sitian@10.232.195.203`.
- Future execution environment is exactly
  `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- Preserve the active-compute-process guard and the 24 GiB per-GPU free-memory
  floor.
- Never claim latency, throughput, cache, GPU-memory, compression, quality, or
  accuracy benefit before real correctness and canonical benchmark artifacts
  pass independent verification.

---

## Completion Audit: Objective to Artifact Checklist

1. **No accuracy regression**
   - Required artifact: real root-logit, cached-continuation, and Engine
     correctness prerequisite bundle accepted by the shared semantic validator.
   - Current evidence: validators and builders are CPU-tested.
   - Gap: real prerequisite bundle is absent.
2. **Faster inference**
   - Required artifact: canonical 70-case benchmark artifact with recompute and
     exact-restore pairs, independent verification, and frozen latency and
     throughput metrics.
   - Current evidence: worker, adapter, assembler, and verifier are CPU-tested.
   - Gap: no real TP4 benchmark has run.
3. **Less cache / physical memory**
   - Required artifact: canonical cache and all-rank CUDA allocator snapshots,
     with independent physical-byte reconstruction and no correctness loss.
   - Current evidence: snapshot transport and schemas are CPU-tested.
   - Gap: no real physical GPU/cache evidence exists.
4. **Safe auditable remote execution**
   - Required artifact: immutable launch plan, one-time authorization, bounded
     PASS/FAILED receipt, package SHA/size, final resource recheck, safe
     extraction, and local independent re-verification.
   - Current evidence: launch plan exists only as data.
   - Gap: benchmark-specific authorization, receipt, executor, and adapter are
     absent.

The objective is not complete. Task 4 is the nearest structural blocker and is
the scope of this plan.

### Task 1: Benchmark Single-Use Authorization

**Files:**
- Create: `tools/qwen35_tp4_hybrid_prefix_benchmark_remote_execution_authorization.py`
- Create: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_remote_execution_authorization.py`

**Interfaces:**
- Consumes: the exact dictionary returned by
  `build_authorized_launch_plan(...)`.
- Produces:
  `produce_authorization(plan, output_path, nonce)`,
  `validate_authorization(plan, payload)`, and
  `consume_authorization(plan, authorization_path, consumed_path)`.

- [ ] Write tests proving the authorization binds the canonical plan SHA,
  run tag, prerequisite/source/model/workload identities, four GPU indices,
  all 70 distinct port pairs, and a safe nonce.
- [ ] Run the test and require missing-module RED.
- [ ] Implement closed-schema validation and atomic publication.
- [ ] Implement claim-before-rewrite single-use consumption.
- [ ] Run the focused test and require GREEN.

### Task 2: Benchmark Execution Receipt

**Files:**
- Create: `tools/qwen35_tp4_hybrid_prefix_benchmark_remote_execution_receipt.py`
- Create: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_remote_execution_receipt.py`

**Interfaces:**
- Consumes: verified launch plan, consumed authorization, and ordered execution
  step results.
- Produces:
  `validate_execution_receipt(...)`,
  `produce_execution_receipt(...)`, and `verify_receipt_files(...)`.

- [ ] Write missing-module RED tests for exact step order, command hashes,
  bounded logs, package SHA/size, preflight/final GPU UUID identity, canonical
  artifact verifier PASS equality, and authorization binding.
- [ ] Implement the minimal closed-schema receipt validator and atomic writer.
- [ ] Run focused tests and require GREEN.

### Task 3: Dependency-Injected Offline Executor

**Files:**
- Create: `tools/qwen35_tp4_hybrid_prefix_benchmark_remote_execution_executor.py`
- Create: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_remote_execution_executor.py`

**Interfaces:**
- Consumes: verified plan file, active authorization file, exact execution
  environment, explicit plan verifier, and explicit command runner.
- Produces: a PASS receipt or bounded prefix-preserving FAILED evidence.

- [ ] Write missing-module RED tests proving there is no default runner,
  authorization is consumed before the first command, outputs cannot preexist,
  package identity is checked, and failures never publish PASS.
- [ ] Implement only the injected execution path and exact
  `KRB5CCNAME` requirement.
- [ ] Run focused tests and require GREEN.

### Task 4: Isolated Subprocess Adapter and Source Contract

**Files:**
- Create: `tools/qwen35_tp4_hybrid_prefix_benchmark_remote_subprocess_adapter.py`
- Create: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_remote_subprocess_adapter.py`
- Modify: `tools/run_qwen35_tp4_hybrid_prefix_benchmark_remote.py`
- Modify: `tools/test_run_qwen35_tp4_hybrid_prefix_benchmark_remote.py`

**Interfaces:**
- Consumes: only frozen command shapes from the verified plan.
- Produces: bounded `{returncode, stdout, stderr}` rows and package output
  SHA/size for the executor.

- [ ] Add AST tests proving only the adapter owns subprocess execution.
- [ ] Add exact executable and command-shape allowlists.
- [ ] Include all protocol modules in the deterministic source inventory.
- [ ] Run focused tests and require GREEN.

### Task 5: Expanded Authority Gate and Documentation

**Files:**
- Modify:
  `docs/superpowers/plans/2026-07-29-qwen35-tp4-hybrid-prefix-performance-cache-authority.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [ ] Run all new protocol suites plus the existing 36-file selected authority
  inventory.
- [ ] Run focused `py_compile`.
- [ ] Run `git diff --check`.
- [ ] Confirm staged file count is zero.
- [ ] Record exact test counts and preserve the no-performance-claim boundary.

## Completion Update

The benchmark protocol now reuses the existing audited Engine subprocess
adapter rather than creating a second process owner. Authorization, receipt,
dependency-injected executor, deterministic 11-step plan builder/verifier,
safe package extraction, and frozen-source local verification are implemented.

CPU-only validation:

```text
focused benchmark protocol gate:      51 passed
expanded selected authority gate:    281 passed across 40 files
embedded command-script compile:       passed
```

No remote or GPU action was executed. Real correctness prerequisites and the
canonical 70-case artifact remain absent, so no performance, cache, memory,
compression, quality, or accuracy benefit is claimable.

## 2026-07-29 Prerequisite Bundle Upload and Stage Hardening

A later completion audit found that the 11-step plan froze only the top-level
prerequisite JSON. Worker commands referenced a remote prerequisite path, but
the plan neither uploaded the complete sibling provenance evidence nor staged
it at that path.

The plan now creates a deterministic second archive:

```text
correctness_prerequisites.tar
  correctness_prerequisites.json
  prerequisites/**
```

Only sorted regular non-symlink files are admitted. The tar uses fixed
metadata and is copied into the self-contained plan directory. The `upload`
step contains exactly two explicit `scp` commands: frozen benchmark source
and frozen prerequisite archive.

The `stage` step now:

1. verifies the prerequisite archive SHA;
2. requires the exact frozen member inventory;
3. rejects absolute paths, parent traversal, symlinks, hardlinks, and
   non-regular members;
4. rejects a pre-existing remote prerequisite;
5. extracts into the unique parent shared by all 70 worker prerequisite
   arguments;
6. rehashes the staged `correctness_prerequisites.json`.

The local plan verifier independently opens the prerequisite tar and
reconstructs its sorted, unique, safe regular-file inventory. It also rehashes
the main JSON from inside the tar and requires exact equality with
`local_inputs.prerequisites_owned_files`. This closes a tamper path where an
attacker could previously rewrite both the inventory field and stage command
without changing the tar.

Strict TDD evidence:

```text
inventory-drift RED:
  prerequisite owned-file inventory drift was accepted
execution plan suite:                   7 passed
focused affected gate:                116 passed across 12 files
expanded selected authority gate:     288 passed across 41 files
focused py_compile:                    passed
embedded command-script compile:       passed
git diff --check:                      passed
staged files:                          0
```

No real execution occurred. The protocol is now structurally capable of
transporting a complete v2 prerequisite bundle, but such a bundle has not yet
been produced from real root-logit, cached-continuation, and Engine authority
runs. Therefore no remote benchmark authorization should be issued yet.
