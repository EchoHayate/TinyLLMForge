# Qwen3.5 Real Checkpoint Load Read-Only Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run a source-bound remote preflight that audits the approved Qwen3.5 checkpoint environment while opening zero `.safetensors` payloads.

**Architecture:** Extend the dependency-light real-checkpoint runner with deterministic source staging and a staged internal remote-audit entrypoint. The audit reads only sidecar/JSON/proc metadata, performs stat-only shard inspection, emits `preflight.json` and `source_manifest.json`, and leaves every execution mode except `preflight` and `dry-run` fail-closed.

**Tech Stack:** Python 3.12/3.11 standard library, JSON, SHA256, tar streams, subprocess injection, SSH ControlMaster, Linux `/proc`, `findmnt`, `nvidia-smi`.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; do not use subagents.
- Do not stage, commit, merge, or delete experiment evidence.
- Use only `sitian@10.232.195.203` through `/tmp/ssh-sitian-10.232.195.203`.
- Open zero `.safetensors` payloads and read zero payload bytes.
- Do not implement or launch the real checkpoint-load worker.
- Keep `run`, `download-only`, and `verify-only` fail-closed.
- Do not modify production tile policy, ModelRunner, Engine, Scheduler, publication, or schema-v2 canonical evidence.
- Do not claim speed, memory, cache, compression, or quality improvement.

---

### Task 1: Freeze the Read-Only Preflight Contract

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_load_safety_gate.py`
- Modify: `tools/qwen35_real_checkpoint_load_contract.py`

**Interfaces:**
- Consumes: existing `validate_preflight(record)` safety contract.
- Produces: exact approved model paths/hashes and validation for payload-zero, source-bound preflight records.

- [x] **Step 1: Add failing fixture tests for the expanded READY record**

Add an approved model manifest/path fixture, package/fs/memory/shard fields,
source file hash maps, payload byte counters, checks, and failure reasons.

- [x] **Step 2: Run RED**

Run:

```bash
python tools/test_qwen35_real_checkpoint_load_safety_gate.py
```

Expected: failure because expanded constants/validation are absent.

- [x] **Step 3: Add minimal constants and validation**

Freeze the approved acquisition manifest/model paths, expected config/index/
shard identities, required package names, artifact allowance, and strict
READY/INCOMPLETE validation.

- [x] **Step 4: Run GREEN**

Run the focused test and require all existing dry-run checks to remain green.

### Task 2: Add Deterministic Source Staging and Remote Audit RED Tests

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_load_safety_gate.py`
- Modify: `tools/run_qwen35_real_checkpoint_load_gate_remote.py`

**Interfaces:**
- Produces: `build_source_tar`, `stage_owned_source`, `build_source_manifest`,
  `build_remote_preflight_script`, `classify_preflight_payload`, and
  `run_remote_preflight`.

- [x] **Step 1: Add failing source tar/hash/staging tests**

Assert deterministic archive contents, exact staged hashes, no overwrite, and
the fixed SSH identity.

- [x] **Step 2: Add failing generated-script safety tests**

Assert the script contains stat-only shard inspection and excludes all
forbidden payload/network/loading APIs.

- [x] **Step 3: Add failing READY/INCOMPLETE fixture tests**

Cover exact success, GPU0 occupancy, source mismatch, missing package,
manifest mismatch, shard stat mismatch, and non-zero payload counters.

- [x] **Step 4: Run RED**

Run the focused test and confirm failures are caused by missing interfaces.

- [x] **Step 5: Implement deterministic source staging**

Use an in-memory tar with normalized ownership/mtime, stream it to a new
remote source directory, and verify exact per-file SHA256 remotely.

- [x] **Step 6: Implement the metadata-only audit**

Generate a dependency-light remote script that reads sidecars/JSON/proc
metadata, stats shard files, audits GPU0, and emits a complete raw record.

- [x] **Step 7: Implement fail-closed classification**

Build explicit checks/reasons, compute composite identities and source-tree
SHA256, validate with the contract, and persist the remote artifacts.

- [x] **Step 8: Run GREEN**

Run the focused test and `py_compile`.

### Task 3: Wire the Public Preflight Mode

**Files:**
- Modify: `tools/test_qwen35_real_checkpoint_load_safety_gate.py`
- Modify: `tools/run_qwen35_real_checkpoint_load_gate_remote.py`

**Interfaces:**
- Consumes: Task 2 staging/audit functions.
- Produces: CLI `preflight --run-tag <tag> [--output-json <path>]`.

- [x] **Step 1: Add failing CLI orchestration test**

Inject command runners and assert exact order: stage, remote audit, artifact
download/persistence, validation; assert no worker command is constructed or
executed.

- [x] **Step 2: Run RED**

Confirm `preflight` is still rejected.

- [x] **Step 3: Implement minimal orchestration**

Allow `preflight`, retain intentional rejection for `run`, `download-only`,
and `verify-only`, and print the validated preflight JSON.

- [x] **Step 4: Run GREEN and regression checks**

Run focused tests, dry-run regeneration checks, `py_compile`, and
`git diff --check`.

### Task 4: Execute and Validate the Live Read-Only Preflight

**Files:**
- Create: `experiments/qwen35_hybrid_state/<run-tag>/preflight.json`
- Create: `experiments/qwen35_hybrid_state/<run-tag>/source_manifest.json`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-07-27-qwen35-real-checkpoint-load-read-only-preflight.md`

**Interfaces:**
- Consumes: public preflight CLI and approved SSH ControlMaster.
- Produces: one immutable local evidence directory and the next-gate decision.

- [x] **Step 1: Verify SSH ControlMaster reachability**

Run a non-interactive identity command through the fixed control path.

- [x] **Step 2: Run one unique live preflight**

Use a timestamped run tag. Do not retry into the same remote directory.

- [x] **Step 3: Validate downloaded artifacts**

Require exact source hashes, approved identity, explicit checks/reasons,
`payload_open_count=0`, `payload_bytes_read=0`, and no extra downloaded
artifact.

- [x] **Step 4: Run fresh full verification**

Run focused tests, `py_compile`, artifact validation, forbidden-source scan,
`git diff --check`, staged-file check, and handoff EOF-heading check.

- [x] **Step 5: Complete the plan and append the unique EOF handoff**

Record what the live status proves, what blocked READY if applicable, and that
worker execution remains unauthorized.

## Live Result

Final run tag:

```text
qwen35-real-load-read-only-preflight-final-20260727-224203
```

Local evidence:

```text
experiments/qwen35_hybrid_state/
  qwen35-real-load-read-only-preflight-final-20260727-224203/
    preflight.json
    source_manifest.json
```

Classification:

```text
INCOMPLETE
```

Passed checks:

```text
source_identity
remote_identity
runtime_dependencies
proc_telemetry
run_root_space
cuda_disabled
payload_zero
```

Blocked checks:

```text
model_identity
model_files
gpu0_idle
```

Observed blockers:

- the approved model directory still contains `config.json`,
  `model.safetensors.index.json`, and one `.safetensors` filename, but its
  approved `model_manifest.json` sidecar is absent;
- without that sidecar, the gate cannot bind the observed shard stat to the
  previously verified repository/revision/full-file SHA256 identity;
- GPU0 had 16 active compute-process rows at observation time.

Hard payload-zero evidence:

```text
payload_open_count: 0
payload_bytes_read: 0
payload_hashes_recomputed: false
payload_zero: true
```

Source-tree SHA256:

```text
4d9d6f724f3ba1dc967533563f8efd7d7a88626a9a91000f627deca7d6eae50d
```

The remote run directory contains exactly the five staged source files plus
the two authorized JSON artifacts. Both JSON artifacts were written
atomically on the remote host, read back through SSH, compared to the local
records, and persisted locally.

The worker remains unimplemented and unauthorized. No real checkpoint load,
model construction, CUDA initialization, assignment, forward, or inference
occurred.
