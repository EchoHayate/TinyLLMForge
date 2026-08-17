# Qwen3.5 Zero-Payload Model Manifest Restore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the missing immutable Qwen3.5 acquisition manifest only after non-payload hashes and shard stat metadata match the historical approved evidence.

**Architecture:** Add a narrowly authorized `restore-model-manifest` runner mode. It stages exact source, sends the canonical historical manifest as data, runs a remote dependency-light validator that never opens payloads, performs create-if-absent/conflict-reject installation, and round-trips a restore artifact before rerunning the existing preflight.

**Tech Stack:** Python standard library, JSON, SHA256, tar source staging, SSH ControlMaster, Linux stat/fsync/link semantics.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not stage, commit, merge, or delete evidence.
- Open/read/hash zero `.safetensors` payloads.
- Do not overwrite an existing remote manifest.
- Do not implement or launch the checkpoint-load worker.
- Keep `run`, `download-only`, and `verify-only` fail-closed.
- Do not claim inference speed, cache, memory, compression, accuracy, or quality improvement.

---

### Task 1: Freeze Restore Contract with TDD

**Files:**
- Modify: `tools/qwen35_real_checkpoint_load_contract.py`
- Modify: `tools/test_qwen35_real_checkpoint_load_safety_gate.py`

- [x] **Step 1: Add RED tests for canonical manifest bytes and digest**
- [x] **Step 2: Add RED tests for RESTORED/ALREADY_PRESENT/INCOMPLETE/CONFLICT**
- [x] **Step 3: Implement minimal restore-record validation**
- [x] **Step 4: Run GREEN**

### Task 2: Implement Zero-Payload Remote Restore

**Files:**
- Modify: `tools/run_qwen35_real_checkpoint_load_gate_remote.py`
- Modify: `tools/test_qwen35_real_checkpoint_load_safety_gate.py`

- [x] **Step 1: Add RED script-safety and exact-check tests**
- [x] **Step 2: Add RED orchestration and round-trip tests**
- [x] **Step 3: Implement remote validation and create-if-absent install**
- [x] **Step 4: Implement restore artifact remote/local round trip**
- [x] **Step 5: Keep all worker modes fail-closed**
- [x] **Step 6: Run GREEN, py_compile, and forbidden scan**

### Task 3: Run Live Restore and Read-Only Preflight

**Files:**
- Create: `experiments/qwen35_hybrid_state/<restore-run>/restore_model_manifest.json`
- Create: `experiments/qwen35_hybrid_state/<restore-run>/source_manifest.json`
- Create: `experiments/qwen35_hybrid_state/<preflight-run>/preflight.json`
- Create: `experiments/qwen35_hybrid_state/<preflight-run>/source_manifest.json`

- [x] **Step 1: Establish the fixed ControlMaster in one shell lifecycle**
- [x] **Step 2: Run one unique restore attempt**
- [x] **Step 3: Independently verify remote manifest bytes and payload-zero**
- [x] **Step 4: Run a new read-only preflight**
- [x] **Step 5: Record READY or exact remaining blockers**

### Task 4: Completion Audit and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-07-27-qwen35-zero-payload-model-manifest-restore.md`

- [x] **Step 1: Run all focused tests and syntax checks**
- [x] **Step 2: Validate both live artifact directories**
- [x] **Step 3: Verify forbidden modes and payload counters**
- [x] **Step 4: Verify unique canonical EOF heading and staged files zero**
- [x] **Step 5: Map objective requirements to established and missing evidence**

## Live Results

Restore run:

```text
qwen35-zero-payload-manifest-restore-20260727-225503
```

Restore classification:

```text
RESTORED
```

All restore checks passed:

```text
target_state
model_directory
non_payload_files
config_identity
index_identity
shard_inventory
payload_zero
```

The restored remote manifest SHA256 is exactly:

```text
3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
```

Payload-zero evidence:

```text
payload_open_count: 0
payload_bytes_read: 0
payload_hashes_recomputed: false
```

Post-restore preflight run:

```text
qwen35-real-load-post-restore-preflight-20260727-225531
```

The restored sidecar resolved the earlier blockers:

```text
model_identity: true
model_files: true
```

The preflight remains:

```text
INCOMPLETE
```

with exactly one failed check:

```text
gpu0_idle: false
```

An independent `nvidia-smi --id=0` query confirmed all 16 process rows carry
GPU0 UUID `GPU-57be086f-e967-c022-3832-93df4fc77bd0`; GPU0 reported about
81 GB allocated. This is real shared-host occupancy, not a filtering bug.

## Objective Completion Audit

Objective translated to concrete deliverables:

1. inference speed must improve on a real workload;
2. cache or memory footprint must decrease;
3. precision/quality must not regress;
4. results must be measured against a baseline under reproducible conditions;
5. optimized production inference path must be connected, not only synthetic
   or metadata gates.

Prompt-to-artifact mapping:

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Preserve accuracy | Existing schema-v2 canonical Qwen3.5 result remains `NO_GO`; no production candidate is promoted | Not established for a faster/cache-saving path |
| Faster inference | Synthetic tile calibration and estimator exist, but no real checkpoint-load or inference timing has run | Missing |
| Smaller cache/memory | Hybrid-state compression research evidence exists, but no production cache policy with end-to-end quality/runtime proof is connected here | Missing |
| Real reproducible benchmark | Source-bound preflight and immutable model identity are now verified; GPU0 is occupied | Blocked |
| Production integration | Worker, independent verifier, native Qwen3.5 runtime load/forward, and ModelRunner/Engine/Scheduler execution remain disconnected | Missing |

Conclusion:

```text
long-term goal: NOT ACHIEVED
current gate: COMPLETE
next blocker: obtain an idle approved GPU window, then implement and run the
CPU-only real checkpoint-load worker/independent verifier before any runtime
or performance promotion
```
