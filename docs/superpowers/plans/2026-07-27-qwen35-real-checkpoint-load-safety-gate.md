# Qwen3.5 Real Checkpoint Load Safety Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed source-bound contract and dry-run remote harness for a future real Qwen3.5 8-versus-16 MiB checkpoint load comparison.

**Architecture:** Freeze the experiment and artifact schemas in a dependency-light contract module. Add a non-destructive runner that validates exact remote/model identities, builds source-bound commands, and emits a local dry-run plan without SSH or payload access; future worker/verifier execution remains explicitly gated.

**Tech Stack:** Python 3.12 standard library, JSON/JSONL, SHA256, argparse, subprocess command construction, existing Qwen3.5 remote-runner conventions.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; do not use subagents.
- Do not stage, commit, merge, or delete experiment evidence.
- Do not SSH or open real checkpoint payloads in this implementation session.
- Remote target is exactly `sitian@10.232.195.203`.
- Future worker uses CPU only with empty `CUDA_VISIBLE_DEVICES`.
- Do not change production tile policy, loader, ModelRunner, Engine, or Scheduler.
- Do not claim real-load or inference improvement.

---

### Task 1: Add Contract and Runner RED Tests

**Files:**
- Create: `tools/test_qwen35_real_checkpoint_load_safety_gate.py`

- [x] **Step 1: Assert frozen identities, matrix, thresholds, artifacts**

- [x] **Step 2: Assert preflight and result fail-closed validation**

- [x] **Step 3: Assert non-destructive SSH/run command construction**

- [x] **Step 4: Assert dry-run JSON persistence without subprocess/SSH**

- [x] **Step 5: Run RED for missing modules**

### Task 2: Implement Dependency-Light Contract

**Files:**
- Create: `tools/qwen35_real_checkpoint_load_contract.py`

- [x] **Step 1: Define schema, identities, matrix, thresholds, artifacts**

- [x] **Step 2: Validate source/model/environment preflight records**

- [x] **Step 3: Validate exact case rows and telemetry records**

- [x] **Step 4: Classify complete synthetic verifier fixtures**

- [x] **Step 5: Run contract GREEN tests**

### Task 3: Implement Non-Destructive Remote Runner Dry-Run

**Files:**
- Create: `tools/run_qwen35_real_checkpoint_load_gate_remote.py`

- [x] **Step 1: Define exact owned source and path identities**

- [x] **Step 2: Build safe SSH and remote command arguments**

- [x] **Step 3: Build source manifest and dry-run execution plan**

- [x] **Step 4: Add atomic local JSON persistence**

- [x] **Step 5: Reject execution modes in this unimplemented session**

- [x] **Step 6: Run full GREEN tests**

### Task 4: Local Dry-Run Evidence and Handoff

**Files:**
- Create: `experiments/qwen35_hybrid_state/20260727-real-checkpoint-load-safety-dry-run.json`
- Modify: `docs/superpowers/plans/2026-07-27-qwen35-real-checkpoint-load-safety-gate.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] **Step 1: Generate local dry-run artifact**

- [x] **Step 2: Verify no SSH/subprocess/payload access occurred**

- [x] **Step 3: Run tests, py_compile, diff/staged/EOF checks**

- [x] **Step 4: Complete plan and append unique EOF canonical handoff**

