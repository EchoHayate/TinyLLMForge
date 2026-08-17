# Qwen3.5 Real Checkpoint Load Independent Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-only fail-closed verifier for Qwen3.5 real-checkpoint-load artifacts with synthetic `GO`, `NO_GO`, and tamper coverage.

**Architecture:** Load the frozen real-load contract dynamically, validate local artifacts in inventory/provenance/process/telemetry/case layers, and reuse `classify_case_rows` only after all evidence is independently consistent. Atomically write verifier-owned outputs without changing the worker input inventory.

**Tech Stack:** Python 3 standard library, JSON/JSONL, SHA256, dependency-light executable tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not SSH, open checkpoint payloads, implement or start the worker, or inspect GPUs live.
- Do not modify production tile policy, ModelRunner, Engine, Scheduler, publication, or schema-v2 canonical evidence.
- Do not stage, commit, merge, or delete untracked experiment evidence.
- The verifier must not import Torch, Transformers, safetensors, or production TinyLLMForge modules.
- Synthetic fixtures prove verifier behavior only and authorize no speed, cache, memory, compression, or quality claim.

---

### Task 1: Synthetic Complete Classification Fixtures

**Files:**
- Create: `tools/test_verify_qwen35_real_checkpoint_load_gate.py`
- Read: `tools/qwen35_real_checkpoint_load_contract.py`

**Interfaces:**
- Consumes: `contract.REQUIRED_ARTIFACTS`, `contract.CASE_ORDER_MIB`, and `contract.classify_case_rows(rows)`.
- Produces: `write_complete_run(path: Path, classification: str) -> Path` and executable tests for `verifier.verify_run`.

- [x] **Step 1: Write a complete synthetic `GO` fixture test**

Create canonical JSON, JSONL, worker logs, and SHA256 manifest entries. Assert
six rows classify as `GO`, outputs are written, and repeated verification is
stable.

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
python3 tools/test_verify_qwen35_real_checkpoint_load_gate.py
```

Expected: failure because the verifier placeholder has no `verify_run`.

- [x] **Step 3: Add a complete synthetic `NO_GO` fixture**

Use identical correctness evidence but set the 16 MiB median wall time below
the frozen five-percent improvement threshold. Assert `NO_GO`, not
`INCOMPLETE`.

- [x] **Step 4: Run the test and retain the expected RED**

Run the same command and confirm the missing verifier API remains the cause.

### Task 2: Inventory And Provenance Verifier

**Files:**
- Modify: `tools/verify_qwen35_real_checkpoint_load_gate.py`
- Test: `tools/test_verify_qwen35_real_checkpoint_load_gate.py`

**Interfaces:**
- Produces: `verify_run(run_dir: Path | str, write_report: bool = False) -> dict`.
- Internal helpers: `_read_json`, `_read_jsonl`, `_sha256`, `_verify_inventory`, `_verify_source`, `_verify_model`, `_verify_preflight`, `_verify_environment`.

- [x] **Step 1: Add RED tamper tests**

Cover unlisted files, hash mismatch, unsafe paths, dirty source, source hash
disagreement, model revision/config/index/manifest mismatch, non-`READY`
preflight, and environment CUDA use.

- [x] **Step 2: Run tests and verify intended failures**

Expected: each new case fails because the corresponding guard is absent.

- [x] **Step 3: Implement minimal inventory and provenance validation**

Use only standard-library readers and hash functions. Treat all malformed or
contradictory provenance as `INCOMPLETE`.

- [x] **Step 4: Run tests and verify GREEN**

Expected: complete `GO`/`NO_GO` fixtures and all provenance tamper tests pass.

### Task 3: Process, GPU, Telemetry, And Case Verification

**Files:**
- Modify: `tools/verify_qwen35_real_checkpoint_load_gate.py`
- Test: `tools/test_verify_qwen35_real_checkpoint_load_gate.py`

**Interfaces:**
- Internal helpers: `_verify_processes`, `_verify_gpu_processes`, `_verify_telemetry`, `_verify_summary`.
- Reuses: `contract.classify_case_rows(case_rows) -> dict`.

- [x] **Step 1: Add RED process and GPU tests**

Cover missing worker, nonzero return code, timeout, non-empty CUDA visibility,
CUDA initialization/allocation, non-empty before/after GPU process lists, and
missing worker logs.

- [x] **Step 2: Add RED telemetry and case tests**

Cover missing/extra/duplicate telemetry, metric disagreement, case
count/order/repeat/budget mismatch, tile peak overflow, assigned/source count,
destination byte, digest, handle, and summary disagreement.

- [x] **Step 3: Run tests and verify RED**

Expected: failures identify the missing semantic guards.

- [x] **Step 4: Implement process, GPU, telemetry, case, and summary guards**

Validate evidence independently, call the frozen classifier only after all
guards pass, and preserve `NO_GO` for complete threshold misses.

- [x] **Step 5: Run tests and verify GREEN**

Expected: all complete and tampered fixture tests pass.

### Task 4: CLI And Atomic Output Persistence

**Files:**
- Modify: `tools/verify_qwen35_real_checkpoint_load_gate.py`
- Test: `tools/test_verify_qwen35_real_checkpoint_load_gate.py`

**Interfaces:**
- CLI: `python3 tools/verify_qwen35_real_checkpoint_load_gate.py --run-dir PATH [--write-report]`.
- Outputs: `independent_verification.json`, `report.md`.

- [x] **Step 1: Add RED CLI/output tests**

Assert JSON stdout, exit code zero for `GO` and `NO_GO`, atomic output files,
claim boundary text, and repeat-run input inventory stability.

- [x] **Step 2: Run tests and verify RED**

Expected: CLI or persistence assertions fail before implementation.

- [x] **Step 3: Implement CLI and atomic writes**

Write `.partial` siblings and replace atomically. Exclude verifier-owned
outputs from worker inventory comparison.

- [x] **Step 4: Run tests and verify GREEN**

Expected: the executable test script reports all verifier tests passed.

### Task 5: Handoff And Completion Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-07-27-qwen35-real-checkpoint-load-independent-verifier.md`

**Interfaces:**
- Produces: one unique canonical EOF handoff section.

- [x] **Step 1: Run focused validation**

```bash
python3 tools/test_verify_qwen35_real_checkpoint_load_gate.py
python3 tools/test_qwen35_real_checkpoint_load_safety_gate.py
python3 -m py_compile \
  tools/qwen35_real_checkpoint_load_contract.py \
  tools/verify_qwen35_real_checkpoint_load_gate.py \
  tools/test_verify_qwen35_real_checkpoint_load_gate.py
git diff --check
git diff --cached --name-only
```

Expected: verifier tests pass, existing safety tests remain at 16 passing
tests, compilation succeeds, no whitespace errors, and no staged files.

- [x] **Step 2: Mark plan checkboxes with actual evidence**

Do not mark a task complete until its fresh command output is observed.

- [x] **Step 3: Replace the canonical EOF handoff heading**

Record files, test counts, claim boundary, current GPU0 blocker, and the next
live gate. Keep exactly one new canonical heading at true EOF.

- [x] **Step 4: Audit the long-term objective**

Keep the goal active unless real inference speed, production cache/memory
reduction, no-accuracy-regression evidence, and native production Qwen3.5
execution are all established.
