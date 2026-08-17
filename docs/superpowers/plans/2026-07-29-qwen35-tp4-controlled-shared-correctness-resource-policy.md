# Qwen3.5 TP4 Controlled-Shared Correctness Resource Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a manifest-bound controlled-shared GPU policy for the real TP4 correctness campaign while preserving strict-exclusive correctness and benchmark behavior.

**Architecture:** Introduce one focused resource-policy module that captures, validates, renders, and verifies GPU/process baselines. Thread its immutable identity through executor configuration, all correctness child plans, authorizations, receipts, and campaign preparation; leave benchmark code on the existing strict-exclusive path.

**Tech Stack:** Python standard library, canonical JSON/SHA256, `nvidia-smi`, `/proc/<pid>/stat`, existing authority protocols, CPU-only unit tests, real SSH/GPU execution after local gates.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, merge, create a branch, or open a PR.
- Remote target is exactly `sitian@10.232.195.203`.
- Execution environment is exactly `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- Use GPU indices `2,4,5,6` only for controlled-shared correctness.
- Do not kill, pause, reprioritize, or modify remote processes.
- Preserve the existing strict-exclusive resource protocol.
- Keep all benchmark plans strict-exclusive and unauthorized until clean resources exist.
- Child authorities execute strictly serially: root-logit, cached continuation, Engine correctness.
- Do not overwrite the existing `20260729-132532` READY preparation.
- Do not claim performance, cache, memory, compression, quality, or accuracy benefit before real evidence passes.

---

### Task 1: Controlled-Shared Resource Policy Core

**Files:**
- Create: `tools/qwen35_tp4_correctness_resource_policy.py`
- Create: `tools/test_qwen35_tp4_correctness_resource_policy.py`

**Interfaces:**
- Produces `validate_baseline_manifest(path, *, ssh_target, gpu_indices)`.
- Produces `capture_command(gpu_indices)`.
- Produces `guard_command(policy, gpu_indices, baseline_path=None, baseline_sha256=None)`.
- Produces `validate_guard_payload(policy, payload, *, gpu_indices, baseline=None)`.

- [ ] **Step 1: Write failing tests**

Cover canonical strict output, valid controlled baseline, disappearing
baseline process, new process rejection, PID reuse, GPU UUID drift, low free
memory, malformed `/proc` identity, symlink baseline, and closed schemas.

- [ ] **Step 2: Run the policy suite and require RED**

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_correctness_resource_policy.py
```

Expected: missing-module `FileNotFoundError`.

- [ ] **Step 3: Implement minimal policy module**

Use canonical JSON, exact schemas, regular-file checks, SHA256, and shell
commands containing no mutation or process-control surface.

- [ ] **Step 4: Run the policy suite and py_compile**

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 tools/test_qwen35_tp4_correctness_resource_policy.py
python3 -m py_compile \
  tools/qwen35_tp4_correctness_resource_policy.py \
  tools/test_qwen35_tp4_correctness_resource_policy.py
```

Expected: all tests pass.

### Task 2: Configuration and Child Plan Binding

**Files:**
- Modify: `tools/build_qwen35_tp4_engine_authority_configuration.py`
- Modify: `tools/qwen35_tp4_engine_correctness_executor.py`
- Modify: `tools/qwen35_tp4_engine_remote_execution_plan.py`
- Modify: `tools/qwen35_tp4_cached_continuation_remote_execution_plan.py`
- Modify: `tools/qwen35_tp4_root_logit_remote_execution_plan.py`
- Modify corresponding `tools/test_*.py` files.

**Interfaces:**
- Executor configuration adds `resource_policy`, `resource_baseline_path`,
  and `resource_baseline_sha256`.
- Child plans bind the same policy and baseline identity.

- [ ] **Step 1: Add failing configuration and plan tests**

Require strict defaults to remain unchanged and controlled-shared plans to
bind the baseline, selected GPUs, UUIDs, guard command, and
`benchmark_execution_authorized=false`.

- [ ] **Step 2: Run focused tests and require RED**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  tools/test_build_qwen35_tp4_engine_authority_configuration.py
PYTHONDONTWRITEBYTECODE=1 python3 \
  tools/test_qwen35_tp4_root_logit_remote_execution_plan.py
PYTHONDONTWRITEBYTECODE=1 python3 \
  tools/test_qwen35_tp4_cached_continuation_remote_execution_plan.py
PYTHONDONTWRITEBYTECODE=1 python3 \
  tools/test_qwen35_tp4_engine_remote_execution_plan.py
```

Expected: failures for missing policy fields and unsupported builder inputs.

- [ ] **Step 3: Implement minimal configuration and plan threading**

Delegate resource command construction to the policy module. Do not change
benchmark plan construction.

- [ ] **Step 4: Run focused tests and py_compile**

Expected: all focused tests pass.

### Task 3: Authorization and Receipt Binding

**Files:**
- Modify: `tools/qwen35_tp4_root_logit_remote_execution_authorization.py`
- Modify: `tools/qwen35_tp4_engine_remote_execution_authorization.py`
- Modify: `tools/qwen35_tp4_root_logit_remote_execution_receipt.py`
- Modify: `tools/qwen35_tp4_cached_continuation_remote_execution_receipt.py`
- Modify: `tools/qwen35_tp4_engine_remote_execution_receipt.py`
- Modify corresponding `tools/test_*.py` files.

**Interfaces:**
- Authorizations freeze policy and baseline SHA.
- Receipts independently reopen the baseline and validate both initial and
  final resource observations.

- [ ] **Step 1: Add failing authorization and receipt tests**

Reject policy drift, baseline drift, GPU/UUID drift, new process, PID reuse,
low memory, missing final recheck, and benchmark authorization.

- [ ] **Step 2: Run focused tests and require RED**

Run the five authorization/receipt suites. Expected: failures for missing
controlled-shared bindings.

- [ ] **Step 3: Implement minimal binding and validation**

Reuse the policy module; do not duplicate process-set validation.

- [ ] **Step 4: Run focused tests and py_compile**

Expected: all focused tests pass.

### Task 4: Preparation and Campaign Propagation

**Files:**
- Modify: `tools/qwen35_tp4_correctness_authority_campaign_preparation.py`
- Modify: `tools/qwen35_tp4_correctness_authority_campaign_plan.py`
- Modify: `tools/qwen35_tp4_correctness_authority_campaign_authorization.py`
- Modify: `tools/qwen35_tp4_correctness_authority_campaign_receipt.py`
- Modify corresponding `tools/test_*.py` files.

**Interfaces:**
- Preparation copies `resource_baseline.json` into `inputs/`.
- Manifest, child rows, campaign plan, authorization, and receipt bind policy
  and baseline SHA.

- [ ] **Step 1: Add failing preparation/campaign tests**

Require all three child policies to match, controlled-shared baseline
identity to match configuration, and benchmark authorization to remain false.

- [ ] **Step 2: Run focused tests and require RED**

Expected: failures for missing propagation.

- [ ] **Step 3: Implement minimal propagation**

Preserve the existing READY bundle and create only new timestamped outputs.

- [ ] **Step 4: Run focused tests and py_compile**

Expected: all focused tests pass.

### Task 5: Local Regression and Safety Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`

- [ ] **Step 1: Run focused resource and campaign suites**

Run all files touched by Tasks 1-4.

- [ ] **Step 2: Run the expanded 51-file authority gate**

Expected: no regressions from the prior 328-test baseline.

- [ ] **Step 3: Audit forbidden surfaces**

Verify controlled-shared commands contain no `kill`, `pkill`, `renice`,
process mutation, benchmark authorization, or strict benchmark changes.

- [ ] **Step 4: Run `git diff --check`**

Expected: PASS.

### Task 6: Capture Real Baseline and Publish New READY Preparation

**Files:**
- Create: timestamped baseline/configuration/preparation directories under
  `experiments/qwen35_hybrid_state/`.
- Modify: `AGENT_HANDOFF_STATE.md`

- [ ] **Step 1: Re-enumerate GPU `2,4,5,6`**

Require at least 24 GiB free on each selected GPU.

- [ ] **Step 2: Capture and locally validate the immutable baseline**

Bind GPU UUIDs, process names, PIDs, start times, used memory, and SHA256.

- [ ] **Step 3: Build a fresh remote configuration**

Use fresh collision-free `TINYVLLM_DIST_PORT` and `MASTER_PORT`.

- [ ] **Step 4: Build and verify a fresh campaign preparation**

Require `classification=READY`, all authorizations unconsumed, no future
runtime outputs, and `benchmark_execution_authorized=false`.

### Task 7: Execute Real Correctness Campaign

**Files:**
- Create: timestamped runtime, receipt, adapter, and bundle evidence.
- Modify: `AGENT_HANDOFF_STATE.md`

- [ ] **Step 1: Run root-logit authority**

Require initial controlled-shared guard PASS, real authority PASS, final
guard PASS, receipt PASS, and independent verification PASS.

- [ ] **Step 2: Run cached-continuation authority**

Require the same chain without overlap with Task 7 Step 1.

- [ ] **Step 3: Run Engine correctness authority**

Require the same chain without overlap with prior children.

- [ ] **Step 4: Adapt authorities and build v2 prerequisite bundle**

Require exactly three independently verified authority inputs.

- [ ] **Step 5: Verify campaign receipt and v2 bundle**

Require PASS and `benchmark_execution_authorized=false`.

### Task 8: Completion Audit and Benchmark Boundary

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`

- [ ] **Step 1: Map objective requirements to real artifacts**

Record correctness receipts, source/model identities, resource policy,
baseline SHA, and v2 bundle.

- [ ] **Step 2: State what correctness proves**

Only claim accuracy/correctness properties directly covered by the real
authority suites.

- [ ] **Step 3: State what remains unproved**

Performance, throughput, cache, and physical-memory improvements remain
unmeasured until strict-exclusive benchmark resources are available.

- [ ] **Step 4: Preserve benchmark block**

Do not generate or consume benchmark authorization from controlled-shared
evidence.
