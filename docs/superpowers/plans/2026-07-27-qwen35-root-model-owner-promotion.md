# Qwen3.5 Root Model Owner Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the complete Qwen3.5 root causal-LM model the exact hybrid-state owner bound into ModelRunner.

**Architecture:** Derive the packed layer stack, transaction, pool, and runtime bridge from an exact `Qwen35PackedForCausalLM`. Preserve the existing one-shot ModelRunner binding contract while replacing the incomplete layer-stack-as-model identity.

**Tech Stack:** Python 3.9, PyTorch CPU, transactional Qwen3.5 root shell, packed layer stack, dependency-light owner/binding tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not stage, commit, merge, or delete experiment evidence.
- CPU/static only; no CUDA, NCCL, checkpoint, local GPU, or remote GPU work.
- Never allocate a second state pool.
- Do not change production ModelRunner model selection.
- Do not add Engine startup binding or Scheduler admission.
- Preserve Qwen3/Qwen3.5 math and transactional root behavior.
- Preserve the immutable schema-v2 canonical `NO_GO`.
- Do not claim performance, cache, memory, compression, or quality benefit.

---

### Task 1: RED Root Owner Factory

**Files:**
- Modify: `tools/test_qwen35_native_model_owner_binding.py`
- Modify after RED: `tinyvllm/engine/qwen35_hybrid_model_owner.py`

- [x] Build test roots from the existing packed stack fixture.
- [x] Assert exact root/stack/transaction/pool/storage identity.
- [x] Reject layer-stack-only, non-root, root-subclass, and transaction-subclass inputs.
- [x] Reject incoherent root/stack/transaction/pool graphs.
- [x] Run focused tests and confirm RED against the old stack-only factory.
- [x] Add `layer_stack` to the owner and derive all state ownership from the exact root.
- [x] Run focused factory tests and confirm GREEN.

### Task 2: RED ModelRunner Root Binding

**Files:**
- Modify: `tools/test_qwen35_native_model_owner_binding.py`
- Modify after RED: `tinyvllm/engine/model_runner.py`

- [x] Set the test runner current model to the exact root.
- [x] Cover forged owner root/stack/transaction/pool/bridge graphs.
- [x] Preserve idempotence and restore pool mismatch tests.
- [x] Update identity-row expectations to use `owner.layer_stack`.
- [x] Run focused tests and confirm RED for the old stack graph checks.
- [x] Validate the complete root ownership graph before mutation.
- [x] Run focused tests under Python 3.9 and 3.12 and confirm GREEN.

### Task 3: Regression and Completion Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

- [x] Run transactional root, packed stack, owner/restore protocol, and Qwen3.5 regressions.
- [x] Run chunked-prefill 97/1/0 matrix.
- [x] Run Python 3.9/3.12 compile for all changed/untracked Python files.
- [x] Run `git diff --check`.
- [x] Confirm staged files empty and experiment evidence present.
- [x] Confirm production ModelRunner still constructs Qwen3.
- [x] Confirm Engine/Scheduler automatic binding remains absent and guard remains.
- [x] Audit spec requirements and update handoff.
- [x] Mark only freshly verified checkboxes complete.

## Completion Audit

Fresh focused evidence:

```text
Python 3.9 and Python 3.12:
qwen35 native model owner binding tests passed (10 tests)
qwen35 transactional root causal lm tests passed (8 tests)
model runner spec_verify tests passed
```

The owner now retains the complete root and exact packed layer stack. Factory
and ModelRunner graph checks require root -> stack -> transaction -> pool ->
runtime bridge identity coherence before mutation.

Compatibility evidence:

```text
CHUNKED_PREFILL_MATRIX passed=97 skipped=1 failed=0 total=98
Python 3.9/3.12 py_compile passed for 60 changed/untracked Python files
git diff --check passed
staged files: 0
automatic Engine/Scheduler binding calls: 0
production Qwen3 constructor count: 1
Scheduler guard count: 1
experiments/qwen35_hybrid_state: present
```

Production construction, checkpoint loading, startup binding, GPU correctness,
and all performance/cache/memory/quality claims remain blocked.

