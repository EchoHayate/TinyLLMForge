# Qwen3.5 Transactional Root Causal-LM Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a complete dependency-light Qwen3.5 embedding-to-logits root shell whose recurrent state commits only after the entire logits path succeeds.

**Architecture:** Split the existing packed layer stack into non-mutating `prepare()` and explicit `commit()` phases while preserving its current `forward()` behavior. Compose embedding, staged stack execution, final norm, and lm head in a new root model whose `run_step()` commits only after valid logits exist.

**Tech Stack:** Python 3.9, PyTorch CPU tensors, existing packed Qwen3.5 stack and cross-layer state transaction, dependency-light executable tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not stage, commit, merge, or delete experiment evidence.
- CPU/static only; no CUDA, NCCL, checkpoint, local GPU, or remote GPU work.
- Never allocate a second state pool.
- Do not change production ModelRunner model selection.
- Do not add Engine startup binding, Scheduler admission, or `LLMEngine.step()` integration.
- Preserve existing Qwen3 and Qwen3.5 math and packed-stack `forward()` behavior.
- Preserve the immutable Qwen3.5 schema-v2 canonical `NO_GO`.
- Do not claim performance, cache, memory, compression, or quality benefit.

---

### Task 1: RED Staged Packed Stack

**Files:**
- Modify: `tools/test_qwen35_transactional_root_causal_lm.py`
- Modify after RED: `tinyvllm/layers/qwen35_packed_layer_stack.py`

**Interfaces:**
- Produces: `Qwen35PackedHeterogeneousLayerStack.prepare(...)`
- Produces: `Qwen35PackedHeterogeneousLayerStack.commit(...)`
- Preserves: existing `forward(...) -> torch.Tensor`

- [x] Create a CPU fixture with real pool, adapters, transaction, and packed stack.
- [x] Test that `prepare()` returns hidden states and candidates without changing pool storage values.
- [x] Test that `commit()` applies the candidates.
- [x] Test that existing `forward()` still performs prepare then commit.
- [x] Run focused tests and confirm RED for missing staged methods.
- [x] Extract the current gather/layer loop into `prepare()`, add `commit()`, and make `forward()` compose them.
- [x] Run staged-stack tests and confirm GREEN.

### Task 2: RED Transactional Root Shell

**Files:**
- Create: `tinyvllm/models/qwen35_packed.py`
- Modify: `tools/test_qwen35_transactional_root_causal_lm.py`

**Interfaces:**
- Consumes: exact `Qwen35PackedHeterogeneousLayerStack`
- Produces: `Qwen35PackedForCausalLM.run_step(...) -> tuple[torch.Tensor, torch.Tensor]`

- [x] Add exact component retention and call-order tests.
- [x] Add manual hidden/logit equivalence and input-embedding override tests.
- [x] Add state-preservation tests for embedding, layer, final-norm, and lm-head failures.
- [x] Add commit-failure rollback coverage.
- [x] Add malformed public input and component-output contract tests.
- [x] Run focused tests and confirm RED because the root module is absent.
- [x] Implement the minimal root shell with public validation and commit-after-logits ordering.
- [x] Run focused tests under Python 3.9 and 3.12 and confirm GREEN.

### Task 3: Regression and Completion Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

- [x] Run packed stack, transaction, model-owner binding, restore, and Qwen3.5 shell regressions.
- [x] Run chunked-prefill 97/1/0 matrix.
- [x] Run Python 3.9/3.12 `py_compile` for all changed/untracked Python files.
- [x] Run `git diff --check`.
- [x] Confirm staged files empty and `experiments/qwen35_hybrid_state` present.
- [x] Confirm production `ModelRunner` still constructs `Qwen3ForCausalLM`.
- [x] Confirm Engine/Scheduler contain zero automatic root-shell calls and the Scheduler guard is unchanged.
- [x] Audit every spec requirement against direct code or test evidence.
- [x] Update handoff with checkpoint/model-selection/startup/GPU/performance gates still blocked.
- [x] Mark only freshly verified checkboxes complete.

## Completion Audit

Fresh focused results:

```text
Python 3.9:
qwen35 transactional root causal lm tests passed (8 tests)

Python 3.12:
qwen35 transactional root causal lm tests passed (8 tests)
```

Direct evidence covers staged prepare/commit, exact call order, manual hidden
and logits equivalence, embedding override, failure before commit, commit
rollback, malformed public inputs, malformed component outputs, and unchanged
production selection/admission boundaries.

Compatibility evidence:

```text
CHUNKED_PREFILL_MATRIX passed=97 skipped=1 failed=0 total=98
Python 3.9/3.12 py_compile passed for 60 changed/untracked Python files
git diff --check passed
staged files: 0
experiments/qwen35_hybrid_state: present
ModelRunner Qwen3 constructor count: 1
Engine/ModelRunner/Scheduler root-shell references: 0
Scheduler guard count: 1
```

This gate does not select the shell in production and does not load a real
checkpoint. GPU/logit equivalence and every performance/cache/memory/quality
claim remain blocked.

