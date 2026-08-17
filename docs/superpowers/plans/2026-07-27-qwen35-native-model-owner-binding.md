# Qwen3.5 Native Model Owner Binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind a real packed Qwen3.5 model stack's exact existing state transaction and pool into ModelRunner without allocating duplicate state tensors.

**Architecture:** A pure owner factory derives transaction, pool, and runtime bridge from the exact packed stack. ModelRunner accepts only an owner whose model is its current model, installs the exact bridge once, and leaves snapshot-cache/coordinator setup to the existing acknowledged owner factory.

**Tech Stack:** Python 3.9, PyTorch CPU tensors, packed Qwen3.5 layer stack, existing HybridStateRuntimeBridge, dependency-light AST/method tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not stage, commit, merge, or delete experiment evidence.
- CPU/static only; no CUDA, NCCL, checkpoint, local GPU, or remote GPU work.
- Never allocate a second state pool.
- Do not change ModelRunner model selection in this phase.
- Do not enable Scheduler admission or change `LLMEngine.step()`.
- Preserve existing Qwen3 and Qwen3.5 math.
- Preserve the immutable Qwen3.5 schema-v2 canonical `NO_GO`.
- Do not claim performance, cache, memory, compression, or quality benefit.

---

### Task 1: RED Pure Model Owner Factory

**Files:**
- Create: `tools/test_qwen35_native_model_owner_binding.py`
- Create after RED: `tinyvllm/engine/qwen35_hybrid_model_owner.py`

- [x] Write exact model/transaction/pool/storage identity tests.
- [x] Write invalid-model and graph-coherence rejection tests.
- [x] Run focused tests and confirm RED because the owner module is absent.
- [x] Implement the immutable owner and pure factory.
- [x] Run focused factory tests and confirm GREEN.

### Task 2: RED ModelRunner Binding

**Files:**
- Modify: `tools/test_qwen35_native_model_owner_binding.py`
- Modify after RED: `tinyvllm/engine/model_runner.py`

- [x] Add current-model identity, exact bridge, and retained-owner tests.
- [x] Add identical idempotence and different-owner/bridge/pool rejection tests.
- [x] Add convenience identity-row and non-Qwen3.5 fail-closed tests.
- [x] Run focused tests and confirm RED for missing binding methods.
- [x] Import owner types, initialize owner field, and implement binding methods.
- [x] Run focused tests and confirm GREEN.

### Task 3: Regression and Completion Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

- [x] Run focused tests under Python 3.9 and 3.12.
- [x] Run owner factory/live transaction/restore/ack protocol tests.
- [x] Run packed stack and Qwen3.5/hybrid CPU regressions.
- [x] Run chunked-prefill 97/1/0 matrix.
- [x] Run Python 3.9/3.12 `py_compile` and `git diff --check`.
- [x] Confirm staged files empty and experiment evidence present.
- [x] Audit spec requirements and claim boundaries.
- [x] Update handoff with model selection/checkpoint/startup gates still blocked.
- [x] Mark only freshly verified checkboxes complete.

## Completion Audit

The audit added exact-type and forged-owner coverage after the initial GREEN:

- derived packed-stack, transaction, and owner types fail closed;
- a hand-constructed owner whose model, transaction, pool, or runtime bridge
  does not form one identity-coherent graph fails before mutation;
- the pure factory still reuses the exact existing pool storage and allocates
  no second state tensor pool.

Fresh focused validation:

```text
Python 3.9:
qwen35 native model owner binding tests passed (10 tests)

Python 3.12:
qwen35 native model owner binding tests passed (10 tests)
```

The production `ModelRunner` still constructs `Qwen3ForCausalLM`; neither
`LLMEngine.step()` nor Scheduler calls either binding method. The Scheduler
guard remains unchanged, so native model selection, checkpoint loading,
startup binding/configuration, Scheduler admission, GPU correctness, and all
performance/cache/memory/quality claims remain blocked.

