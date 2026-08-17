# Generic Native Speculative Runtime Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a callback-driven, model-independent native speculative step that owns plan, transaction, acceptance, commit, rollback, and phase-labelled failures.

**Architecture:** Create `tinyvllm/speculative/runtime.py` with immutable result types and one orchestration function. Model/CUDA-specific preparation and forward execution remain injected callbacks.

**Tech Stack:** Python 3, dataclasses, callables, existing verifier and BlockManager transaction APIs.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- No model-name branches, scheduler integration, MTP model, or KV-offload behavior.
- No branch switch, stage, commit, stash, reset, push, or `git clean`.
- TDD RED before production implementation.
- Preserve accepted KV direct commit and rejected suffix rollback.
- No performance claims.

---

### Task 1: RED Runtime Contract

**Files:**
- Create: `tools/test_speculative_runtime.py`
- Create after RED: `tinyvllm/speculative/runtime.py`

- [x] Test plan/proxy-table callback arguments and callback order.
- [x] Test K=1 skips tail callbacks and marks zero.
- [x] Test zero/one/partial/full acceptance, EOS, and output budget.
- [x] Test callback target validation.
- [x] Test rollback phases for first target, prepare, tail, materialize, and
  commit failures.
- [x] Test rollback failure preserves the original cause.
- [x] Run and witness RED because the runtime module does not exist.

### Task 2: GREEN Runtime Core

**Files:**
- Create: `tinyvllm/speculative/runtime.py`

- [x] Add `NativeTailResult`.
- [x] Add `NativeSpeculativeStepResult`.
- [x] Add `NativeSpeculativeStepError`.
- [x] Add private token/callback validation.
- [x] Implement callback ordering and verifier plan construction.
- [x] Implement transaction begin, materialization, commit, and rollback.
- [x] Implement greedy prefix acceptance and truncation.
- [x] Run focused tests and confirm GREEN.

### Task 3: Profiler Integration

**Files:**
- Modify: `tools/profile_ngram_commit.py`
- Modify: `tools/test_ngram_speculative.py`

- [x] Adapt native preparation/forward into runtime callbacks.
- [x] Build the existing event from `NativeSpeculativeStepResult`.
- [x] Preserve timing/debug/oracle fields.
- [x] Keep legacy mode unchanged.
- [x] Run complete speculative, transaction, and chunked-prefill regressions.
- [x] Update handoff with exact evidence and remaining batch-runtime gap.

## Completion Evidence

```text
python3 -m pytest -q tools/test_speculative_runtime.py
14 passed

python3 -m pytest -q tools/test_ngram_speculative.py
59 passed

python3 -m pytest -q tools/test_speculative_kv_transaction.py
25 passed

PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.12 \
  tools/test_chunked_prefill.py
chunked prefill tests passed
```

The implementation is complete for the scoped generic core and profiler
adapter. Production scheduler batching, learned/MTP proposal adapters, KV
residency integration, and the full promotion matrix remain separate gates.
