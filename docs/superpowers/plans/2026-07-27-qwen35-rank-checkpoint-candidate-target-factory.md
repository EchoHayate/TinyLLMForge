# Qwen3.5 Rank Checkpoint Candidate Target Factory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare one rank-specific, one-shot checkpoint candidate target by composing the concrete Qwen3.5 model assembly with the exact checkpoint binding plan over a caller-supplied sole state pool.

**Architecture:** Build the concrete graph on meta or CPU, bind the supplied tensor plan read-only, verify pool/TP/device identities, and return a one-shot target whose `take()` method matches the existing loader candidate-factory tuple contract.

**Tech Stack:** Python, PyTorch meta/CPU, existing Qwen3.5 concrete assembly and checkpoint binding planner.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Never construct a second hybrid-state pool.
- No checkpoint payload read, assignment, CUDA, Engine wiring, or remote run.
- Preserve schema-v2 canonical `NO_GO`.
- Do not stage, commit, merge, or clean experiment evidence.
- Do not claim production memory or speed benefit.

---

### Task 1: Compact Composition and One-Shot Contract

**Files:**
- Create: `tinyvllm/models/qwen35_checkpoint_candidate_factory.py`
- Create: `tools/test_qwen35_checkpoint_candidate_factory.py`

- [x] Write RED compact TP=1/2 meta/CPU composition, identity, pool preservation, and one-shot tests.
- [x] Run focused test and confirm missing module.
- [x] Implement minimal composition and one-shot target.
- [x] Run focused test and confirm GREEN.

### Task 2: Failure Atomicity and Real Metadata Regression

**Files:**
- Modify: `tools/test_qwen35_checkpoint_candidate_factory.py`
- Modify: `tools/test_qwen35_real_component_binding.py`

- [x] Add failure-path pool preservation tests.
- [x] Route real 24-layer metadata composition through the new factory and retain 320-entry TP=1/2 assertions.
- [x] Run focused and real metadata tests.

### Task 3: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan

- [x] Run concrete assembly, target binding, checkpoint loader, local worker boundary, and owner regressions.
- [x] Run compile, diff, staged, payload-zero, pool-allocation, Engine, and `step()` audits.
- [x] Check plan items and record exact evidence plus remaining real loader/backend gap.

## Completion Evidence

TDD:

```text
initial focused RED:
  ModuleNotFoundError for qwen35_checkpoint_candidate_factory
TP/layout RED:
  TP=1 pool was incorrectly accepted for TP=2 preparation
wider speculative-state RED:
  valid speculative_tokens=3 pool was incorrectly rejected
final focused suite:
  passed (6 tests)
real 24-layer metadata suite:
  passed (1 test, 320 bindings at TP=1/2)
```

Fresh regressions:

```text
concrete component factory:
  passed (2 tests)
checkpoint target binding:
  passed (3 tests)
streamed fresh checkpoint:
  passed (4 tests)
tiled checkpoint loading:
  passed (5 tests)
policy tiled checkpoint loading:
  passed (3 tests)
ModelRunner authorized checkpoint loader:
  passed (7 tests)
bounded checkpoint worker request:
  passed (4 tests)
ModelRunner published candidate binding:
  passed (4 tests)
Engine all-rank candidate binding:
  passed (9 tests)
hybrid model publication:
  passed (2 tests)
native model owner binding:
  passed (13 tests)
real checkpoint authorization:
  passed
real checkpoint safety gate:
  passed (23 tests)
```

Static boundary evidence:

```text
focused py_compile:
  passed
git diff --check:
  passed
staged files:
  0
factory payload/open/SSH/CUDA references:
  0
factory HybridStateTensorPool allocations:
  0
Engine factory references:
  0
LLMEngine.step factory references:
  0
real checkpoint worker:
  remains fail-closed
```

Remaining gap:

```text
production attention backend injection:
  caller responsibility, not wired by Engine
real checkpoint payload worker:
  absent and execution unauthorized
real multi-process load/publication:
  absent
production CUDA/cache/speed benefit:
  unmeasured
```
