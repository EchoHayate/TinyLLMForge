# Qwen3.5 Manifest-Bound Rank Loader Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construct a request-bound local checkpoint loader from immutable manifest identities and fresh rank-specific CPU target providers.

**Architecture:** A frozen manifest identity retains the normalized checkpoint path and verified metadata digests. A frozen rank configuration validates topology/providers without allocation, then builds a callable that rejects request conflicts before creating exactly one fresh pool and delegating to the prepared-target loader adapter.

**Tech Stack:** Python frozen dataclasses, existing bounded request, hybrid-state pool, prepared-target factory, and authorized loader adapter.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- No metadata or checkpoint file reads in configuration code.
- No imports from `tools/`.
- No Engine wiring, `LLMEngine.step()` integration, CUDA, SSH, or remote run.
- Every load invocation must create exactly one fresh pool.
- Preserve schema-v2 canonical `NO_GO`.
- Do not modify or execute the real checkpoint worker.
- Do not stage, commit, merge, or clean experiment evidence.

---

### Task 1: Frozen Identity and Configuration Validation

**Files:**
- Create: `tinyvllm/models/qwen35_checkpoint_loader_configuration.py`
- Create: `tools/test_qwen35_checkpoint_loader_configuration.py`

- [x] Write RED tests for normalized path, canonical SHA256, exact tensor plan, TP range, callable providers, and allocation-free builder.
- [x] Run the focused test and confirm the module is missing.
- [x] Implement the minimal frozen identity/configuration values and builder.
- [x] Run the focused test and confirm GREEN.

### Task 2: Request Binding and Fresh Rank Target

**Files:**
- Modify: `tools/test_qwen35_checkpoint_loader_configuration.py`

- [x] Add RED tests for pre-pool request conflict rejection, one fresh pool per invocation, exact argument forwarding, CPU forcing, candidate passthrough, and failure retry freshness.
- [x] Implement the callable manifest-bound loader.
- [x] Run the focused test and confirm GREEN.

### Task 3: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan

- [x] Run adapter/factory, streamed/tiled, ModelRunner publication/binding, authorization, and safety-gate regressions.
- [x] Run compile, diff, staged, file-read, tools-import, pool-count, Engine, `step()`, CUDA/SSH, and worker-stub audits.
- [x] Check plan items and record exact evidence plus remaining file-verification and worker-execution gaps.

## Completion Evidence

TDD:

```text
initial focused RED:
  ModuleNotFoundError for qwen35_checkpoint_loader_configuration
final focused suite:
  passed (4 tests)
```

Fresh regressions:

```text
prepared-target loader adapter:
  passed (5 tests)
prepared-target factory:
  passed (6 tests)
real 24-layer metadata binding:
  passed (1 test)
streamed fresh checkpoint:
  passed (4 tests)
tiled checkpoint loading:
  passed (5 tests)
policy tiled checkpoint loading:
  passed (3 tests)
ModelRunner authorized checkpoint loader:
  passed (7 tests)
ModelRunner published candidate binding:
  passed (4 tests)
Engine all-rank candidate binding:
  passed (9 tests)
hybrid model publication:
  passed (2 tests)
native model owner binding:
  passed (13 tests)
bounded checkpoint worker request:
  passed (4 tests)
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
configuration file-read/hash imports:
  0
configuration tools imports:
  0
configuration CUDA/SSH/process references:
  0
configuration direct HybridStateTensorPool allocations:
  0
Engine configuration references:
  0
LLMEngine.step configuration references:
  0
real worker fail-closed raise:
  preserved
```

Remaining gap:

```text
runtime verification of config/index/header digests:
  absent
production attention-backend provider:
  caller supplied, not yet constructed by worker
real worker implementation:
  still a fail-closed stub
real worker execution:
  unauthorized
production CUDA/cache/speed benefit:
  unmeasured
```
