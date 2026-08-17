# Qwen3.5 Authorized Prepared-Target Loader Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an explicit authorization-bound callable that creates a fresh CPU Qwen3.5 prepared target and delegates it to the existing streamed checkpoint loader.

**Architecture:** Validate the bounded request and adapter authorization before creating a target. Require an exact fresh CPU prepared target, consume it once through the existing streamed loader, and return the exact loaded-candidate type already accepted by ModelRunner.

**Tech Stack:** Python, PyTorch CPU/meta contracts, existing Qwen3.5 target factory, request contract, and streamed checkpoint loader.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not modify or execute `tools/qwen35_real_checkpoint_load_worker.py`.
- No Engine wiring, `LLMEngine.step()` integration, CUDA, SSH, or remote run.
- Do not change ModelRunner's exact `Qwen35LoadedCheckpointCandidate` contract.
- Preserve schema-v2 canonical `NO_GO`.
- Do not stage, commit, merge, or clean experiment evidence.
- Do not claim production memory or speed benefit.

---

### Task 1: Adapter Contract and Pre-Delegation Safety

**Files:**
- Create: `tinyvllm/models/qwen35_checkpoint_candidate_loader.py`
- Create: `tools/test_qwen35_checkpoint_candidate_loader.py`

**Interfaces:**
- Consumes: `Qwen35CheckpointCandidateLoadRequest`, `Qwen35PreparedCheckpointCandidateTarget`, and `load_qwen35_fresh_checkpoint_candidate(...)`.
- Produces: `Qwen35AuthorizedCheckpointCandidateLoader` and `build_qwen35_authorized_checkpoint_candidate_loader(...)`.

- [x] Write RED tests for builder validation, authorization-before-provider, exact target type, CPU-only target, and exact request forwarding.
- [x] Run the focused test and confirm the module is missing.
- [x] Implement the minimal frozen callable adapter.
- [x] Run the focused test and confirm GREEN.

### Task 2: Fresh-Retry and Delegation Semantics

**Files:**
- Modify: `tools/test_qwen35_checkpoint_candidate_loader.py`

**Interfaces:**
- Consumes: the adapter implemented in Task 1.
- Produces: evidence that every call receives a fresh target and delegated failure cannot reuse a consumed/partially assigned target.

- [x] Add RED tests for delegated failure followed by fresh-target retry and exact candidate passthrough.
- [x] Implement only the behavior required by the new tests.
- [x] Run the focused suite and confirm GREEN.

### Task 3: Regression, Audit, and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan

**Interfaces:**
- Consumes: completed adapter and existing loader/publication boundaries.
- Produces: fresh verification evidence and the next explicit gap.

- [x] Run prepared-target, streamed/tiled loader, ModelRunner authorized-loader, publication/binding, authorization, and safety-gate regressions.
- [x] Run compile, diff, staged, worker-stub, Engine, `step()`, CUDA/SSH, and adapter-publication audits.
- [x] Check all plan items and record exact evidence plus the remaining worker-execution and manifest-verification gap.

## Completion Evidence

TDD:

```text
initial focused RED:
  ModuleNotFoundError for qwen35_checkpoint_candidate_loader
final focused suite:
  passed (5 tests)
```

Fresh regressions:

```text
prepared target factory:
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
adapter CUDA/SSH/process/distributed references:
  0
adapter publication/slot/Engine references:
  0
adapter direct safe_open/open references:
  0
Engine imports or calls of the new builder/module:
  0
LLMEngine.step adapter references:
  0
real worker fail-closed raise:
  preserved
```

Remaining gap:

```text
approved model-manifest verification:
  remains outside the runtime adapter
real worker execution:
  absent and unauthorized
Engine installation/dispatch:
  absent
real checkpoint load through this adapter:
  not executed in this gate
production CUDA/cache/speed benefit:
  unmeasured
```
