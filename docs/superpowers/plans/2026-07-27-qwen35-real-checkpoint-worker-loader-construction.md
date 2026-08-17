# Qwen3.5 Real Checkpoint Worker Loader Construction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an import-only worker function that builds a manifest-bound rank loader from already parsed checkpoint metadata while preserving hard execution rejection.

**Architecture:** Delegate metadata validation to the existing tensor-plan builder, then compose the manifest identity and rank loader configuration. Construction retains providers without invoking them; `main()` remains unchanged and fail-closed.

**Tech Stack:** Python, existing Qwen3.5 checkpoint tensor plan and manifest-bound loader configuration.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Modify the worker implementation only; do not authorize or execute it.
- No file reads, JSON parsing, safetensors access, CUDA, SSH, subprocess, Engine wiring, or publication.
- Keep `main()` as the existing hard rejection.
- Preserve schema-v2 canonical `NO_GO`.
- Do not stage, commit, merge, or clean experiment evidence.

---

### Task 1: Construction-Only Worker Function

**Files:**
- Modify: `tools/qwen35_real_checkpoint_load_worker.py`
- Create: `tools/test_qwen35_real_checkpoint_load_worker.py`

- [x] Write RED tests for tensor-plan construction, manifest/rank forwarding, provider non-invocation, exact loader return, and failure propagation.
- [x] Run the focused test and confirm the function is missing.
- [x] Implement the minimal construction function without changing `main()`.
- [x] Run the focused test and confirm GREEN.

### Task 2: Safety Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan

- [x] Run worker, loader configuration/adapter/factory, authorization, and safety-gate regressions.
- [x] Run compile, diff, staged, worker source, execution rejection, Engine, and `step()` audits.
- [x] Check plan items and record exact evidence plus remaining metadata-read/file-verification/execution gaps.

## Completion Evidence

TDD/debugging:

```text
initial focused RED:
  AttributeError because build_qwen35_real_checkpoint_rank_loader was absent
direct-script dependency RED:
  eager tinyvllm import reached missing flash_attn before main rejection
fix:
  lazy construction dependency resolution
final focused suite:
  passed (3 tests)
direct worker execution:
  exits with the existing execution-is-not-implemented RuntimeError
```

Fresh regressions:

```text
manifest-bound loader configuration:
  passed (4 tests)
prepared-target loader adapter:
  passed (5 tests)
prepared-target factory:
  passed (6 tests)
real 24-layer metadata binding:
  passed (1 test)
streamed fresh checkpoint:
  passed (4 tests)
real checkpoint authorization:
  passed
real checkpoint safety gate:
  passed (23 tests)
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
```

Static boundary evidence:

```text
focused py_compile:
  passed
git diff --check:
  passed
staged files:
  0
worker payload/safetensors references:
  0
worker CUDA/SSH/process references:
  0
worker Engine/publication references:
  0
worker CLI parsing references:
  0
worker main hard rejection:
  preserved
Engine worker-builder references:
  0
LLMEngine.step worker-builder references:
  0
```

Remaining gap:

```text
metadata JSON/header reading:
  absent
runtime digest verification:
  absent
real payload loading:
  not executed
worker execution:
  unauthorized
production attention backend:
  caller supplied
production CUDA/cache/speed benefit:
  unmeasured
```
