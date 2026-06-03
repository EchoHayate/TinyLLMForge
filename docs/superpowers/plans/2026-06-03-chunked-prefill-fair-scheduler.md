# Chunked Prefill Fair Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small scheduler policy that prevents chunked prefill-first mode from monopolizing the engine by yielding to decode after a configurable number of consecutive prefill chunks.

**Architecture:** Keep pure prefill / pure decode batches. Add one config knob and one scheduler counter: when chunked prefill is enabled, `chunked_prefill_decode_first=False`, running decode exists, and the prefill chunk streak reaches the threshold, schedule one decode batch and reset the streak. Reuse the existing latency profiler to compare default, prefill-first, decode-first, and balanced modes.

**Tech Stack:** Python scheduler logic, existing script tests, `tools/profile_chunked_prefill.py`.

---

## File Structure

- Modify `tinyvllm/config.py`: add `chunked_prefill_max_consecutive_chunks`.
- Modify `tinyvllm/engine/scheduler.py`: track consecutive prefill chunks and yield decode when threshold is reached.
- Modify `tools/test_chunked_prefill.py`: add fairness policy regression test.
- Modify `tools/profile_chunked_prefill.py`: add `--max-consecutive-prefill-chunks` CLI knob.
- Modify `docs/qwen3-8b-fixes.md`: add §38 with results.

## Task 1: Scheduler fairness policy

**Files:**
- Modify: `tools/test_chunked_prefill.py`
- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/scheduler.py`

- [ ] **Step 1: Write failing test**

Add a test where one running decode seq and one long prefilling seq coexist. With `chunked_prefill_decode_first=False` and `chunked_prefill_max_consecutive_chunks=2`, the scheduler should return two prefill chunks, then one decode batch, then resume prefill.

- [ ] **Step 2: Run RED**

Run: `python3 tools/test_chunked_prefill.py`

Expected: fail because `chunked_prefill_max_consecutive_chunks` is not implemented.

- [ ] **Step 3: Implement config and scheduler counter**

Add the config field with default `0` (disabled). In Scheduler, add `_consecutive_prefill_chunks`; increment on scheduled prefill; reset on scheduled decode; if threshold is reached and `running` is non-empty, schedule decode first.

- [ ] **Step 4: Run GREEN**

Run: `python3 tools/test_chunked_prefill.py`

Expected: pass.

## Task 2: Profiler balanced mode support

**Files:**
- Modify: `tools/profile_chunked_prefill.py`
- Test: `tools/test_profile_chunked_prefill.py`

- [ ] **Step 1: Add CLI knob**

Add `--max-consecutive-prefill-chunks`, default `0`, and pass it into `LLM(...)` when `--mode chunked`.

- [ ] **Step 2: Verify local tests**

Run: `python3 -m py_compile tools/profile_chunked_prefill.py && python3 tools/test_profile_chunked_prefill.py`.

## Task 3: Remote smoke and docs

**Files:**
- Modify: `docs/qwen3-8b-fixes.md`

- [ ] **Step 1: Run remote balanced profile**

Run Qwen3-0.6B with `--mode chunked --max-num-prefill-tokens-per-step 128 --max-consecutive-prefill-chunks 1` and compare against §37.

- [ ] **Step 2: Add §38**

Document the policy, metrics, and conclusion.

- [ ] **Step 3: Commit**

Commit message: `研究：添加 chunked prefill 公平调度策略`

## Self-Review

- Spec coverage: covers scheduler policy, profiler support, remote smoke, and docs.
- Placeholder scan: no placeholders remain.
- Scope check: intentionally avoids mixed prefill+decode kernel changes; this is a scheduler-only policy experiment.
