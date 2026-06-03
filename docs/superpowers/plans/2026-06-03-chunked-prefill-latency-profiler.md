# Chunked Prefill Latency Profiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small profiler that quantifies whether Chunked Prefill v0 reduces long-prefill blocking by measuring per-step prefill/decode latency and decode gaps.

**Architecture:** Keep the engine unchanged. Add a tool that drives `LLM.add_request()` / `LLM.step()` manually, records each step's duration and token type, and summarizes prefill/decode latency percentiles plus maximum decode gap. Unit-test pure summary helpers locally; run GPU smoke remotely.

**Tech Stack:** Python stdlib, TinyLLMForge `LLM` / `SamplingParams`, JSON output, dependency-light script tests.

---

## File Structure

- Create `tools/profile_chunked_prefill.py`: CLI profiler and pure summary helpers.
- Create `tools/test_profile_chunked_prefill.py`: tests for step classification and percentile/gap summary.
- Modify `docs/qwen3-8b-fixes.md`: add §37 with profiler results.

## Task 1: Summary helpers and tests

**Files:**
- Create: `tools/profile_chunked_prefill.py`
- Create: `tools/test_profile_chunked_prefill.py`

- [ ] **Step 1: Write failing tests**

Test a list of synthetic step records: prefill positive token counts, decode negative token counts, first-token completion events, and gaps between decode steps.

- [ ] **Step 2: Run tests to verify RED**

Run: `python3 tools/test_profile_chunked_prefill.py`

Expected: fail because `tools/profile_chunked_prefill.py` does not exist.

- [ ] **Step 3: Implement minimal summary helpers**

Add `percentile(values, q)`, `summarize_steps(records)`, and JSON-friendly dict output.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python3 tools/test_profile_chunked_prefill.py`

Expected: `chunked prefill profiler tests passed`.

## Task 2: CLI profiler

**Files:**
- Modify: `tools/profile_chunked_prefill.py`

- [ ] **Step 1: Add CLI arguments**

Support `--model`, `--mode {default,chunked}`, `--long-prompt-tokens`, `--num-decode-seqs`, `--decode-prompt-tokens`, `--max-output-len`, `--max-num-prefill-tokens-per-step`, `--out-json`, and existing engine knobs.

- [ ] **Step 2: Implement manual stepping loop**

Construct an LLM, add a small batch of decode prompts plus one long prompt, then manually call `llm.step()` until finished while timing each step with `perf_counter()` and `torch.cuda.synchronize()` when CUDA is available.

- [ ] **Step 3: Print JSON summary**

Include args, step count, prefill/decode latency summaries, first output latency, max decode gap, generated sequence count, and first text previews.

- [ ] **Step 4: Verify syntax and local tests**

Run: `python3 -m py_compile tools/profile_chunked_prefill.py tools/test_profile_chunked_prefill.py && python3 tools/test_profile_chunked_prefill.py`.

## Task 3: Remote smoke and documentation

**Files:**
- Modify: `docs/qwen3-8b-fixes.md`

- [ ] **Step 1: Run remote smoke**

Run Qwen3-0.6B default and chunked modes with the same workload and record JSON outputs.

- [ ] **Step 2: Add §37**

Document what metric was tested, default vs chunked results, and limitations.

- [ ] **Step 3: Commit**

Commit message: `研究：添加 chunked prefill latency profiler`

## Self-Review

- Spec coverage: covers pure helper tests, CLI profiler, remote smoke, and documentation.
- Placeholder scan: no placeholder sections remain.
- Scope check: profiler only measures v0 behavior; it does not add mixed prefill/decode or change scheduler semantics.
