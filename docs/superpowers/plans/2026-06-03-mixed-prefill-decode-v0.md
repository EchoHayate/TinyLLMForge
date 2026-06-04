# Mixed Prefill Decode v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conservative mixed chunked-prefill + decode step that lets one prefill chunk and existing decode sequences run in the same varlen prefill forward pass.

**Architecture:** Keep the existing decode-only and prefill-only paths intact. Add an optional chunked-prefill mode that schedules mixed batches, tags each sequence with its per-step role, prepares a varlen prefill-style batch where decode sequences have query length 1, then postprocesses prefill and decode sequences separately.

**Tech Stack:** Python scheduler/model runner logic, FlashAttention varlen prefill path, existing script tests, `tools/profile_chunked_prefill.py`.

---

## File Structure

- Modify `tinyvllm/config.py`: add `chunked_prefill_mixed_batch: bool = False`.
- Modify `tinyvllm/engine/sequence.py`: add transient step metadata fields for mixed batches.
- Modify `tinyvllm/engine/scheduler.py`: schedule mixed batches and postprocess per sequence role.
- Modify `tinyvllm/engine/model_runner.py`: add `prepare_mixed()` and route `run(..., batch_kind="mixed")`.
- Modify `tinyvllm/engine/llm_engine.py`: accept scheduler metadata and report mixed token counts.
- Modify `tools/test_chunked_prefill.py`: add scheduler/postprocess regression tests.
- Modify `tools/profile_chunked_prefill.py`: add `--mode mixed`.
- Modify `tools/test_profile_chunked_prefill.py`: cover mixed step summarization.
- Modify `docs/qwen3-8b-fixes.md`: record implementation and local smoke status.

## Task 1: Scheduler mixed batch behavior

**Files:**
- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/sequence.py`
- Modify: `tinyvllm/engine/scheduler.py`
- Modify: `tools/test_chunked_prefill.py`

- [ ] **Step 1: Write failing scheduler test**

Add a test where one running decode sequence and one long waiting sequence coexist. With `chunked_prefill_mixed_batch=True`, `schedule()` should return both sequences, `is_prefill=True`, `do_sample=True`, and a mixed batch marker. The prefill sequence should have `step_is_decode=False`; the decode sequence should have `step_is_decode=True`.

- [ ] **Step 2: Implement config and sequence metadata**

Add `chunked_prefill_mixed_batch: bool = False` to config. Add transient fields to `Sequence`: `step_is_decode=False`, `step_do_sample=True`.

- [ ] **Step 3: Implement scheduler mixed path**

In chunked mode, before pure prefill-first scheduling, if mixed mode is enabled and `running` is non-empty, schedule one prefill chunk plus up to `max_num_seqs - 1` decode sequences. Return batch kind `"mixed"`. Reset consecutive prefill counter because decode is included.

- [ ] **Step 4: Implement mixed postprocess**

For mixed batches, commit prefill chunks and append tokens only for final prefill chunks; append tokens for decode sequences; finished sequences are deallocated, unfinished decode/prefill-final sequences return to `running`, unfinished intermediate prefill sequences return to `prefilling`.

- [ ] **Step 5: Run tests**

Run `python3 tools/test_chunked_prefill.py`. Expected: all tests pass.

## Task 2: Model runner mixed preparation

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/llm_engine.py`

- [ ] **Step 1: Add `prepare_mixed()`**

Build `input_ids`, `positions`, `cu_seqlens_q`, `cu_seqlens_k`, `slot_mapping`, and `block_tables` for mixed sequences. Prefill rows use `[prefill_chunk_start, prefill_chunk_end)`. Decode rows use the last token at `position=len(seq)` and the slot from the last token after `may_append()`.

- [ ] **Step 2: Route `run()` by batch kind**

Change `run(seqs, is_prefill, do_sample=True, batch_kind=None)` to use `prepare_mixed()` when `batch_kind == "mixed"`; otherwise preserve old behavior.

- [ ] **Step 3: Update engine step contract**

Make `LLMEngine.step()` accept either 3-tuple old scheduler return or 4-tuple with batch kind, then pass batch kind into `ModelRunner.run()` and scheduler postprocess.

- [ ] **Step 4: Run compile checks**

Run `python3 -m py_compile tinyvllm/engine/model_runner.py tinyvllm/engine/llm_engine.py tinyvllm/engine/scheduler.py`.

## Task 3: Profiler and docs

**Files:**
- Modify: `tools/profile_chunked_prefill.py`
- Modify: `tools/test_profile_chunked_prefill.py`
- Modify: `docs/qwen3-8b-fixes.md`

- [ ] **Step 1: Add profiler mixed mode**

Extend `--mode` choices to include `mixed`; pass `max_num_prefill_tokens_per_step`, `chunked_prefill_decode_first=False`, and `chunked_prefill_mixed_batch=True`.

- [ ] **Step 2: Ensure profiler labels mixed steps**

Keep existing sign convention: positive tokens are prefill/mixed prefill-like steps; add `kind="mixed"` when profiler observes mixed batch kind if exposed, otherwise record mode-level mixed output.

- [ ] **Step 3: Run tests**

Run `python3 tools/test_profile_chunked_prefill.py` and `python3 -m py_compile tools/profile_chunked_prefill.py`.

- [ ] **Step 4: Update docs**

Add §39 documenting mixed v0 scope, safety constraints, local tests, and the remote smoke command to run next.

## Self-Review

- Spec coverage: covers scheduler, runner, engine step contract, profiler, tests, and docs.
- Placeholder scan: no placeholders remain.
- Scope check: intentionally does not add new kernels or optimize Quest/C4 mixed interactions.
