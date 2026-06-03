# Chunked Prefill v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal chunked-prefill scheduler path that caps prefill work per engine step and prevents long prompt prefill from monopolizing decode indefinitely.

**Architecture:** Keep v0 conservative: batches are still either pure prefill or pure decode, never mixed. Chunked prefill tracks per-sequence computed prompt progress, delays prefix-cache publication until KV for a full block has actually been written, and skips sampling for intermediate chunks. Default config keeps old behavior unchanged.

**Tech Stack:** Python, TinyLLMForge scheduler/model-runner/block-manager, FlashAttention existing prefill-with-block-table path, dependency-light script tests.

---

## File Structure

- Modify `tinyvllm/config.py`: add chunked prefill knobs.
- Modify `tinyvllm/engine/sequence.py`: add prompt-computed/chunk boundary fields and serialize them for TP workers.
- Modify `tinyvllm/engine/block_manager.py`: add delayed hash publication for chunked prefill.
- Modify `tinyvllm/engine/scheduler.py`: add chunk scheduling, intermediate no-sample handling, final chunk handoff to decode.
- Modify `tinyvllm/engine/model_runner.py`: make prefill input/slot mapping use chunk boundaries and allow `do_sample=False`.
- Modify `tinyvllm/engine/llm_engine.py`: propagate `do_sample` and report actual chunk token count.
- Create `tools/test_chunked_prefill.py`: scheduler/block-manager unit tests without GPU.
- Modify `docs/qwen3-8b-fixes.md`: record the research question and smoke results after verification.

## Task 1: Chunked prefill scheduler state and tests

**Files:**
- Create: `tools/test_chunked_prefill.py`
- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/sequence.py`
- Modify: `tinyvllm/engine/scheduler.py`

- [ ] **Step 1: Write failing scheduler tests**

Create tests that assert: first chunk does not sample or append; final chunk samples once and moves to running; `decode_first` returns decode when running sequences exist.

- [ ] **Step 2: Run tests to verify RED**

Run: `python3 tools/test_chunked_prefill.py`

Expected: fail because `max_num_prefill_tokens_per_step` and `do_sample` return value do not exist yet.

- [ ] **Step 3: Implement minimal scheduler state**

Add config fields, sequence chunk fields, `prefilling` queue, `schedule() -> (seqs, is_prefill, do_sample)`, and prefill-aware `postprocess()`.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python3 tools/test_chunked_prefill.py`

Expected: scheduler tests pass.

## Task 2: Delayed prefix-cache publication

**Files:**
- Modify: `tinyvllm/engine/block_manager.py`
- Modify: `tools/test_chunked_prefill.py`

- [ ] **Step 1: Add failing cache publication test**

Assert that a future full block is not present in `hash_to_block_id` until the chunk covering that block has been postprocessed.

- [ ] **Step 2: Run test to verify RED**

Run: `python3 tools/test_chunked_prefill.py`

Expected: fail because existing `allocate()` publishes all full blocks immediately.

- [ ] **Step 3: Implement delayed commit**

Add `allocate(seq, publish_hashes=True)` and `commit_prefill(seq, old_end, new_end)`. Chunked scheduling calls `allocate(..., publish_hashes=False)` and postprocess commits only full blocks whose KV was computed.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python3 tools/test_chunked_prefill.py`

Expected: delayed publication test passes.

## Task 3: Model runner chunk boundaries

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Test: `tools/test_chunked_prefill.py`

- [ ] **Step 1: Update prefill preparation**

Use `seq.prefill_chunk_start` / `seq.prefill_chunk_end` for `input_ids`, `positions`, `cu_seqlens_q/k`, and exact per-token `slot_mapping`.

- [ ] **Step 2: Support no-sample intermediate chunks**

Change `ModelRunner.run(seqs, is_prefill, do_sample=True)` to skip sampler and return `None` when `do_sample=False`.

- [ ] **Step 3: Propagate do_sample through engine**

Change `LLMEngine.step()` to consume the 3-tuple scheduler result and pass `do_sample` into model runner and scheduler postprocess. Use actual chunk token count for prefill throughput.

- [ ] **Step 4: Verify local tests and compile**

Run: `python3 -m py_compile tinyvllm/engine/*.py tinyvllm/config.py tools/test_chunked_prefill.py && python3 tools/test_chunked_prefill.py`

Expected: pass.

## Task 4: Smoke and documentation

**Files:**
- Modify: `docs/qwen3-8b-fixes.md`

- [ ] **Step 1: Run local non-GPU tests**

Run: `python3 tools/test_chunked_prefill.py && python3 tools/test_ngram_speculative.py && python3 tools/test_eval_needle_fixed_prompts.py`

- [ ] **Step 2: Run remote generation smoke**

Run a small Qwen3-8B or Qwen3-0.6B generation with `max_num_prefill_tokens_per_step` enabled and compare output shape with default generation.

- [ ] **Step 3: Record §36**

Document v0 scope, correctness invariants, and smoke results.

- [ ] **Step 4: Commit**

Commit message: `研究：添加 chunked prefill v0 调度`

## Self-Review

- Spec coverage: plan covers config, scheduler state, delayed prefix-cache commit, model-runner chunk boundaries, verification, and documentation.
- Placeholder scan: no TBD/TODO placeholders remain.
- Scope check: v0 intentionally avoids mixed prefill+decode and online latency benchmark complexity; it only creates a safe chunked path for follow-up profiling.
