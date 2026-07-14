# Prefix Cache Correctness and Performance Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make TinyLLMForge's existing full-block prefix cache correctness-safe for cross-request reuse, then run a reproducible Qwen3-0.6B correctness and TTFT gate that decides whether deeper APC work is justified.

**Architecture:** Keep the chained hash cache and existing block tables. Add a maximum-reusable-token cap to allocation, use compute-complete publication for both normal and chunked prefill, and enforce that every sampled prefill row contains at least one query token. Add a dependency-light CPU regression suite, a one-model-instance GPU profiler that captures full logits and cold/warm/cache-cleared timings, and an isolated remote runner that never overwrites the existing remote checkout.

**Tech Stack:** Python 3.11, TinyLLMForge scheduler/block manager/model runner, PyTorch, Qwen3-0.6B, standard-library JSON/statistics/argparse/pathlib, Bash/SSH/rsync, existing plain-assert tool tests.

## Global Constraints

- Optimize prompt prefill and TTFT only; do not describe results as decode acceleration.
- Do not describe logical block reuse as reduced physical KV allocation or GPU-memory capacity.
- A newly allocated full block becomes reusable only after its prefill forward completes successfully.
- A sampled prefill request must retain at least one query token.
- With block size `B`, maximum reusable prompt tokens are `floor((prompt_tokens - 1) / B) * B`.
- Requests in one scheduler batch may reuse only blocks computed before that batch began.
- Do not add radix trees, cache-aware scheduling, partial-block reuse, same-batch dependency waves, or final-hidden-state/logits caching in this phase.
- Greedy output token IDs and decoded text must exactly match cold baselines.
- Warm/cache-cleared full-vocabulary logits must keep the cold argmax, `max_abs <= 0.25`, and `mean_abs <= 0.05`.
- The gate is `GO` only if every correctness/lifecycle case passes and warm median TTFT improves by at least 20% for both 1024-token and 2048-token-or-longer shared prefixes, with no warm median regression above 5%.
- Use `sitian@10.232.195.203`, Qwen3-0.6B, block size 256, greedy sampling, CUDA synchronization, and matching dynamic `TINYVLLM_DIST_PORT`/`MASTER_PORT`.
- Preserve the pre-existing uncommitted changes in `tinyvllm/engine/model_runner.py` and `AGENT_HANDOFF_STATE.md`; never stage either file wholesale.
- Remote validation must run from an isolated uploaded source tree and must record SHA-256 hashes of the tested source files.

---

## File Structure

- Modify `tinyvllm/engine/block_manager.py`
  - Add sampleable-prefix capping and reusable-cache clearing.
- Modify `tinyvllm/engine/scheduler.py`
  - Use delayed publication for normal prefill, commit normal prefill after the forward, and enforce positive sampled query lengths.
- Modify `tools/test_chunked_prefill.py`
  - Add CPU regressions for exact block boundaries, same-batch isolation, delayed publication, collision defense, cache clearing, and live-block safety.
- Create `tools/profile_prefix_cache.py`
  - Run correctness cases, capture logits, run cold/warm/cache-cleared repetitions, aggregate metrics, and compute the gate decision.
- Create `tools/test_profile_prefix_cache.py`
  - Test prompt construction, metric aggregation, tolerance checks, and `GO`/`NO_GO` boundaries without loading a model.
- Create `tools/run_prefix_cache_gate_remote.sh`
  - Upload tracked source into an isolated remote directory, select GPU/ports, run preflight/smoke/full gate, and mirror artifacts back.
- Create `experiments/prefix_cache/README.md`
  - Document commands, artifact schema, decision rules, and claim boundaries.
- Create generated canonical result directory
  `experiments/prefix_cache/qwen3_0_6b_gate_20260714/`
  - Store manifest, raw rows, aggregate JSON, Markdown report, logs, and source hashes.
- Modify `README.md`
  - Replace the stale APC checkbox with the validated implementation/gate status and link to the experiment README.
- Modify `AGENT_HANDOFF_STATE.md`
  - Append the final gate result, evidence, limitations, and next branch using exact-hunk staging only.

---

### Task 1: Prefix Match Cap and Cache Lifecycle Primitives

**Files:**
- Modify: `tinyvllm/engine/block_manager.py`
- Modify: `tools/test_chunked_prefill.py`

**Interfaces:**
- Produces: `BlockManager.max_reusable_tokens(seq: Sequence) -> int`
- Changes: `BlockManager.allocate(seq: Sequence, publish_hashes: bool = True, max_cached_tokens: int | None = None) -> None`
- Produces: `BlockManager.clear_reusable_cache() -> int`
- Later scheduler tasks call `max_reusable_tokens(seq)` and pass the result to `allocate(...)`.

- [ ] **Step 1: Add failing exact-boundary prefix-cap tests**

Append these tests after `test_chunked_prefill_restores_reused_cached_block_metadata()`:

```python
def _publish_and_release(block_manager, token_ids):
    seq = make_seq(token_ids, max_tokens=1)
    block_manager.allocate(seq, publish_hashes=False, max_cached_tokens=0)
    block_manager.commit_prefill(seq, 0, len(seq))
    block_table = list(seq.block_table)
    block_manager.deallocate(seq)
    return block_table


def test_max_reusable_tokens_keeps_one_sampleable_token():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=16, block_size=4)

    expected = {
        3: 0,
        4: 0,
        5: 4,
        8: 4,
        9: 8,
    }
    for prompt_tokens, reusable_tokens in expected.items():
        seq = make_seq(range(prompt_tokens), max_tokens=1)
        assert block_manager.max_reusable_tokens(seq) == reusable_tokens


def test_allocate_caps_exact_block_aligned_cache_hit():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=16, block_size=4)
    cached_blocks = _publish_and_release(block_manager, [1, 2, 3, 4])

    seq = make_seq([1, 2, 3, 4], max_tokens=1)
    block_manager.allocate(
        seq,
        publish_hashes=False,
        max_cached_tokens=block_manager.max_reusable_tokens(seq),
    )

    assert seq.num_cached_tokens == 0
    assert seq.num_computed_tokens == 0
    assert seq.block_table[0] != cached_blocks[0]


def test_allocate_reuses_only_blocks_before_sampleable_suffix():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=16, block_size=4)
    cached_blocks = _publish_and_release(block_manager, list(range(1, 9)))

    seq = make_seq(list(range(1, 9)), max_tokens=1)
    block_manager.allocate(
        seq,
        publish_hashes=False,
        max_cached_tokens=block_manager.max_reusable_tokens(seq),
    )

    assert seq.num_cached_tokens == 4
    assert seq.num_computed_tokens == 4
    assert seq.block_table[0] == cached_blocks[0]
    assert seq.block_table[1] != cached_blocks[1]
```

- [ ] **Step 2: Add failing collision and cache-clearing tests**

Append:

```python
def test_allocate_rejects_hash_collision_when_tokens_differ():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=8, block_size=4)
    cached = make_seq([1, 2, 3, 4], max_tokens=1)
    block_manager.allocate(cached, publish_hashes=False, max_cached_tokens=0)
    block_manager.commit_prefill(cached, 0, len(cached))
    cached_block = cached.block_table[0]
    cached_hash = block_manager.blocks[cached_block].hash
    block_manager.deallocate(cached)

    original_compute_hash = block_manager.compute_hash
    block_manager.compute_hash = lambda token_ids, prefix=-1: cached_hash
    try:
        seq = make_seq([9, 8, 7, 6], max_tokens=1)
        block_manager.allocate(seq, publish_hashes=False, max_cached_tokens=4)
    finally:
        block_manager.compute_hash = original_compute_hash

    assert seq.num_cached_tokens == 0
    assert seq.block_table[0] != cached_block


def test_clear_reusable_cache_preserves_live_block_metadata():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=8, block_size=4)
    free_cached = make_seq([1, 2, 3, 4], max_tokens=1)
    block_manager.allocate(free_cached, publish_hashes=False, max_cached_tokens=0)
    block_manager.commit_prefill(free_cached, 0, len(free_cached))
    free_block_id = free_cached.block_table[0]
    block_manager.deallocate(free_cached)

    live = make_seq([5, 6, 7, 8], max_tokens=2)
    block_manager.allocate(live, publish_hashes=False, max_cached_tokens=0)
    block_manager.commit_prefill(live, 0, len(live))
    live_block_id = live.block_table[0]
    live_hash = block_manager.blocks[live_block_id].hash
    live_tokens = list(block_manager.blocks[live_block_id].token_ids)

    cleared = block_manager.clear_reusable_cache()

    assert cleared == 1
    assert block_manager.blocks[free_block_id].hash == -1
    assert block_manager.blocks[free_block_id].token_ids == []
    assert block_manager.blocks[live_block_id].hash == live_hash
    assert block_manager.blocks[live_block_id].token_ids == live_tokens
    assert block_manager.blocks[live_block_id].ref_count == 1


def test_capacity_pressure_never_returns_live_shared_block():
    reset_sequence_state()
    block_manager = BlockManager(num_blocks=3, block_size=4)
    live = make_seq([1, 2, 3, 4], max_tokens=2)
    block_manager.allocate(live, publish_hashes=False, max_cached_tokens=0)
    block_manager.commit_prefill(live, 0, len(live))
    live_block_id = live.block_table[0]

    shared = make_seq([1, 2, 3, 4, 5], max_tokens=1)
    block_manager.allocate(
        shared,
        publish_hashes=False,
        max_cached_tokens=block_manager.max_reusable_tokens(shared),
    )
    assert shared.block_table[0] == live_block_id
    assert block_manager.blocks[live_block_id].ref_count == 2

    other = make_seq([9, 8, 7, 6], max_tokens=1)
    block_manager.allocate(other, publish_hashes=False, max_cached_tokens=0)

    assert other.block_table[0] != live_block_id
    assert live_block_id in block_manager.used_block_ids
    assert live_block_id not in block_manager.free_block_ids
```

Register all new tests in `main()`.

- [ ] **Step 3: Run the focused suite and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
```

Expected: fail with missing `max_reusable_tokens`, unexpected
`max_cached_tokens`, or missing `clear_reusable_cache`.

- [ ] **Step 4: Implement the maximum-reusable-token helper**

Add after `can_allocate()`:

```python
    def max_reusable_tokens(self, seq: Sequence) -> int:
        """Return the full-block prefix cap that leaves one query token."""
        if len(seq) <= 1:
            return 0
        return ((len(seq) - 1) // self.block_size) * self.block_size
```

- [ ] **Step 5: Extend allocation with a cache-hit cap**

Change the signature and prefix loop logic:

```python
    def allocate(
        self,
        seq: Sequence,
        publish_hashes: bool = True,
        max_cached_tokens: int | None = None,
    ):
        assert not seq.block_table
        if max_cached_tokens is None:
            max_cached_tokens = len(seq)
        max_cached_tokens = max(0, min(int(max_cached_tokens), len(seq)))
        max_cached_blocks = max_cached_tokens // self.block_size
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1) if i < max_cached_blocks else -1
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            ...
```

Preserve the existing invariant that after the first miss, all later blocks
are newly allocated. Preserve exact-token comparison after hash lookup.

- [ ] **Step 6: Implement reusable-cache clearing**

Add:

```python
    def clear_reusable_cache(self) -> int:
        """Drop only idle prefix metadata; never mutate live blocks."""
        self.hash_to_block_id.clear()
        cleared = 0
        for block in self.blocks:
            if block.ref_count != 0:
                if block.hash != -1:
                    self.hash_to_block_id[block.hash] = block.block_id
                continue
            if block.hash != -1 or block.token_ids:
                block.hash = -1
                block.token_ids = []
                cleared += 1
        return cleared
```

Reindex live hashed blocks so clearing idle metadata does not make live shared
prefixes undiscoverable.

- [ ] **Step 7: Run tests and verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
```

Expected: `chunked prefill tests passed`.

- [ ] **Step 8: Commit the block-manager primitive**

```bash
git add tinyvllm/engine/block_manager.py tools/test_chunked_prefill.py
git commit -m "Fix prefix cache sampleable suffix"
```

---

### Task 2: Unify Normal and Chunked Prefill Publication

**Files:**
- Modify: `tinyvllm/engine/scheduler.py`
- Modify: `tools/test_chunked_prefill.py`

**Interfaces:**
- Consumes: `BlockManager.max_reusable_tokens(seq)`
- Consumes: `BlockManager.allocate(..., publish_hashes=False, max_cached_tokens=...)`
- Consumes: `BlockManager.commit_prefill(seq, old_end, new_end)`
- Produces scheduler invariant: every sampled prefill row has
  `prefill_chunk_end > prefill_chunk_start`.

- [ ] **Step 1: Add failing normal-prefill delayed-publication test**

Append:

```python
def test_normal_prefill_publishes_only_after_postprocess():
    reset_sequence_state()
    scheduler = Scheduler(make_config(max_num_prefill_tokens_per_step=0))
    seq = make_seq([1, 2, 3, 4, 5], max_tokens=2)
    scheduler.add(seq)

    seqs, is_prefill, do_sample = scheduler.schedule()
    h0 = scheduler.block_manager.compute_hash([1, 2, 3, 4], -1)

    assert seqs == [seq]
    assert is_prefill is True
    assert do_sample is True
    assert seq.prefill_chunk_start == 0
    assert seq.prefill_chunk_end == 5
    assert h0 not in scheduler.block_manager.hash_to_block_id

    scheduler.postprocess(seqs, [99], is_prefill, do_sample)

    assert h0 in scheduler.block_manager.hash_to_block_id
    assert seq.num_computed_tokens == 5
    assert seq.completion_token_ids == [99]
```

- [ ] **Step 2: Add failing same-batch isolation and exact-hit tests**

Append:

```python
def test_normal_prefill_does_not_reuse_prefix_created_in_same_batch():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=0,
        max_num_seqs=3,
        max_num_batched_tokens=32,
    ))
    seq_a = make_seq([1, 2, 3, 4, 5], max_tokens=1)
    seq_b = make_seq([9, 8, 7, 6, 5], max_tokens=1)
    seq_c = make_seq([1, 2, 3, 4, 5], max_tokens=1)
    for seq in (seq_a, seq_b, seq_c):
        scheduler.add(seq)

    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [seq_a, seq_b, seq_c]
    assert is_prefill is True
    assert do_sample is True
    assert [seq.num_cached_tokens for seq in seqs] == [0, 0, 0]
    assert all(seq.prefill_chunk_end > seq.prefill_chunk_start for seq in seqs)
    assert seq_a.block_table[0] != seq_c.block_table[0]


def test_normal_prefill_exact_block_warm_hit_recomputes_final_block():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=0,
        max_num_batched_tokens=32,
    ))
    cold = make_seq([1, 2, 3, 4], max_tokens=1)
    scheduler.add(cold)
    seqs, is_prefill, do_sample = scheduler.schedule()
    scheduler.postprocess(seqs, [70], is_prefill, do_sample)

    warm = make_seq([1, 2, 3, 4], max_tokens=1)
    scheduler.add(warm)
    seqs, is_prefill, do_sample = scheduler.schedule()

    assert seqs == [warm]
    assert warm.num_cached_tokens == 0
    assert warm.prefill_chunk_start == 0
    assert warm.prefill_chunk_end == 4
```

- [ ] **Step 3: Add failing multi-block warm-hit tests for normal and chunked paths**

Append:

```python
def _seed_scheduler_cache(scheduler, token_ids):
    seq = make_seq(token_ids, max_tokens=1)
    scheduler.add(seq)
    seqs, is_prefill, do_sample = scheduler.schedule()
    while not do_sample:
        scheduler.postprocess(seqs, None, is_prefill, do_sample)
        seqs, is_prefill, do_sample = scheduler.schedule()
    scheduler.postprocess(seqs, [71], is_prefill, do_sample)


def test_normal_prefill_warm_hit_reuses_only_complete_prefix_blocks():
    reset_sequence_state()
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=0,
        max_num_batched_tokens=32,
    ))
    _seed_scheduler_cache(scheduler, list(range(1, 9)))

    warm = make_seq(list(range(1, 9)), max_tokens=1)
    scheduler.add(warm)
    seqs, is_prefill, do_sample = scheduler.schedule()

    assert warm.num_cached_tokens == 4
    assert warm.prefill_chunk_start == 4
    assert warm.prefill_chunk_end == 8
    assert do_sample is True


def test_chunked_prefill_uses_same_sampleable_prefix_cap():
    reset_sequence_state()
    scheduler = Scheduler(make_config(max_num_prefill_tokens_per_step=4))
    _seed_scheduler_cache(scheduler, list(range(1, 9)))

    warm = make_seq(list(range(1, 9)), max_tokens=1)
    scheduler.add(warm)
    seqs, is_prefill, do_sample = scheduler.schedule()

    assert warm.num_cached_tokens == 4
    assert warm.prefill_chunk_start == 4
    assert warm.prefill_chunk_end == 8
    assert do_sample is True
```

Register the tests in `main()`.

- [ ] **Step 4: Run tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
```

Expected: failures showing immediate publication, same-batch reuse, or normal
prefill not committing hashes after postprocess.

- [ ] **Step 5: Apply capped delayed allocation to normal prefill**

In the non-chunked `schedule()` loop, replace the allocation/progress block
with:

```python
            max_cached_tokens = self.block_manager.max_reusable_tokens(seq)
            self.block_manager.allocate(
                seq,
                publish_hashes=False,
                max_cached_tokens=max_cached_tokens,
            )
            seq.prefill_chunk_start = seq.num_cached_tokens
            seq.prefill_chunk_end = len(seq)
            seq.prefill_chunk_final = True
            assert seq.prefill_chunk_end > seq.prefill_chunk_start
```

Do not set `seq.num_computed_tokens = len(seq)` before the model forward.

- [ ] **Step 6: Apply the same cap to chunked allocation**

At both chunked allocation call sites, use:

```python
            max_cached_tokens = self.block_manager.max_reusable_tokens(seq)
            self.block_manager.allocate(
                seq,
                publish_hashes=False,
                max_cached_tokens=max_cached_tokens,
            )
```

Keep `_schedule_one_prefill_chunk()`'s defensive full-hit fallback, but add an
assertion after chunk boundaries are assigned:

```python
        if seq.prefill_chunk_final:
            assert seq.prefill_chunk_end > seq.prefill_chunk_start
```

After capped allocation, the full-hit fallback should no longer be reached for
new sampled requests; it remains defensive for restored legacy state.

- [ ] **Step 7: Commit normal prefill before token append**

In `postprocess()`, before the existing normal token loop:

```python
        if is_prefill:
            for seq in seqs:
                old_end = seq.num_computed_tokens
                new_end = seq.prefill_chunk_end
                assert new_end > seq.prefill_chunk_start
                self.block_manager.commit_prefill(seq, old_end, new_end)
                seq.num_computed_tokens = new_end
```

Then retain the existing append/finish/deallocate logic.

- [ ] **Step 8: Run focused and neighboring tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_profile_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_eval_needle_fixed_prompts.py
python3 -m py_compile tinyvllm/engine/block_manager.py tinyvllm/engine/scheduler.py
```

Expected:

```text
chunked prefill tests passed
chunked prefill profiler tests passed
eval_needle fixed-prompt tests passed
```

- [ ] **Step 9: Commit scheduler lifecycle**

```bash
git add tinyvllm/engine/scheduler.py tools/test_chunked_prefill.py
git commit -m "Delay normal prefix cache publication"
```

---

### Task 3: CPU-Testable Prefix Cache Gate Reporting

**Files:**
- Create: `tools/profile_prefix_cache.py`
- Create: `tools/test_profile_prefix_cache.py`

**Interfaces:**
- Produces: `make_token_prompt(length: int, offset: int = 0) -> list[int]`
- Produces: `expected_reusable_tokens(prompt_tokens: int, block_size: int) -> int`
- Produces: `summarize_case_rows(rows: list[dict]) -> dict`
- Produces: `decide_gate(correctness_rows: list[dict], performance_cases: list[dict]) -> dict`
- Task 4 consumes these report helpers and adds GPU-specific logits comparison.

- [ ] **Step 1: Create failing pure-function tests**

Create `tools/test_profile_prefix_cache.py`:

```python
"""Prefix-cache gate report tests.

Run: python3 tools/test_profile_prefix_cache.py
"""

from tools.profile_prefix_cache import (
    decide_gate,
    expected_reusable_tokens,
    make_token_prompt,
    summarize_case_rows,
)


def _perf_case(prefix_tokens, cold_ms, warm_ms, correct=True):
    return {
        "shared_prefix_tokens": prefix_tokens,
        "cold": {"median_ttft_ms": cold_ms},
        "warm": {"median_ttft_ms": warm_ms},
        "all_correct": correct,
        "expected_reusable_tokens": prefix_tokens,
        "warm_median_cached_tokens": prefix_tokens,
        "warm_median_query_tokens": 300,
        "cold_median_query_tokens": prefix_tokens + 300,
    }


def test_expected_reusable_tokens_keeps_sampleable_suffix():
    assert expected_reusable_tokens(255, 256) == 0
    assert expected_reusable_tokens(256, 256) == 0
    assert expected_reusable_tokens(257, 256) == 256
    assert expected_reusable_tokens(512, 256) == 256
    assert expected_reusable_tokens(513, 256) == 512


def test_make_token_prompt_is_deterministic_and_offset_sensitive():
    assert make_token_prompt(8, 0) == make_token_prompt(8, 0)
    assert make_token_prompt(8, 0) != make_token_prompt(8, 11)
    assert len(make_token_prompt(257, 3)) == 257


def test_summarize_case_rows_reports_medians_and_correctness():
    rows = [
        {"state": "warm", "ttft_ms": 10.0, "query_tokens": 300, "cached_tokens": 1024, "correct": True},
        {"state": "warm", "ttft_ms": 12.0, "query_tokens": 300, "cached_tokens": 1024, "correct": True},
        {"state": "warm", "ttft_ms": 11.0, "query_tokens": 300, "cached_tokens": 1024, "correct": True},
    ]
    summary = summarize_case_rows(rows)
    assert summary["median_ttft_ms"] == 11.0
    assert summary["median_query_tokens"] == 300
    assert summary["median_cached_tokens"] == 1024
    assert summary["all_correct"] is True


def test_decide_gate_requires_correctness_and_two_large_prefix_wins():
    correctness = [{"case": "boundary_256", "correct": True}]
    performance = [
        _perf_case(256, 10.0, 10.2),
        _perf_case(1024, 20.0, 15.0),
        _perf_case(2048, 40.0, 28.0),
    ]
    decision = decide_gate(correctness, performance)
    assert decision["decision"] == "GO"

    performance[2]["warm"]["median_ttft_ms"] = 35.0
    decision = decide_gate(correctness, performance)
    assert decision["decision"] == "NO_GO"
    assert "2048" in " ".join(decision["reasons"])


def test_decide_gate_rejects_any_correctness_failure_or_warm_regression():
    performance = [
        _perf_case(256, 10.0, 10.6),
        _perf_case(1024, 20.0, 15.0),
        _perf_case(2048, 40.0, 28.0),
    ]
    decision = decide_gate([{"case": "triple", "correct": False}], performance)
    assert decision["decision"] == "NO_GO"

    decision = decide_gate([{"case": "triple", "correct": True}], performance)
    assert decision["decision"] == "NO_GO"
    assert "regression" in " ".join(decision["reasons"]).lower()


def test_decide_gate_rejects_cached_or_query_token_mismatch():
    performance = [
        _perf_case(256, 10.0, 9.8),
        _perf_case(1024, 20.0, 15.0),
        _perf_case(2048, 40.0, 28.0),
    ]
    performance[1]["warm_median_cached_tokens"] = 768
    decision = decide_gate([{"case": "boundary", "correct": True}], performance)
    assert decision["decision"] == "NO_GO"
    assert "cached-token" in " ".join(decision["reasons"])


def main():
    test_expected_reusable_tokens_keeps_sampleable_suffix()
    test_make_token_prompt_is_deterministic_and_offset_sensitive()
    test_summarize_case_rows_reports_medians_and_correctness()
    test_decide_gate_requires_correctness_and_two_large_prefix_wins()
    test_decide_gate_rejects_any_correctness_failure_or_warm_regression()
    test_decide_gate_rejects_cached_or_query_token_mismatch()
    print("prefix cache profiler tests passed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_profile_prefix_cache.py
```

Expected: `ModuleNotFoundError: No module named 'tools.profile_prefix_cache'`.

- [ ] **Step 3: Implement pure helpers and gate decision**

Create `tools/profile_prefix_cache.py` with:

```python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def make_token_prompt(length: int, offset: int = 0) -> list[int]:
    return [100 + ((index + offset) % 1000) for index in range(length)]


def expected_reusable_tokens(prompt_tokens: int, block_size: int) -> int:
    if prompt_tokens <= 1:
        return 0
    return ((prompt_tokens - 1) // block_size) * block_size


def summarize_case_rows(rows: list[dict]) -> dict:
    return {
        "samples": len(rows),
        "median_ttft_ms": statistics.median(float(row["ttft_ms"]) for row in rows),
        "min_ttft_ms": min(float(row["ttft_ms"]) for row in rows),
        "max_ttft_ms": max(float(row["ttft_ms"]) for row in rows),
        "median_query_tokens": statistics.median(int(row["query_tokens"]) for row in rows),
        "median_cached_tokens": statistics.median(int(row["cached_tokens"]) for row in rows),
        "all_correct": all(bool(row["correct"]) for row in rows),
    }


def decide_gate(correctness_rows: list[dict], performance_cases: list[dict]) -> dict:
    reasons = []
    failed = [row["case"] for row in correctness_rows if not row["correct"]]
    if failed:
        reasons.append("correctness failures: " + ", ".join(failed))
    for case in performance_cases:
        prefix = int(case["shared_prefix_tokens"])
        cold = float(case["cold"]["median_ttft_ms"])
        warm = float(case["warm"]["median_ttft_ms"])
        improvement = (cold - warm) / cold if cold > 0 else 0.0
        case["warm_ttft_improvement_fraction"] = improvement
        if not case["all_correct"]:
            reasons.append(f"{prefix}: incorrect performance sample")
        if prefix >= 1024 and improvement < 0.20:
            reasons.append(f"{prefix}: warm median TTFT improvement below 20%")
        if warm > cold * 1.05:
            reasons.append(f"{prefix}: warm median TTFT regression exceeds 5%")
        if int(case["warm_median_cached_tokens"]) != int(case["expected_reusable_tokens"]):
            reasons.append(f"{prefix}: cached-token accounting mismatch")
        saved_queries = int(case["cold_median_query_tokens"]) - int(case["warm_median_query_tokens"])
        if saved_queries != int(case["expected_reusable_tokens"]):
            reasons.append(f"{prefix}: executed prefill-token reduction mismatch")
    return {"decision": "NO_GO" if reasons else "GO", "reasons": reasons}
```

- [ ] **Step 4: Run pure tests and verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_profile_prefix_cache.py
python3 -m py_compile tools/profile_prefix_cache.py tools/test_profile_prefix_cache.py
```

Expected: `prefix cache profiler tests passed`.

- [ ] **Step 5: Commit report primitives**

```bash
git add tools/profile_prefix_cache.py tools/test_profile_prefix_cache.py
git commit -m "Add prefix cache gate reporting"
```

---

### Task 4: GPU Correctness and Performance Profiler

**Files:**
- Modify: `tools/profile_prefix_cache.py`
- Modify: `tools/test_profile_prefix_cache.py`

**Interfaces:**
- Produces CLI:
  `python tools/profile_prefix_cache.py --model PATH --mode correctness|performance|full --out-dir DIR`
- Produces:
  `compare_logits(reference, candidate) -> dict[str, float | int | bool]`
- Produces files:
  `manifest.json`, `correctness_rows.json`, `performance_rows.json`,
  `summary.json`, and `report.md`.

- [ ] **Step 1: Add parser and manifest tests**

Extend `tools/test_profile_prefix_cache.py`:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.profile_prefix_cache import build_manifest, parse_int_list


def test_parse_int_list_accepts_comma_separated_values():
    assert parse_int_list("256,1024,2048") == [256, 1024, 2048]


def test_build_manifest_records_source_hashes():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source.py"
        source.write_text("print('ok')\n")
        manifest = build_manifest(root, ["source.py"], {"model": "/tmp/model"})
        assert manifest["args"]["model"] == "/tmp/model"
        assert len(manifest["source_sha256"]["source.py"]) == 64
```

Register both tests in `main()`.

- [ ] **Step 2: Implement CLI arguments and source manifest**

Add:

```python
def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(repo_root: Path, source_files: list[str], args: dict) -> dict:
    return {
        "args": args,
        "source_sha256": {
            relative: sha256_file(repo_root / relative)
            for relative in source_files
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["correctness", "performance", "full"], default="full")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shared-prefix-tokens", default="256,1024,2048")
    parser.add_argument("--suffix-tokens", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmup-repetitions", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    return parser.parse_args()
```

- [ ] **Step 3: Add a safe manual-prefill driver**

Implement:

```python
def cuda_sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def schedule_and_run_prefill(llm, prompts, capture_logits=True):
    from tinyvllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    for prompt in prompts:
        llm.add_request(prompt, params)
    scheduled = llm.scheduler.schedule()
    if len(scheduled) == 4:
        seqs, is_prefill, do_sample, batch_kind = scheduled
    else:
        seqs, is_prefill, do_sample = scheduled
        batch_kind = None
    assert is_prefill and do_sample
    metadata = [{
        "seq_id": seq.seq_id,
        "prompt_tokens": len(seq),
        "cached_tokens": int(seq.num_cached_tokens),
        "chunk_start": int(seq.prefill_chunk_start),
        "chunk_end": int(seq.prefill_chunk_end),
        "query_tokens": int(seq.prefill_chunk_end - seq.prefill_chunk_start),
        "block_table": list(seq.block_table),
    } for seq in seqs]
    assert all(row["query_tokens"] > 0 for row in metadata)

    captures = []
    original_forward = llm.model_runner.sampler.forward
    if capture_logits:
        def capture_forward(logits, temperatures):
            captures.append(logits.detach().float().cpu().clone())
            return original_forward(logits, temperatures)
        llm.model_runner.sampler.forward = capture_forward
    try:
        cuda_sync()
        start = time.perf_counter()
        token_ids = llm.model_runner.call("run", seqs, is_prefill, do_sample, batch_kind)
        cuda_sync()
        ttft_ms = (time.perf_counter() - start) * 1000.0
    finally:
        llm.model_runner.sampler.forward = original_forward
    llm.scheduler.postprocess(seqs, token_ids, is_prefill, do_sample, batch_kind)
    logits = captures[0] if captures else None
    return {
        "metadata": metadata,
        "token_ids": [int(token_id) for token_id in token_ids],
        "decoded": [llm.tokenizer.decode([int(token_id)]) for token_id in token_ids],
        "logits": logits,
        "ttft_ms": ttft_ms,
    }
```

Do not call `llm.exit()` explicitly; let the registered `atexit` handler run
once, avoiding the already observed double-exit error.

- [ ] **Step 4: Implement logits comparison**

Add:

```python
def compare_logits(reference, candidate) -> dict:
    delta = (reference - candidate).abs()
    max_abs = float(delta.max())
    mean_abs = float(delta.mean())
    reference_argmax = int(reference.argmax())
    candidate_argmax = int(candidate.argmax())
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "reference_argmax": reference_argmax,
        "candidate_argmax": candidate_argmax,
        "argmax_match": reference_argmax == candidate_argmax,
        "within_tolerance": (
            reference_argmax == candidate_argmax
            and max_abs <= 0.25
            and mean_abs <= 0.05
        ),
    }
```

- [ ] **Step 5: Implement required correctness cases**

Use one `LLM` instance and `block_manager.clear_reusable_cache()` between cold
cases. Run:

1. repeated prompt lengths 255, 256, 257, 512, and 513;
2. same-batch `[P, Q, P]`, with `P` and `Q` selected by trying deterministic
   offsets until independent logits differ by `max_abs > 1.0`;
3. two prompts with one full shared block and distinct suffixes;
4. cache-cleared rerun;
5. CPU collision/lifecycle status imported from the preflight test command.

For every row record:

```python
{
    "case": case_name,
    "state": state,
    "prompt_tokens": prompt_tokens,
    "cached_tokens": metadata["cached_tokens"],
    "query_tokens": metadata["query_tokens"],
    "token_id": token_id,
    "decoded": decoded,
    "logit_diff": comparison,
    "correct": (
        token_id == cold_token_id
        and decoded == cold_decoded
        and comparison["within_tolerance"]
    ),
}
```

For `[P,Q,P]`, additionally assert the third row matches independent `P` and
is not bit-identical to the batch `Q` row.

- [ ] **Step 6: Implement cold/warm/cache-cleared performance cases**

For each shared prefix length:

1. build `prefix + producer_suffix` and `prefix + consumer_suffix`;
2. clear reusable metadata;
3. run the consumer alone for the cold sample;
4. clear metadata, run producer to completion, then run consumer for warm;
5. clear metadata after producer, then run consumer for cache-cleared;
6. repeat with deterministic suffix offsets and discard configured warmups.

Record each sample's actual `cached_tokens`, `query_tokens`, `ttft_ms`, first
token, decoded token, and correctness against the cold reference.

Aggregate each state with `summarize_case_rows()`, then call `decide_gate()`.

- [ ] **Step 7: Write deterministic artifacts**

Write:

```text
<out-dir>/
  manifest.json
  correctness_rows.json
  performance_rows.json
  summary.json
  report.md
```

Use this top-level `summary.json` schema:

```python
summary = {
    "correctness_rows": correctness_rows,
    "performance_cases": performance_cases,
    "decision": decision,
}
```

`report.md` must contain:

- source hashes;
- correctness table;
- cold/warm/cache-cleared TTFT table;
- cached/query-token accounting;
- explicit `GO` or `NO_GO`;
- every rejection reason;
- claim boundaries from the spec.

- [ ] **Step 8: Run local pure tests and compile**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_profile_prefix_cache.py
python3 -m py_compile tools/profile_prefix_cache.py tools/test_profile_prefix_cache.py
```

Expected: tests pass and compile exits 0.

- [ ] **Step 9: Commit the executable profiler**

```bash
git add tools/profile_prefix_cache.py tools/test_profile_prefix_cache.py
git commit -m "Add prefix cache correctness profiler"
```

---

### Task 5: Isolated Remote Runner

**Files:**
- Create: `tools/run_prefix_cache_gate_remote.sh`

**Interfaces:**
- Consumes local tracked/modified source files.
- Creates isolated remote tree:
  `/data00/home/sitian/sitian-workspace01/tllm/prefix-cache-gate-<tag>`
- Mirrors remote artifacts to local
  `experiments/prefix_cache/qwen3_0_6b_gate_<tag>/`.

- [ ] **Step 1: Create the remote runner**

Create a strict Bash script with:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_BASE="${REMOTE_BASE:-/data00/home/sitian/sitian-workspace01/tllm}"
REMOTE_PYTHON="${REMOTE_PYTHON:-${REMOTE_BASE}/env/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B}"
TAG="${TAG:-$(date +%Y%m%d)}"
REMOTE_TREE="${REMOTE_BASE}/prefix-cache-gate-${TAG}"
LOCAL_OUT="${LOCAL_OUT:-experiments/prefix_cache/qwen3_0_6b_gate_${TAG}}"
REMOTE_OUT="${REMOTE_TREE}/gate_out"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
PORT="${PORT:-$((25000 + RANDOM % 12000))}"
REPETITIONS="${REPETITIONS:-7}"

mkdir -p "${LOCAL_OUT}"
ssh "${REMOTE_HOST}" "rm -rf '${REMOTE_TREE}' && mkdir -p '${REMOTE_TREE}'"

git ls-files -z \
  | rsync -a --from0 --files-from=- ./ "${REMOTE_HOST}:${REMOTE_TREE}/"

ssh "${REMOTE_HOST}" "
  set -euo pipefail
  cd '${REMOTE_TREE}'
  export PYTHONPATH='${REMOTE_TREE}'
  export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'
  export TINYVLLM_DIST_PORT='${PORT}'
  export MASTER_PORT='${PORT}'
  export PYTHONDONTWRITEBYTECODE=1
  '${REMOTE_PYTHON}' -m py_compile \
    tinyvllm/engine/block_manager.py \
    tinyvllm/engine/scheduler.py \
    tools/profile_prefix_cache.py
  '${REMOTE_PYTHON}' tools/test_chunked_prefill.py
  '${REMOTE_PYTHON}' tools/test_profile_prefix_cache.py
  '${REMOTE_PYTHON}' tools/profile_prefix_cache.py \
    --model '${MODEL_PATH}' \
    --mode full \
    --out-dir '${REMOTE_OUT}' \
    --shared-prefix-tokens 256,1024,2048 \
    --suffix-tokens 64 \
    --repetitions '${REPETITIONS}' \
    --enforce-eager
"

rsync -a "${REMOTE_HOST}:${REMOTE_OUT}/" "${LOCAL_OUT}/"
```

Do not sync into the existing remote `TinyLLMForge` directory.

- [ ] **Step 2: Add executable bit and shell validation**

Run:

```bash
chmod +x tools/run_prefix_cache_gate_remote.sh
bash -n tools/run_prefix_cache_gate_remote.sh
```

Expected: exit 0.

- [ ] **Step 3: Run a one-repetition remote smoke**

Run:

```bash
TAG=20260714_smoke REPETITIONS=1 \
  tools/run_prefix_cache_gate_remote.sh
```

Expected:

- CPU tests pass remotely;
- all correctness cases finish without zero-query rows;
- local directory
  `experiments/prefix_cache/qwen3_0_6b_gate_20260714_smoke/` exists;
- `summary.json` contains an explicit decision, but the one-repetition smoke is
  not accepted as final performance evidence.

- [ ] **Step 4: Inspect smoke source hashes and row coverage**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("experiments/prefix_cache/qwen3_0_6b_gate_20260714_smoke")
manifest = json.loads((root / "manifest.json").read_text())
correctness = json.loads((root / "correctness_rows.json").read_text())
required = {"repeat_255", "repeat_256", "repeat_257", "repeat_512", "repeat_513",
            "same_batch_p_q_p", "shared_prefix_different_suffix", "cache_cleared"}
seen = {row["case"] for row in correctness}
assert required <= seen, sorted(required - seen)
for path in (
    "tinyvllm/engine/block_manager.py",
    "tinyvllm/engine/scheduler.py",
    "tools/profile_prefix_cache.py",
):
    assert len(manifest["source_sha256"][path]) == 64
print("PREFIX_CACHE_SMOKE_COVERAGE_OK")
PY
```

- [ ] **Step 5: Commit the remote runner**

```bash
git add tools/run_prefix_cache_gate_remote.sh
git commit -m "Add prefix cache remote gate runner"
```

---

### Task 6: Full Gate, Decision, and Documentation

**Files:**
- Create: `experiments/prefix_cache/README.md`
- Create: `experiments/prefix_cache/qwen3_0_6b_gate_20260714/`
- Modify: `README.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes final `summary.json` and `report.md`.
- Produces the canonical APC `GO` or `NO_GO` record and the next research
  branch decision.

- [ ] **Step 1: Run the full remote gate**

Run:

```bash
TAG=20260714 REPETITIONS=7 \
  tools/run_prefix_cache_gate_remote.sh
```

Expected: canonical artifacts appear under
`experiments/prefix_cache/qwen3_0_6b_gate_20260714/`.

- [ ] **Step 2: Verify artifact completeness and recompute the decision**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("experiments/prefix_cache/qwen3_0_6b_gate_20260714")
required_files = {
    "manifest.json",
    "correctness_rows.json",
    "performance_rows.json",
    "summary.json",
    "report.md",
}
missing = [name for name in required_files if not (root / name).is_file()]
assert not missing, missing

summary = json.loads((root / "summary.json").read_text())
assert summary["decision"]["decision"] in {"GO", "NO_GO"}
assert len(summary["correctness_rows"]) >= 8
prefixes = {case["shared_prefix_tokens"] for case in summary["performance_cases"]}
assert {256, 1024, 2048} <= prefixes
assert all(case["cold"]["samples"] == 7 for case in summary["performance_cases"])
assert all(case["warm"]["samples"] == 7 for case in summary["performance_cases"])
assert all(case["cache_cleared"]["samples"] == 7 for case in summary["performance_cases"])
print("PREFIX_CACHE_GATE_ARTIFACTS_OK", summary["decision"]["decision"])
PY
```

- [ ] **Step 3: Write experiment usage and claim boundaries**

Create `experiments/prefix_cache/README.md` with:

```markdown
# Prefix Cache Gate

## Purpose

Validate correctness-safe cross-request full-block KV reuse and measure prefill
and TTFT effects. This does not measure decode acceleration or physical KV
capacity reduction.

## Local Tests

```bash
python3 tools/test_chunked_prefill.py
python3 tools/test_profile_prefix_cache.py
```

## Remote Gate

```bash
TAG=20260714 REPETITIONS=7 tools/run_prefix_cache_gate_remote.sh
```

## Canonical Result

See `qwen3_0_6b_gate_20260714/report.md` and `summary.json`.

The decision is generated from exact output/logits checks, actual executed
prefill tokens, cached-token accounting, and cold/warm/cache-cleared TTFT.
```

Append the actual decision and key metrics only after reading the canonical
summary.

- [ ] **Step 4: Update the root README from actual evidence**

Replace:

```markdown
- [] apply APC(automatic prefix caching)
```

with one of:

```markdown
- [x] APC correctness-safe full-block reuse gate — GO; see `experiments/prefix_cache/README.md`
```

or:

```markdown
- [x] APC correctness gate completed — NO_GO for deeper APC optimization; see `experiments/prefix_cache/README.md`
```

Choose strictly from `summary.json`; do not infer.

- [ ] **Step 5: Append the handoff using exact-hunk staging**

Append a dated section to `AGENT_HANDOFF_STATE.md` containing:

- implementation commits;
- local and remote validation commands;
- canonical artifact path;
- source hashes;
- correctness result;
- 256/1024/2048 cold/warm/cache-cleared median TTFT;
- actual cached/query-token reductions;
- explicit decision and reasons;
- what the result proves and does not prove;
- next branch:
  - if `GO`: design radix versus chained-hash/cache-aware scheduling;
  - if `NO_GO`: retain safety fix and move to the next bounded optimization.

Stage only the appended hunk:

```bash
git diff -- AGENT_HANDOFF_STATE.md
git add -p AGENT_HANDOFF_STATE.md
git diff --cached -- AGENT_HANDOFF_STATE.md
```

Reject any pre-existing unrelated hunk.

- [ ] **Step 6: Run final verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_profile_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_profile_prefix_cache.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_eval_needle_fixed_prompts.py
python3 -m py_compile \
  tinyvllm/engine/block_manager.py \
  tinyvllm/engine/scheduler.py \
  tools/profile_prefix_cache.py
bash -n tools/run_prefix_cache_gate_remote.sh
git diff --check
```

Expected: all tests pass, compile/shell checks exit 0, and no whitespace
errors are reported.

- [ ] **Step 7: Audit prompt-to-artifact coverage**

Create a checklist from the approved spec and verify:

```text
[ ] compute-complete publication: CPU test + scheduler diff
[ ] one query token per sampled row: boundary tests + remote metadata
[ ] 255/256/257/512/513 cases: correctness_rows.json
[ ] [P,Q,P] wrong-row regression: correctness_rows.json
[ ] shared prefix/different suffix: correctness_rows.json
[ ] collision defense: CPU test
[ ] cache clearing/live blocks: CPU test
[ ] cold/warm/cache-cleared: performance_rows.json
[ ] cached and actual query tokens: summary.json
[ ] TTFT thresholds: decision reasons
[ ] source hashes: manifest.json
[ ] claim boundaries: report.md + experiment README
[ ] root README and handoff: actual result only
```

Treat any unchecked item as incomplete and fix it before committing.

- [ ] **Step 8: Commit final evidence and documentation**

Stage only owned files and the exact handoff hunk:

```bash
git add \
  README.md \
  experiments/prefix_cache/README.md \
  experiments/prefix_cache/qwen3_0_6b_gate_20260714
git add -p AGENT_HANDOFF_STATE.md
git diff --cached --name-only
git diff --cached --check
git commit -m "Record prefix cache correctness gate"
```

Verify `tinyvllm/engine/model_runner.py` is not staged.

---

## Self-Review

### Spec Coverage

- Compute-complete publication is implemented and tested in Tasks 1-2.
- The sampleable-suffix formula is implemented and boundary-tested in Tasks
  1-2.
- Same-batch producer/consumer reuse is prevented by delayed publication and
  covered by a `[P,Q,P]` CPU and GPU regression.
- Cross-batch exact-block zero-query crashes are covered by 256/512-token
  boundary tests and remote correctness cases.
- Collision defense, live reference safety, idle cache clearing, chained hash
  publication, and capacity pressure are covered in the block-manager suite.
- Cold/warm/cache-cleared output, logits, cached tokens, executed query tokens,
  prefill latency/TTFT, source hashes, and gate thresholds are covered by the
  profiler and artifact audit.
- Remote work runs from an isolated tree and does not overwrite the divergent
  remote checkout.
- README and handoff updates are driven only by canonical artifacts.

### Placeholder Scan

The plan contains no `TBD`, `TODO`, “implement later”, or unspecified test
steps. Result-dependent documentation explicitly requires selecting text from
`summary.json`, not filling an unbounded placeholder.

### Type and Interface Consistency

- `max_reusable_tokens()` returns an integer token cap consumed by every
  scheduler allocation path.
- `allocate(..., max_cached_tokens=...)` initializes
  `num_cached_tokens/num_computed_tokens`, which scheduler chunk boundaries
  consume.
- `commit_prefill()` remains the sole compute-complete publication method.
- Profiler row fields match aggregation and gate-decision field names.
- Remote runner output paths match the final artifact audit and documentation.

### Scope Check

This plan is one testable subsystem: correctness-safe cross-request full-block
prefix reuse plus its decision gate. Radix trees, cache-aware scheduling,
same-batch dependency waves, and final-state caching remain separate follow-up
designs.
