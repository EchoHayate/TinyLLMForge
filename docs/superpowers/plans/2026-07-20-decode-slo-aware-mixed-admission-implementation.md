# Decode-SLO-Aware Mixed Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement disabled-by-default `P5` decode-SLO-aware mixed admission, calibrate its conservative integer step-cost envelope on the source-bound remote environment, and independently determine whether it removes P4 decode-tail regressions while preserving a repeatable benefit.

**Architecture:** `LLMEngine` owns one injectable monotonic nanosecond clock and passes exactly one decision timestamp plus one synchronous step-end timestamp to `Scheduler`. `Scheduler` reuses P4 backlog hysteresis only as a demand gate, keeps per-sequence decode-progress state locally, selects the largest safe chunk from a fixed integer ladder, and calls the existing transactional mixed helper with the exact oldest runnable decode row protected. The arrival-load harness adds a separate cost-calibration stage and immutable evidence; the independent verifier reconstructs calibration, clock/progress state, demand transitions, safe-chunk choices, structural correctness, and final P5 classification without trusting scheduler labels.

**Tech Stack:** Python 3, dataclasses, integer nanosecond arithmetic, immutable `MappingProxyType` snapshots, deque-based scheduler state, TinyLLMForge block manager, JSON/JSONL source-bound artifacts, dependency-light script tests, Bash remote runner, Qwen3-0.6B on `sitian@10.232.195.203`.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`; never modify `/Users/bytedance/dev/TinyLLMForge`.
- Execute inline in the current session; do not dispatch subagents.
- Use stable policy name `P5` and descriptive name `decode_slo_aware_mixed_admission`.
- Keep `chunked_prefill_slo_mixed=False` by default.
- P5 is mutually exclusive with P3 always-on mixed, P4 adaptive mixed, KV offload, and other experimental scheduler modes.
- Reuse P4 `inactive / active / draining` hysteresis only as a demand gate; do not consume `chunked_prefill_adaptive_max_mixed_steps`.
- Preregister `chunked_prefill_slo_target_gap_ns=64_000_000`, `chunked_prefill_slo_reserve_ns=8_000_000`, `chunked_prefill_slo_min_chunk_tokens=16`, and `max_num_prefill_tokens_per_step=128`.
- Generate the canonical descending ladder exactly as `128, 112, 96, 80, 64, 48, 32, 16`.
- Obtain `chunked_prefill_slo_cost_intercept_ns` and `chunked_prefill_slo_cost_per_prefill_token_ns` only from source/environment-bound remote cost calibration.
- Compute each shape's nearest-rank p99 from at least seven measured iterations; with seven iterations p99 is the observed maximum.
- Inflate each measured p99 using integer arithmetic: `(measured_p99_ns * 5 + 3) // 4`.
- Freeze `predicted_step_ns = cost_intercept_ns + tokens * cost_per_prefill_token_ns`; use signed 64-bit overflow checks.
- `LLMEngine` owns `time.monotonic_ns`; tests inject an integer fake clock.
- Sample one `decision_now_ns` immediately before `Scheduler.schedule(decision_now_ns)`.
- Sample one `step_end_ns` immediately after the synchronous model-runner call returns and pass the exact decision/end timestamps to `Scheduler.postprocess`.
- A clock violation is sticky for the engine lifetime and forces P5 decode-only.
- Keep decode-progress state scheduler-local; never add it to `Sequence` or TP serialization.
- A P5 mixed batch must contain the exact reconstructed oldest runnable decode sequence; never substitute a younger row.
- Before SLO approval, do not mutate queues, sequence status, chunk boundaries, prefix hashes, KV ownership, or call `may_append()`.
- Preserve P0, P3, P4, exact greedy output, lifecycle, prefix-cache semantics, block accounting, and disabled-default behavior.
- Canonical comparison is exactly `P0 / P4 / P5`: 6 scenarios × 3 policies × 3 repetitions = 54 cases.
- P4 is diagnostic only; only independently recomputed P5 may determine top-level `GO`.
- GPU/model work runs only on `sitian@10.232.195.203` using `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python` and `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`.
- Use `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- Give every model process unique dynamic `TINYVLLM_DIST_PORT` and `MASTER_PORT`.
- Do not mutate the remote checkout, use rsync, kill unrelated processes, clear shared `/tmp`, or reuse artifacts across source/policy identities.
- Preserve untracked `experiments/` roots and stage files selectively; never use `git add -A`.
- Do not update README before independently verified `GO`; write `PROMISING_NOT_PROVEN`, `NO_GO`, and `INCOMPLETE` only to `AGENT_HANDOFF_STATE.md`.

---

## File Map

- Modify `tinyvllm/config.py`: add six P5 fields and fail-closed integer/configuration validation.
- Modify `tinyvllm/engine/scheduler.py`: accept decision timestamps, own progress/clock state, derive safe chunks, protect the exact oldest row, publish immutable decision evidence, and postprocess progress.
- Modify `tinyvllm/engine/llm_engine.py`: own/inject the monotonic clock, bracket the synchronous model call, and merge decision/postprocess evidence into `last_step_observation`.
- Modify `tools/test_chunked_prefill.py`: dependency-light scheduler/config/engine TDD for every local P5 contract and P0/P3/P4 regression.
- Modify `tools/arrival_load_driver.py`: preserve engine-owned timing evidence without substituting driver timestamps.
- Modify `tools/test_arrival_load_driver.py`: prove complete immutable P5 snapshots survive JSONL recording.
- Create `tools/arrival_load_cost_calibration.py`: define required shapes, validate raw rows, compute nearest-rank p99, 25% inflation, and the minimal positive integer envelope.
- Create `tools/test_arrival_load_cost_calibration.py`: dependency-light tests for shapes, arithmetic, infeasibility, overflow, and coefficient reconstruction.
- Modify `tools/arrival_load_gate.py`: freeze P0/P4/P5 identity, add cost-calibration artifacts/stage, P5 smoke, canonical matrix, structural counters, and P5-only promotion.
- Modify `tools/test_arrival_load_gate.py`: prove identity, matrix, calibration dependency, smoke coverage, and exact promotion guards.
- Modify `tools/arrival_load_verify.py`: independently reconstruct cost envelope, progress, demand state, oldest row, chunk selection, suppression, timing, metrics, and classification.
- Modify `tools/test_arrival_load_verify.py`: build complete P5 synthetic artifacts and fail every required tamper independently.
- Modify `tools/run_arrival_load_gate_remote.sh`: expose `cost-calibration` and `workload-calibration` modes and enforce the full immutable chain.
- Modify `tools/test_run_arrival_load_gate_remote.py`: assert exact remote runtime, stage order, predecessor tags, dynamic ports, local verification, and prohibited operations.
- Modify `AGENT_HANDOFF_STATE.md` only after a complete source-bound result exists.
- Modify `README.md` only if the independent verifier returns `GO`.
- Create raw artifacts under `experiments/arrival_load/<run-tag>/`; never stage or commit them.

## Shared Interfaces

Use these exact interfaces across tasks:

- `INT64_MAX = (1 << 63) - 1`
- `build_slo_chunk_ladder(max_chunk_tokens: int, min_chunk_tokens: int) -> tuple[int, ...]`
- `select_slo_chunk(*, remaining_slack_ns: int, cost_intercept_ns: int, cost_per_prefill_token_ns: int, token_ladder: tuple[int, ...]) -> tuple[int | None, int | None]`
- `Scheduler.schedule(self, decision_now_ns: int | None = None) -> tuple[list[Sequence], bool, bool] | tuple[list[Sequence], bool, bool, str]`
- `Scheduler.postprocess(self, seqs: list[Sequence], token_ids: list[int] | None, is_prefill: bool = False, do_sample: bool = True, batch_kind: str | None = None, *, decision_now_ns: int | None = None, step_end_ns: int | None = None) -> None`
- `Scheduler._schedule_mixed_prefill_decode(self, *, allow_waiting_admission: bool = True, require_decode: bool = False, required_decode_seq_id: int | None = None, max_prefill_tokens: int | None = None) -> tuple[list[Sequence], bool, bool, str] | tuple[list[Sequence], bool, bool] | None`

The immutable P5 decision snapshot has these exact keys:

```python
P5_DECISION_FIELDS = (
    "decision_now_ns",
    "target_gap_ns",
    "reserve_ns",
    "oldest_decode_seq_id",
    "oldest_decode_progress_ns",
    "oldest_decode_age_ns",
    "remaining_slack_ns",
    "cost_intercept_ns",
    "cost_per_prefill_token_ns",
    "candidate_chunk_tokens",
    "predicted_step_ns",
    "selected_chunk_tokens",
    "actual_prefill_tokens",
    "scheduled_decode_seq_ids",
    "demand_state_before",
    "demand_state_after",
    "suppression_reason",
    "clock_invalid",
    "clock_invalid_reason",
)
```

Postprocess appends only:

```python
P5_POSTPROCESS_FIELDS = (
    "step_end_ns",
    "actual_step_duration_ns",
    "decode_progress_updates",
    "finished_progress_entries_removed",
)
```

---

### Task 1: Add Fail-Closed P5 Configuration and Integer Helpers

**Files:**
- Modify: `tinyvllm/config.py:55`
- Modify: `tinyvllm/config.py:153`
- Modify: `tinyvllm/engine/scheduler.py:9`
- Test: `tools/test_chunked_prefill.py`

**Interfaces:**
- Consumes: existing chunked-prefill, P3, P4, and KV-offload configuration.
- Produces: six P5 config fields, `INT64_MAX`, `build_slo_chunk_ladder()`, and `select_slo_chunk()`.

- [ ] **Step 1: Write failing config/default/overflow tests**

Add tests that parse the real dataclass and instantiate it with fake `transformers`:

```python
def test_slo_mixed_config_defaults_and_fail_closed_contract():
    Config = load_real_config_class()
    with tempfile.TemporaryDirectory() as model:
        common = {
            "model": model,
            "max_num_batched_tokens": 4096,
            "max_model_len": 4096,
            "kvcache_block_size": 256,
        }
        config = Config(**common)
        assert config.chunked_prefill_slo_mixed is False
        assert config.chunked_prefill_slo_target_gap_ns == 0
        assert config.chunked_prefill_slo_reserve_ns == 0
        assert config.chunked_prefill_slo_cost_intercept_ns == 0
        assert config.chunked_prefill_slo_cost_per_prefill_token_ns == 0
        assert config.chunked_prefill_slo_min_chunk_tokens == 16

        enabled = {
            "chunked_prefill_slo_mixed": True,
            "chunked_prefill_slo_target_gap_ns": 64_000_000,
            "chunked_prefill_slo_reserve_ns": 8_000_000,
            "chunked_prefill_slo_cost_intercept_ns": 4_000_000,
            "chunked_prefill_slo_cost_per_prefill_token_ns": 100_000,
            "chunked_prefill_slo_min_chunk_tokens": 16,
            "max_num_prefill_tokens_per_step": 128,
        }
        Config(**common, **enabled)
        invalid = (
            {**enabled, "chunked_prefill_slo_target_gap_ns": 0},
            {**enabled, "chunked_prefill_slo_reserve_ns": 0},
            {**enabled, "chunked_prefill_slo_reserve_ns": 64_000_000},
            {**enabled, "chunked_prefill_slo_cost_intercept_ns": 0},
            {**enabled, "chunked_prefill_slo_cost_per_prefill_token_ns": 0},
            {**enabled, "chunked_prefill_slo_min_chunk_tokens": 0},
            {**enabled, "chunked_prefill_slo_min_chunk_tokens": 256},
            {**enabled, "max_num_prefill_tokens_per_step": 120},
            {**enabled, "chunked_prefill_mixed_batch": True},
            {**enabled, "chunked_prefill_adaptive_mixed": True},
            {**enabled, "kv_offload_mvp0": True},
            {
                **enabled,
                "chunked_prefill_slo_cost_intercept_ns": (1 << 63),
            },
            {
                **enabled,
                "chunked_prefill_slo_cost_per_prefill_token_ns":
                    ((1 << 63) - 1) // 128 + 1,
            },
        )
        for overrides in invalid:
            try:
                Config(**common, **overrides)
            except AssertionError:
                pass
            else:
                raise AssertionError(f"invalid P5 config accepted: {overrides}")
```

- [ ] **Step 2: Write failing ladder and boundary tests**

```python
def test_slo_chunk_ladder_and_largest_safe_boundary():
    scheduler = load_scheduler_module()
    ladder = scheduler.build_slo_chunk_ladder(128, 16)
    assert ladder == (128, 112, 96, 80, 64, 48, 32, 16)
    assert scheduler.select_slo_chunk(
        remaining_slack_ns=16_800_000,
        cost_intercept_ns=4_000_000,
        cost_per_prefill_token_ns=100_000,
        token_ladder=ladder,
    ) == (128, 16_800_000)
    assert scheduler.select_slo_chunk(
        remaining_slack_ns=15_200_000,
        cost_intercept_ns=4_000_000,
        cost_per_prefill_token_ns=100_000,
        token_ladder=ladder,
    ) == (112, 15_200_000)
    assert scheduler.select_slo_chunk(
        remaining_slack_ns=5_599_999,
        cost_intercept_ns=4_000_000,
        cost_per_prefill_token_ns=100_000,
        token_ladder=ladder,
    ) == (None, None)
```

- [ ] **Step 3: Run the focused suite and verify RED**

Run:

```bash
python3 tools/test_chunked_prefill.py
```

Expected: failure because P5 fields and helper functions do not exist.

- [ ] **Step 4: Add exact fields and validation**

Add to `Config`:

```python
chunked_prefill_slo_mixed: bool = False
chunked_prefill_slo_target_gap_ns: int = 0
chunked_prefill_slo_reserve_ns: int = 0
chunked_prefill_slo_cost_intercept_ns: int = 0
chunked_prefill_slo_cost_per_prefill_token_ns: int = 0
chunked_prefill_slo_min_chunk_tokens: int = 16
```

Add before model loading:

```python
int64_max = (1 << 63) - 1
for value in (
    self.chunked_prefill_slo_target_gap_ns,
    self.chunked_prefill_slo_reserve_ns,
    self.chunked_prefill_slo_cost_intercept_ns,
    self.chunked_prefill_slo_cost_per_prefill_token_ns,
):
    assert isinstance(value, int) and not isinstance(value, bool)
    assert 0 <= value <= int64_max
assert (
    isinstance(self.chunked_prefill_slo_min_chunk_tokens, int)
    and not isinstance(self.chunked_prefill_slo_min_chunk_tokens, bool)
    and self.chunked_prefill_slo_min_chunk_tokens > 0
)
if self.chunked_prefill_slo_mixed:
    assert self.chunked_prefill_slo_target_gap_ns > 0
    assert self.chunked_prefill_slo_reserve_ns > 0
    assert (
        self.chunked_prefill_slo_reserve_ns
        < self.chunked_prefill_slo_target_gap_ns
    )
    assert self.chunked_prefill_slo_cost_intercept_ns > 0
    assert self.chunked_prefill_slo_cost_per_prefill_token_ns > 0
    assert self.max_num_prefill_tokens_per_step > 0
    assert (
        self.chunked_prefill_slo_min_chunk_tokens
        <= self.max_num_prefill_tokens_per_step
    )
    assert (
        self.max_num_prefill_tokens_per_step
        % self.chunked_prefill_slo_min_chunk_tokens
        == 0
    )
    assert not self.chunked_prefill_mixed_batch
    assert not self.chunked_prefill_adaptive_mixed
    assert not self.kv_offload_mvp0
    assert (
        self.chunked_prefill_slo_cost_per_prefill_token_ns
        <= (
            int64_max - self.chunked_prefill_slo_cost_intercept_ns
        ) // self.max_num_prefill_tokens_per_step
    )
```

- [ ] **Step 5: Add pure integer helpers**

Add to `scheduler.py`:

```python
INT64_MAX = (1 << 63) - 1


def build_slo_chunk_ladder(
    max_chunk_tokens: int,
    min_chunk_tokens: int,
) -> tuple[int, ...]:
    if (
        isinstance(max_chunk_tokens, bool)
        or isinstance(min_chunk_tokens, bool)
        or not isinstance(max_chunk_tokens, int)
        or not isinstance(min_chunk_tokens, int)
        or min_chunk_tokens <= 0
        or max_chunk_tokens < min_chunk_tokens
        or max_chunk_tokens % min_chunk_tokens != 0
    ):
        raise ValueError("invalid SLO chunk ladder")
    return tuple(
        range(max_chunk_tokens, min_chunk_tokens - 1, -min_chunk_tokens)
    )


def select_slo_chunk(
    *,
    remaining_slack_ns: int,
    cost_intercept_ns: int,
    cost_per_prefill_token_ns: int,
    token_ladder: tuple[int, ...],
) -> tuple[int | None, int | None]:
    for tokens in token_ladder:
        if (
            cost_per_prefill_token_ns
            > (INT64_MAX - cost_intercept_ns) // tokens
        ):
            raise OverflowError("P5 predicted step cost overflows int64")
        predicted = cost_intercept_ns + tokens * cost_per_prefill_token_ns
        if predicted <= remaining_slack_ns:
            return tokens, predicted
    return None, None
```

- [ ] **Step 6: Run GREEN and commit**

Run:

```bash
python3 tools/test_chunked_prefill.py
git diff --check
```

Expected: PASS and no whitespace errors.

Commit selectively:

```bash
git add tinyvllm/config.py tinyvllm/engine/scheduler.py tools/test_chunked_prefill.py
git commit -m "feat(scheduler): add P5 configuration contract"
```

---

### Task 2: Add Engine-Owned Monotonic Timing and Immutable Step Evidence

**Files:**
- Modify: `tinyvllm/engine/llm_engine.py:7`
- Modify: `tinyvllm/engine/scheduler.py:14`
- Test: `tools/test_chunked_prefill.py`

**Interfaces:**
- Consumes: Task 1 P5 configuration.
- Produces: injectable engine clock, timestamped `schedule()`/`postprocess()`, immutable decision snapshot, and complete `last_step_observation`.

- [ ] **Step 1: Write a failing fake-clock engine test**

Load `LLMEngine.step` without creating a real model and assert exact clock use:

```python
class IntegerClock:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return next(self.values)


def test_engine_samples_one_decision_and_one_step_end_timestamp():
    engine = object.__new__(load_llm_engine_class())
    clock = IntegerClock([100, 175])
    engine._clock_ns = clock
    engine.scheduler = FakeTimedScheduler()
    engine.model_runner = FakeTimedModelRunner()
    engine.last_batch_kind = None
    engine.last_scheduled_seqs = []
    engine.last_step_observation = None
    outputs, _ = engine.step()
    assert outputs == []
    assert clock.calls == 2
    assert engine.scheduler.schedule_calls == [100]
    assert engine.scheduler.postprocess_calls == [{
        "decision_now_ns": 100,
        "step_end_ns": 175,
    }]
    assert engine.last_step_observation["decision_now_ns"] == 100
    assert engine.last_step_observation["step_end_ns"] == 175
    assert engine.last_step_observation["actual_step_duration_ns"] == 75
```

- [ ] **Step 2: Write failing snapshot immutability tests**

```python
def test_p5_decision_snapshot_is_immutable_until_postprocess_copy():
    scheduler = make_scheduler(
        chunked_prefill_slo_mixed=True,
        chunked_prefill_slo_target_gap_ns=64_000_000,
        chunked_prefill_slo_reserve_ns=8_000_000,
        chunked_prefill_slo_cost_intercept_ns=4_000_000,
        chunked_prefill_slo_cost_per_prefill_token_ns=100_000,
    )
    scheduler._publish_slo_decision({
        "decision_now_ns": 100,
        "suppression_reason": "inactive",
    })
    snapshot = scheduler.last_slo_decision
    try:
        snapshot["decision_now_ns"] = 200
    except TypeError:
        pass
    else:
        raise AssertionError("P5 decision snapshot is mutable")
```

- [ ] **Step 3: Run RED**

```bash
python3 tools/test_chunked_prefill.py
```

Expected: failure because engine clock injection and P5 snapshot publishing are absent.

- [ ] **Step 4: Add the engine clock and timestamp bracketing**

In `llm_engine.py`:

```python
import time
```

At initialization:

```python
self._clock_ns = kwargs.pop("_clock_ns", time.monotonic_ns)
```

At the start of `step()`:

```python
queue_before = self.scheduler.observation_snapshot()
decision_now_ns = self._clock_ns()
scheduled = self.scheduler.schedule(decision_now_ns)
```

Immediately after the synchronous model call:

```python
token_ids = self.model_runner.call(
    "run", seqs, is_prefill, do_sample, batch_kind
)
step_end_ns = self._clock_ns()
```

Pass both timestamps:

```python
self.scheduler.postprocess(
    seqs,
    token_ids,
    is_prefill,
    do_sample,
    batch_kind,
    decision_now_ns=decision_now_ns,
    step_end_ns=step_end_ns,
)
```

Merge scheduler evidence:

```python
timing_observation = self.scheduler.last_slo_observation()
self.last_step_observation = {
    "policy_branch": self.scheduler.last_policy_branch,
    "batch_kind": batch_kind,
    "is_prefill": bool(is_prefill),
    "do_sample": bool(do_sample),
    "scheduled": scheduled_rows,
    "queue_before": queue_before,
    "queue_after": self.scheduler.observation_snapshot(),
    "new_completion_tokens_by_seq": token_deltas,
    "finished_seq_ids": [
        seq.seq_id for seq in seqs if seq.is_finished
    ],
    "memory": self.model_runner.memory_snapshot(),
    **timing_observation,
}
```

- [ ] **Step 5: Add scheduler snapshot storage without policy behavior**

Use immutable internal snapshots:

```python
from types import MappingProxyType
```

Initialize:

```python
self.last_slo_decision = MappingProxyType({})
self._last_slo_postprocess: dict = {}
```

Add:

```python
def _publish_slo_decision(self, values: dict) -> None:
    self.last_slo_decision = MappingProxyType(dict(values))
    self._last_slo_postprocess = {}


def last_slo_observation(self) -> dict:
    return {
        **dict(self.last_slo_decision),
        **dict(self._last_slo_postprocess),
    }
```

Make `schedule(decision_now_ns=None)` and the extended `postprocess` signature backward compatible while P0/P3/P4 ignore timestamps.

- [ ] **Step 6: Run GREEN and commit**

```bash
python3 tools/test_chunked_prefill.py
python3 -m py_compile tinyvllm/engine/llm_engine.py tinyvllm/engine/scheduler.py
git diff --check
git add tinyvllm/engine/llm_engine.py tinyvllm/engine/scheduler.py tools/test_chunked_prefill.py
git commit -m "feat(engine): expose monotonic P5 step timing"
```

---

### Task 3: Implement Decode Progress and Sticky Clock Validation

**Files:**
- Modify: `tinyvllm/engine/scheduler.py`
- Test: `tools/test_chunked_prefill.py`

**Interfaces:**
- Consumes: Task 2 timestamps.
- Produces: `decode_progress_ns_by_seq_id`, sticky invalid state, oldest-row reconstruction, lifecycle cleanup, and progress evidence.

- [ ] **Step 1: Write failing progress lifecycle tests**

Cover first-token prefill, normal decode, mixed decode, intermediate prefill, finish, preemption, and empty reset:

```python
def test_decode_progress_updates_only_for_completion_tokens():
    scheduler = make_slo_scheduler()
    first = make_seq(seq_id=1, prompt_tokens=32, max_tokens=4)
    first.step_is_decode = False
    first.step_do_sample = True
    scheduler._postprocess_chunked_prefill(
        [first], [101], True, step_end_ns=1_000
    )
    assert scheduler.decode_progress_ns_by_seq_id == {1: 1_000}

    intermediate = make_seq(seq_id=2, prompt_tokens=64, max_tokens=4)
    intermediate.step_do_sample = False
    scheduler._postprocess_chunked_prefill(
        [intermediate], None, False, step_end_ns=1_100
    )
    assert 2 not in scheduler.decode_progress_ns_by_seq_id

    first.step_is_decode = True
    scheduler._postprocess_mixed([first], [102], step_end_ns=1_200)
    assert scheduler.decode_progress_ns_by_seq_id[1] == 1_200
```

```python
def test_progress_survives_preemption_but_is_excluded_until_running():
    scheduler = make_slo_scheduler()
    seq = make_running_seq(seq_id=7)
    scheduler.decode_progress_ns_by_seq_id[7] = 1_000
    scheduler.preempt(seq)
    assert scheduler.decode_progress_ns_by_seq_id[7] == 1_000
    assert scheduler._oldest_runnable_decode(2_000) is None
    scheduler.waiting.remove(seq)
    scheduler.running.append(seq)
    assert scheduler._oldest_runnable_decode(2_000) == (7, 1_000, 1_000)
```

- [ ] **Step 2: Write failing clock regression and missing-progress tests**

```python
def test_clock_regression_is_sticky_and_forces_decode_only():
    scheduler = make_slo_scheduler_with_running_progress(
        decision_now_ns=1_000
    )
    scheduler.schedule(1_000)
    scheduler.schedule(999)
    assert scheduler.slo_clock_invalid is True
    assert scheduler.slo_clock_invalid_reason == "decision_clock_regressed"
    branch = scheduler.last_policy_branch
    assert branch == "slo_mixed_clock_invalid_decode"
    scheduler.schedule(2_000)
    assert scheduler.last_policy_branch == "slo_mixed_clock_invalid_decode"
```

```python
def test_missing_runnable_progress_fails_closed():
    scheduler = make_slo_scheduler()
    scheduler.running.append(make_running_seq(seq_id=9))
    scheduler.waiting.extend(make_waiting_seqs(8))
    scheduler.schedule(10_000)
    assert scheduler.last_policy_branch == "slo_mixed_missing_progress_decode"
    assert scheduler.last_slo_decision["suppression_reason"] == (
        "missing_decode_progress"
    )
```

- [ ] **Step 3: Run RED**

```bash
python3 tools/test_chunked_prefill.py
```

- [ ] **Step 4: Add state and validation helpers**

Initialize:

```python
self.decode_progress_ns_by_seq_id: dict[int, int] = {}
self.slo_clock_invalid = False
self.slo_clock_invalid_reason: str | None = None
self._last_slo_decision_now_ns: int | None = None
```

Add:

```python
def _invalidate_slo_clock(self, reason: str) -> None:
    if not self.slo_clock_invalid:
        self.slo_clock_invalid = True
        self.slo_clock_invalid_reason = reason


def _validate_slo_decision_time(self, decision_now_ns: int | None) -> bool:
    if (
        isinstance(decision_now_ns, bool)
        or not isinstance(decision_now_ns, int)
        or decision_now_ns < 0
    ):
        self._invalidate_slo_clock("invalid_decision_timestamp")
        return False
    if (
        self._last_slo_decision_now_ns is not None
        and decision_now_ns < self._last_slo_decision_now_ns
    ):
        self._invalidate_slo_clock("decision_clock_regressed")
        return False
    self._last_slo_decision_now_ns = decision_now_ns
    return not self.slo_clock_invalid


def _oldest_runnable_decode(
    self,
    decision_now_ns: int,
) -> tuple[int, int, int] | None:
    oldest = None
    for seq in self.running:
        progress_ns = self.decode_progress_ns_by_seq_id.get(seq.seq_id)
        if progress_ns is None:
            return None
        if progress_ns > decision_now_ns:
            self._invalidate_slo_clock("progress_timestamp_in_future")
            return None
        candidate = (progress_ns, seq.seq_id)
        if oldest is None or candidate < oldest:
            oldest = candidate
    if oldest is None:
        return None
    progress_ns, seq_id = oldest
    return seq_id, progress_ns, decision_now_ns - progress_ns
```

- [ ] **Step 5: Update progress only from actual token deltas**

Make `_postprocess_chunked_prefill()` and `_postprocess_mixed()` return:

```python
tuple[dict[int, int], list[int]]
```

For each actual `seq.append_token(token_id)`:

```python
self.decode_progress_ns_by_seq_id[seq.seq_id] = step_end_ns
progress_updates[seq.seq_id] = step_end_ns
```

When a sequence finishes:

```python
if self.decode_progress_ns_by_seq_id.pop(seq.seq_id, None) is not None:
    finished_progress_entries_removed.append(seq.seq_id)
```

When the engine becomes empty:

```python
self.decode_progress_ns_by_seq_id.clear()
self._last_slo_decision_now_ns = None
```

Do not clear `slo_clock_invalid` or its reason.

- [ ] **Step 6: Validate postprocess timestamps and publish append-only fields**

```python
def _validate_slo_step_end(
    self,
    decision_now_ns: int | None,
    step_end_ns: int | None,
) -> bool:
    if (
        isinstance(step_end_ns, bool)
        or not isinstance(step_end_ns, int)
        or step_end_ns < 0
        or isinstance(decision_now_ns, bool)
        or not isinstance(decision_now_ns, int)
        or step_end_ns < decision_now_ns
    ):
        self._invalidate_slo_clock("invalid_step_end_timestamp")
        return False
    return True
```

Publish:

```python
self._last_slo_postprocess = {
    "step_end_ns": step_end_ns,
    "actual_step_duration_ns": step_end_ns - decision_now_ns,
    "decode_progress_updates": {
        str(seq_id): timestamp
        for seq_id, timestamp in sorted(progress_updates.items())
    },
    "finished_progress_entries_removed": sorted(
        finished_progress_entries_removed
    ),
}
```

- [ ] **Step 7: Run GREEN and commit**

```bash
python3 tools/test_chunked_prefill.py
python3 -m py_compile tinyvllm/engine/scheduler.py
git diff --check
git add tinyvllm/engine/scheduler.py tools/test_chunked_prefill.py
git commit -m "feat(scheduler): track P5 decode progress"
```

---

### Task 4: Implement SLO Admission and Exact Oldest-Row Protection

**Files:**
- Modify: `tinyvllm/engine/scheduler.py`
- Test: `tools/test_chunked_prefill.py`

**Interfaces:**
- Consumes: Tasks 1–3 helpers and timing/progress state.
- Produces: `_schedule_slo_mixed()`, stable branches, exact oldest-row reservation, bounded transactional admission, and immutable decision evidence.

- [ ] **Step 1: Write failing slack/chunk/suppression tests**

```python
def test_active_demand_never_overrides_no_slack():
    scheduler = make_slo_scheduler_with_active_demand()
    scheduler.decode_progress_ns_by_seq_id[1] = 1_000
    before = scheduler_state_digest(scheduler)
    scheduler.schedule(57_000_001)
    assert scheduler.last_policy_branch == "slo_mixed_no_slack_decode"
    assert scheduler.last_slo_decision["remaining_slack_ns"] == -1
    assert scheduler.last_slo_decision["selected_chunk_tokens"] is None
    assert no_prefill_mutation(before, scheduler)
```

```python
def test_largest_safe_chunk_is_selected_with_exact_integer_math():
    scheduler = make_slo_scheduler_with_active_demand(
        intercept_ns=4_000_000,
        slope_ns=100_000,
    )
    scheduler.decode_progress_ns_by_seq_id[1] = 1_000
    scheduler.schedule(41_800_000)
    decision = scheduler.last_slo_decision
    assert decision["oldest_decode_age_ns"] == 41_799_000
    assert decision["remaining_slack_ns"] == 14_201_000
    assert decision["candidate_chunk_tokens"] == [
        128, 112, 96, 80, 64, 48, 32, 16
    ]
    assert decision["selected_chunk_tokens"] == 96
    assert decision["predicted_step_ns"] == 13_600_000
```

- [ ] **Step 2: Write failing exact protected-row tests**

```python
def test_mixed_batch_contains_exact_oldest_runnable_decode_row():
    scheduler = make_slo_scheduler_with_active_demand()
    older = make_running_seq(seq_id=11)
    younger = make_running_seq(seq_id=12)
    scheduler.running.extend([younger, older])
    scheduler.decode_progress_ns_by_seq_id.update({
        11: 1_000,
        12: 5_000,
    })
    scheduled = scheduler.schedule(10_000_000)
    seqs = scheduled[0]
    decode_ids = [
        seq.seq_id for seq in seqs if seq.step_is_decode
    ]
    assert 11 in decode_ids
    assert scheduler.last_slo_decision["oldest_decode_seq_id"] == 11
    assert scheduler.last_slo_decision["scheduled_decode_seq_ids"] == (
        decode_ids
    )
```

```python
def test_protected_row_reservation_failure_does_not_substitute_younger():
    scheduler = make_slo_scheduler_with_active_demand()
    older = make_running_seq(seq_id=21, needs_new_block=True)
    younger = make_running_seq(seq_id=22, needs_new_block=False)
    scheduler.running.extend([older, younger])
    scheduler.decode_progress_ns_by_seq_id.update({21: 1_000, 22: 2_000})
    exhaust_free_blocks(scheduler)
    before = scheduler_state_digest(scheduler)
    scheduler.schedule(10_000_000)
    assert scheduler.last_policy_branch == (
        "slo_mixed_transaction_fallback_decode"
    )
    assert scheduler.last_slo_decision["selected_chunk_tokens"] is not None
    assert scheduler.last_slo_decision["actual_prefill_tokens"] == 0
    assert no_prefill_mutation(before, scheduler)
```

- [ ] **Step 3: Run RED**

```bash
python3 tools/test_chunked_prefill.py
```

- [ ] **Step 4: Route P5 before P4 and build a complete decision template**

At the start of `schedule()`:

```python
waiting_depth = len(self.waiting)
self._maybe_reset_adaptive_mixed_controller()
if self.chunked_prefill_enabled and self.chunked_prefill_slo_mixed:
    return self._schedule_slo_mixed(waiting_depth, decision_now_ns)
if self.chunked_prefill_enabled and self.chunked_prefill_adaptive_mixed:
    return self._schedule_adaptive_mixed(waiting_depth)
```

Add:

```python
def _new_slo_decision(
    self,
    *,
    decision_now_ns: int | None,
    demand_state_before: str,
) -> dict:
    return {
        "decision_now_ns": decision_now_ns,
        "target_gap_ns": self.chunked_prefill_slo_target_gap_ns,
        "reserve_ns": self.chunked_prefill_slo_reserve_ns,
        "oldest_decode_seq_id": None,
        "oldest_decode_progress_ns": None,
        "oldest_decode_age_ns": None,
        "remaining_slack_ns": None,
        "cost_intercept_ns": self.chunked_prefill_slo_cost_intercept_ns,
        "cost_per_prefill_token_ns":
            self.chunked_prefill_slo_cost_per_prefill_token_ns,
        "candidate_chunk_tokens": list(self.slo_chunk_ladder),
        "predicted_step_ns": None,
        "selected_chunk_tokens": None,
        "actual_prefill_tokens": 0,
        "scheduled_decode_seq_ids": [],
        "demand_state_before": demand_state_before,
        "demand_state_after": demand_state_before,
        "suppression_reason": None,
        "clock_invalid": self.slo_clock_invalid,
        "clock_invalid_reason": self.slo_clock_invalid_reason,
    }
```

- [ ] **Step 5: Implement deterministic suppression order**

`_schedule_slo_mixed()` must apply this order:

```python
if not self.running:
    branch = "slo_mixed_no_running_prefill"
elif not self._validate_slo_decision_time(decision_now_ns):
    branch = "slo_mixed_clock_invalid_decode"
elif self.slo_clock_invalid:
    branch = "slo_mixed_clock_invalid_decode"
elif oldest is None:
    branch = "slo_mixed_missing_progress_decode"
elif self.adaptive_mixed_state == ADAPTIVE_MIXED_INACTIVE:
    branch = "slo_mixed_inactive_decode"
elif remaining_slack_ns <= 0:
    branch = "slo_mixed_no_slack_decode"
elif selected_chunk_tokens is None:
    branch = "slo_mixed_cost_suppressed_decode"
elif mixed is None:
    branch = "slo_mixed_transaction_fallback_decode"
elif demand_state_after == ADAPTIVE_MIXED_DRAINING:
    branch = "slo_mixed_draining_prefill_decode"
else:
    branch = "slo_mixed_prefill_decode"
```

Every decode-only return must publish the decision before calling `_schedule_decode()`.

- [ ] **Step 6: Extend the mixed helper with exact reservation and budget**

Replace `_mixed_decode_reservation()` with:

```python
def _mixed_decode_reservation(
    self,
    required_decode_seq_id: int | None = None,
) -> tuple[int, int] | None:
    if self.max_num_seqs < 2 or self.max_num_batched_tokens < 2:
        return None
    candidates = (
        [seq for seq in self.running
         if seq.seq_id == required_decode_seq_id]
        if required_decode_seq_id is not None
        else list(self.running)
    )
    for seq in candidates:
        required_free_blocks = int(
            len(seq) % self.block_manager.block_size == 1
        )
        if len(self.block_manager.free_block_ids) >= required_free_blocks:
            return seq.seq_id, required_free_blocks
    return None
```

Extend `_schedule_mixed_prefill_decode()`:

```python
reserved_seq_id = None
reserved_free_blocks = 0
if require_decode or required_decode_seq_id is not None:
    reservation = self._mixed_decode_reservation(required_decode_seq_id)
    if reservation is None:
        return None
    reserved_seq_id, reserved_free_blocks = reservation

prefill_slots = max(1, self.max_num_seqs - 1)
decode_query_tokens = 1 if self.running else 0
prefill_budget = (
    max(1, self.max_num_batched_tokens - decode_query_tokens)
    if max_prefill_tokens is None
    else min(
        max_prefill_tokens,
        max(1, self.max_num_batched_tokens - decode_query_tokens),
    )
)
prefill = self._schedule_chunked_prefill(
    max_prefill_seqs=prefill_slots,
    max_prefill_tokens=prefill_budget,
    allow_waiting_admission=allow_waiting_admission,
    reserved_free_blocks=reserved_free_blocks,
)
```

If `required_decode_seq_id` is supplied, require that exact reservation and never fall back to another row.

- [ ] **Step 7: Prove actual prefill is bounded and publish exact evidence**

After the helper succeeds:

```python
actual_prefill_tokens = sum(
    seq.prefill_chunk_end - seq.prefill_chunk_start
    for seq in mixed[0]
    if not getattr(seq, "step_is_decode", False)
)
scheduled_decode_seq_ids = [
    seq.seq_id
    for seq in mixed[0]
    if getattr(seq, "step_is_decode", False)
]
assert actual_prefill_tokens <= selected_chunk_tokens
assert oldest_decode_seq_id in scheduled_decode_seq_ids
decision.update({
    "selected_chunk_tokens": selected_chunk_tokens,
    "predicted_step_ns": predicted_step_ns,
    "actual_prefill_tokens": actual_prefill_tokens,
    "scheduled_decode_seq_ids": scheduled_decode_seq_ids,
})
```

- [ ] **Step 8: Add P0/P3/P4 regression tests**

Reuse existing fixtures and assert:

```python
def test_p0_p3_p4_scheduling_is_unchanged_by_p5_support():
    assert capture_policy_trace("P0") == EXPECTED_P0_TRACE
    assert capture_policy_trace("P3") == EXPECTED_P3_TRACE
    assert capture_policy_trace("P4") == EXPECTED_P4_TRACE
```

The expected traces must be literal branch/batch/queue digests captured from the pre-P5 committed behavior, not generated by the implementation under test.

- [ ] **Step 9: Run GREEN and commit**

```bash
python3 tools/test_chunked_prefill.py
python3 -m py_compile tinyvllm/engine/scheduler.py
git diff --check
git add tinyvllm/engine/scheduler.py tools/test_chunked_prefill.py
git commit -m "feat(scheduler): enforce decode-SLO mixed admission"
```

---

### Task 5: Preserve P5 Evidence Through the Arrival Driver

**Files:**
- Modify: `tools/arrival_load_driver.py`
- Modify: `tools/test_arrival_load_driver.py`

**Interfaces:**
- Consumes: complete engine-owned P5 observation.
- Produces: append-only scheduler rows retaining engine decision/end timing while keeping driver timing separately named.

- [ ] **Step 1: Write failing evidence-preservation tests**

Extend `FakeEngine.last_step_observation` with all P5 fields and assert exact survival:

```python
def test_driver_preserves_complete_p5_decision_and_postprocess_evidence():
    _, output_dir = _run(engine_factory=P5FakeEngine)
    rows = _jsonl(output_dir / "scheduler_trace.jsonl")
    row = rows[0]
    assert row["decision_now_ns"] == 1_000
    assert row["step_end_ns"] == 1_075
    assert row["actual_step_duration_ns"] == 75
    assert row["oldest_decode_seq_id"] == 4
    assert row["selected_chunk_tokens"] == 64
    assert row["scheduled_decode_seq_ids"] == [4, 5]
    assert row["decode_progress_updates"] == {"4": 1_075, "5": 1_075}
    assert row["driver_step_start_ns"] < row["driver_step_end_ns"]
```

Add a collision test:

```python
def test_driver_never_overwrites_engine_owned_timestamps():
    _, output_dir = _run(engine_factory=P5FakeEngine)
    row = _jsonl(output_dir / "scheduler_trace.jsonl")[0]
    assert row["decision_now_ns"] == 1_000
    assert row["step_end_ns"] == 1_075
    assert row["driver_step_start_ns"] != row["decision_now_ns"]
```

- [ ] **Step 2: Run RED**

```bash
python3 tools/test_arrival_load_driver.py
```

- [ ] **Step 3: Rename driver timing fields and validate P5 schema**

Change:

```python
driver_step_start_ns = clock_ns()
outputs, num_tokens = engine.step()
driver_step_end_ns = clock_ns()
observation.update({
    "step_index": step_index,
    "driver_step_start_ns": driver_step_start_ns,
    "driver_step_end_ns": driver_step_end_ns,
    "num_tokens_returned": num_tokens,
})
```

When `case_spec["policy"] == "P5"`, require all `P5_DECISION_FIELDS` and `P5_POSTPROCESS_FIELDS`. Validate integer/list/dict types, but do not recompute policy decisions in the driver.

- [ ] **Step 4: Use engine time for request token events**

For completion-token timestamps, use:

```python
token_event_ns = observation["step_end_ns"]
```

Use `driver_step_start_ns` only for `first_scheduled_ns`, because that is driver-level dispatch observation rather than P5 policy state.

- [ ] **Step 5: Run GREEN and commit**

```bash
python3 tools/test_arrival_load_driver.py
python3 -m py_compile tools/arrival_load_driver.py
git diff --check
git add tools/arrival_load_driver.py tools/test_arrival_load_driver.py
git commit -m "feat(eval): preserve P5 scheduler timing evidence"
```

---

### Task 6: Build the Source-Bound Cost Calibration Module

**Files:**
- Create: `tools/arrival_load_cost_calibration.py`
- Create: `tools/test_arrival_load_cost_calibration.py`
- Modify: `tools/arrival_load_gate.py`

**Interfaces:**
- Consumes: raw per-iteration synchronous durations and engine/environment limits.
- Produces: required shape manifest, validated raw rows, recomputed shape p99/inflation, minimal integer coefficients, and frozen artifact SHA-256.

- [ ] **Step 1: Write failing shape-manifest tests**

```python
def test_required_shapes_cover_decode_rows_contexts_and_mixed_cross_product():
    shapes = calibration.build_required_shapes(
        max_num_seqs=512,
        max_prefill_tokens=128,
    )
    decode = [shape for shape in shapes if shape["kind"] == "decode"]
    mixed = [shape for shape in shapes if shape["kind"] == "mixed"]
    assert {
        row["decode_rows"] for row in decode
    } == {1, 8, 32, 512}
    assert {
        row["context_class"] for row in decode
    } == {"short", "medium", "long"}
    assert {
        row["prefill_tokens"] for row in mixed
    } == {16, 32, 64, 128}
    assert {row["decode_rows"] for row in mixed} == {1, 8, 32, 512}
    assert all(row["measured_iterations"] == 7 for row in shapes)
    assert all(row["warmup_iterations"] >= 1 for row in shapes)
```

- [ ] **Step 2: Write failing integer-envelope tests**

```python
def test_integer_envelope_uses_observed_max_and_25_percent_ceiling():
    rows = synthetic_complete_rows()
    result = calibration.recompute_cost_envelope(rows)
    assert result["cost_intercept_ns"] == max(
        row["inflated_duration_ns"]
        for row in result["shape_summaries"]
        if row["kind"] == "decode"
    )
    assert all(
        result["cost_intercept_ns"]
        + row["prefill_tokens"]
        * result["cost_per_prefill_token_ns"]
        >= row["inflated_duration_ns"]
        for row in result["shape_summaries"]
        if row["kind"] == "mixed"
    )
    assert result["shape_summaries"][0]["measured_p99_ns"] == max(
        result["shape_summaries"][0]["measured_duration_ns"]
    )
```

Add tests for missing shapes, six samples, non-positive duration, NaN, duplicate iteration, infeasible shape, and int64 overflow.

- [ ] **Step 3: Run RED**

```bash
python3 tools/test_arrival_load_cost_calibration.py
```

- [ ] **Step 4: Implement exact pure functions**

The module must expose:

```python
def build_required_shapes(
    *,
    max_num_seqs: int,
    max_prefill_tokens: int,
) -> list[dict]:
    if max_num_seqs <= 0 or max_prefill_tokens != 128:
        raise ValueError("unsupported P5 calibration limits")
    decode_counts = sorted({
        count for count in (1, 8, 32, max_num_seqs)
        if count <= max_num_seqs
    })
    shapes = []
    for context_class in ("short", "medium", "long"):
        for decode_rows in decode_counts:
            shapes.append({
                "shape_id": (
                    f"decode-{context_class}-d{decode_rows}"
                ),
                "kind": "decode",
                "context_class": context_class,
                "decode_rows": decode_rows,
                "prefill_rows": 0,
                "prefill_tokens": 0,
                "warmup_iterations": 1,
                "measured_iterations": 7,
            })
    for prefill_tokens in (16, 32, 64, 128):
        prefill_rows = sorted({
            1,
            min(8, prefill_tokens // 16),
        })
        for decode_rows in decode_counts:
            for row_count in prefill_rows:
                shapes.append({
                    "shape_id": (
                        f"mixed-p{prefill_tokens}-r{row_count}"
                        f"-d{decode_rows}"
                    ),
                    "kind": "mixed",
                    "context_class": "mixed",
                    "decode_rows": decode_rows,
                    "prefill_rows": row_count,
                    "prefill_tokens": prefill_tokens,
                    "warmup_iterations": 1,
                    "measured_iterations": 7,
                })
    return shapes


def nearest_rank_p99_ns(values: list[int]) -> int:
    if len(values) < 7:
        raise ValueError("cost calibration requires at least seven samples")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        for value in values
    ):
        raise ValueError("invalid calibration duration")
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.99) - 1]


def inflate_duration_ns(measured_p99_ns: int) -> int:
    if (
        isinstance(measured_p99_ns, bool)
        or not isinstance(measured_p99_ns, int)
        or measured_p99_ns <= 0
    ):
        raise ValueError("invalid measured p99 duration")
    inflated = (measured_p99_ns * 5 + 3) // 4
    if inflated > INT64_MAX:
        raise OverflowError("calibration inflation overflows int64")
    return inflated


def recompute_cost_envelope(rows: list[dict]) -> dict:
    grouped = {}
    for row in rows:
        shape_id = row["shape_id"]
        iteration = row["iteration"]
        if iteration in grouped.setdefault(shape_id, {}):
            raise ValueError("duplicate calibration iteration")
        grouped[shape_id][iteration] = row
    shape_summaries = []
    for shape_id in sorted(grouped):
        shape_rows = grouped[shape_id]
        durations = [
            shape_rows[index]["duration_ns"]
            for index in sorted(shape_rows)
        ]
        measured_p99_ns = nearest_rank_p99_ns(durations)
        first = shape_rows[min(shape_rows)]
        shape_summaries.append({
            "shape_id": shape_id,
            "kind": first["kind"],
            "decode_rows": first["decode_rows"],
            "prefill_rows": first["prefill_rows"],
            "prefill_tokens": first["prefill_tokens"],
            "measured_duration_ns": durations,
            "measured_p99_ns": measured_p99_ns,
            "inflated_duration_ns": inflate_duration_ns(
                measured_p99_ns
            ),
        })
    decode_points = [
        row for row in shape_summaries if row["kind"] == "decode"
    ]
    mixed_points = [
        row for row in shape_summaries if row["kind"] == "mixed"
    ]
    if not decode_points or not mixed_points:
        raise ValueError("incomplete calibration shape classes")
    intercept = max(
        row["inflated_duration_ns"] for row in decode_points
    )
    slope = 1
    for point in mixed_points:
        tokens = point["prefill_tokens"]
        excess = max(0, point["inflated_duration_ns"] - intercept)
        slope = max(slope, (excess + tokens - 1) // tokens)
    max_tokens = max(row["prefill_tokens"] for row in mixed_points)
    if slope > (INT64_MAX - intercept) // max_tokens:
        raise OverflowError("cost envelope overflows int64")
    for point in mixed_points:
        predicted = intercept + point["prefill_tokens"] * slope
        if predicted < point["inflated_duration_ns"]:
            raise ValueError("cost envelope does not dominate point")
    return {
        "shape_summaries": shape_summaries,
        "cost_intercept_ns": intercept,
        "cost_per_prefill_token_ns": slope,
    }
```

Compute the minimal positive slope:

```python
slope = 1
for point in mixed_points:
    excess = max(0, point["inflated_duration_ns"] - intercept)
    required = max(1, (excess + point["prefill_tokens"] - 1)
                   // point["prefill_tokens"])
    slope = max(slope, required)
```

Then verify every point and int64 bound.

- [ ] **Step 5: Add a dedicated artifact contract**

Use:

```text
cost_calibration_manifest.jsonl
cost_calibration_rows.jsonl
cost_calibration_summary.json
```

The summary includes:

```python
{
    "status": "PASS",
    "source_tree_sha256": source_tree_sha256,
    "environment_sha256": environment_sha256,
    "engine_config_sha256": engine_config_sha256,
    "required_shape_sha256": required_shape_sha256,
    "raw_rows_sha256": raw_rows_sha256,
    "cost_intercept_ns": envelope["cost_intercept_ns"],
    "cost_per_prefill_token_ns":
        envelope["cost_per_prefill_token_ns"],
    "envelope_sha256": canonical_json_sha256(envelope),
}
```

- [ ] **Step 6: Run GREEN and commit**

```bash
python3 tools/test_arrival_load_cost_calibration.py
python3 -m py_compile tools/arrival_load_cost_calibration.py
git diff --check
git add tools/arrival_load_cost_calibration.py tools/test_arrival_load_cost_calibration.py tools/arrival_load_gate.py
git commit -m "feat(eval): add P5 cost-envelope calibration"
```

---

### Task 7: Freeze P0/P4/P5 Identity, Smoke, and Canonical Gate

**Files:**
- Modify: `tools/arrival_load_gate.py`
- Modify: `tools/test_arrival_load_gate.py`
- Modify: `tools/test_arrival_load_cost_calibration.py`

**Interfaces:**
- Consumes: Task 6 provisional smoke calibration and authoritative cost-calibration summary.
- Produces: P5 resolved identity, P0/P4/P5 54-case matrix, P5 smoke proof, structural counters, and P5-only classification.

- [ ] **Step 1: Write failing identity and matrix tests**

```python
def test_p5_policy_contract_is_frozen_and_source_bound():
    contract = gate._resolved_policy_contract(
        cost_envelope={
            "artifact_sha256": "a" * 64,
            "cost_intercept_ns": 4_000_000,
            "cost_per_prefill_token_ns": 100_000,
        }
    )
    assert tuple(contract["canonical_policy_by_name"]) == ("P0", "P4", "P5")
    p5 = contract["resolved_policy_config_by_name"]["P5"]
    assert p5["chunked_prefill_slo_mixed"] is True
    assert p5["chunked_prefill_slo_target_gap_ns"] == 64_000_000
    assert p5["chunked_prefill_slo_reserve_ns"] == 8_000_000
    assert p5["chunked_prefill_slo_min_chunk_tokens"] == 16
    assert p5["max_num_prefill_tokens_per_step"] == 128
    assert p5["chunked_prefill_slo_cost_intercept_ns"] == 4_000_000
    assert p5["chunked_prefill_slo_cost_per_prefill_token_ns"] == 100_000
    assert p5["chunked_prefill_slo_token_ladder"] == [
        128, 112, 96, 80, 64, 48, 32, 16
    ]
    assert p5["cost_calibration_artifact_sha256"] == "a" * 64
```

```python
def test_canonical_matrix_is_exactly_p0_p4_p5_54_cases():
    manifest = complete_p5_manifest()
    matrix = gate.build_case_matrix(manifest)
    assert len(matrix) == 54
    assert {row["policy"] for row in matrix} == {"P0", "P4", "P5"}
    assert all(row["policy"] != "P3" for row in matrix)
```

- [ ] **Step 2: Write failing smoke-proof tests**

```python
def test_p5_smoke_requires_all_preregistered_policy_paths():
    summary = gate.summarize_p5_smoke(
        synthetic_p5_smoke_rows()
    )
    assert summary["demand_activation_count"] >= 1
    assert summary["largest_chunk_admission_count"] >= 1
    assert summary["smaller_chunk_admission_count"] >= 1
    assert summary["slo_suppression_count"] >= 1
    assert summary["draining_decision_count"] >= 1
    assert summary["distinct_selected_chunk_tokens"] >= 2
    assert summary["classification"] == "SMOKE_ONLY"
```

Deleting every mixed row must produce `INCOMPLETE`, not `NO_GO`.

Because the spec orders smoke before the dedicated cost-calibration stage,
`run-smoke` must execute a source-bound **provisional** full-shape calibration
inside the smoke run before launching P5. Smoke records and verifies that
provisional artifact identity. The later `cost-calibration` stage reruns all
required shapes in fresh isolated processes and publishes the only envelope
that workload calibration and canonical may consume. The provisional smoke
coefficients are never copied into canonical evidence.

- [ ] **Step 3: Write failing P5 promotion guard tests**

Build synthetic P0/P4/P5 case rows and separately violate:

```text
any 10% p99/max-gap/service-bucket guard
mixed_service_fairness bucket > 1.10x
long_prompt_pressure p95 ITL > 1.05x in one repetition
burst median throughput < 1.25x
no burst repetition with three chunk sizes
no non-burst SLO suppression
envelope underprediction count > 0
no median-and-worst benefit path
```

Assert each produces P5 `NO_GO`, and P4 can never determine top-level `GO`.

- [ ] **Step 4: Run RED**

```bash
python3 tools/test_arrival_load_gate.py
```

- [ ] **Step 5: Update policy constants and resolved identity**

Set:

```python
POLICY_NAMES = ("P0", "P4", "P5")
POLICY_ORDER_BY_REPETITION = {
    0: ("P0", "P4", "P5"),
    1: ("P4", "P5", "P0"),
    2: ("P5", "P0", "P4"),
}
```

P5 overrides must include all P5 fields, consumed demand fields, exact ladder, and cost artifact SHA-256. P4 remains unchanged and diagnostic.

- [ ] **Step 6: Separate cost calibration from workload calibration**

Rename existing arrival-rate calibration commands/artifacts semantically to workload calibration while retaining file compatibility:

```text
cost-calibration       -> frozen P5 cost envelope
workload-calibration   -> existing P0 arrival-rate lambda_ref
```

Canonical initialization must require and copy both predecessor identities.

- [ ] **Step 7: Extend case summaries with P5 structural counters**

Per P5 case include:

```python
"p5_policy": {
    "mixed_decision_count": mixed_decision_count,
    "slo_suppression_count": slo_suppression_count,
    "draining_decision_count": draining_decision_count,
    "selected_chunk_histogram": dict(
        sorted(selected_chunk_histogram.items())
    ),
    "envelope_underprediction_count":
        envelope_underprediction_count,
    "missing_progress_count": missing_progress_count,
    "clock_invalid_count": clock_invalid_count,
}
```

These are harness diagnostics only; the independent verifier recomputes them.

- [ ] **Step 8: Implement exact P5 classification**

Retain existing benefit paths and 10% guards. Add preregistered P5 requirements exactly. Return top-level:

```python
classification = candidate_results["P5"]["classification"]
```

Never promote from P4.

- [ ] **Step 9: Run GREEN and commit**

```bash
python3 tools/test_arrival_load_gate.py
python3 tools/test_arrival_load_cost_calibration.py
python3 -m py_compile tools/arrival_load_gate.py
git diff --check
git add tools/arrival_load_gate.py tools/test_arrival_load_gate.py tools/test_arrival_load_cost_calibration.py
git commit -m "feat(gate): freeze P5 arrival-load policy"
```

---

### Task 8: Independently Reconstruct Every P5 Decision

**Files:**
- Modify: `tools/arrival_load_verify.py`
- Modify: `tools/test_arrival_load_verify.py`

**Interfaces:**
- Consumes: raw calibration rows, run identity, scheduler trace, timeline, memory trace, and recorded summaries.
- Produces: independent P5 structural validation and classification.

- [ ] **Step 1: Replace the synthetic artifact with complete P0/P4/P5 evidence**

The fixture must include:

```text
54 canonical case rows
valid source/environment identities
complete cost-calibration manifest/rows/summary
complete workload calibration
P5 progress initialization
P5 active and draining transitions
128-token and smaller mixed admissions
no-slack/cost suppression
actual mixed durations under envelope
exact outputs and complete lifecycle
```

Do not reuse P4 scheduler rows as P5 evidence.

- [ ] **Step 2: Write one failing test per tamper field**

Use a helper:

```python
def _mutate_first_p5_trace(root: Path, mutate) -> None:
    rows = verifier._read_jsonl(root / "scheduler_trace.jsonl")
    row = next(row for row in rows if row["policy"] == "P5")
    mutate(row)
    _write_jsonl(root / "scheduler_trace.jsonl", rows)
    _refresh_hash(root, "scheduler_trace.jsonl")
```

Add separate tests for:

```python
tamper_cases = {
    "decision time": lambda row: row.__setitem__("decision_now_ns", 1),
    "progress timestamp": lambda row: row["decode_progress_updates"].__setitem__("1", 1),
    "oldest identity": lambda row: row.__setitem__("oldest_decode_seq_id", 999),
    "age": lambda row: row.__setitem__("oldest_decode_age_ns", 1),
    "slack": lambda row: row.__setitem__("remaining_slack_ns", 1),
    "coefficient": lambda row: row.__setitem__("cost_per_prefill_token_ns", 1),
    "selected chunk": lambda row: row.__setitem__("selected_chunk_tokens", 16),
    "protected row": lambda row: row.__setitem__("scheduled_decode_seq_ids", [999]),
    "actual prefill": lambda row: row.__setitem__("actual_prefill_tokens", 129),
    "suppression": lambda row: row.__setitem__("suppression_reason", "forged"),
    "progress update": lambda row: row.__setitem__("decode_progress_updates", {}),
}
```

Also independently tamper one calibration row, source identity, environment identity, and workload identity. Every test must raise `ValueError`.

- [ ] **Step 3: Run RED**

```bash
python3 tools/test_arrival_load_verify.py
```

- [ ] **Step 4: Recompute the cost envelope from raw rows**

Load `arrival_load_cost_calibration.py` by file path. Verify required shape set, seven samples, p99, inflation, coefficients, envelope dominance, artifact SHA-256, and source/environment/engine identity.

Never trust published coefficients.

- [ ] **Step 5: Reconstruct P5 state per case**

For each P5 case, initialize:

```python
progress_by_seq_id = {}
demand_state = "inactive"
high_streak = 0
low_streak = 0
clock_invalid = False
last_decision_now_ns = None
```

For every trace row in step order:

1. Verify queue continuity: previous `queue_after` equals current `queue_before`.
2. Reconstruct demand transition from `len(waiting_seq_ids)` sampled once.
3. Verify decision time monotonicity and progress timestamps.
4. Reconstruct runnable decode IDs from `queue_before["running_seq_ids"]`.
5. Require progress for every runnable ID or verify fail-closed missing-progress branch semantics.
6. Recompute oldest row using `(progress_ns, seq_id)` minimum.
7. Recompute age and slack.
8. Rebuild the token ladder.
9. Recompute largest safe selected chunk.
10. If suppressed, prove zero prefill rows/tokens.
11. If mixed, prove both row types, exact oldest-row inclusion, selected-budget bound, and actual duration.
12. Compare actual mixed duration with predicted envelope and count underprediction.
13. Reconstruct token-producing progress updates from `scheduled` plus `new_completion_tokens_by_seq`.
14. Remove finished progress entries.
15. Preserve progress through preemption and exclude non-running rows.

- [ ] **Step 6: Recompute P5 diagnostics and classification**

Derive all P5 structural counters from trace rows, then recompute request/fairness/memory metrics and exact promotion requirements. Compare exact JSON equality with recorded `case_rows.jsonl` and `summary.json`.

- [ ] **Step 7: Run GREEN and commit**

```bash
python3 tools/test_arrival_load_verify.py
python3 -m py_compile tools/arrival_load_verify.py
git diff --check
git add tools/arrival_load_verify.py tools/test_arrival_load_verify.py
git commit -m "feat(verifier): reconstruct P5 SLO admission"
```

---

### Task 9: Harden the Remote Chain and Local Completion Suite

**Files:**
- Modify: `tools/run_arrival_load_gate_remote.sh`
- Modify: `tools/test_run_arrival_load_gate_remote.py`
- Modify: `tools/arrival_load_gate.py`

**Interfaces:**
- Consumes: Tasks 6–8 stage contracts.
- Produces: exact `preflight → smoke → cost-calibration → workload-calibration → canonical → local verify` chain.

- [ ] **Step 1: Write failing runner-mode and predecessor tests**

Assert exact modes:

```python
for mode in (
    "preflight",
    "smoke",
    "cost-calibration",
    "workload-calibration",
    "canonical",
    "download-only",
    "verify-only",
):
    assert mode in runner
```

Assert canonical requires:

```text
SMOKE_RUN_TAG
COST_CALIBRATION_RUN_TAG
WORKLOAD_CALIBRATION_RUN_TAG
```

Assert workload calibration requires smoke and cost calibration.

- [ ] **Step 2: Write failing safety tests**

Require literal:

```text
sitian@10.232.195.203
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python
/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B
```

Reject `rsync`, `pkill`, `killall`, remote checkout paths, `rm -rf /tmp`, and static model ports. Prove local independent verifier runs after smoke and canonical download and its exit code is checked.

- [ ] **Step 3: Run RED**

```bash
python3 tools/test_run_arrival_load_gate_remote.py
```

- [ ] **Step 4: Add exact modes and predecessor arguments**

Update usage and case matching. Map:

```text
cost-calibration      -> run-cost-calibration-remote
workload-calibration  -> run-workload-calibration-remote
canonical             -> run-canonical with all predecessor dirs
```

Keep immutable source snapshot upload and detached model process behavior.

- [ ] **Step 5: Add remote preflight tests for the new module**

Remote preflight runs:

```bash
"${REMOTE_PYTHON}" tools/test_arrival_load_cost_calibration.py
"${REMOTE_PYTHON}" tools/test_arrival_load_gate.py
"${REMOTE_PYTHON}" tools/test_arrival_load_driver.py
"${REMOTE_PYTHON}" tools/test_arrival_load_verify.py
"${REMOTE_PYTHON}" tools/test_chunked_prefill.py
```

- [ ] **Step 6: Keep dynamic ports inside the Python orchestrator**

The shell must not export fixed `TINYVLLM_DIST_PORT` or `MASTER_PORT`. Every isolated calibration shape and every smoke/canonical case obtains a fresh `allocate_port_pair()` and records it in process metadata.

- [ ] **Step 7: Run complete local validation**

```bash
python3 tools/test_chunked_prefill.py
python3 tools/test_arrival_load_cost_calibration.py
python3 tools/test_arrival_load_gate.py
python3 tools/test_arrival_load_driver.py
python3 tools/test_arrival_load_verify.py
python3 tools/test_run_arrival_load_gate_remote.py
python3 -m py_compile \
  tinyvllm/config.py \
  tinyvllm/engine/scheduler.py \
  tinyvllm/engine/llm_engine.py \
  tools/arrival_load_cost_calibration.py \
  tools/arrival_load_gate.py \
  tools/arrival_load_driver.py \
  tools/arrival_load_verify.py
bash -n tools/run_arrival_load_gate_remote.sh
git diff --check
```

Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add \
  tools/run_arrival_load_gate_remote.sh \
  tools/test_run_arrival_load_gate_remote.py \
  tools/arrival_load_gate.py
git commit -m "feat(eval): bind P5 remote evidence chain"
```

---

### Task 10: Execute the Source-Bound Remote Experiment

**Files:**
- Create untracked: `experiments/arrival_load/<run-tags>/`
- Modify tracked docs only after independent verification.

**Interfaces:**
- Consumes: committed source and complete local suite.
- Produces: source-bound P5 result with independent `GO`, `PROMISING_NOT_PROVEN`, `NO_GO`, or `INCOMPLETE`.

- [ ] **Step 1: Record immutable source state**

Run:

```bash
git status --short --branch
git rev-parse HEAD
git diff --check
```

Expected: tracked tree clean; only preserved untracked `experiments/` roots may remain.

- [ ] **Step 2: Establish the approved SSH route**

Use:

```bash
export KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian
ssh -MNf \
  -o BatchMode=yes \
  -o ControlMaster=yes \
  -o ControlPersist=600 \
  -S /tmp/ssh-sitian-10.232.195.203 \
  sitian@10.232.195.203
```

If an existing healthy ControlMaster is present, reuse it. Do not kill unrelated SSH processes.

- [ ] **Step 3: Run preflight**

```bash
RUN_TAG=qwen3-06b-p5-preflight-$(date +%Y%m%d-%H%M%S) \
  bash tools/run_arrival_load_gate_remote.sh preflight
```

Stop on any source/environment/test mismatch.

- [ ] **Step 4: Run smoke**

```bash
SMOKE_RUN_TAG=qwen3-06b-p5-smoke-$(date +%Y%m%d-%H%M%S)
RUN_TAG="${SMOKE_RUN_TAG}" \
  bash tools/run_arrival_load_gate_remote.sh smoke
```

Require local independent verification and all smoke proof counters. If no mixed step is exercised, classify `INCOMPLETE`; do not tune the target or coefficients.

`run-smoke` first runs the complete required calibration-shape matrix in
isolated processes, marks it `purpose="provisional_smoke"`, derives a
source-bound envelope, and then launches P0/P5 smoke with that envelope.
This satisfies the smoke cost-identity contract without guessing constants.
The provisional envelope is diagnostic and cannot be a predecessor of
workload calibration or canonical.

- [ ] **Step 5: Run cost calibration**

```bash
COST_CALIBRATION_RUN_TAG=qwen3-06b-p5-cost-$(date +%Y%m%d-%H%M%S)
RUN_TAG="${COST_CALIBRATION_RUN_TAG}" \
SMOKE_RUN_TAG="${SMOKE_RUN_TAG}" \
  bash tools/run_arrival_load_gate_remote.sh cost-calibration
```

Require a fresh independent execution of all shapes, seven measured
iterations, frozen coefficients, and matching source/environment/engine
identities. Verify that both independent calibrations are valid; canonical
uses only this authoritative `purpose="canonical"` envelope.

- [ ] **Step 6: Run workload calibration**

```bash
WORKLOAD_CALIBRATION_RUN_TAG=qwen3-06b-p5-workload-$(date +%Y%m%d-%H%M%S)
RUN_TAG="${WORKLOAD_CALIBRATION_RUN_TAG}" \
SMOKE_RUN_TAG="${SMOKE_RUN_TAG}" \
COST_CALIBRATION_RUN_TAG="${COST_CALIBRATION_RUN_TAG}" \
  bash tools/run_arrival_load_gate_remote.sh workload-calibration
```

Require frozen `lambda_ref`, frozen workload manifest, and both predecessor identities.

- [ ] **Step 7: Run canonical 54-case matrix**

```bash
CANONICAL_RUN_TAG=qwen3-06b-p5-canonical-$(date +%Y%m%d-%H%M%S)
RUN_TAG="${CANONICAL_RUN_TAG}" \
SMOKE_RUN_TAG="${SMOKE_RUN_TAG}" \
COST_CALIBRATION_RUN_TAG="${COST_CALIBRATION_RUN_TAG}" \
WORKLOAD_CALIBRATION_RUN_TAG="${WORKLOAD_CALIBRATION_RUN_TAG}" \
  bash tools/run_arrival_load_gate_remote.sh canonical
```

Poll until the detached process publishes atomic exit status. On interruption, resume only the same source-bound run with the same predecessor tags.

- [ ] **Step 8: Re-run local independent verification**

```bash
RUN_TAG="${CANONICAL_RUN_TAG}" \
  bash tools/run_arrival_load_gate_remote.sh verify-only
cat "experiments/arrival_load/${CANONICAL_RUN_TAG}/independent-verify/verify.exitcode"
```

Expected: `0`.

- [ ] **Step 9: Audit the actual result before documentation**

Inspect:

```bash
cat "experiments/arrival_load/${CANONICAL_RUN_TAG}/independent-verify/summary.json"
cat "experiments/arrival_load/${CANONICAL_RUN_TAG}/independent-verify/report.md"
```

Confirm:

```text
54 unique canonical cases
P0/P4/P5 only
all required artifacts and hashes
zero correctness/structural failures
zero envelope underprediction for GO
all 10% guards
fairness and long-prompt requirements
burst >=1.25x median throughput
three chunk sizes in one burst repetition
non-burst suppression
median-and-worst benefit path
```

- [ ] **Step 10: Apply the promotion boundary**

If `GO`:

```text
Update README.md with exact model/GPU/source/run scope, independently
recomputed ratios, guards, and limitations.
Update AGENT_HANDOFF_STATE.md with result and next steps.
```

If `PROMISING_NOT_PROVEN`, `NO_GO`, or `INCOMPLETE`:

```text
Do not add a README performance claim.
Update AGENT_HANDOFF_STATE.md with exact classification, evidence path,
what passed, what failed, what the result proves, and the stop-condition
decision.
```

- [ ] **Step 11: Validate and commit documentation only**

```bash
git diff --check
git status --short
git add AGENT_HANDOFF_STATE.md
if grep -q 'Classification: `GO`' \
  "experiments/arrival_load/${CANONICAL_RUN_TAG}/independent-verify/report.md"
then
  git add README.md
fi
git commit -m "docs: record P5 arrival-load result"
```

Never stage raw experiment artifacts.

---

## Final Completion Audit

Before calling P5 complete, restate the concrete objective:

```text
Produce a disabled-by-default decode-SLO-aware mixed-admission policy and a
source-bound independent answer to whether it eliminates P4 decode-tail and
fairness regressions while preserving a repeatable throughput, latency, or
memory benefit.
```

Build and verify this prompt-to-artifact checklist:

| Requirement | Required evidence |
|---|---|
| Disabled default and fail-closed combinations | `tinyvllm/config.py`; `tools/test_chunked_prefill.py` |
| One engine-owned monotonic decision/end timestamp | `tinyvllm/engine/llm_engine.py`; fake-clock test |
| Scheduler-local progress lifecycle | scheduler tests for first token, decode, mixed, preemption, finish, empty |
| Sticky clock invalidation | regression/future/end-before-start tests |
| Exact integer slack and fixed ladder | pure helper tests and trace reconstruction |
| Exact oldest-row inclusion | scheduler reservation tests and independent verifier |
| No mutation before approval | queue/KV/status digest tests |
| Actual prefill bounded by selected chunk | scheduler assertion and verifier |
| P0/P3/P4 unchanged | literal regression traces |
| Complete cost shapes and seven samples | cost manifest/rows and independent recomputation |
| Nearest-rank p99 plus 25% integer headroom | cost-calibration tests and verifier |
| Source/environment/engine identity | preflight, predecessor manifests, hashes |
| Smoke exercises large/small/suppressed/draining paths | independent smoke summary |
| Exact P0/P4/P5 54-case matrix | run manifest, case rows, verifier |
| Exact outputs and lifecycle | timeline/case recomputation |
| Tail/fairness/long-prompt guards | independent summary |
| Burst opportunity and chunk diversity | independent P5 diagnostics |
| Zero envelope underprediction for GO | reconstructed scheduler trace |
| P5-only top-level classification | independent verifier code and test |
| Correct documentation boundary | README only on GO; handoff for all outcomes |
| Raw artifacts untracked | final `git status --short` |

Do not accept local tests, a complete manifest, harness `GO`, or a remote exit code as sufficient by itself. Completion requires the local independent verifier to cover every row above and publish exit code `0`.

If the result is `NO_GO`, enforce the spec stop condition: do not retune the same workload after reading canonical results. Move the next investigation to kernel/CUDA Graph overhead reduction or quantization with quality and memory-capacity gates.
