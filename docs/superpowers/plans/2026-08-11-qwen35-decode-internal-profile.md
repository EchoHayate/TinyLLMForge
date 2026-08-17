# Qwen3.5 TP4 Decode-Internal Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce guarded TP4 evidence that separates first decode-step latency, steady-state latency, TP collective CUDA time, CUDA execution, and the remaining host/synchronization upper bound for recompute versus exact prefix restore.

**Architecture:** Add an opt-in rank-local profiler to `ModelRunner`, expose its snapshots through the existing acknowledged engine RPC, and let the benchmark adapter align engine steps with rank steps. Write a separate `decode_profile.json`, aggregate five paired repetitions with a pure validation module, and run one median-representative pair under Nsight Systems without changing canonical benchmark artifacts.

**Tech Stack:** Python, pytest, PyTorch CUDA Events, `torch.distributed`, NVTX, NVIDIA Nsight Systems 2024.7, TinyLLMForge TP4 engine, JSON artifacts, SSH.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not modify `/Users/bytedance/dev/TinyLLMForge`.
- Do not modify the canonical workload manifest, case matrix, case-row schema, existing `profile.json` schema, or existing r607-r611 artifacts.
- Use the eight-token `w2_long_reuse` diagnostic workload.
- Use fixed GPUs `2,4,5,6`.
- Require at least 25 GiB free and at most 10 percent utilization per GPU at entry and worker entry.
- Allow unrelated low-utilization GPU processes; mark results shared and non-exclusive.
- Do not use dummy reservations or kill unrelated processes.
- Use a fresh run tag and attempt-scoped cleanup.
- Do not switch branches, stage, commit, stash, reset, push, or run `git clean`.
- Avoid per-event `torch.cuda.synchronize()`; resolve CUDA Events once during profile finalization.
- Treat `step_wall_ns - step_cuda_ns` only as a non-CUDA/synchronization upper bound.

---

### Task 1: Pure Decode Profile Contract and Aggregator

**Files:**
- Create: `tools/qwen35_tp4_decode_internal_profile.py`
- Create: `tools/test_qwen35_tp4_decode_internal_profile.py`

**Interfaces:**
- Produces: `validate_decode_profile(payload: dict) -> dict`
- Produces: `aggregate_decode_profiles(root: Path) -> dict`
- Produces: `select_representative_repetition(pairs: list[dict]) -> int`
- Produces: CLI `--input-root`, `--output`, and optional `--nsys-summary`.
- Consumes: ten measured `decode_profile.json` files plus existing case rows.

- [ ] **Step 1: Write RED tests for one valid four-rank profile**

Create fixtures with four aligned ranks, one prefill row, decode ordinals
`0..6`, collective events attached to decode steps, and deterministic wall and
CUDA durations. Assert:

```python
validated = profile.validate_decode_profile(payload)
assert validated["rank_inventory"] == [0, 1, 2, 3]
assert validated["decode_ordinals"] == list(range(7))
assert validated["first_decode_ordinal"] == 0
assert validated["steady_decode_ordinals"] == list(range(1, 7))
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
pytest -q tools/test_qwen35_tp4_decode_internal_profile.py \
  -k 'valid_four_rank_profile'
```

Expected: collection or import failure because
`qwen35_tp4_decode_internal_profile.py` does not exist.

- [ ] **Step 3: Implement schema constants and validation**

Use:

```python
SCHEMA_VERSION = "qwen35.tp4-decode-internal-case.v1"
SUMMARY_SCHEMA_VERSION = "qwen35.tp4-decode-internal-summary.v1"
POLICIES = ("recompute", "exact_restore")
REPETITIONS = tuple(range(5))
RANKS = tuple(range(4))
GENERATED_TOKENS = 8
```

Require identity fields, units `"nanoseconds"`, finalization status
`"complete"`, four ranks, ordered step alignment, non-negative integer
durations, collective-to-step association, and generated-token count eight.

- [ ] **Step 4: Run the focused valid-profile test and verify GREEN**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Write RED rejection tests**

Add one test each for:

```text
missing rank
rank step-order mismatch
request-set digest mismatch
negative duration
collective referencing unknown step
generated_tokens != 8
finalization_status != complete
identity mismatch
```

- [ ] **Step 6: Run rejection tests and verify RED**

Run:

```bash
pytest -q tools/test_qwen35_tp4_decode_internal_profile.py \
  -k 'rejects'
```

Expected: at least one assertion fails because strict validation is incomplete.

- [ ] **Step 7: Complete strict validation and verify GREEN**

Implement precise `ValueError` messages containing the case identity, rank,
step index, or field name. Re-run Step 6; expected: PASS.

- [ ] **Step 8: Write RED aggregation tests**

Build five paired repetitions and assert:

```python
summary["measured_pairs"] == 5
summary["generated_tokens"] == 8
summary["first_step"]["paired_ratios"] == expected_first_ratios
summary["steady_state"]["paired_ratios"] == expected_steady_ratios
summary["representative_repetition"] == expected_nearest_median
summary["by_policy"]["exact_restore"]["median_collective_cuda_ns"] == expected
```

Also assert warmup and `nsys_replay` cases are excluded.

- [ ] **Step 9: Run aggregation tests and verify RED**

Run:

```bash
pytest -q tools/test_qwen35_tp4_decode_internal_profile.py \
  -k 'aggregate or representative or excludes'
```

Expected: FAIL because aggregation is absent.

- [ ] **Step 10: Implement aggregation and frozen classifications**

Use the maximum rank duration as the primary aligned-step duration. Compute:

```text
first-step wall/CUDA/collective
steady-state median and p90 wall/CUDA/collective
non-CUDA upper bound
cross-rank max-minus-min wall/CUDA
paired ratios
ratio of policy medians
direction agreement
four-of-five directional consistency
```

Freeze materiality thresholds:

```python
RELATIVE_REGRESSION = 1.03
FIRST_STEP_ABSOLUTE_NS = 2_000_000
STEADY_STEP_ABSOLUTE_NS = 1_000_000
COLLECTIVE_ABSOLUTE_NS = 500_000
NON_CUDA_ABSOLUTE_NS = 1_000_000
```

Classify in this precedence:

```text
mixed_or_inconclusive
collective_regression
first_step_regression
steady_state_regression
non_cuda_or_sync_upper_bound_regression
no_material_decode_regression
```

Require paired-median and ratio-of-medians direction agreement plus at least
four of five same-direction pairs before assigning a specific regression.

- [ ] **Step 11: Verify aggregation GREEN**

Run all tests in the new test file. Expected: PASS.

### Task 2: Rank-Local Runtime Profiler

**Files:**
- Create: `tinyvllm/engine/decode_internal_profiler.py`
- Create: `tools/test_decode_internal_profiler.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/layers/linear.py`
- Modify: `tinyvllm/layers/embed_head.py`

**Interfaces:**
- Produces: `DecodeInternalProfiler`
- Produces: `ModelRunner.configure_decode_internal_profile(enabled: bool) -> dict`
- Produces: `ModelRunner.finalize_decode_internal_profile() -> dict`
- Consumes: model-runner step metadata and explicit collective call-site labels.

- [ ] **Step 1: Write RED unit tests with fake CUDA Events**

Define a fake event factory whose events record once and resolve to fixed
milliseconds. Test:

```python
profiler.begin_step(...)
with profiler.collective("row_parallel_all_reduce", tensor):
    pass
profiler.end_step()
snapshot = profiler.finalize()
```

Assert one step, one collective, nanosecond conversion, tensor shape/dtype,
rank, step index, decode ordinal, and one finalizer synchronization.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
pytest -q tools/test_decode_internal_profiler.py
```

Expected: import failure because the profiler module is absent.

- [ ] **Step 3: Implement the minimal profiler**

The profiler stores unresolved event pairs in memory. `begin_step()` records
CPU start and a CUDA start Event. `collective()` is a context manager that
records CPU and CUDA boundaries only when an active decode step exists.
`end_step()` records CPU and CUDA end boundaries. `finalize()` performs one
device synchronization, resolves all event pairs, computes:

```python
step_cuda_ns = round(start.elapsed_time(end) * 1_000_000)
non_cuda_upper_bound_ns = max(0, step_wall_ns - step_cuda_ns)
```

and makes the profiler immutable.

- [ ] **Step 4: Verify profiler GREEN**

Run Step 2. Expected: PASS.

- [ ] **Step 5: Write RED lifecycle and failure tests**

Cover:

```text
disabled profiler is no-op
double begin rejects
end without begin rejects
finalize with active step rejects
finalize twice returns identical snapshot
exception closes a collective event
events cannot be added after finalization
prefill rows carry no decode ordinal
first decode is ordinal zero and later decode increments
```

- [ ] **Step 6: Run lifecycle tests and verify RED**

Expected: failures for missing lifecycle guards.

- [ ] **Step 7: Implement lifecycle guards and verify GREEN**

Re-run the complete profiler test file. Expected: PASS.

- [ ] **Step 8: Write RED ModelRunner integration tests**

Use construction-free `ModelRunner.__new__` fixtures. Assert:

```python
runner.configure_decode_internal_profile(True)
runner.run(...)
snapshot = runner.finalize_decode_internal_profile()
```

records the request-set digest, active sequence count, batch kind, dispatch
metadata, and restores clean disabled state after finalization.

- [ ] **Step 9: Run integration tests and verify RED**

Run:

```bash
pytest -q tools/test_decode_internal_profiler.py \
  -k 'model_runner'
```

Expected: missing ModelRunner methods.

- [ ] **Step 10: Integrate profiler around `ModelRunner.run()`**

Configure all ranks before measured execution. Wrap the complete GPU-bearing
run region with `try/finally`. Derive decode classification from `is_prefill`
and `batch_kind`; derive a stable SHA256 digest from sorted sequence IDs.
Publish the current profiler through a process-local context variable so
collective call sites can find it without changing linear-layer signatures.

- [ ] **Step 11: Instrument explicit collective call sites**

Wrap only:

```text
RowParallelLinear.forward all_reduce
chunked RowParallelLinear.forward all_reduce
VocabParallelEmbedding.forward all_reduce
output-head collective reached by this model, if separate
```

Each wrapper uses a stable operation label and calls the original
`dist.all_reduce` exactly once with unchanged arguments.

- [ ] **Step 12: Verify ModelRunner and collective GREEN**

Run:

```bash
pytest -q \
  tools/test_decode_internal_profiler.py \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py
```

Expected: PASS.

### Task 3: Engine RPC, Adapter Alignment, and Worker Artifact

**Files:**
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py`
- Modify: `tools/qwen35_tp4_hybrid_prefix_benchmark_worker.py`
- Modify: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py`
- Modify: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_worker.py`

**Interfaces:**
- Produces: `LLMEngine.configure_decode_internal_profile(enabled, timeout_s)`
- Produces: `LLMEngine.finalize_decode_internal_profile(timeout_s)`
- Extends profile snapshot with `decode_internal`.
- Writes independent `decode_profile.json`.

- [ ] **Step 1: Write RED acknowledged-RPC tests**

Assert configuration and finalization return rank inventory `[0,1,2,3]`, reject
duplicate/missing rank acknowledgements, and preserve rank-specific snapshots.

- [ ] **Step 2: Run and verify RED**

Run the relevant engine adapter test selectors. Expected: missing engine RPC.

- [ ] **Step 3: Implement engine RPC**

Use `call_model_runner_acknowledged()` for enable and finalization. Normalize
the local result and worker acknowledgements into sorted rank rows and reject
incomplete inventories.

- [ ] **Step 4: Verify RPC GREEN**

Re-run Step 2. Expected: PASS.

- [ ] **Step 5: Write RED adapter alignment tests**

Extend fake observations to include prefill and seven decode steps. Assert the
adapter:

```text
enables profiling before workload execution
captures engine step indices and scheduled request identities
finalizes after workload completion
aligns rank rows to engine decode ordinals
rejects rank/engine mismatch
exposes decode_internal in profile_snapshot()
```

- [ ] **Step 6: Run and verify RED**

Expected: failures because adapter alignment is absent.

- [ ] **Step 7: Implement adapter alignment**

Install lightweight step observation collection around existing
`engine.step()`. Do not time restore internals further. On finalization, join
engine observations and rank snapshots by local step order, decode flag,
active sequence count, and request-set digest.

- [ ] **Step 8: Verify adapter GREEN**

Run the full engine adapter test file. Expected: PASS.

- [ ] **Step 9: Write RED worker artifact tests**

Run a profile case with a fake engine and assert:

```text
profile.json remains byte-schema compatible
decode_profile.json exists separately
schema_version is qwen35.tp4-decode-internal-case.v1
resource_policy is shared-low-utilization
exclusive is false
source/workload/case/repetition identities are present
failed finalization writes failure.json and no complete decode profile
```

- [ ] **Step 10: Run and verify RED**

Expected: missing independent artifact.

- [ ] **Step 11: Implement worker artifact emission**

Add `decode_internal=True` as a profile-only worker option. Require
`profiling=True`, `w2_long_reuse`, and generated-token override eight.
Validate the decoded snapshot using Task 1 before atomically writing
`decode_profile.json`.

- [ ] **Step 12: Verify worker GREEN**

Run:

```bash
pytest -q \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_worker.py \
  tools/test_qwen35_tp4_decode_internal_profile.py
```

Expected: PASS.

### Task 4: Nsight Command and Guarded Attempt Orchestration

**Files:**
- Create: `tools/run_qwen35_tp4_decode_internal_profile.py`
- Create: `tools/test_run_qwen35_tp4_decode_internal_profile.py`

**Interfaces:**
- Produces: a local orchestrator for entry guard, staging, 12 structured
  workers, representative-pair selection, two Nsight replay workers, download,
  aggregation, and cleanup.
- Reuses: existing SSH route, resource sampling, source staging, benchmark
  worker command, and attempt cleanup conventions.

- [ ] **Step 1: Write RED command-construction tests**

Assert structured commands include:

```text
--profile
--generated-tokens-override 8
--decode-internal-profile
CUDA_VISIBLE_DEVICES=2,4,5,6 mapping
fresh attempt-scoped output directories
```

Assert Nsight command starts with:

```text
/usr/local/bin/nsys profile
--trace=cuda,nvtx,osrt,nccl
```

and uses a separate `nsys_replay` identity.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
pytest -q tools/test_run_qwen35_tp4_decode_internal_profile.py
```

Expected: import failure because the runner is absent.

- [ ] **Step 3: Implement pure command builders**

Make source path, host, GPU list, run tag, attempt root, and `nsys` path
explicit parameters. Reuse existing helpers rather than duplicating SSH
quoting.

- [ ] **Step 4: Verify command builders GREEN**

Re-run Step 2. Expected: command tests pass.

- [ ] **Step 5: Write RED state-machine tests**

Use a fake command runner to cover:

```text
entry guard blocked -> no staging or worker
worker-entry guard blocked -> no measured worker
structured worker failure -> preserve attempt and cleanup
ten measured profiles -> aggregate and select representative repetition
Nsight unavailable -> structured result preserved with explicit status
Nsight success -> reports exported
cleanup incomplete -> final classification fails
```

- [ ] **Step 6: Run state-machine tests and verify RED**

Expected: failures because orchestration is incomplete.

- [ ] **Step 7: Implement orchestration and receipts**

Run one paired warmup plus five measured pairs. After local aggregation,
select the median-nearest repetition and run recompute/exact replays under
Nsight. Export:

```text
nsys stats --report cuda_gpu_kern_sum
nsys stats --report nccl_sum
```

when supported. Record unavailable reports without fabricating metrics.

- [ ] **Step 8: Verify orchestration GREEN**

Run the complete runner test file. Expected: PASS.

### Task 5: Local Validation and Real Guarded TP4 Run

**Files:**
- Create: a fresh directory under
  `experiments/qwen35_hybrid_state/qwen35-tp4-decode-internal-profile-20260811-<tag>-attempt001/`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Produces: ten measured `decode_profile.json` artifacts, five paired
  summaries, representative Nsight evidence, guard receipts, and cleanup.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
pytest -q \
  tools/test_decode_internal_profiler.py \
  tools/test_qwen35_tp4_decode_internal_profile.py \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_worker.py \
  tools/test_run_qwen35_tp4_decode_internal_profile.py
```

Expected: all pass.

- [ ] **Step 2: Run static validation**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/decode_internal_profiler.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/qwen35_tp4_decode_internal_profile.py \
  tools/qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py \
  tools/qwen35_tp4_hybrid_prefix_benchmark_worker.py \
  tools/run_qwen35_tp4_decode_internal_profile.py
git diff --check
```

Expected: exit code zero.

- [ ] **Step 3: Launch one fresh guarded attempt**

Use the runner with fixed GPUs `2,4,5,6`, 25 GiB, 10 percent utilization, a
fresh r-tag, shared/non-exclusive metadata, and no dummy reservation.

- [ ] **Step 4: Handle blocked guard autonomously**

If entry or worker-entry is blocked, preserve the attempt, monitor for a
qualifying window using the relaxed shared-low-utilization condition, and
launch a new tag when ready. Do not extend a worker timeout as a substitute
for resource monitoring.

- [ ] **Step 5: Verify structured evidence**

Require:

```text
12/12 workers complete
5 measured pairs
20/20 output parity
4/4 aligned ranks per case
7 decode ordinals per request set
collective counts and CUDA durations present
summary regenerates byte-identically
```

- [ ] **Step 6: Verify Nsight evidence**

Require two representative replay reports or an explicit tool/report
unavailability receipt. Keep Nsight replay outside primary medians.

- [ ] **Step 7: Measure profile overhead**

Run one separately labeled unprofiled/structured-profile smoke pair with the
same case configuration and report the ratio. Do not mix it into primary
statistics.

- [ ] **Step 8: Verify cleanup**

Require:

```text
classification: CLEAN
remaining attempt-scoped PIDs: []
matched attempt-scoped GPU PIDs: []
remaining profiler children: []
```

- [ ] **Step 9: Update handoff**

Append exact commands, source and summary hashes, guard state, structured
metrics, Nsight evidence, overhead, cleanup, classification, limitations, and
next opportunities to `AGENT_HANDOFF_STATE.md`.

### Completion Audit

- [ ] Objective maps to concrete evidence for first decode step.
- [ ] Objective maps to concrete evidence for steady-state token latency.
- [ ] Objective maps to per-rank TP collective count and CUDA duration.
- [ ] Objective maps to CUDA kernel and NCCL evidence from Nsight or explicit
  profiler unavailability.
- [ ] Objective maps to a correctly labeled non-CUDA/synchronization upper
  bound.
- [ ] No new restore subdivision was added.
- [ ] Canonical schemas and old artifacts are unchanged.
- [ ] All focused tests and static checks pass.
- [ ] Real run meets guard, parity, rank alignment, reproducibility, and
  cleanup gates.
- [ ] `AGENT_HANDOFF_STATE.md` records what is proved and not proved.
