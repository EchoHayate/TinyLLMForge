# Generic Speculative Runtime Performance Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Execute inline with test-driven development. Follow every RED/GREEN step in order. Do not commit, stage, switch branches, stash, reset, push, clean, or modify `/Users/bytedance/dev/TinyLLMForge`.

## 2026-08-15 Evidence Reconciliation

This plan is reconciled against current source, fresh dependency-light tests,
the retained `20260812T085852Z` artifact, its historical local/remote verifier
receipts, and the durable audit/handoff.

```text
GENERIC_PERFORMANCE_PLAN_TOTAL_STEPS=35
GENERIC_PERFORMANCE_PLAN_CHECKED=29
GENERIC_PERFORMANCE_PLAN_INTENTIONALLY_OPEN=6
GENERIC_PERFORMANCE_FRESH_TESTS=107_PASSED
GENERIC_PERFORMANCE_SOURCE_PYCOMPILE=PASS
GENERIC_PERFORMANCE_RUNNER_BASH_SYNTAX=PASS
```

The six open steps are historical RED observations without retained
transcripts. No completed implementation, execution, verification, or
documentation step is left unchecked.

The retained artifact remains bound to its historical source:

```text
artifact SHA-256:
  d987b288176beec3ab841a7f640bb1d68cabf86445cadad50aec6337bcc4fb9f

historical local receipt:  PASS / NOT_PROMOTABLE
historical remote receipt: PASS / NOT_PROMOTABLE

artifact llm_engine.py SHA-256:
  baf26ee14d3cfe1dbeb0d897e8d4572460c1d9376ac92bbbff7fac718a7e5e12

current llm_engine.py SHA-256:
  2ffceaccfb1ff9e0cd2aa6506a1e6cdda588e71bc17368f2411974918543096b

fresh current-source verification:
  FAIL_CLOSED_SOURCE_HASH_MISMATCH

GENERIC_PERFORMANCE_CURRENT_SOURCE_VERIFICATION=FAIL_CLOSED_SOURCE_DRIFT
```

This source drift does not invalidate the historical receipts, but it means
the current checkout cannot claim a fresh verifier PASS for that artifact.
All nine artifact-bound source hashes were subsequently recovered exactly
into `/tmp/speculative-runtime-performance-frozen-2026-08-15`: five from the
current checkout and four from the content-addressed TRAE contribution
snapshot blob store. The recovered verifier exited zero against the unchanged
artifact and wrote
`/tmp/speculative-runtime-performance-frozen-verify.json`:

```text
GENERIC_PERFORMANCE_FROZEN_SOURCE_CLOSURE=RECOVERED_9_OF_9
GENERIC_PERFORMANCE_FROZEN_VERIFICATION=PASS_NOT_PROMOTABLE
```

This proves local reconstructability of the embedded nine-file closure, not
current-source equivalence. The artifact still has no retained standalone
deterministic source archive or checkpoint-bound model manifest, so it
remains legacy, scope-limited evidence.

```text
GENERIC_PERFORMANCE_HISTORICAL_AUTHORITY=PASS_NOT_PROMOTABLE
GENERIC_PERFORMANCE_CURRENT_SOURCE_VERIFICATION=FAIL_CLOSED_SOURCE_DRIFT
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

**Goal:** Build and run a controlled TP1 4K batch-1/batch-4 baseline-versus-real-ngram performance gate with exact parity, repeated TTFT/TPOT/throughput measurements, rank-wise peak GPU memory, and real MVP-0 movement deltas.

**Architecture:** Add a generic acknowledged peak-memory reset API. Run one isolated worker process per `(policy, batch_size)` cell, with one engine, one warmup, one parity run, and five measured runs. Assemble and independently verify one schema-v1 artifact without reusing the legacy rematerialization profiler.

**Tech Stack:** Python 3, PyTorch/CUDA, TinyLLMForge `LLMEngine`, `EngineSpeculativeRuntime`, `NGramDraftAdapter`, `KVOffloadMVP0`, JSON, SHA-256, pytest, Bash, SSH ControlMaster.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not modify or use `tools/profile_ngram_commit.py` as performance evidence.
- Keep runtime, Scheduler, allocator, verifier, and generic APIs model-name-free and proposal-source-free.
- Use `EngineSpeculativeRuntime(NGramDraftAdapter)` through `LLMEngine.activate_speculative_runtime()`.
- Use only before/after deltas from real `KVOffloadMVP0` summaries; never synthesize movement.
- Keep accepted KV in place and rejected suffix rollback semantics unchanged.
- Use TP1, Qwen3-0.6B, exactly 4096 prompt tokens, 64 output tokens, batch sizes 1 and 4, greedy sampling, one warmup, one parity run, and five measured runs.
- Use `kv_offload_mvp0=True`, full-attention-compatible prefill/decode,
  `kv_offload_gpu_blocks=68`, and one fixed post-first-token clean
  writeback/eviction sequence to produce positive real H2D evidence for every
  policy/batch campaign. Speculative verification must not enable blockwise
  prefill or decode.
- Keep classification `NOT_PROMOTABLE`; do not claim generalized end-to-end optimization.
- Use `sitian@10.232.195.203`, GPU 0, and `/tmp/ssh-sitian-10.232.195.203`.
- Do not commit, stage, switch branches, stash, reset, push, or run `git clean`.

---

### Task 1: Add Rank-Aware Peak-Memory Reset

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/test_engine_speculative_runtime.py`
- Modify: `tools/test_kv_offload_generation_metadata.py`

**Interfaces:**
- Produces: `ModelRunner.reset_peak_memory_stats() -> dict`.
- Produces: `LLMEngine.reset_peak_memory_stats(*, timeout_s: float) -> tuple[dict, ...]`.
- Preserves: existing `memory_snapshot()` schema.

- [x] **Step 1: Write the failing engine acknowledgement tests**

Append AST-loaded tests to `tools/test_engine_speculative_runtime.py`:

```python
def test_reset_peak_memory_stats_returns_rank_ordered_rows():
    method = _load_engine_method("reset_peak_memory_stats")
    engine = SimpleNamespace(
        model_runner=SimpleNamespace(world_size=2),
        call_model_runner_acknowledged=lambda name, timeout_s: (
            {
                "cuda_allocated_bytes": 10,
                "cuda_peak_allocated_bytes": 10,
            },
            (
                SimpleNamespace(
                    rank=1,
                    result={
                        "cuda_allocated_bytes": 20,
                        "cuda_peak_allocated_bytes": 20,
                    },
                ),
            ),
        ),
    )

    rows = method(engine, timeout_s=3.0)

    assert rows == (
        {
            "cuda_allocated_bytes": 10,
            "cuda_peak_allocated_bytes": 10,
            "rank": 0,
        },
        {
            "cuda_allocated_bytes": 20,
            "cuda_peak_allocated_bytes": 20,
            "rank": 1,
        },
    )
```

Add rejection tests for duplicate, missing, non-dict, and inner-rank-mismatch
acknowledgements.

- [ ] **Step 2: Run the focused engine RED**

Run:

```bash
python3 -m pytest \
  tools/test_engine_speculative_runtime.py \
  -k reset_peak_memory_stats \
  -q
```

Expected: FAIL because `LLMEngine.reset_peak_memory_stats` is absent.

- [x] **Step 3: Write the failing ModelRunner behavior test**

Use the dependency-light AST loader pattern already present in
`tools/test_kv_offload_generation_metadata.py`:

```python
def test_model_runner_reset_peak_memory_stats_synchronizes_and_snapshots():
    calls = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            synchronize=lambda: calls.append("synchronize"),
            reset_peak_memory_stats=lambda: calls.append("reset"),
        ),
    )
    method = _load_model_runner_method(
        "reset_peak_memory_stats",
        {"torch": fake_torch},
    )
    runner = SimpleNamespace(
        memory_snapshot=lambda: {
            "cuda_allocated_bytes": 7,
            "cuda_peak_allocated_bytes": 7,
        },
    )

    result = method(runner)

    assert calls == ["synchronize", "reset"]
    assert result["cuda_peak_allocated_bytes"] == 7
```

- [ ] **Step 4: Run the ModelRunner RED**

Run:

```bash
python3 -m pytest \
  tools/test_kv_offload_generation_metadata.py \
  -k reset_peak_memory_stats \
  -q
```

Expected: FAIL because `ModelRunner.reset_peak_memory_stats` is absent.

- [x] **Step 5: Implement the minimal reset methods**

Add to `ModelRunner` next to `memory_snapshot()`:

```python
def reset_peak_memory_stats(self):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return self.memory_snapshot()
```

Add to `LLMEngine` next to `memory_snapshots()`:

```python
def reset_peak_memory_stats(self, *, timeout_s):
    local_result, worker_acks = (
        self.call_model_runner_acknowledged(
            "reset_peak_memory_stats",
            timeout_s=timeout_s,
        )
    )
    ranked = [(0, local_result)]
    ranked.extend(
        (ack.rank, ack.result)
        for ack in worker_acks
    )
    rows = {}
    for outer_rank, row in ranked:
        if not isinstance(row, dict) or outer_rank in rows:
            raise ValueError("peak reset rank mismatch")
        inner_rank = row.get("rank", outer_rank)
        if inner_rank != outer_rank:
            raise ValueError("peak reset rank mismatch")
        normalized = dict(row)
        normalized["rank"] = outer_rank
        rows[outer_rank] = normalized
    expected = tuple(range(self.model_runner.world_size))
    if tuple(sorted(rows)) != expected:
        raise ValueError("peak reset rank inventory mismatch")
    return tuple(rows[rank] for rank in expected)
```

- [x] **Step 6: Run GREEN and nearby regression**

Run:

```bash
python3 -m pytest \
  tools/test_engine_speculative_runtime.py \
  tools/test_kv_offload_generation_metadata.py \
  -q
```

Expected: both suites PASS.

---

### Task 2: Build Pure Metric, Prompt, and Counter Helpers

**Files:**
- Create: `tools/speculative_runtime_performance_gate.py`
- Create: `tools/test_speculative_runtime_performance_gate.py`

**Interfaces:**
- Produces: `build_prompt_token_batches(tokenizer, *, batch_size, prompt_tokens)`.
- Produces: `subtract_counter_summaries(before, after, *, keys)`.
- Produces: `summarize_step_observations(observations)`.
- Produces: `build_run_metrics(...)`.
- Produces: `aggregate_measurements(runs)`.
- Produces: `classify_batch_direction(baseline, candidate)`.

- [x] **Step 1: Write failing prompt and counter tests**

Create dependency-light tests:

```python
def test_prompt_builder_returns_exact_4096_token_rows():
    tokenizer = FakeTokenizer(
        {
            "alpha": [11, 12, 13, 14],
            "beta": [21, 22, 23, 24],
            "gamma": [31, 32, 33, 34],
            "delta": [41, 42, 43, 44],
        }
    )

    rows = gate.build_prompt_token_batches(
        tokenizer,
        batch_size=4,
        prompt_tokens=4096,
    )

    assert len(rows) == 4
    assert all(len(row["token_ids"]) == 4096 for row in rows)
    assert len({row["sha256"] for row in rows}) == 4


def test_counter_delta_uses_only_monotonic_real_summaries():
    result = gate.subtract_counter_summaries(
        {"h2d_copies": 2, "h2d_bytes": 1024},
        {"h2d_copies": 5, "h2d_bytes": 4096},
        keys=("h2d_copies", "h2d_bytes"),
    )

    assert result == {
        "h2d_copies": 3,
        "h2d_bytes": 3072,
    }
```

Reject missing keys, bools, floats, and decreasing counters.

- [x] **Step 2: Write failing timing and aggregation tests**

Use a two-request, three-step synthetic observation:

```python
def test_run_metrics_compute_synchronized_ttft_tpot_and_throughput():
    metrics = gate.build_run_metrics(
        request_start_ns=1_000_000_000,
        request_finish_ns=3_000_000_000,
        token_events={
            0: [(1_500_000_000, 1), (3_000_000_000, 63)],
            1: [(2_000_000_000, 4), (3_000_000_000, 60)],
        },
        finished_at_ns={0: 3_000_000_000, 1: 3_000_000_000},
        expected_output_tokens=64,
    )

    assert metrics["per_request"][0]["ttft_s"] == 0.5
    assert metrics["per_request"][0]["tpot_s"] == pytest.approx(
        1.5 / 63
    )
    assert metrics["batch_token_throughput_tps"] == 64.0
    assert metrics["request_throughput_rps"] == 1.0
```

Also assert:

```python
def test_direction_requires_tpot_and_throughput_to_agree():
    assert gate.classify_batch_direction(
        {"tpot_s": {"median": 0.020},
         "batch_token_throughput_tps": {"median": 50.0}},
        {"tpot_s": {"median": 0.015},
         "batch_token_throughput_tps": {"median": 60.0}},
    ) == "IMPROVED"
```

Cover `REGRESSED` and `MIXED`.

- [ ] **Step 3: Run the pure-helper RED**

Run:

```bash
python3 -m pytest \
  tools/test_speculative_runtime_performance_gate.py \
  -k 'prompt or counter or metrics or direction or aggregate' \
  -q
```

Expected: FAIL because the gate module and helpers are absent.

- [x] **Step 4: Implement minimal pure helpers**

Define fixed constants:

```python
SCHEMA_VERSION = 1
CLASSIFICATION = "NOT_PROMOTABLE"
POLICIES = ("baseline", "ngram")
BATCH_SIZES = (1, 4)
PROMPT_TOKENS = 4096
MAX_OUTPUT_TOKENS = 64
WARMUP_RUNS = 1
PARITY_RUNS = 1
MEASURED_RUNS = 5
NGRAM_SIZE = 3
MAX_PROPOSAL_TOKENS = 4
```

Implement prompt rows as mappings containing `seed`, `token_ids`, and the
SHA-256 of a canonical JSON encoding of the token IDs. Implement aggregation
with `statistics.median`, `min`, `max`, and `statistics.pstdev`. Reject empty
or malformed input rather than returning partial metrics.

- [x] **Step 5: Run pure-helper GREEN**

Run:

```bash
python3 -m pytest \
  tools/test_speculative_runtime_performance_gate.py \
  -k 'prompt or counter or metrics or direction or aggregate' \
  -q
```

Expected: PASS.

---

### Task 3: Implement the Isolated Loaded-Engine Worker

**Files:**
- Create: `tools/speculative_runtime_performance_worker.py`
- Modify: `tools/test_speculative_runtime_performance_gate.py`

**Interfaces:**
- Consumes: fixed policy/batch/model arguments and pure helpers from the gate module.
- Produces: `run_request_batch(...) -> dict`.
- Produces: `run_policy_campaign(...) -> dict`.
- Produces: one worker JSON file for exactly one `(policy, batch_size)` cell.

- [x] **Step 1: Write a failing fake-engine request timing test**

Create a fake engine whose first synchronized step emits one token and whose
second emits the remaining 63:

```python
def test_worker_records_step_end_token_events_and_counter_deltas():
    engine = FakeEngine(
        step_rows=[
            ({}, {
                "new_completion_tokens_by_seq": {0: [10]},
                "finished_seq_ids": [],
            }),
            ({0: list(range(64))}, {
                "new_completion_tokens_by_seq": {0: list(range(63))},
                "finished_seq_ids": [0],
            }),
        ],
        summaries=[
            ({"h2d_copies": 2, "h2d_bytes": 1024},),
            ({"h2d_copies": 5, "h2d_bytes": 4096},),
        ],
    )

    result = worker.run_request_batch(
        engine=engine,
        prompt_rows=[PROMPT_ROW],
        sampling_params=object(),
        expected_output_tokens=64,
        synchronize=lambda: None,
        clock_ns=iter(
            (1_000_000_000, 1_500_000_000, 3_000_000_000)
        ).__next__,
    )

    assert result["movement"]["ranks"][0]["h2d_copies"] == 3
    assert result["timing"]["per_request"][0]["ttft_s"] == 0.5
    assert result["outputs"][0] == list(range(64))
```

The fake engine must also assert call order:

```text
idle -> clear prefix cache -> synchronize -> summary before
-> reset peaks -> synchronize -> add requests -> timed steps
-> synchronize -> summary after -> memory snapshot
```

- [x] **Step 2: Write failing campaign lifecycle tests**

Inject `engine_factory`, `sampling_params_type`, `runtime_type`, and
`adapter_type`:

```python
def test_candidate_campaign_installs_real_adapter_once_and_runs_1_1_5():
    result = worker.run_policy_campaign(
        model_path="/model",
        policy="ngram",
        batch_size=4,
        engine_factory=fake_factory,
        sampling_params_type=FakeSamplingParams,
        runtime_type=FakeRuntime,
        adapter_type=FakeNGramAdapter,
        synchronize=lambda: None,
        clock_ns=fake_clock,
    )

    assert fake_engine.activate_calls == 1
    assert len(result["warmup_runs"]) == 1
    assert len(result["parity_runs"]) == 1
    assert len(result["measured_runs"]) == 5
```

Assert baseline never installs a runtime, every prompt has 4096 tokens, every
output has 64 tokens, and the engine exits exactly once even on failure.

- [ ] **Step 3: Run worker RED**

Run:

```bash
python3 -m pytest \
  tools/test_speculative_runtime_performance_gate.py \
  -k 'worker or campaign' \
  -q
```

Expected: FAIL because the worker module is absent.

- [x] **Step 4: Implement `run_request_batch`**

The implementation must:

```python
engine.clear_reusable_prefix_cache()
before = engine.kv_offload_summaries(timeout_s=60.0)
reset_rows = engine.reset_peak_memory_stats(timeout_s=60.0)
synchronize()
started_ns = clock_ns()
```

After each `engine.step()`, call `synchronize()`, take one timestamp, copy
`last_step_observation`, and record all newly emitted tokens at that timestamp.
After completion, collect after summaries and memory snapshots, calculate
rank-wise movement deltas, and return raw observations plus derived metrics.

- [x] **Step 5: Implement `run_policy_campaign` and worker CLI**

Construct the engine with exactly:

```python
engine_factory(
    model_path,
    tensor_parallel_size=1,
    enforce_eager=True,
    max_model_len=4352,
    max_num_batched_tokens=16384,
    max_num_seqs=batch_size,
    max_num_prefill_tokens_per_step=1024,
    chunked_prefill_mixed_batch=False,
    kv_offload_mvp0=True,
    kv_offload_gpu_blocks=68,
    kv_offload_logical_blocks=128,
    kv_offload_blockwise_decode=False,
    kv_offload_blockwise_prefill=False,
    kv_offload_blockwise_blocks=1,
)
```

After the first completion token in each request batch, write back all active
sequence blocks, synchronize MVP-0, and evict the exact clean resident
`(logical_block, generation)` identities once. Baseline and n-gram cells must
use identical timing and configuration.

For `policy == "ngram"` install:

```python
engine.activate_speculative_runtime(
    EngineSpeculativeRuntime(
        NGramDraftAdapter(
            ngram_size=3,
            max_proposal_tokens=4,
        )
    )
)
```

Use `SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)`.
Write JSON atomically and always call `engine.exit()` in `finally`.

- [x] **Step 6: Run worker GREEN and compile**

Run:

```bash
python3 -m pytest \
  tools/test_speculative_runtime_performance_gate.py \
  -k 'worker or campaign' \
  -q
python3 -m py_compile \
  tools/speculative_runtime_performance_gate.py \
  tools/speculative_runtime_performance_worker.py
```

Expected: PASS.

---

### Task 4: Assemble and Independently Verify the Artifact

**Files:**
- Modify: `tools/speculative_runtime_performance_gate.py`
- Create: `tools/verify_speculative_runtime_performance_gate.py`
- Modify: `tools/test_speculative_runtime_performance_gate.py`

**Interfaces:**
- Produces: `validate_worker_result(worker_result) -> dict`.
- Produces: `build_performance_artifact(worker_results, environment, source_files) -> dict`.
- Produces: `validate_performance_artifact(artifact) -> dict`.
- Produces: `verify_performance_artifact(artifact_path, repo_root) -> dict`.
- Produces: parent CLI that launches four worker subprocesses.

- [x] **Step 1: Write failing artifact validity tests**

Build four minimal worker fixtures and assert:

```python
def test_artifact_requires_four_cells_exact_parity_and_real_paths():
    artifact = gate.build_performance_artifact(
        worker_results=WORKER_FIXTURES,
        environment=ENVIRONMENT,
        source_files={"tinyvllm/engine/llm_engine.py": "a" * 64},
    )

    result = gate.validate_performance_artifact(artifact)

    assert artifact["schema_version"] == 1
    assert artifact["classification"] == "NOT_PROMOTABLE"
    assert result["status"] == "PASS"
    assert set(result["batch_directions"]) == {"1", "4"}
```

Parameterized mutations must reject:

```text
missing policy/batch cell
warmup/parity/measured counts other than 1/1/5
prompt token count other than 4096
output token count other than 64
baseline/candidate output divergence
candidate proposed_tokens == 0
candidate accepted_draft_tokens == 0
missing first-target or tail callbacks
aggregate h2d_copies == 0 or h2d_bytes == 0
negative movement delta
missing peak-reset or final rank row
derived median/direction mismatch
classification other than NOT_PROMOTABLE
```

- [x] **Step 2: Write failing source-hash verifier tests**

Assert the verifier rejects a changed source file, a missing source file,
malformed SHA-256, and any artifact whose recomputed aggregate differs from
the stored aggregate.

- [ ] **Step 3: Run artifact/verifier RED**

Run:

```bash
python3 -m pytest \
  tools/test_speculative_runtime_performance_gate.py \
  -k 'artifact or verifier or source_hash' \
  -q
```

Expected: FAIL because artifact assembly and verifier are absent.

- [x] **Step 4: Implement worker validation and aggregation**

Validate every raw run before deriving aggregates. Reduce per-request TTFT,
TPOT, and completion latency to one median per run, then aggregate the five
run values. Recompute movement totals from raw rank deltas and recompute
acceptance from raw observations.

Store direction only from:

```python
def classify_batch_direction(baseline, candidate):
    baseline_tpot = baseline["tpot_s"]["median"]
    candidate_tpot = candidate["tpot_s"]["median"]
    baseline_throughput = (
        baseline["batch_token_throughput_tps"]["median"]
    )
    candidate_throughput = (
        candidate["batch_token_throughput_tps"]["median"]
    )
    if (
        candidate_tpot < baseline_tpot
        and candidate_throughput > baseline_throughput
    ):
        return "IMPROVED"
    if (
        candidate_tpot > baseline_tpot
        and candidate_throughput < baseline_throughput
    ):
        return "REGRESSED"
    return "MIXED"
```

- [x] **Step 5: Implement parent subprocess orchestration**

For each pair in:

```python
for policy in ("baseline", "ngram"):
    for batch_size in (1, 4):
        ...
```

launch:

```python
worker_output = (
    output_dir
    / f"worker-{policy}-b{batch_size}.json"
)
command = [
    sys.executable,
    "tools/speculative_runtime_performance_worker.py",
    "--model",
    model_path,
    "--policy",
    policy,
    "--batch-size",
    str(batch_size),
    "--out",
    str(worker_output),
]
```

Capture stdout/stderr per cell. Preserve partial worker JSON on failure and
write one diagnostic FAIL artifact when possible.

- [x] **Step 6: Implement independent verification**

The verifier must:

1. load JSON without importing worker/CUDA modules;
2. call the gate's pure artifact validator;
3. recompute every listed source SHA-256 relative to `repo_root`;
4. emit a JSON verification receipt containing `status`, artifact path,
   artifact SHA-256, schema version, classification, and direction.

- [x] **Step 7: Run artifact/verifier GREEN**

Run:

```bash
python3 -m pytest \
  tools/test_speculative_runtime_performance_gate.py \
  -q
python3 -m py_compile \
  tools/speculative_runtime_performance_gate.py \
  tools/speculative_runtime_performance_worker.py \
  tools/verify_speculative_runtime_performance_gate.py
```

Expected: all performance-gate tests PASS.

---

### Task 5: Add the Fixed Remote Runner

**Files:**
- Create: `tools/run_speculative_runtime_performance_gate_remote.sh`
- Modify: `tools/test_speculative_runtime_performance_gate.py`

**Interfaces:**
- Produces: one tagged local directory with `result.json`, four worker JSON
  files, four worker logs, `remote.log`, `verify.remote.json`, and
  `verify.json`.

- [x] **Step 1: Write failing remote-runner source tests**

Assert the script contains and uses:

```text
sitian@10.232.195.203
/tmp/ssh-sitian-10.232.195.203
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python
/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B
/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge
CUDA_VISIBLE_DEVICES=0
```

Also require `tinyvllm/`, all four performance tools, and the performance test
to be synchronized; require a fixed-venv remote `py_compile` preflight, remote
verification, unconditional artifact download, and local verification. The
remote model environment does not provide pytest, so local pytest is the test
authority.

- [ ] **Step 2: Run remote-runner RED**

Run:

```bash
python3 -m pytest \
  tools/test_speculative_runtime_performance_gate.py \
  -k remote_runner \
  -q
```

Expected: FAIL because the script is absent.

- [x] **Step 3: Implement the runner**

Follow the established boundary-runner structure:

```bash
set -euo pipefail
REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
CONTROL_SOCKET="${CONTROL_SOCKET:-/tmp/ssh-sitian-10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B}"
REMOTE_REPO="${REMOTE_REPO:-/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge}"
GPU_ID="${GPU_ID:-0}"
```

Run fixed-venv `py_compile` first, then the parent gate. Use `set +e` only
around the remote campaign, always `rsync` the remote output directory back,
and propagate the remote status after printing `remote.log`.

- [x] **Step 4: Run remote-runner GREEN and shell validation**

Run:

```bash
python3 -m pytest \
  tools/test_speculative_runtime_performance_gate.py \
  -k remote_runner \
  -q
bash -n tools/run_speculative_runtime_performance_gate_remote.sh
```

Expected: PASS.

---

### Task 6: Run the Real TP1 4K Campaign

**Files:**
- Produce: `artifacts/speculative_runtime_performance/${RUN_TAG}/result.json`
- Produce: `artifacts/speculative_runtime_performance/${RUN_TAG}/verify.remote.json`
- Produce: `artifacts/speculative_runtime_performance/${RUN_TAG}/verify.json`

**Interfaces:**
- Consumes: the fixed remote runner and current source tree.
- Produces: the first authoritative performance-direction artifact.

- [x] **Step 1: Run all local dependency-light tests**

Run:

```bash
python3 -m pytest \
  tools/test_engine_speculative_runtime.py \
  tools/test_kv_offload_generation_metadata.py \
  tools/test_speculative_runtime_performance_gate.py \
  -q
```

Expected: PASS.

- [x] **Step 2: Run the remote campaign**

Run:

```bash
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
LOCAL_OUT="artifacts/speculative_runtime_performance/${RUN_TAG}"
RUN_TAG="${RUN_TAG}" LOCAL_OUT="${LOCAL_OUT}" \
  bash tools/run_speculative_runtime_performance_gate_remote.sh
```

Do not interpret partial logs as a pass. Wait for all four worker cells and
both verification receipts.

- [x] **Step 3: Check authoritative gates**

Require:

```text
status=PASS
classification=NOT_PROMOTABLE
four policy/batch cells present
five measured runs per cell
exact baseline/candidate token parity
candidate proposed_tokens > 0
candidate accepted_draft_tokens > 0
candidate first_target_callbacks > 0
candidate tail_callbacks > 0
positive aggregate h2d_copies and h2d_bytes for every cell
remote verifier PASS
local verifier PASS
```

Record the actual per-batch direction. Do not convert `MIXED` or `NEGATIVE`
into an optimization claim.

- [x] **Step 4: Run loaded CUDA KV regression**

Synchronize the current source and run the existing direct suite with remote
CUDA Python:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge \
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python \
  tools/test_kv_offload.py
```

Expected: `kv offload tests passed`.

---

### Task 7: Record the Result and Preserve Claim Boundaries

**Files:**
- Modify: `docs/superpowers/audits/2026-08-12-generic-inference-optimization-goal-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: final artifact and both verification receipts.
- Produces: authoritative continuation state for the next matrix expansion or
  optimization iteration.

- [x] **Step 1: Update the audit**

Record:

```text
artifact path and SHA-256
source commit and dirty-tree limitation
exact fixed matrix and configuration
all five raw-run aggregate summaries
batch-1 and batch-4 TTFT, TPOT, token throughput, request throughput
peak allocated/reserved GPU memory
real H2D/D2H bytes and copies
proposal and acceptance counts/rates
per-batch and campaign direction
what the result proves and does not prove
```

- [x] **Step 2: Update handoff**

If direction is `POSITIVE`, the next task is the 16K/32K expansion design.
If `MIXED` or `NEGATIVE`, the next task is a profile-driven optimization of
the measured bottleneck while preserving exact parity and transactional KV
semantics.

- [x] **Step 3: Run final verification**

Run:

```bash
python3 tools/verify_speculative_runtime_performance_gate.py \
  "artifacts/speculative_runtime_performance/${RUN_TAG}/result.json" \
  . \
  --output \
  "artifacts/speculative_runtime_performance/${RUN_TAG}/verify.json"

python3 -m pytest \
  tools/test_engine_speculative_runtime.py \
  tools/test_kv_offload_generation_metadata.py \
  tools/test_speculative_runtime_performance_gate.py \
  tools/test_speculative_residency_boundary_gate.py \
  tools/test_speculative_tp1_parity_gate.py \
  -q

python3 -m py_compile \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/speculative_runtime_performance_gate.py \
  tools/speculative_runtime_performance_worker.py \
  tools/verify_speculative_runtime_performance_gate.py

bash -n tools/run_speculative_runtime_performance_gate_remote.sh
git diff --check
git diff --cached --quiet
```

Expected: all tests and static checks PASS, both verification receipts remain
PASS, and the staged diff remains empty.

## Execution Record

The completed authoritative campaign used tag `20260812T085852Z`:

```text
artifact:
  artifacts/speculative_runtime_performance/20260812T085852Z/result.json
artifact SHA-256:
  d987b288176beec3ab841a7f640bb1d68cabf86445cadad50aec6337bcc4fb9f
remote verifier:
  PASS
local verifier:
  PASS
batch 1 direction:
  IMPROVED
batch 4 direction:
  IMPROVED
campaign direction:
  POSITIVE
classification:
  NOT_PROMOTABLE
```

The initially planned blockwise configuration was rejected by the generic
spec-verifier compatibility guard. The executed gate therefore used
`kv_offload_gpu_blocks=68`, disabled blockwise prefill/decode, and forced the
same real clean-writeback/eviction/H2D restore sequence in baseline and
candidate workers. The remote environment also lacked pytest; the final
runner used remote `py_compile` preflight and retained local pytest as test
authority.
