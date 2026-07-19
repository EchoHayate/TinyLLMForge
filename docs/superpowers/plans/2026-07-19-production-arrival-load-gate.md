# Production Arrival-Load Continuous-Batching Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and remotely execute a source-auditable arrival-load gate that
compares TinyLLMForge's existing continuous-batching policies under frozen
request arrivals, independently verifies correctness and performance, and
publishes only a verifier-confirmed `GO`, `PROMISING_NOT_PROVEN`, `NO_GO`, or
`INCOMPLETE`.

**Architecture:** Keep workload construction, online model driving, offline
aggregation, and independent verification in separate files. Add only
observation-only scheduler/engine hooks, then run one isolated model process
per policy/scenario/repetition with unique ports; the remote wrapper stages an
immutable source snapshot and the local verifier rebuilds every metric and
classification from append-only raw evidence rather than trusting the
harness summary.

**Tech Stack:** Python 3 standard library, existing TinyLLMForge
`LLMEngine`/scheduler/model runner, PyTorch CUDA memory APIs, dependency-light
Python test scripts, Bash, Git, SSH, Qwen3-0.6B on the remote CUDA host.

## Global Constraints

- The normative design is
  `docs/superpowers/specs/2026-07-18-production-arrival-load-gate-design.md`.
- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`; do not
  modify `/Users/bytedance/dev/TinyLLMForge`.
- Execute inline in the current session; do not dispatch subagents.
- GPU/model work runs only on `sitian@10.232.195.203` with Python
  `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python` and model
  `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`.
- Every model process receives a unique dynamic `TINYVLLM_DIST_PORT` and
  `MASTER_PORT` pair.
- Use `max_num_batched_tokens=16384`, `max_num_seqs=512`,
  `max_num_prefill_tokens_per_step=128`, and `enforce_eager=False` for every
  policy.
- P0 leaves the four policy fields at their repository defaults. P1 resolves
  them explicitly to decode-first. P2 sets
  `chunked_prefill_decode_first=False` and
  `chunked_prefill_max_consecutive_chunks=2`. P3 sets
  `chunked_prefill_decode_first=False` and
  `chunked_prefill_mixed_batch=True`.
- Policy identity is the canonical JSON SHA-256 of the fully resolved
  scheduler-affecting configuration. P1 aliases P0 when identities match and
  cannot add repetitions or vote independently.
- Use greedy fixed-length decoding (`temperature=0.0`, `ignore_eos=True`) so
  every candidate can be compared token-for-token with P0.
- All timestamps come from `time.monotonic_ns()` in the model-driving process.
  Primary latency uses scheduled arrival, never actual injection time.
- The inline driver injects every due request through
  `LLMEngine.add_request()` and advances runnable work through
  `LLMEngine.step()`.
- Each non-duplicate canonical policy/scenario has three measured
  repetitions. Report median and worst repetition; a best repetition never
  creates `GO`.
- The steady rates are exactly `0.6 * lambda_ref`, `0.9 * lambda_ref`, and
  `1.2 * lambda_ref` for moderate, near-saturation, and overload.
- The preregistered benefit paths are exactly: throughput `>=5%` with p95
  TTFT/ITL regressions `<=5%`; or p95 TTFT/ITL improvement `>=10%` with
  throughput regression `<=2%` and the other p95 regression `<=5%`; or peak
  KV/CUDA-reserved improvement `>=5%` with throughput/p95 TTFT/p95 ITL
  regressions `<=2%`.
- Every canonical scenario must also keep p99 TTFT, p99 ITL, p99 E2E,
  `maximum_decode_gap`, and every service-time bucket's p95 E2E regression within
  `10%`.
- Any missing/truncated/non-finite artifact, ambiguous calibration, reused
  port pair, source/environment/workload hash mismatch, fewer than three
  complete repetitions, invalid alias counting, or independent-verifier
  failure is `INCOMPLETE`.
- Peak `CUDA reserved` memory and peak KV bytes are recorded from raw memory
  observations; neither may be inferred from a summary-only field.
- Any output, request-set, finish-state, timestamp, lifecycle, starvation,
  dropped, rejected, truncated, or unfinished mismatch is correctness
  `NO_GO` and cannot be overridden by performance.
- Observation code must not change queue order, scheduling decisions, token
  generation, block allocation, synchronization, or policy defaults.
- Do not kill unrelated GPU processes, clear shared `/tmp`, modify a remote
  checkout, or reuse another run's source/artifact directory.
- Resume may reuse only hash-valid complete repetitions. Replacing a failed
  repetition replaces all of its stale raw records and logs. Calibration
  changes require a new run tag.
- Preserve all existing untracked experiment artifacts. Stage files
  selectively; never use `git add -A`.
- The prior K1/SAM and speculation-router canonical results remain `NO_GO`.
  Do not describe this gate as an improvement until its independent canonical
  verifier confirms `GO`.

## Fixed Workload Defaults

These values are implementation constants and are written into every
manifest before model results are visible:

```python
SCHEMA_VERSION = 1
COMMON_ENGINE_CONFIG = {
    "max_num_batched_tokens": 16384,
    "max_num_seqs": 512,
    "max_num_prefill_tokens_per_step": 128,
    "enforce_eager": False,
}
PROMPT_CLASS_TARGET_TOKENS = {
    "short": 64,
    "medium": 512,
    "long": 1536,
}
OUTPUT_CLASS_TOKENS = {"short": 16, "long": 64}
CALIBRATION_INITIAL_RATE_RPS = 0.5
CALIBRATION_MAX_DOUBLINGS = 8
CALIBRATION_BISECTION_STEPS = 3
CALIBRATION_REQUESTS_PER_RATE = 24
CALIBRATION_DRAIN_TIMEOUT_NS = 120_000_000_000
CANONICAL_WARMUP_REQUESTS = 8
CANONICAL_MEASURED_REQUESTS = 64
FAIRNESS_REQUESTS_PER_BUCKET = 20
CANONICAL_DRAIN_TIMEOUT_NS = 120_000_000_000
STARVATION_DEADLINE_NS = 5_000_000_000
MEASURED_REPETITIONS = 3
ARRIVAL_SEEDS = {
    "steady_moderate": 601,
    "near_saturation": 901,
    "overload": 1201,
    "burst": 1701,
    "long_prompt_pressure": 1901,
    "mixed_service_fairness": 2301,
}
```

Steady scenarios use seeded exponential inter-arrivals. Burst uses four
16-request bursts, each realized inside a fixed `250 ms` interval, followed
by a `2 s` recovery window. Standard scenarios use 64 measured requests;
mixed-service fairness uses 20 requests for each of the six fixed
`prompt_class x output_class` buckets. Long-prompt pressure uses 60% long,
25% medium, and 15% short prompts while keeping short/long outputs balanced.

Calibration doubles from `0.5 requests/s` until it finds the first unstable
point or exhausts eight doublings, then performs three deterministic
bisection rates between the highest stable and lowest unstable rates.
Backlog stability is computed over the final third of the offered-arrival
window by ordinary least squares; a rate is stable only when the slope is
`<= max(0.01 requests/s, 0.02 * offered_rate_rps)`, every request drains, all
metrics are finite, and outputs are exact. `lambda_ref` is the highest stable
rate whose completed-request throughput is at least 95% of the maximum stable
completed-request throughput observed.

## File Structure

- Create `tools/arrival_load_gate.py`: constants, prompt/workload manifests,
  policy resolution and identity, pure aggregation, classification, source
  snapshot, process orchestration, resume reconciliation, artifact
  finalization, report generation, and CLI.
- Create `tools/arrival_load_driver.py`: one-process inline arrival driver
  that initializes one policy, binds requests to sequence IDs, appends raw
  request/scheduler/memory records, and exits fail-closed.
- Create `tools/arrival_load_verify.py`: standalone verifier with its own JSON
  parsing, nearest-rank, metric reconstruction, alias checks, regression
  guards, classification, and report.
- Create `tools/test_arrival_load_gate.py`: dependency-light workload,
  aggregation, classification, source, process, resume, and tamper tests.
- Create `tools/test_arrival_load_driver.py`: fake-engine lifecycle,
  scheduled/actual arrival, sequence binding, multi-token step, timeout, and
  append-only stream tests.
- Create `tools/test_arrival_load_verify.py`: independent recomputation and
  deliberate aggregator/verifier disagreement tests.
- Modify `tinyvllm/engine/scheduler.py`: expose immutable queue/KV snapshots
  and record the actual branch selected by `schedule()`.
- Modify `tinyvllm/engine/llm_engine.py`: capture an observation-only
  `last_step_observation` around the existing step.
- Modify `tinyvllm/engine/model_runner.py`: expose rank-0 CUDA allocated,
  reserved, peak, and KV-byte metadata without synchronization or mutation.
- Modify `tools/test_chunked_prefill.py`: prove instrumentation preserves
  existing default, bounded-prefill, and mixed scheduling semantics.
- Create `tools/run_arrival_load_gate_remote.sh`: immutable staging, remote
  preflight, smoke/calibration/canonical launch, polling, chunked artifact
  recovery, download-only, verify-only, and local independent verification.
- Create `tools/test_run_arrival_load_gate_remote.py`: shell safety, dynamic
  ports, mode, source boundary, atomic exit-code, zero-byte, and recovery
  contract tests.
- Modify `README.md` and `AGENT_HANDOFF_STATE.md` only after verified remote
  evidence exists.
- Generate artifacts only under
  `experiments/arrival_load/$RUN_TAG/`; keep them untracked.

---

### Task 1: Deterministic Workload and Policy Contracts

**Files:**
- Create: `tools/test_arrival_load_gate.py`
- Create: `tools/arrival_load_gate.py`

**Interfaces:**
- Produces:
  `canonical_json_sha256(value: object) -> str`
- Produces:
  `nearest_rank(values: list[float], percentile: float) -> float`
- Produces:
  `build_prompt_bank(tokenizer, *, model_id: str) -> dict`
- Produces:
  `build_calibration_manifest(prompt_bank: dict) -> list[dict]`
- Produces:
  `build_canonical_workload(*, lambda_ref: float, prompt_bank: dict) -> list[dict]`
- Produces:
  `resolve_policy_config(policy_name: str, defaults: dict) -> dict`
- Produces:
  `policy_identity(resolved_config: dict) -> str`
- Produces:
  `deduplicate_policies(resolved: dict[str, dict]) -> dict`

- [ ] **Step 1: Write failing deterministic manifest tests**

Create `tools/test_arrival_load_gate.py` with direct imports through
`importlib.util`, a fake tokenizer whose `encode()` splits on spaces, and
tests asserting:

```python
def test_seeded_steady_and_burst_workloads_are_byte_stable():
    first = gate.build_canonical_workload(
        lambda_ref=4.0,
        prompt_bank=_prompt_bank(),
    )
    second = gate.build_canonical_workload(
        lambda_ref=4.0,
        prompt_bank=_prompt_bank(),
    )
    assert first == second
    assert gate.canonical_json_sha256(first) == (
        gate.canonical_json_sha256(second)
    )
    assert [row["request_id"] for row in first] == sorted(
        row["request_id"] for row in first
    )
    burst = [
        row for row in first if row["scenario"] == "burst"
    ]
    assert len(burst) == 64 + gate.CANONICAL_WARMUP_REQUESTS
    assert max(row["arrival_offset_ns"] for row in burst) > 6_000_000_000


def test_service_buckets_are_fixed_before_execution():
    rows = gate.build_canonical_workload(
        lambda_ref=4.0,
        prompt_bank=_prompt_bank(),
    )
    fairness = [
        row for row in rows
        if row["scenario"] == "mixed_service_fairness"
        and not row["warmup"]
    ]
    counts = Counter(row["service_time_bucket"] for row in fairness)
    assert set(counts) == {
        "short__short", "short__long",
        "medium__short", "medium__long",
        "long__short", "long__long",
    }
    assert set(counts.values()) == {gate.FAIRNESS_REQUESTS_PER_BUCKET}


def test_policy_identity_aliases_explicit_default_only():
    defaults = {
        "chunked_prefill_decode_first": True,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
    }
    resolved = {
        name: gate.resolve_policy_config(name, defaults)
        for name in ("P0", "P1", "P2", "P3")
    }
    aliases = gate.deduplicate_policies(resolved)
    assert aliases["canonical_policy_by_name"] == {
        "P0": "P0", "P1": "P0", "P2": "P2", "P3": "P3",
    }
    assert len(set(aliases["identity_by_name"].values())) == 3
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_arrival_load_gate.py
```

Expected: FAIL because `tools/arrival_load_gate.py` does not exist.

- [ ] **Step 3: Implement constants, canonical hashing, policies, and builders**

Create `tools/arrival_load_gate.py` with the exact constants from **Fixed
Workload Defaults**, canonical JSON encoding using sorted keys and compact
separators, and policy definitions:

```python
POLICY_OVERRIDES = {
    "P0": {},
    "P1": {
        "chunked_prefill_decode_first": True,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
    },
    "P2": {
        "chunked_prefill_decode_first": False,
        "chunked_prefill_max_consecutive_chunks": 2,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
    },
    "P3": {
        "chunked_prefill_decode_first": False,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": True,
        "chunked_prefill_mixed_min_prompt_tokens": 0,
    },
}


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_policy_config(policy_name: str, defaults: dict) -> dict:
    if policy_name not in POLICY_OVERRIDES:
        raise ValueError(f"unknown policy: {policy_name}")
    resolved = {
        **COMMON_ENGINE_CONFIG,
        **{
            key: defaults[key]
            for key in POLICY_OVERRIDES["P1"]
        },
        **POLICY_OVERRIDES[policy_name],
    }
    return resolved


def policy_identity(resolved_config: dict) -> str:
    return canonical_json_sha256(resolved_config)
```

Generate exponential arrivals with `random.Random(seed).expovariate(rate)`;
store realized integer nanosecond offsets. Build burst offsets explicitly,
assign immutable request IDs
`<scenario>-<warmup|measured>-<zero-padded-index>`, and sort by
`(scenario_order, arrival_offset_ns, request_id)`. Store prompt text, prompt
hash, measured token count, requested output count, classes, bucket, sampling
contract, seed, requested rate, and generator version in every row.

- [ ] **Step 4: Add boundary and malformed-policy tests**

Add tests for nearest-rank p50/p95/p99 on one, two, and twenty samples; prompt
bank hash drift; negative/non-finite `lambda_ref`; unknown policy; unexpected
P2/P3 identity collision; and exact moderate/near/overload rates
`0.6/0.9/1.2 * lambda_ref`.

- [ ] **Step 5: Run the workload contract tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_arrival_load_gate.py
```

Expected: all Task 1 tests print
`arrival load gate tests passed`.

- [ ] **Step 6: Commit the deterministic contracts**

```bash
git add tools/arrival_load_gate.py tools/test_arrival_load_gate.py
git commit -m "feat: add arrival load workload contracts"
```

---

### Task 2: Raw Lifecycle Aggregation and Preregistered Classification

**Files:**
- Modify: `tools/test_arrival_load_gate.py`
- Modify: `tools/arrival_load_gate.py`

**Interfaces:**
- Produces:
  `reconstruct_request_metrics(workload_rows, timeline_rows, scheduler_rows) -> list[dict]`
- Produces:
  `summarize_repetition(case: dict, request_metrics: list[dict], memory_rows: list[dict]) -> dict`
- Produces:
  `aggregate_case_repetitions(rows: list[dict]) -> dict`
- Produces:
  `classify_gate(run_manifest: dict, case_rows: list[dict]) -> dict`
- Produces:
  `render_report(run_manifest: dict, summary: dict) -> str`

- [ ] **Step 1: Add failing metric reconstruction tests**

Add synthetic raw rows covering scheduled versus actual arrival, queue delay,
multiple tokens from one step, one-token output without ITL, and memory peaks:

```python
def test_reconstructs_scheduled_arrival_metrics_and_shared_step_tokens():
    workload = [_workload_row("r0", output_tokens=3)]
    timeline = [{
        "request_id": "r0",
        "seq_id": 7,
        "scheduled_arrival_ns": 100,
        "actual_arrival_ns": 120,
        "first_scheduled_ns": 150,
        "first_token_ns": 200,
        "token_timestamps_ns": [200, 260, 260],
        "completion_ns": 260,
        "output_token_ids": [11, 12, 13],
        "finish_reason": "length",
        "error": None,
    }]
    metrics = gate.reconstruct_request_metrics(workload, timeline, [])
    assert metrics == [{
        **metrics[0],
        "injection_lag_ns": 20,
        "queue_delay_ns": 30,
        "ttft_ns": 100,
        "e2e_ns": 160,
        "itl_ns": [60, 0],
        "maximum_decode_gap_ns": 60,
    }]


def test_one_token_output_has_no_itl_sample():
    metrics = gate.reconstruct_request_metrics(
        [_workload_row("r0", output_tokens=1)],
        [_timeline_row("r0", [300])],
        [],
    )
    assert metrics[0]["itl_ns"] == []
    assert metrics[0]["maximum_decode_gap_ns"] is None
```

- [ ] **Step 2: Add failing classification-table tests**

Create table-driven fixtures for:

- throughput `+5%` exactly with TTFT/ITL `+5%` exactly -> `GO`;
- throughput `+4.999%` -> `PROMISING_NOT_PROVEN`;
- p95 TTFT `-10%` exactly with throughput `-2%` -> `GO`;
- peak reserved `-5%` exactly with all guarded metrics `+2%` -> `GO`;
- p99, decode gap, or one service bucket `+10.001%` -> `NO_GO`;
- favorable median but unfavorable worst repetition -> not `GO`;
- missing third repetition, non-finite sample, duplicate port, or improperly
  counted P1 alias -> `INCOMPLETE`;
- output mismatch, unfinished request, starvation, P0 repeat instability, or
  request-set mismatch -> `NO_GO`;
- correct complete neutral candidate -> `NO_GO`.

Assert the returned summary contains separate
`structural_failures`, `correctness_failures`, `candidate_results`, selected
benefit path, median rows, and worst-repetition rows.

- [ ] **Step 3: Run tests and verify missing aggregation functions**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_arrival_load_gate.py
```

Expected: FAIL on missing `reconstruct_request_metrics` or
`classify_gate`.

- [ ] **Step 4: Implement lifecycle validation and metrics**

Implement strict one-record-per-request validation, unique request/sequence
binding, monotonic timestamps, exact token timestamp count, finite values, and
fixed service buckets. Compute:

```python
injection_lag_ns = actual_arrival_ns - scheduled_arrival_ns
queue_delay_ns = first_scheduled_ns - actual_arrival_ns
ttft_ns = first_token_ns - scheduled_arrival_ns
e2e_ns = completion_ns - scheduled_arrival_ns
itl_ns = [
    current - previous
    for previous, current in zip(
        token_timestamps_ns,
        token_timestamps_ns[1:],
    )
]
```

Use nearest-rank over sorted finite samples. Request throughput and
output-token throughput use only measured requests and the frozen measured interval:
`max(completion_ns) - min(scheduled_arrival_ns)`. Report request/s and output
tokens/s, p50/p95/p99 request metrics, maximum injection/decode gap,
unfinished/dropped/truncated/starved counts, per-bucket metrics, Jain's index,
peak CUDA allocated/reserved, peak KV blocks, and peak KV bytes.

- [ ] **Step 5: Implement median/worst aggregation and classification**

Pair P0/candidate by scenario and repetition index. Define ratio as
`candidate / baseline`; lower is better for latency/memory and higher is
better for throughput. Worst repetition is the least favorable paired ratio
for the claimed path, while regression guards use the largest bad ratio.
Implement the four-state precedence:

```python
if structural_failures:
    classification = "INCOMPLETE"
elif correctness_failures:
    classification = "NO_GO"
elif any(candidate["classification"] == "GO" for candidate in candidates):
    classification = "GO"
elif any(
    candidate["classification"] == "PROMISING_NOT_PROVEN"
    for candidate in candidates
):
    classification = "PROMISING_NOT_PROVEN"
else:
    classification = "NO_GO"
```

Do not allow P1 to create a candidate when it aliases P0. An unexpected P2/P3
collision is structural `INCOMPLETE`.

- [ ] **Step 6: Run classification and existing scheduler tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_arrival_load_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
```

Expected: both scripts pass.

- [ ] **Step 7: Commit aggregation and classification**

```bash
git add tools/arrival_load_gate.py tools/test_arrival_load_gate.py
git commit -m "feat: classify arrival load evidence"
```

---

### Task 3: Standalone Independent Verifier

**Files:**
- Create: `tools/arrival_load_verify.py`
- Create: `tools/test_arrival_load_verify.py`
- Modify: `tools/test_arrival_load_gate.py`

**Interfaces:**
- Produces:
  `verify_run(run_dir: Path, *, write_output: bool = True) -> dict`
- Produces CLI:
  `python3 tools/arrival_load_verify.py --run-dir experiments/arrival_load/$RUN_TAG`
- Consumes only artifact files; does not import `arrival_load_gate`.
- Writes:
  `independent-verify/summary.json`,
  `independent-verify/report.md`,
  `independent-verify/verify.stdout`,
  `independent-verify/verify.stderr`,
  `independent-verify/verify.exitcode`.

- [ ] **Step 1: Write a failing verifier independence test**

Create a complete tiny synthetic artifact fixture, finalize it with the
harness, then assert:

```python
def test_verifier_does_not_import_harness_aggregation():
    source = VERIFY_PATH.read_text()
    assert "import arrival_load_gate" not in source
    assert "from arrival_load_gate" not in source


def test_verifier_recomputes_without_trusting_summary():
    run_dir = _complete_artifact()
    recorded = json.loads((run_dir / "summary.json").read_text())
    recorded["classification"] = "GO"
    (run_dir / "summary.json").write_text(json.dumps(recorded))
    _refresh_hash(run_dir, "summary.json")
    try:
        verifier.verify_run(run_dir)
    except ValueError as exc:
        assert "classification disagreement" in str(exc)
    else:
        raise AssertionError("tampered summary must be rejected")
```

Also monkeypatch the harness `nearest_rank()` to return a wrong value and
prove verifier output remains correct.

- [ ] **Step 2: Run tests and verify the missing verifier failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_arrival_load_verify.py
```

Expected: FAIL because `tools/arrival_load_verify.py` does not exist.

- [ ] **Step 3: Implement independent parsing and recomputation**

Implement local copies of canonical JSON hashing, nearest-rank,
request-metric reconstruction, per-repetition aggregation, policy alias
validation, benefit paths, guard checks, and final classification. Read and
validate every required file:

```python
REQUIRED_FILES = (
    "run_manifest.json",
    "calibration_manifest.jsonl",
    "calibration_rows.jsonl",
    "workload_manifest.jsonl",
    "request_timeline.jsonl",
    "scheduler_trace.jsonl",
    "memory_trace.jsonl",
    "case_rows.jsonl",
    "summary.json",
    "report.md",
    "source_evidence.json",
    "source.patch",
    "source_snapshot.tar.gz",
    "artifact_hashes.json",
)
```

Reject malformed/truncated JSONL, duplicate keys, absent final newlines,
unsafe archive members, impossible timestamp ordering, sample-count
differences, missing buckets, reused ports, source/environment/workload hash
differences, and any harness/verifier result disagreement.

- [ ] **Step 4: Add corruption and boundary tests**

Test each of: missing scheduler row, truncated final JSONL record, changed
output token, duplicated request binding, reused port pair, changed workload
offset, changed policy identity, missing service bucket, changed source file,
changed artifact hash, nonzero process exit, and exact threshold boundaries.

- [ ] **Step 5: Run all dependency-light evidence tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_arrival_load_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_arrival_load_verify.py
```

Expected: both pass and verifier tests print
`arrival load verifier tests passed`.

- [ ] **Step 6: Commit the independent verifier**

```bash
git add \
  tools/arrival_load_verify.py \
  tools/test_arrival_load_verify.py \
  tools/test_arrival_load_gate.py
git commit -m "feat: independently verify arrival load artifacts"
```

---

### Task 4: Observation-Only Scheduler and Engine Instrumentation

**Files:**
- Modify: `tinyvllm/engine/scheduler.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_chunked_prefill.py`

**Interfaces:**
- Produces:
  `Scheduler.observation_snapshot() -> dict`
- Produces:
  `Scheduler.last_policy_branch: str | None`
- Produces:
  `ModelRunner.memory_snapshot() -> dict`
- Produces:
  `LLMEngine.last_step_observation: dict | None`
- Preserves existing `LLMEngine.step() -> tuple[list[tuple[int, list[int]]], int]`.

- [ ] **Step 1: Add failing instrumentation semantics tests**

In `tools/test_chunked_prefill.py`, add tests that construct the existing fake
scheduler fixtures and assert branch names:

```python
def test_observation_reports_decode_first_without_changing_result():
    scheduler = Scheduler(make_config(
        max_num_prefill_tokens_per_step=4,
        chunked_prefill_decode_first=True,
    ))
    running = make_seq([1, 2], max_tokens=4)
    waiting = make_seq([3] * 9, max_tokens=4)
    _put_running(scheduler, running)
    scheduler.add(waiting)
    result = scheduler.schedule()
    assert [seq.seq_id for seq in result[0]] == [running.seq_id]
    assert scheduler.last_policy_branch == "decode_first"
    snapshot = scheduler.observation_snapshot()
    assert snapshot["waiting_seq_ids"] == [waiting.seq_id]
    assert snapshot["used_kv_blocks"] == len(
        scheduler.block_manager.used_block_ids
    )
```

Add equivalent tests for `bounded_prefill_yield`, `chunked_prefill`,
`mixed_prefill_decode`, `decode_fallback`, and legacy non-chunked prefill.
Compare complete pre/post queue and token outcomes with the prior expected
values so instrumentation cannot pass by merely naming a branch.

- [ ] **Step 2: Run the scheduler tests and verify missing observation APIs**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
```

Expected: FAIL on `last_policy_branch` or `observation_snapshot`.

- [ ] **Step 3: Add scheduler branch recording through one return helper**

Add:

```python
def _observed_result(self, result, branch: str):
    self.last_policy_branch = branch
    return result


def observation_snapshot(self) -> dict:
    manager = self.block_manager
    return {
        "waiting_seq_ids": [seq.seq_id for seq in self.waiting],
        "prefilling_seq_ids": [seq.seq_id for seq in self.prefilling],
        "running_seq_ids": [seq.seq_id for seq in self.running],
        "free_kv_blocks": len(manager.free_block_ids),
        "used_kv_blocks": len(manager.used_block_ids),
        "total_kv_blocks": len(manager.blocks),
        "kv_block_size_tokens": manager.block_size,
        "consecutive_prefill_chunks": self._consecutive_prefill_chunks,
    }
```

Initialize `last_policy_branch = None` and wrap every existing `schedule()`
return with an exact branch name. Do not change branch conditions, queue
operations, or result tuples.

- [ ] **Step 4: Add rank-0 CUDA/KV memory observation**

In `ModelRunner`, add:

```python
def memory_snapshot(self):
    allocated = int(torch.cuda.memory_allocated())
    reserved = int(torch.cuda.memory_reserved())
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    kv_bytes = int(self.kv_cache.numel() * self.kv_cache.element_size())
    if self.kv_scale is not None:
        kv_bytes += int(
            self.kv_scale.numel() * self.kv_scale.element_size()
        )
    if self.kv_zero is not None:
        kv_bytes += int(
            self.kv_zero.numel() * self.kv_zero.element_size()
        )
    return {
        "cuda_allocated_bytes": allocated,
        "cuda_reserved_bytes": reserved,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "kv_capacity_bytes": kv_bytes,
    }
```

Do not call `torch.cuda.synchronize()`, `empty_cache()`, or reset peak stats
inside this method.

- [ ] **Step 5: Capture step observations without changing the return value**

In `LLMEngine.step()`, capture scheduler snapshots immediately before
`schedule()` and after `postprocess()`, copy scheduled-sequence fields before
postprocess mutates them, compute completion-token deltas, and store:

```python
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
}
```

Keep the existing `outputs, num_tokens` return byte-for-byte compatible.

- [ ] **Step 6: Run scheduler, profiler, and compile regressions**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_profile_chunked_prefill.py
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
  python3 -m py_compile \
  tinyvllm/engine/scheduler.py \
  tinyvllm/engine/llm_engine.py \
  tinyvllm/engine/model_runner.py
```

Expected: all pass.

- [ ] **Step 7: Commit observation-only instrumentation**

```bash
git add \
  tinyvllm/engine/scheduler.py \
  tinyvllm/engine/llm_engine.py \
  tinyvllm/engine/model_runner.py \
  tools/test_chunked_prefill.py
git commit -m "feat: expose arrival load observations"
```

---

### Task 5: Single-Case Inline Arrival Driver

**Files:**
- Create: `tools/arrival_load_driver.py`
- Create: `tools/test_arrival_load_driver.py`

**Interfaces:**
- Produces:
  `run_case(*, case_spec: dict, workload_rows: list[dict], engine_factory, clock_ns, output_dir: Path) -> dict`
- Produces CLI requiring:
  `--case-spec`, `--workload-manifest`, `--model`, `--output-dir`.
- Writes one case-local:
  `request_timeline.jsonl`, `scheduler_trace.jsonl`, `memory_trace.jsonl`,
  `case_result.json`, `stdout.log`, `stderr.log`, `exitcode`.

- [ ] **Step 1: Write a deterministic fake engine**

Create `tools/test_arrival_load_driver.py` with a fake engine that exposes
`scheduler.waiting`, `is_finished()`, `add_request()`, `step()`, and
`last_step_observation`. Use a manually advanced fake monotonic clock so one
request arrives while another decodes and one step emits two completion
tokens for the same request.

- [ ] **Step 2: Add failing driver contract tests**

Assert:

```python
def test_driver_binds_new_waiting_sequence_and_accounts_injection_lag():
    result = driver.run_case(
        case_spec=_case_spec(),
        workload_rows=_workload(),
        engine_factory=FakeEngine,
        clock_ns=FakeClock([100, 120, 150, 200, 260, 300]),
        output_dir=temporary_path,
    )
    rows = _jsonl(temporary_path / "request_timeline.jsonl")
    assert result["status"] == "PASS"
    assert rows[0]["scheduled_arrival_ns"] < rows[0]["actual_arrival_ns"]
    assert len({row["seq_id"] for row in rows}) == len(rows)


def test_driver_records_multiple_tokens_at_one_step_timestamp():
    row = _timeline_for("multi")
    assert row["token_timestamps_ns"][-2:] == [260, 260]


def test_driver_watchdog_preserves_partial_append_only_evidence():
    output_dir = _temporary_output_dir()
    result = driver.run_case(
        case_spec={
            **_case_spec(),
            "drain_timeout_ns": 500,
        },
        workload_rows=_workload(),
        engine_factory=StuckFakeEngine,
        clock_ns=IncrementingClock(start_ns=100, step_ns=100),
        output_dir=output_dir,
    )
    assert result["status"] == "INCOMPLETE"
    assert result["error_type"] == "drain_timeout"
    assert (output_dir / "scheduler_trace.jsonl").read_bytes()
```

Also test admission exception, duplicate/new waiting ambiguity, unexpected
sequence event, token-count delta mismatch, malformed manifest order, and
final newline preservation.

- [ ] **Step 3: Run tests and verify the missing driver failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_arrival_load_driver.py
```

Expected: FAIL because `tools/arrival_load_driver.py` does not exist.

- [ ] **Step 4: Implement append-only JSONL and due-arrival loop**

Implement a writer that flushes and `os.fsync()`s each JSON line. The loop
must:

```python
while pending_requests or not engine.is_finished():
    now_ns = clock_ns()
    while (
        pending_requests
        and epoch_ns + pending_requests[0]["arrival_offset_ns"] <= now_ns
    ):
        request = pending_requests.popleft()
        scheduled_ns = epoch_ns + request["arrival_offset_ns"]
        actual_ns = clock_ns()
        before_ids = {
            seq.seq_id for seq in engine.scheduler.waiting
        }
        engine.add_request(
            request["prompt_token_ids"],
            sampling_params_for(request),
        )
        appended = [
            seq for seq in engine.scheduler.waiting
            if seq.seq_id not in before_ids
        ]
        if len(appended) != 1:
            raise RuntimeError("ambiguous request-to-sequence binding")
        lifecycle_by_request[request["request_id"]] = {
            "request_id": request["request_id"],
            "seq_id": appended[0].seq_id,
            "scheduled_arrival_ns": scheduled_ns,
            "actual_arrival_ns": actual_ns,
            "first_scheduled_ns": None,
            "first_token_ns": None,
            "token_timestamps_ns": [],
            "completion_ns": None,
            "output_token_ids": [],
            "finish_reason": None,
            "error": None,
        }

    if not engine.is_finished():
        step_start_ns = clock_ns()
        outputs, num_tokens = engine.step()
        step_end_ns = clock_ns()
        observation = dict(engine.last_step_observation)
        observation.update({
            "step_index": step_index,
            "step_start_ns": step_start_ns,
            "step_end_ns": step_end_ns,
            "num_tokens_returned": num_tokens,
        })
        scheduler_writer.append(observation)
        memory_writer.append({
            "step_index": step_index,
            "timestamp_ns": step_end_ns,
            **observation["memory"],
            **{
                key: observation["queue_after"][key]
                for key in (
                    "free_kv_blocks",
                    "used_kv_blocks",
                    "total_kv_blocks",
                    "kv_block_size_tokens",
                )
            },
        })
        for scheduled in observation["scheduled"]:
            request_id = request_id_by_seq[scheduled["seq_id"]]
            lifecycle = lifecycle_by_request[request_id]
            if lifecycle["first_scheduled_ns"] is None:
                lifecycle["first_scheduled_ns"] = step_start_ns
        for seq_id, delta in (
            observation["new_completion_tokens_by_seq"].items()
        ):
            request_id = request_id_by_seq[int(seq_id)]
            lifecycle = lifecycle_by_request[request_id]
            if delta:
                if lifecycle["first_token_ns"] is None:
                    lifecycle["first_token_ns"] = step_end_ns
                lifecycle["token_timestamps_ns"].extend(
                    [step_end_ns] * int(delta)
                )
        for seq_id, token_ids in outputs:
            request_id = request_id_by_seq[seq_id]
            lifecycle = lifecycle_by_request[request_id]
            lifecycle["completion_ns"] = step_end_ns
            lifecycle["output_token_ids"] = list(token_ids)
            lifecycle["finish_reason"] = "length"
        step_index += 1
    elif pending_requests:
        remaining_ns = (
            epoch_ns
            + pending_requests[0]["arrival_offset_ns"]
            - clock_ns()
        )
        if remaining_ns > 0:
            time.sleep(min(remaining_ns / 1_000_000_000, 0.001))
```

Use `time.sleep(min(remaining_seconds, 0.001))` only when the engine has no
work. Never sleep while runnable work exists.

- [ ] **Step 5: Implement exact completion and fail-closed finalization**

Track first scheduled, first token, every token delta, completion, output IDs,
finish reason, and errors by immutable request ID. On exit, verify every
manifest request has exactly one binding and lifecycle record. Write
`case_result.json` atomically after raw streams are closed; always write an
integer `exitcode`, including exception paths.

- [ ] **Step 6: Run driver and gate tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_arrival_load_driver.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_arrival_load_gate.py
```

Expected: both pass.

- [ ] **Step 7: Commit the inline driver**

```bash
git add tools/arrival_load_driver.py tools/test_arrival_load_driver.py
git commit -m "feat: drive frozen request arrivals"
```

---

### Task 6: Calibration, Process Matrix, Resume, and Artifact Finalization

**Files:**
- Modify: `tools/arrival_load_gate.py`
- Modify: `tools/test_arrival_load_gate.py`

**Interfaces:**
- Produces:
  `allocate_port_pair() -> tuple[int, int]`
- Produces:
  `select_lambda_ref(calibration_rows: list[dict]) -> dict`
- Produces:
  `build_case_matrix(run_manifest: dict) -> list[dict]`
- Produces:
  `run_calibration(*, run_dir: Path, python_bin: str, model_path: str, run_manifest: dict, resume: bool = False) -> dict`
- Produces:
  `run_canonical(*, run_dir: Path, python_bin: str, model_path: str, run_manifest: dict, resume: bool = False) -> dict`
- Produces:
  `finalize_artifacts(run_dir: Path) -> dict`
- Produces CLI subcommands:
  `snapshot-source`, `run-calibration`, `freeze-workload`, `run-canonical`,
  `finalize-artifacts`, `verify-harness`.

- [ ] **Step 1: Add failing calibration and matrix tests**

Test rate doubling/bisection, slope threshold, no-stable-point,
no-clear-ceiling, 95%-throughput selection, exact 54 non-alias canonical
cases (`3 policies x 6 scenarios x 3 repetitions`), Latin-square policy
ordering by repetition, and unique case keys.

The process order must rotate:

```python
POLICY_ORDER_BY_REPETITION = {
    0: ("P0", "P2", "P3"),
    1: ("P2", "P3", "P0"),
    2: ("P3", "P0", "P2"),
}
```

Within each repetition, interleave by scenario first and policy second.

- [ ] **Step 2: Add failing resume and process tests**

Use a fake subprocess command to assert:

- each launched case receives a never-before-used port pair;
- a complete hash-valid row is immutable on resume;
- failed/incomplete case raw directory is moved to
  `processes/{case_id}.replaced.{time.time_ns()}` before replacement;
- changed source/workload/policy/environment/run tag rejects resume;
- canonical cannot begin without a verified smoke marker;
- calibration cannot change after canonical rows exist.

- [ ] **Step 3: Implement calibration selection and case matrix**

Compute backlog OLS slope from `(relative_time_s, unfinished_count)` samples
in the final third of the offered window. A calibration row is stable only
when all structural/correctness conditions and the fixed slope threshold
pass. Return `INCOMPLETE` unless at least one stable point and one higher
unstable point establish a ceiling.

- [ ] **Step 4: Implement isolated process launch**

Allocate ports by binding two local ephemeral sockets until distinct values
are obtained, close them immediately before launch, pass both through the
child environment, and record command, PID, start/end timestamps, stdout,
stderr, exit code, source/workload/policy hashes, and ports. Retry only a
narrow startup `EADDRINUSE` failure with a fresh pair; never retry correctness
or performance failures automatically.

- [ ] **Step 5: Implement source evidence and immutable snapshot**

Reuse `tools/source_audit.py` with:

```python
OWNED_SOURCE_ROOTS = (
    "tinyvllm",
    "tools/source_audit.py",
    "tools/arrival_load_gate.py",
    "tools/arrival_load_driver.py",
    "tools/arrival_load_verify.py",
    "tools/test_arrival_load_gate.py",
    "tools/test_arrival_load_driver.py",
    "tools/test_arrival_load_verify.py",
    "tools/test_chunked_prefill.py",
    "tools/run_arrival_load_gate_remote.sh",
    "tools/test_run_arrival_load_gate_remote.py",
)
```

Generate `source_evidence.json`, `source.patch`, staged `source/`, and
`source_snapshot.tar.gz` from the same immutable bytes. Reject untracked owned
source and unrelated dirty tracked source before canonical snapshot.

- [ ] **Step 6: Implement raw merge and artifact hashes**

Merge case-local JSONL streams deterministically by
`(scenario_order, repetition, policy_order, step/request key)`. Preserve
case-local files under `processes/{case_id}/`. Hash every final file except
`artifact_hashes.json`, then hash canonical JSON values in
`artifact_hashes.json`. `summary.json` and `report.md` are derived only after
all raw files exist.

- [ ] **Step 7: Run orchestration, source, and resume tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_arrival_load_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_arrival_load_driver.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_arrival_load_verify.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_source_audit.py
```

Expected: all pass.

- [ ] **Step 8: Commit orchestration and artifact finalization**

```bash
git add \
  tools/arrival_load_gate.py \
  tools/test_arrival_load_gate.py
git commit -m "feat: orchestrate arrival load canonical gate"
```

---

### Task 7: Safe Remote Runner and Recovery Modes

**Files:**
- Create: `tools/run_arrival_load_gate_remote.sh`
- Create: `tools/test_run_arrival_load_gate_remote.py`
- Modify: `tools/arrival_load_gate.py`

**Interfaces:**
- Produces modes:
  `preflight`, `smoke`, `calibration`, `canonical`, `download-only`,
  `verify-only`.
- Uses remote root:
  `/data00/home/sitian/sitian-workspace01/tllm/arrival-load-runs/$RUN_TAG`.
- Uses local root:
  `experiments/arrival_load/$RUN_TAG`.

- [ ] **Step 1: Write failing shell contract tests**

Create tests asserting the runner contains the exact remote host/Python/model,
immutable staging before upload, `TMPDIR` under the run directory, detached
launch, atomic `remote_exitcode`, per-file chunk download with zero-byte
support, source preflight, local independent verification, and no `pkill`,
`killall`, shared `/tmp` cleanup, remote checkout mutation, `rsync`, or
`git add -A`.

Assert `download-only` and `verify-only` exit before snapshot, upload, or model
launch. Assert every model command receives dynamic
`TINYVLLM_DIST_PORT`/`MASTER_PORT` from the Python orchestrator rather than
fixed shell constants.

- [ ] **Step 2: Run tests and verify missing runner failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python3 tools/test_run_arrival_load_gate_remote.py
```

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement preflight and immutable upload**

Follow the proven speculation-router transport pattern:

1. create `${LOCAL_OUT}.staging`;
2. run `arrival_load_gate.py snapshot-source`;
3. upload only staging bytes through a stdin-capable SSH transport;
4. remotely extract to `staging.upload`, verify source and package/model/GPU
   identity, then atomically rename to `staging`;
5. write `source_preflight.json` and `capability.json`;
6. download preflight artifacts with the chunk transport.

Run dependency-light tests remotely before any model initialization:

```bash
"${REMOTE_PYTHON}" tools/test_arrival_load_gate.py
"${REMOTE_PYTHON}" tools/test_arrival_load_driver.py
"${REMOTE_PYTHON}" tools/test_arrival_load_verify.py
"${REMOTE_PYTHON}" tools/test_chunked_prefill.py
```

- [ ] **Step 4: Implement detached model modes**

`smoke`, `calibration`, and `canonical` launch under
`nohup bash -c "${REMOTE_COMMAND_Q}"`, redirect to run-local `runner.log`, and publish
`remote_exitcode.tmp` then atomic rename. Poll only that run's exit-code file.
On nonzero exit, download every available artifact before returning failure.

- [ ] **Step 5: Implement recovery and verification**

`download-only` requires explicit `RUN_TAG`, downloads recursively using safe
relative paths and block retries, and never launches work. `verify-only`
runs:

```bash
python3 tools/arrival_load_verify.py --run-dir "${LOCAL_OUT}"
```

without SSH or model work. Successful normal modes download, verify artifact
hashes, execute the independent verifier, and require
`independent-verify/verify.exitcode == 0`.

- [ ] **Step 6: Run shell tests and syntax validation**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python3 tools/test_run_arrival_load_gate_remote.py
bash -n tools/run_arrival_load_gate_remote.sh
```

Expected: both pass.

- [ ] **Step 7: Commit the remote runner**

```bash
git add \
  tools/run_arrival_load_gate_remote.sh \
  tools/test_run_arrival_load_gate_remote.py \
  tools/arrival_load_gate.py
git commit -m "feat: run arrival load gate remotely"
```

---

### Task 8: Full Local Verification Before GPU Work

**Files:**
- Verify all changed implementation and test files.

**Interfaces:**
- Produces a clean implementation checkpoint; no remote model work yet.

- [ ] **Step 1: Run all focused dependency-light tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_arrival_load_gate.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_arrival_load_driver.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_arrival_load_verify.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_run_arrival_load_gate_remote.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_profile_chunked_prefill.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/test_source_audit.py
```

Expected: all pass.

- [ ] **Step 2: Compile changed Python and validate shell**

```bash
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
  python3 -m py_compile \
  tools/arrival_load_gate.py \
  tools/arrival_load_driver.py \
  tools/arrival_load_verify.py \
  tools/test_arrival_load_gate.py \
  tools/test_arrival_load_driver.py \
  tools/test_arrival_load_verify.py \
  tools/test_run_arrival_load_gate_remote.py \
  tinyvllm/engine/scheduler.py \
  tinyvllm/engine/llm_engine.py \
  tinyvllm/engine/model_runner.py
bash -n tools/run_arrival_load_gate_remote.sh
git diff --check
```

Expected: all exit `0`.

- [ ] **Step 3: Audit owned-source and experiment boundaries**

```bash
git status --short
git diff --name-only HEAD
```

Expected: only intended source/docs are modified; all existing
`experiments/adaptive_ngram/20260717-k1-sam-*` and
`experiments/speculation_router/`
directories remain untracked and untouched.

- [ ] **Step 4: Commit any test-only corrections selectively**

If verification required corrections:

```bash
git add \
  tools/arrival_load_gate.py \
  tools/arrival_load_driver.py \
  tools/arrival_load_verify.py \
  tools/test_arrival_load_gate.py \
  tools/test_arrival_load_driver.py \
  tools/test_arrival_load_verify.py \
  tools/test_run_arrival_load_gate_remote.py \
  tinyvllm/engine/scheduler.py \
  tinyvllm/engine/llm_engine.py \
  tinyvllm/engine/model_runner.py \
  tools/test_chunked_prefill.py
git commit -m "test: harden arrival load gate coverage"
```

Do not create an empty commit.

---

### Task 9: Remote Two-to-Four-Request Lifecycle Smoke

**Files:**
- Generate:
  `experiments/arrival_load/$SMOKE_RUN_TAG/`
- Modify source only if the smoke exposes an instrumentation/lifecycle defect.

**Interfaces:**
- Consumes the exact committed source snapshot from Tasks 1-8.
- Produces a verifier-confirmed smoke artifact; cannot classify performance.

- [ ] **Step 1: Run remote preflight**

```bash
RUN_TAG="qwen3-06b-arrival-preflight-$(date +%Y%m%d-%H%M%S)" \
  tools/run_arrival_load_gate_remote.sh preflight
```

Expected: source preflight and all remote dependency-light tests pass; no
model case runs.

- [ ] **Step 2: Run the lifecycle smoke**

```bash
RUN_TAG="qwen3-06b-arrival-smoke-$(date +%Y%m%d-%H%M%S)" \
  tools/run_arrival_load_gate_remote.sh smoke
```

The smoke manifest must contain:

- one decode-active request before a later arrival;
- one long prompt requiring more than one 128-token prefill chunk;
- one output with at least three tokens and therefore ITL samples;
- P0 plus one policy that exercises its named branch;
- two to four total requests;
- exact P0/candidate output equality;
- unique port pairs;
- nonempty request timeline, scheduler trace, and memory trace.

Expected: remote and downloaded independent verification complete, while
classification remains explicitly `SMOKE_ONLY`.

- [ ] **Step 3: Audit the smoke artifact independently**

```bash
SMOKE_RUN_TAG="qwen3-06b-arrival-smoke-20260719-000000"
python3 tools/arrival_load_verify.py \
  --run-dir "experiments/arrival_load/${SMOKE_RUN_TAG}"
python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path("experiments/arrival_load") / os.environ["SMOKE_RUN_TAG"]
summary = json.loads(
    (root / "independent-verify/summary.json").read_text()
)
assert summary["classification"] == "SMOKE_ONLY"
assert summary["lifecycle_complete"] is True
assert summary["exact_outputs"] is True
print("ARRIVAL_SMOKE_AUDIT_OK")
PY
```

Expected: `ARRIVAL_SMOKE_AUDIT_OK`.

- [ ] **Step 4: Fix only root-cause smoke defects and rerun with a new tag**

If the smoke fails, preserve the failed artifact, write a focused failing
local test, make the smallest correction, rerun Tasks 8.1-8.3, commit
selectively, and launch a new smoke tag. Never overwrite or reinterpret a
failed smoke.

---

### Task 10: P0 Saturation Calibration and Frozen Workload

**Files:**
- Generate:
  `experiments/arrival_load/$RUN_TAG/calibration_manifest.jsonl`,
  `experiments/arrival_load/$RUN_TAG/calibration_rows.jsonl`,
  and `workload_manifest.jsonl`.

**Interfaces:**
- Requires a hash-valid smoke marker for the same source/environment identity.
- Produces one frozen `lambda_ref` and canonical workload hash.

- [ ] **Step 1: Start a new canonical run tag with calibration**

```bash
RUN_TAG="qwen3-06b-arrival-canonical-$(date +%Y%m%d-%H%M%S)" \
  SMOKE_RUN_TAG="qwen3-06b-arrival-smoke-20260719-000000" \
  tools/run_arrival_load_gate_remote.sh calibration
```

Expected: P0-only calibration candidates run serially with unique ports; the
artifact records stable/unstable decisions, backlog trajectories, exact
outputs, and selected `lambda_ref`.

- [ ] **Step 2: Verify calibration before canonical launch**

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("experiments/arrival_load/qwen3-06b-arrival-canonical-20260719-000000")
manifest = json.loads((root / "run_manifest.json").read_text())
calibration = manifest["calibration"]
assert calibration["status"] == "PASS"
assert calibration["lambda_ref_rps"] > 0
assert calibration["stable_rate_rps"] < calibration["unstable_rate_rps"]
assert manifest["workload_sha256"]
print(
    "ARRIVAL_CALIBRATION_OK",
    calibration["lambda_ref_rps"],
    manifest["workload_sha256"],
)
PY
```

Expected: output starts with `ARRIVAL_CALIBRATION_OK`, followed by a positive
floating-point rate and a 64-character lowercase workload SHA-256.

- [ ] **Step 3: Stop if calibration is ambiguous**

If no stable point, no higher unstable point, output mismatch, non-finite
metric, or drain failure prevents a valid `lambda_ref`, retain the artifact
as `INCOMPLETE`. Do not guess a rate, change bounds after inspecting results,
or launch canonical cases under that tag.

---

### Task 11: Serial Canonical Policy/Scenario/Repetition Matrix

**Files:**
- Complete:
  `experiments/arrival_load/$RUN_TAG/`

**Interfaces:**
- Consumes the immutable source, calibration, and workload hashes from Task 10.
- Produces all non-alias P0/P2/P3 cases and the P1 alias record.

- [ ] **Step 1: Launch canonical execution**

```bash
RUN_TAG="qwen3-06b-arrival-canonical-20260719-000000" \
  RESUME=1 \
  tools/run_arrival_load_gate_remote.sh canonical
```

Expected: exactly 54 non-alias case processes execute serially in the frozen
interleaved order, each with unique ports and complete process artifacts.

- [ ] **Step 2: Keep polling the detached run to completion**

Use only run-local inspection:

```bash
ssh -n -S /tmp/ssh-sitian-10.232.195.203 \
  sitian@10.232.195.203 \
  "tail -80 '/data00/home/sitian/sitian-workspace01/tllm/arrival-load-runs/qwen3-06b-arrival-canonical-20260719-000000/runner.log'"
```

Do not kill other users' processes or clear system temporary directories.

- [ ] **Step 3: Recover transport failures without relaunching model work**

If SSH/download fails after remote completion:

```bash
RUN_TAG="qwen3-06b-arrival-canonical-20260719-000000" \
  tools/run_arrival_load_gate_remote.sh download-only
RUN_TAG="qwen3-06b-arrival-canonical-20260719-000000" \
  tools/run_arrival_load_gate_remote.sh verify-only
```

If a case is structurally incomplete, use canonical `RESUME=1`; the
orchestrator must preserve complete cases and replace only failed case
directories with fresh ports.

- [ ] **Step 4: Run final independent verification**

```bash
python3 tools/arrival_load_verify.py \
  --run-dir "experiments/arrival_load/qwen3-06b-arrival-canonical-20260719-000000"
cat \
  "experiments/arrival_load/qwen3-06b-arrival-canonical-20260719-000000/independent-verify/verify.exitcode"
```

Expected: exit code `0` means the verifier completed and agreed with the raw
evidence; inspect the classification separately.

- [ ] **Step 5: Perform the completion audit against raw evidence**

Verify:

1. source snapshot reconstructs and matches the recorded tree hash;
2. model, tokenizer, environment, workload, prompt-bank, policy, and port
   identities match;
3. P1 is an alias only if its resolved identity equals P0;
4. all 54 required non-alias cases and three repetitions exist;
5. every admitted request has one binding, exact output, valid finish state,
   complete timeline, and no starvation;
6. all request, scheduler, memory, stdout/stderr, and exit-code evidence is
   present;
7. harness and independent percentiles, medians, worst repetitions, guards,
   and final classification agree;
8. no result relies on dropping requests, reducing output length, omitting a
   bucket, or selecting a best repetition.

Any uncertainty remains `INCOMPLETE`; do not mark the long-running objective
complete.

---

### Task 12: Publish Verified Result and Claim Boundaries

**Files:**
- Modify: `README.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Preserve:
  `experiments/arrival_load/$RUN_TAG/`

**Interfaces:**
- Consumes only the independent-verifier-confirmed artifact.
- Produces durable commands, hashes, result, proof boundary, limitations, and
  next decision.

- [ ] **Step 1: Add the reproducible command and result to README**

Document:

- canonical run tag and artifact path;
- source commit/tree/workload/prompt-bank hashes;
- exact remote Python/model/host;
- calibration `lambda_ref`;
- P0/P1 alias outcome;
- policy/scenario/repetition matrix;
- verifier-confirmed classification;
- winning or best diagnostic candidate with throughput, TTFT, ITL, E2E,
  decode-gap, bucket, and memory ratios;
- the command for `download-only` and `verify-only`;
- explicit statement that prior K1/router results remain `NO_GO`.

- [ ] **Step 2: Update the handoff with proof and non-proof**

In `AGENT_HANDOFF_STATE.md`, record:

- what passed and what it proves;
- what did not pass or remains weak;
- whether any policy may proceed to an integration design;
- that this is an inline synchronous arrival gate, not HTTP/RPC production
  serving;
- model/GPU/workload/generalization limits;
- the next highest-value direction if result is `NO_GO` or
  `PROMISING_NOT_PROVEN`.

- [ ] **Step 3: Validate documentation against the artifact**

Run a small Python audit that extracts every number/hash written into README
and handoff from `run_manifest.json` and
`independent-verify/summary.json`. Then run:

```bash
git diff --check
git status --short
```

Expected: docs match the artifact and experiment directories remain
untracked.

- [ ] **Step 4: Commit documentation selectively**

```bash
git add README.md AGENT_HANDOFF_STATE.md
git commit -m "docs: record arrival load gate result"
```

Do not stage experiment artifacts.

## Completion Criteria

This plan is complete only when all of the following are true:

- dependency-light tests cover deterministic workload generation, lifecycle
  reconstruction, one-token/no-ITL behavior, multi-token same-step behavior,
  metrics, fairness, memory, starvation, aliases, repetitions, every
  classification boundary, resume, source, ports, hashes, and verifier
  independence;
- existing chunked-prefill regressions still pass;
- observation-only instrumentation is verified not to alter scheduler
  outcomes;
- a remote lifecycle smoke passes exact output and independent verification;
- P0 calibration produces an unambiguous frozen `lambda_ref`;
- every required canonical case has three complete repetitions or the run is
  honestly `INCOMPLETE`;
- the standalone verifier reconstructs and agrees with every gated metric and
  final classification;
- README and handoff report the verified result and its limitations without
  promoting K1/router negative results or generalizing beyond Qwen3-0.6B and
  the frozen arrival matrix.
