# Autoregressive Draft Instability Telemetry Implementation Plan

> **For agentic workers:** Execute inline in the current checkout. Do not use
> subagents, create or switch branches/worktrees, stage, commit, push, stash,
> reset, or clean.

**Goal:** Add source-bound, per-repeat GPU and host telemetry to the existing
TP4 Qwen3 batch-4 stability diagnostic without adding CUDA synchronization to
the measured request path.

**Architecture:** Worker results gain wall-clock repeat intervals. A separate
telemetry module parses external sampler output, aligns GPU samples to those
intervals, enforces coverage, and produces an attribution artifact. A focused
remote runner owns sampler lifecycle and retains raw logs, dual verifier
receipts, source hashes, and a complete manifest.

**Tech Stack:** Python 3.11, pytest, JSON, CSV, Bash, SSH, `nvidia-smi`,
`vmstat`, `mpstat`, `pidstat`.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Remote host is `sitian@10.232.195.203`.
- Preserve temperature zero, exact greedy parity, accepted-prefix semantics,
  `MAX_PROPOSAL_TOKENS=4`, and workload-derived Proposal-KV capacity.
- Do not add `torch.cuda.synchronize()` or any system query inside
  `engine.step()`.
- Do not terminate unrelated GPU processes; preserve the existing GPU-7
  `python3` service.
- Do not treat synthetic Proposal-KV copies as real movement.
- Use `/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815` for remote
  source and artifacts; do not write new artifacts under `/data00`.
- Each measured repeat requires at least five telemetry samples for each of
  GPUs `3,4,6,7`.
- Keep target-then-learned order for the first telemetry campaign.

---

### Task 1: Worker Wall-Clock Repeat Intervals

**Files:**
- Modify: `tools/autoregressive_draft_performance_worker.py`
- Modify: `tools/test_autoregressive_draft_performance_gate.py`

**Interfaces:**
- Consumes: existing `run_policy_campaign(...)`.
- Produces: `campaign_interval` on every warmup and measured result, with
  integer `started_at_unix_ns` and `finished_at_unix_ns`.
- Produces: injectable `wall_clock_ns: Callable[[], int]`, defaulting to
  `time.time_ns`.

- [x] **Step 1: Write deterministic interval tests**

Add a fake wall clock and assert exact interval placement:

```python
wall_times = iter([
    1_000, 2_000,
    3_000, 4_000,
])

result = worker.run_policy_campaign(
    ...,
    warmup_runs=1,
    measured_runs=1,
    wall_clock_ns=lambda: next(wall_times),
)

assert result["warmup_runs"][0]["campaign_interval"] == {
    "started_at_unix_ns": 1_000,
    "finished_at_unix_ns": 2_000,
}
assert result["measured_runs"][0]["campaign_interval"] == {
    "started_at_unix_ns": 3_000,
    "finished_at_unix_ns": 4_000,
}
```

Also assert that `run_batch_fn` receives the original repeat values and that
its returned timing/runtime payload remains unchanged.

- [x] **Step 2: Run the focused test and confirm RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_performance_gate.py \
  -k campaign_interval
```

Expected: failure because `wall_clock_ns` and `campaign_interval` do not yet
exist.

- [x] **Step 3: Add the minimal interval wrapper**

Change the signature:

```python
def run_policy_campaign(
    *,
    ...,
    wall_clock_ns=time.time_ns,
    run_batch_fn=run_request_batch,
    ...
) -> dict:
```

Replace the current `run_once` body with:

```python
def run_once(repeat: int):
    started_at_unix_ns = wall_clock_ns()
    result = run_batch_fn(
        engine=engine,
        policy=policy,
        prompt_rows=prompt_rows,
        sampling_params=sampling_params,
        expected_output_tokens=MAX_OUTPUT_TOKENS,
        synchronize=synchronize,
        clock_ns=clock_ns,
        repeat=repeat,
    )
    finished_at_unix_ns = wall_clock_ns()
    if (
        isinstance(started_at_unix_ns, bool)
        or not isinstance(started_at_unix_ns, int)
        or isinstance(finished_at_unix_ns, bool)
        or not isinstance(finished_at_unix_ns, int)
        or started_at_unix_ns <= 0
        or finished_at_unix_ns <= started_at_unix_ns
    ):
        raise ValueError("campaign interval is invalid")
    return {
        **result,
        "campaign_interval": {
            "started_at_unix_ns": started_at_unix_ns,
            "finished_at_unix_ns": finished_at_unix_ns,
        },
    }
```

Pass `wall_clock_ns=time.time_ns` from `_default_dependencies`.

- [x] **Step 4: Run worker and gate tests**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_performance_gate.py
```

Expected: all tests pass, including deterministic interval coverage.

---

### Task 2: GPU Telemetry Parser and Interval Alignment

**Files:**
- Create: `tools/autoregressive_draft_instability_telemetry.py`
- Create: `tools/test_autoregressive_draft_instability_telemetry.py`

**Interfaces:**
- Produces:

```python
GPU_FIELDS = (
    "sampled_at_unix_ns",
    "timestamp",
    "index",
    "uuid",
    "pstate",
    "sm_clock_mhz",
    "memory_clock_mhz",
    "power_w",
    "temperature_c",
    "gpu_utilization_percent",
    "memory_utilization_percent",
    "memory_used_mib",
    "throttle_reasons_active",
)

def parse_gpu_telemetry(text: str) -> list[dict]: ...

def validate_campaign_intervals(worker: dict) -> None: ...

def summarize_gpu_telemetry(
    worker: dict,
    samples: list[dict],
    *,
    expected_gpu_indices: tuple[int, ...] = (3, 4, 6, 7),
    minimum_samples: int = 5,
) -> dict: ...
```

- [x] **Step 1: Write parser RED tests**

Use exact driver-compatible rows:

```python
CSV_ROW = (
    "1786808823101000000, 2026/08/15 23:47:03.101, 3, "
    "GPU-f8904cb4-f9f0-c757-df36-e6fd971b3a9d, "
    "P0, 1410, 1512, 70.38, 41, 93, 12, 72455, "
    "0x0000000000000001"
)
```

Assert that `sampled_at_unix_ns` is used as the alignment key, while the
timezone-less `nvidia-smi` timestamp is retained as text for audit. Also
assert numeric types, UUID, P-state, and parsed throttle mask. Add failures
for missing columns, invalid epoch nanoseconds, malformed audit timestamp,
boolean numeric values, negative power, utilization outside `[0,100]`, and
duplicate GPU index/epoch rows.

- [x] **Step 2: Run parser tests and confirm RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_instability_telemetry.py \
  -k parse
```

Expected: import failure because the telemetry module does not exist.

- [x] **Step 3: Implement strict CSV parsing**

Use `csv.reader`; validate the audit timestamp with
`datetime.strptime(..., "%Y/%m/%d %H:%M:%S.%f")`, but do not convert it to
epoch time. Return sorted rows with these keys:

```python
{
    "sampled_at_unix_ns": int,
    "nvidia_timestamp": str,
    "gpu_index": int,
    "uuid": str,
    "pstate": str,
    "sm_clock_mhz": int,
    "memory_clock_mhz": int,
    "power_w": float,
    "temperature_c": int,
    "gpu_utilization_percent": int,
    "memory_utilization_percent": int,
    "memory_used_mib": int,
    "throttle_reasons_active": int,
}
```

Reject empty input and duplicate `(sampled_at_unix_ns, gpu_index)` keys.

- [x] **Step 4: Write interval and aggregation RED tests**

Construct two measured intervals and samples at the lower boundary, inside,
and at the upper boundary. Define membership as:

```text
started_at_unix_ns <= sampled_at_unix_ns <= finished_at_unix_ns
```

Assert:

- exactly four expected GPU indices;
- at least five samples per repeat/GPU;
- min/median/max for clocks, power, temperature, utilization, and memory;
- sorted distinct P-states;
- throttle mask bitwise OR;
- overlapping or non-monotonic intervals fail;
- missing GPU coverage yields `ValueError`.

- [x] **Step 5: Implement interval validation and aggregation**

Return:

```python
{
    "expected_gpu_indices": [3, 4, 6, 7],
    "minimum_samples_per_repeat_gpu": 5,
    "measured_runs": [
        {
            "repeat": 0,
            "campaign_interval": {...},
            "gpus": [
                {
                    "gpu_index": 3,
                    "sample_count": 5,
                    "sm_clock_mhz": {
                        "minimum": 1245,
                        "median": 1410,
                        "maximum": 1410,
                    },
                    ...
                }
            ],
        }
    ],
}
```

Do not infer causality in this module.

- [x] **Step 6: Run the telemetry unit suite**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_instability_telemetry.py
```

Expected: all parser, interval, coverage, and aggregation tests pass.

---

### Task 3: Source-Bound Artifact and Independent Verifier

**Files:**
- Modify: `tools/autoregressive_draft_instability_telemetry.py`
- Create: `tools/verify_autoregressive_draft_instability_telemetry.py`
- Modify: `tools/test_autoregressive_draft_instability_telemetry.py`

**Interfaces:**
- Produces:

```python
def build_instability_telemetry_artifact(
    *,
    timing_artifact: dict,
    target_worker: dict,
    learned_worker: dict,
    target_gpu_samples: list[dict],
    learned_gpu_samples: list[dict],
    source_files: dict[str, str],
    host_files: dict[str, str],
) -> dict: ...

def verify_instability_telemetry(
    artifact_path: Path,
    repo_root: Path,
) -> dict: ...
```

- [x] **Step 1: Write artifact classification RED tests**

Cover:

```text
INVALID_TELEMETRY:
  any measured repeat/GPU has fewer than five samples

STABLE_BASELINE:
  timing artifact classification is STABLE

ENVIRONMENT_CORRELATED:
  timing is UNSTABLE and a slow repeat contains lower SM clock,
  nonzero throttle mask, changed P-state, or materially elevated host/GPU
  contention

RUNTIME_VARIANCE_SUSPECTED:
  timing is UNSTABLE while sampled GPU P-state, clocks, throttle masks,
  temperature, utilization, and retained host summaries remain within the
  declared stable envelope
```

The first implementation should make only direct, auditable comparisons:

- any nonzero throttle mask during a measured interval;
- multiple P-states during measured intervals;
- minimum SM clock below `95%` of that policy's median maximum clock;
- maximum temperature at least `10 C` above that policy's minimum
  temperature;
- GPU utilization on a selected GPU above `0%` in a gap outside the worker
  interval is retained but not classified as causal.

- [x] **Step 2: Implement artifact assembly**

Artifact schema:

```python
{
    "schema_version": 1,
    "status": "PASS",
    "timing_classification": "UNSTABLE",
    "telemetry_classification": "ENVIRONMENT_CORRELATED",
    "exact_parity": True,
    "policies": {
        "target": {...},
        "learned": {...},
    },
    "host_files": {
        "target_vmstat": {"path": str, "sha256": str},
        ...
    },
    "source_files": {
        "tools/autoregressive_draft_performance_worker.py": sha256,
        "tools/autoregressive_draft_instability_telemetry.py": sha256,
        "tools/verify_autoregressive_draft_instability_telemetry.py": sha256,
        "tools/autoregressive_draft_b4_timing_diagnostic.py": sha256,
        "tools/autoregressive_draft_performance_gate.py": sha256,
    },
    "limitations": [...],
}
```

The artifact must preserve per-repeat summaries and all classification
reasons. Host logs remain raw hash-bound evidence in the first version.

- [x] **Step 3: Write verifier tamper tests**

Assert verifier rejection for:

- changed source hash;
- changed GPU sample count or summary;
- missing selected GPU;
- interval overlap;
- changed timing classification or exact parity;
- unsafe source path;
- missing host log hash.

- [x] **Step 4: Implement independent verifier**

The verifier reloads the artifact, recomputes all deterministic summaries and
classification from embedded normalized samples, verifies repository source
hashes, and returns:

```python
{
    "status": "PASS",
    "schema_version": 1,
    "timing_classification": "UNSTABLE",
    "telemetry_classification": "...",
    "exact_parity": True,
    "source_files_verified": 5,
}
```

- [x] **Step 5: Run artifact and verifier tests**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_instability_telemetry.py
python3 -m py_compile \
  tools/autoregressive_draft_instability_telemetry.py \
  tools/verify_autoregressive_draft_instability_telemetry.py
```

Expected: all tests and compilation pass.

---

### Task 4: Focused Remote Sampler Runner

**Files:**
- Create: `tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh`
- Modify: `tools/test_autoregressive_draft_instability_telemetry.py`

**Interfaces:**
- Consumes the existing worker, timing diagnostic, telemetry assembler, and
  verifier.
- Produces one local artifact directory with raw samplers, worker JSON,
  timing and telemetry artifacts, dual receipts, source tarball, and manifest.

- [x] **Step 1: Add shell-contract tests**

Read the runner text and assert it contains:

- `sitian@10.232.195.203`;
- GPUs `3,4,6,7`;
- a `date +%s%N` epoch prefix and `sleep 0.2` sampling loop;
- all twelve validated GPU query fields;
- `vmstat -t 1`, `mpstat -P ALL 1`, and `pidstat -u -r -d -h 1`;
- two warmups and eight measured runs;
- target before learned;
- cleanup trap that kills only sampler PIDs started by this script;
- remote and local telemetry verifier calls;
- manifest generation excluding only `manifest.sha256`;
- no new `torch.cuda.synchronize` text.

- [x] **Step 2: Run shell tests and confirm RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_instability_telemetry.py \
  -k runner
```

Expected: failure because the runner does not exist.

- [x] **Step 3: Implement sampler lifecycle**

Inside the remote campaign define:

```bash
sampler_pids=()

stop_samplers() {
  local pid
  for pid in "${sampler_pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  sampler_pids=()
}

trap stop_samplers EXIT TERM INT
```

Start only script-owned background processes and store each PID. Stop all
samplers after each policy worker before assembling artifacts.

- [x] **Step 4: Implement the remote workflow**

For target and learned separately:

1. start GPU and host samplers;
2. run the worker with `2` warmups and `8` measured runs;
3. stop samplers;
4. retain sampler exit statuses;
5. run the existing timing diagnostic;
6. run the telemetry assembler;
7. run the telemetry verifier remotely;
8. capture final `nvidia-smi`;
9. rsync artifacts locally;
10. run the verifier locally;
11. generate the complete manifest.

Use the existing remote Python:

```text
/data00/home/sitian/miniconda3/envs/py311/bin/python
```

Use the verified writable base:

```text
/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815
```

- [x] **Step 5: Validate runner syntax and contract**

Run:

```bash
bash -n \
  tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh
python3 -m pytest -q \
  tools/test_autoregressive_draft_instability_telemetry.py \
  -k runner
```

Expected: syntax and runner-contract tests pass.

---

### Task 5: Full Local Gate and Remote Authority Campaign

**Files:**
- Modify after results:
  `docs/superpowers/audits/2026-08-15-phase1-prompt-to-artifact-coverage.md`
- Modify after results: `AGENT_HANDOFF_STATE.md`
- Create after results:
  `experiments/autoregressive_draft/<telemetry-run>/README.md`

**Interfaces:**
- Produces the first telemetry-backed TP4 batch-4 stability authority.

- [x] **Step 1: Run the complete local gate**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_performance_gate.py \
  tools/test_autoregressive_draft_instability_telemetry.py
python3 -m py_compile \
  tinyvllm/engine/autoregressive_draft_executor.py \
  tools/autoregressive_draft_performance_worker.py \
  tools/autoregressive_draft_b4_timing_diagnostic.py \
  tools/autoregressive_draft_instability_telemetry.py \
  tools/verify_autoregressive_draft_instability_telemetry.py
bash -n \
  tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh
```

Expected: all tests, compilation, and shell syntax pass.

- [x] **Step 2: Run the source-bound remote campaign**

Use a new output directory:

```bash
bash tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh \
  --remote-host sitian@10.232.195.203 \
  --remote-python /data00/home/sitian/miniconda3/envs/py311/bin/python \
  --remote-base /dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815 \
  --target-model \
    /dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815/target-qwen3-1.7b \
  --draft-model \
    /dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815/draft \
  --gpu-indices 3,4,6,7 \
  --local-run \
    experiments/autoregressive_draft/tp4-qwen3-b4-instability-telemetry-gpu3467-r1-20260815
```

- [x] **Step 3: Verify the downloaded authority**

Run:

```bash
python3 tools/verify_autoregressive_draft_instability_telemetry.py \
  --artifact \
    experiments/autoregressive_draft/tp4-qwen3-b4-instability-telemetry-gpu3467-r1-20260815/telemetry.json \
  --repo-root "$PWD"

(
  cd \
    experiments/autoregressive_draft/tp4-qwen3-b4-instability-telemetry-gpu3467-r1-20260815
  shasum -a 256 -c manifest.sha256
)
```

Expected: verifier and every manifest entry pass.

- [x] **Step 4: Apply the decision tree**

Record exactly one:

```text
ENVIRONMENT_CORRELATED:
  investigate or control the observed system condition before optimization

RUNTIME_VARIANCE_SUSPECTED:
  run the reverse-order campaign before selecting a runtime optimization

STABLE_BASELINE:
  use the existing critical-rank attribution to choose the next optimization

INVALID_TELEMETRY:
  repair sampling coverage and rerun; make no performance claim
```

- [x] **Step 5: Persist claim boundaries**

Update the bundle README, phase audit, and handoff with:

- exact workload and source hashes;
- sample coverage per repeat/GPU;
- stationarity and telemetry classification;
- exact parity, acceptance, memory, and timing;
- what the telemetry supports and does not support;
- the next experiment;
- unchanged `PHASE_1=NOT_ACHIEVED` unless every original promotion gate has
  independently passed.

- [x] **Step 6: Run the final scoped audit**

Run:

```bash
git diff --check -- \
  tools/autoregressive_draft_performance_worker.py \
  tools/autoregressive_draft_instability_telemetry.py \
  tools/verify_autoregressive_draft_instability_telemetry.py \
  tools/test_autoregressive_draft_instability_telemetry.py \
  tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh \
  docs/superpowers/specs/2026-08-15-autoregressive-draft-instability-telemetry-design.md \
  docs/superpowers/plans/2026-08-15-autoregressive-draft-instability-telemetry.md \
  docs/superpowers/audits/2026-08-15-phase1-prompt-to-artifact-coverage.md \
  AGENT_HANDOFF_STATE.md
```

Expected: no whitespace errors. Do not stage, commit, or push.

## Reverse-Order Extension

- [x] Add strict `--policy-order` support for exactly `target,learned` or
  `learned,target`.
- [x] Preserve `target,learned` as the default first-campaign order.
- [x] Run the source-identical verifier-bound reverse campaign as
  `learned,target`.
- [x] Verify exact parity, timing and telemetry artifacts remotely and
  locally, plus the complete manifest.
- [x] Compare the orders without merging their medians.
- [x] Persist the supported policy-position/process-cadence interpretation
  and the non-causal claim boundary.

Reverse authority:

```text
experiments/autoregressive_draft/
  tp4-qwen3-b4-instability-telemetry-reverse-gpu3467-r4-20260815
```

Decision:

```text
Both orders remain UNSTABLE / RUNTIME_VARIANCE_SUSPECTED.
Both policies are faster in the second process position.
The effect supports a process-cadence control experiment.
It does not establish a specific runtime root cause.
Do not select an optimization before the same-policy priming gate.
```
