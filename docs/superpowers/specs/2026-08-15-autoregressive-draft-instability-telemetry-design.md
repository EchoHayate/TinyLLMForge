# Autoregressive Draft Instability Telemetry Design

## Purpose

The TP4 Qwen3 independent-draft batch-4 diagnostic is source-bound and exact,
but both target and learned timing are non-stationary. The next campaign must
determine whether repeat-to-repeat variation follows GPU or host conditions
before any CUDA Graph, TP-authority, or metadata optimization is attempted.

This design adds observation only. It does not change speculative semantics,
Proposal-KV ownership, proposal capacity, sampling, accepted-prefix handling,
or the measured request path's CUDA synchronization behavior.

## Considered Approaches

### 1. External continuous samplers plus worker repeat boundaries

Run `nvidia-smi`, `vmstat`, `mpstat`, and `pidstat` as background processes.
Add wall-clock start and finish timestamps around each worker repeat so an
offline assembler can align telemetry samples with warmup and measured
intervals.

Advantages:

- no system query runs inside `engine.step()`;
- no new CUDA synchronization;
- preserves raw samples for independent recomputation;
- uses tools verified on the remote host;
- can distinguish clocks, throttling, thermal, utilization, and host-load
  hypotheses.

Cost:

- requires timestamp alignment and sampler lifecycle handling.

### 2. One-shot system queries before and after every repeat

This is simpler to align, but the query latency occurs directly between
repeats and can perturb cadence. Two snapshots also miss short throttling,
power, or contention events inside a multi-second repeat.

### 3. NVML/DCGM telemetry daemon

This can provide richer sampling and lower overhead, but introduces a new
Python or system dependency and a larger validation surface. It is not needed
for the first falsification campaign.

## Decision

Use approach 1. Preserve approach 3 only as a follow-up if `nvidia-smi`
sampling is incomplete or too coarse.

## Worker Timestamp Contract

`run_policy_campaign` receives an injectable `wall_clock_ns` dependency,
defaulting to `time.time_ns`.

Each warmup and measured result gains:

```json
{
  "campaign_interval": {
    "started_at_unix_ns": 0,
    "finished_at_unix_ns": 0
  }
}
```

The start timestamp is taken immediately before `run_batch_fn`; the finish
timestamp is taken immediately after it returns. The timestamps are outside
the existing request timing calculation and do not add
`torch.cuda.synchronize()`.

Validation requires:

- integer, non-boolean timestamps;
- positive values;
- finish strictly greater than start;
- non-overlapping, monotonically ordered intervals;
- existing repeat numbering and measured results unchanged.

## GPU Telemetry

The remote runner starts a line-buffered sampler before each worker process.
Every query batch is prefixed with `date +%s%N`; this Unix-epoch nanosecond
value is authoritative for interval alignment. The `nvidia-smi` timestamp is
retained only for auditing because the remote host reports local `+0800 CST`
time without an offset in each CSV row.

```text
loop:
  sampled_at_unix_ns = date +%s%N
  nvidia-smi
    --query-gpu=
      timestamp,
      index,
      uuid,
      pstate,
      clocks.current.sm,
      clocks.current.memory,
      power.draw,
      temperature.gpu,
      utilization.gpu,
      utilization.memory,
      memory.used,
      clocks_throttle_reasons.active
    --format=csv,noheader,nounits
    --id=3,4,6,7
  prefix every returned row with sampled_at_unix_ns
  sleep 0.2
```

These fields were verified on the remote A100 host with driver `535.261.03`.
The raw CSV is authoritative. Sampler startup and termination status are
retained separately.

No application or unrelated GPU process is terminated. In particular, the
existing GPU-7 `python3` service remains untouched.

## Host Telemetry

The runner retains concurrent raw output from:

- `vmstat -t 1`;
- `mpstat -P ALL 1`;
- `pidstat -u -r -d -h 1`.

The first campaign uses host telemetry as explanatory evidence, not as a hard
pass gate, because process inventory and locale can vary. GPU sample coverage
and worker timestamp integrity are hard gates.

## Artifact Assembly

A dedicated assembler consumes:

- target and learned worker JSON;
- target and learned GPU CSV;
- host telemetry logs;
- source-file hashes.

For every measured repeat and every selected GPU, it records:

- sample count;
- minimum, median, and maximum SM clock;
- minimum and maximum memory clock;
- minimum, median, and maximum power;
- minimum and maximum temperature;
- median and maximum GPU and memory utilization;
- distinct P-states;
- bitwise OR and distinct values of active throttle reasons;
- minimum and maximum memory usage.

The assembler retains raw file hashes and does not discard the original CSV
or host logs.

## Coverage and Classification

Each measured repeat must contain at least five GPU samples per selected GPU.
Failure to meet coverage produces `INVALID_TELEMETRY`, not a performance
classification.

The campaign remains `UNSTABLE` under the existing stationarity rules unless
all primary target and learned metrics pass. Telemetry adds attribution:

- `ENVIRONMENT_CORRELATED`: instability aligns with clock, throttle, thermal,
  power, utilization, or host-load changes;
- `RUNTIME_VARIANCE_SUSPECTED`: timing remains unstable while sampled GPU
  clocks, P-state, throttle reasons, and host load remain materially stable;
- `STABLE_BASELINE`: existing stationarity gates pass.

These labels do not promote an optimization by themselves.

## Campaign Order

The first telemetry campaign preserves the current target-then-learned order
to minimize code and workload changes. If it remains unstable without an
environment correlation, run a second source-identical campaign with reversed
policy order. Do not merge the two orders into one performance median.

## Verification

Focused tests cover:

- deterministic worker interval timestamps;
- invalid or overlapping intervals;
- GPU CSV parsing and field inventory;
- per-repeat/per-GPU sample alignment;
- minimum sample coverage;
- throttle-reason aggregation;
- source hash verification;
- classification boundaries;
- sampler cleanup on success, worker failure, timeout, and signal.

The final remote bundle must retain:

- executor and telemetry test logs;
- worker JSON;
- raw GPU and host telemetry;
- assembled diagnostic artifact;
- independent remote and local verifier receipts;
- before/after GPU process snapshots;
- source tarball and hashes;
- complete manifest.

## Claim Boundary

This campaign can establish whether observed instability correlates with
sampled system conditions. It cannot establish GPU kernel duration, causal
performance improvements, long-context performance, Proposal-KV offload
benefit, or Phase-1 promotion.
