# Autoregressive Draft Host Semantic Alignment Design

## Goal

Determine whether the learned-policy first-position slowdown that remains after
same-policy priming is associated with measurable host pressure during the
individual measured repeats.

This is a diagnostic control, not a runtime optimization. It must not change
the measured worker, speculative semantics, Proposal-KV behavior, or the
existing timing and GPU telemetry artifact schemas.

## Evidence That Motivates the Diagnostic

The source-bound primed campaigns used opposite policy orders:

```text
r5: target,learned
r6: learned,target
```

Same-policy priming moved the two target-policy results close together:

```text
target r5 first E2E:   3.811664153 s
target r6 second E2E:  3.932132042 s
cross-order delta:    +3.16%
```

The learned policy remained substantially slower when it ran first:

```text
learned r6 first E2E:   6.676585212 s
learned r5 second E2E:  5.157472217 s
cross-order delta:    -22.75%

learned TPOT delta:             -22.11%
learned proposal-forward delta: -15.09%
```

GPU clocks, P-state, throttle state, temperature, and sampled utilization did
not establish a GPU-environment cause. The existing `vmstat`, `mpstat`, and
`pidstat` files are hash-bound raw evidence only. They are not aligned to the
nanosecond campaign intervals stored in each worker repeat, so host contention
remains unexcluded.

The current classification remains:

```text
POSITION_EFFECT_REMAINS
TARGET_PRIMED_ORDER_EFFECT=NEAR_CONVERGED
LEARNED_PRIMED_ORDER_EFFECT=REMAINS
SPECIFIC_RUNTIME_ROOT_CAUSE=NOT_ESTABLISHED
HOST_CONTENTION=NOT_EXCLUDED
STABLE_PERFORMANCE_BASELINE=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Selected Approach

Add a lightweight Python `/proc` sampler that emits one JSON object every
`200 ms`. Each sample carries both:

```text
sampled_at_unix_ns
sampled_at_monotonic_ns
```

The Unix timestamp aligns samples to the existing worker
`campaign_interval.started_at_unix_ns` and
`campaign_interval.finished_at_unix_ns` fields. The monotonic timestamp proves
strict local sample order and detects wall-clock discontinuities.

Build a separate host-semantic artifact from:

- the target and learned measured-worker JSON files;
- one host JSONL file per policy;
- the existing timing artifact;
- the existing GPU telemetry artifact;
- source hashes for the sampler, assembler, verifier, worker, and timing
  producer.

The new artifact has its own independent verifier. The existing timing and GPU
telemetry artifacts remain unchanged and continue to be independently
verified.

## Rejected Alternatives

### Parse Existing `vmstat`, `mpstat`, and `pidstat`

This would permit retrospective analysis of r5 and r6, but the formats combine
second-level timestamps, locale-dependent headers, boot-relative counters, and
buffered output. `mpstat` and `pidstat` also require reconstructing the date
from a separate header. That is too fragile for repeat-level authority.

The old files remain retained and hash-bound as supporting evidence, but they
do not satisfy the semantic-alignment requirement.

### Add Host Reads to the Measured Worker

Reading `/proc` in the measured request path would make alignment direct, but
it would perturb the code path whose timing is under investigation and would
couple a diagnostic to the worker schema. The sampler must remain a separate
script-owned process.

### Run Learned A/A Before Host Alignment

An A/A process-boundary control can determine whether the first learned process
is consistently slower without policy interaction. It should be the next
diagnostic only if the host-semantic result is not associated or is
inconclusive. It does not replace measuring host conditions.

## Components

### `/proc` Host Sampler

Create:

```text
tools/autoregressive_draft_host_sampler.py
```

The sampler reads cumulative or instantaneous system-wide counters from:

```text
/proc/stat
/proc/loadavg
/proc/vmstat
/proc/meminfo
/proc/pressure/cpu
/proc/pressure/io
/proc/pressure/memory
```

It writes line-delimited JSON to stdout. The runner redirects stdout to:

```text
host-semantic/target-host.jsonl
host-semantic/learned-host.jsonl
```

and stderr to:

```text
host-semantic/target-host.stderr.log
host-semantic/learned-host.stderr.log
```

The CLI is:

```bash
python tools/autoregressive_draft_host_sampler.py \
  --interval-seconds 0.2
```

The sampler must:

- reject non-finite or non-positive intervals;
- emit the first complete sample immediately;
- sleep against a monotonic deadline rather than accumulating fixed-sleep
  drift;
- flush every JSON line;
- terminate cleanly on `SIGTERM` or `SIGINT`;
- never emit a partial sample;
- report a read or parse failure to stderr and exit nonzero;
- use only the Python standard library.

No per-process scan or process ranking is included. System-wide pressure is
the smallest sufficient diagnostic and avoids periodic traversal of
`/proc/<pid>`.

### Raw Sample Schema

Every row is a flat JSON object with:

```text
schema_version
sampled_at_unix_ns
sampled_at_monotonic_ns

cpu_user_ticks
cpu_nice_ticks
cpu_system_ticks
cpu_idle_ticks
cpu_iowait_ticks
cpu_irq_ticks
cpu_softirq_ticks
cpu_steal_ticks

procs_running
procs_blocked
context_switches_total
processes_forked_total

loadavg_1m
loadavg_5m
loadavg_15m

major_faults_total
page_in_kib_total
page_out_kib_total
swap_in_kib_total
swap_out_kib_total

memory_available_kib
memory_cached_kib
memory_dirty_kib
memory_writeback_kib

cpu_psi_some_total_us
cpu_psi_full_total_us
io_psi_some_total_us
io_psi_full_total_us
memory_psi_some_total_us
memory_psi_full_total_us
```

`page_in_kib_total` and `page_out_kib_total` are derived from `pgpgin` and
`pgpgout`, whose Linux `/proc/vmstat` unit is KiB. Swap counters are converted
from pages to KiB using `SC_PAGE_SIZE`.

PSI totals are used instead of `avg10`, `avg60`, and `avg300` because the
artifact needs interval-local deltas rather than moving averages with windows
larger than a repeat. If a kernel omits `full` CPU PSI, the sampler records
`cpu_psi_full_total_us` as `null`; all other required fields must be present.

All cumulative counters are non-negative integers. Memory values are
non-negative integers in KiB. Load averages are finite non-negative numbers.

### Runner Integration

Modify:

```text
tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh
```

The new sampler is enabled for this diagnostic runner without adding a new
mode flag. It starts in `start_samplers(policy)` after the GPU sampler and
before the legacy one-second host tools. Its PID is appended to the existing
script-owned `sampler_pids` array, so `stop_samplers` remains the only process
cleanup path.

The sampler starts only for the measured worker:

```text
prime(policy)
start samplers
measured(policy)
stop samplers
```

It must never run during same-policy priming. The measured worker remains:

```text
batch size:      4
warmup runs:     2
measured runs:   8
```

The runner must:

- package the sampler, assembler, verifier, and tests;
- compile and test them during remote preflight;
- create `host-semantic/`;
- assemble `host-semantic.json` after the existing timing and GPU artifacts;
- run the independent host verifier remotely;
- run the same verifier locally after download;
- retain separate remote/local receipts and exit-code files;
- include all files in the existing final manifest;
- print host assembler/verifier logs on failure.

No `torch.cuda.synchronize()` may be added to the measured request path.

### Host-Semantic Artifact

Create:

```text
tools/autoregressive_draft_host_semantic_diagnostic.py
```

The artifact schema version is `1`. Its top-level fields are:

```text
schema_version
status
classification
classification_reasons
exact_parity
policy_order
prime_each_policy
timing_artifact_sha256
gpu_telemetry_artifact_sha256
policies
thresholds
source_files
input_files
limitations
```

`status` is `PASS` only when all structural, coverage, exact-parity, and
recomputation checks succeed. A `PASS` status does not imply host association
or Phase-1 promotion.

The campaign-local `classification` is always:

```text
ALIGNED_CAMPAIGN
```

after all campaign-local checks pass. Cross-order host-pressure
classification exists only in the separate comparison artifact.

Every `input_files` entry contains a path relative to the campaign artifact
directory and its SHA-256. Absolute paths and `..` traversal are invalid.

For each policy, `policies[policy]` contains:

```text
worker_sha256
host_jsonl_sha256
sample_count
measured_runs
```

Each measured-repeat row contains:

```text
repeat
campaign_interval
host_sample_interval
sample_count
duration_seconds
metrics
timing
```

The host sample interval is bounded by the first sample at or before the repeat
start and the first sample at or after the repeat finish. Deltas are computed
between those two boundary samples. This avoids dropping the unsampled tail at
either end of the repeat.

The boundary requirements are:

- one sample no more than `400 ms` before or exactly at repeat start;
- one sample exactly at or no more than `400 ms` after repeat finish;
- at least two samples in the boundary interval;
- strictly increasing Unix and monotonic timestamps;
- no adjacent monotonic sample gap above `600 ms`;
- no cumulative counter decrease between the two boundary samples.

The `400 ms` edge allowance is twice the requested interval. The `600 ms`
maximum internal gap permits one delayed sample but not two consecutive missed
periods.

### Derived Repeat Metrics

For each measured repeat, derive:

```text
cpu_busy_fraction
cpu_system_fraction
cpu_iowait_fraction
cpu_steal_fraction

run_queue_mean
run_queue_max
blocked_processes_mean
blocked_processes_max
loadavg_1m_mean

context_switches_per_second
forks_per_second
major_faults_per_second
page_in_kib_per_second
page_out_kib_per_second
swap_in_kib_per_second
swap_out_kib_per_second

memory_available_kib_min
memory_dirty_kib_max
memory_writeback_kib_max

cpu_psi_some_fraction
cpu_psi_full_fraction
io_psi_some_fraction
io_psi_full_fraction
memory_psi_some_fraction
memory_psi_full_fraction
```

CPU fractions use deltas from the aggregate `cpu` counters. PSI fractions are
`delta_total_us / elapsed_us`. Rates use the boundary sample elapsed time.
Queue, load, and memory summaries use every sample whose Unix timestamp lies
inside the inclusive boundary interval.

The corresponding timing row contains:

```text
e2e_s
tpot_s
executor_proposal_forward_ms
```

These values are extracted with the same batch-median semantics already used
by `autoregressive_draft_b4_timing_diagnostic.py`.

### Primary Host Metrics

Classification uses these primary metrics:

```text
cpu_system_fraction
cpu_iowait_fraction
run_queue_mean
context_switches_per_second
major_faults_per_second
io_psi_some_fraction
memory_psi_some_fraction
memory_dirty_kib_max
memory_writeback_kib_max
```

Higher values always mean more host pressure. Other derived metrics remain
retained for diagnosis but do not satisfy the primary-metric count.

For a primary metric to be `worse` in learned-first:

```text
learned-first median > learned-second median
```

and both of these must hold:

```text
absolute difference > 1e-12
relative increase >= 10%
```

If the learned-second median is at most `1e-12`, a positive learned-first
median satisfies the relative condition.

### Correlation

Spearman rank correlation is computed independently for each primary host
metric against:

```text
learned e2e_s
learned executor_proposal_forward_ms
```

using all sixteen learned repeats from r7 and r8. Average ranks are used for
ties. A correlation is valid only when both vectors have nonzero rank variance.

A metric has expected-direction correlation when:

```text
rho >= 0.6
```

Negative correlation does not support host-pressure association because all
primary metrics are oriented so higher means worse.

## Cross-Campaign Comparison

One host-semantic artifact describes one campaign and proves per-repeat
alignment. Cross-order classification requires two verified campaign
artifacts:

```text
r7: target,learned, prime each policy
r8: learned,target, prime each policy
```

The diagnostic CLI accepts the optional pair:

```text
--comparison-artifact <other-host-semantic.json>
```

The pair must have:

- opposite valid policy orders;
- `prime_each_policy == true`;
- identical source hashes;
- eight measured repeats per policy;
- exact parity in both campaigns;
- distinct timing, GPU telemetry, worker, and host-input hashes.

The CLI writes a separate comparison file:

```text
experiments/autoregressive_draft/
  tp4-qwen3-b4-host-semantic-comparison-r7-r8-20260815.json
```

This avoids mutating either campaign-local artifact after its remote and local
verification. The comparison file lives in the common parent of both campaign
directories, so both campaign references remain relative without `..`
traversal.

The comparison schema version is `1`. Its top-level fields are:

```text
schema_version
status
classification
classification_reasons
campaign_artifacts
learned_position_effect
primary_metric_comparison
correlations
thresholds
source_identity
limitations
```

`campaign_artifacts` contains `learned_first` and `learned_second`. Each entry
stores the campaign artifact path relative to the comparison artifact
directory, its SHA-256, and the validated policy order. Paths must be relative
and cannot contain `..`.

`learned_position_effect` stores learned-first and learned-second medians for
E2E, TPOT, and proposal-forward plus the relative delta defined as:

```text
(learned_first - learned_second) / learned_second
```

`primary_metric_comparison` stores each primary metric's two medians, absolute
difference, relative increase, and `worse_in_learned_first` boolean.

`correlations` stores, for every primary metric and each timing metric, the
sample count, rank-variance flags, and either a finite Spearman `rho` or
`null`.

`source_identity` stores the exact common source-hash mapping and proves both
campaign artifacts were produced by identical diagnostic sources.

## Classification Rules

The learned position effect is present when learned-first is at least `10%`
slower than learned-second in the median `e2e_s`. The comparison records TPOT
and proposal-forward deltas as supporting values, but E2E is the required
position-effect gate.

### Host Pressure Associated

Classify:

```text
HOST_PRESSURE_ASSOCIATED
```

only when all conditions hold:

1. learned-first median E2E is at least `10%` slower;
2. at least two primary host metrics are worse in learned-first by at least
   `10%`;
3. at least two primary host metrics have valid expected-direction
   `rho >= 0.6`;
4. among the correlated metrics, at least one correlates with learned E2E and
   at least one correlates with learned proposal-forward;
5. both campaign-local artifacts pass all alignment and source-identity
   checks.

This classification means association only. It does not prove that host
pressure caused the slowdown or identify a process responsible for it.

### Host Pressure Not Supported

Classify:

```text
HOST_PRESSURE_NOT_SUPPORTED
```

when:

1. the learned position effect remains at least `10%`;
2. both campaign-local artifacts pass all alignment checks;
3. fewer than two primary host metrics are worse by at least `10%`, or fewer
   than two primary metrics have expected-direction `rho >= 0.6`.

The next experiment is a primed learned/learned process-boundary A/A control.

### Host Alignment Inconclusive

Classify:

```text
HOST_ALIGNMENT_INCONCLUSIVE
```

when any required alignment, coverage, source-identity, exact-parity,
opposite-order, or sample-continuity condition fails.

Also use this classification when the learned E2E position effect is below
`10%`, because there is no sufficiently large effect to explain in this pair.
Zero-variance primary metrics produce invalid correlations and therefore do
not satisfy the association count; they do not by themselves invalidate an
otherwise complete campaign pair.

Do not select a runtime optimization from an inconclusive result.

## Independent Verification

Create:

```text
tools/verify_autoregressive_draft_host_semantic_diagnostic.py
```

The verifier must not trust stored summaries or classifications. It:

1. loads the artifact and every referenced input;
2. validates raw JSONL samples;
3. recomputes repeat boundaries, deltas, rates, fractions, and timing values;
4. verifies exact parity and policy metadata;
5. verifies every source and input SHA-256;
6. reconstructs the complete expected artifact;
7. requires exact structural equality;
8. emits a receipt with verified source/input counts and repeat coverage.

For a comparison artifact, the verifier independently loads both campaign
artifacts, verifies both against their own bundles, recomputes all medians,
relative deltas, ranks, correlations, reasons, and final classification, and
requires exact equality.

## Campaign Matrix

Run two new source-bound campaigns:

```text
r7:
  policy order: target,learned
  prime each policy: enabled

r8:
  policy order: learned,target
  prime each policy: enabled
```

Keep:

```text
GPUs:                3,4,6,7
batch size:          4
warmup repeats:      2
measured repeats:    8
proposal length:     4
temperature:         0
target model:        target-qwen3-1.7b
learned draft model: draft
```

Do not merge target and learned timing medians. The comparison uses the
learned repeats for the residual position-effect diagnosis and retains target
host summaries as a control.

## Safety and Correctness

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Use `sitian@10.232.195.203`.
- Use `/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815`.
- Do not write new experiment artifacts under `/data00`.
- Preserve `MAX_PROPOSAL_TOKENS=4`.
- Preserve temperature zero, exact greedy parity, and accepted-prefix
  semantics.
- Preserve workload-derived Proposal-KV capacity.
- Do not add `torch.cuda.synchronize()` to the measured request path.
- Do not terminate unrelated GPU processes.
- Preserve the existing GPU-7 `python3` service.
- Do not treat synthetic KV movement as real movement.
- Do not stage, commit, push, stash, reset, clean, or switch branches or
  worktrees.

## Test Requirements

Focused unit and source-contract tests must prove:

- every `/proc` parser accepts representative kernel text and rejects missing,
  duplicate, malformed, negative, or non-finite fields;
- the sampler emits complete schema-v1 JSON objects with increasing timestamps;
- interval validation rejects zero, negative, NaN, and infinity;
- signal handling stops without writing partial JSON;
- repeat alignment accepts valid edge samples and rejects missing boundaries,
  excessive gaps, timestamp regression, counter regression, and fewer than two
  samples;
- CPU, rate, PSI, queue, load, and memory calculations match hand-computed
  fixtures;
- timing extraction preserves existing batch-median semantics;
- Spearman tie handling and zero-variance behavior are deterministic;
- all three classifications are covered, including every association
  prerequisite;
- artifact and comparison recomputation detect tampering;
- source and input hash verification detects missing or modified files;
- runner ordering keeps the host sampler after prime and before measured work;
- runner cleanup targets only script-owned sampler PIDs;
- remote and local independent verifier calls are present;
- existing timing and GPU telemetry assembler/verifier inputs are unchanged;
- no `torch.cuda.synchronize` text is introduced.

The complete local gate is:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_performance_gate.py \
  tools/test_autoregressive_draft_instability_telemetry.py \
  tools/test_autoregressive_draft_host_sampler.py \
  tools/test_autoregressive_draft_host_semantic_diagnostic.py

python3 -m py_compile \
  tinyvllm/engine/autoregressive_draft_executor.py \
  tools/autoregressive_draft_performance_worker.py \
  tools/autoregressive_draft_b4_timing_diagnostic.py \
  tools/autoregressive_draft_instability_telemetry.py \
  tools/verify_autoregressive_draft_instability_telemetry.py \
  tools/autoregressive_draft_host_sampler.py \
  tools/autoregressive_draft_host_semantic_diagnostic.py \
  tools/verify_autoregressive_draft_host_semantic_diagnostic.py

bash -n \
  tools/run_autoregressive_draft_b4_instability_telemetry_remote.sh
```

## Interpretation Boundary

This diagnostic can establish whether repeat-aligned system-wide host pressure
is associated with the learned first-position slowdown. It cannot establish:

- causality;
- the responsible host process;
- a CUDA, allocator, page-cache, JIT, or collective root cause;
- stable long-context performance;
- Proposal-KV offload benefit;
- Phase-1 promotion readiness.

If the result is `HOST_PRESSURE_ASSOCIATED`, the next control must isolate the
responsible host resource or process class. If it is
`HOST_PRESSURE_NOT_SUPPORTED`, the next control is primed learned/learned A/A.
If it is `HOST_ALIGNMENT_INCONCLUSIVE`, repair the evidence gap and rerun the
same two-order campaign before changing the runtime.
