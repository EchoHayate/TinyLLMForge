# Autoregressive Draft Command Timeline and Sync-Debt Design

**Date:** 2026-08-18

**Status:** Approved for implementation

## Goal

Locate the real source of the TP4/B4/Q4 graph-versus-eager TTFT and E2E
dispersion that remains after proposal-forward improves.

The first slice adds default-off observation only. It must decompose each
measured engine step into:

```text
scheduler
  -> command publication
  -> worker queue wait
  -> worker CUDA execution
  -> worker non-CUDA residual
  -> acknowledged-command wait
  -> scheduler prepare/commit/postprocess
```

The diagnostic must distinguish work executed inside the current phase from
asynchronous worker debt inherited from an earlier non-acknowledged command.
It must do so without inserting a completion fence or a new CUDA
synchronization into the measured request path.

Only a fresh source-bound result that passes exact identity, exact parity,
timeline coverage, timing conservation, stationarity, dual verification, and
manifest verification may select the next runtime optimization. This
specification does not preselect that optimization.

## Motivation and Evidence Boundary

The immutable schema-v2 campaign:

```text
20260817-steady-state-schema-v2-tp4-b4-q4-r3
```

established exact TP4/B4/Q4 graph replay and eager/graph correctness parity,
but classified controlled performance as:

```text
NO_GO_PERFORMANCE
```

Proposal-forward improved in all eight measured pairs, with a median
graph-minus-eager delta of approximately `-350.421 ms`. That improvement did
not explain the request-level variance:

```text
correlation(proposal-forward delta, E2E delta):
  approximately 0.19

correlation(TTFT delta, E2E delta):
  approximately 0.84

correlation(commit_metadata_ms delta, E2E delta):
  approximately 0.75
```

The eager path also contained isolated final-step `commit_metadata_ms` spikes
of approximately `598 ms`, `740 ms`, and `729 ms`, while graph
`prompt_bootstrap` contained shared-rank stalls of approximately
`218-474 ms`.

These are localization clues, not proof of a worker queue or synchronization
cause. Existing host/GPU telemetry did not establish a stable environmental
cause, and the completed learned A/A result was inconclusive. The next gate
therefore measures the missing command timeline while reusing the existing
paired-stability admission model and source-bound evidence chain.

## Fixed Scope

The diagnostic scope is exactly:

```text
draft model family:       independent Qwen3 learned drafter
tensor parallel size:     4
batch size:               4
proposal limit:           4
prompt tokens:            256 per request
requested output tokens:  16 per request
sampling:                 greedy, temperature 0
Proposal-KV allocator:    dense direct
Proposal-KV offload:      disabled
graph shape policy:       exact only
```

No padding, shape rounding, mixed graph identity, alternate allocator, KV
offload, stochastic sampling, or changed request set is admitted.

The authoritative checkout is:

```text
/Users/bytedance/Desktop/TinyLLMForge
  -> /Users/bytedance/dev/TinyLLMForge
```

The retired checkout
`/Users/bytedance/dev/TinyLLMForge-adaptive-ngram` must not be read as
execution authority, modified, staged, committed, or used to package a
remote run.

## Selected Approach

Use a new command-timeline diagnostic layered on the existing exact CUDA
Graph gate, paired-stability admission rules, host/GPU telemetry, independent
verifier pattern, and checksum manifest.

The schedule is precommitted:

```text
block 0: eager -> graph
block 1: graph -> eager
block 2: graph -> eager
block 3: eager -> graph
```

Each of the eight epochs is an isolated worker process. Each process runs:

```text
one in-process warmup batch
five measured batches
```

The graph warmup must perform the sole capture. Every measured graph batch
must increase replay counters while capture counters and retained resources
remain unchanged. The eager warmup uses the same workload but must not create
or replay an autoregressive-draft graph.

This produces:

```text
4 balanced blocks
8 isolated worker processes
40 measured batches
160 measured requests
```

No measured repeat may be deleted after execution. One invalid epoch makes
the bundle unsuitable for boundary localization.

## Rejected Approaches

### Force acknowledgements on all model-runner commands

This would make the timeline easier to interpret, but it changes scheduling
semantics and may merely move the same wait earlier. It could reduce overlap
and throughput while disguising queue debt as an intentional fence.

### Pipeline or merge proposal lifecycle acknowledgements now

The current evidence does not identify which lifecycle phase owns the
variance. Combining transactional phases before attribution would enlarge
the rollback and TP-convergence risk surface.

### Add more warmups or discard the first measured batch

That may hide a process-boundary effect without changing production runtime.
Warmup policy is part of the fixed identity, not an optimization result.

### Repeat generic host-pressure telemetry

The existing repeat-aligned host/GPU campaigns are retained and reusable.
The missing evidence is the command and step envelope, not another broad
system sampler.

## Observation Architecture

### Default-off profiler

Add a default-off command timeline profiler owned by `LLMEngine` and each
`ModelRunner`.

When disabled:

- command envelopes and acknowledgement behavior remain unchanged;
- no timeline rows are retained;
- no CUDA Events are created;
- no extra synchronization occurs; and
- existing worker and gate schemas remain valid.

When enabled by the dedicated diagnostic worker:

- every command receives a bounded trace identity;
- rank 0 records dispatch, local execution, and acknowledgement collection;
- worker ranks record command receive, method execution, and ack-send
  boundaries;
- engine step phases record scheduler and postprocess boundaries; and
- existing deferred CUDA Event profiling supplies CUDA duration.

The profiler is diagnostic-only and must not become a production-default
metrics dependency.

### Clock contract

All processes record:

```text
time.monotonic_ns()
time.time_ns()
/proc/sys/kernel/random/boot_id
time.get_clock_info("monotonic")
```

Timeline arithmetic uses monotonic nanoseconds only. Unix time is retained
for alignment with existing external telemetry. Every rank must report the
same boot ID and compatible monotonic clock metadata. A clock identity
mismatch invalidates the timeline.

No cross-host timestamp arithmetic is permitted.

### Command envelope identity

The traced command identity contains:

```text
command_id
method_name
requires_ack
engine_step_id
repeat_index
request_set_sha256
batch_kind
speculative_selected_sequence_ids_sha256
dispatch_started_monotonic_ns
dispatch_published_monotonic_ns
```

The trace fields are metadata only. They must not alter the method arguments,
return values, error propagation, or acknowledgement requirements.

`command_id` remains the ordering authority. Each rank must observe the same
strictly increasing command inventory. Missing, duplicated, reordered, or
unknown command IDs invalidate the epoch.

### Rank-local command rows

Rank 0 records:

```text
dispatch_started_monotonic_ns
dispatch_published_monotonic_ns
local_method_started_monotonic_ns
local_method_finished_monotonic_ns
ack_wait_started_monotonic_ns
ack_wait_finished_monotonic_ns
```

Worker ranks record:

```text
event_woken_monotonic_ns
envelope_read_monotonic_ns
method_started_monotonic_ns
method_finished_monotonic_ns
ack_send_started_monotonic_ns
ack_send_finished_monotonic_ns
```

Ack timestamps are `null` for non-acknowledged commands. Nullability is part
of the schema and may not be inferred from missing keys.

Every row also records status and a bounded error type. Error detail remains
subject to the existing acknowledgement size limit.

### CUDA execution rows

Reuse `DecodeInternalProfiler` CUDA Events rather than adding wall-path
`torch.cuda.synchronize()` calls.

CUDA Events are recorded around the existing profiled model-runner step and
collective boundaries. They are resolved only after the measured batch's
existing terminal synchronization. Finalization may not add an earlier
request-path synchronization.

For every rank and step, retain:

```text
worker_method_wall_ns
cuda_ns
non_cuda_upper_bound_ns
collective_cuda_ns
collective_wall_ns
```

The CUDA duration is rank-local. The bundle uses the maximum rank duration
for critical-path attribution and preserves all four rank rows.

### Engine step envelope

`LLMEngine.step()` records these non-overlapping host spans:

```text
scheduler_schedule
partition_and_step_setup
ordinary_or_first_target_dispatch
speculative_prepare
scheduler_prepare_postprocess
proposal_kv_prepare_commit
proposal_lifecycle_finalize_prepare
scheduler_commit_postprocess
proposal_lifecycle_finalize_commit
side_state_seal
residency_precommit_or_seal
ordinary_scheduler_postprocess
step_residual
```

Only spans actually executed in a step are present, with an explicit
`executed` boolean and zero duration for skipped optional phases.

The existing aggregate keys such as `first_target_batch_ms`,
`tail_batch_ms`, and `commit_metadata_ms` remain unchanged. The new envelope
must reconcile those aggregates to named subspans instead of redefining
their meaning.

## Sync-Debt Decomposition

For every rank and command:

```text
worker_queue_wait_ns =
  method_started_monotonic_ns
  - dispatch_published_monotonic_ns

queued_behind_prior_command_ns =
  max(
    0,
    previous_method_finished_monotonic_ns
    - dispatch_published_monotonic_ns,
  )

worker_ready_delay_ns =
  worker_queue_wait_ns
  - queued_behind_prior_command_ns

worker_non_cuda_upper_bound_ns =
  worker_method_wall_ns
  - cuda_ns
```

For acknowledged commands on rank 0:

```text
ack_wait_ns =
  ack_wait_finished_monotonic_ns
  - ack_wait_started_monotonic_ns

post_local_ack_wait_ns =
  ack_wait_finished_monotonic_ns
  - max(
      ack_wait_started_monotonic_ns,
      local_method_finished_monotonic_ns,
    )
```

The diagnostic attributes an inherited debt only when a non-acknowledged
command is still executing on at least one worker after rank 0 has completed
the same command and before a later phase begins waiting for convergence.

It must retain:

- the producing command ID and method;
- the consuming phase or acknowledged command ID;
- affected ranks;
- overlap duration per rank;
- maximum-rank critical overlap; and
- whether the debt lands inside `prompt_bootstrap`,
  `commit_metadata_ms`, or another named step span.

An ack wait is not automatically classified as debt. It may represent the
acknowledged command's own CUDA work. The classification requires overlap
with an earlier command or a conservation residual that isolates the wait.

## Timing Conservation

For each command:

```text
worker_method_wall_ns
  ~= cuda_ns + worker_non_cuda_upper_bound_ns
```

For each engine step:

```text
step_wall_ns
  ~= scheduler_ns
   + rank0_local_command_critical_path_ns
   + acknowledged_wait_ns
   + scheduler_postprocess_ns
   + explicit_residual_ns
```

Overlapping worker execution is represented as overlap, not double-counted
as serial time.

Admission requires:

```text
absolute conservation residual <= 2 ms
or
relative conservation residual <= 1% of step wall time
```

The larger tolerance applies. Negative durations, impossible nesting, or
overlap greater than either containing interval invalidate the repeat.

## Exact Graph/Eager Identity Binding

Every epoch binds the following worker identity:

```text
source commit and source tree SHA-256
target checkpoint identity
draft checkpoint identity
tokenizer identity
prompt token rows and prompt SHA-256
request order
requested output lengths
TP world size and rank set
GPU UUID set
batch size
temperature
proposal limit
Proposal-KV capacity and allocator mode
Proposal-KV offload flag
CUDA Graph mode
graph identity SHA-256 per rank
warmup and measured repeat counts
command timeline schema version
```

Graph admission additionally requires, on every rank:

```text
successful captures after warmup = 1
capture attempts unchanged across measured repeats
successful captures unchanged across measured repeats
replay count increases in every measured repeat
ready graph entries = 1 and unchanged
retained static/reserved/capture-time resources unchanged
quarantines = 0
pre-replay fallbacks = 0
```

Eager admission requires:

```text
successful captures = 0
replays = 0
ready graph entries = 0
```

The logical proposal rows may remain bounded ragged evidence exactly as in
schema v2, but the execution identity remains exact TP4/B4/Q4. No artifact
may describe a logical short final row as a different graph shape.

## Correctness and Transaction Gates

Every eager/graph block and every repeat must preserve:

- exact target token IDs;
- exact logical proposal token IDs;
- exact accepted-prefix counts;
- exact accepted token IDs;
- exact transaction digest;
- exact acceptance totals and rate;
- Proposal-KV accepted-prefix commit;
- Proposal-KV rejected-suffix rollback;
- zero active Proposal-KV transactions after the batch;
- TP failure convergence; and
- replay-started fail-closed behavior.

Timing evidence is discarded if any correctness or transaction gate fails.

## Paired-Stability Admission

Reuse the paired-stability five-repeat epoch rules:

```text
MAD / median <= 0.10
half_drift <= 0.15
```

The halves are repeats `0,1` and `3,4`; repeat `2` contributes to the epoch
median and MAD only.

Admission applies independently to:

```text
E2E
TTFT
TPOT
proposal_forward
worker_queue_wait
queued_behind_prior_command
CUDA critical path
ack wait
scheduler plus postprocess
step conservation residual
```

The bundle must also pass the existing position-balance and sequence
interaction checks. A mode effect may be reported only when all eight epochs
are admitted.

## Boundary Classification

The canonical diagnostic has four fail-closed classifications:

```text
INVALID_IDENTITY_OR_CORRECTNESS
TIMELINE_INCOMPLETE_OR_NONCONSERVING
PAIRED_PROTOCOL_UNSTABLE
BOUNDARY_LOCALIZED
```

`BOUNDARY_LOCALIZED` requires:

1. all identity, correctness, transaction, timeline, and stationarity gates
   pass;
2. one named boundary explains at least `60%` of the absolute paired E2E
   delta in at least three of four blocks;
3. the same boundary has the same delta sign in at least three of four
   blocks;
4. the median absolute unexplained residual is at most `10%` of median E2E;
5. label and position effects do not reverse the boundary conclusion; and
6. remote and local independent verifier receipts are byte-identical after
   normalizing receipt paths.

The named boundary is one of:

```text
worker_queue_debt
worker_cuda_execution
ack_wait
scheduler_postprocess
mixed_or_unresolved
```

If no single boundary satisfies the thresholds, the classification remains
`PAIRED_PROTOCOL_UNSTABLE` with
`stable_but_unlocalized=true`; it does not authorize a runtime change.

Every result sets:

```text
runtime_optimization_authorized=false
performance_improvement_established=false
phase_1_complete=false
promotion_ready=false
```

## Artifact and Verification Design

Create a new artifact family rather than changing or reinterpreting the
immutable schema-v2 `r3` payload.

The bundle contains:

```text
result.json
command-timeline.json
workers/
telemetry/
source/
source.patch
source_manifest.json
verify.remote.json
verify.local.json
manifest.sha256
```

`workers/` retains all eight raw worker JSON files. `telemetry/` retains the
existing raw GPU and host telemetry per epoch. `source/` is the frozen
source tree used by the remote workload.

### Sitian-only storage override

All generated archives, bundles, raw outputs, telemetry, logs, receipts,
manifests, caches, validation roots, and review artifacts live under:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818
```

The runner uses two immutable remote destinations for a fresh tag:

```text
runs/<run-tag>
controller-verification/<run-tag>
```

The first is the execution bundle. The second is a new no-overwrite copy used
for the independent controller verification. No source archive, downloaded
bundle, validation copy, log, receipt, or cache may be created on the local
machine, local `/tmp`, local `/private/tmp`, remote `/`, or remote `/tmp`.
The historical `verify.command-timeline.local.*` names remain stable detached
attestation filenames, but `local` denotes the second controller verification
invocation, not a physical local-machine artifact directory.

### Canonical assembler

The assembler:

- validates the fixed schedule and all eight epoch identities;
- validates exact eager/graph and workload identity;
- joins command, rank, engine-step, CUDA Event, request timing, and telemetry
  rows;
- computes queue debt, CUDA, ack, scheduler/postprocess, and residual
  components;
- performs timing conservation and stationarity admission;
- computes block-local and aggregate paired effects; and
- emits deterministic canonical JSON.

### Independent verifier

The verifier must not trust assembler summaries. It independently:

- reloads every raw worker and telemetry input;
- verifies every input and source SHA-256;
- validates safe relative paths and complete inventory;
- recomputes exact identity and parity;
- recomputes every timeline join and duration;
- recomputes conservation, admission, paired effects, and classification;
- compares the full canonical structure; and
- verifies the final checksum manifest.

The same verifier runs first against the frozen source inside the primary
remote bundle and then against the current source snapshot from the immutable
remote controller copy. Both receipts must agree on all semantic fields after
removing only `verified_at_utc`, `verification_location`, and `artifact_path`.

### Manifest

The manifest is generated only after canonical assembly and pre-manifest
verification succeed. It covers every authoritative regular file except the
manifest itself and the two final verifier receipts/logs whose creation
follows it.

The final completion audit records a fresh:

```text
shasum -a 256 -c manifest.sha256
```

## Host/GPU Telemetry Reuse

Reuse the existing external samplers and repeat-alignment logic:

- GPU clock, utilization, power, temperature, P-state, throttle reasons, and
  memory;
- host CPU, run queue, faults, IO, memory, and PSI counters; and
- campaign Unix-nanosecond interval boundaries.

Telemetry remains outside the measured request path. Samplers are
runner-owned and only runner-owned sampler processes may be stopped.

Environmental instability can invalidate an epoch, but environmental
correlation does not replace command-timeline conservation.

## Testing Strategy

Local implementation follows RED -> GREEN.

Focused tests must cover:

- command envelope trace identity validation;
- disabled-mode schema and behavior compatibility;
- strict command ordering on all ranks;
- worker queue and inherited-debt arithmetic;
- acknowledged and non-acknowledged command rows;
- rank failure, timeout, malformed ack, and collector poisoning;
- deferred CUDA Event finalization without a new request-path synchronize;
- engine step span nesting and conservation;
- graph/eager exact identity and graph counter rules;
- ragged logical rows without graph identity relaxation;
- exact token, acceptance, and transaction parity;
- five-repeat stationarity boundaries;
- block schedule and position balance;
- localization threshold boundaries;
- tampered raw input, source, timeline, and manifest rejection;
- primary/controller-copy verifier semantic equivalence; and
- runner safety, ownership, Kerberos TTL, GPU cleanliness, and immutable tag
  handling.

The expanded local suite must include existing command-ack, decode-profiler,
speculative runtime, Proposal-KV, CUDA Graph gate, paired-stability,
telemetry, and verifier tests.

## Execution and Authorization Boundary

Writing this spec and its implementation plan authorizes local source and
test changes only.

It does not authorize:

- SSH or remote writes;
- GPU, CUDA, NCCL, or checkpoint execution;
- launching the existing paired-stability runner;
- launching the new command-timeline gate;
- killing or pausing any unrelated process; or
- implementing a runtime optimization before localization.

The fresh remote bundle requires a separate explicit user authorization after
the local implementation, verifier, runner, and preflight tests are green.

## Optimization Decision Gate

After a verified `BOUNDARY_LOCALIZED` result, create and approve a separate
runtime optimization design:

- `worker_queue_debt`: consider a targeted completion boundary or command
  scheduling change at the producing/consuming command pair only;
- `worker_cuda_execution`: optimize the named CUDA/collective region without
  changing transaction semantics;
- `ack_wait`: reduce or batch only the proven lifecycle wait while preserving
  rollback and TP convergence;
- `scheduler_postprocess`: optimize the named host span without moving work
  across semantic commit boundaries;
- `mixed_or_unresolved`: make no runtime change and refine instrumentation.

Every optimization must preserve the fixed identity and correctness gates and
must pass a new never-before-used source-bound controlled performance tag.
Warmup changes or discarded measurements alone do not count as runtime
performance improvements.

## Completion Criteria for This Diagnostic Slice

The diagnostic slice is complete only when all of the following exist and are
verified:

1. the approved written spec and implementation plan are committed and pushed;
2. default-off command and step timeline instrumentation is implemented;
3. focused RED and GREEN evidence is retained;
4. the expanded local test matrix and `git diff --check` pass;
5. the source-bound runner, assembler, independent verifier, and manifest
   contracts pass local fail-closed tests;
6. a separately authorized fresh remote bundle completes all eight epochs;
7. primary and controller-copy verifier receipts agree;
8. the manifest passes a fresh checksum verification;
9. a completion audit maps every requirement to retained evidence; and
10. only then is a boundary-specific optimization proposed.
