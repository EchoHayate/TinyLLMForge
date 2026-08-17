# Qwen3.5 TP4 Decode-Internal Profile Design

## Goal

Identify whether the small and variable decode slowdown observed after exact
hybrid-prefix restore comes from the first decode step, steady-state decode,
TP collectives, CUDA kernels, or host-side and synchronization waiting.

This is a diagnostic follow-up to the successful 64-token r609 and 8-token
r611 profiles. It does not further subdivide restore work and does not change
the canonical benchmark authority.

## Questions

The experiment must answer:

1. Does `exact_restore` change the first decode-step latency relative to
   `recompute`?
2. Does it change the distribution of steady-state per-token latency?
3. Is any difference accompanied by a change in TP collective count or CUDA
   duration?
4. Which CUDA kernels account for the largest policy difference in one
   representative paired run?
5. Is the unexplained part of step wall time large enough to implicate CPU
   orchestration or synchronization waiting?

The experiment is not required to identify a unique root cause when the
observed differences are within shared-GPU noise. An explicit inconclusive
classification is valid.

## Selected Approach

Use a two-level profile:

1. Collect lightweight, structured decode-step and collective measurements
   for all five measured recompute/exact pairs.
2. Capture one additional representative recompute/exact pair with NVIDIA
   Nsight Systems for kernel-level timeline evidence.

This approach is preferred over either extreme:

- Profiling only one pair with Nsight Systems gives rich detail but cannot
  establish repeatability.
- Running Nsight Systems over every pair increases runtime, artifact size,
  and profiler perturbation without improving the primary paired statistics.
- PyTorch profiler alone is easier to integrate but duplicates part of the
  CUDA Event and Nsight evidence while adding a larger Python-side collection
  surface.

## Workload and Resource Policy

The structured profile uses the existing eight-token `w2_long_reuse`
diagnostic workload:

- shared prefix: 3,840 tokens;
- continuation suffix: 64 tokens;
- continuations per case: 4;
- generated tokens: 8;
- policies: `recompute` and `exact_restore`;
- one paired warmup and five paired measured repetitions;
- output-token parity required for all 20 measured continuation pairs.

The resource contract remains:

- fixed GPUs `2,4,5,6`;
- at least 25 GiB free per GPU;
- at most 10 percent utilization per GPU;
- unrelated low-utilization processes are allowed;
- every result is marked shared and non-exclusive;
- no dummy reservation and no unrelated-process termination;
- a fresh attempt tag is used for every remote run;
- cleanup is attempt-scoped.

The entry and worker-entry guards must both pass before measured execution.

## Instrumentation Architecture

### Step Boundary

Instrument the existing decode call chain:

```text
LLMEngine.step()
  -> ModelRunner.run()
  -> ModelRunner.run_model()
```

Each rank records a row for every model-runner step. A row contains:

- rank;
- monotonically increasing local step index;
- batch kind;
- prefill/decode classification;
- active sequence count;
- request identities or a stable request-set digest;
- CPU wall start and end timestamps;
- CUDA Event start and end timestamps around the model-runner execution;
- collective count and cumulative collective CUDA duration;
- execution dispatch metadata when available, including eager versus graph.

The profile records prefill rows only for sequence alignment and rejects them
from decode summaries. Decode ordinal zero is the first decode step after
prefill for a request set. Later decode ordinals are steady-state steps.

### CUDA Timing

CUDA Events are recorded on the active PyTorch stream around:

- the complete `ModelRunner.run()` GPU-bearing region;
- each instrumented TP collective.

Event elapsed time is resolved only after the measured case has completed.
The timed hot path must not call `torch.cuda.synchronize()` once per event.
One bounded synchronization during profile finalization is allowed.

`step_cuda_ns` therefore measures elapsed device-stream time between the
recorded step events. It is not the sum of all kernel durations across
streams and must not be described as GPU occupancy.

### TP Collective Timing

Profile the actual TinyLLMForge collective call sites used by the model:

- row-parallel linear `dist.all_reduce`;
- vocabulary-parallel embedding `dist.all_reduce`;
- any output-head collective reached by the measured decode path.

The wrapper records operation kind, tensor shape, dtype, rank, local step
index, CPU wall interval, and CUDA Event duration. It preserves the original
collective arguments and return value.

Instrumentation must be opt-in and process-local. It must be installed after
distributed initialization, removed during finalization, and remain inactive
for all non-profile runs. It must not globally monkey-patch an unrelated
process.

### Host and Synchronization Upper Bound

For each rank and step:

```text
non_cuda_upper_bound_ns = max(0, step_wall_ns - step_cuda_ns)
```

This value combines CPU preparation, Python orchestration, scheduler or RPC
delay visible inside the measured boundary, CUDA launch gaps, and possible
synchronization waiting. It is only an upper bound. It must never be labeled
as exact CPU time or exact synchronization time.

Cross-rank imbalance is reported from the maximum-minus-minimum rank wall and
CUDA durations for aligned steps. Because TP progress is constrained by the
slowest rank, paired policy summaries use the maximum rank duration for each
aligned step as the primary step metric.

### Nsight Systems Capture

Run one extra diagnostic recompute/exact pair under:

```text
/usr/local/bin/nsys
```

The capture uses CUDA, NVTX, OS runtime, and NCCL tracing when available. The
profiled code emits NVTX ranges for:

- policy and case identity;
- prefill;
- first decode step;
- steady-state decode steps;
- TP collective call sites.

The Nsight pair is selected after the five structured pairs complete. Choose
the measured pair whose exact/recompute decode ratio is closest to the median
paired ratio. The extra capture replays that pair's configuration using a new
diagnostic repetition identity; it does not reuse the original process.

Nsight results are kernel-level supporting evidence, not part of the primary
latency medians. If Nsight or NCCL tracing is unavailable, the structured
five-pair profile remains valid and the missing kernel-level question is
reported explicitly rather than inferred.

## Artifact Contract

Do not modify the canonical workload manifest, case matrix, case-row schema,
or existing `profile.json` schema.

Each profiled worker writes a separate:

```text
decode_profile.json
```

The file contains:

```json
{
  "schema_version": 1,
  "variant": "decode_internal",
  "resource_policy": "shared-low-utilization",
  "exclusive": false,
  "policy": "recompute",
  "generated_tokens": 8,
  "ranks": [],
  "steps": [],
  "collectives": []
}
```

The exact schema is defined by validation code and tests. It must include
source-tree identity, workload identity, case identity, repetition identity,
rank inventory, units, profiler availability, and finalization status.

The attempt root also contains:

- `decode_profile_summary.json` for structured five-pair aggregation;
- an `nsys/` directory for the representative pair's reports and exported
  statistics;
- resource guard, worker receipt, and cleanup evidence using the existing
  attempt conventions.

Existing r609 and r611 artifacts remain immutable.

## Alignment and Validation

The aggregator rejects a case unless:

- all four ranks are present;
- every rank reports the same ordered decode-step inventory;
- request-set identity and decode ordinal agree across ranks;
- every event duration is non-negative and finite;
- collective events map to a known decode step;
- the generated-token count is eight;
- policy, case, repetition, workload, and source identities match;
- profile finalization completed;
- the benchmark case itself passed;
- recompute/exact output tokens match.

Warmup rows are retained for debugging but excluded from measured summaries.
The Nsight pair is retained separately and excluded from the five-pair
structured aggregates.

## Summary Metrics

Report recompute, exact restore, paired ratios, and per-rank dispersion for:

- first decode-step wall time;
- first decode-step CUDA time;
- first decode-step collective CUDA time;
- steady-state step wall-time median and p90;
- steady-state CUDA-time median and p90;
- steady-state collective count and CUDA time;
- non-CUDA upper bound;
- cross-rank wall and CUDA imbalance.

Also report:

- all five paired first-step ratios;
- all five paired steady-state ratios;
- direction agreement between paired median and ratio of policy medians;
- whether the result is consistent across at least four of five pairs;
- top kernel and NCCL duration differences from the representative Nsight
  capture;
- profiler overhead from a separately labeled unprofiled versus structured
  profile smoke comparison.

The aggregator must distinguish:

- `first_step_regression`;
- `steady_state_regression`;
- `collective_regression`;
- `non_cuda_or_sync_upper_bound_regression`;
- `mixed_or_inconclusive`;
- `no_material_decode_regression`.

Classification thresholds are fixed in the implementation plan before the
remote run. They must include both a relative threshold and an absolute
nanosecond floor so tiny timing differences are not promoted to a cause.

## Error Handling and Cleanup

- A profile validation failure fails only the new diagnostic attempt and
  preserves its evidence.
- Partial profiles are never aggregated as complete pairs.
- CUDA Event resolution failures identify the rank, step, and event type.
- Profiler setup and teardown use `try/finally`.
- Attempt-scoped workers, profiler children, report exporters, and temporary
  files are included in cleanup receipts.
- Existing artifacts are never overwritten.
- No branch switch, staging, commit, stash, reset, push, or `git clean` is
  permitted.

## Test Strategy

Write focused tests before implementation for:

1. decode ordinal and first-versus-steady classification;
2. event schema validation and non-negative timing;
3. four-rank step alignment and mismatch rejection;
4. collective-to-step association;
5. maximum-rank primary metric calculation;
6. non-CUDA upper-bound semantics;
7. paired summary and direction-agreement calculation;
8. classification thresholds and inconclusive behavior;
9. representative-pair selection nearest the median ratio;
10. worker emission of independent `decode_profile.json`;
11. profile finalization and wrapper restoration on success and failure;
12. exclusion of warmup and Nsight replay from primary statistics;
13. preservation of the canonical case and `profile.json` schemas;
14. command construction and attempt-scoped cleanup for Nsight.

Run focused unit tests, Python compilation, `git diff --check`, artifact
regeneration, output parity verification, resource receipts, and cleanup
verification before accepting the remote result.

## Success Criteria

1. Five measured recompute/exact pairs complete under the approved shared-GPU
   guard with 20/20 output-token parity.
2. All four ranks provide aligned first-step and steady-state rows.
3. Per-rank TP collective counts and CUDA durations are reported for every
   measured decode step.
4. First-step, steady-state, collective, non-CUDA-upper-bound, and rank
   imbalance summaries are reproducible from downloaded artifacts.
5. One representative paired Nsight capture reports top CUDA kernel and NCCL
   differences, or the artifact explicitly proves profiler unavailability.
6. Structured profiler overhead is measured and labeled.
7. The final conclusion states what the evidence proves, what it does not
   prove, and whether the slowdown is first-step, steady-state, collective,
   non-CUDA/synchronization-bound, mixed, absent, or inconclusive.
8. Cleanup is `CLEAN`, with no attempt-scoped worker or profiler process
   remaining.

## Evidence Boundary

This experiment can localize a repeatable slowdown to coarse decode regions
and correlate it with collectives or kernel timelines. It cannot infer an
exclusive causal mechanism merely because two timings move together.

In particular:

- CUDA Event intervals do not prove kernel occupancy;
- `step_wall - step_cuda` is not exact CPU or synchronization time;
- Nsight evidence from one representative pair does not establish
  repeatability;
- shared-GPU measurements remain vulnerable to unrelated load;
- the diagnostic result does not replace canonical benchmark authority.
