# TP4 Worker Schedstat Tail Diagnostic Design

## Problem

The `20260820-bounded-journal-tpot-source-pair-r911` and
`20260821-sampler-500ms-tpot-source-pair-r1` artifacts repeatedly localize
the dominant tail to eager measured repeat 2, engine step 4, and the second
`run_spec_verify_batch` call. That call always has active batch size 2 and
query length 3, but the same exact shape, selected-sequence hash, and
transaction digest are fast in other epochs. Shape is therefore necessary
for the observed spikes but not sufficient.

The slow calls contain one approximately 0.55--0.84 second
`row_parallel_all_reduce` CUDA interval on ranks that reach the collective
early. During the same interval, other selected GPUs can be idle. This is
consistent with one or more worker host threads reaching the collective late,
which amplifies one rank-local pause into a TP-wide CUDA tail. Existing
system-wide host telemetry cannot identify the late worker.

Spec-verify CUDA Graph admission is not a valid immediate fix: the current
implementation deliberately returns `tp_not_one` whenever tensor parallel
world size is not 1. The failing campaign is TP4.

## Falsifiable Hypothesis

During a slow query-length-3 second spec-verify call, at least one TP worker
has a rank-local scheduling or off-CPU delay that is absent from fast calls
with the same transaction and shape. The delayed worker reaches a
`row_parallel_all_reduce` late, while already-arrived peers wait in NCCL.

The hypothesis is supported only if a fresh run shows a rank-local increase
in `/proc/<pid>/schedstat` run-queue wait, a sustained non-running process
state, or a stable blocking `wchan` overlapping the slow command. If the
worker rows remain continuously runnable/running without a corresponding
scheduling delta, the hypothesis is rejected and the next investigation must
profile untracked CUDA kernels instead.

## Chosen Approach

Add an external process sampler to the command-timeline runner. The sampler
is a separate process and reads only:

- `/proc/<pid>/schedstat`;
- `/proc/<pid>/stat`;
- `/proc/<pid>/status`;
- `/proc/<pid>/wchan`.

It starts only after the runner has observed one owned compute process on
each of the four selected GPU UUIDs. Rank order is the selected
`CUDA_VISIBLE_DEVICES` order already passed to the worker. Each binding also
records the process start-time tick so PID reuse fails closed.

Sampling cadence is 10 ms. This resolves a 300--800 ms pause while avoiding
model-path instrumentation, ptrace, `perf`, CUDA synchronization, logging
inside the worker, or production GC control.

## Data Contract

Each JSONL row contains:

- schema version;
- Unix and monotonic sample timestamps;
- rank, GPU UUID, PID, and process start-time ticks;
- process state, last CPU, thread count, and `wchan`;
- CPU runtime, run-queue wait, and scheduler timeslice counters;
- user/system CPU ticks and block-I/O delay ticks;
- voluntary and involuntary context-switch counters;
- a terminal status when a bound process exits.

Rows must be canonical JSON mappings. Integer counters must be non-negative.
The sampler rejects duplicate ranks, UUIDs, or PIDs and rejects a changed
process start time.

## Runner Lifecycle

The existing GPU and host samplers still start before the worker. The process
sampler starts exactly once from the first complete owned TP4 GPU binding
observed by `_monitor_owned_worker()`.

Artifacts are written only below the immutable remote run root:

```text
telemetry/block-N/<mode>.process.jsonl
telemetry/block-N/<mode>.process.jsonl.stderr
```

The process sampler is terminated and reaped before epoch finalization.
Runner cleanup remains responsible for all owned sampler processes. No file
is written beneath remote `/`, `/tmp`, or `/private/tmp`.

## Verification

Epoch finalization fails unless:

- process-sampler stderr is empty;
- all four expected rank/UUID/PID bindings appear;
- process start times remain stable;
- every warmup and measured campaign interval contains at least one sample
  for every rank;
- per-rank monotonic timestamps and scheduler counters do not decrease;
- no sample names an unowned PID.

The raw process JSONL remains an immutable manifest member. It is not folded
into TPOT, TTFT, throughput, correctness, stationarity, or promotion fields.

## Experiment

After local RED-to-GREEN verification and commit/push, run one fresh
candidate command-timeline campaign on four clean GPUs using a new tag under:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818
```

For every step-4 spec-verify call, correlate the command interval with
per-rank scheduler-counter deltas and process states. Compare slow and fast
calls with identical shape, selected-sequence hash, and transaction digest.

No production optimization is authorized by the sampler alone. A production
change may follow only after the fresh evidence confirms one causal class and
a test can reproduce the intended behavior.

## Rejected Alternatives

- `perf sched`: richer data, but it adds privilege, kernel-tooling, and large
  artifact risks that are unnecessary for the first falsification.
- `strace` or `py-spy`: ptrace overhead and availability are uncontrolled.
- timestamps inside collectives or ModelRunner: this would add profiling to
  the measured production path.
- enabling spec-verify CUDA Graphs for batch 2: current graph support is TP1
  only and therefore cannot affect this TP4 run.
