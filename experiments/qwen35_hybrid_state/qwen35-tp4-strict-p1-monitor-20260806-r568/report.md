# Qwen3.5 TP4 strict-P1 low-frequency monitor

## Superseded job

This read-only monitor was explicitly removed before any launch attempt. It
was superseded by
`qwen35-tp4-strict-p1-monitor-20260806-r570`, which adds a bounded 12-hour
timeout for the canonical workers command.

```text
launchd label:
com.bytedance.tinyllmforge.qwen35.strictp1.r568

monitor tag:
qwen35-tp4-strict-p1-monitor-20260806-r568

canonical run-tag base:
qwen35-tp4-strict-p1-canonical-20260806-r569

launch-attempt pattern:
qwen35-tp4-strict-p1-canonical-20260806-r569-attemptNNN
```

The launchd PID at startup was `27588`.

r562, r564, and r566 were removed before this monitor was submitted. They
performed only read-only resource samples and never started a launch attempt.

## Admission

Only GPUs `2,4,5,6` are eligible. Each must expose:

```text
free bytes >= 25769803776
compute_processes == []
```

Two consecutive `READY` samples separated by at least 60 seconds are required.
The canonical executor then performs a fresh resource preflight before
authorization or remote path creation. A launch-time `BLOCKED_RESOURCES` race
is recorded and monitoring resumes.

No dummy process reserves GPUs. No pre-existing process is signaled, stopped,
reprioritized, or modified.

## Current state

The first active sample was:

```text
observed_at: 2026-08-06T08:32:09.845842+00:00
classification: BLOCKED_RESOURCES

GPU 2: 68,789,731,328 free bytes, 5 compute processes
GPU 4: 50,963,939,328 free bytes, 5 compute processes
GPU 5: 84,979,744,768 free bytes, 0 compute processes
GPU 6: 58,502,152,192 free bytes, 3 compute processes
```

No benchmark, model load, CUDA worker, remote run directory, authorization, or
new GPU allocation has started.

## Cleanup

Cleanup first finds remote root processes whose `/proc/<pid>/cmdline`
contains both:

1. the unique launch-attempt run tag; and
2. that attempt's exact remote run root.

It then snapshots `/proc` parent links and process start-time ticks and expands
those roots to the complete descendant closure. Signals are sent only when the
PID still has the snapshotted start-time tick, protecting against PID reuse.
This covers spawned TP workers whose own command line does not repeat the run
tag.

`SIGTERM` and `SIGINT` received by the detached local monitor are converted to
exceptions during active launch execution so the same cleanup path runs.

After cleanup, the runner queries:

```text
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits
```

Success requires both:

```text
remaining_target_pids == []
matched_gpu_pids_after_cleanup == []
```

## Cleanup integration evidence

A remote no-GPU test created only a unique tagged `bash` root and one `sleep`
child:

```text
root_pids:                      [1223948]
descendants:                    [1223953]
target_pids:                    [1223948, 1223953]
remaining_target_pids:          []
matched_gpu_pids_after_cleanup: []
classification:                 CLEAN
```

The temporary log was removed. The test did not select or signal any
pre-existing process and allocated no GPU memory.

## Verification

```text
ControlMaster transport: 3 tests passed
monitor core:            6 tests passed
live monitor runner:     5 tests passed
py_compile:              PASS

strict-P1 source tree:
88ebda203decaf26a721f6299a5c9ed08b648841feb0b972f1395d6cc609eb9c

strict-P1 source tar:
8f807c0e51010fddd0bd474670ae41c09beda604c8247f672fc047f533c53a5d
```

Monitor files remain outside the frozen 91-file strict-P1 owned-source
inventory.

## Operations

Status:

```bash
launchctl list com.bytedance.tinyllmforge.qwen35.strictp1.r568
tail -n 5 experiments/qwen35_hybrid_state/qwen35-tp4-strict-p1-monitor-20260806-r568/resource_samples.jsonl
```

Stop only when explicitly requested:

```bash
launchctl remove com.bytedance.tinyllmforge.qwen35.strictp1.r568
```

The job is bounded to 1440 samples, approximately 24 hours.
