# Qwen3.5 TP4 strict-P1 low-frequency monitor

## Superseded job

This read-only monitor was explicitly removed before any launch attempt. It
was superseded by
`qwen35-tp4-strict-p1-monitor-20260806-r568`, which adds descendant-closure
cleanup for spawned TP worker processes.

```text
launchd label:
com.bytedance.tinyllmforge.qwen35.strictp1.r566

monitor tag:
qwen35-tp4-strict-p1-monitor-20260806-r566

canonical run-tag base:
qwen35-tp4-strict-p1-canonical-20260806-r567

launch-attempt pattern:
qwen35-tp4-strict-p1-canonical-20260806-r567-attemptNNN
```

r562 and r564 were removed before this job was submitted. They performed only
read-only resource samples and never started a launch attempt.

## Admission gate

Only GPUs `2,4,5,6` are eligible. Every fixed GPU must simultaneously expose:

```text
free bytes >= 25769803776
compute_processes == []
```

The monitor requires two consecutive `READY` samples, separated by at least
60 seconds. The canonical executor then performs its own fresh resource guard
before authorization and remote path creation. A launch-time
`BLOCKED_RESOURCES` race is recorded and returns to monitoring.

No dummy process reserves the GPUs. No pre-existing process is signaled,
stopped, reprioritized, or modified.

## Current state

The first two r566 observations were `BLOCKED_RESOURCES`. At sample 2:

```text
observed_at: 2026-08-06T08:26:02.563098+00:00

GPU 2: 68,789,731,328 free bytes, 5 compute processes
GPU 4: 50,963,939,328 free bytes, 5 compute processes
GPU 5: 84,979,744,768 free bytes, 0 compute processes
GPU 6: 58,502,152,192 free bytes, 3 compute processes
```

No benchmark, model load, CUDA worker, remote run directory, authorization, or
new GPU allocation has started.

## Transport

The monitor reuses the authenticated ControlMaster:

```text
/tmp/ssh-dynamic-token-ppo-a66cd65
```

A pre-launch 4096-byte SSH/SCP round trip passed upload, remote SHA-256,
download identity, cleanup, and absence verification.

## Cleanup

Cleanup is scoped to processes whose `/proc/<pid>/cmdline` contains both:

1. the unique launch-attempt run tag; and
2. that attempt's exact remote run root.

It never selects processes by user, GPU, executable name, or unrelated PID
inventory. `SIGTERM` and `SIGINT` are converted into exceptions during active
launch execution so this cleanup path still runs.

After signaling scoped processes, cleanup queries:

```text
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits
```

The cleanup succeeds only when no still-matching run PID remains in either
`/proc` or the compute-app PID inventory.

The harmless nonexistent-run probe returned:

```text
classification:                         CLEAN
matched_pids:                           []
remaining_pids:                         []
matched_gpu_pids_after_cleanup:         []
```

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

The monitor files remain outside the frozen 91-file strict-P1 owned-source
inventory.

## Operations

Status:

```bash
launchctl list com.bytedance.tinyllmforge.qwen35.strictp1.r566
tail -n 5 experiments/qwen35_hybrid_state/qwen35-tp4-strict-p1-monitor-20260806-r566/resource_samples.jsonl
```

Stop only when explicitly requested:

```bash
launchctl remove com.bytedance.tinyllmforge.qwen35.strictp1.r566
```

The job is bounded to 1440 samples, approximately 24 hours.
