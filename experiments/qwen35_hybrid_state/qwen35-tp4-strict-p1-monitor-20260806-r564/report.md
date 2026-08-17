# Qwen3.5 TP4 strict-P1 low-frequency monitor

## Status

The detached monitor is active under:

```text
launchd label:
com.bytedance.tinyllmforge.qwen35.strictp1.r564

monitor tag:
qwen35-tp4-strict-p1-monitor-20260806-r564

canonical run-tag base:
qwen35-tp4-strict-p1-canonical-20260806-r565
```

Each actual launch attempt receives a unique `-attemptNNN` suffix.

## Gate

The monitor checks only GPUs `2,4,5,6`.

Every GPU must simultaneously satisfy:

```text
free bytes >= 25769803776
compute_processes == []
```

Two consecutive samples must be `READY`. Samples are separated by at least
60 seconds. A launch-time canonical preflight is still required before
authorization or remote path creation.

No dummy process is used to reserve GPUs. No pre-existing process is stopped,
signaled, reprioritized, or modified.

## Current observation

The first two active samples were both `BLOCKED_RESOURCES`:

```text
sample 1: 2026-08-06T08:18:30.150803+00:00
sample 2: 2026-08-06T08:19:32.133695+00:00
interval: 61.982892 seconds

GPU 2: 68,789,731,328 free bytes, 5 compute processes
GPU 4: 50,963,939,328 free bytes, 5 compute processes
GPU 5: 84,979,744,768 free bytes, 0 compute processes
GPU 6: 58,502,152,192 free bytes, 3 compute processes
```

Therefore no benchmark, model load, CUDA worker, remote run directory, or new
GPU allocation had started when this report was written.

## Transport

The canonical command runner uses the already authenticated ControlMaster:

```text
/tmp/ssh-dynamic-token-ppo-a66cd65
```

Before monitor launch, a 4096-byte SSH/SCP round trip passed remote SHA-256,
download identity, cleanup, and absence checks.

## Cleanup boundary

After a completed or failed launch, cleanup matches only remote processes
whose `/proc/<pid>/cmdline` contains both:

1. the unique launch-attempt run tag; and
2. the exact remote run root for that same attempt.

It does not kill by user, GPU, executable name, or an unrelated PID list.

## Verification

```text
ControlMaster transport: 3 tests passed
monitor core:            6 tests passed
live monitor runner:     3 tests passed
py_compile:              PASS
scoped cleanup probe:    CLEAN, matched_pids=[]

strict-P1 source tree:
88ebda203decaf26a721f6299a5c9ed08b648841feb0b972f1395d6cc609eb9c

strict-P1 source tar:
8f807c0e51010fddd0bd474670ae41c09beda604c8247f672fc047f533c53a5d
```

The monitor implementation is outside the frozen 91-file strict-P1 owned
source set and does not change the benchmark source identity.

## Operations

Status:

```bash
launchctl list com.bytedance.tinyllmforge.qwen35.strictp1.r564
tail -n 5 experiments/qwen35_hybrid_state/qwen35-tp4-strict-p1-monitor-20260806-r564/resource_samples.jsonl
```

Stop only when explicitly requested:

```bash
launchctl remove com.bytedance.tinyllmforge.qwen35.strictp1.r564
```

The monitor is bounded to 1440 samples, approximately 24 hours at the
configured interval.
