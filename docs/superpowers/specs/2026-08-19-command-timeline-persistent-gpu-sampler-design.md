# Command-Timeline Persistent GPU Sampler Design

**Date:** 2026-08-19

**Authoritative checkout:** `/Users/bytedance/Desktop/TinyLLMForge`

**Branch:** `feat/kv-sparse-attention`

## Problem

The immutable command-timeline campaigns `r6` and `r7` both stopped during
the first epoch, but for different reasons:

- `r6` exposed a GPU ownership sampling race, fixed by commit `d975a30`;
- `r7` proved that ownership and the worker completed successfully, then
  failed while attaching telemetry because two measured repeats had no GPU
  sample whose timestamp fell strictly inside the repeat interval.

The `r7` measured repeats lasted approximately 1.7 to 2.8 seconds. The
current sampler starts a fresh `nvidia-smi` subprocess for every sample and
queries every GPU before filtering to the selected four. Under the real
TP4 workload, adjacent GPU snapshots were sometimes more than two seconds
apart. Host telemetry remained complete.

This is a sampler-cadence defect. It is not evidence that the repeat lacked
GPU execution, and it must not be hidden by weakening telemetry admission.

## Decision

Replace the repeated `nvidia-smi` subprocess loop with one persistent
`nvidia-smi` process:

```text
nvidia-smi
  --id=<the four selected GPU indices>
  --query-gpu=<the existing twelve fields>
  --format=csv,noheader,nounits
  --loop-ms=200
```

A Python wrapper will read the stream and emit one canonical JSON snapshot
only after it has received exactly one row for every selected GPU. Every row
in that four-GPU snapshot receives the same `sampled_at_unix_ns` and
`sampled_at_monotonic_ns`, captured after the complete snapshot is read.

The wrapper owns the persistent child process. On normal shutdown, SIGTERM,
parse failure, or unexpected child exit, it terminates and reaps only that
owned child. It does not inspect, signal, adopt, or otherwise interact with
unrelated GPU or host processes.

## Evidence Boundary

The existing strict admission remains unchanged:

- every measured repeat must contain at least one complete four-GPU
  snapshot whose timestamp is inside the repeat's Unix interval;
- host telemetry must also contain an in-interval row;
- missing coverage remains a fail-closed error;
- no sample before or after a repeat may be borrowed;
- no sample may be synthetically duplicated across repeats;
- the four clean, process-free GPU preflight remains unchanged;
- immutable failed tags `r6` and `r7` are never reused.

This repair changes telemetry collection only. It does not change worker
commands, model identity, TP4/B4/Q4 shape, graph/eager schedule, warmup or
repeat count, timing decomposition, token parity, Proposal-KV semantics,
localization thresholds, runtime semantics, or optimization authorization.

## Interfaces

`tools/run_autoregressive_draft_command_timeline_remote.py` will add:

```python
def _build_gpu_sampler_script() -> str:
    """Return the dependency-light persistent sampler program."""
```

The generated program consumes one JSON list of four distinct non-negative
GPU indices. It writes canonical JSON Lines to stdout and diagnostics to
stderr. `_start_epoch_samplers(...)` continues returning the same process
and handle tuple, so campaign orchestration and artifact paths do not
change.

## Validation

The focused tests must establish:

1. the sampler command uses one persistent `nvidia-smi` process,
   `--loop-ms=200`, and `--id` limited to the selected GPUs;
2. incomplete and duplicate snapshots are rejected rather than emitted;
3. a complete snapshot emits four rows with identical Unix and monotonic
   timestamps;
4. SIGTERM and normal EOF terminate and reap the owned child;
5. `_attach_epoch_telemetry(...)` still rejects missing strict in-interval
   coverage.

After local GREEN, run the complete command-timeline runner/diagnostic
contract locally and remotely, atomically sync only the changed files,
and use the never-before-used tag:

```text
20260818-command-timeline-tp4-b4-q4-r8
```

The Mac Agent controller must continue polling official preflight and call
`execute` immediately when four clean GPUs are READY. `execute` retains its
built-in second preflight.

## Completion

This repair is complete only when:

- focused and full contract tests pass;
- the remote source transaction receipt binds the pushed commit and exact
  changed-file hashes;
- `r8` completes all eight epochs;
- primary and controller verifiers agree;
- the checksum manifest passes;
- the canonical audit and handoff reconcile `r6`, `r7`, and `r8`;
- all intended changes are committed and pushed.

Even a successful `r8` authorizes no runtime optimization unless its
verified canonical classification is `BOUNDARY_LOCALIZED` and a separate
optimization design is approved.
