# Command-Timeline Concurrent NVML Sampler Design

## Goal

Restore complete in-interval four-GPU telemetry snapshots during active
command-timeline epochs by removing the measured aggregate latency caused by
serializing four independent per-GPU NVML query chains.

The change must preserve the approved telemetry schema, fail-closed behavior,
strict coverage rules, and complete separation from the measured request path.

## Evidence Requiring This Change

Run `20260818-command-timeline-tp4-b4-q4-r15` preserved per-API timing for the
direct-NVML sampler. During the uncovered eager repeat, several normally cheap
queries stalled at the same time:

- `memory_clock`: approximately 566 ms across the four serialized GPUs;
- `performance_state`: approximately 330 ms;
- `clock_throttle_reasons`: approximately 324 ms;
- `power_usage`: approximately 300 ms;
- `sm_clock`: approximately 231 ms;
- `memory_info`: approximately 222 ms.

No single field explains the roughly two-second sampling gap. The gap is the
sum of multiple delayed calls across four GPUs because the current sampler
queries every field on GPU 1, then GPU 2, then GPU 3, then GPU 4. Host telemetry
remained complete, sampler stderr was empty, and the worker returned zero.
Therefore the localized bottleneck is aggregate cross-GPU serialization in the
GPU sidecar.

## Considered Approaches

### 1. Concurrent per-GPU chains in one sampler process

Create one fixed four-worker thread pool. Each worker owns one frozen
`(index, UUID, handle)` tuple and executes that GPU's eight NVML calls in the
existing order. The main sampler waits for all four workers before emitting
the snapshot.

This is the selected approach. It changes only cross-GPU scheduling, keeps the
single process and one JSONL stream, and reduces snapshot latency from the sum
of four device chains toward the slowest device chain.

### 2. One sampler subprocess per GPU

Four independent processes would avoid shared Python and ctypes state, but
would require an additional merger to assign one canonical timestamp and
guarantee all-or-nothing four-row emission. That adds process lifecycle,
buffering, and failure-coordination complexity without evidence that threads
are insufficient.

### 3. Parallelize individual APIs

Running all eight calls concurrently would reduce latency further but would
change the ordering and concurrency semantics within one device. It would make
the timing evidence harder to interpret and introduce unnecessary pressure on
the same NVML device handle. The evidence only requires cross-GPU concurrency.

## Selected Architecture

The generated sampler imports `ThreadPoolExecutor` and creates exactly one
executor with `max_workers=4` after NVML initialization and frozen inventory
validation.

A focused `sample_device(device)` function:

1. allocates fresh ctypes output objects;
2. executes the existing eight timed NVML calls sequentially;
3. preserves the existing canonical query names and order;
4. returns one complete row without printing.

For every sampling iteration, the main thread submits the four frozen devices
in inventory order and materializes all four results before capturing the
shared Unix and monotonic timestamps. `executor.map` preserves input order, so
the emitted GPU row order remains `[2, 3, 4, 6]` for the current inventory even
though completion order may differ.

The executor is reused across snapshots and shut down before `nvmlShutdown`.
No new dependency is introduced; `concurrent.futures` is part of Python's
standard library.

## Failure and Shutdown Semantics

- A query failure in any worker raises through the existing named `check`
  helper.
- The main thread must collect all four results before printing any row.
  Therefore a failed device cannot produce a partial snapshot.
- Other already-running device chains may finish before the exception is
  observed, but their rows are discarded.
- SIGTERM and SIGINT continue setting `stop_requested`. A signal received
  during a snapshot allows the owned query chains to finish, discards that
  snapshot, shuts down the executor, and calls `nvmlShutdown` exactly once.
- An executor shutdown or NVML shutdown failure remains fatal unless the
  existing signal-requested shutdown exception applies to `nvmlShutdown`.
- Frozen index/UUID validation and handle creation remain single-threaded
  before the executor starts.

## Preserved Invariants

- The measured request path receives no synchronization or instrumentation.
- All eight required fields and all eight `query_duration_ns` values remain.
- Calls remain sequential and in the same order within each GPU.
- Only complete four-GPU snapshots receive shared timestamps and are emitted.
- The 200 ms target cadence and monotonic deadline logic remain unchanged.
- Strict in-interval coverage, no boundary borrowing, and no synthetic
  duplication remain unchanged.
- The preflight still requires four distinct process-free GPUs with memory
  used at most 1024 MiB and utilization at most 5%.
- All remote output and scratch state remain beneath
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.

## Test Design

The fake-NVML subprocess contract will add three proofs:

1. A four-party barrier in the first per-GPU query proves that all four device
   chains can be active concurrently. The old serialized implementation fails
   this test because the first call cannot reach the barrier with three peers.
2. Per-device call logs prove that each GPU still observes the exact sequence:
   performance state, SM clock, memory clock, power, temperature, utilization,
   memory, and throttle reasons.
3. The existing injected power-query failure must still return nonzero, name
   the failing API, emit no JSON row, and call `nvmlShutdown` once.

Existing complete-snapshot, UUID mismatch, invalid inventory, duration schema,
SIGTERM, sampler startup, telemetry attachment, and full command-timeline
contracts must remain green locally and remotely.

## Campaign Validation

After local and remote contracts pass, source synchronization will preserve a
detached receipt and the Mac Agent will invoke official `execute` with fresh
tag `20260818-command-timeline-tp4-b4-q4-r16` as soon as official preflight
returns `READY`.

The run succeeds at this layer only if every measured repeat has at least one
complete in-interval four-GPU snapshot. The immutable raw JSONL will also be
checked for four-row timestamp groups, the expected UUID set, and retained
per-API durations. A later strict-gate failure is preserved and diagnosed
under a new tag; it does not justify weakening telemetry.
