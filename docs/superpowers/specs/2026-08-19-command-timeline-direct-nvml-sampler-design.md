# Command-Timeline Direct NVML Sampler Design

**Date:** 2026-08-19

**Authoritative checkout:** `/Users/bytedance/Desktop/TinyLLMForge`

**Branch:** `feat/kv-sparse-attention`

## Problem

The immutable `r8` campaign proved that keeping one persistent
`nvidia-smi --loop-ms=200` process removes process-startup overhead but does
not guarantee strict in-repeat coverage under the real TP4 workload.

`r8` produced 300 complete four-GPU snapshots with a median gap of about
284 ms, but one real `nvidia-smi` query stalled for about 2129 ms. The final
measured repeat lasted about 1691 ms and therefore contained no complete
four-GPU snapshot. The worker exited successfully and GPU ownership was
correct; invariant attachment failed closed with:

```text
ValueError: telemetry coverage is incomplete
```

This localizes the remaining defect to the `nvidia-smi` query path. It does
not justify borrowing boundary samples, duplicating samples, weakening the
four-GPU requirement, or instrumenting the measured request path.

## Considered Approaches

### A. Direct NVML through Python `ctypes` — selected

Load the host-provided `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1` through
`ctypes`, initialize NVML once, bind the four preflight-selected devices,
and query the required fields directly every 200 ms.

This removes the CLI parser and `nvidia-smi` query serialization layer,
adds no Python package dependency, and leaves the worker process and
measured request path unchanged.

### B. Install `pynvml` or `nvidia-ml-py` — rejected

Neither package is present in the pinned runtime or test environment.
Adding a package would expand the environment transaction and introduce a
new dependency when the stable system C ABI is already installed.

### C. Relax telemetry admission — rejected

Boundary borrowing, synthetic duplication, or accepting fewer than four
rows would change the evidence contract and could turn a telemetry hole
into a false performance claim.

## Sampler Architecture

`_build_gpu_sampler_script()` continues producing a standalone Python
program. The program:

1. validates one JSON inventory containing exactly four distinct,
   non-negative GPU indices and four distinct non-empty UUIDs;
2. loads `libnvidia-ml.so.1` and calls `nvmlInit_v2()` once;
3. resolves each device with `nvmlDeviceGetHandleByIndex_v2()`;
4. verifies each resolved handle with `nvmlDeviceGetUUID()` against the
   preflight-frozen UUID at the same position;
5. every 200 ms reads:
   - `nvmlDeviceGetPerformanceState`;
   - `nvmlDeviceGetClockInfo` for SM and memory clocks;
   - `nvmlDeviceGetPowerUsage`;
   - `nvmlDeviceGetTemperature`;
   - `nvmlDeviceGetUtilizationRates`;
   - `nvmlDeviceGetMemoryInfo`;
   - `nvmlDeviceGetCurrentClocksThrottleReasons`;
6. emits no rows until all fields for all four GPUs succeed;
7. assigns one shared Unix and monotonic completion timestamp to the four
   rows;
8. sleeps against a monotonic 200 ms schedule;
9. handles SIGTERM and SIGINT by exiting the loop and calling
   `nvmlShutdown()` exactly once.

All NVML return codes other than `NVML_SUCCESS` are fatal. Diagnostics use
`nvmlErrorString()` when available. An incomplete snapshot is never
emitted.

## Schema Compatibility

The JSON Lines schema remains unchanged:

```text
sampled_at_unix_ns
sampled_at_monotonic_ns
nvidia_timestamp
gpu_index
gpu_uuid
pstate
sm_clock_mhz
memory_clock_mhz
power_w
temperature_c
gpu_utilization_percent
memory_utilization_percent
memory_used_mib
throttle_reasons_active
```

`nvidia_timestamp` remains present for compatibility and is generated from
the shared Unix completion time in UTC. Power is converted from NVML
milliwatts to watts; used memory is converted from bytes to MiB; pstate is
formatted as `P<integer>`.

## Automatic Mac-Agent Controller

The controller is an active launcher, not a terminal notifier. The current
Mac Agent owns the loop:

```text
poll official preflight
  -> not READY: preserve bounded status and retry after 15 seconds
  -> READY: immediately call official execute
      -> built-in second preflight rejects a vanished window safely
      -> accepted: run the complete eight-epoch campaign
```

The loop never waits for another user response after observing `READY`.
Transient SSH failures are retried. Insufficient Kerberos lifetime is a
fail-fast terminal condition because credentials must not be refreshed
manually. The controller never signals, pauses, adopts, or kills unrelated
processes.

## Evidence Boundary

The following remain unchanged:

- four distinct clean, process-free GPUs are required;
- memory used must be at most 1024 MiB;
- utilization must be at most 5%;
- every measured repeat must contain a complete four-GPU snapshot with a
  timestamp strictly inside its Unix interval;
- every measured repeat must contain in-interval host telemetry;
- boundary samples cannot be borrowed;
- snapshots cannot be duplicated across repeats;
- failed tags `r6`, `r7`, and `r8` are immutable;
- the next tag is
  `20260818-command-timeline-tp4-b4-q4-r9`;
- runtime optimization remains unauthorized unless a fully verified
  campaign is classified `BOUNDARY_LOCALIZED` and a separate optimization
  design is approved.

## Validation

Tests inject a fake ctypes-compatible NVML object before executing the
generated script. They must prove:

1. the expected NVML functions have explicit argument and return types;
2. four verified handles produce four schema-compatible rows with shared
   timestamps;
3. index/UUID mismatch fails before sampling;
4. any NVML query failure emits no partial snapshot and reports the API;
5. SIGTERM exits cleanly and calls shutdown;
6. `_start_epoch_samplers()` passes the frozen index/UUID inventory;
7. strict telemetry attachment still rejects boundary-only evidence.

After local and remote contract GREEN, a real remote cadence smoke must run
under the Sitian task root. Only then may the Mac-Agent automatic controller
wait for four clean GPUs and launch fresh `r9`.

## Completion

The work is complete only when:

- focused and full local tests pass;
- focused and full remote tests pass;
- real direct-NVML cadence smoke is valid;
- the implementation commit is pushed and atomically synced with a detached
  receipt;
- the Mac-Agent controller launches `r9` automatically on `READY`;
- all eight epochs complete;
- primary and controller verifiers agree;
- `manifest.sha256` validates;
- the audit and handoff chronologically reconcile `r6` through `r9`;
- all intended documentation is committed and pushed.
