# Command-Timeline Field-Isolated NVML Sampler Design

## Goal

Reduce complete four-GPU snapshot latency below the shortest measured repeat
without changing the measured request path, repeat duration, telemetry
coverage rule, or the eight retained NVML fields.

The sampler will run one persistent process for every frozen
`(GPU, query)` pair. A parent will broadcast one generation to all 32
children, accept only a complete four-GPU by eight-query result matrix, and
emit one shared-timestamp snapshot group.

## Immutable r17 Evidence

Campaign `20260818-command-timeline-tp4-b4-q4-r17` used one persistent NVML
child process per GPU. The Mac-owned controller found `READY` and immediately
launched the official run without another approval round.

The first two epochs completed:

- `block-0:eager:first`;
- `block-0:graph:second`.

The third epoch, `block-1:graph:first`, produced a zero worker status and a
complete raw worker result, but strict telemetry attachment failed with:

```text
ValueError: telemetry coverage is incomplete
```

Measured repeat 2 lasted `1825.127 ms`. It contained nine host rows but no
complete GPU snapshot. The complete GPU groups immediately around the repeat
were `1932.047 ms` apart. The group that crossed the interval boundary
finished `62.841 ms` after the repeat ended.

All four per-GPU children remained alive and returned complete rows. Their
query chains lasted between `1754.733 ms` and `1757.300 ms`. Most of the
eight APIs were delayed on every GPU at the same time:

- performance state: about `202 ms`;
- SM clock: about `228 ms` on three GPUs and `517 ms` on one GPU;
- memory clock: about `286-294 ms`;
- power: about `226-286 ms`;
- temperature: about `218-219 ms` on three GPUs;
- memory info: about `285 ms` on three GPUs;
- throttle reasons: about `237-241 ms`.

The four emitted rows used four distinct child PIDs, all timestamp groups
contained the frozen four UUIDs, sampler stderr was empty, and there were no
malformed or partial groups.

This rules out parent/child loss and process-local cross-GPU serialization as
the remaining cause. The wall-clock gap is the sum of eight driver-facing
query phases that occur sequentially inside each per-GPU child.

## Design Constraints

- Use only `/Users/bytedance/Desktop/TinyLLMForge`.
- Preserve r17 and every earlier tag as immutable evidence.
- Use fresh tag `20260818-command-timeline-tp4-b4-q4-r18` only after the new
  architecture passes its diagnostic and contract gates.
- Do not modify the worker, workload, warmup count, measured repeat count,
  repeat duration, schedule, or invariant verifier.
- Do not add synchronization or instrumentation to the measured request
  path.
- Preserve all eight real NVML values and a real duration for each query.
- Preserve frozen index/UUID validation and frozen output order.
- Emit no row until all 32 query results for one generation are complete.
- Preserve shared parent Unix and monotonic timestamps for the four emitted
  rows.
- Preserve the 200 ms target cadence, with at most one generation in flight.
- Preserve strict in-interval coverage, no boundary borrowing, and no
  synthetic duplication.
- Preserve four-clean-GPU admission and process ownership.
- Do not signal, pause, adopt, or otherwise interfere with unrelated
  processes.
- Keep every remote output, diagnostic artifact, cache, and receipt beneath
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.

## Considered Approaches

### 1. One persistent process per `(GPU, query)` pair

Create 32 persistent children. Each child owns one frozen GPU and exactly one
of the eight existing NVML API calls. The parent broadcasts a generation to
all children before collecting any result.

This is the selected approach. It crosses both process-local NVML boundaries
and the sequential per-device query-chain boundary while retaining the exact
queries and values.

The r17 evidence gives a falsifiable expectation: if the driver permits
different APIs to make progress concurrently, a generation should approach
the slowest individual call, observed at about `517 ms`, instead of the
roughly `1757 ms` sum.

### 2. One process per GPU with eight query threads

This would reduce process count, but all eight threads would share one loaded
NVML library and one process-local synchronization domain. The prior
same-process four-thread experiment did not reduce the roughly two-second
stall. This approach lacks evidence and is rejected.

### 3. Change the benchmark or coverage rule

Lengthening measured repeats, accepting a sample that finishes outside the
repeat, reusing boundary samples, or requiring only a subset of fields would
make the gate easier to pass. Each option changes the evidence contract and
is rejected.

### 4. Use a cached telemetry service

`nvmlDeviceGetFieldValues` exists on the host but its available field IDs do
not cover the required general metrics. `nvmlDeviceGetSamples` covers power,
utilization, and clocks but not the complete field set. DCGM is not installed.
The `nvidia-smi daemon` lifecycle is system-global and cannot be safely owned
or reaped without risking unrelated monitoring. These options cannot satisfy
the full contract on the current host.

## Architecture

### Query identities

The existing eight query identities remain authoritative and ordered:

1. `performance_state`;
2. `sm_clock`;
3. `memory_clock`;
4. `power_usage`;
5. `temperature`;
6. `utilization_rates`;
7. `memory_info`;
8. `clock_throttle_reasons`.

Each query identity has one function that accepts configured NVML state and a
resolved device handle, performs exactly one NVML call, converts the result
to the existing JSON representation, and returns:

```python
{
    "generation": generation,
    "gpu_index": gpu_index,
    "gpu_uuid": gpu_uuid,
    "query_name": query_name,
    "value": converted_value,
    "query_duration_ns": duration_ns,
    "sampler_process_pid": os.getpid(),
}
```

`utilization_rates` and `memory_info` remain one API call each and return
their existing two-value structures. They are not split into synthetic
subqueries.

### Child lifecycle

The parent uses `multiprocessing.get_context("fork")` and creates 32 duplex
pipes and 32 persistent children. Every child:

1. loads and configures its own `libnvidia-ml.so.1`;
2. calls `nvmlInit_v2`;
3. resolves exactly one assigned GPU index;
4. validates the frozen UUID;
5. sends a ready record containing GPU identity, query identity, and PID;
6. waits for `sample`, `stop`, or EOF;
7. performs only its assigned query for each sample generation;
8. sends a primitive-only result or a structured error;
9. calls `nvmlShutdown` exactly once in `finally`.

Startup succeeds only if the parent receives the complete Cartesian product
of four frozen GPUs and eight query identities, with 32 distinct positive
child PIDs.

### Generation protocol

The parent maintains a positive integer generation counter. For each sample:

1. increment the generation;
2. send `("sample", generation)` to all 32 children before receiving any
   result;
3. receive one result from every child;
4. reject stale, duplicate, missing, unexpected, or mismatched generation,
   GPU, UUID, query, or PID data;
5. reconstruct four rows in frozen GPU order;
6. attach the existing field names and the eight real per-query durations;
7. capture one shared Unix timestamp and one shared monotonic timestamp;
8. emit the four rows only after the entire matrix validates.

There is at most one in-flight generation. The design does not pipeline or
overlap generations, so a slow generation cannot be confused with a later
one and cannot create unbounded NVML pressure.

### Failure and shutdown

- A child error is fatal for the generation and names its GPU and query.
- EOF, malformed messages, duplicate matrix cells, missing cells, or a
  nonzero child exit are fatal.
- No child writes telemetry stdout, so no partial group can escape.
- A signal requests parent shutdown. An in-progress generation is discarded,
  owned children receive `stop` when possible, and every owned child is
  joined.
- Only task-owned child PIDs are signaled or joined.
- Failure before complete startup emits no telemetry.

## Diagnostic Gate Before Production Integration

The architecture must first be exercised as an isolated candidate sampler
against the unchanged real graph workload. This diagnostic is not an official
campaign tag and may not be used as performance evidence.

The diagnostic must:

- use the same four-clean-GPU admission classifier;
- use task-owned paths and environment variables;
- launch the unchanged command-timeline performance worker;
- collect candidate sampler rows without modifying the worker;
- retain per-generation timing and all child stderr;
- preserve a detached source/hash receipt;
- cleanly reap every owned process;
- leave unrelated processes untouched.

The architecture is eligible for production integration only if all of the
following hold:

1. every emitted generation contains four frozen UUIDs and all eight query
   identities;
2. every generation contains 32 distinct sampler PIDs;
3. no partial or malformed row is emitted;
4. every query duration is a positive real measurement;
5. every sampler stderr is empty;
6. at least five unchanged measured repeats are observed;
7. every observed repeat contains at least one complete candidate snapshot;
8. the maximum complete-generation gap across the measured interval is less
   than the shortest observed repeat duration;
9. worker output and invariants remain valid;
10. before/after GPU inventory is unchanged and clean.

If any item fails, preserve the diagnostic and do not launch r18. Return to
root-cause analysis instead of weakening the criteria.

## Test Design

The fake-NVML contract will prove:

1. startup creates the exact four-GPU by eight-query matrix;
2. all 32 child PIDs are distinct and differ from the parent;
3. the parent broadcasts one generation before receiving results;
4. output contains four rows in frozen GPU order;
5. each row contains the existing values and eight duration keys;
6. delayed child responses do not permit partial output;
7. stale, duplicate, missing, and wrong-generation messages fail closed;
8. query failure identifies both GPU and API and emits no group;
9. UUID mismatch and invalid inventory fail closed;
10. SIGTERM produces 32 shutdown markers and a clean parent exit;
11. all existing telemetry attachment and command-timeline contracts remain
    green.

The current four-process implementation must fail the new 32-process matrix
contract before production code changes.

## Delivery Sequence

1. Add fake-NVML RED contracts.
2. Implement query-specific child functions and the generation matrix.
3. Run focused and complete local command-timeline contracts.
4. Commit and push exact implementation paths.
5. Atomically sync to the remote source and retain detached hashes.
6. Run focused and complete remote contracts.
7. Run the isolated real-workload diagnostic gate.
8. Only after the diagnostic is fully green, start the Mac-owned condition
   controller.
9. On `READY`, immediately launch immutable official tag
   `20260818-command-timeline-tp4-b4-q4-r18`.
10. Preserve any failure and continue only under a fresh later tag.

## Success Boundary

This design establishes only that the telemetry system can provide strict
in-interval complete snapshots without changing the measured workload. It
does not itself establish a runtime performance improvement.

The command-timeline campaign remains incomplete until one official tag
finishes all eight epochs, both verifiers pass against the retained primary
artifacts, the manifest is complete, the audit and handoff are reconciled,
and the final repository state is committed and pushed.
