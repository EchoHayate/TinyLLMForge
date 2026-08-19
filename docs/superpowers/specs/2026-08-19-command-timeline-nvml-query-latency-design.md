# Command-Timeline NVML Query-Latency Evidence Design

## Goal

Identify which direct-NVML call causes the roughly two-second GPU telemetry
gaps observed during active command-timeline CUDA Graph epochs, without
changing the measured request path, weakening telemetry coverage, or removing
any required GPU field.

## Evidence Requiring This Change

Run `20260818-command-timeline-tp4-b4-q4-r14` completed its graph worker with
status zero, but strict telemetry attachment failed. Measured repeats 1 and 3
had complete host telemetry and no complete four-GPU snapshot inside their
intervals. The nearest complete GPU snapshots were separated by roughly two
seconds, while repeats 0, 2, and 4 each contained seven or eight complete
snapshots. Both sampler stderr files were empty.

This localizes the failure to the direct-NVML sidecar rather than the worker,
host scheduler, telemetry attachment boundary rule, or GPU ownership gate.
The current sampler records only the timestamp after all queries finish, so it
cannot identify the blocking NVML API.

## Selected Design

The generated direct-NVML sampler will time each required query with
`time.monotonic_ns()` and attach one canonical `query_duration_ns` mapping to
each GPU row. Keys will distinguish both clock queries and every other API:

- `performance_state`
- `sm_clock`
- `memory_clock`
- `power_usage`
- `temperature`
- `utilization_rates`
- `memory_info`
- `clock_throttle_reasons`

Each value is the elapsed nanoseconds for exactly one existing NVML call on
that row's GPU. The sampler will still execute the same calls in the same
order, fail closed on any query error, and emit rows only after a complete
four-GPU snapshot is available.

## Data Flow

1. Before each NVML call, capture a monotonic start timestamp.
2. Execute the existing NVML function and pass its return code through the
   existing fail-closed `check` path.
3. Record the elapsed monotonic duration under the canonical query name.
4. Add the completed mapping to the in-memory row.
5. Emit the four rows together with the existing shared snapshot timestamps.
6. Preserve the raw JSONL beneath the immutable run directory. Telemetry
   attachment continues copying only identity and timestamp fields, so
   canonical verifier inputs do not change.

## Invariants

- No synchronization or instrumentation enters the measured request path.
- No NVML field is removed, substituted, cached, or borrowed across snapshots.
- No incomplete snapshot is emitted.
- Query failures remain fatal and name the failing NVML API.
- The 200 ms target cadence and monotonic deadline logic remain unchanged.
- Strict in-interval four-GPU coverage remains unchanged.
- All remote output remains under
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.

## Validation

Unit tests will require every emitted fake-NVML row to contain exactly the
eight non-negative integer duration keys. Existing UUID mismatch, query
failure, invalid inventory, SIGTERM shutdown, and complete-snapshot tests must
remain green.

The complete three-file command-timeline contract will run locally and
remotely. A fresh official campaign tag will then reproduce an active graph
epoch. The preserved raw GPU JSONL will be summarized by API and GPU using
maximum and high-percentile durations. Optimization will be designed only
after one or more blocking calls are identified from that evidence.

## Rejected Alternatives

### Parallelize per-GPU queries immediately

This may improve cadence but would obscure whether one API, one GPU, or the
aggregate serialized workload is responsible. It also changes sampler
concurrency before the root cause is known.

### Remove a suspected expensive field

Dropping power, throttle, or another field without timing evidence weakens the
telemetry contract and risks preserving the actual blocker.

### Borrow boundary snapshots

Assigning the nearest pre- or post-interval snapshot would violate the
approved strict coverage rule and conceal the sidecar failure.
