# Exact CUDA Graph Budget Fallback Fault-Injection Design

## Status

Approved as an additive correctness requirement for the production exact-width
multi-sequence CUDA Graph gate.

This design closes one uncovered requirement in:

- `docs/superpowers/specs/2026-07-22-production-exact-width-multi-sequence-cuda-graph-design.md`
- `docs/superpowers/plans/2026-07-22-production-exact-width-multi-sequence-cuda-graph.md`

The production design requires remote evidence for every budget or terminal
fallback, but the current remote worker does not deliberately produce those
states. Local unit tests alone are not sufficient evidence.

## Goal

Add source-bound remote correctness evidence for exactly these eight terminal
fallback reasons:

```text
entry_limit
static_byte_budget
reserved_byte_budget
single_capture_budget
total_capture_budget
scratch_unavailable
capture_failed
identity_drift
```

For every reason, the gate must prove all of the following:

1. the intended fault was active before the target identity became eligible;
2. the real `ModelRunner` target step executed eagerly and remained correct;
3. the exact target fallback reason was emitted;
4. no target graph replay occurred;
5. the identity became terminally rejected;
6. later eligible steps remained eager with the same reason;
7. no recapture occurred after rejection;
8. the evidence is bound to the candidate source tree and independently
   verified.

These workers are correctness-only. Their initialization time, throughput,
latency, memory, graph-hit rate, and capture time are excluded from all
production performance ratios.

## Non-Goals

This change does not:

- enable exact multi-sequence CUDA Graphs by default;
- change normal cache admission or production fallback behavior;
- add a production CLI, environment variable, or config field that can inject
  faults;
- weaken GPU isolation, source binding, or dynamic-port requirements;
- authorize rounded batch or page-width replay;
- replace the 315-case diagnostic, normal `ModelRunner` correctness matrix, or
  paired arrival-load gate;
- mix Light Doc Cache, Gist KV, token sparsity, low-rank KV compression, weight
  quantization, or other optimization branches into this work;
- contribute samples to performance statistics.

## Considered Approaches

### A. Accept Local Unit Tests

This is inexpensive but rejected. Unit tests prove cache methods in isolation;
they do not prove that a remote Qwen3-0.6B `ModelRunner` step emits the same
reason, preserves eager outputs and KV state, and remains terminal under real
CUDA execution.

### B. One Process That Sequentially Injects All Faults

This reduces model startup cost but is rejected. Cache counters, CUDA allocator
state, graph pools, scratch state, and monkeypatches can leak between faults.
The evidence would not show that each reason is independently sufficient.

### C. Eight Isolated Source-Bound Workers

This is the selected approach. One fresh remote process owns one reason, fresh
dynamic ports, a fresh engine/cache, its own output directory, and before/after
GPU occupancy evidence. It is slower, but it provides the strongest causal and
operational evidence without adding a production fault surface.

## Architecture

### Frozen Contract

`tools/multi_sequence_cuda_graph_contract.py` owns:

```python
BUDGET_FALLBACK_REASONS = (
    "entry_limit",
    "static_byte_budget",
    "reserved_byte_budget",
    "single_capture_budget",
    "total_capture_budget",
    "scratch_unavailable",
    "capture_failed",
    "identity_drift",
)
```

It also freezes the new aggregate artifact:

```text
budget_fallback_rows.jsonl
```

The order is contractual. A canonical correctness result requires exactly one
complete row for every reason in that order. Duplicate, missing, unknown, or
out-of-order reasons fail independent verification.

### Remote Orchestrator

`tools/run_multi_sequence_cuda_graph_production_gate_remote.py` adds
`worker_kind=budget-fallback` and a required
`--budget-fallback-reason=<closed-enum-value>`.

For `correctness-smoke` and `correctness-canonical`, the parent launches eight
additional workers after capacity calibration and before final aggregation.
Each worker receives:

- the same immutable candidate source snapshot;
- `CUDA_VISIBLE_DEVICES=0`;
- fresh distinct `TINYVLLM_DIST_PORT` and `MASTER_PORT`;
- the fixed remote Python and Qwen3-0.6B model;
- one reason only;
- a unique output directory;
- before/after GPU occupancy checks.

Unrelated occupancy makes the parent result `INCOMPLETE`; it is not classified
as a fallback failure or performance loss.

`arrival-smoke` does not rerun the eight workers. `arrival-canonical` accepts
only an independently verified canonical correctness binding that already
contains complete 8/8 evidence.

### Injection Boundary

Fault injection exists only in the gate harness after engine construction. No
production `Config`, environment variable, or `ModelRunner` branch checks for a
fault-injection option.

Two injection classes are allowed:

#### Configuration/State Preconditions

These use real cache ceilings and identities:

- `entry_limit`: capture one seed identity under `max_entries=1`, then present
  a distinct target identity;
- `static_byte_budget`: set the target static ceiling below the independently
  computed target estimate before its third observation;
- `reserved_byte_budget`: set a one-byte reserved ceiling so a real capture's
  retained reserved delta rejects the target at commit;
- `single_capture_budget`: set the single-capture ceiling to one nanosecond and
  keep the total ceiling non-binding;
- `total_capture_budget`: set the total-capture ceiling to one nanosecond and
  keep the single ceiling non-binding.

The worker records the exact effective cache config and pre-target cache
summary so the verifier can prove which ceiling was binding.

#### Harness-Scoped Runtime Faults

These wrap only the constructed worker instance:

- `scratch_unavailable`: make the target capture's scratch restore operation
  fail after the real eager comparison step;
- `capture_failed`: make the target capture operation raise the production
  `_ExactGraphCaptureError("capture_failed", ...)`;
- `identity_drift`: return a distinct exact identity only for the post-capture
  identity rebuild.

Every wrapper is installed immediately before the target identity reaches its
third observation and restored in `finally`. The worker records installation
and restoration booleans. The engine source contains no corresponding runtime
switch.

## Worker Sequence

Every worker executes this sequence:

1. construct a fresh feature-enabled engine and a feature-disabled eager
   reference engine with equal scheduler-visible KV capacity;
2. record source, environment, effective config, capacity, and GPU occupancy;
3. run enough setup steps to establish the fault precondition;
4. run target observations one and two and require
   `fallback_reason=cold_identity`;
5. install or confirm the target fault before observation three;
6. run the eager reference and candidate target step from equivalent inputs;
7. compare output token IDs, logits under the frozen tolerances, and live-slot
   KV hashes;
8. require candidate `dispatch=eager` and the exact target fallback;
9. run at least two later target steps;
10. require the same terminal fallback, zero target graph dispatches, and zero
    later target capture attempts;
11. restore any harness wrapper;
12. write one atomic `budget_fallback_row.json` and the normal raw dispatch,
    capture, correctness, memory, and environment evidence.

The reference and candidate engines may run in separate phases inside the same
isolated worker process, but no other fallback reason may be injected there.

## Evidence Schema

Each aggregate `budget_fallback_rows.jsonl` row contains exactly:

```text
row_id
case_id
reason
source_sha256
worker_pid
tinyvllm_dist_port
master_port
gpu
injection_class
injection_installed
injection_restored
effective_cache_config
pre_target_cache_summary
target_identity_fields
target_identity_sha256
observation_dispatch_row_ids
terminal_dispatch_row_ids
capture_row_ids
eager_output_token_ids
candidate_output_token_ids
logits_allclose
logits_max_abs_diff
eager_live_kv_sha256
candidate_live_kv_sha256
terminal_rejection_reason
target_graph_replay_count
target_capture_attempt_count
post_rejection_capture_attempt_count
complete
```

Raw referenced rows remain in the normal production artifacts. The aggregate
row is a binding index, not a substitute for raw evidence.

## Independent Verification

`tools/verify_multi_sequence_cuda_graph_production.py` must not trust the
producer's `complete` field. For correctness modes it independently proves:

1. the reason domain and order exactly equal
   `BUDGET_FALLBACK_REASONS`;
2. each row and every referenced raw row share the manifest source SHA;
3. worker ports are positive, distinct, and not reused by another process;
4. target identity fields recompute to the recorded SHA;
5. observations one and two are eager/cold;
6. the target and terminal rows are eager with the exact reason;
7. no referenced target row has `dispatch=graph`;
8. the target capture-attempt count matches raw dispatch/capture rows;
9. no capture occurs after the first terminal rejection;
10. eager and candidate token IDs are identical;
11. logits pass the frozen tolerances;
12. live-slot KV hashes are identical;
13. injection installation and restoration are true for runtime faults;
14. effective cache config and pre-target summary make the selected budget
    causally possible;
15. canonical correctness has all eight rows and a complete 315-case diagnostic.

Any failure makes correctness `NO_GO`, except unrelated GPU occupancy or an
interrupted/missing worker, which remains `INCOMPLETE`.

Arrival verification requires its bound correctness result to be canonical
`GO` and to report:

```text
budget_fallback_required = 8
budget_fallback_verified = 8
```

## Tamper Tests

Local verifier fixtures must reject at least:

- one missing reason;
- one duplicate reason;
- an unknown reason;
- a row whose declared reason differs from raw dispatch;
- a forged terminal rejection without a prior target observation;
- a graph replay after rejection;
- a recapture after rejection;
- mismatched output tokens;
- failed logits comparison;
- mismatched live KV hash;
- a source SHA mismatch;
- a changed identity field without a matching SHA;
- reused or identical worker ports;
- a runtime injection that was not restored;
- a budget configuration that cannot produce the declared reason;
- inclusion of fault-worker metrics in performance aggregation.

## Failure Classification

- `GO`: all eight rows pass independent reconstruction, normal remote
  correctness passes, and the 315-case diagnostic passes.
- `NO_GO`: any complete worker demonstrates wrong output/KV, wrong fallback,
  graph replay, recapture, non-terminal state, source mismatch, or verifier
  mismatch.
- `INCOMPLETE`: a worker did not run to completion because of unrelated GPU
  occupancy, transport interruption, missing model/runtime dependency, or
  missing artifact.

No partial count such as 7/8 is sufficient.

## Source and Operational Constraints

- Local edits occur only in
  `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Remote model/GPU work occurs only on
  `sitian@10.232.195.203` as user `sitian`.
- Use `/tmp/ssh-sitian-10.232.195.203` as the SSH ControlMaster path.
- Use remote Python
  `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`.
- Use model
  `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`.
- Use `CUDA_VISIBLE_DEVICES=0`.
- Every worker receives fresh distinct dynamic distributed ports.
- Do not use `rsync`, modify the remote checkout, kill unrelated processes,
  delete shared `/tmp`, or switch GPU.
- Retry only `EADDRINUSE`.
- Preserve untracked `experiments/`; never stage them.
- Do not publish a performance improvement until canonical correctness and
  canonical arrival-load verification both return `GO`.

## Completion Criteria

This additive design is implemented only when:

1. the closed eight-reason contract and artifact schema are frozen;
2. local TDD covers worker commands, isolation, injection lifecycle,
   aggregation, verifier reconstruction, tampering, and performance exclusion;
3. all local exact CUDA Graph suites pass;
4. source-bound remote preflight passes;
5. remote correctness produces independently verified 8/8 evidence;
6. arrival canonical accepts only that correctness binding;
7. no production fault switch exists;
8. canonical performance, if run, excludes every fault-worker sample;
9. the durable handoff records exact commands, source SHA, artifacts,
   classifications, and limitations.
