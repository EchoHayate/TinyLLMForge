# TP4 Decode Replay Hot-Path Capture and Phase Isolation Design

**Date:** 2026-09-06

**Status:** Approved for implementation

## Problem

The r48 Qwen3.8-27B BF16 TP4 decode-replay run cannot satisfy the
frozen capture-cost evidence contract:

- four measured graph cases have no measured capture-cost rows;
- warmup and measured cohorts use different lease-sealed identities;
- warmup captures consume the process-wide `total_capture_ns` budget;
- measured identities are then rejected by `total_capture_budget`;
- observed single-capture durations remain above the frozen
  `2_000_000_000 ns` limit.

The current capture path also executes the same decode protocol twice
after a successful eager decode:

1. one uncaptured forward inside
   `_capture_exact_multi_sequence_graph()`;
2. one forward inside `torch.cuda.graph(...)`.

The first execution is redundant on this hot path because capture is
attempted only after the current eager decode has completed
successfully and the identity has reached its observation threshold.
For lease-transactional Qwen3.8 execution, the redundant execution
also mutates transactional state and scratch KV before both mutations
are rolled back after capture.

These are two separate issues:

- duplicate hot-path execution is a candidate performance
  optimization;
- warmup-to-measured cache leakage is a benchmark evidence-isolation
  defect.

They must remain separately named and tested. Phase isolation is not a
performance claim.

## Frozen Boundaries

This change must not alter any of the following:

- exact lease identity remains sealed by ordered
  `slot_id + generation + request_id`;
- generation-agnostic or dynamic pool-index replay remains a Stage-1
  candidate and is out of scope;
- single capture budget remains `2_000_000_000 ns`;
- total capture budget remains `5_000_000_000 ns`;
- replay coverage remains at least `0.80`;
- shared-capacity evidence remains `DIAGNOSTIC_ONLY`;
- capture rollback must restore transactional model state and scratch
  KV on both success and failure;
- TP4 capture admission continues to use the maximum capture duration
  observed across ranks;
- no r48 artifact may be modified, supplemented, or reclassified.

## Considered Approaches

### A. Retain the internal warmup and only reset the cache

This repairs measured-phase accounting but retains the known duplicate
model execution. It is the lowest-risk evidence repair, but r48
already shows multi-second captures and therefore gives little reason
to expect the frozen single-capture gate to pass.

### B. Capture on the already-hot eager path and reset at the phase boundary

This is the selected approach. The runtime continues to require
successful eager execution before capture admission, but the capture
function performs only the execution inside `torch.cuda.graph(...)`.
The benchmark resets all exact graph state after warmup and before
measured execution.

This approach is narrow, preserves the existing lease-sealed protocol,
and directly tests whether the redundant execution is responsible for
a material part of capture latency.

### C. Make graphs independent of physical lease identity

This could preserve warmup graphs across measured cohorts, but current
hybrid-state gather and commit operations bind captured tensor access
to concrete physical slots. Removing slot or generation fields from
the identity without introducing dynamic device-side indirection would
be unsafe. This remains a separate Stage-1 design if approach B cannot
meet the frozen gate.

## Design

### 1. Capture on the hot path

`ModelRunner.run_model()` keeps the existing order:

1. build the exact identity;
2. miss the ready cache;
3. complete eager logits successfully;
4. record the successful identity observation;
5. if admitted, capture that same identity after the eager result is
   already available.

`_capture_exact_multi_sequence_graph()` will:

1. allocate and populate static graph tensors;
2. snapshot scratch KV;
3. snapshot transactional model state when the protocol is
   `lease_transaction_v1`;
4. set the exact capture context;
5. enter `torch.cuda.graph(...)` immediately;
6. execute exactly one model step inside the graph context;
7. synchronize after capture;
8. restore transactional state and scratch KV;
9. rebuild and compare the exact identity;
10. return the entry for TP-wide duration synchronization and budget
    commit.

It will no longer execute an uncaptured model step or an associated
pre-capture synchronization inside the capture function.

The capture receipt will replace the obsolete
`warmup_forward_completed` and `warmup_synchronize_completed` phases
with `hot_path_eager_prerequisite`. This phase records the runtime
contract: capture was reached only through the post-success eager
admission path. The receipt still records capture begin, capture body
completion, post-capture synchronization, and scratch restoration.

The first fresh remote smoke must enable
`TINYVLLM_EXACT_GRAPH_CAPTURE_RECEIPT_ROOT` under the approved
`/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`
tree. The receipt is diagnostic evidence; it does not replace the
frozen capture-duration fields.

### 2. Exact graph cache phase reset

`ExactCudaGraphCache.reset_phase(synchronize=...)` will provide one
complete phase-boundary operation.

Precondition:

- `capturing` must be empty. A reset during an active capture raises
  and leaves the cache unchanged.

On success it will:

1. reset every ready graph;
2. synchronize once when at least one ready graph was released;
3. clear ready entries;
4. clear observation counts;
5. clear rejected identities;
6. clear counters;
7. set static bytes, retained reserved-delta accounting, and total
   capture time to zero.

The method returns a deterministic receipt containing the number of
released ready entries, cleared observations, cleared rejections, and
the zeroed post-reset summary.

This reset deliberately clears phase-local accounting. It does not
claim that the CUDA allocator has returned every previously reserved
byte to the driver. Measured peak-memory evidence is reset separately
after graph release.

### 3. ModelRunner and LLMEngine acknowledgement

`ModelRunner.reset_exact_cuda_graph_cache()` will:

- call the cache phase reset with `torch.cuda.synchronize`;
- clear `_exact_cuda_graph_pool`;
- return the reset receipt with its rank.

`LLMEngine.reset_exact_cuda_graph_cache(timeout_s=...)` will dispatch
that method through the existing acknowledged command path. It will
require:

- exactly one receipt for every rank;
- each receipt's embedded rank to match the acknowledgement rank;
- all non-rank receipt fields to agree across ranks;
- a post-reset summary with no ready, rejected, capturing, or observed
  identities and zero accounting.

Any missing, malformed, or disagreeing acknowledgement aborts the
benchmark before measured requests begin. A partial reset caused by a
rank failure is not accepted as measured evidence.

### 4. Worker phase order

After the warmup request batch finishes and the engine is idle,
`tp4_decode_replay_worker.run_arm()` will execute:

1. `clear_reusable_prefix_cache()`;
2. `reset_exact_cuda_graph_cache(timeout_s=...)`;
3. `reset_decode_internal_profile(timeout_s=...)`;
4. `reset_peak_memory_stats(timeout_s=...)`;
5. measured request batch.

The graph reset precedes profiler and peak-memory resets so graph
release work cannot contaminate measured profile or peak-memory
evidence.

The eager arm uses the same phase-boundary call. Its exact graph cache
is expected to be empty, which proves the reset protocol is symmetric
and avoids arm-specific harness behavior.

## Failure Handling

- Active capture during reset: fail closed before mutating cache state.
- Graph without callable `reset()`: fail closed and do not begin the
  measured phase.
- Rank receipt mismatch or missing rank: fail closed in `LLMEngine`.
- Capture failure: preserve the existing terminal rejection path and
  rollback behavior.
- Transactional or scratch restore failure: preserve
  `scratch_unavailable` and the chained failure evidence.
- Fresh smoke still above `2 s`: retain `NO_GO`/`INCOMPLETE`; do not
  relax the threshold.
- Fresh smoke cannot capture or replay correctly: revert the
  optimization hypothesis and retain the evidence as a failed
  experiment.

## Test Strategy

### RED 1: hot-path capture executes the model once

Extend the dependency-light model-runner test fixture to count model
executions during direct capture. Assert:

- forward protocol: one model execution, not two;
- lease-transaction protocol: one
  `run_exact_cuda_graph_step()` call, not two;
- state and scratch KV are restored;
- receipt phases contain `hot_path_eager_prerequisite` and do not
  contain either obsolete warmup phase.

The tests must fail against the current two-execution implementation.

### RED 2: cache phase reset clears all accounting

Populate ready, rejected, observed, counter, capture-time, and byte
state. Assert that `reset_phase()`:

- releases ready graphs before synchronization;
- clears all phase state and accounting;
- returns the deterministic receipt;
- rejects an active capture without partial mutation.

### RED 3: worker invokes an acknowledged reset in fixed order

Extend the TP4 worker fake engine with an ordered call log. Assert the
phase boundary is:

`clear_prefix -> reset_graph_cache -> reset_profile -> reset_peak`.

Assert measured capture-cost rows are derived only after the reset.

### GREEN and regression

After minimal implementation:

- run the focused cache, model-runner, engine-wiring, and worker tests;
- run the adjacent TP4 replay suite;
- run syntax compilation and `git diff --check`;
- review exact changed paths before commit.

## Remote Qualification

The first remote validation is a fresh-tag one-case graph smoke, not a
resume or rewrite of r48. It must:

- use a new source revision and run tag;
- write all task data, logs, receipts, and temporary files below the
  approved remote root;
- enable capture phase receipts;
- verify exact output and lifecycle cleanup;
- report per-rank capture duration and receipt phases;
- retain the frozen `2 s` single and `5 s` total limits.

Only if the smoke is correct and passes capture budgets may the same
source revision proceed to the complete 30-case/15-pair gate, dual
verification, manifest, audit, commit, and push.

## Claim Boundary

Before fresh hardware evidence, this change is only:

- a locally verified removal of one duplicate capture-path model
  execution; and
- a locally verified warmup/measured evidence-isolation repair.

It is not yet a Qwen3.8-27B TP4 performance improvement. A performance
claim requires fresh strict-clean hardware evidence with correctness,
capture budgets, replay coverage, throughput, latency, memory, and
lifecycle gates all satisfied.
