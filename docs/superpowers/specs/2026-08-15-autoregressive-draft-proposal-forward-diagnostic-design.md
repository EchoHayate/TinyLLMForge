# Autoregressive Draft Proposal-Forward Diagnostic Design

## Goal

Explain the schema-v2 `proposal_forward` bottleneck without adding a CUDA
synchronization to the measured hot path, then produce a stable repeated
TP4 batch-4 diagnostic that chooses the first optimization target from
evidence.

## Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Keep `max_proposal_tokens=4`, temperature zero, accepted-prefix semantics,
  exact greedy parity, and workload-derived Proposal-KV capacity.
- Use `sitian@10.232.195.203` and GPUs `3,4,6,7`.
- Do not terminate or alter the existing GPU-7 service.
- Do not add `torch.cuda.synchronize()` or CUDA-event synchronization to the
  measured request path.
- Do not stage, commit, push, switch branches, stash, reset, or clean.
- Do not treat direct-allocator zero H2D/D2H as offload evidence.

## Considered Approaches

### 1. Immediate CUDA Graph implementation

This could reduce launch overhead, but schema-v2 does not distinguish model
submission, GPU completion/readback, TP collective wait, authority
convergence, and Proposal-KV bookkeeping. Implementing graphs now would
optimize a plausible cause rather than a demonstrated cause.

### 2. CUDA events around every draft decode

CUDA events can isolate GPU duration, but recording and resolving many events
changes the diagnostic surface and requires a synchronization boundary. It is
valuable after wall-clock attribution, not as the first refinement.

### 3. Non-synchronizing wall-clock substages, then a focused repeated b4 run

This is the selected approach. It preserves the current execution path and
uses `time.perf_counter()` around existing boundaries. Because CUDA work is
asynchronous, the design deliberately distinguishes backend submission from
the later point that waits for selected token IDs. It does not mislabel
submission latency as GPU kernel duration.

## Timing Model

Keep the existing executor parent totals:

```text
prompt_bootstrap
proposal_forward
proposal_finalize
```

Add a nested `proposal_forward_detail_ms` mapping:

```text
setup
backend_submit
selection_collective
decode_authority
token_readback
materialize_register
```

The detail intervals are non-overlapping wall-clock intervals accumulated
inside `_run_exact_q_group()`:

- `setup`: transaction creation, lease acquisition, and decode-row assembly.
- `backend_submit`: only `backend.decode_step_batch()`. It may enqueue CUDA
  work and return before kernels finish.
- `selection_collective`: logit validation/stacking and
  `select_tensor_parallel_greedy_tokens()`.
- `decode_authority`: the TP4 `_converge_stage()` call for each decode step.
- `token_readback`: `selected.tolist()` plus Python token-row publication.
  Any deferred CUDA completion can appear here.
- `materialize_register`: proposal materialization, logical-authority check,
  lifecycle registration, and transaction metadata publication.

The parent `proposal_forward` remains authoritative. Detail values are nested
and must not be added to runtime stage totals. The artifact records:

```text
proposal_forward_detail_sum_ms
proposal_forward_residual_ms
```

Per-key rank maxima are retained but are explicitly non-additive. The
artifact identifies the rank with maximum parent `proposal_forward`, records
that same rank's six-key `critical_rank_ms`, and computes detail sum and
residual only from that coherent critical rank. Residual must be nonnegative
within a small floating tolerance and captures untimed control/cleanup
overhead.

## Artifact Contract

Raise the performance artifact schema to version 3.

For learned measured runs:

- four rank rows must contain all six detail keys;
- all detail values must be finite and nonnegative;
- at least one of `backend_submit`, `selection_collective`,
  `decode_authority`, or `token_readback` must be positive;
- the parent critical rank and its detail row must be recomputed;
- critical-rank detail sum must not exceed that rank's parent
  `proposal_forward` beyond tolerance;
- per-key maxima, critical-rank values, sum, and residual must be recomputed
  by the verifier.

For target runs:

- parent executor timing remains all zero;
- every rank detail row and max-rank detail value is zero;
- detail sum and residual are zero.

The controlled four-cell gate remains available. A new focused diagnostic
reuses the same worker contract for learned batch 4 with:

```text
warmup runs: 2
measured runs: 8
prompt tokens: 256
output tokens: 16
batch size: 4
TP: 4
```

The focused diagnostic must retain every raw repeat, exact parity against a
fresh target reference, per-rank detail timing, GPU snapshots, source archive,
dual verifier receipts, and a checksum manifest.

## Decision Rule

After the focused diagnostic:

- If `backend_submit + selection_collective + token_readback` dominates and
  metadata/authority is small, implement exact-shape independent-draft CUDA
  Graph capture.
- If `decode_authority` dominates, reduce TP authority/collective frequency
  without weakening failure convergence.
- If `materialize_register` or parent residual dominates, optimize
  Proposal-KV/lifecycle metadata before graph work.
- If raw learned-b4 E2E remains strongly non-stationary, classify the
  diagnostic as unstable and investigate environmental/runtime accumulation
  before claiming any optimization delta.

## Validation

- Deterministic-clock executor tests must prove exact detail accumulation.
- Worker/gate tests must prove delta extraction, aggregate recomputation,
  target zero semantics, learned positive evidence, and tamper rejection.
- Full executor tests run in the remote py311 environment.
- The focused remote bundle must pass both source-bound verifiers and
  `sha256sum -c manifest.sha256`.

## Claim Boundary

This diagnostic can establish where wall time is charged and whether the
batch-4 baseline is repeatable. It cannot by itself establish GPU kernel
duration, CUDA Graph speedup, 4K/16K/32K performance, offload benefit, a
second learned model structure, or Phase-1 promotion.
