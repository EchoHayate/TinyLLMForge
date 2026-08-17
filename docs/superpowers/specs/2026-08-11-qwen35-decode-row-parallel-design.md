# Qwen3.5 Decode Row-Parallel Projection Design

## Objective

Reduce Qwen3.5 TP4 decode cost by replacing the correctness-first
`ReplicatedWeightRowParallelLinear` layout used by attention output
projections with true input-dimension weight sharding.

This work targets the approximately 24
`replicated_weight_row_parallel_all_gather` operations observed per decode
step in the r620 profile. It does not further optimize hybrid-prefix restore,
whose remaining contribution is already small relative to steady decode.

## Scope

The first implementation phase covers exactly two Qwen3.5 projection sites:

- linear-attention `out_proj`
- full-attention `output_projection`

It may update:

- `tinyvllm/models/qwen35_components.py`
- `tinyvllm/models/qwen35_checkpoint_binding.py`
- focused tests for row-parallel loading, component construction, checkpoint
  binding, collective profiling, and output equivalence
- the TP4 decode profiling runner or report assembler only when necessary to
  expose the new collective names and local weight shapes

The existing `ReplicatedWeightRowParallelLinear` class remains available.
This phase does not migrate unrelated model families or redesign the complete
decoder hidden-state layout.

## Current Problem

Each affected rank currently owns:

- a local input activation shard with width `input_size / tp_size`
- a complete projection weight with shape `[output_size, input_size]`

Before every projection, each rank gathers all input shards, concatenates
them into the global input, and executes the complete projection:

```text
local input
  -> AllGather all input shards
  -> concatenate global input
  -> full-weight GEMM on every rank
  -> replicated output
```

For TP4 decode, the r620 evidence records approximately 24 such AllGather
operations per generated token. This also duplicates the projection weight
and projection arithmetic on every rank.

## Considered Approaches

### 1. True row-parallel weight sharding

Each rank stores a contiguous input-dimension shard of the weight:

```text
local input x local weight
  -> local partial output
  -> AllReduce partial outputs
  -> replicated output
```

Advantages:

- removes the 24 input AllGather operations
- stores one quarter of each affected weight on every TP4 rank
- executes one quarter of each affected projection GEMM on every rank
- reuses the existing `RowParallelLinear` implementation and loader contract
- preserves a replicated decoder hidden state after the projection

Cost:

- replaces each AllGather with an AllReduce rather than eliminating all
  communication
- floating-point summation order changes, so intermediate outputs need
  tolerance-based comparison rather than bitwise equality

This is the selected first phase.

### 2. End-to-end sharded hidden state

Keep projection output, residual, normalization, and following projections
sharded across ranks, using ReduceScatter only at selected boundaries.

This could reduce or defer collective calls, but it changes the entire
decoder-block ownership model and affects residual paths, RMSNorm, MLP,
embedding, and checkpoint assumptions. It is deferred until phase-one
profiling proves that row-parallel AllReduce remains the dominant decode
cost.

### 3. Optimize the existing AllGather

Use `all_gather_into_tensor`, persistent buffers, or lower-overhead launch
plumbing without changing the weight layout.

This is lower risk but retains complete weights, complete GEMMs, and 24
serial input gathers. It is not selected because it cannot address the
primary structural inefficiency.

## Architecture

### Projection construction

`build_qwen35_concrete_components()` constructs both affected projections as
`RowParallelLinear` instances.

For TP4:

- full-attention local input is the contiguous local query-head output
- linear-attention local input is the contiguous local value-head output
- each projection weight has shape
  `[hidden_size, global_input_width / 4]`

The local activation order must match the contiguous checkpoint column shard
selected for the same TP rank. Focused tests must prove this ordering for
both projection families.

### Checkpoint binding

The Qwen3.5 checkpoint binding contract recognizes both affected targets as
`RowParallelLinear`.

The existing custom parameter loader slices the transformed source tensor
along axis 1:

```text
source[:, rank * local_width : (rank + 1) * local_width]
```

Binding validation must require:

- the global source shape remains `[output_size, input_size]`
- `input_size` is divisible by `tensor_parallel_size`
- the local destination shape is
  `[output_size, input_size / tensor_parallel_size]`
- the selected local columns exactly match the activation shard owned by the
  same rank

No checkpoint file or canonical manifest schema changes are required.

### Forward data flow

For each affected projection:

1. The attention implementation produces a rank-local activation shard.
2. `RowParallelLinear` multiplies the local activation by the local weight.
3. Bias is applied on rank zero only when present.
4. `row_parallel_all_reduce` sums partial outputs across ranks.
5. Every rank receives the same replicated hidden-size output.
6. Existing decoder residual and normalization paths continue unchanged.

This phase deliberately preserves the replicated hidden-state boundary.

## Profiling and Observability

The decode profiler must distinguish:

- legacy `replicated_weight_row_parallel_all_gather`
- replacement `row_parallel_all_reduce`
- existing `vocab_parallel_embedding_all_reduce`

The TP4 run must report, per rank and per steady decode step:

- count and CUDA time for every collective name
- total step wall and CUDA time
- local projection weight shapes
- output token parity

`step_wall_ns - step_cuda_ns` remains only an upper bound for host
orchestration, launch gaps, and possible synchronization waiting.

## Correctness Gates

### CPU or synthetic gates

Focused tests must cover:

1. A global source weight is sliced into the expected axis-1 local shards.
2. The local weight shape is exactly one TP partition.
3. Summed local projection outputs match the unpartitioned reference within
   dtype-appropriate tolerance.
4. Full-attention head ownership maps to the matching checkpoint columns.
5. Linear-attention value-head ownership maps to the matching checkpoint
   columns.
6. Qwen3.5 component assembly constructs both targets as
   `RowParallelLinear`.
7. Checkpoint binding accepts the new layout and rejects replicated or
   incorrectly sized destinations.
8. Profiler wiring records `row_parallel_all_reduce` and no longer records
   the legacy AllGather for the migrated projections.

### Real TP4 gates

Use GPUs `2,4,5,6` only. Before each run, every selected GPU must have at
least 25 GiB free memory and utilization no greater than 10 percent.
Unrelated low-utilization processes are allowed; all results must be labeled
shared and non-exclusive.

The real checkpoint run must demonstrate:

- successful load on all four ranks
- no `replicated_weight_row_parallel_all_gather` rows from the two migrated
  projection families
- the expected `row_parallel_all_reduce` rows
- equal generated token IDs for all controlled before/after request pairs
- no NaN or Inf in checked logits or reported tensors
- clean shutdown without killing unrelated processes

Because the reduction order changes, internal hidden states and logits may
use an explicitly reported BF16 tolerance. Generated-token parity remains a
hard gate for the controlled cases.

## Performance Gates

The before and after runs must use:

- the same model and checkpoint
- the same prompts and output lengths
- the same TP size and GPUs
- the same warmup policy
- a fresh attempt tag that preserves all earlier attempts
- at least three measured repetitions per compared case

Report median and dispersion for:

- steady decode wall time per step
- steady decode CUDA time per step
- collective CUDA time per step
- TPOT and output tokens per second

Performance classification:

- `PERFORMANCE_PASS`: steady decode wall or TPOT improves by at least 5
  percent, steady decode CUDA moves in the same direction, parity passes,
  and no reported tail metric regresses by more than 2 percent
- `STRUCTURAL_ONLY`: correctness and structural gates pass, but the measured
  speedup is below 5 percent or is noisy
- `NO_GO`: parity fails, loading is incorrect, stability fails, or steady
  decode materially regresses

A structural reduction in weight memory, GEMM work, or legacy AllGather
count is not by itself sufficient for a performance claim.

## Failure Handling

- If local activation ordering does not match contiguous checkpoint column
  shards, stop and design a projection-specific loader; do not silently
  reorder checkpoint weights.
- If `RowParallelLinear` changes unrelated Qwen3.5 bindings, use explicit
  target suffixes rather than broad type rules.
- If AllReduce is slower than the legacy AllGather path, retain the evidence
  and classify the attempt as `NO_GO` or `STRUCTURAL_ONLY`; do not claim an
  E2E speedup.
- If the runtime cannot collect trustworthy repeated measurements because of
  shared-GPU interference, preserve the attempt and rerun with a new tag.
- Cleanup may target only processes carrying the current attempt tag.

## Deferred Work

The following are explicitly outside this phase:

- end-to-end hidden-state sharding
- ReduceScatter-based residual ownership
- collective fusion across sequential decoder layers
- changes to hybrid-prefix publication or restore
- changes to canonical manifests, case-matrix schemas, or existing r607-r620
  artifacts
- migration of unrelated `ReplicatedWeightRowParallelLinear` users

## Deliverables

1. Focused implementation and tests for true row-parallel Qwen3.5 output
   projections.
2. A new TP4 performance attempt with structured collective evidence.
3. Before/after correctness and performance classification.
4. A completion audit mapping every design gate to an artifact.
5. Updated `AGENT_HANDOFF_STATE.md` containing the result, limitations, and
   next recommended optimization.
