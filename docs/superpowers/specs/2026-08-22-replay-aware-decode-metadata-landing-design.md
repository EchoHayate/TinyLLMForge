# Replay-Aware Decode Metadata Landing Design

**Date:** 2026-08-22  
**Status:** Approved under the standing autonomous-optimization authorization  
**Stage-1 model:** Qwen3-0.6B  
**Primary target:** batch-1 decode CUDA Graph TPOT and host-side tail latency

## Objective

Reduce steady-state decode overhead by removing avoidable per-token metadata
allocation, redundant device copies, and blanket clearing of CUDA Graph replay
buffers. The optimization must preserve exact output behavior and must report
both benefit and cost.

This is a new runtime optimization for TinyLLMForge. It is not an extension of
the failed Prefix Cache or Chunked Prefill Stage-1 policies, and it does not
authorize a Qwen3-8B claim before the Qwen3-0.6B gate passes.

## Current bottleneck

The ordinary decode path currently performs all of the following on every
token:

1. Build Python lists for input IDs, positions, slot mappings, context lengths,
   and padded block-table rows.
2. Materialize a pageable CPU tensor for each list.
3. Copy each pageable tensor into a reusable pinned-host buffer.
4. Allocate a new CUDA tensor for each metadata field through `.cuda()`.
5. Copy those temporary CUDA tensors into captured CUDA Graph buffers.
6. Clear every captured input and output buffer with `zero_()`, including
   regions that the selected graph cannot read and output storage that the
   graph overwrites completely.

For batch-1 decode, all live scalar fields are overwritten each step.
`context_lens` bounds the valid block-table prefix, and the graph overwrites
the active output row. Clearing the complete static arena therefore does not
carry semantic information.

## Considered approaches

### A. Replay-aware direct metadata landing

Stage pinned-host metadata directly into the existing CUDA Graph input
buffers, overwrite only active and readable slices, and replay without blanket
clears.

Advantages:

- attacks observed work in the exact batch-1 graph path;
- preserves the existing model, scheduler, KV layout, and graph identity;
- has a small, auditable correctness boundary;
- can be disabled and compared against the unchanged baseline.

Costs and risks:

- retains bounded pinned-host staging capacity;
- requires strict shape and lifetime validation;
- asynchronous host-to-device copies must complete in stream order before
  replay;
- stale inactive bytes must never become readable after shape or dispatch
  changes.

### B. Stable-lane batching

Keep request-to-lane placement stable across decode steps to increase
multi-sequence graph reuse.

This may improve graph hit rate, but it introduces padding work, queueing
policy, fairness effects, and a substantially larger benchmark matrix. It is
deferred.

### C. Asynchronous next-step scheduling

Prepare the next host-side schedule and metadata while the current GPU step is
running.

This has a high theoretical ceiling, but sampled tokens, sequence completion,
KV allocation, and transaction commit all affect the next step. It is deferred
until a safe dependency boundary is established.

## Selected design

Implement approach A behind a default-disabled configuration flag named
`replay_aware_decode_metadata`.

### 1. Decode metadata plan

Introduce a dependency-light value object representing the host-side decode
metadata:

- input IDs;
- positions;
- slot mappings;
- context lengths;
- padded block-table rows;
- active batch size;
- readable page-table width.

Construction remains deterministic from the scheduled `Sequence` objects. The
plan owns no CUDA tensor and carries no model-specific policy.

### 2. Reusable pinned-host staging

Add a bounded staging arena owned by `ModelRunner`. It provides typed,
capacity-growing pinned buffers for:

- `int64`: input IDs and positions;
- `int32`: slot mappings, context lengths, and block tables.

The arena writes Python values directly into pinned storage without first
constructing an intermediate pageable `torch.tensor`. Capacity may grow but
never shrinks during a runner lifetime. The arena reports:

- current capacity bytes;
- peak capacity bytes;
- allocation and growth counts;
- bytes staged per step.

The first implementation may use one buffer per field. Packing fields is not
required for Stage 1 because it complicates alignment and view ownership
without first proving that launch count is limiting.

### 3. Replay dispatch

When all of the following are true:

- the flag is enabled;
- execution is ordinary decode;
- active batch size is exactly one;
- the selected legacy CUDA Graph has exactly one active row;
- no KV offload, Quest, compact attention, CPU offload, quantized eager-only
  path, input embedding, or hidden-state return is active;

the runner:

1. copies pinned input IDs, positions, slot mappings, and context lengths
   directly into the corresponding active graph slices;
2. copies only the valid block-table prefix into the active graph row;
3. does not clear graph outputs because replay overwrites the active output;
4. does not clear inactive graph rows because an exact batch-1 graph has none;
5. replays the graph on the same stream after the copies.

The selected graph size must equal the active batch size. A padded graph is not
eligible in Stage 1.

### 4. Fallback

Any unsupported mode, shape mismatch, absent graph state, staging failure, or
identity ambiguity uses the existing path without changing output semantics.
The optimization must fail closed. A fallback event records a stable reason.

The default-disabled flag guarantees that existing users retain current
behavior until the benchmark gate establishes a benefit.

### 5. Observability

Expose a per-run summary with:

- eligible decode steps;
- optimized replay steps;
- fallback counts by reason;
- pinned capacity and peak bytes;
- staging growth count;
- staged host-to-device bytes;
- avoided temporary CUDA tensor count;
- avoided blanket-zero bytes;
- selected graph batch size and page-table width.

Counters are evidence, not proof of speedup. Performance authority comes only
from the paired benchmark.

## Correctness invariants

The optimization must preserve:

- generated token IDs exactly;
- final sequence output text exactly;
- per-step input IDs, positions, slot mappings, context lengths, and readable
  block-table entries;
- KV block ownership and write locations;
- scheduler queue order and completion state;
- graph selection and graph replay count;
- eager and unsupported-mode behavior.

Stale bytes outside the readable block-table prefix are allowed only when the
active attention path proves they are unreachable through `context_lens`.
Tests must fail if the optimized path is used with a padded graph or a
non-batch-1 request.

## Testing strategy

Follow test-driven development.

### Unit tests

Add dependency-light tests that first fail and then prove:

1. the metadata plan exactly represents batch-1 decode inputs;
2. staging reuses pinned buffers and grows capacity monotonically;
3. direct landing overwrites every readable field;
4. stale block-table suffix entries are not cleared and are outside the
   readable prefix;
5. outputs are not zeroed before replay;
6. no temporary CUDA tensor conversion occurs on the optimized path;
7. padded graphs, batch sizes above one, and unsupported modes fall back;
8. telemetry counters and byte accounting are exact;
9. the disabled flag preserves the current path.

Existing model-runner, CUDA Graph, scheduler, prefix, chunked-prefill, and
speculative-runtime tests remain required regression coverage.

### Remote correctness smoke

On the approved remote host, run baseline and candidate with identical:

- source-bound Qwen3-0.6B checkpoint and tokenizer;
- prompts, sampling parameters, and random seeds;
- batch size one;
- prompt and output lengths;
- graph warmup and measured-step inventory.

Require exact token equality for every measured request and no graph, KV, or
runtime error.

## Stage-1 benchmark

Use immutable run tags and store all remote task data under:

`/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`

Do not use remote `/`, `/tmp`, `/private/tmp`, or the legacy checkout as a task
output location.

Run paired OFF/ON repetitions for at least:

- short context: 256 prompt tokens, 128 generated tokens;
- medium context: 2048 prompt tokens, 128 generated tokens;
- long context: 8192 prompt tokens, 128 generated tokens;
- batch size: one;
- five measured repetitions per case after warmup;
- alternating policy order to reduce temporal bias.

Record:

- TTFT;
- per-token latency median, P95, and P99;
- end-to-end latency;
- output tokens per second;
- host wall time and CUDA time for decode steps;
- peak allocated and reserved CUDA memory;
- pinned-host capacity;
- optimized/fallback step counts;
- temporary allocation and zeroing counters.

## Gate

The candidate is `GO_REPLAY_AWARE_METADATA` only if all requirements hold:

1. exact token and text equality in every pair;
2. optimized replay is observed on every measured decode step after warmup;
3. median TPOT improves by at least 5% in at least two context buckets;
4. aggregate P95 TPOT improves by at least 5%;
5. no context bucket regresses median or P95 TPOT by more than 3%;
6. TTFT and end-to-end latency do not regress by more than 3%;
7. throughput does not regress by more than 2%;
8. peak CUDA reserved memory does not regress by more than 1%;
9. pinned-host cost is reported and remains bounded by the configured graph
   metadata capacity;
10. both the producer gate and an independent verifier agree.

Otherwise the result is a specific NO-GO classification. A correct but
non-beneficial result is not promotable.

## Promotion boundary

Stage 1 proves only Qwen3-0.6B batch-1 decode behavior on the tested host and
workloads.

Only a Stage-1 GO permits:

- enabling the feature by default for the proven scope;
- a separate batch-size sweep;
- a Qwen3-8B validation;
- extension to exact multi-sequence graph entries.

No Qwen3-8B or general serving claim may be derived from a Qwen3-0.6B result.

## Deliverables

- default-disabled runtime implementation;
- focused unit and regression tests;
- source-bound paired benchmark runner;
- independent verifier;
- immutable raw artifacts and manifest;
- benefit-and-cost report;
- audit and handoff updates;
- exact-path commits pushed to `origin/feat/kv-sparse-attention`.
