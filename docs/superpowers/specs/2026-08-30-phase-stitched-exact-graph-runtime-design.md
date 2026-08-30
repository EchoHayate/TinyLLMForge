# Phase-Stitched Exact Graph Runtime Design

**Date:** 2026-08-30
**Status:** Approved direction; profile gate required before runtime mutation
**Stage-1 model:** Qwen3-0.6B
**Primary target:** batch-one, completion-only, exact greedy generation

## Objective

Reduce end-to-end latency at the prefill-to-decode boundary by connecting the
existing Exact Prefill CUDA Graph directly to the existing Exact Greedy Decode
Burst runtime.

The candidate must preserve:

- exact output token IDs and text;
- one full target-model forward for every decoded token;
- ordinary Scheduler ownership of sequence state and KV allocation;
- internal host visibility of the first token without waiting for the complete
  decode burst;
- fail-closed behavior before any graph mutates live KV;
- quarantine and request failure after an authoritative replay starts;
- default-disabled configuration and eager fallback for unsupported requests.

This is a new TinyLLMForge execution protocol assembled from existing
mechanisms. CUDA Graph replay, GPU-resident greedy feedback, and multi-token
decode are not claimed as new primitives. The proposed contribution is the
cross-phase lease, visibility protocol, and exact end-to-end composition.

## Current boundary

The current optimized path still contains two independent engine steps:

```text
final prefill
  -> Exact Prefill CUDA Graph replay
  -> graph-external LM head and float32 argmax
  -> first-token D2H
  -> Scheduler prefill commit and sequence requeue
  -> next schedule decision
  -> Exact Greedy Decode Burst lease construction
  -> K8 graph replay
```

The completed Exact Prefill gate shows large TTFT reductions, while the
completed Exact Greedy K8 gate shows large steady-decode reductions. Neither
result proves that joining the two paths is beneficial. The removable
prefill-to-first-decode handoff must be measured directly before adding a
cross-phase transaction.

## Related systems direction

Recent systems work motivates the design without establishing its benefit in
TinyLLMForge:

- NanoFlow overlaps fine-grained inference operations instead of treating a
  model iteration as one indivisible scheduling unit.
- Blink moves steady-state control toward the GPU to reduce host submission
  and synchronization overhead.
- Foundry materializes CUDA Graph execution contexts offline to reduce graph
  setup cost.
- DeepEP overlaps communication and computation for MoE serving.

This design adopts only the general principle of removing avoidable
host-controlled phase boundaries. It does not import model-specific kernels,
approximate attention, or MoE assumptions.

## Considered approaches

### A. Phase-stitched exact graph runtime

Before the final prefill starts, the Scheduler creates a bounded stitch lease.
The prefill graph produces the first exact greedy token into device-resident
storage. That token becomes the first input of a bounded decode burst without
rebuilding ordinary decode metadata on the host.

The first token is copied through a dedicated visibility transfer and becomes
available for the Scheduler prefix commit as soon as its CUDA event completes.
Remaining burst tokens finish through an asynchronous mailbox and are
committed later under the same parent lease. This internal two-phase commit
does not enable ordinary per-token callbacks or external streaming.

Advantages:

- removes the ordinary post-prefill requeue and next-step scheduling gap from
  the GPU critical path;
- reuses two already validated graph mechanisms;
- preserves first-token visibility instead of delaying it until the burst
  ends;
- supports a four-arm benchmark that isolates composition benefit.

Costs and risks:

- the Scheduler must authorize future decode KV writes before prefill;
- prefill and decode graph identities become jointly bound;
- the first-token publication and suffix commit form a two-phase transaction;
- failures after prefill replay starts cannot use eager retry;
- retained static buffers and reserved KV capacity increase.

This is the selected approach, contingent on the profile gate.

### B. Offline CUDA Graph materialization

Persist graph topology and execution context during build or deployment, then
reconstruct it at service startup.

This directly targets the measured approximately 728 ms capture cost, but it
does not improve steady-state TTFT, TPOT, or end-to-end generation latency.
It also introduces strong CUDA-driver, GPU-identity, binary, and address-layout
compatibility requirements. It remains a later cold-start project.

### C. Contract-guided kernel autotuning

Generate or select multiple contract-compatible implementations for small hot
kernels, verify exactness, and retain the fastest implementation per shape.

This could improve LM-head, argmax, metadata packing, or normalization costs,
but it first requires a reusable kernel-contract and compilation service. It is
broader infrastructure with less certain immediate end-to-end value, so it is
not selected for this iteration.

## Stage 0: profile-only admission gate

No runtime stitching code may be implemented until a source-bound profile
shows sufficient removable work.

### Measurement points

For each request, record synchronized timestamps for:

1. final prefill graph replay submitted;
2. final prefill CUDA work completed;
3. first token available on the host;
4. Scheduler prefill commit completed;
5. next decode schedule decision completed;
6. K8 lease prepared;
7. first K8 graph replay submitted;
8. first K8 CUDA work started, when observable without intrusive profiling.

The primary removable interval is:

```text
first K8 replay submission - final prefill CUDA completion
```

Secondary decomposition reports:

- first-token D2H latency;
- Scheduler prefill commit;
- queue/reselection;
- K8 lease preparation;
- decode metadata landing;
- unclassified residual.

Instrumentation must be disabled by default and must not add a synchronization
to either benchmark arm.

### Profile workload

- Qwen3-0.6B BF16;
- A100 80GB PCIe;
- TP1, batch one;
- greedy temperature zero;
- `ignore_eos=true`;
- completion-only visibility;
- prompt lengths 256 and 2048;
- 128 generated tokens;
- Exact Prefill Graph and Exact Greedy K8 both enabled;
- two repetitions with reversed execution order;
- fresh engine per measured case.

### Profile GO threshold

Proceed to runtime implementation only when all of these hold:

- median removable interval is at least 0.15 ms for one prompt shape;
- the interval is at least 3% of measured end-to-end time for one prompt
  shape, or its P95 is at least 0.50 ms;
- timing coverage is complete for every retained request;
- instrumentation-on versus instrumentation-off changes median E2E by no more
  than 1%;
- output token IDs remain exact.

Otherwise classify the proposal `NO_GO_PHASE_STITCH_CEILING` and stop after
publishing the profile evidence.

## Stitch lease

The Scheduler owns an immutable `PhaseStitchLease` created before final
prefill dispatch. It binds:

- sequence ID and sequence generation;
- schedule generation;
- prefill graph identity and generation;
- decode-burst graph identity and generation;
- prompt token count and final prefill KV interval;
- initial completion count and remaining output budget;
- first generated token authority;
- authorized decode replay count;
- exact decode KV write interval and block generations;
- output-visibility policy;
- source identity and runtime feature identity.

The initial candidate emits one token from prefill and authorizes seven decode
replays, producing one eight-token parent transaction. This keeps the maximum
visibility envelope comparable with the existing K8 policy.

The lease is admitted only when:

- both underlying graph components are independently available;
- final prefill is an exact allowlisted shape;
- there is exactly one running sequence and no competing waiting or
  prefilling request;
- sampling is greedy with numeric temperature zero;
- EOS, stop strings, callbacks, per-step logits, and token streaming beyond
  the dedicated first-token publication are disabled;
- the request declares completion-only execution;
- eight output tokens remain;
- all seven decode KV writes fit in the authorized physical block interval;
- TP1/rank-zero and all existing Exact Prefill/K8 compatibility checks pass;
- no speculative, offload, quantized, sparse-attention, mixed-batch, or
  stateful-model path is active;
- no stitch or burst lease is pending.

Every rejection occurs before prefill replay and falls back to the current
independent Prefill Graph plus ordinary K8 path.

## Device execution and visibility

The stitched execution uses retained device tensors:

- prefill hidden output;
- first-token float32 logits and argmax result;
- decode input token;
- decode position, context length, slot mapping, and block table;
- eight-token history;
- replay index and completion count;
- first-token D2H staging storage;
- first-token-ready CUDA event;
- suffix-ready CUDA event or existing split-phase mailbox.

The ordered execution is:

```text
prefill graph replay
  -> LM head
  -> float32 argmax token 0
  -> token 0 history write
  -> token 0 D2H copy + first-token-ready event
  -> seed decode metadata from the stitch lease
  -> seven exact decode graph replays
  -> tokens 1..7 history writes
  -> suffix-ready event
```

The host may wait for and commit token 0 while tokens 1..7 continue on the
device. That prefix commit makes token 0 available to internal latency
instrumentation, but completion-only API delivery remains unchanged.
Committing token 0 does not release or replace the parent lease. The suffix
commit validates the same lease and closes it.

No graph node may read host-mutated sequence state after replay begins.

## Scheduler transaction

The transaction has two visible phases:

1. **Prefix commit:** append exactly token 0, commit the completed prefill KV
   interval, preserve the parent lease, and record internal first-token
   availability without invoking an external per-token callback.
2. **Suffix commit:** append tokens 1..7 in order, publish newly completed KV
   blocks only after the corresponding graph replays finish, then release the
   parent lease.

Both phases use one parent identity. A suffix cannot commit without a
successful prefix commit, and a prefix cannot be committed twice.

The existing Exact Burst split-phase machinery may be reused only through a
new role-based adapter. The stitched path must not masquerade as speculative
output or silently mutate the semantics of the existing K8 lease.

## Failure semantics

### Before prefill replay

Identity drift, insufficient output/KV capacity, unavailable graph entries, or
unsupported request semantics use the current independent path.

### After prefill replay starts

No eager retry is permitted because live prompt KV may have been written.
Any replay, D2H, event, mailbox, validation, or commit failure:

1. quarantines the stitched identity for the runner lifetime;
2. fails the active request;
3. preserves the original exception;
4. prevents duplicate prefill or decode execution;
5. records the last authoritative phase and replay count;
6. rolls back host metadata when possible without claiming the mutated KV is
   reusable.

If token 0 has already been externally published, suffix failure is terminal
and must be reported as partial visibility rather than rolled back as if
nothing was observed.

## Configuration

Add one default-disabled public switch:

```python
phase_stitched_exact_graph_runtime: bool = False
```

The first version inherits the existing prefill allowlist and K8 width. It
does not add dynamic shape buckets, online autotuning, model names, prompt
labels, or GPU-specific policy to core runtime code.

## Four-arm performance gate

Every prompt shape and repetition runs in isolated fresh engines:

| Arm | Exact Prefill Graph | Exact K8 | Phase stitch |
| --- | ---: | ---: | ---: |
| A: eager | off | off | off |
| B: prefill-only | on | off | off |
| C: independent composition | on | on | off |
| D: stitched composition | on | on | on |

The primary causal comparison is D versus C. A and B retain attribution to
the existing components.

The gate reports:

- TTFT median/P95/P99;
- token-0-to-token-1 visible gap;
- amortized TPOT median/P95/P99;
- E2E median/P95/P99;
- output tokens per second;
- host scheduling and phase-handoff intervals;
- graph capture duration and retained static bytes;
- CUDA allocated/reserved deltas;
- preauthorized KV capacity;
- first-token and suffix D2H counts;
- exact token/text equality;
- lease, replay, prefix-commit, suffix-commit, quarantine, and fallback
  counters.

### Runtime GO threshold

`GO_PHASE_STITCHED_EXACT_GRAPH` requires:

- exact output token IDs and text in all retained pairs;
- no capture/replay/transaction failure or quarantine;
- D versus C median E2E improvement of at least 3% for at least one prompt
  shape;
- D versus C aggregate median E2E improvement of at least 2%;
- D versus C token-0-to-token-1 visible-gap improvement of at least 10%;
- no more than 2% TTFT regression for either prompt shape;
- no more than 2% P95/P99 E2E regression for either prompt shape;
- no more than 3% peak reserved-memory regression;
- complete capture, memory, D2H, lease, and visibility accounting;
- independent verifier reconstruction from raw rows and manifest hashes.

If Stage 0 passes but the runtime gate fails, the implementation remains
default-disabled and is classified `NO_GO_PHASE_STITCHED_EXACT_GRAPH`.

## Explicit non-goals

- sampled decoding;
- EOS-aware or stop-string-aware bursts;
- ordinary per-token streaming after token 0;
- multi-sequence or continuous batching;
- tensor parallelism;
- speculative decoding;
- approximate attention or quantization;
- sentinel-filled prefill graph buckets;
- offline CUDA Graph materialization;
- model-specific branches in the generic runtime.

## Evidence and claim boundary

A successful Stage-1 result would establish only that the stitched runtime
improves the frozen Qwen3-0.6B/A100/TP1/batch-one workload relative to the
same TinyLLMForge build with the stitch disabled.

It would not establish:

- superiority over vLLM, SGLang, TensorRT-LLM, or another engine;
- sampled-decoding correctness;
- continuous-batching throughput;
- multi-GPU scaling;
- benefit for arbitrary prompt lengths or model architectures;
- a new CUDA Graph or speculative-decoding primitive.

## References

- NanoFlow, operation-level pipelining for LLM serving:
  `https://arxiv.org/abs/2408.12757`
- Blink, a CPU-free LLM serving engine:
  `https://arxiv.org/abs/2604.07609`
- Foundry, offline materialization of CUDA Graph contexts:
  `https://arxiv.org/abs/2604.06664`
- DeepEP, communication kernels with computation-communication overlap:
  `https://github.com/deepseek-ai/DeepEP`
