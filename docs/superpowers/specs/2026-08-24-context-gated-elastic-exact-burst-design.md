# Context-Gated Elastic Exact-Burst Design

**Date:** 2026-08-24

**Status:** Approved under the standing autonomous-optimization authorization

**Stage-1 model:** Qwen3-0.6B

**Primary target:** reduce completion-only batch-1 greedy-decode TPOT beyond
the current fixed K8 path while preserving the existing 40 ms host-visible
burst-gap limit

## Objective

Add a default-disabled exact-burst policy that selects K16 for short and
medium contexts and falls back to K8 for long contexts or any request that
cannot safely authorize sixteen consecutive writes.

The optimization must preserve:

- exact output-token, sampled-logit, and argmax equality;
- one target-model forward and one CUDA Graph replay per emitted token;
- one final token D2H per burst and no intermediate token D2H;
- completion-only, temperature-zero, `ignore_eos=true`, batch-one
  eligibility;
- scheduler ownership, rollback, quarantine, and fail-closed behavior;
- the 40 ms maximum host-visible burst-gap gate; and
- unchanged request ordering, fairness, model weights, attention math, and KV
  physical layout.

This is a runtime-data-flow-specific original engineering design, not a claim
of academic novelty.

## Motivation and Measured Ceiling

The canonical exact-burst gate selected K8 over K4. Its paired Qwen3-0.6B
results showed:

```text
K8 versus K4 aggregate median TPOT improvement: 6.562618%
K8 versus K4 aggregate P95 TPOT improvement:    9.329137%
K8 maximum host-visible burst gap:              24.035218 ms
```

Using the five paired repetitions from the same source-bound artifact and a
simple `per_token_cost + fixed_burst_cost / K` decomposition gives the
following read-only K16 ceiling estimate:

| Context | K8 median TPOT | Projected K16 TPOT | Projected gain | Projected K16 gap |
| --- | ---: | ---: | ---: | ---: |
| short | 2.160 ms | 2.086 ms | 3.441% | 33.371 ms |
| medium | 2.447 ms | 2.358 ms | 3.657% | 37.724 ms |
| long | 2.995 ms | 2.859 ms | 4.549% | 45.743 ms |

The projection is not performance evidence. It identifies a plausible
benefit and also rejects unconditional K16 because the long-context estimate
exceeds the frozen 40 ms visibility limit.

## Considered Approaches

### A. Context-gated K16 with a shared one-token graph

Reuse the existing complete-step CUDA Graph and select sixteen ordered replays
only when the request is already eligible for exact burst, the initial context
length is at most 2,048 tokens, the current physical block has room for
sixteen writes, at least sixteen output tokens remain, and the K16 policy
health epoch is not quarantined. Otherwise select K8 using the existing path.

Benefits:

- targets a projected 3% to 4% additional TPOT reduction where the visibility
  budget appears feasible;
- keeps selection deterministic and inspectable;
- preserves the proven K8 path as a complete fallback;
- changes no numerical operation;
- adds no duplicate graph capture, retained model output, or static tensor
  allocation.

Costs:

- larger host-visible output batches for selected requests;
- K16-specific lifecycle and transaction coverage;
- one small host-side width-health record and additional counters.

This is the selected approach.

### B. Online latency-predicted burst width

Choose K8 or K16 from a moving estimate of replay latency.

Benefits:

- could adapt to transient hardware load;
- could admit some long-context K16 bursts.

Costs:

- the first K16 observation is not protected by prior evidence;
- selection becomes history-dependent and harder to reproduce;
- noisy estimates can oscillate at the policy boundary;
- a prediction is not a hard deadline guarantee.

This approach is rejected for the first implementation.

### C. Two-burst GPU/host commit pipeline

Launch burst N+1 while the host commits burst N.

Benefits:

- potentially overlaps most host postprocessing rather than merely
  amortizing it.

Costs:

- introduces cross-burst speculative ownership;
- complicates cancellation, rollback, output visibility, and stop handling;
- may execute tokens that cannot be committed;
- requires a substantially larger correctness proof.

This approach is deferred until the simpler elastic-width ceiling is known.

### D. Separately captured K16 graph

Capture a second copy of the same complete-step graph and bind K16 leases to
that copy.

Benefits:

- graph-level quarantine is naturally independent.

Costs:

- the graph body still emits one token per replay, so the second capture does
  not reduce K16 replay work;
- duplicates capture time, retained outputs, static tensors, and graph
  bookkeeping;
- makes a width policy look like a numerically distinct execution graph when
  it is not.

This approach is rejected. Width-specific health is sufficient to isolate K16
failures while retaining the already-proven graph owner.

## Architecture

### 1. Policy

Add a default-disabled flag:

```text
exact_greedy_decode_burst_elastic_k16
```

It requires:

```text
exact_greedy_decode_burst = true
exact_greedy_decode_burst_tokens = 8
exact_greedy_decode_burst_split_phase = false
```

The first implementation does not compose with split phase or ragged
coalescing. It may compose with continuation, one-phase lease-local journal,
and generation-sealed identity only after focused tests prove each contract.

### 2. Deterministic selector

The selector evaluates K16 before the existing K8 decision. K16 is requested
only when all of the following hold:

```text
initial_sequence_length <= 2048
remaining_output_tokens >= 16
writable positions in the current physical block >= 16
shared complete-step graph is available and not quarantined
K16 policy health epoch is not quarantined
no incompatible mode is active
```

Any failed K16 condition falls back to the current K8 selector. A K16
width-policy, D2H, result, visibility, or commit validation failure quarantines
only the K16 policy-health epoch and retries no work inside the same step; the
scheduler cancels the lease and follows the existing safe K8 fallback on the
next engine step. A failure that proves the shared graph itself invalid
continues to quarantine the graph and therefore disables both widths.

The context threshold is frozen before the terminal gate. It is not tuned
from terminal measurements.

### 3. Shared graph ownership and width-scoped health

K8 and K16 use the same captured complete-step graph, static tensors, token
history, and capture receipt. The graph already executes one complete token
step per replay and its history capacity is at least one physical block, so
K16 changes only the authorized replay count.

The lease digest binds the requested and authorized width. A host-side
`ElasticBurstWidthHealth` record owns a monotonically increasing generation,
an optional K16 quarantine reason, and K16 attempt/accept/fallback counters.
The lease captures this width-health generation. Validation rejects stale
width generations before graph state mutation.

K16 performs exactly sixteen ordered replays of the same complete-step graph.
The final device token remains the next autoregressive input during the
burst. Only the completed sixteen-token history slice is copied to the host.

A failure attributable to K16 policy limits, host visibility, result
validation, or width-aware commit quarantines only the K16 health record. A
failure attributable to graph capture identity, graph replay integrity,
static tensors, or live KV identity quarantines the shared graph and therefore
both widths.

### 4. Scheduler transaction

The one-phase lease-local journal is generalized from an exact width of eight
to an allowed width set `{8, 16}`. Its state remains bounded by the burst
width and by at most one write block because K16 admission is clipped at the
current physical block boundary.

The transaction records the actual authorized width. Commit, rollback,
publication, continuation, and counters must use that width rather than a
hard-coded eight.

### 5. Observability

Record:

- requested and authorized width histograms;
- K16 attempts, acceptances, K8 fallbacks, and fallback reasons;
- shared graph identity and capture receipt;
- width-health generation and K16 quarantine reason;
- per-burst host-visible gaps;
- per-context TPOT median, P95, and P99;
- TTFT, E2E, throughput, allocated and reserved CUDA memory;
- target forwards, graph replays, D2H calls and bytes;
- journal attempts, captures, commits, fallbacks, and rollbacks.

## Correctness and Failure Handling

K16 is allowed only for the same completion-only semantics as the current
exact-burst path. EOS inspection, stop strings, token-level callbacks,
streaming, sampling, mixed batches, and multiple active sequences remain
ineligible.

Every failure is fail-closed:

- missing or quarantined shared graph: use the existing ordinary path;
- quarantined K16 policy health: choose K8 before lease creation;
- stale schedule, graph, block table, ownership, or lease identity: reject
  before mutation;
- replay/static-state failure: quarantine the shared graph and cancel the
  pending lease;
- K16 D2H/result/width validation failure: quarantine K16 policy health and
  commit no host metadata;
- prepare or commit failure: use the existing bounded rollback owner;
- rollback failure: surface the compound failure and quarantine the path.

No synchronous per-token D2H fallback is allowed after any K16 replay.

## Validation

### CPU contract gate

Focused tests must prove:

- exact boundary behavior at context 2,048 and 2,049;
- output-budget and physical-block clipping;
- deterministic K16-to-K8 fallback reasons;
- K8 behavior is byte-for-byte unchanged when the flag is off;
- K16 lease/result serialization and identity binding;
- width-aware continuation and one-phase journal behavior;
- stale and fault-injection paths mutate no unauthorized state.

### GPU ceiling probe

Before full implementation promotion, run a source-bound K8/K16 ceiling probe
with Qwen3-0.6B, TP1, batch one, temperature zero, `ignore_eos=true`, 128
generated tokens, and fixed 256, 2,048, 4,096, and 8,192-token prompts.

The 4,096-token and 8,192-token points are K8 controls and must never select
K16.

Proceed to the full gate only if:

```text
short-or-medium K16 median TPOT improvement versus K8 >= 1.5%
maximum K16 host-visible burst gap <= 40 ms
exact output tokens and sampled logits
forwards and replays equal emitted tokens
one final D2H call per burst with token bytes equal emitted-token bytes
```

### Terminal paired gate

Use five repetitions with alternating order for:

```text
fixed_k8
context_gated_elastic_k16
```

Required inventory:

```text
2 policies x 4 contexts x 5 repetitions = 40 performance rows
2 policies x 4 contexts x 4 sample points = 32 correctness rows
```

Promotion requires:

- exact output-token, decoded-text, sampled-logit, and argmax equality;
- K16 selected for every eligible 256/2,048-token row and never at
  4,096/8,192 tokens;
- aggregate eligible-context median TPOT improvement at least 2%;
- aggregate eligible-context P95 TPOT improvement at least 1%;
- no context median or P95 TPOT regression above 2%;
- maximum host-visible burst gap at most 40 ms;
- TTFT, E2E, and TPOT-P99 regression at most 2%;
- throughput regression at most 1%;
- allocated and reserved CUDA-memory regression at most 3%;
- target forwards equal emitted tokens;
- graph replays equal emitted tokens;
- one final token D2H per burst and zero intermediate token D2H;
- zero unexpected fallback, rollback, or quarantine events;
- complete source manifest and agreeing producer, remote verifier, and
  frozen-source local verifier receipts.

## Benefit and Cost Reporting

The terminal report must present both:

- benefit: TPOT median/P95/P99 and throughput changes overall and by context;
- cost: maximum/p95 host-visible gap, incremental capture duration,
  incremental retained static bytes, peak CUDA memory, K8 fallback rate, and
  K16 width-health quarantine count.

A correct implementation that misses the positive TPOT thresholds is
`NO_GO_INSUFFICIENT_INCREMENTAL_BENEFIT`. A faster implementation that exceeds
40 ms is `NO_GO_BURST_GAP`. Neither result authorizes Qwen3-8B or production
default enablement.

## Claim Boundary

A successful Stage-1 gate establishes only a Qwen3-0.6B TP1 batch-one,
completion-only, temperature-zero, ignore-EOS elastic-width result. It does
not establish streaming, EOS-aware generation, multi-sequence scheduling,
tensor parallelism, Qwen3-8B benefit, or production readiness.
