# Exact-Burst Octet-Folded Replay Graph Design

Date: 2026-08-24

## Status

Approved for a default-disabled, ceiling-first implementation after:

1. the r10 one-phase lease-local-journal gate is reconciled;
2. generation-sealed block-table identity is completed; and
3. context-gated elastic K8/K16 passes its own ceiling gate.

This is a runtime-data-flow-specific original engineering design. It does
not claim academic novelty.

## Objective

Reduce batch-1 exact-greedy decode TPOT by replacing repeated host launches
of the existing one-token complete-step CUDA Graph with a graph that contains
eight ordered complete-token steps.

The candidate must preserve:

- exact output tokens, decoded text, sampled logits, and argmax;
- the same number and order of target-model token forwards;
- the same KV slots and writes;
- one final token D2H per committed burst;
- scheduler ownership, publication, rollback, and quarantine semantics; and
- the existing one-token graph as the complete fallback.

Every performance conclusion must report both benefit and cost.

## Motivation and Measured Boundary

The current exact-burst graph captures one complete autoregressive token
step. A K8 lease invokes `graph.replay()` eight times, while a K16 lease
invokes it sixteen times. Device-side token feedback already updates the
next input token, position, context length, slot mapping, token history, and
history cursor without intermediate D2H.

The canonical Qwen3-0.6B artifact showed that K8 preserves 127 target-model
forwards and 127 logical graph replays while reducing D2H and scheduler
publication frequency. For the 2K case, K8 reached approximately 2.47 ms
amortized median TPOT and approximately 18.9 ms maximum host-visible burst
gap.

The remaining opportunity is therefore not to remove model work. It is to
submit the same ordered model work with fewer host CUDA Graph launches.

An alternative lease-time pre-armed commit envelope was evaluated first.
A dependency-light CPU ceiling probe measured:

| Context | Current prepare median | Token-dependent residual | Movable ceiling |
| ---: | ---: | ---: | ---: |
| 249 | 28.583 us | 16.626 us | 41.83% |
| 2,041 | 35.166 us | 17.709 us | 49.64% |
| 8,185 | 50.792 us | 25.083 us | 50.62% |

Although the prepare percentage is substantial, the absolute removable work
is only about 12 to 26 microseconds per burst, or roughly 1 to 3 microseconds
per token at K8. That is unlikely to clear a meaningful end-to-end TPOT gate,
so the pre-armed envelope is rejected.

## Selected Approach

### Octet-folded complete-step graph

Capture exactly eight ordered calls to the existing complete-step body in one
CUDA Graph:

```text
input token
  -> target forward
  -> logits
  -> float32 argmax
  -> token history write
  -> next input / position / context / slot update
  -> repeat seven more times inside the same graph
```

At runtime:

- an eligible K8 lease performs one folded graph launch;
- an eligible K16 lease performs two consecutive folded graph launches;
- a width not divisible by eight uses the existing one-token graph;
- correctness tracing uses a separately captured folded correctness graph;
- no host synchronization or D2H occurs between folded launches; and
- final token extraction remains one D2H after all logical token steps.

The folded graph changes launch grouping only. It does not skip, approximate,
fuse, or reorder model math.

## Why Eight

Eight is the accepted base exact-burst width and composes with the planned
elastic K16 policy:

- K8: `8 one-token launches -> 1 folded launch`
- K16: `16 one-token launches -> 2 folded launches`

A K4 folded graph would save fewer launches. A K16 folded graph would require
more retained capture state, would not serve K8 directly, and would duplicate
an additional width-specific graph. Eight therefore provides one reusable
capture unit for both accepted widths.

## Capability and Ownership

### Configuration

Add:

```text
Config.exact_greedy_decode_burst_octet_folded_graph: bool = false
```

When enabled, require:

- `exact_greedy_decode_burst == true`;
- base width exactly eight;
- split phase disabled;
- ragged coalescing disabled; and
- no incompatible graph-capture mode.

Elastic K16 may be either disabled or enabled. The same folded graph serves
both K8 and K16.

### Graph capability

The folded graph owns:

- `steps_per_launch == 8`;
- a graph generation;
- graph identity;
- static tensor identities;
- capture duration;
- allocated and reserved memory deltas;
- retained static bytes;
- production or correctness-trace role; and
- a dedicated health/quarantine record.

The one-token graph remains independently healthy. A folded-graph failure
quarantines only the folded capability and falls back to the one-token graph
on a later engine step. It never retries after partial execution in the same
step.

### Scheduler lease

The lease continues to authorize a logical token count of eight or sixteen.
It does not authorize a graph-launch count.

The model runner deterministically chooses:

```text
folded_launch_count = authorized_token_count // 8
```

only when:

- the authorized count is divisible by eight;
- the folded capability is healthy;
- static tensor and graph identities match;
- history capacity is sufficient; and
- the lease is otherwise valid for the existing one-token graph.

Otherwise it uses the existing one-token graph without changing the lease.

## Capture

### Warmup

Warm up eight ordered complete steps on the scratch block before capture.
Reset all static state after warmup and before capture.

### Capture body

Inside one capture context, invoke the existing `_run_complete_step(...)`
exactly eight times. Flatten and retain the returned outputs in stable step
order.

The graph identity includes:

- `steps_per_launch = 8`;
- graph generation;
- rank and tensor-parallel size;
- block size and scratch block;
- correctness-trace mode;
- sampled-logit ordinals;
- FlashAttention split setting; and
- all static tensor identities.

### Memory accounting

The capture receipt separately records:

- folded capture duration;
- allocated delta bytes;
- reserved delta bytes;
- retained static bytes;
- retained-output tensor count; and
- graph node grouping width.

The design makes no assumption that the CUDA allocator will reuse
intermediate buffers across the eight captured steps. The measured receipt is
authoritative.

The folded graph may share the existing graph memory pool only if the
runtime proves that the one-token and folded graphs are never replayed
concurrently and always follow a deterministic replay order. Otherwise it
uses a separate pool and reports the full incremental cost.

## Replay

### Cold bind

Cold bind remains unchanged:

- reset static state;
- bind initial token, position, context length, and physical slot;
- materialize and copy the block table once; and
- start history at zero.

### Folded launch

For K8, launch the folded graph once.

For K16, launch the same folded graph twice without resetting static state.
The second launch consumes the token, position, context length, slot mapping,
and history cursor produced by the first launch.

Logical counters remain token-based:

- target-model forwards increase by eight per folded launch;
- logical graph replays increase by eight per folded launch; and
- committed tokens remain equal to the lease authorization.

Add physical launch counters:

- `one_token_cuda_graph_launches`;
- `folded_cuda_graph_launches`;
- `folded_logical_steps`;
- `folded_k8_bursts`;
- `folded_k16_bursts`; and
- folded fallback and quarantine reasons.

### Final D2H

After all folded launches:

- copy exactly the authorized token-history slice to host once;
- preserve sampled-logit trace behavior;
- construct the existing result schema with the same token count; and
- validate against the unchanged lease authority.

## Failure Semantics

Before the first folded launch, any capability, identity, or shape mismatch
falls back to the one-token graph.

After any folded launch begins:

- no same-step retry is allowed;
- a launch or D2H failure invalidates folded continuation state;
- the folded capability is quarantined;
- no scheduler token metadata is committed;
- the pending lease follows the existing terminal failure path; and
- later engine steps may use the healthy one-token graph.

For K16, failure during the second folded launch is treated as partial
speculative device execution. The scheduler still commits zero host tokens.
Existing KV ownership remains authoritative, and uncommitted future slots are
not exposed.

## Rejected Alternatives

### A. Pre-armed host commit envelope

Move token-independent journal capture before GPU replay.

Rejected because the local ceiling leaves only about 12 to 26 microseconds
per burst to hide, which is too small relative to measured TPOT.

### B. Dedicated sixteen-step graph

Capture sixteen complete steps and launch once for K16.

Rejected for the first implementation because it duplicates another width,
raises capture and retained-state cost, and cannot serve K8 directly.

### C. CUDA child-graph construction through a custom extension

Construct a parent graph from child graph nodes using lower-level CUDA APIs.

Deferred because it requires private/native graph-handle ownership outside
the current PyTorch abstraction and introduces a larger portability and
build surface than an ordinary capture loop.

### D. Online variable unroll length

Use conditional graph nodes or runtime-patched topology to choose K8/K16 in a
single launch.

Rejected because width selection would become coupled to lower-level CUDA
graph mutation and would weaken the current deterministic, source-auditable
policy.

## Verification Strategy

### Unit tests

Require:

- capture invokes the complete-step body exactly eight times;
- retained outputs preserve all eight step groups;
- graph identity changes when `steps_per_launch` changes;
- K8 selects one folded launch;
- K16 selects two folded launches;
- K1 through K7 and K9 through K15 use the one-token graph;
- folded K16 does not reset state between launches;
- logical forwards/replays equal authorized tokens;
- physical folded launches equal `authorized_tokens / 8`;
- one final token D2H is preserved;
- exact token and sampled-logit trace parity;
- pre-launch mismatch falls back;
- post-launch failure quarantines folded only;
- the one-token graph remains usable after folded quarantine; and
- default-disabled behavior is byte-for-byte unchanged.

### GPU ceiling probe

Before building a full terminal gate, compare:

```text
accepted stack + one-token graph
accepted stack + octet-folded graph
```

Use Qwen3-0.6B, TP1, batch 1, greedy completion-only decode, identical
prompts, outputs, ordering, and source SHA.

The ceiling probe covers K8 at 256, 2,048, and 8,192 prompt tokens. If elastic
K16 has already passed, also cover K16 at 256 and 2,048.

Proceed only if all are true:

- exact token, text, argmax, and sampled-logit parity;
- target-model forwards and logical replay counts unchanged;
- K8 physical CUDA Graph launches reduced by at least 85%;
- K16 physical CUDA Graph launches reduced by at least 80%;
- aggregate median TPOT improvement at least 1.0%;
- aggregate P95 TPOT improvement at least 0.5%;
- no context median or P95 TPOT regression above 2%;
- no TTFT, E2E, TPOT-P99, or throughput regression above 2%;
- folded capture allocated and reserved deltas each no more than 1% of the
  baseline peak for the tested process;
- folded retained static bytes no more than 128 MiB above baseline; and
- folded capture duration no more than 120 seconds.

If any positive latency threshold fails, classify the candidate
`NO_GO_CEILING` and stop. Do not build the terminal gate.

### Terminal paired gate

If the ceiling passes, require at least:

- 40 complete paired performance rows;
- 32 correctness rows;
- producer, remote verifier, and frozen-source local verifier agreement;
- exact source and patch hashes;
- complete launch, logical-step, D2H, memory, capture, and quarantine
  receipts; and
- benefit and cost reported together.

Promotion thresholds:

- aggregate median TPOT improvement at least 1.5%;
- aggregate P95 TPOT improvement at least 1.0%;
- no context median or P95 regression above 2%;
- TTFT, E2E, TPOT-P99, and throughput regression at most 2%;
- CUDA peak allocated and reserved memory regression at most 1%;
- exact parity and unchanged logical target forwards;
- physical launch reduction thresholds unchanged from the ceiling gate; and
- zero unexpected fallback, rollback, or quarantine events.

## Claim Boundary

If the terminal gate passes, the maximum claim is:

> On the tested Qwen3-0.6B TP1 batch-1 greedy workload, an octet-folded
> complete-step CUDA Graph preserved exact outputs and target-model work while
> reducing physical graph launches for K8/K16 and improving measured TPOT
> within the reported capture-time and memory costs.

The gate does not prove:

- benefit for batching, sampling, TP greater than one, or other models;
- benefit when device compute dominates launch overhead;
- production readiness;
- academic novelty; or
- that the folded graph should replace the one-token fallback.
