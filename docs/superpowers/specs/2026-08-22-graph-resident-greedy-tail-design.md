# Graph-Resident Greedy Tail Design

**Date:** 2026-08-22
**Status:** Approved under the standing autonomous-optimization authorization
**Stage-1 model:** Qwen3-0.6B
**Primary target:** batch-1 zero-temperature decode TPOT and tail latency

## Objective

Reduce ordinary batch-1 greedy decode latency by moving the graph-external
LM-head, float32 conversion, and argmax sequence into a second CUDA Graph
that consumes the existing transformer's static graph output directly.

The optimization must preserve the current float32 greedy result exactly,
must remain default-disabled until a paired hardware gate passes, and must
report graph-memory and capture-time cost together with latency benefit.

This is an independently motivated TinyLLMForge optimization. It does not
assume that the zero-temperature host fast path is beneficial, and it does
not reuse the negative replay-aware metadata result as performance proof.

## Observed Runtime Boundary

The existing ordinary batch-1 CUDA Graph captures only:

```text
input metadata
  -> transformer forward
  -> static hidden-state output
```

After every replay, Python separately invokes:

```text
compute_logits(hidden)
  -> logits.to(float32)
  -> argmax
  -> token_ids.tolist()
```

The final token transfer to Python is required by the current scheduler, but
the LM-head, conversion, and argmax launches do not need Python between them.
The existing transformer graph output is a stable tensor, so a second graph
can read it without a hidden-state copy.

The completed replay-aware metadata experiment established that small
decode-path launch and staging changes can move median TPOT by roughly
2.5-3% on short and medium contexts, but did not meet the 5% gate and
regressed the long-context tail. That result motivates attacking a larger,
still isolated graph-external launch chain.

## Considered Approaches

### A. Graph-resident greedy tail

Capture `compute_logits()`, exact float32 conversion, and argmax in a
dedicated tail graph whose input is the existing transformer's static
batch-1 hidden output.

Advantages:

- removes Python dispatch between LM-head, conversion, and argmax;
- removes repeated eager launch setup for those operations;
- introduces no hidden-state device copy;
- preserves the current token-level synchronization boundary;
- has a narrow fail-closed eligibility contract.

Costs and risks:

- adds one CUDA Graph replay per decode token;
- retains graph-private logits and float32 argmax intermediates;
- adds capture latency and reserved CUDA memory;
- requires exact lifetime binding to the transformer's static output;
- does not help prefill, non-greedy sampling, eager decode, or batch sizes
  above one.

This is the selected approach.

### B. Decode block-table reuse

Cache a graph input's page table and skip its H2D copy until the sequence
crosses a KV-block boundary.

This is safe in a tightly bound graph entry, but at 8K context the avoided
payload is still only a small int32 row. Its expected ceiling is lower than
removing the LM-head/cast/argmax launch chain. It remains a later candidate.

### C. Disable cyclic GC during decode

Move automatic Python garbage collection outside active decode windows.

This is rejected. Previous work identified allocation storms but did not
prove GC as the residual cause, and an earlier approved design explicitly
rejected production GC control because it can trade latency for unbounded
transient host memory and delayed finalization.

## Architecture

### 1. Generic mechanism

Add a focused graph-tail component with no model names in its contract. It
accepts:

- one stable hidden-state tensor owned by an existing batch-1 transformer
  graph;
- a callable that maps hidden states to logits;
- the expected hidden width, vocabulary width, dtype, device, and rank;
- a CUDA event/graph factory supplied by the runner.

It owns:

- one dedicated CUDA Graph;
- the retained logits tensor produced during capture;
- the retained one-element token tensor;
- capture and replay counters;
- capture duration and memory deltas;
- a stable source-hidden identity;
- a terminal quarantine reason after replay failure.

The graph body is exactly:

```python
logits = compute_logits(static_hidden[:1])
token_ids = logits.to(torch.float32).argmax(dim=-1)
```

The mechanism does not inspect model type, tokenizer, request text, or
checkpoint name.

### 2. ModelRunner first integration

`ModelRunner` is the first adopter. During ordinary CUDA Graph construction,
it may construct one greedy tail after the batch-1 transformer graph exists.
The tail binds directly to `graph_vars["outputs"][:1]`.

The first implementation is authorized only when:

- the feature flag is enabled;
- tensor parallel size is one and rank is zero;
- the request is ordinary decode;
- active and selected graph batch sizes are exactly one;
- sampling is enabled;
- exactly one sequence has numeric `temperature == 0.0`;
- no mixed batch, input embedding, hidden-state return, KV offload, CPU
  offload, Quest, compact attention, or quantized eager-only mode is active;
- step-logit recording either consumes the retained graph logits or is
  disabled;
- the bound transformer graph and static hidden tensor identities still
  match the tail's capture receipt.

Every failed condition uses the existing run-model and sampler path.

### 3. Data flow

Eligible decode:

```text
prepare decode metadata
  -> replay existing transformer graph
  -> same-stream replay of greedy tail graph
  -> optional correctness capture from retained logits
  -> one-element token tensor .tolist()
  -> scheduler postprocess
```

There is no hidden-state copy between the two graphs. CUDA stream ordering is
the producer-consumer dependency. The final `.tolist()` remains because the
scheduler currently owns stop conditions, sequence mutation, and KV
allocation on the host.

Fallback decode:

```text
existing transformer graph or eager forward
  -> existing compute_logits()
  -> existing optional greedy fast path or stochastic sampler
```

No request may silently switch sampling semantics.

### 4. Failure handling

Capture failure leaves the feature unavailable and records a stable reason.
Replay failure quarantines the tail for the runner lifetime and propagates
the error; it must not replay the transformer a second time or resample the
same step.

Shape, source identity, device, dtype, rank, or graph-generation drift fails
closed before tail replay. Default-disabled behavior must remain byte-for-byte
equivalent at the Python contract level.

## Genericity Review

### Two-axis verdict

```text
mechanism: reusable candidate
integration: first adopter only
```

The graph-tail mechanism consumes a stable producer tensor plus a logits
callable and survives deletion of the Qwen3 name. Genericity is not yet
proven because only the ordinary `ModelRunner` decode path is integrated and
only Qwen3 is authorized for the first hardware gate.

### Layer map

- mechanism: graph capture, source identity, replay, quarantine, accounting;
- adapter: `ModelRunner` supplies the stable transformer output and
  `compute_logits`;
- policy/config: eligibility flag and exact batch/rank/temperature limits;
- benchmark/profile: Qwen3-0.6B checkpoint, A100 GPU, context buckets,
  repetitions, thresholds, and output oracle.

### Leakage check

Core graph-tail code must not contain Qwen3, checkpoint, prompt, tokenizer,
or workload names. Qwen3 appears only as the Stage-1 benchmark profile and
first-adopter evidence.

### Recommended split

Use one generic graph-tail module, one focused `ModelRunner` integration,
and one model-specific benchmark profile. Do not create a registry or generic
multi-model framework before a second adopter exists.

### Evidence boundary

Dependency-light tests can prove policy, ownership, fallback, accounting,
and simulated replay. Only a real GPU run can prove capture viability,
memory cost, exact logits/output parity, or performance. Qwen3-0.6B evidence
does not establish Qwen3-8B or tensor-parallel behavior.

## Configuration and Observability

Add a strict boolean flag:

```text
graph_resident_greedy_tail = false
```

Expose a runner summary containing:

- eligible, captured, replayed, fallback, and quarantined counts;
- fallback counts by stable reason;
- source-hidden identity and graph generation;
- capture duration;
- allocated and reserved memory deltas;
- retained logits and token tensor bytes;
- avoided external `compute_logits`, float32-conversion, and argmax calls;
- final token D2H count.

Counters explain the mechanism but do not prove a speedup.

## Correctness Invariants

The candidate must preserve:

- exact generated token IDs and decoded-text hashes;
- retained pre-argmax logits with `max_abs <= 0.25`,
  per-pair `mean_abs <= 0.05`, and equal argmax;
- exact float32 argmax semantics;
- transformer graph selection and replay count;
- KV block ownership and write locations;
- scheduler order, completion, and stop behavior;
- ordinary fallback behavior for every ineligible request;
- one and only one model execution and one sampling decision per step.

No fallback may occur after the transformer graph has mutated KV for the
current step.

## TDD and Verification

### Unit and integration tests

Tests must first fail and then prove:

1. exact eligibility and stable fallback reasons;
2. capture binds the exact static hidden tensor and expected graph generation;
3. the graph body performs `compute_logits`, float32 conversion, and argmax;
4. replay returns the retained token tensor without a hidden copy;
5. the normal sampler is not called on an eligible replay;
6. optional logit recording reads the retained graph logits;
7. source, shape, dtype, device, graph-generation, or rank drift fails closed;
8. capture failure stays disabled and replay failure quarantines;
9. disabled and unsupported paths preserve the existing implementation;
10. accounting and memory-byte calculations are exact.

Existing greedy-fast-path, model-runner, CUDA Graph, scheduler, prefix,
chunked-prefill, and source-audit suites remain required.

### Three-arm Stage-1 gate

Use fresh runner instances and alternating order for:

```text
legacy:
  zero_temperature_greedy_fast_path=false
  graph_resident_greedy_tail=false

host_greedy:
  zero_temperature_greedy_fast_path=true
  graph_resident_greedy_tail=false

graph_greedy:
  zero_temperature_greedy_fast_path=true
  graph_resident_greedy_tail=true
```

Run Qwen3-0.6B with batch size one, temperature zero, 128 generated tokens,
two warmups, five measured repetitions, and prompt lengths 256, 2048, and
8192. Keep model, source, GPU UUID, input IDs, output length, and order policy
fixed.

Retain:

- 45 performance rows;
- exact output IDs and text hashes;
- three-point float32 logits sidecars for every arm and context;
- per-token TPOT and decode host/CUDA samples;
- TTFT, E2E, and output throughput;
- peak allocated and reserved CUDA memory;
- graph capture duration and memory deltas;
- replay and avoided-work counters;
- source/workload manifests;
- producer and independent-verifier receipts.

## Promotion Gate

The classification is `GO_GRAPH_RESIDENT_GREEDY_TAIL` only if:

1. all three arms have exact output equality;
2. every retained logits pair meets the frozen numerical thresholds and has
   equal argmax;
3. every measured graph-greedy decode step uses the tail graph after warmup;
4. graph-greedy versus legacy improves median TPOT by at least 5% in at
   least two of three context buckets;
5. graph-greedy versus legacy improves aggregate nearest-rank P95 TPOT by at
   least 5%;
6. graph-greedy versus host-greedy improves aggregate median TPOT by at least
   2% and does not regress any bucket median or P95 by more than 2%;
7. no graph-greedy bucket regresses median or P95 TPOT versus legacy by more
   than 3%;
8. TTFT and E2E do not regress versus legacy by more than 3%;
9. throughput does not regress versus legacy by more than 2%;
10. peak CUDA reserved memory does not regress versus legacy by more than
    2%;
11. capture duration, static bytes, and final token D2H cost are reported;
12. producer and independent verifier agree on classification, comparison
    digest, and manifest digest.

Any failure produces a specific NO-GO classification. Correctness without
measured benefit is not promotable.

## Remote Safety and Promotion Boundary

All remote data must remain below:

`/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`

Use immutable tags, strict-clean GPU admission, source commit equality with
the pushed branch, Kerberos TTL fail-fast, complete manifest download, and
local independent reconstruction. Do not refresh credentials or terminate
unrelated processes automatically.

Stage 1 proves only Qwen3-0.6B, tensor parallel size one, batch size one,
ordinary zero-temperature decode on the tested host. A Stage-1 GO is required
before default enablement, a batch-size extension, Qwen3-8B validation, or
tensor-parallel claims.

## Deliverables

- default-disabled generic graph-tail mechanism;
- focused `ModelRunner` adopter, first validated with Qwen3;
- dependency-light unit and integration tests;
- source-bound three-arm benchmark worker;
- producer gate and independent verifier;
- immutable local and remote artifacts;
- benefit-and-cost report;
- EOF audit and handoff reconciliation;
- exact-path commits pushed to `origin/feat/kv-sparse-attention`.
