# Exact Greedy Decode Burst Design

**Date:** 2026-08-22
**Status:** Approved under the standing autonomous-optimization authorization
**Stage-1 model:** Qwen3-0.6B
**Primary target:** batch-1 greedy decode synchronization and host-control cost

## Objective

Reduce exact greedy decode latency by executing several consecutive,
non-speculative decode forwards without synchronizing the sampled token to
the host between forwards.

The optimization must:

- preserve the exact token sequence produced by ordinary float32 greedy
  decoding;
- execute the full target model once for every emitted token;
- keep generated-token feedback, position advance, context-length advance,
  and within-block KV-slot advance on the GPU during a burst;
- perform one final device-to-host transfer for the burst token vector;
- commit all returned tokens through a bounded Scheduler transaction;
- remain default-disabled until a source-bound hardware gate passes;
- report output-visibility delay, graph capture cost, retained memory, and
  reserved Scheduler/KV capacity together with performance benefit.

This is an independently derived TinyLLMForge optimization. It is not
speculative decoding: no draft token is proposed, no target forward is
skipped, and no approximate acceptance rule is introduced.

## Observed Runtime Boundary

The completed Qwen3-0.6B graph-tail gate shows the following median
steady-decode decomposition for the current host-greedy path:

| Context | Median TPOT | Median model wall/CUDA time | Residual control gap |
| --- | ---: | ---: | ---: |
| 256 | 3.0540 ms | 2.4857/2.4874 ms | approximately 0.57 ms/token |
| 2048 | 3.4349 ms | 2.7575/2.7591 ms | approximately 0.68 ms/token |
| 8192 | 4.1985 ms | 3.1891/3.1908 ms | approximately 1.01 ms/token |

The near equality of model wall and CUDA time shows that the model call
finishes at a device synchronization boundary. The ordinary path then
performs:

```text
transformer CUDA Graph replay
  -> graph-external LM head
  -> float32 argmax
  -> token tensor .tolist()
  -> Scheduler postprocess
  -> next-step metadata construction
  -> next CUDA Graph replay
```

The zero-temperature fast path removes unnecessary stochastic sampling and
improves aggregate median TPOT by only a few percent. Moving LM head and
argmax into a separate tail graph also improves versus legacy by only a few
percent and regresses the host-greedy arm by 1.455072%. Those results bound
the expected value of further single-step launch trimming.

The remaining structural opportunity is the mandatory per-token host
round-trip. Under completion-only, fixed-budget greedy generation, the host
does not need the token value before launching the next target forward if:

1. EOS and stop-token inspection are disabled;
2. all KV write slots for the burst are authorized in advance;
3. the next input token and metadata are advanced on the device;
4. no request requiring an intervening scheduling decision is present.

## Considered Approaches

### A. Exact greedy decode burst

Capture one complete greedy decode step and make its token output feed the
next replay's static input. Issue `K` graph replays back-to-back, then
materialize the `K` exact tokens once and commit them as one Scheduler
transaction.

Advantages:

- removes `K - 1` token D2H synchronization points;
- removes `K - 1` Scheduler/postprocess and metadata-building round trips
  from the GPU critical path;
- retains one full target-model execution per output token;
- reuses the existing batch-1 CUDA Graph and transactional postprocess
  architecture;
- has a measurable ceiling of roughly 0.57-1.01 ms per token in the
  observed Stage-1 workload.

Costs and risks:

- host-visible token delivery is delayed until the burst completes;
- the Scheduler must authorize future KV writes before execution;
- a failure after the first replay cannot fall back because target KV has
  already been mutated;
- the complete-step graph retains logits, token history, counters, and
  metadata state;
- the initial implementation is intentionally narrow.

This is the selected approach.

### B. Ordinary decode Scheduler fast commit

Specialize the non-speculative one-token postprocess path to avoid the
general prepared-transaction machinery and repeated tuple/list construction.

This is lower risk, but it cannot remove the `.tolist()` synchronization or
the gap between GPU submissions. Its expected ceiling is below the observed
0.57-1.01 ms residual because only part of that interval is Scheduler work.

### C. Stable masked decode slots

Keep a fixed graph-width batch and mask inactive rows so request churn does
not change graph identity or rebuild metadata.

This may help concurrent serving, but it intentionally executes work for
inactive slots and does not address the batch-1 synchronization boundary.
It also requires a separate fairness and utilization gate. It remains a
future online-serving candidate.

## Architecture

### 1. Generic burst mechanism

Add a model-agnostic exact-decode-burst component. Its contract uses runtime
roles rather than model names:

- one stable full-step graph entry;
- static input token, position, context-length, slot-mapping, and block-table
  tensors;
- one retained float32 greedy token tensor;
- one retained token-history tensor;
- one retained device-side history index;
- a burst lease describing authorized replay count and KV write interval;
- capture/replay, synchronization, fallback, and quarantine accounting.

The complete-step graph body is logically:

```python
hidden = model(input_token, position)
logits = compute_logits(hidden)
next_token = logits.to(torch.float32).argmax(dim=-1)
token_history.index_copy_(0, history_index.view(1), next_token)
input_token.copy_(next_token)
position.add_(1)
context_length.add_(1)
slot_mapping.add_(1)
history_index.add_(1)
```

The history index is reset to zero before each burst. All mutations execute
in one CUDA stream and are therefore ordered before the next graph replay.
The host may enqueue several replays, but it must not inspect the token,
advance Scheduler state, or rebuild decode metadata between them.

The generic component does not inspect model type, checkpoint, tokenizer,
prompt text, EOS identity, or workload labels.

The hardware gate may request a separate correctness-only capture variant
with a fixed device-side logit-sampling mask. That variant copies only the
declared replay ordinals into a retained float32 logit-history tensor and
materializes the selected rows after the burst. It performs no intermediate
host synchronization. Correctness-only rows are excluded from performance
aggregation, capture-cost comparison, and promotion metrics.

### 2. Complete-step graph

The existing ordinary batch-1 graph captures only the transformer and
returns hidden states. The burst path needs a separate complete-step graph
that captures:

```text
transformer
  -> LM head
  -> float32 argmax
  -> token-history write
  -> autoregressive token feedback
  -> scalar metadata advance
```

It must not replay the existing transformer graph and a separate tail graph,
because the completed graph-tail experiment showed that the extra graph
launch loses to host-greedy.

The complete-step graph may share immutable model weights, but it owns a
dedicated graph pool and static activation/metadata tensors. Capture must
bind:

- graph generation;
- rank and tensor-parallel size;
- active and selected graph batch size;
- hidden size, vocabulary size, dtype, and device;
- block-table width;
- source tensor pointers, shapes, strides, and storage offsets;
- maximum token-history capacity.

Capture must never write a live request's KV slot. The complete-step graph is
captured with a private scratch block owned by the ModelRunner for its entire
lifetime. The scratch block:

- is excluded from Scheduler allocation and prefix-cache publication;
- supplies valid block-table and slot-mapping addresses during warmup and
  capture;
- may be overwritten by capture without changing any live sequence;
- is not counted as request capacity;
- is included in retained-memory and reserved-capacity reporting.

After capture, the static input, position, context length, slot mapping,
history index, and token-history tensors are restored to deterministic
sentinel values before the entry can become replayable. Replay binds only
the already-captured tensor storage; it copies an eligible lease's initial
values into those tensors before the first replay.

### 3. Scheduler burst lease

Before any burst replay, the Scheduler constructs an immutable lease:

```text
sequence identity and generation
schedule generation
requested and authorized token counts
initial completion count
initial sequence length
initial block-table identity
first and last KV write positions
physical block identity
remaining output budget
completion-only visibility authority
```

The authorized token count is:

```text
first_write_position = initial_sequence_length - 1
writable_positions_in_current_block =
    block_size - (first_write_position % block_size)

min(
    configured burst size,
    remaining output tokens,
    writable_positions_in_current_block,
)
```

If this minimum is less than two, the production path falls back before the
first replay. Width one remains available only to the gate's direct
`full_step_graph_k1` causal arm.

Each replay consumes the current input token at `first_write_position + i`,
writes that token's KV, and predicts one new token. Therefore an authorized
burst of width `K` always authorizes exactly `K` consecutive KV writes. The
token predicted by the final replay is not itself decoded again inside the
burst and does not need an additional KV slot. This matches the existing
`prompt + max_tokens - 1` storage invariant.

The burst may consume the remainder of the current physical block, including
its final slot, but it must not increment `slot_mapping` into another physical
block. Even if a later logical block is already present in the sequence's
block table, crossing the boundary is rejected in Stage 1 because adjacent
logical blocks are not guaranteed to have adjacent physical IDs. The lease
binds the current block ID and generation and the exact inclusive write
interval.

The lease is rejected unless all of these are true:

- exactly one running sequence is selected;
- no waiting or prefilling sequence is present;
- the request is ordinary decode, not mixed or speculative;
- temperature is numeric zero;
- `ignore_eos` is true;
- the caller declares completion-only token visibility;
- at least two output tokens remain;
- tensor parallel size is one and rank is zero;
- ordinary batch-1 CUDA Graph execution is available;
- no input embeddings, hidden-state return, KV offload, CPU offload, Quest,
  compact attention, quantized eager path, or step-logit recording is active;
- graph source, block-table width, and physical KV identity match capture;
- no prior burst is pending or quarantined.

Every failed condition uses the existing one-token path before any current
step KV mutation.

The production entrypoint rejects ordinary per-step logit recording. The
gate-only correctness entrypoint is a distinct, explicitly labeled path with
fixed sample ordinals and its own graph identity; it cannot be enabled by the
public generation configuration.

### 4. ModelRunner execution

For an accepted lease:

1. prepare the first step's static decode metadata normally;
2. reset the graph's history index and retained token count;
3. replay the complete-step graph exactly `authorized_token_count` times;
4. perform one `.tolist()` on the populated token-history prefix;
5. return a typed burst result containing the lease identity, exact tokens,
   replay count, final metadata state, and cost counters.

The implementation must prove:

- one target-model graph replay per output token;
- zero intermediate token D2H operations;
- one final burst-token D2H operation;
- no graph-external LM-head, float32 argmax, or ordinary sampler calls;
- the final static input and metadata represent the next unexecuted step;
- no second model execution for any committed token.

The gate-only correctness variant may additionally perform one final D2H for
its bounded sampled-logit tensor. That transfer is recorded separately and
is never counted as a production-path token transfer or performance sample.

### 5. Engine and Scheduler commit

`LLMEngine` validates the returned lease identity before constructing one
multi-token `ScheduledOutputRow` marked `exact_burst=True`. The Scheduler
prepares one bounded postprocess journal and commits the returned tokens in
order.

This is an exact-generation row, not a speculative row. The design must not
mislabel it as accepted draft output merely to reuse an existing branch.
The Scheduler may share lower-level multi-token append and KV-publication
helpers with speculative postprocess, but exact burst ownership and counters
remain distinct.

`ScheduledOutputRow` validation requires exactly one of these forms:

- ordinary: `speculative=False`, `exact_burst=False`, exactly one output;
- exact burst: `speculative=False`, `exact_burst=True`, at least two outputs,
  no accepted-draft tokens, and a matching active burst lease;
- speculative: `speculative=True`, `exact_burst=False`, existing speculative
  invariants unchanged.

Commit must:

- append every returned token exactly once;
- advance completion counts by the replay count;
- publish a newly completed full-block hash only after every token ID in that
  block is materialized on the host and every corresponding target KV slot
  has been written by a completed replay;
- mark the sequence finished at the fixed output budget;
- retain the final generated token as the next input only when generation
  continues;
- release request storage exactly once on completion;
- expose one ordered token delta containing the complete burst.

### 6. Output visibility

The optimization changes token observation cadence even when the final token
sequence is identical. Therefore:

- the first implementation is completion-only;
- per-token streaming, token-level callbacks, stop strings, host logits
  inspection, and external per-step observers are ineligible;
- the benchmark reports both amortized TPOT and maximum host-visible burst
  gap;
- no result may describe amortized TPOT as streaming inter-token latency.

This trade-off is part of the mechanism, not an incidental benchmark detail.

## Failure Semantics

### Before first replay

Any invalid lease, source mismatch, unsupported mode, insufficient capacity,
or capture unavailability falls back to the ordinary one-token path.

### After first replay

No fallback is allowed. Target KV and device metadata may already contain
multiple authoritative steps. A replay, D2H, identity, or commit failure:

1. quarantines the burst component for the runner lifetime;
2. marks the active request/engine step failed;
3. preserves the original exception;
4. does not replay, resample, or partially commit the burst;
5. records whether zero, some, or all authorized replays completed.

The Scheduler journal may roll back host metadata if commit fails, but this
does not make mutated target KV safe for retry. The request remains failed.

## Genericity Review

### Two-axis verdict

```text
mechanism: reusable candidate
integration: first adopter only
```

Deleting the Qwen3 name leaves a coherent contract: a causal decoder with a
stable complete-step graph may execute an exact greedy burst under a
Scheduler-issued KV/visibility lease. Genericity is not yet established
because the first implementation has one ModelRunner caller and one real
checkpoint gate.

### Layer map

- mechanism: lease validation, graph identity, replay loop, token history,
  quarantine, and accounting;
- adapter: `ModelRunner` supplies the model forward, LM head, static graph
  tensors, and current runtime incompatibilities;
- policy/config: enable flag, maximum burst width, completion-only authority,
  and promotion thresholds;
- benchmark/profile: Qwen3-0.6B checkpoint, A100 GPU, prompt lengths, output
  length, repetition order, and Stage-1 thresholds.

### Leakage check

Core burst and Scheduler lease files must not contain Qwen3, checkpoint,
tokenizer, prompt, dataset, GPU model, or benchmark-bucket names. The
Scheduler consumes role-based facts: sequence count, output budget, stopping
requirements, visibility policy, block boundary, and graph capability.

### Recommended split

Use:

1. one dependency-light burst-lease module;
2. one complete-step graph module;
3. focused `Scheduler`, `ModelRunner`, and `LLMEngine` integration;
4. one Qwen3-0.6B benchmark profile;
5. a synthetic second caller proving the generic lease contract without
   claiming second-model GPU performance.

Do not build a graph registry, dynamic online policy, or multi-sequence burst
framework before Stage 1 establishes benefit.

### Evidence boundary

Unit tests can prove lease arithmetic, block-boundary clipping, ownership,
fallback, exact commit ordering, and failure semantics. A fake graph can
prove device-state sequencing structurally. Only a real GPU run can prove
CUDA Graph viability, exact model outputs/logits, synchronization reduction,
memory cost, output-gap cost, or speedup. Qwen3-0.6B TP1 evidence cannot
establish Qwen3-8B, tensor parallelism, online fairness, or streaming benefit.

## Configuration and Observability

Add strict configuration:

```text
exact_greedy_decode_burst = false
exact_greedy_decode_burst_tokens = 4
```

The token count must be an integer in `[2, 8]`. A non-boolean enable value or
invalid width fails during configuration validation.

Expose:

- lease attempts, acceptances, and fallback counts by stable reason;
- requested and authorized burst-width histogram;
- boundary-clipped and output-budget-clipped counts;
- complete-step graph captures, capture failures, replays, and quarantines;
- target-model forward count;
- intermediate and final token D2H operations and bytes;
- graph-external LM-head, float32 conversion, argmax, and sampler calls
  avoided;
- capture duration and allocated/reserved deltas;
- retained static bytes by tensor role;
- host-visible burst gaps;
- Scheduler commit and failure counts;
- pending lease count, which must return to zero.

Counters explain path execution but do not prove performance.

## Correctness Invariants

The candidate must preserve:

- exact generated token IDs and decoded-text hashes;
- exact float32 greedy argmax semantics;
- sampled logits with `max_abs <= 0.25`, per-pair
  `mean_abs <= 0.05`, and equal argmax;
- one full target-model execution per emitted token;
- exact KV write position and block identity for every replay;
- exact token order and completion count;
- full-block hash publication after host token materialization;
- request finish and storage-release semantics;
- default-disabled behavior for all unsupported requests;
- zero intermediate token D2H operations;
- one final token-vector D2H operation per accepted burst;
- zero live burst leases after success or failure.

## TDD and Verification

### Unit and integration tests

Tests must first fail and then prove:

1. strict configuration validation;
2. deterministic lease eligibility and stable fallback reasons;
3. burst width clips at output budget and current physical block boundary;
4. no waiting, prefilling, mixed, speculative, streaming, EOS-sensitive, or
   unsupported execution enters the burst path;
5. lease identity binds sequence, schedule generation, graph generation,
   block table, physical block, and initial token counts;
6. capture warmup and graph capture use only a private scratch block and
   cannot mutate any live request KV slot;
7. complete-step capture contains model, LM head, float32 argmax, token
   history, token feedback, and scalar metadata advance;
8. `K` authorized tokens cause exactly `K` target graph replays and exactly
   `K` consecutive target KV writes;
9. the first `K - 1` replays perform no token D2H;
10. one final D2H returns all tokens in order;
11. Scheduler accepts multi-token exact-burst rows without weakening
    ordinary or speculative row validation;
12. Scheduler commit appends all tokens exactly once and publishes a newly
    completed full block only after its host tokens and target KV are both
    materialized;
13. disabled and ineligible paths preserve the ordinary implementation;
14. pre-replay failure falls back while post-replay failure quarantines and
    never retries;
15. host-visible token cadence is explicitly recorded;
16. all accounting and retained-byte calculations are exact;
17. a synthetic non-Qwen caller satisfies the same lease contract.

Existing greedy-fast-path, graph-tail, metadata-landing, ModelRunner,
Scheduler transaction, CUDA Graph, chunked-prefill, prefix-cache, and
source-audit tests remain required.

## Four-Arm Stage-1 Gate

Run fresh ModelRunner instances in a balanced order:

```text
host_greedy:
  zero_temperature_greedy_fast_path=true
  graph_resident_greedy_tail=false
  exact_greedy_decode_burst=false

full_step_graph_k1:
  exact_greedy_decode_burst=true
  exact_greedy_decode_burst_tokens=1  # gate-only causal arm

decode_burst_k4:
  exact_greedy_decode_burst=true
  exact_greedy_decode_burst_tokens=4

decode_burst_k8:
  exact_greedy_decode_burst=true
  exact_greedy_decode_burst_tokens=8
```

The production configuration rejects width one; the gate-only arm invokes
the same complete-step mechanism directly to isolate graph integration cost
from cross-token amortization.

Use Qwen3-0.6B, TP1, batch size one, temperature zero, `ignore_eos=true`,
128 generated tokens, two warmups, five measured repetitions, and prompt
lengths 256, 2048, and 8192.

Retain:

- 60 performance rows;
- exact output IDs and decoded-text hashes;
- prefill-final, decode-first, decode-middle, and decode-final float32 logits
  for every arm and context;
- target-forward and graph-replay inventories;
- burst widths and final partial-burst widths;
- intermediate/final D2H operations and bytes;
- amortized TPOT and nearest-rank P95/P99;
- host-visible burst-gap median/P95/P99/max;
- TTFT, E2E, and output throughput;
- peak allocated and reserved CUDA memory;
- capture duration and retained bytes;
- Scheduler lease/commit/failure inventory;
- source/workload manifests;
- producer and independent-verifier receipts.

Correctness/logit collection and performance timing use separate fresh
ModelRunner instances. Performance rows must report the production graph
identity with correctness tracing disabled. Correctness rows must report the
gate-only trace graph identity, selected replay ordinals, and exactly one
post-burst sampled-logit D2H operation when a burst contains a requested
sample point.

## Deterministic Candidate Selection

Each burst arm is eligible only if all correctness, lifecycle, and protected
metric gates pass. Among eligible burst arms, select the arm with the largest
aggregate median amortized-TPOT improvement versus `host_greedy`. Exact ties
select the smaller burst width.

The `full_step_graph_k1` arm is causal evidence only and cannot be selected
for promotion.

## Promotion Gate

Classify `GO_EXACT_GREEDY_DECODE_BURST` only if the selected burst arm:

1. has exact output-token and decoded-text equality with both control arms;
2. satisfies every frozen logit tolerance and argmax check;
3. executes exactly one target forward per emitted token;
4. executes the complete-step graph for every measured burst replay;
5. records zero intermediate token D2H and exactly one final D2H per burst;
6. leaves zero pending leases and has no replay, commit, or quarantine event;
7. improves aggregate median amortized TPOT versus `host_greedy` by at least
   10%;
8. improves aggregate nearest-rank P95 amortized TPOT by at least 8%;
9. improves median amortized TPOT by at least 8% in at least two of the three
   context buckets;
10. improves aggregate median amortized TPOT versus `full_step_graph_k1` by
    at least 5%;
11. does not regress any bucket's amortized median or P95 TPOT versus
    `host_greedy` by more than 3%;
12. does not regress TTFT or E2E versus `host_greedy` by more than 3%;
13. does not regress throughput versus `host_greedy` by more than 2%;
14. does not regress peak reserved CUDA memory by more than 3%;
15. keeps maximum host-visible burst gap at or below 40 ms;
16. reports capture duration, retained bytes, lease capacity, and output
    visibility cost;
17. producer and independent verifier agree on classification, selected arm,
    comparison digest, and manifest digest.

Any failure produces a specific NO-GO classification. Exact correctness with
only amortized throughput benefit is not sufficient if the host-visible
burst-gap limit fails.

## Remote Safety and Promotion Boundary

All remote data must remain below:

`/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`

Use a fresh immutable run tag, source commit equality with the pushed branch,
strict-clean GPU admission, Kerberos TTL fail-fast, complete manifest
download, and local independent reconstruction. Do not refresh credentials,
write outside the approved mounted root, or terminate unrelated processes.

Stage 1 proves only completion-only Qwen3-0.6B TP1 batch-1 greedy generation
on the admitted GPU. A Stage-1 GO is required before Qwen3-8B validation.
Neither Stage 1 nor Stage 2 authorizes claims about token streaming, EOS-aware
generation, multi-sequence fairness, tensor parallelism, or production
default enablement.

## Deliverables

- default-disabled model-agnostic burst lease and complete-step graph;
- focused Scheduler, ModelRunner, and LLMEngine integration;
- synthetic second caller for contract-level genericity;
- dependency-light unit and integration tests;
- source-bound four-arm Qwen3-0.6B Stage-1 benchmark;
- producer gate and independent verifier;
- immutable local and remote artifacts;
- benefit-and-cost report including host-visible output delay;
- EOF audit and handoff reconciliation;
- exact-path commits pushed to `origin/feat/kv-sparse-attention`.
