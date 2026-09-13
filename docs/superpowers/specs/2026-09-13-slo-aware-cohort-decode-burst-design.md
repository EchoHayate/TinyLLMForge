# SLO-Aware Cohort Decode Burst Design

**Date:** 2026-09-13
**Status:** Approved design; implementation is not yet authorized
**Stage-0/1 model:** Qwen3-0.6B
**Initial topology:** TP1 on one NVIDIA A100 80GB PCIe
**Primary target:** open-loop multi-request output throughput with protected
P99 host-visible inter-token latency and TTFT

## Objective

Extend TinyLLMForge's default-disabled Exact Greedy Decode Burst from its
proven batch-one path into a multi-request serving experiment that preserves
the scheduler's existing request order while reducing repeated host control,
token device-to-host transfer, and scheduler publication work.

The selected design is an SLO-aware cohort burst:

1. the existing scheduler chooses the decode cohort;
2. a separate controller observes the cohort and all requests that the cohort
   may block;
3. the controller selects the largest safe burst width from `1, 2, 4, 8`;
4. the entire cohort advances by that many exact greedy target-model steps;
5. one bounded transaction validates and publishes the per-request token
   prefixes.

The first source-bound promotion gate targets:

```text
aggregate output-throughput improvement >= 10%
every workload P99 host-visible ITL regression <= 3%
every workload P99 TTFT regression <= 5%
```

Throughput alone is insufficient. A candidate that wins throughput but
violates a protected tail-latency, correctness, starvation, memory, or
evidence-completeness condition is a `NO_GO`.

## Existing Evidence and Boundary

The authoritative Exact Greedy Decode Burst gate selected `decode_burst_k8`
for Qwen3-0.6B, TP1, batch one, zero-temperature, completion-only generation.
Against TinyLLMForge host greedy, that path established:

```text
aggregate median TPOT improvement: 29.361129%
aggregate P95 TPOT improvement:    33.916056%
aggregate P99 TPOT improvement:    33.422108%
aggregate E2E improvement:         28.160104%
output-throughput improvement:     39.198420%
TTFT regression:                    0.335487%
maximum host-visible burst gap:    24.035218 ms
```

Correctness passed with exact output token IDs, exact decoded text, equal
argmax, and zero sampled-logit difference over the declared checks.

That result does not establish multi-request fairness, continuous batching,
EOS-aware behavior, streaming benefit, TP greater than one, larger-model
benefit, or production-default safety. The current implementation explicitly
rejects a burst when:

- the selected decode batch contains more than one sequence;
- any request is waiting;
- any request is still prefilling;
- EOS inspection is active;
- tensor parallel size is greater than one.

Several subsequent attempts narrowed the remaining opportunity:

- generation-sealed identity was correct but not promotable on performance;
- phase-stitched exact graph was correct but missed its end-to-end threshold;
- octet-folded replay reduced physical graph launches but produced negligible
  median TPOT improvement;
- corrected persistent-decode profiling found less than 1% optimistic TPOT
  headroom;
- fixed or elastic wider bursts did not establish a general replacement for
  K8.

This design therefore does not repeat those mechanisms. It targets
multi-request host/runtime amortization while explicitly preserving latency
and fairness.

## Alternatives Considered

### A. Queue-depth thresholds

Choose a burst width directly from waiting depth:

```text
empty queue -> K8
small queue -> K4
congested queue -> K1
```

Advantages:

- small implementation;
- deterministic;
- easy to inspect.

Disadvantages:

- request count is not elapsed-time slack;
- one long step can consume much more time than several short steps;
- thresholds do not directly protect TTFT or ITL;
- a queue can be shallow while one request is already near its deadline.

This approach is rejected as the primary policy.

### B. SLO-aware single-owner burst

Allow one selected request to execute K2/K4/K8 when the predicted burst fits
inside the other requests' slack.

Advantages:

- reuses most of the existing batch-one K8 runtime;
- introduces a real elapsed-time guard;
- lower implementation risk than a cohort graph.

Disadvantages:

- intentionally reduces the selected decode batch to one request;
- sacrifices GPU batch parallelism under concurrency;
- may improve one request's TPOT while reducing aggregate throughput;
- is unlikely to meet the approved 10% multi-request throughput target.

This approach remains a possible diagnostic arm but is not selected.

### C. SLO-aware cohort burst

Keep the scheduler-selected decode cohort intact and advance every eligible
row by the same selected burst width.

Advantages:

- preserves continuous-batching occupancy;
- amortizes host work across both batch size and burst width;
- directly controls elapsed-time exposure;
- does not obtain performance by reordering or starving requests;
- composes with the existing scheduler clock and decode-progress accounting.

Costs and risks:

- requires a new multi-sequence exact-burst graph and transaction;
- cohort-wide failure handling is stricter than batch-one handling;
- EOS may cause post-EOS speculative device work that cannot be committed;
- graph shape coverage and retained memory increase;
- conservative SLO protection may select K1 frequently and erase the benefit.

This is the selected approach.

## Architecture

### Scheduler ownership

The existing scheduler remains the sole owner of:

- request admission;
- queue order;
- prefill/decode selection;
- preemption;
- KV allocation;
- sequence completion;
- host-visible token publication.

The cohort controller does not reorder requests or construct a different
batch. It receives the exact ordered decode cohort already selected by the
scheduler and chooses only the number of steps by which that cohort may
advance.

```text
scheduler selects [A, B, C, D]
        |
        v
SLO controller selects K
        |
        v
exact cohort graph advances [A, B, C, D] by K steps
        |
        v
scheduler validates and atomically publishes per-request prefixes
```

### Module boundaries

Add:

```text
tinyvllm/engine/slo_cohort_burst.py
```

This pure policy module owns:

- request SLO records;
- frozen cost-table validation;
- slack calculation;
- width selection;
- deterministic suppression reasons;
- decision telemetry.

Add:

```text
tinyvllm/engine/exact_greedy_cohort_burst.py
```

This runtime-contract module owns:

- multi-sequence leases;
- per-row write authorities;
- multi-row result validation;
- EOS-prefix accounting;
- cohort transaction identities;
- cohort-specific execution statistics.

Existing modules retain narrow responsibilities:

- `scheduler.py` registers request timing, supplies the ordered cohort,
  creates the transaction, and commits or aborts it;
- `model_runner.py` owns graph capture, static tensors, graph health, and
  replay;
- `llm_engine.py` orchestrates calls but contains no width-selection policy;
- benchmark and verifier tools consume immutable telemetry but never choose
  runtime policy.

The existing batch-one `exact_greedy_decode_burst.py` remains the historical
K8 implementation and fallback authority. The cohort implementation must not
turn that already-large module into a combined policy, scheduler, benchmark,
and multi-sequence owner.

### Request SLO state

The controller maintains a record keyed by sequence ID:

```text
RequestSLOState
  sequence_id
  arrival_ns
  first_token_visible_ns
  last_token_visible_ns
  service_class
```

The state is not added to the generic `Sequence` serialization contract.
Scheduler hooks update it on:

- request admission;
- first token publication;
- every later token publication;
- preemption and resumption;
- completion, cancellation, or failure.

A missing record, invalid timestamp, non-monotonic clock, or timestamp in the
future suppresses cohort bursting and selects ordinary K1.

### Graph identity and shape

The cohort graph cache key is:

```text
batch_size
block_table_width
dtype
device identity
tensor-parallel size
correctness-trace mode
```

Stage 1 supports TP1 only. Tensor-parallel size remains in the identity so a
TP1 graph can never be reused by a later TP2 extension.

Burst width is not a graph-cache dimension. One complete-step cohort graph is
replayed one, two, four, or eight times according to the lease.

Static tensors are row-indexed:

- input tokens;
- positions;
- context lengths;
- slot mappings;
- padded block tables;
- active-row masks;
- token-history rows;
- history indices;
- EOS observations.

Capture uses scheduler-inaccessible scratch KV blocks. It must not mutate a
live request. Graph retained bytes, scratch-block capacity, capture duration,
and capture count are reported by graph shape.

## SLO Decision

### Protected requests

The controller considers all requests whose progress can be delayed by the
candidate burst:

- every sequence in the selected cohort;
- runnable decode sequences omitted from the current cohort;
- admitted waiting requests;
- requests with incomplete prefill.

It is invalid to calculate slack from only the selected cohort.

### Slack

For a request that has emitted at least one token:

```text
itl_slack_ns =
  target_itl_ns
  - (decision_now_ns - last_token_visible_ns)
  - reserve_ns
```

For a request that has not emitted its first token:

```text
ttft_slack_ns =
  target_ttft_ns
  - (decision_now_ns - arrival_ns)
  - reserve_ns
```

The global available slack is:

```text
global_slack_ns = min(all protected request slacks)
```

If no valid positive slack exists, the controller selects K1.

### Frozen cost envelope

Stage 0 creates a source-, model-, GPU-, and configuration-bound cost table:

```text
predicted_cost_ns[batch_size, context_bucket, burst_width]
```

The value is a conservative nearest-rank P99 duration from calibration, not a
mean or an optimistic fitted value. A cohort spanning multiple context
buckets uses the maximum applicable prediction.

The table includes:

- schema and source identity;
- model/checkpoint identity;
- GPU UUID and device properties;
- graph identity inputs;
- batch size;
- context bucket;
- burst width;
- sample count;
- raw sample digest;
- P50/P95/P99;
- table SHA-256.

The canonical candidate consumes a table frozen before candidate execution.
It may not rewrite or retune the table from canonical results.

### Width selection

The controller checks widths in descending order:

```python
for width in (8, 4, 2):
    if structurally_eligible(width):
        if predicted_cost(width) <= global_slack_ns:
            return width
return 1
```

Structural eligibility requires:

- all cohort rows use zero-temperature greedy decoding;
- the cohort graph is available and healthy;
- no prior cohort lease is pending;
- every request has enough output budget;
- every row has enough authorized writable capacity;
- the batch shape and block-table width are supported;
- no incompatible execution mode is active;
- the controller clock and cost-table identity are valid.

The selected width is clipped by the minimum output budget and writable
capacity across the cohort. Unsupported ragged capacity selects the next
smaller width rather than changing cohort membership.

### Fixed fallback precedence

The first applicable reason is recorded:

```text
disabled
clock_invalid
missing_slo_state
cost_table_invalid
non_greedy_request
mixed_mode_unsupported
graph_unavailable
graph_quarantined
pending_lease
cohort_shape_unsupported
insufficient_output_budget
kv_block_boundary
no_slo_slack
predicted_cost_exceeds_slack
```

The verifier reconstructs the same precedence from raw observations.

## Cohort Transaction

### Lease

Before replay, the scheduler creates one immutable cohort lease binding:

- schedule generation;
- ordered sequence IDs;
- per-sequence generation;
- per-sequence block-table identity;
- writable block IDs and generations;
- first and last authorized logical positions;
- first and last authorized physical slots;
- initial completion counts;
- remaining output budgets;
- requested and authorized burst width;
- batch and graph identities;
- controller decision timestamp;
- cost-table identity;
- predicted duration and available slack.

The lease covers the entire cohort. Reordering rows, replacing a request, or
changing any bound block generation invalidates the lease.

### Execution

For K greater than one:

1. bind all row-indexed static tensors;
2. reset token histories and EOS observations;
3. replay the complete-step cohort graph K times;
4. retain intermediate tokens and autoregressive feedback on device;
5. copy the declared token-history and EOS slices to the host once;
6. construct one result bound to the lease and graph identity;
7. validate every row;
8. atomically publish all valid per-request prefixes.

The target model still executes once per logical token per active row. This
is exact greedy execution, not draft-model speculative decoding.

### EOS-aware prefix commit

For an EOS-sensitive row, the committed prefix ends at the first EOS token.
Later tokens calculated for that row in the same graph replay sequence are
not host-visible and are never committed.

The runtime records:

```text
wasted_post_eos_tokens
wasted_post_eos_forwards
wasted_post_eos_cuda_ns
```

An EOS row is terminal, so its uncommitted future-slot content cannot become
authoritative state for a later decode step. The scheduler frees or retires
the request through its ordinary completion path only after the cohort
transaction validates.

### Atomicity and failures

Before the first graph replay, a recognized failure may cancel the lease and
fall back to ordinary K1.

After any graph replay begins:

- no eager or K1 retry is allowed in the same engine step;
- no request in the cohort is partially published;
- the graph identity is quarantined on graph or result-integrity failure;
- the cohort follows the existing terminal-failure path;
- the artifact records completed logical replays and all future KV slots that
  may have been written;
- the pending lease must be closed exactly once.

Normal EOS-prefix variation is not a partial transaction. Every row is
validated, and all per-row prefixes are published together.

## Controller State Machine

```text
IDLE
  -> OBSERVE
  -> SELECT_WIDTH
       -> K1 ordinary path
       -> RESERVE_COHORT_LEASE
            -> pre-replay failure: cancel and use K1
            -> DISPATCH
                 -> VALIDATE_RESULT
                      -> ATOMIC_COMMIT
                      -> post-replay failure: quarantine and terminate
  -> IDLE
```

The controller does not learn or retune policy thresholds during the
canonical gate. Runtime observations are telemetry only.

## Configuration

All new behavior is default-disabled:

```text
exact_greedy_cohort_burst = false
exact_greedy_cohort_burst_widths = [1, 2, 4, 8]
exact_greedy_cohort_burst_max_batch_size = 8
exact_greedy_cohort_burst_target_itl_ns
exact_greedy_cohort_burst_target_ttft_ns
exact_greedy_cohort_burst_reserve_ns
exact_greedy_cohort_burst_cost_table_path
```

Validation rules:

- widths are strictly increasing powers of two beginning with one;
- maximum width is eight;
- maximum batch size is positive and no greater than the captured limit;
- targets and reserve are non-negative integers;
- reserve is smaller than both targets;
- a K greater than one requires a valid cost-table path;
- cohort burst requires the base exact greedy burst capability;
- TP greater than one is rejected in Stage 1;
- disabling the feature preserves current behavior byte-for-byte outside
  unavoidable telemetry version fields.

The cohort configuration does not reuse chunked-prefill policy fields.

## Telemetry

### Decision rows

Every scheduling decision records:

- monotonic decision timestamp;
- schedule generation;
- ordered cohort sequence IDs;
- waiting, prefilling, and running depths;
- batch size and context buckets;
- protected-request ages and slacks;
- global slack;
- K8/K4/K2 predicted costs;
- structural eligibility per width;
- selected width;
- fallback or suppression reason;
- cost-table SHA-256.

### Execution rows

Every accepted cohort burst records:

- lease, result, and graph identities;
- requested and authorized width;
- completed replay count;
- predicted and actual duration;
- host-visible publication gap;
- token D2H calls and bytes;
- per-request generated and committed token counts;
- per-request EOS-discarded counts;
- post-EOS wasted work;
- fallback, failure, rollback, and quarantine state.

### Request rows

Every request retains:

- request and sequence IDs;
- service class;
- arrival timestamp;
- prefill start and completion;
- first-token visibility;
- every host-visible token timestamp;
- completion timestamp;
- output token IDs or an auditable sidecar;
- output text hash;
- terminal reason.

Raw request and decision rows remain authoritative. A summary without those
rows is incomplete.

## Measurement Definitions

```text
TTFT =
  first host-visible token timestamp - request arrival timestamp

ITL[i] =
  host-visible timestamp[token i]
  - host-visible timestamp[token i - 1]

E2E =
  final host-visible token timestamp - request arrival timestamp
```

Tokens published together may share a timestamp. Inter-burst stalls remain
in the ITL distribution and must not be replaced by amortized TPOT.

Output throughput is:

```text
total committed output tokens
/
(last request completion - first frozen arrival)
```

Request throughput uses completed requests over the same window. P50, P95,
and P99 use the frozen nearest-rank rule over raw observations.

## Staged Validation

### Stage 0: ceiling and calibration

Run the ordinary multi-request baseline at batch sizes `1, 2, 4, 8` and
attribute:

- target-model CUDA time;
- graph execution and launch gaps;
- scheduler time;
- token D2H and publication;
- batch construction and metadata binding;
- GPU idle bubbles attributable to host control.

Generate the frozen cost table and a conservative optimistic throughput
ceiling.

Stop without implementing the cohort runtime if neither medium nor high load
has at least 12% optimistic throughput headroom.

```text
classification = NO_GO_CEILING
```

The 12% threshold leaves two percentage points for implementation cost above
the approved 10% final target.

### Stage 1: correctness and lifecycle

Exercise all supported batch and width combinations:

```text
B in 1, 2, 4, 8
K in 1, 2, 4, 8
```

Cover:

- output-budget clipping;
- KV block boundaries;
- heterogeneous context lengths;
- partial and multiple EOS rows;
- graph absence and quarantine;
- stale schedule, graph, and block generations;
- execution-before-replay fallback;
- failure on replay ordinal N;
- D2H and validation failure;
- atomic commit and terminal cleanup.

Required result:

```text
output token IDs:             exact
decoded text:                exact
sampled logit max abs:       0.0
sampled logit mean abs:      0.0
argmax:                      equal
duplicate forwards/commits:  0
unauthorized KV publication: 0
pending leases after case:   0
```

### Stage 2: open-loop serving gate

Calibrate baseline saturation separately, then freeze arrival traces at:

| Load | Fraction of baseline saturation |
| --- | ---: |
| Low | 40% |
| Medium | 70% |
| High | 90% |

Baseline and candidate receive identical request IDs, prompts, arrival
timestamps, output budgets, and EOS settings. Closed-loop request generation
is prohibited.

Use at least 128 measured requests per workload and arm. Run at least five
paired repetitions with balanced order, including:

```text
baseline -> candidate -> candidate -> baseline
```

The exact full order is frozen before execution.

#### Workload 1: decode-heavy steady arrivals

```text
prompt tokens: 256
maximum output tokens: 128
```

This isolates decode throughput and host-control amortization.

#### Workload 2: mixed short and long requests

```text
70%:  256 prompt / 64 output
20%: 2048 prompt / 128 output
10%: 8192 prompt / 128 output
```

This protects short-request TTFT and ITL under long-request pressure.

#### Workload 3: bursty EOS-sensitive arrivals

Requests arrive in frozen waves, use `ignore_eos=false`, and terminate at
different output lengths. This exercises dynamic width contraction,
publication stalls, and post-EOS waste.

## Formal Classification

Evidence, source identity, correctness, and lifecycle close before
performance classification.

`GO_SLO_AWARE_COHORT_DECODE_BURST` requires all of:

```text
aggregate output-throughput improvement >= 10%
medium-load throughput improvement      >= 10%
high-load throughput improvement        >= 10%
any workload throughput regression      <= 2%

any workload P99 ITL regression         <= 3%
any workload P99 TTFT regression        <= 5%
any workload P99 E2E regression         <= 5%
maximum host-visible gap                <= 40 ms

starved requests                         = 0
post-EOS wasted-forward fraction        <= 10%
peak CUDA reserved-memory regression    <= 5%
all transaction inventories             closed
remote and local verifier               agree
```

Fixed failure precedence:

```text
INVALID_SOURCE_OR_EVIDENCE
NO_GO_CORRECTNESS
NO_GO_LIFECYCLE
NO_GO_STARVATION
NO_GO_TAIL_LATENCY
NO_GO_MEMORY
NO_GO_EOS_WASTE
NO_GO_THROUGHPUT
GO_SLO_AWARE_COHORT_DECODE_BURST
```

A valid, complete negative experiment exits successfully and retains its
`NO_GO` classification.

## Testing

### Pure policy tests

Exhaust:

- batch sizes and width ladder;
- positive, zero, and negative slack;
- output budgets and block capacities;
- valid, missing, future, and decreasing timestamps;
- supported and unsupported graph shapes;
- every fixed fallback-precedence pair.

The selector must always choose the largest structurally eligible width whose
predicted P99 cost fits inside global slack.

### State-machine and property tests

Prove:

- one lease commits or closes at most once;
- committed tokens never exceed authorization;
- per-request write ranges do not overlap unauthorized positions;
- cohort order never changes;
- K greater than one requires all identities to be bound;
- no post-replay eager retry exists;
- every terminal path leaves zero pending leases;
- disabled mode preserves the current execution path.

### Fake-clock scheduler tests

Verify:

- idle capacity selects K8 when safe;
- shrinking ITL slack selects K4, K2, then K1;
- a newly waiting request can force immediate width contraction;
- missing SLO state and clock rollback select K1;
- preemption, finish, and EOS remove or update SLO records;
- cohort membership and order match the ordinary scheduler.

### GPU and fault-injection tests

Exercise:

- all supported B/K graph shapes;
- capture and bind failures;
- failures before and after replay;
- stale identities;
- EOS-prefix commits;
- transaction rollback;
- graph quarantine;
- exact D2H and replay accounting.

### Source-bound verification

The remote and local verifiers independently reconstruct:

- every width decision;
- every lease and result identity;
- every per-request output;
- every metric and percentile;
- every benefit and cost threshold;
- final classification.

## Artifact and Remote-Storage Rules

Remote task data is written only below:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
```

Large traces, profiler databases, and raw logs remain remote. The local
checkout retains only the compact final bundle, manifests, verifier receipts,
aggregated reports, and hashes needed for independent reconstruction.

The workflow must not write task data to remote `/`, remote `/tmp`, or a
retired checkout.

## Scope and Claim Boundary

Stage 1 includes:

- Qwen3-0.6B;
- TP1;
- multi-request continuous batching;
- zero-temperature greedy decoding;
- K1/K2/K4/K8;
- EOS-aware prefix publication;
- host-visible streaming timestamps;
- open-loop low/medium/high load;
- default-disabled operation.

Stage 1 excludes:

- K16;
- nonzero-temperature sampling;
- speculative decoding;
- TP2 or TP4;
- Qwen3-8B or Qwen3-27B;
- attention or MLP math changes;
- scheduler request reordering;
- joint chunked-prefill tuning;
- production-default enablement;
- a cross-engine superiority claim.

If the Qwen3-0.6B gate returns GO, later work requires separate designs for:

1. joint Chunked Prefill and Cohort Burst scheduling;
2. Qwen3-8B/27B promotion;
3. TP2 cohort graphs;
4. production-default qualification.

No Stage-0 ceiling result, implementation test, or Qwen3-0.6B gate may be
generalized to those scopes.
