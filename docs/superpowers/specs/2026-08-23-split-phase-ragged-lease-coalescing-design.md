# Split-Phase Ragged-Lease Coalescing Design

**Date:** 2026-08-23
**Status:** Approved under the standing autonomous-optimization authorization
**Stage-1 model:** Qwen3-0.6B
**Primary target:** remove split-phase tail and block-edge fallback churn
without exceeding K4-scale host-visible cadence

## Objective

Keep full K8 decode segments on the split-phase `4 + 4` path, but route
authorized widths `2..7` through bounded one-phase exact bursts instead of
repeatedly asking the K8-only split backend to reject them.

For the canonical 128-token workload, the exact-decode inventory contains
127 tokens:

```text
current split path:
  15 x split K8 = 120 tokens
  K7 request -> reject, ordinary K1
  K6 request -> reject, ordinary K1
  K5 request -> reject, ordinary K1
  K4 request -> reject, ordinary K1
  K3 request -> reject, ordinary K1
  K2 request -> reject, ordinary K1
  K1 request -> scheduler reject, ordinary K1

selected path:
  15 x split K8 = 120 tokens
  one-phase K4
  one-phase K3
```

The selected path removes six avoidable split-backend calls, five ordinary
decode engine steps, and five token-publication synchronizations from the
seven-token tail. It does not reduce target-model forwards: exact greedy
autoregression still requires one forward per generated token.

This is an original engineering proposal for TinyLLMForge's current
split-phase data flow. It is not presented as an academic novelty claim.

## Evidence and Motivation

The source-bound `r4` run reached the first split candidate and exposed the
real tail ownership:

- fifteen successful K8 split commits produce 120 tokens;
- widths `7..2` reach the model runner and fall back with
  `split_phase_requires_k8`;
- width `1` is rejected by the scheduler with
  `insufficient_output_budget`;
- the worker decode profile contains 28 rows, not 29, because the
  scheduler-only rejection never enters the worker profiler.

The same run measured, for the short bucket:

- K4 maximum host-visible gap: `8.780467 ms`;
- K8 maximum host-visible gap: `17.489840 ms`;
- K4 throughput: `382.896598 tokens/s`;
- K8 throughput: `406.314711 tokens/s`.

Historical canonical K8 evidence also shows that a one-phase K7 tail has a
smaller gap than a full K8 burst, but it is still materially larger than K4.
Therefore the optimization should coalesce ragged work without publishing
more than four tokens in one host-visible step.

## Scope

Stage 1 is deliberately narrow:

- TP1, rank 0, batch size 1;
- exact greedy completion-only decode;
- `temperature == 0`;
- `ignore_eos == true`;
- configured K8 exact burst with split phase enabled;
- full authorized K8 leases remain split `4 + 4`;
- ragged authorized capacity `2..7` is capped at four tokens per one-phase
  exact burst;
- ragged capacity can originate from remaining output budget or the current
  writable KV-block boundary;
- capacity `1` remains ordinary decode;
- no padding beyond output budget or writable KV positions;
- no speculative decoding, KV offload, KV quantization, Quest, or
  compact-attention composition;
- no adaptive timing controller in Stage 1.

The feature remains default-disabled until its own source-bound hardware gate
proves benefit and cost.

## Considered Approaches

### A. Unbounded ragged replay

Route every split-ineligible lease of width `2..7` directly through the
existing one-phase exact graph.

Advantages:

- smallest implementation;
- one burst handles the entire canonical seven-token tail;
- minimizes scheduler and model-runner call count.

Costs:

- a K7 publication can approach the host-visible gap of K8;
- it weakens the cadence benefit that motivated split phase;
- maximum latency is controlled only by incidental tail width.

This approach is rejected for Stage 1.

### B. K4-capped ragged-lease coalescing

When split phase is enabled for configured K8, calculate the currently
authorized capacity before issuing the lease:

```text
capacity = min(configured_width, remaining_output_tokens, writable_positions)

if capacity == 8:
    requested_width = 8       # split 4 + 4
elif 2 <= capacity <= 7:
    requested_width = min(4, capacity)  # one-phase exact burst
else:
    ordinary decode
```

The next scheduler step recomputes capacity. A seven-token tail therefore
becomes K4 followed by K3. A three-position block edge becomes K3 followed by
the next full K8 split lease.

Advantages:

- removes repeated runner fallback and ordinary-step churn;
- bounds each ragged publication at K4 cadence;
- reuses the existing exact one-phase graph, lease, validation, commit,
  continuation, and rollback contracts;
- handles both output tails and physical block edges;
- requires no padded writes and no new CUDA graph.

Costs:

- a K7 tail uses two exact bursts rather than one;
- the scheduler now selects an effective width distinct from configured K8;
- evidence and statistics must distinguish full split leases from ragged
  one-phase leases.

This is the selected approach.

### C. Padded K8 replay with masked publication

Replay eight times, but expose only the valid tail tokens.

This is rejected because the extra replays would write KV state beyond the
authorized output budget or writable block range. Hiding those tokens from
the host would not undo the unauthorized device-side mutations.

## Architecture

### 1. Pure width-selection contract

Add a dependency-light selector beside the exact-burst decision contract:

```python
select_exact_greedy_decode_burst_width(
    *,
    configured_width: int,
    remaining_output_tokens: int,
    initial_sequence_length: int,
    block_size: int,
    split_phase_enabled: bool,
    ragged_coalescing_enabled: bool,
) -> int
```

The selector returns the requested width passed into the existing decision
builder. It must not authorize a token or mutate scheduler state.

Rules:

- disabled coalescing returns `configured_width` unchanged;
- non-split configurations return `configured_width` unchanged;
- configured widths other than eight return unchanged;
- capacity eight returns eight;
- capacity `2..7` returns `min(4, capacity)`;
- capacity zero or one returns eight so the existing decision builder emits
  its current scheduler-owned fallback;
- all inputs are validated before selection.

Keeping capacity-one on the existing path preserves the rule that production
exact bursts require at least two tokens.

### 2. Scheduler authority

`Scheduler.prepare_exact_greedy_decode_burst()` computes the effective
requested width before calling `build_exact_greedy_decode_burst_decision()`.
The resulting immutable lease records that effective width in
`requested_token_count`.

The scheduler remains the sole authority for:

- remaining output budget;
- writable block capacity;
- pending lease exclusion;
- sequence status and generation;
- exact physical write range.

No model-runner heuristic may widen or split a lease after authorization.

### 3. Model-runner dispatch

`ModelRunner.run_exact_greedy_decode_burst()` dispatches by the immutable
authorized lease:

```text
split enabled and authorized_token_count == 8
    -> replay_split_phase()

authorized_token_count in 2..7
    -> existing replay()
```

The one-phase path must preserve existing continuation receipt validation,
token-history bounds, D2H accounting, sampled-logit behavior, graph identity,
and quarantine semantics.

An authorized K8 lease must never silently bypass split phase when the split
feature is enabled. An authorized width below eight must never enter the
K8-only split backend.

### 4. Engine and commit behavior

No new engine state machine is introduced.

- K8 split results continue through prefix publication and suffix drain.
- Ragged one-phase results use the existing atomic burst commit.
- Capacity one uses ordinary decode.
- A one-phase ragged commit releases its lease before the next scheduler
  decision.

The canonical expected inventory is:

```text
split parent leases:             15
split prefix commits:            15
split suffix commits:            15
ragged one-phase commits:         2
ragged widths:                  4, 3
split_phase_requires_k8:          0
insufficient_output_budget:       0
total exact graph replays:      127
target-model forwards:          127
```

The final `capacity == 3` lease consumes the entire remaining output budget,
so no capacity-one scheduler attempt is necessary.

### 5. Configuration

Add:

```text
exact_greedy_decode_burst_ragged_coalescing: bool = false
```

Stage-1 validation requires:

- ragged coalescing implies exact burst;
- ragged coalescing implies split phase;
- configured exact-burst width equals eight;

The width cap is the frozen constant four, not a Stage-1 tuning surface.
Source binding, the policy manifest, and selector tests provide evidence for
that fixed behavior.

## Correctness and Failure Semantics

The candidate must preserve:

- exact output-token equality against host greedy, K4, K8, and current split;
- bounded sampled-logit parity at the existing four sampling points;
- exact scheduler lease identity and generation checks;
- no write beyond output budget;
- no write across a physical KV-block boundary;
- no pending suffix on ragged one-phase commits;
- no split mailbox allocation or publication ticket for ragged leases;
- rollback and quarantine behavior of the selected one-phase backend;
- default-off behavior and unchanged non-split policies.

Invalid configuration fails at construction. Runtime invariant violations
use existing fallback or quarantine paths; the feature must not silently
fall back after recording a ragged lease as committed.

## Stage-1 Hardware Gate

### Matrix

- model: Qwen3-0.6B;
- one clean A100 selected by the existing local controller;
- prompt buckets: 256, 2048, and 8192 tokens;
- generated tokens: 128;
- warmups: 2;
- measured repetitions: 5;
- GPU memory utilization: 0.5;
- policies:
  - `decode_burst_k4`;
  - `decode_burst_k8_split_phase`;
  - `decode_burst_k8_split_phase_ragged`;
- performance rows: `3 policies x 3 buckets x 5 repetitions = 45`;
- correctness rows:
  `3 policies x 3 buckets x 4 sampling points = 36`.

The gate must be source-bound to the pushed commit, use a fresh run tag, keep
failed and partial artifacts, and write remote task data only under
`/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.

### Benefit thresholds

Candidate versus current split baseline:

- aggregate paired median seven-token tail latency improves by at least 10%;
- aggregate paired median TPOT does not regress by more than 1%;
- aggregate paired throughput does not regress by more than 1%;
- per-bucket median and P95 TPOT do not regress by more than 2%;
- per-bucket E2E does not regress by more than 2%.

The tail metric is the sum of the final seven entries in
`amortized_tpot_samples_ns`; because each multi-token step contributes its
elapsed time divided across emitted tokens, the sum reconstructs the
host-visible time consumed by the final seven tokens.

### Cost and cadence thresholds

- candidate median maximum host-visible gap versus K4 regresses by no more
  than 3%;
- candidate maximum observed gap versus K4 regresses by no more than 5%;
- TTFT regression versus current split is no more than 3% per bucket;
- peak allocated and reserved memory regression versus current split is no
  more than 3%;
- capture retained static bytes may not increase;
- candidate uses exactly two ragged one-phase token D2H calls per request;
- candidate introduces no new graph capture.

### Correctness and ownership thresholds

- output token IDs and output text digests are identical across policies;
- logits max absolute difference is at most `0.25`;
- logits mean absolute difference is at most `0.05`;
- producer and independent verifier numeric disagreement is at most `1e-9`;
- candidate has zero `split_phase_requires_k8` fallbacks;
- candidate has zero `insufficient_output_budget` fallbacks;
- requested and authorized width histogram is exactly
  `{"3": 1, "4": 1, "8": 15}` per request;
- split inventory remains exactly fifteen ordered prefix/suffix pairs;
- total exact committed tokens and target-model forwards equal 127;
- no block-boundary crossing, pending lease, quarantine, or split failure.

### Classification

- `GO_EXACT_BURST_RAGGED_COALESCING`
- `NO_GO_EXACT_BURST_RAGGED_COALESCING_CORRECTNESS`
- `NO_GO_EXACT_BURST_RAGGED_COALESCING_PERFORMANCE`
- `INCOMPLETE_EXACT_BURST_RAGGED_COALESCING_EVIDENCE`

Only `GO_EXACT_BURST_RAGGED_COALESCING` may authorize enabling the feature
outside the gate. Stage 1 does not make it the production default.

## Testing

Unit and contract tests must cover:

- selector disabled and non-split identity behavior;
- capacities zero and one preserving ordinary fallback;
- capacities two through seven selecting `2, 3, 4, 4, 4, 4`;
- capacity eight preserving K8 split;
- output-budget and block-boundary origins of ragged capacity;
- scheduler lease requested/authorized width identity;
- K8-only split dispatch;
- K2–K4 one-phase dispatch;
- no mailbox or split ticket for ragged commits;
- canonical K4+K3 tail inventory;
- zero old tail fallback ownership;
- rollback, stale generation, and quarantine propagation;
- producer, gate, independent verifier, controller, manifest, and source
  binding.

## Non-Goals

Stage 1 does not claim:

- fewer target-model forwards;
- sampled decoding support;
- EOS-sensitive early termination;
- multi-sequence batching;
- TP greater than one;
- a learned or timing-adaptive width policy;
- benefit for workloads whose output ends exactly on a full K8 boundary;
- production enablement before the hardware gate;
- academic novelty.
