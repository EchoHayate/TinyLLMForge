# Adaptive N-Gram Speculative Decoding Gate Design

Date: 2026-07-14

## Objective

Determine whether a small per-sequence adaptive draft-length policy makes
TinyLLMForge's existing n-gram speculative verify/commit path faster or less
wasteful than both normal greedy decoding and fixed draft lengths.

The first phase must:

1. Preserve exact greedy output through the existing proposal, target
   verification, acceptance, and commit path.
2. Select the maximum n-gram draft length independently for one sequence from
   `K in {1, 2, 4}`.
3. React quickly to first-token rejection and increase draft length only after
   sustained strong acceptance.
4. Compare adaptive `K` against fixed `K=1`, `K=2`, and `K=4` with identical
   prompts, generation settings, and target-model code.
5. Measure wall-clock performance, target verify cost, zero-accept work,
   drafted-token waste, and the selected-`K` trajectory.
6. Produce an explicit go/no-go decision before moving the policy into the
   general engine decode scheduler.

This phase is single-sequence only. It does not implement or claim ragged
batched speculative decoding, batch-aware draft scheduling, or correctness for
multiple sequences verified in one target forward.

## Motivation

TinyLLMForge already has the important correctness machinery for an online
n-gram candidate:

- `propose_ngram_draft()` produces deterministic CPU-only draft tokens.
- `verify_and_commit_block()` runs the target model, accepts only the matching
  prefix, and commits accepted tokens and KV metadata.
- `tools/profile_ngram_commit.py` records proposals, accepted tokens, target
  verify timings, output equality, and end-to-end timing.

The current profiler uses one fixed `--max-draft-tokens` value for the whole
request. That is inefficient when acceptance quality changes during a
generation:

- a large `K` can avoid more autoregressive steps in repetitive regions;
- the same large `K` wastes target verification work in low-repeat or
  transition regions;
- a permanently small `K` limits downside but leaves high-acceptance regions
  underexploited.

Historical measurements demonstrate both regimes. A short Qwen3-0.6B warm
smoke with fixed n-gram size 5 and `K=4` reached only `0.5625` acceptance and
improved from `27.33 tok/s` to `28.65 tok/s`. A synthetic repeated
long-context case reached `0.9038` acceptance and improved from `25.29 tok/s`
to `50.54 tok/s`. The latter is useful evidence that accepted multi-token
commits can help, but its repetition is too synthetic to justify a global
fixed `K=4` policy.

The bounded question is therefore not whether speculative decoding can work in
principle. It is whether a transparent online controller can retain useful
large drafts while reducing rejection-heavy verification waste.

## Decision

Use a deterministic per-sequence AIMD/EMA controller over the discrete levels
`{1, 2, 4}`. The controller changes only the maximum token count passed to the
existing n-gram proposal function.

It does not change:

- which tokens the n-gram matcher proposes;
- target-model logits or sampling;
- accepted-prefix calculation;
- KV reservation or commit behavior;
- finish and EOS handling;
- the normal target decode step after an unsuccessful proposal.

All fixed and adaptive candidates therefore share the same correctness path.

## Adaptive Policy

### State

Each speculative candidate sequence owns an `AdaptiveDraftState` with:

```text
levels = (1, 2, 4)
level_index = 1                 # initial K = 2
acceptance_ema = 0.5
full_accept_streak = 0
proposal_events = 0
```

The policy is profiler-owned in this phase. No state is added to the production
scheduler or `Sequence`.

### Observation

The state updates only after a non-empty proposal has been target-verified.
For one verify event:

```text
proposed = len(draft_tokens)
accepted = accepted_count
event_acceptance = accepted / proposed
acceptance_ema =
    0.5 * event_acceptance + 0.5 * previous_acceptance_ema
```

No-match decode positions do not update the EMA or selected level because they
provide no evidence about whether a different draft cap would have helped.

### Transition Rule

Transitions happen after the current verify/commit event and affect only the
next proposal:

1. **First-token rejection:** if `accepted == 0`, reset
   `full_accept_streak=0` and jump directly to `K=1`.
2. **Weak partial acceptance:** otherwise, if `event_acceptance < 0.5` or the
   updated `acceptance_ema < 0.5`, reset the streak and move down one level.
3. **Strong full acceptance:** if `accepted == proposed`, increment
   `full_accept_streak`. When the updated EMA is at least `0.75` and the streak
   reaches two, move up one level and reset the streak.
4. **Hold:** all other outcomes keep the current level. A partial acceptance
   always resets `full_accept_streak`.

Level changes saturate at `K=1` and `K=4`.

This is additive increase over the ordered levels and aggressive multiplicative
decrease on complete rejection. The two-event promotion rule prevents one easy
match from immediately expanding target verification work.

### Short Proposal Semantics

The selected `K` is a cap, not a promise. If the n-gram source can produce only
one token while the state selects `K=4`, the event records:

```text
selected_k = 4
proposed_tokens = 1
```

Acceptance and waste use the actual proposal length. Promotion may still occur
after two fully accepted short proposals because those are valid positive
observations, but the gate separately reports selected-`K` and actual proposal
length distributions.

## Alternatives Considered

### 1. Offline Oracle Followed by Fitted Rules

Replay fixed `K` choices against completed generations, choose the best `K` at
each position, and fit a policy from those labels.

This can estimate an upper bound, but the oracle sees future target tokens and
latency outcomes. A fitted policy can therefore inherit hindsight bias, and a
small prompt bank is insufficient for a trustworthy learned rule. Offline
replay may be added later as analysis, but it is not the first online policy.

### 2. Learned Latency or Acceptance Predictor

Train a model using hidden states, token features, recent acceptance, batch
load, and target timings to select `K` or disable speculation.

This may eventually outperform a hand-written controller, but it introduces
training data, feature stability, model overhead, calibration, and deployment
questions before TinyLLMForge has established a robust adaptive baseline.

### 3. Batch- and Load-Aware `K=0..N`

Select both whether to speculate and how many tokens to draft using current
batch size, queue pressure, and a ragged batched target verifier.

This is the likely production direction if the single-sequence gate succeeds.
It is excluded now because `profile_ngram_commit.py` verifies each sequence
individually. Treating that loop as a batched verifier would overstate both
correctness coverage and performance relevance.

## Scope

### Included

- A pure, unit-tested adaptive draft-length state machine.
- Fixed and adaptive policy dispatch for the n-gram draft source.
- Per-event recording of selected `K`, previous and updated EMA, transition
  reason, proposed tokens, accepted tokens, and waste.
- Single-sequence baseline, fixed-policy, and adaptive-policy remote runs.
- A deterministic committed prompt bank spanning low-, mixed-, and
  high-repeat behavior.
- Repeated isolated-process measurements on Qwen3-0.6B.
- Exact output-token comparison against normal greedy decoding.
- Canonical JSON/Markdown artifacts with a go/no-go decision.
- README and `AGENT_HANDOFF_STATE.md` updates after the result is known.

### Excluded

- Multiple candidate sequences in one target verify forward.
- Ragged batch metadata, masks, or KV reservation.
- Batch-size, queue-load, or memory-pressure policy inputs.
- A `K=0` speculation bypass decision.
- Learned policies or offline-oracle-trained thresholds.
- N-gram lookup algorithm changes.
- Production scheduler or public `LLM.generate()` integration.
- Non-greedy sampling equivalence.
- Claims for models or hardware not measured by the gate.

## Components

### 1. Adaptive Policy Module

`tinyvllm/speculative/ngram.py` will define the policy state and a side-effect
limited update helper. The helper accepts the current state plus actual
`proposed` and `accepted` counts and returns a JSON-friendly transition record.

The module remains independent of Torch, CUDA, the scheduler, and KV-cache
objects. Unit tests can therefore cover every boundary transition without
loading a model.

Required transition tests include:

1. initial `K=2`;
2. two strong full accepts promote `2 -> 4`;
3. promotion saturates at 4;
4. first-token rejection jumps `4 -> 1`;
5. weak partial acceptance moves `4 -> 2`;
6. weak EMA moves `2 -> 1`;
7. partial acceptance resets the promotion streak;
8. no-match positions leave state unchanged;
9. invalid counts are rejected;
10. transition summaries are JSON-friendly.

### 2. Proposal Policy Dispatch

`tools/profile_ngram_commit.py` will distinguish:

```text
draft_policy = fixed
draft_policy = adaptive
```

For fixed policies, `--max-draft-tokens` retains its current meaning. For the
adaptive policy, the current sequence state's selected `K` is supplied to
`propose_ngram_draft()` for that event.

The adaptive option is valid only with `--draft-source ngram`. Other profiler
draft sources continue to use their existing fixed cap and reject an adaptive
configuration with a clear CLI error.

### 3. Verify/Commit Integration

The existing `verify_and_commit_block()` remains the sole target verification
and accepted-token commit implementation.

The adaptive loop:

1. reads the sequence's current selected `K`;
2. requests an n-gram proposal capped by that value;
3. runs the unchanged target verify/commit function;
4. updates policy state from actual proposed and accepted counts;
5. attaches the policy transition record to the verify event.

An empty n-gram proposal skips target verification and records a no-draft
position without updating policy state.

### 4. Gate Driver

A dedicated gate driver under `tools/` will orchestrate separate profiler
processes for:

- normal greedy baseline;
- fixed `K=1`;
- fixed `K=2`;
- fixed `K=4`;
- adaptive `K in {1,2,4}`.

Each process loads the same model configuration and runs exactly one sequence
at a time. The driver randomizes candidate order within each repetition using a
recorded deterministic seed, performs one untimed warmup, and writes raw rows
before computing summaries.

The driver must not use the profiler's paired two-sequence mode as primary
wall-clock evidence. Paired mode is useful for local output checks but mixes
baseline and speculative work in one scheduler loop.

### 5. Prompt Bank

The implementation will commit a deterministic single-sequence prompt bank
with literal prompt text, expected workload class, maximum output length, and a
stable prompt hash. It must include at least:

1. natural prose with low expected repetition;
2. structured list or code continuation with mixed repetition;
3. repeated long-context text with high expected repetition;
4. a transition-heavy prompt intended to expose zero-accept events.

The bank is a gate fixture, not a claim of production representativeness.
Artifacts must report results per prompt class so the synthetic repeated case
cannot dominate or hide regressions in natural prompts.

## Metrics

Each raw candidate row records:

- model path and resolved model identifier;
- source commit and dirty-state flag;
- prompt name, class, hash, token count, and output token count;
- repetition, randomized run order, and seed;
- policy and fixed `K` where applicable;
- exact output token IDs or their stable hash plus equality result;
- elapsed wall-clock seconds and output tokens per second;
- proposal events and no-draft positions;
- drafted, accepted, and wasted draft tokens;
- acceptance rate and draft-waste rate;
- zero-accept event count and rate;
- total target verify/commit timing and its existing subcomponents;
- selected-`K` counts and transition trajectory for adaptive runs;
- autoregressive steps avoided;
- gate failure reasons.

Draft waste is:

```text
wasted_draft_tokens = drafted_tokens - accepted_tokens
draft_waste_rate = wasted_draft_tokens / drafted_tokens
```

Zero-accept verify cost is the sum of `verify_commit_total_ms` for events with
`accepted_count == 0`.

Aggregate performance uses total output tokens divided by total wall-clock
seconds across all prompt classes within one repetition. The canonical summary
reports the median across seven measured repetitions. Per-prompt medians remain
visible and are never replaced by only the aggregate.

## Correctness and Decision Gate

### Mandatory Correctness

The gate is an unconditional `NO_GO` if any of the following occurs:

- a fixed or adaptive candidate's greedy output token IDs differ from its
  isolated normal-decode baseline;
- target verify/commit reports an internal failure;
- output length differs before an allowed EOS termination;
- an adaptive event selects a value outside `{1,2,4}`;
- the event trajectory cannot be reproduced from its recorded transitions;
- any measured process uses more than one active sequence.

All correctness cases must pass in all seven measured repetitions.

### Adaptive-Policy Exercise

The result is `NO_GO` for adaptation if the canonical workload does not:

- produce at least one non-empty proposal in every repeat-capable prompt;
- exercise at least two selected `K` levels across the suite;
- include at least one promotion or demotion event.

This prevents a fixed `K` in disguise from passing as adaptive evidence.

### Performance Decision

After mandatory correctness passes, the adaptive policy is `GO` only when:

1. median aggregate throughput is at least 5% above isolated normal greedy
   decoding; and
2. adaptive throughput is either:
   - at least 2% above the best fixed-`K` aggregate throughput, or
   - within 1% of the best fixed policy while reducing both drafted-token waste
     by at least 20% and zero-accept verify cost by at least 15% relative to
     fixed `K=4`; and
3. no natural or transition-heavy prompt is more than 5% slower than normal
   greedy decoding by its seven-run median.

Thresholds are fixed before measurement and must not be relaxed after seeing
the result. A correctness-safe result that misses them is recorded as
`NO_GO`, not reclassified as a success.

The gate may separately report that fixed n-gram speculation is useful even if
adaptation is `NO_GO`, but it must not conflate those conclusions.

## Artifact Layout

Canonical output will live under:

```text
experiments/adaptive_ngram/<model-and-date>/
```

and contain:

- `manifest.json`: source/model/environment, prompt bank, commands, seeds, and
  fixed thresholds;
- `raw_rows.json`: one row per process run;
- `event_rows.json`: proposal, verify, and adaptive transition events;
- `summary.json`: per-policy and per-prompt medians plus decision fields;
- `report.md`: concise interpretation, limitations, and next direction.

The driver writes artifacts atomically enough that a partial run cannot be
mistaken for canonical completion. `summary.json` includes expected and
observed row counts, and the decision remains `INCOMPLETE` until every required
row and event audit passes.

## Remote Execution

Remote validation uses:

```text
sitian@10.232.195.203
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python
```

The gate runner uploads only the required source files into an isolated remote
directory and does not modify or trust an existing remote checkout.

Before running, it must discover and record the actual Qwen3-0.6B model path on
the remote host rather than assuming whether the directory is spelled
`Qwen3-0.6B` or `Qwen3-0___6B`.

Every model process sets distinct dynamic values for both:

```text
TINYVLLM_DIST_PORT
MASTER_PORT
```

Remote failures, missing model files, incomplete repetitions, or port
collisions produce `INCOMPLETE`, not `NO_GO`.

## Error Handling

- Invalid policy parameters fail before model loading.
- `accepted > proposed`, negative counts, or unknown levels raise explicit
  errors in the pure policy helper.
- Adaptive policy with a non-n-gram draft source is rejected.
- A failed subprocess records stdout, stderr, exit code, command, and run key.
- Missing rows or duplicate run keys fail artifact validation.
- Non-finite timings fail summary generation.
- An interrupted run may resume only by preserving already validated unique
  rows; it must not silently average partial duplicates.

## Testing and Validation

### Local

- Dependency-light policy transition tests in
  `tools/test_ngram_speculative.py`.
- Profiler helper and argument-validation tests.
- Gate aggregation and threshold tests with synthetic rows.
- Artifact completeness and trajectory-replay tests.
- `python3 -m py_compile` for changed Python files.
- `bash -n` for the remote runner.
- `git diff --check`.

### Remote

1. One-repetition smoke validating model discovery, dynamic ports, all five
   policies, and artifact shape.
2. Exact-output audit for every smoke row.
3. Seven-repetition canonical run.
4. Independent artifact verifier that recomputes row counts, trajectories,
   medians, deltas, and the final decision from raw rows.
5. Post-download verification from the committed local artifact directory.

## Claim Boundaries

A `GO` means only that the specified single-sequence adaptive n-gram policy
passed the committed Qwen3-0.6B prompt bank and hardware gate.

It does not establish:

- production batch throughput improvement;
- ragged batched speculative correctness;
- serving-tail-latency improvement under queueing;
- memory-capacity reduction;
- transfer to other models, sampling modes, or accelerators;
- superiority over draft-model methods such as EAGLE, Medusa, or DFlash.

If this gate passes, the next design should add a real ragged batched target
verify path and a load-aware `K=0..N` policy. If it fails, retain reusable
correctness and measurement improvements, record the negative result, and
prefer the best measured fixed policy or move to a higher-quality draft source.
