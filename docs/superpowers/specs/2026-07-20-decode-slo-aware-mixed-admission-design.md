# Decode-SLO-Aware Mixed Admission Design

Date: 2026-07-20

## Terminology

This document defines experimental arrival-load policy `P5`.

Its stable descriptive name is:

```text
decode_slo_aware_mixed_admission
```

`P5` is a successor experiment to `P4`
`sam_backlog_adaptive_mixed_prefill`. It does not replace, promote, or change
the classification of `P4`.

In this document:

- **decode progress age** is the elapsed monotonic time since a runnable
  sequence most recently produced a completion token;
- **decode gap target** is the configured maximum progress age that the
  scheduler attempts to preserve;
- **slack** is the decode gap target minus the oldest runnable decode progress
  age and a fixed safety reserve;
- **cost envelope** is a source-bound, environment-bound upper estimate of
  synchronous mixed-step duration as a function of admitted prefill tokens;
- **SLO suppression** means choosing a decode-only step because no admissible
  mixed chunk fits inside the remaining slack.

This is a scheduler experiment. It is not an external request router,
multi-tenant priority system, or production service-level guarantee.

## Objective

Add a disabled-by-default scheduler policy that preserves the useful burst
region of mixed prefill-plus-decode while preventing prefill admission from
consuming the remaining decode progress budget.

The first phase must:

1. Reuse the existing transactional mixed prefill-plus-decode implementation.
2. Reuse the proven `P4` backlog hysteresis only as a demand gate.
3. Track decode progress with a monotonic clock owned by the engine process.
4. Compute the oldest runnable decode progress age before every scheduling
   decision.
5. Use a frozen, source-bound cost envelope to choose the largest safe prefill
   chunk from a fixed descending token ladder.
6. Schedule decode-only when no mixed chunk fits within remaining slack.
7. Keep all admission, queue ownership, KV reservation, and `may_append()`
   mutations transactional.
8. Preserve exact greedy output, lifecycle, prefix-cache semantics, block
   accounting, and repository-default behavior when disabled.
9. Expose enough immutable decision evidence for an independent verifier to
   reconstruct every P5 choice without trusting scheduler labels.
10. Compare `P5` with repository default `P0` and predecessor `P4` under a new
    source-bound remote chain.
11. Permit a performance claim only if the independent `P5` verifier returns
    `GO`.

The first phase does not:

- learn a policy online;
- tune thresholds from canonical results;
- predict individual request completion time;
- add weighted priorities or user-visible service classes;
- split prefill and decode across GPUs;
- change kernels, attention preparation, CUDA Graph capture, or model weights;
- combine P5 with KV offload, speculative decoding, or quantization.

## Evidence and Problem Statement

The authoritative P4 canonical artifact is:

```text
experiments/arrival_load/qwen3-06b-sam-p4-canonical-v2-20260720-142635
```

Its independent verifier completed successfully and classified P4 as
`NO_GO`, with no correctness or structural failures.

P4 preserved a useful burst region:

- burst request-throughput ratios were approximately `1.99x` to `2.40x`;
- burst TTFT and E2E latency improved materially.

However, backlog hysteresis plus a fixed "two mixed steps, one decode yield"
rule did not protect decode tails:

- median p95 ITL ratio: `1.225369`;
- worst p95 ITL ratio: `12.327660`;
- worst p99 ITL ratio: `5.889306`;
- worst maximum decode-gap ratio: `7.641892`;
- worst request-throughput ratio: `0.954690`;
- median peak KV-byte ratio: `3.142857`.

The strongest failure was `mixed_service_fairness r0`. Five service buckets
had p95 E2E ratios from `8.241x` to `38.662x`. Long-prompt pressure also had
p95 ITL ratios from approximately `3.77x` to `3.92x` in all three
repetitions.

The final traces show that P4 was exercised rather than dormant. In
`mixed_service_fairness r0`, P4 executed:

```text
35 mixed prefill+decode steps
16 forced decode-yield steps
76 decode-fallback steps
maximum waiting depth 15
```

The synchronous step-duration audit further shows why a fixed step-count
yield is too coarse:

```text
P0 decode median       3.448 ms
P0 decode p95          4.476 ms
P0 decode maximum     75.814 ms
P4 decode median       3.478 ms
P4 decode p95          9.225 ms
P4 decode maximum    330.203 ms
P4 mixed median       35.823 ms
P4 mixed p95          61.084 ms
P4 mixed maximum     370.166 ms
```

Two mixed steps can therefore consume a few milliseconds or hundreds of
milliseconds. Counting steps cannot bound elapsed decode delay.

## Industry Direction

The design follows three established serving principles without importing an
external serving stack:

1. Sarathi-Serve uses chunked prefill and a token budget to reduce generation
   stalls caused by long prefills.
2. vLLM's chunked-prefill scheduling documentation describes prioritizing
   decode work and filling the remaining token budget with prefills.
3. DistServe treats prefill/decode interference as an SLO problem, but solves
   it with phase disaggregation; that larger architecture is out of scope for
   this single-GPU scheduler experiment.

Primary references:

```text
Sarathi-Serve: https://arxiv.org/abs/2403.02310
DistServe:     https://arxiv.org/abs/2401.09670
vLLM docs:    https://docs.vllm.ai/en/latest/configuration/optimization.html
```

These references motivate elapsed-time and token-budget control. They do not
prove that P5 will improve TinyLLMForge; only the repository's source-bound
gate can do that.

## Alternatives Considered

### 1. Recommended: Frozen Cost Envelope Plus Per-Sequence Slack

Run a dedicated source-bound remote cost calibration before canonical
execution. Freeze a conservative mixed-step duration envelope, then admit the
largest prefill chunk whose predicted duration fits under the oldest runnable
decode sequence's remaining slack.

Advantages:

- directly controls elapsed decode exposure rather than counting steps;
- fails closed before queue/KV mutation;
- deterministic for a frozen source, environment, calibration artifact, and
  queue snapshot;
- independently reconstructable;
- preserves the existing mixed model-runner path;
- changes prefill chunk size only when mixed mode is active.

Risks:

- a cost envelope calibrated on one model/GPU cannot be reused elsewhere;
- an envelope can underpredict unseen batch shapes;
- conservative upper bounds may suppress most mixed opportunities;
- wall-clock state adds an explicit engine/scheduler interface.

### 2. Reactive EWMA Controller

Estimate mixed-step cost from recently observed production steps and shrink
or expand the next chunk online.

This is rejected for phase one:

- the first large mixed step can violate the decode target before feedback
  exists;
- decisions become history-sensitive in ways that are harder to reproduce;
- canonical repetitions could diverge because of timing noise;
- verifier reconstruction would need to reproduce floating-point EWMA and
  initialization details.

### 3. Prefill/Decode Disaggregation

Run prefill and decode on separate devices and transfer KV state.

This is deferred:

- it changes deployment topology and resource accounting;
- it requires KV transfer, placement, and backpressure design;
- it cannot be validated as a small successor to the current single-GPU gate;
- the current objective is to determine whether safe mixed admission can
  preserve the already observed burst benefit.

## Decision

Implement Alternative 1 only after this written design is reviewed and a
separate implementation plan is approved.

P5 must remain disabled by default and mutually exclusive with P3, P4, KV
offload, and other experimental scheduler modes.

## Policy Matrix

The P5 canonical comparison is exactly:

```text
P0  repository-default decode-first chunked prefill
P4  predecessor backlog-adaptive mixed policy, diagnostic only
P5  decode-SLO-aware mixed admission, only promotable candidate
```

The matrix remains:

```text
6 scenarios × 3 policies × 3 measured repetitions = 54 cases
```

The six existing scenarios and frozen workload generator remain unchanged:

```text
steady_moderate
near_saturation
overload
burst
long_prompt_pressure
mixed_service_fairness
```

P4 cannot produce a top-level `GO`. The top-level classification is exactly
the independently recomputed P5 classification.

## Configuration Contract

Add these fields:

```python
chunked_prefill_slo_mixed: bool = False
chunked_prefill_slo_target_gap_ns: int = 0
chunked_prefill_slo_reserve_ns: int = 0
chunked_prefill_slo_cost_intercept_ns: int = 0
chunked_prefill_slo_cost_per_prefill_token_ns: int = 0
chunked_prefill_slo_min_chunk_tokens: int = 16
```

P5 also consumes the existing P4 demand-gate fields:

```python
chunked_prefill_adaptive_enter_waiting
chunked_prefill_adaptive_exit_waiting
chunked_prefill_adaptive_transition_steps
```

It does not consume
`chunked_prefill_adaptive_max_mixed_steps`; P5 replaces fixed mixed-step
counting with elapsed-time slack.

Validation is fail closed:

- `chunked_prefill_slo_mixed=False` preserves current behavior;
- when enabled, target, reserve, intercept, per-token cost, and minimum chunk
  must all be positive;
- `reserve_ns < target_gap_ns`;
- `min_chunk_tokens <= max_num_prefill_tokens_per_step`;
- the maximum prefill chunk must be divisible by the minimum chunk size;
- P5 requires chunked prefill;
- P5 and P3 always-on mixed are mutually exclusive;
- P5 and P4 are mutually exclusive;
- P5 and KV offload are mutually exclusive;
- calibration values must fit signed 64-bit nanoseconds and multiplication
  must be overflow-safe;
- invalid configuration fails before model startup.

The policy identity includes all P5 fields, all consumed demand-gate fields,
the exact token ladder, and the cost-calibration artifact SHA-256.

The first canonical P5 candidate is preregistered as:

```text
chunked_prefill_slo_target_gap_ns        64_000_000
chunked_prefill_slo_reserve_ns            8_000_000
chunked_prefill_slo_min_chunk_tokens             16
max_num_prefill_tokens_per_step                 128
```

Only the cost intercept and per-token slope come from source-bound remote
calibration. The target, reserve, minimum chunk, and maximum chunk are not
changed after smoke or canonical observation.

## Source-Bound Cost Calibration

P5 must not contain guessed hardware constants.

The remote chain gains a dedicated `cost-calibration` stage after smoke and
before workload calibration:

```text
preflight
→ smoke
→ cost-calibration
→ workload-calibration
→ canonical
→ local independent verify
```

Cost calibration runs on the same:

- source tree;
- remote Python;
- model path;
- GPU identity;
- dtype and engine configuration;
- `max_num_seqs`;
- `max_num_batched_tokens`;
- `max_num_prefill_tokens_per_step`.

It uses isolated model processes and unique dynamic ports.

### Calibration Shapes

Measure synchronous step duration for:

- decode-only batches at runnable decode counts `1`, `8`, `32`, and the largest
  feasible count not exceeding `max_num_seqs`;
- mixed batches at prefill-token counts `16`, `32`, `64`, and `128`;
- each mixed token count at decode-row counts `1`, `8`, `32`, and the largest
  feasible count;
- each mixed token count at prefill-row counts `1` and
  `min(8, prefill_tokens / 16)`, with duplicate one-row shapes removed;
- at least one short-, medium-, and long-context decode shape;
- warmup plus at least seven measured iterations per shape.

If a configured shape is infeasible because of model length or KV capacity,
the stage is `INCOMPLETE`; it must not silently omit the shape.

### Integer Cost Envelope

For every measured shape, compute the nearest-rank p99 synchronous step
duration. Seven measured iterations make this p99 the observed maximum for
that shape.

Inflate every measured point with integer 25% headroom:

```text
inflated_duration_ns = ceil(measured_p99_ns * 5 / 4)
```

Freeze an integer upper envelope:

```text
predicted_step_ns(tokens) =
    cost_intercept_ns
    + tokens * cost_per_prefill_token_ns
```

where:

- `cost_intercept_ns` is at least the maximum inflated decode-only duration
  across all calibrated decode-row/context shapes;
- `cost_per_prefill_token_ns` is the smallest positive integer slope for which
  every inflated mixed point lies at or below the envelope;
- all division rounds upward;
- the published envelope must dominate every inflated point;
- the independent verifier recomputes both coefficients from raw calibration
  rows.

The calibration stage is `INCOMPLETE` if:

- any required process fails;
- any duration is non-positive or non-finite;
- any required shape is missing;
- the integer envelope overflows;
- the published coefficients do not dominate every measured point;
- source, environment, or engine configuration identity differs from the
  subsequent canonical run.

Canonical results must never modify or refit the envelope.

## Monotonic Time Contract

P5 requires one explicit monotonic clock domain.

`LLMEngine` owns the real clock:

```python
time.monotonic_ns
```

Tests inject a fake integer clock.

For each `step()`:

1. sample `decision_now_ns` immediately before `Scheduler.schedule()`;
2. execute the synchronous model-runner call;
3. sample `step_end_ns` immediately after the call returns;
4. pass `step_end_ns` into scheduler postprocess;
5. update decode progress only for rows that actually produced a completion
   token.

The model-runner call is treated as synchronous only because the current path
synchronizes CUDA before returning. If that contract changes, P5 must fail
closed until end-of-step timing is re-established.

The scheduler never calls a second clock while making one decision. Every
calculation for that decision uses the single supplied `decision_now_ns`.

Clock validation:

- both timestamps are non-negative integers;
- `step_end_ns >= decision_now_ns`;
- decision timestamps never move backward;
- progress timestamps never exceed the current decision timestamp;
- a clock violation sets a sticky invalid flag and P5 schedules decode-only
  for the remainder of the engine lifetime;
- the invalid flag and reason are emitted in every observation snapshot.

No wall-clock time, Unix time, or driver-side timestamp may enter P5
decisions.

## Decode Progress State

The scheduler owns:

```python
decode_progress_ns_by_seq_id: dict[int, int]
slo_clock_invalid: bool
slo_clock_invalid_reason: str | None
last_slo_decision: immutable snapshot
```

Progress initialization:

- a request that finishes prefill and produces its first completion token gets
  progress timestamp `step_end_ns`;
- a normal decode row that produces a token gets `step_end_ns`;
- a mixed decode row that produces a token gets `step_end_ns`;
- a prefill-only intermediate chunk does not create decode progress.

Lifecycle:

- preemption preserves a sequence's existing progress timestamp;
- a preempted sequence is excluded while it is not in `running`;
- when it becomes runnable again, its preserved age immediately participates
  in protection;
- finishing removes the sequence's progress entry;
- an empty engine clears all P5 timing state;
- a runnable decode sequence without progress state makes the current
  decision fail closed to decode-only and records
  `missing_decode_progress`.

The map is scheduler-local and keyed by stable `seq_id`; it is not serialized
inside `Sequence` or sent to tensor-parallel workers.

## Demand Gate

P5 reuses the P4 `inactive / active / draining` backlog controller:

- enter active after two eligible observations with waiting depth at least 8;
- stop new admission after two eligible observations with waiting depth at
  most 2;
- drain already-admitted prefill work without admitting another waiting
  request;
- reset controller state only when the engine is empty.

The controller samples `len(waiting)` once at the beginning of the scheduling
decision.

Demand state answers only whether mixed admission is potentially useful.
SLO slack independently answers whether it is safe now.

An active demand state never overrides an SLO suppression.

## Slack Computation

For one scheduling decision:

```text
oldest_progress_ns =
    min(decode_progress_ns_by_seq_id[seq_id]
        for seq in running)

oldest_decode_age_ns =
    decision_now_ns - oldest_progress_ns

remaining_slack_ns =
    target_gap_ns
    - reserve_ns
    - oldest_decode_age_ns
```

If `remaining_slack_ns <= 0`, schedule decode-only.

The reserve is not learned or changed online. It absorbs scheduler overhead,
Python bookkeeping, and cost-envelope residual risk.

P5 does not average ages. One oldest runnable sequence is sufficient to
suppress mixed admission.

The sequence identified by `oldest_decode_seq_id` is the protected decode
row for the current decision. A P5 mixed batch is valid only if it includes
that exact sequence and produces one completion token for it.

## Safe Chunk Selection

The fixed descending token ladder is generated from configuration:

```text
max_chunk,
max_chunk - min_chunk,
...,
min_chunk
```

For the canonical configuration:

```text
128, 112, 96, 80, 64, 48, 32, 16
```

For each candidate token count in descending order:

```text
predicted_step_ns =
    cost_intercept_ns
    + candidate_tokens * cost_per_prefill_token_ns
```

Choose the first candidate satisfying:

```text
predicted_step_ns <= remaining_slack_ns
```

Then pass that exact value as the maximum prefill-token budget into the
existing transactional mixed helper.

If no candidate fits, schedule decode-only.

If the helper cannot form a real mixed batch containing at least one prefill
row and one decode row, schedule decode-only.

The helper may return fewer prefill tokens than the selected budget because
of prompt completion, prefix reuse, sequence limits, or KV capacity. It may
never exceed the selected budget.

## Transactional Scheduling Contract

Before SLO approval, P5 may only read:

- queues and sequence metadata;
- decode progress timestamps;
- block-manager capacity estimates;
- frozen configuration;
- frozen cost-envelope values.

Before selecting a safe chunk, it must not:

- pop or reorder `waiting`, `prefilling`, or `running`;
- allocate or deallocate KV blocks;
- call `may_append()`;
- mutate sequence status;
- mutate chunk boundaries;
- publish prefix hashes;
- increment controller counters that depend on a successful mixed admission.

After selecting a safe chunk, P5 calls the existing transactional mixed helper
with:

```text
require_decode=True
required_decode_seq_id=<oldest_decode_seq_id>
max_prefill_tokens=<selected safe budget>
allow_waiting_admission=<active, not draining>
```

The existing decode reservation and rollback-free ordering remain mandatory:

1. identify the protected oldest decode row read-only;
2. reserve its potential free-block requirement;
3. select and commit bounded prefill admission;
4. locate the same protected decode sequence;
5. call `may_append()` for that sequence;
6. add additional feasible decode rows;
7. emit a batch only if both prefill and decode rows exist.

If the protected row cannot be reserved, found, or appended under the
read-only capacity estimate, P5 suppresses mixed admission and schedules
decode-only. It must not substitute a younger decode row while still claiming
the oldest sequence's slack.

## Scheduler Branches

P5 emits stable branch names:

```text
slo_mixed_no_running_prefill
slo_mixed_inactive_decode
slo_mixed_clock_invalid_decode
slo_mixed_missing_progress_decode
slo_mixed_no_slack_decode
slo_mixed_cost_suppressed_decode
slo_mixed_transaction_fallback_decode
slo_mixed_prefill_decode
slo_mixed_draining_prefill_decode
```

Every branch is derived from independently visible state. Branch labels are
diagnostic and never accepted as proof by themselves.

## Immutable Decision Evidence

For every P5 scheduling decision, publish:

```text
decision_now_ns
target_gap_ns
reserve_ns
oldest_decode_seq_id
oldest_decode_progress_ns
oldest_decode_age_ns
remaining_slack_ns
cost_intercept_ns
cost_per_prefill_token_ns
candidate_chunk_tokens
predicted_step_ns
selected_chunk_tokens
actual_prefill_tokens
scheduled_decode_seq_ids
demand_state_before
demand_state_after
suppression_reason
clock_invalid
clock_invalid_reason
```

The decision snapshot is immutable after `schedule()` returns. Postprocess may
append:

```text
step_end_ns
actual_step_duration_ns
decode_progress_updates
finished_progress_entries_removed
```

It may not rewrite the pre-execution decision fields.

`LLMEngine.last_step_observation` carries the complete snapshot into the
arrival-load trace.

## Independent Verifier

The verifier must not trust:

- policy branch strings;
- published age/slack values;
- selected chunk size;
- predicted duration;
- progress updates;
- cost-envelope coefficients;
- top-level classification.

It independently:

1. verifies P5 resolved configuration and policy identity;
2. recomputes the cost envelope from raw calibration rows;
3. verifies source/environment/config identity across all stages;
4. reconstructs progress timestamps from token-producing trace rows;
5. reconstructs active/draining demand transitions;
6. recomputes the oldest runnable decode sequence for every decision;
7. recomputes age and remaining slack from integer timestamps;
8. reconstructs the descending token ladder;
9. proves that the selected chunk is the largest safe candidate;
10. proves that suppressed decisions schedule no prefill row;
11. proves that mixed decisions contain both prefill and decode rows;
12. proves every mixed decision includes the reconstructed oldest decode row;
13. proves actual prefill tokens do not exceed selected tokens;
14. proves every actual mixed duration is recorded and finite;
15. detects any mixed-step envelope underprediction;
16. recomputes all request, fairness, memory, and performance metrics;
17. classifies P5 independently.

Any transition, timing, envelope, mutation-order, or branch mismatch is a
structural failure.

An actual mixed duration exceeding its predicted envelope does not
automatically make evidence incomplete if the trace is otherwise valid. It is
a P5 correctness-of-controller failure and therefore `NO_GO`, because the
policy admitted work on an invalid safety premise.

## Local TDD Contract

Dependency-light scheduler tests must cover:

- disabled defaults;
- all invalid configuration combinations;
- one monotonic timestamp sampled per decision;
- initial first-token progress creation;
- normal decode progress update;
- mixed decode progress update;
- intermediate prefill not updating progress;
- preemption preserving but excluding progress;
- finish removing progress;
- engine-empty reset;
- clock regression sticky fail-closed behavior;
- missing runnable progress fail-closed behavior;
- exact integer age/slack arithmetic;
- overflow rejection;
- descending token-ladder construction;
- largest-safe-chunk selection at exact boundaries;
- no-safe-chunk decode-only suppression;
- demand-active but SLO-suppressed behavior;
- draining without new waiting admission;
- transactional no-mutation before SLO approval;
- protected oldest decode reservation and inclusion;
- protected-row reservation failure suppressing mixed admission;
- actual prefill tokens bounded by selected budget;
- mixed batch requiring both row types;
- P0, P3, and P4 behavior unchanged.

Synthetic verifier tests must tamper each independently reconstructed field:

- decision time;
- progress timestamp;
- oldest sequence identity;
- age;
- slack;
- coefficient;
- selected chunk;
- protected decode sequence identity;
- actual prefill count;
- suppression reason;
- progress update;
- calibration row;
- source/environment/workload identity.

Every tamper case must fail closed.

## Remote Evidence Chain

All model/GPU work remains on:

```text
host    sitian@10.232.195.203
python  /data00/home/sitian/sitian-workspace01/tllm/env/bin/python
model   /data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B
```

Use:

```text
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian
```

Every model process receives unique dynamic:

```text
TINYVLLM_DIST_PORT
MASTER_PORT
```

The runner must not:

- mutate the remote checkout;
- use rsync;
- kill unrelated processes;
- clear shared `/tmp`;
- reuse calibration or canonical evidence across source identities;
- reuse P4 canonical rows as P5 evidence;
- modify thresholds or coefficients after observing canonical results.

Any source change after preflight requires a complete new:

```text
preflight
→ smoke
→ cost-calibration
→ workload-calibration
→ canonical
→ local independent verify
```

## Smoke Contract

Smoke uses only P0 and P5 and must prove:

- source/environment identity;
- cost-calibration identity;
- at least one demand activation;
- at least one safe admission at the largest chunk allowed by the frozen
  envelope and 64 ms target;
- at least one smaller selected chunk;
- at least one no-slack or cost-suppressed decode-only decision;
- at least one draining decision;
- exact outputs and complete lifecycle;
- independent verifier success.

If the frozen cost envelope suppresses every mixed step, smoke is
`INCOMPLETE`, not `NO_GO`, because the intended policy path was not exercised.

## Canonical Classification

The existing arrival-load correctness and structural rules remain mandatory.

P5 is `NO_GO` for any:

- output, lifecycle, starvation, or request-set failure;
- state-transition or progress reconstruction mismatch;
- clock violation;
- missing progress;
- unsafe mixed admission;
- actual prefill count above the selected budget;
- mixed batch without both row types;
- actual mixed duration above its frozen predicted envelope;
- p99 TTFT, p99 ITL, p99 E2E, maximum decode gap, or service-bucket p95 E2E
  regression above 10%;
- worst-repetition violation;
- lack of a valid benefit path.

P5 may be `GO` only if all guards pass and at least one existing
preregistered benefit path holds for both median and worst-repetition
direction:

1. throughput path;
2. latency path;
3. memory path.

P5 does not receive a weaker threshold because it is SLO-aware.

Additional P5 promotion requirements:

- `mixed_service_fairness` has no bucket above `1.10x` p95 E2E;
- `long_prompt_pressure` p95 ITL is at most `1.05x` in every repetition;
- burst request throughput median is at least `1.25x`;
- at least one burst repetition selects three distinct chunk sizes;
- at least one non-burst scenario records SLO suppression;
- envelope-underprediction count is zero.

These requirements prevent a trivial decode-only policy from passing safety
while discarding the known burst opportunity.

`PROMISING_NOT_PROVEN` remains diagnostic only and cannot update README
performance claims.

## Documentation and Promotion Boundary

Before source-bound evidence:

- only this design and its implementation plan may describe expected
  behavior;
- no README performance claim is allowed.

After canonical:

- `GO`: update README and handoff with exact scope and evidence;
- `PROMISING_NOT_PROVEN`: update handoff only;
- `NO_GO`: update handoff only and keep P5 disabled;
- `INCOMPLETE`: update handoff with the missing evidence and do not claim a
  performance result.

Raw experiment artifacts remain untracked and are never staged with
`git add -A`.

## Success Criteria

The design is successful only if implementation produces a source-bound,
independently verified answer to both questions:

1. Can elapsed-time slack eliminate P4's decode-tail and service-bucket
   regressions?
2. Can it preserve enough mixed admission to retain a repeatable burst
   throughput or latency benefit?

Passing local tests alone is not success. Safe tails without a benefit path
are not success. Burst speedup with any guard failure is not success.

## Stop Conditions

Stop P5 scheduler work and do not tune the same workload further if:

- the frozen envelope suppresses nearly all mixed work in two independently
  calibrated source-bound runs;
- P5 passes safety but has no throughput, latency, or memory benefit path;
- P5 still violates long-prompt or mixed-service tail guards;
- useful admission requires weakening the 10% guards;
- coefficients or targets must be adjusted after reading canonical results.

After a P5 `NO_GO`, the next optimization investigation should move away from
scheduler-level mixed-prefill tuning and evaluate a structurally different
source of benefit, preferably:

1. kernel/CUDA Graph overhead reduction; or
2. quantization with exact quality and memory-capacity gates.
