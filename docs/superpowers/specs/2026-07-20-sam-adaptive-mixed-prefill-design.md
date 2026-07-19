# SAM Backlog-Adaptive Mixed-Prefill Design

Date: 2026-07-20

## Terminology

In this document, **SAM** means **Scheduler-Adaptive Mixed-prefill**.

It does not mean either of the other SAM terms already used in this
repository:

- the suffix-automaton speculative draft source;
- Source-Auditable Manifest.

The policy name used by the arrival-load gate is `P4`. Its stable descriptive
name is `sam_backlog_adaptive_mixed_prefill`.

## Objective

Add a disabled-by-default scheduler controller that uses the existing mixed
prefill-plus-decode implementation only while a sustained waiting backlog
justifies its latency cost.

The first phase must:

1. Preserve repository-default decode-first scheduling while the controller
   is disabled or inactive.
2. Enter mixed mode only after `waiting >= 8` is observed on two consecutive
   eligible scheduling decisions.
3. Stop admitting new mixed prefill work after `waiting <= 2` is observed on
   two consecutive eligible scheduling decisions.
4. Drain any prefill request already admitted before returning to inactive
   decode-first scheduling.
5. Force one decode-only step after at most two consecutive mixed steps when
   runnable decode work remains.
6. Never choose a prefill-only batch while runnable decode work exists under
   the adaptive policy.
7. Preserve exact greedy output, request lifecycle, queue ownership, block
   allocation, prefix-cache behavior, and mixed-batch model-runner semantics.
8. Expose enough immutable state and transition evidence for the independent
   arrival-load verifier to prove which policy branch ran.
9. Compare `P4` against both repository default `P0` and always-on mixed
   policy `P3` under the frozen arrival-load gate.
10. Produce a performance claim only if the source-bound independent
    canonical verifier returns `GO`.

The first phase does not tune thresholds online, predict kernel duration,
change the mixed forward implementation, add request priorities, or implement
a general serving admission controller.

## Evidence and Problem Statement

The independently verified canonical artifact
`experiments/arrival_load/qwen3-06b-arrival-final-calibration-20260719-2317`
classified both existing non-default candidates as `NO_GO`.

The verifier found no structural or correctness failures, but always-on mixed
policy `P3` had these global median ratios relative to `P0`:

- request throughput: `0.99999`;
- p95 TTFT: `0.62014`;
- p95 ITL: `4.21280`;
- p99 ITL: `1.07546`;
- maximum decode gap: `1.16455`;
- peak KV bytes: `3.5`.

`P3` therefore improved admission latency in some regions but paid an
unacceptable decode-latency and KV-residency cost when mixed mode was used
without demand gating.

The burst scenario shows a distinct useful region. Across the three measured
repetitions:

- `P0` request throughput was approximately `1.66`, `1.66`, and `1.78`
  requests/s;
- `P3` request throughput was approximately `4.35`, `4.14`, and `4.25`
  requests/s;
- `P3` substantially reduced burst TTFT and E2E latency;
- `P3` kept maximum decode gap near the `P0` result in two repetitions, while
  the global workload still failed tail guards.

The scheduler trace also separates the burst region by backlog. Under `P3`:

- burst waiting depth had p50 `16` and p95 `67`;
- steady-moderate, near-saturation, overload, long-prompt-pressure, and
  mixed-service-fairness each had waiting-depth p50 `0` and p95 `0`.

This supports a conservative demand gate around the existing mixed path. It
does not prove that thresholds `8` and `2` are optimal. They are
preregistered first-candidate values that must survive the same full canonical
gate as every other scheduler policy.

The artifact above executed source tree
`153b12d9c827157229950264b078c873300b462cab43e4f3bd1b02017725a5c3`.
It motivates this design but cannot authorize a newer source. Commit
`7e848c997d1c5c2c1ddff9c853a5d4dba2e59a56` must still complete its own
source-bound remote chain after Kerberos authentication is restored.

## Alternatives Considered

### 1. Recommended: Backlog Hysteresis Plus Decode-Service Bound

Use waiting depth as a demand signal, require consecutive observations before
state changes, stop new admission at a lower exit threshold, and force a
decode-only step after two mixed steps.

Advantages:

- directly targets the trace region where `P3` was useful;
- remains local to scheduler policy selection;
- uses only state already owned by the scheduler;
- avoids wall-clock feedback and nondeterministic online tuning;
- bounds exposure to the long mixed-forward latency that caused the ITL
  regression;
- preserves the existing mixed data path instead of creating another one.

Risks:

- waiting depth is a coarse demand signal and does not encode request size;
- two fixed thresholds may not generalize to other models or GPU types;
- forced decode-only steps may surrender part of the burst throughput gain;
- the policy may still increase KV residency while active.

### 2. Always-On Mixed Mode With a Smaller Prefill Chunk

Keep `P3` active for every eligible step but reduce
`max_num_prefill_tokens_per_step`.

This is rejected for the first phase. It changes the amount of target work in
every mixed forward, does not isolate burst demand, and adds a second tuning
dimension before the existing `128`-token gate is understood.

### 3. Latency-Feedback or Learned Controller

Enable mixed mode from an online EMA of step latency, TTFT, ITL, or a learned
profitability model.

This is deferred. Runtime latency feedback would depend on synchronization,
measurement placement, and hardware-specific timing. A learned policy would
need a separate training and out-of-sample evidence contract. Neither is
justified before a deterministic backlog controller is measured.

## Decision

Implement Alternative 1 after this written spec is reviewed and a separate
implementation plan is approved.

The implementation must reuse the existing mixed scheduler, model-runner, and
postprocess paths. It must not fork or duplicate mixed attention preparation.

## Configuration Contract

Add these scheduler-affecting configuration fields:

```python
chunked_prefill_adaptive_mixed: bool = False
chunked_prefill_adaptive_enter_waiting: int = 8
chunked_prefill_adaptive_exit_waiting: int = 2
chunked_prefill_adaptive_transition_steps: int = 2
chunked_prefill_adaptive_max_mixed_steps: int = 2
```

Validation is fail closed:

- all integer values must be positive except the exit threshold, which may be
  zero;
- `exit_waiting < enter_waiting`;
- adaptive mode requires `max_num_prefill_tokens_per_step > 0`;
- adaptive mode and legacy `chunked_prefill_mixed_batch=True` are mutually
  exclusive;
- adaptive mode is incompatible with `kv_offload_mvp0`, matching the current
  mixed-batch restriction;
- invalid combinations fail during configuration initialization, before model
  startup or scheduler mutation.

All five fields are included in policy identity, source-bound manifests, and
independent-verifier configuration checks.

Default values preserve all current behavior because
`chunked_prefill_adaptive_mixed=False`.

## Eligibility

A scheduling decision is eligible for adaptive transition accounting only
when all of the following are true:

- adaptive mode is enabled;
- chunked prefill is enabled;
- at least one sequence is runnable in `running`;
- at least one prefill candidate exists in `waiting` or `prefilling`.

Ineligible decisions clear both transition streaks. They do not carry stale
activation evidence into a later unrelated workload.

The controller samples queue depth exactly once at the beginning of
`Scheduler.schedule()`:

```python
waiting_depth = len(self.waiting)
```

It must not derive the signal after allocating, preempting, or reordering a
sequence.

Only `waiting` contributes to the threshold. `prefilling` is tracked
separately so that a request already admitted can be drained without
pretending it is new backlog.

## State Machine

The controller has three explicit states:

- `INACTIVE`: repository-default decode-first behavior;
- `ACTIVE`: may admit and schedule mixed prefill-plus-decode work;
- `DRAINING`: admits no new waiting request but may complete a request already
  in `prefilling`.

The scheduler owns:

```python
adaptive_mixed_state
adaptive_high_streak
adaptive_low_streak
adaptive_consecutive_mixed_steps
```

### INACTIVE

On each eligible decision:

- if `waiting_depth >= 8`, increment `adaptive_high_streak`;
- otherwise reset `adaptive_high_streak` to zero;
- keep `adaptive_low_streak` and
  `adaptive_consecutive_mixed_steps` at zero.

After two consecutive high observations, transition to `ACTIVE` before
selecting the current step. This allows the second confirming decision to use
mixed mode.

While the high streak is below two, run the existing decode-first path.

### ACTIVE

On each eligible decision:

- if `waiting_depth <= 2`, increment `adaptive_low_streak`;
- otherwise reset `adaptive_low_streak` to zero.

After two consecutive low observations:

- transition directly to `INACTIVE` if `prefilling` is empty;
- otherwise transition to `DRAINING`.

The transition is evaluated before new prefill admission, so the confirming
low observation cannot pull another request from `waiting`.

### DRAINING

`DRAINING` never admits a new sequence from `waiting`.

If `prefilling` is non-empty and decode work is runnable, the existing prefill
sequence may progress through a mixed batch subject to the decode-service
bound. Once `prefilling` is empty, transition to `INACTIVE` before selecting
the next step.

If `waiting_depth >= 8` for two consecutive eligible decisions while
draining, transition back to `ACTIVE`. This avoids completing a disable cycle
while a new sustained burst has already arrived.

### Empty and Terminal Queues

When `waiting`, `prefilling`, and `running` are all empty, reset the controller
to its initial `INACTIVE` state with all counters zero.

Preemption does not directly change controller state. Its resulting queue
depth is observed only at the next scheduling boundary.

## Scheduling Contract

### Inactive Behavior

`INACTIVE` must take the exact existing decode-first branch whenever
`running` is non-empty. A disabled or inactive adaptive controller must not
change:

- selected sequence IDs;
- queue order;
- block allocation;
- `last_policy_branch`;
- batch kind;
- generated tokens.

### Active Mixed Behavior

An adaptive mixed batch is valid only when it contains:

- at least one prefill sequence; and
- at least one decode sequence.

The policy must never report an adaptive mixed branch for a prefill-only
batch.

Before mutating queue or block state, the scheduler must prove that at least
one runnable decode sequence can be included under sequence, token, and KV
capacity limits. If it cannot, the step fails closed to the existing
decode-only path.

The existing `_schedule_mixed_prefill_decode()` implementation may be
refactored to provide this transactional guarantee, but its successful batch
composition and postprocess semantics must remain shared by `P3` and `P4`.

`ACTIVE` may admit a new sequence from `waiting`. `DRAINING` may pass only the
existing `prefilling` sequence to the mixed scheduler.

### Decode-Service Bound

Every successful adaptive mixed batch increments
`adaptive_consecutive_mixed_steps`.

If runnable decode work remains and the counter has reached two, the next
step is decode-only:

- do not admit or progress prefill in that step;
- reset `adaptive_consecutive_mixed_steps` to zero;
- preserve `ACTIVE` or `DRAINING` state;
- report branch `adaptive_mixed_decode_yield`.

A normal decode-only fallback also resets the counter. A prefill-only batch is
not permitted while adaptive mode has runnable decode work.

This is a scheduling-step bound, not a wall-clock guarantee. The synchronous
scheduler cannot know the future mixed-forward duration without introducing
timing feedback. The canonical p99 ITL and maximum-decode-gap guards remain
the authoritative wall-clock safety checks.

### No Running Decode Work

When `running` is empty, use the existing chunked-prefill path. Adaptive mixed
state must not invent an empty decode row or delay progress solely to preserve
a mixed label.

## Observation and Evidence Contract

`Scheduler.observation_snapshot()` must add immutable scalar fields:

```python
{
    "adaptive_mixed_state": "inactive|active|draining",
    "adaptive_high_streak": 0,
    "adaptive_low_streak": 0,
    "adaptive_consecutive_mixed_steps": 0,
}
```

Add policy branches:

- `adaptive_mixed_decode_first`;
- `adaptive_mixed_prefill_decode`;
- `adaptive_mixed_decode_yield`;
- `adaptive_mixed_decode_fallback`;
- `adaptive_mixed_chunked_prefill`.

Each scheduler trace row already records queue snapshots, selected sequence
roles, batch kind, and policy branch. The arrival-load driver must preserve
the new controller fields in both `queue_before` and `queue_after`.

The independent verifier must reject:

- an illegal state name;
- a negative or over-limit counter;
- activation without two preceding high observations;
- new waiting admission during `DRAINING`;
- more than two consecutive adaptive mixed branches while decode work was
  runnable;
- an adaptive mixed branch without both prefill and decode roles;
- a policy identity or resolved-config mismatch;
- missing controller fields for `P4`.

These are structural failures and classify the run as `INCOMPLETE`, not as a
performance loss.

## Policy Matrix

The adaptive canonical gate runs:

- `P0`: repository-default decode-first control;
- `P3`: existing always-on mixed diagnostic;
- `P4`: SAM backlog-adaptive mixed candidate.

`P4` resolves:

```python
{
    "chunked_prefill_decode_first": True,
    "chunked_prefill_max_consecutive_chunks": 0,
    "chunked_prefill_mixed_batch": False,
    "chunked_prefill_mixed_min_prompt_tokens": 0,
    "chunked_prefill_adaptive_mixed": True,
    "chunked_prefill_adaptive_enter_waiting": 8,
    "chunked_prefill_adaptive_exit_waiting": 2,
    "chunked_prefill_adaptive_transition_steps": 2,
    "chunked_prefill_adaptive_max_mixed_steps": 2,
}
```

The common engine configuration, prompt bank, arrivals, seeds, repetition
count, calibration procedure, output contract, and performance thresholds
remain those in the production arrival-load design.

`P3` is diagnostic and cannot make `P4` pass. `P4` must independently beat
`P0` through a preregistered benefit path while satisfying every global,
worst-repetition, service-bucket, correctness, source, and structural guard.

No threshold may be changed after any `P4` model result is visible. A threshold
change creates a new policy identity, run tag, source snapshot, and full
preflight-to-canonical chain.

## Correctness and Safety Invariants

The implementation must preserve:

- exact token equality with `P0` under greedy fixed-length decoding;
- one queue owner per live sequence;
- no duplicate or missing sequence IDs across queue snapshots;
- no request drop, truncation, starvation, or unfinished lifecycle;
- no extra model forward caused only by state observation;
- no CUDA synchronization added by the controller;
- no block allocation or `may_append()` call before the selected branch is
  committed;
- no hidden reset of prefix-cache metadata;
- no change to policy defaults when adaptive mode is disabled;
- no mixed batch under `kv_offload_mvp0`.

Any correctness mismatch is `NO_GO`. Missing, contradictory, truncated, or
unverifiable evidence is `INCOMPLETE`.

## Test Strategy

### Dependency-Light Scheduler Tests

Extend `tools/test_chunked_prefill.py` to prove:

1. disabled adaptive mode is byte-for-byte equivalent in returned schedule
   metadata and queue state to the current default;
2. one high observation does not activate;
3. the second consecutive `waiting >= 8` observation activates and may mix;
4. an intervening low observation resets the high streak;
5. one low observation does not deactivate;
6. the second consecutive `waiting <= 2` observation stops new admission;
7. an already-prefilling sequence enters `DRAINING` and completes without
   admitting another waiting request;
8. two mixed steps force one decode-only yield;
9. a mixed attempt that cannot include decode falls back without prefill
   allocation or queue mutation;
10. empty queues reset all state;
11. preemption affects only the next observation;
12. observation snapshots and branch names match the state transition;
13. invalid configuration combinations fail before scheduling;
14. `P3` successful mixed behavior remains unchanged after any shared-helper
    refactor.

### Arrival-Load Gate and Verifier Tests

Extend the gate, driver, shell-runner, and independent-verifier tests to
prove:

- `P4` policy identity includes all adaptive fields;
- frozen manifests preserve thresholds and transition counts;
- scheduler traces reconstruct every state transition independently;
- tampered state, counters, branch roles, or resolved configuration fail
  closed;
- warmup lifecycle is validated but warmup requests remain excluded from
  performance aggregation;
- resume cannot reuse a repetition from another adaptive policy identity;
- source changes require a new source tree and run tag.

### Remote Execution Order

After local tests pass and Kerberos is valid, execute only on
`sitian@10.232.195.203`:

1. source-bound preflight;
2. strengthened fixed-length smoke;
3. calibration;
4. full three-repetition canonical matrix;
5. local independent verification from downloaded raw artifacts.

Every model process uses a unique dynamic `TINYVLLM_DIST_PORT` and
`MASTER_PORT`. The runner must use the run-local immutable source snapshot and
must not modify a remote checkout, kill unrelated processes, or clear shared
temporary directories.

## Promotion Criteria

`P4` is promoted only if all of the following hold:

- independent verifier classification is `GO`;
- no structural or correctness failure exists;
- all non-duplicate cases have three complete measured repetitions;
- a preregistered benefit path passes against `P0`;
- p99 TTFT, p99 ITL, p99 E2E, maximum decode gap, and every service-bucket p95
  E2E guard pass;
- worst-repetition guards pass;
- output and lifecycle evidence are exact;
- the source tree, workload, model runner, environment, ports, and policy
  identities are verified;
- the report shows `P4` activates materially in burst demand and remains
  mostly inactive outside the intended backlog region.

If `P4` improves burst throughput but fails any global guard, classify it
`NO_GO`; do not call it an engine-wide speedup.

If `P4` passes correctness but does not reach a benefit path, classify it
`NO_GO`.

If evidence cannot prove the state machine or source boundary, classify it
`INCOMPLETE`.

## Non-Goals

This phase does not:

- claim network-serving latency;
- add async request admission;
- use request priority or deadline scheduling;
- tune thresholds per prompt class;
- use GPU utilization, free-memory EMA, or observed latency as controller
  inputs;
- change the `128`-token prefill chunk;
- change CUDA graph policy;
- optimize or replace the mixed varlen kernel;
- combine adaptive mixed scheduling with KV offload;
- update README or handoff claims before verified remote evidence exists.

## Implementation Boundary

Expected implementation files are:

- `tinyvllm/config.py`;
- `tinyvllm/engine/scheduler.py`;
- `tools/arrival_load_gate.py`;
- `tools/arrival_load_verify.py`;
- `tools/test_chunked_prefill.py`;
- `tools/test_arrival_load_gate.py`;
- `tools/test_arrival_load_driver.py`;
- `tools/test_arrival_load_verify.py`;
- `tools/run_arrival_load_gate_remote.sh`;
- `tools/test_run_arrival_load_gate_remote.py`.

`tinyvllm/engine/model_runner.py` and mixed postprocessing should not require
new behavior. They may receive only compatibility-preserving changes if the
transactional mixed-admission guarantee cannot otherwise be expressed.

README and `AGENT_HANDOFF_STATE.md` remain unchanged until a source-bound
remote result exists.

## Rollback

Rollback is configuration-only: leave
`chunked_prefill_adaptive_mixed=False`.

Because the feature is disabled by default and does not replace the existing
`P0` or `P3` path, a failed remote gate requires no model-format, cache-format,
or artifact migration.
