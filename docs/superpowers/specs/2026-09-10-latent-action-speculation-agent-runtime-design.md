# Latent Action Speculation for Agent Runtimes Design

**Date:** 2026-09-10
**Status:** Written design pending user review
**Source anchor:** `5b511da768f4fde1670aeb801c8c29f991322769`
**Target branch:** `feat/kv-sparse-attention`
**Scope:** action-level speculation contract and shared-serving profitability
**Default:** disabled
**Stage 0 result:** `PASS` (analytic only, no measured speedup claimed)

## 1. Objective

Qualify whether an agent runtime can speculate the *next action* using the
actor's latent state instead of a drafted text tool call, and establish under
what shared-serving conditions that speculation is net positive.

The line is deliberately split so that each claim is separately falsifiable:

1. a latent action drafter that emits a *set* of candidate action signatures
   with calibrated confidences in a single pass, rather than one sampled text
   tool call;
2. a drafter context built from compressed KV rather than the full text
   trajectory;
3. an explicit break-even model for a *shared* engine, where drafter GPU cost
   is not free even while the actor waits on a tool.

Stage 0 delivers only item 3 plus the contracts for items 1 and 2. It contains
no model and asserts no speedup.

## 2. Motivation

### 2.1 The bottleneck in an agent step is not decode

Published action-level speculation work reports that tool initialization
accounts for under twenty percent of overall tool latency; the dominant cost is
the strictly serial `think -> act -> observe` loop. Token-level speculative
decoding, including everything already qualified in
`tinyvllm/speculative/`, attacks the wrong term for this workload. The
addressable term is the tool wait, and the mechanism is overlapping it with the
actor's own thinking.

### 2.2 Every published drafter is a text drafter

The systems this design builds on draft with a smaller text model: the drafter
generates the full tool call as text and the actor's action is compared against
it. Reported single-branch next-action accuracy is about fifty-five percent,
which converts to roughly twenty percent latency reduction. Two costs follow.

First, the drafter pays to decode every argument token, so its cost scales with
argument length rather than with decision entropy. Second, one text sample
commits to one branch, so raising the effective match probability requires
sampling repeatedly, which multiplies drafter cost.

A drafter that reads the actor's latent state does not have this shape. It can
emit a ranked candidate set in one pass, because the decision is a
classification over tools and argument codes rather than a generation of
argument text. Widening from one branch to `b` branches then raises the match
probability at constant drafter GPU cost, paying only in extra speculative tool
invocations.

Separately, the discretisation literature is relevant here in the opposite
direction from latent reasoning. Continuous latent chains lose the
per-step reset that keeps errors from accumulating; work that re-discretises
latent thoughts with a learned codebook recovers it. An action drafter wants
exactly that property, because an action *is* a discrete object and because a
discrete code is auditable while a continuous vector is not.

### 2.3 The literature prices the drafter at zero

This is the gap that Stage 0 addresses and the reason this design starts with a
cost model rather than with a model.

Published results are reported on an otherwise idle actor: the drafter runs
while the actor waits for a tool, so its GPU cost is treated as free. On a
shared serving engine that assumption fails, because the tool wait is not GPU
idle time. The GPU is serving another request. Charging the drafter correctly
produces three consequences that the action-speculation literature does not
state:

1. `rho * (1 + tau) < 1` is a hard stability constraint, where `rho` is
   baseline GPU utilization and `tau` is the drafter's GPU tax relative to the
   actor. A drafter can push a stable engine into overload. That is a capacity
   failure, not a latency regression.
2. Peak capacity falls by `1 / (1 + tau)` whether or not proposals are
   accepted, because the drafter always runs.
3. There is a critical utilization above which speculation is net negative at a
   fixed, good match probability.

## 3. Non-goals

- No change to token-level speculative decoding semantics. The distributional
  losslessness of `tinyvllm/speculative/` is not weakened, extended, or
  reinterpreted.
- No claim that action-level commit-on-match is equivalent to token-level
  rejection sampling. It is not; see section 6.4.
- No latent reasoning inside the actor. The actor keeps emitting text actions.
  Only the *drafter* consumes latent state.
- No trained drafter in this design. Stage 1 introduces one behind its own gate.
- No interpretability claim. Reading a latent action code is not reading a
  reasoning trace, and no probe or lens output is treated as an explanation.
- No queueing-theory contribution. The M/M/1 term in the cost model is a
  declared modelling assumption, not a result.

## 4. Considered approaches

### 4.1 Scale the text drafter

Keep the published design and use a stronger small text model to raise the
fifty-five percent match rate. Rejected as the primary path because the Stage 0
model shows that a drafter whose GPU tax reaches one has no headroom at all:
scheduled first, it consumes `tau / (1 + tau)` of a step that is itself
`(1 + tau)` times longer, so it finishes exactly when the actor would have.
At `tau >= 1` the verdict is `infeasible_no_match_benefit` regardless of
accuracy. Accuracy is the wrong axis to buy first.

### 4.2 Continuous latent action drafter

Read the actor's residual stream and regress a continuous action embedding,
then nearest-neighbour it against a tool-argument index. Kept as a comparison
arm, not the candidate. Two objections. A continuous proposal is not auditable,
which is the property this line most needs to preserve. And nearest-neighbour
retrieval over a large argument space reintroduces a cost that scales with
argument cardinality.

### 4.3 Discrete-code action drafter over compressed KV

Candidate. The drafter consumes the actor's hidden state plus a compressed KV
view of the trajectory and emits a ranked set of `(tool, argument-code)` pairs
with calibrated confidences. Argument codes come from a learned codebook over
observed tool arguments, so the drafter classifies instead of generating.

Three properties motivate the choice. The emitted object is discrete, so it can
be logged, diffed, and reviewed. The candidate set is produced in one pass, so
branch widening is free on the GPU. And the drafter reads compressed KV rather
than the full trajectory, so its cost does not grow with trajectory length,
which is the term that grows fastest in long agent runs.

## 5. Existing runtime boundaries to preserve

### 5.1 Proposal interface

`tinyvllm/speculative/adapter.py` already admits a drafter that declares
`requires_target_hidden` and receives `target_hidden` in its context.
`tinyvllm/agentspec/latent_adapter.py` mirrors that shape at action
granularity on purpose, so the two speculation levels stay reviewable side by
side and so a future runtime can host both without a second abstraction.

### 5.2 Token authority

The target model remains the sole authority over tokens. Nothing in
`agentspec` may sample, accept, or rewrite a token.

### 5.3 Trajectory authority

The actor remains the sole authority over the committed action sequence. A
speculative observation may be reused only on an exact action-signature match.

### 5.4 Route naming

`tinyvllm/agentspec/router.py` follows `tinyvllm/speculative/router.py`: the
route set is fixed, every non-speculative route is named, and an unknown
condition fails closed rather than degrading into speculation.

## 6. Candidate architecture

### 6.1 Action identity

`ActionSignature` is `(tool_name, canonical_arguments_json)` with a SHA-256
digest. Canonicalisation pins key order, separators, and escaping so the match
test cannot be influenced by dictionary iteration order or formatting. Two
actions are identical only when their digests are equal.

### 6.2 Side-effect classification

Token-level speculation can discard a rejected token for free. An agent action
cannot. `ToolContract` therefore declares a `side_effect_class` in
`read_only`, `sandboxable`, `reversible`, `irreversible`, `unknown`, plus a
`rollback_seconds` cost. Only the first three are speculation eligible, and
`sandboxable` additionally requires an available sandbox. `unknown` fails
closed.

### 6.3 Latent action drafter contract

`LatentActionDraftCapabilities` declares the representation
(`text`, `continuous_latent`, `discrete_code`), whether target hidden state and
a compressed KV handle are required, the candidate-set width, and the action
horizon. A non-text representation must require hidden state; that is enforced,
not documented.

`validate_proposal` treats confidences as marginal probabilities over mutually
exclusive next actions: within `[0, 1]`, summing to at most one, unique by
digest, sorted by descending confidence so branch selection is deterministic. A
drafter that does not declare calibrated confidence must emit exactly one
candidate at confidence one, so an uncalibrated drafter cannot smuggle in a
fake distribution.

### 6.4 Commit-on-match and what it does and does not guarantee

The actor's action sequence is authoritative. A speculative observation is
reused only when the speculated signature is byte-identical to the actor's.
Otherwise the speculative work is discarded and the actor path executes.

This yields *trajectory equivalence*: the committed action sequence is
identical to what non-speculative execution would have produced. It does not
yield distributional equivalence, and it is strictly weaker than the
token-level guarantee. Any document, commit message, or report that describes
action-level commit-on-match as lossless in the token-level sense is wrong and
must be corrected.

The relevant practical consequence is that trajectory equivalence is the
property an operator actually needs for replay and post-hoc investigation, so
the weaker guarantee is not a weaker product claim in this specific respect.

### 6.5 Shared-engine cost model

`tinyvllm/agentspec/cost_model.py` models one agent step over two resources.
GPU is an M/M/1 queue with mean sojourn `D / (1 - rho)`; the tool is pure delay
with ample parallelism.

```text
W_base   = D / (1 - rho)
T_base   = W_base + T

rho_spec = rho * (1 + tau)
W_spec   = D * (1 + tau) / (1 - rho_spec)
t_draft  = alpha * W_spec
T_hit    = max(W_spec, t_draft + T)
T_miss   = W_spec + T + R
T_spec   = p * T_hit + (1 - p) * T_miss
```

`alpha` defaults to `tau / (1 + tau)`: the drafter is scheduled first, so its
share of step GPU work determines when the speculative tool call launches.
`alpha` is an explicit knob because it is the single most load-bearing
assumption in the model.

Closed forms the gate reports and tests pin:

- stability bound `tau_max = (1 - rho) / rho`;
- indifference probability `p_min = (T_miss - T_base) / (T_miss - T_hit)`,
  reported as `None` when `T_hit >= T_base` because then no accuracy can win;
- capacity ratio `1 / (1 + tau)`;
- wasted GPU fraction `(1 - p) * tau / (1 + tau)`;
- `critical_utilization` and `critical_draft_tax`, each located by grid scan
  plus bisection and each verified by the tests to be a real crossing.

### 6.6 Profitability router

Guard order is safety, then capacity, then profitability:
`baseline_no_proposal`, `baseline_side_effect_guard`,
`baseline_capacity_guard`, `baseline_unprofitable`,
`speculative_commit_on_match`. A net-negative point can be forced only through
an explicit `allow_unprofitable` flag, which is recorded in the route's
`fallback_reason`.

### 6.7 Optional macro horizon

`max_horizon_actions` exists in the capabilities because Stage 0 produced a
quantitative argument for it; see finding 4 in section 7. It is not exercised
before Stage 4.

## 7. Stage 0 findings

Reproduce with:

```bash
python3 tools/agentspec_breakeven_gate.py --print-summary
python3 -m pytest tools/test_agentspec_breakeven_gate.py -q
```

Frozen matrix: `actor_gpu_seconds = 0.080`, tool latency in
`{0.2, 1.0, 5.0}` s, baseline utilization in `{0.0, 0.3, 0.6, 0.8, 0.9}`,
drafter tax in `{0.10 discrete_code, 0.25 continuous_latent, 1.00
text_small_model}`, match probability in `{0.55, 0.75, 0.90}`, rollback in
`{0.0, 0.5}` s. 270 rows, 16 invariants, all `PASS`.

Verdict distribution over the 270 declared operating points:

| verdict | rows |
|---|---:|
| `unstable_capacity` | 90 |
| `net_positive` | 76 |
| `infeasible_no_match_benefit` | 66 |
| `net_negative` | 38 |

**Finding 1: only 28 percent of declared operating points are net positive.**
The published framing, where the drafter is free, corresponds to the
`rho = 0` slice. Off that slice, most of the space is not profitable.

**Finding 2: drafter cost dominates drafter accuracy.** At tool latency 1.0 s,
zero rollback, and `rho = 0.3`:

| drafter | tax | p=0.55 | p=0.75 | p=0.90 | critical rho at p=0.55 |
|---|---:|---:|---:|---:|---:|
| `discrete_code` | 0.10 | 1.046 | 1.070 | 1.088 | 0.818 |
| `continuous_latent` | 0.25 | 1.023 | 1.047 | 1.067 | 0.545 |
| `text_small_model` | 1.00 | infeasible | infeasible | infeasible | none |

Moving the drafter tax from 0.25 to 0.10 buys more critical utilization
headroom (0.545 to 0.818) than moving accuracy from 0.55 to 0.90 does
(0.545 to 0.722). This is the quantitative case for the cheap discrete-code
drafter over both a stronger text drafter and a continuous regression drafter.

**Finding 3: rollback cost, not accuracy, is the usual killer.** At tax 0.10
and tool latency 1.0 s with `R = 0.5` s, the indifference probability is 0.876
at `rho = 0`, 0.835 at `rho = 0.3`, 0.760 at `rho = 0.6`, and 0.714 at
`rho = 0.8`. The published fifty-five percent match rate is far below all of
them. Speculation on a tool with real rollback cost is not viable at published
accuracy; speculation must be restricted to read-only or genuinely sandboxed
tools, where `R` is approximately zero.

**Finding 4: single-step speculation cannot hide a long tool call.** The saving
on a hit is bounded by the actor's GPU sojourn, not by tool latency, because
overlapping one tool call can hide at most one think. At tax 0.10,
`rho = 0.3`, `p = 0.55`, the speedup is 1.183 at 0.2 s tool latency, 1.046 at
1.0 s, and 1.010 at 5.0 s. Relative speedup decays as the tool dominates. This
is the cost-model argument for a multi-action horizon rather than a single next
action, and it is asserted as a gate invariant.

**Finding 5: branch widening is the cheap axis for a latent drafter.** With a
fixed candidate set at confidences `(0.55, 0.18, 0.09, 0.05)`, tax 0.10,
`rho = 0.6`, tool latency 1.0 s, widening from one to four branches raises the
aggregate match probability from 0.55 to 0.87 and the speedup from 1.063 to
1.138 at *constant* drafter GPU tax, paying only in speculative tool calls.
This is the property a text drafter does not have.

## 8. Evidence and artifact contract

`tools/agentspec_breakeven_gate.py` writes
`agentspec_breakeven_report.json` containing `gate`, `schema_version`,
`status`, `claim_boundary`, `headline`, all 16 `invariants`, `branch_rows`,
`guard_rows`, the full 270-row `matrix_rows`, and `payload_sha256` over the
canonical serialisation. The report is deterministic: two builds in one process
and two CLI invocations produce the same digest, which the tests assert.

Stage 0 evidence is analytic and self-contained. It uses no GPU, no checkpoint,
no network, and no third-party package, and runs in under one second.

## 9. Failure and cleanup policy

Any invariant failure makes the gate exit non-zero with `status = FAIL`; there
is no partial pass. Stage 1 and later gates may not reuse a Stage 0 artifact
whose `payload_sha256` does not match a fresh local run of the same source
revision. The `alpha` knob and every frozen matrix constant must be changed by
editing the gate, never by an environment variable, so that a changed
assumption changes the digest.

## 10. Claim boundary

Stage 0 `PASS` means the contracts validate and the sixteen analytic invariants
hold over the declared matrix. It is not evidence of any measured speedup, and
it must not be cited as such. Every number in section 7 is a consequence of
declared inputs plus a declared queueing assumption. The only externally
sourced quantity is the fifty-five percent reference match probability, which
is taken from published action-level speculation results and used as a
reference point, not reproduced here.

## 11. Originality statement

Action-level speculative execution with commit-on-match is prior art, as is
compressed-KV context for a drafter, as is discrete codebook representation of
latent reasoning steps. What this design contributes is the combination and its
pricing:

1. an action drafter that reads latent state and emits a *confidence-ranked
   candidate set* in one pass, so branch width is decoupled from drafter cost;
2. a drafter context built from compressed KV rather than the text trajectory,
   so drafter cost is decoupled from trajectory length;
3. a shared-engine break-even model that charges the drafter for the capacity
   it consumes, yielding a stability bound, a critical utilization, and the
   four findings in section 7 that the zero-cost-drafter framing cannot
   express.

No claim is made that items 1 and 2 have never been proposed separately. The
falsifiable claim is that the pricing in item 3 changes which design one should
build, and section 7 finding 2 is the specific evidence for that.

## 12. Prompt-to-artifact checklist

- [x] Stage 0 contracts implemented under `tinyvllm/agentspec/`.
- [x] Stage 0 gate implemented at `tools/agentspec_breakeven_gate.py`.
- [x] Stage 0 tests at `tools/test_agentspec_breakeven_gate.py`, 48 passing.
- [x] Deterministic artifact digest asserted by tests.
- [x] Claim boundary stated in code, artifact, and this design.
- [ ] Stage 1 offline action-prediction quality gate.
- [ ] Stage 2 sandboxed trajectory-equivalence gate.
- [ ] Stage 3 shared-load profitability gate on real hardware.
- [ ] Stage 4 macro-horizon qualification.

## 13. Qualification sequence

### Stage 0: analytic contract gate

Status `PASS`. Deliverables as in section 12. Exit criterion for Stage 1 is a
fresh local `PASS` at the Stage 1 source revision.

### Stage 1: offline action-prediction quality gate

Train nothing in-repo yet. Collect agent traces, fit the argument codebook,
and measure top-`b` action-prediction accuracy and calibration offline. The
gate must report `p` per branch width and per tool class, plus the measured
drafter tax `tau` against the actor. `NO_GO` if measured `tau` at the target
accuracy exceeds the Stage 0 `critical_draft_tax` for the intended load point.

### Stage 2: sandboxed trajectory-equivalence gate

Prove that speculative execution produces a byte-identical committed action
sequence against a non-speculative control on the same traces, with all
speculative tool calls restricted to `read_only` and sandboxed tools. Any
divergence is `NO_GO_CORRECTNESS`.

### Stage 3: shared-load profitability gate

Measure on real hardware under real concurrency, and compare measured speedup
and capacity loss against the Stage 0 prediction at the same operating point.
The falsifiable prediction is the section 7 finding 2 ordering. If a measured
text drafter beats a measured discrete-code drafter, the cost model is wrong
and this design is retired.

### Stage 4: macro-horizon qualification

Only after Stage 3. Extend to `max_horizon_actions > 1` and test whether the
finding 4 decay is recovered.

## 14. References

Cited by name because Stage 0 imports no external artifact:

- action-level speculative execution with commit-on-match, and its reported
  fifty-five percent single-branch next-action accuracy converting to about
  twenty percent latency reduction;
- memory-augmented action speculation, reporting nineteen to thirty-nine
  percent relative gains in action-prediction accuracy from online transition,
  episodic, and confusion memories;
- self-speculative forking for agentic inference, which contributes the
  break-even cost-model framing this design extends to shared load;
- macro-commit style speculation of multi-action skeletons, which motivates
  section 6.7;
- compressed-KV and asymmetric-context drafting, which motivates section 4.3;
- discrete-codebook re-quantisation of latent reasoning steps, which motivates
  the choice of `discrete_code` over `continuous_latent` in section 4.
