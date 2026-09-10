# Latent Action Speculation Stage 0 Implementation Plan

**Design:** `docs/superpowers/specs/2026-09-10-latent-action-speculation-agent-runtime-design.md`
**Date:** 2026-09-10
**Status:** Stage 0 complete, Stage 1 not started
**Default:** disabled; nothing in this plan is wired into the engine

## Global Constraints

1. Stage 0 is analytic and contract-only. No task in this plan may import
   torch, transformers, or any other third-party package, load a checkpoint,
   touch a GPU, or reach the network.
2. Stage 0 must remain runnable on a laptop in under one second, because it is
   the entry gate every later stage re-runs.
3. No task may modify `tinyvllm/speculative/`. Token-level distributional
   losslessness is out of scope and must not be weakened by adjacency.
4. No task may register `agentspec` into the engine. `tinyvllm/__init__.py`
   stays untouched, and the gate loads submodules through stub parent packages
   so that production code keeps ordinary absolute imports.
5. Every profitability statement must be reported as a consequence of declared
   inputs. No task may phrase an analytic result as a measurement.
6. Safety guards must dominate profitability guards in the router, and this
   ordering must be asserted by a test rather than described in a comment.

## File map

### Runtime files

- `tinyvllm/agentspec/__init__.py` scope declaration and claim boundary.
- `tinyvllm/agentspec/action.py` action identity, canonical argument
  serialisation, side-effect classification, rollback cost.
- `tinyvllm/agentspec/latent_adapter.py` latent action drafter capabilities,
  context, candidate set, proposal validation, branch aggregation.
- `tinyvllm/agentspec/cost_model.py` shared-engine break-even model.
- `tinyvllm/agentspec/router.py` fail-closed route selection.

### Gate files

- `tools/agentspec_breakeven_gate.py` frozen matrix, invariants, deterministic
  JSON artifact, CLI.

### Tests

- `tools/test_agentspec_breakeven_gate.py` dependency-light tests over all
  four runtime modules plus the gate.

### Documentation

- `docs/superpowers/specs/2026-09-10-latent-action-speculation-agent-runtime-design.md`
- this plan
- `AGENT_HANDOFF_STATE.md` Stage 0 section

### Task 1: Freeze action identity and side-effect classification

Implement `ActionSignature` with a SHA-256 digest over
`(tool_name, canonical_arguments_json)`, and pin canonicalisation to sorted
keys, compact separators, and ASCII escaping so the match test cannot be
influenced by dictionary iteration order or formatting. Implement
`ToolContract` with `side_effect_class`, `rollback_seconds`, and
`sandbox_available`, and make `unknown` and `irreversible` ineligible.
`sandboxable` is eligible only with an available sandbox.

Status: done. Covered by ten tests including the digest field-separation case
and both sandbox directions.

### Task 2: Implement the shared-engine cost model

Implement the two-resource model in the design section 6.5 with validated
inputs, the `tau_max` stability bound, `p_min`, capacity ratio, wasted GPU
fraction, and the four verdicts. Locate `critical_utilization` and
`critical_draft_tax` by grid scan plus bisection rather than by assuming
monotonicity, and report `None` rather than guessing when the domain has no
crossing.

Status: done. Both critical values are asserted by tests to be real crossings,
and `p_min` is asserted to be the indifference point by re-evaluating the model
at that probability and checking latency equality to `1e-9`.

### Task 3: Implement the latent action drafter contract

Mirror `tinyvllm/speculative/adapter.py` at action granularity. Enforce that a
non-text representation requires target hidden state, that a declared
compressed-KV requirement is satisfied by the context, that confidences form a
valid marginal distribution, that candidates are unique by digest and sorted by
descending confidence, and that an uncalibrated drafter cannot emit a fake
distribution.

Status: done. Also implements `eligible_candidates`, which drops candidates
whose tool is ineligible or undeclared, and `aggregate_match_probability`, which
is the branch-widening axis.

### Task 4: Implement the fail-closed router

Fixed route set, named fallbacks, guard order safety then capacity then
profitability, and an explicit `allow_unprofitable` escape hatch that records
its reason.

Status: done. Guard precedence is asserted by constructing a point that is
simultaneously side-effect ineligible, capacity unstable, and profitable, and
checking that the side-effect route wins.

### Task 5: Build the Stage 0 gate

Freeze the 270-row matrix, evaluate every row through the model and the router,
check sixteen invariants, and write a deterministic JSON artifact with a
`payload_sha256` over the canonical serialisation. Load `agentspec` through
stub parent packages so the gate stays dependency-light.

Status: done. Gate returns `PASS` with all sixteen invariants passing.

### Task 6: Write dependency-light tests

Cover input validation, closed forms, both critical boundaries, contract
rejection paths, router precedence, gate determinism across two in-process
builds and one subprocess CLI run, and the artifact schema.

Status: done. 48 tests, 0.6 s.

### Task 7: Record the findings and their claim boundary

Write the design document with the five Stage 0 findings, state the claim
boundary in the package docstring, the gate artifact, and the design, and make
the Stage 3 falsifiable prediction explicit so the line can be retired.

Status: done.

### Task 8: Reconcile handoff, commit, and push

Add the Stage 0 section to `AGENT_HANDOFF_STATE.md`, commit the runtime, gate,
tests, and documentation together, and push.

Status: done.

## Plan self-review record

Two invariants failed on first run and both were resolved by correcting the
plan's assumption rather than the model.

1. `free_drafter_rows_are_profitable` failed at `tau = 1`. Inspection showed
   the model is right: at zero utilization and zero rollback, a hit beats
   baseline if and only if `tau < 1`, because a drafter scheduled first
   consumes `tau / (1 + tau)` of a step that is `(1 + tau)` times longer. The
   invariant was split into `idle_capacity_rows_profitable_below_unit_tax` and
   `unit_tax_drafter_has_no_headroom`, and the latter is now a design finding.
2. A test asserted that longer tool latency raises speedup. It does not. The
   hit saving is bounded by the actor GPU sojourn, so relative speedup decays
   once the tool dominates. The test was replaced by
   `test_hit_saving_is_capped_by_actor_sojourn` and
   `test_relative_speedup_decays_when_tool_dominates`, two gate invariants were
   added, and the result became the argument for a macro horizon.

Both corrections are recorded because they are the two places where the plan
was wrong and the model was right, which is the evidence that the invariants
are load-bearing rather than decorative.

## Completion criteria

- [x] `python3 tools/agentspec_breakeven_gate.py --print-summary` prints
      `status PASS` and exits zero.
- [x] `python3 -m pytest tools/test_agentspec_breakeven_gate.py -q` reports
      48 passed.
- [x] `python3 -m py_compile` clean for all five runtime files, the gate, and
      the tests.
- [x] No third-party import anywhere in the Stage 0 surface.
- [x] `tinyvllm/speculative/` unmodified.
- [x] Design, plan, and handoff state committed together with the code.

## Stage 1 entry criteria

Stage 1 may not begin until a fresh local Stage 0 run at the Stage 1 source
revision returns `PASS`, and until the intended load point is declared in
advance as `(rho, tool_seconds, rollback_seconds)`. Stage 1 is `NO_GO` if the
measured drafter tax at the target accuracy exceeds the Stage 0
`critical_draft_tax` for that declared point.
