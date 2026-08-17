# ModelRunner Atomic Checkpoint Candidate Binding Design

## Objective

Bind a loaded Qwen3.5 checkpoint candidate's exact model owner and retained
model-manifest fingerprint to ModelRunner as one single-assignment operation.

## Interface

`ModelRunner` adds:

```text
bind_qwen35_loaded_checkpoint_candidate(candidate)
```

The candidate must be an exact `Qwen35LoadedCheckpointCandidate`.

## Atomicity Contract

Before mutation, the method:

1. validates the candidate type;
2. validates and derives runtime identity from
   `candidate.owner` and `candidate.model_fingerprint`;
3. requires both owner and runtime identity slots to be empty.

It then calls the existing owner binder. That binder performs all model,
ownership-graph, bridge, restore-owner, and participant compatibility checks
before assigning owner state.

After owner binding returns, identity assignment cannot fail because the
identity was already fully constructed. The method stores the identity and
its exact owner, then returns the rank row.

## Idempotency

An exact repeat is allowed only when:

- bound owner is the same object as `candidate.owner`;
- bound identity owner is the same object;
- bound identity equals the candidate-derived identity.

Any different candidate or partial pre-existing owner/identity state fails
closed without mutation.

## Scope Boundary

This gate binds only the streaming candidate type used by the existing
one-shot owner publication slot. Tiled candidate promotion remains separate.
It does not add Engine orchestration or automatic runtime enablement.

## Tests

Tests cover first bind, exact repeat, wrong type, wrong current model,
invalid fingerprint, forged owner graph, partial pre-existing state, and
different candidate rejection. Failure cases assert no new owner, bridge, or
identity state.

## Claim Boundary

Passing proves atomic dependency-light ModelRunner owner+identity binding. It
does not prove real worker checkpoint loading or Engine all-rank candidate
dispatch.
