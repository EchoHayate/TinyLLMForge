# Qwen3.5 Checkpoint Candidate Provenance Design

## Objective

Retain the verified model-manifest SHA256 from checkpoint loading through the
immutable loaded candidate and one-shot owner publication boundary.

## Loader Contract

Streaming, tiled, and policy-tiled loaders require:

```text
model_fingerprint=<lowercase 64-character SHA256>
```

The value is validated before opening any checkpoint shard.

## Candidate Contract

`Qwen35LoadedCheckpointCandidate` and
`Qwen35TiledLoadedCheckpointCandidate` add:

```text
model_fingerprint: str
```

The policy-tiled wrapper inherits provenance through its `loaded` candidate.
Candidates remain frozen dataclasses.

## Publication Contract

`Qwen35HybridModelOwnerPublicationSlot` stores both:

```text
owner
model_fingerprint
```

Both are `None` before publication. A successful one-shot publication stores
the exact candidate owner and fingerprint atomically. Invalid or incoherent
candidates and later replacement attempts leave both original values
unchanged.

## Scope Boundary

This gate does not modify the frozen model owner graph or automatically bind
the fingerprint into ModelRunner. The next gate may consume the publication
slot fingerprint when binding the owner.

## Tests

Tests cover required/invalid fingerprint rejection before shard open,
streaming/tiled/policy candidate retention, pickle-safe candidate provenance,
and publication-slot atomic preservation.

## Claim Boundary

Passing proves provenance retention from verified loader input through
publication slot. It does not prove the real checkpoint worker supplies the
authorized manifest SHA256 or that ModelRunner consumes it automatically.
