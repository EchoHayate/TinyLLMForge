# ModelRunner Hybrid Prefix Publication Methods Design

## Objective

Expose the reversible rank-local publication participant through acknowledged
ModelRunner command methods without enabling automatic runtime publication.

## Owner Graph

`Qwen35HybridPrefixRestoreOwner` remains the shared prefix-cache owner and gains:

```python
publication_participant:
    Qwen35HybridPrefixPublicationParticipant
```

Restore and publication participants share the exact same pool, transaction,
and snapshot cache.

## ModelRunner State

Add:

```text
qwen35_hybrid_prefix_publication_participant
```

Configuration installs both restore and publication participants atomically
from the same owner.

## Methods

```text
prepare_hybrid_prefix_publication
precommit_hybrid_prefix_publication
finalize_hybrid_prefix_publication
seal_hybrid_prefix_publication
rollback_hybrid_prefix_publication
```

Each method:

1. requires an installed publication participant;
2. delegates exactly once;
3. validates ticket ID, participant ID, operation, status, and detail;
4. returns a pickle-safe exact dict:

```text
ticket_id
participant_id
operation
status
detail
```

Allowed statuses:

```text
prepare: prepared | rejected | error
precommit: precommitted | error
finalize: finalized | error
seal: committed | error
rollback: rolled_back | error
```

## Scope Boundary

Do not add Engine transport or call these methods from `run()`, `step()`, or
Scheduler postprocess in this phase.

## Tests

- owner graph shares one cache across both participants;
- install validates type, rank, and pool;
- configuration installs both participants idempotently;
- all five methods delegate and validate exact result schemas;
- malformed acknowledgements fail closed;
- uninstalled methods fail closed.

## Claim Boundary

Passing proves rank-local ModelRunner method availability only. It does not
prove multi-process transport, runtime publication, cache hits, or performance.
