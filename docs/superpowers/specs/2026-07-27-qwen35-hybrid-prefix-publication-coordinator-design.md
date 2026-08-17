# Qwen3.5 Hybrid Prefix Publication Coordinator Design

## Objective

Coordinate exact hybrid-prefix snapshot publication across tensor-parallel
rank participants without exposing any rank during prepare or precommit.

All participants must prepare and precommit the same prefix identity before
finalization begins. Prepare/precommit rejection or failure rolls back every
rank and leaves all visible caches unchanged.

## Honest Atomicity Boundary

Precommit moves all expected hashing, equality, allocation, and interning
failures before visibility. Finalize is designed as a local non-allocating
entry/accounting mutation.

However, Python process death, transport loss, or an injected unexpected
exception can still interrupt sequential finalization after an earlier rank
became visible. The first coordinator therefore:

- guarantees all-rank rollback through prepare and precommit;
- validates every acknowledgement and exact payload identity;
- poisons permanently on any finalize inconsistency or failure;
- does not claim recoverable strong atomicity after partial finalize.

A later reversible-finalize/seal protocol is required before production
runtime wiring.

## Coordinator

Create:

```text
tinyvllm/engine/qwen35_hybrid_prefix_publication_coordinator.py
```

```python
class Qwen35HybridPrefixPublicationCoordinator:
    def __init__(participants)
    def publish(payloads) -> bool
```

Participants are sorted by unique participant ID. Payloads are sorted and
must provide exactly one `participant_id`-bound payload per participant.

Across ranks, payloads must match on:

```text
ticket_id
request_id
key
token_ids
block_identities
```

`participant_id` differs by rank and must cover contiguous IDs from zero.
Leases may differ by slot/generation but must have the same request ID.
`key.tensor_parallel_size` must equal participant count.

## Flow

1. validate coordinator health and payload matrix;
2. call prepare on every participant;
3. if any rank rejects, rollback all prepared ranks and return `False`;
4. if any rank errors or acknowledgement is invalid, rollback all prepared
   ranks and raise;
5. call precommit on every participant;
6. on any precommit failure, rollback every prepared/precommitted rank and
   raise;
7. call commit/finalize on every participant;
8. require every rank to report committed;
9. return `True`.

Rollback acknowledgement failure poisons the coordinator.
Any finalize failure poisons the coordinator because an earlier rank may
already be visible.

## Tests

1. two ranks publish exact same identity with rank-local state;
2. prepare rejection rolls back earlier rank and returns false;
3. prepare error rolls back earlier rank and raises;
4. precommit error rolls back every rank and leaves caches invisible;
5. malformed/duplicate/missing participant payloads reject before calls;
6. identity mismatch rejects before calls;
7. finalize failure poisons coordinator and blocks reuse;
8. successful publication is idempotent only at participant ticket level;
   coordinator ticket replay rejects as terminal participant acknowledgements.

## Claim Boundary

Passing proves all-rank identity validation and no visibility before every rank
precommits. It proves rollback for all prepare/precommit failures.

It does not prove recoverable atomicity after partial finalize, runtime
publication, cache hit rate, physical production memory reduction, or speed.
