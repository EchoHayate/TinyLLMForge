# Qwen3.5 Hybrid Prefix Publication Participant Design

## Objective

Wrap cache-local staged publication in a rank-local, ticket-bound participant
contract suitable for a later all-rank coordinator.

This gate does not claim all-rank atomic commit. It proves that each rank can
prepare, commit, or rollback the exact same publication payload
deterministically and idempotently without exposing state during prepare.

## Payload

Create `tinyvllm/engine/qwen35_hybrid_prefix_publication_ticket.py`.

```python
@dataclass(frozen=True)
class Qwen35HybridPrefixPublicationPayload:
    ticket_id: int
    participant_id: int
    request_id: int
    key: Qwen35HybridPrefixKey
    token_ids: tuple[int, ...]
    block_identities: tuple[tuple[int, int, int], ...]
    lease: HybridStateLease
```

The payload validates exact cache identity, binds one participant rank, and
requires the lease request ID to match `request_id`.

## Participant

```python
class Qwen35HybridPrefixPublicationParticipant:
    def prepare(payload) -> ack
    def commit(payload) -> ack
    def rollback(payload) -> ack
```

Statuses:

```text
prepare: prepared | rejected | error
commit: committed | error
rollback: rolled_back | error
```

Prepare calls `cache.prepare_publication()`. Oversize returns `rejected`.
Exceptions return `error` and leave no ticket state unless cache cleanup fails.

The participant binds one exact payload to each ticket ID. Repeating the same
operation with the same payload is idempotent. Reusing a ticket ID with a
different payload returns `error`.

Commit is accepted only for the exact prepared payload. Rollback is accepted
for the exact prepared payload and is idempotent after successful rollback.
Committed tickets reject rollback; rolled-back tickets reject commit.

## Failure Boundary

If cache commit fails, the participant retains the prepared ticket and returns
`error`, allowing the coordinator to issue rollback. If rollback itself fails,
the ticket remains prepared and reports the cleanup error.

## Tests

Cover:

1. prepare is invisible and records an exact ticket;
2. repeated prepare is idempotent;
3. changed payload on the same ticket rejects;
4. oversize prepare rejects without state;
5. commit publishes exact staged values and is idempotent;
6. rollback preserves previous visible entry and is idempotent;
7. commit failure retains prepared state for rollback;
8. terminal-state cross-operation attempts reject;
9. participant/cache/pool coherence validation.

## Claim Boundary

Passing proves rank-local publication ticket semantics only. It does not prove
all-rank coordination, production publication, cache hit rate, memory savings,
or speed.
