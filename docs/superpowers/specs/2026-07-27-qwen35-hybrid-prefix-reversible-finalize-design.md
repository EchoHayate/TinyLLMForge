# Qwen3.5 Hybrid Prefix Reversible Finalize and Seal Design

## Objective

Make visible publication reversible until every rank has finalized.

The coordinator foundation can roll back all prepare/precommit failures, but a
rank may fail after an earlier rank finalizes. This gate adds a local
`finalize -> seal` protocol so the coordinator can undo earlier visible ranks
before declaring the ticket committed.

## Cache State

Extend the in-flight transaction:

```text
prepared
precommitted
finalized_unsealed
sealed / consumed
```

`finalize_publication()` makes the new entry visible but retains enough
ownership to restore the exact previous entry and LRU position.

## Replacement Journal

Before finalize, capture:

- entry key;
- previous snapshot if present;
- previous LRU index;
- new snapshot;
- counters/accounting deltas caused by finalize;
- entries evicted while enforcing limits, including snapshots and positions.

Because evicted entries may be needed for rollback, finalize does not release
their intern refs until seal. It removes them from visible accounting/index
but keeps journal ownership.

## Rollback Finalized

`rollback_publication()` from `finalized_unsealed`:

1. removes the new entry if still present;
2. restores previous and evicted entries at exact LRU positions;
3. restores visible bytes/logical bytes and event counters;
4. releases new snapshot refs;
5. consumes the transaction as rolled back.

## Seal

`seal_publication()`:

1. validates the exact finalized handle;
2. releases journal ownership for replaced/evicted snapshots;
3. consumes the transaction;
4. records one sealed publication.

Only seal makes replacement/eviction irreversible.

## Convenience Commit

Local `commit_publication()` becomes:

```text
precommit
finalize
seal
```

The distributed participant exposes `finalize`, `rollback`, and `seal`
separately.

## Tests

1. finalize new entry then rollback removes it;
2. replacement finalize then rollback restores old exact snapshot;
3. byte-limit evictions during finalize are restored in exact LRU order;
4. finalize then seal retains new entry and releases journal refs;
5. handle replay/foreign/wrong phase rejects;
6. coordinator rank-1 finalize failure rolls rank-0 back to invisible;
7. seal failure poisons after all entries are visible and must remain
   fail-stop.

## Claim Boundary

Passing proves dependency-light reversible cache visibility and coordinator
recovery from injected partial finalize. It still does not prove process-crash
recovery, durable distributed consensus, runtime publication, or production
performance.
