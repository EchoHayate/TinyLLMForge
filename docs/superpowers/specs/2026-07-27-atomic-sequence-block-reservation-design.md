# Atomic Sequence Block Reservation Design

## Objective

Add the KV-side transaction required before hybrid prefix reuse can enter the
scheduler.

The previous phase can reserve an exact reusable prefix and restore its
aligned Qwen3.5 linear state. A real request also needs KV blocks for its
uncached suffix. The current `BlockManager.allocate()` mutates
`Sequence.block_table` while it walks the request, so it cannot be composed
with a later state-restore failure without exposing or unwinding partial
request metadata.

This phase creates one private reservation for:

```text
exact reusable prefix references
+ newly allocated uncached suffix blocks
= complete future request block table
```

The reservation can be attached to one request in a final non-throwing commit
or released completely on any earlier failure.

This is still a dependency-light CPU gate. It does not enable hybrid prefix
reuse in `Scheduler`, move state tensors into the scheduler process, or modify
ModelRunner.

## Runtime Ownership Boundary

The current runtime has a deliberate split:

- scheduler process: `BlockManager`, request queues, and
  `HybridStateSlotAllocator`;
- ModelRunner process: `HybridStateTensorPool`, layer adapters, and actual
  state tensors.

The snapshot restore therefore cannot be made a scheduler-local operation in
the real engine. A later integration must use an explicit Engine↔ModelRunner
prepare/commit/rollback ticket.

This phase only provides the complete KV reservation needed by that ticket.
It does not introduce a second state pool or cache in the scheduler.

## Alternatives

### 1. Call Existing `allocate()` and Undo on Failure

Rejected. It mutates `Sequence.block_table`, cached/computed counters, free
lists, used sets, and refcounts incrementally. Rollback would depend on
partially visible request state.

### 2. Reserve Prefix, Attach It, Then Allocate Suffix

Rejected. Suffix exhaustion or state failure would expose a prefix-only block
table.

### 3. Reserve Complete KV Ownership Privately, Then Attach

Selected. Prefix and suffix ownership are acquired before any request metadata
changes. The final commit only copies prevalidated values.

### 4. Move Hybrid State Pool into Scheduler

Rejected. ModelRunner executes the layers and owns the tensors. Duplicating
that ownership would make multi-process and GPU lifecycle semantics incorrect.

## Data Model

Extend `tinyvllm/engine/block_manager.py`:

```python
@dataclass
class SequenceBlockReservation:
    block_ids: tuple[int, ...]
    block_identities: tuple[tuple[int, int, int], ...]
    cached_tokens: int
    prefix_block_count: int
    new_block_count: int
    state: str = "reserved"
```

Every `block_id` already holds exactly one future request reference:

- reusable prefix blocks gained one refcount;
- idle prefix blocks were generation-preserving activated;
- suffix blocks were allocated as new content with `ref_count == 1`.

`block_identities` covers only the reusable prefix because newly allocated
suffix blocks have no valid hash until their KV is materialized.

Add:

```python
def reserve_sequence_blocks(
    self,
    seq: Sequence,
    *,
    max_cached_tokens: Optional[int] = None,
) -> SequenceBlockReservation

def attach_sequence_reservation(
    self,
    reservation: SequenceBlockReservation,
    seq: Sequence,
) -> None

def release_sequence_reservation(
    self,
    reservation: SequenceBlockReservation,
) -> None
```

## Reservation Algorithm

`reserve_sequence_blocks()` requires a pristine sequence:

```text
block_table == []
num_cached_tokens == 0
num_computed_tokens == 0
```

It applies the same cache cap as existing `allocate()`:

- `max_cached_tokens` defaults to `len(seq)`;
- clamp to `[0, len(seq)]`;
- only full blocks below the cap are reusable;
- stop prefix reuse at the first miss;
- every remaining request block is a new-content allocation.

The transaction order is:

1. validate the request and cache cap;
2. compute the exact reusable prefix chain without mutation;
3. ensure enough free blocks exist for all non-live-prefix request blocks;
4. reserve one reference for every reusable prefix block;
5. allocate every suffix block privately;
6. build the complete ordered block table;
7. record exact reusable prefix identities;
8. return the reservation.

Any exception after ownership starts releases newly allocated suffix blocks
and every prefix reference, then re-raises.

An ordinary zero-prefix request is valid and returns a reservation containing
only new blocks.

## Attachment

`attach_sequence_reservation()` validates before mutation:

- reservation state is `reserved`;
- sequence is pristine;
- block count equals `seq.num_blocks`;
- cached tokens equal `prefix_block_count * block_size`;
- cached tokens do not exceed `max_reusable_tokens(seq)`;
- every reserved block is live with a positive refcount;
- reusable prefix identities still match current block ID, generation, hash,
  and exact request tokens.

It then sets:

```text
seq.block_table
seq.num_cached_tokens
seq.num_computed_tokens
reservation.state = "attached"
```

No refcount changes occur during attachment because ownership transfers from
the reservation to the request.

## Release

`release_sequence_reservation()` accepts only `reserved`.

It decrements exactly one reference for every reserved block in reverse table
order. Zero-ref blocks return to the free list. New suffix blocks retain no
published hash metadata, while prefix blocks retain reusable metadata under
the existing lifecycle.

The reservation becomes `released`. Double release or attach fails closed.

## Existing API Compatibility

`BlockManager.allocate()` remains available and behavior-compatible in this
phase. It may later be implemented through the new reservation API after the
scheduler integration is proven, but this phase does not refactor it.

`PrefixBlockReservation` and
`Qwen35HybridPrefixAcquireCoordinator` remain valid focused primitives. The
future cross-process integration may use the new complete reservation rather
than letting the coordinator own prefix attachment.

## Correctness Test Matrix

Extend `tools/test_chunked_prefill.py`:

1. cold one-block and multi-block sequence reservations do not mutate the
   request;
2. warm one-block and multi-block prefixes produce the same table and cached
   token count as existing `allocate()`;
3. exact block-aligned prompts retain the sampleable-token cap;
4. live and idle prefix blocks reserve correctly;
5. hash collision with different tokens is a miss;
6. prefix miss causes all later blocks to be newly allocated;
7. insufficient capacity fails before mutation;
8. injected later suffix allocation failure rolls back prefix and earlier
   suffix ownership;
9. attachment transfers complete-table ownership without refcount changes;
10. release returns cold suffix blocks and preserves idle prefix metadata;
11. stale prefix generation/hash before attachment fails without request
    mutation and the reservation remains releasable;
12. dirty requests, malformed reservations, and repeated terminal operations
    fail closed;
13. existing `allocate()`, chunked prefill, speculative append, cache
    collision, and admission tests remain green.

## Acceptance Gate

This phase is complete only when:

- reservation RED/GREEN tests pass;
- the full Python 3.12 zero-argument chunked-prefill matrix passes with only
  the existing Config AST skip;
- Qwen3.5 acquisition/cache and hybrid-state regression remains green;
- Python 3.9 and Python 3.12 compilation passes;
- `git diff --check` passes;
- staged files remain empty;
- handoff records the cross-process integration boundary.

Allowed conclusion:

> TinyLLMForge can privately reserve and atomically attach a complete KV block
> table containing an exact reusable prefix and an uncached suffix.

Not established:

- scheduler hybrid prefix admission;
- ModelRunner state snapshot restore;
- cross-process rollback;
- GPU correctness or any performance/cache improvement.
