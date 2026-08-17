# Qwen3.5 Hybrid Prefix Runtime Acquisition Design

## Objective

Build the dependency-light CPU transaction that safely acquires an exact
Qwen3.5 shared prefix.

The previous phase proved that one cache entry can bind an exact KV block
identity chain to one cross-layer convolution/recurrent state snapshot. It did
not prevent those KV blocks from being reused while state restoration was in
progress.

This phase closes that ownership gap:

```text
reserve exact KV block identities
+ allocate and activate destination hybrid-state leases
+ restore the matching state snapshot
+ publish both resources to destination requests
```

All request-visible metadata changes only after every resource and state
operation succeeds. A normal miss or any exception releases every newly
acquired KV reference and hybrid-state lease.

This remains a CPU correctness gate. It does not connect the coordinator to
the scheduler, ModelRunner, GPU KV tensors, or a real checkpoint.

## Existing Ownership Model

`BlockManager` remains the only KV owner:

- `used_block_ids` and `free_block_ids` describe physical block availability;
- `Block.ref_count` describes live request ownership;
- chained hashes plus exact token comparison identify reusable full blocks;
- an idle cached block keeps its hash and token metadata until the physical
  storage is assigned to different content.

`HybridStateSlotAllocator` owns request-to-slot leases.
`HybridStateTensorPool` owns the corresponding state tensors.
`Qwen35HybridPrefixSnapshotCache` owns only cloned state snapshots.

No second KV index or refcount system is introduced.

## Alternatives

### 1. Restore State Before Reserving KV

Rejected. Capacity pressure could recycle a matching block between lookup and
restore, pairing old state with unrelated KV bytes.

### 2. Attach KV Blocks to Requests Before State Restore

Rejected. A snapshot miss or copy failure would expose a half-acquired request
and force rollback through scheduler-visible metadata.

### 3. Reserve Exact KV, Restore State, Then Commit Metadata

Selected.

The coordinator keeps all resources private until the exact snapshot restore
succeeds. It then performs a small, prevalidated metadata commit.

### 4. Integrate Directly into `Scheduler._allocate_request_storage`

Deferred. The current scheduler allocates an entire block table, while this
phase proves only exact reusable-prefix ownership. Suffix-block reservation,
admission accounting, publication points, and scheduler/ModelRunner command
ordering require a separate integration gate.

## Block Generation

Add `generation` to `Block`.

The generation identifies the content lifetime of physical KV storage:

- a block starts at generation `0`;
- assigning a free block to new, unknown content increments the generation;
- deallocation does not increment it;
- reactivating an idle exact-token cache hit does not increment it;
- clearing only idle cache metadata does not increment it;
- later assigning that cleared/free block to new content increments it.

The existing `_allocate_block()` path becomes the new-content path. Exact
idle-prefix reuse uses a separate activation path that preserves hash, tokens,
and generation.

This distinction is required because the KV bytes remain valid across
deallocation and exact reactivation. Incrementing generation on every
refcount transition would reject valid snapshots; failing to increment on
new-content assignment would accept stale snapshots.

## Prefix Reservation

Add a focused reservation type to `tinyvllm/engine/block_manager.py`:

```python
@dataclass
class PrefixBlockReservation:
    block_ids: tuple[int, ...]
    block_identities: tuple[tuple[int, int, int], ...]
    token_count: int
    owner_count: int
    state: str = "reserved"
```

`state` is one of `reserved`, `attached`, or `released`. It prevents double
commit and double release.

Add these APIs:

```python
def reserve_exact_prefix(
    self,
    token_ids: tuple[int, ...],
    *,
    owner_count: int = 1,
) -> Optional[PrefixBlockReservation]

def attach_prefix_reservation(
    self,
    reservation: PrefixBlockReservation,
    sequences: tuple[Sequence, ...],
) -> None

def release_prefix_reservation(
    self,
    reservation: PrefixBlockReservation,
) -> None
```

`reserve_exact_prefix()` accepts only a positive, full-block-aligned token
tuple and a positive owner count. It computes the chained hashes and requires
an exact token match for every block. A partial chain is a normal miss and
returns `None` without mutation.

After the complete chain is found, reservation acquires `owner_count`
references for every block:

- a live block increases its existing `ref_count`;
- an idle cached block is activated without changing its generation, then
  receives the remaining references.

The reservation records identities only after all references are held:

```text
(block_id, generation, block_hash)
```

If acquisition raises, already acquired references are released before the
exception propagates.

`attach_prefix_reservation()` requires:

- reservation state is `reserved`;
- sequence count equals `owner_count`;
- sequence objects and `seq_id` values are unique;
- every destination has an empty block table;
- cached/computed token counts are zero.

It attaches the same reserved prefix block IDs to every destination, sets
`num_cached_tokens` and `num_computed_tokens` to `token_count`, and changes
the reservation state to `attached`. Refcounts do not change because
reservation ownership is transferred to the requests.

`release_prefix_reservation()` accepts only the `reserved` state, removes
exactly `owner_count` references from every block, returns zero-ref blocks to
the free list, and marks the reservation `released`.

## Acquisition Coordinator

Create
`tinyvllm/engine/qwen35_hybrid_prefix_acquisition.py`.

```python
class Qwen35HybridPrefixAcquireCoordinator:
    def __init__(
        self,
        block_manager: BlockManager,
        state_allocator: HybridStateSlotAllocator,
        state_pool: HybridStateTensorPool,
        snapshot_cache: Qwen35HybridPrefixSnapshotCache,
    )

    def acquire(
        self,
        sequences: tuple[Sequence, ...],
        key: Qwen35HybridPrefixKey,
        token_ids: tuple[int, ...],
    ) -> bool
```

Constructor validation requires the snapshot cache transaction to use the
same `HybridStateTensorPool` instance passed to the coordinator.

`acquire()` first validates all destination requests without mutation:

- `sequences` is a non-empty tuple;
- objects and `seq_id` values are unique;
- each request has no KV blocks and no hybrid-state metadata;
- `key.token_count == len(token_ids)`;
- `key.block_size == block_manager.block_size`.
- every destination prompt has at least `key.token_count` tokens and starts
  with the exact `token_ids` tuple.

The transaction order is:

1. reserve the exact KV prefix for `len(sequences)` owners;
2. return `False` immediately if the exact chain is absent;
3. allocate one hybrid-state lease per destination request;
4. activate every lease in the state tensor pool;
5. call `snapshot_cache.acquire()` with the reservation identities and all
   leases;
6. on a normal snapshot miss, release pool bindings, allocator leases, and KV
   reservation, then return `False`;
7. on success, attach the KV reservation to every request;
8. write each lease slot/generation into its request;
9. return `True`.

Any exception after reservation runs the same cleanup and re-raises. Cleanup
is ordered state-pool release, allocator release, then KV release.

All request metadata is prevalidated before resources are acquired.
Attachment assignments are the final non-throwing commit section. Therefore a
failed attempt leaves:

```text
block_table == []
num_cached_tokens == 0
num_computed_tokens == 0
hybrid_state_slot_id == -1
hybrid_state_generation == 0
```

## Scope Boundary

A successful coordinator call attaches only the exact reusable prefix. It
does not allocate uncached suffix blocks.

The next scheduler integration phase must add one transaction that reserves
the suffix blocks and commits the complete block table without exposing a
partially allocated request. Until that phase passes, the existing scheduler
fail-closed check remains unchanged.

## Correctness Test Matrix

Extend `tools/test_chunked_prefill.py`:

1. new-content assignment increments generation;
2. deallocation preserves generation;
3. idle exact-prefix reactivation preserves generation;
4. assigning the same physical block to different tokens increments
   generation and removes stale hash mappings;
5. exact one-block and multi-block reservations hold the requested refcount;
6. partial-prefix miss has no side effects;
7. reservation release restores live and idle ownership correctly;
8. reservation attachment transfers ownership to one and multiple sequences;
9. double attach/release and malformed destinations fail closed.

Create `tools/test_qwen35_hybrid_prefix_acquisition.py` with real
`BlockManager`, `HybridStateSlotAllocator`, `HybridStateTensorPool`,
cross-layer transaction, and snapshot cache:

1. exact one-destination acquisition restores state and attaches both
   resource identities;
2. one cached source row broadcasts to multiple out-of-order destinations;
3. missing KV chain returns `False` without allocating state;
4. stale snapshot generation returns `False` and releases all resources;
5. state-slot exhaustion releases the KV reservation;
6. injected state-pool activation failure releases prior leases and KV;
7. injected restore failure releases leases and KV while preserving request
   metadata;
8. snapshot miss after lease allocation releases every resource;
9. constructor and request/key validation fail before mutation;
10. wrong-prefix and too-short destination prompts fail before mutation;
11. successful request deallocation later returns the transferred KV
    references through existing `BlockManager.deallocate()`.

## Acceptance Gate

This phase is complete only when:

- focused generation/reservation tests pass;
- focused coordinator tests pass;
- prior prefix-cache, hybrid-state, state-transaction, scheduler fail-closed,
  and chunked-prefill tests pass;
- Python 3.9 and Python 3.12 compilation passes where supported;
- `git diff --check` passes;
- staged files remain empty;
- `AGENT_HANDOFF_STATE.md` records exact proof and remaining integration
  limits.

The allowed conclusion is:

> TinyLLMForge can reserve an exact KV prefix, restore its aligned Qwen3.5
> linear state, and publish both to CPU request metadata as one failure-atomic
> acquisition.

This phase does not establish native Qwen3.5 runtime support, exact
checkpoint logits/tokens, GPU correctness, TTFT, throughput, cache hit rate,
compression ratio, or physical memory reduction.
