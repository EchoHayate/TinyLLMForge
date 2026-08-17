# Generic Speculative KV Transaction Design

## Objective

Create the first model-independent runtime primitive required by generic
MTP/speculative decoding: a transactional KV append reservation with explicit
ownership, generation validation, materialization acknowledgement, and
exactly-once commit or rollback.

The current speculative profiler reserves append capacity as a bare
`list[int]`. The caller must remember which sequence owns the blocks, whether
the verifier wrote KV into them, which generations were reserved, and whether
commit or cleanup already happened. That is sufficient for a single profiler
function, but it is not a safe contract for scheduler batching, MTP adapters,
or KV offload.

This phase adds a dependency-light CPU contract. It does not move speculative
execution into the scheduler, implement an MTP draft model, change model math,
or claim a throughput improvement.

## Scope

This phase covers:

- one live sequence per transaction;
- append-capacity reservation for a proposed speculative token count;
- exact logical block ownership and generation capture;
- explicit acknowledgement of how many appended token KV entries were
  materialized;
- partial/full/zero-accept metadata commit;
- release of the rejected or unused reserved suffix;
- rollback after verifier or acceptance failure;
- stale generation, wrong sequence, sequence drift, and repeated terminal
  operation rejection;
- CPU-only model-independent tests.

This phase does not cover:

- scheduler batching or fairness;
- a draft-source adapter or MTP model;
- target-model forward execution;
- CUDA Graph capture;
- physical KV residency, H2D copies, writeback, or eviction;
- cross-process transaction serialization;
- performance benchmarks.

## Existing Boundary

The existing path is:

```text
profiler verify_and_commit_block()
  -> BlockManager.reserve_append_blocks()
  -> target decode + speculative tail forward
  -> BlockManager.commit_accepted_tokens()
  -> release_reserved_blocks() on failure
```

The verifier can already write accepted KV directly when
`verifier_mode="native"`. The missing primitive is lifecycle ownership:

- reservation returns untyped block IDs;
- no sequence identity is attached;
- no block generation is recorded;
- no state transition proves verifier materialization happened;
- commit and cleanup are caller-owned and only assertion-protected;
- repeated commit/release and stale ownership are not fail-closed APIs.

## Alternatives

### 1. Keep Bare Block IDs and Add More Caller Checks

Rejected. Every future caller would need to reproduce sequence, generation,
materialization, and exactly-once checks. Scheduler batching would multiply
the number of failure paths.

### 2. Store Speculative Transaction State on `Sequence`

Rejected. `Sequence` is serialized across engine boundaries and represents
request metadata, while a reservation is temporary BlockManager ownership.
Embedding allocator lifecycle state in `Sequence` would couple request
serialization to one speculative implementation and complicate rollback.

### 3. BlockManager-Owned Transaction Object

Selected. `BlockManager` already owns block allocation, generations,
refcounts, prefix publication, and release. A transaction object records the
minimum immutable request snapshot plus reserved ownership, while manager
methods enforce transitions and mutate the sequence only at commit.

## Data Model

Add to `tinyvllm/engine/block_manager.py`:

```python
@dataclass
class SpeculativeKVTransaction:
    sequence_id: int
    original_num_tokens: int
    original_last_token: int
    original_block_table: tuple[int, ...]
    reserved_block_ids: tuple[int, ...]
    reserved_block_generations: tuple[int, ...]
    proposed_token_count: int
    materialized_token_count: int = 0
    state: str = "reserved"
```

Field semantics:

- `sequence_id` is the logical request owner.
- `original_num_tokens`, `original_last_token`, and
  `original_block_table` form the commit-time sequence snapshot.
- `reserved_block_ids` are private new-content blocks not yet visible in the
  request block table.
- `reserved_block_generations` reject ABA reuse of a block ID.
- `proposed_token_count` is the maximum number of draft tokens this
  transaction may accept.
- `materialized_token_count` counts appended token positions whose KV was
  written by the verifier. It does not count the pre-existing pending token.
- `state` follows the explicit state machine below.

The transaction deliberately carries logical block IDs only. It does not
store physical KV slots, CUDA pointers, offload residency, or transfer
statistics.

## API

Add:

```python
def begin_speculative_kv_transaction(
    self,
    seq: Sequence,
    proposed_token_count: int,
) -> SpeculativeKVTransaction
```

```python
def mark_speculative_kv_materialized(
    self,
    transaction: SpeculativeKVTransaction,
    materialized_token_count: int,
) -> None
```

```python
def commit_speculative_kv_transaction(
    self,
    transaction: SpeculativeKVTransaction,
    seq: Sequence,
    accepted_tokens: list[int],
) -> None
```

```python
def rollback_speculative_kv_transaction(
    self,
    transaction: SpeculativeKVTransaction,
    seq: Sequence,
) -> None
```

The existing `reserve_append_blocks()`, `release_reserved_blocks()`, and
`commit_accepted_tokens()` remain behavior-compatible during this phase.
Profiler/runtime migration is a later integration step.

## State Machine

```text
reserved
  -> materialized
       -> committed
       -> rolled_back
  -> rolled_back
```

Rules:

- begin returns `reserved`;
- materialization may be acknowledged exactly once, including a count of zero;
- commit requires `materialized`;
- rollback accepts `reserved` or `materialized`;
- `committed` and `rolled_back` are terminal;
- every repeated or invalid transition raises without mutating the sequence
  or allocator.

Materialization is an acknowledgement from the execution layer, not an
inference made by BlockManager. This prevents a logical reservation from
being mistaken for a completed GPU write or H2D transfer.

## Reservation Semantics

`begin_speculative_kv_transaction()` validates before allocation:

- `seq` is a `Sequence`;
- `proposed_token_count` is a positive integer and not `bool`;
- `seq.seq_id` is a non-negative integer;
- every current block-table ID is valid, live, and positively referenced;
- enough free blocks exist for all verifier-visible appended KV positions.

For `N` proposed tokens, the verifier can materialize at most `N - 1`
appended token positions. The last accepted token remains the normal pending
decode token. Required capacity is therefore:

```text
materialized_end = len(seq) + max(0, N - 1)
required_blocks = ceil(materialized_end / block_size)
missing_blocks = max(0, required_blocks - len(seq.block_table))
```

Every missing block is allocated as unpublished new content. The transaction
captures each block generation after allocation. The sequence is not mutated.

If allocation fails after partial ownership acquisition, all acquired blocks
are released before the exception escapes.

## Materialization Semantics

`mark_speculative_kv_materialized()` accepts an integer in:

```text
0 <= materialized_token_count <= proposed_token_count - 1
```

It validates all reserved block IDs, generations, live ownership, refcount
`1`, and unpublished hash state before changing the transaction state.

The runtime calls this method only after the target verifier has completed
the corresponding KV writes. A KV-offload adapter must separately ensure
physical residency and writeback requirements before acknowledgement.

## Commit Semantics

Commit validates all conditions before request mutation:

- transaction structure and state are valid;
- `seq.seq_id` matches `sequence_id`;
- token count, last token, and block table equal the original snapshot;
- every original and reserved block is still live;
- every reserved generation matches and each block is still private,
  unpublished new content;
- accepted count does not exceed the proposal;
- accepted tokens are integers and not booleans;
- the accepted materialized prefix
  `max(0, len(accepted_tokens) - 1)` does not exceed
  `materialized_token_count`;
- reserved capacity covers the accepted materialized prefix.

After validation:

1. attach only the reserved blocks needed by accepted materialized KV;
2. append accepted tokens to the sequence;
3. publish only full blocks covered by `materialized_tokens =
   original_num_tokens + max(0, accepted_count - 1)`;
4. release every unused reserved block;
5. set state to `committed`.

Zero acceptance is a valid commit: all reserved blocks are released, the
sequence remains unchanged, and the transaction becomes `committed`.

The rejected suffix is never appended and its private blocks are returned to
the free list. Accepted KV is not replayed or copied by this contract.

## Rollback Semantics

Rollback validates the logical sequence owner but intentionally does not
require the original sequence snapshot to remain unchanged. Cleanup must
remain possible after a verifier-side failure mutates unrelated request
metadata.

Rollback:

- accepts only `reserved` or `materialized`;
- requires the provided sequence ID to match;
- validates reserved block generations and private ownership;
- releases each reserved block exactly once;
- leaves sequence tokens and block table untouched;
- sets state to `rolled_back`.

A wrong sequence, stale generation, or repeated rollback fails closed rather
than releasing a block that may now belong to another owner.

## Failure Atomicity

All expected validation and capacity failures happen before sequence
mutation.

Reservation allocation uses explicit partial-acquisition cleanup.

Commit computes and validates the final block-table extension, accepted token
metadata, and full-block publication inputs before the commit point. After the
commit point it performs only local list/scalar updates, deterministic prefix
registration, and release of transaction-owned blocks; it invokes no model,
offload, scheduler, or user callback.

This phase does not promise recovery from process termination or Python
`MemoryError`. Those require a persistent or cross-process transaction log,
which is outside this CPU runtime slice.

## KV Offload Boundary

The transaction is storage-topology agnostic:

- it owns logical blocks, not physical slots;
- it does not call `ensure_resident()`;
- it does not perform or simulate H2D copies;
- it does not mark blocks dirty or write them back;
- it does not report saved bytes.

Future integration order:

```text
begin transaction
-> offload adapter resolves/reserves physical residency
-> target verifier writes KV
-> offload adapter completes required writeback/ownership step
-> mark materialized
-> acceptance
-> commit or rollback
```

This boundary allows the same transaction to serve resident KV, real KV
offload, MTP, n-gram, and other draft sources without model-name branches.

## Correctness Matrix

Create `tools/test_speculative_kv_transaction.py` with CPU-only tests for:

1. reservation does not mutate the sequence;
2. capacity uses verifier-visible `N - 1` appended KV positions;
3. insufficient capacity fails without allocator or request mutation;
4. transaction records exact reserved generations;
5. materialization accepts zero and the maximum valid count;
6. invalid and repeated materialization fail closed;
7. commit rejects wrong sequence, sequence drift, stale generations, and
   unpublished-ownership violations before mutation;
8. commit rejects acceptance above the proposal or above materialized KV;
9. zero, one, partial, and full acceptance commit the correct tokens and
   blocks;
10. rejected/unused blocks are released;
11. full materialized blocks publish the same prefix metadata as the normal
    block manager path;
12. rollback works from reserved and materialized states;
13. rollback tolerates same-owner sequence drift but rejects a wrong owner;
14. commit and rollback are exactly once;
15. existing speculative and chunked-prefill tests remain green.

## Completion Boundary

Passing this phase proves only that speculative append KV ownership has a
model-independent, fail-closed CPU transaction contract.

It does not prove:

- scheduler/runtime integration;
- real MTP support;
- exact greedy model parity;
- TP1/TP4 behavior;
- 4K/16K/32K context behavior;
- TPOT, TTFT, throughput, memory, H2D-byte, or acceptance improvement.

Those measurements become mandatory after the transaction is integrated into
the real batched runtime and exercised on at least two model structures.
