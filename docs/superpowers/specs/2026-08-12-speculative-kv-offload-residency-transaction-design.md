# Speculative KV Offload Residency Transaction Design

**Date:** 2026-08-12

**Status:** Approved direction, implementation pending

**Classification before and after this gate:** `NOT_PROMOTABLE`

## Goal

Connect the existing generic speculative KV transaction to real
`KVOffloadMVP0` residency so that:

- verifier-written accepted KV remains in place without replay or KV copy;
- rejected speculative reserved blocks are invalidated exactly once without
  dirty writeback;
- allocator ownership, Scheduler metadata, and rank-local residency publish
  in a defined transaction order;
- real H2D/D2H/copy/wait/eviction counters come only from
  `KVOffloadMVP0`;
- an offload-enabled loaded-model TP1 gate can prove exact greedy parity.

This design is source-agnostic and model-name-free. N-gram is the first gate
source only because it is stateless and already batch capable.

## Current State

The repository already has:

- logical allocator ownership in `BlockManager`;
- private speculative reserved blocks;
- native multi-token verifier writes into the target KV cache;
- token-free allocator commit plans;
- rollback-safe Scheduler metadata preparation;
- logical KV blocks mapped to physical GPU slots by `KVOffloadMVP0`;
- pinned CPU backing, async H2D/D2H, batched copies, dirty tracking,
  writeback, eviction, and real movement counters.

The two ownership systems are currently independent:

- `BlockManager` runs in the engine process and owns logical block lifetime;
- `KVOffloadMVP0` lives in each ModelRunner rank and owns physical GPU slot
  residency plus copy events;
- `ModelRunner._validate_spec_verify_compatibility()` explicitly rejects
  `kv_offload_mvp0`;
- speculative verifier block tables are therefore currently interpreted as
  direct physical block IDs.

The integration cannot be implemented by calling `KVOffloadMVP0` from
`BlockManager`. Rank-local GPU residency must remain behind ModelRunner RPCs.

## Non-Goals

This gate does not implement:

- learned draft models or MTP heads;
- stochastic speculative decoding;
- recurrent or convolution state transactions;
- TP4;
- a second model structure;
- KV8 or KV4 offload;
- variable-Q CUDA Graph capture;
- fused verification, sampling, or commit kernels;
- proactive speculative prefetch beyond blocks required by the verifier;
- stable TPOT, TTFT, throughput, or memory improvement claims;
- restoration of the exact pre-transaction physical GPU slot layout.

## Alternatives

### BlockManager directly controls residency

Rejected. The allocator and GPU residency have different process and rank
ownership. Direct coupling would make TP workers invisible to the transaction
and would mix allocator semantics with CUDA events.

### Monolithic verifier/offload rewrite

Rejected. Enabling offload, staging blocks, tracking private writes, deciding
acceptance, committing allocator ownership, and handling distributed failure
in one change would make rollback evidence ambiguous.

### Engine-coordinated two-phase residency transaction

Selected. Each ModelRunner rank owns a private residency ticket. The engine
coordinates prepare, precommit, rollback, and seal acknowledgements around
the existing allocator and Scheduler transaction.

## Semantic Rollback Boundary

Rollback restores semantic ownership and visibility, not the exact prior GPU
slot arrangement.

Preparing a speculative verifier batch may evict an unrelated resident block.
If that block is dirty, existing `KVOffloadMVP0` logic writes it to pinned CPU
backing before slot reuse. On rollback:

- speculative reserved mappings are discarded;
- accepted ownership is not published;
- no rejected speculative block is written back;
- evicted ordinary blocks remain CPU-valid and may reload later.

The transaction does not move an ordinary block back to its old physical slot
solely to reproduce a previous layout. Such a copy is unnecessary for
correctness and would make rollback itself create avoidable H2D traffic.

## Residency Ticket

Add a focused rank-local module:

```text
tinyvllm/engine/speculative_residency.py
```

The module owns validation and state transitions but delegates all mapping,
copy, dirty, and invalidation operations to the existing residency manager.

```python
@dataclass(frozen=True)
class SpeculativeResidencyRow:
    sequence_id: int
    original_block_identities: tuple[tuple[int, int], ...]
    reserved_block_identities: tuple[tuple[int, int], ...]
    materialized_block_identities: tuple[tuple[int, int], ...]


@dataclass
class SpeculativeResidencyTicket:
    ticket_id: int
    rows: tuple[SpeculativeResidencyRow, ...]
    state: str = "prepared"
    committed_block_identities: tuple[tuple[int, int], ...] = ()
    rejected_block_identities: tuple[tuple[int, int], ...] = ()
```

Required states:

```text
prepared -> precommitted -> sealed
prepared -> rolled_back
precommitted -> rolled_back
```

All other transitions fail closed. Repeated rollback of a rolled-back ticket
and repeated seal of a sealed ticket are rejected rather than silently
changing counters twice.

## Generation-Aware Residency Identity

Allocator logical block IDs are reusable. `Block.reset()` increments a block
generation, and speculative transactions already record
`reserved_block_generations`.

`KVOffloadMVP0` currently keys mapping, CPU validity, dirty state, and copy
events only by logical block ID. Without generation binding, a newly allocated
speculative block can inherit:

- a stale resident slot from a previous owner;
- stale pinned-CPU validity;
- stale H2D or D2H events;
- stale dirty state.

Therefore every residency ticket and RPC uses:

```text
(logical_block_id, allocator_generation)
```

as the ownership identity.

Extend `KVOffloadMVP0` with generation metadata and one validated operation:

```python
def bind_logical_block_identity(
    self,
    logical_block: int,
    generation: int,
) -> None:
    ...
```

Binding the same generation is idempotent. Binding a newer generation:

- discards the previous owner's resident mapping without D2H;
- clears stale CPU validity, dirty state, pending waits, and copy events;
- records the new generation before the block can stage or receive writes.

Binding an older generation fails closed.

An unbound logical block may be bound only when it has no resident mapping,
CPU-valid backing, dirty state, pending wait, or copy event. The manager must
never attach a generation label to pre-existing bytes by assumption.

Therefore this gate also propagates allocator identities through every
ordinary `KVOffloadMVP0` access before speculative residency is enabled:

- the engine derives `(block_id, generation)` from the allocator for every
  logical block table sent to an offload-enabled ModelRunner;
- the ModelRunner binds those identities before ordinary staging or writes;
- a newly allocated generation clears stale state before its first write;
- a reactivated cached block keeps the same generation and may reuse its
  valid backing;
- release requires no rank RPC because a later newer-generation bind performs
  the invalidation.

This propagation is limited to `KVOffloadMVP0`; other offload modes remain
outside this gate. Speculative prepare binds every reserved identity before
`ensure_resident(require_valid=False)` and requires every original identity
to match an already bound ordinary-path identity.

Extend `SpeculativeKVTransaction` with:

```python
original_block_generations: tuple[int, ...]
```

`begin_speculative_kv_transaction()` snapshots these generations alongside
`original_block_table`, and every later transaction validation requires the
original `(block_id, generation)` pairs to remain current. The engine derives
all rank-local original and reserved identities from this allocator-owned
snapshot; ModelRunner ranks never invent or query allocator generations.

## Block Classification

For a sequence with history length `L` and verifier input positions:

```text
plan.logical_slots
```

the ModelRunner derives:

- historical read blocks: blocks covering positions `[0, L)`;
- materialized write blocks: blocks containing `plan.logical_slots`;
- original write blocks: materialized blocks already in the original block
  table;
- reserved write blocks: materialized blocks from the allocator reservation.

Rejected token slots inside an original partially filled block remain
invisible because the committed sequence length does not include them.
They do not require zeroing or a whole-block invalidation. Future committed
decode writes overwrite those positions.

Only rejected reserved logical blocks lose residency ownership.

## Prepare Flow

After proposal generation and allocator reservation, before the verifier
forward:

1. the engine sends each rank the selected rows, original logical block
   tables, proxy logical block tables, and verify plans;
2. the rank validates that every proxy suffix is composed only of the
   provided reserved logical block identities;
3. every reserved `(block_id, generation)` is bound, clearing stale state
   from a previous allocator owner;
4. historical read blocks stage through
   `KVOffloadMVP0.ensure_resident(require_valid=True)`;
5. reserved write blocks stage through
   `KVOffloadMVP0.ensure_resident(require_valid=False)`;
6. historical and write blocks are protected during the staging operation;
7. the ticket records original, reserved, and materialized block identities;
8. physical block tables and slot mappings are derived from the resulting
   `logical_to_slot` mapping;
9. required H2D events are awaited before the verifier forward.

No speculative block is added to global dirty/writeback state during prepare.
The ticket privately owns the verifier write until acceptance is known.

If prepare fails after assigning some reserved mappings, the rank discards
only those speculative reserved mappings. Ordinary evictions remain valid
because dirty victims were already written to CPU backing by the existing
manager.

## Verifier Forward

`run_spec_verify_batch()` may execute with `kv_offload_mvp0` only when all
rows have a valid prepared residency ticket.

The forward:

- uses physical slot mappings derived from the ticket;
- performs one fixed-Q target forward per existing grouping rules;
- does not call the ordinary `_kv_offload_after_forward()` path;
- does not write back verifier-written speculative blocks before acceptance;
- marks the ticket's materialized block set after a successful forward.

The existing fail-closed checks remain for:

- KV quantization;
- blockwise offload modes not covered by this first full-attention gate;
- Quest, AM compact, and KV cartridge modes;
- mixed prefill/decode verifier execution;
- non-greedy sampling;
- non-KV hybrid state.

## Precommit Flow

After greedy acceptance and before allocator mutation, the engine sends each
rank a generation-aware projection of the allocator
`SpeculativeKVCommitPlan`:

```text
sequence_id
committed_reserved_block_identities
unused_reserved_block_identities
accepted_materialized_end
```

The existing allocator plan remains ID-based. The engine converts its
`committed_block_ids` and `unused_block_ids` to identities by exact lookup in
the transaction's zipped
`(reserved_block_ids, reserved_block_generations)` snapshot. A missing,
duplicate, reordered, or cross-partition ID fails before any rank RPC.

Each rank validates:

- ticket and sequence identity;
- exact disjoint partition of reserved blocks into committed and unused;
- every committed reserved block was materialized;
- no unused block is reported as committed;
- accepted materialized positions do not exceed the verifier write range.

Precommit records the accepted and rejected residency plan but performs no
destructive invalidation. It remains rollbackable.

All ranks must acknowledge precommit before allocator commit begins.

## Engine Publication Order

The engine publication order is:

```text
1. prepare rank-local residency tickets
2. run verifier and compute acceptance
3. precommit every rank-local residency ticket
4. commit allocator ownership batch
5. commit prepared Scheduler metadata
6. seal every rank-local residency ticket
7. synchronize optional draft lifecycle
```

Before step 4, any failure rolls back every prepared or precommitted rank
ticket and every active allocator reservation.

Steps 4 and 5 retain the existing allocator/Scheduler transaction behavior.
This design does not weaken their current rollback tests.

Residency seal happens only after target tokens and allocator ownership are
authoritative. Seal is prevalidated and consists only of deterministic local
mapping/set transitions. A transport or rank failure during seal poisons the
speculative runtime and blocks further selected work; it must not pretend the
already committed target tokens were rolled back.

## Seal Semantics

For each precommitted ticket:

- original blocks remain mapped as before;
- committed reserved blocks keep their current physical slot and become
  ordinary residency;
- blocks containing accepted materialized positions are marked dirty;
- when `writeback_on_evict` is false, only accepted dirty blocks follow the
  existing immediate writeback policy;
- unused reserved blocks are discarded from `logical_to_slot`,
  `slot_to_logical`, dirty state, pending wait state, and stale copy-event
  indexes without D2H;
- rejected speculative slots in an original block remain invisible beyond
  sequence length.

No accepted KV replay, target recomputation, GPU-to-GPU copy, or
accepted-block H2D is allowed during seal.

## Rollback Semantics

Rollback from `prepared` or `precommitted`:

- discards all reserved logical block mappings owned by the ticket;
- clears their dirty, pending-wait, H2D-event, and D2H-event entries;
- performs no rejected-block D2H;
- leaves original block mappings and CPU backing validity intact;
- marks the ticket `rolled_back`;
- increments rollback/invalidation counters exactly once.

Rollback does not clear speculative bytes written into unused positions of an
original block because those positions are not visible to attention.

## KVOffloadMVP0 Extensions

Extend the existing manager rather than creating a second copy manager.

Required operations:

```python
def discard_resident_blocks(
    self,
    block_identities: tuple[tuple[int, int], ...],
    *,
    allow_dirty: bool,
) -> tuple[tuple[int, int], ...]:
    ...


def speculative_residency_summary(self) -> dict:
    ...
```

`discard_resident_blocks()`:

- validates the complete input before mutation;
- requires every generation to match the bound residency identity;
- requires each discarded block to be resident;
- rejects dirty blocks unless `allow_dirty=True`;
- clears both mapping directions and all block-local pending/event state;
- never enqueues D2H or H2D;
- returns the discarded block identities in input order;
- restores its metadata snapshot if an injected mutation failure occurs.

New real counters:

```text
speculative_residency_prepares
speculative_residency_precommits
speculative_residency_seals
speculative_residency_rollbacks
speculative_residency_committed_blocks
speculative_residency_rejected_blocks
speculative_residency_rejected_d2h_copies
```

The rejected-D2H counter must remain zero. Existing `h2d_bytes`,
`d2h_bytes`, copy counts, batches, waits, evictions, and dirty evictions
remain the movement source of truth.

## ModelRunner RPC Boundary

Add rank-local methods:

```text
prepare_speculative_residency_batch
precommit_speculative_residency_batch
rollback_speculative_residency_batch
seal_speculative_residency_batch
```

The engine calls these through the existing `ModelRunner.call()` transport.
Every response includes:

```text
ticket_id
operation
status
sequence_ids
committed_block_identities
rejected_block_identities
detail
```

The engine requires exact rank agreement on ticket identity, operation,
status, sequence IDs, and committed/rejected block identities. Missing,
duplicate, reordered, stale-generation, or error acknowledgements fail
closed.

## Failure Handling

Required injected failures:

- invalid proxy logical block table;
- unreadable historical block;
- insufficient GPU staging slots;
- failure after one speculative mapping assignment;
- verifier forward failure;
- precommit partition mismatch;
- one-rank precommit failure;
- allocator commit failure after successful residency precommit;
- Scheduler commit failure after allocator commit;
- rollback invalidation failure;
- seal acknowledgement failure.

Before allocator commit, failures must leave:

- no active speculative reserved mapping;
- no rejected D2H;
- no Scheduler token publication;
- allocator reservations rollbackable or rolled back;
- runtime reusable unless rollback itself fails.

Rollback failure poisons the runtime. Seal failure also poisons the runtime
because target state is already authoritative.

## Testing Strategy

### Dependency-light state machine

Create focused tests using a fake residency manager:

- exact state transitions;
- exact committed/rejected partition;
- zero accepted, partial accepted, full accepted;
- original-block-only writes;
- one and multiple reserved blocks;
- idempotence rejection;
- prepare/precommit rollback;
- injected discard failure and poison boundary.

### KVOffload metadata tests

Use a fake tensor/event surface to prove:

- reused logical block IDs cannot inherit stale resident/CPU/event state;
- same-generation binding is idempotent and older-generation binding fails;
- mixed read/write staging distinguishes `require_valid=True/False`;
- rejected reserved blocks are discarded without D2H;
- accepted reserved blocks retain the same physical slot;
- accepted blocks are marked dirty exactly once;
- rejected slots in an original block do not invalidate that block;
- unrelated dirty victim eviction remains CPU-valid;
- movement counters are not synthesized.

### Engine transaction tests

Extend engine speculative runtime tests to prove:

- all-rank prepare before verifier;
- all-rank precommit before allocator commit;
- allocator and Scheduler publication order;
- rollback RPC on all precommit-phase failures;
- seal only after Scheduler commit;
- post-commit seal failure poisons the runtime;
- ordinary non-speculative execution remains unchanged.

### Loaded-model gate

Extend the existing TP1 parity artifact with:

- `kv_offload_mvp0=true`;
- exact baseline/speculative output token equality;
- source hashes;
- residency prepare/precommit/seal/rollback counters;
- real H2D/D2H/copy/wait/eviction/writeback deltas;
- accepted and rejected residency block counts;
- assertion that rejected speculative D2H copies equal zero.

This first GPU gate is correctness evidence only. Controlled repeated
performance measurements are a later gate.

## Success Criteria

The implementation gate is green only when:

1. spec-verify remains fail closed without a valid residency ticket when
   offload is enabled;
2. reused allocator block IDs cannot observe stale residency generations;
3. accepted materialized KV keeps its physical slot through seal;
4. no accepted KV replay or copy occurs;
5. rejected reserved mappings invalidate exactly once;
6. rejected speculative blocks perform zero D2H;
7. allocator, Scheduler, and rank-local failure injection passes;
8. non-offload speculative behavior remains unchanged;
9. offload-enabled loaded-model TP1 exact greedy parity passes;
10. the artifact independently verifies source identity and real movement
   counters;
11. the audit remains `NOT_PROMOTABLE`.

## Promotion Boundary

Passing this design proves one real offload-enabled TP1 correctness path for
transactional speculative residency.

It does not prove:

- lower TPOT, TTFT, or memory;
- reduced H2D under controlled workloads;
- TP4 correctness;
- a second model structure;
- 4K/16K/32K+ context coverage;
- batch 1/4/mixed promotion coverage;
- learned-drafter or MTP support;
- KV8/KV4 residency;
- production readiness.

Those remain mandatory before promotion.
