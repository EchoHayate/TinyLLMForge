# Qwen3.5 Hybrid Prefix Restore Ticket Design

## Objective

Define the Engine↔ModelRunner transaction required to reuse a Qwen3.5 hybrid
prefix without publishing KV or recurrent-state metadata before every runtime
participant has restored the aligned state successfully.

The current CPU primitives can already:

- privately reserve a complete KV block table containing an exact cached
  prefix plus uncached suffix blocks;
- allocate a generation-checked hybrid-state lease;
- restore one exact Qwen3.5 prefix snapshot into one or more state-pool rows;
- attach KV and lease metadata in one process.

The real runtime splits those owners:

- Scheduler/Engine owns `BlockManager`, `Sequence`, request queues, and
  `HybridStateSlotAllocator`;
- each ModelRunner rank owns its `HybridStateTensorPool`, layer adapters, and
  rank-local snapshot tensors.

This phase adds a dependency-light CPU protocol that models this split with an
explicit restore ticket, per-participant acknowledgement, and deterministic
rollback. It does not wire the protocol into `LLMEngine.step()` or the current
shared-memory worker RPC.

## Alternatives

### 1. Restore State Directly in Scheduler

Rejected. Scheduler does not own the tensors used by model execution.
Duplicating pools or snapshots there would create incorrect process and GPU
ownership.

### 2. Publish Sequence Metadata Before Worker Restore

Rejected. A worker miss or failure would leave the request advertising cached
KV and hybrid-state metadata that are not usable on every rank.

### 3. Fire-and-Forget Restore Through Existing Worker RPC

Rejected. `ModelRunner.loop()` discards return values and does not report
worker exceptions to rank 0. Rank 0 therefore cannot prove that every TP
participant prepared.

### 4. Two-Phase Restore Ticket With Explicit Acknowledgements

Selected. Engine-owned resources remain private while every participant
prepares. Only an all-prepared ticket can publish request metadata.

## Components

Create `tinyvllm/engine/qwen35_hybrid_prefix_restore_ticket.py`.

### Restore Payload

`Qwen35HybridPrefixRestorePayload` is immutable and contains only values that
can cross a process boundary:

```python
@dataclass(frozen=True)
class Qwen35HybridPrefixRestorePayload:
    ticket_id: int
    request_id: int
    key: Qwen35HybridPrefixKey
    token_ids: tuple[int, ...]
    block_identities: tuple[tuple[int, int, int], ...]
    lease: HybridStateLease
```

The payload binds one request, one exact token boundary, one exact KV identity
chain, and one generation-checked state slot. It does not contain
`SequenceBlockReservation`, `Sequence`, `BlockManager`, or tensor objects.

### Prepare Acknowledgement

`Qwen35HybridPrefixPrepareAck` is immutable:

```python
@dataclass(frozen=True)
class Qwen35HybridPrefixPrepareAck:
    ticket_id: int
    participant_id: int
    status: str
    detail: str = ""
```

`status` is exactly one of:

- `prepared`: the participant owns an activated lease row containing the
  restored snapshot;
- `miss`: no matching snapshot was restored and the participant retained no
  lease-row binding;
- `error`: validation or restore failed and the participant retained no
  lease-row binding.

The participant converts ordinary validation and restore exceptions into an
`error` acknowledgement. Process death and transport failure remain RPC-layer
errors for the later runtime integration.

### Engine-Owned Ticket

`Qwen35HybridPrefixRestoreTicket` is local control-plane state:

```python
@dataclass
class Qwen35HybridPrefixRestoreTicket:
    payload: Qwen35HybridPrefixRestorePayload
    sequence: Sequence
    reservation: SequenceBlockReservation
    participant_ids: tuple[int, ...]
    acknowledgements: tuple[Qwen35HybridPrefixPrepareAck, ...] = ()
    state: str = "reserved"
```

Ticket states are:

```text
reserved -> prepared -> committed
    |           |
    +-----------+-> rolled_back
                    rollback_failed
```

`rollback_failed` is a terminal failure state used when participant cleanup
cannot be proven complete. No transition out of `committed`, `rolled_back`,
or `rollback_failed` is valid.

### ModelRunner Participant

`Qwen35HybridPrefixRestoreParticipant` owns:

- one non-negative `participant_id`;
- one `HybridStateTensorPool`;
- one rank-local `Qwen35HybridPrefixSnapshotCache`;
- an internal map of prepared ticket IDs to exact payloads.

`prepare(payload)` performs:

1. payload and participant-local identity validation;
2. `pool.activate(payload.lease)`;
3. exact snapshot lookup and cross-layer restore;
4. recording the payload as prepared.

On miss or error it releases an activated row before returning the
acknowledgement. Duplicate prepare with the identical payload is idempotent;
the same ticket ID with different contents returns `error`.

`validate_prepared(payload)` is a read-only precommit check.

`commit(payload)` accepts only an exactly prepared payload and removes the
private prepared marker while leaving the restored lease row active for model
execution. After `validate_prepared()` succeeds in the single-threaded CPU
contract, `commit()` is a non-allocating dictionary deletion and must not
perform tensor work.

`rollback(payload)` accepts an exactly prepared payload, releases and zeroes
the pool row, and removes the prepared marker. Rolling back an unknown or
already terminal payload fails closed.

## Coordinator

`Qwen35HybridPrefixRestoreCoordinator` owns:

- `BlockManager`;
- `HybridStateSlotAllocator`;
- a non-empty tuple of participants with unique participant IDs;
- a monotonic local ticket-ID counter.

It deliberately does not own a tensor pool or snapshot cache.

### Reserve

```python
def reserve(
    self,
    sequence: Sequence,
    key: Qwen35HybridPrefixKey,
    token_ids: tuple[int, ...],
) -> Optional[Qwen35HybridPrefixRestoreTicket]
```

Reserve validates before mutation:

- destination KV and hybrid-state metadata are pristine;
- no allocator lease already exists for the request;
- key and exact tokens are valid;
- key block size matches `BlockManager`;
- destination prompt begins with the exact token tuple;
- key tensor-parallel size equals participant count.

It then:

1. calls `reserve_sequence_blocks(..., max_cached_tokens=key.token_count)`;
2. treats a shorter cached prefix as a clean miss and releases the complete
   reservation;
3. requires the reserved prefix identities to cover exactly the key boundary;
4. allocates one central lease;
5. returns a `reserved` ticket.

Any exception after resource acquisition releases both the lease and complete
KV reservation. A clean KV miss returns `None` with a pristine request.

### Prepare

```python
def prepare(
    self,
    ticket: Qwen35HybridPrefixRestoreTicket,
) -> tuple[Qwen35HybridPrefixPrepareAck, ...]
```

Prepare is valid only from `reserved`. It invokes every participant in stable
participant-ID order and records every acknowledgement.

If every acknowledgement is `prepared`, the ticket becomes `prepared`.

On the first `miss` or `error`, the coordinator:

1. rolls back every participant that already acknowledged `prepared`;
2. releases the allocator lease;
3. releases the complete KV reservation;
4. marks the ticket `rolled_back`;
5. returns the acknowledgements observed through the failing participant.

The destination request remains pristine.

If coordinator code itself raises unexpectedly, the same rollback is
attempted and the original exception is re-raised.

### Commit

```python
def commit(
    self,
    ticket: Qwen35HybridPrefixRestoreTicket,
) -> None
```

Commit is valid only from `prepared`. Before any request mutation it verifies:

- acknowledgement count and participant IDs exactly match the configured
  participants;
- every acknowledgement status is `prepared`;
- allocator lease ownership still matches;
- sequence KV and hybrid-state metadata remain pristine;
- every participant still validates the exact prepared payload;
- the complete KV reservation remains attachable.

The publication sequence is:

1. attach `SequenceBlockReservation`;
2. publish `hybrid_state_slot_id` and `hybrid_state_generation`;
3. call each participant's prevalidated, non-tensor `commit()`;
4. mark the ticket `committed`.

The CPU protocol is single-threaded. It does not claim crash consistency
between these in-process assignments. The later cross-process runtime must
pair this precommit validation with a real acknowledgement barrier and worker
liveness policy.

### Explicit Rollback

```python
def rollback(
    self,
    ticket: Qwen35HybridPrefixRestoreTicket,
) -> None
```

Rollback is valid from `reserved` or `prepared`:

- prepared participants release their pool rows;
- the allocator lease is released;
- the complete KV reservation is released;
- the ticket becomes `rolled_back`;
- the request remains pristine.

Participant rollback is best-effort across every prepared participant. Engine
resources are still released when one participant cleanup fails, but the
ticket becomes `rollback_failed` and the cleanup exception is raised rather
than falsely reporting complete rollback. The coordinator is then poisoned
and rejects new reservations because a participant may still hold a stale
binding for an allocator slot that has returned to the free list. Recovery
requires reconstructing the coordinator, allocator, and participants.

Rollback after commit or repeated rollback fails closed.

## Invariants

At all observable boundaries:

1. a `reserved` ticket owns KV and allocator resources but publishes no
   request metadata and no participant is prepared;
2. a `prepared` ticket additionally owns restored rows on every participant
   but still publishes no request metadata;
3. a `committed` ticket has transferred KV and lease identity to the request,
   and all participant rows remain active;
4. a `rolled_back` ticket owns no KV references, allocator lease, pool binding,
   or request metadata;
5. exact token comparison is required in addition to chained block hashes;
6. block generation, model/layout fingerprint, TP size, dtype, and state-lease
   generation remain part of the identity;
7. a partial participant prepare can never make the request schedulable.

## Correctness Test Matrix

Create `tools/test_qwen35_hybrid_prefix_restore_ticket.py` using real
`BlockManager`, allocator, pools, adapters, transactions, and snapshot caches.

The test matrix covers:

1. reserve privately holds a complete prefix-plus-suffix table and one lease
   without mutating the request;
2. a shorter KV prefix is a clean miss with complete rollback;
3. exact destination tokens, TP size, key identity, and pristine request
   validation fail before publication;
4. one participant prepare restores the expected cross-layer state;
5. multiple participants restore rank-local snapshots and return explicit
   acknowledgements;
6. participant cache miss rolls back earlier prepared participants and every
   Engine-owned resource;
7. participant validation/restore error returns `error` and produces the same
   rollback;
8. explicit rollback from `reserved` and `prepared` restores pristine resource
   observations;
9. commit publishes the complete KV table and lease metadata only after all
   participants are prepared;
10. stale allocator, participant, reservation, or destination metadata fails
    before publication and leaves the ticket explicitly rollbackable;
11. duplicate prepare, conflicting ticket payloads, repeated terminal
    transitions, and rollback after commit fail closed;
12. payload and acknowledgement objects are pickle round-trip safe;
13. existing block reservation, snapshot cache, acquisition, transaction,
    scheduler, sequence, runtime-bridge, and ModelRunner dependency-light
    regressions remain green.

## Acceptance Gate

This phase is complete only when:

- the focused ticket tests show an observed RED before implementation and pass
  after implementation;
- the full Python 3.12 zero-argument chunked-prefill matrix retains only the
  documented Config AST skip;
- all Qwen3.5 hybrid prefix and hybrid-state CPU regressions pass;
- Python 3.9 and Python 3.12 `py_compile` pass;
- `git diff --check` passes;
- `git diff --cached --name-only` remains empty;
- no `experiments/` evidence is removed;
- handoff records both the achieved CPU protocol and the missing RPC barrier.

Allowed conclusion:

> TinyLLMForge has a CPU-tested two-phase hybrid-prefix restore ticket that
> keeps complete KV ownership, state leases, and per-rank restored state
> private until every modeled participant acknowledges prepare.

Not established:

- scheduler admission or `LLMEngine.step()` integration;
- real worker return/error acknowledgement;
- TP process or GPU correctness;
- checkpoint logits or generated-token equivalence;
- TTFT, throughput, cache hit rate, compression, or memory improvement.

