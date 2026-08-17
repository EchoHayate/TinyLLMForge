# Engine Live Hybrid Prefix Restore Transaction Design

## Objective

Compose the already tested Engine acknowledged ModelRunner restore methods with
private KV reservation and the central hybrid-state allocator so one explicit
Engine control-plane call can restore an exact Qwen3.5 hybrid prefix without
publishing partial ownership.

This phase remains opt-in and CPU/static only. It does not call the coordinator
from `LLMEngine.step()`, does not change Scheduler admission, and does not
automatically construct checkpoint-specific pools, snapshot caches, or
participants.

## Chosen Architecture

Create a focused module:

```text
tinyvllm/engine/qwen35_hybrid_prefix_engine_restore.py
```

It owns:

```python
Qwen35HybridPrefixEngineRestoreTicket
Qwen35HybridPrefixEngineRestoreCoordinator
```

The coordinator receives:

- the live `LLMEngine` control-plane object;
- the Scheduler-owned `BlockManager`;
- the Scheduler-owned `HybridStateSlotAllocator`;
- a positive acknowledged-command timeout.

`LLMEngine` exposes explicit one-shot installation and acquisition methods. The
coordinator is not installed automatically.

Alternatives rejected:

1. Put the transaction in Scheduler admission. Rejected because Scheduler
   cannot own or copy ModelRunner state tensors and the runtime gate is not yet
   ready.
2. Extend the CPU-local participant coordinator with transport branches.
   Rejected because local object calls and uncertain multiprocess operations
   have different failure boundaries.
3. Put all transaction logic directly in `llm_engine.py`. Rejected because it
   would mix transport, resource transaction, and scheduling responsibilities.

## Request and Identity Validation

The coordinator accepts:

```python
acquire(
    sequence: Sequence,
    key: Qwen35HybridPrefixKey,
    token_ids: tuple[int, ...],
) -> bool
```

Before reserving resources it requires:

- a healthy, unpoisoned coordinator;
- a pristine destination Sequence with no KV or hybrid-state metadata;
- no allocator lease already owned by `sequence.seq_id`;
- exact prefix tokens at the beginning of the destination Sequence;
- key block size equal to the live BlockManager block size;
- key tensor-parallel size equal to `engine.model_runner.world_size`;
- an exact terminal chained block hash for `token_ids`;
- a positive timeout.

The Engine installation contract requires:

- the exact coordinator type;
- coordinator Engine identity equal to `self`;
- coordinator BlockManager identity equal to
  `self.scheduler.block_manager`;
- coordinator allocator identity equal to
  `self.scheduler.hybrid_state_allocator`;
- one-shot installation, except reinstalling the identical object.

## Transaction Flow

### 1. Private reserve

Reserve the complete Sequence block table:

```python
reservation = block_manager.reserve_sequence_blocks(
    sequence,
    max_cached_tokens=key.token_count,
)
```

The reservation privately holds both the exact cached prefix and all uncached
suffix blocks. It remains detached from the Sequence.

The reservation must have:

- `cached_tokens == key.token_count`;
- exact generation-aware prefix identities;
- terminal block hash equal to `key.terminal_block_hash`.

A prefix miss releases the reservation and returns `False`.

Then allocate one central:

```python
lease = state_allocator.allocate(sequence.seq_id)
```

Build a `Qwen35HybridPrefixRestorePayload` with a monotonic ticket ID.

### 2. All-rank prepare

Call:

```python
engine.prepare_model_runner_hybrid_prefix_restore(
    payload,
    timeout_s=timeout_s,
)
```

Success requires every ordered rank result to have status `prepared`.

If any valid result is `miss` or `error`, broadcast rollback to every rank,
then release the central allocator lease and private KV reservation, and
return `False`.

To make all-rank broadcast cleanup deterministic, participant rollback becomes
idempotent for an exact ticket already terminal as `rolled_back`. A committed
ticket remains non-rollbackable.

### 3. Pre-publication validation

Before publishing Sequence metadata:

- revalidate the allocator lease;
- revalidate the destination Sequence is pristine;
- revalidate the complete reservation structure and live block ownership;
- revalidate generation-aware exact prefix identities;
- call all-rank
  `validate_model_runner_hybrid_prefix_restore()`.

Any failure triggers all-rank rollback plus central lease/KV release. If any
cleanup operation is uncertain or fails, the coordinator is poisoned.

### 4. Publication

Only after all precommit checks succeed:

```python
block_manager.attach_sequence_reservation(reservation, sequence)
sequence.hybrid_state_slot_id = lease.slot_id
sequence.hybrid_state_generation = lease.generation
```

This is the publication boundary. The ticket state becomes `published`.

### 5. All-rank commit

Call:

```python
engine.commit_model_runner_hybrid_prefix_restore(
    payload,
    timeout_s=timeout_s,
)
```

On success the ticket becomes `committed` and `acquire()` returns `True`.

Any exception or uncertain result after publication poisons the coordinator
and the Engine acknowledgement channel. The coordinator must not detach or
reuse the now-published allocator/KV resources because rank commit state is
unknown.

## Ticket States

The live ticket uses:

```text
reserved
prepared
published
committed
rolled_back
rollback_failed
commit_failed
```

Only `reserved` and `prepared` are privately rollbackable. `published` is the
irreversible control-plane boundary for this phase.

The coordinator retains the last ticket for dependency-light inspection and
failure evidence. This is not a public scheduling API.

## Poison Semantics

The coordinator is fail-stop after:

- any failed or uncertain all-rank rollback;
- central allocator or KV release failure;
- any exception after publication;
- malformed or transport-poisoning nested rank results.

Future `acquire()` calls fail before resource reservation.

Before publication, the original prepare/validate error may be re-raised only
after successful cleanup. Valid all-rank prepare `miss`/`error` rows produce a
clean `False`.

After publication, `commit_failed` is always raised as a runtime error and no
performance path may continue.

## LLMEngine Surface

Add:

```python
self.qwen35_hybrid_prefix_engine_restore_coordinator = None
```

Explicit methods:

```python
install_qwen35_hybrid_prefix_engine_restore_coordinator(coordinator)
acquire_qwen35_hybrid_prefix(
    sequence,
    key,
    token_ids,
) -> bool
```

The acquisition method fails closed when no coordinator is installed and only
delegates to the coordinator. `LLMEngine.step()` remains unchanged.

## Correctness Tests

Create:

```text
tools/test_engine_live_hybrid_prefix_restore_transaction.py
```

Dependency-light CPU tests cover:

1. constructor and request identity validation;
2. full Sequence reservation and central lease remain private during prepare;
3. exact prefix miss releases KV and does not allocate a lease;
4. all-rank prepared flow validates, publishes, and commits in order;
5. valid rank miss/error broadcasts idempotent rollback and releases all
   private resources;
6. prepare transport failure attempts rollback and poisons only when cleanup is
   uncertain;
7. stale reservation, dirty Sequence, or stale allocator lease before
   publication rolls back and releases resources;
8. rollback failure enters `rollback_failed`, poisons, and blocks reuse;
9. publication happens only after all-rank validation;
10. commit failure after publication enters `commit_failed`, poisons, and
    preserves published Sequence ownership;
11. Engine installation identity and one-shot rules;
12. Engine acquisition fails closed when uninstalled and delegates when
    installed;
13. Scheduler guard and `LLMEngine.step()` remain unchanged.

Existing restore-ticket tests are updated for exact rolled-back idempotency and
committed rollback rejection.

## Acceptance Gate

Complete only when:

- focused tests demonstrate RED then GREEN;
- restore ticket, ModelRunner methods, command ack, and live wiring regressions
  pass under Python 3.9 and 3.12 where applicable;
- Qwen3.5/hybrid CPU regression scripts pass;
- chunked-prefill function matrix remains 97 pass / 1 known skip / 0 fail;
- Python 3.9/3.12 `py_compile` and `git diff --check` pass;
- staged files remain empty and untracked experiment evidence remains present;
- handoff records that Scheduler admission and automatic runtime owner
  construction are still blocked.

Allowed conclusion:

> TinyLLMForge has an explicit Engine control-plane transaction that privately
> reserves complete KV and hybrid-state ownership, prepares and validates
> installed rank-local restore participants, publishes Sequence metadata only
> after all-rank validation, and fails stop on uncertain cleanup or commit.

Not established:

- automatic Qwen3.5 owner construction;
- Scheduler admission or `LLMEngine.step()` integration;
- CUDA/NCCL or checkpoint logits/token correctness;
- latency, throughput, cache, memory, compression, or quality improvement.

