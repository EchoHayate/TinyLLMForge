# Speculative KV Offload Residency Transaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate generic speculative KV transactions with real `KVOffloadMVP0` residency using generation-aware rank-local prepare/precommit/rollback/seal tickets, then prove TP1 exact-greedy correctness with real movement counters.

**Architecture:** The engine remains the transaction coordinator and allocator authority. Each ModelRunner rank owns a `SpeculativeResidencyParticipant` that stages logical blocks through its local `KVOffloadMVP0`, keeps verifier writes private until allocator and Scheduler publication succeed, and discards rejected reserved mappings without D2H. Allocator generations are propagated through every ordinary MVP-0 access so speculative original blocks can be validated rather than adopted by assumption.

**Tech Stack:** Python 3.12, PyTorch/CUDA, multiprocessing shared-memory ModelRunner transport, pytest dependency-light tests, direct Torch scripts, remote TP1 execution over `sitian@10.232.195.203`.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, or run `git clean`.
- Generic runtime/core/scheduler/verifier code must not branch on model names or proposal-source behavior.
- Accepted KV commits in place; rejected suffix rolls back without replay, accepted KV copy, or per-token rematerialization.
- Variable-Q verification remains grouped by distinct fixed `Q`; do not pad groups.
- Non-KV recurrent or convolution state without transaction semantics remains fail closed.
- Non-zero temperature remains suppressed or fail closed.
- Real H2D/D2H evidence comes only from `KVOffloadMVP0` counters.
- Do not claim TPOT, TTFT, throughput, memory, long-context, TP4, second-model, learned-drafter, MTP, or promotion gains.
- Remote GPU validation uses `sitian@10.232.195.203`.
- Use short-lived shell commands; do not create persistent PTYs.
- Local Torch scripts use `/opt/homebrew/bin/python3.12`; run them directly because that interpreter has no pytest.
- Final classification remains `NOT_PROMOTABLE`.

---

## File Map

- Modify `tinyvllm/engine/block_manager.py`: allocator generation snapshots and exact identity lookup.
- Modify `tinyvllm/speculative/batch_runtime.py`: carry allocator-owned original/reserved identities into verifier tail items.
- Create `tinyvllm/engine/speculative_residency.py`: dependency-light payloads, acknowledgements, ticket state machine, and rank-local participant.
- Modify `tinyvllm/engine/model_runner.py`: generation-aware `KVOffloadMVP0`, ordinary identity binding, residency RPCs, and offload-enabled verifier mapping.
- Modify `tinyvllm/engine/speculative_model_runner.py`: pass ordinary identity rows and residency ticket IDs through existing callback helpers.
- Modify `tinyvllm/engine/llm_engine.py`: build identity rows, dispatch acknowledged residency RPCs, enforce publication order, and poison on post-commit failure.
- Modify `tools/test_speculative_kv_transaction.py`: allocator original-generation regression coverage.
- Modify `tools/test_kv_offload.py`: generation binding, discard atomicity, and real-counter metadata tests.
- Create `tools/test_speculative_residency.py`: dependency-light state-machine tests.
- Create `tools/test_model_runner_speculative_residency.py`: ModelRunner offload verifier and RPC contract tests.
- Modify `tools/test_engine_speculative_runtime.py`: all-rank prepare/precommit/rollback/seal ordering and poison tests.
- Modify `tools/test_speculative_model_runner_callbacks.py`: callback argument and fixed-Q ticket propagation tests.
- Modify `tools/speculative_tp1_parity_gate.py`: offload-enabled artifact schema and movement evidence.
- Modify `tools/verify_speculative_tp1_parity_gate.py`: independent schema and source-hash verification.
- Modify `tools/run_speculative_tp1_parity_gate_remote.sh`: invoke the offload gate and preserve failure artifacts.
- Modify `tools/test_speculative_tp1_parity_gate.py`: schema, zero rejected-D2H, and divergence artifact tests.
- Modify `docs/superpowers/audits/2026-08-12-generic-inference-optimization-goal-audit.md`: evidence and limitations.
- Modify `AGENT_HANDOFF_STATE.md`: authoritative continuation state.

---

### Task 1: Make Allocator Block Identities Exact

**Files:**
- Modify: `tinyvllm/engine/block_manager.py`
- Modify: `tools/test_speculative_kv_transaction.py`

**Interfaces:**
- Produces: `BlockManager.block_identities(block_ids: tuple[int, ...]) -> tuple[tuple[int, int], ...]`
- Produces: `SpeculativeKVTransaction.original_block_generations: tuple[int, ...]`
- Consumes later: ordinary offload identity rows and speculative residency prepare rows.

- [ ] **Step 1: Add failing allocator identity tests**

Add tests that require exact order, duplicate rejection, stale original-generation rejection, and rollback preservation:

```python
def test_block_identities_returns_allocator_generations_in_order():
    manager, seq = _allocated_sequence(block_size=2, token_ids=[1, 2, 3])

    identities = manager.block_identities(tuple(seq.block_table))

    assert identities == tuple(
        (block_id, manager.blocks[block_id].generation)
        for block_id in seq.block_table
    )


def test_speculative_transaction_rejects_stale_original_generation():
    manager, seq = _allocated_sequence(block_size=2, token_ids=[1, 2, 3])
    transaction = manager.begin_speculative_kv_transaction(
        seq,
        proposed_token_count=2,
    )
    original_block = seq.block_table[0]
    manager.blocks[original_block].generation += 1

    with pytest.raises(
        RuntimeError,
        match="original block ownership is stale",
    ):
        manager.mark_speculative_kv_materialized(transaction, 1)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python3 -m pytest \
  tools/test_speculative_kv_transaction.py \
  -k 'block_identities or original_generation' -q
```

Expected: failures because `block_identities()` and `original_block_generations` do not exist.

- [ ] **Step 3: Implement exact identity lookup and transaction snapshots**

Extend the dataclass:

```python
@dataclass
class SpeculativeKVTransaction:
    sequence_id: int
    original_num_tokens: int
    original_last_token: int
    original_block_table: tuple[int, ...]
    original_block_generations: tuple[int, ...]
    reserved_block_ids: tuple[int, ...]
    reserved_block_generations: tuple[int, ...]
    proposed_token_count: int
    materialized_token_count: int = 0
    state: str = "reserved"
```

Add the allocator API:

```python
def block_identities(
    self,
    block_ids: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    if not isinstance(block_ids, tuple):
        raise ValueError("block_ids must be a tuple")
    if len(set(block_ids)) != len(block_ids):
        raise ValueError("block_ids must be unique")
    identities = []
    for block_id in block_ids:
        if (
            isinstance(block_id, bool)
            or not isinstance(block_id, int)
            or block_id < 0
            or block_id >= len(self.blocks)
        ):
            raise ValueError("block id is out of range")
        block = self.blocks[block_id]
        if block_id not in self.used_block_ids or block.ref_count <= 0:
            raise RuntimeError("block identity ownership is stale")
        identities.append((block_id, block.generation))
    return tuple(identities)
```

Snapshot original generations in `begin_speculative_kv_transaction()` and validate tuple length, values, and exact current generations in `_validate_speculative_original_blocks()`.

- [ ] **Step 4: Run allocator transaction tests and verify GREEN**

Run:

```bash
python3 -m pytest tools/test_speculative_kv_transaction.py -q
```

Expected: `28+ passed`, with no existing transaction regression.

- [ ] **Step 5: Check the focused diff**

Run:

```bash
git diff --check -- \
  tinyvllm/engine/block_manager.py \
  tools/test_speculative_kv_transaction.py
```

Expected: no output.

---

### Task 2: Add Generation-Aware KVOffload Metadata

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_kv_offload.py`

**Interfaces:**
- Consumes: allocator `(block_id, generation)` identities.
- Produces: `KVOffloadMVP0.bind_logical_block_identity(logical_block, generation) -> None`
- Produces: `KVOffloadMVP0.discard_resident_blocks(block_identities, allow_dirty=False) -> tuple[tuple[int, int], ...]`
- Produces: `KVOffloadMVP0.speculative_residency_summary() -> dict`

- [ ] **Step 1: Add failing generation and discard tests**

Add a helper with complete metadata:

```python
def _identity_manager():
    manager = _NoopKVOffload()
    manager.bound_generations = [None] * manager.logical_blocks
    manager.h2d_done = {}
    manager.d2h_done = {}
    manager.stats.update({
        "speculative_residency_prepares": 0,
        "speculative_residency_precommits": 0,
        "speculative_residency_seals": 0,
        "speculative_residency_rollbacks": 0,
        "speculative_residency_committed_blocks": 0,
        "speculative_residency_rejected_blocks": 0,
        "speculative_residency_rejected_d2h_copies": 0,
    })
    return manager
```

Cover these exact cases:

```python
def test_newer_generation_clears_stale_owner_without_copy():
    manager = _identity_manager()
    manager.bound_generations[1] = 3
    manager.logical_to_slot[1] = 0
    manager.slot_to_logical[0] = 1
    manager.cpu_valid[1] = True
    manager.dirty_logical_blocks.add(1)
    manager.pending_wait_blocks.add(1)
    manager.h2d_done[1] = object()
    manager.d2h_done[1] = object()

    manager.bind_logical_block_identity(1, 4)

    assert manager.bound_generations[1] == 4
    assert 1 not in manager.logical_to_slot
    assert manager.slot_to_logical[0] is None
    assert manager.cpu_valid[1] is False
    assert 1 not in manager.dirty_logical_blocks
    assert 1 not in manager.pending_wait_blocks
    assert 1 not in manager.h2d_done
    assert 1 not in manager.d2h_done
    assert manager.enqueue_d2h_calls == 0
    assert manager.enqueue_h2d_calls == 0


def test_discard_resident_blocks_rejects_generation_mismatch_before_mutation():
    manager = _identity_manager()
    manager.bound_generations[1] = 7
    manager.logical_to_slot[1] = 0
    manager.slot_to_logical[0] = 1

    with pytest.raises(RuntimeError, match="generation mismatch"):
        manager.discard_resident_blocks(((1, 6),), allow_dirty=False)

    assert manager.logical_to_slot == {1: 0}
    assert manager.slot_to_logical == [1, None]
```

Also test same-generation idempotence, older-generation rejection, unbound-with-bytes rejection, dirty rejection, input prevalidation, no D2H/H2D, and injected mutation rollback.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/opt/homebrew/bin/python3.12 tools/test_kv_offload.py
```

Expected: failure at the first new method or metadata assertion.

- [ ] **Step 3: Implement generation metadata and atomic discard**

Initialize:

```python
self.bound_generations: list[int | None] = [
    None
] * self.logical_blocks
```

Add one metadata clearing primitive:

```python
def _clear_logical_block_metadata(
    self,
    logical_block: int,
) -> None:
    slot = self.logical_to_slot.pop(logical_block, None)
    if slot is not None:
        if self.slot_to_logical[slot] != logical_block:
            raise RuntimeError("KV offload mapping is inconsistent")
        self.slot_to_logical[slot] = None
    self.cpu_valid[logical_block] = False
    self.dirty_logical_blocks.discard(logical_block)
    self.pending_wait_blocks.discard(logical_block)
    self.h2d_done.pop(logical_block, None)
    self.d2h_done.pop(logical_block, None)
```

Implement binding:

```python
def bind_logical_block_identity(
    self,
    logical_block: int,
    generation: int,
) -> None:
    self._check_logical_block(logical_block)
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise ValueError("KV offload generation must be non-negative")
    current = self.bound_generations[logical_block]
    if current == generation:
        return
    if current is not None and generation < current:
        raise RuntimeError("KV offload generation moved backwards")
    if current is None and (
        logical_block in self.logical_to_slot
        or self.cpu_valid[logical_block]
        or logical_block in self.dirty_logical_blocks
        or logical_block in self.pending_wait_blocks
        or logical_block in self.h2d_done
        or logical_block in self.d2h_done
    ):
        raise RuntimeError(
            "cannot bind an unowned KV offload block with existing state"
        )
    if current is not None:
        self._clear_logical_block_metadata(logical_block)
    self.bound_generations[logical_block] = generation
```

Implement `discard_resident_blocks()` with complete prevalidation and snapshots of both mapping directions, validity, dirty/pending sets, event dictionaries, and generations. Use a private `_discard_validated_resident_block()` helper so the test can inject one mid-batch failure and prove restoration.

- [ ] **Step 4: Expose only real counters**

Extend `stats` with the seven speculative residency counters from the spec. `speculative_residency_summary()` must return those counters plus current active ticket-independent metadata counts; it must not derive or synthesize H2D/D2H values.

- [ ] **Step 5: Run the direct Torch suite and verify GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 tools/test_kv_offload.py
```

Expected: all tests pass.

---

### Task 3: Propagate Ordinary MVP-0 Block Identities

**Files:**
- Create: `tinyvllm/engine/speculative_residency.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/speculative_model_runner.py`
- Modify: `tools/test_engine_speculative_runtime.py`
- Modify: `tools/test_speculative_model_runner_callbacks.py`
- Create: `tools/test_model_runner_speculative_residency.py`

**Interfaces:**
- Produces: `KVBlockIdentityRow(sequence_id: int, block_identities: tuple[tuple[int, int], ...])`
- Produces: `build_kv_block_identity_rows(block_manager, seqs) -> tuple[KVBlockIdentityRow, ...]`
- Produces: `ModelRunner.bind_kv_block_identity_rows(seqs, rows) -> None`
- Changes: `ModelRunner.run(..., kv_block_identity_rows=())`
- Changes: `run_spec_first_target_batch(..., kv_block_identity_rows=())`

- [ ] **Step 1: Write failing dependency-light identity-row tests**

The new module starts with:

```python
@dataclass(frozen=True)
class KVBlockIdentityRow:
    sequence_id: int
    block_identities: tuple[tuple[int, int], ...]
```

Add tests that require exact sequence order, exact block-table order, unique sequence IDs, unique block IDs per row, and allocator lookup:

```python
def test_build_identity_rows_uses_allocator_generation_order():
    rows = build_kv_block_identity_rows(
        block_manager,
        (sequence_a, sequence_b),
    )
    assert rows == (
        KVBlockIdentityRow(7, ((1, 4), (3, 2))),
        KVBlockIdentityRow(9, ((5, 8),)),
    )
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python3 -m pytest \
  tools/test_model_runner_speculative_residency.py \
  tools/test_speculative_model_runner_callbacks.py \
  -q
```

Expected: import/signature failures.

- [ ] **Step 3: Implement engine-side identity construction**

Use only `BlockManager.block_identities()`:

```python
def build_kv_block_identity_rows(
    block_manager,
    seqs: tuple[object, ...],
) -> tuple[KVBlockIdentityRow, ...]:
    rows = []
    for seq in seqs:
        rows.append(
            KVBlockIdentityRow(
                sequence_id=int(seq.seq_id),
                block_identities=block_manager.block_identities(
                    tuple(seq.block_table)
                ),
            )
        )
    return tuple(rows)
```

Add `LLMEngine._kv_offload_identity_rows(seqs)` that returns `()` when MVP-0 is disabled and otherwise calls this helper.

- [ ] **Step 4: Bind identities before every ordinary offload access**

Add ModelRunner validation:

```python
def bind_kv_block_identity_rows(
    self,
    seqs: tuple[Sequence, ...],
    rows: tuple[KVBlockIdentityRow, ...],
) -> None:
    if self.kv_offload is None:
        if rows:
            raise RuntimeError(
                "KV block identities require kv_offload_mvp0"
            )
        return
    if len(rows) != len(seqs):
        raise ValueError("KV block identity row count mismatch")
    for seq, row in zip(seqs, rows):
        if row.sequence_id != seq.seq_id:
            raise ValueError("KV block identity sequence mismatch")
        if tuple(block for block, _ in row.block_identities) != tuple(
            seq.block_table
        ):
            raise ValueError("KV block identity table mismatch")
        for block_id, generation in row.block_identities:
            self.kv_offload.bind_logical_block_identity(
                block_id,
                generation,
            )
```

Call it before `_run_model_step()` and before `prepare_decode()` in `run_spec_first_target_batch()`. Pass the rows through `LLMEngine.step()` for ordinary, suppressed, and first-target execution. Preserve the existing `released_hybrid_state_leases` argument order by adding identity rows as the final optional parameter.

- [ ] **Step 5: Prove fail-closed behavior**

Tests must show:

- offload-enabled `run()` rejects missing rows before staging;
- non-offload `run()` remains unchanged with empty rows;
- stale generation fails before model forward;
- first-target callbacks preserve row order and hidden/logit options;
- the shared-memory payload remains pickleable.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
python3 -m pytest \
  tools/test_model_runner_speculative_residency.py \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_engine_speculative_runtime.py \
  -k 'identity or first_target or ordinary' -q
```

Expected: all selected tests pass.

---

### Task 4: Implement the Rank-Local Residency Ticket State Machine

**Files:**
- Modify: `tinyvllm/engine/speculative_residency.py`
- Create: `tools/test_speculative_residency.py`

**Interfaces:**
- Produces: `SpeculativeResidencyPrepareRow`
- Produces: `SpeculativeResidencyPrecommitRow`
- Produces: `SpeculativeResidencyResult`
- Produces: `SpeculativeResidencyParticipant.prepare_batch()`
- Produces: `SpeculativeResidencyParticipant.mark_materialized()`
- Produces: `SpeculativeResidencyParticipant.precommit_batch()`
- Produces: `SpeculativeResidencyParticipant.rollback_batch()`
- Produces: `SpeculativeResidencyParticipant.seal_batch()`

- [ ] **Step 1: Define exact payloads in failing tests**

Use immutable payloads:

```python
@dataclass(frozen=True)
class SpeculativeResidencyPrepareRow:
    sequence_id: int
    original_block_identities: tuple[tuple[int, int], ...]
    reserved_block_identities: tuple[tuple[int, int], ...]
    proxy_block_table: tuple[int, ...]
    logical_slots: tuple[int, ...]


@dataclass(frozen=True)
class SpeculativeResidencyPrecommitRow:
    sequence_id: int
    committed_block_identities: tuple[tuple[int, int], ...]
    rejected_block_identities: tuple[tuple[int, int], ...]
    accepted_materialized_end: int


@dataclass(frozen=True)
class SpeculativeResidencyResult:
    ticket_id: int
    participant_id: int
    operation: str
    status: str
    sequence_ids: tuple[int, ...]
    committed_block_identities: tuple[tuple[int, int], ...] = ()
    rejected_block_identities: tuple[tuple[int, int], ...] = ()
    detail: str = ""
```

Tests cover `prepared -> precommitted -> sealed`, prepared/precommitted rollback, repeated operation rejection, zero/partial/full acceptance, original-only writes, and multiple reserved blocks.

- [ ] **Step 2: Run the state-machine suite and verify RED**

Run:

```bash
python3 -m pytest tools/test_speculative_residency.py -q
```

Expected: import failures.

- [ ] **Step 3: Implement prepare classification and staging**

`prepare_batch(ticket_id, rows)` must:

1. validate all rows before mutation;
2. require unique sequence IDs and disjoint reserved identities;
3. require proxy prefix IDs to equal original IDs;
4. require proxy suffix IDs to equal reserved IDs;
5. derive materialized block IDs from `logical_slots`;
6. bind original identities as exact matches;
7. bind reserved identities;
8. stage original visible blocks with `require_valid=True`;
9. stage reserved materialized blocks with `require_valid=False`;
10. protect the union during staging;
11. on failure, discard only reserved mappings assigned by this prepare.

The participant stores a private mutable ticket but returns only `SpeculativeResidencyResult`.

- [ ] **Step 4: Implement materialize, precommit, rollback, and seal**

`mark_materialized(ticket_id, sequence_ids)` records successful verifier ownership without dirty publication.

`precommit_batch()` requires an exact committed/rejected partition of every reserved identity and validates `accepted_materialized_end`.

`rollback_batch()` calls:

```python
self.manager.discard_resident_blocks(
    ticket.reserved_block_identities,
    allow_dirty=False,
)
```

`seal_batch()`:

- marks accepted original and committed-reserved materialized blocks dirty;
- immediately writes accepted dirty blocks only when existing policy requires it;
- discards rejected identities with `allow_dirty=False`;
- increments counters once;
- never stages, copies, or recomputes accepted KV.

- [ ] **Step 5: Add injected failure tests**

Cover prepare failure after one mapping, unreadable history, insufficient slots, precommit partition mismatch, discard failure, and ensure rollback/seal failures leave the ticket in an explicit failed state rather than reusable.

- [ ] **Step 6: Run the state-machine suite and verify GREEN**

Run:

```bash
python3 -m pytest tools/test_speculative_residency.py -q
```

Expected: all tests pass.

---

### Task 5: Enable Offload-Aware ModelRunner Verification

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/speculative_model_runner.py`
- Modify: `tools/test_model_runner_speculative_residency.py`
- Modify: `tools/test_model_runner_batch_spec_verify_source.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**
- Produces ModelRunner RPCs:
  - `prepare_speculative_residency_batch(ticket_id, rows)`
  - `precommit_speculative_residency_batch(ticket_id, rows)`
  - `rollback_speculative_residency_batch(ticket_id)`
  - `seal_speculative_residency_batch(ticket_id)`
- Changes: `run_spec_verify_batch(items, residency_ticket_id=None)`
- Changes: `run_model_runner_tail_batch(model_runner, items, residency_ticket_id=None)`

- [ ] **Step 1: Add failing RPC and verifier tests**

Tests require:

```python
result = runner.prepare_speculative_residency_batch(
    41,
    (prepare_row,),
)
assert result == {
    "ticket_id": 41,
    "participant_id": 0,
    "operation": "prepare",
    "status": "prepared",
    "sequence_ids": (7,),
    "committed_block_identities": (),
    "rejected_block_identities": (),
    "detail": "",
}
```

Also require:

- MVP-0 verifier without a ticket fails before forward;
- prepared ticket maps logical proxy blocks to physical blocks;
- physical slots derive from `logical_to_slot`;
- successful forward marks the ticket materialized;
- forward exception leaves the ticket rollbackable;
- fixed-Q groups reuse the same ticket ID without padding.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python3 -m pytest \
  tools/test_model_runner_speculative_residency.py \
  tools/test_model_runner_batch_spec_verify_source.py \
  tools/test_model_runner_spec_verify.py \
  -q
```

Expected: missing RPC/signature failures.

- [ ] **Step 3: Wire the participant into ModelRunner**

After KV cache allocation:

```python
self.speculative_residency = (
    SpeculativeResidencyParticipant(
        participant_id=self.rank,
        manager=self.kv_offload,
        block_size=self.block_size,
    )
    if self.kv_offload is not None
    else None
)
```

Each public RPC validates the participant exists, delegates, and converts the immutable result to an exact dict with no extra fields.

- [ ] **Step 4: Translate verifier metadata through the ticket**

Change compatibility validation so MVP-0 is allowed only when:

```python
residency_ticket_id is not None
and self.speculative_residency is not None
and self.speculative_residency.is_prepared_for(
    residency_ticket_id,
    sequence_ids,
)
```

Validate logical slots against logical proxy rows first. Then use the participant's manager mapping to create physical block-table rows and physical slot mappings. Do not call `_kv_offload_after_forward()`.

After `run_model()` succeeds:

```python
self.speculative_residency.mark_materialized(
    residency_ticket_id,
    tuple(row.sequence_id for row in metadata.rows),
)
```

- [ ] **Step 5: Preserve non-offload verifier behavior**

When MVP-0 is disabled, `residency_ticket_id` must be `None`, and the current direct-block behavior remains byte-for-byte equivalent.

- [ ] **Step 6: Run ModelRunner tests and verify GREEN**

Run:

```bash
python3 -m pytest \
  tools/test_model_runner_speculative_residency.py \
  tools/test_model_runner_batch_spec_verify_source.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_speculative_model_runner_callbacks.py \
  -q
```

Expected: all tests pass.

---

### Task 6: Coordinate the Engine Two-Phase Residency Transaction

**Files:**
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tinyvllm/engine/speculative_execution.py`
- Modify: `tinyvllm/engine/speculative_model_runner.py`
- Modify: `tinyvllm/speculative/batch_runtime.py`
- Modify: `tools/test_engine_speculative_execution.py`
- Modify: `tools/test_engine_speculative_runtime.py`

**Interfaces:**
- Produces: generation-aware `TailBatchItem.original_block_identities`
- Produces: generation-aware `TailBatchItem.reserved_block_identities`
- Produces: exact precommit projections from `SpeculativeKVCommitPlan`
- Produces: engine methods for acknowledged prepare/precommit/rollback/seal.

- [ ] **Step 1: Add failing tail-item identity tests**

Extend:

```python
@dataclass(frozen=True)
class TailBatchItem:
    sequence_id: int
    plan: SpecVerifyPlan
    proxy_block_table: tuple[int, ...]
    original_block_identities: tuple[tuple[int, int], ...]
    reserved_block_identities: tuple[tuple[int, int], ...]
```

The batch runtime constructs identities only from transaction snapshots:

```python
original_block_identities=tuple(zip(
    transaction.original_block_table,
    transaction.original_block_generations,
)),
reserved_block_identities=tuple(zip(
    transaction.reserved_block_ids,
    transaction.reserved_block_generations,
)),
```

Tests reject mismatched tuple lengths and proxy suffixes.

- [ ] **Step 2: Add failing engine ordering tests**

Extend `_run_selected_step_with_transaction()` with recorded residency RPCs and require:

```text
residency_prepare
verifier
kv_prepare
residency_precommit
kv_commit
scheduler_commit
residency_seal
lifecycle_sync
```

Failure cases:

- verifier failure -> residency rollback + allocator rollback;
- one-rank precommit failure -> all-rank rollback, no allocator commit;
- allocator commit failure -> residency rollback;
- Scheduler commit failure -> Scheduler snapshot restore + residency rollback;
- rollback failure -> runtime poisoned;
- seal failure after Scheduler commit -> target tokens stay committed and runtime poisoned.

- [ ] **Step 3: Run ordering tests and verify RED**

Run:

```bash
python3 -m pytest \
  tools/test_engine_speculative_execution.py \
  tools/test_engine_speculative_runtime.py \
  -k 'residency or publication_order or seal or precommit' -q
```

Expected: missing engine coordination behavior.

- [ ] **Step 4: Add exact acknowledged-result validation**

Use `call_model_runner_acknowledged()` for every residency mutation. Validate local rank plus worker acknowledgements against:

```python
required = {
    "ticket_id",
    "participant_id",
    "operation",
    "status",
    "sequence_ids",
    "committed_block_identities",
    "rejected_block_identities",
    "detail",
}
```

Require participant IDs `0..world_size-1`, exact ticket/operation/status, exact sequence order, and exact committed/rejected identities. Poison the acknowledgement collector on malformed nested results.

- [ ] **Step 5: Integrate prepare around the tail callback**

Allocate the ticket ID in the engine and keep it in the selected-step scope:

```python
residency_ticket_id = next(
    self._speculative_residency_ticket_ids
)
```

The `run_tail_batch` closure:

1. builds `SpeculativeResidencyPrepareRow` values from `TailBatchItem`;
2. acknowledged-prepare on all ranks;
3. calls `run_model_runner_tail_batch(..., residency_ticket_id)`;
4. leaves the ticket prepared/materialized for acceptance.

If no tail items exist, do not create a ticket.

- [ ] **Step 6: Integrate precommit and publication order**

After `kv_plans` are prepared, convert plan IDs to identities by exact lookup in each transaction snapshot. Dispatch acknowledged precommit before:

```python
self.scheduler.block_manager.commit_speculative_kv_commit_batch(
    kv_plans
)
self.scheduler.commit_prepared_postprocess(
    prepared_scheduler
)
```

Dispatch acknowledged seal only after both calls succeed. Then set `prepared_runtime.state = "committed"` and synchronize optional draft lifecycle.

- [ ] **Step 7: Implement phase-aware rollback and poison rules**

Before allocator commit, any exception rolls back residency first and allocator reservations second. Scheduler commit exceptions rely on the existing Scheduler snapshot restore, then roll back residency and native speculative reservations. A residency rollback failure sets:

```python
self.speculative_runtime_poisoned = True
self.speculative_runtime_poison_reason = (
    "speculative residency rollback failed: "
    f"{error}"
)
```

A seal failure uses `"speculative residency seal failed: ..."`, does not call allocator rollback, and leaves target tokens authoritative.

- [ ] **Step 8: Run engine tests and verify GREEN**

Run:

```bash
python3 -m pytest \
  tools/test_engine_speculative_execution.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_speculative_model_runner_callbacks.py \
  -q
```

Expected: all tests pass.

---

### Task 7: Extend the Loaded-Model TP1 Offload Gate

**Files:**
- Modify: `tools/speculative_tp1_parity_gate.py`
- Modify: `tools/verify_speculative_tp1_parity_gate.py`
- Modify: `tools/run_speculative_tp1_parity_gate_remote.sh`
- Modify: `tools/test_speculative_tp1_parity_gate.py`

**Interfaces:**
- Produces artifact schema version `2`.
- Produces exact-token parity plus real `KVOffloadMVP0` summary fields.
- Preserves failure artifact creation before raising.

- [ ] **Step 1: Add failing artifact-schema tests**

Require:

```python
artifact["environment"]["kv_offload_mvp0"] is True
artifact["speculative"]["residency"]["speculative_residency_prepares"] > 0
artifact["speculative"]["residency"]["speculative_residency_precommits"] > 0
artifact["speculative"]["residency"]["speculative_residency_seals"] > 0
artifact["speculative"]["residency"][
    "speculative_residency_rejected_d2h_copies"
] == 0
```

Require real movement keys:

```python
REAL_MOVEMENT_KEYS = (
    "h2d_copies",
    "d2h_copies",
    "h2d_bytes",
    "d2h_bytes",
    "copy_waits",
    "evictions",
    "evict_clean",
    "evict_dirty",
)
```

Reject missing, negative, non-integer, or synthesized fields. Preserve the token-divergence FAIL artifact.

- [ ] **Step 2: Run artifact tests and verify RED**

Run:

```bash
python3 -m pytest tools/test_speculative_tp1_parity_gate.py -q
```

Expected: schema version and residency evidence failures.

- [ ] **Step 3: Enable MVP-0 in both baseline and speculative cases**

Construct both engines with:

```python
engine = engine_factory(
    model_path,
    tensor_parallel_size=1,
    enforce_eager=True,
    max_model_len=4096,
    max_num_seqs=max(4, len(prompts)),
    kv_offload_mvp0=True,
    kv_offload_gpu_blocks=8,
    kv_offload_logical_blocks=64,
)
```

Add `ModelRunner.kv_offload_summary()` and `LLMEngine.kv_offload_summaries(timeout_s=60.0)` using acknowledged rank collection. Capture the final rank-0 summary before `engine.exit()`.

- [ ] **Step 4: Build and independently validate schema version 2**

Add `tinyvllm/engine/speculative_residency.py` to `SOURCE_FILES`. Store baseline movement separately from speculative movement. The claim scope states TP1 BF16 greedy exact-token parity with MVP-0 transactional speculative residency and explicitly states correctness-only evidence.

- [ ] **Step 5: Update the remote runner**

Keep full `tinyvllm/` sync. Invoke the schema-v2 gate, always download `remote.log` and any `result.json`, and run the independent verifier remotely and locally.

- [ ] **Step 6: Run local artifact tests and verify GREEN**

Run:

```bash
python3 -m pytest tools/test_speculative_tp1_parity_gate.py -q
python3 -m py_compile \
  tools/speculative_tp1_parity_gate.py \
  tools/verify_speculative_tp1_parity_gate.py
```

Expected: all tests pass and compile succeeds.

- [ ] **Step 7: Run the real remote TP1 gate**

Run:

```bash
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)" \
LOCAL_OUT="artifacts/speculative_tp1_parity/${RUN_TAG}" \
bash tools/run_speculative_tp1_parity_gate_remote.sh
```

Expected:

- exact baseline/speculative output equality;
- residency prepare/precommit/seal counters are positive;
- accepted and rejected residency counts match transaction observations;
- rejected speculative D2H copies equal zero;
- independent source-hash verification passes;
- elapsed times are recorded but not interpreted as performance gains.

---

### Task 8: Run the Full Gate and Update Authoritative Evidence

**Files:**
- Modify: `docs/superpowers/audits/2026-08-12-generic-inference-optimization-goal-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: all local test results and the newest remote artifact.
- Produces: an explicit `NOT_PROMOTABLE` evidence boundary and next actions.

- [ ] **Step 1: Run dependency-light suites in isolated processes**

Run:

```bash
python3 -m pytest tools/test_speculative_kv_transaction.py -q
python3 -m pytest tools/test_speculative_residency.py -q
python3 -m pytest tools/test_engine_speculative_execution.py -q
python3 -m pytest tools/test_engine_speculative_runtime.py -q
python3 -m pytest tools/test_speculative_model_runner_callbacks.py -q
python3 -m pytest tools/test_model_runner_speculative_residency.py -q
python3 -m pytest tools/test_model_runner_batch_spec_verify_source.py -q
python3 -m pytest tools/test_model_runner_spec_verify.py -q
python3 -m pytest tools/test_speculative_tp1_parity_gate.py -q
```

Expected: all pass. Keep `tools/test_speculative_kv_transaction.py` isolated because it installs dependency-light module stubs.

- [ ] **Step 2: Run the direct Torch KVOffload suite**

Run:

```bash
/opt/homebrew/bin/python3.12 tools/test_kv_offload.py
```

Expected: all direct assertions pass.

- [ ] **Step 3: Run broad speculative regression coverage**

Run:

```bash
python3 -m pytest \
  tools/test_native_verifier_contract.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_model_runner_batch_spec_verify_source.py \
  tools/test_speculative_batch_runtime.py \
  tools/test_engine_speculative_execution.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_speculative_runtime.py \
  tools/test_ngram_speculative.py \
  -q
```

Expected: all pass.

- [ ] **Step 4: Run static validation**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/block_manager.py \
  tinyvllm/engine/speculative_residency.py \
  tinyvllm/engine/speculative_model_runner.py \
  tinyvllm/engine/speculative_execution.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tinyvllm/speculative/batch_runtime.py \
  tools/speculative_tp1_parity_gate.py \
  tools/verify_speculative_tp1_parity_gate.py
git diff --check
git status --short
```

Expected: compile and diff checks pass; no staged changes.

- [ ] **Step 5: Update the audit**

Record:

- exact local commands and counts;
- remote artifact path and independent verifier path;
- proposed/accepted counts;
- residency prepare/precommit/seal/rollback counts;
- real H2D/D2H bytes/copies/waits/evictions;
- rejected speculative D2H count;
- exact-token parity result;
- elapsed baseline/speculative times as observations only;
- all unproven promotion dimensions.

- [ ] **Step 6: Update the handoff**

Append a new authoritative continuation section containing:

- files changed;
- transaction publication order;
- failure/poison boundaries;
- latest green artifact;
- known limits;
- next recommended gate: controlled repeated offload movement and TPOT measurements only after correctness remains green.

- [ ] **Step 7: Final evidence check**

Run:

```bash
git diff --check
git diff --cached --quiet
```

Expected: no whitespace errors and an empty staged diff.

