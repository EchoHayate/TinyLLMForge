# Autoregressive Draft-Model Proposal Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## 2026-08-15 Evidence Reconciliation

This plan was reconciled against current source, the current test inventory,
fresh offline CPU Torch regression, static source-neutral checks, and the
durable handoff/audit:

```text
AUTOREGRESSIVE_DRAFT_EXECUTOR_PLAN_TOTAL_STEPS=34
AUTOREGRESSIVE_DRAFT_EXECUTOR_PLAN_CHECKED=26
AUTOREGRESSIVE_DRAFT_EXECUTOR_PLAN_INTENTIONALLY_OPEN=8

full local regression matrix:
  437 passed in 23.60s

interface/config/source-neutral assertions:
  PASS

production and gate py_compile:
  PASS
```

The eight open steps are the historical RED executions for Tasks 1-8. Their
expected initial failures are described by the plan, but no retained failure
transcripts were found. Current tests, implementations, GREEN runs, static
checks, and documentation are directly provable and are checked below.

Later Proposal-KV residency and TP4 work moved
`Qwen3DraftPhysicalSlotStore` from
`tinyvllm/engine/qwen3_draft_backend.py` into
`tinyvllm/engine/qwen3_draft_proposal_kv.py`; the backend still imports and
exports the symbol. This is a tested superseding refactor, not a missing Task
4 implementation.

The local implementation and harness are established, but the real Qwen3
draft plus Qwen3.5 target checkpoint gate has not run:

```text
AUTOREGRESSIVE_DRAFT_EXECUTOR_CONTRACT=ESTABLISHED_LOCAL
AUTOREGRESSIVE_DRAFT_TP1_GATE_HARNESS=ESTABLISHED_LOCAL
SECOND_LEARNED_STRUCTURE=NOT_ESTABLISHED
LEARNED_DRAFTER_LOADED_PARITY=NOT_ESTABLISHED
REAL_AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

**Goal:** Add a generic batch-native autoregressive learned-draft executor, a Qwen3 dense backend with isolated transactional proposal KV, source-neutral ModelRunner registration, and a TP1 real-checkpoint exact-greedy gate for a Qwen3 drafter against a Qwen3.5 target.

**Architecture:** Keep proposal orchestration and lifecycle ownership in `AutoregressiveDraftProposalExecutor`; inject a model-specific batch backend that owns forward contexts and logits. The first backend wraps `Qwen3ForCausalLM`, binds every attention layer to one block-size-one multi-layer proposal-KV store, and is registered only after checkpoint and tokenizer compatibility validation.

**Tech Stack:** Python dataclasses and protocols, PyTorch, existing `Qwen3ForCausalLM`, `ProposalKVCache`, `ProposalKVLifecycleCoordinator`, `ModelRunnerProposalExecutorRegistry`, Transformers tokenizer metadata, dependency-light pytest.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not create or switch worktrees or branches.
- Do not stage, commit, push, stash, reset, or clean.
- Do not use subagents.
- Do not run remote or GPU workloads without separate authorization.
- Preserve exact greedy parity and `MAX_PROPOSAL_TOKENS=4`.
- Keep target KV at 68 GPU blocks, 640 logical blocks, and block size 256 in later real gates; proposal KV remains GPU-resident.
- Diagnostics remain default-off and may not add a target forward.
- Do not change verifier selection, fallback indexing, accepted-prefix semantics, target-KV transactions, recurrent side-state selection, or offload counters.
- Proposal token 0 is always `first_target_token`; learned decode produces only tokens `1..Q-1`.
- Proposal staged entry count is `Q - 1`; proposal commit count is `max(accepted_proposal_tokens - 1, 0)`.
- Qwen3 proposal KV uses `block_size=1`; one proposal slot ID is also one paged-attention block ID across every draft layer.
- The first implementation is TP1 only. Enabled configuration with `tensor_parallel_size != 1` must fail before draft weights or slots are allocated.
- Token remapping, vocabulary projection, and same-size-only tokenizer compatibility are forbidden.
- Do not claim a real second learned structure until the real checkpoint gate passes.
- Do not claim TP4, 4K/16K/32K coverage, performance improvement, KV8/KV4, production readiness, or Phase 1 completion from local synthetic tests.
- Every production change follows RED -> GREEN with the focused failing test observed before implementation.

---

### Task 1: Generic Executor Contracts and Prefill Observation

**Files:**
- Create: `tinyvllm/engine/autoregressive_draft_executor.py`
- Create: `tools/test_autoregressive_draft_executor.py`

**Interfaces:**
- Consumes: `ProposalKVTransaction`, `ProposalKVCache`, `TargetPrefillObservation`, and `DraftCapabilities`.
- Produces: `AutoregressiveDraftPrefillRow`, `AutoregressiveDraftDecodeRow`, `AutoregressiveDraftBackend`, and the observation/capability surface of `AutoregressiveDraftProposalExecutor`.

- [x] **Step 1: Write failing contract and observation tests**

Create CPU-only fixtures:

```python
from types import SimpleNamespace

import pytest
import torch

from tinyvllm.engine.autoregressive_draft_executor import (
    AutoregressiveDraftProposalExecutor,
)
from tinyvllm.engine.proposal_kv_cache import ProposalKVCache
from tinyvllm.engine.speculative_proposal_executor import (
    TargetPrefillObservation,
)


class _PhysicalStore:
    def __init__(self):
        self.next_slot = 0

    def reserve_slots(self, count):
        slots = tuple(range(self.next_slot, self.next_slot + count))
        self.next_slot += count
        return slots

    def release_slots(self, slot_ids):
        assert isinstance(slot_ids, tuple)


class _Backend:
    device = torch.device("cpu")
    backend_identity = "fake-autoregressive"
    model_fingerprint = "model-sha256"
    tokenizer_fingerprint = "tokenizer-sha256"

    def prefill_batch(self, rows):
        raise AssertionError("prefill is not used by observation tests")

    def decode_step_batch(self, rows):
        raise AssertionError("decode is not used by observation tests")


def _executor():
    return AutoregressiveDraftProposalExecutor(
        backend=_Backend(),
        proposal_kv_cache=ProposalKVCache(_PhysicalStore()),
        max_proposal_tokens=4,
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
    )


def _row(token_ids, positions, *, final, epoch=7):
    return TargetPrefillObservation(
        sequence_id=11,
        sequence_epoch=epoch,
        token_ids=tuple(token_ids),
        positions=torch.tensor(positions, dtype=torch.int64),
        target_hidden=torch.full((len(token_ids), 3), 91.0),
        is_final_chunk=final,
    )


def test_capabilities_are_source_neutral_and_lifecycle_enabled():
    capabilities = _executor().capabilities
    assert capabilities.source_type == "independent_draft_model"
    assert capabilities.supports_batch is True
    assert capabilities.requires_target_hidden is False
    assert capabilities.requires_target_logits is False
    assert capabilities.max_proposal_tokens == 4
    assert capabilities.execution_domain == "model_runner"
    assert capabilities.requires_proposal_lifecycle is True
    assert capabilities.requires_full_token_history is False


def test_chunked_prefill_persists_only_tokens_positions_and_epoch():
    executor = _executor()
    executor.observe_target_prefill((_row((3, 4), (0, 1), final=False),))
    executor.observe_target_prefill((_row((5,), (2,), final=True),))

    pending = executor.pending_prompt(11)
    assert pending.sequence_epoch == 7
    assert pending.token_ids == (3, 4, 5)
    assert pending.positions == (0, 1, 2)
    assert pending.is_final is True
    assert not hasattr(pending, "target_hidden")


@pytest.mark.parametrize(
    ("first", "second", "message"),
    (
        (_row((3,), (1,), final=True), None, "start at position zero"),
        (
            _row((3,), (0,), final=False),
            _row((4,), (2,), final=True),
            "contiguous",
        ),
        (
            _row((3,), (0,), final=False),
            _row((4,), (1,), final=True, epoch=8),
            "epoch",
        ),
    ),
)
def test_invalid_prefill_chunks_fail_closed(first, second, message):
    executor = _executor()
    with pytest.raises((ValueError, RuntimeError), match=message):
        executor.observe_target_prefill((first,))
        if second is not None:
            executor.observe_target_prefill((second,))
```

Also add tests that reject duplicate sequence IDs in one observation batch,
non-integer token IDs, non-rank-one positions, non-integer position dtype,
row-count mismatch between tokens/positions/target hidden, a second final
chunk, and appending after a final chunk.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONPATH=$PWD /tmp/tinyllmforge-trace-test-venv/bin/python \
  -m pytest -q tools/test_autoregressive_draft_executor.py
```

Expected: collection fails because
`tinyvllm.engine.autoregressive_draft_executor` does not exist.

- [x] **Step 3: Implement contracts, constructor validation, and prompt accumulation**

Create these exact public definitions:

```python
@dataclass(frozen=True)
class AutoregressiveDraftPrefillRow:
    transaction: ProposalKVTransaction
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]


@dataclass(frozen=True)
class AutoregressiveDraftDecodeRow:
    transaction: ProposalKVTransaction
    step: int
    input_token_id: int
    position: int


@dataclass(frozen=True)
class AutoregressiveDraftPendingPrompt:
    sequence_id: int
    sequence_epoch: int
    token_ids: tuple[int, ...]
    positions: tuple[int, ...]
    is_final: bool


class AutoregressiveDraftBackend(Protocol):
    device: object
    backend_identity: str
    model_fingerprint: str
    tokenizer_fingerprint: str

    def prefill_batch(
        self,
        rows: tuple[AutoregressiveDraftPrefillRow, ...],
    ) -> None:
        pass

    def decode_step_batch(
        self,
        rows: tuple[AutoregressiveDraftDecodeRow, ...],
    ) -> tuple[object, ...]:
        pass
```

`AutoregressiveDraftProposalExecutor.__init__()` must:

- require callable `prefill_batch` and `decode_step_batch`;
- require non-empty string backend/model/tokenizer identities;
- require a real `ProposalKVCache`;
- require `1 <= max_proposal_tokens <= 4`;
- validate rank bounds;
- raise `RuntimeError("autoregressive draft executor currently requires TP1")`
  when `tensor_parallel_size != 1`;
- construct `ProposalKVLifecycleCoordinator` with namespace
  `"autoregressive-draft"`;
- publish the exact capabilities asserted by the test;
- initialize pending prompt, bootstrapped-sequence, proposal-Q, selected-token,
  bootstrap, and timing evidence containers.

`observe_target_prefill()` must validate the complete input tuple before
mutating pending state. Convert positions to immutable Python integer tuples,
check exact contiguous values, validate target-hidden row count, and never
retain the hidden tensor or read its values.

- [x] **Step 4: Run the focused test and verify GREEN**

Run the Task 1 command. Expected: all Task 1 tests pass.

---

### Task 2: Transactional Bootstrap and Batch-Native Proposal Execution

**Files:**
- Modify: `tinyvllm/engine/autoregressive_draft_executor.py`
- Modify: `tools/test_autoregressive_draft_executor.py`

**Interfaces:**
- Consumes: contracts from Task 1, `ModelRunnerProposalInput`, `DraftProposal`, `ProposalKVRegistration`, and `select_tensor_parallel_greedy_tokens`.
- Produces: `propose_batch()`, bootstrap state, coordinator-backed proposal registration, and exact-Q batch behavior.

- [x] **Step 1: Write failing bootstrap and proposal tests**

Replace the inert fake backend with a recording backend whose
`decode_step_batch()` returns deterministic finite logits:

```python
class _RecordingBackend:
    device = torch.device("cpu")
    backend_identity = "recording-backend"
    model_fingerprint = "model-sha256"
    tokenizer_fingerprint = "tokenizer-sha256"

    def __init__(self):
        self.prefill_calls = []
        self.decode_calls = []

    def prefill_batch(self, rows):
        self.prefill_calls.append(rows)

    def decode_step_batch(self, rows):
        self.decode_calls.append(rows)
        outputs = []
        for row in rows:
            logits = torch.full((8,), -100.0)
            logits[(row.input_token_id + 1) % 8] = 10.0
            outputs.append(logits)
        return tuple(outputs)
```

Add focused tests proving:

```python
def test_first_proposal_bootstraps_prompt_once_and_batches_decode():
    executor, backend, cache = ready_executor(
        prompts={
            1: ((2, 3), (0, 1)),
            2: ((4, 5, 6), (0, 1, 2)),
        }
    )
    proposals = executor.propose_batch((
        proposal_input(1, first_target_token=6, exact_q=4),
        proposal_input(2, first_target_token=2, exact_q=4),
    ))

    assert len(backend.prefill_calls) == 1
    assert tuple(row.token_ids for row in backend.prefill_calls[0]) == (
        (2, 3),
        (4, 5, 6),
    )
    assert [len(call) for call in backend.decode_calls] == [2, 2, 2]
    assert proposals[0].token_ids == (6, 7, 0, 1)
    assert proposals[1].token_ids == (2, 3, 4, 5)
    assert cache.committed_length(1) == 2
    assert cache.committed_length(2) == 3
```

Add tests for:

- mixed `Q=0,1,2,3,4` rows preserve input order;
- `Q=0` has no transaction;
- `Q=1` has a non-empty proposal and zero staged entries;
- staged entry counts are exactly `Q-1`;
- proposal token 0 exactly equals `first_target_token`;
- repeated proposals do not replay prompt bootstrap;
- malformed logit count, rank, dtype, device, or non-finite values abort every
  still-owned new transaction in reverse order;
- injected prefill failure leaves zero owned slots, immutable pending prompts,
  and retryable state;
- bootstrap retry is rejected if any committed prompt slot, active
  transaction, or active ticket exists;
- registration failure cleans the entire new proposal batch but preserves
  transactions from an earlier successful batch.

- [ ] **Step 2: Run the new tests and verify RED**

Run the Task 1 command.

Expected: failures for missing `propose_batch()`, bootstrap, and proposal
registration behavior.

- [x] **Step 3: Implement batch bootstrap and proposal generation**

Implement private helpers with these exact responsibilities:

```python
def _bootstrap_sequences(
    self,
    sequence_ids: tuple[int, ...],
) -> None:
    pass


def _validate_logit_rows(
    self,
    rows: object,
    *,
    expected_count: int,
) -> tuple[torch.Tensor, ...]:
    pass


def _proposal_exact_q(
    self,
    input_row: ModelRunnerProposalInput,
) -> int:
    pass
```

The bootstrap algorithm is fixed:

1. Validate every requested sequence has one complete final prompt and no
   stale bootstrapped epoch.
2. Validate all retry preconditions before reserving any slot.
3. Begin one `prompt_token_count` transaction per unbootstrapped sequence.
4. Call `backend.prefill_batch()` once with rows in input order.
5. Mark every transaction fully materialized.
6. Prepare each underlying ticket with
   `accepted_proposal_tokens=prompt_token_count + 1`.
7. Commit tickets in input order and record that this count is bootstrap
   encoding, not verifier acceptance.
8. On any pre-commit failure, rollback prepared tickets and abort
   unprepared transactions in reverse order.
9. Treat commit failure as terminal by raising without attempting synthetic
   partial recovery.

The proposal algorithm is fixed:

1. Validate all inputs and compute every `exact_q` before bootstrap or slot
   allocation.
2. Return empty proposals directly for `exact_q == 0`; these rows do not
   bootstrap the draft model or allocate proposal slots.
3. Bootstrap only non-empty sequence IDs not yet bootstrapped for the current
   epoch.
4. Begin `exact_q - 1` entries for every non-empty proposal.
5. Initialize each proposal with `first_target_token`.
6. Group non-empty rows by `exact_q`, retaining first-seen group order.
7. For each group and step, call one `decode_step_batch()` containing every
   active sequence in stable input order.
8. Validate one rank-one floating finite logit tensor per row on
   `backend.device`.
9. Stack the validated rows into `[batch_size, vocab_size]` and call
   `select_tensor_parallel_greedy_tokens()` with the executor rank, world
   size, batch size, and backend device.
10. Mark transactions fully materialized and register the complete batch
    through `ProposalKVLifecycleCoordinator.register_batch()`.

Do not call the target model, sampler, verifier, target KV cache, offload
manager, or scheduler from this file.

- [x] **Step 4: Run the focused test and verify GREEN**

Run the Task 1 command. Expected: all executor tests pass.

---

### Task 3: Finalize, Release, and Tensor-Free Authority

**Files:**
- Modify: `tinyvllm/engine/autoregressive_draft_executor.py`
- Modify: `tools/test_autoregressive_draft_executor.py`

**Interfaces:**
- Consumes: Task 2 proposal registrations and the generic lifecycle coordinator.
- Produces: `prepare_finalize_batch()`, `commit_finalize_batch()`, `rollback_finalize_batch()`, `release_sequence()`, and `authority_snapshot()`.

- [x] **Step 1: Write failing lifecycle tests**

Add:

```python
def test_partial_acceptance_commits_exact_accepted_minus_one_entries():
    executor, _, cache = ready_executor(prompts={1: ((2, 3), (0, 1))})
    proposal = executor.propose_batch((
        proposal_input(1, first_target_token=4, exact_q=4),
    ))[0]
    ticket = executor.prepare_finalize_batch((
        ProposalFinalizeRow(
            sequence_id=1,
            proposal_transaction_id=proposal.proposal_transaction_id,
            accepted_proposal_tokens=3,
        ),
    ))
    executor.commit_finalize_batch(ticket)

    assert cache.committed_length(1) == 4
```

The expected length is two prompt slots plus `max(3 - 1, 0)` proposal slots.

Also add tests proving:

- accepted counts 0 through 4 commit exactly `max(A-1, 0)` staged entries;
- rollback releases the complete staged suffix and retains prior committed
  prompt/accepted slots;
- multiple accepted rounds append committed slots without bootstrap replay;
- release rejects active transactions and prepared tickets;
- stale epoch release fails;
- successful release clears pending prompt, bootstrap state, committed slots,
  selected-token rows, and proposal-Q rows for that sequence;
- final release returns every physical slot;
- authority snapshots contain no tensors and include backend/model/tokenizer
  identity, rank/size, bootstrap rows, selected tokens, coordinator snapshot,
  timing buckets, and live-slot counts.

- [ ] **Step 2: Run lifecycle tests and verify RED**

Run the Task 1 command.

Expected: failures for missing lifecycle delegation and authority APIs.

- [x] **Step 3: Implement lifecycle delegation and evidence**

Delegate finalize calls directly to `self.proposal_kv_lifecycle`.

Release in this order:

1. call `assert_sequence_releasable(sequence_id, sequence_epoch)`;
2. validate pending/bootstrap epoch identity;
3. call coordinator `release_sequence()`;
4. clear only that sequence's model-specific immutable state and evidence.

Expose:

```python
def authority_snapshot(self) -> dict:
    return {
        "source_type": "independent_draft_model",
        "backend_identity": self.backend.backend_identity,
        "model_fingerprint": self.backend.model_fingerprint,
        "tokenizer_fingerprint": self.backend.tokenizer_fingerprint,
        "tensor_parallel_rank": self.tensor_parallel_rank,
        "tensor_parallel_size": self.tensor_parallel_size,
        "bootstrap_rows": tuple(self._bootstrap_rows),
        "selected_token_rows": tuple(self._selected_token_rows),
        "proposal_exact_q": tuple(self._proposal_exact_q_rows),
        "timing_ms": dict(self._timing_ms),
        "proposal_kv_lifecycle": (
            self.proposal_kv_lifecycle.authority_snapshot()
        ),
    }
```

Call `assert_tensor_free()` on the completed snapshot before returning it.
Timing must have separate `prompt_bootstrap`, `proposal_forward`, and
`proposal_finalize` buckets. Do not include checkpoint load time in
steady-state proposal timing.

- [x] **Step 4: Run lifecycle tests and verify GREEN**

Run the Task 1 command. Expected: all executor tests pass.

---

### Task 4: Qwen3 Multi-Layer Physical Proposal-KV Store

**Files:**
- Create: `tinyvllm/engine/qwen3_draft_backend.py`
- Create: `tools/test_qwen3_draft_backend.py`

**Interfaces:**
- Consumes: instantiated Qwen3 attention modules exposing `attn`, `num_kv_heads`, and `head_dim`.
- Produces: `Qwen3DraftPhysicalSlotStore`.

- [x] **Step 1: Write failing physical-store tests**

Create fake Qwen3 layers and assert exact shapes and ownership:

```python
def _model(layer_count=3, local_kv_heads=2, head_dim=4):
    layers = []
    for _ in range(layer_count):
        backend = SimpleNamespace(
            k_cache=torch.Tensor(),
            v_cache=torch.Tensor(),
            kv_quant_bits=None,
        )
        attention = SimpleNamespace(
            num_kv_heads=local_kv_heads,
            head_dim=head_dim,
            attn=backend,
        )
        layers.append(
            SimpleNamespace(
                self_attn=attention,
            )
        )
    return SimpleNamespace(
        model=SimpleNamespace(layers=layers),
    )


def test_store_binds_one_block_size_one_slice_per_layer():
    model = _model()
    store = Qwen3DraftPhysicalSlotStore(
        model,
        capacity=7,
        dtype=torch.bfloat16,
        device="cpu",
    )

    assert store.key_cache.shape == (3, 7, 1, 2, 4)
    assert store.value_cache.shape == (3, 7, 1, 2, 4)
    for layer_index, layer in enumerate(model.model.layers):
        assert layer.self_attn.attn.k_cache is store.key_cache[layer_index]
        assert layer.self_attn.attn.v_cache is store.value_cache[layer_index]
        assert layer.self_attn.attn.kv_quant_bits == 0
```

Also test deterministic reservation, exhaustion, duplicate/stale release,
zeroing all layer slices on release, block/slot identity, mismatched
per-layer local KV shapes, pre-bound foreign caches, invalid capacity/dtype,
and tensor-free authority counts.

- [ ] **Step 2: Run physical-store tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD /tmp/tinyllmforge-trace-test-venv/bin/python \
  -m pytest -q tools/test_qwen3_draft_backend.py
```

Expected: collection fails because `Qwen3DraftPhysicalSlotStore` is absent.

- [x] **Step 3: Implement the multi-layer store**

`Qwen3DraftPhysicalSlotStore` must:

- derive layer count and local KV shape from every instantiated attention;
- require identical local KV shape across layers;
- allocate K/V as
  `[layers, capacity, 1, local_kv_heads, head_dim]`;
- bind layer `i` to `key_cache[i]` and `value_cache[i]`;
- reject replacing a non-empty foreign cache;
- expose `capacity`, `block_size=1`, `layer_count`, `local_kv_heads`,
  `head_dim`, `dtype`, and `device`;
- reserve the lowest free slot IDs;
- zero K and V for every layer before returning released IDs to the free set;
- expose `slot_identity(slot_id)` as per-layer K/V data-pointer tuples;
- expose a tensor-free `authority_snapshot()` with allocated/free counts only.

Do not reference target KV, target block IDs, or the target offload manager.

- [x] **Step 4: Run physical-store tests and verify GREEN**

Run the Task 4 command. Expected: all physical-store tests pass.

---

### Task 5: Qwen3 Batch Backend and Attention Contexts

**Files:**
- Modify: `tinyvllm/engine/qwen3_draft_backend.py`
- Modify: `tools/test_qwen3_draft_backend.py`

**Interfaces:**
- Consumes: Task 1 prefill/decode rows, Task 4 physical store, `temporary_context`, and `Qwen3ForCausalLM`.
- Produces: `Qwen3AutoregressiveDraftBackend`.

- [x] **Step 1: Write failing backend context tests**

Use an injected fake model that records the active context from
`tinyvllm.utils.context.get_context()` and returns deterministic hidden rows.
Its `compute_logits()` returns a full-vocabulary floating tensor.

Add tests proving:

```python
def test_prefill_batch_packs_rows_into_one_model_forward():
    backend, model, cache = backend_fixture()
    first = cache.begin(1, 0, 2)
    second = cache.begin(2, 0, 3)

    backend.prefill_batch((
        AutoregressiveDraftPrefillRow(first, (7, 8), (0, 1)),
        AutoregressiveDraftPrefillRow(second, (4, 5, 6), (0, 1, 2)),
    ))

    assert model.forward_calls == 1
    call = model.calls[0]
    assert call.input_ids.tolist() == [7, 8, 4, 5, 6]
    assert call.positions.tolist() == [0, 1, 0, 1, 2]
    assert call.context.mode == "prefill"
    assert call.context.slot_mapping.tolist() == (
        list(first.staged_slot_ids) + list(second.staged_slot_ids)
    )
    assert call.context.cu_seqlens_q.tolist() == [0, 2, 5]
```

Add decode tests asserting:

- one model forward for all decode rows;
- slot mapping contains only each row's current staged slot;
- block tables contain committed slots plus staged prefix, padded with `-1`;
- context lengths match visible slot counts;
- `max_seqlen_q=1`;
- target offload manager is `None` and blockwise offload flags are false;
- output logits are one rank-one row per input row in input order;
- malformed model hidden/logit shape, dtype, device, and non-finite values
  fail before returning tokens.

- [ ] **Step 2: Run backend tests and verify RED**

Run the Task 4 command.

Expected: failures for missing `Qwen3AutoregressiveDraftBackend`.

- [x] **Step 3: Implement batch prefill and decode**

Constructor:

```python
class Qwen3AutoregressiveDraftBackend:
    def __init__(
        self,
        *,
        model,
        proposal_kv_cache,
        backend_identity,
        model_fingerprint,
        tokenizer_fingerprint,
    ):
        pass
```

Require a model with callable `forward` and `compute_logits`, a
`ProposalKVCache` whose physical store is `Qwen3DraftPhysicalSlotStore`, and
non-empty identity strings.

`prefill_batch()` must flatten rows once, build packed integer tensors on the
backend device, and enter one temporary prefill context with:

```text
mode="prefill"
is_prefill=True
slot_mapping=all staged slots in row order
block_tables=None
cu_seqlens_q=cu_seqlens_k=[0, cumulative token counts]
max_seqlen_q=max prompt length
max_seqlen_k=max prompt length
kv_offload_manager=None
kv_offload_blockwise_decode=False
kv_offload_blockwise_prefill=False
```

Run `model(input_ids, positions)` once. Validate only hidden row count because
prefill logits are intentionally not part of the executor contract.

`decode_step_batch()` must build one padded block-table tensor, one
context-length tensor, one slot-mapping tensor, and one input/position tensor.
Run the model once, compute logits once, validate shape
`[batch_size, vocab_size]`, floating dtype, backend device, and finite values,
then return stable rank-one views.

- [x] **Step 4: Run backend tests and verify GREEN**

Run the Task 4 command. Expected: all backend/store tests pass.

---

### Task 6: Checkpoint and Tokenizer Compatibility Contract

**Files:**
- Create: `tinyvllm/engine/autoregressive_draft_registration.py`
- Create: `tools/test_autoregressive_draft_registration.py`

**Interfaces:**
- Consumes: local target/draft checkpoint directories and local Transformers tokenizers.
- Produces: `CheckpointFingerprint`, `TokenizerContract`, `build_checkpoint_fingerprint()`, `build_tokenizer_contract()`, and `validate_tokenizer_compatibility()`.

- [x] **Step 1: Write failing identity and compatibility tests**

Create temporary checkpoint directories with small deterministic files and
fake tokenizers exposing `get_vocab()`, tokenizer class, init kwargs, and
special-token IDs.

Test:

```python
def test_same_size_different_token_order_is_rejected(tmp_path):
    target = tokenizer_contract(
        tmp_path / "target",
        vocab={"a": 0, "b": 1},
        eos_token_id=1,
    )
    draft = tokenizer_contract(
        tmp_path / "draft",
        vocab={"b": 0, "a": 1},
        eos_token_id=1,
    )

    with pytest.raises(ValueError, match="ordered token-to-ID"):
        validate_tokenizer_compatibility(target, draft)
```

Also cover:

- exact mapping and special-token equality passes;
- same mapping with different BOS/EOS/PAD/stop IDs fails;
- tokenizer class or normalization/init configuration mismatch fails;
- artifact content mismatch fails when both sides provide the same tokenizer
  artifact name;
- missing optional artifact is recorded but cannot replace mapping evidence;
- config and every `.safetensors` file contribute to checkpoint composite
  SHA256;
- changed file bytes change the composite hash;
- empty/no-shard checkpoint fails;
- snapshots contain paths only as normalized strings and no tensors.

- [ ] **Step 2: Run registration-contract tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD /tmp/tinyllmforge-trace-test-venv/bin/python \
  -m pytest -q tools/test_autoregressive_draft_registration.py
```

Expected: collection fails because the registration module does not exist.

- [x] **Step 3: Implement immutable fingerprints and exact compatibility**

Use these dataclasses:

```python
@dataclass(frozen=True)
class CheckpointFingerprint:
    model_path: str
    config_sha256: str
    shard_sha256: tuple[tuple[str, str], ...]
    composite_sha256: str


@dataclass(frozen=True)
class TokenizerContract:
    model_path: str
    tokenizer_class: str
    normalization_sha256: str
    ordered_token_to_id_sha256: str
    vocab_size: int
    bos_token_id: int | None
    eos_token_id: int | tuple[int, ...] | None
    pad_token_id: int | None
    stop_token_ids: tuple[int, ...]
    artifact_sha256: tuple[tuple[str, str], ...]
    composite_sha256: str
```

Hash canonical JSON with sorted keys and compact separators. Build the
ordered vocabulary payload by sorting `(token_id, token_string)` pairs, not
by token text. Hash these tokenizer artifact names when present:

```text
tokenizer.json
tokenizer_config.json
special_tokens_map.json
vocab.json
merges.txt
```

Compatibility requires exact equality for tokenizer class, normalization
hash, ordered mapping hash, vocabulary size, BOS/EOS/PAD IDs, stop-token IDs,
and every common artifact hash. Error messages must name the mismatched
contract field.

- [x] **Step 4: Run registration-contract tests and verify GREEN**

Run the Task 6 command. Expected: all contract tests pass.

---

### Task 7: Config and Source-Neutral ModelRunner Registration

**Files:**
- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/autoregressive_draft_registration.py`
- Create: `tools/test_autoregressive_draft_model_runner_integration.py`
- Modify: `tools/test_model_runner_proposal_prefill_observation.py`

**Interfaces:**
- Consumes: Tasks 1-6 and existing `load_model`, `Qwen3ForCausalLM`, `AutoConfig`, `AutoTokenizer`, and registry descriptor.
- Produces: validated independent-draft config, `_maybe_register_autoregressive_draft_executor()`, `autoregressive_draft_authority_snapshot()`, and generic registry entry `"autoregressive-draft"`.

- [x] **Step 1: Write failing config and ModelRunner integration tests**

Add config fields with expected defaults to test fixtures and assert:

```python
def test_enabled_tp4_fails_before_registration_dependencies_are_called():
    runner = _runner(
        enabled=True,
        tensor_parallel_size=4,
    )
    dependencies = _Dependencies()

    with pytest.raises(RuntimeError, match="currently requires TP1"):
        register(
            runner,
            registration_dependencies=dependencies,
        )

    assert dependencies.calls == []
    assert runner.speculative_proposal_executors.rows == {}
```

Add tests proving:

- disabled config performs no draft work;
- enabled config requires non-empty draft model path;
- backend must equal `"qwen3"`;
- max proposal tokens is in `1..4`;
- slot capacity is positive;
- TP1 registration order is fingerprint target, fingerprint draft, build both
  tokenizer contracts, validate compatibility, load Qwen3 config/model,
  load weights, move/eval model, build store/cache/backend/executor, register;
- failed compatibility allocates no model and no proposal slots;
- failed model loading leaves the target model and existing Qwen3.5 MTP
  executor usable;
- Qwen3.5 MTP and autoregressive draft descriptors coexist under distinct
  source-neutral registry IDs;
- prefill observation is delivered to both lifecycle executors without a
  source-specific branch;
- authority snapshot contains checkpoint/tokenizer fingerprints and executor
  snapshot but no tensors;
- generic Engine/Scheduler/verifier/finalize files contain no Qwen3 draft
  checkpoint or backend branch.

- [ ] **Step 2: Run integration tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD /tmp/tinyllmforge-trace-test-venv/bin/python \
  -m pytest -q \
  tools/test_autoregressive_draft_model_runner_integration.py \
  tools/test_model_runner_proposal_prefill_observation.py
```

Expected: failures for absent config and ModelRunner registration APIs.

- [x] **Step 3: Add config validation and registration dependencies**

Add exact config fields:

```python
autoregressive_draft_enabled: bool = False
autoregressive_draft_model: str | None = None
autoregressive_draft_backend: str = "qwen3"
autoregressive_draft_max_proposal_tokens: int = 4
autoregressive_draft_gpu_slot_capacity: int = 0
```

Validation rules:

- enabled must be bool;
- model is required and non-empty only when enabled;
- backend must equal `"qwen3"`;
- proposal token limit is integer `1..4`;
- enabled slot capacity must be positive;
- disabled capacity may be zero.

Create an injectable registration dependency bundle in
`autoregressive_draft_registration.py` with callables for fingerprinting,
tokenizer construction/validation, Qwen3 config/model construction, weight
loading, device transfer, physical store/cache/backend/executor construction,
and descriptor construction.

In `ModelRunner.__init__`, initialize:

```text
autoregressive_draft_model
autoregressive_draft_physical_store
autoregressive_draft_executor
autoregressive_draft_executor_descriptor
autoregressive_draft_registration_error
autoregressive_draft_checkpoint_identity
autoregressive_draft_tokenizer_contract
```

Call `_maybe_register_autoregressive_draft_executor()` immediately after
`_maybe_register_qwen35_mtp_executor()`.

Registration must:

1. return immediately when disabled;
2. fail closed on TP != 1 before invoking dependencies;
3. fingerprint target and draft checkpoints;
4. build and compare tokenizer contracts;
5. require draft HF config `model_type == "qwen3"`;
6. construct/load the draft model independently from target weights;
7. move it to the target model device and dtype, set `eval()`;
8. build and bind the proposal physical store;
9. build `ProposalKVCache`, Qwen3 backend, and generic executor;
10. register descriptor ID `"autoregressive-draft"`;
11. publish object references only after successful registry insertion;
12. store a typed registration error and release partial local references on
    failure without mutating the target model or existing registry entries.

Do not add a Qwen3 branch to Engine, Scheduler, verifier, proposal finalize,
or target-KV code.

- [x] **Step 4: Run integration tests and verify GREEN**

Run the Task 7 command. Expected: all integration tests pass.

---

### Task 8: Real TP1 Checkpoint Gate Harness and Full Local Regression

**Files:**
- Create: `tools/autoregressive_draft_tp1_engine_gate.py`
- Create: `tools/test_autoregressive_draft_tp1_engine_gate.py`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: a source-attributed local Qwen3 checkpoint, a source-attributed local Qwen3.5 checkpoint, and completed Tasks 1-7.
- Produces: a machine-verifiable TP1 exact-greedy gate artifact and an honest handoff status.

- [x] **Step 1: Write failing gate-schema tests**

The gate test must use fake engine factories and no CUDA. Require output:

```python
assert payload["schema_version"] == 1
assert payload["gate"] == "autoregressive_draft_tp1_engine"
assert payload["configuration"] == {
    "tensor_parallel_size": 1,
    "dtype": "bfloat16",
    "temperature": 0.0,
    "max_proposal_tokens": 4,
}
assert payload["checkpoint_identity"]["draft"]["composite_sha256"]
assert payload["checkpoint_identity"]["target"]["composite_sha256"]
assert payload["tokenizer_contract"]["compatible"] is True
assert payload["cases"]["batch_1"]["exact_output_parity"] is True
assert payload["cases"]["batch_4"]["exact_output_parity"] is True
assert payload["evidence"]["extra_target_forward_count"] == 0
assert payload["evidence"]["proposal_kv_live_slots_after_release"] == 0
assert payload["evidence"]["real_draft_forward_count"] > 0
assert payload["evidence"]["proposal_kv_bytes"] > 0
assert payload["evidence"]["target_kv_bytes"] > 0
```

Add negative tests for token mismatch, missing checkpoint hashes, tokenizer
incompatibility, zero draft forwards, nonzero extra target forwards, proposal
KV leak, missing acceptance rows, and conflated target/proposal KV bytes.

- [ ] **Step 2: Run gate tests and verify RED**

Run:

```bash
PYTHONPATH=$PWD /tmp/tinyllmforge-trace-test-venv/bin/python \
  -m pytest -q tools/test_autoregressive_draft_tp1_engine_gate.py
```

Expected: collection fails because the gate module does not exist.

- [x] **Step 3: Implement the gate harness**

The CLI must require:

```text
--target-model
--draft-model
--output
--prompt-file
```

It must run target-only and learned-draft engines with identical prompts,
BF16, TP1, temperature zero, and output limit. Run batch 1 first and batch 4
second. Compare complete output token ID tuples exactly.

Record:

- immutable target/draft checkpoint hashes;
- tokenizer artifact and ordered mapping hashes;
- exact engine configuration;
- per-case prompts and complete output token IDs;
- proposal tokens and accepted-prefix counts;
- bootstrap/proposal/finalize/verification timing separated from model load;
- real Qwen3 forward call count;
- target first-token and tail-forward counts proving no extra target forward;
- proposal and target KV allocated/live/released bytes separately;
- proposal slot ownership before and after release;
- explicit `performance_pass_criterion=False`.

Exit nonzero when any correctness or evidence requirement fails. Do not call
this harness during local implementation without separate GPU/remote
authorization.

- [x] **Step 4: Run dependency-light gate tests and the focused regression matrix**

Run:

```bash
PYTHONPATH=$PWD /tmp/tinyllmforge-trace-test-venv/bin/python -m pytest -q \
  tools/test_autoregressive_draft_executor.py \
  tools/test_qwen3_draft_backend.py \
  tools/test_autoregressive_draft_registration.py \
  tools/test_autoregressive_draft_model_runner_integration.py \
  tools/test_autoregressive_draft_tp1_engine_gate.py \
  tools/test_proposal_kv_cache.py \
  tools/test_proposal_kv_lifecycle.py \
  tools/test_model_runner_proposal_executor.py \
  tools/test_model_runner_proposal_prefill_observation.py \
  tools/test_qwen35_mtp_executor.py \
  tools/test_qwen35_mtp_model_runner_integration.py \
  tools/test_engine_speculative_execution.py \
  tools/test_engine_speculative_runtime.py
```

Expected: all tests pass.

- [x] **Step 5: Compile and run static source-neutral checks**

Run:

```bash
PYTHONPATH=$PWD /tmp/tinyllmforge-trace-test-venv/bin/python -m py_compile \
  tinyvllm/engine/autoregressive_draft_executor.py \
  tinyvllm/engine/qwen3_draft_backend.py \
  tinyvllm/engine/autoregressive_draft_registration.py \
  tinyvllm/config.py \
  tinyvllm/engine/model_runner.py \
  tools/autoregressive_draft_tp1_engine_gate.py

! rg -n \
  "Qwen3AutoregressiveDraftBackend|autoregressive_draft_model|Qwen3-0\\.6B" \
  tinyvllm/engine/llm_engine.py \
  tinyvllm/engine/scheduler.py \
  tinyvllm/speculative/verifier.py \
  tinyvllm/speculative/batch_runtime.py \
  tinyvllm/engine/proposal_kv_cache.py \
  tinyvllm/engine/proposal_kv_lifecycle.py
```

Expected: compilation succeeds and the static search returns no matches.

- [x] **Step 6: Update the handoff without overclaiming**

Record:

- local test commands and exact pass counts;
- whether the real checkpoint gate was run;
- checkpoint/tokenizer paths only if freshly verified;
- exact gate artifact path and SHA256 if run;
- remaining TP4, 4K/16K/32K, batch/multi-sequence, performance, KV H2D,
  KV8/KV4, and production-readiness gaps.

If the real gate has not run, retain:

```text
SECOND_LEARNED_STRUCTURE=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

If the real gate passes, only
`SECOND_LEARNED_STRUCTURE_TP1_CORRECTNESS=ESTABLISHED` may be promoted.
Phase 1 remains `NOT_ACHIEVED` until the full objective matrix is covered.
