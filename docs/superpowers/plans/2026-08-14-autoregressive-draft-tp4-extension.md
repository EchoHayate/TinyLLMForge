# Autoregressive Draft TP4 Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not dispatch subagents.

**Goal:** Extend the independent Qwen3 autoregressive draft executor from TP1 to sharded TP4 with rank-local proposal KV, root-only full logits, synchronized logical lifecycle authority, and failure-atomic all-rank registration.

**Architecture:** Each target rank constructs one corresponding Qwen3 draft shard, one local physical proposal-KV store, and one local executor. A dedicated tensor-parallel coordinator uses structured one-time registration gathering and fixed-size runtime digest gathering so all ranks execute the same logical transitions while physical transaction IDs, slot IDs, K/V tensors, and storage pointers remain rank-local.

**Tech Stack:** Python dataclasses and protocols, PyTorch distributed collectives, existing Qwen3 tensor-parallel layers, `ProposalKVCache`, `ProposalKVLifecycleCoordinator`, `ModelRunnerProposalExecutorRegistry`, dependency-light pytest with injected collectives.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not create or switch worktrees or branches.
- Do not use subagents.
- Do not stage, commit, push, stash, reset, or clean.
- Do not run remote, GPU, NCCL, or real-checkpoint workloads without separate authorization.
- Preserve unrelated modified and untracked files in the dirty workspace.
- Preserve exact greedy parity and `MAX_PROPOSAL_TOKENS=4`.
- Support only `tensor_parallel_size in {1, 4}` for the independent autoregressive drafter; every other topology remains fail-closed.
- Qwen3 weights and proposal K/V remain TP-sharded and rank-local.
- All ranks execute identical bootstrap, proposal, prepare, commit, rollback, and release transitions.
- Only selected token IDs, registration status, and normalized logical authority digests cross ranks.
- Never compare or gather physical proposal slot IDs, proposal transaction IDs, K/V storage pointers, or K/V tensors.
- Proposal token zero remains the target-produced `first_target_token`.
- Learned decode performs at most `Q - 1` forwards.
- Proposal staged-entry count remains `Q - 1`.
- Proposal committed-entry count remains `max(accepted_proposal_tokens - 1, 0)`.
- Do not change verifier token selection, fallback indexing, accepted-prefix semantics, target-KV transactions, speculative side-state selection, offload counters, scheduler publication, n-gram, SAM, or Qwen3.5 native-MTP behavior.
- Registry publication occurs only after all-rank private construction, registry preflight, and identity consensus succeed.
- Every predictable registration failure occurs before registry mutation because the registry has no unregister operation.
- Every fallible lifecycle phase captures local errors, converges the stage outcome across ranks, cleans still-owned local state on successful peers, and then raises a common stage-attributed failure.
- A process death or failure inside a Qwen3 tensor-parallel collective remains a poisoned distributed process-group failure; this plan does not attempt recovery.
- Passing local synthetic tests does not establish a real second learned structure, real TP4 Engine parity, 4K/16K/32K correctness, performance improvement, KV8/KV4, production readiness, or Phase 1 completion.
- Every production behavior change follows RED -> GREEN with the focused failure observed before implementation.

---

## File Map

- Create `tinyvllm/engine/autoregressive_draft_tp.py`
  - Own canonical logical encoding, registration status gathering, fixed-size runtime digest gathering, and common stage convergence.
- Create `tools/test_autoregressive_draft_tp.py`
  - Cover topology validation, canonical encoding, digest equality, injected all-rank failures, and physical-identity exclusion.
- Modify `tinyvllm/engine/speculative_proposal_executor.py`
  - Add mutation-free registry preflight using the exact validation shared by `register()`.
- Modify `tools/test_autoregressive_draft_model_runner_integration.py`
  - Cover registry preflight, all-rank registration, publication ordering, root-only ModelRunner return behavior, and generic-boundary source checks.
- Modify `tinyvllm/engine/qwen3_draft_backend.py`
  - Accept root full logits and non-root `None`; expose local shard geometry, parameter bytes, proposal-KV bytes, and real forward counters.
- Modify `tools/test_qwen3_draft_backend.py`
  - Cover root/non-root logits, local geometry, malformed root logits, and non-root local-vocabulary rejection.
- Modify `tinyvllm/engine/autoregressive_draft_executor.py`
  - Permit TP1/TP4, inject the coordinator, synchronize proposal and lifecycle phases, and remove physical IDs from cross-rank authority.
- Modify `tools/test_autoregressive_draft_executor.py`
  - Cover TP4 construction, Q=1..4, mixed-Q ordering, root/non-root token selection, cleanup, and lifecycle convergence.
- Modify `tinyvllm/engine/autoregressive_draft_registration.py`
  - Add private registration candidates and deterministic status/identity helpers.
- Modify `tinyvllm/engine/model_runner.py`
  - Build private local candidates on every rank, publish only after consensus, and extend authority snapshots.
- Modify `tinyvllm/config.py`
  - Permit enabled independent draft execution only at TP1 or TP4.
- Modify `tools/test_autoregressive_draft_registration.py`
  - Cover candidate and status identities without loading real checkpoints.
- Modify `tools/test_autoregressive_draft_tp1_engine_gate.py`
  - Assert TP1 gate semantics and classification remain unchanged.
- Create `tools/test_autoregressive_draft_tp4_local_gate.py`
  - Aggregate four injected rank snapshots and enforce the local TP4 evidence contract without NCCL or real checkpoints.

---

### Task 1: Tensor-Parallel Coordinator and Canonical Logical Authority

**Files:**
- Create: `tinyvllm/engine/autoregressive_draft_tp.py`
- Create: `tools/test_autoregressive_draft_tp.py`

**Interfaces:**
- Consumes: rank, world size, draft device, injected registration gather, and injected fixed-size digest gather.
- Produces: `AutoregressiveDraftRankRegistrationStatus`, `AutoregressiveDraftTensorParallelCoordinator.collect_registration_status()`, `assert_logical_authority()`, and `converge_stage()`.

- [ ] **Step 1: Write failing topology and registration-gather tests**

Create fixtures that simulate four ranks without initializing `torch.distributed`:

```python
from dataclasses import replace

import pytest
import torch

from tinyvllm.engine.autoregressive_draft_tp import (
    AutoregressiveDraftRankRegistrationStatus,
    AutoregressiveDraftTensorParallelCoordinator,
)


def _status(rank, *, success=True, stage="ready", message=None):
    return AutoregressiveDraftRankRegistrationStatus(
        rank=rank,
        world_size=4,
        success=success,
        stage=stage,
        error_type=None if success else "RuntimeError",
        message=message,
        target_checkpoint_sha256="target",
        draft_checkpoint_sha256="draft",
        target_tokenizer_sha256="target-tokenizer",
        draft_tokenizer_sha256="draft-tokenizer",
        backend_identity="qwen3",
        executor_id="autoregressive-draft",
        capabilities_sha256="capabilities",
    )


@pytest.mark.parametrize("world_size", (2, 3, 5, 8))
def test_coordinator_rejects_unsupported_world_sizes(world_size):
    with pytest.raises(RuntimeError, match="TP1 or TP4"):
        AutoregressiveDraftTensorParallelCoordinator(
            rank=0,
            world_size=world_size,
            device="cpu",
        )


def test_registration_collects_exactly_one_status_per_rank():
    statuses = tuple(_status(rank) for rank in range(4))
    coordinator = AutoregressiveDraftTensorParallelCoordinator(
        rank=2,
        world_size=4,
        device="cpu",
        gather_registration_status=lambda local: statuses,
    )

    assert coordinator.collect_registration_status(statuses[2]) == statuses


@pytest.mark.parametrize(
    "statuses,message",
    (
        (
            (_status(0), _status(1), _status(2)),
            "exactly world_size",
        ),
        (
            (_status(0), _status(1), _status(1), _status(3)),
            "ranks 0..world_size-1",
        ),
        (
            tuple(
                replace(_status(rank), world_size=1)
                for rank in range(4)
            ),
            "world_size",
        ),
    ),
)
def test_registration_rejects_malformed_status_sets(statuses, message):
    coordinator = AutoregressiveDraftTensorParallelCoordinator(
        rank=0,
        world_size=4,
        device="cpu",
        gather_registration_status=lambda local: statuses,
    )
    with pytest.raises(RuntimeError, match=message):
        coordinator.collect_registration_status(_status(0))
```

- [ ] **Step 2: Run the coordinator tests and verify RED**

Run:

```bash
uv run --offline --python 3.12 --with pytest --with torch \
  pytest -q tools/test_autoregressive_draft_tp.py
```

Expected: collection fails because
`tinyvllm.engine.autoregressive_draft_tp` does not exist.

- [ ] **Step 3: Add failing canonical-encoding and fixed-size digest tests**

Add tests proving:

```python
def test_logical_digest_is_stable_for_sorted_dictionary_keys():
    gathered = []

    def gather(local):
        gathered.append(local.detach().clone())
        return tuple(local.detach().clone() for _ in range(4))

    coordinator = AutoregressiveDraftTensorParallelCoordinator(
        rank=0,
        world_size=4,
        device="cpu",
        gather_digest=gather,
    )
    first = coordinator.assert_logical_authority(
        stage="proposal_preflight",
        rows={"b": [2, 3], "a": {"value": 1}},
    )
    second = coordinator.assert_logical_authority(
        stage="proposal_preflight",
        rows={"a": {"value": 1}, "b": [2, 3]},
    )

    assert first == second
    assert gathered[0].dtype == torch.uint8
    assert gathered[0].shape == (33,)
    assert gathered[0][0].item() == 1


@pytest.mark.parametrize(
    "value,message",
    (
        (torch.tensor([1]), "tensor"),
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        ({1, 2}, "set"),
        (object(), "unsupported"),
    ),
)
def test_logical_encoder_rejects_noncanonical_values(value, message):
    coordinator = AutoregressiveDraftTensorParallelCoordinator(
        rank=0,
        world_size=1,
        device="cpu",
    )
    with pytest.raises((TypeError, ValueError), match=message):
        coordinator.assert_logical_authority(
            stage="invalid",
            rows=value,
        )


def test_physical_ids_are_not_part_of_logical_rows():
    coordinator = AutoregressiveDraftTensorParallelCoordinator(
        rank=0,
        world_size=1,
        device="cpu",
    )
    left = coordinator.assert_logical_authority(
        stage="materialized",
        rows={
            "sequence_id": 7,
            "proposal_token_ids": (4, 5, 6),
            "staged_entry_count": 2,
        },
    )
    right = coordinator.assert_logical_authority(
        stage="materialized",
        rows={
            "sequence_id": 7,
            "proposal_token_ids": (4, 5, 6),
            "staged_entry_count": 2,
        },
    )
    assert left == right
```

Add an unequal-gather fixture that changes one digest byte and assert all
simulated ranks receive a `RuntimeError` containing the stage name and
`logical authority mismatch`.

- [x] **Step 4: Implement the coordinator**

Create these public definitions:

```python
@dataclass(frozen=True)
class AutoregressiveDraftRankRegistrationStatus:
    rank: int
    world_size: int
    success: bool
    stage: str
    error_type: str | None
    message: str | None
    target_checkpoint_sha256: str | None
    draft_checkpoint_sha256: str | None
    target_tokenizer_sha256: str | None
    draft_tokenizer_sha256: str | None
    backend_identity: str | None
    executor_id: str | None
    capabilities_sha256: str | None


```

Create `AutoregressiveDraftTensorParallelCoordinator` with these exact
public signatures:

- `__init__(self, *, rank: int, world_size: int, device: object, gather_registration_status=None, gather_digest=None)`
- `collect_registration_status(self, status: AutoregressiveDraftRankRegistrationStatus) -> tuple[AutoregressiveDraftRankRegistrationStatus, ...]`
- `assert_logical_authority(self, *, stage: str, rows: object) -> str`
- `converge_stage(self, *, stage: str, rows: object, local_error: BaseException | None) -> str`

Use a recursive canonicalizer that accepts only `None`, booleans, integers,
finite floats, strings, tuples/lists, string-keyed dictionaries, and
dataclasses recursively converted to these values. Encode canonical values
with compact sorted JSON and compute SHA-256.

For TP1, return the local status or digest without a distributed operation.
For TP4 registration, default to `torch.distributed.all_gather_object`.
For TP4 runtime stages, build one contiguous `torch.uint8[33]` tensor:

```python
payload = torch.empty(33, dtype=torch.uint8, device=self.device)
payload[0] = 1 if local_error is None else 0
payload[1:] = torch.tensor(
    tuple(digest_bytes),
    dtype=torch.uint8,
    device=self.device,
)
```

Default `gather_digest` performs `torch.distributed.all_gather` into exactly
four tensors of the same shape. A success row hashes `{"stage": stage,
"rows": rows}`. A failure row hashes `{"stage": stage, "error_type":
type(local_error).__name__, "message": str(local_error)}`. Reject any failure
bit and any unequal success digest with one stage-attributed `RuntimeError`.
Preserve the local exception as `raise common_error from local_error` when
the current rank failed.

- [ ] **Step 5: Add failing one-rank failure-convergence tests**

Simulate rank 2 reporting a failure row while the local rank reports success:

```python
def test_peer_failure_forces_successful_rank_to_raise_common_stage_error():
    def gather(local):
        rows = [local.detach().clone() for _ in range(4)]
        rows[2][0] = 0
        rows[2][1:] = torch.arange(32, dtype=torch.uint8)
        return tuple(rows)

    coordinator = AutoregressiveDraftTensorParallelCoordinator(
        rank=0,
        world_size=4,
        device="cpu",
        gather_digest=gather,
    )
    with pytest.raises(RuntimeError, match="bootstrap_prepare"):
        coordinator.converge_stage(
            stage="bootstrap_prepare",
            rows={"sequence_id": 9},
            local_error=None,
        )
```

Also assert that local failure in TP1 raises the common stage error with the
original exception as `__cause__`.

- [x] **Step 6: Run Task 1 tests and verify GREEN**

Run the Task 1 command.

Expected: all coordinator tests pass without initializing a process group.

---

### Task 2: Mutation-Free Proposal Registry Preflight

**Files:**
- Modify: `tinyvllm/engine/speculative_proposal_executor.py`
- Modify: `tools/test_autoregressive_draft_model_runner_integration.py`

**Interfaces:**
- Consumes: executor ID, executor object, and `DraftCapabilities`.
- Produces: `ModelRunnerProposalExecutorRegistry.preflight_registration()` and one shared validation helper used by both preflight and publication.

- [ ] **Step 1: Write failing registry preflight tests**

Add:

```python
def test_registry_preflight_validates_without_mutation():
    registry = ModelRunnerProposalExecutorRegistry()
    executor = _FakeExecutor()

    normalized = registry.preflight_registration(
        "autoregressive-draft",
        executor,
        executor.capabilities,
    )

    assert normalized == executor.capabilities
    assert registry.lifecycle_executor_ids() == ()


def test_registry_preflight_and_register_reject_the_same_invalid_executor():
    registry = ModelRunnerProposalExecutorRegistry()
    executor = _FakeExecutor()
    executor.prepare_finalize_batch = None

    with pytest.raises(ValueError, match="prepare_finalize_batch"):
        registry.preflight_registration(
            "autoregressive-draft",
            executor,
            executor.capabilities,
        )
    with pytest.raises(ValueError, match="prepare_finalize_batch"):
        registry.register(
            "autoregressive-draft",
            executor,
            executor.capabilities,
        )
    assert registry.lifecycle_executor_ids() == ()


def test_registry_preflight_rejects_duplicate_id_without_mutation():
    registry = ModelRunnerProposalExecutorRegistry()
    executor = _FakeExecutor()
    registry.register("autoregressive-draft", executor, executor.capabilities)

    with pytest.raises(ValueError, match="already registered"):
        registry.preflight_registration(
            "autoregressive-draft",
            _FakeExecutor(),
            _FakeExecutor.capabilities,
        )
    assert registry.lifecycle_executor_ids() == ("autoregressive-draft",)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
uv run --offline --python 3.12 --with pytest --with torch \
  pytest -q tools/test_autoregressive_draft_model_runner_integration.py \
  -k 'registry_preflight'
```

Expected: tests fail because `preflight_registration()` is missing.

- [x] **Step 3: Extract one exact registry validator**

Add:

```python
def _validated_registration(
    self,
    executor_id: str,
    executor: ProposalExecutor,
    capabilities: DraftCapabilities,
) -> DraftCapabilities:
    normalized = validate_draft_capabilities(
        capabilities,
        expected_execution_domain="model_runner",
    )
    if not isinstance(executor_id, str) or not executor_id:
        raise ValueError(
            "proposal executor ID must be a non-empty string"
        )
    if executor_id in self._entries:
        raise ValueError(
            "proposal executor ID is already registered"
        )
    if getattr(executor, "capabilities", None) != normalized:
        raise ValueError(
            "proposal executor capabilities must exactly match"
        )
    if not callable(getattr(executor, "propose_batch", None)):
        raise ValueError(
            "proposal executor must expose callable propose_batch"
        )
    if normalized.requires_proposal_lifecycle:
        for method_name in (
            "observe_target_prefill",
            "prepare_finalize_batch",
            "commit_finalize_batch",
            "rollback_finalize_batch",
            "release_sequence",
        ):
            if not callable(getattr(executor, method_name, None)):
                raise ValueError(
                    "lifecycle proposal executor must expose "
                    f"callable {method_name}"
                )
    return normalized


def preflight_registration(
    self,
    executor_id: str,
    executor: ProposalExecutor,
    capabilities: DraftCapabilities,
) -> DraftCapabilities:
    return self._validated_registration(
        executor_id,
        executor,
        capabilities,
    )
```

Move the current complete validation from `register()` into
`_validated_registration()` without weakening any check. `register()` must
become:

```python
normalized = self._validated_registration(
    executor_id,
    executor,
    capabilities,
)
self._entries[executor_id] = (executor, normalized)
```

Preflight must never modify `_entries`.

- [x] **Step 4: Run focused and generic registry tests and verify GREEN**

Run:

```bash
uv run --offline --python 3.12 --with pytest --with torch \
  pytest -q \
  tools/test_autoregressive_draft_model_runner_integration.py \
  tools/test_speculative_model_runner_callbacks.py \
  -k 'registry or registration'
```

Expected: all selected tests pass and no registry entry appears after
preflight-only calls.

---

### Task 3: Topology-Aware Qwen3 Draft Backend Logits and Rank-Local Evidence

**Files:**
- Modify: `tinyvllm/engine/qwen3_draft_backend.py`
- Modify: `tools/test_qwen3_draft_backend.py`

**Interfaces:**
- Consumes: Qwen3 model shard whose `compute_logits()` returns full
  `[batch, vocab]` logits on rank zero and `None` on non-root ranks.
- Produces: `decode_step_batch() -> tuple[torch.Tensor, ...] | None` and
  authority fields for local parameter bytes, local geometry, local forward
  counts, and local proposal-KV bytes.

- [ ] **Step 1: Write failing root/non-root backend tests**

Extend `_FakeQwen3` so `compute_logits()` can return a configured value, then
add:

```python
def test_tp4_root_backend_returns_full_logit_rows():
    backend, cache, model = _backend_fixture(rank=0, world_size=4)
    transaction = _decode_transaction(cache)
    model.logits_result = torch.tensor([
        [0.0, 4.0, 1.0],
        [3.0, 2.0, 1.0],
    ])

    rows = backend.decode_step_batch(_decode_rows(transaction, count=2))

    assert isinstance(rows, tuple)
    assert len(rows) == 2
    assert all(row.shape == (3,) for row in rows)


def test_tp4_non_root_backend_requires_none_logits():
    backend, cache, model = _backend_fixture(rank=2, world_size=4)
    transaction = _decode_transaction(cache)
    model.logits_result = None

    assert backend.decode_step_batch(
        _decode_rows(transaction, count=1)
    ) is None


def test_tp4_non_root_local_vocabulary_logits_fail_closed():
    backend, cache, model = _backend_fixture(rank=1, world_size=4)
    transaction = _decode_transaction(cache)
    model.logits_result = torch.ones(1, 8)

    with pytest.raises(ValueError, match="non-root logits must be None"):
        backend.decode_step_batch(
            _decode_rows(transaction, count=1)
        )
```

Add root tests for unequal row width, integer dtype, wrong device,
non-finite values, and wrong batch row count.

- [ ] **Step 2: Run backend tests and verify RED**

Run:

```bash
uv run --offline --python 3.12 --with pytest --with torch \
  pytest -q tools/test_qwen3_draft_backend.py \
  -k 'tp4 or non_root or local_vocabulary'
```

Expected: constructor or decode tests fail because the backend has no
topology-aware logits contract.

- [x] **Step 3: Extend backend construction and decode return contract**

Add constructor parameters:

```python
tensor_parallel_rank: int = 0,
tensor_parallel_size: int = 1,
```

Validate `tensor_parallel_size in (1, 4)` and rank bounds. Store both values.
Change:

```python
def decode_step_batch(
    self,
    rows: tuple[AutoregressiveDraftDecodeRow, ...],
) -> tuple[torch.Tensor, ...] | None:
```

After `compute_logits(hidden)`:

```python
if self.tensor_parallel_rank != 0:
    if logits is not None:
        raise ValueError("non-root logits must be None")
    return None
if not isinstance(logits, torch.Tensor):
    raise ValueError("root model logits must be a tensor")
```

Keep all current rank-zero shape, dtype, device, and finite checks. Never
return local-vocabulary logits from a worker.

- [ ] **Step 4: Add failing local geometry and byte-evidence tests**

Add:

```python
def test_backend_snapshot_reports_rank_local_geometry_and_bytes():
    backend, _, model = _backend_fixture(
        rank=3,
        world_size=4,
        local_query_heads=8,
        local_kv_heads=2,
    )
    snapshot = backend.authority_snapshot()

    assert snapshot["tensor_parallel_rank"] == 3
    assert snapshot["tensor_parallel_size"] == 4
    assert snapshot["local_query_heads"] == 8
    assert snapshot["local_kv_heads"] == 2
    assert snapshot["local_model_parameter_bytes"] == sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    assert snapshot["local_proposal_kv_bytes"] > 0
```

Ensure the fixture gives each fake attention module both
`num_heads`/`num_kv_heads`, and gives the fake model real parameters.

- [x] **Step 5: Extend backend authority snapshot**

Derive local query/KV geometry from instantiated layers and reject
inconsistent per-layer local geometry. Compute:

```python
local_model_parameter_bytes = sum(
    parameter.numel() * parameter.element_size()
    for parameter in self.model.parameters()
)
```

Publish:

```python
{
    "tensor_parallel_rank": self.tensor_parallel_rank,
    "tensor_parallel_size": self.tensor_parallel_size,
    "local_model_parameter_bytes": local_model_parameter_bytes,
    "local_proposal_kv_bytes": proposal_kv_bytes,
    "local_query_heads": self.local_query_heads,
    "local_kv_heads": self.physical_store.local_kv_heads,
    "local_prefill_forward_count": self._prefill_forward_count,
    "local_decode_forward_count": self._decode_forward_count,
}
```

Retain existing compatibility keys if current gate tests consume them.

- [x] **Step 6: Run complete backend tests and verify GREEN**

Run:

```bash
uv run --offline --python 3.12 --with pytest --with torch \
  pytest -q tools/test_qwen3_draft_backend.py
```

Expected: all physical-store and backend tests pass.

---

### Task 4: TP4 Executor Proposal Selection and Transaction Cleanup

**Files:**
- Modify: `tinyvllm/engine/autoregressive_draft_executor.py`
- Modify: `tools/test_autoregressive_draft_executor.py`

**Interfaces:**
- Consumes: Task 1 coordinator, Task 3 topology-aware backend, and
  `select_tensor_parallel_greedy_tokens()`.
- Produces: TP1/TP4 executor construction, synchronized proposal preflight,
  root-only logits validation, all-rank token selection, materialized
  authority, and cleanup before lifecycle registration.

- [ ] **Step 1: Write failing TP4 constructor and logits tests**

Add a coordinator fixture that records stages and returns local success:

```python
class _RecordingCoordinator:
    def __init__(self):
        self.stages = []

    def assert_logical_authority(self, *, stage, rows):
        self.stages.append((stage, rows))
        return f"{stage}-digest"

    def converge_stage(self, *, stage, rows, local_error):
        self.stages.append((stage, rows))
        if local_error is not None:
            raise RuntimeError(f"{stage} failed") from local_error
        return f"{stage}-digest"
```

Add:

```python
@pytest.mark.parametrize("rank", (0, 1, 2, 3))
def test_tp4_executor_accepts_every_valid_rank(rank):
    executor = _executor(
        rank=rank,
        world_size=4,
        coordinator=_RecordingCoordinator(),
    )
    assert executor.tensor_parallel_rank == rank
    assert executor.tensor_parallel_size == 4


@pytest.mark.parametrize("world_size", (2, 3, 5))
def test_executor_rejects_every_other_topology(world_size):
    with pytest.raises(RuntimeError, match="TP1 or TP4"):
        _executor(rank=0, world_size=world_size)


def test_tp4_non_root_passes_none_to_greedy_selector(monkeypatch):
    executor, backend, _ = _ready_executor(
        rank=2,
        world_size=4,
        backend_logits=None,
    )
    calls = []

    def select(logits, **kwargs):
        calls.append((logits, kwargs))
        return torch.tensor([7], dtype=torch.int64)

    monkeypatch.setattr(
        "tinyvllm.engine.autoregressive_draft_executor."
        "select_tensor_parallel_greedy_tokens",
        select,
    )
    proposal = executor.propose_batch(
        (_proposal_input(1, first_target_token=4, exact_q=2),)
    )[0]

    assert proposal.token_ids == (4, 7)
    assert calls[0][0] is None
```

Add rank-zero full-logit, non-root local-logit, malformed-root-logit, and
broadcast-failure tests. For every failure, assert zero new active
transactions and unchanged committed prompt slots.

- [ ] **Step 2: Run focused executor tests and verify RED**

Run:

```bash
uv run --offline --python 3.12 --with pytest --with torch \
  pytest -q tools/test_autoregressive_draft_executor.py \
  -k 'tp4 or non_root or broadcast_failure'
```

Expected: TP4 construction fails at the current TP1-only guard.

- [x] **Step 3: Change executor topology and protocol**

Import `AutoregressiveDraftTensorParallelCoordinator`. Change the backend
protocol method to the exact signature
`decode_step_batch(self, rows: tuple[AutoregressiveDraftDecodeRow, ...]) -> tuple[object, ...] | None`.

Add optional constructor injection:

```python
tensor_parallel_coordinator: (
    AutoregressiveDraftTensorParallelCoordinator | None
) = None,
```

Require world size 1 or 4. Construct the default coordinator with executor
rank, size, and backend device. Retain all TP1 behavior.

- [x] **Step 4: Add synchronized proposal preflight and materialized stages**

Before any proposal slot allocation, build immutable rows in input order:

```python
proposal_preflight_rows = tuple({
    "batch_index": input_index,
    "sequence_id": input_row.sequence_id,
    "sequence_epoch": (
        self._pending_prompts[input_row.sequence_id].sequence_epoch
    ),
    "context_token_count": context_token_count,
    "exact_q": exact_q,
    "first_target_token": input_row.first_target_token,
} for (
    input_index,
    input_row,
    context_token_count,
    exact_q,
) in normalized)
```

Call:

```python
self.tensor_parallel_coordinator.assert_logical_authority(
    stage="proposal_preflight",
    rows=proposal_preflight_rows,
)
```

In `_run_exact_q_group()`, treat backend output as
`root_logits_or_none`. Rank zero calls `_validate_logit_rows()` and stacks
the result; non-root requires `None` and passes `None` to the greedy
selector. Wrap local validation and selection in:

```python
local_error = None
selected = None
try:
    root_logits_or_none = self.backend.decode_step_batch(
        decode_rows
    )
    if self.tensor_parallel_rank == 0:
        logit_rows = self._validate_logit_rows(
            root_logits_or_none,
            expected_count=len(decode_rows),
        )
        logits = torch.stack(logit_rows, dim=0)
    else:
        if root_logits_or_none is not None:
            raise ValueError("non-root logits must be None")
        logits = None
    selected = select_tensor_parallel_greedy_tokens(
        logits,
        rank=self.tensor_parallel_rank,
        world_size=self.tensor_parallel_size,
        batch_size=len(decode_rows),
        device=torch.device(self.backend.device),
    )
except BaseException as error:
    local_error = error
self.tensor_parallel_coordinator.converge_stage(
    stage=f"proposal_decode_step_{step}",
    rows={
        "sequence_ids": tuple(
            transaction.sequence_id for transaction in transactions
        ),
        "step": step,
        "exact_q": exact_q,
    },
    local_error=local_error,
)
if local_error is not None:
    raise local_error
```

After materialization, compare:

```python
{
    "batch_index": input_index,
    "sequence_id": transaction.sequence_id,
    "sequence_epoch": transaction.sequence_epoch,
    "exact_q": exact_q,
    "proposal_token_ids": tuple(tokens),
    "staged_entry_count": exact_q - 1,
    "logical_state": "materialized",
}
```

Only after `proposal_materialized` succeeds may the local lifecycle
coordinator register the transaction.

- [ ] **Step 5: Add failing mismatch and cleanup tests**

Inject a coordinator that fails `proposal_preflight` and assert no physical
slot reservation occurs. Inject failure at `proposal_materialized` and
assert every new local transaction is aborted in reverse order before
`register_batch()`.

Add parameterized Q=1..4 tests on ranks 0..3:

```python
@pytest.mark.parametrize("exact_q", (1, 2, 3, 4))
@pytest.mark.parametrize("rank", (0, 1, 2, 3))
def test_tp4_exact_q_preserves_q_minus_one_staged_entries(
    exact_q,
    rank,
):
    executor, _, cache = _ready_executor(
        prompts={1: ((2, 3), (0, 1))},
        rank=rank,
        world_size=4,
        coordinator=_RecordingCoordinator(),
    )
    proposal = executor.propose_batch((
        _proposal_input(
            1,
            first_target_token=4,
            exact_q=exact_q,
        ),
    ))[0]
    assert proposal.metadata["staged_entry_count"] == exact_q - 1
    transaction = cache.transaction(
        proposal.proposal_transaction_id
    )
    assert len(transaction.staged_slot_ids) == exact_q - 1
```

Add a mixed-Q batch test proving stable input order on every rank and one
backend decode call per active step. Fixture token broadcasts alone must not
increment `local_decode_forward_count`.

- [x] **Step 6: Run complete executor tests and verify GREEN**

Run:

```bash
uv run --offline --python 3.12 --with pytest --with torch \
  pytest -q tools/test_autoregressive_draft_executor.py
```

Expected: all TP1 and new TP4 executor tests pass.

---

### Task 5: Synchronized Bootstrap, Finalize, Rollback, and Release

**Files:**
- Modify: `tinyvllm/engine/autoregressive_draft_executor.py`
- Modify: `tools/test_autoregressive_draft_executor.py`

**Interfaces:**
- Consumes: Task 1 stage convergence and existing local
  `ProposalKVLifecycleCoordinator`.
- Produces: all-rank bootstrap/finalize/release logical authority with
  pre-commit cleanup and explicit poisoned post-commit boundaries.

- [ ] **Step 1: Write failing bootstrap convergence tests**

Add tests for:

```python
def test_bootstrap_preflight_mismatch_allocates_no_slots():
    executor, _, cache = _ready_executor(
        coordinator=_FailingCoordinator("bootstrap_preflight"),
    )
    with pytest.raises(RuntimeError, match="bootstrap_preflight"):
        executor.propose_batch((
            _proposal_input(
                1,
                first_target_token=4,
                exact_q=4,
            ),
        ))
    assert cache.authority_snapshot()["active_transaction_count"] == 0


def test_peer_bootstrap_prepare_failure_rolls_back_successful_local_rank():
    executor, _, cache = _ready_executor(
        coordinator=_FailingCoordinator("bootstrap_prepared"),
    )
    with pytest.raises(RuntimeError, match="bootstrap_prepared"):
        executor.propose_batch((
            _proposal_input(
                1,
                first_target_token=4,
                exact_q=4,
            ),
        ))
    assert cache.committed_length(1) == 0
    assert cache.authority_snapshot()["active_ticket_count"] == 0
```

Record physical store releases and assert reverse cleanup order for a
multi-sequence bootstrap.

- [ ] **Step 2: Run bootstrap tests and verify RED**

Run:

```bash
uv run --offline --python 3.12 --with pytest --with torch \
  pytest -q tools/test_autoregressive_draft_executor.py \
  -k 'bootstrap_preflight or bootstrap_prepare_failure'
```

Expected: slots are allocated before authority or successful peers do not
observe the injected peer failure.

- [x] **Step 3: Implement bootstrap stage ordering**

Before `ProposalKVCache.begin()`, compare `bootstrap_preflight` rows with:

```python
{
    "sequence_id": pending.sequence_id,
    "sequence_epoch": pending.sequence_epoch,
    "prompt_token_ids": pending.token_ids,
    "prompt_positions": pending.positions,
    "final_chunk_seen": pending.is_final,
}
```

Converge these fallible phases:

1. `bootstrap_begin`
2. `bootstrap_prefill`
3. `bootstrap_materialize`
4. `bootstrap_prepare`
5. `bootstrap_prepared`

The `bootstrap_prepared` logical rows contain sequence, epoch, prompt count,
and `logical_state="prepared"`. If any pre-commit phase fails, rollback
prepared tickets and abort unprepared transactions in reverse order on every
successful rank.

After local commit, compare `bootstrap_committed` rows containing sequence,
epoch, prompt count, committed logical length, and
`logical_state="committed"`. Any local commit failure or post-commit digest
mismatch raises:

```text
autoregressive draft runtime poisoned after bootstrap commit
```

Do not attempt synthetic cross-rank partial-commit recovery.

- [ ] **Step 4: Write failing finalize and release convergence tests**

Add tests covering:

- `finalize_preflight` mismatch calls no local prepare.
- one-rank local prepare failure makes successful peers rollback prepared
  tickets and abort still-owned proposal transactions;
- `finalize_prepared` mismatch leaves zero prepared tickets;
- local commit failure and `finalize_committed` mismatch are classified as
  poisoned runtime boundaries;
- rollback success compares `finalize_rolled_back` logical state and retains
  exactly prior committed entries;
- rollback peer failure is poisoned;
- `release_preflight` mismatch preserves local sequence state;
- successful release requires zero active transactions, zero prepared
  tickets, zero committed logical entries, and zero live physical slots;
- peer release failure is poisoned and the sequence is not advertised as
  reusable.

- [x] **Step 5: Implement lifecycle stage rows**

For prepare, compare:

```python
{
    "batch_index": batch_index,
    "sequence_id": row.sequence_id,
    "sequence_epoch": transaction.sequence_epoch,
    "exact_q": self._proposal_exact_q_by_transaction[
        row.proposal_transaction_id
    ],
    "proposal_token_ids": proposal.token_ids,
    "accepted_proposal_tokens": row.accepted_proposal_tokens,
    "committed_proposal_entries": max(
        row.accepted_proposal_tokens - 1,
        0,
    ),
}
```

Use stage names exactly:

```text
finalize_preflight
finalize_prepare
finalize_prepared
finalize_commit
finalize_committed
finalize_rollback
finalize_rolled_back
release_preflight
release_local
release_complete
```

`release_complete` rows include:

```python
{
    "sequence_id": sequence_id,
    "sequence_epoch": sequence_epoch,
    "active_transaction_count": 0,
    "active_ticket_count": 0,
    "committed_logical_entries": 0,
    "live_local_slot_count": 0,
}
```

Never include local transaction IDs or slot IDs in these rows.

- [x] **Step 6: Extend executor authority evidence**

Record a bounded tuple of logical stage rows and publish:

```python
{
    "rank": self.tensor_parallel_rank,
    "world_size": self.tensor_parallel_size,
    "logical_authority_rows": tuple(self._logical_authority_rows),
    "logical_authority_digest_count": len(
        self._logical_authority_digests
    ),
    "last_logical_authority_sha256": (
        None
        if not self._logical_authority_digests
        else self._logical_authority_digests[-1]
    ),
}
```

Keep physical store evidence rank-local. Ensure `assert_tensor_free()` still
passes.

- [x] **Step 7: Run complete executor tests and verify GREEN**

Run the Task 4 complete executor command.

Expected: TP1 behavior remains green and all synchronized lifecycle tests
pass.

---

### Task 6: Private Registration Candidates and All-Rank Identity Consensus

**Files:**
- Modify: `tinyvllm/engine/autoregressive_draft_registration.py`
- Modify: `tools/test_autoregressive_draft_registration.py`
- Modify: `tools/test_autoregressive_draft_model_runner_integration.py`

**Interfaces:**
- Consumes: checkpoint/tokenizer fingerprints, locally constructed Qwen3
  shard objects, descriptor, and registry preflight.
- Produces: `AutoregressiveDraftRegistrationCandidate`, deterministic
  capability hash, rank status, consensus validation, and no-publication
  failure behavior.

- [ ] **Step 1: Write failing candidate and identity tests**

Add:

```python
def test_registration_candidate_preserves_private_objects():
    candidate = AutoregressiveDraftRegistrationCandidate(
        target_checkpoint=_checkpoint("target"),
        draft_checkpoint=_checkpoint("draft"),
        target_tokenizer_contract=_tokenizer("target-tokenizer"),
        draft_tokenizer_contract=_tokenizer("draft-tokenizer"),
        model=object(),
        physical_store=object(),
        proposal_kv_cache=object(),
        backend=object(),
        executor=_FakeExecutor(),
        descriptor=ModelRunnerProposalExecutorDescriptor(
            executor_id="autoregressive-draft",
            capabilities=_FakeExecutor.capabilities,
        ),
    )
    assert candidate.descriptor.executor_id == "autoregressive-draft"


def test_registration_status_hashes_capabilities_deterministically():
    first = build_autoregressive_draft_registration_status(
        rank=0,
        world_size=4,
        stage="ready",
        candidate=_candidate(),
        error=None,
    )
    second = build_autoregressive_draft_registration_status(
        rank=3,
        world_size=4,
        stage="ready",
        candidate=_candidate(),
        error=None,
    )
    assert first.capabilities_sha256 == second.capabilities_sha256
```

Add tests that failed status rows contain no candidate identities from
partially initialized objects and retain exact stage/error type/message.

- [ ] **Step 2: Run registration tests and verify RED**

Run:

```bash
uv run --offline --python 3.12 --with pytest --with torch \
  pytest -q tools/test_autoregressive_draft_registration.py \
  -k 'candidate or registration_status'
```

Expected: imports fail because candidate and status helpers do not exist.

- [x] **Step 3: Add candidate and status helpers**

Add:

```python
@dataclass(frozen=True)
class AutoregressiveDraftRegistrationCandidate:
    target_checkpoint: CheckpointFingerprint
    draft_checkpoint: CheckpointFingerprint
    target_tokenizer_contract: TokenizerContract
    draft_tokenizer_contract: TokenizerContract
    model: object
    physical_store: object
    proposal_kv_cache: object
    backend: object
    executor: object
    descriptor: object
```

Add a deterministic capability hash based on `asdict(capabilities)` through
the existing canonical hashing helper. Add:

```python
def build_autoregressive_draft_registration_status(
    *,
    rank: int,
    world_size: int,
    stage: str,
    candidate: AutoregressiveDraftRegistrationCandidate | None,
    error: BaseException | None,
) -> AutoregressiveDraftRankRegistrationStatus:
    if error is not None:
        return AutoregressiveDraftRankRegistrationStatus(
            rank=rank,
            world_size=world_size,
            success=False,
            stage=stage,
            error_type=type(error).__name__,
            message=str(error),
            target_checkpoint_sha256=None,
            draft_checkpoint_sha256=None,
            target_tokenizer_sha256=None,
            draft_tokenizer_sha256=None,
            backend_identity=None,
            executor_id=None,
            capabilities_sha256=None,
        )
    if candidate is None:
        raise ValueError(
            "successful registration status requires candidate"
        )
    return AutoregressiveDraftRankRegistrationStatus(
        rank=rank,
        world_size=world_size,
        success=True,
        stage=stage,
        error_type=None,
        message=None,
        target_checkpoint_sha256=(
            candidate.target_checkpoint.composite_sha256
        ),
        draft_checkpoint_sha256=(
            candidate.draft_checkpoint.composite_sha256
        ),
        target_tokenizer_sha256=(
            candidate.target_tokenizer_contract.composite_sha256
        ),
        draft_tokenizer_sha256=(
            candidate.draft_tokenizer_contract.composite_sha256
        ),
        backend_identity=candidate.backend.backend_identity,
        executor_id=candidate.descriptor.executor_id,
        capabilities_sha256=_hash_payload(
            asdict(candidate.descriptor.capabilities)
        ),
    )
```

Success rows publish only logical SHA-256 identities. Failure rows publish
rank, world size, `success=False`, stage, error type, and message; identity
fields are `None`.

- [ ] **Step 4: Write failing all-rank consensus tests**

Add a pure validator:

```python
def test_matching_four_rank_statuses_return_one_consensus_hash():
    statuses = tuple(_successful_status(rank) for rank in range(4))
    consensus = validate_autoregressive_draft_registration_consensus(
        statuses,
        world_size=4,
    )
    assert len(consensus) == 64


@pytest.mark.parametrize(
    "field",
    (
        "target_checkpoint_sha256",
        "draft_checkpoint_sha256",
        "target_tokenizer_sha256",
        "draft_tokenizer_sha256",
        "backend_identity",
        "executor_id",
        "capabilities_sha256",
    ),
)
def test_one_rank_identity_mismatch_rejects_consensus(field):
    statuses = list(_successful_status(rank) for rank in range(4))
    statuses[2] = replace(statuses[2], **{field: "mismatch"})
    with pytest.raises(RuntimeError, match=field):
        validate_autoregressive_draft_registration_consensus(
            tuple(statuses),
            world_size=4,
        )
```

Add failure-row tests that name the failing rank and stage.

- [x] **Step 5: Implement consensus validation**

Require exactly `world_size` statuses, ranks `0..world_size-1` once each,
matching world sizes, all success bits, and identical logical identity
fields. Return SHA-256 of the canonical tuple of matching status identity
fields as `registration_consensus_sha256`.

- [x] **Step 6: Extend dependency construction for TP shard topology**

Change registration dependencies so `build_model` and backend construction
can receive rank and world size:

```python
build_model(draft_hf_config, *, tensor_parallel_rank, tensor_parallel_size)
build_backend(
    model=draft_model,
    proposal_kv_cache=proposal_kv_cache,
    backend_identity=config.autoregressive_draft_backend,
    model_fingerprint=draft_checkpoint.composite_sha256,
    tokenizer_fingerprint=(
        draft_tokenizer_contract.composite_sha256
    ),
    tensor_parallel_rank=tensor_parallel_rank,
    tensor_parallel_size=tensor_parallel_size,
)
```

Construct `Qwen3ForCausalLM` inside the existing ModelRunner distributed
environment so its tensor-parallel layers observe the current rank and world
size. Do not create a second process group.

- [x] **Step 7: Run complete registration tests and verify GREEN**

Run:

```bash
uv run --offline --python 3.12 --with pytest --with torch \
  pytest -q \
  tools/test_autoregressive_draft_registration.py \
  tools/test_autoregressive_draft_model_runner_integration.py \
  -k 'candidate or registration_status or consensus or registry_preflight'
```

Expected: all private-candidate and consensus tests pass without model or
checkpoint loading.

---

### Task 7: Failure-Atomic ModelRunner TP4 Registration and Authority

**Files:**
- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_autoregressive_draft_model_runner_integration.py`

**Interfaces:**
- Consumes: Task 1 coordinator, Task 2 registry preflight, and Task 6
  candidate/status/consensus helpers.
- Produces: all-rank private construction, publication after consensus,
  closed-feature failure state, root-only fused proposal return, and
  rank-local authority snapshots.

- [ ] **Step 1: Replace the old TP4 fail-closed test with topology tests**

Change the existing test that expects TP4 rejection. Add:

```python
@pytest.mark.parametrize("tensor_parallel_size", (2, 3, 5, 8))
def test_enabled_unsupported_tp_fails_before_dependencies_are_called(
    tmp_path,
    tensor_parallel_size,
):
    runner = _runner(
        tmp_path,
        tensor_parallel_size=tensor_parallel_size,
    )
    dependencies = _Dependencies()
    with pytest.raises(RuntimeError, match="TP1 or TP4"):
        runner._maybe_register_autoregressive_draft_executor(
            registration_dependencies=dependencies,
        )
    assert dependencies.calls == []


@pytest.mark.parametrize("rank", (0, 1, 2, 3))
def test_tp4_privately_constructs_local_candidate_before_publication(
    tmp_path,
    rank,
):
    runner = _runner(
        tmp_path,
        tensor_parallel_size=4,
        rank=rank,
    )
    dependencies = _Dependencies(rank=rank, world_size=4)
    coordinator = _RegistrationCoordinator.matching(rank)

    descriptor = runner._maybe_register_autoregressive_draft_executor(
        registration_dependencies=dependencies,
        tensor_parallel_coordinator=coordinator,
    )

    assert descriptor.executor_id == "autoregressive-draft"
    assert dependencies.calls.index("registry_preflight") < (
        dependencies.calls.index("collect_registration_status")
    )
    assert dependencies.calls.index("collect_registration_status") < (
        dependencies.calls.index("register_executor")
    )
```

Make `_runner()` accept rank separately from world size.

- [ ] **Step 2: Run ModelRunner TP4 registration tests and verify RED**

Run:

```bash
uv run --offline --python 3.12 --with pytest --with torch \
  pytest -q tools/test_autoregressive_draft_model_runner_integration.py \
  -k 'unsupported_tp or privately_constructs'
```

Expected: TP4 still raises `autoregressive draft currently requires TP1`.

- [x] **Step 3: Permit TP1/TP4 in Config and ModelRunner**

In `Config.__post_init__()`, when autoregressive draft is enabled, require:

```python
if self.tensor_parallel_size not in (1, 4):
    raise ValueError(
        "autoregressive draft requires tensor_parallel_size 1 or 4"
    )
```

In ModelRunner, require:

```python
if (
    tensor_parallel_size not in (1, 4)
    or self.world_size != tensor_parallel_size
    or self.rank < 0
    or self.rank >= self.world_size
):
    raise RuntimeError(
        "autoregressive draft requires matching TP1 or TP4 topology"
    )
```

Remove only the two obsolete TP1-only messages after the focused tests for
all replacement behavior are present.

- [x] **Step 4: Refactor registration into private construction**

Add an optional coordinator parameter:

```python
tensor_parallel_coordinator=None,
```

Construct a default
`AutoregressiveDraftTensorParallelCoordinator(rank=self.rank,
world_size=self.world_size, device=target_device)` when one is not injected.

Keep local construction stage-attributed. Do not write any
`self.autoregressive_draft_*` publication field during construction. Build
one `AutoregressiveDraftRegistrationCandidate`, then call:

```python
self.speculative_proposal_executors.preflight_registration(
    candidate.descriptor.executor_id,
    candidate.executor,
    candidate.descriptor.capabilities,
)
```

Build local status, gather statuses, and validate consensus. Only then call
`register()` and expose local candidate fields.

- [ ] **Step 5: Add failing no-publication matrix**

Parameterize every predictable local stage:

```text
fingerprint_target_checkpoint
fingerprint_draft_checkpoint
load_target_tokenizer
build_target_tokenizer_contract
load_draft_tokenizer
build_draft_tokenizer_contract
validate_tokenizer_compatibility
load_draft_hf_config
build_draft_model
load_draft_weights
move_and_eval_draft_model
build_proposal_physical_store
build_proposal_kv_cache
build_qwen3_draft_backend
build_autoregressive_draft_executor
build_executor_descriptor
registry_preflight
registration_consensus
```

For each injected one-rank failure, assert:

```python
assert registry.lifecycle_executor_ids() == existing_ids
assert runner.autoregressive_draft_model is None
assert runner.autoregressive_draft_physical_store is None
assert runner.autoregressive_draft_executor is None
assert runner.autoregressive_draft_executor_descriptor is None
assert runner.autoregressive_draft_checkpoint_identity is None
assert runner.autoregressive_draft_tokenizer_contract is None
```

The stored registration error must name the failing rank and stage on every
simulated rank.

- [x] **Step 6: Implement all-rank failure handling and publication**

On local construction failure, build a failed status rather than returning
early. Gather statuses on every rank. If consensus fails:

- discard the successful local candidate reference;
- keep every publication field `None`;
- store the common all-rank error;
- do not call registry `register()`;
- return `None`.

On consensus success:

1. call registry `register()` locally;
2. expose local model/store/cache/backend/executor/descriptor;
3. expose checkpoint and tokenizer contracts;
4. expose `registration_consensus_sha256`;
5. clear registration error;
6. return the descriptor.

- [x] **Step 7: Extend ModelRunner authority snapshot**

Publish:

```python
{
    "rank": self.rank,
    "world_size": self.world_size,
    "registration_consensus_sha256": (
        self.autoregressive_draft_registration_consensus_sha256
    ),
    "checkpoint_identity": {
        name: asdict(identity)
        for name, identity
        in self.autoregressive_draft_checkpoint_identity.items()
    },
    "tokenizer_contract": {
        name: asdict(contract)
        for name, contract
        in self.autoregressive_draft_tokenizer_contract.items()
    },
    "executor": executor.authority_snapshot(),
}
```

Require executor snapshot topology to match ModelRunner rank and world size.
Retain exact target/draft checkpoint and tokenizer composite hashes on every
rank. Keep the snapshot tensor-free.

- [x] **Step 8: Verify root-only fused proposal return**

Add an AST-shell or direct fixture around
`run_spec_first_target_and_proposal_batch()` proving:

- every rank calls the registered proposal executor;
- all ranks receive identical selected target token IDs;
- rank zero returns fused `FirstTargetProposalResult` rows;
- ranks 1, 2, and 3 return `None`;
- no Engine or Scheduler return type changes.

- [x] **Step 9: Run complete integration tests and verify GREEN**

Run:

```bash
uv run --offline --python 3.12 --with pytest --with torch \
  pytest -q tools/test_autoregressive_draft_model_runner_integration.py
```

Expected: TP1 registration order remains valid, TP4 all-rank registration
passes injected consensus, and every failure path publishes on no rank.

### 2026-08-15 Task 1-7 Fresh Reconciliation

Tasks 1-7 were reconciled against the current source and tests without
running Task 8 or any GPU, remote, NCCL, real-checkpoint, or performance
workload.

Fresh focused verification:

```text
tools/test_autoregressive_draft_tp.py
tools/test_autoregressive_draft_executor.py
tools/test_qwen3_draft_backend.py
tools/test_autoregressive_draft_registration.py
tools/test_autoregressive_draft_model_runner_integration.py
tools/test_speculative_model_runner_callbacks.py
tools/test_tensor_parallel_greedy.py

271 passed in 5.95s
```

The neighboring source-neutral regression files cannot safely share one
pytest process because dependency-light tests install incompatible
`sys.modules` stubs. The combined collection failed when the stubbed
`tinyvllm.utils.context` lacked `temporary_context`. Running each file in an
isolated Python process produced:

```text
tools/test_proposal_kv_cache.py:                       17 passed
tools/test_proposal_kv_lifecycle.py:                   11 passed
tools/test_speculative_runtime.py:                     14 passed
tools/test_speculative_batch_runtime.py:               36 passed
tools/test_speculative_selection_record.py:            28 passed
tools/test_speculative_side_state.py:                  20 passed
tools/test_ngram_speculative.py:                       59 passed
tools/test_qwen35_mtp_executor.py:                     24 passed
tools/test_qwen35_mtp_model_runner_integration.py:     15 passed
total:                                                224 passed
```

Production `tools/test_kv_offload.py` still does not collect in the offline
Python 3.12 + Torch 2.12 environment:

```text
ModuleNotFoundError: No module named 'flash_attn'
```

Static checks:

```text
Task 1-7 source/test py_compile:             PASS
obsolete TP1-only message scan:              PASS
repo-global git diff --check:                PASS
staged diff:                                 empty
```

Checkbox policy follows the same evidence rule as the paired-trace
reconciliation: current implementation and freshly observed GREEN steps are
checked, while historical RED/test-first steps whose original failure was
not observed in this continuation remain unchecked. Therefore:

```text
Tasks 1-7 implementation/GREEN steps checked: 24
Tasks 1-7 historical RED steps unchecked:     21
Task 8 steps unchecked:                        8
whole-plan checked:                           24
whole-plan unchecked:                         29
```

This establishes local TP4 implementation, topology contracts, logical
authority convergence, rank-local storage evidence, lifecycle convergence,
failure-atomic registration, and root-only fused return. It does not
establish a real TP4 checkpoint run, 4K/16K/32K Engine parity, real
Proposal-KV H2D/D2H, performance improvement, Phase 1 completion, or
promotion.

---

### Task 8: Local Four-Rank Evidence Gate and Regression Boundary

**Files:**
- Create: `tools/test_autoregressive_draft_tp4_local_gate.py`
- Modify: `tools/test_autoregressive_draft_tp1_engine_gate.py`
- Modify: `tools/test_autoregressive_draft_model_runner_integration.py`

**Interfaces:**
- Consumes: four rank-local authority snapshots produced by Tasks 3, 5, and
  7.
- Produces: a dependency-light local TP4 aggregate validator and explicit
  non-promotion classification.

- [ ] **Step 1: Write the local aggregate contract**

Create
`validate_autoregressive_draft_tp4_local_evidence(snapshots: tuple[dict, ...]) -> dict`.

Tests must require:

```python
def test_local_gate_requires_exactly_four_distinct_rank_snapshots():
    with pytest.raises(ValueError, match="exactly four"):
        validate_autoregressive_draft_tp4_local_evidence(
            tuple(_rank_snapshot(rank) for rank in range(3))
        )


def test_local_gate_accepts_rank_local_physical_identity_differences():
    snapshots = tuple(_rank_snapshot(rank) for rank in range(4))
    aggregate = validate_autoregressive_draft_tp4_local_evidence(
        snapshots
    )
    assert aggregate["rank_count"] == 4
    assert aggregate["total_proposal_kv_bytes"] == sum(
        row["executor"]["backend"]["local_proposal_kv_bytes"]
        for row in snapshots
    )
    assert aggregate["classification"] == "NOT_PROMOTABLE"
```

- [ ] **Step 2: Run the local gate test and verify RED**

Run:

```bash
uv run --offline --python 3.12 --with pytest --with torch \
  pytest -q tools/test_autoregressive_draft_tp4_local_gate.py
```

Expected: collection fails because the validator file does not exist.

- [ ] **Step 3: Implement the aggregate validator**

Require:

- exactly four snapshots;
- ranks `{0, 1, 2, 3}` exactly once;
- `world_size == 4` on every rank;
- identical non-empty registration consensus hash;
- identical target/draft checkpoint composite hashes;
- identical target/draft tokenizer composite hashes;
- identical backend identity, executor ID, and logical capabilities;
- `local_prefill_forward_count > 0` on every rank;
- `local_decode_forward_count > 0` on every rank;
- logical authority digest agreement for matched stage rows;
- four distinct proposal-KV storage identities;
- zero active transactions;
- zero active prepared tickets;
- zero live proposal slots after release.

Compute total proposal-KV bytes as the sum of rank-local bytes. Compute stage
latency as the maximum rank-local latency, never the sum. Return:

```python
{
    "schema_version": 1,
    "rank_count": 4,
    "registration_consensus_sha256": consensus,
    "total_proposal_kv_bytes": total_bytes,
    "max_rank_timing_ms": max_timing,
    "classification": "NOT_PROMOTABLE",
    "promotion_boundary": {
        "real_checkpoint_tp4": "NOT_ESTABLISHED",
        "second_learned_structure": "NOT_ESTABLISHED",
        "contexts_4k_16k_32k": "NOT_ESTABLISHED",
        "performance": "NOT_ESTABLISHED",
        "real_kv_movement": "NOT_ESTABLISHED",
        "phase_1": "NOT_ACHIEVED",
    },
}
```

- [ ] **Step 4: Add source-boundary regression checks**

Extend the generic-runtime source test so these files contain no Qwen3 draft
special case:

```text
tinyvllm/engine/llm_engine.py
tinyvllm/engine/scheduler.py
tinyvllm/speculative/verifier.py
tinyvllm/engine/proposal_kv_cache.py
tinyvllm/engine/proposal_kv_lifecycle.py
tinyvllm/engine/speculative_side_state.py
```

Permit generic registry and ModelRunner integration references only where
the source-neutral executor boundary already exists.

- [ ] **Step 5: Preserve TP1 gate classification**

Add assertions to `tools/test_autoregressive_draft_tp1_engine_gate.py` that
the existing gate remains TP1, requires exact token parity, requires
physical proposal-KV evidence, and does not claim TP4 or Phase 1 completion.

- [ ] **Step 6: Run the focused TP4 extension matrix**

Run:

```bash
uv run --offline --python 3.12 --with pytest --with torch \
  pytest -q \
  tools/test_autoregressive_draft_tp.py \
  tools/test_autoregressive_draft_executor.py \
  tools/test_qwen3_draft_backend.py \
  tools/test_autoregressive_draft_registration.py \
  tools/test_autoregressive_draft_model_runner_integration.py \
  tools/test_autoregressive_draft_tp1_engine_gate.py \
  tools/test_autoregressive_draft_tp4_local_gate.py \
  tools/test_tensor_parallel_greedy.py
```

Expected: all tests pass. This is the first point where both obsolete
TP1-only fail-closed messages may be absent.

- [ ] **Step 7: Run neighboring speculative regression tests**

Run:

```bash
uv run --offline --python 3.12 --with pytest --with torch \
  pytest -q \
  tools/test_proposal_kv_cache.py \
  tools/test_proposal_kv_lifecycle.py \
  tools/test_speculative_runtime.py \
  tools/test_speculative_batch_runtime.py \
  tools/test_speculative_selection_record.py \
  tools/test_speculative_side_state.py \
  tools/test_kv_offload.py \
  tools/test_ngram_speculative.py \
  tools/test_qwen35_mtp_executor.py \
  tools/test_qwen35_mtp_model_runner_integration.py
```

Expected: all selected generic, target-KV, offload, n-gram, and native-MTP
regressions pass. Any unrelated pre-existing failure is recorded verbatim
and is not silently classified as TP4 success.

- [ ] **Step 8: Run static plan-boundary verification**

Run:

```bash
rg -n \
  'autoregressive draft executor currently requires TP1|autoregressive draft currently requires TP1' \
  tinyvllm tools
git diff --check
git status --short
```

Expected:

- no obsolete TP1-only message remains;
- `git diff --check` exits zero;
- only intended files from this plan plus pre-existing dirty files appear;
- no file is staged, committed, pushed, stashed, reset, or cleaned.

---

## Real TP4 Checkpoint Gate Boundary

Do not execute a remote or GPU campaign as part of this implementation plan.
After the local matrix is green and separate authorization is received, a
new plan must create and run the real TP4 gate with all of these mandatory
conditions:

- source-attributed immutable Qwen3 draft checkpoint hashes;
- source-attributed immutable Qwen3.5 target checkpoint hashes;
- exact tokenizer and ordered token-to-ID compatibility;
- TP4 BF16;
- greedy temperature zero;
- `MAX_PROPOSAL_TOKENS=4`;
- 4K context;
- batch 1, batch 4, and true multi-sequence execution;
- target-only and learned-draft Engine runs over identical prompts;
- exact output-token parity for every sequence;
- all-rank registration consensus;
- nonzero real draft prefill and decode forward counts on every rank;
- all-rank logical proposal and lifecycle digest agreement;
- accepted-prefix proposal-KV commit evidence;
- rejected-suffix rollback evidence;
- zero proposal-KV leaks after release;
- no extra target forward beyond the generic speculative contract;
- separate target-KV and proposal-KV byte accounting;
- no claim that simulated copy counters represent real H2D movement.

The real gate fails if a non-root rank has zero real draft forwards, proposal
tokens come from fixtures, only rank-zero authority is present, or physical
proposal K/V is gathered across ranks. The 16K and 32K learned-draft
campaigns remain separate work after the 4K gate.

## Spec Coverage Matrix

| Approved spec requirement | Plan evidence |
|---|---|
| Sharded Qwen3 drafter on every target rank | Tasks 3, 6, and 7 construct topology-aware Qwen3 backend/model objects on ranks 0 through 3. |
| Rank-local proposal K/V and physical identity independence | Tasks 3, 4, 5, and 8 derive local KV geometry, exclude physical IDs from digests, and require distinct local stores. |
| Root-only full logits and compact token broadcast | Tasks 3 and 4 preserve `ParallelLMHead` root gathering, validate non-root `None`, call exact `argmax` through `select_tensor_parallel_greedy_tokens()`, and broadcast only `torch.int64[batch]`. |
| Fixed-size runtime authority consensus | Task 1 implements the `torch.uint8[33]` success-bit plus SHA-256 protocol. |
| Failure convergence before rank progress | Tasks 1, 4, and 5 capture local errors, converge stages, clean successful peers, and raise common stage errors. |
| Bootstrap preflight/prepare/commit authority | Task 5 covers `bootstrap_preflight`, `bootstrap_prepared`, and `bootstrap_committed`. |
| Proposal preflight/materialized authority | Task 4 covers `proposal_preflight`, per-step convergence, and `proposal_materialized`. |
| Finalize prepare/commit/rollback authority | Task 5 covers all finalize preflight, prepared, committed, and rolled-back rows. |
| Sequence release authority and zero leaks | Tasks 5 and 8 require release preflight, release completion, zero active state, and zero live local slots. |
| Failure-atomic all-rank registration | Tasks 2, 6, and 7 add mutation-free preflight, private candidates, structured status collection, consensus, and post-consensus publication. |
| TP1 remains unchanged | Tasks 3 through 8 retain TP1 paths and rerun the TP1 executor/backend/registration/Engine tests. |
| TP sizes other than 1 or 4 fail closed | Tasks 1, 4, and 7 test and enforce the exact supported topology set. |
| Q=1 through Q=4 and mixed-Q stable order | Task 4 adds parameterized exact-Q and mixed-batch tests. |
| Every rank performs real backend forwards | Tasks 3, 4, and 8 require nonzero local prefill/decode counters and reject fixture-only evidence. |
| Generic verifier, target-KV, side-state, scheduler, offload, n-gram, SAM, and native-MTP semantics remain unchanged | Task 8 runs source-boundary checks and neighboring regression suites. |
| Real TP4 4K batch 1/4/multi-sequence gate | The Real TP4 Checkpoint Gate Boundary lists the exact separately authorized gate; local completion remains `NOT_PROMOTABLE`. |
| 16K/32K, performance, and real H2D movement are not overclaimed | The Real TP4 Checkpoint Gate Boundary and Completion Classification keep each item `NOT_ESTABLISHED`. |

## Completion Classification

When every local task in this plan is green, report exactly:

```text
TP4_SHARDED_DRAFT_LOCAL_CONTRACT=ESTABLISHED
TP4_FAILURE_ATOMIC_REGISTRATION_LOCAL_CONTRACT=ESTABLISHED
TP4_LOGICAL_LIFECYCLE_AUTHORITY_LOCAL_CONTRACT=ESTABLISHED
SECOND_LEARNED_STRUCTURE=NOT_ESTABLISHED
SECOND_LEARNED_STRUCTURE_TP1_CORRECTNESS=NOT_ESTABLISHED
TP4_INDEPENDENT_DRAFT_REAL_CHECKPOINT=NOT_ESTABLISHED
TP4_4K_ENGINE_PARITY=NOT_ESTABLISHED
TP4_16K_ENGINE_PARITY=NOT_ESTABLISHED
TP4_32K_ENGINE_PARITY=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
REAL_KV_MOVEMENT_BENEFIT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```
