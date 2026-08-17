# Qwen3.5 Native MTP TP4/4K Engine Transactional Correctness Implementation Plan

> **For agentic workers:** Execute inline in this worktree. Do not dispatch subagents. Follow every task in order with strict RED -> GREEN validation.

**Goal:** Extend the real Qwen3.5 native MTP Engine path from TP1 to eager TP4 and establish a real-checkpoint 4K transactional correctness authority for batch 1 and 4.

**Architecture:** Every rank constructs its native MTP decoder shard and owns rank-local proposal KV. Rank 0 alone receives gathered LM-head logits and selects greedy token IDs; a shared tensor-only selector broadcasts `int64` token IDs to the other ranks. All ranks execute matching proposal, finalize, release, and cleanup lifecycles, while only rank 0 returns host-side proposals and generation results.

**Tech Stack:** Python 3, PyTorch distributed/NCCL, TinyLLMForge ModelRunner and generic speculative runtime, pytest, JSON authority artifacts, SSH remote execution on four GPUs.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, push, switch branches/worktrees, stash, reset, or clean.
- Do not use subagents.
- Every behavior change must follow RED -> minimal GREEN -> focused regression.
- Keep target and MTP execution eager for this authority.
- Keep target KV offload disabled.
- Keep MTP CUDA Graphs disabled and preserve `RuntimeError("Qwen3.5 MTP CUDA graphs require TP1")`.
- Broadcast only `torch.int64` token tensors; never broadcast logits, hidden states, Python objects, or proposal metadata.
- Rank 0 is the sole host token and `DraftProposal` authority.
- Every rank must execute proposal KV bootstrap, proposal, finalize, release, and cleanup.
- Accepted target prefixes must never be replayed by a second full target forward.
- Remote access must use `sitian@10.232.195.203`, `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`, `ControlMaster=no`, and `ControlPath=none`.
- Do not kill unrelated GPU processes.
- A passing gate remains `NOT_PROMOTABLE`.

---

### Task 1: Make Native MTP Construction TP-Aware

**Files:**
- Modify: `tools/test_qwen35_mtp.py`
- Modify: `tinyvllm/models/qwen35_mtp.py`

**Interfaces:**
- Consumes: `build_qwen35_full_attention_decoder_layer(...)` and `_distributed_construction_context(size, rank)`.
- Produces: `build_qwen35_native_mtp(..., tensor_parallel_size: int, tensor_parallel_rank: int, ...)` for valid TP1 and TP4 ranks.

- [x] **Step 1: Write the failing TP4 construction tests**

Add tests that monkeypatch the distributed construction context and component
factory, then call the real builder for ranks zero through three:

```python
@pytest.mark.parametrize("rank", range(4))
def test_builder_uses_requested_tp4_context_and_rank_local_backend(
    monkeypatch,
    rank,
):
    entered = []
    backend_rows = []

    @contextmanager
    def fake_context(size, context_rank):
        entered.append((size, context_rank))
        yield

    def fake_backend(layer_index, local_queries, local_keys, dimension):
        backend_rows.append(
            (layer_index, local_queries, local_keys, dimension)
        )
        return object()

    monkeypatch.setattr(
        qwen35_mtp,
        "_distributed_construction_context",
        fake_context,
    )
    module = qwen35_mtp.build_qwen35_native_mtp(
        _config(),
        embed_tokens=_shared_embedding(rank=rank, size=4),
        lm_head=_shared_lm_head(rank=rank, size=4),
        tensor_parallel_size=4,
        tensor_parallel_rank=rank,
        build_attention_backend=fake_backend,
        parameter_device="cpu",
    )

    assert entered == [(4, rank)]
    assert module.embed_tokens is module.lm_head
    assert backend_rows[0][0] == 0
```

Add invalid topology cases for rank `-1`, rank `4`, size `0`, non-integer
values, and geometry that is not divisible under the existing component
rules.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest -q tools/test_qwen35_mtp.py \
  -k 'tp4_context or invalid_tensor_parallel'
```

Expected: FAIL because the builder raises
`ValueError("Qwen3.5 native MTP first slice requires TP1")`.

- [x] **Step 3: Implement minimal TP-aware construction**

Replace the TP1 guard with exact TP validation and pass the requested topology
to the existing construction context:

```python
def _tensor_parallel_context(size: int, rank: int) -> tuple[int, int]:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(
            "tensor_parallel_size must be a positive integer"
        )
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 0
        or rank >= size
    ):
        raise ValueError(
            "tensor_parallel_rank must be in "
            "[0, tensor_parallel_size)"
        )
    return size, rank
```

Use:

```python
tensor_parallel_size, tensor_parallel_rank = (
    _tensor_parallel_context(
        tensor_parallel_size,
        tensor_parallel_rank,
    )
)
with _distributed_construction_context(
    tensor_parallel_size,
    tensor_parallel_rank,
), torch.device(device):
    ...
```

Do not shard `mtp.fc`; retain `ReplicatedLinear`.

- [x] **Step 4: Run focused and existing construction tests**

Run:

```bash
pytest -q tools/test_qwen35_mtp.py
```

Expected: PASS.

- [x] **Step 5: Record working-tree status without committing**

Run:

```bash
git diff --check -- \
  tinyvllm/models/qwen35_mtp.py \
  tools/test_qwen35_mtp.py
```

Expected: no output. Do not stage or commit.

### Task 2: Add the Tensor-Only TP Greedy Selector

**Files:**
- Create: `tinyvllm/engine/tensor_parallel_greedy.py`
- Create: `tools/test_tensor_parallel_greedy.py`

**Interfaces:**
- Produces:

```python
def select_tensor_parallel_greedy_tokens(
    logits: torch.Tensor | None,
    *,
    rank: int,
    world_size: int,
    batch_size: int,
    device: torch.device,
    broadcast=None,
) -> torch.Tensor:
    ...
```

- Returns identical contiguous `torch.int64[batch_size]` token IDs on all ranks.

- [x] **Step 1: Write selector RED tests**

Use an in-memory broadcast harness that first executes rank 0 and then copies
the recorded source tensor into worker-rank buffers:

```python
def test_tp4_rank0_selects_and_workers_receive_only_token_ids():
    bus = _BroadcastBus()
    logits = torch.tensor([
        [0.0, 5.0, 1.0],
        [7.0, 2.0, 3.0],
    ])

    root = select_tensor_parallel_greedy_tokens(
        logits,
        rank=0,
        world_size=4,
        batch_size=2,
        device=logits.device,
        broadcast=bus.root,
    )
    workers = tuple(
        select_tensor_parallel_greedy_tokens(
            None,
            rank=rank,
            world_size=4,
            batch_size=2,
            device=logits.device,
            broadcast=bus.worker,
        )
        for rank in (1, 2, 3)
    )

    assert root.tolist() == [1, 0]
    assert all(torch.equal(row, root) for row in workers)
    assert bus.payloads == [
        (torch.int64, (2,), [1, 0]),
    ]
```

Add tests that reject:

- non-root `logits` tensors;
- root `logits is None`;
- wrong logit rank or row count;
- non-floating logits;
- invalid rank/world size;
- malformed broadcast output;
- negative received token IDs; and
- TP1 behavior that invokes a broadcast.

- [ ] **Step 2: Run selector tests and verify RED**

Run:

```bash
pytest -q tools/test_tensor_parallel_greedy.py
```

Expected: collection failure because
`tinyvllm.engine.tensor_parallel_greedy` does not exist.

- [x] **Step 3: Implement the minimal selector**

Implement exact validation and default to `torch.distributed.broadcast` only
when `world_size > 1`:

```python
def select_tensor_parallel_greedy_tokens(
    logits,
    *,
    rank,
    world_size,
    batch_size,
    device,
    broadcast=None,
):
    rank, world_size = _validate_topology(rank, world_size)
    batch_size = _positive_integer(batch_size, "batch_size")
    device = torch.device(device)
    if rank == 0:
        _validate_root_logits(logits, batch_size, device)
        token_ids = logits.argmax(dim=-1).to(
            device=device,
            dtype=torch.int64,
        ).contiguous()
    else:
        if logits is not None:
            raise ValueError("non-root logits must be None")
        token_ids = torch.empty(
            batch_size,
            dtype=torch.int64,
            device=device,
        )
    if world_size > 1:
        operation = dist.broadcast if broadcast is None else broadcast
        operation(token_ids, src=0)
    _validate_token_ids(token_ids, batch_size, device)
    return token_ids
```

Do not add object collectives or vocabulary-sized payloads.

- [x] **Step 4: Run selector tests**

Run:

```bash
pytest -q tools/test_tensor_parallel_greedy.py
```

Expected: PASS.

- [x] **Step 5: Run static validation**

Run:

```bash
python -m py_compile \
  tinyvllm/engine/tensor_parallel_greedy.py \
  tools/test_tensor_parallel_greedy.py
git diff --check -- \
  tinyvllm/engine/tensor_parallel_greedy.py \
  tools/test_tensor_parallel_greedy.py
```

Expected: both commands succeed. Do not stage or commit.

### Task 3: Make the Native MTP Executor Rank-Aware

**Files:**
- Modify: `tinyvllm/engine/qwen35_mtp_executor.py`
- Modify: `tools/test_qwen35_mtp_executor.py`

**Interfaces:**
- Consumes: `select_tensor_parallel_greedy_tokens(...)`.
- Extends:

```python
Qwen35MTPProposalExecutor(
    *,
    module,
    proposal_kv_cache,
    max_proposal_tokens,
    graph_runner=None,
    tensor_parallel_rank=0,
    tensor_parallel_size=1,
    token_broadcast=None,
)
```

- [x] **Step 1: Write executor RED tests**

Add a fake MTP whose `forward_step()` returns full logits on rank 0 and
`None` on worker ranks. Run four executors against a shared test broadcast
bus and assert:

```python
assert proposals_by_rank[0].token_ids == (13, 14, 15, 16)
assert all(
    proposal.token_ids == proposals_by_rank[0].token_ids
    for proposal in proposals_by_rank[1:]
)
assert [
    proposal.proposal_transaction_id
    for proposal in proposals_by_rank
] == ["proposal-2"] * 4
assert bus.payload_shapes == [(1,), (1,), (1,)]
```

Add a worker-rank test where the fake MTP returns logits and require
`ValueError("non-root logits must be None")`. Add a root-rank test where the
fake MTP returns `None` and require a fail-closed error. Preserve all existing
TP1 executor tests unchanged.

- [ ] **Step 2: Run focused executor tests and verify RED**

Run:

```bash
pytest -q tools/test_qwen35_mtp_executor.py \
  -k 'tp4 or worker_logits or root_logits'
```

Expected: FAIL because the constructor has no TP arguments and `_run_proposal`
requires a tensor logits row on every rank.

- [x] **Step 3: Implement minimal executor integration**

Store validated TP topology and replace the local `argmax` block with:

```python
token_ids = select_tensor_parallel_greedy_tokens(
    logits,
    rank=self.tensor_parallel_rank,
    world_size=self.tensor_parallel_size,
    batch_size=1,
    device=current_hidden.device,
    broadcast=self.token_broadcast,
)
current_token = int(token_ids[0].item())
```

Retain hidden-output validation on every rank. Root validates full logits
through the selector; worker ranks require `None`.

Do not change proposal transaction allocation, registration, or finalization
order.

- [x] **Step 4: Run executor tests**

Run:

```bash
pytest -q tools/test_qwen35_mtp_executor.py
```

Expected: PASS.

- [x] **Step 5: Run proposal cache regressions**

Run:

```bash
pytest -q \
  tools/test_proposal_kv_cache.py \
  tools/test_qwen35_mtp_executor.py
```

Expected: PASS. Do not stage or commit.

### Task 4: Register One Native MTP Executor Per Rank

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_qwen35_mtp_model_runner_integration.py`

**Interfaces:**
- Consumes: rank-aware `build_qwen35_native_mtp()` and
  `Qwen35MTPProposalExecutor(...)`.
- Produces: descriptor `native_checkpoint_proposal` on every TP4 rank.

- [x] **Step 1: Write TP4 registration RED tests**

Parameterize the existing runner fixture by rank:

```python
@pytest.mark.parametrize("rank", range(4))
def test_tp4_registration_builds_rank_local_executor(rank):
    runner = _runner(tensor_parallel_size=4, rank=rank)
    dependencies = _Dependencies(runner.events)

    descriptor = _load_runner_method(
        "_maybe_register_qwen35_mtp_executor"
    )(
        runner,
        registration_dependencies=dependencies,
    )

    assert descriptor.executor_id == "native_checkpoint_proposal"
    assert dependencies.build_topology == [(4, rank)]
    assert dependencies.executor_topology == [(4, rank)]
    assert runner.qwen35_mtp_physical_store is dependencies.physical_store
```

Add tests that TP4 still refuses KV offload and TP4 graph mode, while TP1
registration remains unchanged.

- [ ] **Step 2: Run registration tests and verify RED**

Run:

```bash
pytest -q tools/test_qwen35_mtp_model_runner_integration.py \
  -k 'tp4_registration or tp4_graph or tp4_offload'
```

Expected: FAIL because registration returns `None` for world size four and
passes hard-coded `(1, 0)`.

- [x] **Step 3: Implement all-rank registration**

Change the eligibility check to require:

```python
tensor_parallel_size = getattr(
    config,
    "tensor_parallel_size",
    None,
)
if (
    tensor_parallel_size != self.world_size
    or self.rank < 0
    or self.rank >= self.world_size
    or getattr(config, "kv_offload_mvp0", False)
    ...
):
    return None
```

Pass:

```python
tensor_parallel_size=self.world_size,
tensor_parallel_rank=self.rank,
```

through module and executor construction. Keep graph runner creation disabled
for TP4; if `qwen35_mtp_cuda_graphs` is true, preserve the existing TP1-only
runtime error.

- [x] **Step 4: Run integration tests**

Run:

```bash
pytest -q tools/test_qwen35_mtp_model_runner_integration.py
```

Expected: PASS.

- [x] **Step 5: Run checkpoint binding regressions**

Run:

```bash
pytest -q \
  tools/test_qwen35_mtp_checkpoint.py \
  tools/test_qwen35_mtp_model_runner_integration.py
```

Expected: PASS. Do not stage or commit.

### Task 5: Execute First-Target and Proposal Work on Every Rank

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_model_runner_proposal_executor.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**
- Consumes: `select_tensor_parallel_greedy_tokens(...)`.
- Produces: all-rank proposal execution with rank-0-only
  `FirstTargetProposalResult` return.

- [x] **Step 1: Write all-rank ModelRunner RED tests**

Create AST-loaded method fixtures for ranks zero and one. Assert that rank 1:

1. receives `logits=None` from the fake TP LM head;
2. receives the rank-0 first-target token through the selector;
3. calls `speculative_proposal_executors.execute_batch(...)`;
4. passes its rank-local hidden row into the proposal input; and
5. returns `None` only after execution.

The root test asserts that it returns `FirstTargetProposalResult` rows and
that root and worker execute identical ordered sequence IDs.

Add a failure test where the worker proposal executor raises. The method must
propagate the error rather than returning early.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest -q \
  tools/test_model_runner_proposal_executor.py \
  tools/test_model_runner_spec_verify.py \
  -k 'all_rank or worker_proposal or tp4_first_target'
```

Expected: FAIL at the TP1-only guard or because rank 1 returns before proposal
execution.

- [x] **Step 3: Implement all-rank first-target selection and proposal**

Remove the world-size-one rejection. After `run_model(...)`, call:

```python
target_token_tensor = select_tensor_parallel_greedy_tokens(
    logits,
    rank=self.rank,
    world_size=self.world_size,
    batch_size=len(seqs),
    device=input_ids.device,
)
target_tokens = target_token_tensor.tolist()
```

Build and execute `proposal_inputs` on every rank. Keep `target_logits=None`
for non-root ranks and only support descriptors that do not require target
logits at TP4.

Move:

```python
if self.rank != 0:
    return None
```

to immediately after `execute_batch(...)`. Construct
`FirstTargetProposalResult` only on rank 0.

- [x] **Step 4: Run ModelRunner tests**

Run:

```bash
pytest -q \
  tools/test_model_runner_proposal_executor.py \
  tools/test_model_runner_spec_verify.py
```

Expected: PASS.

- [x] **Step 5: Run generic speculative callback regressions**

Run:

```bash
pytest -q \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_speculative_batch_runtime.py \
  tools/test_engine_speculative_runtime.py
```

Expected: PASS. Do not stage or commit.

### Task 6: Relax Runtime Activation Only for the Supported TP4 Contract

**Files:**
- Modify: `tinyvllm/engine/speculative_runtime.py`
- Modify: `tools/test_engine_speculative_runtime.py`

**Interfaces:**
- Produces: ModelRunner proposal runtime activation for world sizes `1` and
  `4` when KV offload is disabled and the descriptor does not require target
  logits.

- [x] **Step 1: Write activation RED tests**

Add:

```python
def test_model_runner_native_executor_allows_tp4_without_offload():
    runner = _runner(world_size=4, kv_offload_mvp0=False)
    runtime = _native_runtime(requires_target_logits=False)
    config = build_speculative_selection_config(runner, runtime)
    assert config.enabled is True


def test_model_runner_tp4_rejects_target_logit_requirement():
    runner = _runner(world_size=4, kv_offload_mvp0=False)
    runtime = _native_runtime(requires_target_logits=True)
    with pytest.raises(ValueError, match="target logits"):
        build_speculative_selection_config(runner, runtime)
```

Retain rejection for world size two, world size eight, and all offload-enabled
ModelRunner proposal execution.

- [ ] **Step 2: Run activation tests and verify RED**

Run:

```bash
pytest -q tools/test_engine_speculative_runtime.py \
  -k 'native_executor_allows_tp4 or tp4_rejects_target'
```

Expected: FAIL with
`ValueError("model runner proposal execution supports TP1 only")`.

- [x] **Step 3: Implement the bounded activation rule**

Use:

```python
world_size = getattr(model_runner, "world_size", None)
if world_size not in (1, 4):
    raise ValueError(
        "model runner proposal execution supports TP1 or TP4"
    )
if world_size > 1 and capabilities.requires_target_logits:
    raise ValueError(
        "TP4 model runner proposal execution cannot require "
        "target logits on worker ranks"
    )
```

Do not generalize to arbitrary world sizes in this gate.

- [x] **Step 4: Run runtime tests**

Run:

```bash
pytest -q tools/test_engine_speculative_runtime.py
```

Expected: PASS.

- [x] **Step 5: Run source-neutral runtime regressions**

Run:

```bash
pytest -q \
  tools/test_speculative_runtime.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_engine_speculative_execution.py
```

Expected: PASS. Do not stage or commit.

### Task 7: Add TP4 Rank and Transaction Evidence Surfaces

**Files:**
- Modify: `tinyvllm/engine/qwen35_mtp_executor.py`
- Modify: `tinyvllm/engine/proposal_kv_cache.py`
- Create: `tools/test_qwen35_mtp_tp4_rank_evidence.py`

**Interfaces:**
- Produces tensor-free executor snapshot:

```python
def tp4_authority_snapshot(self) -> dict:
    ...
```

- Snapshot contains rank topology, lifecycle counters, ordered token-broadcast
  digests, logical transaction rows, finalize rows, and leak counts.

- [x] **Step 1: Write snapshot RED tests**

After a proposal with partial acceptance and sequence release, require:

```python
snapshot = executor.tp4_authority_snapshot()
assert snapshot["tensor_parallel_rank"] == rank
assert snapshot["tensor_parallel_size"] == 4
assert snapshot["proposal_transactions"][0]["sequence_id"] == 7
assert snapshot["proposal_transactions"][0]["exact_q"] == 4
assert snapshot["proposal_transactions"][0]["accepted"] == 2
assert snapshot["proposal_transactions"][0]["rejected"] == 2
assert snapshot["active_transactions"] == 0
assert snapshot["prepared_tickets"] == 0
assert snapshot["pending_sequences"] == 0
assert snapshot["bootstrapped_sequences"] == 0
assert snapshot["allocated_physical_slots"] == 0
```

Run four deterministic executors and assert logical transaction and ticket
IDs match while physical slot IDs are validated per rank rather than compared
as global addresses.

- [ ] **Step 2: Run evidence tests and verify RED**

Run:

```bash
pytest -q tools/test_qwen35_mtp_tp4_rank_evidence.py
```

Expected: FAIL because `tp4_authority_snapshot()` does not exist.

- [x] **Step 3: Implement bounded tensor-free evidence**

Record only primitive values and tuples during existing lifecycle operations.
Do not expose tensors, module objects, physical store objects, or checkpoint
parameters. Return a fresh dictionary so the authority worker cannot mutate
executor state.

- [x] **Step 4: Run evidence and lifecycle tests**

Run:

```bash
pytest -q \
  tools/test_qwen35_mtp_tp4_rank_evidence.py \
  tools/test_qwen35_mtp_executor.py \
  tools/test_proposal_kv_cache.py
```

Expected: PASS.

- [x] **Step 5: Verify no behavior drift**

Run:

```bash
git diff --check -- \
  tinyvllm/engine/qwen35_mtp_executor.py \
  tinyvllm/engine/proposal_kv_cache.py \
  tools/test_qwen35_mtp_tp4_rank_evidence.py
```

Expected: no output. Do not stage or commit.

### Task 8: Build the Frozen TP4/4K Authority Tooling

**Files:**
- Create: `tools/qwen35_native_mtp_tp4_4k_engine_worker.py`
- Create: `tools/qwen35_native_mtp_tp4_4k_engine_gate.py`
- Create: `tools/verify_qwen35_native_mtp_tp4_4k_engine_gate.py`
- Create: `tools/run_qwen35_native_mtp_tp4_4k_engine_gate_remote.sh`
- Create: `tools/test_qwen35_native_mtp_tp4_4k_engine_gate.py`
- Create: `tools/test_verify_qwen35_native_mtp_tp4_4k_engine_gate.py`

**Interfaces:**
- Produces schema
  `qwen35.native-mtp-tp4-4k-engine-transactional-correctness.v1`.
- Produces classification
  `QWEN35_NATIVE_MTP_TP4_4K_ENGINE_ESTABLISHED`.
- Keeps promotion status `NOT_PROMOTABLE`.

- [x] **Step 1: Write gate and verifier RED tests**

Freeze exact validation for:

- world size and rank inventories;
- four cells `baseline/native_mtp x b1/b4`;
- 4,096 prompt tokens and 32 output tokens;
- baseline/native-MTP and TP1/TP4 token parity;
- positive proposed, accepted, and rejected learned tokens;
- rank-0-only logits and one token broadcast per proposal step;
- matching logical transaction and ticket rows across ranks;
- per-rank callback and lifecycle counters;
- zero accepted-prefix replay;
- zero transaction, ticket, sequence, and physical-slot leaks;
- complete acknowledged finalize/release rows;
- unchanged GPU process inventory; and
- exact source/checkpoint/TP1-authority hashes.

Mutation tests must independently delete or alter each required rank, token
digest, transaction row, cleanup field, and parity row, and require verifier
failure.

- [ ] **Step 2: Run tooling tests and verify RED**

Run:

```bash
pytest -q \
  tools/test_qwen35_native_mtp_tp4_4k_engine_gate.py \
  tools/test_verify_qwen35_native_mtp_tp4_4k_engine_gate.py
```

Expected: collection failure because the TP4 authority modules do not exist.

- [x] **Step 3: Implement the worker and gate**

Adapt the established TP1 worker structure, but:

- construct `tensor_parallel_size=4`;
- collect acknowledged per-rank executor snapshots;
- store token-broadcast and logical transaction parity rows;
- keep each cell in a fresh process group and Engine;
- capture GPU process inventory before and after;
- bind the frozen TP1 authority result;
- write output atomically only after all cells validate; and
- never reuse TP1 schema or overwrite TP1 artifacts.

- [x] **Step 4: Implement the independent verifier**

The verifier must import only standard-library modules. It recomputes hashes,
digests, inventories, parity, lifecycle invariants, and cleanup invariants
from JSON.

- [x] **Step 5: Implement the remote runner**

The shell runner must:

```bash
export KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian
ssh \
  -o ControlMaster=no \
  -o ControlPath=none \
  -o BatchMode=yes \
  sitian@10.232.195.203 \
  ...
```

Use bounded serial retries. Perform remote GPU inventory before launch and
after cleanup. Do not kill unrelated processes.

- [x] **Step 6: Run tooling tests**

Run:

```bash
pytest -q \
  tools/test_qwen35_native_mtp_tp4_4k_engine_gate.py \
  tools/test_verify_qwen35_native_mtp_tp4_4k_engine_gate.py
python -m py_compile \
  tools/qwen35_native_mtp_tp4_4k_engine_worker.py \
  tools/qwen35_native_mtp_tp4_4k_engine_gate.py \
  tools/verify_qwen35_native_mtp_tp4_4k_engine_gate.py
bash -n tools/run_qwen35_native_mtp_tp4_4k_engine_gate_remote.sh
```

Expected: all commands succeed. Do not stage or commit.

### Task 9: Run Focused Local Regression and Real TP4 Authority

**Files:**
- Modify: `docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Create: `artifacts/qwen35_native_mtp_tp4_4k_engine/<opaque-run-id>/...`

**Interfaces:**
- Consumes all implementation and authority tooling from Tasks 1-8.
- Produces local and remote independent verifier receipts.

- [x] **Step 1: Run the focused local suite**

Run:

```bash
pytest -q \
  tools/test_tensor_parallel_greedy.py \
  tools/test_qwen35_mtp.py \
  tools/test_qwen35_mtp_checkpoint.py \
  tools/test_qwen35_mtp_executor.py \
  tools/test_qwen35_mtp_model_runner_integration.py \
  tools/test_qwen35_mtp_tp4_rank_evidence.py \
  tools/test_model_runner_proposal_executor.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_qwen35_native_mtp_tp1_4k_engine_gate.py \
  tools/test_qwen35_native_mtp_tp4_4k_engine_gate.py \
  tools/test_verify_qwen35_native_mtp_tp4_4k_engine_gate.py
```

Expected: PASS with no skipped TP4 contract tests.

- [x] **Step 2: Run static validation**

Run:

```bash
python -m py_compile \
  tinyvllm/models/qwen35_mtp.py \
  tinyvllm/engine/tensor_parallel_greedy.py \
  tinyvllm/engine/qwen35_mtp_executor.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/speculative_runtime.py
bash -n tools/run_qwen35_native_mtp_tp4_4k_engine_gate_remote.sh
git diff --check
```

Expected: all commands succeed.

- [x] **Step 3: Run remote preflight**

Verify:

- Kerberos ticket is valid;
- SSH route works with ControlMaster disabled;
- model and MTP manifests match the frozen hashes;
- four approved GPUs are available;
- no unrelated process will be terminated;
- remote source tree matches the packaged local source; and
- TP1 authority artifact is present and verifies.

Any mismatch is a hard stop for the authority run.

- [x] **Step 4: Run the remote TP4 authority**

Run:

```bash
bash tools/run_qwen35_native_mtp_tp4_4k_engine_gate_remote.sh
```

Expected: four cells complete and the remote verifier writes:

```json
{"classification":"PASS","failures":[]}
```

- [x] **Step 5: Run the independent verifier locally**

Run:

```bash
python tools/verify_qwen35_native_mtp_tp4_4k_engine_gate.py \
  artifacts/qwen35_native_mtp_tp4_4k_engine/<opaque-run-id>/local-authority/result.json \
  --output \
  artifacts/qwen35_native_mtp_tp4_4k_engine/<opaque-run-id>/verify.local.json
```

Expected:

```json
{"classification":"PASS","failures":[]}
```

- [x] **Step 6: Update the audit and handoff**

Record:

- exact artifact paths;
- source and checkpoint hashes;
- test totals and commands;
- b1/b4 proposed, accepted, and rejected counts;
- per-rank callback, broadcast, transaction, finalize, release, and cleanup
  evidence;
- zero accepted-prefix replay;
- unchanged GPU process inventory;
- limitations and `NOT_PROMOTABLE` status; and
- the next ordered gate: native MTP TP4/16K correctness before offload or
  performance claims.

- [x] **Step 7: Perform final completion audit**

Build a prompt-to-artifact checklist against the design and inspect every
artifact field rather than relying on the top-level PASS result. Treat any
missing rank, weak proxy, unverified cleanup row, or parity ambiguity as
unfinished work.

- [x] **Step 8: Report status without committing**

Run:

```bash
git status --short
git diff --check
```

Report changed paths, exact validation evidence, remaining boundaries, and
the next gate. Do not stage, commit, or push.
