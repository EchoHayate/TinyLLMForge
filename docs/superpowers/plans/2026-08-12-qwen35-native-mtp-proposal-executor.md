# Qwen3.5 Native MTP Proposal Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task in the
> current session. Subagents are prohibited for this work. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Load and execute the real Qwen3.5 `mtp.*` checkpoint head inside
`ModelRunner`, maintain transactionally aligned MTP KV, and produce exact-Q
greedy proposals through the existing source-neutral speculative runtime.

**Architecture:** Extend the ModelRunner proposal executor with generic
prefill-observation and two-phase finalization lifecycle hooks. Implement a
model-specific Qwen3.5 MTP module, checkpoint binding, KV owner, exact-Q
executor, and registration factory while keeping Engine, Scheduler, verifier,
and target-KV code free of model/source branches.

**Tech Stack:** Python 3, PyTorch, dataclasses, typing protocols, safetensors
metadata, existing Qwen3.5 packed components, existing exact CUDA graph cache,
pytest, remote CUDA validation.

## Global Constraints

- Modify files only under
  `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, clean, or create
  another worktree.
- Do not use subagents.
- Preserve all unrelated modified and untracked files.
- First slice supports TP1 only.
- First slice requires KV offload disabled.
- First slice supports one native Qwen3.5 MTP layer with shared target
  embedding and LM head.
- Proposal sampling is greedy.
- Every effective proposal length uses a distinct exact-Q execution family.
- Do not pad, round, bucket, or merge different Q values.
- Target and MTP CUDA tensors must remain inside `ModelRunner`.
- Generic Engine, Scheduler, verifier, residency, and target-KV code must not
  branch on Qwen3.5, MTP, learned-drafter, checkpoint, or `source_type`.
- Accepted target and MTP KV remains in its existing physical slots.
- Rejected suffix KV rolls back directly.
- Do not add accepted-KV replay, copy, or per-token rematerialization.
- A failure before target replay may use the existing fallback.
- A CUDA failure after target replay starts must propagate without eager retry.
- Do not claim TP4, KV offload, a second architecture, long-context promotion,
  CUDA correctness, or performance improvement without corresponding evidence.
- Every task follows RED, GREEN, focused regression, `py_compile`, and
  `git diff --check`.
- No task contains a commit step because repository operations are prohibited.

---

## File Structure

### New Production Modules

- `tinyvllm/models/qwen35_mtp_checkpoint.py`
  - Owns the supported MTP config contract, exact 15-tensor metadata plan,
    destination bindings, and all-or-nothing assignment.
- `tinyvllm/models/qwen35_mtp.py`
  - Owns the native MTP module and component factory using shared target
    embedding/LM head.
- `tinyvllm/engine/proposal_kv_cache.py`
  - Owns source-neutral logical proposal-KV slots and reversible
    prepare/commit/rollback transactions.
- `tinyvllm/engine/qwen35_mtp_executor.py`
  - Owns prefill bootstrap, greedy autoregressive proposal execution, exact-Q
    grouping, finalization tickets, and sequence cleanup.
- `tinyvllm/engine/qwen35_mtp_graph.py`
  - Owns exact-Q CUDA graph identity, capture buffers, replay, and eager
    fallback for the Qwen3.5 executor.

### Existing Production Modules

- `tinyvllm/speculative/adapter.py`
  - Adds explicit proposal transaction IDs and lifecycle capability.
- `tinyvllm/speculative/batch_runtime.py`
  - Preserves proposal transaction IDs and produces generic accepted-count
    finalization rows.
- `tinyvllm/speculative/__init__.py`
  - Exports generic lifecycle result types if public API tests require them.
- `tinyvllm/engine/speculative_proposal_executor.py`
  - Adds prefill observation and two-phase finalization registry methods.
- `tinyvllm/engine/speculative_model_runner.py`
  - Adds tensor-free ModelRunner bridge callbacks for lifecycle operations.
- `tinyvllm/engine/speculative_runtime.py`
  - Validates lifecycle capability for ModelRunner executors.
- `tinyvllm/engine/model_runner.py`
  - Observes target prefill hidden states, owns registered executors, and
    exposes lifecycle command methods.
- `tinyvllm/engine/llm_engine.py`
  - Orchestrates MTP finalization prepare/commit/rollback around target KV and
    Scheduler publication.
- `tinyvllm/models/qwen35_components.py`
  - Exposes a focused full-attention decoder-layer component factory reused by
    both target and MTP construction.
- `tinyvllm/models/qwen35_checkpoint.py`
  - Retains target-only loading and reports MTP sources for the separate MTP
    planner without loading them into target destinations.
- `tinyvllm/config.py`
  - Adds explicit first-slice MTP enablement and exact-Q graph controls.

### Focused Tests

- `tools/test_speculative_adapter.py`
- `tools/test_model_runner_proposal_executor.py`
- `tools/test_speculative_batch_runtime.py`
- `tools/test_speculative_model_runner_callbacks.py`
- `tools/test_engine_speculative_runtime.py`
- `tools/test_engine_speculative_execution.py`
- `tools/test_qwen35_mtp_checkpoint.py`
- `tools/test_qwen35_mtp.py`
- `tools/test_proposal_kv_cache.py`
- `tools/test_qwen35_mtp_executor.py`
- `tools/test_qwen35_mtp_graph.py`
- `tools/test_qwen35_mtp_model_runner_integration.py`
- `tools/test_speculative_public_api.py`

### Remote Gate

- `tools/qwen35_mtp_real_checkpoint_gate.py`
- `tools/run_qwen35_mtp_real_checkpoint_gate_remote.sh`
- `tools/test_qwen35_mtp_real_checkpoint_gate.py`

---

### Task 1: Add Generic Proposal Lifecycle Contracts

**Files:**
- Modify: `tinyvllm/speculative/adapter.py`
- Modify: `tinyvllm/engine/speculative_proposal_executor.py`
- Modify: `tinyvllm/speculative/__init__.py`
- Modify: `tools/test_speculative_adapter.py`
- Modify: `tools/test_model_runner_proposal_executor.py`
- Modify: `tools/test_speculative_public_api.py`

**Interfaces:**
- Produces:
  `DraftCapabilities.requires_proposal_lifecycle: bool = False`.
- Produces:
  `DraftProposal.proposal_transaction_id: str | None = None`.
- Produces:
  `TargetPrefillObservation`.
- Produces:
  `ProposalFinalizeRow`.
- Produces registry methods:
  `observe_target_prefill()`, `prepare_finalize_batch()`,
  `commit_finalize_batch()`, `rollback_finalize_batch()`.

- [x] **Step 1: Write failing capability and proposal-ID tests**

Add tests equivalent to:

```python
def test_lifecycle_capability_defaults_false():
    capabilities = _capabilities()
    assert capabilities.requires_proposal_lifecycle is False


def test_host_proposal_defaults_to_no_transaction():
    proposal = DraftProposal(1, (7,), "ngram")
    assert proposal.proposal_transaction_id is None


def test_lifecycle_proposal_requires_transaction_id():
    capabilities = _capabilities(
        execution_domain="model_runner",
        requires_proposal_lifecycle=True,
    )
    with pytest.raises(ValueError, match="transaction"):
        _validate_proposals(
            (DraftProposal(1, (7,), "native"),),
            (_input(1),),
            capabilities,
        )
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_speculative_adapter.py \
  tools/test_model_runner_proposal_executor.py \
  tools/test_speculative_public_api.py -q
```

Expected: failures because lifecycle fields and types do not exist.

- [x] **Step 3: Add explicit lifecycle fields**

Implement:

```python
@dataclass(frozen=True)
class DraftCapabilities:
    source_type: str
    supports_batch: bool
    requires_target_hidden: bool
    requires_target_logits: bool
    max_proposal_tokens: int
    execution_domain: str = "host"
    requires_proposal_lifecycle: bool = False


@dataclass(frozen=True)
class DraftProposal:
    sequence_id: int
    token_ids: tuple[int, ...]
    source_type: str
    metadata: object | None = None
    timing_ms: dict[str, float] | None = None
    proposal_transaction_id: str | None = None
```

Validate that lifecycle is boolean, is allowed only for
`execution_domain == "model_runner"`, and requires a non-empty transaction ID
for every non-empty lifecycle proposal.

- [x] **Step 4: Add ModelRunner-local lifecycle types and protocol**

Implement:

```python
@dataclass(frozen=True)
class TargetPrefillObservation:
    sequence_id: int
    sequence_epoch: int
    token_ids: tuple[int, ...]
    positions: object
    target_hidden: object
    is_final_chunk: bool


@dataclass(frozen=True)
class ProposalFinalizeRow:
    sequence_id: int
    proposal_transaction_id: str
    accepted_proposal_tokens: int


class ProposalExecutor(Protocol):
    @property
    def capabilities(self) -> DraftCapabilities: ...

    def observe_target_prefill(
        self,
        rows: tuple[TargetPrefillObservation, ...],
    ) -> None: ...

    def propose_batch(
        self,
        inputs: tuple[ModelRunnerProposalInput, ...],
    ) -> tuple[DraftProposal, ...]: ...

    def prepare_finalize_batch(
        self,
        rows: tuple[ProposalFinalizeRow, ...],
    ) -> str: ...

    def commit_finalize_batch(self, ticket_id: str) -> None: ...

    def rollback_finalize_batch(self, ticket_id: str) -> None: ...
```

Add registry methods with exact executor-ID/capability matching and single
non-empty string ticket validation. Registry lifecycle methods reject
executors whose capabilities do not require lifecycle.

- [x] **Step 5: Add lifecycle registry tests**

Cover:

```python
def test_registry_prepares_commits_and_rolls_back_ticket():
    registry, executor, capabilities = _registered_lifecycle_executor()
    ticket = registry.prepare_finalize_batch(
        "native",
        (_finalize_row(1, "tx-1", 2),),
        capabilities,
    )
    assert ticket == "ticket-1"
    registry.commit_finalize_batch("native", ticket, capabilities)
    assert executor.events == [
        ("prepare", (1, "tx-1", 2)),
        ("commit", "ticket-1"),
    ]
```

Also reject empty IDs, booleans as counts, negative counts, duplicate sequence
IDs, duplicate transaction IDs, mismatched capabilities, and tensor values in
returned public rows.

- [x] **Step 6: Run focused tests and checks**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_speculative_adapter.py \
  tools/test_model_runner_proposal_executor.py \
  tools/test_speculative_public_api.py -q
python3 -m py_compile \
  tinyvllm/speculative/adapter.py \
  tinyvllm/engine/speculative_proposal_executor.py
git diff --check -- \
  tinyvllm/speculative/adapter.py \
  tinyvllm/engine/speculative_proposal_executor.py \
  tinyvllm/speculative/__init__.py \
  tools/test_speculative_adapter.py \
  tools/test_model_runner_proposal_executor.py \
  tools/test_speculative_public_api.py
```

Expected: all commands exit 0.

---

### Task 2: Carry Proposal Finalization Rows Through the Prepared Runtime

**Files:**
- Modify: `tinyvllm/speculative/batch_runtime.py`
- Modify: `tools/test_speculative_batch_runtime.py`

**Interfaces:**
- Consumes:
  `DraftProposal.proposal_transaction_id`.
- Produces:
  `PreparedProposalFinalizeRow`.
- Produces:
  `build_prepared_proposal_finalize_rows(prepared)`.

- [x] **Step 1: Write failing prepared-row tests**

Add:

```python
def test_prepared_batch_exposes_lifecycle_finalize_rows():
    prepared = _prepare(
        proposal=_proposal(
            1,
            (11, 12, 13),
            proposal_transaction_id="tx-1",
        ),
        target_tokens=(11, 12, 99),
    )
    rows = build_prepared_proposal_finalize_rows(prepared)
    assert rows == (
        PreparedProposalFinalizeRow(
            sequence_id=1,
            proposal_transaction_id="tx-1",
            accepted_proposal_tokens=2,
        ),
    )
```

Add cases for zero accepted tokens, `Q=1`, empty proposal, missing transaction
on lifecycle proposals, and a host proposal with no transaction.

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_speculative_batch_runtime.py -q
```

Expected: import or attribute failures for the new finalize-row API.

- [x] **Step 3: Add prepared finalization rows**

Implement:

```python
@dataclass(frozen=True)
class PreparedProposalFinalizeRow:
    sequence_id: int
    proposal_transaction_id: str
    accepted_proposal_tokens: int


def build_prepared_proposal_finalize_rows(
    prepared: PreparedNativeSpeculativeBatch,
) -> tuple[PreparedProposalFinalizeRow, ...]:
    _require_prepared_batch(prepared)
    rows = []
    for row in prepared.sequences:
        transaction_id = row.proposal.proposal_transaction_id
        if transaction_id is None:
            continue
        rows.append(PreparedProposalFinalizeRow(
            sequence_id=row.sequence_id,
            proposal_transaction_id=transaction_id,
            accepted_proposal_tokens=len(row.accepted_tokens),
        ))
    return tuple(rows)
```

Validate exact sequence ordering, unique transaction IDs, and accepted count
not exceeding proposal length.

- [x] **Step 4: Verify target rollback does not silently consume proposal rows**

Add a test asserting
`rollback_prepared_native_speculative_batch()` changes only target
transactions. Proposal finalization remains an explicit Engine/ModelRunner
operation; no hidden source-specific callback belongs in the batch runtime.

- [x] **Step 5: Run focused tests and checks**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_speculative_batch_runtime.py -q
python3 -m py_compile tinyvllm/speculative/batch_runtime.py
git diff --check -- \
  tinyvllm/speculative/batch_runtime.py \
  tools/test_speculative_batch_runtime.py
```

Expected: all commands exit 0.

---

### Task 3: Add Source-Neutral ModelRunner Lifecycle Bridges

**Files:**
- Modify: `tinyvllm/engine/speculative_model_runner.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_speculative_model_runner_callbacks.py`
- Modify: `tools/test_model_runner_proposal_executor.py`

**Interfaces:**
- Consumes registry lifecycle methods from Task 1.
- Produces ModelRunner methods:
  `observe_speculative_target_prefill_batch()`,
  `prepare_speculative_proposal_finalize_batch()`,
  `commit_speculative_proposal_finalize_batch()`,
  `rollback_speculative_proposal_finalize_batch()`.
- Produces strict tensor-free bridge helpers with the same operation names.

- [x] **Step 1: Write failing bridge tests**

Add tests equivalent to:

```python
def test_prepare_finalize_bridge_round_trips_ticket():
    runner = _Runner()
    ticket = prepare_model_runner_proposal_finalize_batch(
        runner,
        _descriptor(),
        (_prepared_finalize_row(1, "tx-1", 2),),
    )
    assert ticket == "ticket-1"
    assert runner.calls == [(
        "prepare_speculative_proposal_finalize_batch",
        "native",
        (_proposal_finalize_row(1, "tx-1", 2),),
    )]
```

Test commit/rollback operation names, empty/malformed acknowledgement rejection,
and no tensor leakage.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_model_runner_proposal_executor.py -q
```

Expected: lifecycle bridge functions are missing.

- [x] **Step 3: Add ModelRunner command methods**

Implement focused methods:

```python
def prepare_speculative_proposal_finalize_batch(
    self,
    executor_id,
    rows,
):
    capabilities = (
        self.speculative_proposal_executors
        .capabilities_for(executor_id)
    )
    return self.speculative_proposal_executors.prepare_finalize_batch(
        executor_id,
        rows,
        capabilities,
    )
```

Commit and rollback methods resolve the same registered capability and require a
single-use non-empty ticket. Observation invokes only lifecycle-aware registered
executors.

- [x] **Step 4: Add bridge validation**

Bridge functions convert public prepared rows to
`ProposalFinalizeRow`, call `model_runner.call(...)`, and validate:

- exact operation;
- exact executor ID;
- non-empty string ticket;
- no tensors or opaque objects;
- exact one acknowledgement.

- [x] **Step 5: Run focused tests and checks**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_model_runner_proposal_executor.py -q
python3 -m py_compile \
  tinyvllm/engine/speculative_model_runner.py \
  tinyvllm/engine/model_runner.py
git diff --check -- \
  tinyvllm/engine/speculative_model_runner.py \
  tinyvllm/engine/model_runner.py \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_model_runner_proposal_executor.py
```

Expected: all commands exit 0.

---

### Task 4: Observe Final Target Prefill Hidden States

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/speculative_proposal_executor.py`
- Create: `tools/test_model_runner_proposal_prefill_observation.py`
- Modify: `tools/test_context_modes.py`

**Interfaces:**
- Consumes:
  `TargetPrefillObservation`.
- Produces:
  `_proposal_prefill_observation_required()`.
- Produces:
  `_observe_proposal_target_prefill(...)`.

- [x] **Step 1: Write failing constructed-ModelRunner tests**

Cover:

```python
def test_final_prefill_returns_hidden_only_for_lifecycle_executor():
    runner = _constructed_runner(lifecycle=True)
    runner._run_model_step([_final_prefill_seq()], True, True, None, ())
    observation = runner.executor.observations[0]
    assert observation.sequence_id == 7
    assert observation.token_ids == (10, 11, 12)
    assert observation.sequence_epoch == 0
    assert observation.is_final_chunk is True
    assert observation.target_hidden.shape == (3, runner.hidden_size)
```

Also cover:

- non-final chunk accumulation;
- chunk token slicing;
- multiple prefill sequences;
- mixed batch rejection for the first slice;
- no extra hidden return when no lifecycle executor is registered;
- observation tensors never appear in Engine return values.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_model_runner_proposal_prefill_observation.py \
  tools/test_context_modes.py -q
```

Expected: no observation path exists.

- [x] **Step 3: Request target hidden locally**

In `_run_model_step`, compute:

```python
observe_proposal_prefill = (
    is_prefill
    and self._proposal_prefill_observation_required()
)
outputs = self.run_model(
    input_ids,
    positions,
    is_prefill,
    return_hidden=observe_proposal_prefill,
    execution_mode=execution_mode,
)
```

When enabled, split normalized hidden rows using the exact prefill chunk token
counts and create `TargetPrefillObservation` rows. Keep the normal sampled-token
return contract unchanged.

- [x] **Step 4: Make chunk accumulation explicit**

Observation rows carry only the exact computed chunk. The executor owns
accumulation and validates contiguous positions. ModelRunner must not retain
duplicate hidden tensors.

- [x] **Step 5: Run focused and prefill regressions**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_model_runner_proposal_prefill_observation.py \
  tools/test_context_modes.py \
  tools/test_chunked_prefill.py \
  tools/test_qwen35_cached_prefill_eager_attention.py -q
python3 -m py_compile \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/speculative_proposal_executor.py
git diff --check -- \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/speculative_proposal_executor.py \
  tools/test_model_runner_proposal_prefill_observation.py \
  tools/test_context_modes.py
```

Expected: all commands exit 0.

---

### Task 5: Add the Exact Qwen3.5 MTP Checkpoint Plan and Binding

**Files:**
- Create: `tinyvllm/models/qwen35_mtp_checkpoint.py`
- Modify: `tinyvllm/models/qwen35_checkpoint.py`
- Create: `tools/test_qwen35_mtp_checkpoint.py`

**Interfaces:**
- Produces:
  `Qwen35MTPCheckpointTensor`.
- Produces:
  `Qwen35MTPCheckpointPlan`.
- Produces:
  `build_qwen35_mtp_checkpoint_plan(hf_config, index_payload, shard_headers)`.
- Produces:
  `bind_qwen35_mtp_checkpoint(module, plan, tensor_reader)`.

- [x] **Step 1: Write exact config and source-set tests**

Define the supported source contract in the test:

```python
EXPECTED_MTP = {
    "mtp.fc.weight": ("BF16", (2048, 4096)),
    "mtp.layers.0.input_layernorm.weight": ("BF16", (2048,)),
    "mtp.layers.0.self_attn.q_proj.weight": ("BF16", (4096, 2048)),
    "mtp.layers.0.self_attn.k_proj.weight": ("BF16", (512, 2048)),
    "mtp.layers.0.self_attn.v_proj.weight": ("BF16", (512, 2048)),
    "mtp.layers.0.self_attn.o_proj.weight": ("BF16", (2048, 2048)),
    "mtp.layers.0.self_attn.q_norm.weight": ("BF16", (256,)),
    "mtp.layers.0.self_attn.k_norm.weight": ("BF16", (256,)),
    "mtp.layers.0.post_attention_layernorm.weight": ("BF16", (2048,)),
    "mtp.layers.0.mlp.gate_proj.weight": ("BF16", (6144, 2048)),
    "mtp.layers.0.mlp.up_proj.weight": ("BF16", (6144, 2048)),
    "mtp.layers.0.mlp.down_proj.weight": ("BF16", (2048, 6144)),
    "mtp.norm.weight": ("BF16", (2048,)),
    "mtp.pre_fc_norm_embedding.weight": ("BF16", (2048,)),
    "mtp.pre_fc_norm_hidden.weight": ("BF16", (2048,)),
}
```

Reject:

- `mtp_num_hidden_layers != 1`;
- `mtp_use_dedicated_embeddings != False`;
- `tie_word_embeddings != True`;
- missing, duplicate, unexpected MTP sources;
- non-BF16 dtype;
- any wrong dimension;
- header/index shard disagreement.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_qwen35_mtp_checkpoint.py -q
```

Expected: module import failure.

- [x] **Step 3: Implement immutable tensor planning**

Implement dataclasses containing source name, shard, dtype, shape, byte range,
destination path, and packed slot. Reuse the existing bounded safetensors
metadata parser rather than loading tensor payloads during planning.

Map:

```python
PACKED = {
    "mtp.layers.0.self_attn.q_proj.weight": (
        "layer.decoder_layer.full_attention.q_projection.weight",
        "q",
    ),
    "mtp.layers.0.self_attn.k_proj.weight": (
        "layer.decoder_layer.full_attention.k_projection.weight",
        "k",
    ),
    "mtp.layers.0.self_attn.v_proj.weight": (
        "layer.decoder_layer.full_attention.v_projection.weight",
        "v",
    ),
    "mtp.layers.0.mlp.gate_proj.weight": (
        "layer.decoder_layer.mlp.gate_up_proj.weight",
        0,
    ),
    "mtp.layers.0.mlp.up_proj.weight": (
        "layer.decoder_layer.mlp.gate_up_proj.weight",
        1,
    ),
}
```

- [x] **Step 4: Implement all-or-nothing binding**

Resolve and validate every destination before reading or assigning payloads.
Snapshot destination tensors, assign through existing `weight_loader` methods,
and restore all snapshots if any assignment fails. Return the exact sorted
loaded source set.

- [x] **Step 5: Retain target/MTP source separation**

Keep `build_qwen35_checkpoint_weight_plan()` target-only. Add a helper that
returns the sorted MTP skips so the registration factory can prove that every
skipped `mtp.*` source is consumed by the separate MTP plan.

- [x] **Step 6: Run tests and checks**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_qwen35_mtp_checkpoint.py \
  tools/test_qwen35_checkpoint_metadata.py \
  tools/test_qwen35_checkpoint_target_binding.py -q
python3 -m py_compile \
  tinyvllm/models/qwen35_mtp_checkpoint.py \
  tinyvllm/models/qwen35_checkpoint.py
git diff --check -- \
  tinyvllm/models/qwen35_mtp_checkpoint.py \
  tinyvllm/models/qwen35_checkpoint.py \
  tools/test_qwen35_mtp_checkpoint.py
```

Expected: all commands exit 0.

---

### Task 6: Build the Native Qwen3.5 MTP Module

**Files:**
- Create: `tinyvllm/models/qwen35_mtp.py`
- Modify: `tinyvllm/models/qwen35_components.py`
- Create: `tools/test_qwen35_mtp.py`
- Modify: `tools/test_qwen35_packed_full_decoder_layer.py`

**Interfaces:**
- Produces:
  `build_qwen35_full_attention_decoder_layer(...)`.
- Produces:
  `Qwen35NativeMTP`.
- Produces:
  `build_qwen35_native_mtp(...)`.
- `Qwen35NativeMTP.forward_step(...) -> tuple[Tensor, Tensor]`.

- [x] **Step 1: Write failing component-reuse tests**

Add a test requiring the target component assembly and MTP assembly to use the
same public full-attention builder. Reject TP sizes other than one in the MTP
factory.

- [x] **Step 2: Write independent math-oracle tests**

Use deterministic tiny modules:

```python
def reference_mtp(module, input_ids, positions, target_hidden):
    embedded = module.embed_tokens(input_ids)
    embedded = offset_rmsnorm_reference(
        embedded,
        module.pre_fc_norm_embedding.weight,
        module.pre_fc_norm_embedding.eps,
    )
    hidden = offset_rmsnorm_reference(
        target_hidden,
        module.pre_fc_norm_hidden.weight,
        module.pre_fc_norm_hidden.eps,
    )
    fused = F.linear(torch.cat((embedded, hidden), dim=-1), module.fc.weight)
    decoded = module.layer((len(input_ids),), positions, fused)
    normalized = offset_rmsnorm_reference(
        decoded,
        module.norm.weight,
        module.norm.eps,
    )
    return normalized, module.lm_head(normalized)
```

Assert hidden and logits parity, concatenation order, FC orientation, exact
shape validation, shared embedding identity, and shared LM-head identity.

- [ ] **Step 3: Run tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_qwen35_mtp.py \
  tools/test_qwen35_packed_full_decoder_layer.py -q
```

Expected: missing module and public builder.

- [x] **Step 4: Extract the focused full-attention builder**

Move only the full-attention construction logic from the nested target factory
into:

```python
def build_qwen35_full_attention_decoder_layer(
    *,
    hidden_size: int,
    intermediate_size: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    norm_eps: float,
    rotary_dim: int,
    rope_theta: float,
    mrope_section: tuple[int, int, int],
    build_attention_backend,
) -> Qwen35DecoderLayerShell:
    ...
```

The target factory calls this helper unchanged. Do not refactor linear-attention
construction.

- [x] **Step 5: Implement the native module**

Implement:

```python
class Qwen35NativeMTP(nn.Module):
    def __init__(
        self,
        *,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        fc: nn.Module,
        layer: Qwen35PackedFullDecoderLayer,
        norm: nn.Module,
        pre_fc_norm_embedding: nn.Module,
        pre_fc_norm_hidden: nn.Module,
    ): ...

    def forward_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.pre_fc_norm_embedding(
            self.embed_tokens(input_ids)
        )
        hidden = self.pre_fc_norm_hidden(hidden_states)
        fused = self.fc(torch.cat((embedded, hidden), dim=-1))
        decoded = self.layer((int(input_ids.shape[0]),), positions, fused)
        normalized = self.norm(decoded)
        return normalized, self.lm_head(normalized)
```

Use `ReplicatedLinear(2 * hidden_size, hidden_size, bias=False)` for `mtp.fc`
in TP1. All parameter modules use BF16.

- [x] **Step 6: Run focused and target regressions**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_qwen35_mtp.py \
  tools/test_qwen35_packed_full_decoder_layer.py \
  tools/test_qwen35_concrete_component_factory.py \
  tools/test_qwen35_norm_query_gate.py \
  tools/test_qwen35_partial_interleaved_mrope.py -q
python3 -m py_compile \
  tinyvllm/models/qwen35_mtp.py \
  tinyvllm/models/qwen35_components.py
git diff --check -- \
  tinyvllm/models/qwen35_mtp.py \
  tinyvllm/models/qwen35_components.py \
  tools/test_qwen35_mtp.py \
  tools/test_qwen35_packed_full_decoder_layer.py
```

Expected: all commands exit 0.

---

### Task 7: Add Reversible Proposal KV Transactions

**Files:**
- Create: `tinyvllm/engine/proposal_kv_cache.py`
- Create: `tools/test_proposal_kv_cache.py`

**Interfaces:**
- Produces:
  `ProposalKVSequenceState`.
- Produces:
  `ProposalKVTransaction`.
- Produces:
  `ProposalKVFinalizeTicket`.
- Produces:
  `ProposalKVCache.begin()`, `.mark_materialized()`,
  `.prepare_finalize()`, `.commit_finalize()`,
  `.rollback_finalize()`, `.abort()`, `.release_sequence()`.

- [x] **Step 1: Write the full state-machine test matrix**

For every `q` in `(1, 2, 3, 5)` and `accepted` in `range(q + 1)`:

```python
transaction = cache.begin(
    sequence_id=7,
    sequence_epoch=3,
    staged_entry_count=max(q - 1, 0),
)
cache.mark_materialized(transaction, max(q - 1, 0))
ticket = cache.prepare_finalize(
    transaction.transaction_id,
    accepted_proposal_tokens=accepted,
)
cache.commit_finalize(ticket.ticket_id)
assert cache.committed_length(7) == max(accepted - 1, 0)
```

Track slot IDs and assert accepted slots retain identity while rejected slots
return to the free list.

- [x] **Step 2: Add failure tests**

Reject:

- overlapping sequence transactions;
- stale sequence epoch;
- materialized count beyond staged count;
- accepted count beyond `staged + 1`;
- unknown/duplicate transaction;
- duplicate prepare;
- commit after rollback;
- rollback after commit;
- ticket reuse;
- releasing a sequence with active transaction/ticket.

- [ ] **Step 3: Run tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_proposal_kv_cache.py -q
```

Expected: module import failure.

- [x] **Step 4: Implement metadata-first transactions**

Use dataclasses with:

```python
@dataclass
class ProposalKVTransaction:
    transaction_id: str
    sequence_id: int
    sequence_epoch: int
    original_committed_length: int
    staged_slot_ids: tuple[int, ...]
    materialized_entry_count: int = 0
    state: str = "reserved"


@dataclass
class ProposalKVFinalizeTicket:
    ticket_id: str
    transaction_id: str
    commit_entry_count: int
    release_slot_ids: tuple[int, ...]
    state: str = "prepared"
```

`prepare_finalize()` performs every structural and ownership validation but
does not change committed length or free slots. `commit_finalize()` publishes
the accepted prefix and releases only the rejected suffix.

- [x] **Step 5: Prove no replay/copy path exists**

Use a fake physical store whose only methods are `reserve_slots()` and
`release_slots()`. Assert commit never invokes tensor copy, rematerialization,
or a model forward callback.

- [x] **Step 6: Run tests and checks**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_proposal_kv_cache.py -q
python3 -m py_compile tinyvllm/engine/proposal_kv_cache.py
git diff --check -- \
  tinyvllm/engine/proposal_kv_cache.py \
  tools/test_proposal_kv_cache.py
```

Expected: all commands exit 0.

---

### Task 8: Implement Qwen3.5 MTP Bootstrap and Eager Proposal Execution

**Files:**
- Create: `tinyvllm/engine/qwen35_mtp_executor.py`
- Create: `tools/test_qwen35_mtp_executor.py`

**Interfaces:**
- Consumes:
  `Qwen35NativeMTP`, `ProposalKVCache`,
  `TargetPrefillObservation`, `ProposalFinalizeRow`.
- Produces:
  `Qwen35MTPProposalExecutor`.
- Produces:
  `Qwen35MTPProposalExecutor.capabilities`.

- [x] **Step 1: Write failing prefill accumulation tests**

Cover contiguous chunks:

```python
executor.observe_target_prefill((
    _observation(7, epoch=0, tokens=(10, 11), positions=(0, 1),
                 final=False),
))
executor.observe_target_prefill((
    _observation(7, epoch=0, tokens=(12,), positions=(2,), final=True),
))
assert executor.pending_prefix(7).token_ids == (10, 11, 12)
```

Reject gaps, overlaps, changed epochs, duplicate final chunks, changed hidden
width/dtype/device, and a second bootstrap.

- [x] **Step 2: Write failing bootstrap alignment tests**

Given prompt `(10, 11, 12)` and first sampled token `13`, assert bootstrap calls
the MTP module with:

```text
input_ids = (11, 12, 13)
positions = (0, 1, 2)
target_hidden = hidden(prompt positions 0, 1, 2)
```

Assert committed MTP KV length becomes `3`, bootstrap logits are discarded, and
pending target hidden tensors are released.

- [x] **Step 3: Write failing proposal-semantics tests**

For effective Q values:

```python
assert propose(q=0).token_ids == ()
assert propose(q=1).token_ids == (first_target,)
assert propose(q=4).token_ids == (
    first_target,
    greedy_1,
    greedy_2,
    greedy_3,
)
assert module.forward_call_count == 3
```

Assert call zero consumes target hidden and later calls consume previous MTP
hidden. Verify the proposal carries a non-empty transaction ID and stages
exactly `Q - 1` KV entries.

- [ ] **Step 4: Run tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_qwen35_mtp_executor.py -q
```

Expected: module import failure.

- [x] **Step 5: Implement capabilities and lifecycle**

Use:

```python
self._capabilities = DraftCapabilities(
    source_type="native_model_runner",
    supports_batch=True,
    requires_target_hidden=True,
    requires_target_logits=False,
    max_proposal_tokens=max_proposal_tokens,
    execution_domain="model_runner",
    requires_proposal_lifecycle=True,
)
```

Do not use `source_type` for routing. The executor:

- accumulates prefill observations;
- bootstraps once;
- groups proposal inputs by effective exact Q;
- creates one KV transaction per non-empty proposal;
- runs eager `Q - 1` forwards;
- returns input order;
- delegates prepare/commit/rollback finalization to `ProposalKVCache`;
- releases all state on sequence completion.

- [x] **Step 6: Validate tensor-free output**

Proposal metadata may contain scalar counters and exact-Q identity strings but
no tensors, physical slot IDs, allocator objects, or graph objects.

- [x] **Step 7: Run tests and checks**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_qwen35_mtp_executor.py \
  tools/test_proposal_kv_cache.py \
  tools/test_model_runner_proposal_executor.py -q
python3 -m py_compile tinyvllm/engine/qwen35_mtp_executor.py
git diff --check -- \
  tinyvllm/engine/qwen35_mtp_executor.py \
  tools/test_qwen35_mtp_executor.py
```

Expected: all commands exit 0.

---

### Task 9: Add Distinct Exact-Q CUDA Graph Families

**Files:**
- Create: `tinyvllm/engine/qwen35_mtp_graph.py`
- Modify: `tinyvllm/engine/qwen35_mtp_executor.py`
- Modify: `tinyvllm/config.py`
- Create: `tools/test_qwen35_mtp_graph.py`
- Modify: `tools/test_qwen35_mtp_executor.py`

**Interfaces:**
- Produces:
  `Qwen35MTPGraphIdentity`.
- Produces:
  `Qwen35MTPExactGraphRunner.run(...)`.
- Adds config:
  `qwen35_mtp_cuda_graphs`,
  `qwen35_mtp_cuda_graph_q_allowlist`,
  `qwen35_mtp_cuda_graph_batch_allowlist`,
  capture-count and memory/time budgets.

- [x] **Step 1: Write exact identity tests**

Assert identity differs for every change to:

```text
exact_q
exact_batch_size
device_index
compute_dtype
hidden_size
mtp_layer_count
block_table_width
```

Reject Q below 2 because Q=1 performs no MTP forward.

- [x] **Step 2: Write no-padding mixed-Q tests**

For input effective Q values `(2, 4, 2, 3)`, assert execution events are:

```text
run(q=2, sequence_ids=(first, third))
run(q=4, sequence_ids=(second,))
run(q=3, sequence_ids=(fourth,))
```

No call may receive a padded Q or emit discarded suffix tokens.

- [x] **Step 3: Write eager/capture/replay state tests**

Reuse `ExactCudaGraphCache` admission semantics:

- first observations execute eager;
- eligible identity captures only after successful eager runs;
- capture uses private scratch KV slots;
- live proposal transaction slots are not mutated by capture;
- exact identity replays;
- capture failure quarantines only that identity;
- a post-replay CUDA error propagates without eager retry.

- [ ] **Step 4: Run tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_qwen35_mtp_graph.py \
  tools/test_qwen35_mtp_executor.py -q
```

Expected: graph module and config fields are missing.

- [x] **Step 5: Implement exact-Q graph runner**

The graph body statically unrolls exactly `Q - 1` calls. Static buffers contain
input IDs, positions, hidden states, logits/token outputs, MTP block tables, and
slot mappings for the exact batch size.

Capture clones committed prefix state into private scratch, captures staged
suffix writes there, discards capture output, and publishes the entry only
after scratch rollback succeeds.

- [x] **Step 6: Integrate executor grouping**

Replace each eager exact-Q group call with:

```python
result = self.graph_runner.run(
    exact_q=effective_q,
    rows=group,
    eager=self._run_exact_q_eager,
)
```

The eager callable remains the correctness oracle.

- [x] **Step 7: Run focused and graph-cache regressions**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_qwen35_mtp_graph.py \
  tools/test_qwen35_mtp_executor.py \
  tools/test_spec_verify_exact_cuda_graph_cache.py \
  tools/test_model_runner_spec_verify_cuda_graph.py -q
python3 -m py_compile \
  tinyvllm/engine/qwen35_mtp_graph.py \
  tinyvllm/engine/qwen35_mtp_executor.py \
  tinyvllm/config.py
git diff --check -- \
  tinyvllm/engine/qwen35_mtp_graph.py \
  tinyvllm/engine/qwen35_mtp_executor.py \
  tinyvllm/config.py \
  tools/test_qwen35_mtp_graph.py \
  tools/test_qwen35_mtp_executor.py
```

Expected: all commands exit 0.

---

### Task 10: Register the Real Executor from the Qwen3.5 Model Factory

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/speculative_runtime.py`
- Modify: `tinyvllm/models/qwen35_components.py`
- Modify: `tinyvllm/models/qwen35_checkpoint.py`
- Modify: `tinyvllm/config.py`
- Create: `tools/test_qwen35_mtp_model_runner_integration.py`
- Modify: `tools/test_engine_speculative_runtime.py`

**Interfaces:**
- Produces:
  `ModelRunner._maybe_register_qwen35_mtp_executor()`.
- Produces executor ID:
  `"native_checkpoint_proposal"`.
- Produces a `ModelRunnerProposalExecutorDescriptor` from loaded capability,
  without generic model-name dispatch.

- [x] **Step 1: Write failing registration tests**

Construct a ModelRunner with a fake Qwen3.5 target and MTP plan. Assert:

- target load completes before MTP construction;
- target embedding and MTP embedding are the same object;
- target LM head and MTP LM head are the same object;
- all 15 sources load exactly once;
- executor registers only after binding succeeds;
- TP2/TP4, KV offload, dedicated embeddings, multiple MTP layers, and disabled
  config do not register the executor;
- failed MTP registration leaves the target model usable.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_qwen35_mtp_model_runner_integration.py \
  tools/test_engine_speculative_runtime.py -q
```

Expected: no registration factory or config exists.

- [x] **Step 3: Add explicit config**

Add validated fields:

```text
qwen35_mtp_enabled: bool = false
qwen35_mtp_max_proposal_tokens: int
qwen35_mtp_cuda_graphs: bool
qwen35_mtp_cuda_graph_q_allowlist: tuple[int, ...]
qwen35_mtp_cuda_graph_batch_allowlist: tuple[int, ...]
```

Enabling MTP requires TP1, no KV offload, greedy speculative runtime, and a
positive proposal limit.

- [x] **Step 4: Implement model-specific registration**

The factory:

1. detects the supported MTP config only inside the Qwen3.5 load path;
2. builds the exact MTP tensor plan;
3. constructs the MTP module with shared target modules;
4. binds all MTP weights;
5. creates proposal KV and graph owners;
6. constructs `Qwen35MTPProposalExecutor`;
7. registers it under the source-neutral executor ID;
8. returns the descriptor capabilities to runtime construction.

No generic runtime file may inspect Qwen3.5 config fields.

- [x] **Step 5: Add static source-neutrality gates**

Scan generic files and reject case-insensitive occurrences of:

```text
qwen
mtp
learned
source_type ==
source_type in
checkpoint
```

Allow the generic lifecycle type name
`requires_proposal_lifecycle`; it does not identify a source.

- [x] **Step 6: Run focused tests and checks**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_qwen35_mtp_model_runner_integration.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_engine_speculative_execution.py -q
python3 -m py_compile \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/speculative_runtime.py \
  tinyvllm/config.py
git diff --check -- \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/speculative_runtime.py \
  tinyvllm/models/qwen35_components.py \
  tinyvllm/models/qwen35_checkpoint.py \
  tinyvllm/config.py \
  tools/test_qwen35_mtp_model_runner_integration.py \
  tools/test_engine_speculative_runtime.py
```

Expected: all commands exit 0.

---

### Task 11: Orchestrate Two-Phase MTP Finalization in Engine

**Files:**
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tinyvllm/engine/speculative_model_runner.py`
- Modify: `tinyvllm/speculative/batch_runtime.py`
- Modify: `tools/test_engine_speculative_execution.py`
- Modify: `tools/test_speculative_model_runner_callbacks.py`
- Modify: `tools/test_speculative_batch_runtime.py`

**Interfaces:**
- Consumes:
  `build_prepared_proposal_finalize_rows()`.
- Consumes ModelRunner lifecycle bridges from Task 3.
- Produces exact prepare/commit/rollback ordering around target publication.

- [x] **Step 1: Write failing success-order test**

Record events and require:

```python
assert events == [
    "verify_complete",
    "target_commit_plans_prepared",
    "proposal_finalize_prepared",
    "target_kv_committed",
    "scheduler_committed",
    "proposal_finalize_committed",
]
```

The accepted count supplied to the executor must equal
`len(prepared_row.accepted_tokens)`.

- [x] **Step 2: Write pre-publication failure tests**

If target KV commit or Scheduler commit fails after proposal finalization
prepare:

```python
assert events[-1] == "proposal_finalize_rolled_back"
assert runtime_poisoned is False
```

Target transaction rollback remains governed by the existing prepared-runtime
path.

- [x] **Step 3: Write post-publication failure test**

If proposal finalization commit fails after target/Scheduler publication:

```python
assert engine.speculative_runtime_poisoned is True
assert "proposal finalization commit failed" in reason
assert eager_retry_count == 0
```

The next speculative step must fail before dispatch.

- [ ] **Step 4: Run tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_engine_speculative_execution.py \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_speculative_batch_runtime.py -q
```

Expected: Engine does not call lifecycle finalization.

- [x] **Step 5: Add source-neutral two-phase orchestration**

After building target `kv_plans`:

```python
finalize_rows = build_prepared_proposal_finalize_rows(
    prepared_runtime
)
proposal_finalize_ticket = None
if finalize_rows:
    proposal_finalize_ticket = (
        prepare_model_runner_proposal_finalize_batch(
            self.model_runner,
            runtime.model_runner_executor,
            finalize_rows,
        )
    )
```

On any failure before target/Scheduler publication, roll back the ticket. After
successful publication, commit it. A commit failure poisons the runtime and
propagates.

- [x] **Step 6: Preserve host-adapter behavior**

Host proposals have no transaction IDs, produce no finalization rows, and make
zero lifecycle RPCs. Existing n-gram/SAM behavior and callback counts remain
unchanged.

- [x] **Step 7: Run focused and broad speculative regressions**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_engine_speculative_execution.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_speculative_batch_runtime.py \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_speculative_kv_transaction.py \
  tools/test_ngram_speculative.py -q
python3 -m py_compile \
  tinyvllm/engine/llm_engine.py \
  tinyvllm/engine/speculative_model_runner.py \
  tinyvllm/speculative/batch_runtime.py
git diff --check -- \
  tinyvllm/engine/llm_engine.py \
  tinyvllm/engine/speculative_model_runner.py \
  tinyvllm/speculative/batch_runtime.py \
  tools/test_engine_speculative_execution.py \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_speculative_batch_runtime.py
```

Expected: all commands exit 0.

---

### Task 12: Add Real-Checkpoint CUDA Gate and Update Evidence

**Files:**
- Create: `tools/qwen35_mtp_real_checkpoint_gate.py`
- Create: `tools/run_qwen35_mtp_real_checkpoint_gate_remote.sh`
- Create: `tools/test_qwen35_mtp_real_checkpoint_gate.py`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md`

**Interfaces:**
- Produces a machine-readable gate result with:
  checkpoint identity, config/tensor contract, shared-weight identity,
  eager/reference parity, exact-Q parity, transaction slot identity, and
  explicit promotion classification.

- [x] **Step 1: Write gate-contract tests**

Require schema fields:

```text
schema_version
checkpoint_path
checkpoint_manifest_sha256
device_name
torch_version
cuda_version
q_values
batch_sizes
loader_passed
shared_embedding_identity
shared_lm_head_identity
eager_reference_max_abs_diff
eager_reference_argmax_equal
graph_eager_argmax_equal
transaction_cases
accepted_slot_identity_preserved
rejected_slots_released
post_rollback_continuation_equal
status
promotion_classification
limitations
```

Reject a PASS when any required Q/batch case is absent or when the report claims
TP4, KV offload, long-context, second-architecture, or performance coverage.

- [ ] **Step 2: Run contract tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_qwen35_mtp_real_checkpoint_gate.py -q
```

Expected: gate module import failure.

- [x] **Step 3: Implement the gate**

The gate must:

1. read the selected checkpoint without mutation;
2. load the target and native MTP module;
3. verify shared object/storage identity;
4. compare MTP hidden/logits with an independent equation-level reference;
5. run batch sizes 1 and 4;
6. run at least Q values 1, 2, 3, and 4;
7. compare exact-Q eager and graph argmax;
8. exercise every accepted count from 0 through Q;
9. record physical slot IDs before and after finalization;
10. rerun continuation after partial rollback and compare tokens;
11. emit `NOT_PROMOTABLE` even when this slice passes.

- [x] **Step 4: Add the remote wrapper**

The local wrapper must create a unique isolated remote run root instead of
editing the canonical remote checkout. Use:

```bash
#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_BASE="${REMOTE_BASE:-/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/sitian-workspace01/tllm/env/bin/python}"
TAG="${TAG:-qwen35-mtp-$(date +%Y%m%d-%H%M%S)}"
REMOTE_RUN_ROOT="/data00/home/sitian/sitian-workspace01/tllm/qwen35-mtp-runs/${TAG}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20)

"${SSH[@]}" "${REMOTE_HOST}" \
  "mkdir -p '${REMOTE_RUN_ROOT}' && cp -a '${REMOTE_BASE}/.' '${REMOTE_RUN_ROOT}/'"

SOURCE_FILES=(
  tinyvllm/config.py
  tinyvllm/speculative/__init__.py
  tinyvllm/speculative/adapter.py
  tinyvllm/speculative/batch_runtime.py
  tinyvllm/engine/speculative_proposal_executor.py
  tinyvllm/engine/speculative_model_runner.py
  tinyvllm/engine/speculative_runtime.py
  tinyvllm/engine/model_runner.py
  tinyvllm/engine/llm_engine.py
  tinyvllm/engine/proposal_kv_cache.py
  tinyvllm/engine/qwen35_mtp_executor.py
  tinyvllm/engine/qwen35_mtp_graph.py
  tinyvllm/models/qwen35_checkpoint.py
  tinyvllm/models/qwen35_components.py
  tinyvllm/models/qwen35_mtp_checkpoint.py
  tinyvllm/models/qwen35_mtp.py
  tools/qwen35_mtp_real_checkpoint_gate.py
)

rsync -a --relative \
  -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
  "${SOURCE_FILES[@]}" \
  "${REMOTE_HOST}:${REMOTE_RUN_ROOT}/"

"${SSH[@]}" "${REMOTE_HOST}" \
  "cd '${REMOTE_RUN_ROOT}' && \
   mkdir -p '${REMOTE_RUN_ROOT}/artifacts' && \
   CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-0}' \
   PYTHONPATH='${REMOTE_RUN_ROOT}' \
   '${REMOTE_PYTHON}' tools/qwen35_mtp_real_checkpoint_gate.py \
     --checkpoint /data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model \
     --q-values 1,2,3,4 \
     --batch-sizes 1,4 \
     --output '${REMOTE_RUN_ROOT}/artifacts/qwen35_mtp_real_checkpoint_gate.json'"
```

The local launcher uses the established `sitian@10.232.195.203` route and
`KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`. Run serially; do not depend on
the absent ControlMaster socket. Keep `SOURCE_FILES` as the explicit production
inventory above; never derive it from the dirty working tree.

- [x] **Step 5: Run all local tests before remote execution**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_speculative_adapter.py \
  tools/test_model_runner_proposal_executor.py \
  tools/test_speculative_batch_runtime.py \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_engine_speculative_execution.py \
  tools/test_qwen35_mtp_checkpoint.py \
  tools/test_qwen35_mtp.py \
  tools/test_proposal_kv_cache.py \
  tools/test_qwen35_mtp_executor.py \
  tools/test_qwen35_mtp_graph.py \
  tools/test_qwen35_mtp_model_runner_integration.py \
  tools/test_qwen35_mtp_real_checkpoint_gate.py -q
```

Expected: all tests pass.

- [x] **Step 6: Run the remote gate**

Run serially:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
bash tools/run_qwen35_mtp_real_checkpoint_gate_remote.sh
```

Expected: process exits 0 and writes a gate JSON whose `status` is `PASS` and
whose `promotion_classification` remains `NOT_PROMOTABLE`. Record the exact
unique remote run root and download the JSON into a matching local
`artifacts/qwen35-mtp-runs/<TAG>/` directory before any optional remote cleanup.

- [x] **Step 7: Run final static and syntax gates**

Run:

```bash
python3 -m py_compile \
  tinyvllm/speculative/adapter.py \
  tinyvllm/speculative/batch_runtime.py \
  tinyvllm/engine/speculative_proposal_executor.py \
  tinyvllm/engine/speculative_model_runner.py \
  tinyvllm/engine/proposal_kv_cache.py \
  tinyvllm/engine/qwen35_mtp_executor.py \
  tinyvllm/engine/qwen35_mtp_graph.py \
  tinyvllm/models/qwen35_mtp_checkpoint.py \
  tinyvllm/models/qwen35_mtp.py

rg -n -i \
  'qwen|mtp|learned|checkpoint|source_type[[:space:]]*(==|in)' \
  tinyvllm/speculative/batch_runtime.py \
  tinyvllm/speculative/verifier.py \
  tinyvllm/engine/speculative_runtime.py \
  tinyvllm/engine/speculative_model_runner.py \
  tinyvllm/engine/scheduler.py

git diff --check -- \
  tinyvllm \
  tools/test_speculative_adapter.py \
  tools/test_model_runner_proposal_executor.py \
  tools/test_speculative_batch_runtime.py \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_engine_speculative_execution.py \
  tools/test_qwen35_mtp_checkpoint.py \
  tools/test_qwen35_mtp.py \
  tools/test_proposal_kv_cache.py \
  tools/test_qwen35_mtp_executor.py \
  tools/test_qwen35_mtp_graph.py \
  tools/test_qwen35_mtp_model_runner_integration.py \
  tools/test_qwen35_mtp_real_checkpoint_gate.py
```

Expected:

- `py_compile` exits 0;
- the source-neutrality scan prints no prohibited production matches after
  excluding lifecycle field names and error strings that do not dispatch;
- `git diff --check` exits 0.

- [x] **Step 8: Update handoff and objective audit**

Record:

- exact files changed;
- local test count and commands;
- remote checkpoint/gate artifact path;
- what eager/reference and graph/eager parity prove;
- target and MTP transaction slot evidence;
- limitations;
- next recommended slice;
- overall `NOT_PROMOTABLE` classification.

Do not claim the broader phase-one objective is achieved unless a separate
completion audit proves both-model, TP4, long-context, multi-batch, performance,
memory, real H2D, and acceptance gates.

---

## Plan Self-Review Checklist

- [x] Every spec requirement maps to at least one task.
- [x] Generic lifecycle and two-phase finalization precede model-specific code.
- [x] Prefill bootstrap has an explicit target-hidden observation path.
- [x] The exact 15 BF16 checkpoint tensors and shapes are tested.
- [x] Proposal token zero is the target first token.
- [x] Q proposals perform exactly `Q - 1` MTP forwards.
- [x] MTP KV commits exactly `max(k - 1, 0)` staged entries.
- [x] Finalization validates before target publication.
- [x] A post-publication MTP commit failure poisons without retry.
- [x] Mixed Q values remain distinct and unpadded.
- [x] CUDA graph capture uses private scratch state.
- [x] Host n-gram/SAM paths remain lifecycle-free.
- [x] TP1/no-offload gates are explicit.
- [x] Real-checkpoint evidence is required before a real-MTP correctness claim.
- [x] Broader promotion remains `NOT_PROMOTABLE`.
