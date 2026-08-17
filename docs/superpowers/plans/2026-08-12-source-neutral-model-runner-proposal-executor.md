# Source-Neutral ModelRunner Proposal Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task in the
> current session. Subagents are prohibited for this work.

**Goal:** Let a ModelRunner-owned proposal executor produce validated
`DraftProposal` rows without exporting target CUDA hidden states or logits
to Engine/Scheduler code, while preserving host n-gram/SAM adapters and the
existing transactional KV verifier path.

**Architecture:** Add a source-neutral execution domain to proposal
capabilities, a focused ModelRunner-local executor registry, and one fused
first-target/proposal callback. Refactor the generic batch runtime to consume
uniform `FirstTargetProposalResult` rows so host and ModelRunner proposal
sources share all transaction, exact-Q verification, acceptance, and commit
logic.

**Tech Stack:** Python 3, dataclasses, typing protocols, pytest, TinyLLMForge
ModelRunner shared-memory command dispatch, existing speculative batch
runtime and transactional KV manager.

## Global Constraints

- Modify files only under
  `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, clean, or create
  another worktree.
- Do not use subagents.
- First slice supports TP1 only.
- First slice requires KV offload disabled.
- First slice uses exact proposal lengths and exact-Q verifier families.
- Generic runtime, Scheduler, verifier, and transaction code must not branch
  on model names, checkpoint names, `source_type`, learned-drafter names, or
  MTP names.
- CUDA hidden states and logits must never be returned to Engine or
  Scheduler for ModelRunner proposal execution.
- Accepted speculative KV commits in place; rejected suffix KV rolls back.
- Do not add accepted-KV replay, copy, or per-token rematerialization.
- Do not claim CUDA correctness, performance, H2D/D2H reduction, TP4, KV
  offload, real learned-drafter, or real MTP support.
- Preserve all unrelated modified and untracked files.
- Every implementation task follows RED, GREEN, focused regression, and
  `git diff --check`; no task contains a commit step.

---

## File Structure

### New Focused Module

- `tinyvllm/engine/speculative_proposal_executor.py`
  - Owns ModelRunner-local proposal input, executor protocol, executor
    registry, capability matching, proposal validation, and recursive
    tensor-leak detection.

### Existing Production Files

- `tinyvllm/speculative/adapter.py`
  - Adds and validates the source-neutral execution domain.
- `tinyvllm/speculative/batch_runtime.py`
  - Adds uniform first-target/proposal rows and changes the transaction
    runtime to consume a provider callback.
- `tinyvllm/engine/speculative_runtime.py`
  - Represents mutually exclusive host-adapter and ModelRunner-executor
    runtime configuration.
- `tinyvllm/engine/speculative_model_runner.py`
  - Provides host and fused ModelRunner bridge functions and strict result
    validation.
- `tinyvllm/engine/model_runner.py`
  - Owns executor registration and fused first-target/proposal execution.
- `tinyvllm/engine/llm_engine.py`
  - Selects a provider by capability execution domain and passes one
    provider callback into the generic batch runtime.
- `tinyvllm/speculative/__init__.py`
  - Exports the new public generic result contract if existing public API
    tests require it.

### Focused Tests

- `tools/test_speculative_adapter.py`
- `tools/test_model_runner_proposal_executor.py`
- `tools/test_speculative_model_runner_callbacks.py`
- `tools/test_speculative_batch_runtime.py`
- `tools/test_engine_speculative_runtime.py`
- `tools/test_engine_speculative_execution.py` only if the runtime descriptor
  reaches its fixtures.
- `tools/test_speculative_public_api.py` only if a new public export is
  required.

---

### Task 1: Add Source-Neutral Capability Execution Domain

**Files:**
- Modify: `tinyvllm/speculative/adapter.py`
- Test: `tools/test_speculative_adapter.py`
- Test: `tools/test_speculative_source_adapters.py`

**Interfaces:**
- Produces:
  `DraftCapabilities.execution_domain: str = "host"`.
- Produces:
  `validate_draft_capabilities(capabilities, *, expected_execution_domain=None)`.
- Existing host adapters continue constructing `DraftCapabilities` without a
  new positional argument.

- [ ] **Step 1: Write failing default and validation tests**

Add tests equivalent to:

```python
def test_capabilities_default_to_host_execution():
    capabilities = _capabilities()
    assert capabilities.execution_domain == "host"


@pytest.mark.parametrize("execution_domain", ["", "gpu", "mtp", 1, None])
def test_rejects_invalid_execution_domain(execution_domain):
    adapter = _Adapter(
        _capabilities(execution_domain=execution_domain),
        (_proposal(1, (10,)),),
    )
    with pytest.raises(ValueError, match="execution domain"):
        validate_draft_adapter_batch(adapter, (_context(1),))


def test_rejects_model_runner_capability_in_host_adapter_validator():
    adapter = _Adapter(
        _capabilities(execution_domain="model_runner"),
        (_proposal(1, (10,)),),
    )
    with pytest.raises(ValueError, match="host"):
        validate_draft_adapter_batch(adapter, (_context(1),))
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_speculative_adapter.py -q
```

Expected: failures because `execution_domain` and the shared capability
validator do not exist.

- [ ] **Step 3: Implement the minimal capability contract**

Add:

```python
_EXECUTION_DOMAINS = frozenset({"host", "model_runner"})


@dataclass(frozen=True)
class DraftCapabilities:
    source_type: str
    supports_batch: bool
    requires_target_hidden: bool
    requires_target_logits: bool
    max_proposal_tokens: int
    execution_domain: str = "host"


def validate_draft_capabilities(
    capabilities: object,
    *,
    expected_execution_domain: str | None = None,
) -> DraftCapabilities:
    if not isinstance(capabilities, DraftCapabilities):
        raise ValueError("capabilities must be DraftCapabilities")
    if capabilities.execution_domain not in _EXECUTION_DOMAINS:
        raise ValueError("capability execution domain is unsupported")
    if (
        expected_execution_domain is not None
        and capabilities.execution_domain != expected_execution_domain
    ):
        raise ValueError(
            "capability execution domain does not match "
            f"{expected_execution_domain}"
        )
    # Retain the existing source type, batching, boolean, and limit checks.
    return capabilities
```

Make `validate_draft_adapter_batch()` call
`validate_draft_capabilities(..., expected_execution_domain="host")`.

- [ ] **Step 4: Run capability and source-adapter regressions**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_speculative_adapter.py \
  tools/test_speculative_source_adapters.py -q
```

Expected: all tests pass and existing n-gram/SAM capabilities remain host
capabilities.

- [ ] **Step 5: Check syntax and whitespace**

Run:

```bash
python3 -m py_compile tinyvllm/speculative/adapter.py
git diff --check -- \
  tinyvllm/speculative/adapter.py \
  tools/test_speculative_adapter.py
```

Expected: both commands exit 0.

---

### Task 2: Add the ModelRunner-Local Executor Contract and Registry

**Files:**
- Create: `tinyvllm/engine/speculative_proposal_executor.py`
- Create: `tools/test_model_runner_proposal_executor.py`

**Interfaces:**
- Consumes:
  `DraftCapabilities`, `DraftProposal`, and
  `validate_draft_capabilities()` from Task 1.
- Produces:
  `ModelRunnerProposalInput`.
- Produces:
  `ProposalExecutor` protocol.
- Produces:
  `ModelRunnerProposalExecutorRegistry.register()` and `.execute_batch()`.
- Produces:
  `assert_tensor_free(value, *, name)`.

- [ ] **Step 1: Write failing registry and executor tests**

Create focused tests covering:

```python
class _Executor:
    def __init__(self, capabilities, proposals):
        self.capabilities = capabilities
        self.proposals = proposals
        self.calls = []

    def propose_batch(self, inputs):
        self.calls.append(inputs)
        return self.proposals


def test_registry_executes_and_restores_input_order():
    capabilities = _capabilities(
        execution_domain="model_runner",
    )
    registry = ModelRunnerProposalExecutorRegistry()
    executor = _Executor(
        capabilities,
        (
            DraftProposal(2, (), "fixture"),
            DraftProposal(1, (11, 12), "fixture"),
        ),
    )
    registry.register("fixture-executor", executor, capabilities)
    rows = registry.execute_batch(
        "fixture-executor",
        (_input(1), _input(2)),
        capabilities,
    )
    assert tuple(row.sequence_id for row in rows) == (1, 2)


def test_registry_rejects_missing_executor():
    registry = ModelRunnerProposalExecutorRegistry()
    with pytest.raises(ValueError, match="not registered"):
        registry.execute_batch(
            "missing",
            (_input(1),),
            _capabilities(execution_domain="model_runner"),
        )


def test_registry_rejects_capability_mismatch():
    registered = _capabilities(
        execution_domain="model_runner",
        max_proposal_tokens=4,
    )
    requested = _capabilities(
        execution_domain="model_runner",
        max_proposal_tokens=3,
    )
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register(
        "fixture",
        _Executor(registered, ()),
        registered,
    )
    with pytest.raises(ValueError, match="capabilities"):
        registry.execute_batch(
            "fixture",
            (_input(1),),
            requested,
        )


def test_executor_failure_propagates():
    capabilities = _capabilities(
        execution_domain="model_runner",
    )
    executor = _Executor(capabilities, ())
    executor.propose_batch = lambda inputs: (
        (_ for _ in ()).throw(RuntimeError("executor failed"))
    )
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register("fixture", executor, capabilities)
    with pytest.raises(RuntimeError, match="executor failed"):
        registry.execute_batch(
            "fixture",
            (_input(1),),
            capabilities,
        )


def test_assert_tensor_free_rejects_nested_tensor():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="tensor"):
        assert_tensor_free(
            {"nested": [torch.zeros(1)]},
            name="result",
        )
```

Also cover invalid/duplicate/missing sequence IDs, proposal source mismatch,
proposal limits, missing required hidden/logit payloads, and invalid executor
IDs.

- [ ] **Step 2: Run the new test and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_model_runner_proposal_executor.py -q
```

Expected: collection failure because the new module does not exist.

- [ ] **Step 3: Implement the focused module**

Implement the following concrete structure:

```python
@dataclass(frozen=True)
class ModelRunnerProposalInput:
    sequence_id: int
    token_ids: tuple[int, ...]
    remaining_output_tokens: int
    max_proposal_tokens: int
    first_target_token: int
    target_hidden: object | None = None
    target_logits: object | None = None


class ProposalExecutor(Protocol):
    @property
    def capabilities(self) -> DraftCapabilities:
        pass

    def propose_batch(
        self,
        inputs: tuple[ModelRunnerProposalInput, ...],
    ) -> tuple[DraftProposal, ...]:
        pass


class ModelRunnerProposalExecutorRegistry:
    def __init__(self):
        self._entries: dict[str, tuple[ProposalExecutor, DraftCapabilities]] = {}

    def register(self, executor_id, executor, capabilities):
        normalized = validate_draft_capabilities(
            capabilities,
            expected_execution_domain="model_runner",
        )
        if not isinstance(executor_id, str) or not executor_id:
            raise ValueError("executor ID must be non-empty")
        if executor_id in self._entries:
            raise ValueError("executor ID is already registered")
        if getattr(executor, "capabilities", None) != normalized:
            raise ValueError("executor capabilities must exactly match")
        if not callable(getattr(executor, "propose_batch", None)):
            raise ValueError("executor must expose propose_batch")
        self._entries[executor_id] = (executor, normalized)

    def execute_batch(self, executor_id, inputs, capabilities):
        entry = self._entries.get(executor_id)
        if entry is None:
            raise ValueError("proposal executor is not registered")
        executor, registered = entry
        requested = validate_draft_capabilities(
            capabilities,
            expected_execution_domain="model_runner",
        )
        if requested != registered:
            raise ValueError("proposal executor capabilities mismatch")
        sequence_ids = validate_model_runner_proposal_inputs(
            inputs,
            requested,
        )
        proposals = validate_model_runner_proposals(
            executor.propose_batch(inputs),
            inputs,
            requested,
        )
        proposals_by_id = {
            proposal.sequence_id: proposal
            for proposal in proposals
        }
        return tuple(
            proposals_by_id[sequence_id]
            for sequence_id in sequence_ids
        )
```

Use a shared proposal-row validator rather than constructing an Engine-side
`DraftContext`; device payload presence is checked against
`ModelRunnerProposalInput`.

Implement recursive tensor detection for dataclasses, dictionaries, tuples,
lists, and sets without importing torch eagerly:

```python
def _is_tensor(value):
    value_type = type(value)
    return (
        value_type.__name__ == "Tensor"
        and value_type.__module__.startswith("torch")
    )
```

- [ ] **Step 4: Run the executor contract tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_model_runner_proposal_executor.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Check syntax and whitespace**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/speculative_proposal_executor.py
git diff --check -- \
  tinyvllm/engine/speculative_proposal_executor.py \
  tools/test_model_runner_proposal_executor.py
```

Expected: both commands exit 0.

---

### Task 3: Add Uniform First-Target/Proposal Rows and Bridges

**Files:**
- Modify: `tinyvllm/speculative/batch_runtime.py`
- Modify: `tinyvllm/engine/speculative_model_runner.py`
- Test: `tools/test_speculative_model_runner_callbacks.py`

**Interfaces:**
- Consumes: registry and tensor-free validator from Task 2.
- Produces:
  `FirstTargetProposalResult(sequence_id, target_token, proposal, first_target_metadata=None, proposal_metadata=None)`.
- Produces:
  `run_host_first_targets_and_proposals(...)`.
- Produces:
  `run_model_runner_first_targets_and_proposals(...)`.

- [ ] **Step 1: Write failing host and fused bridge tests**

Add tests equivalent to:

```python
def test_host_provider_combines_first_targets_and_adapter_proposals():
    rows = run_host_first_targets_and_proposals(
        seqs,
        adapter,
        lambda values: first_target_rows,
    )
    assert rows == (
        FirstTargetProposalResult(
            sequence_id=8,
            target_token=101,
            proposal=proposal_8,
            first_target_metadata={"batch_index": 0},
        ),
        FirstTargetProposalResult(
            sequence_id=4,
            target_token=201,
            proposal=proposal_4,
            first_target_metadata={"batch_index": 1},
        ),
    )


def test_fused_bridge_uses_one_rpc_and_restores_order():
    rows = run_model_runner_first_targets_and_proposals(
        runner,
        seqs,
        descriptor,
        identity_rows,
    )
    assert runner.calls == [
        (
            "run_spec_first_target_and_proposal_batch",
            (seqs, descriptor, identity_rows),
        )
    ]
    assert tuple(row.sequence_id for row in rows) == (8, 4)


def test_fused_bridge_rejects_nested_tensor_result():
    torch = pytest.importorskip("torch")
    result = FirstTargetProposalResult(
        sequence_id=8,
        target_token=101,
        proposal=DraftProposal(
            sequence_id=8,
            token_ids=(17,),
            source_type="fixture",
            metadata={"tensor": torch.zeros(1)},
        ),
    )
    runner = _FakeModelRunner(
        lambda method_name, args: (result,)
    )
    with pytest.raises(ValueError, match="tensor"):
        run_model_runner_first_targets_and_proposals(
            runner,
            (SimpleNamespace(seq_id=8),),
            _descriptor(),
        )


def test_fused_bridge_rejects_nested_proposal_id_mismatch():
    result = FirstTargetProposalResult(
        sequence_id=8,
        target_token=101,
        proposal=DraftProposal(
            sequence_id=4,
            token_ids=(17,),
            source_type="fixture",
        ),
    )
    runner = _FakeModelRunner(
        lambda method_name, args: (result,)
    )
    with pytest.raises(ValueError, match="proposal sequence ID"):
        run_model_runner_first_targets_and_proposals(
            runner,
            (SimpleNamespace(seq_id=8),),
            _descriptor(),
        )
```

Retain existing first-target bridge tests for host compatibility.

- [ ] **Step 2: Run focused callback tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_speculative_model_runner_callbacks.py -q
```

Expected: failures because the fused result and bridge functions do not
exist.

- [ ] **Step 3: Implement the result and bridge validators**

Add the result dataclass to `batch_runtime.py`:

```python
@dataclass(frozen=True)
class FirstTargetProposalResult:
    sequence_id: int
    target_token: int
    proposal: DraftProposal
    first_target_metadata: object | None = None
    proposal_metadata: object | None = None
```

Implement a shared validator in `speculative_model_runner.py` that:

```python
def _validate_first_target_proposal_results(
    results,
    sequence_ids,
    capabilities,
):
    assert_tensor_free(results, name="fused proposal result")
    if not isinstance(results, tuple):
        raise ValueError("fused proposal result must be a tuple")
    rows = {}
    for result in results:
        if not isinstance(result, FirstTargetProposalResult):
            raise ValueError(
                "fused rows must be FirstTargetProposalResult"
            )
        if result.sequence_id in rows:
            raise ValueError("fused sequence IDs must be unique")
        if result.proposal.sequence_id != result.sequence_id:
            raise ValueError("proposal sequence ID must match fused row")
        validate_proposal_for_capabilities(
            result.proposal,
            capabilities,
        )
        rows[result.sequence_id] = result
    if set(rows) != set(sequence_ids):
        raise ValueError(
            "fused result sequence IDs must exactly match input"
        )
    return tuple(rows[sequence_id] for sequence_id in sequence_ids)
```

The host bridge may internally use the existing first-target and adapter
validators. The fused bridge must call ModelRunner exactly once.

- [ ] **Step 4: Run callback and adapter tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_speculative_adapter.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Check syntax and whitespace**

Run:

```bash
python3 -m py_compile \
  tinyvllm/speculative/batch_runtime.py \
  tinyvllm/engine/speculative_model_runner.py
git diff --check -- \
  tinyvllm/speculative/batch_runtime.py \
  tinyvllm/engine/speculative_model_runner.py \
  tools/test_speculative_model_runner_callbacks.py
```

Expected: both commands exit 0.

---

### Task 4: Refactor the Generic Batch Runtime to Consume a Provider

**Files:**
- Modify: `tinyvllm/speculative/batch_runtime.py`
- Modify: `tinyvllm/speculative/__init__.py`
- Test: `tools/test_speculative_batch_runtime.py`
- Test: `tools/test_speculative_kv_transaction.py`
- Test: `tools/test_speculative_public_api.py`

**Interfaces:**
- Consumes:
  `FirstTargetProposalResult` from Task 3.
- Changes:
  `prepare_native_speculative_batch(..., run_first_targets_and_proposals, ...)`.
- Changes:
  `execute_native_speculative_batch(..., run_first_targets_and_proposals, ...)`.
- Removes proposal-source execution from the transaction runtime.

- [ ] **Step 1: Convert tests to the provider contract and add invariants**

Replace fixture pairs of `draft_adapter` and `run_first_targets` with:

```python
def run_first_targets_and_proposals(seqs):
    return tuple(
        FirstTargetProposalResult(
            sequence_id=seq.seq_id,
            target_token=100 + seq.seq_id,
            proposal=proposal_by_id[seq.seq_id],
            first_target_metadata={
                "sequence_id": seq.seq_id,
            },
        )
        for seq in seqs
    )
```

Add explicit tests:

```python
def test_provider_failure_creates_no_transaction():
    block_manager = _RecordingBlockManager()
    with pytest.raises(
        NativeSpeculativeBatchError,
        match="first_target_batch",
    ):
        prepare_native_speculative_batch(
            block_manager=block_manager,
            seqs=(_sequence(1),),
            eos_token=99,
            run_first_targets_and_proposals=lambda seqs: (
                (_ for _ in ()).throw(RuntimeError("provider failed"))
            ),
            run_tail_batch=lambda items: (),
        )
    assert block_manager.begin_calls == []


def test_empty_provider_proposal_creates_no_transaction():
    block_manager = _RecordingBlockManager()
    prepared = prepare_native_speculative_batch(
        block_manager=block_manager,
        seqs=(_sequence(1),),
        eos_token=99,
        run_first_targets_and_proposals=lambda seqs: (
            FirstTargetProposalResult(
                sequence_id=1,
                target_token=10,
                proposal=DraftProposal(1, (), "fixture"),
            ),
        ),
        run_tail_batch=lambda items: (),
    )
    assert block_manager.begin_calls == []
    assert prepared.sequences[0].plan is None


def test_mixed_provider_rows_share_existing_exact_q_tail_path():
    proposals = {
        1: DraftProposal(1, (11, 12), "fixture"),
        2: DraftProposal(2, (21, 22, 23), "fixture"),
    }
    prepared = prepare_native_speculative_batch(
        block_manager=_RecordingBlockManager(),
        seqs=(_sequence(1), _sequence(2)),
        eos_token=99,
        run_first_targets_and_proposals=lambda seqs: tuple(
            FirstTargetProposalResult(
                sequence_id=seq.seq_id,
                target_token=100 + seq.seq_id,
                proposal=proposals[seq.seq_id],
            )
            for seq in seqs
        ),
        run_tail_batch=_recording_tail_callback(),
    )
    assert tuple(
        row.plan.query_len for row in prepared.sequences
    ) == (1, 2)
```

- [ ] **Step 2: Run the batch-runtime tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_speculative_batch_runtime.py \
  tools/test_speculative_kv_transaction.py -q
```

Expected: failures because the runtime still expects a draft adapter and
separate first-target callback.

- [ ] **Step 3: Implement the provider-only preparation phase**

Change the preparation boundary to:

```python
def prepare_native_speculative_batch(
    *,
    block_manager,
    seqs,
    eos_token,
    run_first_targets_and_proposals,
    run_tail_batch,
) -> PreparedNativeSpeculativeBatch:
    sequence_ids = _validate_sequences(seqs)
    first_target_proposals = (
        _validate_first_target_proposal_results(
            run_first_targets_and_proposals(seqs),
            sequence_ids,
        )
    )
    return _prepare_transactions_and_tail(
        block_manager=block_manager,
        seqs=seqs,
        first_target_proposals=first_target_proposals,
        eos_token=eos_token,
        run_tail_batch=run_tail_batch,
    )
```

Replace the current first-target/context/adapter sequence with:

```python
started_at = time.perf_counter()
first_target_proposals = _validate_first_target_proposal_results(
    run_first_targets_and_proposals(seqs),
    sequence_ids,
)
timing_ms["first_target_batch_ms"] = (
    time.perf_counter() - started_at
) * 1000.0
timing_ms["draft_proposal_ms"] = sum(
    float(row.proposal.timing_ms.get("draft_ms", 0.0))
    for row in first_target_proposals
    if row.proposal.timing_ms is not None
)
```

Build transaction records from each row's proposal. Do not move transaction
creation before the provider callback succeeds.

Preserve the public timing keys even though the fused path cannot split
wall-clock target and proposal time without ModelRunner metadata.

- [ ] **Step 4: Run batch, transaction, and public API regressions**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_speculative_batch_runtime.py \
  tools/test_speculative_kv_transaction.py \
  tools/test_speculative_public_api.py -q
```

Expected: all tests pass; transactional KV assertions remain unchanged.

- [ ] **Step 5: Check syntax and whitespace**

Run:

```bash
python3 -m py_compile \
  tinyvllm/speculative/batch_runtime.py \
  tinyvllm/speculative/__init__.py
git diff --check -- \
  tinyvllm/speculative/batch_runtime.py \
  tinyvllm/speculative/__init__.py \
  tools/test_speculative_batch_runtime.py \
  tools/test_speculative_kv_transaction.py \
  tools/test_speculative_public_api.py
```

Expected: both commands exit 0.

---

### Task 5: Add Mutually Exclusive Engine Runtime Configuration

**Files:**
- Modify: `tinyvllm/engine/speculative_runtime.py`
- Test: `tools/test_engine_speculative_runtime.py`

**Interfaces:**
- Produces:
  `ModelRunnerProposalExecutorDescriptor`.
- Changes:
  `EngineSpeculativeRuntime` to accept exactly one of `draft_adapter` or
  `model_runner_executor`.
- Produces:
  `EngineSpeculativeRuntime.capabilities`.

- [ ] **Step 1: Write failing runtime configuration tests**

Add tests equivalent to:

```python
def test_host_runtime_exposes_host_capabilities():
    runtime = EngineSpeculativeRuntime(draft_adapter=_Adapter())
    assert runtime.capabilities.execution_domain == "host"


def test_model_runner_runtime_exposes_descriptor_capabilities():
    descriptor = ModelRunnerProposalExecutorDescriptor(
        executor_id="fixture",
        capabilities=_capabilities(
            execution_domain="model_runner",
        ),
    )
    runtime = EngineSpeculativeRuntime(
        model_runner_executor=descriptor,
    )
    assert runtime.capabilities is descriptor.capabilities


@pytest.mark.parametrize(
    "runtime",
    [
        EngineSpeculativeRuntime(),
        EngineSpeculativeRuntime(
            draft_adapter=_Adapter(),
            model_runner_executor=_descriptor(),
        ),
    ],
)
def test_runtime_requires_exactly_one_proposal_source(runtime):
    with pytest.raises(ValueError, match="exactly one"):
        build_engine_speculative_selection_config(
            runtime,
            model_runner=_Runner(),
        )


def test_model_runner_runtime_rejects_tp4():
    runner = _Runner(world_size=4, kv_offload_enabled=False)
    with pytest.raises(ValueError, match="TP1"):
        build_engine_speculative_selection_config(
            EngineSpeculativeRuntime(
                model_runner_executor=_descriptor(),
            ),
            model_runner=runner,
        )


def test_model_runner_runtime_rejects_kv_offload():
    runner = _Runner(world_size=1, kv_offload_enabled=True)
    with pytest.raises(ValueError, match="KV offload"):
        build_engine_speculative_selection_config(
            EngineSpeculativeRuntime(
                model_runner_executor=_descriptor(),
            ),
            model_runner=runner,
        )
```

- [ ] **Step 2: Run the focused runtime tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_engine_speculative_runtime.py -q
```

Expected: failures because the descriptor and alternate runtime source do
not exist.

- [ ] **Step 3: Implement descriptor and validation**

Add:

```python
@dataclass(frozen=True)
class ModelRunnerProposalExecutorDescriptor:
    executor_id: str
    capabilities: DraftCapabilities


@dataclass(frozen=True)
class EngineSpeculativeRuntime:
    draft_adapter: DraftAdapter | None = None
    model_runner_executor: (
        ModelRunnerProposalExecutorDescriptor | None
    ) = None
    lifecycle: DraftLifecycle | None = None

    @property
    def capabilities(self) -> DraftCapabilities:
        configured = tuple(
            value
            for value in (
                self.draft_adapter,
                self.model_runner_executor,
            )
            if value is not None
        )
        if len(configured) != 1:
            raise ValueError(
                "runtime must configure exactly one proposal source"
            )
        if self.draft_adapter is not None:
            return self.draft_adapter.capabilities
        return self.model_runner_executor.capabilities
```

For ModelRunner-domain validation, inspect:

```python
model_runner.world_size
model_runner.config.cpu_offload
```

and the actual configured KV-offload flag used by the existing runtime. Do
not guess a field name: locate it with `rg "kv_offload" tinyvllm/config.py
tinyvllm/engine/model_runner.py` before implementation and bind the gate to
the authoritative field.

- [ ] **Step 4: Run runtime tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_engine_speculative_runtime.py -q
```

Expected: all tests pass, including existing host runtime installation and
rollback behavior.

- [ ] **Step 5: Check syntax and whitespace**

Run:

```bash
python3 -m py_compile tinyvllm/engine/speculative_runtime.py
git diff --check -- \
  tinyvllm/engine/speculative_runtime.py \
  tools/test_engine_speculative_runtime.py
```

Expected: both commands exit 0.

---

### Task 6: Implement Fused ModelRunner Execution and Engine Routing

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/speculative_model_runner.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Test: `tools/test_model_runner_spec_verify.py`
- Test: `tools/test_speculative_model_runner_callbacks.py`
- Test: `tools/test_engine_speculative_runtime.py`

**Interfaces:**
- Consumes: executor registry from Task 2.
- Consumes: runtime descriptor from Task 5.
- Produces:
  `ModelRunner.register_speculative_proposal_executor(...)`.
- Produces:
  `ModelRunner.run_spec_first_target_and_proposal_batch(...)`.
- Engine selects host or ModelRunner provider by
  `runtime.capabilities.execution_domain`.

- [x] **Step 1: Write failing ModelRunner and Engine routing tests**

Add a constructed ModelRunner test that bypasses heavyweight initialization
using the existing test pattern:

```python
def test_fused_model_runner_keeps_hidden_local():
    runner = _constructed_runner()
    runner.speculative_proposal_executors.register(
        "fixture",
        executor,
        capabilities,
    )
    runner.run_model = lambda *args, **kwargs: (
        logits,
        hidden_states,
    )
    rows = runner.run_spec_first_target_and_proposal_batch(
        seqs,
        descriptor,
        identity_rows,
    )
    assert executor.calls[0][0].target_hidden is hidden_states[0]
    assert rows[0].proposal.token_ids == (17, 18)
    assert not hasattr(rows[0], "target_hidden")
```

Add Engine routing assertions:

```python
def test_engine_host_domain_uses_host_provider():
    events = []
    engine = _engine_with_runtime(
        EngineSpeculativeRuntime(draft_adapter=_Adapter())
    )
    _run_provider_selection(
        engine,
        host_provider=lambda *args: events.append("host"),
        model_runner_provider=lambda *args: events.append(
            "model_runner"
        ),
    )
    assert events == ["host"]


def test_engine_model_runner_domain_uses_fused_provider():
    events = []
    engine = _engine_with_runtime(
        EngineSpeculativeRuntime(
            model_runner_executor=_descriptor(),
        )
    )
    _run_provider_selection(
        engine,
        host_provider=lambda *args: events.append("host"),
        model_runner_provider=lambda *args: events.append(
            "model_runner"
        ),
    )
    assert events == ["model_runner"]


def test_engine_routing_never_checks_source_type():
    source = (
        Path(REPO_ROOT, "tinyvllm/engine/llm_engine.py")
        .read_text()
    )
    assert "source_type ==" not in source
    assert "source_type in" not in source
```

- [x] **Step 2: Run focused tests and confirm RED**

Run each file in a separate process to avoid the known attention-module stub
pollution:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_model_runner_spec_verify.py -q
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_speculative_model_runner_callbacks.py -q
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_engine_speculative_runtime.py -q
```

Expected: new tests fail because fused execution and routing do not exist.

- [x] **Step 3: Initialize and expose the registry**

In `ModelRunner.__init__`:

```python
self.speculative_proposal_executors = (
    ModelRunnerProposalExecutorRegistry()
)
```

Add:

```python
def register_speculative_proposal_executor(
    self,
    executor_id,
    executor,
    capabilities,
):
    self.speculative_proposal_executors.register(
        executor_id,
        executor,
        capabilities,
    )
```

Registration must occur consistently on all ranks before Engine activation.
The first slice may register only test/mock executors; do not add a real
checkpoint loader.

- [x] **Step 4: Implement the fused ModelRunner method**

Follow the existing `run_spec_first_target_batch()` preparation and KV
binding path. The method must:

```python
def run_spec_first_target_and_proposal_batch(
    self,
    seqs,
    descriptor,
    kv_block_identity_rows=(),
):
    # Validate TP1 and descriptor.
    # Bind KV identities.
    # prepare_decode -> offload guard -> run_model(return_hidden as needed).
    # Build ModelRunnerProposalInput rows locally.
    # Execute the registered executor once.
    # Return tensor-free FirstTargetProposalResult rows on rank 0 only.
    # Always reset_context() in finally.
```

Use the existing `_kv_offload_before_forward()` and
`_kv_offload_after_forward()` calls only to preserve current forward
structure. Runtime installation must already have rejected enabled KV
offload; do not claim offload support.

- [x] **Step 5: Route one provider into the generic runtime**

In `LLMEngine.step()`, replace direct adapter use with:

```python
capabilities = runtime.capabilities
if capabilities.execution_domain == "host":
    run_first_targets_and_proposals = lambda selected_seqs: (
        run_host_first_targets_and_proposals(
            self.model_runner,
            selected_seqs,
            runtime.draft_adapter,
            kv_block_identity_rows_for(selected_seqs),
        )
    )
elif capabilities.execution_domain == "model_runner":
    run_first_targets_and_proposals = lambda selected_seqs: (
        run_model_runner_first_targets_and_proposals(
            self.model_runner,
            selected_seqs,
            runtime.model_runner_executor,
            kv_block_identity_rows_for(selected_seqs),
        )
    )
else:
    raise RuntimeError("unsupported proposal execution domain")
```

Pass that callback to `prepare_native_speculative_batch()`. Do not inspect
`source_type` or model names.

- [x] **Step 6: Run focused integration tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_model_runner_spec_verify.py -q
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_speculative_model_runner_callbacks.py -q
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_engine_speculative_runtime.py -q
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_speculative_batch_runtime.py -q
```

Expected: each independent process passes.

- [x] **Step 7: Check syntax and scoped whitespace**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/speculative_model_runner.py \
  tinyvllm/engine/llm_engine.py
git diff --check -- \
  tinyvllm/engine/speculative_model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_engine_speculative_runtime.py
```

Expected: both commands exit 0. Do not include the entire pre-existing
`model_runner.py` in `git diff --check` because unrelated historical trailing
whitespace is already recorded; inspect only newly changed hunks separately.

---

### Task 7: Add Source-Neutrality and Ownership Regression Gates

**Files:**
- Create or Modify: `tools/test_model_runner_proposal_executor.py`
- Modify: `tools/test_engine_speculative_runtime.py`
- Modify: `tools/test_speculative_model_runner_callbacks.py`

**Interfaces:**
- Verifies the complete first-slice ownership and genericity boundary.
- Produces no new runtime behavior.

- [x] **Step 1: Add static source-neutrality tests**

Add a test that reads only the generic files:

```python
GENERIC_FILES = (
    "tinyvllm/engine/scheduler.py",
    "tinyvllm/speculative/batch_runtime.py",
    "tinyvllm/speculative/verifier.py",
    "tinyvllm/engine/block_manager.py",
    "tinyvllm/engine/speculative_model_runner.py",
)


def test_generic_runtime_has_no_model_or_source_dispatch():
    forbidden = (
        "qwen",
        "llama",
        "mtp",
        "learned",
        'source_type ==',
        'source_type in',
    )
    for relative_path in GENERIC_FILES:
        source = (REPO_ROOT / relative_path).read_text().lower()
        for needle in forbidden:
            assert needle not in source
```

Before adopting a broad token such as `"mtp"`, inspect existing comments and
identifiers. Narrow the assertion to dispatch expressions if the token
already exists for documentation rather than behavior.

- [x] **Step 2: Add tensor ownership and transaction-order tests**

Cover:

- ModelRunner executor sees local hidden/logits when required.
- Fused result recursive scan rejects tensors in metadata and timing.
- Engine receives no target payload fields.
- Executor error occurs before any
  `begin_speculative_kv_transaction()` call.
- Empty proposal creates no transaction.
- Non-empty proposal reaches the unchanged exact-Q verifier and commit path.

- [x] **Step 3: Run ownership and genericity tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tools/test_model_runner_proposal_executor.py \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_engine_speculative_runtime.py -q
```

Expected: all tests pass.

- [x] **Step 4: Run the first-slice independent-process regression matrix**

Run each command separately:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tools/test_speculative_adapter.py -q
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tools/test_speculative_source_adapters.py -q
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tools/test_model_runner_proposal_executor.py -q
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tools/test_speculative_batch_runtime.py -q
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tools/test_engine_speculative_runtime.py -q
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tools/test_speculative_model_runner_callbacks.py -q
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tools/test_speculative_kv_transaction.py -q
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tools/test_speculative_residency_boundary_gate.py -q
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tools/test_speculative_public_api.py -q
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tools/test_model_runner_spec_verify.py -q
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tools/test_model_runner_spec_verify_cuda_graph.py -q
```

Expected: every independent process passes. Record per-file counts; do not
combine files that install incompatible import stubs.

- [x] **Step 5: Run final syntax, placeholder, and scoped diff checks**

Run:

```bash
python3 -m py_compile \
  tinyvllm/speculative/adapter.py \
  tinyvllm/speculative/batch_runtime.py \
  tinyvllm/engine/speculative_proposal_executor.py \
  tinyvllm/engine/speculative_runtime.py \
  tinyvllm/engine/speculative_model_runner.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py

rg -n "TBD|TODO|FIXME|NotImplementedError" \
  tinyvllm/engine/speculative_proposal_executor.py \
  tinyvllm/engine/speculative_runtime.py \
  tinyvllm/engine/speculative_model_runner.py \
  tools/test_model_runner_proposal_executor.py

git diff --check -- \
  tinyvllm/speculative/adapter.py \
  tinyvllm/speculative/batch_runtime.py \
  tinyvllm/engine/speculative_proposal_executor.py \
  tinyvllm/engine/speculative_runtime.py \
  tinyvllm/engine/speculative_model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_speculative_adapter.py \
  tools/test_model_runner_proposal_executor.py \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_speculative_batch_runtime.py \
  tools/test_engine_speculative_runtime.py
```

Expected:

- `py_compile` exits 0;
- placeholder scan has no output;
- scoped `git diff --check` exits 0.

- [x] **Step 6: Update handoff and audit boundaries**

Append the completed first-slice evidence to:

- `AGENT_HANDOFF_STATE.md`
- `docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md`

Record:

- exact files changed;
- independent-process test counts;
- ownership guarantees proved;
- TP1/no-offload/exact-Q limits;
- no real learned/MTP checkpoint;
- no CUDA or performance claim;
- the next required real executor and GPU gates.

Do not mark Phase 1 complete or promotable.

---

## Plan Self-Review Checklist

- [x] Every selected design requirement maps to a task.
- [x] Host n-gram/SAM compatibility is covered in Tasks 1, 3, 4, and 7.
- [x] CUDA tensor ownership is covered in Tasks 2, 3, 6, and 7.
- [x] ModelRunner executor configuration and TP1/no-offload gates are covered
  in Tasks 5 and 6.
- [x] Transaction creation remains after successful proposal execution in
  Task 4.
- [x] Exact-Q verification and in-place KV commit/rollback are regression
  requirements in Tasks 4 and 7.
- [x] No task adds a real model loader, TP4 support, KV offload, or
  performance claim.
- [x] No task contains a git stage, commit, push, reset, clean, stash, branch,
  or worktree operation.
- [x] No task uses subagents.
