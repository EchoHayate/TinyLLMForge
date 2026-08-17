# Source-Neutral ModelRunner Proposal Executor Design

**Date:** 2026-08-12

**Status:** Selected continuation design

## Objective

Add a source-neutral proposal execution boundary that lets learned draft
models and MTP heads generate speculative proposals inside `ModelRunner`
without exposing CUDA hidden states or logits to `LLMEngine`, `Scheduler`, or
the generic speculative batch runtime.

The design must preserve the existing host-side n-gram and SAM adapters and
must reuse the current batch-native verifier and transactional KV
commit/rollback path.

## First-Slice Scope

The first implementation slice is deliberately limited to:

- tensor parallel size 1;
- KV offload disabled;
- greedy decoding;
- exact proposal lengths and exact-Q verifier families;
- one ModelRunner-resident mock proposal executor;
- batch sizes already supported by the generic speculative runtime;
- proposal output through the existing `DraftProposal` contract.

The slice does not claim:

- a real learned checkpoint;
- a real MTP checkpoint or head;
- TP4 correctness;
- KV-offload compatibility;
- performance improvement;
- H2D or D2H reduction;
- promotion readiness.

## Existing Problem

The current generic runtime performs these operations:

1. `LLMEngine` calls `run_spec_first_target_batch`.
2. `ModelRunner` optionally returns target hidden states or logits.
3. `LLMEngine` constructs `DraftContext` rows containing those payloads.
4. `DraftAdapter.propose_batch()` runs in the Engine process.
5. The runtime reserves transactional KV and executes the tail verifier.

This is correct for host-only adapters such as n-gram and SAM. It is not a
safe ownership boundary for learned or MTP proposal sources because CUDA
tensors would leave `ModelRunner` and become visible to the Engine process.
It also creates pressure to serialize device payloads through the
ModelRunner command boundary.

## Considered Architectures

### 1. Fused First-Target and Proposal RPC

`ModelRunner` executes the target first-token forward, keeps any required
hidden states or logits local, invokes a registered proposal executor, and
returns only first-target tokens plus validated `DraftProposal` rows.

This is the selected design. It has no cross-call tensor handle lifetime,
does not expose CUDA tensors, and preserves the existing transaction and
verification phases.

### 2. Two-Stage Opaque Handle

The first-target RPC returns an opaque handle that is passed back in a
second proposal RPC.

This was rejected for the first slice because it requires handle expiry,
batch identity, retry, worker restart, and exception cleanup semantics
without adding correctness value.

### 3. Move Every Adapter into ModelRunner

n-gram, SAM, learned models, and MTP would all execute in `ModelRunner`.

This was rejected because it couples CPU-only proposal sources to the GPU
worker lifecycle and adds unnecessary token-history transport.

## Capability Contract

`DraftCapabilities` gains a source-neutral execution domain:

```python
@dataclass(frozen=True)
class DraftCapabilities:
    source_type: str
    supports_batch: bool
    requires_target_hidden: bool
    requires_target_logits: bool
    max_proposal_tokens: int
    execution_domain: str = "host"
```

Allowed values in the first slice are:

- `"host"` for existing Engine-side adapters;
- `"model_runner"` for ModelRunner-resident executors.

The field defaults to `"host"` so existing adapters and fixtures remain
compatible.

Execution domain must not be inferred from
`requires_target_hidden`/`requires_target_logits`. An independent GPU draft
model may execute in `ModelRunner` without consuming target hidden states.

Generic runtime code may branch on `execution_domain`. It must not branch on
model names, checkpoint names, `source_type`, learned-drafter names, or MTP
names.

## Runtime Descriptor

`EngineSpeculativeRuntime` supports exactly one of two proposal sources:

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
```

Validation requires:

- exactly one proposal source is configured;
- a host adapter has `execution_domain == "host"`;
- a ModelRunner descriptor has
  `execution_domain == "model_runner"`;
- descriptor capabilities support batch proposals;
- `executor_id` is a non-empty source-neutral registry key;
- Scheduler and proposal limits match;
- TP size is 1 for the first slice;
- KV offload is disabled for the first slice.

`executor_id` selects an already-constructed executor. It is not a model
name dispatch branch in Scheduler or the batch runtime.

## ModelRunner Executor Contract

The ModelRunner-local protocol is:

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
    def capabilities(self) -> DraftCapabilities: ...

    def propose_batch(
        self,
        inputs: tuple[ModelRunnerProposalInput, ...],
    ) -> tuple[DraftProposal, ...]: ...
```

`ModelRunnerProposalInput` is internal to the ModelRunner process. Its
hidden/logit fields may contain CUDA tensors and must never be returned from
the ModelRunner call.

The first slice registers a deterministic mock executor through an explicit
ModelRunner API. Registration is capability-based and validates that the
descriptor and executor capabilities match exactly.

## Fused Result Contract

The fused bridge returns:

```python
@dataclass(frozen=True)
class FirstTargetProposalResult:
    sequence_id: int
    target_token: int
    proposal: DraftProposal
    first_target_metadata: object | None = None
    proposal_metadata: object | None = None
```

The result must not contain:

- target hidden states;
- target logits;
- CUDA tensors;
- opaque device handles;
- executor objects.

Both ModelRunner and Engine-side bridge validation require:

- tuple result type;
- exact input/result sequence-ID equality;
- unique sequence IDs;
- stable restoration to input order;
- nested proposal sequence ID equal to the row sequence ID;
- proposal source type equal to descriptor capabilities;
- proposal length within capability and per-request limits;
- integer, non-boolean token IDs;
- valid non-negative finite timing fields;
- no tensor reachable from the returned row.

## Provider Boundary

The generic batch runtime consumes one provider callback rather than owning
proposal-source routing:

```python
run_first_targets_and_proposals(
    seqs,
) -> tuple[FirstTargetProposalResult, ...]
```

Two provider implementations produce this common result:

### Host Provider

1. Call the existing `run_spec_first_target_batch`.
2. Build the existing Engine-side `DraftContext` rows.
3. Call and validate the host `DraftAdapter`.
4. Combine first-target and proposal rows.

This preserves n-gram and SAM behavior.

### ModelRunner Provider

1. Call the fused ModelRunner method once for the selected batch.
2. ModelRunner executes the normal target decode forward.
3. ModelRunner invokes the registered executor using local target payloads.
4. ModelRunner validates proposals.
5. The bridge validates tensor-free fused rows.

The generic batch runtime is unaware of which provider produced the rows.

## Data Flow

For a ModelRunner executor:

1. Scheduler selects eligible decode sequences using only generic proposal
   limits and runtime health.
2. Engine builds sequence metadata and exact KV identity rows.
3. Engine invokes the fused ModelRunner callback.
4. Every ModelRunner rank receives the same source-neutral executor ID and
   batch metadata.
5. TP1 target first-token forward writes the normal target KV.
6. Local hidden/logit payloads are passed only to the local executor.
7. Rank 0 returns tensor-free fused rows.
8. Engine validates and reorders rows.
9. Empty proposals continue as ordinary first-target decode.
10. Non-empty proposals begin private speculative KV transactions.
11. Existing exact-Q tail verification executes.
12. Existing acceptance logic commits accepted KV in place and rolls back
    rejected suffix KV.

No speculative transaction exists during first-target or proposal
execution.

## Error Semantics

Configuration errors fail during runtime installation:

- unsupported execution domain;
- both or neither proposal sources configured;
- missing executor;
- capability mismatch;
- TP size other than 1;
- KV offload enabled;
- non-greedy configuration.

Runtime errors before transaction creation fail closed:

- target forward failure;
- executor failure;
- invalid proposal output;
- tensor leakage in the fused result;
- result identity mismatch.

These errors are not converted into empty proposals. Empty proposals are
valid only when deliberately returned by a successful executor.

Existing transaction error semantics remain authoritative after reserve:

- pre-replay failures may follow the existing eager fallback boundary;
- replay-started CUDA failures must not retry eagerly;
- accepted KV commits in place;
- rejected suffix KV rolls back;
- rollback failure poisons the speculative runtime.

## ModelRunner Rank Semantics

`ModelRunner.call()` broadcasts method name and arguments to worker ranks.
The executor must therefore be registered consistently on every rank before
the fused call is allowed.

For the first TP1-only slice:

- rank-consistency validation is represented in the API and tests;
- no TP4 correctness claim is made;
- only rank 0 returns fused result rows;
- non-zero-rank return values remain `None`, consistent with current target
  forward behavior.

Future TP support must define whether the executor consumes sharded or
replicated hidden state and how proposal-token agreement is enforced.

## Exact-Q and Transactional KV Invariants

This design does not change:

- exact proposal length;
- one exact-Q verifier family per distinct proposal tail length;
- no padding, rounding, or merging of Q families;
- private speculative KV reservation;
- accepted-token in-place commit;
- rejected-suffix rollback;
- no accepted-KV replay, copy, or per-token rematerialization;
- graph contents limited to target transformer forward and KV writes;
- logits, acceptance, and transaction finalize outside the graph.

## Testing Strategy

### Capability and Runtime Tests

- default execution domain remains `"host"`;
- invalid execution domains fail;
- host adapter and ModelRunner descriptor are mutually exclusive;
- host/domain mismatch fails;
- descriptor/domain mismatch fails;
- executor ID and capability validation;
- TP1 and no-offload gates;
- Scheduler proposal-limit agreement.

### Executor Contract Tests

- deterministic batched proposal generation;
- exact sequence-ID match and input-order restoration;
- empty proposals;
- mixed proposal lengths;
- capability and request-budget limits;
- required hidden/logit payloads;
- invalid tokens and timing fields;
- executor exceptions propagate before transaction creation.

### Fused Bridge Tests

- exactly one fused ModelRunner RPC per selected batch;
- exact KV identity rows are forwarded;
- no hidden/logit argument is returned to Engine;
- recursive tensor-leak detection;
- nested proposal/row ID consistency;
- invalid, duplicate, missing, and extra result rows fail;
- rank-0 return contract.

### Batch Runtime Tests

- host provider preserves existing n-gram/SAM behavior;
- ModelRunner provider reaches the same transaction/verifier path;
- empty proposal creates no transaction;
- non-empty proposal creates one transaction per sequence;
- mixed batch ordering remains stable;
- exact-Q tail grouping remains unchanged;
- executor failure creates no transaction;
- commit and rollback behavior remains unchanged.

### Source-Neutrality Tests

Static scans reject model-name or proposal-source branches in:

- Scheduler;
- generic batch runtime;
- verifier;
- transactional KV manager;
- ModelRunner callback bridge.

### Regression Tests

Run existing focused suites for:

- draft adapter;
- source adapters;
- speculative batch runtime;
- Engine speculative runtime;
- ModelRunner callbacks;
- transactional KV;
- speculative residency;
- exact-Q CUDA Graph configuration and CPU schema.

Real CUDA correctness and performance gates remain separately blocked until
an idle GPU is available.

## Implementation Boundaries

Expected production files:

- `tinyvllm/speculative/adapter.py`
- `tinyvllm/speculative/batch_runtime.py`
- `tinyvllm/engine/speculative_runtime.py`
- `tinyvllm/engine/speculative_model_runner.py`
- `tinyvllm/engine/model_runner.py`
- `tinyvllm/engine/llm_engine.py`

Expected focused test files:

- `tools/test_speculative_adapter.py`
- `tools/test_speculative_batch_runtime.py`
- `tools/test_engine_speculative_runtime.py`
- `tools/test_speculative_model_runner_callbacks.py`
- a focused ModelRunner proposal-executor contract test file

No Scheduler model-specific branch, new KV policy, CUDA Graph family, real
checkpoint loader, or performance gate belongs in this slice.

## Promotion Boundary

Completion of this slice proves only that a ModelRunner-owned proposal
source can participate in the generic speculative runtime without exporting
target CUDA payloads and while preserving transactional KV semantics.

Promotion still requires:

- a real learned drafter;
- a real MTP executor;
- two model structures;
- TP1 and TP4;
- 4K, 16K, and 32K contexts;
- batch 1, batch 4, and multi-sequence workloads;
- exact greedy parity;
- TPOT, TTFT, throughput, memory, real KV H2D bytes, and acceptance metrics;
- real CUDA variable-Q correctness and performance evidence;
- explicit proof that any KV movement is real rather than simulated.
