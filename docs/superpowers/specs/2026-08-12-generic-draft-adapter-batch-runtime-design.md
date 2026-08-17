# Generic Draft Adapter and Batch Runtime Design

**Date:** 2026-08-12  
**Status:** Approved by the existing model-independent optimization direction  
**Scope:** Dependency-light adapter and batch orchestration core; no model-name
branches and no performance claims

## Goal

Define one runtime-facing draft proposal contract for n-gram, SAM, independent
draft models, and MTP heads, then add a batch-native speculative orchestration
core that:

- executes one normal first-target batch;
- asks an injected adapter for per-sequence proposals;
- reserves private transactional KV for each non-empty proposal;
- executes one variable-length tail-verifier batch;
- commits each accepted prefix directly;
- rolls back each rejected or failed suffix without accepted-KV
  rematerialization.

This design is the next layer above
`execute_native_speculative_step()`. It does not replace the single-sequence
core until production integration proves equivalent behavior.

## Non-Goals

- Loading or implementing a specific MTP head.
- Loading a specific independent draft model.
- Adding model-name checks.
- Changing sampling away from exact greedy acceptance.
- Integrating CUDA Graphs, TP collectives, KV precision tiers, or offload
  residency in this slice.
- Claiming TPOT, TTFT, throughput, memory, or H2D improvement.
- Modifying Qwen3.5-specific projection code.

## Alternatives

### A. Put drafter branches directly in `Scheduler`

The scheduler would inspect a source name and call n-gram, SAM, learned-model,
or MTP code directly.

Rejected because it couples queue policy to model execution and would require
model-name/source-name branches throughout scheduling.

### B. Loop over `execute_native_speculative_step()`

Each sequence would use the existing single-sequence runtime independently.

This is useful as a correctness oracle, but rejected as the production batch
shape because it performs first-target and tail callbacks per sequence. It
cannot prove a batch-native verifier or amortize target-forward overhead.

### C. Staged batch runtime with an injected adapter

The engine executes first targets for the selected decode batch, builds
adapter contexts, obtains proposals, starts per-sequence transactions, and
executes one tail callback over all non-empty verifier plans.

Selected because it isolates proposal policy, preserves per-sequence KV
ownership, and gives the model runner one explicit batch verifier boundary.

## Components

### 1. Draft adapter contract

Create `tinyvllm/speculative/adapter.py`.

```python
@dataclass(frozen=True)
class DraftCapabilities:
    source_type: str
    supports_batch: bool
    requires_target_hidden: bool
    requires_target_logits: bool
    max_proposal_tokens: int


@dataclass(frozen=True)
class DraftContext:
    sequence_id: int
    token_ids: tuple[int, ...]
    remaining_output_tokens: int
    max_proposal_tokens: int
    first_target_token: int
    target_hidden: object | None = None
    target_logits: object | None = None


@dataclass(frozen=True)
class DraftProposal:
    sequence_id: int
    token_ids: tuple[int, ...]
    source_type: str
    metadata: object | None = None
    timing_ms: dict[str, float] | None = None


class DraftAdapter(Protocol):
    @property
    def capabilities(self) -> DraftCapabilities: ...

    def propose_batch(
        self,
        contexts: tuple[DraftContext, ...],
    ) -> tuple[DraftProposal, ...]: ...
```

The runtime validates:

- sequence IDs are unique and exactly match the input contexts;
- token IDs are integers, not booleans;
- proposal lengths do not exceed the context or capability limits;
- required hidden/logit payloads are present;
- timing values are finite and non-negative;
- the adapter does not mutate sequence metadata.

An empty `token_ids` tuple means ordinary first-target decode with no
speculative tail.

`remaining_output_tokens` is advisory input to the adapter. The adapter may
return a longer bounded proposal; the runtime remains authoritative for
output-budget truncation so that acceptance and cleanup semantics do not
depend on adapter behavior.

### 2. First-target batch result

Create immutable runtime types in
`tinyvllm/speculative/batch_runtime.py`.

```python
@dataclass(frozen=True)
class FirstTargetResult:
    sequence_id: int
    target_token: int
    target_hidden: object | None = None
    target_logits: object | None = None
    metadata: object | None = None


@dataclass(frozen=True)
class TailBatchItem:
    sequence_id: int
    plan: SpecVerifyPlan
    proxy_block_table: tuple[int, ...]


@dataclass(frozen=True)
class TailBatchResult:
    sequence_id: int
    target_tokens: tuple[int, ...]
    metadata: object | None = None
    auxiliary: object | None = None
```

The batch callbacks are:

```python
run_first_targets(
    seqs: tuple[object, ...],
) -> tuple[FirstTargetResult, ...]

run_tail_batch(
    items: tuple[TailBatchItem, ...],
) -> tuple[TailBatchResult, ...]
```

The callback result order is not trusted. Results are matched by unique
`sequence_id`.

### 3. Batch orchestration

```python
execute_native_speculative_batch(
    *,
    block_manager,
    seqs,
    draft_adapter,
    eos_token,
    run_first_targets,
    run_tail_batch,
) -> NativeSpeculativeBatchResult
```

Data flow:

1. Validate unique sequence IDs and callback/adapter shape.
2. Execute one first-target callback for the full selected batch.
3. Build immutable `DraftContext` rows from pre-commit sequence history,
   remaining output budget, and first-target payloads.
4. Execute one adapter batch proposal.
5. For every non-empty proposal:
   - build `SpecVerifyPlan`;
   - begin `SpeculativeKVTransaction`;
   - construct the private proxy block table;
   - add one `TailBatchItem` when `query_len > 0`.
6. Execute at most one tail callback for all tail items.
7. Validate all tail rows before materializing or committing any transaction.
8. Mark each transaction with its exact `query_len`.
9. Compute exact greedy accepted prefixes, EOS truncation, and output-budget
   truncation independently per sequence.
10. Commit accepted prefixes in stable input order.
11. Roll back all uncommitted transactions on failure.
12. Return per-sequence results plus batch timing and callback counts.

K=1 proposals have `query_len == 0`, skip the tail callback, mark zero
materialized KV, and may still accept the first draft token.

Empty proposals do not create transactions. Their first target is returned as
ordinary-decode output for the future engine adapter; the dependency-light
core does not append it to `Sequence`.

## Result Contract

```python
@dataclass(frozen=True)
class NativeSpeculativeSequenceResult:
    sequence_id: int
    first_target_token: int
    proposal: DraftProposal
    plan: SpecVerifyPlan | None
    target_tokens: tuple[int, ...]
    greedy_accepted_count: int
    accepted_tokens: tuple[int, ...]
    eos_truncated: bool
    output_budget_truncated: bool
    reserved_blocks: tuple[int, ...]
    proxy_block_table: tuple[int, ...]
    committed_blocks: tuple[int, ...]
    released_blocks: tuple[int, ...]
    tail_metadata: object | None = None
    tail_auxiliary: object | None = None


@dataclass(frozen=True)
class NativeSpeculativeBatchResult:
    sequences: tuple[NativeSpeculativeSequenceResult, ...]
    first_target_callback_count: int
    tail_callback_count: int
    timing_ms: dict[str, float]
```

The sequence result order always matches the input `seqs` order.

## Failure and Rollback Semantics

`NativeSpeculativeBatchError` records:

- phase;
- original cause;
- sequence IDs whose transactions were already committed;
- sequence IDs successfully rolled back;
- per-sequence rollback failures.

Phases are:

- `first_target_batch`;
- `draft_proposal`;
- `reserve`;
- `tail_batch`;
- `kv_materialize`;
- `acceptance`;
- `metadata_commit`.

No transaction exists during first-target or proposal failures.

Reservation and tail failures roll back every started transaction.
Validation of the complete tail result set occurs before any materialization.
Materialization failures roll back all active transactions.

Commits occur in stable sequence order. If a later commit fails, already
committed sequences remain committed because the allocator API has no inverse
commit. All not-yet-committed transactions are rolled back and the error
explicitly reports the partial commit set. Production integration must treat
this as a fatal engine invariant failure, not retry the batch.

Rollback failure never hides the original failure.

## Scheduler and Engine Boundary

This slice does not change `Scheduler.schedule()` yet. The subsequent
integration will:

1. let the scheduler select eligible running decode sequences using only
   capability and budget metadata;
2. let `LLMEngine.step()`/`ModelRunner` execute the first-target batch;
3. call the adapter and batch runtime;
4. pass committed sequence results to scheduler postprocessing;
5. keep ordinary decode for empty proposals or ineligible sequences.

The scheduler never imports n-gram, SAM, learned-model, MTP, CUDA, or model
classes.

## Testing

### Adapter tests

Create `tools/test_speculative_adapter.py` covering:

- capability validation;
- exact context/proposal ID matching;
- hidden/logit requirements;
- proposal length and remaining-budget bounds;
- invalid tokens and timing values;
- empty proposals;
- no sequence mutation.

### Batch runtime tests

Create `tools/test_speculative_batch_runtime.py` covering:

- one first-target callback for batch 1 and batch 4;
- one tail callback for mixed K=0/K=1/K>1 rows;
- out-of-order callback results matched by sequence ID;
- zero/one/partial/full acceptance independently per row;
- EOS and output-budget truncation;
- exact transaction begin/materialize/commit/rollback counts;
- direct accepted-KV commit and rejected suffix release;
- first-target, adapter, reserve, tail, materialize, acceptance, and commit
  failures;
- rollback failure preserving the original cause;
- explicit partial-commit reporting;
- no model names in the core modules.

### Regression

Run:

```text
python3 -m pytest -q tools/test_speculative_adapter.py
python3 -m pytest -q tools/test_speculative_batch_runtime.py
python3 -m pytest -q tools/test_speculative_runtime.py
python3 -m pytest -q tools/test_speculative_kv_transaction.py
python3 -m pytest -q tools/test_ngram_speculative.py
```

## Promotion Boundary

Passing this slice proves only model-independent batch ownership and callback
semantics. It does not prove:

- production scheduler or model-runner integration;
- a real MTP or learned draft model;
- GPU, TP1, or TP4 correctness;
- long-context behavior;
- exact greedy model parity;
- TPOT, TTFT, throughput, memory, real KV H2D, or acceptance improvement.
