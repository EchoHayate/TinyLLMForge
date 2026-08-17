# Scheduler-Visible Speculative Selection Record Design

**Date:** 2026-08-12  
**Status:** Approved by the existing generic speculative-runtime direction  
**Scope:** Source-agnostic scheduler selection metadata only; no target
execution, proposal generation, transaction execution, or sequence commit

## Goal

Give `Scheduler` an immutable, validated record that identifies which rows of
an already selected ordinary batch are eligible for a speculative attempt.

The record must:

- preserve the existing dynamic three/four-element `schedule()` return shape;
- carry no n-gram, SAM, learned-drafter, MTP, or model-name logic;
- distinguish decode rows from prefill rows in ordinary and mixed batches;
- cap requested speculative work by a generic configured maximum and output
  budget;
- snapshot sequence identity and token counts so `LLMEngine` can reject stale
  selection;
- remain observability-only until the real engine/runtime integration is
  implemented.

## Non-Goals

- Generating draft tokens inside `Scheduler`.
- Importing concrete draft adapters into `Scheduler`.
- Beginning speculative KV transactions during scheduling.
- Running first-target or tail-verifier model batches.
- Committing output tokens or synchronizing SAM state.
- Adding speculative fields to `Sequence` serialization.
- Replacing the existing scheduler return tuple in this slice.
- Claiming latency, throughput, memory, or KV-movement improvement.

## Current Constraints

`Scheduler.schedule()` currently returns:

```text
(seqs, is_prefill, do_sample)
(seqs, is_prefill, do_sample, "mixed")
```

`LLMEngine.step()` checks `len(scheduled)` and unpacks either shape. Existing
scheduler tests also index and unpack these tuples directly.

Mixed batches mark decode rows with `seq.step_is_decode`; ordinary decode
batches use `is_prefill == False`. Final prefill rows may sample the first
output token, but they are still prefill execution and are excluded from this
initial speculative-selection slice.

## Alternatives

### A. Store speculative selection on each `Sequence`

Rejected because `Sequence` is serialized to worker processes and already
contains mutable per-step prefill/decode fields. Adding proposal policy and
selection lifecycle there would create stale state and expand the wire
contract.

### B. Add a fifth scheduler tuple element or replace the tuple

Rejected for this slice because the current three/four-element shape is used
throughout engine and scheduler tests. A new envelope may be appropriate
later, but it is unnecessary to establish the selection contract.

### C. Publish an immutable generation-bound sidecar record

Selected. `Scheduler` keeps the legacy return value and synchronously updates
`last_speculative_selection` from the same scheduled batch. The record carries
the scheduler generation and exact sequence snapshots. `LLMEngine` must read
and validate it immediately after `schedule()`.

The sidecar owns no resources, so reading it is not destructive and does not
need transaction-like exactly-once semantics.

## Components

Create `tinyvllm/engine/speculative_selection.py`.

### Configuration

```python
@dataclass(frozen=True)
class SpeculativeSelectionConfig:
    enabled: bool
    max_proposal_tokens: int
```

Validation:

- `enabled` must be boolean;
- disabled configuration requires `max_proposal_tokens == 0`;
- enabled configuration requires integer, non-boolean
  `max_proposal_tokens >= 2`.

The minimum of two expresses scheduler intent to attempt a multi-token
speculative step. The concrete adapter may still return zero or one token;
proposal quality remains outside the scheduler.

No source identity appears in this configuration. The future engine owner
binds one configured adapter to the selected batch and verifies the selection
maximum does not exceed adapter capabilities.

### Per-row record

```python
@dataclass(frozen=True)
class SpeculativeSelectionRow:
    sequence_id: int
    batch_index: int
    token_count_snapshot: int
    completion_token_count_snapshot: int
    remaining_output_tokens: int
    selected: bool
    max_proposal_tokens: int
    suppression_reason: str | None
```

Selected rows require:

```text
selected = true
max_proposal_tokens >= 2
suppression_reason = None
```

Suppressed rows require:

```text
selected = false
max_proposal_tokens = 0
suppression_reason = one non-empty stable reason
```

Initial suppression reasons:

- `disabled`;
- `prefill`;
- `not_sampling`;
- `insufficient_output_budget`.

`remaining_output_tokens` is:

```python
max(0, seq.max_tokens - seq.num_completion_tokens)
```

For an eligible decode row:

```python
max_proposal_tokens = min(
    config.max_proposal_tokens,
    remaining_output_tokens,
)
```

Rows with fewer than two remaining output tokens are suppressed because the
ordinary first-target step already satisfies the remaining request and there
is no multi-token speculative opportunity.

### Batch record

```python
@dataclass(frozen=True)
class SpeculativeSelectionRecord:
    schedule_generation: int
    policy_branch: str
    is_prefill: bool
    do_sample: bool
    batch_kind: str | None
    scheduled_sequence_ids: tuple[int, ...]
    rows: tuple[SpeculativeSelectionRow, ...]

    @property
    def selected_rows(
        self,
    ) -> tuple[SpeculativeSelectionRow, ...]: ...
```

Record order exactly matches scheduled batch order. Every scheduled sequence
has exactly one row, including suppressed prefill and output-budget rows.

### Builder

```python
def build_speculative_selection_record(
    *,
    seqs: tuple[object, ...],
    is_prefill: bool,
    do_sample: bool,
    batch_kind: str | None,
    policy_branch: str,
    schedule_generation: int,
    config: SpeculativeSelectionConfig,
) -> SpeculativeSelectionRecord: ...
```

The builder is dependency-light and reads only:

- `seq_id`;
- `num_tokens`;
- `num_completion_tokens`;
- `max_tokens`;
- mixed-batch `step_is_decode`;
- mixed-batch `step_do_sample`.

Eligibility:

1. disabled config suppresses every row;
2. ordinary prefill suppresses every row;
3. mixed prefill rows are suppressed;
4. non-sampling rows are suppressed;
5. rows with fewer than two remaining output tokens are suppressed;
6. all other decode rows are selected.

The builder does not inspect free blocks, adapter source, target hidden state,
target logits, token contents, model type, or KV residency.

### Stale-record validation

```python
def validate_speculative_selection_record(
    record: SpeculativeSelectionRecord,
    seqs: tuple[object, ...],
    *,
    expected_schedule_generation: int,
) -> tuple[object, ...]: ...
```

Validation requires:

- exact sequence ID coverage and order;
- matching expected generation;
- matching `num_tokens`;
- matching `num_completion_tokens`;
- selected-row output budget still at least two;
- selected row maximum no greater than current remaining budget.

It returns selected sequence objects in record order. Validation failure occurs
before proposal generation or KV transaction creation.

## Scheduler Integration

Add default-disabled scheduler state:

```python
self.speculative_selection_config = (
    SpeculativeSelectionConfig(
        enabled=False,
        max_proposal_tokens=0,
    )
)
self.schedule_generation = 0
self.last_speculative_selection = None
```

Add:

```python
def install_speculative_selection(
    self,
    config: SpeculativeSelectionConfig,
) -> None: ...
```

Installation is idempotent for an equal config and rejects replacement with a
different config after installation. This mirrors other fail-closed scheduler
hook installation.

`_return_schedule()` becomes the single publication boundary:

1. preserve `last_policy_branch`;
2. increment `schedule_generation`;
3. parse the existing scheduled tuple;
4. build and store `last_speculative_selection`;
5. return the original tuple unchanged.

The deterministic builder cannot invoke user callbacks and cannot mutate
queues, sequences, block ownership, or scheduler policy.

## Engine Consumption Boundary

Future `LLMEngine.step()` integration must:

1. call `scheduler.schedule()`;
2. read `scheduler.last_speculative_selection`;
3. validate it against the returned sequence batch and generation;
4. execute ordinary model-runner behavior when no rows are selected;
5. otherwise pass only selected decode rows to the generic speculative batch
   runtime while preserving ordinary execution for suppressed rows;
6. append committed output metadata;
7. synchronize SAM only after metadata commit;
8. include the full selection record in step observations.

This design does not yet define how one mixed ordinary/speculative target
forward is fused. The first production integration may split selected and
suppressed decode rows into separate model-runner calls, but must measure that
overhead before promotion.

## Failure Semantics

- Configuration validation fails before scheduling begins.
- Record construction treats missing or invalid sequence fields as an engine
  invariant error.
- Publication failure must not alter the already constructed legacy schedule
  tuple or begin speculative resources. Because the builder is deterministic
  and side-effect free, such failure indicates corrupt sequence metadata and
  stops the step.
- Stale-record validation fails before adapter proposal or transaction begin.
- A disabled record is still published so observations cannot accidentally
  reuse a previous enabled selection.

## Testing

Create `tools/test_speculative_selection_record.py` with dependency-light
coverage for:

- configuration validation;
- ordinary decode selection and output-budget cap;
- ordinary prefill suppression;
- mixed decode/prefill row classification;
- non-sampling suppression;
- insufficient-budget suppression;
- stable row ordering and duplicate-ID rejection;
- immutable records;
- stale token-count, completion-count, generation, and sequence-order
  rejection;
- selected sequence return order.

Add a focused scheduler integration test proving:

- disabled-by-default publication;
- installation idempotence and replacement rejection;
- generation increments exactly once per `_return_schedule()`;
- existing schedule tuple identity/length/content are unchanged;
- `last_speculative_selection` corresponds to the same batch and
  `last_policy_branch`.

## Promotion Boundary

Passing this slice proves only that scheduler-selected decode rows have a
source-agnostic, immutable, stale-detectable speculative-attempt record.

It does not prove:

- adapter proposal execution from `LLMEngine`;
- real first-target or tail-verifier batching;
- scheduler-owned KV transaction lifecycle;
- ordinary/speculative mixed-row correctness;
- SAM post-commit synchronization;
- GPU, TP, long-context, exact-model parity, or performance improvement.

Overall classification remains `NOT_PROMOTABLE`.
