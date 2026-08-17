# Engine Speculative Selection and Commit Bridge Design

**Date:** 2026-08-12
**Status:** Approved by the existing generic speculative-runtime direction
**Scope:** Dependency-light engine selection consumption and output-token
commit planning; no real ModelRunner tail batch in this slice

## Goal

Bridge the scheduler-visible speculative selection record to the generic
speculative batch runtime without inventing a fake production verifier.

This slice must define:

1. exact selected/suppressed sequence partitioning from the scheduler record;
2. stale-record rejection before any model or KV operation;
3. conversion of `NativeSpeculativeBatchResult` into the final token sequence
   that the engine must append for each request;
4. default-off `LLMEngine.step()` consumption of the selection record;
5. fail-closed behavior if speculative rows are selected before a real engine
   runtime is installed.

## Why This Slice Is Necessary

`NativeSpeculativeBatchResult.accepted_tokens` is not always the complete
output to append.

For a K-token draft:

- zero acceptance must append the first target token;
- partial acceptance must append the accepted draft prefix followed by the
  target mismatch token;
- full acceptance appends the accepted draft tokens;
- an accepted EOS or exhausted output budget suppresses the mismatch token;
- an empty proposal still emits the ordinary first target token.

The generic batch runtime intentionally owns KV transaction and acceptance
mechanics, not `Sequence` metadata mutation. The engine therefore needs one
explicit commit-planning boundary before scheduler postprocessing can be made
multi-token aware.

## Non-Goals

- Removing the current `ModelRunner` single-sequence `spec_verify` restriction.
- Implementing a variable-length multi-sequence tail forward.
- Calling `execute_native_speculative_batch()` from production in this slice.
- Mutating `Sequence.token_ids` from the dependency-light bridge.
- Deallocating finished sequences or changing scheduler queues.
- Synchronizing SAM state.
- Supporting mixed ordinary/speculative execution in one target forward.
- Claiming any performance improvement.

## Alternatives

### A. Append only `accepted_tokens`

Rejected because zero and partial rejection would drop the target fallback
token and diverge from exact greedy decoding.

### B. Let `batch_runtime.py` mutate `Sequence`

Rejected because the generic runtime would become coupled to engine metadata,
finish policy, scheduler queues, and SAM lifecycle.

### C. Add a dependency-light engine partition and commit planner

Selected. The bridge validates immutable runtime results and produces exact
commit rows without mutating engine state. `LLMEngine` can then apply those
rows only after real ModelRunner callbacks are connected.

## Components

Create `tinyvllm/engine/speculative_execution.py`.

### Selection partition

```python
@dataclass(frozen=True)
class EngineSpeculativePartition:
    schedule_generation: int
    scheduled_sequence_ids: tuple[int, ...]
    selected_sequence_ids: tuple[int, ...]
    suppressed_sequence_ids: tuple[int, ...]
    selected_sequences: tuple[object, ...]
    suppressed_sequences: tuple[object, ...]
```

```python
def build_engine_speculative_partition(
    record: SpeculativeSelectionRecord,
    seqs: tuple[object, ...],
    *,
    expected_schedule_generation: int,
) -> EngineSpeculativePartition: ...
```

The function delegates stale checks to
`validate_speculative_selection_record()`, then preserves original batch order
within selected and suppressed subsets.

It validates exact disjoint coverage:

```text
selected IDs union suppressed IDs == scheduled IDs
selected IDs intersect suppressed IDs == empty
```

### Commit row

```python
@dataclass(frozen=True)
class EngineSpeculativeCommitRow:
    sequence_id: int
    output_tokens: tuple[int, ...]
    accepted_draft_tokens: tuple[int, ...]
    fallback_target_token: int | None
    finished_by_eos: bool
    finished_by_output_budget: bool
```

```python
def build_engine_speculative_commit_rows(
    result: NativeSpeculativeBatchResult,
    seqs: tuple[object, ...],
    *,
    eos_token: int,
) -> tuple[EngineSpeculativeCommitRow, ...]: ...
```

The result and sequence IDs must match exactly in order.

For each row:

1. compute the pre-commit remaining output budget;
2. validate `accepted_tokens` is an exact prefix of proposal tokens;
3. start output with `accepted_tokens`;
4. if the proposal is empty, use `first_target_token` as fallback;
5. if greedy acceptance is partial, use
   `target_tokens[greedy_accepted_count]` as fallback;
6. suppress fallback after accepted EOS when EOS is active;
7. append fallback only when output budget remains;
8. never append a bonus token after full acceptance because the current target
   result contains no K+1 target row;
9. classify EOS and output-budget completion from the final output tuple.

The fallback target token is not claimed as transaction-committed KV. Like an
ordinary decoded token, its KV is materialized when it becomes the next decode
input.

### Empty proposal

An adapter may return an empty proposal for a selected sequence. The generic
runtime creates no transaction and returns the first target. The commit row is:

```text
accepted_draft_tokens = ()
fallback_target_token = first_target_token
output_tokens = (first_target_token,)
```

subject to EOS and output budget.

### Zero acceptance

For proposal `(d0, d1, ...)` and first target `t0 != d0`:

```text
accepted_draft_tokens = ()
fallback_target_token = t0
output_tokens = (t0,)
```

### Partial acceptance

For proposal `(d0, d1, d2)` and target `(d0, x1, ...)`:

```text
accepted_draft_tokens = (d0,)
fallback_target_token = x1
output_tokens = (d0, x1)
```

Only `d0` belongs to the accepted speculative KV prefix.

### Full acceptance

For proposal and target both `(d0, d1, d2)`:

```text
accepted_draft_tokens = (d0, d1, d2)
fallback_target_token = None
output_tokens = (d0, d1, d2)
```

## LLMEngine Default-Off Consumption

Modify `LLMEngine.step()` immediately after scheduler tuple parsing:

```python
partition = build_engine_speculative_partition(
    self.scheduler.last_speculative_selection,
    tuple(seqs),
    expected_schedule_generation=(
        self.scheduler.schedule_generation
    ),
)
```

Default scheduler configuration produces no selected rows, so ordinary model
execution remains unchanged.

Until a real engine speculative runtime is installed:

```python
if partition.selected_sequences:
    raise RuntimeError(
        "speculative rows selected before engine runtime installation"
    )
```

The failure occurs before `ModelRunner.run()`, proposal generation, or KV
transaction begin. This prevents a configured speculative scheduler from
silently falling back to ordinary decode while observations claim selection.

Add selection IDs and generation to `last_step_observation`.

## Failure Semantics

- Stale selection fails before model execution.
- Invalid runtime-result IDs fail before commit-row construction.
- Invalid accepted-prefix or target-token shape fails before metadata mutation.
- Output-budget and EOS decisions are deterministic and per sequence.
- This slice performs no partial metadata commit because it does not mutate
  sequences.
- Once mutation is added, a later slice must prevalidate all rows before any
  append and define fatal handling for post-KV metadata failure.

## Testing

Create `tools/test_engine_speculative_execution.py` with dependency-light
coverage for:

- selected/suppressed stable partition;
- stale generation and token snapshots through the existing validator;
- exact disjoint coverage;
- empty proposal fallback;
- zero, one, partial, and full acceptance;
- EOS in accepted prefix;
- EOS fallback token;
- output-budget suppression of fallback;
- output-budget completion;
- result ID reorder/missing/duplicate rejection;
- accepted tokens that are not a proposal prefix;
- invalid target-token count.

Create `tools/test_llm_engine_speculative_selection_source.py` to verify the
heavy `LLMEngine` source:

- imports `build_engine_speculative_partition`;
- consumes `last_speculative_selection` immediately after `schedule()`;
- checks selected rows before `model_runner.call("run", ...)`;
- includes selection generation and IDs in step observation.

The source test is not a substitute for future behavioral engine tests. It
only protects the default-off wiring in an environment without the full
Torch/Transformers runtime.

## Promotion Boundary

Passing this slice proves exact engine-side output-token semantics and
default-off selection consumption. It does not prove:

- real speculative execution from `LLMEngine`;
- batch-native `ModelRunner` verification;
- multi-token scheduler postprocessing;
- SAM post-commit synchronization;
- exact model parity or performance improvement.

Overall classification remains `NOT_PROMOTABLE`.
