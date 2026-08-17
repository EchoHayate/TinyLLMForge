# Speculative ModelRunner Callback Bridge Design

**Date:** 2026-08-12

## Goal

Provide engine-side callbacks that adapt `ModelRunner.call()` to
`execute_native_speculative_batch()`:

- one first-target RPC for the selected sequence batch;
- stable fixed-query-length tail grouping;
- one tail RPC per distinct query length;
- ordered merge into generic `TailBatchResult` rows.

This slice builds the bridge but does not yet invoke it from `LLMEngine.step()`.

## Current Gap

The generic batch runtime accepts:

```python
run_first_targets(seqs)
run_tail_batch(items)
```

`ModelRunner` now has matching production-shaped methods, but:

1. no engine adapter calls them through the all-rank `ModelRunner.call()`
   transport;
2. `run_spec_first_target_batch()` exposes keyword-only capability flags,
   while `ModelRunner.call()` and its shared-memory command envelope carry
   positional arguments only;
3. one generic tail callback may contain variable query lengths, while the
   real verifier intentionally accepts only homogeneous fixed-Q batches.

## Considered Approaches

### 1. Teach `ModelRunner.call()` to carry arbitrary keyword arguments

This changes the shared-memory command envelope and every command-dispatch
consumer for one callback.

**Rejected for this slice.**

### 2. Pad variable-Q rows into one verifier forward

Padding would create false KV writes and logits because
`flash_attn_with_kvcache()` does not accept per-row query lengths.

**Rejected for correctness.**

### 3. Positional first-target flags plus stable fixed-Q grouping

Allow the two booleans as positional arguments, call first-target once, group
tail items by exact `plan.query_len`, call ModelRunner once per distinct Q, and
merge results back to original item order.

**Selected.**

## New Engine Module

Create:

```text
tinyvllm/engine/speculative_model_runner.py
```

It contains dependency-light validation and no model-name or proposal-source
branches.

## Contracts

```python
@dataclass(frozen=True)
class FixedQTailBatch:
    query_len: int
    items: tuple[TailBatchItem, ...]


def build_fixed_q_tail_batches(
    items: tuple[TailBatchItem, ...],
) -> tuple[FixedQTailBatch, ...]:
    ...


def run_model_runner_first_targets(
    model_runner,
    seqs: tuple[object, ...],
    capabilities: DraftCapabilities,
) -> tuple[FirstTargetResult, ...]:
    ...


def run_model_runner_tail_batch(
    model_runner,
    items: tuple[TailBatchItem, ...],
) -> tuple[TailBatchResult, ...]:
    ...
```

## First-Target Data Flow

1. Validate `DraftCapabilities`.
2. Call exactly once:

   ```python
   model_runner.call(
       "run_spec_first_target_batch",
       seqs,
       capabilities.requires_target_hidden,
       capabilities.requires_target_logits,
   )
   ```

3. Require a tuple of `FirstTargetResult`.
4. Require exact sequence-ID coverage and uniqueness.
5. Return rows in the input sequence order, regardless of ModelRunner result
   order.

`ModelRunner.run_spec_first_target_batch()` removes the keyword-only separator
for the two booleans so the RPC can pass them positionally. Existing keyword
callers remain valid.

## Tail Grouping

`build_fixed_q_tail_batches()`:

- requires a non-empty tuple of `TailBatchItem`;
- validates positive query lengths and unique sequence IDs;
- groups by exact query length;
- orders groups by first occurrence of each query length;
- preserves item order within each group.

Example:

```text
input:  seq8/Q2, seq4/Q1, seq2/Q2, seq9/Q3
groups: Q2=(seq8,seq2), Q1=(seq4), Q3=(seq9)
```

## Tail RPC and Merge

For every fixed-Q group, call exactly once:

```python
model_runner.call(
    "run_spec_verify_batch",
    group.items,
)
```

The returned rows must be `SpecVerifyBatchResultRow` with exact group
sequence-ID coverage. Convert each row to:

```python
TailBatchResult(
    sequence_id=row.sequence_id,
    target_tokens=row.target_tokens,
    metadata={
        "query_len": group.query_len,
        "fixed_q_group_index": group_index,
        "fixed_q_group_count": group_count,
    },
)
```

Finally return `TailBatchResult` rows in the original `items` order.

If any group fails or returns invalid rows, propagate the failure. The generic
batch runtime owns rollback of all active transactions, including KV written
by earlier successful groups.

## Observability Boundary

The generic runtime's existing `tail_callback_count` remains one because it
counts callback invocations. Physical target-forward count equals the number
of fixed-Q groups and is exposed through each tail row's metadata.

Dedicated aggregate observations can be added during `LLMEngine` wiring.

## Tests

Dependency-light tests cover:

- first-target uses one RPC and capability flags;
- reversed first-target rows are restored to sequence order;
- fixed-Q grouping stability;
- batch-4 variable Q produces one RPC per distinct Q, never per sequence;
- reversed per-group results merge to original item order;
- duplicate IDs, zero Q, missing rows, extra rows, wrong result types, and
  worker-style `None` fail closed;
- source test proves ModelRunner first-target flags can be positional;
- generic source scan remains model/proposal-source free.

## Non-Goals

- invoking the bridge from `LLMEngine.step()`;
- executing suppressed rows alongside selected rows;
- changing the generic runtime callback-count schema;
- variable-Q FlashAttention;
- scheduler metadata commit;
- performance claims.

## Result Boundary

```text
real ModelRunner callback adapter:
  implemented
variable proposal lengths:
  supported through one forward per distinct fixed Q
LLMEngine runtime execution:
  not implemented
end-to-end performance:
  unmeasured
overall classification:
  NOT_PROMOTABLE
```
