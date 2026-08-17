# Batch-Native ModelRunner Tail Verifier Design

**Date:** 2026-08-12
**Status:** Approved by the standing generic speculative-runtime direction

## Goal

Replace the `ModelRunner` verifier's exactly-one-sequence restriction with a
real homogeneous-query batch boundary:

- one flattened input tensor for all verifier rows;
- one target-model forward for the batch;
- one `flash_attn_with_kvcache()` invocation per attention layer;
- direct KV writes to each transaction's final physical slots;
- one ordered target-token result row per sequence.

This slice must not loop over the existing single-sequence verifier and call it
once per request.

## Scope

This slice implements the production-capable tail-forward primitive needed by
`execute_native_speculative_batch()`. It does not yet call that runtime from
`LLMEngine`.

The first-target callback continues to use the existing ordinary batched decode
forward. This slice adds the missing multi-token tail callback.

## Current Root Cause

The existing verifier is single-sequence at two independent boundaries:

1. `ModelRunner._validate_spec_verify_compatibility()` rejects
   `seq_count != 1`.
2. `_flash_attn_spec_verify()` requires exactly one context length and one
   block-table row, then reshapes flattened queries to `[1, Q, H, D]`.

The rest of the data path is already compatible with flattened multi-token
execution:

- `store_kvcache()` accepts a flattened slot mapping;
- model layers consume flattened token rows;
- `run_model(..., execution_mode="spec_verify")` already forces eager
  execution and preserves every logits row;
- each `SpecVerifyPlan` already contains exact input tokens, model positions,
  logical slots, context length, and visible block count.

## Alternatives

### A. Loop over `prepare_spec_verify()` and `run_model()`

Rejected. It would perform one target forward per sequence and would only
rename a single-sequence verifier. It cannot amortize model or collective
overhead.

### B. Support arbitrary ragged query lengths immediately

Deferred. `flash_attn_with_kvcache()` accepts a dense
`[batch, query_len, heads, dim]` query tensor and does not accept per-row query
lengths. Padding would create fake KV writes and fake logits. Grouping every
distinct query length inside each attention layer can degenerate into one
kernel per request and is not an acceptable first proof of batch-native
execution.

### C. Require one homogeneous query length per verifier batch

Selected. All rows use the same `Q > 0`, allowing:

```text
flattened q [B * Q, H, D]
  -> view [B, Q, H, D]
  -> one flash_attn_with_kvcache call
  -> flatten [B * Q, H, D]
```

Variable proposal lengths remain supported by the generic runtime contract,
but the future engine callback must bucket selected rows by verifier query
length before invoking this primitive.

## Public Contracts

Extend `tinyvllm/speculative/verifier.py`.

```python
@dataclass(frozen=True)
class SpecVerifyBatchRowMetadata:
    sequence_id: int
    batch_index: int
    query_offset: int
    query_len: int
    input_tokens: tuple[int, ...]
    positions: tuple[int, ...]
    logical_slots: tuple[int, ...]
    physical_slots: tuple[int, ...]
    context_len: int
    block_table: tuple[int, ...]
```

```python
@dataclass(frozen=True)
class SpecVerifyBatchMetadata:
    rows: tuple[SpecVerifyBatchRowMetadata, ...]
    query_len: int
    total_query_tokens: int
    block_table_width: int
```

```python
@dataclass(frozen=True)
class SpecVerifyBatchResultRow:
    sequence_id: int
    target_tokens: tuple[int, ...]
```

```python
def split_spec_verify_batch_target_tokens(
    metadata: SpecVerifyBatchMetadata,
    flat_target_tokens: tuple[int, ...],
) -> tuple[SpecVerifyBatchResultRow, ...]: ...
```

The splitter validates exact target-token count and preserves metadata order.

## ModelRunner Preparation

Add:

```python
def prepare_spec_verify_batch(
    self,
    items: tuple[object, ...],
) -> tuple[torch.Tensor, torch.Tensor, SpecVerifyBatchMetadata]: ...
```

Each item must expose:

```text
sequence_id: int
plan: SpecVerifyPlan
proxy_block_table: tuple[int, ...]
```

This matches the existing `TailBatchItem` without importing proposal-source
logic into `ModelRunner`.

Preparation must validate all rows before any CUDA upload:

- `items` is a non-empty tuple;
- sequence IDs are unique;
- every plan is `SpecVerifyPlan`;
- every `query_len` is greater than zero;
- every row has the same `query_len`;
- plan positions and logical slots have exact lengths;
- each proxy block table covers `visible_block_count`;
- all referenced block IDs are non-negative;
- physical slots are derived by `validate_spec_verify_slots()`;
- compatibility checks pass for the entire batch.

Rows are flattened in input order:

```text
input_ids
positions
slot_mapping
```

Block-table rows are padded with `-1` to one batch width after each row's
visible prefix has been validated.

The installed context contains:

```text
mode = "spec_verify"
context_lens = [context_len_0, ..., context_len_B-1]
block_tables = [B, max_visible_blocks]
slot_mapping = [B * Q]
spec_verify_query_lens = (Q, ..., Q)
flash_attn_num_splits = 16
```

The existing `prepare_spec_verify()` remains as a backward-compatible
single-row wrapper over the same preparation logic.

## Attention Execution

Extend `Context` with:

```python
spec_verify_query_lens: tuple[int, ...] = ()
```

`_flash_attn_spec_verify()` validates:

- context lengths and block-table rows have the same positive batch size;
- `spec_verify_query_lens` contains one positive integer per row;
- all query lengths are identical;
- `q.size(0) == batch_size * query_len`.

It then performs exactly one:

```python
flash_attn_with_kvcache(
    q.view(batch_size, query_len, num_heads, head_dim),
    ...,
    cache_seqlens=context.context_lens,
    block_table=context.block_tables,
    causal=True,
)
```

The helper returns a flattened tensor with the original `q` shape.

## ModelRunner Tail Method

Add:

```python
def run_spec_verify_batch(
    self,
    items: tuple[object, ...],
) -> tuple[SpecVerifyBatchResultRow, ...] | None: ...
```

Execution:

1. call `prepare_spec_verify_batch()` once;
2. call `run_model(..., execution_mode="spec_verify")` once;
3. on rank 0, compute greedy token IDs for every logits row;
4. split flat token IDs with
   `split_spec_verify_batch_target_tokens()`;
5. on worker ranks, return `None`;
6. reset attention context in `finally`.

This method does not begin, commit, or rollback KV transactions. The generic
batch runtime remains the owner of transaction lifetime.

## Failure Semantics

- Any invalid item, heterogeneous query length, unsupported feature, slot
  error, or block-table error fails before tensor upload and KV mutation.
- An exception during the forward propagates to the generic batch runtime,
  which rolls back all active transactions.
- No per-row result is returned until every target row has been produced.
- The method is greedy-only and eager-only.
- Quantized KV, KV offload, blockwise attention, Quest, Attention Matching,
  KV cartridge, and mixed prefill/decode remain fail-closed.

## Testing

Dependency-light tests must cover:

- batch metadata and target-token splitting;
- two-row preparation with one flattened upload per tensor;
- stable row order and exact physical slots;
- duplicate IDs;
- empty rows;
- zero query length;
- heterogeneous query lengths;
- insufficient/invalid block tables;
- compatibility allowing `seq_count > 1`;
- attention reshaping `[B * Q, H, D] -> [B, Q, H, D]`;
- exactly one FlashAttention call for `B > 1`;
- invalid query metadata rejection;
- source/runtime contract proving one `run_model()` call and no loop invoking
  `prepare_spec_verify()` or `run_model()` per item.

The focused regression must retain all existing speculative transaction,
runtime, scheduler selection, engine commit, and chunked-prefill tests.

## Promotion Boundary

Passing this slice proves a real fixed-Q batch-native tail target forward and
direct KV materialization. It does not prove:

- variable-Q execution;
- `LLMEngine` callback wiring;
- multi-token scheduler metadata commit;
- SAM synchronization;
- CUDA Graph execution;
- exact model parity on TP1/TP4;
- TPOT, throughput, memory, or KV-H2D improvement.

Overall classification remains `NOT_PROMOTABLE`.
