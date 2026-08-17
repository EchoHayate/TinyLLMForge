# Qwen3.5 Batched Layer-State Transaction Design

## Objective

Extend `Qwen35LayerStateAdapter` with dependency-light, slot-indexed batched
gather and all-or-nothing batched commit for one linear-attention layer.

This is the next correctness prerequisite between the existing single-request
stateful decoder wrapper and any future packed multi-request execution.

## Interface

Extend `tinyvllm/engine/qwen35_layer_state.py`:

```python
class Qwen35LayerStateAdapter:
    def gather_batch(
        self,
        leases: tuple[HybridStateLease, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]

    def commit_batch(
        self,
        leases: tuple[HybridStateLease, ...],
        convolution_states: torch.Tensor,
        recurrent_states: torch.Tensor,
    ) -> None
```

The existing single-request `gather()` and `commit()` behavior remains
unchanged.

## Gather Contract

`gather_batch()`:

1. requires a non-empty tuple of `HybridStateLease` values;
2. validates every lease before reading any component;
3. rejects duplicate slot IDs, even if the repeated lease is otherwise valid;
4. preserves lease order in the leading batch dimension;
5. returns cloned, contiguous tensors shaped:

```text
[batch, *linear_convolution_shape]
[batch, *linear_recurrent_shape]
```

Mutating either result cannot change the pool.

## Commit Contract

`commit_batch()`:

1. requires a non-empty tuple of leases;
2. validates all lease bindings and rejects duplicate slot IDs before any
   write;
3. validates both complete batched candidate tensors before any write;
4. requires exact leading batch size, component shape, dtype, and device;
5. snapshots all selected convolution and recurrent rows;
6. copies rows in lease order using the existing `_copy_component` seam;
7. restores every selected row if any convolution or recurrent copy raises.

Rows outside the selected batch are never modified.

The method deliberately does not implement partial success. Callers either
commit the full layer-state batch or observe the original pool.

## Rejected Alternatives

### Repeated single-request `commit()`

This can commit request zero before request one fails, violating batch
atomicity.

### One advanced-index assignment per component

This is compact but does not expose a deterministic failure point between
individual row/component writes, making the rollback guarantee difficult to
test. It also obscures lease-order semantics.

### ModelRunner integration in the same change

This would combine state transaction correctness with packed-token boundaries,
model construction, and runtime dispatch. Those remain separate later gates.

## Failure Contracts

- Empty, list-based, or non-lease batches fail closed.
- A stale lease anywhere in the batch leaves all rows unchanged.
- Duplicate slots fail before gather or commit.
- Batch-size, component-shape, dtype, or device mismatch fails before writes.
- A failure on any later row or second component restores all earlier writes.
- Non-selected rows remain byte-for-byte unchanged.

## Test Gate

Extend `tools/test_qwen35_layer_state_adapter.py` with a three-slot fixture and
cover:

- out-of-order gather preserving lease order;
- clone isolation and contiguous stacked output;
- successful two-row commit with an untouched third row;
- stale later lease causing no partial write;
- duplicate slot rejection;
- empty/list/non-lease input rejection;
- candidate batch-size, shape, dtype, and device rejection;
- injected failure after multiple successful copies with rollback of every
  selected row and preservation of the unselected row;
- unchanged single-request adapter regression.

## Claim Boundary

Passing proves only CPU rank-local batched gather and rollback-protected commit
for one Qwen3.5 linear-attention layer.

It does not prove packed token-to-request boundaries, batched decoder
execution, cross-layer transactionality, ModelRunner wiring, checkpoint
loading, CUDA atomicity, native model correctness, compression ratio, quality,
latency, throughput, or memory improvement. The immutable Qwen3.5 schema-v2
canonical result remains `NO_GO`.
