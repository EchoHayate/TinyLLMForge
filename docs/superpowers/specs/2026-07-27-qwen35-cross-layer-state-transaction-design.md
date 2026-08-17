# Qwen3.5 Cross-Layer State Transaction Design

## Objective

Add an all-or-nothing transaction across every linear-attention layer touched
by one packed model forward.

Per-layer `commit_batch()` is atomic only within one layer. A later model layer
failure must not leave earlier linear layers committed.

## Interface

Create `tinyvllm/engine/qwen35_state_transaction.py`:

```python
class Qwen35CrossLayerStateTransaction:
    def __init__(
        self,
        adapters: tuple[Qwen35LayerStateAdapter, ...],
    )

    def gather(
        self,
        leases: tuple[HybridStateLease, ...],
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        ...,
    ]

    def commit(
        self,
        leases: tuple[HybridStateLease, ...],
        candidates: tuple[
            tuple[torch.Tensor, torch.Tensor],
            ...,
        ],
    ) -> None
```

The adapter order is the transaction layer order. `gather()` returns one
batched `(convolution, recurrent)` pair per adapter in that order.

## Construction Contract

- Adapters are a non-empty tuple.
- Every entry is a `Qwen35LayerStateAdapter`.
- All adapters reference the same `HybridStateTensorPool`.
- Layer indices are unique.

## Commit Contract

1. Validate the lease tuple through every adapter before writes.
2. Require exactly one candidate pair per adapter.
3. Validate every convolution and recurrent batch before writes.
4. Snapshot every selected row in every layer.
5. Copy layer/request/component in deterministic order.
6. If any copy raises, directly restore every snapshotted row in every layer.

Rollback bypasses the injectable `_copy_component` seam so the same injected
failure cannot prevent restoration.

## Failure Contracts

- Invalid construction fails before transaction use.
- Stale lease in any adapter fails before writes.
- Candidate tuple count, pair arity, tensor, shape, dtype, or device mismatch
  fails before writes.
- Failure in a later layer restores earlier-layer and current-layer writes.
- Rows not selected by leases remain unchanged across all layers.

## Test Gate

Use two linear layers and three slots. Cover:

- ordered gather for both layers and out-of-order leases;
- clone isolation;
- successful two-layer commit with untouched third slot;
- stale lease and malformed candidates with no writes;
- injected failure in layer 1 after layer 0 has fully copied;
- complete rollback across both layers and preservation of unselected rows;
- invalid adapter tuple, duplicate layers, and mixed pools.

## Claim Boundary

Passing proves only CPU rank-local cross-layer transactionality for fixed
linear state. It does not prove heterogeneous model execution, full-attention
KV transactionality, scheduler/ModelRunner wiring, checkpoint equivalence,
CUDA atomicity, native support, or performance/memory/quality gains.
Schema-v2 remains `NO_GO`.
