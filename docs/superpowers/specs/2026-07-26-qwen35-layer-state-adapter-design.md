# Qwen3.5 Layer-State Adapter Design

## Objective

Add a rank-local, generation-safe adapter between one Qwen3.5 linear-attention
layer and `HybridStateTensorPool`.

## Interface

Create `tinyvllm/engine/qwen35_layer_state.py`:

```python
class Qwen35LayerStateAdapter:
    def __init__(self, pool: HybridStateTensorPool, layer_index: int)

    def gather(
        self,
        lease: HybridStateLease,
    ) -> tuple[torch.Tensor, torch.Tensor]

    def commit(
        self,
        lease: HybridStateLease,
        convolution_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> None
```

`gather()` validates the lease and returns clones, never writable pool views.

`commit()`:

1. validates the lease generation/owner;
2. validates both candidate shape/dtype/device contracts before writing;
3. snapshots both current pool rows;
4. writes convolution then recurrent state;
5. restores both snapshots if either copy raises.

This gives transactional behavior at the Python adapter boundary.

## Failure Contracts

Reject invalid layer indices, missing layer components, stale/wrong-owner
leases, and candidate shape/dtype/device mismatches. Failed validation or
copy leaves both pool rows unchanged.

## Test Gate

Cover successful gather/commit, clone isolation, stale generation rejection,
candidate validation without mutation, and injected second-copy failure with
rollback of the first component.

## Claim Boundary

Passing proves only CPU rank-local pool gather/transactional commit. It does
not prove batched slot dispatch, ModelRunner wiring, CUDA atomicity, or native
model correctness. Schema-v2 canonical remains `NO_GO`.

