# Qwen3.5 Packed Heterogeneous Layer Stack Design

## Objective

Combine packed request-isolated full-attention layers and packed stateful
linear-attention layers in one explicit Qwen3.5 layer schedule, while committing
all linear-layer state only after the complete stack succeeds.

## Interface

Create `tinyvllm/layers/qwen35_packed_layer_stack.py`:

```python
class Qwen35PackedHeterogeneousLayerStack(nn.Module):
    def __init__(
        self,
        layers: tuple[Qwen35DecoderLayerShell, ...],
        state_transaction: Qwen35CrossLayerStateTransaction,
    )

    def forward(
        self,
        leases: tuple[HybridStateLease, ...],
        token_counts: tuple[int, ...],
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor
```

Tuple position is the model layer index. The transaction adapter layer indices
must exactly equal the indices of all `linear_attention` layers.

## Execution

1. Validate packed metadata and layer/adapter alignment.
2. Gather every linear layer state through the cross-layer transaction.
3. Traverse layers in tuple order.
4. For full-attention layers, slice positions and hidden states per request and
   execute each request separately.
5. For linear-attention layers, slice hidden states per request, pair each
   segment with the matching gathered state row, and retain candidate rows.
6. After every layer succeeds, call one cross-layer commit with all linear
   candidates.
7. Return final packed hidden states only after commit succeeds.

## Failure Semantics

- Full-attention requests cannot see tokens from another request.
- Linear state rows cannot cross requests or layers.
- Any full layer, linear layer, MLP, candidate validation, or later layer
  failure leaves all linear pool state unchanged.
- Cross-layer commit failure restores every linear layer.
- Inputs are not mutated.

## Test Gate

Use schedule:

```text
layer 0: linear_attention
layer 1: full_attention
layer 2: linear_attention
```

with packed request counts `(2, 1, 3)`. Cover exact independent numerical and
state oracles, observable layer/request call order, full-attention leakage
protection, BF16/non-contiguous input, constructor alignment failures,
later-full/later-linear failure without state commit, and cross-layer commit
rollback.

## Claim Boundary

Passing proves only a dependency-light CPU heterogeneous decoder-layer stack
with packed request isolation and atomic fixed-state commit. It does not prove
embedding/final norm/lm-head behavior, paged full-attention KV state,
checkpoint loading, scheduler/ModelRunner wiring, logits, CUDA correctness,
native support, or performance/memory/quality gains. Schema-v2 remains
`NO_GO`.
