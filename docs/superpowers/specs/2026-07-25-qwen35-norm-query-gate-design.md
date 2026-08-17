# Qwen3.5 Offset RMSNorm and Query-Gate Design

## Status

Approved for inline CPU-only execution. This slice does not add attention,
RoPE, a model, checkpoint loading, or GPU execution.

## Objective

Add two official-formula correctness primitives:

1. Qwen3.5 offset RMSNorm:

```text
normalized = x_fp32 * rsqrt(mean(x_fp32^2) + eps)
output = normalized * (1 + weight_fp32)
```

2. full-attention query-output gating:

```text
output = attention_output * sigmoid(query_gate)
```

The gate is applied after attention heads are flattened and before `o_proj`.

## Isolation Decision

Do not change `tinyvllm.layers.layernorm.RMSNorm`. Its weight is initialized
to one and directly scales normalized values, which is correct for existing
Qwen3. Changing it to offset semantics would silently break current models.

Create `tinyvllm/layers/qwen35_primitives.py` with:

```python
class Qwen35OffsetRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6): ...
    def forward(self, tensor: torch.Tensor) -> torch.Tensor: ...

def qwen35_apply_query_gate(
    attention_output: torch.Tensor,
    query_gate: torch.Tensor,
) -> torch.Tensor: ...
```

## Contracts

`Qwen35OffsetRMSNorm`:

- positive non-boolean integer hidden size;
- positive finite epsilon;
- weight shape `[hidden_size]`, initialized to zeros;
- input last dimension exactly `hidden_size`;
- floating-point input only;
- FP32 normalization and scale multiplication;
- output preserves input dtype and shape;
- input is not mutated.

`qwen35_apply_query_gate`:

- both tensors floating point;
- exact shape equality, with no broadcasting;
- deterministic sigmoid multiplication;
- output dtype and shape match attention output;
- inputs are not mutated.

Exact shape equality is deliberate. The official model reshapes the query
gate to the flattened attention-output shape before multiplication. Silent
broadcasting could gate the wrong head or token.

## Test Gate

CPU tests must cover:

- zero weight equals plain FP32 RMS normalization;
- non-zero positive and negative offsets;
- distinction from legacy direct-weight RMSNorm;
- BF16 input/output with FP32 oracle;
- arbitrary leading dimensions;
- no input mutation;
- gate values at zero, large positive, and large negative logits;
- exact manual sigmoid oracle;
- shape, dtype, hidden-size, and epsilon failures.

## Claim Boundary

Passing proves only the isolated formulas. It does not prove Qwen3.5
attention, RoPE, checkpoint loading, full-layer equivalence, or performance.

