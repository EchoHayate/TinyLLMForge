# Qwen3.5 Packed Full-Attention Decoder Design

## Objective

Add the missing dependency-light request-isolation boundary for applying one
Qwen3.5 full-attention decoder layer to packed multi-request hidden states.

Without this wrapper, a full-attention backend receiving all packed tokens
could mix tokens across requests.

## Interface

Create `tinyvllm/layers/qwen35_packed_full_decoder_layer.py`:

```python
class Qwen35PackedFullDecoderLayer(nn.Module):
    def __init__(self, decoder_layer: Qwen35DecoderLayerShell)

    def forward(
        self,
        token_counts: tuple[int, ...],
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor
```

The decoder shell must use `block_type == "full_attention"`.

## Request Isolation

`token_counts` defines contiguous packed request segments. The wrapper slices
both hidden states and position IDs per request, invokes the full decoder shell
separately for each segment, and concatenates outputs in the original order.

For one-dimensional positions:

```text
position_ids[offset:offset + token_count]
```

For one-row or three-row positions:

```text
position_ids[:, offset:offset + token_count]
```

The full-attention backend never receives tokens from two requests in one call.

## Validation and Failure Contracts

- `token_counts` is a non-empty tuple of positive non-boolean integers.
- Count sum equals packed hidden and position token width.
- Hidden states are rank-two floating tensors.
- Positions are integer rank-one or rank-two tensors with one or three rows.
- Position and hidden devices match.
- A failure in any request raises without returning a partial packed output.
- Inputs are never mutated.

There is no persistent full-attention KV transaction in this CPU shell. Future
runtime KV cache integration remains separate.

## Test Gate

Use `(2, 1, 3)` packed segments and an attention fixture whose result depends
on the segment mean. Prove:

- exact independent per-request oracle;
- observable attention calls of token lengths `2, 1, 3`;
- no cross-request leakage when unrelated request tokens change;
- one-dimensional, one-row, and three-row position slicing;
- BF16 and non-contiguous hidden input;
- metadata and constructor failures;
- later-request failure without partial return or input mutation.

## Claim Boundary

Passing proves only CPU packed request isolation for one full-attention decoder
layer. It does not prove paged KV behavior, cache growth, scheduler metadata,
heterogeneous model execution, CUDA attention, native support, or any
performance/memory/quality gain. Schema-v2 remains `NO_GO`.
