# Qwen3.5 Packed Stateful Linear Decoder Design

## Objective

Add a dependency-light packed multi-request wrapper for one Qwen3.5
linear-attention decoder layer. Explicit token counts map each packed token
segment to exactly one lease and one persistent state row.

## Interface

Create `tinyvllm/layers/qwen35_packed_stateful_decoder_layer.py`:

```python
class Qwen35PackedStatefulLinearDecoderLayer(nn.Module):
    def __init__(
        self,
        decoder_layer: Qwen35DecoderLayerShell,
        state_adapter: Qwen35LayerStateAdapter,
    )

    def forward(
        self,
        leases: tuple[HybridStateLease, ...],
        token_counts: tuple[int, ...],
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor
```

`token_counts[i]` is the contiguous token count belonging to `leases[i]`.
Every count is positive and their sum equals `hidden_states.shape[0]`.
`position_ids` remains a uniform decoder input and must have one or three rows
with its final dimension equal to the packed token count; linear attention does
not consume it.

## Transaction Flow

1. Validate hidden, position, lease, and token-count boundaries.
2. Call `state_adapter.gather_batch(leases)` once.
3. Slice packed hidden states in request order.
4. For each request, run input norm, its own stateful linear mixer, both
   residual paths, post norm, and MLP.
5. Accumulate output segments and candidate state rows without pool writes.
6. Stack all candidate rows and call `commit_batch()` once.
7. Concatenate and return packed outputs only after commit succeeds.

Any request failure prevents the single batch commit. A commit failure is
rolled back by the adapter.

## Failure Contracts

- Lease and token-count batch sizes must match and be non-empty tuples.
- Counts reject booleans, zero, negatives, and non-integers.
- Packed token sum must match hidden and position token dimensions.
- Every mixer output must be a three-item tuple with valid hidden output.
- Candidate state rows are validated by `commit_batch`.
- A failure in any later request leaves every pool row unchanged.
- Input hidden and position tensors are not mutated.

## Test Gate

Use three requests with token counts `(2, 1, 3)` and distinct initial states.
Prove packed output/state equality against independent per-request execution,
request-order state isolation, BF16/non-contiguous hidden support, malformed
metadata rejection, later-request failure without commit, invalid candidate
batch rejection, and adapter commit rollback propagation.

## Claim Boundary

Passing proves only CPU sequential execution of packed request segments through
one shared linear decoder layer with all-batch state commit. It does not prove
kernel-level batching, scheduler metadata wiring, cross-layer model execution,
ModelRunner integration, checkpoint loading, CUDA correctness, native support,
or any performance/memory/quality gain. Schema-v2 remains `NO_GO`.
