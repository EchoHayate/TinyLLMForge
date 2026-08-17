# Qwen3.5 Stateful Linear Decoder-Layer Design

## Objective

Add the smallest dependency-light transaction boundary that combines one
`Qwen35DecoderLayerShell`, its stateful `Qwen35LinearAttentionShell`, and one
`Qwen35LayerStateAdapter`.

The wrapper proves this order:

```text
gather
input layer norm
linear-attention forward
first residual
post-attention layer norm
MLP
second residual
commit
```

Persistent pool state changes only after the complete decoder layer succeeds.

## Chosen Boundary

Create `tinyvllm/layers/qwen35_stateful_decoder_layer.py`:

```python
class Qwen35StatefulLinearDecoderLayer(nn.Module):
    def __init__(
        self,
        decoder_layer: Qwen35DecoderLayerShell,
        state_adapter: Qwen35LayerStateAdapter,
    )

    def forward(
        self,
        lease: HybridStateLease,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor
```

The uniform decoder-layer call keeps `position_ids`, but the linear-attention
branch does not consume it. The wrapper must not mutate it.

The constructor accepts only a decoder shell configured with
`block_type == "linear_attention"`. It reuses the decoder shell's existing
normalization, mixer, MLP, and component-output validation boundaries rather
than adding another set of modules.

## Transaction Flow

1. Validate `hidden_states` using the same rank-two floating contract as the
   stateless decoder shell.
2. Call `state_adapter.gather(lease)` to obtain isolated convolution and
   recurrent clones.
3. Normalize the input.
4. Call the configured linear-attention module with
   `(normalized, convolution_state, recurrent_state)`.
5. Validate that the mixer returns exactly
   `(mixed, candidate_convolution_state, candidate_recurrent_state)`.
6. Complete both residual paths, post-attention normalization, and MLP.
7. Call `state_adapter.commit(...)` only after the final hidden output has
   passed all component contracts.
8. Return the final hidden output only after commit succeeds.

There is no attempt to roll back arbitrary module side effects. The guaranteed
transaction covers the persistent hybrid-state pool because gather returns
clones and commit is the only pool write.

## Failure Contracts

- A stale or unbound lease fails during gather before any layer component runs.
- Input norm, mixer, post-attention norm, or MLP failure leaves the pool
  unchanged.
- Malformed mixer tuple arity or hidden output fails closed before commit.
- Invalid candidate states fail in the adapter before pool writes.
- A second adapter copy failure rolls back both pool components through the
  existing adapter transaction.
- A full-attention decoder shell is rejected at construction.

## Test Gate

Create `tools/test_qwen35_stateful_linear_decoder_layer.py` with an actual
`HybridStateTensorPool`, `Qwen35LayerStateAdapter`, and
`Qwen35DecoderLayerShell`.

Cover:

- exact operation order and independent numerical oracle;
- successful dual-state commit;
- clone isolation and hidden/position input nonmutation;
- BF16 and non-contiguous hidden input;
- stale lease rejection before module execution;
- mixer failure after gather with unchanged pool;
- post-attention norm and MLP failure after candidate creation with unchanged
  pool;
- malformed mixer return and hidden-output contracts;
- invalid candidate state rejection;
- injected second-copy commit failure with rollback of both pool rows;
- rejection of a non-linear decoder shell.

## Claim Boundary

Passing proves only a CPU, single-request, rank-local decoder-layer transaction
from pool gather through successful dual-state commit.

It does not prove packed multi-request dispatch, batching, ModelRunner wiring,
checkpoint loading, CUDA behavior, kernel optimization, native model
correctness, compression ratio, quality retention, latency, throughput, or
memory improvement. The immutable Qwen3.5 schema-v2 canonical result remains
`NO_GO`.
