# Qwen3.5 Decoder-Layer Shell Design

## Status

Approved under the standing inline-execution direction for the Qwen3.5
native correctness path. This is a CPU-only dependency-light module gate.

## Objective

Prove the official Qwen3.5 decoder-layer data flow for both configured block
types without adding a production model, cache mutation, or GPU runtime:

```text
residual = hidden_states
normalized = input_layernorm(hidden_states)
mixed = selected token mixer(normalized)
hidden_states = residual + mixed
residual = hidden_states
normalized = post_attention_layernorm(hidden_states)
mlp_output = mlp(normalized)
hidden_states = residual + mlp_output
```

The selected token mixer is determined by `block_type`:

```text
full_attention   -> full_attention(position_ids, normalized)
linear_attention -> linear_attention(normalized)
```

## Official Source

The source audited for this gate is:

```text
/tmp/modeling_qwen3_5.py
SHA-256 15d5425ee6e771f8fbca10468c280fe62afa79fab3eff73ad1a8852162799d48
```

`Qwen3_5DecoderLayer.forward` performs:

1. save the incoming hidden states as the first residual;
2. apply `input_layernorm`;
3. dispatch to the layer-type-specific token mixer;
4. add the first residual;
5. save that result as the second residual;
6. apply `post_attention_layernorm`;
7. apply the MLP;
8. add the second residual.

The official full-attention and linear-attention modules expose different
runtime arguments. This dependency-light shell deliberately proves only the
layer ordering and branch selection:

- full attention receives `position_ids` and normalized hidden states;
- linear attention receives normalized hidden states;
- recurrent/cache state is outside this gate.

## Alternatives Considered

### A. Injected Decoder-Layer Shell

Create a small shell with injected norms, token mixers, and MLP.

Advantages:

- exact operation order is observable on CPU;
- both `layer_types` branches can be tested without optional kernels;
- residual and boundary contracts are isolated from runtime state;
- no changes to existing Qwen3 code or fused residual norm behavior.

This is the selected approach.

### B. Reuse Existing Fused `RMSNorm(x, residual)`

This could more closely resemble TinyLLMForge's optimized Qwen3 residual
pipeline, but Qwen3.5 uses offset RMSNorm parameters and the existing fused
module has direct-weight semantics. Reusing it now would either be
mathematically wrong or require premature fused-kernel changes.

Rejected for this gate.

### C. Build the Native Qwen3.5 Decoder and Model Directly

This would combine token-mixer state, checkpoint loading, model selection,
and layer math in one step. Failures would not identify which contract is
wrong.

Rejected as too broad.

## Component

Create:

```text
tinyvllm/layers/qwen35_decoder_layer.py
```

with:

```python
class Qwen35DecoderLayerShell(nn.Module):
    def __init__(
        self,
        *,
        block_type: str,
        input_layernorm: nn.Module,
        post_attention_layernorm: nn.Module,
        mlp: nn.Module,
        full_attention: Optional[nn.Module] = None,
        linear_attention: Optional[nn.Module] = None,
    )

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor
```

Constructor rules:

- `block_type` must be exactly `"full_attention"` or
  `"linear_attention"`;
- the selected module must be provided;
- the unselected module may be omitted and must never be called.

The shell does not own model dimensions. It derives the exact output contract
from `hidden_states`.

## Data Flow

For rank-two token-major hidden states:

```text
hidden_states: [tokens, hidden_size]
```

Full-attention branch:

```text
first_residual = hidden_states
normalized = input_layernorm(hidden_states)
mixed = full_attention(position_ids, normalized)
after_mixer = first_residual + mixed
second_residual = after_mixer
normalized = post_attention_layernorm(after_mixer)
mlp_output = mlp(normalized)
output = second_residual + mlp_output
```

Linear-attention branch:

```text
first_residual = hidden_states
normalized = input_layernorm(hidden_states)
mixed = linear_attention(normalized)
after_mixer = first_residual + mixed
second_residual = after_mixer
normalized = post_attention_layernorm(after_mixer)
mlp_output = mlp(normalized)
output = second_residual + mlp_output
```

Residual addition follows PyTorch's native input dtype, matching the official
plain tensor additions. The shell must not mutate `hidden_states` or
`position_ids`.

## Fail-Closed Contracts

The shell rejects:

- unsupported block types;
- missing selected token-mixer modules;
- non-tensor, non-rank-two, or non-floating hidden states;
- input norm, selected token mixer, post-attention norm, or MLP outputs that
  are not tensors;
- any component output changing shape, dtype, or device;
- component outputs that are not floating point.

Checks occur immediately after each component.

Position-id semantic validation remains the selected full-attention module's
responsibility. The decoder shell only forwards the object unchanged and
does not inspect it for the linear-attention branch.

## Test Gate

CPU tests must cover:

- exact observable full-attention event order;
- exact observable linear-attention event order;
- independent numerical oracles for both branches;
- only the selected branch being called;
- first residual addition before post-attention norm;
- second residual addition after MLP;
- full attention receiving the original position ids;
- linear attention not receiving position ids;
- FP32 and BF16 behavior;
- unchanged hidden states and position ids;
- non-contiguous valid hidden/component tensors;
- constructor failures;
- input rank/dtype failures;
- shape/dtype/device/non-floating failures after every component boundary.

## Claim Boundary

Passing proves:

- exact dependency-light Qwen3.5 decoder residual and normalization order;
- correct dispatch between full-attention and linear-attention layer types;
- strict CPU boundary validation for the layer shell.

It does not prove:

- a production Qwen3.5 linear-attention module;
- recurrent or convolution state mutation;
- paged KV-cache behavior;
- a native Qwen3.5 MLP projection implementation;
- checkpoint-name mapping or real checkpoint loading;
- decoder-layer equivalence against official weights;
- model construction, logits, distributed execution, or GPU correctness;
- compression, quality, latency, throughput, or memory improvement.

The immutable Qwen3.5 schema-v2 canonical result remains `NO_GO`.

