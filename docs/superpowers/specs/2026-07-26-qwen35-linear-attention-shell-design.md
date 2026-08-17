# Qwen3.5 Linear-Attention Shell Design

## Status

Approved under the standing inline-execution direction. This is a CPU-only
dependency-light correctness gate.

## Objective

Connect the proven Qwen3.5 causal-convolution and gated-delta recurrent
primitives into the official linear-attention data flow:

```text
in_proj_qkv
in_proj_z
in_proj_b
in_proj_a
causal depthwise convolution
Q/K/V split and reshape
Q/K head repeat when value_heads > key_heads
gated-delta recurrence
RMSNorm before SiLU(z) gate
out_proj
candidate convolution and recurrent states
```

The shell must not mutate persistent input state. It returns candidate new
states only after the complete forward, including output projection, succeeds.
The future pool integration is responsible for committing those candidates.

## Official Source

The source audited for this gate is:

```text
/tmp/modeling_qwen3_5.py
SHA-256 15d5425ee6e771f8fbca10468c280fe62afa79fab3eff73ad1a8852162799d48
```

The official gated norm is:

```text
normalized = core / sqrt(mean(core^2) + eps)
scaled = norm_weight * normalized
gated = scaled * silu(z)
```

Normalization occurs before the SiLU gate. Core normalization and gate math
use FP32 and return to the input dtype.

## Component

Create:

```text
tinyvllm/layers/qwen35_linear_attention.py
```

with:

```python
class Qwen35LinearAttentionShell(nn.Module):
    def __init__(
        self,
        *,
        local_key_heads: int,
        local_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        norm_eps: float,
        in_proj_qkv: nn.Module,
        in_proj_z: nn.Module,
        in_proj_b: nn.Module,
        in_proj_a: nn.Module,
        out_proj: nn.Module,
        conv_weight: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
    )

    def forward(
        self,
        hidden_states: torch.Tensor,
        convolution_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

No batch/request dimension is added in this gate. One invocation represents
one packed request segment with token-major hidden states.

## Shapes

Definitions:

```text
key_width = local_key_heads * key_head_dim
value_width = local_value_heads * value_head_dim
conv_width = 2 * key_width + value_width
```

Inputs:

```text
hidden_states:     [tokens, hidden_size]
convolution_state: [conv_width, history_width]
recurrent_state:   [local_value_heads, value_head_dim, key_head_dim]
```

Projection outputs:

```text
qkv: [tokens, conv_width]
z:   [tokens, value_width]
b:   [tokens, local_value_heads]
a:   [tokens, local_value_heads]
```

After causal convolution:

```text
query: [tokens, local_key_heads, key_head_dim]
key:   [tokens, local_key_heads, key_head_dim]
value: [tokens, local_value_heads, value_head_dim]
```

When `local_value_heads / local_key_heads > 1`, query and key repeat
interleave across heads to match local value heads. The ratio must be an
integer.

The recurrent primitive consumes physical recurrent orientation:

```text
[local_value_heads, value_head_dim, key_head_dim]
```

and returns the same orientation.

## Transactional State Semantics

The existing primitives are side-effect-free:

- convolution returns a new window;
- recurrence returns a new physical recurrent tensor.

The shell retains this behavior:

1. validate input states;
2. compute candidate convolution state;
3. compute candidate recurrent state;
4. complete gated norm and output projection;
5. return output and both candidates.

If any stage fails, the input state tensors remain byte-for-byte unchanged and
no external pool is written.

## Fail-Closed Contracts

Reject:

- invalid positive integer dimensions;
- value-head count not divisible by key-head count;
- non-positive or non-finite norm epsilon;
- parameter shape/dtype/device mismatches;
- non-rank-two or non-floating hidden states;
- convolution/recurrent state shape, dtype, or device mismatches;
- projection rank/token/feature/dtype/device failures;
- malformed primitive outputs;
- output projection rank/token/dtype/device failures.

## Test Gate

CPU tests cover:

- exact projection/conv/recurrent/gated-norm/output order;
- independent full numerical oracle;
- asymmetric key/value dimensions to detect recurrent orientation errors;
- key-head repeat into more value heads;
- one-shot versus split continuation;
- FP32 and BF16;
- input and persistent-state non-mutation;
- output-projection failure leaves both input states unchanged;
- constructor and every boundary failure.

## Claim Boundary

Passing proves a dependency-light CPU linear-attention chain and transactional
candidate-state semantics.

It does not prove:

- HybridStateTensorPool commit integration;
- batched slot gather/scatter;
- packed multi-request sequence boundaries;
- official checkpoint loading;
- optimized causal-conv/FLA/Triton kernels;
- decoder/model equivalence or GPU performance.

The immutable Qwen3.5 schema-v2 canonical result remains `NO_GO`.

