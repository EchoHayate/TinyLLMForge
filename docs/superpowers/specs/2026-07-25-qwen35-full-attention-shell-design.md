# Qwen3.5 Full-Attention Projection Shell Design

## Status

Approved by the standing inline-execution direction for the Qwen3.5 native
correctness path. This is a CPU-only dependency-light module gate.

## Objective

Connect the already proven Qwen3.5 primitives into the exact full-attention
operation order without adding a production model:

```text
head-paired q/query-gate projection
separate k and v projections
per-head offset q/k RMSNorm
partial interleaved MRoPE
injected attention backend
flattened query-output sigmoid gate
injected output projection
```

The gate must also prove the official q/query-gate checkpoint row layout and
its tensor-parallel sharding rule.

## Official Layout Correction

The official Hugging Face source:

```text
/tmp/modeling_qwen3_5.py
SHA-256 15d5425ee6e771f8fbca10468c280fe62afa79fab3eff73ad1a8852162799d48
```

constructs:

```python
q_proj_output.view(tokens, num_heads, 2 * head_dim)
torch.chunk(..., 2, dim=-1)
```

Therefore checkpoint rows are head-major:

```text
[query_head_0, gate_head_0,
 query_head_1, gate_head_1,
 ...]
```

They are not:

```text
[all_query_heads, all_gate_heads]
```

At TP size `P`, rank `r` must receive one contiguous range containing:

```text
num_heads / P complete head pairs
```

This is ordinary column-parallel sharding with an additional alignment
contract, not `SegmentedColumnParallelLinear`.

## Components

### Head-Paired Projection

Add to `tinyvllm/layers/linear.py`:

```python
class HeadPairedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        num_heads: int,
        head_dim: int,
        bias: bool = False,
    ): ...

    def split_query_gate(
        self,
        projected: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
```

Constructor requirements:

- positive non-boolean integer `num_heads` and `head_dim`;
- `num_heads` divisible by TP size;
- output size `num_heads * 2 * head_dim`;
- local head count `num_heads / TP`.

`split_query_gate` requires rank-two floating output shaped:

```text
[tokens, local_heads * 2 * head_dim]
```

It reshapes to:

```text
[tokens, local_heads, 2 * head_dim]
```

then chunks each head into:

```text
query: [tokens, local_heads, head_dim]
gate:  [tokens, local_heads * head_dim]
```

### Dependency-Light Shell

Create:

```text
tinyvllm/layers/qwen35_full_attention.py
```

with:

```python
class Qwen35FullAttentionShell(nn.Module):
    def __init__(
        self,
        *,
        head_dim: int,
        local_query_heads: int,
        local_kv_heads: int,
        q_projection: nn.Module,
        k_projection: nn.Module,
        v_projection: nn.Module,
        q_norm: nn.Module,
        k_norm: nn.Module,
        rotary: nn.Module,
        attention_backend: nn.Module,
        output_projection: nn.Module,
    ): ...

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor: ...
```

The shell accepts injected modules so CPU tests can prove data flow without
initializing distributed process groups, paged KV, or CUDA kernels.

## Operation Order and Shapes

For `tokens = T`:

```text
paired_qg = q_projection(hidden_states)
  [T, local_query_heads * 2 * head_dim]

paired_qg -> [T, local_query_heads, 2 * head_dim]
query, gate = chunk(..., 2, dim=-1)
query = [T, local_query_heads, head_dim]
gate = [T, local_query_heads * head_dim]

key = k_projection(hidden_states)
  [T, local_kv_heads * head_dim]
value = v_projection(hidden_states)
  [T, local_kv_heads * head_dim]

query = q_norm(query)
key = k_norm(key.view(T, local_kv_heads, head_dim))

query, key = rotary(
  position_ids,
  query.reshape(T, -1),
  key.reshape(T, -1),
)

attention_output = attention_backend(
  query,
  key,
  value,
)
  [T, local_query_heads * head_dim]

gated = qwen35_apply_query_gate(attention_output, gate)
output = output_projection(gated)
```

The backend receives flattened token-major query, key, and value projections,
matching the existing TinyLLMForge attention boundary.

## Fail-Closed Contracts

The head-paired projection rejects:

- invalid head counts or dimensions;
- query heads not divisible by TP size;
- projected output rank, dtype, or exact feature mismatch.

The shell rejects:

- invalid local head counts or dimensions;
- non-floating or non-rank-two hidden states;
- any projection output with incorrect rank, token count, feature count,
  dtype, or device;
- q/k norm outputs with changed shape, dtype, or device;
- rotary outputs with changed shape, dtype, or device;
- attention output not exactly
  `[tokens, local_query_heads * head_dim]`;
- output projection changing token count or returning a non-floating tensor.

All checks occur at the closest component boundary.

## Test Gate

CPU tests must cover:

- TP=1/2/4 head-coded q/gate source rows;
- all ranks receiving contiguous complete head pairs;
- a counterexample proving global `[Q_all,gate_all]` segmented sharding gives
  different rows;
- exact per-head query/gate split;
- asymmetric local query and KV head counts;
- q/k normalization before rotary;
- rotary before attention backend;
- value bypassing q/k normalization and rotary;
- attention output gating before output projection;
- gate zero and non-zero behavior through the complete shell;
- no input mutation;
- BF16 dtype preservation through the shell;
- every shape/dtype/device boundary failure.

## Claim Boundary

Passing proves:

- correct head-paired q/query-gate contiguous TP layout;
- exact dependency-light full-attention projection and primitive ordering.

It does not prove:

- the production paged-attention backend;
- checkpoint-name mapping or real checkpoint loading;
- a decoder layer or native Qwen3.5 model;
- distributed execution;
- layer, hidden-state, or logit equivalence;
- cached/chunked/interleaved request correctness;
- any compression, quality, latency, throughput, or memory improvement.

The immutable Qwen3.5 schema-v2 canonical result remains `NO_GO`.
