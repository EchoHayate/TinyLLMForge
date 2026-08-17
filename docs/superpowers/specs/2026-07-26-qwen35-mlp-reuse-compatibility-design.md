# Qwen3.5 MLP Reuse Compatibility Design

## Status

Approved under the standing inline-execution direction. This gate determines
whether TinyLLMForge needs a separate Qwen3.5 MLP implementation.

## Objective

Prove or reject reuse of the existing `tinyvllm.models.qwen3.Qwen3MLP` for
Qwen3.5.

The official Qwen3.5 formula is:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

The existing TinyLLMForge Qwen3 MLP is:

```text
gate_up = gate_up_proj(x)
activated = SiluAndMul(gate_up)
output = down_proj(activated)
```

with gate/up column-parallel sharding and down row-parallel sharding.

## Decision

Do not create a duplicate Qwen3.5 MLP class if a TP-aware compatibility test
proves exact equivalence. Reuse reduces code and checkpoint-layout drift.

## Test Method

Instantiate the real existing `Qwen3MLP` under synthetic TP layouts:

```text
TP = 1, 2, 4
```

For every rank:

1. load the global gate weight through
   `MergedColumnParallelLinear.weight_loader(..., loaded_shared_id=0)`;
2. load the global up weight through
   `MergedColumnParallelLinear.weight_loader(..., loaded_shared_id=1)`;
3. load the global down weight through
   `RowParallelLinear.weight_loader`;
4. run the real `Qwen3MLP.forward`;
5. treat each rank output as its row-parallel partial sum.

Sum rank outputs and compare with an independent full-tensor oracle:

```python
gate = F.linear(x, gate_weight)
up = F.linear(x, up_weight)
expected = F.linear(F.silu(gate) * up, down_weight)
```

The synthetic row codes must prove:

- local fused order is `[gate_rank_shard, up_rank_shard]`;
- each down-projection rank consumes the corresponding intermediate shard;
- the sum of rank-local down outputs equals the official global result.

## Coverage

Tests cover:

- TP=1/2/4;
- all ranks;
- deterministic unequal gate and up values;
- FP32 numerical equivalence;
- BF16 dtype preservation and FP32 comparison;
- unchanged input;
- the existing `hidden_act == "silu"` guard.

## Claim Boundary

Passing proves the existing unquantized Qwen3 MLP math and TP checkpoint
layout are compatible with the official Qwen3.5 MLP.

It does not prove:

- a complete Qwen3.5 checkpoint-name traversal;
- quantized MLP loading or kernels;
- distributed collective execution;
- real-model layer equivalence;
- model construction or performance improvement.

The immutable Qwen3.5 schema-v2 canonical result remains `NO_GO`.

