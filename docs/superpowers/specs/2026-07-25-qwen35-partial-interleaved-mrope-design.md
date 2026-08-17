# Qwen3.5 Partial Interleaved MRoPE Design

## Status

Approved by the standing inline-execution direction for the Qwen3.5 native
correctness path. This is a CPU-only primitive gate.

## Objective

Add an isolated Qwen3.5 rotary primitive that matches the official text-model
semantics:

1. compute rotary frequencies only for an explicit even prefix of each head;
2. select temporal, height, and width position frequencies using interleaved
   MRoPE sections;
3. rotate only the prefix and preserve the remaining head suffix exactly.

This phase does not add a Qwen3.5 attention module, model, checkpoint loader,
or GPU execution.

## Official Source

The reviewed Hugging Face Qwen3.5 source is:

```text
/tmp/modeling_qwen3_5.py
SHA-256 15d5425ee6e771f8fbca10468c280fe62afa79fab3eff73ad1a8852162799d48
```

The official implementation:

- computes `rotary_dim = int(head_dim * partial_rotary_factor)`;
- creates inverse frequencies over `rotary_dim / 2`;
- expands one-dimensional text positions across T/H/W;
- starts with temporal frequencies;
- replaces indices `1,4,7,...` with height frequencies;
- replaces indices `2,5,8,...` with width frequencies;
- duplicates the selected half-frequency vector for split-half rotation;
- applies rotary math only to the first `rotary_dim` features.

For the Qwen3.5 pattern `(11, 11, 10)`, the half-frequency layout contains
32 entries:

```text
T,H,W,T,H,W,...,T,H
```

## Isolation Decision

Do not change `tinyvllm/layers/rotary_embedding.py`. Its current
`RotaryEmbedding` requires `rotary_dim == head_size` and is correct for
existing Qwen3 models.

Create:

```text
tinyvllm/layers/qwen35_rotary_embedding.py
```

with:

```python
class Qwen35PartialInterleavedRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        base: float,
        mrope_section: tuple[int, int, int],
    ): ...

    def forward(
        self,
        position_ids: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
```

The public input layout matches TinyLLMForge attention projections:

```text
query: [tokens, query_heads * head_dim]
key:   [tokens, key_heads * head_dim]
```

Position ids are either:

```text
[tokens]    ordinary text positions, replicated across T/H/W
[3, tokens] explicit temporal/height/width positions
```

## Frequency and Rotation Semantics

Initialization computes:

```text
inv_freq[i] = 1 / base ** (2 * i / rotary_dim)
```

For explicit T/H/W positions:

```text
freqs[axis, token, i] = position_ids[axis, token] * inv_freq[i]
```

The selected half-frequency vector starts as temporal frequencies. Height and
width entries replace the official interleaved indices:

```text
height indices = 1, 4, 7, ... within 3 * mrope_section[1]
width indices  = 2, 5, 8, ... within 3 * mrope_section[2]
```

The selected vector is duplicated:

```text
embedding = concat(selected_freqs, selected_freqs)
cos = cos(embedding)
sin = sin(embedding)
```

For each head:

```text
rotated_prefix =
  prefix * cos + rotate_half(prefix) * sin
output = concat(rotated_prefix, untouched_suffix)
```

All frequency and trigonometric math runs in FP32. Query and key outputs
preserve their respective input dtypes. Inputs are not mutated.

## Contracts

Fail closed unless:

- `head_dim` and `rotary_dim` are positive non-boolean integers;
- `rotary_dim` is even and no larger than `head_dim`;
- `base` is finite and greater than one;
- `mrope_section` contains exactly three positive non-boolean integers;
- `sum(mrope_section) == rotary_dim / 2`;
- position ids have shape `[tokens]` or `[3, tokens]`;
- position ids use an integer dtype;
- query and key are rank-two floating-point tensors;
- query and key have the same token count;
- their feature dimensions are positive multiples of `head_dim`;
- query and key use the same dtype and device as each other;
- position ids are on the same device.

Exact section coverage prevents silent omission or reuse of frequency lanes.
The primitive supports different query-head and key-head counts.

## Test Gate

The dependency-light CPU test must cover:

- an asymmetric synthetic `(2, 1, 1)` section fixture with distinct T/H/W
  positions and a manual scalar oracle;
- explicit proof of the selected lane pattern `T,H,W,T`;
- one-dimensional text positions equaling replicated three-axis positions;
- partial rotation leaving the suffix bitwise unchanged;
- different query-head and key-head counts;
- position zero identity;
- BF16 inputs with an FP32 oracle and preserved output dtype;
- no input mutation;
- rejection of odd or oversized rotary dimensions;
- rejection of section-length or section-sum mismatch;
- rejection of invalid position shape/dtype;
- rejection of query/key rank, token, feature, dtype, and device mismatch.

## Claim Boundary

Passing proves only the isolated partial/interleaved rotary formula and
token-major shape contract.

It does not prove:

- Qwen3.5 attention or `o_proj` integration;
- native model or checkpoint loading;
- layer, hidden-state, or logit equivalence;
- cached/chunked/interleaved request correctness;
- any latency, throughput, compression, quality, or GPU-memory improvement.

The immutable Qwen3.5 schema-v2 canonical result remains `NO_GO`.
