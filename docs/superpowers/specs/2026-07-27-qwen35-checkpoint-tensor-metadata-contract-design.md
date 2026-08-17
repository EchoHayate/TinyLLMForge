# Qwen3.5 Checkpoint Tensor-Metadata Contract Design

## Goal

Extend the completed weight-name gate with a CPU-only, read-only safetensors
metadata contract. The new gate must validate the exact dtype, shape, byte
size, and data-offset layout of every language-model tensor before any tensor
payload is read or any model object is constructed.

## Source Evidence

The fixed source remains:

```text
repository: Qwen/Qwen3.5-2B
revision: 15852e8c16360a2fea060d615a32b45270f8a8fc
shard size: 4548221488
index metadata total_size: 4548144832
```

Only these byte ranges were read:

```text
bytes 0-7: safetensors header length
bytes 8-76655: 76648-byte JSON header
```

No tensor payload byte was requested. The parsed header contains 632 entries:

```text
language model: 320
visual: 297
MTP: 15
BF16: 596
F32: 36
```

The language-model subset is:

```text
BF16: 284
F32: 36
```

All 36 F32 tensors are the 18 linear-attention `A_log` values and the 18
linear-attention normalization weights.

## Critical Findings

### Mixed parameter dtypes are intentional

The checkpoint is not uniformly BF16:

```text
linear_attn.A_log:      F32 [linear_num_value_heads]
linear_attn.norm.weight F32 [linear_value_head_dim]
```

The metadata gate must preserve this evidence. It must not normalize the two
families to the model compute dtype.

### Convolution storage has a singleton channel dimension

The checkpoint stores:

```text
linear_attn.conv1d.weight:
[conv_width, 1, linear_conv_kernel_dim]
```

The current CPU shell accepts the logical kernel:

```text
[conv_width, linear_conv_kernel_dim]
```

The later tensor-loader gate therefore needs one explicit, lossless
`squeeze(1)` transform. The metadata gate records this transform but does not
apply it.

### Current shell dtype construction is not load-ready

`Qwen35LinearAttentionShell` currently requires `conv_weight`, `A_log`,
`dt_bias`, and `norm_weight` to share one dtype. The real checkpoint uses:

```text
conv_weight: BF16
A_log: F32
dt_bias: BF16
norm_weight: F32
```

Therefore direct checkpoint loading into the current shell must remain
blocked. This gate reports source metadata only; a later component-construction
gate must deliberately support mixed stable-parameter dtypes before loading.

## Alternatives Considered

### 1. Use `safe_open()` and inspect tensors

Rejected because even metadata-only use couples the gate to safetensors and
the local shard path, and a future refactor could accidentally call
`get_tensor()`.

### 2. Load tensors on CPU and compare `shape`/`dtype`

Rejected because it reads the 4.5 GB payload and mixes metadata validation
with allocation and mutation concerns.

### 3. Parse the safetensors JSON header only

Selected. The safetensors format exposes dtype, shape, and data offsets in a
bounded JSON header. The planner can validate the entire payload layout
without reading payload bytes.

## Architecture

Extend `tinyvllm/models/qwen35_checkpoint.py` with:

```python
@dataclass(frozen=True)
class Qwen35CheckpointTensorMetadata:
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


@dataclass(frozen=True)
class Qwen35CheckpointTensorLoad:
    weight: Qwen35CheckpointLoadTarget
    metadata: Qwen35CheckpointTensorMetadata
    transform: str


@dataclass(frozen=True)
class Qwen35CheckpointTensorPlan:
    loads: tuple[Qwen35CheckpointTensorLoad, ...]
    skips: tuple[Qwen35CheckpointSkip, ...]
    payload_bytes: int
```

The public factory is:

```python
def build_qwen35_checkpoint_tensor_plan(
    hf_config,
    index_payload: Mapping[str, object],
    shard_headers: Mapping[str, Mapping[str, object]],
) -> Qwen35CheckpointTensorPlan
```

The factory first calls `build_qwen35_checkpoint_weight_plan()`. It then
validates each declared shard header and attaches metadata to every planned
language-model load.

## Config Contract

Resolve through `text_config` and require the existing weight-name fields plus:

```text
dtype
hidden_size
intermediate_size
vocab_size
linear_num_key_heads
linear_num_value_heads
linear_key_head_dim
linear_value_head_dim
linear_conv_kernel_dim
num_attention_heads
num_key_value_heads
head_dim
```

Accepted checkpoint dtype strings are:

```text
bfloat16 -> BF16
float32  -> F32
```

The config compute dtype determines all ordinary text parameters. `A_log` and
linear norm are always expected as `F32` for this fixed Qwen3.5 contract.

## Expected Shape Contract

Let:

```text
H  = hidden_size
I  = intermediate_size
V  = vocab_size
QH = num_attention_heads
KVH = num_key_value_heads
D  = head_dim
LKH = linear_num_key_heads
LVH = linear_num_value_heads
LKD = linear_key_head_dim
LVD = linear_value_head_dim
K = linear_conv_kernel_dim
KW = LKH * LKD
VW = LVH * LVD
CW = 2 * KW + VW
```

Root:

```text
embed_tokens.weight [V, H] compute dtype
norm.weight         [H]    compute dtype
```

Every layer:

```text
input_layernorm.weight          [H]    compute dtype
post_attention_layernorm.weight [H]    compute dtype
mlp.gate_proj.weight            [I, H] compute dtype
mlp.up_proj.weight              [I, H] compute dtype
mlp.down_proj.weight            [H, I] compute dtype
```

Linear-attention layer:

```text
in_proj_qkv.weight [CW, H] compute dtype
in_proj_z.weight   [VW, H] compute dtype
in_proj_b.weight   [LVH, H] compute dtype
in_proj_a.weight   [LVH, H] compute dtype
out_proj.weight    [H, VW] compute dtype
conv1d.weight      [CW, 1, K] compute dtype
A_log              [LVH] F32
dt_bias            [LVH] compute dtype
norm.weight        [LVD] F32
```

Full-attention layer:

```text
q_proj.weight [QH * 2 * D, H] compute dtype
k_proj.weight [KVH * D, H]    compute dtype
v_proj.weight [KVH * D, H]    compute dtype
o_proj.weight [H, QH * D]     compute dtype
q_norm.weight [D]             compute dtype
k_norm.weight [D]             compute dtype
```

## Transform Contract

Only two transform values are accepted:

```text
identity
squeeze_conv_channel
```

Every source uses `identity` except:

```text
*.linear_attn.conv1d.weight -> squeeze_conv_channel
```

The transform is descriptive. This gate does not allocate an output tensor.

## Header Validation

Every shard in the weight plan must have exactly one header mapping. Unknown
header shards and missing planned shards fail closed.

Each tensor entry must contain:

```text
dtype: non-empty string
shape: list/tuple of positive exact integers
data_offsets: two non-negative exact integers, end > start
```

The byte width table is:

```text
BF16: 2
F32: 4
```

For every entry:

```text
end - start == product(shape) * byte_width(dtype)
```

Within each shard:

- offsets must not overlap;
- the first payload offset must be zero;
- entries may appear in any JSON key order;
- sorted intervals must be contiguous with no holes;
- the final offset is that shard's validated payload byte count.

Across all planned shards:

```text
sum(shard final offsets)
==
index_payload["metadata"]["total_size"]
```

The complete header source-name set must equal the index weight-map set.
Metadata for visual/MTP entries is validated structurally and for byte layout
even though those entries remain explicit skips.

## Failure Atomicity

The factory builds local immutable values only after:

1. the weight-name plan succeeds;
2. all headers match planned shards;
3. all 632 names match;
4. all metadata fields and byte sizes are valid;
5. all intervals form one contiguous payload;
6. all 320 text tensors match expected dtype/shape.

Any failure publishes no tensor plan and performs no I/O or mutation.

## Testing

Add focused tests to
`tools/test_qwen35_checkpoint_weight_name_contract.py`.

The tests will:

1. generate exact synthetic headers from a small interleaved config;
2. verify identity and convolution-squeeze transforms;
3. verify mixed BF16/F32 expectations;
4. reject wrong shape or dtype for every parameter family;
5. reject malformed shapes/offsets, unknown dtypes, overlaps, holes, and byte
   count mismatches;
6. reject missing/extra header names and shard mismatches;
7. validate the fixed real 632-entry header fixture from parsed JSON metadata;
8. prove the function performs no file or tensor I/O;
9. run under Python 3.9 and Python 3.12;
10. rerun weight-name and Qwen3.5 root/owner/factory regressions.

The 76,648-byte real header remains temporary evidence and is not copied into
the repository.

## Allowed Conclusion

After this gate passes:

> TinyLLMForge can validate the exact dtype, shape, byte size, and contiguous
> payload layout of all 632 tensors in the verified Qwen3.5-2B safetensors
> header, and can attach exact metadata plus required lossless transforms to
> all 320 language-model load targets without reading tensor payloads.

This gate does not establish:

- compatibility with the current linear-attention shell dtype constraints;
- TP-local target shapes;
- tensor materialization or assignment;
- embedding/LM-head storage aliasing;
- production Qwen3.5 construction or selection;
- checkpoint token/logit equivalence;
- any performance, cache, memory, compression, or quality benefit.

The immutable Qwen3.5 schema-v2 canonical result remains `NO_GO`.
