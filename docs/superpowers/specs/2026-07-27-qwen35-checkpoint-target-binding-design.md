# Qwen3.5 Checkpoint Target-Binding Design

## Status

Approved under the standing inline-execution direction. This is a CPU-only,
read-only correctness gate between the completed checkpoint tensor plan and an
assembled native Qwen3.5 model graph.

## Goal

Validate that every checkpoint tensor-plan entry resolves to exactly one
compatible TinyLLMForge destination tensor with:

- the intended concrete component type;
- the correct Parameter or Buffer kind;
- the correct source transform;
- the correct TP-local destination shape;
- the checkpoint dtype preserved locally;
- the correct packed destination slice when two sources share one parameter;
- the tied embedding/lm-head storage alias.

The gate does not read tensor payloads or write model storage.

## Current Gap

The checkpoint planner publishes logical target names such as:

```text
embed_tokens.weight
final_norm.weight
layers.0.linear_attention.in_proj_qkv.weight
layers.0.mlp.gate_up_proj.weight
```

The assembled root registers layers under:

```text
model.layer_stack.layers
```

Therefore `model.get_parameter(logical_target)` is not a valid binding
strategy. The binding gate must deliberately map the stable logical namespace
to the assembled root without changing the planner's names.

The model graph also mixes destination kinds:

```text
projection/norm/embedding weights -> Parameter
conv_weight/A_log/dt_bias/norm_weight -> Buffer
```

Finally, two source tensors:

```text
mlp.gate_proj.weight
mlp.up_proj.weight
```

share one TP-local packed destination:

```text
mlp.gate_up_proj.weight
```

with distinct packed slots.

## Alternatives Considered

### 1. Call `model.get_parameter()` on planner targets

Rejected. Logical layer paths do not match root registration paths and four
linear-attention checkpoint families target buffers, not parameters.

### 2. Reuse the generic safetensors loader

Rejected. It opens shards and calls `get_tensor()`, cannot consume the
planner's explicit transforms, and does not validate the Qwen3.5 component
selection before mutation.

### 3. Add a pure binding planner over the assembled model

Selected. A dedicated module resolves logical paths, validates component
types and local storage contracts, and returns immutable binding records
without materializing source tensors.

## Public API

Create:

```text
tinyvllm/models/qwen35_checkpoint_binding.py
```

with:

```python
@dataclass(frozen=True)
class Qwen35CheckpointTensorBinding:
    load: Qwen35CheckpointTensorLoad
    destination_name: str
    destination: torch.Tensor
    destination_kind: str
    loader_kind: str
    local_shape: tuple[int, ...]
    destination_slice: tuple[int, int] | None


@dataclass(frozen=True)
class Qwen35CheckpointBindingPlan:
    bindings: tuple[Qwen35CheckpointTensorBinding, ...]
    tensor_parallel_size: int
    tensor_parallel_rank: int


def build_qwen35_checkpoint_binding_plan(
    model: Qwen35PackedForCausalLM,
    tensor_plan: Qwen35CheckpointTensorPlan,
    *,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
) -> Qwen35CheckpointBindingPlan:
    ...
```

The tensor reference is retained so a later materialization gate can execute
the already validated operation without resolving a different object. The
record is frozen, but this gate never mutates the referenced tensor.

## Logical Path Resolution

Root targets map as:

```text
embed_tokens.* -> model.embed_tokens.*
final_norm.*   -> model.final_norm.*
layers.N.*     -> model.layer_stack.layers[N].*
```

The resolver rejects:

- malformed target syntax;
- out-of-range layer indices;
- missing attributes;
- a non-module intermediate;
- a destination that is neither a registered Parameter nor registered
  Buffer;
- duplicate `(target, packed_slot)` entries;
- an unplanned registered destination selected by ambiguous resolution.

The input model must be an exact `Qwen35PackedForCausalLM`.

## Concrete Component Contract

The immediate parent module for each target must be:

```text
embed_tokens.weight
  exact VocabParallelEmbedding

final_norm.weight
layer input/post norm weight
full q/k norm weight
  exact Qwen35OffsetRMSNorm

mlp.gate_up_proj.weight
  exact MergedColumnParallelLinear

mlp.down_proj.weight
linear_attention.out_proj.weight
full_attention.output_projection.weight
  exact RowParallelLinear

linear_attention.in_proj_qkv.weight
  exact SegmentedColumnParallelLinear

linear_attention.in_proj_z/a/b.weight
full_attention.k/v_projection.weight
  exact ColumnParallelLinear

full_attention.q_projection.weight
  exact HeadPairedColumnParallelLinear

linear_attention conv/A_log/dt_bias/norm buffers
  exact Qwen35LinearAttentionShell
```

This connects already tested loader semantics to the selected destination. It
does not execute the loaders.

## TP-Local Shape Contract

Let source metadata describe the full checkpoint tensor after the declared
lossless transform.

### Replicated

Keep the transformed full shape:

```text
all offset RMSNorm weights
linear_attention.norm_weight
```

### Vocabulary column

Shard axis 0:

```text
embed_tokens.weight [V / TP, H]
```

### Ordinary output-column sharding

Shard axis 0:

```text
linear in_proj_qkv/z/a/b
full q/k/v projections
A_log
dt_bias
conv_weight after squeeze
```

For `in_proj_qkv`, require the selected
`SegmentedColumnParallelLinear.output_sizes` to sum to the source output
dimension and every segment to be TP-divisible.

For the full q projection, require the selected
`HeadPairedColumnParallelLinear` to preserve complete head pairs.

### Row sharding

Shard axis 1:

```text
mlp.down_proj.weight
linear_attention.out_proj.weight
full_attention.output_projection.weight
```

### Packed gate/up

Each source has local shape:

```text
[intermediate_size / TP, hidden_size]
```

The shared destination shape is:

```text
[2 * intermediate_size / TP, hidden_size]
```

Packed slot `0` binds destination rows:

```text
[0, intermediate_size / TP)
```

Packed slot `1` binds:

```text
[intermediate_size / TP, 2 * intermediate_size / TP)
```

The binding record stores these `(offset, length)` slices.

Every sharded dimension must divide exactly by TP size. Rank must be in
`[0, TP)`. The destination's own `tp_size` and `tp_rank`, when exposed, must
match the requested binding context.

## Dtype and Loader Contract

Map:

```text
BF16 -> torch.bfloat16
F32  -> torch.float32
```

The destination tensor dtype must exactly match checkpoint metadata.

Loader kinds:

```text
custom_parameter_loader
default_parameter_copy
direct_buffer_copy
```

TP-sharded projection and embedding parameters require a callable custom
`weight_loader`. Offset RMSNorm parameters use default copy. Linear-attention
buffers use direct copy after the future source transform/shard operation.

## Embedding Alias Contract

The exact root `lm_head` must be a `ParallelLMHead`. Its weight must:

- have the same local shape and dtype as `embed_tokens.weight`;
- share the same underlying storage and storage offset.

The checkpoint has tied embeddings and only one source tensor. A copied
second allocation is rejected.

## Atomicity and Read-Only Semantics

The function validates all entries before publishing the tuple. It must not:

- open files;
- call a weight loader;
- call `copy_`, `set_`, `resize_`, or assign `.data`;
- change tensor values, object identity, storage pointers, dtype, or device;
- change the state pool.

Any failure returns no partial binding plan.

## Test Gate

Build real CPU component graphs at TP=1 and TP=2 using the existing:

```text
VocabParallelEmbedding / ParallelLMHead
Qwen35OffsetRMSNorm
MergedColumnParallelLinear
RowParallelLinear
SegmentedColumnParallelLinear
ColumnParallelLinear
HeadPairedColumnParallelLinear
Qwen35LinearAttentionShell
Qwen35FullAttentionShell
Qwen35DecoderLayerShell
Qwen35PackedForCausalLM
```

Use a small two-layer linear/full topology and a synthetic tensor plan.

Prove:

- every logical target resolves;
- Parameter/Buffer kind and loader kind are exact;
- TP=1 and TP=2 local shapes are exact;
- gate/up share one destination with disjoint packed slices;
- conv singleton-channel transform binds a rank-local 2-D buffer;
- F32 stable buffers remain F32 while compute destinations remain BF16;
- embedding/lm-head storage is tied;
- model tensors, storage pointers, and pool values remain unchanged.

Reject:

- wrong model type;
- invalid TP rank/size;
- layer count or block-type mismatch;
- missing/wrong component type;
- missing destination;
- parameter/buffer kind mismatch;
- wrong local shape or dtype;
- missing custom loader;
- wrong component TP metadata;
- bad packed slot or overlapping slice;
- untied lm head;
- duplicate binding entry.

## Non-Goals

This gate does not:

- create the concrete production component factory;
- read or materialize safetensors payloads;
- execute transforms or TP slicing;
- call any weight loader;
- change parameters or buffers;
- integrate the generic loader, ModelRunner, Engine, or Scheduler;
- run GPU/distributed collectives;
- establish token/logit equivalence;
- establish performance, cache, memory, compression, or quality improvement.

The immutable Qwen3.5 schema-v2 canonical result remains `NO_GO`.

## Success Criteria

The gate passes when the complete 27-entry two-layer grammar binds at TP=1/2
and all negative component/shape/dtype/alias cases fail before mutation. A
320-entry binding remains a later concrete 24-layer component-factory gate.

