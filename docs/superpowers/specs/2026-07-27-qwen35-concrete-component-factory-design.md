# Qwen3.5 Concrete Component Factory Design

## Status

Approved under the standing inline-execution direction. This is a local
CPU/static correctness gate. It does not load checkpoint payloads, execute a
model forward pass, initialize CUDA, or connect production runtime selection.

## Goal

Construct a complete checkpoint-bearing TinyLLMForge Qwen3.5 graph directly
from:

- a verified Hugging Face-style Qwen3.5 config;
- an already-created `HybridStateTensorPool`;
- an explicit tensor-parallel size and rank;
- an injected full-attention backend factory.

The graph must contain all 24 concrete decoder layers and every destination
required by the real 320-entry language-model tensor plan. The existing
read-only checkpoint binding planner must then resolve all 320 entries without
reading payload bytes or mutating model/state storage.

## Current Gap

`assemble_qwen35_packed_model()` validates topology and state ownership, but
requires a caller-supplied decoder-layer callback. The current 27-entry binding
fixture constructs concrete components manually. Therefore it proves the
binding grammar for one linear layer and one full-attention layer, but does not
prove that a single config-driven factory can construct the verified 24-layer
Qwen3.5-2B topology.

The real checkpoint-bearing graph is approximately 4.5 GB in BF16/F32 storage.
Allocating those parameter payloads on local CPU would provide no additional
binding evidence and would make this metadata-only gate unnecessarily
expensive.

## Alternatives Considered

### 1. Allocate a real CPU graph

Rejected. It would allocate roughly the full checkpoint payload while this gate
only inspects tensor identity, shape, dtype, type, loader capability, and
logical binding.

### 2. Build synthetic small layers and extrapolate to 24 layers

Rejected. The completed 27-entry test already covers that level. It cannot
detect config-field mistakes, 24-layer topology mistakes, or missing
destinations in the real 320-entry grammar.

### 3. Build all checkpoint-bearing tensors on `meta`

Selected. PyTorch `meta` tensors preserve module structure, shape, dtype,
Parameter/Buffer registration, and loader attributes without allocating tensor
payload storage. The supplied state pool remains real and unchanged because it
is the state owner, not checkpoint weight storage.

## Public API

Create:

```text
tinyvllm/models/qwen35_components.py
```

with:

```python
@dataclass(frozen=True)
class Qwen35ConcreteComponentAssembly:
    packed: Qwen35PackedModelAssembly
    tensor_parallel_size: int
    tensor_parallel_rank: int
    parameter_device: torch.device
    compute_dtype: torch.dtype
    stable_dtype: torch.dtype


def build_qwen35_concrete_component_assembly(
    hf_config,
    *,
    pool: HybridStateTensorPool,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
    build_attention_backend: Callable[
        [int, int, int, int],
        nn.Module,
    ],
    parameter_device: str | torch.device = "meta",
) -> Qwen35ConcreteComponentAssembly:
    ...
```

The callback arguments are:

```text
layer_index
local_query_heads
local_kv_heads
head_dim
```

The callback must return an `nn.Module`. The factory stores it in the
corresponding `Qwen35FullAttentionShell` but never calls it.

## Config Contract

The factory accepts `hf_config.text_config` when present, otherwise
`hf_config`. It validates these fields before publishing an assembly:

```text
dtype == "bfloat16"
hidden_size
intermediate_size
vocab_size
num_hidden_layers
layer_types
linear_num_key_heads
linear_num_value_heads
linear_key_head_dim
linear_value_head_dim
linear_conv_kernel_dim
num_attention_heads
num_key_value_heads
head_dim
rms_norm_eps
hidden_act == "silu"
tie_word_embeddings is True
rope_parameters
```

`rope_parameters` must provide:

```text
rope_theta
partial_rotary_factor
mrope_section
```

The rotary dimension is:

```text
rotary_dim = int(head_dim * partial_rotary_factor)
```

It must be positive, even, no larger than `head_dim`, and satisfy:

```text
sum(mrope_section) == rotary_dim / 2
```

All global head counts, projection segment widths, intermediate size, and
vocabulary size that are TP-sharded must be exactly divisible by
`tensor_parallel_size`.

## Device and Dtype Contract

The default and real-gate device is `meta`. Only `meta` and `cpu` are accepted
by this static construction gate; CUDA devices are rejected.

Checkpoint-bearing compute tensors use:

```text
torch.bfloat16
```

This includes:

- embedding and LM-head weight;
- all RMSNorm weights except the linear-attention internal norm;
- all MLP, linear-attention, and full-attention projection weights;
- linear-attention `conv_weight`;
- linear-attention `dt_bias`.

Stable linear-attention tensors use:

```text
A_log      -> torch.float32
norm_weight -> torch.float32
```

The factory must not downcast stable tensors or upcast compute tensors.

## Tensor-Parallel Construction

Existing TinyLLMForge TP layers read rank and world size from
`torch.distributed` during construction. The factory accepts explicit TP
arguments and enters a narrowly scoped construction context that makes those
values observable only while modules are created.

The context must:

- validate rank in `[0, tensor_parallel_size)`;
- restore the original distributed callables even after an exception;
- avoid process-group initialization;
- avoid collective operations because no forward pass occurs.

Every created TP component must retain the explicit `tp_size` and `tp_rank`.

## Concrete Graph

Root:

```text
embed_tokens -> exact VocabParallelEmbedding
final_norm   -> exact Qwen35OffsetRMSNorm
lm_head      -> exact ParallelLMHead
```

`lm_head.weight` must be the exact same `Parameter` object as
`embed_tokens.weight`. This is stronger than storage-pointer equality and is
required for `meta` tensors because independent meta storages can both expose
pointer zero.

Every layer contains:

```text
input_layernorm          -> exact Qwen35OffsetRMSNorm
post_attention_layernorm -> exact Qwen35OffsetRMSNorm
mlp.gate_up_proj         -> exact MergedColumnParallelLinear
mlp.down_proj            -> exact RowParallelLinear
```

The MLP is a small Qwen3.5-specific module in
`qwen35_components.py`. It uses `torch.nn.functional.silu` and elementwise
multiplication in `forward`; construction and binding do not invoke it.

Linear-attention layers contain:

```text
in_proj_qkv -> exact SegmentedColumnParallelLinear
in_proj_z   -> exact ColumnParallelLinear
in_proj_b   -> exact ColumnParallelLinear
in_proj_a   -> exact ColumnParallelLinear
out_proj    -> exact RowParallelLinear
shell       -> exact Qwen35LinearAttentionShell
```

The registered shell buffers have TP-local shapes:

```text
conv_weight:
  [
    (
      2 * linear_num_key_heads * linear_key_head_dim
      + linear_num_value_heads * linear_value_head_dim
    ) / TP,
    linear_conv_kernel_dim,
  ]

A_log:
  [linear_num_value_heads / TP]

dt_bias:
  [linear_num_value_heads / TP]

norm_weight:
  [linear_value_head_dim]
```

Full-attention layers contain:

```text
q_projection     -> exact HeadPairedColumnParallelLinear
k_projection     -> exact ColumnParallelLinear
v_projection     -> exact ColumnParallelLinear
q_norm           -> exact Qwen35OffsetRMSNorm
k_norm           -> exact Qwen35OffsetRMSNorm
rotary           -> exact Qwen35PartialInterleavedRotaryEmbedding
attention_backend -> injected module
output_projection -> exact RowParallelLinear
shell            -> exact Qwen35FullAttentionShell
```

The layer callback passed to `assemble_qwen35_packed_model()` returns exact
`Qwen35DecoderLayerShell` values and uses the supplied state adapter only
through the existing assembly factory. The supplied pool remains the sole
state-tensor owner.

## Meta-Safe Alias Validation

The checkpoint binding planner currently validates tied storage by pointer and
offset. That is sufficient for allocated CPU tensors but insufficient on
`meta`, where unrelated storages can both report pointer zero.

Strengthen `_validate_embedding_alias()` so:

```text
meta destination -> embed_tokens.weight is lm_head.weight
allocated destination -> exact shared storage pointer and offset
```

Shape and dtype equality remain required in both cases. A factory or fixture
that creates two independent meta Parameters must fail closed.

## Real 320-Entry Gate

The test consumes the verified bounded metadata files:

```text
/tmp/qwen35-2b-15852e8-config.json
/tmp/qwen35-2b-15852e8-model.safetensors.index.json
/tmp/qwen35-safetensors-header.json
```

It parses only JSON and calls:

```python
weight_plan = build_qwen35_checkpoint_weight_plan(config, index_payload)
tensor_plan = build_qwen35_checkpoint_tensor_plan(
    config,
    index_payload,
    shard_headers,
)
assembly = build_qwen35_concrete_component_assembly(...)
binding_plan = build_qwen35_checkpoint_binding_plan(
    assembly.packed.model,
    tensor_plan,
    tensor_parallel_size=...,
    tensor_parallel_rank=...,
)
```

The gate runs TP=1 and TP=2 for every rank. It asserts:

```text
24 layers
18 linear-attention layers
6 full-attention layers
320 tensor loads
320 bindings
284 BF16 destinations
36 F32 destinations
18 squeeze_conv_channel loads
2 root targets
252 linear-layer targets
66 full-layer targets
```

It also verifies:

- exact layer-type sequence from the real config;
- exact concrete component types;
- exact Parameter/Buffer counts implied by the binding plan;
- tied embedding/LM-head object identity;
- all checkpoint-bearing destination tensors remain on `meta`;
- every pool tensor preserves object identity, device, dtype, shape, and value;
- planner calls do not mutate any module registration or tensor reference.

The test must not open any `.safetensors` shard. It must not call
`safe_open()`, `get_tensor()`, a destination loader, or model forward.

## Failure Policy

Fail closed on:

- missing, malformed, or unsupported config fields;
- unsupported dtype, activation, untied embeddings, or CUDA device;
- invalid TP size/rank or TP indivisibility;
- pool topology inconsistent with config;
- malformed RoPE parameters;
- a backend callback that is not callable or returns a non-module;
- any wrong component type, registration kind, local shape, dtype, TP metadata,
  packed slice, or tied alias exposed by the binding planner.

The factory must restore the distributed construction context on all failures.

## Test Strategy

### TDD cycle 1: meta alias hardening

Add a focused test proving two independent meta Parameters are rejected even
though both expose pointer zero. Verify the existing shared-object case passes.

### TDD cycle 2: small concrete factory

Use a two-layer config and a real supplied CPU state pool. Validate TP=1/2,
exact components, dtypes, shapes, tied object identity, callback arguments,
read-only pool behavior, and representative failures.

### TDD cycle 3: real 24-layer binding

Use the three verified JSON artifacts, construct the full graph on `meta`, and
bind the real 320-entry tensor plan at TP=1/2.

### Regression

Run the existing checkpoint metadata, target-binding, TP primitive, linear/full
attention shell, root assembly, transactional root, and native-owner tests.
Run focused Python compilation, `git diff --check`, production-wiring guards,
and staged-file count.

## Non-Goals

This gate does not establish:

- source tensor materialization;
- safetensors payload reads;
- transform or TP-slice execution;
- loader invocation or assignment transaction;
- CPU or GPU model forward correctness;
- production `ModelRunner` Qwen3.5 selection;
- `LLMEngine.step()` native Qwen3.5 execution;
- Scheduler admission;
- checkpoint token/logit equivalence;
- optimized kernels;
- speed, latency, throughput, cache, memory, compression, or quality gains.

The Qwen3.5 schema-v2 canonical `NO_GO` remains unchanged.

## Allowed Conclusion

After this gate passes:

> TinyLLMForge can construct the complete verified 24-layer Qwen3.5-2B
> checkpoint-bearing component graph without allocating checkpoint payload
> storage, preserve the supplied hybrid-state pool as the sole state owner,
> and bind all 320 language-model tensor-plan entries to exact TP-local
> Parameters/Buffers at TP=1/2 without reading payloads or mutating the model.
