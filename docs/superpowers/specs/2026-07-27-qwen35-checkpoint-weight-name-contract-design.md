# Qwen3.5 Checkpoint Weight-Name Contract Design

## Goal

Define a CPU-only, read-only contract that validates the official
Qwen3.5-2B checkpoint index and produces an immutable text-model weight plan.
This gate proves source-name coverage and scope isolation before any
safetensors shard is opened, any tensor is materialized, or any model
parameter is mutated.

## Scope

This gate consumes:

- a parsed Hugging Face config or its `text_config`;
- a parsed `model.safetensors.index.json` payload;
- the expected Qwen3.5 layer topology.

It produces:

- one immutable plan entry for every checkpoint key;
- exact counts for text-model entries and explicitly skipped scopes;
- an ordered source-name inventory for the 320 language-model tensors;
- a source-name-to-logical-target contract for later loader work.

This gate does not:

- open `.safetensors` files;
- call `safe_open()` or `get_tensor()`;
- allocate checkpoint tensors;
- inspect or mutate a live `nn.Module`;
- perform TP slicing, dtype conversion, device transfer, or quantization;
- install the Qwen3.5 root into `ModelRunner`;
- change Engine or Scheduler wiring.

## Source Evidence

The source checkpoint is fixed to the already acquired manifest evidence:

```text
repository: Qwen/Qwen3.5-2B
revision: 15852e8c16360a2fea060d615a32b45270f8a8fc
index size: 64460 bytes
index sha256: aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9
weight shard: model.safetensors-00001-of-00001.safetensors
index metadata total_size: 4548144832
```

The verified index contains 632 keys:

```text
model.language_model.*: 320
model.visual.*: 297
mtp.*: 15
```

This explains the existing official load log:

```text
Loading weights: 320/320
```

The official causal language model materializes the language-model subset,
not all 632 tensors stored in the multimodal/MTP checkpoint.

## Alternatives Considered

### 1. Extend the generic loader immediately

Modify `tinyvllm/utils/loader.py` to special-case Qwen3.5 names and load the
real shard.

Rejected for this gate because it combines source-name discovery, target-name
mapping, tensor materialization, TP slicing, and mutation. A missing or
ambiguous mapping could partially mutate the model before failing.

### 2. Compare the index directly with `model.named_parameters()`

Construct a live Qwen3.5 model and compare source keys with runtime parameter
names.

Rejected for this gate because concrete TP/CUDA construction is not complete,
the current shell uses intentionally different internal names, and live model
construction would make this gate dependent on process groups and devices.

### 3. Build an index-first immutable plan

Parse only JSON metadata, validate the exact topology-derived source grammar,
and classify every key before a later loader consumes the plan.

Selected because it is deterministic, dependency-light, fail-closed, and
keeps discovery separate from mutation.

## Architecture

Create `tinyvllm/models/qwen35_checkpoint.py` with four frozen data types:

```python
@dataclass(frozen=True)
class Qwen35CheckpointSource:
    name: str
    shard: str


@dataclass(frozen=True)
class Qwen35CheckpointLoadTarget:
    source: Qwen35CheckpointSource
    target: str
    packed_slot: str | int | None


@dataclass(frozen=True)
class Qwen35CheckpointSkip:
    source: Qwen35CheckpointSource
    scope: str


@dataclass(frozen=True)
class Qwen35CheckpointWeightPlan:
    loads: tuple[Qwen35CheckpointLoadTarget, ...]
    skips: tuple[Qwen35CheckpointSkip, ...]
    shards: tuple[str, ...]
```

The public factory is:

```python
def build_qwen35_checkpoint_weight_plan(
    hf_config,
    index_payload: Mapping[str, object],
) -> Qwen35CheckpointWeightPlan
```

The returned tuples are lexicographically ordered by source name. The
function retains only strings and immutable tuples; it never retains or
accepts tensor values.

## Accepted Source Grammar

The config must resolve through `getattr(hf_config, "text_config", hf_config)`
and provide:

- exact positive `num_hidden_layers`;
- exact `layer_types` length;
- only `linear_attention` or `full_attention` layer types;
- `tie_word_embeddings is True`.

The index payload must contain one non-empty `weight_map` mapping exact
non-empty source-name strings to exact non-empty relative shard-name strings.
Absolute paths, `..` path segments, duplicate normalized names, and shard
names that do not end in `.safetensors` are rejected.

Only these top-level scopes are accepted:

```text
model.language_model.
model.visual.
mtp.
```

Unknown scopes fail closed rather than being silently skipped.

### Root text weights

Exactly these two language-model root weights are required:

```text
model.language_model.embed_tokens.weight
model.language_model.norm.weight
```

No independent `lm_head.weight` is accepted because the fixed config uses
`tie_word_embeddings=true`. The later model-construction gate must alias the
LM head to the embedding storage; this contract does not create that alias.

### Shared decoder weights

Every decoder layer requires:

```text
input_layernorm.weight
post_attention_layernorm.weight
mlp.gate_proj.weight
mlp.up_proj.weight
mlp.down_proj.weight
```

### Linear-attention layer weights

Every `linear_attention` layer additionally requires:

```text
linear_attn.in_proj_qkv.weight
linear_attn.in_proj_z.weight
linear_attn.in_proj_b.weight
linear_attn.in_proj_a.weight
linear_attn.out_proj.weight
linear_attn.conv1d.weight
linear_attn.A_log
linear_attn.dt_bias
linear_attn.norm.weight
```

For the validated Qwen3.5-2B topology, each linear layer has 14 source
weights.

### Full-attention layer weights

Every `full_attention` layer additionally requires:

```text
self_attn.q_proj.weight
self_attn.k_proj.weight
self_attn.v_proj.weight
self_attn.o_proj.weight
self_attn.q_norm.weight
self_attn.k_norm.weight
```

For the validated Qwen3.5-2B topology, each full layer has 11 source weights.

Any missing required name, unexpected language-model name, cross-type mixer
name, layer index outside the config range, or duplicate logical
target-and-packed-slot pair fails the whole plan.

## Logical Target Names

The plan uses canonical TinyLLMForge logical targets, not current
`state_dict()` paths. This avoids coupling the contract to wrapper names such
as `layer_stack.layers` or future construction details.

Root targets:

```text
embed_tokens.weight
final_norm.weight
```

Layer targets start with:

```text
layers.{index}.
```

Shared mappings:

```text
input_layernorm.weight              -> input_layernorm.weight
post_attention_layernorm.weight     -> post_attention_layernorm.weight
mlp.down_proj.weight                -> mlp.down_proj.weight
mlp.gate_proj.weight                -> mlp.gate_up_proj.weight, packed_slot=0
mlp.up_proj.weight                  -> mlp.gate_up_proj.weight, packed_slot=1
```

Linear-attention mappings:

```text
linear_attn.in_proj_qkv.weight -> linear_attention.in_proj_qkv.weight
linear_attn.in_proj_z.weight   -> linear_attention.in_proj_z.weight
linear_attn.in_proj_b.weight   -> linear_attention.in_proj_b.weight
linear_attn.in_proj_a.weight   -> linear_attention.in_proj_a.weight
linear_attn.out_proj.weight    -> linear_attention.out_proj.weight
linear_attn.conv1d.weight      -> linear_attention.conv_weight
linear_attn.A_log              -> linear_attention.A_log
linear_attn.dt_bias            -> linear_attention.dt_bias
linear_attn.norm.weight        -> linear_attention.norm_weight
```

Full-attention mappings:

```text
self_attn.q_proj.weight -> full_attention.q_projection.weight
self_attn.k_proj.weight -> full_attention.k_projection.weight
self_attn.v_proj.weight -> full_attention.v_projection.weight
self_attn.o_proj.weight -> full_attention.output_projection.weight
self_attn.q_norm.weight -> full_attention.q_norm.weight
self_attn.k_norm.weight -> full_attention.k_norm.weight
```

`packed_slot` is non-`None` only for MLP gate/up sources. Full-attention Q/K/V
remain separate because the current Qwen3.5 shell deliberately models the
query projection as a paired query/gate projection and has not yet defined a
safe checkpoint packing rule for that component. This gate records names
only and does not claim shape compatibility.

## Skip Contract

Every `model.visual.*` entry becomes:

```text
scope = "visual"
```

Every `mtp.*` entry becomes:

```text
scope = "mtp"
```

Skipping is explicit and counted. The plan must contain every input key
exactly once across `loads` and `skips`.

For the verified Qwen3.5-2B index the expected result is:

```text
loads: 320
skips: 312
visual skips: 297
mtp skips: 15
shards: ("model.safetensors-00001-of-00001.safetensors",)
```

## Failure Atomicity

The factory builds local lists, validates full coverage and uniqueness, then
publishes frozen tuples. Any error returns no partial plan.

The function has no model argument and no callback, so failure cannot mutate:

- model parameters or buffers;
- the hybrid state pool;
- Scheduler or Engine state;
- filesystem contents.

## Testing

Add `tools/test_qwen35_checkpoint_weight_name_contract.py` as a
dependency-light executable suite.

The suite will:

1. build a small synthetic interleaved topology and verify exact mappings;
2. verify gate/up packed slots while Q/K/V remain separate;
3. verify explicit visual and MTP skips plus total coverage;
4. reject missing, extra, duplicate-target, cross-type, and out-of-range text
   names;
5. reject unknown top-level scopes and unsafe shard paths;
6. reject untied embeddings and malformed config/index values;
7. parse the SHA256-verified official Qwen3.5-2B index fixture generated from
   the exact source-name inventory and assert `320 + 297 + 15 = 632`;
8. monkeypatch tensor-loading entry points to prove the contract does not call
   safetensors or `torch.load`;
9. run under Python 3.9 and Python 3.12;
10. run the existing Qwen3.5 root/owner/factory regressions and
    `git diff --check`.

The official 64 KB index is evidence used to derive the grammar, but it is not
copied into the repository. The test fixture generates the exact expected
grammar from the fixed 24-layer topology and fixed scope counts, so the new
code remains small and reviewable.

## Allowed Conclusion

After this gate passes, the only new allowed conclusion is:

> TinyLLMForge can validate and classify every weight name in the verified
> Qwen3.5-2B checkpoint index into a complete 320-entry language-model load
> plan and explicit visual/MTP skips without opening weight shards or mutating
> a model.

This gate does not establish:

- tensor shape or dtype compatibility;
- TP shard loading;
- actual checkpoint assignment;
- embedding/LM-head storage aliasing;
- concrete Qwen3.5 component construction;
- production model selection or startup binding;
- token/logit equivalence;
- latency, throughput, cache, memory, compression, or quality benefit.

The immutable Qwen3.5 schema-v2 canonical result remains `NO_GO`.
