# Qwen3.5 Concrete Component Factory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construct the verified 24-layer Qwen3.5 checkpoint-bearing graph on meta tensors from config and an existing state pool, then bind the real 320-entry tensor plan read-only at TP=1/2.

**Architecture:** Add a focused config-driven component factory that creates every concrete checkpoint destination while delegating only the runtime full-attention backend. Use a scoped TP construction context and meta tensors to preserve exact types, shapes, dtypes, registrations, and loaders without allocating the 4.5 GB payload; reuse the existing root assembly and binding planners.

**Tech Stack:** Python 3.12, PyTorch 2.12 CPU/meta tensors, dataclasses, existing TinyLLMForge Qwen3.5 shells, TP layers, state pool, checkpoint metadata planner, and target-binding planner.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; do not use subagents.
- Do not stage, commit, merge, or delete untracked experiment evidence.
- Do not start local or remote GPU/checkpoint execution.
- Do not open or read any safetensors payload.
- Do not call destination weight loaders or mutate checkpoint destinations.
- Reuse the supplied `HybridStateTensorPool`; do not create a second state pool.
- Keep production `ModelRunner` constructing `Qwen3ForCausalLM`.
- Keep `LLMEngine.step()` and Scheduler admission unconnected.
- Preserve `RuntimeError("hybrid prefix reuse requires aligned state snapshot")`.
- Do not modify the Qwen3.5 schema-v2 canonical `NO_GO`.
- Do not claim performance, cache, memory, compression, or quality benefit.
- This session intentionally omits all commit steps.

---

### Task 1: Make Tied-Embedding Validation Meta-Safe

**Files:**
- Modify: `tools/test_qwen35_checkpoint_target_binding.py`
- Modify: `tinyvllm/models/qwen35_checkpoint_binding.py`

**Interfaces:**
- Consumes: `build_qwen35_checkpoint_binding_plan(...)`.
- Produces: meta-safe tied embedding validation.

- [x] **Step 1: Add the failing meta-alias test**

Create a helper that moves the two-layer fixture to `meta`, first preserving
the shared embedding `Parameter`, then replacing `lm_head.weight` with an
independent meta `Parameter` of identical shape and dtype.

Assert the shared-object graph binds and the independent-object graph fails
with:

```text
embed_tokens and lm_head must share storage
```

- [x] **Step 2: Run RED**

Run:

```bash
/opt/homebrew/bin/python3.12 tools/test_qwen35_checkpoint_target_binding.py
```

Expected: the independent meta parameters are incorrectly accepted because
both storage pointers are zero.

- [x] **Step 3: Implement the minimal alias rule**

In `_validate_embedding_alias()`:

```python
if embedding.device.type == "meta" or lm_head.device.type == "meta":
    if embedding is not lm_head:
        raise ValueError(
            "embed_tokens and lm_head must share storage"
        )
elif (
    embedding.untyped_storage().data_ptr()
    != lm_head.untyped_storage().data_ptr()
    or embedding.storage_offset() != lm_head.storage_offset()
):
    raise ValueError(
        "embed_tokens and lm_head must share storage"
    )
```

Keep the existing shape/dtype check.

- [x] **Step 4: Run GREEN**

Run the same command. Expected:

```text
qwen35 checkpoint target binding tests passed
```

### Task 2: Write Concrete Component Factory RED Tests

**Files:**
- Create: `tools/test_qwen35_concrete_component_factory.py`

**Interfaces:**
- Consumes:

```python
build_qwen35_concrete_component_assembly(
    hf_config,
    *,
    pool,
    tensor_parallel_size,
    tensor_parallel_rank,
    build_attention_backend,
    parameter_device="meta",
)
```

- Produces construction, topology, dtype, alias, TP, callback, and failure
  requirements for `tinyvllm/models/qwen35_components.py`.

- [x] **Step 1: Add dependency-light imports**

Use the same namespace-stub pattern as the existing target-binding test so
optional flash-attention/Triton imports are not loaded. Import the exact
engine, layer, factory, and packed-root modules required by the new factory.

- [x] **Step 2: Add a two-layer config and supplied state pool**

Use:

```text
hidden_size=8
intermediate_size=12
vocab_size=32
layer_types=(linear_attention, full_attention)
linear key/value heads=2
linear key/value head dims=2
full query/KV heads=2
head_dim=8
conv kernel=3
partial_rotary_factor=0.75
mrope_section=(1, 1, 1)
dtype=bfloat16
hidden_act=silu
tie_word_embeddings=true
```

Create the pool with `build_qwen35_hybrid_state_layout()` and
`HybridStateTensorPool`; snapshot every pool tensor.

- [x] **Step 3: Assert TP=1/2 concrete construction**

For every rank:

- assert exact root, layer, norm, MLP, linear-attention, full-attention, and
  rotary component types;
- assert all checkpoint-bearing tensors are on `meta`;
- assert BF16 compute tensors and F32 stable buffers;
- assert exact TP-local shapes and `tp_size`/`tp_rank`;
- assert `lm_head.weight is embed_tokens.weight`;
- assert backend callback receives `(layer_index, local_q, local_kv, head_dim)`;
- assert assembly reuses the exact supplied pool and adapters reference it;
- assert pool object ids, values, dtype, device, and shape are unchanged.

- [x] **Step 4: Add fail-closed cases**

Cover:

```text
invalid TP size/rank
unsupported dtype
unsupported hidden_act
tie_word_embeddings false
TP indivisibility
missing/malformed rope_parameters
CUDA parameter_device
non-callable backend factory
backend factory returning non-module
pool topology mismatch
```

After every failure, assert the TP construction context was restored and pool
storage remained unchanged.

- [x] **Step 5: Run RED**

Run:

```bash
/opt/homebrew/bin/python3.12 \
  tools/test_qwen35_concrete_component_factory.py
```

Expected:

```text
ModuleNotFoundError:
No module named 'tinyvllm.models.qwen35_components'
```

### Task 3: Implement the Concrete Component Factory

**Files:**
- Create: `tinyvllm/models/qwen35_components.py`

**Interfaces:**
- Produces:

```python
Qwen35ConcreteComponentAssembly
build_qwen35_concrete_component_assembly(...)
```

- [x] **Step 1: Add strict scalar/config helpers**

Implement positive integer, positive finite float, config unwrapping, TP
context, dtype, activation, tied-embedding, and parameter-device validation.
Reject CUDA and accept only `meta` or `cpu`.

- [x] **Step 2: Add scoped TP construction context**

Temporarily override `torch.distributed.get_rank()` and
`torch.distributed.get_world_size()` only while components are constructed.
Restore both callables in `finally`.

- [x] **Step 3: Add the Qwen3.5 MLP**

Create a private `nn.Module` with exact:

```text
gate_up_proj -> MergedColumnParallelLinear
down_proj    -> RowParallelLinear
```

Its forward computes:

```python
gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
return self.down_proj(torch.nn.functional.silu(gate) * up)
```

- [x] **Step 4: Build root and shared layer components**

Inside `with torch.device(parameter_device)` and the TP context, construct:

```text
VocabParallelEmbedding
ParallelLMHead
Qwen35OffsetRMSNorm
Qwen35 MLP
```

Convert checkpoint-bearing compute modules to BF16. Assign:

```python
lm_head.weight = embed_tokens.weight
```

so the same `Parameter` object is registered at both paths.

- [x] **Step 5: Build linear-attention components**

Derive global and local widths from config. Construct all exact projections
and the `Qwen35LinearAttentionShell` with:

```text
conv_weight -> BF16 meta tensor
A_log -> F32 meta tensor
dt_bias -> BF16 meta tensor
norm_weight -> F32 meta tensor
```

- [x] **Step 6: Build full-attention components**

Construct exact paired query, key, value, output, q/k norm, and partial MRoPE
modules. Validate and store the injected backend module without invoking it.

- [x] **Step 7: Delegate topology/state assembly**

Call `assemble_qwen35_packed_model()` with the supplied pool and a decoder
callback returning exact `Qwen35DecoderLayerShell` values. Return the frozen
component assembly record.

- [x] **Step 8: Run GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 \
  tools/test_qwen35_concrete_component_factory.py
```

Expected:

```text
qwen35 concrete component factory tests passed
```

### Task 4: Bind the Real 320-Entry Plan

**Files:**
- Create: `tools/test_qwen35_real_component_binding.py`

**Interfaces:**
- Consumes:

```text
/tmp/qwen35-2b-15852e8-config.json
/tmp/qwen35-2b-15852e8-model.safetensors.index.json
/tmp/qwen35-safetensors-header.json
```

- Consumes the concrete factory, tensor metadata planner, and target-binding
  planner.
- Produces real 24-layer/320-entry read-only evidence at TP=1/2.

- [x] **Step 1: Parse bounded JSON metadata only**

Load the three JSON files and convert nested config mappings to attribute
namespaces. Do not inspect or open any `.safetensors` file.

- [x] **Step 2: Build the real tensor plan**

Run:

```python
tensor_plan = build_qwen35_checkpoint_tensor_plan(
    config,
    index_payload,
    {only_shard_name: shard_header},
)
```

Assert:

```text
loads=320
skips=312
payload_bytes=4548144832
BF16=284
F32=36
squeeze_conv_channel=18
```

- [x] **Step 3: Construct and bind TP=1/2**

For each rank at TP=1 and TP=2:

- build the state layout and supplied CPU pool;
- construct the 24-layer graph on `meta`;
- build the binding plan;
- assert 320 bindings and the exact real topology/counts;
- assert all checkpoint destinations are `meta`;
- assert root embedding/LM-head object identity;
- assert every state-pool tensor remains unchanged.

- [x] **Step 4: Guard against payload/runtime activity**

Temporarily forbid:

```text
builtins.open for *.safetensors
model.run_step
destination weight_loader calls
```

The test must complete using JSON metadata and object inspection only.

- [x] **Step 5: Run the gate**

Run:

```bash
/opt/homebrew/bin/python3.12 \
  tools/test_qwen35_real_component_binding.py
```

Expected:

```text
qwen35 real component binding tests passed
```

### Task 5: Regression, Static Guards, and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-07-27-qwen35-concrete-component-factory.md`

**Interfaces:**
- Produces fresh evidence and the next explicit TODO.

- [x] **Step 1: Run focused Qwen3.5 regressions**

Run:

```bash
for test_file in \
  tools/test_qwen35_concrete_component_factory.py \
  tools/test_qwen35_real_component_binding.py \
  tools/test_qwen35_checkpoint_target_binding.py \
  tools/test_qwen35_checkpoint_weight_name_contract.py \
  tools/test_segmented_column_parallel_linear.py \
  tools/test_qwen35_head_paired_projection.py \
  tools/test_qwen35_mlp_reuse_compatibility.py \
  tools/test_qwen35_linear_attention_shell.py \
  tools/test_qwen35_full_attention_shell.py \
  tools/test_qwen35_root_model_assembly_factory.py \
  tools/test_qwen35_transactional_root_causal_lm.py \
  tools/test_qwen35_native_model_owner_binding.py
do
  /opt/homebrew/bin/python3.12 "$test_file"
done
```

Expected: every test exits zero.

- [x] **Step 2: Run focused compilation**

Run:

```bash
/opt/homebrew/bin/python3.12 -m py_compile \
  tinyvllm/models/qwen35_components.py \
  tinyvllm/models/qwen35_checkpoint_binding.py \
  tools/test_qwen35_concrete_component_factory.py \
  tools/test_qwen35_real_component_binding.py \
  tools/test_qwen35_checkpoint_target_binding.py
```

Run the available Python 3.9 environment for the same files if present.

- [x] **Step 3: Run static runtime-boundary guards**

Verify:

```text
ModelRunner still constructs Qwen3ForCausalLM
Scheduler aligned-state guard remains present
new factory/binding tests contain no CUDA execution
no source loader is invoked
LLMEngine native Qwen3.5 execution remains unconnected
git diff --check passes
staged files count is zero
```

- [x] **Step 4: Update plan checkboxes and append handoff**

Append one unique EOF section to `AGENT_HANDOFF_STATE.md` containing:

- exact files changed;
- meta graph and TP construction contract;
- real 24-layer/320-entry counts;
- RED/GREEN evidence;
- regression/static evidence;
- allowed conclusion and explicit non-claims;
- next gate: source tensor materialization/transform/shard execution and
  transactional assignment, still without production runtime wiring.

Verify the new heading appears exactly once and is at the true EOF.
