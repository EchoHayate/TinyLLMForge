# Qwen3.5 Checkpoint Target-Binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind every immutable Qwen3.5 checkpoint tensor-plan entry to the exact assembled Parameter or Buffer and validate component type, TP-local shape, dtype, packed slice, loader capability, and tied embedding storage without reading payloads or mutating the model.

**Architecture:** Add a dedicated torch-aware binding planner beside the dependency-free checkpoint metadata planner. It maps stable logical targets into the packed root, enforces the already tested TinyLLMForge TP component classes, derives local shapes from source metadata and TP context, and publishes frozen tensor-reference records only after complete validation.

**Tech Stack:** Python 3.12, PyTorch CPU, dataclasses, existing TinyLLMForge TP layers and Qwen3.5 shells.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; do not use subagents.
- Do not stage, commit, merge, or delete untracked experiment evidence.
- Do not start local or remote GPU/checkpoint work.
- Do not open safetensors shards or materialize source tensors.
- Do not call any destination weight loader or mutate model/state storage.
- Keep the dependency-free `qwen35_checkpoint.py` free of torch imports.
- Do not modify the Qwen3.5 schema-v2 canonical `NO_GO`.
- Keep production `ModelRunner` fixed to `Qwen3ForCausalLM`.
- Keep `LLMEngine.step()` and Scheduler admission unconnected.
- Preserve the aligned-state Scheduler guard.
- Do not claim performance, cache, memory, compression, or quality benefit.
- This session intentionally omits all commit steps.

---

### Task 1: Write Target-Binding RED Tests

**Files:**
- Create: `tools/test_qwen35_checkpoint_target_binding.py`

**Interfaces:**
- Consumes `Qwen35CheckpointTensorPlan` and a real assembled
  `Qwen35PackedForCausalLM`.
- Produces expectations for:

```python
build_qwen35_checkpoint_binding_plan(
    model,
    tensor_plan,
    tensor_parallel_size=...,
    tensor_parallel_rank=...,
) -> Qwen35CheckpointBindingPlan
```

- [x] **Step 1: Build a real two-layer TP fixture**

Use one linear-attention and one full-attention decoder layer with hidden size
8, intermediate size 12, vocabulary size 32, two linear key/value heads, two
full query/KV heads, and head dimensions divisible by TP=1/2.

Use the existing concrete TP classes for every load-bearing projection and
the exact Qwen3.5 shell/root classes. Tie `lm_head.weight` storage to
`embed_tokens.weight`.

- [x] **Step 2: Build a synthetic complete tensor plan**

Create metadata entries for every logical target in the two-layer topology:

```text
root 2
shared per layer 5
linear-specific 9
full-specific 6
total 27
```

Use BF16 for compute tensors, F32 for linear `A_log`/norm, and
`squeeze_conv_channel` only for convolution.

- [x] **Step 3: Assert TP=1 and TP=2 positive bindings**

For each rank, assert all 27 entries bind with exact:

```text
destination object
destination_name
Parameter/Buffer kind
custom/default/direct loader kind
local shape
dtype
packed destination slice
```

Assert gate/up share one object with disjoint slices and embedding/lm-head
share storage.

- [x] **Step 4: Assert read-only behavior**

Snapshot every destination value, object id, storage pointer, storage offset,
dtype, device, and pool tensor before planning. Assert complete equality
after success and after representative failures.

- [x] **Step 5: Add fail-closed matrix**

Cover wrong model type, invalid TP context, wrong block type, missing target,
wrong component class, wrong destination kind, wrong shape/dtype, missing
custom loader, mismatched component `tp_size`/`tp_rank`, invalid packed slot,
untied lm head, and duplicate plan entry.

- [x] **Step 6: Run RED**

Run:

```bash
/opt/homebrew/bin/python3.12 tools/test_qwen35_checkpoint_target_binding.py
```

Expected: fail because `tinyvllm/models/qwen35_checkpoint_binding.py` does not
exist.

### Task 2: Implement the Read-Only Binding Planner

**Files:**
- Create: `tinyvllm/models/qwen35_checkpoint_binding.py`

**Interfaces:**
- Produces:

```python
Qwen35CheckpointTensorBinding
Qwen35CheckpointBindingPlan
build_qwen35_checkpoint_binding_plan(...)
```

- [x] **Step 1: Add frozen binding records**

Retain the exact tensor-plan load and destination tensor reference plus
destination path, kind, loader kind, local shape, and optional packed slice.

- [x] **Step 2: Add strict logical target resolution**

Map root and `layers.N` targets into the packed root. Require exact root type,
exact layer count implied by targets, exact block-specific attributes, and
registered Parameter/Buffer identity.

- [x] **Step 3: Validate concrete component selection**

Use exact type checks from the design for embedding, norms, merged/row/
segmented/column/head-paired projections, and linear-attention buffers.

- [x] **Step 4: Derive transformed TP-local shapes**

Apply only the descriptive convolution squeeze to shape metadata, then derive
replicated, axis-0, axis-1, and packed local shapes. Validate exact divisibility
and component TP metadata.

- [x] **Step 5: Validate dtype, loader, packed slices, and alias**

Require exact BF16/F32 destination dtype, custom loaders for TP parameters,
default-copy norms, direct-copy buffers, disjoint gate/up slots, and exact
embedding/lm-head storage alias.

- [x] **Step 6: Publish atomically**

Return source-sorted frozen bindings only after all entries validate. Do not
invoke loaders or mutate any tensor.

- [x] **Step 7: Run GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 tools/test_qwen35_checkpoint_target_binding.py
```

Expected:

```text
qwen35 checkpoint target binding tests passed
```

### Task 3: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-07-27-qwen35-checkpoint-target-binding.md`

- [x] **Step 1: Run focused regressions**

Run the new binding suite plus checkpoint metadata, TP projection/MLP, mixed
linear shell, root assembly, transactional root, and native owner suites.

- [x] **Step 2: Run static boundaries**

Compile the new module/test under Python 3.9 and 3.12. Require:

```text
no safe_open/get_tensor/torch.load/cuda/distributed in binding module
no generic loader/Engine/ModelRunner references
git diff --check passes
staged files remain zero
```

- [x] **Step 3: Update handoff at true EOF**

Record exact binding count, TP=1/2 shape evidence, packed slices, alias proof,
RED/GREEN, non-mutation evidence, regressions, allowed conclusion, and
remaining materialization/production gates.

- [x] **Step 4: Mark all plan boxes complete**

Rerun static checks after documentation changes.

