# Qwen3.5 Checkpoint Tensor-Metadata Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the exact dtype, shape, byte size, and contiguous safetensors payload layout for the verified Qwen3.5 checkpoint and attach immutable metadata plus lossless transforms to the existing 320-entry language-model weight plan without reading tensor payloads.

**Architecture:** Extend the dependency-free checkpoint planner with frozen tensor-metadata records and a second pure factory. The factory reuses the completed name plan, derives expected full-source shapes from config, validates all shard headers and payload intervals, then publishes a source-sorted tensor plan only after complete coverage succeeds.

**Tech Stack:** Python 3.9/3.12, standard-library dataclasses, `collections.abc.Mapping`, integer byte arithmetic, dependency-light executable tests.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, merge, or delete untracked experiment evidence.
- Do not call `safe_open()`, `get_tensor()`, `torch.load()`, or read safetensors payload bytes.
- Do not allocate torch tensors, start GPU work, or run remote compute.
- Do not modify the current linear-attention shell dtype contract in this gate.
- Do not integrate with `tinyvllm/utils/loader.py`, `ModelRunner`, Engine, or Scheduler.
- Keep production `ModelRunner` fixed to `Qwen3ForCausalLM`.
- Preserve the immutable Qwen3.5 schema-v2 canonical `NO_GO`.
- Treat real F32 `A_log` and linear norm metadata as authoritative.
- Record but do not apply the convolution channel-squeeze transform.
- This session intentionally omits all commit steps.

---

### Task 1: Extend the Test Suite and Observe RED

**Files:**
- Modify: `tools/test_qwen35_checkpoint_weight_name_contract.py`

**Interfaces:**
- Consumes the existing weight-plan API.
- Produces test expectations for:

```python
build_qwen35_checkpoint_tensor_plan(
    hf_config,
    index_payload: Mapping[str, object],
    shard_headers: Mapping[str, Mapping[str, object]],
) -> Qwen35CheckpointTensorPlan
```

- [x] **Step 1: Add a complete synthetic config**

Extend `_config()` with:

```python
dtype="bfloat16"
hidden_size=8
intermediate_size=12
vocab_size=32
linear_num_key_heads=2
linear_num_value_heads=2
linear_key_head_dim=3
linear_value_head_dim=4
linear_conv_kernel_dim=5
num_attention_heads=2
num_key_value_heads=1
head_dim=4
```

Keep asymmetric key/value head dimensions so shape formulas are observable.

- [x] **Step 2: Add exact metadata generation**

Create a test-only helper that derives expected source metadata from the same
public config contract, assigns contiguous offsets in source-name order, and
returns:

```python
{
    SHARD: {
        source_name: {
            "dtype": "BF16" or "F32",
            "shape": [...],
            "data_offsets": [start, end],
        },
    },
}
```

The helper must generate valid visual/MTP metadata too so full index/header
coverage is exercised.

- [x] **Step 3: Test exact metadata and transforms**

Assert:

```text
320-style load count follows the synthetic topology
all loads retain the original weight-plan record
ordinary weights use BF16
A_log and linear norm use F32
conv1d shape is [conv_width, 1, kernel]
conv1d transform is squeeze_conv_channel
all other transforms are identity
payload_bytes equals final contiguous offset
```

- [x] **Step 4: Test shape/dtype failures**

For representative root, MLP, linear-attention, full-attention, `A_log`,
linear norm, and convolution entries, independently mutate shape or dtype and
assert stable fail-closed errors.

- [x] **Step 5: Test structural header failures**

Independently reject:

```text
missing planned shard header
unknown shard header
missing source metadata
extra source metadata
non-mapping metadata entry
unknown dtype
zero/negative/bool shape dimension
malformed offsets
byte count mismatch
overlap
hole
payload total mismatch
```

- [x] **Step 6: Test immutable/read-only behavior**

Patch `builtins.open` to raise, call the tensor planner with parsed mappings,
assert inputs remain unchanged, and assert all new dataclasses are frozen.

- [x] **Step 7: Run RED under Python 3.9 and Python 3.12**

Run:

```bash
/usr/bin/python3 tools/test_qwen35_checkpoint_weight_name_contract.py
/opt/homebrew/bin/python3.12 tools/test_qwen35_checkpoint_weight_name_contract.py
```

Expected: both fail because tensor metadata APIs are absent.

### Task 2: Implement the Tensor-Metadata Planner

**Files:**
- Modify: `tinyvllm/models/qwen35_checkpoint.py`

**Interfaces:**
- Consumes: existing `Qwen35CheckpointWeightPlan`, config, parsed index, and parsed shard headers.
- Produces:

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

- [x] **Step 1: Add frozen public metadata types**

Use standard-library types only. Do not import torch, safetensors, loader,
Engine, or model modules.

- [x] **Step 2: Add strict config field and dtype helpers**

Require all fields from the spec. Map:

```python
{"bfloat16": "BF16", "float32": "F32"}
```

Reject missing, boolean, non-integer, or non-positive dimensions and unknown
config dtypes.

- [x] **Step 3: Derive exact source metadata**

Build:

```python
source_name -> (expected_dtype, expected_shape, transform)
```

for every language-model name generated by the weight-name contract.

- [x] **Step 4: Parse and validate all header records**

Validate shard-header set, exact source coverage, metadata mapping fields,
supported dtypes, positive shapes, ordered offsets, exact byte count, and
per-shard contiguous intervals starting at zero.

- [x] **Step 5: Validate global payload bytes**

Require:

```python
index_payload["metadata"]["total_size"]
```

to be a non-negative exact integer equal to the sum of final per-shard
offsets.

- [x] **Step 6: Attach metadata atomically**

Validate all 320 language entries against expected dtype/shape. Return
source-sorted frozen loads, reuse the original explicit skips, and publish
only after all validation succeeds.

- [x] **Step 7: Run GREEN under both interpreters**

Run:

```bash
/usr/bin/python3 tools/test_qwen35_checkpoint_weight_name_contract.py
/opt/homebrew/bin/python3.12 tools/test_qwen35_checkpoint_weight_name_contract.py
```

Expected:

```text
qwen35 checkpoint weight-name contract tests passed
```

### Task 3: Validate the Real Header and Integration Boundaries

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-07-27-qwen35-checkpoint-tensor-metadata-contract.md`

**Interfaces:**
- Consumes the completed tensor planner and temporary real header evidence.
- Produces durable verification evidence and checked plan boxes.

- [x] **Step 1: Run the real 632-entry header through the planner**

Use the already downloaded, bounded temporary files:

```text
/tmp/qwen35-2b-15852e8-config.json
/tmp/qwen35-2b-15852e8-model.safetensors.index.json
/tmp/qwen35-safetensors-header.json
```

Assert:

```text
loads=320
skips=312
payload_bytes=4548144832
BF16 language loads=284
F32 language loads=36
squeeze_conv_channel transforms=18
```

- [x] **Step 2: Run focused regressions**

Run under Python 3.12:

```text
tools/test_qwen35_root_model_assembly_factory.py
tools/test_qwen35_transactional_root_causal_lm.py
tools/test_qwen35_native_model_owner_binding.py
tools/test_qwen35_linear_attention_shell.py
tools/test_qwen35_full_attention_shell.py
```

- [x] **Step 3: Run static boundary checks**

Run:

```bash
/usr/bin/python3 -m py_compile \
  tinyvllm/models/qwen35_checkpoint.py \
  tools/test_qwen35_checkpoint_weight_name_contract.py
/opt/homebrew/bin/python3.12 -m py_compile \
  tinyvllm/models/qwen35_checkpoint.py \
  tools/test_qwen35_checkpoint_weight_name_contract.py
rg -n 'safe_open|get_tensor|torch\\.load|load_model|cuda|distributed' \
  tinyvllm/models/qwen35_checkpoint.py
git diff --check
git diff --cached --name-only
```

Expected: compilation succeeds, forbidden references are zero, diff check
passes, and no file is staged.

- [x] **Step 4: Verify production wiring remains absent**

Assert zero tensor-planner references in Engine and the generic loader; retain
one production `Qwen3ForCausalLM` constructor and one Scheduler aligned-state
guard.

- [x] **Step 5: Update handoff and mark the plan complete**

Record:

- bounded header byte ranges;
- exact BF16/F32 and shape findings;
- the current shell mixed-dtype incompatibility;
- RED/GREEN evidence;
- real-header and regression evidence;
- allowed conclusion and remaining gates;
- unchanged schema-v2 canonical `NO_GO`.

Rerun all Task 3 commands after documentation changes.
