# Segmented Column-Parallel Linear Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CPU-tested column-parallel linear layer that independently shards logical fused output segments before assembling each TP rank's local parameter.

**Architecture:** `SegmentedColumnParallelLinear` subclasses the existing `ColumnParallelLinear` so forward and quantization behavior remain unchanged. Its parameter loader validates the full source contract, then either independently slices every segment from a fused source or writes one explicitly selected segment into the same rank-local packed layout.

**Tech Stack:** Python 3, PyTorch CPU tensors, existing `tinyvllm.layers.linear` abstractions, dependency-light script tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Use inline execution; do not dispatch subagents.
- Do not start any local or remote GPU model process.
- Do not modify the generic safetensors traversal in this slice.
- Do not add a native Qwen3.5 model or claim checkpoint support.
- Preserve the immutable Qwen3.5 schema-v2 canonical `NO_GO`.
- Preserve all untracked experiment evidence.
- Do not add dependencies.

---

### Task 1: Fused Segmented TP Loading

**Files:**
- Modify: `tinyvllm/layers/linear.py`
- Create: `tools/test_segmented_column_parallel_linear.py`

**Interfaces:**
- Produces:
  - `SegmentedColumnParallelLinear(input_size, output_sizes, bias=False)`;
  - `weight_loader(param, loaded_weight, loaded_segment_id=None)`.

- [x] **Step 1: Write failing TP=1/2/4 fused-source tests**

Load `linear.py` directly with dependency stubs. Construct layers for
synthetic rank/world-size pairs and compare the local parameter against an
explicit per-segment `narrow` oracle. Use unequal segment sizes and row-coded
weights.

- [x] **Step 2: Run and confirm RED**

Run:

```bash
/opt/homebrew/bin/python3.12 \
  tools/test_segmented_column_parallel_linear.py
```

Expected: missing `SegmentedColumnParallelLinear`.

- [x] **Step 3: Implement constructor and fused loader**

Validate segment sizes before calling `ColumnParallelLinear`. Precompute
global and local segment offsets. For a fused source, validate every shape,
dtype, and device condition before copying each rank-local segment.

- [x] **Step 4: Add bias and forward tests**

Prove bias uses the same segmented row selection and inherited forward equals
`F.linear` with the assembled local weight and bias.

- [x] **Step 5: Run and confirm GREEN**

Expected: fused segmented TP tests pass.

### Task 2: Separate Segment Loading and Fail-Closed Contracts

**Files:**
- Modify: `tinyvllm/layers/linear.py`
- Modify: `tools/test_segmented_column_parallel_linear.py`

**Interfaces:**
- Consumes:
  - `SegmentedColumnParallelLinear`;
  - `weight_loader(..., loaded_segment_id=int)`.

- [x] **Step 1: Write failing separate-source equivalence tests**

Load each logical source segment separately in non-monotonic id order and
assert the final weight and bias exactly match fused loading for every rank.

- [x] **Step 2: Run and confirm RED**

Expected: the loader rejects or mishandles `loaded_segment_id`.

- [x] **Step 3: Implement separate-source loading**

Validate the selected id and exact segment source shape, select the TP rank's
rows, and copy only the corresponding local parameter range.

- [x] **Step 4: Add constructor, shape, dtype, device, and atomicity failures**

Cover empty/invalid/non-divisible sizes, invalid ids, wrong input/output
shapes, dtype mismatch, device mismatch when available, and unchanged
destination after failed fused validation.

- [x] **Step 5: Run and confirm GREEN**

Expected:

```text
segmented column parallel linear tests passed
```

### Task 3: Regression Verification and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] **Step 1: Run focused tests**

Run:

```bash
/opt/homebrew/bin/python3.12 \
  tools/test_segmented_column_parallel_linear.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_qwen35_gated_delta_reference.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_qwen35_hybrid_state_layout.py
```

- [x] **Step 2: Run syntax and diff checks**

Run:

```bash
/opt/homebrew/bin/python3.12 -m py_compile \
  tinyvllm/layers/linear.py \
  tools/test_segmented_column_parallel_linear.py
git diff --check
```

- [x] **Step 3: Update handoff with exact claim boundary**

Record the source/destination layouts, TP=1/2/4 evidence, failure contracts,
and explicitly state that no native model, real checkpoint, distributed GPU
execution, correctness gate, or performance result has been established.

