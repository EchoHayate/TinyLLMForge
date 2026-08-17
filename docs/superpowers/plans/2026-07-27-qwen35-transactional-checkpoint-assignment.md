# Qwen3.5 Transactional Checkpoint Assignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply a complete in-memory Qwen3.5 checkpoint tensor mapping to the existing two-layer CPU graph at TP=1/2 with exact transforms/loaders and all-or-nothing rollback.

**Architecture:** Add a focused assignment executor after the immutable binding planner. It prevalidates and prepares every operation without mutation, snapshots each unique destination once, executes existing custom loaders or exact direct copies, and restores all destinations if any write fails.

**Tech Stack:** Python 3.12, PyTorch CPU, dataclasses, existing Qwen3.5 checkpoint/binding records, TinyLLMForge TP weight loaders.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; do not use subagents.
- Do not stage, commit, merge, or delete experiment evidence.
- Do not open safetensors files or start local/remote GPU work.
- Do not create or replace the supplied state pool.
- Do not connect ModelRunner, Engine, or Scheduler.
- Keep production `ModelRunner` constructing `Qwen3ForCausalLM`.
- Preserve the Scheduler aligned-state fail-closed guard.
- Do not modify schema-v2 canonical `NO_GO`.
- Do not claim performance/cache/memory/compression/quality benefit.
- This session intentionally omits commit steps.

---

### Task 1: Write Transactional Assignment RED Tests

**Files:**
- Create: `tools/test_qwen35_checkpoint_assignment.py`

**Interfaces:**
- Consumes the existing two-layer fixture and:

```python
assign_qwen35_checkpoint_tensors(
    binding_plan,
    source_tensors,
) -> Qwen35CheckpointAssignmentResult
```

- [x] **Step 1: Reuse the complete 27-entry fixture**

Load `tools/test_qwen35_checkpoint_target_binding.py` as a helper module and
reuse `_fixture()`, `_tensor_plan()`, and the binding planner. Create
deterministic source tensors keyed by each
`binding.load.weight.source.name`.

- [x] **Step 2: Add exact TP=1/2 success assertions**

For every rank, initialize destinations to sentinels, assign all 27 bindings,
and assert exact values for:

```text
embedding axis-0 shard
replicated norms
merged gate/up packed slices
row-parallel axis-1 shards
segmented Q/K/V local concatenation
ordinary column shards
head-paired q projection rows
convolution squeeze plus axis-0 shard
F32 A_log/norm and BF16 compute buffers
```

Assert tied embedding/LM-head identity, unchanged sources/pool/registrations,
and result counts.

- [x] **Step 3: Add prevalidation failure matrix**

Cover wrong plan type, non-mapping input, missing/extra/non-tensor source,
wrong shape/dtype/device, unknown transform, invalid loader kind, missing
custom loader, and meta destination. Snapshot destinations before every case
and assert no mutation.

- [x] **Step 4: Add injected mid-transaction failure**

Replace a later destination's loader with a callable that raises after earlier
operations have written. Assert all unique destinations are restored exactly,
source tensors remain unchanged, aliases/registrations remain unchanged, and
the raised error identifies the source and target.

- [x] **Step 5: Run RED**

Run:

```bash
/opt/homebrew/bin/python3.12 \
  tools/test_qwen35_checkpoint_assignment.py
```

Expected:

```text
ModuleNotFoundError:
No module named 'tinyvllm.models.qwen35_checkpoint_assignment'
```

### Task 2: Implement Prevalidation and Operation Preparation

**Files:**
- Create: `tinyvllm/models/qwen35_checkpoint_assignment.py`

**Interfaces:**
- Produces:

```python
Qwen35CheckpointAssignmentResult
assign_qwen35_checkpoint_tensors(...)
```

- [x] **Step 1: Add frozen result and private operation records**

Store the binding, original source tensor, transformed tensor, and optional
direct-buffer local tensor. Do not clone source tensors.

- [x] **Step 2: Validate exact source coverage and metadata**

Require exact key equality, CPU tensor values, metadata dtype/shape, CPU
non-meta destinations, and unchanged binding TP context.

- [x] **Step 3: Apply supported transforms**

Implement:

```python
identity
squeeze_conv_channel -> source.squeeze(1)
```

Validate transformed shape before publishing an operation.

- [x] **Step 4: Prepare loader-specific contracts**

For custom loaders, validate callable and packed-slot rules. For default copy,
require transformed shape equal destination. For direct buffers, derive
replicated or rank-local axis-0 tensor and require exact `local_shape`.

- [x] **Step 5: Run tests to confirm failures move past missing module**

Run the focused test and confirm only unimplemented transaction behavior
remains.

### Task 3: Implement Transaction and Rollback

**Files:**
- Modify: `tinyvllm/models/qwen35_checkpoint_assignment.py`

**Interfaces:**
- Consumes the prepared immutable operation tuple.
- Produces all-or-nothing destination assignment.

- [x] **Step 1: Snapshot unique destinations**

Clone each distinct destination object once, preserving packed/shared
destination semantics.

- [x] **Step 2: Execute each loader kind**

Under `torch.no_grad()`:

```text
custom -> loader(destination, transformed[, packed_slot])
default -> destination.copy_(transformed)
buffer -> destination.copy_(local_tensor)
```

- [x] **Step 3: Restore all destinations on failure**

Attempt every restore, then raise contextual assignment failure or rollback
failure with exception chaining. Never return partial success.

- [x] **Step 4: Return exact counts on success**

Return binding count, unique destination count, and source count.

- [x] **Step 5: Run GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 \
  tools/test_qwen35_checkpoint_assignment.py
```

Expected:

```text
qwen35 checkpoint assignment tests passed
```

### Task 4: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-07-27-qwen35-transactional-checkpoint-assignment.md`

**Interfaces:**
- Produces fresh evidence and the next explicit TODO.

- [x] **Step 1: Run focused regressions**

Run assignment, target-binding, metadata, TP primitive, component factory,
real binding, root assembly, transactional root, and native-owner tests.

- [x] **Step 2: Run focused compilation and static guards**

Run Python 3.12 `py_compile`, `git diff --check`, staged-file count,
ModelRunner constructor check, Scheduler guard check, and Engine reference
check.

- [x] **Step 3: Complete plan and append unique EOF handoff**

Record RED/GREEN, TP=1/2 exact assignment evidence, rollback evidence,
limitations, and next gate:

```text
bounded safetensors reader/streaming into the transactional executor,
still CPU-only and production-unwired
```

Verify the canonical heading is unique and the final `##` at true EOF.
