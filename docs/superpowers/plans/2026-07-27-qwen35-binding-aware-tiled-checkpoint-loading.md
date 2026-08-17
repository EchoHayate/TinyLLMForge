# Qwen3.5 Binding-Aware Tiled Checkpoint Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build exact TP-local safetensors slice plans for all Qwen3.5 checkpoint bindings and load a fresh unpublished CPU candidate through bounded tiles rather than complete source tensors.

**Architecture:** Add a pure tile planner that classifies each binding into axis-0, axis-1, segmented axis-0, squeezed convolution, or replicated direct-copy regions. Add a fresh-candidate tiled loader that validates the complete plan and all paths before opening shards, calls `get_slice()` only, materializes one bounded tile at a time, copies it directly into its final destination region, and discards failures without rollback.

**Tech Stack:** Python 3.12, PyTorch CPU, safetensors 0.7.0 `PySafeSlice`, pathlib, dataclasses, weak references.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; do not use subagents.
- Do not stage, commit, merge, or delete experiment evidence.
- Do not open the real checkpoint payload.
- Real config/index/header metadata may be read from the existing `/tmp` files.
- Use only temporary small safetensors shards for payload tests.
- Do not start local or remote GPU work.
- Do not connect generic loader, ModelRunner, Engine, or Scheduler.
- Production `ModelRunner` must continue constructing `Qwen3ForCausalLM`.
- Preserve `RuntimeError("hybrid prefix reuse requires aligned state snapshot")`.
- State tensors remain owned only by the model/root owner and supplied pool.
- Do not modify schema-v2 canonical `NO_GO`.
- Do not claim performance/cache/memory/compression/quality benefit.

---

### Task 1: Add Pure Tile Planner RED Tests

**Files:**
- Create: `tools/test_qwen35_checkpoint_tiles.py`

**Interfaces:**
- Consumes `Qwen35CheckpointBindingPlan`.
- Produces requirements for:

```python
Qwen35CheckpointTile
Qwen35CheckpointTilePlan
build_qwen35_checkpoint_tile_plan(...)
```

Each tile records `binding_index`, complete `source_tensor_shape`, and
materialized `tile_shape` so packed bindings and slice-shape validation remain
unambiguous.

- [x] **Step 1: Cover all five tile classes with the two-layer fixture**

Build TP=1/2 plans with a budget that forces multiple tiles. Assert exact
source and destination slices for embedding/column/packed axis-0,
row-parallel axis-1, segmented Q/K/V, squeezed convolution, and replicated
vectors.

- [x] **Step 2: Independently prove destination coverage**

Convert every tile destination slice to element coordinates. Assert every
binding destination region is covered exactly once, allowing only disjoint
gate/up regions on one packed destination.

- [x] **Step 3: Add planner failure matrix**

Cover invalid plan/budget, budget below indivisible row bytes, wrong binding
type, unsupported loader/transform/target, malformed destination slice,
conflicting duplicate source metadata, and mismatched segmented owner sizes.

- [x] **Step 4: Run planner RED**

Run:

```bash
python3.12 tools/test_qwen35_checkpoint_tiles.py
```

Expected missing module:

```text
tinyvllm.models.qwen35_checkpoint_tiles
```

### Task 2: Implement Pure Binding-Aware Tile Planner

**Files:**
- Create: `tinyvllm/models/qwen35_checkpoint_tiles.py`

**Interfaces:**
- Produces frozen tile/plan records and
  `build_qwen35_checkpoint_tile_plan()`.

- [x] **Step 1: Validate exact plan, TP context, budget, and bindings**

Reject malformed contracts before creating any tile. Map metadata dtype to
exact torch dtype and byte width.

- [x] **Step 2: Implement axis-0 and packed destination tiles**

Compute rank-local global row intervals, destination row offsets, row byte
units, and budget-derived row counts.

- [x] **Step 3: Implement segmented axis-0 tiles**

Recover the exact `SegmentedColumnParallelLinear` owner from the bound loader.
Validate global/local segment sizes and emit independent tiles per segment.

- [x] **Step 4: Implement axis-1 tiles**

Select the rank-local source column interval and tile across source/destination
rows.

- [x] **Step 5: Implement convolution and replicated tiles**

Use integer channel selection for convolution's singleton axis and
one-dimensional ranges for replicated vectors.

- [x] **Step 6: Validate complete plan invariants**

Verify positive shapes/bytes, exact tile byte calculation, budget compliance,
binding/source counts, deterministic ordering, and exact destination coverage.

- [x] **Step 7: Run planner GREEN**

Run:

```bash
python3.12 tools/test_qwen35_checkpoint_tiles.py
```

Expected:

```text
qwen35 checkpoint tile planner tests passed
```

### Task 3: Prove All 320 Real Bindings Are Tile-Plan Compatible

**Files:**
- Create: `tools/test_qwen35_real_checkpoint_tiles.py`

**Interfaces:**
- Consumes verified `/tmp` config/index/header metadata, real tensor plan, and
  24-layer meta binding graph.
- Produces static compatibility evidence only.

- [x] **Step 1: Build TP=1/2 real tile plans**

Use `12,288` bytes at TP=1 and `6,144` bytes at TP=2. Cover every rank without
opening the safetensors shard.

- [x] **Step 2: Assert exact class counts**

Require:

```text
axis-0=157
axis-1=48
segmented-axis-0=18
squeeze-axis-0=18
replicated=79
```

- [x] **Step 3: Assert hard budget and complete coverage**

Require 320 bindings, 320 unique sources, every tile within budget, exact
rank-local destination bytes, and no safetensors payload open.

- [x] **Step 4: Assert minimum feasible budget boundary**

Require TP=1 rejection at `12,287` bytes and TP=2 rejection at `6,143` bytes,
with the row-parallel down projection identified as the limiting binding.

- [x] **Step 5: Run real static GREEN**

Expected:

```text
qwen35 real checkpoint tile tests passed
```

### Task 4: Add Fresh Tiled Loader RED Tests

**Files:**
- Create: `tools/test_qwen35_tiled_checkpoint_loading.py`

**Interfaces:**
- Consumes the existing fresh candidate fixture and tile planner.
- Produces requirements for:

```python
Qwen35TiledCheckpointLoadStats
Qwen35TiledLoadedCheckpointCandidate
load_qwen35_fresh_checkpoint_candidate_tiled(...)
```

- [x] **Step 1: Assert TP=1/2 exact values with multiple tiles**

Use temporary two-shard files and a small valid budget. Compare every
destination to independent full-source expectations.

- [x] **Step 2: Prove only `get_slice()` is used**

Wrap shard handles so `get_tensor()` raises, record every `get_slice()` call,
and verify requested source coverage.

- [x] **Step 3: Prove custom loaders never execute**

Replace every callable destination `weight_loader` with a failure sentinel
after binding. Tiled loading must still produce exact values.

- [x] **Step 4: Prove one-tile lifetime**

Weak-reference each materialized tile and force collection before the next
copy. Assert the returned candidate retains no tile tensors.

- [x] **Step 5: Add loader failure matrix**

Cover invalid factory/path, missing source, wrong slice shape metadata,
materialized wrong dtype/shape, and injected destination-copy failure. Assert
balanced handle cleanup, no loaded result, and private-candidate discard.

- [x] **Step 6: Run loader RED**

Expected missing module:

```text
tinyvllm.models.qwen35_checkpoint_tiled_loading
```

### Task 5: Implement Fresh Tiled Candidate Loading

**Files:**
- Create: `tinyvllm/models/qwen35_checkpoint_tiled_loading.py`

**Interfaces:**
- Produces frozen stats/candidate records and
  `load_qwen35_fresh_checkpoint_candidate_tiled()`.

- [x] **Step 1: Validate fresh candidate and build tile plan**

Reuse the completed fresh-candidate ownership rules. Build the complete tile
plan and validate all safe shard paths before opening any file.

- [x] **Step 2: Validate shard slice shape before materialization**

For each source, call `get_slice()` once per shard open and require
`get_shape()` to equal immutable source metadata.

- [x] **Step 3: Materialize and validate one tile**

Index the `PySafeSlice` with the planned source slices. Require exact CPU
dtype, shape, and bytes.

- [x] **Step 4: Copy directly into final destination region**

Under `torch.no_grad()`, validate destination slice shape and copy the tile.
Release the tensor before the next tile.

- [x] **Step 5: Build owner only after all handles close**

Return no source/tile tensors. Report exact binding/source/shard/tile counts,
destination/materialized bytes, and peak tile bytes.

- [x] **Step 6: Run loader GREEN**

Expected:

```text
qwen35 tiled checkpoint loading tests passed
```

### Task 6: Regression, Static Guards, and Handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-07-27-qwen35-binding-aware-tiled-checkpoint-loading.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes all checkpoint/factory/owner gates.
- Produces a unique canonical EOF handoff section.

- [x] **Step 1: Run tile planner/real static/tiled loader tests**

- [x] **Step 2: Run prior streamed, publication, reader, assignment, binding, graph, root, and owner regressions**

- [x] **Step 3: Run Python 3.12 compile and production/static guards**

Verify no real payload path, CUDA, Engine wiring, or custom-loader execution
exists in the tiled gate.

- [x] **Step 4: Run `git diff --check` and verify zero staged files**

- [x] **Step 5: Complete the plan and append unique EOF handoff**

Record exact tile budgets/counts, RED/GREEN evidence, what the hard bound does
and does not prove, and the next conservative gate.

