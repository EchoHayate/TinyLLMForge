# Qwen3.5 Coalesced Tile Budget Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select the smallest bounded Qwen3.5 tile budget satisfying an explicit tile-count cap and load a fresh temporary-shard candidate with that exact plan and one factory invocation.

**Architecture:** Add a pure deterministic power-of-two budget selector over the completed tile planner. Refactor the tiled loader around an internal exact-plan loading function, then add a policy-driven public wrapper that constructs one candidate, selects one plan, and loads it without rebuilding.

**Tech Stack:** Python 3.12, PyTorch CPU, safetensors 0.7.0, dataclasses.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; do not use subagents.
- Do not stage, commit, merge, or delete experiment evidence.
- Do not open the real checkpoint payload.
- Use only existing real metadata and temporary payload shards.
- Do not start local or remote GPU work.
- Do not connect generic loader, ModelRunner, Engine, or Scheduler.
- Preserve production `Qwen3ForCausalLM` and Scheduler aligned-state guard.
- Preserve supplied state-pool ownership and schema-v2 canonical `NO_GO`.
- Do not claim runtime/cache/memory/compression/quality benefit.

---

### Task 1: Add Pure Tile-Budget Policy RED Tests

**Files:**
- Create: `tools/test_qwen35_checkpoint_tile_policy.py`

**Interfaces:**
- Produces requirements for evaluation/decision records and
  `select_qwen35_checkpoint_tile_budget()`.

- [x] **Step 1: Assert smallest satisfying power-of-two candidate**

Use the two-layer TP=1/2 fixture. Independently enumerate candidate plans and
assert the decision selects the first plan within the tile-count cap.

- [x] **Step 2: Assert exact final non-power-of-two cap**

Choose a cap between adjacent powers of two where only the exact final cap
satisfies the count constraint.

- [x] **Step 3: Add input/failure/determinism matrix**

Cover invalid plan/caps, cap below minimum feasible unit, no satisfying
candidate, repeated identical decisions, immutable destinations, and ordered
evaluation records.

- [x] **Step 4: Run policy RED**

Expected missing module:

```text
tinyvllm.models.qwen35_checkpoint_tile_policy
```

### Task 2: Implement Pure Tile-Budget Selection

**Files:**
- Create: `tinyvllm/models/qwen35_checkpoint_tile_policy.py`

**Interfaces:**
- Produces frozen evaluation/decision records and selector.

- [x] **Step 1: Validate exact plan and positive caps**

- [x] **Step 2: Find the minimum feasible byte unit**

Use fail-closed planner probes and exact error handling; do not inspect files.

- [x] **Step 3: Generate deterministic candidate budgets**

Generate powers of two up to the cap and append an exact non-power-of-two cap
when needed.

- [x] **Step 4: Evaluate and return the first satisfying plan**

Record every attempted valid plan. Reject with final count/caps when none
satisfies.

- [x] **Step 5: Run policy GREEN**

Expected:

```text
qwen35 checkpoint tile policy tests passed
```

### Task 3: Add Real 320-Entry Policy Gate

**Files:**
- Create: `tools/test_qwen35_real_checkpoint_tile_policy.py`

**Interfaces:**
- Consumes real metadata/meta binding helper and pure selector.

- [x] **Step 1: Assert 16 MiB/512 decisions**

Require TP=1 `16 MiB / 488` tiles and TP=2 `8 MiB / 488` tiles for every
rank, proving selection is per exact rank plan rather than globally aligned.

- [x] **Step 2: Assert 8 MiB split behavior**

Require TP=1 failure and TP=2 selection with `488` tiles.

- [x] **Step 3: Guard every real safetensors payload open**

- [x] **Step 4: Run real policy GREEN**

Expected:

```text
qwen35 real checkpoint tile policy tests passed
```

### Task 4: Add Policy-Driven Loader RED Tests

**Files:**
- Create: `tools/test_qwen35_policy_tiled_checkpoint_loading.py`

**Interfaces:**
- Produces requirements for policy-loaded candidate and wrapper API.

- [x] **Step 1: Assert one factory invocation and exact plan identity**

- [x] **Step 2: Assert TP=1/2 exact destination values**

- [x] **Step 3: Preserve get-slice-only/custom-loader/one-tile properties**

- [x] **Step 4: Assert policy failure before shard open**

- [x] **Step 5: Assert load failure closes handles and discards candidate**

- [x] **Step 6: Run loader RED**

Expected missing policy-driven API.

### Task 5: Refactor and Implement Policy-Driven Loading

**Files:**
- Modify: `tinyvllm/models/qwen35_checkpoint_tiled_loading.py`

**Interfaces:**
- Produces `Qwen35PolicyTiledLoadedCheckpointCandidate` and
  `load_qwen35_fresh_checkpoint_candidate_with_tile_policy()`.

- [x] **Step 1: Extract exact-plan internal loader**

Preserve all existing public tiled loader behavior and tests.

- [x] **Step 2: Implement one-factory policy wrapper**

Validate/invoke factory once, select the decision, pass its exact tile plan to
the internal loader, and return nested result plus decision.

- [x] **Step 3: Run policy loader and prior tiled loader GREEN**

### Task 6: Regression and Handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-07-27-qwen35-coalesced-tile-budget-policy.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] **Step 1: Run policy, real policy, policy loader, and tiled regressions**

- [x] **Step 2: Run prior checkpoint/graph/root/owner regressions**

- [x] **Step 3: Run py_compile, production guards, diff check, and staged check**

- [x] **Step 4: Complete plan and append unique EOF canonical handoff**

