# Qwen3.5 Weighted Tile-Policy Estimator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a deterministic static estimator that combines real Qwen3.5 per-kind tile distributions with synthetic per-kind calibration fits.

**Architecture:** Fit one non-negative auditable intercept/slope model per tile grammar from the persisted five-binding synthetic artifact. Apply those models to real payload-free TP=1/2 tile plans, report per-kind contributions and Pareto trade-offs, and persist JSON without changing production policy.

**Tech Stack:** Python 3.12, standard-library JSON/statistics/math, existing Qwen3.5 real metadata/meta binding fixtures, existing tile planner.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; do not use subagents.
- Do not stage, commit, merge, or delete experiment evidence.
- Do not open real safetensors payloads.
- Do not start local or remote GPU work.
- Do not modify production tile-policy selection or runtime wiring.
- Treat proxy seconds as model scores, not real-load latency.
- Do not claim inference, cache, memory, compression, accuracy, or quality improvement.

---

### Task 1: Add Estimator RED Tests

**Files:**
- Create: `tools/test_qwen35_weighted_tile_policy_estimator.py`

- [x] **Step 1: Assert exact per-kind OLS fit contract**

- [x] **Step 2: Assert contribution math and Pareto dominance**

- [x] **Step 3: Assert malformed calibration rejection**

- [x] **Step 4: Assert CLI artifact persistence with payload-open guard**

- [x] **Step 5: Run RED for missing estimator module**

### Task 2: Implement Pure Calibration and Estimator

**Files:**
- Create: `tools/estimate_qwen35_weighted_tile_policy.py`

- [x] **Step 1: Parse and validate calibration artifact**

- [x] **Step 2: Fit per-kind intercept/slope/residual records**

- [x] **Step 3: Aggregate exact real-plan per-kind distributions**

- [x] **Step 4: Compute proxy contributions, deltas, and Pareto frontier**

- [x] **Step 5: Add CLI and atomic JSON write**

- [x] **Step 6: Run GREEN**

Expected:

```text
qwen35 weighted tile policy estimator tests passed
```

### Task 3: Run Real Static TP Matrix

**Files:**
- Create: `experiments/qwen35_hybrid_state/20260727-weighted-tile-policy-estimator.json`

- [x] **Step 1: Run TP=1/2 ranks over 4/8/16/32 MiB**

- [x] **Step 2: Independently verify per-kind totals and zero payload opens**

- [x] **Step 3: Summarize 8->16 and 16->32 proxy/peak trade-offs**

### Task 4: Regression and Handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-07-27-qwen35-weighted-tile-policy-estimator.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] **Step 1: Run focused tests and py_compile**

- [x] **Step 2: Verify production policy and runtime remain untouched**

- [x] **Step 3: Run diff/staged/EOF checks**

- [x] **Step 4: Complete plan and append unique EOF canonical handoff**

