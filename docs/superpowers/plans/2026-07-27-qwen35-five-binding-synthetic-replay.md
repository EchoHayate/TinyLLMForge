# Qwen3.5 Five-Binding Synthetic Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a correctness-checked CPU safetensors replay benchmark over all five Qwen3.5 checkpoint tile grammars.

**Architecture:** Construct one exact synthetic binding plan per grammar, use the existing public planner to generate every source/destination slice, replay each plan against a temporary safetensors shard, and validate the complete final destination outside timing. Persist a stable JSON matrix for host-local policy calibration only.

**Tech Stack:** Python 3.12, PyTorch CPU, safetensors 0.7.0, existing Qwen3.5 checkpoint binding/tile planner, tempfile, JSON, statistics.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; do not use subagents.
- Do not stage, commit, merge, or delete existing experiment evidence.
- Generate only temporary synthetic payloads; do not open real checkpoints.
- Do not start local or remote GPU work.
- Do not connect ModelRunner, Engine, Scheduler, or publication.
- Do not assert timing order in unit tests.
- Do not claim model-load or inference performance.

---

### Task 1: Add Five-Binding Replay RED Tests

**Files:**
- Create: `tools/test_qwen35_five_binding_tile_replay.py`

- [x] **Step 1: Assert reduced exact schema and five kinds**

- [x] **Step 2: Assert planner accounting and exact destinations**

- [x] **Step 3: Assert invalid input matrix**

- [x] **Step 4: Assert CLI atomic JSON persistence**

- [x] **Step 5: Run RED for missing benchmark module**

### Task 2: Implement Planner-Driven Replay Harness

**Files:**
- Create: `tools/benchmark_qwen35_five_binding_tile_replay.py`

- [x] **Step 1: Build exact synthetic binding/source/expected cases**

- [x] **Step 2: Build plans only through the public tile planner**

- [x] **Step 3: Replay exact planner slices and validate full destination**

- [x] **Step 4: Build stable environment/configuration/case schema**

- [x] **Step 5: Add CLI and atomic JSON persistence**

- [x] **Step 6: Run GREEN**

Expected:

```text
qwen35 five binding tile replay tests passed
```

### Task 3: Run Default Matrix and Persist Evidence

**Files:**
- Create: `experiments/qwen35_hybrid_state/20260727-five-binding-synthetic-tile-replay.json`

- [x] **Step 1: Run 4/8/16/32 MiB matrix with three repeats**

- [x] **Step 2: Independently validate all case records and exact flags**

- [x] **Step 3: Summarize grammar-sensitive diminishing returns**

### Task 4: Regression and Handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-07-27-qwen35-five-binding-synthetic-replay.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] **Step 1: Run replay tests and focused py_compile**

- [x] **Step 2: Verify temporary-only payload and production/GPU isolation**

- [x] **Step 3: Run diff/staged/EOF checks**

- [x] **Step 4: Complete plan and append unique EOF canonical handoff**

