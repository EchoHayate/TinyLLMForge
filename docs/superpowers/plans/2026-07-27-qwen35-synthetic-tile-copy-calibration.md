# Qwen3.5 Synthetic Tile-Copy Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a correctness-checked CPU safetensors tile-copy microbenchmark and persist JSON evidence for tile-policy calibration.

**Architecture:** Add one standalone benchmark tool with a callable pure-result API and CLI. Unit tests validate input handling, full-copy correctness, call/byte accounting, schema, and JSON persistence without asserting unstable timing relationships. Run the default 64 MiB matrix separately and interpret it conservatively.

**Tech Stack:** Python 3.12, PyTorch CPU, safetensors 0.7.0, tempfile, JSON, statistics, platform.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; do not use subagents.
- Do not stage, commit, merge, or delete existing experiment evidence.
- Generate only temporary synthetic payloads; do not open real checkpoints.
- Do not start local or remote GPU work.
- Do not connect ModelRunner, Engine, or Scheduler.
- Do not assert timing order in unit tests.
- Do not claim model-load or inference performance.

---

### Task 1: Add Benchmark Harness RED Tests

**Files:**
- Create: `tools/test_qwen35_synthetic_tile_copy_calibration.py`

- [x] **Step 1: Assert tiny exact matrix schema and accounting**

- [x] **Step 2: Assert full destination checksum and final short tile**

- [x] **Step 3: Assert invalid input matrix**

- [x] **Step 4: Assert CLI JSON persistence**

- [x] **Step 5: Run RED**

Expected missing benchmark module/file.

### Task 2: Implement Calibration Harness

**Files:**
- Create: `tools/benchmark_qwen35_safetensors_tile_copy.py`

- [x] **Step 1: Validate inputs and build deterministic BF16 tensor**

- [x] **Step 2: Implement symmetric baseline and tiled timed repeats**

- [x] **Step 3: Validate full checksum outside timing**

- [x] **Step 4: Build stable JSON-serializable schema and environment record**

- [x] **Step 5: Implement CLI and atomic JSON write**

- [x] **Step 6: Run GREEN**

Expected:

```text
qwen35 synthetic tile copy calibration tests passed
```

### Task 3: Run Default Calibration and Persist Evidence

**Files:**
- Create: `experiments/qwen35_hybrid_state/20260727-synthetic-tile-copy-calibration.json`

- [x] **Step 1: Run 64 MiB default matrix**

- [x] **Step 2: Validate artifact schema and checksums**

- [x] **Step 3: Summarize diminishing-return ratios without overclaiming**

### Task 4: Regression and Handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-07-27-qwen35-synthetic-tile-copy-calibration.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] **Step 1: Run harness tests and py_compile**

- [x] **Step 2: Verify no real-path/GPU/production references**

- [x] **Step 3: Run diff/staged checks**

- [x] **Step 4: Complete plan and append unique EOF canonical handoff**

