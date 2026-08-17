# Qwen3.5 Packed Stateful Linear Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Map packed request token segments to distinct Qwen3.5 state rows and commit the entire layer-state batch only after every request succeeds.

**Architecture:** A focused wrapper gathers batched state once, executes each explicit token segment with its matching state row, accumulates candidates, and invokes one rollback-protected batched commit.

**Tech Stack:** RL Python 3.9, PyTorch CPU.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not commit or stage.
- No GPU, scheduler, ModelRunner, checkpoint-loader, or kernel optimization.
- Preserve immutable schema-v2 evidence and untracked experiments.
- Do not claim performance, memory, compression, quality, or native support.

---

### Task 1: Packed Wrapper RED

**Files:**
- Create: `tools/test_qwen35_packed_stateful_linear_decoder_layer.py`
- Create after RED: `tinyvllm/layers/qwen35_packed_stateful_decoder_layer.py`

- [x] Build a three-request `(2, 1, 3)` real-pool fixture.
- [x] Test exact packed output and per-request candidate states.
- [x] Test request-order isolation, BF16, and non-contiguous hidden input.
- [x] Test lease/count/position/hidden metadata boundaries.
- [x] Test later-request failures leave the full pool unchanged.
- [x] Test invalid candidate and commit-copy failure leave the pool unchanged.
- [x] Confirm missing-module RED.

### Task 2: Minimal GREEN

**Files:**
- Create: `tinyvllm/layers/qwen35_packed_stateful_decoder_layer.py`

- [x] Validate packed metadata before gather.
- [x] Gather once, execute explicit request segments, and accumulate candidates.
- [x] Commit once only after all request segments complete.
- [x] Confirm focused GREEN.

### Task 3: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

- [x] Run full Qwen3.5/hybrid-state dependency-light regression.
- [x] Run Python 3.9/3.12 compile and `git diff --check`.
- [x] Mark all checkboxes complete from fresh evidence.
- [x] Record proof and remaining runtime/performance boundaries.
