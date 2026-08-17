# Qwen3.5 Packed Heterogeneous Layer Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute a packed mixed full/linear Qwen3.5 decoder schedule and atomically commit all linear-layer state after complete success.

**Architecture:** Validate explicit layer-index alignment, gather all linear states once, execute request-isolated full and linear paths in model order, then perform one cross-layer commit.

**Tech Stack:** RL Python 3.9, PyTorch CPU.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not commit or stage.
- No GPU, checkpoint loading, scheduler, ModelRunner, embeddings, final norm, or lm head.
- Preserve schema-v2 evidence and untracked experiments.
- Do not claim performance, memory, compression, quality, or native support.

---

### Task 1: RED

**Files:**
- Create: `tools/test_qwen35_packed_layer_stack.py`
- Create after RED: `tinyvllm/layers/qwen35_packed_layer_stack.py`

- [x] Build linear/full/linear schedule with two adapters and three requests.
- [x] Test exact hidden and state oracles plus call order.
- [x] Test full-attention request isolation.
- [x] Test BF16 and non-contiguous input.
- [x] Test constructor layer/adapter alignment.
- [x] Test later-layer and commit failures leave all state unchanged.
- [x] Confirm missing-module RED.

### Task 2: GREEN

**Files:**
- Create: `tinyvllm/layers/qwen35_packed_layer_stack.py`

- [x] Validate packed metadata and exact linear-index alignment.
- [x] Gather all linear state once.
- [x] Execute full and linear request segments without intermediate commits.
- [x] Commit all linear candidates once after complete stack success.
- [x] Confirm focused GREEN.

### Task 3: Regression and Completion Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

- [x] Run complete Qwen3.5/hybrid-state regression.
- [x] Run compile and `git diff --check`.
- [x] Audit every active-goal deliverable against concrete files and tests.
- [x] Mark plan and goal complete only if no required gate remains.
