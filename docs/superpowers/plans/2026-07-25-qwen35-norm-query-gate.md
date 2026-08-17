# Qwen3.5 Offset RMSNorm and Query-Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement dependency-light CPU primitives for Qwen3.5 offset RMSNorm and full-attention query-output gating.

**Architecture:** A dedicated Qwen3.5 primitive module preserves existing Qwen3 RMSNorm semantics. Both operations validate exact shape/dtype contracts, compute numerically sensitive math in FP32, and return the caller's dtype without mutating inputs.

**Tech Stack:** Python 3, PyTorch CPU tensors, dependency-light script tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not start local or remote GPU processes.
- Do not change existing `RMSNorm` behavior.
- Do not add attention, RoPE, model, or checkpoint support.
- Preserve schema-v2 canonical `NO_GO` and all experiment evidence.

---

### Task 1: Offset RMSNorm

**Files:**
- Create: `tinyvllm/layers/qwen35_primitives.py`
- Create: `tools/test_qwen35_norm_query_gate.py`

- [x] **Step 1: Write official-formula, offset, BF16, and failure tests**
- [x] **Step 2: Run with the RL Python and confirm missing-module RED**
- [x] **Step 3: Implement `Qwen35OffsetRMSNorm` with FP32 math**
- [x] **Step 4: Run and confirm RMSNorm GREEN**

### Task 2: Query-Output Gate

**Files:**
- Modify: `tinyvllm/layers/qwen35_primitives.py`
- Modify: `tools/test_qwen35_norm_query_gate.py`

- [x] **Step 1: Write sigmoid oracle, saturation, mutation, and shape tests**
- [x] **Step 2: Run and confirm missing-function RED**
- [x] **Step 3: Implement `qwen35_apply_query_gate`**
- [x] **Step 4: Run and confirm full GREEN**

### Task 3: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] **Step 1: Run new, segmented-loader, and GDN tests**
- [x] **Step 2: Run `py_compile` and `git diff --check`**
- [x] **Step 3: Record formulas, RED/GREEN evidence, and claim boundary**

