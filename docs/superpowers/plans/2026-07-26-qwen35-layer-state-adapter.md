# Qwen3.5 Layer-State Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add generation-safe clone-based gather and rollback-protected dual-state commit for one Qwen3.5 linear-attention layer.

**Architecture:** A focused adapter wraps `HybridStateTensorPool`, validates both layer components and a lease, and rolls back both pool rows if either copy fails.

**Tech Stack:** RL Python 3.9, PyTorch CPU.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not commit or stage.
- No GPU or ModelRunner integration.
- Preserve immutable schema-v2 evidence and untracked experiments.

---

### Task 1: Adapter TDD

**Files:**
- Create: `tinyvllm/engine/qwen35_layer_state.py`
- Create: `tools/test_qwen35_layer_state_adapter.py`

- [x] Write gather/commit, clone-isolation, validation, stale-lease, and
  rollback tests.
- [x] Confirm missing-module RED.
- [x] Implement minimal adapter with `_copy_component` seam for deterministic
  second-copy failure injection.
- [x] Confirm adapter GREEN.

### Task 2: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

- [x] Run complete Qwen3.5 CPU correctness suite.
- [x] Run py_compile and `git diff --check`.
- [x] Record transactional evidence and remaining ModelRunner/GPU boundaries.

