# Qwen3.5 Cross-Layer State Transaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commit all packed Qwen3.5 linear-layer state batches atomically across model layers.

**Architecture:** A coordinator validates and snapshots all adapters before copying any layer, then restores every selected row directly if any later layer/component copy fails.

**Tech Stack:** RL Python 3.9, PyTorch CPU.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not commit or stage.
- No heterogeneous model shell, GPU, ModelRunner, checkpoint loader, or KV transaction.
- Preserve schema-v2 evidence and untracked experiments.
- Do not claim performance, memory, compression, quality, or native support.

---

### Task 1: RED

**Files:**
- Create: `tools/test_qwen35_cross_layer_state_transaction.py`
- Create after RED: `tinyvllm/engine/qwen35_state_transaction.py`

- [x] Build two-layer, three-slot real-pool fixture.
- [x] Test ordered gather and clone isolation.
- [x] Test successful cross-layer commit and untouched rows.
- [x] Test complete prevalidation before writes.
- [x] Test later-layer failure rolls back all layers.
- [x] Test constructor pool/layer/type boundaries.
- [x] Confirm missing-module RED.

### Task 2: GREEN

**Files:**
- Create: `tinyvllm/engine/qwen35_state_transaction.py`

- [x] Validate constructor and lease batches.
- [x] Gather every adapter in order.
- [x] Validate all candidates before snapshots/writes.
- [x] Copy deterministically and directly restore all snapshots on failure.
- [x] Confirm focused GREEN.

### Task 3: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

- [x] Run complete Qwen3.5/hybrid-state regression.
- [x] Run compile and `git diff --check`.
- [x] Mark plan complete and record heterogeneous-stack readiness.
