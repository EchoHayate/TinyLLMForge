# Qwen3.5 Packed Full-Attention Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent cross-request attention by applying one full-attention decoder layer separately to each explicit packed request segment.

**Architecture:** Validate packed metadata, slice hidden and position tensors by token counts, call the existing full-attention decoder shell per request, and concatenate outputs.

**Tech Stack:** RL Python 3.9, PyTorch CPU.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not commit or stage.
- No GPU, KV-cache runtime, scheduler, ModelRunner, or checkpoint loading.
- Preserve schema-v2 evidence and untracked experiments.
- Do not claim performance, memory, compression, quality, or native support.

---

### Task 1: RED

**Files:**
- Create: `tools/test_qwen35_packed_full_decoder_layer.py`
- Create after RED: `tinyvllm/layers/qwen35_packed_full_decoder_layer.py`

- [x] Build `(2, 1, 3)` packed full-attention fixture.
- [x] Test exact oracle, call lengths, and request isolation.
- [x] Test 1D/1-row/3-row position slicing.
- [x] Test BF16 and non-contiguous hidden input.
- [x] Test metadata, constructor, and later-request failures.
- [x] Confirm missing-module RED.

### Task 2: GREEN

**Files:**
- Create: `tinyvllm/layers/qwen35_packed_full_decoder_layer.py`

- [x] Validate metadata before layer execution.
- [x] Slice positions and hidden states per request.
- [x] Concatenate only after all requests succeed.
- [x] Confirm focused GREEN.

### Task 3: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

- [x] Run complete Qwen3.5/hybrid-state regression.
- [x] Run compile and `git diff --check`.
- [x] Mark plan complete and record boundaries.
