# Qwen3.5 Batched Layer-State Transaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ordered clone-based batched gather and rollback-protected all-batch commit to one Qwen3.5 layer-state adapter.

**Architecture:** Extend the existing adapter rather than creating a parallel abstraction. Validate the complete lease/candidate batch before writes, snapshot all selected rows, use the existing deterministic copy seam, and restore the entire selected batch on any failure.

**Tech Stack:** RL Python 3.9, PyTorch CPU.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not commit or stage.
- Preserve existing single-request adapter behavior.
- No GPU, packed decoder, ModelRunner, or checkpoint-loader integration.
- Preserve immutable schema-v2 evidence and all untracked experiments.
- Do not claim performance, memory, compression, quality, or native-support gains.

---

### Task 1: Batched Adapter RED

**Files:**
- Modify: `tools/test_qwen35_layer_state_adapter.py`
- Modify after RED: `tinyvllm/engine/qwen35_layer_state.py`

**Interfaces:**
- Consumes: `HybridStateLease`, `HybridStateTensorPool`.
- Produces:

```python
def gather_batch(
    self,
    leases: tuple[HybridStateLease, ...],
) -> tuple[torch.Tensor, torch.Tensor]

def commit_batch(
    self,
    leases: tuple[HybridStateLease, ...],
    convolution_states: torch.Tensor,
    recurrent_states: torch.Tensor,
) -> None
```

- [x] Add a three-slot fixture with distinct values and out-of-order leases.
- [x] Test ordered clone-based contiguous gather.
- [x] Test successful two-row commit and untouched non-selected row.
- [x] Test stale later lease and duplicate slots fail before writes.
- [x] Test empty, list-based, and non-lease batches fail closed.
- [x] Test candidate batch-size, shape, dtype, and device validation.
- [x] Test a later copy failure restores every selected row.
- [x] Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_qwen35_layer_state_adapter.py
```

Expected: `AttributeError` because `gather_batch` is absent.

### Task 2: Minimal GREEN

**Files:**
- Modify: `tinyvllm/engine/qwen35_layer_state.py`

- [x] Add shared tuple/lease/duplicate validation without changing single-slot
  behavior.
- [x] Implement ordered clone-based `gather_batch()`.
- [x] Implement full candidate validation before writes.
- [x] Implement snapshot, ordered row/component copies, and all-selected-row
  rollback.
- [x] Run the focused adapter test and require:

```text
qwen35 layer state adapter tests passed
```

### Task 3: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

- [x] Run adapter, stateful decoder, linear-attention, decoder, GDN, MLP,
  full-attention, projection, RoPE, hybrid-state, scheduler/runtime bridge,
  and ModelRunner dependency-light tests.
- [x] Run Python 3.9 and Python 3.12 `py_compile` for their respective files.
- [x] Run `git diff --check`.
- [x] Mark every plan checkbox complete only after fresh evidence.
- [x] Record batched transaction proof and remaining packed-runtime boundaries
  in `AGENT_HANDOFF_STATE.md`.
