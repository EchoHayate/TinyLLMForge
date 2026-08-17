# Qwen3.5 Linear-Attention Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect Qwen3.5 projections, causal convolution, gated-delta recurrence, gated RMSNorm, and output projection with side-effect-free candidate state updates.

**Architecture:** Add a CPU shell that reuses the proven pure PyTorch GDN primitives, injects projection/output modules, validates exact local shapes, repeats Q/K heads when required, and returns candidate states only after complete forward success.

**Tech Stack:** RL Python 3.9, PyTorch, standalone dependency-light tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not commit or stage changes.
- Do not start local or remote GPU processes.
- Do not modify immutable schema-v2 canonical evidence.
- Do not add model selection, checkpoint traversal, pool writes, or runtime integration.
- Preserve all untracked `experiments/` evidence.

---

### Task 1: Gated RMSNorm Primitive

**Files:**
- Modify: `tinyvllm/layers/gated_delta.py`
- Modify: `tools/test_qwen35_gated_delta_reference.py`

- [x] **Step 1: Write gated RMSNorm RED tests**

Cover the official norm-before-SiLU-gate formula, FP32 accumulation, BF16
return dtype, non-mutation, and shape/dtype/device failures.

- [x] **Step 2: Confirm missing-function RED**

Run the GDN reference suite and expect missing
`qwen35_gated_rmsnorm`.

- [x] **Step 3: Implement the minimal primitive**

Add:

```python
def qwen35_gated_rmsnorm(
    core: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor
```

- [x] **Step 4: Confirm GDN primitive GREEN**

Expected:

```text
qwen35 gated delta reference tests passed
```

### Task 2: Linear-Attention Shell

**Files:**
- Create: `tinyvllm/layers/qwen35_linear_attention.py`
- Create: `tools/test_qwen35_linear_attention_shell.py`

- [x] **Step 1: Write operation-order and independent oracle tests**

Use asymmetric dimensions and deterministic projections. Independently
reproduce convolution, head repeat, recurrent update, gated RMSNorm, and
output projection.

- [x] **Step 2: Confirm missing-module RED**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_qwen35_linear_attention_shell.py
```

Expected missing module.

- [x] **Step 3: Implement constructor, forward, and boundary validation**

Reuse:

```text
qwen35_causal_depthwise_conv
qwen35_gated_delta_recurrent
qwen35_gated_rmsnorm
```

Return `(output, candidate_convolution_state, candidate_recurrent_state)`.

- [x] **Step 4: Add continuation, BF16, transaction, and failure tests**

Prove split continuation, unchanged input states, output-projection failure
without state mutation, and all shape/dtype/device boundaries.

- [x] **Step 5: Confirm shell GREEN**

Expected:

```text
qwen35 linear attention shell tests passed
```

### Task 3: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-07-26-qwen35-linear-attention-shell.md`

- [x] **Step 1: Run complete Qwen3.5 CPU correctness suite**

Include linear shell, GDN primitive, MLP reuse, decoder/full-attention shells,
projection, MRoPE, norm/query-gate, segmented loader, hybrid state/runtime,
and ModelRunner tests.

- [x] **Step 2: Run py_compile and diff checks**

Use validated Python 3.9/3.12 interpreters and require `git diff --check`.

- [x] **Step 3: Record state transaction and claim boundaries**

Document source formula/SHA, RED/GREEN, head-repeat/orientation coverage,
candidate-state semantics, commands, and remaining pool/runtime/GPU gates.

