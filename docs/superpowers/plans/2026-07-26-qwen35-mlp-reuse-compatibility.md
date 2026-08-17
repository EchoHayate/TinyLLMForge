# Qwen3.5 MLP Reuse Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove whether the existing Qwen3 MLP can be reused unchanged for Qwen3.5 at TP=1/2/4.

**Architecture:** Add no production module. A dependency-light test imports the real `Qwen3MLP`, drives its real gate/up and down weight loaders under synthetic TP ranks, sums row-parallel partial outputs, and compares with the official full-tensor SiLU-gated MLP oracle.

**Tech Stack:** Python 3.12, PyTorch, standalone dependency-light test.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not commit or stage changes.
- Do not start local or remote GPU processes.
- Do not modify immutable schema-v2 canonical evidence.
- Do not add model selection, checkpoint traversal, runtime integration, or a duplicate MLP class.
- Preserve all untracked `experiments/` evidence.

---

### Task 1: TP-Aware Existing-MLP Compatibility

**Files:**
- Create: `tools/test_qwen35_mlp_reuse_compatibility.py`

- [x] **Step 1: Write TP=1/2/4 real-loader equivalence test**

Import the real `Qwen3MLP`, load separate global gate/up weights and global
down weight for every synthetic rank, run the real forward, sum rank partial
outputs, and compare with the independent official formula.

- [x] **Step 2: Confirm an intentional RED**

Initially assert an incorrect local fused order `[up, gate]` at TP=2 and
observe a numerical mismatch against the official oracle. Then restore the
official `[gate, up]` expectation before implementation completion.

- [x] **Step 3: Add BF16, mutation, and activation-guard coverage**

Require BF16 output, compare in FP32, prove input non-mutation, and prove the
existing constructor rejects non-SiLU activation.

- [x] **Step 4: Run and confirm GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 \
  tools/test_qwen35_mlp_reuse_compatibility.py
```

Expected:

```text
qwen35 mlp reuse compatibility tests passed
```

### Task 2: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-07-26-qwen35-mlp-reuse-compatibility.md`

- [x] **Step 1: Run the complete Qwen3.5 CPU suite**

Run the MLP compatibility test plus decoder, full-attention, projection,
MRoPE, norm/query-gate, GDN, segmented loader, hybrid state/runtime, and
ModelRunner dependency-light tests.

- [x] **Step 2: Run static checks**

Compile the new test with Python 3.12, compile prior Python 3.9 files with
their validated interpreter, and run `git diff --check`.

- [x] **Step 3: Record reuse conclusion and boundaries**

Document source formula, TP=1/2/4 evidence, RED/GREEN, why no duplicate class
was added, and all remaining checkpoint/quantization/distributed/equivalence
and performance gates.

