# Qwen3.5 Full-Attention Projection Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove Qwen3.5 head-paired q/query-gate TP loading and the complete dependency-light full-attention operation order.

**Architecture:** Add a constrained contiguous column-parallel projection for official head-major query/gate rows. Add a CPU shell with injected projections, norms, rotary, attention backend, and output projection so tests can observe every boundary without production model or GPU dependencies.

**Tech Stack:** Python 3.12 for `linear.py`, RL Python/PyTorch for dependency-light shell tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not commit or stage changes.
- Do not start local or remote GPU processes.
- Do not modify the immutable schema-v2 canonical evidence.
- Do not add model selection, checkpoint traversal, paged attention, or runtime integration.
- Preserve all untracked `experiments/` evidence.

---

### Task 1: Head-Paired Column Projection

**Files:**
- Modify: `tinyvllm/layers/linear.py`
- Create: `tools/test_qwen35_head_paired_projection.py`

- [x] **Step 1: Write TP=1/2/4 head-coded loader and split tests**

Encode every global row by head id and query/gate half. Assert that every rank
receives complete contiguous head pairs and that reshaping/chunking recovers
the correct local query and gate values.

- [x] **Step 2: Add a segmented-layout counterexample**

Construct the incorrect expected rows for global `[Q_all,gate_all]` segmented
sharding and assert they differ from the official head-paired rank rows at
TP=2 and TP=4.

- [x] **Step 3: Run with Python 3.12 and confirm missing-class RED**

Run:

```bash
/opt/homebrew/bin/python3.12 \
  tools/test_qwen35_head_paired_projection.py
```

Expected: missing `HeadPairedColumnParallelLinear`.

- [x] **Step 4: Implement the minimal subclass and split contract**

Subclass `ColumnParallelLinear`, enforce complete head-pair alignment, retain
its existing contiguous loader/forward, and implement exact per-head split.

- [x] **Step 5: Add constructor and projected-output failure tests**

Cover invalid heads/dimensions, TP divisibility, projected rank, floating
dtype, and exact local output width.

- [x] **Step 6: Run and confirm projection GREEN**

Expected:

```text
qwen35 head paired projection tests passed
```

### Task 2: Dependency-Light Full-Attention Shell

**Files:**
- Create: `tinyvllm/layers/qwen35_full_attention.py`
- Create: `tools/test_qwen35_full_attention_shell.py`

- [x] **Step 1: Write an observable operation-order fixture**

Use deterministic injected modules that record:

```text
q_projection
k_projection
v_projection
q_norm
k_norm
rotary
attention_backend
output_projection
```

The expected numerical oracle must independently reproduce the official
per-head split, q/k transforms, rotary transform, backend result, sigmoid
gate, and output projection.

- [x] **Step 2: Run and confirm missing-module RED**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_qwen35_full_attention_shell.py
```

Expected: missing `tinyvllm/layers/qwen35_full_attention.py`.

- [x] **Step 3: Implement constructor and forward boundary checks**

Implement only the shape-driven shell from the design. Reuse
`qwen35_apply_query_gate`; do not initialize distributed or attention runtime
objects.

- [x] **Step 4: Add BF16, mutation, asymmetric-head, and failure tests**

Cover:

- local query heads different from local KV heads;
- BF16 preservation;
- unchanged hidden states and position ids;
- malformed projection, norm, rotary, backend, and output-projection results.

- [x] **Step 5: Run and confirm shell GREEN**

Expected:

```text
qwen35 full attention shell tests passed
```

### Task 3: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-07-25-qwen35-full-attention-shell.md`

- [x] **Step 1: Run all Qwen3.5 primitive and projection tests**

Run the head-paired projection, full-attention shell, MRoPE, norm/query-gate,
GDN reference, segmented loader, hybrid layout, and runtime bridge suites.

- [x] **Step 2: Run `py_compile` and `git diff --check`**

Compile all newly touched implementation/test files with their validated
interpreters and require `git diff --check` exit zero.

- [x] **Step 3: Record the corrected layout, RED/GREEN, and claim boundary**

Document the official head-major row order, why segmented sharding is wrong
for this tensor, all fresh commands, and the remaining production/model gates.
