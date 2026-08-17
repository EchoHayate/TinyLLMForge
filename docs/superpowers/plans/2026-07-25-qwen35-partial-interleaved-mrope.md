# Qwen3.5 Partial Interleaved MRoPE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and validate a dependency-light CPU primitive for Qwen3.5 partial, interleaved text MRoPE.

**Architecture:** Add a Qwen3.5-specific rotary module without changing the existing Qwen3 full-head RoPE. The module accepts TinyLLMForge token-major flattened query/key projections, builds official interleaved T/H/W frequencies in FP32, rotates only an explicit head prefix, and preserves the suffix and caller dtype.

**Tech Stack:** Python 3, PyTorch CPU tensors, dependency-light script tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not commit or stage changes.
- Do not start local or remote GPU processes.
- Do not change `tinyvllm/layers/rotary_embedding.py`.
- Do not add attention, model, checkpoint, or runtime integration.
- Preserve schema-v2 canonical `NO_GO` and all experiment evidence.

---

### Task 1: Official Interleaved Frequency and Partial Rotation

**Files:**
- Create: `tinyvllm/layers/qwen35_rotary_embedding.py`
- Create: `tools/test_qwen35_partial_interleaved_mrope.py`

**Interfaces:**
- Produces:

```python
class Qwen35PartialInterleavedRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        base: float,
        mrope_section: tuple[int, int, int],
    ): ...

    def forward(
        self,
        position_ids: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
```

- [x] **Step 1: Write the failing official-formula test**

Use `head_dim=12`, `rotary_dim=8`, and `mrope_section=(2,1,1)`.
Use explicit T/H/W positions and a manual scalar oracle that selects
half-frequency lanes as `T,H,W,T`, duplicates them, rotates the first eight
features, and preserves the final four features.

- [x] **Step 2: Run the focused test and confirm missing-module RED**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_qwen35_partial_interleaved_mrope.py
```

Expected: failure because
`tinyvllm/layers/qwen35_rotary_embedding.py` does not exist.

- [x] **Step 3: Implement constructor validation and FP32 frequencies**

Validate all contracts from the design before registering `inv_freq`.
Require:

```text
sum(mrope_section) == rotary_dim / 2
```

Construct inverse frequencies in FP32 and register them as a non-persistent
buffer.

- [x] **Step 4: Implement token-major partial rotation**

Normalize `[tokens]` position ids by replicating them to `[3,tokens]`.
Build the three frequency planes in FP32, select official interleaved lanes,
duplicate the selected half, and apply split-half rotation to only the first
`rotary_dim` features of every query/key head.

- [x] **Step 5: Run and confirm official-formula GREEN**

Run the focused test and require:

```text
qwen35 partial interleaved mrope tests passed
```

### Task 2: Dtype, Text, Shape, and Failure Matrix

**Files:**
- Modify: `tools/test_qwen35_partial_interleaved_mrope.py`
- Modify: `tinyvllm/layers/qwen35_rotary_embedding.py`

- [x] **Step 1: Add text-position, BF16, multi-head, and mutation tests**

Cover:

- `[tokens]` positions equal explicit replicated `[3,tokens]`;
- different query/key head counts;
- zero-position identity;
- BF16 against an FP32 oracle;
- exact unchanged suffix;
- no mutation of position, query, or key inputs.

- [x] **Step 2: Add fail-closed contract tests**

Cover invalid:

- boolean, non-integer, non-positive, odd, and oversized dimensions;
- base values at or below one, infinity, and NaN;
- malformed, non-positive, or wrong-sum sections;
- position rank, leading axis, token count, dtype, and device;
- query/key rank, token mismatch, non-multiple features, integer dtype,
  unequal dtype, and unequal device.

- [x] **Step 3: Run focused GREEN**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_qwen35_partial_interleaved_mrope.py
```

Expected:

```text
qwen35 partial interleaved mrope tests passed
```

### Task 3: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-07-25-qwen35-partial-interleaved-mrope.md`

- [x] **Step 1: Run Qwen3.5 primitive regressions**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_qwen35_partial_interleaved_mrope.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_qwen35_norm_query_gate.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_qwen35_gated_delta_reference.py
/opt/homebrew/bin/python3.12 \
  tools/test_segmented_column_parallel_linear.py
```

- [x] **Step 2: Run static checks**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python -m py_compile \
  tinyvllm/layers/qwen35_rotary_embedding.py \
  tools/test_qwen35_partial_interleaved_mrope.py
git diff --check
```

- [x] **Step 3: Record formulas, RED/GREEN evidence, and claim boundary**

Append the exact official source SHA, lane selection, partial-prefix behavior,
test matrix, validation commands, and remaining non-claims to
`AGENT_HANDOFF_STATE.md`. Mark every plan checkbox only after fresh evidence.
