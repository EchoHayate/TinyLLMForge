# Qwen3.5 Native GDN Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a pure-PyTorch CPU reference for Qwen3.5 causal depthwise convolution and gated-delta recurrent updates with explicit vLLM-compatible physical state orientation.

**Architecture:** Token-major projected Q/K/V is transformed by a fixed-width causal depthwise convolution state, then a deterministic FP32 recurrent oracle updates mathematical `[head, key, value]` state while reading/writing physical `[head, value, key]` storage. The primitive is side-effect-free and independent of ModelRunner, CUDA, FLA, or causal-conv1d.

**Tech Stack:** Python 3, PyTorch CPU tensors, existing dependency-light script tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Use inline execution; do not dispatch subagents.
- Do not start any local or remote GPU model process.
- Do not implement or claim native Qwen3.5 model support in this slice.
- Preserve the immutable Qwen3.5 schema-v2 canonical `NO_GO`.
- Physical recurrent state is `[head, value_dim, key_dim]`.
- Mathematical recurrence state is `[head, key_dim, value_dim]`.
- Accumulate recurrent math in FP32.
- Do not add FLA, causal-conv1d, Triton, or other dependencies.
- Preserve all untracked experiment evidence.

---

### Task 1: Recurrent Gated-Delta Oracle

**Files:**
- Create: `tinyvllm/layers/gated_delta.py`
- Create: `tools/test_qwen35_gated_delta_reference.py`

**Interfaces:**
- Produces:
  - `qwen35_l2norm(tensor, *, eps=1e-6)`;
  - `qwen35_gated_delta_recurrent(query, key, value, a, b, A_log, dt_bias, recurrent_state_v_k)`.

- [x] **Step 1: Write failing asymmetric-orientation and scalar-oracle tests**

Use `key_dim=2`, `value_dim=3`, one head, and two tokens. Build the expected
state with an explicit Python token loop in mathematical `[K,V]` orientation,
then compare the production result and returned physical `[V,K]` state.

- [x] **Step 2: Run and confirm RED**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_qwen35_gated_delta_reference.py
```

Expected: import failure because `tinyvllm.layers.gated_delta` does not exist.

- [x] **Step 3: Implement validation, L2 norm, and recurrent loop**

Validate rank, token/head agreement, local-head parameter lengths, and
physical orientation. Convert q/k/value/gates/state to FP32, apply the official
equations token by token, then cast output and physical state back.

- [x] **Step 4: Add continuation, isolation, and dtype tests**

Cover:

- one-shot equals two split calls;
- non-zero initial state;
- two independent request states do not cross-write;
- BF16 input/output with FP32 oracle tolerance;
- invalid orientation, token count, and parameter shapes fail.

- [x] **Step 5: Run and confirm GREEN**

Expected: recurrent reference tests pass.

### Task 2: Causal Depthwise Convolution Oracle

**Files:**
- Modify: `tinyvllm/layers/gated_delta.py`
- Modify: `tools/test_qwen35_gated_delta_reference.py`

**Interfaces:**
- Produces:
  - `qwen35_causal_depthwise_conv(projected_qkv, conv_state, weight, *, activation="silu")`.

- [x] **Step 1: Write failing one-shot/chunk continuation tests**

Use token-major `[T,C]`, physical state `[C,K]`, and weight `[C,K]`. Compare
one-shot output/state with a split execution carrying the returned state.
Use channel-specific weights to catch accidental dense convolution.

- [x] **Step 2: Run and confirm RED**

Expected: missing convolution function.

- [x] **Step 3: Implement the minimal fixed-window convolution**

For each token, append the projected value to each channel's history, retain
the latest `K` values, compute elementwise weighted sum over the window, and
apply SiLU. Return token-major output and the new physical state.

- [x] **Step 4: Add state-window and failure tests**

Cover exact latest-window content, no input mutation, unsupported activation,
channel/kernel mismatch, and dtype preservation.

- [x] **Step 5: Run and confirm GREEN**

Expected:

```text
qwen35 gated delta reference tests passed
```

### Task 3: Regression Verification and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] **Step 1: Run focused and existing hybrid tests**

Run the new reference test plus:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_hybrid_state.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_qwen35_hybrid_state_layout.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_hybrid_state_runtime_bridge.py
```

- [x] **Step 2: Run syntax and diff checks**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python -m py_compile \
  tinyvllm/layers/gated_delta.py \
  tools/test_qwen35_gated_delta_reference.py
git diff --check
```

- [x] **Step 3: Update handoff with exact claim boundary**

Record upstream source SHAs, orientation finding, equations, RED/GREEN
evidence, and state explicitly that no model, loader, GPU kernel, correctness
gate, compression, speed, or memory result has been established.
