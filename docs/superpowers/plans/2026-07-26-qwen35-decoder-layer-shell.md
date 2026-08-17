# Qwen3.5 Decoder-Layer Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove Qwen3.5 full/linear token-mixer dispatch and exact decoder-layer normalization and residual order in a dependency-light CPU shell.

**Architecture:** Add one injected shell that selects exactly one token mixer from `block_type`, validates every component boundary, and performs the two official residual additions. Deterministic test modules expose call order and provide independent numerical oracles without model, checkpoint, cache, distributed, or GPU dependencies.

**Tech Stack:** RL Python 3.9, PyTorch, standalone dependency-light test scripts.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not commit or stage changes.
- Do not start local or remote GPU processes.
- Do not modify immutable schema-v2 canonical evidence.
- Do not add model selection, checkpoint traversal, cache mutation, paged attention, or runtime integration.
- Preserve all untracked `experiments/` evidence.

---

### Task 1: Decoder-Layer Dispatch and Residual Math

**Files:**
- Create: `tinyvllm/layers/qwen35_decoder_layer.py`
- Create: `tools/test_qwen35_decoder_layer_shell.py`

**Interfaces:**
- Consumes:
  injected `nn.Module` norms, full/linear token mixers, and MLP.
- Produces:
  `Qwen35DecoderLayerShell.forward(position_ids, hidden_states) -> Tensor`.

- [x] **Step 1: Write full-attention operation-order and oracle test**

Use deterministic affine modules and record:

```text
input_layernorm
full_attention
post_attention_layernorm
mlp
```

Independently compute:

```python
normalized = hidden_states * input_scale + input_bias
mixed = normalized * attention_scale + position_term
after_mixer = hidden_states + mixed
normalized = after_mixer * post_scale + post_bias
mlp_output = normalized * mlp_scale + mlp_bias
expected = after_mixer + mlp_output
```

Assert original hidden states and position ids are unchanged.

- [x] **Step 2: Write linear-attention dispatch and oracle test**

Record:

```text
input_layernorm
linear_attention
post_attention_layernorm
mlp
```

Use a linear-attention fixture accepting only normalized hidden states. Pass a
full-attention fixture that raises if called, proving branch isolation.

- [x] **Step 3: Run and confirm missing-module RED**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_qwen35_decoder_layer_shell.py
```

Expected:

```text
FileNotFoundError:
tinyvllm/layers/qwen35_decoder_layer.py
```

- [x] **Step 4: Implement the minimal shell**

Create:

```python
class Qwen35DecoderLayerShell(nn.Module):
    def __init__(
        self,
        *,
        block_type: str,
        input_layernorm: nn.Module,
        post_attention_layernorm: nn.Module,
        mlp: nn.Module,
        full_attention: Optional[nn.Module] = None,
        linear_attention: Optional[nn.Module] = None,
    )

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor
```

Implement only selected-branch dispatch, immediate exact
shape/dtype/device/floating validation, and the two residual additions.

- [x] **Step 5: Add BF16, non-contiguous, and failure tests**

Cover:

- BF16 output and independent oracle;
- valid non-contiguous hidden/component tensors;
- unsupported block type and missing selected modules;
- hidden-state rank and dtype;
- shape, dtype, device, non-floating, and non-tensor failures for input norm,
  selected token mixer, post-attention norm, and MLP.

- [x] **Step 6: Run and confirm decoder shell GREEN**

Expected:

```text
qwen35 decoder layer shell tests passed
```

### Task 2: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-07-26-qwen35-decoder-layer-shell.md`

**Interfaces:**
- Consumes:
  completed decoder shell and prior Qwen3.5 primitive/shell gates.
- Produces:
  fresh regression evidence and explicit claim boundaries.

- [x] **Step 1: Run the Qwen3.5 CPU correctness suite**

Run decoder shell, full-attention shell, head-paired projection, MRoPE,
norm/query-gate, GDN reference, segmented loader, hybrid layout/state/runtime,
and ModelRunner dependency-light tests.

- [x] **Step 2: Run static checks**

Run Python 3.9 `py_compile` for the decoder shell and test, Python 3.12
`py_compile` for `linear.py` and its tests, and:

```bash
git diff --check
```

- [x] **Step 3: Record RED/GREEN and claim boundary**

Document:

- official decoder operation order and source SHA;
- both branch event orders;
- numerical, BF16, mutation, non-contiguous, and fail-closed coverage;
- every fresh command result;
- remaining production/model/equivalence/performance gates;
- immutable schema-v2 canonical `NO_GO`.

