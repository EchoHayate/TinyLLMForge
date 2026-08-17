# Qwen3.5 Stateful Linear Decoder-Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dependency-light wrapper that gathers Qwen3.5 linear-attention state, completes the whole decoder layer, and commits both candidate states only after success.

**Architecture:** A focused `nn.Module` composes the existing decoder shell and layer-state adapter. It reuses decoder component validation, treats linear-attention state as isolated candidates, and makes adapter commit the sole persistent-state write.

**Tech Stack:** RL Python 3.9, PyTorch CPU.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; no subagents.
- Do not commit or stage.
- Do not modify the existing stateless decoder-shell behavior.
- No GPU, ModelRunner, checkpoint-loader, or batched-slot integration.
- Preserve immutable schema-v2 evidence and all untracked experiments.
- Do not claim performance, memory, compression, quality, or native-support gains.

---

### Task 1: Stateful Wrapper RED

**Files:**
- Create: `tools/test_qwen35_stateful_linear_decoder_layer.py`
- Create after RED: `tinyvllm/layers/qwen35_stateful_decoder_layer.py`

**Interfaces:**
- Consumes: `Qwen35DecoderLayerShell`, `Qwen35LayerStateAdapter`,
  `HybridStateLease`.
- Produces:

```python
class Qwen35StatefulLinearDecoderLayer(nn.Module):
    def __init__(
        self,
        decoder_layer: Qwen35DecoderLayerShell,
        state_adapter: Qwen35LayerStateAdapter,
    )

    def forward(
        self,
        lease: HybridStateLease,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor
```

- [x] Write a dependency-light test fixture with a real tensor pool, adapter,
  decoder shell, deterministic stateful mixer, norms, and MLP.
- [x] Test exact gather-to-commit order, numerical output, dual-state commit,
  and input nonmutation.
- [x] Test BF16 and non-contiguous hidden input.
- [x] Test stale lease rejection before any component call.
- [x] Test mixer, post-attention norm, and MLP failures leave both pool rows
  unchanged.
- [x] Test malformed mixer return, hidden output, and candidate-state
  contracts fail before commit.
- [x] Test second-copy commit failure rolls back both pool rows.
- [x] Test constructor rejects a full-attention decoder shell.
- [x] Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_qwen35_stateful_linear_decoder_layer.py
```

Expected: missing-module `FileNotFoundError` for
`tinyvllm/layers/qwen35_stateful_decoder_layer.py`.

### Task 2: Minimal GREEN

**Files:**
- Create: `tinyvllm/layers/qwen35_stateful_decoder_layer.py`

- [x] Implement constructor validation for a linear-attention decoder shell.
- [x] Implement gather, decoder component execution, mixer tuple validation,
  candidate preservation, commit-after-complete-layer, and return-after-commit.
- [x] Reuse `Qwen35DecoderLayerShell._validate_component_output` for all hidden
  tensor boundaries.
- [x] Run the focused test and require the final marker:

```text
qwen35 stateful linear decoder layer tests passed
```

### Task 3: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: this plan.

- [x] Run the focused wrapper, adapter, linear-attention, decoder, GDN, MLP,
  full-attention, projection, RoPE, hybrid-state, scheduler/runtime bridge,
  and ModelRunner dependency-light tests.
- [x] Run Python 3.9 `py_compile` for the new implementation and test.
- [x] Run `git diff --check`.
- [x] Mark every plan checkbox complete only after fresh evidence.
- [x] Record the transaction proof and unchanged claim boundaries in
  `AGENT_HANDOFF_STATE.md`.
