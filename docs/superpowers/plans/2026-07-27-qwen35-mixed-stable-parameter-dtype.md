# Qwen3.5 Mixed Stable-Parameter Dtype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept the verified Qwen3.5 BF16 compute plus F32 `A_log`/normalization checkpoint layout in the existing CPU linear-attention shell without changing output/state dtypes, reference math, state ownership, or runtime wiring.

**Architecture:** Keep convolution and `dt_bias` in the hidden compute dtype, preserve `A_log` and normalization scale as floating stable parameters, and rely on the existing FP32 recurrent and gated-normalization accumulation. Change only fail-closed dtype validation and add checkpoint-like numerical, continuation, non-mutation, and failure tests.

**Tech Stack:** Python 3.12, PyTorch CPU, dependency-light executable tests.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; do not use subagents.
- Do not stage, commit, merge, or delete untracked experiment evidence.
- Do not start local or remote GPU/checkpoint work.
- Do not materialize or assign safetensors payloads.
- Do not modify the Qwen3.5 schema-v2 canonical `NO_GO`.
- Do not change state ownership or create a second state pool.
- Keep production `ModelRunner` fixed to `Qwen3ForCausalLM`.
- Keep `LLMEngine.step()` and Scheduler admission unconnected.
- Preserve `RuntimeError("hybrid prefix reuse requires aligned state snapshot")`.
- Do not claim speed, cache, memory, compression, or quality improvement.
- This session intentionally omits all commit steps.

---

### Task 1: Write Mixed-Dtype RED Tests

**Files:**
- Modify: `tools/test_qwen35_gated_delta_reference.py`
- Modify: `tools/test_qwen35_linear_attention_shell.py`

**Interfaces:**
- Consumes:

```python
qwen35_gated_rmsnorm(core, gate, weight, eps=1e-6)
Qwen35LinearAttentionShell(...)
```

- Produces the required BF16 compute/F32 stable-parameter behavior.

- [x] **Step 1: Add a primitive mixed-weight test**

Use BF16 `core` and `gate`, F32 `weight`, and an independent FP32 formula:

```python
expected = core.float() * torch.rsqrt(
    core.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6
)
expected = (
    expected * weight.float() * F.silu(gate.float())
).to(torch.bfloat16)
```

Assert BF16 output, exact input non-mutation, and closeness to `expected`.
Keep the existing BF16 core versus F64 gate case as a dtype failure so only
the weight dtype rule is relaxed.

- [x] **Step 2: Generalize shell fixtures without changing old cases**

Change `_parameters()` to accept:

```python
compute_dtype: torch.dtype
stable_dtype: torch.dtype | None = None
```

Create projection, convolution, `dt_bias`, and output values in
`compute_dtype`; create `A_log` and `norm_weight` in `stable_dtype` when
provided. Make `_manual_oracle()` and `_new_shell()` accept the same stable
dtype so the oracle uses exactly the shell's source tensors.

- [x] **Step 3: Add checkpoint-like shell correctness**

Construct:

```python
shell = _new_shell(
    [],
    dtype=torch.bfloat16,
    stable_dtype=torch.float32,
)
```

Run BF16 hidden/convolution/recurrent inputs and assert:

```text
output/candidate states are BF16
A_log/norm_weight are F32
all outputs match the independent FP32 oracle within rtol=2e-2, atol=2e-2
hidden/states/A_log/norm_weight remain unchanged
```

- [x] **Step 4: Add checkpoint-like split continuation**

Run one-shot and `[:1]` plus `[1:]` continuation using the mixed shell.
Compare concatenated output and final states within BF16 tolerance.

- [x] **Step 5: Update fail-closed dtype cases**

Replace the obsolete F64 `norm_weight` constructor failure with:

```text
F64 dt_bias against F32 conv_weight -> "compute parameter dtype"
```

Add forward-time replacement cases:

```text
conv_weight dtype differs from hidden -> "conv_weight dtype"
dt_bias dtype differs from hidden     -> "dt_bias dtype"
A_log wrong device                    -> "A_log device"
norm_weight wrong device              -> "norm_weight device"
```

- [x] **Step 6: Run focused RED**

Run:

```bash
/opt/homebrew/bin/python3.12 tools/test_qwen35_gated_delta_reference.py
/opt/homebrew/bin/python3.12 tools/test_qwen35_linear_attention_shell.py
```

Expected failures:

```text
primitive: "core, gate, and weight dtype must match"
shell:     "linear-attention parameter dtype must match"
```

The failures must come from the old restrictions, not fixture errors.

### Task 2: Implement Minimal Mixed-Dtype Validation

**Files:**
- Modify: `tinyvllm/layers/gated_delta.py`
- Modify: `tinyvllm/layers/qwen35_linear_attention.py`

**Interfaces:**
- Keeps all public signatures unchanged.
- Produces mixed stable-parameter acceptance with unchanged math.

- [x] **Step 1: Relax only the gated RMSNorm weight dtype**

Replace:

```python
if core.dtype != gate.dtype or core.dtype != weight.dtype:
    raise ValueError("core, gate, and weight dtype must match")
```

with:

```python
if core.dtype != gate.dtype:
    raise ValueError("core and gate dtype must match")
```

Keep the common-device validation and existing FP32 formula unchanged.

- [x] **Step 2: Split constructor dtype validation**

After floating checks, require:

```python
if conv_weight.dtype != dt_bias.dtype:
    raise ValueError(
        "linear-attention compute parameter dtype must match"
    )
```

Do not compare `A_log.dtype` or `norm_weight.dtype` with the compute group.
Keep the all-parameter common-device requirement.

- [x] **Step 3: Validate each parameter at forward time**

Require:

```text
conv_weight dtype/device == hidden dtype/device
dt_bias dtype/device == hidden dtype/device
A_log device == hidden device
norm_weight device == hidden device
all parameters remain floating
```

Use parameter-specific errors so tests identify the exact broken buffer. Do
not cast or replace any registered buffer.

- [x] **Step 4: Run focused GREEN**

Run:

```bash
/opt/homebrew/bin/python3.12 tools/test_qwen35_gated_delta_reference.py
/opt/homebrew/bin/python3.12 tools/test_qwen35_linear_attention_shell.py
```

Expected:

```text
qwen35 gated delta reference tests passed
qwen35 linear attention shell tests passed
```

### Task 3: Regression, Static Boundaries, and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-07-27-qwen35-mixed-stable-parameter-dtype.md`

**Interfaces:**
- Consumes the completed mixed-dtype shell.
- Produces fresh evidence and a durable next-gate handoff.

- [x] **Step 1: Run focused Qwen3.5 regressions**

Run under Python 3.12:

```bash
for test_file in \
  tools/test_qwen35_gated_delta_reference.py \
  tools/test_qwen35_linear_attention_shell.py \
  tools/test_qwen35_full_attention_shell.py \
  tools/test_qwen35_packed_stateful_linear_decoder_layer.py \
  tools/test_qwen35_packed_layer_stack.py \
  tools/test_qwen35_transactional_root_causal_lm.py \
  tools/test_qwen35_native_model_owner_binding.py \
  tools/test_qwen35_root_model_assembly_factory.py
do
  /opt/homebrew/bin/python3.12 "$test_file"
done
```

Expected: every executable test reports its pass sentinel.

- [x] **Step 2: Compile changed Python under both interpreters**

Run:

```bash
/usr/bin/python3 -m py_compile \
  tinyvllm/layers/gated_delta.py \
  tinyvllm/layers/qwen35_linear_attention.py \
  tools/test_qwen35_gated_delta_reference.py \
  tools/test_qwen35_linear_attention_shell.py
/opt/homebrew/bin/python3.12 -m py_compile \
  tinyvllm/layers/gated_delta.py \
  tinyvllm/layers/qwen35_linear_attention.py \
  tools/test_qwen35_gated_delta_reference.py \
  tools/test_qwen35_linear_attention_shell.py
```

- [x] **Step 3: Verify unchanged production boundaries**

Run:

```bash
rg -n "Qwen3ForCausalLM" tinyvllm/engine/model_runner.py
rg -n "hybrid prefix reuse requires aligned state snapshot" \
  tinyvllm/engine/scheduler.py
rg -n "qwen35_linear_attention|mixed stable|compute parameter dtype" \
  tinyvllm/engine/llm_engine.py
git diff --check
git diff --cached --name-only
```

Expected:

```text
production Qwen3ForCausalLM constructor remains present
Scheduler aligned-state guard remains present
LLMEngine contains no mixed-dtype shell wiring
diff check passes
staged file list is empty
```

- [x] **Step 4: Update durable handoff**

Append one uniquely titled EOF section recording:

- checkpoint-like dtype layout;
- exact RED failures;
- exact GREEN/regression commands;
- output/state/stable-parameter dtype and non-mutation evidence;
- unchanged production and state-ownership boundaries;
- allowed conclusion and remaining gates.

Do not use `NO_GO` as an insertion anchor.

- [x] **Step 5: Mark this plan complete and rerun static checks**

Change every checkbox to `[x]`, then rerun:

```bash
git diff --check
git diff --cached --name-only
```

