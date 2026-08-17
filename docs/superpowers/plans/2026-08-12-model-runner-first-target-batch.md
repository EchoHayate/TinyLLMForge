# ModelRunner First-Target Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one greedy KV-only multi-sequence first-target target forward
that returns the generic speculative runtime's `FirstTargetResult` rows.

**Architecture:** `ModelRunner` validates the speculative execution boundary,
prepares one ordinary decode batch, executes one eager target forward, and
packages ordered per-row target tokens plus optional hidden/logit tensors. The
existing speculative compatibility gate rejects active non-KV/hybrid state so
the first-target and fixed-Q tail methods cannot mutate state that the KV-only
transaction cannot roll back.

**Tech Stack:** Python 3.9+, PyTorch tensor contracts, existing
`FirstTargetResult`, pytest, dependency-light source loading, AST source gates.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, or run
  `git clean`.
- Execute the first-target target model exactly once per batch.
- Do not call `ModelRunner.run()` or `_run_model_step()` from the callback.
- Require `temperature == 0`; stochastic speculative decoding is out of scope.
- Reject active non-KV/hybrid state until it has transaction semantics.
- Keep generic speculative code free of model-name and proposal-source
  branches.
- Preserve ordinary execution behavior.
- Do not claim `LLMEngine` wiring, variable-Q support, CUDA Graph support, GPU
  parity, or end-to-end performance gains.

---

### Task 1: First-Target Execution and Source RED Tests

**Files:**
- Modify: `tools/test_model_runner_spec_verify.py`
- Create: `tools/test_model_runner_first_target_batch_source.py`

**Interfaces:**
- Consumes: existing dependency-light `ModelRunner` loader and
  `FirstTargetResult`.
- Produces: executable behavior and AST constraints for
  `run_spec_first_target_batch()`.

- [x] **Step 1: Add execution RED tests**

Add fake decode rows with:

```text
sequence IDs: (8, 4)
temperatures: (0, 0)
input IDs:    [10, 20]
positions:    [5, 9]
```

Patch `prepare_decode()`, `_kv_offload_before_forward()`,
`_kv_offload_after_forward()`, and `run_model()` to record calls. Return fake
logits whose greedy IDs are `[101, 201]` and fake hidden rows.

Assert:

```text
prepare_decode calls:             1
run_model calls:                  1
is_prefill:                       False
execution_mode:                   decode
return_hidden:                    True
before/after offload hook calls:  1/1
result sequence order:            (8, 4)
result target tokens:             (101, 201)
context reset calls:              1
```

Add coverage for `return_logits=True`, worker rank returning `None` after the
same forward, reset after forward failure, non-greedy rejection before
preparation, and active hybrid-state rejection before preparation.

- [x] **Step 2: Add source RED tests**

Parse `ModelRunner.run_spec_first_target_batch()` and assert:

- exactly one `prepare_decode()` call;
- exactly one `run_model()` call;
- no `run()` or `_run_model_step()` call;
- no `run_model()` call under `for`, list/set/dict comprehensions, or generator
  expressions;
- one `reset_context()` call inside a `finally` block.

- [x] **Step 3: Run RED**

```bash
python3 -m pytest -q \
  tools/test_model_runner_spec_verify.py \
  tools/test_model_runner_first_target_batch_source.py \
  -k 'first_target_batch or hybrid_state_spec_verify'
```

Expected: failures because `run_spec_first_target_batch()` and the hybrid-state
compatibility rejection do not exist.

---

### Task 2: Greedy KV-Only First-Target Batch

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`

**Interfaces:**
- Consumes:

```python
FirstTargetResult
prepare_decode(seqs)
run_model(..., execution_mode="decode")
```

- Produces:

```python
run_spec_first_target_batch(
    seqs,
    *,
    return_hidden=False,
    return_logits=False,
) -> tuple[FirstTargetResult, ...] | None
```

- [x] **Step 1: Extend speculative compatibility validation**

In `_validate_spec_verify_compatibility()`, reject:

```python
if self.hybrid_state_runtime_bridge is not None:
    raise RuntimeError(
        "speculative verification requires transactional "
        "non-KV state"
    )
```

Use `getattr(self, "hybrid_state_runtime_bridge", None)` so dependency-light
fixtures and legacy runner construction remain compatible.

- [x] **Step 2: Implement input validation**

Before decode preparation, require:

```text
seqs is a non-empty tuple
sequence IDs are unique non-negative integers
temperature is int or float but not bool
temperature equals zero
```

Raise `ValueError` for malformed input and `RuntimeError` for non-greedy
execution.

- [x] **Step 3: Implement one-forward execution**

Import `FirstTargetResult` from `tinyvllm.speculative.batch_runtime` and add:

```python
def run_spec_first_target_batch(
    self,
    seqs: tuple[Sequence, ...],
    *,
    return_hidden: bool = False,
    return_logits: bool = False,
) -> tuple[FirstTargetResult, ...] | None:
    try:
        self._validate_spec_first_target_batch(seqs)
        input_ids, positions = self.prepare_decode(list(seqs))
        self._kv_offload_before_forward()
        outputs = self.run_model(
            input_ids,
            positions,
            False,
            return_hidden=return_hidden,
            execution_mode="decode",
        )
        if return_hidden:
            logits, hidden_states = outputs
        else:
            logits = outputs
            hidden_states = None
        self._kv_offload_after_forward()
        if self.rank != 0:
            return None
        target_tokens = logits.argmax(dim=-1).tolist()
        return tuple(
            FirstTargetResult(
                sequence_id=int(seq.seq_id),
                target_token=int(target_tokens[batch_index]),
                target_hidden=(
                    hidden_states[batch_index]
                    if hidden_states is not None
                    else None
                ),
                target_logits=(
                    logits[batch_index]
                    if return_logits
                    else None
                ),
                metadata={
                    "batch_index": batch_index,
                    "execution_mode": "decode",
                },
            )
            for batch_index, seq in enumerate(seqs)
        )
    finally:
        reset_context()
```

The comprehension packages results only; it must not contain the model
forward.

- [x] **Step 4: Run GREEN**

```bash
python3 -m pytest -q \
  tools/test_model_runner_spec_verify.py \
  tools/test_model_runner_first_target_batch_source.py \
  -k 'first_target_batch or hybrid_state_spec_verify'
```

Expected: all selected tests pass.

---

### Task 3: Regression, Evidence, and Handoff

**Files:**
- Modify:
  `docs/superpowers/audits/2026-08-12-generic-inference-optimization-goal-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/plans/2026-08-12-model-runner-first-target-batch.md`

**Interfaces:**
- Consumes: completed first-target and fixed-Q tail callback pair.
- Produces: fresh evidence and explicit remaining engine/stateful-model gaps.

- [x] **Step 1: Run focused regression**

```bash
python3 -m pytest -q \
  tools/test_spec_verify_batch_contract.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_model_runner_first_target_batch_source.py \
  tools/test_model_runner_batch_spec_verify_source.py \
  tools/test_engine_speculative_execution.py \
  tools/test_llm_engine_speculative_selection_source.py \
  tools/test_speculative_selection_record.py \
  tools/test_scheduler_speculative_selection.py \
  tools/test_speculative_source_adapters.py \
  tools/test_speculative_adapter.py \
  tools/test_speculative_batch_runtime.py \
  tools/test_speculative_public_api.py \
  tools/test_speculative_runtime.py \
  tools/test_speculative_kv_transaction.py \
  tools/test_ngram_speculative.py

PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.12 \
  tools/test_native_verifier_attention.py

PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.12 \
  tools/test_chunked_prefill.py
```

- [x] **Step 2: Run compatibility and hygiene gates**

Run Python 3.9 and 3.12 `py_compile` for all changed Python files. Scan
`tinyvllm/speculative`, the new callback method, verifier context, and
attention code for model-name/proposal-source branching. Run
`git diff --check`, verify no unchecked boxes remain after completion, and
verify the staged diff is empty.

- [x] **Step 3: Update strict evidence**

Record exact fresh results and:

```text
greedy KV-only first-target ModelRunner batch:
  implemented
fixed-Q tail ModelRunner batch:
  implemented
non-KV/hybrid speculative state:
  fail closed pending transaction design
non-greedy speculative decoding:
  fail closed
LLMEngine callback wiring:
  not implemented
end-to-end performance:
  unmeasured
overall classification:
  NOT_PROMOTABLE
```

- [x] **Step 4: Complete the plan**

Only after fresh verification, change every checkbox in this plan to `[x]`.
