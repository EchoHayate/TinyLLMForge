# Speculative ModelRunner Callback Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt the generic speculative batch callbacks to all-rank
`ModelRunner.call()` execution with stable fixed-Q tail grouping.

**Architecture:** A new dependency-light engine module validates capabilities,
performs one first-target RPC, partitions tail items by exact query length,
performs one verifier RPC per distinct Q, validates each result group, and
merges rows into original order. The ModelRunner first-target flags become
positional-RPC compatible.

**Tech Stack:** Python 3.9+, frozen dataclasses, existing speculative contracts,
pytest, AST/source tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, or run
  `git clean`.
- Never pad heterogeneous query lengths.
- Never invoke the single-sequence verifier in a loop.
- One tail target forward is allowed per distinct positive query length, not
  per sequence.
- Preserve original sequence order in callback results.
- Keep generic code free of model-name and proposal-source branches.
- Do not wire `LLMEngine.step()` or claim end-to-end gains in this slice.

---

### Task 1: Callback Bridge RED Tests

**Files:**
- Create: `tools/test_speculative_model_runner_callbacks.py`
- Modify: `tools/test_model_runner_first_target_batch_source.py`

- [x] **Step 1: Add fixed-Q grouping RED tests**

Construct items:

```text
seq8/Q2
seq4/Q1
seq2/Q2
seq9/Q3
```

Assert group order `Q2, Q1, Q3`, group membership `(8,2), (4), (9)`, and
rejection of empty input, duplicate sequence IDs, and zero query length.

- [x] **Step 2: Add first-target RPC RED tests**

Use a fake `model_runner.call()` that records:

```text
method: run_spec_first_target_batch
args:   seqs, requires_hidden, requires_logits
calls:  1
```

Return reversed `FirstTargetResult` rows and assert the bridge restores input
order. Reject `None`, missing/extra IDs, duplicates, and wrong row types.

- [x] **Step 3: Add tail RPC/merge RED tests**

Return reversed `SpecVerifyBatchResultRow` rows for every group. Assert:

```text
RPC calls:              3
RPC query lengths:      2, 1, 3
result sequence order:  8, 4, 2, 9
metadata group count:   3
```

Reject `None`, missing/extra IDs, duplicates, wrong row types, and wrong target
token counts.

- [x] **Step 4: Add positional signature source test**

Parse `run_spec_first_target_batch()` and assert `return_hidden` and
`return_logits` are regular positional-or-keyword arguments, not
keyword-only arguments.

- [x] **Step 5: Run RED**

```bash
python3 -m pytest -q \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_model_runner_first_target_batch_source.py
```

Expected: bridge module is missing and the first-target flags are
keyword-only.

---

### Task 2: Implement the Callback Bridge

**Files:**
- Create: `tinyvllm/engine/speculative_model_runner.py`
- Modify: `tinyvllm/engine/model_runner.py`

- [x] **Step 1: Make first-target flags RPC-compatible**

Change:

```python
def run_spec_first_target_batch(
    self,
    seqs,
    *,
    return_hidden=False,
    return_logits=False,
):
```

to positional-or-keyword boolean parameters by removing `*`.

- [x] **Step 2: Implement immutable fixed-Q groups**

Add `FixedQTailBatch` and `build_fixed_q_tail_batches()` with exact tuple,
type, sequence-ID, positive-Q, uniqueness, and stable-order validation.

- [x] **Step 3: Implement first-target adapter**

Validate `DraftCapabilities`, call ModelRunner once with positional booleans,
validate exact unique IDs and row types, and return rows in sequence order.

- [x] **Step 4: Implement tail adapter**

Build fixed-Q groups, call `run_spec_verify_batch` once per group, validate
`SpecVerifyBatchResultRow` rows and target counts, convert to
`TailBatchResult`, and merge in original order.

- [x] **Step 5: Run GREEN**

```bash
python3 -m pytest -q \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_model_runner_first_target_batch_source.py
```

Expected: all tests pass.

---

### Task 3: Regression and Evidence

**Files:**
- Modify:
  `docs/superpowers/audits/2026-08-12-generic-inference-optimization-goal-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/plans/2026-08-12-speculative-model-runner-callback-bridge.md`

- [x] **Step 1: Run focused regression**

Run the complete ModelRunner/speculative/engine pytest gate, native verifier
attention script, and chunked-prefill script.

- [x] **Step 2: Run compatibility and hygiene**

Run Python 3.9 and 3.12 `py_compile`, generic source scan,
`git diff --check`, unchecked-plan scan, and staged-diff-empty validation.

- [x] **Step 3: Update evidence**

Record:

```text
first-target ModelRunner RPC:
  one per selected batch
tail ModelRunner RPC:
  one per distinct fixed Q
LLMEngine callback invocation:
  not implemented
overall classification:
  NOT_PROMOTABLE
```

- [x] **Step 4: Complete the plan**

Only after fresh verification, change every checkbox to `[x]`.
