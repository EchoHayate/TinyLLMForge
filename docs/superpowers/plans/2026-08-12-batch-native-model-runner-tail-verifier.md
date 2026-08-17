# Batch-Native ModelRunner Tail Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the exactly-one-sequence verifier restriction with one
fixed-query-length multi-sequence target forward that writes verifier KV
directly to final transaction slots.

**Architecture:** Extend the dependency-light verifier contract with immutable
batch metadata and ordered result splitting. `ModelRunner` flattens homogeneous
tail rows into one context and one model forward. Attention reshapes the
flattened rows into `[B, Q, H, D]` and performs one paged FlashAttention call
per layer.

**Tech Stack:** Python 3.9+, dataclasses, PyTorch tensor contracts,
FlashAttention `flash_attn_with_kvcache`, pytest, dependency-light source
loading.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, or run
  `git clean`.
- Do not invoke the single-sequence verifier in a loop.
- One verifier batch must execute `run_model()` exactly once.
- One homogeneous verifier batch must invoke
  `flash_attn_with_kvcache()` exactly once per attention layer.
- Reject heterogeneous query lengths before CUDA upload or KV mutation.
- Keep generic verifier/runtime code free of model-name and proposal-source
  branches.
- Keep accepted-KV direct commit and rejected-suffix rollback ownership in the
  generic batch runtime.
- Do not claim variable-Q, CUDA Graph, TPOT, or end-to-end performance gains.

---

### Task 1: Batch Metadata and Result Splitter

**Files:**
- Modify: `tinyvllm/speculative/verifier.py`
- Create: `tools/test_spec_verify_batch_contract.py`

**Interfaces:**
- Produces:
  - `SpecVerifyBatchRowMetadata`
  - `SpecVerifyBatchMetadata`
  - `SpecVerifyBatchResultRow`
  - `split_spec_verify_batch_target_tokens()`

- [x] **Step 1: Write metadata and splitter RED tests**

Create dependency-light tests that construct:

```python
metadata = SpecVerifyBatchMetadata(
    rows=(
        SpecVerifyBatchRowMetadata(
            sequence_id=8,
            batch_index=0,
            query_offset=0,
            query_len=2,
            input_tokens=(10, 11),
            positions=(5, 6),
            logical_slots=(4, 5),
            physical_slots=(20, 21),
            context_len=6,
            block_table=(5, 6),
        ),
        SpecVerifyBatchRowMetadata(
            sequence_id=4,
            batch_index=1,
            query_offset=2,
            query_len=2,
            input_tokens=(20, 21),
            positions=(9, 10),
            logical_slots=(8, 9),
            physical_slots=(40, 41),
            context_len=10,
            block_table=(10, 11, 12),
        ),
    ),
    query_len=2,
    total_query_tokens=4,
    block_table_width=3,
)
```

Assert flat tokens `(101, 102, 201, 202)` split into ordered sequence rows.
Add rejection for non-tuple input, wrong token count, duplicate sequence IDs,
non-contiguous offsets, and row query length mismatch.

- [x] **Step 2: Run RED**

```bash
python3 -m pytest -q tools/test_spec_verify_batch_contract.py
```

Expected: import failure because the batch contracts do not exist.

- [x] **Step 3: Implement immutable batch contracts**

Add frozen dataclasses and strict integer/tuple validation. Implement:

```python
def split_spec_verify_batch_target_tokens(
    metadata,
    flat_target_tokens,
):
    validate_spec_verify_batch_metadata(metadata)
    if not isinstance(flat_target_tokens, tuple):
        raise ValueError("flat target tokens must be a tuple")
    if len(flat_target_tokens) != metadata.total_query_tokens:
        raise ValueError("flat target token count mismatch")
    return tuple(
        SpecVerifyBatchResultRow(
            sequence_id=row.sequence_id,
            target_tokens=flat_target_tokens[
                row.query_offset:
                row.query_offset + row.query_len
            ],
        )
        for row in metadata.rows
    )
```

- [x] **Step 4: Run GREEN**

```bash
python3 -m pytest -q tools/test_spec_verify_batch_contract.py
```

Expected: all tests pass.

---

### Task 2: Batch Preparation and Context Contract

**Files:**
- Modify: `tinyvllm/utils/context.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**
- Consumes: batch metadata contracts from Task 1.
- Produces:
  - `Context.spec_verify_query_lens`
  - `ModelRunner.prepare_spec_verify_batch(items)`
  - multi-sequence compatibility validation.

- [x] **Step 1: Add batch preparation RED tests**

Extend the existing dependency-light ModelRunner tests with two
`TailBatchItem`-compatible fixtures. Assert:

```text
sequence IDs:       (8, 4)
query lengths:      (2, 2)
flat input IDs:     [10, 11, 20, 21]
flat positions:     [5, 6, 9, 10]
flat physical slots:[20, 21, 40, 41]
context lengths:    [6, 10]
block tables:       [[5, 6, -1], [10, 11, 12]]
```

Assert metadata offsets `(0, 2)`, one upload per flattened tensor, and stable
row order.

Add fail-closed cases for:

```text
empty items
duplicate sequence IDs
query_len == 0
heterogeneous query lengths
non-SpecVerifyPlan plan
insufficient proxy block table
negative visible block ID
```

Change the compatibility test so `seq_count=2` is accepted while zero rows,
non-linear drafts, non-greedy acceptance, mixed batches, and existing
unsupported features still fail.

- [x] **Step 2: Run RED**

```bash
python3 -m pytest -q \
  tools/test_model_runner_spec_verify.py \
  -k 'spec_verify'
```

Expected: failures because batch preparation and context fields are absent.

- [x] **Step 3: Extend Context**

Add:

```python
spec_verify_query_lens: tuple[int, ...] = ()
```

to `Context`, add a matching `set_context()` argument, normalize it to an
integer tuple, and pass it into the created context.

- [x] **Step 4: Implement batch preparation**

Refactor compatibility validation to require `seq_count > 0`, then implement
`prepare_spec_verify_batch()` with complete host-side validation before the
first `_list_to_cuda()` call.

Build per-row metadata with:

```python
query_offset = batch_index * query_len
physical_slots = validate_spec_verify_slots(
    plan,
    visible_block_table,
    self.block_size,
)
```

Pad visible block-table rows only after validation and install one
`spec_verify` context.

Keep `prepare_spec_verify()` behavior unchanged by making it use the same
internal row-preparation helper and return the existing
`SpecVerifyMetadata`.

- [x] **Step 5: Run GREEN**

```bash
python3 -m pytest -q \
  tools/test_model_runner_spec_verify.py \
  -k 'spec_verify'
```

Expected: all selected tests pass.

---

### Task 3: Batch-Native Attention

**Files:**
- Modify: `tinyvllm/layers/attention.py`
- Modify: `tools/test_native_verifier_attention.py`

**Interfaces:**
- Consumes:
  - `context.context_lens`
  - `context.block_tables`
  - `context.spec_verify_query_lens`
- Produces one FlashAttention call for a homogeneous verifier batch.

- [x] **Step 1: Add multi-row attention RED tests**

Upgrade `FakeTensor` with `view()` and `view_as()` support. Add a two-row,
two-query test:

```text
q input shape:       (4, 4, 8)
query lengths:       (2, 2)
FlashAttention q:    (2, 2, 4, 8)
context lengths:     [6, 10]
block-table rows:    [[5, 6, -1], [10, 11, 12]]
output shape:        (4, 4, 8)
FlashAttention calls: 1
```

Add rejection for missing query lengths, row-count mismatch, heterogeneous
query lengths, zero query length, and flattened query-count mismatch.

- [x] **Step 2: Run RED**

```bash
python3 tools/test_native_verifier_attention.py
```

Expected: the existing exactly-one-row guard rejects the new batch.

- [x] **Step 3: Implement homogeneous batch attention**

Replace the one-row guard with:

```python
batch_size = int(context.context_lens.numel())
query_lens = tuple(context.spec_verify_query_lens)
query_len = query_lens[0]
if any(length != query_len for length in query_lens):
    raise RuntimeError(
        "spec_verify requires homogeneous query lengths"
    )
if q.size(0) != batch_size * query_len:
    raise RuntimeError(
        "spec_verify flattened query count mismatch"
    )
batched_q = q.view(
    batch_size,
    query_len,
    q.size(1),
    q.size(2),
)
```

Call `flash_attn_with_kvcache()` once and flatten with `view_as(q)`.

- [x] **Step 4: Run GREEN**

```bash
python3 tools/test_native_verifier_attention.py
```

Expected final lines:

```text
native verifier attention dispatch tests passed
CUDA numerical capability cases deferred to remote gate
```

---

### Task 4: ModelRunner Tail Method

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_model_runner_spec_verify.py`
- Create: `tools/test_model_runner_batch_spec_verify_source.py`

**Interfaces:**
- Consumes:
  - `prepare_spec_verify_batch()`
  - `split_spec_verify_batch_target_tokens()`
- Produces:

```python
ModelRunner.run_spec_verify_batch(
    items,
) -> tuple[SpecVerifyBatchResultRow, ...] | None
```

- [x] **Step 1: Add execution/source RED tests**

Add a dependency-light execution test with a fake runner that records:

```text
prepare_spec_verify_batch calls: 1
run_model calls:                  1
execution_mode:                   spec_verify
is_prefill:                       False
reset_context calls:              1
```

Return fake logits whose `argmax(dim=-1).tolist()` is
`[101, 102, 201, 202]` and assert ordered result rows.

Add rank-1 coverage returning `None` after the same model forward.

Parse `run_spec_verify_batch()` with `ast` and assert:

- exactly one `run_model()` call;
- no call to `prepare_spec_verify()`;
- no `run_model()` call nested inside a `for` or comprehension;
- context reset occurs in a `finally` block.

- [x] **Step 2: Run RED**

```bash
python3 -m pytest -q \
  tools/test_model_runner_spec_verify.py \
  tools/test_model_runner_batch_spec_verify_source.py \
  -k 'spec_verify'
```

Expected: failures because `run_spec_verify_batch()` does not exist.

- [x] **Step 3: Implement the tail method**

Implement:

```python
def run_spec_verify_batch(self, items):
    try:
        input_ids, positions, metadata = (
            self.prepare_spec_verify_batch(items)
        )
        logits = self.run_model(
            input_ids,
            positions,
            is_prefill=False,
            execution_mode="spec_verify",
        )
        if self.rank != 0:
            return None
        flat_target_tokens = tuple(
            int(token_id)
            for token_id in logits.argmax(dim=-1).tolist()
        )
        return split_spec_verify_batch_target_tokens(
            metadata,
            flat_target_tokens,
        )
    finally:
        reset_context()
```

- [x] **Step 4: Run GREEN**

```bash
python3 -m pytest -q \
  tools/test_model_runner_spec_verify.py \
  tools/test_model_runner_batch_spec_verify_source.py \
  -k 'spec_verify'
```

Expected: all selected tests pass.

---

### Task 5: Regression, Audit, and Handoff

**Files:**
- Modify:
  `docs/superpowers/audits/2026-08-12-generic-inference-optimization-goal-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/plans/2026-08-12-batch-native-model-runner-tail-verifier.md`

- [x] **Step 1: Run focused regression**

```bash
python3 -m pytest -q \
  tools/test_spec_verify_batch_contract.py \
  tools/test_model_runner_spec_verify.py \
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

Run Python 3.9 and 3.12 `py_compile` for all changed Python files, scan generic
verifier/context/attention code for model-name and proposal-source branches,
run `git diff --check`, and verify the staged diff is empty.

- [x] **Step 3: Update evidence and strict limitations**

Record exact fresh counts and:

```text
fixed-Q batch-native ModelRunner tail forward:
  implemented
variable-Q verifier:
  not implemented
LLMEngine callback wiring:
  not implemented
multi-token scheduler postprocess:
  not implemented
SAM post-commit synchronization:
  not implemented
GPU numerical parity and performance:
  unmeasured
overall classification:
  NOT_PROMOTABLE
```

- [x] **Step 4: Complete the plan**

Only after fresh verification, change every checkbox in this plan to `[x]`.
