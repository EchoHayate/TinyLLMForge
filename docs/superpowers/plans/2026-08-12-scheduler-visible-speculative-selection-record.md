# Scheduler-Visible Speculative Selection Record Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish an immutable, source-agnostic speculative selection record
for every scheduler batch without changing the existing scheduler return
tuple.

**Architecture:** A dependency-light builder converts the already scheduled
rows into selected/suppressed records with generation and token-count
snapshots. `Scheduler._return_schedule()` publishes the record and keeps the
legacy tuple untouched; future engine integration validates the sidecar before
proposal or KV transaction work.

**Tech Stack:** Python 3.9+, dataclasses, existing `Scheduler` and `Sequence`
metadata, pytest.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, run `git clean`,
  or modify index state.
- No n-gram, SAM, learned-drafter, MTP, or model-name branches in scheduler
  code.
- Preserve the exact existing three/four-element scheduler return shapes.
- Do not modify `Sequence` serialization.
- Do not begin KV transactions, generate proposals, execute models, append
  tokens, or synchronize SAM in this slice.
- Keep default behavior disabled and ordinary execution unchanged.
- No performance claim.

---

### Task 1: Selection Record RED Tests

**Files:**
- Create: `tools/test_speculative_selection_record.py`

**Interfaces:**
- Produces behavior requirements for:
  - `SpeculativeSelectionConfig`
  - `SpeculativeSelectionRow`
  - `SpeculativeSelectionRecord`
  - `build_speculative_selection_record()`
  - `validate_speculative_selection_record()`

- [x] **Step 1: Add dependency-light fake sequences**

Create fake rows exposing only:

```python
seq_id
num_tokens
num_completion_tokens
max_tokens
step_is_decode
step_do_sample
```

- [x] **Step 2: Add configuration and ordinary-batch tests**

Cover disabled `max_proposal_tokens=0`, enabled minimum K=2, invalid booleans,
ordinary decode selection, output-budget capping, prefill suppression, and
insufficient output budget.

- [x] **Step 3: Add mixed-batch classification tests**

Build one mixed prefill row and two mixed decode rows. Assert only sampling
decode rows with at least two remaining tokens are selected, while row order
still matches the complete scheduled batch.

- [x] **Step 4: Add stale-record validation tests**

Cover schedule-generation mismatch, sequence reorder, duplicate IDs,
`num_tokens` change, completion-count change, and output-budget shrink.

- [x] **Step 5: Run RED**

Run:

```bash
python3 -m pytest -q tools/test_speculative_selection_record.py
```

Expected: collection failure because
`tinyvllm.engine.speculative_selection` does not exist.

---

### Task 2: Dependency-Light Selection Contract

**Files:**
- Create: `tinyvllm/engine/speculative_selection.py`
- Test: `tools/test_speculative_selection_record.py`

**Interfaces:**
- Consumes generic sequence metadata only.
- Produces the five public selection types/functions from Task 1.

- [x] **Step 1: Implement immutable types and config validation**

Reject disabled nonzero K and enabled K below two. Validate integer
non-boolean IDs, indices, counts, and generation.

- [x] **Step 2: Implement row classification**

Apply stable suppression precedence:

```text
disabled
prefill
not_sampling
insufficient_output_budget
selected
```

For selected rows cap maximum K by current remaining output tokens.

- [x] **Step 3: Implement stale validation**

Require exact sequence order and generation, then compare token and completion
snapshots before returning selected sequence objects.

- [x] **Step 4: Run contract GREEN**

Run:

```bash
python3 -m pytest -q tools/test_speculative_selection_record.py
```

Expected: all dependency-light selection tests pass.

---

### Task 3: Scheduler Publication

**Files:**
- Modify: `tinyvllm/engine/scheduler.py`
- Create: `tools/test_scheduler_speculative_selection.py`

**Interfaces:**
- Consumes:
  - `SpeculativeSelectionConfig`
  - `build_speculative_selection_record()`
- Produces:
  - `Scheduler.install_speculative_selection(config)`
  - `Scheduler.schedule_generation`
  - `Scheduler.last_speculative_selection`

- [x] **Step 1: Add scheduler integration RED tests**

Use dependency-light scheduler loading consistent with existing scheduler
tests. Assert default-disabled publication, equal-config idempotence,
different-config rejection, and exact legacy tuple preservation.

- [x] **Step 2: Verify scheduler RED**

Run:

```bash
python3 -m pytest -q tools/test_scheduler_speculative_selection.py
```

Expected: failure because scheduler installation and publication fields do not
exist.

- [x] **Step 3: Add default state and installation**

Initialize disabled configuration, generation zero, and no last record.
Implement one-time idempotent installation without source/model imports.

- [x] **Step 4: Publish at `_return_schedule()`**

Parse the existing three/four-element tuple, increment generation once, build
the record, set `last_policy_branch`, save the record, and return the exact
original tuple object.

- [x] **Step 5: Run scheduler GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_speculative_selection_record.py \
  tools/test_scheduler_speculative_selection.py
```

Expected: all selection and scheduler-publication tests pass.

---

### Task 4: Regression, Audit, and Handoff

**Files:**
- Modify: `docs/superpowers/audits/2026-08-12-generic-inference-optimization-goal-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes the completed selection record and scheduler publication.
- Produces exact validation evidence and the next engine-integration boundary.

- [x] **Step 1: Run scheduler and speculative regressions**

Run:

```bash
python3 -m pytest -q \
  tools/test_speculative_selection_record.py \
  tools/test_scheduler_speculative_selection.py \
  tools/test_speculative_source_adapters.py \
  tools/test_speculative_adapter.py \
  tools/test_speculative_batch_runtime.py \
  tools/test_speculative_public_api.py \
  tools/test_speculative_runtime.py \
  tools/test_speculative_kv_transaction.py \
  tools/test_ngram_speculative.py \
  tools/test_scheduler_prefill_commit_hook.py \
  tools/test_hybrid_state_scheduler.py

PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.12 \
  tools/test_chunked_prefill.py
```

Expected: all pytest tests pass and chunked prefill prints
`chunked prefill tests passed`.

Environment note: the focused selection/scheduler/speculative matrix and
chunked-prefill script are authoritative for this slice. The separate
`tools/test_scheduler_prefill_commit_hook.py` and
`tools/test_hybrid_state_scheduler.py` collections require local `torch`;
when unavailable they must be reported as unexecuted, not passing or failing.

- [x] **Step 2: Run compatibility and hygiene checks**

Run Python 3.9 and 3.12 `py_compile` on the new module and tests, then:

```bash
rg -n \
  "Qwen|Llama|Mistral|\\bngram\\b|\\bsam\\b|draft_model|\\bmtp\\b" \
  tinyvllm/engine/speculative_selection.py

git diff --check
git diff --cached --name-only
```

Expected: generic source scan has no matches, diff check passes, and staged
diff is empty.

- [x] **Step 3: Update audit and handoff**

Record the immutable record API, scheduler generation/publication semantics,
fresh test counts, and unchanged limitations:

```text
LLMEngine record consumption:
  not implemented
real ModelRunner callbacks:
  not implemented
scheduler/engine-owned speculative transaction execution:
  not implemented
GPU/TP/long-context/exact-model parity:
  unproven
performance improvement:
  unmeasured
overall classification:
  NOT_PROMOTABLE
```

- [x] **Step 4: Mark this plan complete**

Only after fresh verification, change every checkbox in this plan to `[x]`.
