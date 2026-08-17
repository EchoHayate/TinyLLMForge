# Engine Speculative Selection and Commit Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add dependency-light engine selection partitioning and exact
speculative output-token commit planning, then consume scheduler records in
`LLMEngine.step()` with default-off behavior.

**Architecture:** `speculative_execution.py` validates scheduler selection and
runtime result contracts without mutating sequences. `LLMEngine.step()` always
consumes the scheduler record, continues ordinary execution when no rows are
selected, and fails before ModelRunner execution if selection is enabled
before a real runtime is installed.

**Tech Stack:** Python 3.9+, dataclasses, existing
`SpeculativeSelectionRecord`, `NativeSpeculativeBatchResult`, pytest, AST
source checks for heavy engine wiring.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, run `git clean`,
  or modify index state.
- No model-name or proposal-source branches.
- Do not mutate `Sequence`, scheduler queues, KV transactions, or SAM state in
  the dependency-light bridge.
- Do not fake a batch-native verifier by looping the single-sequence
  `ModelRunner.spec_verify` path.
- Preserve default ordinary `LLMEngine.step()` execution.
- Fail before ModelRunner execution when selected rows have no installed
  engine runtime.
- No performance claim.

---

### Task 1: Engine Bridge RED Tests

**Files:**
- Create: `tools/test_engine_speculative_execution.py`

**Interfaces:**
- Produces requirements for:
  - `EngineSpeculativePartition`
  - `EngineSpeculativeCommitRow`
  - `build_engine_speculative_partition()`
  - `build_engine_speculative_commit_rows()`

- [x] **Step 1: Add dependency-light sequence and result fixtures**

Use the real immutable selection and batch-runtime dataclasses while stubbing
package imports to avoid top-level Torch/Transformers dependencies.

- [x] **Step 2: Add partition tests**

Cover selected/suppressed stable ordering and stale generation, token-count,
completion-count, sequence reorder, and output-budget rejection.

- [x] **Step 3: Add commit semantic tests**

Construct runtime rows for:

```text
empty proposal
zero acceptance
one-token acceptance
partial acceptance plus mismatch target
full acceptance
accepted EOS
fallback EOS
budget-blocked fallback
```

- [x] **Step 4: Add fail-closed validation tests**

Cover missing/extra/duplicate/reordered IDs, accepted tokens not matching the
proposal prefix, greedy count outside proposal bounds, and insufficient target
tokens for a partial mismatch fallback.

- [x] **Step 5: Run RED**

Run:

```bash
python3 -m pytest -q tools/test_engine_speculative_execution.py
```

Expected: collection failure because
`tinyvllm.engine.speculative_execution` does not exist.

---

### Task 2: Dependency-Light Engine Bridge

**Files:**
- Create: `tinyvllm/engine/speculative_execution.py`
- Test: `tools/test_engine_speculative_execution.py`

**Interfaces:**
- Consumes:
  - `validate_speculative_selection_record()`
  - `NativeSpeculativeBatchResult`
- Produces the four public bridge contracts from Task 1.

- [x] **Step 1: Implement stable partition**

Delegate stale validation to the selection module, preserve batch order, and
validate selected/suppressed disjoint exact coverage.

- [x] **Step 2: Implement runtime-result validation**

Require exact ordered IDs, proposal-prefix accepted tokens, bounded greedy
counts, integer tokens, and sufficient target rows for fallback.

- [x] **Step 3: Implement output-token planning**

Apply empty/zero/partial/full acceptance, EOS, and output-budget rules from the
design. Return immutable commit rows without sequence mutation.

- [x] **Step 4: Run GREEN**

Run:

```bash
python3 -m pytest -q tools/test_engine_speculative_execution.py
```

Expected: all bridge tests pass.

---

### Task 3: LLMEngine Default-Off Consumption

**Files:**
- Modify: `tinyvllm/engine/llm_engine.py`
- Create: `tools/test_llm_engine_speculative_selection_source.py`

**Interfaces:**
- Consumes: `build_engine_speculative_partition()`.
- Produces:
  - default-off record validation on every step;
  - fail-closed selected-row behavior before ModelRunner execution;
  - selection fields in `last_step_observation`.

- [x] **Step 1: Add source-contract RED test**

Parse `LLMEngine.step()` with `ast` and assert it does not yet contain the
partition builder call, selected-row guard, and observation keys.

- [x] **Step 2: Run source RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_llm_engine_speculative_selection_source.py
```

Expected: assertions fail because the engine does not consume the selection
record.

- [x] **Step 3: Add default-off engine wiring**

Import the bridge, build the partition immediately after scheduler return
parsing, fail on selected rows before release draining or ModelRunner calls,
and add:

```text
speculative_schedule_generation
speculative_selected_seq_ids
speculative_suppressed_seq_ids
```

to the step observation.

- [x] **Step 4: Run source GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_llm_engine_speculative_selection_source.py
```

Expected: all source-contract assertions pass.

---

### Task 4: Regression, Audit, and Handoff

**Files:**
- Modify: `docs/superpowers/audits/2026-08-12-generic-inference-optimization-goal-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] **Step 1: Run focused regression**

Run:

```bash
python3 -m pytest -q \
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
  tools/test_chunked_prefill.py
```

- [x] **Step 2: Run compatibility and hygiene checks**

Run Python 3.9 and 3.12 `py_compile`, source/model-name scans, plan checkbox
checks, `git diff --check`, and staged-diff inspection.

- [x] **Step 3: Update audit and handoff**

Record exact APIs, token-commit semantics, fresh test counts, and these strict
limitations:

```text
real LLMEngine speculative execution:
  not implemented
batch-native ModelRunner tail verification:
  not implemented
multi-token scheduler postprocess:
  not implemented
SAM post-commit synchronization:
  not implemented
performance improvement:
  unmeasured
overall classification:
  NOT_PROMOTABLE
```

- [x] **Step 4: Mark this plan complete**

Only after fresh verification, change every checkbox to `[x]`.
