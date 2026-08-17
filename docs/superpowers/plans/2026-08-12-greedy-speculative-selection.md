# Greedy Speculative Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Suppress nonzero-temperature rows before the greedy-only
speculative runtime and stale-check temperature at engine consumption.

**Architecture:** Add a normalized finite temperature snapshot to each
immutable selection row. Selection applies a `non_greedy` suppression reason
after prefill/not-sampling checks, while record validation compares the live
temperature against the snapshot and rejects selected stochastic rows.

**Tech Stack:** Python 3.9+, frozen dataclasses, pytest, dependency-light
scheduler loading.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, or run
  `git clean`.
- Preserve scheduler tuple identity and shape.
- Preserve suppression precedence for disabled, prefill, and non-sampling
  rows.
- Keep the ModelRunner greedy check as defense in depth.
- Do not implement stochastic speculative decoding.
- Do not claim engine execution or performance improvement.

---

### Task 1: Selection Contract RED Tests

**Files:**
- Modify: `tools/test_speculative_selection_record.py`
- Modify: `tools/test_scheduler_speculative_selection.py`

**Interfaces:**
- Produces required `temperature_snapshot` and `non_greedy` behavior.

- [x] **Step 1: Extend fixtures with temperature**

Give both dependency-light `_Sequence` fixtures a `temperature=0.0` default.

- [x] **Step 2: Add selection RED tests**

Assert:

```text
temperature 0.0:
  selected
temperature 0.7:
  selected=false, reason=non_greedy, max_proposal_tokens=0
mixed greedy/stochastic decode:
  only greedy row selected
temperature mutation after publication:
  validation raises stale temperature
```

Add invalid values:

```python
True
"0"
float("nan")
float("inf")
```

- [x] **Step 3: Add scheduler RED test**

Install enabled selection, return a decode row with temperature `0.7`, and
assert the published record contains no selected rows and reason
`non_greedy`.

- [x] **Step 4: Run RED**

```bash
python3 -m pytest -q \
  tools/test_speculative_selection_record.py \
  tools/test_scheduler_speculative_selection.py
```

Expected: missing `temperature_snapshot` and stochastic rows remain selected.

---

### Task 2: Temperature Snapshot and Suppression

**Files:**
- Modify: `tinyvllm/engine/speculative_selection.py`

**Interfaces:**
- Produces:

```python
SpeculativeSelectionRow.temperature_snapshot: float
```

- [x] **Step 1: Normalize temperature**

Add a helper that rejects booleans, non-numeric values, NaN, and infinity, and
returns `float(value)`.

- [x] **Step 2: Add the immutable snapshot**

Extend `_sequence_values()` and `SpeculativeSelectionRow` with normalized
temperature. Populate `temperature_snapshot` for every row.

- [x] **Step 3: Suppress stochastic rows**

After `not_sampling` and before output-budget checks:

```python
elif temperature != 0.0:
    suppression_reason = "non_greedy"
```

- [x] **Step 4: Validate freshness**

Reject a live temperature that differs from the row snapshot. Also reject any
selected row whose live temperature is nonzero.

- [x] **Step 5: Run GREEN**

```bash
python3 -m pytest -q \
  tools/test_speculative_selection_record.py \
  tools/test_scheduler_speculative_selection.py
```

Expected: all tests pass.

---

### Task 3: Regression and Evidence

**Files:**
- Modify:
  `docs/superpowers/audits/2026-08-12-generic-inference-optimization-goal-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/plans/2026-08-12-greedy-speculative-selection.md`

- [x] **Step 1: Run focused regression**

Run the complete first-target/fixed-Q/speculative/engine pytest gate, native
verifier attention script, and chunked-prefill script.

- [x] **Step 2: Run hygiene gates**

Run Python 3.9 and 3.12 `py_compile`, generic source scan,
`git diff --check`, unchecked-plan scan, and staged-diff-empty validation.

- [x] **Step 3: Update strict evidence**

Record:

```text
non-greedy scheduler selection:
  suppressed and stale-checked
stochastic speculative decoding:
  not implemented
LLMEngine callback wiring:
  not implemented
overall classification:
  NOT_PROMOTABLE
```

- [x] **Step 4: Complete the plan**

Only after fresh verification, change every checkbox to `[x]`.
