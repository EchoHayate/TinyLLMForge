# Sequence Temperature Serialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve sequence temperature across TP worker pickle transport while
remaining compatible with older state tuple schemas.

**Architecture:** Extend the state tuple from 14 to 15 fields by inserting
temperature immediately before the final token payload. Restore new states
strictly and assign `0.0` to older schemas.

**Tech Stack:** Python 3.9+, pickle, dependency-light sequence tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, or run
  `git clean`.
- Preserve the final payload position.
- Preserve old 14/13/11/5-field restore behavior.
- Do not add unrelated sampling fields.
- Do not claim engine execution or performance gains.

---

### Task 1: Serialization RED Tests

**Files:**
- Modify: `tools/test_hybrid_state_sequence.py`
- Modify: `tools/test_chunked_prefill.py`

- [x] **Step 1: Add nonzero round-trip tests**

Use prompt and decode sequences with temperatures `0.7` and `0.25`; assert the
restored values match exactly.

- [x] **Step 2: Add old-schema default tests**

Construct an explicit old 14-field state and assert `temperature == 0.0`.
Extend existing 13/11/5-field compatibility assertions with the same default.

- [x] **Step 3: Strengthen chunked TP transport test**

Set a nonzero temperature before pickle round trip and assert it survives.

- [x] **Step 4: Run RED**

```bash
python3 -m pytest -q tools/test_hybrid_state_sequence.py
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.12 \
  tools/test_chunked_prefill.py
```

Expected: restored sequences lack temperature or do not preserve the value.

---

### Task 2: Implement Schema 15

**Files:**
- Modify: `tinyvllm/engine/sequence.py`

- [x] **Step 1: Emit temperature**

Insert `self.temperature` before the final payload in `__getstate__()`.

- [x] **Step 2: Restore new and old schemas**

Initialize `self.temperature = 0.0`. For `len(state) >= 15`, validate index 13
as a finite numeric non-boolean value and restore it. Leave the default for
older schemas.

- [x] **Step 3: Update schema documentation**

Describe the new 15-field state and old 14-field compatibility accurately.

- [x] **Step 4: Run GREEN**

Run the commands from Task 1 and require both to pass.

---

### Task 3: Regression and Evidence

**Files:**
- Modify:
  `docs/superpowers/audits/2026-08-12-generic-inference-optimization-goal-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/plans/2026-08-12-sequence-temperature-serialization.md`

- [x] **Step 1: Run complete focused regression**

Run the ModelRunner/speculative/engine gate, native verifier attention, and
chunked-prefill script.

- [x] **Step 2: Run hygiene**

Run Python 3.9/3.12 `py_compile`, `git diff --check`, unchecked-plan scan, and
staged-diff-empty validation.

- [x] **Step 3: Update evidence and limitations**

Record TP temperature transport as implemented while engine runtime wiring and
performance remain unimplemented.

- [x] **Step 4: Complete the plan**

Only after fresh verification, change every checkbox to `[x]`.

## Fresh Completion Evidence

```text
serialization RED:
  8 failed for missing temperature restore and missing strict validation
serialization focused GREEN:
  10 passed
chunked TP transport:
  passed with temperature=0.6 round trip
ModelRunner/speculative/engine focused regression:
  314 passed
native verifier attention:
  passed; CUDA numerical cases deferred to remote gate
chunked prefill:
  passed
Python 3.9 and 3.12 py_compile:
  passed
generic source scan and git diff hygiene:
  passed; staged diff empty
```

Strict boundary:

```text
TP worker temperature transport:
  implemented with schema-15 finite numeric validation
old 14/13/11/5-field states:
  restore temperature=0.0
LLMEngine speculative callback invocation:
  not implemented
end-to-end performance improvement:
  unmeasured
overall classification:
  NOT_PROMOTABLE
```
