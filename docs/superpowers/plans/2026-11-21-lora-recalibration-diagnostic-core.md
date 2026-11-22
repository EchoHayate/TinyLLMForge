# 72-Generation Paired Calibration Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the versioned v2 paired `base`/`text` × `initial`/`feedback`/`retry` schedule, candidate syntax/signature validator, public-feedback projector, and 576-position final scorer + calibration opportunity verifier (`recalibration_diagnostic_scoring.py`), plus remote GPU generation/execution driver (`recalibration_diagnostic_remote.py`) and local controller (`recalibration_diagnostic_controller.py`), strictly following `docs/superpowers/specs/2026-11-17-lora-development-recalibration-design.md`.

**Architecture:** `recalibration_diagnostic_scoring.py` consumes the frozen 6-file `recalibration_v2_r1` dataset, constructs the deterministic 24-episode / 72-generation / 576-case schedule (`C01..C12`, counterbalanced arm and branch orders matching `recalibration_tokenizer_inventory.py`), validates candidate single-function ASTs without executing candidate code, projects public `visible_tests` / `invalid_source` / `results_withheld` continuation messages, and scores all 72 candidates after generation finishes into the five states (`PASS`, `SEMANTIC_FAIL`, `EXECUTION_FAIL`, `UNKNOWN`, `INVALID_EVIDENCE`) plus paired transitions, net gains, public-vs-private error exposure, and the four-level calibration status (`INVALID_PROTOCOL`, `INCONCLUSIVE_INCOMPLETE`, `INCONCLUSIVE_CALIBRATION`, `DEVELOPMENT_DIAGNOSTIC_USABLE`). `recalibration_diagnostic_remote.py` and `recalibration_diagnostic_controller.py` execute the 72 greedy generations (`Qwen3-8B` BF16 eager no-cache on 1× A100 80GB, `<= 4096` input, `<= 256` output) and `bwrap`+`seccomp` public/final candidate evaluations in separate phases without leaking private cases into generation.

**Tech Stack:** Python 3.9+ standard library and pinned repository modules (`pilot_protocol`, `validation_contract`, `recalibration_data_contract`, `recalibration_rendering`, `observed_generation`, `generation_core`, `tokenizer_measurement`, `validation_remote`).

## Global Constraints

- Spec authority: `docs/superpowers/specs/2026-11-17-lora-development-recalibration-design.md`.
- Preserve all frozen v1 and recalibration modules byte-for-byte.
- Complete all 72 generations before executing any private scoring case (`S01..S06`).
- Share each `(task, arm)` `initial` candidate across `feedback` and `retry` without sharing branch KV/state, answers, or scores.
- Keep `input_tokens <= 4096`, `output_tokens <= 256`, greedy decoding, eager attention, no KV cache.
- No inline comments (`#`) and no single-letter variable names in Python source or test files.

---

### Task 1: Schedule, Candidate Validator, And 576-Position Calibration Scorer (`recalibration_diagnostic_scoring.py`)

- [x] **Step 1: Write failing tests (`test_recalibration_diagnostic_scoring.py`)**
- [x] **Step 2: Run focused RED and save log**
- [x] **Step 3: Implement `recalibration_diagnostic_scoring.py`**
- [x] **Step 4: Run focused GREEN and save log**

---

### Task 2: Remote Diagnostic Driver And Local Controller (`recalibration_diagnostic_remote.py`, `recalibration_diagnostic_controller.py`)

- [x] **Step 1: Write failing tests**
- [x] **Step 2: Run focused RED and save log**
- [x] **Step 3: Implement remote driver and local controller**
- [x] **Step 4: Run focused GREEN and independent review**
