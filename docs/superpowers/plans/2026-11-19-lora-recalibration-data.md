# LoRA Recalibration Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Materialize the approved twelve-task development dataset with manually derived cases, independently authored references, explicit incorrect-program witnesses and static review.

**Architecture:** Keep candidate-visible prompts separate from scoring material. Author cases before reference sources, seal reference B before exposing reference A, and preserve all v1 files. Embedded Python is experiment data, not host-executable application code.

**Tech Stack:** UTF-8 JSON/Markdown, Python standard-library JSON/AST/hash inspection only.

## Global Constraints

- User approved the written November 19 data design with “可以，继续吧”.
- Design: `docs/superpowers/specs/2026-11-19-lora-recalibration-data-design.md`, SHA256 `9e80f6b516577a4b2b3d19e24b1cb08c846d67fa0b681459845e0ffaa9186f10`.
- Work in the current `experiments/feedback-active-verification` checkout; no new branch, commit, push, installation or cleanup of existing untracked files.
- Do not execute/import/compile-to-run embedded task source, load a model/tokenizer, use SSH/GPU or resume an old session. AST parsing is permitted, not dynamic validation.
- No application runtime or unit-test suite is being added in this slice. Deliberately wrong programs are data; do not “repair” their specified defects. Validate their syntax and manually derive witnesses instead of running them.
- Preserve all 154 existing LoRA files and the reviewed design. The older 153-file tree excluding the November 19 design-review report has digest `980273dad4501834a1a312c15d31f781129e503dd096cc663aca54123ef0be6d`.
- Each task has P01/P02 plus H01–H06, eight distinct full argument vectors, two independent reference functions, one initial function, two incorrect variants and a rejection witness for each incorrect source.
- Keep the two approved public examples byte-value equivalent; private cases instantiate the six coverage requirements. Record handwritten derivations before sources.
- No inline source comments or single-letter variable names. Functions are standalone, use only deterministic builtins, and have the exact approved entry point/signature.
- No prior transfer/retention answers or model outcomes may inform new cases. Structural comparisons use S01–S08 and public D01–D04 only.
- Static, dynamic and tokenizer-length states are separate. A static pass never authorizes model execution or establishes LoRA benefit.

## Task 1: Cases and structural inventory

**Files:** private staging ` /private/tmp/lora-recalibration-v2-20261119/cases.json`; final `experiments/lora_necessity/data/recalibration_v2/structure.json`.

**Interfaces:** staging is a list of objects with `id` and `cases`; each case has `id`, `args`, `expected` (`kind=return`, JSON `value`), `derivation`, and `coverage`. It contains no source code.

- [x] Materialize all 24 approved public and 72 private cases, deriving every expected result without executing a task implementation.
- [x] Parse JSON, verify identifiers, argument counts, input domains and uniqueness. Inspect public examples against the approved design.
- [x] Record task mechanics and actual repair obligations. Enumerate 96 supervision pairs, 48 old-development pairs and 66 internal pairs with explicit reasons; retain near-template concerns instead of labeling every pair independent.
- [ ] Obtain independent case/structure review and resolve confirmed issues before marking static acceptance.

Review completed: R1 derivation corrected and closed. Structural acceptance is
not granted: C01:S01 and C04:S03 remain near-template blockers. This combined
acceptance checkbox intentionally stays open rather than claiming resolution.

## Task 2: Independent reference B

**Files:** `/private/tmp/lora-recalibration-v2-20261119/reference-b.json` and `reference-b-report.md`.

**Interfaces:** JSON list of `id`, `entry_point`, `source`, `decomposition`. Use the second decomposition named in each design section.

- [x] Give a fresh author only the design and source-free cases; prohibit reading reference A, initial or variant source.
- [x] Write twelve standalone reference functions using apply_patch.
- [x] Parse all twelve ASTs and check signatures, imports, comments and variable names. Do not invoke functions or evaluate source.
- [x] Seal and report reference-B SHA256 plus author identity and allowed inputs; independently review before integration.

## Task 3: Primary reference, incorrect sources and separation

**Files:** staging `primary-sources.json`; final `prompts.json` and `oracles.json` in `experiments/lora_necessity/data/recalibration_v2/`.

**Interfaces:**

- `prompts.json` is a list of `id`, `entry_point`, `spec`, `initial_source`, `public_cases`. Each public case exposes only `id`, `args`, `expected`.
- `oracles.json` is a list of `id`, `entry_point`, `cases`, `references`, `incorrect_programs`.
- Each reference is `id` (`reference_a`/`reference_b`), `source`, `decomposition`, `author`.
- Each incorrect program is `id` (`initial`/`variant_1`/`variant_2`), `source`, `defect`, `witness` (`case_id`, `predicted`, `reason`). Predicted outcome has `kind=return,value` or `kind=raises,exception`.
- `initial` source must be byte-identical between prompts and oracles. Reference-B sources must be byte-identical to the sealed independent file.

- [x] Author primary/reference-A, initial and two variant functions per task, using the first decomposition and specified defect families.
- [x] Derive each of the 36 witness outcomes by reading the actual authored source; initial witnesses must be public cases. Keep known execution errors distinct from semantic mismatches.
- [x] Seal primary sources before reading reference B. Then combine both without rewriting the independent implementation.
- [x] Integrate cases and sources into the two final JSON files with apply_patch; do not expose derivations, references, variant sources or private cases in prompts.
- [x] Run JSON/AST/signature/projection/witness-identity checks only. Independent review must consider actual code and hand calculations, not just author claims.

## Task 4: Acceptance, provenance and closeout

**Files:** `review.md`, `manifest.json`, `experiments/lora_necessity/20261119-recalibration-data-results.md`; task ledger `.superpowers/sdd/lora-recalibration-data-20261119.md`.

**Interfaces:** manifest binds the other four data files and original supervision/design by SHA256, records counts and independent authorship, and names `static_status`, `dynamic_status`, `tokenizer_length_status`, `run_authorized`, `benchmark_eligible`, `formal_evaluation_ready`, `net_benefit_established`, `isolated`.

- [x] Review the complete untracked file set rather than an empty HEAD diff; retain review findings and dispositions.
- [ ] Resolve blocking static findings, rerun only affected metadata checks, and re-review corrected material.
- [x] Recompute exact file hashes, counts, public/scoring separation and unchanged v1 baseline.
- [ ] Set completed static acceptance only if both structural and semantic reviews pass; dynamic and tokenizer states remain pending, all run/benefit/readiness flags false.
- [x] Save a compact report distinguishing 96 authored cases and 60 syntax-checked sources from the 480 future dynamic positions. No claim of dynamic correctness or model difficulty.

Closeout is a blocked static-review snapshot, not a successful whole-set
acceptance. Both near-template findings remain open; only C04's understated
classification was corrected. Source-defect redesign is a separate design
decision, not an unrecorded implementation change.

## Validation commands and boundaries

Use `python3 -m json.tool FILE` for syntax; use `ast.parse(source)` for each stored source string, never `exec`, `eval`, import of data source or loading it through a runner. Check exactly C01–C12, 24 public/72 private cases, 24 references/36 incorrect sources and 36 bound witnesses. Signature argument names/order must equal the design. Count comparisons are strict about JSON booleans versus integers.

The independent review must hand-trace every expected value and every named incorrect witness. Use scoped source reads and preserve reviewer identity. A second agent is not independent human sign-off.

No task ends in git commit despite generic skill templates. No new test framework, application validator or execution harness is part of this data-authoring plan.
