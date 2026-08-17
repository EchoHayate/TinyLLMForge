# Qwen3.5 W2 Short-Output Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a guarded TP4 eight-token `w2_long_reuse` diagnostic that determines whether long decode diluted the measured prefix-reuse benefit.

**Architecture:** Add a profile-only generated-token override to the existing worker, propagate one copied effective workload payload through engine execution and validation, and record explicit experimental metadata in the separate profile artifact. Extend the focused aggregator to validate and report the effective output length without changing canonical benchmark rows or manifests.

**Tech Stack:** Python, pytest, TinyLLMForge TP4 engine, JSON artifacts, SSH remote execution.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not modify the canonical workload manifest, case matrix, case-row schema, or existing r607/r608/r609 artifacts.
- Use fixed GPUs `2,4,5,6`.
- Require at least 25 GiB free and at most 10 percent utilization per GPU.
- Record the run as shared-low-utilization and non-exclusive.
- Do not use dummy reservations or kill unrelated processes.
- Use a new run tag and attempt-scoped cleanup.
- Do not stage, commit, stash, reset, push, or run `git clean`.

---

### Task 1: Profile-Only Generated-Token Override

**Files:**
- Modify: `tools/qwen35_tp4_hybrid_prefix_benchmark_worker.py`
- Modify: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_worker.py`

**Interfaces:**
- Consumes: canonical `w2_long_reuse` payload.
- Produces: `--generated-tokens-override 8` accepted only with `--profile`.
- Produces: an effective workload payload used by configuration, execution,
  validation, and profile metadata.

- [x] Add a failing test that requests an eight-token override and expects the
  fake engine to receive `spec.generated_tokens == 8`.
- [x] Add failing tests that reject the override without profiling, on a
  non-`w2_long_reuse` workload, at zero, and above the canonical count.
- [x] Run the focused worker tests and confirm failures are caused by the
  missing override interface.
- [x] Implement the minimal override validation and copied effective payload.
- [x] Validate requests against the effective generated-token count.
- [x] Record `variant`, `canonical_generated_tokens`, and `generated_tokens`
  in `profile.json`.
- [x] Re-run the worker tests.

### Task 2: Eight-Token Profile Aggregation

**Files:**
- Modify: `tools/qwen35_tp4_w2_restore_profile.py`
- Modify: `tools/test_qwen35_tp4_w2_restore_profile.py`

**Interfaces:**
- Consumes: ten measured profile cases with explicit generated-token metadata.
- Produces: summary fields for generated-token count, paired makespan ratios,
  ratio-of-medians, and stable-direction classification.

- [x] Add failing tests for eight-token metadata validation.
- [x] Add a failing test that rejects a row with a mismatched generated-token
  count.
- [x] Add a failing test for ratio-of-medians and direction-agreement output.
- [x] Run focused aggregation tests and confirm RED.
- [x] Implement generated-token validation and comparison fields.
- [x] Preserve compatibility with the existing 64-token r609 artifact.
- [x] Re-run aggregation and worker tests.

### Task 3: Guarded Remote Experiment

**Files:**
- Create a new directory under:
  `experiments/qwen35_hybrid_state/qwen35-tp4-w2-short-output-profile-20260811-<tag>-attempt001/`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Runs: one paired warmup plus five measured pairs.
- Produces: downloaded case artifacts, `profile_summary.json`, resource guards,
  case receipts, and cleanup receipt.

- [x] Package the current source tree under a new immutable run tag.
- [x] Run entry and worker-entry resource guards for GPUs `2,4,5,6`.
- [x] If either guard is not `READY`, preserve the attempt and launch no worker.
- [x] Run 12 workers with `--profile --generated-tokens-override 8`.
- [x] Download artifacts and aggregate with `--generated-tokens 8`.
- [x] Verify 20/20 continuation token parity.
- [x] Perform attempt-scoped cleanup and require `CLEAN`.
- [x] Append commands, hashes, measurements, evidence boundaries, and the
  conclusion to `AGENT_HANDOFF_STATE.md`.

### Completion Gate

- [x] Five measured pairs are present.
- [x] Every continuation generated exactly eight tokens.
- [x] Output-token parity is 20/20.
- [x] Reuse and executed-prefill counts match the 64-token experiment.
- [x] Decode share is compared directly with 84.722 percent.
- [x] Both paired-median and ratio-of-medians makespan summaries are reported.
- [x] Cleanup is `CLEAN`.
- [x] Focused tests, `py_compile`, artifact regeneration, and
  `git diff --check` pass.

