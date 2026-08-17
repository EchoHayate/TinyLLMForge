# Qwen3.5 W2 Restore Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a focused TP4 `w2_long_reuse` timing artifact that explains where the 15,360-token reuse saving is spent without rerunning the 70-case benchmark.

**Architecture:** Add opt-in profiling to the benchmark engine adapter rather than changing the canonical benchmark row schema. The adapter wraps synchronous engine restore boundaries and records request admission, restore transaction phases, first-token latency, and decode time into a separate profile payload. A dedicated focused runner executes only `w2_long_reuse` recompute and exact-restore cases and preserves the existing resource guard and cleanup rules.

**Tech Stack:** Python, pytest, TinyLLMForge TP4 engine, JSON artifacts, existing remote execution helpers.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not modify the canonical benchmark `CASE_ROW_FIELDS` or the existing `r607` artifact.
- Use fixed GPUs `2,4,5,6`.
- Require at least 25 GiB free memory and at most 10% utilization per GPU; unrelated low-utilization processes are allowed and must be recorded.
- Do not use dummy GPU reservations and do not kill unrelated processes.
- Use a new run tag and preserve all previous experiment directories.
- Cleanup must remain attempt-scoped.
- Do not stage, commit, stash, reset, push, or run `git clean`.

---

### Task 1: Opt-In Restore Timing Collector

**Files:**
- Modify: `tools/qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py`
- Test: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py`

**Interfaces:**
- Produces: `BenchmarkEngineAdapter.profile_snapshot() -> dict`
- Produces profile events with `name`, `duration_ns`, `request_id`, and optional `operation`.
- Wraps: `flush_pending_hybrid_state_releases`, `acquire_qwen35_hybrid_prefix`, `prepare_model_runner_hybrid_prefix_restore`, `validate_model_runner_hybrid_prefix_restore`, `commit_model_runner_hybrid_prefix_restore`, and `rollback_model_runner_hybrid_prefix_restore`.

- [ ] Write a failing adapter test that enables profiling, admits an exact-restore request, and expects nested phase events with non-negative durations.
- [ ] Run `python3 -m pytest -q tools/test_qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py` and confirm the new test fails because `profile_snapshot` is absent.
- [ ] Implement an opt-in collector using the adapter clock and instance-local method wrappers; leave behavior unchanged when profiling is disabled.
- [ ] Record per-request `admission_ns`, `ttft_ns`, `decode_ns`, and `e2e_ns` from the existing lifecycle timestamps.
- [ ] Re-run the focused test and the complete adapter test file.

### Task 2: Focused W2 Profile Artifact

**Files:**
- Create: `tools/qwen35_tp4_w2_restore_profile.py`
- Create: `tools/test_qwen35_tp4_w2_restore_profile.py`

**Interfaces:**
- Consumes: canonical `w2_long_reuse` workload payload and the adapter profiling API.
- Produces: `profile.json` with schema `qwen35.tp4-w2-restore-profile.v1`.
- Produces paired summaries for `recompute` and `exact_restore`.

- [ ] Write failing tests for profile schema validation and aggregation of five paired repetitions.
- [ ] Verify RED with `python3 -m pytest -q tools/test_qwen35_tp4_w2_restore_profile.py`.
- [ ] Implement aggregation for median and per-repetition values of restore phases, TTFT, decode, makespan, executed prefill tokens, and reused KV tokens.
- [ ] Include an explicit evidence boundary: prepare duration combines rank work and acknowledgement transport until rank-level instrumentation is added.
- [ ] Re-run both focused test files and `python3 -m py_compile` for the new/modified tools.

### Task 3: Focused Remote Execution

**Files:**
- Create: `tools/run_qwen35_tp4_w2_restore_profile_remote.py`
- Test: `tools/test_run_qwen35_tp4_w2_restore_profile_remote.py`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Uses the existing SSH/control-master path and resource guard.
- Runs only warmup plus five measured paired `w2_long_reuse` cases.
- Writes a new experiment directory and cleanup receipt.

- [ ] Write failing tests for fixed GPU selection, shared-low-utilization resource metadata, unique run tags, exact case inventory, and scoped cleanup.
- [ ] Implement the remote runner by reusing existing transport and guard helpers.
- [ ] Run the focused runner tests and existing remote-runner regression tests.
- [ ] Execute one guarded remote profile run on GPUs `2,4,5,6`; if the resource gate is not READY, stop without launching workers and preserve the gate artifact.
- [ ] Verify the downloaded artifact, calculate the dominant phase, and distinguish measured evidence from inference.
- [ ] Append commands, hashes, timing results, limitations, cleanup state, and next action to `AGENT_HANDOFF_STATE.md`.

### Completion Gate

- [ ] The profile artifact contains five paired measured repetitions.
- [ ] Exact restore reports `15,360` reused KV tokens and `256` executed prefill tokens per four-request case.
- [ ] Restore transaction, TTFT, decode, and makespan are separately reported.
- [ ] The report does not claim prepare-internal communication versus rank-local copy attribution unless rank-level evidence exists.
- [ ] Remote cleanup is `CLEAN` with no attempt-scoped PIDs remaining.
- [ ] Focused tests, relevant regressions, `py_compile`, and `git diff --check` pass.
