# Qwen3.5 TP4 Root-Logit Remote Runner Implementation Plan

**Goal:** Add a dependency-light, fail-closed local runner for the frozen TP4
root-logit authority chain.

## Tasks

1. [x] Add dependency-light tests for constants, safe tags, exact-five inventory,
   SSH command construction, resource classification, safe tar extraction, and
   authority ordering.
2. [x] Run the tests and confirm RED because the runner module does not exist.
3. [x] Implement `tools/run_qwen35_tp4_real_root_logit_gate_remote.py` with
   `preflight`, `run`, `download-only`, `verify-only`, and `authority` modes.
4. [x] Run focused tests, `py_compile`, and `git diff --check`.
5. [x] Execute current-server `preflight`; require fail-closed classification while
   all GPUs have active compute processes.
6. [x] Add the copyable resume command and evidence path to
   `AGENT_HANDOFF_STATE.md`.
7. [x] Add a source-bound `native-smoke` mode that is preflight-first, launches
   no reference worker, validates the four frozen production native ranks, and
   publishes a non-authoritative three-file smoke result.
8. [x] Add RED/GREEN tests for native-smoke ordering, BLOCKED short-circuit,
   frozen source identity, reference exclusion, and artifact separation.
9. [x] Execute live BLOCKED validation and prove no remote smoke/work/publish
   path is created.
10. [x] Audit READY-path native child cleanup, reproduce the partial-start leak
    with a RED test, add bounded terminate/kill/reap cleanup, and pass the full
    remote preflight suite.
11. [x] Freeze the cleanup-hardened 80-file source bundle, migrate the runner
    identity, and repeat live BLOCKED native-smoke/authority validation.
12. [x] Replace sequential per-rank timeout budgets with one monotonic group
    deadline, pass RED/GREEN and the full remote preflight suite, then freeze
    and migrate the final source identity again.
13. [x] Remove the timeout branch's duplicate full-runtime reap waits, route
    timeout cleanup through the same bounded emergency cleanup, pass the full
    remote suite, and freeze the final active identity.

## Constraints

- Do not stage, commit, merge, or create a PR.
- Do not modify or overwrite the frozen source bundle.
- Do not kill remote processes or weaken the resource selector.
- Do not execute `run` unless preflight returns four eligible GPUs.
