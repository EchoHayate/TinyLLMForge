# Autoregressive Draft Command Timeline SDD Progress

- Plan: `docs/superpowers/plans/2026-08-18-autoregressive-draft-command-timeline-sync-debt.md`
- Authoritative checkout: `/Users/bytedance/Desktop/TinyLLMForge`
- Branch: `feat/kv-sparse-attention`
- Starting base SHA: `c5aa2d7d3c8050c61cd1b0ce735bd717ba862fd3`
- Baseline verification: `64 passed in 1.34s`
- Execution mode: subagent-driven development, sequential tasks

## Task 1: Default-Off Command Timeline Core

- Status: completed
- Base SHA: `c5aa2d7d3c8050c61cd1b0ce735bd717ba862fd3`
- Brief: `.superpowers/sdd/task-1-brief.md`
- Required clarification: `CommandClockIdentity` must record a wall-clock capture field named `captured_at_unix_ns`, sourced from `time.time_ns()`, in addition to monotonic/boot identity.
- Required invariants:
  - timeline remains default-off;
  - ordinary commands remain `requires_ack=False`;
  - no completion fence or `torch.cuda.synchronize()` is added to the measured request path;
  - implementation follows RED, GREEN, focused verification, commit, and report.
- Implementer: Codex
- Implementer report: `.superpowers/sdd/task-1-implementer-report.md`
- Implementer commit: `SELF/HEAD`
- Task review: `.superpowers/sdd/task-1-review-1.md` — Needs fixes
- Task re-review: `.superpowers/sdd/task-1-review-2.md` — Approved
- Review fix RED: `2 failed, 9 deselected in 0.06s`
- Review fix focused GREEN: `2 passed, 9 deselected in 0.04s`
- Review fix Task 1 GREEN: `12 passed, 19 deselected in 0.09s`
- Review fix regression: `59 passed in 0.86s`
- Review fix commit: `SELF/HEAD`

## Task 2: Shared-Memory Command and Ack-Wait Wiring

- Status: completed
- Base SHA: `13c0c017222d65a252e7b1a019510b4413520963`
- Brief: `.superpowers/sdd/task-2-brief.md`
- Required invariants:
  - trace identity is stamped only while the timeline is enabled and an engine step/repeat context is active;
  - the final publish timestamp is serialized before worker events are set;
  - rank zero and workers share the same command ID and trace identity;
  - TP1 remains local-only with no worker ack wait;
  - ordinary `call()` commands remain `requires_ack=False`;
  - timeline management commands use acknowledged all-rank calls and do not contaminate measured snapshots;
  - no completion fence or `torch.cuda.synchronize()` is added.
- Implementer: Codex
- Implementer report: `.superpowers/sdd/task-2-implementer-report.md`
- RED: `5 failed, 30 deselected in 0.21s`
- TP1 lifecycle RED: `1 failed, 19 deselected in 0.13s`
- Focused GREEN: `7 passed, 30 deselected in 0.25s`
- Task 1 + Task 2 focused regression: `48 passed in 0.94s`
- Exact planned regression: `49 passed, 6 failed in 1.16s`
  - all six failures are the known frozen
    `test_qwen35_real_binding_engine_ack_transport_preflight.py`
    source-fingerprint family;
  - the prerequisite closure had 23 mismatches at the Task 2 base SHA and
    still has the same 23 mismatches after Task 2;
  - immutable fingerprint expectations were not rewritten.
- Syntax verification: PASS
- Initial implementer commit:
  `e2479a3ad3adff6375c44de214653b3b0c1549e8`
- Task review: `.superpowers/sdd/task-2-review-1.md` — Needs fixes
- Task re-review 1: `.superpowers/sdd/task-2-review-2.md` — Needs fixes
- Task re-review 2: `.superpowers/sdd/task-2-review-3.md` — Approved
- Review fix RED: `6 failed in 0.37s`
- Review fix focused GREEN: `6 passed in 0.26s`
- Review fix Task 1 + Task 2 regression: `54 passed in 1.11s`
- Review fix exact planned regression: `55 passed, 6 failed in 1.32s`
  - all six failures remain in the known frozen
    `test_qwen35_real_binding_engine_ack_transport_preflight.py`
    source-fingerprint family;
  - five failures stop at
    `ValueError: LLMEngine source hash is invalid`;
  - one failure reports the inherited prerequisite source-closure mismatch;
  - immutable fingerprint expectations were not rewritten.
- Review fix syntax verification: PASS
- Review fix commit: `SELF/HEAD`
- Second-review fix RED: `2 failed, 24 deselected in 0.24s`
- Second-review fix focused GREEN:
  `2 passed, 24 deselected in 0.17s`
- Second-review fix live-wiring GREEN: `26 passed in 0.75s`
- Second-review fix Task 1 + Task 2 regression:
  `56 passed in 1.27s`
- Second-review fix exact planned regression:
  `57 passed, 6 failed in 1.49s`
  - all six failures remain in the known frozen
    `test_qwen35_real_binding_engine_ack_transport_preflight.py`
    source-fingerprint family;
  - five failures stop at
    `ValueError: LLMEngine source hash is invalid`;
  - one failure reports the inherited prerequisite source-closure mismatch;
  - immutable fingerprint expectations were not rewritten.
- Second-review fix syntax verification: PASS
- Second-review fix commit: `SELF/HEAD`

## Remaining Tasks

- Task 3: pending
- Task 4: pending
- Task 5: pending
- Task 6: pending
- Task 7: pending
- Task 8: pending
- Whole-branch review: pending
- Remote execution authorization: not granted
