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

## Task 3: Engine Step Envelope and Conservation

- Status: completed
- Base SHA: `7de6f15f357c919185b1dfe5f810089da4b80cce`
- Brief: `.superpowers/sdd/task-3-brief.md`
- Required invariants:
  - fixed phase inventory with explicit skipped rows;
  - one active step and at most one active phase, with nested/repeated phases rejected;
  - active ContextVar identity integrates with Task 2 dispatch without circular-import masking;
  - existing scheduler/speculative/Proposal-KV/lifecycle/side-state operation order is unchanged;
  - step failures finalize telemetry without suppressing the original exception;
  - conservation uses `max(2_000_000 ns, 1% step wall)` and fails closed;
  - no completion fence or `torch.cuda.synchronize()` is added.
- Implementer: Codex
- Implementer report: `.superpowers/sdd/task-3-implementer-report.md`
- RED: `14 failed, 1 passed, 27 deselected in 0.21s`
- Review-fix RED cycles:
  - reset clears engine rows/repeat: `1 failed, 15 deselected in 0.07s`;
  - active reset rejects before dispatch:
    `1 failed, 16 deselected in 0.13s`;
  - missing command data fails closed:
    `1 failed, 1 passed, 16 deselected in 0.06s`;
  - finish-clock failure cleans ContextVar:
    `1 failed, 18 deselected in 0.07s`.
- Focused GREEN: `21 passed, 27 deselected in 0.20s`
- Planned focused regression: `208 passed in 5.76s`
- Task 1 + Task 2 regression: `56 passed in 1.29s`
- Exact inherited-fingerprint regression:
  `57 passed, 6 failed in 1.44s`
  - five failures stop at
    `ValueError: LLMEngine source hash is invalid`;
  - one failure reports the inherited prerequisite source-closure mismatch;
  - frozen source fingerprints were not rewritten.
- Syntax verification: PASS
- Implementer commit: `SELF/HEAD`
- Task review: `.superpowers/sdd/task-3-review-1.md` — Needs fixes
- Task re-review 1: `.superpowers/sdd/task-3-review-2.md` — Approved with Minor follow-up
- Task re-review 2: `.superpowers/sdd/task-3-review-3.md` — Approved
- Review-fix RED cycles:
  - operation exception plus phase-exit clock failure, and clock-only
    phase-exit failure:
    `2 failed, 19 deselected in 0.11s`;
  - failed-step observation ownership:
    `1 failed, 21 deselected in 0.08s`;
  - phase interval outside step envelope:
    `2 failed, 22 deselected in 0.07s`;
  - disabled hot path requests a phase context:
    `1 failed, 24 deselected in 0.10s`.
- Review-fix combined mandatory GREEN:
  `6 passed, 19 deselected in 0.08s`
- Review-fix focused GREEN:
  `27 passed, 27 deselected in 0.22s`
- Review-fix planned four-file regression:
  `214 passed in 2.15s`
- Review-fix Task 1 + Task 2 regression:
  `56 passed in 1.30s`
- Review-fix exact inherited-fingerprint regression:
  `57 passed, 6 failed in 1.48s`
  - five failures stop at
    `ValueError: LLMEngine source hash is invalid`;
  - one failure reports the inherited prerequisite source-closure mismatch;
  - frozen source fingerprints were not rewritten.
- Review-fix syntax verification: PASS
- Review-fix commit: `SELF/HEAD`
- Review-2 Minor RED:
  `1 failed, 29 deselected in 0.17s`
- Review-2 Minor targeted GREEN:
  `1 passed, 29 deselected in 0.08s`
- Review-2 Minor full Task 3 focused GREEN:
  `28 passed, 27 deselected in 0.25s`
- Review-2 Minor planned four-file regression:
  `215 passed in 7.02s`
- Review-2 Minor Task 1 + Task 2 regression:
  `56 passed in 1.28s`
- Review-2 Minor syntax verification: PASS
- Review-2 Minor commit: `SELF/HEAD`

## Task 4: Deferred CUDA Identity and Worker Export

- Status: Review-1 fix complete
- Base SHA: `8da831bb9d4622adc8bb0ed556ebd3f8830a5933`
- Brief: `.superpowers/sdd/task-4-brief.md`
- Required invariants:
  - CUDA rows bind exact command, engine-step, and repeat identity;
  - deferred profiler may reuse an existing synchronization and never adds a new measured-path fence;
  - timeline disabled preserves the current worker schema and defaults;
  - timeline enabled configures all ranks once, resets immediately before timing, and snapshots only after the existing final synchronization;
  - each measured repeat exports four rank command snapshots, four CUDA rank snapshots, and one engine-step snapshot;
  - canonical JSON is used for request and selection digests;
  - schema-v2 default warmup/measured counts remain unchanged outside the new diagnostic mode.
- Implementer: Codex
- Implementer report: `.superpowers/sdd/task-4-implementer-report.md`
- RED: `12 failed, 1 passed, 51 deselected in 0.25s`
- Review-fix RED cycles:
  - non-negative warmup timeline identity:
    `1 failed in 0.16s`;
  - disabled callback keyword compatibility:
    `1 failed in 0.11s`.
- Review-fix targeted GREEN: `2 passed in 0.05s`
- Focused GREEN: `13 passed, 51 deselected in 0.20s`
- Complete Task 4 regression: `117 passed in 2.87s`
- Task 1 + Task 2 regression: `56 passed in 1.46s`
- Task 3 four-file regression: `215 passed in 7.62s`
- Exact inherited-fingerprint regression:
  `57 passed, 6 failed in 1.92s`
  - five failures stop at
    `ValueError: LLMEngine source hash is invalid`;
  - one failure reports the inherited prerequisite source-closure mismatch;
  - frozen source fingerprints were not rewritten.
- Syntax verification: PASS
- Static no-new-sync/wait/fence check: PASS
- Implementer self-review: no remaining P0-P2 findings
- Implementer commit: `SELF/HEAD`
- Task review: `.superpowers/sdd/task-4-review-1.md` — Needs fixes
- Review-1 fix RED:
  `13 failed, 38 deselected in 0.28s`
  - every case failed with `DID NOT RAISE`, proving cardinality-only
    validation accepted empty, dropped, stale, malformed, or unrelated
    evidence.
- Review-1 fix focused GREEN:
  `28 passed, 51 deselected in 0.28s`
- Review-1 fix complete Task 4 regression:
  `132 passed in 2.99s`
- Review-1 fix Task 1 + Task 2 regression:
  `56 passed in 1.55s`
- Review-1 fix Task 3 four-file regression:
  `215 passed in 3.30s`
- Review-1 fix exact inherited-fingerprint regression:
  `57 passed, 6 failed in 1.86s`
  - five failures stop at
    `ValueError: LLMEngine source hash is invalid`;
  - one failure reports the inherited prerequisite source-closure mismatch;
  - frozen source fingerprints were not rewritten.
- Review-1 fix syntax verification: PASS
- Review-1 fix static no-new-sync/wait/fence check: PASS
- Review-1 fix self-review: no remaining P0-P2 findings
- Review-1 fix commit: `SELF/HEAD`

- Task 5: pending
- Task 6: pending
- Task 7: pending
- Task 8: pending
- Whole-branch review: pending
- Remote execution authorization: not granted
