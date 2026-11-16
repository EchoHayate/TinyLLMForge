# Full Tokenizer Audit Implementation Plan

**Goal:** Execute one fresh audit of the unchanged frozen 324-row inventory
with the reviewed tokenizer implementation and obtain a fully verified report.

**Architecture:** Add a manifest-bound optional `work_timeout_seconds` to the
existing launcher. Default remains 540 seconds; this run explicitly uses 1200.
The local absolute work deadline and the remote grant ceiling use the same
value. The cleanup allowance remains 60 seconds and is not proof of cleanup.
Do not alter tokenization, inventory, stop-on-overflow, or report semantics.

**Tech Stack:** Existing Python unittest; fixed remote Python 3.12.13 /
Transformers 4.57.1, same SSH target and five tokenizer file pins.

## Scope And Rationale

Latest user continuation follows the completed real tokenizer timing revision
and identifies full-inventory budget acceptance as the next step.
The reviewed short-input timing suggests approximately 13 minutes for 272
prompt-only and 52 answer rows, before additional overhead. That estimate is
not a completion guarantee. A new bounded 20-minute attempt avoids knowingly
using the old nine-minute ceiling; it does not replace the completed performance
fix or weaken the 4096 input / 256 target token constraints.

Rejected alternatives: silently reuse the consumed launch identity; keep the
known short deadline and hope; remove intermediate tokenizer checks; split the
inventory and treat partial sessions as a complete chained report.

No model weights, generation, training, dependency installation, library audit,
cleanup requirement, commit/push, branch/worktree changes or old-session resume.
No automatic remote retry. Preserve all prior failed and successful evidence.

## Steps

- [x] Verify current branch, reviewed source seal, credentials and frozen inputs.
- [x] Add RED tests for an explicit 1200-second absolute deadline, unchanged
  540-second default, invalid values, expiry/no renewal, manifest binding and
  local/remote timeout propagation.
- [x] Modify `tokenizer_launch_window.py`: keyword-only
  `LaunchWindow(work_timeout_seconds=540)`, exact integer in `[1, 1200]`.
  Work deadline = connection start + configured seconds; total adds 60 seconds.
- [x] Modify `tokenizer_host_remote.py`: validate the same integer parameter,
  pass it to `remaining_seconds` at both grant checks; default remains 540.
- [x] Modify `tokenizer_host_launch.py`: freeze the timeout in config, propagate
  it through `execute_once` and collector. Short fractional collector-only test
  timeouts remain supported; outer collector deadlines are never renewed.
- [x] Run focused and full regression; independently review only these changes.
- [x] Freeze a new profile/run ID and current source bundle with byte-identical
  inventory. Run exactly once with 1200 seconds and existing fixed SSH prefix.
- [x] Reconstruct the report from private records, verify all input/run/hash
  bindings, inspect stop/failure and coverage, then publish only sanitized counts.
- [x] Record result and remaining LoRA experimental prerequisites. Budget
  acceptance is not formal evaluation or evidence of LoRA benefit.

Prelaunch verification: 367 full-suite tests, 365 passed and two platform skips.
Independent review found a short-integer-budget accounting mismatch; the RED
test reproduced it, the collector now propagates integer budgets exactly, and
33 independently run rereview tests passed. Default/fallback remains 540.
The first prepared candidate was not executed. The reviewed candidate has run
ID `cadc2426b5c2499f8dbf30ba0dcfc8a1` and manifest SHA256
`e9d41022a9c490ccb4b7a19abc9d9e08d3d74495f5e863f0af6f1dc3aabdbb4e`.

Final result: one remote attempt completed in 709.736 seconds with SSH exit 0.
All 324 records passed token budgets; max input 2432, max target 104.
Full report reconstruction matched exactly and finalization covers 324 records.
No dynamic task execution, generation or training occurred.
Results: `experiments/lora_necessity/20261115-full-tokenizer-audit-results.md`.

## Evidence

Private root:
`/private/var/folders/0c/hl2r_50x28j2ysjvv2120t8h0000gn/T/tlf-tokenizer-full-20261115-8JzI7WOn`.
