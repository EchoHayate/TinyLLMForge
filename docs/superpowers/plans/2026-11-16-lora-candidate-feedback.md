# Candidate Feedback Implementation Plan

> **For agentic workers:** Implement in the existing experimental checkout, with test-driven-development and independent review. No old agent/session resume.

**Goal:** Bind generated source and public-only feedback without executing candidates.

**Architecture:** Pure functions read pinned prompts, validate source syntax and construct answer-free requests. A reducer consumes trusted executor receipts; the existing renderer builds the second prompt.

**Tech Stack:** Python standard library, existing unittest runner.

## Global Constraints

- Follow `docs/superpowers/specs/2026-11-16-lora-candidate-feedback-design.md`.
- No candidate execution, hidden scoring access, training, install, telemetry, commit or push.
- Preserve all prior files; existing experimental branch stays unchanged.
- No added aggregate quotas or automatic disconnect cleanup gate.

## Task 1: Candidate and feedback contract

Create `experiments/lora_necessity/candidate_feedback.py` and
`experiments/lora_necessity/test_candidate_feedback.py`.

- [x] Snapshot pre-existing experiment/docs files and previous private stage artifacts.
- [x] Write failing tests using import-after-file-existence assertion, as in test_prompt_rendering.
- [x] Run RED: `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -m unittest discover -s experiments/lora_necessity -p 'test_candidate_feedback.py' -q`.
- [x] Implement the three exact signatures in the design; use ast.parse/compile without exec, frozen load, strict parse_json, fields and digest. No edits to existing modules.
- [x] Run GREEN with the same command; confirm requests omit expected values, receipt corruption raises ValueError and second_round accepts reduced feedback.
- [x] Independent reviewer checks source/test/design files, without candidate execution or hidden data.
- [x] Run complete existing local suite with `-p 'test_*.py'`; report pass/skip counts, not model efficacy.
- [x] Verify snapshot preservation and record runtime observations and result limits.

Example caller shape, using separately admitted executor receipts:

```python
requests = visible_requests(prompts_raw, "D01", "base", candidate,
                            attempt_id=attempt_id)
feedback = visible_feedback(prompts_raw, "D01", "base", candidate,
                            receipts_raw, attempt_id=attempt_id)
messages = second_round(supervision_raw, prompts_raw, "D01", "base",
                        candidate, feedback)
```

This plan intentionally supplies no executor or launch command.

Completion: 17 RED then 17 GREEN; independent Spec PASS/component approval.
Full regression: 583 run, 581 passed, 2 skipped, exit 0 in 220.111 seconds.
1100 old file hashes preserved. Real candidate execution remains blocked, not
completed by this component. No commit/push or old-session resume.
