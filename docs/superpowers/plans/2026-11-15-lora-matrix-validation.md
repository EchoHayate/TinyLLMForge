# Remaining Matrix Validation Implementation Plan

> **For agentic workers:** Use subagent-driven-development for the pure reporter and integrated requesting-code-review; main owns the coupled backend/driver and remote calls.

**Goal:** Validate remaining1384 frozen calls once and combine with accepted D01.
**Architecture:** Existing remote worker, bounded32-row chunks, pure local full matrix report.
**Tech Stack:** Python3.12 standard library/unittest, existing SSH/bwrap/libseccomp.

## Global Constraints

Binding design: `../specs/2026-11-15-lora-matrix-validation-design.md`.
Do not change frozen data/worker/D01 reporter. No training/tokenizer/install/
telemetry/Git publication/worktrees. Never rerun D01 or retry consumed identities.
900s stage,90s batch,2s item; existing IO limits remain. Answers stay local.

## Task 1: Backend and driver (main)

- [x] RED: actual run_bundle synthetic execution preserves raw formatted bundle
  digest; matrix runtime absence fails before collector;1/8/32 accepted,33 rejected.
- [x] Fix raw digest and add matrix mode/runtime gate. Exact changed expression:
  `hashlib.sha256(bundle_raw).hexdigest()`.
- [x] GREEN focused backend tests and unchanged D01 tests.
- [x] Private driver prepare/execute with44 frozen chunks, new exclusive stage
  and per-batch claims, exact source/bundle/runtime/D01 bindings,900s deadline.
- [x] Driver negative tests: changed bytes and existing claim prohibit collector,
  infrastructure failure stops next batch; use synthetic collectors only.

## Task 2: Pure reporter (independent owner)

Create `experiments/lora_necessity/validation_matrix.py` and
`test_validation_matrix.py`. Read exact signatures/schemas in binding design.

- [x] RED before implementation:
  `self.assertTrue((ROOT / "validation_matrix.py").is_file())`.
- [x] Implement batch_ranges, build_batch_report, aggregate_matrix using strict
  frozen projections and raw receipts, with no execution imports.
- [x] Synthetic tests:32 and final8, nonzero global ordinals, missing rows,
  duplicate/out-of-order/wrong hashes/extra fields, runtime mismatch,
  stopped stream, failed transport, duplicate batch/run, gap, and after-failure.
- [x] Synthetic full1416 report:44 tasks/354 references/132 witnesses; flip
  reference, initial scoring and witness independently to reject; flags stayfalse.
- [x] GREEN focused unittest; unchanged D01 report tests continue passing.

Command for each focused file:
`PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -m unittest discover -s experiments/lora_necessity -p 'test_validation_matrix.py' -q`

## Task 3: Review, execute and close

- [x] Review full additions and backend diff, driver and report integration;
  fix material findings with focused RED/GREEN, preserve previous evidence.
- [x] Full directory unittest, final new-backend artificial probes, source hashes.
- [x] Verify new profile and previous D01 seal, prepare44 bundles, execute once.
  Attempt terminated incomplete on batch34 at jump authentication:1056 new
  received plus32 D01,328 missing. This checkbox records the bounded attempt,
  not completion of the scientific/data-validation goal; do not replay it.
- [x] Pure report/summary reconstruction and independent evidence review.
- [x] Update README/results/progress and seal new artifacts. No publication.

## Terminal Outcome

Partial evidence accepted, not full matrix acceptance. The failed stage is
permanently incomplete:1088/1416 observed,328 missing. Independent review found
no P0/P1/P2 and rebuilt all34 attempted batch reports and aggregate prefixes.
Closeout suite442 tests:440 passed,2 skipped. Evidence inventory is recorded in
the private root's final-verification.json.

Future tail scope must distinguish32 already-attempted rows from296 rows without
claims. Fresh identities alone cannot waive no-retry; the former need explicit
resubmission approval. No continuation execution or training occurred here.
