# Tail Continuation Implementation Plan

> **For agentic workers:** Use subagent-driven-development for the pure reporter;
> main owns the coupled private driver and all remote calls. Do not commit.

**Goal:** Execute only328 authorized tail requests once, preserving old failure.
**Architecture:** Unchanged remote backend; separate one-shot driver and pure
cross-stage aggregate that retains both attempt history and selected observations.
**Tech Stack:** Python3.12 standard-library unittest, existing SSH/bwrap collector.

## Global Constraints

Binding spec: ../specs/2026-11-15-lora-tail-validation-design.md.
No training/model loading/generation, tokenizer rerun, dependencies, telemetry,
Git publication, branches or worktrees. Work in the existing experiment checkout.
Correct answers and witness predictions stay local. Remote request fields remain
exactly request_id/source/entry_point/args. Only main performs remote calls.
No automatic retry, no replay of old stage or completed1088. New deadline900s;
item2s, batch90s, SSH110s, stdout8388608/stderr131072.
Preserve old sealed artifacts and frozen data/backend/worker/reporters.

## Task 1: Pure continuation report (independent implementer)

Files: create experiments/lora_necessity/validation_tail.py and
experiments/lora_necessity/test_validation_tail.py.
Requirements/API/output exactly as in the Pure reporter section of the spec.

- [x] First add failing existence/import assertion and run it:
  `self.assertTrue((ROOT / "validation_tail.py").is_file())`.
- [x] Add strict aggregate_tail with no executor imports. Its composition uses
  `prior = aggregate_matrix(..., previous_records, ...)` and
  `combined = aggregate_matrix(..., previous_records[:-1] + tail_records, ...)`
  only after validation; preserve prior, auth and selected batch stage IDs.
- [x] Synthetic fixture uses ValidationMatrixTests.setUpClass without invoking
  frozen source. Make a distinct failed prior start1088 bundle/run and transport;
  preserve complete synthetic tail to reach1416. Distinguish all IDs.
- [x] Test full/partial, authority exact fields/types/hash, reused failed ID,
  wrong prior count/start/partial observations, tail gap/duplicate/after-stop,
  runtime drift, final8 and T06 cross-stage, negative reference and witness.
- [x] Run focused unittest with Python3.12 -B and PYTHONDONTWRITEBYTECODE=1;
  save RED/GREEN output and report in new private root/reporter-task.md.
- [x] Independent spec/quality review before remote integration approval.

## Task 2: Private one-shot driver (main)

Files: new private root/tail.py and test_tail_driver.py. Do not edit old matrix.py.

- [x] RED tests for prepare(old_root)/execute(stage), exact eleven bundles,
  consumed stage refusal, source/bundle/approval drift before collector, first
  transport failure stopping, missing done stopping, malformed-output preservation,
  collector exception preservation, synthetic complete328 and1416 combined.
- [x] Implement prepare snapshots and strict bindings per spec. Use `open("xb")`
  for claims/artifacts and uuid4 hex for new stage/run IDs.
- [x] Implement sequential collector with inherited exact approved SSH prefix;
  save raw before report; aggregate_tail verifies prior and all tail observations.
  Stop on incomplete or exception; terminal summary always retains strict prefix.
- [x] GREEN focused tests; run full repository experiment unittest once.
- [x] Generate four-file review diff/hashes and send final integrated review.

## Task 3: Execute and close

- [x] Verify old seal, current hashes and approval; prepare one fresh stage.
- [x] Execute11 ordered batches once after independent review approval.
- [x] Rebuild all raw reports and combined result without source execution.
- [x] Independent evidence review; update README/new dated report/progress,
  seal new artifacts separately. No publication or further task expansion.

## Terminal Outcome

Completed once:328 new normal observations,1416 total,0 missing. All354
references,132 designated witnesses and44 task checks passed. Independent raw
reconstruction accepted this data result; the one README status finding was
fixed and closed in R2. Original failure and prior377 private files remain
unchanged. The new private final-verification.json is the closeout inventory.
No training/model generation, benchmark admission or LoRA benefit was established.
