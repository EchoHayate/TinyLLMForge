# Development coordinator qualification

Date: 2026-11-17. Component accepted; real development run not performed here.

`development_coordinator.py` connects the fixed schedule, observed generation,
public feedback, final execution and scoring into eight serial episodes:
D01 base/text, D02 text/base, D03 base/text, D04 text/base.
Each episode uses both generation rounds, even after first-round public success.
Only the original public feedback enters the second prompt. Final scoring uses
the saved second-generation bytes, never the better of two answers.

## What is verified

- Exclusive records bind planned IDs, callback entry, saved claims, observed
  forwards, generation completion and candidate-child evidence separately.
- Failure preserves the original exception, readable evidence, completed work
  and all missing positions. No automatic retries or replacement IDs.
- Unknown costs stay null. Nested generation, observer, execution and outer
  times remain separate rather than being summed twice.
- The final case inventory stays fixed. Unexecuted means no callback entered;
  an entered call without a receipt is not assumed to have done no work.
- First-round invalidity survives a later successful correction. Functional
  scores and development calibration are distinct.
- Post-publication temporary-file cleanup failure does not reverse a committed
  success or trigger another call.

## Review and tests

The first independent review found two defects: finite execution durations
could overflow their aggregate and prevent failure-report publication; failed
audits could contain contradictory input-completion and byte-count fields.
Both were reproduced and fixed, then independently re-reviewed as closed.

- Repair RED: 7 methods, 12 assertion/subtest failures.
- Final coordinator suite: 105 tests passed.
- Author's named pure regression: 205 tests passed.
- Main's independent named regression: 332 tests passed in 18.219 seconds,
  exit 0, with no warnings or traceback noise.
- Independent specification and scoped quality review: PASS after repair.

Counts overlap; they are not additive distinct coverage. Tests use synthetic
task bytes and artifact-writing callbacks, not a model or evaluated candidate.
The all-pass fixture completes eight episodes, sixteen generations, sixteen
public receipts and twenty-four final receipts. Those final-case counts belong
to the synthetic fixture, not the frozen real tasks.

## Failure-report contract

`costs.execution_elapsed_seconds.aggregation_error` is normally null. If finite
individual durations cannot be represented as a finite aggregate, the original
values remain available while total/subtotal become null with a diagnostic;
calibration is incomplete, not a fabricated model failure.

If the report reducer itself fails, the degraded report retains all eight
episode ledgers and the original cause, but `counts`, `costs` and `by_arm` are
null. Its `report_reduction_failed` calibration is not a normal complete
calibration result. Consumers must use the retained evidence, not interpret
null summaries as zero work.

## Identities and remaining work

```
development_coordinator.py
3d03838977ddd20982db180393557e040ac38f6261f30c6a92496b4257633623
test_development_coordinator.py
38c7e62e8b6b77050243b0a86e40f1a5c7979b4958a8ba026450931b03e926ff
```

Private evidence under `/tmp/tlf-candidate-executor-20261117-gb1gZX`:
`coordinator-runtime-report.md`, `coordinator-runtime-review.md`,
`coordinator-runtime-review-r2.md`, and
`coordinator-runtime-main-r2-combined.log`. Original review findings remain
preserved alongside their closure.

Actual callback provenance, model/tokenizer integrity, remote isolation,
transport and raw-result reconstruction remain the remote driver's integration
work. Coordinator bindings remain caller declarations and all scientific
readiness flags remain false. Neither these tests nor development calibration
prove LoRA effectiveness or authorize automatic transfer/retention evaluation.
