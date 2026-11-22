# Development schedule qualification

Date: 2026-11-17. Pure scheduling component; no model or candidate execution.

`development_schedule.py` freezes eight episodes in this order:
D01 base/text, D02 text/base, D03 base/text, D04 text/base.
It preserves the original renderer's messages and eight-example text arm,
assigns sixteen generation IDs and eight public-attempt IDs, and binds all
four frozen data identities.

The scheduler checks agreement between public and private D-task projections.
The trusted-side final inventory contains only ordered case identities and
input/expected hashes, not private values, reasons or reference implementations.
Logical IDs do not replace durable claims or authenticate actual execution.

## Verification

- Missing-component RED: 28 explicit `setUp` assertion failures before source
  creation; not 28 test-method-body failures.
- Focused GREEN: 28 passed.
- Protocol/scoring/schedule aggregate: 100 passed.
- Main named eight-module aggregate: 214 passed in 5.756 seconds, exit 0.
- Independent specification and scoped quality review: PASS.
- All 13 source/dependency/brief/log hashes reported by the author match.

Tests use synthetic views and the real frozen loaders/renderer, restoring all
four shared pins after each test. Coverage includes exact order and digests,
public/private drift, full case inventory, no aliases, no private data leakage,
and cold imports without model or execution modules.

The review notes a minor coverage limit: the collision test does not isolate a
cross-purpose collision. The implementation checks all 24 IDs in one shared set.
This is not represented as an additional independently exercised test case.

## Identities and evidence

```
development_schedule.py
f7785d194d3e9adfb986ab627fd5b5e968c9bb42f44e40e3a07ee33a04bc8bb1
test_development_schedule.py
721bbfc6bb5fcf0338327a549666d298e630b72364b20eca4fb518e49d48b3c1
```

Private reports/logs are under `/tmp/tlf-candidate-executor-20261117-gb1gZX`:
`development-schedule-report.md`, `development-schedule-review.md`,
`development-schedule-green.log`, `development-schedule-aggregate-green.log`
and `development-schedule-safe-combined.log`.

Eight episodes and sixteen calls are planned, not observed. Coordinator
persistence, two-round execution, failure accounting, calibration and the
new remote run remain separate work. No LoRA-effect conclusion follows.
