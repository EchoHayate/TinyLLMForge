# Development final-scoring component

Date: 2026-11-17. This report covers a pure planner/reducer, not real task results.

## Delivered

`development_final_scoring.py` rebuilds the final public-plus-scoring case union
from the original pinned task and manifest bytes. It binds the exact second-round
generation bytes, unchanged candidate source, complete case set and per-case
request identities. It never executes reference or candidate code.

The reducer reconstructs its own plan rather than accepting a caller-edited
case list. Only an ordered receipt prefix is valid. Missing cases remain missing;
invalid source retains the full scheduled denominator with zero executions.
Functional pass requires every public and independent scoring case to pass.

Expected outcomes remain in the trusted-side plan, not in worker payloads or
score output. Canonical comparison distinguishes bool, int, float and string.
Completed ValueError outcomes, mismatches, program errors, timeouts and output
limits remain distinct. Admission/infrastructure errors are not model feedback.

## Verification

- Focused synthetic suite: 44 passed.
- Final protocol plus scoring: 72 passed.
- Main's named old/new protocol, admission, controller and scoring aggregate:
  186 passed in 5.680 seconds, exit 0.
- Independent static specification and quality review: PASS, no concrete
  Critical/Important defect found.
- All 16 source/dependency/spec/plan/log hashes in the author report match.

Tests build synthetic 44-task metadata and use the real frozen loader with
test-scoped pins. They do not read the real private task bundle. Coverage includes
complete case binding, raw/candidate hashes, malformed and reordered receipts,
invalid/missing denominators, outcome byte/depth limits, answer-free projections,
and restoration of shared pins.

The initial RED was one missing-module import-discovery error: no test-body
assertions ran. That limitation is retained, not rewritten as behavioral RED.
A later missing-case-group regression genuinely reproduced `KeyError` and passed
after the validation fix. GREEN results do not erase the initial limitation.

## Reproduce

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lora_necessity \
  /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error -m unittest \
  test_candidate_execution_protocol test_candidate_execution_admission \
  test_candidate_execution_controller test_development_final_protocol \
  test_development_final_admission test_development_final_controller \
  test_development_final_scoring -q
```

Source SHA256:

```
development_final_scoring.py
29244847a45d6b29228306632eee1a51eb19a1263f5ff86dafbe275d0bdf2aab
test_development_final_scoring.py
a41b331c6b4a358d0912ac357255331c18bf4e7281e7fc005603cc7910cf5b4f
```

Private evidence under `/tmp/tlf-candidate-executor-20261117-gb1gZX`:
`development-final-scoring-report.md`, `development-final-scoring-review.md`,
`development-final-scoring-green.log`,
`development-final-scoring-aggregate-green.log` and
`development-final-scoring-safe-combined.log`.

## Boundaries

Byte hashes bind records; they do not authenticate their origin. The coordinator
still must link the real generation claim/observation and model/tokenizer identity
to final requests, and reconstruct execution lifecycle/receipts from trusted
artifacts. This component does not authorize candidate execution.

No real D-case count, development score, calibration outcome or LoRA gain was
measured here. All scientific/readiness/isolation flags remain false, including
on a synthetic functional pass.
