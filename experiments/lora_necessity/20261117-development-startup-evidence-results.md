# Development startup-evidence qualification

Date: 2026-11-17. Pure diagnostic validation; no new remote execution.

`development_startup_evidence.py` checks the fixed four-line pre-input stdout:
the three ordered launcher/worker markers and the strict diagnostic JSON.
It requires exactly 17 successful checks, five zero capability sets, the
41 ordered syscalls, Python 3.12.13, and both pre-input execution flags false.
The original parsed record is returned without changing its meaning.

## Verification

- Behavioral RED: 12 methods with 160 assertion/subtest failures against a
  temporary nonfunctional implementation, retained separately from the initial
  missing-file checks.
- Final component/protocol GREEN: 41 tests passed.
- Fresh Main component check: 13 tests passed in 0.006 seconds, exit 0.
- Independent specification and scoped quality review: PASS, no P0–P2 finding.
- Static comparison matched the sealed worker's checks and constants.
- Read-only compatibility parsing matched four preserved fixed-fixture startup
  streams. This did not replay their probes or create new observations.

Tests cover marker/framing errors, duplicate and nonfinite JSON, exact fields,
boolean versus integer values, missing/extra checks, capability and syscall
drift, and otherwise valid records at exactly 65,536 and 65,537 bytes.
The exact-size test was added after the initial GREEN, not claimed as a new
behavioral RED cycle.

## Identities and limits

```
development_startup_evidence.py
8a8e47d9cf360598aabdd5b3ca9c8f26fe6dc1cd2bb5c3e7f897023f32702b57
test_development_startup_evidence.py
ba1f8f2aea7cbc230814a025e8c5b33143e42f0ba9ca785127ddd66c1d773279
```

Private evidence is under `/tmp/tlf-candidate-executor-20261117-gb1gZX`:
`startup-evidence-review.md`, `startup-evidence-final-green.log`,
`startup-evidence-main-recheck.log`, and `startup-evidence-compatibility.log`.

A valid diagnostic does not establish producer identity, actual isolation,
READY/input delivery, child reaping, or whether subsequent candidate execution
occurred. The remote driver must bind these bytes to the pinned launcher and
parent lifecycle records. This helper is not launch approval or evidence of
LoRA effectiveness.
