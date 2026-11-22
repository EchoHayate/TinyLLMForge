# Development final-domain execution qualification

Date: 2026-11-17. Branch: `experiments/feedback-active-verification`.

## Delivered

Second-round development requests now have separate admission and controller
entry points. The final controller retains the sealed first-round lifecycle
implementation, changing only its request-validator import. Final admission
retains computation policy v2 and records a distinct final-domain admission.

The admission record must be exclusively created, fully written and successfully
closed before controller launch. Requests are snapshotted through canonical JSON;
the controller receipt must match the admitted request digest. A failed
admission/persistence step aborts rather than becoming model feedback.

These entry points accept D01–D04/base/text/round-two requests. They do not
authenticate the caller's generation or frozen-case hashes by themselves.

## Verification

- Missing-module RED: 39 new failures, with 28 existing protocol tests passing.
- Final focused GREEN: 68 passed, comprising 14 admission, 26 controller and
  28 protocol tests.
- Main safe old/new aggregate: 142 passed in 5.414 seconds, exit 0.
- Independent review: specification PASS and scoped quality PASS.
- Complete controller-byte comparison: exact prescribed import-only derivative.
- Admission-function comparison: identical policy body except evidence kind.
- Source/log hash recheck: all ten author-report entries matched.

The 26 controller tests include all 20 inherited lifecycle tests. Coverage
includes READY-before-input, limits, startup failure, backpressure, duplicate
claims/results, stdout spoofing, ValueError, result-then-hang/flood, and kill/reap.
Admission tests cover snapshot independence, refusal and persistence failures,
descriptor closure, digest mismatch, and literal trusted-helper integration.

No helper interprets candidate payload source. Its fixed return value despite
a raising candidate body tests framing, not repair correctness or isolation.

The explicit aggregate command excludes worker tests that execute synthetic
payloads; it reconciles the initial overbroad test pattern with this component's
control-frame-only test scope:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lora_necessity \
  /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error -m unittest \
  test_candidate_execution_protocol test_candidate_execution_admission \
  test_candidate_execution_controller test_development_final_protocol \
  test_development_final_admission test_development_final_controller -v
```

## Identities

```
development_final_admission.py
8af6e5c1534fbec4da1897119b5431c0f35e2b69fdecb76df235219f112c094e
development_final_controller.py
4b70a321227425296cdbae47d15d7237a399627e18c8443da70955649f3df7cd
computation policy v2
16fb67e48d8adba83ad0cc832053d65d2b4d71a6b0075407d1c4a0e006410859
```

Private evidence under `/tmp/tlf-candidate-executor-20261117-gb1gZX`:

- `development-final-execution-report.md`
- `development-final-execution-review.md`
- `development-final-execution-final-green.log`
- `development-final-execution-safe-combined.log`

## Still Separate

Actual generated-candidate execution, current sandbox/launcher qualification,
second-round generation provenance, complete frozen-case binding and final
scoring remain separate gates. No real development candidate, model, remote
probe, training or LoRA comparison ran during this component qualification.
