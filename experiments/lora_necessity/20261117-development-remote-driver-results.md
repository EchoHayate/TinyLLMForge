# Development remote-driver qualification

Date: 2026-11-17. Driver component accepted; no real development run performed
in this qualification.

`development_remote_driver.py` connects the fixed coordinator to observed
generation and separate public/final admission callbacks. It checks the
runtime, checkpoint, base state, tokenizer state and loaded source bindings,
preserves failed-call evidence, and exports the owned artifact directory.
The fixed workload remains D01–D04, base/text, eight episodes and sixteen
scheduled generations. No adapter is mounted and no training is invoked.

## Review outcome

Independent review identified three defects, all reproduced and repaired:

- Export could continue after deadline cancellation. It now stops subsequent
  file/chunk work on `TimeoutError` or `KeyboardInterrupt`, while retaining
  ordinary unreadable-file diagnostics and other readable files.
- Source checks covered provider definitions but not imported consumer aliases.
  The fixed twenty-module graph now checks those aliases against the verified
  provider slots before constructing callbacks.
- Failed runtime and initial-integrity phases lost their measurable duration.
  Entered phases now retain elapsed time without fabricating successful checks.

Independent repair re-review returned specification PASS and quality APPROVED,
with all three findings closed and no remaining concrete P0–P2 in that scope.
This is component review, not proof of arbitrary hostile-code safety or remote
execution provenance.

## Validation

- Repair RED: five test methods produced nineteen assertion/subtest failures.
- Driver-focused GREEN: 45 tests passed.
- Author's named regression: 263 tests passed.
- Main's fresh named regression: 377 tests passed in 25.290 seconds, exit 0,
  with no warning or traceback diagnostics.
- Static source inspection confirmed that the fixed twenty-module order
  resolves local imports, including the startup and training-state helpers.

These suites overlap; their counts are not additive. Tests use synthetic
callbacks, task bytes, runtime objects and streams. They do not load a real
model, execute a generated candidate or contact a remote server.

## Consumer boundaries

The committed driver result has `export=null`: export has not happened when
that immutable file is published. Export frames and the final emitted envelope
carry the export receipt and committed terminal identity separately.
The emitted terminal cannot contain its own emission or final return duration;
only the returned/error envelope contains those later measurements.

Cancellation during inventory can leave an incomplete `EXPORT_END` without an
`EXPORT_BEGIN`. That is interrupted evidence, not a completed export. Missing
frames do not prove remote non-execution. `COORDINATOR_ENTERED` is retrospective;
its absence likewise does not prove that no coordinator work occurred.

Nested timing fields remain separate, unknown measurements remain null, and
the original failure is preserved alongside secondary collection/export errors.
No aggregate memory/disk quota or disconnect-cleanup guarantee is introduced.

## Frozen identities

```
development_remote_driver.py
2636fb740cec87c351e9d1c2493a1f9c9af795d2a96dc0d34196a1d86528cedf
test_development_remote_driver.py
c45591b16f52c262a077a9894ef0fdc8d014b115e021c3523feee76639eab7cf
```

Private evidence under `/tmp/tlf-candidate-executor-20261117-gb1gZX`:
`development-remote-driver-report.md`,
`development-remote-driver-review.md`,
`development-remote-driver-review-r2.md`,
`development-remote-driver-fix-review.diff` and
`development-remote-driver-main-r2-combined.log`.
The original review and failed logs remain preserved.

Next work is the pinned stdin package, one-shot transport and independent
offline reconstruction. Their acceptance, exact-package review and the actual
four-task run remain separate gates. This qualification establishes no LoRA
benefit, generalization result or automatic permission for a larger evaluation.
