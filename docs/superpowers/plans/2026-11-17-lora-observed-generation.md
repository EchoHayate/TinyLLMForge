# Observed Generation Implementation Plan

> **For agentic workers:** Use subagent-driven-development with independent review.

**Goal:** Preserve one generation attempt and its observable work on success or failure.

**Architecture:** A thin wrapper around unchanged generation_core owns an exclusive
claim, root-forward hooks and separate raw-generation/observation records.

**Tech Stack:** Existing Python 3.12 unittest and Torch-compatible hooks; no dependencies.

## Global Constraints

Follow `../specs/2026-11-17-lora-observed-generation-design.md`.
Only create observed_generation.py and its test file under experiments/lora_necessity.
No modifications to old generation, training, executor or frozen-data files.
No remote operations, actual pretrained weights, generated-code execution, installs,
telemetry, commit or push during component implementation.
Do not overlap test-file RED edits with the currently running full regression.

## Task 1: One-attempt observer

- [ ] Read the complete design and generation_core.py; write the focused tests first.
- [ ] Verify missing-module RED with the following invocation:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error \
  -m unittest discover -s experiments/lora_necessity -p 'test_observed_generation.py' -v
```

- [ ] Implement `generate_observed` and `GenerationAttemptError` exactly as specified.
  Reuse the existing generation function, message digest/encoding functions and
  repository test conventions; expose no new tuning parameters or retry path.
- [ ] Use a trusted synthetic model/hook fixture and patched generator to exercise
  counting and failures. Include an actual small synthetic generation integration
  case using existing fixtures if their interfaces fit without unrelated edits.
- [ ] Run focused GREEN with the same command; retain RED/GREEN logs and source hashes.
- [ ] Deliver a report with exact scope, commands, output, unresolved concerns and
  any specification ambiguity. Do not claim remote or task acceptance.
- [ ] Independent reviewer checks claim/record lifecycle, cleanup and unknown-cost
  semantics; fix concrete findings with reproducing tests and re-review.

## Task 2: Next-stage handoff

- [ ] After component acceptance, pin it in a separate real-base generation-only
  diagnostic with two predeclared non-task message sequences.
- [ ] Keep real tokenizer/model checks, selected GPU observation and process
  deadline in that outer diagnostic. Do not use the consumed training launcher.
- [ ] Preserve current remote toy-executor and random-model results as prior evidence.

## Review Fix: Publication Commit Boundary

Scope: only observed_generation.py, test_observed_generation.py and this observer
design/plan; preserve earlier source, sealed review artifacts and prior reports.
This fix does not authorize Task 2, remote/model loads or old experiment restarts.

- [x] Capture regression RED for unlink failure after generation, completed
  observation and failed observation publication, plus write/link failures with
  simultaneous cleanup failure. Include a one-shot completed-observation unlink
  failure that reproduces the disagreement between disk and the live API.
- [x] Treat successful completed-observation publication as the attempt commit
  point, not generation publication alone. Keep pre-publication failures fatal,
  preserve primary exception chains and never overwrite a terminal record.
- [x] Make temporary-name cleanup nonfatal and separately count/report it in
  result/exception `record_cleanup`; do not mutate the persisted observation or
  introduce required post-commit persistence. Hook removal remains fatal.
- [x] Document recovery precedence, residual-file maintenance, and that failure
  counts are root-hook observations rather than universally exact body or kernel
  completions. Test existing pre-hook and global post-hook exceptions.
- [x] Run Python 3.12 `-B -W error` focused and combined generation tests.
  Save exact commands, RED/GREEN logs, counts and hashes in
  `/tmp/tlf-candidate-executor-20261117-gb1gZX/observed-generation-fix-report.md`.
- [ ] Hand off for independent re-review; do not claim real-base acceptance.
