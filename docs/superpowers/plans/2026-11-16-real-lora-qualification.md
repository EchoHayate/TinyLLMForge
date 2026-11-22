# Real LoRA Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run one fixed pretrained Qwen3 LoRA engineering roundtrip remotely.

**Architecture:** Add one thin orchestration module over the reviewed training and
adapter APIs. A private, source-pinned launcher first qualifies the remote dependency
versions with random tiny Qwen, then performs the single real run.

**Tech Stack:** Existing Python, unittest, PyTorch, Transformers, strict SSH.

## Completion Record

All three tasks completed on 2026-11-16. The original checklists below are retained
as the execution specification, not outstanding launch instructions:

- [x] Task 1: wrapper, 12 RED/GREEN cases, full regression and independent review.
- [x] Task 2: reviewed R2 launcher, remote compatibility, one real 16-step run;
  all three equality checks passed, final DONE and exit0 verified.
- [x] Task 3: adapter collected and hash-verified, raw receipts and preservation
  reconciled, dated real-lora-qualification result report written.

See `experiments/lora_necessity/20261116-real-lora-qualification-results.md` and
the stage progress ledger. The training claim is consumed; do not rerun it.

## Global Constraints

- Follow `docs/superpowers/specs/2026-11-16-real-lora-qualification-design.md`.
- Preserve all existing files. Add new files only; no commits, pushes, installations,
  telemetry, memory updates or new worktrees. Continue on the experimental branch.
- Real pretrained execution is remote only. CPU synthetic tests are allowed.
- No new training configurations, model generations, candidate execution or scored
  evaluation; no access to transfer/retention reference data.
- Test-generation telemetry/bootstrap conflicts with no-install/no-telemetry;
  use existing unittest directly. Do not run those scripts.

## Task 1: Training roundtrip

Create `experiments/lora_necessity/training_roundtrip.py` and
`experiments/lora_necessity/test_training_roundtrip.py`.

Interface: `run_roundtrip(model, dataset, artifact_path, *, emit=None) -> dict`.
Dependencies: `training_inputs.validate_training`, `training_worker.train_adapter`,
`adapter_artifact.save_adapter/load_adapter/mounted_adapter`.

- [ ] Write failing integration tests using `tiny_model(torch)` and
  `synthetic_dataset()` from `test_training_worker`, lazy torch imports.
  Assert ordered 16 training calls plus five prompt-only forward calls;
  original state/modes/grad flags remain intact; `checks` contains exactly
  `initial_equals_base`, `reload_equals_final`, `restored_equals_base`, all true.
- [ ] Cover existing artifact refusal before training, a forward exception,
  nonfinite probe logits, corrupted saved artifact and malformed dataset.
  Use a temporary directory per case; preserve intentional failures in test logs.
- [ ] Run:
  `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -m unittest discover -s experiments/lora_necessity -p test_training_roundtrip.py -v`
  Expected RED: the roundtrip module is absent.
- [ ] Implement the minimal sequence from the spec. Keep the original core receipt
  unchanged under `training_receipt`; return `checks`, `artifact`, `timings_ns`,
  `peak_allocated_bytes`, `peak_reserved_bytes`, plus false admission flags.
  Require caller model eval/frozen; reject autocast/inference-mode before mutation.
  Dataset is validated before the first forward. Prompt logits are cloned on CPU.
  `emit` is optional and must not include dataset/token/logit content.
- [ ] Run the focused suite to GREEN, then the existing full directory regression.
  No new dependency installation or shared-process eager torch imports.

## Task 2: Remote qualification and one-shot execution

Create private `bootstrap.py`, `remote_job.py`, `driver.py`, `supervisor.py` beneath
`tlf-real-lora-qualification-20261116-8ry4aa_s` (outside Git).
Main agent owns these; independent reviewer must inspect them before launch.

- [ ] Pin exact source closure, eight-row supervision file, prior runtime metadata
  and checkpoint file observations. Save manifest and source hashes with exclusive
  creation; compile all new source locally without executing model code.
- [ ] Implement driver commands `prepare`, `compatibility`, `train`, `collect`.
  Each stage has an exclusive local claim, exact saved argv, separate raw stdout/
  stderr and terminal result. Source is streamed to strict SSH; no old controllers.
  `train` cannot run without successful compatibility and a source-bound review.
- [ ] Use nonblocking deadline-aware pipe framing and input writes; validate the
  exact stage/event sequence and serialized result hash. An independent parent
  enforces the model child's deadline and records its actual termination outcome.
- [ ] Bootstrap validates hostname/UID before accepting private payload, creates
  the exact new private directory, verifies every file and rejects reuse.
  All later jobs rehash staged sources before import; never auto-retry failed jobs.
- [ ] Tiny compatibility uses the same roundtrip with random Qwen on CPU;
  real execution uses only the pinned existing pretrained model on one fixed UUID.
  Save training receipt, roundtrip result and adapter in the private remote run.
  Model-load time and hash-check time are recorded outside core training time.
- [ ] Review the module, tests and private launch package independently. Fix and
  re-review any important findings; retain both failure and successful evidence.
- [ ] Execute one compatibility attempt, then one real training attempt only
  after GPU recheck. Verify exit, stage sequence and receipt hashes.

## Task 3: Evidence closeout

- [ ] Reconstruct 16-step counts, adapter bytes/hashes, equality checks and cost
  scopes from raw records, without reading any evaluation answers.
- [ ] Verify existing code/data hashes against `baseline.json`; check HEAD/index.
- [ ] Add `experiments/lora_necessity/20261116-real-lora-qualification-results.md`.
  Report completed versus missing work and actual failure status, never inference
  benefit from loss reduction. Keep sensitive execution records outside Git.
