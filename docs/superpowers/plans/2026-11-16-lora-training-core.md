# LoRA Training Core Implementation Plan

> **For agentic workers:** Use test-driven-development and task-scoped review.
> The main worker handles the training core; a fresh worker may handle the
> independent input module. Do not modify each other's files.

**Goal:** Implement answer-only training inputs and a real, testable PyTorch
gradient-LoRA core without loading a pretrained model.

**Architecture:** A standard-library input module binds frozen supervision to
validated token rows. The tensor core consumes that interface, injects q/v
adapters temporarily, trains exactly sixteen steps and returns evidence.

**Tech Stack:** Python 3.12, standard unittest, existing PyTorch.

## Global Constraints

- Follow `../specs/2026-11-16-lora-training-core-design.md`.
- Stay in the current experimental branch; preserve unrelated untracked work.
- Do not change frozen files, existing protocol, sealed evidence or old results.
- No installation, telemetry, network, real model load, remote launch or task execution.
- Only tiny synthetic tensors may perform forward/backward locally.
- No commit/push or new branch/worktree in this implementation stage.
- Existing unittest conventions override optional toolkit bootstrap/reporting.
- No inline code comments or one-letter variable names.

## Task 1: Answer-only input construction

Files:
- Create `experiments/lora_necessity/training_inputs.py`.
- Create `experiments/lora_necessity/test_training_inputs.py`.

Interfaces:
- `prepare_training(tokenizer, supervision_raw) -> dict`.
- `validate_training(dataset) -> str`.
- Consume unchanged `training_jobs`, `measure`, `PINS`, `digest`.
- Produce exactly the schema and token/label semantics in the design.

- [x] Write failing unittest cases before implementing either interface.
- [x] Observe missing-module assertion failure.
- [x] Implement preparation: measure each frozen supervision job; reject limits
  or state changes; take prompt + target IDs, not trailing template tokens.
  Set attention to ones and labels to `[-100] * prompt_count + target_ids`.
  Bind fixed source pin and tokenizer state, calculate the canonical digest.
- [x] Implement independent validation: strict JSON types/keys; exactly eight
  sorted IDs; positive bounded prompt/target counts; mask all ones; labels
  exactly as above; terminal EOS unique in target; valid hashes/source pin.
  Reject boolean integer aliases, missing rows, extra keys and rebound but
  semantically invalid structures.
- [x] Verify focused tests including immutability, corrupted source, token
  limits, post-EOS tail exclusion, answer-label alignment and tokenizer drift.
- [x] Write a concise TDD report in
  `.superpowers/sdd/lora-training-inputs-20261116-report.md`.

Command:
```sh
PYTHONDONTWRITEBYTECODE=1 python3.12 -B -m unittest discover \
  -s experiments/lora_necessity -p 'test_training_inputs.py' -v
```

## Task 2: Tensor training core

Files:
- Create `experiments/lora_necessity/training_worker.py`.
- Create `experiments/lora_necessity/test_training_worker.py`.

Interfaces:
- Consume `validate_training(dataset) -> str`.
- `LoRAProjection(base)` exposes FP32 `adapter_a`, `adapter_b`, frozen `base`.
- `answer_loss(logits, labels) -> torch.Tensor` uses causal shift and ignores -100.
- `train_adapter(model, dataset) -> dict` returns `receipt`, `initial_adapter`,
  `adapter`; tensor mappings are detached CPU clones.

- [x] Write RED tests for zero delta, initial first-step gradient behavior and
  causal answer-only loss. Verify expected failures before code.
- [x] Implement wrappers using `torch.nn.functional.linear`, fixed rank/alpha,
  and answer CE on only selected preceding logits. Tests compare directly to
  independently selected logits, not to another invocation of the same helper.
- [x] Write RED end-to-end tiny-model tests for 16 updates and failure cleanup.
- [x] Implement validation before mutation; preserve modes/RNG/grad flags;
  inject wrappers; snapshot base; run `configuration()["training"]` exactly;
  check finite tensors and clipped norm; hash final state; restore caller.
- [x] Verify targeted tests, including malformed topology/dtype, duplicate
  module aliases, finite-state failure, disabled autograd, and no cache.
- [x] Independently review both modules and resolve substantive findings.
- [x] Run full existing suite plus new tests and document observed limits.

Command:
```sh
PYTHONDONTWRITEBYTECODE=1 python3.12 -B -m unittest discover \
  -s experiments/lora_necessity -p 'test_training_worker.py' -v
```

## Closeout

- [x] Record results in `experiments/lora_necessity/20261116-training-core-results.md`.
- [x] Keep existing sealed README/results unchanged; newer result states the
  actual implementation status without altering historical evidence.
- [x] Check git diff scope and source hashes. Report implemented versus
  remotely unvalidated work separately; do not declare LoRA benefit.

Completed: 25 input tests and 17 tensor-core tests; final complete suite502,
500 passed/2 platform skips in74.031s. R1 found two issues; R2 closed both.
Synthetic input-to-core integration also passed after fixes. No real model
execution or new publication occurred.
