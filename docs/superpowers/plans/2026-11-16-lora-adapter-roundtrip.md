# LoRA Adapter Round Trip Implementation Plan

> **For agentic workers:** Use subagent-driven-development for the bounded module
> task and independent spec/quality review. Preserve the existing worktree.

**Goal:** Persist, verify, reload and temporarily mount the trained adapter.

**Architecture:** One adapter-artifact module consumes the existing training-core
result and reuses its tensor hashing, layout checks and low-rank wrapper.
The real-model launcher is deliberately outside this component.

**Tech Stack:** Existing Python 3.12, PyTorch and unittest; no installation.

## Global Constraints

- Real pretrained execution remains remote.
- Keep all four admission flags false.
- Preserve existing files, sealed data, remote evidence and unrelated work.
- Do not retry SSH, rerun the 1416 observations, install, commit or push.
- Keep the current `experiments/feedback-active-verification` branch.
- Use apply_patch; no inline code comments or one-letter variable names.
- CPU tests use only synthetic tensors/tokenizers.
- No adapter output-change requirement after BF16 rounding.

## Task 1: Artifact and frozen mounting

Read `docs/superpowers/specs/2026-11-16-lora-adapter-roundtrip-design.md` in full.

Files:
- Create `experiments/lora_necessity/adapter_artifact.py`.
- Create `experiments/lora_necessity/test_adapter_artifact.py`.
- Write report `.superpowers/sdd/lora-adapter-roundtrip-20261116-report.md`.
- Do not modify `training_worker.py` or `training_inputs.py`.

Interfaces:
- `save_adapter(path, result)` returns
  `{"sha256": str, "file_bytes": int, "save_ns": int}`.
- `load_adapter(path, expected_sha256=...)` returns
  `{"result": dict, "load_ns": int}`.
- `mounted_adapter(model, result, state="adapter")` yields the model.
- Existing `train_adapter(model, dataset)` returns the result to persist.
- Existing tests export `tiny_model(torch)` and `synthetic_dataset()`.

- [x] Write focused tests before the implementation. Fail with an assertion that
  the module/API is absent, not unrelated fixture/import failures.
- [x] Observe RED:
  `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -m unittest discover -s experiments/lora_necessity -p test_adapter_artifact.py -v`.
- [x] Implement the three APIs and the validation in the spec. Use exclusive
  `os.open` with mode 0600, serialize an exact kind/version/result envelope,
  hash serialized bytes, and require the caller's expected digest before
  `torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)`.
  No unsafe fallback. Preserve incomplete writes instead of retrying.
- [x] Verify real tiny-model round trip with this API sequence:

  ```python
  result = train_adapter(model, synthetic_dataset())
  saved = save_adapter(path, result)
  loaded = load_adapter(path, expected_sha256=saved["sha256"])["result"]
  with mounted_adapter(model, loaded, state="initial_adapter") as active:
      initial = active(**inputs).logits
  with mounted_adapter(model, loaded) as active:
      final = active(**inputs).logits
  ```

  Compare `initial` to base, and `final` to the same in-memory trained adapter.
  Verify restored base output, modules, flags and RNG. Do not assert final
  logits must differ from base. Add malformed-state and exception cases listed
  in the spec, using real tensors. Save RED/GREEN command/output in the report.
- [x] Run focused GREEN plus `test_training_*.py`, then self-review.
- [x] Obtain independent spec/quality review; fix substantive findings through
  reproduced regressions and re-review before marking complete.

## Main closeout

- [x] Preserve a source-bound diff and independent review report.
- [x] Run the full `test_*.py` directory regression after review. First run exposed
  import-order interference; preserve it, isolate new tensor cases in child
  processes, obtain independent review, and rerun. Final:530 run,528pass/2skip.
- [x] Verify old seal bindings and unchanged existing-file inventory.
- [x] Record fresh SSH failure separately from local adapter results.
- [x] State remaining remote dependency and scientific gaps without claiming
  that local implementation qualifies a real model or closes the pilot.
