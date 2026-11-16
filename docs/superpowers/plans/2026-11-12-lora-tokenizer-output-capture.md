# Tokenizer Output Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retain bounded binary output with explicit EOF and fail-stop behavior.

**Architecture:** One dependency-light receiver with three owned buffers;
public metadata and private bytes remain separate. Transport, process
supervision and OS resource enforcement are not part of this component.

**Tech Stack:** Python 3.12 standard library and unittest.

## Global Constraints

- Existing checkout only; no branch/worktree, commit, push, network/SSH, installation, telemetry, real metadata/tokenizer/model loading, GPU, training or candidate/task-source execution.
- Standard-library local tests only; preserve accepted sources, tests, data, documents and all sealed evidence.
- Use `apply_patch`; no inline code comments or alphabetic one-letter names.
- Fixed accepted payload caps: result 67,108,864 bytes, stdout 1,048,576 bytes, stderr 1,048,576 bytes. No configurable limits, truncation or retry.
- A feed chunk is exact built-in bytes, nonempty and at most 65,536 bytes.
- Public snapshots contain no output text, token IDs or content hashes.
- All five admission flags remain false; cleanup is `CLEANUP_UNVERIFIED`.
- Root/resource/deadline/termination enforcement and cleanup verification remain false; complete capture is not a valid audit or real admission.

### Task 1: Bounded binary capture

**Files:**
- Create `experiments/lora_necessity/tokenizer_output_capture.py`.
- Create `experiments/lora_necessity/test_tokenizer_output_capture.py`.

**Requirements:** Read
`docs/superpowers/specs/2026-11-12-lora-tokenizer-output-capture-design.md`
completely. Its exact state/method/schema/error contract is binding.

**Interfaces:** `OutputCapture`, `OutputCaptureError`; `feed(stream, chunk)`,
`end(stream)`, `finish()`, `abort()`, `snapshot()`, `private_outputs()`.
No changes to, or integration with, sealed timing/controller modules.

- [x] Start with a missing-module assertion through `find_spec`; write the
  first construction/binary preservation tests before implementation.
  Example after module import:

  ```python
  capture = module.OutputCapture()
  capture.feed("result", b"\x00\xff{}")
  for stream in ("result", "stdout", "stderr"):
      capture.end(stream)
  capture.finish()
  self.assertEqual(capture.private_outputs()["result"], b"\x00\xff{}")
  self.assertTrue(capture.snapshot()["capture_complete"])
  self.assertFalse(capture.snapshot()["benchmark_eligible"])
  ```

- [x] Run focused RED from repo root and retain numbered exclusive evidence:
  `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -m unittest discover -s experiments/lora_necessity -p 'test_tokenizer_output_capture.py' -v`.
  Expected failures identify missing behavior, not environment/import mistakes.
- [x] Implement minimal private state and owned byte buffers. Follow the
  exact design transitions; derive counts from retained lengths, validate
  before appending, reject a whole offending chunk, preserve first failure.
- [x] Expand RED/GREEN cycles for all design boundary cases. Test real
  production caps using repeated 65,536-byte chunks rather than enormous
  per-call chunks or patched smaller constants. At least one test fills all
  three production capacities. Keep one full-capacity capture alive at a time.
  Use bounded injected buffer fixtures only for append/copy failures and
  cancellation identity; never trigger actual resource exhaustion.
- [x] Run final focused GREEN and one complete suite:
  `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -m unittest discover -s experiments/lora_necessity -p 'test_*.py' -v`.
  All prior 218 tests and additions must pass without unexpected output.
- [x] Self-review and write `task-1-report.md` under
  `/tmp/tlf-tokenizer-output-20261112-ZnitkP`, including exact commands,
  RED/GREEN receipts, current source hashes, file list and limitations.
  No commit. Own only this source/test pair and fresh `task-1-*` evidence.
- [x] Main packages full additions and obtains independent task Spec/Quality
  review. Resolve findings before completion.

## Main closeout

- [x] Extend preceding sealed protection manifest and record entry Git state.
- [x] Independently rerun full regression, syntax/name checks, cold import
  probe and exact addition reconstruction with source-bound receipts.
- [x] Obtain fresh whole-component review of source, tests, design and evidence.
- [x] Reverify preservation, record results and seal new evidence exclusively.

Closeout completion is conditional on the successfully written and verified
`/tmp/tlf-tokenizer-output-20261112-ZnitkP/final-verification.json`, which binds
this plan and the final ledger. This checkbox alone is not sealing evidence.

## Plan check

One coherent implementation task; no hidden backend selection or launch.
No root-budget guarantee is inferred from the sum of output caps. No public
log safety is inferred from bytes-only storage. No smaller test cap substitutes
for the required production-limit boundary cases.
