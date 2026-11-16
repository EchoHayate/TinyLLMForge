# Tokenizer Pipe Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read real nonblocking output pipes through sealed capacity and time-window components.

**Architecture:** One main-thread, one-shot reader owns duplicate read descriptors
and selector lifetime; it borrows the caller's AUDITING window. It never owns
processes or interprets an audit.

**Tech Stack:** Python 3.12 POSIX standard library and unittest.

## Global Constraints

- Existing checkout only; no branch/worktree, commit or push.
- Local stdlib tests only; no network/SSH, installations, telemetry, real
  metadata discovery, tokenizer/model loading, GPU, training or task-source
  execution. Synthetic bytes through local anonymous pipes are permitted.
- Preserve every accepted source/test/data/document and sealed evidence root.
- Use `apply_patch`; no inline code comments or alphabetic one-letter names.
- Reuse `OutputCapture` unchanged: result 67,108,864 bytes, stdout and stderr
  1,048,576 bytes each; maximum read 65,536 bytes, no truncation or retries.
- Reuse an already AUDITING exact `LaunchWindow` unchanged. Never create a
  replacement window, claim connections/audits, reset deadlines or close it.
- Selector waits are at most 0.1 seconds and at most the observed remaining
  work budget. Before and after each wait and each read, verify work remains.
- No configurable budgets, root-write/resource enforcement, process signals
  or admission promotion. All ten capture flags stay false and cleanup stays
  `CLEANUP_UNVERIFIED`.

### Task 1: Real bounded pipe reader

**Files:**
- Create `experiments/lora_necessity/tokenizer_output_reader.py`.
- Create `experiments/lora_necessity/test_tokenizer_output_reader.py`.

**Requirements:** Read the complete binding design
`docs/superpowers/specs/2026-11-13-lora-tokenizer-output-reader-design.md`.
All API, ownership, error precedence and exact output schema are defined
there. Do not implement a process backend or modify sealed dependencies.

**Interfaces:** `OutputReader()`, `run(window, streams)`, `snapshot()`,
`private_outputs()`, `OutputReaderError(ValueError)`. Consume sealed
`OutputCapture` and `LaunchWindow`; use their public methods only.

- [x] Write construction and real-pipe completion tests first, using a
  missing-module assertion rather than an accidental import error:

  ```python
  self.assertIsNotNone(importlib.util.find_spec("tokenizer_output_reader"))
  reader = importlib.import_module("tokenizer_output_reader").OutputReader()
  self.assertEqual(reader.snapshot()["state"], "NEW")
  self.assertTrue(reader.snapshot()["local_handles_released"])
  ```

  For completion create three pipes, set read ends nonblocking, write small
  binary payloads, close write ends, claim a real window's connection/audit,
  call run, assert exact private bytes and FINISHED, then fstat originals.
  Use fixture cleanup for every FD on all outcomes.
- [x] Run focused RED and retain fresh numbered log plus receipt with exit
  code, command, log hash and before/after source hashes:
  `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -m unittest discover -s experiments/lora_necessity -p 'test_tokenizer_output_reader.py' -v`.
- [x] Implement only the tested state/ownership/read path, then focused GREEN.
  Skeleton responsibilities:

  ```python
  class OutputReaderError(ValueError):
      def __init__(self):
          super().__init__("tokenizer output reader invalid")
  ```

  `run` consumes the attempt, validates/binds an AUDITING window and exact
  stream map, validates pipe properties before consuming data, duplicates
  and registers only three read ends, checks the window around select/read,
  feeds before the next read, recognizes only actual EOF, and always releases
  only its duplicates/selector. No blanket success in finally.
- [x] Expand RED/GREEN for the complete design validation section: wrong
  types/states/threads/aliases/blocking or write FDs; real capacity overflow;
  pre/post wait/read deadlines; transient read conditions; setup/I/O/capture/
  release faults and cancellation. Preserve first failures and stop reads.
  Bound producer-thread lifetime; no reduced-capacity substitute.
- [x] Run focused final GREEN and one complete suite:
  `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -m unittest discover -s experiments/lora_necessity -p 'test_*.py' -v`.
  Preserve all previous 245 unique test IDs and pristine final output.
- [x] Self-review, record exact scope/test results/TDD limitations in
  `/tmp/tlf-tokenizer-reader-20261113-oERiBR/task-1-report.md`.
  Own only the source/test pair and fresh `task-1-*` evidence files.
- [x] Main packages full additions and obtains independent task review;
  resolve findings and source-bound evidence before task completion.

## Main closeout

- [x] Extend previous protection manifest, including its complete final seal.
- [x] Independently verify regression, static checks, cold import/construction,
  full-addition package and current source hashes.
- [x] Obtain a fresh whole-component review and generate local reports.
- [x] Reverify preservation/Git and exclusively seal current-stage evidence.

Closeout completion is conditional on the successfully written and verified
`/tmp/tlf-tokenizer-reader-20261113-oERiBR/final-verification.json`, binding
this plan and the final ledger. Checkboxes alone are not sealing evidence.

## Plan check

One cohesive task, no remote assumptions. Both implementations of time and
capacity remain immutable dependencies. Explicit close failures preclude a
success receipt; local handle release never promotes process cleanup flags.
