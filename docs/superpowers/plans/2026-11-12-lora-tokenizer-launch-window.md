# Tokenizer Launch Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Locally validate shared absolute deadlines and per-instance single-attempt accounting.

**Architecture:** One synchronous main-thread timing primitive; no external operation callbacks and no integration into sealed components. The future launcher owns and enforces it.

**Tech Stack:** Python 3.12 standard library, unittest and unittest.mock.

## Global Constraints

- Existing checkout only; no branch/worktree creation, commits or pushes.
- Local stdlib tests only; no SSH/network, installations, telemetry, real metadata discovery, tokenizer/model loading, weights, GPU, training, or candidate/task-source execution.
- Preserve all accepted source/tests/data/docs and every sealed evidence root.
- Use `apply_patch`; no inline code comments or alphabetic one-letter names.
- Connection claim fixes work deadline at start + 540 seconds and total deadline at start + 600 seconds, in integer nanoseconds.
- At most one connection claim and one audit claim per instance.
- Cleanup lasts at most 60 seconds and never beyond the total deadline; phase changes and repeated calls cannot reset either deadline.
- All five admission flags remain false; cleanup is `CLEANUP_UNVERIFIED`.
- Resource enforcement, process termination and cleanup verification remain explicitly false. Snapshot data is last-observed accounting, not live proof.

## Task 1: LaunchWindow source and tests

**Files:**
- Create `experiments/lora_necessity/tokenizer_launch_window.py`.
- Create `experiments/lora_necessity/test_tokenizer_launch_window.py`.

**Binding detailed contract:** Read
`docs/superpowers/specs/2026-11-12-lora-tokenizer-launch-window-design.md` in full.
It defines all states, methods, precedence and exact snapshot keys.
Do not modify the parent design or any sealed module.

**Interfaces:** `LaunchWindow`, `LaunchWindowError`; methods
`claim_connection()`, `claim_audit()`, `work_remaining_ns()`,
`begin_cleanup(reason)`, `cleanup_remaining_ns()`, `close()`, `snapshot()`.

- [x] Write a failing construction/claim test before source implementation.
  Use `importlib.util.find_spec` with an assertion so absent implementation
  produces an intentional assertion failure, not an accidental import error.
  A first behavioral test after import is:

  ```python
  with patch("tokenizer_launch_window.time.monotonic_ns", return_value=100):
      window = LaunchWindow()
      window.claim_connection()
      self.assertEqual(window.work_remaining_ns(), 540_000_000_000)
      self.assertEqual(window.snapshot()["total_deadline_ns"], 600_000_000_100)
  ```

- [x] Run focused RED, preserving exclusive numbered log/receipt files in
  `/tmp/tlf-tokenizer-window-20261112-osGkZH`. Run from repo root:
  `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -m unittest discover -s experiments/lora_necessity -p 'test_tokenizer_launch_window.py' -v`.
  Confirm failures name missing behavior, not unrelated environment trouble.
- [x] Implement the minimal class using internal clock observation and one
  shared cleanup transition helper. Keep all state private, validate exact
  primitive types, preserve first stop reason, and keep thread guards before
  clock or mutations. Follow the design table for each method; add no loaders,
  CLI, resource backend, clock parameter or external-operation callbacks.
- [x] Expand test-first cycles for the design's boundary matrix. Include
  work expiry at deadline minus one / equal / plus one nanosecond; no audit
  claim at expiry; early cleanup ending before overall deadline; delayed
  observation after total; repeated/misordered operations in each state;
  bool/int-subclass/negative/regressing clocks; ordinary errors and original
  KeyboardInterrupt/SystemExit objects; wrong-thread calls without claims;
  snapshot mutation and stable false flags. Keep numbered RED/GREEN evidence.
- [x] Run focused GREEN and then the complete suite once:
  `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -m unittest discover -s experiments/lora_necessity -p 'test_*.py' -v`.
  All prior 191 tests and all new tests must pass without unexpected output.
- [x] Self-review source/tests and write `task-1-report.md` in the fresh
  evidence root, with commands, results, TDD evidence, changed files and limits.
  No commit; all additions remain untracked.
- [x] Main builds a complete addition diff and hash manifest, obtains a fresh
  task Spec/Quality review, and resolves findings before marking complete.

## Main closeout checkpoints

- [x] Verify prior controller seal; extend protection baseline to its sources,
  documents and all evidence. Record Git branch/HEAD without network access.
- [x] Run independent main regression, syntax/name/import checks and exact
  full-addition package verification, binding output to current source hashes.
- [x] Obtain fresh whole-component review of the two additions, child docs
  and evidence; use the advertised explicit reviewer route with no fork.
- [x] Recheck protected files and tracked/index cleanliness, update this plan
  and fresh ledger, then write exclusive final receipt. Seal only this local
  component; no real launch or scientific benefit claim.

## Self-review

All design behavior belongs to Task 1; main checkpoints validate preservation
and evidence rather than changing implementation. No conflicting retry,
cleanup success, resource enforcement or publication requirement.

## Local completion

Task and final reviews both returned Spec PASS / Quality APPROVED, with no
findings. Main independently ran 218 tests (191 preserved + 27 new) in
15.430 seconds, exit 0, no unexpected output. Static and fresh-process
import/construction checks passed; complete additions reconstruct exactly.
All 477 prior protected files match, and tracked/index diffs remain empty.

Accepted source: 161 lines; paired tests: 683 lines. No accepted dependency
was edited. Main corrected its task-brief helper's accidental overwrite of
one unrelated tracked brief back to exact clean-entry bytes; the incident
and fresh task brief are retained in this stage's evidence.

Exclusive seal:
`/tmp/tlf-tokenizer-window-20261112-osGkZH/final-verification.json`.
That receipt is authoritative for successful closeout; do not rerun helpers
over existing outputs or restart completed agents. The child design's
reviewed behavior is unchanged.

This completes local timing/accounting only. No real tokenizer, external
connection, process termination, aggregate hard limit, cleanup verification,
training or scientific benefit was established. Keep all admission flags
false and proceed to remaining launch gates under a separate local plan.
