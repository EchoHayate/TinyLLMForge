# Existing-Host Tokenizer Launch Implementation Plan

> **For agentic workers:** Use subagent-driven-development for the implementation
> task and an independent read-only review before the single remote attempt.

**Goal:** Run the accepted 324-row tokenizer audit once in the observed remote
environment and return private evidence without changing the audit semantics.

**Architecture:** A stdlib-only remote bootstrap checks identity and then accepts
hash-bound inputs on SSH stdin. It invokes the existing snapshot and audit APIs;
a local nonblocking collector bounds transport, captures evidence and enforces
the existing launch window. No container, scheduler or cleanup service.

**Tech Stack:** Python 3.12 stdlib, existing tokenizer modules, OpenSSH/GSSAPI;
the remote tokenizer stack remains the observed, unmodified installation.

## Execution Outcome

Implementation, focused/combined tests, independent review and one authorized
remote attempt completed. The attempt returned a validated OPEN_FAILED report,
with 0/324 rows measured. This plan's engineering/execution work is complete;
the scientific token-limit question remains unanswered.

Evidence and next-step boundary:
`experiments/lora_necessity/20261113-existing-host-tokenizer-results.md`.
Do not rerun this run ID or treat the implementation review as audit acceptance.

## Global Constraints

- Approved scope: `experiments/lora_necessity/20261113-existing-host-tokenizer-scope.md`.
- Approval is recorded in `/tmp/tlf-tokenizer-host-launch-20261113-tilybrwi/authorization.json`.
- Preserve the prepared inventory/profile, accepted source modules and old seals.
- One connection, one audit; no retry, generation, GPU, training, installation,
  publication, branch/worktree changes, commits or pushes.
- Work/total limits: 540/600 seconds from initial connection claim.
- Result <= 67,108,864 bytes; each diagnostic stream <= 1,048,576 bytes.
- Input <= 4096 tokens; target including unique EOS <= 256; stop on first failure.
- Directory/file modes: 0700/0600, except the accepted read-only snapshot.
- Automatic disconnect cleanup and aggregate memory/file budgets are waived.
- Retain the eight-task ceiling without an account-wide `RLIMIT_NPROC` or a
  container platform: remote bootstrap must remain single-threaded and deny
  process/thread creation before importing any third-party library. Fail closed
  if this lightweight Linux protection or its behavioral checks fail.
- Do not treat environment variables or process sampling as that hard limit.
- SSH service infrastructure is pre-existing; the new remote audit process is
  the owned task. Any new remote helper must be explicitly counted.
- Keep legacy cleanup/admission fields unchanged; report counts separately.

## Task 1: Minimal Existing-Host Transport And Bootstrap

**Create only:**
- `experiments/lora_necessity/tokenizer_host_remote.py`
- `experiments/lora_necessity/tokenizer_host_launch.py`
- `experiments/lora_necessity/test_tokenizer_host.py`

**Interfaces and ownership:**
- Remote stdlib module: bounded framing helpers, input validation, bootstrap,
  isolated path setup, private staging and invocation of accepted APIs.
- Local module: build the bound command/payload, collect one subprocess with
  nonblocking pipes, write private evidence; executable launch must require
  explicit arguments and exclusive local attempt creation, never run on import.
- Tests: synthetic local subprocesses only; never connect or use private data.
- Main agent owns all approval/plan/ledger/result artifacts and actual SSH launch.

- [x] Write failing tests first, using `unittest` and actual small subprocesses.
  Cover short/fragmented frames, unknown/duplicate channels, trailing data,
  oversized declarations, EOF/truncation, nonzero exit, stdout/stderr flood,
  timeout/reaping, input hash mismatch, safe filenames and exclusive creation.
  Example behavior:
  ```python
  with self.assertRaises(ValueError):
      decode_frames(b"not a framed result")
  ```
- [x] Run the focused suite and retain the expected failure in the fresh
  launch evidence root before implementing.
  ```bash
  PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 \
    -m unittest discover -s experiments/lora_necessity -p 'test_tokenizer_host.py' -v
  ```
- [x] Implement one bootstrap using the approved interpreter with `-I -S -B`.
  Check expected hostname/UID before emitting READY. The client must receive
  and validate READY before sending any private input. Bind exact source hashes,
  inventory hash, profile hash and run ID; reject drift and unsafe filenames.
  Add only explicit source/site-packages paths, never execute `.pth` startup.
  Before imports, fix offline/thread environment and install fail-closed
  Linux task-creation denial; behavior-check fork/thread denial within this same
  bootstrap. Reference primary kernel documentation if using seccomp.
- [x] Reuse accepted APIs, in order:
  ```python
  validate_inventory(inventory_raw, inventory_sha256)
  validate_profile(profile_raw, profile_sha256)
  create_snapshot(source, parent, profile_raw, profile_sha256)
  report = run_audit(
      parent / run_id, profile_raw, profile_sha256,
      inventory_raw, inventory_sha256)
  ```
  `create_snapshot` owns creation of `parent/run_id`; do not precreate that path.
  Stage bootstrap sources privately outside `run_id`, inside an exclusively
  created per-attempt directory, with a clear recorded path.
- [x] Keep SSH stdout for framed result plus bounded application diagnostics;
  SSH stderr is transport diagnostics. Redirect native fd 1/2 during imports and
  audit, not just Python `sys.stdout`. Enforce diagnostic bounds at production
  as well as reception. Keep no unbounded disk log; result travels privately.
  Refuse frame truncation, channel duplication, identity mismatch and trailing
  bytes. Audit failure is valid failure evidence, not transport success => PASS.
- [x] Reuse `LaunchWindow` locally. Send a conservative remaining deadline
  after READY; remote ordinary timeout must subtract time already elapsed
  receiving inputs, not restart a fresh 540 seconds. Reap only the owned local
  SSH subprocess/group on failure; remote state stays unknown after disconnect.
  Save return code, exact hashes, elapsed time and partial bounded outputs.
- [x] Repeat the focused test command until passing, then run the existing
  tokenizer tests. Report exact counts, skipped Linux-only coverage and risks.

## Task 2: Review And One Real Audit

- [x] Generate a diff for the three new files (they are untracked), not an empty
  `git diff HEAD`, and obtain independent spec/quality review.
- [x] Resolve concrete findings and rerun the affected checks before execution.
- [x] Freeze command, payload, all source hashes and approval references in the
  fresh launch evidence root. Recheck the full 785-file baseline; only the
  append-only progress ledger may change.
- [x] Run one authorized connection through the existing secure SSH options.
  Record the attempt before starting. No hidden test connection or retry.
- [x] Validate returned private evidence by reconstructing the report:
  ```python
  rebuilt = build_report(
      inventory_raw, inventory_sha256, profile_raw, profile_sha256,
      result["private"]["records"], result["private"]["finalization"],
      failure=failure_without_reason, abort_failed=result["public"]["abort_failed"])
  assert rebuilt == result
  ```
  Derive `failure_without_reason` as `None` or the returned phase/ordinal pair.
  Also compare the private inventory to the approved local inventory.
- [x] Publish only a local, non-private summary: per-category scheduled/checked/
  overflow/unexecuted counts, token maxima, stop reason, and remaining limitations.
  Preserve private answers/IDs outside Git; append ledger and seal new evidence.
