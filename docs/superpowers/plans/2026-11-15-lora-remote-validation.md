# Remote D01 Validation Implementation Plan

> **For agentic workers:** Use subagent-driven-development for the independent report task and requesting-code-review for the integrated changes. Main owns the tightly coupled remote execution path.

**Goal:** Obtain one bounded real remote D01 matrix, without training.

**Architecture:** Local frozen controller sends only requests. Remote nonprivileged
bwrap runs a fresh Python worker for each request and returns bounded raw observations.
Local strict report compares them to answers never uploaded.

**Tech Stack:** Existing Python 3.12, unittest, SSH, bwrap, libseccomp via ctypes.

## Global Constraints

- Binding design: `../specs/2026-11-15-lora-remote-validation-design.md`.
- No install, telemetry, commit, push, new branch/worktree, training or generation.
- Preserve all frozen data and old simulation/tokenizer components.
- D01 only: 32 calls out of 1416; 2 seconds/item, 90 seconds/batch.
- stdout65536/stderr4096, SSH8MiB/128KiB and110seconds. No retry.
- Correct answers stay local; generated-candidate admission remains false.

## Task 1: Remote execution

Files: create `experiments/lora_necessity/validation_task_worker.py`,
`validation_remote.py`, `test_validation_remote.py`.

- [x] RED tests for missing worker/backend; fixed artificial return/ValueError,
  strict request fields, stdout pollution, program errors and serialization errors.
- [x] RED tests for collector: successful echo, simultaneous stdout/stderr, bounded
  overflow, timeout, early child exit while sending input, expired deadline.
- [x] Implement standalone worker and `collect_process(argv, input_raw, *,
  timeout_seconds=2, stdout_limit=65536, stderr_limit=4096, pass_fds=())`.
- [x] Implement `run_bundle(bundle_raw)` with the design's exact JSONL schema.
  Load libseccomp normally; export native filter to memfd; set no_new_privs before
  any bwrap launch. No silent dropping of unknown required syscalls.
- [x] GREEN focused standard-library tests. Remote artificial probes use the actual
  backend, namespace identity and benign paths, not production task strings.

Command:
`PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -m unittest discover -s experiments/lora_necessity -p 'test_validation_remote.py' -q`

## Task 2: Independent local report

Files: create `experiments/lora_necessity/validation_real_report.py` and
`test_validation_real_report.py`. Full schema in design; no imports of remote code.

- [x] RED for absence using `assertTrue(Path(...).is_file())` before import.
- [x] Build explicitly synthetic fixture observations from frozen schedule; check
  true matching semantics without claiming that fixtures ran remotely.
- [x] Implement `build_d01_report(source_raw, manifest_raw, bundle_raw, transport)`.
- [x] Test all32 versus missing31, wrong reference/witness, bool-vs-int outcome,
  malformed JSON/extra fields, duplicate/incorrect ordinal/run/request/bundle hashes,
  missing done, failed transport, nonempty stderr, incomplete stdin, unreaped worker,
  ignored-timeout spoofing and events after failed row.
- [x] GREEN; original `test_validation_contract.py` and `test_validation_report.py`
  remain unchanged and pass.

Command:
`PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -m unittest discover -s experiments/lora_necessity -p 'test_validation*.py' -q`

## Task 3: Integration and actual first batch

- [x] Review full additions, not an empty tracked git diff; fix material findings.
- [x] Run existing whole experiment unittest suite and snapshot hashes.
- [x] Save new probe bundle/remote code/SSH prefix and observations in private evidence.
- [x] Verify artificial probes and unchanged remote runtime; if any fails, do not send
  D01. Preserve failure; do not replay a consumed bundle.
- [x] With a new run_id, build requests from `worker_requests(... )[:32]`, save local
  bundle and command, then execute once. Collect events and actual transport.
- [x] Rebuild `build_d01_report(...)` locally, save full private report and sanitized
  count summary. Preserve negative results and missing rows.
- [x] Update README/results/ledger and seal evidence. No publication.

Closeout: D01 run `7aa42d991cab47eb83e916aefd2c4f23` received32/32;
reference8/8 and witnesses3/3 satisfy frozen checks. Independent code and receipt
reviews accepted this canonical driver path. One nonblocking raw/canonical JSON
digest compatibility finding remains documented. Full matrix has1384 rows left;
training and generated-candidate admission remain false. Evidence manifest:
`/private/tmp/tlf-remote-validation-20261115-WdUFIzYw/final-verification.json`.
