# Candidate Executor Implementation Plan

**Goal:** Connect the existing public-request contract to a bounded, parent-owned
executor, and qualify only reviewed fixed fixtures before generated candidates.

**Architecture:** A pure protocol module validates requests and turns parent
observations into receipts. A selector-based controller owns processes and byte
limits. A child-only worker runs after the existing V3 isolation bootstrap.

**Tech Stack:** Python 3.12 standard library, existing bubblewrap/libseccomp.

## Constraints

Follow `docs/superpowers/specs/2026-11-17-lora-candidate-executor-design.md`.
Use the existing experimental checkout; no new worktree, installation, training,
hidden scoring access, prior-stage replay, commit or push in this component stage.
All previous files except this stage's design/plan remain unchanged.
No arbitrary candidate execution on the local host.
Implementation tasks are tightly coupled: main implements the controller while
one scoped worker implements the pure protocol, with disjoint files.
Independent review precedes any new remote fixed-fixture execution.

## Task 1: Pure protocol

- [ ] Create `candidate_execution_protocol.py` and its focused test file.
- [ ] Run missing-module assertion RED before implementation, then GREEN.
- [ ] Export `READY = b"TLF_CANDIDATE_READY_V1\n"`.
- [ ] Export `prepare_request(request) -> (request_sha256, payload_bytes)`.
  Accept only the existing exact version/round/attempt/task/arm/case/entry/source/args
  fields. Use strict finite canonical JSON, reject bool version/round and extra
  fields, valid identities and source UTF-8 <=65536 bytes. Output only
  source/entry_point/args plus newline, at most 1048576 bytes.
- [ ] Export `InfrastructureError(RuntimeError)`.
- [ ] Export `make_receipt(request_sha256, *, returncode, ready, input_complete,
  reaped, eof, termination, result_raw)`. Parent facts must be valid; failed
  initialization, incomplete input or unreaped process raise InfrastructureError.
  Parent timeout/output_limit takes precedence over child data. Otherwise require
  all EOF and normal exit for completed; abnormal exit/malformed outcome is
  program_error. Accept only return/value or raises/ValueError, strict JSON <=16384.
- [ ] Test pure facts, duplicate/malformed output, spoofed receipt/status fields,
  type distinctions and existing feedback integration without executing candidates.

## Task 2: Controller

- [ ] Create `candidate_execution_controller.py` and its test file.
- [ ] RED then GREEN using fixed local helper programs, never generated candidates.
- [ ] Export `run_request(request, argv, *, artifact_dir, cwd, env,
  startup_timeout=10.0, candidate_timeout=10.0, output_limit=65536,
  result_limit=16384, reap_timeout=2.0) -> {"receipt": dict, "audit": dict}`.
- [ ] Validate all configuration and request before exclusive claim. Require an
  existing private artifact directory. Never invoke a shell. Replace only the
  literal `{control_fd}` token in trusted argv, passing that write descriptor.
- [ ] Selector loop owns stdout/stderr/control reads and nonblocking stdin writes.
  Do not send payload before exact READY. Bound each read to 8192 and combined
  retention to output_limit plus one detection byte. Close stdin after full input.
- [ ] Start candidate deadline on first attempted send, not after write completion.
  Include waiting for exit/EOF. Kill/reap owned process group on timeout, overflow,
  collector failures and exceptional exit paths. No retry, no receipt on infra error.
- [ ] Persist bounded raw streams, parent observation and receipt exclusively.
  Parent-observed first termination reason cannot be overridden by late output.
- [ ] Test normal/raises, noise/spoofing, double result, exit without result,
  output floods on both streams/control, handshake mismatch/timeout, large input
  backpressure, late result/hanging child, one-shot claim and input-not-sent-before-ready.

## Task 3: Child worker and pinned isolation integration

- [ ] Create `candidate_execution_worker.py` and focused tests.
- [ ] Compile/parse tests and fixed local toy functions only; no frozen task execution.
- [ ] Worker requires an explicit invocation after trusted isolation setup.
  READY goes to the inherited control descriptor, not stdout.
  Reads bounded source/entry/args; validates single-function syntax without repair.
  Only outcome goes to control; stdout/stderr remain untrusted diagnostics.
  Bind trusted serializer/writer before invoking candidate; validate returned
  built-in JSON types before serialization. Do not claim this is an unforgeable VM.
- [ ] Generate private bootstrap from pinned V3, retaining capability/filter policy.
  No source is sent until isolation/setup passes. Verify byte transformations.
- [ ] Independent reviewer examines exact modules, fixed fixtures and bootstrap.
- [ ] If local tests and review pass, run one bounded remote fixed-fixture batch
  with current route/runtime pins. Do not retry automatically on authentication.
  Stop batch on infrastructure failure; normal fixture errors are expected results.

## Task 4: Verification and handoff

- [ ] Fix findings using reproducing tests; re-review relevant changes.
- [ ] Run focused tests then existing experiment suite.
- [ ] Reconstruct remote observations, preserve prior seal hashes, record exact
  admission limits and remaining work in a dated result file.
- [ ] Do not start four development tasks until full candidate admission and
  generation/cost integration are independently established.
