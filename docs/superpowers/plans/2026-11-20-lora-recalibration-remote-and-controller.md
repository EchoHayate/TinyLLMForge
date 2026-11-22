# Fixed-Source Remote Supervisor And Local Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the local implementation of the fixed-source 480-item validation execution stack (`recalibration_execution_remote.py` and `recalibration_execution_controller.py`) with non-cyclic launch package binding, five-frame wire emission, exclusive stage/batch claims, and strict failure/cancellation cleanup, without executing stored tasks or launching live SSH during tests.

**Architecture:** `recalibration_execution_remote.py` is a pure standard-library trusted remote supervisor that verifies a non-cyclic bootstrap envelope (`LRB01` channels 1 and 2) and pinned backend source before emitting `READY` (`LRV01` channel 1), awaits controller grant and exact bundle bytes (`LRB01` channels 3 and 4) under a single monotonic anchor established before `READY`, runs the backend in an owned isolated child session matching preflight file-descriptor layout, and emits bounded `BACKEND_STDOUT`, `BACKEND_STDERR`, `DIAGNOSTIC`, and `DONE` (`LRV01` channels 2..5). `recalibration_execution_controller.py` builds and validates the non-cyclic full launch package (`config_sha256` -> 15 exact SSH commands -> `package_sha256`), enforces local exclusive `0700`/`0600` stage and batch claims, conducts the two-phase stdin/stdout SSH handshake across up to 15 serial batches within the 1800s stage budget, stops immediately on infrastructure/unknown/persistence failures while continuing on normal reference/witness mismatches, and wraps `recalibration_execution_evidence.build_evidence` while keeping `trusted_collection_accepted` and `run_authorized` false.

**Tech Stack:** Python 3.9+ standard library (`hashlib`, `json`, `math`, `os`, `platform`, `re`, `selectors`, `shlex`, `signal`, `stat`, `struct`, `subprocess`, `sys`, `time`) and pinned pure repository modules (`pilot_protocol`, `validation_contract`, `recalibration_data_contract`, `recalibration_validation`, `recalibration_execution_package`, `recalibration_execution_wire`, `recalibration_execution_evidence`), tested via `unittest` with artificial process/selector/clock boundaries.

## Global Constraints

- Spec authority: `docs/superpowers/specs/2026-11-20-lora-recalibration-fixed-execution-design.md` and `/private/tmp/lora-recalibration-evidence-20261120-dtPPVL/next-integration-design.md`.
- Private verification root: `/private/tmp/lora-recalibration-controller-20261120-sSYngX` (196 baseline LoRA files frozen).
- Do not edit any of the 196 frozen baseline files in `experiments/lora_necessity/`.
- Do not import or execute `validation_remote.py`, `validation_task_worker.py`, or any stored task source in host Python.
- Production entrypoints must NOT accept `session_factory`, `transport_callback`, `real=True`, or `approved=True`.
- No inline comments (`#`) and no single-letter variable names in Python source or test files.
- No real `subprocess.Popen` or `os.killpg` calls during unit tests; patch internal process/signal/selector/clock boundaries in `setUp`.
- Do not rerun unchanged historical component suites; run focused suites once per changed source/test hash and retain raw RED and GREEN logs plus exit codes.
- Do not commit, push, switch branches/worktrees, install packages, or run live SSH/GPU/model commands during implementation.

---

### Task 1: Trusted Remote Supervisor (`recalibration_execution_remote.py`)

**Files:**
- Create: `experiments/lora_necessity/recalibration_execution_remote.py`
- Test: `experiments/lora_necessity/test_recalibration_execution_remote.py`

**Interfaces:**
- Pure Python standard library only (zero repository imports so remote `-I -S -B -c` execution requires no repository search path).
- Inbound control framing (`STDIN_MAGIC = b"LRB01"`, `HEADER = struct.Struct("!5sBQ")`):
  - `encode_control_frame(channel, payload_raw) -> bytes`
  - `decode_control_frame(frame_raw, *, expected_channel) -> bytes`
  - Channel 1 (`BOOTSTRAP`, <= 65536 bytes): strict JSON `{"kind": "LORA_RECALIBRATION_BOOTSTRAP", "version": 1, "package_sha256": <64-hex>, "config_sha256": <64-hex>, "batch_index": <0..14>, "launch_config": <stage_config_dict>}`
  - Channel 2 (`BACKEND_SOURCE`, <= 65536 bytes): exact `validation_remote.py` raw bytes matching `BACKEND_SHA256 = "9be98748dfd2bdcdd132c6ed4c04c3dd67f11d49d6049fcbe7960bde622690da"`
  - Channel 3 (`GRANT`, <= 4096 bytes): strict JSON `{"kind": "LORA_RECALIBRATION_GRANT", "version": 1, "ready_sha256": <64-hex>, "granted_work_ns": <1..95_000_000_000>, "bundle_sha256": <64-hex>, "bundle_bytes": <1..131072>}`
  - Channel 4 (`BUNDLE`, <= 131072 bytes): exact `bundle_raw` bytes matching `bundle_sha256` and `bundle_bytes`, followed by immediate stdin EOF.
- Outbound wire framing (`WIRE_MAGIC = b"LRV01"`, `HEADER = struct.Struct("!5sBQ")`):
  - Emits exact 5 frames compatible with `recalibration_execution_wire.decode_wire` and `recalibration_execution_evidence.decode_receipt`.
- Entrypoint:
  - `supervise_batch(expected_config_sha256, expected_batch_index) -> int`: reads from `_stdin_fd()`, writes framed bytes to `_stdout_fd()`, verifies host/UID/executable/roots, creates owned `0700` `stage_root` (if first batch; verifies ownership/mode on subsequent batches) and exclusive `0700` `batch_root`, saves exclusive `0600` bootstrap/backend/expected_runtime claims before `READY`, enforces `<= 10s` `READY` budget and `<= min(95s, granted_work_ns)` total work budget from the pre-`READY` monotonic anchor (never reset on grant/bundle), spawns backend in a clean session (`close_fds=True`, `pass_fds=()`, `cwd=batch_root`) with matching file-descriptor layout, reaps child and verifies local group absence in a separate `<= 2s` cleanup window, and emits `BACKEND_STDOUT`, `BACKEND_STDERR`, `DIAGNOSTIC`, and `DONE` only after backend termination/cleanup attempt.

- [x] **Step 1: Write failing tests (`test_recalibration_execution_remote.py`)**

Cover: pure-stdlib import closure and absence of import-time side effects; `LRB01` control framing validation; full 5-frame roundtrip decoded by real `recalibration_execution_evidence.decode_receipt`; pre-`READY` deadline (>10s) rejection; host/UID/interpreter/config digest mismatch before `READY`; exclusive `batch_root` refusal on replay; grant tightening vs never extending the pre-`READY` 95s anchor; EOF/invalid grant/truncated bundle/trailing stdin bytes producing valid 5-frame `status="input_error"` with `backend=None` and zero backend launches; backend launch `OSError` producing `status="launch_failed"` with `backend=None` and `payload_received=True`; backend timeout, nonzero exit, unreaped child, surviving process group, and stdout/stderr overflow (`output_limit`); cancellation before backend start (`status="cancelled"`) and cancellation during backend run (kills/reaps owned group and re-raises original `BaseException` without forging a completed `DONE`).

- [x] **Step 2: Run focused RED and save log**

Run:
```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_execution_remote -v
```
Expected: FAIL with absent implementation assertion failures, saved to `remote-red.log` and `remote-red.exit`.

- [x] **Step 3: Implement `recalibration_execution_remote.py`**

Implement the minimal complete pure-stdlib supervisor adhering strictly to `READY_FIELDS`, `DONE_FIELDS`, `PROCESS_FIELDS`, and `LRV01` frame limits.

- [x] **Step 4: Run focused GREEN and save log**

Run the exact unittest command above; expect exit 0 with zero skips, saved to `remote-green.log` and `remote-green.exit`.

---

### Task 2: Full Launch Package And Serial Local Controller (`recalibration_execution_controller.py`)

**Files:**
- Create: `experiments/lora_necessity/recalibration_execution_controller.py`
- Test: `experiments/lora_necessity/test_recalibration_execution_controller.py`

**Interfaces:**
- Consumes: `recalibration_execution_package`, `recalibration_execution_wire`, `recalibration_execution_evidence`, `recalibration_execution_remote` (source bytes + control framing constants), `recalibration_validation`, `recalibration_data_contract`, `pilot_protocol`, `validation_contract`.
- Public functions:
  - `build_launch_package(raw_files, worker_raw, backend_raw, full_source_raw, preflight_raw, scope_raw, *, stage_id, run_ids, ssh_argv_prefix, ssh_environment, local_stage_root, remote_stage_root) -> dict`
  - `validate_launch_package(package, raw_files, worker_raw, backend_raw, full_source_raw, preflight_raw, scope_raw) -> dict`
  - `run_stage(package, raw_files, worker_raw, backend_raw, full_source_raw, preflight_raw, scope_raw) -> dict`
- Non-cyclic binding order:
  1. Verify 6 data pins, `backend_raw`, `worker_raw`, all 10 peer Python modules in `full_source_raw` (`pilot_protocol.py`, `validation_contract.py`, `validation_report.py`, `validation_real_report.py`, `recalibration_data_contract.py`, `recalibration_validation.py`, `recalibration_execution_package.py`, `recalibration_execution_wire.py`, `recalibration_execution_evidence.py`, `recalibration_execution_remote.py`) plus self (`recalibration_execution_controller.py` read from `__file__` at runtime verification), `preflight_raw` (strict JSON binding `expected_runtime` and verified artificial probe results), and `scope_raw` (strict JSON binding `stage_id`, `local_stage_root`, `remote_stage_root`, `ssh_argv_prefix_sha256`, and `environment_sha256`).
  2. Build `preparation` via `build_preparation(...)`.
  3. Construct `launch_config` (`config_sha256 = digest(launch_config)`), excluding `package_sha256` and command digests.
  4. Construct the 15 exact SSH `argv` lists invoking `exec <remote_python> -I -S -B -c <quoted_remote_source> <config_sha256> <batch_index>`, recording `argv_sha256` publicly and full `argv` in `package["private"]`.
  5. Construct `package["public"]` (`package_sha256 = digest(manifest)`) with `launch_package_structure_valid=True` and `run_authorized=False`, `trusted_collection_accepted=False`, `dynamic_checks_passed=False`.
- Serial execution (`run_stage`):
  1. Re-validate `package` and all input/source bytes; create exclusive canonical `0700` `local_stage_root` (consuming the stage identity permanently before any SSH launch) and persist `stage-claim.json` and `package-public.json` (`0600`, fsync + dir fsync).
  2. Start single 1800.0s stage monotonic clock with 5.0s finalization reserve (`work_cutoff = stage_start + 1795.0`).
  3. For each batch `0..14`:
     - Re-verify input/source digests and check remaining stage work window >= `102.0s` (`95 + 2 + 5`); if insufficient, stop before creating batch claim (`first_stop.reason = "stage_budget_exhausted"`).
     - Create exclusive `0700` `batch-XX-<run_id>` directory and persist `batch-claim.json` before `Popen`. Any persistence error stops immediately with zero further dispatches.
     - Spawn one owned SSH process group (`cwd="/"`, exact `ssh_environment`, `close_fds=True`, `start_new_session=True`, `shell=False`) under 110.0s SSH work deadline and 2.0s cleanup window.
     - Stream `LRB01` frames 1 & 2 (`bootstrap` + `backend_raw`), await `LRV01` frame 1 (`READY`) within min(remaining, 110s), validate via `validate_ready`, verify remaining stage/SSH budget >= `102.0s` (or compute tightened `granted_work_ns` when configured within stage/SSH budget while requiring >= `102.0s` before dispatch), stream `LRB01` frames 3 & 4 (`grant` + `bundle_raw`) and close `stdin`. If `READY` fails validation or budget check fails after `READY`, close `stdin` immediately without sending bundle bytes.
     - Collect remaining `LRV01` stdout frames (capped at `8388608` total wire bytes with `+1` overflow probe) and SSH stderr (capped at `131072` bytes with `+1` overflow probe), reap direct child and verify local group absence.
     - Persist raw `wire.bin`, `ssh-stderr.bin`, and `batch-transport.json` (`0600`, fsync + dir fsync).
     - Reconstruct cumulative evidence via `build_evidence(...)`. If the latest batch `can_continue` is `False` (any `receipt_error`, supervisor non-completed status, SSH non-completed status, backend `program_error`/`timeout`/`output_limit`, or incomplete 32-row observation) OR persistence failed, break immediately! Note: reference or witness semantic value mismatches where all 32 items completed normally have `stop_reason is None` (`can_continue is True`) and therefore continue through all 15 batches.
  4. Persist `execution-evidence.json` (the unaltered `build_evidence` dict) and `controller-summary.json` (`binding_complete=True` when all bindings/persistence succeeded, while `trusted_collection_accepted=False`, `run_authorized=False`, `dynamic_checks_passed=False`), preserving and re-raising any primary cancellation (`BaseException`) after bounded cleanup and best-effort persistence.

- [x] **Step 1: Write failing tests (`test_recalibration_execution_controller.py`)**

Cover: non-cyclic `build_launch_package` / `validate_launch_package` determinism; rejection of any tampered data file, source pin, controller self-source, preflight runtime, scope digest, duplicate run ID, or non-canonical root; zero task oracle/expected/witness leakage in remote bootstrap or SSH argv; full 15-batch 480-item simulated stage driving the REAL `recalibration_execution_remote` framing logic in memory -> `collection_complete=True`, `binding_complete=True`, `trusted_collection_accepted=False`; normal reference/witness semantic mismatch continuing all 15 batches while setting `data_checks_satisfied=False`; batch 0 infrastructure failure (`program_error`, `timeout`, missing `DONE`, SSH nonzero exit, wire overflow) stopping immediately at batch 0 (`not_dispatched` for batches 1..14, `unknown_after_dispatch` when wire lacks valid receipt); `READY` mismatch or insufficient remaining stage budget refusing to transmit bundle payload; exclusive stage/batch claim preventing replay; batch persistence failure preventing subsequent batch dispatch; cancellation preserving original `BaseException` after owned-group cleanup and persistence.

- [x] **Step 2: Run focused RED and save log**

Run:
```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_execution_controller -v
```
Expected: FAIL with absent implementation assertion failures, saved to `controller-red.log` and `controller-red.exit`.

- [x] **Step 3: Implement `recalibration_execution_controller.py`**

Implement the package builder, validator, and serial stage controller.

- [x] **Step 4: Run focused GREEN and narrow execution-stack regression**

Run `test_recalibration_execution_controller` first, then run the new execution-stack regression (`test_recalibration_execution_remote` + `test_recalibration_execution_controller`) once and save logs/exit codes. Verify all 196 baseline files remain byte-identical and AST/style/import closure checks pass.

- [x] **Step 5: Independent code review and seal**

Dispatch a fresh code-review subagent to inspect `recalibration_execution_remote.py`, `test_recalibration_execution_remote.py`, `recalibration_execution_controller.py`, `test_recalibration_execution_controller.py`, the plan, and saved RED/GREEN logs. Fix any P0/P1/P2 findings with TDD before sealing `final-verification.json`, updating `experiments/lora_necessity/20261120-recalibration-controller-results.md`, and appending to `.superpowers/sdd/lora-recalibration-validation-adapter-20261119.md`.
