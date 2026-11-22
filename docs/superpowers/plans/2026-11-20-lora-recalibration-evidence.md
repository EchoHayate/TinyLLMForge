# Recalibration Semantic Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate contextual READY/DONE identities and lifecycle consistency, and reconstruct the fixed 480-position report from saved backend bytes without declaring simulated evidence real.

**Architecture:** Add one pure evidence module over the existing preparation, wire and content-report modules. Its launch context is explicitly an unverified declaration, not a complete executable package: receipt consistency can be true, but full launch binding and trusted collection remain false. The future full launch-package validator must supply the missing source/preflight/command/claim bindings before any real collection acceptance.

**Tech Stack:** Python standard library, existing strict JSON helpers, unittest; no new dependencies.

## Global Constraints

- Approved parent design: `docs/superpowers/specs/2026-11-20-lora-recalibration-fixed-execution-design.md`.
- Keep all frozen data, existing source/test files, preparation-only flags and original report flags unchanged.
- No stored task execution/import, SSH, network, model/tokenizer loading, training, installation, telemetry, bootstrap/flush, cleanup, branch/worktree changes, commit or push.
- Parent implements the immediate dependency locally; a fresh independent reviewer checks actual files afterward. The parallel research auditor has a disjoint private report path.
- Evidence root: `/private/tmp/lora-recalibration-evidence-20261120-dtPPVL`.
- Tests use manually constructed process/JSONL/frame receipts. Expected or witness values used in a fixture are artificial observations, never executed outcomes.
- Use apply_patch, no inline code comments or single-letter names. Standard-library unittest style overrides test-skill defaults. Test workflow is Python/default/basic/lite; prohibited bootstrap/telemetry/flush are omitted, with local evidence retained.
- This is not the remote supervisor, controller, complete launch manifest, or a real execution acceptance record.

## Task 1: Semantic Receipts And Content Reconstruction

**Files:**
- Create `experiments/lora_necessity/recalibration_execution_evidence.py`.
- Create `experiments/lora_necessity/test_recalibration_execution_evidence.py`.
- Parent appends the dedicated ledger and creates the result note after review.

**Interfaces:**

```python
def validate_ready(ready_raw, *, expected):
    ...

def decode_receipt(wire_raw, *, expected_sha256, expected):
    ...

def build_evidence(preparation, raw_files, worker_raw, backend_raw, source_raw,
                   records, *, stage_id, run_ids, context):
    ...
```

All arguments are plain builtin values, not subclasses. Local malformed
preparation/context/record shapes raise ValueError. A malformed attempted remote
receipt is retained by hash as a stopped batch with no verifiable transport.

### Exact Context And READY

Context has exactly `package_sha256`, `config_sha256`, `stage_root`,
`expected_runtime`. Hashes are 64 lowercase hex. Stage root is a canonical
absolute POSIX path, ASCII components `[A-Za-z0-9_.-]+`, no empty/dot/dot-dot
components, at most 256 bytes, final component equal to stage ID. Runtime has the
exact backend runtime shape: hostname, uid, python, python_root, python_sha256,
bwrap_sha256, seccomp_sha256, argv_prefix, no_new_privs, host_namespaces.
Preserve all argv elements and namespace values verbatim; no FD normalization.
This shape check is not a preflight or sandbox acceptance.

Expected READY has exactly:
`kind/version/stage_id/batch_index/run_id/preparation_sha256/package_sha256/
config_sha256/bundle_sha256/backend_sha256/worker_sha256/hostname/uid/
stage_root/batch_root/expected_runtime_sha256`.
Kind `LORA_RECALIBRATION_READY`, version strict int 1; batch index strict int 0–14;
stage/run IDs strict lowercase hex32 and distinct; fixed backend/worker pins.
`batch_root = stage_root + "/batch-%02d-" % batch_index + run_id`.
Observed READY has those keys plus strict int `ready_elapsed_ns` in [0,10e9].
Parse <=4096 bytes with duplicate/nonfinite rejection, validate all builtin
types, then exact canonical equality to expected excluding timing.
Return a detached parsed object. Expected values do not authenticate themselves.

### Exact DONE

Kind `LORA_RECALIBRATION_DONE`, version strict int 1. Exact fields:
`kind/version/stage_id/batch_index/run_id/package_sha256/config_sha256/
ready_sha256/payload_sha256/prior_frames_sha256/status/payload_received/
backend_pid/backend/work_elapsed_ns/cleanup_elapsed_ns`.
Identity equals READY. `ready_sha256` hashes raw READY payload.
`prior_frames_sha256` is exactly four lowercase SHA256 strings hashing each
complete frame, header included, in wire order. Hash before parsing READY/DONE.

`payload_received` is bool; payload hash equals bundle hash iff received, else
null. If backend is present, payload must be received, PID strict positive int,
and backend exact fields `status/returncode/elapsed_ns/process_reaped/stdin_complete`.
Rebuild stdout_hex/stderr_hex directly from frames and apply old `_process_result`.
Reaped iff returncode non-null; backend elapsed <= work+cleanup.
Supervisor status equals backend status and cannot be launch_failed when a PID
exists. Backend absent requires null PID and empty backend stdout/stderr;
status only input_error, launch_failed, timeout or cancelled; received payload
is required for launch_failed. No fabricated process result when absent.

Work/cleanup elapsed are nonnegative strict ints. Completed supervision requires
work <=95e9, cleanup <=2e9 and backend completed (thus exit0, input complete,
reaped). Failure timings may exceed budgets and remain retained failed evidence.
Other supervisor statuses are program_error, timeout, output_limit, launch_failed,
input_error and cancelled. Failure does not imply unreaped; conversely an
unreaped failure remains a valid but incomplete lifecycle observation.

Return exactly `ready/done/transport/receipt_binding_consistent/
backend_lifecycle_complete/payload_received/backend_started`.
Receipt consistency true is not whole-launch binding.

### Stage Records And Stops

Records are a list of at most 15, each exact
`batch_index/bundle_raw/ssh/wire_sha256`, contiguous indexes from 0. SSH is the
old strict process-result schema, its stdout is the raw wire; stderr is preserved
by the supplied record digest, never substituted for backend stderr. Failed SSH
may carry the existing one-byte overflow sentinel; a complete wire still obeys
the exact aggregate budget. Hash expected wire before semantic decoding.

No following record after the previous batch cannot continue. Continue requires:
verified receipt, supervisor completed, backend reaped/exit0/input complete,
SSH completed/reaped/exit0/input complete, SSH elapsed <=112e9, and original
content batch_complete. Semantic answer mismatch does not stop.
Received valid DONE with failed SSH retains valid backend content but cannot
continue. Backend absent or unverifiable wire maps the attempted batch to
`transport=None`; original report therefore retains unknown_after_dispatch.

Use original build_report once over the continuous mapped prefix. Check its
per-batch content completion before admitting a later record; a later record
after an outer failure must raise even if content alone is complete.
Output kind `LORA_RECALIBRATION_EXECUTION_EVIDENCE`, version1, stage/preparation/
declared context hashes, raw record hashes, content_report, batch observations
and first stop. `receipt_binding_consistent` is conjunction over a nonempty
supplied receipt prefix; `collection_complete` requires all 15 continuable
batches. `binding_complete`, `trusted_collection_accepted`, `dynamic_checks_passed`,
`launch_ready`, `run_authorized`, `tokenizer_length_validated` and research flags
are always false. `missing_bindings` explicitly lists full launch package,
live preflight, actual controller/claim evidence and independent acceptance.
No `real` or approval parameter. Never mutate original content_report flags.

- [x] **Step 1: Write complete RED tests**

Discovery-safe import with `setUp` assertion:

```python
if importlib.util.find_spec("recalibration_execution_evidence") is not None:
    import recalibration_execution_evidence as evidence
else:
    evidence = None

def test_raw_backend_report_is_reconstructed(self):
    report = self.report([self.record(0)])
    self.assertEqual(report["content_report"]["completed"], 32)
    self.assertFalse(report["trusted_collection_accepted"])
    self.assertFalse(report["binding_complete"])
```

Cover full 15-batch reconstruction; empty prefix; stale hashes; READY every
identity field; strict types including bool/int, subclasses, duplicate keys and
nonfinite JSON; malformed runtime and preserved FD changes; DONE frame hashes,
payload/identity/PID/reap/elapsed contradictions; backend absent; valid failures;
SSH failure versus backend success; missing/truncated DONE; no record after any
outer/content stop; reference mismatch and witness mismatch without stopping;
source/preparation mutation; forbidden approval kwargs; input/output non-aliasing;
AST/import audit excludes I/O, task backend, model and tokenizer.

- [x] **Step 2: Execute and preserve RED**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_execution_evidence -v
```

Save raw log and immediate exit under evidence root. Expect assertion failures
for absent implementation, not import errors. Do not execute stored programs.

- [x] **Step 3: Implement pure evidence module**

Build READY expectations only from reconstructed preparation and validated
context. Use parse_json/fields/_plain/encoded and decode_wire, never custom JSON
semantics. Decode each batch once. Map valid backend transport into original
build_report; failures keep attempted prefix and first-stop details. Do not
change prior modules to accommodate new fixtures.

- [x] **Step 4: Verify GREEN and narrow regression**

Run focused command above once per changed source/test identity, then exactly:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/experiments/lora_necessity" python3 -m unittest test_recalibration_execution_evidence test_recalibration_execution_wire test_recalibration_validation -v
```

Retain both raw logs and immediate exit codes. Do not rerun other old suites.
Independently check baseline preservation, module import closure and actual
source/test hashes. Report synthetic data as synthetic.

- [x] **Step 5: Fresh review and continued integration**

Fresh reviewer reads actual source/tests/plan and saved raw results, reports
spec compliance then code findings. Fix supported defects with failing tests,
reverify affected tests and request rereview. Append result and ledger only
after checks. Continue full launch/supervisor/controller work rather than
presenting this pure component as the research conclusion.

## Plan Self-Review

This covers semantic READY/DONE, raw backend reconstruction and dual lifecycle
stops from parent sections 4/6. Deliberately does not assert full section3 launch
binding: caller-supplied package/config hashes remain unverified declarations.
The report exposes this gap rather than upgrading PREPARATION_ONLY. Source
manifest, preflight, claim, streaming cancellation and real acceptance stay with
later execution responsibilities. Existing report's unknown semantics remain.

Completed local task on 2026-11-20: 29 expected RED assertion failures, 29 GREEN
passes, 87 narrow regression passes (including 29 focused), no errors/skips.
Parent and independent Russell review both verified the unchanged 190-file
baseline and exact diff/source/test identities. Spec compliant and quality
approved, zero P0/P1/P2 findings. No production/test edits after GREEN.
The overall research goal remains active. Full launch/supervisor/controller
integration is the next task, not a claim that these responsibilities are done.
