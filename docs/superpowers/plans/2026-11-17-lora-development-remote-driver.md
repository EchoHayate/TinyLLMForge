# Development Remote Driver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the accepted eight-episode coordinator to the verified real base and unchanged isolated executor, then package and independently reconstruct one separately authorized run.

**Architecture:** Two review units: a fixed driver with synthetic tests; then its pinned stdin package, one-shot transport and independent evidence reconstruction with synthetic tests. No job framework, coordinator rewrite or replay of the consumed smoke.

**Tech Stack:** Existing Python 3.12, stdlib/unittest, approved observer/admission/scoring modules and existing offline Torch/Transformers only inside the future real driver.

## Global Constraints

- Implement `../specs/2026-11-17-lora-development-remote-driver-design.md`; use `/tmp/tlf-candidate-executor-20261117-gb1gZX/development-remote-integration-notes.md` as the source-backed integration boundary.
- D01–D04/base/text, eight episodes, sixteen scheduled generations. No training, adapters, LoRA loading, transfer/retention execution, extra smoke calls or hidden-answer feedback.
- During implementation: synthetic bytes/callbacks only; no real data/model/remote/worker execution, installs, telemetry, commits, pushes, stage/protected edits or consumed-entry replay.
- Use `apply_patch`; do not change coordinator, accepted dependencies, old harnesses or their reports. All future files below are new and require Main's implementation dispatch.
- No aggregate quotas or automatic cleanup/retry. Preserve existing isolation, per-child limits and reap checks.
- Source/manifest hashes bind bytes; provenance additionally requires actual loaded-source/runtime/model checks and independent reconstruction. All existing scientific/readiness FLAGS remain false.
- This document contains future commands, not a preparation or launch grant.

## Contracts and Acceptance Gates

Coordinator implementation is now frozen at
`3d03838977ddd20982db180393557e040ac38f6261f30c6a92496b4257633623`.
Source reference:
`/tmp/tlf-candidate-executor-20261117-gb1gZX/coordinator-runtime-report.md`.
Main's earlier324-test baseline exposed no test failures; independent review
then found finite elapsed aggregation and failed-input consistency defects.
Both fixes are delivered with105 coordinator/205 pure-regression tests;
Main's fresh332-test regression passed. Independent re-review in
`coordinator-runtime-review-r2.md` closed both findings with spec/quality PASS.
Main accepts this coordinator component, not remote execution or provenance.
These are overlapping reported runs, not counts to add.
The concrete API below matches the frozen source already inspected. At
implementation handoff bind its acceptance decision/hash; reopen interfaces
only if review changes that source.

```text
run_development(supervision_raw, prompts_raw, source_raw, manifest_raw, *,
                run_id, artifact_dir, bindings, generate, execute_public, execute_final)
generate(messages, *, expected_tokenizer_sha256, call_id, artifact_dir)
execute_public(request, *, artifact_dir)
execute_final(request, *, artifact_dir)
```

Coordinator creates its artifact directory exclusively. Generation returns exactly
`generation/observation/record_cleanup`; execution returns exactly
`admission/audit/receipt`. `DevelopmentRunError` carries `report`, `terminal_path`,
`terminal_sha256`, `record_cleanup`, `record_cleanup_complete`,
`persistence_errors`, with the first error as its cause. Confirmed driver
violations use `BoundaryProtocolError`; unexplained I/O/transport failures do not.
Elapsed aggregation has an additive `aggregation_error` field. Finite individual
durations may yield an unrepresentable total: preserve them and null total/subtotal
with incomplete calibration, not zero or infinity. An unexpected reducer failure
uses a degraded report with null `counts`, `costs`, `by_arm`, while retaining
the eight episode ledgers and `failure.report_error`. Driver export must preserve
that report; independent reconstruction must not assume summaries always exist.

The inspected coordinator's bindings have exactly `component_sha256`,
`runtime_sha256`, `model_sha256`, `tokenizer_state`, `execution`.
Use its exact `COMPONENTS` mapping, not the full package closure as extra keys.
Public/final execution each binds `argv`, `startup_timeout`, `candidate_timeout`,
`output_limit`, `result_limit`, `reap_timeout`. Runtime/model binding meanings:
`runtime_sha256=digest(runtime_manifest)` and `model_sha256=expected_base_sha256`.
The driver also records full checkpoint-file pins and complete package sources.

Package approval must fix run ID, host/UID, Python path/hash/version, offline
package path/version inventory, GPU UUID, checkpoint path/file sizes/hashes,
expected base/tokenizer state, source closure, unchanged launcher/worker,
per-child limits and three positive deadlines:
`driver_timeout_seconds`, `export_reserve_seconds`, `transport_timeout_seconds`.
Require transport > driver + export reserve. Main binds the new work horizon to
the existing `pilot_protocol.configuration()["limits"]["workflow_seconds"]`:
driver3600s, export reserve120s, transport3900s. This is a new run's explicit
timeout configuration, not an extension or replay of the old540s/30s probes.
For both public/final children retain the accepted controller defaults:
startup10s, candidate10s, output65536 bytes, result16384 bytes, reap2s.
No aggregate memory/disk quota or disconnect-cleanup promise is added.

The runtime manifest has exactly: `hostname`, `uid`, `python` (version string),
`executable` (absolute host interpreter path), `executable_sha256`, `packages`,
`versions`, `gpu_uuid`, `model_root`, `checkpoint_files`, `artifact_parent`,
`expected_base_sha256`, `expected_tokenizer_sha256`. Do not mistake the retained
training manifest's `python="3.12.13"` for an executable path. Checkpoint entries
are exact `bytes`/`sha256` objects. Pin actual values at separately reviewed
preparation; the inherited training manifest is a source of evidence, not the
new package itself. `limits` has exactly `public`, `final` and the three named
deadline fields; each domain contains the five child limits above, without argv.
The driver constructs argv from verified executable/launcher/worker snapshots.

Decoded packages keep the same outer keys as wire JSON, but `modules` maps
allowlisted module names to bytes, `data` maps four frozen basenames to bytes,
and `launcher`/`worker` are bytes. Runtime manifest and limits remain native JSON
objects. The package decoder validates wire entry sizes/hashes before returning
this form. Synthetic driver tests supply this same decoded structure.

## Unit 1: Fixed Real Driver and Synthetic Tests

**Create:** `experiments/lora_necessity/development_remote_driver.py`
and `experiments/lora_necessity/test_development_remote_driver.py`.

**Interfaces:**

```text
run_driver(package, *, artifact_root, deadline_ns, emit) -> driver envelope
make_callbacks(model, tokenizer, *, bindings, launcher, worker, driver_state)
    -> (generate, execute_public, execute_final)
export_evidence(root, *, run_id, emit) -> export receipt
```

`package` is Unit 2's fully verified/decoded snapshot, not a filesystem path.
`artifact_root` is a new private driver directory; pass its nonexistent
`coordinator` child to `run_development`. Derive the root once as
`Path(runtime_manifest["artifact_parent"]) / ("development-" + run_id)`;
use exclusive creation, not a fallback randomly named directory.
`deadline_ns` is the absolute monotonic driver horizon set by the bootstrap,
not a fresh allowance starting after imports. `emit(record)` writes one canonical
JSON line and flushes. The public functions never accept replacement production
loaders/admission modules; tests patch narrowly scoped helpers.

- [ ] **RED:** Add file-existence assertions before deferred imports. Patch
  `_prepare_base` to return sentinel model/tokenizer objects and patch the three
  approved call boundaries; forbid Torch/Transformers imports, network access
  and subprocess creation. Test one coordinator call and exact callback kwargs.
  Run only:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error -m unittest discover -s experiments/lora_necessity -p test_development_remote_driver.py -v
```

- [ ] **Base preparation:** Implement `_prepare_base(package, root, state)` by
  extracting the following explicit blocks from consumed
  `/tmp/tlf-candidate-executor-20261117-gb1gZX/real_generation_smoke.py` as reviewed
  implementation material. Do not import/run that file or copy its top-level body.

| Source block | Driver behavior and evidence |
| --- | --- |
| Python/host/UID/version inventory | Check 3.12.13, isolated/no-site/no-bytecode flags, actual host/UID, exact installed inventory; record facts before model load |
| Offline environment and GPU checks | Preserve offline/no-telemetry/single-thread flags; target exact UUID, A100, free≥65536 MiB, used≤512 MiB, no compute apps; save both raw `nvidia-smi` outputs; never kill others |
| Checkpoint loop | Hash each declared file; verify bytes and size/mtime_ns/inode before/after reading and again after load |
| Base loading | `local_files_only=True`, `trust_remote_code=False`, safetensors, BF16 eager, CUDA eval/no-grad, `use_cache=False`; 36 layers; unchanged dtype/base/tokenizer identities |
| Determinism/integrity | Seed42, deterministic algorithms, TF32 off; preserve RNG snapshots after load; verify base digest, gradients, tokenizer state and RNG at finish/failure when available |

Reuse `training_worker._state/_tensor_digest` only for state hashing, imported
inside verified runtime setup. Package `training_inputs` too, as required by
the source closure; never call `train_adapter`, instantiate `LoRAProjection`
or load an adapter. No new general model abstraction.

Remove the old boundary search, fixed smoke messages, fixed smoke call IDs,
extra 4097 request, smoke result schema and old alarm. Use no old entry point.
Record separate GPU/checkpoint/load/initial-integrity/final-integrity/measurement/
coordinator/driver wall durations; retain nulls for unavailable observations.
These nested durations are not summed into a claimed disjoint total.

- [ ] **Generation closure:** Preserve the exact coordinator return envelope.
  Save `driver-measurement.json` in that call's directory using observer `_save`;
  it contains `call_id`, messages, measurement, measurement elapsed and hashes.
  Call real `measure(tokenizer, messages, answer=None)` once in this callback,
  in addition to the generator's own unchanged internal measurement. It is not
  another model call. Critically, `INPUT_LIMIT` must still reach the observer:

```python
def generate(messages, *, expected_tokenizer_sha256, call_id, artifact_dir):
    measured = measure(tokenizer, messages, answer=None)
    save_measurement(artifact_dir, call_id, messages, measured, driver_state)
    result = generate_observed(
        model, tokenizer, messages,
        expected_tokenizer_sha256=expected_tokenizer_sha256,
        call_id=call_id, artifact_dir=artifact_dir)
    try:
        verify_real_generation(result, messages, measured, tokenizer, bindings)
    except BoundaryProtocolError as error:
        error.record_cleanup = result["record_cleanup"]
        raise
    return result
```

`save_measurement` and `verify_real_generation` are private functions in the
same driver module. Record measurement timing around the call in the actual
implementation. Save failure association/cleanup in a guarded `finally` without
overwriting an original exception; no observer retry.

`verify_real_generation` checks measured state/message/prompt hashes and token
IDs/counts against the successful generation and saved observer records, one
model call/root-forward counts, ≤4096 input, 1..256 output, EOS/length rules,
actual vocabulary membership, CUDA positive allocated/reserved peaks and false
FLAGS. Decode returned output IDs with the actual tokenizer, excluding only
terminal EOS, using `skip_special_tokens=False,
clean_up_tokenization_spaces=False`; require exact candidate text/hash.
Recheck tokenizer state after this validation.

On normal input >4096, preserve the observer's failed claim/observation and
original exception. Record the **actual** measured count (not necessarily4097);
never adapt the smoke's exactly4097 acceptance predicate or fabricate a zero
cost generation. If supplementary measurement or saving itself fails, abort
before model entry and retain that different failure, rather than inventing
observer evidence.
Measurement entry/failure before `generate_observed` is only driver callback/
measurement activity: never record it as an actual model call or root forward,
and never fabricate observer claim/observation records or zero token costs.
For a genuine measured4097 request whose measurement was saved successfully,
invoke `generate_observed` once and preserve its original refusal, claim and
failed observation. The driver must not replace that refusal with a precheck
exception lacking observer evidence.

- [ ] **Execution closures:** Construct the same unchanged launcher/worker argv
  for both domains, with one `{control_fd}` and the sealed worker's base64/hash.
  Host argv uses verified Python `-I -S -B -c`, launcher text, worker base64,
  worker hash, control token. Keep launcher/worker source hashes unchanged.
  Host cwd is the owned case directory; use the reviewed minimal explicit host
  environment, never inherit the model/credential environment into the worker.

```python
def execute_final(request, *, artifact_dir):
    result = development_final_admission.run_candidate_request(
        request, bindings["execution"]["final"]["argv"],
        artifact_dir=artifact_dir, cwd=artifact_dir, env=host_environment,
        **limits_without_argv(bindings["execution"]["final"]))
    verify_startup((artifact_dir / "stdout.bin").read_bytes(), worker)
    return result
```

Public closure is identical except it calls
`candidate_execution_admission.run_candidate_request` and uses `"public"` limits.
`limits_without_argv` returns the other five bound fields unchanged.
Every successfully returned runner result must pass
`verify_startup(saved_stdout_bytes, worker)` using the preserved `stdout.bin`,
not returned summaries or a separately generated diagnostic.
`verify_startup` checks the first four stdout lines: userns policy,
host no-new-privileges, all-capabilities-zero, then the fixed JSON diagnostic.
Never search later candidate stdout for a replacement passing diagnostic.
Require exact kind/python/capability/syscall schema and the exact 17 true
checks from pinned worker source, before accepting the receipt. Its
`generated_candidate_executed=false` and `candidate_execution_admitted=false`
remain explicitly **pre-input** facts, not a later execution denial.

On execution exceptions, inspect any existing diagnostic/streams in `finally`.
After callback failure, missing, partial or unreadable startup evidence alone is
incomplete, not automatically a protocol violation. Do not feed absent/truncated
bytes through the successful-run validator and relabel its missing-evidence
exception as a safety failure. Collect every readable diagnostic and record
collection errors separately. Only positively inconsistent evidence, such as a
complete diagnostic with a false fixed check or conflicting runtime identity,
adds a protocol error. If another exception already exists, retain that original
exception and its cause first and record the additional violation separately.
Never rewrite admission, audit, raw streams or receipt, and never retry.

- [ ] **Driver outcome/cleanup/export:** Put setup, one coordinator invocation,
  final integrity and evidence export inside one primary-error-preserving flow.
  Save `driver-claim.json` before loading the model; emit `STARTED`,
  `RUNTIME_VERIFIED`, `GPU_VERIFIED`, `MODEL_VERIFIED`, `COORDINATOR_ENTERED`
  only after their named facts. Failed/absent phases do not imply zero work.
  Save `driver-result.json` or `driver-failure.json`; retain any coordinator
  `DevelopmentRunError` envelope unchanged, alongside driver-level violations.

Driver envelope fields are `run_id`, `status`, `coordinator` (return/error
envelope or null), `integrity`, `costs`, `failure`, `secondary_errors`,
`record_cleanup`, `export`, `training_steps=0`, `adapter_mounted=false`, and
false FLAGS. Status is completed/aborted; calibration remains a separate field
inside coordinator evidence. Late integrity failure can invalidate driver
acceptance without rewriting the already committed coordinator terminal.

Reuse observer `_save/_failure`; link is commit. Postcommit temporary unlink
diagnostics remain nonfatal/separate; do not retry or reverse a committed
terminal. Before-commit failures remain primary. Best-effort failure integrity
must not replace the original error or reuse missing RNG/model snapshots.
The approved driver deadline stops new work; an explicit export reserve follows.
Timeout/interrupt paths attempt export, but cannot promise cleanup on host loss,
SIGKILL or blocked native code. Never imply killing local SSH proves remote stop.

`export_evidence` snapshots only regular files beneath the owned root, rejecting
symlinks/escapes and recording stat/hash changes or unreadable files. Export all
available coordinator/current-call records, raw streams, driver events,
measurements, cleanup and partial results. Exclude checkpoint/runtime files.
Emit `EXPORT_BEGIN` with inventory, then ordered base64 `FILE_CHUNK` records of
≤65536 raw bytes, and `EXPORT_END` with count/inventory digest/errors.
Every frame has `phase` and `run_id`; each inventory entry has `path`, `bytes`,
`sha256`, `mode`; each chunk has `path`, `index`, `offset`, `bytes`, `base64`.
Zero-byte files still appear in inventory. Paths are root-relative POSIX names.
Chunking is not an aggregate quota. Failed reads/writes produce incomplete
export, never a complete-looking inventory. Keep remote files; do not delete.
Terminal `DONE` requires completed driver + completed export; otherwise emit
`FAILED`/`EXPORT_INCOMPLETE` if possible. A broken output channel leaves local
transport uncertain and remote evidence intact, not a fabricated terminal.

Avoid self-referential export and late file mutation: freeze event persistence
before inventory creation. The committed driver result/failure has `export=null`
because export has not occurred yet; do not rewrite it afterward. Export frames
and the final returned/emitted envelope carry the export receipt and the saved
terminal's path/hash. The exporter does not write its own growing stream into
the root it is snapshotting. Failure while emitting retains the original driver
error and records transport/export incompleteness separately.

- [ ] **GREEN tests:** Besides the two examples below, cover wrong runtime/GPU/
  checkpoint pins before model entry; failed partial setup; state/RNG drift;
  callback envelope identity; strict stdout prefix/17 checks; no final-to-public
  conversion; observer exception after saved records; postcommit unlink failure;
  primary+secondary failure; missing measurement and interrupted export.

```python
def test_input_limit_still_enters_observer_once(self):
    self.measure.return_value = self.input_limit_measurement(5000)
    failure = self.observer_failure()
    self.observer.side_effect = failure
    with self.assertRaises(type(failure)) as raised:
        self.generate(self.messages, **self.call_keywords)
    self.assertIs(raised.exception, failure)
    self.observer.assert_called_once()


def test_fake_later_diagnostic_cannot_replace_bad_prefix(self):
    self.write_stdout(self.bad_prefix + self.passing_diagnostic)
    with self.assertRaises(self.module.BoundaryProtocolError):
        self.module.verify_startup(
            (self.case_root / "stdout.bin").read_bytes(), self.worker_snapshot)
```

Synthetic helpers create only literal records/streams; no candidate is evaluated.
Patch base preparation and approved boundaries, not coordinator's control flow.
Add explicit paired regressions: callback failure plus absent/truncated/unreadable
startup preserves the original exception with incomplete evidence and no invented
protocol error; the same failure plus a readable contradictory complete diagnostic
preserves the original exception and adds the separate protocol finding.
Measurement failure before observer entry must assert zero observer invocations
and no asserted model-forward evidence; genuine4097 must assert one invocation
and retention of the original observer refusal and saved failed observation.
Run the focused Unit1 command; save RED/GREEN output and actual counts privately.
Independent review closes this unit before packaging a real payload.

## Unit 2: Pinned Package, Transport and Independent Reconstruction

**Create privately under `/tmp/tlf-candidate-executor-20261117-gb1gZX/`:**
`development_remote_package.py`, `development_remote_transport.py`,
`verify_development_remote.py`, `test_development_remote_package.py`.
Do not reuse any consumed prepare/run/verifier CLI or create a real package
during this implementation task.

**Interfaces:**

```text
build_package(module_sources, frozen_bytes, launcher, worker, *,
              run_id, runtime_manifest, limits) -> bytes
decode_package(package_raw) -> decoded package
build_transport(package_raw, *, approved_ssh_argv, remote_python, deadlines) -> dict
execute_once(transport, package_raw, *, receipt_dir) -> transport receipt
validate_export(stdout_raw, stderr_raw, *, expected) -> decoded evidence
verify_development(evidence, *, approved_package_raw, transport_receipt) -> dict
```

All except `execute_once` are pure functions over supplied snapshots.
The execution function has no import-time/automatic call; tests replace Popen/
subprocess with a deterministic fake. CLI exposure requires explicit arguments
and a fresh claim; no retry flag or resume-by-new-ID helper.

- [ ] **RED:** Write in-memory package fixtures, one-bit mutations, a stdlib
  fake subprocess, and literal export streams. Test missing modules with deferred
  import assertions and run only:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lora_necessity /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error -m unittest discover -s /tmp/tlf-candidate-executor-20261117-gb1gZX -p test_development_remote_package.py -v
```

- [ ] **Package:** Canonical JSON contains exact fields `kind`, `version`,
  `run_id`, `runtime_manifest`, `limits`, `module_order`, `modules`, `data`,
  `launcher`, `worker`. Byte entries contain `bytes`, `sha256`, `base64`;
  module names/data basenames are unique and strictly allowlisted. Data is the
  four original frozen views, validated against original PINS/loader schemas
  on the trusted side; never synthesize actual task material.

Use the notes' local dependency order, then append accepted coordinator and
driver: pilot_protocol, validation_contract, candidate_execution_protocol,
candidate_feedback, development_final_protocol, development_final_scoring,
prompt_rendering, development_schedule, candidate_execution_controller,
candidate_execution_admission, development_final_controller,
development_final_admission, tokenizer_measurement, generation_core,
observed_generation, training_inputs, training_worker, development_coordinator,
development_startup_evidence, development_remote_driver. Recheck static imports
after acceptance. The startup helper is outside the coordinator's narrower
COMPONENTS map but is a required driver dependency. Torch must
be enabled only after verified offline environment/runtime setup, before loading
training_worker; its helper dependency does not authorize training.

Package validation rejects missing/extra closure members, filename aliases,
duplicate JSON keys, bad base64, count/hash mismatch and reordered unresolved
dependencies. Validate **all** source/data/launcher bytes before executing any
embedded module. Source material comes from verified byte snapshots, not a later
path reread. Bind package builder, transport, verifier and test evidence in the
external preparation manifest too; avoid a self-referential package hash.

- [ ] **Stdin bootstrap:** Do not embed the whole package/program in one SSH
  argument. `build_transport` returns `ssh_argv`, `remote_command`, package
  length/hash, bootstrap hash and deadline fields. Build the shell command from
  an argv list using `shlex.join`; its final Python arguments are `-I -S -B -c`,
  the small bootstrap, expected decimal length and expected SHA256.
  Input collection must finish before parsing or executing source:

```python
length = int(sys.argv[1])
expected = sys.argv[2]
raw = sys.stdin.buffer.read(length + 1)
if len(raw) != length or hashlib.sha256(raw).hexdigest() != expected:
    raise ValueError("incomplete or changed package")
```

The transport closes stdin after the exact package, so the extra-byte/EOF
check can complete. Bootstrap then performs strict stdlib package validation,
fresh Python/host/runtime checks, offline path setup, verified ordered module
loading, and one call to `run_driver`. No partial stream is executable.
Capture `time.monotonic_ns()` before reading stdin. After package verification,
derive the absolute driver horizon from that start and approved driver timeout,
arm its remaining time before imports, and reject an already exhausted horizon.
Pass it to the driver unchanged; reserve export time only after work stops.
Main's transport margin must cover SSH startup as well as that driver/export
horizon. An expired outer transport never authorizes a replacement run.
The bootstrap code is a literal pinned constant in the package module; tests
exercise its byte gate with a sentinel loader, not a real driver/model.

- [ ] **One-shot transport:** Validate approved SSH argv without reading routes/
  credentials in tests. Preserve authentication, host-key policy and explicit
  Kerberos environment from the independently approved driver route; do not
  weaken authentication or infer its current state from old receipts.
  Before any subprocess entry, exclusively persist a 0600 local claim binding
  run ID, package length/hash, complete argv/command hash, bootstrap and deadlines.
  Open private binary stdout/stderr receipts exclusively and stream SSH output
  there; pass package bytes as stdin, not DEVNULL. No shell=True, inherited
  credential serialization, output truncation or auto retry.

Record exact start/end/returncode/timeout/bytes hashes and one invocation.
Retain a consumed claim on local launch failure, timeout or disconnect.
Local timeout may terminate/reap owned SSH, but remote status remains unknown
without evidence. No DONE means not fully observed, not unexecuted. A later
read-only recovery of the same remote root requires Main authorization; this
transport does not create a new run to compensate.

- [ ] **Independent reconstruction:** `validate_export` rejects path traversal,
  symlinks, duplicate/missing/reordered chunks, changed sizes/hashes, malformed
  frames, impossible phase order, duplicated terminal and missing export
  terminal; retain raw streams regardless.
  Verification reads approved package snapshots only after stdlib hash checks,
  imports only accepted **pure** schedule/protocol/feedback/scoring helpers,
  and never calls coordinator, driver, observer, tokenizer/model or executors.

Rebuild eight identities and sixteen planned calls from frozen inputs. For each
entered call independently match original/second-round messages, measurement
prompt/IDs/count/hash, tokenizer state, saved claim/generation/observation,
candidate bytes, EOS, CUDA and costs. Reconstruct public requests, each receipt
from exact control bytes/audit and its verified launcher diagnostic, then only
the original visible feedback. Rebuild final plans from saved second-generation
bytes; score ordered receipt prefixes and preserve every missing/invalid case.
Compare independently computed counts/calibration with coordinator fields,
including first-invalid recovery and both-arm saturation precedence.
Match the frozen runtime's clarified semantics: `missing_episode_positions`
contains ordinals; `unexecuted_final_cases` is scheduled cases minus final
callback entries, not minus receipts. An entered callback without positive
child-start evidence remains unknown, not unexecuted.

Validate driver host/runtime/checkpoint/base/RNG evidence and full module pins,
not just caller declarations. Missing authenticity/cost/stream evidence stays
unknown/incomplete; confirmed mismatches are invalid protocol. Local inspection
does not independently re-render a real tokenizer or prove a remote process
from arbitrary self-reported JSON; acceptance depends on the reviewed fixed
driver, bound transport and preserved evidence, not hash strings alone.

Result schema: `kind`, `version`, `run_id`, `process_status`,
`verification_status` (verified/incomplete/invalid), `functional_by_arm`,
`calibration`, `counts`, `costs`, `checks`, `reasons`, `source_sha256`,
`raw_stream_sha256`, false FLAGS. No LoRA benefit/generalization/launch claim.
Only an explicitly invoked verifier CLI publishes
`development-remote-verification.json`, exclusively; pure synthetic tests publish
no real acceptance file. Preserve primary errors and nonfatal postcommit cleanup
diagnostics as in the approved observer/publication approach.

- [ ] **GREEN/mutations:** Use both focused commands above. Add cases for package
  >old120000 representation without a giant argv; short/extra stdin before any
  loader; source/hash/module-order drift; duplicate claims and launch exceptions;
  timeout with partial stdout; missing export end; cleanup errors; late forged
  diagnostic; wrong candidate/call/argv/limits; first-invalid recovery; final
  missing denominator; >4096 observed rejection; fake zero costs; callback
  completed field inconsistent with raw evidence. No actual SSH/launcher/model.
  Preserve actual RED/GREEN commands, logs, hashes and counts in
  `development-remote-driver-implementation-report.md` under the private root.

## Exact Review Handoff

Main must bind Carver's accepted coordinator and all new driver/package/transport/
verifier source identities after review. Preserve sealed launcher SHA256
`e15df5c44094b5cf8bb3fadc40006cab4ec7f4ba01df373f9879f0e63a8cd465`
and worker SHA256
`20d7cd8abb3f09350c01ec47d30613761804dd0849620f808b82606110690896`.

The deadline and child-limit values are bound above by Main. No new scheduling
or policy decision is required. Runtime/model/host/GPU pins and
route are supplied by Main's approved manifest, not discovered by this sidecar.
After both units pass independent review, separately prepare and review the exact
one-shot package/transport; **do not launch as part of completing this plan**.
