# Development Coordinator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the fixed eight-episode, two-round development coordinator with evidence-linked accounting and private final scoring.

**Architecture:** One serial coordinator consumes the separately accepted schedule, calls three injected trusted boundaries, and reduces their preserved records. Reuse the existing observer's record publisher, error formatter and generation validator; reuse the original public-feedback renderer and the separate final scorer. There is no scheduler framework, model loader, subprocess launcher, CLI or automatic next stage.

**Tech Stack:** Existing Python 3.12, standard library and unittest; in-memory synthetic frozen views and artifact-writing synthetic callbacks.

## Global Constraints

- Implements `../specs/2026-11-17-lora-development-coordinator-design.md`.
- D01–D04 × base/text only: eight episodes, two rounds each, at most sixteen generation callback invocations.
- Order: D01 base/text, D02 text/base, D03 base/text, D04 text/base. Complete both rounds and final scoring of one episode before entering the next.
- No adapters, training, transfer/retention execution, private-data reads during implementation, real model/candidate execution, remote access, installs, telemetry, commits or pushes.
- Only create `experiments/lora_necessity/development_coordinator.py` and `experiments/lora_necessity/test_development_coordinator.py` when Main authorizes implementation. Use `apply_patch`, no inline code comments.
- Do not change any sealed protocol, policy, observer, renderer, execution/scoring component, schedule contract, stage ledger or protected file.
- Synthetic callbacks write fixed evidence; never interpret or execute request source. Do not run worker tests, broad discovery or another agent's suite.
- Initial invalid source still receives the second generation. Public success does not skip round two. No retry, truncation, prompt substitution, best-of-two, adaptive order or extra feedback.
- Final expected values, reasons, scores, admission diagnostics, raw stdout and failure messages never enter model feedback. Only `visible_feedback` goes into `second_round`.
- Ordinary mismatch, supported ValueError outcome and executor `program_error`/`timeout`/`output_limit` are observations, not infrastructure exceptions.
- Unknown costs remain `None`; counts of callback entries, claims, forwards, completed generation records and child starts are separate.
- All `pilot_protocol.FLAGS` remain false. Synthetic callback success is not a model experiment, authenticated provenance, isolation qualification or launch authority.
- No new aggregate disk/memory quota, connection cleanup policy or general scheduling abstraction.

## Prerequisites and Review Gates

This plan was written against:

| Dependency | State at planning | SHA256 |
| --- | --- | --- |
| `development_schedule.py` | Independently accepted; Main rehashed source before dispatch | `f7785d194d3e9adfb986ab627fd5b5e968c9bb42f44e40e3a07ee33a04bc8bb1` |
| `development_final_protocol.py` | Accepted | `9315bfec3ed19604c21d8ec6a944d264828f8e92618416748d377d77f36ad141` |
| `development_final_admission.py` | Accepted | `8af6e5c1534fbec4da1897119b5431c0f35e2b69fdecb76df235219f112c094e` |
| `development_final_controller.py` | Accepted | `4b70a321227425296cdbae47d15d7237a399627e18c8443da70955649f3df7cd` |
| `development_final_scoring.py` | Independently accepted per Main's planning checkpoint | `29244847a45d6b29228306632eee1a51eb19a1263f5ff86dafbe275d0bdf2aab` |
| `observed_generation.py` | Approved additive cleanup interface | `8fd18d126cce0b97cce5c90c2edeb4f2859959d3a4ad76864c2711b8571c0efd` |

Schedule and scoring are both independently accepted per Main's latest message.
The accepted schedule source identity is recorded above. Consume its exact documented return schema, not a
guessed adapter. Scoring source was read and hashed before acceptance. The schedule
delivery counts above are Main-reported, not tests run by this planning sidecar.
The execution acceptance is in
`/tmp/tlf-candidate-executor-20261117-gb1gZX/development-final-execution-review.md`;
the scoring implementation evidence is in `development-final-scoring-report.md`
in that same directory.

## Exact Public API

```text
run_development(supervision_raw, prompts_raw, source_raw, manifest_raw, *,
                run_id, artifact_dir, bindings,
                generate, execute_public, execute_final) -> dict
```

Four inputs are exact immutable `bytes`, not paths. `artifact_dir` names a **new**
directory whose parent already exists. Create it exclusively with mode 0700;
never adopt a preexisting directory, resume, clean it or overwrite its records.

`bindings` is native JSON with exactly:

```python
bindings = {
    "component_sha256": component_sha256,
    "runtime_sha256": runtime_sha256,
    "model_sha256": model_sha256,
    "tokenizer_state": {
        "template_sha256": template_sha256,
        "vocabulary_sha256": vocabulary_sha256,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
    },
    "execution": {
        "public": public_execution,
        "final": final_execution,
    },
}
```

`component_sha256` is an exact mapping for these basenames:
`development_coordinator.py`, `development_schedule.py`,
`development_final_scoring.py`, `development_final_protocol.py`,
`development_final_admission.py`, `development_final_controller.py`,
`candidate_execution_admission.py`, `candidate_execution_controller.py`,
`candidate_execution_protocol.py`, `candidate_feedback.py`,
`prompt_rendering.py`, `observed_generation.py`, `generation_core.py`,
`tokenizer_measurement.py`, `validation_contract.py`, `pilot_protocol.py`.
Every digest is an exact lowercase 64-hex string. Token IDs are exact nonnegative
integers. An execution binding has exactly `argv`, `startup_timeout`,
`candidate_timeout`, `output_limit`, `result_limit`, `reap_timeout`, with the
accepted controller's type/range checks and exactly one `{control_fd}` token
across a nonempty list of strings. No shell evaluation or environment handling.

These are **caller-declared bindings**, committed before callbacks, not authenticated
identities. The coordinator verifies the actual four input hashes through the
schedule's frozen loaders and compares emitted records with these declarations.
It cannot establish that the declaration came from a genuine model, observer,
launcher, remote host or independently verified source file. That remains the
future fixed remote driver's and independent receipt reviewer's responsibility.
Do not add `verified_runtime=true` or infer isolation from hashes.

### Callback signatures

```python
def generate(messages, *, expected_tokenizer_sha256, call_id, artifact_dir):
    return {"generation": generation, "observation": observation,
            "record_cleanup": record_cleanup}


def execute_public(request, *, artifact_dir):
    return {"admission": admission, "audit": audit, "receipt": receipt}


def execute_final(request, *, artifact_dir):
    return {"admission": admission, "audit": audit, "receipt": receipt}
```

The driver binds model/tokenizer to `generate_observed`; public/final closures
bind the correct accepted `run_candidate_request`, argv, cwd, environment and
limits. The coordinator passes only a fresh messages/request snapshot and these
named keyword arguments, **not** the schedule, frozen source, expected outcomes,
bindings dictionary, earlier episode results or scorer plan.

Callbacks must preserve the existing canonical observer records or existing
controller records in the supplied directory, including execution raw streams.
They do not return artifact paths to follow. Read only fixed names under the
coordinator-owned directory. Returned callback objects must agree with the saved
records. The closed-over callback's genuine identity is outside this component.

Successful `run_development` returns exactly:

```python
{
    "report": report,
    "terminal_path": "development-report.json",
    "terminal_sha256": digest(report),
    "record_cleanup": coordinator_record_cleanup,
    "record_cleanup_complete": record_cleanup_complete,
    "persistence_errors": [],
}
```

On any preterminal failure raise `DevelopmentRunError` from the **first** exception;
attributes are `report`, `terminal_path` (relative path or `None`),
`terminal_sha256` (digest or `None`), `record_cleanup`,
`record_cleanup_complete`, `persistence_errors`.
Its `report` is available even if every attempted failure-record write fails.
`BoundaryProtocolError` is the named exception for confirmed identity/protocol/
isolation failures, including a driver-detected violation. Untyped callback
exceptions are infrastructure/incomplete, not automatically a model error or
a confirmed isolation breach. Locally detected admission denials are protocol
failures; admission persistence failures remain infrastructure failures.

## Schemas and Accounting

New root claim, prepared, episode and terminal records have `kind`,
`version=1`, `run_id`. Reused schedule/feedback records and embedded association
rows retain their explicit schemas below rather than acquiring extra fields. Store exact native
JSON, independent of mutable callback objects. No data is silently discarded
because a higher-priority calibration reason exists.

### Root artifacts

```text
claim.json
schedule.json
episodes/00-D01-base/episode.json
episodes/00-D01-base/generation-1/prepared.json
episodes/00-D01-base/generation-1/claim.json
episodes/00-D01-base/generation-1/generation.json
episodes/00-D01-base/generation-1/observation.json
episodes/00-D01-base/generation-1/association.json
episodes/00-D01-base/public-00-P01/prepared.json
episodes/00-D01-base/public-00-P01/admission.json
episodes/00-D01-base/public-00-P01/claim.json
episodes/00-D01-base/public-00-P01/observation.json
episodes/00-D01-base/public-00-P01/receipt.json
episodes/00-D01-base/public-00-P01/stdout.bin
episodes/00-D01-base/public-00-P01/stderr.bin
episodes/00-D01-base/public-00-P01/control.bin
episodes/00-D01-base/public-00-P01/association.json
episodes/00-D01-base/feedback.json
episodes/00-D01-base/generation-2/
episodes/00-D01-base/final-binding.json
episodes/00-D01-base/final-000-public-P01/
episodes/00-D01-base/result.json
failure.json
partial-report.json
development-report.json
```

The public path uses the original `visible_requests` order: the `00` entry may
be P02 in synthetic fixtures. Do **not** reorder it to match the illustrative path.
Final paths use the scorer's public-first/sorted order, including group and ordinal.
Only one terminal report is published. No `failure.json`/`partial-report.json`
on a successful run. On abort, untouched future episode directories are absent
but their eight fixed positions remain in the report.

Root claim fields, in addition to the common header:
`schedule_sha256`, `data_sha256`, `bindings`, `bindings_authenticity="CALLER_DECLARED"`,
`scheduled_episodes=8`, `scheduled_generation_calls=16`, `input_token_limit=4096`,
`output_token_limit=256`, `episode_order` (eight task/arm pairs),
`calibration_precedence` (the four labels in order).
`schedule.json` is exactly the accepted schedule result. Its episode rows are:
`ordinal`, `task_id`, `arm`, `messages`, `examples_sha256`, `task_sha256`,
`messages_sha256`, `first_call_id`, `second_call_id`, `public_attempt_id`,
`scheduled_final_cases`, `final_cases`.
Each inventory item has only `case_group`, `case_id`, `input_sha256`,
`expected_sha256`. Do not add private expected values to that inventory.

Prepared records commit `phase`, `episode_ordinal`, `round`, `call_id` for
generation or `request_sha256`/`attempt_id`/case identity for execution,
messages digest or full execution request, and the relative directory.
Prepared publication and directory existence are **intent**, not proof the
callback entered or that a model/child started.

Every association row contains:

```python
{
    "phase": phase,
    "relative_dir": relative_dir,
    "prepared": False,
    "callback_entered": False,
    "callback_returned": False,
    "claim_verified": False,
    "observation_verified": False,
    "observation_status": None,
    "generation_verified": False,
    "receipt_verified": False,
    "process_started": None,
    "forward_started": None,
    "forward_completed": None,
    "input_tokens": None,
    "output_tokens": None,
    "generation_timings_ns": None,
    "observer_elapsed_ns": None,
    "execution_elapsed_seconds": None,
    "peak_allocated_bytes": None,
    "peak_reserved_bytes": None,
    "record_cleanup": None,
    "receipt": None,
    "files": {},
    "failure": None,
    "evidence_errors": [],
}
```

Extend generation rows with `round`, `call_id`, `messages_sha256`;
execution rows with `round`, `case_group`, `case_id`, `request_sha256`.
`files` maps fixed filenames to `{sha256, bytes}` for bytes actually read,
including malformed JSON. Original bytes remain untouched on disk.
Absent/unreadable records do not get fabricated hashes. Failed observation
forward counters/elapsed can be retained after their identities/types validate;
without a validated generation record, token counts remain `None`.

Episode rows contain `ordinal`, `task_id`, `arm`, `scheduled_final_cases`,
`first_round_valid`/`final_round_valid` (bool or `None`), `feedback` (safe
projection or `None`), `feedback_opportunities` (0 or 1), `generations`,
`public_executions`, `final_executions`, `final_score` (scorer result or `None`),
`completed` (bool). First-invalid recovery must preserve both
`first_round_valid=False` and `final_score.status="pass"`.

Terminal report fields:

```python
{
    "kind": "LORA_DEVELOPMENT_REPORT",
    "version": 1,
    "run_id": run_id,
    "status": status,
    "schedule_sha256": schedule_sha256,
    "bindings_authenticity": "CALLER_DECLARED",
    "scheduled_episodes": 8,
    "scheduled_generation_calls": 16,
    "episodes": episodes,
    "counts": counts,
    "by_arm": by_arm,
    "costs": costs,
    "calibration": {"status": decision, "reasons": reasons},
    "failure": failure,
    "persistence_errors": persistence_errors,
    "record_cleanup_before_terminal": cleanup_snapshot,
    "record_cleanup_complete_before_terminal": record_cleanup_complete,
    "boundary_authenticity": "NOT_ATTESTED",
    **{flag: False for flag in FLAGS},
}
```

`status` is `completed` or `aborted`, distinct from calibration and task pass/fail.
`by_arm[arm]` has `scheduled=4`, `completed`, `passed`, and four ordered
`{task_id, final_status}` rows; missing is not fail.

`counts` has `completed_episodes`, `generation_callbacks_entered`,
`generation_callbacks_returned`, `verified_generation_claims`,
`verified_generation_records`, `completed_generation_calls`,
`public_feedback_opportunities`, `public_callbacks_entered`,
`final_callbacks_entered`, `public_verified_child_starts`,
`final_verified_child_starts`, `public_unknown_child_starts`,
`final_unknown_child_starts`, `public_verified_receipts`,
`final_verified_receipts`, `scheduled_final_cases`, `received_final_cases`,
`passed_final_cases`, `missing_final_cases`, `invalid_final_cases`,
`unexecuted_final_cases`, `missing_episode_positions`.
`missing_episode_positions` is the ordered list of ordinals whose episode result
was not committed. `unexecuted_final_cases` counts cases for which no final
execution callback was entered: scheduled cases minus final callback entries.
It is a known-not-submitted count, not scheduled minus received receipts.
Entered callbacks without positive child-start evidence remain unknown in
`final_unknown_child_starts`; a missing receipt never proves non-execution.
Completed generation calls require a verified completed observation and
generation, not merely returned callbacks. Unknown child-start counts include
entered execution callbacks without positive launch evidence. A generation
claim does not prove one model call; an execution claim does not prove a child.

Before a valid schedule exists, case denominators are `None`, not zero; all
eight episode positions and sixteen planned calls are nevertheless retained.
With a schedule, final invalid source means all scheduled rows are invalid,
zero received, all unexecuted, and no missing *status* rows. If final planning
never occurred, inventory rows are missing. Prefix execution retains every
remaining missing row by calling `score_final` on the verified receipt prefix.

`costs` retains per-association records and:
`input_tokens`, `output_tokens`, `forward_started`, `forward_completed`,
`generation_wall_ns`, `observer_elapsed_ns`, `execution_elapsed_seconds`,
`coordinator_wall_ns`, `driver_model_load_ns=None`,
`driver_shared_preparation_ns=None`, `driver_outer_wall_ns=None`.
Each of the first six generation metrics is
`{total, observed_subtotal, known_calls, scheduled_calls:16}`. `total=None`
unless all sixteen values are known. Execution elapsed uses the corresponding
executed-attempt rows and also reports unknown entries, not a guessed full
scheduled-work cost. Never add nested generator wall, observer elapsed,
execution audit elapsed and coordinator wall as if they were disjoint.

## Task 1: Exclusive Records and Fixed Ledger

**Files:** Create the coordinator and its test file only.

**Consumes:** `observed_generation._save`, `_failure`;
`candidate_execution_admission._plain_json`; `validation_contract.encoded`,
`digest`, `parse_json`; `development_schedule.build_schedule`.

**Produces:** `BoundaryProtocolError`, `DevelopmentRunError`, `_clone`,
`_publish`, `_new_state`, `_claim`, `_attempt`, and the public API's failure envelope.

- [ ] Write missing-component RED tests with imports inside `setUp`, so discovery
  does not fail before assertions execute:

```python
class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("development_coordinator.py").is_file(),
                        "coordinator implementation missing")
        self.module = importlib.import_module("development_coordinator")
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name) / "run"
```

- [ ] Run and retain RED, without running any other suite:

```bash
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error -m unittest discover -s experiments/lora_necessity -p test_development_coordinator.py -v 2>&1 | tee /tmp/tlf-candidate-executor-20261117-gb1gZX/development-coordinator-red.log
```

- [ ] Implement the persistence primitives below. `_publish` does not catch
  precommit errors. The observer's hard link is the commit point; failed
  postcommit unlink becomes a cleanup entry, never an exception/retry.

```python
import copy
import hashlib
import math
from pathlib import Path
import re
import time

from candidate_execution_admission import _plain_json
from observed_generation import _failure, _save, _validate
from pilot_protocol import FLAGS
from validation_contract import digest, encoded, parse_json


class BoundaryProtocolError(ValueError):
    pass


class DevelopmentRunError(RuntimeError):
    def __init__(self, envelope):
        super().__init__("development run aborted; preserved evidence is authoritative")
        self.report = envelope["report"]
        self.terminal_path = envelope["terminal_path"]
        self.terminal_sha256 = envelope["terminal_sha256"]
        self.record_cleanup = envelope["record_cleanup"]
        self.record_cleanup_complete = envelope["record_cleanup_complete"]
        self.persistence_errors = envelope["persistence_errors"]


def _need(condition, message):
    if not condition:
        raise BoundaryProtocolError(message)


def _clone(value):
    _plain_json(value)
    raw = encoded(value)
    return parse_json(raw, limit=len(raw))


def _checked(function, *arguments, **keywords):
    try:
        return function(*arguments, **keywords)
    except BoundaryProtocolError:
        raise
    except (ValueError, KeyError, TypeError, RecursionError) as error:
        raise BoundaryProtocolError(str(error)) from error


def _publish(root, name, value, state):
    _save(root, name, _clone(value), cleanup=state["record_cleanup"])


def _save_error(state, path, error):
    state["persistence_errors"].append({"path": path, "error": _failure(error)})
```

Use the exact fixed-order constructor, not a caller-provided iterable:

```python
ORDER = tuple(
    (f"D0{ordinal}", arm)
    for ordinal in range(1, 5)
    for arm in (("base", "text") if ordinal % 2 else ("text", "base"))
)


def _new_state(run_id):
    return {
        "run_id": run_id, "schedule": None, "active": None,
        "record_cleanup": {"failed_count": 0, "failures": []},
        "persistence_errors": [], "protocol_errors": [], "failure": None,
        "episodes": [
            {"ordinal": ordinal, "task_id": task_id, "arm": arm,
             "scheduled_final_cases": None, "first_round_valid": None,
             "final_round_valid": None, "feedback": None,
             "feedback_opportunities": 0, "generations": [],
             "public_executions": [], "final_executions": [],
             "final_score": None, "completed": False}
            for ordinal, (task_id, arm) in enumerate(ORDER)
        ],
    }
```

`_claim(root, state, raw, bindings)` must call
`_checked(build_schedule, *raw, run_id=state["run_id"])`, validate its exact eight
ordered identities and 8/16 caps, and check each messages hash before publishing
`claim.json`, then `schedule.json`. Copy scheduled denominators to all episode
rows before the first callback. Do not regenerate IDs in the coordinator:
use `first_call_id`, `second_call_id`, `public_attempt_id`. Use the accepted final
scorer for its different, frozen per-case attempt algorithm.

`raw` is the four-byte-snapshot tuple in public API argument order. Validate
`type(run_id) is str` and lowercase 32-hex before forming paths. Reject non-bytes
raw inputs, non-callable callbacks and nonnative/incorrectly shaped bindings
before any callback. Invalid run identities appear as `None` in an in-memory
preflight error report rather than invoking arbitrary object serialization.
Retain independent messages/request snapshots: calculate expected hashes from
the coordinator-owned snapshot and pass a separate `_clone` into the callback.
Compare both the supplied copy and preserved records against that original
after return; mutation must not redefine the expected hash.

- [ ] Implement this shared **callback transaction**, not a scheduling engine:

```python
def _attempt(root, relative, prepared, row, state, invoke, inspect):
    directory = root / relative
    state["active"] = relative
    directory.mkdir(mode=0o700)
    _publish(directory, "prepared.json", prepared, state)
    row["prepared"] = True
    returned = None
    primary = None
    row["callback_entered"] = True
    try:
        returned = invoke(directory)
        row["callback_returned"] = True
    except BaseException as error:
        primary = error
    verified = None
    try:
        verified = inspect(directory, returned, primary, row)
    except BaseException as error:
        row["evidence_errors"].append(_failure(error))
        if isinstance(error, BoundaryProtocolError):
            state["protocol_errors"].append(_failure(error))
        if primary is None:
            primary = error
    if primary is not None:
        row["failure"] = _failure(primary)
    try:
        _publish(directory, "association.json", row, state)
    except BaseException as error:
        _save_error(state, relative + "/association.json", error)
        if primary is None:
            primary = error
    if primary is not None:
        raise primary
    return verified
```

Append the initialized row to its episode **before** calling `_attempt`.
A directory/prepared-write failure therefore leaves a visible missing attempt
without falsely setting `callback_entered`. `inspect` always uses captured
saved bytes, including after a callback exception; it can retain independently
verified costs without making that callback successful.

- [ ] Add concrete publication regression tests:

```python
def test_record_publish_unlink_failure_is_postcommit_diagnostic(self):
    state = self.module._new_state("a" * 32)
    self.root.mkdir(mode=0o700)
    with mock.patch("observed_generation.os.unlink", side_effect=OSError("cleanup")):
        self.module._publish(self.root, "terminal.json", {"status": "completed"}, state)
    self.assertEqual(parse_json((self.root / "terminal.json").read_bytes()),
                     {"status": "completed"})
    self.assertEqual(state["record_cleanup"]["failed_count"], 1)
    self.assertTrue(state["record_cleanup"]["failures"][0]["published"])


def test_record_publish_preserves_link_failure_when_unlink_also_fails(self):
    state = self.module._new_state("a" * 32)
    self.root.mkdir(mode=0o700)
    failure = OSError("primary link failure")
    with mock.patch("observed_generation.os.link", side_effect=failure), \
            mock.patch("observed_generation.os.unlink", side_effect=OSError("cleanup")):
        with self.assertRaises(OSError) as raised:
            self.module._publish(self.root, "terminal.json", {}, state)
    self.assertIs(raised.exception, failure)
    self.assertFalse((self.root / "terminal.json").exists())
    self.assertFalse(state["record_cleanup"]["failures"][0]["published"])
```

Also inject `_write` failure with unlink failure; require the original write
exception object. Precreate the run directory and assert no callback or existing
file mutation. Patch first `prepared.json` publication to fail and assert zero
callback entries/claims/forwards. Do not run a synthetic callback to stand in for
a persistence failure.

- [ ] Run the same focused command, with log
  `/tmp/tlf-candidate-executor-20261117-gb1gZX/development-coordinator-records-green.log`;
  expected: the task's implemented tests pass. Preserve actual counts, not an
  estimated number. Do not claim later tasks pass before they exist.

## Task 2: Generation and Execution Evidence Boundaries

**Files:** Modify only the new coordinator and test file.

**Consumes:** Task 1 transaction; approved observer `_validate`;
`candidate_execution_protocol.READY`, `make_receipt`, original `prepare_request`;
final `prepare_request`; original/final `admit_request`.

**Produces:**
`_generation(root, episode_dir, rows, messages, round_number, job, bindings, generate, state)`
returning exact `generation.json` bytes;
`_execution(root, episode_dir, rows, request, phase, bindings, execute, state)`
returning the verified receipt. No model or process invocation except the supplied
callbacks.
Both `rows` parameters are the episode's mutable association list. Each helper
constructs and appends its own initialized row before entering `_attempt`.

### Fixed-file snapshots

- [ ] Implement reads once per transaction. For generation, inspect only
  `claim.json`, `observation.json`, `generation.json`; execution additionally has
  `admission.json`, `receipt.json`, `stdout.bin`, `stderr.bin`, `control.bin` and
  no generation file. Record all readable file hashes before validating any one
  object. Missing files are allowed as failure evidence, not successful results.
  OSError is evidence failure, not absence. Preserve malformed bytes on disk.

```python
def _capture(directory, names, row):
    blobs = {}
    first_error = None
    for name in names:
        try:
            raw = (directory / name).read_bytes()
        except FileNotFoundError:
            continue
        except OSError as error:
            row["evidence_errors"].append({"file": name, "error": _failure(error)})
            if first_error is None:
                first_error = error
            continue
        blobs[name] = raw
        row["files"][name] = {
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return blobs, first_error


def _json(blobs, name):
    _need(name in blobs, "missing successful record: " + name)
    return _checked(parse_json, blobs[name], limit=len(blobs[name]))


def _same(actual, expected, message):
    _need(encoded(actual) == encoded(expected), message)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _duration(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0
```

### Generation verifier

- [ ] Use saved bytes as authoritative, not mutable returned objects. The
  expected claim is exactly:

```python
expected_claim = {
    "kind": "LORA_GENERATION_CLAIM",
    "call_id": call_id,
    "messages_sha256": digest(messages),
    "expected_tokenizer_sha256": digest(bindings["tokenizer_state"]),
}
```

Validate observation's exact eight fields:
`kind`, `call_id`, `status`, `messages_sha256`, `forward_started`,
`forward_completed`, `elapsed_ns`, `failure`.
Require `kind="LORA_GENERATION_OBSERVATION"`, matching IDs/messages,
nonnegative exact integer counters and elapsed; completed cannot exceed started.
`status` is `completed` with `failure=None`, or `failed` with a diagnostic object.
A failed observation is useful cost evidence, not a completed call.

Validate generation using this concrete function; it never measures a real tokenizer:

```python
def _generation_values(generation, observation, messages, bindings):
    state = bindings["tokenizer_state"]
    _checked(_validate, generation, messages, digest(messages), digest(state),
             observation["forward_started"], observation["forward_completed"])
    names = {
        "kind", "device_type", "tokenizer_state_sha256", "messages_sha256",
        "prompt_sha256", "input_tokens", "output_tokens", "output_ids",
        "candidate", "candidate_sha256", "stop_reason", "model_calls",
        "forward_calls", "timings_ns", "peak_allocated_bytes", "peak_reserved_bytes",
        *FLAGS,
    }
    _need(set(generation) == names, "generation fields")
    _need(_integer(generation["input_tokens"], 1)
          and generation["input_tokens"] <= 4096, "input token budget")
    _need(type(generation["prompt_sha256"]) is str
          and re.fullmatch("[0-9a-f]{64}", generation["prompt_sha256"]),
          "prompt identity format")
    output_ids = generation["output_ids"]
    eos = state["eos_token_id"]
    if generation["stop_reason"] == "eos":
        _need(output_ids[-1] == eos and eos not in output_ids[:-1], "EOS boundary")
    else:
        _need(generation["stop_reason"] == "length"
              and len(output_ids) == 256 and eos not in output_ids, "length boundary")
    timings = generation["timings_ns"]
    _need(type(timings) is dict
          and set(timings) == {"preparation", "generation", "decoding", "wall"}
          and all(_integer(value) for value in timings.values()), "generation timings")
    _need(sum(timings[name] for name in ("preparation", "generation", "decoding"))
          <= timings["wall"] <= observation["elapsed_ns"], "nested generation times")
    _need(all(generation[flag] is False for flag in FLAGS), "generation flags")
    allocated = generation["peak_allocated_bytes"]
    reserved = generation["peak_reserved_bytes"]
    if generation["device_type"] == "cpu":
        _need(allocated is None and reserved is None, "unknown CPU memory")
    else:
        _need(generation["device_type"] == "cuda"
              and _integer(allocated) and _integer(reserved)
              and allocated <= reserved, "CUDA peak fields")
```

This validates declared prompt hash shape, count bounds and EOS consistency,
**not** re-tokenization, rendered prompt authenticity or vocabulary membership.
Those need the future fixed driver's actual tokenizer/state evidence. Do not
claim that `_validate` performs checks absent from its source.

On return, require exact callback envelope fields; canonical
`encoded(returned["generation"])` must equal saved generation bytes and
`encoded(returned["observation"])` must equal saved observation bytes.
Validate `record_cleanup` as exact `failed_count` and list `failures`, with count
equal to list length; each failure has `record`, `temporary`, `published`, `error`
and error has `type`, `message` strings. Preserve nonzero diagnostics.
On callback failure inspect `error.record_cleanup` if present; absence is
unknown, not a fabricated clean result.

For partial inspection, update row fields as each independent check succeeds:
claim first, matching observation next, then generation. If generation bytes
validate with a failed observation's counters, retain token/timing evidence
but do not count a completed generation. If the callback raises despite intact
completed saved records, preserve their verified cost/completion evidence and
still abort on the callback exception. Never call it again.

`_generation` creates `generation-1` or `generation-2`, takes the corresponding
schedule ID, records the supplied messages digest, calls `_attempt`, and returns
the immutable saved generation bytes only after successful verification.
Before setting `callback_entered`, require that the total already-entered
generation rows is below 16. A successful run must verify exactly sixteen.

### Execution verifier

- [ ] Compute expected admission locally with the correct pure `admit_request`
  before the callback. This detects a real policy denial separately from an
  OSError raised while the callback persists admission. Do not execute source.
  Catch `CandidateAdmissionError` from this **local pure call** and raise
  `BoundaryProtocolError` from it; do not classify every callback exception of
  that class as a safety denial.

Successful expected controller claim is exactly:

```python
expected_claim = {
    "request_sha256": request_sha256,
    "argv": execution["argv"],
    "payload_sha256": hashlib.sha256(payload).hexdigest(),
    "startup_timeout": execution["startup_timeout"],
    "candidate_timeout": execution["candidate_timeout"],
    "output_limit": execution["output_limit"],
    "result_limit": execution["result_limit"],
}
```

`reap_timeout` stays bound in the root claim/prepared association, but the sealed
controller claim does not contain it. Do not invent such a field or claim it
was independently observed there.

Require saved admission equal locally computed admission; saved claim equal
the above; callback admission/audit/receipt equal the corresponding saved
JSON objects under canonical encoding. Unlike observer records, controller
JSON is pretty-printed: do not require raw bytes equal canonical `encoded`.
Keep their actual raw hashes.

Reconstruct the receipt from the saved raw control stream and audit:

```python
def _receipt_from_evidence(request_sha256, payload, execution, audit, blobs):
    _need(audit["request_sha256"] == request_sha256, "audit request binding")
    _need(audit["failure"] is None, "failed execution audit")
    for name in ("ready", "input_complete", "reaped", "eof"):
        _need(type(audit[name]) is bool, "audit lifecycle type")
    _need(_integer(audit["input_bytes_sent"])
          and audit["input_bytes_sent"] == len(payload), "input byte count")
    _need(_duration(audit["elapsed_seconds"])
          and _duration(audit["send_started_seconds"])
          and audit["send_started_seconds"] <= audit["elapsed_seconds"], "execution times")
    control = blobs["control.bin"]
    _need(control.startswith(READY), "control READY")
    observed = sum(len(blobs[name + ".bin"]) for name in ("stdout", "stderr", "control"))
    _need(_integer(audit["output_bytes_observed"])
          and audit["output_bytes_observed"] == observed
          and observed <= execution["output_limit"] + 1, "stream byte accounting")
    _same(audit["stream_sha256"], {
        name: hashlib.sha256(blobs[name + ".bin"]).hexdigest()
        for name in ("stdout", "stderr", "control")
    }, "stream hashes")
    if len(control) - len(READY) > execution["result_limit"]:
        _need(audit["termination"] == "output_limit", "result-limit termination")
    if observed > execution["output_limit"]:
        _need(audit["termination"] == "output_limit", "output-limit termination")
    return _checked(
        make_receipt, request_sha256, returncode=audit["returncode"],
        ready=audit["ready"], input_complete=audit["input_complete"],
        reaped=audit["reaped"], eof=audit["eof"], termination=audit["termination"],
        result_raw=control[len(READY):])
```

Require exact audit fields from the accepted controller source; reject extra or
missing fields and bool-as-integer substitutions. Inspect `returncode` as exact
int or `None`; `termination` as `None`, `timeout` or `output_limit`; `failure`
as `None` or string. Successful reconstruction requires the sealed lifecycle.
Nonzero exit, duplicate/missing frame, timeout and output-limit are reduced by
`make_receipt`, not reclassified by a new policy.

On failed callbacks retain readable audit elapsed and positive child-start
evidence only after request/claim/stream identities validate. `ready=True` or an
integer return code positively shows a child; absent/failed-reap evidence leaves
start unknown. Do not set `process_started=False` merely from missing receipt.
Do not reconstruct a success from a failure audit. Keep all original files.

- [ ] Write synthetic generation/receipt mutations before these validators:
  swap call IDs/messages/tokenizer/candidate hash/observation fields; input4097;
  EOS in the middle; length<256; false forward count; missing generation file;
  returned-vs-saved disagreement; nonfinite/negative elapsed; flags true;
  admission kind/hash mismatch; wrong argv/limits; raw-control mutation; wrong
  stream hash; missing successful records or incomplete successful lifecycle.
  Each confirmed identity/contract mutation must abort without another callback
  and retain the prior readable evidence. Coherent complete lifecycle records
  whose result stream contains a missing or duplicate frame reduce through
  `make_receipt` to `program_error`, not a new protocol exception. An unreaped or
  incomplete lifecycle cannot support a successful callback. Wrap
  `InfrastructureError` raised by `make_receipt` during successful-callback
  validation as `BoundaryProtocolError`; do not apply that conversion to an
  exception raised by the callback itself.

Use the synthetic callback implementations in Task 4; add mutations with this
specific adapter, not an actual observer/model invocation:

```python
def altered(callback, mutate):
    def invoke(*arguments, **keywords):
        result = callback(*arguments, **keywords)
        mutate(result, keywords["artifact_dir"])
        return result
    return invoke
```

For example:

```python
def test_returned_generation_cannot_disagree_with_saved_bytes(self):
    generate = altered(self.generate, lambda result, directory:
                       result["generation"].update(candidate="not the saved candidate"))
    with self.assertRaises(self.module.DevelopmentRunError) as raised:
        self.run_development(generate=generate)
    self.assertEqual(len(self.calls), 1)
    self.assertEqual(raised.exception.report["calibration"]["status"], "INVALID_PROTOCOL")
```

- [ ] Run the focused command to
  `development-coordinator-boundaries-green.log`. Report actual test counts.

## Task 3: Two-Round Flow, Partial Prefixes and Calibration

**Files:** Modify only the new coordinator and test file.

**Consumes:** Task 2 verified bytes/receipts; unchanged
`candidate_valid`, `visible_requests`, `visible_feedback`, `second_round`;
accepted `build_final_plan` and `score_final`.

**Produces:** Completed `run_development`, `_run_episode`, `_partial_score`,
`_report`, `_finish`, `_calibration`; exact schemas above.

- [ ] Write flow tests before implementation: fixed alternating order; exactly
  sixteen successful generation callbacks even with all public tests passing;
  first-invalid recovery with zero public callbacks for that episode; final
  invalid source with zero final callbacks and complete denominator.

- [ ] Implement the per-episode core with these exact helper calls. `_execution`
  takes only the individual request, never `plan["expected_outcomes"]`.

```python
def _run_episode(root, job, episode, state, raw, bindings, callbacks):
    supervision_raw, prompts_raw, source_raw, manifest_raw = raw
    generate, execute_public, execute_final = callbacks
    episode_dir = f'episodes/{job["ordinal"]:02d}-{job["task_id"]}-{job["arm"]}'
    directory = root / episode_dir
    directory.mkdir(mode=0o700)
    _publish(directory, "episode.json", {
        "kind": "LORA_DEVELOPMENT_EPISODE", "version": 1, "run_id": state["run_id"],
        "ordinal": job["ordinal"], "task_id": job["task_id"], "arm": job["arm"],
        "messages_sha256": job["messages_sha256"],
        "first_call_id": job["first_call_id"], "second_call_id": job["second_call_id"],
        "public_attempt_id": job["public_attempt_id"],
        "scheduled_final_cases": job["scheduled_final_cases"],
    }, state)
    first_raw = _generation(root, episode_dir, episode["generations"], job["messages"],
                            1, job, bindings, generate, state)
    first = parse_json(first_raw)
    valid = _checked(candidate_valid, prompts_raw, job["task_id"], job["arm"],
                     first["candidate"])
    episode["first_round_valid"] = valid
    requests = _checked(visible_requests, prompts_raw, job["task_id"], job["arm"],
                        first["candidate"], attempt_id=job["public_attempt_id"])
    _need(len(requests) == (2 if valid else 0), "public request coverage")
    receipts = []
    for request in requests:
        receipts.append(_execution(root, episode_dir, episode["public_executions"],
                                   request, "public", bindings, execute_public, state))
    feedback = _checked(
        visible_feedback, prompts_raw, job["task_id"], job["arm"], first["candidate"],
        encoded(receipts), attempt_id=job["public_attempt_id"])
    episode["feedback"] = feedback
    _publish(directory, "feedback.json", feedback, state)
    messages = _checked(second_round, supervision_raw, prompts_raw, job["task_id"],
                        job["arm"], first["candidate"], feedback)
    episode["feedback_opportunities"] = 1
    second_raw = _generation(root, episode_dir, episode["generations"], messages,
                             2, job, bindings, generate, state)
    identity = {
        "run_id": state["run_id"], "task_id": job["task_id"], "arm": job["arm"],
        "generation_call_id": job["second_call_id"], "generation_raw": second_raw,
        "expected_generation_sha256": hashlib.sha256(second_raw).hexdigest(),
    }
    state["final_identity"][job["ordinal"]] = identity
    plan = _checked(build_final_plan, source_raw, manifest_raw, **identity)
    _match_inventory(job, plan)
    episode["final_round_valid"] = plan["candidate_valid"]
    _publish(directory, "final-binding.json", {
        "kind": "LORA_DEVELOPMENT_FINAL_BINDING", "version": 1,
        "run_id": state["run_id"], "generation_call_id": job["second_call_id"],
        "generation_sha256": identity["expected_generation_sha256"],
        "plan_sha256": digest(plan), "case_set_sha256": plan["case_set_sha256"],
        "scheduled_cases": plan["scheduled_cases"], "inventory": job["final_cases"],
        "request_sha256": [digest(request) for request in plan["requests"]],
    }, state)
    for request in plan["requests"]:
        _execution(root, episode_dir, episode["final_executions"], request,
                   "final", bindings, execute_final, state)
    _partial_score(episode, state, source_raw, manifest_raw)
    completed = {**episode, "completed": True}
    _publish(directory, "result.json", completed, state)
    episode["completed"] = True
```

Initialize `state["final_identity"]={}` as a **transient**, nonserialized map.
It may contain saved generation bytes; never pass it to callbacks or include it
in `_report`. Keep output schemas strictly JSON.

`_match_inventory` compares `task_sha256`, scheduled count, group/id order and
expected hashes; for valid plans also compare request input hashes:

```python
def _match_inventory(job, plan):
    _need(plan["task_sha256"] == job["task_sha256"]
          and plan["scheduled_cases"] == job["scheduled_final_cases"], "final task binding")
    expected = [
        {"case_group": case["case_group"], "case_id": case["case_id"],
         "expected_sha256": digest(case["expected"])}
        for case in plan["expected_outcomes"]
    ]
    _same(expected, [
        {name: case[name] for name in ("case_group", "case_id", "expected_sha256")}
        for case in job["final_cases"]
    ], "final expected inventory")
    if plan["candidate_valid"]:
        _same([
            {"case_group": request["case_group"], "case_id": request["case_id"],
             "input_sha256": digest(request["args"])}
            for request in plan["requests"]
        ], [
            {name: case[name] for name in ("case_group", "case_id", "input_sha256")}
            for case in job["final_cases"]
        ], "final input inventory")


def _partial_score(episode, state, source_raw, manifest_raw):
    identity = state["final_identity"].get(episode["ordinal"])
    if identity is None:
        return
    receipts = []
    for row in episode["final_executions"]:
        if not row["receipt_verified"]:
            break
        receipts.append(row["receipt"])
    episode["final_score"] = _checked(
        score_final, source_raw, manifest_raw, encoded(receipts), **identity)
```

On any failure, call `_partial_score` for the active episode if a verified
second generation identity exists. This is pure re-reduction, not retry.
Never skip a missing middle receipt and count later rows as a valid prefix.
If planning/scoring validation fails, retain it as additional protocol evidence
without overwriting the first callback/persistence exception.

### Cost and calibration reduction

- [ ] Implement exact `None` semantics:

```python
def _cost(rows, name, scheduled=16):
    values = [row[name] for row in rows if row[name] is not None]
    return {
        "total": sum(values) if len(values) == scheduled else None,
        "observed_subtotal": sum(values) if values else None,
        "known_calls": len(values), "scheduled_calls": scheduled,
    }
```

An observed subtotal of 0 is permitted only when at least one validated record
reports 0. No observed records means subtotal `None`. Do not infer an unstarted
model's token count from a directory, exception string or a zero forward count.

Use the following precedence, with **all** accumulated reasons retained:

```python
PRECEDENCE = (
    "INVALID_PROTOCOL",
    "INCONCLUSIVE_INCOMPLETE",
    "INCONCLUSIVE_CALIBRATION",
    "DEVELOPMENT_CHECKS_SATISFIED",
)


def _calibration(episodes, protocol_errors, aborted, necessary_costs_known):
    reasons = {label: [] for label in PRECEDENCE[:-1]}
    if protocol_errors:
        reasons["INVALID_PROTOCOL"].append("confirmed_protocol_identity_or_isolation_failure")
    incomplete = reasons["INCONCLUSIVE_INCOMPLETE"]
    if aborted:
        incomplete.append("run_aborted")
    if any(not episode["completed"] for episode in episodes):
        incomplete.append("missing_episode")
    generations = [row for episode in episodes for row in episode["generations"]]
    if len(generations) != 16 or any(
            not row["generation_verified"] or not row["observation_verified"]
            or row.get("observation_status") != "completed" for row in generations):
        incomplete.append("missing_or_failed_generation")
    if not necessary_costs_known:
        incomplete.append("unknown_necessary_cost")
    statuses = [
        row["receipt"]["status"]
        for episode in episodes
        for row in episode["public_executions"] + episode["final_executions"]
        if row["receipt_verified"]
    ]
    if "timeout" in statuses:
        incomplete.append("timeout")
    calibration = reasons["INCONCLUSIVE_CALIBRATION"]
    if any(episode["first_round_valid"] is False for episode in episodes):
        calibration.append("first_round_invalid_source")
    if any(episode["final_round_valid"] is False for episode in episodes):
        calibration.append("final_round_invalid_source")
    for status in ("program_error", "output_limit"):
        if status in statuses:
            calibration.append(status)
    if all(episode["final_score"] is not None
           and episode["final_score"]["status"] != "incomplete" for episode in episodes):
        passed = {
            arm: sum(episode["final_score"]["status"] == "pass"
                     for episode in episodes if episode["arm"] == arm)
            for arm in ("base", "text")
        }
        if passed["base"] == passed["text"] == 4:
            calibration.append("both_arms_all_pass")
        if passed["base"] == passed["text"] == 0:
            calibration.append("both_arms_all_fail")
    decision = next((label for label in PRECEDENCE[:-1] if reasons[label]), PRECEDENCE[-1])
    return {"status": decision, "reasons": reasons}
```

Necessary costs for a completed run mean sixteen known input/output token
counts, generation timing dictionaries and observer elapsed values, plus finite
elapsed values for every entered execution attempt that returned a verified
receipt. Memory and driver load/preparation/outer timings are explicitly
out-of-component and may remain `None` without forcing incomplete. Neither
driver authenticity being unassessed nor CPU peak memory being unavailable is
a confirmed protocol failure. All externally meaningful FLAGS remain false.

Add `observation_status` (string or `None`) to generation association rows; it
does not alter the observer's record schema.

### Root control and publication

- [ ] Implement the root flow in this order:
  native input/binding validation; exclusive 0700 root creation; schedule/claim
  publication; `episodes` directory creation; fixed loop; aggregate report;
  terminal publish. No exception handler around the loop may reenter it.
  Invalid input before root ownership still raises an in-memory error envelope
  with eight missing positions; never touch an existing root to save the error.

Use this terminal helper, adding the common report fields in `_report` exactly
as specified above. `state["failure"]` is the first error plus `active` location.
`_report(state, coordinator_wall_ns, *, aborted)` is a total, side-effect-free
reducer over the initialized ledger: it must work with no schedule, partial
associations and missing scores. It never reads files or calls a callback.
The coordinator wall measurement ends immediately before report assembly;
the future driver's outer wall includes publication and is reported separately.

Record cleanup is a separate check, not an excuse to reverse a committed
terminal or silently retry. Its scope is coordinator/observer record-file
maintenance, not driver/model/remote cleanup:

```python
def _record_cleanup_complete(state):
    if state["record_cleanup"]["failed_count"]:
        return False
    values = [
        row["record_cleanup"]
        for episode in state["episodes"] for row in episode["generations"]
        if row["callback_entered"]
    ]
    if any(value is not None and value["failed_count"] for value in values):
        return False
    if any(value is None for value in values):
        return None
    return True


def _finish(root, state, report, primary):
    terminal = "development-report.json" if primary is None else "partial-report.json"
    if primary is not None:
        try:
            _publish(root, "failure.json", state["failure"], state)
        except BaseException as error:
            _save_error(state, "failure.json", error)
    report["persistence_errors"] = _clone(state["persistence_errors"])
    report["record_cleanup_before_terminal"] = _clone(state["record_cleanup"])
    report["record_cleanup_complete_before_terminal"] = _record_cleanup_complete(state)
    try:
        _publish(root, terminal, report, state)
    except BaseException as error:
        _save_error(state, terminal, error)
        if primary is None:
            raise
        terminal = None
    return {
        "report": report,
        "terminal_path": terminal,
        "terminal_sha256": None if terminal is None else digest(report),
        "record_cleanup": _clone(state["record_cleanup"]),
        "record_cleanup_complete": _record_cleanup_complete(state),
        "persistence_errors": _clone(state["persistence_errors"]),
    }
```

If success-terminal publication fails **before** the link, preserve that
failure as primary, create an aborted report from the already-completed
episode ledger, and call `_finish` once for failure publication. Do not redo an
episode or call a model. If failure-terminal publication also fails, return the
in-memory error envelope and raise `DevelopmentRunError` from the original
success-publication exception.

If terminal publication commits but temporary unlink fails, `_save` returns
normally: return the committed successful envelope and its separate nonzero
`record_cleanup`. Do not rewrite the published report, emit an aborted report
or raise a contradictory exception. A diagnostics snapshot inside a committed
report cannot include cleanup events that occur after its bytes were linked;
the returned envelope is the explicit postcommit diagnostics channel.

For errors before root ownership use `terminal_path=None`,
`terminal_sha256=None`, no disk writes. For owned-root preflight failures,
publish `failure.json`/`partial-report.json` if possible; scheduled final counts
remain `None` until a schedule was validated.

At the root exception boundary, append a caught `BoundaryProtocolError` to
`state["protocol_errors"]` even when it came directly from the callback rather
than its record inspector. Inspectors' secondary protocol violations also
remain there; neither replaces `state["failure"]` or the original cause.
Compute a partial final score before building the report. Guard any exception
from that pure re-reduction as additional evidence; never let it replace the
first error. An error during success-terminal publication enters failure
publication only once; an error during failure publication returns the error
envelope without recursion.

- [ ] Add tests for first failure plus failing association/failure/partial
  publication; inspect `DevelopmentRunError.__cause__ is original_error`.
  Add success-terminal unlink failure: function returns success, report on disk
  remains completed, returned diagnostics retain the failure.
  Add success-terminal link failure: no committed success file, completed
  episode evidence retained, aborted partial report, sixteen calls not seventeen.

## Task 4: Synthetic Full Flow and Mutation Matrix

**Files:** Modify only `test_development_coordinator.py`; fix defects only in the
new coordinator, not dependencies.

**Consumes:** Accepted scoring's `synthetic_tasks`, `frozen_bytes` via a module
reference, the actual accepted schedule/renderers/scorer, Tasks 1–3.

**Produces:** Focused RED/GREEN evidence for fixed flow and failure accounting.

- [ ] Build all task material in memory. Reuse
  `import test_development_final_scoring as scoring_fixtures`, not its TestCase.
  Construct synthetic source/manifest and public views; patch only the four
  `validation_contract.PINS` entries inside tests, with automatic restoration.

```python
def synthetic_views():
    tasks = scoring_fixtures.synthetic_tasks()
    for task in tasks:
        task["spec"] = "Synthetic identity task"
        task["initial_witness"] = {
            "public_case_id": "P01",
            "predicted_initial": {"kind": "return", "value": 99},
        }
        if task["split"] == "supervision":
            task["reference_source"] = scoring_fixtures.CANDIDATE

    def public(task):
        return {
            "id": task["id"], "entry_point": task["entry_point"],
            "initial_source": task["initial_source"], "spec": task["spec"],
            "public_cases": [
                {name: copy.deepcopy(case[name]) for name in ("id", "args", "expected")}
                for case in task["public_cases"]
            ],
        }

    supervision = [
        {**public(task), "reference_source": task["reference_source"],
         "initial_witness": copy.deepcopy(task["initial_witness"])}
        for task in tasks if task["split"] == "supervision"
    ]
    prompts = [public(task) for task in tasks if task["split"] != "supervision"]
    source_raw, manifest_raw = scoring_fixtures.frozen_bytes(tasks)
    return encoded(supervision), encoded(prompts), source_raw, manifest_raw
```

Do not read frozen JSON files. Original `episode_jobs`, `_tasks`, `build_schedule`,
`visible_feedback`, `second_round`, `build_final_plan` and `score_final` must
actually run on these synthetic bytes; do not mock their main results.

- [ ] Implement synthetic callback evidence with literal values. Use a fixture
  state whose EOS token is 0 and output IDs are `[1, 0]`. Every generated
  candidate string is preselected by the test; no tokenizer/model or body runs:

```python
def generation_records(messages, call_id, expected_sha256, candidate):
    generation = {
        "kind": "LORA_GREEDY_CANDIDATE_UNATTESTED", "device_type": "cpu",
        "tokenizer_state_sha256": expected_sha256, "messages_sha256": digest(messages),
        "prompt_sha256": "f" * 64, "input_tokens": 10, "output_tokens": 2,
        "output_ids": [1, 0], "candidate": candidate,
        "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
        "stop_reason": "eos", "model_calls": 1, "forward_calls": 2,
        "timings_ns": {"preparation": 1, "generation": 2, "decoding": 1, "wall": 5},
        "peak_allocated_bytes": None, "peak_reserved_bytes": None,
        **{flag: False for flag in FLAGS},
    }
    observation = {
        "kind": "LORA_GENERATION_OBSERVATION", "call_id": call_id, "status": "completed",
        "messages_sha256": digest(messages), "forward_started": 2,
        "forward_completed": 2, "elapsed_ns": 6, "failure": None,
    }
    claim = {
        "kind": "LORA_GENERATION_CLAIM", "call_id": call_id,
        "messages_sha256": digest(messages), "expected_tokenizer_sha256": expected_sha256,
    }
    return claim, generation, observation


def save_json(directory, name, value):
    with (directory / name).open("xb") as output:
        output.write(encoded(value))
```

`self.generate` records only its supplied arguments in `self.calls`, selects
the next literal candidate from a sixteen-item test list, writes those three
records, and returns the exact observer envelope plus clean metadata. Nonfatal
cleanup tests return a valid nonempty diagnostics list without changing files.

Execution fixtures call the real **pure** admission and request-preparation
functions, construct the exact controller claim from their test bindings, and
use this fixed control stream:

```python
value = {"P01": 1, "P02": 2, "H01": 3}[request["case_id"]]
control = READY + encoded({"kind": "return", "value": value}) + b"\n"
audit = {
    "request_sha256": digest(request), "ready": True, "input_complete": True,
    "input_bytes_sent": len(payload), "reaped": True, "eof": True,
    "returncode": 0, "termination": None, "elapsed_seconds": 0.01,
    "send_started_seconds": 0.001, "output_bytes_observed": len(control),
    "failure": None,
    "stream_sha256": {
        "stdout": hashlib.sha256(b"").hexdigest(),
        "stderr": hashlib.sha256(b"").hexdigest(),
        "control": hashlib.sha256(control).hexdigest(),
    },
}
receipt = make_receipt(
    digest(request), returncode=0, ready=True, input_complete=True,
    reaped=True, eof=True, termination=None, result_raw=control[len(READY):])
```

Save admission, claim, observation, receipt and all three `.bin` files
exclusively; return only admission/audit/receipt. Mismatch variants substitute
literal value 99 by fixed task/arm/case lookup. Timeout/program-error/output-limit
fixtures construct corresponding coherent audit/control/receipt records using
`make_receipt`, never actual hanging/flooding processes.

- [ ] Add at least these explicit full-flow assertions:

```python
def test_all_public_pass_still_generates_sixteen_times(self):
    result = self.run_development()
    report = result["report"]
    self.assertEqual(len(self.calls), 16)
    self.assertEqual(report["counts"]["completed_episodes"], 8)
    self.assertEqual(report["counts"]["completed_generation_calls"], 16)
    self.assertEqual(report["counts"]["public_verified_receipts"], 16)
    self.assertEqual(report["counts"]["final_verified_receipts"], 24)
    self.assertEqual(report["costs"]["input_tokens"]["total"], 160)
    self.assertEqual(report["costs"]["output_tokens"]["total"], 32)
    self.assertEqual(report["calibration"]["status"], "INCONCLUSIVE_CALIBRATION")
    self.assertEqual(report["by_arm"]["base"]["passed"], 4)
    self.assertEqual(report["by_arm"]["text"]["passed"], 4)


def test_first_invalid_can_recover_without_erasing_warning(self):
    self.candidates[0] = "not valid python"
    result = self.run_development()
    episode = result["report"]["episodes"][0]
    self.assertIs(episode["first_round_valid"], False)
    self.assertEqual(episode["public_executions"], [])
    self.assertEqual(episode["feedback"], {"kind": "invalid_source"})
    self.assertEqual(episode["final_score"]["status"], "pass")
    self.assertEqual(len(self.calls), 16)
    self.assertIn("first_round_invalid_source",
                  result["report"]["calibration"]["reasons"]["INCONCLUSIVE_CALIBRATION"])


def test_final_invalid_keeps_full_case_denominator(self):
    self.candidates[1] = "not valid python"
    result = self.run_development()
    episode = result["report"]["episodes"][0]
    self.assertEqual(episode["final_executions"], [])
    self.assertEqual(episode["final_score"]["scheduled_cases"], 3)
    self.assertEqual(episode["final_score"]["received_cases"], 0)
    self.assertEqual(episode["final_score"]["status"], "invalid_source")
    self.assertEqual(result["report"]["counts"]["scheduled_final_cases"], 24)
```

Three cases per episode and 24 total are **synthetic fixture numbers**, not
claims about the real frozen D case counts.

- [ ] Add the following named mutation/failure tests. Each test has a concrete
  injection and expected invariant; keep parameterized subcases explicit:

| Test | Injection | Required result |
| --- | --- | --- |
| `test_exact_schedule_order_and_ids` | Compare captured calls with actual schedule IDs and messages | Eight alternating episodes; two consecutive calls per episode; no repeats |
| `test_second_messages_only_original_public_projection` | Compare second callback messages to actual `second_round` result | Equal canonical bytes; no private reference/reason sentinel or final-score keys |
| `test_execution_callbacks_receive_no_expectations` | Capture callback request key sets | Exact old/final request schemas; neither expected values nor full scorer plan |
| `test_input_4097_rejection_preserves_first_cost` | Second generation writes matching failed observation, zero forwards, raises | Two entries, one complete generation, first costs retained, failed token counts `None`, no second-call retry |
| `test_generation_failure_after_forward_is_not_zero_cost` | Failed observation started=3/completed=2, no generation | Observed forward totals include 3/2; input/output totals unknown; abort |
| `test_generation_exception_after_saved_success_still_aborts` | Save complete synthetic records, then raise | Completed saved evidence counted; callback not returned; no next callback |
| `test_final_failure_preserves_prefix` | Raise on second final callback | First receipt retained; remaining final rows missing; other seven positions missing |
| `test_public_failure_has_no_feedback_retry` | Raise on second public callback | One verified receipt retained; no feedback projection or round two |
| `test_policy_denial_is_not_model_error` | Literal syntactically valid `return open('/tmp/x').read()` | Pure admission denies before callback; protocol reason, no execution |
| `test_unknown_callback_error_is_incomplete` | Plain RuntimeError with no protocol evidence | Aborted/incomplete, not fabricated invalid_source or protocol assertion |
| `test_confirmed_driver_violation_has_priority` | Raise `BoundaryProtocolError` | INVALID_PROTOCOL plus incomplete reasons retained |
| `test_callback_mutation_cannot_change_schedule` | Mutate supplied messages/request | Snapshot mismatch aborts; persisted schedule and prior request remain unchanged |
| `test_prepared_and_claim_are_not_completed_work` | Save claim then fail before observation | Entered/claim verified distinct; no completed generation, forwards/tokens unknown |
| `test_final_receipt_is_not_first_round_feedback` | Fail private case but pass public cases | Final fail stored; next episode starts original messages only |
| `test_timeout_priority_keeps_invalid_warning` | First invalid then coherent final timeout | INCONCLUSIVE_INCOMPLETE and first-invalid calibration reason both present |
| `test_program_and_output_errors_are_not_infrastructure` | Coherent error receipts | Continue all sixteen calls; INCONCLUSIVE_CALIBRATION |
| `test_only_both_saturated_arms_trigger_calibration` | Fixed final outcomes 4/2, 0/2, 2/3 | DEVELOPMENT_CHECKS_SATISFIED if no other defects |
| `test_both_zero_and_both_four_trigger_calibration` | Fixed final outcomes 0/0 and 4/4 | INCONCLUSIVE_CALIBRATION |
| `test_failure_save_errors_keep_original_cause` | Callback error plus association/failure/terminal write errors | Original cause retained; diagnostics separately listed; no retry |
| `test_source_hash_failure_never_enters_callback` | Change one raw view without repinning | Protocol failure; all eight positions missing; case counts unknown |
| `test_cold_import_has_no_model_or_launcher_side_effects` | Import new module in an isolated stdlib subprocess with import guards | No Torch/Transformers load and no launcher invocation |

Use only a trusted import-check subprocess for the last test, never a candidate
worker. If the local task grant disallows even that subprocess, use an in-process
guard around a cold import and record the narrower verification.

### Final focused validation and handoff

- [ ] Run the full new synthetic suite and then the explicitly named pure
  dependency suite; no broad worker/execution discovery:

```bash
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error -m unittest discover -s experiments/lora_necessity -p test_development_coordinator.py -v 2>&1 | tee /tmp/tlf-candidate-executor-20261117-gb1gZX/development-coordinator-green.log
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/lora_necessity /opt/homebrew/opt/python@3.12/bin/python3.12 -B -W error -m unittest test_development_final_protocol test_development_final_scoring test_development_schedule test_development_coordinator -v 2>&1 | tee /tmp/tlf-candidate-executor-20261117-gb1gZX/development-coordinator-pure-regression.log
git diff --no-index --check /dev/null experiments/lora_necessity/development_coordinator.py
git diff --no-index --check /dev/null experiments/lora_necessity/test_development_coordinator.py
shasum -a 256 experiments/lora_necessity/development_coordinator.py experiments/lora_necessity/test_development_coordinator.py
```

Expected unittest result: all collected tests pass; publish actual method/subtest
counts and logs. Do not invent RED history or count import-discovery failure as
executed behavioral assertions. Main may run a separately authorized broader
suite; this plan does not authorize it.

- [ ] Save implementation evidence in
  `/tmp/tlf-candidate-executor-20261117-gb1gZX/development-coordinator-report.md`:
  source hashes, accepted dependency identities, commands/logs/counts, callbacks
  remaining synthetic, publication/partial accounting results and limitations.
- [ ] Independent review gate: fixed order/caps, original feedback only,
  first-invalid history, complete denominators, receipt/raw-file binding,
  unknown-cost accounting, error precedence and postcommit cleanup consistency.
- [ ] Stop after component handoff. Main owns schedule/scoring acceptance,
  future fixed remote driver, callback/model/runtime/launcher authenticity,
  original stream attestation, any real run and all stage transitions.

## Main-Facing Integration Questions

No blocking design conflict or unresolved product decision was identified.
The following concrete handoff checks are for Main's review, not reasons to
expand this component:

1. Resolved by Main: the accepted schedule source was rehashed and recorded in
   the prerequisite table. Existing accepted reviews are not restarted.
2. Accepted by Main: the driver uses `BoundaryProtocolError` only for positively identified
   protocol/identity/isolation failures. An arbitrary callback RuntimeError is
   insufficient evidence for that label and remains an incomplete abort.
   Missing records after a failed callback are absence of evidence, not by
   themselves a protocol violation. Malformed or contradictory readable records
   remain positive protocol evidence. A successful callback missing required
   records violates its success contract. The local pure admission call is the
   only automatic classification of `CandidateAdmissionError` as policy denial.

These are handoff checks for Main, not a request to the user for execution
permission, and do not justify any launch from this planning task.
