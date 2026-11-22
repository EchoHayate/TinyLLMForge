import hashlib
import math
from pathlib import Path
import re
import time

import candidate_execution_admission
import candidate_execution_protocol
import development_final_admission
import development_final_protocol
from candidate_feedback import candidate_valid, visible_feedback, visible_requests
from development_final_scoring import build_final_plan, score_final
from candidate_execution_admission import CandidateAdmissionError, _plain_json
from development_schedule import build_schedule
from observed_generation import _failure, _save, _validate
from pilot_protocol import FLAGS
from prompt_rendering import second_round
from validation_contract import digest, encoded, parse_json


ORDER = tuple(
    (f"D0{ordinal}", arm)
    for ordinal in range(1, 5)
    for arm in (("base", "text") if ordinal % 2 else ("text", "base"))
)

COMPONENTS = (
    "development_coordinator.py", "development_schedule.py",
    "development_final_scoring.py", "development_final_protocol.py",
    "development_final_admission.py", "development_final_controller.py",
    "candidate_execution_admission.py", "candidate_execution_controller.py",
    "candidate_execution_protocol.py", "candidate_feedback.py", "prompt_rendering.py",
    "observed_generation.py", "generation_core.py", "tokenizer_measurement.py",
    "validation_contract.py", "pilot_protocol.py",
)

CALIBRATION_PRECEDENCE = (
    "INVALID_PROTOCOL", "INCONCLUSIVE_INCOMPLETE", "INCONCLUSIVE_CALIBRATION",
    "DEVELOPMENT_CHECKS_SATISFIED",
)


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


def _preflight(run_id, raw, bindings, callbacks):
    _need(_hex(run_id, 32), "run identity")
    _need(type(raw) is tuple and len(raw) == 4
          and all(type(value) is bytes for value in raw), "four immutable byte snapshots")
    _need(type(callbacks) is tuple and len(callbacks) == 3
          and all(callable(callback) for callback in callbacks), "three callable boundaries")
    return _checked(_bindings, bindings)


def _owned_root(artifact_dir):
    root = Path(artifact_dir)
    root.mkdir(mode=0o700)
    return root


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


def _claim(root, state, raw, bindings):
    declaration = _checked(_bindings, bindings)
    schedule = _checked(build_schedule, *raw, run_id=state["run_id"])
    schedule = _checked(_validate_schedule, schedule, state["run_id"], raw)
    state["schedule"] = schedule
    for episode, job in zip(state["episodes"], schedule["episodes"]):
        episode["scheduled_final_cases"] = job["scheduled_final_cases"]
    _publish(root, "claim.json", {
        "kind": "LORA_DEVELOPMENT_CLAIM", "version": 1, "run_id": state["run_id"],
        "schedule_sha256": digest(schedule), "data_sha256": schedule["data_sha256"],
        "bindings": declaration, "bindings_authenticity": "CALLER_DECLARED",
        "scheduled_episodes": 8, "scheduled_generation_calls": 16,
        "input_token_limit": 4096, "output_token_limit": 256,
        "episode_order": [{"task_id": task_id, "arm": arm} for task_id, arm in ORDER],
        "calibration_precedence": list(CALIBRATION_PRECEDENCE),
    }, state)
    _publish(root, "schedule.json", schedule, state)
    return schedule


def _validate_schedule(schedule, run_id, raw):
    try:
        schedule = _clone(schedule)
    except CandidateAdmissionError as error:
        raise BoundaryProtocolError(str(error)) from error
    _fields(schedule, (
        "kind", "version", "run_id", "data_sha256", "scheduled_episodes",
        "scheduled_generation_calls", "episodes",
    ))
    _need(schedule["kind"] == "LORA_DEVELOPMENT_SCHEDULE"
          and type(schedule["version"]) is int and schedule["version"] == 1
          and schedule["run_id"] == run_id, "schedule identity")
    for name, count in (("scheduled_episodes", 8), ("scheduled_generation_calls", 16)):
        _need(type(schedule[name]) is int and schedule[name] == count, "schedule cap")
    _need(schedule["data_sha256"] == {
        name: hashlib.sha256(value).hexdigest()
        for name, value in zip((
            "supervision_v1.json", "prompts_v1.json",
            "task_sources_v1.json", "manifest_v1.json",
        ), raw)
    }, "schedule data identity")
    jobs = schedule["episodes"]
    _need(type(jobs) is list and len(jobs) == 8, "eight scheduled episodes")
    identifiers = set()
    for ordinal, (job, identity) in enumerate(zip(jobs, ORDER)):
        _fields(job, (
            "ordinal", "task_id", "arm", "messages", "examples_sha256", "task_sha256",
            "messages_sha256", "first_call_id", "second_call_id", "public_attempt_id",
            "scheduled_final_cases", "final_cases",
        ))
        _need(type(job["ordinal"]) is int and job["ordinal"] == ordinal
              and (job["task_id"], job["arm"]) == identity, "ordered episode identity")
        _need(_hex(job["task_sha256"]) and _hex(job["messages_sha256"])
              and digest(job["messages"]) == job["messages_sha256"], "episode hashes")
        _need(job["examples_sha256"] is None if job["arm"] == "base"
              else _hex(job["examples_sha256"]), "examples identity")
        for name in ("first_call_id", "second_call_id", "public_attempt_id"):
            identifier = job[name]
            _need(_hex(identifier, 32) and identifier not in identifiers,
                  "unique logical identity")
            identifiers.add(identifier)
        inventory = job["final_cases"]
        _need(type(inventory) is list and type(job["scheduled_final_cases"]) is int
              and job["scheduled_final_cases"] == len(inventory) > 0,
              "scheduled case denominator")
        case_order = []
        for case in inventory:
            _fields(case, ("case_group", "case_id", "input_sha256", "expected_sha256"))
            _need(case["case_group"] in ("public", "scoring")
                  and type(case["case_id"]) is str and bool(case["case_id"])
                  and _hex(case["input_sha256"]) and _hex(case["expected_sha256"]),
                  "hash-only final inventory")
            case_order.append((case["case_group"] == "scoring", case["case_id"]))
        _need(case_order == sorted(set(case_order)), "unique ordered final cases")
    return schedule


def _hex(value, length=64):
    return type(value) is str and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def _fields(value, names):
    _need(type(value) is dict and set(value) == set(names), "unexpected fields")


def _bindings(bindings):
    try:
        snapshot = _clone(bindings)
    except CandidateAdmissionError as error:
        raise BoundaryProtocolError(str(error)) from error
    _fields(snapshot, (
        "component_sha256", "runtime_sha256", "model_sha256", "tokenizer_state", "execution",
    ))
    _fields(snapshot["component_sha256"], COMPONENTS)
    _need(all(_hex(value) for value in snapshot["component_sha256"].values()),
          "component SHA256")
    _need(_hex(snapshot["runtime_sha256"]) and _hex(snapshot["model_sha256"]),
          "runtime/model SHA256")
    tokenizer = snapshot["tokenizer_state"]
    _fields(tokenizer, ("template_sha256", "vocabulary_sha256", "eos_token_id", "pad_token_id"))
    _need(_hex(tokenizer["template_sha256"]) and _hex(tokenizer["vocabulary_sha256"]),
          "tokenizer SHA256")
    _need(all(type(tokenizer[name]) is int and tokenizer[name] >= 0
              for name in ("eos_token_id", "pad_token_id")), "tokenizer token ID")
    _fields(snapshot["execution"], ("public", "final"))
    for execution in snapshot["execution"].values():
        _fields(execution, (
            "argv", "startup_timeout", "candidate_timeout", "output_limit",
            "result_limit", "reap_timeout",
        ))
        for name in ("startup_timeout", "candidate_timeout", "reap_timeout"):
            value = execution[name]
            try:
                _need(type(value) in (int, float) and math.isfinite(value) and value > 0,
                      "invalid deadline")
            except OverflowError as error:
                raise BoundaryProtocolError("invalid deadline") from error
        for name, maximum in (("output_limit", 65536), ("result_limit", 16384)):
            value = execution[name]
            _need(type(value) is int and 0 < value <= maximum, "invalid byte limit")
        argv = execution["argv"]
        _need(type(argv) is list and bool(argv)
              and all(type(value) is str and "\0" not in value for value in argv)
              and sum(value.count("{control_fd}") for value in argv) == 1,
              "trusted argv requires exactly one control descriptor token")
    return snapshot


def _new_state(run_id):
    return {
        "run_id": run_id if _hex(run_id, 32) else None, "schedule": None, "active": None,
        "record_cleanup": {"failed_count": 0, "failures": []},
        "persistence_errors": [], "protocol_errors": [], "failure": None,
        "final_identity": {},
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


def _clone(value):
    _plain_json(value)
    raw = encoded(value)
    return parse_json(raw, limit=len(raw))


def _generation(root, episode_dir, rows, messages, round_number, job, bindings, generate, state):
    _need(round_number in (1, 2) and type(round_number) is int, "generation round")
    snapshot = _native(messages)
    supplied = _clone(snapshot)
    call_id = job["first_call_id" if round_number == 1 else "second_call_id"]
    relative = f"{episode_dir}/generation-{round_number}"
    row = _association("generation", relative)
    row.update(round=round_number, call_id=call_id, messages_sha256=digest(snapshot))
    rows.append(row)
    _need(sum(item["callback_entered"] for episode in state["episodes"]
              for item in episode["generations"]) < 16, "generation callback cap")
    expected = {
        "kind": "LORA_GENERATION_CLAIM", "call_id": call_id,
        "messages_sha256": digest(snapshot),
        "expected_tokenizer_sha256": digest(bindings["tokenizer_state"]),
    }
    prepared = {
        "kind": "LORA_DEVELOPMENT_PREPARED", "version": 1, "run_id": state["run_id"],
        "phase": "generation", "episode_ordinal": job["ordinal"], "round": round_number,
        "call_id": call_id, "messages_sha256": digest(snapshot), "relative_dir": relative,
    }

    def invoke(directory):
        return generate(supplied, expected_tokenizer_sha256=expected["expected_tokenizer_sha256"],
                        call_id=call_id, artifact_dir=directory)

    def inspect(directory, returned, primary, current):
        names = ("claim.json", "observation.json", "generation.json")
        blobs, records, errors = _records(directory, names, current, primary)
        if "claim.json" in records:
            if _inspect_check(errors, _same, records["claim.json"], expected, "generation claim"):
                current["claim_verified"] = True
        observation = None
        if "observation.json" in records:
            observation = _inspect_check(
                errors, _generation_observation, records["observation.json"], expected)
            if observation is not None:
                current.update(observation_verified=True, observation_status=observation["status"],
                               forward_started=observation["forward_started"],
                               forward_completed=observation["forward_completed"],
                               observer_elapsed_ns=observation["elapsed_ns"])
        if "generation.json" in records and observation is not None:
            generation = _inspect_check(errors, _generation_values, records["generation.json"],
                                        observation, snapshot, bindings)
            if generation is not None:
                current.update(generation_verified=True, input_tokens=generation["input_tokens"],
                               output_tokens=generation["output_tokens"],
                               generation_timings_ns=_clone(generation["timings_ns"]),
                               peak_allocated_bytes=generation["peak_allocated_bytes"],
                               peak_reserved_bytes=generation["peak_reserved_bytes"])
        _inspect_check(errors, _same, supplied, snapshot, "callback changed messages")
        if primary is None:
            if observation is not None:
                _inspect_check(errors, _need, observation["status"] == "completed",
                               "successful callback saved failed observation")
            envelope = _inspect_check(errors, _envelope, returned,
                                      ("generation", "observation", "record_cleanup"))
            if envelope is not None:
                current["record_cleanup"] = _inspect_check(
                    errors, _cleanup, envelope["record_cleanup"])
                for key in ("generation", "observation"):
                    if key + ".json" in blobs:
                        _inspect_check(errors, _need,
                                       encoded(envelope[key]) == blobs[key + ".json"],
                                       "returned generation evidence changed")
            if not any("file" in item for item in current["evidence_errors"]):
                _inspect_check(errors, _need, current["generation_verified"]
                               and current["observation_verified"]
                               and current["observation_status"] == "completed",
                               "successful generation evidence incomplete")
        else:
            cleanup = _inspect_check(errors, getattr, primary, "record_cleanup", None)
            if cleanup is not None:
                current["record_cleanup"] = _inspect_check(errors, _cleanup, cleanup)
        _inspection_done(errors, current, state)
        return blobs.get("generation.json")

    return _attempt(root, relative, prepared, row, state, invoke, inspect)


def _execution(root, episode_dir, rows, request, phase, bindings, execute, state):
    _need(phase in ("public", "final"), "execution phase")
    snapshot = _native(request)
    supplied = _clone(snapshot)
    protocol = development_final_protocol if phase == "final" else candidate_execution_protocol
    admission_module = development_final_admission if phase == "final" else candidate_execution_admission
    request_sha256, payload = _checked(protocol.prepare_request, snapshot)
    group = snapshot.get("case_group", "public")
    suffix = f"final-{len(rows):03d}-{group}-{snapshot['case_id']}" if phase == "final" \
        else f"public-{len(rows):02d}-{snapshot['case_id']}"
    relative = f"{episode_dir}/{suffix}"
    row = _association(phase, relative)
    row.update(round=snapshot["round"], case_group=group, case_id=snapshot["case_id"],
               request_sha256=request_sha256)
    rows.append(row)
    state["active"] = relative
    try:
        admission = admission_module.admit_request(snapshot)
    except CandidateAdmissionError as error:
        raise BoundaryProtocolError(str(error)) from error
    execution = bindings["execution"][phase]
    expected_claim = {
        "request_sha256": request_sha256, "argv": execution["argv"],
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        **{name: execution[name] for name in
           ("startup_timeout", "candidate_timeout", "output_limit", "result_limit")},
    }
    ordinal = ORDER.index((snapshot["task_id"], snapshot["arm"]))
    prepared = {
        "kind": "LORA_DEVELOPMENT_PREPARED", "version": 1, "run_id": state["run_id"],
        "phase": phase, "episode_ordinal": ordinal, "round": snapshot["round"],
        "attempt_id": snapshot["attempt_id"], "request_sha256": request_sha256,
        "case_group": group, "case_id": snapshot["case_id"], "request": snapshot,
        "execution": _clone(execution), "relative_dir": relative,
    }

    def invoke(directory):
        return execute(supplied, artifact_dir=directory)

    def inspect(directory, returned, primary, current):
        names = ("admission.json", "claim.json", "observation.json", "receipt.json",
                 "stdout.bin", "stderr.bin", "control.bin")
        blobs, records, errors = _records(directory, names, current, primary)
        if "admission.json" in records:
            _inspect_check(errors, _same, records["admission.json"], admission, "admission binding")
        if "claim.json" in records:
            if _inspect_check(errors, _same, records["claim.json"], expected_claim, "execution claim"):
                current["claim_verified"] = True
        audit = None
        if "observation.json" in records:
            audit = _inspect_check(errors, _audit_values, records["observation.json"],
                                   request_sha256, payload)
        if primary is None and audit is not None:
            _inspect_check(errors, _successful_audit, audit, payload)
        streams = all(name + ".bin" in blobs for name in ("stdout", "stderr", "control"))
        if audit is not None and streams:
            consistent = _inspect_check(errors, _execution_streams, audit, blobs, execution)
            if consistent and current["claim_verified"]:
                current.update(observation_verified=True,
                               execution_elapsed_seconds=audit["elapsed_seconds"])
                if audit["ready"] or type(audit["returncode"]) is int:
                    current["process_started"] = True
        receipt = None
        if audit is not None and current["observation_verified"] and "receipt.json" in records:
            complete = audit["failure"] is None and audit["ready"] and audit["input_complete"] \
                and audit["reaped"] and (audit["eof"] or audit["termination"] is not None) \
                and type(audit["returncode"]) is int
            if primary is None or complete:
                receipt = _inspect_check(errors, _receipt_from_evidence, request_sha256, payload,
                                         execution, audit, blobs)
                if receipt is not None and _inspect_check(
                        errors, _same, records["receipt.json"], receipt, "receipt reconstruction"):
                    current.update(receipt_verified=True, receipt=_clone(receipt))
        _inspect_check(errors, _same, supplied, snapshot, "callback changed request")
        if primary is None:
            envelope = _inspect_check(errors, _envelope, returned, ("admission", "audit", "receipt"))
            if envelope is not None:
                for key, filename in (("admission", "admission.json"), ("audit", "observation.json"),
                                      ("receipt", "receipt.json")):
                    if filename in records:
                        _inspect_check(errors, _same, envelope[key], records[filename],
                                       "returned execution evidence changed")
            if not any("file" in item for item in current["evidence_errors"]):
                _inspect_check(errors, _need, current["receipt_verified"], "successful receipt missing")
        _inspection_done(errors, current, state)
        return receipt

    return _attempt(root, relative, prepared, row, state, invoke, inspect)


def _native(value):
    try:
        return _checked(_clone, value)
    except CandidateAdmissionError as error:
        raise BoundaryProtocolError(str(error)) from error


def _same(actual, expected, message):
    _need(encoded(_native(actual)) == encoded(expected), message)
    return True


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _duration(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _association(phase, relative):
    return {
        "phase": phase, "relative_dir": relative, "prepared": False,
        "callback_entered": False, "callback_returned": False, "claim_verified": False,
        "observation_verified": False, "observation_status": None,
        "generation_verified": False, "receipt_verified": False,
        "process_started": None, "forward_started": None, "forward_completed": None,
        "input_tokens": None, "output_tokens": None, "generation_timings_ns": None,
        "observer_elapsed_ns": None, "execution_elapsed_seconds": None,
        "peak_allocated_bytes": None, "peak_reserved_bytes": None, "record_cleanup": None,
        "receipt": None, "files": {}, "failure": None, "evidence_errors": [],
    }


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
        row["files"][name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return blobs, first_error


def _json(blobs, name):
    _need(name in blobs, "missing successful record: " + name)
    return _checked(parse_json, blobs[name], limit=len(blobs[name]))


def _records(directory, names, row, primary):
    blobs, read_error = _capture(directory, names, row)
    errors = [] if read_error is None else [read_error]
    unreadable = {item["file"] for item in row["evidence_errors"] if "file" in item}
    records = {}
    for name in names:
        if name not in blobs:
            if primary is None and name not in unreadable:
                errors.append(BoundaryProtocolError("missing successful record: " + name))
        elif name.endswith(".json"):
            record = _inspect_check(errors, _json, blobs, name)
            if record is not None:
                records[name] = record
            elif blobs[name].strip() == b"null":
                errors.append(BoundaryProtocolError("record object required: " + name))
    return blobs, records, errors


def _inspect_check(errors, function, *arguments):
    try:
        return _checked(function, *arguments)
    except BaseException as error:
        errors.append(error)
        return None


def _inspection_done(errors, row, state):
    if errors:
        for error in errors[1:]:
            row["evidence_errors"].append(_failure(error))
            if isinstance(error, BoundaryProtocolError):
                state["protocol_errors"].append(_failure(error))
        raise errors[0]


def _envelope(value, names):
    snapshot = _native(value)
    _fields(snapshot, names)
    return snapshot


def _diagnostic(value):
    _need(type(value) is dict and set(value) in (
        {"type", "message"}, {"type", "message", "cleanup_errors"}), "diagnostic fields")
    _need(type(value["type"]) is str and type(value["message"]) is str, "diagnostic strings")
    if "cleanup_errors" in value:
        _need(type(value["cleanup_errors"]) is list, "additional cleanup errors")
        for error in value["cleanup_errors"]:
            _fields(error, ("type", "message"))
            _diagnostic(error)


def _cleanup(value):
    snapshot = _envelope(value, ("failed_count", "failures"))
    _need(_integer(snapshot["failed_count"]) and type(snapshot["failures"]) is list
          and snapshot["failed_count"] == len(snapshot["failures"]), "record cleanup count")
    for item in snapshot["failures"]:
        _fields(item, ("record", "temporary", "published", "error"))
        _need(type(item["record"]) is str and type(item["temporary"]) is str
              and type(item["published"]) is bool, "record cleanup fields")
        _fields(item["error"], ("type", "message"))
        _diagnostic(item["error"])
    return snapshot


def _generation_observation(observation, claim):
    _fields(observation, ("kind", "call_id", "status", "messages_sha256", "forward_started",
                          "forward_completed", "elapsed_ns", "failure"))
    _need(observation["kind"] == "LORA_GENERATION_OBSERVATION"
          and observation["call_id"] == claim["call_id"]
          and observation["messages_sha256"] == claim["messages_sha256"], "observation identity")
    _need(all(_integer(observation[name]) for name in
              ("forward_started", "forward_completed", "elapsed_ns"))
          and observation["forward_completed"] <= observation["forward_started"],
          "observation counters")
    if observation["status"] == "completed":
        _need(observation["failure"] is None, "completed observation failure")
    else:
        _need(observation["status"] == "failed", "observation status")
        _diagnostic(observation["failure"])
    return observation


def _generation_values(generation, observation, messages, bindings):
    tokenizer = bindings["tokenizer_state"]
    _checked(_validate, generation, messages, digest(messages), digest(tokenizer),
             observation["forward_started"], observation["forward_completed"])
    _fields(generation, {
        "kind", "device_type", "tokenizer_state_sha256", "messages_sha256", "prompt_sha256",
        "input_tokens", "output_tokens", "output_ids", "candidate", "candidate_sha256",
        "stop_reason", "model_calls", "forward_calls", "timings_ns",
        "peak_allocated_bytes", "peak_reserved_bytes", *FLAGS,
    })
    _need(_integer(generation["input_tokens"], 1)
          and generation["input_tokens"] <= 4096, "input token budget")
    _need(_hex(generation["prompt_sha256"]), "prompt identity format")
    output_ids = generation["output_ids"]
    eos = tokenizer["eos_token_id"]
    if generation["stop_reason"] == "eos":
        _need(output_ids[-1] == eos and eos not in output_ids[:-1], "EOS boundary")
    else:
        _need(generation["stop_reason"] == "length" and len(output_ids) == 256
              and eos not in output_ids, "length boundary")
    timings = generation["timings_ns"]
    _fields(timings, ("preparation", "generation", "decoding", "wall"))
    _need(all(_integer(value) for value in timings.values()), "generation timings")
    _need(sum(timings[name] for name in ("preparation", "generation", "decoding"))
          <= timings["wall"] <= observation["elapsed_ns"], "nested generation times")
    _need(all(generation[flag] is False for flag in FLAGS), "generation flags")
    allocated, reserved = generation["peak_allocated_bytes"], generation["peak_reserved_bytes"]
    if generation["device_type"] == "cpu":
        _need(allocated is None and reserved is None, "unknown CPU memory")
    else:
        _need(generation["device_type"] == "cuda" and _integer(allocated) and _integer(reserved)
              and allocated <= reserved, "CUDA peak fields")
    return generation


def _audit_values(audit, request_sha256, payload):
    _fields(audit, (
        "request_sha256", "ready", "input_complete", "input_bytes_sent", "reaped", "eof",
        "returncode", "termination", "elapsed_seconds", "send_started_seconds",
        "output_bytes_observed", "failure", "stream_sha256",
    ))
    _need(audit["request_sha256"] == request_sha256, "audit request binding")
    _need(all(type(audit[name]) is bool for name in ("ready", "input_complete", "reaped", "eof")),
          "audit lifecycle types")
    _need(_integer(audit["input_bytes_sent"]) and audit["input_bytes_sent"] <= len(payload),
          "audit input bytes")
    _need(audit["input_complete"] == (audit["input_bytes_sent"] == len(payload)),
          "audit input completion contradicts byte count")
    _need(audit["returncode"] is None or type(audit["returncode"]) is int, "audit return code")
    _need(audit["termination"] in (None, "timeout", "output_limit"), "audit termination")
    _need(audit["failure"] is None or type(audit["failure"]) is str, "audit failure")
    _need(_duration(audit["elapsed_seconds"]) and (
        audit["send_started_seconds"] is None or _duration(audit["send_started_seconds"])
        and audit["send_started_seconds"] <= audit["elapsed_seconds"]), "audit times")
    _need(_integer(audit["output_bytes_observed"]), "audit output bytes")
    _fields(audit["stream_sha256"], ("stdout", "stderr", "control"))
    _need(all(_hex(value) for value in audit["stream_sha256"].values()), "audit stream digests")
    return audit


def _execution_streams(audit, blobs, execution):
    observed = sum(len(blobs[name + ".bin"]) for name in ("stdout", "stderr", "control"))
    _need(observed == audit["output_bytes_observed"]
          and observed <= execution["output_limit"] + 1, "stream byte accounting")
    _same(audit["stream_sha256"], {
        name: hashlib.sha256(blobs[name + ".bin"]).hexdigest()
        for name in ("stdout", "stderr", "control")
    }, "stream hashes")
    if audit["ready"]:
        _need(blobs["control.bin"].startswith(candidate_execution_protocol.READY), "control READY")
    return True


def _successful_audit(audit, payload):
    _need(audit["failure"] is None, "failed execution audit")
    _need(audit["input_bytes_sent"] == len(payload), "input byte count")
    _need(_duration(audit["send_started_seconds"]), "execution send time")
    _need(audit["ready"] and audit["input_complete"] and audit["reaped"]
          and (audit["eof"] or audit["termination"] is not None)
          and type(audit["returncode"]) is int, "incomplete successful execution lifecycle")


def _receipt_from_evidence(request_sha256, payload, execution, audit, blobs):
    _successful_audit(audit, payload)
    control = blobs["control.bin"]
    ready = candidate_execution_protocol.READY
    _need(control.startswith(ready), "control READY")
    if (len(control) - len(ready) > execution["result_limit"]
            or audit["output_bytes_observed"] > execution["output_limit"]):
        _need(audit["termination"] == "output_limit", "output-limit termination")
    try:
        return candidate_execution_protocol.make_receipt(
            request_sha256, returncode=audit["returncode"], ready=audit["ready"],
            input_complete=audit["input_complete"], reaped=audit["reaped"], eof=audit["eof"],
            termination=audit["termination"], result_raw=control[len(ready):])
    except candidate_execution_protocol.InfrastructureError as error:
        raise BoundaryProtocolError(str(error)) from error


def run_development(supervision_raw, prompts_raw, source_raw, manifest_raw, *,
                    run_id, artifact_dir, bindings, generate, execute_public, execute_final):
    started = time.perf_counter_ns()
    state = _new_state(run_id)
    raw = (supervision_raw, prompts_raw, source_raw, manifest_raw)
    callbacks = (generate, execute_public, execute_final)
    root = None
    active_episode = None
    report_error = None
    try:
        declaration = _preflight(run_id, raw, bindings, callbacks)
        root = _owned_root(artifact_dir)
        schedule = _claim(root, state, raw, declaration)
        (root / "episodes").mkdir(mode=0o700)
        for job, episode in zip(schedule["episodes"], state["episodes"]):
            active_episode = episode
            _run_episode(root, job, episode, state, raw, declaration, callbacks)
        state["active"] = "development-report.json"
        try:
            report = _report(state, time.perf_counter_ns() - started, aborted=False)
        except BaseException as error:
            report_error = error
            raise
        return _finish(root, state, report, None)
    except BaseException as primary:
        state["failure"] = {
            "kind": "LORA_DEVELOPMENT_FAILURE", "version": 1, "run_id": state["run_id"],
            "active": state["active"], "error": _failure(primary),
        }
        if isinstance(primary, BoundaryProtocolError):
            state["protocol_errors"].append(_failure(primary))
        if active_episode is not None:
            try:
                _partial_score(active_episode, state, source_raw, manifest_raw)
            except BaseException as error:
                state["failure"]["partial_score_error"] = _failure(error)
                if isinstance(error, BoundaryProtocolError):
                    state["protocol_errors"].append(_failure(error))
        if report_error is None:
            try:
                report = _report(state, time.perf_counter_ns() - started, aborted=True)
            except BaseException as error:
                report_error = error
        if report_error is not None:
            state["failure"]["report_error"] = _failure(report_error)
            report = _report_snapshot(state, aborted=True)
        if root is None:
            envelope = {
                "report": report, "terminal_path": None, "terminal_sha256": None,
                "record_cleanup": _clone(state["record_cleanup"]),
                "record_cleanup_complete": _record_cleanup_complete(state),
                "persistence_errors": _clone(state["persistence_errors"]),
            }
        else:
            envelope = _finish(root, state, report, primary)
        raise DevelopmentRunError(envelope) from primary


def _cost(rows, name, scheduled=16):
    values = [row[name] for row in rows if row[name] is not None]
    return {
        "total": sum(values) if len(values) == scheduled else None,
        "observed_subtotal": sum(values) if values else None,
        "known_calls": len(values), "scheduled_calls": scheduled,
    }


def _execution_cost(rows):
    elapsed = [row["execution_elapsed_seconds"] for row in rows
               if row["execution_elapsed_seconds"] is not None]
    aggregation_error = None
    try:
        subtotal = sum(elapsed) if elapsed else None
        if type(subtotal) is float and not math.isfinite(subtotal):
            raise OverflowError("execution elapsed aggregate is not finite")
    except OverflowError as error:
        subtotal = None
        aggregation_error = _failure(error)
    return {
        "total": subtotal if len(elapsed) == len(rows) else None,
        "observed_subtotal": subtotal,
        "known_attempts": len(elapsed), "entered_attempts": len(rows),
        "unknown_attempts": len(rows) - len(elapsed), "aggregation_error": aggregation_error,
    }


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


def _run_episode(root, job, episode, state, raw, bindings, callbacks):
    supervision_raw, prompts_raw, source_raw, manifest_raw = raw
    generate, execute_public, execute_final = callbacks
    episode_dir = f'episodes/{job["ordinal"]:02d}-{job["task_id"]}-{job["arm"]}'
    state["active"] = episode_dir
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
    valid = _checked(candidate_valid, prompts_raw, job["task_id"], job["arm"], first["candidate"])
    episode["first_round_valid"] = valid
    requests = _checked(visible_requests, prompts_raw, job["task_id"], job["arm"],
                        first["candidate"], attempt_id=job["public_attempt_id"])
    _need(len(requests) == (2 if valid else 0), "public request coverage")
    receipts = []
    for request in requests:
        receipts.append(_execution(root, episode_dir, episode["public_executions"], request,
                                   "public", bindings, execute_public, state))
    feedback = _checked(visible_feedback, prompts_raw, job["task_id"], job["arm"],
                        first["candidate"], encoded(receipts), attempt_id=job["public_attempt_id"])
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
        "kind": "LORA_DEVELOPMENT_FINAL_BINDING", "version": 1, "run_id": state["run_id"],
        "generation_call_id": job["second_call_id"],
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
    _publish(directory, "result.json", {
        "kind": "LORA_DEVELOPMENT_EPISODE_RESULT", "version": 1, "run_id": state["run_id"],
        **completed,
    }, state)
    episode["completed"] = True


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


def _calibration(episodes, protocol_errors, aborted, necessary_costs_known):
    reasons = {label: [] for label in CALIBRATION_PRECEDENCE[:-1]}
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
        row["receipt"]["status"] for episode in episodes
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
    decision = next((label for label in CALIBRATION_PRECEDENCE[:-1] if reasons[label]),
                    CALIBRATION_PRECEDENCE[-1])
    return {"status": decision, "reasons": reasons}


def _report(state, coordinator_wall_ns, *, aborted):
    episodes = state["episodes"]
    generations = [row for episode in episodes for row in episode["generations"]]
    public = [row for episode in episodes for row in episode["public_executions"]]
    final = [row for episode in episodes for row in episode["final_executions"]]
    scheduled = None if state["schedule"] is None else sum(
        episode["scheduled_final_cases"] for episode in episodes)
    scores = [episode["final_score"] for episode in episodes if episode["final_score"] is not None]
    counts = {
        "completed_episodes": sum(episode["completed"] for episode in episodes),
        "generation_callbacks_entered": sum(row["callback_entered"] for row in generations),
        "generation_callbacks_returned": sum(row["callback_returned"] for row in generations),
        "verified_generation_claims": sum(row["claim_verified"] for row in generations),
        "verified_generation_records": sum(row["generation_verified"] for row in generations),
        "completed_generation_calls": sum(
            row["generation_verified"] and row["observation_verified"]
            and row["observation_status"] == "completed" for row in generations),
        "public_feedback_opportunities": sum(episode["feedback_opportunities"] for episode in episodes),
        "scheduled_final_cases": scheduled,
        "received_final_cases": None if scheduled is None else sum(
            score["received_cases"] for score in scores),
        "passed_final_cases": None if scheduled is None else sum(score["passed_cases"] for score in scores),
        "missing_final_cases": None if scheduled is None else sum(
            episode["scheduled_final_cases"] if episode["final_score"] is None else
            sum(case["status"] == "missing" for case in episode["final_score"]["cases"])
            for episode in episodes),
        "invalid_final_cases": None if scheduled is None else sum(
            case["status"] == "invalid_source" for score in scores for case in score["cases"]),
        "unexecuted_final_cases": None if scheduled is None else
            scheduled - sum(row["callback_entered"] for row in final),
        "missing_episode_positions": [episode["ordinal"] for episode in episodes
                                      if not episode["completed"]],
    }
    for phase, rows in (("public", public), ("final", final)):
        counts.update({
            phase + "_callbacks_entered": sum(row["callback_entered"] for row in rows),
            phase + "_verified_child_starts": sum(row["process_started"] is True for row in rows),
            phase + "_unknown_child_starts": sum(
                row["callback_entered"] and row["process_started"] is not True for row in rows),
            phase + "_verified_receipts": sum(row["receipt_verified"] for row in rows),
        })
    costs = {name: _cost(generations, name) for name in (
        "input_tokens", "output_tokens", "forward_started", "forward_completed", "observer_elapsed_ns",
    )}
    costs["generation_wall_ns"] = _cost([
        {"wall": None if row["generation_timings_ns"] is None else row["generation_timings_ns"]["wall"]}
        for row in generations
    ], "wall")
    entered = [row for row in public + final if row["callback_entered"]]
    costs.update({
        "execution_elapsed_seconds": _execution_cost(entered),
        "generation_records": generations, "execution_records": public + final,
        "coordinator_wall_ns": coordinator_wall_ns,
        "driver_model_load_ns": None, "driver_shared_preparation_ns": None, "driver_outer_wall_ns": None,
    })
    necessary = all(costs[name]["known_calls"] == 16 for name in (
        "input_tokens", "output_tokens", "generation_wall_ns", "observer_elapsed_ns",
    )) and all(row["execution_elapsed_seconds"] is not None for row in entered if row["receipt_verified"]) \
        and costs["execution_elapsed_seconds"]["aggregation_error"] is None
    by_arm = {}
    for arm in ("base", "text"):
        selected = [episode for episode in episodes if episode["arm"] == arm]
        by_arm[arm] = {
            "scheduled": 4, "completed": sum(episode["completed"] for episode in selected),
            "passed": sum(episode["final_score"] is not None
                          and episode["final_score"]["status"] == "pass" for episode in selected),
            "episodes": [
                {"task_id": episode["task_id"], "final_status": "missing"
                 if episode["final_score"] is None else episode["final_score"]["status"]}
                for episode in selected
            ],
        }
    return _report_snapshot(
        state, aborted=aborted, counts=counts, by_arm=by_arm, costs=costs,
        calibration=_calibration(episodes, state["protocol_errors"], aborted, necessary))


def _report_snapshot(state, *, aborted, counts=None, by_arm=None, costs=None, calibration=None):
    if calibration is None:
        protocol = bool(state["protocol_errors"])
        calibration = {
            "status": "INVALID_PROTOCOL" if protocol else "INCONCLUSIVE_INCOMPLETE",
            "reasons": {
                "INVALID_PROTOCOL": ["confirmed_protocol_identity_or_isolation_failure"] if protocol else [],
                "INCONCLUSIVE_INCOMPLETE": ["run_aborted", "report_reduction_failed"],
                "INCONCLUSIVE_CALIBRATION": [],
            },
        }
    return _clone({
        "kind": "LORA_DEVELOPMENT_REPORT", "version": 1, "run_id": state["run_id"],
        "status": "aborted" if aborted else "completed",
        "schedule_sha256": None if state["schedule"] is None else digest(state["schedule"]),
        "bindings_authenticity": "CALLER_DECLARED", "scheduled_episodes": 8,
        "scheduled_generation_calls": 16, "episodes": state["episodes"], "counts": counts,
        "by_arm": by_arm, "costs": costs,
        "calibration": calibration,
        "failure": state["failure"], "persistence_errors": state["persistence_errors"],
        "record_cleanup_before_terminal": state["record_cleanup"],
        "record_cleanup_complete_before_terminal": _record_cleanup_complete(state),
        "boundary_authenticity": "NOT_ATTESTED", **dict.fromkeys(FLAGS, False),
    })


def _record_cleanup_complete(state):
    if state["record_cleanup"]["failed_count"]:
        return False
    values = [
        row["record_cleanup"] for episode in state["episodes"] for row in episode["generations"]
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
        report["persistence_errors"] = _clone(state["persistence_errors"])
        report["record_cleanup_before_terminal"] = _clone(state["record_cleanup"])
        report["record_cleanup_complete_before_terminal"] = _record_cleanup_complete(state)
    return {
        "report": report, "terminal_path": terminal,
        "terminal_sha256": None if terminal is None else digest(report),
        "record_cleanup": _clone(state["record_cleanup"]),
        "record_cleanup_complete": _record_cleanup_complete(state),
        "persistence_errors": _clone(state["persistence_errors"]),
    }
