import copy
import hashlib
import os
from pathlib import Path
import re
import tempfile
import time

import generation_core
from pilot_protocol import require
from validation_contract import digest, encoded


__all__ = ("generate_observed", "GenerationAttemptError")


class GenerationAttemptError(RuntimeError):
    def __init__(self, message, *, observation_error=None, record_cleanup=None):
        super().__init__(message)
        self.observation_error = observation_error
        self.record_cleanup = record_cleanup


def _write(descriptor, record):
    try:
        output = os.fdopen(descriptor, "wb")
    except BaseException:
        os.close(descriptor)
        raise
    with output:
        raw = encoded(record)
        if output.write(raw) != len(raw):
            raise OSError("incomplete generation record write")


def _save(root, name, record, *, cleanup):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=root)
    published = False
    try:
        _write(descriptor, record)
        os.link(temporary, root / name)
        published = True
    finally:
        try:
            os.unlink(temporary)
        except BaseException as error:
            cleanup["failures"].append({
                "record": name, "temporary": Path(temporary).name,
                "published": published, "error": _failure(error),
            })
            cleanup["failed_count"] = len(cleanup["failures"])


def _failure(error):
    try:
        message = str(error)[:1024]
    except BaseException:
        message = "<exception message unavailable>"
    return {"type": type(error).__name__[:128], "message": message}


def _validate(generation, snapshot, messages_sha256, expected_sha256,
              forward_started, forward_completed):
    require(type(generation) is dict, "generation record")
    require(generation["kind"] == "LORA_GREEDY_CANDIDATE_UNATTESTED", "generation kind")
    require(digest(snapshot) == messages_sha256
            and generation["messages_sha256"] == messages_sha256, "messages changed")
    require(generation["tokenizer_state_sha256"] == expected_sha256,
            "tokenizer identity mismatch")
    candidate = generation["candidate"]
    require(type(candidate) is str, "candidate text")
    candidate_bytes = candidate.encode("utf-8")
    require(len(candidate_bytes) <= 65536
            and hashlib.sha256(candidate_bytes).hexdigest() == generation["candidate_sha256"],
            "candidate text identity")
    require(type(generation["model_calls"]) is int and generation["model_calls"] == 1,
            "generation model call count")
    output_tokens = generation["output_tokens"]
    require(type(output_tokens) is int and 1 <= output_tokens <= 256,
            "generation token count")
    require(type(generation["forward_calls"]) is int
            and generation["forward_calls"] == output_tokens
            and forward_started == output_tokens and forward_completed == output_tokens,
            "observed forward count mismatch")
    require(type(generation["output_ids"]) is list
            and len(generation["output_ids"]) == output_tokens
            and all(type(token) is int and token >= 0 for token in generation["output_ids"]),
            "generation output token identities")


def generate_observed(model, tokenizer, messages, *, expected_tokenizer_sha256,
                      call_id, artifact_dir):
    started = time.perf_counter_ns()
    forward_started = 0
    forward_completed = 0
    handles = []
    cleanup_errors = []
    record_cleanup = {"failed_count": 0, "failures": []}
    failure = None
    claimed = False

    def entered(module, arguments):
        nonlocal forward_started
        forward_started += 1

    def completed(module, arguments, output):
        nonlocal forward_completed
        forward_completed += 1

    def observation(status, error=None):
        details = None if error is None else _failure(error)
        additional = [_failure(item) for item in cleanup_errors if item is not error]
        if additional:
            details["cleanup_errors"] = additional
        return {
            "kind": "LORA_GENERATION_OBSERVATION", "call_id": call_id, "status": status,
            "messages_sha256": messages_sha256, "forward_started": forward_started,
            "forward_completed": forward_completed,
            "elapsed_ns": time.perf_counter_ns() - started, "failure": details,
        }

    try:
        require(type(call_id) is str and re.fullmatch("[0-9a-f]{32}", call_id),
                "invalid generation call identity")
        require(type(expected_tokenizer_sha256) is str
                and re.fullmatch("[0-9a-f]{64}", expected_tokenizer_sha256),
                "invalid expected tokenizer SHA256")
        root = Path(artifact_dir)
        require(root.is_dir(), "existing artifact directory required")
        snapshot = copy.deepcopy(messages)
        messages_sha256 = digest(snapshot)
        claim = {
            "kind": "LORA_GENERATION_CLAIM", "call_id": call_id,
            "messages_sha256": messages_sha256,
            "expected_tokenizer_sha256": expected_tokenizer_sha256,
        }
        descriptor = os.open(root / "claim.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                             0o600)
        claimed = True
        _write(descriptor, claim)
        handles.append(model.register_forward_pre_hook(entered))
        handles.append(model.register_forward_hook(completed, prepend=True))
        generation = generation_core.generate_candidate(
            model, tokenizer, snapshot, expected_tokenizer_sha256=expected_tokenizer_sha256)
        _validate(generation, snapshot, messages_sha256, expected_tokenizer_sha256,
                  forward_started, forward_completed)
    except BaseException as error:
        failure = error
    finally:
        for handle in reversed(handles):
            try:
                handle.remove()
            except BaseException as error:
                cleanup_errors.append(error)
        if failure is None and cleanup_errors:
            failure = cleanup_errors[0]

    if failure is None:
        try:
            _save(root, "generation.json", generation, cleanup=record_cleanup)
            observed = observation("completed")
            _save(root, "observation.json", observed, cleanup=record_cleanup)
        except BaseException as error:
            failure = error
        else:
            return {"generation": generation, "observation": observed,
                    "record_cleanup": record_cleanup}

    if not claimed:
        raise GenerationAttemptError(
            "generation attempt not claimed", record_cleanup=record_cleanup,
        ) from failure
    try:
        _save(root, "observation.json", observation("failed", failure),
              cleanup=record_cleanup)
    except BaseException as observation_error:
        raise GenerationAttemptError(
            "generation attempt failed; failed observation could not be persisted",
            observation_error=observation_error, record_cleanup=record_cleanup,
        ) from failure
    raise GenerationAttemptError(
        "generation attempt failed; failed observation saved", record_cleanup=record_cleanup,
    ) from failure
