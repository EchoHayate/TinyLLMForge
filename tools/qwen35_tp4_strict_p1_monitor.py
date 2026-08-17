from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time


SCHEMA_VERSION = "qwen35.tp4-strict-p1-monitor.v1"
MINIMUM_INTERVAL_S = 60
MAX_ERROR_CHARS = 4096


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path, payload):
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run_cleanup(cleanup_fn):
    try:
        cleanup = cleanup_fn()
    except BaseException as error:
        return {
            "classification": "CLEANUP_FAILED",
            "error": str(error)[-MAX_ERROR_CHARS:],
        }
    if not isinstance(cleanup, dict):
        return {
            "classification": "CLEANUP_FAILED",
            "error": "cleanup result is invalid",
        }
    return cleanup


def resumable_sample_count(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise ValueError("monitor output directory is not resumable")
    if any(
        (output_dir / name).exists()
        for name in ("monitor_result.json", "monitor_failure.json")
    ):
        raise ValueError("completed monitor output is not resumable")
    samples_path = output_dir / "resource_samples.jsonl"
    if not samples_path.is_file():
        raise ValueError("monitor sample ledger is missing")
    expected_sample_id = 1
    with samples_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "monitor sample ledger is invalid"
                ) from error
            if (
                not isinstance(sample, dict)
                or sample.get("sample_id") != expected_sample_id
            ):
                raise ValueError("monitor sample ledger is invalid")
            expected_sample_id += 1
    return expected_sample_id - 1


def _validate_configuration(
    *,
    monitor_tag,
    output_dir,
    sample_fn,
    launch_fn,
    cleanup_fn,
    sleep_fn,
    interval_s,
    required_ready_samples,
    max_samples,
    resume_existing,
):
    if (
        not isinstance(monitor_tag, str)
        or not monitor_tag
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in monitor_tag
        )
    ):
        raise ValueError("monitor tag is invalid")
    if not isinstance(output_dir, Path):
        raise ValueError("monitor output directory must be a Path")
    if any(
        not callable(value)
        for value in (sample_fn, launch_fn, cleanup_fn, sleep_fn)
    ):
        raise ValueError("monitor callbacks are invalid")
    if (
        isinstance(interval_s, bool)
        or not isinstance(interval_s, (int, float))
        or interval_s < MINIMUM_INTERVAL_S
    ):
        raise ValueError("monitor interval must be at least 60 seconds")
    if (
        isinstance(required_ready_samples, bool)
        or not isinstance(required_ready_samples, int)
        or required_ready_samples <= 0
    ):
        raise ValueError("required READY sample count is invalid")
    if (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, int)
        or max_samples <= 0
    ):
        raise ValueError("monitor sample budget is invalid")
    if not isinstance(resume_existing, bool):
        raise ValueError("monitor resume policy is invalid")


def monitor_until_launch(
    *,
    monitor_tag,
    output_dir,
    sample_fn,
    launch_fn,
    cleanup_fn,
    sleep_fn=time.sleep,
    interval_s=60,
    required_ready_samples=2,
    max_samples=1440,
    resume_existing=False,
):
    output_dir = Path(output_dir)
    _validate_configuration(
        monitor_tag=monitor_tag,
        output_dir=output_dir,
        sample_fn=sample_fn,
        launch_fn=launch_fn,
        cleanup_fn=cleanup_fn,
        sleep_fn=sleep_fn,
        interval_s=interval_s,
        required_ready_samples=required_ready_samples,
        max_samples=max_samples,
        resume_existing=resume_existing,
    )
    if output_dir.exists():
        if not resume_existing:
            raise ValueError("monitor output directory already exists")
        existing_sample_count = resumable_sample_count(output_dir)
    else:
        output_dir.mkdir(parents=True)
        existing_sample_count = 0
    if existing_sample_count >= max_samples:
        raise ValueError("monitor sample budget is exhausted")
    samples_path = output_dir / "resource_samples.jsonl"
    ready_samples = []
    for sample_index in range(existing_sample_count, max_samples):
        try:
            sample = sample_fn()
        except Exception as error:
            sample = {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_index + 1,
                "classification": "SAMPLE_FAILED",
                "error": str(error)[-MAX_ERROR_CHARS:],
            }
        if not isinstance(sample, dict):
            raise ValueError("resource sample is invalid")
        _append_jsonl(samples_path, sample)
        if sample.get("classification") == "READY":
            ready_samples.append(sample)
            ready_samples = ready_samples[-required_ready_samples:]
        else:
            ready_samples = []
        if len(ready_samples) == required_ready_samples:
            trigger_ids = [
                row.get("sample_id") for row in ready_samples
            ]
            try:
                launch_result = launch_fn()
            except BaseException as error:
                cleanup = _run_cleanup(cleanup_fn)
                failure = {
                    "schema_version": SCHEMA_VERSION,
                    "monitor_tag": monitor_tag,
                    "classification": "FAILED",
                    "error": str(error)[-MAX_ERROR_CHARS:],
                    "trigger_sample_ids": trigger_ids,
                    "cleanup": cleanup,
                }
                _atomic_write_json(
                    output_dir / "monitor_failure.json",
                    failure,
                )
                raise
            cleanup = _run_cleanup(cleanup_fn)
            if launch_result.get("classification") == (
                "BLOCKED_RESOURCES"
            ):
                if cleanup.get("classification") != "CLEAN":
                    result = {
                        "schema_version": SCHEMA_VERSION,
                        "monitor_tag": monitor_tag,
                        "classification": "CLEANUP_FAILED",
                        "sample_count": sample_index + 1,
                        "trigger_sample_ids": trigger_ids,
                        "launch_result": launch_result,
                        "cleanup": cleanup,
                    }
                    _atomic_write_json(
                        output_dir / "monitor_result.json",
                        result,
                    )
                    return result
                _append_jsonl(
                    output_dir / "launch_attempts.jsonl",
                    {
                        "classification": "BLOCKED_RESOURCES",
                        "trigger_sample_ids": trigger_ids,
                        "launch_result": launch_result,
                        "cleanup": cleanup,
                    },
                )
                ready_samples = []
                if sample_index + 1 < max_samples:
                    sleep_fn(interval_s)
                    continue
                break
            classification = launch_result.get(
                "classification",
                "UNKNOWN",
            )
            if cleanup.get("classification") != "CLEAN":
                classification = "CLEANUP_FAILED"
            result = {
                "schema_version": SCHEMA_VERSION,
                "monitor_tag": monitor_tag,
                "classification": classification,
                "sample_count": sample_index + 1,
                "trigger_sample_ids": trigger_ids,
                "launch_result": launch_result,
                "cleanup": cleanup,
            }
            _atomic_write_json(
                output_dir / "monitor_result.json",
                result,
            )
            return result
        if sample_index + 1 < max_samples:
            sleep_fn(interval_s)
    result = {
        "schema_version": SCHEMA_VERSION,
        "monitor_tag": monitor_tag,
        "classification": "MONITOR_EXPIRED",
        "sample_count": max_samples,
        "trigger_sample_ids": [],
    }
    _atomic_write_json(output_dir / "monitor_result.json", result)
    return result
