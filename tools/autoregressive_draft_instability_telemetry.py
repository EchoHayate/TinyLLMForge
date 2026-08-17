#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import copy
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import statistics
import sys


EXPECTED_GPU_INDICES = (3, 4, 6, 7)
MINIMUM_SAMPLES_PER_REPEAT_GPU = 5
GPU_EDGE_ALLOWANCE_SECONDS = 0.6
GPU_EDGE_ALLOWANCE_NS = int(GPU_EDGE_ALLOWANCE_SECONDS * 1e9)
GPU_FIELD_COUNT = 13
SCHEMA_VERSION = 1
LIMITATIONS = (
    "telemetry correlation is not causal proof",
    "wall-clock buckets are not GPU kernel durations",
    "host telemetry is retained as hash-bound raw evidence",
    "campaign does not establish long-context performance",
    "campaign does not establish Proposal-KV offload benefit",
    "campaign does not establish Phase-1 promotion",
)
SOURCE_FILE_PATHS = (
    "tools/autoregressive_draft_performance_worker.py",
    "tools/autoregressive_draft_instability_telemetry.py",
    "tools/verify_autoregressive_draft_instability_telemetry.py",
    "tools/autoregressive_draft_b4_timing_diagnostic.py",
    "tools/autoregressive_draft_performance_gate.py",
)


def _parse_int(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is invalid") from error
    return parsed


def _parse_float(value: str, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is invalid") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{name} is invalid")
    return parsed


def parse_gpu_telemetry(text: str) -> list[dict]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("GPU telemetry is empty")
    rows = []
    seen = set()
    reader = csv.reader(io.StringIO(text), skipinitialspace=True)
    for line_number, values in enumerate(reader, start=1):
        if len(values) != GPU_FIELD_COUNT:
            raise ValueError(
                f"GPU telemetry field count is invalid at line "
                f"{line_number}"
            )
        values = [value.strip() for value in values]
        sampled_at_unix_ns = _parse_int(
            values[0],
            name="sampled_at_unix_ns",
        )
        if sampled_at_unix_ns <= 0:
            raise ValueError("sampled_at_unix_ns is invalid")
        try:
            datetime.strptime(
                values[1],
                "%Y/%m/%d %H:%M:%S.%f",
            )
        except ValueError as error:
            raise ValueError(
                "nvidia timestamp is invalid"
            ) from error
        gpu_index = _parse_int(values[2], name="gpu index")
        uuid = values[3]
        if not uuid.startswith("GPU-"):
            raise ValueError("GPU uuid is invalid")
        pstate = values[4]
        if not pstate.startswith("P") or not pstate[1:].isdigit():
            raise ValueError("GPU pstate is invalid")
        sm_clock_mhz = _parse_int(
            values[5],
            name="SM clock",
        )
        memory_clock_mhz = _parse_int(
            values[6],
            name="memory clock",
        )
        power_w = _parse_float(values[7], name="power")
        temperature_c = _parse_int(
            values[8],
            name="temperature",
        )
        gpu_utilization_percent = _parse_int(
            values[9],
            name="GPU utilization",
        )
        memory_utilization_percent = _parse_int(
            values[10],
            name="memory utilization",
        )
        memory_used_mib = _parse_int(
            values[11],
            name="memory used",
        )
        try:
            throttle_reasons_active = int(values[12], 0)
        except ValueError as error:
            raise ValueError(
                "throttle reasons are invalid"
            ) from error
        for name, value in (
            ("SM clock", sm_clock_mhz),
            ("memory clock", memory_clock_mhz),
            ("temperature", temperature_c),
            ("memory used", memory_used_mib),
            ("throttle reasons", throttle_reasons_active),
        ):
            if value < 0:
                raise ValueError(f"{name} is invalid")
        if power_w < 0.0:
            raise ValueError("power is invalid")
        for name, value in (
            ("GPU utilization", gpu_utilization_percent),
            (
                "memory utilization",
                memory_utilization_percent,
            ),
        ):
            if value < 0 or value > 100:
                raise ValueError(f"{name} is invalid")
        key = (sampled_at_unix_ns, gpu_index)
        if key in seen:
            raise ValueError("GPU telemetry contains duplicate rows")
        seen.add(key)
        rows.append({
            "sampled_at_unix_ns": sampled_at_unix_ns,
            "nvidia_timestamp": values[1],
            "gpu_index": gpu_index,
            "uuid": uuid,
            "pstate": pstate,
            "sm_clock_mhz": sm_clock_mhz,
            "memory_clock_mhz": memory_clock_mhz,
            "power_w": power_w,
            "temperature_c": temperature_c,
            "gpu_utilization_percent": (
                gpu_utilization_percent
            ),
            "memory_utilization_percent": (
                memory_utilization_percent
            ),
            "memory_used_mib": memory_used_mib,
            "throttle_reasons_active": (
                throttle_reasons_active
            ),
        })
    rows.sort(
        key=lambda row: (
            row["sampled_at_unix_ns"],
            row["gpu_index"],
        )
    )
    return rows


def _normalize_interval(run: dict) -> dict:
    interval = run.get("campaign_interval")
    if not isinstance(interval, dict):
        raise ValueError("campaign interval is missing")
    started_at_unix_ns = interval.get("started_at_unix_ns")
    finished_at_unix_ns = interval.get("finished_at_unix_ns")
    if (
        isinstance(started_at_unix_ns, bool)
        or not isinstance(started_at_unix_ns, int)
        or isinstance(finished_at_unix_ns, bool)
        or not isinstance(finished_at_unix_ns, int)
        or started_at_unix_ns <= 0
        or finished_at_unix_ns <= started_at_unix_ns
    ):
        raise ValueError("campaign interval is invalid")
    return {
        "started_at_unix_ns": started_at_unix_ns,
        "finished_at_unix_ns": finished_at_unix_ns,
    }


def validate_campaign_intervals(worker: dict) -> None:
    if not isinstance(worker, dict):
        raise ValueError("worker artifact is invalid")
    runs = []
    for key in ("warmup_runs", "measured_runs"):
        rows = worker.get(key)
        if not isinstance(rows, list):
            raise ValueError(f"{key} is invalid")
        runs.extend(rows)
    previous_finish = None
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("worker run is invalid")
        interval = _normalize_interval(run)
        if (
            previous_finish is not None
            and interval["started_at_unix_ns"] <= previous_finish
        ):
            raise ValueError("campaign intervals overlap")
        previous_finish = interval["finished_at_unix_ns"]


def _numeric_summary(values: list[int | float]) -> dict:
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def _gpu_summary(
    samples: list[dict],
    *,
    gpu_index: int,
) -> dict:
    uuids = sorted({row["uuid"] for row in samples})
    if len(uuids) != 1:
        raise ValueError("GPU telemetry UUID changed")
    throttle_values = sorted({
        row["throttle_reasons_active"]
        for row in samples
    })
    throttle_or = 0
    for value in throttle_values:
        throttle_or |= value
    return {
        "gpu_index": gpu_index,
        "uuid": uuids[0],
        "sample_count": len(samples),
        "sm_clock_mhz": _numeric_summary([
            row["sm_clock_mhz"] for row in samples
        ]),
        "memory_clock_mhz": _numeric_summary([
            row["memory_clock_mhz"] for row in samples
        ]),
        "power_w": _numeric_summary([
            row["power_w"] for row in samples
        ]),
        "temperature_c": _numeric_summary([
            row["temperature_c"] for row in samples
        ]),
        "gpu_utilization_percent": _numeric_summary([
            row["gpu_utilization_percent"]
            for row in samples
        ]),
        "memory_utilization_percent": _numeric_summary([
            row["memory_utilization_percent"]
            for row in samples
        ]),
        "memory_used_mib": _numeric_summary([
            row["memory_used_mib"] for row in samples
        ]),
        "pstates": sorted({
            row["pstate"] for row in samples
        }),
        "throttle_reasons_active_values": throttle_values,
        "throttle_reasons_active_or": throttle_or,
    }


def summarize_gpu_telemetry(
    worker: dict,
    samples: list[dict],
    *,
    expected_gpu_indices: tuple[int, ...] = (
        EXPECTED_GPU_INDICES
    ),
    minimum_samples: int = (
        MINIMUM_SAMPLES_PER_REPEAT_GPU
    ),
) -> dict:
    validate_campaign_intervals(worker)
    if (
        not isinstance(expected_gpu_indices, tuple)
        or not expected_gpu_indices
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in expected_gpu_indices
        )
        or len(set(expected_gpu_indices))
        != len(expected_gpu_indices)
    ):
        raise ValueError("expected GPU indices are invalid")
    if (
        isinstance(minimum_samples, bool)
        or not isinstance(minimum_samples, int)
        or minimum_samples <= 0
    ):
        raise ValueError("minimum samples is invalid")
    if not isinstance(samples, list):
        raise ValueError("GPU telemetry samples are invalid")
    measured = []
    for run in worker["measured_runs"]:
        interval = _normalize_interval(run)
        gpu_rows = []
        for gpu_index in expected_gpu_indices:
            gpu_samples = sorted(
                (
                    row for row in samples
                    if row.get("gpu_index") == gpu_index
                ),
                key=lambda row: row.get("sampled_at_unix_ns", -1),
            )
            matching = [
                row
                for row in gpu_samples
                if interval["started_at_unix_ns"]
                <= row.get("sampled_at_unix_ns", -1)
                <= interval["finished_at_unix_ns"]
            ]
            if len(matching) < minimum_samples:
                before = [
                    row for row in gpu_samples
                    if row["sampled_at_unix_ns"]
                    < interval["started_at_unix_ns"]
                ]
                if (
                    before
                    and interval["started_at_unix_ns"]
                    - before[-1]["sampled_at_unix_ns"]
                    <= GPU_EDGE_ALLOWANCE_NS
                ):
                    matching.insert(0, before[-1])
            if len(matching) < minimum_samples:
                after = [
                    row for row in gpu_samples
                    if row["sampled_at_unix_ns"]
                    > interval["finished_at_unix_ns"]
                ]
                if (
                    after
                    and after[0]["sampled_at_unix_ns"]
                    - interval["finished_at_unix_ns"]
                    <= GPU_EDGE_ALLOWANCE_NS
                ):
                    matching.append(after[0])
            if len(matching) < minimum_samples:
                raise ValueError(
                    "insufficient GPU telemetry coverage"
                )
            gpu_rows.append(
                _gpu_summary(
                    matching,
                    gpu_index=gpu_index,
                )
            )
        measured.append({
            "repeat": run.get("repeat"),
            "campaign_interval": interval,
            "gpus": gpu_rows,
        })
    return {
        "expected_gpu_indices": list(expected_gpu_indices),
        "minimum_samples_per_repeat_gpu": minimum_samples,
        "edge_allowance_seconds": GPU_EDGE_ALLOWANCE_SECONDS,
        "measured_runs": measured,
    }


def _validate_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(f"{name} digest is invalid")
    return value


def _validate_relative_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} path is unsafe")
    return value


def _validate_source_files(source_files: object) -> dict[str, str]:
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("telemetry source files must be non-empty")
    normalized = {}
    for path, digest in source_files.items():
        normalized[_validate_relative_path(
            path,
            name="source",
        )] = _validate_digest(digest, name="source")
    return normalized


def _validate_host_files(host_files: object) -> dict[str, dict]:
    if not isinstance(host_files, dict) or not host_files:
        raise ValueError("telemetry host files must be non-empty")
    normalized = {}
    for name, row in host_files.items():
        if not isinstance(name, str) or not name:
            raise ValueError("telemetry host file name is invalid")
        if not isinstance(row, dict):
            raise ValueError("telemetry host file row is invalid")
        normalized[name] = {
            "path": _validate_relative_path(
                row.get("path"),
                name="host file",
            ),
            "sha256": _validate_digest(
                row.get("sha256"),
                name="host file",
            ),
        }
    return normalized


def _worker_interval_view(worker: dict) -> dict:
    validate_campaign_intervals(worker)
    view = {}
    for key in ("warmup_runs", "measured_runs"):
        view[key] = [
            {
                "repeat": run.get("repeat"),
                "campaign_interval": _normalize_interval(run),
            }
            for run in worker[key]
        ]
    return view


def _environment_reasons(policies: dict) -> list[str]:
    reasons = []
    for policy in ("target", "learned"):
        measured_runs = policies[policy]["summary"][
            "measured_runs"
        ]
        by_gpu = {}
        for run in measured_runs:
            repeat = run["repeat"]
            for gpu in run["gpus"]:
                gpu_index = gpu["gpu_index"]
                by_gpu.setdefault(gpu_index, []).append(gpu)
                throttle_mask = gpu[
                    "throttle_reasons_active_or"
                ]
                if throttle_mask:
                    reasons.append(
                        f"{policy} repeat {repeat} GPU "
                        f"{gpu_index} active throttle mask "
                        f"{throttle_mask:#x}"
                    )
                if len(gpu["pstates"]) > 1:
                    reasons.append(
                        f"{policy} repeat {repeat} GPU "
                        f"{gpu_index} multiple P-states "
                        f"{','.join(gpu['pstates'])}"
                    )
        for gpu_index, rows in sorted(by_gpu.items()):
            maximum_clocks = [
                row["sm_clock_mhz"]["maximum"]
                for row in rows
            ]
            reference_clock = statistics.median(
                maximum_clocks
            )
            minimum_clock = min(
                row["sm_clock_mhz"]["minimum"]
                for row in rows
            )
            if minimum_clock < reference_clock * 0.95:
                reasons.append(
                    f"{policy} GPU {gpu_index} SM clock "
                    f"dropped below 95% reference"
                )
            minimum_temperature = min(
                row["temperature_c"]["minimum"]
                for row in rows
            )
            maximum_temperature = max(
                row["temperature_c"]["maximum"]
                for row in rows
            )
            if (
                maximum_temperature - minimum_temperature
                >= 10
            ):
                reasons.append(
                    f"{policy} GPU {gpu_index} temperature "
                    f"range reached 10 C"
                )
    return reasons


def build_instability_telemetry_artifact(
    *,
    timing_artifact: dict,
    target_worker: dict,
    learned_worker: dict,
    target_gpu_samples: list[dict],
    learned_gpu_samples: list[dict],
    source_files: dict[str, str],
    host_files: dict[str, dict],
) -> dict:
    if not isinstance(timing_artifact, dict):
        raise ValueError("timing artifact is invalid")
    if timing_artifact.get("status") != "PASS":
        raise ValueError("timing artifact status is invalid")
    timing_classification = timing_artifact.get(
        "classification"
    )
    if timing_classification not in ("STABLE", "UNSTABLE"):
        raise ValueError("timing classification is invalid")
    if timing_artifact.get("exact_parity") is not True:
        raise ValueError("timing exact parity is invalid")
    policies = {
        "target": {
            "worker": _worker_interval_view(target_worker),
            "gpu_samples": copy.deepcopy(target_gpu_samples),
            "summary": summarize_gpu_telemetry(
                target_worker,
                target_gpu_samples,
            ),
        },
        "learned": {
            "worker": _worker_interval_view(learned_worker),
            "gpu_samples": copy.deepcopy(learned_gpu_samples),
            "summary": summarize_gpu_telemetry(
                learned_worker,
                learned_gpu_samples,
            ),
        },
    }
    reasons = _environment_reasons(policies)
    if timing_classification == "STABLE":
        telemetry_classification = "STABLE_BASELINE"
    elif reasons:
        telemetry_classification = "ENVIRONMENT_CORRELATED"
    else:
        telemetry_classification = (
            "RUNTIME_VARIANCE_SUSPECTED"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "timing_classification": timing_classification,
        "telemetry_classification": (
            telemetry_classification
        ),
        "exact_parity": True,
        "classification_reasons": reasons,
        "policies": policies,
        "host_files": _validate_host_files(host_files),
        "source_files": _validate_source_files(source_files),
        "limitations": list(LIMITATIONS),
    }


def validate_instability_telemetry_artifact(
    artifact: object,
) -> dict:
    if not isinstance(artifact, dict):
        raise ValueError("telemetry artifact must be a mapping")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("telemetry schema version mismatch")
    policies = artifact.get("policies")
    if not isinstance(policies, dict):
        raise ValueError("telemetry policies are invalid")
    expected = build_instability_telemetry_artifact(
        timing_artifact={
            "status": artifact.get("status"),
            "classification": artifact.get(
                "timing_classification"
            ),
            "exact_parity": artifact.get("exact_parity"),
        },
        target_worker=policies.get("target", {}).get(
            "worker"
        ),
        learned_worker=policies.get("learned", {}).get(
            "worker"
        ),
        target_gpu_samples=policies.get("target", {}).get(
            "gpu_samples"
        ),
        learned_gpu_samples=policies.get(
            "learned",
            {},
        ).get("gpu_samples"),
        source_files=artifact.get("source_files"),
        host_files=artifact.get("host_files"),
    )
    if json.dumps(
        artifact,
        sort_keys=True,
        separators=(",", ":"),
    ) != json.dumps(
        expected,
        sort_keys=True,
        separators=(",", ":"),
    ):
        raise ValueError(
            "telemetry artifact recomputation mismatch"
        )
    return {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "timing_classification": expected[
            "timing_classification"
        ],
        "telemetry_classification": expected[
            "telemetry_classification"
        ],
        "exact_parity": True,
    }


def _read_json(path: Path, *, name: str) -> dict:
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unreadable") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a mapping")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes(repo_root: Path) -> dict[str, str]:
    hashes = {}
    for relative_path in SOURCE_FILE_PATHS:
        path = Path(repo_root) / relative_path
        if not path.is_file():
            raise ValueError(
                f"source file is missing: {relative_path}"
            )
        hashes[relative_path] = _sha256(path)
    return hashes


def _host_hashes(
    rows: list[str],
    *,
    artifact_root: Path,
) -> dict[str, dict]:
    if not rows:
        raise ValueError("at least one host file is required")
    result = {}
    for row in rows:
        if not isinstance(row, str) or "=" not in row:
            raise ValueError("host file argument is invalid")
        name, relative_path = row.split("=", 1)
        if not name or name in result:
            raise ValueError("host file name is invalid")
        normalized_path = _validate_relative_path(
            relative_path,
            name="host file",
        )
        path = artifact_root / normalized_path
        if not path.is_file():
            raise ValueError(
                f"host file is missing: {normalized_path}"
            )
        result[name] = {
            "path": normalized_path,
            "sha256": _sha256(path),
        }
    return result


def _write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-artifact", required=True)
    parser.add_argument("--target-worker", required=True)
    parser.add_argument("--learned-worker", required=True)
    parser.add_argument("--target-gpu-csv", required=True)
    parser.add_argument("--learned-gpu-csv", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument(
        "--host-file",
        action="append",
        default=[],
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_path = Path(args.out)
    artifact = build_instability_telemetry_artifact(
        timing_artifact=_read_json(
            Path(args.timing_artifact),
            name="timing artifact",
        ),
        target_worker=_read_json(
            Path(args.target_worker),
            name="target worker",
        ),
        learned_worker=_read_json(
            Path(args.learned_worker),
            name="learned worker",
        ),
        target_gpu_samples=parse_gpu_telemetry(
            Path(args.target_gpu_csv).read_text(
                encoding="utf-8"
            )
        ),
        learned_gpu_samples=parse_gpu_telemetry(
            Path(args.learned_gpu_csv).read_text(
                encoding="utf-8"
            )
        ),
        source_files=_source_hashes(Path(args.repo_root)),
        host_files=_host_hashes(
            args.host_file,
            artifact_root=output_path.parent,
        ),
    )
    _write_json_atomic(output_path, artifact)
    return 0


if __name__ == "__main__":
    sys.exit(main())
