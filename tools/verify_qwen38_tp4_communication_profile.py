#!/usr/bin/env python3
"""Independently verify a Qwen3.8-27B TP4 communication profile bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import statistics
import tempfile

if __package__:
    from tools.qwen38_nsys_intervals import parse_nsys_sqlite
else:
    from qwen38_nsys_intervals import parse_nsys_sqlite


VERIFICATION_SCHEMA = (
    "qwen38.tp4-communication-profile-independent-verification.v1"
)
MANIFEST_SCHEMA = "qwen38.tp4-communication-profile-manifest.v1"
PROFILE_SCHEMA = "qwen38.communication-profile-row.v1"
SUMMARY_SCHEMA = "qwen38.communication-exposure-summary.v1"
SOURCE_SCHEMA = "tinyllmforge.source-manifest.v1"
MODEL_SCHEMA = "tinyllmforge.qwen38-model-manifest.v1"
ENVIRONMENT_SCHEMA = "qwen38.tp4-profile-environment.v1"
TOPOLOGY_SCHEMA = "qwen38.tp4-profile-topology.v1"
WORKLOAD_SCHEMA = "qwen38.tp4-profile-workloads.v1"
LAYER_SUMMARY_SCHEMA = "qwen38.layer-summary.v1"
MODEL_REPOSITORY = "Qwen/Qwen3.8-27B"
APPROVED_REMOTE_ROOT = PurePosixPath(
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
WORKLOADS = {
    "P0": {
        "workload_family": "causal",
        "prompt_tokens": 256,
        "output_tokens": 128,
        "concurrency": 1,
    },
    "P1": {
        "workload_family": "causal",
        "prompt_tokens": 2048,
        "output_tokens": 128,
        "concurrency": 1,
    },
    "Q0": {
        "workload_family": "online",
        "prompt_tokens": 256,
        "output_tokens": 128,
        "concurrency": 4,
    },
    "Q1": {
        "workload_family": "online",
        "prompt_tokens": 256,
        "output_tokens": 128,
        "concurrency": 8,
    },
    "Q2": {
        "workload_family": "online",
        "prompt_tokens": 2048,
        "output_tokens": 128,
        "concurrency": 4,
    },
}
RANKS = (0, 1, 2, 3)
WARMUP_REPETITIONS = (0, 1)
MEASURED_REPETITIONS = (0, 1, 2, 3, 4)
MAX_PROFILER_OVERHEAD_RATIO = 0.03
GO_EXPOSURE_RATIO = 0.10
GO_HEADROOM_RATIO = 0.05
NO_GO_EXPOSURE_RATIO = 0.05
NO_GO_HEADROOM_RATIO = 0.02
METRIC_TOLERANCE = 1e-12
LAYER_METRICS = (
    "gemm_ns",
    "collective_ns",
    "compute_ns",
    "exposed_collective_ns",
    "compute_collective_overlap_ns",
    "gpu_idle_ns",
    "collective_count",
    "collective_bytes",
    "critical_path_ns",
)
BASE_FILES = {
    "source_manifest.json",
    "model_manifest.json",
    "environment.json",
    "gpu_topology.json",
    "workload_manifest.json",
    "correctness_rows.jsonl",
    "profile_rows.jsonl",
    "layer_summary.json",
    "communication_exposure_summary.json",
    "online_metrics.json",
    "memory_summary.json",
    "resource_samples.jsonl",
    "report.md",
}


def _expected_worker_ids() -> list[str]:
    result = ["correctness"]
    for workload in WORKLOADS:
        for repetition in WARMUP_REPETITIONS:
            result.append(f"{workload}__warmup__r{repetition}")
        for phase in ("measured", "nsys_replay"):
            for repetition in MEASURED_REPETITIONS:
                result.append(
                    f"{workload}__{phase}__r{repetition}"
                )
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"artifact is missing: {path.name}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"artifact is missing: {path.name}")
    with path.open("r", encoding="utf-8") as handle:
        return [
            json.loads(
                line,
                parse_constant=_reject_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
            for line in handle
            if line.strip()
        ]


def _write_json_atomic(path: Path, payload) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload) -> str:
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value, name: str, lengths=(64,)) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")
    return value


def _integer(value, name: str, *, minimum=0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _number(value, name: str, *, minimum=0.0, positive=False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite and numeric")
    result = float(value)
    if positive and result <= minimum:
        raise ValueError(f"{name} must be greater than {minimum}")
    if not positive and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _string(value, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_clean_gpu_inventory(
    value,
    *,
    expected_indices: list[int],
    expected_uuids: list[str],
    label: str,
) -> None:
    if not isinstance(value, list) or len(value) != len(RANKS):
        raise ValueError(f"{label} strict-clean inventory mismatch")
    indices = []
    uuids = []
    for row in value:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "gpu_index",
                "gpu_uuid",
                "memory_used_mib",
                "utilization_percent",
                "compute_processes",
            }
        ):
            raise ValueError(f"{label} strict-clean inventory mismatch")
        indices.append(_integer(row["gpu_index"], "GPU index"))
        uuids.append(_string(row["gpu_uuid"], "GPU UUID"))
        if (
            _integer(
                row["memory_used_mib"],
                "GPU memory used",
            )
            > 1024
            or _integer(
                row["utilization_percent"],
                "GPU utilization",
            )
            > 5
            or row["compute_processes"] != []
        ):
            raise ValueError(f"{label} strict-clean admission failed")
    if indices != expected_indices or uuids != expected_uuids:
        raise ValueError(f"{label} strict-clean GPU identity mismatch")


def _percentile(values, percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile input is empty")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


def _actual_artifacts(root: Path) -> set[str]:
    artifacts = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink() or any(
            part.startswith(".") for part in relative.parts
        ):
            raise ValueError("artifact inventory contains unsafe path")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("artifact inventory contains unsafe path")
        relative_name = relative.as_posix()
        if relative_name == "manifest.sha256":
            continue
        artifacts.add(relative_name)
    return artifacts


def _expected_trace_names() -> set[str]:
    return {
        f"nsys/{workload}-r{repetition}.sqlite"
        for workload in WORKLOADS
        for repetition in MEASURED_REPETITIONS
    }


def _verify_manifest(root: Path) -> dict:
    manifest = _load_json(root / "manifest.sha256")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or not isinstance(manifest.get("artifacts"), dict)
    ):
        raise ValueError("manifest schema mismatch")
    actual = _actual_artifacts(root)
    allowed = BASE_FILES | _expected_trace_names() | {
        "independent_verification.json"
    }
    required = BASE_FILES | _expected_trace_names()
    if not required <= actual:
        raise ValueError(
            f"artifact is missing: {sorted(required - actual)}"
        )
    if not actual <= allowed:
        raise ValueError(
            f"artifact inventory contains extra files: "
            f"{sorted(actual - allowed)}"
        )
    recorded = set(manifest["artifacts"])
    if recorded != actual:
        raise ValueError("manifest artifact inventory mismatch")
    for relative, expected in manifest["artifacts"].items():
        _sha256(expected, f"manifest digest for {relative}")
        path = root / relative
        if path.resolve().parent != root.resolve() and (
            root.resolve() not in path.resolve().parents
        ):
            raise ValueError("manifest artifact path escapes bundle")
        if _sha256_file(path) != expected:
            raise ValueError(f"manifest digest mismatch for {relative}")
    return manifest


def _validate_identity(root: Path) -> dict:
    model = _load_json(root / "model_manifest.json")
    source = _load_json(root / "source_manifest.json")
    environment = _load_json(root / "environment.json")
    topology = _load_json(root / "gpu_topology.json")
    workload = _load_json(root / "workload_manifest.json")
    if (
        model.get("schema_version") != MODEL_SCHEMA
        or model.get("repository") != MODEL_REPOSITORY
    ):
        raise ValueError("model manifest identity mismatch")
    model_revision = _sha256(
        model.get("resolved_revision"),
        "model revision",
        lengths=(40,),
    )
    files = model.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("model manifest file inventory is missing")
    normalized_files = {}
    for relative, metadata in files.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or "\\" in relative
            or Path(relative).as_posix() != relative
            or not isinstance(metadata, dict)
            or set(metadata) != {"sha256", "size"}
        ):
            raise ValueError("model manifest file inventory mismatch")
        normalized_files[relative] = {
            "sha256": _sha256(
                metadata.get("sha256"),
                "model file SHA-256",
            ),
            "size": _integer(
                metadata.get("size"),
                "model file size",
                minimum=1,
            ),
        }
    model_root = model.get("model_root")
    model_root_path = (
        PurePosixPath(model_root)
        if isinstance(model_root, str)
        else None
    )
    if (
        model_root_path is None
        or not model_root
        or not model_root_path.is_absolute()
        or ".." in model_root_path.parts
        or model_root_path == APPROVED_REMOTE_ROOT
        or not model_root_path.is_relative_to(APPROVED_REMOTE_ROOT)
    ):
        raise ValueError(
            "model manifest model root must be below approved remote root"
        )
    config_sha256 = _sha256(
        model.get("config_sha256"),
        "model config SHA-256",
    )
    if (
        "config.json" not in normalized_files
        or normalized_files["config.json"]["sha256"] != config_sha256
    ):
        raise ValueError("model config inventory mismatch")
    _sha256(
        model.get("text_config_sha256"),
        "text config SHA-256",
    )
    checkpoint_index = model.get("checkpoint_index")
    if (
        checkpoint_index != "model.safetensors.index.json"
        or checkpoint_index not in normalized_files
        or normalized_files[checkpoint_index]["sha256"]
        != _sha256(
            model.get("checkpoint_index_sha256"),
            "checkpoint index SHA-256",
        )
    ):
        raise ValueError("model checkpoint index mismatch")
    _integer(
        model.get("checkpoint_tensor_count"),
        "checkpoint tensor count",
        minimum=1,
    )
    checkpoint_shards = model.get("checkpoint_shards")
    if (
        not isinstance(checkpoint_shards, list)
        or not checkpoint_shards
        or len(checkpoint_shards) != len(set(checkpoint_shards))
        or any(
            not isinstance(relative, str)
            or not relative.endswith(".safetensors")
            or relative not in normalized_files
            for relative in checkpoint_shards
        )
    ):
        raise ValueError("model checkpoint shard inventory mismatch")
    tokenizer_files = model.get("tokenizer_files")
    if (
        not isinstance(tokenizer_files, list)
        or not tokenizer_files
        or len(tokenizer_files) != len(set(tokenizer_files))
        or any(
            not isinstance(relative, str)
            or relative not in normalized_files
            for relative in tokenizer_files
        )
    ):
        raise ValueError("model tokenizer inventory mismatch")
    tokenizer_inventory_sha256 = _sha256(
        model.get("tokenizer_inventory_sha256"),
        "tokenizer inventory SHA-256",
    )
    if tokenizer_inventory_sha256 != _canonical_sha256({
        relative: normalized_files[relative]
        for relative in tokenizer_files
    }):
        raise ValueError("model tokenizer inventory digest mismatch")
    if source.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("source manifest schema mismatch")
    source_revision = _sha256(
        source.get("source_revision"),
        "source revision",
        lengths=(40, 64),
    )
    source_tree_sha256 = _sha256(
        source.get("source_tree_sha256"),
        "source tree SHA-256",
    )
    if source.get("model_manifest_sha256") != _sha256_file(
        root / "model_manifest.json"
    ):
        raise ValueError("model manifest SHA-256 mismatch")
    if (
        environment.get("schema_version") != ENVIRONMENT_SCHEMA
        or environment.get("source_revision") != source_revision
    ):
        raise ValueError("source revision mismatch")
    if environment.get("model_revision") != model_revision:
        raise ValueError("model revision mismatch")
    if (
        environment.get("dtype") != "bfloat16"
        or environment.get("tensor_parallel_size") != 4
        or environment.get("decoding") != "greedy"
        or environment.get("temperature") != 0.0
        or environment.get("fixed_output_tokens") != 128
        or environment.get("scheduler_policy") != "identical"
        or environment.get("cuda_graph_policy") != "identical"
    ):
        raise ValueError("environment contract mismatch")
    if workload.get("schema_version") != WORKLOAD_SCHEMA:
        raise ValueError("workload manifest schema mismatch")
    if (
        workload.get("source_revision") != source_revision
        or workload.get("model_revision") != model_revision
    ):
        raise ValueError("workload revision mismatch")
    if (
        workload.get("order") != list(WORKLOADS)
        or workload.get("warmup_repetitions")
        != list(WARMUP_REPETITIONS)
        or workload.get("measured_repetitions")
        != list(MEASURED_REPETITIONS)
        or workload.get("rank_inventory") != list(RANKS)
        or workload.get("workloads") != WORKLOADS
    ):
        raise ValueError("workload manifest inventory mismatch")
    if topology.get("schema_version") != TOPOLOGY_SCHEMA:
        raise ValueError("GPU topology schema mismatch")
    rank_mapping = topology.get("rank_mapping")
    if (
        not isinstance(rank_mapping, list)
        or len(rank_mapping) != 4
        or [row.get("rank") for row in rank_mapping] != list(RANKS)
    ):
        raise ValueError("rank inventory mismatch in GPU topology")
    gpu_uuids = []
    gpu_indices = []
    for rank, row in enumerate(rank_mapping):
        if not isinstance(row, dict):
            raise ValueError("GPU UUID mapping mismatch")
        gpu_uuids.append(_string(row.get("gpu_uuid"), "GPU UUID"))
        gpu_indices.append(_integer(row.get("gpu_index"), "GPU index"))
        if row.get("rank") != rank:
            raise ValueError("rank inventory mismatch in GPU topology")
    if len(set(gpu_uuids)) != 4 or len(set(gpu_indices)) != 4:
        raise ValueError(
            "GPU mapping must contain four distinct UUIDs and indices"
        )
    gpu_rows = topology.get("gpu_rows")
    normalized_gpu_rows = []
    if not isinstance(gpu_rows, list) or len(gpu_rows) != 4:
        raise ValueError("GPU topology inventory mismatch")
    for row in gpu_rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"gpu_index", "gpu_uuid", "pci_bus_id"}
            or re.fullmatch(
                r"[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:"
                r"[0-9A-Fa-f]{2}\.[0-7]",
                str(row.get("pci_bus_id", "")),
            )
            is None
        ):
            raise ValueError("GPU topology schema mismatch")
        normalized_gpu_rows.append((
            _integer(row.get("gpu_index"), "GPU topology index"),
            _string(row.get("gpu_uuid"), "GPU topology UUID"),
            row["pci_bus_id"].lower(),
        ))
    if (
        {(index, uuid) for index, uuid, _ in normalized_gpu_rows}
        != set(zip(gpu_indices, gpu_uuids))
        or len({pci_bus_id for _, _, pci_bus_id in normalized_gpu_rows}) != 4
    ):
        raise ValueError(
            "GPU topology identity mismatch: GPU UUID/index/PCI drift"
        )
    matrix = topology.get("interconnect_matrix")
    if (
        not isinstance(matrix, str)
        or not matrix.strip()
        or any(f"GPU{gpu_index}" not in matrix for gpu_index in gpu_indices)
    ):
        raise ValueError("GPU topology interconnect matrix mismatch")
    limits = topology.get("strict_clean_limits")
    if limits != {
        "maximum_memory_used_mib": 1024,
        "maximum_utilization_percent": 5,
        "compute_processes": [],
    }:
        raise ValueError("strict-clean GPU limits mismatch")
    _validate_clean_gpu_inventory(
        topology.get("controller_entry_inventory"),
        expected_indices=gpu_indices,
        expected_uuids=gpu_uuids,
        label="controller entry",
    )
    worker_entries = topology.get("worker_entry_inventories")
    expected_worker_ids = _expected_worker_ids()
    if (
        not isinstance(worker_entries, list)
        or len(worker_entries) != len(expected_worker_ids)
        or [
            row.get("worker_id")
            for row in worker_entries
            if isinstance(row, dict)
        ]
        != expected_worker_ids
    ):
        raise ValueError("worker-entry inventory mismatch")
    for worker_entry in worker_entries:
        if (
            not isinstance(worker_entry, dict)
            or set(worker_entry)
            != {
                "worker_id",
                "capture_source",
                "capture_stage",
                "captured_at_unix_ns",
                "gpu_rows",
            }
        ):
            raise ValueError("worker-entry inventory mismatch")
        worker_id = worker_entry["worker_id"]
        expected_source = (
            "controller/gpu_admission_samples.jsonl"
            if worker_id == "correctness"
            else (
                "controller/nsys-resource-samples.raw.jsonl"
                if "__nsys_replay__" in worker_id
                else "controller/structured-resource-samples.raw.jsonl"
            )
        )
        if (
            worker_entry["capture_source"] != expected_source
            or not isinstance(worker_entry["capture_stage"], str)
            or not worker_entry["capture_stage"]
            or _integer(
                worker_entry["captured_at_unix_ns"],
                "worker-entry capture timestamp",
                minimum=1,
            )
            <= 0
        ):
            raise ValueError("worker-entry inventory mismatch")
        _validate_clean_gpu_inventory(
            worker_entry["gpu_rows"],
            expected_indices=gpu_indices,
            expected_uuids=gpu_uuids,
            label=f"worker-entry {worker_entry['worker_id']}",
        )
    cleanup = environment.get("cleanup")
    cleanup_valid = (
        isinstance(cleanup, dict)
        and cleanup.get("process_groups_destroyed") is True
        and cleanup.get("rank_exit_codes") == [0, 0, 0, 0]
        and cleanup.get("owned_children_remaining") == []
    )
    final_inventory = (
        cleanup.get("final_gpu_inventory")
        if isinstance(cleanup, dict)
        else None
    )
    if not isinstance(final_inventory, list) or len(final_inventory) != 4:
        cleanup_valid = False
    else:
        final_by_uuid = {
            row.get("gpu_uuid"): row
            for row in final_inventory
            if isinstance(row, dict)
        }
        cleanup_valid = cleanup_valid and set(final_by_uuid) == set(gpu_uuids)
        for gpu_uuid in gpu_uuids:
            row = final_by_uuid.get(gpu_uuid, {})
            cleanup_valid = cleanup_valid and (
                _number(
                    row.get("memory_used_mib"),
                    "cleanup memory used",
                )
                <= 1024
                and _number(
                    row.get("utilization_percent"),
                    "cleanup utilization",
                )
                <= 5
                and row.get("compute_processes") == []
            )
    if not cleanup_valid:
        raise ValueError("cleanup evidence is incomplete")
    return {
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "model_revision": model_revision,
        "gpu_uuids": gpu_uuids,
        "strict_clean_worker_entry_count": len(worker_entries),
        "cleanup_valid": True,
    }


def _normalize_layer(layer) -> dict:
    if not isinstance(layer, dict):
        raise ValueError("profile layer must be an object")
    result = {
        "layer_index": _integer(layer.get("layer_index"), "layer index"),
        "layer_role": _string(layer.get("layer_role"), "layer role"),
    }
    operations = layer.get("operation_inventory")
    if not isinstance(operations, list) or not operations:
        raise ValueError("operation inventory must be non-empty")
    normalized_operations = []
    ordinals = set()
    for operation in operations:
        if not isinstance(operation, list) or len(operation) != 3:
            raise ValueError("operation inventory row is invalid")
        ordinal = _integer(operation[0], "operation ordinal")
        if ordinal in ordinals:
            raise ValueError("operation inventory has duplicate ordinal")
        ordinals.add(ordinal)
        normalized_operations.append([
            ordinal,
            _string(operation[1], "operation class"),
            _string(operation[2], "operation name"),
        ])
    if [row[0] for row in normalized_operations] != sorted(ordinals):
        raise ValueError("operation inventory order mismatch")
    result["operation_inventory"] = normalized_operations
    byte_rows = layer.get("collective_byte_inventory")
    if not isinstance(byte_rows, list):
        raise ValueError("collective byte inventory must be a list")
    expected_collectives = {
        operation[0]
        for operation in normalized_operations
        if operation[1] == "collective"
    }
    normalized_bytes = []
    byte_ordinals = set()
    for row in byte_rows:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("collective byte inventory row is invalid")
        ordinal = _integer(row[0], "collective operation ordinal")
        byte_count = _integer(row[1], "collective bytes")
        if ordinal in byte_ordinals:
            raise ValueError(
                "collective byte inventory has duplicate ordinal"
            )
        byte_ordinals.add(ordinal)
        normalized_bytes.append([ordinal, byte_count])
    if (
        byte_ordinals != expected_collectives
        or [row[0] for row in normalized_bytes]
        != sorted(byte_ordinals)
    ):
        raise ValueError("collective byte inventory mismatch")
    result["collective_byte_inventory"] = normalized_bytes
    for metric in LAYER_METRICS:
        result[metric] = _integer(layer.get(metric), metric)
    result["step_critical_interval_ns"] = _integer(
        layer.get("step_critical_interval_ns"),
        "step critical interval",
        minimum=1,
    )
    overlap = result["compute_collective_overlap_ns"]
    if (
        overlap > result["compute_ns"]
        or overlap > result["collective_ns"]
        or result["exposed_collective_ns"] + overlap
        != result["collective_ns"]
        or result["gemm_ns"] > result["compute_ns"]
    ):
        raise ValueError("profile interval arithmetic mismatch")
    if (
        result["collective_count"] != len(normalized_bytes)
        or result["collective_bytes"]
        != sum(row[1] for row in normalized_bytes)
    ):
        raise ValueError("collective byte accounting mismatch")
    if (
        result["compute_ns"] + result["exposed_collective_ns"]
        > result["step_critical_interval_ns"]
    ):
        raise ValueError("profile interval arithmetic exceeds critical interval")
    for field in ("cpu_global_tids", "stream_ids"):
        values = layer.get(field)
        if not isinstance(values, list):
            raise ValueError(f"{field} must be a list")
        result[field] = [
            _integer(value, field) for value in values
        ]
    return result


def _validate_profiles(rows: list[dict], identity: dict) -> dict:
    if not isinstance(rows, list) or len(rows) != 140:
        raise ValueError("profile row inventory mismatch")
    normalized = []
    sequences = set()
    cases = {}
    attempts = set()
    gpu_by_rank = {}
    alignment_by_workload = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("profile row must be an object")
        sequence = _integer(raw.get("sequence_index"), "sequence index")
        if sequence in sequences:
            raise ValueError("duplicate profile sequence index")
        sequences.add(sequence)
        if raw.get("schema_version") != PROFILE_SCHEMA:
            raise ValueError("profile row schema mismatch")
        attempts.add(_string(raw.get("attempt"), "attempt identity"))
        _string(raw.get("process_identity"), "process identity")
        if raw.get("source_tree_sha256") != identity["source_tree_sha256"]:
            raise ValueError("source revision or tree identity drift")
        if raw.get("model_revision") != identity["model_revision"]:
            raise ValueError("model revision drift")
        workload = raw.get("workload")
        if workload not in WORKLOADS:
            raise ValueError("workload inventory drift")
        for field, expected in WORKLOADS[workload].items():
            if raw.get(field) != expected:
                raise ValueError(f"workload contract drift for {workload}")
        phase = raw.get("phase")
        repetitions = {
            "warmup": WARMUP_REPETITIONS,
            "measured": MEASURED_REPETITIONS,
        }
        if phase not in repetitions:
            raise ValueError("profile phase mismatch")
        repetition = _integer(raw.get("repetition"), "repetition")
        if repetition not in repetitions[phase]:
            raise ValueError("profile repetition inventory mismatch")
        rank = _integer(raw.get("rank"), "rank")
        if rank not in RANKS:
            raise ValueError("rank inventory mismatch")
        gpu_uuid = _string(raw.get("gpu_uuid"), "GPU UUID")
        if gpu_uuid != identity["gpu_uuids"][rank]:
            raise ValueError("GPU UUID drift for rank")
        gpu_by_rank.setdefault(rank, gpu_uuid)
        if (
            raw.get("finalization_status") != "complete"
            or raw.get("trace_coverage") != "COMPLETE"
        ):
            raise ValueError("profile finalization or trace coverage incomplete")
        steps = raw.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("profile steps are missing")
        normalized_steps = []
        decode_ordinals = set()
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError("profile step must be an object")
            decode_ordinal = _integer(
                step.get("decode_ordinal"),
                "decode ordinal",
            )
            if decode_ordinal in decode_ordinals:
                raise ValueError("duplicate decode ordinal")
            decode_ordinals.add(decode_ordinal)
            critical_rank = _integer(
                step.get("critical_rank"),
                "critical rank",
            )
            if critical_rank not in RANKS:
                raise ValueError("critical rank mismatch")
            layers = step.get("layers")
            if not isinstance(layers, list) or not layers:
                raise ValueError("profile step layers must be non-empty")
            layer_identities = [
                (
                    layer.get("layer_index"),
                    layer.get("layer_role"),
                )
                for layer in layers
                if isinstance(layer, dict)
            ]
            if (
                len(layer_identities) != len(layers)
                or len(layer_identities) != len(set(layer_identities))
            ):
                raise ValueError("profile step layers contain duplicates")
            normalized_steps.append({
                "request_set_sha256": _sha256(
                    step.get("request_set_sha256"),
                    "request set SHA-256",
                ),
                "decode_ordinal": decode_ordinal,
                "critical_rank": critical_rank,
                "final_required_offset_ns": _integer(
                    step.get("final_required_offset_ns"),
                    "final required offset",
                    minimum=1,
                ),
                "layers": [
                    _normalize_layer(layer)
                    for layer in layers
                ],
            })
        normalized_steps.sort(key=lambda row: row["decode_ordinal"])
        row = dict(raw)
        row.update({
            "sequence_index": sequence,
            "repetition": repetition,
            "rank": rank,
            "decode_time_ns": _integer(
                raw.get("decode_time_ns"),
                "decode time",
                minimum=1,
            ),
            "steps": normalized_steps,
        })
        key = (workload, phase, repetition)
        if rank in cases.setdefault(key, {}):
            raise ValueError("duplicate rank row")
        cases[key][rank] = row
        normalized.append(row)
    if sequences != set(range(140)):
        raise ValueError("profile sequence inventory mismatch")
    if len(attempts) != 1:
        raise ValueError("profile attempt identity drift")
    expected_cases = {
        (workload, phase, repetition)
        for workload in WORKLOADS
        for phase, repetitions in (
            ("warmup", WARMUP_REPETITIONS),
            ("measured", MEASURED_REPETITIONS),
        )
        for repetition in repetitions
    }
    if set(cases) != expected_cases:
        raise ValueError("workload or repetition inventory mismatch")
    for key, rank_rows in cases.items():
        if set(rank_rows) != set(RANKS):
            raise ValueError("rank inventory mismatch")
        signatures = {
            tuple(
                (
                    step["request_set_sha256"],
                    step["decode_ordinal"],
                    step["critical_rank"],
                    tuple(
                        (
                            layer["layer_index"],
                            layer["layer_role"],
                            tuple(
                                tuple(operation)
                                for operation in layer[
                                    "operation_inventory"
                                ]
                            ),
                        )
                        for layer in step["layers"]
                    ),
                )
                for step in row["steps"]
            )
            for row in rank_rows.values()
        }
        if len(signatures) != 1:
            raise ValueError("rank alignment mismatch")
        for step_index, reference_step in enumerate(rank_rows[0]["steps"]):
            computed = max(
                RANKS,
                key=lambda candidate: (
                    rank_rows[candidate]["steps"][step_index][
                        "final_required_offset_ns"
                    ],
                    candidate,
                ),
            )
            if reference_step["critical_rank"] != computed:
                raise ValueError("critical rank does not match timeline")
        if key[1] == "measured":
            cross_signature = tuple(
                (
                    step["request_set_sha256"],
                    step["decode_ordinal"],
                    tuple(
                        (
                            layer["layer_index"],
                            layer["layer_role"],
                            tuple(
                                tuple(operation)
                                for operation in layer[
                                    "operation_inventory"
                                ]
                            ),
                        )
                        for layer in step["layers"]
                    ),
                )
                for step in rank_rows[0]["steps"]
            )
            previous = alignment_by_workload.setdefault(
                key[0],
                cross_signature,
            )
            if previous != cross_signature:
                raise ValueError("cross-repetition alignment mismatch")
    return {
        "rows": sorted(normalized, key=lambda row: row["sequence_index"]),
        "cases": cases,
    }


def _structured_rows(rank_rows: dict[int, dict]) -> list[dict]:
    result = []
    for rank in RANKS:
        row = rank_rows[rank]
        for step in row["steps"]:
            for layer in step["layers"]:
                byte_by_ordinal = dict(
                    layer["collective_byte_inventory"]
                )
                for ordinal, operation_class, operation_name in layer[
                    "operation_inventory"
                ]:
                    structured = {
                        "attempt": row["attempt"],
                        "workload": row["workload"],
                        "repetition": row["repetition"],
                        "request_set_sha256": step["request_set_sha256"],
                        "decode_ordinal": step["decode_ordinal"],
                        "rank": rank,
                        "layer_index": layer["layer_index"],
                        "layer_role": layer["layer_role"],
                        "operation_ordinal": ordinal,
                        "operation_class": operation_class,
                        "operation_name": operation_name,
                    }
                    if operation_class == "collective":
                        structured["collective_bytes"] = byte_by_ordinal[
                            ordinal
                        ]
                    result.append(structured)
    return result


def _verify_nsys(root: Path, profiles: dict) -> None:
    metric_fields = (
        "step_critical_interval_ns",
        *LAYER_METRICS,
        "cpu_global_tids",
        "stream_ids",
    )
    for workload in WORKLOADS:
        for repetition in MEASURED_REPETITIONS:
            rank_rows = profiles["cases"][
                (workload, "measured", repetition)
            ]
            parsed = parse_nsys_sqlite(
                root / "nsys" / f"{workload}-r{repetition}.sqlite",
                _structured_rows(rank_rows),
            )
            if parsed.get("classification") != "COMPLETE":
                raise ValueError(
                    "Nsight correlation is incomplete: "
                    f"{parsed.get('coverage_errors')}"
                )
            parsed_rows = {
                (
                    row["rank"],
                    row["decode_ordinal"],
                    row["layer_index"],
                    row["layer_role"],
                ): row
                for row in parsed["rows"]
            }
            expected_rows = {}
            for rank, profile in rank_rows.items():
                for step in profile["steps"]:
                    for layer in step["layers"]:
                        expected_rows[(
                            rank,
                            step["decode_ordinal"],
                            layer["layer_index"],
                            layer["layer_role"],
                        )] = layer
            if set(parsed_rows) != set(expected_rows):
                raise ValueError("Nsight/profile layer inventory mismatch")
            for key, expected in expected_rows.items():
                actual = parsed_rows[key]
                if any(actual[field] != expected[field] for field in metric_fields):
                    raise ValueError("Nsight/profile interval arithmetic mismatch")
            critical_rows = {
                row["decode_ordinal"]: row
                for row in parsed["critical_rows"]
            }
            for step_index, step in enumerate(rank_rows[0]["steps"]):
                actual = critical_rows.get(step["decode_ordinal"])
                if (
                    actual is None
                    or actual["critical_rank"] != step["critical_rank"]
                    or actual["final_required_offset_ns"]
                    != step["final_required_offset_ns"]
                ):
                    raise ValueError("Nsight/profile critical-rank mismatch")


def _repetition_metrics(rank_rows: dict[int, dict]) -> dict:
    exposed = 0
    independent_compute = 0
    critical_interval = 0
    for step_index, reference in enumerate(rank_rows[0]["steps"]):
        critical = rank_rows[reference["critical_rank"]]["steps"][step_index]
        intervals = {
            layer["step_critical_interval_ns"]
            for layer in critical["layers"]
        }
        if len(intervals) != 1:
            raise ValueError("critical interval mismatch across layers")
        critical_interval += next(iter(intervals))
        for layer in critical["layers"]:
            exposed += layer["exposed_collective_ns"]
            independent_compute += (
                layer["compute_ns"]
                - layer["compute_collective_overlap_ns"]
            )
    if (
        critical_interval <= 0
        or exposed + independent_compute > critical_interval
    ):
        raise ValueError("critical interval arithmetic mismatch")
    return {
        "exposed_collective_ns": exposed,
        "independent_compute_ns": independent_compute,
        "step_critical_interval_ns": critical_interval,
        "exposed_communication_ratio": exposed / critical_interval,
        "overlap_headroom_lower_bound": (
            min(exposed, independent_compute) / critical_interval
        ),
    }


def _layer_summary(rank_rows_by_repetition) -> list[dict]:
    values = {}
    for rank_rows in rank_rows_by_repetition:
        for step_index, reference in enumerate(rank_rows[0]["steps"]):
            critical = rank_rows[reference["critical_rank"]]["steps"][
                step_index
            ]
            for layer in critical["layers"]:
                key = (layer["layer_index"], layer["layer_role"])
                bucket = values.setdefault(
                    key,
                    {metric: [] for metric in LAYER_METRICS},
                )
                for metric in LAYER_METRICS:
                    bucket[metric].append(layer[metric])
    return [
        {
            "layer_index": layer_index,
            "layer_role": layer_role,
            **{
                f"median_{metric}": statistics.median(bucket[metric])
                for metric in LAYER_METRICS
            },
        }
        for (layer_index, layer_role), bucket in sorted(values.items())
    ]


def _online_summary(payload: dict, workload: str) -> dict:
    rows = [
        row
        for row in payload.get("rows", [])
        if isinstance(row, dict) and row.get("workload") == workload
    ]
    if (
        len(rows) != 5
        or {row.get("repetition") for row in rows}
        != set(MEASURED_REPETITIONS)
    ):
        raise ValueError("online workload inventory mismatch")
    qps = []
    tokens_per_second = []
    distributions = {
        "ttft_ms": [],
        "tpot_ms": [],
        "e2e_latency_ms": [],
    }
    for row in rows:
        count = _integer(row.get("request_count"), "request count", minimum=1)
        elapsed = _number(row.get("elapsed_s"), "elapsed seconds", positive=True)
        output_count = _integer(
            row.get("output_token_count"),
            "output token count",
            minimum=1,
        )
        if output_count != count * WORKLOADS[workload]["output_tokens"]:
            raise ValueError("online output token count mismatch")
        qps.append(count / elapsed)
        tokens_per_second.append(output_count / elapsed)
        for field, values in distributions.items():
            observed = row.get(field)
            if not isinstance(observed, list) or len(observed) != count:
                raise ValueError(f"{field} inventory mismatch")
            values.extend(_number(value, field) for value in observed)
    return {
        "median_request_qps": statistics.median(qps),
        "median_output_tokens_per_s": statistics.median(tokens_per_second),
        **{
            field: {
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
                "p99": _percentile(values, 0.99),
            }
            for field, values in distributions.items()
        },
    }


def _memory_summary(payload: dict, workload: str) -> dict:
    rows = [
        row
        for row in payload.get("rows", [])
        if isinstance(row, dict) and row.get("workload") == workload
    ]
    inventory = set()
    allocated = {}
    reserved = {}
    for row in rows:
        repetition = _integer(row.get("repetition"), "memory repetition")
        rank = _integer(row.get("rank"), "memory rank")
        key = (repetition, rank)
        if key in inventory:
            raise ValueError("duplicate memory row")
        inventory.add(key)
        allocated[rank] = max(
            allocated.get(rank, 0),
            _integer(row.get("peak_allocated_bytes"), "allocated bytes"),
        )
        reserved[rank] = max(
            reserved.get(rank, 0),
            _integer(row.get("peak_reserved_bytes"), "reserved bytes"),
        )
    expected = {
        (repetition, rank)
        for repetition in MEASURED_REPETITIONS
        for rank in RANKS
    }
    if inventory != expected:
        raise ValueError("memory rank inventory mismatch")
    return {
        "peak_allocated_bytes_by_rank": {
            str(rank): allocated[rank] for rank in RANKS
        },
        "peak_reserved_bytes_by_rank": {
            str(rank): reserved[rank] for rank in RANKS
        },
    }


def _resource_summary(
    rows: list[dict],
    workload: str,
    gpu_uuids: list[str],
) -> dict:
    selected = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("workload") == workload
    ]
    inventory = set()
    utilization = []
    power = []
    for row in selected:
        repetition = _integer(row.get("repetition"), "resource repetition")
        gpu_uuid = _string(row.get("gpu_uuid"), "resource GPU UUID")
        if gpu_uuid not in gpu_uuids:
            raise ValueError("resource GPU UUID drift")
        key = (repetition, gpu_uuid)
        if key in inventory:
            raise ValueError("duplicate resource sample")
        inventory.add(key)
        utilization.append(
            _number(
                row.get("gpu_utilization_percent"),
                "GPU utilization",
            )
        )
        power.append(_number(row.get("power_watts"), "GPU power"))
    expected = {
        (repetition, gpu_uuid)
        for repetition in MEASURED_REPETITIONS
        for gpu_uuid in gpu_uuids
    }
    if inventory != expected:
        raise ValueError("resource sample inventory mismatch")
    return {
        "gpu_utilization_percent": {
            "p50": _percentile(utilization, 0.50),
            "p95": _percentile(utilization, 0.95),
            "max": max(utilization),
        },
        "power_watts": {
            "p50": _percentile(power, 0.50),
            "p95": _percentile(power, 0.95),
            "max": max(power),
        },
    }


def _correctness_summary(rows: list[dict]) -> tuple[dict, bool]:
    expected = {
        (workload, repetition, rank)
        for workload in WORKLOADS
        for repetition in MEASURED_REPETITIONS
        for rank in RANKS
    }
    actual = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("correctness row must be an object")
        key = (
            row.get("workload"),
            _integer(row.get("repetition"), "correctness repetition"),
            _integer(row.get("rank"), "correctness rank"),
        )
        if key in actual:
            raise ValueError("duplicate correctness row")
        actual.add(key)
    summary = {
        "row_count": len(rows),
        "exact_token_match_rows": sum(
            row.get("exact_token_match") is True for row in rows
        ),
        "argmax_match_rows": sum(
            row.get("argmax_match") is True for row in rows
        ),
        "finite_logit_rows": sum(
            row.get("finite_logits") is True for row in rows
        ),
        "numeric_tolerance_rows": sum(
            row.get("within_numeric_tolerance") is True for row in rows
        ),
        "max_abs_logit_error": max(
            (
                _number(
                    row.get("max_abs_logit_error"),
                    "maximum absolute logit error",
                )
                for row in rows
            ),
            default=0.0,
        ),
        "max_rel_logit_error": max(
            (
                _number(
                    row.get("max_rel_logit_error"),
                    "maximum relative logit error",
                )
                for row in rows
            ),
            default=0.0,
        ),
    }
    valid = (
        actual == expected
        and len(rows) == len(expected)
        and all(
            summary[field] == len(expected)
            for field in (
                "exact_token_match_rows",
                "argmax_match_rows",
                "finite_logit_rows",
                "numeric_tolerance_rows",
            )
        )
    )
    return summary, valid


def _profiler_overhead(payload: dict, identity: dict) -> float:
    controls = payload.get("overhead_controls")
    if not isinstance(controls, list):
        raise ValueError("profiler overhead controls must be a list")
    expected = {
        (workload, repetition)
        for workload in WORKLOADS
        for repetition in MEASURED_REPETITIONS
    }
    actual = set()
    ratios = []
    for row in controls:
        if not isinstance(row, dict):
            raise ValueError("profiler overhead row must be an object")
        key = (
            row.get("workload"),
            _integer(row.get("repetition"), "overhead repetition"),
        )
        if key in actual:
            raise ValueError("duplicate profiler overhead row")
        actual.add(key)
        if (
            row.get("source_tree_sha256") != identity["source_tree_sha256"]
            or row.get("model_revision") != identity["model_revision"]
            or row.get("rank_inventory") != list(RANKS)
            or row.get("gpu_uuids") != identity["gpu_uuids"]
        ):
            raise ValueError("profiler overhead identity mismatch")
        unprofiled = _number(
            row.get("unprofiled_ns"),
            "unprofiled duration",
            positive=True,
        )
        profiled = _number(
            row.get("profiled_ns"),
            "profiled duration",
            positive=True,
        )
        ratios.append(profiled / unprofiled - 1.0)
    if actual != expected:
        raise ValueError("profiler overhead inventory mismatch")
    return max(ratios)


def _direction(ratio: float, headroom: float) -> str:
    if ratio >= GO_EXPOSURE_RATIO and headroom >= GO_HEADROOM_RATIO:
        return "GO"
    if ratio < NO_GO_EXPOSURE_RATIO and headroom < NO_GO_HEADROOM_RATIO:
        return "NO_GO"
    return "MIDDLE"


def _classify(summary: dict) -> str:
    if summary["correctness_valid"] is not True:
        return "INVALID_CORRECTNESS"
    if summary["resource_identity_valid"] is not True:
        return "INVALID_RESOURCE_IDENTITY"
    if (
        summary["trace_coverage_complete"] is not True
        or summary["complete_four_rank_alignment"] is not True
    ):
        return "INCONCLUSIVE_TRACE_COVERAGE"
    directions = {}
    for workload, contract in WORKLOADS.items():
        payload = summary["workloads"][workload]
        if payload["workload_family"] != contract["workload_family"]:
            raise ValueError("workload family mismatch")
        repetitions = payload["repetitions"]
        ratios = [row["exposed_communication_ratio"] for row in repetitions]
        headrooms = [
            row["overlap_headroom_lower_bound"] for row in repetitions
        ]
        direction = _direction(
            statistics.median(ratios),
            statistics.median(headrooms),
        )
        if sum(
            _direction(ratio, headroom) == direction
            for ratio, headroom in zip(ratios, headrooms)
        ) < 4:
            return "INCONCLUSIVE_VARIANCE"
        directions[workload] = direction
    causal_go = any(
        directions[workload] == "GO"
        and WORKLOADS[workload]["workload_family"] == "causal"
        for workload in WORKLOADS
    )
    online_go = any(
        directions[workload] == "GO"
        and WORKLOADS[workload]["workload_family"] == "online"
        for workload in WORKLOADS
    )
    if (
        causal_go
        and online_go
        and summary["profiler_overhead_ratio"]
        <= MAX_PROFILER_OVERHEAD_RATIO
    ):
        return "GO_COMMUNICATION_OVERLAP"
    if all(direction == "NO_GO" for direction in directions.values()):
        return "NO_GO_ALREADY_HIDDEN"
    return "INCONCLUSIVE_LOW_HEADROOM"


def _assert_equivalent(actual, expected, message: str, path="root") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{message}: field inventory differs at {path}")
        for key in expected:
            _assert_equivalent(
                actual[key],
                expected[key],
                message,
                f"{path}.{key}",
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{message}: list inventory differs at {path}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_equivalent(
                left,
                right,
                message,
                f"{path}[{index}]",
            )
        return
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        if not math.isclose(
            float(actual),
            float(expected),
            rel_tol=METRIC_TOLERANCE,
            abs_tol=METRIC_TOLERANCE,
        ):
            raise ValueError(f"{message}: metric differs at {path}")
        return
    if actual != expected:
        raise ValueError(f"{message}: value differs at {path}")


def _reconstruct_summary(
    root: Path,
    profiles: dict,
    identity: dict,
) -> dict:
    online = _load_json(root / "online_metrics.json")
    memory = _load_json(root / "memory_summary.json")
    resources = _load_jsonl(root / "resource_samples.jsonl")
    correctness_rows = _load_jsonl(root / "correctness_rows.jsonl")
    if not isinstance(online, dict) or not isinstance(memory, dict):
        raise ValueError("auxiliary summary schema mismatch")
    online_rows = online.get("rows")
    memory_rows = memory.get("rows")
    if (
        not isinstance(online_rows, list)
        or len(online_rows) != len(WORKLOADS) * len(MEASURED_REPETITIONS)
        or any(
            not isinstance(row, dict) or row.get("workload") not in WORKLOADS
            for row in online_rows
        )
    ):
        raise ValueError("online row inventory mismatch")
    if (
        not isinstance(memory_rows, list)
        or len(memory_rows)
        != len(WORKLOADS) * len(MEASURED_REPETITIONS) * len(RANKS)
        or any(
            not isinstance(row, dict) or row.get("workload") not in WORKLOADS
            for row in memory_rows
        )
    ):
        raise ValueError("memory row inventory mismatch")
    if (
        len(resources)
        != len(WORKLOADS) * len(MEASURED_REPETITIONS) * len(RANKS)
        or any(
            not isinstance(row, dict) or row.get("workload") not in WORKLOADS
            for row in resources
        )
    ):
        raise ValueError("resource row inventory mismatch")
    correctness, correctness_valid = _correctness_summary(correctness_rows)
    if not correctness_valid:
        raise ValueError("correctness evidence is invalid")
    overhead = _profiler_overhead(online, identity)
    if overhead > MAX_PROFILER_OVERHEAD_RATIO:
        raise ValueError(
            "profiler overhead exceeds the three percent verification limit"
        )
    workloads = {}
    for workload, contract in WORKLOADS.items():
        rank_rows_by_repetition = [
            profiles["cases"][(workload, "measured", repetition)]
            for repetition in MEASURED_REPETITIONS
        ]
        repetitions = [
            {
                "repetition": repetition,
                **_repetition_metrics(rank_rows),
            }
            for repetition, rank_rows in zip(
                MEASURED_REPETITIONS,
                rank_rows_by_repetition,
            )
        ]
        critical_times = {
            repetition: max(
                row["decode_time_ns"] for row in rank_rows.values()
            )
            for repetition, rank_rows in zip(
                MEASURED_REPETITIONS,
                rank_rows_by_repetition,
            )
        }
        median_time = statistics.median(critical_times.values())
        representative = min(
            critical_times,
            key=lambda repetition: (
                abs(critical_times[repetition] - median_time),
                repetition,
            ),
        )
        workloads[workload] = {
            "workload_family": contract["workload_family"],
            "repetitions": repetitions,
            "median_exposed_communication_ratio": statistics.median(
                row["exposed_communication_ratio"] for row in repetitions
            ),
            "median_overlap_headroom_lower_bound": statistics.median(
                row["overlap_headroom_lower_bound"] for row in repetitions
            ),
            "representative_repetition": representative,
            "layer_summary": _layer_summary(rank_rows_by_repetition),
            "online": _online_summary(online, workload),
            "memory": _memory_summary(memory, workload),
            "resources": _resource_summary(
                resources,
                workload,
                identity["gpu_uuids"],
            ),
        }
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "source_tree_sha256": identity["source_tree_sha256"],
        "model_revision": identity["model_revision"],
        "rank_inventory": list(RANKS),
        "gpu_uuids": identity["gpu_uuids"],
        "correctness": correctness,
        "correctness_valid": True,
        "resource_identity_valid": True,
        "trace_coverage_complete": True,
        "complete_four_rank_alignment": True,
        "profiler_overhead_ratio": overhead,
        "workloads": workloads,
    }
    summary["classification"] = _classify(summary)
    return summary


def _verify_producer_outputs(
    root: Path,
    summary: dict,
    identity: dict,
) -> None:
    producer = _load_json(root / "communication_exposure_summary.json")
    if producer.get("classification") != summary["classification"]:
        raise ValueError("producer/verifier classification mismatch")
    _assert_equivalent(
        producer,
        summary,
        "communication summary drift",
    )
    expected_layers = {
        "schema_version": LAYER_SUMMARY_SCHEMA,
        "source_tree_sha256": summary["source_tree_sha256"],
        "model_revision": summary["model_revision"],
        "workloads": {
            workload: payload["layer_summary"]
            for workload, payload in summary["workloads"].items()
        },
    }
    _assert_equivalent(
        _load_json(root / "layer_summary.json"),
        expected_layers,
        "layer summary drift",
    )
    report = (root / "report.md").read_text(encoding="utf-8")
    expected_classification = (
        f"Classification: `{summary['classification']}`"
    )
    if expected_classification not in report:
        raise ValueError("report classification does not match machine result")
    for label, value in (
        ("Source revision", identity["source_revision"]),
        ("Model revision", identity["model_revision"]),
    ):
        if f"{label}: `{value}`" not in report:
            raise ValueError("report identity does not match machine result")


def _rewrite_manifest(root: Path) -> None:
    artifacts = {
        relative: _sha256_file(root / relative)
        for relative in sorted(_actual_artifacts(root))
    }
    _write_json_atomic(root / "manifest.sha256", {
        "schema_version": MANIFEST_SCHEMA,
        "artifacts": artifacts,
    })


def verify_bundle(root: Path) -> dict:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("bundle root must be an existing directory")
    _verify_manifest(root)
    identity = _validate_identity(root)
    profiles = _validate_profiles(
        _load_jsonl(root / "profile_rows.jsonl"),
        identity,
    )
    _verify_nsys(root, profiles)
    summary = _reconstruct_summary(root, profiles, identity)
    _verify_producer_outputs(root, summary, identity)
    result = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS",
        "source_revision": identity["source_revision"],
        "source_tree_sha256": identity["source_tree_sha256"],
        "model_revision": identity["model_revision"],
        "rank_inventory": list(RANKS),
        "gpu_uuids": identity["gpu_uuids"],
        "profile_row_count": len(profiles["rows"]),
        "correctness_row_count": summary["correctness"]["row_count"],
        "nsys_trace_count": len(_expected_trace_names()),
        "artifact_hashes_verified": True,
        "complete_four_rank_alignment": True,
        "trace_coverage_complete": True,
        "correctness_valid": True,
        "cleanup_valid": identity["cleanup_valid"],
        "strict_clean_worker_entry_count": identity[
            "strict_clean_worker_entry_count"
        ],
        "profiler_overhead_ratio": summary["profiler_overhead_ratio"],
        "producer_classification": summary["classification"],
        "reconstructed_classification": summary["classification"],
        "workloads": {
            workload: {
                "median_exposed_communication_ratio": payload[
                    "median_exposed_communication_ratio"
                ],
                "median_overlap_headroom_lower_bound": payload[
                    "median_overlap_headroom_lower_bound"
                ],
                "representative_repetition": payload[
                    "representative_repetition"
                ],
            }
            for workload, payload in summary["workloads"].items()
        },
    }
    _write_json_atomic(root / "independent_verification.json", result)
    _rewrite_manifest(root)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Independently verify a Qwen3.8-27B TP4 communication "
            "profile artifact bundle."
        )
    )
    parser.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args(argv)
    result = verify_bundle(args.bundle)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
