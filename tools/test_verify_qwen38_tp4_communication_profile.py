#!/usr/bin/env python3
"""Contracts for the independent Qwen3.8 TP4 profile verifier."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

from tools.verify_qwen38_tp4_communication_profile import (
    _structured_rows as _verifier_structured_rows,
    verify_bundle,
)


SOURCE_REVISION = "1" * 40
SOURCE_TREE_SHA256 = "a" * 64
MODEL_REVISION = "2" * 40
GPU_UUIDS = tuple(f"GPU-{rank}" for rank in range(4))
APPROVED_REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
WORKLOADS = {
    "P0": ("causal", 256, 128, 1),
    "P1": ("causal", 2048, 128, 1),
    "Q0": ("online", 256, 128, 4),
    "Q1": ("online", 256, 128, 8),
    "Q2": ("online", 2048, 128, 4),
}


def _worker_ids() -> list[str]:
    result = ["correctness"]
    for workload in WORKLOADS:
        for repetition in range(2):
            result.append(f"{workload}__warmup__r{repetition}")
        for phase in ("measured", "nsys_replay"):
            for repetition in range(5):
                result.append(
                    f"{workload}__{phase}__r{repetition}"
                )
    return result


def _clean_gpu_inventory() -> list[dict]:
    return [
        {
            "gpu_index": rank,
            "gpu_uuid": GPU_UUIDS[rank],
            "memory_used_mib": 0,
            "utilization_percent": 0,
            "compute_processes": [],
        }
        for rank in range(4)
    ]


def _canonical_bytes(payload) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _write_json(path: Path, payload) -> None:
    path.write_bytes(_canonical_bytes(payload))


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(_canonical_bytes(row).decode() for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _operation_inventory():
    return [
        [0, "gemm", "qkv_projection"],
        [1, "collective", "row_parallel_all_reduce"],
        [2, "attention", "flash_attention"],
    ]


def _profile_layer(rank: int) -> dict:
    return {
        "layer_index": 0,
        "layer_role": "full_attention",
        "operation_inventory": _operation_inventory(),
        "step_critical_interval_ns": 210 + rank * 10,
        "gemm_ns": 60,
        "collective_ns": 80,
        "compute_ns": 130 + rank * 10,
        "exposed_collective_ns": 60,
        "compute_collective_overlap_ns": 20,
        "gpu_idle_ns": 20,
        "collective_count": 1,
        "collective_bytes": 4096,
        "collective_byte_inventory": [[1, 4096]],
        "critical_path_ns": 190 + rank * 10,
        "cpu_global_tids": [((100 + rank) << 24) | 7],
        "stream_ids": [7, 11],
    }


def test_structured_rows_preserve_each_collective_byte_count():
    rank_rows = {}
    for rank in range(4):
        layer = _profile_layer(rank)
        layer["operation_inventory"] = [
            [0, "gemm", "qkv_projection"],
            [1, "collective", "row_parallel_all_reduce"],
            [2, "collective", "row_parallel_all_reduce"],
        ]
        layer["collective_count"] = 2
        layer["collective_bytes"] = 3072
        layer["collective_byte_inventory"] = [
            [1, 1024],
            [2, 2048],
        ]
        rank_rows[rank] = {
            "attempt": "attempt-a",
            "workload": "Q0",
            "repetition": 0,
            "rank": rank,
            "steps": [{
                "request_set_sha256": "c" * 64,
                "decode_ordinal": 0,
                "layers": [layer],
            }],
        }

    rows = _verifier_structured_rows(rank_rows)

    assert [
        row["collective_bytes"]
        for row in rows
        if row["rank"] == 0
        and row["operation_class"] == "collective"
    ] == [1024, 2048]


def _profile_rows() -> list[dict]:
    rows = []
    sequence_index = 0
    for workload, (
        family,
        prompt_tokens,
        output_tokens,
        concurrency,
    ) in WORKLOADS.items():
        for phase, repetitions in (
            ("warmup", range(2)),
            ("measured", range(5)),
        ):
            for repetition in repetitions:
                for rank in range(4):
                    rows.append({
                        "schema_version": (
                            "qwen38.communication-profile-row.v1"
                        ),
                        "sequence_index": sequence_index,
                        "attempt": "attempt-a",
                        "source_tree_sha256": SOURCE_TREE_SHA256,
                        "model_revision": MODEL_REVISION,
                        "workload": workload,
                        "workload_family": family,
                        "phase": phase,
                        "repetition": repetition,
                        "rank": rank,
                        "gpu_uuid": GPU_UUIDS[rank],
                        "process_identity": (
                            f"worker-{workload}-{phase}-"
                            f"{repetition}-r{rank}"
                        ),
                        "finalization_status": "complete",
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                        "concurrency": concurrency,
                        "decode_time_ns": (
                            (100 + repetition) * 1_000_000 + rank
                        ),
                        "trace_coverage": "COMPLETE",
                        "steps": [{
                            "request_set_sha256": "c" * 64,
                            "decode_ordinal": 0,
                            "critical_rank": 3,
                            "final_required_offset_ns": 250 + rank * 10,
                            "layers": [_profile_layer(rank)],
                        }],
                    })
                    sequence_index += 1
    return rows


def _structured_rows(workload: str, repetition: int) -> list[dict]:
    rows = []
    for rank in range(4):
        identity = {
            "attempt": "attempt-a",
            "workload": workload,
            "repetition": repetition,
            "request_set_sha256": "c" * 64,
            "decode_ordinal": 0,
            "rank": rank,
            "layer_index": 0,
            "layer_role": "full_attention",
        }
        rows.extend((
            identity | {
                "operation_ordinal": 0,
                "operation_class": "gemm",
                "operation_name": "qkv_projection",
            },
            identity | {
                "operation_ordinal": 1,
                "operation_class": "collective",
                "operation_name": "row_parallel_all_reduce",
                "collective_bytes": 4096,
            },
            identity | {
                "operation_ordinal": 2,
                "operation_class": "attention",
                "operation_name": "flash_attention",
            },
        ))
    return rows


def _create_trace(
    path: Path,
    workload: str,
    repetition: int,
) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE StringIds ("
        "id INTEGER PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE NVTX_EVENTS ("
        "start INTEGER NOT NULL, end INTEGER, eventType INTEGER NOT NULL, "
        "text TEXT, globalTid INTEGER, textId INTEGER)"
    )
    connection.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME ("
        "start INTEGER NOT NULL, end INTEGER NOT NULL, "
        "globalTid INTEGER NOT NULL, correlationId INTEGER NOT NULL, "
        "nameId INTEGER)"
    )
    connection.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ("
        "start INTEGER NOT NULL, end INTEGER NOT NULL, "
        "deviceId INTEGER NOT NULL, contextId INTEGER NOT NULL, "
        "streamId INTEGER NOT NULL, correlationId INTEGER, "
        "globalPid INTEGER, demangledName INTEGER, "
        "shortName INTEGER NOT NULL)"
    )
    connection.executemany(
        "INSERT INTO StringIds(id, value) VALUES (?, ?)",
        (
            (1, "cutlass_gemm"),
            (2, "ncclKernel_AllReduce_RING_LL"),
            (3, "flash_attention"),
            (4, "cudaLaunchKernel"),
        ),
    )
    for rank in range(4):
        base = rank * 1_000
        global_pid = (100 + rank) << 24
        global_tid = global_pid | 7
        prefix = (
            f"decode_internal/attempt=attempt-a/workload={workload}/"
            f"repetition={repetition}/rank={rank}"
        )
        connection.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?)",
            (
                (
                    base + 100,
                    base + 500,
                    59,
                    f"{prefix}/decode_steady",
                    global_tid,
                    None,
                ),
                (
                    base + 110,
                    base + 490,
                    59,
                    f"{prefix}/layer/0/full_attention",
                    global_tid,
                    None,
                ),
                (
                    base + 120,
                    base + 130,
                    59,
                    f"{prefix}/operation/0/gemm/qkv_projection",
                    global_tid,
                    None,
                ),
                (
                    base + 210,
                    base + 220,
                    59,
                    f"{prefix}/operation/1/collective/"
                    "row_parallel_all_reduce",
                    global_tid,
                    None,
                ),
                (
                    base + 230,
                    base + 240,
                    59,
                    f"{prefix}/operation/2/attention/flash_attention",
                    global_tid,
                    None,
                ),
            ),
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME "
            "VALUES (?, ?, ?, ?, ?)",
            (
                (
                    base + 122,
                    base + 128,
                    global_tid,
                    rank * 10 + 1,
                    4,
                ),
                (
                    base + 212,
                    base + 218,
                    global_tid,
                    rank * 10 + 2,
                    4,
                ),
                (
                    base + 232,
                    base + 238,
                    global_tid,
                    rank * 10 + 3,
                    4,
                ),
            ),
        )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    base + 140,
                    base + 200,
                    rank,
                    1,
                    7,
                    rank * 10 + 1,
                    global_pid,
                    1,
                    1,
                ),
                (
                    base + 220,
                    base + 300,
                    rank,
                    1,
                    11,
                    rank * 10 + 2,
                    global_pid,
                    2,
                    2,
                ),
                (
                    base + 280,
                    base + 350 + rank * 10,
                    rank,
                    1,
                    7,
                    rank * 10 + 3,
                    global_pid,
                    3,
                    3,
                ),
            ),
        )
    connection.commit()
    connection.close()


def _online_payload() -> dict:
    rows = []
    controls = []
    for workload in WORKLOADS:
        for repetition in range(5):
            rows.append({
                "workload": workload,
                "repetition": repetition,
                "request_count": 2,
                "elapsed_s": 1.0,
                "output_token_count": 256,
                "ttft_ms": [10.0 + repetition, 12.0 + repetition],
                "tpot_ms": [2.0 + repetition, 4.0 + repetition],
                "e2e_latency_ms": [
                    100.0 + repetition,
                    110.0 + repetition,
                ],
            })
            controls.append({
                "workload": workload,
                "repetition": repetition,
                "source_tree_sha256": SOURCE_TREE_SHA256,
                "model_revision": MODEL_REVISION,
                "rank_inventory": [0, 1, 2, 3],
                "gpu_uuids": list(GPU_UUIDS),
                "unprofiled_ns": 1000,
                "profiled_ns": 1020,
            })
    return {"rows": rows, "overhead_controls": controls}


def _memory_rows() -> list[dict]:
    return [
        {
            "workload": workload,
            "repetition": repetition,
            "rank": rank,
            "peak_allocated_bytes": 10_000 + repetition * 100 + rank,
            "peak_reserved_bytes": 20_000 + repetition * 100 + rank,
        }
        for workload in WORKLOADS
        for repetition in range(5)
        for rank in range(4)
    ]


def _resource_rows() -> list[dict]:
    return [
        {
            "workload": workload,
            "repetition": repetition,
            "gpu_uuid": GPU_UUIDS[rank],
            "gpu_utilization_percent": 60 + repetition + rank * 5,
            "power_watts": 250.0 + repetition + rank * 5,
        }
        for workload in WORKLOADS
        for repetition in range(5)
        for rank in range(4)
    ]


def _correctness_rows() -> list[dict]:
    return [
        {
            "workload": workload,
            "repetition": repetition,
            "rank": rank,
            "exact_token_match": True,
            "argmax_match": True,
            "finite_logits": True,
            "within_numeric_tolerance": True,
            "max_abs_logit_error": 0.005 + rank * 0.001,
            "max_rel_logit_error": 0.001 + rank * 0.0001,
        }
        for workload in WORKLOADS
        for repetition in range(5)
        for rank in range(4)
    ]


def _producer_summary() -> dict:
    workloads = {}
    for workload, (family, _, _, _) in WORKLOADS.items():
        workloads[workload] = {
            "workload_family": family,
            "repetitions": [
                {
                    "repetition": repetition,
                    "exposed_collective_ns": 60,
                    "independent_compute_ns": 140,
                    "step_critical_interval_ns": 240,
                    "exposed_communication_ratio": 0.25,
                    "overlap_headroom_lower_bound": 0.25,
                }
                for repetition in range(5)
            ],
            "median_exposed_communication_ratio": 0.25,
            "median_overlap_headroom_lower_bound": 0.25,
            "representative_repetition": 2,
            "layer_summary": [{
                "layer_index": 0,
                "layer_role": "full_attention",
                "median_gemm_ns": 60,
                "median_collective_ns": 80,
                "median_compute_ns": 160,
                "median_exposed_collective_ns": 60,
                "median_compute_collective_overlap_ns": 20,
                "median_gpu_idle_ns": 20,
                "median_collective_count": 1,
                "median_collective_bytes": 4096,
                "median_critical_path_ns": 220,
            }],
            "online": {
                "median_request_qps": 2.0,
                "median_output_tokens_per_s": 256.0,
                "ttft_ms": {"p50": 13.0, "p95": 15.55, "p99": 15.91},
                "tpot_ms": {"p50": 5.0, "p95": 7.55, "p99": 7.91},
                "e2e_latency_ms": {
                    "p50": 107.0,
                    "p95": 113.55,
                    "p99": 113.91,
                },
            },
            "memory": {
                "peak_allocated_bytes_by_rank": {
                    str(rank): 10_400 + rank for rank in range(4)
                },
                "peak_reserved_bytes_by_rank": {
                    str(rank): 20_400 + rank for rank in range(4)
                },
            },
            "resources": {
                "gpu_utilization_percent": {
                    "p50": 69.5,
                    "p95": 78.05,
                    "max": 79.0,
                },
                "power_watts": {
                    "p50": 259.5,
                    "p95": 268.05,
                    "max": 269.0,
                },
            },
        }
    return {
        "schema_version": "qwen38.communication-exposure-summary.v1",
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "model_revision": MODEL_REVISION,
        "rank_inventory": [0, 1, 2, 3],
        "gpu_uuids": list(GPU_UUIDS),
        "correctness": {
            "row_count": 100,
            "exact_token_match_rows": 100,
            "argmax_match_rows": 100,
            "finite_logit_rows": 100,
            "numeric_tolerance_rows": 100,
            "max_abs_logit_error": 0.008,
            "max_rel_logit_error": 0.0013,
        },
        "correctness_valid": True,
        "resource_identity_valid": True,
        "trace_coverage_complete": True,
        "complete_four_rank_alignment": True,
        "profiler_overhead_ratio": 0.02,
        "workloads": workloads,
        "classification": "GO_COMMUNICATION_OVERLAP",
    }


def _refresh_manifest(root: Path) -> None:
    artifacts = {}
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.name == "manifest.sha256"
            or path.name.startswith(".")
        ):
            continue
        artifacts[path.relative_to(root).as_posix()] = _sha256(path)
    _write_json(root / "manifest.sha256", {
        "schema_version": (
            "qwen38.tp4-communication-profile-manifest.v1"
        ),
        "artifacts": artifacts,
    })


def _write_bundle(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    nsys = root / "nsys"
    nsys.mkdir()
    model_files = {
        "config.json": {"sha256": "d" * 64, "size": 1024},
        "model.safetensors.index.json": {
            "sha256": "e" * 64,
            "size": 2048,
        },
        "model-00001-of-00002.safetensors": {
            "sha256": "f" * 64,
            "size": 4096,
        },
        "model-00002-of-00002.safetensors": {
            "sha256": "0" * 64,
            "size": 4096,
        },
        "tokenizer.json": {"sha256": "3" * 64, "size": 512},
    }
    tokenizer_files = ["tokenizer.json"]
    model_manifest = {
        "schema_version": "tinyllmforge.qwen38-model-manifest.v1",
        "repository": "Qwen/Qwen3.8-27B",
        "resolved_revision": MODEL_REVISION,
        "model_root": f"{APPROVED_REMOTE_ROOT}/models/Qwen3.8-27B",
        "config_sha256": model_files["config.json"]["sha256"],
        "text_config_sha256": "4" * 64,
        "tokenizer_inventory_sha256": hashlib.sha256(
            _canonical_bytes({
                name: model_files[name]
                for name in tokenizer_files
            })
        ).hexdigest(),
        "tokenizer_files": tokenizer_files,
        "checkpoint_index": "model.safetensors.index.json",
        "checkpoint_index_sha256": model_files[
            "model.safetensors.index.json"
        ]["sha256"],
        "checkpoint_tensor_count": 1024,
        "checkpoint_shards": [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
        "files": model_files,
    }
    _write_json(root / "model_manifest.json", model_manifest)
    _write_json(root / "source_manifest.json", {
        "schema_version": "tinyllmforge.source-manifest.v1",
        "source_revision": SOURCE_REVISION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "model_manifest_sha256": _sha256(root / "model_manifest.json"),
    })
    _write_json(root / "environment.json", {
        "schema_version": "qwen38.tp4-profile-environment.v1",
        "source_revision": SOURCE_REVISION,
        "model_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "tensor_parallel_size": 4,
        "decoding": "greedy",
        "temperature": 0.0,
        "fixed_output_tokens": 128,
        "scheduler_policy": "identical",
        "cuda_graph_policy": "identical",
        "cleanup": {
            "process_groups_destroyed": True,
            "rank_exit_codes": [0, 0, 0, 0],
            "owned_children_remaining": [],
            "final_gpu_inventory": [
                {
                    "gpu_uuid": gpu_uuid,
                    "memory_used_mib": 0,
                    "utilization_percent": 0,
                    "compute_processes": [],
                }
                for gpu_uuid in GPU_UUIDS
            ],
        },
    })
    _write_json(root / "gpu_topology.json", {
        "schema_version": "qwen38.tp4-profile-topology.v1",
        "rank_mapping": [
            {
                "rank": rank,
                "gpu_index": rank,
                "gpu_uuid": GPU_UUIDS[rank],
            }
            for rank in range(4)
        ],
        "gpu_rows": [
            {
                "gpu_index": rank,
                "gpu_uuid": GPU_UUIDS[rank],
                "pci_bus_id": f"00000000:{rank + 1:02x}:00.0",
            }
            for rank in range(4)
        ],
        "interconnect_matrix": (
            "        GPU0 GPU1 GPU2 GPU3\n"
            "GPU0    X    NV18 NV18 NV18\n"
            "GPU1    NV18 X    NV18 NV18\n"
            "GPU2    NV18 NV18 X    NV18\n"
            "GPU3    NV18 NV18 NV18 X\n"
        ),
        "controller_entry_inventory": _clean_gpu_inventory(),
        "worker_entry_inventories": [
            {
                "worker_id": worker_id,
                "capture_source": (
                    "controller/gpu_admission_samples.jsonl"
                    if worker_id == "correctness"
                    else (
                        "controller/nsys-resource-samples.raw.jsonl"
                        if "__nsys_replay__" in worker_id
                        else (
                            "controller/"
                            "structured-resource-samples.raw.jsonl"
                        )
                    )
                ),
                "capture_stage": (
                    "before_tinyllmforge_tp4"
                    if worker_id == "correctness"
                    else "pre_gpu_mutation"
                ),
                "captured_at_unix_ns": 1_000_000 + index,
                "gpu_rows": _clean_gpu_inventory(),
            }
            for index, worker_id in enumerate(_worker_ids())
        ],
        "strict_clean_limits": {
            "maximum_memory_used_mib": 1024,
            "maximum_utilization_percent": 5,
            "compute_processes": [],
        },
    })
    _write_json(root / "workload_manifest.json", {
        "schema_version": "qwen38.tp4-profile-workloads.v1",
        "source_revision": SOURCE_REVISION,
        "model_revision": MODEL_REVISION,
        "order": list(WORKLOADS),
        "warmup_repetitions": [0, 1],
        "measured_repetitions": [0, 1, 2, 3, 4],
        "rank_inventory": [0, 1, 2, 3],
        "workloads": {
            workload: {
                "workload_family": family,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "concurrency": concurrency,
            }
            for workload, (
                family,
                prompt_tokens,
                output_tokens,
                concurrency,
            ) in WORKLOADS.items()
        },
    })
    _write_jsonl(root / "profile_rows.jsonl", _profile_rows())
    _write_jsonl(root / "correctness_rows.jsonl", _correctness_rows())
    _write_json(root / "online_metrics.json", _online_payload())
    _write_json(root / "memory_summary.json", {"rows": _memory_rows()})
    _write_jsonl(root / "resource_samples.jsonl", _resource_rows())
    producer = _producer_summary()
    _write_json(root / "layer_summary.json", {
        "schema_version": "qwen38.layer-summary.v1",
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "model_revision": MODEL_REVISION,
        "workloads": {
            workload: payload["layer_summary"]
            for workload, payload in producer["workloads"].items()
        },
    })
    _write_json(
        root / "communication_exposure_summary.json",
        producer,
    )
    for workload in WORKLOADS:
        for repetition in range(5):
            _create_trace(
                nsys / f"{workload}-r{repetition}.sqlite",
                workload,
                repetition,
            )
    (root / "report.md").write_text(
        "# Qwen3.8 TP4 Communication Exposure\n\n"
        "Classification: `GO_COMMUNICATION_OVERLAP`\n\n"
        f"Source revision: `{SOURCE_REVISION}`\n\n"
        f"Model revision: `{MODEL_REVISION}`\n",
        encoding="utf-8",
    )
    _refresh_manifest(root)


@pytest.fixture(scope="session")
def pristine_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("qwen38-verifier-pristine")
    _write_bundle(root)
    return root


@pytest.fixture
def bundle(tmp_path, pristine_bundle):
    root = tmp_path / "bundle"
    shutil.copytree(pristine_bundle, root)
    return root


def test_verify_bundle_recomputes_complete_bundle_and_updates_manifest(bundle):
    source = (
        ROOT / "tools/verify_qwen38_tp4_communication_profile.py"
    ).read_text(encoding="utf-8")
    assert "qwen38_communication_exposure import" not in source

    result = verify_bundle(bundle)

    assert result["schema_version"] == (
        "qwen38.tp4-communication-profile-independent-verification.v1"
    )
    assert result["status"] == "PASS"
    assert result["reconstructed_classification"] == (
        "GO_COMMUNICATION_OVERLAP"
    )
    assert result["producer_classification"] == (
        "GO_COMMUNICATION_OVERLAP"
    )
    assert result["profile_row_count"] == 140
    assert result["correctness_row_count"] == 100
    assert result["nsys_trace_count"] == 25
    assert result["rank_inventory"] == [0, 1, 2, 3]
    assert result["gpu_uuids"] == list(GPU_UUIDS)
    assert result["trace_coverage_complete"] is True
    assert result["correctness_valid"] is True
    assert result["cleanup_valid"] is True
    assert result["strict_clean_worker_entry_count"] == 61
    assert result["profiler_overhead_ratio"] == pytest.approx(0.02)
    assert result["workloads"]["P0"][
        "median_exposed_communication_ratio"
    ] == pytest.approx(0.25)
    assert result["workloads"]["P0"][
        "median_overlap_headroom_lower_bound"
    ] == pytest.approx(0.25)
    verification = json.loads(
        (bundle / "independent_verification.json").read_text(
            encoding="utf-8"
        )
    )
    assert verification == result
    manifest = json.loads(
        (bundle / "manifest.sha256").read_text(encoding="utf-8")
    )
    assert manifest["artifacts"]["independent_verification.json"] == (
        _sha256(bundle / "independent_verification.json")
    )


def test_verifier_cli_can_run_as_a_direct_script():
    completed = subprocess.run(
        [
            sys.executable,
            os.fspath(
                ROOT / "tools/verify_qwen38_tp4_communication_profile.py"
            ),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--bundle" in completed.stdout


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda root: (root / "memory_summary.json").unlink(),
            "missing",
        ),
        (
            lambda root: (root / "unexpected.txt").write_text(
                "extra\n",
                encoding="utf-8",
            ),
            "inventory",
        ),
        (
            lambda root: (root / "report.md").write_text(
                "tampered\n",
                encoding="utf-8",
            ),
            "digest mismatch",
        ),
    ),
)
def test_verify_bundle_rejects_missing_extra_or_hash_mismatch(
    bundle,
    mutation,
    message,
):
    mutation(bundle)

    with pytest.raises(ValueError, match=message):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda root: (root / "nsys/manifest.sha256").write_text(
            "{}\n",
            encoding="utf-8",
        ),
        lambda root: (root / "linked-nsys").symlink_to(
            root / "nsys",
            target_is_directory=True,
        ),
    ),
)
def test_verify_bundle_rejects_ignored_or_unsafe_inventory_entries(
    bundle,
    mutation,
):
    mutation(bundle)

    with pytest.raises(ValueError, match="inventory|unsafe"):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    ("filename", "mutate", "message"),
    (
        (
            "layer_summary.json",
            lambda payload: payload["workloads"]["P0"][0].update(
                {"median_exposed_collective_ns": 999}
            ),
            "layer summary drift",
        ),
        (
            "communication_exposure_summary.json",
            lambda payload: payload["workloads"]["P0"].update(
                {"median_exposed_communication_ratio": 0.99}
            ),
            "communication summary drift",
        ),
        (
            "communication_exposure_summary.json",
            lambda payload: payload.update(
                {"classification": "NO_GO_ALREADY_HIDDEN"}
            ),
            "classification mismatch",
        ),
    ),
)
def test_verify_bundle_rejects_summary_only_tamper(
    bundle,
    filename,
    mutate,
    message,
):
    path = bundle / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(path, payload)
    _refresh_manifest(bundle)

    with pytest.raises(ValueError, match=message):
        verify_bundle(bundle)


def test_verify_bundle_rejects_report_machine_result_mismatch(bundle):
    (bundle / "report.md").write_text(
        "# Report\n\nClassification: `NO_GO_ALREADY_HIDDEN`\n",
        encoding="utf-8",
    )
    _refresh_manifest(bundle)

    with pytest.raises(ValueError, match="report classification"):
        verify_bundle(bundle)


def test_verify_bundle_rejects_report_source_revision_mismatch(bundle):
    report = (bundle / "report.md").read_text(encoding="utf-8")
    (bundle / "report.md").write_text(
        report.replace(SOURCE_REVISION, "8" * 40),
        encoding="utf-8",
    )
    _refresh_manifest(bundle)

    with pytest.raises(ValueError, match="report identity"):
        verify_bundle(bundle)


def test_verify_bundle_rejects_duplicate_json_keys(bundle):
    environment = bundle / "environment.json"
    text = environment.read_text(encoding="utf-8")
    environment.write_text(
        text.replace(
            '"source_revision":',
            f'"source_revision":"{"9" * 40}","source_revision":',
            1,
        ),
        encoding="utf-8",
    )
    _refresh_manifest(bundle)

    with pytest.raises(ValueError, match="duplicate JSON key"):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda root: _mutate_json(
                root / "environment.json",
                lambda payload: payload.update(
                    {"source_revision": "9" * 40}
                ),
            ),
            "source revision",
        ),
        (
            lambda root: _mutate_json(
                root / "workload_manifest.json",
                lambda payload: payload["workloads"]["P0"].update(
                    {"prompt_tokens": 257}
                ),
            ),
            "workload",
        ),
        (
            lambda root: _mutate_jsonl(
                root / "profile_rows.jsonl",
                lambda rows: rows[0].update({"rank": 7}),
            ),
            "rank",
        ),
        (
            lambda root: _mutate_json(
                root / "gpu_topology.json",
                lambda payload: payload["rank_mapping"][0].update(
                    {"gpu_uuid": "GPU-wrong"}
                ),
            ),
            "GPU UUID",
        ),
        (
            lambda root: _mutate_model_manifest(
                root,
                lambda payload: payload.update({
                    "model_root": "/models/Qwen3.8-27B",
                }),
            ),
            "approved remote root",
        ),
    ),
)
def test_verify_bundle_rejects_revision_workload_rank_or_gpu_drift(
    bundle,
    mutate,
    message,
):
    mutate(bundle)
    _refresh_manifest(bundle)

    with pytest.raises(ValueError, match=message):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda payload: payload[
                "controller_entry_inventory"
            ][0].update({"memory_used_mib": 1025}),
            "strict-clean",
        ),
        (
            lambda payload: payload[
                "worker_entry_inventories"
            ].pop(),
            "worker-entry",
        ),
        (
            lambda payload: payload[
                "worker_entry_inventories"
            ][0].pop("capture_source"),
            "worker-entry",
        ),
        (
            lambda payload: payload[
                "worker_entry_inventories"
            ][0].update({"captured_at_unix_ns": 0}),
            "worker-entry",
        ),
    ),
)
def test_verify_bundle_rejects_incomplete_gpu_admission(
    bundle,
    mutate,
    message,
):
    _mutate_json(bundle / "gpu_topology.json", mutate)
    _refresh_manifest(bundle)

    with pytest.raises(ValueError, match=message):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda root: _mutate_json(
                root / "environment.json",
                lambda payload: payload.update(
                    {"scheduler_policy": "different"}
                ),
            ),
            "environment contract",
        ),
        (
            lambda root: _mutate_json(
                root / "environment.json",
                lambda payload: payload.update(
                    {"cuda_graph_policy": "different"}
                ),
            ),
            "environment contract",
        ),
        (
            lambda root: _mutate_json(
                root / "gpu_topology.json",
                lambda payload: payload["gpu_rows"][0].update(
                    {"gpu_index": 99}
                ),
            ),
            "GPU topology",
        ),
    ),
)
def test_verify_bundle_rejects_environment_or_topology_identity_drift(
    bundle,
    mutate,
    message,
):
    mutate(bundle)
    _refresh_manifest(bundle)

    with pytest.raises(ValueError, match=message):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    ("filename", "container", "row"),
    (
        (
            "online_metrics.json",
            "rows",
            {
                "workload": "UNKNOWN",
                "repetition": 0,
                "request_count": 1,
                "elapsed_s": 1.0,
                "output_token_count": 128,
                "ttft_ms": [1.0],
                "tpot_ms": [1.0],
                "e2e_latency_ms": [1.0],
            },
        ),
        (
            "memory_summary.json",
            "rows",
            {
                "workload": "UNKNOWN",
                "repetition": 0,
                "rank": 0,
                "peak_allocated_bytes": 1,
                "peak_reserved_bytes": 1,
            },
        ),
    ),
)
def test_verify_bundle_rejects_extra_auxiliary_json_rows(
    bundle,
    filename,
    container,
    row,
):
    _mutate_json(
        bundle / filename,
        lambda payload: payload[container].append(row),
    )
    _refresh_manifest(bundle)

    with pytest.raises(ValueError, match="row inventory"):
        verify_bundle(bundle)


def test_verify_bundle_rejects_extra_resource_row(bundle):
    rows = _resource_rows()
    rows.append({
        "workload": "UNKNOWN",
        "repetition": 0,
        "gpu_uuid": GPU_UUIDS[0],
        "gpu_utilization_percent": 1,
        "power_watts": 1.0,
    })
    _write_jsonl(bundle / "resource_samples.jsonl", rows)
    _refresh_manifest(bundle)

    with pytest.raises(ValueError, match="row inventory"):
        verify_bundle(bundle)


def _mutate_json(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(path, payload)


def _mutate_model_manifest(root: Path, mutate) -> None:
    _mutate_json(root / "model_manifest.json", mutate)
    _mutate_json(
        root / "source_manifest.json",
        lambda payload: payload.update({
            "model_manifest_sha256": _sha256(
                root / "model_manifest.json"
            ),
        }),
    )


def _mutate_jsonl(path: Path, mutate) -> None:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    mutate(rows)
    _write_jsonl(path, rows)


def test_verify_bundle_rejects_incomplete_nsys_correlation(bundle):
    trace = bundle / "nsys/P0-r0.sqlite"
    with sqlite3.connect(trace) as connection:
        connection.execute(
            "DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME "
            "WHERE correlationId = 2"
        )
        connection.commit()
    _refresh_manifest(bundle)

    with pytest.raises(ValueError, match="Nsight correlation"):
        verify_bundle(bundle)


def test_verify_bundle_rejects_profile_interval_drift_from_nsys(bundle):
    def mutate(rows):
        target = next(
            row
            for row in rows
            if row["workload"] == "P0"
            and row["phase"] == "measured"
            and row["repetition"] == 0
            and row["rank"] == 3
        )
        layer = target["steps"][0]["layers"][0]
        layer["collective_ns"] = 81
        layer["exposed_collective_ns"] = 61

    _mutate_jsonl(bundle / "profile_rows.jsonl", mutate)
    _refresh_manifest(bundle)

    with pytest.raises(ValueError, match="Nsight/profile interval"):
        verify_bundle(bundle)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda rows: rows[0].update({"attempt": "attempt-b"}),
            "attempt identity",
        ),
        (
            lambda rows: rows[0]["steps"][0].update({"layers": []}),
            "layers",
        ),
    ),
)
def test_verify_bundle_rejects_incomplete_profile_semantics(
    bundle,
    mutate,
    message,
):
    _mutate_jsonl(bundle / "profile_rows.jsonl", mutate)
    _refresh_manifest(bundle)

    with pytest.raises(ValueError, match=message):
        verify_bundle(bundle)


def test_verify_bundle_rejects_invalid_correctness(bundle):
    _mutate_jsonl(
        bundle / "correctness_rows.jsonl",
        lambda rows: rows[0].update(
            {
                "exact_token_match": False,
                "within_numeric_tolerance": False,
            }
        ),
    )
    _refresh_manifest(bundle)

    with pytest.raises(ValueError, match="correctness"):
        verify_bundle(bundle)


def test_verify_bundle_rejects_incomplete_cleanup(bundle):
    _mutate_json(
        bundle / "environment.json",
        lambda payload: payload["cleanup"].update(
            {"owned_children_remaining": [4321]}
        ),
    )
    _refresh_manifest(bundle)

    with pytest.raises(ValueError, match="cleanup"):
        verify_bundle(bundle)


def test_verify_bundle_classifies_high_profiler_overhead_without_go(bundle):
    _mutate_json(
        bundle / "online_metrics.json",
        lambda payload: payload["overhead_controls"][0].update(
            {"profiled_ns": 1040}
        ),
    )
    _mutate_json(
        bundle / "communication_exposure_summary.json",
        lambda payload: payload.update({
            "profiler_overhead_ratio": 0.04,
            "classification": "INCONCLUSIVE_LOW_HEADROOM",
        }),
    )
    report_path = bundle / "report.md"
    report_path.write_text(
        report_path.read_text(encoding="utf-8").replace(
            "Classification: `GO_COMMUNICATION_OVERLAP`",
            "Classification: `INCONCLUSIVE_LOW_HEADROOM`",
        ),
        encoding="utf-8",
    )
    _refresh_manifest(bundle)

    result = verify_bundle(bundle)

    assert result["status"] == "PASS"
    assert result["profiler_overhead_ratio"] == pytest.approx(0.04)
    assert result["reconstructed_classification"] == (
        "INCONCLUSIVE_LOW_HEADROOM"
    )
