from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.assemble_lease_sealed_state_commit_overlap import (
    PRODUCER_ARTIFACTS,
    _load_json,
    assemble_raw_attempt,
    assemble_bundle,
)
from tools.test_lease_sealed_state_commit_overlap import (
    passing_memory,
    passing_rows,
)


ATTEMPT = "20260907-lease-sealed-state-commit-overlap-stage0-r1"
SOURCE_REVISION = "a" * 40
SOURCE_TREE_SHA256 = "b" * 64


def passing_inputs():
    rows = passing_rows()
    for row in rows:
        row["attempt"] = ATTEMPT
        row["source_revision"] = SOURCE_REVISION
        row["source_tree_sha256"] = SOURCE_TREE_SHA256
    identity = {
        "attempt": ATTEMPT,
        "source_revision": SOURCE_REVISION,
        "source_tree_sha256": SOURCE_TREE_SHA256,
    }
    return {
        "source_identity": {
            "schema_version": (
                "lease-sealed-state-commit-overlap-source.v1"
            ),
            **identity,
            "environment": {
                "hostname": "gpu-host",
                "python_version": "3.12.0",
                "torch_version": "2.8.0",
                "cuda_version": "12.8",
                "runtime_capabilities": {
                    "rank_rows": [
                        {
                            "rank": rank,
                            "device_index": rank,
                            "device_name": "NVIDIA A100",
                            "device_uuid": f"GPU-{rank}",
                            "compute_capability": [8, 0],
                            "hostname": "gpu-host",
                            "python_version": "3.12.0",
                            "driver_version": "550.54.15",
                            "cuda_version": "12.8",
                            "torch_version": "2.8.0",
                            "nccl_available": True,
                            "nccl_version": "(2, 21, 5)",
                            "world_size": 4,
                            "hidden_size": 5120,
                            "collective_dtype": "float32",
                            "output_dtype": "bfloat16",
                            "state_dtype": "bfloat16",
                        }
                        for rank in range(4)
                    ]
                },
            },
            "gpu_rank_rows": [
                {
                    "rank": rank,
                    "device_index": rank,
                    "device_uuid": f"GPU-{rank}",
                }
                for rank in range(4)
            ],
            "admission": {
                "classification": "STRICT_CLEAN",
                "rank_rows": [
                    {
                        "rank": rank,
                        "memory_mib": 0,
                        "utilization_percent": 0,
                        "compute_processes": [],
                    }
                    for rank in range(4)
                ],
            },
        },
        "rows": rows,
        "memory": passing_memory(),
        "lifecycle": {
            **identity,
            "rank_rows": [
                {
                    "rank": rank,
                    "active_tokens": active_tokens,
                    "active_state_preserved_before_publish": True,
                    "published_state_exact": True,
                    "abort_preserved_old_state": True,
                    "commit_identity_match": True,
                }
                for active_tokens in (1, 4, 8)
                for rank in range(4)
            ],
        },
        "cleanup": {
            **identity,
            "classification": "CLEAN",
            "rank_rows": [
                {
                    "rank": rank,
                    "streams_released": True,
                    "events_released": True,
                    "timed_out": False,
                    "process_group_destroyed": True,
                }
                for rank in range(4)
            ],
            "owned_children_remaining": [],
            "exact_tag_scans": [[], [], []],
        },
    }


def test_assembler_writes_complete_manifested_go_bundle(tmp_path):
    result = assemble_bundle(output_root=tmp_path, **passing_inputs())

    assert result["classification"] == (
        "GO_LEASE_SEALED_OVERLAP_MICROGATE"
    )
    assert {path.name for path in tmp_path.iterdir()} == set(
        PRODUCER_ARTIFACTS
    )
    producer = json.loads(
        (tmp_path / "producer_result.json").read_text()
    )
    assert producer["stage1_authorized"] is True
    assert producer["measurement_row_count"] == 180


def test_assembler_rejects_nonempty_output_identity_drift_and_nan(tmp_path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "existing").write_text("occupied")
    with pytest.raises(ValueError, match="must be empty"):
        assemble_bundle(output_root=occupied, **passing_inputs())

    inputs = passing_inputs()
    inputs["rows"][0]["attempt"] = "different-attempt"
    with pytest.raises(ValueError, match="identity"):
        assemble_bundle(output_root=tmp_path / "identity", **inputs)

    inputs = passing_inputs()
    inputs["memory"]["rank_rows"][0][
        "maximum_reserved_delta_bytes"
    ] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        assemble_bundle(output_root=tmp_path / "nan", **inputs)


def test_strict_json_loader_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"attempt":"a","attempt":"b"}\n')

    with pytest.raises(ValueError, match="duplicate JSON key"):
        _load_json(path)


def test_cli_exposes_raw_attempt_assembly_arguments():
    script = Path(__file__).with_name(
        "assemble_lease_sealed_state_commit_overlap.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--raw-root" in result.stdout
    assert "--source-identity" in result.stdout
    assert "--admission" in result.stdout
    assert "--output-root" in result.stdout


def test_raw_attempt_includes_runtime_capabilities_in_environment(tmp_path):
    inputs = passing_inputs()
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    source_path = tmp_path / "source.json"
    admission_path = tmp_path / "admission.json"
    output_root = tmp_path / "bundle"
    source = inputs["source_identity"]
    admission = source.pop("admission")
    source_path.write_text(json.dumps(source))
    admission_path.write_text(json.dumps(admission))
    (raw_root / "measurement_rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in inputs["rows"])
    )
    (raw_root / "memory.json").write_text(json.dumps(inputs["memory"]))
    (raw_root / "lifecycle.json").write_text(
        json.dumps(inputs["lifecycle"])
    )
    (raw_root / "cleanup.json").write_text(json.dumps(inputs["cleanup"]))
    capabilities = {
        "rank_rows": [
            {
                "rank": rank,
                "device_name": "NVIDIA A100",
                "device_uuid": f"GPU-{rank}",
                "device_index": rank,
                "compute_capability": [8, 0],
                "hostname": "gpu-host",
                "python_version": "3.12.0",
                "driver_version": "550.54.15",
                "torch_version": "2.8.0",
                "cuda_version": "12.8",
                "nccl_available": True,
                "nccl_version": "(2, 21, 5)",
                "world_size": 4,
                "hidden_size": 5120,
                "collective_dtype": "float32",
                "output_dtype": "bfloat16",
                "state_dtype": "bfloat16",
            }
            for rank in range(4)
        ]
    }
    (raw_root / "runtime_capabilities.json").write_text(
        json.dumps(capabilities)
    )

    assemble_raw_attempt(
        raw_root=raw_root,
        source_identity_path=source_path,
        admission_path=admission_path,
        output_root=output_root,
    )

    environment = json.loads(
        (output_root / "environment_manifest.json").read_text()
    )
    assert environment["runtime_capabilities"] == capabilities


def test_assembler_rejects_runtime_gpu_uuid_drift(tmp_path):
    inputs = passing_inputs()
    inputs["source_identity"]["environment"]["runtime_capabilities"][
        "rank_rows"
    ][0]["device_uuid"] = "GPU-different"

    with pytest.raises(ValueError, match="runtime capability"):
        assemble_bundle(output_root=tmp_path, **inputs)


def test_assembler_rejects_duplicate_gpu_rank_identity(tmp_path):
    inputs = passing_inputs()
    inputs["source_identity"]["gpu_rank_rows"][1][
        "device_uuid"
    ] = "GPU-0"
    inputs["source_identity"]["environment"]["runtime_capabilities"][
        "rank_rows"
    ][1]["device_uuid"] = "GPU-0"

    with pytest.raises(ValueError, match="runtime capability"):
        assemble_bundle(output_root=tmp_path, **inputs)


def test_assembler_rejects_mislabeled_dirty_admission(tmp_path):
    inputs = passing_inputs()
    inputs["source_identity"]["admission"]["rank_rows"][0][
        "memory_mib"
    ] = 1025

    with pytest.raises(ValueError, match="admission"):
        assemble_bundle(output_root=tmp_path, **inputs)


def clone_inputs():
    return copy.deepcopy(passing_inputs())
