from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.assemble_qwen38_tp4_collective_reduction import (
    PRODUCER_ARTIFACTS,
    _load_json_strict,
    assemble_bundle,
)
from tools.qwen38_collective_reduction import (
    build_qwen38_static_collective_catalog,
)
from tools.qwen38_tp4_collective_reduction_worker import (
    WORKLOADS,
    build_collective_reduction_cases,
    collective_reduction_case_id,
)


ATTEMPT = "20260827-qwen38-tp4-collective-reduction-r1"
SOURCE_REVISION = "a" * 40
MODEL_REVISION = "b" * 40


def _profile():
    return {
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "vocab_size": 248320,
        "dtype": "bfloat16",
    }


def _sampled_ordinals(workload, repetition, budget):
    seed = (
        f"{SOURCE_REVISION}\0{ATTEMPT}\0"
        f"{workload}\0{repetition}\0{0}"
    ).encode()
    cohort = int.from_bytes(
        hashlib.sha256(seed).digest()[:8],
        "big",
    ) % 17
    width = math.ceil(130 / 17)
    start = cohort * width % 130
    return {
        (start + offset) % 130
        for offset in range(min(budget, 130))
    }


def _rank_snapshot(rank, row):
    sampled = _sampled_ordinals(
        row["workload"],
        row["repetition"],
        row["budget"],
    )
    collectives = []
    for ordinal, site in enumerate(
        build_qwen38_static_collective_catalog(
            _profile(),
            tensor_parallel_size=4,
        )
    ):
        is_token = site["site_role"] == "greedy_token_broadcast"
        collectives.append({
            "attempt": ATTEMPT,
            "workload": row["workload"],
            "repetition": row["repetition"],
            "rank": rank,
            "decode_ordinal": 0,
            "collective_ordinal": ordinal,
            "site_id": site["site_id"],
            "site_role": site["site_role"],
            "collective_kind": site["collective_kind"],
            "process_group": "tensor_parallel",
            "tensor_shape": [1] if is_token else [1, 5120],
            "tensor_dtype": (
                "torch.int64" if is_token else "torch.bfloat16"
            ),
            "tensor_bytes": 8 if is_token else 10240,
            "event_sampled": ordinal in sampled,
            "cuda_ns": 2_000 if ordinal in sampled else None,
            "status": "completed",
        })
    return {
        "schema": "tinyllmforge.synchronous-collective-census.v1",
        "rank": rank,
        "enabled": True,
        "finalization_status": "complete",
        "source_revision": SOURCE_REVISION,
        "attempt": ATTEMPT,
        "workload": row["workload"],
        "repetition": row["repetition"],
        "sample_budget": row["budget"],
        "cohort_count": 17,
        "expected_collective_count": 130,
        "steps": [{
            "decode_ordinal": 0,
            "collective_count": 130,
            "status": "completed",
        }],
        "collectives": collectives,
    }


def _arm(row, arm):
    control_ns = 1_000_000
    calibration_overhead = {
        0: 0.01,
        8: 0.02,
        16: 0.029,
        32: 0.06,
    }[row["budget"]]
    instrumented_ns = int(
        control_ns * (
            1.02
            if row["campaign_phase"] == "terminal"
            else 1.0 + calibration_overhead
        )
    )
    decode_time_ns = (
        control_ns if arm == "control" else instrumented_ns
    )
    requests = [
        {
            "request_id": f"request-{index}",
            "output_token_ids": [7] * row["output_tokens"],
            "ttft_ns": 100_000,
            "tpot_ns": 10_000 if arm == "control" else 10_200,
            "e2e_ns": 1_370_000,
        }
        for index in range(row["concurrency"])
    ]
    census = {
        "enabled": arm == "instrumented",
        "rank_inventory": [0, 1, 2, 3],
        "ranks": (
            [_rank_snapshot(rank, row) for rank in range(4)]
            if arm == "instrumented"
            else [
                {
                    "schema": (
                        "tinyllmforge."
                        "synchronous-collective-census.v1"
                    ),
                    "rank": rank,
                    "enabled": False,
                    "finalization_status": "complete",
                    "steps": [],
                    "collectives": [],
                }
                for rank in range(4)
            ]
        ),
    }
    return {
        "arm": arm,
        "policy": (
            {"enabled": False}
            if arm == "control"
            else {
                "enabled": True,
                "sample_budget": row["budget"],
                "cohort_count": 17,
                "expected_collective_count": 130,
                "source_revision": SOURCE_REVISION,
                "attempt": ATTEMPT,
                "workload": row["workload"],
                "repetition": row["repetition"],
            }
        ),
        "requests": requests,
        "decode_time_ns": decode_time_ns,
        "census": census,
        "memory": [
            {
                "rank": rank,
                "cuda_peak_allocated_bytes": 1_000 + rank,
                "cuda_peak_reserved_bytes": 2_000 + rank,
            }
            for rank in range(4)
        ],
    }


def _case(row):
    arm_order = (
        ("control", "instrumented")
        if row["repetition"] % 2 == 0
        else ("instrumented", "control")
    )
    return {
        "schema_version": "qwen38.tp4-collective-reduction-worker.v1",
        "classification": "PASS",
        "case_id": collective_reduction_case_id(**{
            key: row[key]
            for key in (
                "campaign_phase",
                "workload",
                "phase",
                "repetition",
                "budget",
            )
        }),
        "attempt": ATTEMPT,
        "source_revision": SOURCE_REVISION,
        **row,
        "arm_order": list(arm_order),
        "arms": [_arm(row, arm) for arm in arm_order],
    }


def _inputs():
    matrix = build_collective_reduction_cases(selected_budget=16)
    cases = [_case(row) for row in (
        matrix["calibration"] + matrix["terminal"]
    )]
    gpu_rows = [
        {
            "rank": rank,
            "gpu_index": rank + 2,
            "gpu_uuid": f"GPU-{rank}",
        }
        for rank in range(4)
    ]
    return {
        "source_identity": {
            "schema_version": (
                "qwen38.tp4-collective-reduction-source.v1"
            ),
            "attempt": ATTEMPT,
            "source_revision": SOURCE_REVISION,
            "source_tree_sha256": "c" * 64,
        },
        "model_manifest": {
            "schema_version": "tinyllmforge.qwen38-model-manifest.v1",
            "repository": "Qwen/Qwen3.8-27B",
            "revision": MODEL_REVISION,
            "text_profile": _profile(),
        },
        "gpu_topology": {
            "schema_version": (
                "qwen38.tp4-collective-reduction-topology.v1"
            ),
            "rank_mapping": gpu_rows,
            "interconnect_matrix": "GPU0 GPU1 GPU2 GPU3",
        },
        "cases": cases,
        "resource_samples": [
            {
                "case_id": case["case_id"],
                "owned_pids": [],
                "selected_gpus": [
                    {
                        "gpu_index": row["gpu_index"],
                        "gpu_uuid": row["gpu_uuid"],
                        "memory_used_mib": 0,
                        "utilization_percent": 0,
                        "compute_processes": [],
                    }
                    for row in gpu_rows
                ],
            }
            for case in cases
        ],
        "cleanup": {
            "schema_version": (
                "qwen38.tp4-collective-reduction-cleanup.v1"
            ),
            "complete": True,
            "process_group_destroyed": True,
            "owned_children_remaining": [],
            "exact_tag_scans": [[], [], []],
        },
    }


def test_assembler_writes_exact_terminal_contract(tmp_path):
    result = assemble_bundle(output_root=tmp_path, **_inputs())

    assert result["classification"] == "GO_SYNC_COLLECTIVE_REDUCTION"
    assert set(path.name for path in tmp_path.iterdir()) == set(
        PRODUCER_ARTIFACTS
    )
    manifest = json.loads(
        (tmp_path / "manifest.sha256").read_text()
    )
    assert set(manifest["artifacts"]) == (
        set(PRODUCER_ARTIFACTS) - {"manifest.sha256"}
    )
    assert result["selected_event_budget"] == 16


@pytest.mark.parametrize(
    "mutation",
    ("missing_case", "extra_case", "output_drift", "rank_divergence"),
)
def test_assembler_rejects_incomplete_or_divergent_cases(
    tmp_path,
    mutation,
):
    inputs = _inputs()
    if mutation == "missing_case":
        inputs["cases"].pop()
    elif mutation == "extra_case":
        inputs["cases"].append(copy.deepcopy(inputs["cases"][-1]))
        inputs["cases"][-1]["case_id"] += "-extra"
    elif mutation == "output_drift":
        inputs["cases"][-1]["arms"][1]["requests"][0][
            "output_token_ids"
        ][-1] = 8
    else:
        snapshot = inputs["cases"][-1]["arms"][1]["census"]["ranks"][3]
        snapshot["collectives"][7]["tensor_bytes"] += 2

    with pytest.raises(ValueError):
        assemble_bundle(output_root=tmp_path, **inputs)


def test_assembler_accepts_active_owned_gpu_processes(tmp_path):
    inputs = _inputs()
    for sample in inputs["resource_samples"]:
        sample["owned_pids"] = [101, 102, 103, 104]
        for offset, row in enumerate(sample["selected_gpus"]):
            row["memory_used_mib"] = 20_000 + offset
            row["utilization_percent"] = 80 + offset
            row["compute_processes"] = [{
                "pid": 101 + offset,
                "process_name": "python",
                "used_memory_mib": 20_000 + offset,
            }]

    result = assemble_bundle(output_root=tmp_path, **inputs)

    assert result["classification"] == "GO_SYNC_COLLECTIVE_REDUCTION"


def test_assembler_rejects_foreign_gpu_process(tmp_path):
    inputs = _inputs()
    sample = inputs["resource_samples"][0]
    sample["owned_pids"] = [101]
    sample["selected_gpus"][0]["compute_processes"] = [{
        "pid": 999,
        "process_name": "foreign",
        "used_memory_mib": 1,
    }]

    with pytest.raises(ValueError, match="resource identity"):
        assemble_bundle(output_root=tmp_path, **inputs)


def test_assembler_rejects_nonfinite_values_and_incomplete_cleanup(
    tmp_path,
):
    inputs = _inputs()
    inputs["cases"][-1]["arms"][1]["decode_time_ns"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        assemble_bundle(output_root=tmp_path / "nan", **inputs)

    inputs = _inputs()
    inputs["cleanup"]["owned_children_remaining"] = [123]
    with pytest.raises(ValueError, match="cleanup"):
        assemble_bundle(output_root=tmp_path / "cleanup", **inputs)


def test_strict_json_loader_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"classification":"A","classification":"B"}')

    with pytest.raises(ValueError, match="duplicate JSON key"):
        _load_json_strict(path)


def test_no_passing_event_budget_finishes_without_terminal_cases(
    tmp_path,
):
    inputs = _inputs()
    inputs["cases"] = [
        case
        for case in inputs["cases"]
        if case["campaign_phase"] == "calibration"
    ]
    for case in inputs["cases"]:
        if case["budget"] != 0:
            by_arm = {row["arm"]: row for row in case["arms"]}
            by_arm["instrumented"]["decode_time_ns"] = 1_060_000
    case_ids = {case["case_id"] for case in inputs["cases"]}
    inputs["resource_samples"] = [
        row
        for row in inputs["resource_samples"]
        if row["case_id"] in case_ids
    ]

    result = assemble_bundle(output_root=tmp_path, **inputs)

    assert result["classification"] == "INCONCLUSIVE_PROFILER_OVERHEAD"
    assert result["selected_event_budget"] is None


@pytest.mark.parametrize(
    "script_name",
    (
        "assemble_qwen38_tp4_collective_reduction.py",
        "verify_qwen38_tp4_collective_reduction.py",
    ),
)
def test_bundle_scripts_start_from_an_unrelated_working_directory(
    tmp_path,
    script_name,
):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    script = Path(__file__).with_name(script_name)

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
