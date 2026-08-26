from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.assemble_qwen38_tp4_communication_profile import (
    assemble_bundle,
    build_case_correctness_rows,
    build_profile_case_rows,
)


GPU_UUIDS = [f"GPU-{rank}" for rank in range(4)]
SOURCE_TREE_SHA256 = "a" * 64
MODEL_REVISION = "b" * 40
WORKLOADS = {
    "P0": ("causal", 256, 1),
    "P1": ("causal", 2048, 1),
    "Q0": ("online", 256, 4),
    "Q1": ("online", 256, 8),
    "Q2": ("online", 2048, 4),
}


def _layer(rank: int) -> dict:
    return {
        "layer_index": 0,
        "layer_role": "full_attention",
        "operation_inventory": [
            [0, "gemm", "qkv_projection"],
            [1, "collective", "row_parallel_all_reduce"],
        ],
        "collective_byte_inventory": [[1, 4096]],
        "step_critical_interval_ns": 1000 + rank,
        "gemm_ns": 100,
        "collective_ns": 80,
        "compute_ns": 180,
        "exposed_collective_ns": 50,
        "compute_collective_overlap_ns": 30,
        "gpu_idle_ns": 20,
        "collective_count": 1,
        "collective_bytes": 4096,
        "critical_path_ns": 230,
        "cpu_global_tids": [],
        "stream_ids": [],
    }


def _case(*, phase: str, decode_time_ns: int) -> dict:
    return {
        "attempt": "attempt-a",
        "case_id": f"P0__{phase}__r0",
        "classification": "PASS",
        "workload": "P0",
        "workload_family": "causal",
        "phase": phase,
        "repetition": 0,
        "prompt_tokens": 256,
        "output_tokens": 128,
        "concurrency": 1,
        "decode_time_ns": decode_time_ns,
        "pid": 123,
        "rank_inventory": [0, 1, 2, 3],
        "requests": [{
            "request_id": "request-0",
            "prompt_tokens": 256,
            "generated_tokens": 128,
            "output_token_ids": [11, 22],
            "ttft_ns": 10_000_000,
            "tpot_ns": 2_000_000.0,
            "e2e_ns": 264_000_000,
        }],
        "memory": [
            {
                "rank": rank,
                "peak_allocated_bytes": 1000 + rank,
                "peak_reserved_bytes": 2000 + rank,
            }
            for rank in range(4)
        ],
        "profile": {
            "enabled": True,
            "rank_inventory": [0, 1, 2, 3],
            "ranks": [
                {
                    "enabled": True,
                    "finalization_status": "complete",
                    "rank": rank,
                    "steps": [{
                        "request_set_sha256": "c" * 64,
                        "decode_ordinal": 0,
                        "critical_rank": 3,
                        "final_required_offset_ns": 1200 + rank,
                        "layers": [_layer(rank)],
                    }],
                }
                for rank in range(4)
            ],
        },
    }


def _parsed_trace() -> dict:
    rows = []
    step_rows = []
    for rank in range(4):
        rows.append({
            "attempt": "attempt-a",
            "workload": "P0",
            "repetition": 0,
            "request_set_sha256": "c" * 64,
            "decode_ordinal": 0,
            "rank": rank,
            "layer_index": 0,
            "layer_role": "full_attention",
            "step_critical_interval_ns": 3000 + rank,
            "gemm_ns": 300,
            "collective_ns": 200,
            "compute_ns": 450,
            "exposed_collective_ns": 150,
            "compute_collective_overlap_ns": 50,
            "gpu_idle_ns": 100,
            "collective_count": 1,
            "collective_bytes": 4096,
            "critical_path_ns": 600,
            "cpu_global_tids": [rank + 100],
            "stream_ids": [7, 11],
        })
        step_rows.append({
            "attempt": "attempt-a",
            "workload": "P0",
            "repetition": 0,
            "request_set_sha256": "c" * 64,
            "decode_ordinal": 0,
            "rank": rank,
            "step_critical_interval_ns": 3000 + rank,
            "final_required_offset_ns": 4000 + rank,
        })
    return {
        "classification": "COMPLETE",
        "coverage_errors": [],
        "rows": rows,
        "step_rows": step_rows,
        "critical_rows": [{
            "attempt": "attempt-a",
            "workload": "P0",
            "repetition": 0,
            "request_set_sha256": "c" * 64,
            "decode_ordinal": 0,
            "critical_rank": 3,
            "step_critical_interval_ns": 3003,
            "final_required_offset_ns": 4003,
        }],
    }


def test_measured_profile_rows_use_unprofiled_time_and_nsys_metrics():
    structured = _case(phase="measured", decode_time_ns=9000)
    replay = _case(phase="nsys_replay", decode_time_ns=12000)

    rows = build_profile_case_rows(
        structured,
        sequence_start=20,
        source_tree_sha256=SOURCE_TREE_SHA256,
        model_revision=MODEL_REVISION,
        gpu_uuids=GPU_UUIDS,
        replay_case=replay,
        parsed_trace=_parsed_trace(),
    )

    assert len(rows) == 4
    assert [row["sequence_index"] for row in rows] == [20, 21, 22, 23]
    for rank, row in enumerate(rows):
        assert row["phase"] == "measured"
        assert row["decode_time_ns"] == 9000
        assert row["gpu_uuid"] == GPU_UUIDS[rank]
        assert row["steps"][0]["critical_rank"] == 3
        assert (
            row["steps"][0]["final_required_offset_ns"]
            == 4000 + rank
        )
        layer = row["steps"][0]["layers"][0]
        assert layer["gemm_ns"] == 300
        assert layer["exposed_collective_ns"] == 150
        assert layer["operation_inventory"] == [
            [0, "gemm", "qkv_projection"],
            [1, "collective", "row_parallel_all_reduce"],
        ]


def test_case_correctness_rows_reject_request_output_drift():
    structured = _case(phase="measured", decode_time_ns=9000)
    replay = _case(phase="nsys_replay", decode_time_ns=12000)
    replay["requests"][0]["output_token_ids"] = [11, 99]

    with pytest.raises(ValueError, match="request output mismatch"):
        build_case_correctness_rows(
            structured,
            replay,
            verification={
                "exact_generated_tokens": True,
                "exact_argmax_positions": True,
                "finite_logits_all_ranks": True,
                "within_numeric_tolerance": True,
                "max_abs_logit_error": 0.0,
                "max_rel_logit_error": 0.0,
            },
        )


def test_case_correctness_rows_expand_verified_authority_to_four_ranks():
    structured = _case(phase="measured", decode_time_ns=9000)
    replay = copy.deepcopy(structured)
    replay["phase"] = "nsys_replay"
    replay["case_id"] = "P0__nsys_replay__r0"

    rows = build_case_correctness_rows(
        structured,
        replay,
        verification={
            "exact_generated_tokens": True,
            "exact_argmax_positions": True,
            "finite_logits_all_ranks": True,
            "within_numeric_tolerance": True,
            "max_abs_logit_error": 0.0,
            "max_rel_logit_error": 0.0,
        },
    )

    assert [row["rank"] for row in rows] == [0, 1, 2, 3]
    assert all(row["workload"] == "P0" for row in rows)
    assert all(row["repetition"] == 0 for row in rows)
    assert all(row["exact_token_match"] is True for row in rows)
    assert all(row["argmax_match"] is True for row in rows)
    assert all(row["finite_logits"] is True for row in rows)
    assert all(row["within_numeric_tolerance"] is True for row in rows)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _matrix_case(
    workload: str,
    phase: str,
    repetition: int,
) -> dict:
    family, prompt_tokens, concurrency = WORKLOADS[workload]
    case = _case(
        phase=phase,
        decode_time_ns=1_000_000_000 + repetition * 1_000_000,
    )
    case.update({
        "case_id": f"{workload}__{phase}__r{repetition}",
        "workload": workload,
        "workload_family": family,
        "phase": phase,
        "repetition": repetition,
        "prompt_tokens": prompt_tokens,
        "concurrency": concurrency,
        "requests": [
            {
                "request_id": f"request-{index}",
                "prompt_tokens": prompt_tokens,
                "generated_tokens": 128,
                "output_token_ids": [11, 22],
                "ttft_ns": 10_000_000 + index,
                "tpot_ns": 2_000_000.0 + index,
                "e2e_ns": 264_000_000 + index,
            }
            for index in range(concurrency)
        ],
    })
    for rank_profile in case["profile"]["ranks"]:
        for step in rank_profile["steps"]:
            step["request_set_sha256"] = "c" * 64
    return case


def _write_raw_attempt(root: Path) -> None:
    structured_cases = root / "artifacts/structured/cases"
    replay_cases = root / "artifacts/nsys_replay/cases"
    traces = root / "nsys"
    structured_cases.mkdir(parents=True)
    replay_cases.mkdir(parents=True)
    traces.mkdir()
    for workload in WORKLOADS:
        for phase, repetitions in (
            ("warmup", range(2)),
            ("measured", range(5)),
        ):
            for repetition in repetitions:
                case = _matrix_case(workload, phase, repetition)
                _write_json(
                    structured_cases
                    / f"{workload}__{phase}__r{repetition}.json",
                    case,
                )
                if phase == "measured":
                    replay = copy.deepcopy(case)
                    replay["phase"] = "nsys_replay"
                    replay["case_id"] = (
                        f"{workload}__nsys_replay__r{repetition}"
                    )
                    replay["decode_time_ns"] = int(
                        case["decode_time_ns"] * 1.02
                    )
                    _write_json(
                        replay_cases
                        / (
                            f"{workload}__nsys_replay__"
                            f"r{repetition}.json"
                        ),
                        replay,
                    )
                    (traces / f"{workload}-r{repetition}.sqlite").write_bytes(
                        b"trace"
                    )

    controller = root / "controller"
    controller.mkdir()
    clean = [
        {
            "gpu_index": rank + 2,
            "gpu_uuid": GPU_UUIDS[rank],
            "memory_used_mib": 3,
            "gpu_utilization_percent": 0,
            "power_watts": 70.0 + rank,
            "compute_processes": [],
        }
        for rank in range(4)
    ]
    _write_json(controller / "nsys-admission.json", {
        "selected_gpus": clean,
    })
    _write_json(controller / "environment_identity.json", {
        "source_revision": "d" * 40,
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "model_revision": MODEL_REVISION,
        "torch": "2.7.1+cu126",
        "transformers": "5.8.1",
        "cuda_runtime": "12.6",
        "driver": "535.261.03",
        "nccl": [2, 26, 2],
        "gpu_topology": "GPU2 GPU3 GPU4 GPU5",
    })
    _write_json(controller / "correctness_controller_result.json", {
        "classification": "PASS",
        "verification": {
            "exact_generated_tokens": True,
            "exact_argmax_positions": True,
            "finite_logits_all_ranks": True,
            "within_numeric_tolerance": True,
            "max_abs_logit_error": 0.0,
            "max_rel_logit_error": 0.0,
        },
    })
    resource_rows = [
        {
            "workload": workload,
            "repetition": repetition,
            "gpu_uuid": GPU_UUIDS[rank],
            "gpu_utilization_percent": 80,
            "power_watts": 250.0,
        }
        for workload in WORKLOADS
        for repetition in range(5)
        for rank in range(4)
    ]
    _write_json(controller / "structured-resume-receipt.json", {
        "classification": "PASS",
        "completed_case_count": 35,
        "resource_rows": resource_rows,
    })
    overhead = [
        {
            "workload": workload,
            "repetition": repetition,
            "source_tree_sha256": SOURCE_TREE_SHA256,
            "model_revision": MODEL_REVISION,
            "rank_inventory": [0, 1, 2, 3],
            "gpu_uuids": GPU_UUIDS,
            "unprofiled_ns": 1_000_000_000,
            "profiled_ns": 1_020_000_000,
            "relative_overhead": 0.02,
        }
        for workload in WORKLOADS
        for repetition in range(5)
    ]
    _write_json(controller / "nsys-receipt.json", {
        "classification": "PASS",
        "completed_case_count": 25,
        "overhead_controls": overhead,
        "process_groups_destroyed": True,
        "owned_children_remaining": [],
    })
    correctness = root / "artifacts/correctness"
    correctness.mkdir()
    _write_json(correctness / "model_manifest.json", {
        "schema_version": "tinyllmforge.qwen38-model-manifest.v1",
        "repository": "Qwen/Qwen3.8-27B",
        "resolved_revision": MODEL_REVISION,
    })
    _write_json(correctness / "source_manifest.json", {
        "schema_version": "tinyllmforge.source-manifest.v1",
        "source_tree_sha256": SOURCE_TREE_SHA256,
    })


def _fake_parse(_path: Path, structured_rows: list[dict]) -> dict:
    identity = structured_rows[0]
    workload = identity["workload"]
    repetition = identity["repetition"]
    trace = _parsed_trace()
    for collection in ("rows", "step_rows", "critical_rows"):
        for row in trace[collection]:
            row["workload"] = workload
            row["repetition"] = repetition
    for row in trace["rows"]:
        row.update({
            "gemm_ns": 300,
            "collective_ns": 650,
            "compute_ns": 700,
            "exposed_collective_ns": 600,
            "compute_collective_overlap_ns": 50,
            "gpu_idle_ns": 100,
            "critical_path_ns": 1350,
        })
    return trace


def test_assemble_bundle_writes_complete_producer_inventory(tmp_path):
    attempt = tmp_path / "attempt"
    bundle = tmp_path / "bundle"
    _write_raw_attempt(attempt)
    final_inventory = [
        {
            "gpu_uuid": gpu_uuid,
            "memory_used_mib": 3,
            "utilization_percent": 0,
            "compute_processes": [],
        }
        for gpu_uuid in GPU_UUIDS
    ]
    pci_rows = [
        {
            "gpu_index": rank + 2,
            "gpu_uuid": GPU_UUIDS[rank],
            "pci_bus_id": f"00000000:{rank + 2:02x}:00.0",
        }
        for rank in range(4)
    ]

    summary = assemble_bundle(
        attempt,
        bundle,
        final_gpu_inventory=final_inventory,
        gpu_pci_rows=pci_rows,
        parse_trace=_fake_parse,
    )

    assert summary["classification"] == "GO_COMMUNICATION_OVERLAP"
    assert len((bundle / "profile_rows.jsonl").read_text().splitlines()) == 140
    assert (
        len((bundle / "correctness_rows.jsonl").read_text().splitlines())
        == 100
    )
    assert len(list((bundle / "nsys").glob("*.sqlite"))) == 25
    manifest = json.loads(
        (bundle / "manifest.sha256").read_text(encoding="utf-8")
    )
    assert len(manifest["artifacts"]) == 38
    assert "independent_verification.json" not in manifest["artifacts"]
    report = (bundle / "report.md").read_text(encoding="utf-8")
    assert "Classification: `GO_COMMUNICATION_OVERLAP`" in report
    assert "Profiler overhead" in report
