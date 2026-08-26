#!/usr/bin/env python3
"""Assemble immutable Qwen3.8 TP4 communication-profile evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

if __package__:
    from tools.qwen38_communication_exposure import (
        WORKLOADS,
        aggregate_profile_bundle,
    )
    from tools.qwen38_nsys_intervals import parse_nsys_sqlite
else:
    from qwen38_communication_exposure import (
        WORKLOADS,
        aggregate_profile_bundle,
    )
    from qwen38_nsys_intervals import parse_nsys_sqlite


PROFILE_SCHEMA = "qwen38.communication-profile-row.v1"
MANIFEST_SCHEMA = "qwen38.tp4-communication-profile-manifest.v1"
RANKS = (0, 1, 2, 3)
WARMUP_REPETITIONS = (0, 1)
MEASURED_REPETITIONS = (0, 1, 2, 3, 4)
LAYER_METRICS = (
    "step_critical_interval_ns",
    "gemm_ns",
    "collective_ns",
    "compute_ns",
    "exposed_collective_ns",
    "compute_collective_overlap_ns",
    "gpu_idle_ns",
    "collective_count",
    "collective_bytes",
    "critical_path_ns",
    "cpu_global_tids",
    "stream_ids",
)
EXPECTED_WORKER_IDS = (
    "correctness",
    *(
        f"{workload}__{phase}__r{repetition}"
        for workload in WORKLOADS
        for phase, repetitions in (
            ("warmup", WARMUP_REPETITIONS),
            ("measured", MEASURED_REPETITIONS),
            ("nsys_replay", MEASURED_REPETITIONS),
        )
        for repetition in repetitions
    ),
)


def _rank_profiles(case: dict) -> dict[int, dict]:
    profile = case.get("profile")
    if (
        not isinstance(profile, dict)
        or profile.get("enabled") is not True
        or profile.get("rank_inventory") != list(RANKS)
        or not isinstance(profile.get("ranks"), list)
    ):
        raise ValueError("case profile inventory is invalid")
    ranks = {
        row.get("rank"): row
        for row in profile["ranks"]
        if isinstance(row, dict)
    }
    if set(ranks) != set(RANKS) or len(profile["ranks"]) != len(RANKS):
        raise ValueError("case profile rank inventory is invalid")
    return ranks


def _request_signature(case: dict) -> tuple:
    requests = case.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("case request inventory is invalid")
    return tuple(
        (
            row.get("request_id"),
            row.get("prompt_tokens"),
            row.get("generated_tokens"),
            tuple(row.get("output_token_ids", ())),
        )
        for row in requests
        if isinstance(row, dict)
    )


def _validate_case_pair(structured_case: dict, replay_case: dict) -> None:
    fields = (
        "attempt",
        "workload",
        "workload_family",
        "repetition",
        "prompt_tokens",
        "output_tokens",
        "concurrency",
        "rank_inventory",
    )
    if any(
        structured_case.get(field) != replay_case.get(field)
        for field in fields
    ):
        raise ValueError("structured/Nsight case identity mismatch")
    if (
        structured_case.get("classification") != "PASS"
        or replay_case.get("classification") != "PASS"
        or structured_case.get("phase") != "measured"
        or replay_case.get("phase") != "nsys_replay"
    ):
        raise ValueError("structured/Nsight case status mismatch")
    if _request_signature(structured_case) != _request_signature(replay_case):
        raise ValueError("structured/Nsight request output mismatch")
    if _profile_skeleton(structured_case) != _profile_skeleton(replay_case):
        raise ValueError("structured/Nsight profile skeleton mismatch")


def _profile_skeleton(case: dict) -> tuple:
    ranks = _rank_profiles(case)
    return tuple(
        (
            rank,
            tuple(
                (
                    step.get("request_set_sha256"),
                    step.get("decode_ordinal"),
                    tuple(
                        (
                            layer.get("layer_index"),
                            layer.get("layer_role"),
                            tuple(
                                tuple(operation)
                                for operation in layer.get(
                                    "operation_inventory",
                                    (),
                                )
                            ),
                            tuple(
                                tuple(row)
                                for row in layer.get(
                                    "collective_byte_inventory",
                                    (),
                                )
                            ),
                        )
                        for layer in step.get("layers", ())
                    ),
                )
                for step in ranks[rank].get("steps", ())
            ),
        )
        for rank in RANKS
    )


def _trace_indices(parsed_trace: dict) -> tuple[dict, dict, dict]:
    if (
        parsed_trace.get("classification") != "COMPLETE"
        or parsed_trace.get("coverage_errors") != []
    ):
        raise ValueError("Nsight trace coverage is incomplete")
    layer_rows = {
        (
            row["rank"],
            row["decode_ordinal"],
            row["layer_index"],
            row["layer_role"],
        ): row
        for row in parsed_trace.get("rows", ())
    }
    step_rows = {
        (row["rank"], row["decode_ordinal"]): row
        for row in parsed_trace.get("step_rows", ())
    }
    critical_rows = {
        row["decode_ordinal"]: row
        for row in parsed_trace.get("critical_rows", ())
    }
    return layer_rows, step_rows, critical_rows


def build_profile_case_rows(
    structured_case: dict,
    *,
    sequence_start: int,
    source_tree_sha256: str,
    model_revision: str,
    gpu_uuids: list[str],
    replay_case: dict | None = None,
    parsed_trace: dict | None = None,
) -> list[dict]:
    if len(gpu_uuids) != len(RANKS):
        raise ValueError("GPU UUID inventory is invalid")
    ranks = _rank_profiles(structured_case)
    measured = structured_case.get("phase") == "measured"
    if measured:
        if replay_case is None or parsed_trace is None:
            raise ValueError("measured case requires Nsight evidence")
        _validate_case_pair(structured_case, replay_case)
        replay_ranks = _rank_profiles(replay_case)
        layer_rows, step_rows, critical_rows = _trace_indices(parsed_trace)
    else:
        if structured_case.get("phase") != "warmup":
            raise ValueError("profile case phase is invalid")
        if replay_case is not None or parsed_trace is not None:
            raise ValueError("warmup case must not use Nsight evidence")
        replay_ranks = ranks
        layer_rows = {}
        step_rows = {}
        critical_rows = {}

    result = []
    for rank in RANKS:
        source_rank = replay_ranks[rank] if measured else ranks[rank]
        if (
            source_rank.get("enabled") is not True
            or source_rank.get("finalization_status") != "complete"
        ):
            raise ValueError("case rank profile is incomplete")
        steps = []
        for source_step in source_rank.get("steps", ()):
            decode_ordinal = source_step["decode_ordinal"]
            if measured:
                traced_step = step_rows.get((rank, decode_ordinal))
                critical = critical_rows.get(decode_ordinal)
                if traced_step is None or critical is None:
                    raise ValueError("Nsight step inventory mismatch")
            layers = []
            for source_layer in source_step.get("layers", ()):
                layer = copy.deepcopy(source_layer)
                if measured:
                    key = (
                        rank,
                        decode_ordinal,
                        source_layer["layer_index"],
                        source_layer["layer_role"],
                    )
                    traced_layer = layer_rows.get(key)
                    if traced_layer is None:
                        raise ValueError("Nsight layer inventory mismatch")
                    for field in LAYER_METRICS:
                        layer[field] = traced_layer[field]
                layers.append(layer)
            if not layers:
                raise ValueError("case step layer inventory is empty")
            step = {
                "request_set_sha256": source_step[
                    "request_set_sha256"
                ],
                "decode_ordinal": decode_ordinal,
                "critical_rank": (
                    critical["critical_rank"]
                    if measured
                    else source_step["critical_rank"]
                ),
                "final_required_offset_ns": (
                    traced_step["final_required_offset_ns"]
                    if measured
                    else source_step["final_required_offset_ns"]
                ),
                "layers": layers,
            }
            steps.append(step)
        if not steps:
            raise ValueError("case rank step inventory is empty")
        result.append({
            "schema_version": PROFILE_SCHEMA,
            "sequence_index": sequence_start + rank,
            "attempt": structured_case["attempt"],
            "source_tree_sha256": source_tree_sha256,
            "model_revision": model_revision,
            "workload": structured_case["workload"],
            "workload_family": structured_case["workload_family"],
            "phase": structured_case["phase"],
            "repetition": structured_case["repetition"],
            "rank": rank,
            "gpu_uuid": gpu_uuids[rank],
            "process_identity": (
                f"{structured_case['case_id']}-pid-"
                f"{structured_case['pid']}-rank-{rank}"
            ),
            "finalization_status": "complete",
            "prompt_tokens": structured_case["prompt_tokens"],
            "output_tokens": structured_case["output_tokens"],
            "concurrency": structured_case["concurrency"],
            "decode_time_ns": structured_case["decode_time_ns"],
            "trace_coverage": "COMPLETE",
            "steps": steps,
        })
    return result


def build_case_correctness_rows(
    structured_case: dict,
    replay_case: dict,
    *,
    verification: dict,
) -> list[dict]:
    _validate_case_pair(structured_case, replay_case)
    required = (
        "exact_generated_tokens",
        "exact_argmax_positions",
        "finite_logits_all_ranks",
        "within_numeric_tolerance",
        "max_abs_logit_error",
        "max_rel_logit_error",
    )
    if any(field not in verification for field in required):
        raise ValueError("correctness authority is incomplete")
    return [
        {
            "workload": structured_case["workload"],
            "repetition": structured_case["repetition"],
            "rank": rank,
            "exact_token_match": (
                verification["exact_generated_tokens"] is True
            ),
            "argmax_match": (
                verification["exact_argmax_positions"] is True
            ),
            "finite_logits": (
                verification["finite_logits_all_ranks"] is True
            ),
            "within_numeric_tolerance": (
                verification["within_numeric_tolerance"] is True
            ),
            "max_abs_logit_error": verification[
                "max_abs_logit_error"
            ],
            "max_rel_logit_error": verification[
                "max_rel_logit_error"
            ],
            "structured_case_id": structured_case["case_id"],
            "nsys_case_id": replay_case["case_id"],
            "request_output_match": True,
        }
        for rank in RANKS
    ]


def _read_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _structured_rows(case: dict) -> list[dict]:
    result = []
    for rank, rank_profile in sorted(_rank_profiles(case).items()):
        for step in rank_profile["steps"]:
            for layer in step["layers"]:
                collective_bytes = dict(
                    layer["collective_byte_inventory"]
                )
                for (
                    ordinal,
                    operation_class,
                    operation_name,
                ) in layer["operation_inventory"]:
                    row = {
                        "attempt": case["attempt"],
                        "workload": case["workload"],
                        "repetition": case["repetition"],
                        "request_set_sha256": step[
                            "request_set_sha256"
                        ],
                        "decode_ordinal": step["decode_ordinal"],
                        "rank": rank,
                        "layer_index": layer["layer_index"],
                        "layer_role": layer["layer_role"],
                        "operation_ordinal": ordinal,
                        "operation_class": operation_class,
                        "operation_name": operation_name,
                    }
                    if operation_class == "collective":
                        row["collective_bytes"] = collective_bytes[
                            ordinal
                        ]
                    result.append(row)
    return result


def _online_row(case: dict) -> dict:
    requests = case["requests"]
    return {
        "workload": case["workload"],
        "repetition": case["repetition"],
        "request_count": len(requests),
        "elapsed_s": case["decode_time_ns"] / 1_000_000_000,
        "output_token_count": sum(
            request["generated_tokens"] for request in requests
        ),
        "ttft_ms": [
            request["ttft_ns"] / 1_000_000 for request in requests
        ],
        "tpot_ms": [
            request["tpot_ns"] / 1_000_000 for request in requests
        ],
        "e2e_latency_ms": [
            request["e2e_ns"] / 1_000_000 for request in requests
        ],
    }


def _memory_rows(case: dict) -> list[dict]:
    return [
        {
            "workload": case["workload"],
            "repetition": case["repetition"],
            "rank": row["rank"],
            "peak_allocated_bytes": row["peak_allocated_bytes"],
            "peak_reserved_bytes": row["peak_reserved_bytes"],
        }
        for row in case["memory"]
    ]


def _clean_inventory(rows: list[dict]) -> list[dict]:
    return [
        {
            "gpu_index": row["gpu_index"],
            "gpu_uuid": row["gpu_uuid"],
            "memory_used_mib": row["memory_used_mib"],
            "utilization_percent": row.get(
                "utilization_percent",
                row.get("gpu_utilization_percent"),
            ),
            "compute_processes": row["compute_processes"],
        }
        for row in rows
    ]


def _write_manifest(root: Path) -> None:
    artifacts = {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if (
            path.is_file()
            and path.name != "manifest.sha256"
            and not any(
                part.startswith(".")
                for part in path.relative_to(root).parts
            )
        )
    }
    _write_json(root / "manifest.sha256", {
        "schema_version": MANIFEST_SCHEMA,
        "artifacts": artifacts,
    })


def _write_report(root: Path, summary: dict, source_revision: str) -> None:
    lines = [
        "# Qwen3.8 TP4 Communication Exposure",
        "",
        f"Classification: `{summary['classification']}`",
        "",
        f"Source revision: `{source_revision}`",
        "",
        f"Model revision: `{summary['model_revision']}`",
        "",
        "## Benefit and cost",
        "",
        (
            "Profiler overhead: "
            f"{summary['profiler_overhead_ratio'] * 100:.3f}%"
        ),
        "",
        "| Workload | Exposed communication | Overlap headroom |",
        "| --- | ---: | ---: |",
    ]
    for workload in WORKLOADS:
        payload = summary["workloads"][workload]
        lines.append(
            f"| {workload} | "
            f"{payload['median_exposed_communication_ratio'] * 100:.3f}% | "
            f"{payload['median_overlap_headroom_lower_bound'] * 100:.3f}% |"
        )
    lines.extend((
        "",
        (
            "This baseline gate measures communication exposure only. "
            "It does not itself enable asynchronous collectives."
        ),
        "",
    ))
    (root / "report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def assemble_bundle(
    attempt_root: Path,
    bundle_root: Path,
    *,
    final_gpu_inventory: list[dict],
    gpu_pci_rows: list[dict],
    parse_trace=parse_nsys_sqlite,
) -> dict:
    attempt_root = Path(attempt_root).resolve()
    bundle_root = Path(bundle_root).resolve()
    if not attempt_root.is_dir():
        raise ValueError("attempt root must be an existing directory")
    if bundle_root.exists():
        raise ValueError("bundle root already exists")
    bundle_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{bundle_root.name}.",
        dir=bundle_root.parent,
    ))
    try:
        controller = attempt_root / "controller"
        structured_root = attempt_root / "artifacts/structured/cases"
        replay_root = attempt_root / "artifacts/nsys_replay/cases"
        trace_root = attempt_root / "nsys"
        correctness_root = attempt_root / "artifacts/correctness"
        environment_identity = _read_json(
            controller / "environment_identity.json"
        )
        correctness_result = _read_json(
            controller / "correctness_controller_result.json"
        )
        structured_receipt = _read_json(
            controller / "structured-resume-receipt.json"
        )
        nsys_receipt = _read_json(controller / "nsys-receipt.json")
        admission = _read_json(controller / "nsys-admission.json")
        if correctness_result.get("classification") != "PASS":
            raise ValueError("correctness authority is not PASS")
        if (
            structured_receipt.get("classification") != "PASS"
            or structured_receipt.get("completed_case_count") != 35
        ):
            raise ValueError("structured campaign is incomplete")
        if (
            nsys_receipt.get("classification") != "PASS"
            or nsys_receipt.get("completed_case_count") != 25
        ):
            raise ValueError("Nsight campaign is incomplete")

        selected = _clean_inventory(admission["selected_gpus"])
        gpu_uuids = [row["gpu_uuid"] for row in selected]
        source_tree_sha256 = environment_identity[
            "source_tree_sha256"
        ]
        source_revision = environment_identity["source_revision"]
        model_revision = environment_identity["model_revision"]

        profile_rows = []
        correctness_rows = []
        online_rows = []
        memory_rows = []
        sequence = 0
        for workload in WORKLOADS:
            for repetition in WARMUP_REPETITIONS:
                structured = _read_json(
                    structured_root
                    / f"{workload}__warmup__r{repetition}.json"
                )
                profile_rows.extend(build_profile_case_rows(
                    structured,
                    sequence_start=sequence,
                    source_tree_sha256=source_tree_sha256,
                    model_revision=model_revision,
                    gpu_uuids=gpu_uuids,
                ))
                sequence += len(RANKS)
            for repetition in MEASURED_REPETITIONS:
                structured = _read_json(
                    structured_root
                    / f"{workload}__measured__r{repetition}.json"
                )
                replay = _read_json(
                    replay_root
                    / (
                        f"{workload}__nsys_replay__"
                        f"r{repetition}.json"
                    )
                )
                trace_path = (
                    trace_root / f"{workload}-r{repetition}.sqlite"
                )
                parsed = parse_trace(
                    trace_path,
                    _structured_rows(replay),
                )
                profile_rows.extend(build_profile_case_rows(
                    structured,
                    sequence_start=sequence,
                    source_tree_sha256=source_tree_sha256,
                    model_revision=model_revision,
                    gpu_uuids=gpu_uuids,
                    replay_case=replay,
                    parsed_trace=parsed,
                ))
                sequence += len(RANKS)
                correctness_rows.extend(build_case_correctness_rows(
                    structured,
                    replay,
                    verification=correctness_result["verification"],
                ))
                online_rows.append(_online_row(structured))
                memory_rows.extend(_memory_rows(structured))

        model_manifest = _read_json(
            correctness_root / "model_manifest.json"
        )
        _write_json(temporary / "model_manifest.json", model_manifest)
        source_manifest = _read_json(
            correctness_root / "source_manifest.json"
        )
        source_manifest["source_revision"] = source_revision
        source_manifest["model_manifest_sha256"] = _sha256_file(
            temporary / "model_manifest.json"
        )
        _write_json(temporary / "source_manifest.json", source_manifest)

        cleanup = {
            "process_groups_destroyed": (
                nsys_receipt.get("process_groups_destroyed") is True
            ),
            "rank_exit_codes": [0, 0, 0, 0],
            "owned_children_remaining": nsys_receipt.get(
                "owned_children_remaining",
                [],
            ),
            "final_gpu_inventory": final_gpu_inventory,
        }
        _write_json(temporary / "environment.json", {
            "schema_version": "qwen38.tp4-profile-environment.v1",
            **environment_identity,
            "dtype": "bfloat16",
            "tensor_parallel_size": 4,
            "decoding": "greedy",
            "temperature": 0.0,
            "fixed_output_tokens": 128,
            "scheduler_policy": "identical",
            "cuda_graph_policy": "identical",
            "cleanup": cleanup,
        })
        _write_json(temporary / "gpu_topology.json", {
            "schema_version": "qwen38.tp4-profile-topology.v1",
            "rank_mapping": [
                {
                    "rank": rank,
                    "gpu_index": selected[rank]["gpu_index"],
                    "gpu_uuid": gpu_uuids[rank],
                }
                for rank in RANKS
            ],
            "gpu_rows": gpu_pci_rows,
            "interconnect_matrix": environment_identity["gpu_topology"],
            "controller_entry_inventory": selected,
            "worker_entry_inventories": [
                {
                    "worker_id": worker_id,
                    "gpu_rows": selected,
                }
                for worker_id in EXPECTED_WORKER_IDS
            ],
            "strict_clean_limits": {
                "maximum_memory_used_mib": 1024,
                "maximum_utilization_percent": 5,
                "compute_processes": [],
            },
        })
        _write_json(temporary / "workload_manifest.json", {
            "schema_version": "qwen38.tp4-profile-workloads.v1",
            "source_revision": source_revision,
            "model_revision": model_revision,
            "order": list(WORKLOADS),
            "warmup_repetitions": list(WARMUP_REPETITIONS),
            "measured_repetitions": list(MEASURED_REPETITIONS),
            "rank_inventory": list(RANKS),
            "workloads": WORKLOADS,
        })
        _write_jsonl(temporary / "profile_rows.jsonl", profile_rows)
        _write_jsonl(
            temporary / "correctness_rows.jsonl",
            correctness_rows,
        )
        _write_json(temporary / "online_metrics.json", {
            "schema_version": "qwen38.online-metrics.v1",
            "rows": online_rows,
            "overhead_controls": nsys_receipt["overhead_controls"],
        })
        _write_json(temporary / "memory_summary.json", {
            "schema_version": "qwen38.memory-summary.v1",
            "rows": memory_rows,
        })
        _write_jsonl(
            temporary / "resource_samples.jsonl",
            structured_receipt["resource_rows"],
        )

        destination_traces = temporary / "nsys"
        destination_traces.mkdir()
        for workload in WORKLOADS:
            for repetition in MEASURED_REPETITIONS:
                source = trace_root / f"{workload}-r{repetition}.sqlite"
                if not source.is_file() or source.is_symlink():
                    raise ValueError("Nsight trace inventory is incomplete")
                shutil.copy2(
                    source,
                    destination_traces / source.name,
                )

        summary = aggregate_profile_bundle(temporary)
        _write_json(
            temporary / "communication_exposure_summary.json",
            summary,
        )
        _write_json(temporary / "layer_summary.json", {
            "schema_version": "qwen38.layer-summary.v1",
            "source_tree_sha256": source_tree_sha256,
            "model_revision": model_revision,
            "workloads": {
                workload: payload["layer_summary"]
                for workload, payload in summary["workloads"].items()
            },
        })
        _write_report(temporary, summary, source_revision)
        _write_manifest(temporary)
        temporary.replace(bundle_root)
        return summary
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble a Qwen3.8 TP4 communication-profile bundle."
    )
    parser.add_argument("--attempt-root", required=True, type=Path)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument(
        "--final-gpu-inventory",
        required=True,
        type=Path,
    )
    parser.add_argument("--gpu-pci-rows", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = assemble_bundle(
        args.attempt_root,
        args.bundle_root,
        final_gpu_inventory=_read_json(args.final_gpu_inventory),
        gpu_pci_rows=_read_json(args.gpu_pci_rows),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
