#!/usr/bin/env python3
"""Independently verify a Qwen3.8 TP4 collective-reduction bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import tempfile

if __package__:
    from tools.qwen38_collective_reduction import (
        EVENT_BUDGETS,
        build_consumer_dependency_proofs,
        build_qwen38_static_collective_catalog,
        classify_collective_reduction,
        estimate_reduction_ceiling,
        select_event_budget,
        validate_collective_census,
    )
    from tools.qwen38_tp4_collective_reduction_worker import (
        WORKLOADS,
        build_collective_reduction_cases,
        collective_reduction_case_id,
    )
else:
    from qwen38_collective_reduction import (
        EVENT_BUDGETS,
        build_consumer_dependency_proofs,
        build_qwen38_static_collective_catalog,
        classify_collective_reduction,
        estimate_reduction_ceiling,
        select_event_budget,
        validate_collective_census,
    )
    from qwen38_tp4_collective_reduction_worker import (
        WORKLOADS,
        build_collective_reduction_cases,
        collective_reduction_case_id,
    )


MANIFEST_SCHEMA = "qwen38.tp4-collective-reduction-manifest.v1"
VERIFICATION_SCHEMA = (
    "qwen38.tp4-collective-reduction-independent-verification.v1"
)
PRODUCER_FILES = frozenset({
    "source_identity.json",
    "model_manifest.json",
    "gpu_topology.json",
    "workload_manifest.json",
    "static_collective_catalog.json",
    "consumer_dependency_proofs.json",
    "profiler_calibration.json",
    "collective_census.jsonl",
    "collective_timing_samples.jsonl",
    "paired_online_metrics.json",
    "correctness.jsonl",
    "resource_samples.jsonl",
    "reduction_ceiling.json",
    "classification.json",
    "cleanup.json",
})
RANKS = (0, 1, 2, 3)


def _duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"JSON number must be finite: {value}")


def _load_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_duplicate_keys,
                parse_constant=_nonfinite,
            )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error


def _load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(
                    line,
                    object_pairs_hook=_duplicate_keys,
                    parse_constant=_nonfinite,
                ))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from error
    return rows


def _write_json_atomic(path, payload):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
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
    temporary.replace(path)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(root):
    manifest = _load_json(root / "manifest.sha256")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or not isinstance(manifest.get("artifacts"), dict)
    ):
        raise ValueError("manifest is invalid")
    actual_files = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "manifest.sha256"
    }
    allowed = PRODUCER_FILES | {"independent_verification.json"}
    if (
        actual_files not in {
            PRODUCER_FILES,
            PRODUCER_FILES | {"independent_verification.json"},
        }
        or set(manifest["artifacts"]) != actual_files
        or not actual_files.issubset(allowed)
    ):
        raise ValueError("manifest artifact inventory mismatch")
    for name, expected in manifest["artifacts"].items():
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or _sha256_file(root / name) != expected
        ):
            raise ValueError("manifest artifact hash mismatch")


def _expected_rows(selected_budget):
    matrix = build_collective_reduction_cases(
        selected_budget=16 if selected_budget is None else selected_budget
    )
    rows = list(matrix["calibration"])
    if selected_budget is not None:
        rows.extend(matrix["terminal"])
    return rows


def _case_id(row):
    return collective_reduction_case_id(**{
        key: row[key]
        for key in (
            "campaign_phase",
            "workload",
            "phase",
            "repetition",
            "budget",
        )
    })


def _verify_identity(root):
    source = _load_json(root / "source_identity.json")
    if (
        source.get("schema_version")
        != "qwen38.tp4-collective-reduction-source.v1"
        or not isinstance(source.get("attempt"), str)
        or not isinstance(source.get("source_revision"), str)
        or len(source["source_revision"]) != 40
        or not isinstance(source.get("source_tree_sha256"), str)
        or len(source["source_tree_sha256"]) != 64
    ):
        raise ValueError("source identity is invalid")
    model = _load_json(root / "model_manifest.json")
    if (
        model.get("schema_version")
        != "tinyllmforge.qwen38-model-manifest.v1"
        or model.get("repository") != "Qwen/Qwen3.8-27B"
        or not isinstance(model.get("revision"), str)
        or len(model["revision"]) != 40
        or not isinstance(model.get("text_profile"), dict)
    ):
        raise ValueError("model identity is invalid")
    topology = _load_json(root / "gpu_topology.json")
    mapping = topology.get("rank_mapping")
    if (
        topology.get("schema_version")
        != "qwen38.tp4-collective-reduction-topology.v1"
        or not isinstance(mapping, list)
        or len(mapping) != 4
        or [row.get("rank") for row in mapping] != list(RANKS)
        or len({row.get("gpu_index") for row in mapping}) != 4
        or len({row.get("gpu_uuid") for row in mapping}) != 4
        or not isinstance(topology.get("interconnect_matrix"), str)
        or not topology["interconnect_matrix"]
    ):
        raise ValueError("GPU rank map is invalid")
    return source, model, topology


def _verify_catalog(root, model):
    expected = list(build_qwen38_static_collective_catalog(
        model["text_profile"],
        tensor_parallel_size=4,
    ))
    catalog = _load_json(root / "static_collective_catalog.json")
    if (
        catalog.get("schema_version")
        != "qwen38.tp4-static-collective-catalog.v1"
        or catalog.get("rows") != expected
    ):
        raise ValueError("static catalog mismatch")
    expected_proofs = list(build_consumer_dependency_proofs(expected))
    proofs = _load_json(root / "consumer_dependency_proofs.json")
    if (
        proofs.get("schema_version")
        != "qwen38.tp4-consumer-dependency-proofs.v1"
        or proofs.get("rows") != expected_proofs
    ):
        raise ValueError("consumer proof mismatch")
    return expected, expected_proofs


def _verify_calibration(root):
    payload = _load_json(root / "profiler_calibration.json")
    rows = payload.get("rows")
    if (
        payload.get("schema_version")
        != "qwen38.tp4-collective-reduction-calibration.v1"
        or not isinstance(rows, list)
        or [row.get("budget") for row in rows] != list(EVENT_BUDGETS)
        or payload.get("median_overhead_ceiling") != 0.03
        or payload.get("maximum_overhead_ceiling") != 0.05
    ):
        raise ValueError("profiler calibration is invalid")
    reconstructed = []
    for row in rows:
        pairs = row.get("pairs")
        if not isinstance(pairs, list) or len(pairs) != 15:
            raise ValueError("profiler calibration coverage is invalid")
        ratios = []
        identities = set()
        for pair in pairs:
            control = pair.get("control_decode_time_ns")
            instrumented = pair.get("instrumented_decode_time_ns")
            if (
                isinstance(control, bool)
                or not isinstance(control, (int, float))
                or not math.isfinite(control)
                or control <= 0
                or isinstance(instrumented, bool)
                or not isinstance(instrumented, (int, float))
                or not math.isfinite(instrumented)
                or instrumented <= 0
            ):
                raise ValueError("profiler calibration timing is invalid")
            ratio = instrumented / control - 1.0
            if pair.get("relative_overhead") != ratio:
                raise ValueError("profiler overhead reconstruction mismatch")
            identity = (pair.get("workload"), pair.get("repetition"))
            if identity in identities:
                raise ValueError("profiler calibration pair duplicate")
            identities.add(identity)
            ratios.append(ratio)
        rebuilt = {
            "budget": row["budget"],
            "pair_count": 15,
            "median_overhead_ratio": median(ratios),
            "maximum_overhead_ratio": max(ratios),
            "pairs": pairs,
        }
        if row != rebuilt:
            raise ValueError("profiler overhead aggregate mismatch")
        reconstructed.append(rebuilt)
    selected = select_event_budget(reconstructed)
    if payload.get("selected_event_budget") != selected:
        raise ValueError("selected event budget mismatch")
    return reconstructed, selected


def _verify_workloads(root, source, selected_budget):
    payload = _load_json(root / "workload_manifest.json")
    rows = _expected_rows(selected_budget)
    expected_ids = [_case_id(row) for row in rows]
    expected = {
        "schema_version": (
            "qwen38.tp4-collective-reduction-workloads.v1"
        ),
        "attempt": source["attempt"],
        "selected_event_budget": selected_budget,
        "calibration_workloads": ["P0", "P1", "Q1"],
        "terminal_workloads": (
            [] if selected_budget is None else list(WORKLOADS)
        ),
        "warmup_repetitions": [0, 1],
        "measured_repetitions": [0, 1, 2, 3, 4],
        "case_ids": expected_ids,
    }
    if payload != expected:
        raise ValueError("workload matrix mismatch")
    return rows, expected_ids


def _sampled_ordinals(snapshot, decode_ordinal):
    seed = (
        f"{snapshot['source_revision']}\0{snapshot['attempt']}\0"
        f"{snapshot['workload']}\0{snapshot['repetition']}\0"
        f"{decode_ordinal}"
    ).encode("utf-8")
    cohort = int.from_bytes(
        hashlib.sha256(seed).digest()[:8],
        "big",
    ) % snapshot["cohort_count"]
    width = math.ceil(
        snapshot["expected_collective_count"]
        / snapshot["cohort_count"]
    )
    start = cohort * width % snapshot["expected_collective_count"]
    return {
        (start + offset) % snapshot["expected_collective_count"]
        for offset in range(snapshot["sample_budget"])
    }


def _verify_census_and_timing(
    root,
    source,
    catalog,
    selected_budget,
):
    census_rows = _load_jsonl(root / "collective_census.jsonl")
    timing_rows = _load_jsonl(
        root / "collective_timing_samples.jsonl"
    )
    expected_terminal = {
        _case_id(row)
        for row in _expected_rows(selected_budget)
        if (
            row["campaign_phase"] == "terminal"
            and row["phase"] == "measured"
        )
    }
    by_case = {}
    reconstructed_timing = []
    for snapshot in census_rows:
        case_id = snapshot.get("case_id")
        if (
            case_id not in expected_terminal
            or snapshot.get("source_revision")
            != source["source_revision"]
            or snapshot.get("attempt") != source["attempt"]
            or snapshot.get("sample_budget") != selected_budget
        ):
            raise ValueError("collective census identity mismatch")
        by_case.setdefault(case_id, []).append(snapshot)
        for collective in snapshot.get("collectives", ()):
            sampled = (
                collective.get("collective_ordinal")
                in _sampled_ordinals(
                    snapshot,
                    collective.get("decode_ordinal"),
                )
            )
            if (
                collective.get("event_sampled") is not sampled
                or (
                    sampled
                    and (
                        type(collective.get("cuda_ns")) is not int
                        or collective["cuda_ns"] < 0
                    )
                )
                or (
                    not sampled
                    and collective.get("cuda_ns") is not None
                )
            ):
                raise ValueError("timing cohort mismatch")
            if sampled:
                reconstructed_timing.append({
                    "case_id": case_id,
                    "workload": snapshot["workload"],
                    "repetition": snapshot["repetition"],
                    "rank": snapshot["rank"],
                    "decode_ordinal": collective["decode_ordinal"],
                    "collective_ordinal": (
                        collective["collective_ordinal"]
                    ),
                    "site_id": collective["site_id"],
                    "cuda_ns": collective["cuda_ns"],
                })
    if set(by_case) != expected_terminal:
        raise ValueError("collective census coverage is incomplete")
    coverage = [
        validate_collective_census(by_case[case_id], catalog)
        for case_id in sorted(by_case)
    ]
    if timing_rows != reconstructed_timing:
        raise ValueError("collective timing samples mismatch")
    return timing_rows, coverage


def _verify_online(root, selected_budget):
    payload = _load_json(root / "paired_online_metrics.json")
    pairs = payload.get("pairs")
    if (
        payload.get("schema_version")
        != "qwen38.tp4-collective-reduction-online-metrics.v1"
        or payload.get("selected_event_budget") != selected_budget
        or not isinstance(pairs, list)
    ):
        raise ValueError("paired online metrics are invalid")
    if selected_budget is None:
        if payload.get("pair_count") != 0 or pairs:
            raise ValueError("paired online metrics are invalid")
        return payload
    expected_ids = {
        _case_id(row)
        for row in _expected_rows(selected_budget)
        if (
            row["campaign_phase"] == "terminal"
            and row["phase"] == "measured"
        )
    }
    if (
        len(pairs) != 25
        or payload.get("pair_count") != 25
        or {row.get("case_id") for row in pairs} != expected_ids
        or payload.get("workloads") != sorted(WORKLOADS)
        or not isinstance(payload.get("median_tpot_ns"), (int, float))
        or payload["median_tpot_ns"] <= 0
    ):
        raise ValueError("paired online coverage is incomplete")
    for pair in pairs:
        control = pair.get("control_decode_time_ns")
        instrumented = pair.get("instrumented_decode_time_ns")
        if (
            not isinstance(control, (int, float))
            or isinstance(control, bool)
            or control <= 0
            or not isinstance(instrumented, (int, float))
            or isinstance(instrumented, bool)
            or instrumented <= 0
            or pair.get("relative_overhead")
            != instrumented / control - 1.0
        ):
            raise ValueError("paired online timing mismatch")
    return payload


def _verify_correctness(root, expected_rows):
    rows = _load_jsonl(root / "correctness.jsonl")
    expected = {
        _case_id(row): row for row in expected_rows
    }
    grouped = {}
    for row in rows:
        key = (row.get("case_id"), row.get("request_id"))
        grouped.setdefault(key, {})[row.get("arm")] = row
    for (case_id, _request_id), arms in grouped.items():
        if (
            case_id not in expected
            or set(arms) != {"control", "instrumented"}
            or arms["control"].get("output_token_ids")
            != arms["instrumented"].get("output_token_ids")
            or len(arms["control"].get("output_token_ids", ()))
            != expected[case_id]["output_tokens"]
        ):
            raise ValueError("correctness evidence mismatch")
    expected_pairs = sum(
        row["concurrency"] for row in expected_rows
    )
    if len(grouped) != expected_pairs or len(rows) != 2 * expected_pairs:
        raise ValueError("correctness coverage is incomplete")
    return rows


def _verify_resources(root, expected_ids, topology):
    rows = _load_jsonl(root / "resource_samples.jsonl")
    mapping = {
        (row["gpu_index"], row["gpu_uuid"])
        for row in topology["rank_mapping"]
    }
    if (
        len(rows) != len(expected_ids)
        or {row.get("case_id") for row in rows} != set(expected_ids)
    ):
        raise ValueError("resource sample coverage is incomplete")
    for sample in rows:
        owned_pids = sample.get("owned_pids")
        gpu_rows = sample.get("selected_gpus")
        owned_pid_set = (
            set(owned_pids) if isinstance(owned_pids, list) else set()
        )
        if (
            not isinstance(owned_pids, list)
            or any(
                type(pid) is not int or pid <= 0
                for pid in owned_pids
            )
            or len(owned_pids) != len(set(owned_pids))
            or not isinstance(gpu_rows, list)
            or len(gpu_rows) != 4
            or {
                (row.get("gpu_index"), row.get("gpu_uuid"))
                for row in gpu_rows
            }
            != mapping
            or any(
                not isinstance(row, dict)
                or type(row.get("memory_used_mib")) is not int
                or row["memory_used_mib"] < 0
                or type(row.get("utilization_percent")) is not int
                or not 0 <= row["utilization_percent"] <= 100
                or not isinstance(row.get("compute_processes"), list)
                or any(
                    not isinstance(process, dict)
                    or type(process.get("pid")) is not int
                    or process["pid"] not in owned_pid_set
                    for process in row["compute_processes"]
                )
                for row in gpu_rows
            )
        ):
            raise ValueError("resource identity mismatch")
    return rows


def _verify_cleanup(root):
    cleanup = _load_json(root / "cleanup.json")
    if (
        cleanup.get("schema_version")
        != "qwen38.tp4-collective-reduction-cleanup.v1"
        or cleanup.get("complete") is not True
        or cleanup.get("process_group_destroyed") is not True
        or cleanup.get("owned_children_remaining") != []
        or cleanup.get("exact_tag_scans") != [[], [], []]
    ):
        raise ValueError("cleanup evidence mismatch")
    return cleanup


def _rebuild_ceiling(timing_rows, online, proofs, coverage):
    embedding = [
        row["cuda_ns"]
        for row in timing_rows
        if row["site_id"] == "embedding.input"
    ]
    timing = {}
    if embedding:
        timing["replicate_embedding"] = {
            "sampled_collective_cuda_ns": int(median(embedding)),
            "profiler_uncertainty_ns": max(
                int(
                    pair["control_tpot_ns"]
                    * max(0.0, pair["relative_overhead"])
                )
                for pair in online["pairs"]
            ),
        }
    return estimate_reduction_ceiling(
        {
            "coverage_complete": bool(coverage),
        },
        timing,
        proofs,
        online,
    )


def _rewrite_manifest(root):
    artifacts = {
        path.name: _sha256_file(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "manifest.sha256"
    }
    _write_json_atomic(root / "manifest.sha256", {
        "schema_version": MANIFEST_SCHEMA,
        "artifacts": artifacts,
    })


def verify_bundle(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("bundle root must be an existing directory")
    _verify_manifest(root)
    source, model, topology = _verify_identity(root)
    catalog, proofs = _verify_catalog(root, model)
    calibration, selected_budget = _verify_calibration(root)
    expected_rows, expected_ids = _verify_workloads(
        root,
        source,
        selected_budget,
    )
    timing_rows, coverage = _verify_census_and_timing(
        root,
        source,
        catalog,
        selected_budget,
    )
    online = _verify_online(root, selected_budget)
    correctness = _verify_correctness(root, expected_rows)
    resources = _verify_resources(root, expected_ids, topology)
    _verify_cleanup(root)
    if selected_budget is None:
        ceiling = {"median_tpot_ns": None, "candidates": []}
    else:
        ceiling = _rebuild_ceiling(
            timing_rows,
            online,
            proofs,
            coverage,
        )
    if _load_json(root / "reduction_ceiling.json") != ceiling:
        raise ValueError("reduction ceiling mismatch")
    reconstructed = classify_collective_reduction({
        "correctness_pass": True,
        "resource_identity_pass": True,
        "coverage_complete": (
            True if selected_budget is None else len(coverage) == 25
        ),
        "profiler_overhead_pass": selected_budget is not None,
        "candidates": ceiling["candidates"],
    })
    producer = _load_json(root / "classification.json")
    expected_classification = {
        "schema_version": (
            "qwen38.tp4-collective-reduction-classification.v1"
        ),
        "classification": reconstructed,
        "selected_event_budget": selected_budget,
        "coverage_complete": (
            selected_budget is not None and len(coverage) == 25
        ),
        "correctness_pass": True,
        "resource_identity_pass": True,
        "cleanup_pass": True,
        "profiler_overhead_pass": selected_budget is not None,
        "minimum_lower_bound_opportunity": 0.05,
    }
    if producer != expected_classification:
        raise ValueError("producer classification mismatch")
    result = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS",
        "source_revision": source["source_revision"],
        "source_tree_sha256": source["source_tree_sha256"],
        "model_revision": model["revision"],
        "rank_inventory": list(RANKS),
        "gpu_uuids": [
            row["gpu_uuid"] for row in topology["rank_mapping"]
        ],
        "selected_event_budget": selected_budget,
        "calibration_pair_count": sum(
            row["pair_count"] for row in calibration
        ),
        "terminal_case_count": len(coverage),
        "census_rank_snapshot_count": len(coverage) * 4,
        "timing_sample_count": len(timing_rows),
        "correctness_row_count": len(correctness),
        "resource_sample_count": len(resources),
        "artifact_hashes_verified": True,
        "complete_four_rank_alignment": (
            selected_budget is None or len(coverage) == 25
        ),
        "timing_cohort_coverage_complete": True,
        "correctness_valid": True,
        "resource_identity_valid": True,
        "cleanup_valid": True,
        "producer_classification": reconstructed,
        "reconstructed_classification": reconstructed,
    }
    _write_json_atomic(
        root / "independent_verification.json",
        result,
    )
    _rewrite_manifest(root)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True, type=Path)
    args = parser.parse_args(argv)
    result = verify_bundle(args.bundle_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
