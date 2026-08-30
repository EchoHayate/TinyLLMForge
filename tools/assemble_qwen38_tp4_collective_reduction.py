#!/usr/bin/env python3
"""Assemble immutable Qwen3.8 TP4 collective-reduction evidence."""

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
        EXPECTED_DECODE_COLLECTIVE_COUNT,
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
        EXPECTED_DECODE_COLLECTIVE_COUNT,
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
CLASSIFICATION_SCHEMA = (
    "qwen38.tp4-collective-reduction-classification.v1"
)
PRODUCER_ARTIFACTS = (
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
    "manifest.sha256",
)
RANKS = (0, 1, 2, 3)


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value):
    raise ValueError(f"JSON number must be finite: {value}")


def _load_json_strict(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error


def _require_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("numeric evidence must be finite")
    if isinstance(value, dict):
        for child in value.values():
            _require_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _require_finite(child)


def _write_json_atomic(path, payload):
    _require_finite(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _write_jsonl_atomic(path, rows):
    _require_finite(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ))
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


def _validate_source_identity(source_identity):
    if (
        not isinstance(source_identity, dict)
        or source_identity.get("schema_version")
        != "qwen38.tp4-collective-reduction-source.v1"
        or not isinstance(source_identity.get("attempt"), str)
        or not source_identity["attempt"]
        or not isinstance(source_identity.get("source_revision"), str)
        or len(source_identity["source_revision"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in source_identity["source_revision"]
        )
        or not isinstance(source_identity.get("source_tree_sha256"), str)
        or len(source_identity["source_tree_sha256"]) != 64
    ):
        raise ValueError("source identity is invalid")
    return dict(source_identity)


def _validate_model_manifest(model_manifest):
    if (
        not isinstance(model_manifest, dict)
        or model_manifest.get("schema_version")
        != "tinyllmforge.qwen38-model-manifest.v1"
        or model_manifest.get("repository") != "Qwen/Qwen3.8-27B"
        or not isinstance(model_manifest.get("revision"), str)
        or len(model_manifest["revision"]) != 40
        or not isinstance(model_manifest.get("text_profile"), dict)
    ):
        raise ValueError("model manifest is invalid")
    return dict(model_manifest)


def _validate_gpu_topology(gpu_topology):
    if (
        not isinstance(gpu_topology, dict)
        or gpu_topology.get("schema_version")
        != "qwen38.tp4-collective-reduction-topology.v1"
        or not isinstance(gpu_topology.get("rank_mapping"), list)
        or not isinstance(gpu_topology.get("interconnect_matrix"), str)
        or not gpu_topology["interconnect_matrix"]
    ):
        raise ValueError("GPU topology is invalid")
    rows = gpu_topology["rank_mapping"]
    if (
        len(rows) != 4
        or [row.get("rank") for row in rows] != list(RANKS)
        or len({row.get("gpu_uuid") for row in rows}) != 4
        or len({row.get("gpu_index") for row in rows}) != 4
    ):
        raise ValueError("GPU rank map is invalid")
    return dict(gpu_topology)


def _validate_cleanup(cleanup):
    if (
        not isinstance(cleanup, dict)
        or cleanup.get("complete") is not True
        or cleanup.get("process_group_destroyed") is not True
        or cleanup.get("owned_children_remaining") != []
        or cleanup.get("exact_tag_scans") != [[], [], []]
    ):
        raise ValueError("cleanup evidence is incomplete")
    return dict(cleanup)


def _case_id(row):
    return collective_reduction_case_id(**{
        key: row.get(key)
        for key in (
            "campaign_phase",
            "workload",
            "phase",
            "repetition",
            "budget",
        )
    })


def _arm_by_name(case):
    arms = case.get("arms")
    if (
        not isinstance(arms, list)
        or len(arms) != 2
        or {row.get("arm") for row in arms} != {
            "control",
            "instrumented",
        }
    ):
        raise ValueError("case arm inventory is invalid")
    by_arm = {row["arm"]: row for row in arms}
    expected_order = (
        ["control", "instrumented"]
        if case["repetition"] % 2 == 0
        else ["instrumented", "control"]
    )
    if case.get("arm_order") != expected_order:
        raise ValueError("case arm order is invalid")
    return by_arm


def _validate_requests(case, by_arm):
    signatures = []
    rows = []
    for arm_name in ("control", "instrumented"):
        arm = by_arm[arm_name]
        requests = arm.get("requests")
        if (
            not isinstance(requests, list)
            or len(requests) != case["concurrency"]
            or not isinstance(arm.get("decode_time_ns"), (int, float))
            or isinstance(arm.get("decode_time_ns"), bool)
            or not math.isfinite(arm["decode_time_ns"])
            or arm["decode_time_ns"] <= 0
        ):
            raise ValueError("case timing evidence is invalid")
        signature = []
        for request in requests:
            output_ids = request.get("output_token_ids")
            if (
                not isinstance(request, dict)
                or not isinstance(request.get("request_id"), str)
                or not isinstance(output_ids, list)
                or len(output_ids) != case["output_tokens"]
                or any(type(token) is not int for token in output_ids)
            ):
                raise ValueError("case request evidence is invalid")
            for name in ("ttft_ns", "tpot_ns", "e2e_ns"):
                value = request.get(name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError("case request timing is invalid")
            signature.append(
                (request["request_id"], tuple(output_ids))
            )
            rows.append({
                "case_id": case["case_id"],
                "campaign_phase": case["campaign_phase"],
                "phase": case["phase"],
                "workload": case["workload"],
                "repetition": case["repetition"],
                "budget": case["budget"],
                "arm": arm_name,
                **request,
            })
        signatures.append(tuple(signature))
    if signatures[0] != signatures[1]:
        raise ValueError("control/instrumented output mismatch")
    return rows


def _validate_memory(case, by_arm):
    rows = []
    for arm_name, arm in by_arm.items():
        memory = arm.get("memory")
        if (
            not isinstance(memory, list)
            or len(memory) != 4
            or sorted(row.get("rank") for row in memory) != list(RANKS)
        ):
            raise ValueError("case memory rank inventory is invalid")
        for row in memory:
            rows.append({
                "case_id": case["case_id"],
                "arm": arm_name,
                **row,
            })
    return rows


def _calibration_rows(cases):
    by_budget = {budget: [] for budget in EVENT_BUDGETS}
    for case in cases:
        if (
            case["campaign_phase"] == "calibration"
            and case["phase"] == "measured"
        ):
            arms = _arm_by_name(case)
            ratio = (
                arms["instrumented"]["decode_time_ns"]
                / arms["control"]["decode_time_ns"]
                - 1.0
            )
            by_budget[case["budget"]].append({
                "case_id": case["case_id"],
                "workload": case["workload"],
                "repetition": case["repetition"],
                "control_decode_time_ns": (
                    arms["control"]["decode_time_ns"]
                ),
                "instrumented_decode_time_ns": (
                    arms["instrumented"]["decode_time_ns"]
                ),
                "relative_overhead": ratio,
            })
    if any(len(rows) != 15 for rows in by_budget.values()):
        raise ValueError("calibration case coverage is incomplete")
    return [
        {
            "budget": budget,
            "pair_count": len(by_budget[budget]),
            "median_overhead_ratio": median(
                row["relative_overhead"]
                for row in by_budget[budget]
            ),
            "maximum_overhead_ratio": max(
                row["relative_overhead"]
                for row in by_budget[budget]
            ),
            "pairs": by_budget[budget],
        }
        for budget in EVENT_BUDGETS
    ]


def _expected_case_ids(selected_budget):
    matrix = build_collective_reduction_cases(
        selected_budget=16 if selected_budget is None else selected_budget
    )
    rows = list(matrix["calibration"])
    if selected_budget is not None:
        rows.extend(matrix["terminal"])
    return tuple(_case_id(row) for row in rows)


def _validate_case_inventory(cases, source_identity):
    if not isinstance(cases, (list, tuple)):
        raise ValueError("case inventory must be a list")
    normalized = []
    seen = set()
    for raw in cases:
        if not isinstance(raw, dict):
            raise ValueError("case artifact is invalid")
        case = dict(raw)
        if (
            case.get("schema_version")
            != "qwen38.tp4-collective-reduction-worker.v1"
            or case.get("classification") != "PASS"
            or case.get("attempt") != source_identity["attempt"]
            or case.get("source_revision")
            != source_identity["source_revision"]
            or case.get("workload") not in WORKLOADS
            or case.get("campaign_phase")
            not in {"calibration", "terminal"}
            or case.get("phase") not in {"warmup", "measured"}
            or case.get("budget") not in EVENT_BUDGETS
        ):
            raise ValueError("case identity is invalid")
        expected_workload = WORKLOADS[case["workload"]]
        if (
            (
                case.get("workload_family"),
                case.get("prompt_tokens"),
                case.get("output_tokens"),
                case.get("concurrency"),
            )
            != expected_workload
            or case.get("case_id") != _case_id(case)
            or case["case_id"] in seen
        ):
            raise ValueError("case identity is invalid")
        seen.add(case["case_id"])
        by_arm = _arm_by_name(case)
        request_rows = _validate_requests(case, by_arm)
        memory_rows = _validate_memory(case, by_arm)
        normalized.append({
            "case": case,
            "arms": by_arm,
            "request_rows": request_rows,
            "memory_rows": memory_rows,
        })
    calibration = _calibration_rows(
        [row["case"] for row in normalized]
    )
    selected_budget = select_event_budget(calibration)
    expected = _expected_case_ids(selected_budget)
    by_id = {row["case"]["case_id"]: row for row in normalized}
    if set(by_id) != set(expected) or len(by_id) != len(expected):
        raise ValueError("case coverage is incomplete")
    return [by_id[case_id] for case_id in expected], calibration, selected_budget


def _validate_resource_samples(resource_samples, case_ids, gpu_topology):
    if not isinstance(resource_samples, (list, tuple)):
        raise ValueError("resource sample inventory is invalid")
    expected_gpus = {
        (row["gpu_index"], row["gpu_uuid"])
        for row in gpu_topology["rank_mapping"]
    }
    by_case = {}
    normalized = []
    for sample in resource_samples:
        owned_pids = sample.get("owned_pids")
        if (
            not isinstance(sample, dict)
            or sample.get("case_id") not in case_ids
            or sample["case_id"] in by_case
            or not isinstance(owned_pids, list)
            or any(
                type(pid) is not int or pid <= 0
                for pid in owned_pids
            )
            or len(owned_pids) != len(set(owned_pids))
            or not isinstance(sample.get("selected_gpus"), list)
        ):
            raise ValueError("resource sample inventory is invalid")
        owned_pid_set = set(owned_pids)
        gpu_rows = sample["selected_gpus"]
        observed = {
            (row.get("gpu_index"), row.get("gpu_uuid"))
            for row in gpu_rows
            if isinstance(row, dict)
        }
        if (
            len(gpu_rows) != 4
            or observed != expected_gpus
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
            raise ValueError("resource identity is invalid")
        by_case[sample["case_id"]] = sample
        normalized.append(dict(sample))
    if set(by_case) != set(case_ids):
        raise ValueError("resource sample coverage is incomplete")
    return normalized


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
    start = (
        cohort * width
    ) % snapshot["expected_collective_count"]
    return {
        (start + offset) % snapshot["expected_collective_count"]
        for offset in range(snapshot["sample_budget"])
    }


def _validate_sampling(snapshot):
    if (
        snapshot.get("source_revision") is None
        or snapshot.get("attempt") is None
        or snapshot.get("workload") is None
        or type(snapshot.get("repetition")) is not int
        or snapshot.get("sample_budget") not in EVENT_BUDGETS
        or snapshot.get("cohort_count") != 17
        or snapshot.get("expected_collective_count")
        != EXPECTED_DECODE_COLLECTIVE_COUNT
    ):
        raise ValueError("timing cohort identity is invalid")
    for row in snapshot["collectives"]:
        expected = (
            row["collective_ordinal"]
            in _sampled_ordinals(snapshot, row["decode_ordinal"])
        )
        if (
            row.get("event_sampled") is not expected
            or (expected and type(row.get("cuda_ns")) is not int)
            or (not expected and row.get("cuda_ns") is not None)
            or (expected and row["cuda_ns"] < 0)
        ):
            raise ValueError("timing cohort coverage is invalid")


def _derive_terminal_evidence(rows, catalog, selected_budget):
    census_rows = []
    timing_rows = []
    online_rows = []
    correctness_rows = []
    memory_rows = []
    coverage = []
    for normalized in rows:
        case = normalized["case"]
        correctness_rows.extend(normalized["request_rows"])
        memory_rows.extend(normalized["memory_rows"])
        if (
            case["campaign_phase"] != "terminal"
            or case["phase"] != "measured"
        ):
            continue
        arms = normalized["arms"]
        instrumented = arms["instrumented"]
        census = instrumented.get("census")
        if (
            not isinstance(census, dict)
            or census.get("enabled") is not True
            or census.get("rank_inventory") != list(RANKS)
            or not isinstance(census.get("ranks"), list)
        ):
            raise ValueError("terminal census inventory is invalid")
        rank_rows = census["ranks"]
        coverage.append(validate_collective_census(rank_rows, catalog))
        for snapshot in rank_rows:
            _validate_sampling(snapshot)
            census_rows.append({
                "case_id": case["case_id"],
                **snapshot,
            })
            for collective in snapshot["collectives"]:
                if collective["event_sampled"]:
                    timing_rows.append({
                        "case_id": case["case_id"],
                        "workload": case["workload"],
                        "repetition": case["repetition"],
                        "rank": snapshot["rank"],
                        "decode_ordinal": collective["decode_ordinal"],
                        "collective_ordinal": (
                            collective["collective_ordinal"]
                        ),
                        "site_id": collective["site_id"],
                        "cuda_ns": collective["cuda_ns"],
                    })
        for arm_name, arm in arms.items():
            for request in arm["requests"]:
                online_rows.append({
                    "case_id": case["case_id"],
                    "workload": case["workload"],
                    "repetition": case["repetition"],
                    "arm": arm_name,
                    "request_id": request["request_id"],
                    "ttft_ns": request["ttft_ns"],
                    "tpot_ns": request["tpot_ns"],
                    "e2e_ns": request["e2e_ns"],
                })
    if selected_budget is not None and len(coverage) != 25:
        raise ValueError("terminal workload coverage is incomplete")
    return {
        "census_rows": census_rows,
        "timing_rows": timing_rows,
        "online_rows": online_rows,
        "correctness_rows": correctness_rows,
        "memory_rows": memory_rows,
        "coverage": coverage,
    }


def _build_online_metrics(rows, selected_budget):
    terminal = [
        row
        for row in rows
        if (
            row["case"]["campaign_phase"] == "terminal"
            and row["case"]["phase"] == "measured"
        )
    ]
    if selected_budget is None:
        return {
            "schema_version": (
                "qwen38.tp4-collective-reduction-online-metrics.v1"
            ),
            "selected_event_budget": None,
            "pair_count": 0,
            "workloads": [],
            "median_tpot_ns": None,
            "pairs": [],
        }
    pairs = []
    control_tpots = []
    for row in terminal:
        case = row["case"]
        arms = row["arms"]
        control = [
            request["tpot_ns"]
            for request in arms["control"]["requests"]
        ]
        instrumented = [
            request["tpot_ns"]
            for request in arms["instrumented"]["requests"]
        ]
        control_tpots.extend(control)
        pairs.append({
            "case_id": case["case_id"],
            "workload": case["workload"],
            "repetition": case["repetition"],
            "control_decode_time_ns": (
                arms["control"]["decode_time_ns"]
            ),
            "instrumented_decode_time_ns": (
                arms["instrumented"]["decode_time_ns"]
            ),
            "relative_overhead": (
                arms["instrumented"]["decode_time_ns"]
                / arms["control"]["decode_time_ns"]
                - 1.0
            ),
            "control_tpot_ns": median(control),
            "instrumented_tpot_ns": median(instrumented),
        })
    return {
        "schema_version": (
            "qwen38.tp4-collective-reduction-online-metrics.v1"
        ),
        "selected_event_budget": selected_budget,
        "pair_count": len(pairs),
        "workloads": sorted({row["workload"] for row in pairs}),
        "median_tpot_ns": median(control_tpots),
        "pairs": pairs,
    }


def _timing_summary(timing_rows, online):
    embedding = [
        row["cuda_ns"]
        for row in timing_rows
        if row["site_id"] == "embedding.input"
    ]
    if not embedding:
        return {}
    uncertainty = max(
        int(
            pair["control_tpot_ns"]
            * max(0.0, pair["relative_overhead"])
        )
        for pair in online["pairs"]
    )
    return {
        "replicate_embedding": {
            "sampled_collective_cuda_ns": int(median(embedding)),
            "profiler_uncertainty_ns": uncertainty,
        },
    }


def _write_manifest(root):
    artifacts = {
        name: _sha256_file(root / name)
        for name in PRODUCER_ARTIFACTS
        if name != "manifest.sha256"
    }
    _write_json_atomic(root / "manifest.sha256", {
        "schema_version": MANIFEST_SCHEMA,
        "artifacts": artifacts,
    })


def assemble_bundle(
    *,
    output_root,
    source_identity,
    model_manifest,
    gpu_topology,
    cases,
    resource_samples,
    cleanup,
):
    source_identity = _validate_source_identity(source_identity)
    model_manifest = _validate_model_manifest(model_manifest)
    gpu_topology = _validate_gpu_topology(gpu_topology)
    cleanup = _validate_cleanup(cleanup)
    _require_finite({
        "source_identity": source_identity,
        "model_manifest": model_manifest,
        "gpu_topology": gpu_topology,
        "cases": cases,
        "resource_samples": resource_samples,
        "cleanup": cleanup,
    })
    catalog = list(build_qwen38_static_collective_catalog(
        model_manifest["text_profile"],
        tensor_parallel_size=4,
    ))
    proofs = list(build_consumer_dependency_proofs(catalog))
    normalized, calibration, selected_budget = (
        _validate_case_inventory(cases, source_identity)
    )
    case_ids = [row["case"]["case_id"] for row in normalized]
    resources = _validate_resource_samples(
        resource_samples,
        case_ids,
        gpu_topology,
    )
    evidence = _derive_terminal_evidence(
        normalized,
        catalog,
        selected_budget,
    )
    online = _build_online_metrics(normalized, selected_budget)
    coverage_complete = (
        selected_budget is not None
        and len(evidence["coverage"]) == 25
    )
    census_summary = {
        "coverage_complete": coverage_complete,
        "rank_inventory": list(RANKS),
        "case_count": len(evidence["coverage"]),
    }
    ceiling = (
        estimate_reduction_ceiling(
            census_summary,
            _timing_summary(evidence["timing_rows"], online),
            proofs,
            online,
        )
        if selected_budget is not None
        else {"median_tpot_ns": None, "candidates": []}
    )
    classification = classify_collective_reduction({
        "correctness_pass": True,
        "resource_identity_pass": True,
        "coverage_complete": (
            True if selected_budget is None else coverage_complete
        ),
        "profiler_overhead_pass": selected_budget is not None,
        "candidates": ceiling["candidates"],
    })
    workload_manifest = {
        "schema_version": (
            "qwen38.tp4-collective-reduction-workloads.v1"
        ),
        "attempt": source_identity["attempt"],
        "selected_event_budget": selected_budget,
        "calibration_workloads": ["P0", "P1", "Q1"],
        "terminal_workloads": (
            [] if selected_budget is None else list(WORKLOADS)
        ),
        "warmup_repetitions": [0, 1],
        "measured_repetitions": [0, 1, 2, 3, 4],
        "case_ids": case_ids,
    }
    calibration_payload = {
        "schema_version": (
            "qwen38.tp4-collective-reduction-calibration.v1"
        ),
        "rows": calibration,
        "selected_event_budget": selected_budget,
        "median_overhead_ceiling": 0.03,
        "maximum_overhead_ceiling": 0.05,
    }
    classification_payload = {
        "schema_version": CLASSIFICATION_SCHEMA,
        "classification": classification,
        "selected_event_budget": selected_budget,
        "coverage_complete": coverage_complete,
        "correctness_pass": True,
        "resource_identity_pass": True,
        "cleanup_pass": True,
        "profiler_overhead_pass": selected_budget is not None,
        "minimum_lower_bound_opportunity": 0.05,
    }

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("bundle output directory must be empty")
    _write_json_atomic(root / "source_identity.json", source_identity)
    _write_json_atomic(root / "model_manifest.json", model_manifest)
    _write_json_atomic(root / "gpu_topology.json", gpu_topology)
    _write_json_atomic(root / "workload_manifest.json", workload_manifest)
    _write_json_atomic(
        root / "static_collective_catalog.json",
        {
            "schema_version": (
                "qwen38.tp4-static-collective-catalog.v1"
            ),
            "rows": catalog,
        },
    )
    _write_json_atomic(
        root / "consumer_dependency_proofs.json",
        {
            "schema_version": (
                "qwen38.tp4-consumer-dependency-proofs.v1"
            ),
            "rows": proofs,
        },
    )
    _write_json_atomic(
        root / "profiler_calibration.json",
        calibration_payload,
    )
    _write_jsonl_atomic(
        root / "collective_census.jsonl",
        evidence["census_rows"],
    )
    _write_jsonl_atomic(
        root / "collective_timing_samples.jsonl",
        evidence["timing_rows"],
    )
    _write_json_atomic(
        root / "paired_online_metrics.json",
        online,
    )
    _write_jsonl_atomic(
        root / "correctness.jsonl",
        evidence["correctness_rows"],
    )
    _write_jsonl_atomic(
        root / "resource_samples.jsonl",
        resources,
    )
    _write_json_atomic(root / "reduction_ceiling.json", ceiling)
    _write_json_atomic(
        root / "classification.json",
        classification_payload,
    )
    _write_json_atomic(root / "cleanup.json", cleanup)
    _write_manifest(root)
    return {
        "classification": classification,
        "selected_event_budget": selected_budget,
        "bundle_root": str(root),
        "artifact_count": len(PRODUCER_ARTIFACTS),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", required=True, type=Path)
    parser.add_argument("--bundle-root", type=Path)
    args = parser.parse_args(argv)
    attempt_root = args.attempt_root.resolve()
    bundle_root = (
        args.bundle_root.resolve()
        if args.bundle_root is not None
        else attempt_root / "final_bundle"
    )
    cases = [
        _load_json_strict(path)
        for path in sorted((attempt_root / "cases").glob("*.json"))
    ]
    result = assemble_bundle(
        output_root=bundle_root,
        source_identity=_load_json_strict(
            attempt_root / "controller/source_identity.json"
        ),
        model_manifest=_load_json_strict(
            attempt_root / "controller/model_manifest.json"
        ),
        gpu_topology=_load_json_strict(
            attempt_root / "controller/gpu_topology.json"
        ),
        cases=cases,
        resource_samples=_load_json_strict(
            attempt_root / "controller/resource_samples.json"
        ),
        cleanup=_load_json_strict(
            attempt_root / "controller/cleanup.json"
        ),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
