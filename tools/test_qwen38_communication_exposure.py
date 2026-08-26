from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/qwen38_communication_exposure.py"


def _load():
    assert MODULE_PATH.is_file(), (
        "Qwen3.8 communication-exposure aggregator is missing"
    )
    spec = importlib.util.spec_from_file_location(
        "qwen38_communication_exposure_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exposure = _load()


WORKLOADS = {
    "P0": ("causal", 256, 128, 1),
    "P1": ("causal", 2048, 128, 1),
    "Q0": ("online", 256, 128, 4),
    "Q1": ("online", 256, 128, 8),
    "Q2": ("online", 2048, 128, 4),
}


def _layer(
    rank,
    *,
    exposed_ns=120,
    compute_ns=400,
    overlap_ns=100,
):
    return {
        "layer_index": 0,
        "layer_role": "full_attention",
        "operation_inventory": [
            [0, "gemm", "qkv_projection"],
            [1, "collective", "row_parallel_all_reduce"],
            [2, "attention", "flash_attention"],
        ],
        "step_critical_interval_ns": 1000,
        "gemm_ns": 250 + rank,
        "collective_ns": exposed_ns + overlap_ns,
        "compute_ns": compute_ns,
        "exposed_collective_ns": exposed_ns,
        "compute_collective_overlap_ns": overlap_ns,
        "gpu_idle_ns": 80,
        "collective_count": 1,
        "collective_bytes": 40960,
        "critical_path_ns": 700 + rank,
        "cpu_global_tids": [((100 + rank) << 24) | 7],
        "stream_ids": [7, 11],
    }


def _profile_rows(
    *,
    exposed_by_workload=None,
    process_prefix="worker",
):
    exposed_by_workload = exposed_by_workload or {
        workload: [120] * 5
        for workload in WORKLOADS
    }
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
                    exposed_ns = (
                        0
                        if phase == "warmup"
                        else exposed_by_workload[workload][repetition]
                    )
                    rows.append({
                        "schema_version": (
                            "qwen38.communication-profile-row.v1"
                        ),
                        "sequence_index": sequence_index,
                        "attempt": "attempt-a",
                        "source_tree_sha256": "a" * 64,
                        "model_revision": "b" * 40,
                        "workload": workload,
                        "workload_family": family,
                        "phase": phase,
                        "repetition": repetition,
                        "rank": rank,
                        "gpu_uuid": f"GPU-{rank}",
                        "process_identity": (
                            f"{process_prefix}-{workload}-{phase}-"
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
                            "final_required_offset_ns": 900 + rank,
                            "layers": [
                                _layer(
                                    rank,
                                    exposed_ns=exposed_ns,
                                )
                            ],
                        }],
                    })
                    sequence_index += 1
    return rows


def _summary(
    *,
    ratios=None,
    headrooms=None,
):
    ratios = ratios or {
        workload: [0.12] * 5
        for workload in WORKLOADS
    }
    headrooms = headrooms or {
        workload: [0.08] * 5
        for workload in WORKLOADS
    }
    workloads = {}
    for workload, (family, _, _, _) in WORKLOADS.items():
        repetitions = [
            {
                "repetition": repetition,
                "exposed_communication_ratio": (
                    ratios[workload][repetition]
                ),
                "overlap_headroom_lower_bound": (
                    headrooms[workload][repetition]
                ),
            }
            for repetition in range(5)
        ]
        workloads[workload] = {
            "workload_family": family,
            "repetitions": repetitions,
        }
    return {
        "correctness_valid": True,
        "resource_identity_valid": True,
        "trace_coverage_complete": True,
        "complete_four_rank_alignment": True,
        "profiler_overhead_ratio": 0.02,
        "workloads": workloads,
    }


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path, rows):
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_bundle(root, profile_rows):
    _write_jsonl(root / "profile_rows.jsonl", profile_rows)
    online_rows = []
    controls = []
    memory_rows = []
    resource_rows = []
    correctness_rows = []
    for workload in WORKLOADS:
        for repetition in range(5):
            online_rows.append({
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
                "source_tree_sha256": "a" * 64,
                "model_revision": "b" * 40,
                "rank_inventory": [0, 1, 2, 3],
                "gpu_uuids": [f"GPU-{rank}" for rank in range(4)],
                "unprofiled_ns": 1000,
                "profiled_ns": 1020,
            })
            for rank in range(4):
                memory_rows.append({
                    "workload": workload,
                    "repetition": repetition,
                    "rank": rank,
                    "peak_allocated_bytes": (
                        10_000 + repetition * 100 + rank
                    ),
                    "peak_reserved_bytes": (
                        20_000 + repetition * 100 + rank
                    ),
                })
                correctness_rows.append({
                    "workload": workload,
                    "repetition": repetition,
                    "rank": rank,
                    "exact_token_match": True,
                    "argmax_match": True,
                    "finite_logits": True,
                    "within_numeric_tolerance": True,
                    "max_abs_logit_error": 0.005 + rank * 0.001,
                    "max_rel_logit_error": 0.001 + rank * 0.0001,
                })
            resource_rows.extend(
                {
                    "workload": workload,
                    "repetition": repetition,
                    "gpu_uuid": f"GPU-{rank}",
                    "gpu_utilization_percent": (
                        60 + repetition + rank * 5
                    ),
                    "power_watts": 250.0 + repetition + rank * 5,
                }
                for rank in range(4)
            )
    _write_json(
        root / "online_metrics.json",
        {"rows": online_rows, "overhead_controls": controls},
    )
    _write_json(root / "memory_summary.json", {"rows": memory_rows})
    _write_jsonl(root / "resource_samples.jsonl", resource_rows)
    _write_jsonl(root / "correctness_rows.jsonl", correctness_rows)


def test_validate_profile_rows_accepts_exact_matrix_and_alignment():
    validated = exposure.validate_profile_rows(_profile_rows())

    assert validated["workloads"] == list(WORKLOADS)
    assert validated["warmup_repetitions"] == [0, 1]
    assert validated["measured_repetitions"] == [0, 1, 2, 3, 4]
    assert validated["rank_inventory"] == [0, 1, 2, 3]
    assert validated["complete_four_rank_alignment"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda rows: rows.__setitem__(
                slice(None),
                [row for row in rows if row["workload"] != "Q2"],
            ),
            "workload",
        ),
        (
            lambda rows: rows.append(
                rows[0] | {
                    "workload": "Q3",
                    "sequence_index": len(rows),
                }
            ),
            "workload",
        ),
        (
            lambda rows: rows.__setitem__(
                slice(None),
                [
                    row
                    for row in rows
                    if not (
                        row["workload"] == "P0"
                        and row["phase"] == "warmup"
                        and row["repetition"] == 1
                    )
                ],
            ),
            "warmup",
        ),
        (
            lambda rows: rows.__setitem__(
                slice(None),
                [
                    row
                    for row in rows
                    if not (
                        row["workload"] == "P0"
                        and row["phase"] == "measured"
                        and row["repetition"] == 4
                        and row["rank"] == 3
                    )
                ],
            ),
            "rank",
        ),
        (
            lambda rows: rows[73]["steps"][0].update(
                {"request_set_sha256": "d" * 64}
            ),
            "request",
        ),
        (
            lambda rows: rows[73]["steps"][0]["layers"][0].update(
                {"operation_inventory": [[0, "gemm", "other"]]}
            ),
            "operation",
        ),
        (
            lambda rows: next(
                row for row in rows if row["workload"] == "Q1"
            ).update({"concurrency": 4}),
            "workload contract",
        ),
        (
            lambda rows: next(
                row for row in rows if row["workload"] == "P0"
            ).update({"concurrency": True}),
            "workload contract",
        ),
    ),
)
def test_validate_profile_rows_rejects_inventory_drift(
    mutation,
    message,
):
    rows = _profile_rows()
    mutation(rows)

    with pytest.raises(ValueError, match=message):
        exposure.validate_profile_rows(rows)


def test_validate_profile_rows_rejects_process_reuse_after_failure():
    rows = _profile_rows()
    for row in rows:
        row["sequence_index"] += 1
    reused_identity = rows[0]["process_identity"]
    rows.insert(0, {
        **copy.deepcopy(rows[0]),
        "sequence_index": 0,
        "phase": "failed",
        "finalization_status": "failed",
        "steps": [],
    })

    with pytest.raises(ValueError, match="reused after failed finalization"):
        exposure.validate_profile_rows(rows)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda rows: rows[0].update({"schema_version": "wrong"}),
            "schema version",
        ),
        (
            lambda rows: rows[-1].update({"attempt": "attempt-b"}),
            "attempt",
        ),
        (
            lambda rows: [
                row.update({"gpu_uuid": "GPU-0"})
                for row in rows
                if row["rank"] == 1
            ],
            "distinct GPU",
        ),
    ),
)
def test_validate_profile_rows_rejects_identity_drift(mutation, message):
    rows = _profile_rows()
    mutation(rows)

    with pytest.raises(ValueError, match=message):
        exposure.validate_profile_rows(rows)


def test_validate_profile_rows_rejects_incorrect_critical_rank():
    rows = _profile_rows()
    for row in rows:
        if (
            row["workload"] == "P0"
            and row["phase"] == "measured"
            and row["repetition"] == 0
        ):
            row["steps"][0]["critical_rank"] = 0

    with pytest.raises(ValueError, match="critical rank"):
        exposure.validate_profile_rows(rows)


def test_validate_profile_rows_rejects_whole_repetition_inventory_drift():
    rows = _profile_rows()
    for row in rows:
        if (
            row["workload"] == "P0"
            and row["phase"] == "measured"
            and row["repetition"] == 4
        ):
            row["steps"][0]["request_set_sha256"] = "d" * 64
            row["steps"][0]["layers"][0]["operation_inventory"][0][2] = (
                "different_qkv_projection"
            )

    with pytest.raises(ValueError, match="cross-repetition alignment"):
        exposure.validate_profile_rows(rows)


def test_validate_profile_rows_rejects_impossible_layer_duration():
    rows = _profile_rows()
    target = next(
        row
        for row in rows
        if (
            row["workload"] == "P0"
            and row["phase"] == "measured"
            and row["repetition"] == 0
            and row["rank"] == 0
        )
    )
    target["steps"][0]["layers"][0]["gemm_ns"] = 401

    with pytest.raises(ValueError, match="GEMM duration exceeds compute"):
        exposure.validate_profile_rows(rows)


def test_select_representative_repetition_uses_critical_rank_decode_time():
    rows = _profile_rows()
    p0 = [
        row
        for row in rows
        if row["workload"] == "P0" and row["phase"] == "measured"
    ]
    critical_times = [90, 120, 100, 110, 130]
    for row in p0:
        if row["rank"] == 3:
            row["decode_time_ns"] = (
                critical_times[row["repetition"]] * 1_000_000
            )

    assert exposure.select_representative_repetition(p0) == 3


def test_select_representative_repetition_rejects_duplicate_rank():
    rows = [
        row
        for row in _profile_rows()
        if row["workload"] == "P0" and row["phase"] == "measured"
    ]
    duplicate = copy.deepcopy(rows[0])
    duplicate["decode_time_ns"] += 999
    rows.append(duplicate)

    with pytest.raises(ValueError, match="duplicate rank"):
        exposure.select_representative_repetition(rows)


def test_aggregate_profile_bundle_recomputes_metrics_and_costs(tmp_path):
    rows = _profile_rows()
    _write_bundle(tmp_path, rows)

    summary = exposure.aggregate_profile_bundle(tmp_path)

    p0 = summary["workloads"]["P0"]
    assert p0["median_exposed_communication_ratio"] == pytest.approx(0.12)
    assert p0["median_overlap_headroom_lower_bound"] == pytest.approx(0.12)
    assert p0["representative_repetition"] == 2
    assert p0["layer_summary"] == [{
        "layer_index": 0,
        "layer_role": "full_attention",
        "median_gemm_ns": 253,
        "median_collective_ns": 220,
        "median_compute_ns": 400,
        "median_exposed_collective_ns": 120,
        "median_compute_collective_overlap_ns": 100,
        "median_gpu_idle_ns": 80,
        "median_collective_count": 1,
        "median_collective_bytes": 40960,
        "median_critical_path_ns": 703,
    }]
    assert p0["online"]["median_request_qps"] == pytest.approx(2.0)
    assert p0["online"]["median_output_tokens_per_s"] == pytest.approx(256.0)
    assert p0["online"]["ttft_ms"] == pytest.approx({
        "p50": 13.0,
        "p95": 15.55,
        "p99": 15.91,
    })
    assert p0["online"]["tpot_ms"] == pytest.approx({
        "p50": 5.0,
        "p95": 7.55,
        "p99": 7.91,
    })
    assert p0["online"]["e2e_latency_ms"] == pytest.approx({
        "p50": 107.0,
        "p95": 113.55,
        "p99": 113.91,
    })
    assert p0["memory"]["peak_allocated_bytes_by_rank"] == {
        "0": 10400,
        "1": 10401,
        "2": 10402,
        "3": 10403,
    }
    assert p0["memory"]["peak_reserved_bytes_by_rank"] == {
        "0": 20400,
        "1": 20401,
        "2": 20402,
        "3": 20403,
    }
    assert p0["resources"]["gpu_utilization_percent"]["max"] == 79
    assert p0["resources"]["power_watts"]["max"] == pytest.approx(269.0)
    assert summary["correctness_valid"] is True
    assert summary["correctness"] == {
        "row_count": 100,
        "exact_token_match_rows": 100,
        "argmax_match_rows": 100,
        "finite_logit_rows": 100,
        "numeric_tolerance_rows": 100,
        "max_abs_logit_error": pytest.approx(0.008),
        "max_rel_logit_error": pytest.approx(0.0013),
    }
    assert summary["resource_identity_valid"] is True
    assert summary["trace_coverage_complete"] is True
    assert summary["profiler_overhead_ratio"] == pytest.approx(0.02)
    assert summary["classification"] == "GO_COMMUNICATION_OVERLAP"


def test_aggregate_requires_all_five_exposure_repetitions(tmp_path):
    rows = _profile_rows()
    rows = [
        row
        for row in rows
        if not (
            row["workload"] == "P0"
            and row["phase"] == "measured"
            and row["repetition"] == 4
        )
    ]
    _write_bundle(tmp_path, rows)

    with pytest.raises(ValueError, match="measured"):
        exposure.aggregate_profile_bundle(tmp_path)


def test_aggregate_rejects_exposure_larger_than_critical_interval(tmp_path):
    rows = _profile_rows()
    target = next(
        row
        for row in rows
        if (
            row["workload"] == "P0"
            and row["phase"] == "measured"
            and row["repetition"] == 0
            and row["rank"] == 3
        )
    )
    layer = target["steps"][0]["layers"][0]
    layer["exposed_collective_ns"] = 1200
    layer["collective_ns"] = 1300
    _write_bundle(tmp_path, rows)

    with pytest.raises(ValueError, match="critical interval"):
        exposure.aggregate_profile_bundle(tmp_path)


def test_profiler_overhead_controls_require_matching_identity(tmp_path):
    rows = _profile_rows()
    _write_bundle(tmp_path, rows)
    online = json.loads(
        (tmp_path / "online_metrics.json").read_text(encoding="utf-8")
    )
    online["overhead_controls"][0]["gpu_uuids"] = ["GPU-wrong"]
    _write_json(tmp_path / "online_metrics.json", online)

    with pytest.raises(ValueError, match="overhead control identity"):
        exposure.aggregate_profile_bundle(tmp_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda root: (
                lambda payload: (
                    payload["rows"][0].update(
                        {"output_token_count": 255}
                    ),
                    _write_json(root / "online_metrics.json", payload),
                )
            )(
                json.loads(
                    (root / "online_metrics.json").read_text(
                        encoding="utf-8"
                    )
                )
            ),
            "output token count",
        ),
        (
            lambda root: (
                lambda payload: (
                    payload.update({
                        "rows": [
                            row
                            for row in payload["rows"]
                            if not (
                                row["workload"] == "P0"
                                and row["repetition"] == 4
                                and row["rank"] == 3
                            )
                        ]
                    }),
                    _write_json(root / "memory_summary.json", payload),
                )
            )(
                json.loads(
                    (root / "memory_summary.json").read_text(
                        encoding="utf-8"
                    )
                )
            ),
            "memory.*inventory",
        ),
        (
            lambda root: _write_jsonl(
                root / "resource_samples.jsonl",
                [
                    row
                    for row in exposure._read_jsonl(
                        root / "resource_samples.jsonl"
                    )
                    if not (
                        row["workload"] == "P0"
                        and row["gpu_uuid"] == "GPU-3"
                    )
                ],
            ),
            "resource sample inventory",
        ),
    ),
)
def test_aggregate_rejects_incomplete_or_inconsistent_cost_rows(
    tmp_path,
    mutate,
    message,
):
    rows = _profile_rows()
    _write_bundle(tmp_path, rows)
    mutate(tmp_path)

    with pytest.raises(ValueError, match=message):
        exposure.aggregate_profile_bundle(tmp_path)


def test_aggregate_rejects_duplicate_resource_sample(tmp_path):
    rows = _profile_rows()
    _write_bundle(tmp_path, rows)
    resource_path = tmp_path / "resource_samples.jsonl"
    resource_rows = exposure._read_jsonl(resource_path)
    resource_rows.append(copy.deepcopy(resource_rows[0]))
    _write_jsonl(resource_path, resource_rows)

    with pytest.raises(ValueError, match="resource sample inventory"):
        exposure.aggregate_profile_bundle(tmp_path)


def test_aggregate_marks_numeric_correctness_failure_invalid(tmp_path):
    rows = _profile_rows()
    _write_bundle(tmp_path, rows)
    correctness_path = tmp_path / "correctness_rows.jsonl"
    correctness_rows = exposure._read_jsonl(correctness_path)
    correctness_rows[0]["within_numeric_tolerance"] = False
    correctness_rows[0]["max_abs_logit_error"] = 1.0
    _write_jsonl(correctness_path, correctness_rows)

    summary = exposure.aggregate_profile_bundle(tmp_path)

    assert summary["correctness_valid"] is False
    assert summary["classification"] == "INVALID_CORRECTNESS"


@pytest.mark.parametrize(
    "artifact",
    (
        "online",
        "resource",
        "correctness",
        "overhead",
    ),
)
def test_aggregate_rejects_noninteger_auxiliary_repetition(
    tmp_path,
    artifact,
):
    rows = _profile_rows()
    _write_bundle(tmp_path, rows)
    if artifact in {"online", "overhead"}:
        path = tmp_path / "online_metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = "rows" if artifact == "online" else "overhead_controls"
        payload[key][0]["repetition"] = False
        _write_json(path, payload)
    else:
        filename = (
            "resource_samples.jsonl"
            if artifact == "resource"
            else "correctness_rows.jsonl"
        )
        path = tmp_path / filename
        payload = exposure._read_jsonl(path)
        payload[0]["repetition"] = False
        _write_jsonl(path, payload)

    with pytest.raises(ValueError, match="repetition"):
        exposure.aggregate_profile_bundle(tmp_path)


@pytest.mark.parametrize(
    ("filename", "payload_key"),
    (
        ("online_metrics.json", "rows"),
        ("memory_summary.json", "rows"),
        ("resource_samples.jsonl", None),
    ),
)
def test_aggregate_rejects_unknown_auxiliary_workload(
    tmp_path,
    filename,
    payload_key,
):
    rows = _profile_rows()
    _write_bundle(tmp_path, rows)
    path = tmp_path / filename
    if payload_key is None:
        payload = exposure._read_jsonl(path)
        payload.append(copy.deepcopy(payload[0]) | {"workload": "Q3"})
        _write_jsonl(path, payload)
    else:
        document = json.loads(path.read_text(encoding="utf-8"))
        document[payload_key].append(
            copy.deepcopy(document[payload_key][0]) | {"workload": "Q3"}
        )
        _write_json(path, document)

    with pytest.raises(ValueError, match="workload inventory"):
        exposure.aggregate_profile_bundle(tmp_path)


def test_classification_precedence_and_all_terminal_results():
    invalid_correctness = _summary()
    invalid_correctness.update({
        "correctness_valid": False,
        "resource_identity_valid": False,
        "trace_coverage_complete": False,
    })
    assert exposure.classify_communication_exposure(
        invalid_correctness
    ) == "INVALID_CORRECTNESS"

    invalid_resource = _summary()
    invalid_resource.update({
        "resource_identity_valid": False,
        "trace_coverage_complete": False,
    })
    assert exposure.classify_communication_exposure(
        invalid_resource
    ) == "INVALID_RESOURCE_IDENTITY"

    incomplete_trace = _summary()
    incomplete_trace["trace_coverage_complete"] = False
    assert exposure.classify_communication_exposure(
        incomplete_trace
    ) == "INCONCLUSIVE_TRACE_COVERAGE"

    variable_ratios = {
        workload: [0.12] * 5
        for workload in WORKLOADS
    }
    variable_headrooms = {
        workload: [0.08] * 5
        for workload in WORKLOADS
    }
    variable_ratios["P0"] = [0.12, 0.03, 0.12, 0.03, 0.07]
    variable_headrooms["P0"] = [0.08, 0.01, 0.08, 0.01, 0.04]
    assert exposure.classify_communication_exposure(
        _summary(
            ratios=variable_ratios,
            headrooms=variable_headrooms,
        )
    ) == "INCONCLUSIVE_VARIANCE"

    assert exposure.classify_communication_exposure(
        _summary()
    ) == "GO_COMMUNICATION_OVERLAP"

    no_go_ratios = {
        workload: [0.03] * 5
        for workload in WORKLOADS
    }
    no_go_headrooms = {
        workload: [0.01] * 5
        for workload in WORKLOADS
    }
    assert exposure.classify_communication_exposure(
        _summary(
            ratios=no_go_ratios,
            headrooms=no_go_headrooms,
        )
    ) == "NO_GO_ALREADY_HIDDEN"

    low_ratios = {
        workload: [0.07] * 5
        for workload in WORKLOADS
    }
    low_headrooms = {
        workload: [0.04] * 5
        for workload in WORKLOADS
    }
    assert exposure.classify_communication_exposure(
        _summary(
            ratios=low_ratios,
            headrooms=low_headrooms,
        )
    ) == "INCONCLUSIVE_LOW_HEADROOM"


def test_go_requires_causal_and_online_qualifiers_and_low_overhead():
    ratios = {
        workload: [0.07] * 5
        for workload in WORKLOADS
    }
    headrooms = {
        workload: [0.04] * 5
        for workload in WORKLOADS
    }
    ratios["P0"] = [0.12] * 5
    headrooms["P0"] = [0.08] * 5
    only_causal = _summary(ratios=ratios, headrooms=headrooms)
    assert exposure.classify_communication_exposure(
        only_causal
    ) == "INCONCLUSIVE_LOW_HEADROOM"

    ratios["Q0"] = [0.12] * 5
    headrooms["Q0"] = [0.08] * 5
    valid_go = _summary(ratios=ratios, headrooms=headrooms)
    assert exposure.classify_communication_exposure(
        valid_go
    ) == "GO_COMMUNICATION_OVERLAP"

    valid_go["profiler_overhead_ratio"] = 0.031
    assert exposure.classify_communication_exposure(
        valid_go
    ) != "GO_COMMUNICATION_OVERLAP"


def test_classification_threshold_boundaries_are_exact():
    ratios = {
        workload: [0.10] * 5
        for workload in WORKLOADS
    }
    headrooms = {
        workload: [0.05] * 5
        for workload in WORKLOADS
    }
    at_go_boundary = _summary(ratios=ratios, headrooms=headrooms)
    at_go_boundary["profiler_overhead_ratio"] = 0.03
    assert exposure.classify_communication_exposure(
        at_go_boundary
    ) == "GO_COMMUNICATION_OVERLAP"

    ratios = {
        workload: [0.05] * 5
        for workload in WORKLOADS
    }
    headrooms = {
        workload: [0.02] * 5
        for workload in WORKLOADS
    }
    assert exposure.classify_communication_exposure(
        _summary(ratios=ratios, headrooms=headrooms)
    ) == "INCONCLUSIVE_LOW_HEADROOM"


@pytest.mark.parametrize(
    ("ratio", "headroom", "message"),
    (
        (1.01, 0.05, "exposed communication ratio"),
        (0.10, 0.11, "headroom cannot exceed exposure"),
    ),
)
def test_classification_rejects_impossible_ratios(
    ratio,
    headroom,
    message,
):
    summary = _summary()
    summary["workloads"]["P0"]["repetitions"][0].update({
        "exposed_communication_ratio": ratio,
        "overlap_headroom_lower_bound": headroom,
    })

    with pytest.raises(ValueError, match=message):
        exposure.classify_communication_exposure(summary)


def test_classification_rejects_noninteger_repetition_identity():
    summary = _summary()
    summary["workloads"]["P0"]["repetitions"][0]["repetition"] = False

    with pytest.raises(ValueError, match="classification repetition"):
        exposure.classify_communication_exposure(summary)
