from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
GATE_PATH = HERE / "kvcapacity_kv8_quest_adaptive_gate.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "kvcapacity_kv8_quest_adaptive_gate_under_test",
        GATE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


gate = _load_gate()

BATCHES = (4, 6, 8, 10, 12, 16, 19)
REVISION = "a" * 40


def _activation_event(
    observation_id,
    *,
    top_k,
    min_saved_blocks,
    batch,
):
    if top_k <= 0:
        reason = "disabled"
        resolved = -1
        saved = None
    else:
        saved = batch * 16
        active = min_saved_blocks == 0 or saved >= min_saved_blocks
        reason = "active" if active else "below_saved_blocks"
        resolved = top_k if active else -1
    return {
        "status": "valid",
        "observation_id": observation_id,
        "requested_top_k": top_k,
        "resolved_top_k": resolved,
        "min_seq_len": 512,
        "min_saved_blocks": min_saved_blocks,
        "saved_blocks": saved,
        "batch_size": batch,
        "sequence_lengths": [8192] * batch,
        "sequence_block_counts": [32] * batch,
        "reason": reason,
    }


def make_perf(bits, top_k, min_saved_blocks, times):
    rows = []
    for batch, step_ms in zip(BATCHES, times):
        events = [
            _activation_event(
                index + 1,
                top_k=top_k,
                min_saved_blocks=min_saved_blocks,
                batch=batch,
            )
            for index in range(24)
        ]
        reason_counts = {}
        resolved_counts = {}
        saved = []
        for event in events:
            reason_counts[event["reason"]] = (
                reason_counts.get(event["reason"], 0) + 1
            )
            key = str(event["resolved_top_k"])
            resolved_counts[key] = resolved_counts.get(key, 0) + 1
            if event["saved_blocks"] is not None:
                saved.append(event["saved_blocks"])
        rows.append(
            {
                "context_length": 8192,
                "batch": batch,
                "measured": True,
                "observed_batch_is_stable": True,
                "prefill_tokens_match": True,
                "prompt_digests": [
                    f"prompt-{batch}-{index}"
                    for index in range(batch)
                ],
                "step": {"median_ms": step_ms, "count": 24},
                "dispatch_measured": {
                    "all_graph": False,
                    "graph_steps": 0,
                    "steps": 24,
                    "counts": {"eager:feature_disabled": 24},
                },
                "quest_activation_measured": {
                    "steps": 24,
                    "status_counts": {"valid": 24},
                    "reason_counts": reason_counts,
                    "resolved_top_k_counts": resolved_counts,
                    "saved_blocks_min": min(saved) if saved else None,
                    "saved_blocks_max": max(saved) if saved else None,
                    "all_valid": True,
                },
                "quest_activation_measured_events": events,
            }
        )
    return {
        "grid_spec": "8192:4,6,8,10,12,16,19",
        "configuration": {
            "enforce_eager": True,
            "gpu_memory_utilization": 0.85,
            "kv_blocks_requested": 640,
            "kv_quant_bits": bits,
            "measured_steps": 24,
            "model_path": "/models/Qwen3-8B",
            "quest_min_saved_blocks": min_saved_blocks,
            "quest_min_seq_len": 512,
            "quest_top_k_blocks": top_k,
            "seed": 20260913,
            "warmup_steps": 24,
        },
        "engines": [
            {
                "context_length": 8192,
                "identity": {
                    "enforce_eager": True,
                    "kv_blocks_requested": 640,
                    "num_kvcache_blocks": 640,
                    "kv_quant_bits": bits,
                    "quest_min_saved_blocks": min_saved_blocks,
                    "quest_min_seq_len": 512,
                    "quest_top_k_blocks": top_k,
                },
            }
        ],
        "rows": rows,
    }


def make_provenance(revision=REVISION):
    return {
        "source_revision": revision,
        "tinyvllm_dirty_paths": 0,
        "worker_sha256": "b" * 64,
        "cuda_device": "2",
        "mode": "sweep",
    }


def make_quality():
    details = []
    for depth in gate.EXPECTED_DEPTHS:
        for trial in range(5):
            magic = 10000 + int(depth * 100) * 10 + trial
            details.append(
                {
                    "ctx_len": 8192,
                    "depth": depth,
                    "trial": trial,
                    "magic": magic,
                    "answer": str(magic),
                    "hit": True,
                    "raw": str(magic),
                }
            )
    adaptive_events = [
        _activation_event(
            index + 17,
            top_k=16,
            min_saved_blocks=128,
            batch=8 if index < 12 else 4,
        )
        for index in range(16)
    ]
    return {
        "args": {
            "context_lens": [8192],
            "depths": list(gate.EXPECTED_DEPTHS),
            "enforce_eager": True,
            "fixed_prompts": True,
            "gpu_memory_utilization": 0.85,
            "kv_quant_bits": 8,
            "max_model_len": 16384,
            "max_num_seqs": 32,
            "max_output_len": 16,
            "model": "/models/Qwen3-8B",
            "needle_style": "newline",
            "num_trials": 5,
            "quest_min_saved_blocks": 128,
            "quest_min_seq_len": 512,
            "seed": 0,
            "tp_size": 1,
            "top_k_blocks_list": [-1, 16],
        },
        "results": [
            {
                "top_k": -1,
                "overall_accuracy": 1.0,
                "details": copy.deepcopy(details),
                "quest_activation": {
                    "steps": 16,
                    "reason_counts": {"disabled": 16},
                    "resolved_top_k_counts": {"-1": 16},
                    "first_observation_id": 1,
                    "last_observation_id": 16,
                },
            },
            {
                "top_k": 16,
                "overall_accuracy": 1.0,
                "details": copy.deepcopy(details),
                "quest_activation": {
                    "steps": 16,
                    "reason_counts": {
                        "active": 12,
                        "below_saved_blocks": 4,
                    },
                    "resolved_top_k_counts": {
                        "-1": 4,
                        "16": 12,
                    },
                    "first_observation_id": 17,
                    "last_observation_id": 32,
                    "events": adaptive_events,
                    "events_dropped": 0,
                },
            },
        ],
    }


def make_quality_provenance(revision=REVISION):
    return {
        "source_revision": revision,
        "source_dirty_paths": 0,
        "remote_dir": (
            "/data00/home/sitian/tllm/kvcapacity-runs/"
            "20260914-kv8quest-adaptive-quality-test"
        ),
        "cuda_device": "2",
        "model": "/models/Qwen3-8B",
        "fixed_prompts": True,
        "needle_style": "newline",
        "context_lens": [8192],
        "depths": list(gate.EXPECTED_DEPTHS),
        "num_trials": 5,
        "kv_quant_bits": 8,
        "quest_top_k_blocks": 16,
        "quest_min_seq_len": 512,
        "quest_min_saved_blocks": 128,
    }


def good_inputs():
    performance = {
        "bf16": make_perf(
            0,
            -1,
            0,
            [40, 45, 50, 55, 60, 70, 80],
        ),
        "kv8": make_perf(
            8,
            -1,
            0,
            [70, 90, 120, 150, 180, 220, 260],
        ),
        "fixed": make_perf(
            8,
            16,
            0,
            [88, 95, 86, 95, 105, 125, 145],
        ),
        "adaptive": make_perf(
            8,
            16,
            128,
            [70.5, 91, 87, 96, 106, 126, 146],
        ),
    }
    provenance = {
        name: make_provenance()
        for name in performance
    }
    return (
        performance,
        provenance,
        make_quality(),
        make_quality_provenance(),
    )


def classify(performance, provenance, quality, quality_provenance):
    return gate.build_report(
        performance,
        provenance,
        quality,
        quality_provenance,
    )


def test_all_thresholds_pass():
    report = classify(*good_inputs())

    assert report["classification"] == (
        "GO_KV8_QUEST_AMORTIZATION_POLICY"
    )
    assert report["identity_failures"] == []
    assert report["threshold_failures"] == []


def test_non_adaptive_arms_do_not_require_quest_events():
    inputs = good_inputs()
    for arm in ("bf16", "kv8"):
        for row in inputs[0][arm]["rows"]:
            row.pop("quest_activation_measured")
            row.pop("quest_activation_measured_events")

    report = classify(*inputs)

    assert report["classification"] == (
        "GO_KV8_QUEST_AMORTIZATION_POLICY"
    )
    assert report["identity_failures"] == []


def test_missing_boundary_cell_is_inconclusive():
    inputs = good_inputs()
    inputs[0]["adaptive"]["rows"] = [
        row
        for row in inputs[0]["adaptive"]["rows"]
        if row["batch"] != 6
    ]

    report = classify(*inputs)

    assert report["classification"] == (
        "INCONCLUSIVE_KV8_QUEST_AMORTIZATION_POLICY"
    )
    assert "adaptive_missing_cell_8192x6" in report["identity_failures"]


def test_source_mismatch_is_inconclusive():
    inputs = good_inputs()
    inputs[1]["adaptive"]["source_revision"] = "c" * 40

    report = classify(*inputs)

    assert report["classification"] == (
        "INCONCLUSIVE_KV8_QUEST_AMORTIZATION_POLICY"
    )
    assert "source_revision_mismatch" in report["identity_failures"]


def test_worker_mismatch_is_inconclusive():
    inputs = good_inputs()
    inputs[1]["adaptive"]["worker_sha256"] = "c" * 64

    report = classify(*inputs)

    assert report["classification"] == (
        "INCONCLUSIVE_KV8_QUEST_AMORTIZATION_POLICY"
    )
    assert "worker_sha256_mismatch" in report["identity_failures"]


def test_quality_model_must_match_performance_model():
    inputs = good_inputs()
    inputs[2]["args"]["model"] = "/models/Other-8B"
    inputs[3]["model"] = "/models/Other-8B"

    report = classify(*inputs)

    assert report["classification"] == (
        "INCONCLUSIVE_KV8_QUEST_AMORTIZATION_POLICY"
    )
    assert "quality_performance_model_mismatch" in report[
        "identity_failures"
    ]


def test_common_identity_and_sample_mutations_are_inconclusive():
    mutations = (
        (
            lambda inputs: inputs[0]["adaptive"]["configuration"].__setitem__(
                "model_path",
                "/models/Other-8B",
            ),
            "adaptive_model_path_mismatch",
        ),
        (
            lambda inputs: inputs[0]["adaptive"]["configuration"].__setitem__(
                "kv_blocks_requested",
                639,
            ),
            "adaptive_kv_blocks_requested_mismatch",
        ),
        (
            lambda inputs: inputs[0]["adaptive"]["configuration"].__setitem__(
                "seed",
                7,
            ),
            "adaptive_seed_mismatch",
        ),
        (
            lambda inputs: inputs[0]["adaptive"]["rows"][0]["step"].__setitem__(
                "count",
                23,
            ),
            "adaptive_8192x4_sample_count_mismatch",
        ),
    )
    for mutate, expected_failure in mutations:
        inputs = good_inputs()
        mutate(inputs)

        report = classify(*inputs)

        assert report["classification"] == (
            "INCONCLUSIVE_KV8_QUEST_AMORTIZATION_POLICY"
        )
        assert expected_failure in report["identity_failures"]


def test_missing_repeated_and_invalid_activation_events_are_inconclusive():
    mutations = (
        (
            lambda row: row.pop("quest_activation_measured_events"),
            "adaptive_8192x8_activation_event_count_mismatch",
        ),
        (
            lambda row: row["quest_activation_measured_events"][1].__setitem__(
                "observation_id",
                1,
            ),
            "adaptive_8192x8_activation_observation_ids_invalid",
        ),
        (
            lambda row: row["quest_activation_measured_events"][0].__setitem__(
                "status",
                "invalid",
            ),
            "adaptive_activation_mismatch_8192x8",
        ),
    )
    for mutate, expected_failure in mutations:
        inputs = good_inputs()
        row = inputs[0]["adaptive"]["rows"][2]
        mutate(row)

        report = classify(*inputs)

        assert report["classification"] == (
            "INCONCLUSIVE_KV8_QUEST_AMORTIZATION_POLICY"
        )
        assert expected_failure in report["identity_failures"]


def test_performance_activation_event_requires_worker_valid_status():
    inputs = good_inputs()
    event = inputs[0]["adaptive"]["rows"][2][
        "quest_activation_measured_events"
    ][0]
    event.pop("status")

    report = classify(*inputs)

    assert report["classification"] == (
        "INCONCLUSIVE_KV8_QUEST_AMORTIZATION_POLICY"
    )
    assert "adaptive_8192x8_activation_status_invalid" in report[
        "identity_failures"
    ]


def test_fixed_quest_arm_requires_complete_active_events():
    inputs = good_inputs()
    row = inputs[0]["fixed"]["rows"][2]
    row.pop("quest_activation_measured_events")

    report = classify(*inputs)

    assert report["classification"] == (
        "INCONCLUSIVE_KV8_QUEST_AMORTIZATION_POLICY"
    )
    assert "fixed_8192x8_activation_event_count_mismatch" in report[
        "identity_failures"
    ]


def test_performance_activation_event_must_match_cell_batch():
    inputs = good_inputs()
    row = inputs[0]["adaptive"]["rows"][2]
    row["quest_activation_measured_events"][0] = _activation_event(
        1,
        top_k=16,
        min_saved_blocks=128,
        batch=10,
    )

    report = classify(*inputs)

    assert report["classification"] == (
        "INCONCLUSIVE_KV8_QUEST_AMORTIZATION_POLICY"
    )
    assert "adaptive_activation_batch_mismatch_8192x8" in report[
        "identity_failures"
    ]


def test_adaptive_activation_event_must_match_registered_threshold():
    inputs = good_inputs()
    row = inputs[0]["adaptive"]["rows"][2]
    row["quest_activation_measured_events"] = [
        _activation_event(
            index + 1,
            top_k=16,
            min_saved_blocks=64,
            batch=8,
        )
        for index in range(24)
    ]

    report = classify(*inputs)

    assert report["classification"] == (
        "INCONCLUSIVE_KV8_QUEST_AMORTIZATION_POLICY"
    )
    assert "adaptive_activation_configuration_mismatch_8192x8" in report[
        "identity_failures"
    ]


def test_performance_prompt_digests_must_match_across_arms():
    inputs = good_inputs()
    inputs[0]["adaptive"]["rows"][2]["prompt_digests"][0] = (
        "different-prompt"
    )

    report = classify(*inputs)

    assert report["classification"] == (
        "INCONCLUSIVE_KV8_QUEST_AMORTIZATION_POLICY"
    )
    assert "adaptive_prompt_digests_mismatch_8192x8" in report[
        "identity_failures"
    ]


def test_missing_step_median_is_inconclusive_instead_of_crashing():
    inputs = good_inputs()
    inputs[0]["adaptive"]["rows"][2]["step"].pop("median_ms")

    report = classify(*inputs)

    assert report["classification"] == (
        "INCONCLUSIVE_KV8_QUEST_AMORTIZATION_POLICY"
    )
    assert "adaptive_8192x8_step_median_missing" in report[
        "identity_failures"
    ]


def test_fallback_cell_must_not_activate():
    inputs = good_inputs()
    row = inputs[0]["adaptive"]["rows"][0]
    row["quest_activation_measured_events"] = [
        _activation_event(
            index + 1,
            top_k=16,
            min_saved_blocks=0,
            batch=4,
        )
        for index in range(24)
    ]

    report = classify(*inputs)

    assert "adaptive_activation_mismatch_8192x4" in report[
        "identity_failures"
    ]


def test_active_cell_must_not_fallback():
    inputs = good_inputs()
    row = inputs[0]["adaptive"]["rows"][2]
    row["quest_activation_measured_events"] = [
        _activation_event(
            index + 1,
            top_k=16,
            min_saved_blocks=256,
            batch=8,
        )
        for index in range(24)
    ]

    report = classify(*inputs)

    assert "adaptive_activation_mismatch_8192x8" in report[
        "identity_failures"
    ]


def test_fallback_regression_over_three_percent_is_no_go():
    inputs = good_inputs()
    inputs[0]["adaptive"]["rows"][0]["step"]["median_ms"] = 73

    report = classify(*inputs)

    assert report["classification"] == (
        "NO_GO_KV8_QUEST_AMORTIZATION_POLICY"
    )
    assert "fallback_regression_over_3pct_8192x4" in report[
        "threshold_failures"
    ]


def test_active_regression_over_three_percent_is_no_go():
    inputs = good_inputs()
    inputs[0]["adaptive"]["rows"][2]["step"]["median_ms"] = 90

    report = classify(*inputs)

    assert "active_regression_over_3pct_8192x8" in report[
        "threshold_failures"
    ]


def test_wall_recovery_below_fifty_percent_is_no_go():
    inputs = good_inputs()
    inputs[0]["adaptive"]["rows"][-1]["step"]["median_ms"] = 180

    report = classify(*inputs)

    assert "wall_excess_latency_recovery_below_50pct" in report[
        "threshold_failures"
    ]


def test_quality_requires_an_active_step():
    inputs = good_inputs()
    adaptive = inputs[2]["results"][1]
    adaptive["quest_activation"]["reason_counts"] = {
        "below_saved_blocks": 16,
    }
    adaptive["quest_activation"]["resolved_top_k_counts"] = {
        "-1": 16,
    }
    adaptive["quest_activation"]["events"] = [
        _activation_event(
            index + 17,
            top_k=16,
            min_saved_blocks=128,
            batch=4,
        )
        for index in range(16)
    ]

    report = classify(*inputs)

    assert "adaptive_quality_has_no_active_step" in report[
        "identity_failures"
    ]


def test_quality_accepts_raw_runtime_events_without_worker_status():
    inputs = good_inputs()
    events = inputs[2]["results"][1]["quest_activation"]["events"]
    for event in events:
        event.pop("status")

    report = classify(*inputs)

    assert report["classification"] == (
        "GO_KV8_QUEST_AMORTIZATION_POLICY"
    )
    assert "adaptive_quality_activation_mismatch" not in report[
        "identity_failures"
    ]


def test_quality_activation_observation_ids_must_be_unique():
    inputs = good_inputs()
    events = inputs[2]["results"][1]["quest_activation"]["events"]
    events[1]["observation_id"] = events[0]["observation_id"]

    report = classify(*inputs)

    assert report["classification"] == (
        "INCONCLUSIVE_KV8_QUEST_AMORTIZATION_POLICY"
    )
    assert "adaptive_quality_activation_observation_ids_invalid" in report[
        "identity_failures"
    ]


def test_malformed_quality_activation_event_is_inconclusive():
    inputs = good_inputs()
    event = inputs[2]["results"][1]["quest_activation"]["events"][0]
    event.pop("status")
    event["sequence_block_counts"][0] = "not-an-integer"

    report = classify(*inputs)

    assert report["classification"] == (
        "INCONCLUSIVE_KV8_QUEST_AMORTIZATION_POLICY"
    )
    assert "adaptive_quality_activation_mismatch" in report[
        "identity_failures"
    ]


def test_non_mapping_activation_events_are_inconclusive():
    mutations = (
        (
            lambda inputs: inputs[0]["adaptive"]["rows"][2][
                "quest_activation_measured_events"
            ].__setitem__(0, None),
            "adaptive_8192x8_activation_observation_ids_invalid",
        ),
        (
            lambda inputs: inputs[2]["results"][1][
                "quest_activation"
            ]["events"].__setitem__(0, None),
            "adaptive_quality_activation_mismatch",
        ),
    )
    for mutate, expected_failure in mutations:
        inputs = good_inputs()
        mutate(inputs)

        report = classify(*inputs)

        assert report["classification"] == (
            "INCONCLUSIVE_KV8_QUEST_AMORTIZATION_POLICY"
        )
        assert expected_failure in report["identity_failures"]


def test_quality_hit_is_recomputed_from_answer():
    inputs = good_inputs()
    inputs[2]["results"][1]["details"][0]["answer"] = "99999"
    inputs[2]["results"][1]["details"][0]["hit"] = True

    report = classify(*inputs)

    assert report["quality"]["adaptive_overall_accuracy"] == 24 / 25


def test_duplicate_quality_case_is_inconclusive():
    inputs = good_inputs()
    duplicate = copy.deepcopy(
        inputs[2]["results"][1]["details"][0]
    )
    inputs[2]["results"][1]["details"].append(duplicate)

    report = classify(*inputs)

    assert report["classification"] == (
        "INCONCLUSIVE_KV8_QUEST_AMORTIZATION_POLICY"
    )
    assert "quality_adaptive_duplicate_case" in report[
        "identity_failures"
    ]


def test_quality_cases_must_cover_the_exact_registered_grid():
    inputs = good_inputs()
    for setting in inputs[2]["results"]:
        setting["details"][0]["depth"] = 0.1

    report = classify(*inputs)

    assert report["classification"] == (
        "INCONCLUSIVE_KV8_QUEST_AMORTIZATION_POLICY"
    )
    assert "quality_full_case_grid_mismatch" in report[
        "identity_failures"
    ]
    assert "quality_adaptive_case_grid_mismatch" in report[
        "identity_failures"
    ]


def test_overall_quality_loss_over_five_points_is_no_go():
    inputs = good_inputs()
    for row in inputs[2]["results"][1]["details"][:2]:
        row["answer"] = "99999"

    report = classify(*inputs)

    assert "overall_quality_loss_over_5pp" in report[
        "threshold_failures"
    ]


def test_depth_quality_loss_over_twenty_points_is_no_go():
    inputs = good_inputs()
    changed = 0
    for row in inputs[2]["results"][1]["details"]:
        if row["depth"] == 0.5 and changed < 2:
            row["answer"] = "99999"
            changed += 1

    report = classify(*inputs)

    assert "depth_quality_loss_over_20pp_0.5" in report[
        "threshold_failures"
    ]


def test_markdown_reports_overall_and_per_depth_quality_deltas():
    report = classify(*good_inputs())

    markdown = gate.render_markdown(report)

    assert "## Quality" in markdown
    assert (
        "| Overall | 100.000% | 100.000% | +0.000 pp |"
        in markdown
    )
    for depth in gate.EXPECTED_DEPTHS:
        assert (
            f"| Depth {depth:.2f} | 100.000% | "
            "100.000% | +0.000 pp |"
        ) in markdown
