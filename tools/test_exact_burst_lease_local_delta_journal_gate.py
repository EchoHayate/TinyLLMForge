from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tools"
    / "exact_burst_lease_local_delta_journal_gate.py"
)
RUN_TAG = "20260823-qwen3-06b-delta-journal-fixture"
SOURCE_SHA = "a" * 40


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "exact_burst_lease_local_delta_journal_gate_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_workload_manifest_has_fixed_60_24_inventory():
    gate = _load_module()

    manifest = gate.build_workload_manifest()

    assert manifest["performance_row_count"] == 60
    assert manifest["correctness_row_count"] == 24
    assert manifest["policies"] == (
        "generic",
        "lease_local_delta",
    )
    assert manifest["contexts"] == (
        "short",
        "medium",
        "long",
    )
    assert manifest["performance_repetitions"] == 10
    assert manifest["correctness_sampling_points"] == 4
    for repetition in range(10):
        for context_index, context in enumerate(
            manifest["contexts"]
        ):
            order = manifest["policy_order"][
                str(repetition)
            ][context]
            assert set(order) == set(manifest["policies"])
            assert len(order) == 2
            assert order[0] == manifest["policies"][
                (repetition + context_index) % 2
            ]


def passing_metrics():
    return {
        "performance_row_count": 60,
        "correctness_row_count": 24,
        "output_exact": True,
        "sampled_logit_max_abs_diff": 0.0,
        "candidate_fallbacks": 0,
        "candidate_rollbacks": 0,
        "forward_inventory_equal": True,
        "replay_inventory_equal": True,
        "d2h_call_inventory_equal": True,
        "d2h_byte_inventory_equal": True,
        "long_prepare_median_improvement_pct": 75.0,
        "long_prepare_p95_improvement_pct": 70.0,
        "short_prepare_median_regression_pct": -20.0,
        "short_prepare_p95_regression_pct": -15.0,
        "medium_prepare_median_regression_pct": -50.0,
        "medium_prepare_p95_regression_pct": -45.0,
        "aggregate_tpot_median_regression_pct": 0.0,
        "aggregate_tpot_p95_regression_pct": 1.0,
        "aggregate_ttft_regression_pct": 1.0,
        "aggregate_e2e_regression_pct": 1.0,
        "throughput_regression_pct": 1.0,
        "reserved_memory_regression_pct": 0.5,
    }


@pytest.mark.parametrize(
    ("field", "value", "classification"),
    (
        ("output_exact", False, "NO_GO_CORRECTNESS"),
        (
            "sampled_logit_max_abs_diff",
            1e-5,
            "NO_GO_CORRECTNESS",
        ),
        (
            "candidate_fallbacks",
            1,
            "NO_GO_TRANSACTIONAL_SAFETY",
        ),
        (
            "candidate_rollbacks",
            1,
            "NO_GO_TRANSACTIONAL_SAFETY",
        ),
        (
            "forward_inventory_equal",
            False,
            "NO_GO_CORRECTNESS",
        ),
        (
            "long_prepare_median_improvement_pct",
            49.9,
            "NO_GO_PERFORMANCE",
        ),
        (
            "long_prepare_p95_improvement_pct",
            49.9,
            "NO_GO_PERFORMANCE",
        ),
        (
            "aggregate_tpot_p95_regression_pct",
            3.01,
            "NO_GO_PERFORMANCE",
        ),
        (
            "reserved_memory_regression_pct",
            1.01,
            "NO_GO_PERFORMANCE",
        ),
        (
            "performance_row_count",
            59,
            "NO_GO_EVIDENCE_INCOMPLETE",
        ),
        (
            "correctness_row_count",
            23,
            "NO_GO_EVIDENCE_INCOMPLETE",
        ),
    ),
)
def test_classification_rejects_each_failed_contract(
    field,
    value,
    classification,
):
    gate = _load_module()
    metrics = passing_metrics()
    metrics[field] = value

    assert gate.classify(metrics) == classification


def test_classification_accepts_complete_exact_evidence():
    gate = _load_module()

    assert gate.classify(passing_metrics()) == (
        "GO_EXACT_BURST_LEASE_LOCAL_DELTA_JOURNAL"
    )


def make_fixture_rows():
    gate = _load_module()
    performance = []
    correctness = []
    for repetition in range(10):
        for context_index, context in enumerate(gate.CONTEXTS):
            for order_position, policy in enumerate(
                gate.policy_order(repetition, context_index)
            ):
                baseline_prepare = {
                    "short": 1000,
                    "medium": 2000,
                    "long": 4000,
                }[context]
                prepare = (
                    baseline_prepare
                    if policy == "generic"
                    else {
                        "short": 800,
                        "medium": 900,
                        "long": 1000,
                    }[context]
                )
                performance.append({
                    "schema": gate.PERFORMANCE_ROW_SCHEMA,
                    "run_tag": RUN_TAG,
                    "source_sha": SOURCE_SHA,
                    "policy": policy,
                    "context": context,
                    "repetition": repetition,
                    "order_position": order_position,
                    "prompt_digest": context * 8,
                    "generated_tokens": 128,
                    "output_tokens": list(range(128)),
                    "phase_prepare_ns": [prepare] * 10,
                    "ttft_ns": 1000,
                    "tpot_samples_ns": [1000] * 127,
                    "e2e_ns": 128000,
                    "output_tokens_per_second": 1000.0,
                    "cuda_peak_allocated_bytes": 1000000,
                    "cuda_peak_reserved_bytes": 1000000,
                    "target_model_forwards": 128,
                    "graph_replays": 128,
                    "d2h_calls": 32,
                    "d2h_bytes": 1024,
                    "delta_attempts": (
                        32 if policy == "lease_local_delta" else 0
                    ),
                    "delta_captures": (
                        32 if policy == "lease_local_delta" else 0
                    ),
                    "delta_commits": (
                        32 if policy == "lease_local_delta" else 0
                    ),
                    "delta_rollbacks": 0,
                    "delta_published_blocks": (
                        8 if policy == "lease_local_delta" else 0
                    ),
                    "delta_fallbacks": {},
                })
    for context in gate.CONTEXTS:
        for policy in gate.POLICIES:
            for sampling_point in gate.SAMPLING_POINTS:
                correctness.append({
                    "schema": gate.CORRECTNESS_ROW_SCHEMA,
                    "run_tag": RUN_TAG,
                    "source_sha": SOURCE_SHA,
                    "policy": policy,
                    "context": context,
                    "sampling_point": sampling_point,
                    "output_token_ids": [1, 2, 3],
                    "sampled_logits": [0.25, -0.5, 1.0],
                    "target_model_forwards": 128,
                    "graph_replays": 128,
                    "d2h_calls": 32,
                    "d2h_bytes": 1024,
                })
    return performance, correctness


def write_fixture_bundle(root: Path) -> Path:
    gate = _load_module()
    run_dir = root / "primary"
    run_dir.mkdir(parents=True)
    performance, correctness = make_fixture_rows()
    gate.write_jsonl(
        run_dir / "performance_rows.jsonl",
        performance,
    )
    gate.write_jsonl(
        run_dir / "correctness_rows.jsonl",
        correctness,
    )
    (run_dir / "phase_samples.jsonl").write_text(
        "".join(
            json.dumps({
                "policy": row["policy"],
                "context": row["context"],
                "repetition": row["repetition"],
                "phase_prepare_ns": row["phase_prepare_ns"],
            }, sort_keys=True)
            + "\n"
            for row in performance
        )
    )
    gate.write_json(
        run_dir / "workload_manifest.json",
        gate.build_workload_manifest(
            run_tag=RUN_TAG,
            source_sha=SOURCE_SHA,
        ),
    )
    summary = gate.summarize_evidence(
        performance,
        correctness,
    )
    gate.write_json(run_dir / "summary.json", summary)
    gate.write_json(
        run_dir / "runner_receipt.json",
        {
            "schema": gate.RUNNER_RECEIPT_SCHEMA,
            "run_tag": RUN_TAG,
            "source_sha": SOURCE_SHA,
            "exit_code": 0,
        },
    )
    artifact_names = (
        "workload_manifest.json",
        "performance_rows.jsonl",
        "correctness_rows.jsonl",
        "phase_samples.jsonl",
    )
    gate.write_json(
        run_dir / "source_manifest.json",
        {
            "schema": gate.SOURCE_MANIFEST_SCHEMA,
            "run_tag": RUN_TAG,
            "source_sha": SOURCE_SHA,
            "source_patch_sha256": hashlib.sha256(b"").hexdigest(),
            "source_file_sha256": {
                relative: hashlib.sha256(
                    (ROOT / relative).read_bytes()
                ).hexdigest()
                for relative in (
                    "tools/"
                    "exact_burst_lease_local_delta_journal_gate.py",
                    "tools/"
                    "exact_burst_lease_local_delta_journal_verify.py",
                )
            },
            "artifact_sha256": {
                name: hashlib.sha256(
                    (run_dir / name).read_bytes()
                ).hexdigest()
                for name in artifact_names
            },
        },
    )
    return run_dir
