from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from tools import context_gated_elastic_exact_burst_gate as gate
from tools import profile_context_gated_elastic_exact_burst as profile
from tools.test_profile_context_gated_elastic_exact_burst import (
    RUN_TAG,
    SOURCE_COMMIT,
    _case_row,
    _summary,
)


ROOT = Path(__file__).resolve().parents[1]


def _valid_metrics() -> dict:
    context = {
        str(length): {
            "tpot_median_improvement_pct": 2.0,
            "tpot_p95_improvement_pct": 1.0,
            "tpot_p99_regression_pct": 2.0,
            "ttft_regression_pct": 2.0,
            "e2e_regression_pct": 2.0,
            "throughput_regression_pct": 1.0,
            "allocated_memory_regression_pct": 3.0,
            "reserved_memory_regression_pct": 3.0,
        }
        for length in profile.CONTEXT_LENGTHS
    }
    return {
        "evidence_complete": True,
        "correctness_exact": True,
        "width_policy_exact": True,
        "runtime_inventory_exact": True,
        "zero_unexpected_lifecycle_events": True,
        "eligible_aggregate": {
            "tpot_median_improvement_pct": 2.0,
            "tpot_p95_improvement_pct": 1.0,
        },
        "by_context": context,
        "maximum_selected_k16_host_visible_gap_ns": 40_000_000,
    }


def _performance_rows() -> list[dict]:
    rows = []
    for repetition, context, policy in gate.expected_performance_identities():
        row = _case_row(
            policy,
            context_length=context,
            repetition=repetition,
        )
        if policy == "context_gated_elastic_k16" and context <= 2048:
            row["amortized_tpot_samples_ns"] = [980_000.0] * 127
            row["amortized_tpot_median_ns"] = 980_000.0
            row["amortized_tpot_p95_ns"] = 980_000.0
            row["amortized_tpot_p99_ns"] = 980_000.0
            row["e2e_ns"] = 137_200_000
            row["output_tokens_per_second"] = (
                profile.GENERATED_TOKENS / 0.1372
            )
        rows.append(row)
    return rows


def _correctness_rows(run_dir: Path) -> list[dict]:
    rows = []
    for context, policy, point in gate.expected_correctness_identities():
        summary = _summary(policy, context)
        summary["sampled_logit_d2h_calls"] = 3
        summary["capture_receipts"][0]["correctness_trace"] = True
        sidecar = profile.write_float32_sidecar(
            run_dir,
            f"logits/{context}-{policy}-{point}.f32",
            (1.0, 4.0, 3.0),
        )
        rows.append({
            "schema_version": profile.CORRECTNESS_SCHEMA_VERSION,
            "run_tag": RUN_TAG,
            "source_commit": SOURCE_COMMIT,
            "policy": policy,
            "context_length": context,
            "generated_tokens": profile.GENERATED_TOKENS,
            "sampling_point": point,
            "prompt_sha256": "b" * 64,
            "output_token_ids": list(range(profile.GENERATED_TOKENS)),
            "output_text_sha256": "e" * 64,
            "argmax_token_id": 1,
            "logits_path": sidecar["path"],
            "logits_shape": [1, 3],
            "logits_element_count": sidecar["element_count"],
            "logits_byte_length": sidecar["byte_length"],
            "logits_sha256": sidecar["sha256"],
            "correctness_trace": True,
            "exact_greedy_decode_burst_summary": summary,
        })
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def write_fixture_bundle(root: Path) -> Path:
    run_dir = root / "run"
    run_dir.mkdir(parents=True)
    performance = _performance_rows()
    correctness = _correctness_rows(run_dir)
    workload = profile.build_workload_manifest(
        model="/models/Qwen3-0.6B",
        device="cuda:0",
        run_tag=RUN_TAG,
        source_commit=SOURCE_COMMIT,
        gpu_memory_utilization=0.5,
        environment={"fixture": True},
    )
    _write_json(run_dir / "workload_manifest.json", workload)
    _write_json(
        run_dir / "source_manifest.json",
        {
            "schema_version": profile.SOURCE_SCHEMA_VERSION,
            "run_tag": RUN_TAG,
            "source_commit": SOURCE_COMMIT,
            "source_sha256": {
                relative: gate.sha256_file(ROOT / relative)
                for relative in profile.SOURCE_FILES
            },
        },
    )
    (run_dir / "source.patch").write_bytes(b"")
    _write_jsonl(run_dir / "performance_rows.jsonl", performance)
    _write_jsonl(run_dir / "correctness_rows.jsonl", correctness)
    worker_summary = profile.summarize_rows(performance)
    worker_summary["correctness_row_count"] = len(correctness)
    _write_json(run_dir / "profile_summary.json", worker_summary)
    return run_dir


def test_terminal_constants_and_exact_thresholds_are_frozen() -> None:
    assert gate.TERMINAL_REPETITIONS == 5
    assert gate.PERFORMANCE_ROW_COUNT == 40
    assert gate.CORRECTNESS_ROW_COUNT == 32
    assert gate.MINIMUM_ELIGIBLE_MEDIAN_TPOT_IMPROVEMENT_PCT == 2.0
    assert gate.MINIMUM_ELIGIBLE_P95_TPOT_IMPROVEMENT_PCT == 1.0
    assert gate.MAXIMUM_PER_CONTEXT_TPOT_REGRESSION_PCT == 2.0
    assert gate.MAXIMUM_LATENCY_REGRESSION_PCT == 2.0
    assert gate.MAXIMUM_THROUGHPUT_REGRESSION_PCT == 1.0
    assert gate.MAXIMUM_MEMORY_REGRESSION_PCT == 3.0
    assert gate.MAXIMUM_K16_HOST_VISIBLE_GAP_NS == 40_000_000
    assert gate.classify(_valid_metrics()) == (
        gate.GO_CONTEXT_GATED_ELASTIC_EXACT_BURST
    )


@pytest.mark.parametrize(
    ("mutate", "expected"),
    (
        (
            lambda value: value.update(evidence_complete=False),
            "NO_GO_EVIDENCE_INCOMPLETE",
        ),
        (
            lambda value: value.update(correctness_exact=False),
            "NO_GO_CORRECTNESS",
        ),
        (
            lambda value: value.update(width_policy_exact=False),
            "NO_GO_WIDTH_POLICY",
        ),
        (
            lambda value: value.update(runtime_inventory_exact=False),
            "NO_GO_RUNTIME_INVARIANT",
        ),
        (
            lambda value: value.update(
                zero_unexpected_lifecycle_events=False
            ),
            "NO_GO_RUNTIME_INVARIANT",
        ),
        (
            lambda value: value.update(
                maximum_selected_k16_host_visible_gap_ns=40_000_001
            ),
            "NO_GO_BURST_GAP",
        ),
        (
            lambda value: value["eligible_aggregate"].update(
                tpot_median_improvement_pct=1.999999
            ),
            "NO_GO_INSUFFICIENT_INCREMENTAL_BENEFIT",
        ),
        (
            lambda value: value["eligible_aggregate"].update(
                tpot_p95_improvement_pct=0.999999
            ),
            "NO_GO_INSUFFICIENT_INCREMENTAL_BENEFIT",
        ),
        (
            lambda value: value["by_context"]["4096"].update(
                tpot_median_improvement_pct=-2.000001
            ),
            "NO_GO_PROTECTED_REGRESSION",
        ),
        (
            lambda value: value["by_context"]["8192"].update(
                tpot_p95_improvement_pct=-2.000001
            ),
            "NO_GO_PROTECTED_REGRESSION",
        ),
        (
            lambda value: value["by_context"]["256"].update(
                tpot_p99_regression_pct=2.000001
            ),
            "NO_GO_PROTECTED_REGRESSION",
        ),
        (
            lambda value: value["by_context"]["256"].update(
                ttft_regression_pct=2.000001
            ),
            "NO_GO_PROTECTED_REGRESSION",
        ),
        (
            lambda value: value["by_context"]["256"].update(
                e2e_regression_pct=2.000001
            ),
            "NO_GO_PROTECTED_REGRESSION",
        ),
        (
            lambda value: value["by_context"]["256"].update(
                throughput_regression_pct=1.000001
            ),
            "NO_GO_PROTECTED_REGRESSION",
        ),
        (
            lambda value: value["by_context"]["256"].update(
                allocated_memory_regression_pct=3.000001
            ),
            "NO_GO_PROTECTED_REGRESSION",
        ),
        (
            lambda value: value["by_context"]["256"].update(
                reserved_memory_regression_pct=3.000001
            ),
            "NO_GO_PROTECTED_REGRESSION",
        ),
    ),
)
def test_each_terminal_classification_is_independent(
    mutate,
    expected: str,
) -> None:
    metrics = _valid_metrics()
    mutate(metrics)
    assert gate.classify(metrics) == expected


def test_raw_rows_reconstruct_complete_terminal_go(tmp_path: Path) -> None:
    result = gate.summarize_evidence(
        _performance_rows(),
        _correctness_rows(tmp_path),
        run_dir=tmp_path,
    )

    assert result["classification"] == (
        gate.GO_CONTEXT_GATED_ELASTIC_EXACT_BURST
    )
    assert result["performance_row_count"] == 40
    assert result["correctness_row_count"] == 32
    assert result["correctness_exact"] is True
    assert result["width_policy_exact"] is True
    assert result["runtime_inventory_exact"] is True
    assert result["zero_unexpected_lifecycle_events"] is True
    assert result["eligible_aggregate"][
        "tpot_median_improvement_pct"
    ] == pytest.approx(2.0)
    assert result["eligible_aggregate"][
        "tpot_p95_improvement_pct"
    ] == pytest.approx(2.0)
    assert result["elastic_incremental_capture_duration_ns"] == 0
    assert result["elastic_incremental_retained_static_bytes"] == 0
    assert result["k16_width_health_quarantine_count"] == 0
    assert 0.0 <= result["k8_fallback_rate"] <= 1.0


def test_incomplete_and_duplicate_inventory_fail_closed(
    tmp_path: Path,
) -> None:
    performance = _performance_rows()
    correctness = _correctness_rows(tmp_path)
    incomplete = gate.summarize_evidence(
        performance[:-1],
        correctness,
        run_dir=tmp_path,
    )
    assert incomplete["classification"] == (
        gate.NO_GO_EVIDENCE_INCOMPLETE
    )
    duplicate = gate.summarize_evidence(
        performance + [deepcopy(performance[0])],
        correctness,
        run_dir=tmp_path,
    )
    assert duplicate["classification"] == (
        gate.NO_GO_EVIDENCE_INCOMPLETE
    )


def test_unexpected_fallback_rollback_and_quarantine_fail(
    tmp_path: Path,
) -> None:
    for mutation in ("fallback", "rollback", "quarantine"):
        performance = _performance_rows()
        candidate = next(
            row for row in performance
            if row["policy"] == "context_gated_elastic_k16"
            and row["context_length"] == 256
        )
        summary = candidate["exact_greedy_decode_burst_summary"]
        if mutation == "fallback":
            summary["fallback_counts"] = {"unexpected": 1}
        elif mutation == "rollback":
            summary["lease_local_delta_journal_rollbacks"] = 1
        else:
            summary["quarantines"] = 1
            summary["quarantine_reason"] = "fixture"
        result = gate.summarize_evidence(
            performance,
            _correctness_rows(tmp_path / mutation),
            run_dir=tmp_path / mutation,
        )
        assert result["classification"] == (
            gate.NO_GO_RUNTIME_INVARIANT
        )


def test_malformed_performance_measurement_is_rejected(
    tmp_path: Path,
) -> None:
    performance = _performance_rows()
    performance[0]["decode_host_ns"] = -1

    with pytest.raises(ValueError, match="decode_host_ns"):
        gate.summarize_evidence(
            performance,
            _correctness_rows(tmp_path),
            run_dir=tmp_path,
        )


def test_inconsistent_k16_selection_is_explicit_width_no_go(
    tmp_path: Path,
) -> None:
    performance = _performance_rows()
    candidate = next(
        row for row in performance
        if row["policy"] == "context_gated_elastic_k16"
        and row["context_length"] == 256
    )
    candidate["exact_greedy_decode_burst_summary"][
        "authorized_width_histogram"
    ] = {"8": 8}

    result = gate.summarize_evidence(
        performance,
        _correctness_rows(tmp_path),
        run_dir=tmp_path,
    )

    assert result["classification"] == gate.NO_GO_WIDTH_POLICY


@pytest.mark.parametrize(
    ("relative", "mutate", "error"),
    (
        (
            "workload_manifest.json",
            lambda payload: payload.update(temperature=1.0),
            "workload manifest",
        ),
        (
            "source_manifest.json",
            lambda payload: payload.update(schema_version="wrong"),
            "manifest",
        ),
    ),
)
def test_producer_rejects_manifest_contract_tamper(
    tmp_path: Path,
    relative: str,
    mutate,
    error: str,
) -> None:
    run_dir = write_fixture_bundle(tmp_path)
    path = run_dir / relative
    payload = json.loads(path.read_text())
    mutate(payload)
    _write_json(path, payload)

    with pytest.raises(ValueError, match=error):
        gate.produce_artifacts(run_dir, source_root=ROOT)


def test_producer_writes_hash_bound_terminal_bundle(
    tmp_path: Path,
) -> None:
    run_dir = write_fixture_bundle(tmp_path)

    receipt = gate.produce_artifacts(
        run_dir,
        source_root=ROOT,
    )

    assert receipt["classification"] == (
        gate.GO_CONTEXT_GATED_ELASTIC_EXACT_BURST
    )
    assert receipt["performance_row_count"] == 40
    assert receipt["correctness_row_count"] == 32
    manifest = json.loads(
        (run_dir / "terminal_manifest.json").read_text()
    )
    assert set(manifest["artifact_sha256"]) == (
        set(gate.PRIMARY_ARTIFACTS)
        | {
            row["logits_path"]
            for row in _correctness_rows(tmp_path / "sidecar-list")
        }
    )
    assert all(
        len(digest) == 64
        for digest in manifest["artifact_sha256"].values()
    )
    assert (
        manifest["source_patch_sha256"]
        == hashlib.sha256(b"").hexdigest()
    )
