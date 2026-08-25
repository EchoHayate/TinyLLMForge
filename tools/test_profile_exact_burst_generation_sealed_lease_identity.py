from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tools"
    / "profile_exact_burst_generation_sealed_lease_identity.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "generation_sealed_lease_identity_profile_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_profile_cases_are_fixed_and_complete():
    profile = _load_module()

    assert profile.CONTEXT_LENGTHS == (249, 2041, 8185)
    assert profile.POLICIES == (
        "full_identity",
        "generation_sealed",
    )
    assert profile.DEFAULT_REPETITIONS == 100
    assert {
        (case["policy"], case["sequence_length"])
        for case in profile.build_profile_cases()
    } == {
        (policy, sequence_length)
        for policy in profile.POLICIES
        for sequence_length in profile.CONTEXT_LENGTHS
    }


@pytest.mark.parametrize(
    "policy",
    ("full_identity", "generation_sealed"),
)
def test_profile_case_reports_identity_lifecycle(policy):
    profile = _load_module()

    row = profile.run_profile_case(
        policy=policy,
        sequence_length=249,
        repetitions=3,
    )

    assert row["schema"] == (
        "exact_burst_generation_sealed_"
        "lease_identity_cpu_profile_v1"
    )
    assert row["policy"] == policy
    assert row["sequence_length"] == 249
    assert row["sample_count"] == 3
    assert row["lease_grant_median_us"] >= 0
    assert row["lease_grant_p95_us"] >= (
        row["lease_grant_median_us"]
    )
    assert row["lease_lifecycle_median_us"] >= (
        row["lease_grant_median_us"]
    )
    assert row["lease_lifecycle_p95_us"] >= 0
    assert len(row["lease_grant_samples_us"]) == 3
    assert len(row["lease_lifecycle_samples_us"]) == 3
    assert row["positive_python_allocation_bytes"] >= 0
    assert row["fallback_counts"] == {}
    assert row["identity_seal_fallback_counts"] == {}
    assert row["failure_count"] == 0
    assert row["pending_lease_count"] == 0
    assert row["rollback_count"] == 13
    if policy == "generation_sealed":
        assert row["identity_rows_visited"] == 0
        assert row["identity_seal_cold_captures"] == 1
        assert row["identity_seal_hot_reuses"] == 12
        assert row["identity_seal_validations"] == 39
    else:
        assert row["identity_rows_visited"] > 0
        assert row["identity_seal_cold_captures"] == 0
        assert row["identity_seal_hot_reuses"] == 0
        assert row["identity_seal_validations"] == 0


def test_profile_collects_latency_before_allocation_tracing(
    monkeypatch,
):
    profile = _load_module()
    events = []
    original_collect = profile._collect_timing_samples
    original_start = profile.tracemalloc.start

    def collect_before_tracing(*args, **kwargs):
        events.append(
            ("timing", profile.tracemalloc.is_tracing())
        )
        return original_collect(*args, **kwargs)

    def start_after_timing():
        events.append(("allocation", False))
        original_start()

    monkeypatch.setattr(
        profile,
        "_collect_timing_samples",
        collect_before_tracing,
    )
    monkeypatch.setattr(
        profile.tracemalloc,
        "start",
        start_after_timing,
    )

    profile.run_profile_case(
        policy="generation_sealed",
        sequence_length=249,
        repetitions=1,
    )

    assert events == [
        ("timing", False),
        ("allocation", False),
    ]


def test_profile_summary_uses_ratio_safe_aggregate_and_classifies_gate():
    profile = _load_module()

    rows = []
    for sequence_length in profile.CONTEXT_LENGTHS:
        baseline_samples = (
            [100.0, 100.0, 100.0]
            if sequence_length != 8185
            else [200.0, 200.0, 200.0]
        )
        candidate_samples = (
            [75.0, 75.0, 75.0]
            if sequence_length != 8185
            else [120.0, 120.0, 120.0]
        )
        for policy, samples in (
            ("full_identity", baseline_samples),
            ("generation_sealed", candidate_samples),
        ):
            rows.append({
                "schema": profile.PROFILE_SCHEMA,
                "policy": policy,
                "sequence_length": sequence_length,
                "sample_count": 3,
                "warmup_count": 10,
                "lease_grant_samples_us": list(samples),
                "lease_lifecycle_samples_us": list(samples),
                "lease_grant_median_us": samples[1],
                "lease_grant_p95_us": samples[-1],
                "lease_lifecycle_median_us": samples[1],
                "lease_lifecycle_p95_us": samples[-1],
                "identity_rows_visited": (
                    0 if policy == "generation_sealed" else 3
                ),
                "identity_seal_cold_captures": (
                    1 if policy == "generation_sealed" else 0
                ),
                "identity_seal_hot_reuses": (
                    12 if policy == "generation_sealed" else 0
                ),
                "identity_seal_validations": (
                    39 if policy == "generation_sealed" else 0
                ),
                "positive_python_allocation_bytes": 0,
                "fallback_counts": {},
                "identity_seal_fallback_counts": {},
                "failure_count": 0,
                "pending_lease_count": 0,
                "rollback_count": 13,
            })

    summary = profile.summarize_profile(tuple(rows))

    assert summary["aggregate"]["aggregation"] == (
        "geometric_mean_of_per_context_median_ratios"
    )
    assert summary["aggregate"]["context_count"] == 3
    assert (
        summary["aggregate"][
            "lifecycle_median_improvement_pct"
        ]
        == pytest.approx(
            100.0
            * (
                1.0
                - (0.75 * 0.75 * 0.60) ** (1.0 / 3.0)
            )
        )
    )
    assert (
        summary["by_context"]["8185"][
            "lifecycle_median_improvement_pct"
        ]
        == 40
    )
    assert summary["checks"] == {
        "8k_lifecycle_median_improvement": True,
        "8k_lifecycle_p95_improvement": True,
        "aggregate_lifecycle_median_improvement": True,
        "candidate_hot_path_identity_rows_zero": True,
        "candidate_one_cold_capture_per_fixture": True,
        "candidate_no_fallback_or_rollback_failures": True,
    }
    assert summary["classification"] == "GO"


def test_profile_summary_rejects_incomplete_matrix():
    profile = _load_module()

    with pytest.raises(ValueError, match="complete profile matrix"):
        profile.summarize_profile(())


def test_profile_rejects_invalid_inputs():
    profile = _load_module()

    with pytest.raises(ValueError, match="policy"):
        profile.run_profile_case(
            policy="unknown",
            sequence_length=249,
            repetitions=3,
        )
    with pytest.raises(ValueError, match="sequence_length"):
        profile.run_profile_case(
            policy="full_identity",
            sequence_length=250,
            repetitions=3,
        )
    with pytest.raises(ValueError, match="repetitions"):
        profile.run_profile_case(
            policy="full_identity",
            sequence_length=249,
            repetitions=0,
        )


def test_profile_writes_summary_and_jsonl(tmp_path):
    profile = _load_module()
    rows = tuple(
        profile.run_profile_case(
            policy=case["policy"],
            sequence_length=case["sequence_length"],
            repetitions=1,
        )
        for case in profile.build_profile_cases()
    )

    profile.write_profile_artifacts(tmp_path, rows)

    summary = json.loads(
        (tmp_path / "summary.json").read_text()
    )
    written_rows = tuple(
        json.loads(line)
        for line in (tmp_path / "rows.jsonl").read_text().splitlines()
    )
    assert summary["schema"] == (
        "exact_burst_generation_sealed_"
        "lease_identity_cpu_profile_v1"
    )
    assert summary["row_count"] == 6
    assert summary["classification"] in {"GO", "NO_GO"}
    assert set(summary["checks"]) == {
        "8k_lifecycle_median_improvement",
        "8k_lifecycle_p95_improvement",
        "aggregate_lifecycle_median_improvement",
        "candidate_hot_path_identity_rows_zero",
        "candidate_one_cold_capture_per_fixture",
        "candidate_no_fallback_or_rollback_failures",
    }
    assert written_rows == rows


def test_profile_cli_runs_from_repository_root(tmp_path):
    completed = subprocess.run(
        (
            sys.executable,
            str(MODULE_PATH),
            "--output-dir",
            str(tmp_path),
            "--repetitions",
            "1",
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "rows.jsonl").is_file()
