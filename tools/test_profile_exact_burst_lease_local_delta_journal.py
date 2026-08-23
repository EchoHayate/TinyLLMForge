from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tools"
    / "profile_exact_burst_lease_local_delta_journal.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "profile_exact_burst_lease_local_delta_journal_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_profile_cases_are_fixed_and_complete():
    profile = _load_module()

    cases = profile.build_profile_cases()

    assert len(cases) == 6
    assert {
        (row["policy"], row["sequence_length"])
        for row in cases
    } == {
        (policy, length)
        for policy in ("generic", "lease_local_delta")
        for length in (249, 2041, 8185)
    }


@pytest.mark.parametrize(
    "policy",
    ("generic", "lease_local_delta"),
)
def test_profile_case_reports_timing_allocation_and_authority(
    policy,
):
    profile = _load_module()

    row = profile.run_profile_case(
        policy=policy,
        sequence_length=249,
        repetitions=3,
    )

    assert row["schema"] == (
        "exact_burst_lease_local_delta_journal_cpu_profile_v1"
    )
    assert row["policy"] == policy
    assert row["sequence_length"] == 249
    assert row["sample_count"] == 3
    assert row["prepare_median_us"] >= 0
    assert row["prepare_p95_us"] >= row["prepare_median_us"]
    assert row["positive_python_allocation_bytes"] >= 0
    assert row["journal_touched_block_count"] >= 0
    assert row["journal_hash_key_count"] >= 0
    assert row["compute_hash_calls"] >= 0
    if policy == "lease_local_delta":
        assert row["journal_touched_block_count"] <= 1
        assert row["journal_hash_key_count"] <= 1
        assert row["delta_attempts"] == 13
        assert row["delta_captures"] == 13
        assert row["delta_fallbacks"] == {}
    else:
        assert row["delta_attempts"] == 0
        assert row["delta_captures"] == 0


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
            policy="generic",
            sequence_length=250,
            repetitions=3,
        )
    with pytest.raises(ValueError, match="repetitions"):
        profile.run_profile_case(
            policy="generic",
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

    summary_path = tmp_path / "summary.json"
    rows_path = tmp_path / "rows.jsonl"
    assert summary_path.is_file()
    assert rows_path.is_file()
    summary = json.loads(summary_path.read_text())
    written_rows = tuple(
        json.loads(line)
        for line in rows_path.read_text().splitlines()
    )
    assert summary["schema"] == (
        "exact_burst_lease_local_delta_journal_cpu_profile_v1"
    )
    assert summary["row_count"] == 6
    assert written_rows == rows
