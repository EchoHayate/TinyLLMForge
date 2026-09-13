from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from tools import profile_slo_cohort_burst_ceiling as profile


@dataclass(frozen=True)
class FakeCase:
    case_id: str = "medium-b4-c2048-k4-r0"
    load: str = "medium"
    batch_size: int = 4
    context_bucket: int = 2048
    burst_width: int = 4
    source_commit: str = "a" * 40


class FakeEngine:
    def profile_baseline_case(self, case, *, clock_ns):
        assert case == FakeCase()
        assert clock_ns() == 101
        return {
            "target_cuda_ns": 50,
            "graph_launch_gap_ns": 10,
            "scheduler_ns": 15,
            "token_d2h_publication_ns": 15,
            "batch_binding_ns": 5,
            "wall_ns": 100,
            "committed_tokens": 4,
            "cuda_reserved_bytes": 1_024,
        }


def test_profile_baseline_case_attributes_wall_time_once() -> None:
    row = profile.profile_baseline_case(
        FakeEngine(),
        FakeCase(),
        clock_ns=lambda: 101,
    )
    assert row["schema_version"] == profile.PROFILE_ROW_SCHEMA_VERSION
    assert row["component_ns"] == {
        "target_cuda": 50,
        "graph_launch_gap": 10,
        "scheduler": 15,
        "token_d2h_publication": 15,
        "batch_binding": 5,
        "unattributed": 5,
    }
    assert sum(row["component_ns"].values()) == row["wall_ns"]


def _row(load: str, wall_ns: int, removable_ns: int) -> dict:
    return {
        "schema_version": profile.PROFILE_ROW_SCHEMA_VERSION,
        "case_id": f"{load}-r0",
        "load": load,
        "source_commit": "a" * 40,
        "batch_size": 4,
        "context_bucket": 2048,
        "burst_width": 4,
        "component_ns": {
            "target_cuda": wall_ns - removable_ns,
            "graph_launch_gap": removable_ns // 2,
            "scheduler": 0,
            "token_d2h_publication": removable_ns - removable_ns // 2,
            "batch_binding": 0,
            "unattributed": 0,
        },
        "wall_ns": wall_ns,
        "committed_tokens": 4,
        "cuda_reserved_bytes": 1_024,
    }


def test_build_ceiling_summary_computes_optimistic_headroom() -> None:
    summary = profile.build_ceiling_summary([
        _row("low", 100, 5),
        _row("medium", 100, 20),
        _row("high", 100, 25),
    ])
    assert summary["evidence_complete"] is True
    assert summary["source_exact"] is True
    assert summary["medium_headroom_ratio"] == pytest.approx(0.25)
    assert summary["high_headroom_ratio"] == pytest.approx(1 / 3)
    assert summary["classification"] == "CONTINUE_RUNTIME"


def test_summary_requires_each_frozen_load_and_unique_case_ids() -> None:
    with pytest.raises(ValueError, match="load inventory"):
        profile.build_ceiling_summary([
            _row("low", 100, 5),
            _row("medium", 100, 20),
        ])

    rows = [
        _row("low", 100, 5),
        _row("medium", 100, 20),
        _row("high", 100, 25),
    ]
    rows[-1]["case_id"] = rows[0]["case_id"]
    with pytest.raises(ValueError, match="duplicate case ID"):
        profile.build_ceiling_summary(rows)


def test_profile_row_rejects_overlapping_or_invalid_components() -> None:
    engine = FakeEngine()
    original = engine.profile_baseline_case

    def invalid(case, *, clock_ns):
        row = original(case, clock_ns=clock_ns)
        row["target_cuda_ns"] = 101
        return row

    engine.profile_baseline_case = invalid
    with pytest.raises(ValueError, match="exceed wall time"):
        profile.profile_baseline_case(
            engine,
            FakeCase(),
            clock_ns=lambda: 101,
        )


def test_write_ceiling_bundle_emits_the_immutable_stage0_inventory(
    tmp_path: Path,
) -> None:
    rows = [
        _row("low", 100, 5),
        _row("medium", 100, 20),
        _row("high", 100, 25),
    ]
    source_identity = {
        "source_commit": "a" * 40,
        "source_patch_sha256": "b" * 64,
        "model": "Qwen3-0.6B",
        "checkpoint_sha256": "c" * 64,
        "gpu_uuid": "GPU-a",
        "gpu_name": "NVIDIA A100 80GB PCIe",
        "tensor_parallel_size": 1,
        "dtype": "float16",
        "config_sha256": "d" * 64,
    }
    cost_rows = [
        {
            "schema_version": "slo-cohort-burst.cost-sample.v1",
            "sample_id": f"{row['case_id']}-sample",
            "batch_size": row["batch_size"],
            "context_bucket": row["context_bucket"],
            "burst_width": row["burst_width"],
            "duration_ns": row["wall_ns"],
        }
        for row in rows
    ]

    receipt = profile.write_ceiling_bundle(
        output_dir=tmp_path,
        profile_rows=rows,
        cost_rows=cost_rows,
        source_identity=source_identity,
    )

    assert receipt["classification"] == "CONTINUE_RUNTIME"
    assert {
        path.name for path in tmp_path.iterdir()
    } == {
        "raw_rows.jsonl",
        "cost_table.json",
        "ceiling_summary.json",
        "source_manifest.json",
        "remote_verify.json",
    }
    assert len(
        (tmp_path / "raw_rows.jsonl").read_text().splitlines()
    ) == 3
    verify = json.loads(
        (tmp_path / "remote_verify.json").read_text()
    )
    assert verify["verified"] is True
    assert verify["classification"] == "CONTINUE_RUNTIME"


def test_write_ceiling_bundle_is_immutable(tmp_path: Path) -> None:
    (tmp_path / "raw_rows.jsonl").write_text("{}\n")

    with pytest.raises(ValueError, match="destination is not empty"):
        profile.write_ceiling_bundle(
            output_dir=tmp_path,
            profile_rows=[],
            cost_rows=[],
            source_identity={},
        )
