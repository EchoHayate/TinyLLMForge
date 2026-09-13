from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import profile_slo_cohort_burst_ceiling as profile


def test_direct_script_entrypoint_can_import_tools() -> None:
    result = subprocess.run(
        [sys.executable, str(Path(profile.__file__)), "--help"],
        cwd=Path(profile.__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


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


def _timeline_step(*, wall_ns: int = 100) -> dict:
    names = (
        "scheduler_schedule",
        "partition_and_step_setup",
        "ordinary_or_first_target_dispatch",
        "speculative_prepare",
        "scheduler_prepare_postprocess",
        "proposal_kv_prepare_commit",
        "proposal_lifecycle_finalize_prepare",
        "scheduler_commit_postprocess",
        "proposal_lifecycle_finalize_commit",
        "side_state_seal",
        "residency_precommit_or_seal",
        "ordinary_scheduler_postprocess",
    )
    durations = {
        "scheduler_schedule": 10,
        "partition_and_step_setup": 5,
        "ordinary_or_first_target_dispatch": 60,
        "scheduler_prepare_postprocess": 4,
        "scheduler_commit_postprocess": 3,
        "ordinary_scheduler_postprocess": 2,
    }
    return {
        "step_wall_ns": wall_ns,
        "phases": {
            name: {"duration_ns": durations.get(name, 0)}
            for name in names
        },
    }


def test_timeline_components_leave_only_target_cuda_irreducible() -> None:
    components = profile.components_from_timeline_step(
        _timeline_step(),
    )

    assert components == {
        "target_cuda": 60,
        "graph_launch_gap": 16,
        "scheduler": 19,
        "token_d2h_publication": 0,
        "batch_binding": 5,
        "unattributed": 0,
    }
    assert sum(components.values()) == 100


def test_optimistic_cost_rows_cover_every_supported_width() -> None:
    row = _row("medium", 100, 20)
    row["case_id"] = "medium-b4-c2048-r0"
    row["component_ns"] = profile.components_from_timeline_step(
        _timeline_step(),
    )

    cost_rows = profile.build_optimistic_cost_rows([row])

    assert [item["burst_width"] for item in cost_rows] == [1, 2, 4, 8]
    assert [item["duration_ns"] for item in cost_rows] == [
        100,
        179,
        337,
        653,
    ]


def test_scheduler_and_unattributed_time_are_not_optimistically_removed() -> None:
    row = _row("medium", 100, 20)
    row["component_ns"] = {
        "target_cuda": 50,
        "graph_launch_gap": 10,
        "scheduler": 20,
        "token_d2h_publication": 5,
        "batch_binding": 5,
        "unattributed": 10,
    }

    assert profile._optimistic_headroom_ratio(row) == pytest.approx(0.25)


def test_frozen_case_inventory_covers_loads_batches_and_arrivals() -> None:
    cases = profile.build_frozen_case_inventory("a" * 40)

    assert len(cases) == 12
    assert {
        (case.load, case.batch_size)
        for case in cases
    } == {
        (load, batch_size)
        for load in ("low", "medium", "high")
        for batch_size in (1, 2, 4, 8)
    }
    assert all(case.context_bucket == 2048 for case in cases)
    assert all(case.burst_width == 1 for case in cases)
    assert all(
        len(case.arrival_offsets_ns) == case.batch_size
        for case in cases
    )
    assert all(
        tuple(sorted(case.arrival_offsets_ns))
        == case.arrival_offsets_ns
        for case in cases
    )
    assert {
        case.arrival_offsets_ns
        for case in cases
        if case.batch_size == 4
    } == {
        (0, 4_000_000, 8_000_000, 12_000_000),
        (0, 1_000_000, 2_000_000, 3_000_000),
        (0, 0, 0, 0),
    }


def test_timeline_components_do_not_double_count_async_cuda() -> None:
    components = profile.components_from_timeline_step(
        _timeline_step(wall_ns=150),
        target_cuda_ns=120,
    )

    assert components == {
        "target_cuda": 120,
        "graph_launch_gap": 6,
        "scheduler": 19,
        "token_d2h_publication": 0,
        "batch_binding": 5,
        "unattributed": 0,
    }
    assert sum(components.values()) == 150


class _FakeStepTimer:
    def __init__(self, values):
        self._values = iter(values)

    def measure(self, operation):
        result = operation()
        return result, next(self._values)


class _FakeProfileEngine:
    def __init__(self):
        self._step = 0
        self._finished = False
        self.added = []
        self.last_step_observation = None

    def add_request(self, prompt, sampling_params):
        self.added.append((list(prompt), sampling_params))

    def is_finished(self):
        return self._finished

    def step(self):
        self._step += 1
        self._finished = self._step == 5
        self.last_step_observation = {
            "command_timeline_step": _timeline_step(wall_ns=100),
            "memory": {"cuda_reserved_bytes": 4096},
        }
        return [], (-1 if self._step == 1 else -2)


def test_real_case_profiler_uses_timeline_cuda_and_offered_arrivals() -> None:
    case = profile.CeilingProfileCase(
        case_id="medium-b2-c2048-r0",
        load="medium",
        batch_size=2,
        context_bucket=2048,
        burst_width=1,
        source_commit="a" * 40,
        arrival_offsets_ns=(0, 1_000_000),
        requested_output_tokens=8,
        warmup_steps=1,
        measured_steps=2,
    )
    engine = _FakeProfileEngine()
    clock_values = iter((
        0,
        0,
        100,
        200,
        1_000_000,
        1_000_100,
        1_000_200,
        1_000_300,
        1_000_400,
        1_000_500,
        1_000_600,
        1_000_700,
        1_000_800,
        1_000_900,
        1_001_000,
        1_001_100,
    ))
    sleeps = []

    rows = profile.run_profile_case(
        engine,
        case,
        sampling_params_factory=lambda **kwargs: kwargs,
        step_timer=_FakeStepTimer((60, 61, 62, 63, 64)),
        clock_ns=lambda: next(clock_values),
        sleep=lambda seconds: sleeps.append(seconds),
    )

    assert sleeps == []
    assert len(engine.added) == 2
    assert engine.is_finished() is True
    assert len(rows) == 2
    assert [row["component_ns"]["target_cuda"] for row in rows] == [
        62,
        63,
    ]
    assert all(
        row["offered_arrival_offsets_ns"] == [0, 1_000_000]
        for row in rows
    )
    assert all(row["committed_tokens"] == 2 for row in rows)
    assert all(row["cuda_reserved_bytes"] == 4096 for row in rows)


def test_run_inventory_writes_source_bound_five_file_bundle(
    tmp_path: Path,
) -> None:
    cases = profile.build_frozen_case_inventory("a" * 40)
    source_identity = {
        "source_commit": "a" * 40,
        "source_patch_sha256": "b" * 64,
        "model": "Qwen3-0.6B",
        "checkpoint_sha256": "c" * 64,
        "gpu_uuid": "GPU-a",
        "gpu_name": "NVIDIA A100 80GB PCIe",
        "tensor_parallel_size": 1,
        "config_sha256": "d" * 64,
    }
    engines = []

    def engine_factory(_model, **_config):
        engine = SimpleNamespace(
            exit=lambda: None,
            model_runner=SimpleNamespace(
                config=SimpleNamespace(
                    hf_config=SimpleNamespace(
                        torch_dtype="torch.bfloat16",
                    ),
                ),
            ),
        )
        engines.append(engine)
        return engine

    def case_runner(_engine, case, **_kwargs):
        removable = {"low": 5, "medium": 20, "high": 25}[case.load]
        rows = []
        for sample_index in range(case.measured_steps):
            row = _row(case.load, 100, removable)
            row.update({
                "case_id": case.case_id + f"-s{sample_index}",
                "batch_size": case.batch_size,
                "context_bucket": case.context_bucket,
                "burst_width": case.burst_width,
                "committed_tokens": case.batch_size,
                "offered_arrival_offsets_ns": list(
                    case.arrival_offsets_ns
                ),
            })
            rows.append(row)
        return rows

    receipt = profile.run_profile_inventory(
        model="/models/qwen",
        cases=cases,
        output_dir=tmp_path,
        source_identity=source_identity,
        engine_factory=engine_factory,
        case_runner=case_runner,
        sampling_params_factory=lambda **kwargs: kwargs,
        step_timer_factory=lambda _engine: object(),
    )

    assert receipt["classification"] == "CONTINUE_RUNTIME"
    assert len(engines) == 1
    assert {
        path.name for path in tmp_path.iterdir()
    } == {
        "raw_rows.jsonl",
        "cost_table.json",
        "ceiling_summary.json",
        "source_manifest.json",
        "remote_verify.json",
    }
    manifest = json.loads(
        (tmp_path / "source_manifest.json").read_text()
    )
    assert manifest == {
        **source_identity,
        "dtype": "torch.bfloat16",
    }


def test_frozen_inventory_rejects_a_missing_sample() -> None:
    cases = profile.build_frozen_case_inventory("a" * 40)
    rows = []
    for case in cases:
        for sample_index in range(case.measured_steps):
            row = _row(case.load, 100, 20)
            row.update({
                "case_id": case.case_id + f"-s{sample_index}",
                "batch_size": case.batch_size,
                "context_bucket": case.context_bucket,
                "burst_width": case.burst_width,
                "committed_tokens": case.batch_size,
                "offered_arrival_offsets_ns": list(
                    case.arrival_offsets_ns
                ),
            })
            rows.append(row)

    with pytest.raises(ValueError, match="frozen profile inventory"):
        profile.validate_frozen_profile_inventory(
            rows[:-1],
            source_commit="a" * 40,
        )


def test_bundle_verifier_rebuilds_summary_and_binds_row_source(
    tmp_path: Path,
) -> None:
    cases = profile.build_frozen_case_inventory("a" * 40)
    rows = []
    for case in cases:
        for sample_index in range(case.measured_steps):
            row = _row(case.load, 100, 20)
            row.update({
                "case_id": case.case_id + f"-s{sample_index}",
                "batch_size": case.batch_size,
                "context_bucket": case.context_bucket,
                "burst_width": case.burst_width,
                "committed_tokens": case.batch_size,
                "offered_arrival_offsets_ns": list(
                    case.arrival_offsets_ns
                ),
            })
            rows.append(row)
    source_identity = {
        "source_commit": "a" * 40,
        "source_patch_sha256": "b" * 64,
        "model": "Qwen3-0.6B",
        "checkpoint_sha256": "c" * 64,
        "gpu_uuid": "GPU-a",
        "gpu_name": "NVIDIA A100 80GB PCIe",
        "tensor_parallel_size": 1,
        "dtype": "torch.bfloat16",
        "config_sha256": "d" * 64,
    }
    profile.write_ceiling_bundle(
        output_dir=tmp_path,
        profile_rows=rows,
        cost_rows=profile.build_optimistic_cost_rows(rows),
        source_identity=source_identity,
    )
    summary_path = tmp_path / "ceiling_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["medium_headroom_ratio"] = 99.0
    summary["classification"] = "CONTINUE_RUNTIME"
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="summary does not match"):
        profile.verify_ceiling_bundle(tmp_path)


def test_cli_supports_run_and_verify_modes(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(
        profile,
        "run_cli",
        lambda args: calls.append(("run", args.run_tag)) or 0,
    )
    monkeypatch.setattr(
        profile,
        "verify_cli",
        lambda args: calls.append(("verify", args.artifact_dir)) or 0,
    )

    assert profile.main([
        "--mode",
        "run",
        "--model",
        "/models/qwen",
        "--run-tag",
        "stage0-r1",
        "--source-commit",
        "a" * 40,
        "--output-dir",
        str(tmp_path / "run"),
    ]) == 0
    assert profile.main([
        "--mode",
        "verify",
        "--artifact-dir",
        str(tmp_path / "run"),
        "--output",
        str(tmp_path / "verify.json"),
    ]) == 0
    assert calls == [
        ("run", "stage0-r1"),
        ("verify", str(tmp_path / "run")),
    ]
