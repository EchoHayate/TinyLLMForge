from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "tools/qwen38_tp4_collective_reduction_worker.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "qwen38_tp4_collective_reduction_worker_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeEngine:

    def __init__(self, outputs_by_arm):
        self.outputs_by_arm = outputs_by_arm
        self.model_runner = SimpleNamespace(rank=0, world_size=4)
        self.configurations = []
        self.reset_calls = 0
        self.exit_calls = 0
        self.arm = None

    def clear_reusable_prefix_cache(self):
        return 0

    def configure_decode_internal_profile(
        self,
        enabled,
        profile_label,
        *,
        timeout_s,
    ):
        assert enabled is False
        assert profile_label == "disabled"
        return {"enabled": False, "rank_inventory": [0, 1, 2, 3]}

    def configure_synchronous_collective_census(
        self,
        policy,
        *,
        timeout_s,
    ):
        self.arm = "instrumented" if policy["enabled"] else "control"
        self.configurations.append(dict(policy))
        return {
            "enabled": policy["enabled"],
            "sample_budget": policy.get("sample_budget", 0),
            "cohort_count": policy.get("cohort_count", 0),
            "rank_inventory": [0, 1, 2, 3],
        }

    def reset_peak_memory_stats(self, *, timeout_s):
        self.reset_calls += 1
        return ()

    def run_requests(self, requests, *, timeout_s):
        return {
            "requests": [
                {
                    "request_id": "request-0",
                    "output_token_ids": list(
                        self.outputs_by_arm[self.arm]
                    ),
                    "ttft_ns": 100,
                    "tpot_ns": 1000,
                    "e2e_ns": 2000,
                }
            ],
            "decode_time_ns": 1000,
        }

    def finalize_synchronous_collective_census(
        self,
        *,
        already_synchronized_rank,
        timeout_s,
    ):
        return {
            "enabled": self.arm == "instrumented",
            "rank_inventory": [0, 1, 2, 3],
            "ranks": [
                {
                    "schema": (
                        "tinyllmforge."
                        "synchronous-collective-census.v1"
                    ),
                    "rank": rank,
                    "enabled": self.arm == "instrumented",
                    "finalization_status": "complete",
                    "steps": [],
                    "collectives": [],
                }
                for rank in range(4)
            ],
        }

    def memory_snapshots(self, *, timeout_s):
        return tuple(
            {
                "rank": rank,
                "cuda_peak_allocated_bytes": 100 + rank,
                "cuda_peak_reserved_bytes": 200 + rank,
            }
            for rank in range(4)
        )

    def exit(self):
        self.exit_calls += 1
        return {
            "process_group_destroyed": True,
            "rank_exit_codes": [0, 0, 0, 0],
            "owned_children_remaining": [],
        }


def test_build_cases_freezes_calibration_and_terminal_matrix():
    worker = _load()
    cases = worker.build_collective_reduction_cases(
        selected_budget=16
    )

    assert {row["budget"] for row in cases["calibration"]} == {
        0, 8, 16, 32
    }
    assert {
        row["workload"] for row in cases["calibration"]
    } == {"P0", "P1", "Q1"}
    assert {
        row["workload"] for row in cases["terminal"]
    } == {"P0", "P1", "Q0", "Q1", "Q2"}
    assert {
        row["budget"] for row in cases["terminal"]
    } == {16}


def test_build_cases_assigns_unique_artifact_identity_to_every_case():
    worker = _load()
    cases = worker.build_collective_reduction_cases(
        selected_budget=16
    )
    rows = cases["calibration"] + cases["terminal"]
    identities = {
        worker.collective_reduction_case_id(**{
            key: row[key]
            for key in (
                "campaign_phase",
                "workload",
                "phase",
                "repetition",
                "budget",
            )
        })
        for row in rows
    }

    assert len(identities) == len(rows)


def test_worker_preserves_exact_request_outputs_across_pair():
    worker = _load()
    engine = _FakeEngine({
        "control": [7, 8],
        "instrumented": [7, 8],
    })
    result = worker.run_collective_reduction_pair(
        engine=engine,
        attempt="attempt-r1",
        source_revision="a" * 40,
        campaign_phase="calibration",
        workload="P0",
        phase="measured",
        repetition=0,
        budget=8,
        timeout_s=30.0,
        reset_sequence_ids=lambda: None,
    )

    assert result["classification"] == "PASS"
    assert [
        row["arm"] for row in result["arms"]
    ] == ["control", "instrumented"]
    assert engine.configurations[0] == {"enabled": False}
    assert engine.configurations[1]["sample_budget"] == 8
    assert engine.configurations[1]["expected_collective_count"] == 130


def test_worker_case_carries_static_workload_identity_for_assembler():
    worker = _load()
    engine = _FakeEngine({
        "control": [7, 8],
        "instrumented": [7, 8],
    })

    result = worker.run_collective_reduction_pair(
        engine=engine,
        attempt="attempt-r1",
        source_revision="a" * 40,
        campaign_phase="calibration",
        workload="P0",
        phase="measured",
        repetition=0,
        budget=8,
        timeout_s=30.0,
        reset_sequence_ids=lambda: None,
    )

    assert (
        result["workload_family"],
        result["prompt_tokens"],
        result["output_tokens"],
        result["concurrency"],
    ) == worker.WORKLOADS["P0"]


def test_worker_clears_reusable_prefix_cache_before_each_arm():
    worker = _load()

    class PrefixSensitiveEngine(_FakeEngine):

        def __init__(self):
            super().__init__({
                "control": [7, 8],
                "instrumented": [7, 8],
            })
            self.prefix_cache_dirty = True
            self.prefix_cache_clear_calls = 0

        def clear_reusable_prefix_cache(self):
            self.prefix_cache_dirty = False
            self.prefix_cache_clear_calls += 1
            return 1

        def run_requests(self, requests, *, timeout_s):
            if self.prefix_cache_dirty:
                raise RuntimeError(
                    "hybrid prefix reuse requires aligned state snapshot"
                )
            result = super().run_requests(
                requests,
                timeout_s=timeout_s,
            )
            self.prefix_cache_dirty = True
            return result

    engine = PrefixSensitiveEngine()

    result = worker.run_collective_reduction_pair(
        engine=engine,
        attempt="attempt-r1",
        source_revision="a" * 40,
        campaign_phase="calibration",
        workload="P1",
        phase="warmup",
        repetition=0,
        budget=0,
        timeout_s=30.0,
        reset_sequence_ids=lambda: None,
    )

    assert result["classification"] == "PASS"
    assert engine.prefix_cache_clear_calls == 2


def test_worker_rejects_output_mismatch_and_closes_once():
    worker = _load()
    engine = _FakeEngine({
        "control": [7, 8],
        "instrumented": [7, 9],
    })

    with pytest.raises(RuntimeError, match="output mismatch"):
        worker.run_collective_reduction_campaign(
            attempt="attempt-r1",
            source_revision="a" * 40,
            cases=[{
                "campaign_phase": "calibration",
                "workload": "P0",
                "phase": "measured",
                "repetition": 0,
                "budget": 8,
            }],
            model_root=Path("/model"),
            timeout_s=30.0,
            engine_factory=lambda *_args, **_kwargs: engine,
            case_sink=lambda _row: None,
            reset_sequence_ids=lambda: None,
        )

    assert engine.exit_calls == 1


def test_campaign_streams_full_cases_and_retains_bounded_receipts():
    worker = _load()
    engine = _FakeEngine({
        "control": [7, 8],
        "instrumented": [7, 8],
    })
    streamed = []
    result = worker.run_collective_reduction_campaign(
        attempt="attempt-r1",
        source_revision="a" * 40,
        cases=[{
            "campaign_phase": "calibration",
            "workload": "P0",
            "phase": "measured",
            "repetition": 0,
            "budget": 0,
        }],
        model_root=Path("/model"),
        timeout_s=30.0,
        engine_factory=lambda *_args, **_kwargs: engine,
        case_sink=streamed.append,
        reset_sequence_ids=lambda: None,
    )

    assert len(streamed) == 1
    assert "arms" in streamed[0]
    assert result["cases"] == [{
        "case_id": "calibration__P0__budget0__measured__r0",
        "classification": "PASS",
        "budget": 0,
    }]
    assert engine.exit_calls == 1


def test_campaign_executes_generated_case_rows():
    worker = _load()
    engine = _FakeEngine({
        "control": [7, 8],
        "instrumented": [7, 8],
    })
    case = worker.build_collective_reduction_cases(
        selected_budget=16
    )["calibration"][0]

    result = worker.run_collective_reduction_campaign(
        attempt="attempt-r1",
        source_revision="a" * 40,
        cases=[case],
        model_root=Path("/model"),
        timeout_s=30.0,
        engine_factory=lambda *_args, **_kwargs: engine,
        reset_sequence_ids=lambda: None,
    )

    assert result["cases"] == [{
        "case_id": "calibration__P0__budget0__warmup__r0",
        "classification": "PASS",
        "budget": 0,
    }]
    assert engine.exit_calls == 1


def test_calibration_cases_select_largest_budget_within_overhead_limits():
    worker = _load()
    cases = []
    for workload in ("P0", "P1", "Q1"):
        for budget, instrumented_ns in (
            (0, 1_000),
            (8, 1_020),
            (16, 1_029),
            (32, 1_060),
        ):
            for repetition in range(5):
                cases.append({
                    "campaign_phase": "calibration",
                    "workload": workload,
                    "phase": "measured",
                    "repetition": repetition,
                    "budget": budget,
                    "arms": [
                        {
                            "arm": "control",
                            "decode_time_ns": 1_000,
                        },
                        {
                            "arm": "instrumented",
                            "decode_time_ns": instrumented_ns,
                        },
                    ],
                })

    assert worker.select_event_budget_from_cases(cases) == 16


def test_full_campaign_merges_actual_phase_receipts():
    worker = _load()
    calibration_case = {
        "campaign_phase": "calibration",
        "workload": "P0",
        "phase": "measured",
        "repetition": 0,
        "budget": 0,
    }
    terminal_case = {
        "campaign_phase": "terminal",
        "workload": "P0",
        "phase": "measured",
        "repetition": 0,
        "budget": 16,
    }
    phase_results = iter((
        {
            "schema_version": worker.WORKER_SCHEMA,
            "classification": "PASS",
            "attempt": "attempt-r1",
            "source_revision": "a" * 40,
            "cases": [{
                "case_id": (
                    "calibration__P0__budget0__measured__r0"
                ),
                "classification": "PASS",
                "budget": 0,
            }],
            "cleanup": {
                "process_group_destroyed": True,
                "rank_exit_codes": [0, 0, 0, 0],
                "owned_children_remaining": [],
            },
        },
        {
            "schema_version": worker.WORKER_SCHEMA,
            "classification": "PASS",
            "attempt": "attempt-r1",
            "source_revision": "a" * 40,
            "cases": [{
                "case_id": "terminal__P0__budget16__measured__r0",
                "classification": "PASS",
                "budget": 16,
            }],
            "cleanup": {
                "process_group_destroyed": True,
                "rank_exit_codes": [0, 0, 0, 0],
                "owned_children_remaining": [],
            },
        },
    ))
    streamed = []

    result = worker.run_full_collective_reduction_campaign(
        attempt="attempt-r1",
        source_revision="a" * 40,
        model_root=Path("/model"),
        timeout_s=30.0,
        case_sink=streamed.append,
        phase_runner=lambda **_kwargs: next(phase_results),
        case_matrix_builder=lambda selected_budget: {
            "calibration": (calibration_case,),
            "terminal": (
                terminal_case
                if selected_budget == 16
                else ()
            ),
        },
        budget_selector=lambda _rows: 16,
        pid_resolver=lambda: 4242,
    )

    assert result["classification"] == "PASS"
    assert result["selected_budget"] == 16
    assert result["owned_pids"] == [4242]
    assert [row["case_id"] for row in result["cases"]] == [
        "calibration__P0__budget0__measured__r0",
        "terminal__P0__budget16__measured__r0",
    ]
    assert len(result["phase_cleanups"]) == 2


def test_full_campaign_calls_case_matrix_builder_by_keyword():
    worker = _load()
    selected_budgets = []

    def build_matrix(*, selected_budget):
        selected_budgets.append(selected_budget)
        return {
            "calibration": (),
            "terminal": (),
        }

    result = worker.run_full_collective_reduction_campaign(
        attempt="attempt-r1",
        source_revision="a" * 40,
        model_root=Path("/model"),
        timeout_s=30.0,
        phase_runner=lambda **_kwargs: {
            "schema_version": worker.WORKER_SCHEMA,
            "classification": "PASS",
            "attempt": "attempt-r1",
            "source_revision": "a" * 40,
            "cases": [],
            "cleanup": {
                "process_group_destroyed": True,
                "rank_exit_codes": [0, 0, 0, 0],
                "owned_children_remaining": [],
            },
        },
        case_matrix_builder=build_matrix,
        budget_selector=lambda _rows: 16,
        pid_resolver=lambda: 4242,
    )

    assert result["classification"] == "PASS"
    assert selected_budgets == [16, 16]


def test_full_campaign_finishes_after_calibration_when_no_budget_passes():
    worker = _load()
    phase_calls = []
    selected_budgets = []

    def build_matrix(*, selected_budget):
        selected_budgets.append(selected_budget)
        return {
            "calibration": ({"case_id": "calibration-case"},),
            "terminal": ({"case_id": "terminal-case"},),
        }

    def run_phase(**kwargs):
        phase_calls.append(tuple(kwargs["cases"]))
        return {
            "schema_version": worker.WORKER_SCHEMA,
            "classification": "PASS",
            "attempt": "attempt-r1",
            "source_revision": "a" * 40,
            "cases": [{
                "case_id": "calibration-case",
                "classification": "PASS",
            }],
            "cleanup": {
                "process_group_destroyed": True,
                "rank_exit_codes": [0, 0, 0, 0],
                "owned_children_remaining": [],
            },
        }

    result = worker.run_full_collective_reduction_campaign(
        attempt="attempt-r1",
        source_revision="a" * 40,
        model_root=Path("/model"),
        timeout_s=30.0,
        phase_runner=run_phase,
        case_matrix_builder=build_matrix,
        budget_selector=lambda _rows: None,
        pid_resolver=lambda: 4242,
    )

    assert result["classification"] == "PASS"
    assert result["selected_budget"] is None
    assert selected_budgets == [16]
    assert phase_calls == [({"case_id": "calibration-case"},)]
    assert len(result["phase_cleanups"]) == 1


def test_worker_cli_full_campaign_selects_budget_internally(
    tmp_path,
    monkeypatch,
    capsys,
):
    worker = _load()
    output = tmp_path / "worker.json"
    output_dir = tmp_path / "cases"
    calls = []
    monkeypatch.setattr(
        worker,
        "run_full_collective_reduction_campaign",
        lambda **kwargs: calls.append(kwargs) or {
            "schema_version": worker.WORKER_SCHEMA,
            "classification": "PASS",
            "attempt": "attempt-r1",
            "source_revision": "a" * 40,
            "selected_budget": 16,
            "owned_pids": [4242],
            "cases": [],
            "phase_cleanups": [],
        },
    )

    return_code = worker.main([
        "--attempt",
        "attempt-r1",
        "--source-revision",
        "a" * 40,
        "--model-root",
        str(tmp_path / "model"),
        "--output",
        str(output),
        "--output-dir",
        str(output_dir),
        "--phase",
        "full",
    ])

    assert return_code == 0
    assert len(calls) == 1
    assert calls[0]["model_root"] == tmp_path / "model"
    assert json.loads(output.read_text())["selected_budget"] == 16
    assert json.loads(capsys.readouterr().out)["selected_budget"] == 16


def test_worker_script_starts_from_an_unrelated_working_directory(
    tmp_path,
):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--help"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
