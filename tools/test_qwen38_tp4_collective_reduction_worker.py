from __future__ import annotations

import importlib.util
from pathlib import Path
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
        workload="P0",
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
                "workload": "P0",
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
            "workload": "P0",
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
        "case_id": "P0__budget0__r0",
        "classification": "PASS",
        "budget": 0,
    }]
    assert engine.exit_calls == 1
