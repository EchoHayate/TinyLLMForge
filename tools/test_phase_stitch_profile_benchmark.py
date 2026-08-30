#!/usr/bin/env python3
"""Contracts for the paired Phase-Stitch profile benchmark."""

from __future__ import annotations

from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _modules():
    return tuple(
        importlib.import_module(name)
        for name in (
            "tools.phase_stitch_profile_contract",
            "tools.phase_stitch_profile_worker",
        )
    )


class StepClock:

    def __init__(self, step_ns=1_000_000):
        self.value = 0
        self.step_ns = step_ns

    def __call__(self):
        self.value += self.step_ns
        return self.value


class FakePrefillCache:

    def __init__(self):
        self.replays = 0

    def summary(self):
        return {
            "capture_attempts": 2,
            "capture_successes": 2,
            "capture_failures": 0,
            "replays": self.replays,
            "replay_failures": 0,
            "quarantines": 0,
            "fallbacks": 0,
            "static_bytes": 4_764_704,
            "reserved_delta_bytes": 41_943_040,
        }


class FakeRunner:

    def __init__(self):
        self.exact_prefill_cuda_graph_cache = FakePrefillCache()
        self.graph_replays = 0
        self.acceptances = 0
        self.exited = False

    def reset_peak_memory_stats(self):
        return self.memory_snapshot()

    @staticmethod
    def memory_snapshot():
        return {
            "cuda_allocated_bytes": 2_000_000_000,
            "cuda_reserved_bytes": 2_100_000_000,
            "cuda_peak_allocated_bytes": 2_010_000_000,
            "cuda_peak_reserved_bytes": 2_120_000_000,
        }

    def exact_greedy_decode_burst_summary(self):
        return {
            "attempts": self.acceptances,
            "acceptances": self.acceptances,
            "graph_replays": self.graph_replays,
            "failures": 0,
            "quarantines": 0,
            "pending_leases": 0,
            "fallback_counts": {},
            "quarantine_reason": None,
        }


class FakeEngine:

    def __init__(self, spec):
        self.profile_enabled = (
            spec["arm"] == "instrumentation_on"
        )
        self.model_runner = FakeRunner()
        self.tokenizer = SimpleNamespace(
            decode=lambda token_ids: " ".join(map(str, token_ids))
        )
        self.pending = None
        self.generated = []
        self.profile_rows = []
        self.last_step_observation = None
        self.clear_calls = 0
        self.exited = False

    def add_request(self, prompt, sampling_params):
        self.pending = {
            "prompt": list(prompt),
            "max_tokens": sampling_params.max_tokens,
        }
        self.generated = []

    def is_finished(self):
        return (
            self.pending is None
            or len(self.generated) >= self.pending["max_tokens"]
        )

    def step(self, *, completion_only):
        assert completion_only is True
        first_step = not self.generated
        emitted = min(
            1 if first_step else 8,
            self.pending["max_tokens"] - len(self.generated),
        )
        tokens = [
            1000 + len(self.generated) + index
            for index in range(emitted)
        ]
        self.generated.extend(tokens)
        if first_step:
            self.model_runner.exact_prefill_cuda_graph_cache.replays += 1
        else:
            self.model_runner.graph_replays += 1
            self.model_runner.acceptances += 1
        self.last_step_observation = {
            "is_prefill": first_step,
            "new_completion_tokens_by_seq": {1: tokens},
        }
        if self.is_finished() and self.profile_enabled:
            output_hash = _modules()[0].canonical_json_sha256(
                self.generated
            )
            base = 10_000_000 + len(self.profile_rows) * 1_000_000
            events = (
                "prefill_dispatch_finished",
                "first_token_host_available",
                "prefill_scheduler_commit_finished",
                "next_schedule_started",
                "next_schedule_finished",
                "k8_lease_prepare_finished",
                "first_k8_dispatch_started",
            )
            timestamps = (
                base,
                base + 100_000,
                base + 200_000,
                base + 300_000,
                base + 400_000,
                base + 500_000,
                base + 700_000,
            )
            self.profile_rows.append({
                "sequence_id": len(self.profile_rows),
                "prompt_tokens": len(self.pending["prompt"]),
                "status": "complete",
                "events": list(events),
                **{
                    f"{event}_ns": timestamp
                    for event, timestamp in zip(events, timestamps)
                },
                "adjacent_intervals_ns": {},
                "removable_host_gap_ns": 600_000,
                "output_token_ids_sha256": output_hash,
                "event_coverage_complete": True,
            })
        outputs = (
            [(1, list(self.generated))]
            if self.is_finished()
            else []
        )
        return outputs, -emitted

    def phase_stitch_profile_snapshot(self):
        return {
            "schema_version": 1,
            "enabled": self.profile_enabled,
            "active": [],
            "rows": deepcopy(self.profile_rows),
        }

    def configure_phase_stitch_profile(self, enabled):
        assert enabled is self.profile_enabled
        self.profile_rows = []
        return {"enabled": enabled, "schema_version": 1}

    def clear_reusable_prefix_cache(self):
        self.clear_calls += 1
        return 1

    def exit(self):
        self.exited = True
        return {"rank": 0, "process_group_destroyed": True}


def test_contract_freezes_balanced_pair_matrix_and_engine_controls():
    contract, _worker = _modules()
    cases = contract.build_case_matrix()

    assert contract.ARMS == (
        "instrumentation_off",
        "instrumentation_on",
    )
    assert contract.PROMPT_TOKEN_COUNTS == (256, 2048)
    assert contract.ROUNDS == 2
    assert contract.WARMUP_REPETITIONS == 2
    assert contract.MEASURED_REPETITIONS == 5
    assert contract.GENERATED_TOKENS == 128
    assert len(cases) == 8
    assert [
        (row["round"], row["prompt_tokens"], row["arm"])
        for row in cases
    ] == [
        (0, 256, "instrumentation_off"),
        (0, 256, "instrumentation_on"),
        (0, 2048, "instrumentation_off"),
        (0, 2048, "instrumentation_on"),
        (1, 256, "instrumentation_on"),
        (1, 256, "instrumentation_off"),
        (1, 2048, "instrumentation_on"),
        (1, 2048, "instrumentation_off"),
    ]
    for case in cases:
        config = case["engine_config"]
        assert config["tensor_parallel_size"] == 1
        assert config["max_num_seqs"] == 1
        assert config["prefill_cuda_graphs"] is True
        assert config["prefill_cuda_graph_token_allowlist"] == [256, 2048]
        assert config["exact_greedy_decode_burst"] is True
        assert config["exact_greedy_decode_burst_tokens"] == 8
        assert config["phase_stitch_profile"] is (
            case["arm"] == "instrumentation_on"
        )


def test_contract_hash_and_case_validation_fail_closed():
    contract, _worker = _modules()
    first = contract.build_case_matrix()[0]

    assert len(contract.contract_sha256()) == 64
    assert contract.validate_case_spec(first) == first

    drifted = deepcopy(first)
    drifted["engine_config"]["exact_greedy_decode_burst_tokens"] = 4
    with pytest.raises(ValueError, match="frozen contract"):
        contract.validate_case_spec(drifted)


@pytest.mark.parametrize(
    "arm",
    ("instrumentation_off", "instrumentation_on"),
)
def test_worker_runs_isolated_case_and_writes_valid_result(
    tmp_path,
    arm,
):
    contract, worker = _modules()
    spec = next(
        case
        for case in contract.build_case_matrix()
        if case["arm"] == arm and case["prompt_tokens"] == 256
    )
    engines = []

    def engine_factory(case):
        engine = FakeEngine(case)
        engines.append(engine)
        return engine

    output_dir = tmp_path / arm
    result = worker.run_worker(
        spec,
        model="/models/Qwen3-0.6B",
        output_dir=output_dir,
        engine_factory=engine_factory,
        clock_ns=StepClock(),
        synchronize=lambda: None,
    )

    assert engines[0].exited is True
    assert len(result["rows"]) == contract.MEASURED_REPETITIONS
    assert result == json.loads(
        (output_dir / "result.json").read_text(encoding="utf-8")
    )
    assert all(
        len(row["output_token_ids"]) == contract.GENERATED_TOKENS
        for row in result["rows"]
    )
    assert all(
        row["prefill_graph_replay_delta"] == 1
        and row["exact_burst_replay_delta"] > 0
        and row["exact_burst_acceptance_delta"] > 0
        for row in result["rows"]
    )
    if arm == "instrumentation_on":
        assert all(
            row["phase_stitch_profile"]["removable_host_gap_ns"]
            == 600_000
            for row in result["rows"]
        )
    else:
        assert all(
            row["phase_stitch_profile"] is None
            for row in result["rows"]
        )
    contract.validate_case_result(result)


def test_worker_rejects_existing_output_directory(tmp_path):
    contract, worker = _modules()
    spec = contract.build_case_matrix()[0]
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    with pytest.raises(FileExistsError):
        worker.run_worker(
            spec,
            model="/models/Qwen3-0.6B",
            output_dir=output_dir,
            engine_factory=FakeEngine,
            clock_ns=StepClock(),
            synchronize=lambda: None,
        )


def test_result_validation_rejects_missing_profile_and_bad_tokens(
    tmp_path,
):
    contract, worker = _modules()
    spec = next(
        case
        for case in contract.build_case_matrix()
        if case["arm"] == "instrumentation_on"
        and case["prompt_tokens"] == 256
    )
    result = worker.run_worker(
        spec,
        model="/models/Qwen3-0.6B",
        output_dir=tmp_path / "case",
        engine_factory=FakeEngine,
        clock_ns=StepClock(),
        synchronize=lambda: None,
    )

    missing_profile = deepcopy(result)
    missing_profile["rows"][0]["phase_stitch_profile"] = None
    with pytest.raises(ValueError, match="profile"):
        contract.validate_case_result(missing_profile)

    bad_tokens = deepcopy(result)
    bad_tokens["rows"][0]["output_token_ids"] = [1]
    with pytest.raises(ValueError, match="token"):
        contract.validate_case_result(bad_tokens)
