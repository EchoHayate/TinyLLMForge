#!/usr/bin/env python3
"""Contracts for the exact-prefill CUDA Graph paired benchmark."""

from __future__ import annotations

from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _modules():
    return tuple(
        importlib.import_module(name)
        for name in (
            "tools.exact_prefill_cuda_graph_benchmark_contract",
            "tools.exact_prefill_cuda_graph_benchmark_worker",
            "tools.exact_prefill_cuda_graph_gate",
            "tools.exact_prefill_cuda_graph_verify",
        )
    )


class StepClock:
    def __init__(self, step_ns: int = 1_000_000):
        self.value = 0
        self.step_ns = step_ns

    def __call__(self) -> int:
        self.value += self.step_ns
        return self.value


class FakeCache:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.replays = 0

    def summary(self) -> dict:
        return {
            "ready_entries": (
                ("a" * 64, "b" * 64) if self.enabled else ()
            ),
            "capture_attempts": 2 if self.enabled else 0,
            "capture_successes": 2 if self.enabled else 0,
            "capture_failures": 0,
            "replays": self.replays,
            "replay_failures": 0,
            "quarantines": 0,
            "fallbacks": 0,
            "static_bytes": 1_058_848 if self.enabled else 0,
            "allocated_delta_bytes": 0,
            "reserved_delta_bytes": 46_137_344 if self.enabled else 0,
            "total_capture_ns": 730_000_000 if self.enabled else 0,
            "quarantined": {},
            "capture_errors_by_token": {},
        }


class FakeRunner:
    def __init__(self, enabled: bool):
        self.exact_prefill_cuda_graph_cache = FakeCache(enabled)

    def reset_peak_memory_stats(self):
        return self.memory_snapshot()

    @staticmethod
    def memory_snapshot():
        return {
            "cuda_allocated_bytes": 2_000_000_000,
            "cuda_reserved_bytes": 2_100_000_000,
            "cuda_peak_allocated_bytes": 2_010_000_000,
            "cuda_peak_reserved_bytes": 2_120_000_000,
            "kv_capacity_bytes": 128_000_000,
        }


class FakeEngine:
    def __init__(self, spec: dict):
        enabled = spec["arm"] == "exact_prefill_graph"
        self.model_runner = FakeRunner(enabled)
        self.pending = None
        self.generated = []
        self.last_step_observation = None
        self.clear_calls = 0
        self.exited = False
        self.tokenizer = SimpleNamespace(
            decode=lambda token_ids: " ".join(map(str, token_ids))
        )

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
        token = 1000 + len(self.generated)
        self.generated.append(token)
        if (
            len(self.generated) == 1
            and self.model_runner.exact_prefill_cuda_graph_cache.enabled
        ):
            self.model_runner.exact_prefill_cuda_graph_cache.replays += 1
        self.last_step_observation = {
            "is_prefill": len(self.generated) == 1,
            "new_completion_tokens_by_seq": {1: [token]},
            "memory": self.model_runner.memory_snapshot(),
        }
        outputs = (
            [(1, list(self.generated))]
            if self.is_finished()
            else []
        )
        return outputs, 1

    def clear_reusable_prefix_cache(self):
        self.clear_calls += 1
        return 1

    def exit(self):
        self.exited = True
        return {"rank": 0, "process_group_destroyed": True}


def _case_result(
    contract,
    case: dict,
    *,
    ttft_ns: int,
    tpot_ns: int = 2_000_000,
    e2e_ns: int = 31_000_000,
) -> dict:
    rows = []
    graph = case["arm"] == "exact_prefill_graph"
    for sample_index in range(contract.MEASURED_REPETITIONS):
        rows.append({
            "schema_version": contract.ROW_SCHEMA_VERSION,
            "case_id": case["case_id"],
            "round": case["round"],
            "order_position": case["order_position"],
            "arm": case["arm"],
            "prompt_tokens": case["prompt_tokens"],
            "sample_index": sample_index,
            "generated_tokens": contract.GENERATED_TOKENS,
            "prompt_sha256": f"{sample_index + 1:064x}",
            "output_token_ids": list(range(contract.GENERATED_TOKENS)),
            "output_text_sha256": "c" * 64,
            "ttft_ns": ttft_ns,
            "tpot_samples_ns": [
                tpot_ns
            ] * (contract.GENERATED_TOKENS - 1),
            "e2e_ns": e2e_ns,
            "output_tokens_per_second": (
                contract.GENERATED_TOKENS / (e2e_ns / 1e9)
            ),
            "prefill_graph_replay_delta": 1 if graph else 0,
            "cuda_peak_allocated_bytes": 2_010_000_000,
            "cuda_peak_reserved_bytes": (
                2_146_137_344 if graph else 2_100_000_000
            ),
        })
    return {
        "schema_version": contract.RESULT_SCHEMA_VERSION,
        "case": deepcopy(case),
        "rows": rows,
        "prefill_graph_summary": {
            "ready_entries": (
                ["a" * 64, "b" * 64] if graph else []
            ),
            "capture_attempts": 2 if graph else 0,
            "capture_successes": 2 if graph else 0,
            "capture_failures": 0,
            "replays": (
                contract.WARMUP_REPETITIONS
                + contract.MEASURED_REPETITIONS
                if graph
                else 0
            ),
            "replay_failures": 0,
            "quarantines": 0,
            "fallbacks": 0,
            "static_bytes": 1_058_848 if graph else 0,
            "allocated_delta_bytes": 0,
            "reserved_delta_bytes": 46_137_344 if graph else 0,
            "total_capture_ns": 730_000_000 if graph else 0,
            "quarantined": {},
            "capture_errors_by_token": {},
        },
    }


def _write_complete_fixture(root: Path):
    contract, _worker, gate, _verifier = _modules()
    run_dir = root / "run"
    cases_dir = run_dir / "cases"
    cases_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({
            "schema_version": contract.RUN_SCHEMA_VERSION,
            "run_tag": "synthetic-exact-prefill-r1",
            "source_base_commit": "1" * 40,
            "source_files": {
                relative: "a" * 64
                for relative in contract.SOURCE_FILES
            },
            "model": "/models/Qwen3-0.6B",
            "python": "/env/bin/python",
            "cuda_visible_devices": "0",
            "clean_gpu_admission": True,
            "gpu_inventory": [{
                "index": 0,
                "uuid": "GPU-fixture",
                "name": "NVIDIA A100 80GB PCIe",
                "memory_total_mb": 81920,
                "memory_used_mb": 0,
                "utilization_gpu_percent": 0,
                "compute_processes": [],
            }],
            "case_order": list(contract.expected_case_ids()),
            "contract_sha256": contract.contract_sha256(),
        }, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for case in contract.build_case_matrix():
        eager = case["arm"] == "eager"
        ttft_ns = (
            40_000_000
            if eager
            else (
                28_000_000
                if case["prompt_tokens"] == 256
                else 40_400_000
            )
        )
        result = _case_result(
            contract,
            case,
            ttft_ns=ttft_ns,
            tpot_ns=2_000_000 if eager else 2_010_000,
            e2e_ns=31_000_000 if eager else 31_200_000,
        )
        destination = cases_dir / case["case_id"]
        destination.mkdir()
        (destination / "result.json").write_text(
            json.dumps(result, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return contract, gate, run_dir


def test_contract_freezes_balanced_pair_matrix_and_engine_controls():
    contract, _worker, _gate, _verifier = _modules()
    cases = contract.build_case_matrix()

    assert contract.ARMS == ("eager", "exact_prefill_graph")
    assert contract.PROMPT_TOKEN_COUNTS == (256, 2048)
    assert contract.WARMUP_REPETITIONS == 2
    assert contract.MEASURED_REPETITIONS == 5
    assert contract.GENERATED_TOKENS == 16
    assert len(cases) == 8
    assert [
        (row["round"], row["prompt_tokens"], row["arm"])
        for row in cases
    ] == [
        (0, 256, "eager"),
        (0, 256, "exact_prefill_graph"),
        (0, 2048, "eager"),
        (0, 2048, "exact_prefill_graph"),
        (1, 256, "exact_prefill_graph"),
        (1, 256, "eager"),
        (1, 2048, "exact_prefill_graph"),
        (1, 2048, "eager"),
    ]
    for case in cases:
        config = case["engine_config"]
        assert config["tensor_parallel_size"] == 1
        assert config["enforce_eager"] is False
        assert (
            config["max_num_batched_tokens"]
            >= config["max_model_len"]
        )
        assert config["prefill_cuda_graph_token_allowlist"] == [256, 2048]
        assert config["prefill_cuda_graphs"] is (
            case["arm"] == "exact_prefill_graph"
        )
        assert case["sampling"] == {
            "temperature": 0.0,
            "max_tokens": 16,
            "ignore_eos": True,
        }


def test_worker_records_exact_rows_replay_and_capture_cost(tmp_path: Path):
    contract, worker, _gate, _verifier = _modules()
    case = next(
        row
        for row in contract.build_case_matrix()
        if row["arm"] == "exact_prefill_graph"
        and row["prompt_tokens"] == 256
    )
    engine = None

    def engine_factory(spec):
        nonlocal engine
        engine = FakeEngine(spec)
        return engine

    result = worker.run_worker(
        case,
        model="/models/Qwen3-0.6B",
        output_dir=tmp_path / "case",
        engine_factory=engine_factory,
        clock_ns=StepClock(),
        synchronize=lambda: None,
    )

    assert len(result["rows"]) == contract.MEASURED_REPETITIONS
    assert all(
        row["prefill_graph_replay_delta"] == 1
        for row in result["rows"]
    )
    assert all(
        len(row["tpot_samples_ns"])
        == contract.GENERATED_TOKENS - 1
        for row in result["rows"]
    )
    assert result["prefill_graph_summary"]["capture_successes"] == 2
    assert result["prefill_graph_summary"]["total_capture_ns"] > 0
    assert (
        result["prefill_graph_summary"]["reserved_delta_bytes"] > 0
    )
    assert engine is not None
    assert engine.clear_calls == (
        contract.WARMUP_REPETITIONS
        + contract.MEASURED_REPETITIONS
    )
    assert engine.exited is True
    persisted = json.loads(
        (tmp_path / "case" / "result.json").read_text()
    )
    assert persisted == result


def test_gate_classifies_complete_exact_fixture_go(tmp_path: Path):
    contract, gate, run_dir = _write_complete_fixture(tmp_path)
    result = gate.produce_gate(run_dir)
    comparison = json.loads(
        (run_dir / "comparison.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )

    assert result["classification"] == gate.GO_EXACT_PREFILL_GRAPH
    assert comparison["schema_version"] == (
        contract.COMPARISON_SCHEMA_VERSION
    )
    assert summary["schema_version"] == contract.GATE_SCHEMA_VERSION
    assert result["correctness"]["all_token_ids_exact"] is True
    assert result["mechanism"]["candidate_replayed_every_sample"] is True
    assert (
        result["performance"]["256"][
            "ttft_improvement_fraction"
        ]
        >= contract.TTFT_256_IMPROVEMENT_MINIMUM
    )
    assert (
        result["performance"]["2048"]["ttft_regression_fraction"]
        <= contract.TTFT_2048_REGRESSION_LIMIT
    )
    assert result["cost"]["capture_duration_ns"]["available"] is True
    assert result["cost"]["reserved_delta_bytes"]["available"] is True
    assert {
        "run_manifest.json",
        "comparison.json",
        "summary.json",
        "manifest.json",
        "report.md",
    } <= {path.name for path in run_dir.iterdir()}


def test_gate_accepts_any_single_clean_selected_gpu(tmp_path: Path):
    _contract, gate, run_dir = _write_complete_fixture(tmp_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cuda_visible_devices"] = "3"
    manifest["gpu_inventory"][0]["index"] = 3
    manifest["gpu_inventory"][0]["uuid"] = "GPU-fixture-3"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    assert gate.produce_gate(run_dir)["classification"] == (
        gate.GO_EXACT_PREFILL_GRAPH
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("token", "NO_GO_CORRECTNESS"),
        ("ttft256", "NO_GO_PERFORMANCE"),
        ("tpot", "NO_GO_PERFORMANCE"),
        ("e2e", "NO_GO_PERFORMANCE"),
        ("replay", "NO_GO_MECHANISM"),
        ("cost", "NO_GO_EVIDENCE_INCOMPLETE"),
        ("row_contract", "NO_GO_EVIDENCE_INCOMPLETE"),
    ),
)
def test_gate_fails_closed_for_each_frozen_requirement(
    tmp_path: Path,
    mutation: str,
    expected: str,
):
    contract, gate, run_dir = _write_complete_fixture(tmp_path)
    candidate_paths = sorted(
        path
        for path in (run_dir / "cases").glob("*/result.json")
        if "exact-prefill-graph" in path.parent.name
    )
    assert candidate_paths
    targets = candidate_paths
    if mutation == "ttft256":
        targets = [
            path for path in targets if "p256" in path.parent.name
        ]
    for path in targets:
        result = json.loads(path.read_text())
        if mutation == "token":
            result["rows"][0]["output_token_ids"][-1] = 9999
        elif mutation == "ttft256":
            for row in result["rows"]:
                row["ttft_ns"] = 39_000_000
        elif mutation == "tpot":
            for row in result["rows"]:
                row["tpot_samples_ns"] = [2_100_000] * 15
        elif mutation == "e2e":
            for row in result["rows"]:
                row["e2e_ns"] = 32_000_000
        elif mutation == "replay":
            result["rows"][0]["prefill_graph_replay_delta"] = 0
        elif mutation == "cost":
            result["prefill_graph_summary"].pop(
                "reserved_delta_bytes"
            )
        elif mutation == "row_contract":
            result["rows"][0]["schema_version"] = "wrong.row.schema"
        path.write_text(
            json.dumps(result, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    assert gate.produce_gate(run_dir)["classification"] == expected


def test_independent_verifier_reconstructs_and_rejects_tamper(
    tmp_path: Path,
):
    _contract, gate, run_dir = _write_complete_fixture(tmp_path)
    gate.produce_gate(run_dir)
    _contract, _worker, _gate, verifier = _modules()

    receipt = verifier.verify_artifact_directory(run_dir)
    assert receipt["verified"] is True
    assert receipt["classification"] == gate.GO_EXACT_PREFILL_GRAPH
    assert receipt["raw_metrics_reconstructed"] is True
    assert "exact_prefill_cuda_graph_gate" not in (
        ROOT
        / "tools"
        / "exact_prefill_cuda_graph_verify.py"
    ).read_text(encoding="utf-8")

    comparison = json.loads(
        (run_dir / "comparison.json").read_text(encoding="utf-8")
    )
    comparison["classification"] = "NO_GO_PERFORMANCE"
    (run_dir / "comparison.json").write_text(
        json.dumps(comparison, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest hash"):
        verifier.verify_artifact_directory(run_dir)


def test_independent_verifier_rejects_nonpositive_latency_samples(
    tmp_path: Path,
):
    _contract, _gate, run_dir = _write_complete_fixture(tmp_path)
    _contract, _worker, _gate, verifier = _modules()
    results = verifier._load_results(run_dir)

    for result in results:
        graph = result["case"]["arm"] == "exact_prefill_graph"
        for row in result["rows"]:
            row["ttft_ns"] = -50 if graph else -100
            row["e2e_ns"] = -50 if graph else -100
            row["tpot_samples_ns"] = [
                -50.0 if graph else -100.0
            ] * (verifier.contract.GENERATED_TOKENS - 1)

    reconstructed = verifier._classify(results)

    assert reconstructed["classification"] == (
        verifier.NO_GO_EVIDENCE_INCOMPLETE
    )
    assert reconstructed["incomplete_evidence"]


def test_independent_verifier_fails_closed_on_zero_baseline_latency(
    tmp_path: Path,
):
    _contract, _gate, run_dir = _write_complete_fixture(tmp_path)
    _contract, _worker, _gate, verifier = _modules()
    results = verifier._load_results(run_dir)
    for result in results:
        if result["case"]["arm"] == "eager":
            for row in result["rows"]:
                row["ttft_ns"] = 0

    reconstructed = verifier._classify(results)

    assert reconstructed["classification"] == (
        verifier.NO_GO_EVIDENCE_INCOMPLETE
    )
    assert reconstructed["incomplete_evidence"]
