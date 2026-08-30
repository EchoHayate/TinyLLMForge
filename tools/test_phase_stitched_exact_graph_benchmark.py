#!/usr/bin/env python3
"""Frozen contracts for the phase-stitched exact-graph benchmark."""

from __future__ import annotations

from copy import deepcopy
import hashlib
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
            "tools.phase_stitched_exact_graph_contract",
            "tools.phase_stitched_exact_graph_worker",
            "tools.phase_stitched_exact_graph_gate",
            "tools.phase_stitched_exact_graph_verify",
        )
    )


class StepClock:

    def __init__(self, step_ns: int = 1_000_000):
        self.value = 0
        self.step_ns = step_ns

    def __call__(self) -> int:
        self.value += self.step_ns
        return self.value


class FakeRunner:

    def __init__(self, arm: str):
        self.arm = arm
        self.config = SimpleNamespace(
            hf_config=SimpleNamespace(torch_dtype="bfloat16")
        )
        self.request_index = 0
        self.peak_reset_calls = 0
        self.phase = {
            "attempts": 0,
            "successes": 0,
            "prefill_graph_replays": 0,
            "decode_graph_replays": 0,
            "target_model_forwards": 0,
            "failures": 0,
            "fallback_counts": {},
            "quarantined_joint_identities": [],
        }
        self.prefill = {
            "capture_attempts": 2 if arm != "eager" else 0,
            "capture_successes": 2 if arm != "eager" else 0,
            "capture_failures": 0,
            "replays": 0,
            "replay_failures": 0,
            "quarantines": 0,
            "fallbacks": 0,
            "static_bytes": 1_000_000 if arm != "eager" else 0,
            "allocated_delta_bytes": 0,
            "reserved_delta_bytes": (
                40_000_000 if arm != "eager" else 0
            ),
            "total_capture_ns": (
                700_000_000 if arm != "eager" else 0
            ),
        }
        self.burst = {
            "attempts": 0,
            "acceptances": 0,
            "graph_replays": 0,
            "commits": 0,
            "failures": 0,
            "fallback_counts": {},
            "quarantine_reason": None,
            "pending_leases": 0,
        }

    def reset_peak_memory_stats(self):
        self.peak_reset_calls += 1
        return self.memory_snapshot()

    def memory_snapshot(self):
        reserved = (
            2_120_000_000
            if self.arm == "stitched_composition"
            else 2_100_000_000
        )
        return {
            "cuda_allocated_bytes": 2_000_000_000,
            "cuda_reserved_bytes": reserved,
            "cuda_peak_allocated_bytes": 2_010_000_000,
            "cuda_peak_reserved_bytes": reserved,
            "kv_capacity_bytes": 128_000_000,
        }

    def phase_stitch_summary(self):
        return deepcopy(self.phase)

    def exact_greedy_decode_burst_summary(self):
        return deepcopy(self.burst)

    @property
    def exact_prefill_cuda_graph_cache(self):
        return SimpleNamespace(summary=lambda: deepcopy(self.prefill))


class FakeEngine:

    def __init__(self, spec: dict):
        self.spec = spec
        self.arm = spec["arm"]
        self.model_runner = FakeRunner(self.arm)
        self.pending = None
        self.generated = []
        self.last_step_observation = None
        self.exited = False
        self.clear_calls = 0
        self.tokenizer = SimpleNamespace(
            decode=lambda values: " ".join(map(str, values))
        )

    def add_request(self, prompt, sampling_params):
        self.pending = {
            "prompt": list(prompt),
            "max_tokens": sampling_params.max_tokens,
        }
        self.generated = []
        self.model_runner.request_index += 1

    def is_finished(self):
        return (
            self.pending is None
            or len(self.generated) >= self.pending["max_tokens"]
        )

    def step(self, *, completion_only):
        assert completion_only is True
        remaining = self.pending["max_tokens"] - len(self.generated)
        width = 1
        if self.arm == "stitched_composition":
            if not self.generated:
                width = 1
            elif len(self.generated) == 1:
                width = min(7, remaining)
            else:
                width = min(8, remaining)
        elif self.arm == "independent_composition":
            width = 1 if not self.generated else min(8, remaining)
        width = min(width, remaining)
        start = len(self.generated)
        emitted = [1000 + index for index in range(start, start + width)]
        self.generated.extend(emitted)
        if start == 0 and self.arm != "eager":
            self.model_runner.prefill["replays"] += 1
        if self.arm == "stitched_composition" and start == 0:
            phase = self.model_runner.phase
            phase["attempts"] += 1
            phase["successes"] += 1
            phase["prefill_graph_replays"] += 1
            phase["decode_graph_replays"] += 7
            phase["target_model_forwards"] += 8
        if (
            self.arm == "independent_composition"
            and start > 0
        ) or (
            self.arm == "stitched_composition"
            and start >= 8
        ):
            burst = self.model_runner.burst
            burst["attempts"] += 1
            burst["acceptances"] += 1
            burst["graph_replays"] += width
            burst["commits"] += 1
        self.last_step_observation = {
            "new_completion_tokens_by_seq": {1: emitted},
            "phase_stitch_attempted": (
                self.arm == "stitched_composition" and start == 0
            ),
            "phase_stitch_accepted": (
                self.arm == "stitched_composition" and start == 0
            ),
            "phase_published": (
                "prefix"
                if self.arm == "stitched_composition" and start == 0
                else (
                    "suffix"
                    if self.arm == "stitched_composition" and start == 1
                    else None
                )
            ),
            "prefix_d2h_calls": (
                1
                if self.arm == "stitched_composition" and start == 0
                else 0
            ),
            "suffix_d2h_calls": (
                1
                if self.arm == "stitched_composition" and start == 1
                else 0
            ),
            "prefix_d2h_bytes": (
                8
                if self.arm == "stitched_composition" and start == 0
                else 0
            ),
            "suffix_d2h_bytes": (
                56
                if self.arm == "stitched_composition" and start == 1
                else 0
            ),
            "pending_suffix": (
                self.arm == "stitched_composition" and start == 0
            ),
        }
        outputs = (
            [(1, list(self.generated))]
            if self.is_finished()
            else []
        )
        return outputs, -width

    def clear_reusable_prefix_cache(self):
        self.clear_calls += 1
        return 1

    def exit(self):
        self.exited = True
        return {"rank": 0, "process_group_destroyed": True}


def _row(contract, case, sample_index, *, metrics):
    arm = case["arm"]
    stitched = arm == "stitched_composition"
    independent = arm == "independent_composition"
    graph = arm != "eager"
    return {
        "schema_version": contract.ROW_SCHEMA_VERSION,
        "case_id": case["case_id"],
        "round": case["round"],
        "order_position": case["order_position"],
        "arm": arm,
        "prompt_tokens": case["prompt_tokens"],
        "sample_index": sample_index,
        "generated_tokens": contract.GENERATED_TOKENS,
        "prompt_sha256": f"{sample_index + 1:064x}",
        "output_token_ids": list(range(contract.GENERATED_TOKENS)),
        "output_text_sha256": "c" * 64,
        "ttft_ns": metrics["ttft_ns"],
        "token_0_to_1_gap_ns": metrics["gap_ns"],
        "tpot_samples_ns": [
            metrics["tpot_ns"]
        ] * (contract.GENERATED_TOKENS - 1),
        "tpot_median_ns": metrics["tpot_ns"],
        "e2e_ns": metrics["e2e_ns"],
        "output_tokens_per_second": (
            contract.GENERATED_TOKENS
            / (metrics["e2e_ns"] / 1_000_000_000)
        ),
        "cuda_peak_allocated_bytes": 2_010_000_000,
        "cuda_peak_reserved_bytes": metrics["reserved_bytes"],
        "prefill_graph_replay_delta": 1 if graph else 0,
        "exact_burst_replay_delta": (
            127
            if independent
            else (120 if stitched else 0)
        ),
        "phase_stitch_attempt_delta": 1 if stitched else 0,
        "phase_stitch_success_delta": 1 if stitched else 0,
        "phase_stitch_prefill_replay_delta": 1 if stitched else 0,
        "phase_stitch_decode_replay_delta": 7 if stitched else 0,
        "phase_stitch_target_forward_delta": 8 if stitched else 0,
        "phase_stitch_failure_delta": 0,
        "phase_stitch_quarantine_delta": 0,
        "phase_stitch_prefix_d2h_calls": 1 if stitched else 0,
        "phase_stitch_suffix_d2h_calls": 1 if stitched else 0,
        "phase_stitch_prefix_d2h_bytes": 8 if stitched else 0,
        "phase_stitch_suffix_d2h_bytes": 56 if stitched else 0,
        "phase_stitch_prefix_commits": 1 if stitched else 0,
        "phase_stitch_suffix_commits": 1 if stitched else 0,
        "phase_stitch_pending_leases": 0,
        "phase_stitch_fallback_count": 0,
        "preauthorized_kv_tokens": 7 if stitched else 0,
    }


def _write_complete_fixture(root: Path):
    contract, _worker, gate, _verifier = _modules()
    run_dir = root / "run"
    cases_dir = run_dir / "cases"
    cases_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({
            "schema_version": contract.RUN_SCHEMA_VERSION,
            "run_tag": "synthetic-phase-stitched-r1",
            "source_base_commit": "1" * 40,
            "source_files": {
                relative: hashlib.sha256(
                    (ROOT / relative).read_bytes()
                ).hexdigest()
                for relative in contract.SOURCE_FILES
            },
            "model": "/models/Qwen3-0.6B",
            "precision": "bfloat16",
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
    metrics_by_arm = {
        "eager": {
            "ttft_ns": 100_000_000,
            "gap_ns": 10_000_000,
            "tpot_ns": 2_200_000,
            "e2e_ns": 400_000_000,
            "reserved_bytes": 2_000_000_000,
        },
        "prefill_only": {
            "ttft_ns": 85_000_000,
            "gap_ns": 10_000_000,
            "tpot_ns": 2_200_000,
            "e2e_ns": 385_000_000,
            "reserved_bytes": 2_010_000_000,
        },
        "independent_composition": {
            "ttft_ns": 86_000_000,
            "gap_ns": 8_000_000,
            "tpot_ns": 2_000_000,
            "e2e_ns": 360_000_000,
            "reserved_bytes": 2_020_000_000,
        },
        "stitched_composition": {
            "ttft_ns": 86_500_000,
            "gap_ns": 6_800_000,
            "tpot_ns": 1_900_000,
            "e2e_ns": 345_000_000,
            "reserved_bytes": 2_060_000_000,
        },
    }
    for case in contract.build_case_matrix():
        rows = [
            _row(
                contract,
                case,
                sample_index,
                metrics=metrics_by_arm[case["arm"]],
            )
            for sample_index in range(contract.MEASURED_REPETITIONS)
        ]
        graph = case["arm"] != "eager"
        result = {
            "schema_version": contract.RESULT_SCHEMA_VERSION,
            "case": deepcopy(case),
            "model": "/models/Qwen3-0.6B",
            "model_dtype": "bfloat16",
            "rows": rows,
            "prefill_graph_summary": {
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
                "static_bytes": 1_000_000 if graph else 0,
                "allocated_delta_bytes": 0,
                "reserved_delta_bytes": (
                    40_000_000 if graph else 0
                ),
                "total_capture_ns": (
                    700_000_000 if graph else 0
                ),
            },
            "exact_burst_summary": {
                "attempts": 0,
                "acceptances": 0,
                "replays": 0,
                "commits": 0,
                "failures": 0,
                "quarantines": 0,
                "fallback_count": 0,
                "fallback_counts": {},
                "quarantine_reason": None,
                "pending_leases": 0,
            },
            "phase_stitch_summary": {
                "attempts": 0,
                "successes": 0,
                "prefill_graph_replays": 0,
                "decode_graph_replays": 0,
                "target_model_forwards": 0,
                "failures": 0,
                "quarantines": 0,
                "pending_leases": 0,
                "fallback_count": 0,
                "fallback_counts": {},
                "quarantined_joint_identities": [],
            },
        }
        destination = cases_dir / case["case_id"]
        destination.mkdir()
        (destination / "result.json").write_text(
            json.dumps(result, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return contract, gate, run_dir


def test_contract_freezes_four_arm_balanced_matrix():
    contract, _worker, _gate, _verifier = _modules()
    cases = contract.build_case_matrix()

    assert contract.ARMS == (
        "eager",
        "prefill_only",
        "independent_composition",
        "stitched_composition",
    )
    assert contract.PROMPT_TOKEN_COUNTS == (256, 2048)
    assert contract.ROUNDS == 2
    assert contract.WARMUP_REPETITIONS == 2
    assert contract.MEASURED_REPETITIONS == 5
    assert contract.GENERATED_TOKENS == 128
    assert {
        "tools/run_staged_inference_benchmark_remote.py",
        "tools/run_zero_temperature_greedy_fast_path_remote.py",
    } <= set(contract.SOURCE_FILES)
    assert len(cases) == 16
    assert [row["arm"] for row in cases[:4]] == list(contract.ARMS)
    assert [row["arm"] for row in cases[8:12]] == list(
        reversed(contract.ARMS)
    )
    for case in cases:
        config = case["engine_config"]
        assert config["tensor_parallel_size"] == 1
        assert config["max_num_seqs"] == 1
        assert case["sampling"] == {
            "temperature": 0.0,
            "max_tokens": 128,
            "ignore_eos": True,
        }
        assert case["precision"] == "bfloat16"
        assert case["completion_only"] is True
        assert config["prefill_cuda_graphs"] is (
            case["arm"] != "eager"
        )
        assert config["exact_greedy_decode_burst"] is (
            case["arm"] in (
                "independent_composition",
                "stitched_composition",
            )
        )
        assert config["phase_stitched_exact_graph_runtime"] is (
            case["arm"] == "stitched_composition"
        )


def test_worker_records_complete_stitched_accounting(tmp_path: Path):
    contract, worker, _gate, _verifier = _modules()
    case = next(
        row
        for row in contract.build_case_matrix()
        if row["arm"] == "stitched_composition"
        and row["prompt_tokens"] == 256
    )
    engines = []

    def engine_factory(spec):
        engine = FakeEngine(spec)
        engines.append(engine)
        return engine

    result = worker.run_worker(
        case,
        model="/models/Qwen3-0.6B",
        output_dir=tmp_path / "case",
        engine_factory=engine_factory,
        clock_ns=StepClock(),
        synchronize=lambda: None,
    )

    assert len(engines) == 1
    assert engines[0].exited is True
    assert len(result["rows"]) == contract.MEASURED_REPETITIONS
    assert result["model_dtype"] == "bfloat16"
    for row in result["rows"]:
        assert len(row["output_token_ids"]) == 128
        assert len(row["tpot_samples_ns"]) == 127
        assert row["token_0_to_1_gap_ns"] > 0
        assert row["phase_stitch_attempt_delta"] == 1
        assert row["phase_stitch_success_delta"] == 1
        assert row["phase_stitch_decode_replay_delta"] == 7
        assert row["phase_stitch_target_forward_delta"] == 8
        assert row["exact_burst_replay_delta"] == 120
        assert row["phase_stitch_prefix_d2h_calls"] == 1
        assert row["phase_stitch_suffix_d2h_calls"] == 1
        assert row["phase_stitch_prefix_d2h_bytes"] == 8
        assert row["phase_stitch_suffix_d2h_bytes"] == 56
        assert row["phase_stitch_prefix_commits"] == 1
        assert row["phase_stitch_suffix_commits"] == 1
        assert row["phase_stitch_pending_leases"] == 0
        assert row["preauthorized_kv_tokens"] == 7
    assert engines[0].model_runner.peak_reset_calls == (
        contract.MEASURED_REPETITIONS
    )
    assert (tmp_path / "case" / "result.json").is_file()


def test_gate_and_independent_verifier_reconstruct_go(tmp_path: Path):
    contract, gate, run_dir = _write_complete_fixture(tmp_path)
    produced = gate.produce_gate(run_dir)
    _contract, _worker, _gate, verifier = _modules()
    verified = verifier.verify_artifact_directory(run_dir)

    assert produced["classification"] == (
        gate.GO_PHASE_STITCHED_EXACT_GRAPH
    )
    assert verified["verified"] is True
    assert verified["classification"] == produced["classification"]
    assert verified["raw_metrics_reconstructed"] is True
    assert produced["correctness"]["all_token_ids_exact"] is True
    assert produced["mechanism"]["complete_accounting"] is True
    assert (
        produced["performance"]["aggregate"][
            "d_vs_c_e2e_improvement_fraction"
        ]
        >= contract.E2E_AGGREGATE_IMPROVEMENT_MINIMUM
    )
    assert {
        "summary.json",
        "gate.json",
        "manifest.json",
        "producer_receipt.json",
        "independent_verifier_receipt.json",
    } <= {path.name for path in run_dir.iterdir()}
    assert "phase_stitched_exact_graph_gate" not in (
        ROOT / "tools" / "phase_stitched_exact_graph_verify.py"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("mutation", "expected_fragment"),
    (
        ("token", "CORRECTNESS"),
        ("missing_metric", "EVIDENCE"),
        ("failure", "MECHANISM"),
        ("quarantine", "MECHANISM"),
        ("shape_gain", "PERFORMANCE"),
        ("aggregate_gain", "PERFORMANCE"),
        ("gap", "PERFORMANCE"),
        ("ttft", "PERFORMANCE"),
        ("tail", "PERFORMANCE"),
        ("memory", "PERFORMANCE"),
        ("config", "EVIDENCE"),
        ("graph", "MECHANISM"),
        ("burst_failure", "MECHANISM"),
        ("phase_fallback", "MECHANISM"),
        ("capture_metric", "EVIDENCE"),
    ),
)
def test_gate_fails_closed_for_each_frozen_requirement(
    tmp_path: Path,
    mutation: str,
    expected_fragment: str,
):
    contract, gate, run_dir = _write_complete_fixture(tmp_path)
    paths = sorted((run_dir / "cases").glob("*/result.json"))
    stitched_paths = [
        path
        for path in paths
        if path.parent.name.endswith("stitched-composition")
    ]
    independent_paths = [
        path
        for path in paths
        if path.parent.name.endswith("independent-composition")
    ]
    targets = stitched_paths
    if mutation == "token":
        payload = json.loads(targets[0].read_text())
        payload["rows"][0]["output_token_ids"][-1] = 9999
        targets[0].write_text(json.dumps(payload) + "\n")
    elif mutation == "missing_metric":
        payload = json.loads(targets[0].read_text())
        payload["rows"][0].pop("token_0_to_1_gap_ns")
        targets[0].write_text(json.dumps(payload) + "\n")
    elif mutation in ("failure", "quarantine"):
        field = (
            "phase_stitch_failure_delta"
            if mutation == "failure"
            else "phase_stitch_quarantine_delta"
        )
        payload = json.loads(targets[0].read_text())
        payload["rows"][0][field] = 1
        targets[0].write_text(json.dumps(payload) + "\n")
    elif mutation == "graph":
        payload = json.loads(targets[0].read_text())
        payload["rows"][0]["phase_stitch_decode_replay_delta"] = 0
        targets[0].write_text(json.dumps(payload) + "\n")
    elif mutation == "config":
        payload = json.loads(targets[0].read_text())
        payload["case"]["engine_config"]["max_num_seqs"] = 2
        targets[0].write_text(json.dumps(payload) + "\n")
    elif mutation == "burst_failure":
        payload = json.loads(targets[0].read_text())
        payload["exact_burst_summary"]["failures"] = 1
        targets[0].write_text(json.dumps(payload) + "\n")
    elif mutation == "phase_fallback":
        payload = json.loads(targets[0].read_text())
        payload["phase_stitch_summary"]["fallback_counts"] = {
            "unexpected": 1,
        }
        payload["phase_stitch_summary"]["fallback_count"] = 1
        targets[0].write_text(json.dumps(payload) + "\n")
    elif mutation == "capture_metric":
        payload = json.loads(targets[0].read_text())
        payload["prefill_graph_summary"].pop("total_capture_ns")
        targets[0].write_text(json.dumps(payload) + "\n")
    else:
        for path in targets:
            payload = json.loads(path.read_text())
            for row in payload["rows"]:
                if mutation == "shape_gain":
                    row["e2e_ns"] = 357_000_000
                elif mutation == "aggregate_gain":
                    row["e2e_ns"] = 354_000_000
                elif mutation == "gap":
                    row["token_0_to_1_gap_ns"] = 7_500_000
                elif mutation == "ttft":
                    row["ttft_ns"] = 88_000_000
                elif mutation == "tail":
                    row["e2e_ns"] = 370_000_000
                elif mutation == "memory":
                    row["cuda_peak_reserved_bytes"] = 2_090_000_000
            path.write_text(json.dumps(payload) + "\n")
        if mutation == "shape_gain":
            for path in independent_paths:
                payload = json.loads(path.read_text())
                for row in payload["rows"]:
                    row["e2e_ns"] = 360_000_000
                path.write_text(json.dumps(payload) + "\n")
    result = gate.produce_gate(run_dir)
    assert expected_fragment in result["classification"]


def test_independent_verifier_rejects_tamper(tmp_path: Path):
    _contract, gate, run_dir = _write_complete_fixture(tmp_path)
    gate.produce_gate(run_dir)
    _contract, _worker, _gate, verifier = _modules()
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["classification"] = "NO_GO_PERFORMANCE"
    summary_path.write_text(json.dumps(summary) + "\n")

    with pytest.raises(ValueError, match="manifest hash"):
        verifier.verify_artifact_directory(run_dir)


def test_independent_verifier_rejects_invalid_source_hash_inventory(
    tmp_path: Path,
):
    _contract, _gate, run_dir = _write_complete_fixture(tmp_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    first_source = next(iter(manifest["source_files"]))
    manifest["source_files"][first_source] = "not-a-sha256"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    _contract, _worker, _gate, verifier = _modules()

    with pytest.raises(ValueError, match="source hash inventory"):
        verifier._reconstruct(run_dir)


def test_independent_verifier_reconstructs_source_hashes(
    tmp_path: Path,
):
    _contract, _gate, run_dir = _write_complete_fixture(tmp_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    first_source = next(iter(manifest["source_files"]))
    manifest["source_files"][first_source] = "0" * 64
    manifest_path.write_text(json.dumps(manifest) + "\n")
    _contract, _worker, _gate, verifier = _modules()

    with pytest.raises(ValueError, match="source hash mismatch"):
        verifier._reconstruct(run_dir)


def test_producer_fails_closed_on_source_hash_mismatch(tmp_path: Path):
    _contract, gate, run_dir = _write_complete_fixture(tmp_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    first_source = next(iter(manifest["source_files"]))
    manifest["source_files"][first_source] = "0" * 64
    manifest_path.write_text(json.dumps(manifest) + "\n")

    produced = gate.produce_gate(run_dir)

    assert produced["classification"] == "NO_GO_EVIDENCE"
    assert "source hash mismatch" in produced["evidence_errors"][0]


def test_independent_verifier_rejects_raw_row_identity_drift(
    tmp_path: Path,
):
    _contract, _gate, run_dir = _write_complete_fixture(tmp_path)
    result_path = sorted((run_dir / "cases").glob("*/result.json"))[0]
    result = json.loads(result_path.read_text())
    result["rows"][0]["arm"] = "stitched_composition"
    result_path.write_text(json.dumps(result) + "\n")
    _contract, _worker, _gate, verifier = _modules()

    with pytest.raises(ValueError, match="raw row identity"):
        verifier._reconstruct(run_dir)


def test_independent_verifier_reconstructs_mechanism_no_go(
    tmp_path: Path,
):
    _contract, gate, run_dir = _write_complete_fixture(tmp_path)
    target = next(
        path
        for path in sorted((run_dir / "cases").glob("*/result.json"))
        if path.parent.name.endswith("stitched-composition")
    )
    payload = json.loads(target.read_text())
    payload["exact_burst_summary"]["failures"] = 1
    target.write_text(json.dumps(payload) + "\n")

    produced = gate.produce_gate(run_dir)
    _contract, _worker, _gate, verifier = _modules()
    verified = verifier.verify_artifact_directory(run_dir)

    assert produced["classification"] == "NO_GO_MECHANISM"
    assert verified["verified"] is True
    assert verified["classification"] == "NO_GO_MECHANISM"
