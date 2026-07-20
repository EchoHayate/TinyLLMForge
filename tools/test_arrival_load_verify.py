"""Dependency-light tests for the independent arrival-load verifier."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tarfile
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_PATH = REPO_ROOT / "tools" / "arrival_load_verify.py"
GATE_PATH = REPO_ROOT / "tools" / "arrival_load_gate.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "arrival_load_verify",
        VERIFY_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load arrival_load_verify")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = _load_verifier()


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "arrival_load_gate_for_verify_test",
        GATE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load arrival_load_gate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate()

ADAPTIVE_DEFAULTS = {
    "chunked_prefill_decode_first": True,
    "chunked_prefill_max_consecutive_chunks": 0,
    "chunked_prefill_mixed_batch": False,
    "chunked_prefill_mixed_min_prompt_tokens": 0,
    "chunked_prefill_adaptive_mixed": False,
    "chunked_prefill_adaptive_enter_waiting": 8,
    "chunked_prefill_adaptive_exit_waiting": 2,
    "chunked_prefill_adaptive_transition_steps": 2,
    "chunked_prefill_adaptive_max_mixed_steps": 2,
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(_canonical_bytes(row).decode("utf-8") + "\n")


def _policy_id(config: dict) -> str:
    return hashlib.sha256(_canonical_bytes(config)).hexdigest()


def _case_id(policy: str, repetition: int) -> str:
    return f"{policy}-steady_moderate-r{repetition}"


def _complete_artifact() -> tuple[tempfile.TemporaryDirectory, Path]:
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    source_root = root / "snapshot-source"
    source_root.mkdir()
    (source_root / "marker.txt").write_text("arrival-load-source\n")
    source_files = [{
        "path": "marker.txt",
        "size_bytes": (source_root / "marker.txt").stat().st_size,
        "sha256": _sha256_file(source_root / "marker.txt"),
    }]
    source_tree_sha256 = hashlib.sha256(
        _canonical_bytes(source_files)
    ).hexdigest()
    source_evidence = {
        "schema_version": 1,
        "base_commit": "1" * 40,
        "tree_sha256": source_tree_sha256,
        "files": source_files,
        "patch_size_bytes": 0,
        "patch_sha256": hashlib.sha256(b"").hexdigest(),
    }
    _write_json(root / "source_evidence.json", source_evidence)
    (root / "source.patch").write_bytes(b"")
    with tarfile.open(root / "source_snapshot.tar.gz", "w:gz") as archive:
        archive.add(source_root, arcname="source")

    resolved_policy_config_by_name = {
        "P0": dict(ADAPTIVE_DEFAULTS),
        "P3": {
            **ADAPTIVE_DEFAULTS,
            "chunked_prefill_decode_first": False,
            "chunked_prefill_mixed_batch": True,
        },
        "P4": {
            **ADAPTIVE_DEFAULTS,
            "chunked_prefill_adaptive_mixed": True,
        },
    }
    policy_identity_by_name = {
        name: _policy_id(config)
        for name, config in resolved_policy_config_by_name.items()
    }
    manifest = {
        "schema_version": 1,
        "run_tag": "synthetic-arrival-load",
        "source_tree_sha256": source_tree_sha256,
        "required_scenarios": ["steady_moderate"],
        "measured_repetitions": 3,
        "policy_identity_by_name": policy_identity_by_name,
        "canonical_policy_by_name": {
            "P0": "P0",
            "P3": "P3",
            "P4": "P4",
        },
        "resolved_policy_config_by_name": (
            resolved_policy_config_by_name
        ),
        "process_port_pairs": [],
        "expected_case_ids": [],
    }
    calibration_manifest = [{
        "calibration_id": "p0-rate-00",
        "policy": "P0",
        "requested_rate_rps": 1.0,
    }]
    calibration_rows = [{
        "calibration_id": "p0-rate-00",
        "status": "PASS",
        "stable": True,
        "completed_request_throughput_rps": 1.0,
    }]
    workload_rows = [{
        "schema_version": 1,
        "request_id": "steady_moderate-measured-0000",
        "scenario": "steady_moderate",
        "warmup": False,
        "arrival_offset_ns": 0,
        "prompt_token_count": 4,
        "requested_output_tokens": 2,
        "prompt_class": "short",
        "output_class": "short",
        "service_time_bucket": "short__short",
    }]
    timeline_rows = []
    scheduler_rows = []
    memory_rows = []
    case_rows = []
    ports = 31000
    for repetition in range(3):
        for policy in ("P0", "P3", "P4"):
            case_id = _case_id(policy, repetition)
            manifest["expected_case_ids"].append(case_id)
            manifest["process_port_pairs"].append({
                "case_id": case_id,
                "tinyvllm_dist_port": ports,
                "master_port": ports + 1,
            })
            ports += 2
            duration_ns = 1_000_000_000
            seq_id = repetition * 10 + {"P0": 0, "P3": 1, "P4": 2}[policy]
            timeline_rows.append({
                "case_id": case_id,
                "policy": policy,
                "scenario": "steady_moderate",
                "repetition": repetition,
                "request_id": workload_rows[0]["request_id"],
                "seq_id": seq_id,
                "scheduled_arrival_ns": 100,
                "actual_arrival_ns": 110,
                "first_scheduled_ns": 120,
                "first_token_ns": 200,
                "token_timestamps_ns": [200, duration_ns + 100],
                "completion_ns": duration_ns + 100,
                "output_token_ids": [11, 12],
                "finish_reason": "length",
                "error": None,
            })
            if policy == "P4":
                waiting_high = list(range(100, 108))
                waiting_low = [100, 101]
                running_ids = [seq_id]
                prefilling_ids = [200]
                scheduler_rows.extend([
                    {
                        "case_id": case_id,
                        "step_index": 0,
                        "policy_branch": "adaptive_mixed_decode_first",
                        "scheduled": [{"seq_id": seq_id, "is_decode": True}],
                        "queue_before": {
                            "adaptive_mixed_state": "inactive",
                            "adaptive_high_streak": 0,
                            "adaptive_low_streak": 0,
                            "adaptive_consecutive_mixed_steps": 0,
                            "waiting_seq_ids": waiting_high,
                            "prefilling_seq_ids": [],
                            "running_seq_ids": running_ids,
                        },
                        "queue_after": {
                            "adaptive_mixed_state": "inactive",
                            "adaptive_high_streak": 1,
                            "adaptive_low_streak": 0,
                            "adaptive_consecutive_mixed_steps": 0,
                            "waiting_seq_ids": waiting_high,
                            "prefilling_seq_ids": [],
                            "running_seq_ids": running_ids,
                        },
                    },
                    {
                        "case_id": case_id,
                        "step_index": 1,
                        "policy_branch": "adaptive_mixed_prefill_decode",
                        "scheduled": [
                            {"seq_id": 100, "is_decode": False},
                            {"seq_id": seq_id, "is_decode": True},
                        ],
                        "queue_before": {
                            "adaptive_mixed_state": "inactive",
                            "adaptive_high_streak": 1,
                            "adaptive_low_streak": 0,
                            "adaptive_consecutive_mixed_steps": 0,
                            "waiting_seq_ids": waiting_high,
                            "prefilling_seq_ids": [],
                            "running_seq_ids": running_ids,
                        },
                        "queue_after": {
                            "adaptive_mixed_state": "active",
                            "adaptive_high_streak": 0,
                            "adaptive_low_streak": 0,
                            "adaptive_consecutive_mixed_steps": 1,
                            "waiting_seq_ids": waiting_high[1:],
                            "prefilling_seq_ids": [100],
                            "running_seq_ids": running_ids,
                        },
                    },
                    {
                        "case_id": case_id,
                        "step_index": 2,
                        "policy_branch": "adaptive_mixed_prefill_decode",
                        "scheduled": [
                            {"seq_id": 100, "is_decode": False},
                            {"seq_id": seq_id, "is_decode": True},
                        ],
                        "queue_before": {
                            "adaptive_mixed_state": "active",
                            "adaptive_high_streak": 0,
                            "adaptive_low_streak": 0,
                            "adaptive_consecutive_mixed_steps": 1,
                            "waiting_seq_ids": waiting_high[1:],
                            "prefilling_seq_ids": [100],
                            "running_seq_ids": running_ids,
                        },
                        "queue_after": {
                            "adaptive_mixed_state": "active",
                            "adaptive_high_streak": 0,
                            "adaptive_low_streak": 0,
                            "adaptive_consecutive_mixed_steps": 2,
                            "waiting_seq_ids": waiting_high[1:],
                            "prefilling_seq_ids": [100],
                            "running_seq_ids": running_ids,
                        },
                    },
                    {
                        "case_id": case_id,
                        "step_index": 3,
                        "policy_branch": "adaptive_mixed_decode_yield",
                        "scheduled": [{"seq_id": seq_id, "is_decode": True}],
                        "queue_before": {
                            "adaptive_mixed_state": "active",
                            "adaptive_high_streak": 0,
                            "adaptive_low_streak": 0,
                            "adaptive_consecutive_mixed_steps": 2,
                            "waiting_seq_ids": waiting_low,
                            "prefilling_seq_ids": prefilling_ids,
                            "running_seq_ids": running_ids,
                        },
                        "queue_after": {
                            "adaptive_mixed_state": "active",
                            "adaptive_high_streak": 0,
                            "adaptive_low_streak": 1,
                            "adaptive_consecutive_mixed_steps": 0,
                            "waiting_seq_ids": waiting_low,
                            "prefilling_seq_ids": prefilling_ids,
                            "running_seq_ids": running_ids,
                        },
                    },
                    {
                        "case_id": case_id,
                        "step_index": 4,
                        "policy_branch": "adaptive_mixed_prefill_decode",
                        "scheduled": [
                            {"seq_id": 200, "is_decode": False},
                            {"seq_id": seq_id, "is_decode": True},
                        ],
                        "queue_before": {
                            "adaptive_mixed_state": "active",
                            "adaptive_high_streak": 0,
                            "adaptive_low_streak": 1,
                            "adaptive_consecutive_mixed_steps": 0,
                            "waiting_seq_ids": waiting_low,
                            "prefilling_seq_ids": prefilling_ids,
                            "running_seq_ids": running_ids,
                        },
                        "queue_after": {
                            "adaptive_mixed_state": "draining",
                            "adaptive_high_streak": 0,
                            "adaptive_low_streak": 0,
                            "adaptive_consecutive_mixed_steps": 1,
                            "waiting_seq_ids": waiting_low,
                            "prefilling_seq_ids": [],
                            "running_seq_ids": running_ids,
                        },
                    },
                    {
                        "case_id": case_id,
                        "step_index": 5,
                        "policy_branch": "adaptive_mixed_decode_first",
                        "scheduled": [{"seq_id": seq_id, "is_decode": True}],
                        "queue_before": {
                            "adaptive_mixed_state": "draining",
                            "adaptive_high_streak": 0,
                            "adaptive_low_streak": 0,
                            "adaptive_consecutive_mixed_steps": 1,
                            "waiting_seq_ids": waiting_low,
                            "prefilling_seq_ids": [],
                            "running_seq_ids": running_ids,
                        },
                        "queue_after": {
                            "adaptive_mixed_state": "inactive",
                            "adaptive_high_streak": 0,
                            "adaptive_low_streak": 0,
                            "adaptive_consecutive_mixed_steps": 0,
                            "waiting_seq_ids": waiting_low,
                            "prefilling_seq_ids": [],
                            "running_seq_ids": running_ids,
                        },
                    },
                ])
            else:
                scheduler_rows.append({
                    "case_id": case_id,
                    "step_index": 0,
                    "step_start_ns": 120,
                    "step_end_ns": duration_ns + 100,
                    "scheduled_seq_ids": [seq_id],
                    "finished_seq_ids": [seq_id],
                })
            memory_rows.append({
                "case_id": case_id,
                "step_index": 0,
                "cuda_allocated_bytes": 100,
                "cuda_reserved_bytes": 200,
                "used_kv_blocks": 2,
                "kv_block_bytes": 64,
            })
            case_rows.append({
                "case_id": case_id,
                "policy": policy,
                "scenario": "steady_moderate",
                "repetition": repetition,
                "status": "PASS",
                "correctness": {
                    "exact_outputs": True,
                    "complete_requests": True,
                    "no_starvation": True,
                    "valid_lifecycle": True,
                    "stable_p0_outputs": True,
                },
                "metrics": {
                    "request_throughput_rps": 1.0,
                    "output_token_throughput_tps": 2.0,
                    "maximum_injection_lag_ns": 10.0,
                    "p50_injection_lag_ns": 10.0,
                    "p95_injection_lag_ns": 10.0,
                    "p99_injection_lag_ns": 10.0,
                    "p50_queue_delay_ns": 10.0,
                    "p95_queue_delay_ns": 10.0,
                    "p99_queue_delay_ns": 10.0,
                    "p50_ttft_ns": 100.0,
                    "p95_ttft_ns": 100.0,
                    "p50_itl_ns": 999_999_900.0,
                    "p95_itl_ns": 999_999_900.0,
                    "p50_e2e_ns": 1_000_000_000.0,
                    "p95_e2e_ns": 1_000_000_000.0,
                    "p99_ttft_ns": 100.0,
                    "p99_itl_ns": 999_999_900.0,
                    "p99_e2e_ns": 1_000_000_000.0,
                    "maximum_decode_gap_ns": 999_999_900.0,
                    "peak_cuda_allocated_bytes": 100,
                    "peak_cuda_reserved_bytes": 200,
                    "peak_used_kv_blocks": 2,
                    "peak_kv_bytes": 128,
                    "service_buckets": {
                        "short__short": {
                            "p50_e2e_ns": 1_000_000_000.0,
                            "p95_e2e_ns": 1_000_000_000.0,
                            "p99_e2e_ns": 1_000_000_000.0,
                            "completed_requests": 1,
                            "request_throughput_rps": 1.0,
                            "worst_e2e_ns": 1_000_000_000.0,
                        },
                    },
                    "jain_service_rate_index": 1.0,
                },
            })
    _write_json(root / "run_manifest.json", manifest)
    _write_jsonl(
        root / "calibration_manifest.jsonl",
        calibration_manifest,
    )
    _write_jsonl(root / "calibration_rows.jsonl", calibration_rows)
    _write_jsonl(root / "workload_manifest.jsonl", workload_rows)
    _write_jsonl(root / "request_timeline.jsonl", timeline_rows)
    _write_jsonl(root / "scheduler_trace.jsonl", scheduler_rows)
    _write_jsonl(root / "memory_trace.jsonl", memory_rows)
    _write_jsonl(root / "case_rows.jsonl", case_rows)
    summary = {
        "classification": "NO_GO",
        "structural_failures": [],
        "correctness_failures": [],
        "candidate_results": {
            "P4": {
                "policy": "P4",
                "classification": "NO_GO",
                "benefit_path": None,
                "median_ratios": {
                    "request_throughput_rps": 1.0,
                    "p95_ttft_ns": 1.0,
                    "p95_itl_ns": 1.0,
                    "p99_ttft_ns": 1.0,
                    "p99_itl_ns": 1.0,
                    "p99_e2e_ns": 1.0,
                    "maximum_decode_gap_ns": 1.0,
                    "peak_cuda_reserved_bytes": 1.0,
                    "peak_kv_bytes": 1.0,
                },
                "worst_repetition_ratios": {
                    "request_throughput_rps": 1.0,
                    "p95_ttft_ns": 1.0,
                    "p95_itl_ns": 1.0,
                    "p99_ttft_ns": 1.0,
                    "p99_itl_ns": 1.0,
                    "p99_e2e_ns": 1.0,
                    "maximum_decode_gap_ns": 1.0,
                    "peak_cuda_reserved_bytes": 1.0,
                    "peak_kv_bytes": 1.0,
                },
                "guard_failures": [],
            },
            "P3": {
                "policy": "P3",
                "classification": "NO_GO",
                "benefit_path": None,
                "median_ratios": {
                    "request_throughput_rps": 1.0,
                    "p95_ttft_ns": 1.0,
                    "p95_itl_ns": 1.0,
                    "p99_ttft_ns": 1.0,
                    "p99_itl_ns": 1.0,
                    "p99_e2e_ns": 1.0,
                    "maximum_decode_gap_ns": 1.0,
                    "peak_cuda_reserved_bytes": 1.0,
                    "peak_kv_bytes": 1.0,
                },
                "worst_repetition_ratios": {
                    "request_throughput_rps": 1.0,
                    "p95_ttft_ns": 1.0,
                    "p95_itl_ns": 1.0,
                    "p99_ttft_ns": 1.0,
                    "p99_itl_ns": 1.0,
                    "p99_e2e_ns": 1.0,
                    "maximum_decode_gap_ns": 1.0,
                    "peak_cuda_reserved_bytes": 1.0,
                    "peak_kv_bytes": 1.0,
                },
                "guard_failures": [],
            },
        },
    }
    _write_json(root / "summary.json", summary)
    (root / "report.md").write_text(
        "# Production Arrival-Load Gate\n\n"
        "Classification: `NO_GO`\n"
    )
    required = (
        "run_manifest.json",
        "calibration_manifest.jsonl",
        "calibration_rows.jsonl",
        "workload_manifest.jsonl",
        "request_timeline.jsonl",
        "scheduler_trace.jsonl",
        "memory_trace.jsonl",
        "case_rows.jsonl",
        "summary.json",
        "report.md",
        "source_evidence.json",
        "source.patch",
        "source_snapshot.tar.gz",
    )
    _write_json(
        root / "artifact_hashes.json",
        {name: _sha256_file(root / name) for name in required},
    )
    return temporary, root


def _refresh_hash(root: Path, name: str) -> None:
    hashes = json.loads((root / "artifact_hashes.json").read_text())
    hashes[name] = _sha256_file(root / name)
    _write_json(root / "artifact_hashes.json", hashes)


def _p5_trace(
    *,
    case_id: str,
    scenario: str,
    repetition: int,
    decode_seq_id: int,
    prefill_seq_id: int,
    intercept_ns: int,
    per_token_ns: int,
) -> list[dict]:
    ladder = [128, 112, 96, 80, 64, 48, 32, 16]
    waiting_high = list(range(prefill_seq_id, prefill_seq_id + 8))
    queue0 = {
        "waiting_seq_ids": waiting_high,
        "prefilling_seq_ids": [],
        "running_seq_ids": [],
    }
    queue1 = {
        "waiting_seq_ids": waiting_high,
        "prefilling_seq_ids": [],
        "running_seq_ids": [decode_seq_id],
    }
    queue2 = {
        "waiting_seq_ids": waiting_high,
        "prefilling_seq_ids": [],
        "running_seq_ids": [decode_seq_id],
    }
    queue3 = {
        "waiting_seq_ids": waiting_high[1:],
        "prefilling_seq_ids": [prefill_seq_id],
        "running_seq_ids": [decode_seq_id],
    }
    queue4 = {
        "waiting_seq_ids": [prefill_seq_id + 1],
        "prefilling_seq_ids": [prefill_seq_id],
        "running_seq_ids": [decode_seq_id],
    }
    queue5 = {
        "waiting_seq_ids": [decode_seq_id, prefill_seq_id + 1],
        "prefilling_seq_ids": [prefill_seq_id],
        "running_seq_ids": [],
    }
    queue6 = {
        "waiting_seq_ids": [prefill_seq_id + 1],
        "prefilling_seq_ids": [prefill_seq_id],
        "running_seq_ids": [decode_seq_id],
    }
    queue7 = {
        "waiting_seq_ids": [prefill_seq_id + 1],
        "prefilling_seq_ids": [prefill_seq_id],
        "running_seq_ids": [decode_seq_id],
    }
    queue8 = {
        "waiting_seq_ids": [prefill_seq_id + 1],
        "prefilling_seq_ids": [prefill_seq_id],
        "running_seq_ids": [],
    }
    common = {
        "case_id": case_id,
        "policy": "P5",
        "scenario": scenario,
        "repetition": repetition,
        "target_gap_ns": 64_000_000,
        "reserve_ns": 8_000_000,
        "cost_intercept_ns": intercept_ns,
        "cost_per_prefill_token_ns": per_token_ns,
        "candidate_chunk_tokens": ladder,
        "clock_invalid": False,
        "clock_invalid_reason": None,
        "finished_progress_entries_removed": [],
    }

    def row(
        step_index,
        *,
        decision_now_ns,
        step_end_ns,
        queue_before,
        queue_after,
        demand_before,
        demand_after,
        scheduled,
        progress_updates,
        token_deltas,
        oldest=None,
        progress=None,
        age=None,
        slack=None,
        selected=None,
        predicted=None,
        suppression=None,
        finished_removed=None,
    ):
        return {
            **common,
            "step_index": step_index,
            "decision_now_ns": decision_now_ns,
            "step_end_ns": step_end_ns,
            "actual_step_duration_ns":
                step_end_ns - decision_now_ns,
            "queue_before": queue_before,
            "queue_after": queue_after,
            "scheduled": scheduled,
            "new_completion_tokens_by_seq": token_deltas,
            "decode_progress_updates": progress_updates,
            "oldest_decode_seq_id": oldest,
            "oldest_decode_progress_ns": progress,
            "oldest_decode_age_ns": age,
            "remaining_slack_ns": slack,
            "predicted_step_ns": predicted,
            "selected_chunk_tokens": selected,
            "actual_prefill_tokens": sum(
                item["prefill_chunk_end"]
                - item["prefill_chunk_start"]
                for item in scheduled
                if item["is_decode"] is False
            ),
            "scheduled_decode_seq_ids": [
                item["seq_id"] for item in scheduled
                if item["is_decode"] is True
            ],
            "demand_state_before": demand_before,
            "demand_state_after": demand_after,
            "suppression_reason": suppression,
            "finished_progress_entries_removed":
                finished_removed or [],
        }

    progress0 = 1_001_000_000
    decision2 = 1_003_000_000
    slack2 = 56_000_000 - (decision2 - progress0)
    predicted128 = intercept_ns + 128 * per_token_ns
    predicted64 = intercept_ns + 64 * per_token_ns
    predicted16 = intercept_ns + 16 * per_token_ns
    decision3 = progress0 + 56_000_000 - predicted64
    step3_end = decision3 + 1_000_000
    slack3 = 56_000_000 - (decision3 - progress0)
    decision4 = step3_end + 56_000_000 - (predicted16 - 1)
    step4_end = decision4 + 1_000_000
    slack4 = 56_000_000 - (decision4 - step3_end)
    decision5 = step4_end + 100_000
    step5_end = decision5 + 100_000
    decision6 = step3_end + 57_000_000
    slack6 = 56_000_000 - (decision6 - step3_end)
    decision7 = decision6 + 2_000_000
    slack7 = 56_000_000 - (decision7 - step3_end)
    return [
        row(
            0,
            decision_now_ns=1_000_000_000,
            step_end_ns=progress0,
            queue_before=queue0,
            queue_after=queue1,
            demand_before="inactive",
            demand_after="inactive",
            scheduled=[{
                "seq_id": decode_seq_id,
                "is_decode": True,
            }],
            progress_updates={str(decode_seq_id): progress0},
            token_deltas={str(decode_seq_id): [11]},
        ),
        row(
            1,
            decision_now_ns=1_002_000_000,
            step_end_ns=decision2,
            queue_before=queue1,
            queue_after=queue2,
            demand_before="inactive",
            demand_after="inactive",
            scheduled=[{
                "seq_id": decode_seq_id,
                "is_decode": True,
            }],
            progress_updates={},
            token_deltas={str(decode_seq_id): []},
            oldest=decode_seq_id,
            progress=progress0,
            age=1_000_000,
            slack=55_000_000,
            suppression="inactive",
        ),
        row(
            2,
            decision_now_ns=decision2,
            step_end_ns=1_004_000_000,
            queue_before=queue2,
            queue_after=queue3,
            demand_before="inactive",
            demand_after="active",
            scheduled=[{
                "seq_id": decode_seq_id,
                "is_decode": True,
            }, {
                "seq_id": prefill_seq_id,
                "is_decode": False,
                "prefill_chunk_start": 0,
                "prefill_chunk_end": 128,
            }],
            progress_updates={},
            token_deltas={str(decode_seq_id): []},
            oldest=decode_seq_id,
            progress=progress0,
            age=decision2 - progress0,
            slack=slack2,
            selected=128,
            predicted=predicted128,
        ),
        row(
            3,
            decision_now_ns=decision3,
            step_end_ns=step3_end,
            queue_before=queue3,
            queue_after=queue4,
            demand_before="active",
            demand_after="active",
            scheduled=[{
                "seq_id": decode_seq_id,
                "is_decode": True,
            }, {
                "seq_id": prefill_seq_id,
                "is_decode": False,
                "prefill_chunk_start": 128,
                "prefill_chunk_end": 192,
            }],
            progress_updates={
                str(decode_seq_id): step3_end,
            },
            token_deltas={str(decode_seq_id): [12]},
            oldest=decode_seq_id,
            progress=progress0,
            age=decision3 - progress0,
            slack=slack3,
            selected=64,
            predicted=predicted64,
        ),
        row(
            4,
            decision_now_ns=decision4,
            step_end_ns=step4_end,
            queue_before=queue4,
            queue_after=queue5,
            demand_before="active",
            demand_after="active",
            scheduled=[{
                "seq_id": decode_seq_id,
                "is_decode": True,
            }],
            progress_updates={},
            token_deltas={str(decode_seq_id): []},
            oldest=decode_seq_id,
            progress=step3_end,
            age=decision4 - step3_end,
            slack=slack4,
            suppression="cost_suppressed",
        ),
        row(
            5,
            decision_now_ns=decision5,
            step_end_ns=step5_end,
            queue_before=queue5,
            queue_after=queue6,
            demand_before="active",
            demand_after="active",
            scheduled=[{
                "seq_id": decode_seq_id,
                "is_decode": False,
                "prefill_chunk_start": 0,
                "prefill_chunk_end": 16,
            }],
            progress_updates={},
            token_deltas={str(decode_seq_id): []},
        ),
        row(
            6,
            decision_now_ns=decision6,
            step_end_ns=decision6 + 1_000_000,
            queue_before=queue6,
            queue_after=queue7,
            demand_before="active",
            demand_after="active",
            scheduled=[{
                "seq_id": decode_seq_id,
                "is_decode": True,
            }],
            progress_updates={},
            token_deltas={str(decode_seq_id): []},
            oldest=decode_seq_id,
            progress=step3_end,
            age=decision6 - step3_end,
            slack=slack6,
            suppression="no_slack",
        ),
        row(
            7,
            decision_now_ns=decision7,
            step_end_ns=decision7 + 1_000_000,
            queue_before=queue7,
            queue_after=queue8,
            demand_before="active",
            demand_after="draining",
            scheduled=[{
                "seq_id": decode_seq_id,
                "is_decode": True,
            }],
            progress_updates={
                str(decode_seq_id): decision7 + 1_000_000,
            },
            token_deltas={str(decode_seq_id): [13]},
            oldest=decode_seq_id,
            progress=step3_end,
            age=decision7 - step3_end,
            slack=slack7,
            suppression="no_slack",
            finished_removed=[decode_seq_id],
        ),
    ]


_legacy_complete_artifact = _complete_artifact


def _complete_artifact() -> tuple[tempfile.TemporaryDirectory, Path]:
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    source_root = root / "snapshot-source"
    source_root.mkdir()
    (source_root / "marker.txt").write_text("arrival-load-source\n")
    source_files = [{
        "path": "marker.txt",
        "size_bytes": (source_root / "marker.txt").stat().st_size,
        "sha256": _sha256_file(source_root / "marker.txt"),
    }]
    source_tree_sha256 = hashlib.sha256(
        _canonical_bytes(source_files)
    ).hexdigest()
    environment_sha256 = "e" * 64
    _write_json(root / "source_evidence.json", {
        "schema_version": 1,
        "base_commit": "1" * 40,
        "tree_sha256": source_tree_sha256,
        "files": source_files,
        "patch_size_bytes": 0,
        "patch_sha256": hashlib.sha256(b"").hexdigest(),
    })
    (root / "source.patch").write_bytes(b"")
    with tarfile.open(root / "source_snapshot.tar.gz", "w:gz") as archive:
        archive.add(source_root, arcname="source")

    calibration = verifier._load_cost_calibration_module()
    cost_capacity = calibration.build_capacity_evidence(
        base_engine_config=verifier.EXPECTED_COST_ENGINE_CONFIG,
        num_kvcache_blocks=448,
        block_size=256,
    )
    shapes = calibration.build_required_shapes(
        max_num_seqs=512,
        max_prefill_tokens=128,
        num_kvcache_blocks=cost_capacity["num_kvcache_blocks"],
        block_size=cost_capacity["block_size"],
    )
    cost_rows = []
    for shape_index, shape in enumerate(shapes):
        for iteration in range(7):
            cost_rows.append({
                **shape,
                "iteration": iteration,
                "duration_ns": (
                    1_000_000
                    + shape["prefill_tokens"] * 10_000
                    + shape_index
                    + iteration
                ),
            })
    cost_summary = calibration.build_cost_calibration_summary(
        source_tree_sha256=source_tree_sha256,
        environment_sha256=environment_sha256,
        engine_config_sha256=verifier._canonical_identity(
            cost_capacity["resolved_engine_config"]
        ),
        required_shapes=shapes,
        raw_rows=cost_rows,
    )
    cost_summary["purpose"] = "authoritative"
    _write_json(
        root / "cost_calibration_capacity.json",
        cost_capacity,
    )
    _write_jsonl(root / "cost_calibration_manifest.jsonl", shapes)
    _write_jsonl(root / "cost_calibration_rows.jsonl", cost_rows)
    _write_json(root / "cost_calibration_summary.json", cost_summary)
    artifact_sha256 = _sha256_file(
        root / "cost_calibration_summary.json"
    )
    intercept_ns = cost_summary["cost_intercept_ns"]
    per_token_ns = cost_summary["cost_per_prefill_token_ns"]

    defaults = {
        **ADAPTIVE_DEFAULTS,
        "chunked_prefill_slo_mixed": False,
        "chunked_prefill_slo_target_gap_ns": 0,
        "chunked_prefill_slo_reserve_ns": 0,
        "chunked_prefill_slo_cost_intercept_ns": 0,
        "chunked_prefill_slo_cost_per_prefill_token_ns": 0,
        "chunked_prefill_slo_min_chunk_tokens": 16,
    }
    resolved = {
        "P0": dict(defaults),
        "P4": {
            **defaults,
            "chunked_prefill_adaptive_mixed": True,
        },
        "P5": {
            **defaults,
            "chunked_prefill_slo_mixed": True,
            "chunked_prefill_slo_target_gap_ns": 64_000_000,
            "chunked_prefill_slo_reserve_ns": 8_000_000,
            "chunked_prefill_slo_cost_intercept_ns": intercept_ns,
            "chunked_prefill_slo_cost_per_prefill_token_ns":
                per_token_ns,
            "chunked_prefill_slo_token_ladder": [
                128, 112, 96, 80, 64, 48, 32, 16,
            ],
            "cost_calibration_artifact_sha256": artifact_sha256,
        },
    }
    scenarios = list(gate.CANONICAL_SCENARIOS)
    manifest = {
        "schema_version": 1,
        "run_tag": "synthetic-p5-canonical",
        "run_type": "canonical",
        "source_tree_sha256": source_tree_sha256,
        "environment_sha256": environment_sha256,
        "required_scenarios": scenarios,
        "measured_repetitions": 3,
        "resolved_policy_config_by_name": resolved,
        "policy_identity_by_name": {
            name: _policy_id(config)
            for name, config in resolved.items()
        },
        "canonical_policy_by_name": {
            "P0": "P0",
            "P4": "P4",
            "P5": "P5",
        },
        "cost_calibration_verification": {
            "status": "PASS",
            "run_tag": "authoritative-cost",
            "artifact_sha256": artifact_sha256,
            "source_tree_sha256": source_tree_sha256,
            "environment_sha256": environment_sha256,
        },
        "process_port_pairs": [],
        "expected_case_ids": [],
    }
    workload_rows = []
    for scenario in scenarios:
        workload_rows.append({
            "schema_version": 1,
            "request_id": f"{scenario}-measured-0000",
            "scenario": scenario,
            "warmup": False,
            "arrival_offset_ns": 0,
            "prompt_token_count": 4,
            "requested_output_tokens": 2,
            "prompt_class": "short",
            "output_class": "short",
            "service_time_bucket": "short__short",
        })
    manifest["workload_sha256"] = verifier._canonical_identity(
        workload_rows
    )
    _write_jsonl(root / "calibration_manifest.jsonl", [{
        "calibration_id": "p0-rate-00",
        "policy": "P0",
        "requested_rate_rps": 1.0,
    }])
    _write_jsonl(root / "calibration_rows.jsonl", [{
        "calibration_id": "p0-rate-00",
        "status": "PASS",
        "stable": True,
        "completed_request_throughput_rps": 1.0,
    }])
    _write_jsonl(root / "workload_manifest.jsonl", workload_rows)

    timeline_rows = []
    scheduler_rows = []
    memory_rows = []
    ports = 31_000
    for scenario_index, scenario in enumerate(scenarios):
        request_id = f"{scenario}-measured-0000"
        for repetition in range(3):
            for policy_index, policy in enumerate(("P0", "P4", "P5")):
                case_id = f"{scenario}__{policy}__r{repetition}"
                manifest["expected_case_ids"].append(case_id)
                manifest["process_port_pairs"].append({
                    "case_id": case_id,
                    "tinyvllm_dist_port": ports,
                    "master_port": ports + 1,
                })
                ports += 2
                seq_id = (
                    scenario_index * 1000
                    + repetition * 100
                    + policy_index * 10
                    + 1
                )
                duration_ns = (
                    800_000_000
                    if policy == "P5" and scenario == "burst"
                    else 1_000_000_000
                )
                timeline_rows.append({
                    "case_id": case_id,
                    "policy": policy,
                    "scenario": scenario,
                    "repetition": repetition,
                    "request_id": request_id,
                    "seq_id": seq_id,
                    "scheduled_arrival_ns": 100,
                    "actual_arrival_ns": 110,
                    "first_scheduled_ns": 120,
                    "first_token_ns": 200,
                    "token_timestamps_ns": [200, duration_ns + 100],
                    "completion_ns": duration_ns + 100,
                    "output_token_ids": [11, 12],
                    "finish_reason": "length",
                    "error": None,
                })
                if policy == "P5":
                    scheduler_rows.extend(_p5_trace(
                        case_id=case_id,
                        scenario=scenario,
                        repetition=repetition,
                        decode_seq_id=seq_id,
                        prefill_seq_id=seq_id + 20,
                        intercept_ns=intercept_ns,
                        per_token_ns=per_token_ns,
                    ))
                elif policy == "P4":
                    empty_queue = {
                        "adaptive_mixed_state": "inactive",
                        "adaptive_high_streak": 0,
                        "adaptive_low_streak": 0,
                        "adaptive_consecutive_mixed_steps": 0,
                        "waiting_seq_ids": [],
                        "prefilling_seq_ids": [],
                        "running_seq_ids": [],
                    }
                    scheduler_rows.append({
                        "case_id": case_id,
                        "policy": policy,
                        "scenario": scenario,
                        "repetition": repetition,
                        "step_index": 0,
                        "policy_branch":
                            "adaptive_mixed_chunked_prefill",
                        "queue_before": empty_queue,
                        "queue_after": empty_queue,
                        "scheduled": [{
                            "seq_id": seq_id,
                            "is_decode": False,
                        }],
                    })
                else:
                    scheduler_rows.append({
                        "case_id": case_id,
                        "policy": policy,
                        "scenario": scenario,
                        "repetition": repetition,
                        "step_index": 0,
                        "step_start_ns": 120,
                        "step_end_ns": duration_ns + 100,
                        "scheduled": [{
                            "seq_id": seq_id,
                            "is_decode": True,
                        }],
                    })
                memory_rows.append({
                    "case_id": case_id,
                    "policy": policy,
                    "scenario": scenario,
                    "repetition": repetition,
                    "step_index": 0,
                    "cuda_allocated_bytes": 100,
                    "cuda_reserved_bytes": 200,
                    "used_kv_blocks": 2,
                    "kv_block_bytes": 64,
                })
    _write_json(root / "run_manifest.json", manifest)
    _write_jsonl(root / "request_timeline.jsonl", timeline_rows)
    _write_jsonl(root / "scheduler_trace.jsonl", scheduler_rows)
    _write_jsonl(root / "memory_trace.jsonl", memory_rows)

    workload_by_id = {row["request_id"]: row for row in workload_rows}
    case_rows = []
    for case_id in manifest["expected_case_ids"]:
        case_scheduler = [
            row for row in scheduler_rows
            if row["case_id"] == case_id
        ]
        p5_policy = (
            gate.summarize_p5_policy(case_scheduler)
            if case_scheduler[0]["policy"] == "P5"
            else None
        )
        case_rows.append(verifier._recompute_case(
            case_id,
            timeline_rows,
            scheduler_rows,
            memory_rows,
            workload_by_id,
            p5_policy=p5_policy,
        ))
    _write_jsonl(root / "case_rows.jsonl", case_rows)
    summary = verifier._classify(manifest, case_rows)
    _write_json(root / "summary.json", summary)
    (root / "report.md").write_text(verifier._render_report(summary))
    required = set(verifier.REQUIRED_FILES) - {
        "artifact_hashes.json"
    }
    _write_json(root / "artifact_hashes.json", {
        name: _sha256_file(root / name)
        for name in sorted(required)
    })
    return temporary, root


def _add_warmup_request(root: Path) -> None:
    workload_rows = verifier._read_jsonl(
        root / "workload_manifest.jsonl"
    )
    warmup_request_id = "steady_moderate-warmup-0000"
    workload_rows.append({
        **workload_rows[0],
        "request_id": warmup_request_id,
        "warmup": True,
        "arrival_offset_ns": -1_000_000_000,
    })
    _write_jsonl(root / "workload_manifest.jsonl", workload_rows)
    _refresh_hash(root, "workload_manifest.jsonl")
    manifest = json.loads((root / "run_manifest.json").read_text())
    manifest["workload_sha256"] = verifier._canonical_identity(
        workload_rows
    )
    _write_json(root / "run_manifest.json", manifest)
    _refresh_hash(root, "run_manifest.json")

    timeline_rows = verifier._read_jsonl(
        root / "request_timeline.jsonl"
    )
    warmup_rows = []
    for row in timeline_rows:
        warmup_rows.append({
            **row,
            "request_id": warmup_request_id,
            "seq_id": row["seq_id"] + 1000,
            "scheduled_arrival_ns": 0,
            "actual_arrival_ns": 10,
            "first_scheduled_ns": 20,
            "first_token_ns": 30,
            "token_timestamps_ns": [30, 40],
            "completion_ns": 40,
        })
    _write_jsonl(
        root / "request_timeline.jsonl",
        warmup_rows + timeline_rows,
    )
    _refresh_hash(root, "request_timeline.jsonl")


def test_verifier_does_not_import_harness_aggregation():
    source = VERIFY_PATH.read_text()
    assert "import arrival_load_gate" not in source
    assert "from arrival_load_gate" not in source


def test_p5_verifier_contract_replaces_p3_and_requires_cost_artifacts():
    required = set(verifier.REQUIRED_FILES)
    assert {
        "cost_calibration_manifest.jsonl",
        "cost_calibration_rows.jsonl",
        "cost_calibration_summary.json",
    } <= required
    manifest = {
        "canonical_policy_by_name": {
            "P0": "P0",
            "P4": "P4",
            "P5": "P5",
        },
        "resolved_policy_config_by_name": {
            "P0": dict(ADAPTIVE_DEFAULTS),
            "P4": {
                **ADAPTIVE_DEFAULTS,
                "chunked_prefill_adaptive_mixed": True,
            },
            "P5": {
                **ADAPTIVE_DEFAULTS,
                "chunked_prefill_slo_mixed": True,
                "chunked_prefill_slo_target_gap_ns": 64_000_000,
                "chunked_prefill_slo_reserve_ns": 8_000_000,
                "chunked_prefill_slo_cost_intercept_ns": 4_000_000,
                "chunked_prefill_slo_cost_per_prefill_token_ns": 100_000,
                "chunked_prefill_slo_min_chunk_tokens": 16,
                "chunked_prefill_slo_token_ladder": [
                    128, 112, 96, 80, 64, 48, 32, 16,
                ],
                "cost_calibration_artifact_sha256": "a" * 64,
            },
        },
    }
    manifest["policy_identity_by_name"] = {
        name: _policy_id(config)
        for name, config in manifest[
            "resolved_policy_config_by_name"
        ].items()
    }
    p5 = verifier._verify_policy_manifest(manifest)
    assert p5[
        "cost_calibration_artifact_sha256"
    ] == "a" * 64


def test_output_equality_uses_recorded_case_metadata_not_case_id_format():
    manifest = {
        "required_scenarios": ["steady_moderate"],
        "measured_repetitions": 1,
    }
    common = {
        "scenario": "steady_moderate",
        "repetition": 0,
        "request_id": "request-0",
    }
    rows = [
        {
            **common,
            "case_id": "steady_moderate__P0__r0",
            "policy": "P0",
            "output_token_ids": [1, 2],
        },
        {
            **common,
            "case_id": "steady_moderate__P4__r0",
            "policy": "P4",
            "output_token_ids": [1, 3],
        },
        {
            **common,
            "case_id": "steady_moderate__P3__r0",
            "policy": "P3",
            "output_token_ids": [1, 2],
        },
    ]
    try:
        verifier._verify_output_equality(rows, manifest)
    except ValueError as exc:
        assert "output token mismatch" in str(exc)
    else:
        raise AssertionError("canonical case-id output drift was missed")


def test_smoke_summary_is_lifecycle_only():
    rows = [
        {
            "case_id": "lifecycle_smoke__P0__r0",
            "policy": "P0",
            "scenario": "lifecycle_smoke",
            "repetition": 0,
            "status": "PASS",
            "correctness": {
                "exact_outputs": True,
                "complete_requests": True,
                "no_starvation": True,
                "valid_lifecycle": True,
                "stable_p0_outputs": True,
            },
        },
        {
            "case_id": "lifecycle_smoke__P4__r0",
            "policy": "P4",
            "scenario": "lifecycle_smoke",
            "repetition": 0,
            "status": "PASS",
            "correctness": {
                "exact_outputs": True,
                "complete_requests": True,
                "no_starvation": True,
                "valid_lifecycle": True,
                "stable_p0_outputs": True,
            },
        },
    ]
    summary = verifier._smoke_summary(rows)
    assert summary == {
        "classification": "SMOKE_ONLY",
        "lifecycle_complete": True,
        "exact_outputs": True,
        "case_count": 2,
    }


def test_verifier_recomputes_complete_artifact():
    temporary, root = _complete_artifact()
    try:
        result = verifier.verify_run(root, write_output=True)
        assert result["classification"] == "NO_GO"
        assert (
            root / "independent-verify/summary.json"
        ).is_file()
        assert (
            root / "independent-verify/verify.exitcode"
        ).read_text().strip() == "0"
    finally:
        temporary.cleanup()


def test_p5_fixture_covers_required_structural_paths():
    temporary, root = _complete_artifact()
    try:
        case_rows = verifier._read_jsonl(root / "case_rows.jsonl")
        assert len(case_rows) == 54
        rows = [
            row for row in verifier._read_jsonl(
                root / "scheduler_trace.jsonl"
            )
            if row["policy"] == "P5"
        ]
        assert {
            row["selected_chunk_tokens"]
            for row in rows
            if row["selected_chunk_tokens"] is not None
        } >= {128, 64}
        assert {
            row["suppression_reason"]
            for row in rows
            if row["suppression_reason"] is not None
        } >= {"inactive", "cost_suppressed", "no_slack"}
        assert {
            row["demand_state_after"] for row in rows
        } >= {"active", "draining"}
        assert any(
            row["finished_progress_entries_removed"]
            for row in rows
        )
        assert any(
            row["queue_before"]["running_seq_ids"]
            and not row["queue_after"]["running_seq_ids"]
            and set(row["queue_before"]["running_seq_ids"])
            <= set(row["queue_after"]["waiting_seq_ids"])
            for row in rows
        )
        assert any(
            row["step_index"] == 6
            and row["oldest_decode_progress_ns"]
            < row["decision_now_ns"]
            for row in rows
        )
        assert all(
            row["actual_step_duration_ns"]
            <= row["predicted_step_ns"]
            for row in rows
            if row["selected_chunk_tokens"] is not None
        )
    finally:
        temporary.cleanup()


def test_verifier_excludes_warmup_requests_from_case_metrics():
    temporary, root = _complete_artifact()
    try:
        _add_warmup_request(root)
        result = verifier.verify_run(root, write_output=False)
        assert result["classification"] == "NO_GO"
    finally:
        temporary.cleanup()


def test_verifier_still_validates_warmup_request_lifecycle():
    temporary, root = _complete_artifact()
    try:
        _add_warmup_request(root)
        timeline_rows = verifier._read_jsonl(
            root / "request_timeline.jsonl"
        )
        warmup = next(
            row for row in timeline_rows
            if row["request_id"] == "steady_moderate-warmup-0000"
        )
        warmup["first_scheduled_ns"] = warmup["actual_arrival_ns"] - 1
        _write_jsonl(root / "request_timeline.jsonl", timeline_rows)
        _refresh_hash(root, "request_timeline.jsonl")

        try:
            verifier.verify_run(root, write_output=False)
        except ValueError as exc:
            assert "impossible timestamp ordering" in str(exc)
        else:
            raise AssertionError("invalid warmup lifecycle must fail")
    finally:
        temporary.cleanup()


def test_verifier_rejects_summary_tampering_even_when_rehashed():
    temporary, root = _complete_artifact()
    try:
        recorded = json.loads((root / "summary.json").read_text())
        recorded["classification"] = "GO"
        _write_json(root / "summary.json", recorded)
        _refresh_hash(root, "summary.json")
        try:
            verifier.verify_run(root, write_output=False)
        except ValueError as exc:
            assert "classification disagreement" in str(exc)
        else:
            raise AssertionError("tampered summary must be rejected")
    finally:
        temporary.cleanup()


def test_verifier_rejects_truncated_jsonl_and_duplicate_ports():
    temporary, root = _complete_artifact()
    try:
        path = root / "request_timeline.jsonl"
        path.write_bytes(path.read_bytes()[:-1])
        _refresh_hash(root, "request_timeline.jsonl")
        try:
            verifier.verify_run(root, write_output=False)
        except ValueError as exc:
            assert "final newline" in str(exc)
        else:
            raise AssertionError("truncated JSONL must fail")
    finally:
        temporary.cleanup()

    temporary, root = _complete_artifact()
    try:
        manifest = json.loads((root / "run_manifest.json").read_text())
        manifest["process_port_pairs"][1][
            "tinyvllm_dist_port"
        ] = manifest["process_port_pairs"][0]["tinyvllm_dist_port"]
        manifest["process_port_pairs"][1][
            "master_port"
        ] = manifest["process_port_pairs"][0]["master_port"]
        _write_json(root / "run_manifest.json", manifest)
        _refresh_hash(root, "run_manifest.json")
        try:
            verifier.verify_run(root, write_output=False)
        except ValueError as exc:
            assert "duplicate process port pair" in str(exc)
        else:
            raise AssertionError("duplicate ports must fail")
    finally:
        temporary.cleanup()


def test_verifier_rejects_rehashed_source_output_and_scheduler_tampering():
    temporary, root = _complete_artifact()
    try:
        (root / "source.patch").write_bytes(b"tampered patch\n")
        _refresh_hash(root, "source.patch")
        try:
            verifier.verify_run(root, write_output=False)
        except ValueError as exc:
            assert "source patch" in str(exc)
        else:
            raise AssertionError("rehashed source patch tampering must fail")
    finally:
        temporary.cleanup()

    temporary, root = _complete_artifact()
    try:
        rows = [
            json.loads(line)
            for line in (root / "request_timeline.jsonl").read_text().splitlines()
        ]
        candidate = next(
            row for row in rows
            if row["case_id"] == "steady_moderate__P5__r0"
        )
        candidate["output_token_ids"][-1] = 99
        _write_jsonl(root / "request_timeline.jsonl", rows)
        _refresh_hash(root, "request_timeline.jsonl")
        try:
            verifier.verify_run(root, write_output=False)
        except ValueError as exc:
            assert "output token mismatch" in str(exc)
        else:
            raise AssertionError("candidate output tampering must fail")
    finally:
        temporary.cleanup()

    temporary, root = _complete_artifact()
    try:
        rows = [
            json.loads(line)
            for line in (root / "scheduler_trace.jsonl").read_text().splitlines()
        ]
        rows = [
            row for row in rows
            if row["case_id"] != "steady_moderate__P5__r2"
        ]
        _write_jsonl(root / "scheduler_trace.jsonl", rows)
        _refresh_hash(root, "scheduler_trace.jsonl")
        try:
            verifier.verify_run(root, write_output=False)
        except ValueError as exc:
            assert "missing scheduler trace" in str(exc)
        else:
            raise AssertionError("missing scheduler trace must fail")
    finally:
        temporary.cleanup()


def _mutate_first_p5_trace(root: Path, mutate) -> None:
    rows = verifier._read_jsonl(root / "scheduler_trace.jsonl")
    row = next(
        row for row in rows
        if row["policy"] == "P5"
    )
    mutate(row)
    _write_jsonl(root / "scheduler_trace.jsonl", rows)
    _refresh_hash(root, "scheduler_trace.jsonl")


def test_verifier_rejects_each_p5_decision_field_tamper():
    tamper_cases = {
        "decision time": lambda row: row.__setitem__(
            "decision_now_ns", 1
        ),
        "progress timestamp": lambda row: row[
            "decode_progress_updates"
        ].__setitem__(next(iter(row["decode_progress_updates"])), 1),
        "oldest identity": lambda row: row.__setitem__(
            "oldest_decode_seq_id", 999
        ),
        "age": lambda row: row.__setitem__(
            "oldest_decode_age_ns", 1
        ),
        "slack": lambda row: row.__setitem__(
            "remaining_slack_ns", 1
        ),
        "coefficient": lambda row: row.__setitem__(
            "cost_per_prefill_token_ns", 1
        ),
        "selected chunk": lambda row: row.__setitem__(
            "selected_chunk_tokens", 16
        ),
        "protected row": lambda row: row.__setitem__(
            "scheduled_decode_seq_ids", [999]
        ),
        "actual prefill": lambda row: row.__setitem__(
            "actual_prefill_tokens", 129
        ),
        "suppression": lambda row: row.__setitem__(
            "suppression_reason", "forged"
        ),
        "progress update": lambda row: row.__setitem__(
            "decode_progress_updates", {}
        ),
        "finished removal": lambda row: row.__setitem__(
            "finished_progress_entries_removed", [999]
        ),
    }
    for label, mutate in tamper_cases.items():
        temporary, root = _complete_artifact()
        try:
            _mutate_first_p5_trace(root, mutate)
            try:
                verifier.verify_run(root, write_output=False)
            except ValueError:
                pass
            else:
                raise AssertionError(f"P5 tamper accepted: {label}")
        finally:
            temporary.cleanup()


def test_verifier_rejects_raw_cost_calibration_tamper():
    temporary, root = _complete_artifact()
    try:
        rows = verifier._read_jsonl(
            root / "cost_calibration_rows.jsonl"
        )
        rows[0]["duration_ns"] += 1
        _write_jsonl(root / "cost_calibration_rows.jsonl", rows)
        _refresh_hash(root, "cost_calibration_rows.jsonl")
        try:
            verifier.verify_run(root, write_output=False)
        except ValueError:
            pass
        else:
            raise AssertionError("raw cost calibration tamper accepted")
    finally:
        temporary.cleanup()


def test_verifier_rejects_capacity_calibration_tamper():
    temporary, root = _complete_artifact()
    try:
        capacity = json.loads(
            (root / "cost_calibration_capacity.json").read_text()
        )
        capacity["num_kvcache_blocks"] += 1
        _write_json(
            root / "cost_calibration_capacity.json",
            capacity,
        )
        _refresh_hash(root, "cost_calibration_capacity.json")
        try:
            verifier.verify_run(root, write_output=False)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "capacity calibration tamper accepted"
            )
    finally:
        temporary.cleanup()


def test_verifier_rejects_source_environment_and_workload_identity_tamper():
    def tamper_source(root):
        manifest = json.loads((root / "run_manifest.json").read_text())
        manifest["source_tree_sha256"] = "f" * 64
        _write_json(root / "run_manifest.json", manifest)
        _refresh_hash(root, "run_manifest.json")

    def tamper_environment(root):
        manifest = json.loads((root / "run_manifest.json").read_text())
        manifest["environment_sha256"] = "f" * 64
        _write_json(root / "run_manifest.json", manifest)
        _refresh_hash(root, "run_manifest.json")

    def tamper_workload(root):
        rows = verifier._read_jsonl(root / "workload_manifest.jsonl")
        rows[0]["prompt_token_count"] += 1
        _write_jsonl(root / "workload_manifest.jsonl", rows)
        _refresh_hash(root, "workload_manifest.jsonl")

    for label, tamper in {
        "source": tamper_source,
        "environment": tamper_environment,
        "workload": tamper_workload,
    }.items():
        temporary, root = _complete_artifact()
        try:
            tamper(root)
            try:
                verifier.verify_run(root, write_output=False)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"{label} identity tamper accepted"
                )
        finally:
            temporary.cleanup()


def test_verifier_rejects_p5_policy_identity_drift():
    temporary, root = _complete_artifact()
    try:
        manifest = json.loads((root / "run_manifest.json").read_text())
        manifest["resolved_policy_config_by_name"]["P5"][
            "chunked_prefill_slo_reserve_ns"
        ] += 1
        _write_json(root / "run_manifest.json", manifest)
        _refresh_hash(root, "run_manifest.json")
        try:
            verifier.verify_run(root, write_output=False)
        except ValueError as exc:
            assert "policy identity mismatch" in str(exc)
        else:
            raise AssertionError("P5 policy identity drift accepted")
    finally:
        temporary.cleanup()


def main():
    test_verifier_does_not_import_harness_aggregation()
    test_p5_verifier_contract_replaces_p3_and_requires_cost_artifacts()
    test_output_equality_uses_recorded_case_metadata_not_case_id_format()
    test_smoke_summary_is_lifecycle_only()
    test_verifier_recomputes_complete_artifact()
    test_p5_fixture_covers_required_structural_paths()
    test_verifier_excludes_warmup_requests_from_case_metrics()
    test_verifier_still_validates_warmup_request_lifecycle()
    test_verifier_rejects_summary_tampering_even_when_rehashed()
    test_verifier_rejects_truncated_jsonl_and_duplicate_ports()
    test_verifier_rejects_rehashed_source_output_and_scheduler_tampering()
    test_verifier_rejects_each_p5_decision_field_tamper()
    test_verifier_rejects_raw_cost_calibration_tamper()
    test_verifier_rejects_capacity_calibration_tamper()
    test_verifier_rejects_source_environment_and_workload_identity_tamper()
    test_verifier_rejects_p5_policy_identity_drift()
    print("arrival load verifier tests passed")


if __name__ == "__main__":
    main()
