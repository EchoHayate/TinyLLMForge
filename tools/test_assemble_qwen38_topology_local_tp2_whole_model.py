from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.assemble_qwen38_topology_local_tp2_whole_model import (
    CORE_ARTIFACTS,
    assemble_attempt,
    classify,
    nearest_rank_percentile,
)
import tools.assemble_qwen38_topology_local_tp2_whole_model as assembler


WORKLOADS = ("P0", "P1", "Q0", "Q1", "Q2")
ARMS = ("baseline", "candidate", "candidate", "baseline")


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ))


def _correctness_evidence(concurrency):
    linear_layer_indices = tuple(
        index for index in range(64) if index % 4 != 3
    )
    requests = [{
        "request_id": f"request-{request_index}",
        "runtime_request_id": request_index,
        "output_token_ids": [
            step + request_index for step in range(128)
        ],
    } for request_index in range(concurrency)]
    proofs = [[{
        "rank": rank,
        "sequence_ids": list(range(concurrency)),
        "finite_logits": True,
        "token_ids": [
            step + request_index
            for request_index in range(concurrency)
        ],
        "top_logit_values": [
            float(step + request_index)
            for request_index in range(concurrency)
        ],
    } for rank in range(4)] for step in range(128)]
    commit_counts = {
        "pre_migration": 0,
        "token_1": 0,
        "post_migration": 1,
        "token_4": 3,
        "token_8": 7,
        "token_32": 31,
        "token_128": 127,
    }
    def component(layer, source_rank):
        return {
            "layer_index": layer,
            "logical_rank": source_rank // 2,
            "source_rank": source_rank,
            "convolution_query_sha256": (
                f"{layer * 16 + source_rank * 4:064x}"
            ),
            "convolution_key_sha256": (
                f"{layer * 16 + source_rank * 4 + 1:064x}"
            ),
            "convolution_value_sha256": (
                f"{layer * 16 + source_rank * 4 + 2:064x}"
            ),
            "recurrent_sha256": (
                f"{layer * 16 + source_rank * 4 + 3:064x}"
            ),
        }

    baseline_checkpoints = {}
    checkpoints = {}
    for name, commit_count in commit_counts.items():
        baseline_rows = []
        rows = []
        for rank in range(4):
            active = name not in {"pre_migration", "token_1"}
            cohort = [{
                "slot_id": request_index,
                "generation": 1,
                "request_id": request_index,
            } for request_index in range(concurrency)]
            baseline_rows.append({
                "rank": rank,
                "pair_id": 0 if rank < 2 else 1,
                "logical_rank": rank % 2,
                "state_layout": "tp4_source_quarter",
                "cohort": copy.deepcopy(cohort),
                "canonical_state_components": [
                    component(layer, rank)
                    for layer in linear_layer_indices
                ],
            })
            rows.append({
                "rank": rank,
                "pair_id": 0 if rank < 2 else 1,
                "logical_rank": rank % 2,
                "state_layout": (
                    "tp2_logical_half"
                    if active
                    else "tp4_source_quarter"
                ),
                "cohort": copy.deepcopy(cohort),
                "output_digests": [{
                    "layer_index": layer,
                    "sha256": f"{layer:064x}",
                } for layer in linear_layer_indices] if active else [],
                "state_digests": [{
                    "layer_index": layer,
                    "convolution_sha256": (
                        f"{layer * 2 + rank % 2:064x}"
                    ),
                    "recurrent_sha256": (
                        f"{layer * 2 + rank % 2 + 1:064x}"
                    ),
                } for layer in linear_layer_indices] if active else [],
                "canonical_state_components": [
                    component(layer, source_rank)
                    for layer in linear_layer_indices
                    for source_rank in (
                        (
                            2 * (rank % 2),
                            2 * (rank % 2) + 1,
                        )
                        if active
                        else (rank,)
                    )
                ],
                "runtime_snapshot": {
                    "state": {
                        "commit_count": commit_count,
                        "rollback_count": 0,
                        "temporary_live_tensors": 0,
                    },
                },
            })
        baseline_checkpoints[name] = baseline_rows
        checkpoints[name] = rows
    return {
        "baseline_requests": requests,
        "candidate_requests": copy.deepcopy(requests),
        "baseline_step_proofs": proofs,
        "candidate_step_proofs": copy.deepcopy(proofs),
        "baseline_state_checkpoints": baseline_checkpoints,
        "candidate_state_checkpoints": checkpoints,
    }


def _timing_correctness_evidence(requests):
    replay_requests = [{
        "request_id": request["request_id"],
        "runtime_request_id": request_index,
        "output_token_ids": list(request["output_token_ids"]),
        "stop_position": request["stop_position"],
        "stop_reason": request["stop_reason"],
        "decoded_text": request["decoded_text"],
        "decoded_text_sha256": request["decoded_text_sha256"],
    } for request_index, request in enumerate(requests)]
    proofs = [[{
        "rank": rank,
        "sequence_ids": list(range(len(replay_requests))),
        "finite_logits": True,
        "token_ids": [
            request["output_token_ids"][step]
            for request in replay_requests
        ],
        "top_logit_values": [
            float(request["output_token_ids"][step])
            for request in replay_requests
        ],
    } for rank in range(4)] for step in range(128)]
    return {
        "rank_token_agreement": True,
        "finite_logits": True,
        "top_logit_values_match": True,
        "timing_correctness_replay": {
            "requests": replay_requests,
            "step_proofs": proofs,
        },
    }


def passing_summary():
    return {
        "correctness_pass": True,
        "resource_identity_pass": True,
        "candidate_coverage_pass": True,
        "memory_pass": True,
        "measurement_complete": True,
        "tail_ttft_pass": True,
        "throughput_pass": True,
        "migration_pass": True,
        "aggregate_median_tpot_improvement_percent": 5.0,
        "improving_workload_count": 4,
        "median_regressing_workloads": [],
        "pair_direction_counts": {
            workload: 7 for workload in WORKLOADS
        },
    }


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (
            {"correctness_pass": False},
            "NO_GO_CORRECTNESS_OR_LIFECYCLE",
        ),
        (
            {"resource_identity_pass": False},
            "NO_GO_RESOURCE_IDENTITY",
        ),
        (
            {"candidate_coverage_pass": False},
            "NO_GO_CANDIDATE_NOT_EXERCISED",
        ),
        (
            {"memory_pass": False},
            "NO_GO_MEMORY_OR_ALLOCATION",
        ),
        (
            {"measurement_complete": False},
            "INCONCLUSIVE_ENVIRONMENT_OR_MEASUREMENT",
        ),
        (
            {"tail_ttft_pass": False},
            "NO_GO_TAIL_OR_TTFT",
        ),
        (
            {"throughput_pass": False},
            "NO_GO_THROUGHPUT",
        ),
        (
            {"migration_pass": False},
            "NO_GO_MIGRATION_AMORTIZATION",
        ),
        (
            {"aggregate_median_tpot_improvement_percent": 4.99},
            "NO_GO_PERFORMANCE",
        ),
        ({}, "GO_TOPOLOGY_LOCAL_TP2_WHOLE_MODEL_GATE"),
    ),
)
def test_classifier_precedence(mutation, expected):
    summary = passing_summary()
    summary.update(mutation)

    assert classify(summary) == expected


def test_nearest_rank_percentile_uses_raw_values():
    values = list(range(1, 101))

    assert nearest_rank_percentile(values, 95) == 95.0
    assert nearest_rank_percentile(values, 99) == 99.0


def test_assembler_cli_writes_to_explicit_fresh_output(tmp_path):
    calls = []
    attempt = tmp_path / "attempt"
    output = tmp_path / "bundle"

    assert assembler.main(
        [
            "--attempt-root",
            str(attempt),
            "--output-root",
            str(output),
        ],
        assemble=lambda attempt_root, output_root: (
            calls.append((attempt_root, output_root))
            or {"classification": "NO_GO_PERFORMANCE"}
        ),
        printer=lambda value: calls.append(value),
    ) == 0

    assert calls[0] == (attempt, output)
    assert json.loads(calls[1])["classification"] == "NO_GO_PERFORMANCE"


def passing_attempt(root: Path) -> Path:
    raw = root / "raw"
    source_revision = "a" * 40
    model_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    identity = {
        "source_revision": source_revision,
        "model_revision": model_revision,
    }
    cleanup_records = []
    cleanup_labels = (
        ("correctness/baseline", False, 4),
        ("correctness/candidate", True, 4),
        ("epoch/0/baseline", False, 4),
        ("epoch/1/candidate", True, 4),
        ("epoch/2/candidate", True, 4),
        ("epoch/3/baseline", False, 4),
        ("service/Q0/replica/0", False, 2),
        ("service/Q0/replica/1", False, 2),
        ("service/Q1/replica/0", False, 2),
        ("service/Q1/replica/1", False, 2),
        ("service/Q2/replica/0", False, 2),
        ("service/Q2/replica/1", False, 2),
    )
    for label, candidate_enabled, rank_count in cleanup_labels:
        cleanup_records.append({
            "label": label,
            "candidate_enabled": candidate_enabled,
            "receipt": {
                "process_group_destroyed": True,
                "rank_exit_codes": [0] * rank_count,
                "owned_children_remaining": [],
                "cleanup_started_ns": 10,
                "cleanup_finished_ns": 110,
                "cleanup_duration_ns": 100,
                "rank_cleanup_receipts": [{
                    "rank": rank,
                    "process_group_destroyed": True,
                    "qwen38_topology_local_tp2_cleanup": (
                        {
                            "pair_groups_destroyed": 2,
                            "candidate_state_released": True,
                            "published_generations_remaining": 0,
                            "temporary_live_tensors": 0,
                        }
                        if candidate_enabled
                        else None
                    ),
                } for rank in range(rank_count)],
            },
        })
    json_payloads = {
        "source_manifest.json": {
            **identity,
            "attempt_tag": "fixture-attempt",
            "source_archive_complete": True,
            "source_tree_sha256": "b" * 64,
        },
        "model_manifest.json": {
            **identity,
            "model_repository": "Qwen/Qwen3.8-27B",
            "num_hidden_layers": 64,
            "linear_attention_layer_count": 48,
            "full_attention_layer_count": 16,
        },
        "environment_manifest.json": {
            **identity,
            "environment_complete": True,
        },
        "gpu_topology.json": {
            **identity,
            "selection_frozen": True,
            "rows": [
                {
                    "left_rank": left,
                    "right_rank": right,
                    "link": (
                        "PIX"
                        if tuple(sorted((left, right)))
                        in {(0, 1), (2, 3)}
                        else "SYS"
                    ),
                }
                for left in range(4)
                for right in range(4)
                if left != right
            ],
        },
        "gpu_rank_manifest.json": {
            **identity,
            "ranks": list(range(4)),
            "mapping": [{
                "rank": rank,
                "gpu_index": rank,
                "gpu_uuid": f"GPU-{rank}",
            } for rank in range(4)],
        },
        "pair_group_manifest.json": {
            **identity,
            "pair_groups": [[0, 1], [2, 3]],
            "optimal_matching": True,
        },
        "workload_manifest.json": {
            **identity,
            "workloads": list(WORKLOADS),
        },
        "campaign_epoch_manifest.json": {
            **identity,
            "epochs": [
                {
                    "epoch": epoch,
                    "arm": arm,
                    "workload_order": (
                        list(WORKLOADS)
                        if epoch in (0, 2)
                        else list(reversed(WORKLOADS))
                    ),
                }
                for epoch, arm in enumerate(ARMS)
            ],
        },
        "feature_contract.json": {
            **identity,
            "default_off": True,
            "eager": True,
            "tensor_parallel_size": 4,
        },
        "weight_layout_manifest.json": {
            **identity,
            "baseline_tp4_decode_accumulation_retained": False,
            "steady_increment_bytes_per_rank": 1_800 * 1024**2,
        },
        "state_layout_manifest.json": {
            **identity,
            "temporary_objects_released": True,
        },
        "cleanup.json": {
            **identity,
            "complete": True,
            "retained_generations": 0,
            "retained_leases": 0,
            "retained_tensors": 0,
            "retained_process_groups": 0,
            "owned_processes_remaining": [],
            "foreign_process_actions": [],
            "worker_cleanup_receipts": cleanup_records,
            "cleanup_durations_ns": [
                record["receipt"]["cleanup_duration_ns"]
                for record in cleanup_records
            ],
            "validated_worker_cleanups": 12,
            "validated_rank_cleanup_receipts": 36,
            "task_paths": [
                (
                    "/data00/home/sitian/tinyllmforge-workspaces/"
                    "command-timeline-20260818/attempts/a/raw"
                )
            ],
        },
    }
    for name, payload in json_payloads.items():
        _write_json(raw / name, payload)

    request_rows = []
    scheduler_rows = []
    candidate_rows = []
    migration_rows = []
    collective_rows = []
    memory_rows = []
    for epoch, arm in enumerate(ARMS):
        for workload_index, workload in enumerate(WORKLOADS):
            concurrency = (1, 1, 4, 8, 4)[workload_index]
            for repetition in range(5):
                digest = hashlib.sha256(
                    f"{workload}:{repetition}".encode()
                ).hexdigest()
                base_tpot = 10_000 + workload_index * 100
                tpot = (
                    base_tpot
                    if arm == "baseline"
                    else base_tpot * 94 // 100
                )
                requests = []
                for request_index in range(concurrency):
                    admitted_ns = 1_000_000
                    token_timestamps_ns = [
                        admitted_ns + 1_000 + step * tpot
                        for step in range(128)
                    ]
                    decoded_text = ",".join(
                        str(token) for token in range(128)
                    )
                    requests.append({
                        "request_id": f"{workload}-{repetition}-{request_index}",
                        "runtime_request_id": request_index,
                        "admitted_ns": admitted_ns,
                        "first_scheduled_ns": admitted_ns + 500,
                        "queueing_ns": 500,
                        "token_timestamps_ns": token_timestamps_ns,
                        "output_token_ids": list(range(128)),
                        "token_gaps_ns": [tpot] * 127,
                        "ttft_ns": 1_000.0,
                        "tpot_ns": tpot,
                        "e2e_ns": 1_000.0 + tpot * 127,
                        "completion_ns": token_timestamps_ns[-1],
                        "complete": True,
                        "prompt_tokens": (
                            2_048 if workload in {"P1", "Q2"} else 256
                        ),
                        "generated_tokens": 128,
                        "stop_position": 128,
                        "stop_reason": "length",
                        "decoded_text": decoded_text,
                        "decoded_text_sha256": hashlib.sha256(
                            decoded_text.encode("utf-8")
                        ).hexdigest(),
                    })
                row = {
                    **identity,
                    "epoch": epoch,
                    "arm": arm,
                    "workload_id": workload,
                    "repetition": repetition,
                    "request_set_digest": digest,
                    "requests": requests,
                    "cohort_makespan_ns": max(
                        request["completion_ns"] for request in requests
                    ) - min(
                        request["admitted_ns"] for request in requests
                    ),
                    **_timing_correctness_evidence(requests),
                }
                request_rows.append(row)
                scheduler_rows.append({
                    **identity,
                    "epoch": epoch,
                    "arm": arm,
                    "workload_id": workload,
                    "repetition": repetition,
                    "request_set_digest": digest,
                    "startup_model_load_started_ns": 100,
                    "startup_model_load_finished_ns": 1_000_100,
                    "startup_model_load_duration_ns": 1_000_000,
                    "steps": [{
                        "step_index": step,
                        "is_prefill": step == 0,
                        "batch_kind": (
                            "prefill" if step == 0 else "decode"
                        ),
                        "request_ids": [
                            request["request_id"] for request in requests
                        ],
                        "step_start_ns": admitted_ns + 500 + step * tpot,
                        "step_end_ns": (
                            admitted_ns + 580 + step * tpot
                        ),
                        "step_duration_ns": 80,
                        "host_submission_ns": 60,
                    } for step in range(128)],
                    "decode_steps": 127,
                    "token_one_segments": concurrency * 127,
                })
                if arm == "candidate":
                    expected_segments = concurrency * 127
                    candidate_rows.append({
                        **identity,
                        "epoch": epoch,
                        "arm": arm,
                        "workload_id": workload,
                        "repetition": repetition,
                        "request_set_digest": digest,
                        "tp2_decode_calls": expected_segments * 48,
                        "recurrent_token_one_calls": expected_segments * 48,
                        "short_chunk_calls": 0,
                        "ordinary_chunk_calls": 0,
                        "global_tp4_linear_decode_all_reduce_calls": 0,
                        "full_attention_tp4_collective_calls": (
                            expected_segments * 16
                        ),
                        "migration_publications": concurrency,
                        "fallback_calls": 0,
                        "post_warmup_request_path_allocations": 0,
                        "retry_after_mutation_calls": 0,
                        "duplicate_commit_calls": 0,
                    })
                    migration_rows.append({
                        **identity,
                        "epoch": epoch,
                        "arm": arm,
                        "workload_id": workload,
                        "repetition": repetition,
                        "request_set_digest": digest,
                        "latency_ns": (base_tpot - tpot) * 8.0,
                        "break_even_output_tokens": 8.0,
                        "temporary_live_tensors": 0,
                    })
                    collective_rows.append({
                        **identity,
                        "epoch": epoch,
                        "arm": arm,
                        "workload_id": workload,
                        "repetition": repetition,
                        "request_set_digest": digest,
                        "pair_local_calls": expected_segments * 48,
                        "pair_local_bytes": expected_segments * 48 * 5120 * 4,
                        "full_attention_tp4_calls": expected_segments * 16,
                        "full_attention_tp4_bytes": (
                            expected_segments * 16 * 5120 * 2
                        ),
                        "pair_local_sequence_match": True,
                    })
        for rank in range(4):
            memory_rows.append({
                **identity,
                "epoch": epoch,
                "arm": arm,
                "rank": rank,
                "peak_allocated_bytes": 70 * 1024**3,
                "peak_reserved_bytes": 72 * 1024**3,
                "physical_memory_bytes": 80 * 1024**3,
            })

    correctness_rows = []
    for workload_index, workload in enumerate(WORKLOADS):
        concurrency = (1, 1, 4, 8, 4)[workload_index]
        for repetition in range(5):
            correctness_rows.append({
                **identity,
                "workload_id": workload,
                "repetition": repetition,
                "output_tokens_match": True,
                "rank_token_agreement": True,
                "finite_logits": True,
                "top_logit_values_match": True,
                "state_checkpoints_complete": True,
                "single_commit_per_step": True,
                "pair_replica_digest_match": True,
                "baseline_candidate_state_match": True,
                **_correctness_evidence(concurrency),
            })
    resource_stages = (
        "entry",
        "pre_correctness",
        "post_correctness",
        *(f"pre_epoch_{index}" for index in range(4)),
        *(f"post_launch_{index}" for index in range(4)),
        "pre_service_control",
        "post_service_control",
        "terminal",
    )
    resource_rows = [{
        **identity,
        "attempt_tag": "fixture-attempt",
        "stage": stage,
        "measurement_scope": "boundary",
        "run_label": None,
        "sample_index": None,
        "gpu_inventory": [{
            "gpu_index": rank,
            "gpu_uuid": f"GPU-{rank}",
            "memory_used_mib": 0,
            "utilization_percent": 0,
            "power_watts": 70.0 + rank,
            "compute_processes": [],
        } for rank in range(4)],
        "process_rows": [],
        "strict_clean": True,
        "identity_match": True,
        "foreign_processes": [],
    } for stage in resource_stages]
    resource_rows.extend({
        **identity,
        "attempt_tag": "fixture-attempt",
        "stage": f"runtime_{run_label}_0000",
        "measurement_scope": "runtime",
        "run_label": run_label,
        "sample_index": 0,
        "gpu_inventory": [{
            "gpu_index": rank,
            "gpu_uuid": f"GPU-{rank}",
            "memory_used_mib": 4096,
            "utilization_percent": 80,
            "power_watts": 250.0 + rank,
            "compute_processes": [{"pid": 1000 + rank}],
        } for rank in range(4)],
        "process_rows": [],
        "strict_clean": False,
        "identity_match": True,
        "foreign_processes": [],
    } for run_label in (
        "correctness",
        *(f"epoch_{index}" for index in range(4)),
        "service_control",
    ))
    service_rows = []
    for workload in ("Q0", "Q1", "Q2"):
        baseline = next(
            row
            for row in request_rows
            if row["epoch"] == 0
            and row["arm"] == "baseline"
            and row["workload_id"] == workload
            and row["repetition"] == 0
        )
        replicas = []
        for replica_index, pair_devices in enumerate(((0, 1), (2, 3))):
            replica_requests = copy.deepcopy(
                baseline["requests"][replica_index::2]
            )
            replica_makespan_ns = (
                max(row["completion_ns"] for row in replica_requests)
                - min(row["admitted_ns"] for row in replica_requests)
            )
            replicas.append({
                "replica_index": replica_index,
                "pair_devices": list(pair_devices),
                "replica_tensor_parallel_size": 2,
                "request_set_digest": hashlib.sha256(
                    json.dumps(
                        replica_requests,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "requests": replica_requests,
                "memory": [{
                    "rank": rank,
                    "cuda_peak_allocated_bytes": 35 * 1024**3,
                    "cuda_peak_reserved_bytes": 36 * 1024**3,
                    "physical_memory_bytes": 80 * 1024**3,
                } for rank in range(2)],
                "request_count": len(replica_requests),
                "output_token_count": len(replica_requests) * 128,
                "cohort_makespan_ns": replica_makespan_ns,
                "request_qps": (
                    len(replica_requests) * 1e9 / replica_makespan_ns
                ),
                "output_tokens_per_second": (
                    len(replica_requests) * 128 * 1e9
                    / replica_makespan_ns
                ),
                "ttft_ns": {
                    "p50": nearest_rank_percentile(
                        [row["ttft_ns"] for row in replica_requests],
                        50,
                    ),
                    "p95": nearest_rank_percentile(
                        [row["ttft_ns"] for row in replica_requests],
                        95,
                    ),
                    "p99": nearest_rank_percentile(
                        [row["ttft_ns"] for row in replica_requests],
                        99,
                    ),
                },
                "tpot_ns": {
                    "p50": nearest_rank_percentile(
                        [row["tpot_ns"] for row in replica_requests],
                        50,
                    ),
                    "p95": nearest_rank_percentile(
                        [row["tpot_ns"] for row in replica_requests],
                        95,
                    ),
                    "p99": nearest_rank_percentile(
                        [row["tpot_ns"] for row in replica_requests],
                        99,
                    ),
                },
                "peak_memory_by_gpu": [{
                    "gpu_index": gpu_index,
                    "peak_allocated_bytes": 35 * 1024**3,
                    "peak_reserved_bytes": 36 * 1024**3,
                    "physical_memory_bytes": 80 * 1024**3,
                } for gpu_index in pair_devices],
            })
        makespan_ns = (
            max(row["completion_ns"] for row in baseline["requests"])
            - min(row["admitted_ns"] for row in baseline["requests"])
        )
        ttfts = [row["ttft_ns"] for row in baseline["requests"]]
        tpots = [row["tpot_ns"] for row in baseline["requests"]]
        service_rows.append({
            **identity,
            "workload_id": workload,
            "arm": "TP2_X2_SERVICE_CONTROL",
            "classification_authority": False,
            "request_set_digest": baseline["request_set_digest"],
            "requests": [
                copy.deepcopy(request)
                for replica in replicas
                for request in replica["requests"]
            ],
            "replicas": replicas,
            "request_count": len(baseline["requests"]),
            "output_token_count": len(baseline["requests"]) * 128,
            "cohort_makespan_ns": makespan_ns,
            "request_qps": len(baseline["requests"]) * 1e9 / makespan_ns,
            "output_tokens_per_second": (
                len(baseline["requests"]) * 128 * 1e9 / makespan_ns
            ),
            "ttft_ns": {
                "p50": nearest_rank_percentile(ttfts, 50),
                "p95": nearest_rank_percentile(ttfts, 95),
                "p99": nearest_rank_percentile(ttfts, 99),
            },
            "tpot_ns": {
                "p50": nearest_rank_percentile(tpots, 50),
                "p95": nearest_rank_percentile(tpots, 95),
                "p99": nearest_rank_percentile(tpots, 99),
            },
            "replica_balance": {
                "request_counts": [
                    replica["request_count"] for replica in replicas
                ],
                "request_qps_max_to_min": (
                    max(replica["request_qps"] for replica in replicas)
                    / min(replica["request_qps"] for replica in replicas)
                ),
                "output_tokens_per_second_max_to_min": (
                    max(
                        replica["output_tokens_per_second"]
                        for replica in replicas
                    )
                    / min(
                        replica["output_tokens_per_second"]
                        for replica in replicas
                    )
                ),
            },
            "global_tp4_baseline_output_parity": True,
        })
    for name, rows in {
        "migration_rows.jsonl": migration_rows,
        "correctness_rows.jsonl": correctness_rows,
        "request_rows.jsonl": request_rows,
        "scheduler_step_rows.jsonl": scheduler_rows,
        "candidate_hit_rows.jsonl": candidate_rows,
        "collective_rows.jsonl": collective_rows,
        "memory_rows.jsonl": memory_rows,
        "resource_rows.jsonl": resource_rows,
        "service_control_rows.jsonl": service_rows,
    }.items():
        _write_jsonl(raw / name, rows)
    return root


def test_assemble_attempt_reconstructs_go_and_exact_inventory(tmp_path):
    attempt = passing_attempt(tmp_path / "attempt")
    output = tmp_path / "bundle"

    result = assemble_attempt(attempt, output)

    assert result["classification"] == (
        "GO_TOPOLOGY_LOCAL_TP2_WHOLE_MODEL_GATE"
    )
    assert result["aggregate_median_tpot_improvement_percent"] >= 5.0
    assert result["raw_gap_p99_authority"] is True
    assert result["measurement_complete"] is True
    assert result["workloads"]["P0"]["baseline"]["queueing_ns"]["p50"] == 500
    assert (
        result["workloads"]["P0"]["baseline"][
            "scheduler_step_duration_ns"
        ]["p50"]
        == 80
    )
    assert (
        result["workloads"]["P0"]["baseline"]["host_submission_ns"]["p50"]
        == 60
    )
    assert result["startup_model_load_ns"]["baseline"]["p50"] == 1_000_000
    assert result["cleanup_duration_ns"]["p50"] == 100
    assert result["gpu_utilization_percent"]["p50"] == 80
    assert result["gpu_power_watts"]["p50"] == 251.5
    report = (output / "report.md").read_text()
    assert "TP2 x2 service control" in report
    assert "request QPS" in report
    assert "output tokens/s" in report
    assert "TTFT P50/P95/P99" in report
    assert "TPOT P50/P95/P99" in report
    assert "replica balance" in report
    assert "peak memory by GPU" in report
    assert "global TP4 output parity" in report
    assert "Primary A/B benefit and protected metrics" in report
    assert "Measured mechanism and memory cost" in report
    assert "Queueing P50/P95/P99" in report
    assert "Scheduler-step P50/P95/P99" in report
    assert "Host-submission P50/P99" in report
    assert "Startup/model-load duration" in report
    assert "Runtime GPU utilization P50/P95/P99" in report
    assert "Runtime GPU power P50/P95/P99" in report
    assert "Cleanup duration P50/P95/P99" in report
    assert "Production-default enablement: prohibited" in report
    for workload in WORKLOADS:
        assert f"| {workload} |" in report
    assert {
        path.name for path in output.iterdir()
    } == set(CORE_ARTIFACTS) | {
        "classification.json",
        "report.md",
        "manifest.json",
        "manifest.sha256",
    }


def test_assemble_rejects_boundary_only_gpu_telemetry(tmp_path):
    attempt = passing_attempt(tmp_path / "attempt")
    path = attempt / "raw" / "resource_rows.jsonl"
    rows = [
        json.loads(line) for line in path.read_text().splitlines()
    ]
    _write_jsonl(
        path,
        [
            row
            for row in rows
            if row["measurement_scope"] == "boundary"
        ],
    )

    with pytest.raises(
        ValueError,
        match="runtime resource sample inventory",
    ):
        assemble_attempt(attempt, tmp_path / "bundle")


def test_assemble_runtime_metrics_ignore_unselected_host_gpus(tmp_path):
    attempt = passing_attempt(tmp_path / "attempt")
    path = attempt / "raw" / "resource_rows.jsonl"
    rows = [
        json.loads(line) for line in path.read_text().splitlines()
    ]
    for row in rows:
        if row["measurement_scope"] == "runtime":
            row["gpu_inventory"].append({
                "gpu_index": 7,
                "gpu_uuid": "GPU-unselected",
                "memory_used_mib": 8192,
                "utilization_percent": 100,
                "power_watts": 999.0,
                "compute_processes": [{"pid": 9999}],
            })
    _write_jsonl(path, rows)

    result = assemble_attempt(attempt, tmp_path / "bundle")

    assert result["gpu_utilization_percent"]["p50"] == 80
    assert result["gpu_power_watts"]["p50"] == 251.5


def test_service_control_output_mismatch_is_reported_but_non_authoritative(
    tmp_path,
):
    attempt = passing_attempt(tmp_path / "attempt")
    path = attempt / "raw" / "service_control_rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["replicas"][0]["requests"][0]["output_token_ids"][0] = 999
    rows[0]["requests"][0]["output_token_ids"][0] = 999
    rows[0]["global_tp4_baseline_output_parity"] = False
    _write_jsonl(path, rows)

    result = assemble_attempt(attempt, tmp_path / "bundle")

    assert result["classification"] == (
        "GO_TOPOLOGY_LOCAL_TP2_WHOLE_MODEL_GATE"
    )
    assert (
        result["service_control"]["Q0"][
            "global_tp4_baseline_output_parity"
        ]
        is False
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("epoch_arm", "A/B/B/A"),
        ("epoch_identity", "epoch"),
        ("epoch_workload_order", "epoch"),
        ("environment_incomplete", "environment"),
        ("feature_contract", "feature"),
        ("topology_matching", "topology"),
        ("rank_mapping_extra", "rank mapping"),
        ("rank_mapping_order", "rank mapping"),
        ("source_revision_charset", "source archive"),
        ("source_tree_charset", "source archive"),
        ("row_count", "ten baseline"),
        ("request_digest", "request-set digest"),
        ("request_concurrency", "concurrency"),
        ("request_timing_derivation", "timing"),
        ("token_gaps", "127 token gaps"),
        ("candidate_hits", "candidate hit"),
        ("candidate_source", "candidate evidence"),
        ("short_chunk", "short-chunk"),
        ("service_authority", "service-control"),
        ("service_identity", "service-control"),
        ("service_metric", "service-control"),
        ("service_output_parity", "service-control"),
        ("service_memory", "service-control"),
        ("nonfinite", "finite"),
        ("migration_break_even", "migration evidence"),
        ("migration_identity", "migration evidence"),
        ("duplicate_identity", "duplicate"),
        ("correctness_proof", "correctness"),
        ("correctness_top_logit", "correctness"),
        ("correctness_checkpoint", "correctness"),
        ("correctness_boolean_commit", "correctness"),
        ("correctness_source", "correctness"),
        ("correctness_layer_inventory", "correctness"),
        ("correctness_baseline_state", "correctness"),
        ("correctness_cohort", "correctness"),
        ("timing_correctness_proof", "timing correctness"),
        ("decoded_text", "decoded text"),
        ("resource_stage", "resource sample"),
        ("resource_scope", "resource sample"),
        ("resource_runtime_telemetry", "resource sample"),
        ("resource_source", "resource sample"),
        ("resource_raw_identity", "resource sample"),
        ("memory_physical_capacity", "memory evidence"),
        ("memory_identity", "memory evidence"),
        ("path_escape", "approved remote root"),
        ("source_archive", "source archive"),
        ("foreign_action", "foreign process"),
        ("cleanup", "cleanup"),
        ("cleanup_raw_receipt", "cleanup"),
        ("cleanup_raw_inventory", "cleanup"),
    ),
)
def test_assemble_attempt_rejects_invalid_evidence(
    tmp_path,
    mutation,
    message,
):
    attempt = passing_attempt(tmp_path / mutation)
    raw = attempt / "raw"
    if mutation == "epoch_arm":
        payload = json.loads(
            (raw / "campaign_epoch_manifest.json").read_text()
        )
        payload["epochs"][1]["arm"] = "baseline"
        _write_json(raw / "campaign_epoch_manifest.json", payload)
    elif mutation == "epoch_identity":
        payload = json.loads(
            (raw / "campaign_epoch_manifest.json").read_text()
        )
        payload["epochs"][1]["epoch"] = 0
        _write_json(raw / "campaign_epoch_manifest.json", payload)
    elif mutation == "epoch_workload_order":
        payload = json.loads(
            (raw / "campaign_epoch_manifest.json").read_text()
        )
        payload["epochs"][1]["workload_order"] = list(WORKLOADS)
        _write_json(raw / "campaign_epoch_manifest.json", payload)
    elif mutation == "environment_incomplete":
        payload = json.loads(
            (raw / "environment_manifest.json").read_text()
        )
        payload["environment_complete"] = False
        _write_json(raw / "environment_manifest.json", payload)
    elif mutation == "feature_contract":
        payload = json.loads((raw / "feature_contract.json").read_text())
        payload["default_off"] = False
        _write_json(raw / "feature_contract.json", payload)
    elif mutation == "topology_matching":
        payload = json.loads((raw / "gpu_topology.json").read_text())
        for row in payload["rows"]:
            pair = tuple(sorted((row["left_rank"], row["right_rank"])))
            row["link"] = (
                "PIX" if pair in {(0, 2), (1, 3)} else "SYS"
            )
        _write_json(raw / "gpu_topology.json", payload)
    elif mutation == "rank_mapping_extra":
        payload = json.loads((raw / "gpu_rank_manifest.json").read_text())
        payload["mapping"].append({})
        _write_json(raw / "gpu_rank_manifest.json", payload)
    elif mutation == "rank_mapping_order":
        payload = json.loads((raw / "gpu_rank_manifest.json").read_text())
        payload["mapping"].reverse()
        _write_json(raw / "gpu_rank_manifest.json", payload)
    elif mutation == "source_revision_charset":
        payload = json.loads((raw / "source_manifest.json").read_text())
        payload["source_revision"] = "z" * 40
        _write_json(raw / "source_manifest.json", payload)
    elif mutation == "source_tree_charset":
        payload = json.loads((raw / "source_manifest.json").read_text())
        payload["source_tree_sha256"] = "z" * 64
        _write_json(raw / "source_manifest.json", payload)
    elif mutation == "row_count":
        rows = [
            json.loads(line)
            for line in (raw / "request_rows.jsonl").read_text().splitlines()
        ]
        _write_jsonl(raw / "request_rows.jsonl", rows[:-1])
    elif mutation == "request_digest":
        rows = [
            json.loads(line)
            for line in (raw / "request_rows.jsonl").read_text().splitlines()
        ]
        rows[0]["request_set_digest"] = "f" * 64
        _write_jsonl(raw / "request_rows.jsonl", rows)
    elif mutation == "request_concurrency":
        path = raw / "request_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["workload_id"] == "Q1" and row["repetition"] == 0:
                row["requests"].pop()
                replay = row["timing_correctness_replay"]
                replay["requests"].pop()
                for step_rows in replay["step_proofs"]:
                    for rank_row in step_rows:
                        rank_row["sequence_ids"].pop()
                        rank_row["token_ids"].pop()
                        rank_row["top_logit_values"].pop()
        _write_jsonl(path, rows)
    elif mutation == "request_timing_derivation":
        path = raw / "request_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["requests"][0]["tpot_ns"] += 1
        _write_jsonl(path, rows)
    elif mutation == "token_gaps":
        rows = [
            json.loads(line)
            for line in (raw / "request_rows.jsonl").read_text().splitlines()
        ]
        rows[0]["requests"][0]["token_gaps_ns"].pop()
        _write_jsonl(raw / "request_rows.jsonl", rows)
    elif mutation == "candidate_hits":
        rows = [
            json.loads(line)
            for line in (
                raw / "candidate_hit_rows.jsonl"
            ).read_text().splitlines()
        ]
        rows[0]["tp2_decode_calls"] -= 1
        _write_jsonl(raw / "candidate_hit_rows.jsonl", rows)
    elif mutation == "candidate_source":
        path = raw / "candidate_hit_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["source_revision"] = "f" * 40
        _write_jsonl(path, rows)
    elif mutation == "short_chunk":
        rows = [
            json.loads(line)
            for line in (
                raw / "candidate_hit_rows.jsonl"
            ).read_text().splitlines()
        ]
        rows[0]["short_chunk_calls"] = 1
        _write_jsonl(raw / "candidate_hit_rows.jsonl", rows)
    elif mutation == "service_authority":
        rows = [
            json.loads(line)
            for line in (
                raw / "service_control_rows.jsonl"
            ).read_text().splitlines()
        ]
        rows[0]["classification_authority"] = True
        _write_jsonl(raw / "service_control_rows.jsonl", rows)
    elif mutation == "service_identity":
        path = raw / "service_control_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows.append(copy.deepcopy(rows[0]))
        _write_jsonl(path, rows)
    elif mutation == "service_metric":
        path = raw / "service_control_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["request_qps"] += 1
        _write_jsonl(path, rows)
    elif mutation == "service_output_parity":
        path = raw / "service_control_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["replicas"][0]["requests"][0]["output_token_ids"][0] = 999
        _write_jsonl(path, rows)
    elif mutation == "service_memory":
        path = raw / "service_control_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        memory = rows[0]["replicas"][0]["peak_memory_by_gpu"][0]
        memory["peak_reserved_bytes"] = (
            memory["physical_memory_bytes"] + 1
        )
        _write_jsonl(path, rows)
    elif mutation == "nonfinite":
        path = raw / "migration_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["latency_ns"] = float("nan")
        _write_jsonl(path, rows)
    elif mutation == "migration_break_even":
        path = raw / "migration_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["break_even_output_tokens"] = 0.0
        _write_jsonl(path, rows)
    elif mutation == "migration_identity":
        path = raw / "migration_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["epoch"] = rows[0]["epoch"]
        rows[1]["workload_id"] = rows[0]["workload_id"]
        rows[1]["repetition"] = rows[0]["repetition"]
        _write_jsonl(path, rows)
    elif mutation == "duplicate_identity":
        path = raw / "request_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows.append(copy.deepcopy(rows[0]))
        _write_jsonl(path, rows)
    elif mutation == "correctness_proof":
        path = raw / "correctness_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["candidate_step_proofs"][10][3]["token_ids"][0] = 999
        _write_jsonl(path, rows)
    elif mutation == "correctness_top_logit":
        path = raw / "correctness_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["candidate_step_proofs"][10][3][
            "top_logit_values"
        ][0] += 1.0
        _write_jsonl(path, rows)
    elif mutation == "correctness_checkpoint":
        path = raw / "correctness_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["candidate_state_checkpoints"]["token_32"][2][
            "runtime_snapshot"
        ]["state"]["commit_count"] += 1
        _write_jsonl(path, rows)
    elif mutation == "correctness_boolean_commit":
        path = raw / "correctness_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["candidate_state_checkpoints"]["post_migration"][0][
            "runtime_snapshot"
        ]["state"]["commit_count"] = True
        _write_jsonl(path, rows)
    elif mutation == "correctness_source":
        path = raw / "correctness_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["source_revision"] = "f" * 40
        _write_jsonl(path, rows)
    elif mutation == "correctness_layer_inventory":
        path = raw / "correctness_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for rank_row in rows[0]["candidate_state_checkpoints"]["token_32"]:
            rank_row["output_digests"][0]["layer_index"] = 999
            rank_row["state_digests"][0]["layer_index"] = 999
        _write_jsonl(path, rows)
    elif mutation == "correctness_baseline_state":
        path = raw / "correctness_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["baseline_state_checkpoints"]["token_32"][2][
            "canonical_state_components"
        ][0]["recurrent_sha256"] = "f" * 64
        _write_jsonl(path, rows)
    elif mutation == "correctness_cohort":
        path = raw / "correctness_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["candidate_state_checkpoints"]["token_32"][0][
            "cohort"
        ][0]["generation"] += 1
        _write_jsonl(path, rows)
    elif mutation == "timing_correctness_proof":
        path = raw / "request_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["timing_correctness_replay"]["step_proofs"][10][3][
            "token_ids"
        ][0] = 999
        _write_jsonl(path, rows)
    elif mutation == "decoded_text":
        path = raw / "request_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["requests"][0]["decoded_text"] += "tampered"
        _write_jsonl(path, rows)
    elif mutation == "resource_stage":
        path = raw / "resource_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        _write_jsonl(path, rows[:-1])
    elif mutation == "resource_scope":
        path = raw / "resource_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        runtime = next(
            row for row in rows
            if row["measurement_scope"] == "runtime"
        )
        runtime["measurement_scope"] = "boundary"
        _write_jsonl(path, rows)
    elif mutation == "resource_runtime_telemetry":
        path = raw / "resource_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        runtime = next(
            row for row in rows
            if row["measurement_scope"] == "runtime"
        )
        runtime["gpu_inventory"][0]["power_watts"] = -1
        runtime["strict_clean"] = True
        _write_jsonl(path, rows)
    elif mutation == "resource_source":
        path = raw / "resource_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["source_revision"] = "f" * 40
        _write_jsonl(path, rows)
    elif mutation == "resource_raw_identity":
        path = raw / "resource_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["gpu_inventory"][0]["gpu_uuid"] = "GPU-drift"
        _write_jsonl(path, rows)
    elif mutation == "memory_physical_capacity":
        path = raw / "memory_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["physical_memory_bytes"] = 0
        _write_jsonl(path, rows)
    elif mutation == "memory_identity":
        path = raw / "memory_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["epoch"] = rows[0]["epoch"]
        rows[1]["arm"] = rows[0]["arm"]
        rows[1]["rank"] = rows[0]["rank"]
        _write_jsonl(path, rows)
    elif mutation == "path_escape":
        payload = json.loads((raw / "cleanup.json").read_text())
        payload["task_paths"].append("/tmp/escaped")
        _write_json(raw / "cleanup.json", payload)
    elif mutation == "source_archive":
        payload = json.loads((raw / "source_manifest.json").read_text())
        payload["source_archive_complete"] = False
        _write_json(raw / "source_manifest.json", payload)
    elif mutation == "foreign_action":
        payload = json.loads((raw / "cleanup.json").read_text())
        payload["foreign_process_actions"] = [999]
        _write_json(raw / "cleanup.json", payload)
    elif mutation == "cleanup":
        payload = json.loads((raw / "cleanup.json").read_text())
        payload["retained_leases"] = 1
        _write_json(raw / "cleanup.json", payload)
    elif mutation == "cleanup_raw_receipt":
        payload = json.loads((raw / "cleanup.json").read_text())
        payload["worker_cleanup_receipts"][1]["receipt"][
            "rank_cleanup_receipts"
        ][0]["qwen38_topology_local_tp2_cleanup"][
            "candidate_state_released"
        ] = False
        _write_json(raw / "cleanup.json", payload)
    elif mutation == "cleanup_raw_inventory":
        payload = json.loads((raw / "cleanup.json").read_text())
        payload["worker_cleanup_receipts"].pop()
        _write_json(raw / "cleanup.json", payload)

    with pytest.raises((ValueError, RuntimeError), match=message):
        assemble_attempt(attempt, tmp_path / f"{mutation}-bundle")


def test_assemble_rejects_missing_or_extra_raw_file(tmp_path):
    missing = passing_attempt(tmp_path / "missing")
    (missing / "raw" / "cleanup.json").unlink()
    with pytest.raises(ValueError, match="artifact inventory"):
        assemble_attempt(missing, tmp_path / "missing-bundle")

    extra = passing_attempt(tmp_path / "extra")
    _write_json(extra / "raw" / "unexpected.json", {})
    with pytest.raises(ValueError, match="artifact inventory"):
        assemble_attempt(extra, tmp_path / "extra-bundle")


def test_nearest_rank_rejects_nonfinite():
    with pytest.raises(ValueError, match="finite"):
        nearest_rank_percentile([1.0, math.inf], 99)
