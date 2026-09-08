from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from tools.assemble_qwen38_topology_local_tp2_whole_model import (
    CORE_ARTIFACTS,
    assemble_attempt,
    classify,
    nearest_rank_percentile,
)


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


def passing_attempt(root: Path) -> Path:
    raw = root / "raw"
    source_revision = "a" * 40
    model_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    identity = {
        "source_revision": source_revision,
        "model_revision": model_revision,
    }
    json_payloads = {
        "source_manifest.json": {
            **identity,
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
        },
        "gpu_rank_manifest.json": {
            **identity,
            "ranks": list(range(4)),
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
                {"epoch": epoch, "arm": arm}
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
                base_tpot = 100.0 + workload_index
                tpot = base_tpot if arm == "baseline" else base_tpot * 0.94
                requests = []
                for request_index in range(concurrency):
                    requests.append({
                        "request_id": f"{workload}-{repetition}-{request_index}",
                        "output_token_ids": list(range(128)),
                        "token_gaps_ns": [tpot] * 127,
                        "ttft_ns": 1_000.0,
                        "tpot_ns": tpot,
                        "e2e_ns": 1_000.0 + tpot * 127,
                    })
                row = {
                    **identity,
                    "epoch": epoch,
                    "arm": arm,
                    "workload_id": workload,
                    "repetition": repetition,
                    "request_set_digest": digest,
                    "requests": requests,
                    "cohort_makespan_ns": 2_000_000.0,
                }
                request_rows.append(row)
                if arm == "candidate":
                    expected_segments = concurrency * 127
                    scheduler_rows.append({
                        **identity,
                        "epoch": epoch,
                        "workload_id": workload,
                        "repetition": repetition,
                        "request_set_digest": digest,
                        "decode_steps": 127,
                        "token_one_segments": expected_segments,
                    })
                    candidate_rows.append({
                        **identity,
                        "epoch": epoch,
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
                        "workload_id": workload,
                        "repetition": repetition,
                        "request_set_digest": digest,
                        "latency_ns": 10_000.0,
                        "break_even_output_tokens": 8.0,
                        "temporary_live_tensors": 0,
                    })
                    collective_rows.append({
                        **identity,
                        "epoch": epoch,
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
                "physical_memory_bytes": 80 * 1024**3,
            })

    correctness_rows = [{
        **identity,
        "workload_id": workload,
        "repetition": repetition,
        "output_tokens_match": True,
        "rank_token_agreement": True,
        "finite_logits": True,
        "state_checkpoints_complete": True,
        "single_commit_per_step": True,
        "pair_replica_digest_match": True,
    } for workload in WORKLOADS for repetition in range(5)]
    resource_stages = (
        "entry",
        *(f"pre_epoch_{index}" for index in range(4)),
        *(f"post_launch_{index}" for index in range(4)),
        "pre_service_control",
        "post_service_control",
        "terminal",
    )
    resource_rows = [{
        **identity,
        "stage": stage,
        "strict_clean": True,
        "identity_match": True,
        "foreign_processes": [],
    } for stage in resource_stages]
    service_rows = [{
        **identity,
        "workload_id": workload,
        "arm": "TP2_X2_SERVICE_CONTROL",
        "classification_authority": False,
    } for workload in ("Q0", "Q1", "Q2")]
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
    assert {
        path.name for path in output.iterdir()
    } == set(CORE_ARTIFACTS) | {
        "classification.json",
        "report.md",
        "manifest.json",
        "manifest.sha256",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("epoch_arm", "A/B/B/A"),
        ("row_count", "ten baseline"),
        ("request_digest", "request-set digest"),
        ("token_gaps", "127 token gaps"),
        ("candidate_hits", "candidate hit"),
        ("short_chunk", "short-chunk"),
        ("service_authority", "service-control"),
        ("nonfinite", "finite"),
        ("duplicate_identity", "duplicate"),
        ("resource_stage", "resource sample"),
        ("path_escape", "approved remote root"),
        ("source_archive", "source archive"),
        ("foreign_action", "foreign process"),
        ("cleanup", "cleanup"),
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
    elif mutation == "nonfinite":
        path = raw / "migration_rows.jsonl"
        path.write_text(path.read_text().replace("10000.0", "NaN", 1))
    elif mutation == "duplicate_identity":
        path = raw / "request_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows.append(copy.deepcopy(rows[0]))
        _write_jsonl(path, rows)
    elif mutation == "resource_stage":
        path = raw / "resource_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        _write_jsonl(path, rows[:-1])
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
