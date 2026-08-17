from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = (
    ROOT / "tools" / "verify_spec_verify_cuda_graph_gate.py"
)


def _load_verifier():
    assert VERIFIER_PATH.is_file(), (
        f"missing verifier: {VERIFIER_PATH}"
    )
    spec = importlib.util.spec_from_file_location(
        "verify_spec_verify_cuda_graph_gate_test_module",
        VERIFIER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _family(batch_size: int, query_len: int, width: int) -> dict:
    family_id = f"b{batch_size}-q{query_len}-w{width}"
    logits_sha256 = _sha(f"{family_id}:logits")
    kv_sha256 = _sha(f"{family_id}:accepted-kv")
    target_tokens = [
        batch_size * 1000 + query_len * 100 + width + offset
        for offset in range(batch_size * query_len)
    ]
    final_tokens = [
        target_tokens[
            row * query_len:(row + 1) * query_len
        ]
        for row in range(batch_size)
    ]
    identity = {
        "active_batch_size": batch_size,
        "query_len": query_len,
        "page_table_width": width,
    }
    return {
        "family_id": family_id,
        "identity": identity,
        "eager_baseline": {
            "logits_sha256": logits_sha256,
            "target_tokens": target_tokens,
            "accepted_lengths": [query_len - 1] * batch_size,
            "final_tokens": final_tokens,
            "accepted_prefix_kv_sha256": kv_sha256,
            "target_forward_count": 1,
        },
        "warmed_graph": {
            "identity": dict(identity),
            "logits_sha256": logits_sha256,
            "target_tokens": list(target_tokens),
            "accepted_lengths": [query_len - 1] * batch_size,
            "final_tokens": copy.deepcopy(final_tokens),
            "accepted_prefix_kv_sha256": kv_sha256,
            "target_forward_count": 1,
            "eager_forward_count": 0,
            "graph_replay_count": 1,
            "warmed_latency_ns": 1000,
            "capture_latency_ns": 5000,
        },
        "transaction_results": {
            "accepted_prefix_kv_parity": True,
            "rejected_suffix_released": True,
            "transaction_states": ["committed"] * batch_size,
            "materialized_token_counts": [query_len] * batch_size,
            "committed_materialized_token_counts": [
                max(0, query_len - 2)
            ] * batch_size,
            "rejected_materialized_token_counts": [
                query_len - max(0, query_len - 2)
            ] * batch_size,
            "unused_block_ids": [],
            "released_block_ids": [],
            "all_unused_blocks_released": True,
        },
        "replay_failure_injection": {
            "graph_replay_count": 1,
            "eager_retry_count": 0,
            "error_propagated": True,
            "quarantine_reason": "replay_failed",
            "stable_quarantine_reason": True,
        },
    }


def _artifact() -> dict:
    batch_sizes = [1, 4]
    query_lengths = [1, 3]
    widths = [1, 2]
    families = [
        _family(batch_size, query_len, width)
        for batch_size in batch_sizes
        for query_len in query_lengths
        for width in widths
    ]
    return {
        "schema_version": 1,
        "source_sha256": _sha("source-tree"),
        "source_files": {
            "tinyvllm/engine/model_runner.py": _sha(
                "model-runner-source"
            ),
        },
        "model": "Qwen3-0.6B",
        "checkpoint": "/models/Qwen3-0.6B",
        "device_name": "NVIDIA A100-SXM4-80GB",
        "device_compute_capability": [8, 0],
        "torch_version": "2.4.1",
        "cuda_version": "12.4",
        "flash_attn_version": "2.7.4",
        "configuration": {
            "tensor_parallel_size": 1,
            "kv_offload_mvp0": False,
            "context_length": 4096,
            "batch_sizes": batch_sizes,
            "query_lengths": query_lengths,
            "page_table_widths": widths,
            "identity_policy": "exact_b_q_w",
            "capture_latency_excluded_from_warmed_hit": True,
        },
        "families": families,
        "eager_baseline": {
            "family_count": len(families),
        },
        "warmed_graph": {
            "family_count": len(families),
        },
        "replay_failure_injection": {
            "family_count": len(families),
        },
        "transaction_results": {
            "family_count": len(families),
        },
        "claims": {
            "kv_offload_benefit": False,
            "h2d_d2h_benefit": False,
        },
        "classification": "NOT_PROMOTABLE",
    }


def _performance_artifact() -> dict:
    artifact = _artifact()
    artifact["configuration"]["measure_performance"] = True
    family_ids = [
        family["family_id"]
        for family in artifact["families"]
    ]
    artifact["performance"] = {
        "warmup_count": 2,
        "measurement_count": len(family_ids) * 5,
        "prompt_lengths": {
            family_id: 256
            for family_id in family_ids
        },
        "proposal_length_distribution": {
            "1": 10,
            "3": 30,
        },
        "batch_distribution": {
            "1": 4,
            "4": 4,
        },
        "eager_baseline": {
            "ttft_ns": 20_000,
            "tpot_ns": 4_000,
            "throughput_tokens_per_second": 500.0,
            "gpu_allocated_bytes": 1_000_000,
            "gpu_reserved_bytes": 2_000_000,
        },
        "warmed_exact_graph_hits": {
            "measurement_count": len(family_ids) * 5,
            "latency_ns_by_family": {
                family_id: [1_000] * 5
                for family_id in family_ids
            },
            "gpu_allocated_bytes": 1_100_000,
            "gpu_reserved_bytes": 2_100_000,
        },
        "mixed_hit_rate": {
            "measurement_count": 56,
            "hit_count": 40,
            "miss_count": 16,
            "end_to_end_tpot_ns": 3_500,
            "ttft_ns": 21_000,
            "throughput_tokens_per_second": 550.0,
        },
        "capture": {
            "duration_ns_by_family": {
                family_id: 5_000
                for family_id in family_ids
            },
            "allocated_delta_bytes_by_family": {
                family_id: 10_000
                for family_id in family_ids
            },
            "reserved_delta_bytes_by_family": {
                family_id: 20_000
                for family_id in family_ids
            },
        },
        "cache_counts": {
            "hits": 40,
            "misses": 16,
            "evictions": 0,
            "quarantines": 0,
        },
        "acceptance": {
            "proposed_tokens": 40,
            "accepted_draft_tokens": 20,
            "acceptance_rate": 0.5,
        },
    }
    return artifact


def test_valid_exact_family_matrix_passes():
    result = _load_verifier().validate_artifact(_artifact())

    assert result["status"] == "PASS"
    assert result["classification"] == "NOT_PROMOTABLE"
    assert result["family_count"] == 8
    assert result["batch_sizes"] == [1, 4]
    assert result["query_lengths"] == [1, 3]
    assert result["page_table_widths"] == [1, 2]


def test_valid_performance_evidence_passes_separately():
    result = _load_verifier().validate_artifact(
        _performance_artifact()
    )

    assert result["status"] == "PASS"
    assert result["performance_evidence"] is True
    assert result["performance_measurement_count"] == 40


def test_rejects_non_mvp_performance_warmup_count():
    artifact = _performance_artifact()
    artifact["performance"]["warmup_count"] = 1

    with pytest.raises(ValueError, match="exactly 2"):
        _load_verifier().validate_artifact(artifact)


def test_rejects_non_mvp_warmed_measurement_count():
    artifact = _performance_artifact()
    artifact["performance"]["measurement_count"] = 8
    artifact["performance"]["warmed_exact_graph_hits"][
        "measurement_count"
    ] = 8

    with pytest.raises(ValueError, match="exactly 40"):
        _load_verifier().validate_artifact(artifact)


def test_rejects_non_mvp_warmed_measurements_per_family():
    artifact = _performance_artifact()
    artifact["performance"]["warmed_exact_graph_hits"][
        "latency_ns_by_family"
    ]["b1-q1-w1"] = [1_000] * 4

    with pytest.raises(ValueError, match="exactly 5"):
        _load_verifier().validate_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        (
            "proposal_length_distribution",
            {"1": 10, "3": 29},
            "proposal-length distribution counts",
        ),
        (
            "batch_distribution",
            {"1": 4, "4": 3},
            "batch distribution counts",
        ),
    ),
)
def test_rejects_non_mvp_performance_distributions(
    field,
    value,
    match,
):
    artifact = _performance_artifact()
    artifact["performance"][field] = value

    with pytest.raises(ValueError, match=match):
        _load_verifier().validate_artifact(artifact)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    (
        (
            ("performance", "warmup_count"),
            0,
            "warmup",
        ),
        (
            ("performance", "measurement_count"),
            0,
            "measurement",
        ),
        (
            (
                "performance",
                "warmed_exact_graph_hits",
                "latency_ns_by_family",
            ),
            {},
            "warmed",
        ),
        (
            (
                "performance",
                "mixed_hit_rate",
                "hit_count",
            ),
            0,
            "mixed",
        ),
        (
            (
                "performance",
                "capture",
                "reserved_delta_bytes_by_family",
            ),
            {},
            "capture",
        ),
        (
            (
                "performance",
                "acceptance",
                "acceptance_rate",
            ),
            1.5,
            "acceptance",
        ),
    ),
)
def test_rejects_partial_or_contradictory_performance_evidence(
    path,
    value,
    match,
):
    artifact = _performance_artifact()
    target = artifact
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=match):
        _load_verifier().validate_artifact(artifact)


def test_measure_performance_requires_performance_section():
    artifact = _artifact()
    artifact["configuration"]["measure_performance"] = True

    with pytest.raises(ValueError, match="performance"):
        _load_verifier().validate_artifact(artifact)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    (
        (
            ("configuration", "tensor_parallel_size"),
            2,
            "TP1",
        ),
        (
            ("configuration", "kv_offload_mvp0"),
            True,
            "KV offload",
        ),
        (
            ("configuration", "context_length"),
            2048,
            "4096",
        ),
        (
            ("configuration", "batch_sizes"),
            [1],
            "batch",
        ),
        (
            ("configuration", "query_lengths"),
            [],
            "query",
        ),
        (
            ("configuration", "query_lengths"),
            [1, 2],
            "exactly",
        ),
        (
            ("configuration", "page_table_widths"),
            [1],
            "width",
        ),
        (
            ("configuration", "page_table_widths"),
            [2, 3],
            "exactly",
        ),
        (
            ("configuration", "identity_policy"),
            "rounded",
            "exact",
        ),
        (
            (
                "configuration",
                "capture_latency_excluded_from_warmed_hit",
            ),
            False,
            "capture latency",
        ),
        (
            ("classification",),
            "PROMOTABLE",
            "NOT_PROMOTABLE",
        ),
        (
            ("claims", "kv_offload_benefit"),
            True,
            "KV offload",
        ),
        (
            ("claims", "h2d_d2h_benefit"),
            True,
            "H2D/D2H",
        ),
    ),
)
def test_rejects_unsupported_or_overclaimed_configuration(
    path,
    value,
    match,
):
    artifact = _artifact()
    target = artifact
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=match):
        _load_verifier().validate_artifact(artifact)


@pytest.mark.parametrize(
    ("section", "field", "value", "match"),
    (
        (
            "warmed_graph",
            "logits_sha256",
            _sha("different-logits"),
            "logits",
        ),
        (
            "warmed_graph",
            "target_tokens",
            [999],
            "target-token",
        ),
        (
            "warmed_graph",
            "accepted_lengths",
            [1],
            "accepted-length",
        ),
        (
            "warmed_graph",
            "final_tokens",
            [[999]],
            "final-token",
        ),
        (
            "warmed_graph",
            "accepted_prefix_kv_sha256",
            _sha("different-kv"),
            "accepted-prefix KV",
        ),
        (
            "transaction_results",
            "accepted_prefix_kv_parity",
            False,
            "accepted-prefix KV",
        ),
        (
            "transaction_results",
            "rejected_suffix_released",
            False,
            "rejected suffix",
        ),
        (
            "warmed_graph",
            "target_forward_count",
            2,
            "one target forward",
        ),
        (
            "warmed_graph",
            "eager_forward_count",
            1,
            "eager forward",
        ),
        (
            "warmed_graph",
            "graph_replay_count",
            0,
            "graph replay",
        ),
        (
            "replay_failure_injection",
            "eager_retry_count",
            1,
            "eager retry",
        ),
        (
            "replay_failure_injection",
            "error_propagated",
            False,
            "propagated",
        ),
        (
            "replay_failure_injection",
            "quarantine_reason",
            "other",
            "quarantine",
        ),
        (
            "replay_failure_injection",
            "stable_quarantine_reason",
            False,
            "stable quarantine",
        ),
    ),
)
def test_rejects_missing_family_correctness_or_failure_evidence(
    section,
    field,
    value,
    match,
):
    artifact = _artifact()
    artifact["families"][0][section][field] = value

    with pytest.raises(ValueError, match=match):
        _load_verifier().validate_artifact(artifact)


def test_rejects_missing_enabled_family_and_rounded_identity():
    artifact = _artifact()
    artifact["families"].pop()
    with pytest.raises(ValueError, match="family matrix"):
        _load_verifier().validate_artifact(artifact)

    artifact = _artifact()
    artifact["families"][0]["warmed_graph"]["identity"][
        "page_table_width"
    ] = 2
    with pytest.raises(ValueError, match="exact identity"):
        _load_verifier().validate_artifact(artifact)


def test_allows_rejected_suffix_without_independent_released_block():
    artifact = _artifact()
    transaction = artifact["families"][0]["transaction_results"]

    assert transaction["rejected_materialized_token_counts"] == [1]
    assert transaction["unused_block_ids"] == []
    assert transaction["released_block_ids"] == []
    assert (
        _load_verifier().validate_artifact(artifact)["status"]
        == "PASS"
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        (
            "transaction_states",
            ["materialized"],
            "transaction state",
        ),
        (
            "rejected_materialized_token_counts",
            [0],
            "rejected materialized",
        ),
        (
            "committed_materialized_token_counts",
            [1],
            "materialized token count",
        ),
        (
            "released_block_ids",
            [123],
            "unused block release",
        ),
        (
            "all_unused_blocks_released",
            False,
            "unused blocks",
        ),
    ),
)
def test_rejects_contradictory_rejected_suffix_evidence(
    field,
    value,
    match,
):
    artifact = _artifact()
    artifact["families"][0]["transaction_results"][field] = value

    with pytest.raises(ValueError, match=match):
        _load_verifier().validate_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("source_sha256", "", "source"),
        ("model", "", "model"),
        ("checkpoint", "", "checkpoint"),
        ("device_name", "", "device"),
        ("device_compute_capability", [8], "compute capability"),
        ("torch_version", "", "torch"),
        ("cuda_version", "", "CUDA"),
        ("flash_attn_version", "", "FlashAttention"),
    ),
)
def test_rejects_missing_source_config_or_device_identity(
    field,
    value,
    match,
):
    artifact = _artifact()
    artifact[field] = value

    with pytest.raises(ValueError, match=match):
        _load_verifier().validate_artifact(artifact)


def test_verify_artifact_rejects_stale_source_hash(tmp_path):
    source_path = tmp_path / "tinyvllm" / "engine"
    source_path.mkdir(parents=True)
    model_runner_path = source_path / "model_runner.py"
    model_runner_path.write_text(
        "current source\n",
        encoding="utf-8",
    )
    artifact = _artifact()
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source hash"):
        _load_verifier().verify_artifact(
            artifact_path,
            tmp_path,
        )
