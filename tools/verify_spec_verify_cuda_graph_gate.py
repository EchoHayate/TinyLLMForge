from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


SCHEMA_VERSION = 1
CLASSIFICATION = "NOT_PROMOTABLE"
REQUIRED_BATCH_SIZES = (1, 4)
REQUIRED_QUERY_LENGTHS = (1, 3)
REQUIRED_PAGE_TABLE_WIDTHS = (1, 2)
REQUIRED_PERFORMANCE_WARMUP_COUNT = 2
REQUIRED_PERFORMANCE_MEASUREMENTS_PER_FAMILY = 5


def _require_dict(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dictionary")
    return value


def _require_list(value, name: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _require_string(value, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value, name: str) -> str:
    value = _require_string(value, name)
    if len(value) != 64:
        raise ValueError(f"{name} must be a SHA256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a SHA256 digest"
        ) from exc
    return value


def _require_int(value, name: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _require_bool(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_number(
    value,
    name: str,
    *,
    minimum: float = 0.0,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ValueError(
            f"{name} must be a finite number >= {minimum}"
        )
    return float(value)


def _require_count_list(
    value,
    name: str,
    *,
    expected_length: int,
) -> list[int]:
    rows = _require_list(value, name)
    if len(rows) != expected_length:
        raise ValueError(
            f"{name} must contain {expected_length} rows"
        )
    return [
        _require_int(row, name)
        for row in rows
    ]


def _require_block_id_list(value, name: str) -> list[int]:
    rows = _require_list(value, name)
    normalized = [
        _require_int(row, name)
        for row in rows
    ]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique block IDs")
    return normalized


def _require_exact_int_list(
    value,
    name: str,
    *,
    minimum: int = 1,
) -> list[int]:
    rows = _require_list(value, name)
    if not rows:
        raise ValueError(f"{name} must be non-empty")
    normalized = [
        _require_int(row, name, minimum=minimum)
        for row in rows
    ]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique values")
    if normalized != sorted(normalized):
        raise ValueError(f"{name} must be sorted")
    return normalized


def _identity_tuple(identity, name: str) -> tuple[int, int, int]:
    identity = _require_dict(identity, name)
    expected_fields = {
        "active_batch_size",
        "query_len",
        "page_table_width",
    }
    if set(identity) != expected_fields:
        raise ValueError(
            f"{name} must contain exact B/Q/W identity"
        )
    return (
        _require_int(
            identity["active_batch_size"],
            f"{name}.active_batch_size",
            minimum=1,
        ),
        _require_int(
            identity["query_len"],
            f"{name}.query_len",
            minimum=1,
        ),
        _require_int(
            identity["page_table_width"],
            f"{name}.page_table_width",
            minimum=1,
        ),
    )


def _validate_baseline(
    baseline,
    *,
    name: str,
    batch_size: int,
    query_len: int,
) -> dict:
    baseline = _require_dict(baseline, name)
    _require_sha256(
        baseline.get("logits_sha256"),
        f"{name}.logits_sha256",
    )
    target_tokens = _require_list(
        baseline.get("target_tokens"),
        f"{name}.target_tokens",
    )
    if (
        len(target_tokens)
        != batch_size * query_len
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            for token in target_tokens
        )
    ):
        raise ValueError(
            f"{name} target-token evidence is invalid"
        )
    accepted_lengths = _require_list(
        baseline.get("accepted_lengths"),
        f"{name}.accepted_lengths",
    )
    if (
        len(accepted_lengths) != batch_size
        or any(
            isinstance(length, bool)
            or not isinstance(length, int)
            or length < 0
            or length > query_len
            for length in accepted_lengths
        )
    ):
        raise ValueError(
            f"{name} accepted-length evidence is invalid"
        )
    final_tokens = _require_list(
        baseline.get("final_tokens"),
        f"{name}.final_tokens",
    )
    if (
        len(final_tokens) != batch_size
        or any(not isinstance(row, list) for row in final_tokens)
    ):
        raise ValueError(
            f"{name} final-token evidence is invalid"
        )
    _require_sha256(
        baseline.get("accepted_prefix_kv_sha256"),
        f"{name}.accepted-prefix KV",
    )
    if (
        _require_int(
            baseline.get("target_forward_count"),
            f"{name}.target_forward_count",
        )
        != 1
    ):
        raise ValueError(
            f"{name} must record one target forward"
        )
    return baseline


def _validate_family(
    family,
    *,
    enabled_batches: set[int],
    enabled_queries: set[int],
    enabled_widths: set[int],
) -> tuple[int, int, int]:
    family = _require_dict(family, "family")
    batch_size, query_len, width = _identity_tuple(
        family.get("identity"),
        "family.identity",
    )
    identity = (batch_size, query_len, width)
    if (
        batch_size not in enabled_batches
        or query_len not in enabled_queries
        or width not in enabled_widths
    ):
        raise ValueError(
            "family identity is outside the enabled exact matrix"
        )
    expected_family_id = (
        f"b{batch_size}-q{query_len}-w{width}"
    )
    if family.get("family_id") != expected_family_id:
        raise ValueError("family_id does not match exact identity")

    eager = _validate_baseline(
        family.get("eager_baseline"),
        name=f"{expected_family_id}.eager_baseline",
        batch_size=batch_size,
        query_len=query_len,
    )
    warmed = _validate_baseline(
        family.get("warmed_graph"),
        name=f"{expected_family_id}.warmed_graph",
        batch_size=batch_size,
        query_len=query_len,
    )
    if _identity_tuple(
        warmed.get("identity"),
        f"{expected_family_id}.warmed_graph.identity",
    ) != identity:
        raise ValueError(
            "warmed graph must use the exact identity"
        )
    if warmed["logits_sha256"] != eager["logits_sha256"]:
        raise ValueError(
            "eager-versus-graph logits parity failed"
        )
    if warmed["target_tokens"] != eager["target_tokens"]:
        raise ValueError(
            "eager-versus-graph target-token parity failed"
        )
    if (
        warmed["accepted_lengths"]
        != eager["accepted_lengths"]
    ):
        raise ValueError(
            "eager-versus-graph accepted-length parity failed"
        )
    if warmed["final_tokens"] != eager["final_tokens"]:
        raise ValueError(
            "eager-versus-graph final-token parity failed"
        )
    if (
        warmed["accepted_prefix_kv_sha256"]
        != eager["accepted_prefix_kv_sha256"]
    ):
        raise ValueError(
            "eager-versus-graph accepted-prefix KV parity failed"
        )
    if warmed["target_forward_count"] != 1:
        raise ValueError(
            "warmed graph must record one target forward"
        )
    if (
        _require_int(
            warmed.get("eager_forward_count"),
            "warmed_graph.eager_forward_count",
        )
        != 0
    ):
        raise ValueError(
            "warmed graph must add zero eager forward calls"
        )
    if (
        _require_int(
            warmed.get("graph_replay_count"),
            "warmed_graph.graph_replay_count",
        )
        != 1
    ):
        raise ValueError(
            "warmed graph must record one graph replay"
        )
    _require_int(
        warmed.get("warmed_latency_ns"),
        "warmed_graph.warmed_latency_ns",
        minimum=1,
    )
    _require_int(
        warmed.get("capture_latency_ns"),
        "warmed_graph.capture_latency_ns",
    )

    transaction = _require_dict(
        family.get("transaction_results"),
        "transaction_results",
    )
    if (
        _require_bool(
            transaction.get("accepted_prefix_kv_parity"),
            "accepted-prefix KV parity",
        )
        is not True
    ):
        raise ValueError(
            "accepted-prefix KV parity must pass"
        )
    if (
        _require_bool(
            transaction.get("rejected_suffix_released"),
            "rejected suffix release",
        )
        is not True
    ):
        raise ValueError(
            "rejected suffix must be released"
        )
    transaction_states = _require_list(
        transaction.get("transaction_states"),
        "transaction states",
    )
    if (
        len(transaction_states) != batch_size
        or any(
            state != "committed"
            for state in transaction_states
        )
    ):
        raise ValueError(
            "transaction state evidence must be committed"
        )
    materialized_counts = _require_count_list(
        transaction.get("materialized_token_counts"),
        "materialized token counts",
        expected_length=batch_size,
    )
    committed_counts = _require_count_list(
        transaction.get(
            "committed_materialized_token_counts"
        ),
        "committed materialized token counts",
        expected_length=batch_size,
    )
    rejected_counts = _require_count_list(
        transaction.get(
            "rejected_materialized_token_counts"
        ),
        "rejected materialized token counts",
        expected_length=batch_size,
    )
    for (
        materialized_count,
        committed_count,
        rejected_count,
        accepted_length,
    ) in zip(
        materialized_counts,
        committed_counts,
        rejected_counts,
        eager["accepted_lengths"],
    ):
        if materialized_count != query_len:
            raise ValueError(
                "materialized token count must equal exact Q"
            )
        if committed_count != max(0, accepted_length - 1):
            raise ValueError(
                "committed materialized token count must match "
                "accepted materialized prefix"
            )
        if (
            rejected_count <= 0
            or committed_count + rejected_count
            != materialized_count
        ):
            raise ValueError(
                "rejected materialized token count is "
                "contradictory"
            )
    unused = _require_block_id_list(
        transaction.get("unused_block_ids"),
        "unused_block_ids",
    )
    released = _require_block_id_list(
        transaction.get("released_block_ids"),
        "released_block_ids",
    )
    if released != unused:
        raise ValueError(
            "unused block release evidence must exactly match"
        )
    if (
        _require_bool(
            transaction.get("all_unused_blocks_released"),
            "all unused blocks released",
        )
        is not True
    ):
        raise ValueError(
            "all unused blocks must be released"
        )

    failure = _require_dict(
        family.get("replay_failure_injection"),
        "replay_failure_injection",
    )
    if (
        _require_int(
            failure.get("graph_replay_count"),
            "failure graph replay count",
        )
        != 1
    ):
        raise ValueError(
            "failure injection must attempt one graph replay"
        )
    if (
        _require_int(
            failure.get("eager_retry_count"),
            "failure eager retry count",
        )
        != 0
    ):
        raise ValueError(
            "failure injection must perform zero eager retry"
        )
    if (
        _require_bool(
            failure.get("error_propagated"),
            "failure error propagated",
        )
        is not True
    ):
        raise ValueError(
            "failure injection error must be propagated"
        )
    if (
        failure.get("quarantine_reason")
        != "replay_failed"
    ):
        raise ValueError(
            "failure injection quarantine reason is invalid"
        )
    if (
        _require_bool(
            failure.get("stable_quarantine_reason"),
            "stable quarantine reason",
        )
        is not True
    ):
        raise ValueError(
            "failure injection requires stable quarantine reason"
        )
    return identity


def _require_exact_family_metric_map(
    value,
    name: str,
    *,
    family_ids: set[str],
    list_values: bool,
    minimum: int,
) -> dict:
    rows = _require_dict(value, name)
    if set(rows) != family_ids:
        raise ValueError(
            f"{name} must cover every exact family"
        )
    for family_id, metric in rows.items():
        if list_values:
            values = _require_list(
                metric,
                f"{name}.{family_id}",
            )
            if not values:
                raise ValueError(
                    f"{name} warmed measurements must be non-empty"
                )
            for row in values:
                _require_int(
                    row,
                    f"{name}.{family_id}",
                    minimum=minimum,
                )
        else:
            _require_int(
                metric,
                f"{name}.{family_id}",
                minimum=minimum,
            )
    return rows


def _validate_memory_metrics(value, name: str) -> dict:
    value = _require_dict(value, name)
    allocated = _require_int(
        value.get("gpu_allocated_bytes"),
        f"{name}.gpu_allocated_bytes",
    )
    reserved = _require_int(
        value.get("gpu_reserved_bytes"),
        f"{name}.gpu_reserved_bytes",
    )
    if reserved < allocated:
        raise ValueError(
            f"{name} reserved GPU memory is contradictory"
        )
    return value


def _validate_performance(
    performance,
    *,
    family_ids: set[str],
    batch_sizes: list[int],
    query_lengths: list[int],
) -> dict:
    performance = _require_dict(
        performance,
        "performance",
    )
    warmup_count = _require_int(
        performance.get("warmup_count"),
        "performance warmup_count",
        minimum=1,
    )
    if warmup_count != REQUIRED_PERFORMANCE_WARMUP_COUNT:
        raise ValueError(
            "performance warmup_count must be exactly 2"
        )
    measurement_count = _require_int(
        performance.get("measurement_count"),
        "performance measurement_count",
        minimum=1,
    )
    expected_measurement_count = (
        len(family_ids)
        * REQUIRED_PERFORMANCE_MEASUREMENTS_PER_FAMILY
    )
    if measurement_count != expected_measurement_count:
        raise ValueError(
            "performance measurement_count must be exactly "
            f"{expected_measurement_count}"
        )

    _require_exact_family_metric_map(
        performance.get("prompt_lengths"),
        "performance prompt_lengths",
        family_ids=family_ids,
        list_values=False,
        minimum=1,
    )

    proposal_distribution = _require_dict(
        performance.get("proposal_length_distribution"),
        "performance proposal-length distribution",
    )
    if set(proposal_distribution) != {
        str(value) for value in query_lengths
    }:
        raise ValueError(
            "performance proposal-length distribution must "
            "cover every exact Q"
        )
    normalized_proposal_distribution = {}
    for query_len, count in proposal_distribution.items():
        normalized_proposal_distribution[query_len] = _require_int(
            count,
            "performance proposal-length count",
            minimum=1,
        )
    width_count = len(family_ids) // (
        len(batch_sizes) * len(query_lengths)
    )
    expected_proposal_distribution = {
        str(query_len): (
            sum(batch_sizes) * query_len * width_count
        )
        for query_len in query_lengths
    }
    if (
        normalized_proposal_distribution
        != expected_proposal_distribution
    ):
        raise ValueError(
            "performance proposal-length distribution counts "
            "must match the exact MVP matrix"
        )

    batch_distribution = _require_dict(
        performance.get("batch_distribution"),
        "performance batch distribution",
    )
    if set(batch_distribution) != {
        str(value) for value in batch_sizes
    }:
        raise ValueError(
            "performance batch distribution must cover "
            "batch 1 and 4"
        )
    normalized_batch_distribution = {}
    for batch_size, count in batch_distribution.items():
        normalized_batch_distribution[batch_size] = _require_int(
            count,
            "performance batch count",
            minimum=1,
        )
    expected_batch_distribution = {
        str(batch_size): len(query_lengths) * width_count
        for batch_size in batch_sizes
    }
    if normalized_batch_distribution != expected_batch_distribution:
        raise ValueError(
            "performance batch distribution counts must match "
            "the exact MVP matrix"
        )

    eager = _validate_memory_metrics(
        performance.get("eager_baseline"),
        "performance eager_baseline",
    )
    _require_int(
        eager.get("ttft_ns"),
        "performance eager TTFT",
        minimum=1,
    )
    _require_int(
        eager.get("tpot_ns"),
        "performance eager TPOT",
        minimum=1,
    )
    _require_number(
        eager.get("throughput_tokens_per_second"),
        "performance eager throughput",
        minimum=0.0,
    )
    if eager["throughput_tokens_per_second"] <= 0:
        raise ValueError(
            "performance eager throughput must be positive"
        )

    warmed = _validate_memory_metrics(
        performance.get("warmed_exact_graph_hits"),
        "performance warmed exact graph hits",
    )
    if (
        _require_int(
            warmed.get("measurement_count"),
            "performance warmed measurement_count",
            minimum=1,
        )
        != measurement_count
    ):
        raise ValueError(
            "performance warmed measurement count mismatch"
        )
    latency_by_family = _require_exact_family_metric_map(
        warmed.get("latency_ns_by_family"),
        "performance warmed latency",
        family_ids=family_ids,
        list_values=True,
        minimum=1,
    )
    if any(
        len(rows)
        != REQUIRED_PERFORMANCE_MEASUREMENTS_PER_FAMILY
        for rows in latency_by_family.values()
    ):
        raise ValueError(
            "performance warmed latency must contain exactly 5 "
            "measurements per family"
        )

    mixed = _require_dict(
        performance.get("mixed_hit_rate"),
        "performance mixed hit rate",
    )
    mixed_measurements = _require_int(
        mixed.get("measurement_count"),
        "performance mixed measurement_count",
        minimum=1,
    )
    hit_count = _require_int(
        mixed.get("hit_count"),
        "performance mixed hit_count",
        minimum=1,
    )
    miss_count = _require_int(
        mixed.get("miss_count"),
        "performance mixed miss_count",
        minimum=1,
    )
    if hit_count + miss_count != mixed_measurements:
        raise ValueError(
            "performance mixed hit/miss counts are contradictory"
        )
    _require_int(
        mixed.get("end_to_end_tpot_ns"),
        "performance mixed TPOT",
        minimum=1,
    )
    _require_int(
        mixed.get("ttft_ns"),
        "performance mixed TTFT",
        minimum=1,
    )
    if (
        _require_number(
            mixed.get("throughput_tokens_per_second"),
            "performance mixed throughput",
        )
        <= 0
    ):
        raise ValueError(
            "performance mixed throughput must be positive"
        )

    capture = _require_dict(
        performance.get("capture"),
        "performance capture",
    )
    _require_exact_family_metric_map(
        capture.get("duration_ns_by_family"),
        "performance capture duration",
        family_ids=family_ids,
        list_values=False,
        minimum=1,
    )
    _require_exact_family_metric_map(
        capture.get("allocated_delta_bytes_by_family"),
        "performance capture allocated delta",
        family_ids=family_ids,
        list_values=False,
        minimum=0,
    )
    _require_exact_family_metric_map(
        capture.get("reserved_delta_bytes_by_family"),
        "performance capture reserved delta",
        family_ids=family_ids,
        list_values=False,
        minimum=0,
    )

    cache_counts = _require_dict(
        performance.get("cache_counts"),
        "performance cache counts",
    )
    for name in (
        "hits",
        "misses",
        "evictions",
        "quarantines",
    ):
        minimum = 1 if name in ("hits", "misses") else 0
        _require_int(
            cache_counts.get(name),
            f"performance cache {name}",
            minimum=minimum,
        )

    acceptance = _require_dict(
        performance.get("acceptance"),
        "performance acceptance",
    )
    proposed = _require_int(
        acceptance.get("proposed_tokens"),
        "performance acceptance proposed_tokens",
        minimum=1,
    )
    accepted = _require_int(
        acceptance.get("accepted_draft_tokens"),
        "performance acceptance accepted_draft_tokens",
    )
    if accepted > proposed:
        raise ValueError(
            "performance acceptance token counts are contradictory"
        )
    rate = _require_number(
        acceptance.get("acceptance_rate"),
        "performance acceptance_rate",
    )
    if rate > 1.0 or not math.isclose(
        rate,
        accepted / proposed,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "performance acceptance rate is contradictory"
        )
    return {
        "measurement_count": measurement_count,
    }


def validate_artifact(artifact) -> dict:
    artifact = _require_dict(artifact, "artifact")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version is unsupported")
    _require_sha256(
        artifact.get("source_sha256"),
        "source_sha256",
    )
    source_files = _require_dict(
        artifact.get("source_files"),
        "source_files",
    )
    if not source_files:
        raise ValueError("source_files must be non-empty")
    for relative_path, digest in source_files.items():
        _require_string(relative_path, "source path")
        _require_sha256(digest, "source hash")

    _require_string(artifact.get("model"), "model")
    _require_string(artifact.get("checkpoint"), "checkpoint")
    _require_string(artifact.get("device_name"), "device_name")
    capability = _require_list(
        artifact.get("device_compute_capability"),
        "device compute capability",
    )
    if (
        len(capability) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in capability
        )
    ):
        raise ValueError(
            "device compute capability must contain two integers"
        )
    _require_string(artifact.get("torch_version"), "torch_version")
    _require_string(artifact.get("cuda_version"), "CUDA version")
    _require_string(
        artifact.get("flash_attn_version"),
        "FlashAttention version",
    )

    configuration = _require_dict(
        artifact.get("configuration"),
        "configuration",
    )
    if configuration.get("tensor_parallel_size") != 1:
        raise ValueError("only TP1 evidence is accepted")
    if configuration.get("kv_offload_mvp0") is not False:
        raise ValueError("KV offload evidence is unsupported")
    if configuration.get("context_length") != 4096:
        raise ValueError("context length must be exactly 4096")
    batch_sizes = _require_exact_int_list(
        configuration.get("batch_sizes"),
        "batch sizes",
    )
    if tuple(batch_sizes) != REQUIRED_BATCH_SIZES:
        raise ValueError("batch sizes must be exactly [1, 4]")
    query_lengths = _require_exact_int_list(
        configuration.get("query_lengths"),
        "query lengths",
    )
    if tuple(query_lengths) != REQUIRED_QUERY_LENGTHS:
        raise ValueError("query lengths must be exactly [1, 3]")
    widths = _require_exact_int_list(
        configuration.get("page_table_widths"),
        "page table widths",
    )
    if tuple(widths) != REQUIRED_PAGE_TABLE_WIDTHS:
        raise ValueError(
            "page table widths must be exactly [1, 2]"
        )
    if configuration.get("identity_policy") != "exact_b_q_w":
        raise ValueError("identity policy must be exact B/Q/W")
    if (
        configuration.get(
            "capture_latency_excluded_from_warmed_hit"
        )
        is not True
    ):
        raise ValueError(
            "capture latency must be excluded from warmed-hit latency"
        )
    measure_performance = configuration.get(
        "measure_performance",
        False,
    )
    if not isinstance(measure_performance, bool):
        raise ValueError(
            "measure_performance must be a boolean"
        )

    if artifact.get("classification") != CLASSIFICATION:
        raise ValueError(
            "classification must be NOT_PROMOTABLE"
        )
    claims = _require_dict(artifact.get("claims"), "claims")
    if claims.get("kv_offload_benefit") is not False:
        raise ValueError("KV offload benefit claims are forbidden")
    if claims.get("h2d_d2h_benefit") is not False:
        raise ValueError("H2D/D2H benefit claims are forbidden")

    families = _require_list(
        artifact.get("families"),
        "families",
    )
    expected_identities = {
        (batch_size, query_len, width)
        for batch_size in batch_sizes
        for query_len in query_lengths
        for width in widths
    }
    observed_identities = [
        _validate_family(
            family,
            enabled_batches=set(batch_sizes),
            enabled_queries=set(query_lengths),
            enabled_widths=set(widths),
        )
        for family in families
    ]
    if (
        len(set(observed_identities))
        != len(observed_identities)
    ):
        raise ValueError("family matrix contains duplicates")
    if set(observed_identities) != expected_identities:
        raise ValueError(
            "family matrix does not cover every exact B/Q/W family"
        )
    family_ids = {
        f"b{batch_size}-q{query_len}-w{width}"
        for batch_size, query_len, width
        in expected_identities
    }

    performance_result = None
    if measure_performance:
        performance_result = _validate_performance(
            artifact.get("performance"),
            family_ids=family_ids,
            batch_sizes=batch_sizes,
            query_lengths=query_lengths,
        )
    elif artifact.get("performance") is not None:
        raise ValueError(
            "performance evidence requires "
            "measure_performance=true"
        )

    for section_name in (
        "eager_baseline",
        "warmed_graph",
        "replay_failure_injection",
        "transaction_results",
    ):
        section = _require_dict(
            artifact.get(section_name),
            section_name,
        )
        if section.get("family_count") != len(families):
            raise ValueError(
                f"{section_name} family_count mismatch"
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "classification": CLASSIFICATION,
        "family_count": len(families),
        "batch_sizes": batch_sizes,
        "query_lengths": query_lengths,
        "page_table_widths": widths,
        "performance_evidence": bool(
            performance_result is not None
        ),
        "performance_measurement_count": (
            0
            if performance_result is None
            else performance_result["measurement_count"]
        ),
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_artifact(
    artifact_path: Path,
    repo_root: Path,
) -> dict:
    artifact_path = Path(artifact_path)
    repo_root = Path(repo_root)
    artifact = json.loads(
        artifact_path.read_text(encoding="utf-8")
    )
    result = validate_artifact(artifact)
    for relative_path, expected_digest in artifact[
        "source_files"
    ].items():
        source_path = repo_root / relative_path
        if not source_path.is_file():
            raise ValueError(
                f"source file is missing: {relative_path}"
            )
        if _sha256_file(source_path) != expected_digest:
            raise ValueError(
                f"source hash mismatch: {relative_path}"
            )
    return {
        **result,
        "artifact_sha256": _sha256_file(artifact_path),
        "source_sha256": artifact["source_sha256"],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = verify_artifact(
        Path(args.artifact),
        args.repo_root,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(
            payload + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
