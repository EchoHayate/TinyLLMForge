#!/usr/bin/env python3
"""Assemble and classify Qwen3.8 topology-local TP2 whole-model evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import statistics


WORKLOADS = ("P0", "P1", "Q0", "Q1", "Q2")
ONLINE_WORKLOADS = ("Q0", "Q1", "Q2")
WORKLOAD_SHAPES = {
    "P0": (256, 128, 1),
    "P1": (2048, 128, 1),
    "Q0": (256, 128, 4),
    "Q1": (256, 128, 8),
    "Q2": (2048, 128, 4),
}
EPOCH_ARMS = ("baseline", "candidate", "candidate", "baseline")
CORRECTNESS_RANKS = (0, 1, 2, 3)
CORRECTNESS_CHECKPOINTS = (
    "pre_migration",
    "post_migration",
    "token_1",
    "token_4",
    "token_8",
    "token_32",
    "token_128",
)
CORRECTNESS_COMMIT_DELTAS = {
    "pre_migration": 0,
    "token_1": 0,
    "post_migration": 1,
    "token_4": 3,
    "token_8": 7,
    "token_32": 31,
    "token_128": 127,
}
TOPOLOGY_LINK_COST = {
    "PIX": 0,
    "PXB": 1,
    "PHB": 2,
    "NODE": 3,
    "SYS": 4,
}
CORRECTNESS_LINEAR_LAYER_INDICES = frozenset(
    index for index in range(64) if index % 4 != 3
)
APPROVED_REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
EXPECTED_CLEANUP_WORKERS = {
    "correctness/baseline": (False, 4),
    "correctness/candidate": (True, 4),
    "epoch/0/baseline": (False, 4),
    "epoch/1/candidate": (True, 4),
    "epoch/2/candidate": (True, 4),
    "epoch/3/baseline": (False, 4),
    **{
        f"service/{workload}/replica/{replica}": (False, 2)
        for workload in ONLINE_WORKLOADS
        for replica in range(2)
    },
}
BOUNDARY_RESOURCE_STAGES = frozenset({
    "entry",
    "pre_correctness",
    "post_correctness",
    *(f"pre_epoch_{index}" for index in range(4)),
    *(f"post_launch_{index}" for index in range(4)),
    "pre_service_control",
    "post_service_control",
    "terminal",
})
RUNTIME_RESOURCE_LABELS = frozenset({
    "correctness",
    *(f"epoch_{index}" for index in range(4)),
    "service_control",
})
CORE_ARTIFACTS = (
    "source_manifest.json",
    "model_manifest.json",
    "environment_manifest.json",
    "gpu_topology.json",
    "gpu_rank_manifest.json",
    "pair_group_manifest.json",
    "workload_manifest.json",
    "campaign_epoch_manifest.json",
    "feature_contract.json",
    "weight_layout_manifest.json",
    "state_layout_manifest.json",
    "migration_rows.jsonl",
    "correctness_rows.jsonl",
    "request_rows.jsonl",
    "scheduler_step_rows.jsonl",
    "candidate_hit_rows.jsonl",
    "collective_rows.jsonl",
    "memory_rows.jsonl",
    "resource_rows.jsonl",
    "service_control_rows.jsonl",
    "cleanup.json",
)
JSONL_ARTIFACTS = frozenset(
    name for name in CORE_ARTIFACTS if name.endswith(".jsonl")
)
MANIFEST_SCHEMA = (
    "qwen38.topology-local-tp2-whole-model-manifest.v1"
)


def _duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"JSON number must be finite: {value}")


def _load_json(path: Path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_duplicate_keys,
                parse_constant=_nonfinite,
            )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error


def _load_jsonl(path: Path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(
                    line,
                    object_pairs_hook=_duplicate_keys,
                    parse_constant=_nonfinite,
                ))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from error
    return rows


def _require_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("numeric evidence must be finite")
    if isinstance(value, dict):
        for child in value.values():
            _require_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _require_finite(child)


def _write_json(path: Path, payload: object) -> None:
    _require_finite(payload)
    Path(path).write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_lower_hex(value, length):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def geometric_mean(values) -> float:
    values = tuple(float(value) for value in values)
    if not values or any(
        not math.isfinite(value) or value <= 0
        for value in values
    ):
        raise ValueError(
            "geometric mean inputs must be finite and positive"
        )
    return math.exp(
        sum(math.log(value) for value in values) / len(values)
    )


def nearest_rank_percentile(values, percentile) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise ValueError(
            "percentile inputs must be finite and non-empty"
        )
    percentile = float(percentile)
    if not math.isfinite(percentile) or not 0 < percentile <= 100:
        raise ValueError("percentile must be finite and in (0, 100]")
    rank = max(1, math.ceil((percentile / 100.0) * len(ordered)))
    return ordered[rank - 1]


def classify(summary) -> str:
    if summary.get("correctness_pass") is not True:
        return "NO_GO_CORRECTNESS_OR_LIFECYCLE"
    if summary.get("resource_identity_pass") is not True:
        return "NO_GO_RESOURCE_IDENTITY"
    if summary.get("candidate_coverage_pass") is not True:
        return "NO_GO_CANDIDATE_NOT_EXERCISED"
    if summary.get("memory_pass") is not True:
        return "NO_GO_MEMORY_OR_ALLOCATION"
    if summary.get("measurement_complete") is not True:
        return "INCONCLUSIVE_ENVIRONMENT_OR_MEASUREMENT"
    if summary.get("tail_ttft_pass") is not True:
        return "NO_GO_TAIL_OR_TTFT"
    if summary.get("throughput_pass") is not True:
        return "NO_GO_THROUGHPUT"
    if summary.get("migration_pass") is not True:
        return "NO_GO_MIGRATION_AMORTIZATION"
    direction_counts = summary.get("pair_direction_counts", {})
    if (
        float(summary.get(
            "aggregate_median_tpot_improvement_percent",
            float("-inf"),
        )) < 5.0
        or int(summary.get("improving_workload_count", 0)) < 4
        or summary.get("median_regressing_workloads")
        or set(direction_counts) != set(WORKLOADS)
        or any(int(direction_counts[name]) < 7 for name in WORKLOADS)
    ):
        return "NO_GO_PERFORMANCE"
    return "GO_TOPOLOGY_LOCAL_TP2_WHOLE_MODEL_GATE"


def _row_identity(row):
    return (
        row.get("epoch"),
        row.get("arm"),
        row.get("workload_id"),
        row.get("repetition"),
    )


def _case_identity(row):
    return (
        row.get("epoch"),
        row.get("workload_id"),
        row.get("repetition"),
    )


def _validate_request_timing(request):
    admitted_ns = request.get("admitted_ns")
    first_scheduled_ns = request.get("first_scheduled_ns")
    queueing_ns = request.get("queueing_ns")
    completion_ns = request.get("completion_ns")
    timestamps = request.get("token_timestamps_ns")
    if (
        isinstance(admitted_ns, bool)
        or not isinstance(admitted_ns, int)
        or admitted_ns < 0
        or isinstance(first_scheduled_ns, bool)
        or not isinstance(first_scheduled_ns, int)
        or first_scheduled_ns < admitted_ns
        or isinstance(queueing_ns, bool)
        or not isinstance(queueing_ns, int)
        or queueing_ns != first_scheduled_ns - admitted_ns
        or isinstance(completion_ns, bool)
        or not isinstance(completion_ns, int)
        or not isinstance(timestamps, list)
        or len(timestamps) != 128
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in timestamps
        )
        or any(
            current < previous
            for previous, current in zip(timestamps, timestamps[1:])
        )
        or timestamps[0] < admitted_ns
        or completion_ns != timestamps[-1]
    ):
        raise ValueError("raw request timing evidence is invalid")
    gaps = [
        current - previous
        for previous, current in zip(timestamps, timestamps[1:])
    ]
    derived = {
        "ttft_ns": timestamps[0] - admitted_ns,
        "tpot_ns": (timestamps[-1] - timestamps[0]) / 127,
        "e2e_ns": completion_ns - admitted_ns,
    }
    if (
        request.get("token_gaps_ns") != gaps
        or any(
            isinstance(request.get(field), bool)
            or not isinstance(request.get(field), (int, float))
            or not math.isclose(
                float(request[field]),
                float(value),
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            for field, value in derived.items()
        )
    ):
        raise ValueError(
            "reported request timing does not match raw timestamps"
        )
    return admitted_ns, completion_ns


def _validate_topology_matching(payload):
    rows = payload.get("rows")
    if payload.get("selection_frozen") is not True or not isinstance(
        rows,
        list,
    ):
        raise ValueError("GPU topology evidence is invalid")
    directed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("GPU topology evidence is invalid")
        left = row.get("left_rank")
        right = row.get("right_rank")
        link = row.get("link")
        key = (left, right)
        if (
            isinstance(left, bool)
            or not isinstance(left, int)
            or isinstance(right, bool)
            or not isinstance(right, int)
            or left not in CORRECTNESS_RANKS
            or right not in CORRECTNESS_RANKS
            or left == right
            or key in directed
            or link not in TOPOLOGY_LINK_COST
        ):
            raise ValueError("GPU topology evidence is invalid")
        directed[key] = TOPOLOGY_LINK_COST[link]
    if set(directed) != {
        (left, right)
        for left in CORRECTNESS_RANKS
        for right in CORRECTNESS_RANKS
        if left != right
    }:
        raise ValueError("GPU topology evidence is incomplete")
    costs = {}
    for left in CORRECTNESS_RANKS:
        for right in CORRECTNESS_RANKS:
            if left >= right:
                continue
            if directed[(left, right)] != directed[(right, left)]:
                raise ValueError("GPU topology evidence is asymmetric")
            costs[(left, right)] = directed[(left, right)]
    matchings = (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )
    return min(
        matchings,
        key=lambda matching: (
            tuple(sorted(costs[pair] for pair in matching)),
            matching,
        ),
    )


def _validate_source_and_manifests(payloads):
    source = payloads["source_manifest.json"]
    model = payloads["model_manifest.json"]
    source_revision = source.get("source_revision")
    model_revision = source.get("model_revision")
    attempt_tag = source.get("attempt_tag")
    if (
        not _is_lower_hex(source_revision, 40)
        or not _is_lower_hex(model_revision, 40)
        or not _is_lower_hex(source.get("source_tree_sha256"), 64)
        or not isinstance(attempt_tag, str)
        or not attempt_tag
        or not attempt_tag[0].isalnum()
        or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz0123456789._-"
            for character in attempt_tag
        )
        or source.get("source_archive_complete") is not True
    ):
        raise ValueError("source archive or revision is invalid")
    if (
        model.get("model_repository") != "Qwen/Qwen3.8-27B"
        or model.get("model_revision") != model_revision
        or model.get("source_revision") != source_revision
        or model.get("num_hidden_layers") != 64
        or model.get("linear_attention_layer_count") != 48
        or model.get("full_attention_layer_count") != 16
    ):
        raise ValueError("model identity or layer manifest mismatch")
    for name, payload in payloads.items():
        if name in JSONL_ARTIFACTS:
            continue
        if (
            name == "source_manifest.json"
            or not isinstance(payload, dict)
            or payload.get("source_revision") != source_revision
            or payload.get("model_revision") != model_revision
        ):
            if name != "source_manifest.json":
                raise ValueError(f"{name} source or model identity drift")
    expected_epochs = [
        {
            "epoch": epoch,
            "arm": arm,
            "workload_order": (
                list(WORKLOADS)
                if epoch in (0, 2)
                else list(reversed(WORKLOADS))
            ),
        }
        for epoch, arm in enumerate(EPOCH_ARMS)
    ]
    if payloads["campaign_epoch_manifest.json"].get("epochs") != expected_epochs:
        raise ValueError(
            "campaign epoch identity/order must be exact A/B/B/A"
        )
    if (
        payloads["environment_manifest.json"].get(
            "environment_complete"
        )
        is not True
    ):
        raise ValueError("environment manifest is incomplete")
    feature = payloads["feature_contract.json"]
    if (
        feature.get("default_off") is not True
        or feature.get("eager") is not True
        or feature.get("tensor_parallel_size") != 4
    ):
        raise ValueError("feature contract is invalid")
    rank_manifest = payloads["gpu_rank_manifest.json"]
    mapping = rank_manifest.get("mapping")
    if (
        rank_manifest.get("ranks") != list(range(4))
        or not isinstance(mapping, list)
        or [row.get("rank") for row in mapping if isinstance(row, dict)]
        != list(range(4))
    ):
        raise ValueError("GPU rank mapping identity mismatch")
    topology_matching = _validate_topology_matching(
        payloads["gpu_topology.json"]
    )
    if (
        topology_matching != ((0, 1), (2, 3))
        or payloads["pair_group_manifest.json"].get("pair_groups")
        != [list(pair) for pair in topology_matching]
        or payloads["pair_group_manifest.json"].get("optimal_matching")
        is not True
    ):
        raise ValueError("pair-group topology identity mismatch")
    if (
        payloads["workload_manifest.json"].get("workloads")
        != list(WORKLOADS)
    ):
        raise ValueError("workload manifest mismatch")
    return source_revision, model_revision


def _validate_request_rows(rows, source_revision, model_revision):
    expected = {
        (epoch, arm, workload, repetition)
        for epoch, arm in enumerate(EPOCH_ARMS)
        for workload in WORKLOADS
        for repetition in range(5)
    }
    indexed = {}
    for row in rows:
        _require_finite(row)
        identity = _row_identity(row)
        if identity in indexed:
            raise ValueError("duplicate request row identity")
        indexed[identity] = row
        if (
            row.get("source_revision") != source_revision
            or row.get("model_revision") != model_revision
        ):
            raise ValueError("request source or model identity drift")
        requests = row.get("requests")
        if not isinstance(requests, list) or not requests:
            raise ValueError("request inventory is incomplete")
        shape = WORKLOAD_SHAPES.get(row.get("workload_id"))
        if shape is None or len(requests) != shape[2]:
            raise ValueError("request concurrency does not match workload")
        request_ids = []
        runtime_request_ids = []
        timing_bounds = []
        for request in requests:
            gaps = request.get("token_gaps_ns")
            request_id = request.get("request_id")
            runtime_request_id = request.get("runtime_request_id")
            decoded_text = request.get("decoded_text")
            decoded_sha256 = request.get("decoded_text_sha256")
            if (
                not isinstance(request_id, str)
                or not request_id
                or isinstance(runtime_request_id, bool)
                or not isinstance(runtime_request_id, int)
                or runtime_request_id < 0
                or not isinstance(request.get("output_token_ids"), list)
                or len(request["output_token_ids"]) != 128
                or not isinstance(gaps, list)
                or len(gaps) != 127
                or request.get("complete") is not True
                or request.get("prompt_tokens") != shape[0]
                or request.get("generated_tokens") != shape[1]
                or request.get("stop_position") != 128
                or request.get("stop_reason") != "length"
            ):
                raise ValueError(
                    "each request requires 128 tokens and 127 token gaps"
                )
            request_ids.append(request_id)
            runtime_request_ids.append(runtime_request_id)
            timing_bounds.append(_validate_request_timing(request))
            if (
                not isinstance(decoded_text, str)
                or not isinstance(decoded_sha256, str)
                or decoded_sha256
                != hashlib.sha256(
                    decoded_text.encode("utf-8")
                ).hexdigest()
            ):
                raise ValueError("decoded text digest is invalid")
            for field in ("ttft_ns", "tpot_ns", "e2e_ns"):
                value = request.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or value < 0
                ):
                    raise ValueError(
                        "request duration must be finite and non-negative"
                    )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
                for value in gaps
            ):
                raise ValueError(
                    "token gaps must be finite and non-negative"
                )
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request identity is duplicated")
        if len(set(runtime_request_ids)) != len(runtime_request_ids):
            raise ValueError("runtime request identity is duplicated")
        cohort_makespan = row.get("cohort_makespan_ns")
        derived_makespan = (
            max(completion for _, completion in timing_bounds)
            - min(admitted for admitted, _ in timing_bounds)
        )
        if (
            isinstance(cohort_makespan, bool)
            or not isinstance(cohort_makespan, (int, float))
            or not math.isfinite(float(cohort_makespan))
            or not math.isclose(
                float(cohort_makespan),
                float(derived_makespan),
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                "cohort makespan does not match raw request timing"
            )
        replay = row.get("timing_correctness_replay")
        proof = (
            _correctness_step_proof(
                replay.get("requests"),
                replay.get("step_proofs"),
            )
            if isinstance(replay, dict)
            else None
        )
        measured_identity = tuple(
            (
                request.get("request_id"),
                tuple(request.get("output_token_ids", ())),
                request.get("stop_position"),
                request.get("stop_reason"),
                request.get("decoded_text"),
                request.get("decoded_text_sha256"),
            )
            for request in requests
        )
        replay_identity = (
            tuple(
                (
                    request.get("request_id"),
                    tuple(request.get("output_token_ids", ())),
                    request.get("stop_position"),
                    request.get("stop_reason"),
                    request.get("decoded_text"),
                    request.get("decoded_text_sha256"),
                )
                for request in replay.get("requests", ())
            )
            if isinstance(replay, dict)
            else ()
        )
        if (
            proof is None
            or measured_identity != replay_identity
            or row.get("rank_token_agreement") is not True
            or row.get("finite_logits") is not True
            or row.get("top_logit_values_match") is not True
        ):
            raise RuntimeError(
                "timing correctness token replay evidence is invalid"
            )
    if set(indexed) != expected:
        raise ValueError(
            "each workload requires ten baseline and ten candidate rows"
        )
    for workload in WORKLOADS:
        for repetition in range(5):
            cohort = [
                indexed[(epoch, EPOCH_ARMS[epoch], workload, repetition)]
                for epoch in range(4)
            ]
            digests = {
                row.get("request_set_digest") for row in cohort
            }
            if len(digests) != 1:
                raise ValueError("paired request-set digest mismatch")
            tokens = [
                tuple(tuple(request["output_token_ids"])
                      for request in row["requests"])
                for row in cohort
            ]
            if len(set(tokens)) != 1:
                raise RuntimeError("correctness output token mismatch")
            decoded_hashes = [
                tuple(
                    request["decoded_text_sha256"]
                    for request in row["requests"]
                )
                for row in cohort
            ]
            if len(set(decoded_hashes)) != 1:
                raise RuntimeError("decoded text mismatch")
    return indexed


def _correctness_requests(raw_requests):
    if not isinstance(raw_requests, list) or not raw_requests:
        return None
    request_ids = []
    sequence_ids = []
    output_tokens = []
    for request in raw_requests:
        if not isinstance(request, dict):
            return None
        request_id = request.get("request_id")
        sequence_id = request.get("runtime_request_id")
        tokens = request.get("output_token_ids")
        if (
            not isinstance(request_id, str)
            or not request_id
            or isinstance(sequence_id, bool)
            or not isinstance(sequence_id, int)
            or not isinstance(tokens, list)
            or len(tokens) != 128
            or any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                for token in tokens
            )
        ):
            return None
        request_ids.append(request_id)
        sequence_ids.append(sequence_id)
        output_tokens.append(tuple(tokens))
    if (
        len(set(request_ids)) != len(request_ids)
        or len(set(sequence_ids)) != len(sequence_ids)
    ):
        return None
    return {
        "request_ids": tuple(request_ids),
        "sequence_ids": tuple(sequence_ids),
        "output_tokens": tuple(output_tokens),
    }


def _correctness_step_proof(raw_requests, raw_proofs):
    requests = _correctness_requests(raw_requests)
    if (
        requests is None
        or not isinstance(raw_proofs, list)
        or len(raw_proofs) != 128
    ):
        return None
    sequence_ids = requests["sequence_ids"]
    emitted = [[] for _ in sequence_ids]
    top_values = []
    for raw_step in raw_proofs:
        if not isinstance(raw_step, list) or len(raw_step) != 4:
            return None
        if any(not isinstance(rank_row, dict) for rank_row in raw_step):
            return None
        ranked = sorted(raw_step, key=lambda rank_row: rank_row.get("rank", -1))
        if [rank_row.get("rank") for rank_row in ranked] != list(
            CORRECTNESS_RANKS
        ):
            return None
        tokens_by_rank = []
        values_by_rank = []
        for rank_row in ranked:
            tokens = rank_row.get("token_ids")
            values = rank_row.get("top_logit_values")
            if (
                tuple(rank_row.get("sequence_ids", ())) != sequence_ids
                or rank_row.get("finite_logits") is not True
                or not isinstance(tokens, list)
                or len(tokens) != len(sequence_ids)
                or any(
                    isinstance(token, bool)
                    or not isinstance(token, int)
                    or token < 0
                    for token in tokens
                )
                or not isinstance(values, list)
                or len(values) != len(sequence_ids)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in values
                )
            ):
                return None
            tokens_by_rank.append(tuple(tokens))
            values_by_rank.append(tuple(float(value) for value in values))
        if len(set(tokens_by_rank)) != 1:
            return None
        reference_values = values_by_rank[0]
        if any(
            not math.isclose(
                value,
                reference_values[index],
                rel_tol=2e-3,
                abs_tol=2e-2,
            )
            for values in values_by_rank[1:]
            for index, value in enumerate(values)
        ):
            return None
        for index, token in enumerate(tokens_by_rank[0]):
            emitted[index].append(token)
        top_values.append(reference_values)
    if tuple(tuple(tokens) for tokens in emitted) != requests["output_tokens"]:
        return None
    return {
        "requests": requests,
        "top_logit_values": tuple(top_values),
    }


def _correctness_digest_map(rows, value_fields):
    if not isinstance(rows, list) or len(rows) != 48:
        return None
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            return None
        layer_index = row.get("layer_index")
        values = tuple(row.get(field) for field in value_fields)
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index in result
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in value
                )
                for value in values
            )
        ):
            return None
        result[layer_index] = values
    if set(result) != CORRECTNESS_LINEAR_LAYER_INDICES:
        return None
    return result


def _correctness_component_map(rank_row, expected_source_ranks):
    rows = rank_row.get("canonical_state_components")
    if not isinstance(rows, list):
        return None
    expected_keys = {
        (layer_index, source_rank)
        for layer_index in CORRECTNESS_LINEAR_LAYER_INDICES
        for source_rank in expected_source_ranks
    }
    result = {}
    digest_fields = (
        "convolution_query_sha256",
        "convolution_key_sha256",
        "convolution_value_sha256",
        "recurrent_sha256",
    )
    for row in rows:
        if not isinstance(row, dict):
            return None
        layer_index = row.get("layer_index")
        source_rank = row.get("source_rank")
        key = (layer_index, source_rank)
        values = tuple(row.get(field) for field in digest_fields)
        if (
            key not in expected_keys
            or key in result
            or row.get("logical_rank") != source_rank // 2
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in value
                )
                for value in values
            )
        ):
            return None
        result[key] = values
    if set(result) != expected_keys:
        return None
    return result


def _correctness_cohort(rank_row):
    rows = rank_row.get("cohort")
    if not isinstance(rows, list) or not rows:
        return None
    result = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        values = tuple(
            row.get(field)
            for field in ("slot_id", "generation", "request_id")
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in values
        ):
            return None
        result.append(values)
    if (
        len({row[0] for row in result}) != len(result)
        or len({row[2] for row in result}) != len(result)
    ):
        return None
    return tuple(result)


def _correctness_state_comparison(
    raw_baseline_checkpoints,
    raw_candidate_checkpoints,
):
    if (
        not isinstance(raw_baseline_checkpoints, dict)
        or not isinstance(raw_candidate_checkpoints, dict)
        or set(raw_baseline_checkpoints) != set(CORRECTNESS_CHECKPOINTS)
        or set(raw_candidate_checkpoints) != set(CORRECTNESS_CHECKPOINTS)
    ):
        return False, False, False
    pair_replica_match = True
    baseline_candidate_match = True
    expected_cohort = None
    for name in CORRECTNESS_CHECKPOINTS:
        baseline_rows = raw_baseline_checkpoints.get(name)
        candidate_rows = raw_candidate_checkpoints.get(name)
        if (
            not isinstance(baseline_rows, list)
            or not isinstance(candidate_rows, list)
            or len(baseline_rows) != len(CORRECTNESS_RANKS)
            or len(candidate_rows) != len(CORRECTNESS_RANKS)
            or any(not isinstance(row, dict) for row in baseline_rows)
            or any(not isinstance(row, dict) for row in candidate_rows)
        ):
            return False, False, False
        baseline_rows = sorted(
            baseline_rows,
            key=lambda row: row.get("rank", -1),
        )
        candidate_rows = sorted(
            candidate_rows,
            key=lambda row: row.get("rank", -1),
        )
        if (
            [row.get("rank") for row in baseline_rows]
            != list(CORRECTNESS_RANKS)
            or [row.get("rank") for row in candidate_rows]
            != list(CORRECTNESS_RANKS)
        ):
            return False, False, False
        candidate_active = name not in {"pre_migration", "token_1"}
        baseline_cohorts = [
            _correctness_cohort(row) for row in baseline_rows
        ]
        candidate_cohorts = [
            _correctness_cohort(row) for row in candidate_rows
        ]
        checkpoint_cohort = baseline_cohorts[0]
        if (
            checkpoint_cohort is None
            or any(
                cohort != checkpoint_cohort
                for cohort in baseline_cohorts
            )
            or any(
                cohort != checkpoint_cohort
                for cohort in candidate_cohorts
            )
            or (
                expected_cohort is not None
                and checkpoint_cohort != expected_cohort
            )
        ):
            return False, False, False
        expected_cohort = checkpoint_cohort
        baseline_maps = []
        candidate_maps = []
        for rank in CORRECTNESS_RANKS:
            baseline_row = baseline_rows[rank]
            candidate_row = candidate_rows[rank]
            if (
                baseline_row.get("pair_id") != rank // 2
                or baseline_row.get("logical_rank") != rank % 2
                or baseline_row.get("state_layout")
                != "tp4_source_quarter"
                or candidate_row.get("pair_id") != rank // 2
                or candidate_row.get("logical_rank") != rank % 2
                or candidate_row.get("state_layout")
                != (
                    "tp2_logical_half"
                    if candidate_active
                    else "tp4_source_quarter"
                )
            ):
                return False, False, False
            baseline_maps.append(_correctness_component_map(
                baseline_row,
                (rank,),
            ))
            candidate_maps.append(_correctness_component_map(
                candidate_row,
                (
                    (2 * (rank % 2), 2 * (rank % 2) + 1)
                    if candidate_active
                    else (rank,)
                ),
            ))
        if (
            any(mapping is None for mapping in baseline_maps)
            or any(mapping is None for mapping in candidate_maps)
        ):
            return False, False, False
        baseline_map = {
            key: value
            for mapping in baseline_maps
            for key, value in mapping.items()
        }
        if candidate_active:
            if (
                candidate_maps[0] != candidate_maps[2]
                or candidate_maps[1] != candidate_maps[3]
            ):
                pair_replica_match = False
            candidate_map = {
                **candidate_maps[0],
                **candidate_maps[1],
            }
        else:
            candidate_map = {
                key: value
                for mapping in candidate_maps
                for key, value in mapping.items()
            }
        if candidate_map != baseline_map:
            baseline_candidate_match = False
    return True, pair_replica_match, baseline_candidate_match


def _correctness_checkpoints(raw_checkpoints):
    if (
        not isinstance(raw_checkpoints, dict)
        or set(raw_checkpoints) != set(CORRECTNESS_CHECKPOINTS)
    ):
        return None
    ranked_checkpoints = {}
    for name in CORRECTNESS_CHECKPOINTS:
        checkpoint = raw_checkpoints.get(name)
        if (
            not isinstance(checkpoint, list)
            or len(checkpoint) != len(CORRECTNESS_RANKS)
            or any(not isinstance(rank_row, dict) for rank_row in checkpoint)
        ):
            return None
        ranked = sorted(
            checkpoint,
            key=lambda rank_row: rank_row.get("rank", -1),
        )
        if [rank_row.get("rank") for rank_row in ranked] != list(
            CORRECTNESS_RANKS
        ):
            return None
        if any(
            rank_row.get("pair_id") != (rank // 2)
            or rank_row.get("logical_rank") != (rank % 2)
            for rank, rank_row in enumerate(ranked)
        ):
            return None
        ranked_checkpoints[name] = ranked
    base_counts = []
    for rank_row in ranked_checkpoints["pre_migration"]:
        snapshot = rank_row.get("runtime_snapshot")
        state = snapshot.get("state") if isinstance(snapshot, dict) else None
        base_count = state.get("commit_count") if isinstance(state, dict) else None
        if isinstance(base_count, bool) or not isinstance(base_count, int):
            return None
        base_counts.append(base_count)
    for name, expected_delta in CORRECTNESS_COMMIT_DELTAS.items():
        ranked = ranked_checkpoints[name]
        for rank, rank_row in enumerate(ranked):
            snapshot = rank_row.get("runtime_snapshot")
            state = (
                snapshot.get("state")
                if isinstance(snapshot, dict)
                else None
            )
            if (
                not isinstance(state, dict)
                or isinstance(state.get("commit_count"), bool)
                or not isinstance(state.get("commit_count"), int)
                or isinstance(state.get("rollback_count"), bool)
                or not isinstance(state.get("rollback_count"), int)
                or isinstance(
                    state.get("temporary_live_tensors"),
                    bool,
                )
                or not isinstance(
                    state.get("temporary_live_tensors"),
                    int,
                )
                or state.get("commit_count")
                != base_counts[rank] + expected_delta
                or state.get("rollback_count") != 0
                or state.get("temporary_live_tensors") != 0
            ):
                return None
        if name in {"pre_migration", "token_1"}:
            continue
        output_maps = [
            _correctness_digest_map(
                rank_row.get("output_digests"),
                ("sha256",),
            )
            for rank_row in ranked
        ]
        state_maps = [
            _correctness_digest_map(
                rank_row.get("state_digests"),
                ("convolution_sha256", "recurrent_sha256"),
            )
            for rank_row in ranked
        ]
        if (
            any(mapping is None for mapping in output_maps)
            or any(mapping is None for mapping in state_maps)
            or not all(mapping == output_maps[0] for mapping in output_maps[1:])
            or state_maps[0] != state_maps[2]
            or state_maps[1] != state_maps[3]
            or set(output_maps[0]) != set(state_maps[0])
        ):
            return None
    return True


def _reconstruct_correctness(row):
    baseline = _correctness_step_proof(
        row.get("baseline_requests"),
        row.get("baseline_step_proofs"),
    )
    candidate = _correctness_step_proof(
        row.get("candidate_requests"),
        row.get("candidate_step_proofs"),
    )
    output_tokens_match = (
        baseline is not None
        and candidate is not None
        and baseline["requests"]["request_ids"]
        == candidate["requests"]["request_ids"]
        and baseline["requests"]["output_tokens"]
        == candidate["requests"]["output_tokens"]
    )
    top_logit_values_match = (
        baseline is not None
        and candidate is not None
        and len(baseline["top_logit_values"])
        == len(candidate["top_logit_values"])
        and all(
            len(baseline_values) == len(candidate_values)
            and all(
                math.isclose(
                    baseline_value,
                    candidate_value,
                    rel_tol=2e-3,
                    abs_tol=2e-2,
                )
                for baseline_value, candidate_value in zip(
                    baseline_values,
                    candidate_values,
                )
            )
            for baseline_values, candidate_values in zip(
                baseline["top_logit_values"],
                candidate["top_logit_values"],
            )
        )
    )
    candidate_checkpoints_valid = (
        _correctness_checkpoints(
            row.get("candidate_state_checkpoints")
        )
        is True
    )
    (
        state_components_complete,
        component_pair_replica_match,
        baseline_candidate_state_match,
    ) = _correctness_state_comparison(
        row.get("baseline_state_checkpoints"),
        row.get("candidate_state_checkpoints"),
    )
    checkpoints_valid = (
        candidate_checkpoints_valid and state_components_complete
    )
    proof_valid = baseline is not None and candidate is not None
    return {
        "output_tokens_match": output_tokens_match,
        "rank_token_agreement": proof_valid,
        "finite_logits": proof_valid,
        "top_logit_values_match": top_logit_values_match,
        "state_checkpoints_complete": checkpoints_valid,
        "single_commit_per_step": candidate_checkpoints_valid,
        "pair_replica_digest_match": (
            candidate_checkpoints_valid
            and component_pair_replica_match
        ),
        "baseline_candidate_state_match": (
            baseline_candidate_state_match
        ),
    }


def _validate_correctness(rows, source_revision, model_revision):
    expected = {
        (workload, repetition)
        for workload in WORKLOADS
        for repetition in range(5)
    }
    indexed = {}
    required_true = (
        "output_tokens_match",
        "rank_token_agreement",
        "finite_logits",
        "top_logit_values_match",
        "state_checkpoints_complete",
        "single_commit_per_step",
        "pair_replica_digest_match",
        "baseline_candidate_state_match",
    )
    for row in rows:
        key = (row.get("workload_id"), row.get("repetition"))
        if key in indexed:
            raise ValueError("duplicate correctness row identity")
        indexed[key] = row
        if (
            row.get("source_revision") != source_revision
            or row.get("model_revision") != model_revision
        ):
            raise ValueError("correctness source or model identity drift")
        reconstructed = _reconstruct_correctness(row)
        if any(
            row.get(field) is not reconstructed[field]
            for field in required_true
        ):
            raise RuntimeError(
                "correctness summary does not match raw evidence"
            )
        if any(reconstructed[field] is not True for field in required_true):
            return False
    if set(indexed) != expected:
        raise ValueError("correctness row inventory mismatch")
    return True


def _validate_candidate_coverage(
    scheduler_rows,
    hit_rows,
    collective_rows,
    request_rows,
    source_revision,
    model_revision,
):
    scheduler = {}
    for row in scheduler_rows:
        key = _case_identity(row)
        if key in scheduler:
            raise ValueError("duplicate scheduler row identity")
        scheduler[key] = row
    hits = {}
    for row in hit_rows:
        key = _case_identity(row)
        if key in hits:
            raise ValueError("duplicate candidate hit row identity")
        hits[key] = row
    collectives = {}
    for row in collective_rows:
        key = _case_identity(row)
        if key in collectives:
            raise ValueError("duplicate collective row identity")
        collectives[key] = row
    expected_scheduler_keys = {
        (epoch, workload, repetition)
        for epoch in range(len(EPOCH_ARMS))
        for workload in WORKLOADS
        for repetition in range(5)
    }
    expected_candidate_keys = {
        (epoch, workload, repetition)
        for epoch, arm in enumerate(EPOCH_ARMS)
        if arm == "candidate"
        for workload in WORKLOADS
        for repetition in range(5)
    }
    if (
        set(scheduler) != expected_scheduler_keys
        or set(hits) != expected_candidate_keys
        or set(collectives) != expected_candidate_keys
    ):
        raise ValueError("candidate hit evidence inventory mismatch")
    startup_by_epoch = {}
    for key in expected_scheduler_keys:
        scheduler_row = scheduler[key]
        request_row = request_rows.get((
            key[0],
            EPOCH_ARMS[key[0]],
            key[1],
            key[2],
        ))
        started = scheduler_row.get("startup_model_load_started_ns")
        finished = scheduler_row.get("startup_model_load_finished_ns")
        duration = scheduler_row.get("startup_model_load_duration_ns")
        steps = scheduler_row.get("steps")
        if (
            request_row is None
            or scheduler_row.get("source_revision") != source_revision
            or scheduler_row.get("model_revision") != model_revision
            or scheduler_row.get("arm") != EPOCH_ARMS[key[0]]
            or scheduler_row.get("request_set_digest")
            != request_row.get("request_set_digest")
            or isinstance(started, bool)
            or not isinstance(started, int)
            or started < 0
            or isinstance(finished, bool)
            or not isinstance(finished, int)
            or finished < started
            or isinstance(duration, bool)
            or not isinstance(duration, int)
            or duration != finished - started
            or not isinstance(steps, list)
            or not steps
        ):
            raise ValueError("scheduler timing evidence is invalid")
        previous = startup_by_epoch.setdefault(key[0], (started, finished))
        if previous != (started, finished):
            raise ValueError("startup/model-load evidence is inconsistent")
        expected_request_ids = {
            request["request_id"] for request in request_row["requests"]
        }
        first_scheduled = {}
        for step_index, step in enumerate(steps):
            step_start = step.get("step_start_ns")
            step_end = step.get("step_end_ns")
            step_duration = step.get("step_duration_ns")
            host_submission = step.get("host_submission_ns")
            request_ids = step.get("request_ids")
            if (
                step.get("step_index") != step_index
                or isinstance(step_start, bool)
                or not isinstance(step_start, int)
                or step_start < 0
                or isinstance(step_end, bool)
                or not isinstance(step_end, int)
                or step_end < step_start
                or isinstance(step_duration, bool)
                or not isinstance(step_duration, int)
                or step_duration != step_end - step_start
                or isinstance(host_submission, bool)
                or not isinstance(host_submission, int)
                or host_submission < 0
                or host_submission > step_duration
                or not isinstance(request_ids, list)
                or not set(request_ids).issubset(expected_request_ids)
            ):
                raise ValueError("scheduler timing evidence is invalid")
            for request_id in request_ids:
                first_scheduled.setdefault(request_id, step_start)
        if set(first_scheduled) != expected_request_ids:
            raise ValueError("scheduler request inventory is incomplete")
        for request in request_row["requests"]:
            if (
                request.get("first_scheduled_ns")
                != first_scheduled[request["request_id"]]
            ):
                raise ValueError("request queueing evidence is invalid")
    for key in expected_candidate_keys:
        scheduler_row = scheduler[key]
        hit = hits[key]
        collective = collectives[key]
        request_row = request_rows.get((
            key[0],
            "candidate",
            key[1],
            key[2],
        ))
        if (
            request_row is None
            or any(
                row.get("source_revision") != source_revision
                or row.get("model_revision") != model_revision
                or row.get("arm") != "candidate"
                for row in (scheduler_row, hit, collective)
            )
            or scheduler_row.get("request_set_digest")
            != hit.get("request_set_digest")
            or scheduler_row.get("request_set_digest")
            != collective.get("request_set_digest")
            or scheduler_row.get("request_set_digest")
            != request_row.get("request_set_digest")
        ):
            raise ValueError("candidate evidence identity mismatch")
        segments = scheduler_row.get("token_one_segments")
        if (
            scheduler_row.get("decode_steps") != 127
            or isinstance(segments, bool)
            or not isinstance(segments, int)
            or segments <= 0
        ):
            raise ValueError("candidate scheduler evidence is invalid")
        if (
            hit.get("tp2_decode_calls") != segments * 48
            or hit.get("recurrent_token_one_calls") != segments * 48
            or hit.get("global_tp4_linear_decode_all_reduce_calls") != 0
            or hit.get("full_attention_tp4_collective_calls")
            != segments * 16
            or hit.get("migration_publications") != segments // 127
            or hit.get("fallback_calls") != 0
            or hit.get("post_warmup_request_path_allocations") != 0
            or hit.get("retry_after_mutation_calls") != 0
            or hit.get("duplicate_commit_calls") != 0
        ):
            raise RuntimeError("candidate hit counts are invalid")
        if hit.get("short_chunk_calls") != 0:
            raise RuntimeError("short-chunk calls must be zero")
        if hit.get("ordinary_chunk_calls") != 0:
            raise RuntimeError("ordinary-chunk calls must be zero")
        if (
            collective.get("pair_local_calls") != segments * 48
            or collective.get("pair_local_bytes")
            != segments * 48 * 5120 * 4
            or collective.get("full_attention_tp4_calls")
            != segments * 16
            or collective.get("full_attention_tp4_bytes")
            != segments * 16 * 5120 * 2
            or collective.get("pair_local_sequence_match") is not True
        ):
            raise RuntimeError("candidate collective evidence mismatch")
    return True


def _validate_resources(
    rows,
    source_revision,
    model_revision,
    expected_gpu_by_index,
    attempt_tag,
):
    valid = True
    boundary_stages = set()
    runtime_indices = {
        label: set() for label in RUNTIME_RESOURCE_LABELS
    }
    observed_stages = set()
    for row in rows:
        inventory = row.get("gpu_inventory") if isinstance(row, dict) else None
        processes = row.get("process_rows") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or row.get("source_revision") != source_revision
            or row.get("model_revision") != model_revision
            or row.get("attempt_tag") != attempt_tag
            or not isinstance(inventory, list)
            or not isinstance(processes, list)
            or any(not isinstance(item, dict) for item in inventory)
            or any(not isinstance(item, dict) for item in processes)
        ):
            raise ValueError("resource sample identity mismatch")
        observed = {}
        for item in inventory:
            gpu_index = item.get("gpu_index")
            if (
                isinstance(gpu_index, bool)
                or not isinstance(gpu_index, int)
                or gpu_index in observed
            ):
                raise ValueError("resource sample identity mismatch")
            observed[gpu_index] = item
        selected = [
            observed.get(gpu_index)
            for gpu_index in expected_gpu_by_index
        ]
        identity_match = all(
            isinstance(item, dict)
            and item.get("gpu_uuid") == expected_gpu_by_index[gpu_index]
            for gpu_index, item in zip(expected_gpu_by_index, selected)
        )
        telemetry_valid = identity_match and all(
            isinstance(item.get("memory_used_mib"), int)
            and not isinstance(item.get("memory_used_mib"), bool)
            and item["memory_used_mib"] >= 0
            and isinstance(item.get("utilization_percent"), int)
            and not isinstance(item.get("utilization_percent"), bool)
            and 0 <= item["utilization_percent"] <= 100
            and isinstance(item.get("power_watts"), (int, float))
            and not isinstance(item.get("power_watts"), bool)
            and math.isfinite(float(item["power_watts"]))
            and item["power_watts"] >= 0
            and isinstance(item.get("compute_processes"), list)
            for item in selected
        )
        strict_clean = telemetry_valid and all(
            item["memory_used_mib"] <= 1024
            and item["utilization_percent"] <= 5
            and item["compute_processes"] == []
            for item in selected
        )
        foreign_processes = [
            item
            for item in processes
            if item.get("attempt_tag") != attempt_tag
        ]
        if (
            row.get("identity_match") is not identity_match
            or row.get("strict_clean") is not strict_clean
            or row.get("foreign_processes") != foreign_processes
        ):
            raise ValueError("resource sample summary mismatch")
        stage = row.get("stage")
        scope = row.get("measurement_scope")
        if not isinstance(stage, str) or stage in observed_stages:
            raise ValueError("required resource sample inventory mismatch")
        observed_stages.add(stage)
        if scope == "boundary":
            if (
                stage not in BOUNDARY_RESOURCE_STAGES
                or row.get("run_label") is not None
                or row.get("sample_index") is not None
            ):
                raise ValueError("boundary resource sample inventory mismatch")
            boundary_stages.add(stage)
        elif scope == "runtime":
            run_label = row.get("run_label")
            sample_index = row.get("sample_index")
            if (
                run_label not in RUNTIME_RESOURCE_LABELS
                or isinstance(sample_index, bool)
                or not isinstance(sample_index, int)
                or sample_index < 0
                or stage != f"runtime_{run_label}_{sample_index:04d}"
                or sample_index in runtime_indices[run_label]
            ):
                raise ValueError("runtime resource sample inventory mismatch")
            runtime_indices[run_label].add(sample_index)
        else:
            raise ValueError("resource sample scope mismatch")
        valid = (
            valid
            and identity_match
            and telemetry_valid
            and (scope == "runtime" or strict_clean)
            and not foreign_processes
        )
    if boundary_stages != BOUNDARY_RESOURCE_STAGES:
        raise ValueError("required resource sample inventory mismatch")
    if any(
        indices != set(range(len(indices))) or not indices
        for indices in runtime_indices.values()
    ):
        raise ValueError("runtime resource sample inventory mismatch")
    return valid


def _service_metrics(requests):
    timing_bounds = []
    ttfts = []
    tpots = []
    output_tokens = 0
    for request in requests:
        timing_bounds.append(_validate_request_timing(request))
        ttfts.append(float(request["ttft_ns"]))
        tpots.append(float(request["tpot_ns"]))
        output_tokens += len(request["output_token_ids"])
    makespan = (
        max(completion for _, completion in timing_bounds)
        - min(admitted for admitted, _ in timing_bounds)
    )
    if makespan <= 0:
        raise ValueError("service-control makespan is invalid")
    return {
        "request_count": len(requests),
        "output_token_count": output_tokens,
        "cohort_makespan_ns": makespan,
        "request_qps": len(requests) * 1e9 / makespan,
        "output_tokens_per_second": output_tokens * 1e9 / makespan,
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
    }


def _validate_service_metric_summary(container, expected):
    for field in (
        "request_count",
        "output_token_count",
        "cohort_makespan_ns",
        "request_qps",
        "output_tokens_per_second",
    ):
        value = container.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not math.isclose(
                float(value),
                float(expected[field]),
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                "service-control reported metrics do not match raw evidence"
            )
    for field in ("ttft_ns", "tpot_ns"):
        reported = container.get(field)
        if not isinstance(reported, dict) or set(reported) != {
            "p50",
            "p95",
            "p99",
        }:
            raise ValueError("service-control distribution is invalid")
        if any(
            isinstance(reported[key], bool)
            or not isinstance(reported[key], (int, float))
            or not math.isfinite(float(reported[key]))
            or not math.isclose(
                float(reported[key]),
                float(expected[field][key]),
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
            for key in ("p50", "p95", "p99")
        ):
            raise ValueError(
                "service-control distribution does not match raw evidence"
            )


def _service_output_identity(requests):
    return {
        request.get("request_id"): (
            request.get("output_token_ids"),
            request.get("stop_position"),
            request.get("stop_reason"),
            request.get("decoded_text_sha256"),
        )
        for request in requests
    }


def _validate_service_rows(
    rows,
    request_index,
    expected_pair_devices,
    source_revision,
    model_revision,
):
    indexed = {}
    summaries = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("service-control row is invalid")
        workload = row.get("workload_id")
        if workload in indexed:
            raise ValueError("service-control identity is duplicated")
        indexed[workload] = row
        if (
            workload not in ONLINE_WORKLOADS
            or row.get("source_revision") != source_revision
            or row.get("model_revision") != model_revision
            or row.get("arm") != "TP2_X2_SERVICE_CONTROL"
            or row.get("classification_authority") is not False
        ):
            raise ValueError("service-control evidence is invalid")
        shape = WORKLOAD_SHAPES[workload]
        baseline_rows = [
            request_index[(epoch, "baseline", workload, 0)]
            for epoch in (0, 3)
        ]
        if (
            row.get("request_set_digest")
            != baseline_rows[0]["request_set_digest"]
            or baseline_rows[0]["request_set_digest"]
            != baseline_rows[1]["request_set_digest"]
        ):
            raise ValueError(
                "service-control request-set digest mismatch"
            )
        replicas = row.get("replicas")
        requests = row.get("requests")
        if (
            not isinstance(replicas, list)
            or len(replicas) != 2
            or not isinstance(requests, list)
            or len(requests) != shape[2]
        ):
            raise ValueError("service-control inventory mismatch")
        baseline_requests = baseline_rows[0]["requests"]
        baseline_identity = _service_output_identity(baseline_requests)
        if (
            baseline_identity
            != _service_output_identity(baseline_rows[1]["requests"])
        ):
            raise ValueError("service-control TP4 baseline is unstable")
        flattened = []
        replica_metrics = []
        peak_memory_by_gpu = []
        for replica_index, replica in enumerate(replicas):
            expected_requests = baseline_requests[replica_index::2]
            replica_requests = replica.get("requests")
            if (
                replica.get("replica_index") != replica_index
                or replica.get("pair_devices")
                != list(expected_pair_devices[replica_index])
                or replica.get("replica_tensor_parallel_size") != 2
                or not isinstance(replica_requests, list)
                or [
                    request.get("request_id")
                    for request in replica_requests
                ] != [
                    request["request_id"]
                    for request in expected_requests
                ]
            ):
                raise ValueError(
                    "service-control replica identity or split is invalid"
                )
            for request in replica_requests:
                if (
                    request.get("complete") is not True
                    or request.get("prompt_tokens") != shape[0]
                    or request.get("generated_tokens") != shape[1]
                    or request.get("stop_position") != 128
                    or request.get("stop_reason") != "length"
                    or not isinstance(
                        request.get("output_token_ids"),
                        list,
                    )
                    or len(request["output_token_ids"]) != 128
                    or not isinstance(request.get("decoded_text"), str)
                    or request.get("decoded_text_sha256")
                    != hashlib.sha256(
                        request["decoded_text"].encode("utf-8")
                    ).hexdigest()
                ):
                    raise ValueError(
                        "service-control request evidence is invalid"
                    )
            expected_metrics = _service_metrics(replica_requests)
            _validate_service_metric_summary(replica, expected_metrics)
            memory = replica.get("memory")
            reported_memory = replica.get("peak_memory_by_gpu")
            if (
                not isinstance(memory, list)
                or not isinstance(reported_memory, list)
                or len(memory) != 2
                or len(reported_memory) != 2
            ):
                raise ValueError(
                    "service-control memory inventory is invalid"
                )
            memory_by_rank = {}
            for memory_row in memory:
                rank = memory_row.get("rank")
                values = tuple(
                    memory_row.get(field)
                    for field in (
                        "cuda_peak_allocated_bytes",
                        "cuda_peak_reserved_bytes",
                        "physical_memory_bytes",
                    )
                )
                if (
                    isinstance(rank, bool)
                    or not isinstance(rank, int)
                    or rank in memory_by_rank
                    or rank not in (0, 1)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        for value in values
                    )
                    or values[0] < 0
                    or values[1] < values[0]
                    or values[2] <= 0
                    or values[1] > values[2]
                ):
                    raise ValueError(
                        "service-control memory evidence is invalid"
                    )
                memory_by_rank[rank] = memory_row
            expected_memory = [{
                "gpu_index": expected_pair_devices[replica_index][rank],
                "peak_allocated_bytes": memory_by_rank[rank][
                    "cuda_peak_allocated_bytes"
                ],
                "peak_reserved_bytes": memory_by_rank[rank][
                    "cuda_peak_reserved_bytes"
                ],
                "physical_memory_bytes": memory_by_rank[rank][
                    "physical_memory_bytes"
                ],
            } for rank in (0, 1)] if set(memory_by_rank) == {0, 1} else []
            if reported_memory != expected_memory:
                raise ValueError(
                    "service-control peak memory does not match raw evidence"
                )
            flattened.extend(replica_requests)
            replica_metrics.append(expected_metrics)
            peak_memory_by_gpu.extend(expected_memory)
        if requests != flattened:
            raise ValueError(
                "service-control flattened request evidence is invalid"
            )
        expected_metrics = _service_metrics(requests)
        _validate_service_metric_summary(row, expected_metrics)
        expected_balance = {
            "request_counts": [
                metrics["request_count"] for metrics in replica_metrics
            ],
            "request_qps_max_to_min": (
                max(metrics["request_qps"] for metrics in replica_metrics)
                / min(metrics["request_qps"] for metrics in replica_metrics)
            ),
            "output_tokens_per_second_max_to_min": (
                max(
                    metrics["output_tokens_per_second"]
                    for metrics in replica_metrics
                )
                / min(
                    metrics["output_tokens_per_second"]
                    for metrics in replica_metrics
                )
            ),
        }
        if row.get("replica_balance") != expected_balance:
            raise ValueError(
                "service-control replica balance is invalid"
            )
        parity = (
            _service_output_identity(requests) == baseline_identity
            and len(_service_output_identity(requests)) == len(requests)
        )
        if row.get("global_tp4_baseline_output_parity") is not parity:
            raise ValueError(
                "service-control output parity is invalid"
            )
        summaries[workload] = {
            **expected_metrics,
            "replica_balance": expected_balance,
            "replicas": replica_metrics,
            "peak_memory_by_gpu": peak_memory_by_gpu,
            "global_tp4_baseline_output_parity": parity,
        }
    if set(indexed) != set(ONLINE_WORKLOADS):
        raise ValueError("service-control inventory mismatch")
    return summaries


def _validate_memory_rows(rows, source_revision, model_revision):
    expected = {
        (epoch, arm, rank)
        for epoch, arm in enumerate(EPOCH_ARMS)
        for rank in CORRECTNESS_RANKS
    }
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("memory evidence row is invalid")
        identity = (
            row.get("epoch"),
            row.get("arm"),
            row.get("rank"),
        )
        if identity in indexed:
            raise ValueError("memory evidence identity is duplicated")
        indexed[identity] = row
        values = tuple(
            row.get(field)
            for field in (
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "physical_memory_bytes",
            )
        )
        if (
            row.get("source_revision") != source_revision
            or row.get("model_revision") != model_revision
            or identity not in expected
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                for value in values
            )
            or values[0] < 0
            or values[1] < values[0]
            or values[2] <= 0
            or values[1] > values[2]
        ):
            raise ValueError("memory evidence is invalid")
    if set(indexed) != expected:
        raise ValueError("memory evidence inventory mismatch")
    return all(
        row["peak_allocated_bytes"] / row["physical_memory_bytes"] < 0.98
        for row in indexed.values()
    )


def _validate_migration_rows(
    rows,
    request_rows,
    source_revision,
    model_revision,
):
    expected = {
        (epoch, workload, repetition)
        for epoch, arm in enumerate(EPOCH_ARMS)
        if arm == "candidate"
        for workload in WORKLOADS
        for repetition in range(5)
    }
    baseline_for_candidate = {1: 3, 2: 0}
    indexed = {}
    passes = True
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("migration evidence row is invalid")
        key = (
            row.get("epoch"),
            row.get("workload_id"),
            row.get("repetition"),
        )
        if key in indexed:
            raise ValueError("migration evidence identity is duplicated")
        indexed[key] = row
        candidate = request_rows.get((
            row.get("epoch"),
            "candidate",
            row.get("workload_id"),
            row.get("repetition"),
        ))
        baseline = request_rows.get((
            baseline_for_candidate.get(row.get("epoch")),
            "baseline",
            row.get("workload_id"),
            row.get("repetition"),
        ))
        latency = row.get("latency_ns")
        temporary = row.get("temporary_live_tensors")
        reported_break_even = row.get("break_even_output_tokens")
        if (
            row.get("source_revision") != source_revision
            or row.get("model_revision") != model_revision
            or row.get("arm") != "candidate"
            or key not in expected
            or candidate is None
            or baseline is None
            or row.get("request_set_digest")
            != candidate.get("request_set_digest")
            or isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or latency < 0
            or isinstance(temporary, bool)
            or not isinstance(temporary, int)
            or temporary < 0
            or isinstance(reported_break_even, bool)
            or not isinstance(reported_break_even, (int, float))
            or not math.isfinite(float(reported_break_even))
            or reported_break_even < 0
        ):
            raise ValueError("migration evidence is invalid")
        candidate_tpot = statistics.mean(
            float(request["tpot_ns"])
            for request in candidate["requests"]
        )
        baseline_tpot = statistics.mean(
            float(request["tpot_ns"])
            for request in baseline["requests"]
        )
        savings = baseline_tpot - candidate_tpot
        break_even = (
            float(latency) / savings
            if savings > 0
            else 1e30
        )
        if not math.isclose(
            float(reported_break_even),
            break_even,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "migration evidence break-even does not match raw timing"
            )
        passes = (
            passes
            and break_even <= 32
            and temporary == 0
        )
    if set(indexed) != expected:
        raise ValueError("migration evidence inventory mismatch")
    return passes


def _validate_cleanup(cleanup):
    if not isinstance(cleanup, dict):
        raise RuntimeError("cleanup evidence is incomplete")
    for path in cleanup.get("task_paths", []):
        candidate = PurePosixPath(path)
        if (
            not candidate.is_absolute()
            or not candidate.is_relative_to(
                PurePosixPath(APPROVED_REMOTE_ROOT)
            )
        ):
            raise ValueError(
                "task path escapes approved remote root"
            )
    if cleanup.get("foreign_process_actions") != []:
        raise RuntimeError("cleanup modified a foreign process")
    raw_receipts = cleanup.get("worker_cleanup_receipts")
    if (
        not isinstance(raw_receipts, list)
        or len(raw_receipts) != len(EXPECTED_CLEANUP_WORKERS)
    ):
        raise RuntimeError("cleanup worker receipt inventory is incomplete")
    observed_labels = set()
    validated_rank_receipts = 0
    retained_generations = 0
    retained_tensors = 0
    for record in raw_receipts:
        if not isinstance(record, dict):
            raise RuntimeError("cleanup worker receipt is invalid")
        label = record.get("label")
        expected = EXPECTED_CLEANUP_WORKERS.get(label)
        if expected is None or label in observed_labels:
            raise RuntimeError("cleanup worker receipt inventory is invalid")
        observed_labels.add(label)
        candidate_enabled, rank_count = expected
        receipt = record.get("receipt")
        cleanup_started_ns = (
            receipt.get("cleanup_started_ns")
            if isinstance(receipt, dict)
            else None
        )
        cleanup_finished_ns = (
            receipt.get("cleanup_finished_ns")
            if isinstance(receipt, dict)
            else None
        )
        cleanup_duration_ns = (
            receipt.get("cleanup_duration_ns")
            if isinstance(receipt, dict)
            else None
        )
        if (
            record.get("candidate_enabled") is not candidate_enabled
            or not isinstance(receipt, dict)
            or receipt.get("process_group_destroyed") is not True
            or receipt.get("rank_exit_codes") != [0] * rank_count
            or receipt.get("owned_children_remaining") != []
            or isinstance(cleanup_started_ns, bool)
            or not isinstance(cleanup_started_ns, int)
            or cleanup_started_ns < 0
            or isinstance(cleanup_finished_ns, bool)
            or not isinstance(cleanup_finished_ns, int)
            or cleanup_finished_ns < cleanup_started_ns
            or isinstance(cleanup_duration_ns, bool)
            or not isinstance(cleanup_duration_ns, int)
            or cleanup_duration_ns
            != cleanup_finished_ns - cleanup_started_ns
        ):
            raise RuntimeError("cleanup worker receipt is invalid")
        rank_receipts = receipt.get("rank_cleanup_receipts")
        if (
            not isinstance(rank_receipts, list)
            or len(rank_receipts) != rank_count
            or sorted(
                row.get("rank")
                for row in rank_receipts
                if isinstance(row, dict)
            ) != list(range(rank_count))
        ):
            raise RuntimeError("cleanup rank receipt inventory is invalid")
        for row in rank_receipts:
            if (
                not isinstance(row, dict)
                or row.get("process_group_destroyed") is not True
            ):
                raise RuntimeError("cleanup rank receipt is invalid")
            candidate = row.get("qwen38_topology_local_tp2_cleanup")
            if candidate_enabled:
                if (
                    not isinstance(candidate, dict)
                    or candidate.get("pair_groups_destroyed") != 2
                    or candidate.get("candidate_state_released") is not True
                    or isinstance(
                        candidate.get("published_generations_remaining"),
                        bool,
                    )
                    or not isinstance(
                        candidate.get("published_generations_remaining"),
                        int,
                    )
                    or candidate["published_generations_remaining"] < 0
                    or isinstance(
                        candidate.get("temporary_live_tensors"),
                        bool,
                    )
                    or not isinstance(
                        candidate.get("temporary_live_tensors"),
                        int,
                    )
                    or candidate["temporary_live_tensors"] < 0
                ):
                    raise RuntimeError(
                        "cleanup candidate rank receipt is invalid"
                    )
                retained_generations += candidate[
                    "published_generations_remaining"
                ]
                retained_tensors += candidate["temporary_live_tensors"]
            elif candidate is not None:
                raise RuntimeError(
                    "cleanup baseline rank receipt is invalid"
                )
        validated_rank_receipts += rank_count
    if observed_labels != set(EXPECTED_CLEANUP_WORKERS):
        raise RuntimeError("cleanup worker receipt inventory is incomplete")
    if (
        cleanup.get("complete") is not True
        or cleanup.get("retained_generations") != retained_generations
        or cleanup.get("retained_leases") != retained_generations
        or cleanup.get("retained_tensors") != retained_tensors
        or cleanup.get("retained_process_groups") != 0
        or cleanup.get("owned_processes_remaining") != []
        or cleanup.get("validated_worker_cleanups")
        != len(EXPECTED_CLEANUP_WORKERS)
        or cleanup.get("validated_rank_cleanup_receipts")
        != validated_rank_receipts
        or cleanup.get("cleanup_durations_ns") != [
            record["receipt"]["cleanup_duration_ns"]
            for record in raw_receipts
        ]
        or retained_generations != 0
        or retained_tensors != 0
    ):
        raise RuntimeError("cleanup evidence is incomplete")
    return True


def _metric_summary(
    request_rows,
    scheduler_rows,
    resource_rows,
    cleanup,
    selected_gpu_indices,
):
    def distribution(values):
        values = [float(value) for value in values]
        return {
            "p50": statistics.median(values),
            "p95": nearest_rank_percentile(values, 95),
            "p99": nearest_rank_percentile(values, 99),
        }

    scheduler_index = {
        (
            row["epoch"],
            row["arm"],
            row["workload_id"],
            row["repetition"],
        ): row
        for row in scheduler_rows
    }

    def arm_metrics(rows):
        requests = [
            request for row in rows for request in row["requests"]
        ]
        request_tpots = [
            float(request["tpot_ns"]) for request in requests
        ]
        gaps = [
            float(gap)
            for request in requests
            for gap in request["token_gaps_ns"]
        ]
        ttfts = [float(request["ttft_ns"]) for request in requests]
        e2e = [float(request["e2e_ns"]) for request in requests]
        queueing = [float(request["queueing_ns"]) for request in requests]
        matching_scheduler_rows = [
            scheduler_index[(
                row["epoch"],
                row["arm"],
                row["workload_id"],
                row["repetition"],
            )]
            for row in rows
        ]
        step_durations = [
            float(step["step_duration_ns"])
            for row in matching_scheduler_rows
            for step in row["steps"]
        ]
        host_submission = [
            float(step["host_submission_ns"])
            for row in matching_scheduler_rows
            for step in row["steps"]
        ]
        completion_spreads = [
            max(
                float(request["completion_ns"])
                for request in row["requests"]
            )
            - min(
                float(request["completion_ns"])
                for request in row["requests"]
            )
            for row in rows
        ]
        makespan = sum(float(row["cohort_makespan_ns"]) for row in rows)
        output_tokens = sum(
            int(request["generated_tokens"]) for request in requests
        )
        return {
            "median_tpot_ns": statistics.median(request_tpots),
            "gap_p95_ns": nearest_rank_percentile(gaps, 95),
            "gap_p99_ns": nearest_rank_percentile(gaps, 99),
            "ttft_median_ns": statistics.median(ttfts),
            "ttft_p99_ns": nearest_rank_percentile(ttfts, 99),
            "request_qps": len(requests) * 1e9 / makespan,
            "output_tokens_per_second": output_tokens * 1e9 / makespan,
            "request_tpot_ns": distribution(request_tpots),
            "token_gap_ns": distribution(gaps),
            "ttft_ns": distribution(ttfts),
            "e2e_ns": distribution(e2e),
            "queueing_ns": distribution(queueing),
            "scheduler_step_duration_ns": distribution(step_durations),
            "host_submission_ns": {
                "p50": statistics.median(host_submission),
                "p99": nearest_rank_percentile(host_submission, 99),
            },
            "completion_spread_ns": distribution(completion_spreads),
        }

    workload_metrics = {}
    ratios = []
    improving = 0
    regressions = []
    direction_counts = {}
    tail_ttft_pass = True
    throughput_pass = True
    for workload in WORKLOADS:
        arm_rows = {
            arm: [
                row
                for row in request_rows.values()
                if row["workload_id"] == workload
                and row["arm"] == arm
            ]
            for arm in ("baseline", "candidate")
        }
        values = {}
        for arm, rows in arm_rows.items():
            values[arm] = arm_metrics(rows)
        ratio = (
            values["candidate"]["median_tpot_ns"]
            / values["baseline"]["median_tpot_ns"]
        )
        ratios.append(ratio)
        if ratio < 1.0:
            improving += 1
        elif ratio > 1.0:
            regressions.append(workload)
        if (
            values["candidate"]["gap_p99_ns"]
            / values["baseline"]["gap_p99_ns"] > 1.02
            or values["candidate"]["ttft_median_ns"]
            / values["baseline"]["ttft_median_ns"] > 1.02
            or values["candidate"]["ttft_p99_ns"]
            / values["baseline"]["ttft_p99_ns"] > 1.02
        ):
            tail_ttft_pass = False
        if workload in ONLINE_WORKLOADS and (
            values["candidate"]["request_qps"]
            / values["baseline"]["request_qps"] < 0.98
        ):
            throughput_pass = False
        direction_count = 0
        for repetition in range(5):
            for baseline_epoch, candidate_epoch in ((0, 2), (3, 1)):
                baseline = request_rows[(
                    baseline_epoch,
                    "baseline",
                    workload,
                    repetition,
                )]
                candidate = request_rows[(
                    candidate_epoch,
                    "candidate",
                    workload,
                    repetition,
                )]
                baseline_median = statistics.median(
                    request["tpot_ns"]
                    for request in baseline["requests"]
                )
                candidate_median = statistics.median(
                    request["tpot_ns"]
                    for request in candidate["requests"]
                )
                direction_count += candidate_median < baseline_median
        direction_counts[workload] = direction_count
        workload_metrics[workload] = values
    aggregate = {
        arm: arm_metrics([
            row for row in request_rows.values() if row["arm"] == arm
        ])
        for arm in ("baseline", "candidate")
    }
    online_aggregate = {
        arm: arm_metrics([
            row
            for row in request_rows.values()
            if row["arm"] == arm
            and row["workload_id"] in ONLINE_WORKLOADS
        ])
        for arm in ("baseline", "candidate")
    }
    online_request_qps_ratio = (
        online_aggregate["candidate"]["request_qps"]
        / online_aggregate["baseline"]["request_qps"]
    )
    online_output_throughput_ratio = (
        online_aggregate["candidate"]["output_tokens_per_second"]
        / online_aggregate["baseline"]["output_tokens_per_second"]
    )
    throughput_pass = (
        throughput_pass
        and online_request_qps_ratio >= 0.98
        and online_output_throughput_ratio >= 0.98
    )
    aggregate_ratio = geometric_mean(ratios)
    startup_model_load_ns = {}
    for arm in ("baseline", "candidate"):
        by_epoch = {}
        for row in scheduler_rows:
            if row["arm"] == arm:
                by_epoch.setdefault(
                    row["epoch"],
                    float(row["startup_model_load_duration_ns"]),
                )
        startup_model_load_ns[arm] = distribution(by_epoch.values())
    utilization = []
    power = []
    for row in resource_rows:
        if row["measurement_scope"] != "runtime":
            continue
        for gpu in row["gpu_inventory"]:
            if gpu["gpu_index"] not in selected_gpu_indices:
                continue
            utilization.append(float(gpu["utilization_percent"]))
            power.append(float(gpu["power_watts"]))
    cleanup_durations = [
        float(record["receipt"]["cleanup_duration_ns"])
        for record in cleanup["worker_cleanup_receipts"]
    ]
    return {
        "workloads": workload_metrics,
        "aggregate": aggregate,
        "online_aggregate": online_aggregate,
        "online_request_qps_ratio": online_request_qps_ratio,
        "online_output_throughput_ratio": (
            online_output_throughput_ratio
        ),
        "aggregate_median_tpot_ratio": aggregate_ratio,
        "aggregate_median_tpot_improvement_percent": (
            (1.0 - aggregate_ratio) * 100.0
        ),
        "improving_workload_count": improving,
        "median_regressing_workloads": regressions,
        "pair_direction_counts": direction_counts,
        "tail_ttft_pass": tail_ttft_pass,
        "throughput_pass": throughput_pass,
        "raw_gap_p99_authority": True,
        "startup_model_load_ns": startup_model_load_ns,
        "gpu_utilization_percent": distribution(utilization),
        "gpu_power_watts": distribution(power),
        "cleanup_duration_ns": distribution(cleanup_durations),
    }


def _write_manifest(output_root: Path) -> None:
    files = {}
    for path in sorted(output_root.iterdir()):
        if (
            not path.is_file()
            or path.name in {"manifest.json", "manifest.sha256"}
        ):
            continue
        files[path.name] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    manifest_path = output_root / "manifest.json"
    _write_json(manifest_path, {
        "schema_version": MANIFEST_SCHEMA,
        "files": files,
    })
    (output_root / "manifest.sha256").write_text(
        _sha256(manifest_path) + "\n",
        encoding="utf-8",
    )


def _service_control_report(service_control) -> str:
    lines = [
        "## TP2 x2 service control (non-authoritative)",
        "",
        (
            "| workload | request QPS | output tokens/s | "
            "TTFT P50/P95/P99 (ms) | TPOT P50/P95/P99 (ms) | "
            "replica balance | peak memory by GPU | "
            "global TP4 output parity |"
        ),
        (
            "| --- | ---: | ---: | --- | --- | --- | --- | --- |"
        ),
    ]
    for workload in ONLINE_WORKLOADS:
        values = service_control[workload]
        ttft = values["ttft_ns"]
        tpot = values["tpot_ns"]
        balance = values["replica_balance"]
        memory = ", ".join(
            (
                f"GPU {row['gpu_index']}: "
                f"{row['peak_allocated_bytes'] / 1024**3:.2f}/"
                f"{row['peak_reserved_bytes'] / 1024**3:.2f} GiB "
                "(allocated/reserved)"
            )
            for row in values["peak_memory_by_gpu"]
        )
        lines.append(
            f"| {workload} | {values['request_qps']:.3f} | "
            f"{values['output_tokens_per_second']:.3f} | "
            f"{ttft['p50'] / 1e6:.3f}/"
            f"{ttft['p95'] / 1e6:.3f}/"
            f"{ttft['p99'] / 1e6:.3f} | "
            f"{tpot['p50'] / 1e6:.3f}/"
            f"{tpot['p95'] / 1e6:.3f}/"
            f"{tpot['p99'] / 1e6:.3f} | "
            f"requests={balance['request_counts']}; "
            f"QPS max/min={balance['request_qps_max_to_min']:.3f}; "
            "tokens/s max/min="
            f"{balance['output_tokens_per_second_max_to_min']:.3f} | "
            f"{memory} | "
            f"{values['global_tp4_baseline_output_parity']} |"
        )
    return "\n".join(lines) + "\n"


def _primary_report(
    summary,
    *,
    migration_rows,
    memory_rows,
    candidate_rows,
    collective_rows,
    weight_layout,
    cleanup,
) -> str:
    lines = [
        "## Primary A/B benefit and protected metrics",
        "",
        (
            "| workload | baseline/candidate request TPOT "
            "P50/P95/P99 (ms) | median improvement | "
            "baseline/candidate token-gap P99 (ms) | "
            "baseline/candidate TTFT P50/P95/P99 (ms) | "
            "baseline/candidate E2E P50/P95/P99 (ms) | "
            "baseline/candidate request QPS | "
            "baseline/candidate output tokens/s |"
        ),
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for workload in WORKLOADS:
        values = summary["workloads"][workload]
        baseline = values["baseline"]
        candidate = values["candidate"]
        improvement = (
            1.0
            - candidate["median_tpot_ns"]
            / baseline["median_tpot_ns"]
        ) * 100.0

        def triple(metric):
            return "/".join(
                f"{metric[key] / 1e6:.3f}"
                for key in ("p50", "p95", "p99")
            )

        lines.append(
            f"| {workload} | "
            f"{triple(baseline['request_tpot_ns'])} / "
            f"{triple(candidate['request_tpot_ns'])} | "
            f"{improvement:.3f}% | "
            f"{baseline['token_gap_ns']['p99'] / 1e6:.3f} / "
            f"{candidate['token_gap_ns']['p99'] / 1e6:.3f} | "
            f"{triple(baseline['ttft_ns'])} / "
            f"{triple(candidate['ttft_ns'])} | "
            f"{triple(baseline['e2e_ns'])} / "
            f"{triple(candidate['e2e_ns'])} | "
            f"{baseline['request_qps']:.3f} / "
            f"{candidate['request_qps']:.3f} | "
            f"{baseline['output_tokens_per_second']:.3f} / "
            f"{candidate['output_tokens_per_second']:.3f} |"
        )
    online = summary["online_aggregate"]
    lines.extend([
        "",
        (
            "Aggregate median TPOT improvement: "
            f"`{summary['aggregate_median_tpot_improvement_percent']:.6f}%`."
        ),
        (
            "Aggregate online request-QPS ratio "
            f"(candidate/baseline): "
            f"`{summary['online_request_qps_ratio']:.9f}` "
            f"({online['baseline']['request_qps']:.6f} -> "
            f"{online['candidate']['request_qps']:.6f})."
        ),
        (
            "Aggregate online output-throughput ratio "
            f"(candidate/baseline): "
            f"`{summary['online_output_throughput_ratio']:.9f}` "
            f"({online['baseline']['output_tokens_per_second']:.6f} -> "
            f"{online['candidate']['output_tokens_per_second']:.6f})."
        ),
        "",
        (
            "| workload | arm | Queueing P50/P95/P99 (ms) | "
            "Scheduler-step P50/P95/P99 (ms) | "
            "Host-submission P50/P99 (ms) |"
        ),
        "| --- | --- | --- | --- | --- |",
    ])
    for workload in WORKLOADS:
        for arm in ("baseline", "candidate"):
            values = summary["workloads"][workload][arm]

            def telemetry_triple(name):
                metric = values[name]
                return "/".join(
                    f"{metric[key] / 1e6:.3f}"
                    for key in ("p50", "p95", "p99")
                )

            host = values["host_submission_ns"]
            lines.append(
                f"| {workload} | {arm} | "
                f"{telemetry_triple('queueing_ns')} | "
                f"{telemetry_triple('scheduler_step_duration_ns')} | "
                f"{host['p50'] / 1e6:.3f}/"
                f"{host['p99'] / 1e6:.3f} |"
            )
    startup = summary["startup_model_load_ns"]
    utilization = summary["gpu_utilization_percent"]
    power = summary["gpu_power_watts"]
    cleanup_duration = summary["cleanup_duration_ns"]
    lines.extend([
        "",
        "## Measured mechanism and memory cost",
        "",
        (
            "- Startup/model-load duration P50/P95/P99 "
            "(baseline/candidate, ms): "
            f"{startup['baseline']['p50'] / 1e6:.3f}/"
            f"{startup['baseline']['p95'] / 1e6:.3f}/"
            f"{startup['baseline']['p99'] / 1e6:.3f} / "
            f"{startup['candidate']['p50'] / 1e6:.3f}/"
            f"{startup['candidate']['p95'] / 1e6:.3f}/"
            f"{startup['candidate']['p99'] / 1e6:.3f}."
        ),
        (
            "- Runtime GPU utilization P50/P95/P99 (%): "
            f"{utilization['p50']:.3f}/"
            f"{utilization['p95']:.3f}/"
            f"{utilization['p99']:.3f}."
        ),
        (
            "- Runtime GPU power P50/P95/P99 (W): "
            f"{power['p50']:.3f}/"
            f"{power['p95']:.3f}/"
            f"{power['p99']:.3f}."
        ),
        (
            "- Cleanup duration P50/P95/P99 (ms): "
            f"{cleanup_duration['p50'] / 1e6:.3f}/"
            f"{cleanup_duration['p95'] / 1e6:.3f}/"
            f"{cleanup_duration['p99'] / 1e6:.3f}."
        ),
    ])
    for workload in WORKLOADS:
        rows = [
            row
            for row in migration_rows
            if row["workload_id"] == workload
        ]
        latencies = [float(row["latency_ns"]) for row in rows]
        break_even = max(
            float(row["break_even_output_tokens"]) for row in rows
        )
        lines.append(
            f"- {workload} migration latency P50/P95/P99: "
            f"{statistics.median(latencies) / 1e6:.3f}/"
            f"{nearest_rank_percentile(latencies, 95) / 1e6:.3f}/"
            f"{nearest_rank_percentile(latencies, 99) / 1e6:.3f} ms; "
            f"worst observed break-even: {break_even:.3f} output tokens."
        )
    max_peak_ratio = max(
        float(row["peak_allocated_bytes"])
        / float(row["physical_memory_bytes"])
        for row in memory_rows
    )
    lines.extend([
        (
            "- Calculated steady-state increment per rank: "
            f"{weight_layout['steady_increment_bytes_per_rank'] / 1024**2:.3f} "
            "MiB."
        ),
        (
            "- Maximum measured peak-allocated/physical-memory ratio: "
            f"{max_peak_ratio:.9f}."
        ),
        (
            "- Candidate execution totals: "
            f"{sum(row['tp2_decode_calls'] for row in candidate_rows)} "
            "TP2 decode calls; "
            f"{sum(row['fallback_calls'] for row in candidate_rows)} "
            "fallback calls; "
            f"{sum(row['pair_local_calls'] for row in collective_rows)} "
            "pair-local collectives; "
            f"{sum(row['full_attention_tp4_calls'] for row in collective_rows)} "
            "full-attention TP4 collectives."
        ),
        (
            "- Cleanup proof: "
            f"{cleanup['validated_worker_cleanups']} worker receipts and "
            f"{cleanup['validated_rank_cleanup_receipts']} rank receipts; "
            "zero retained generations, leases, tensors, process groups, "
            "owned processes, and foreign-process actions."
        ),
        "",
        "## Claim boundary",
        "",
        "Production-default enablement: prohibited.",
        (
            "This fixed-cohort result does not establish dynamic-arrival, "
            "preemption, prefix-restore, abort, speculative-rollback, or "
            "mixed-layout continuous-batching safety."
        ),
        "",
    ])
    return "\n".join(lines)


def assemble_attempt(attempt_root: Path, output_root: Path) -> dict:
    attempt_root = Path(attempt_root)
    raw_root = (
        attempt_root / "raw"
        if (attempt_root / "raw").is_dir()
        else attempt_root
    )
    actual = {
        path.name for path in raw_root.iterdir() if path.is_file()
    }
    if actual != set(CORE_ARTIFACTS):
        raise ValueError("raw artifact inventory mismatch")
    payloads = {
        name: (
            _load_jsonl(raw_root / name)
            if name in JSONL_ARTIFACTS
            else _load_json(raw_root / name)
        )
        for name in CORE_ARTIFACTS
    }
    source_revision, model_revision = _validate_source_and_manifests(
        payloads
    )
    request_index = _validate_request_rows(
        payloads["request_rows.jsonl"],
        source_revision,
        model_revision,
    )
    correctness_pass = (
        _validate_correctness(
            payloads["correctness_rows.jsonl"],
            source_revision,
            model_revision,
        )
        and _validate_cleanup(payloads["cleanup.json"])
    )
    candidate_coverage_pass = _validate_candidate_coverage(
        payloads["scheduler_step_rows.jsonl"],
        payloads["candidate_hit_rows.jsonl"],
        payloads["collective_rows.jsonl"],
        request_index,
        source_revision,
        model_revision,
    )
    rank_mapping_rows = payloads["gpu_rank_manifest.json"].get("mapping")
    if not isinstance(rank_mapping_rows, list):
        raise ValueError("GPU rank mapping is invalid")
    gpu_index_by_rank = {
        row.get("rank"): row.get("gpu_index")
        for row in rank_mapping_rows
        if isinstance(row, dict)
    }
    if (
        set(gpu_index_by_rank) != set(CORRECTNESS_RANKS)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in gpu_index_by_rank.values()
        )
        or len(set(gpu_index_by_rank.values())) != len(CORRECTNESS_RANKS)
    ):
        raise ValueError("GPU rank mapping is invalid")
    resource_identity_pass = _validate_resources(
        payloads["resource_rows.jsonl"],
        source_revision,
        model_revision,
        {
            gpu_index_by_rank[rank]: rank_mapping_rows[rank]["gpu_uuid"]
            for rank in CORRECTNESS_RANKS
        },
        payloads["source_manifest.json"]["attempt_tag"],
    )
    service_control = _validate_service_rows(
        payloads["service_control_rows.jsonl"],
        request_index,
        (
            (gpu_index_by_rank[0], gpu_index_by_rank[1]),
            (gpu_index_by_rank[2], gpu_index_by_rank[3]),
        ),
        source_revision,
        model_revision,
    )

    migration_pass = _validate_migration_rows(
        payloads["migration_rows.jsonl"],
        request_index,
        source_revision,
        model_revision,
    )
    weight = payloads["weight_layout_manifest.json"]
    state = payloads["state_layout_manifest.json"]
    memory_rows = payloads["memory_rows.jsonl"]
    memory_capacity_pass = _validate_memory_rows(
        memory_rows,
        source_revision,
        model_revision,
    )
    memory_pass = (
        weight.get("baseline_tp4_decode_accumulation_retained") is False
        and weight.get("steady_increment_bytes_per_rank", math.inf)
        <= 1920 * 1024**2
        and state.get("temporary_objects_released") is True
        and memory_capacity_pass
    )
    metrics = _metric_summary(
        request_index,
        payloads["scheduler_step_rows.jsonl"],
        payloads["resource_rows.jsonl"],
        payloads["cleanup.json"],
        frozenset(gpu_index_by_rank.values()),
    )
    summary = {
        "schema_version": (
            "qwen38.topology-local-tp2-whole-model-classification.v1"
        ),
        "source_revision": source_revision,
        "source_tree_sha256": payloads[
            "source_manifest.json"
        ]["source_tree_sha256"],
        "model_revision": model_revision,
        "correctness_pass": correctness_pass,
        "resource_identity_pass": resource_identity_pass,
        "candidate_coverage_pass": candidate_coverage_pass,
        "memory_pass": memory_pass,
        "measurement_complete": True,
        "tail_ttft_pass": metrics["tail_ttft_pass"],
        "throughput_pass": metrics["throughput_pass"],
        "migration_pass": migration_pass,
        "service_control": service_control,
        **metrics,
    }
    summary["classification"] = classify(summary)

    output_root = Path(output_root)
    if output_root.exists():
        if any(output_root.iterdir()):
            raise ValueError("output bundle must be fresh")
    else:
        output_root.mkdir(parents=True)
    for name in CORE_ARTIFACTS:
        shutil.copyfile(raw_root / name, output_root / name)
    _write_json(output_root / "classification.json", summary)
    (output_root / "report.md").write_text(
        "# Qwen3.8 topology-local TP2 whole-model gate\n\n"
        f"Classification: `{summary['classification']}`\n\n"
        "The classification was reconstructed from raw request, "
        "scheduler, candidate-hit, collective, memory, migration, "
        "resource, service-control, and cleanup evidence.\n\n"
        + _primary_report(
            summary,
            migration_rows=payloads["migration_rows.jsonl"],
            memory_rows=memory_rows,
            candidate_rows=payloads["candidate_hit_rows.jsonl"],
            collective_rows=payloads["collective_rows.jsonl"],
            weight_layout=weight,
            cleanup=payloads["cleanup.json"],
        )
        + _service_control_report(service_control),
        encoding="utf-8",
    )
    _write_manifest(output_root)
    return summary


def main(
    argv=None,
    *,
    assemble=assemble_attempt,
    printer=print,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = assemble(args.attempt_root, args.output_root)
    printer(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
