from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import statistics


SCHEMA_VERSION = 2
GATE_NAME = "autoregressive_draft_exact_cuda_graph"
EXACT_CONFIGURATION = {
    "tensor_parallel_size": 4,
    "batch_size": 4,
    "exact_q": 4,
    "prompt_tokens": 256,
    "output_tokens": 16,
    "temperature": 0.0,
    "allocator_mode": "direct",
    "proposal_kv_offload": False,
    "in_process_warmup_runs": 1,
    "warmup_pairs": 2,
    "measured_pairs": 8,
}


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        canonical_json_bytes(value)
    ).hexdigest()


def _sha256(value, name: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(
            f"{name} must be lowercase hexadecimal"
        )
    return value


def _nonnegative_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a nonnegative integer"
        )
    return value


def _positive_number(value, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(
            f"{name} must be a positive finite number"
        )
    return float(value)


def _nonnegative_number(value, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(
            f"{name} must be a nonnegative finite number"
        )
    return float(value)


def _token_rows(value, name: str):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")

    def validate(node):
        if isinstance(node, list):
            if not node:
                raise ValueError(
                    f"{name} must not contain empty rows"
                )
            return [validate(item) for item in node]
        return _nonnegative_integer(node, name)

    return validate(value)


def _target_token_rows(value):
    rows = _token_rows(value, "target token rows")
    if (
        len(rows) != EXACT_CONFIGURATION["batch_size"]
        or any(
            not isinstance(row, list)
            or len(row) != EXACT_CONFIGURATION["output_tokens"]
            for row in rows
        )
    ):
        raise ValueError(
            "target token shape must be exact B4 x output16"
        )
    return rows


def _proposal_token_rows(value):
    calls = _token_rows(value, "proposal token rows")
    if any(
        not isinstance(call, list)
        or len(call) > EXACT_CONFIGURATION["batch_size"]
        or any(
            not isinstance(row, list)
            or len(row) > EXACT_CONFIGURATION["exact_q"]
            or any(not isinstance(token, int) for token in row)
            for row in call
        )
        for call in calls
    ):
        raise ValueError(
            "proposal token shape must contain at most B4 rows "
            "and at most Q4 tokens per row"
        )
    return calls


def _rank_graph_counters(value, *, graph: bool):
    if (
        not isinstance(value, list)
        or len(value) != 4
        or [row.get("rank") for row in value]
        != [0, 1, 2, 3]
    ):
        raise ValueError(
            "rank graph counters must contain ranks 0..3"
        )
    normalized = []
    for row in value:
        normalized_row = {"rank": row["rank"]}
        for name in (
            "capture_attempts",
            "captures",
            "replays",
            "quarantines",
            "fallback_pre_replay",
        ):
            normalized_row[name] = _nonnegative_integer(
                row.get(name),
                f"graph counter {name}",
            )
        if not graph and any(
            normalized_row[name] != 0
            for name in (
                "capture_attempts",
                "captures",
                "replays",
                "quarantines",
                "fallback_pre_replay",
            )
        ):
            raise ValueError(
                "eager row must not report graph counters"
            )
        normalized.append(normalized_row)
    return normalized


def _rank_graph_resources(value, *, graph: bool):
    if (
        not isinstance(value, list)
        or len(value) != 4
        or [row.get("rank") for row in value]
        != [0, 1, 2, 3]
    ):
        raise ValueError(
            "rank graph resources must contain ranks 0..3"
        )
    normalized = []
    for row in value:
        normalized_row = {"rank": row["rank"]}
        for name in (
            "ready_entry_count",
            "static_bytes",
            "reserved_bytes",
            "total_capture_ns",
        ):
            normalized_row[name] = _nonnegative_integer(
                row.get(name),
                f"graph resource {name}",
            )
        if not graph and any(
            normalized_row[name] != 0
            for name in (
                "ready_entry_count",
                "static_bytes",
                "reserved_bytes",
                "total_capture_ns",
            )
        ):
            raise ValueError(
                "eager row must not report graph resources"
            )
        normalized.append(normalized_row)
    return normalized


def _validate_steady_state_graph(
    *,
    graph: bool,
    warmup_counters,
    measured_counters,
    warmup_resources,
    measured_resources,
) -> None:
    if not graph:
        return
    for rank, (
        warmup_counter,
        measured_counter,
        warmup_resource,
        measured_resource,
    ) in enumerate(zip(
        warmup_counters,
        measured_counters,
        warmup_resources,
        measured_resources,
    )):
        if (
            warmup_counter["capture_attempts"] != 1
            or warmup_counter["captures"] != 1
            or measured_counter["capture_attempts"]
            != warmup_counter["capture_attempts"]
            or measured_counter["captures"]
            != warmup_counter["captures"]
        ):
            raise ValueError(
                "graph capture changed during measured replay "
                f"on rank {rank}"
            )
        if (
            warmup_counter["replays"] < 1
            or measured_counter["replays"]
            <= warmup_counter["replays"]
        ):
            raise ValueError(
                "graph replay did not increase during measured "
                f"run on rank {rank}"
            )
        for name in (
            "quarantines",
            "fallback_pre_replay",
        ):
            if (
                warmup_counter[name] != 0
                or measured_counter[name] != 0
            ):
                raise ValueError(
                    "graph steady-state failure counter "
                    f"{name} is nonzero on rank {rank}"
                )
        if (
            warmup_resource["ready_entry_count"] != 1
            or warmup_resource["static_bytes"] <= 0
            or warmup_resource["reserved_bytes"] <= 0
            or warmup_resource["total_capture_ns"] <= 0
            or measured_resource != warmup_resource
        ):
            raise ValueError(
                "graph capture resource changed during measured "
                f"replay on rank {rank}"
            )


def _rank_memory_rows(value):
    if (
        not isinstance(value, list)
        or len(value) != 4
        or [row.get("rank") for row in value]
        != [0, 1, 2, 3]
    ):
        raise ValueError(
            "memory rows must contain ranks 0..3"
        )
    return [
        {
            "rank": row["rank"],
            "peak_allocated_bytes": _nonnegative_integer(
                row.get("peak_allocated_bytes"),
                "memory peak allocated bytes",
            ),
            "peak_reserved_bytes": _nonnegative_integer(
                row.get("peak_reserved_bytes"),
                "memory peak reserved bytes",
            ),
        }
        for row in value
    ]


def _timing(value):
    if not isinstance(value, dict):
        raise ValueError("timing must be a mapping")
    detail = value.get("proposal_detail_ns")
    if not isinstance(detail, dict):
        raise ValueError(
            "proposal detail timing must be a mapping"
        )
    detail_names = (
        "setup",
        "backend_submit",
        "selection_collective",
        "decode_authority",
        "token_readback",
        "materialize_register",
    )
    if set(detail) != set(detail_names):
        raise ValueError(
            "proposal detail timing inventory mismatch"
        )
    return {
        "e2e_ns": _positive_number(
            value.get("e2e_ns"),
            "e2e timing",
        ),
        "throughput_tokens_per_second": _positive_number(
            value.get("throughput_tokens_per_second"),
            "throughput",
        ),
        "ttft_ns": _positive_number(
            value.get("ttft_ns"),
            "TTFT",
        ),
        "tpot_ns": _positive_number(
            value.get("tpot_ns"),
            "TPOT",
        ),
        "proposal_forward_ns": _positive_number(
            value.get("proposal_forward_ns"),
            "proposal forward timing",
        ),
        "proposal_detail_ns": {
            name: _nonnegative_number(
                detail[name],
                f"proposal detail {name}",
            )
            for name in detail_names
        },
    }


def _acceptance(value):
    if not isinstance(value, dict):
        raise ValueError("acceptance must be a mapping")
    proposed = _nonnegative_integer(
        value.get("proposed_tokens"),
        "proposed tokens",
    )
    accepted = _nonnegative_integer(
        value.get("accepted_tokens"),
        "accepted tokens",
    )
    if accepted > proposed:
        raise ValueError(
            "accepted tokens exceed proposed tokens"
        )
    accepted_per_call = _nonnegative_number(
        value.get("accepted_tokens_per_target_call"),
        "accepted tokens per target call",
    )
    rate = _nonnegative_number(
        value.get("rate"),
        "acceptance rate",
    )
    if rate > 1:
        raise ValueError("acceptance rate exceeds one")
    expected_rate = (
        accepted / proposed if proposed else 0.0
    )
    if not math.isclose(
        rate,
        expected_rate,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("acceptance rate mismatch")
    return {
        "proposed_tokens": proposed,
        "accepted_tokens": accepted,
        "accepted_tokens_per_target_call": accepted_per_call,
        "rate": rate,
    }


def _mode_row(value, expected_mode: str):
    if (
        not isinstance(value, dict)
        or value.get("mode") != expected_mode
    ):
        raise ValueError(
            f"{expected_mode} row mode mismatch"
        )
    graph = expected_mode == "graph"
    warmup_counters = _rank_graph_counters(
        value.get("warmup_rank_graph_counters"),
        graph=graph,
    )
    measured_counters = _rank_graph_counters(
        value.get("rank_graph_counters"),
        graph=graph,
    )
    warmup_resources = _rank_graph_resources(
        value.get("warmup_rank_graph_resources"),
        graph=graph,
    )
    measured_resources = _rank_graph_resources(
        value.get("rank_graph_resources"),
        graph=graph,
    )
    _validate_steady_state_graph(
        graph=graph,
        warmup_counters=warmup_counters,
        measured_counters=measured_counters,
        warmup_resources=warmup_resources,
        measured_resources=measured_resources,
    )
    return {
        "mode": expected_mode,
        "target_token_rows": _target_token_rows(
            value.get("target_token_rows")
        ),
        "proposal_token_rows": _proposal_token_rows(
            value.get("proposal_token_rows")
        ),
        "accepted_prefix_counts": [
            _nonnegative_integer(
                count,
                "accepted prefix count",
            )
            for count in value.get(
                "accepted_prefix_counts",
                (),
            )
        ],
        "transaction_digest": _sha256(
            value.get("transaction_digest"),
            "transaction digest",
        ),
        "active_transaction_count": _nonnegative_integer(
            value.get("active_transaction_count"),
            "active transaction count",
        ),
        "warmup_rank_graph_counters": warmup_counters,
        "rank_graph_counters": measured_counters,
        "warmup_rank_graph_resources": warmup_resources,
        "rank_graph_resources": measured_resources,
        "rank_memory_rows": _rank_memory_rows(
            value.get("rank_memory_rows")
        ),
        "timing": _timing(value.get("timing")),
        "acceptance": _acceptance(
            value.get("acceptance")
        ),
    }


def _pair(value, expected_index: int, *, warmup: bool):
    if not isinstance(value, dict):
        raise ValueError("pair must be a mapping")
    index_name = "warmup_index" if warmup else "pair_index"
    if value.get(index_name) != expected_index:
        raise ValueError(f"{index_name} is not canonical")
    result = {
        index_name: expected_index,
        "eager": _mode_row(value.get("eager"), "eager"),
        "graph": _mode_row(value.get("graph"), "graph"),
    }
    if not warmup:
        order = value.get("order")
        if order not in ("eager_graph", "graph_eager"):
            raise ValueError("pair order is invalid")
        result["order"] = order
    return result


def _bootstrap_ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    generator = random.Random(20260817)
    samples = []
    for _ in range(10_000):
        samples.append(statistics.mean(
            values[
                generator.randrange(len(values))
            ]
            for _ in values
        ))
    samples.sort()
    return (
        samples[int(0.025 * len(samples))],
        samples[int(0.975 * len(samples)) - 1],
    )


def _derive_summary(
    *,
    environment,
    warmups,
    pairs,
) -> dict:
    correctness_failures = []
    every_rank_replayed = True
    for pair in pairs:
        eager = pair["eager"]
        graph = pair["graph"]
        for field, label in (
            ("target_token_rows", "target token mismatch"),
            ("proposal_token_rows", "proposal token mismatch"),
            (
                "accepted_prefix_counts",
                "accepted prefix mismatch",
            ),
            (
                "transaction_digest",
                "transaction digest mismatch",
            ),
            ("acceptance", "acceptance mismatch"),
        ):
            if eager[field] != graph[field]:
                correctness_failures.append(
                    f"pair {pair['pair_index']} {label}"
                )
        if (
            eager["active_transaction_count"] != 0
            or graph["active_transaction_count"] != 0
        ):
            correctness_failures.append(
                f"pair {pair['pair_index']} active transaction leak"
            )
        for rank_row in graph["rank_graph_counters"]:
            if (
                rank_row["replays"] <= 0
                or rank_row["captures"] <= 0
                or rank_row["quarantines"] != 0
                or rank_row["fallback_pre_replay"] != 0
            ):
                every_rank_replayed = False
    order_counts = {
        order: sum(pair["order"] == order for pair in pairs)
        for order in ("eager_graph", "graph_eager")
    }
    eager_throughputs = [
        pair["eager"]["timing"][
            "throughput_tokens_per_second"
        ]
        for pair in pairs
    ]
    graph_throughputs = [
        pair["graph"]["timing"][
            "throughput_tokens_per_second"
        ]
        for pair in pairs
    ]
    eager_tpots = [
        pair["eager"]["timing"]["tpot_ns"]
        for pair in pairs
    ]
    graph_tpots = [
        pair["graph"]["timing"]["tpot_ns"]
        for pair in pairs
    ]
    throughput_deltas = [
        graph - eager
        for eager, graph in zip(
            eager_throughputs,
            graph_throughputs,
        )
    ]
    ci_low, ci_high = _bootstrap_ci(
        throughput_deltas
    )
    correctness_passed = not correctness_failures
    position_balanced = (
        len(pairs) >= 8
        and order_counts["eager_graph"]
        == order_counts["graph_eager"]
    )
    environment_inconclusive = bool(
        environment["interference_detected"]
    ) or not position_balanced
    median_eager_throughput = statistics.median(
        eager_throughputs
    )
    median_graph_throughput = statistics.median(
        graph_throughputs
    )
    median_eager_tpot = statistics.median(eager_tpots)
    median_graph_tpot = statistics.median(graph_tpots)
    if not correctness_passed or not every_rank_replayed:
        classification = "NO_GO_CORRECTNESS"
    elif environment_inconclusive:
        classification = "INCONCLUSIVE_ENVIRONMENT"
    elif (
        median_graph_throughput
        > median_eager_throughput
        and median_graph_tpot <= median_eager_tpot
        and ci_low > 0
    ):
        classification = "GO"
    else:
        classification = "NO_GO_PERFORMANCE"
    return {
        "classification": classification,
        "correctness_passed": correctness_passed,
        "correctness_failures": correctness_failures,
        "every_rank_replayed": every_rank_replayed,
        "measured_pair_count": len(pairs),
        "warmup_pair_count": len(warmups),
        "order_counts": order_counts,
        "position_balanced": position_balanced,
        "median_eager_throughput": (
            median_eager_throughput
        ),
        "median_graph_throughput": (
            median_graph_throughput
        ),
        "median_eager_tpot_ns": median_eager_tpot,
        "median_graph_tpot_ns": median_graph_tpot,
        "paired_throughput_delta_mean": (
            statistics.mean(throughput_deltas)
        ),
        "paired_throughput_delta_ci_low": ci_low,
        "paired_throughput_delta_ci_high": ci_high,
        "peak_eager_reserved_bytes": max(
            row["peak_reserved_bytes"]
            for pair in pairs
            for row in pair["eager"]["rank_memory_rows"]
        ),
        "peak_graph_reserved_bytes": max(
            row["peak_reserved_bytes"]
            for pair in pairs
            for row in pair["graph"]["rank_memory_rows"]
        ),
    }


def _provenance(value):
    if not isinstance(value, dict):
        raise ValueError("provenance must be a mapping")
    result = {
        "source_commit": _sha256(
            value.get("source_commit"),
            "source_commit",
            length=40,
        ),
        "source_patch_sha256": _sha256(
            value.get("source_patch_sha256"),
            "source_patch_sha256",
        ),
        "source_tree_sha256": _sha256(
            value.get("source_tree_sha256"),
            "source_tree_sha256",
        ),
        "target_model_fingerprint": _sha256(
            value.get("target_model_fingerprint"),
            "target_model_fingerprint",
        ),
        "draft_model_fingerprint": _sha256(
            value.get("draft_model_fingerprint"),
            "draft_model_fingerprint",
        ),
        "tokenizer_fingerprint": _sha256(
            value.get("tokenizer_fingerprint"),
            "tokenizer_fingerprint",
        ),
    }
    for name in (
        "python_version",
        "torch_version",
        "cuda_version",
        "nccl_version",
    ):
        field = value.get(name)
        if not isinstance(field, str) or not field:
            raise ValueError(f"{name} must be non-empty")
        result[name] = field
    gpu_uuids = value.get("gpu_uuids")
    if (
        not isinstance(gpu_uuids, list)
        or len(gpu_uuids) != 4
        or len(set(gpu_uuids)) != 4
        or any(
            not isinstance(gpu_uuid, str)
            or not gpu_uuid.startswith("GPU-")
            for gpu_uuid in gpu_uuids
        )
    ):
        raise ValueError(
            "gpu_uuids must contain four unique UUIDs"
        )
    result["gpu_uuids"] = list(gpu_uuids)
    return result


def _environment(value):
    if not isinstance(value, dict):
        raise ValueError("environment must be a mapping")
    host = value.get("host")
    if not isinstance(host, str) or not host:
        raise ValueError("environment host must be non-empty")
    interference = value.get("interference_detected")
    if not isinstance(interference, bool):
        raise ValueError(
            "environment interference flag must be bool"
        )
    for name in ("gpu_before", "gpu_after"):
        rows = value.get(name)
        if not isinstance(rows, list) or len(rows) != 4:
            raise ValueError(
                f"environment {name} must contain four rows"
            )
    return {
        "host": host,
        "interference_detected": interference,
        "gpu_before": copy.deepcopy(value["gpu_before"]),
        "gpu_after": copy.deepcopy(value["gpu_after"]),
    }


def build_gate_payload(
    *,
    provenance,
    environment,
    warmups,
    pairs,
) -> dict:
    normalized_provenance = _provenance(provenance)
    normalized_environment = _environment(environment)
    normalized_warmups = [
        _pair(row, index, warmup=True)
        for index, row in enumerate(warmups)
    ]
    normalized_pairs = [
        _pair(row, index, warmup=False)
        for index, row in enumerate(pairs)
    ]
    summary = _derive_summary(
        environment=normalized_environment,
        warmups=normalized_warmups,
        pairs=normalized_pairs,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "gate_name": GATE_NAME,
        "configuration": copy.deepcopy(
            EXACT_CONFIGURATION
        ),
        "provenance": normalized_provenance,
        "environment": normalized_environment,
        "warmups": normalized_warmups,
        "pairs": normalized_pairs,
        "summary": summary,
    }


def validate_gate_payload(payload) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("gate payload must be a mapping")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("gate schema version mismatch")
    if payload.get("gate_name") != GATE_NAME:
        raise ValueError("gate name mismatch")
    if payload.get("configuration") != EXACT_CONFIGURATION:
        raise ValueError("gate configuration mismatch")
    provenance = _provenance(payload.get("provenance"))
    environment = _environment(payload.get("environment"))
    warmup_rows = payload.get("warmups")
    pair_rows = payload.get("pairs")
    if (
        not isinstance(warmup_rows, list)
        or len(warmup_rows) < 2
    ):
        raise ValueError(
            "at least two warmup pairs are required"
        )
    if (
        not isinstance(pair_rows, list)
        or len(pair_rows) < 8
    ):
        raise ValueError(
            "at least eight measured pairs are required"
        )
    warmups = [
        _pair(row, index, warmup=True)
        for index, row in enumerate(warmup_rows)
    ]
    pairs = [
        _pair(row, index, warmup=False)
        for index, row in enumerate(pair_rows)
    ]
    derived = _derive_summary(
        environment=environment,
        warmups=warmups,
        pairs=pairs,
    )
    stored = payload.get("summary")
    if not isinstance(stored, dict):
        raise ValueError("summary must be a mapping")
    if not derived["position_balanced"]:
        raise ValueError(
            "measured pair order is not position balanced"
        )
    if (
        derived["correctness_failures"]
        and stored.get("classification")
        != "NO_GO_CORRECTNESS"
    ):
        raise ValueError(
            derived["correctness_failures"][0]
        )
    if (
        not derived["every_rank_replayed"]
        and stored.get("every_rank_replayed") is not False
    ):
        raise ValueError(
            "graph replay evidence mismatch"
        )
    if stored != derived:
        raise ValueError("summary aggregate mismatch")
    normalized_payload = {
        "schema_version": SCHEMA_VERSION,
        "gate_name": GATE_NAME,
        "configuration": copy.deepcopy(
            EXACT_CONFIGURATION
        ),
        "provenance": provenance,
        "environment": environment,
        "warmups": warmups,
        "pairs": pairs,
        "summary": derived,
    }
    if normalized_payload != payload:
        raise ValueError(
            "payload is not canonical"
        )
    return copy.deepcopy(derived)
