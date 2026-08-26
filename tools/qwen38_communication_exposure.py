from __future__ import annotations

import json
from pathlib import Path
import statistics


SCHEMA_VERSION = "qwen38.communication-profile-row.v1"
WORKLOADS = {
    "P0": {
        "workload_family": "causal",
        "prompt_tokens": 256,
        "output_tokens": 128,
        "concurrency": 1,
    },
    "P1": {
        "workload_family": "causal",
        "prompt_tokens": 2048,
        "output_tokens": 128,
        "concurrency": 1,
    },
    "Q0": {
        "workload_family": "online",
        "prompt_tokens": 256,
        "output_tokens": 128,
        "concurrency": 4,
    },
    "Q1": {
        "workload_family": "online",
        "prompt_tokens": 256,
        "output_tokens": 128,
        "concurrency": 8,
    },
    "Q2": {
        "workload_family": "online",
        "prompt_tokens": 2048,
        "output_tokens": 128,
        "concurrency": 4,
    },
}
RANKS = (0, 1, 2, 3)
WARMUP_REPETITIONS = (0, 1)
MEASURED_REPETITIONS = (0, 1, 2, 3, 4)
GO_EXPOSURE_RATIO = 0.10
GO_HEADROOM_RATIO = 0.05
NO_GO_EXPOSURE_RATIO = 0.05
NO_GO_HEADROOM_RATIO = 0.02
MAX_PROFILER_OVERHEAD_RATIO = 0.03

_LAYER_METRICS = (
    "gemm_ns",
    "collective_ns",
    "compute_ns",
    "exposed_collective_ns",
    "compute_collective_overlap_ns",
    "gpu_idle_ns",
    "collective_count",
    "collective_bytes",
    "critical_path_ns",
)


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [
            json.loads(line)
            for line in handle
            if line.strip()
        ]


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _number(value, name, *, minimum=0.0, strictly_positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if strictly_positive and result <= minimum:
        raise ValueError(f"{name} must be greater than {minimum}")
    if not strictly_positive and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return result


def _sha256(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _non_empty_string(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _percentile(values, percentile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return (
        ordered[lower]
        + (ordered[upper] - ordered[lower]) * fraction
    )


def _percentiles(values):
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _validate_auxiliary_workloads(rows, artifact_name):
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("workload") not in WORKLOADS
        ):
            raise ValueError(
                f"{artifact_name} workload inventory mismatch"
            )


def _validate_operation_inventory(value):
    if not isinstance(value, list) or not value:
        raise ValueError("operation inventory must be a non-empty list")
    result = []
    ordinals = set()
    for operation in value:
        if not isinstance(operation, (list, tuple)) or len(operation) != 3:
            raise ValueError("operation inventory row is invalid")
        ordinal = _integer(operation[0], "operation ordinal")
        if ordinal in ordinals:
            raise ValueError("operation inventory contains duplicate ordinal")
        ordinals.add(ordinal)
        result.append((
            ordinal,
            _non_empty_string(operation[1], "operation class"),
            _non_empty_string(operation[2], "operation name"),
        ))
    if [row[0] for row in result] != sorted(ordinals):
        raise ValueError("operation inventory order mismatch")
    return tuple(result)


def _validate_collective_byte_inventory(value, operations):
    if not isinstance(value, list):
        raise ValueError("collective byte inventory must be a list")
    expected_ordinals = {
        operation[0]
        for operation in operations
        if operation[1] == "collective"
    }
    result = []
    seen = set()
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("collective byte inventory row is invalid")
        ordinal = _integer(row[0], "collective operation ordinal")
        byte_count = _integer(row[1], "collective bytes")
        if ordinal in seen:
            raise ValueError(
                "collective byte inventory contains duplicate ordinal"
            )
        seen.add(ordinal)
        result.append((ordinal, byte_count))
    if seen != expected_ordinals or [row[0] for row in result] != sorted(seen):
        raise ValueError("collective operation byte inventory mismatch")
    return tuple(result)


def _validate_layers(value):
    if not isinstance(value, list) or not value:
        raise ValueError("step layers must be a non-empty list")
    layers = []
    identities = set()
    for layer in value:
        if not isinstance(layer, dict):
            raise ValueError("layer row must be an object")
        layer_index = _integer(layer.get("layer_index"), "layer index")
        layer_role = _non_empty_string(
            layer.get("layer_role"),
            "layer role",
        )
        identity = (layer_index, layer_role)
        if identity in identities:
            raise ValueError("duplicate layer identity")
        identities.add(identity)
        normalized = dict(layer)
        normalized["layer_index"] = layer_index
        normalized["layer_role"] = layer_role
        operations = _validate_operation_inventory(
            layer.get("operation_inventory")
        )
        normalized["operation_inventory"] = operations
        byte_inventory = _validate_collective_byte_inventory(
            layer.get("collective_byte_inventory"),
            operations,
        )
        normalized["collective_byte_inventory"] = byte_inventory
        for metric in _LAYER_METRICS:
            normalized[metric] = _integer(
                layer.get(metric),
                metric,
            )
        normalized["step_critical_interval_ns"] = _integer(
            layer.get("step_critical_interval_ns"),
            "step_critical_interval_ns",
            minimum=1,
        )
        overlap = normalized["compute_collective_overlap_ns"]
        if overlap > normalized["compute_ns"]:
            raise ValueError("compute/collective overlap exceeds compute")
        if overlap > normalized["collective_ns"]:
            raise ValueError("compute/collective overlap exceeds collective")
        if (
            normalized["exposed_collective_ns"] + overlap
            != normalized["collective_ns"]
        ):
            raise ValueError("exposed collective arithmetic mismatch")
        if normalized["gemm_ns"] > normalized["compute_ns"]:
            raise ValueError("GEMM duration exceeds compute duration")
        if normalized["collective_count"] != len(byte_inventory):
            raise ValueError("collective count does not match inventory")
        if normalized["collective_bytes"] != sum(
            row[1] for row in byte_inventory
        ):
            raise ValueError("collective bytes do not match inventory")
        critical_interval_ns = normalized["step_critical_interval_ns"]
        for metric in (
            "collective_ns",
            "compute_ns",
            "exposed_collective_ns",
            "gpu_idle_ns",
            "critical_path_ns",
        ):
            if normalized[metric] > critical_interval_ns:
                raise ValueError(f"{metric} exceeds step critical interval")
        if (
            normalized["compute_ns"]
            + normalized["exposed_collective_ns"]
            > critical_interval_ns
        ):
            raise ValueError(
                "compute and exposed collective exceed step critical interval"
            )
        normalized["compute_collective_overlap_ns"] = overlap
        layers.append(normalized)
    layers.sort(key=lambda layer: (
        layer["layer_index"],
        layer["layer_role"],
    ))
    return layers


def _validate_steps(value):
    if not isinstance(value, list) or not value:
        raise ValueError("profile row steps must be a non-empty list")
    steps = []
    ordinals = set()
    for step in value:
        if not isinstance(step, dict):
            raise ValueError("step row must be an object")
        decode_ordinal = _integer(
            step.get("decode_ordinal"),
            "decode ordinal",
        )
        if decode_ordinal in ordinals:
            raise ValueError("duplicate decode ordinal")
        ordinals.add(decode_ordinal)
        critical_rank = _integer(
            step.get("critical_rank"),
            "critical rank",
        )
        if critical_rank not in RANKS:
            raise ValueError("critical rank is invalid")
        steps.append({
            "request_set_sha256": _sha256(
                step.get("request_set_sha256"),
                "request set digest",
            ),
            "decode_ordinal": decode_ordinal,
            "critical_rank": critical_rank,
            "final_required_offset_ns": _integer(
                step.get("final_required_offset_ns"),
                "final required offset",
                minimum=1,
            ),
            "layers": _validate_layers(step.get("layers")),
        })
    steps.sort(key=lambda step: step["decode_ordinal"])
    return steps


def _alignment_signature(row):
    return tuple(
        (
            step["request_set_sha256"],
            step["decode_ordinal"],
            step["critical_rank"],
            tuple(
                (
                    layer["layer_index"],
                    layer["layer_role"],
                    layer["operation_inventory"],
                )
                for layer in step["layers"]
            ),
        )
        for step in row["steps"]
    )


def _cross_repetition_signature(row):
    return tuple(
        (
            step["request_set_sha256"],
            step["decode_ordinal"],
            tuple(
                (
                    layer["layer_index"],
                    layer["layer_role"],
                    layer["operation_inventory"],
                )
                for layer in step["layers"]
            ),
        )
        for step in row["steps"]
    )


def validate_profile_rows(rows: list[dict]) -> dict:
    if not isinstance(rows, list) or not rows:
        raise ValueError("profile rows must be a non-empty list")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("profile row must be an object")
    normalized = []
    sequence_indices = set()
    failed_processes = set()
    attempts = set()
    source_identities = set()
    model_revisions = set()
    gpu_by_rank = {}
    for raw in sorted(rows, key=lambda row: row.get("sequence_index", -1)):
        sequence_index = _integer(
            raw.get("sequence_index"),
            "sequence index",
        )
        if sequence_index in sequence_indices:
            raise ValueError("duplicate sequence index")
        sequence_indices.add(sequence_index)
        process_identity = _non_empty_string(
            raw.get("process_identity"),
            "process identity",
        )
        if process_identity in failed_processes:
            raise ValueError(
                "process identity reused after failed finalization"
            )
        finalization_status = raw.get("finalization_status")
        if finalization_status == "failed":
            failed_processes.add(process_identity)
            continue
        if finalization_status != "complete":
            raise ValueError("finalization status is invalid")
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("profile row schema version mismatch")
        attempts.add(_non_empty_string(raw.get("attempt"), "attempt"))
        workload = raw.get("workload")
        if workload not in WORKLOADS:
            raise ValueError("workload inventory mismatch")
        contract = WORKLOADS[workload]
        for field, expected in contract.items():
            actual = raw.get(field)
            if isinstance(expected, int):
                try:
                    actual = _integer(actual, field, minimum=1)
                except ValueError as error:
                    raise ValueError(
                        f"workload contract mismatch for {workload}: {field}"
                    ) from error
            if actual != expected:
                raise ValueError(
                    f"workload contract mismatch for {workload}: {field}"
                )
        phase = raw.get("phase")
        if phase not in {"warmup", "measured"}:
            raise ValueError("profile phase is invalid")
        repetition = _integer(raw.get("repetition"), "repetition")
        rank = _integer(raw.get("rank"), "rank")
        if rank not in RANKS:
            raise ValueError("rank inventory mismatch")
        source_identity = _sha256(
            raw.get("source_tree_sha256"),
            "source_tree_sha256",
        )
        model_revision = _non_empty_string(
            raw.get("model_revision"),
            "model revision",
        )
        source_identities.add(source_identity)
        model_revisions.add(model_revision)
        gpu_uuid = _non_empty_string(raw.get("gpu_uuid"), "GPU UUID")
        previous_gpu = gpu_by_rank.setdefault(rank, gpu_uuid)
        if previous_gpu != gpu_uuid:
            raise ValueError("GPU UUID drift for rank")
        trace_coverage = raw.get("trace_coverage")
        if trace_coverage not in {
            "COMPLETE",
            "INCONCLUSIVE_TRACE_COVERAGE",
        }:
            raise ValueError("trace coverage status is invalid")
        normalized.append(dict(raw) | {
            "sequence_index": sequence_index,
            "phase": phase,
            "repetition": repetition,
            "rank": rank,
            "decode_time_ns": _integer(
                raw.get("decode_time_ns"),
                "decode_time_ns",
                minimum=1,
            ),
            "trace_coverage": trace_coverage,
            "steps": _validate_steps(raw.get("steps")),
        })
    if set(row["workload"] for row in normalized) != set(WORKLOADS):
        raise ValueError("workload inventory mismatch")
    if len(source_identities) != 1:
        raise ValueError("source identity drift")
    if len(model_revisions) != 1:
        raise ValueError("model revision drift")
    if len(attempts) != 1:
        raise ValueError("attempt identity drift")
    if set(gpu_by_rank) != set(RANKS):
        raise ValueError("rank inventory mismatch")
    if len(set(gpu_by_rank.values())) != len(RANKS):
        raise ValueError("ranks must map to four distinct GPU UUIDs")

    rows_by_case = {}
    for row in normalized:
        case_key = (
            row["workload"],
            row["phase"],
            row["repetition"],
        )
        rank_rows = rows_by_case.setdefault(case_key, {})
        if row["rank"] in rank_rows:
            raise ValueError("duplicate rank row")
        rank_rows[row["rank"]] = row
    for workload in WORKLOADS:
        for phase, expected_repetitions in (
            ("warmup", WARMUP_REPETITIONS),
            ("measured", MEASURED_REPETITIONS),
        ):
            actual_repetitions = {
                key[2]
                for key in rows_by_case
                if key[:2] == (workload, phase)
            }
            if actual_repetitions != set(expected_repetitions):
                raise ValueError(
                    f"{workload} {phase} repetition inventory mismatch"
                )
            for repetition in expected_repetitions:
                rank_rows = rows_by_case[
                    (workload, phase, repetition)
                ]
                if set(rank_rows) != set(RANKS):
                    raise ValueError(
                        f"{workload} {phase} repetition {repetition} "
                        "rank inventory mismatch"
                    )
                signatures = {
                    _alignment_signature(row)
                    for row in rank_rows.values()
                }
                if len(signatures) != 1:
                    reference = next(iter(rank_rows.values()))
                    reference_requests = tuple(
                        (
                            step["request_set_sha256"],
                            step["decode_ordinal"],
                        )
                        for step in reference["steps"]
                    )
                    request_signatures = {
                        tuple(
                            (
                                step["request_set_sha256"],
                                step["decode_ordinal"],
                            )
                            for step in row["steps"]
                        )
                        for row in rank_rows.values()
                    }
                    if len(request_signatures) != 1:
                        raise ValueError("request alignment mismatch")
                    raise ValueError("layer or operation alignment mismatch")
                reference = rank_rows[0]
                for step_index, step in enumerate(reference["steps"]):
                    computed_critical_rank = max(
                        RANKS,
                        key=lambda rank: (
                            rank_rows[rank]["steps"][step_index][
                                "final_required_offset_ns"
                            ],
                            rank,
                        ),
                    )
                    if step["critical_rank"] != computed_critical_rank:
                        raise ValueError(
                            "critical rank does not match timeline"
                        )
        measured_signatures = {
            _cross_repetition_signature(row)
            for row in normalized
            if row["workload"] == workload
            and row["phase"] == "measured"
        }
        if len(measured_signatures) != 1:
            raise ValueError(
                f"{workload} cross-repetition alignment mismatch"
            )
    return {
        "rows": sorted(normalized, key=lambda row: row["sequence_index"]),
        "workloads": list(WORKLOADS),
        "warmup_repetitions": list(WARMUP_REPETITIONS),
        "measured_repetitions": list(MEASURED_REPETITIONS),
        "rank_inventory": list(RANKS),
        "source_tree_sha256": next(iter(source_identities)),
        "model_revision": next(iter(model_revisions)),
        "gpu_uuids": [gpu_by_rank[rank] for rank in RANKS],
        "complete_four_rank_alignment": True,
        "trace_coverage_complete": all(
            row["trace_coverage"] == "COMPLETE"
            for row in normalized
            if row["phase"] == "measured"
        ),
    }


def select_representative_repetition(rows: list[dict]) -> int:
    if not isinstance(rows, list) or not rows:
        raise ValueError("representative selection requires rows")
    by_repetition = {}
    for row in rows:
        if row.get("phase") != "measured":
            raise ValueError("representative selection requires measured rows")
        repetition = _integer(row.get("repetition"), "repetition")
        rank = _integer(row.get("rank"), "rank")
        rank_times = by_repetition.setdefault(repetition, {})
        if rank in rank_times:
            raise ValueError("representative selection has duplicate rank")
        rank_times[rank] = _integer(
            row.get("decode_time_ns"),
            "decode_time_ns",
            minimum=1,
        )
    if set(by_repetition) != set(MEASURED_REPETITIONS):
        raise ValueError("representative selection needs five repetitions")
    critical_times = {}
    for repetition, rank_times in by_repetition.items():
        if set(rank_times) != set(RANKS):
            raise ValueError("representative selection needs four ranks")
        critical_times[repetition] = max(rank_times.values())
    target = statistics.median(critical_times.values())
    return min(
        critical_times,
        key=lambda repetition: (
            abs(critical_times[repetition] - target),
            repetition,
        ),
    )


def _repetition_metrics(rank_rows):
    exposed_ns = 0
    independent_compute_ns = 0
    critical_interval_ns = 0
    reference = rank_rows[0]
    for step_index, reference_step in enumerate(reference["steps"]):
        critical_rank = reference_step["critical_rank"]
        critical_step = rank_rows[critical_rank]["steps"][step_index]
        intervals = {
            layer["step_critical_interval_ns"]
            for layer in critical_step["layers"]
        }
        if len(intervals) != 1:
            raise ValueError("step critical interval mismatch across layers")
        critical_interval_ns += next(iter(intervals))
        for layer in critical_step["layers"]:
            exposed_ns += layer["exposed_collective_ns"]
            independent_compute_ns += (
                layer["compute_ns"]
                - layer["compute_collective_overlap_ns"]
            )
    if critical_interval_ns <= 0:
        raise ValueError("step critical interval must be positive")
    if (
        exposed_ns > critical_interval_ns
        or independent_compute_ns > critical_interval_ns
        or exposed_ns + independent_compute_ns > critical_interval_ns
    ):
        raise ValueError(
            "exposed or independent compute time exceeds critical interval"
        )
    return {
        "exposed_collective_ns": exposed_ns,
        "independent_compute_ns": independent_compute_ns,
        "step_critical_interval_ns": critical_interval_ns,
        "exposed_communication_ratio": (
            exposed_ns / critical_interval_ns
        ),
        "overlap_headroom_lower_bound": (
            min(exposed_ns, independent_compute_ns)
            / critical_interval_ns
        ),
    }


def _layer_summary(measured_rows):
    values = {}
    by_case = {}
    for row in measured_rows:
        by_case.setdefault(
            (row["workload"], row["repetition"]),
            {},
        )[row["rank"]] = row
    for rank_rows in by_case.values():
        reference = rank_rows[0]
        for step_index, reference_step in enumerate(reference["steps"]):
            critical_rank = reference_step["critical_rank"]
            critical_step = rank_rows[critical_rank]["steps"][step_index]
            for layer in critical_step["layers"]:
                key = (layer["layer_index"], layer["layer_role"])
                bucket = values.setdefault(
                    key,
                    {metric: [] for metric in _LAYER_METRICS},
                )
                for metric in _LAYER_METRICS:
                    bucket[metric].append(layer[metric])
    result = []
    for (layer_index, layer_role), bucket in sorted(values.items()):
        result.append({
            "layer_index": layer_index,
            "layer_role": layer_role,
        } | {
            f"median_{metric}": statistics.median(bucket[metric])
            for metric in _LAYER_METRICS
        })
    return result


def _online_summary(rows, workload, output_tokens):
    selected = [row for row in rows if row.get("workload") == workload]
    repetitions = [
        _integer(row.get("repetition"), "online repetition")
        for row in selected
    ]
    if (
        set(repetitions) != set(MEASURED_REPETITIONS)
        or len(selected) != len(MEASURED_REPETITIONS)
    ):
        raise ValueError(f"{workload} online metric inventory mismatch")
    qps = []
    output_tokens_per_s = []
    ttft = []
    tpot = []
    e2e = []
    for row in selected:
        request_count = _integer(
            row.get("request_count"),
            "request_count",
            minimum=1,
        )
        elapsed_s = _number(
            row.get("elapsed_s"),
            "elapsed_s",
            strictly_positive=True,
        )
        output_token_count = _integer(
            row.get("output_token_count"),
            "output token count",
            minimum=1,
        )
        if output_token_count != request_count * output_tokens:
            raise ValueError("output token count mismatch")
        qps.append(request_count / elapsed_s)
        output_tokens_per_s.append(output_token_count / elapsed_s)
        for field, destination in (
            ("ttft_ms", ttft),
            ("tpot_ms", tpot),
            ("e2e_latency_ms", e2e),
        ):
            values = row.get(field)
            if (
                not isinstance(values, list)
                or len(values) != request_count
            ):
                raise ValueError(
                    f"{field} count must match request_count"
                )
            destination.extend(
                _number(value, field)
                for value in values
            )
    return {
        "median_request_qps": statistics.median(qps),
        "median_output_tokens_per_s": statistics.median(
            output_tokens_per_s
        ),
        "ttft_ms": _percentiles(ttft),
        "tpot_ms": _percentiles(tpot),
        "e2e_latency_ms": _percentiles(e2e),
    }


def _memory_summary(rows, workload):
    selected = [row for row in rows if row.get("workload") == workload]
    allocated = {}
    reserved = {}
    inventory = set()
    for row in selected:
        repetition = _integer(
            row.get("repetition"),
            "memory repetition",
        )
        rank = _integer(row.get("rank"), "memory rank")
        if rank not in RANKS:
            raise ValueError("memory rank inventory mismatch")
        key = (repetition, rank)
        if key in inventory:
            raise ValueError("duplicate memory inventory row")
        inventory.add(key)
        allocated[rank] = max(
            allocated.get(rank, 0),
            _integer(
                row.get("peak_allocated_bytes"),
                "peak_allocated_bytes",
            ),
        )
        reserved[rank] = max(
            reserved.get(rank, 0),
            _integer(
                row.get("peak_reserved_bytes"),
                "peak_reserved_bytes",
            ),
        )
    expected_inventory = {
        (repetition, rank)
        for repetition in MEASURED_REPETITIONS
        for rank in RANKS
    }
    if inventory != expected_inventory:
        raise ValueError(f"{workload} memory rank inventory mismatch")
    return {
        "peak_allocated_bytes_by_rank": {
            str(rank): allocated[rank] for rank in RANKS
        },
        "peak_reserved_bytes_by_rank": {
            str(rank): reserved[rank] for rank in RANKS
        },
    }


def _resource_summary(rows, workload, gpu_uuids):
    selected = [row for row in rows if row.get("workload") == workload]
    if not selected:
        raise ValueError(f"{workload} resource samples are missing")
    expected_uuids = set(gpu_uuids)
    if any(row.get("gpu_uuid") not in expected_uuids for row in selected):
        raise ValueError("resource sample GPU identity mismatch")
    inventory = set()
    for row in selected:
        inventory.add((
            _integer(row.get("repetition"), "resource repetition"),
            row.get("gpu_uuid"),
        ))
    expected_inventory = {
        (repetition, gpu_uuid)
        for repetition in MEASURED_REPETITIONS
        for gpu_uuid in gpu_uuids
    }
    if (
        inventory != expected_inventory
        or len(selected) != len(expected_inventory)
    ):
        raise ValueError(f"{workload} resource sample inventory mismatch")
    utilization = [
        _number(
            row.get("gpu_utilization_percent"),
            "gpu_utilization_percent",
        )
        for row in selected
    ]
    power = [
        _number(row.get("power_watts"), "power_watts")
        for row in selected
    ]
    return {
        "gpu_utilization_percent": {
            "p50": _percentile(utilization, 0.50),
            "p95": _percentile(utilization, 0.95),
            "max": max(utilization),
        },
        "power_watts": {
            "p50": _percentile(power, 0.50),
            "p95": _percentile(power, 0.95),
            "max": max(power),
        },
    }


def _correctness_summary(rows):
    expected = {
        (workload, repetition, rank)
        for workload in WORKLOADS
        for repetition in MEASURED_REPETITIONS
        for rank in RANKS
    }
    actual = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("correctness row must be an object")
        actual.add((
            row.get("workload"),
            _integer(row.get("repetition"), "correctness repetition"),
            _integer(row.get("rank"), "correctness rank"),
        ))
    result = {
        "row_count": len(rows),
        "exact_token_match_rows": sum(
            row.get("exact_token_match") is True for row in rows
        ),
        "argmax_match_rows": sum(
            row.get("argmax_match") is True for row in rows
        ),
        "finite_logit_rows": sum(
            row.get("finite_logits") is True for row in rows
        ),
        "numeric_tolerance_rows": sum(
            row.get("within_numeric_tolerance") is True for row in rows
        ),
        "max_abs_logit_error": max(
            (
                _number(
                    row.get("max_abs_logit_error"),
                    "max_abs_logit_error",
                )
                for row in rows
            ),
            default=0.0,
        ),
        "max_rel_logit_error": max(
            (
                _number(
                    row.get("max_rel_logit_error"),
                    "max_rel_logit_error",
                )
                for row in rows
            ),
            default=0.0,
        ),
    }
    valid = (
        actual == expected
        and len(rows) == len(expected)
        and result["exact_token_match_rows"] == len(expected)
        and result["argmax_match_rows"] == len(expected)
        and result["finite_logit_rows"] == len(expected)
        and result["numeric_tolerance_rows"] == len(expected)
    )
    return result, valid


def _profiler_overhead(controls, validated):
    if not isinstance(controls, list):
        raise ValueError("overhead controls must be a list")
    expected = {
        (workload, repetition)
        for workload in WORKLOADS
        for repetition in MEASURED_REPETITIONS
    }
    actual = set()
    ratios = []
    for control in controls:
        key = (
            control.get("workload"),
            _integer(
                control.get("repetition"),
                "overhead repetition",
            ),
        )
        if key in actual:
            raise ValueError("duplicate overhead control")
        actual.add(key)
        if (
            control.get("source_tree_sha256")
            != validated["source_tree_sha256"]
            or control.get("model_revision")
            != validated["model_revision"]
            or control.get("rank_inventory") != list(RANKS)
            or control.get("gpu_uuids") != validated["gpu_uuids"]
        ):
            raise ValueError("overhead control identity mismatch")
        unprofiled = _number(
            control.get("unprofiled_ns"),
            "unprofiled_ns",
            strictly_positive=True,
        )
        profiled = _number(
            control.get("profiled_ns"),
            "profiled_ns",
            strictly_positive=True,
        )
        ratios.append(profiled / unprofiled - 1.0)
    if actual != expected:
        raise ValueError("overhead control inventory mismatch")
    return max(ratios)


def aggregate_profile_bundle(root: Path) -> dict:
    root = Path(root)
    required_paths = {
        "profile": root / "profile_rows.jsonl",
        "online": root / "online_metrics.json",
        "memory": root / "memory_summary.json",
        "resources": root / "resource_samples.jsonl",
        "correctness": root / "correctness_rows.jsonl",
    }
    missing = [
        name for name, path in required_paths.items() if not path.is_file()
    ]
    if missing:
        raise ValueError(f"profile bundle missing artifacts {missing}")
    validated = validate_profile_rows(
        _read_jsonl(required_paths["profile"])
    )
    online_payload = _read_json(required_paths["online"])
    memory_payload = _read_json(required_paths["memory"])
    if not isinstance(online_payload, dict):
        raise ValueError("online metrics must be an object")
    if not isinstance(memory_payload, dict):
        raise ValueError("memory summary must be an object")
    online_rows = online_payload.get("rows")
    memory_rows = memory_payload.get("rows")
    if not isinstance(online_rows, list):
        raise ValueError("online metric rows must be a list")
    if not isinstance(memory_rows, list):
        raise ValueError("memory rows must be a list")
    resource_rows = _read_jsonl(required_paths["resources"])
    correctness_rows = _read_jsonl(required_paths["correctness"])
    _validate_auxiliary_workloads(online_rows, "online metric")
    _validate_auxiliary_workloads(memory_rows, "memory")
    _validate_auxiliary_workloads(resource_rows, "resource sample")
    _validate_auxiliary_workloads(correctness_rows, "correctness")
    correctness, correctness_valid = _correctness_summary(
        correctness_rows
    )
    measured_rows = [
        row for row in validated["rows"] if row["phase"] == "measured"
    ]
    workloads = {}
    for workload, contract in WORKLOADS.items():
        workload_rows = [
            row for row in measured_rows if row["workload"] == workload
        ]
        by_repetition = {}
        for row in workload_rows:
            by_repetition.setdefault(row["repetition"], {})[
                row["rank"]
            ] = row
        repetitions = []
        for repetition in MEASURED_REPETITIONS:
            metrics = _repetition_metrics(by_repetition[repetition])
            repetitions.append({
                "repetition": repetition,
                **metrics,
            })
        workloads[workload] = {
            "workload_family": contract["workload_family"],
            "repetitions": repetitions,
            "median_exposed_communication_ratio": statistics.median(
                row["exposed_communication_ratio"]
                for row in repetitions
            ),
            "median_overlap_headroom_lower_bound": statistics.median(
                row["overlap_headroom_lower_bound"]
                for row in repetitions
            ),
            "representative_repetition": (
                select_representative_repetition(workload_rows)
            ),
            "layer_summary": _layer_summary(workload_rows),
            "online": _online_summary(
                online_rows,
                workload,
                contract["output_tokens"],
            ),
            "memory": _memory_summary(memory_rows, workload),
            "resources": _resource_summary(
                resource_rows,
                workload,
                validated["gpu_uuids"],
            ),
        }
    summary = {
        "schema_version": "qwen38.communication-exposure-summary.v1",
        "source_tree_sha256": validated["source_tree_sha256"],
        "model_revision": validated["model_revision"],
        "rank_inventory": list(RANKS),
        "gpu_uuids": validated["gpu_uuids"],
        "correctness": correctness,
        "correctness_valid": correctness_valid,
        "resource_identity_valid": True,
        "trace_coverage_complete": (
            validated["trace_coverage_complete"]
        ),
        "complete_four_rank_alignment": (
            validated["complete_four_rank_alignment"]
        ),
        "profiler_overhead_ratio": _profiler_overhead(
            online_payload.get("overhead_controls"),
            validated,
        ),
        "workloads": workloads,
    }
    summary["classification"] = classify_communication_exposure(summary)
    return summary


def _direction(ratio, headroom):
    if ratio >= GO_EXPOSURE_RATIO and headroom >= GO_HEADROOM_RATIO:
        return "GO"
    if (
        ratio < NO_GO_EXPOSURE_RATIO
        and headroom < NO_GO_HEADROOM_RATIO
    ):
        return "NO_GO"
    return "MIDDLE"


def classify_communication_exposure(summary: dict) -> str:
    if not isinstance(summary, dict):
        raise ValueError("communication exposure summary must be an object")
    if summary.get("correctness_valid") is not True:
        return "INVALID_CORRECTNESS"
    if summary.get("resource_identity_valid") is not True:
        return "INVALID_RESOURCE_IDENTITY"
    if (
        summary.get("trace_coverage_complete") is not True
        or summary.get("complete_four_rank_alignment") is not True
    ):
        return "INCONCLUSIVE_TRACE_COVERAGE"
    workloads = summary.get("workloads")
    if not isinstance(workloads, dict) or set(workloads) != set(WORKLOADS):
        raise ValueError("classification workload inventory mismatch")
    directions = {}
    for workload, contract in WORKLOADS.items():
        payload = workloads[workload]
        if payload.get("workload_family") != contract["workload_family"]:
            raise ValueError("classification workload family mismatch")
        repetitions = payload.get("repetitions")
        if not isinstance(repetitions, list):
            raise ValueError("classification repetition inventory mismatch")
        repetition_ids = [
            _integer(
                row.get("repetition"),
                "classification repetition",
            )
            for row in repetitions
        ]
        if (
            len(repetitions) != len(MEASURED_REPETITIONS)
            or set(repetition_ids) != set(MEASURED_REPETITIONS)
        ):
            raise ValueError("classification repetition inventory mismatch")
        ratios = [
            _number(
                row.get("exposed_communication_ratio"),
                "exposed communication ratio",
            )
            for row in repetitions
        ]
        headrooms = [
            _number(
                row.get("overlap_headroom_lower_bound"),
                "overlap headroom lower bound",
            )
            for row in repetitions
        ]
        if any(ratio > 1.0 for ratio in ratios):
            raise ValueError(
                "exposed communication ratio must not exceed 1"
            )
        if any(
            headroom > ratio
            for ratio, headroom in zip(ratios, headrooms)
        ):
            raise ValueError("headroom cannot exceed exposure")
        median_direction = _direction(
            statistics.median(ratios),
            statistics.median(headrooms),
        )
        agreement = sum(
            _direction(ratio, headroom) == median_direction
            for ratio, headroom in zip(ratios, headrooms)
        )
        if agreement < 4:
            return "INCONCLUSIVE_VARIANCE"
        directions[workload] = median_direction
    causal_go = any(
        directions[workload] == "GO"
        and WORKLOADS[workload]["workload_family"] == "causal"
        for workload in WORKLOADS
    )
    online_go = any(
        directions[workload] == "GO"
        and WORKLOADS[workload]["workload_family"] == "online"
        for workload in WORKLOADS
    )
    overhead = _number(
        summary.get("profiler_overhead_ratio"),
        "profiler overhead ratio",
    )
    if (
        causal_go
        and online_go
        and overhead <= MAX_PROFILER_OVERHEAD_RATIO
    ):
        return "GO_COMMUNICATION_OVERLAP"
    if all(
        direction == "NO_GO" for direction in directions.values()
    ):
        return "NO_GO_ALREADY_HIDDEN"
    return "INCONCLUSIVE_LOW_HEADROOM"
