#!/usr/bin/env python3
"""Assemble and classify Qwen3.8 topology-local TP2 whole-model evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import statistics


WORKLOADS = ("P0", "P1", "Q0", "Q1", "Q2")
ONLINE_WORKLOADS = ("Q0", "Q1", "Q2")
EPOCH_ARMS = ("baseline", "candidate", "candidate", "baseline")
APPROVED_REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
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


def _validate_source_and_manifests(payloads):
    source = payloads["source_manifest.json"]
    model = payloads["model_manifest.json"]
    source_revision = source.get("source_revision")
    model_revision = source.get("model_revision")
    if (
        not isinstance(source_revision, str)
        or len(source_revision) != 40
        or not isinstance(model_revision, str)
        or len(model_revision) != 40
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
    epochs = payloads["campaign_epoch_manifest.json"].get("epochs")
    observed = tuple(
        row.get("arm")
        for row in sorted(epochs or [], key=lambda row: row.get("epoch", -1))
    )
    if observed != EPOCH_ARMS:
        raise ValueError("campaign epoch order must be A/B/B/A")
    if payloads["gpu_rank_manifest.json"].get("ranks") != list(range(4)):
        raise ValueError("GPU rank identity mismatch")
    if (
        payloads["pair_group_manifest.json"].get("pair_groups")
        != [[0, 1], [2, 3]]
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
        for request in requests:
            gaps = request.get("token_gaps_ns")
            if (
                not isinstance(request.get("output_token_ids"), list)
                or len(request["output_token_ids"]) != 128
                or not isinstance(gaps, list)
                or len(gaps) != 127
            ):
                raise ValueError(
                    "each request requires 128 tokens and 127 token gaps"
                )
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
    return indexed


def _validate_correctness(rows):
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
        "state_checkpoints_complete",
        "single_commit_per_step",
        "pair_replica_digest_match",
    )
    for row in rows:
        key = (row.get("workload_id"), row.get("repetition"))
        if key in indexed:
            raise ValueError("duplicate correctness row identity")
        indexed[key] = row
        if any(row.get(field) is not True for field in required_true):
            return False
    if set(indexed) != expected:
        raise ValueError("correctness row inventory mismatch")
    return True


def _validate_candidate_coverage(scheduler_rows, hit_rows, collective_rows):
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
    expected_keys = {
        (epoch, workload, repetition)
        for epoch, arm in enumerate(EPOCH_ARMS)
        if arm == "candidate"
        for workload in WORKLOADS
        for repetition in range(5)
    }
    if (
        set(scheduler) != expected_keys
        or set(hits) != expected_keys
        or set(collectives) != expected_keys
    ):
        raise ValueError("candidate hit evidence inventory mismatch")
    for key in expected_keys:
        scheduler_row = scheduler[key]
        hit = hits[key]
        collective = collectives[key]
        if (
            scheduler_row.get("request_set_digest")
            != hit.get("request_set_digest")
            or scheduler_row.get("request_set_digest")
            != collective.get("request_set_digest")
        ):
            raise ValueError("candidate request-set digest mismatch")
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


def _validate_resources(rows):
    required = {
        "entry",
        *(f"pre_epoch_{index}" for index in range(4)),
        *(f"post_launch_{index}" for index in range(4)),
        "pre_service_control",
        "post_service_control",
        "terminal",
    }
    stages = [row.get("stage") for row in rows]
    if len(stages) != len(set(stages)) or set(stages) != required:
        raise ValueError("required resource sample inventory mismatch")
    return all(
        row.get("strict_clean") is True
        and row.get("identity_match") is True
        and row.get("foreign_processes") == []
        for row in rows
    )


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
    if (
        cleanup.get("complete") is not True
        or cleanup.get("retained_generations") != 0
        or cleanup.get("retained_leases") != 0
        or cleanup.get("retained_tensors") != 0
        or cleanup.get("retained_process_groups") != 0
        or cleanup.get("owned_processes_remaining") != []
    ):
        raise RuntimeError("cleanup evidence is incomplete")
    return True


def _metric_summary(request_rows):
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
            request_tpots = [
                float(request["tpot_ns"])
                for row in rows
                for request in row["requests"]
            ]
            gaps = [
                float(gap)
                for row in rows
                for request in row["requests"]
                for gap in request["token_gaps_ns"]
            ]
            ttfts = [
                float(request["ttft_ns"])
                for row in rows
                for request in row["requests"]
            ]
            request_count = sum(len(row["requests"]) for row in rows)
            makespan = sum(float(row["cohort_makespan_ns"]) for row in rows)
            values[arm] = {
                "median_tpot_ns": statistics.median(request_tpots),
                "gap_p95_ns": nearest_rank_percentile(gaps, 95),
                "gap_p99_ns": nearest_rank_percentile(gaps, 99),
                "ttft_median_ns": statistics.median(ttfts),
                "ttft_p99_ns": nearest_rank_percentile(ttfts, 99),
                "request_qps": request_count * 1e9 / makespan,
                "output_tokens_per_second": (
                    request_count * 128 * 1e9 / makespan
                ),
            }
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
    aggregate_ratio = geometric_mean(ratios)
    return {
        "workloads": workload_metrics,
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
        _validate_correctness(payloads["correctness_rows.jsonl"])
        and _validate_cleanup(payloads["cleanup.json"])
    )
    candidate_coverage_pass = _validate_candidate_coverage(
        payloads["scheduler_step_rows.jsonl"],
        payloads["candidate_hit_rows.jsonl"],
        payloads["collective_rows.jsonl"],
    )
    resource_identity_pass = _validate_resources(
        payloads["resource_rows.jsonl"]
    )
    service_rows = payloads["service_control_rows.jsonl"]
    if (
        {row.get("workload_id") for row in service_rows}
        != set(ONLINE_WORKLOADS)
        or any(
            row.get("arm") != "TP2_X2_SERVICE_CONTROL"
            or row.get("classification_authority") is not False
            for row in service_rows
        )
    ):
        raise ValueError("service-control authority or inventory mismatch")

    migration_rows = payloads["migration_rows.jsonl"]
    if len(migration_rows) != 50:
        raise ValueError("migration evidence inventory mismatch")
    migration_pass = all(
        row.get("break_even_output_tokens", math.inf) <= 32
        and row.get("temporary_live_tensors") == 0
        and row.get("latency_ns", -1) >= 0
        for row in migration_rows
    )
    weight = payloads["weight_layout_manifest.json"]
    state = payloads["state_layout_manifest.json"]
    memory_rows = payloads["memory_rows.jsonl"]
    memory_pass = (
        weight.get("baseline_tp4_decode_accumulation_retained") is False
        and weight.get("steady_increment_bytes_per_rank", math.inf)
        <= 1920 * 1024**2
        and state.get("temporary_objects_released") is True
        and len(memory_rows) == 16
        and all(
            row.get("peak_allocated_bytes", math.inf)
            / row.get("physical_memory_bytes", 0) < 0.98
            for row in memory_rows
            if row.get("physical_memory_bytes", 0) > 0
        )
    )
    metrics = _metric_summary(request_index)
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
        "resource, and cleanup evidence.\n",
        encoding="utf-8",
    )
    _write_manifest(output_root)
    return summary
