#!/usr/bin/env python3
"""Independently verify a topology-local TP2 whole-model evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics


WORKLOADS = ("P0", "P1", "Q0", "Q1", "Q2")
ONLINE_WORKLOADS = ("Q0", "Q1", "Q2")
EPOCH_ARMS = ("baseline", "candidate", "candidate", "baseline")
APPROVED_REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
CORE_FILES = {
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
}
GENERATED_FILES = {
    "classification.json",
    "report.md",
}
MANIFEST_SCHEMA = (
    "qwen38.topology-local-tp2-whole-model-manifest.v1"
)
VERIFICATION_SCHEMA = (
    "qwen38.topology-local-tp2-whole-model-verification.v1"
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


def _load_json(path):
    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_keys,
            parse_constant=_nonfinite,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error


def _load_jsonl(path):
    rows = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
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


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _percentile(values, percentile):
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise ValueError("percentile inputs must be finite")
    return ordered[
        max(1, math.ceil(percentile / 100.0 * len(ordered))) - 1
    ]


def _geometric_mean(values):
    values = tuple(float(value) for value in values)
    if not values or any(
        not math.isfinite(value) or value <= 0 for value in values
    ):
        raise ValueError("geometric mean inputs must be finite")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _classification(summary):
    precedence = (
        ("correctness_pass", "NO_GO_CORRECTNESS_OR_LIFECYCLE"),
        ("resource_identity_pass", "NO_GO_RESOURCE_IDENTITY"),
        ("candidate_coverage_pass", "NO_GO_CANDIDATE_NOT_EXERCISED"),
        ("memory_pass", "NO_GO_MEMORY_OR_ALLOCATION"),
        (
            "measurement_complete",
            "INCONCLUSIVE_ENVIRONMENT_OR_MEASUREMENT",
        ),
        ("tail_ttft_pass", "NO_GO_TAIL_OR_TTFT"),
        ("throughput_pass", "NO_GO_THROUGHPUT"),
        ("migration_pass", "NO_GO_MIGRATION_AMORTIZATION"),
    )
    for field, result in precedence:
        if summary.get(field) is not True:
            return result
    directions = summary.get("pair_direction_counts", {})
    if (
        summary.get("aggregate_median_tpot_improvement_percent", -math.inf)
        < 5.0
        or summary.get("improving_workload_count", 0) < 4
        or summary.get("median_regressing_workloads")
        or set(directions) != set(WORKLOADS)
        or any(directions[name] < 7 for name in WORKLOADS)
    ):
        return "NO_GO_PERFORMANCE"
    return "GO_TOPOLOGY_LOCAL_TP2_WHOLE_MODEL_GATE"


def _verify_manifest(root):
    root = Path(root)
    manifest = _load_json(root / "manifest.json")
    digest_text = (root / "manifest.sha256").read_text().strip()
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or digest_text != _sha256(root / "manifest.json")
    ):
        raise ValueError("manifest digest or schema mismatch")
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_file()
        and path.name not in {"manifest.json", "manifest.sha256"}
    }
    if actual != CORE_FILES | GENERATED_FILES:
        raise ValueError("manifest artifact inventory mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != actual:
        raise ValueError("manifest artifact inventory mismatch")
    for name, identity in files.items():
        path = root / name
        if (
            identity.get("size_bytes") != path.stat().st_size
            or identity.get("sha256") != _sha256(path)
        ):
            raise ValueError("manifest artifact hash mismatch")


def _validate_manifests(payloads, classification):
    source = payloads["source_manifest.json"]
    model = payloads["model_manifest.json"]
    source_revision = source.get("source_revision")
    source_tree = source.get("source_tree_sha256")
    model_revision = source.get("model_revision")
    if (
        source.get("source_archive_complete") is not True
        or not isinstance(source_revision, str)
        or len(source_revision) != 40
        or not isinstance(source_tree, str)
        or len(source_tree) != 64
        or classification.get("source_revision") != source_revision
        or classification.get("source_tree_sha256") != source_tree
    ):
        raise ValueError("source identity mismatch")
    if (
        model.get("model_repository") != "Qwen/Qwen3.8-27B"
        or model.get("source_revision") != source_revision
        or model.get("model_revision") != model_revision
        or classification.get("model_revision") != model_revision
        or model.get("num_hidden_layers") != 64
        or model.get("linear_attention_layer_count") != 48
        or model.get("full_attention_layer_count") != 16
    ):
        raise ValueError("model identity mismatch")
    observed = tuple(
        row.get("arm")
        for row in sorted(
            payloads["campaign_epoch_manifest.json"].get("epochs", []),
            key=lambda row: row.get("epoch", -1),
        )
    )
    if observed != EPOCH_ARMS:
        raise ValueError("epoch arm order mismatch")
    if payloads["gpu_rank_manifest.json"].get("ranks") != list(range(4)):
        raise ValueError("rank identity mismatch")
    if (
        payloads["pair_group_manifest.json"].get("pair_groups")
        != [[0, 1], [2, 3]]
        or payloads["pair_group_manifest.json"].get("optimal_matching")
        is not True
    ):
        raise ValueError("pair identity mismatch")
    return source_revision, model_revision


def _request_index(rows, source_revision, model_revision):
    expected = {
        (epoch, arm, workload, repetition)
        for epoch, arm in enumerate(EPOCH_ARMS)
        for workload in WORKLOADS
        for repetition in range(5)
    }
    indexed = {}
    for row in rows:
        key = (
            row.get("epoch"),
            row.get("arm"),
            row.get("workload_id"),
            row.get("repetition"),
        )
        if key in indexed:
            raise ValueError("duplicate request identity")
        indexed[key] = row
        if (
            row.get("source_revision") != source_revision
            or row.get("model_revision") != model_revision
        ):
            raise ValueError("request identity drift")
        for request in row.get("requests", []):
            tokens = request.get("output_token_ids")
            gaps = request.get("token_gaps_ns")
            if (
                not isinstance(tokens, list)
                or len(tokens) != 128
                or not isinstance(gaps, list)
                or len(gaps) != 127
            ):
                raise ValueError("token or token-gap inventory mismatch")
            for value in [
                *gaps,
                request.get("ttft_ns"),
                request.get("tpot_ns"),
                request.get("e2e_ns"),
            ]:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or value < 0
                ):
                    raise ValueError("request timing is invalid")
    if set(indexed) != expected:
        raise ValueError("request measurement inventory mismatch")
    for workload in WORKLOADS:
        for repetition in range(5):
            rows = [
                indexed[(epoch, EPOCH_ARMS[epoch], workload, repetition)]
                for epoch in range(4)
            ]
            if len({row.get("request_set_digest") for row in rows}) != 1:
                raise ValueError("request-set identity mismatch")
            token_sets = {
                tuple(tuple(request["output_token_ids"])
                      for request in row["requests"])
                for row in rows
            }
            if len(token_sets) != 1:
                raise RuntimeError("request token mismatch")
    return indexed


def _candidate_coverage(payloads):
    def index(rows, label):
        result = {}
        for row in rows:
            key = (
                row.get("epoch"),
                row.get("workload_id"),
                row.get("repetition"),
            )
            if key in result:
                raise ValueError(f"duplicate {label} identity")
            result[key] = row
        return result

    scheduler = index(payloads["scheduler_step_rows.jsonl"], "scheduler")
    hits = index(payloads["candidate_hit_rows.jsonl"], "candidate")
    collectives = index(payloads["collective_rows.jsonl"], "collective")
    expected = {
        (epoch, workload, repetition)
        for epoch, arm in enumerate(EPOCH_ARMS)
        if arm == "candidate"
        for workload in WORKLOADS
        for repetition in range(5)
    }
    if set(scheduler) != expected or set(hits) != expected:
        raise ValueError("candidate evidence inventory mismatch")
    if set(collectives) != expected:
        raise ValueError("collective evidence inventory mismatch")
    for key in expected:
        segments = scheduler[key].get("token_one_segments")
        hit = hits[key]
        collective = collectives[key]
        if (
            scheduler[key].get("decode_steps") != 127
            or not isinstance(segments, int)
            or segments <= 0
            or hit.get("tp2_decode_calls") != segments * 48
            or hit.get("recurrent_token_one_calls") != segments * 48
            or hit.get("global_tp4_linear_decode_all_reduce_calls") != 0
            or hit.get("full_attention_tp4_collective_calls")
            != segments * 16
            or hit.get("fallback_calls") != 0
            or hit.get("post_warmup_request_path_allocations") != 0
            or hit.get("retry_after_mutation_calls") != 0
            or hit.get("duplicate_commit_calls") != 0
        ):
            raise RuntimeError("candidate hit evidence mismatch")
        if hit.get("migration_publications") != segments // 127:
            raise RuntimeError("candidate migration publication mismatch")
        if (
            hit.get("short_chunk_calls") != 0
            or hit.get("ordinary_chunk_calls") != 0
        ):
            raise RuntimeError("candidate short or ordinary chunk mismatch")
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
            raise RuntimeError("collective evidence mismatch")
    return True


def _correctness(rows):
    if len(rows) != 25:
        raise ValueError("correctness inventory mismatch")
    required = (
        "output_tokens_match",
        "rank_token_agreement",
        "finite_logits",
        "state_checkpoints_complete",
        "single_commit_per_step",
        "pair_replica_digest_match",
    )
    return all(
        all(row.get(field) is True for field in required)
        for row in rows
    )


def _resource_identity(rows):
    expected = {
        "entry",
        *(f"pre_epoch_{index}" for index in range(4)),
        *(f"post_launch_{index}" for index in range(4)),
        "pre_service_control",
        "post_service_control",
        "terminal",
    }
    if {row.get("stage") for row in rows} != expected:
        raise ValueError("resource inventory mismatch")
    return all(
        row.get("strict_clean") is True
        and row.get("identity_match") is True
        and row.get("foreign_processes") == []
        for row in rows
    )


def _cleanup(payload):
    for path in payload.get("task_paths", []):
        candidate = PurePosixPath(path)
        if (
            not candidate.is_absolute()
            or not candidate.is_relative_to(
                PurePosixPath(APPROVED_REMOTE_ROOT)
            )
        ):
            raise ValueError("cleanup path identity mismatch")
    if payload.get("foreign_process_actions") != []:
        raise RuntimeError("cleanup foreign process action detected")
    if (
        payload.get("complete") is not True
        or payload.get("retained_generations") != 0
        or payload.get("retained_leases") != 0
        or payload.get("retained_tensors") != 0
        or payload.get("retained_process_groups") != 0
        or payload.get("owned_processes_remaining") != []
    ):
        raise RuntimeError("cleanup evidence is incomplete")
    return True


def _metrics(indexed):
    ratios = []
    improving = 0
    regressions = []
    directions = {}
    tail_pass = True
    throughput_pass = True
    workload_metrics = {}
    for workload in WORKLOADS:
        values = {}
        for arm in ("baseline", "candidate"):
            rows = [
                row for row in indexed.values()
                if row["workload_id"] == workload and row["arm"] == arm
            ]
            tpots = [
                float(request["tpot_ns"])
                for row in rows for request in row["requests"]
            ]
            gaps = [
                float(gap)
                for row in rows
                for request in row["requests"]
                for gap in request["token_gaps_ns"]
            ]
            ttfts = [
                float(request["ttft_ns"])
                for row in rows for request in row["requests"]
            ]
            request_count = sum(len(row["requests"]) for row in rows)
            makespan = sum(float(row["cohort_makespan_ns"]) for row in rows)
            values[arm] = {
                "median_tpot_ns": statistics.median(tpots),
                "gap_p95_ns": _percentile(gaps, 95),
                "gap_p99_ns": _percentile(gaps, 99),
                "ttft_median_ns": statistics.median(ttfts),
                "ttft_p99_ns": _percentile(ttfts, 99),
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
        improving += ratio < 1.0
        if ratio > 1.0:
            regressions.append(workload)
        if (
            values["candidate"]["gap_p99_ns"]
            / values["baseline"]["gap_p99_ns"] > 1.02
            or values["candidate"]["ttft_median_ns"]
            / values["baseline"]["ttft_median_ns"] > 1.02
            or values["candidate"]["ttft_p99_ns"]
            / values["baseline"]["ttft_p99_ns"] > 1.02
        ):
            tail_pass = False
        if workload in ONLINE_WORKLOADS and (
            values["candidate"]["request_qps"]
            / values["baseline"]["request_qps"] < 0.98
        ):
            throughput_pass = False
        count = 0
        for repetition in range(5):
            for baseline_epoch, candidate_epoch in ((0, 2), (3, 1)):
                baseline = indexed[(
                    baseline_epoch,
                    "baseline",
                    workload,
                    repetition,
                )]
                candidate = indexed[(
                    candidate_epoch,
                    "candidate",
                    workload,
                    repetition,
                )]
                count += statistics.median(
                    request["tpot_ns"]
                    for request in candidate["requests"]
                ) < statistics.median(
                    request["tpot_ns"]
                    for request in baseline["requests"]
                )
        directions[workload] = count
        workload_metrics[workload] = values
    aggregate = _geometric_mean(ratios)
    return {
        "workloads": workload_metrics,
        "aggregate_median_tpot_ratio": aggregate,
        "aggregate_median_tpot_improvement_percent": (1 - aggregate) * 100,
        "improving_workload_count": improving,
        "median_regressing_workloads": regressions,
        "pair_direction_counts": directions,
        "tail_ttft_pass": tail_pass,
        "throughput_pass": throughput_pass,
        "raw_gap_p99_authority": True,
    }


def verify_bundle(bundle, output_path=None):
    root = Path(bundle)
    _verify_manifest(root)
    payloads = {}
    for name in CORE_FILES:
        payloads[name] = (
            _load_jsonl(root / name)
            if name.endswith(".jsonl")
            else _load_json(root / name)
        )
    producer = _load_json(root / "classification.json")
    source_revision, model_revision = _validate_manifests(
        payloads,
        producer,
    )
    indexed = _request_index(
        payloads["request_rows.jsonl"],
        source_revision,
        model_revision,
    )
    correctness_pass = (
        _correctness(payloads["correctness_rows.jsonl"])
        and _cleanup(payloads["cleanup.json"])
    )
    candidate_coverage_pass = _candidate_coverage(payloads)
    resource_identity_pass = _resource_identity(
        payloads["resource_rows.jsonl"]
    )
    service = payloads["service_control_rows.jsonl"]
    if (
        {row.get("workload_id") for row in service}
        != set(ONLINE_WORKLOADS)
        or any(
            row.get("arm") != "TP2_X2_SERVICE_CONTROL"
            or row.get("classification_authority") is not False
            for row in service
        )
    ):
        raise ValueError("service-control evidence is invalid")
    migration = payloads["migration_rows.jsonl"]
    if len(migration) != 50:
        raise ValueError("migration inventory mismatch")
    migration_pass = all(
        row.get("break_even_output_tokens", math.inf) <= 32
        and row.get("temporary_live_tensors") == 0
        and row.get("latency_ns", -1) >= 0
        for row in migration
    )
    memory = payloads["memory_rows.jsonl"]
    weight = payloads["weight_layout_manifest.json"]
    state = payloads["state_layout_manifest.json"]
    memory_pass = (
        len(memory) == 16
        and weight.get("baseline_tp4_decode_accumulation_retained") is False
        and weight.get("steady_increment_bytes_per_rank", math.inf)
        <= 1920 * 1024**2
        and state.get("temporary_objects_released") is True
        and all(
            row.get("physical_memory_bytes", 0) > 0
            and row.get("peak_allocated_bytes", math.inf)
            / row["physical_memory_bytes"] < 0.98
            for row in memory
        )
    )
    metrics = _metrics(indexed)
    reconstructed = {
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
    reconstructed["classification"] = _classification(reconstructed)
    if reconstructed != producer:
        raise ValueError("producer classification does not match raw evidence")
    checks = {
        "manifest": True,
        "source_identity": True,
        "model_identity": True,
        "epoch_inventory": True,
        "request_tokens_and_timings": True,
        "candidate_coverage": True,
        "collectives": True,
        "memory": True,
        "cleanup": True,
        "service_control_non_authoritative": True,
    }
    semantic_digest = hashlib.sha256(_canonical_bytes({
        "classification": reconstructed,
        "checks": checks,
    })).hexdigest()
    receipt = {
        "schema_version": VERIFICATION_SCHEMA,
        "classification": reconstructed["classification"],
        "checks": checks,
        "semantic_digest": semantic_digest,
    }
    if output_path is not None:
        Path(output_path).write_bytes(_canonical_bytes(receipt) + b"\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    verify_bundle(args.bundle, output_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
