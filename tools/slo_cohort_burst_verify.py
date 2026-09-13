#!/usr/bin/env python3
"""Independent verifier for SLO-aware cohort decode-burst evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
from typing import Mapping, Sequence


VERIFICATION_SCHEMA_VERSION = (
    "slo-cohort-burst.independent-verification.v1"
)
MANIFEST_SCHEMA_VERSION = "slo-cohort-burst.manifest.v1"
SUMMARY_SCHEMA_VERSION = "slo-cohort-burst.summary.v1"

INVALID_SOURCE_OR_EVIDENCE = "INVALID_SOURCE_OR_EVIDENCE"
NO_GO_CORRECTNESS = "NO_GO_CORRECTNESS"
NO_GO_LIFECYCLE = "NO_GO_LIFECYCLE"
NO_GO_STARVATION = "NO_GO_STARVATION"
NO_GO_TAIL_LATENCY = "NO_GO_TAIL_LATENCY"
NO_GO_MEMORY = "NO_GO_MEMORY"
NO_GO_EOS_WASTE = "NO_GO_EOS_WASTE"
NO_GO_THROUGHPUT = "NO_GO_THROUGHPUT"
GO_SLO_AWARE_COHORT_DECODE_BURST = (
    "GO_SLO_AWARE_COHORT_DECODE_BURST"
)

FAILURE_PRECEDENCE = (
    INVALID_SOURCE_OR_EVIDENCE,
    NO_GO_CORRECTNESS,
    NO_GO_LIFECYCLE,
    NO_GO_STARVATION,
    NO_GO_TAIL_LATENCY,
    NO_GO_MEMORY,
    NO_GO_EOS_WASTE,
    NO_GO_THROUGHPUT,
    GO_SLO_AWARE_COHORT_DECODE_BURST,
)

ARTIFACT_KEYS = {
    "source_manifest.json": "source_manifest",
    "environment.json": "environment",
    "cost_profile_rows.jsonl": "cost_profile_rows",
    "arrival_traces.json": "arrival_traces",
    "cost_table.json": "cost_table",
    "decision_rows.jsonl": "decision_rows",
    "execution_rows.jsonl": "execution_rows",
    "request_rows.jsonl": "request_rows",
    "correctness_rows.jsonl": "correctness_rows",
    "summary.json": "summary",
    "manifest.json": "manifest",
}
AUTHORITATIVE_ARTIFACTS = tuple(
    relative
    for relative in ARTIFACT_KEYS
    if relative != "manifest.json"
)
CORRECTNESS_ARTIFACT_KEYS = {
    "source_manifest.json": "source_manifest",
    "environment.json": "environment",
    "cost_profile_rows.jsonl": "cost_profile_rows",
    "cost_table.json": "cost_table",
    "correctness_rows.jsonl": "correctness_rows",
    "manifest.json": "manifest",
}
CORRECTNESS_AUTHORITATIVE_ARTIFACTS = tuple(
    relative
    for relative in CORRECTNESS_ARTIFACT_KEYS
    if relative != "manifest.json"
)

WORKLOADS = ("decode_heavy", "mixed", "bursty_eos")
LOADS = ("low", "medium", "high")
ARMS = ("baseline", "candidate")
LOAD_FRACTIONS = {"low": 0.40, "medium": 0.70, "high": 0.90}
REQUESTS_PER_REPETITION = 26
FROZEN_ARM_ORDER = (
    ("baseline", "candidate"),
    ("candidate", "baseline"),
    ("baseline", "candidate"),
    ("candidate", "baseline"),
    ("baseline", "candidate"),
)
WIDTHS = (1, 2, 4, 8)
BURST_WIDTHS_DESCENDING = (8, 4, 2)
CONTEXT_BUCKETS = (256, 2048, 8192)
PROFILE_CONTEXT_BUCKETS = (512, 4096, 16384)
PROFILE_MEASURED_STEPS = 16
PROFILE_ARRIVAL_GAP_NS = {
    "low": 4_000_000,
    "medium": 1_000_000,
    "high": 0,
}
QUALIFICATION_SOURCE_PATHS = (
    "tinyvllm/config.py",
    "tinyvllm/engine/exact_greedy_cohort_burst.py",
    "tinyvllm/engine/llm_engine.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tinyvllm/engine/slo_cohort_burst.py",
    "tools/profile_slo_cohort_burst_ceiling.py",
    "tools/run_slo_cohort_burst_remote.py",
    "tools/slo_cohort_burst_ceiling.py",
    "tools/slo_cohort_burst_gate.py",
    "tools/slo_cohort_burst_verify.py",
)
SOURCE_TREE_TOOL_PATHS = (
    "tools/slo_cohort_burst_ceiling.py",
    "tools/profile_slo_cohort_burst_ceiling.py",
    "tools/run_slo_cohort_burst_remote.py",
)
CEILING_PROFILE_SCHEMA_VERSION = (
    "slo-cohort-burst.ceiling-profile-row.v1"
)
CEILING_COMPONENTS = {
    "target_cuda",
    "graph_launch_gap",
    "scheduler",
    "token_d2h_publication",
    "batch_binding",
    "unattributed",
}
AMORTIZABLE_CEILING_COMPONENTS = (
    "graph_launch_gap",
    "token_d2h_publication",
    "batch_binding",
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _artifact_bytes(relative: str, payload: object) -> bytes:
    if relative.endswith(".jsonl"):
        if not isinstance(payload, list):
            raise ValueError(f"{relative} must contain a row list")
        return b"".join(_canonical_json(row) + b"\n" for row in payload)
    return _canonical_json(payload) + b"\n"


def _sha256_payload(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_tree_sha256(source_root: Path) -> str:
    root = Path(source_root).resolve()
    paths = [
        path
        for path in (root / "tinyvllm").rglob("*.py")
        if path.is_file()
    ]
    paths.extend(root / relative for relative in SOURCE_TREE_TOOL_PATHS)
    if not paths or any(
        not path.is_file() or path.is_symlink()
        for path in paths
    ):
        raise ValueError("source tree is incomplete")
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    return digest.hexdigest()


def _digest(
    value: object,
    field: str,
    *,
    lengths: tuple[int, ...] = (64,),
) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} is not a lowercase digest")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(
    value: object,
    field: str,
    *,
    minimum: int = 0,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(
            f"{field} must be an integer greater than or equal to "
            f"{minimum}"
        )
    return value


def _number(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be finite")
    normalized = float(value)
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return normalized


def _nearest_rank(
    values: Sequence[int | float],
    percentile: float,
) -> float:
    if not values:
        raise ValueError("metric samples are empty")
    ordered = sorted(
        _number(value, "metric sample", minimum=0.0)
        for value in values
    )
    return ordered[max(1, math.ceil(percentile * len(ordered))) - 1]


def _relative_change(baseline: float, candidate: float) -> float:
    baseline = _number(baseline, "baseline metric", minimum=0.0)
    candidate = _number(candidate, "candidate metric", minimum=0.0)
    if baseline <= 0.0:
        if candidate == baseline:
            return 0.0
        raise ValueError("baseline metric must be positive")
    return (candidate - baseline) / baseline


def _improvement(baseline: float, candidate: float) -> float:
    return -_relative_change(baseline, candidate)


def _safe_relative(root: Path, relative: object) -> Path:
    text = _text(relative, "relative path")
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("relative path escapes the evidence root")
    root = root.resolve()
    candidate = root / text
    if candidate.is_symlink():
        raise ValueError(f"source artifact is missing: {text}")
    path = candidate.resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"source artifact is missing: {text}")
    return path


def _require_fields(
    payload: object,
    required: set[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError(f"{name} fields mismatch")
    return payload


def _validate_manifest(
    bundle: Mapping[str, object],
    *,
    artifact_keys: Mapping[str, str] = ARTIFACT_KEYS,
    authoritative_artifacts: Sequence[str] = AUTHORITATIVE_ARTIFACTS,
) -> str:
    manifest = _require_fields(
        bundle.get("manifest"),
        {"schema_version", "artifact_sha256"},
        "manifest",
    )
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("manifest schema mismatch")
    hashes = manifest["artifact_sha256"]
    if (
        not isinstance(hashes, Mapping)
        or set(hashes) != set(authoritative_artifacts)
    ):
        raise ValueError("manifest artifact inventory mismatch")
    for relative in authoritative_artifacts:
        expected = _digest(
            hashes[relative],
            f"manifest digest for {relative}",
        )
        actual = hashlib.sha256(
            _artifact_bytes(
                relative,
                bundle[artifact_keys[relative]],
            )
        ).hexdigest()
        if expected != actual:
            raise ValueError(f"artifact hash mismatch: {relative}")
    return hashlib.sha256(
        _artifact_bytes("manifest.json", manifest)
    ).hexdigest()


def _validate_source_and_environment(
    bundle: Mapping[str, object],
    source_root: Path,
) -> tuple[dict, dict]:
    manifest = _require_fields(
        bundle.get("source_manifest"),
        {
            "schema_version",
            "source_identity",
            "dirty",
            "source_sha256",
        },
        "source manifest",
    )
    if (
        manifest["schema_version"]
        != "slo-cohort-burst.source-manifest.v1"
        or manifest["dirty"] is not False
    ):
        raise ValueError("source manifest is not a clean frozen source")
    identity = _require_fields(
        manifest["source_identity"],
        {
            "source_commit",
            "source_patch_sha256",
            "model",
            "checkpoint_sha256",
            "gpu_uuid",
            "gpu_name",
            "tensor_parallel_size",
            "dtype",
            "config_sha256",
        },
        "source identity",
    )
    _digest(identity["source_commit"], "source commit", lengths=(40, 64))
    for field in (
        "source_patch_sha256",
        "checkpoint_sha256",
        "config_sha256",
    ):
        _digest(identity[field], field)
    for field in ("model", "gpu_uuid", "gpu_name", "dtype"):
        _text(identity[field], field)
    if _integer(
        identity["tensor_parallel_size"],
        "source tensor parallel size",
        minimum=1,
    ) != 1:
        raise ValueError("Stage-1 source identity must use TP1")
    sources = manifest["source_sha256"]
    if (
        not isinstance(sources, Mapping)
        or set(sources) != set(QUALIFICATION_SOURCE_PATHS)
    ):
        raise ValueError("source digest inventory mismatch")
    for relative, expected in sources.items():
        expected = _digest(expected, f"source hash for {relative}")
        if _sha256_file(_safe_relative(source_root, relative)) != expected:
            raise ValueError(f"source hash mismatch: {relative}")
    if identity["source_patch_sha256"] != _source_tree_sha256(source_root):
        raise ValueError("source tree digest mismatch")

    environment = _require_fields(
        bundle.get("environment"),
        {
            "schema_version",
            "source_commit",
            "model",
            "checkpoint_sha256",
            "gpu_uuid",
            "gpu_name",
            "tensor_parallel_size",
            "temperature",
            "completion_only",
            "eos_token_id",
            "target_itl_ns",
            "target_ttft_ns",
            "reserve_ns",
            "graph_identity_sha256_by_batch",
        },
        "environment",
    )
    if environment["schema_version"] != "slo-cohort-burst.environment.v1":
        raise ValueError("environment schema mismatch")
    for field in (
        "source_commit",
        "model",
        "checkpoint_sha256",
        "gpu_uuid",
        "gpu_name",
        "tensor_parallel_size",
    ):
        if environment[field] != identity[field]:
            raise ValueError(f"source/environment identity mismatch: {field}")
    if (
        environment["model"] != "Qwen3-0.6B"
        or "A100" not in environment["gpu_name"]
        or environment["tensor_parallel_size"] != 1
        or environment["temperature"] != 0.0
        or environment["completion_only"] is not True
    ):
        raise ValueError("unsupported Stage-1 environment")
    _integer(environment["eos_token_id"], "EOS token ID")
    target_itl = _integer(
        environment["target_itl_ns"],
        "target ITL",
        minimum=1,
    )
    target_ttft = _integer(
        environment["target_ttft_ns"],
        "target TTFT",
        minimum=1,
    )
    reserve = _integer(environment["reserve_ns"], "reserve")
    if reserve >= min(target_itl, target_ttft):
        raise ValueError("reserve must be below both SLO targets")
    if (
        target_itl,
        target_ttft,
        reserve,
    ) != (
        40_000_000,
        1_000_000_000,
        2_000_000,
    ):
        raise ValueError("qualification SLO policy mismatch")
    graph_identities = environment["graph_identity_sha256_by_batch"]
    if (
        not isinstance(graph_identities, Mapping)
        or set(graph_identities) != {str(value) for value in WIDTHS}
    ):
        raise ValueError("graph identity inventory mismatch")
    for batch_size, digest in graph_identities.items():
        _digest(digest, f"graph identity for batch {batch_size}")
    return dict(identity), dict(environment)


def _validate_cost_table(
    cost_table: object,
    source_identity: Mapping[str, object],
    profile_rows: object,
) -> tuple[dict, dict[tuple[int, int, int], int]]:
    return verify_cost_table_against_profile_rows(
        cost_table,
        source_identity,
        profile_rows,
    )


def _cost_samples_from_profile_rows(
    profile_rows: object,
    *,
    source_commit: str,
) -> dict[tuple[int, int, int], list[int]]:
    if not isinstance(profile_rows, list) or not profile_rows:
        raise ValueError("cost profile row inventory is empty")
    seen_case_ids = set()
    grouped: dict[tuple[int, int, int], list[int]] = {}
    profile_counts = {
        (load, batch_size, context_bucket): 0
        for load in LOADS
        for batch_size in WIDTHS
        for context_bucket in PROFILE_CONTEXT_BUCKETS
    }
    for raw in profile_rows:
        row = _require_fields(
            raw,
            {
                "schema_version",
                "case_id",
                "load",
                "batch_size",
                "context_bucket",
                "burst_width",
                "source_commit",
                "offered_arrival_offsets_ns",
                "component_ns",
                "wall_ns",
                "committed_tokens",
                "cuda_reserved_bytes",
            },
            "cost profile row",
        )
        if row["schema_version"] != CEILING_PROFILE_SCHEMA_VERSION:
            raise ValueError("cost profile schema mismatch")
        case_id = _text(row["case_id"], "cost profile case ID")
        if case_id in seen_case_ids:
            raise ValueError("duplicate cost profile case ID")
        seen_case_ids.add(case_id)
        if row["source_commit"] != source_commit:
            raise ValueError("cost profile source mismatch")
        load = row["load"]
        if load not in LOADS:
            raise ValueError("cost profile load mismatch")
        batch = _integer(
            row["batch_size"],
            "cost profile batch",
            minimum=1,
        )
        context = _integer(
            row["context_bucket"],
            "cost profile context",
            minimum=1,
        )
        profile_key = (load, batch, context)
        if profile_key not in profile_counts:
            raise ValueError("frozen profile inventory is invalid")
        expected_case_prefix = (
            f"{load}-b{batch}-c{context}-r0-s"
        )
        step_suffix = case_id.removeprefix(expected_case_prefix)
        if (
            not case_id.startswith(expected_case_prefix)
            or not step_suffix.isdigit()
            or int(step_suffix) <= 0
        ):
            raise ValueError("frozen profile inventory is invalid")
        if _integer(
            row["burst_width"],
            "cost profile width",
            minimum=1,
        ) != 1:
            raise ValueError("cost profile must use baseline width one")
        if _integer(
            row["committed_tokens"],
            "cost profile committed tokens",
            minimum=1,
        ) != batch:
            raise ValueError("cost profile committed token mismatch")
        offsets = row["offered_arrival_offsets_ns"]
        expected_offsets = [
            index * PROFILE_ARRIVAL_GAP_NS[load]
            for index in range(batch)
        ]
        if (
            not isinstance(offsets, list)
            or len(offsets) != batch
            or offsets != expected_offsets
        ):
            raise ValueError("cost profile arrival inventory mismatch")
        for offset in offsets:
            _integer(offset, "cost profile arrival offset")
        components = _require_fields(
            row["component_ns"],
            CEILING_COMPONENTS,
            "cost profile components",
        )
        normalized_components = {
            name: _integer(
                value,
                f"cost profile component {name}",
            )
            for name, value in components.items()
        }
        wall_ns = _integer(
            row["wall_ns"],
            "cost profile wall duration",
            minimum=1,
        )
        _integer(
            row["cuda_reserved_bytes"],
            "cost profile reserved memory",
        )
        if sum(normalized_components.values()) != wall_ns:
            raise ValueError("cost profile component accounting mismatch")
        amortized_ns = sum(
            normalized_components[name]
            for name in AMORTIZABLE_CEILING_COMPONENTS
        )
        irreducible_ns = wall_ns - amortized_ns
        if irreducible_ns <= 0:
            raise ValueError("cost profile irreducible duration is invalid")
        for width in WIDTHS:
            grouped.setdefault((batch, context, width), []).append(
                irreducible_ns * width + amortized_ns
            )
        profile_counts[profile_key] += 1
    if any(
        count != PROFILE_MEASURED_STEPS
        for count in profile_counts.values()
    ):
        raise ValueError("frozen profile inventory is incomplete")
    return grouped


def verify_cost_table_against_profile_rows(
    cost_table: object,
    source_identity: Mapping[str, object],
    profile_rows: object,
) -> tuple[dict, dict[tuple[int, int, int], int]]:
    table = _require_fields(
        cost_table,
        {
            "schema_version",
            "source_identity",
            "entries",
            "table_sha256",
        },
        "cost table",
    )
    if table["schema_version"] != "slo-cohort-burst.cost-table.v1":
        raise ValueError("cost table schema mismatch")
    if table["source_identity"] != source_identity:
        raise ValueError("cost table source identity mismatch")
    expected_table_sha = _sha256_payload({
        "schema_version": table["schema_version"],
        "source_identity": table["source_identity"],
        "entries": table["entries"],
    })
    if table["table_sha256"] != expected_table_sha:
        raise ValueError("cost table identity mismatch")
    samples_by_key = _cost_samples_from_profile_rows(
        profile_rows,
        source_commit=_text(
            source_identity.get("source_commit"),
            "cost table source commit",
        ),
    )
    entries = table["entries"]
    if (
        not isinstance(entries, Mapping)
        or not entries
        or set(samples_by_key) != {
            (
                entry.get("batch_size"),
                entry.get("context_bucket"),
                entry.get("burst_width"),
            )
            for entry in entries.values()
            if isinstance(entry, Mapping)
        }
    ):
        raise ValueError("cost table/profile inventory mismatch")
    predictions = {}
    for name, raw in entries.items():
        entry = _require_fields(
            raw,
            {
                "batch_size",
                "context_bucket",
                "burst_width",
                "sample_count",
                "raw_sample_sha256",
                "p50_ns",
                "p95_ns",
                "p99_ns",
            },
            "cost table entry",
        )
        batch = _integer(entry["batch_size"], "cost batch", minimum=1)
        context = _integer(
            entry["context_bucket"],
            "cost context",
            minimum=1,
        )
        width = _integer(entry["burst_width"], "cost width", minimum=1)
        key = (batch, context, width)
        if (
            width not in WIDTHS
            or name != f"b{batch}-c{context}-k{width}"
        ):
            raise ValueError("cost table key identity mismatch")
        samples = sorted(samples_by_key[key])
        if len(samples) != _integer(
            entry["sample_count"],
            "cost sample count",
            minimum=1,
        ):
            raise ValueError("cost sample inventory mismatch")
        if entry["raw_sample_sha256"] != _sha256_payload(samples):
            raise ValueError("cost raw sample identity mismatch")
        expected_percentiles = (
            _nearest_rank(samples, 0.50),
            _nearest_rank(samples, 0.95),
            _nearest_rank(samples, 0.99),
        )
        if (
            entry["p50_ns"],
            entry["p95_ns"],
            entry["p99_ns"],
        ) != expected_percentiles:
            raise ValueError("cost percentile mismatch")
        predictions[key] = int(entry["p99_ns"])
    return dict(table), predictions


def _case_identity(case: object) -> tuple[str, str, int, str | None]:
    if not isinstance(case, Mapping):
        raise ValueError("case identity must be a mapping")
    fields = set(case)
    allowed = {"workload", "load", "repetition", "arm"}
    if fields not in (allowed, allowed - {"arm"}):
        raise ValueError("case identity fields mismatch")
    workload = case["workload"]
    load = case["load"]
    repetition = case["repetition"]
    arm = case.get("arm")
    if (
        workload not in WORKLOADS
        or load not in LOADS
        or isinstance(repetition, bool)
        or not isinstance(repetition, int)
        or repetition < 0
        or (arm is not None and arm not in ARMS)
    ):
        raise ValueError("case identity is invalid")
    return workload, load, repetition, arm


def _frozen_prompt_sha256(
    *,
    source_commit: str,
    workload: str,
    ordinal: int,
    prompt_tokens: int,
) -> str:
    seed = int(
        hashlib.sha256(
            f"{source_commit}:{workload}:{ordinal}".encode("utf-8")
        ).hexdigest()[:8],
        16,
    )
    prompt = [
        100 + ((seed + token_index * 997) % 30_000)
        for token_index in range(prompt_tokens)
    ]
    return hashlib.sha256(_canonical_json(prompt)).hexdigest()


def _frozen_saturation_rates(
    predictions: Mapping[tuple[int, int, int], int],
) -> dict[str, float]:
    def token_rate(context: int) -> float:
        predicted = _context_prediction(
            predictions,
            batch_size=8,
            contexts=[context],
            width=1,
        )
        if predicted is None or predicted <= 0:
            raise ValueError("cost table lacks a conservative B8 entry")
        return 8_000_000_000.0 / predicted

    short_rate = token_rate(384)
    mixed_rate = min(token_rate(384), token_rate(8_320))
    return {
        "decode_heavy": short_rate / 128.0,
        "mixed": mixed_rate / 128.0,
        "bursty_eos": short_rate / 68.0,
    }


def _expected_arrival_trace_cases(
    *,
    source_commit: str,
    predictions: Mapping[tuple[int, int, int], int],
) -> list[dict]:
    saturation = _frozen_saturation_rates(predictions)
    cases = []
    for workload in WORKLOADS:
        for load in LOADS:
            arrival_gap_ns = max(
                1,
                round(
                    1_000_000_000
                    / (saturation[workload] * LOAD_FRACTIONS[load])
                ),
            )
            for repetition in range(len(FROZEN_ARM_ORDER)):
                requests = []
                for request_index in range(REQUESTS_PER_REPETITION):
                    ordinal = (
                        repetition * REQUESTS_PER_REPETITION
                        + request_index
                    )
                    if workload == "mixed":
                        prompt_tokens = (
                            256
                            if ordinal < 91
                            else 2048
                            if ordinal < 117
                            else 8192
                        )
                        maximum_output_tokens = (
                            64 if prompt_tokens == 256 else 128
                        )
                    else:
                        prompt_tokens = 256
                        maximum_output_tokens = (
                            128
                            if workload == "decode_heavy"
                            else 8 * (1 + ordinal % 16)
                        )
                    arrival_offset_ns = (
                        (request_index // 5) * arrival_gap_ns * 5
                        if workload == "bursty_eos"
                        else request_index * arrival_gap_ns
                    )
                    requests.append({
                        "request_id": (
                            f"{workload}-{load}-r{repetition}"
                            f"-q{request_index}"
                        ),
                        "prompt_sha256": _frozen_prompt_sha256(
                            source_commit=source_commit,
                            workload=workload,
                            ordinal=ordinal,
                            prompt_tokens=prompt_tokens,
                        ),
                        "arrival_offset_ns": arrival_offset_ns,
                        "prompt_tokens": prompt_tokens,
                        "maximum_output_tokens": maximum_output_tokens,
                        "ignore_eos": workload != "bursty_eos",
                    })
                cases.append({
                    "workload": workload,
                    "load": load,
                    "repetition": repetition,
                    "requests": requests,
                })
    return cases


def _validate_arrival_traces(
    payload: object,
    *,
    source_commit: str,
    predictions: Mapping[tuple[int, int, int], int],
) -> tuple[dict, dict[tuple[str, str, int], dict[str, dict]]]:
    traces = _require_fields(
        payload,
        {
            "schema_version",
            "minimum_requests_per_workload_load_arm",
            "minimum_repetitions",
            "arm_order_by_repetition",
            "cases",
        },
        "arrival traces",
    )
    if traces["schema_version"] != "slo-cohort-burst.arrival-traces.v1":
        raise ValueError("arrival trace schema mismatch")
    minimum_requests = _integer(
        traces["minimum_requests_per_workload_load_arm"],
        "minimum request count",
        minimum=128,
    )
    minimum_repetitions = _integer(
        traces["minimum_repetitions"],
        "minimum repetitions",
        minimum=5,
    )
    orders = traces["arm_order_by_repetition"]
    if (
        not isinstance(orders, list)
        or minimum_repetitions != len(FROZEN_ARM_ORDER)
        or orders != [list(order) for order in FROZEN_ARM_ORDER]
    ):
        raise ValueError("paired arm order is not frozen and balanced")
    cases = traces["cases"]
    if not isinstance(cases, list):
        raise ValueError("arrival trace cases must be a list")
    indexed = {}
    counts = {(workload, load): 0 for workload in WORKLOADS for load in LOADS}
    for case in cases:
        required = {"workload", "load", "repetition", "requests"}
        case = _require_fields(case, required, "arrival trace case")
        workload, load, repetition, arm = _case_identity({
            key: case[key] for key in ("workload", "load", "repetition")
        })
        assert arm is None
        identity = (workload, load, repetition)
        if identity in indexed:
            raise ValueError("duplicate arrival trace case")
        requests = case["requests"]
        if not isinstance(requests, list) or not requests:
            raise ValueError("arrival trace request inventory is empty")
        request_index = {}
        for request in requests:
            request = _require_fields(
                request,
                {
                    "request_id",
                    "prompt_sha256",
                    "arrival_offset_ns",
                    "prompt_tokens",
                    "maximum_output_tokens",
                    "ignore_eos",
                },
                "arrival trace request",
            )
            request_id = _text(request["request_id"], "request ID")
            if request_id in request_index:
                raise ValueError("duplicate request ID in arrival trace")
            _digest(request["prompt_sha256"], "prompt digest")
            _integer(request["arrival_offset_ns"], "arrival offset")
            prompt_tokens = _integer(
                request["prompt_tokens"],
                "prompt tokens",
                minimum=1,
            )
            if prompt_tokens not in CONTEXT_BUCKETS:
                raise ValueError("unsupported prompt/context bucket")
            _integer(
                request["maximum_output_tokens"],
                "maximum output tokens",
                minimum=1,
            )
            if not isinstance(request["ignore_eos"], bool):
                raise ValueError("ignore_eos must be boolean")
            request_index[request_id] = dict(request)
        indexed[identity] = request_index
        counts[(workload, load)] += len(request_index)
    expected_case_ids = {
        (workload, load, repetition)
        for workload in WORKLOADS
        for load in LOADS
        for repetition in range(minimum_repetitions)
    }
    if set(indexed) != expected_case_ids:
        raise ValueError("arrival trace case inventory mismatch")
    if any(count < minimum_requests for count in counts.values()):
        raise ValueError("arrival trace request inventory is incomplete")
    mixed_prompts = [
        request["prompt_tokens"]
        for identity, requests in indexed.items()
        if identity[0] == "mixed"
        for request in requests.values()
    ]
    counts_by_prompt = {
        prompt: mixed_prompts.count(prompt) for prompt in CONTEXT_BUCKETS
    }
    total = len(mixed_prompts)
    if (
        counts_by_prompt[256] / total != 0.70
        or counts_by_prompt[2048] / total != 0.20
        or counts_by_prompt[8192] / total != 0.10
    ):
        raise ValueError("mixed workload composition mismatch")
    if any(
        request["prompt_tokens"] != 256
        for identity, requests in indexed.items()
        if identity[0] == "decode_heavy"
        for request in requests.values()
    ):
        raise ValueError("decode-heavy prompt shape mismatch")
    if any(
        request["ignore_eos"] is not False
        for identity, requests in indexed.items()
        if identity[0] == "bursty_eos"
        for request in requests.values()
    ):
        raise ValueError("EOS-sensitive trace must honor natural EOS")
    expected = {
        "schema_version": "slo-cohort-burst.arrival-traces.v1",
        "minimum_requests_per_workload_load_arm": 128,
        "minimum_repetitions": len(FROZEN_ARM_ORDER),
        "arm_order_by_repetition": [
            list(order) for order in FROZEN_ARM_ORDER
        ],
        "cases": _expected_arrival_trace_cases(
            source_commit=source_commit,
            predictions=predictions,
        ),
    }
    if dict(traces) != expected:
        raise ValueError("arrival trace differs from the frozen workload")
    return dict(traces), indexed


def _normalize_request_payload(payload: object) -> dict:
    row = _require_fields(
        payload,
        {
            "schema_version",
            "request_id",
            "sequence_id",
            "service_class",
            "arrival_ns",
            "prefill_start_ns",
            "prefill_complete_ns",
            "first_token_visible_ns",
            "token_visible_ns",
            "completion_ns",
            "output_token_ids",
            "output_text_sha256",
            "terminal_reason",
        },
        "request telemetry",
    )
    if row["schema_version"] != "slo-cohort-burst.request.v1":
        raise ValueError("request telemetry schema mismatch")
    request_id = _text(row["request_id"], "request ID")
    sequence_id = _integer(row["sequence_id"], "sequence ID")
    _text(row["service_class"], "service class")
    arrival = _integer(row["arrival_ns"], "arrival timestamp")
    prefill_start = _integer(
        row["prefill_start_ns"],
        "prefill start timestamp",
    )
    prefill_complete = _integer(
        row["prefill_complete_ns"],
        "prefill completion timestamp",
    )
    first = _integer(
        row["first_token_visible_ns"],
        "first token timestamp",
    )
    completion = _integer(row["completion_ns"], "completion timestamp")
    visible = row["token_visible_ns"]
    tokens = row["output_token_ids"]
    if (
        not isinstance(visible, list)
        or not visible
        or not isinstance(tokens, list)
        or not tokens
        or len(visible) != len(tokens)
    ):
        raise ValueError("request token/timestamp inventory mismatch")
    visible = [
        _integer(value, "token visibility timestamp") for value in visible
    ]
    tokens = [_integer(value, "output token ID") for value in tokens]
    timeline = [
        arrival,
        prefill_start,
        prefill_complete,
        first,
        *visible[1:],
        completion,
    ]
    if (
        first != visible[0]
        or completion != visible[-1]
        or any(current < prior for prior, current in zip(
            timeline,
            timeline[1:],
        ))
    ):
        raise ValueError("request timestamp identity mismatch")
    _digest(row["output_text_sha256"], "output text digest")
    _text(row["terminal_reason"], "terminal reason")
    normalized = dict(row)
    normalized["token_visible_ns"] = visible
    normalized["output_token_ids"] = tokens
    normalized["request_id"] = request_id
    normalized["sequence_id"] = sequence_id
    return normalized


def _validate_frozen_arrival_schedule(
    grouped: Mapping[tuple[str, str, int, str], Sequence[dict]],
    traces: Mapping[tuple[str, str, int], Mapping[str, dict]],
) -> None:
    for group_key, group in grouped.items():
        frozen_requests = traces[group_key[:3]]
        actual_by_request = {
            wrapper["request"]["request_id"]: wrapper["request"][
                "arrival_ns"
            ]
            for wrapper in group
        }
        epoch_ns = min(actual_by_request.values())
        expected_by_request = {
            request_id: epoch_ns + int(request["arrival_offset_ns"])
            for request_id, request in frozen_requests.items()
        }
        if actual_by_request != expected_by_request:
            raise ValueError(
                "request arrival schedule differs from frozen trace"
            )


def _validate_terminal_request_budget(
    request: Mapping[str, object],
    *,
    maximum_output_tokens: int,
    ignore_eos: bool,
    eos_token_id: int,
) -> None:
    output_tokens = request["output_token_ids"]
    terminal_reason = request["terminal_reason"]
    if ignore_eos:
        if (
            terminal_reason != "length"
            or len(output_tokens) != maximum_output_tokens
        ):
            raise ValueError("request output budget mismatch")
        return
    if terminal_reason == "eos":
        if (
            not output_tokens
            or output_tokens[-1] != eos_token_id
            or len(output_tokens) > maximum_output_tokens
        ):
            raise ValueError("request EOS termination mismatch")
        return
    if terminal_reason == "length":
        if (
            len(output_tokens) != maximum_output_tokens
            or eos_token_id in output_tokens
        ):
            raise ValueError("request output budget mismatch")
        return
    if terminal_reason == "starved":
        if len(output_tokens) > maximum_output_tokens:
            raise ValueError("request output budget mismatch")
        return
    raise ValueError("request terminal reason is unsupported")


def _validate_request_rows(
    rows: object,
    traces: Mapping[tuple[str, str, int], Mapping[str, dict]],
    *,
    eos_token_id: int,
) -> tuple[
    list[dict],
    dict[tuple[str, str, int, str], list[dict]],
    dict[tuple[str, str, int, str, int], dict],
]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("request row inventory is empty")
    grouped = {}
    by_sequence = {}
    seen = set()
    for wrapper in rows:
        wrapper = _require_fields(
            wrapper,
            {
                "schema_version",
                "case",
                "prompt_sha256",
                "maximum_output_tokens",
                "ignore_eos",
                "peak_cuda_reserved_bytes",
                "request",
            },
            "request evidence row",
        )
        if (
            wrapper["schema_version"]
            != "slo-cohort-burst.request-evidence.v1"
        ):
            raise ValueError("request evidence schema mismatch")
        workload, load, repetition, arm = _case_identity(wrapper["case"])
        if arm is None:
            raise ValueError("request evidence arm is missing")
        case_key = (workload, load, repetition)
        if case_key not in traces:
            raise ValueError("request evidence lacks frozen arrival trace")
        request = _normalize_request_payload(wrapper["request"])
        identity = (*case_key, arm, request["request_id"])
        if identity in seen:
            raise ValueError("duplicate request evidence identity")
        seen.add(identity)
        frozen = traces[case_key].get(request["request_id"])
        if frozen is None:
            raise ValueError("request is absent from frozen arrival trace")
        for field in (
            "prompt_sha256",
            "maximum_output_tokens",
            "ignore_eos",
        ):
            if wrapper[field] != frozen[field]:
                raise ValueError(f"frozen request mismatch: {field}")
        _validate_terminal_request_budget(
            request,
            maximum_output_tokens=int(wrapper["maximum_output_tokens"]),
            ignore_eos=bool(wrapper["ignore_eos"]),
            eos_token_id=eos_token_id,
        )
        _number(
            wrapper["peak_cuda_reserved_bytes"],
            "peak reserved memory",
            minimum=0.0,
        )
        normalized = dict(wrapper)
        normalized["request"] = request
        normalized["_prompt_tokens"] = frozen["prompt_tokens"]
        group_key = (*case_key, arm)
        grouped.setdefault(group_key, []).append(normalized)
        sequence_key = (*group_key, request["sequence_id"])
        if sequence_key in by_sequence:
            raise ValueError("duplicate sequence identity")
        by_sequence[sequence_key] = normalized
    expected_groups = {
        (workload, load, repetition, arm)
        for workload, load, repetition in traces
        for arm in ARMS
    }
    if set(grouped) != expected_groups:
        raise ValueError("request arm inventory mismatch")
    for group_key, group in grouped.items():
        trace_ids = set(traces[group_key[:3]])
        if {row["request"]["request_id"] for row in group} != trace_ids:
            raise ValueError("baseline/candidate arrival trace mismatch")
    _validate_frozen_arrival_schedule(grouped, traces)
    for case_key in traces:
        paired = {
            arm: {
                row["request"]["request_id"]: row
                for row in grouped[(*case_key, arm)]
            }
            for arm in ARMS
        }
        for request_id in traces[case_key]:
            baseline = paired["baseline"][request_id]
            candidate = paired["candidate"][request_id]
            for field in (
                "prompt_sha256",
                "maximum_output_tokens",
                "ignore_eos",
            ):
                if baseline[field] != candidate[field]:
                    raise ValueError("paired request identity mismatch")
            for field in ("output_token_ids", "output_text_sha256"):
                if (
                    baseline["request"][field]
                    != candidate["request"][field]
                ):
                    raise ValueError("paired output correctness mismatch")
    return list(rows), grouped, by_sequence


def _context_prediction(
    predictions: Mapping[tuple[int, int, int], int],
    *,
    batch_size: int,
    contexts: Sequence[int],
    width: int,
) -> int | None:
    values = []
    for context in contexts:
        candidates = [
            (bucket, value)
            for (batch, bucket, candidate_width), value
            in predictions.items()
            if (
                batch == batch_size
                and candidate_width == width
                and bucket >= context
            )
        ]
        if not candidates:
            return None
        _bucket, value = min(candidates, key=lambda item: item[0])
        values.append(int(value))
    return max(values)


def _request_state_at_decision(
    request_wrapper: Mapping[str, object],
    decision_now_ns: int,
) -> tuple[int, int, int]:
    request = request_wrapper["request"]
    arrival_ns = int(request["arrival_ns"])
    if decision_now_ns < arrival_ns:
        raise ValueError("protected request has not arrived")
    visible_count = sum(
        timestamp <= decision_now_ns
        for timestamp in request["token_visible_ns"]
    )
    anchor_ns = (
        request["token_visible_ns"][visible_count - 1]
        if visible_count
        else arrival_ns
    )
    remaining_output_tokens = (
        int(request_wrapper["maximum_output_tokens"])
        - visible_count
    )
    if remaining_output_tokens < 0:
        raise ValueError("request output exceeds its frozen budget")
    prompt_tokens = _integer(
        request_wrapper["_prompt_tokens"],
        "frozen prompt tokens",
        minimum=1,
    )
    return (
        int(anchor_ns),
        remaining_output_tokens,
        prompt_tokens + visible_count,
    )


def _validate_decisions(
    rows: object,
    *,
    environment: Mapping[str, object],
    cost_table_sha256: str,
    predictions: Mapping[tuple[int, int, int], int],
    request_by_sequence: Mapping[
        tuple[str, str, int, str, int],
        dict,
    ],
) -> tuple[list[dict], dict[tuple, dict]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("decision row inventory is empty")
    indexed = {}
    for wrapper in rows:
        wrapper = _require_fields(
            wrapper,
            {"schema_version", "case", "decision"},
            "decision evidence row",
        )
        if (
            wrapper["schema_version"]
            != "slo-cohort-burst.decision-evidence.v1"
        ):
            raise ValueError("decision evidence schema mismatch")
        workload, load, repetition, arm = _case_identity(wrapper["case"])
        if arm != "candidate":
            raise ValueError("decision evidence must belong to candidate")
        decision = _require_fields(
            wrapper["decision"],
            {
                "schema_version",
                "decision_now_ns",
                "schedule_generation",
                "batch_size",
                "ordered_cohort_sequence_ids",
                "queue_depths",
                "context_buckets",
                "protected_requests",
                "global_slack_ns",
                "predicted_cost_ns_by_width",
                "structural_eligibility_by_width",
                "selected_width",
                "reason",
                "cost_table_sha256",
            },
            "decision telemetry",
        )
        if decision["schema_version"] != "slo-cohort-burst.decision.v1":
            raise ValueError("decision telemetry schema mismatch")
        now = _integer(decision["decision_now_ns"], "decision timestamp")
        generation = _integer(
            decision["schedule_generation"],
            "schedule generation",
            minimum=1,
        )
        sequence_ids = decision["ordered_cohort_sequence_ids"]
        if (
            not isinstance(sequence_ids, list)
            or not sequence_ids
            or len(sequence_ids) != len(set(sequence_ids))
            or len(sequence_ids) != _integer(
                decision["batch_size"],
                "decision batch size",
                minimum=1,
            )
        ):
            raise ValueError("decision cohort inventory mismatch")
        contexts = decision["context_buckets"]
        if (
            not isinstance(contexts, list)
            or len(contexts) != len(sequence_ids)
        ):
            raise ValueError("decision context inventory mismatch")
        cohort_state_by_sequence = {}
        for row in contexts:
            row = _require_fields(
                row,
                {
                    "sequence_id",
                    "context_bucket",
                    "remaining_output_tokens",
                    "writable_tokens",
                },
                "decision context row",
            )
            sequence_id = _integer(
                row["sequence_id"],
                "context sequence ID",
            )
            cohort_state_by_sequence[sequence_id] = {
                "context_bucket": _integer(
                    row["context_bucket"],
                    "context bucket",
                    minimum=1,
                ),
                "remaining_output_tokens": _integer(
                    row["remaining_output_tokens"],
                    "remaining output tokens",
                ),
                "writable_tokens": _integer(
                    row["writable_tokens"],
                    "writable tokens",
                ),
            }
        if list(cohort_state_by_sequence) != sequence_ids:
            raise ValueError("decision cohort/context row order mismatch")
        state_by_sequence = {}
        for sequence_id in sequence_ids:
            request_wrapper = request_by_sequence.get((
                workload,
                load,
                repetition,
                "candidate",
                sequence_id,
            ))
            if request_wrapper is None:
                raise ValueError("decision request identity mismatch")
            state_by_sequence[sequence_id] = (
                _request_state_at_decision(request_wrapper, now)
            )
        if any(
            cohort_state_by_sequence[sequence_id]["context_bucket"]
            != state_by_sequence[sequence_id][2]
            or cohort_state_by_sequence[sequence_id][
                "remaining_output_tokens"
            ] != state_by_sequence[sequence_id][1]
            for sequence_id in sequence_ids
        ):
            raise ValueError("decision cohort state reconstruction mismatch")
        queue_depths = _require_fields(
            decision["queue_depths"],
            {"waiting", "prefilling", "running"},
            "decision queue depths",
        )
        queue_depths = {
            name: _integer(value, f"{name} queue depth")
            for name, value in queue_depths.items()
        }
        protected = decision["protected_requests"]
        if not isinstance(protected, list) or not protected:
            raise ValueError("protected request inventory is empty")
        expected_slacks = []
        protected_sequence_ids = []
        protected_category_counts = {
            "cohort": 0,
            "omitted_decode": 0,
            "waiting": 0,
            "incomplete_prefill": 0,
        }
        for protected_row in protected:
            protected_row = _require_fields(
                protected_row,
                {
                    "sequence_id",
                    "category",
                    "service_class",
                    "age_ns",
                    "slack_ns",
                },
                "protected request",
            )
            sequence_id = _integer(
                protected_row["sequence_id"],
                "protected sequence ID",
            )
            category = _text(
                protected_row["category"],
                "protected request category",
            )
            if category not in protected_category_counts:
                raise ValueError(
                    "protected request category is unsupported"
                )
            request_wrapper = request_by_sequence.get((
                workload,
                load,
                repetition,
                "candidate",
                sequence_id,
            ))
            if request_wrapper is None:
                raise ValueError("protected request identity mismatch")
            if (
                protected_row["service_class"]
                != request_wrapper["request"]["service_class"]
            ):
                raise ValueError(
                    "protected request service class mismatch"
                )
            anchor, _remaining, _context = (
                _request_state_at_decision(
                    request_wrapper,
                    now,
                )
            )
            age = now - anchor
            target = (
                environment["target_itl_ns"]
                if anchor != request_wrapper["request"]["arrival_ns"]
                else environment["target_ttft_ns"]
            )
            slack = target - age - environment["reserve_ns"]
            if (
                protected_row["age_ns"] != age
                or protected_row["slack_ns"] != slack
            ):
                raise ValueError("protected request slack mismatch")
            expected_slacks.append(slack)
            protected_sequence_ids.append(sequence_id)
            protected_category_counts[category] += 1
        if (
            len(protected_sequence_ids) != len(set(protected_sequence_ids))
            or protected_sequence_ids[:len(sequence_ids)] != sequence_ids
            or any(
                protected[index]["category"] != "cohort"
                for index in range(len(sequence_ids))
            )
            or protected_category_counts["cohort"] != len(sequence_ids)
            or queue_depths["running"] != (
                protected_category_counts["cohort"]
                + protected_category_counts["omitted_decode"]
            )
            or queue_depths["waiting"]
            != protected_category_counts["waiting"]
            or queue_depths["prefilling"]
            != protected_category_counts["incomplete_prefill"]
        ):
            raise ValueError(
                "decision queue/protected request inventory mismatch"
            )
        predicted = decision["predicted_cost_ns_by_width"]
        eligible = decision["structural_eligibility_by_width"]
        if (
            not isinstance(predicted, Mapping)
            or set(predicted) != {"8", "4", "2"}
            or not isinstance(eligible, Mapping)
            or set(eligible) != {"8", "4", "2"}
        ):
            raise ValueError("decision width inventory mismatch")
        expected_eligibility = {}
        minimum_remaining_output = min(
            state_by_sequence[sequence_id][1]
            for sequence_id in sequence_ids
        )
        for width in BURST_WIDTHS_DESCENDING:
            cost = _context_prediction(
                predictions,
                batch_size=len(sequence_ids),
                contexts=[
                    cohort_state_by_sequence[sequence_id][
                        "context_bucket"
                    ]
                    for sequence_id in sequence_ids
                ],
                width=width,
            )
            if predicted[str(width)] != cost:
                raise ValueError("decision predicted cost mismatch")
            expected_eligibility[str(width)] = all(
                cohort_state_by_sequence[sequence_id][
                    "remaining_output_tokens"
                ] >= width
                and cohort_state_by_sequence[sequence_id][
                    "writable_tokens"
                ] >= width
                for sequence_id in sequence_ids
            )
            if eligible[str(width)] is not expected_eligibility[str(width)]:
                raise ValueError(
                    "decision structural eligibility mismatch"
                )
        minimum_writable_tokens = min(
            cohort_state_by_sequence[sequence_id]["writable_tokens"]
            for sequence_id in sequence_ids
        )
        selected_width = _integer(
            decision["selected_width"],
            "selected width",
            minimum=1,
        )
        reason = _text(decision["reason"], "decision reason")
        if reason == "insufficient_output_budget":
            if (
                selected_width != 1
                or decision["global_slack_ns"] != 0
                or minimum_remaining_output >= 2
                or dict(eligible) != expected_eligibility
            ):
                raise ValueError("decision width reconstruction mismatch")
        elif reason == "kv_block_boundary":
            if (
                selected_width != 1
                or decision["global_slack_ns"] != 0
                or minimum_remaining_output < 2
                or minimum_writable_tokens >= 2
                or dict(eligible) != expected_eligibility
            ):
                raise ValueError("decision width reconstruction mismatch")
        else:
            if reason not in {
                "no_slo_slack",
                "predicted_cost_exceeds_slack",
                "selected",
            }:
                raise ValueError(
                    "decision fallback reason is not canonical"
                )
            global_slack = min(expected_slacks)
            if decision["global_slack_ns"] != global_slack:
                raise ValueError("global slack mismatch")
            if (
                minimum_remaining_output < 2
                or minimum_writable_tokens < 2
                or not eligible["2"]
            ):
                raise ValueError(
                    "decision structural eligibility mismatch"
                )
            selected = 1
            if global_slack > 0:
                for width in BURST_WIDTHS_DESCENDING:
                    if (
                        eligible[str(width)]
                        and predicted[str(width)] <= global_slack
                    ):
                        selected = width
                        break
            expected_reason = (
                "no_slo_slack"
                if global_slack <= 0
                else "selected"
                if selected > 1
                else "predicted_cost_exceeds_slack"
            )
            if (
                selected_width != selected
                or reason != expected_reason
            ):
                raise ValueError(
                    "decision width reconstruction mismatch"
                )
        if decision["cost_table_sha256"] != cost_table_sha256:
            raise ValueError("decision cost table identity mismatch")
        key = (
            workload,
            load,
            repetition,
            generation,
            tuple(sequence_ids),
        )
        if key in indexed:
            raise ValueError("duplicate decision identity")
        indexed[key] = dict(decision)
    return list(rows), indexed


def _validate_lease_rows(lease: Mapping[str, object]) -> list[dict]:
    rows = lease["rows"]
    width = lease["authorized_width"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("lease row inventory is empty")
    normalized = []
    physical_ranges = []
    for row in rows:
        row = _require_fields(
            row,
            {
                "sequence_id",
                "sequence_generation",
                "block_table_identity",
                "writable_block_identities",
                "first_write_position",
                "last_write_position",
                "first_physical_slot",
                "last_physical_slot",
                "initial_completion_count",
                "initial_sequence_length",
                "remaining_output_budget",
            },
            "lease row",
        )
        sequence_id = _integer(row["sequence_id"], "lease sequence ID")
        for field in (
            "sequence_generation",
            "first_write_position",
            "last_write_position",
            "first_physical_slot",
            "last_physical_slot",
            "initial_completion_count",
            "remaining_output_budget",
        ):
            _integer(row[field], field)
        _integer(
            row["initial_sequence_length"],
            "initial sequence length",
            minimum=1,
        )
        block_table = row["block_table_identity"]
        writable = row["writable_block_identities"]
        if (
            not isinstance(block_table, list)
            or not block_table
            or not isinstance(writable, list)
            or not writable
            or any(
                not isinstance(pair, list)
                or len(pair) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in pair
                )
                for pair in block_table + writable
            )
            or len({pair[0] for pair in block_table}) != len(block_table)
            or len({pair[0] for pair in writable}) != len(writable)
            or not {
                tuple(pair) for pair in writable
            }.issubset({tuple(pair) for pair in block_table})
        ):
            raise ValueError("lease block identity mismatch")
        if (
            row["remaining_output_budget"] < width
            or row["first_write_position"]
            != row["initial_sequence_length"] - 1
            or row["last_write_position"]
            != row["first_write_position"] + width - 1
            or row["last_physical_slot"]
            != row["first_physical_slot"] + width - 1
        ):
            raise ValueError("lease write authority mismatch")
        physical_ranges.append((
            row["first_physical_slot"],
            row["last_physical_slot"],
        ))
        normalized.append(dict(row))
    for prior, current in zip(
        sorted(physical_ranges),
        sorted(physical_ranges)[1:],
    ):
        if current[0] <= prior[1]:
            raise ValueError("lease physical write ranges overlap")
    if [row["sequence_id"] for row in normalized] != (
        lease["ordered_sequence_ids"]
    ):
        raise ValueError("lease row order mismatch")
    return normalized


def _publication_matches_request_segment(
    *,
    request_output_token_ids: Sequence[int],
    initial_completion_count: int,
    commit_tokens: Sequence[int],
) -> bool:
    start = _integer(
        initial_completion_count,
        "initial completion count",
    )
    stop = start + len(commit_tokens)
    return list(request_output_token_ids[start:stop]) == list(
        commit_tokens
    )


def _validate_executions(
    rows: object,
    *,
    decisions: Mapping[tuple, dict],
    environment: Mapping[str, object],
    cost_table_sha256: str,
    request_by_sequence: Mapping[tuple, dict],
) -> tuple[list[dict], int, int]:
    if not isinstance(rows, list):
        raise ValueError("execution row inventory is invalid")
    seen_decisions = set()
    total_wasted_forwards = 0
    total_forward_slots = 0
    for wrapper in rows:
        wrapper = _require_fields(
            wrapper,
            {
                "schema_version",
                "case",
                "lease",
                "lease_identity_sha256",
                "result",
                "result_identity_sha256",
                "publication",
                "execution",
            },
            "execution evidence row",
        )
        if (
            wrapper["schema_version"]
            != "slo-cohort-burst.execution-evidence.v1"
        ):
            raise ValueError("execution evidence schema mismatch")
        workload, load, repetition, arm = _case_identity(wrapper["case"])
        if arm != "candidate":
            raise ValueError("execution evidence must belong to candidate")
        lease = _require_fields(
            wrapper["lease"],
            {
                "schema_version",
                "schedule_generation",
                "graph_generation",
                "graph_identity_sha256",
                "ordered_sequence_ids",
                "requested_width",
                "authorized_width",
                "decision_now_ns",
                "cost_table_sha256",
                "predicted_duration_ns",
                "global_slack_ns",
                "rows",
            },
            "lease",
        )
        if lease["schema_version"] != "exact-greedy-cohort-burst.lease.v1":
            raise ValueError("lease schema mismatch")
        sequence_ids = lease["ordered_sequence_ids"]
        decision_key = (
            workload,
            load,
            repetition,
            lease["schedule_generation"],
            tuple(sequence_ids),
        )
        decision = decisions.get(decision_key)
        if decision is None or decision_key in seen_decisions:
            raise ValueError("lease/decision identity mismatch")
        seen_decisions.add(decision_key)
        if (
            lease["graph_identity_sha256"]
            != environment["graph_identity_sha256_by_batch"][
                str(len(sequence_ids))
            ]
            or lease["cost_table_sha256"] != cost_table_sha256
            or lease["requested_width"] != decision["selected_width"]
            or lease["authorized_width"] != decision["selected_width"]
            or lease["decision_now_ns"] != decision["decision_now_ns"]
            or lease["global_slack_ns"] != decision["global_slack_ns"]
            or lease["predicted_duration_ns"]
            != decision["predicted_cost_ns_by_width"][
                str(decision["selected_width"])
            ]
        ):
            raise ValueError("lease authority mismatch")
        _integer(lease["graph_generation"], "graph generation", minimum=1)
        if lease["requested_width"] not in BURST_WIDTHS_DESCENDING:
            raise ValueError("lease burst width is unsupported")
        lease_rows = _validate_lease_rows(lease)
        lease_row_by_sequence = {
            row["sequence_id"]: row for row in lease_rows
        }
        decision_state_by_sequence = {
            row["sequence_id"]: row
            for row in decision["context_buckets"]
        }
        lease_identity = _sha256_payload(lease)
        if (
            wrapper["lease_identity_sha256"] != lease_identity
            or not isinstance(wrapper["lease_identity_sha256"], str)
        ):
            raise ValueError("lease identity mismatch")

        result = _require_fields(
            wrapper["result"],
            {
                "schema_version",
                "lease_identity_sha256",
                "graph_identity_sha256",
                "graph_generation",
                "replay_count",
                "rows",
                "token_d2h_calls",
                "sampled_logit_d2h_calls",
            },
            "result",
        )
        if (
            result["schema_version"]
            != "exact-greedy-cohort-burst.result-identity.v1"
            or result["lease_identity_sha256"] != lease_identity
            or result["graph_identity_sha256"]
            != lease["graph_identity_sha256"]
            or result["graph_generation"] != lease["graph_generation"]
            or result["replay_count"] != lease["authorized_width"]
            or result["token_d2h_calls"] != 1
            or result["sampled_logit_d2h_calls"] != 0
        ):
            raise ValueError("result/lease/graph identity mismatch")
        result_rows = result["rows"]
        if (
            not isinstance(result_rows, list)
            or [row.get("sequence_id") for row in result_rows]
            != sequence_ids
        ):
            raise ValueError("result row order mismatch")
        result_identity = _sha256_payload(result)
        if wrapper["result_identity_sha256"] != result_identity:
            raise ValueError("result identity mismatch")

        publication = _require_fields(
            wrapper["publication"],
            {"ordered_sequence_ids", "rows"},
            "publication",
        )
        if publication["ordered_sequence_ids"] != sequence_ids:
            raise ValueError("publication row order mismatch")
        publication_rows = publication["rows"]
        if (
            not isinstance(publication_rows, list)
            or [row.get("sequence_id") for row in publication_rows]
            != sequence_ids
        ):
            raise ValueError("publication row inventory mismatch")
        generated_counts = {}
        committed_counts = {}
        discarded_counts = {}
        expected_waste = 0
        for result_row, publication_row in zip(
            result_rows,
            publication_rows,
        ):
            sequence_id = _integer(
                result_row.get("sequence_id"),
                "result sequence ID",
            )
            if set(result_row) != {
                "sequence_id",
                "sequence_generation",
                "tokens",
                "final_position",
                "final_context_length",
                "final_physical_slot",
                "sampled_logits",
            }:
                raise ValueError("result row fields mismatch")
            authority = lease_row_by_sequence[sequence_id]
            decision_state = decision_state_by_sequence[sequence_id]
            if (
                authority["initial_sequence_length"]
                != decision_state["context_bucket"]
                or authority["remaining_output_budget"]
                != decision_state["remaining_output_tokens"]
                or authority["initial_completion_count"]
                != (
                    int(request_by_sequence[(
                        workload,
                        load,
                        repetition,
                        "candidate",
                        sequence_id,
                    )]["maximum_output_tokens"])
                    - authority["remaining_output_budget"]
                )
                or lease["authorized_width"]
                > decision_state["writable_tokens"]
            ):
                raise ValueError("lease/decision row state mismatch")
            tokens = result_row["tokens"]
            if (
                not isinstance(tokens, list)
                or len(tokens) != lease["authorized_width"]
                or any(
                    isinstance(token, bool)
                    or not isinstance(token, int)
                    or token < 0
                    for token in tokens
                )
            ):
                raise ValueError("result token inventory mismatch")
            request_wrapper = request_by_sequence.get((
                workload,
                load,
                repetition,
                "candidate",
                sequence_id,
            ))
            if request_wrapper is None:
                raise ValueError("result request identity mismatch")
            if (
                _integer(
                    result_row["sequence_generation"],
                    "result sequence generation",
                )
                != authority["sequence_generation"]
                or _integer(
                    result_row["final_position"],
                    "result final position",
                )
                != (
                    authority["first_write_position"]
                    + result["replay_count"]
                )
                or _integer(
                    result_row["final_context_length"],
                    "result final context length",
                    minimum=1,
                )
                != (
                    authority["initial_sequence_length"]
                    + result["replay_count"]
                )
                or _integer(
                    result_row["final_physical_slot"],
                    "result final physical slot",
                )
                != authority["last_physical_slot"] + 1
                or result_row["sampled_logits"] != []
            ):
                raise ValueError("result/lease row state mismatch")
            expected_prefix = list(tokens)
            if request_wrapper["ignore_eos"]:
                pass
            elif environment["eos_token_id"] in expected_prefix:
                eos_index = expected_prefix.index(environment["eos_token_id"])
                expected_prefix = expected_prefix[:eos_index + 1]
            if (
                set(publication_row) != {"sequence_id", "commit_tokens"}
                or publication_row["commit_tokens"] != expected_prefix
            ):
                raise ValueError("EOS publication prefix mismatch")
            request_output = request_wrapper["request"][
                "output_token_ids"
            ]
            initial_completion_count = lease_row_by_sequence[
                sequence_id
            ]["initial_completion_count"]
            if (
                not _publication_matches_request_segment(
                    request_output_token_ids=request_output,
                    initial_completion_count=initial_completion_count,
                    commit_tokens=expected_prefix,
                )
            ):
                raise ValueError("publication/request output mismatch")
            if (
                not request_wrapper["ignore_eos"]
                and environment["eos_token_id"] in tokens
                and len(request_output)
                != initial_completion_count + len(expected_prefix)
            ):
                raise ValueError("request continued after published EOS")
            generated_counts[str(sequence_id)] = len(tokens)
            committed_counts[str(sequence_id)] = len(expected_prefix)
            discarded = len(tokens) - len(expected_prefix)
            discarded_counts[str(sequence_id)] = discarded
            expected_waste += discarded
        execution = _require_fields(
            wrapper["execution"],
            {
                "schema_version",
                "lease_identity_sha256",
                "result_identity_sha256",
                "graph_identity_sha256",
                "requested_width",
                "authorized_width",
                "completed_replay_count",
                "predicted_duration_ns",
                "actual_duration_ns",
                "host_visible_publication_gap_ns",
                "token_d2h_calls",
                "token_d2h_bytes",
                "sampled_logit_d2h_calls",
                "generated_token_counts",
                "committed_token_counts",
                "eos_discarded_token_counts",
                "post_eos_wasted_tokens",
                "post_eos_wasted_forwards",
                "post_eos_wasted_forward_fraction",
                "fallback_reason",
                "failure_reason",
                "rollback_reason",
                "quarantined",
                "quarantine_reason",
                "pending_inventory",
            },
            "execution telemetry",
        )
        total_slots = result["replay_count"] * len(result_rows)
        expected_fraction = expected_waste / total_slots
        if (
            execution["schema_version"]
            != "exact-greedy-cohort-burst.execution.v1"
            or execution["lease_identity_sha256"] != lease_identity
            or execution["result_identity_sha256"] != result_identity
            or execution["graph_identity_sha256"]
            != lease["graph_identity_sha256"]
            or execution["requested_width"] != lease["requested_width"]
            or execution["authorized_width"] != lease["authorized_width"]
            or execution["completed_replay_count"]
            != result["replay_count"]
            or execution["predicted_duration_ns"]
            != lease["predicted_duration_ns"]
            or execution["actual_duration_ns"]
            != execution["host_visible_publication_gap_ns"]
            or execution["token_d2h_calls"] != result["token_d2h_calls"]
            or execution["sampled_logit_d2h_calls"]
            != result["sampled_logit_d2h_calls"]
            or execution["token_d2h_bytes"] != total_slots * 8
            or execution["generated_token_counts"] != generated_counts
            or execution["committed_token_counts"] != committed_counts
            or execution["eos_discarded_token_counts"] != discarded_counts
            or execution["post_eos_wasted_tokens"] != expected_waste
            or execution["post_eos_wasted_forwards"] != expected_waste
            or execution["post_eos_wasted_forward_fraction"]
            != expected_fraction
            or execution["fallback_reason"] is not None
            or execution["failure_reason"] is not None
            or execution["rollback_reason"] is not None
            or execution["quarantined"] is not False
            or execution["quarantine_reason"] is not None
            or execution["pending_inventory"]
            != {"leases": 0, "transactions": 0}
        ):
            raise ValueError("execution lifecycle inventory mismatch")
        _integer(
            execution["actual_duration_ns"],
            "actual duration",
            minimum=1,
        )
        _integer(
            execution["host_visible_publication_gap_ns"],
            "publication gap",
        )
        _integer(execution["token_d2h_bytes"], "token D2H bytes")
        total_wasted_forwards += expected_waste
        total_forward_slots += total_slots
    expected_execution_decisions = {
        key
        for key, decision in decisions.items()
        if decision["selected_width"] > 1
    }
    if seen_decisions != expected_execution_decisions:
        raise ValueError("decision/execution inventory mismatch")
    return list(rows), total_wasted_forwards, total_forward_slots


def _validate_correctness(rows: object) -> bool:
    if not isinstance(rows, list):
        raise ValueError("correctness rows must be a list")
    identities = set()
    exact = True
    for case in rows:
        case = _require_fields(
            case,
            {
                "schema_version",
                "batch_size",
                "burst_width",
                "rows",
                "duplicate_forwards",
                "duplicate_commits",
                "unauthorized_kv_publications",
                "pending_leases_after_case",
            },
            "correctness case",
        )
        if (
            case["schema_version"]
            != "slo-cohort-burst.correctness-case.v1"
        ):
            raise ValueError("correctness schema mismatch")
        identity = (
            _integer(case["batch_size"], "correctness batch", minimum=1),
            _integer(case["burst_width"], "correctness width", minimum=1),
        )
        if identity in identities or identity[0] not in WIDTHS or (
            identity[1] not in WIDTHS
        ):
            raise ValueError("correctness case identity mismatch")
        identities.add(identity)
        case_rows = case["rows"]
        if not isinstance(case_rows, list) or len(case_rows) != identity[0]:
            raise ValueError("correctness row inventory mismatch")
        for row_index, row in enumerate(case_rows):
            row = _require_fields(
                row,
                {
                    "row_index",
                    "baseline_output_token_ids",
                    "candidate_output_token_ids",
                    "baseline_output_text_sha256",
                    "candidate_output_text_sha256",
                    "baseline_sampled_logits_sha256",
                    "candidate_sampled_logits_sha256",
                    "baseline_argmax_token_ids",
                    "candidate_argmax_token_ids",
                },
                "correctness row",
            )
            baseline_tokens = row["baseline_output_token_ids"]
            candidate_tokens = row["candidate_output_token_ids"]
            baseline_argmax = row["baseline_argmax_token_ids"]
            candidate_argmax = row["candidate_argmax_token_ids"]
            if (
                _integer(
                    row["row_index"],
                    "correctness row index",
                ) != row_index
                or not isinstance(baseline_tokens, list)
                or len(baseline_tokens) != identity[1]
                or not isinstance(candidate_tokens, list)
                or len(candidate_tokens) != identity[1]
                or not isinstance(baseline_argmax, list)
                or len(baseline_argmax) != identity[1]
                or not isinstance(candidate_argmax, list)
                or len(candidate_argmax) != identity[1]
            ):
                raise ValueError("correctness row shape mismatch")
            for value in (
                baseline_tokens
                + candidate_tokens
                + baseline_argmax
                + candidate_argmax
            ):
                _integer(value, "correctness token ID")
            baseline_logits_sha = _digest(
                row["baseline_sampled_logits_sha256"],
                "baseline sampled logits digest",
            )
            candidate_logits_sha = _digest(
                row["candidate_sampled_logits_sha256"],
                "candidate sampled logits digest",
            )
            baseline_text_sha = _digest(
                row["baseline_output_text_sha256"],
                "baseline output text digest",
            )
            candidate_text_sha = _digest(
                row["candidate_output_text_sha256"],
                "candidate output text digest",
            )
            exact = exact and (
                baseline_tokens == candidate_tokens
                and baseline_text_sha == candidate_text_sha
                and baseline_logits_sha == candidate_logits_sha
                and baseline_argmax == candidate_argmax
                and baseline_argmax == baseline_tokens
            )
        exact = exact and all(
            _integer(case[field], field) == 0
            for field in (
                "duplicate_forwards",
                "duplicate_commits",
                "unauthorized_kv_publications",
                "pending_leases_after_case",
            )
        )
    if identities != {
        (batch_size, width)
        for batch_size in WIDTHS
        for width in WIDTHS
    }:
        raise ValueError("correctness B x K inventory mismatch")
    return exact


def _request_metrics(rows: Sequence[dict]) -> dict[str, float]:
    requests = [row["request"] for row in rows]
    start = min(row["arrival_ns"] for row in requests)
    end = max(row["completion_ns"] for row in requests)
    duration = end - start
    if duration <= 0:
        raise ValueError("request measurement window is not positive")
    ttft = [
        row["first_token_visible_ns"] - row["arrival_ns"]
        for row in requests
    ]
    itl = [
        current - prior
        for row in requests
        for prior, current in zip(
            row["token_visible_ns"],
            row["token_visible_ns"][1:],
        )
    ]
    if not itl:
        raise ValueError("request evidence contains no ITL samples")
    e2e = [row["completion_ns"] - row["arrival_ns"] for row in requests]
    output_tokens = sum(len(row["output_token_ids"]) for row in requests)
    return {
        "duration_ns": float(duration),
        "committed_output_tokens": float(output_tokens),
        "output_throughput_tps": (
            output_tokens * 1_000_000_000.0 / duration
        ),
        "p99_ttft_ns": _nearest_rank(ttft, 0.99),
        "p99_itl_ns": _nearest_rank(itl, 0.99),
        "p99_e2e_ns": _nearest_rank(e2e, 0.99),
        "maximum_host_visible_gap_ns": max(itl),
        "starved_requests": float(sum(
            row["terminal_reason"] == "starved" for row in requests
        )),
        "peak_reserved_bytes": max(
            float(row["peak_cuda_reserved_bytes"]) for row in rows
        ),
    }


def _reconstruct_summary(
    *,
    grouped_requests: Mapping[tuple[str, str, int, str], list[dict]],
    correctness_passed: bool,
    lifecycle_closed: bool,
    wasted_forwards: int,
    total_forward_slots: int,
) -> dict[str, object]:
    paired_metrics = {}
    for workload in WORKLOADS:
        for load in LOADS:
            for repetition in sorted({
                key[2]
                for key in grouped_requests
                if key[:2] == (workload, load)
            }):
                key = (workload, load, repetition)
                paired_metrics[key] = {
                    arm: _request_metrics(
                        grouped_requests[(*key, arm)]
                    )
                    for arm in ARMS
                }
    throughput_improvements = {
        key: _relative_change(
            values["baseline"]["output_throughput_tps"],
            values["candidate"]["output_throughput_tps"],
        )
        for key, values in paired_metrics.items()
    }
    metric_regressions = {}
    for metric in ("p99_itl_ns", "p99_ttft_ns", "p99_e2e_ns"):
        metric_regressions[metric] = [
            max(
                0.0,
                _relative_change(
                    values["baseline"][metric],
                    values["candidate"][metric],
                ),
            )
            for values in paired_metrics.values()
        ]
    memory_regressions = [
        max(
            0.0,
            _relative_change(
                values["baseline"]["peak_reserved_bytes"],
                values["candidate"]["peak_reserved_bytes"],
            ),
        )
        for values in paired_metrics.values()
    ]
    starved = int(sum(
        values["candidate"]["starved_requests"]
        for values in paired_metrics.values()
    ))
    candidate_gaps = [
        values["candidate"]["maximum_host_visible_gap_ns"]
        for values in paired_metrics.values()
    ]
    aggregate = statistics.mean(throughput_improvements.values())
    by_load = {
        load: statistics.mean(
            value
            for key, value in throughput_improvements.items()
            if key[1] == load
        )
        for load in LOADS
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "evidence_complete": True,
        "source_exact": True,
        "verifier_agreement": True,
        "correctness_passed": correctness_passed,
        "lifecycle_closed": lifecycle_closed,
        "aggregate_throughput_improvement": aggregate,
        "medium_throughput_improvement": by_load["medium"],
        "high_throughput_improvement": by_load["high"],
        "worst_throughput_regression": max(
            0.0,
            -min(throughput_improvements.values()),
        ),
        "worst_p99_itl_regression": max(
            metric_regressions["p99_itl_ns"]
        ),
        "worst_p99_ttft_regression": max(
            metric_regressions["p99_ttft_ns"]
        ),
        "worst_p99_e2e_regression": max(
            metric_regressions["p99_e2e_ns"]
        ),
        "maximum_host_visible_gap_ns": int(max(candidate_gaps)),
        "starved_requests": starved,
        "post_eos_wasted_forward_fraction": (
            wasted_forwards / total_forward_slots
            if total_forward_slots
            else 0.0
        ),
        "peak_reserved_memory_regression": max(memory_regressions),
    }
    summary["classification"] = _classify(summary)
    return summary


def _classify(summary: Mapping[str, object]) -> str:
    required = {
        "schema_version",
        "evidence_complete",
        "source_exact",
        "verifier_agreement",
        "correctness_passed",
        "lifecycle_closed",
        "aggregate_throughput_improvement",
        "medium_throughput_improvement",
        "high_throughput_improvement",
        "worst_throughput_regression",
        "worst_p99_itl_regression",
        "worst_p99_ttft_regression",
        "worst_p99_e2e_regression",
        "maximum_host_visible_gap_ns",
        "starved_requests",
        "post_eos_wasted_forward_fraction",
        "peak_reserved_memory_regression",
    }
    if set(summary) != required or summary["schema_version"] != (
        SUMMARY_SCHEMA_VERSION
    ):
        return INVALID_SOURCE_OR_EVIDENCE
    boolean_fields = (
        "evidence_complete",
        "source_exact",
        "verifier_agreement",
        "correctness_passed",
        "lifecycle_closed",
    )
    if any(not isinstance(summary[field], bool) for field in boolean_fields):
        return INVALID_SOURCE_OR_EVIDENCE
    try:
        values = {
            field: _number(summary[field], field)
            for field in (
                "aggregate_throughput_improvement",
                "medium_throughput_improvement",
                "high_throughput_improvement",
                "worst_throughput_regression",
                "worst_p99_itl_regression",
                "worst_p99_ttft_regression",
                "worst_p99_e2e_regression",
                "post_eos_wasted_forward_fraction",
                "peak_reserved_memory_regression",
            )
        }
        maximum_gap = _integer(
            summary["maximum_host_visible_gap_ns"],
            "maximum host-visible gap",
        )
        starved = _integer(
            summary["starved_requests"],
            "starved request count",
        )
    except ValueError:
        return INVALID_SOURCE_OR_EVIDENCE
    if any(
        values[field] < 0.0
        for field in (
            "worst_throughput_regression",
            "worst_p99_itl_regression",
            "worst_p99_ttft_regression",
            "worst_p99_e2e_regression",
            "post_eos_wasted_forward_fraction",
            "peak_reserved_memory_regression",
        )
    ):
        return INVALID_SOURCE_OR_EVIDENCE
    if not all(summary[field] for field in boolean_fields[:3]):
        return INVALID_SOURCE_OR_EVIDENCE
    if not summary["correctness_passed"]:
        return NO_GO_CORRECTNESS
    if not summary["lifecycle_closed"]:
        return NO_GO_LIFECYCLE
    if starved:
        return NO_GO_STARVATION
    if (
        values["worst_p99_itl_regression"] > 0.03
        or values["worst_p99_ttft_regression"] > 0.05
        or values["worst_p99_e2e_regression"] > 0.05
        or maximum_gap > 40_000_000
    ):
        return NO_GO_TAIL_LATENCY
    if values["peak_reserved_memory_regression"] > 0.05:
        return NO_GO_MEMORY
    if values["post_eos_wasted_forward_fraction"] > 0.10:
        return NO_GO_EOS_WASTE
    if (
        values["aggregate_throughput_improvement"] < 0.10
        or values["medium_throughput_improvement"] < 0.10
        or values["high_throughput_improvement"] < 0.10
        or values["worst_throughput_regression"] > 0.02
    ):
        return NO_GO_THROUGHPUT
    return GO_SLO_AWARE_COHORT_DECODE_BURST


def _equal_evidence(expected: object, actual: object) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return math.isclose(
                float(expected),
                float(actual),
                rel_tol=1e-12,
                abs_tol=0.0,
            )
        except (TypeError, ValueError):
            return False
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        return (
            set(expected) == set(actual)
            and all(
                _equal_evidence(expected[key], actual[key])
                for key in expected
            )
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return (
            len(expected) == len(actual)
            and all(
                _equal_evidence(left, right)
                for left, right in zip(expected, actual)
            )
        )
    return expected == actual


def verify_slo_cohort_burst_bundle(
    bundle: Mapping[str, object],
    *,
    source_root: Path,
) -> dict[str, object]:
    if not isinstance(bundle, Mapping) or set(bundle) != set(
        ARTIFACT_KEYS.values()
    ):
        raise ValueError("SLO cohort-burst bundle inventory mismatch")
    manifest_sha256 = _validate_manifest(bundle)
    source_identity, environment = _validate_source_and_environment(
        bundle,
        Path(source_root),
    )
    cost_table, predictions = _validate_cost_table(
        bundle["cost_table"],
        source_identity,
        bundle["cost_profile_rows"],
    )
    _traces, trace_index = _validate_arrival_traces(
        bundle["arrival_traces"],
        source_commit=source_identity["source_commit"],
        predictions=predictions,
    )
    request_rows, grouped, request_by_sequence = _validate_request_rows(
        bundle["request_rows"],
        trace_index,
        eos_token_id=environment["eos_token_id"],
    )
    decision_rows, decisions = _validate_decisions(
        bundle["decision_rows"],
        environment=environment,
        cost_table_sha256=cost_table["table_sha256"],
        predictions=predictions,
        request_by_sequence=request_by_sequence,
    )
    execution_rows, wasted, total_slots = _validate_executions(
        bundle["execution_rows"],
        decisions=decisions,
        environment=environment,
        cost_table_sha256=cost_table["table_sha256"],
        request_by_sequence=request_by_sequence,
    )
    correctness_rows = bundle["correctness_rows"]
    correctness_passed = _validate_correctness(correctness_rows)
    lifecycle_closed = all(
        row["execution"]["pending_inventory"]
        == {"leases": 0, "transactions": 0}
        and row["execution"]["failure_reason"] is None
        and row["execution"]["rollback_reason"] is None
        and row["execution"]["quarantined"] is False
        for row in execution_rows
    )
    reconstructed = _reconstruct_summary(
        grouped_requests=grouped,
        correctness_passed=correctness_passed,
        lifecycle_closed=lifecycle_closed,
        wasted_forwards=wasted,
        total_forward_slots=total_slots,
    )
    recorded = bundle["summary"]
    if not _equal_evidence(recorded, reconstructed):
        raise ValueError("summary or classification drift")
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "verified": True,
        "source_commit": source_identity["source_commit"],
        "classification": reconstructed["classification"],
        "request_row_count": len(request_rows),
        "decision_row_count": len(decision_rows),
        "execution_row_count": len(execution_rows),
        "correctness_case_count": len(correctness_rows),
        "cost_table_sha256": cost_table["table_sha256"],
        "manifest_sha256": manifest_sha256,
        "authoritative_artifact_sha256": dict(
            bundle["manifest"]["artifact_sha256"]
        ),
        "reconstructed_summary": reconstructed,
        "failure_precedence": list(FAILURE_PRECEDENCE),
    }


def verify_correctness_bundle(
    bundle: Mapping[str, object],
    *,
    source_root: Path,
) -> dict[str, object]:
    if not isinstance(bundle, Mapping) or set(bundle) != set(
        CORRECTNESS_ARTIFACT_KEYS.values()
    ):
        raise ValueError("correctness bundle inventory mismatch")
    manifest_sha256 = _validate_manifest(
        bundle,
        artifact_keys=CORRECTNESS_ARTIFACT_KEYS,
        authoritative_artifacts=CORRECTNESS_AUTHORITATIVE_ARTIFACTS,
    )
    source_identity, _environment = _validate_source_and_environment(
        bundle,
        Path(source_root),
    )
    cost_table, _predictions = _validate_cost_table(
        bundle["cost_table"],
        source_identity,
        bundle["cost_profile_rows"],
    )
    correctness_rows = bundle["correctness_rows"]
    correctness_passed = _validate_correctness(correctness_rows)
    if not correctness_passed:
        raise ValueError("correctness evidence is not exact")
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "verified": True,
        "source_commit": source_identity["source_commit"],
        "classification": "PASS_CORRECTNESS_AND_LIFECYCLE",
        "correctness_case_count": len(correctness_rows),
        "cost_table_sha256": cost_table["table_sha256"],
        "manifest_sha256": manifest_sha256,
        "authoritative_artifact_sha256": dict(
            bundle["manifest"]["artifact_sha256"]
        ),
    }


def _load_json(path: Path) -> object:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required artifact is missing: {path.name}")
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required artifact is missing: {path.name}")
    return [
        json.loads(line, parse_constant=_reject_constant)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)


def verify_artifact_directory(
    path: Path,
    *,
    source_root: Path,
    output: Path | None = None,
    stage: str = "canonical",
) -> dict[str, object]:
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("artifact directory is invalid")
    artifact_keys = (
        CORRECTNESS_ARTIFACT_KEYS
        if stage == "correctness"
        else ARTIFACT_KEYS
        if stage == "canonical"
        else None
    )
    if artifact_keys is None:
        raise ValueError("unsupported verifier stage")
    bundle = {
        key: (
            _load_jsonl(root / relative)
            if relative.endswith(".jsonl")
            else _load_json(root / relative)
        )
        for relative, key in artifact_keys.items()
    }
    result = (
        verify_correctness_bundle(
            bundle,
            source_root=source_root,
        )
        if stage == "correctness"
        else verify_slo_cohort_burst_bundle(
            bundle,
            source_root=source_root,
        )
    )
    if output is not None:
        destination = Path(output)
        if destination.exists() or destination.is_symlink():
            raise ValueError("verification output already exists")
        _write_json(destination, result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Independently verify a SLO cohort-burst bundle",
    )
    parser.add_argument("artifact_directory", type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--stage",
        choices=("correctness", "canonical"),
        default="canonical",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = verify_artifact_directory(
        args.artifact_directory,
        source_root=args.source_root,
        output=args.output,
        stage=args.stage,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
