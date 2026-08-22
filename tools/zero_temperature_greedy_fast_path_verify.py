#!/usr/bin/env python3
"""Independent verifier for zero-temperature greedy fast-path evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


CASE_SCHEMA = "zero-temperature-greedy-fast-path.case.v1"
CORRECTNESS_SCHEMA = (
    "zero-temperature-greedy-fast-path.correctness.v1"
)
COMPARISON_SCHEMA = (
    "zero-temperature-greedy-fast-path.comparison.v1"
)
GATE_SCHEMA = "zero-temperature-greedy-fast-path.gate.v1"
MANIFEST_SCHEMA = (
    "zero-temperature-greedy-fast-path.manifest.v1"
)
VERIFICATION_SCHEMA = (
    "zero-temperature-greedy-fast-path."
    "independent-verification.v1"
)
SOURCE_SCHEMA = "zero-temperature-greedy-fast-path.source.v1"
WORKLOAD_SCHEMA = (
    "zero-temperature-greedy-fast-path.workload.v1"
)
CONTEXTS = (
    ("short", 256, 128),
    ("medium", 2048, 128),
    ("long", 8192, 128),
)
POINTS = ("prefill-final", "decode-first", "decode-final")
SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/greedy_sampling_fast_path.py",
    "tinyvllm/engine/model_runner.py",
    "tools/profile_zero_temperature_greedy_fast_path.py",
    "tools/test_profile_zero_temperature_greedy_fast_path.py",
    "tools/zero_temperature_greedy_fast_path_gate.py",
    "tools/test_zero_temperature_greedy_fast_path_gate.py",
    "tools/zero_temperature_greedy_fast_path_verify.py",
    "tools/test_zero_temperature_greedy_fast_path_verify.py",
    "tools/run_zero_temperature_greedy_fast_path_remote.py",
    "tools/test_run_zero_temperature_greedy_fast_path_remote.py",
    "tools/run_staged_inference_benchmark_remote.py",
    "tools/test_run_staged_inference_benchmark_remote.py",
)
PRIMARY_ARTIFACTS = {
    "case_rows.jsonl",
    "correctness_rows.jsonl",
    "source_manifest.json",
    "workload_manifest.json",
    "summary.json",
    "comparison.json",
    "gate.json",
}
COUNTER_FIELDS = (
    "eligible_steps",
    "optimized_steps",
    "avoided_temperature_h2d_bytes",
    "avoided_softmax_calls",
    "avoided_gumbel_rng_calls",
    "avoided_stochastic_divisions",
    "avoided_stochastic_argmax_calls",
    "avoided_where_calls",
)
MEDIAN_THRESHOLD = 0.05
P95_THRESHOLD = 0.05
TPOT_REGRESSION_LIMIT = 0.03
LATENCY_REGRESSION_LIMIT = 0.03
THROUGHPUT_REGRESSION_LIMIT = 0.02
MEMORY_REGRESSION_LIMIT = 0.01
LOGIT_MAX_LIMIT = 0.25
LOGIT_MEAN_LIMIT = 0.05


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def _load_json(path: Path):
    if not path.is_file():
        raise ValueError(
            f"primary artifact is missing: {path.name}"
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=_reject_constant,
        )


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(
            f"primary artifact is missing: {path.name}"
        )
    with path.open("r", encoding="utf-8") as handle:
        return [
            json.loads(
                line,
                parse_constant=_reject_constant,
            )
            for line in handle
            if line.strip()
        ]


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"artifact is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _nearest_rank(values, percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty percentile input")
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _relative_change(baseline, candidate) -> float:
    baseline = float(baseline)
    candidate = float(candidate)
    if baseline <= 0.0:
        if candidate == baseline:
            return 0.0
        raise ValueError(
            "relative comparison baseline must be positive"
        )
    return (candidate - baseline) / baseline


def _improvement(baseline, candidate) -> float:
    return -_relative_change(baseline, candidate)


def _assert_finite_tree(value, path="root") -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(
                f"non-finite numeric value at {path}"
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_tree(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_tree(item, f"{path}.{key}")
        return
    raise ValueError(f"unsupported evidence type at {path}")


def _validate_performance_rows(rows) -> list[dict]:
    if len(rows) != 30:
        raise ValueError("expected exactly 30 measured rows")
    shapes = {
        bucket: (prompt, generated)
        for bucket, prompt, generated in CONTEXTS
    }
    identities = set()
    for row in rows:
        _assert_finite_tree(row)
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != CASE_SCHEMA
        ):
            raise ValueError("case row schema mismatch")
        bucket = row.get("context_bucket")
        policy = row.get("policy")
        repetition = row.get("repetition")
        if (
            bucket not in shapes
            or policy not in {"off", "on"}
            or isinstance(repetition, bool)
            or not isinstance(repetition, int)
            or repetition not in range(5)
        ):
            raise ValueError("case identity mismatch")
        identity = (bucket, repetition, policy)
        if identity in identities:
            raise ValueError("duplicate case identity")
        identities.add(identity)
        if (
            row.get("prompt_tokens"),
            row.get("generated_tokens"),
        ) != shapes[bucket]:
            raise ValueError("case shape mismatch")
        generated = row["generated_tokens"]
        if (
            not isinstance(row.get("output_token_ids"), list)
            or len(row["output_token_ids"]) != generated
            or not isinstance(row.get("output_text_sha256"), str)
            or len(row["output_text_sha256"]) != 64
        ):
            raise ValueError("output evidence mismatch")
        for field in (
            "tpot_samples_ns",
            "decode_host_ns",
            "decode_cuda_ns",
        ):
            values = row.get(field)
            if (
                not isinstance(values, list)
                or len(values) != generated - 1
            ):
                raise ValueError(f"{field} inventory mismatch")
        for field in (
            "ttft_ns",
            "e2e_ns",
            "output_tokens_per_second",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
        ):
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
            ):
                raise ValueError(f"{field} is invalid")
        summary = row.get("greedy_fast_path_summary")
        if (
            not isinstance(summary, dict)
            or set(COUNTER_FIELDS) - set(summary)
            or "fallback_counts" not in summary
        ):
            raise ValueError(
                "greedy fast-path summary mismatch"
            )
    expected = {
        (bucket, repetition, policy)
        for bucket, _prompt, _generated in CONTEXTS
        for repetition in range(5)
        for policy in ("off", "on")
    }
    if identities != expected:
        raise ValueError("case inventory mismatch")
    if len({row.get("run_tag") for row in rows}) != 1:
        raise ValueError("run tag mismatch")
    if len({row.get("source_commit") for row in rows}) != 1:
        raise ValueError("source commit mismatch")
    return rows


def _read_sidecar(run_dir: Path, row: dict) -> tuple[float, ...]:
    raw_path = row.get("logits_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("sidecar path mismatch")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("sidecar path escapes run directory")
    path = run_dir / relative
    payload = path.read_bytes() if path.is_file() else None
    if payload is None:
        raise ValueError(f"sidecar is missing: {raw_path}")
    expected_bytes = row.get("logits_byte_length")
    expected_count = row.get("logits_element_count")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
        or expected_bytes != expected_count * 4
        or len(payload) != expected_bytes
    ):
        raise ValueError("sidecar byte length mismatch")
    if hashlib.sha256(payload).hexdigest() != row.get(
        "logits_sha256"
    ):
        raise ValueError("sidecar digest mismatch")
    values = struct.unpack(f"<{expected_count}f", payload)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("sidecar contains non-finite values")
    return tuple(values)


def _validate_correctness_rows(
    rows,
    *,
    run_dir: Path,
) -> dict:
    if len(rows) != 18:
        raise ValueError(
            "expected exactly 18 correctness rows"
        )
    shapes = {
        bucket: (prompt, generated)
        for bucket, prompt, generated in CONTEXTS
    }
    identities = {}
    values_by_identity = {}
    for row in rows:
        _assert_finite_tree(row)
        if (
            not isinstance(row, dict)
            or row.get("schema_version")
            != CORRECTNESS_SCHEMA
        ):
            raise ValueError("correctness row schema mismatch")
        identity = (
            row.get("context_bucket"),
            row.get("sampling_point"),
            row.get("policy"),
        )
        bucket, point, policy = identity
        if (
            bucket not in shapes
            or point not in POINTS
            or policy not in {"off", "on"}
        ):
            raise ValueError(
                "correctness identity mismatch"
            )
        if identity in identities:
            raise ValueError(
                "duplicate correctness identity"
            )
        identities[identity] = row
        if (
            row.get("prompt_tokens"),
            row.get("generated_tokens"),
        ) != shapes[bucket]:
            raise ValueError(
                "correctness context shape mismatch"
            )
        generated = row["generated_tokens"]
        if (
            not isinstance(row.get("output_token_ids"), list)
            or len(row["output_token_ids"]) != generated
            or not isinstance(row.get("output_text_sha256"), str)
            or len(row["output_text_sha256"]) != 64
        ):
            raise ValueError(
                "correctness output evidence mismatch"
            )
        shape = row.get("logits_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or shape[0] != 1
            or isinstance(shape[1], bool)
            or not isinstance(shape[1], int)
            or shape[1] <= 0
            or shape[0] * shape[1]
            != row.get("logits_element_count")
        ):
            raise ValueError(
                "correctness logits shape mismatch"
            )
        values_by_identity[identity] = _read_sidecar(
            run_dir,
            row,
        )
    expected = {
        (bucket, point, policy)
        for bucket, _prompt, _generated in CONTEXTS
        for point in POINTS
        for policy in ("off", "on")
    }
    if set(identities) != expected:
        raise ValueError(
            "correctness inventory mismatch"
        )
    pairs = []
    maximum = 0.0
    worst_mean = 0.0
    total_abs = 0.0
    total_count = 0
    all_argmax = True
    all_ids = True
    all_text = True
    for bucket, _prompt, _generated in CONTEXTS:
        for point in POINTS:
            off_row = identities[(bucket, point, "off")]
            on_row = identities[(bucket, point, "on")]
            off_values = values_by_identity[
                (bucket, point, "off")
            ]
            on_values = values_by_identity[
                (bucket, point, "on")
            ]
            if (
                off_row["logits_shape"] != on_row["logits_shape"]
                or len(off_values) != len(on_values)
            ):
                raise ValueError("paired logits shape mismatch")
            differences = [
                abs(left - right)
                for left, right in zip(off_values, on_values)
            ]
            pair_max = max(differences)
            pair_mean = sum(differences) / len(differences)
            off_argmax = max(
                range(len(off_values)),
                key=off_values.__getitem__,
            )
            on_argmax = max(
                range(len(on_values)),
                key=on_values.__getitem__,
            )
            argmax_equal = off_argmax == on_argmax
            ids_equal = (
                off_row["output_token_ids"]
                == on_row["output_token_ids"]
            )
            text_equal = (
                off_row["output_text_sha256"]
                == on_row["output_text_sha256"]
            )
            maximum = max(maximum, pair_max)
            worst_mean = max(worst_mean, pair_mean)
            total_abs += sum(differences)
            total_count += len(differences)
            all_argmax = all_argmax and argmax_equal
            all_ids = all_ids and ids_equal
            all_text = all_text and text_equal
            pairs.append({
                "context_bucket": bucket,
                "sampling_point": point,
                "element_count": len(differences),
                "max_abs": pair_max,
                "mean_abs": pair_mean,
                "off_argmax": off_argmax,
                "on_argmax": on_argmax,
                "argmax_equal": argmax_equal,
                "output_ids_exact": ids_equal,
                "output_text_exact": text_equal,
            })
    return {
        "row_count": len(rows),
        "pair_count": len(pairs),
        "max_abs": maximum,
        "mean_abs": worst_mean,
        "aggregate_mean_abs": total_abs / total_count,
        "argmax_equal": all_argmax,
        "output_ids_exact": all_ids,
        "output_text_exact": all_text,
        "pairs": pairs,
    }


def _metrics(off_rows, on_rows) -> dict:
    off_tpot = [
        float(value)
        for row in off_rows
        for value in row["tpot_samples_ns"]
    ]
    on_tpot = [
        float(value)
        for row in on_rows
        for value in row["tpot_samples_ns"]
    ]
    off_median = statistics.median(off_tpot)
    on_median = statistics.median(on_tpot)
    off_p95 = _nearest_rank(off_tpot, 0.95)
    on_p95 = _nearest_rank(on_tpot, 0.95)
    off_p99 = _nearest_rank(off_tpot, 0.99)
    on_p99 = _nearest_rank(on_tpot, 0.99)
    off_ttft = statistics.median(
        float(row["ttft_ns"]) for row in off_rows
    )
    on_ttft = statistics.median(
        float(row["ttft_ns"]) for row in on_rows
    )
    off_e2e = statistics.median(
        float(row["e2e_ns"]) for row in off_rows
    )
    on_e2e = statistics.median(
        float(row["e2e_ns"]) for row in on_rows
    )
    off_rate = statistics.median(
        float(row["output_tokens_per_second"])
        for row in off_rows
    )
    on_rate = statistics.median(
        float(row["output_tokens_per_second"])
        for row in on_rows
    )
    off_allocated = max(
        int(row["cuda_peak_allocated_bytes"])
        for row in off_rows
    )
    on_allocated = max(
        int(row["cuda_peak_allocated_bytes"])
        for row in on_rows
    )
    off_reserved = max(
        int(row["cuda_peak_reserved_bytes"])
        for row in off_rows
    )
    on_reserved = max(
        int(row["cuda_peak_reserved_bytes"])
        for row in on_rows
    )
    return {
        "sample_count_per_policy": len(off_tpot),
        "off_tpot_median_ns": off_median,
        "on_tpot_median_ns": on_median,
        "tpot_median_improvement_fraction":
            _improvement(off_median, on_median),
        "off_tpot_p95_ns": off_p95,
        "on_tpot_p95_ns": on_p95,
        "tpot_p95_improvement_fraction":
            _improvement(off_p95, on_p95),
        "off_tpot_p99_ns": off_p99,
        "on_tpot_p99_ns": on_p99,
        "tpot_p99_improvement_fraction":
            _improvement(off_p99, on_p99),
        "off_ttft_median_ns": off_ttft,
        "on_ttft_median_ns": on_ttft,
        "ttft_regression_fraction":
            _relative_change(off_ttft, on_ttft),
        "off_e2e_median_ns": off_e2e,
        "on_e2e_median_ns": on_e2e,
        "e2e_regression_fraction":
            _relative_change(off_e2e, on_e2e),
        "off_output_tokens_per_second_median": off_rate,
        "on_output_tokens_per_second_median": on_rate,
        "throughput_regression_fraction":
            _relative_change(on_rate, off_rate),
        "off_cuda_peak_allocated_bytes": off_allocated,
        "on_cuda_peak_allocated_bytes": on_allocated,
        "cuda_allocated_delta_bytes": on_allocated - off_allocated,
        "off_cuda_peak_reserved_bytes": off_reserved,
        "on_cuda_peak_reserved_bytes": on_reserved,
        "cuda_reserved_delta_bytes": on_reserved - off_reserved,
        "cuda_reserved_regression_fraction":
            _relative_change(off_reserved, on_reserved),
    }


def _reconstruct_comparison(
    rows,
    correctness_rows,
    *,
    run_dir: Path,
) -> dict:
    rows = _validate_performance_rows(rows)
    by_identity = {
        (
            row["context_bucket"],
            row["repetition"],
            row["policy"],
        ): row
        for row in rows
    }
    exact_outputs = True
    optimized_complete = True
    for bucket, _prompt, generated in CONTEXTS:
        for repetition in range(5):
            off = by_identity[(bucket, repetition, "off")]
            on = by_identity[(bucket, repetition, "on")]
            exact_outputs = exact_outputs and (
                off["output_token_ids"] == on["output_token_ids"]
                and off["output_text_sha256"]
                == on["output_text_sha256"]
            )
            off_summary = off["greedy_fast_path_summary"]
            on_summary = on["greedy_fast_path_summary"]
            optimized_complete = optimized_complete and (
                off_summary["eligible_steps"] == 0
                and off_summary["optimized_steps"] == 0
                and on_summary["eligible_steps"] == generated
                and on_summary["optimized_steps"] == generated
                and not on_summary["fallback_counts"]
            )
    correctness = _validate_correctness_rows(
        correctness_rows,
        run_dir=run_dir,
    )
    correctness_passed = (
        exact_outputs
        and correctness["output_ids_exact"]
        and correctness["output_text_exact"]
        and correctness["max_abs"] <= LOGIT_MAX_LIMIT
        and correctness["mean_abs"] <= LOGIT_MEAN_LIMIT
        and correctness["argmax_equal"]
    )
    by_bucket = {}
    for bucket, _prompt, _generated in CONTEXTS:
        selected = [
            row for row in rows
            if row["context_bucket"] == bucket
        ]
        by_bucket[bucket] = _metrics(
            [row for row in selected if row["policy"] == "off"],
            [row for row in selected if row["policy"] == "on"],
        )
    aggregate = _metrics(
        [row for row in rows if row["policy"] == "off"],
        [row for row in rows if row["policy"] == "on"],
    )
    winning = sum(
        metric["tpot_median_improvement_fraction"]
        >= MEDIAN_THRESHOLD
        for metric in by_bucket.values()
    )
    regressions = []
    for bucket, metric in by_bucket.items():
        if (
            metric["tpot_median_improvement_fraction"]
            < -TPOT_REGRESSION_LIMIT
        ):
            regressions.append(f"{bucket}:median_tpot")
        if (
            metric["tpot_p95_improvement_fraction"]
            < -TPOT_REGRESSION_LIMIT
        ):
            regressions.append(f"{bucket}:p95_tpot")
        if (
            metric["ttft_regression_fraction"]
            > LATENCY_REGRESSION_LIMIT
        ):
            regressions.append(f"{bucket}:ttft")
        if (
            metric["e2e_regression_fraction"]
            > LATENCY_REGRESSION_LIMIT
        ):
            regressions.append(f"{bucket}:e2e")
        if (
            metric["throughput_regression_fraction"]
            > THROUGHPUT_REGRESSION_LIMIT
        ):
            regressions.append(f"{bucket}:throughput")
    if (
        aggregate["cuda_reserved_regression_fraction"]
        > MEMORY_REGRESSION_LIMIT
    ):
        regressions.append("aggregate:cuda_reserved")
    if not correctness_passed:
        classification = "NO_GO_CORRECTNESS"
    elif not optimized_complete:
        classification = (
            "NO_GO_OPTIMIZED_PATH_INCOMPLETE"
        )
    elif winning < 2:
        classification = "NO_GO_TPOT_MEDIAN"
    elif (
        aggregate["tpot_p95_improvement_fraction"]
        < P95_THRESHOLD
    ):
        classification = "NO_GO_TPOT_P95"
    elif regressions:
        classification = "NO_GO_PROTECTED_REGRESSION"
    else:
        classification = (
            "GO_ZERO_TEMPERATURE_GREEDY_FAST_PATH"
        )
    on_summaries = [
        row["greedy_fast_path_summary"]
        for row in rows
        if row["policy"] == "on"
    ]
    avoided_work = {
        field: sum(summary[field] for summary in on_summaries)
        for field in (
            "avoided_temperature_h2d_bytes",
            "avoided_softmax_calls",
            "avoided_gumbel_rng_calls",
            "avoided_stochastic_divisions",
            "avoided_stochastic_argmax_calls",
            "avoided_where_calls",
        )
    }
    return {
        "schema_version": COMPARISON_SCHEMA,
        "run_tag": rows[0]["run_tag"],
        "source_commit": rows[0]["source_commit"],
        "classification": classification,
        "correctness_passed": correctness_passed,
        "optimized_path_complete": optimized_complete,
        "median_tpot_winning_bucket_count": winning,
        "protected_regressions": regressions,
        "thresholds": {
            "logit_max_abs_limit": LOGIT_MAX_LIMIT,
            "logit_mean_abs_limit": LOGIT_MEAN_LIMIT,
            "median_tpot_min_improvement_fraction":
                MEDIAN_THRESHOLD,
            "aggregate_p95_min_improvement_fraction":
                P95_THRESHOLD,
            "tpot_max_regression_fraction":
                TPOT_REGRESSION_LIMIT,
            "latency_max_regression_fraction":
                LATENCY_REGRESSION_LIMIT,
            "throughput_max_regression_fraction":
                THROUGHPUT_REGRESSION_LIMIT,
            "reserved_memory_max_regression_fraction":
                MEMORY_REGRESSION_LIMIT,
        },
        "correctness": correctness,
        "by_bucket": by_bucket,
        "aggregate": aggregate,
        "cost": {
            "persistent_cuda_memory_delta_bytes": 0,
            "host_counter_integer_fields": 8,
            "host_fallback_counter_mapping": True,
            "avoided_work": avoided_work,
            "cuda_peak_allocated_delta_bytes":
                aggregate["cuda_allocated_delta_bytes"],
            "cuda_peak_reserved_delta_bytes":
                aggregate["cuda_reserved_delta_bytes"],
        },
    }


def _validate_manifest(
    run_dir: Path,
    manifest,
    correctness_rows,
) -> None:
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA
    ):
        raise ValueError("manifest schema mismatch")
    sidecars = {
        row.get("logits_path")
        for row in correctness_rows
    }
    if None in sidecars:
        raise ValueError("sidecar path mismatch")
    expected = PRIMARY_ARTIFACTS | sidecars
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected:
        raise ValueError("manifest file inventory mismatch")
    for name in sorted(expected):
        actual = _sha256_file(run_dir / name)
        if artifacts[name] != actual:
            raise ValueError(
                f"manifest digest mismatch: {name}"
            )


def _validate_source(repo_root: Path, source) -> None:
    if (
        not isinstance(source, dict)
        or source.get("schema_version") != SOURCE_SCHEMA
        or set(source.get("source_sha256", {}))
        != set(SOURCE_FILES)
    ):
        raise ValueError("source manifest mismatch")
    for relative in SOURCE_FILES:
        if source["source_sha256"][relative] != _sha256_file(
            repo_root / relative
        ):
            raise ValueError(
                f"source digest mismatch: {relative}"
            )


def _validate_workload(workload) -> None:
    if (
        not isinstance(workload, dict)
        or workload.get("schema_version") != WORKLOAD_SCHEMA
    ):
        raise ValueError("workload manifest mismatch")
    expected_cases = [
        {
            "context_bucket": bucket,
            "prompt_tokens": prompt,
            "generated_tokens": generated,
        }
        for bucket, prompt, generated in CONTEXTS
    ]
    expected = {
        "context_cases": expected_cases,
        "repetitions": 5,
        "warmup_repetitions": 2,
        "batch_size": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "policy_order": {
            str(index): (
                ["off", "on"]
                if index % 2 == 0
                else ["on", "off"]
            )
            for index in range(5)
        },
        "correctness_sampling_points": list(POINTS),
    }
    for field, value in expected.items():
        if workload.get(field) != value:
            raise ValueError(
                f"workload manifest mismatch: {field}"
            )


def verify_bundle(
    run_dir: Path,
    *,
    repo_root: Path,
) -> dict:
    run_dir = Path(run_dir)
    rows = _load_jsonl(run_dir / "case_rows.jsonl")
    correctness_rows = _load_jsonl(
        run_dir / "correctness_rows.jsonl"
    )
    manifest_path = run_dir / "manifest.sha256"
    manifest = _load_json(manifest_path)
    _validate_manifest(run_dir, manifest, correctness_rows)
    source = _load_json(run_dir / "source_manifest.json")
    workload = _load_json(run_dir / "workload_manifest.json")
    comparison = _load_json(run_dir / "comparison.json")
    gate = _load_json(run_dir / "gate.json")
    summary = _load_json(run_dir / "summary.json")
    _validate_source(Path(repo_root), source)
    _validate_workload(workload)
    reconstructed = _reconstruct_comparison(
        rows,
        correctness_rows,
        run_dir=run_dir,
    )
    if comparison != reconstructed:
        raise ValueError("comparison drift")
    if (
        not isinstance(gate, dict)
        or gate.get("schema_version") != GATE_SCHEMA
        or gate.get("classification")
        != reconstructed["classification"]
        or gate.get("run_tag") != reconstructed["run_tag"]
        or gate.get("source_commit")
        != reconstructed["source_commit"]
        or gate.get("comparison_sha256")
        != _sha256_file(run_dir / "comparison.json")
    ):
        raise ValueError("classification drift")
    if (
        summary.get("row_count") != 30
        or summary.get("pair_count") != 15
        or summary.get("correctness_row_count") != 18
        or summary.get("all_outputs_exact") is not True
    ):
        raise ValueError("worker summary drift")
    identities = {
        manifest.get("run_tag"),
        source.get("run_tag"),
        workload.get("run_tag"),
        reconstructed.get("run_tag"),
        gate.get("run_tag"),
    }
    commits = {
        manifest.get("source_commit"),
        source.get("source_commit"),
        workload.get("source_commit"),
        reconstructed.get("source_commit"),
        gate.get("source_commit"),
    }
    if len(identities) != 1 or len(commits) != 1:
        raise ValueError("source-bound identity mismatch")
    result = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS",
        "run_tag": reconstructed["run_tag"],
        "source_commit": reconstructed["source_commit"],
        "reconstructed_classification":
            reconstructed["classification"],
        "comparison_sha256": _sha256_file(
            run_dir / "comparison.json"
        ),
        "manifest_sha256": _sha256_file(manifest_path),
    }
    output = run_dir / "independent-verification.json"
    output.write_text(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = verify_bundle(
        Path(args.run_dir),
        repo_root=Path(args.repo_root),
    )
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
