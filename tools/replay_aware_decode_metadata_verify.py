#!/usr/bin/env python3
"""Independent verification for replay-aware decode metadata evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


CASE_SCHEMA = "replay-aware-decode-metadata.case.v1"
COMPARISON_SCHEMA = (
    "replay-aware-decode-metadata.comparison.v1"
)
VERIFICATION_SCHEMA = (
    "replay-aware-decode-metadata.independent-verification.v1"
)
MANIFEST_SCHEMA = (
    "replay-aware-decode-metadata.manifest.v1"
)
PRIMARY_ARTIFACTS = {
    "case_rows.jsonl",
    "source_manifest.json",
    "workload_manifest.json",
    "summary.json",
    "comparison.json",
    "gate.json",
}
SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/decode_metadata_landing.py",
    "tinyvllm/engine/model_runner.py",
    "tools/profile_replay_aware_decode_metadata.py",
    "tools/test_profile_replay_aware_decode_metadata.py",
    "tools/replay_aware_decode_metadata_gate.py",
    "tools/test_replay_aware_decode_metadata_gate.py",
    "tools/replay_aware_decode_metadata_verify.py",
    "tools/test_replay_aware_decode_metadata_verify.py",
    "tools/run_replay_aware_decode_metadata_remote.py",
    "tools/test_run_replay_aware_decode_metadata_remote.py",
)
CONTEXTS = (
    ("short", 256, 128),
    ("medium", 2048, 128),
    ("long", 8192, 128),
)
MEDIAN_THRESHOLD = 0.05
P95_THRESHOLD = 0.05
TPOT_REGRESSION_LIMIT = 0.03
LATENCY_REGRESSION_LIMIT = 0.03
THROUGHPUT_REGRESSION_LIMIT = 0.02
MEMORY_REGRESSION_LIMIT = 0.01
PINNED_LIMIT_BYTES = 1_792


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


def _nearest_rank(values, percentile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty percentile input")
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _relative_change(baseline, candidate):
    baseline = float(baseline)
    candidate = float(candidate)
    if baseline <= 0:
        if baseline == candidate:
            return 0.0
        raise ValueError(
            "relative comparison baseline must be positive"
        )
    return (candidate - baseline) / baseline


def _improvement(baseline, candidate):
    return -_relative_change(baseline, candidate)


def _assert_finite_tree(value, path="root"):
    if isinstance(value, bool) or value is None:
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
    if not isinstance(value, str):
        raise ValueError(
            f"unsupported evidence type at {path}"
        )


def _validate_rows(rows):
    if len(rows) != 30:
        raise ValueError(
            "expected exactly 30 measured rows"
        )
    context_shapes = {
        bucket: (prompt, generated)
        for bucket, prompt, generated in CONTEXTS
    }
    identities = set()
    normalized = []
    for row in rows:
        _assert_finite_tree(row)
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != CASE_SCHEMA
        ):
            raise ValueError("case row schema mismatch")
        bucket = row.get("context_bucket")
        if bucket not in context_shapes:
            raise ValueError("context bucket mismatch")
        if (
            row.get("prompt_tokens"),
            row.get("generated_tokens"),
        ) != context_shapes[bucket]:
            raise ValueError("context shape mismatch")
        policy = row.get("policy")
        repetition = row.get("repetition")
        if (
            policy not in {"off", "on"}
            or isinstance(repetition, bool)
            or not isinstance(repetition, int)
            or repetition not in range(5)
        ):
            raise ValueError("case identity mismatch")
        identity = (bucket, repetition, policy)
        if identity in identities:
            raise ValueError("duplicate case identity")
        identities.add(identity)
        generated = row["generated_tokens"]
        if (
            not isinstance(row.get("output_token_ids"), list)
            or len(row["output_token_ids"]) != generated
            or not isinstance(row.get("output_text_sha256"), str)
            or len(row["output_text_sha256"]) != 64
        ):
            raise ValueError("output evidence mismatch")
        expected_decode = generated - 1
        for field in (
            "tpot_samples_ns",
            "decode_host_ns",
            "decode_cuda_ns",
        ):
            if (
                not isinstance(row.get(field), list)
                or len(row[field]) != expected_decode
            ):
                raise ValueError(
                    f"{field} inventory mismatch"
                )
        landing = row.get("landing_summary")
        if not isinstance(landing, dict):
            raise ValueError("landing summary mismatch")
        for field in (
            "eligible_steps",
            "optimized_steps",
            "peak_pinned_capacity_bytes",
            "fallback_counts",
        ):
            if field not in landing:
                raise ValueError(
                    "landing summary field mismatch"
                )
        if policy == "off" and (
            landing["eligible_steps"] != 0
            or landing["optimized_steps"] != 0
        ):
            raise ValueError(
                "disabled policy reached landing"
            )
        normalized.append(row)
    expected = {
        (bucket, repetition, policy)
        for bucket, _prompt, _generated in CONTEXTS
        for repetition in range(5)
        for policy in ("off", "on")
    }
    if identities != expected:
        raise ValueError("case inventory mismatch")
    if len({row["run_tag"] for row in normalized}) != 1:
        raise ValueError("run tag mismatch")
    if len(
        {row["source_commit"] for row in normalized}
    ) != 1:
        raise ValueError("source commit mismatch")
    return normalized


def _metrics(off_rows, on_rows):
    off_tpot = [
        float(sample)
        for row in off_rows
        for sample in row["tpot_samples_ns"]
    ]
    on_tpot = [
        float(sample)
        for row in on_rows
        for sample in row["tpot_samples_ns"]
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
        "off_cuda_peak_reserved_bytes": off_reserved,
        "on_cuda_peak_reserved_bytes": on_reserved,
        "cuda_reserved_regression_fraction":
            _relative_change(off_reserved, on_reserved),
    }


def _reconstruct_comparison(rows):
    rows = _validate_rows(rows)
    by_identity = {
        (
            row["context_bucket"],
            row["repetition"],
            row["policy"],
        ): row
        for row in rows
    }
    exact = True
    optimized = True
    for bucket, _prompt, generated in CONTEXTS:
        for repetition in range(5):
            off = by_identity[(bucket, repetition, "off")]
            on = by_identity[(bucket, repetition, "on")]
            exact = exact and (
                off["output_token_ids"]
                == on["output_token_ids"]
                and off["output_text_sha256"]
                == on["output_text_sha256"]
            )
            landing = on["landing_summary"]
            optimized = optimized and (
                landing["eligible_steps"] == generated - 1
                and landing["optimized_steps"] == generated - 1
                and landing["fallback_counts"] == {}
            )
    by_bucket = {}
    for bucket, _prompt, _generated in CONTEXTS:
        selected = [
            row for row in rows
            if row["context_bucket"] == bucket
        ]
        by_bucket[bucket] = _metrics(
            [
                row for row in selected
                if row["policy"] == "off"
            ],
            [
                row for row in selected
                if row["policy"] == "on"
            ],
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
    pinned_peak = max(
        row["landing_summary"][
            "peak_pinned_capacity_bytes"
        ]
        for row in rows
        if row["policy"] == "on"
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
    if pinned_peak > PINNED_LIMIT_BYTES:
        regressions.append("aggregate:pinned_capacity")
    if not exact:
        classification = "NO_GO_CORRECTNESS"
    elif not optimized:
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
        classification = "GO_REPLAY_AWARE_METADATA"
    return {
        "schema_version": COMPARISON_SCHEMA,
        "run_tag": rows[0]["run_tag"],
        "source_commit": rows[0]["source_commit"],
        "classification": classification,
        "exact_outputs": exact,
        "optimized_path_complete": optimized,
        "median_tpot_winning_bucket_count": winning,
        "pinned_peak_bytes": pinned_peak,
        "pinned_capacity_limit_bytes": PINNED_LIMIT_BYTES,
        "protected_regressions": regressions,
        "thresholds": {
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
        "by_bucket": by_bucket,
        "aggregate": aggregate,
    }


def _reconstruct_summary(rows):
    by_identity = {
        (
            row["context_bucket"],
            row["repetition"],
            row["policy"],
        ): row
        for row in rows
    }
    pair_keys = sorted({
        (bucket, repetition)
        for bucket, repetition, _policy in by_identity
    })
    pairs = []
    for bucket, repetition in pair_keys:
        off = by_identity[(bucket, repetition, "off")]
        on = by_identity[(bucket, repetition, "on")]
        if (
            off["output_token_ids"]
            != on["output_token_ids"]
            or off["output_text_sha256"]
            != on["output_text_sha256"]
        ):
            raise ValueError("worker summary is not reconstructable")
        pairs.append({
            "context_bucket": bucket,
            "repetition": repetition,
            "off_tpot_median_ns": statistics.median(
                off["tpot_samples_ns"]
            ),
            "on_tpot_median_ns": statistics.median(
                on["tpot_samples_ns"]
            ),
            "off_tpot_p95_ns": _nearest_rank(
                off["tpot_samples_ns"],
                0.95,
            ),
            "on_tpot_p95_ns": _nearest_rank(
                on["tpot_samples_ns"],
                0.95,
            ),
        })
    on_rows = [row for row in rows if row["policy"] == "on"]
    return {
        "schema_version":
            "replay-aware-decode-metadata.summary.v1",
        "run_tag": rows[0]["run_tag"],
        "source_commit": rows[0]["source_commit"],
        "row_count": len(rows),
        "pair_count": len(pairs),
        "all_outputs_exact": True,
        "all_on_steps_optimized": True,
        "peak_pinned_capacity_bytes": max(
            row["landing_summary"][
                "peak_pinned_capacity_bytes"
            ]
            for row in on_rows
        ),
        "pairs": pairs,
    }


def _verify_manifest(run_dir, manifest):
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or set(manifest.get("artifacts", {}))
        != PRIMARY_ARTIFACTS
    ):
        raise ValueError("manifest file inventory mismatch")
    for name in PRIMARY_ARTIFACTS:
        path = run_dir / name
        if not path.is_file():
            raise ValueError(
                f"primary artifact is missing: {name}"
            )
        actual = _sha256_file(path)
        if manifest["artifacts"][name] != actual:
            raise ValueError(
                f"manifest digest mismatch: {name}"
            )


def _verify_source(source, repo_root):
    if (
        not isinstance(source, dict)
        or source.get("schema_version")
        != "replay-aware-decode-metadata.source.v1"
        or set(source.get("source_sha256", {}))
        != set(SOURCE_FILES)
    ):
        raise ValueError("source manifest mismatch")
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise ValueError(
                f"source file is missing: {relative}"
            )
        if (
            source["source_sha256"][relative]
            != _sha256_file(path)
        ):
            raise ValueError(
                f"source digest mismatch: {relative}"
            )


def _verify_workload(workload):
    expected_cases = [
        {
            "context_bucket": bucket,
            "prompt_tokens": prompt,
            "generated_tokens": generated,
        }
        for bucket, prompt, generated in CONTEXTS
    ]
    if (
        not isinstance(workload, dict)
        or workload.get("schema_version")
        != "replay-aware-decode-metadata.workload.v1"
        or workload.get("context_cases") != expected_cases
        or workload.get("repetitions") != 5
        or workload.get("warmup_repetitions") != 2
        or workload.get("batch_size") != 1
        or workload.get("temperature") != 0.0
        or workload.get("ignore_eos") is not True
        or workload.get("policy_order") != {
            str(index): (
                ["off", "on"]
                if index % 2 == 0
                else ["on", "off"]
            )
            for index in range(5)
        }
    ):
        raise ValueError("workload manifest mismatch")


def verify_bundle(
    run_dir: Path,
    *,
    repo_root: Path,
) -> dict:
    run_dir = Path(run_dir)
    repo_root = Path(repo_root)
    manifest_path = run_dir / "manifest.sha256"
    manifest = _load_json(manifest_path)
    _verify_manifest(run_dir, manifest)
    source = _load_json(run_dir / "source_manifest.json")
    workload = _load_json(
        run_dir / "workload_manifest.json"
    )
    summary = _load_json(run_dir / "summary.json")
    comparison = _load_json(
        run_dir / "comparison.json"
    )
    gate = _load_json(run_dir / "gate.json")
    rows = _validate_rows(
        _load_jsonl(run_dir / "case_rows.jsonl")
    )
    _verify_source(source, repo_root)
    _verify_workload(workload)
    identities = {
        source.get("run_tag"),
        workload.get("run_tag"),
        manifest.get("run_tag"),
        comparison.get("run_tag"),
        gate.get("run_tag"),
        *(row.get("run_tag") for row in rows),
    }
    commits = {
        source.get("source_commit"),
        workload.get("source_commit"),
        manifest.get("source_commit"),
        comparison.get("source_commit"),
        gate.get("source_commit"),
        *(row.get("source_commit") for row in rows),
    }
    if len(identities) != 1 or len(commits) != 1:
        raise ValueError("source-bound identity mismatch")
    reconstructed_summary = _reconstruct_summary(rows)
    if summary != reconstructed_summary:
        raise ValueError("worker summary drift")
    reconstructed = _reconstruct_comparison(rows)
    if comparison != reconstructed:
        raise ValueError("comparison drift")
    if gate.get("classification") != reconstructed[
        "classification"
    ]:
        raise ValueError("classification drift")
    comparison_digest = _sha256_json(reconstructed)
    if gate.get("comparison_sha256") != comparison_digest:
        raise ValueError("gate comparison digest drift")
    return {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS",
        "run_tag": reconstructed["run_tag"],
        "source_commit": reconstructed["source_commit"],
        "reconstructed_classification": reconstructed[
            "classification"
        ],
        "comparison_sha256": comparison_digest,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _write_json(path, payload):
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    result = verify_bundle(
        run_dir,
        repo_root=Path(args.repo_root),
    )
    output = (
        Path(args.output)
        if args.output is not None
        else run_dir / "independent-verification.json"
    )
    _write_json(output, result)
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
