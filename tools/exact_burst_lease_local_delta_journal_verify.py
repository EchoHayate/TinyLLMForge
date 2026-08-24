#!/usr/bin/env python3
"""Independent verifier for lease-local delta-journal evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


VERIFY_SCHEMA = (
    "exact_burst_lease_local_delta_journal_verify_v1"
)
GATE_SCHEMA = "exact_burst_lease_local_delta_journal_gate_v1"
PERFORMANCE_ROW_SCHEMA = (
    "exact_burst_lease_local_delta_journal_performance_v1"
)
CORRECTNESS_ROW_SCHEMA = (
    "exact_burst_lease_local_delta_journal_correctness_v1"
)
WORKLOAD_SCHEMA = (
    "exact_burst_lease_local_delta_journal_workload_v1"
)
SOURCE_SCHEMA = (
    "exact_burst_lease_local_delta_journal_source_v1"
)
RUNNER_SCHEMA = (
    "exact_burst_lease_local_delta_journal_runner_v1"
)
POLICIES = ("generic", "lease_local_delta")
CONTEXTS = ("short", "medium", "long")
SAMPLING_POINTS = (
    "prefill-final",
    "decode-first",
    "decode-middle",
    "decode-final",
)
PERFORMANCE_REPETITIONS = 10
PERFORMANCE_ROW_COUNT = 60
CORRECTNESS_ROW_COUNT = 24
ALLOWED_FALLBACK_REASONS = {
    "terminal_suffix",
    "write_block_position_mismatch",
    "write_block_already_published",
    "predecessor_hash_unavailable",
    "unsupported_phase_shape",
}
REQUIRED_FILES = (
    "workload_manifest.json",
    "performance_rows.jsonl",
    "correctness_rows.jsonl",
    "phase_samples.jsonl",
    "summary.json",
    "source_manifest.json",
    "runner_receipt.json",
)


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def _load_json(path: Path):
    if not path.is_file():
        raise ValueError(f"required artifact is missing: {path.name}")
    return json.loads(
        path.read_text(),
        parse_constant=_reject_constant,
    )


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"required artifact is missing: {path.name}")
    return [
        json.loads(line, parse_constant=_reject_constant)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nearest_rank(values, percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or any(
        not math.isfinite(value) for value in ordered
    ):
        raise ValueError("metric samples must be finite and non-empty")
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _median(values) -> float:
    normalized = [float(value) for value in values]
    if not normalized or any(
        not math.isfinite(value) for value in normalized
    ):
        raise ValueError("metric samples must be finite and non-empty")
    return float(statistics.median(normalized))


def _regression_pct(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        if candidate == baseline:
            return 0.0
        raise ValueError("metric baseline must be positive")
    return (candidate - baseline) / baseline * 100.0


def _throughput_regression_pct(
    baseline: float,
    candidate: float,
) -> float:
    if baseline <= 0:
        if candidate == baseline:
            return 0.0
        raise ValueError("throughput baseline must be positive")
    return (baseline - candidate) / baseline * 100.0


def _performance_index(rows: list[dict]) -> dict:
    expected = {
        (repetition, context, policy)
        for repetition in range(PERFORMANCE_REPETITIONS)
        for context in CONTEXTS
        for policy in POLICIES
    }
    indexed = {}
    for row in rows:
        if row.get("schema") != PERFORMANCE_ROW_SCHEMA:
            raise ValueError("performance row schema mismatch")
        fallback_counts = row.get("delta_fallbacks")
        if not isinstance(fallback_counts, dict):
            raise ValueError("delta fallback counts are invalid")
        unknown = set(fallback_counts) - ALLOWED_FALLBACK_REASONS
        if unknown:
            raise ValueError("unknown fallback reason")
        key = (
            row.get("repetition"),
            row.get("context"),
            row.get("policy"),
        )
        if key in indexed:
            raise ValueError("duplicate performance row")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("performance row inventory is incomplete")
    return indexed


def _correctness_index(rows: list[dict]) -> dict:
    expected = {
        (context, policy, sampling_point)
        for context in CONTEXTS
        for policy in POLICIES
        for sampling_point in SAMPLING_POINTS
    }
    indexed = {}
    for row in rows:
        if row.get("schema") != CORRECTNESS_ROW_SCHEMA:
            raise ValueError("correctness row schema mismatch")
        key = (
            row.get("context"),
            row.get("policy"),
            row.get("sampling_point"),
        )
        if key in indexed:
            raise ValueError("duplicate correctness row")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("correctness row inventory is incomplete")
    return indexed


def _classification(metrics: dict) -> str:
    if (
        metrics["performance_row_count"]
        != PERFORMANCE_ROW_COUNT
        or metrics["correctness_row_count"]
        != CORRECTNESS_ROW_COUNT
    ):
        return "NO_GO_EVIDENCE_INCOMPLETE"
    if (
        not metrics["output_exact"]
        or metrics["sampled_logit_max_abs_diff"] != 0.0
        or not metrics["forward_inventory_equal"]
        or not metrics["replay_inventory_equal"]
        or not metrics["d2h_call_inventory_equal"]
        or not metrics["d2h_byte_inventory_equal"]
    ):
        return "NO_GO_CORRECTNESS"
    if (
        metrics["candidate_fallbacks"] != 0
        or metrics["candidate_rollbacks"] != 0
    ):
        return "NO_GO_TRANSACTIONAL_SAFETY"
    if (
        metrics["long_prepare_median_improvement_pct"] < 50.0
        or metrics["long_prepare_p95_improvement_pct"] < 50.0
        or any(
            metrics[field] > 3.0
            for field in (
                "short_prepare_median_regression_pct",
                "short_prepare_p95_regression_pct",
                "medium_prepare_median_regression_pct",
                "medium_prepare_p95_regression_pct",
                "aggregate_tpot_median_regression_pct",
                "aggregate_tpot_p95_regression_pct",
                "aggregate_ttft_regression_pct",
                "aggregate_e2e_regression_pct",
                "throughput_regression_pct",
            )
        )
        or metrics["reserved_memory_regression_pct"] > 1.0
    ):
        return "NO_GO_PERFORMANCE"
    return "GO_EXACT_BURST_LEASE_LOCAL_DELTA_JOURNAL"


def _reconstruct(
    performance_rows: list[dict],
    correctness_rows: list[dict],
) -> dict:
    performance = _performance_index(performance_rows)
    correctness = _correctness_index(correctness_rows)
    all_rows = performance_rows + correctness_rows
    run_tags = {row.get("run_tag") for row in all_rows}
    source_shas = {row.get("source_sha") for row in all_rows}
    if len(run_tags) != 1 or None in run_tags:
        raise ValueError("run tag authority mismatch")
    if len(source_shas) != 1 or None in source_shas:
        raise ValueError("source SHA authority mismatch")
    output_exact = True
    inventory_fields = (
        "target_model_forwards",
        "graph_replays",
        "d2h_calls",
        "d2h_bytes",
    )
    inventory_equal = {
        field: True for field in inventory_fields
    }
    for repetition in range(PERFORMANCE_REPETITIONS):
        for context in CONTEXTS:
            generic = performance[
                (repetition, context, "generic")
            ]
            candidate = performance[
                (repetition, context, "lease_local_delta")
            ]
            output_exact = output_exact and (
                generic["output_tokens"]
                == candidate["output_tokens"]
            )
            for field in inventory_fields:
                inventory_equal[field] = (
                    inventory_equal[field]
                    and generic[field] == candidate[field]
                )
    maximum_logit_diff = 0.0
    for context in CONTEXTS:
        for sampling_point in SAMPLING_POINTS:
            generic = correctness[
                (context, "generic", sampling_point)
            ]
            candidate = correctness[
                (
                    context,
                    "lease_local_delta",
                    sampling_point,
                )
            ]
            output_exact = output_exact and (
                generic["output_token_ids"]
                == candidate["output_token_ids"]
            )
            left = generic["sampled_logits"]
            right = candidate["sampled_logits"]
            if len(left) != len(right):
                output_exact = False
            else:
                maximum_logit_diff = max(
                    maximum_logit_diff,
                    max(
                        (
                            abs(float(a) - float(b))
                            for a, b in zip(left, right)
                        ),
                        default=0.0,
                    ),
                )
            for field in inventory_fields:
                inventory_equal[field] = (
                    inventory_equal[field]
                    and generic[field] == candidate[field]
                )
    result = {
        "schema": GATE_SCHEMA,
        "run_tag": next(iter(run_tags)),
        "source_sha": next(iter(source_shas)),
        "performance_row_count": len(performance_rows),
        "correctness_row_count": len(correctness_rows),
        "output_exact": output_exact,
        "sampled_logit_max_abs_diff": maximum_logit_diff,
        "forward_inventory_equal": inventory_equal[
            "target_model_forwards"
        ],
        "replay_inventory_equal": inventory_equal[
            "graph_replays"
        ],
        "d2h_call_inventory_equal": inventory_equal[
            "d2h_calls"
        ],
        "d2h_byte_inventory_equal": inventory_equal[
            "d2h_bytes"
        ],
    }
    for context in CONTEXTS:
        generic = [
            value
            for repetition in range(PERFORMANCE_REPETITIONS)
            for value in performance[
                (repetition, context, "generic")
            ]["phase_prepare_ns"]
        ]
        candidate = [
            value
            for repetition in range(PERFORMANCE_REPETITIONS)
            for value in performance[
                (
                    repetition,
                    context,
                    "lease_local_delta",
                )
            ]["phase_prepare_ns"]
        ]
        generic_median = _median(generic)
        candidate_median = _median(candidate)
        generic_p95 = _nearest_rank(generic, 0.95)
        candidate_p95 = _nearest_rank(candidate, 0.95)
        result[
            f"{context}_prepare_median_regression_pct"
        ] = _regression_pct(generic_median, candidate_median)
        result[
            f"{context}_prepare_p95_regression_pct"
        ] = _regression_pct(generic_p95, candidate_p95)
        result[
            f"{context}_prepare_median_improvement_pct"
        ] = -result[
            f"{context}_prepare_median_regression_pct"
        ]
        result[
            f"{context}_prepare_p95_improvement_pct"
        ] = -result[
            f"{context}_prepare_p95_regression_pct"
        ]

    def aggregate(field, *, throughput=False):
        baseline = _median(
            row[field]
            for row in performance_rows
            if row["policy"] == "generic"
        )
        candidate = _median(
            row[field]
            for row in performance_rows
            if row["policy"] == "lease_local_delta"
        )
        return (
            _throughput_regression_pct(baseline, candidate)
            if throughput
            else _regression_pct(baseline, candidate)
        )

    result["aggregate_tpot_median_regression_pct"] = (
        _regression_pct(
            _median(
                _median(row["tpot_samples_ns"])
                for row in performance_rows
                if row["policy"] == "generic"
            ),
            _median(
                _median(row["tpot_samples_ns"])
                for row in performance_rows
                if row["policy"] == "lease_local_delta"
            ),
        )
    )
    result["aggregate_tpot_p95_regression_pct"] = (
        _regression_pct(
            _median(
                _nearest_rank(row["tpot_samples_ns"], 0.95)
                for row in performance_rows
                if row["policy"] == "generic"
            ),
            _median(
                _nearest_rank(row["tpot_samples_ns"], 0.95)
                for row in performance_rows
                if row["policy"] == "lease_local_delta"
            ),
        )
    )
    result["aggregate_ttft_regression_pct"] = aggregate(
        "ttft_ns"
    )
    result["aggregate_e2e_regression_pct"] = aggregate(
        "e2e_ns"
    )
    result["throughput_regression_pct"] = aggregate(
        "output_tokens_per_second",
        throughput=True,
    )
    baseline_reserved = max(
        int(row["cuda_peak_reserved_bytes"])
        for row in performance_rows
        if row["policy"] == "generic"
    )
    candidate_reserved = max(
        int(row["cuda_peak_reserved_bytes"])
        for row in performance_rows
        if row["policy"] == "lease_local_delta"
    )
    result["reserved_memory_regression_pct"] = _regression_pct(
        baseline_reserved,
        candidate_reserved,
    )
    candidate_rows = [
        row
        for row in performance_rows
        if row["policy"] == "lease_local_delta"
    ]
    result["candidate_fallbacks"] = sum(
        sum(int(value) for value in row["delta_fallbacks"].values())
        for row in candidate_rows
    )
    result["candidate_rollbacks"] = sum(
        int(row["delta_rollbacks"])
        for row in candidate_rows
    )
    result["classification"] = _classification(result)
    return result


def _validate_phase_samples(
    rows: list[dict],
    performance_rows: list[dict],
) -> None:
    if len(rows) != PERFORMANCE_ROW_COUNT:
        raise ValueError("phase sample inventory is incomplete")
    expected = {
        (
            row["repetition"],
            row["context"],
            row["policy"],
        ): row["phase_prepare_ns"]
        for row in performance_rows
    }
    actual = {}
    for row in rows:
        key = (
            row.get("repetition"),
            row.get("context"),
            row.get("policy"),
        )
        if key in actual:
            raise ValueError("duplicate phase sample row")
        actual[key] = row.get("phase_prepare_ns")
    if actual != expected:
        raise ValueError("phase sample inventory mismatch")


def verify_artifact_directory(path: Path) -> dict:
    run_dir = Path(path)
    repo_root = Path(__file__).resolve().parents[1]
    for name in REQUIRED_FILES:
        if not (run_dir / name).is_file():
            raise ValueError(f"required artifact is missing: {name}")
    workload = _load_json(run_dir / "workload_manifest.json")
    source = _load_json(run_dir / "source_manifest.json")
    receipt = _load_json(run_dir / "runner_receipt.json")
    stored_summary = _load_json(run_dir / "summary.json")
    performance_rows = _load_jsonl(
        run_dir / "performance_rows.jsonl"
    )
    correctness_rows = _load_jsonl(
        run_dir / "correctness_rows.jsonl"
    )
    phase_rows = _load_jsonl(run_dir / "phase_samples.jsonl")
    if workload.get("schema") != WORKLOAD_SCHEMA:
        raise ValueError("workload schema mismatch")
    if source.get("schema") != SOURCE_SCHEMA:
        raise ValueError("source manifest schema mismatch")
    if receipt.get("schema") != RUNNER_SCHEMA:
        raise ValueError("runner receipt schema mismatch")
    if receipt.get("exit_code") != 0:
        raise ValueError("runner did not exit successfully")
    if (
        workload.get("performance_row_count")
        != PERFORMANCE_ROW_COUNT
        or workload.get("correctness_row_count")
        != CORRECTNESS_ROW_COUNT
    ):
        raise ValueError("workload inventory mismatch")
    declared_hashes = source.get("artifact_sha256")
    if not isinstance(declared_hashes, dict):
        raise ValueError("artifact digest manifest is missing")
    for name, expected_digest in declared_hashes.items():
        if _sha256(run_dir / name) != expected_digest:
            raise ValueError("artifact digest mismatch")
    source_hashes = source.get("source_file_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("source file digest manifest is missing")
    for relative, expected_digest in source_hashes.items():
        source_path = repo_root / relative
        if (
            source_path.resolve() == repo_root
            or repo_root not in source_path.resolve().parents
            or not source_path.is_file()
            or _sha256(source_path) != expected_digest
        ):
            raise ValueError("source file digest mismatch")
    patch_digest = source.get("source_patch_sha256")
    if (
        not isinstance(patch_digest, str)
        or len(patch_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in patch_digest
        )
    ):
        raise ValueError("source patch digest is invalid")
    reconstructed = _reconstruct(
        performance_rows,
        correctness_rows,
    )
    _validate_phase_samples(phase_rows, performance_rows)
    source_sha = reconstructed["source_sha"]
    if any(
        value != source_sha
        for value in (
            workload.get("source_sha"),
            source.get("source_sha"),
            receipt.get("source_sha"),
        )
    ):
        raise ValueError("source SHA authority mismatch")
    run_tag = reconstructed["run_tag"]
    if any(
        value != run_tag
        for value in (
            workload.get("run_tag"),
            source.get("run_tag"),
            receipt.get("run_tag"),
        )
    ):
        raise ValueError("run tag authority mismatch")
    if stored_summary != reconstructed:
        raise ValueError("summary mismatch")
    return {
        "schema": VERIFY_SCHEMA,
        "verified": True,
        "classification": reconstructed["classification"],
        "performance_row_count": len(performance_rows),
        "correctness_row_count": len(correctness_rows),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = verify_artifact_directory(
        args.artifact_directory
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True)
            + "\n"
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
