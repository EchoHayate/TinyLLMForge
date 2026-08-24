#!/usr/bin/env python3
"""Independent verifier for the one-phase lease-local journal gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


VERIFY_SCHEMA = (
    "exact_burst_one_phase_lease_local_journal_verify_v1"
)
GATE_SCHEMA = (
    "exact_burst_one_phase_lease_local_journal_gate_v1"
)
PERFORMANCE_ROW_SCHEMA = (
    "exact_burst_one_phase_lease_local_journal_performance_v1"
)
CORRECTNESS_ROW_SCHEMA = (
    "exact_burst_one_phase_lease_local_journal_correctness_v1"
)
WORKLOAD_SCHEMA = (
    "exact_burst_one_phase_lease_local_journal_workload_v1"
)
SOURCE_SCHEMA = (
    "exact_burst_one_phase_lease_local_journal_source_v1"
)
RUNNER_SCHEMA = (
    "exact_burst_one_phase_lease_local_journal_runner_v1"
)
GO = "GO_EXACT_BURST_ONE_PHASE_LEASE_LOCAL_JOURNAL"
NO_GO_PERFORMANCE = "NO_GO_PERFORMANCE"
NO_GO_CORRECTNESS = "NO_GO_CORRECTNESS"
NO_GO_TRANSACTIONAL_SAFETY = "NO_GO_TRANSACTIONAL_SAFETY"
NO_GO_EVIDENCE_INCOMPLETE = "NO_GO_EVIDENCE_INCOMPLETE"
EXPECTED_SOURCE_PATCH_SHA256 = hashlib.sha256(b"").hexdigest()
POLICIES = ("generic", "lease_local_delta")
CONTEXTS = ("2k", "4k", "8k")
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
    "unsupported_burst_shape",
    "terminal_one_phase",
    "write_block_boundary_crossed",
    "write_block_position_mismatch",
    "write_block_already_published",
    "predecessor_hash_unavailable",
}
SOURCE_FILES = {
    "tinyvllm/config.py",
    "tinyvllm/engine/block_manager.py",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
    "tinyvllm/engine/llm_engine.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tools/profile_exact_greedy_decode_burst.py",
    "tools/exact_burst_one_phase_lease_local_journal_gate.py",
    "tools/exact_burst_one_phase_lease_local_journal_verify.py",
}
HASHED_ARTIFACTS = {
    "workload_manifest.json",
    "performance_rows.jsonl",
    "correctness_rows.jsonl",
    "prepare_samples.jsonl",
}
REQUIRED_FILES = HASHED_ARTIFACTS | {
    "summary.json",
    "source_manifest.json",
    "runner_receipt.json",
}


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def _load_json(path: Path):
    if not path.is_file():
        raise ValueError(
            f"required artifact is missing: {path.name}"
        )
    return json.loads(
        path.read_text(),
        parse_constant=_reject_constant,
    )


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(
            f"required artifact is missing: {path.name}"
        )
    return [
        json.loads(line, parse_constant=_reject_constant)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_float(value, *, field: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be finite")
    return normalized


def _median(values) -> float:
    normalized = [
        _finite_float(value, field="metric sample")
        for value in values
    ]
    if not normalized:
        raise ValueError(
            "metric samples must be finite and non-empty"
        )
    return float(statistics.median(normalized))


def _nearest_rank(values, percentile: float) -> float:
    ordered = sorted(
        _finite_float(value, field="metric sample")
        for value in values
    )
    if not ordered:
        raise ValueError(
            "metric samples must be finite and non-empty"
        )
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _regression_pct(
    baseline: float,
    candidate: float,
) -> float:
    if baseline <= 0:
        if candidate == baseline:
            return 0.0
        raise ValueError("metric baseline must be positive")
    return (candidate - baseline) / baseline * 100.0


def _improvement_pct(
    baseline: float,
    candidate: float,
) -> float:
    return -_regression_pct(baseline, candidate)


def _throughput_regression_pct(
    baseline: float,
    candidate: float,
) -> float:
    if baseline <= 0:
        if candidate == baseline:
            return 0.0
        raise ValueError("throughput baseline must be positive")
    return (baseline - candidate) / baseline * 100.0


def _validate_performance(rows: list[dict]) -> dict:
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
        fallbacks = row.get("one_phase_fallbacks")
        if not isinstance(fallbacks, dict):
            raise ValueError(
                "one-phase fallback counts are invalid"
            )
        if set(fallbacks) - ALLOWED_FALLBACK_REASONS:
            raise ValueError("unknown fallback reason")
        for field in (
            "ttft_ns",
            "e2e_ns",
            "output_tokens_per_second",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
        ):
            _finite_float(row.get(field), field=field)
        for field in ("prepare_ns", "tpot_samples_ns"):
            values = row.get(field)
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"{field} must be finite and non-empty"
                )
            for value in values:
                _finite_float(value, field=field)
        key = (
            row.get("repetition"),
            row.get("context"),
            row.get("policy"),
        )
        if key in indexed:
            raise ValueError("duplicate performance row")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError(
            "performance row inventory is incomplete"
        )
    return indexed


def _validate_correctness(rows: list[dict]) -> dict:
    expected = {
        (context, policy, point)
        for context in CONTEXTS
        for policy in POLICIES
        for point in SAMPLING_POINTS
    }
    indexed = {}
    for row in rows:
        if row.get("schema") != CORRECTNESS_ROW_SCHEMA:
            raise ValueError("correctness row schema mismatch")
        logits = row.get("sampled_logits")
        if not isinstance(logits, list) or not logits:
            raise ValueError(
                "sampled logits must be finite and non-empty"
            )
        for value in logits:
            _finite_float(value, field="sampled logits")
        key = (
            row.get("context"),
            row.get("policy"),
            row.get("sampling_point"),
        )
        if key in indexed:
            raise ValueError("duplicate correctness row")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError(
            "correctness row inventory is incomplete"
        )
    return indexed


def _same_execution_inventory(left: dict, right: dict) -> bool:
    return all(
        left[field] == right[field]
        for field in (
            "target_model_forwards",
            "graph_replays",
            "d2h_calls",
            "d2h_bytes",
        )
    )


def _classification(metrics: dict) -> str:
    if (
        metrics["performance_row_count"]
        != PERFORMANCE_ROW_COUNT
        or metrics["correctness_row_count"]
        != CORRECTNESS_ROW_COUNT
    ):
        return NO_GO_EVIDENCE_INCOMPLETE
    if (
        metrics["output_exact"] is not True
        or metrics["sampled_argmax_exact"] is not True
        or metrics["sampled_logit_max_abs_diff"] != 0.0
        or metrics["execution_inventory_equal"] is not True
    ):
        return NO_GO_CORRECTNESS
    if (
        metrics["candidate_counter_authority"] is not True
        or metrics["candidate_generic_journal_captures"] != 0
        or metrics["candidate_one_phase_fallbacks"] != 0
        or metrics["candidate_one_phase_rollbacks"] != 0
    ):
        return NO_GO_TRANSACTIONAL_SAFETY
    if (
        metrics["8k_prepare_median_improvement_pct"] < 50.0
        or metrics["8k_prepare_p95_improvement_pct"] < 50.0
        or metrics[
            "aggregate_prepare_median_improvement_pct"
        ]
        < 35.0
        or metrics["aggregate_prepare_p95_improvement_pct"]
        < 35.0
        or metrics["aggregate_tpot_median_improvement_pct"]
        < 1.0
        or metrics["aggregate_tpot_p95_improvement_pct"]
        < 1.0
        or any(
            metrics[field] > 2.0
            for field in (
                "aggregate_tpot_p99_regression_pct",
                "aggregate_ttft_regression_pct",
                "aggregate_e2e_regression_pct",
                "throughput_regression_pct",
            )
        )
        or any(
            metrics[field] > 1.0
            for field in (
                "allocated_memory_regression_pct",
                "reserved_memory_regression_pct",
            )
        )
    ):
        return NO_GO_PERFORMANCE
    return GO


def _reconstruct(
    performance_rows: list[dict],
    correctness_rows: list[dict],
) -> dict:
    performance = _validate_performance(performance_rows)
    correctness = _validate_correctness(correctness_rows)
    all_rows = performance_rows + correctness_rows
    run_tags = {row.get("run_tag") for row in all_rows}
    source_shas = {row.get("source_sha") for row in all_rows}
    if len(run_tags) != 1 or None in run_tags:
        raise ValueError("run tag authority mismatch")
    if len(source_shas) != 1 or None in source_shas:
        raise ValueError("source SHA authority mismatch")

    output_exact = True
    sampled_argmax_exact = True
    sampled_logit_max_abs_diff = 0.0
    inventory_equal = True
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
            inventory_equal = (
                inventory_equal
                and _same_execution_inventory(
                    generic,
                    candidate,
                )
            )
    for context in CONTEXTS:
        for point in SAMPLING_POINTS:
            generic = correctness[(context, "generic", point)]
            candidate = correctness[
                (context, "lease_local_delta", point)
            ]
            output_exact = output_exact and (
                generic["output_token_ids"]
                == candidate["output_token_ids"]
            )
            sampled_argmax_exact = (
                sampled_argmax_exact
                and generic["sampled_argmax"]
                == candidate["sampled_argmax"]
            )
            left = generic["sampled_logits"]
            right = candidate["sampled_logits"]
            if len(left) != len(right):
                output_exact = False
            else:
                sampled_logit_max_abs_diff = max(
                    sampled_logit_max_abs_diff,
                    max(
                        (
                            abs(float(a) - float(b))
                            for a, b in zip(left, right)
                        ),
                        default=0.0,
                    ),
                )
            inventory_equal = (
                inventory_equal
                and _same_execution_inventory(
                    generic,
                    candidate,
                )
            )

    metrics = {
        "schema": GATE_SCHEMA,
        "run_tag": next(iter(run_tags)),
        "source_sha": next(iter(source_shas)),
        "performance_row_count": len(performance_rows),
        "correctness_row_count": len(correctness_rows),
        "output_exact": output_exact,
        "sampled_argmax_exact": sampled_argmax_exact,
        "sampled_logit_max_abs_diff": sampled_logit_max_abs_diff,
        "execution_inventory_equal": inventory_equal,
    }
    for context in CONTEXTS:
        generic = [
            sample
            for row in performance_rows
            if row["policy"] == "generic"
            and row["context"] == context
            for sample in row["prepare_ns"]
        ]
        candidate = [
            sample
            for row in performance_rows
            if row["policy"] == "lease_local_delta"
            and row["context"] == context
            for sample in row["prepare_ns"]
        ]
        metrics[
            f"{context}_prepare_median_improvement_pct"
        ] = _improvement_pct(
            _median(generic),
            _median(candidate),
        )
        metrics[
            f"{context}_prepare_p95_improvement_pct"
        ] = _improvement_pct(
            _nearest_rank(generic, 0.95),
            _nearest_rank(candidate, 0.95),
        )

    generic_prepare = [
        sample
        for row in performance_rows
        if row["policy"] == "generic"
        for sample in row["prepare_ns"]
    ]
    candidate_prepare = [
        sample
        for row in performance_rows
        if row["policy"] == "lease_local_delta"
        for sample in row["prepare_ns"]
    ]
    metrics["aggregate_prepare_median_improvement_pct"] = (
        _improvement_pct(
            _median(generic_prepare),
            _median(candidate_prepare),
        )
    )
    metrics["aggregate_prepare_p95_improvement_pct"] = (
        _improvement_pct(
            _nearest_rank(generic_prepare, 0.95),
            _nearest_rank(candidate_prepare, 0.95),
        )
    )

    def per_row_stat(policy, field, percentile=None):
        values = []
        for row in performance_rows:
            if row["policy"] != policy:
                continue
            if percentile is None:
                values.append(float(row[field]))
            else:
                values.append(
                    _nearest_rank(row[field], percentile)
                )
        return _median(values)

    generic_tpot_median = _median(
        _median(row["tpot_samples_ns"])
        for row in performance_rows
        if row["policy"] == "generic"
    )
    candidate_tpot_median = _median(
        _median(row["tpot_samples_ns"])
        for row in performance_rows
        if row["policy"] == "lease_local_delta"
    )
    metrics["aggregate_tpot_median_improvement_pct"] = (
        _improvement_pct(
            generic_tpot_median,
            candidate_tpot_median,
        )
    )
    for percentile, label in ((0.95, "p95"), (0.99, "p99")):
        baseline = per_row_stat(
            "generic",
            "tpot_samples_ns",
            percentile,
        )
        candidate = per_row_stat(
            "lease_local_delta",
            "tpot_samples_ns",
            percentile,
        )
        metrics[
            f"aggregate_tpot_{label}_improvement_pct"
        ] = _improvement_pct(baseline, candidate)
        metrics[
            f"aggregate_tpot_{label}_regression_pct"
        ] = _regression_pct(baseline, candidate)
    for field, output in (
        ("ttft_ns", "aggregate_ttft_regression_pct"),
        ("e2e_ns", "aggregate_e2e_regression_pct"),
    ):
        metrics[output] = _regression_pct(
            per_row_stat("generic", field),
            per_row_stat("lease_local_delta", field),
        )
    metrics["throughput_regression_pct"] = (
        _throughput_regression_pct(
            per_row_stat(
                "generic",
                "output_tokens_per_second",
            ),
            per_row_stat(
                "lease_local_delta",
                "output_tokens_per_second",
            ),
        )
    )
    for field, output in (
        (
            "cuda_peak_allocated_bytes",
            "allocated_memory_regression_pct",
        ),
        (
            "cuda_peak_reserved_bytes",
            "reserved_memory_regression_pct",
        ),
    ):
        baseline = max(
            int(row[field])
            for row in performance_rows
            if row["policy"] == "generic"
        )
        candidate = max(
            int(row[field])
            for row in performance_rows
            if row["policy"] == "lease_local_delta"
        )
        metrics[output] = _regression_pct(
            baseline,
            candidate,
        )

    candidate_rows = [
        row
        for row in performance_rows
        if row["policy"] == "lease_local_delta"
    ]
    metrics["candidate_generic_journal_captures"] = sum(
        int(row["generic_journal_captures"])
        for row in candidate_rows
    )
    metrics["candidate_one_phase_fallbacks"] = sum(
        sum(
            int(value)
            for value in row["one_phase_fallbacks"].values()
        )
        for row in candidate_rows
    )
    metrics["candidate_one_phase_rollbacks"] = sum(
        int(row["one_phase_rollbacks"])
        for row in candidate_rows
    )
    metrics["candidate_counter_authority"] = all(
        int(row["one_phase_attempts"])
        == int(row["eligible_bursts"])
        == int(row["one_phase_captures"])
        == int(row["one_phase_commits"])
        for row in candidate_rows
    )
    metrics["classification"] = _classification(metrics)
    return metrics


def _validate_prepare_samples(
    rows: list[dict],
    performance_rows: list[dict],
) -> None:
    if len(rows) != PERFORMANCE_ROW_COUNT:
        raise ValueError(
            "prepare sample inventory is incomplete"
        )
    expected = {
        (
            row["repetition"],
            row["context"],
            row["policy"],
        ): row["prepare_ns"]
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
            raise ValueError("duplicate prepare sample row")
        actual[key] = row.get("prepare_ns")
    if actual != expected:
        raise ValueError("prepare sample inventory mismatch")


def _validate_manifest_hashes(
    run_dir: Path,
    source: dict,
    repo_root: Path,
) -> None:
    artifact_hashes = source.get("artifact_sha256")
    if (
        not isinstance(artifact_hashes, dict)
        or set(artifact_hashes) != HASHED_ARTIFACTS
    ):
        raise ValueError("artifact digest manifest is invalid")
    for name, expected in artifact_hashes.items():
        if _sha256(run_dir / name) != expected:
            raise ValueError("artifact digest mismatch")

    source_hashes = source.get("source_file_sha256")
    if (
        not isinstance(source_hashes, dict)
        or set(source_hashes) != SOURCE_FILES
    ):
        raise ValueError(
            "source file digest manifest is invalid"
        )
    resolved_root = repo_root.resolve()
    for relative, expected in source_hashes.items():
        source_path = repo_root / relative
        resolved = source_path.resolve()
        if (
            resolved == resolved_root
            or resolved_root not in resolved.parents
            or not source_path.is_file()
            or _sha256(source_path) != expected
        ):
            raise ValueError("source file digest mismatch")
    if (
        source.get("source_patch_sha256")
        != EXPECTED_SOURCE_PATCH_SHA256
    ):
        raise ValueError("source patch digest is not empty")


def _validate_workload(workload: dict) -> None:
    if workload.get("schema") != WORKLOAD_SCHEMA:
        raise ValueError("workload schema mismatch")
    if (
        workload.get("policies") != list(POLICIES)
        or workload.get("contexts") != list(CONTEXTS)
        or workload.get("sampling_points")
        != list(SAMPLING_POINTS)
        or workload.get("performance_repetitions")
        != PERFORMANCE_REPETITIONS
        or workload.get("performance_row_count")
        != PERFORMANCE_ROW_COUNT
        or workload.get("correctness_row_count")
        != CORRECTNESS_ROW_COUNT
        or workload.get("execution_shape") != "one_phase_k8"
        or workload.get("split_phase_enabled") is not False
    ):
        raise ValueError("workload inventory mismatch")
    expected_order = {
        str(repetition): {
            context: list(
                POLICIES
                if (repetition + index) % 2 == 0
                else reversed(POLICIES)
            )
            for index, context in enumerate(CONTEXTS)
        }
        for repetition in range(PERFORMANCE_REPETITIONS)
    }
    if workload.get("policy_order") != expected_order:
        raise ValueError("workload policy order mismatch")


def verify_artifact_directory(path: Path) -> dict:
    run_dir = Path(path)
    repo_root = Path(__file__).resolve().parents[1]
    for name in REQUIRED_FILES:
        if not (run_dir / name).is_file():
            raise ValueError(
                f"required artifact is missing: {name}"
            )
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
    prepare_rows = _load_jsonl(
        run_dir / "prepare_samples.jsonl"
    )

    _validate_workload(workload)
    if source.get("schema") != SOURCE_SCHEMA:
        raise ValueError("source manifest schema mismatch")
    if receipt.get("schema") != RUNNER_SCHEMA:
        raise ValueError("runner receipt schema mismatch")
    _validate_manifest_hashes(run_dir, source, repo_root)
    reconstructed = _reconstruct(
        performance_rows,
        correctness_rows,
    )
    _validate_prepare_samples(
        prepare_rows,
        performance_rows,
    )

    source_sha = reconstructed["source_sha"]
    if (
        len(source_sha) != 40
        or any(
            character not in "0123456789abcdef"
            for character in source_sha
        )
        or any(
            value != source_sha
            for value in (
                workload.get("source_sha"),
                source.get("source_sha"),
                receipt.get("source_sha"),
            )
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
    if (
        receipt.get("performance_rows")
        != PERFORMANCE_ROW_COUNT
        or receipt.get("correctness_rows")
        != CORRECTNESS_ROW_COUNT
    ):
        raise ValueError("runner inventory mismatch")
    if stored_summary != reconstructed:
        raise ValueError("summary mismatch")
    if (
        receipt.get("classification")
        != reconstructed["classification"]
    ):
        raise ValueError("runner classification mismatch")
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
