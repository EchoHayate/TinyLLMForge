#!/usr/bin/env python3
"""Independent verifier for generation-sealed lease-identity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


VERIFY_SCHEMA = "exact_burst_generation_sealed_lease_identity_verify_v1"
GATE_SCHEMA = "exact_burst_generation_sealed_lease_identity_gate_v1"
PERFORMANCE_ROW_SCHEMA = (
    "exact_burst_generation_sealed_lease_identity_performance_v1"
)
CORRECTNESS_ROW_SCHEMA = (
    "exact_burst_generation_sealed_lease_identity_correctness_v1"
)
WORKLOAD_SCHEMA = (
    "exact_burst_generation_sealed_lease_identity_workload_v1"
)
SOURCE_SCHEMA = "exact_burst_generation_sealed_lease_identity_source_v1"
RUNNER_SCHEMA = "exact_burst_generation_sealed_lease_identity_runner_v1"
GO = "GO_EXACT_BURST_GENERATION_SEALED_LEASE_IDENTITY"
NO_GO_PERFORMANCE = "NO_GO_PERFORMANCE"
NO_GO_CORRECTNESS = "NO_GO_CORRECTNESS"
NO_GO_TRANSACTIONAL_SAFETY = "NO_GO_TRANSACTIONAL_SAFETY"
NO_GO_EVIDENCE_INCOMPLETE = "NO_GO_EVIDENCE_INCOMPLETE"
EXPECTED_SOURCE_PATCH_SHA256 = hashlib.sha256(b"").hexdigest()
POLICIES = ("full_identity", "generation_sealed")
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
SOURCE_FILES = {
    "tinyvllm/config.py",
    "tinyvllm/engine/block_manager.py",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
    "tinyvllm/engine/llm_engine.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tinyvllm/engine/sequence.py",
    "tools/profile_exact_greedy_decode_burst.py",
    "tools/exact_burst_generation_sealed_lease_identity_gate.py",
    "tools/exact_burst_generation_sealed_lease_identity_verify.py",
}
HASHED_ARTIFACTS = {
    "workload_manifest.json",
    "performance_rows.jsonl",
    "correctness_rows.jsonl",
    "lifecycle_samples.jsonl",
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


def _finite(value, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _samples(row: dict, field: str) -> list[float]:
    values = row.get(field)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field} must be finite and non-empty")
    return [_finite(value, field) for value in values]


def _median(values) -> float:
    normalized = [_finite(value, "metric sample") for value in values]
    if not normalized:
        raise ValueError("metric samples must be finite and non-empty")
    return float(statistics.median(normalized))


def _nearest_rank(values, percentile: float) -> float:
    ordered = sorted(_finite(value, "metric sample") for value in values)
    if not ordered:
        raise ValueError("metric samples must be finite and non-empty")
    return ordered[max(1, math.ceil(percentile * len(ordered))) - 1]


def _regression(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        if baseline == candidate:
            return 0.0
        raise ValueError("metric baseline must be positive")
    return (candidate - baseline) / baseline * 100.0


def _improvement(baseline: float, candidate: float) -> float:
    return -_regression(baseline, candidate)


def _throughput_regression(
    baseline: float,
    candidate: float,
) -> float:
    if baseline <= 0:
        if baseline == candidate:
            return 0.0
        raise ValueError("throughput baseline must be positive")
    return (baseline - candidate) / baseline * 100.0


def _policy_order(repetition: int, context_index: int) -> tuple[str, ...]:
    if (repetition + context_index) % 2:
        return tuple(reversed(POLICIES))
    return POLICIES


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
        policy = row.get("policy")
        context = row.get("context")
        repetition = row.get("repetition")
        if policy not in POLICIES:
            raise ValueError("performance row policy is invalid")
        if context not in CONTEXTS:
            raise ValueError("performance row context is invalid")
        if (
            not isinstance(repetition, int)
            or isinstance(repetition, bool)
            or repetition not in range(PERFORMANCE_REPETITIONS)
        ):
            raise ValueError("performance row repetition is invalid")
        if row.get("order_position") != _policy_order(
            repetition,
            CONTEXTS.index(context),
        ).index(policy):
            raise ValueError("performance row policy order is invalid")
        for field in (
            "ttft_ns",
            "e2e_ns",
            "output_tokens_per_second",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
        ):
            _finite(row.get(field), field)
        for field in (
            "lease_grant_ns",
            "scheduler_lifecycle_ns",
            "tpot_samples_ns",
        ):
            _samples(row, field)
        fallbacks = row.get("identity_seal_fallbacks")
        if not isinstance(fallbacks, dict):
            raise ValueError("identity seal fallback counts are invalid")
        for reason, count in fallbacks.items():
            if (
                not isinstance(reason, str)
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
            ):
                raise ValueError("identity seal fallback counts are invalid")
        key = (repetition, context, policy)
        if key in indexed:
            raise ValueError("duplicate performance row")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("performance row inventory is incomplete")
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
        policy = row.get("policy")
        if policy not in POLICIES:
            raise ValueError("correctness row policy is invalid")
        logits = row.get("sampled_logits")
        if not isinstance(logits, list) or not logits:
            raise ValueError("sampled logits must be finite and non-empty")
        for value in logits:
            _finite(value, "sampled logits")
        key = (
            row.get("context"),
            policy,
            row.get("sampling_point"),
        )
        if key in indexed:
            raise ValueError("duplicate correctness row")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("correctness row inventory is incomplete")
    return indexed


def _same_execution(left: dict, right: dict) -> bool:
    return all(
        left.get(field) == right.get(field)
        for field in (
            "target_model_forwards",
            "graph_replays",
            "d2h_calls",
            "d2h_bytes",
        )
    )


def _row_stat(
    rows: list[dict],
    policy: str,
    field: str,
    percentile: float | None = None,
) -> float:
    values = []
    for row in rows:
        if row["policy"] != policy:
            continue
        if percentile is None:
            values.append(_finite(row[field], field))
        else:
            values.append(_nearest_rank(row[field], percentile))
    return _median(values)


def _classification(metrics: dict) -> str:
    if (
        metrics["performance_row_count"] != PERFORMANCE_ROW_COUNT
        or metrics["correctness_row_count"] != CORRECTNESS_ROW_COUNT
    ):
        return NO_GO_EVIDENCE_INCOMPLETE
    if (
        metrics["output_exact"] is not True
        or metrics["sampled_argmax_exact"] is not True
        or metrics["sampled_logit_max_abs_diff"] != 0.0
        or metrics["execution_inventory_equal"] is not True
        or metrics["paired_workload_equal"] is not True
    ):
        return NO_GO_CORRECTNESS
    if (
        metrics["baseline_identity_counters_zero"] is not True
        or metrics["candidate_counter_authority"] is not True
        or metrics["candidate_hot_reuse_accounting"] is not True
        or metrics["candidate_identity_seal_fallbacks"] != 0
        or metrics["candidate_exact_burst_failures"] != 0
        or metrics["candidate_one_phase_rollbacks"] != 0
    ):
        return NO_GO_TRANSACTIONAL_SAFETY
    if (
        metrics["8k_lifecycle_median_improvement_pct"] < 25.0
        or metrics["8k_lifecycle_p95_improvement_pct"] < 25.0
        or metrics["aggregate_lifecycle_median_improvement_pct"] < 15.0
        or metrics["aggregate_lifecycle_p95_improvement_pct"] < 15.0
        or metrics["aggregate_tpot_median_improvement_pct"] < 0.5
        or metrics["aggregate_tpot_p95_improvement_pct"] < 0.5
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
    max_logit_diff = 0.0
    execution_equal = True
    paired_workload_equal = True
    for repetition in range(PERFORMANCE_REPETITIONS):
        for context in CONTEXTS:
            baseline = performance[
                (repetition, context, "full_identity")
            ]
            candidate = performance[
                (repetition, context, "generation_sealed")
            ]
            output_exact = output_exact and (
                baseline.get("output_tokens")
                == candidate.get("output_tokens")
            )
            execution_equal = execution_equal and _same_execution(
                baseline,
                candidate,
            )
            paired_workload_equal = paired_workload_equal and all(
                baseline.get(field) == candidate.get(field)
                for field in (
                    "prompt_digest",
                    "generated_tokens",
                )
            )
    for context in CONTEXTS:
        for point in SAMPLING_POINTS:
            baseline = correctness[(context, "full_identity", point)]
            candidate = correctness[
                (context, "generation_sealed", point)
            ]
            output_exact = output_exact and (
                baseline.get("output_token_ids")
                == candidate.get("output_token_ids")
            )
            sampled_argmax_exact = sampled_argmax_exact and (
                baseline.get("sampled_argmax")
                == candidate.get("sampled_argmax")
            )
            left = baseline["sampled_logits"]
            right = candidate["sampled_logits"]
            if len(left) != len(right):
                max_logit_diff = math.inf
            else:
                max_logit_diff = max(
                    max_logit_diff,
                    max(
                        (
                            abs(float(a) - float(b))
                            for a, b in zip(left, right)
                        ),
                        default=0.0,
                    ),
                )
            execution_equal = execution_equal and _same_execution(
                baseline,
                candidate,
            )

    metrics = {
        "schema": GATE_SCHEMA,
        "run_tag": next(iter(run_tags)),
        "source_sha": next(iter(source_shas)),
        "performance_row_count": len(performance_rows),
        "correctness_row_count": len(correctness_rows),
        "output_exact": output_exact,
        "sampled_argmax_exact": sampled_argmax_exact,
        "sampled_logit_max_abs_diff": max_logit_diff,
        "execution_inventory_equal": execution_equal,
        "paired_workload_equal": paired_workload_equal,
    }
    for context in CONTEXTS:
        baseline = [
            sample
            for row in performance_rows
            if row["policy"] == "full_identity"
            and row["context"] == context
            for sample in row["scheduler_lifecycle_ns"]
        ]
        candidate = [
            sample
            for row in performance_rows
            if row["policy"] == "generation_sealed"
            and row["context"] == context
            for sample in row["scheduler_lifecycle_ns"]
        ]
        metrics[f"{context}_lifecycle_median_improvement_pct"] = (
            _improvement(_median(baseline), _median(candidate))
        )
        metrics[f"{context}_lifecycle_p95_improvement_pct"] = (
            _improvement(
                _nearest_rank(baseline, 0.95),
                _nearest_rank(candidate, 0.95),
            )
        )
    baseline_lifecycle = [
        sample
        for row in performance_rows
        if row["policy"] == "full_identity"
        for sample in row["scheduler_lifecycle_ns"]
    ]
    candidate_lifecycle = [
        sample
        for row in performance_rows
        if row["policy"] == "generation_sealed"
        for sample in row["scheduler_lifecycle_ns"]
    ]
    metrics["aggregate_lifecycle_median_improvement_pct"] = (
        _improvement(
            _median(baseline_lifecycle),
            _median(candidate_lifecycle),
        )
    )
    metrics["aggregate_lifecycle_p95_improvement_pct"] = (
        _improvement(
            _nearest_rank(baseline_lifecycle, 0.95),
            _nearest_rank(candidate_lifecycle, 0.95),
        )
    )
    baseline_tpot = _median(
        _median(row["tpot_samples_ns"])
        for row in performance_rows
        if row["policy"] == "full_identity"
    )
    candidate_tpot = _median(
        _median(row["tpot_samples_ns"])
        for row in performance_rows
        if row["policy"] == "generation_sealed"
    )
    metrics["aggregate_tpot_median_improvement_pct"] = _improvement(
        baseline_tpot,
        candidate_tpot,
    )
    for percentile, label in ((0.95, "p95"), (0.99, "p99")):
        baseline = _row_stat(
            performance_rows,
            "full_identity",
            "tpot_samples_ns",
            percentile,
        )
        candidate = _row_stat(
            performance_rows,
            "generation_sealed",
            "tpot_samples_ns",
            percentile,
        )
        metrics[f"aggregate_tpot_{label}_improvement_pct"] = (
            _improvement(baseline, candidate)
        )
        metrics[f"aggregate_tpot_{label}_regression_pct"] = _regression(
            baseline,
            candidate,
        )
    for field, output in (
        ("ttft_ns", "aggregate_ttft_regression_pct"),
        ("e2e_ns", "aggregate_e2e_regression_pct"),
    ):
        metrics[output] = _regression(
            _row_stat(performance_rows, "full_identity", field),
            _row_stat(performance_rows, "generation_sealed", field),
        )
    metrics["throughput_regression_pct"] = _throughput_regression(
        _row_stat(
            performance_rows,
            "full_identity",
            "output_tokens_per_second",
        ),
        _row_stat(
            performance_rows,
            "generation_sealed",
            "output_tokens_per_second",
        ),
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
            if row["policy"] == "full_identity"
        )
        candidate = max(
            int(row[field])
            for row in performance_rows
            if row["policy"] == "generation_sealed"
        )
        metrics[output] = _regression(baseline, candidate)
    candidate_rows = [
        row
        for row in performance_rows
        if row["policy"] == "generation_sealed"
    ]
    baseline_rows = [
        row
        for row in performance_rows
        if row["policy"] == "full_identity"
    ]
    metrics["baseline_identity_counters_zero"] = all(
        int(row["identity_seal_cold_captures"]) == 0
        and int(row["identity_seal_hot_reuses"]) == 0
        and int(row["identity_seal_validations"]) == 0
        and not row["identity_seal_fallbacks"]
        for row in baseline_rows
    )
    metrics["candidate_identity_seal_fallbacks"] = sum(
        sum(int(value) for value in row["identity_seal_fallbacks"].values())
        for row in candidate_rows
    )
    metrics["candidate_exact_burst_failures"] = sum(
        int(row["exact_burst_failures"]) for row in candidate_rows
    )
    metrics["candidate_one_phase_rollbacks"] = sum(
        int(row["one_phase_rollbacks"]) for row in candidate_rows
    )
    metrics["candidate_counter_authority"] = all(
        int(row["eligible_bursts"]) > 0
        and int(row["eligible_bursts"])
        == len(row["lease_grant_ns"])
        == len(row["scheduler_lifecycle_ns"])
        for row in candidate_rows
    )
    metrics["candidate_hot_reuse_accounting"] = all(
        int(row["identity_seal_hot_reuses"])
        == int(row["eligible_bursts"])
        - int(row["identity_seal_cold_captures"])
        and int(row["identity_seal_validations"])
        >= int(row["eligible_bursts"])
        for row in candidate_rows
    )
    metrics["classification"] = _classification(metrics)
    return metrics


def _validate_lifecycle_samples(
    rows: list[dict],
    performance_rows: list[dict],
) -> None:
    if len(rows) != PERFORMANCE_ROW_COUNT:
        raise ValueError("lifecycle sample inventory is incomplete")
    expected = {
        (
            row["repetition"],
            row["context"],
            row["policy"],
        ): {
            "lease_grant_ns": row["lease_grant_ns"],
            "scheduler_lifecycle_ns": row["scheduler_lifecycle_ns"],
        }
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
            raise ValueError("duplicate lifecycle sample row")
        actual[key] = {
            "lease_grant_ns": row.get("lease_grant_ns"),
            "scheduler_lifecycle_ns": row.get(
                "scheduler_lifecycle_ns"
            ),
        }
    if actual != expected:
        raise ValueError("lifecycle sample inventory mismatch")


def _validate_hashes(
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
        raise ValueError("source file digest manifest is invalid")
    resolved_root = repo_root.resolve()
    for relative, expected in source_hashes.items():
        path = repo_root / relative
        resolved = path.resolve()
        if (
            resolved == resolved_root
            or resolved_root not in resolved.parents
            or not path.is_file()
            or _sha256(path) != expected
        ):
            raise ValueError("source file digest mismatch")
    if source.get("source_patch_sha256") != EXPECTED_SOURCE_PATCH_SHA256:
        raise ValueError("source patch digest is not empty")


def _validate_workload(workload: dict) -> None:
    if workload.get("schema") != WORKLOAD_SCHEMA:
        raise ValueError("workload schema mismatch")
    if (
        workload.get("policies") != list(POLICIES)
        or workload.get("contexts") != list(CONTEXTS)
        or workload.get("sampling_points") != list(SAMPLING_POINTS)
        or workload.get("performance_repetitions")
        != PERFORMANCE_REPETITIONS
        or workload.get("performance_row_count")
        != PERFORMANCE_ROW_COUNT
        or workload.get("correctness_row_count")
        != CORRECTNESS_ROW_COUNT
        or workload.get("execution_shape") != "one_phase_k8"
        or workload.get("split_phase_enabled") is not False
        or workload.get("lease_local_delta_journal_enabled") is not True
        or workload.get("only_variable")
        != "exact_greedy_decode_burst_generation_sealed_identity"
    ):
        raise ValueError("workload inventory mismatch")
    expected_order = {
        str(repetition): {
            context: list(_policy_order(repetition, index))
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
    lifecycle_rows = _load_jsonl(
        run_dir / "lifecycle_samples.jsonl"
    )

    _validate_workload(workload)
    if source.get("schema") != SOURCE_SCHEMA:
        raise ValueError("source manifest schema mismatch")
    if receipt.get("schema") != RUNNER_SCHEMA:
        raise ValueError("runner receipt schema mismatch")
    _validate_hashes(run_dir, source, repo_root)
    reconstructed = _reconstruct(performance_rows, correctness_rows)
    _validate_lifecycle_samples(lifecycle_rows, performance_rows)

    source_sha = reconstructed["source_sha"]
    if (
        not isinstance(source_sha, str)
        or len(source_sha) != 40
        or any(character not in "0123456789abcdef" for character in source_sha)
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
        receipt.get("performance_rows") != PERFORMANCE_ROW_COUNT
        or receipt.get("correctness_rows") != CORRECTNESS_ROW_COUNT
    ):
        raise ValueError("runner receipt row inventory mismatch")
    if stored_summary != reconstructed:
        raise ValueError("summary mismatch")
    if receipt.get("classification") != reconstructed["classification"]:
        raise ValueError("runner receipt classification mismatch")
    return {
        "schema": VERIFY_SCHEMA,
        "verified": True,
        "classification": reconstructed["classification"],
        "performance_row_count": len(performance_rows),
        "correctness_row_count": len(correctness_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_artifact_directory(args.run_dir),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
