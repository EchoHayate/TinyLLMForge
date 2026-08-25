#!/usr/bin/env python3
"""Paired evidence gate for generation-sealed exact-burst lease identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from types import MethodType


GATE_SCHEMA = "exact_burst_generation_sealed_lease_identity_gate_v1"
PERFORMANCE_ROW_SCHEMA = (
    "exact_burst_generation_sealed_lease_identity_performance_v1"
)
CORRECTNESS_ROW_SCHEMA = (
    "exact_burst_generation_sealed_lease_identity_correctness_v1"
)
WORKLOAD_MANIFEST_SCHEMA = (
    "exact_burst_generation_sealed_lease_identity_workload_v1"
)
SOURCE_MANIFEST_SCHEMA = (
    "exact_burst_generation_sealed_lease_identity_source_v1"
)
RUNNER_RECEIPT_SCHEMA = (
    "exact_burst_generation_sealed_lease_identity_runner_v1"
)
POLICIES = ("full_identity", "generation_sealed")
CONTEXTS = ("2k", "4k", "8k")
CONTEXT_CASES = (
    ("2k", 2048, 128),
    ("4k", 4096, 128),
    ("8k", 8192, 128),
)
SAMPLING_POINTS = (
    "prefill-final",
    "decode-first",
    "decode-middle",
    "decode-final",
)
PERFORMANCE_REPETITIONS = 10
PERFORMANCE_ROW_COUNT = 60
CORRECTNESS_ROW_COUNT = 24
DEFAULT_MODEL = "/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B"
GO = "GO_EXACT_BURST_GENERATION_SEALED_LEASE_IDENTITY"
NO_GO_PERFORMANCE = "NO_GO_PERFORMANCE"
NO_GO_CORRECTNESS = "NO_GO_CORRECTNESS"
NO_GO_TRANSACTIONAL_SAFETY = "NO_GO_TRANSACTIONAL_SAFETY"
NO_GO_EVIDENCE_INCOMPLETE = "NO_GO_EVIDENCE_INCOMPLETE"
SOURCE_FILES = (
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
)

_ACTIVE_POLICY = "full_identity"
_LAST_LEASE_GRANT_NS: tuple[int, ...] = ()
_LAST_COMMIT_NS: tuple[int, ...] = ()
_LAST_COUNTERS: dict = {}


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def read_json(path: Path):
    return json.loads(
        Path(path).read_text(),
        parse_constant=_reject_constant,
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line, parse_constant=_reject_constant)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )


def write_jsonl(path: Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        )
    )


def policy_order(
    repetition: int,
    context_index: int,
) -> tuple[str, ...]:
    if (repetition + context_index) % 2:
        return tuple(reversed(POLICIES))
    return POLICIES


def build_workload_manifest(
    *,
    run_tag: str | None = None,
    source_sha: str | None = None,
) -> dict:
    return {
        "schema": WORKLOAD_MANIFEST_SCHEMA,
        "run_tag": run_tag,
        "source_sha": source_sha,
        "policies": list(POLICIES),
        "contexts": list(CONTEXTS),
        "performance_repetitions": PERFORMANCE_REPETITIONS,
        "sampling_points": list(SAMPLING_POINTS),
        "performance_row_count": PERFORMANCE_ROW_COUNT,
        "correctness_row_count": CORRECTNESS_ROW_COUNT,
        "execution_shape": "one_phase_k8",
        "split_phase_enabled": False,
        "lease_local_delta_journal_enabled": True,
        "only_variable": (
            "exact_greedy_decode_burst_generation_sealed_identity"
        ),
        "policy_order": {
            str(repetition): {
                context: list(policy_order(repetition, context_index))
                for context_index, context in enumerate(CONTEXTS)
            }
            for repetition in range(PERFORMANCE_REPETITIONS)
        },
    }


def policy_runtime_config(policy: str) -> dict[str, object]:
    if policy not in POLICIES:
        raise ValueError("policy is invalid")
    return {
        "exact_greedy_decode_burst": True,
        "exact_greedy_decode_burst_tokens": 8,
        "exact_greedy_decode_burst_split_phase": False,
        "exact_greedy_decode_burst_ragged_coalescing": False,
        "exact_greedy_decode_burst_continuation": False,
        "exact_greedy_decode_burst_lease_local_delta_journal": True,
        "exact_greedy_decode_burst_generation_sealed_identity": (
            policy == "generation_sealed"
        ),
    }


def combine_scheduler_lifecycle_samples(
    lease_grant_ns,
    commit_ns,
) -> list[int]:
    if len(lease_grant_ns) != len(commit_ns):
        raise ValueError("scheduler lifecycle sample inventory mismatch")
    combined = []
    for grant, commit in zip(lease_grant_ns, commit_ns):
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in (grant, commit)
        ):
            raise ValueError(
                "scheduler lifecycle samples must be non-negative integers"
            )
        combined.append(grant + commit)
    if not combined:
        raise ValueError("scheduler lifecycle samples are empty")
    return combined


def _finite_float(value, *, field: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be finite")
    return normalized


def _finite_samples(row: dict, field: str) -> list[float]:
    values = row.get(field)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field} must be finite and non-empty")
    return [_finite_float(value, field=field) for value in values]


def _median(values) -> float:
    normalized = [
        _finite_float(value, field="metric sample") for value in values
    ]
    if not normalized:
        raise ValueError("metric samples must be finite and non-empty")
    return float(statistics.median(normalized))


def _nearest_rank(values, percentile: float) -> float:
    ordered = sorted(
        _finite_float(value, field="metric sample") for value in values
    )
    if not ordered:
        raise ValueError("metric samples must be finite and non-empty")
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _regression_pct(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        if candidate == baseline:
            return 0.0
        raise ValueError("metric baseline must be positive")
    return (candidate - baseline) / baseline * 100.0


def _improvement_pct(baseline: float, candidate: float) -> float:
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


def _validate_performance_rows(
    rows: list[dict],
) -> dict[tuple[int, str, str], dict]:
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
        if policy not in POLICIES:
            raise ValueError("performance row policy is invalid")
        context = row.get("context")
        repetition = row.get("repetition")
        order_position = row.get("order_position")
        if context not in CONTEXTS:
            raise ValueError("performance row context is invalid")
        if (
            not isinstance(repetition, int)
            or isinstance(repetition, bool)
            or repetition not in range(PERFORMANCE_REPETITIONS)
        ):
            raise ValueError("performance row repetition is invalid")
        expected_order = policy_order(
            repetition,
            CONTEXTS.index(context),
        )
        if order_position != expected_order.index(policy):
            raise ValueError("performance row policy order is invalid")
        for field in (
            "ttft_ns",
            "e2e_ns",
            "output_tokens_per_second",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
        ):
            _finite_float(row.get(field), field=field)
        for field in (
            "lease_grant_ns",
            "scheduler_lifecycle_ns",
            "tpot_samples_ns",
        ):
            _finite_samples(row, field)
        fallbacks = row.get("identity_seal_fallbacks")
        if not isinstance(fallbacks, dict):
            raise ValueError("identity seal fallback counts are invalid")
        for reason, count in fallbacks.items():
            if not isinstance(reason, str) or not isinstance(count, int):
                raise ValueError("identity seal fallback counts are invalid")
            if isinstance(count, bool) or count < 0:
                raise ValueError("identity seal fallback counts are invalid")
        key = (repetition, context, policy)
        if key in indexed:
            raise ValueError("duplicate performance row")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("performance row inventory is incomplete")
    return indexed


def _validate_correctness_rows(
    rows: list[dict],
) -> dict[tuple[str, str, str], dict]:
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
            _finite_float(value, field="sampled logits")
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


def _paired_inventory_equal(left: dict, right: dict) -> bool:
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
            values.append(_finite_float(row[field], field=field))
        else:
            values.append(_nearest_rank(row[field], percentile))
    return _median(values)


def classify(metrics: dict) -> str:
    if (
        metrics.get("performance_row_count") != PERFORMANCE_ROW_COUNT
        or metrics.get("correctness_row_count")
        != CORRECTNESS_ROW_COUNT
    ):
        return NO_GO_EVIDENCE_INCOMPLETE
    if (
        metrics.get("output_exact") is not True
        or metrics.get("sampled_argmax_exact") is not True
        or float(metrics.get("sampled_logit_max_abs_diff", math.inf))
        != 0.0
        or metrics.get("execution_inventory_equal") is not True
        or metrics.get("paired_workload_equal") is not True
    ):
        return NO_GO_CORRECTNESS
    if (
        metrics.get("baseline_identity_counters_zero") is not True
        or metrics.get("candidate_counter_authority") is not True
        or metrics.get("candidate_hot_reuse_accounting") is not True
        or int(metrics.get("candidate_identity_seal_fallbacks", -1))
        != 0
        or int(metrics.get("candidate_exact_burst_failures", -1)) != 0
        or int(metrics.get("candidate_one_phase_rollbacks", -1)) != 0
    ):
        return NO_GO_TRANSACTIONAL_SAFETY
    if (
        float(
            metrics.get(
                "8k_lifecycle_median_improvement_pct",
                -math.inf,
            )
        )
        < 25.0
        or float(
            metrics.get(
                "8k_lifecycle_p95_improvement_pct",
                -math.inf,
            )
        )
        < 25.0
        or float(
            metrics.get(
                "aggregate_lifecycle_median_improvement_pct",
                -math.inf,
            )
        )
        < 15.0
        or float(
            metrics.get(
                "aggregate_lifecycle_p95_improvement_pct",
                -math.inf,
            )
        )
        < 15.0
        or float(
            metrics.get(
                "aggregate_tpot_median_improvement_pct",
                -math.inf,
            )
        )
        < 0.5
        or float(
            metrics.get(
                "aggregate_tpot_p95_improvement_pct",
                -math.inf,
            )
        )
        < 0.5
        or any(
            float(metrics.get(field, math.inf)) > 2.0
            for field in (
                "aggregate_tpot_p99_regression_pct",
                "aggregate_ttft_regression_pct",
                "aggregate_e2e_regression_pct",
                "throughput_regression_pct",
            )
        )
        or any(
            float(metrics.get(field, math.inf)) > 1.0
            for field in (
                "allocated_memory_regression_pct",
                "reserved_memory_regression_pct",
            )
        )
    ):
        return NO_GO_PERFORMANCE
    return GO


def summarize_evidence(
    performance_rows: list[dict],
    correctness_rows: list[dict],
) -> dict:
    performance = _validate_performance_rows(performance_rows)
    correctness = _validate_correctness_rows(correctness_rows)
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
    execution_inventory_equal = True
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
            execution_inventory_equal = (
                execution_inventory_equal
                and _paired_inventory_equal(baseline, candidate)
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
                sampled_logit_max_abs_diff = math.inf
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
            execution_inventory_equal = (
                execution_inventory_equal
                and _paired_inventory_equal(baseline, candidate)
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
        "execution_inventory_equal": execution_inventory_equal,
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
            _improvement_pct(_median(baseline), _median(candidate))
        )
        metrics[f"{context}_lifecycle_p95_improvement_pct"] = (
            _improvement_pct(
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
        _improvement_pct(
            _median(baseline_lifecycle),
            _median(candidate_lifecycle),
        )
    )
    metrics["aggregate_lifecycle_p95_improvement_pct"] = (
        _improvement_pct(
            _nearest_rank(baseline_lifecycle, 0.95),
            _nearest_rank(candidate_lifecycle, 0.95),
        )
    )
    baseline_tpot_median = _median(
        _median(row["tpot_samples_ns"])
        for row in performance_rows
        if row["policy"] == "full_identity"
    )
    candidate_tpot_median = _median(
        _median(row["tpot_samples_ns"])
        for row in performance_rows
        if row["policy"] == "generation_sealed"
    )
    metrics["aggregate_tpot_median_improvement_pct"] = (
        _improvement_pct(baseline_tpot_median, candidate_tpot_median)
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
            _improvement_pct(baseline, candidate)
        )
        metrics[f"aggregate_tpot_{label}_regression_pct"] = (
            _regression_pct(baseline, candidate)
        )
    for field, output in (
        ("ttft_ns", "aggregate_ttft_regression_pct"),
        ("e2e_ns", "aggregate_e2e_regression_pct"),
    ):
        metrics[output] = _regression_pct(
            _row_stat(performance_rows, "full_identity", field),
            _row_stat(performance_rows, "generation_sealed", field),
        )
    metrics["throughput_regression_pct"] = _throughput_regression_pct(
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
        metrics[output] = _regression_pct(baseline, candidate)

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
    metrics["classification"] = classify(metrics)
    return metrics


def produce_summary(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    summary = summarize_evidence(
        read_jsonl(run_dir / "performance_rows.jsonl"),
        read_jsonl(run_dir / "correctness_rows.jsonl"),
    )
    write_json(run_dir / "summary.json", summary)
    return summary


def _counter_difference(before: dict, after: dict, field: str) -> int:
    difference = int(after.get(field, 0)) - int(before.get(field, 0))
    if difference < 0:
        raise RuntimeError(f"counter decreased: {field}")
    return difference


def _fallback_difference(before: dict, after: dict) -> dict[str, int]:
    field = "identity_seal_fallback_counts"
    before_map = before.get(field, {})
    after_map = after.get(field, {})
    result = {}
    for reason in sorted(set(before_map) | set(after_map)):
        difference = int(after_map.get(reason, 0)) - int(
            before_map.get(reason, 0)
        )
        if difference < 0:
            raise RuntimeError("identity seal fallback counter decreased")
        if difference:
            result[reason] = difference
    return result


def _activate_gpu_harness():
    from tools import profile_exact_greedy_decode_burst as base

    base.CONTEXT_CASES = CONTEXT_CASES
    original_run_request = base._run_request
    original_combined_summary = base._combined_summary

    def construct_llm(
        *,
        model,
        prompt_tokens,
        generated_tokens,
        gpu_memory_utilization,
        policy,
    ):
        del policy
        from tinyvllm import LLM

        config = policy_runtime_config(_ACTIVE_POLICY)
        llm = LLM(
            model,
            max_num_batched_tokens=prompt_tokens + generated_tokens,
            max_num_seqs=1,
            max_model_len=prompt_tokens + generated_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            enforce_eager=False,
            zero_temperature_greedy_fast_path=True,
            graph_resident_greedy_tail=False,
            **config,
        )
        scheduler = llm.scheduler
        scheduler._generation_sealed_lease_grant_ns = []
        scheduler._generation_sealed_commit_ns = []
        original_grant = scheduler.prepare_exact_greedy_decode_burst
        original_commit = (
            scheduler.prepare_exact_greedy_decode_burst_commit
        )

        def timed_grant(_owner, *args, **kwargs):
            started_ns = time.perf_counter_ns()
            lease = original_grant(*args, **kwargs)
            elapsed_ns = time.perf_counter_ns() - started_ns
            if lease is not None:
                scheduler._generation_sealed_lease_grant_ns.append(
                    elapsed_ns
                )
            return lease

        def timed_commit(_owner, *args, **kwargs):
            started_ns = time.perf_counter_ns()
            prepared = original_commit(*args, **kwargs)
            scheduler._generation_sealed_commit_ns.append(
                time.perf_counter_ns() - started_ns
            )
            return prepared

        scheduler.prepare_exact_greedy_decode_burst = MethodType(
            timed_grant,
            scheduler,
        )
        scheduler.prepare_exact_greedy_decode_burst_commit = MethodType(
            timed_commit,
            scheduler,
        )
        return llm

    def run_request(llm, **kwargs):
        global _LAST_LEASE_GRANT_NS, _LAST_COMMIT_NS
        measured = kwargs.get("profile_label") is not None
        if measured:
            llm.scheduler._generation_sealed_lease_grant_ns.clear()
            llm.scheduler._generation_sealed_commit_ns.clear()
        result = original_run_request(llm, **kwargs)
        if measured:
            _LAST_LEASE_GRANT_NS = tuple(
                llm.scheduler._generation_sealed_lease_grant_ns
            )
            _LAST_COMMIT_NS = tuple(
                llm.scheduler._generation_sealed_commit_ns
            )
        return result

    def combined_summary(
        llm,
        before,
        *,
        correctness_trace=False,
    ):
        global _LAST_COUNTERS
        after = llm.scheduler.exact_greedy_decode_burst_summary()
        fields = (
            "identity_seal_cold_captures",
            "identity_seal_hot_reuses",
            "identity_seal_validations",
            "failures",
            "lease_local_delta_journal_one_phase_attempts",
            "lease_local_delta_journal_one_phase_rollbacks",
        )
        _LAST_COUNTERS = {
            field: _counter_difference(before[1], after, field)
            for field in fields
        }
        _LAST_COUNTERS["identity_seal_fallbacks"] = (
            _fallback_difference(before[1], after)
        )
        return original_combined_summary(
            llm,
            before,
            correctness_trace=correctness_trace,
        )

    base._construct_llm = construct_llm
    base._run_request = run_request
    base._combined_summary = combined_summary
    return base


def _d2h_inventory(summary: dict) -> tuple[int, int]:
    calls = (
        int(summary.get("intermediate_token_d2h_calls", 0))
        + int(summary.get("final_token_d2h_calls", 0))
        + int(summary.get("sampled_logit_d2h_calls", 0))
    )
    return calls, int(summary.get("final_token_d2h_bytes", 0))


def _performance_row(
    legacy: dict,
    *,
    policy: str,
    source_sha: str,
    order_position: int,
    prompt_digest: str,
) -> dict:
    summary = legacy["exact_greedy_decode_burst_summary"]
    d2h_calls, d2h_bytes = _d2h_inventory(summary)
    lease_grant = list(_LAST_LEASE_GRANT_NS)
    lifecycle = combine_scheduler_lifecycle_samples(
        lease_grant,
        list(_LAST_COMMIT_NS),
    )
    counters = dict(_LAST_COUNTERS)
    eligible = int(
        counters.get(
            "lease_local_delta_journal_one_phase_attempts",
            0,
        )
    )
    if eligible != len(lifecycle):
        raise RuntimeError(
            "eligible burst and lifecycle sample inventories differ"
        )
    return {
        "schema": PERFORMANCE_ROW_SCHEMA,
        "run_tag": legacy["run_tag"],
        "source_sha": source_sha,
        "policy": policy,
        "context": legacy["context_bucket"],
        "repetition": legacy["repetition"],
        "order_position": order_position,
        "prompt_digest": prompt_digest,
        "generated_tokens": legacy["generated_tokens"],
        "output_tokens": legacy["output_token_ids"],
        "lease_grant_ns": lease_grant,
        "scheduler_lifecycle_ns": lifecycle,
        "ttft_ns": legacy["ttft_ns"],
        "tpot_samples_ns": legacy["amortized_tpot_samples_ns"],
        "e2e_ns": legacy["e2e_ns"],
        "output_tokens_per_second": legacy[
            "output_tokens_per_second"
        ],
        "cuda_peak_allocated_bytes": legacy[
            "cuda_peak_allocated_bytes"
        ],
        "cuda_peak_reserved_bytes": legacy[
            "cuda_peak_reserved_bytes"
        ],
        "target_model_forwards": summary["target_model_forwards"],
        "graph_replays": summary["graph_replays"],
        "d2h_calls": d2h_calls,
        "d2h_bytes": d2h_bytes,
        "eligible_bursts": eligible,
        "identity_seal_cold_captures": int(
            counters.get("identity_seal_cold_captures", 0)
        ),
        "identity_seal_hot_reuses": int(
            counters.get("identity_seal_hot_reuses", 0)
        ),
        "identity_seal_validations": int(
            counters.get("identity_seal_validations", 0)
        ),
        "identity_seal_fallbacks": dict(
            counters.get("identity_seal_fallbacks", {})
        ),
        "exact_burst_failures": int(counters.get("failures", 0)),
        "one_phase_rollbacks": int(
            counters.get(
                "lease_local_delta_journal_one_phase_rollbacks",
                0,
            )
        ),
    }


def _correctness_rows_from_legacy(
    legacy_rows: list[dict],
    *,
    base,
    run_dir: Path,
    policy: str,
    source_sha: str,
) -> list[dict]:
    rows = []
    for legacy in legacy_rows:
        summary = legacy["exact_greedy_decode_burst_summary"]
        d2h_calls, d2h_bytes = _d2h_inventory(summary)
        logits = base.read_float32_sidecar(
            run_dir,
            path=legacy["logits_path"],
            expected_element_count=legacy["logits_element_count"],
            expected_byte_length=legacy["logits_byte_length"],
            expected_sha256=legacy["logits_sha256"],
        )
        rows.append({
            "schema": CORRECTNESS_ROW_SCHEMA,
            "run_tag": legacy["run_tag"],
            "source_sha": source_sha,
            "policy": policy,
            "context": legacy["context_bucket"],
            "sampling_point": legacy["sampling_point"],
            "output_token_ids": legacy["output_token_ids"],
            "sampled_logits": list(logits),
            "sampled_argmax": max(
                range(len(logits)),
                key=logits.__getitem__,
            ),
            "target_model_forwards": summary["target_model_forwards"],
            "graph_replays": summary["graph_replays"],
            "d2h_calls": d2h_calls,
            "d2h_bytes": d2h_bytes,
        })
    return rows


def run_hardware_gate(
    *,
    model: str,
    output_dir: Path,
    source_sha: str,
    run_tag: str,
    gpu_memory_utilization: float = 0.5,
) -> dict:
    if (
        len(source_sha) != 40
        or any(character not in "0123456789abcdef" for character in source_sha)
    ):
        raise ValueError("source SHA must be a full lowercase SHA-1")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    base = _activate_gpu_harness()
    performance_rows = []
    lifecycle_rows = []
    for repetition in range(PERFORMANCE_REPETITIONS):
        for context_index, (
            context,
            prompt_tokens,
            generated_tokens,
        ) in enumerate(CONTEXT_CASES):
            prompt = base._make_prompt(
                prompt_tokens,
                offset=repetition * 10_007,
            )
            prompt_digest = hashlib.sha256(
                json.dumps(prompt).encode("utf-8")
            ).hexdigest()
            for order_position, policy in enumerate(
                policy_order(repetition, context_index)
            ):
                global _ACTIVE_POLICY
                _ACTIVE_POLICY = policy
                legacy = base.run_case(
                    model=model,
                    run_tag=run_tag,
                    source_commit=source_sha,
                    policy="decode_burst_k8",
                    repetition=repetition,
                    context_bucket=context,
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated_tokens,
                    warmup_repetitions=2,
                    gpu_memory_utilization=gpu_memory_utilization,
                )
                row = _performance_row(
                    legacy,
                    policy=policy,
                    source_sha=source_sha,
                    order_position=order_position,
                    prompt_digest=prompt_digest,
                )
                performance_rows.append(row)
                lifecycle_rows.append({
                    "policy": policy,
                    "context": context,
                    "repetition": repetition,
                    "lease_grant_ns": row["lease_grant_ns"],
                    "scheduler_lifecycle_ns": row[
                        "scheduler_lifecycle_ns"
                    ],
                })
                write_jsonl(
                    output_dir / "performance_rows.jsonl",
                    performance_rows,
                )
                write_jsonl(
                    output_dir / "lifecycle_samples.jsonl",
                    lifecycle_rows,
                )
    correctness_rows = []
    for context, prompt_tokens, generated_tokens in CONTEXT_CASES:
        for policy in POLICIES:
            _ACTIVE_POLICY = policy
            legacy_rows = base.run_correctness_probe(
                model=model,
                run_dir=output_dir,
                run_tag=run_tag,
                source_commit=source_sha,
                policy="decode_burst_k8",
                context_bucket=context,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            correctness_rows.extend(
                _correctness_rows_from_legacy(
                    legacy_rows,
                    base=base,
                    run_dir=output_dir,
                    policy=policy,
                    source_sha=source_sha,
                )
            )
            write_jsonl(
                output_dir / "correctness_rows.jsonl",
                correctness_rows,
            )
    write_json(
        output_dir / "workload_manifest.json",
        build_workload_manifest(
            run_tag=run_tag,
            source_sha=source_sha,
        ),
    )
    summary = summarize_evidence(performance_rows, correctness_rows)
    write_json(output_dir / "summary.json", summary)
    artifact_names = (
        "workload_manifest.json",
        "performance_rows.jsonl",
        "correctness_rows.jsonl",
        "lifecycle_samples.jsonl",
    )
    repo_root = Path(__file__).resolve().parents[1]
    write_json(
        output_dir / "source_manifest.json",
        {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "run_tag": run_tag,
            "source_sha": source_sha,
            "source_patch_sha256": hashlib.sha256(b"").hexdigest(),
            "source_file_sha256": {
                relative: hashlib.sha256(
                    (repo_root / relative).read_bytes()
                ).hexdigest()
                for relative in SOURCE_FILES
            },
            "artifact_sha256": {
                name: hashlib.sha256(
                    (output_dir / name).read_bytes()
                ).hexdigest()
                for name in artifact_names
            },
        },
    )
    write_json(
        output_dir / "runner_receipt.json",
        {
            "schema": RUNNER_RECEIPT_SCHEMA,
            "run_tag": run_tag,
            "source_sha": source_sha,
            "performance_rows": len(performance_rows),
            "correctness_rows": len(correctness_rows),
            "classification": summary["classification"],
        },
    )
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source-sha")
    parser.add_argument("--run-tag")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
    )
    args = parser.parse_args(argv)
    if args.hardware:
        if not args.source_sha or not args.run_tag:
            raise SystemExit(
                "--hardware requires --source-sha and --run-tag"
            )
        summary = run_hardware_gate(
            model=args.model,
            output_dir=args.run_dir,
            source_sha=args.source_sha,
            run_tag=args.run_tag,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    else:
        summary = produce_summary(args.run_dir)
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
