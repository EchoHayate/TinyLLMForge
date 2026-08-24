#!/usr/bin/env python3
"""Paired gate for the one-phase exact-burst lease-local journal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from types import MethodType


GATE_SCHEMA = (
    "exact_burst_one_phase_lease_local_journal_gate_v1"
)
PERFORMANCE_ROW_SCHEMA = (
    "exact_burst_one_phase_lease_local_journal_performance_v1"
)
CORRECTNESS_ROW_SCHEMA = (
    "exact_burst_one_phase_lease_local_journal_correctness_v1"
)
WORKLOAD_MANIFEST_SCHEMA = (
    "exact_burst_one_phase_lease_local_journal_workload_v1"
)
SOURCE_MANIFEST_SCHEMA = (
    "exact_burst_one_phase_lease_local_journal_source_v1"
)
RUNNER_RECEIPT_SCHEMA = (
    "exact_burst_one_phase_lease_local_journal_runner_v1"
)
POLICIES = ("generic", "lease_local_delta")
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
DEFAULT_MODEL = (
    "/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B"
)
GO = "GO_EXACT_BURST_ONE_PHASE_LEASE_LOCAL_JOURNAL"
NO_GO_PERFORMANCE = "NO_GO_PERFORMANCE"
NO_GO_CORRECTNESS = "NO_GO_CORRECTNESS"
NO_GO_TRANSACTIONAL_SAFETY = (
    "NO_GO_TRANSACTIONAL_SAFETY"
)
NO_GO_EVIDENCE_INCOMPLETE = "NO_GO_EVIDENCE_INCOMPLETE"
SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/block_manager.py",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
    "tinyvllm/engine/llm_engine.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tools/exact_burst_one_phase_lease_local_journal_gate.py",
    "tools/exact_burst_one_phase_lease_local_journal_verify.py",
)

_ACTIVE_DELTA_POLICY = False
_LAST_PREPARE_NS: tuple[int, ...] = ()
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
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
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
        "policy_order": {
            str(repetition): {
                context: list(
                    policy_order(repetition, context_index)
                )
                for context_index, context in enumerate(CONTEXTS)
            }
            for repetition in range(PERFORMANCE_REPETITIONS)
        },
    }


def _nearest_rank(values, percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or any(
        not math.isfinite(value) for value in ordered
    ):
        raise ValueError(
            "metric samples must be finite and non-empty"
        )
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _median(values) -> float:
    normalized = [float(value) for value in values]
    if not normalized or any(
        not math.isfinite(value) for value in normalized
    ):
        raise ValueError(
            "metric samples must be finite and non-empty"
        )
    return float(statistics.median(normalized))


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


def _validate_correctness_rows(
    rows: list[dict],
) -> dict[tuple[str, str, str], dict]:
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
        logits = row.get("sampled_logits")
        if (
            not isinstance(logits, list)
            or not logits
            or any(
                not math.isfinite(float(value))
                for value in logits
            )
        ):
            raise ValueError(
                "sampled logits must be finite and non-empty"
            )
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


def _paired_inventory_equal(
    generic: dict,
    candidate: dict,
) -> bool:
    return all(
        generic[field] == candidate[field]
        for field in (
            "target_model_forwards",
            "graph_replays",
            "d2h_calls",
            "d2h_bytes",
        )
    )


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
                and _paired_inventory_equal(generic, candidate)
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
                and _paired_inventory_equal(generic, candidate)
            )

    metrics = {
        "schema": GATE_SCHEMA,
        "run_tag": next(iter(run_tags)),
        "source_sha": next(iter(source_shas)),
        "performance_row_count": len(performance_rows),
        "correctness_row_count": len(correctness_rows),
        "output_exact": output_exact,
        "sampled_argmax_exact": sampled_argmax_exact,
        "sampled_logit_max_abs_diff": (
            sampled_logit_max_abs_diff
        ),
        "execution_inventory_equal": inventory_equal,
    }

    for context in CONTEXTS:
        generic_samples = [
            sample
            for row in performance_rows
            if row["policy"] == "generic"
            and row["context"] == context
            for sample in row["prepare_ns"]
        ]
        candidate_samples = [
            sample
            for row in performance_rows
            if row["policy"] == "lease_local_delta"
            and row["context"] == context
            for sample in row["prepare_ns"]
        ]
        metrics[
            f"{context}_prepare_median_improvement_pct"
        ] = _improvement_pct(
            _median(generic_samples),
            _median(candidate_samples),
        )
        metrics[
            f"{context}_prepare_p95_improvement_pct"
        ] = _improvement_pct(
            _nearest_rank(generic_samples, 0.95),
            _nearest_rank(candidate_samples, 0.95),
        )

    all_generic_prepare = [
        sample
        for row in performance_rows
        if row["policy"] == "generic"
        for sample in row["prepare_ns"]
    ]
    all_candidate_prepare = [
        sample
        for row in performance_rows
        if row["policy"] == "lease_local_delta"
        for sample in row["prepare_ns"]
    ]
    metrics["aggregate_prepare_median_improvement_pct"] = (
        _improvement_pct(
            _median(all_generic_prepare),
            _median(all_candidate_prepare),
        )
    )
    metrics["aggregate_prepare_p95_improvement_pct"] = (
        _improvement_pct(
            _nearest_rank(all_generic_prepare, 0.95),
            _nearest_rank(all_candidate_prepare, 0.95),
        )
    )

    def per_row_stat(policy, field, percentile=None):
        values = []
        for row in performance_rows:
            if row["policy"] != policy:
                continue
            value = row[field]
            if percentile is None:
                values.append(float(value))
            else:
                values.append(_nearest_rank(value, percentile))
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
        generic = per_row_stat(
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
        ] = _improvement_pct(generic, candidate)
        metrics[
            f"aggregate_tpot_{label}_regression_pct"
        ] = _regression_pct(generic, candidate)

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
    metrics["classification"] = classify(metrics)
    return metrics


def classify(metrics: dict) -> str:
    if (
        metrics.get("performance_row_count")
        != PERFORMANCE_ROW_COUNT
        or metrics.get("correctness_row_count")
        != CORRECTNESS_ROW_COUNT
    ):
        return NO_GO_EVIDENCE_INCOMPLETE
    if (
        metrics.get("output_exact") is not True
        or metrics.get("sampled_argmax_exact") is not True
        or float(
            metrics.get(
                "sampled_logit_max_abs_diff",
                math.inf,
            )
        )
        != 0.0
        or metrics.get("execution_inventory_equal") is not True
    ):
        return NO_GO_CORRECTNESS
    if (
        metrics.get("candidate_counter_authority") is not True
        or int(
            metrics.get(
                "candidate_generic_journal_captures",
                -1,
            )
        )
        != 0
        or int(
            metrics.get("candidate_one_phase_fallbacks", -1)
        )
        != 0
        or int(
            metrics.get("candidate_one_phase_rollbacks", -1)
        )
        != 0
    ):
        return NO_GO_TRANSACTIONAL_SAFETY
    if (
        float(
            metrics.get(
                "8k_prepare_median_improvement_pct",
                -math.inf,
            )
        )
        < 50.0
        or float(
            metrics.get(
                "8k_prepare_p95_improvement_pct",
                -math.inf,
            )
        )
        < 50.0
        or float(
            metrics.get(
                "aggregate_prepare_median_improvement_pct",
                -math.inf,
            )
        )
        < 35.0
        or float(
            metrics.get(
                "aggregate_prepare_p95_improvement_pct",
                -math.inf,
            )
        )
        < 35.0
        or float(
            metrics.get(
                "aggregate_tpot_median_improvement_pct",
                -math.inf,
            )
        )
        < 1.0
        or float(
            metrics.get(
                "aggregate_tpot_p95_improvement_pct",
                -math.inf,
            )
        )
        < 1.0
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


def produce_summary(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    summary = summarize_evidence(
        read_jsonl(run_dir / "performance_rows.jsonl"),
        read_jsonl(run_dir / "correctness_rows.jsonl"),
    )
    write_json(run_dir / "summary.json", summary)
    return summary


def _counter_difference(
    before: dict,
    after: dict,
    field: str,
) -> int:
    difference = int(after.get(field, 0)) - int(
        before.get(field, 0)
    )
    if difference < 0:
        raise RuntimeError(f"counter decreased: {field}")
    return difference


def _fallback_difference(
    before: dict,
    after: dict,
) -> dict[str, int]:
    field = (
        "lease_local_delta_journal_one_phase_fallback_counts"
    )
    before_map = before.get(field, {})
    after_map = after.get(field, {})
    result = {}
    for reason in sorted(set(before_map) | set(after_map)):
        difference = int(after_map.get(reason, 0)) - int(
            before_map.get(reason, 0)
        )
        if difference < 0:
            raise RuntimeError(
                "one-phase fallback counter decreased"
            )
        if difference:
            result[reason] = difference
    return result


def _activate_gpu_harness():
    from tools import profile_exact_greedy_decode_burst as base

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
        from tinyvllm import LLM

        llm = LLM(
            model,
            max_num_batched_tokens=(
                prompt_tokens + generated_tokens
            ),
            max_num_seqs=1,
            max_model_len=prompt_tokens + generated_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            enforce_eager=False,
            zero_temperature_greedy_fast_path=True,
            graph_resident_greedy_tail=False,
            exact_greedy_decode_burst=True,
            exact_greedy_decode_burst_split_phase=False,
            exact_greedy_decode_burst_ragged_coalescing=False,
            exact_greedy_decode_burst_continuation=False,
            exact_greedy_decode_burst_tokens=8,
            exact_greedy_decode_burst_lease_local_delta_journal=(
                _ACTIVE_DELTA_POLICY
            ),
        )
        scheduler = llm.scheduler
        scheduler._one_phase_journal_prepare_ns = []
        scheduler._one_phase_generic_captures = 0
        original_prepare = (
            scheduler.prepare_exact_greedy_decode_burst_commit
        )

        def timed_prepare(_owner, *args, **kwargs):
            started_ns = time.perf_counter_ns()
            prepared = original_prepare(*args, **kwargs)
            scheduler._one_phase_journal_prepare_ns.append(
                time.perf_counter_ns() - started_ns
            )
            if isinstance(
                prepared.snapshot,
                __import__(
                    "tinyvllm.engine.scheduler",
                    fromlist=["SchedulerPostprocessJournal"],
                ).SchedulerPostprocessJournal,
            ):
                scheduler._one_phase_generic_captures += 1
            return prepared

        scheduler.prepare_exact_greedy_decode_burst_commit = (
            MethodType(timed_prepare, scheduler)
        )
        return llm

    def run_request(llm, **kwargs):
        global _LAST_PREPARE_NS
        measured = kwargs.get("profile_label") is not None
        if measured:
            llm.scheduler._one_phase_journal_prepare_ns.clear()
            llm.scheduler._one_phase_generic_captures = 0
        result = original_run_request(llm, **kwargs)
        if measured:
            _LAST_PREPARE_NS = tuple(
                llm.scheduler._one_phase_journal_prepare_ns
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
            "lease_local_delta_journal_one_phase_attempts",
            "lease_local_delta_journal_one_phase_captures",
            "lease_local_delta_journal_one_phase_commits",
            "lease_local_delta_journal_one_phase_rollbacks",
            (
                "lease_local_delta_journal_one_phase_"
                "published_blocks"
            ),
        )
        _LAST_COUNTERS = {
            field: _counter_difference(
                before[1],
                after,
                field,
            )
            for field in fields
        }
        _LAST_COUNTERS["fallbacks"] = _fallback_difference(
            before[1],
            after,
        )
        _LAST_COUNTERS["generic_journal_captures"] = int(
            llm.scheduler._one_phase_generic_captures
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
    byte_count = int(summary.get("final_token_d2h_bytes", 0))
    return calls, byte_count


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
    prepare_samples = list(_LAST_PREPARE_NS)
    if not prepare_samples:
        raise RuntimeError("one-phase prepare samples are missing")
    counters = dict(_LAST_COUNTERS)
    eligible = int(counters.get(
        "lease_local_delta_journal_one_phase_attempts",
        0,
    ))
    if policy == "generic":
        eligible = int(
            counters.get("generic_journal_captures", 0)
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
        "prepare_ns": prepare_samples,
        "ttft_ns": legacy["ttft_ns"],
        "tpot_samples_ns": legacy[
            "amortized_tpot_samples_ns"
        ],
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
        "target_model_forwards": summary[
            "target_model_forwards"
        ],
        "graph_replays": summary["graph_replays"],
        "d2h_calls": d2h_calls,
        "d2h_bytes": d2h_bytes,
        "eligible_bursts": eligible,
        "generic_journal_captures": int(
            counters.get("generic_journal_captures", 0)
        ),
        "one_phase_attempts": int(counters.get(
            "lease_local_delta_journal_one_phase_attempts",
            0,
        )),
        "one_phase_captures": int(counters.get(
            "lease_local_delta_journal_one_phase_captures",
            0,
        )),
        "one_phase_commits": int(counters.get(
            "lease_local_delta_journal_one_phase_commits",
            0,
        )),
        "one_phase_rollbacks": int(counters.get(
            "lease_local_delta_journal_one_phase_rollbacks",
            0,
        )),
        "one_phase_published_blocks": int(counters.get(
            (
                "lease_local_delta_journal_one_phase_"
                "published_blocks"
            ),
            0,
        )),
        "one_phase_fallbacks": dict(
            counters.get("fallbacks", {})
        ),
    }


def _correctness_rows(
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
            expected_element_count=(
                legacy["logits_element_count"]
            ),
            expected_byte_length=(
                legacy["logits_byte_length"]
            ),
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
            "target_model_forwards": summary[
                "target_model_forwards"
            ],
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
        or any(
            character not in "0123456789abcdef"
            for character in source_sha
        )
    ):
        raise ValueError(
            "source SHA must be a full lowercase SHA-1"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    base = _activate_gpu_harness()
    performance_rows = []
    prepare_rows = []
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
                global _ACTIVE_DELTA_POLICY
                _ACTIVE_DELTA_POLICY = (
                    policy == "lease_local_delta"
                )
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
                    gpu_memory_utilization=(
                        gpu_memory_utilization
                    ),
                )
                row = _performance_row(
                    legacy,
                    policy=policy,
                    source_sha=source_sha,
                    order_position=order_position,
                    prompt_digest=prompt_digest,
                )
                performance_rows.append(row)
                prepare_rows.append({
                    "policy": policy,
                    "context": context,
                    "repetition": repetition,
                    "prepare_ns": row["prepare_ns"],
                })
                write_jsonl(
                    output_dir / "performance_rows.jsonl",
                    performance_rows,
                )
                write_jsonl(
                    output_dir / "prepare_samples.jsonl",
                    prepare_rows,
                )
    correctness_rows = []
    for context, prompt_tokens, generated_tokens in CONTEXT_CASES:
        for policy in POLICIES:
            _ACTIVE_DELTA_POLICY = (
                policy == "lease_local_delta"
            )
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
                _correctness_rows(
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
    summary = summarize_evidence(
        performance_rows,
        correctness_rows,
    )
    write_json(output_dir / "summary.json", summary)
    artifact_names = (
        "workload_manifest.json",
        "performance_rows.jsonl",
        "correctness_rows.jsonl",
        "prepare_samples.jsonl",
    )
    repo_root = Path(__file__).resolve().parents[1]
    write_json(
        output_dir / "source_manifest.json",
        {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "run_tag": run_tag,
            "source_sha": source_sha,
            "source_patch_sha256": hashlib.sha256(
                b""
            ).hexdigest(),
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
    print(
        json.dumps(
            summary,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
