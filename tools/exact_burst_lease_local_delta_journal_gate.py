#!/usr/bin/env python3
"""Gate contract for exact-burst lease-local delta journals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from types import MethodType


GATE_SCHEMA = "exact_burst_lease_local_delta_journal_gate_v1"
PERFORMANCE_ROW_SCHEMA = (
    "exact_burst_lease_local_delta_journal_performance_v1"
)
CORRECTNESS_ROW_SCHEMA = (
    "exact_burst_lease_local_delta_journal_correctness_v1"
)
WORKLOAD_MANIFEST_SCHEMA = (
    "exact_burst_lease_local_delta_journal_workload_v1"
)
SOURCE_MANIFEST_SCHEMA = (
    "exact_burst_lease_local_delta_journal_source_v1"
)
RUNNER_RECEIPT_SCHEMA = (
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
DEFAULT_MODEL = (
    "/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B"
)
CONTEXT_CASES = (
    ("short", 256, 128),
    ("medium", 2048, 128),
    ("long", 8192, 128),
)
SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/block_manager.py",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
    "tinyvllm/engine/exact_greedy_decode_burst_split_phase.py",
    "tinyvllm/engine/llm_engine.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tools/exact_burst_lease_local_delta_journal_gate.py",
    "tools/exact_burst_lease_local_delta_journal_verify.py",
)
GO = "GO_EXACT_BURST_LEASE_LOCAL_DELTA_JOURNAL"
NO_GO_PERFORMANCE = "NO_GO_PERFORMANCE"
NO_GO_CORRECTNESS = "NO_GO_CORRECTNESS"
NO_GO_TRANSACTIONAL_SAFETY = (
    "NO_GO_TRANSACTIONAL_SAFETY"
)
NO_GO_EVIDENCE_INCOMPLETE = (
    "NO_GO_EVIDENCE_INCOMPLETE"
)


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
        "policies": POLICIES,
        "contexts": CONTEXTS,
        "performance_repetitions": PERFORMANCE_REPETITIONS,
        "correctness_sampling_points": len(SAMPLING_POINTS),
        "sampling_points": SAMPLING_POINTS,
        "performance_row_count": PERFORMANCE_ROW_COUNT,
        "correctness_row_count": CORRECTNESS_ROW_COUNT,
        "policy_order": {
            str(repetition): {
                context: policy_order(
                    repetition,
                    context_index,
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


def _validate_performance_rows(rows: list[dict]) -> dict:
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
        raise ValueError("performance row inventory is incomplete")
    return indexed


def _validate_correctness_rows(rows: list[dict]) -> dict:
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


def summarize_evidence(
    performance_rows: list[dict],
    correctness_rows: list[dict],
) -> dict:
    performance = _validate_performance_rows(performance_rows)
    correctness = _validate_correctness_rows(correctness_rows)
    run_tags = {
        row.get("run_tag")
        for row in performance_rows + correctness_rows
    }
    source_shas = {
        row.get("source_sha")
        for row in performance_rows + correctness_rows
    }
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

    sampled_logit_max_abs_diff = 0.0
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
            for field in inventory_fields:
                inventory_equal[field] = (
                    inventory_equal[field]
                    and generic[field] == candidate[field]
                )

    metrics = {
        "schema": GATE_SCHEMA,
        "run_tag": next(iter(run_tags)),
        "source_sha": next(iter(source_shas)),
        "performance_row_count": len(performance_rows),
        "correctness_row_count": len(correctness_rows),
        "output_exact": output_exact,
        "sampled_logit_max_abs_diff": (
            sampled_logit_max_abs_diff
        ),
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
        generic_samples = [
            sample
            for repetition in range(PERFORMANCE_REPETITIONS)
            for sample in performance[
                (repetition, context, "generic")
            ]["phase_prepare_ns"]
        ]
        candidate_samples = [
            sample
            for repetition in range(PERFORMANCE_REPETITIONS)
            for sample in performance[
                (
                    repetition,
                    context,
                    "lease_local_delta",
                )
            ]["phase_prepare_ns"]
        ]
        generic_median = _median(generic_samples)
        candidate_median = _median(candidate_samples)
        generic_p95 = _nearest_rank(generic_samples, 0.95)
        candidate_p95 = _nearest_rank(candidate_samples, 0.95)
        metrics[
            f"{context}_prepare_median_regression_pct"
        ] = _regression_pct(
            generic_median,
            candidate_median,
        )
        metrics[
            f"{context}_prepare_p95_regression_pct"
        ] = _regression_pct(
            generic_p95,
            candidate_p95,
        )
        metrics[
            f"{context}_prepare_median_improvement_pct"
        ] = _improvement_pct(
            generic_median,
            candidate_median,
        )
        metrics[
            f"{context}_prepare_p95_improvement_pct"
        ] = _improvement_pct(
            generic_p95,
            candidate_p95,
        )

    def aggregate(field: str, *, throughput: bool = False):
        generic = _median(
            row[field]
            for row in performance_rows
            if row["policy"] == "generic"
        )
        candidate = _median(
            row[field]
            for row in performance_rows
            if row["policy"] == "lease_local_delta"
        )
        compare = (
            _throughput_regression_pct
            if throughput
            else _regression_pct
        )
        return compare(generic, candidate)

    metrics.update({
        "aggregate_tpot_median_regression_pct": (
            _regression_pct(
                _median(
                    _median(row["tpot_samples_ns"])
                    for row in performance_rows
                    if row["policy"] == "generic"
                ),
                _median(
                    _median(row["tpot_samples_ns"])
                    for row in performance_rows
                    if row["policy"]
                    == "lease_local_delta"
                ),
            )
        ),
        "aggregate_tpot_p95_regression_pct": (
            _regression_pct(
                _median(
                    _nearest_rank(
                        row["tpot_samples_ns"],
                        0.95,
                    )
                    for row in performance_rows
                    if row["policy"] == "generic"
                ),
                _median(
                    _nearest_rank(
                        row["tpot_samples_ns"],
                        0.95,
                    )
                    for row in performance_rows
                    if row["policy"]
                    == "lease_local_delta"
                ),
            )
        ),
        "aggregate_ttft_regression_pct": aggregate(
            "ttft_ns"
        ),
        "aggregate_e2e_regression_pct": aggregate("e2e_ns"),
        "throughput_regression_pct": aggregate(
            "output_tokens_per_second",
            throughput=True,
        ),
    })
    generic_reserved = max(
        int(row["cuda_peak_reserved_bytes"])
        for row in performance_rows
        if row["policy"] == "generic"
    )
    candidate_reserved = max(
        int(row["cuda_peak_reserved_bytes"])
        for row in performance_rows
        if row["policy"] == "lease_local_delta"
    )
    metrics["reserved_memory_regression_pct"] = (
        _regression_pct(
            generic_reserved,
            candidate_reserved,
        )
    )
    candidate_rows = [
        row
        for row in performance_rows
        if row["policy"] == "lease_local_delta"
    ]
    metrics["candidate_fallbacks"] = sum(
        sum(int(value) for value in row["delta_fallbacks"].values())
        for row in candidate_rows
    )
    metrics["candidate_rollbacks"] = sum(
        int(row["delta_rollbacks"])
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
        or float(
            metrics.get(
                "sampled_logit_max_abs_diff",
                math.inf,
            )
        )
        != 0.0
        or metrics.get("forward_inventory_equal") is not True
        or metrics.get("replay_inventory_equal") is not True
        or metrics.get("d2h_call_inventory_equal") is not True
        or metrics.get("d2h_byte_inventory_equal") is not True
    ):
        return NO_GO_CORRECTNESS
    if (
        int(metrics.get("candidate_fallbacks", -1)) != 0
        or int(metrics.get("candidate_rollbacks", -1)) != 0
    ):
        return NO_GO_TRANSACTIONAL_SAFETY
    if (
        float(
            metrics.get(
                "long_prepare_median_improvement_pct",
                -math.inf,
            )
        )
        < 50.0
        or float(
            metrics.get(
                "long_prepare_p95_improvement_pct",
                -math.inf,
            )
        )
        < 50.0
        or any(
            float(metrics.get(field, math.inf)) > 3.0
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
        or float(
            metrics.get(
                "reserved_memory_regression_pct",
                math.inf,
            )
        )
        > 1.0
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


_ACTIVE_DELTA_POLICY = False
_LAST_PHASE_PREPARE_NS = ()
_LAST_DELTA_COUNTERS = {}


def _counter_difference(
    before: dict,
    after: dict,
    field: str,
) -> int:
    result = int(after.get(field, 0)) - int(
        before.get(field, 0)
    )
    if result < 0:
        raise RuntimeError(
            f"delta journal counter decreased: {field}"
        )
    return result


def _fallback_difference(
    before: dict,
    after: dict,
) -> dict[str, int]:
    before_counts = before.get(
        "lease_local_delta_journal_fallback_counts",
        {},
    )
    after_counts = after.get(
        "lease_local_delta_journal_fallback_counts",
        {},
    )
    result = {}
    for reason in sorted(set(before_counts) | set(after_counts)):
        difference = int(after_counts.get(reason, 0)) - int(
            before_counts.get(reason, 0)
        )
        if difference < 0:
            raise RuntimeError(
                "delta journal fallback counter decreased"
            )
        if difference:
            result[reason] = difference
    return result


def _activate_gpu_harness():
    from tools import profile_exact_burst_split_phase as split
    from tools import profile_exact_greedy_decode_burst as base

    original_run_request = split._run_request
    original_combined_summary = split._combined_summary

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
            exact_greedy_decode_burst_split_phase=True,
            exact_greedy_decode_burst_ragged_coalescing=False,
            exact_greedy_decode_burst_continuation=False,
            exact_greedy_decode_burst_tokens=8,
            exact_greedy_decode_burst_lease_local_delta_journal=(
                _ACTIVE_DELTA_POLICY
            ),
        )
        scheduler = llm.scheduler
        scheduler._delta_gate_phase_prepare_ns = []
        original_prepare = (
            scheduler
            .prepare_exact_greedy_decode_burst_phase_commit
        )

        def timed_prepare(_owner, *args, **kwargs):
            started_ns = time.perf_counter_ns()
            try:
                return original_prepare(*args, **kwargs)
            finally:
                scheduler._delta_gate_phase_prepare_ns.append(
                    time.perf_counter_ns() - started_ns
                )

        scheduler.prepare_exact_greedy_decode_burst_phase_commit = (
            MethodType(timed_prepare, scheduler)
        )
        return llm

    def run_request(llm, **kwargs):
        global _LAST_PHASE_PREPARE_NS
        measured = kwargs.get("profile_label") is not None
        if measured:
            llm.scheduler._delta_gate_phase_prepare_ns.clear()
        result = original_run_request(llm, **kwargs)
        if measured:
            _LAST_PHASE_PREPARE_NS = tuple(
                llm.scheduler._delta_gate_phase_prepare_ns
            )
        return result

    def combined_summary(
        llm,
        before,
        *,
        correctness_trace=False,
    ):
        global _LAST_DELTA_COUNTERS
        after = (
            llm.scheduler.exact_greedy_decode_burst_summary()
        )
        fields = (
            "lease_local_delta_journal_attempts",
            "lease_local_delta_journal_captures",
            "lease_local_delta_journal_commits",
            "lease_local_delta_journal_rollbacks",
            "lease_local_delta_journal_published_blocks",
        )
        _LAST_DELTA_COUNTERS = {
            field: _counter_difference(
                before[1],
                after,
                field,
            )
            for field in fields
        }
        _LAST_DELTA_COUNTERS["fallback_counts"] = (
            _fallback_difference(before[1], after)
        )
        return original_combined_summary(
            llm,
            before,
            correctness_trace=correctness_trace,
        )

    base._construct_llm = construct_llm
    split._construct_llm = construct_llm
    base._run_request = run_request
    split._run_request = run_request
    base._combined_summary = combined_summary
    split._combined_summary = combined_summary
    return split, base


def _d2h_inventory(summary: dict) -> tuple[int, int]:
    calls = (
        int(summary.get("prefix_token_d2h_calls", 0))
        + int(summary.get("suffix_token_d2h_calls", 0))
        + int(summary.get("final_token_d2h_calls", 0))
        + int(summary.get("sampled_logit_d2h_calls", 0))
    )
    byte_count = (
        int(summary.get("prefix_token_d2h_bytes", 0))
        + int(summary.get("suffix_token_d2h_bytes", 0))
        + int(summary.get("final_token_d2h_bytes", 0))
    )
    return calls, byte_count


def _performance_row(
    legacy: dict,
    *,
    policy: str,
    source_sha: str,
    order_position: int,
    prompt_digest: str,
) -> dict:
    counters = dict(_LAST_DELTA_COUNTERS)
    raw_fallbacks = dict(counters.get("fallback_counts", {}))
    expected_fallbacks = {
        reason: count
        for reason, count in raw_fallbacks.items()
        if reason == "terminal_suffix"
    }
    unexpected_fallbacks = {
        reason: count
        for reason, count in raw_fallbacks.items()
        if reason != "terminal_suffix"
    }
    summary = legacy["exact_greedy_decode_burst_summary"]
    d2h_calls, d2h_bytes = _d2h_inventory(summary)
    phase_samples = list(_LAST_PHASE_PREPARE_NS)
    if not phase_samples:
        raise RuntimeError("phase prepare samples are missing")
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
        "phase_prepare_ns": phase_samples,
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
        "delta_attempts": int(
            counters.get(
                "lease_local_delta_journal_attempts",
                0,
            )
        ),
        "delta_captures": int(
            counters.get(
                "lease_local_delta_journal_captures",
                0,
            )
        ),
        "delta_commits": int(
            counters.get(
                "lease_local_delta_journal_commits",
                0,
            )
        ),
        "delta_rollbacks": int(
            counters.get(
                "lease_local_delta_journal_rollbacks",
                0,
            )
        ),
        "delta_published_blocks": int(
            counters.get(
                "lease_local_delta_journal_published_blocks",
                0,
            )
        ),
        "delta_expected_fallbacks": expected_fallbacks,
        "delta_fallbacks": unexpected_fallbacks,
    }


def _correctness_rows(
    legacy_rows: list[dict],
    *,
    split,
    run_dir: Path,
    policy: str,
    source_sha: str,
) -> list[dict]:
    rows = []
    for legacy in legacy_rows:
        summary = legacy["exact_greedy_decode_burst_summary"]
        d2h_calls, d2h_bytes = _d2h_inventory(summary)
        logits = split.read_float32_sidecar(
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
        or any(character not in "0123456789abcdef" for character in source_sha)
    ):
        raise ValueError("source SHA must be a full lowercase SHA-1")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    split, base = _activate_gpu_harness()
    performance_rows = []
    phase_rows = []
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
                legacy = split.run_case(
                    model=model,
                    run_tag=run_tag,
                    source_commit=source_sha,
                    policy="decode_burst_k8_split_phase",
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
                phase_rows.append({
                    "policy": policy,
                    "context": context,
                    "repetition": repetition,
                    "phase_prepare_ns": row[
                        "phase_prepare_ns"
                    ],
                })
                write_jsonl(
                    output_dir / "performance_rows.jsonl",
                    performance_rows,
                )
                write_jsonl(
                    output_dir / "phase_samples.jsonl",
                    phase_rows,
                )
    correctness_rows = []
    for context, prompt_tokens, generated_tokens in CONTEXT_CASES:
        for policy in POLICIES:
            _ACTIVE_DELTA_POLICY = (
                policy == "lease_local_delta"
            )
            legacy_rows = split.run_correctness_probe(
                model=model,
                run_dir=output_dir,
                run_tag=run_tag,
                source_commit=source_sha,
                policy="decode_burst_k8_split_phase",
                context_bucket=context,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            correctness_rows.extend(
                _correctness_rows(
                    legacy_rows,
                    split=split,
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
        "phase_samples.jsonl",
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
            "exit_code": 0,
        },
    )
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input-dir", type=Path)
    mode.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source-sha")
    parser.add_argument("--run-tag")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
    )
    args = parser.parse_args(argv)
    if args.input_dir is not None:
        summary = produce_summary(args.input_dir)
    else:
        if not args.source_sha or not args.run_tag:
            parser.error(
                "--source-sha and --run-tag are required "
                "with --output-dir"
            )
        summary = run_hardware_gate(
            model=args.model,
            output_dir=args.output_dir,
            source_sha=args.source_sha,
            run_tag=args.run_tag,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    print(summary["classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
