#!/usr/bin/env python3
"""CPU profile for generation-sealed exact-burst lease identity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
import tracemalloc
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import profile_exact_burst_lease_local_delta_journal as base


PROFILE_SCHEMA = (
    "exact_burst_generation_sealed_"
    "lease_identity_cpu_profile_v1"
)
CONTEXT_LENGTHS = (249, 2041, 8185)
POLICIES = ("full_identity", "generation_sealed")
DEFAULT_REPETITIONS = 100
WARMUP_REPETITIONS = 10
BLOCK_SIZE = 256
MIN_8K_LIFECYCLE_MEDIAN_IMPROVEMENT_PCT = 30.0
MIN_8K_LIFECYCLE_P95_IMPROVEMENT_PCT = 25.0
MIN_AGGREGATE_LIFECYCLE_MEDIAN_IMPROVEMENT_PCT = 20.0


def _config(policy: str):
    return SimpleNamespace(
        **{
            **vars(base._config(delta_enabled=True)),
            (
                "exact_greedy_decode_burst_"
                "generation_sealed_identity"
            ): policy == "generation_sealed",
        }
    )


def _build_fixture(
    policy: str,
    sequence_length: int,
):
    (
        scheduler_module,
        sequence_module,
        sampling_module,
        _split_module,
    ) = base._load_runtime()
    Sequence = sequence_module.Sequence
    SequenceStatus = sequence_module.SequenceStatus
    SamplingParams = sampling_module.SamplingParams
    Scheduler = scheduler_module.Scheduler
    Sequence.block_size = BLOCK_SIZE
    scheduler = Scheduler(_config(policy))
    sequence = Sequence(
        list(range(sequence_length)),
        SamplingParams(
            temperature=0.0,
            max_tokens=32,
            ignore_eos=True,
        ),
    )
    scheduler.block_manager.allocate(sequence)
    sequence.num_computed_tokens = len(sequence)
    sequence.status = SequenceStatus.RUNNING
    scheduler.running.append(sequence)
    scheduler.schedule_generation = 1
    return scheduler_module, scheduler, sequence


def build_profile_cases() -> tuple[dict, ...]:
    return tuple(
        {
            "policy": policy,
            "sequence_length": sequence_length,
        }
        for sequence_length in CONTEXT_LENGTHS
        for policy in POLICIES
    )


def _nearest_rank(
    values: list[float],
    percentile: float,
) -> float:
    if not values:
        raise ValueError("nearest-rank input cannot be empty")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _collect_timing_samples(
    run_lifecycle,
    repetitions: int,
) -> tuple[list[float], list[float]]:
    grant_durations_us = []
    lifecycle_durations_us = []
    for _ in range(repetitions):
        grant_duration_us, lifecycle_duration_us = (
            run_lifecycle(measure=True)
        )
        grant_durations_us.append(grant_duration_us)
        lifecycle_durations_us.append(lifecycle_duration_us)
    return grant_durations_us, lifecycle_durations_us


def run_profile_case(
    *,
    policy: str,
    sequence_length: int,
    repetitions: int,
) -> dict:
    if policy not in POLICIES:
        raise ValueError("policy must be a supported profile policy")
    if sequence_length not in CONTEXT_LENGTHS:
        raise ValueError(
            "sequence_length must be a fixed profile context"
        )
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions <= 0
    ):
        raise ValueError("repetitions must be a positive integer")

    scheduler_module, scheduler, sequence = _build_fixture(
        policy,
        sequence_length,
    )
    identity_rows_visited = 0
    original_block_identities = (
        scheduler.block_manager.block_identities
    )

    def count_block_identities(block_ids):
        nonlocal identity_rows_visited
        identity_rows_visited += len(block_ids)
        return original_block_identities(block_ids)

    scheduler.block_manager.block_identities = (
        count_block_identities
    )

    def run_lifecycle(
        *,
        measure: bool,
    ) -> tuple[float, float] | None:
        lifecycle_started_ns = (
            time.perf_counter_ns() if measure else 0
        )
        grant_started_ns = (
            time.perf_counter_ns() if measure else 0
        )
        lease = scheduler.prepare_exact_greedy_decode_burst(
            (sequence,),
            schedule_generation=1,
            graph_generation=7,
            enabled=True,
            configured_width=8,
            is_prefill=False,
            do_sample=True,
            batch_kind=None,
            completion_only=True,
            tensor_parallel_size=1,
            rank=0,
            graph_available=True,
            incompatible_modes=(),
            allow_single_token_gate=False,
            split_phase_enabled=False,
            ragged_coalescing_enabled=False,
        )
        grant_duration_us = (
            (
                time.perf_counter_ns() - grant_started_ns
            )
            / 1000.0
            if measure
            else 0.0
        )
        if lease is None or lease.authorized_token_count != 8:
            raise RuntimeError(
                "failed to build one-phase K8 exact-burst lease"
            )
        tokens = (11, 12, 13, 14, 15, 16, 17, 18)
        result = scheduler_module.ExactGreedyDecodeBurstResult(
            lease_identity_sha256=lease.identity_sha256,
            tokens=tokens,
            replay_count=8,
            final_input_token=tokens[-1],
            final_position=lease.first_write_position + 8,
            final_context_length=lease.initial_sequence_length + 8,
            final_physical_slot=lease.last_physical_slot + 1,
            graph_identity_sha256="a" * 64,
            token_d2h_calls=1,
            sampled_logit_d2h_calls=0,
        )
        prepared = (
            scheduler.prepare_exact_greedy_decode_burst_commit(
                (sequence,),
                lease,
                result,
            )
        )
        scheduler.rollback_prepared_postprocess(prepared)
        scheduler.cancel_exact_greedy_decode_burst(
            lease,
            "profile_cleanup",
        )
        if not measure:
            return None
        lifecycle_duration_us = (
            time.perf_counter_ns() - lifecycle_started_ns
        ) / 1000.0
        return grant_duration_us, lifecycle_duration_us

    for _ in range(WARMUP_REPETITIONS):
        run_lifecycle(measure=True)
    identity_rows_visited = 0
    (
        grant_durations_us,
        lifecycle_durations_us,
    ) = _collect_timing_samples(
        run_lifecycle,
        repetitions,
    )
    timed_identity_rows_visited = identity_rows_visited
    summary = scheduler.exact_greedy_decode_burst_summary()
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    try:
        for _ in range(repetitions):
            run_lifecycle(measure=False)
        after = tracemalloc.take_snapshot()
    finally:
        tracemalloc.stop()
    positive_allocation_bytes = sum(
        statistic.size_diff
        for statistic in after.compare_to(before, "lineno")
        if statistic.size_diff > 0
    )
    return {
        "schema": PROFILE_SCHEMA,
        "policy": policy,
        "sequence_length": sequence_length,
        "sample_count": repetitions,
        "warmup_count": WARMUP_REPETITIONS,
        "lease_grant_median_us": statistics.median(
            grant_durations_us
        ),
        "lease_grant_p95_us": _nearest_rank(
            grant_durations_us,
            0.95,
        ),
        "lease_lifecycle_median_us": statistics.median(
            lifecycle_durations_us
        ),
        "lease_lifecycle_p95_us": _nearest_rank(
            lifecycle_durations_us,
            0.95,
        ),
        "lease_grant_samples_us": grant_durations_us,
        "lease_lifecycle_samples_us": lifecycle_durations_us,
        "identity_rows_visited": timed_identity_rows_visited,
        "identity_seal_cold_captures": summary[
            "identity_seal_cold_captures"
        ],
        "identity_seal_hot_reuses": summary[
            "identity_seal_hot_reuses"
        ],
        "identity_seal_validations": summary[
            "identity_seal_validations"
        ],
        "positive_python_allocation_bytes": (
            positive_allocation_bytes
        ),
        "fallback_counts": summary[
            "identity_seal_fallback_counts"
        ],
        "identity_seal_fallback_counts": summary[
            "identity_seal_fallback_counts"
        ],
        "failure_count": summary["failures"],
        "pending_lease_count": summary["pending_leases"],
        "rollback_count": summary[
            "lease_local_delta_journal_one_phase_rollbacks"
        ],
    }


def _improvement_pct(
    baseline: float,
    candidate: float,
) -> float:
    if baseline <= 0:
        raise ValueError("baseline timing must be positive")
    return (baseline - candidate) / baseline * 100.0


def summarize_profile(rows: tuple[dict, ...]) -> dict:
    expected_keys = {
        (policy, sequence_length)
        for sequence_length in CONTEXT_LENGTHS
        for policy in POLICIES
    }
    indexed = {
        (row["policy"], row["sequence_length"]): row
        for row in rows
    }
    if len(rows) != len(expected_keys) or set(indexed) != expected_keys:
        raise ValueError("complete profile matrix is required")

    by_context = {}
    for sequence_length in CONTEXT_LENGTHS:
        baseline = indexed[("full_identity", sequence_length)]
        candidate = indexed[("generation_sealed", sequence_length)]
        by_context[str(sequence_length)] = {
            "baseline_lifecycle_median_us": baseline[
                "lease_lifecycle_median_us"
            ],
            "candidate_lifecycle_median_us": candidate[
                "lease_lifecycle_median_us"
            ],
            "lifecycle_median_improvement_pct": _improvement_pct(
                baseline["lease_lifecycle_median_us"],
                candidate["lease_lifecycle_median_us"],
            ),
            "baseline_lifecycle_p95_us": baseline[
                "lease_lifecycle_p95_us"
            ],
            "candidate_lifecycle_p95_us": candidate[
                "lease_lifecycle_p95_us"
            ],
            "lifecycle_p95_improvement_pct": _improvement_pct(
                baseline["lease_lifecycle_p95_us"],
                candidate["lease_lifecycle_p95_us"],
            ),
            "candidate_to_baseline_median_ratio": (
                candidate["lease_lifecycle_median_us"]
                / baseline["lease_lifecycle_median_us"]
            ),
        }

    context_median_ratios = [
        by_context[str(sequence_length)][
            "candidate_to_baseline_median_ratio"
        ]
        for sequence_length in CONTEXT_LENGTHS
    ]
    geometric_mean_ratio = math.prod(
        context_median_ratios
    ) ** (
        1.0 / len(context_median_ratios)
    )
    aggregate = {
        "aggregation": (
            "geometric_mean_of_per_context_median_ratios"
        ),
        "context_count": len(context_median_ratios),
        "candidate_to_baseline_median_ratios": (
            context_median_ratios
        ),
        "geometric_mean_candidate_to_baseline_ratio": (
            geometric_mean_ratio
        ),
        "lifecycle_median_improvement_pct": (
            (1.0 - geometric_mean_ratio) * 100.0
        ),
    }

    candidate_rows = [
        indexed[("generation_sealed", sequence_length)]
        for sequence_length in CONTEXT_LENGTHS
    ]
    long_context = by_context["8185"]
    checks = {
        "8k_lifecycle_median_improvement": (
            long_context["lifecycle_median_improvement_pct"]
            >= MIN_8K_LIFECYCLE_MEDIAN_IMPROVEMENT_PCT
        ),
        "8k_lifecycle_p95_improvement": (
            long_context["lifecycle_p95_improvement_pct"]
            >= MIN_8K_LIFECYCLE_P95_IMPROVEMENT_PCT
        ),
        "aggregate_lifecycle_median_improvement": (
            aggregate["lifecycle_median_improvement_pct"]
            >= MIN_AGGREGATE_LIFECYCLE_MEDIAN_IMPROVEMENT_PCT
        ),
        "candidate_hot_path_identity_rows_zero": all(
            row["identity_rows_visited"] == 0
            for row in candidate_rows
        ),
        "candidate_one_cold_capture_per_fixture": all(
            row["identity_seal_cold_captures"] == 1
            for row in candidate_rows
        ),
        "candidate_no_fallback_or_rollback_failures": all(
            not row["identity_seal_fallback_counts"]
            and row["failure_count"] == 0
            and row["pending_lease_count"] == 0
            and row["rollback_count"]
            == row["warmup_count"] + row["sample_count"]
            for row in candidate_rows
        ),
    }
    return {
        "by_context": by_context,
        "aggregate": aggregate,
        "checks": checks,
        "classification": (
            "GO" if all(checks.values()) else "NO_GO"
        ),
    }


def write_profile_artifacts(
    output_dir: Path,
    rows: tuple[dict, ...],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rows.jsonl").write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        )
    )
    evaluation = summarize_profile(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema": PROFILE_SCHEMA,
                "row_count": len(rows),
                "contexts": list(CONTEXT_LENGTHS),
                "policies": list(POLICIES),
                "rows": list(rows),
                **evaluation,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
    )
    args = parser.parse_args(argv)
    rows = tuple(
        run_profile_case(
            policy=case["policy"],
            sequence_length=case["sequence_length"],
            repetitions=args.repetitions,
        )
        for case in build_profile_cases()
    )
    write_profile_artifacts(args.output_dir, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
