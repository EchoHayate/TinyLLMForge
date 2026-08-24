#!/usr/bin/env python3
"""CPU profile for one-phase exact-burst lease-local journals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
import tracemalloc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import profile_exact_burst_lease_local_delta_journal as base


PROFILE_SCHEMA = (
    "exact_burst_one_phase_"
    "lease_local_journal_cpu_profile_v1"
)
CONTEXT_LENGTHS = (249, 2041, 8185)
POLICIES = ("generic", "lease_local_delta")
DEFAULT_REPETITIONS = 100
WARMUP_REPETITIONS = 10
BLOCK_SIZE = 256


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
    Sequence = scheduler_module.Sequence
    SequenceStatus = scheduler_module.SequenceStatus
    SamplingParams = sampling_module.SamplingParams
    Scheduler = scheduler_module.Scheduler
    Sequence.block_size = BLOCK_SIZE
    scheduler = Scheduler(
        base._config(
            delta_enabled=policy == "lease_local_delta"
        )
    )
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
    return scheduler_module, scheduler, sequence, lease, result


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
    (
        scheduler_module,
        scheduler,
        sequence,
        lease,
        result,
    ) = _build_fixture(policy, sequence_length)
    compute_hash_calls = 0
    generic_journal_captures = 0
    original_compute_hash = scheduler.block_manager.compute_hash

    def count_compute_hash(token_ids, prefix=-1):
        nonlocal compute_hash_calls
        compute_hash_calls += 1
        return original_compute_hash(token_ids, prefix)

    scheduler.block_manager.compute_hash = count_compute_hash

    def prepare_and_rollback():
        nonlocal generic_journal_captures
        prepared = (
            scheduler.prepare_exact_greedy_decode_burst_commit(
                (sequence,),
                lease,
                result,
            )
        )
        if isinstance(
            prepared.snapshot,
            scheduler_module.SchedulerPostprocessJournal,
        ):
            generic_journal_captures += 1
        scheduler.rollback_prepared_postprocess(prepared)

    for _ in range(WARMUP_REPETITIONS):
        prepare_and_rollback()
    compute_hash_calls = 0
    durations_us = []
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    try:
        for _ in range(repetitions):
            started_ns = time.perf_counter_ns()
            prepare_and_rollback()
            durations_us.append(
                (time.perf_counter_ns() - started_ns) / 1000.0
            )
        after = tracemalloc.take_snapshot()
    finally:
        tracemalloc.stop()
    positive_allocation_bytes = sum(
        statistic.size_diff
        for statistic in after.compare_to(before, "lineno")
        if statistic.size_diff > 0
    )
    summary = scheduler.exact_greedy_decode_burst_summary()
    return {
        "schema": PROFILE_SCHEMA,
        "policy": policy,
        "sequence_length": sequence_length,
        "sample_count": repetitions,
        "warmup_count": WARMUP_REPETITIONS,
        "prepare_median_us": statistics.median(durations_us),
        "prepare_p95_us": _nearest_rank(
            durations_us,
            0.95,
        ),
        "positive_python_allocation_bytes": (
            positive_allocation_bytes
        ),
        "compute_hash_calls": compute_hash_calls,
        "generic_journal_captures": generic_journal_captures,
        "one_phase_attempts": summary[
            "lease_local_delta_journal_one_phase_attempts"
        ],
        "one_phase_captures": summary[
            "lease_local_delta_journal_one_phase_captures"
        ],
        "one_phase_rollbacks": summary[
            "lease_local_delta_journal_one_phase_rollbacks"
        ],
        "one_phase_fallbacks": summary[
            "lease_local_delta_journal_one_phase_fallback_counts"
        ],
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
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema": PROFILE_SCHEMA,
                "row_count": len(rows),
                "contexts": list(CONTEXT_LENGTHS),
                "policies": list(POLICIES),
                "rows": list(rows),
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
