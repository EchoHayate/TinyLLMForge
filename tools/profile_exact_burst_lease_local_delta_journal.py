from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time
import tracemalloc
import types
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCHEMA = (
    "exact_burst_lease_local_delta_journal_cpu_profile_v1"
)
CONTEXT_LENGTHS = (249, 2041, 8185)
POLICIES = ("generic", "lease_local_delta")
DEFAULT_REPETITIONS = 100
WARMUP_REPETITIONS = 10
BLOCK_SIZE = 256


class _FakeXXH64:
    def __init__(self):
        self._hash = hashlib.blake2b(digest_size=8)

    def update(self, data):
        self._hash.update(data)

    def intdigest(self):
        return int.from_bytes(self._hash.digest(), "little")


class _TorchDType:
    def __init__(self, name, itemsize):
        self.name = name
        self.itemsize = itemsize

    def __str__(self):
        return f"torch.{self.name}"


def _load_file_module(module_name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load runtime module {module_name}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_runtime():
    scheduler_module = sys.modules.get(
        "tinyvllm.engine.scheduler"
    )
    if scheduler_module is not None:
        return (
            scheduler_module,
            sys.modules["tinyvllm.engine.sequence"],
            sys.modules["tinyvllm.sampling_params"],
            sys.modules[
                "tinyvllm.engine."
                "exact_greedy_decode_burst_split_phase"
            ],
        )

    for package_name in ("tinyvllm", "tinyvllm.engine"):
        package = types.ModuleType(package_name)
        package.__path__ = [
            str(ROOT / package_name.replace(".", "/"))
        ]
        sys.modules[package_name] = package

    config_module = types.ModuleType("tinyvllm.config")
    config_module.Config = object
    sys.modules["tinyvllm.config"] = config_module

    try:
        import xxhash  # noqa: F401
    except ImportError:
        xxhash_module = types.ModuleType("xxhash")
        xxhash_module.xxh64 = _FakeXXH64
        sys.modules["xxhash"] = xxhash_module

    try:
        import torch  # noqa: F401
    except ImportError:
        torch_module = types.ModuleType("torch")
        torch_module.float16 = _TorchDType("float16", 2)
        torch_module.bfloat16 = _TorchDType("bfloat16", 2)
        torch_module.float32 = _TorchDType("float32", 4)
        sys.modules["torch"] = torch_module

    sampling_module = _load_file_module(
        "tinyvllm.sampling_params",
        "tinyvllm/sampling_params.py",
    )
    sequence_module = _load_file_module(
        "tinyvllm.engine.sequence",
        "tinyvllm/engine/sequence.py",
    )
    _load_file_module(
        "tinyvllm.engine.block_manager",
        "tinyvllm/engine/block_manager.py",
    )
    _load_file_module(
        "tinyvllm.engine.exact_greedy_decode_burst",
        "tinyvllm/engine/exact_greedy_decode_burst.py",
    )
    split_module = _load_file_module(
        "tinyvllm.engine.exact_greedy_decode_burst_split_phase",
        "tinyvllm/engine/exact_greedy_decode_burst_split_phase.py",
    )
    _load_file_module(
        "tinyvllm.engine.hybrid_state",
        "tinyvllm/engine/hybrid_state.py",
    )
    _load_file_module(
        "tinyvllm.engine.speculative_selection",
        "tinyvllm/engine/speculative_selection.py",
    )
    scheduler_module = _load_file_module(
        "tinyvllm.engine.scheduler",
        "tinyvllm/engine/scheduler.py",
    )
    return (
        scheduler_module,
        sequence_module,
        sampling_module,
        split_module,
    )


def _config(*, delta_enabled: bool):
    return SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=16384,
        max_model_len=16384,
        max_num_prefill_tokens_per_step=0,
        chunked_prefill_decode_first=True,
        chunked_prefill_max_consecutive_chunks=0,
        chunked_prefill_mixed_batch=False,
        chunked_prefill_mixed_min_prompt_tokens=0,
        chunked_prefill_adaptive_mixed=False,
        chunked_prefill_adaptive_enter_waiting=8,
        chunked_prefill_adaptive_exit_waiting=2,
        chunked_prefill_adaptive_transition_steps=2,
        chunked_prefill_adaptive_max_mixed_steps=2,
        chunked_prefill_slo_mixed=False,
        chunked_prefill_slo_target_gap_ns=0,
        chunked_prefill_slo_reserve_ns=0,
        chunked_prefill_slo_cost_intercept_ns=0,
        chunked_prefill_slo_cost_per_prefill_token_ns=0,
        chunked_prefill_slo_min_chunk_tokens=1,
        eos=99,
        num_kvcache_blocks=64,
        kvcache_block_size=BLOCK_SIZE,
        exact_greedy_decode_burst_lease_local_delta_journal=(
            delta_enabled
        ),
    )


class _ReadyCompletion:
    def synchronize(self):
        return None


class _TokenMailbox:
    def __init__(self, tokens):
        self._tokens = tuple(tokens)

    def tolist(self):
        return list(self._tokens)


def _build_fixture(policy: str, sequence_length: int):
    (
        scheduler_module,
        sequence_module,
        sampling_module,
        split_module,
    ) = _load_runtime()
    Sequence = sequence_module.Sequence
    SequenceStatus = sequence_module.SequenceStatus
    SamplingParams = sampling_module.SamplingParams
    Scheduler = scheduler_module.Scheduler
    Sequence.block_size = BLOCK_SIZE
    scheduler = Scheduler(
        _config(delta_enabled=policy == "lease_local_delta")
    )
    sequence = Sequence(
        list(range(sequence_length)),
        SamplingParams(
            temperature=0.0,
            max_tokens=16,
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
        split_phase_enabled=True,
        ragged_coalescing_enabled=False,
    )
    if lease is None:
        raise RuntimeError("failed to build exact-burst lease")
    prefix_ticket, suffix_ticket = (
        split_module.build_exact_burst_publication_tickets(
            parent_lease_identity_sha256=lease.identity_sha256,
            first_write_position=lease.first_write_position,
            first_physical_slot=lease.first_physical_slot,
            parent_token_count=lease.authorized_token_count,
            prefix_token_count=4,
        )
    )

    def transfer(ticket, tokens):
        return split_module.ExactBurstPhaseTransfer(
            ticket=ticket,
            mailbox_generation=1,
            token_count=len(tokens),
            byte_count=len(tokens) * 8,
            completion=_ReadyCompletion(),
            mailbox=_TokenMailbox(tokens),
        )

    split_result = (
        split_module.ExactGreedyDecodeBurstSplitResult(
            parent_lease_identity_sha256=lease.identity_sha256,
            graph_identity_sha256="b" * 64,
            replay_count=8,
            prefix=transfer(
                prefix_ticket,
                (11, 12, 13, 14),
            ),
            suffix=transfer(
                suffix_ticket,
                (15, 16, 17, 18),
            ),
            sampled_logit_d2h_calls=0,
            sampled_logits=(),
        )
    )
    return scheduler_module, scheduler, sequence, lease, split_result


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
    rank = max(
        1,
        math.ceil(percentile * len(ordered)),
    )
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
        split_result,
    ) = _build_fixture(policy, sequence_length)
    compute_hash_calls = 0
    original_compute_hash = scheduler.block_manager.compute_hash

    def count_compute_hash(token_ids, prefix=-1):
        nonlocal compute_hash_calls
        compute_hash_calls += 1
        return original_compute_hash(token_ids, prefix)

    scheduler.block_manager.compute_hash = count_compute_hash
    touched_block_counts = []
    hash_key_counts = []

    def prepare_and_rollback():
        prepared = (
            scheduler.prepare_exact_greedy_decode_burst_phase_commit(
                (sequence,),
                lease,
                split_result,
                phase="prefix",
                tokens=split_result.prefix.wait_tokens(),
            )
        )
        journal = prepared.snapshot
        if isinstance(
            journal,
            scheduler_module.SchedulerPostprocessJournal,
        ):
            touched_block_counts.append(
                journal.touched_block_count
            )
            hash_key_counts.append(len(journal.hashes))
        else:
            touched_block_counts.append(
                int(journal.publication_plan.will_publish)
            )
            hash_key_counts.append(
                int(
                    journal.publication_plan.planned_block_hash
                    is not None
                )
            )
        scheduler.rollback_prepared_postprocess(prepared)

    for _ in range(WARMUP_REPETITIONS):
        prepare_and_rollback()
    touched_block_counts.clear()
    hash_key_counts.clear()
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
        "journal_touched_block_count": max(
            touched_block_counts,
            default=0,
        ),
        "journal_hash_key_count": max(
            hash_key_counts,
            default=0,
        ),
        "compute_hash_calls": compute_hash_calls,
        "delta_attempts": summary[
            "lease_local_delta_journal_attempts"
        ],
        "delta_captures": summary[
            "lease_local_delta_journal_captures"
        ],
        "delta_rollbacks": summary[
            "lease_local_delta_journal_rollbacks"
        ],
        "delta_fallbacks": summary[
            "lease_local_delta_journal_fallback_counts"
        ],
    }


def write_profile_artifacts(
    output_dir: Path,
    rows: tuple[dict, ...],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    rows_path.write_text(
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
    summary = {
        "schema": PROFILE_SCHEMA,
        "row_count": len(rows),
        "contexts": list(CONTEXT_LENGTHS),
        "policies": list(POLICIES),
        "rows": list(rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(
            summary,
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
