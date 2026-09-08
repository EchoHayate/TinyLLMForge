#!/usr/bin/env python3
"""Run the Qwen3.8 topology-local TP2 whole-model gate workloads."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Callable, Mapping


WORKLOADS = {
    "P0": ("causal", 256, 128, 1),
    "P1": ("causal", 2048, 128, 1),
    "Q0": ("online", 256, 128, 4),
    "Q1": ("online", 256, 128, 8),
    "Q2": ("online", 2048, 128, 4),
}
EPOCH_ARMS = ("baseline", "candidate", "candidate", "baseline")
MEASURED_REPETITIONS = 5
WARMUP_REPETITIONS = 2
STATE_CHECKPOINTS = (
    "pre_migration",
    "post_migration",
    "token_1",
    "token_4",
    "token_8",
    "token_32",
    "token_128",
)
RANKS = (0, 1, 2, 3)
WORKER_SCHEMA = "qwen38.topology-local-tp2-whole-model-worker.v1"


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _atomic_write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
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


def _atomic_write_jsonl(path: Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ) + "\n")
        handle.flush()
    temporary.replace(path)


def _default_engine_factory(model_root: Path, **kwargs):
    from tinyvllm.engine.llm_engine import LLMEngine

    return LLMEngine(str(model_root), **kwargs)


def _default_sampling_params_factory(**kwargs):
    from tinyvllm.sampling_params import SamplingParams

    return SamplingParams(**kwargs)


def build_request_specs(
    prompt_tokens: int,
    output_tokens: int,
    concurrency: int,
    seed_namespace: str,
) -> tuple[dict, ...]:
    """Build deterministic prompts whose namespace fixes their identity."""
    prompt_tokens = _positive_integer(prompt_tokens, "prompt_tokens")
    output_tokens = _positive_integer(output_tokens, "output_tokens")
    concurrency = _positive_integer(concurrency, "concurrency")
    if not isinstance(seed_namespace, str) or not seed_namespace:
        raise ValueError("seed_namespace must be a non-empty string")

    rows = []
    for request_index in range(concurrency):
        prompt = []
        block_index = 0
        while len(prompt) < prompt_tokens:
            digest = sha256(
                (
                    f"{seed_namespace}\0{request_index}\0"
                    f"{block_index}"
                ).encode("utf-8")
            ).digest()
            prompt.extend(
                11 + int.from_bytes(digest[offset:offset + 2], "big") % 2000
                for offset in range(0, len(digest), 2)
            )
            block_index += 1
        rows.append({
            "request_id": (
                f"{seed_namespace.replace('/', '-')}-{request_index}"
            ),
            "prompt_token_ids": prompt[:prompt_tokens],
            "output_tokens": output_tokens,
        })
    return tuple(rows)


def _request_set_digest(request_specs) -> str:
    payload = json.dumps(
        list(request_specs),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def reconstruct_request_metrics(
    *,
    admitted_ns: int,
    token_timestamps_ns: tuple[int, ...] | list[int],
) -> dict:
    if isinstance(admitted_ns, bool) or not isinstance(admitted_ns, int):
        raise ValueError("admitted_ns must be an integer")
    timestamps = tuple(token_timestamps_ns)
    if len(timestamps) != 128:
        raise ValueError("exactly 128 token timestamps are required")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in timestamps
    ):
        raise ValueError("token timestamps must be integers")
    if any(current < previous for previous, current in zip(
        timestamps,
        timestamps[1:],
    )):
        raise ValueError("token timestamps must be monotonic")
    token_gaps_ns = [
        current - previous
        for previous, current in zip(timestamps, timestamps[1:])
    ]
    return {
        "ttft_ns": timestamps[0] - admitted_ns,
        "tpot_ns": (timestamps[-1] - timestamps[0]) / 127,
        "token_gaps_ns": token_gaps_ns,
        "e2e_ns": timestamps[-1] - admitted_ns,
    }


def _nearest_rank_percentile(values, percentile: int) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile inputs must not be empty")
    rank = max(
        1,
        math.ceil((float(percentile) / 100.0) * len(ordered)),
    )
    return ordered[rank - 1]


def _service_metrics(requests) -> dict:
    requests = list(requests)
    if not requests:
        raise RuntimeError("service-control request inventory is empty")
    timing_bounds = []
    ttfts = []
    tpots = []
    output_tokens = 0
    for request in requests:
        reconstructed = reconstruct_request_metrics(
            admitted_ns=request.get("admitted_ns"),
            token_timestamps_ns=request.get("token_timestamps_ns"),
        )
        if (
            request.get("completion_ns")
            != request["token_timestamps_ns"][-1]
            or request.get("token_gaps_ns")
            != reconstructed["token_gaps_ns"]
            or any(
                not math.isclose(
                    float(request.get(field, math.inf)),
                    float(reconstructed[field]),
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
                for field in ("ttft_ns", "tpot_ns", "e2e_ns")
            )
        ):
            raise RuntimeError(
                "service-control request timing evidence is invalid"
            )
        tokens = request.get("output_token_ids")
        if not isinstance(tokens, list) or len(tokens) != 128:
            raise RuntimeError(
                "service-control output token inventory is invalid"
            )
        timing_bounds.append((
            int(request["admitted_ns"]),
            int(request["completion_ns"]),
        ))
        ttfts.append(reconstructed["ttft_ns"])
        tpots.append(reconstructed["tpot_ns"])
        output_tokens += len(tokens)
    makespan_ns = (
        max(completion for _, completion in timing_bounds)
        - min(admitted for admitted, _ in timing_bounds)
    )
    if makespan_ns <= 0:
        raise RuntimeError("service-control makespan is invalid")
    return {
        "request_count": len(requests),
        "output_token_count": output_tokens,
        "cohort_makespan_ns": makespan_ns,
        "request_qps": len(requests) * 1e9 / makespan_ns,
        "output_tokens_per_second": output_tokens * 1e9 / makespan_ns,
        "ttft_ns": {
            "p50": _nearest_rank_percentile(ttfts, 50),
            "p95": _nearest_rank_percentile(ttfts, 95),
            "p99": _nearest_rank_percentile(ttfts, 99),
        },
        "tpot_ns": {
            "p50": _nearest_rank_percentile(tpots, 50),
            "p95": _nearest_rank_percentile(tpots, 95),
            "p99": _nearest_rank_percentile(tpots, 99),
        },
    }


def _service_peak_memory(memory, pair_devices) -> list[dict]:
    by_rank = {}
    for row in memory:
        rank = row.get("rank")
        values = tuple(
            row.get(field)
            for field in (
                "cuda_peak_allocated_bytes",
                "cuda_peak_reserved_bytes",
                "physical_memory_bytes",
            )
        )
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank in by_rank
            or rank not in (0, 1)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                for value in values
            )
            or values[0] < 0
            or values[1] < values[0]
            or values[2] <= 0
            or values[1] > values[2]
        ):
            raise RuntimeError(
                "service-control memory evidence is invalid"
            )
        by_rank[rank] = row
    if set(by_rank) != {0, 1}:
        raise RuntimeError(
            "service-control memory inventory is incomplete"
        )
    return [{
        "gpu_index": int(pair_devices[rank]),
        "peak_allocated_bytes": int(
            by_rank[rank]["cuda_peak_allocated_bytes"]
        ),
        "peak_reserved_bytes": int(
            by_rank[rank]["cuda_peak_reserved_bytes"]
        ),
        "physical_memory_bytes": int(
            by_rank[rank]["physical_memory_bytes"]
        ),
    } for rank in (0, 1)]


def _ranked_snapshots(rows, label: str) -> dict[int, dict]:
    if not isinstance(rows, (tuple, list)):
        raise ValueError(f"{label} rank snapshots are invalid")
    ranked = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label} rank snapshots are invalid")
        rank = row.get("rank")
        if rank in ranked:
            raise ValueError(f"{label} rank snapshot is duplicated")
        ranked[rank] = row
    if tuple(sorted(ranked)) != RANKS:
        raise ValueError(f"{label} rank inventory mismatch")
    return ranked


def _expected_cohort(request_rows: list[dict]) -> tuple[tuple[int, int, int], ...]:
    cohort = []
    for index, row in enumerate(request_rows):
        cohort.append((
            int(row.get("slot_id", index)),
            int(row.get("generation", 1)),
            int(row.get("runtime_request_id", row.get("seq_id", index))),
        ))
    return tuple(cohort)


def _validate_cleanup(cleanup: dict) -> dict:
    if (
        not isinstance(cleanup, dict)
        or cleanup.get("process_group_destroyed") is not True
        or cleanup.get("rank_exit_codes") != [0, 0, 0, 0]
        or cleanup.get("owned_children_remaining") != []
    ):
        raise RuntimeError("cleanup evidence is incomplete")
    receipts = cleanup.get("rank_cleanup_receipts")
    if not isinstance(receipts, list) or len(receipts) != 4:
        raise RuntimeError("cleanup rank receipt inventory is incomplete")
    if sorted(row.get("rank") for row in receipts) != list(RANKS):
        raise RuntimeError("cleanup rank receipt inventory is incomplete")
    for row in receipts:
        candidate = row.get("qwen38_topology_local_tp2_cleanup")
        if (
            row.get("process_group_destroyed") is not True
            or not isinstance(candidate, dict)
            or candidate.get("pair_groups_destroyed") != 2
            or candidate.get("candidate_state_released") is not True
            or candidate.get("published_generations_remaining") != 0
            or candidate.get("temporary_live_tensors") != 0
        ):
            raise RuntimeError("cleanup receipt is incomplete")
    return dict(cleanup)


def validate_candidate_evidence(
    *,
    request_rows,
    scheduler_step_rows,
    token_count_rows,
    collective_rows,
    model_manifest,
    before_snapshots,
    after_snapshots,
    cleanup,
) -> dict:
    """Re-derive candidate mechanism totals from raw evidence."""
    if not isinstance(request_rows, list) or not request_rows:
        raise ValueError("request evidence is missing")
    request_ids = []
    for row in request_rows:
        output = row.get("output_token_ids")
        if not isinstance(output, list) or len(output) != 128:
            raise ValueError("output token count mismatch")
        if row.get("rank_token_agreement") is not True:
            raise ValueError("output token rank agreement is missing")
        if row.get("finite_logits") is not True:
            raise ValueError("finite logits evidence is missing")
        if (
            row.get("stop_position") != 128
            or row.get("stop_reason") != "length"
        ):
            raise ValueError("output token stop evidence is invalid")
        request_ids.append(row.get("request_id"))
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request identity is duplicated")

    layer_types = model_manifest.get("layer_types")
    if (
        model_manifest.get("num_hidden_layers") != 64
        or not isinstance(layer_types, list)
        or len(layer_types) != 64
    ):
        raise ValueError("model layer manifest mismatch")
    linear_indices = tuple(
        index
        for index, layer_type in enumerate(layer_types)
        if layer_type == "linear_attention"
    )
    full_indices = tuple(
        index
        for index, layer_type in enumerate(layer_types)
        if layer_type == "full_attention"
    )
    if len(linear_indices) != 48 or len(full_indices) != 16:
        raise ValueError("model layer role inventory mismatch")

    decode_steps = {}
    for row in scheduler_step_rows:
        if row.get("is_prefill") is True:
            continue
        if (
            row.get("batch_kind") != "decode"
            or row.get("request_ids") != request_ids
        ):
            raise ValueError("fixed candidate cohort changed")
        decode_steps[row.get("step_index")] = row
    if len(decode_steps) != 127:
        raise ValueError("decode scheduler step inventory mismatch")

    token_counts = {}
    for row in token_count_rows:
        key = (row.get("step_index"), row.get("request_id"))
        if key in token_counts:
            raise ValueError("duplicate token-count row")
        token_counts[key] = row.get("token_count")
    expected_token_keys = {
        (step_index, request_id)
        for step_index in decode_steps
        for request_id in request_ids
    }
    if set(token_counts) != expected_token_keys or any(
        count != 1 for count in token_counts.values()
    ):
        raise ValueError("token-count evidence mismatch")
    expected_segments = sum(token_counts.values())

    before = _ranked_snapshots(before_snapshots, "before")
    after = _ranked_snapshots(after_snapshots, "after")
    expected_cohort = _expected_cohort(request_rows)
    authoritative_totals = None
    migration_publications = None
    for rank in RANKS:
        before_row = before[rank]
        after_row = after[rank]
        if (
            after_row.get("enabled") is not True
            or tuple(after_row.get("linear_layer_indices", ()))
            != linear_indices
        ):
            raise ValueError("candidate layer inventory mismatch")
        if tuple(after_row.get("fixed_cohort") or ()) != expected_cohort:
            raise ValueError("candidate cohort mismatch")
        if (
            int(after_row.get("transition_count", -1))
            - int(before_row.get("transition_count", -1))
            != 1
        ):
            raise ValueError("candidate transition count mismatch")
        before_state = before_row.get("state", {})
        after_state = after_row.get("state", {})
        publications = (
            int(after_state.get("publication_count", -1))
            - int(before_state.get("publication_count", -1))
        )
        if publications != len(request_rows):
            raise ValueError("candidate migration publication mismatch")
        if after_state.get("temporary_live_tensors") != 0:
            raise RuntimeError("candidate migration temporary was retained")
        if after_state.get("rollback_count", 0) != 0:
            raise RuntimeError("candidate migration rollback was observed")

        mixers = after_row.get("mixers")
        if not isinstance(mixers, (tuple, list)) or len(mixers) != 48:
            raise ValueError("candidate mixer layer inventory mismatch")
        totals = {
            "tp2_decode_calls": sum(
                int(row.get("tp2_decode_calls", -1)) for row in mixers
            ),
            "recurrent_token_one_calls": sum(
                int(row.get("recurrent_token_one_calls", -1))
                for row in mixers
            ),
            "short_chunk_calls": sum(
                int(row.get("short_chunk_calls", -1)) for row in mixers
            ),
            "ordinary_chunk_calls": sum(
                int(row.get("ordinary_chunk_calls",
                            row.get("chunk_64_calls", -1)))
                for row in mixers
            ),
            "global_tp4_linear_decode_all_reduce_calls": sum(
                int(row.get(
                    "global_tp4_decode_all_reduce_calls",
                    -1,
                ))
                for row in mixers
            ),
            "pair_local_all_reduce_calls": sum(
                int(row.get("pair_local_all_reduce_calls", -1))
                for row in mixers
            ),
        }
        expected_linear_calls = expected_segments * 48
        if (
            totals["tp2_decode_calls"] != expected_linear_calls
            or totals["recurrent_token_one_calls"]
            != expected_linear_calls
            or totals["pair_local_all_reduce_calls"]
            != expected_linear_calls
            or totals["short_chunk_calls"] != 0
            or totals["ordinary_chunk_calls"] != 0
            or totals[
                "global_tp4_linear_decode_all_reduce_calls"
            ] != 0
        ):
            raise RuntimeError("candidate mixer call evidence mismatch")
        zero_fields = (
            "fallback_calls",
            "post_warmup_request_path_allocations",
            "prefix_restore_calls",
            "prefix_publication_calls",
            "retry_after_mutation_calls",
            "duplicate_commit_calls",
            "pair_replica_comparison_failures",
        )
        for field in zero_fields:
            before_value = before_row.get(field)
            after_value = after_row.get(field)
            if (
                isinstance(before_value, bool)
                or not isinstance(before_value, int)
                or isinstance(after_value, bool)
                or not isinstance(after_value, int)
                or after_value - before_value != 0
            ):
                message = (
                    "duplicate commit evidence is nonzero"
                    if field == "duplicate_commit_calls"
                    else f"{field} evidence is nonzero"
                )
                raise RuntimeError(message)
        if authoritative_totals is None:
            authoritative_totals = totals
            migration_publications = publications
        elif totals != authoritative_totals:
            raise RuntimeError("rank mixer totals disagree")

    collectives = {}
    for row in collective_rows:
        key = (
            row.get("rank"),
            row.get("step_index"),
            row.get("layer_index"),
        )
        if key in collectives:
            raise ValueError("collective evidence is duplicated")
        collectives[key] = row
    expected_collective_keys = {
        (rank, step_index, layer_index)
        for rank in RANKS
        for step_index in decode_steps
        for layer_index in range(64)
    }
    if set(collectives) != expected_collective_keys:
        raise ValueError("collective rank or layer inventory mismatch")

    hidden_size = int(model_manifest.get("hidden_size", 0))
    pair_dtype_bytes = int(
        model_manifest.get("pair_local_accumulation_dtype_bytes", 0)
    )
    full_dtype_bytes = int(
        model_manifest.get("full_attention_collective_dtype_bytes", 0)
    )
    pair_sequence_match = True
    for step_index in decode_steps:
        for layer_index, layer_type in enumerate(layer_types):
            rows = [
                collectives[(rank, step_index, layer_index)]
                for rank in RANKS
            ]
            expected_scope = (
                "pair_local"
                if layer_type == "linear_attention"
                else "global_tp4"
            )
            dtype_bytes = (
                pair_dtype_bytes
                if layer_type == "linear_attention"
                else full_dtype_bytes
            )
            for row in rows:
                if (
                    row.get("layer_role") != layer_type
                    or row.get("scope") != expected_scope
                    or row.get("calls") != len(request_rows)
                    or row.get("bytes")
                    != len(request_rows) * hidden_size * dtype_bytes
                ):
                    raise RuntimeError(
                        "collective call or byte evidence mismatch"
                    )
            if layer_type == "linear_attention":
                sequences = {
                    tuple(row.get("sequence", ())) for row in rows
                }
                if len(sequences) != 1:
                    pair_sequence_match = False
    if not pair_sequence_match:
        raise RuntimeError("pair-local collective sequence mismatch")

    _validate_cleanup(cleanup)
    result = dict(authoritative_totals or {})
    result.update({
        "expected_segments": expected_segments,
        "migration_publications": migration_publications,
        "full_attention_layer_count": len(full_indices),
        "full_attention_tp4_collective_calls": (
            expected_segments * len(full_indices)
        ),
        "pair_local_collective_sequence_match": True,
        "fallback_calls": 0,
        "post_warmup_request_path_allocations": 0,
        "prefix_restore_calls": 0,
        "prefix_publication_calls": 0,
        "retry_after_mutation_calls": 0,
        "duplicate_commit_calls": 0,
    })
    return result


def run_engine_case(
    *,
    model_root: Path,
    arm: str,
    workload_id: str,
    request_specs: tuple[dict, ...],
    warmup: bool,
    epoch: int,
    repetition: int,
    engine=None,
    close_engine: bool = True,
    tensor_parallel_size: int = 4,
    engine_factory: Callable = _default_engine_factory,
    sampling_params_factory: Callable = _default_sampling_params_factory,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    timeout_s: float = 120.0,
    correctness_authority: bool = False,
) -> dict:
    if arm not in {"baseline", "candidate"}:
        raise ValueError("arm must be baseline or candidate")
    if workload_id not in WORKLOADS:
        raise ValueError("unknown workload")
    if not request_specs:
        raise ValueError("request_specs must not be empty")
    if (
        isinstance(tensor_parallel_size, bool)
        or tensor_parallel_size not in (2, 4)
        or (arm == "candidate" and tensor_parallel_size != 4)
    ):
        raise ValueError("engine tensor-parallel size is invalid")
    prompt_tokens = len(request_specs[0]["prompt_token_ids"])
    output_tokens = int(request_specs[0]["output_tokens"])
    concurrency = len(request_specs)
    if (
        output_tokens != 128
        or any(
            len(row.get("prompt_token_ids", ())) != prompt_tokens
            or row.get("output_tokens") != output_tokens
            for row in request_specs
        )
    ):
        raise ValueError("request shape mismatch")

    if engine is None:
        engine = engine_factory(
            Path(model_root),
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=True,
            max_num_seqs=max(8, concurrency),
            max_model_len=prompt_tokens + output_tokens,
            max_num_batched_tokens=prompt_tokens * concurrency,
            qwen38_topology_local_tp2_islands=(arm == "candidate"),
        )
    before_snapshots = ()
    cleanup = None
    correctness_step_proofs = []
    correctness_state_checkpoints = {}
    try:
        if (
            getattr(getattr(engine, "model_runner", None), "rank", None)
            != 0
            or getattr(
                getattr(engine, "model_runner", None),
                "world_size",
                None,
            ) != tensor_parallel_size
        ):
            raise RuntimeError("engine tensor-parallel ownership mismatch")
        flush_releases = getattr(
            engine,
            "flush_pending_hybrid_state_releases",
            None,
        )
        if flush_releases is not None:
            flush_releases(timeout_s=float(timeout_s))
        if arm == "candidate":
            before_snapshots = (
                engine.qwen38_topology_local_tp2_snapshots(
                    timeout_s=float(timeout_s)
                )
            )
        if correctness_authority:
            receipt = engine.enable_qwen38_correctness_proof(
                True,
                timeout_s=float(timeout_s),
            )
            if receipt.get("enabled") is not True:
                raise RuntimeError(
                    "correctness proof recording was not enabled"
                )

        lifecycle = {}
        for request in request_specs:
            admitted_ns = clock_ns()
            sampling = sampling_params_factory(
                temperature=0.0,
                max_tokens=output_tokens,
                ignore_eos=True,
            )
            seq_id = engine.add_request(
                request["prompt_token_ids"],
                sampling,
            )
            if (
                isinstance(seq_id, bool)
                or not isinstance(seq_id, int)
                or seq_id in lifecycle
            ):
                raise RuntimeError(
                    "engine request identity is invalid"
                )
            lifecycle[seq_id] = {
                "request_id": request["request_id"],
                "runtime_request_id": seq_id,
                "admitted_ns": admitted_ns,
                "first_scheduled_ns": None,
                "token_timestamps_ns": [],
                "output_token_ids": [],
                "complete": False,
            }

        scheduler_step_rows = []
        token_count_rows = []
        step_index = 0
        while not engine.is_finished():
            outputs, _ = engine.step()
            observation = engine.last_step_observation
            if not isinstance(observation, dict):
                raise RuntimeError("step observation is missing")
            step_end_ns = observation.get("step_end_ns")
            step_start_ns = observation.get("step_start_ns")
            token_deltas = observation.get(
                "new_completion_tokens_by_seq"
            )
            timeline = observation.get("command_timeline_step")
            phases = (
                timeline.get("phases")
                if isinstance(timeline, Mapping)
                else None
            )
            dispatch_phase = (
                phases.get("ordinary_or_first_target_dispatch")
                if isinstance(phases, Mapping)
                else None
            )
            host_submission_ns = (
                dispatch_phase.get("duration_ns")
                if isinstance(dispatch_phase, Mapping)
                else None
            )
            if (
                isinstance(step_end_ns, bool)
                or not isinstance(step_end_ns, int)
                or isinstance(step_start_ns, bool)
                or not isinstance(step_start_ns, int)
                or step_start_ns < 0
                or step_end_ns < step_start_ns
                or isinstance(host_submission_ns, bool)
                or not isinstance(host_submission_ns, int)
                or host_submission_ns < 0
                or not isinstance(token_deltas, dict)
            ):
                raise RuntimeError("step timing observation is invalid")
            if correctness_authority:
                proofs = engine.qwen38_correctness_step_proofs(
                    timeout_s=float(timeout_s)
                )
                proof_sequence_ids = tuple(
                    proofs[0].get("sequence_ids", ())
                ) if proofs else ()
                expected_token_ids = tuple(
                    token_deltas.get(sequence_id, ())
                    for sequence_id in proof_sequence_ids
                )
                if (
                    len(proofs) != tensor_parallel_size
                    or [row.get("rank") for row in proofs]
                    != list(range(tensor_parallel_size))
                    or any(
                        row.get("finite_logits") is not True
                        for row in proofs
                    )
                    or len({
                        tuple(row.get("token_ids", ()))
                        for row in proofs
                    }) != 1
                    or any(
                        tuple(row.get("sequence_ids", ()))
                        != proof_sequence_ids
                        for row in proofs
                    )
                    or any(len(tokens) != 1 for tokens in expected_token_ids)
                    or tuple(proofs[0].get("token_ids", ()))
                    != tuple(tokens[0] for tokens in expected_token_ids)
                ):
                    raise RuntimeError(
                        "rank-local correctness proof mismatch"
                    )
                correctness_step_proofs.append(list(proofs))
            scheduled = observation.get("scheduled", [])
            scheduled_ids = [
                lifecycle[int(row["seq_id"])]["request_id"]
                for row in scheduled
            ]
            for row in scheduled:
                lifecycle_row = lifecycle[int(row["seq_id"])]
                if lifecycle_row["first_scheduled_ns"] is None:
                    lifecycle_row["first_scheduled_ns"] = step_start_ns
            scheduler_step_rows.append({
                "step_index": step_index,
                "is_prefill": observation.get("is_prefill"),
                "batch_kind": observation.get("batch_kind"),
                "request_ids": scheduled_ids,
                "step_start_ns": step_start_ns,
                "step_end_ns": step_end_ns,
                "step_duration_ns": step_end_ns - step_start_ns,
                "host_submission_ns": host_submission_ns,
            })
            for raw_seq_id, tokens in token_deltas.items():
                seq_id = int(raw_seq_id)
                row = lifecycle.get(seq_id)
                if (
                    row is None
                    or not isinstance(tokens, list)
                    or any(
                        isinstance(token, bool)
                        or not isinstance(token, int)
                        or token < 0
                        for token in tokens
                    )
                ):
                    raise RuntimeError(
                        "token delta observation is invalid"
                    )
                if tokens:
                    row["token_timestamps_ns"].extend(
                        [step_end_ns] * len(tokens)
                    )
                    row["output_token_ids"].extend(tokens)
                    token_count_rows.append({
                        "step_index": step_index,
                        "request_id": row["request_id"],
                        "token_count": len(tokens),
                    })
            for raw_seq_id, output in outputs:
                seq_id = int(raw_seq_id)
                row = lifecycle.get(seq_id)
                if row is None or list(output) != row["output_token_ids"]:
                    raise RuntimeError("terminal token output mismatch")
                row["complete"] = True
            if correctness_authority:
                completed_tokens = {
                    len(row["output_token_ids"])
                    for row in lifecycle.values()
                }
                if len(completed_tokens) == 1:
                    token_count = completed_tokens.pop()
                    checkpoint_names = {
                        1: ("pre_migration", "token_1"),
                        2: ("post_migration",),
                        4: ("token_4",),
                        8: ("token_8",),
                        32: ("token_32",),
                        128: ("token_128",),
                    }.get(token_count, ())
                    if checkpoint_names:
                        snapshot = (
                            engine.qwen38_correctness_state_checkpoints(
                                timeout_s=float(timeout_s)
                            )
                        )
                        for checkpoint_name in checkpoint_names:
                            correctness_state_checkpoints[
                                checkpoint_name
                            ] = snapshot
            step_index += 1

        after_snapshots = ()
        if arm == "candidate":
            after_snapshots = (
                engine.qwen38_topology_local_tp2_snapshots(
                    timeout_s=float(timeout_s)
                )
            )
        memory = engine.memory_snapshots(timeout_s=float(timeout_s))
        requests = []
        for seq_id in lifecycle:
            row = lifecycle[seq_id]
            if (
                row["complete"] is not True
                or len(row["output_token_ids"]) != output_tokens
            ):
                raise RuntimeError(
                    "request completion evidence is incomplete"
                )
            metrics = reconstruct_request_metrics(
                admitted_ns=row["admitted_ns"],
                token_timestamps_ns=row["token_timestamps_ns"],
            )
            tokenizer = getattr(engine, "tokenizer", None)
            decode = getattr(tokenizer, "decode", None)
            if not callable(decode):
                raise RuntimeError(
                    "engine tokenizer decode is unavailable"
                )
            decoded_text = decode(
                row["output_token_ids"],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if not isinstance(decoded_text, str):
                raise RuntimeError("decoded output text is invalid")
            requests.append({
                **row,
                **metrics,
                "queueing_ns": (
                    row["first_scheduled_ns"] - row["admitted_ns"]
                ),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": output_tokens,
                "completion_ns": row["token_timestamps_ns"][-1],
                "rank_token_agreement": (
                    True if correctness_authority else None
                ),
                "finite_logits": (
                    True if correctness_authority else None
                ),
                "stop_position": output_tokens,
                "stop_reason": "length",
                "decoded_text": decoded_text,
                "decoded_text_sha256": sha256(
                    decoded_text.encode("utf-8")
                ).hexdigest(),
            })
    finally:
        if correctness_authority:
            engine.enable_qwen38_correctness_proof(
                False,
                timeout_s=float(timeout_s),
            )
        if close_engine:
            cleanup_started_ns = clock_ns()
            cleanup = dict(engine.exit())
            cleanup_finished_ns = clock_ns()
            cleanup.update({
                "cleanup_started_ns": cleanup_started_ns,
                "cleanup_finished_ns": cleanup_finished_ns,
                "cleanup_duration_ns": (
                    cleanup_finished_ns - cleanup_started_ns
                ),
            })

    return {
        "schema_version": WORKER_SCHEMA,
        "arm": arm,
        "workload_id": workload_id,
        "warmup": bool(warmup),
        "epoch": int(epoch),
        "repetition": int(repetition),
        "requests": requests,
        "scheduler_step_rows": scheduler_step_rows,
        "token_count_rows": token_count_rows,
        "before_snapshots": before_snapshots,
        "after_snapshots": after_snapshots,
        "memory": tuple(memory),
        "cleanup": cleanup,
        "timing_authority": not warmup,
        "request_set_digest": _request_set_digest(request_specs),
        "cohort_makespan_ns": (
            max(row["completion_ns"] for row in requests)
            - min(row["admitted_ns"] for row in requests)
        ),
        "correctness_step_proofs": correctness_step_proofs,
        "correctness_state_checkpoints": (
            correctness_state_checkpoints
        ),
    }


def run_service_replica_case(
    *,
    model_root: Path,
    workload_id: str,
    request_specs: tuple[dict, ...],
    engine_factory: Callable = _default_engine_factory,
    sampling_params_factory: Callable = _default_sampling_params_factory,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    timeout_s: float = 120.0,
) -> dict:
    row = run_engine_case(
        model_root=Path(model_root),
        arm="baseline",
        workload_id=workload_id,
        request_specs=request_specs,
        warmup=False,
        epoch=-2,
        repetition=0,
        tensor_parallel_size=2,
        engine_factory=engine_factory,
        sampling_params_factory=sampling_params_factory,
        clock_ns=clock_ns,
        timeout_s=timeout_s,
    )
    row["replica_tensor_parallel_size"] = 2
    row["classification_authority"] = False
    return row


def _run_service_replicas_in_subprocesses(
    *,
    model_root: Path,
    output_root: Path,
    pair_devices: tuple[tuple[int, int], tuple[int, int]],
    workload_id: str,
    request_specs_by_replica: tuple[tuple[dict, ...], tuple[dict, ...]],
    shared_start_ns: int,
    popen_factory: Callable = subprocess.Popen,
    killpg: Callable[[int, int], None] = os.killpg,
    timeout_s: float = 1800.0,
) -> tuple[dict, dict]:
    del shared_start_ns
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".service-{workload_id}.",
        dir=output_root,
    ) as temporary_root:
        temporary = Path(temporary_root)
        barrier = temporary / "start"
        processes = []
        outputs = []
        for replica_index, (devices, request_specs) in enumerate(zip(
            pair_devices,
            request_specs_by_replica,
        )):
            input_path = temporary / f"requests-{replica_index}.json"
            output_path = temporary / f"result-{replica_index}.json"
            _atomic_write_json(input_path, list(request_specs))
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(device) for device in devices
            )
            process = popen_factory(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "service-replica",
                    "--model-root",
                    str(model_root),
                    "--workload-id",
                    workload_id,
                    "--request-specs-path",
                    str(input_path),
                    "--output-path",
                    str(output_path),
                    "--start-barrier",
                    str(barrier),
                ],
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            processes.append(process)
            outputs.append(output_path)
        barrier.write_text("start\n", encoding="utf-8")
        failures = []
        try:
            for replica_index, process in enumerate(processes):
                stdout, stderr = process.communicate(timeout=timeout_s)
                if process.returncode != 0:
                    failures.append(
                        f"replica {replica_index}: "
                        f"{stderr or stdout or process.returncode}"
                    )
        except BaseException:
            for process in processes:
                if process.poll() is None:
                    try:
                        killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for process in processes:
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    try:
                        killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5.0)
            raise
        if failures:
            raise RuntimeError(
                "service-control replica failed: " + "; ".join(failures)
            )
        return tuple(
            json.loads(path.read_text(encoding="utf-8"))
            for path in outputs
        )


def run_performance_epoch(
    *,
    model_root: Path,
    output_root: Path,
    epoch: int,
    arm: str,
    workload_order: tuple[str, ...],
    case_runner: Callable = run_engine_case,
    engine_factory: Callable | None = None,
    row_sink: Callable[[dict], None] | None = None,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> dict:
    workload_order = tuple(workload_order)
    if (
        not workload_order
        or len(set(workload_order)) != len(workload_order)
        or any(workload_id not in WORKLOADS for workload_id in workload_order)
    ):
        raise ValueError("workload order is not frozen")
    forward_positions = [tuple(WORKLOADS).index(item) for item in workload_order]
    reverse_positions = [
        tuple(reversed(WORKLOADS)).index(item)
        for item in workload_order
    ]
    if (
        forward_positions != sorted(forward_positions)
        and reverse_positions != sorted(reverse_positions)
    ):
        raise ValueError("workload order is not frozen")
    if EPOCH_ARMS[int(epoch)] != arm:
        raise ValueError("epoch arm mismatch")
    engine = None
    cleanup = None
    startup_model_load_started_ns = None
    startup_model_load_finished_ns = None
    if engine_factory is not None:
        startup_model_load_started_ns = clock_ns()
        engine = engine_factory(
            Path(model_root),
            tensor_parallel_size=4,
            enforce_eager=True,
            max_num_seqs=8,
            max_model_len=2176,
            max_num_batched_tokens=8192,
            qwen38_topology_local_tp2_islands=(arm == "candidate"),
        )
        startup_model_load_finished_ns = clock_ns()
    rows = []
    measured_rows = []
    try:
        for workload_id in workload_order:
            _, prompt_tokens, output_tokens, concurrency = WORKLOADS[
                workload_id
            ]
            for repetition in range(
                WARMUP_REPETITIONS + MEASURED_REPETITIONS
            ):
                warmup = repetition < WARMUP_REPETITIONS
                measured_repetition = (
                    repetition
                    if warmup
                    else repetition - WARMUP_REPETITIONS
                )
                request_specs = build_request_specs(
                    prompt_tokens,
                    output_tokens,
                    concurrency,
                    (
                        f"timing/{workload_id}/"
                        f"r{measured_repetition}"
                    ),
                )
                case_kwargs = {
                    "model_root": Path(model_root),
                    "arm": arm,
                    "workload_id": workload_id,
                    "request_specs": request_specs,
                    "warmup": warmup,
                    "epoch": int(epoch),
                    "repetition": measured_repetition,
                }
                if engine is not None:
                    case_kwargs.update({
                        "engine": engine,
                        "close_engine": False,
                    })
                row = case_runner(**case_kwargs)
                rows.append(row)
                if not warmup:
                    measured_rows.append((row, request_specs))
        for measured_row, request_specs in measured_rows:
            replay_kwargs = {
                "model_root": Path(model_root),
                "arm": arm,
                "workload_id": measured_row["workload_id"],
                "request_specs": request_specs,
                "warmup": True,
                "epoch": int(epoch),
                "repetition": int(measured_row["repetition"]),
                "correctness_authority": True,
            }
            if engine is not None:
                replay_kwargs.update({
                    "engine": engine,
                    "close_engine": False,
                })
            measured_row["timing_correctness_replay"] = case_runner(
                **replay_kwargs
            )
        if row_sink is not None:
            for row in rows:
                row_sink(row)
    finally:
        if engine is not None:
            cleanup_started_ns = clock_ns()
            cleanup = dict(engine.exit())
            cleanup_finished_ns = clock_ns()
            cleanup.update({
                "cleanup_started_ns": cleanup_started_ns,
                "cleanup_finished_ns": cleanup_finished_ns,
                "cleanup_duration_ns": (
                    cleanup_finished_ns - cleanup_started_ns
                ),
            })
    result = {
        "schema_version": WORKER_SCHEMA,
        "phase": "performance_epoch",
        "epoch": int(epoch),
        "arm": arm,
        "workload_order": list(workload_order),
        "startup_model_load_started_ns": startup_model_load_started_ns,
        "startup_model_load_finished_ns": startup_model_load_finished_ns,
        "startup_model_load_duration_ns": (
            startup_model_load_finished_ns - startup_model_load_started_ns
            if startup_model_load_started_ns is not None
            and startup_model_load_finished_ns is not None
            else None
        ),
        "rows": rows,
        "cleanup": cleanup,
    }
    if row_sink is None:
        _atomic_write_json(
            Path(output_root) / f"epoch-{epoch}.json",
            result,
        )
    return result


def _counter_delta(before, after, field):
    before_mixers = before.get("mixers")
    after_mixers = after.get("mixers")
    if (
        not isinstance(before_mixers, (tuple, list))
        or not isinstance(after_mixers, (tuple, list))
        or len(before_mixers) != 48
        or len(after_mixers) != 48
    ):
        raise RuntimeError("candidate mixer inventory is invalid")
    values = []
    for label, mixers in (
        ("before", before_mixers),
        ("after", after_mixers),
    ):
        counters = []
        for row in mixers:
            value = row.get(field) if isinstance(row, Mapping) else None
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise RuntimeError(
                    f"candidate mixer counter is invalid: {field} "
                    f"({label})"
                )
            counters.append(value)
        values.append(sum(counters))
    if values[1] < values[0]:
        raise RuntimeError(
            f"candidate mixer counter regressed: {field}"
        )
    return values[1] - values[0]


def _runtime_counter_delta(before, after, field):
    before_value = before.get(field)
    after_value = after.get(field)
    if (
        isinstance(before_value, bool)
        or not isinstance(before_value, int)
        or isinstance(after_value, bool)
        or not isinstance(after_value, int)
        or after_value < before_value
    ):
        raise RuntimeError(
            f"candidate runtime counter is invalid: {field}"
        )
    return after_value - before_value


def build_performance_artifact_rows(
    epoch_result: Mapping[str, object],
    *,
    source_revision: str,
    model_revision: str,
) -> dict[str, list[dict]]:
    epoch = int(epoch_result["epoch"])
    arm = str(epoch_result["arm"])
    identity = {
        "source_revision": source_revision,
        "model_revision": model_revision,
    }
    artifacts = {
        "request_rows.jsonl": [],
        "scheduler_step_rows.jsonl": [],
        "candidate_hit_rows.jsonl": [],
        "collective_rows.jsonl": [],
        "migration_rows.jsonl": [],
        "memory_rows.jsonl": [],
    }
    memory_by_rank = {}
    for case in epoch_result.get("rows", ()):
        if case.get("warmup") is True:
            continue
        workload_id = case["workload_id"]
        repetition = int(case["repetition"])
        digest = case["request_set_digest"]
        row_identity = {
            **identity,
            "epoch": epoch,
            "arm": arm,
            "workload_id": workload_id,
            "repetition": repetition,
            "request_set_digest": digest,
        }
        replay = case.get("timing_correctness_replay")
        if not isinstance(replay, Mapping):
            raise RuntimeError(
                "timing correctness replay evidence is missing"
            )
        replay_proof = _correctness_proof_summary(replay)
        measured_requests = case.get("requests")
        replay_requests = replay.get("requests")
        if (
            replay.get("timing_authority") is not False
            or replay_proof["rank_token_agreement"] is not True
            or replay_proof["finite_logits"] is not True
            or replay_proof["top_logit_values_match"] is not True
            or not isinstance(measured_requests, list)
            or not isinstance(replay_requests, list)
            or [
                (
                    row.get("request_id"),
                    row.get("output_token_ids"),
                    row.get("stop_position"),
                    row.get("stop_reason"),
                    row.get("decoded_text"),
                    row.get("decoded_text_sha256"),
                )
                for row in measured_requests
            ] != [
                (
                    row.get("request_id"),
                    row.get("output_token_ids"),
                    row.get("stop_position"),
                    row.get("stop_reason"),
                    row.get("decoded_text"),
                    row.get("decoded_text_sha256"),
                )
                for row in replay_requests
            ]
        ):
            raise RuntimeError(
                "timing correctness replay evidence is invalid"
            )
        artifacts["request_rows.jsonl"].append({
            **row_identity,
            "requests": measured_requests,
            "cohort_makespan_ns": case["cohort_makespan_ns"],
            "rank_token_agreement": True,
            "finite_logits": True,
            "top_logit_values_match": True,
            "timing_correctness_replay": {
                "requests": replay_requests,
                "step_proofs": replay.get("correctness_step_proofs"),
            },
        })
        scheduler_steps = [dict(row) for row in case["scheduler_step_rows"]]
        decode_steps = [
            row
            for row in scheduler_steps
            if row.get("is_prefill") is not True
        ]
        token_one_segments = sum(
            int(row["token_count"])
            for row in case["token_count_rows"]
            if any(
                step["step_index"] == row["step_index"]
                for step in decode_steps
            )
        )
        artifacts["scheduler_step_rows.jsonl"].append({
            **row_identity,
            "startup_model_load_started_ns": epoch_result.get(
                "startup_model_load_started_ns"
            ),
            "startup_model_load_finished_ns": epoch_result.get(
                "startup_model_load_finished_ns"
            ),
            "startup_model_load_duration_ns": epoch_result.get(
                "startup_model_load_duration_ns"
            ),
            "steps": scheduler_steps,
            "decode_steps": len(decode_steps),
            "token_one_segments": token_one_segments,
        })
        for memory in case.get("memory", ()):
            rank = int(memory["rank"])
            previous = memory_by_rank.get(rank)
            if previous is None:
                memory_by_rank[rank] = dict(memory)
                continue
            if (
                int(memory["physical_memory_bytes"])
                != int(previous["physical_memory_bytes"])
            ):
                raise RuntimeError(
                    "physical memory changed within one engine epoch"
                )
            previous["cuda_peak_allocated_bytes"] = max(
                int(previous["cuda_peak_allocated_bytes"]),
                int(memory["cuda_peak_allocated_bytes"]),
            )
            previous["cuda_peak_reserved_bytes"] = max(
                int(previous["cuda_peak_reserved_bytes"]),
                int(memory["cuda_peak_reserved_bytes"]),
            )
        if arm != "candidate":
            continue
        before = _ranked_snapshots(
            case["before_snapshots"],
            "before",
        )
        after = _ranked_snapshots(
            case["after_snapshots"],
            "after",
        )
        rank_totals = []
        for rank in RANKS:
            totals = {
                field: _counter_delta(before[rank], after[rank], field)
                for field in (
                    "tp2_decode_calls",
                    "recurrent_token_one_calls",
                    "short_chunk_calls",
                    "chunk_64_calls",
                    "global_tp4_decode_all_reduce_calls",
                    "pair_local_all_reduce_calls",
                )
            }
            totals.update({
                field: _runtime_counter_delta(
                    before[rank],
                    after[rank],
                    field,
                )
                for field in (
                    "fallback_calls",
                    "post_warmup_request_path_allocations",
                    "prefix_restore_calls",
                    "prefix_publication_calls",
                    "retry_after_mutation_calls",
                    "duplicate_commit_calls",
                    "pair_replica_comparison_failures",
                )
            })
            rank_totals.append(totals)
        if any(row != rank_totals[0] for row in rank_totals[1:]):
            raise RuntimeError("candidate rank counters disagree")
        totals = rank_totals[0]
        publication_counts = [
            int(after[rank]["state"]["publication_count"])
            - int(before[rank]["state"]["publication_count"])
            for rank in RANKS
        ]
        if len(set(publication_counts)) != 1:
            raise RuntimeError("candidate publication counts disagree")
        artifacts["candidate_hit_rows.jsonl"].append({
            **row_identity,
            "tp2_decode_calls": totals["tp2_decode_calls"],
            "recurrent_token_one_calls":
                totals["recurrent_token_one_calls"],
            "short_chunk_calls": totals["short_chunk_calls"],
            "ordinary_chunk_calls": totals["chunk_64_calls"],
            "global_tp4_linear_decode_all_reduce_calls":
                totals["global_tp4_decode_all_reduce_calls"],
            "full_attention_tp4_collective_calls":
                token_one_segments * 16,
            "migration_publications": publication_counts[0],
            "fallback_calls": totals["fallback_calls"],
            "post_warmup_request_path_allocations":
                totals["post_warmup_request_path_allocations"],
            "prefix_restore_calls": totals["prefix_restore_calls"],
            "prefix_publication_calls":
                totals["prefix_publication_calls"],
            "retry_after_mutation_calls":
                totals["retry_after_mutation_calls"],
            "duplicate_commit_calls":
                totals["duplicate_commit_calls"],
        })
        artifacts["collective_rows.jsonl"].append({
            **row_identity,
            "pair_local_calls": totals["pair_local_all_reduce_calls"],
            "pair_local_bytes": (
                totals["pair_local_all_reduce_calls"] * 5120 * 4
            ),
            "full_attention_tp4_calls": token_one_segments * 16,
            "full_attention_tp4_bytes":
                token_one_segments * 16 * 5120 * 2,
            "pair_local_sequence_match": all(
                rank_totals[rank][
                    "pair_replica_comparison_failures"
                ] == 0
                for rank in RANKS
            ),
        })
        transition_latencies = [
            after[rank].get("last_transition_latency_ns")
            for rank in RANKS
        ]
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in transition_latencies
        ):
            raise RuntimeError(
                "candidate migration latency evidence is invalid"
            )
        artifacts["migration_rows.jsonl"].append({
            **row_identity,
            "latency_ns": max(transition_latencies),
            "temporary_live_tensors": max(
                int(after[rank]["state"]["temporary_live_tensors"])
                for rank in RANKS
            ),
        })
    for rank in sorted(memory_by_rank):
        memory = memory_by_rank[rank]
        artifacts["memory_rows.jsonl"].append({
            **identity,
            "epoch": epoch,
            "arm": arm,
            "rank": rank,
            "peak_allocated_bytes": int(
                memory["cuda_peak_allocated_bytes"]
            ),
            "peak_reserved_bytes": int(
                memory["cuda_peak_reserved_bytes"]
            ),
            "physical_memory_bytes": int(
                memory["physical_memory_bytes"]
            ),
        })
    return artifacts


def _correctness_proof_summary(
    arm_result: Mapping[str, object],
) -> dict:
    requests = arm_result.get("requests")
    proofs = arm_result.get("correctness_step_proofs")
    if (
        not isinstance(requests, list)
        or not requests
        or not isinstance(proofs, list)
        or len(proofs) != 128
    ):
        return {
            "rank_token_agreement": False,
            "finite_logits": False,
            "top_logit_values_match": False,
            "top_logit_values": (),
        }
    sequence_ids = tuple(
        row.get("runtime_request_id") for row in requests
    )
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in sequence_ids
        )
        or len(set(sequence_ids)) != len(sequence_ids)
    ):
        return {
            "rank_token_agreement": False,
            "finite_logits": False,
            "top_logit_values_match": False,
            "top_logit_values": (),
        }
    emitted = [[] for _ in requests]
    top_values = []
    rank_agreement = True
    finite_logits = True
    top_logit_values_match = True
    for step_rows in proofs:
        if not isinstance(step_rows, list) or len(step_rows) != 4:
            return {
                "rank_token_agreement": False,
                "finite_logits": False,
                "top_logit_values_match": False,
                "top_logit_values": (),
            }
        ranked = sorted(step_rows, key=lambda row: row.get("rank", -1))
        if [row.get("rank") for row in ranked] != list(RANKS):
            rank_agreement = False
        tokens_by_rank = [
            tuple(row.get("token_ids", ())) for row in ranked
        ]
        values_by_rank = [
            tuple(row.get("top_logit_values", ())) for row in ranked
        ]
        if any(
            tuple(row.get("sequence_ids", ())) != sequence_ids
            for row in ranked
        ):
            rank_agreement = False
        if (
            any(row.get("finite_logits") is not True for row in ranked)
            or any(
                len(values) != len(sequence_ids)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in values
                )
                for values in values_by_rank
            )
        ):
            finite_logits = False
        if (
            any(len(tokens) != len(sequence_ids) for tokens in tokens_by_rank)
            or len(set(tokens_by_rank)) != 1
        ):
            rank_agreement = False
        if values_by_rank and all(
            len(values) == len(sequence_ids)
            for values in values_by_rank
        ):
            reference = values_by_rank[0]
            if any(
                not math.isclose(
                    float(value),
                    float(reference[index]),
                    rel_tol=2e-3,
                    abs_tol=2e-2,
                )
                for values in values_by_rank[1:]
                for index, value in enumerate(values)
            ):
                top_logit_values_match = False
            top_values.append(tuple(float(value) for value in reference))
        else:
            top_logit_values_match = False
        if tokens_by_rank and len(tokens_by_rank[0]) == len(sequence_ids):
            for index, token in enumerate(tokens_by_rank[0]):
                emitted[index].append(token)
    if any(
        emitted[index] != request.get("output_token_ids")
        for index, request in enumerate(requests)
    ):
        rank_agreement = False
    return {
        "rank_token_agreement": rank_agreement,
        "finite_logits": finite_logits,
        "top_logit_values_match": top_logit_values_match,
        "top_logit_values": tuple(top_values),
    }


def _canonical_state_component_map(
    rank_row: Mapping[str, object],
    *,
    expected_source_ranks: tuple[int, ...],
) -> dict | None:
    rows = rank_row.get("canonical_state_components")
    if not isinstance(rows, (tuple, list)):
        return None
    expected_layers = {
        index for index in range(64) if index % 4 != 3
    }
    expected_keys = {
        (layer_index, source_rank)
        for layer_index in expected_layers
        for source_rank in expected_source_ranks
    }
    result = {}
    digest_fields = (
        "convolution_query_sha256",
        "convolution_key_sha256",
        "convolution_value_sha256",
        "recurrent_sha256",
    )
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        layer_index = row.get("layer_index")
        source_rank = row.get("source_rank")
        key = (layer_index, source_rank)
        digests = tuple(row.get(field) for field in digest_fields)
        if (
            key not in expected_keys
            or key in result
            or row.get("logical_rank") != source_rank // 2
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in digest
                )
                for digest in digests
            )
        ):
            return None
        result[key] = digests
    if set(result) != expected_keys:
        return None
    return result


def _checkpoint_cohort(rank_row: Mapping[str, object]) -> tuple | None:
    rows = rank_row.get("cohort")
    if not isinstance(rows, (tuple, list)) or not rows:
        return None
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        values = tuple(
            row.get(field)
            for field in ("slot_id", "generation", "request_id")
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in values
        ):
            return None
        result.append(values)
    if (
        len({row[0] for row in result}) != len(result)
        or len({row[2] for row in result}) != len(result)
    ):
        return None
    return tuple(result)


def build_correctness_artifact_rows(
    correctness_result: Mapping[str, object],
    *,
    source_revision: str,
    model_revision: str,
) -> list[dict]:
    rows = []
    expected_commit_counts = {
        "pre_migration": 0,
        "token_1": 0,
        "post_migration": 1,
        "token_4": 3,
        "token_8": 7,
        "token_32": 31,
        "token_128": 127,
    }
    for case in correctness_result.get("rows", ()):
        baseline = case["baseline"]
        candidate = case["candidate"]
        baseline_requests = baseline["requests"]
        candidate_requests = candidate["requests"]
        output_tokens_match = (
            [row.get("request_id") for row in baseline_requests]
            == [row.get("request_id") for row in candidate_requests]
            and [
                row.get("output_token_ids") for row in baseline_requests
            ] == [
                row.get("output_token_ids") for row in candidate_requests
            ]
        )
        baseline_proof = _correctness_proof_summary(baseline)
        candidate_proof = _correctness_proof_summary(candidate)
        rank_token_agreement = (
            baseline_proof["rank_token_agreement"]
            and candidate_proof["rank_token_agreement"]
        )
        finite_logits = (
            baseline_proof["finite_logits"]
            and candidate_proof["finite_logits"]
        )
        baseline_top_values = baseline_proof["top_logit_values"]
        candidate_top_values = candidate_proof["top_logit_values"]
        top_logit_values_match = (
            baseline_proof["top_logit_values_match"]
            and candidate_proof["top_logit_values_match"]
            and len(baseline_top_values) == len(candidate_top_values)
            and all(
                len(baseline_row) == len(candidate_row)
                and all(
                    math.isclose(
                        baseline_value,
                        candidate_value,
                        rel_tol=2e-3,
                        abs_tol=2e-2,
                    )
                    for baseline_value, candidate_value in zip(
                        baseline_row,
                        candidate_row,
                    )
                )
                for baseline_row, candidate_row in zip(
                    baseline_top_values,
                    candidate_top_values,
                )
            )
        )
        baseline_checkpoints = baseline.get(
            "correctness_state_checkpoints",
            {},
        )
        checkpoints = candidate.get("correctness_state_checkpoints", {})
        state_checkpoints_complete = (
            set(baseline_checkpoints) == set(STATE_CHECKPOINTS)
            and
            set(checkpoints) == set(STATE_CHECKPOINTS)
            and all(
                len(baseline_checkpoints[name]) == 4
                for name in STATE_CHECKPOINTS
            )
            and all(len(checkpoints[name]) == 4 for name in STATE_CHECKPOINTS)
        )
        single_commit_per_step = state_checkpoints_complete
        pair_replica_digest_match = state_checkpoints_complete
        baseline_candidate_state_match = state_checkpoints_complete
        if state_checkpoints_complete:
            expected_cohort = None
            base_counts = [
                row.get("runtime_snapshot", row)
                .get("state", {})
                .get("commit_count")
                for row in checkpoints["pre_migration"]
            ]
            for name, expected_count in expected_commit_counts.items():
                baseline_checkpoint_rows = sorted(
                    baseline_checkpoints[name],
                    key=lambda row: row.get("rank", -1),
                )
                checkpoint_rows = checkpoints[name]
                checkpoint_rows = sorted(
                    checkpoint_rows,
                    key=lambda row: row.get("rank", -1),
                )
                if (
                    [row.get("rank") for row in baseline_checkpoint_rows]
                    != list(RANKS)
                    or [row.get("rank") for row in checkpoint_rows]
                    != list(RANKS)
                ):
                    state_checkpoints_complete = False
                    single_commit_per_step = False
                    pair_replica_digest_match = False
                    baseline_candidate_state_match = False
                    continue
                baseline_cohorts = [
                    _checkpoint_cohort(rank_row)
                    for rank_row in baseline_checkpoint_rows
                ]
                candidate_cohorts = [
                    _checkpoint_cohort(rank_row)
                    for rank_row in checkpoint_rows
                ]
                checkpoint_cohort = baseline_cohorts[0]
                if (
                    checkpoint_cohort is None
                    or any(
                        cohort != checkpoint_cohort
                        for cohort in baseline_cohorts
                    )
                    or any(
                        cohort != checkpoint_cohort
                        for cohort in candidate_cohorts
                    )
                    or (
                        expected_cohort is not None
                        and checkpoint_cohort != expected_cohort
                    )
                ):
                    state_checkpoints_complete = False
                    single_commit_per_step = False
                    pair_replica_digest_match = False
                    baseline_candidate_state_match = False
                    continue
                expected_cohort = checkpoint_cohort
                for rank, rank_row in enumerate(checkpoint_rows):
                    snapshot = rank_row.get(
                        "runtime_snapshot",
                        rank_row,
                    )
                    state = snapshot.get("state", {})
                    if (
                        not isinstance(base_counts[rank], int)
                        or state.get("commit_count")
                        != base_counts[rank] + expected_count
                        or state.get("rollback_count") != 0
                        or state.get("temporary_live_tensors") != 0
                    ):
                        single_commit_per_step = False
                baseline_maps = [
                    _canonical_state_component_map(
                        rank_row,
                        expected_source_ranks=(rank,),
                    )
                    for rank, rank_row in enumerate(
                        baseline_checkpoint_rows
                    )
                ]
                candidate_active = name not in {
                    "pre_migration",
                    "token_1",
                }
                candidate_maps = [
                    _canonical_state_component_map(
                        rank_row,
                        expected_source_ranks=(
                            (
                                2 * (rank % 2),
                                2 * (rank % 2) + 1,
                            )
                            if candidate_active
                            else (rank,)
                        ),
                    )
                    for rank, rank_row in enumerate(checkpoint_rows)
                ]
                if (
                    any(mapping is None for mapping in baseline_maps)
                    or any(mapping is None for mapping in candidate_maps)
                ):
                    baseline_candidate_state_match = False
                else:
                    baseline_map = {
                        key: value
                        for mapping in baseline_maps
                        for key, value in mapping.items()
                    }
                    if candidate_active:
                        if (
                            candidate_maps[0] != candidate_maps[2]
                            or candidate_maps[1] != candidate_maps[3]
                        ):
                            pair_replica_digest_match = False
                        candidate_map = {
                            **candidate_maps[0],
                            **candidate_maps[1],
                        }
                    else:
                        candidate_map = {
                            key: value
                            for mapping in candidate_maps
                            for key, value in mapping.items()
                        }
                    if candidate_map != baseline_map:
                        baseline_candidate_state_match = False
                if name in {"pre_migration", "token_1"}:
                    continue
                output_maps = [
                    {
                        row["layer_index"]: row["sha256"]
                        for row in rank_row.get("output_digests", ())
                    }
                    for rank_row in checkpoint_rows
                ]
                state_maps = [
                    {
                        row["layer_index"]: (
                            row["convolution_sha256"],
                            row["recurrent_sha256"],
                        )
                        for row in rank_row.get("state_digests", ())
                    }
                    for rank_row in checkpoint_rows
                ]
                if (
                    any(len(mapping) != 48 for mapping in output_maps)
                    or len({tuple(sorted(mapping.items()))
                            for mapping in output_maps}) != 1
                    or any(len(mapping) != 48 for mapping in state_maps)
                    or state_maps[0] != state_maps[2]
                    or state_maps[1] != state_maps[3]
                ):
                    pair_replica_digest_match = False
        rows.append({
            "source_revision": source_revision,
            "model_revision": model_revision,
            "workload_id": case["workload_id"],
            "repetition": int(case["repetition"]),
            "output_tokens_match": output_tokens_match,
            "rank_token_agreement": rank_token_agreement,
            "finite_logits": finite_logits,
            "top_logit_values_match": top_logit_values_match,
            "state_checkpoints_complete": state_checkpoints_complete,
            "single_commit_per_step": single_commit_per_step,
            "pair_replica_digest_match": pair_replica_digest_match,
            "baseline_candidate_state_match":
                baseline_candidate_state_match,
            "baseline_requests": [{
                "request_id": request.get("request_id"),
                "runtime_request_id": request.get("runtime_request_id"),
                "output_token_ids": request.get("output_token_ids"),
            } for request in baseline_requests],
            "candidate_requests": [{
                "request_id": request.get("request_id"),
                "runtime_request_id": request.get("runtime_request_id"),
                "output_token_ids": request.get("output_token_ids"),
            } for request in candidate_requests],
            "baseline_step_proofs": baseline.get(
                "correctness_step_proofs"
            ),
            "candidate_step_proofs": candidate.get(
                "correctness_step_proofs"
            ),
            "baseline_state_checkpoints": baseline_checkpoints,
            "candidate_state_checkpoints": checkpoints,
        })
    return rows


def _summarize_resource_sample(
    plan: Mapping[str, object],
    sample: Mapping[str, object],
) -> dict:
    mapping = plan.get("gpu_rank_mapping")
    inventory = sample.get("gpu_inventory")
    processes = sample.get("process_rows", ())
    if (
        not isinstance(mapping, list)
        or sorted(row.get("rank") for row in mapping) != list(RANKS)
        or not isinstance(inventory, (tuple, list))
        or not isinstance(processes, (tuple, list))
    ):
        raise RuntimeError("resource identity evidence is incomplete")
    expected = {
        int(row["gpu_index"]): row.get("gpu_uuid")
        for row in mapping
    }
    if (
        len(expected) != len(RANKS)
        or any(
            not isinstance(gpu_uuid, str) or not gpu_uuid
            for gpu_uuid in expected.values()
        )
    ):
        raise RuntimeError("resource identity evidence is incomplete")
    observed = {}
    duplicate_identity = False
    for row in inventory:
        if not isinstance(row, Mapping):
            duplicate_identity = True
            continue
        gpu_index = row.get("gpu_index")
        if gpu_index in observed:
            duplicate_identity = True
        observed[gpu_index] = row
    selected = [
        observed.get(gpu_index)
        for gpu_index in expected
    ]
    identity_match = (
        not duplicate_identity
        and all(
            isinstance(row, Mapping)
            and row.get("gpu_uuid") == expected[gpu_index]
            for gpu_index, row in zip(expected, selected)
        )
    )
    strict_clean = (
        identity_match
        and all(
            isinstance(row.get("memory_used_mib"), int)
            and not isinstance(row.get("memory_used_mib"), bool)
            and row["memory_used_mib"] <= 1024
            and isinstance(row.get("utilization_percent"), int)
            and not isinstance(row.get("utilization_percent"), bool)
            and row["utilization_percent"] <= 5
            and row.get("compute_processes") == []
            for row in selected
        )
    )
    return {
        "attempt_tag": plan.get("attempt_tag"),
        "stage": sample.get("stage"),
        "measurement_scope": sample.get("measurement_scope"),
        "run_label": sample.get("run_label"),
        "sample_index": sample.get("sample_index"),
        "gpu_inventory": [
            dict(row) if isinstance(row, Mapping) else row
            for row in inventory
        ],
        "process_rows": [
            dict(row) if isinstance(row, Mapping) else row
            for row in processes
        ],
        "strict_clean": strict_clean,
        "identity_match": identity_match,
        "foreign_processes": [
            dict(row)
            for row in processes
            if (
                isinstance(row, Mapping)
                and row.get("attempt_tag") != plan.get("attempt_tag")
            )
        ],
    }


def _summarize_cleanup_receipts(
    records,
    *,
    task_paths,
) -> dict:
    retained_generations = 0
    retained_leases = 0
    retained_tensors = 0
    retained_process_groups = 0
    validated_rank_receipts = 0
    validated_workers = 0
    cleanup_durations_ns = []
    owned_processes_remaining = []
    for label, receipt, candidate_enabled in records:
        if not isinstance(receipt, Mapping):
            raise RuntimeError(
                f"worker cleanup evidence is incomplete: {label}"
            )
        exit_codes = receipt.get("rank_exit_codes")
        rank_receipts = receipt.get("rank_cleanup_receipts")
        cleanup_started_ns = receipt.get("cleanup_started_ns")
        cleanup_finished_ns = receipt.get("cleanup_finished_ns")
        cleanup_duration_ns = receipt.get("cleanup_duration_ns")
        if (
            not isinstance(exit_codes, list)
            or not exit_codes
            or any(
                isinstance(code, bool)
                or not isinstance(code, int)
                or code != 0
                for code in exit_codes
            )
            or receipt.get("process_group_destroyed") is not True
            or receipt.get("owned_children_remaining") != []
            or not isinstance(rank_receipts, list)
            or len(rank_receipts) != len(exit_codes)
            or isinstance(cleanup_started_ns, bool)
            or not isinstance(cleanup_started_ns, int)
            or cleanup_started_ns < 0
            or isinstance(cleanup_finished_ns, bool)
            or not isinstance(cleanup_finished_ns, int)
            or cleanup_finished_ns < cleanup_started_ns
            or isinstance(cleanup_duration_ns, bool)
            or not isinstance(cleanup_duration_ns, int)
            or cleanup_duration_ns
            != cleanup_finished_ns - cleanup_started_ns
        ):
            raise RuntimeError(
                f"worker cleanup evidence is incomplete: {label}"
            )
        expected_ranks = list(range(len(exit_codes)))
        if sorted(row.get("rank") for row in rank_receipts) != expected_ranks:
            raise RuntimeError(
                f"rank cleanup evidence is incomplete: {label}"
            )
        for row in rank_receipts:
            if row.get("process_group_destroyed") is not True:
                retained_process_groups += 1
                raise RuntimeError(
                    f"rank cleanup evidence is incomplete: {label}"
                )
            candidate = row.get(
                "qwen38_topology_local_tp2_cleanup"
            )
            if candidate_enabled:
                if (
                    not isinstance(candidate, Mapping)
                    or candidate.get("pair_groups_destroyed") != 2
                    or candidate.get("candidate_state_released") is not True
                    or not isinstance(
                        candidate.get("published_generations_remaining"),
                        int,
                    )
                    or not isinstance(
                        candidate.get("temporary_live_tensors"),
                        int,
                    )
                ):
                    raise RuntimeError(
                        f"candidate cleanup evidence is incomplete: {label}"
                    )
                retained_generations += int(
                    candidate["published_generations_remaining"]
                )
                retained_leases += int(
                    candidate["published_generations_remaining"]
                )
                retained_tensors += int(
                    candidate["temporary_live_tensors"]
                )
            elif candidate is not None:
                raise RuntimeError(
                    f"unexpected candidate cleanup evidence: {label}"
                )
        validated_workers += 1
        validated_rank_receipts += len(rank_receipts)
        cleanup_durations_ns.append(cleanup_duration_ns)
    complete = (
        validated_workers == len(records)
        and retained_generations == 0
        and retained_leases == 0
        and retained_tensors == 0
        and retained_process_groups == 0
        and not owned_processes_remaining
    )
    if not complete:
        raise RuntimeError("worker cleanup evidence is incomplete")
    return {
        "complete": complete,
        "retained_generations": retained_generations,
        "retained_leases": retained_leases,
        "retained_tensors": retained_tensors,
        "retained_process_groups": retained_process_groups,
        "owned_processes_remaining": owned_processes_remaining,
        "foreign_process_actions": [],
        "task_paths": list(task_paths),
        "validated_worker_cleanups": validated_workers,
        "validated_rank_cleanup_receipts": validated_rank_receipts,
        "cleanup_durations_ns": cleanup_durations_ns,
        "worker_cleanup_receipts": [
            {
                "label": label,
                "candidate_enabled": candidate_enabled,
                "receipt": receipt,
            }
            for label, receipt, candidate_enabled in records
        ],
    }


def _candidate_weight_layout(
    epoch_results,
    *,
    steady_increment_bytes_per_rank: int,
) -> dict:
    release_rows = []
    for result in epoch_results:
        if result.get("arm") != "candidate":
            continue
        for case in result.get("rows", ()):
            ranked = _ranked_snapshots(
                case.get("after_snapshots"),
                "candidate after",
            )
            release_rows.extend(ranked.values())
    if not release_rows:
        raise RuntimeError("candidate weight release evidence is missing")
    layer_counts = {
        row.get("released_layer_count") for row in release_rows
    }
    released_bytes = {
        row.get("released_bytes") for row in release_rows
    }
    if (
        layer_counts != {48}
        or len(released_bytes) != 1
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in released_bytes
        )
    ):
        raise RuntimeError("candidate weight release evidence is incomplete")
    return {
        "baseline_tp4_decode_accumulation_retained": False,
        "released_layer_count_per_rank": 48,
        "released_bytes_per_rank": next(iter(released_bytes)),
        "steady_increment_bytes_per_rank": int(
            steady_increment_bytes_per_rank
        ),
    }


def build_raw_artifact_payloads(
    *,
    plan: Mapping[str, object],
    correctness_result: Mapping[str, object],
    epoch_results: tuple[Mapping[str, object], ...],
    service_result: Mapping[str, object],
    resource_samples: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    identity = {
        "source_revision": plan["source_revision"],
        "model_revision": plan["model_revision"],
    }
    artifacts = {
        name: []
        for name in (
            "request_rows.jsonl",
            "scheduler_step_rows.jsonl",
            "candidate_hit_rows.jsonl",
            "collective_rows.jsonl",
            "migration_rows.jsonl",
            "memory_rows.jsonl",
        )
    }
    for epoch_result in epoch_results:
        rows = build_performance_artifact_rows(
            epoch_result,
            **identity,
        )
        for name, values in rows.items():
            artifacts[name].extend(values)

    request_index = {
        (
            row["epoch"],
            row["workload_id"],
            row["repetition"],
        ): row
        for row in artifacts["request_rows.jsonl"]
    }
    baseline_for_candidate = {1: 3, 2: 0}
    for row in artifacts["migration_rows.jsonl"]:
        candidate = request_index[(
            row["epoch"],
            row["workload_id"],
            row["repetition"],
        )]
        baseline = request_index[(
            baseline_for_candidate[row["epoch"]],
            row["workload_id"],
            row["repetition"],
        )]
        candidate_tpot = sum(
            request["tpot_ns"] for request in candidate["requests"]
        ) / len(candidate["requests"])
        baseline_tpot = sum(
            request["tpot_ns"] for request in baseline["requests"]
        ) / len(baseline["requests"])
        savings = baseline_tpot - candidate_tpot
        row["break_even_output_tokens"] = (
            row["latency_ns"] / savings
            if savings > 0
            else 1e30
        )

    memory = artifacts["memory_rows.jsonl"]
    peak_by_arm_rank = {}
    for row in memory:
        key = (row["arm"], row["rank"])
        peak_by_arm_rank[key] = max(
            int(row["peak_allocated_bytes"]),
            peak_by_arm_rank.get(key, 0),
        )
    increments = [
        max(
            0,
            peak_by_arm_rank.get(("candidate", rank), 0)
            - peak_by_arm_rank.get(("baseline", rank), 0),
        )
        for rank in range(4)
    ]
    cleanup_records = [
        (
            f"correctness/{arm}",
            correctness_result.get("cleanup", {}).get(arm),
            arm == "candidate",
        )
        for arm in ("baseline", "candidate")
    ]
    cleanup_records.extend(
        (
            f"epoch/{result.get('epoch')}/{result.get('arm')}",
            result.get("cleanup"),
            result.get("arm") == "candidate",
        )
        for result in epoch_results
    )
    cleanup_records.extend(
        (
            f"service/{row.get('workload_id')}/replica/{replica_index}",
            replica.get("cleanup"),
            False,
        )
        for row in service_result.get("rows", ())
        for replica_index, replica in enumerate(row.get("replicas", ()))
    )
    resource_rows = [{
        **identity,
        **_summarize_resource_sample(plan, sample),
    } for sample in resource_samples]
    cleanup = {
        **identity,
        **_summarize_cleanup_receipts(
            cleanup_records,
            task_paths=(
                plan["attempt_root"],
                plan["source_root"],
                plan["raw_root"],
                plan["controller_root"],
                plan["bundle_root"],
                *plan["environment"].values(),
            ),
        ),
    }
    weight_layout = _candidate_weight_layout(
        epoch_results,
        steady_increment_bytes_per_rank=max(increments),
    )
    static = {
        "source_manifest.json": {
            **identity,
            "attempt_tag": plan["attempt_tag"],
            "source_tree_sha256": plan["source_tree_sha256"],
            "source_archive_complete": True,
        },
        "model_manifest.json": {
            **identity,
            "model_repository": plan["model_repository"],
            "num_hidden_layers": 64,
            "linear_attention_layer_count": 48,
            "full_attention_layer_count": 16,
        },
        "environment_manifest.json": {
            **identity,
            "environment_complete": True,
            "paths": dict(plan["environment"]),
        },
        "gpu_topology.json": {
            **identity,
            **plan["topology"],
            "selection_frozen": True,
        },
        "gpu_rank_manifest.json": {
            **identity,
            "ranks": list(range(4)),
            "mapping": list(plan["gpu_rank_mapping"]),
        },
        "pair_group_manifest.json": {
            **identity,
            "pair_groups": list(plan["pair_groups"]),
            "optimal_matching": True,
        },
        "workload_manifest.json": {
            **identity,
            "workloads": list(WORKLOADS),
        },
        "campaign_epoch_manifest.json": {
            **identity,
            "epochs": [
                {
                    "epoch": row["epoch"],
                    "arm": row["arm"],
                    "workload_order": list(row["workload_order"]),
                }
                for row in plan["campaign_epochs"]
            ],
        },
        "feature_contract.json": {
            **identity,
            "default_off": True,
            "eager": True,
            "tensor_parallel_size": 4,
        },
        "weight_layout_manifest.json": {
            **identity,
            **weight_layout,
        },
        "state_layout_manifest.json": {
            **identity,
            "temporary_objects_released": all(
                row.get("temporary_live_tensors") == 0
                for row in artifacts["migration_rows.jsonl"]
            ),
        },
        "cleanup.json": cleanup,
    }
    return {
        **static,
        **artifacts,
        "correctness_rows.jsonl": build_correctness_artifact_rows(
            correctness_result,
            **identity,
        ),
        "resource_rows.jsonl": resource_rows,
        "service_control_rows.jsonl": [
            {
                **identity,
                **{
                    key: value
                    for key, value in row.items()
                    if key != "replicas"
                },
                "replicas": [{
                    key: value
                    for key, value in replica.items()
                    if key not in {"cleanup", "request_set_digest"}
                } for replica in row.get("replicas", ())],
                "global_tp4_baseline_output_parity": (
                    sorted(
                        (
                            request.get("request_id"),
                            request.get("output_token_ids"),
                            request.get("stop_position"),
                            request.get("stop_reason"),
                            request.get("decoded_text_sha256"),
                        )
                        for request in row.get("requests", ())
                    )
                    == sorted(
                        (
                            request.get("request_id"),
                            request.get("output_token_ids"),
                            request.get("stop_position"),
                            request.get("stop_reason"),
                            request.get("decoded_text_sha256"),
                        )
                        for request in request_index[(
                            0,
                            row["workload_id"],
                            0,
                        )]["requests"]
                    )
                ),
            }
            for row in service_result.get("rows", ())
        ],
    }


def run_correctness_campaign(
    *,
    model_root: Path,
    output_root: Path,
    workloads: Mapping[str, tuple] = WORKLOADS,
    seed_namespace: str = "correctness",
    case_runner: Callable = run_engine_case,
    engine_factory: Callable | None = None,
    row_sink: Callable[[dict], None] | None = None,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> dict:
    case_specs = []
    for workload_id, (_, prompt_tokens, output_tokens, concurrency) in (
        workloads.items()
    ):
        for repetition in range(5):
            case_specs.append((
                workload_id,
                repetition,
                build_request_specs(
                    prompt_tokens,
                    output_tokens,
                    concurrency,
                    f"{seed_namespace}/{workload_id}/r{repetition}",
                ),
            ))
    arm_results = {"baseline": {}, "candidate": {}}
    cleanup = {}
    for arm in ("baseline", "candidate"):
        engine = None
        if engine_factory is not None:
            engine = engine_factory(
                Path(model_root),
                tensor_parallel_size=4,
                enforce_eager=True,
                max_num_seqs=8,
                max_model_len=2176,
                max_num_batched_tokens=8192,
                qwen38_topology_local_tp2_islands=(arm == "candidate"),
            )
        try:
            for workload_id, repetition, request_specs in case_specs:
                kwargs = {
                    "model_root": Path(model_root),
                    "arm": arm,
                    "workload_id": workload_id,
                    "request_specs": request_specs,
                    "warmup": True,
                    "epoch": -1,
                    "repetition": repetition,
                    "correctness_authority": True,
                }
                if engine is not None:
                    kwargs.update({
                        "engine": engine,
                        "close_engine": False,
                    })
                arm_results[arm][(workload_id, repetition)] = (
                    case_runner(**kwargs)
                )
                arm_results[arm][
                    (workload_id, repetition)
                ]["timing_authority"] = False
        finally:
            if engine is not None:
                cleanup_started_ns = clock_ns()
                receipt = dict(engine.exit())
                cleanup_finished_ns = clock_ns()
                receipt.update({
                    "cleanup_started_ns": cleanup_started_ns,
                    "cleanup_finished_ns": cleanup_finished_ns,
                    "cleanup_duration_ns": (
                        cleanup_finished_ns - cleanup_started_ns
                    ),
                })
                cleanup[arm] = receipt

    rows = []
    for workload_id, repetition, _request_specs in case_specs:
        arm_rows = {
            arm: arm_results[arm][(workload_id, repetition)]
            for arm in ("baseline", "candidate")
        }
        baseline_tokens = [
            row["output_token_ids"]
            for row in arm_rows["baseline"]["requests"]
        ]
        candidate_tokens = [
            row["output_token_ids"]
            for row in arm_rows["candidate"]["requests"]
        ]
        if baseline_tokens != candidate_tokens:
            raise RuntimeError("correctness output token mismatch")
        rows.append({
            "workload_id": workload_id,
            "repetition": repetition,
            "state_checkpoints": STATE_CHECKPOINTS,
            "baseline": arm_rows["baseline"],
            "candidate": arm_rows["candidate"],
            "timing_authority": False,
        })
        if row_sink is not None:
            row_sink(rows[-1])
    result = {
        "schema_version": WORKER_SCHEMA,
        "phase": "correctness",
        "rows": rows,
        "cleanup": cleanup,
    }
    if row_sink is None:
        _atomic_write_json(Path(output_root) / "correctness.json", result)
    return result


def run_service_control(
    *,
    model_root: Path,
    output_root: Path,
    pair_devices: tuple[tuple[int, int], tuple[int, int]],
    workloads: tuple[str, ...],
    replica_runner: Callable | None = None,
    parallel_runner: Callable | None = None,
    row_sink: Callable[[dict], None] | None = None,
) -> dict:
    if (
        len(pair_devices) != 2
        or any(len(pair) != 2 for pair in pair_devices)
        or set(pair_devices[0]) & set(pair_devices[1])
    ):
        raise ValueError("service-control device pairs must be disjoint")
    if replica_runner is None and parallel_runner is None:
        raise RuntimeError(
            "service control requires a process-isolated replica runner"
        )
    sink_rows = []
    sink = row_sink or sink_rows.append
    workload_rows = []
    for workload_id in workloads:
        if workload_id not in WORKLOADS:
            raise ValueError("unknown service-control workload")
        _, prompt_tokens, output_tokens, concurrency = WORKLOADS[
            workload_id
        ]
        request_specs = build_request_specs(
            prompt_tokens,
            output_tokens,
            concurrency,
            f"timing/{workload_id}/r0",
        )
        request_specs_by_replica = tuple(
            tuple(
                row
                for request_index, row in enumerate(request_specs)
                if request_index % 2 == replica_index
            )
            for replica_index in range(2)
        )
        if parallel_runner is not None:
            replicas = tuple(parallel_runner(
                model_root=Path(model_root),
                pair_devices=pair_devices,
                workload_id=workload_id,
                request_specs_by_replica=request_specs_by_replica,
                shared_start_ns=0,
            ))
        else:
            replicas = []
            for devices, selected in zip(
                pair_devices,
                request_specs_by_replica,
            ):
                replica = replica_runner(
                    model_root=Path(model_root),
                    pair_devices=devices,
                    workload_id=workload_id,
                    request_specs=selected,
                    shared_start_ns=0,
                )
                replicas.append(replica)
        if len(replicas) != 2:
            raise RuntimeError(
                "service-control replica inventory mismatch"
            )
        normalized_replicas = []
        for replica_index, (devices, selected, replica) in enumerate(zip(
            pair_devices,
            request_specs_by_replica,
            replicas,
        )):
            if (
                not isinstance(replica, Mapping)
                or replica.get("replica_tensor_parallel_size") != 2
                or replica.get("request_set_digest")
                != _request_set_digest(selected)
            ):
                raise RuntimeError(
                    "service-control replica identity mismatch"
                )
            replica_requests = replica.get("requests")
            if (
                not isinstance(replica_requests, list)
                or [
                    request.get("request_id")
                    for request in replica_requests
                ] != [
                    request["request_id"] for request in selected
                ]
            ):
                raise RuntimeError(
                    "service-control replica request split mismatch"
                )
            normalized_replicas.append({
                "replica_index": replica_index,
                "pair_devices": list(devices),
                "replica_tensor_parallel_size": 2,
                "request_set_digest": replica["request_set_digest"],
                "requests": replica_requests,
                "memory": list(replica.get("memory", ())),
                "peak_memory_by_gpu": _service_peak_memory(
                    replica.get("memory", ()),
                    devices,
                ),
                **_service_metrics(replica_requests),
                "cleanup": replica.get("cleanup"),
            })
        requests = [
            request
            for replica in normalized_replicas
            for request in replica["requests"]
        ]
        if len(requests) != concurrency:
            raise RuntimeError(
                "service-control request inventory mismatch"
            )
        metrics = _service_metrics(requests)
        replica_qps = [
            replica["request_qps"] for replica in normalized_replicas
        ]
        replica_throughput = [
            replica["output_tokens_per_second"]
            for replica in normalized_replicas
        ]
        row = {
            "arm": "TP2_X2_SERVICE_CONTROL",
            "classification_authority": False,
            "workload_id": workload_id,
            "request_set_digest": _request_set_digest(request_specs),
            "requests": requests,
            "replicas": normalized_replicas,
            **metrics,
            "replica_balance": {
                "request_counts": [
                    replica["request_count"]
                    for replica in normalized_replicas
                ],
                "request_qps_max_to_min": (
                    max(replica_qps) / min(replica_qps)
                ),
                "output_tokens_per_second_max_to_min": (
                    max(replica_throughput) / min(replica_throughput)
                ),
            },
        }
        sink(row)
        workload_rows.append(row)
    result = {
        "schema_version": WORKER_SCHEMA,
        "arm": "TP2_X2_SERVICE_CONTROL",
        "classification_authority": False,
        "rows": workload_rows,
    }
    if row_sink is None:
        _atomic_write_json(
            Path(output_root) / "tp2-x2-service-control.json",
            result,
        )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    correctness = subparsers.add_parser("correctness")
    correctness.add_argument("--model-root", type=Path, required=True)
    correctness.add_argument("--output-root", type=Path, required=True)
    correctness.add_argument("--source-revision", required=True)
    correctness.add_argument("--model-revision", required=True)
    performance = subparsers.add_parser("performance-epoch")
    performance.add_argument("--model-root", type=Path, required=True)
    performance.add_argument("--output-root", type=Path, required=True)
    performance.add_argument("--epoch", type=int, required=True)
    performance.add_argument(
        "--arm",
        choices=("baseline", "candidate"),
        required=True,
    )
    performance.add_argument("--workload-order", required=True)
    performance.add_argument("--source-revision", required=True)
    performance.add_argument("--model-revision", required=True)
    service = subparsers.add_parser("service-control")
    service.add_argument("--model-root", type=Path, required=True)
    service.add_argument("--output-root", type=Path, required=True)
    service.add_argument("--pair-devices", required=True)
    service.add_argument("--workloads", required=True)
    replica = subparsers.add_parser("service-replica")
    replica.add_argument("--model-root", type=Path, required=True)
    replica.add_argument("--workload-id", required=True)
    replica.add_argument("--request-specs-path", type=Path, required=True)
    replica.add_argument("--output-path", type=Path, required=True)
    replica.add_argument("--start-barrier", type=Path, required=True)
    finalize = subparsers.add_parser("finalize-artifacts")
    finalize.add_argument("--plan", type=Path, required=True)
    finalize.add_argument("--worker-output-root", type=Path, required=True)
    finalize.add_argument("--resource-samples", type=Path, required=True)
    finalize.add_argument("--raw-root", type=Path, required=True)
    return parser


def main(
    argv=None,
    *,
    performance_runner: Callable = run_performance_epoch,
    correctness_runner: Callable = run_correctness_campaign,
    service_runner: Callable = run_service_control,
    replica_runner: Callable | None = None,
    service_replica_runner: Callable = run_service_replica_case,
) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.mode == "correctness":
        result = correctness_runner(
            model_root=args.model_root,
            output_root=args.output_root,
            engine_factory=_default_engine_factory,
        )
        _atomic_write_json(
            args.output_root / "correctness-artifact-rows.json",
            build_correctness_artifact_rows(
                result,
                source_revision=args.source_revision,
                model_revision=args.model_revision,
            ),
        )
    elif args.mode == "performance-epoch":
        result = performance_runner(
            model_root=args.model_root,
            output_root=args.output_root,
            epoch=args.epoch,
            arm=args.arm,
            workload_order=tuple(args.workload_order.split(",")),
            engine_factory=_default_engine_factory,
        )
        _atomic_write_json(
            args.output_root
            / f"epoch-{args.epoch}-artifact-rows.json",
            build_performance_artifact_rows(
                result,
                source_revision=args.source_revision,
                model_revision=args.model_revision,
            ),
        )
    elif args.mode == "service-control":
        pair_devices = tuple(
            tuple(int(device) for device in pair.split(","))
            for pair in args.pair_devices.split(";")
        )
        service_kwargs = {
            "model_root": args.model_root,
            "output_root": args.output_root,
            "pair_devices": pair_devices,
            "workloads": tuple(args.workloads.split(",")),
        }
        if replica_runner is None:
            service_kwargs["parallel_runner"] = (
                lambda **kwargs: _run_service_replicas_in_subprocesses(
                    output_root=args.output_root,
                    **kwargs,
                )
            )
        else:
            service_kwargs["replica_runner"] = replica_runner
        result = service_runner(**service_kwargs)
    elif args.mode == "service-replica":
        deadline = time.monotonic() + 120.0
        while not args.start_barrier.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "service-control start barrier timed out"
                )
            time.sleep(0.01)
        request_specs = tuple(json.loads(
            args.request_specs_path.read_text(encoding="utf-8")
        ))
        result = service_replica_runner(
            model_root=args.model_root,
            workload_id=args.workload_id,
            request_specs=request_specs,
        )
        _atomic_write_json(args.output_path, result)
        return 0
    elif args.mode == "finalize-artifacts":
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        worker_root = args.worker_output_root
        correctness_result = json.loads(
            (worker_root / "correctness" / "worker-receipt.json")
            .read_text(encoding="utf-8")
        )
        epoch_results = tuple(
            json.loads(
                (worker_root / f"epoch-{epoch}" / "worker-receipt.json")
                .read_text(encoding="utf-8")
            )
            for epoch in range(4)
        )
        service_result = json.loads(
            (worker_root / "service-control" / "worker-receipt.json")
            .read_text(encoding="utf-8")
        )
        resource_samples = tuple(json.loads(
            args.resource_samples.read_text(encoding="utf-8")
        ))
        payloads = build_raw_artifact_payloads(
            plan=plan,
            correctness_result=correctness_result,
            epoch_results=epoch_results,
            service_result=service_result,
            resource_samples=resource_samples,
        )
        if args.raw_root.exists() and any(args.raw_root.iterdir()):
            raise ValueError("raw artifact root must be fresh")
        args.raw_root.mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            if name.endswith(".jsonl"):
                _atomic_write_jsonl(args.raw_root / name, payload)
            else:
                _atomic_write_json(args.raw_root / name, payload)
        print(json.dumps({
            "raw_root": str(args.raw_root),
            "artifact_count": len(payloads),
        }, sort_keys=True))
        return 0
    else:
        raise ValueError("worker mode is unsupported")
    _atomic_write_json(
        args.output_root / "worker-receipt.json",
        result,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
