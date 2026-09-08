#!/usr/bin/env python3
"""Run the Qwen3.8 topology-local TP2 whole-model gate workloads."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
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
            if after_row.get(field, 0) != 0:
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
    engine_factory: Callable = _default_engine_factory,
    sampling_params_factory: Callable = _default_sampling_params_factory,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    timeout_s: float = 120.0,
) -> dict:
    if arm not in {"baseline", "candidate"}:
        raise ValueError("arm must be baseline or candidate")
    if workload_id not in WORKLOADS:
        raise ValueError("unknown workload")
    if not request_specs:
        raise ValueError("request_specs must not be empty")
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

    engine = engine_factory(
        Path(model_root),
        tensor_parallel_size=4,
        enforce_eager=True,
        max_num_seqs=max(8, concurrency),
        max_model_len=prompt_tokens + output_tokens,
        max_num_batched_tokens=prompt_tokens * concurrency,
        qwen38_topology_local_tp2_islands=(arm == "candidate"),
    )
    before_snapshots = ()
    cleanup = None
    try:
        if (
            getattr(getattr(engine, "model_runner", None), "rank", None)
            != 0
            or getattr(
                getattr(engine, "model_runner", None),
                "world_size",
                None,
            ) != 4
        ):
            raise RuntimeError("TP4 engine ownership mismatch")
        if arm == "candidate":
            before_snapshots = (
                engine.qwen38_topology_local_tp2_snapshots(
                    timeout_s=float(timeout_s)
                )
            )

        lifecycle = {}
        for seq_id, request in enumerate(request_specs):
            admitted_ns = clock_ns()
            sampling = sampling_params_factory(
                temperature=0.0,
                max_tokens=output_tokens,
                ignore_eos=True,
            )
            engine.add_request(
                request["prompt_token_ids"],
                sampling,
            )
            lifecycle[seq_id] = {
                "request_id": request["request_id"],
                "admitted_ns": admitted_ns,
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
            token_deltas = observation.get(
                "new_completion_tokens_by_seq"
            )
            if (
                isinstance(step_end_ns, bool)
                or not isinstance(step_end_ns, int)
                or not isinstance(token_deltas, dict)
            ):
                raise RuntimeError("step timing observation is invalid")
            scheduled = observation.get("scheduled", [])
            scheduled_ids = [
                lifecycle[int(row["seq_id"])]["request_id"]
                for row in scheduled
            ]
            scheduler_step_rows.append({
                "step_index": step_index,
                "is_prefill": observation.get("is_prefill"),
                "batch_kind": observation.get("batch_kind"),
                "request_ids": scheduled_ids,
                "step_start_ns": observation.get("step_start_ns"),
                "step_end_ns": step_end_ns,
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
        for seq_id in range(concurrency):
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
            requests.append({
                **row,
                **metrics,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": output_tokens,
                "completion_ns": row["token_timestamps_ns"][-1],
            })
    finally:
        cleanup = engine.exit()

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
    }


def run_performance_epoch(
    *,
    model_root: Path,
    output_root: Path,
    epoch: int,
    arm: str,
    workload_order: tuple[str, ...],
    case_runner: Callable = run_engine_case,
    row_sink: Callable[[dict], None] | None = None,
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
    rows = []
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
            row = case_runner(
                model_root=Path(model_root),
                arm=arm,
                workload_id=workload_id,
                request_specs=request_specs,
                warmup=warmup,
                epoch=int(epoch),
                repetition=measured_repetition,
            )
            rows.append(row)
            if row_sink is not None:
                row_sink(row)
    result = {
        "schema_version": WORKER_SCHEMA,
        "phase": "performance_epoch",
        "epoch": int(epoch),
        "arm": arm,
        "workload_order": list(workload_order),
        "rows": rows,
    }
    if row_sink is None:
        _atomic_write_json(
            Path(output_root) / f"epoch-{epoch}.json",
            result,
        )
    return result


def run_correctness_campaign(
    *,
    model_root: Path,
    output_root: Path,
    workloads: Mapping[str, tuple] = WORKLOADS,
    seed_namespace: str = "correctness",
    case_runner: Callable = run_engine_case,
) -> dict:
    rows = []
    for workload_id, (_, prompt_tokens, output_tokens, concurrency) in (
        workloads.items()
    ):
        for repetition in range(5):
            request_specs = build_request_specs(
                prompt_tokens,
                output_tokens,
                concurrency,
                f"{seed_namespace}/{workload_id}/r{repetition}",
            )
            arm_rows = {}
            for arm in ("baseline", "candidate"):
                arm_rows[arm] = case_runner(
                    model_root=Path(model_root),
                    arm=arm,
                    workload_id=workload_id,
                    request_specs=request_specs,
                    warmup=True,
                    epoch=-1,
                    repetition=repetition,
                )
                arm_rows[arm]["timing_authority"] = False
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
    result = {
        "schema_version": WORKER_SCHEMA,
        "phase": "correctness",
        "rows": rows,
    }
    _atomic_write_json(Path(output_root) / "correctness.json", result)
    return result


def run_service_control(
    *,
    model_root: Path,
    output_root: Path,
    pair_devices: tuple[tuple[int, int], tuple[int, int]],
    workloads: tuple[str, ...],
    replica_runner: Callable | None = None,
    row_sink: Callable[[dict], None] | None = None,
) -> dict:
    if (
        len(pair_devices) != 2
        or any(len(pair) != 2 for pair in pair_devices)
        or set(pair_devices[0]) & set(pair_devices[1])
    ):
        raise ValueError("service-control device pairs must be disjoint")
    if replica_runner is None:
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
            f"service/{workload_id}/r0",
        )
        replicas = []
        for replica_index, devices in enumerate(pair_devices):
            selected = tuple(
                row
                for request_index, row in enumerate(request_specs)
                if request_index % 2 == replica_index
            )
            replica = replica_runner(
                model_root=Path(model_root),
                pair_devices=devices,
                workload_id=workload_id,
                request_specs=selected,
                shared_start_ns=0,
            )
            replicas.append(replica)
        requests = [
            request
            for replica in replicas
            for request in replica.get("requests", [])
        ]
        if len(requests) != concurrency:
            raise RuntimeError(
                "service-control request inventory mismatch"
            )
        earliest = min(row["admitted_ns"] for row in requests)
        latest = max(row["completion_ns"] for row in requests)
        row = {
            "arm": "TP2_X2_SERVICE_CONTROL",
            "classification_authority": False,
            "workload_id": workload_id,
            "requests": requests,
            "replicas": replicas,
            "makespan_ns": latest - earliest,
            "request_qps": (
                concurrency * 1_000_000_000 / (latest - earliest)
                if latest > earliest
                else math.inf
            ),
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
