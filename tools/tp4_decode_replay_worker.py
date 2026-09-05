#!/usr/bin/env python3
"""Run one canonical Qwen3.8 TP4 eager or decode-replay arm."""

from __future__ import annotations

import argparse
from itertools import count
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import tempfile
import time

import tp4_decode_replay_contract as contract


WORKER_SCHEMA = "tinyllmforge.tp4-decode-replay-worker.v1"
STRICT_CLEAN = "strict_clean"
SHARED_CAPACITY = "shared_capacity"


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
        os.fsync(handle.fileno())
    temporary.replace(path)


def _default_engine_factory(model_root, **kwargs):
    from tinyvllm.engine.llm_engine import LLMEngine

    return LLMEngine(str(model_root), **kwargs)


def _free_rendezvous_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _rendezvous_address_in_use(error: BaseException) -> bool:
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current)
        if (
            "EADDRINUSE" in message
            or "address already in use" in message.lower()
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _cleanup_failed_engine_children() -> None:
    import multiprocessing

    children = tuple(multiprocessing.active_children())
    for child in children:
        if child.is_alive():
            child.terminate()
    for child in children:
        child.join(timeout=10.0)
    lingering = tuple(child for child in children if child.is_alive())
    for child in lingering:
        child.kill()
    for child in lingering:
        child.join(timeout=10.0)
    if any(child.is_alive() for child in lingering):
        raise RuntimeError(
            "failed engine children remained after rendezvous retry cleanup"
        )


def create_engine_with_rendezvous_retry(
    model_root,
    *,
    engine_config,
    port_factory,
    engine_factory=_default_engine_factory,
    environment=os.environ,
    cleanup_failed_attempt=_cleanup_failed_engine_children,
    sleep=time.sleep,
    maximum_attempts=3,
    retry_delay_s=0.25,
):
    if maximum_attempts <= 0:
        raise ValueError("maximum_attempts must be positive")
    for attempt in range(maximum_attempts):
        port = int(port_factory())
        environment["TINYVLLM_DIST_PORT"] = str(port)
        try:
            return (
                engine_factory(
                    Path(model_root),
                    **dict(engine_config),
                ),
                port,
            )
        except RuntimeError as exc:
            if not _rendezvous_address_in_use(exc):
                raise
            cleanup_failed_attempt()
            if attempt + 1 == maximum_attempts:
                raise RuntimeError(
                    "rendezvous port retries exhausted"
                ) from exc
            sleep(float(retry_delay_s))
    raise RuntimeError("rendezvous retry loop exhausted unexpectedly")


def _default_sampling_params_factory(**kwargs):
    from tinyvllm.sampling_params import SamplingParams

    return SamplingParams(**kwargs)


def _reset_sequence_ids() -> None:
    from tinyvllm.engine.sequence import Sequence

    Sequence.counter = count()


def build_engine_config(*, arm: str, workload: str) -> dict:
    if arm not in contract.ARMS:
        raise ValueError("arm is invalid")
    if workload not in contract.WORKLOADS:
        raise ValueError("workload is invalid")
    profile = contract.WORKLOADS[workload]
    concurrency = int(profile["concurrency"])
    admission_mode = os.environ.get(
        "TINYLLMFORGE_TP4_ADMISSION_MODE",
        STRICT_CLEAN,
    )
    if admission_mode not in {STRICT_CLEAN, SHARED_CAPACITY}:
        raise ValueError("admission mode is invalid")
    max_model_len = (
        int(profile["prompt_tokens"])
        + int(profile["output_tokens"])
    )
    engine_config = {
        "tensor_parallel_size": 4,
        # Q2 rank 0 materializes a 3.79 GiB full-vocabulary BF16
        # projection before selecting the final token rows. Keep enough
        # allocator headroom for that projection and the bounded graph pool.
        # ModelRunner treats this as a whole-device ceiling and subtracts
        # global usage, including shared-host occupants. Shared admission
        # therefore needs enough ceiling for model initialization, while an
        # explicit workload-sized KV cache below prevents opportunistic
        # allocation of all remaining capacity.
        "gpu_memory_utilization": (
            0.95 if admission_mode == SHARED_CAPACITY else 0.84
        ),
        "enforce_eager": arm == "eager",
        "multi_sequence_cuda_graphs": arm == "graph",
        "multi_sequence_cuda_graph_batch_allowlist": (2, 4, 8),
        "max_num_seqs": max(8, concurrency),
        "max_model_len": max_model_len,
        "max_num_batched_tokens": (
            int(profile["prompt_tokens"]) * concurrency
        ),
    }
    if admission_mode == SHARED_CAPACITY:
        engine_config["num_kvcache_blocks"] = (
            concurrency * math.ceil(max_model_len / 256)
        )
    return engine_config


def _expected_case(case: dict) -> dict:
    matching = {
        row["case_id"]: row for row in contract.build_case_matrix()
    }.get(case.get("case_id"))
    if matching is None or case != matching:
        raise ValueError("case does not match the frozen matrix")
    return matching


def _request_specs(case: dict) -> tuple[dict, ...]:
    profile = case["profile"]
    requests = []
    for request_index in range(profile["concurrency"]):
        offset = request_index * 257
        prompt = [
            11 + ((position + offset) % 2000)
            for position in range(profile["prompt_tokens"])
        ]
        requests.append({
            "request_id": (
                f"{case['case_id']}:request-{request_index}"
            ),
            "prompt_token_ids": prompt,
            "prompt_sha256": contract.canonical_json_sha256(prompt),
        })
    return tuple(requests)


def _ranked_results(local, acknowledgements) -> list[tuple[int, object]]:
    ranked = [(0, local)]
    ranked.extend(
        (acknowledgement.rank, acknowledgement.result)
        for acknowledgement in acknowledgements
    )
    if (
        len(ranked) != len(contract.RANKS)
        or tuple(sorted(rank for rank, _ in ranked)) != contract.RANKS
        or len({rank for rank, _ in ranked}) != len(contract.RANKS)
    ):
        raise RuntimeError("graph observation rank inventory is incomplete")
    return sorted(ranked)


def collect_rank_graph_observations(
    engine,
    *,
    case_id: str,
    phase: str,
    step_index: int,
    timeout_s: float,
) -> list[dict]:
    if phase not in {"warmup", "measured"}:
        raise ValueError("phase is invalid")
    if (
        isinstance(step_index, bool)
        or not isinstance(step_index, int)
        or step_index < 0
    ):
        raise ValueError("step_index must be a non-negative integer")
    case = _expected_case(
        next(
            row
            for row in contract.build_case_matrix()
            if row["case_id"] == case_id
        )
    )
    local, acknowledgements = engine.call_model_runner_acknowledged(
        "cuda_graph_dispatch_observation",
        timeout_s=float(timeout_s),
    )
    rows = []
    for rank, observation in _ranked_results(local, acknowledgements):
        if not isinstance(observation, dict):
            raise RuntimeError("graph observation is missing")
        row = dict(observation)
        row.update({
            "row_id": (
                f"{case_id}:{phase}:step-{step_index}:rank-{rank}"
            ),
            "case_id": case_id,
            "pair_id": case["pair_id"],
            "workload": case["workload"],
            "repetition": case["repetition"],
            "arm": case["arm"],
            "phase": phase,
            "step_index": step_index,
            "rank": rank,
            "world_size": 4,
            "graph_eligible": (
                phase == "measured"
                and observation.get("mode") == "decode"
                and case["arm"] == "graph"
            ),
            "graph_replay_count": (
                int(observation.get("graph_replay_count", 1))
                if observation.get("dispatch") == "graph"
                else 0
            ),
        })
        rows.append(row)
    agreement_fields = (
        "mode",
        "active_batch_size",
        "page_table_width",
        "effective_num_splits",
        "graph_identity_sha256",
        "feature_enabled",
        "dispatch",
        "cache_state",
        "capture_attempted",
        "fallback_reason",
    )
    reference = tuple(rows[0].get(field) for field in agreement_fields)
    if any(
        tuple(row.get(field) for field in agreement_fields) != reference
        for row in rows[1:]
    ):
        raise RuntimeError("graph observations disagree across ranks")
    return rows


def _validate_cleanup(receipt: object) -> dict:
    if (
        not isinstance(receipt, dict)
        or receipt.get("process_group_destroyed") is not True
        or receipt.get("rank_exit_codes") != [0, 0, 0, 0]
        or receipt.get("owned_children_remaining") != []
        or not isinstance(receipt.get("rank_cleanup_receipts"), list)
        or sorted(
            row.get("rank")
            for row in receipt["rank_cleanup_receipts"]
        )
        != list(contract.RANKS)
        or any(
            row.get("process_group_destroyed") is not True
            for row in receipt["rank_cleanup_receipts"]
        )
    ):
        raise RuntimeError("TP4 worker cleanup receipt is incomplete")
    return dict(receipt)


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _run_request_batch(
    *,
    engine,
    case: dict,
    phase: str,
    timeout_s: float,
    sampling_params_factory,
    clock_ns,
    reset_sequence_ids,
) -> dict:
    requests = _request_specs(case)
    output_tokens = int(case["profile"]["output_tokens"])
    reset_sequence_ids()
    lifecycle = {}
    for sequence_id, request in enumerate(requests):
        admitted_ns = int(clock_ns())
        sampling = sampling_params_factory(
            temperature=0.0,
            max_tokens=output_tokens,
            ignore_eos=True,
        )
        engine.add_request(request["prompt_token_ids"], sampling)
        lifecycle[sequence_id] = {
            **request,
            "admitted_ns": admitted_ns,
            "first_token_ns": None,
            "token_timestamps_ns": [],
            "output_token_ids": [],
            "complete": False,
        }
    dispatch_rows = []
    step_index = 0
    while not engine.is_finished():
        outputs, _ = engine.step()
        observation = engine.last_step_observation
        if not isinstance(observation, dict):
            raise RuntimeError("step observation is missing")
        step_end_ns = observation.get("step_end_ns")
        token_deltas = observation.get("new_completion_tokens_by_seq")
        if (
            isinstance(step_end_ns, bool)
            or not isinstance(step_end_ns, int)
            or not isinstance(token_deltas, dict)
        ):
            raise RuntimeError("step timing observation is invalid")
        dispatch_rows.extend(
            collect_rank_graph_observations(
                engine,
                case_id=case["case_id"],
                phase=phase,
                step_index=step_index,
                timeout_s=timeout_s,
            )
        )
        for raw_sequence_id, tokens in token_deltas.items():
            sequence_id = int(raw_sequence_id)
            row = lifecycle.get(sequence_id)
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
                raise RuntimeError("token delta observation is invalid")
            if tokens:
                if row["first_token_ns"] is None:
                    row["first_token_ns"] = step_end_ns
                row["token_timestamps_ns"].extend(
                    [step_end_ns] * len(tokens)
                )
                row["output_token_ids"].extend(tokens)
        for raw_sequence_id, output in outputs:
            sequence_id = int(raw_sequence_id)
            row = lifecycle.get(sequence_id)
            if row is None or list(output) != row["output_token_ids"]:
                raise RuntimeError("terminal token output mismatch")
            row["complete"] = True
        step_index += 1
    request_rows = []
    for sequence_id in range(len(requests)):
        row = lifecycle[sequence_id]
        timestamps = row["token_timestamps_ns"]
        if (
            row["complete"] is not True
            or row["first_token_ns"] is None
            or len(row["output_token_ids"]) != output_tokens
            or len(timestamps) != output_tokens
        ):
            raise RuntimeError("request completion evidence is incomplete")
        decode_intervals = [
            current - previous
            for previous, current in zip(timestamps, timestamps[1:])
        ]
        ttft_ns = row["first_token_ns"] - row["admitted_ns"]
        e2e_ns = timestamps[-1] - row["admitted_ns"]
        request_rows.append({
            "row_id": (
                f"{case['case_id']}:{phase}:request-{sequence_id}"
            ),
            "case_id": case["case_id"],
            "pair_id": case["pair_id"],
            "workload": case["workload"],
            "repetition": case["repetition"],
            "arm": case["arm"],
            "phase": phase,
            "request_id": row["request_id"],
            "prompt_sha256": row["prompt_sha256"],
            "prompt_tokens": case["profile"]["prompt_tokens"],
            "generated_tokens": output_tokens,
            "output_token_ids": list(row["output_token_ids"]),
            "output_length": len(row["output_token_ids"]),
            "stop_reason": "length",
            "ttft_ns": ttft_ns,
            "tpot_ns": (
                sum(decode_intervals) / len(decode_intervals)
                if decode_intervals
                else 0.0
            ),
            "e2e_ns": e2e_ns,
            "admitted_ns": row["admitted_ns"],
            "completed_ns": timestamps[-1],
        })
    return {
        "request_rows": request_rows,
        "rank_dispatch_rows": dispatch_rows,
    }


def _collective_rows(profile: dict, case: dict) -> list[dict]:
    if (
        not isinstance(profile, dict)
        or profile.get("enabled") is not True
        or profile.get("rank_inventory") != list(contract.RANKS)
        or not isinstance(profile.get("ranks"), list)
    ):
        raise RuntimeError("decode internal profile is incomplete")
    rows = []
    for rank_row in profile["ranks"]:
        rank = rank_row.get("rank")
        if (
            rank not in contract.RANKS
            or rank_row.get("finalization_status") != "complete"
            or not isinstance(rank_row.get("collectives"), list)
        ):
            raise RuntimeError("decode collective profile is invalid")
        signature = []
        for collective in rank_row["collectives"]:
            signature.append({
                "step_index": collective.get("step_index"),
                "operation_ordinal": collective.get(
                    "operation_ordinal"
                ),
                "collective_kind": collective.get("collective_kind"),
                "tensor_shape": collective.get("tensor_shape"),
                "tensor_dtype": collective.get("tensor_dtype"),
            })
        rows.append({
            "row_id": f"{case['case_id']}:collectives:rank-{rank}",
            "case_id": case["case_id"],
            "pair_id": case["pair_id"],
            "workload": case["workload"],
            "repetition": case["repetition"],
            "arm": case["arm"],
            "rank": rank,
            "world_size": 4,
            "collective_count": len(signature),
            "collective_order_sha256": (
                contract.canonical_json_sha256(signature)
            ),
            "complete": True,
        })
    if sorted(row["rank"] for row in rows) != list(contract.RANKS):
        raise RuntimeError("decode collective rank inventory mismatch")
    return sorted(rows, key=lambda row: row["rank"])


def _memory_rows(snapshots, case: dict) -> list[dict]:
    rows = []
    for snapshot in snapshots:
        rank = snapshot.get("rank")
        if rank not in contract.RANKS:
            raise RuntimeError("memory rank inventory mismatch")
        rows.append({
            "row_id": f"{case['case_id']}:memory:rank-{rank}",
            "case_id": case["case_id"],
            "pair_id": case["pair_id"],
            "workload": case["workload"],
            "repetition": case["repetition"],
            "arm": case["arm"],
            "rank": rank,
            "peak_allocated_bytes": int(
                snapshot["cuda_peak_allocated_bytes"]
            ),
            "peak_reserved_bytes": int(
                snapshot["cuda_peak_reserved_bytes"]
            ),
        })
    if sorted(row["rank"] for row in rows) != list(contract.RANKS):
        raise RuntimeError("memory rank inventory mismatch")
    return sorted(rows, key=lambda row: row["rank"])


def _performance_row(case: dict, request_rows: list[dict]) -> dict:
    ttft_ms = [row["ttft_ns"] / 1_000_000 for row in request_rows]
    tpot_ms = [row["tpot_ns"] / 1_000_000 for row in request_rows]
    e2e_ms = [row["e2e_ns"] / 1_000_000 for row in request_rows]
    started = min(row["admitted_ns"] for row in request_rows)
    completed = max(row["completed_ns"] for row in request_rows)
    duration_s = max(1, completed - started) / 1_000_000_000
    generated = sum(row["generated_tokens"] for row in request_rows)
    return {
        "row_id": f"{case['case_id']}:performance",
        "case_id": case["case_id"],
        "pair_id": case["pair_id"],
        "workload": case["workload"],
        "repetition": case["repetition"],
        "arm": case["arm"],
        "output_tokens_per_second": generated / duration_s,
        "qps": len(request_rows) / duration_s,
        "median_tpot_ms": statistics.median(tpot_ms),
        "p95_tpot_ms": _nearest_rank(tpot_ms, 0.95),
        "p99_tpot_ms": _nearest_rank(tpot_ms, 0.99),
        "median_e2e_ms": statistics.median(e2e_ms),
        "p99_e2e_ms": _nearest_rank(e2e_ms, 0.99),
        "ttft_ms": statistics.median(ttft_ms),
        "initialization_ms": 0.0,
    }


def _capture_cost_rows(
    dispatch_rows: list[dict],
    case: dict,
) -> list[dict]:
    by_rank = {}
    for row in dispatch_rows:
        if int(row.get("capture_duration_ns", 0)) <= 0:
            continue
        by_rank[row["rank"]] = {
            "row_id": (
                f"{case['case_id']}:capture:rank-{row['rank']}"
            ),
            "case_id": case["case_id"],
            "pair_id": case["pair_id"],
            "workload": case["workload"],
            "repetition": case["repetition"],
            "arm": case["arm"],
            "rank": row["rank"],
            "graph_identity_sha256": row["graph_identity_sha256"],
            "capture_duration_ns": int(row["capture_duration_ns"]),
            "static_bytes": int(row["capture_static_bytes"]),
            "allocated_delta_bytes": int(
                row["capture_allocated_delta_bytes"]
            ),
            "reserved_delta_bytes": int(
                row["capture_reserved_delta_bytes"]
            ),
            "complete": True,
        }
    return [by_rank[rank] for rank in sorted(by_rank)]


def run_arm(
    *,
    model_root: Path,
    case: dict,
    output_dir: Path,
    timeout_s: float = 600.0,
    engine_factory=_default_engine_factory,
    sampling_params_factory=_default_sampling_params_factory,
    clock_ns=time.monotonic_ns,
    reset_sequence_ids=_reset_sequence_ids,
) -> dict:
    case = _expected_case(dict(case))
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or timeout_s <= 0
    ):
        raise ValueError("timeout_s must be finite and positive")
    engine = None
    cleanup = None
    initialization_started_ns = int(clock_ns())
    try:
        engine, _rendezvous_port = create_engine_with_rendezvous_retry(
            Path(model_root),
            engine_config=build_engine_config(
                arm=case["arm"],
                workload=case["workload"],
            ),
            port_factory=_free_rendezvous_port,
            engine_factory=engine_factory,
        )
        initialization_finished_ns = int(clock_ns())
        if (
            getattr(getattr(engine, "model_runner", None), "rank", None)
            != 0
            or getattr(
                getattr(engine, "model_runner", None),
                "world_size",
                None,
            )
            != 4
        ):
            raise RuntimeError("TP4 engine ownership mismatch")
        profile_label = f"{case['case_id']}:warmup-and-measured"
        configured = engine.configure_decode_internal_profile(
            True,
            profile_label,
            timeout_s=float(timeout_s),
        )
        if configured != {
            "enabled": True,
            "rank_inventory": list(contract.RANKS),
        }:
            raise RuntimeError("decode internal profile setup mismatch")
        warmup = _run_request_batch(
            engine=engine,
            case=case,
            phase="warmup",
            timeout_s=timeout_s,
            sampling_params_factory=sampling_params_factory,
            clock_ns=clock_ns,
            reset_sequence_ids=reset_sequence_ids,
        )
        engine.clear_reusable_prefix_cache()
        engine.reset_decode_internal_profile(timeout_s=float(timeout_s))
        engine.reset_peak_memory_stats(timeout_s=float(timeout_s))
        measured = _run_request_batch(
            engine=engine,
            case=case,
            phase="measured",
            timeout_s=timeout_s,
            sampling_params_factory=sampling_params_factory,
            clock_ns=clock_ns,
            reset_sequence_ids=reset_sequence_ids,
        )
        profile = engine.finalize_decode_internal_profile(
            timeout_s=float(timeout_s),
        )
        memory_rows = _memory_rows(
            engine.memory_snapshots(timeout_s=float(timeout_s)),
            case,
        )
    finally:
        if engine is not None:
            cleanup = _validate_cleanup(engine.exit())

    performance = _performance_row(case, measured["request_rows"])
    performance["initialization_ms"] = (
        initialization_finished_ns - initialization_started_ns
    ) / 1_000_000
    lifecycle_rows = [
        {
            "row_id": f"{case['case_id']}:lifecycle:rank-{rank}",
            "case_id": case["case_id"],
            "pair_id": case["pair_id"],
            "workload": case["workload"],
            "repetition": case["repetition"],
            "arm": case["arm"],
            "rank": rank,
            "world_size": 4,
            "complete": True,
            "exit_code": cleanup["rank_exit_codes"][rank],
            "process_group_destroyed": (
                cleanup["rank_cleanup_receipts"][rank][
                    "process_group_destroyed"
                ]
            ),
            "replay_exception": False,
        }
        for rank in contract.RANKS
    ]
    result = {
        "schema_version": WORKER_SCHEMA,
        "case_id": case["case_id"],
        "pair_id": case["pair_id"],
        "workload": case["workload"],
        "repetition": case["repetition"],
        "arm": case["arm"],
        "request_rows": measured["request_rows"],
        "performance_rows": [performance],
        "rank_dispatch_rows": (
            warmup["rank_dispatch_rows"]
            + measured["rank_dispatch_rows"]
        ),
        "rank_collective_rows": _collective_rows(profile, case),
        "rank_lifecycle_rows": lifecycle_rows,
        "memory_rows": memory_rows,
        "capture_cost_rows": _capture_cost_rows(
            measured["rank_dispatch_rows"],
            case,
        ),
        "cleanup": cleanup,
    }
    _atomic_write_json(
        Path(output_dir) / f"{case['case_id']}.json",
        result,
    )
    return result


def run_pair(
    *,
    model_root: Path,
    pair_cases: tuple[dict, dict],
    output_dir: Path,
    timeout_s: float = 600.0,
    engine_factory=_default_engine_factory,
    sampling_params_factory=_default_sampling_params_factory,
    clock_ns=time.monotonic_ns,
    reset_sequence_ids=_reset_sequence_ids,
) -> dict:
    ordered = _ordered_pair_cases(pair_cases)
    results = [
        run_arm(
            model_root=model_root,
            case=case,
            output_dir=output_dir,
            timeout_s=timeout_s,
            engine_factory=engine_factory,
            sampling_params_factory=sampling_params_factory,
            clock_ns=clock_ns,
            reset_sequence_ids=reset_sequence_ids,
        )
        for case in ordered
    ]
    return assemble_pair_result(
        pair_cases=ordered,
        arm_results=results,
    )


def _ordered_pair_cases(
    pair_cases: tuple[dict, dict],
) -> tuple[dict, dict]:
    if len(pair_cases) != 2:
        raise ValueError("pair_cases must contain exactly two arms")
    ordered = tuple(
        sorted(pair_cases, key=lambda row: row["order_index"])
    )
    if (
        {row["arm"] for row in ordered} != set(contract.ARMS)
        or ordered[0]["pair_id"] != ordered[1]["pair_id"]
    ):
        raise ValueError("pair_cases do not form one eager/graph pair")
    return ordered


def assemble_pair_result(
    *,
    pair_cases: tuple[dict, dict],
    arm_results: list[dict],
) -> dict:
    ordered = _ordered_pair_cases(pair_cases)
    if (
        len(arm_results) != 2
        or {result.get("arm") for result in arm_results}
        != set(contract.ARMS)
    ):
        raise ValueError("arm_results do not form one eager/graph pair")
    by_arm = {result["arm"]: result for result in arm_results}
    if any(
        result.get("pair_id") != ordered[0]["pair_id"]
        for result in arm_results
    ):
        raise ValueError("arm_results pair identity mismatch")
    eager_requests = {
        row["request_id"].split(":request-", 1)[1]: row
        for row in by_arm["eager"]["request_rows"]
    }
    graph_requests = {
        row["request_id"].split(":request-", 1)[1]: row
        for row in by_arm["graph"]["request_rows"]
    }
    eager_outputs = []
    graph_outputs = []
    exact_match = set(eager_requests) == set(graph_requests)
    for request_index in sorted(eager_requests, key=int):
        eager = eager_requests[request_index]
        graph = graph_requests.get(request_index)
        eager_output = {
            "request_id": (
                f"{ordered[0]['pair_id']}:request-{request_index}"
            ),
            "prompt_sha256": eager["prompt_sha256"],
            "output_token_ids": eager["output_token_ids"],
            "output_length": eager["output_length"],
            "stop_reason": eager["stop_reason"],
        }
        eager_outputs.append(eager_output)
        if graph is None:
            exact_match = False
            continue
        graph_output = {
            "request_id": (
                f"{ordered[0]['pair_id']}:request-{request_index}"
            ),
            "prompt_sha256": graph["prompt_sha256"],
            "output_token_ids": graph["output_token_ids"],
            "output_length": graph["output_length"],
            "stop_reason": graph["stop_reason"],
        }
        graph_outputs.append(graph_output)
        if (
            graph["prompt_sha256"],
            graph["output_token_ids"],
            graph["output_length"],
            graph["stop_reason"],
        ) != (
            eager["prompt_sha256"],
            eager["output_token_ids"],
            eager["output_length"],
            eager["stop_reason"],
        ):
            exact_match = False
    correctness = {
        "row_id": f"{ordered[0]['pair_id']}:correctness",
        "pair_id": ordered[0]["pair_id"],
        "workload": ordered[0]["workload"],
        "repetition": ordered[0]["repetition"],
        "eager_outputs": eager_outputs,
        "graph_outputs": graph_outputs,
        "exact_match": exact_match,
    }
    return {
        "pair_id": ordered[0]["pair_id"],
        "arm_results": [by_arm[case["arm"]] for case in ordered],
        "correctness_row": correctness,
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--case-json", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    case = json.loads(args.case_json)
    result = run_arm(
        model_root=args.model_root,
        case=case,
        output_dir=args.output_dir,
        timeout_s=args.timeout_s,
    )
    print(json.dumps({
        "schema_version": WORKER_SCHEMA,
        "case_id": result["case_id"],
        "classification": "PASS",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
