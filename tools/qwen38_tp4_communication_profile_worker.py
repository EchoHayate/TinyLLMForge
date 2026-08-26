#!/usr/bin/env python3
"""Run source-bound Qwen3.8 TP4 communication-profile workloads."""

from __future__ import annotations

import argparse
from itertools import count
import json
import math
import os
from pathlib import Path
import tempfile
import time


WORKLOADS = {
    "P0": ("causal", 256, 128, 1),
    "P1": ("causal", 2048, 128, 1),
    "Q0": ("online", 256, 128, 4),
    "Q1": ("online", 256, 128, 8),
    "Q2": ("online", 2048, 128, 4),
}
PHASES = {"warmup", "measured", "nsys_replay"}
RANKS = (0, 1, 2, 3)
WORKER_SCHEMA = "qwen38.tp4-communication-profile-worker.v1"


def _positive_integer(value, name):
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
        os.fsync(handle.fileno())
    temporary.replace(path)


def build_request_specs(
    *,
    prompt_tokens: int,
    output_tokens: int,
    concurrency: int,
) -> tuple[dict, ...]:
    prompt_tokens = _positive_integer(prompt_tokens, "prompt_tokens")
    output_tokens = _positive_integer(output_tokens, "output_tokens")
    concurrency = _positive_integer(concurrency, "concurrency")
    requests = []
    for request_index in range(concurrency):
        offset = request_index * 257
        prompt = [
            11 + ((position + offset) % 2000)
            for position in range(prompt_tokens)
        ]
        requests.append({
            "request_id": f"request-{request_index}",
            "prompt_token_ids": prompt,
            "output_tokens": output_tokens,
        })
    return tuple(requests)


def _default_engine_factory(model_root, **kwargs):
    from tinyvllm.engine.llm_engine import LLMEngine

    return LLMEngine(str(model_root), **kwargs)


def _default_sampling_params_factory(**kwargs):
    from tinyvllm.sampling_params import SamplingParams

    return SamplingParams(**kwargs)


def _reset_sequence_ids() -> None:
    from tinyvllm.engine.sequence import Sequence

    Sequence.counter = count()


def _validate_cleanup(receipt):
    if (
        not isinstance(receipt, dict)
        or receipt.get("process_group_destroyed") is not True
        or receipt.get("rank_exit_codes") != [0, 0, 0, 0]
        or receipt.get("owned_children_remaining") != []
    ):
        raise RuntimeError("TP4 worker cleanup receipt is incomplete")
    return dict(receipt)


def _validate_profile(profile):
    if (
        not isinstance(profile, dict)
        or profile.get("enabled") is not True
        or profile.get("rank_inventory") != list(RANKS)
        or not isinstance(profile.get("ranks"), list)
        or [row.get("rank") for row in profile["ranks"]] != list(RANKS)
        or any(
            row.get("finalization_status") != "complete"
            for row in profile["ranks"]
        )
    ):
        raise RuntimeError("decode internal profile is incomplete")
    return profile


def _validate_memory(rows):
    if (
        not isinstance(rows, (tuple, list))
        or [row.get("rank") for row in rows] != list(RANKS)
    ):
        raise RuntimeError("memory snapshot rank inventory mismatch")
    result = []
    for rank, row in enumerate(rows):
        allocated = row.get("cuda_peak_allocated_bytes")
        reserved = row.get("cuda_peak_reserved_bytes")
        if (
            isinstance(allocated, bool)
            or not isinstance(allocated, int)
            or allocated < 0
            or isinstance(reserved, bool)
            or not isinstance(reserved, int)
            or reserved < 0
        ):
            raise RuntimeError("memory snapshot is invalid")
        result.append({
            "rank": rank,
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
        })
    return result


def _tensor_nbytes(row):
    shape = row.get("tensor_shape")
    dtype = row.get("tensor_dtype")
    widths = {
        "torch.bfloat16": 2,
        "torch.float16": 2,
        "torch.float32": 4,
        "torch.int32": 4,
        "torch.int64": 8,
        "torch.uint8": 1,
        "torch.int8": 1,
    }
    if (
        not isinstance(shape, list)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in shape
        )
        or dtype not in widths
    ):
        raise RuntimeError("collective tensor identity is invalid")
    elements = 1
    for value in shape:
        elements *= value
    return elements * widths[dtype]


def compact_internal_profile(profile):
    profile = _validate_profile(profile)
    compact_ranks = []
    for rank_row in profile["ranks"]:
        rank = rank_row["rank"]
        decode_steps = {
            step["step_index"]: step
            for step in rank_row.get("steps", [])
            if step.get("is_decode") is True
        }
        layers_by_step = {}
        layer_keys = set()
        for layer in rank_row.get("layers", []):
            if layer.get("decode_ordinal") is None:
                continue
            key = (
                layer.get("step_index"),
                layer.get("layer_index"),
                layer.get("layer_role"),
            )
            if (
                key[0] not in decode_steps
                or isinstance(key[1], bool)
                or not isinstance(key[1], int)
                or key[1] < 0
                or not isinstance(key[2], str)
                or not key[2]
                or key in layer_keys
            ):
                raise RuntimeError("profile layer identity is invalid")
            layer_keys.add(key)
            layers_by_step.setdefault(key[0], []).append(layer)
        operations_by_layer = {}
        operation_keys = set()
        for operation in rank_row.get("operations", []):
            if operation.get("decode_ordinal") is None:
                continue
            layer_key = (
                operation.get("step_index"),
                operation.get("layer_index"),
                operation.get("layer_role"),
            )
            ordinal = operation.get("operation_ordinal")
            key = (layer_key[0], ordinal)
            if (
                layer_key not in layer_keys
                or isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or ordinal < 0
                or key in operation_keys
            ):
                raise RuntimeError(
                    "profile operation has no unique layer ownership"
                )
            operation_keys.add(key)
            operations_by_layer.setdefault(layer_key, []).append(operation)
        collectives = {}
        for collective in rank_row.get("collectives", []):
            if collective.get("decode_ordinal") is None:
                continue
            key = (
                collective.get("step_index"),
                collective.get("operation_ordinal"),
            )
            if key in collectives or key not in operation_keys:
                raise RuntimeError("profile collective identity is invalid")
            collectives[key] = collective
        steps = []
        for step_index, step in sorted(decode_steps.items()):
            layers = []
            for layer in sorted(
                layers_by_step.get(step_index, []),
                key=lambda row: (
                    row["layer_index"],
                    row["layer_role"],
                ),
            ):
                layer_key = (
                    step_index,
                    layer["layer_index"],
                    layer["layer_role"],
                )
                operations = sorted(
                    operations_by_layer.get(layer_key, []),
                    key=lambda row: row["operation_ordinal"],
                )
                if not operations:
                    raise RuntimeError(
                        "profile layer has no operation inventory"
                    )
                operation_inventory = [
                    [
                        operation["operation_ordinal"],
                        operation["operation_class"],
                        operation["operation_name"],
                    ]
                    for operation in operations
                ]
                byte_inventory = []
                for operation in operations:
                    if operation["operation_class"] != "collective":
                        continue
                    collective = collectives.get((
                        step_index,
                        operation["operation_ordinal"],
                    ))
                    if collective is None:
                        raise RuntimeError(
                            "profile collective metadata is missing"
                        )
                    byte_inventory.append([
                        operation["operation_ordinal"],
                        _tensor_nbytes(collective),
                    ])
                collective_ns = sum(
                    operation["cuda_ns"]
                    for operation in operations
                    if operation["operation_class"] == "collective"
                )
                compute_ns = sum(
                    operation["cuda_ns"]
                    for operation in operations
                    if operation["operation_class"] in {
                        "gemm",
                        "attention",
                        "recurrent",
                        "normalization",
                        "other_compute",
                    }
                )
                step_cuda_ns = step["cuda_ns"]
                layers.append({
                    "layer_index": layer["layer_index"],
                    "layer_role": layer["layer_role"],
                    "operation_inventory": operation_inventory,
                    "step_critical_interval_ns": step_cuda_ns,
                    "gemm_ns": sum(
                        operation["cuda_ns"]
                        for operation in operations
                        if operation["operation_class"] == "gemm"
                    ),
                    "collective_ns": collective_ns,
                    "compute_ns": compute_ns,
                    "exposed_collective_ns": collective_ns,
                    "compute_collective_overlap_ns": 0,
                    "gpu_idle_ns": max(
                        0,
                        step_cuda_ns - compute_ns - collective_ns,
                    ),
                    "collective_count": len(byte_inventory),
                    "collective_bytes": sum(
                        row[1] for row in byte_inventory
                    ),
                    "collective_byte_inventory": byte_inventory,
                    "critical_path_ns": min(
                        layer["cuda_ns"],
                        step_cuda_ns,
                    ),
                    "cpu_global_tids": [],
                    "stream_ids": [],
                })
            if not layers:
                raise RuntimeError("profile decode step has no layers")
            steps.append({
                "request_set_sha256": step["request_set_sha256"],
                "decode_ordinal": step["decode_ordinal"],
                "critical_rank": rank,
                "final_required_offset_ns": step["cuda_ns"],
                "layers": layers,
                "_cuda_ns": step["cuda_ns"],
            })
        if not steps:
            raise RuntimeError("profile rank has no decode steps")
        compact_ranks.append({
            "rank": rank,
            "enabled": True,
            "finalization_status": "complete",
            "steps": steps,
        })
    reference = compact_ranks[0]["steps"]
    for step_index, reference_step in enumerate(reference):
        aligned = [row["steps"][step_index] for row in compact_ranks]
        signature = (
            reference_step["request_set_sha256"],
            reference_step["decode_ordinal"],
        )
        if any(
            (
                step["request_set_sha256"],
                step["decode_ordinal"],
            )
            != signature
            for step in aligned
        ):
            raise RuntimeError("profile rank step alignment mismatch")
        critical_rank = max(
            RANKS,
            key=lambda candidate: (
                aligned[candidate]["_cuda_ns"],
                candidate,
            ),
        )
        critical_interval_ns = aligned[critical_rank]["_cuda_ns"]
        for step in aligned:
            step["critical_rank"] = critical_rank
            step["final_required_offset_ns"] = step["_cuda_ns"]
            step.pop("_cuda_ns")
            for layer in step["layers"]:
                layer["step_critical_interval_ns"] = critical_interval_ns
    return {
        "enabled": True,
        "rank_inventory": list(RANKS),
        "ranks": compact_ranks,
    }


def _validate_case_identity(
    *,
    attempt: str,
    workload: str,
    workload_family: str,
    phase: str,
    repetition: int,
    prompt_tokens: int,
    output_tokens: int,
    concurrency: int,
    timeout_s: float,
) -> None:
    expected = WORKLOADS.get(workload)
    if expected is not None and expected != (
        workload_family,
        prompt_tokens,
        output_tokens,
        concurrency,
    ):
        raise ValueError("workload contract mismatch")
    if phase not in PHASES:
        raise ValueError("phase is invalid")
    if (
        not isinstance(attempt, str)
        or not attempt
        or isinstance(repetition, bool)
        or not isinstance(repetition, int)
        or repetition < 0
        or isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or timeout_s <= 0
    ):
        raise ValueError("worker identity or timeout is invalid")


def _validate_engine_ownership(engine) -> None:
    if (
        getattr(getattr(engine, "model_runner", None), "rank", None) != 0
        or getattr(
            getattr(engine, "model_runner", None),
            "world_size",
            None,
        )
        != 4
    ):
        raise RuntimeError("TP4 engine ownership mismatch")


def _run_profile_case_with_engine(
    *,
    engine,
    attempt: str,
    workload: str,
    workload_family: str,
    phase: str,
    repetition: int,
    prompt_tokens: int,
    output_tokens: int,
    concurrency: int,
    timeout_s: float,
    sampling_params_factory,
    clock_ns,
    reset_sequence_ids,
) -> dict:
    _validate_case_identity(
        attempt=attempt,
        workload=workload,
        workload_family=workload_family,
        phase=phase,
        repetition=repetition,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        concurrency=concurrency,
        timeout_s=timeout_s,
    )
    requests = build_request_specs(
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        concurrency=concurrency,
    )
    reset_sequence_ids()
    _validate_engine_ownership(engine)
    profile_label = (
        f"attempt={attempt}/workload={workload}/"
        f"repetition={repetition}"
    )
    configured = engine.configure_decode_internal_profile(
        True,
        profile_label,
        timeout_s=float(timeout_s),
    )
    if configured != {
        "enabled": True,
        "rank_inventory": list(RANKS),
    }:
        raise RuntimeError("decode internal profile setup mismatch")
    engine.reset_peak_memory_stats(timeout_s=float(timeout_s))
    lifecycle = {}
    for seq_id, request in enumerate(requests):
        admitted_ns = clock_ns()
        sampling = sampling_params_factory(
            temperature=0.0,
            max_tokens=output_tokens,
            ignore_eos=True,
        )
        engine.add_request(request["prompt_token_ids"], sampling)
        lifecycle[seq_id] = {
            "request_id": request["request_id"],
            "admitted_ns": admitted_ns,
            "first_token_ns": None,
            "token_timestamps_ns": [],
            "output_token_ids": [],
            "complete": False,
        }
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
                raise RuntimeError("token delta observation is invalid")
            if tokens:
                if row["first_token_ns"] is None:
                    row["first_token_ns"] = step_end_ns
                row["token_timestamps_ns"].extend(
                    [step_end_ns] * len(tokens)
                )
                row["output_token_ids"].extend(tokens)
        for raw_seq_id, output in outputs:
            seq_id = int(raw_seq_id)
            row = lifecycle.get(seq_id)
            if row is None or list(output) != row["output_token_ids"]:
                raise RuntimeError("terminal token output mismatch")
            row["complete"] = True
    profile = compact_internal_profile(
        engine.finalize_decode_internal_profile(
            timeout_s=float(timeout_s),
        ),
    )
    memory = _validate_memory(
        engine.memory_snapshots(timeout_s=float(timeout_s))
    )
    request_rows = []
    for seq_id in range(concurrency):
        row = lifecycle[seq_id]
        timestamps = row["token_timestamps_ns"]
        if (
            row["complete"] is not True
            or row["first_token_ns"] is None
            or len(row["output_token_ids"]) != output_tokens
            or len(timestamps) != output_tokens
        ):
            raise RuntimeError("request completion evidence is incomplete")
        ttft_ns = row["first_token_ns"] - row["admitted_ns"]
        e2e_ns = timestamps[-1] - row["admitted_ns"]
        decode_intervals = [
            current - previous
            for previous, current in zip(timestamps, timestamps[1:])
        ]
        request_rows.append({
            "request_id": row["request_id"],
            "prompt_tokens": prompt_tokens,
            "generated_tokens": output_tokens,
            "ttft_ns": ttft_ns,
            "tpot_ns": (
                sum(decode_intervals) / len(decode_intervals)
                if decode_intervals
                else 0.0
            ),
            "e2e_ns": e2e_ns,
            "output_token_ids": list(row["output_token_ids"]),
        })
    decode_time_ns = max(
        row["e2e_ns"] - row["ttft_ns"]
        for row in request_rows
    )
    return {
        "schema_version": WORKER_SCHEMA,
        "classification": "PASS",
        "case_id": f"{workload}__{phase}__r{repetition}",
        "attempt": attempt,
        "workload": workload,
        "workload_family": workload_family,
        "phase": phase,
        "repetition": repetition,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "concurrency": concurrency,
        "rank_inventory": list(RANKS),
        "pid": os.getpid(),
        "decode_time_ns": max(1, int(decode_time_ns)),
        "requests": request_rows,
        "profile": profile,
        "memory": memory,
    }


def run_profile_case(
    *,
    attempt: str,
    workload: str,
    workload_family: str,
    phase: str,
    repetition: int,
    prompt_tokens: int,
    output_tokens: int,
    concurrency: int,
    model_root: Path,
    timeout_s: float,
    engine_factory=_default_engine_factory,
    sampling_params_factory=_default_sampling_params_factory,
    clock_ns=time.monotonic_ns,
    reset_sequence_ids=_reset_sequence_ids,
) -> dict:
    _validate_case_identity(
        attempt=attempt,
        workload=workload,
        workload_family=workload_family,
        phase=phase,
        repetition=repetition,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        concurrency=concurrency,
        timeout_s=timeout_s,
    )
    engine = None
    cleanup = None
    try:
        engine = engine_factory(
            Path(model_root),
            tensor_parallel_size=4,
            enforce_eager=True,
            max_num_seqs=max(8, concurrency),
            max_model_len=prompt_tokens + output_tokens,
            max_num_batched_tokens=prompt_tokens * concurrency,
        )
        result = _run_profile_case_with_engine(
            engine=engine,
            attempt=attempt,
            workload=workload,
            workload_family=workload_family,
            phase=phase,
            repetition=repetition,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            concurrency=concurrency,
            timeout_s=timeout_s,
            sampling_params_factory=sampling_params_factory,
            clock_ns=clock_ns,
            reset_sequence_ids=reset_sequence_ids,
        )
    finally:
        if engine is not None:
            cleanup = _validate_cleanup(engine.exit())
    result["cleanup"] = cleanup
    return result


def run_profile_campaign(
    *,
    attempt: str,
    cases,
    model_root: Path,
    timeout_s: float,
    engine_factory=_default_engine_factory,
    sampling_params_factory=_default_sampling_params_factory,
    clock_ns=time.monotonic_ns,
    reset_sequence_ids=_reset_sequence_ids,
    case_sink=None,
    retain_case_profiles=True,
) -> dict:
    cases = tuple(dict(case) for case in cases)
    if not cases:
        raise ValueError("campaign cases must not be empty")
    if case_sink is not None and not callable(case_sink):
        raise ValueError("case_sink must be callable")
    if not isinstance(retain_case_profiles, bool):
        raise ValueError("retain_case_profiles must be a bool")
    for case in cases:
        _validate_case_identity(
            attempt=attempt,
            timeout_s=timeout_s,
            **case,
        )
    max_concurrency = max(case["concurrency"] for case in cases)
    engine = None
    cleanup = None
    try:
        engine = engine_factory(
            Path(model_root),
            tensor_parallel_size=4,
            enforce_eager=True,
            max_num_seqs=max(8, max_concurrency),
            max_model_len=max(
                case["prompt_tokens"] + case["output_tokens"]
                for case in cases
            ),
            max_num_batched_tokens=max(
                case["prompt_tokens"] * case["concurrency"]
                for case in cases
            ),
        )
        _validate_engine_ownership(engine)
        results = []
        for case in cases:
            case_result = _run_profile_case_with_engine(
                engine=engine,
                attempt=attempt,
                timeout_s=timeout_s,
                sampling_params_factory=sampling_params_factory,
                clock_ns=clock_ns,
                reset_sequence_ids=reset_sequence_ids,
                **case,
            )
            if case_sink is not None:
                case_sink(case_result)
            results.append(
                case_result
                if retain_case_profiles
                else {
                    "classification": case_result["classification"],
                    "case_id": case_result["case_id"],
                    "decode_time_ns": case_result["decode_time_ns"],
                }
            )
    finally:
        if engine is not None:
            cleanup = _validate_cleanup(engine.exit())
    return {
        "schema_version": WORKER_SCHEMA,
        "classification": "PASS",
        "attempt": attempt,
        "cases": results,
        "cleanup": cleanup,
    }


def build_structured_cases() -> tuple[dict, ...]:
    cases = []
    for workload, (
        workload_family,
        prompt_tokens,
        output_tokens,
        concurrency,
    ) in WORKLOADS.items():
        for phase, repetitions in (
            ("warmup", range(2)),
            ("measured", range(5)),
        ):
            for repetition in repetitions:
                cases.append({
                    "workload": workload,
                    "workload_family": workload_family,
                    "phase": phase,
                    "repetition": repetition,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "concurrency": concurrency,
                })
    return tuple(cases)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--workload", choices=tuple(WORKLOADS))
    parser.add_argument("--phase", choices=tuple(sorted(PHASES)))
    parser.add_argument("--repetition", type=int)
    parser.add_argument("--structured-campaign", action="store_true")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    args = parser.parse_args(argv)
    if args.structured_campaign:
        if (
            args.output_dir is None
            or args.workload is not None
            or args.phase is not None
            or args.repetition is not None
        ):
            parser.error(
                "--structured-campaign requires --output-dir and excludes "
                "--workload/--phase/--repetition"
            )

        def write_case(case_result):
            _atomic_write_json(
                args.output_dir / f"{case_result['case_id']}.json",
                case_result,
            )

        result = run_profile_campaign(
            attempt=args.attempt,
            cases=build_structured_cases(),
            model_root=args.model_root,
            timeout_s=args.timeout_s,
            engine_factory=_default_engine_factory,
            sampling_params_factory=_default_sampling_params_factory,
            clock_ns=time.monotonic_ns,
            reset_sequence_ids=_reset_sequence_ids,
            case_sink=write_case,
            retain_case_profiles=False,
        )
        _atomic_write_json(args.output, result)
        print(json.dumps({
            "classification": result["classification"],
            "case_count": len(result["cases"]),
            "output": str(args.output),
            "output_dir": str(args.output_dir),
        }, sort_keys=True))
        return 0
    if (
        args.workload is None
        or args.phase is None
        or args.repetition is None
        or args.output_dir is not None
    ):
        parser.error(
            "single-case mode requires --workload, --phase, and "
            "--repetition and excludes --output-dir"
        )
    family, prompt_tokens, output_tokens, concurrency = WORKLOADS[
        args.workload
    ]
    result = run_profile_case(
        attempt=args.attempt,
        workload=args.workload,
        workload_family=family,
        phase=args.phase,
        repetition=args.repetition,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        concurrency=concurrency,
        model_root=args.model_root,
        timeout_s=args.timeout_s,
        engine_factory=_default_engine_factory,
        sampling_params_factory=_default_sampling_params_factory,
        clock_ns=time.monotonic_ns,
        reset_sequence_ids=_reset_sequence_ids,
    )
    _atomic_write_json(args.output, result)
    print(json.dumps({
        "classification": result["classification"],
        "case_id": result["case_id"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
