from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time

import qwen35_generic_speculative_tp4_16k_performance_gate as gate


TOOLS = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_correctness_worker = _load_module(
    "_qwen35_tp4_16k_performance_correctness_worker",
    TOOLS / "qwen35_generic_speculative_tp4_16k_worker.py",
)
build_prompt_rows = _correctness_worker.build_prompt_rows
distributed_environment = _correctness_worker.distributed_environment
_cleanup_observations = _correctness_worker._cleanup_observations
_merge_cleanup_receipt = _correctness_worker._merge_cleanup_receipt


def _validate_gpu_indices(value: object) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or len(value) != gate.WORLD_SIZE
        or len(set(value)) != gate.WORLD_SIZE
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in value
        )
    ):
        raise ValueError("GPU indices must contain four unique devices")
    return value


def _validate_rank_rows(
    rows: object,
    *,
    name: str,
) -> tuple[dict, ...]:
    if (
        not isinstance(rows, tuple)
        or len(rows) != gate.WORLD_SIZE
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError(
            f"{name} must contain exactly four rank mappings"
        )
    return tuple(dict(row) for row in rows)


def movement_delta(
    before_rows: tuple[dict, ...],
    after_rows: tuple[dict, ...],
) -> dict:
    before_rows = _validate_rank_rows(
        before_rows,
        name="before movement summaries",
    )
    after_rows = _validate_rank_rows(
        after_rows,
        name="after movement summaries",
    )
    ranks = []
    totals = {key: 0 for key in gate.REAL_MOVEMENT_KEYS}
    for rank, (before, after) in enumerate(
        zip(before_rows, after_rows)
    ):
        delta = gate.subtract_counter_summaries(
            before,
            after,
            keys=gate.REAL_MOVEMENT_KEYS,
        )
        row = {"rank": rank, **delta}
        ranks.append(row)
        for key, value in delta.items():
            totals[key] += value
    return gate.validate_movement({
        "ranks": ranks,
        "totals": totals,
    })


def memory_result(
    reset_rows: tuple[dict, ...],
    final_rows: tuple[dict, ...],
) -> dict:
    reset_rows = _validate_rank_rows(
        reset_rows,
        name="peak reset rows",
    )
    final_rows = _validate_rank_rows(
        final_rows,
        name="final memory rows",
    )
    ranks = []
    for rank, (reset, final) in enumerate(
        zip(reset_rows, final_rows)
    ):
        reset = {"rank": rank, **reset}
        final = {"rank": rank, **final}
        ranks.append({
            "rank": rank,
            "reset": reset,
            "final": final,
            "peak_allocated_delta_bytes": max(
                0,
                final["cuda_peak_allocated_bytes"]
                - reset["cuda_allocated_bytes"],
            ),
            "peak_reserved_delta_bytes": max(
                0,
                final["cuda_peak_reserved_bytes"]
                - reset["cuda_reserved_bytes"],
            ),
        })
    value = {
        "ranks": ranks,
        "peak_allocated_bytes": max(
            row["final"]["cuda_peak_allocated_bytes"]
            for row in ranks
        ),
        "peak_reserved_bytes": max(
            row["final"]["cuda_peak_reserved_bytes"]
            for row in ranks
        ),
        "peak_allocated_delta_bytes": max(
            row["peak_allocated_delta_bytes"]
            for row in ranks
        ),
        "peak_reserved_delta_bytes": max(
            row["peak_reserved_delta_bytes"]
            for row in ranks
        ),
    }
    return gate.validate_memory(value)


def run_request_batch(
    *,
    engine,
    prompt_rows: list[dict],
    sampling_params,
    expected_output_tokens: int,
    synchronize,
    clock_ns,
) -> dict:
    if not engine.is_finished():
        raise RuntimeError("engine must be idle before a measured run")
    engine.clear_reusable_prefix_cache()
    before_rows = _validate_rank_rows(
        engine.kv_offload_summaries(timeout_s=60.0),
        name="before movement summaries",
    )
    reset_rows = _validate_rank_rows(
        engine.reset_peak_memory_stats(timeout_s=60.0),
        name="peak reset rows",
    )
    synchronize()
    started_ns = clock_ns()
    for prompt_row in prompt_rows:
        token_ids = prompt_row.get("token_ids")
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != gate.PROMPT_TOKENS
        ):
            raise ValueError(
                "worker prompt must contain exactly "
                f"{gate.PROMPT_TOKENS} tokens"
            )
        engine.add_request(token_ids, sampling_params)
    outputs_by_id = {}
    token_events = {}
    finished_at_ns = {}
    observations = []
    last_step_ns = started_ns
    while not engine.is_finished():
        step_outputs, _ = engine.step()
        synchronize()
        last_step_ns = clock_ns()
        observation = getattr(
            engine,
            "last_step_observation",
            None,
        )
        if not isinstance(observation, dict):
            raise RuntimeError(
                "engine step observation is unavailable"
            )
        observations.append(copy.deepcopy(observation))
        deltas = observation.get(
            "new_completion_tokens_by_seq",
            {},
        )
        if not isinstance(deltas, dict):
            raise ValueError(
                "completion token deltas must be a mapping"
            )
        for sequence_id, token_ids in deltas.items():
            if token_ids:
                token_events.setdefault(
                    int(sequence_id),
                    [],
                ).append((last_step_ns, len(token_ids)))
        for sequence_id in observation.get(
            "finished_seq_ids",
            [],
        ):
            finished_at_ns[int(sequence_id)] = last_step_ns
        for sequence_id, token_ids in step_outputs:
            outputs_by_id[int(sequence_id)] = [
                int(token_id)
                for token_id in token_ids
            ]
    after_rows = _validate_rank_rows(
        engine.kv_offload_summaries(timeout_s=60.0),
        name="after movement summaries",
    )
    final_rows = _validate_rank_rows(
        engine.memory_snapshots(timeout_s=60.0),
        name="final memory rows",
    )
    outputs = [
        outputs_by_id[sequence_id]
        for sequence_id in sorted(outputs_by_id)
    ]
    if len(outputs) != len(prompt_rows):
        raise RuntimeError(
            "engine did not return one output per prompt"
        )
    if any(
        len(token_ids) != expected_output_tokens
        for token_ids in outputs
    ):
        raise RuntimeError(
            "engine output token count does not match budget"
        )
    return {
        "outputs": outputs,
        "timing": gate.build_run_metrics(
            request_start_ns=started_ns,
            request_finish_ns=last_step_ns,
            token_events=token_events,
            finished_at_ns=finished_at_ns,
            expected_output_tokens=expected_output_tokens,
        ),
        "runtime": gate._frozen.summarize_step_observations(
            observations
        ),
        "movement": movement_delta(
            before_rows,
            after_rows,
        ),
        "memory": memory_result(
            reset_rows,
            final_rows,
        ),
        "observations": observations,
    }


def run_policy_campaign(
    *,
    model_path: str,
    gpu_indices: tuple[int, ...],
    policy: str,
    batch_size: int,
    dist_port: int,
    master_port: int,
    engine_factory,
    sampling_params_type,
    runtime_type,
    adapter_type,
    synchronize,
    clock_ns,
    run_batch_fn=run_request_batch,
    cleanup_observations_fn=_cleanup_observations,
    merge_cleanup_fn=_merge_cleanup_receipt,
) -> dict:
    gpu_indices = _validate_gpu_indices(gpu_indices)
    gate.cell_key(policy, batch_size)
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("model path must be non-empty")
    engine = None
    result = None
    cleanup_observations = None
    exit_receipt = None
    runtime_poisoned = False
    with distributed_environment(
        gpu_indices=gpu_indices,
        dist_port=dist_port,
        master_port=master_port,
    ):
        try:
            engine = engine_factory(
                model_path,
                tensor_parallel_size=gate.WORLD_SIZE,
                enforce_eager=True,
                max_model_len=gate.MAX_MODEL_LEN,
                max_num_batched_tokens=(
                    gate.MAX_NUM_BATCHED_TOKENS
                ),
                max_num_seqs=batch_size,
                max_num_prefill_tokens_per_step=(
                    gate.MAX_NUM_PREFILL_TOKENS_PER_STEP
                ),
                chunked_prefill_decode_first=False,
                chunked_prefill_mixed_batch=False,
                kv_offload_mvp0=True,
                kv_offload_gpu_blocks=(
                    gate.KV_OFFLOAD_GPU_BLOCKS
                ),
                kv_offload_logical_blocks=(
                    gate.KV_OFFLOAD_LOGICAL_BLOCKS
                ),
                kv_offload_blockwise_decode=True,
                kv_offload_blockwise_prefill=True,
                kv_offload_blockwise_blocks=(
                    gate.KV_OFFLOAD_BLOCKWISE_BLOCKS
                ),
            )
            if policy == "ngram":
                engine.activate_speculative_runtime(
                    runtime_type(
                        adapter_type(
                            ngram_size=gate.NGRAM_SIZE,
                            max_proposal_tokens=(
                                gate.MAX_PROPOSAL_TOKENS
                            ),
                        )
                    )
                )
            prompt_rows = build_prompt_rows(
                engine.tokenizer,
                batch_size,
            )
            sampling_params = sampling_params_type(
                temperature=0.0,
                max_tokens=gate.MAX_OUTPUT_TOKENS,
                ignore_eos=True,
            )

            def run_once():
                return run_batch_fn(
                    engine=engine,
                    prompt_rows=prompt_rows,
                    sampling_params=sampling_params,
                    expected_output_tokens=(
                        gate.MAX_OUTPUT_TOKENS
                    ),
                    synchronize=synchronize,
                    clock_ns=clock_ns,
                )

            result = {
                "policy": policy,
                "batch_size": batch_size,
                "prompt_rows": prompt_rows,
                "warmup_runs": [
                    run_once()
                    for _ in range(gate.WARMUP_RUNS)
                ],
                "parity_runs": [
                    run_once()
                    for _ in range(gate.PARITY_RUNS)
                ],
                "measured_runs": [
                    run_once()
                    for _ in range(gate.MEASURED_RUNS)
                ],
                "tokenizer_identifier": str(
                    getattr(
                        engine.tokenizer,
                        "name_or_path",
                        type(engine.tokenizer).__name__,
                    )
                ),
                "dtype": str(
                    getattr(
                        getattr(engine, "config", None),
                        "dtype",
                        "unknown",
                    )
                ),
            }
            cleanup_observations = cleanup_observations_fn(engine)
            runtime_poisoned = bool(
                getattr(
                    engine,
                    "speculative_runtime_poisoned",
                    False,
                )
            )
        finally:
            if engine is not None:
                exit_receipt = engine.exit()
    if result is None or cleanup_observations is None:
        raise RuntimeError(
            "TP4 performance worker did not produce a result"
        )
    result["cleanup_receipt"] = merge_cleanup_fn(
        exit_receipt,
        cleanup_observations,
        runtime_poisoned=runtime_poisoned,
    )
    return result


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _default_dependencies():
    import torch

    from tinyvllm import LLM
    from tinyvllm.engine.speculative_runtime import (
        EngineSpeculativeRuntime,
    )
    from tinyvllm.sampling_params import SamplingParams
    from tinyvllm.speculative.ngram_adapter import (
        NGramDraftAdapter,
    )

    return {
        "engine_factory": LLM,
        "sampling_params_type": SamplingParams,
        "runtime_type": EngineSpeculativeRuntime,
        "adapter_type": NGramDraftAdapter,
        "synchronize": torch.cuda.synchronize,
        "clock_ns": time.perf_counter_ns,
    }


def _gpu_indices(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "GPU indices must be comma-separated integers"
        ) from error
    try:
        return _validate_gpu_indices(parsed)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--gpu-indices",
        required=True,
        type=_gpu_indices,
    )
    parser.add_argument(
        "--policy",
        required=True,
        choices=gate.POLICIES,
    )
    parser.add_argument(
        "--batch-size",
        required=True,
        type=int,
        choices=gate.BATCH_SIZES,
    )
    parser.add_argument("--dist-port", required=True, type=int)
    parser.add_argument("--master-port", required=True, type=int)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_policy_campaign(
        model_path=args.model,
        gpu_indices=args.gpu_indices,
        policy=args.policy,
        batch_size=args.batch_size,
        dist_port=args.dist_port,
        master_port=args.master_port,
        **_default_dependencies(),
    )
    _write_json_atomic(Path(args.out), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
