from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def _load_gate_module():
    module_name = "generic_speculative_tp4_gate"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = (
        Path(__file__).resolve().parent
        / "generic_speculative_tp4_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate_module()

DEFAULT_PROMPT_SEEDS = (
    "TP4 speculative verification repeats alpha beta gamma. ",
    "Collective authority cycles red green blue amber. ",
    "Transactional residency follows north east south west. ",
    "Generic n-gram parity echoes one two three four. ",
)


def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_gpu_indices(value: object) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or len(value) != gate.WORLD_SIZE
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in value
        )
        or len(set(value)) != gate.WORLD_SIZE
    ):
        raise ValueError(
            "GPU indices must contain four distinct "
            "non-negative integers"
        )
    return value


def build_prompt_token_batches(
    tokenizer,
    *,
    batch_size: int,
) -> list[dict]:
    if batch_size not in gate.BATCH_SIZES:
        raise ValueError("unsupported batch size")
    rows = []
    for prompt_index, seed in enumerate(
        DEFAULT_PROMPT_SEEDS[:batch_size]
    ):
        encoded = tokenizer.encode(
            seed,
            add_special_tokens=False,
        )
        if (
            not isinstance(encoded, (list, tuple))
            or not encoded
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in encoded
            )
        ):
            raise ValueError(
                f"prompt seed {prompt_index} encoded invalid tokens"
            )
        repeats = (
            gate.CONTEXT_TOKENS + len(encoded) - 1
        ) // len(encoded)
        token_ids = (
            list(encoded) * repeats
        )[:gate.CONTEXT_TOKENS]
        digest = hashlib.sha256(
            json.dumps(
                token_ids,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        rows.append({
            "prompt_index": prompt_index,
            "seed": seed,
            "token_ids": token_ids,
            "token_count": len(token_ids),
            "sha256": digest,
        })
    return rows


def summarize_step_observations(
    observations: list[dict],
) -> dict[str, int]:
    if not isinstance(observations, list):
        raise ValueError("step observations must be a list")
    summary = {
        name: 0
        for name in gate.RUNTIME_KEYS
    }
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError(
                "step observation must be a mapping"
            )
        proposal_counts = observation.get(
            "speculative_proposal_token_counts",
            {},
        )
        accepted_counts = observation.get(
            "speculative_accepted_draft_token_counts",
            {},
        )
        if (
            not isinstance(proposal_counts, dict)
            or not isinstance(accepted_counts, dict)
        ):
            raise ValueError(
                "step speculative token counts are invalid"
            )
        summary["proposal_rows"] += int(
            observation.get(
                "speculative_proposal_row_count",
                0,
            )
        )
        summary["proposed_tokens"] += sum(
            int(value)
            for value in proposal_counts.values()
        )
        summary["accepted_draft_tokens"] += sum(
            int(value)
            for value in accepted_counts.values()
        )
        summary["first_target_callbacks"] += int(
            observation.get(
                "speculative_first_target_callback_count",
                0,
            )
        )
        summary["tail_callbacks"] += int(
            observation.get(
                "speculative_fixed_q_group_count",
                0,
            )
        )
    return summary


def run_generation(
    *,
    engine,
    prompt_rows,
    sampling_params,
    expected_output_tokens,
    synchronize,
):
    if not engine.is_finished():
        raise RuntimeError(
            "engine must be idle before generation"
        )
    for prompt_row in prompt_rows:
        engine.add_request(
            prompt_row["token_ids"],
            sampling_params,
        )
    outputs_by_id = {}
    observations = []
    while not engine.is_finished():
        step_outputs, _ = engine.step()
        synchronize()
        observation = getattr(
            engine,
            "last_step_observation",
            None,
        )
        if not isinstance(observation, dict):
            raise RuntimeError(
                "engine step observation is unavailable"
            )
        observations.append(dict(observation))
        for sequence_id, token_ids in step_outputs:
            outputs_by_id[int(sequence_id)] = list(token_ids)
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
        "runtime": summarize_step_observations(
            observations
        ),
    }


def _movement_delta(before_rows, after_rows):
    if (
        not isinstance(before_rows, tuple)
        or not isinstance(after_rows, tuple)
        or len(before_rows) != gate.WORLD_SIZE
        or len(after_rows) != gate.WORLD_SIZE
    ):
        raise ValueError("KV movement rank inventory mismatch")
    result = []
    for rank, (before, after) in enumerate(
        zip(before_rows, after_rows)
    ):
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError(
                "KV movement summary must be a mapping"
            )
        row = {"rank": rank}
        for name in gate.MOVEMENT_KEYS:
            before_value = before.get(name)
            after_value = after.get(name)
            if (
                isinstance(before_value, bool)
                or not isinstance(before_value, int)
                or before_value < 0
                or isinstance(after_value, bool)
                or not isinstance(after_value, int)
                or after_value < before_value
            ):
                raise ValueError(
                    f"KV movement counter {name} is invalid"
                )
            row[name] = after_value - before_value
        result.append(row)
    return result


@contextmanager
def capture_residency_phases(engine):
    captured = []
    original = engine._call_speculative_residency_phase

    def recorded(method_name, ticket_id, *args, **kwargs):
        rows = original(
            method_name,
            ticket_id,
            *args,
            **kwargs,
        )
        captured.append({
            "ticket_id": ticket_id,
            "operation": kwargs["expected_operation"],
            "status": kwargs["expected_status"],
            "rows": [
                dict(row)
                for row in rows
            ],
        })
        return rows

    engine._call_speculative_residency_phase = recorded
    try:
        yield captured
    finally:
        engine._call_speculative_residency_phase = original


@contextmanager
def distributed_environment(
    *,
    gpu_indices: tuple[int, ...],
    dist_port: int,
    master_port: int,
):
    names = (
        "CUDA_VISIBLE_DEVICES",
        "TINYVLLM_DIST_PORT",
        "MASTER_PORT",
    )
    previous = {
        name: os.environ.get(name)
        for name in names
    }
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(index) for index in gpu_indices
    )
    os.environ["TINYVLLM_DIST_PORT"] = str(dist_port)
    os.environ["MASTER_PORT"] = str(master_port)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_policy_cell(
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
    run_generation_fn=run_generation,
) -> dict:
    gpu_indices = _validate_gpu_indices(gpu_indices)
    gate.cell_key(policy, batch_size)
    _positive_integer(dist_port, "distributed port")
    _positive_integer(master_port, "master port")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("model path must be non-empty")
    for dependency, name in (
        (engine_factory, "engine factory"),
        (sampling_params_type, "sampling params type"),
        (runtime_type, "runtime type"),
        (adapter_type, "adapter type"),
        (synchronize, "synchronize"),
        (run_generation_fn, "generation runner"),
    ):
        if not callable(dependency):
            raise ValueError(f"{name} must be callable")

    engine = None
    cell = None
    cleanup_receipt = None
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
                max_model_len=4352,
                max_num_batched_tokens=16384,
                max_num_seqs=batch_size,
                max_num_prefill_tokens_per_step=1024,
                chunked_prefill_decode_first=False,
                chunked_prefill_mixed_batch=False,
                kv_offload_mvp0=True,
                kv_offload_gpu_blocks=68,
                kv_offload_logical_blocks=640,
                kv_offload_blockwise_decode=True,
                kv_offload_blockwise_prefill=True,
                kv_offload_blockwise_blocks=8,
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
            profile_configuration = (
                engine.configure_decode_internal_profile(
                    True,
                    f"generic_tp4/{policy}/b{batch_size}",
                    timeout_s=60.0,
                )
            )
            prompt_rows = build_prompt_token_batches(
                engine.tokenizer,
                batch_size=batch_size,
            )
            sampling_params = sampling_params_type(
                temperature=0.0,
                max_tokens=gate.MAX_OUTPUT_TOKENS,
                ignore_eos=True,
            )
            run_generation_fn(
                engine=engine,
                prompt_rows=prompt_rows,
                sampling_params=sampling_params,
                expected_output_tokens=(
                    gate.MAX_OUTPUT_TOKENS
                ),
                synchronize=synchronize,
            )
            engine.clear_reusable_prefix_cache()
            before_rows = engine.kv_offload_summaries(
                timeout_s=60.0
            )
            with capture_residency_phases(
                engine
            ) as residency_phases:
                recorded = run_generation_fn(
                    engine=engine,
                    prompt_rows=prompt_rows,
                    sampling_params=sampling_params,
                    expected_output_tokens=(
                        gate.MAX_OUTPUT_TOKENS
                    ),
                    synchronize=synchronize,
                )
            after_rows = engine.kv_offload_summaries(
                timeout_s=60.0
            )
            profile = (
                engine.finalize_decode_internal_profile(
                    timeout_s=60.0
                )
            )
            config = getattr(engine, "config", None)
            if config is None:
                config = getattr(
                    getattr(engine, "model_runner", None),
                    "config",
                    None,
                )
            cell = {
                "schema_version": gate.SCHEMA_VERSION,
                "classification": gate.CLASSIFICATION,
                "policy": policy,
                "context_tokens": gate.CONTEXT_TOKENS,
                "batch_size": batch_size,
                "world_size": gate.WORLD_SIZE,
                "rank_inventory": list(
                    profile_configuration[
                        "rank_inventory"
                    ]
                ),
                "ack_ranks": [1, 2, 3],
                "prompt_rows": prompt_rows,
                "outputs": recorded["outputs"],
                "runtime": recorded["runtime"],
                "kv_rank_deltas": _movement_delta(
                    before_rows,
                    after_rows,
                ),
                "residency_phases": list(
                    residency_phases
                ),
                "profile": profile,
                "tokenizer_identifier": str(
                    getattr(
                        engine.tokenizer,
                        "name_or_path",
                        type(engine.tokenizer).__name__,
                    )
                ),
                "dtype": str(
                    getattr(config, "dtype", "unknown")
                ),
            }
        finally:
            if engine is not None:
                cleanup_receipt = engine.exit()
    if cell is None:
        raise RuntimeError(
            "TP4 cell execution did not produce a result"
        )
    cell["cleanup_receipt"] = cleanup_receipt
    return gate.validate_cell_result(cell)


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
    }


def _gpu_indices(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(
            int(item)
            for item in value.split(",")
        )
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
    parser.add_argument(
        "--dist-port",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--master-port",
        required=True,
        type=int,
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_policy_cell(
        model_path=args.model,
        gpu_indices=args.gpu_indices,
        policy=args.policy,
        batch_size=args.batch_size,
        dist_port=args.dist_port,
        master_port=args.master_port,
        **_default_dependencies(),
    )
    gate.atomic_write_json(Path(args.out), result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
