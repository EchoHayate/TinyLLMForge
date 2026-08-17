from __future__ import annotations

import argparse
from pathlib import Path
import sys

from blockwise_speculative_verifier_gate import (
    BATCH_SIZES,
    BLOCKWISE_BLOCKS,
    CLASSIFICATION,
    CONTEXT_TOKENS,
    GPU_BLOCKS,
    LOGICAL_BLOCKS,
    MAX_OUTPUT_TOKENS,
    MAX_PROPOSAL_TOKENS,
    NGRAM_SIZE,
    POLICIES,
    REAL_MOVEMENT_KEYS,
    SCHEMA_VERSION,
    atomic_write_json,
    build_prompt_token_batches,
    subtract_counter_summaries,
    validate_worker_result,
)
from speculative_runtime_performance_gate import (
    summarize_step_observations,
)


def _validate_rank_rows(rows, name):
    if (
        not isinstance(rows, tuple)
        or not rows
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError(
            f"{name} must be a non-empty tuple of mappings"
        )
    return tuple(dict(row) for row in rows)


def _movement_delta(before_rows, after_rows):
    before_rows = _validate_rank_rows(
        before_rows,
        "before movement summaries",
    )
    after_rows = _validate_rank_rows(
        after_rows,
        "after movement summaries",
    )
    if len(before_rows) != len(after_rows):
        raise ValueError("movement rank inventory mismatch")
    totals = {key: 0 for key in REAL_MOVEMENT_KEYS}
    for before, after in zip(before_rows, after_rows):
        delta = subtract_counter_summaries(
            before,
            after,
            keys=REAL_MOVEMENT_KEYS,
        )
        for key, value in delta.items():
            totals[key] += value
    return totals


def run_generation(
    *,
    engine,
    prompt_rows,
    sampling_params,
    expected_output_tokens,
    synchronize,
):
    if not engine.is_finished():
        raise RuntimeError("engine must be idle before generation")
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


def run_policy_cell(
    *,
    model_path: str,
    policy: str,
    context_tokens: int,
    batch_size: int,
    engine_factory,
    sampling_params_type,
    runtime_type,
    adapter_type,
    synchronize,
    run_generation_fn=run_generation,
) -> dict:
    if policy not in POLICIES:
        raise ValueError(f"unsupported policy: {policy}")
    if context_tokens not in CONTEXT_TOKENS:
        raise ValueError(
            f"unsupported context length: {context_tokens}"
        )
    if batch_size not in BATCH_SIZES:
        raise ValueError(
            f"unsupported batch size: {batch_size}"
        )
    engine = engine_factory(
        model_path,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=33024,
        max_num_batched_tokens=132096,
        max_num_seqs=batch_size,
        max_num_prefill_tokens_per_step=1024,
        chunked_prefill_decode_first=False,
        chunked_prefill_mixed_batch=False,
        kv_offload_mvp0=True,
        kv_offload_gpu_blocks=GPU_BLOCKS,
        kv_offload_logical_blocks=LOGICAL_BLOCKS,
        kv_offload_blockwise_decode=True,
        kv_offload_blockwise_prefill=True,
        kv_offload_blockwise_blocks=BLOCKWISE_BLOCKS,
    )
    try:
        if policy == "ngram":
            engine.activate_speculative_runtime(
                runtime_type(
                    adapter_type(
                        ngram_size=NGRAM_SIZE,
                        max_proposal_tokens=(
                            MAX_PROPOSAL_TOKENS
                        ),
                    )
                )
            )
        prompt_rows = build_prompt_token_batches(
            engine.tokenizer,
            batch_size=batch_size,
            prompt_tokens=context_tokens,
        )
        sampling_params = sampling_params_type(
            temperature=0.0,
            max_tokens=MAX_OUTPUT_TOKENS,
            ignore_eos=True,
        )
        run_generation_fn(
            engine=engine,
            prompt_rows=prompt_rows,
            sampling_params=sampling_params,
            expected_output_tokens=MAX_OUTPUT_TOKENS,
            synchronize=synchronize,
        )
        engine.clear_reusable_prefix_cache()
        before_rows = engine.kv_offload_summaries(
            timeout_s=60.0
        )
        recorded = run_generation_fn(
            engine=engine,
            prompt_rows=prompt_rows,
            sampling_params=sampling_params,
            expected_output_tokens=MAX_OUTPUT_TOKENS,
            synchronize=synchronize,
        )
        after_rows = engine.kv_offload_summaries(
            timeout_s=60.0
        )
        config = getattr(engine, "config", None)
        if config is None:
            config = getattr(
                getattr(engine, "model_runner", None),
                "config",
                None,
            )
        result = {
            "schema_version": SCHEMA_VERSION,
            "classification": CLASSIFICATION,
            "policy": policy,
            "context_tokens": context_tokens,
            "batch_size": batch_size,
            "prompt_rows": prompt_rows,
            "outputs": recorded["outputs"],
            "runtime": recorded["runtime"],
            "movement": _movement_delta(
                before_rows,
                after_rows,
            ),
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
            "visible_logical_blocks": (
                (context_tokens + 255) // 256
            ) * batch_size,
        }
        return validate_worker_result(result)
    finally:
        engine.exit()


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--policy",
        required=True,
        choices=POLICIES,
    )
    parser.add_argument(
        "--context-tokens",
        required=True,
        type=int,
        choices=CONTEXT_TOKENS,
    )
    parser.add_argument(
        "--batch-size",
        required=True,
        type=int,
        choices=BATCH_SIZES,
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = run_policy_cell(
        model_path=args.model,
        policy=args.policy,
        context_tokens=args.context_tokens,
        batch_size=args.batch_size,
        **_default_dependencies(),
    )
    atomic_write_json(Path(args.out), result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
