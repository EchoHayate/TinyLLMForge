from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

from speculative_runtime_performance_gate import (
    BATCH_SIZES,
    MAX_OUTPUT_TOKENS,
    MAX_PROPOSAL_TOKENS,
    MEASURED_RUNS,
    NGRAM_SIZE,
    PARITY_RUNS,
    POLICIES,
    PROMPT_TOKENS,
    REAL_MOVEMENT_KEYS,
    WARMUP_RUNS,
    build_prompt_token_batches,
    build_run_metrics,
    subtract_counter_summaries,
    summarize_step_observations,
)


def _validate_rank_rows(
    rows,
    *,
    name: str,
) -> tuple[dict, ...]:
    if (
        not isinstance(rows, tuple)
        or not rows
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError(
            f"{name} must be a non-empty tuple of mappings"
        )
    return tuple(dict(row) for row in rows)


def _movement_delta(
    before_rows: tuple[dict, ...],
    after_rows: tuple[dict, ...],
) -> dict:
    if len(before_rows) != len(after_rows):
        raise ValueError("movement rank inventory mismatch")
    ranks = []
    totals = {key: 0 for key in REAL_MOVEMENT_KEYS}
    for rank, (before, after) in enumerate(
        zip(before_rows, after_rows)
    ):
        delta = subtract_counter_summaries(
            before,
            after,
            keys=REAL_MOVEMENT_KEYS,
        )
        ranks.append({"rank": rank, **delta})
        for key, value in delta.items():
            totals[key] += value
    return {
        "ranks": ranks,
        "totals": totals,
    }


def _memory_result(
    reset_rows: tuple[dict, ...],
    final_rows: tuple[dict, ...],
) -> dict:
    if len(reset_rows) != len(final_rows):
        raise ValueError("memory rank inventory mismatch")
    ranks = []
    for rank, (reset, final) in enumerate(
        zip(reset_rows, final_rows)
    ):
        reset_rank = reset.get("rank", rank)
        final_rank = final.get("rank", rank)
        if reset_rank != rank or final_rank != rank:
            raise ValueError("memory rank mismatch")
        required = (
            "cuda_allocated_bytes",
            "cuda_reserved_bytes",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
            "kv_capacity_bytes",
        )
        for name, row in (("reset", reset), ("final", final)):
            for key in required:
                value = row.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise ValueError(
                        f"{name} memory {key} is invalid"
                    )
        ranks.append({
            "rank": rank,
            "reset": dict(reset),
            "final": dict(final),
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
    return {
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


def evict_active_history(engine):
    scheduler = getattr(engine, "scheduler", None)
    running = tuple(
        getattr(scheduler, "running", ())
    )
    if not running:
        raise RuntimeError(
            "active-history eviction requires running sequences"
        )
    model_runner = getattr(engine, "model_runner", None)
    manager = getattr(model_runner, "kv_offload", None)
    if manager is None:
        raise RuntimeError(
            "active-history eviction requires KVOffloadMVP0"
        )
    logical_blocks = []
    seen = set()
    for sequence in running:
        for logical_block in sequence.block_table:
            logical_block = int(logical_block)
            if logical_block in seen:
                continue
            logical_blocks.append(logical_block)
            seen.add(logical_block)
    if not logical_blocks:
        raise RuntimeError(
            "active-history eviction found no logical blocks"
        )
    manager.writeback_dirty(logical_blocks)
    manager.synchronize_copies()
    identities = []
    for logical_block in logical_blocks:
        generation = manager.bound_generations[
            logical_block
        ]
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise RuntimeError(
                "active-history eviction requires bound generations"
            )
        identities.append((logical_block, generation))
    return manager.evict_clean_resident_blocks(
        tuple(identities)
    )


def run_request_batch(
    *,
    engine,
    prompt_rows: list[dict],
    sampling_params,
    expected_output_tokens: int,
    synchronize,
    clock_ns,
    evict_fn=evict_active_history,
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
            or len(token_ids) != PROMPT_TOKENS
        ):
            raise ValueError(
                "worker prompt must contain exactly "
                f"{PROMPT_TOKENS} tokens"
            )
        engine.add_request(token_ids, sampling_params)
    observations = []
    outputs_by_id = {}
    token_events = {}
    finished_at_ns = {}
    last_step_ns = started_ns
    evicted_block_identities = ()
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
            if not token_ids:
                continue
            token_events.setdefault(
                int(sequence_id),
                [],
            ).append((last_step_ns, len(token_ids)))
        if (
            not evicted_block_identities
            and any(bool(token_ids) for token_ids in deltas.values())
        ):
            evicted_block_identities = tuple(
                tuple(identity)
                for identity in evict_fn(engine)
            )
        for sequence_id in observation.get(
            "finished_seq_ids",
            [],
        ):
            finished_at_ns[int(sequence_id)] = last_step_ns
        for sequence_id, token_ids in step_outputs:
            outputs_by_id[int(sequence_id)] = list(token_ids)
    after_rows = _validate_rank_rows(
        engine.kv_offload_summaries(timeout_s=60.0),
        name="after movement summaries",
    )
    final_memory_rows = _validate_rank_rows(
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
    timing = build_run_metrics(
        request_start_ns=started_ns,
        request_finish_ns=last_step_ns,
        token_events=token_events,
        finished_at_ns=finished_at_ns,
        expected_output_tokens=expected_output_tokens,
    )
    return {
        "outputs": outputs,
        "timing": timing,
        "runtime": summarize_step_observations(
            observations
        ),
        "movement": _movement_delta(
            before_rows,
            after_rows,
        ),
        "memory": _memory_result(
            reset_rows,
            final_memory_rows,
        ),
        "evicted_block_identities": [
            list(identity)
            for identity in evicted_block_identities
        ],
        "observations": observations,
    }


def run_policy_campaign(
    *,
    model_path: str,
    policy: str,
    batch_size: int,
    engine_factory,
    sampling_params_type,
    runtime_type,
    adapter_type,
    synchronize,
    clock_ns,
    run_batch_fn=run_request_batch,
) -> dict:
    if policy not in POLICIES:
        raise ValueError(f"unsupported policy: {policy}")
    if batch_size not in BATCH_SIZES:
        raise ValueError(
            f"unsupported batch size: {batch_size}"
        )
    engine = engine_factory(
        model_path,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=4352,
        max_num_batched_tokens=16384,
        max_num_seqs=batch_size,
        max_num_prefill_tokens_per_step=1024,
        chunked_prefill_mixed_batch=False,
        kv_offload_mvp0=True,
        kv_offload_gpu_blocks=68,
        kv_offload_logical_blocks=128,
        kv_offload_blockwise_decode=False,
        kv_offload_blockwise_prefill=False,
        kv_offload_blockwise_blocks=1,
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
            prompt_tokens=PROMPT_TOKENS,
        )
        sampling_params = sampling_params_type(
            temperature=0.0,
            max_tokens=MAX_OUTPUT_TOKENS,
            ignore_eos=True,
        )

        def run_once():
            return run_batch_fn(
                engine=engine,
                prompt_rows=prompt_rows,
                sampling_params=sampling_params,
                expected_output_tokens=MAX_OUTPUT_TOKENS,
                synchronize=synchronize,
                clock_ns=clock_ns,
            )

        warmup_runs = [
            run_once() for _ in range(WARMUP_RUNS)
        ]
        parity_runs = [
            run_once() for _ in range(PARITY_RUNS)
        ]
        measured_runs = [
            run_once() for _ in range(MEASURED_RUNS)
        ]
        config = getattr(engine, "config", None)
        if config is None:
            config = getattr(
                getattr(engine, "model_runner", None),
                "config",
                None,
            )
        return {
            "policy": policy,
            "batch_size": batch_size,
            "prompt_rows": prompt_rows,
            "warmup_runs": warmup_runs,
            "parity_runs": parity_runs,
            "measured_runs": measured_runs,
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
        engine.exit()


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--policy",
        required=True,
        choices=POLICIES,
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
    result = run_policy_campaign(
        model_path=args.model,
        policy=args.policy,
        batch_size=args.batch_size,
        **_default_dependencies(),
    )
    _write_json_atomic(Path(args.out), result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
