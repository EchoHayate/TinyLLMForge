from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys


def _load_module(name: str, filename: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_module(
    "qwen35_native_mtp_tp4_16k_target_kv_offload_gate",
    "qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py",
)
_frozen_worker = _load_module(
    "_qwen35_native_mtp_tp4_4k_worker_helpers",
    "qwen35_native_mtp_tp4_4k_engine_worker.py",
)
_frozen_worker.gate = gate
tp1_worker = _frozen_worker.tp1_worker
tp1_worker.gate = gate

target_model_manifest_sha256 = (
    _frozen_worker.target_model_manifest_sha256
)
mtp_checkpoint_manifest_sha256 = (
    _frozen_worker.mtp_checkpoint_manifest_sha256
)
distributed_environment = (
    _frozen_worker.distributed_environment
)
_collect_rank_snapshots = (
    _frozen_worker._collect_rank_snapshots
)
_ack_ranks = _frozen_worker._ack_ranks
_compact_receipts = _frozen_worker._compact_receipts

CHALLENGE_TAIL_TOKENS = 1024


def compact_target_logits(
    logits,
    *,
    sequence_ids: tuple[int, ...],
    top_k: int = 5,
) -> list[dict]:
    rows = logits.tolist()
    if len(rows) != len(sequence_ids):
        raise ValueError(
            "target logits row count does not match sequence IDs"
        )
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k <= 0
    ):
        raise ValueError("target logits top_k must be positive")
    compact = []
    for sequence_id, values in zip(sequence_ids, rows):
        ranked = sorted(
            enumerate(values),
            key=lambda item: (-float(item[1]), item[0]),
        )[:top_k]
        top_logits = [float(value) for _, value in ranked]
        compact.append({
            "sequence_id": int(sequence_id),
            "top_tokens": [
                int(token_id) for token_id, _ in ranked
            ],
            "top_logits": top_logits,
            "top1_margin": (
                float(top_logits[0] - top_logits[1])
                if len(top_logits) > 1
                else None
            ),
        })
    return compact


def run_generation_with_target_logit_diagnostics(
    *,
    engine,
    prompt_rows: list[dict],
    sampling_params,
    synchronize,
) -> tuple[list[dict], list[dict]]:
    for row in prompt_rows:
        engine.add_request(
            row["token_ids"],
            sampling_params,
        )
    outputs_by_id = {}
    observations = []
    engine.model_runner.enable_step_logits_recording(True)
    try:
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
            observation = dict(observation)
            logits = engine.model_runner.last_step_logits()
            if logits is None:
                raise RuntimeError(
                    "target logits were not recorded"
                )
            sequence_ids = tuple(
                int(sequence_id)
                for sequence_id in observation[
                    "new_completion_tokens_by_seq"
                ]
            )
            compact = compact_target_logits(
                logits,
                sequence_ids=sequence_ids,
            )
            observation["authority_target_logits"] = compact
            print(
                "AUTHORITY_TARGET_LOGITS "
                + json.dumps(
                    compact,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            observations.append(observation)
            for sequence_id, token_ids in step_outputs:
                outputs_by_id[int(sequence_id)] = [
                    int(token_id) for token_id in token_ids
                ]
    finally:
        engine.model_runner.enable_step_logits_recording(False)
    output_rows = []
    for prompt_index, sequence_id in enumerate(
        sorted(outputs_by_id)
    ):
        token_ids = outputs_by_id[sequence_id]
        if len(token_ids) != gate.MAX_OUTPUT_TOKENS:
            raise RuntimeError(
                "engine output token count mismatch"
            )
        output_rows.append({
            "prompt_index": prompt_index,
            "token_count": len(token_ids),
            "token_ids": token_ids,
            "sha256": gate._json_sha256(token_ids),
        })
    if len(output_rows) != len(prompt_rows):
        raise RuntimeError(
            "engine output inventory does not match prompts"
        )
    return output_rows, observations


_RUNTIME_COUNTER_FIELDS = (
    "proposal_rows",
    "proposed_tokens",
    "accepted_draft_tokens",
    "rejected_draft_tokens",
    "first_target_callbacks",
    "verify_callbacks",
    "first_target_target_forwards",
    "verify_target_forwards",
    "accepted_prefix_target_replays",
)


def normalize_rank_snapshots(
    snapshots: tuple[dict, ...],
    *,
    policy: str,
    batch_size: int,
    runtime: dict | None,
    finalize_ack_ranks: tuple[int, ...],
    release_ack_ranks: tuple[int, ...],
) -> list[dict]:
    try:
        return _frozen_worker.normalize_rank_snapshots(
            snapshots,
            policy=policy,
            batch_size=batch_size,
            runtime=runtime,
            finalize_ack_ranks=finalize_ack_ranks,
            release_ack_ranks=release_ack_ranks,
        )
    except ValueError as error:
        if policy != "native_mtp" or not isinstance(runtime, dict):
            raise
        counters = ", ".join(
            f"{field}={runtime.get(field)!r}"
            for field in _RUNTIME_COUNTER_FIELDS
        )
        raise ValueError(
            f"{error}; runtime counters: {counters}"
        ) from error


def summarize_runtime(
    observations: list[dict],
    *,
    capture: dict,
    native_binding: dict | None,
) -> dict:
    runtime = tp1_worker.summarize_runtime(
        observations,
        capture=capture,
        native_binding=native_binding,
    )
    accepted_prefix_target_replays = 0
    for observation in observations:
        selected = {
            int(sequence_id)
            for sequence_id in observation.get(
                "speculative_selected_seq_ids",
                (),
            )
        }
        suppressed = {
            int(sequence_id)
            for sequence_id in observation.get(
                "speculative_suppressed_seq_ids",
                (),
            )
        }
        if selected.intersection(suppressed):
            raise RuntimeError(
                "speculative selected and suppressed rows overlap"
            )
        accepted = {
            int(sequence_id)
            for sequence_id, count in observation.get(
                "speculative_accepted_draft_token_counts",
                {},
            ).items()
            if gate._integer(
                count,
                "accepted draft token count",
            ) > 0
        }
        if not accepted.issubset(selected):
            raise RuntimeError(
                "accepted speculative rows are not selected"
            )
        ordinary_calls = gate._integer(
            observation.get(
                "authority_normal_decode_target_forward_calls",
                0,
            ),
            "ordinary decode target forward calls",
        )
        if ordinary_calls and not suppressed:
            accepted_prefix_target_replays += ordinary_calls
    runtime["accepted_prefix_target_replays"] = (
        accepted_prefix_target_replays
    )
    return runtime


def build_prompt_rows(tokenizer, batch_size: int) -> list[dict]:
    rows = tp1_worker.build_prompt_rows(tokenizer, batch_size)
    for prompt_index, row in enumerate(rows):
        pool = []
        for text in (
            (
                f" challenge {prompt_index} unordered proof "
                "delta epsilon contradiction"
            ),
            (
                f" challenge {prompt_index} code template "
                "lambda coroutine pointer"
            ),
            (
                f" challenge {prompt_index} json null true "
                "matrix tensor boundary"
            ),
            (
                f" challenge {prompt_index} multilingual "
                "sequence random walk"
            ),
        ):
            pool.extend(tp1_worker._encode_seed(tokenizer, text))
        if len(set(pool)) <= 8:
            raise RuntimeError(
                "challenge prompt token pool is not diverse"
            )
        state = 0x9E3779B9 ^ (prompt_index + 1)
        tail = []
        for offset in range(CHALLENGE_TAIL_TOKENS):
            state = (
                1664525 * state + 1013904223
            ) & 0xFFFFFFFF
            tail.append(
                pool[(state + 17 * offset) % len(pool)]
            )
        token_ids = list(row["token_ids"])
        token_ids[-CHALLENGE_TAIL_TOKENS:] = tail
        row["token_ids"] = token_ids
        row["token_count"] = len(token_ids)
        row["sha256"] = gate._json_sha256(token_ids)
    if len({row["sha256"] for row in rows}) != batch_size:
        raise RuntimeError(
            "deterministic challenge prompts must be distinct"
        )
    return rows


def engine_kwargs(
    *,
    policy: str,
    batch_size: int,
) -> dict:
    gate.cell_key(policy, batch_size)
    return {
        "tensor_parallel_size": gate.WORLD_SIZE,
        "enforce_eager": True,
        "max_model_len": gate.MAX_MODEL_LEN,
        "max_num_batched_tokens": (
            gate.MAX_NUM_BATCHED_TOKENS
        ),
        "max_num_prefill_tokens_per_step": (
            gate.MAX_NUM_PREFILL_TOKENS_PER_STEP
        ),
        "max_num_seqs": batch_size,
        "kvcache_block_size": gate.BLOCK_SIZE,
        "chunked_prefill_decode_first": False,
        "chunked_prefill_mixed_batch": False,
        "kv_offload_mvp0": True,
        "kv_offload_gpu_blocks": (
            gate.KV_OFFLOAD_GPU_BLOCKS
        ),
        "kv_offload_logical_blocks": (
            gate.KV_OFFLOAD_LOGICAL_BLOCKS
        ),
        "kv_offload_blockwise_prefill": True,
        "kv_offload_blockwise_decode": True,
        "kv_offload_blockwise_blocks": (
            gate.KV_OFFLOAD_BLOCKWISE_BLOCKS
        ),
        "qwen35_mtp_enabled": policy == "native_mtp",
        "qwen35_mtp_cuda_graphs": False,
        "qwen35_mtp_max_proposal_tokens": (
            gate.MAX_PROPOSAL_TOKENS
        ),
    }


def movement_delta(
    before_rows: tuple[dict, ...],
    after_rows: tuple[dict, ...],
) -> list[dict]:
    if (
        not isinstance(before_rows, tuple)
        or not isinstance(after_rows, tuple)
        or len(before_rows) != gate.WORLD_SIZE
        or len(after_rows) != gate.WORLD_SIZE
    ):
        raise ValueError(
            "KV movement rank inventory mismatch"
        )
    result = []
    for rank, (before, after) in enumerate(
        zip(before_rows, after_rows)
    ):
        if (
            not isinstance(before, dict)
            or not isinstance(after, dict)
        ):
            raise ValueError(
                "KV movement summary must be a mapping"
            )
        row = {
            "rank": rank,
            "provenance": "engine.kv_offload_summaries",
        }
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


def capacity_rows(
    rows: tuple[dict, ...],
) -> list[dict]:
    if (
        not isinstance(rows, tuple)
        or len(rows) != gate.WORLD_SIZE
    ):
        raise ValueError(
            "KV capacity rank inventory mismatch"
        )
    result = []
    for rank, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(
                "KV capacity summary must be a mapping"
            )
        result.append({
            "rank": rank,
            "provenance": "engine.kv_offload_summaries",
            "gpu_blocks": row.get("gpu_blocks"),
            "logical_blocks": row.get("logical_blocks"),
            "resident_blocks": row.get("resident_blocks"),
            "peak_resident_blocks": row.get(
                "peak_resident_blocks"
            ),
        })
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
    synchronize,
    run_generation_fn=(
        run_generation_with_target_logit_diagnostics
    ),
    target_manifest_resolver=target_model_manifest_sha256,
    mtp_manifest_resolver=mtp_checkpoint_manifest_sha256,
    prompt_builder=build_prompt_rows,
    rank_snapshot_collector=_collect_rank_snapshots,
    model_identity_fn=tp1_worker._model_identity,
) -> dict:
    gate.cell_key(policy, batch_size)
    target_digest = target_manifest_resolver(model_path)
    mtp_digest = mtp_manifest_resolver(model_path)
    if target_digest != gate.TARGET_MODEL_MANIFEST_SHA256:
        raise RuntimeError(
            "target model manifest does not match authority"
        )
    if mtp_digest != gate.MTP_CHECKPOINT_MANIFEST_SHA256:
        raise RuntimeError(
            "MTP checkpoint manifest does not match authority"
        )
    engine = None
    cell = None
    exit_receipt = None
    with distributed_environment(
        gpu_indices=gpu_indices,
        dist_port=dist_port,
        master_port=master_port,
    ):
        try:
            native = policy == "native_mtp"
            config = engine_kwargs(
                policy=policy,
                batch_size=batch_size,
            )
            engine = engine_factory(model_path, **config)
            executor = None
            if native:
                descriptor, _, executor, _ = (
                    tp1_worker.validate_native_registration(
                        engine.model_runner
                    )
                )
                engine.activate_speculative_runtime(
                    runtime_type(
                        model_runner_executor=descriptor
                    )
                )
            prompt_rows = prompt_builder(
                engine.tokenizer,
                batch_size,
            )
            sampling_params = sampling_params_type(
                temperature=0.0,
                max_tokens=gate.MAX_OUTPUT_TOKENS,
                ignore_eos=True,
            )
            before_rows = engine.kv_offload_summaries(
                timeout_s=60.0
            )
            empty_capture = {
                "method_names": [],
                "ordinary_decode_target_forward_calls": 0,
                "proposal_finalize_receipts": [],
                "side_state_receipts": [],
                "proposal_kv_receipts": [],
                "lifecycle_events": [],
            }
            with capture_residency_phases(
                engine
            ) as residency_phases:
                if native:
                    with tp1_worker.capture_runtime_receipts(
                        engine,
                        executor,
                    ) as capture:
                        output_rows, observations = (
                            run_generation_fn(
                                engine=engine,
                                prompt_rows=prompt_rows,
                                sampling_params=sampling_params,
                                synchronize=synchronize,
                                target_forward_capture=capture,
                            )
                        )
                else:
                    capture = empty_capture
                    output_rows, observations = (
                        run_generation_fn(
                            engine=engine,
                            prompt_rows=prompt_rows,
                            sampling_params=sampling_params,
                            synchronize=synchronize,
                        )
                    )
            after_rows = engine.kv_offload_summaries(
                timeout_s=60.0
            )
            engine.flush_pending_hybrid_state_releases(
                timeout_s=60.0
            )
            raw_snapshots = rank_snapshot_collector(engine)
            if native:
                runtime = summarize_runtime(
                    observations,
                    capture=capture,
                    native_binding={"registered": True},
                )
                runtime["target_prefill_observations"] = (
                    batch_size
                )
                finalize_ack_ranks = _ack_ranks(
                    engine,
                    "speculative_proposal_finalize_batch",
                )
                release_ack_ranks = _ack_ranks(
                    engine,
                    "release_speculative_proposal_sequence",
                )
                side_state_receipts = _compact_receipts(
                    capture["side_state_receipts"],
                    batch_size=batch_size,
                    operations=[
                        "prepare",
                        "select",
                        "apply",
                        "seal",
                    ],
                    name="side-state",
                )
                target_kv_receipts = [
                    {
                        "sequence_id": sequence_id,
                        "operations": ["prepare", "commit"],
                    }
                    for sequence_id in range(batch_size)
                ]
            else:
                runtime = None
                finalize_ack_ranks = ()
                release_ack_ranks = ()
                side_state_receipts = []
                target_kv_receipts = []
            rank_snapshots = normalize_rank_snapshots(
                raw_snapshots,
                policy=policy,
                batch_size=batch_size,
                runtime=runtime,
                finalize_ack_ranks=finalize_ack_ranks,
                release_ack_ranks=release_ack_ranks,
            )
            cell = {
                "schema_version": gate.SCHEMA_VERSION,
                "policy": policy,
                "batch_size": batch_size,
                "world_size": gate.WORLD_SIZE,
                "rank_inventory": list(gate.RANKS),
                "gpu_indices": list(gpu_indices),
                "prompt_token_count": gate.PROMPT_TOKENS,
                "max_output_tokens": gate.MAX_OUTPUT_TOKENS,
                "max_proposal_tokens": (
                    gate.MAX_PROPOSAL_TOKENS
                ),
                "model_identity": model_identity_fn(
                    engine,
                    target_digest=target_digest,
                    mtp_digest=mtp_digest,
                ),
                "engine_config": config,
                "prompt_rows": prompt_rows,
                "output_rows": output_rows,
                "rank_snapshots": rank_snapshots,
                "side_state_receipts": side_state_receipts,
                "target_kv_receipts": target_kv_receipts,
                "residency_phases": list(residency_phases),
                "kv_rank_deltas": movement_delta(
                    before_rows,
                    after_rows,
                ),
                "kv_capacity_rows": capacity_rows(after_rows),
                "runtime_poisoned": bool(
                    engine.speculative_runtime_poisoned
                ),
                "cleanup": None,
            }
        finally:
            if engine is not None:
                exit_receipt = engine.exit()
    if cell is None or not isinstance(exit_receipt, dict):
        raise RuntimeError(
            "TP4/16K worker did not produce a complete cell"
        )
    rank_exit_codes = list(
        exit_receipt.get("rank_exit_codes", [])
    )
    owned_children = list(
        exit_receipt.get("owned_children_remaining", [])
    )
    process_group_destroyed = bool(
        exit_receipt.get("process_group_destroyed")
    )
    cell["cleanup"] = {
        "rank_exit_codes": rank_exit_codes,
        "process_group_destroyed": process_group_destroyed,
        "shared_memory_released": (
            process_group_destroyed and not owned_children
        ),
        "owned_children_remaining": owned_children,
        "engine_exit_called": True,
    }
    return gate.validate_cell_result(cell)


def _default_dependencies():
    import torch

    from tinyvllm import LLM
    from tinyvllm.engine.speculative_runtime import (
        EngineSpeculativeRuntime,
    )
    from tinyvllm.sampling_params import SamplingParams

    return {
        "engine_factory": LLM,
        "sampling_params_type": SamplingParams,
        "runtime_type": EngineSpeculativeRuntime,
        "synchronize": torch.cuda.synchronize,
    }


def _gpu_indices(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "GPU indices must be comma-separated integers"
        ) from error
    if (
        len(parsed) != gate.WORLD_SIZE
        or len(set(parsed)) != gate.WORLD_SIZE
        or any(index < 0 for index in parsed)
    ):
        raise argparse.ArgumentTypeError(
            "GPU indices must contain four distinct "
            "non-negative integers"
        )
    return parsed


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
