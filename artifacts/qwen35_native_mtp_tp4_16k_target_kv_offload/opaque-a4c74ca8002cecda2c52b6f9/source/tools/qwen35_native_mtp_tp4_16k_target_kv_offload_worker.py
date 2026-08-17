from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
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
    run_generation_fn=tp1_worker._run_generation,
    target_manifest_resolver=target_model_manifest_sha256,
    mtp_manifest_resolver=mtp_checkpoint_manifest_sha256,
    prompt_builder=tp1_worker.build_prompt_rows,
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
                runtime = tp1_worker.summarize_runtime(
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
