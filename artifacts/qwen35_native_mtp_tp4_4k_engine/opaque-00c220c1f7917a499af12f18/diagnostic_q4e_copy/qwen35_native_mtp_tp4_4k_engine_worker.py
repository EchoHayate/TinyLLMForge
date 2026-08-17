from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
import torch.distributed as _trace_dist
from torch.distributed.distributed_c10d import _get_default_group

_TRACE_DIRECTORY = Path('/data00/home/sitian/sitian-workspace01/tllm/qwen35-native-mtp-tp4-4k-engine-runs/opaque-00c220c1f7917a499af12f18/diagnostic-q4e/traces')
_TRACE_INDEX = 0


def _collective_sequence():
    if not _trace_dist.is_initialized():
        return -1
    return int(
        _get_default_group()._get_sequence_number_for_group()
    )


def _argument_summary(args):
    rows = []
    for argument in args[:2]:
        if hasattr(argument, "shape"):
            rows.append(
                f"shape={tuple(argument.shape)} "
                f"dtype={getattr(argument, 'dtype', None)}"
            )
        elif isinstance(argument, (list, tuple)):
            rows.append(f"{type(argument).__name__}[{len(argument)}]")
        else:
            rows.append(type(argument).__name__)
    return " ".join(rows)


def _install_collective_trace(name):
    original = getattr(_trace_dist, name)
    def traced(*args, **kwargs):
        global _TRACE_INDEX
        rank = _trace_dist.get_rank() if _trace_dist.is_initialized() else -1
        index = _TRACE_INDEX; _TRACE_INDEX += 1
        sequence_before = _collective_sequence()
        stack = traceback.extract_stack(limit=7)[:-1]
        location = ' <- '.join(f'{Path(frame.filename).name}:{frame.lineno}:{frame.name}' for frame in stack[-4:])
        trace_path = _TRACE_DIRECTORY / f'rank{rank}.log'
        with trace_path.open('a', encoding='utf-8') as trace:
            trace.write(
                f'{index:04d} BEGIN {name} '
                f'seq={sequence_before} {_argument_summary(args)} '
                f'{location}\n'
            )
            trace.flush()
        result = original(*args, **kwargs)
        sequence_after = _collective_sequence()
        with trace_path.open('a', encoding='utf-8') as trace:
            trace.write(
                f'{index:04d} END {name} seq={sequence_after}\n'
            )
            trace.flush()
        return result
    setattr(_trace_dist, name, traced)
for _collective_name in (
    'all_reduce',
    'broadcast',
    'gather',
    'all_gather',
    'barrier',
):
    _install_collective_trace(_collective_name)


def _load_module(module_name: str, filename: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_module(
    "qwen35_native_mtp_tp4_4k_engine_gate",
    "qwen35_native_mtp_tp4_4k_engine_gate.py",
)
tp1_worker = _load_module(
    "qwen35_native_mtp_tp1_4k_engine_worker_for_tp4",
    "qwen35_native_mtp_tp1_4k_engine_worker.py",
)

target_model_manifest_sha256 = (
    tp1_worker.target_model_manifest_sha256
)
mtp_checkpoint_manifest_sha256 = (
    tp1_worker.mtp_checkpoint_manifest_sha256
)


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
        raise ValueError(
            "GPU indices must contain four distinct "
            "non-negative integers"
        )
    return value


@contextmanager
def distributed_environment(
    *,
    gpu_indices: tuple[int, ...],
    dist_port: int,
    master_port: int,
):
    gpu_indices = _validate_gpu_indices(gpu_indices)
    for value, name in (
        (dist_port, "distributed port"),
        (master_port, "master port"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
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


def load_tp1_output_rows(
    result_path: Path,
    *,
    batch_size: int,
    expected_sha256: str = gate.TP1_AUTHORITY_SHA256,
) -> list[dict]:
    gate.cell_key("native_mtp", batch_size)
    result_path = Path(result_path)
    if not result_path.is_file():
        raise ValueError("TP1 authority result is missing")
    expected_sha256 = gate._sha256(
        expected_sha256,
        "TP1 authority",
    )
    actual = hashlib.sha256(result_path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError("TP1 authority digest mismatch")
    try:
        result = json.loads(
            result_path.read_text(encoding="utf-8")
        )
        rows = result["cells"][
            gate.cell_key("native_mtp", batch_size)
        ]["output_rows"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            "TP1 authority output rows are invalid"
        ) from error
    return gate._validate_token_rows(
        rows,
        batch_size=batch_size,
        token_count=gate.MAX_OUTPUT_TOKENS,
        name="TP1 output",
    )


def _baseline_rank_snapshot(rank: int) -> dict:
    return {
        "rank": rank,
        "world_size": gate.WORLD_SIZE,
        "registered": False,
        "module_type": None,
        "physical_store_type": None,
        "shared_embed_tokens": False,
        "shared_lm_head": False,
        "local_query_heads": 0,
        "local_kv_heads": 0,
        "target_prefill_observations": 0,
        "bootstrap_rows": 0,
        "proposal_rows": 0,
        "proposed_tokens": 0,
        "accepted_draft_tokens": 0,
        "rejected_draft_tokens": 0,
        "first_target_callbacks": 0,
        "verify_callbacks": 0,
        "first_target_target_forwards": 0,
        "verify_target_forwards": 0,
        "accepted_prefix_target_replays": 0,
        "lm_head_logits_rows": 0,
        "token_broadcasts": 0,
        "token_broadcast_shape": [],
        "token_broadcast_dtype": None,
        "token_broadcast_source_rank": None,
        "selected_tokens_sha256": gate._json_sha256([]),
        "finalize_ack_ranks": [],
        "release_ack_ranks": [],
        "executor": None,
    }


def normalize_rank_snapshots(
    snapshots: tuple[dict, ...],
    *,
    policy: str,
    batch_size: int,
    runtime: dict | None,
    finalize_ack_ranks: tuple[int, ...],
    release_ack_ranks: tuple[int, ...],
) -> list[dict]:
    gate.cell_key(policy, batch_size)
    if (
        not isinstance(snapshots, tuple)
        or len(snapshots) != gate.WORLD_SIZE
        or tuple(
            snapshot.get("rank")
            for snapshot in snapshots
            if isinstance(snapshot, dict)
        )
        != gate.RANKS
    ):
        raise ValueError("rank snapshot inventory mismatch")
    if policy == "baseline":
        if runtime is not None:
            raise ValueError("baseline runtime must be absent")
        normalized = []
        for rank, snapshot in enumerate(snapshots):
            if snapshot != {
                "rank": rank,
                "world_size": gate.WORLD_SIZE,
                "registered": False,
                "executor": None,
            }:
                raise ValueError(
                    "baseline rank snapshot is invalid"
                )
            normalized.append(_baseline_rank_snapshot(rank))
        return normalized
    if not isinstance(runtime, dict):
        raise ValueError("native runtime summary is missing")
    required_runtime = {
        "target_prefill_observations",
        "proposal_rows",
        "proposed_tokens",
        "accepted_draft_tokens",
        "rejected_draft_tokens",
        "first_target_callbacks",
        "verify_callbacks",
        "first_target_target_forwards",
        "verify_target_forwards",
        "accepted_prefix_target_replays",
    }
    if not required_runtime.issubset(runtime):
        raise ValueError("native runtime summary is incomplete")
    if finalize_ack_ranks != gate.WORKER_RANKS:
        raise ValueError("finalize acknowledgement ranks mismatch")
    if release_ack_ranks != gate.WORKER_RANKS:
        raise ValueError("release acknowledgement ranks mismatch")
    normalized = []
    for rank, snapshot in enumerate(snapshots):
        expected_fields = {
            "rank",
            "world_size",
            "registered",
            "module_type",
            "physical_store_type",
            "shared_embed_tokens",
            "shared_lm_head",
            "local_query_heads",
            "local_kv_heads",
            "executor",
        }
        if (
            not isinstance(snapshot, dict)
            or set(snapshot) != expected_fields
            or snapshot["rank"] != rank
            or snapshot["world_size"] != gate.WORLD_SIZE
            or snapshot["registered"] is not True
            or not isinstance(snapshot["executor"], dict)
        ):
            raise ValueError("native rank snapshot is invalid")
        selected_tokens = snapshot["executor"].get(
            "selected_tokens"
        )
        if not isinstance(selected_tokens, list):
            raise ValueError("selected token evidence is missing")
        normalized.append({
            **snapshot,
            "target_prefill_observations": runtime[
                "target_prefill_observations"
            ],
            "bootstrap_rows": batch_size,
            "proposal_rows": runtime["proposal_rows"],
            "proposed_tokens": runtime["proposed_tokens"],
            "accepted_draft_tokens": runtime[
                "accepted_draft_tokens"
            ],
            "rejected_draft_tokens": runtime[
                "rejected_draft_tokens"
            ],
            "first_target_callbacks": runtime[
                "first_target_callbacks"
            ],
            "verify_callbacks": runtime["verify_callbacks"],
            "first_target_target_forwards": runtime[
                "first_target_target_forwards"
            ],
            "verify_target_forwards": runtime[
                "verify_target_forwards"
            ],
            "accepted_prefix_target_replays": runtime[
                "accepted_prefix_target_replays"
            ],
            "lm_head_logits_rows": (
                len(selected_tokens) if rank == 0 else 0
            ),
            "token_broadcasts": len(selected_tokens),
            "token_broadcast_shape": [1],
            "token_broadcast_dtype": "torch.int64",
            "token_broadcast_source_rank": 0,
            "selected_tokens_sha256": gate._json_sha256(
                selected_tokens
            ),
            "finalize_ack_ranks": list(
                finalize_ack_ranks
            ),
            "release_ack_ranks": list(
                release_ack_ranks
            ),
        })
    return gate._validate_rank_snapshots(
        normalized,
        policy=policy,
        batch_size=batch_size,
    )


def _collect_rank_snapshots(engine) -> tuple[dict, ...]:
    local, worker_acks = engine.call_model_runner_acknowledged(
        "qwen35_mtp_authority_snapshot",
        timeout_s=60.0,
    )
    rows = {0: local}
    for acknowledgement in worker_acks:
        rank = getattr(acknowledgement, "rank", None)
        result = getattr(acknowledgement, "result", None)
        if rank in rows or not isinstance(result, dict):
            raise RuntimeError(
                "MTP authority acknowledgement is invalid"
            )
        rows[rank] = result
    if tuple(sorted(rows)) != gate.RANKS:
        raise RuntimeError(
            "MTP authority acknowledgement ranks are incomplete"
        )
    return tuple(rows[rank] for rank in gate.RANKS)


def _ack_ranks(engine, fragment: str) -> tuple[int, ...]:
    rows = getattr(
        engine,
        "speculative_proposal_lifecycle_ack_rows",
        (),
    )
    matching = [
        row
        for row in rows
        if fragment in row.get("method_name", "")
    ]
    if not matching:
        raise RuntimeError(
            f"{fragment} acknowledgement evidence is missing"
        )
    ranks = {
        rank
        for row in matching
        for rank in row.get("worker_ranks", ())
    }
    return tuple(sorted(ranks))


def _compact_receipts(
    receipts: list[dict],
    *,
    batch_size: int,
    operations: list[str],
    name: str,
) -> list[dict]:
    by_sequence = {
        sequence_id: []
        for sequence_id in range(batch_size)
    }
    for receipt in receipts:
        if not isinstance(receipt, dict):
            raise RuntimeError(f"{name} receipt is invalid")
        sequence_id = receipt.get("sequence_id")
        operation = receipt.get("operation")
        if sequence_id in by_sequence:
            by_sequence[sequence_id].append(operation)
    for sequence_id, observed in by_sequence.items():
        cursor = 0
        for operation in observed:
            if (
                cursor < len(operations)
                and operation == operations[cursor]
            ):
                cursor += 1
        if cursor != len(operations):
            raise RuntimeError(
                f"{name} lifecycle is incomplete for "
                f"sequence {sequence_id}"
            )
    return [
        {
            "sequence_id": sequence_id,
            "operations": list(operations),
        }
        for sequence_id in range(batch_size)
    ]


def run_policy_cell(
    *,
    model_path: str,
    gpu_indices: tuple[int, ...],
    policy: str,
    batch_size: int,
    dist_port: int,
    master_port: int,
    tp1_result_path: Path,
    engine_factory,
    sampling_params_type,
    runtime_type,
    synchronize,
    target_manifest_resolver=target_model_manifest_sha256,
    mtp_manifest_resolver=mtp_checkpoint_manifest_sha256,
) -> dict:
    gpu_indices = _validate_gpu_indices(gpu_indices)
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
    tp1_output_rows = (
        load_tp1_output_rows(
            tp1_result_path,
            batch_size=batch_size,
        )
        if policy == "native_mtp"
        else None
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
            engine = engine_factory(
                model_path,
                tensor_parallel_size=gate.WORLD_SIZE,
                enforce_eager=True,
                max_model_len=8192,
                max_num_batched_tokens=16384,
                max_num_prefill_tokens_per_step=1024,
                max_num_seqs=batch_size,
                num_kvcache_blocks=64,
                kv_offload_mvp0=False,
                qwen35_mtp_enabled=native,
                qwen35_mtp_cuda_graphs=False,
                qwen35_mtp_max_proposal_tokens=(
                    gate.MAX_PROPOSAL_TOKENS
                ),
            )
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
            prompt_rows = tp1_worker.build_prompt_rows(
                engine.tokenizer,
                batch_size,
            )
            sampling_params = sampling_params_type(
                temperature=0.0,
                max_tokens=4,
                ignore_eos=True,
            )
            empty_capture = {
                "method_names": [],
                "ordinary_decode_target_forward_calls": 0,
                "proposal_finalize_receipts": [],
                "side_state_receipts": [],
                "proposal_kv_receipts": [],
                "lifecycle_events": [],
            }
            if native:
                with tp1_worker.capture_runtime_receipts(
                    engine,
                    executor,
                ) as capture:
                    output_rows, observations = (
                        tp1_worker._run_generation(
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
                    tp1_worker._run_generation(
                        engine=engine,
                        prompt_rows=prompt_rows,
                        sampling_params=sampling_params,
                        synchronize=synchronize,
                    )
                )
            engine.flush_pending_hybrid_state_releases(
                timeout_s=60.0
            )
            raw_snapshots = _collect_rank_snapshots(engine)
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
            runtime_poisoned = bool(
                engine.speculative_runtime_poisoned
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
                "model_identity": tp1_worker._model_identity(
                    engine,
                    target_digest=target_digest,
                    mtp_digest=mtp_digest,
                ),
                "prompt_rows": prompt_rows,
                "output_rows": output_rows,
                "tp1_output_rows": tp1_output_rows,
                "rank_snapshots": rank_snapshots,
                "side_state_receipts": side_state_receipts,
                "target_kv_receipts": target_kv_receipts,
                "runtime_poisoned": runtime_poisoned,
                "cleanup": None,
            }
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            if engine is not None:
                exit_receipt = engine.exit()
    if cell is None or not isinstance(exit_receipt, dict):
        raise RuntimeError(
            "TP4 worker did not produce a complete cell"
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
    parser.add_argument("--tp1-result", required=True)
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
        tp1_result_path=Path(args.tp1_result),
        **_default_dependencies(),
    )
    gate.atomic_write_json(Path(args.out), result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
