from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import importlib.util
import os
from pathlib import Path
import sys


def _load_gate_module():
    path = (
        Path(__file__).resolve().parent
        / "qwen35_generic_speculative_tp4_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "qwen35_generic_speculative_tp4_gate",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate_module()

DEFAULT_PROMPT_SEEDS = (
    " repeated alpha 0 beta gamma delta",
    " repeated alpha 1 beta gamma delta",
    " repeated alpha 2 beta gamma delta",
    " repeated alpha 3 beta gamma delta",
)


def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_gpu_indices(
    value: object,
) -> tuple[int, ...]:
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


def _integer_mapping(
    value: object,
    name: str,
) -> dict[int, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    normalized = {}
    for key, count in value.items():
        try:
            sequence_id = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{name} sequence ID is invalid"
            ) from error
        if (
            sequence_id < 0
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise ValueError(f"{name} count is invalid")
        normalized[sequence_id] = count
    return normalized


def _token_id_mapping(
    value: object,
    name: str,
) -> dict[int, list[int]]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    normalized = {}
    for key, token_ids in value.items():
        try:
            sequence_id = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{name} sequence ID is invalid"
            ) from error
        if (
            sequence_id < 0
            or not isinstance(token_ids, list)
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in token_ids
            )
        ):
            raise ValueError(f"{name} token IDs are invalid")
        normalized[sequence_id] = list(token_ids)
    return normalized


def normalize_side_state_receipts(
    receipts: list[dict],
    *,
    rank: int,
) -> list[dict]:
    rank = gate._integer(rank, "side-state rank")
    if rank >= gate.WORLD_SIZE:
        raise ValueError("side-state rank is out of range")
    if not isinstance(receipts, list):
        raise ValueError(
            "side-state receipts must be a list"
        )
    normalized = []
    for receipt in receipts:
        if not isinstance(receipt, dict):
            raise ValueError(
                "side-state receipt must be a mapping"
            )
        transaction_id = receipt.get("transaction_id")
        operation = receipt.get("operation")
        status = receipt.get("status")
        sequence_ids = receipt.get("sequence_ids")
        if (
            not isinstance(transaction_id, str)
            or not transaction_id
            or operation
            not in {
                "prepare",
                "select",
                "apply",
                "seal",
                "rollback",
            }
            or not isinstance(status, str)
            or not status
            or not isinstance(sequence_ids, list)
            or not sequence_ids
        ):
            raise ValueError(
                "side-state receipt is incomplete"
            )
        for sequence_id in sequence_ids:
            normalized.append({
                "rank": rank,
                "sequence_id": gate._integer(
                    sequence_id,
                    "receipt sequence ID",
                ),
                "handle_id": transaction_id,
                "operation": operation,
                "state": status,
            })
    return normalized


def _encode_seed(tokenizer, text: str) -> list[int]:
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )
    if (
        not isinstance(token_ids, (list, tuple))
        or len(token_ids) < 4
        or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in token_ids
        )
    ):
        raise RuntimeError(
            "tokenizer did not produce a usable prompt seed"
        )
    return [int(token_id) for token_id in token_ids]


def build_prompt_rows(
    tokenizer,
    batch_size: int,
) -> list[dict]:
    gate.cell_key("baseline", batch_size)
    rows = []
    for prompt_index, seed in enumerate(
        DEFAULT_PROMPT_SEEDS[:batch_size]
    ):
        acceptance = _encode_seed(tokenizer, seed)
        divergence = _encode_seed(
            tokenizer,
            f" divergent omega {prompt_index} sigma tau lambda",
        )
        pattern = acceptance + divergence
        token_ids = (
            pattern
            * (
                (
                    gate.CONTEXT_TOKENS
                    // len(pattern)
                )
                + 1
            )
        )[:gate.CONTEXT_TOKENS]
        token_ids[-len(acceptance):] = acceptance
        rows.append({
            "prompt_index": prompt_index,
            "token_count": len(token_ids),
            "token_ids": token_ids,
            "sha256": gate._json_sha256(token_ids),
        })
    return rows


def _model_identity(engine) -> dict:
    config = getattr(engine, "config", None)
    if config is None:
        config = getattr(
            getattr(engine, "model_runner", None),
            "config",
            None,
        )
    hf_config = getattr(config, "hf_config", None)
    if hf_config is None:
        raise RuntimeError(
            "loaded model Hugging Face config is unavailable"
        )
    text_config = getattr(
        hf_config,
        "text_config",
        hf_config,
    )
    layer_types = tuple(
        getattr(text_config, "layer_types", ())
    )
    return {
        "model_type": str(
            getattr(hf_config, "model_type", "")
        ),
        "architectures": list(
            getattr(hf_config, "architectures", ()) or ()
        ),
        "text_layer_count": len(layer_types),
        "linear_layer_count": layer_types.count(
            "linear_attention"
        ),
        "full_attention_layer_count": layer_types.count(
            "full_attention"
        ),
    }


@contextmanager
def capture_rank_side_state_receipts(engine):
    model_runner = engine.model_runner
    original_call = model_runner.call
    ranked = {
        rank: []
        for rank in range(gate.WORLD_SIZE)
    }
    method_names = {
        "prepare_speculative_side_state_batch",
        "select_speculative_side_state_batch",
        "apply_speculative_side_state_batch",
        "seal_speculative_side_state_batch",
        "rollback_speculative_side_state_batch",
    }

    def recorded_call(method_name, *args):
        if method_name not in method_names:
            return original_call(method_name, *args)
        local, worker_acks = (
            engine.call_model_runner_acknowledged(
                method_name,
                *args,
                timeout_s=60.0,
            )
        )
        ranked[0].append(dict(local))
        for acknowledgement in worker_acks:
            rank = gate._integer(
                getattr(acknowledgement, "rank", None),
                "side-state acknowledgement rank",
            )
            result = getattr(
                acknowledgement,
                "result",
                None,
            )
            if (
                rank == 0
                or rank >= gate.WORLD_SIZE
                or not isinstance(result, dict)
            ):
                raise RuntimeError(
                    "side-state acknowledgement is invalid"
                )
            ranked[rank].append(dict(result))
        if any(not ranked[rank] for rank in range(gate.WORLD_SIZE)):
            raise RuntimeError(
                "side-state acknowledgement ranks are incomplete"
            )
        return local

    model_runner.call = recorded_call
    try:
        yield ranked
    finally:
        model_runner.call = original_call


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


def summarize_step_observations(
    observations: list[dict],
    *,
    rank_receipts: dict[int, list[dict]],
    policy: str,
    batch_size: int,
) -> dict:
    gate.cell_key(policy, batch_size)
    if (
        not isinstance(rank_receipts, dict)
        or set(rank_receipts) != set(range(gate.WORLD_SIZE))
    ):
        raise ValueError(
            "rank receipt inventory mismatch"
        )
    normalized_receipts = []
    for rank in range(gate.WORLD_SIZE):
        normalized_receipts.extend(
            normalize_side_state_receipts(
                rank_receipts[rank],
                rank=rank,
            )
        )
    runtime = {
        "proposal_rows": 0,
        "proposed_tokens": 0,
        "accepted_draft_tokens": 0,
        "rejected_draft_tokens": 0,
        "first_target_callbacks": 0,
        "verify_callbacks": 0,
        "accepted_prefix_replays": 0,
        "consumed_input_mappings": [],
    }
    mappings = []
    transaction_rows = []
    transaction_ordinals = {
        sequence_id: 0
        for sequence_id in range(batch_size)
    }
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError(
                "engine observation must be a mapping"
            )
        proposal_counts = _integer_mapping(
            observation.get(
                "speculative_proposal_token_counts",
                {},
            ),
            "proposal counts",
        )
        accepted_counts = _integer_mapping(
            observation.get(
                "speculative_accepted_draft_token_counts",
                {},
            ),
            "accepted counts",
        )
        proposal_token_ids = _token_id_mapping(
            observation.get(
                "speculative_proposal_token_ids_by_seq",
                {},
            ),
            "proposal token IDs",
        )
        accepted_token_ids = _token_id_mapping(
            observation.get(
                "speculative_accepted_draft_token_ids_by_seq",
                {},
            ),
            "accepted draft token IDs",
        )
        if (
            set(proposal_token_ids) != set(proposal_counts)
            or set(accepted_token_ids)
            != set(accepted_counts)
        ):
            raise ValueError(
                "speculative token ID inventory mismatch"
            )
        runtime["proposal_rows"] += gate._integer(
            observation.get(
                "speculative_proposal_row_count",
                0,
            ),
            "proposal row count",
        )
        runtime["first_target_callbacks"] += gate._integer(
            observation.get(
                "speculative_first_target_callback_count",
                0,
            ),
            "first-target callback count",
        )
        runtime["verify_callbacks"] += gate._integer(
            observation.get(
                "speculative_fixed_q_group_count",
                0,
            ),
            "verify callback count",
        )
        for sequence_id, proposal_count in proposal_counts.items():
            if sequence_id not in transaction_ordinals:
                raise ValueError(
                    "proposal sequence inventory mismatch"
                )
            accepted_count = accepted_counts.get(
                sequence_id,
                0,
            )
            if accepted_count > proposal_count:
                raise ValueError(
                    "accepted count exceeds proposal count"
                )
            verify_count = max(0, proposal_count - 1)
            committed_tail = min(
                accepted_count,
                verify_count,
            )
            proposal_ids = proposal_token_ids[sequence_id]
            accepted_ids = accepted_token_ids[sequence_id]
            if (
                len(proposal_ids) != proposal_count
                or len(accepted_ids) != accepted_count
                or accepted_ids
                != proposal_ids[:accepted_count]
            ):
                raise ValueError(
                    "speculative token ID counts mismatch"
                )
            transaction_ordinal = transaction_ordinals[
                sequence_id
            ]
            transaction_ordinals[sequence_id] += 1
            mapping = {
                "sequence_id": sequence_id,
                "transaction_ordinal": transaction_ordinal,
                "proposal_token_count": proposal_count,
                "accepted_draft_count": accepted_count,
                "verify_input_count": verify_count,
                "committed_tail_input_count": committed_tail,
                "committed_input_count": 1 + committed_tail,
            }
            mappings.append(mapping)
            transaction_rows.append({
                **mapping,
                "proposal_token_ids": proposal_ids,
                "acceptance_mask": (
                    [True] * accepted_count
                    + [False]
                    * (proposal_count - accepted_count)
                ),
            })
            runtime["proposed_tokens"] += proposal_count
            runtime["accepted_draft_tokens"] += accepted_count
            runtime["rejected_draft_tokens"] += (
                proposal_count - accepted_count
            )
    runtime["consumed_input_mappings"] = mappings
    if policy == "baseline":
        if mappings or normalized_receipts:
            raise ValueError(
                "baseline speculative evidence must be empty"
            )
        return {
            "runtime": runtime,
            "rank_side_state_receipts": [],
            "rank_evidence": [
                {
                    "rank": rank,
                    "checkpoint_loaded": True,
                    "transactions": [],
                    "side_state_receipts": [],
                    "failure_path_rollbacks": [],
                }
                for rank in range(gate.WORLD_SIZE)
            ],
        }

    candidate_observation_count = sum(
        1
        for observation in observations
        if observation.get(
            "speculative_proposal_token_counts",
            {},
        )
    )
    rank_evidence = []
    for rank in range(gate.WORLD_SIZE):
        raw_receipts = rank_receipts[rank]
        groups = []
        by_handle = {}
        for receipt in raw_receipts:
            if not isinstance(receipt, dict):
                raise ValueError(
                    "side-state receipt must be a mapping"
                )
            handle_id = receipt.get("transaction_id")
            if (
                not isinstance(handle_id, str)
                or not handle_id
            ):
                raise ValueError(
                    "side-state transaction ID is invalid"
                )
            if handle_id not in by_handle:
                by_handle[handle_id] = []
                groups.append(by_handle[handle_id])
            by_handle[handle_id].append(receipt)
        if len(groups) != candidate_observation_count:
            raise ValueError(
                "side-state transaction receipt count mismatch"
            )
        checkpoint_by_transaction = {}
        for transaction_index, group in enumerate(groups):
            select_receipts = [
                receipt
                for receipt in group
                if receipt.get("operation") == "select"
            ]
            if len(select_receipts) != 1:
                raise ValueError(
                    "side-state select receipt is missing"
                )
            rows = select_receipts[0].get("rows")
            if not isinstance(rows, list):
                raise ValueError(
                    "side-state select rows are missing"
                )
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(
                        "side-state select row is invalid"
                    )
                sequence_id = gate._integer(
                    row.get("sequence_id"),
                    "side-state select sequence ID",
                )
                checkpoint = gate._integer(
                    row.get("checkpoint_index"),
                    "side-state checkpoint index",
                    minimum=1,
                )
                checkpoint_by_transaction[
                    (sequence_id, transaction_index)
                ] = checkpoint
        transactions = []
        for row in transaction_rows:
            key = (
                row["sequence_id"],
                row["transaction_ordinal"],
            )
            checkpoint = checkpoint_by_transaction.get(key)
            if checkpoint is None:
                raise ValueError(
                    "side-state selected checkpoint is missing"
                )
            transactions.append(
                gate._validate_sequence_transaction({
                    "rank": rank,
                    "cell_key": gate.cell_key(
                        policy,
                        batch_size,
                    ),
                    **row,
                    "kv_decision": (
                        "commit_prefix_"
                        f"{row['committed_input_count']}"
                        "_rollback_suffix"
                    ),
                    "selected_checkpoint_id": (
                        f"checkpoint:{checkpoint}"
                    ),
                }, rank=rank)
            )
        rank_evidence.append({
            "rank": rank,
            "checkpoint_loaded": True,
            "transactions": transactions,
            "side_state_receipts": (
                normalize_side_state_receipts(
                    raw_receipts,
                    rank=rank,
                )
            ),
            "failure_path_rollbacks": [],
        })
    return {
        "runtime": runtime,
        "rank_side_state_receipts": normalized_receipts,
        "rank_evidence": rank_evidence,
    }


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
            outputs_by_id[int(sequence_id)] = [
                int(token_id)
                for token_id in token_ids
            ]
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
        "output_rows": [
            {
                "prompt_index": prompt_index,
                "token_count": len(token_ids),
                "token_ids": token_ids,
                "sha256": gate._json_sha256(token_ids),
            }
            for prompt_index, token_ids in enumerate(outputs)
        ],
        "observations": observations,
    }


def _movement_delta(before_rows, after_rows):
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


def _cleanup_observations(engine) -> list[dict]:
    local, worker_acks = (
        engine.call_model_runner_acknowledged(
            "speculative_cleanup_observation",
            timeout_s=60.0,
        )
    )
    ranked = [(0, local)]
    ranked.extend(
        (ack.rank, ack.result)
        for ack in worker_acks
    )
    rows = {}
    for outer_rank, row in ranked:
        if (
            not isinstance(row, dict)
            or row.get("rank") != outer_rank
            or outer_rank in rows
        ):
            raise RuntimeError(
                "cleanup observation acknowledgement is invalid"
            )
        rows[outer_rank] = {
            "rank": outer_rank,
            "active_transaction_count": gate._integer(
                row.get("active_transaction_count"),
                "active transaction count",
            ),
            "live_lease_count": gate._integer(
                row.get("live_lease_count"),
                "live lease count",
            ),
        }
    if tuple(sorted(rows)) != gate.EXPECTED_RANKS:
        raise RuntimeError(
            "cleanup observation ranks are incomplete"
        )
    return [
        rows[rank]
        for rank in gate.EXPECTED_RANKS
    ]


def _merge_cleanup_receipt(
    exit_receipt: object,
    observations: list[dict],
    *,
    runtime_poisoned: bool,
) -> dict:
    if not isinstance(exit_receipt, dict):
        raise RuntimeError(
            "Engine cleanup receipt is unavailable"
        )
    exit_rows = exit_receipt.get(
        "rank_cleanup_receipts"
    )
    if not isinstance(exit_rows, list):
        raise RuntimeError(
            "Engine rank cleanup receipts are unavailable"
        )
    exit_by_rank = {
        row.get("rank"): row
        for row in exit_rows
        if isinstance(row, dict)
    }
    observation_by_rank = {
        row["rank"]: row
        for row in observations
    }
    exit_codes = exit_receipt.get("rank_exit_codes")
    if (
        set(exit_by_rank) != set(gate.EXPECTED_RANKS)
        or set(observation_by_rank)
        != set(gate.EXPECTED_RANKS)
        or not isinstance(exit_codes, list)
        or len(exit_codes) != gate.WORLD_SIZE
    ):
        raise RuntimeError(
            "Engine cleanup rank inventory is incomplete"
        )
    rank_rows = []
    for rank in gate.EXPECTED_RANKS:
        exit_row = exit_by_rank[rank]
        observed = observation_by_rank[rank]
        rank_rows.append({
            "rank": rank,
            "worker_exit_code": exit_codes[rank],
            "process_group_initialized": not bool(
                exit_row.get("process_group_destroyed")
            ),
            "engine_exit_called": True,
            "live_lease_count": observed[
                "live_lease_count"
            ],
            "prepared_transaction_count": observed[
                "active_transaction_count"
            ],
            "runtime_poisoned": runtime_poisoned,
        })
    return {
        "process_group_destroyed": bool(
            exit_receipt.get("process_group_destroyed")
        ),
        "rank_exit_codes": list(exit_codes),
        "owned_children_remaining": list(
            exit_receipt.get(
                "owned_children_remaining",
                [],
            )
        ),
        "rank_cleanup_receipts": rank_rows,
    }


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
    cleanup_observations = None
    cleanup_receipt = None
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
            engine.configure_decode_internal_profile(
                True,
                f"qwen35_generic_tp4/{policy}/b{batch_size}",
                timeout_s=60.0,
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
            before_rows = engine.kv_offload_summaries(
                timeout_s=60.0
            )
            with capture_residency_phases(
                engine
            ) as residency_phases:
                with capture_rank_side_state_receipts(
                    engine
                ) as rank_receipts:
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
            engine.flush_pending_hybrid_state_releases(
                timeout_s=60.0
            )
            summary = summarize_step_observations(
                recorded["observations"],
                rank_receipts=rank_receipts,
                policy=policy,
                batch_size=batch_size,
            )
            profile = (
                engine.finalize_decode_internal_profile(
                    timeout_s=60.0
                )
            )
            cleanup_observations = (
                _cleanup_observations(engine)
            )
            runtime_poisoned = bool(
                engine.speculative_runtime_poisoned
            )
            cell = {
                "schema_version": gate.SCHEMA_VERSION,
                "classification": gate.CLASSIFICATION,
                "policy": policy,
                "context_tokens": gate.CONTEXT_TOKENS,
                "batch_size": batch_size,
                "world_size": gate.WORLD_SIZE,
                "model_identity": _model_identity(engine),
                "prompt_rows": prompt_rows,
                "output_rows": recorded["output_rows"],
                "runtime": summary["runtime"],
                "rank_evidence": summary["rank_evidence"],
                "profile": profile,
                "kv_rank_deltas": _movement_delta(
                    before_rows,
                    after_rows,
                ),
                "residency_phases": list(
                    residency_phases
                ),
            }
        finally:
            if engine is not None:
                cleanup_receipt = engine.exit()
    if cell is None or cleanup_observations is None:
        raise RuntimeError(
            "TP4 cell execution did not produce a result"
        )
    cell["cleanup_receipt"] = _merge_cleanup_receipt(
        cleanup_receipt,
        cleanup_observations,
        runtime_poisoned=runtime_poisoned,
    )
    return gate.validate_cell_result(cell)


def _default_dependencies():
    import torch

    from tinyvllm import LLM
    from tinyvllm.engine.speculative_runtime import EngineSpeculativeRuntime
    from tinyvllm.sampling_params import SamplingParams
    from tinyvllm.speculative.ngram_adapter import NGramDraftAdapter

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
        choices=gate.POLICIES,
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=gate.BATCH_SIZES,
        required=True,
    )
    parser.add_argument("--dist-port", type=int, required=True)
    parser.add_argument("--master-port", type=int, required=True)
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
    Path(args.out).write_text(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
