from __future__ import annotations

import argparse
import hashlib
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


SCHEMA_VERSION = (
    "qwen35.generic-speculative-tp4-"
    "transactional-correctness.v1"
)
CLASSIFICATION = "SECOND_MODEL_TP4_4K_ESTABLISHED"
CLAIM_SCOPE = "second_model_tp4_4k_only"
LIMITATIONS = (
    "phase1_not_promotable",
    "context_16k_not_established",
    "context_32k_not_established",
    "performance_not_established",
    "learned_drafter_not_established",
    "kv_quantization_not_established",
)
POLICIES = ("baseline", "ngram")
BATCH_SIZES = (1, 4)
WORLD_SIZE = 4
CONTEXT_TOKENS = 4096
NGRAM_SIZE = 3
MAX_PROPOSAL_TOKENS = 4
MAX_OUTPUT_TOKENS = 8
MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38fe"
    "69982efedaf4a3e631ad0b14aad7dd0"
)
DEFAULT_SOURCE_FILES = (
    "tinyvllm/engine/llm_engine.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/qwen35_speculative_state.py",
    "tinyvllm/engine/speculative_model_runner.py",
    "tinyvllm/engine/speculative_side_state.py",
    "tinyvllm/layers/gated_delta.py",
    "tinyvllm/layers/qwen35_linear_attention.py",
    "tinyvllm/layers/qwen35_packed_layer_stack.py",
    "tinyvllm/models/qwen35_packed.py",
    "tinyvllm/speculative/batch_runtime.py",
    "tools/qwen35_generic_speculative_tp4_gate.py",
    "tools/qwen35_generic_speculative_tp4_worker.py",
    "tools/verify_qwen35_generic_speculative_tp4_gate.py",
)
EXPECTED_RANKS = tuple(range(WORLD_SIZE))
SPECULATIVE_BATCH_KINDS = (
    "spec_first_target",
    "spec_verify",
)
SUCCESSFUL_RESIDENCY_OPERATIONS = (
    "prepare",
    "precommit",
    "seal",
)
RESIDENCY_STATUS = {
    "prepare": "prepared",
    "precommit": "precommitted",
    "seal": "sealed",
}
RESIDENCY_ROW_FIELDS = {
    "ticket_id",
    "participant_id",
    "operation",
    "status",
    "sequence_ids",
    "committed_block_identities",
    "rejected_block_identities",
    "detail",
}
MOVEMENT_KEYS = (
    "h2d_copies",
    "h2d_bytes",
    "d2h_copies",
    "d2h_bytes",
    "copy_waits",
    "evictions",
    "evict_clean",
    "speculative_residency_committed_blocks",
    "speculative_residency_rejected_blocks",
    "speculative_residency_rejected_d2h_copies",
)


def cell_key(policy: str, batch_size: int) -> str:
    if policy not in POLICIES:
        raise ValueError("unsupported policy")
    if batch_size not in BATCH_SIZES:
        raise ValueError("unsupported batch size")
    return f"{policy}:b{batch_size}"


def _integer(
    value: object,
    name: str,
    *,
    minimum: int = 0,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_model_identity(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("model identity must be a mapping")
    if value.get("model_type") != "qwen3_5":
        raise ValueError("model type must be qwen3_5")
    architectures = value.get("architectures")
    if architectures != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("model architecture inventory mismatch")
    text_layers = _integer(
        value.get("text_layer_count"),
        "text layer count",
        minimum=1,
    )
    linear_layers = _integer(
        value.get("linear_layer_count"),
        "linear layer count",
        minimum=1,
    )
    full_layers = _integer(
        value.get("full_attention_layer_count"),
        "full attention layer count",
        minimum=1,
    )
    if (
        text_layers != 24
        or linear_layers != 18
        or full_layers != 6
    ):
        raise ValueError("Qwen3.5 hybrid layer inventory mismatch")
    return {
        "model_type": "qwen3_5",
        "architectures": list(architectures),
        "text_layer_count": text_layers,
        "linear_layer_count": linear_layers,
        "full_attention_layer_count": full_layers,
    }


def _validate_mapping(row: object) -> dict:
    if not isinstance(row, dict):
        raise ValueError(
            "consumed input mapping must be a mapping"
        )
    proposal_count = _integer(
        row.get("proposal_token_count"),
        "proposal token count",
        minimum=1,
    )
    accepted_count = _integer(
        row.get("accepted_draft_count"),
        "accepted draft count",
    )
    if accepted_count > proposal_count:
        raise ValueError(
            "accepted draft count exceeds proposal count"
        )
    verify_count = max(0, proposal_count - 1)
    committed_tail = min(accepted_count, verify_count)
    committed_input = 1 + committed_tail
    if row.get("verify_input_count") != verify_count:
        raise ValueError("verify input count mismatch")
    if (
        row.get("committed_tail_input_count")
        != committed_tail
    ):
        raise ValueError(
            "committed tail input count mismatch"
        )
    if row.get("committed_input_count") != committed_input:
        raise ValueError("committed input count mismatch")
    return {
        "sequence_id": _integer(
            row.get("sequence_id"),
            "sequence ID",
        ),
        "transaction_ordinal": _integer(
            row.get("transaction_ordinal"),
            "transaction ordinal",
        ),
        "proposal_token_count": proposal_count,
        "accepted_draft_count": accepted_count,
        "verify_input_count": verify_count,
        "committed_tail_input_count": committed_tail,
        "committed_input_count": committed_input,
    }


def _transaction_semantic_digest(row: dict) -> str:
    if not isinstance(row, dict):
        raise ValueError(
            "sequence transaction must be a mapping"
        )
    semantic_row = {
        "cell_key": row.get("cell_key"),
        "sequence_id": row.get("sequence_id"),
        "transaction_ordinal": row.get(
            "transaction_ordinal"
        ),
        "proposal_token_ids": row.get("proposal_token_ids"),
        "acceptance_mask": row.get("acceptance_mask"),
        "proposal_token_count": row.get(
            "proposal_token_count"
        ),
        "accepted_draft_count": row.get(
            "accepted_draft_count"
        ),
        "verify_input_count": row.get(
            "verify_input_count"
        ),
        "committed_tail_input_count": row.get(
            "committed_tail_input_count"
        ),
        "committed_input_count": row.get(
            "committed_input_count"
        ),
        "kv_decision": row.get("kv_decision"),
        "selected_checkpoint_id": row.get(
            "selected_checkpoint_id"
        ),
    }
    return hashlib.sha256(
        json.dumps(
            semantic_row,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_sequence_transaction(
    row: object,
    *,
    rank: int,
) -> dict:
    if not isinstance(row, dict):
        raise ValueError(
            "sequence transaction must be a mapping"
        )
    expected_rank = _integer(rank, "expected rank")
    actual_rank = _integer(row.get("rank"), "transaction rank")
    if actual_rank != expected_rank:
        raise ValueError("transaction rank mismatch")
    mapping = _validate_mapping(row)
    proposal_ids = row.get("proposal_token_ids")
    acceptance_mask = row.get("acceptance_mask")
    if (
        not isinstance(proposal_ids, list)
        or len(proposal_ids)
        != mapping["proposal_token_count"]
        or any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in proposal_ids
        )
    ):
        raise ValueError("proposal token IDs are invalid")
    if (
        not isinstance(acceptance_mask, list)
        or len(acceptance_mask) != len(proposal_ids)
        or any(
            not isinstance(accepted, bool)
            for accepted in acceptance_mask
        )
        or sum(acceptance_mask)
        != mapping["accepted_draft_count"]
    ):
        raise ValueError("acceptance mask is invalid")
    cell = row.get("cell_key")
    kv_decision = row.get("kv_decision")
    checkpoint = row.get("selected_checkpoint_id")
    if not isinstance(cell, str) or not cell:
        raise ValueError("transaction cell key is invalid")
    if not isinstance(kv_decision, str) or not kv_decision:
        raise ValueError("KV decision is invalid")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(
            "selected checkpoint identity is invalid"
        )
    normalized = {
        "rank": actual_rank,
        "cell_key": cell,
        **mapping,
        "proposal_token_ids": list(proposal_ids),
        "acceptance_mask": list(acceptance_mask),
        "kv_decision": kv_decision,
        "selected_checkpoint_id": checkpoint,
    }
    normalized["semantic_digest"] = (
        _transaction_semantic_digest(normalized)
    )
    return normalized


def _validate_side_state(
    receipts: object,
    failure_rollbacks: object,
    *,
    rank: int,
) -> list[dict]:
    expected_rank = _integer(rank, "expected rank")
    if not isinstance(receipts, list):
        raise ValueError(
            "side-state receipts must be a list"
        )
    if not isinstance(failure_rollbacks, list):
        raise ValueError(
            "failure-path rollbacks must be a list"
        )
    normalized = []
    by_lifecycle: dict[
        tuple[int, str, int],
        list[str],
    ] = {}
    for receipt in receipts:
        if not isinstance(receipt, dict):
            raise ValueError(
                "side-state receipt must be a mapping"
            )
        receipt_rank = _integer(
            receipt.get("rank"),
            "side-state receipt rank",
        )
        if receipt_rank >= WORLD_SIZE:
            raise ValueError(
                "side-state receipt rank is out of range"
            )
        handle_id = receipt.get("handle_id")
        sequence_id = receipt.get("sequence_id")
        operation = receipt.get("operation")
        state = receipt.get("state")
        if (
            not isinstance(handle_id, str)
            or not handle_id
            or isinstance(sequence_id, bool)
            or not isinstance(sequence_id, int)
            or sequence_id < 0
            or operation
            not in {
                "prepare",
                "select",
                "apply",
                "seal",
                "rollback",
            }
            or not isinstance(state, str)
            or not state
        ):
            raise ValueError("side-state receipt is invalid")
        expected_state = {
            "prepare": "prepared",
            "select": "selected",
            "apply": "applied",
            "seal": "sealed",
            "rollback": "rolled_back",
        }[operation]
        if state != expected_state:
            raise ValueError(
                "side-state lifecycle state mismatch"
            )
        key = (receipt_rank, handle_id, sequence_id)
        by_lifecycle.setdefault(key, []).append(operation)
        normalized.append({
            "rank": receipt_rank,
            "handle_id": handle_id,
            "sequence_id": sequence_id,
            "operation": operation,
            "state": state,
        })
    if not any(
        receipt_rank == expected_rank
        for receipt_rank, _, _ in by_lifecycle
    ):
        raise ValueError(
            "expected rank side-state receipt is missing"
        )
    required = ["prepare", "select", "apply", "seal"]
    successful = [
        operations
        for operations in by_lifecycle.values()
        if "seal" in operations
    ]
    if not successful or any(
        operations != required
        for operations in successful
    ):
        raise ValueError(
            "side-state lifecycle receipts are incomplete"
        )
    rollback_keys = {
        key
        for key, operations in by_lifecycle.items()
        if "rollback" in operations
    }
    for rollback in failure_rollbacks:
        if not isinstance(rollback, dict):
            raise ValueError(
                "failure-path rollback receipt is invalid"
            )
        key = (
            rollback.get("rank"),
            rollback.get("handle_id"),
            rollback.get("sequence_id"),
        )
        if key not in rollback_keys:
            raise ValueError(
                "failure-path rollback receipt is missing"
            )
    return normalized


def _callback_identity(step: dict) -> tuple:
    return (
        step["step_index"],
        step["batch_kind"],
        step["is_decode"],
        step["decode_ordinal"],
        step["active_sequence_count"],
        step["request_set_sha256"],
        step["dispatch"],
    )


def _collective_identity(row: dict) -> tuple:
    return (
        row["step_index"],
        row["decode_ordinal"],
        row["operation"],
        tuple(row["tensor_shape"]),
        row["tensor_dtype"],
    )


def _validate_profile_step(step: object, rank: int) -> dict:
    if not isinstance(step, dict):
        raise ValueError(
            "profile callback row must be a mapping"
        )
    if step.get("rank") != rank:
        raise ValueError("profile callback rank mismatch")
    _integer(
        step.get("step_index"),
        "profile callback step index",
    )
    batch_kind = step.get("batch_kind")
    if not isinstance(batch_kind, str) or not batch_kind:
        raise ValueError(
            "profile callback batch kind must be non-empty"
        )
    is_decode = step.get("is_decode")
    if not isinstance(is_decode, bool):
        raise ValueError(
            "profile callback decode flag must be a boolean"
        )
    if (
        batch_kind in SPECULATIVE_BATCH_KINDS
        and not is_decode
    ):
        raise ValueError(
            "speculative callback must be a decode execution"
        )
    decode_ordinal = step.get("decode_ordinal")
    if is_decode:
        _integer(
            decode_ordinal,
            "profile callback decode ordinal",
        )
    elif decode_ordinal is not None:
        raise ValueError(
            "profile prefill decode ordinal must be null"
        )
    if _integer(
        step.get("active_sequence_count"),
        "profile callback active sequence count",
    ) <= 0:
        raise ValueError(
            "profile callback active sequence count must be positive"
        )
    _sha256(
        step.get("request_set_sha256"),
        "profile callback request set",
    )
    if step.get("dispatch") != "eager":
        raise ValueError(
            "profile callback dispatch must be eager"
        )
    for name in (
        "wall_ns",
        "cuda_ns",
        "non_cuda_upper_bound_ns",
    ):
        _integer(
            step.get(name),
            f"profile callback {name}",
        )
    return dict(step)


def _validate_profile_collective(
    row: object,
    rank: int,
) -> dict:
    if not isinstance(row, dict):
        raise ValueError(
            "profile collective row must be a mapping"
        )
    if row.get("rank") != rank:
        raise ValueError("profile collective rank mismatch")
    _integer(
        row.get("step_index"),
        "profile collective step index",
    )
    _integer(
        row.get("decode_ordinal"),
        "profile collective decode ordinal",
    )
    operation = row.get("operation")
    if not isinstance(operation, str) or not operation:
        raise ValueError(
            "profile collective operation must be non-empty"
        )
    tensor_shape = row.get("tensor_shape")
    if (
        not isinstance(tensor_shape, list)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in tensor_shape
        )
    ):
        raise ValueError(
            "profile collective tensor shape is invalid"
        )
    tensor_dtype = row.get("tensor_dtype")
    if not isinstance(tensor_dtype, str) or not tensor_dtype:
        raise ValueError(
            "profile collective tensor dtype must be non-empty"
        )
    for name in ("wall_ns", "cuda_ns"):
        _integer(
            row.get(name),
            f"profile collective {name}",
        )
    return dict(row)


def validate_rank_profile(
    value: object,
    *,
    policy: str,
) -> dict:
    if policy not in POLICIES:
        raise ValueError("unsupported policy")
    if not isinstance(value, dict):
        raise ValueError("rank profile must be a mapping")
    if value.get("enabled") is not True:
        raise ValueError("rank profile must be enabled")
    if value.get("rank_inventory") != list(EXPECTED_RANKS):
        raise ValueError(
            "rank profile rank inventory mismatch"
        )
    rank_rows = value.get("ranks")
    if not isinstance(rank_rows, list):
        raise ValueError(
            "rank profile rank inventory is invalid"
        )
    rows_by_rank = {}
    for row in rank_rows:
        if not isinstance(row, dict):
            raise ValueError(
                "rank profile row must be a mapping"
            )
        rank = row.get("rank")
        if rank in rows_by_rank:
            raise ValueError(
                "rank profile rank inventory is duplicated"
            )
        if rank not in EXPECTED_RANKS:
            raise ValueError(
                "rank profile rank inventory mismatch"
            )
        if (
            row.get("enabled") is not True
            or row.get("finalization_status") != "complete"
        ):
            raise ValueError("rank profile row is incomplete")
        steps = row.get("steps")
        collectives = row.get("collectives")
        if not isinstance(steps, list):
            raise ValueError(
                "rank profile callbacks are invalid"
            )
        if not isinstance(collectives, list):
            raise ValueError(
                "rank profile collectives are invalid"
            )
        rows_by_rank[rank] = {
            **dict(row),
            "steps": [
                _validate_profile_step(step, rank)
                for step in steps
            ],
            "collectives": [
                _validate_profile_collective(
                    collective,
                    rank,
                )
                for collective in collectives
            ],
        }
    if tuple(sorted(rows_by_rank)) != EXPECTED_RANKS:
        raise ValueError(
            "rank profile rank inventory mismatch"
        )

    callback_identities = {}
    collective_identities = {}
    for rank in EXPECTED_RANKS:
        row = rows_by_rank[rank]
        speculative_steps = [
            step
            for step in row["steps"]
            if step["batch_kind"]
            in SPECULATIVE_BATCH_KINDS
        ]
        if policy == "baseline":
            if speculative_steps:
                raise ValueError(
                    "baseline speculative callback is forbidden"
                )
            callback_identities[rank] = ()
            collective_identities[rank] = ()
            continue
        kinds = {
            step["batch_kind"]
            for step in speculative_steps
        }
        if kinds != set(SPECULATIVE_BATCH_KINDS):
            raise ValueError(
                "candidate callback inventory is incomplete"
            )
        callback_identities[rank] = tuple(
            _callback_identity(step)
            for step in speculative_steps
        )
        step_indices = {
            step["step_index"]
            for step in speculative_steps
        }
        speculative_collectives = [
            collective
            for collective in row["collectives"]
            if collective["step_index"] in step_indices
        ]
        if (
            not speculative_collectives
            or {
                row["step_index"]
                for row in speculative_collectives
            }
            != step_indices
        ):
            raise ValueError(
                "candidate collective coverage is incomplete"
            )
        collective_identities[rank] = tuple(
            _collective_identity(row)
            for row in speculative_collectives
        )
    callback_reference = callback_identities[0]
    if any(
        callback_identities[rank] != callback_reference
        for rank in EXPECTED_RANKS[1:]
    ):
        raise ValueError(
            "candidate callback identity mismatch"
        )
    collective_reference = collective_identities[0]
    if any(
        collective_identities[rank]
        != collective_reference
        for rank in EXPECTED_RANKS[1:]
    ):
        raise ValueError(
            "candidate collective identity mismatch"
        )
    return {
        "enabled": True,
        "rank_inventory": list(EXPECTED_RANKS),
        "ranks": [
            copy.deepcopy(rows_by_rank[rank])
            for rank in EXPECTED_RANKS
        ],
    }


def _normalize_sequence_ids(value: object) -> list[int]:
    if (
        not isinstance(value, (list, tuple))
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            for item in value
        )
    ):
        raise ValueError("sequence IDs are invalid")
    return list(value)


def _normalize_integer_rows(
    value: object,
    name: str,
) -> list[list[int]]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{name} must be a list or tuple"
        )
    normalized = []
    for row in value:
        if (
            not isinstance(row, (list, tuple))
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 0
                for item in row
            )
        ):
            raise ValueError(f"{name} row is invalid")
        normalized.append(list(row))
    return normalized


def validate_residency_phases(value: object) -> list[dict]:
    if (
        not isinstance(value, list)
        or not value
        or len(value)
        % len(SUCCESSFUL_RESIDENCY_OPERATIONS)
    ):
        raise ValueError(
            "residency phase order must contain complete tickets"
        )
    normalized = []
    active_ticket = None
    for phase_index, phase in enumerate(value):
        if not isinstance(phase, dict):
            raise ValueError(
                "residency phase must be a mapping"
            )
        expected_operation = SUCCESSFUL_RESIDENCY_OPERATIONS[
            phase_index
            % len(SUCCESSFUL_RESIDENCY_OPERATIONS)
        ]
        if phase.get("operation") != expected_operation:
            raise ValueError("residency phase order mismatch")
        expected_status = RESIDENCY_STATUS[
            expected_operation
        ]
        if phase.get("status") != expected_status:
            raise ValueError(
                "residency phase status mismatch"
            )
        ticket_id = _integer(
            phase.get("ticket_id"),
            "residency ticket ID",
        )
        if expected_operation == "prepare":
            active_ticket = ticket_id
        elif ticket_id != active_ticket:
            raise ValueError(
                "residency ticket sequence mismatch"
            )
        rows = phase.get("rows")
        if not isinstance(rows, list):
            raise ValueError(
                "residency rank inventory is invalid"
            )
        rows_by_rank = {}
        reference = None
        for row in rows:
            if (
                not isinstance(row, dict)
                or set(row) != RESIDENCY_ROW_FIELDS
            ):
                raise ValueError(
                    "residency acknowledgement row is invalid"
                )
            rank = row.get("participant_id")
            if rank in rows_by_rank:
                raise ValueError(
                    "residency rank inventory is duplicated"
                )
            if rank not in EXPECTED_RANKS:
                raise ValueError(
                    "residency rank inventory mismatch"
                )
            if (
                row.get("ticket_id") != ticket_id
                or row.get("operation")
                != expected_operation
                or row.get("status") != expected_status
                or row.get("detail") != ""
            ):
                raise ValueError(
                    "residency acknowledgement identity mismatch"
                )
            normalized_row = {
                "ticket_id": ticket_id,
                "participant_id": rank,
                "operation": expected_operation,
                "status": expected_status,
                "sequence_ids": _normalize_sequence_ids(
                    row.get("sequence_ids")
                ),
                "committed_block_identities": (
                    _normalize_integer_rows(
                        row.get(
                            "committed_block_identities"
                        ),
                        "committed block identities",
                    )
                ),
                "rejected_block_identities": (
                    _normalize_integer_rows(
                        row.get(
                            "rejected_block_identities"
                        ),
                        "rejected block identities",
                    )
                ),
                "detail": "",
            }
            non_rank = {
                key: item
                for key, item in normalized_row.items()
                if key != "participant_id"
            }
            if reference is None:
                reference = non_rank
            elif non_rank != reference:
                raise ValueError(
                    "residency acknowledgement rank mismatch"
                )
            rows_by_rank[rank] = normalized_row
        if tuple(sorted(rows_by_rank)) != EXPECTED_RANKS:
            raise ValueError(
                "residency rank inventory mismatch"
            )
        if expected_operation == "prepare" and (
            reference["committed_block_identities"]
            or reference["rejected_block_identities"]
        ):
            raise ValueError(
                "residency prepare cannot classify blocks"
            )
        normalized.append({
            "ticket_id": ticket_id,
            "operation": expected_operation,
            "status": expected_status,
            "rows": [
                rows_by_rank[rank]
                for rank in EXPECTED_RANKS
            ],
        })
    return normalized


def _validate_kv_rank_deltas(value: object) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("KV rank deltas must be a list")
    rows_by_rank = {}
    for row in value:
        if not isinstance(row, dict):
            raise ValueError(
                "KV rank delta must be a mapping"
            )
        rank = row.get("rank")
        if rank in rows_by_rank:
            raise ValueError(
                "KV rank inventory is duplicated"
            )
        if rank not in EXPECTED_RANKS:
            raise ValueError("KV rank inventory mismatch")
        if (
            row.get("provenance")
            != "engine.kv_offload_summaries"
        ):
            raise ValueError(
                "KV movement provenance is invalid"
            )
        normalized = {
            "rank": rank,
            "provenance": "engine.kv_offload_summaries",
            **{
                name: _integer(
                    row.get(name),
                    f"KV movement {name}",
                )
                for name in MOVEMENT_KEYS
            },
        }
        if (
            normalized[
                "speculative_residency_rejected_d2h_copies"
            ]
            != 0
        ):
            raise ValueError(
                "rejected speculative blocks copied to host"
            )
        rows_by_rank[rank] = normalized
    if tuple(sorted(rows_by_rank)) != EXPECTED_RANKS:
        raise ValueError("KV rank inventory mismatch")
    return [
        rows_by_rank[rank]
        for rank in EXPECTED_RANKS
    ]


def validate_cleanup_receipt(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError(
            "cleanup receipt must be a mapping"
        )
    if value.get("process_group_destroyed") is not True:
        raise ValueError(
            "cleanup process group was not destroyed"
        )
    if value.get("rank_exit_codes") != [0, 0, 0, 0]:
        raise ValueError(
            "cleanup rank exit codes are invalid"
        )
    if value.get("owned_children_remaining") != []:
        raise ValueError("cleanup owned children remain")
    rows = value.get("rank_cleanup_receipts")
    if not isinstance(rows, list):
        raise ValueError(
            "cleanup rank inventory is invalid"
        )
    rows_by_rank = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(
                "rank cleanup receipt must be a mapping"
            )
        rank = row.get("rank")
        if rank in rows_by_rank:
            raise ValueError(
                "cleanup rank inventory is duplicated"
            )
        if rank not in EXPECTED_RANKS:
            raise ValueError(
                "cleanup rank inventory mismatch"
            )
        if (
            row.get("worker_exit_code") != 0
            or row.get("process_group_initialized")
            is not False
            or row.get("engine_exit_called") is not True
            or row.get("live_lease_count") != 0
            or row.get("prepared_transaction_count") != 0
            or row.get("runtime_poisoned") is not False
        ):
            raise ValueError(
                "rank cleanup receipt is incomplete"
            )
        rows_by_rank[rank] = dict(row)
    if tuple(sorted(rows_by_rank)) != EXPECTED_RANKS:
        raise ValueError(
            "cleanup rank inventory mismatch"
        )
    return {
        **copy.deepcopy(value),
        "rank_cleanup_receipts": [
            rows_by_rank[rank]
            for rank in EXPECTED_RANKS
        ],
    }


def _validate_token_rows(
    value: object,
    *,
    batch_size: int,
    name: str,
) -> list[dict]:
    if not isinstance(value, list) or len(value) != batch_size:
        raise ValueError(f"{name} row inventory mismatch")
    normalized = []
    for prompt_index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"{name} row must be a mapping")
        token_ids = row.get("token_ids")
        if (
            not isinstance(token_ids, list)
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in token_ids
            )
        ):
            raise ValueError(f"{name} token IDs are invalid")
        if row.get("prompt_index") != prompt_index:
            raise ValueError(f"{name} prompt index mismatch")
        if row.get("token_count") != len(token_ids):
            raise ValueError(f"{name} token count mismatch")
        if row.get("sha256") != _json_sha256(token_ids):
            raise ValueError(f"{name} digest mismatch")
        normalized.append({
            "prompt_index": prompt_index,
            "token_count": len(token_ids),
            "token_ids": list(token_ids),
            "sha256": row["sha256"],
        })
    return normalized


def _validate_runtime(
    value: object,
    *,
    policy: str,
    batch_size: int,
) -> dict:
    if not isinstance(value, dict):
        raise ValueError("runtime must be a mapping")
    names = (
        "proposal_rows",
        "proposed_tokens",
        "accepted_draft_tokens",
        "rejected_draft_tokens",
        "first_target_callbacks",
        "verify_callbacks",
        "accepted_prefix_replays",
    )
    counters = {
        name: _integer(
            value.get(name),
            name.replace("_", " "),
        )
        for name in names
    }
    mappings_value = value.get("consumed_input_mappings")
    if not isinstance(mappings_value, list):
        raise ValueError(
            "consumed input mappings must be a list"
        )
    if policy == "baseline":
        if any(counters.values()) or mappings_value:
            raise ValueError(
                "baseline speculative evidence must be empty"
            )
        mappings = []
    else:
        for name in (
            "proposal_rows",
            "proposed_tokens",
            "accepted_draft_tokens",
            "rejected_draft_tokens",
            "first_target_callbacks",
            "verify_callbacks",
        ):
            if counters[name] == 0:
                raise ValueError(
                    f"{name.replace('_', ' ')} must be positive"
                )
        if counters["accepted_prefix_replays"] != 0:
            raise ValueError(
                "accepted-prefix replay count must be zero"
            )
        mappings = [
            _validate_mapping(row)
            for row in mappings_value
        ]
        if not mappings or {
            row["sequence_id"]
            for row in mappings
        } != set(range(batch_size)):
            raise ValueError(
                "consumed input mapping inventory mismatch"
            )
        mapping_keys = [
            (
                row["sequence_id"],
                row["transaction_ordinal"],
            )
            for row in mappings
        ]
        if len(set(mapping_keys)) != len(mapping_keys):
            raise ValueError(
                "consumed input mapping inventory is duplicated"
            )
        for sequence_id in range(batch_size):
            ordinals = sorted(
                row["transaction_ordinal"]
                for row in mappings
                if row["sequence_id"] == sequence_id
            )
            if ordinals != list(range(len(ordinals))):
                raise ValueError(
                    "transaction ordinal inventory mismatch"
                )
        if (
            sum(
                row["proposal_token_count"]
                for row in mappings
            )
            != counters["proposed_tokens"]
            or sum(
                row["accepted_draft_count"]
                for row in mappings
            )
            != counters["accepted_draft_tokens"]
        ):
            raise ValueError(
                "runtime mapping counters mismatch"
            )
    return {
        **counters,
        "consumed_input_mappings": mappings,
    }


def _validate_rank_evidence(
    value: object,
    *,
    policy: str,
    batch_size: int,
) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError(
            "rank evidence must be a list"
        )
    rows_by_rank = {}
    digests_by_transaction: dict[
        tuple[int, int],
        dict[int, str],
    ] = {}
    for row in value:
        if not isinstance(row, dict):
            raise ValueError(
                "rank evidence row must be a mapping"
            )
        rank = row.get("rank")
        if rank in rows_by_rank:
            raise ValueError(
                "rank evidence inventory is duplicated"
            )
        if rank not in EXPECTED_RANKS:
            raise ValueError(
                "rank evidence inventory mismatch"
            )
        if row.get("checkpoint_loaded") is not True:
            raise ValueError(
                "rank checkpoint load receipt is missing"
            )
        transactions_value = row.get("transactions")
        receipts = row.get("side_state_receipts")
        failure_rollbacks = row.get(
            "failure_path_rollbacks"
        )
        if (
            not isinstance(transactions_value, list)
            or not isinstance(receipts, list)
            or not isinstance(failure_rollbacks, list)
        ):
            raise ValueError(
                "rank transaction evidence is invalid"
            )
        if any(
            receipt.get("rank") != rank
            for receipt in receipts
            if isinstance(receipt, dict)
        ):
            raise ValueError(
                "side-state receipt rank mismatch"
            )
        if policy == "baseline":
            if transactions_value or receipts or failure_rollbacks:
                raise ValueError(
                    "baseline rank transaction evidence must be empty"
                )
            transactions = []
            normalized_receipts = []
        else:
            transactions = [
                _validate_sequence_transaction(
                    transaction,
                    rank=rank,
                )
                for transaction in transactions_value
            ]
            if not transactions or {
                transaction["sequence_id"]
                for transaction in transactions
            } != set(range(batch_size)):
                raise ValueError(
                    "rank sequence transaction inventory mismatch"
                )
            transaction_keys = [
                (
                    transaction["sequence_id"],
                    transaction["transaction_ordinal"],
                )
                for transaction in transactions
            ]
            if (
                len(set(transaction_keys))
                != len(transaction_keys)
            ):
                raise ValueError(
                    "rank sequence transaction inventory is duplicated"
                )
            for sequence_id in range(batch_size):
                ordinals = sorted(
                    transaction["transaction_ordinal"]
                    for transaction in transactions
                    if transaction["sequence_id"]
                    == sequence_id
                )
                if ordinals != list(range(len(ordinals))):
                    raise ValueError(
                        "rank transaction ordinal inventory mismatch"
                    )
            normalized_receipts = _validate_side_state(
                receipts,
                failure_rollbacks,
                rank=rank,
            )
            for transaction in transactions:
                if (
                    transaction["cell_key"]
                    != cell_key(policy, batch_size)
                ):
                    raise ValueError(
                        "transaction cell identity mismatch"
                    )
                key = (
                    transaction["sequence_id"],
                    transaction["transaction_ordinal"],
                )
                digests_by_transaction.setdefault(
                    key,
                    {},
                )[rank] = transaction["semantic_digest"]
        rows_by_rank[rank] = {
            "rank": rank,
            "checkpoint_loaded": True,
            "transactions": transactions,
            "side_state_receipts": normalized_receipts,
            "failure_path_rollbacks": list(
                failure_rollbacks
            ),
        }
    if tuple(sorted(rows_by_rank)) != EXPECTED_RANKS:
        raise ValueError(
            "rank evidence inventory mismatch"
        )
    if policy == "ngram":
        reference_keys = {
            (
                transaction["sequence_id"],
                transaction["transaction_ordinal"],
            )
            for transaction in rows_by_rank[0][
                "transactions"
            ]
        }
        if set(digests_by_transaction) != reference_keys:
            raise ValueError(
                "cross-rank transaction inventory mismatch"
            )
        for rank_digests in digests_by_transaction.values():
            if (
                set(rank_digests) != set(EXPECTED_RANKS)
                or len(set(rank_digests.values())) != 1
            ):
                raise ValueError(
                    "cross-rank transaction digest mismatch"
                )
    return [
        rows_by_rank[rank]
        for rank in EXPECTED_RANKS
    ]


def validate_cell_result(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("cell must be a mapping")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("cell schema version mismatch")
    if value.get("classification") != CLASSIFICATION:
        raise ValueError("cell classification mismatch")
    policy = value.get("policy")
    batch_size = value.get("batch_size")
    cell_key(policy, batch_size)
    if value.get("context_tokens") != CONTEXT_TOKENS:
        raise ValueError("cell context length mismatch")
    if value.get("world_size") != WORLD_SIZE:
        raise ValueError("cell world size mismatch")
    model_identity = _validate_model_identity(
        value.get("model_identity")
    )
    prompt_rows = _validate_token_rows(
        value.get("prompt_rows"),
        batch_size=batch_size,
        name="prompt",
    )
    output_rows = _validate_token_rows(
        value.get("output_rows"),
        batch_size=batch_size,
        name="output",
    )
    if any(
        row["token_count"] != CONTEXT_TOKENS
        for row in prompt_rows
    ):
        raise ValueError("prompt context length mismatch")
    if any(
        row["token_count"] != MAX_OUTPUT_TOKENS
        for row in output_rows
    ):
        raise ValueError("output token count mismatch")
    runtime = _validate_runtime(
        value.get("runtime"),
        policy=policy,
        batch_size=batch_size,
    )
    rank_evidence = _validate_rank_evidence(
        value.get("rank_evidence"),
        policy=policy,
        batch_size=batch_size,
    )
    if policy == "ngram":
        mapping_keys = {
            (
                row["sequence_id"],
                row["transaction_ordinal"],
            )
            for row in runtime["consumed_input_mappings"]
        }
        for rank_row in rank_evidence:
            transaction_keys = {
                (
                    row["sequence_id"],
                    row["transaction_ordinal"],
                )
                for row in rank_row["transactions"]
            }
            if transaction_keys != mapping_keys:
                raise ValueError(
                    "rank transaction mapping inventory mismatch"
                )
    profile = validate_rank_profile(
        value.get("profile"),
        policy=policy,
    )
    movement = _validate_kv_rank_deltas(
        value.get("kv_rank_deltas")
    )
    phases_value = value.get("residency_phases")
    if policy == "baseline":
        if phases_value != []:
            raise ValueError(
                "baseline residency phases must be empty"
            )
        phases = []
    else:
        phases = validate_residency_phases(phases_value)
    cleanup = validate_cleanup_receipt(
        value.get("cleanup_receipt")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "policy": policy,
        "context_tokens": CONTEXT_TOKENS,
        "batch_size": batch_size,
        "world_size": WORLD_SIZE,
        "model_identity": model_identity,
        "prompt_rows": prompt_rows,
        "output_rows": output_rows,
        "runtime": runtime,
        "rank_evidence": rank_evidence,
        "profile": profile,
        "kv_rank_deltas": movement,
        "residency_phases": phases,
        "cleanup_receipt": cleanup,
    }


def validate_result(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("result must be a mapping")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("result schema version mismatch")
    if value.get("classification") != CLASSIFICATION:
        raise ValueError(
            "result classification mismatch"
        )
    if value.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("result claim scope mismatch")
    if value.get("limitations") != list(LIMITATIONS):
        raise ValueError("result limitations mismatch")
    source_digest = _sha256(
        value.get("source_tree_sha256"),
        "source tree",
    )
    model_digest = _sha256(
        value.get("model_manifest_sha256"),
        "model manifest",
    )
    if model_digest != MODEL_MANIFEST_SHA256:
        raise ValueError(
            "model manifest does not match authority"
        )
    if value.get("world_size") != WORLD_SIZE:
        raise ValueError("result world size mismatch")
    gpu_indices = value.get("gpu_indices")
    if (
        not isinstance(gpu_indices, list)
        or len(gpu_indices) != WORLD_SIZE
        or len(set(gpu_indices)) != WORLD_SIZE
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in gpu_indices
        )
    ):
        raise ValueError(
            "result GPU inventory mismatch"
        )
    cells = value.get("cells")
    expected_cells = {
        cell_key(policy, batch_size)
        for batch_size in BATCH_SIZES
        for policy in POLICIES
    }
    if (
        not isinstance(cells, dict)
        or set(cells) != expected_cells
    ):
        raise ValueError(
            "result cell inventory mismatch"
        )
    normalized_cells = {
        key: validate_cell_result(cells[key])
        for key in sorted(cells)
    }
    for batch_size in BATCH_SIZES:
        baseline = normalized_cells[
            cell_key("baseline", batch_size)
        ]
        candidate = normalized_cells[
            cell_key("ngram", batch_size)
        ]
        if (
            baseline["prompt_rows"]
            != candidate["prompt_rows"]
            or baseline["output_rows"]
            != candidate["output_rows"]
        ):
            raise ValueError(
                f"output parity mismatch for batch {batch_size}"
            )
    expected_parity = {
        f"b{batch_size}": True
        for batch_size in BATCH_SIZES
    }
    if value.get("parity") != expected_parity:
        raise ValueError("result parity summary mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "claim_scope": CLAIM_SCOPE,
        "limitations": list(LIMITATIONS),
        "source_tree_sha256": source_digest,
        "model_manifest_sha256": model_digest,
        "world_size": WORLD_SIZE,
        "gpu_indices": list(gpu_indices),
        "cells": normalized_cells,
        "parity": expected_parity,
    }


def atomic_write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def hash_source_files(
    root: Path,
    source_files: tuple[str, ...],
) -> dict[str, str]:
    root = Path(root)
    return {
        name: sha256_file(root / name)
        for name in source_files
    }


def source_tree_sha256(
    root: Path,
    source_files: tuple[str, ...],
) -> str:
    root = Path(root)
    if (
        not isinstance(source_files, tuple)
        or not source_files
        or any(
            not isinstance(name, str) or not name
            for name in source_files
        )
    ):
        raise ValueError(
            "source file inventory must be a non-empty tuple"
        )
    digest = hashlib.sha256()
    for name in sorted(source_files):
        path = root / name
        payload = path.read_bytes()
        encoded_name = name.encode("utf-8")
        digest.update(
            len(encoded_name).to_bytes(8, "big")
        )
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def model_manifest_sha256(model_path: str) -> str:
    root = Path(model_path)
    if not root.is_dir():
        raise ValueError(
            "model path must be a checkpoint directory"
        )
    manifest_path = root.parent / "model_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            "approved model manifest is missing"
        )
    return sha256_file(manifest_path)


def _load_default_verifier():
    path = (
        Path(__file__).resolve().parent
        / "verify_qwen35_generic_speculative_tp4_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "verify_qwen35_generic_speculative_tp4_gate",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_run


def run_campaign(
    *,
    model_path: str,
    gpu_indices: tuple[int, ...],
    output_dir: Path,
    dist_port_base: int,
    master_port_base: int,
    repo_root: Path | None = None,
    worker_script: Path | None = None,
    python_executable: str = sys.executable,
    source_files: tuple[str, ...] = DEFAULT_SOURCE_FILES,
    verifier=None,
) -> dict:
    if (
        not isinstance(gpu_indices, tuple)
        or len(gpu_indices) != WORLD_SIZE
        or len(set(gpu_indices)) != WORLD_SIZE
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in gpu_indices
        )
    ):
        raise ValueError("campaign GPU inventory mismatch")
    dist_port_base = _integer(
        dist_port_base,
        "distributed port base",
        minimum=1,
    )
    master_port_base = _integer(
        master_port_base,
        "master port base",
        minimum=1,
    )
    repo_root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else Path(repo_root)
    )
    worker_script = (
        Path(__file__).resolve().parent
        / "qwen35_generic_speculative_tp4_worker.py"
        if worker_script is None
        else Path(worker_script)
    )
    output_dir = Path(output_dir)
    failed_dir = output_dir.with_name(
        f"{output_dir.name}.failed"
    )
    if output_dir.exists() or failed_dir.exists():
        raise ValueError(
            "campaign output directory already exists"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ))
    try:
        cell_dir = temporary_dir / "cells"
        cell_dir.mkdir()
        cells = {}
        cell_index = 0
        for batch_size in BATCH_SIZES:
            for policy in POLICIES:
                key = cell_key(policy, batch_size)
                cell_path = cell_dir / f"{key}.json"
                log_path = cell_dir / f"{key}.log"
                command = [
                    python_executable,
                    str(worker_script),
                    "--model",
                    model_path,
                    "--gpu-indices",
                    ",".join(
                        str(index)
                        for index in gpu_indices
                    ),
                    "--policy",
                    policy,
                    "--batch-size",
                    str(batch_size),
                    "--dist-port",
                    str(dist_port_base + cell_index),
                    "--master-port",
                    str(master_port_base + cell_index),
                    "--out",
                    str(cell_path),
                ]
                with log_path.open(
                    "w",
                    encoding="utf-8",
                ) as log:
                    completed = subprocess.run(
                        command,
                        cwd=repo_root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"TP4 worker failed: {key}; "
                        f"log={log_path}"
                    )
                try:
                    payload = json.loads(
                        cell_path.read_text(
                            encoding="utf-8"
                        )
                    )
                except (
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                ) as error:
                    raise RuntimeError(
                        f"TP4 worker output is invalid: {key}"
                    ) from error
                cells[key] = validate_cell_result(payload)
                cell_index += 1

        source_hashes = hash_source_files(
            repo_root,
            source_files,
        )
        source_digest = source_tree_sha256(
            repo_root,
            source_files,
        )
        model_digest = model_manifest_sha256(model_path)
        if model_digest != MODEL_MANIFEST_SHA256:
            raise RuntimeError(
                "model manifest does not match approved checkpoint"
            )
        result = validate_result({
            "schema_version": SCHEMA_VERSION,
            "classification": CLASSIFICATION,
            "claim_scope": CLAIM_SCOPE,
            "limitations": list(LIMITATIONS),
            "source_tree_sha256": source_digest,
            "model_manifest_sha256": model_digest,
            "world_size": WORLD_SIZE,
            "gpu_indices": list(gpu_indices),
            "cells": cells,
            "parity": {
                f"b{batch_size}": True
                for batch_size in BATCH_SIZES
            },
        })
        atomic_write_json(
            temporary_dir / "result.json",
            result,
        )
        atomic_write_json(
            temporary_dir / "source_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "source_tree_sha256": source_digest,
                "model_manifest_sha256": model_digest,
                "source_files": source_hashes,
                "artifacts": {
                    "result.json": sha256_file(
                        temporary_dir / "result.json"
                    ),
                },
            },
        )
        verify = (
            _load_default_verifier()
            if verifier is None
            else verifier
        )
        verification = verify(
            temporary_dir,
            repo_root,
        )
        atomic_write_json(
            temporary_dir / "verify.json",
            verification,
        )
        if (
            verification.get("classification") != "PASS"
            or verification.get("failures") != []
        ):
            raise RuntimeError(
                "independent verification failed: "
                + json.dumps(
                    verification.get("failures", []),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        os.replace(temporary_dir, output_dir)
        return result
    except Exception as error:
        if temporary_dir.exists():
            os.replace(temporary_dir, failed_dir)
        raise RuntimeError(
            f"{error}; failed_artifacts={failed_dir}"
        ) from error
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def _gpu_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(
            int(item)
            for item in value.split(",")
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "GPU indices must be comma-separated integers"
        ) from error
    if (
        len(indices) != WORLD_SIZE
        or len(set(indices)) != WORLD_SIZE
        or any(index < 0 for index in indices)
    ):
        raise argparse.ArgumentTypeError(
            "exactly four distinct GPU indices are required"
        )
    return indices


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--gpu-indices",
        required=True,
        type=_gpu_indices,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dist-port-base",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--master-port-base",
        required=True,
        type=int,
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_campaign(
        model_path=args.model,
        gpu_indices=args.gpu_indices,
        output_dir=Path(args.output_dir),
        dist_port_base=args.dist_port_base,
        master_port_base=args.master_port_base,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
