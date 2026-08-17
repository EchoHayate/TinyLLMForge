from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


SCHEMA_VERSION = 1
CLASSIFICATION = "NOT_PROMOTABLE"
WORLD_SIZE = 4
POLICIES = ("baseline", "ngram")
CONTEXT_TOKENS = 4096
BATCH_SIZES = (1, 4)
MAX_OUTPUT_TOKENS = 8
NGRAM_SIZE = 3
MAX_PROPOSAL_TOKENS = 4
EXPECTED_RANKS = (0, 1, 2, 3)
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
RUNTIME_KEYS = (
    "proposal_rows",
    "proposed_tokens",
    "accepted_draft_tokens",
    "first_target_callbacks",
    "tail_callbacks",
)
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
CLAIM_SCOPE = (
    "Qwen3-0.6B generic host n-gram TP4 correctness "
    "and collective authority"
)
LIMITATIONS = (
    "no TP4 performance claim",
    "no 16K/32K TP4 performance direction",
    "no second-model evidence",
    "no model-runner proposal TP4 evidence",
    "no learned-drafter or MTP plus KV-offload evidence",
    "no KV8/KV4 evidence",
    "no Phase 1 completion claim",
)
DEFAULT_SOURCE_FILES = (
    "tinyvllm/engine/decode_internal_profiler.py",
    "tinyvllm/engine/llm_engine.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/speculative_execution.py",
    "tinyvllm/engine/speculative_model_runner.py",
    "tinyvllm/engine/speculative_residency.py",
    "tinyvllm/engine/speculative_runtime.py",
    "tools/generic_speculative_tp4_gate.py",
    "tools/generic_speculative_tp4_worker.py",
    "tools/verify_generic_speculative_tp4_gate.py",
    "tools/test_generic_speculative_tp4_gate.py",
)


def cell_key(policy: str, batch_size: int) -> str:
    if policy not in POLICIES:
        raise ValueError("unsupported policy")
    if batch_size not in BATCH_SIZES:
        raise ValueError("unsupported batch size")
    return f"{policy}:b{batch_size}"


def _non_negative_integer(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a non-negative integer"
        )
    return value


def _validate_sha256(value: object, name: str) -> str:
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
        raise ValueError("profile callback row must be a mapping")
    if step.get("rank") != rank:
        raise ValueError("profile callback rank mismatch")
    _non_negative_integer(
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
        _non_negative_integer(
            decode_ordinal,
            "profile callback decode ordinal",
        )
    elif decode_ordinal is not None:
        raise ValueError(
            "profile prefill decode ordinal must be null"
        )
    if _non_negative_integer(
        step.get("active_sequence_count"),
        "profile callback active sequence count",
    ) <= 0:
        raise ValueError(
            "profile callback active sequence count must be positive"
        )
    _validate_sha256(
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
        _non_negative_integer(
            step.get(name),
            f"profile callback {name}",
        )
    return dict(step)


def _validate_profile_collective(row: object, rank: int) -> dict:
    if not isinstance(row, dict):
        raise ValueError("profile collective row must be a mapping")
    if row.get("rank") != rank:
        raise ValueError("profile collective rank mismatch")
    _non_negative_integer(
        row.get("step_index"),
        "profile collective step index",
    )
    _non_negative_integer(
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
        _non_negative_integer(
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
        raise ValueError("rank profile rank inventory mismatch")
    rank_rows = value.get("ranks")
    if not isinstance(rank_rows, list):
        raise ValueError("rank profile rank inventory is invalid")
    rows_by_rank = {}
    for row in rank_rows:
        if not isinstance(row, dict):
            raise ValueError("rank profile row must be a mapping")
        rank = row.get("rank")
        if rank in rows_by_rank:
            raise ValueError("rank profile rank inventory is duplicated")
        if rank not in EXPECTED_RANKS:
            raise ValueError("rank profile rank inventory mismatch")
        if (
            row.get("enabled") is not True
            or row.get("finalization_status") != "complete"
        ):
            raise ValueError("rank profile row is incomplete")
        steps = row.get("steps")
        collectives = row.get("collectives")
        if not isinstance(steps, list):
            raise ValueError("rank profile callbacks are invalid")
        if not isinstance(collectives, list):
            raise ValueError("rank profile collectives are invalid")
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
        raise ValueError("rank profile rank inventory mismatch")

    callback_identities = {}
    collective_identities = {}
    for rank in EXPECTED_RANKS:
        row = rows_by_rank[rank]
        speculative_steps = [
            step
            for step in row["steps"]
            if step["batch_kind"] in SPECULATIVE_BATCH_KINDS
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
        speculative_step_indices = {
            step["step_index"]
            for step in speculative_steps
        }
        speculative_collectives = [
            collective
            for collective in row["collectives"]
            if collective["step_index"]
            in speculative_step_indices
        ]
        collective_step_indices = {
            collective["step_index"]
            for collective in speculative_collectives
        }
        if (
            not speculative_collectives
            or collective_step_indices
            != speculative_step_indices
        ):
            raise ValueError(
                "candidate collective coverage is incomplete"
            )
        collective_identities[rank] = tuple(
            _collective_identity(collective)
            for collective in speculative_collectives
        )

    callback_reference = callback_identities[0]
    if any(
        callback_identities[rank] != callback_reference
        for rank in EXPECTED_RANKS[1:]
    ):
        raise ValueError("candidate callback identity mismatch")
    collective_reference = collective_identities[0]
    if any(
        len(collective_identities[rank])
        != len(collective_reference)
        for rank in EXPECTED_RANKS[1:]
    ):
        raise ValueError("candidate collective count mismatch")
    if any(
        collective_identities[rank]
        != collective_reference
        for rank in EXPECTED_RANKS[1:]
    ):
        raise ValueError("candidate collective identity mismatch")

    return {
        "enabled": True,
        "rank_inventory": list(EXPECTED_RANKS),
        "ranks": [
            copy.deepcopy(rows_by_rank[rank])
            for rank in EXPECTED_RANKS
        ],
    }


def _normalize_integer_rows(
    value: object,
    name: str,
) -> list:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    normalized = []
    for row in value:
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"{name} row must be a sequence")
        normalized.append([
            _non_negative_integer(
                item,
                f"{name} value",
            )
            for item in row
        ])
    return normalized


def _normalize_sequence_ids(
    value: object,
) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            "residency sequence IDs must be non-empty"
        )
    normalized = [
        _non_negative_integer(
            sequence_id,
            "residency sequence ID",
        )
        for sequence_id in value
    ]
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "residency sequence IDs must be unique"
        )
    return normalized


def validate_residency_phases(value: object) -> list[dict]:
    if (
        not isinstance(value, list)
        or not value
        or len(value) % len(SUCCESSFUL_RESIDENCY_OPERATIONS)
    ):
        raise ValueError(
            "residency phase order must contain complete tickets"
        )
    normalized = []
    active_ticket = None
    for phase_index, phase in enumerate(value):
        if not isinstance(phase, dict):
            raise ValueError("residency phase must be a mapping")
        expected_operation = SUCCESSFUL_RESIDENCY_OPERATIONS[
            phase_index % len(SUCCESSFUL_RESIDENCY_OPERATIONS)
        ]
        if phase.get("operation") != expected_operation:
            raise ValueError("residency phase order mismatch")
        expected_status = RESIDENCY_STATUS[expected_operation]
        if phase.get("status") != expected_status:
            raise ValueError("residency phase status mismatch")
        ticket_id = _non_negative_integer(
            phase.get("ticket_id"),
            "residency ticket ID",
        )
        if expected_operation == "prepare":
            active_ticket = ticket_id
        elif ticket_id != active_ticket:
            raise ValueError("residency ticket sequence mismatch")
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
                or row.get("operation") != expected_operation
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


def validate_cleanup_receipt(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("cleanup receipt must be a mapping")
    if value.get("process_group_destroyed") is not True:
        raise ValueError(
            "cleanup process group was not destroyed"
        )
    if value.get("rank_exit_codes") != [0, 0, 0, 0]:
        raise ValueError("cleanup rank exit codes are invalid")
    if value.get("owned_children_remaining") != []:
        raise ValueError(
            "cleanup owned children remain"
        )
    rows = value.get("rank_cleanup_receipts")
    if not isinstance(rows, list):
        raise ValueError(
            "cleanup rank inventory is invalid"
        )
    rows_by_rank = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(
                "cleanup rank receipt must be a mapping"
            )
        rank = row.get("rank")
        if rank in rows_by_rank:
            raise ValueError(
                "cleanup rank inventory is duplicated"
            )
        if (
            rank not in EXPECTED_RANKS
            or row.get("process_group_destroyed") is not True
        ):
            raise ValueError(
                "cleanup rank inventory mismatch"
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


def _validate_prompt_rows(
    value: object,
    *,
    batch_size: int,
) -> list[dict]:
    if not isinstance(value, list) or len(value) != batch_size:
        raise ValueError("prompt row count mismatch")
    normalized = []
    for prompt_index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError("prompt row must be a mapping")
        if row.get("prompt_index") != prompt_index:
            raise ValueError("prompt index mismatch")
        if row.get("token_count") != CONTEXT_TOKENS:
            raise ValueError("prompt token count mismatch")
        _validate_sha256(
            row.get("sha256"),
            "prompt token digest",
        )
        normalized.append(copy.deepcopy(row))
    return normalized


def _validate_outputs(
    value: object,
    *,
    batch_size: int,
) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != batch_size:
        raise ValueError("output row count mismatch")
    normalized = []
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != MAX_OUTPUT_TOKENS
        ):
            raise ValueError("output token shape mismatch")
        normalized.append([
            _non_negative_integer(
                token_id,
                "output token ID",
            )
            for token_id in row
        ])
    return normalized


def _validate_runtime(
    value: object,
    *,
    policy: str,
) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError("runtime evidence must be a mapping")
    normalized = {
        name: _non_negative_integer(
            value.get(name),
            f"runtime {name}",
        )
        for name in RUNTIME_KEYS
    }
    if policy == "baseline":
        if any(normalized.values()):
            raise ValueError(
                "baseline runtime evidence must be zero"
            )
    elif any(normalized[name] <= 0 for name in RUNTIME_KEYS):
        raise ValueError(
            "candidate runtime evidence is incomplete"
        )
    return normalized


def _validate_kv_rank_deltas(value: object) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("KV rank deltas must be a list")
    rows_by_rank = {}
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("KV rank delta must be a mapping")
        rank = row.get("rank")
        if rank in rows_by_rank:
            raise ValueError("KV rank inventory is duplicated")
        if rank not in EXPECTED_RANKS:
            raise ValueError("KV rank inventory mismatch")
        normalized = {
            "rank": rank,
            **{
                name: _non_negative_integer(
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


def validate_cell_result(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("cell result must be a mapping")
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
    if value.get("rank_inventory") != list(EXPECTED_RANKS):
        raise ValueError("cell rank inventory mismatch")
    if value.get("ack_ranks") != [1, 2, 3]:
        raise ValueError(
            "cell acknowledgement rank inventory mismatch"
        )
    prompt_rows = _validate_prompt_rows(
        value.get("prompt_rows"),
        batch_size=batch_size,
    )
    outputs = _validate_outputs(
        value.get("outputs"),
        batch_size=batch_size,
    )
    runtime = _validate_runtime(
        value.get("runtime"),
        policy=policy,
    )
    kv_rank_deltas = _validate_kv_rank_deltas(
        value.get("kv_rank_deltas")
    )
    residency_value = value.get("residency_phases")
    if policy == "baseline":
        if residency_value != []:
            raise ValueError(
                "baseline residency phases must be empty"
            )
        residency_phases = []
    else:
        residency_phases = validate_residency_phases(
            residency_value
        )
    profile = validate_rank_profile(
        value.get("profile"),
        policy=policy,
    )
    tokenizer_identifier = value.get(
        "tokenizer_identifier"
    )
    dtype = value.get("dtype")
    if (
        not isinstance(tokenizer_identifier, str)
        or not tokenizer_identifier
        or not isinstance(dtype, str)
        or not dtype
    ):
        raise ValueError(
            "cell tokenizer or dtype is invalid"
        )
    cleanup_receipt = validate_cleanup_receipt(
        value.get("cleanup_receipt")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "policy": policy,
        "context_tokens": CONTEXT_TOKENS,
        "batch_size": batch_size,
        "world_size": WORLD_SIZE,
        "rank_inventory": list(EXPECTED_RANKS),
        "ack_ranks": [1, 2, 3],
        "prompt_rows": prompt_rows,
        "outputs": outputs,
        "runtime": runtime,
        "kv_rank_deltas": kv_rank_deltas,
        "residency_phases": residency_phases,
        "profile": profile,
        "tokenizer_identifier": tokenizer_identifier,
        "dtype": dtype,
        "cleanup_receipt": cleanup_receipt,
    }


def validate_result(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("result must be a mapping")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("result schema version mismatch")
    if value.get("classification") != CLASSIFICATION:
        raise ValueError("result classification mismatch")
    claim_scope = value.get("claim_scope")
    if not isinstance(claim_scope, str) or not claim_scope:
        raise ValueError("result claim scope is invalid")
    limitations = value.get("limitations")
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(
            not isinstance(item, str) or not item
            for item in limitations
        )
    ):
        raise ValueError("result limitations are invalid")
    source_digest = _validate_sha256(
        value.get("source_tree_sha256"),
        "source tree",
    )
    model_digest = _validate_sha256(
        value.get("model_manifest_sha256"),
        "model manifest",
    )
    if value.get("world_size") != WORLD_SIZE:
        raise ValueError("result world size mismatch")
    gpu_indices = value.get("gpu_indices")
    if (
        not isinstance(gpu_indices, list)
        or len(gpu_indices) != WORLD_SIZE
        or len(set(gpu_indices)) != WORLD_SIZE
    ):
        raise ValueError("result GPU inventory mismatch")
    normalized_gpu_indices = [
        _non_negative_integer(index, "GPU index")
        for index in gpu_indices
    ]
    cells = value.get("cells")
    expected_keys = {
        cell_key(policy, batch_size)
        for batch_size in BATCH_SIZES
        for policy in POLICIES
    }
    if (
        not isinstance(cells, dict)
        or set(cells) != expected_keys
    ):
        raise ValueError("result cell inventory mismatch")
    normalized_cells = {
        key: validate_cell_result(cells[key])
        for key in sorted(cells)
    }
    parity = {}
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
            or baseline["outputs"] != candidate["outputs"]
        ):
            raise ValueError(
                f"output parity mismatch for batch {batch_size}"
            )
        parity[f"b{batch_size}"] = True
    if value.get("parity") != parity:
        raise ValueError("result parity summary mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "claim_scope": claim_scope,
        "limitations": list(limitations),
        "source_tree_sha256": source_digest,
        "model_manifest_sha256": model_digest,
        "world_size": WORLD_SIZE,
        "gpu_indices": normalized_gpu_indices,
        "cells": normalized_cells,
        "parity": parity,
    }


def source_tree_sha256(
    root: Path,
    files: tuple[str, ...],
) -> str:
    root = Path(root)
    if (
        not isinstance(files, tuple)
        or not files
        or any(
            not isinstance(name, str) or not name
            for name in files
        )
    ):
        raise ValueError(
            "source file inventory must be a non-empty tuple"
        )
    digest = hashlib.sha256()
    for name in sorted(files):
        path = root / name
        payload = path.read_bytes()
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
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
    files: tuple[str, ...],
) -> dict[str, str]:
    root = Path(root)
    return {
        name: sha256_file(root / name)
        for name in files
    }


def model_manifest_sha256(model_path: str) -> str:
    root = Path(model_path)
    if not root.is_dir():
        raise ValueError(
            "model path must be a local checkpoint directory"
        )
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.suffix
            in {
                ".json",
                ".model",
                ".safetensors",
                ".bin",
                ".txt",
            }
            or path.name
            in {
                "tokenizer_config.json",
                "special_tokens_map.json",
            }
        )
    )
    if not files:
        raise ValueError(
            "model checkpoint manifest is empty"
        )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        relative_bytes = relative.encode("utf-8")
        file_digest = bytes.fromhex(sha256_file(path))
        digest.update(
            len(relative_bytes).to_bytes(8, "big")
        )
        digest.update(relative_bytes)
        digest.update(file_digest)
    return digest.hexdigest()


def _load_default_verifier():
    module_name = "verify_generic_speculative_tp4_gate"
    module = sys.modules.get(module_name)
    if module is None:
        path = (
            Path(__file__).resolve().parent
            / "verify_generic_speculative_tp4_gate.py"
        )
        spec = importlib.util.spec_from_file_location(
            module_name,
            path,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
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
    dist_port_base = _non_negative_integer(
        dist_port_base,
        "distributed port base",
    )
    master_port_base = _non_negative_integer(
        master_port_base,
        "master port base",
    )
    if dist_port_base == 0 or master_port_base == 0:
        raise ValueError("campaign port bases must be positive")
    repo_root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else Path(repo_root)
    )
    worker_script = (
        Path(__file__).resolve().parent
        / "generic_speculative_tp4_worker.py"
        if worker_script is None
        else Path(worker_script)
    )
    output_dir = Path(output_dir)
    failed_dir = output_dir.with_name(
        f"{output_dir.name}.failed"
    )
    if output_dir.exists():
        raise ValueError(
            "campaign output directory already exists"
        )
    if failed_dir.exists():
        raise ValueError(
            "campaign failed-artifact directory already exists"
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
        model_digest = model_manifest_sha256(
            model_path
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
        if verification.get("classification") != "PASS":
            raise RuntimeError(
                "TP4 campaign independent verification failed: "
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
        parsed = tuple(
            int(item)
            for item in value.split(",")
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "GPU indices must be comma-separated integers"
        ) from error
    if (
        len(parsed) != WORLD_SIZE
        or len(set(parsed)) != WORLD_SIZE
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
        "--dist-port-base",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--master-port-base",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
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
