from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


SCHEMA_VERSION = (
    "qwen35.native-mtp-tp1-4k-engine-"
    "transactional-correctness.v1"
)
CLASSIFICATION = (
    "QWEN35_NATIVE_MTP_TP1_4K_ENGINE_ESTABLISHED"
)
PROMOTION_CLASSIFICATION = "NOT_PROMOTABLE"
POLICIES = ("baseline", "native_mtp")
BATCH_SIZES = (1, 4)
PROMPT_TOKENS = 4096
MAX_OUTPUT_TOKENS = 32
MAX_PROPOSAL_TOKENS = 4
WORLD_SIZE = 1
TARGET_MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38fe"
    "69982efedaf4a3e631ad0b14aad7dd0"
)
MTP_CHECKPOINT_MANIFEST_SHA256 = (
    "9a975bdcf0383774183cae560594dd60"
    "b522b83fe9c4cd595c47c12e2403702b"
)
REQUIRED_LIMITATIONS = (
    "TP1 only",
    "4K prompt only",
    "KV offload disabled",
    "eager native MTP only",
    "not production ready",
)
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_FILES = tuple(sorted(
    [
        str(path.relative_to(_SOURCE_ROOT))
        for path in (_SOURCE_ROOT / "tinyvllm").rglob("*.py")
    ]
    + [
        "tools/qwen35_native_mtp_tp1_4k_engine_gate.py",
        "tools/qwen35_native_mtp_tp1_4k_engine_worker.py",
        "tools/verify_qwen35_native_mtp_tp1_4k_engine_gate.py",
    ]
))


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


def _validate_token_rows(
    value: object,
    *,
    batch_size: int,
    token_count: int,
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
            or len(token_ids) != token_count
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
        if row.get("token_count") != token_count:
            raise ValueError(f"{name} token count mismatch")
        if row.get("sha256") != _json_sha256(token_ids):
            raise ValueError(f"{name} digest mismatch")
        normalized.append({
            "prompt_index": prompt_index,
            "token_count": token_count,
            "token_ids": list(token_ids),
            "sha256": row["sha256"],
        })
    return normalized


def _validate_model_identity(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("model identity must be a mapping")
    if value.get("model_type") != "qwen3_5":
        raise ValueError("model type must be qwen3_5")
    architectures = value.get("architectures")
    if (
        not isinstance(architectures, list)
        or not architectures
        or any(
            not isinstance(name, str) or not name
            for name in architectures
        )
    ):
        raise ValueError("model architecture inventory is invalid")
    target_digest = _sha256(
        value.get("target_model_manifest_sha256"),
        "target model manifest",
    )
    if target_digest != TARGET_MODEL_MANIFEST_SHA256:
        raise ValueError(
            "target model manifest does not match authority"
        )
    mtp_digest = _sha256(
        value.get("mtp_checkpoint_manifest_sha256"),
        "MTP checkpoint manifest",
    )
    if mtp_digest != MTP_CHECKPOINT_MANIFEST_SHA256:
        raise ValueError(
            "MTP checkpoint manifest does not match authority"
        )
    return {
        "model_type": "qwen3_5",
        "architectures": list(architectures),
        "target_model_manifest_sha256": target_digest,
        "mtp_checkpoint_manifest_sha256": mtp_digest,
    }


def _validate_native_binding(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("native binding must be a mapping")
    if value.get("executor_id") != "native_checkpoint_proposal":
        raise ValueError("native executor ID mismatch")
    if value.get("source_type") != "native_model_runner":
        raise ValueError("native source type mismatch")
    if value.get("module_type") != "Qwen35NativeMTP":
        raise ValueError("native MTP module type mismatch")
    if (
        value.get("physical_store_type")
        != "Qwen35MTPPhysicalSlotStore"
    ):
        raise ValueError("native physical store type mismatch")
    tensor_count = _integer(
        value.get("checkpoint_tensor_count"),
        "native checkpoint tensor count",
        minimum=1,
    )
    if tensor_count != 15:
        raise ValueError(
            "native checkpoint tensor count must be 15"
        )
    return {
        "executor_id": "native_checkpoint_proposal",
        "source_type": "native_model_runner",
        "module_type": "Qwen35NativeMTP",
        "physical_store_type": (
            "Qwen35MTPPhysicalSlotStore"
        ),
        "checkpoint_tensor_count": 15,
    }


def _validate_operation_receipts(
    value: object,
    *,
    name: str,
    required_operations: tuple[str, ...],
) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} receipts are missing")
    normalized = []
    operations_by_transaction = {}
    for row in value:
        if not isinstance(row, dict):
            raise ValueError(f"{name} receipt must be a mapping")
        sequence_id = _integer(
            row.get("sequence_id"),
            f"{name} sequence ID",
        )
        transaction_id = row.get("transaction_id")
        operation = row.get("operation")
        if (
            not isinstance(transaction_id, str)
            or not transaction_id
        ):
            raise ValueError(f"{name} transaction ID is invalid")
        if operation not in required_operations:
            raise ValueError(f"{name} operation is invalid")
        key = (sequence_id, transaction_id)
        operations_by_transaction.setdefault(key, []).append(
            operation
        )
        normalized.append({
            "sequence_id": sequence_id,
            "transaction_id": transaction_id,
            "operation": operation,
        })
    if any(
        tuple(operations) != required_operations
        for operations in operations_by_transaction.values()
    ):
        raise ValueError(f"{name} lifecycle is incomplete")
    return normalized


def _validate_proposal_kv_receipts(
    value: object,
) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise ValueError("proposal KV receipts are missing")
    normalized = []
    for row in value:
        if not isinstance(row, dict):
            raise ValueError(
                "proposal KV receipt must be a mapping"
            )
        accepted = _integer(
            row.get("accepted_token_count"),
            "proposal KV accepted token count",
        )
        rejected = _integer(
            row.get("rejected_token_count"),
            "proposal KV rejected token count",
        )
        if accepted + rejected <= 0:
            raise ValueError(
                "proposal KV receipt must cover proposal tokens"
            )
        if (
            row.get("accepted_slot_identity_preserved")
            is not True
        ):
            raise ValueError(
                "accepted slot identity was not preserved"
            )
        if row.get("rejected_slots_released") is not True:
            raise ValueError("rejected slots were not released")
        transaction_id = row.get("transaction_id")
        if (
            not isinstance(transaction_id, str)
            or not transaction_id
        ):
            raise ValueError(
                "proposal KV transaction ID is invalid"
            )
        normalized.append({
            "sequence_id": _integer(
                row.get("sequence_id"),
                "proposal KV sequence ID",
            ),
            "transaction_id": transaction_id,
            "accepted_token_count": accepted,
            "rejected_token_count": rejected,
            "accepted_slot_identity_preserved": True,
            "rejected_slots_released": True,
        })
    return normalized


def _validate_lifecycle_events(value: object) -> list[dict]:
    publication = (
        "proposal_finalize_prepare",
        "side_state_apply",
        "target_kv_commit",
        "scheduler_commit",
        "proposal_finalize_commit",
        "side_state_seal",
    )
    release = (
        "proposal_sequence_release",
    )
    if not isinstance(value, list) or not value:
        raise ValueError("proposal lifecycle events are missing")
    normalized = []
    operations_by_sequence = {}
    for row in value:
        if not isinstance(row, dict):
            raise ValueError(
                "proposal lifecycle event must be a mapping"
            )
        sequence_id = _integer(
            row.get("sequence_id"),
            "proposal lifecycle sequence ID",
        )
        operation = row.get("operation")
        if operation not in publication + release:
            raise ValueError(
                "proposal lifecycle operation is invalid"
            )
        operations_by_sequence.setdefault(
            sequence_id,
            [],
        ).append(operation)
        normalized.append({
            "sequence_id": sequence_id,
            "operation": operation,
        })
    for operations in operations_by_sequence.values():
        if (
            not operations
            or operations[-1] != release[0]
            or len(operations[:-1]) % len(publication) != 0
            or any(
                tuple(
                    operations[
                        start:start + len(publication)
                    ]
                )
                != publication
                for start in range(
                    0,
                    len(operations) - 1,
                    len(publication),
                )
            )
        ):
            raise ValueError(
                "proposal lifecycle ordering is incomplete"
            )
    return normalized


def _validate_runtime(
    value: object,
    *,
    policy: str,
) -> dict:
    if not isinstance(value, dict):
        raise ValueError("runtime must be a mapping")
    counter_names = (
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
    counters = {
        name: _integer(
            value.get(name),
            name.replace("_", " "),
        )
        for name in counter_names
    }
    receipt_names = (
        "proposal_finalize_receipts",
        "side_state_receipts",
        "proposal_kv_receipts",
        "lifecycle_events",
    )
    if policy == "baseline":
        if any(counters.values()):
            raise ValueError(
                "baseline speculative counters must be zero"
            )
        if value.get("native_binding") is not None or any(
            value.get(name) != []
            for name in receipt_names
        ):
            raise ValueError(
                "baseline speculative evidence must be empty"
            )
        return {
            "native_binding": None,
            **counters,
            **{name: [] for name in receipt_names},
        }
    binding = _validate_native_binding(
        value.get("native_binding")
    )
    for name in (
        "proposal_rows",
        "proposed_tokens",
        "accepted_draft_tokens",
        "rejected_draft_tokens",
        "first_target_callbacks",
        "verify_callbacks",
    ):
        if counters[name] <= 0:
            label = name.replace("_", " ")
            if name == "first_target_callbacks":
                label = "first-target callbacks"
            raise ValueError(f"{label} must be positive")
    if counters["accepted_prefix_target_replays"] != 0:
        raise ValueError(
            "accepted-prefix target replay count must be zero"
        )
    if (
        counters["first_target_target_forwards"]
        != counters["first_target_callbacks"]
    ):
        raise ValueError(
            "first-target target forward count must match "
            "first-target callback count"
        )
    if (
        counters["verify_target_forwards"]
        != counters["verify_callbacks"]
    ):
        raise ValueError(
            "verify target forward count must match "
            "verify callback count"
        )
    finalize = _validate_operation_receipts(
        value.get("proposal_finalize_receipts"),
        name="proposal finalize",
        required_operations=("prepare", "commit"),
    )
    side_state = _validate_operation_receipts(
        value.get("side_state_receipts"),
        name="side-state",
        required_operations=(
            "prepare",
            "select",
            "apply",
            "seal",
        ),
    )
    proposal_kv = _validate_proposal_kv_receipts(
        value.get("proposal_kv_receipts")
    )
    lifecycle = _validate_lifecycle_events(
        value.get("lifecycle_events")
    )
    if sum(
        row["accepted_token_count"]
        for row in proposal_kv
    ) > counters["accepted_draft_tokens"]:
        raise ValueError(
            "proposal KV accepted count exceeds runtime total"
        )
    if sum(
        row["rejected_token_count"]
        for row in proposal_kv
    ) > counters["rejected_draft_tokens"]:
        raise ValueError(
            "proposal KV rejected count exceeds runtime total"
        )
    return {
        "native_binding": binding,
        **counters,
        "proposal_finalize_receipts": finalize,
        "side_state_receipts": side_state,
        "proposal_kv_receipts": proposal_kv,
        "lifecycle_events": lifecycle,
    }


def _validate_native_state_snapshot(
    value: object,
    *,
    policy: str,
):
    if policy == "baseline":
        if value is not None:
            raise ValueError(
                "baseline native state snapshot must be empty"
            )
        return None
    if not isinstance(value, dict):
        raise ValueError(
            "native state snapshot must be a mapping"
        )
    names = (
        "pending_prefix_count",
        "bootstrapped_sequence_count",
        "proposal_transaction_count",
        "batch_ticket_count",
        "batch_ticket_transaction_count",
        "allocated_physical_slot_count",
    )
    normalized = {
        name: _integer(
            value.get(name),
            name.replace("_", " "),
        )
        for name in names
    }
    if any(normalized.values()):
        raise ValueError("native state snapshot is not empty")
    return normalized


def _validate_cleanup(
    value: object,
    *,
    policy: str,
) -> dict:
    if not isinstance(value, dict):
        raise ValueError("cleanup receipt must be a mapping")
    if value.get("proposal_transactions_open") != []:
        raise ValueError("proposal transaction leak detected")
    if value.get("proposal_finalize_tickets_open") != []:
        raise ValueError("proposal finalize ticket leak detected")
    if value.get("proposal_sequence_ids") != []:
        raise ValueError("proposal sequence leak detected")
    if value.get("proposal_kv_slots_in_use") != 0:
        raise ValueError("proposal KV slot leak detected")
    native_state_snapshot = _validate_native_state_snapshot(
        value.get("native_state_snapshot"),
        policy=policy,
    )
    leases_before = _integer(
        value.get("hybrid_state_leases_before"),
        "hybrid-state leases before",
    )
    leases_after = _integer(
        value.get("hybrid_state_leases_after"),
        "hybrid-state leases after",
    )
    if leases_after != leases_before:
        raise ValueError("hybrid-state lease leak detected")
    if (
        value.get("owned_children_remaining") != []
        or value.get("engine_exit_called") is not True
        or value.get("worker_exit_code") != 0
    ):
        raise ValueError("cleanup receipt is incomplete")
    return {
        "proposal_transactions_open": [],
        "proposal_finalize_tickets_open": [],
        "proposal_sequence_ids": [],
        "proposal_kv_slots_in_use": 0,
        "native_state_snapshot": native_state_snapshot,
        "hybrid_state_leases_before": leases_before,
        "hybrid_state_leases_after": leases_after,
        "owned_children_remaining": [],
        "engine_exit_called": True,
        "worker_exit_code": 0,
    }


def validate_cell_result(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("cell must be a mapping")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("cell schema version mismatch")
    policy = value.get("policy")
    batch_size = value.get("batch_size")
    cell_key(policy, batch_size)
    if value.get("world_size") != WORLD_SIZE:
        raise ValueError("cell world size mismatch")
    if value.get("prompt_token_count") != PROMPT_TOKENS:
        raise ValueError("cell prompt token count mismatch")
    if value.get("max_output_tokens") != MAX_OUTPUT_TOKENS:
        raise ValueError("cell output token count mismatch")
    if (
        value.get("max_proposal_tokens")
        != MAX_PROPOSAL_TOKENS
    ):
        raise ValueError("cell proposal token count mismatch")
    gpu_index = _integer(value.get("gpu_index"), "GPU index")
    model_identity = _validate_model_identity(
        value.get("model_identity")
    )
    prompt_rows = _validate_token_rows(
        value.get("prompt_rows"),
        batch_size=batch_size,
        token_count=PROMPT_TOKENS,
        name="prompt",
    )
    output_rows = _validate_token_rows(
        value.get("output_rows"),
        batch_size=batch_size,
        token_count=MAX_OUTPUT_TOKENS,
        name="output",
    )
    runtime = _validate_runtime(
        value.get("runtime"),
        policy=policy,
    )
    cleanup = _validate_cleanup(
        value.get("cleanup"),
        policy=policy,
    )
    if value.get("runtime_poisoned") is not False:
        raise ValueError("runtime is poisoned")
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": policy,
        "batch_size": batch_size,
        "world_size": WORLD_SIZE,
        "gpu_index": gpu_index,
        "prompt_token_count": PROMPT_TOKENS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_proposal_tokens": MAX_PROPOSAL_TOKENS,
        "model_identity": model_identity,
        "prompt_rows": prompt_rows,
        "output_rows": output_rows,
        "runtime": runtime,
        "cleanup": cleanup,
        "runtime_poisoned": False,
    }


def validate_result(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("result must be a mapping")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("result schema version mismatch")
    if value.get("classification") != CLASSIFICATION:
        raise ValueError("result classification mismatch")
    if (
        value.get("promotion_classification")
        != PROMOTION_CLASSIFICATION
    ):
        raise ValueError("result promotion classification mismatch")
    target_digest = _sha256(
        value.get("target_model_manifest_sha256"),
        "target model manifest",
    )
    if target_digest != TARGET_MODEL_MANIFEST_SHA256:
        raise ValueError(
            "target model manifest does not match authority"
        )
    mtp_digest = _sha256(
        value.get("mtp_checkpoint_manifest_sha256"),
        "MTP checkpoint manifest",
    )
    if mtp_digest != MTP_CHECKPOINT_MANIFEST_SHA256:
        raise ValueError(
            "MTP checkpoint manifest does not match authority"
        )
    source_digest = _sha256(
        value.get("source_tree_sha256"),
        "source tree",
    )
    if value.get("world_size") != WORLD_SIZE:
        raise ValueError("result world size mismatch")
    gpu_index = _integer(value.get("gpu_index"), "GPU index")
    cells = value.get("cells")
    expected = {
        cell_key(policy, batch_size)
        for batch_size in BATCH_SIZES
        for policy in POLICIES
    }
    if not isinstance(cells, dict) or set(cells) != expected:
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
        native = normalized_cells[
            cell_key("native_mtp", batch_size)
        ]
        if (
            baseline["prompt_rows"] != native["prompt_rows"]
            or baseline["output_rows"] != native["output_rows"]
        ):
            raise ValueError(
                f"output parity mismatch for batch {batch_size}"
            )
        parity[f"b{batch_size}"] = True
    if value.get("parity") != parity:
        raise ValueError("parity summary mismatch")
    limitations = value.get("limitations")
    if limitations != list(REQUIRED_LIMITATIONS):
        raise ValueError("authority limitations mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "promotion_classification": (
            PROMOTION_CLASSIFICATION
        ),
        "target_model_manifest_sha256": target_digest,
        "mtp_checkpoint_manifest_sha256": mtp_digest,
        "source_tree_sha256": source_digest,
        "world_size": WORLD_SIZE,
        "gpu_index": gpu_index,
        "cells": normalized_cells,
        "parity": parity,
        "limitations": list(REQUIRED_LIMITATIONS),
    }


def assemble_authority(
    cells: dict,
    *,
    source_tree_sha256: str,
    target_model_manifest_sha256: str,
    mtp_checkpoint_manifest_sha256: str,
    gpu_index: int,
    limitations: list[str],
) -> dict:
    return validate_result({
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "promotion_classification": (
            PROMOTION_CLASSIFICATION
        ),
        "target_model_manifest_sha256": (
            target_model_manifest_sha256
        ),
        "mtp_checkpoint_manifest_sha256": (
            mtp_checkpoint_manifest_sha256
        ),
        "source_tree_sha256": source_tree_sha256,
        "world_size": WORLD_SIZE,
        "gpu_index": gpu_index,
        "cells": cells,
        "parity": {
            f"b{batch_size}": True
            for batch_size in BATCH_SIZES
        },
        "limitations": limitations,
    })


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


def source_tree_sha256(
    root: Path,
    files: tuple[str, ...],
) -> str:
    hashes = hash_source_files(root, files)
    return source_hashes_sha256(hashes)


def source_hashes_sha256(
    hashes: dict[str, str],
) -> str:
    digest = hashlib.sha256()
    for name in sorted(hashes):
        name_bytes = name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(bytes.fromhex(hashes[name]))
    return digest.hexdigest()


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


def publish_authority(
    output_dir: Path,
    result: dict,
    *,
    source_files: dict[str, str],
) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("authority output path already exists")
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.",
        dir=parent,
    ))
    try:
        normalized = validate_result(result)
        normalized_source_files = {}
        for name, digest in source_files.items():
            if not isinstance(name, str) or not name:
                raise ValueError("source file name is invalid")
            normalized_source_files[name] = _sha256(
                digest,
                "source file digest",
            )
        atomic_write_json(
            temporary_dir / "result.json",
            normalized,
        )
        status = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "classification": CLASSIFICATION,
            "promotion_classification": (
                PROMOTION_CLASSIFICATION
            ),
        }
        atomic_write_json(
            temporary_dir / "status.json",
            status,
        )
        atomic_write_json(
            temporary_dir / "source_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "source_tree_sha256": normalized[
                    "source_tree_sha256"
                ],
                "target_model_manifest_sha256": normalized[
                    "target_model_manifest_sha256"
                ],
                "mtp_checkpoint_manifest_sha256": normalized[
                    "mtp_checkpoint_manifest_sha256"
                ],
                "source_files": normalized_source_files,
                "artifacts": {
                    "result.json": sha256_file(
                        temporary_dir / "result.json"
                    ),
                    "status.json": sha256_file(
                        temporary_dir / "status.json"
                    ),
                },
            },
        )
        os.replace(temporary_dir, output_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_default_verifier():
    module = _load_module(
        "verify_qwen35_native_mtp_tp1_4k_engine_gate",
        Path(__file__).resolve().parent
        / "verify_qwen35_native_mtp_tp1_4k_engine_gate.py",
    )
    return module.verify_run


def run_campaign(
    *,
    model_path: str,
    gpu_index: int,
    output_dir: Path,
    repo_root: Path | None = None,
    worker_script: Path | None = None,
    python_executable: str = sys.executable,
    source_files: tuple[str, ...] = DEFAULT_SOURCE_FILES,
    verifier=None,
) -> dict:
    gpu_index = _integer(gpu_index, "GPU index")
    repo_root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else Path(repo_root)
    )
    worker_script = (
        Path(__file__).resolve().parent
        / "qwen35_native_mtp_tp1_4k_engine_worker.py"
        if worker_script is None
        else Path(worker_script)
    )
    output_dir = Path(output_dir)
    failed_dir = output_dir.with_name(
        f"{output_dir.name}.failed"
    )
    if output_dir.exists() or failed_dir.exists():
        raise ValueError(
            "authority output path already exists"
        )
    source_hashes = hash_source_files(
        repo_root,
        source_files,
    )
    source_digest = source_hashes_sha256(source_hashes)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.campaign.",
        dir=output_dir.parent,
    ))
    try:
        cell_dir = temporary_root / "cells"
        cell_dir.mkdir()
        cells = {}
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
                    "--gpu-index",
                    str(gpu_index),
                    "--policy",
                    policy,
                    "--batch-size",
                    str(batch_size),
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
                        f"worker failed for {key}; "
                        f"log={log_path}"
                    )
                cells[key] = validate_cell_result(
                    json.loads(
                        cell_path.read_text(
                            encoding="utf-8"
                        )
                    )
                )
        worker_module = _load_module(
            "qwen35_native_mtp_tp1_4k_engine_worker",
            worker_script,
        )
        target_digest = (
            worker_module.target_model_manifest_sha256(
                model_path
            )
        )
        mtp_digest = (
            worker_module.mtp_checkpoint_manifest_sha256(
                model_path
            )
        )
        final_source_hashes = hash_source_files(
            repo_root,
            source_files,
        )
        if final_source_hashes != source_hashes:
            raise RuntimeError(
                "source changed during campaign"
            )
        result = assemble_authority(
            cells,
            source_tree_sha256=source_digest,
            target_model_manifest_sha256=target_digest,
            mtp_checkpoint_manifest_sha256=mtp_digest,
            gpu_index=gpu_index,
            limitations=list(REQUIRED_LIMITATIONS),
        )
        authority_dir = temporary_root / "authority"
        publish_authority(
            authority_dir,
            result,
            source_files=source_hashes,
        )
        shutil.copytree(
            cell_dir,
            authority_dir / "cells",
        )
        verify = (
            _load_default_verifier()
            if verifier is None
            else verifier
        )
        verification = verify(authority_dir, repo_root)
        atomic_write_json(
            authority_dir / "verify.json",
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
                )
            )
        os.replace(authority_dir, output_dir)
        shutil.rmtree(temporary_root)
        return result
    except Exception as error:
        if temporary_root.exists():
            os.replace(temporary_root, failed_dir)
        raise RuntimeError(
            f"{error}; failed_artifacts={failed_dir}"
        ) from error
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-index", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_campaign(
        model_path=args.model,
        gpu_index=args.gpu_index,
        output_dir=Path(args.output_dir),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
