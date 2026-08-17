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


def _load_frozen_gate():
    path = (
        Path(__file__).resolve().parent
        / "qwen35_native_mtp_tp4_4k_engine_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_qwen35_native_mtp_tp4_4k_frozen_gate",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_frozen_gate = _load_frozen_gate()


def _load_generic_gate():
    path = (
        Path(__file__).resolve().parent
        / "qwen35_generic_speculative_tp4_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_qwen35_generic_speculative_tp4_validation_gate",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_generic_gate = _load_generic_gate()

SCHEMA_VERSION = (
    "qwen35.native-mtp-tp4-16k-target-kv-offload.v1"
)
CLASSIFICATION = (
    "QWEN35_NATIVE_MTP_TP4_16K_TARGET_KV_OFFLOAD_ESTABLISHED"
)
PROMOTION_CLASSIFICATION = "NOT_PROMOTABLE"
POLICIES = ("baseline", "native_mtp")
BATCH_SIZES = (1, 4)
PROMPT_TOKENS = 16384
MAX_OUTPUT_TOKENS = 8
MAX_PROPOSAL_TOKENS = 4
WORLD_SIZE = 4
RANKS = tuple(range(WORLD_SIZE))
WORKER_RANKS = tuple(range(1, WORLD_SIZE))
MAX_MODEL_LEN = 33024
MAX_NUM_BATCHED_TOKENS = 132096
MAX_NUM_PREFILL_TOKENS_PER_STEP = 1024
KV_OFFLOAD_GPU_BLOCKS = 68
KV_OFFLOAD_LOGICAL_BLOCKS = 640
KV_OFFLOAD_BLOCKWISE_BLOCKS = 8
BLOCK_SIZE = 256

TARGET_MODEL_MANIFEST_SHA256 = (
    _frozen_gate.TARGET_MODEL_MANIFEST_SHA256
)
MTP_CHECKPOINT_MANIFEST_SHA256 = (
    _frozen_gate.MTP_CHECKPOINT_MANIFEST_SHA256
)

REQUIRED_LIMITATIONS = (
    "phase1_not_promotable",
    "proposal_kv_offload_not_established",
    "tp1_16k_not_established",
    "context_32k_not_established",
    "performance_not_established",
    "kv_quantization_not_established",
    "second_learned_structure_not_established",
)

_TOOLS = Path(__file__).resolve().parent
_ROOT = _TOOLS.parent
_WORKER = (
    _TOOLS
    / "qwen35_native_mtp_tp4_16k_target_kv_offload_worker.py"
)
_VERIFIER = (
    _TOOLS
    / "verify_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py"
)
DEFAULT_SOURCE_FILES = tuple(sorted(
    [
        str(path.relative_to(_ROOT))
        for path in (
            _ROOT / "tinyvllm"
        ).rglob("*.py")
    ]
    + [
        "tools/qwen35_generic_speculative_tp4_gate.py",
        "tools/qwen35_native_mtp_tp4_4k_engine_gate.py",
        "tools/qwen35_native_mtp_tp4_4k_engine_worker.py",
        "tools/qwen35_native_mtp_tp1_4k_engine_gate.py",
        "tools/qwen35_native_mtp_tp1_4k_engine_worker.py",
        (
            "tools/qwen35_native_mtp_tp4_16k_"
            "target_kv_offload_gate.py"
        ),
        (
            "tools/qwen35_native_mtp_tp4_16k_"
            "target_kv_offload_worker.py"
        ),
        (
            "tools/verify_qwen35_native_mtp_tp4_16k_"
            "target_kv_offload_gate.py"
        ),
    ]
))

CELL_FIELDS = {
    "schema_version",
    "policy",
    "batch_size",
    "world_size",
    "rank_inventory",
    "gpu_indices",
    "prompt_token_count",
    "max_output_tokens",
    "max_proposal_tokens",
    "model_identity",
    "engine_config",
    "prompt_rows",
    "output_rows",
    "rank_snapshots",
    "side_state_receipts",
    "target_kv_receipts",
    "residency_phases",
    "kv_rank_deltas",
    "kv_capacity_rows",
    "runtime_poisoned",
    "cleanup",
}

RESULT_FIELDS = {
    "schema_version",
    "classification",
    "promotion_classification",
    "target_model_manifest_sha256",
    "mtp_checkpoint_manifest_sha256",
    "source_tree_sha256",
    "world_size",
    "rank_inventory",
    "gpu_indices",
    "gpu_process_inventory_before",
    "gpu_process_inventory_after",
    "cells",
    "parity",
    "limitations",
}

_exact_keys = _frozen_gate._exact_keys
_integer = _frozen_gate._integer
_sha256 = _frozen_gate._sha256
_json_sha256 = _frozen_gate._json_sha256
_validate_model_identity = _frozen_gate._validate_model_identity
_validate_token_rows = _frozen_gate._validate_token_rows
_validate_receipts = _frozen_gate._validate_receipts
_validate_cleanup = _frozen_gate._validate_cleanup
atomic_write_json = _frozen_gate.atomic_write_json

_rank_gate = _load_frozen_gate()
_rank_gate.PROMPT_TOKENS = PROMPT_TOKENS
_rank_gate.MAX_OUTPUT_TOKENS = MAX_OUTPUT_TOKENS
_rank_gate.MAX_PROPOSAL_TOKENS = MAX_PROPOSAL_TOKENS
_validate_rank_snapshots = _rank_gate._validate_rank_snapshots

MOVEMENT_KEYS = _generic_gate.MOVEMENT_KEYS
validate_residency_phases = (
    _generic_gate.validate_residency_phases
)
_validate_kv_rank_deltas = (
    _generic_gate._validate_kv_rank_deltas
)


def cell_key(policy: str, batch_size: int) -> str:
    if policy not in POLICIES:
        raise ValueError("unsupported policy")
    if batch_size not in BATCH_SIZES:
        raise ValueError("unsupported batch size")
    return f"{policy}:b{batch_size}"


def validate_engine_config(
    value: object,
    *,
    policy: str,
    batch_size: int,
) -> dict:
    expected = {
        "tensor_parallel_size": WORLD_SIZE,
        "enforce_eager": True,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_num_prefill_tokens_per_step": (
            MAX_NUM_PREFILL_TOKENS_PER_STEP
        ),
        "max_num_seqs": batch_size,
        "kvcache_block_size": BLOCK_SIZE,
        "chunked_prefill_decode_first": False,
        "chunked_prefill_mixed_batch": False,
        "kv_offload_mvp0": True,
        "kv_offload_gpu_blocks": KV_OFFLOAD_GPU_BLOCKS,
        "kv_offload_logical_blocks": (
            KV_OFFLOAD_LOGICAL_BLOCKS
        ),
        "kv_offload_blockwise_prefill": True,
        "kv_offload_blockwise_decode": True,
        "kv_offload_blockwise_blocks": (
            KV_OFFLOAD_BLOCKWISE_BLOCKS
        ),
        "qwen35_mtp_enabled": policy == "native_mtp",
        "qwen35_mtp_cuda_graphs": False,
        "qwen35_mtp_max_proposal_tokens": (
            MAX_PROPOSAL_TOKENS
        ),
    }
    if value != expected:
        raise ValueError("engine configuration mismatch")
    return dict(expected)


def validate_kv_rank_deltas(value: object) -> list[dict]:
    return _validate_kv_rank_deltas(value)


def validate_kv_capacity_rows(value: object) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("target-KV capacity rows must be a list")
    rows_by_rank = {}
    fields = {
        "rank",
        "provenance",
        "gpu_blocks",
        "logical_blocks",
        "resident_blocks",
        "peak_resident_blocks",
    }
    for row in value:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError(
                "target-KV capacity row is invalid"
            )
        rank = row["rank"]
        if rank not in RANKS or rank in rows_by_rank:
            raise ValueError(
                "target-KV capacity rank inventory mismatch"
            )
        if (
            row["provenance"]
            != "engine.kv_offload_summaries"
        ):
            raise ValueError(
                "target-KV capacity provenance is invalid"
            )
        if row["gpu_blocks"] != KV_OFFLOAD_GPU_BLOCKS:
            raise ValueError(
                "target-KV GPU block capacity mismatch"
            )
        if (
            row["logical_blocks"]
            != KV_OFFLOAD_LOGICAL_BLOCKS
        ):
            raise ValueError(
                "target-KV logical block capacity mismatch"
            )
        resident = row["resident_blocks"]
        peak = row["peak_resident_blocks"]
        if (
            isinstance(resident, bool)
            or not isinstance(resident, int)
            or resident < 0
        ):
            raise ValueError(
                "target-KV resident block count is invalid"
            )
        if resident > KV_OFFLOAD_GPU_BLOCKS:
            raise ValueError(
                "target-KV resident blocks exceed GPU capacity"
            )
        if (
            isinstance(peak, bool)
            or not isinstance(peak, int)
            or peak < resident
        ):
            raise ValueError(
                "target-KV peak residency is invalid"
            )
        if peak > KV_OFFLOAD_GPU_BLOCKS:
            raise ValueError(
                "target-KV peak residency exceeds GPU capacity"
            )
        rows_by_rank[rank] = dict(row)
    if tuple(sorted(rows_by_rank)) != RANKS:
        raise ValueError(
            "target-KV capacity rank inventory mismatch"
        )
    return [rows_by_rank[rank] for rank in RANKS]


def validate_cell_result(value: object) -> dict:
    _exact_keys(value, CELL_FIELDS, "cell")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("cell schema version mismatch")
    policy = value["policy"]
    batch_size = value["batch_size"]
    cell_key(policy, batch_size)
    if (
        value["world_size"] != WORLD_SIZE
        or value["rank_inventory"] != list(RANKS)
    ):
        raise ValueError("cell rank inventory mismatch")
    gpu_indices = value["gpu_indices"]
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
        raise ValueError("cell GPU inventory mismatch")
    if value["prompt_token_count"] != PROMPT_TOKENS:
        raise ValueError("cell prompt token count mismatch")
    if value["max_output_tokens"] != MAX_OUTPUT_TOKENS:
        raise ValueError("cell output token count mismatch")
    if (
        value["max_proposal_tokens"]
        != MAX_PROPOSAL_TOKENS
    ):
        raise ValueError("cell proposal token count mismatch")
    model_identity = _validate_model_identity(
        value["model_identity"]
    )
    prompt_rows = _validate_token_rows(
        value["prompt_rows"],
        batch_size=batch_size,
        token_count=PROMPT_TOKENS,
        name="prompt",
    )
    output_rows = _validate_token_rows(
        value["output_rows"],
        batch_size=batch_size,
        token_count=MAX_OUTPUT_TOKENS,
        name="output",
    )
    rank_snapshots = _validate_rank_snapshots(
        value["rank_snapshots"],
        policy=policy,
        batch_size=batch_size,
    )
    if policy == "baseline":
        if value["side_state_receipts"] != []:
            raise ValueError(
                "baseline side-state activity detected"
            )
        if value["target_kv_receipts"] != []:
            raise ValueError(
                "baseline target KV activity detected"
            )
        if value["residency_phases"] != []:
            raise ValueError(
                "baseline residency activity detected"
            )
        side_state = []
        target_kv = []
        residency = []
    else:
        side_state = _validate_receipts(
            value["side_state_receipts"],
            batch_size=batch_size,
            operations=[
                "prepare",
                "select",
                "apply",
                "seal",
            ],
            name="side-state",
        )
        target_kv = _validate_receipts(
            value["target_kv_receipts"],
            batch_size=batch_size,
            operations=["prepare", "commit"],
            name="target KV",
        )
        residency = validate_residency_phases(
            value["residency_phases"]
        )
    if value["runtime_poisoned"] is not False:
        raise ValueError("runtime is poisoned")
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": policy,
        "batch_size": batch_size,
        "world_size": WORLD_SIZE,
        "rank_inventory": list(RANKS),
        "gpu_indices": list(gpu_indices),
        "prompt_token_count": PROMPT_TOKENS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_proposal_tokens": MAX_PROPOSAL_TOKENS,
        "model_identity": model_identity,
        "engine_config": validate_engine_config(
            value["engine_config"],
            policy=policy,
            batch_size=batch_size,
        ),
        "prompt_rows": prompt_rows,
        "output_rows": output_rows,
        "rank_snapshots": rank_snapshots,
        "side_state_receipts": side_state,
        "target_kv_receipts": target_kv,
        "residency_phases": residency,
        "kv_rank_deltas": validate_kv_rank_deltas(
            value["kv_rank_deltas"]
        ),
        "kv_capacity_rows": validate_kv_capacity_rows(
            value["kv_capacity_rows"]
        ),
        "runtime_poisoned": False,
        "cleanup": _validate_cleanup(value["cleanup"]),
    }


def _require_native_batch4_movement(
    cell: dict,
) -> None:
    movement = cell["kv_rank_deltas"]
    for key, message in (
        (
            "d2h_copies",
            "native batch-4 requires real target-KV D2H copies",
        ),
        (
            "d2h_bytes",
            "native batch-4 requires real target-KV D2H bytes",
        ),
        (
            "h2d_copies",
            "native batch-4 requires real target-KV H2D copies",
        ),
        (
            "h2d_bytes",
            "native batch-4 requires real target-KV H2D bytes",
        ),
    ):
        if sum(row[key] for row in movement) <= 0:
            raise ValueError(message)


def validate_result(value: object) -> dict:
    _exact_keys(value, RESULT_FIELDS, "result")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("result schema version mismatch")
    if value["classification"] != CLASSIFICATION:
        raise ValueError("result classification mismatch")
    if (
        value["promotion_classification"]
        != PROMOTION_CLASSIFICATION
    ):
        raise ValueError(
            "result promotion classification mismatch"
        )
    target = _sha256(
        value["target_model_manifest_sha256"],
        "target model manifest",
    )
    mtp = _sha256(
        value["mtp_checkpoint_manifest_sha256"],
        "MTP checkpoint manifest",
    )
    source = _sha256(
        value["source_tree_sha256"],
        "source tree",
    )
    if target != TARGET_MODEL_MANIFEST_SHA256:
        raise ValueError(
            "target model manifest does not match authority"
        )
    if mtp != MTP_CHECKPOINT_MANIFEST_SHA256:
        raise ValueError(
            "MTP checkpoint manifest does not match authority"
        )
    if (
        value["world_size"] != WORLD_SIZE
        or value["rank_inventory"] != list(RANKS)
    ):
        raise ValueError("result rank inventory mismatch")
    gpu_indices = value["gpu_indices"]
    if (
        not isinstance(gpu_indices, list)
        or len(gpu_indices) != WORLD_SIZE
        or len(set(gpu_indices)) != WORLD_SIZE
    ):
        raise ValueError("result GPU inventory mismatch")
    before = value["gpu_process_inventory_before"]
    after = value["gpu_process_inventory_after"]
    if not isinstance(before, list) or not isinstance(
        after,
        list,
    ):
        raise ValueError(
            "GPU process inventory is invalid"
        )
    if before != after:
        raise ValueError("GPU process inventory changed")
    expected_cells = {
        cell_key(policy, batch_size)
        for policy in POLICIES
        for batch_size in BATCH_SIZES
    }
    cells = value["cells"]
    if (
        not isinstance(cells, dict)
        or set(cells) != expected_cells
    ):
        raise ValueError("result cell inventory mismatch")
    normalized_cells = {
        key: validate_cell_result(cells[key])
        for key in sorted(cells)
    }
    parity = {
        "baseline_native": {
            f"b{batch_size}": True
            for batch_size in BATCH_SIZES
        },
    }
    for batch_size in BATCH_SIZES:
        baseline = normalized_cells[
            cell_key("baseline", batch_size)
        ]
        native = normalized_cells[
            cell_key("native_mtp", batch_size)
        ]
        if (
            baseline["gpu_indices"] != gpu_indices
            or native["gpu_indices"] != gpu_indices
        ):
            raise ValueError("cell GPU inventory mismatch")
        if (
            baseline["prompt_rows"] != native["prompt_rows"]
            or baseline["output_rows"] != native["output_rows"]
        ):
            raise ValueError(
                "baseline/native output parity mismatch "
                f"for batch {batch_size}"
            )
    _require_native_batch4_movement(
        normalized_cells[cell_key("native_mtp", 4)]
    )
    if value["parity"] != parity:
        raise ValueError("parity summary mismatch")
    if value["limitations"] != list(REQUIRED_LIMITATIONS):
        raise ValueError("authority limitations mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "promotion_classification": (
            PROMOTION_CLASSIFICATION
        ),
        "target_model_manifest_sha256": target,
        "mtp_checkpoint_manifest_sha256": mtp,
        "source_tree_sha256": source,
        "world_size": WORLD_SIZE,
        "rank_inventory": list(RANKS),
        "gpu_indices": list(gpu_indices),
        "gpu_process_inventory_before": list(before),
        "gpu_process_inventory_after": list(after),
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
    gpu_indices: list[int],
    gpu_process_inventory_before: list,
    gpu_process_inventory_after: list,
    limitations: list[str],
) -> dict:
    return validate_result({
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "promotion_classification": PROMOTION_CLASSIFICATION,
        "target_model_manifest_sha256": (
            target_model_manifest_sha256
        ),
        "mtp_checkpoint_manifest_sha256": (
            mtp_checkpoint_manifest_sha256
        ),
        "source_tree_sha256": source_tree_sha256,
        "world_size": WORLD_SIZE,
        "rank_inventory": list(RANKS),
        "gpu_indices": list(gpu_indices),
        "gpu_process_inventory_before": list(
            gpu_process_inventory_before
        ),
        "gpu_process_inventory_after": list(
            gpu_process_inventory_after
        ),
        "cells": cells,
        "parity": {
            "baseline_native": {
                f"b{batch_size}": True
                for batch_size in BATCH_SIZES
            },
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


def source_tree_sha256(
    root: Path,
    files: tuple[str, ...],
) -> str:
    return source_hashes_sha256(
        hash_source_files(root, files)
    )


def publish_authority(
    output_dir: Path,
    result: dict,
    *,
    source_files: dict[str, str],
) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(
            "authority output path already exists"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ))
    try:
        normalized = validate_result(result)
        normalized_sources = {
            name: _sha256(
                digest,
                "source file digest",
            )
            for name, digest in source_files.items()
        }
        if source_hashes_sha256(normalized_sources) != (
            normalized["source_tree_sha256"]
        ):
            raise ValueError(
                "source tree digest mismatch"
            )
        atomic_write_json(
            temporary / "result.json",
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
            temporary / "status.json",
            status,
        )
        atomic_write_json(
            temporary / "source_manifest.json",
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
                "source_files": normalized_sources,
                "artifacts": {
                    "result.json": sha256_file(
                        temporary / "result.json"
                    ),
                    "status.json": sha256_file(
                        temporary / "status.json"
                    ),
                },
            },
        )
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_default_verifier():
    module = _load_module(
        (
            "verify_qwen35_native_mtp_tp4_16k_"
            "target_kv_offload_gate"
        ),
        _VERIFIER,
    )
    return module.verify_run


def _default_gpu_process_inventory(
    gpu_indices: tuple[int, ...],
) -> list[str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            ",".join(
                str(index)
                for index in gpu_indices
            ),
            (
                "--query-compute-apps="
                "gpu_uuid,pid,process_name,used_gpu_memory"
            ),
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "GPU process inventory failed: "
            + completed.stderr.strip()
        )
    return sorted(
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    )


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
    source_files: tuple[str, ...] | None = None,
    gpu_process_inventory=_default_gpu_process_inventory,
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
        raise ValueError(
            "campaign GPU indices are invalid"
        )
    repo_root = (
        _ROOT
        if repo_root is None
        else Path(repo_root)
    )
    worker_script = (
        _WORKER
        if worker_script is None
        else Path(worker_script)
    )
    source_files = (
        DEFAULT_SOURCE_FILES
        if source_files is None
        else tuple(source_files)
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
    source_digest = source_hashes_sha256(
        source_hashes
    )
    before = gpu_process_inventory(gpu_indices)
    if not isinstance(before, list):
        raise ValueError(
            "GPU process inventory is invalid"
        )
    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_root = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.campaign.",
        dir=output_dir.parent,
    ))
    try:
        atomic_write_json(
            temporary_root / "source_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "source_tree_sha256": source_digest,
                "target_model_manifest_sha256": (
                    TARGET_MODEL_MANIFEST_SHA256
                ),
                "mtp_checkpoint_manifest_sha256": (
                    MTP_CHECKPOINT_MANIFEST_SHA256
                ),
                "source_files": source_hashes,
            },
        )
        cell_dir = temporary_root / "cells"
        cell_dir.mkdir()
        cells = {}
        ordinal = 0
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
                    str(dist_port_base + ordinal),
                    "--master-port",
                    str(master_port_base + ordinal),
                    "--out",
                    str(cell_path),
                ]
                ordinal += 1
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
                        "worker failed for "
                        f"{key}; log={log_path}"
                    )
                cells[key] = validate_cell_result(
                    json.loads(
                        cell_path.read_text(
                            encoding="utf-8"
                        )
                    )
                )
        worker_module = _load_module(
            (
                "qwen35_native_mtp_tp4_16k_"
                "target_kv_offload_worker"
            ),
            worker_script,
        )
        target_digest = (
            worker_module
            .target_model_manifest_sha256(
                model_path
            )
        )
        mtp_digest = (
            worker_module
            .mtp_checkpoint_manifest_sha256(
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
        after = gpu_process_inventory(gpu_indices)
        if not isinstance(after, list):
            raise ValueError(
                "GPU process inventory is invalid"
            )
        result = assemble_authority(
            cells,
            source_tree_sha256=source_digest,
            target_model_manifest_sha256=target_digest,
            mtp_checkpoint_manifest_sha256=mtp_digest,
            gpu_indices=list(gpu_indices),
            gpu_process_inventory_before=before,
            gpu_process_inventory_after=after,
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
        verification = verify(
            authority_dir,
            repo_root,
        )
        atomic_write_json(
            authority_dir / "verify.json",
            verification,
        )
        if verification != {
            "classification": "PASS",
            "failures": [],
        }:
            raise RuntimeError(
                "independent verification failed: "
                + json.dumps(
                    verification.get(
                        "failures",
                        [],
                    ),
                    sort_keys=True,
                )
            )
        os.replace(authority_dir, output_dir)
        shutil.rmtree(temporary_root)
        return result
    except Exception as error:
        if temporary_root.exists():
            os.replace(
                temporary_root,
                failed_dir,
            )
        raise RuntimeError(
            f"{error}; failed_artifacts={failed_dir}"
        ) from error
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def _gpu_indices_argument(
    value: str,
) -> tuple[int, ...]:
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
            "GPU indices must contain four distinct integers"
        )
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--gpu-indices",
        required=True,
        type=_gpu_indices_argument,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
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
