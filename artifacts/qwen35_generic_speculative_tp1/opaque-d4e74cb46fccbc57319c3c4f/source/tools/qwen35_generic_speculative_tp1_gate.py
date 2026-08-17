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
    "qwen35.generic-speculative-tp1-"
    "transactional-correctness.v1"
)
CLASSIFICATION = "SECOND_MODEL_TP1_ESTABLISHED"
POLICIES = ("baseline", "ngram")
BATCH_SIZES = (1, 4)
WORLD_SIZE = 1
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
    "tools/qwen35_generic_speculative_tp1_gate.py",
    "tools/qwen35_generic_speculative_tp1_worker.py",
    "tools/verify_qwen35_generic_speculative_tp1_gate.py",
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
            not isinstance(item, str) or not item
            for item in architectures
        )
    ):
        raise ValueError("model architecture inventory is invalid")
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
    return {
        "model_type": "qwen3_5",
        "architectures": list(architectures),
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
        "proposal_token_count": proposal_count,
        "accepted_draft_count": accepted_count,
        "verify_input_count": verify_count,
        "committed_tail_input_count": committed_tail,
        "committed_input_count": committed_input,
    }


def _validate_side_state(
    receipts: object,
    failure_rollbacks: object,
) -> list[dict]:
    if not isinstance(receipts, list):
        raise ValueError(
            "side-state receipts must be a list"
        )
    if not isinstance(failure_rollbacks, list):
        raise ValueError(
            "failure-path rollbacks must be a list"
        )
    normalized = []
    by_lifecycle: dict[tuple[str, int], list[str]] = {}
    for receipt in receipts:
        if not isinstance(receipt, dict):
            raise ValueError(
                "side-state receipt must be a mapping"
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
            not in {"prepare", "select", "apply", "seal", "rollback"}
            or not isinstance(state, str)
            or not state
        ):
            raise ValueError("side-state receipt is invalid")
        by_lifecycle.setdefault(
            (handle_id, sequence_id),
            [],
        ).append(operation)
        normalized.append(dict(receipt))
    required = ["prepare", "select", "apply", "seal"]
    successful_handles = [
        operations
        for operations in by_lifecycle.values()
        if "seal" in operations
    ]
    if not successful_handles or any(
        operations != required
        for operations in successful_handles
    ):
        raise ValueError(
            "side-state lifecycle receipts are incomplete"
        )
    rollback_handles = {
        handle_id
        for (handle_id, _), operations in by_lifecycle.items()
        if "rollback" in operations
    }
    for row in failure_rollbacks:
        if (
            not isinstance(row, dict)
            or row.get("handle_id") not in rollback_handles
        ):
            raise ValueError(
                "failure-path rollback receipt is missing"
            )
    return normalized


def _validate_runtime(
    value: object,
    *,
    policy: str,
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
        name: _integer(value.get(name), name.replace("_", " "))
        for name in names
    }
    mappings_value = value.get("consumed_input_mappings")
    if not isinstance(mappings_value, list):
        raise ValueError(
            "consumed input mappings must be a list"
        )
    failure_rollbacks = value.get("failure_path_rollbacks")
    if policy == "baseline":
        if any(counters.values()):
            raise ValueError(
                "baseline speculative counters must be zero"
            )
        if mappings_value or value.get("side_state_receipts"):
            raise ValueError(
                "baseline side-state evidence must be empty"
            )
        if failure_rollbacks:
            raise ValueError(
                "baseline rollback evidence must be empty"
            )
        mappings = []
        receipts = []
    else:
        for name in (
            "proposal_rows",
            "proposed_tokens",
            "accepted_draft_tokens",
            "rejected_draft_tokens",
        ):
            if counters[name] == 0:
                raise ValueError(
                    f"{name.replace('_', ' ')} must be positive"
                )
        if counters["first_target_callbacks"] == 0:
            raise ValueError(
                "first-target callbacks must be positive"
            )
        if counters["verify_callbacks"] == 0:
            raise ValueError(
                "verify callback count must be positive"
            )
        if counters["accepted_prefix_replays"] != 0:
            raise ValueError(
                "accepted-prefix replay count must be zero"
            )
        mappings = [
            _validate_mapping(row)
            for row in mappings_value
        ]
        if not mappings:
            raise ValueError(
                "consumed input mappings are missing"
            )
        receipts = _validate_side_state(
            value.get("side_state_receipts"),
            failure_rollbacks,
        )
    return {
        **counters,
        "consumed_input_mappings": mappings,
        "side_state_receipts": receipts,
        "failure_path_rollbacks": list(
            failure_rollbacks or []
        ),
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
    gpu_index = _integer(value.get("gpu_index"), "GPU index")
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
    runtime = _validate_runtime(
        value.get("runtime"),
        policy=policy,
    )
    leases = value.get("lease_inventory")
    if not isinstance(leases, dict):
        raise ValueError("lease inventory must be a mapping")
    before = _integer(leases.get("before"), "lease count before")
    after = _integer(leases.get("after"), "lease count after")
    leaked = leases.get("leaked_sequence_ids")
    if (
        after != before
        or not isinstance(leaked, list)
        or leaked
    ):
        raise ValueError("hybrid-state lease leak detected")
    if value.get("runtime_poisoned") is not False:
        raise ValueError("runtime is poisoned")
    cleanup = value.get("cleanup_receipt")
    if (
        not isinstance(cleanup, dict)
        or cleanup.get("engine_exit_called") is not True
        or cleanup.get("worker_exit_code") != 0
        or cleanup.get("owned_children_remaining") != []
    ):
        raise ValueError("cleanup receipt is incomplete")
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": policy,
        "batch_size": batch_size,
        "world_size": WORLD_SIZE,
        "gpu_index": gpu_index,
        "model_identity": model_identity,
        "prompt_rows": prompt_rows,
        "output_rows": output_rows,
        "runtime": runtime,
        "lease_inventory": {
            "before": before,
            "after": after,
            "leaked_sequence_ids": [],
        },
        "runtime_poisoned": False,
        "cleanup_receipt": dict(cleanup),
    }


def validate_result(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("result must be a mapping")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("result schema version mismatch")
    if value.get("classification") != CLASSIFICATION:
        raise ValueError("result classification mismatch")
    model_digest = _sha256(
        value.get("model_manifest_sha256"),
        "model manifest",
    )
    if model_digest != MODEL_MANIFEST_SHA256:
        raise ValueError("model manifest does not match authority")
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
        candidate = normalized_cells[
            cell_key("ngram", batch_size)
        ]
        if (
            baseline["prompt_rows"] != candidate["prompt_rows"]
            or baseline["output_rows"]
            != candidate["output_rows"]
        ):
            raise ValueError(
                f"output parity mismatch for batch {batch_size}"
            )
        parity[f"b{batch_size}"] = True
    if value.get("parity") != parity:
        raise ValueError("parity summary mismatch")
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "model_manifest_sha256": model_digest,
        "source_tree_sha256": source_digest,
        "world_size": WORLD_SIZE,
        "gpu_index": gpu_index,
        "cells": normalized_cells,
        "parity": parity,
    }
    opaque_run_id = value.get("opaque_run_id")
    if opaque_run_id is not None:
        if not isinstance(opaque_run_id, str) or not opaque_run_id:
            raise ValueError("opaque run ID is invalid")
        normalized["opaque_run_id"] = opaque_run_id
    return normalized


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
    digest = hashlib.sha256()
    for name in sorted(hashes):
        name_bytes = name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(bytes.fromhex(hashes[name]))
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


def _load_default_verifier():
    path = (
        Path(__file__).resolve().parent
        / "verify_qwen35_generic_speculative_tp1_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "verify_qwen35_generic_speculative_tp1_gate",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
        / "qwen35_generic_speculative_tp1_worker.py"
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
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ))
    try:
        cell_dir = temporary_dir / "cells"
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
                with log_path.open("w", encoding="utf-8") as log:
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
                        f"worker failed for {key}; log={log_path}"
                    )
                cells[key] = validate_cell_result(
                    json.loads(
                        cell_path.read_text(encoding="utf-8")
                    )
                )
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
            "model_manifest_sha256": model_digest,
            "source_tree_sha256": source_digest,
            "world_size": WORLD_SIZE,
            "gpu_index": gpu_index,
            "cells": cells,
            "parity": {"b1": True, "b4": True},
        })
        atomic_write_json(temporary_dir / "result.json", result)
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
        verification = verify(temporary_dir, repo_root)
        atomic_write_json(
            temporary_dir / "verify.json",
            verification,
        )
        if (
            verification.get("classification")
            != CLASSIFICATION
            or verification.get("failures") != []
        ):
            raise RuntimeError(
                "independent verification failed: "
                + json.dumps(
                    verification.get("failures", []),
                    sort_keys=True,
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
