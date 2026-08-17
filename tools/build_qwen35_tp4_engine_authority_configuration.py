from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import posixpath
import shutil
import sys
import tempfile


def _load_module(module_name, filename):
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


executor_module = _load_module(
    "qwen35_tp4_engine_correctness_executor",
    "qwen35_tp4_engine_correctness_executor.py",
)
benchmark_contract = _load_module(
    "qwen35_tp4_hybrid_prefix_benchmark_contract",
    "qwen35_tp4_hybrid_prefix_benchmark_contract.py",
)
source_runner = _load_module(
    "run_qwen35_tp4_hybrid_prefix_benchmark_remote",
    "run_qwen35_tp4_hybrid_prefix_benchmark_remote.py",
)


CONFIGURATION_NAME = "executor_configuration.json"
WORKLOAD_MANIFEST_NAME = "workload_manifest.json"
SOURCE_INVENTORY_NAME = "source_inventory.json"
AUTHORITY_OWNED_SOURCE_PATHS = (
    "tinyvllm",
    "tools/qwen35_tp4_hybrid_prefix_benchmark_contract.py",
    "tools/qwen35_tp4_engine_correctness_contract.py",
    "tools/qwen35_tp4_engine_correctness_executor.py",
    "tools/qwen35_tp4_engine_correctness_producer.py",
    "tools/qwen35_tp4_engine_backend_session.py",
    "tools/qwen35_tp4_engine_reference_tokens.py",
    "tools/qwen35_tp4_engine_reference_tokens_producer.py",
    "tools/qwen35_tp4_engine_official_reference_executor.py",
    "tools/qwen35_tp4_cached_continuation_correctness_contract.py",
    "tools/qwen35_tp4_cached_continuation_correctness_executor.py",
    "tools/qwen35_tp4_cached_continuation_backend_session.py",
    "tools/qwen35_tp4_cached_first_divergence_probe.py",
    "tools/qwen35_tp4_cached_partition_diagnostic.py",
    "tools/qwen35_tp4_cached_continuation_correctness_producer.py",
    "tools/run_qwen35_tp4_cached_continuation_authority.py",
    "tools/verify_qwen35_tp4_cached_continuation_correctness_gate.py",
    "tools/verify_qwen35_tp4_engine_reference_tokens.py",
    "tools/verify_qwen35_tp4_engine_correctness_gate.py",
    "tools/run_qwen35_tp4_engine_correctness_authority.py",
    "tools/verify_qwen35_tp4_engine_correctness_authority.py",
    "tools/qwen35_tp4_engine_remote_execution_plan.py",
    "tools/qwen35_tp4_engine_remote_execution_receipt.py",
    "tools/qwen35_tp4_engine_remote_execution_executor.py",
    "tools/qwen35_tp4_engine_remote_execution_source_contract.py",
    "tools/qwen35_tp4_engine_remote_execution_authorization.py",
    "tools/qwen35_tp4_engine_remote_subprocess_adapter.py",
    "tools/qwen35_tp4_correctness_resource_policy.py",
    "tools/qwen35_tp4_cached_continuation_remote_execution_plan.py",
    "tools/qwen35_tp4_cached_continuation_remote_execution_receipt.py",
    "tools/qwen35_tp4_cached_continuation_remote_execution_executor.py",
)


def _write_json(path, payload):
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regular_model_manifest(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("model manifest is missing")
    return path.resolve()


def _load_model_manifest(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("model manifest is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError("model manifest is invalid")
    return payload


def _remote_absolute_path(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be an absolute POSIX path")
    normalized = posixpath.normpath(value)
    path = PurePosixPath(normalized)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute POSIX path")
    return path.as_posix()


def _publish_configuration(
    *,
    repo_root,
    output_dir,
    model_dir,
    model_manifest_path,
    model_fingerprint,
    gpu_indices,
    dist_port,
    master_port,
    max_cache_entries,
    max_cache_bytes,
    timeout_s,
):
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ))
    try:
        files = source_runner._owned_source_files(
            repo_root,
            AUTHORITY_OWNED_SOURCE_PATHS,
        )
        source_tree_sha256 = source_runner._source_tree_sha256(files)
        workload_manifest = (
            benchmark_contract.workload_manifest_payload()
        )
        workload_path = temporary / WORKLOAD_MANIFEST_NAME
        _write_json(workload_path, workload_manifest)
        configuration = executor_module.ExecutorConfiguration(
            model_dir=str(model_dir),
            model_manifest_path=str(model_manifest_path),
            model_manifest_sha256=_sha256(model_manifest_path),
            source_tree_sha256=source_tree_sha256,
            workload_manifest_path=str(
                output_dir / WORKLOAD_MANIFEST_NAME
            ),
            workload_manifest_sha256=_sha256(workload_path),
            model_fingerprint=model_fingerprint,
            gpu_indices=tuple(gpu_indices),
            dist_port=dist_port,
            master_port=master_port,
            max_cache_entries=max_cache_entries,
            max_cache_bytes=max_cache_bytes,
            timeout_s=timeout_s,
        )
        payload = configuration.to_payload()
        _write_json(temporary / CONFIGURATION_NAME, payload)
        _write_json(
            temporary / SOURCE_INVENTORY_NAME,
            {
                "owned_files": [
                    relative for relative, _ in files
                ],
                "source_tree_sha256": source_tree_sha256,
            },
        )
        os.replace(temporary, output_dir)
        return payload
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def build_configuration(
    *,
    repo_root,
    output_dir,
    model_dir,
    model_manifest_path,
    model_fingerprint,
    gpu_indices,
    dist_port,
    master_port,
    max_cache_entries,
    max_cache_bytes,
    timeout_s,
):
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    model_dir = Path(model_dir).resolve()
    if output_dir.exists():
        raise ValueError("configuration output directory already exists")
    if not repo_root.is_dir():
        raise ValueError("repository root is missing")
    if not model_dir.is_dir():
        raise ValueError("model directory is missing")
    model_manifest_path = _regular_model_manifest(model_manifest_path)
    return _publish_configuration(
        repo_root=repo_root,
        output_dir=output_dir,
        model_dir=model_dir,
        model_manifest_path=model_manifest_path,
        model_fingerprint=model_fingerprint,
        gpu_indices=gpu_indices,
        dist_port=dist_port,
        master_port=master_port,
        max_cache_entries=max_cache_entries,
        max_cache_bytes=max_cache_bytes,
        timeout_s=timeout_s,
    )


def build_remote_configuration(
    *,
    repo_root,
    output_dir,
    model_manifest_path,
    remote_model_dir,
    model_fingerprint,
    gpu_indices,
    dist_port,
    master_port,
    max_cache_entries,
    max_cache_bytes,
    timeout_s,
):
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("configuration output directory already exists")
    if not repo_root.is_dir():
        raise ValueError("repository root is missing")
    model_manifest_path = _regular_model_manifest(model_manifest_path)
    manifest = _load_model_manifest(model_manifest_path)
    remote_path = _remote_absolute_path(
        remote_model_dir,
        "remote model directory",
    )
    manifest_remote_path = _remote_absolute_path(
        manifest.get("remote_model_dir"),
        "manifest remote model directory",
    )
    if remote_path != manifest_remote_path:
        raise ValueError("remote model directory does not match manifest")
    return _publish_configuration(
        repo_root=repo_root,
        output_dir=output_dir,
        model_dir=remote_path,
        model_manifest_path=model_manifest_path,
        model_fingerprint=model_fingerprint,
        gpu_indices=gpu_indices,
        dist_port=dist_port,
        master_port=master_port,
        max_cache_entries=max_cache_entries,
        max_cache_bytes=max_cache_bytes,
        timeout_s=timeout_s,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-dir", required=True)
    model_mode = parser.add_mutually_exclusive_group(required=True)
    model_mode.add_argument("--model-dir")
    model_mode.add_argument("--remote-model-dir")
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--gpu-indices", required=True)
    parser.add_argument("--dist-port", required=True, type=int)
    parser.add_argument("--master-port", required=True, type=int)
    parser.add_argument("--max-cache-entries", required=True, type=int)
    parser.add_argument("--max-cache-bytes", required=True, type=int)
    parser.add_argument("--timeout-s", required=True, type=float)
    args = parser.parse_args(argv)
    gpu_indices = tuple(
        int(value) for value in args.gpu_indices.split(",")
    )
    common = {
        "repo_root": args.repo_root,
        "output_dir": args.output_dir,
        "model_manifest_path": args.model_manifest,
        "model_fingerprint": args.model_fingerprint,
        "gpu_indices": gpu_indices,
        "dist_port": args.dist_port,
        "master_port": args.master_port,
        "max_cache_entries": args.max_cache_entries,
        "max_cache_bytes": args.max_cache_bytes,
        "timeout_s": args.timeout_s,
    }
    if args.remote_model_dir is not None:
        result = build_remote_configuration(
            remote_model_dir=args.remote_model_dir,
            **common,
        )
    else:
        result = build_configuration(
            model_dir=args.model_dir,
            **common,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
