from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys


def _load_source_bundle():
    name = "qwen35_tp4_32k_h2d_slot_reuse_source_bundle"
    module = sys.modules.get(name)
    if module is not None:
        return module
    path = (
        Path(__file__).resolve().parent
        / "qwen35_tp4_32k_h2d_slot_reuse_source_bundle.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load focused H2D source bundle")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


source_bundle = _load_source_bundle()
SCHEMA = "qwen35.tp4-32k-h2d-source-bound-campaign-plan.v1"
AUTHORIZATION_TEXT = (
    "允许只运行一个 source-bound focused-H2D four-cell campaign"
)
SSH_TARGET = "sitian@10.232.195.203"
REMOTE_ROOT = (
    "/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815"
)
CELLS = (
    "observe:b1",
    "observe:b4",
    "control:b1",
    "control:b4",
)
WORLD_SIZE = 4
GPU_INDICES = (0, 1, 2, 3)
PLAN_NAME = "campaign_plan.json"
WORKER_ARGV = (
    "python",
    (
        "tools/qwen35_tp4_32k_h2d_slot_reuse_"
        "causal_diagnostic_worker.py"
    ),
)
VERIFIER_ARGV = (
    "python",
    (
        "tools/verify_qwen35_tp4_32k_h2d_slot_reuse_"
        "causal_diagnostic.py"
    ),
)
EXECUTION_BOUNDARY_FIELDS = (
    "ssh_authorized",
    "remote_write_authorized",
    "gpu_authorized",
    "cuda_authorized",
    "nccl_authorized",
    "campaign_authorized",
)
PLAN_FIELDS = frozenset({
    "schema",
    "run_tag",
    "authorization_text",
    "ssh_target",
    "remote_root",
    "remote_run",
    "remote_python",
    "remote_model_dir",
    "cells",
    "repetitions_per_cell",
    "world_size",
    "gpu_indices",
    "ports",
    "source_inventory",
    "source_inventory_sha256",
    "source_tar",
    "source_tar_sha256",
    "source_tree_sha256",
    "checkpoint_manifest",
    "checkpoint_manifest_sha256",
    "commands",
    "execution_boundary",
    "claim_boundary",
})


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _expected_commands() -> dict:
    return {
        "worker_argv": list(WORKER_ARGV),
        "verifier_argv": list(VERIFIER_ARGV),
    }


def _expected_execution_boundary() -> dict:
    return {
        field: False for field in EXECUTION_BOUNDARY_FIELDS
    }


def _safe_run_tag(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(
            character not in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789-_"
            )
            for character in value
        )
    ):
        raise ValueError("run tag is unsafe")
    return value


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is invalid")
    return value


def _safe_remote_absolute(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is invalid")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"{label} is unsafe")
    return value


def _validate_environment_path(value: object, label: str) -> str:
    value = _safe_remote_absolute(value, label)
    if not (
        value == REMOTE_ROOT
        or value.startswith(f"{REMOTE_ROOT}/")
        or value == "/data00"
        or value.startswith("/data00/")
    ):
        raise ValueError(f"{label} is outside approved roots")
    return value


def _validate_port(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1024 <= value <= 65535
    ):
        raise ValueError(f"{label} is invalid")
    return value


def _build_plan(
    *,
    source: dict,
    checkpoint_manifest_path: Path,
    run_tag: str,
    remote_python: str,
    remote_model_dir: str,
    repetitions_per_cell: int,
    gpu_indices: tuple[int, ...],
    dist_port: int,
    master_port: int,
) -> dict:
    run_tag = _safe_run_tag(run_tag)
    remote_python = _validate_environment_path(
        remote_python,
        "remote Python",
    )
    remote_model_dir = _validate_environment_path(
        remote_model_dir,
        "remote model directory",
    )
    if (
        isinstance(repetitions_per_cell, bool)
        or not isinstance(repetitions_per_cell, int)
        or repetitions_per_cell <= 0
    ):
        raise ValueError("repetitions per cell must be positive")
    if (
        not isinstance(gpu_indices, tuple)
        or gpu_indices != GPU_INDICES
    ):
        raise ValueError("campaign GPU indices must be 0,1,2,3")
    dist_port = _validate_port(dist_port, "dist port")
    master_port = _validate_port(master_port, "master port")
    if dist_port == master_port:
        raise ValueError("campaign ports must be distinct")
    if (
        not checkpoint_manifest_path.is_file()
        or checkpoint_manifest_path.is_symlink()
    ):
        raise ValueError("checkpoint manifest is missing")
    remote_run = f"{REMOTE_ROOT}/focused_h2d_runs/{run_tag}"
    return {
        "schema": SCHEMA,
        "run_tag": run_tag,
        "authorization_text": AUTHORIZATION_TEXT,
        "ssh_target": SSH_TARGET,
        "remote_root": REMOTE_ROOT,
        "remote_run": remote_run,
        "remote_python": remote_python,
        "remote_model_dir": remote_model_dir,
        "cells": list(CELLS),
        "repetitions_per_cell": repetitions_per_cell,
        "world_size": WORLD_SIZE,
        "gpu_indices": list(gpu_indices),
        "ports": {
            "dist_port": dist_port,
            "master_port": master_port,
        },
        "source_inventory": source["source_inventory"],
        "source_inventory_sha256": source[
            "source_inventory_sha256"
        ],
        "source_tar": source["source_tar"],
        "source_tar_sha256": source["source_tar_sha256"],
        "source_tree_sha256": source["source_tree_sha256"],
        "checkpoint_manifest": str(checkpoint_manifest_path),
        "checkpoint_manifest_sha256": source_bundle.sha256_file(
            checkpoint_manifest_path
        ),
        "commands": _expected_commands(),
        "execution_boundary": _expected_execution_boundary(),
        "claim_boundary": (
            "local source/provenance/authorization preparation only; "
            "no SSH, remote write, GPU, CUDA, NCCL, campaign execution, "
            "causality, correctness, performance, Phase-1, or promotion claim"
        ),
    }


def validate_campaign_plan(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != PLAN_FIELDS:
        raise ValueError("campaign plan fields mismatch")
    plan = dict(value)
    if (
        plan["schema"] != SCHEMA
        or plan["authorization_text"] != AUTHORIZATION_TEXT
        or plan["ssh_target"] != SSH_TARGET
        or plan["remote_root"] != REMOTE_ROOT
        or plan["cells"] != list(CELLS)
        or plan["world_size"] != WORLD_SIZE
        or plan["commands"] != _expected_commands()
        or plan["execution_boundary"]
        != _expected_execution_boundary()
    ):
        raise ValueError("campaign plan frozen contract mismatch")
    run_tag = _safe_run_tag(plan["run_tag"])
    if plan["remote_run"] != (
        f"{REMOTE_ROOT}/focused_h2d_runs/{run_tag}"
    ):
        raise ValueError("campaign remote output path mismatch")
    _validate_environment_path(plan["remote_python"], "remote Python")
    _validate_environment_path(
        plan["remote_model_dir"],
        "remote model directory",
    )
    repetitions = plan["repetitions_per_cell"]
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions <= 0
    ):
        raise ValueError("campaign repetition count mismatch")
    gpu_indices = plan["gpu_indices"]
    if gpu_indices != list(GPU_INDICES):
        raise ValueError("campaign GPU inventory mismatch")
    ports = plan["ports"]
    if not isinstance(ports, dict) or set(ports) != {
        "dist_port",
        "master_port",
    }:
        raise ValueError("campaign ports mismatch")
    dist_port = _validate_port(ports["dist_port"], "dist port")
    master_port = _validate_port(ports["master_port"], "master port")
    if dist_port == master_port:
        raise ValueError("campaign ports must be distinct")
    source = source_bundle.validate_source_bundle(
        inventory_path=plan["source_inventory"],
        tar_path=plan["source_tar"],
    )
    for field in (
        "source_inventory_sha256",
        "source_tar_sha256",
        "source_tree_sha256",
    ):
        _require_sha256(plan[field], field)
        if plan[field] != source[field]:
            raise ValueError(f"campaign {field} mismatch")
    checkpoint = Path(plan["checkpoint_manifest"])
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise ValueError("campaign checkpoint manifest is missing")
    _require_sha256(
        plan["checkpoint_manifest_sha256"],
        "checkpoint manifest SHA",
    )
    if (
        source_bundle.sha256_file(checkpoint)
        != plan["checkpoint_manifest_sha256"]
    ):
        raise ValueError("campaign checkpoint manifest SHA mismatch")
    if not isinstance(plan["claim_boundary"], str) or not plan[
        "claim_boundary"
    ]:
        raise ValueError("campaign claim boundary is missing")
    return plan


def prepare_local_campaign(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    run_tag: str,
    checkpoint_manifest_path: str | Path,
    remote_python: str,
    remote_model_dir: str,
    repetitions_per_cell: int,
    gpu_indices: tuple[int, ...],
    dist_port: int,
    master_port: int,
) -> dict:
    output = Path(output_dir)
    if output.exists():
        raise ValueError("campaign output already exists")
    output.mkdir(parents=True)
    source = source_bundle.build_source_bundle(
        repo_root=repo_root,
        output_dir=output / "source_bundle",
    )
    plan = _build_plan(
        source=source,
        checkpoint_manifest_path=Path(
            checkpoint_manifest_path
        ).resolve(),
        run_tag=run_tag,
        remote_python=remote_python,
        remote_model_dir=remote_model_dir,
        repetitions_per_cell=repetitions_per_cell,
        gpu_indices=gpu_indices,
        dist_port=dist_port,
        master_port=master_port,
    )
    validate_campaign_plan(plan)
    plan_path = output / PLAN_NAME
    plan_path.write_bytes(canonical_bytes(plan) + b"\n")
    return {
        "plan": plan,
        "plan_path": str(plan_path),
        "plan_sha256": canonical_sha256(plan),
        "source_bundle": source,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repo-root", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--run-tag", required=True)
    prepare.add_argument("--checkpoint-manifest", required=True)
    prepare.add_argument("--remote-python", required=True)
    prepare.add_argument("--remote-model-dir", required=True)
    prepare.add_argument("--repetitions", required=True, type=int)
    prepare.add_argument("--dist-port", required=True, type=int)
    prepare.add_argument("--master-port", required=True, type=int)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--plan", required=True)
    return parser.parse_args(argv)


def _load_plan(path: str | Path) -> dict:
    plan_path = Path(path)
    if not plan_path.is_file() or plan_path.is_symlink():
        raise ValueError("campaign plan is missing")
    try:
        value = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("campaign plan is invalid") from error
    if not isinstance(value, dict):
        raise ValueError("campaign plan must be an object")
    return value


def main(
    argv=None,
    *,
    prepare_fn=None,
    validate_fn=None,
) -> int:
    args = parse_args(argv)
    prepare_fn = prepare_local_campaign if prepare_fn is None else prepare_fn
    validate_fn = validate_campaign_plan if validate_fn is None else validate_fn
    if args.command == "prepare":
        prepared = prepare_fn(
            repo_root=Path(args.repo_root),
            output_dir=Path(args.output_dir),
            run_tag=args.run_tag,
            checkpoint_manifest_path=Path(
                args.checkpoint_manifest
            ),
            remote_python=args.remote_python,
            remote_model_dir=args.remote_model_dir,
            repetitions_per_cell=args.repetitions,
            gpu_indices=GPU_INDICES,
            dist_port=args.dist_port,
            master_port=args.master_port,
        )
        source = prepared["source_bundle"]
        result = {
            "classification": "PREPARED_LOCAL_ONLY",
            "plan_path": prepared["plan_path"],
            "plan_sha256": prepared["plan_sha256"],
            "source_tree_sha256": source[
                "source_tree_sha256"
            ],
            "source_tar_sha256": source["source_tar_sha256"],
        }
    else:
        plan = _load_plan(args.plan)
        validate_fn(plan)
        result = {
            "classification": "VALID_LOCAL_PLAN",
            "plan_sha256": canonical_sha256(plan),
        }
    sys.stdout.write(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
