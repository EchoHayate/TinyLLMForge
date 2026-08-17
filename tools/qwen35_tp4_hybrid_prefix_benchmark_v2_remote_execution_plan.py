from __future__ import annotations

import copy
import importlib.util
from pathlib import Path, PurePosixPath
import sys


def _load_contract():
    name = "qwen35_tp4_hybrid_prefix_benchmark_v2_contract_for_remote_plan"
    module = sys.modules.get(name)
    if module is not None:
        return module
    path = Path(__file__).resolve().parent / (
        "qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract()


def _canonicalize_ports(rows):
    if not isinstance(rows, list):
        raise ValueError("case port pairs are invalid")
    result = copy.deepcopy(rows)
    for row in result:
        if "dist_port" in row and "tinyvllm_dist_port" not in row:
            row["tinyvllm_dist_port"] = row.pop("dist_port")
    contract._validate_case_port_pairs(result)
    return result


def _gpu_assignments():
    result = [
        {
            "rank": rank,
            "gpu_index": gpu_index,
            "cuda_visible_device": str(rank),
        }
        for rank, gpu_index in enumerate(contract.REQUIRED_GPU_INDICES)
    ]
    contract._validate_gpu_assignments(result)
    return result


def _artifact_paths(artifact_root):
    root = PurePosixPath(artifact_root)
    if root.is_absolute() or ".." in root.parts:
        raise ValueError("artifact root is invalid")
    value = {
        "remote_run": f"{root}/remote-run",
        "remote_artifact": f"{root}/remote-run/artifact",
        "package": f"{root}/package.tar",
        "local_extract": f"{root}/local-artifact",
    }
    contract._validate_artifact_paths(value)
    return value


def _physical_root_sha256(path):
    return contract.physical_directory_sha256(path)


def build_remote_execution_plan(
    *,
    preflight,
    case_port_pairs,
    artifact_root,
    authority_root,
    physical_artifact_root,
):
    contract.validate_evidence_document("preflight", preflight)
    if preflight.get("classification") != "READY":
        raise ValueError("preflight must be READY")
    if (
        preflight["worker_authorized"] is not True
        or preflight["blocking_reasons"] != []
        or preflight["remote_path_created"] is not False
        or preflight["source_staged"] is not False
        or preflight["worker_launched"] is not False
    ):
        raise ValueError("preflight must be READY and side-effect free")
    ports = _canonicalize_ports(case_port_pairs)
    provenance = {
        field: preflight[field]
        for field in contract.EXECUTION_PROVENANCE_FIELDS
        if field != "command_manifest_sha256"
    }
    provisional = {
        "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[
            "execution_plan"
        ],
        "run_tag": preflight["run_tag"],
        "nonce": preflight["nonce"],
        **provenance,
        "command_manifest_sha256": "0" * 64,
        "authority_root_sha256": _physical_root_sha256(authority_root),
        "physical_artifact_root_sha256": _physical_root_sha256(
            physical_artifact_root
        ),
        "required_gpu_indices": list(contract.REQUIRED_GPU_INDICES),
        "world_size": contract.WORLD_SIZE,
        "gpu_assignments": _gpu_assignments(),
        "case_port_pairs": ports,
        "artifact_paths": _artifact_paths(artifact_root),
        "command_order": list(contract.EXECUTION_COMMAND_ORDER),
    }
    commands = contract.canonical_execution_commands(provisional)
    provisional["command_manifest_sha256"] = contract.canonical_json_sha256(
        [
            {
                "name": name,
                "command_sha256": contract.execution_command_sha256(
                    commands[name]
                ),
            }
            for name in contract.EXECUTION_COMMAND_ORDER
        ]
    )
    verify_remote_execution_plan(provisional)
    return provisional


def verify_remote_execution_plan(plan):
    contract.validate_evidence_document("execution_plan", plan)
    contract._validate_gpu_assignments(plan["gpu_assignments"])
    contract._validate_case_port_pairs(plan["case_port_pairs"])
    contract._validate_artifact_paths(plan["artifact_paths"])
    commands = contract.canonical_execution_commands(plan)
    contract.validate_execution_command_semantics(
        commands,
        expected_order=contract.EXECUTION_COMMAND_ORDER,
        execution_plan=plan,
    )
    expected = contract.canonical_json_sha256(
        [
            {
                "name": name,
                "command_sha256": contract.execution_command_sha256(
                    commands[name]
                ),
            }
            for name in contract.EXECUTION_COMMAND_ORDER
        ]
    )
    if plan["command_manifest_sha256"] != expected:
        raise ValueError("command manifest identity is invalid")
    return copy.deepcopy(plan)
