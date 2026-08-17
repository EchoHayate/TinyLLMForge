#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import socket
import subprocess
import sys
import tarfile
import tempfile

import qwen35_tp4_hybrid_prefix_benchmark_contract as contract


ROOT = Path(__file__).resolve().parents[1]
SSH_TARGET = "sitian@10.232.195.203"
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "env/bin/python"
)
REMOTE_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-tp4-hybrid-prefix-benchmark-runs"
)
OUTPUT_ROOT = ROOT / "experiments" / "qwen35_hybrid_state"
MIN_GPU_FREE_BYTES = contract.MIN_GPU_FREE_BYTES
REQUIRED_GPU_INDICES = (2, 4, 5, 6)
STRICT_EXCLUSIVE_RESOURCE_POLICY = "strict-exclusive"
SHARED_LOW_UTILIZATION_RESOURCE_POLICY = "shared-low-utilization"
MODES = (
    "preflight",
    "smoke",
    "canonical",
    "download-only",
    "verify-only",
)
SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ControlMaster=no",
    "-o",
    "ControlPath=none",
    "-o",
    "ConnectTimeout=20",
)
BENCHMARK_OWNED_SOURCE_PATHS = (
    "tinyvllm",
    "tools/qwen35_tp4_hybrid_prefix_benchmark_contract.py",
    "tools/qwen35_tp4_decode_internal_profile.py",
    "tools/qwen35_tp4_hybrid_prefix_benchmark_worker.py",
    "tools/qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py",
    "tools/qwen35_tp4_hybrid_prefix_benchmark_assembler.py",
    (
        "tools/qwen35_tp4_hybrid_prefix_benchmark_"
        "remote_execution_authorization.py"
    ),
    (
        "tools/qwen35_tp4_hybrid_prefix_benchmark_"
        "remote_execution_receipt.py"
    ),
    (
        "tools/qwen35_tp4_hybrid_prefix_benchmark_"
        "remote_execution_executor.py"
    ),
    (
        "tools/qwen35_tp4_hybrid_prefix_benchmark_"
        "remote_execution_plan.py"
    ),
    "tools/qwen35_tp4_engine_remote_execution_source_contract.py",
    "tools/qwen35_tp4_engine_remote_subprocess_adapter.py",
    "tools/verify_qwen35_tp4_hybrid_prefix_benchmark.py",
)


@dataclass(frozen=True)
class WorkerAuthorization:
    prerequisites_sha256: str
    source_tree_sha256: str
    model_manifest_sha256: str
    workload_manifest_sha256: str
    gpu_indices: tuple[int, int, int, int]

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("worker authorization must be a dictionary")
        try:
            gpu_indices = tuple(value["gpu_indices"])
            authorization = cls(
                prerequisites_sha256=value[
                    "prerequisites_sha256"
                ],
                source_tree_sha256=value[
                    "source_tree_sha256"
                ],
                model_manifest_sha256=value[
                    "model_manifest_sha256"
                ],
                workload_manifest_sha256=value[
                    "workload_manifest_sha256"
                ],
                gpu_indices=gpu_indices,
            )
        except (KeyError, TypeError) as error:
            raise ValueError(
                "worker authorization is invalid"
            ) from error
        validate_worker_authorization(authorization)
        return authorization


@dataclass(frozen=True)
class WorkerRuntimeArtifacts:
    model_dir: str
    model_manifest_path: str
    correctness_prerequisites_path: str
    workload_manifest_path: str


def safe_run_tag(run_tag):
    if (
        not isinstance(run_tag, str)
        or not run_tag
        or any(
            character not in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789-_"
            )
            for character in run_tag
        )
    ):
        raise ValueError("run tag contains unsupported characters")
    return run_tag


def _ssh_argv(remote_argv):
    remote_command = " ".join(
        shlex.quote(str(value)) for value in remote_argv
    )
    return [
        "ssh",
        *SSH_OPTIONS,
        SSH_TARGET,
        "bash",
        "-lc",
        shlex.quote(remote_command),
    ]


def allocate_port_pair():
    handles = []
    ports = []
    try:
        while len(ports) < 2:
            handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            handle.bind(("127.0.0.1", 0))
            handles.append(handle)
            port = int(handle.getsockname()[1])
            if port not in ports:
                ports.append(port)
    finally:
        for handle in handles:
            handle.close()
    return ports[0], ports[1]


def allocate_unique_port_pairs(count, *, allocator=allocate_port_pair):
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("port pair count must be non-negative")
    pairs = []
    used = set()
    for _ in range(count):
        dist_port, master_port = allocator()
        if (
            isinstance(dist_port, bool)
            or not isinstance(dist_port, int)
            or isinstance(master_port, bool)
            or not isinstance(master_port, int)
            or dist_port <= 0
            or master_port <= 0
            or dist_port == master_port
        ):
            raise ValueError("worker port pair is invalid")
        if dist_port in used or master_port in used:
            raise ValueError("duplicate worker port")
        used.update((dist_port, master_port))
        pairs.append((dist_port, master_port))
    return pairs


def allocate_remote_port_pairs(
    count,
    *,
    command_runner,
    execution_env,
):
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("port pair count must be non-negative")
    if not callable(command_runner):
        raise ValueError("remote port command runner is invalid")
    script = "\n".join([
        "import json,socket,sys",
        "from pathlib import Path",
        "count=int(sys.argv[1])",
        "ephemeral_start=int(Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split()[0])",
        "handles=[]",
        "ports=[]",
        "try:",
        " for port in range(ephemeral_start-1,1023,-1):",
        "  if len(ports)>=count*2: break",
        "  handle=socket.socket(socket.AF_INET,socket.SOCK_STREAM)",
        "  try:",
        "   handle.bind(('0.0.0.0',port))",
        "  except OSError:",
        "   handle.close()",
        "   continue",
        "  handles.append(handle)",
        "  ports.append(port)",
        " if len(ports)!=count*2:",
        "  raise SystemExit('insufficient non-ephemeral ports')",
        " pairs=[ports[index:index+2] for index in range(0,len(ports),2)]",
        " print(json.dumps({'ephemeral_start':ephemeral_start,'pairs':pairs},separators=(',',':')))",
        "finally:",
        " for handle in handles: handle.close()",
    ])
    result = command_runner(
        name="remote_port_inventory",
        argv=_ssh_argv([REMOTE_PYTHON, "-c", script, str(count)]),
        stdout_path=None,
        env=execution_env,
    )
    if result.get("returncode") != 0:
        raise RuntimeError("remote port inventory failed")
    try:
        payload = json.loads(result["stdout"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("remote port inventory is invalid") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"ephemeral_start", "pairs"}
        or isinstance(payload["ephemeral_start"], bool)
        or not isinstance(payload["ephemeral_start"], int)
        or payload["ephemeral_start"] <= 1024
    ):
        raise ValueError("remote port inventory is invalid")
    ephemeral_start = payload["ephemeral_start"]
    rows = payload["pairs"]
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("remote port inventory count mismatch")
    if any(
        not isinstance(row, list)
        or len(row) != 2
        or any(
            isinstance(port, bool)
            or not isinstance(port, int)
            or port < 1024
            or port >= ephemeral_start
            for port in row
        )
        for row in rows
    ):
        raise ValueError("remote port inventory must be non-ephemeral")
    values = iter(tuple(row) if isinstance(row, list) else row for row in rows)
    return allocate_unique_port_pairs(
        count,
        allocator=lambda: next(values),
    )


def validate_worker_authorization(authorization):
    if not isinstance(authorization, WorkerAuthorization):
        raise ValueError("worker authorization is required")
    if (
        not isinstance(authorization.source_tree_sha256, str)
        or len(authorization.source_tree_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in authorization.source_tree_sha256
        )
    ):
        raise ValueError("worker authorization source tree mismatch")
    if (
        authorization.model_manifest_sha256
        != contract.MODEL_MANIFEST_SHA256
    ):
        raise ValueError("worker authorization model manifest mismatch")
    if (
        authorization.workload_manifest_sha256
        != contract.canonical_json_file_sha256(
            contract.workload_manifest_payload()
        )
    ):
        raise ValueError("worker authorization workload manifest mismatch")
    if (
        not isinstance(authorization.prerequisites_sha256, str)
        or len(authorization.prerequisites_sha256) != 64
    ):
        raise ValueError("worker authorization prerequisite SHA invalid")
    if (
        len(authorization.gpu_indices) != contract.WORLD_SIZE
        or len(set(authorization.gpu_indices)) != contract.WORLD_SIZE
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in authorization.gpu_indices
        )
    ):
        raise ValueError("worker authorization GPU identity mismatch")
    return authorization


def validate_worker_runtime_artifacts(runtime_artifacts):
    if not isinstance(runtime_artifacts, WorkerRuntimeArtifacts):
        raise ValueError("worker runtime artifacts are required")
    for name, value in asdict(runtime_artifacts).items():
        if (
            not isinstance(value, str)
            or not value
            or not PurePosixPath(value).is_absolute()
            or ".." in PurePosixPath(value).parts
        ):
            raise ValueError(
                f"worker runtime artifacts {name} is invalid"
            )
    return runtime_artifacts


def _canonical_case(case):
    fields = (
        "case_id",
        "policy",
        "workload",
        "phase",
        "repetition",
    )
    try:
        identity = tuple(getattr(case, name) for name in fields)
    except AttributeError as error:
        raise ValueError("benchmark case is not canonical") from error
    for candidate in contract.build_case_matrix():
        if tuple(getattr(candidate, name) for name in fields) == identity:
            return candidate
    raise ValueError("benchmark case is not canonical")


def build_worker_command(
    *,
    case,
    remote_source,
    case_output_dir,
    dist_port,
    master_port,
    authorization,
    runtime_artifacts=None,
):
    authorization = validate_worker_authorization(authorization)
    runtime_artifacts = validate_worker_runtime_artifacts(
        runtime_artifacts
    )
    case = _canonical_case(case)
    if dist_port == master_port:
        raise ValueError("worker ports must be distinct")
    gpu_csv = ",".join(
        str(index) for index in authorization.gpu_indices
    )
    argv = [
        REMOTE_PYTHON,
        "tools/qwen35_tp4_hybrid_prefix_benchmark_worker.py",
        "--policy",
        case.policy,
        "--workload",
        case.workload,
        "--phase",
        case.phase,
        "--repetition",
        str(case.repetition),
        "--output-dir",
        case_output_dir,
        "--prerequisites-sha256",
        authorization.prerequisites_sha256,
        "--source-tree-sha256",
        authorization.source_tree_sha256,
        "--model-manifest-sha256",
        authorization.model_manifest_sha256,
        "--model-dir",
        runtime_artifacts.model_dir,
        "--model-manifest",
        runtime_artifacts.model_manifest_path,
        "--correctness-prerequisites",
        runtime_artifacts.correctness_prerequisites_path,
        "--workload-manifest",
        runtime_artifacts.workload_manifest_path,
        "--workload-manifest-sha256",
        authorization.workload_manifest_sha256,
    ]
    return {
        "case_id": case.case_id,
        "policy": case.policy,
        "workload": case.workload,
        "phase": case.phase,
        "repetition": case.repetition,
        "dist_port": dist_port,
        "master_port": master_port,
        "cwd": remote_source,
        "env": {
            "CUDA_VISIBLE_DEVICES": gpu_csv,
            "TINYVLLM_DIST_PORT": str(dist_port),
            "MASTER_PORT": str(master_port),
            "PYTHONPATH": remote_source,
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "argv": argv,
    }


def build_case_commands(
    *,
    remote_source,
    remote_output,
    ports,
    authorization,
    runtime_artifacts,
):
    matrix = contract.build_case_matrix()
    if len(ports) != len(matrix):
        raise ValueError("worker port inventory mismatch")
    commands = []
    for case, (dist_port, master_port) in zip(matrix, ports):
        commands.append(build_worker_command(
            case=case,
            remote_source=remote_source,
            case_output_dir=(
                f"{remote_output}/cases/{case.case_id}"
            ),
            dist_port=dist_port,
            master_port=master_port,
            authorization=authorization,
            runtime_artifacts=runtime_artifacts,
        ))
    return commands


def build_authorized_launch_plan(
    *,
    run_tag,
    preflight,
    remote_model_dir,
    remote_model_manifest,
    remote_prerequisites,
    port_allocator=allocate_port_pair,
):
    run_tag = safe_run_tag(run_tag)
    if (
        not isinstance(preflight, dict)
        or preflight.get("classification") != "READY"
        or preflight.get("authorized") is not True
    ):
        raise ValueError("launch plan requires READY preflight")
    resource_policy = preflight.get(
        "resource_policy",
        STRICT_EXCLUSIVE_RESOURCE_POLICY,
    )
    maximum_gpu_utilization_percent = preflight.get(
        "maximum_gpu_utilization_percent"
    )
    if resource_policy not in {
        STRICT_EXCLUSIVE_RESOURCE_POLICY,
        SHARED_LOW_UTILIZATION_RESOURCE_POLICY,
    }:
        raise ValueError("launch plan resource policy is invalid")
    if resource_policy == SHARED_LOW_UTILIZATION_RESOURCE_POLICY:
        if (
            isinstance(maximum_gpu_utilization_percent, bool)
            or not isinstance(maximum_gpu_utilization_percent, int)
            or not 0 <= maximum_gpu_utilization_percent <= 100
        ):
            raise ValueError(
                "launch plan maximum GPU utilization is invalid"
            )
    elif maximum_gpu_utilization_percent is not None:
        raise ValueError(
            "exclusive launch plan cannot set utilization limit"
        )
    authorization = WorkerAuthorization.from_dict(
        preflight.get("worker_authorization")
    )
    source_bundle = preflight.get("source_bundle")
    if not isinstance(source_bundle, dict):
        raise ValueError("launch plan source bundle is invalid")
    source_tree_sha256 = _validate_sha256(
        source_bundle.get("source_tree_sha256"),
        "launch plan source tree",
    )
    if source_tree_sha256 != authorization.source_tree_sha256:
        raise ValueError("launch plan source authorization mismatch")
    tar_sha256 = _validate_sha256(
        source_bundle.get("tar_sha256"),
        "launch plan source tar",
    )
    tar_path = source_bundle.get("tar_path")
    if (
        not isinstance(tar_path, str)
        or not tar_path
        or not Path(tar_path).is_absolute()
    ):
        raise ValueError("launch plan local source tar is invalid")
    runtime_artifacts = validate_worker_runtime_artifacts(
        WorkerRuntimeArtifacts(
            model_dir=remote_model_dir,
            model_manifest_path=remote_model_manifest,
            correctness_prerequisites_path=remote_prerequisites,
            workload_manifest_path=(
                f"{REMOTE_ROOT}/{run_tag}/workload_manifest.json"
            ),
        )
    )
    remote_run = f"{REMOTE_ROOT}/{run_tag}"
    remote_source = f"{remote_run}/source"
    remote_output = f"{remote_run}/output"
    remote_cases = f"{remote_output}/cases"
    remote_logs = f"{remote_run}/logs"
    remote_assembly = f"{remote_run}/assembly"
    remote_artifact = f"{remote_run}/artifact"
    ports = allocate_unique_port_pairs(
        len(contract.build_case_matrix()),
        allocator=port_allocator,
    )
    case_commands = build_case_commands(
        remote_source=remote_source,
        remote_output=remote_output,
        ports=ports,
        authorization=authorization,
        runtime_artifacts=runtime_artifacts,
    )
    for row in case_commands:
        row["log_path"] = f"{remote_logs}/{row['case_id']}.log"
    selected_gpus = preflight.get("selected_gpus")
    if (
        not isinstance(selected_gpus, list)
        or len(selected_gpus) != contract.WORLD_SIZE
    ):
        raise ValueError("launch plan GPU assignments are invalid")
    gpu_assignments = []
    for rank, row in enumerate(selected_gpus):
        if not isinstance(row, dict):
            raise ValueError("launch plan GPU assignments are invalid")
        gpu_assignments.append({
            "rank": rank,
            "gpu_index": row.get("gpu_index"),
            "gpu_uuid": row.get("gpu_uuid"),
            "free_bytes": row.get("free_bytes"),
            "utilization_percent": row.get("utilization_percent"),
            "compute_processes": row.get("compute_processes"),
        })
    command_rows = [
        {
            name: row[name]
            for name in (
                "case_id",
                "policy",
                "workload",
                "phase",
                "repetition",
                "dist_port",
                "master_port",
            )
        }
        for row in case_commands
    ]
    worker_logs = {
        row["case_id"]: row["log_path"]
        for row in case_commands
    }
    assembly_metadata = {
        "source_manifest.json": {
            "schema_version": contract.SCHEMA_VERSION,
            "source_tree_sha256": source_tree_sha256,
            "model_manifest_sha256": (
                contract.MODEL_MANIFEST_SHA256
            ),
        },
        "environment.json": {
            "schema_version": contract.SCHEMA_VERSION,
            "world_size": contract.WORLD_SIZE,
            "python": REMOTE_PYTHON,
        },
        "gpu_assignments.json": {
            "schema_version": contract.SCHEMA_VERSION,
            "resource_policy": preflight.get("resource_policy"),
            "maximum_gpu_utilization_percent": preflight.get(
                "maximum_gpu_utilization_percent"
            ),
            "assignments": gpu_assignments,
        },
        "commands.json": {
            "schema_version": contract.SCHEMA_VERSION,
            "commands": command_rows,
        },
        "worker_logs.json": {
            "schema_version": contract.SCHEMA_VERSION,
            "worker_logs": worker_logs,
        },
    }
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "run_tag": run_tag,
        "worker_authorization": {
            **asdict(authorization),
            "gpu_indices": list(authorization.gpu_indices),
        },
        "local_source_tar": tar_path,
        "source_tar_sha256": tar_sha256,
        "remote_source_tar": (
            f"{REMOTE_ROOT}/{run_tag}-{tar_sha256}.tar"
        ),
        "remote_source": remote_source,
        "remote_output": remote_output,
        "remote_cases": remote_cases,
        "remote_logs": remote_logs,
        "remote_assembly": remote_assembly,
        "remote_artifact": remote_artifact,
        "remote_workload_manifest": (
            runtime_artifacts.workload_manifest_path
        ),
        "stage_command": build_remote_stage_command(
            run_tag=run_tag,
            source_tree_sha256=source_tree_sha256,
            tar_sha256=tar_sha256,
        ),
        "resource_policy": resource_policy,
        "maximum_gpu_utilization_percent": (
            maximum_gpu_utilization_percent
        ),
        "case_commands": case_commands,
        "assembly_metadata": assembly_metadata,
        "assembler_command": {
            "cwd": remote_source,
            "argv": [
                REMOTE_PYTHON,
                "tools/qwen35_tp4_hybrid_prefix_benchmark_assembler.py",
                "--output-dir",
                remote_artifact,
                "--cases-root",
                remote_cases,
                "--correctness-prerequisites",
                runtime_artifacts.correctness_prerequisites_path,
                "--workload-manifest",
                runtime_artifacts.workload_manifest_path,
                "--source-manifest",
                f"{remote_assembly}/source_manifest.json",
                "--environment",
                f"{remote_assembly}/environment.json",
                "--gpu-assignments",
                f"{remote_assembly}/gpu_assignments.json",
                "--commands",
                f"{remote_assembly}/commands.json",
                "--worker-logs",
                f"{remote_assembly}/worker_logs.json",
            ],
        },
    }


def _atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} SHA256 is invalid")
    return value


def _owned_source_files(repo_root, owned_paths):
    repo_root = Path(repo_root).resolve()
    if (
        not isinstance(owned_paths, (tuple, list))
        or not owned_paths
    ):
        raise ValueError("owned source inventory is required")
    files = {}

    def collect(path):
        if path.name == "__pycache__" and path.is_dir():
            return
        if path.suffix in {".pyc", ".pyo"} and path.is_file():
            return
        if path.is_symlink():
            raise ValueError("owned source path must not be a link")
        if path.is_file():
            relative = path.relative_to(repo_root).as_posix()
            files[relative] = path
            return
        if not path.is_dir():
            raise ValueError("owned source path is not regular")
        for child in sorted(path.iterdir(), key=lambda value: value.name):
            collect(child)

    for raw_path in owned_paths:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("owned source path is invalid")
        relative = PurePosixPath(raw_path)
        if (
            relative.is_absolute()
            or relative.as_posix() in {".", ""}
            or ".." in relative.parts
            or "\\" in raw_path
        ):
            raise ValueError("owned source path is unsafe")
        candidate = repo_root.joinpath(*relative.parts)
        if not candidate.exists():
            raise ValueError("owned source path is missing")
        try:
            candidate = candidate.resolve(strict=True)
            candidate.relative_to(repo_root)
        except ValueError as error:
            raise ValueError(
                "owned source path escapes repository"
            ) from error
        collect(candidate)
    if not files:
        raise ValueError("owned source inventory has no files")
    return [(name, files[name]) for name in sorted(files)]


def _source_tree_sha256(files):
    digest = hashlib.sha256()
    for relative, path in files:
        relative_bytes = relative.encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build_deterministic_source_bundle(
    *,
    repo_root,
    owned_paths,
    output_tar,
):
    repo_root = Path(repo_root).resolve()
    files = _owned_source_files(repo_root, owned_paths)
    output_tar = Path(output_tar).resolve()
    for _, source_path in files:
        if output_tar == source_path:
            raise ValueError("output tar overlaps owned source")
    for raw_path in owned_paths:
        owned_root = repo_root.joinpath(
            *PurePosixPath(raw_path).parts
        ).resolve(strict=True)
        if owned_root.is_dir():
            try:
                output_tar.relative_to(owned_root)
            except ValueError:
                pass
            else:
                raise ValueError("output tar is inside owned source")
    output_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(
        output_tar,
        mode="w",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        for relative, path in files:
            stat = path.stat()
            info = tarfile.TarInfo(relative)
            info.size = stat.st_size
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o644
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    return {
        "owned_files": [relative for relative, _ in files],
        "source_tree_sha256": _source_tree_sha256(files),
        "tar_sha256": _sha256(output_tar),
    }


def prepare_benchmark_source_bundle(
    *,
    repo_root=ROOT,
    output_dir,
):
    output_dir = Path(output_dir)
    output_tar = output_dir / "benchmark_source.tar"
    result = build_deterministic_source_bundle(
        repo_root=repo_root,
        owned_paths=BENCHMARK_OWNED_SOURCE_PATHS,
        output_tar=output_tar,
    )
    return {
        **result,
        "owned_paths": list(BENCHMARK_OWNED_SOURCE_PATHS),
        "tar_path": str(output_tar),
    }


def build_remote_stage_command(
    *,
    run_tag,
    source_tree_sha256,
    tar_sha256,
):
    run_tag = safe_run_tag(run_tag)
    source_tree_sha256 = _validate_sha256(
        source_tree_sha256,
        "source tree",
    )
    tar_sha256 = _validate_sha256(tar_sha256, "source tar")
    remote_run = f"{REMOTE_ROOT}/{run_tag}"
    remote_tar = f"{REMOTE_ROOT}/{run_tag}-{tar_sha256}.tar"
    remote_source = f"{remote_run}/source"
    remote_workload_manifest = (
        f"{remote_run}/workload_manifest.json"
    )
    workload_manifest_sha256 = (
        contract.canonical_json_file_sha256(
            contract.workload_manifest_payload()
        )
    )
    extract_script = "\n".join([
        "import sys,tarfile",
        "from pathlib import Path,PurePosixPath",
        "archive_path=Path(sys.argv[1])",
        "destination=Path(sys.argv[2])",
        "with tarfile.open(archive_path,'r') as archive:",
        " members=archive.getmembers()",
        " assert members, 'source tar is empty'",
        " for member in members:",
        "  path=PurePosixPath(member.name)",
        (
            "  assert not path.is_absolute() and '..' not in path.parts,"
            "'unsafe source tar member'"
        ),
        (
            "  assert member.isfile() and not member.issym() "
            "and not member.islnk(),"
            "'unsafe source tar member'"
        ),
        " archive.extractall(destination,members=members)",
    ])
    verify_script = "\n".join([
        "import hashlib,sys",
        "from pathlib import Path",
        "root=Path(sys.argv[1])",
        "expected=sys.argv[2]",
        "digest=hashlib.sha256()",
        "files=sorted(path for path in root.rglob('*') if path.is_file())",
        "assert files, 'source tree is empty'",
        "for path in files:",
        " relative=path.relative_to(root).as_posix().encode('utf-8')",
        " digest.update(len(relative).to_bytes(8,'big'))",
        " digest.update(relative)",
        " with path.open('rb') as handle:",
        "  for chunk in iter(lambda:handle.read(1024*1024),b''):",
        "   digest.update(chunk)",
        (
            "assert digest.hexdigest()==expected,"
            "'source tree mismatch'"
        ),
    ])
    workload_script = "\n".join([
        "import hashlib,importlib.util,sys",
        "from pathlib import Path",
        "source=Path(sys.argv[1])",
        "destination=Path(sys.argv[2])",
        "expected=sys.argv[3]",
        (
            "path=source/'tools/"
            "qwen35_tp4_hybrid_prefix_benchmark_contract.py'"
        ),
        (
            "spec=importlib.util.spec_from_file_location("
            "'qwen35_benchmark_staged_contract',path)"
        ),
        "module=importlib.util.module_from_spec(spec)",
        "sys.modules[spec.name]=module",
        "spec.loader.exec_module(module)",
        (
            "data=module.canonical_json_bytes("
            "module.workload_manifest_payload())+b'\\n'"
        ),
        "assert hashlib.sha256(data).hexdigest()==expected",
        "destination.write_bytes(data)",
        "assert hashlib.sha256(destination.read_bytes()).hexdigest()==expected",
    ])
    script = " && ".join([
        "set -eu",
        f"test ! -e {shlex.quote(remote_run)}",
        (
            f"test \"$(sha256sum {shlex.quote(remote_tar)} "
            "| awk '{print $1}')\" = "
            f"{shlex.quote(tar_sha256)}"
        ),
        f"mkdir {shlex.quote(remote_run)}",
        f"mkdir {shlex.quote(remote_source)}",
        (
            f"{shlex.quote(REMOTE_PYTHON)} -c "
            f"{shlex.quote(extract_script)} "
            f"{shlex.quote(remote_tar)} "
            f"{shlex.quote(remote_source)}"
        ),
        (
            f"{shlex.quote(REMOTE_PYTHON)} -c "
            f"{shlex.quote(verify_script)} "
            f"{shlex.quote(remote_source)} "
            f"{shlex.quote(source_tree_sha256)}"
        ),
        (
            f"{shlex.quote(REMOTE_PYTHON)} -c "
            f"{shlex.quote(workload_script)} "
            f"{shlex.quote(remote_source)} "
            f"{shlex.quote(remote_workload_manifest)} "
            f"{shlex.quote(workload_manifest_sha256)}"
        ),
        (
            f"printf '%s\\n' {shlex.quote(source_tree_sha256)} > "
            f"{shlex.quote(remote_run + '/source_tree_sha256.txt')}"
        ),
    ])
    return ["bash", "-lc", script]


def validate_download_inventory(
    *,
    top_level_files,
    top_level_directories,
):
    if (
        not isinstance(top_level_files, (tuple, list))
        or not isinstance(top_level_directories, (tuple, list))
        or any(not isinstance(value, str) for value in top_level_files)
        or any(
            not isinstance(value, str)
            for value in top_level_directories
        )
        or len(top_level_files) != len(set(top_level_files))
        or len(top_level_directories)
        != len(set(top_level_directories))
        or set(top_level_files) != set(contract.TOP_LEVEL_ARTIFACTS)
        or set(top_level_directories)
        != set(contract.NESTED_ARTIFACT_DIRECTORIES)
    ):
        raise ValueError("download inventory is not canonical")
    return {
        "top_level_files": sorted(top_level_files),
        "top_level_directories": sorted(top_level_directories),
    }


def _select_tp4_gpu_resources(rows):
    if not isinstance(rows, list):
        raise ValueError("GPU resource rows must be a list")
    candidates = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("GPU resource row is invalid")
        gpu_index = row.get("gpu_index")
        gpu_uuid = row.get("gpu_uuid")
        free_bytes = row.get("free_bytes")
        compute_processes = row.get("compute_processes")
        if (
            isinstance(gpu_index, bool)
            or not isinstance(gpu_index, int)
            or gpu_index < 0
            or not isinstance(gpu_uuid, str)
            or not gpu_uuid.startswith("GPU-")
            or isinstance(free_bytes, bool)
            or not isinstance(free_bytes, int)
            or not isinstance(compute_processes, list)
        ):
            raise ValueError("GPU resource row is invalid")
        if (
            free_bytes < MIN_GPU_FREE_BYTES
            or compute_processes != []
        ):
            continue
        candidates.append(dict(row))
    by_index = {row["gpu_index"]: row for row in candidates}
    if any(index not in by_index for index in REQUIRED_GPU_INDICES):
        raise ValueError(
            "eligible GPUs 2,4,5,6 are required"
        )
    selected = [by_index[index] for index in REQUIRED_GPU_INDICES]
    if (
        len({row["gpu_index"] for row in selected})
        != contract.WORLD_SIZE
        or len({row["gpu_uuid"] for row in selected})
        != contract.WORLD_SIZE
    ):
        raise ValueError("selected GPU identities must be unique")
    return selected


def _default_remote_query():
    script = "\n".join([
        "import importlib.util,json,sys",
        "from pathlib import Path",
        (
            "path=Path('/data00/home/sitian/sitian-workspace01/tllm/"
            "qwen35-tp4-root-logit-tests/"
            "qwen35-tp4-source-prep-20260729-010400/source/"
            "tools/qwen35_tp4_real_root_logit_correctness_preflight.py')"
        ),
        (
            "spec=importlib.util.spec_from_file_location("
            "'qwen35_benchmark_gpu_query',path)"
        ),
        "module=importlib.util.module_from_spec(spec)",
        "sys.modules[spec.name]=module",
        "spec.loader.exec_module(module)",
        "print(json.dumps({'gpus':list(module._query_tp4_gpu_resources())},sort_keys=True,separators=(',',':')))",
    ])
    result = subprocess.run(
        _ssh_argv([REMOTE_PYTHON, "-c", script]),
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            detail or "remote GPU resource query failed"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("remote GPU resource JSON is invalid") from error
    return payload


def run_preflight(
    *,
    run_tag,
    prerequisites_path,
    output_root=OUTPUT_ROOT,
    remote_query=_default_remote_query,
    resource_selector=_select_tp4_gpu_resources,
    resource_policy=STRICT_EXCLUSIVE_RESOURCE_POLICY,
    maximum_gpu_utilization_percent=None,
    benchmark_source_tree_sha256=None,
    source_bundle_builder=prepare_benchmark_source_bundle,
):
    run_tag = safe_run_tag(run_tag)
    output_dir = Path(output_root) / run_tag
    if output_dir.exists():
        raise ValueError("local run tag already exists")
    output_dir.mkdir(parents=True)
    prerequisite_status = contract.validate_prerequisites(
        prerequisites_path
    )
    if not prerequisite_status.authorized:
        result = {
            "schema_version": contract.SCHEMA_VERSION,
            "run_tag": run_tag,
            "classification": "BLOCKED_CORRECTNESS",
            "authorized": False,
            "reasons": list(prerequisite_status.reasons),
            "remote_query_executed": False,
            "remote_path_created": False,
        }
        _atomic_write_json(
            output_dir / "benchmark_preflight.json",
            result,
        )
        return result

    prerequisites_path = Path(prerequisites_path)
    prerequisites_sha256 = _sha256(prerequisites_path)
    if benchmark_source_tree_sha256 is None:
        source_bundle = source_bundle_builder(output_dir=output_dir)
        if not isinstance(source_bundle, dict):
            raise ValueError("benchmark source bundle is invalid")
        benchmark_source_tree_sha256 = source_bundle.get(
            "source_tree_sha256"
        )
    else:
        source_bundle = {
            "source_tree_sha256": benchmark_source_tree_sha256,
        }
    benchmark_source_tree_sha256 = _validate_sha256(
        benchmark_source_tree_sha256,
        "benchmark source tree",
    )
    if not callable(resource_selector):
        raise ValueError("resource selector must be callable")
    if resource_policy not in {
        STRICT_EXCLUSIVE_RESOURCE_POLICY,
        SHARED_LOW_UTILIZATION_RESOURCE_POLICY,
    }:
        raise ValueError("resource policy is invalid")
    if resource_policy == SHARED_LOW_UTILIZATION_RESOURCE_POLICY:
        if (
            isinstance(maximum_gpu_utilization_percent, bool)
            or not isinstance(maximum_gpu_utilization_percent, int)
            or not 0 <= maximum_gpu_utilization_percent <= 100
        ):
            raise ValueError("maximum GPU utilization is invalid")
    elif maximum_gpu_utilization_percent is not None:
        raise ValueError(
            "exclusive resource policy cannot set utilization limit"
        )
    payload = remote_query()
    rows = payload.get("gpus") if isinstance(payload, dict) else None
    try:
        selected = resource_selector(rows)
    except ValueError as error:
        result = {
            "schema_version": contract.SCHEMA_VERSION,
            "run_tag": run_tag,
            "classification": "BLOCKED_RESOURCES",
            "authorized": False,
            "reasons": [str(error)],
            "remote_query_executed": True,
            "remote_path_created": False,
            "gpu_rows": rows if isinstance(rows, list) else [],
            "source_bundle": source_bundle,
        }
        _atomic_write_json(
            output_dir / "benchmark_preflight.json",
            result,
        )
        return result

    authorization = WorkerAuthorization(
        prerequisites_sha256=prerequisites_sha256,
        source_tree_sha256=benchmark_source_tree_sha256,
        model_manifest_sha256=contract.MODEL_MANIFEST_SHA256,
        workload_manifest_sha256=(
            contract.canonical_json_file_sha256(
                contract.workload_manifest_payload()
            )
        ),
        gpu_indices=tuple(
            row["gpu_index"] for row in selected
        ),
    )
    validate_worker_authorization(authorization)
    result = {
        "schema_version": contract.SCHEMA_VERSION,
        "run_tag": run_tag,
        "classification": "READY",
        "authorized": True,
        "remote_query_executed": True,
        "remote_path_created": False,
        "source_bundle": source_bundle,
        "resource_policy": resource_policy,
        "maximum_gpu_utilization_percent": (
            maximum_gpu_utilization_percent
        ),
        "selected_gpus": selected,
        "worker_authorization": {
            **asdict(authorization),
            "gpu_indices": list(authorization.gpu_indices),
        },
    }
    _atomic_write_json(
        output_dir / "benchmark_preflight.json",
        result,
    )
    return result


def safe_download_member(name, *, is_file, is_link):
    if not isinstance(name, str) or not name:
        raise ValueError("download member path is invalid")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or is_link
        or not is_file
    ):
        raise ValueError("download member is unsafe")
    return path.as_posix()


def _verify_only(run_tag, output_root):
    run_dir = Path(output_root) / safe_run_tag(run_tag)
    verifier_path = (
        ROOT / "tools/verify_qwen35_tp4_hybrid_prefix_benchmark.py"
    )
    spec = __import__("importlib.util").util.spec_from_file_location(
        "qwen35_benchmark_verifier_for_runner",
        verifier_path,
    )
    module = __import__("importlib.util").util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.verify_run(run_dir)


def _load_tool_module(module_name, filename):
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = ROOT / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def execute_benchmark_launch(
    *,
    mode,
    run_tag,
    prerequisites_path,
    local_model_manifest,
    remote_model_dir,
    remote_model_manifest,
    authorization_nonce,
    output_root=OUTPUT_ROOT,
    preflight_runner=run_preflight,
    launch_plan_builder=build_authorized_launch_plan,
    plan_module=None,
    authorization_module=None,
    executor_module=None,
    command_runner=None,
    resource_policy=STRICT_EXCLUSIVE_RESOURCE_POLICY,
    maximum_gpu_utilization_percent=None,
):
    if mode not in {"smoke", "canonical"}:
        raise ValueError("benchmark launch mode is invalid")
    run_tag = safe_run_tag(run_tag)
    output_root = Path(output_root)
    run_dir = output_root / run_tag
    if run_dir.exists():
        raise ValueError("local run tag already exists")
    preflight = preflight_runner(
        run_tag=run_tag,
        prerequisites_path=Path(prerequisites_path),
        output_root=output_root,
        resource_policy=resource_policy,
        maximum_gpu_utilization_percent=(
            maximum_gpu_utilization_percent
        ),
    )
    if preflight.get("classification") != "READY":
        return preflight
    run_dir.mkdir(parents=True, exist_ok=True)
    if command_runner is None:
        subprocess_adapter = _load_tool_module(
            "qwen35_tp4_engine_remote_subprocess_adapter",
            "qwen35_tp4_engine_remote_subprocess_adapter.py",
        )
        command_runner = subprocess_adapter.run_command
    remote_prerequisites = (
        f"{REMOTE_ROOT}/{run_tag}/prerequisites/"
        "correctness_prerequisites.json"
    )
    launch_plan_kwargs = {
        "run_tag": run_tag,
        "preflight": preflight,
        "remote_model_dir": remote_model_dir,
        "remote_model_manifest": remote_model_manifest,
        "remote_prerequisites": remote_prerequisites,
    }
    if launch_plan_builder is build_authorized_launch_plan:
        port_pairs = allocate_remote_port_pairs(
            len(contract.build_case_matrix()),
            command_runner=command_runner,
            execution_env=(
                {
                    "KRB5CCNAME": (
                        "FILE:/Users/bytedance/krb5cc_sitian"
                    ),
                }
            ),
        )
        port_pair_iterator = iter(port_pairs)
        launch_plan_kwargs["port_allocator"] = (
            lambda: next(port_pair_iterator)
        )
    launch_plan = launch_plan_builder(
        **launch_plan_kwargs,
    )
    if plan_module is None:
        plan_module = _load_tool_module(
            (
                "qwen35_tp4_hybrid_prefix_benchmark_"
                "remote_execution_plan"
            ),
            (
                "qwen35_tp4_hybrid_prefix_benchmark_"
                "remote_execution_plan.py"
            ),
        )
    if authorization_module is None:
        authorization_module = _load_tool_module(
            (
                "qwen35_tp4_hybrid_prefix_benchmark_"
                "remote_execution_authorization"
            ),
            (
                "qwen35_tp4_hybrid_prefix_benchmark_"
                "remote_execution_authorization.py"
            ),
        )
    if executor_module is None:
        executor_module = _load_tool_module(
            (
                "qwen35_tp4_hybrid_prefix_benchmark_"
                "remote_execution_executor"
            ),
            (
                "qwen35_tp4_hybrid_prefix_benchmark_"
                "remote_execution_executor.py"
            ),
        )
    plan_dir = run_dir / "plan"
    plan = plan_module.build_remote_execution_plan(
        launch_plan=launch_plan,
        output_dir=plan_dir,
        local_prerequisites=Path(prerequisites_path),
        local_model_manifest=Path(local_model_manifest),
    )
    plan_path = plan_dir / plan_module.PLAN_NAME
    runtime_dir = run_dir / "runtime"
    authorization_path = runtime_dir / "authorization.json"
    authorization_module.produce_authorization(
        plan=plan,
        output_path=authorization_path,
        nonce=authorization_nonce,
    )
    return executor_module.execute_verified_plan_file(
        plan_path=plan_path,
        authorization_path=authorization_path,
        consumed_authorization_path=(
            runtime_dir / "consumed_authorization.json"
        ),
        output_path=runtime_dir / "execution_receipt.json",
        failure_path=runtime_dir / "execution_failure.json",
        command_runner=command_runner,
        plan_verifier=plan_module.verify_remote_execution_plan,
        execution_env=executor_module.REQUIRED_EXECUTION_ENV,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument(
        "--prerequisites",
        default=(
            ROOT
            / "experiments/qwen35_hybrid_state/"
            "qwen35-tp4-performance-correctness-prerequisites.json"
        ),
        type=Path,
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--local-model-manifest", type=Path)
    parser.add_argument("--remote-model-dir")
    parser.add_argument("--remote-model-manifest")
    parser.add_argument("--authorization-nonce")
    args = parser.parse_args(argv)
    if args.mode == "verify-only":
        result = _verify_only(args.run_tag, args.output_root)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.mode in {"download-only"}:
        raise SystemExit(
            "download-only is not enabled before correctness prerequisites"
        )
    if args.mode in {"smoke", "canonical"}:
        required = {
            "--local-model-manifest": args.local_model_manifest,
            "--remote-model-dir": args.remote_model_dir,
            "--remote-model-manifest": args.remote_model_manifest,
            "--authorization-nonce": args.authorization_nonce,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(
                "benchmark launch requires " + ", ".join(missing)
            )
        result = execute_benchmark_launch(
            mode=args.mode,
            run_tag=args.run_tag,
            prerequisites_path=args.prerequisites,
            local_model_manifest=args.local_model_manifest,
            remote_model_dir=args.remote_model_dir,
            remote_model_manifest=args.remote_model_manifest,
            authorization_nonce=args.authorization_nonce,
            output_root=args.output_root,
        )
    else:
        result = run_preflight(
            run_tag=args.run_tag,
            prerequisites_path=args.prerequisites,
            output_root=args.output_root,
        )
    print(json.dumps(result, sort_keys=True))
    if result["classification"] == "BLOCKED_CORRECTNESS":
        return 2
    if result["classification"] == "BLOCKED_RESOURCES":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
