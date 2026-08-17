from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import sys
import tarfile
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


source_runner = _load_module(
    "run_qwen35_tp4_hybrid_prefix_benchmark_remote",
    "run_qwen35_tp4_hybrid_prefix_benchmark_remote.py",
)
engine_plan = _load_module(
    "qwen35_tp4_engine_remote_execution_plan",
    "qwen35_tp4_engine_remote_execution_plan.py",
)
contract = _load_module(
    "qwen35_tp4_hybrid_prefix_benchmark_contract_for_remote_plan",
    "qwen35_tp4_hybrid_prefix_benchmark_contract.py",
)


SCHEMA_VERSION = (
    "qwen35.tp4-hybrid-prefix-benchmark-remote-execution-plan.v1"
)
BENCHMARK_SCHEMA_VERSION = contract.SCHEMA_VERSION
SSH_TARGET = source_runner.SSH_TARGET
REMOTE_PYTHON = source_runner.REMOTE_PYTHON
MIN_GPU_FREE_BYTES = contract.MIN_GPU_FREE_BYTES
PLAN_NAME = "remote_execution_plan.json"
SOURCE_TAR_NAME = "benchmark_source.tar"
PREREQUISITES_NAME = "correctness_prerequisites.json"
PREREQUISITES_TAR_NAME = "correctness_prerequisites.tar"
MODEL_MANIFEST_NAME = "model_manifest.json"
LOCAL_VERIFIER_SOURCE_NAME = "local_verifier_source"
COMMAND_ORDER = [
    "reserve_remote",
    "upload",
    "stage",
    "resource_guard",
    "workers",
    "assembly",
    "remote_verify",
    "final_resource_guard",
    "package_download",
    "safe_extract",
    "local_verify",
]
COMPLETION_MARKER = "QWEN35_TP4_BENCHMARK_WORKER_COMPLETE"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path, label):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _regular_file(path, label):
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file")
    return path


def _source_tree_sha256_from_tar(path):
    digest = hashlib.sha256()
    try:
        with tarfile.open(path, "r:") as handle:
            members = handle.getmembers()
            names = [member.name for member in members]
            if names != sorted(names) or len(names) != len(set(names)):
                raise ValueError("source tar inventory mismatch")
            for member in members:
                name = member.name
                pure = PurePosixPath(name)
                if (
                    pure.is_absolute()
                    or ".." in pure.parts
                    or not member.isfile()
                    or member.issym()
                    or member.islnk()
                ):
                    raise ValueError("unsafe source tar member")
                encoded = name.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
                source = handle.extractfile(member)
                if source is None:
                    raise ValueError("source tar member is unreadable")
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
    except (OSError, tarfile.TarError) as error:
        raise ValueError("source tar is invalid") from error
    return digest.hexdigest()


def _prerequisite_files(prerequisite_path):
    prerequisite_path = Path(prerequisite_path).resolve()
    root = prerequisite_path.parent
    files = [(PREREQUISITES_NAME, prerequisite_path)]
    nested = root / "prerequisites"
    if nested.exists():
        if not nested.is_dir() or nested.is_symlink():
            raise ValueError("prerequisite nested directory is invalid")
        for path in sorted(nested.rglob("*")):
            if path.is_symlink() or (not path.is_file() and not path.is_dir()):
                raise ValueError("prerequisite bundle member is invalid")
            if path.is_file():
                files.append((path.relative_to(root).as_posix(), path))
    return files


def _write_prerequisite_tar(path, files):
    with tarfile.open(path, "w:") as handle:
        for name, source in files:
            info = tarfile.TarInfo(name)
            info.size = source.stat().st_size
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with source.open("rb") as stream:
                handle.addfile(info, stream)


def _prerequisite_tar_inventory(path, prerequisite_sha256):
    try:
        with tarfile.open(path, "r:") as handle:
            members = handle.getmembers()
            names = [member.name for member in members]
            if names != sorted(names) or len(names) != len(set(names)):
                raise ValueError("prerequisite tar inventory mismatch")
            for member in members:
                pure = PurePosixPath(member.name)
                if (
                    pure.is_absolute()
                    or ".." in pure.parts
                    or not member.isfile()
                    or member.issym()
                    or member.islnk()
                ):
                    raise ValueError(
                        "unsafe prerequisite tar member"
                    )
            try:
                prerequisite_member = members[
                    names.index(PREREQUISITES_NAME)
                ]
            except ValueError as error:
                raise ValueError(
                    "prerequisite tar inventory mismatch"
                ) from error
            source = handle.extractfile(prerequisite_member)
            if source is None:
                raise ValueError(
                    "prerequisite tar member is unreadable"
                )
            digest = hashlib.sha256()
            for chunk in iter(
                lambda: source.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)
            if digest.hexdigest() != prerequisite_sha256:
                raise ValueError(
                    "prerequisite tar input identity mismatch"
                )
    except (OSError, tarfile.TarError) as error:
        raise ValueError("prerequisite tar is invalid") from error
    return names


def _remote_prerequisite_path(case_commands):
    paths = set()
    for row in case_commands:
        argv = row.get("argv")
        if not isinstance(argv, list):
            raise ValueError("worker command argv is invalid")
        try:
            index = argv.index("--correctness-prerequisites")
            path = argv[index + 1]
        except (ValueError, IndexError) as error:
            raise ValueError(
                "worker prerequisite path is missing"
            ) from error
        paths.add(_safe_remote(path, "worker prerequisite path"))
    if len(paths) != 1:
        raise ValueError("worker prerequisite path drifted")
    return paths.pop()


def _prerequisite_stage_command(
    *,
    remote_tar,
    remote_prerequisite,
    tar_sha256,
    prerequisite_sha256,
    owned_files,
):
    script = "\n".join([
        "import hashlib,json,sys,tarfile",
        "from pathlib import Path,PurePosixPath",
        "archive=Path(sys.argv[1])",
        "destination=Path(sys.argv[2])",
        "expected_tar=sys.argv[3]",
        "expected_prerequisite=sys.argv[4]",
        "expected_names=json.loads(sys.argv[5])",
        "if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected_tar:",
        " raise SystemExit('prerequisite bundle tar SHA mismatch')",
        "with tarfile.open(archive,'r:') as handle:",
        " members=handle.getmembers()",
        " names=[member.name for member in members]",
        " if names!=expected_names:",
        "  raise SystemExit('prerequisite bundle inventory mismatch')",
        " for member in members:",
        "  path=PurePosixPath(member.name)",
        "  if path.is_absolute() or '..' in path.parts or not member.isfile() or member.issym() or member.islnk():",
        "   raise SystemExit('unsafe prerequisite bundle member')",
        " if (destination/'correctness_prerequisites.json').exists():",
        "  raise SystemExit('remote prerequisite already exists')",
        " handle.extractall(destination,members=members)",
        "if hashlib.sha256((destination/'correctness_prerequisites.json').read_bytes()).hexdigest()!=expected_prerequisite:",
        " raise SystemExit('remote prerequisite SHA mismatch')",
    ])
    return (
        f"{shlex.quote(REMOTE_PYTHON)} -c {shlex.quote(script)} "
        f"{shlex.quote(remote_tar)} "
        f"{shlex.quote(str(PurePosixPath(remote_prerequisite).parent))} "
        f"{shlex.quote(tar_sha256)} "
        f"{shlex.quote(prerequisite_sha256)} "
        f"{shlex.quote(json.dumps(owned_files, separators=(',', ':')))}"
    )


def _safe_remote(value, label):
    if (
        not isinstance(value, str)
        or not value
        or not PurePosixPath(value).is_absolute()
        or ".." in PurePosixPath(value).parts
    ):
        raise ValueError(f"{label} is not a safe remote path")
    return value


def _canonical_metadata_files(output_dir, metadata):
    root = output_dir / "assembly_metadata"
    root.mkdir()
    result = {}
    for name in sorted(metadata):
        if name not in {
            "source_manifest.json",
            "environment.json",
            "gpu_assignments.json",
            "commands.json",
            "worker_logs.json",
        }:
            raise ValueError("assembly metadata inventory is invalid")
        path = root / name
        _write_json(path, metadata[name])
        result[name] = str(path)
    if len(result) != 5:
        raise ValueError("assembly metadata inventory is incomplete")
    return result


def _metadata_stage_commands(remote_assembly, metadata):
    commands = [f"mkdir {shlex.quote(remote_assembly)}"]
    for name, payload in sorted(metadata.items()):
        encoded = base64.b64encode(
            (
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        ).decode("ascii")
        remote_path = f"{remote_assembly}/{name}"
        script = (
            "import base64,sys;"
            "open(sys.argv[1],'wb').write(base64.b64decode(sys.argv[2]))"
        )
        commands.append(
            f"{shlex.quote(REMOTE_PYTHON)} -c "
            f"{shlex.quote(script)} "
            f"{shlex.quote(remote_path)} {shlex.quote(encoded)}"
        )
    return commands


def _worker_shell(case_commands):
    commands = ["set -eu"]
    expected = []
    for row in case_commands:
        case_id = row["case_id"]
        expected.append(case_id)
        env = " ".join(
            f"{name}={shlex.quote(value)}"
            for name, value in sorted(row["env"].items())
        )
        command = shlex.join([str(value) for value in row["argv"]])
        log_path = shlex.quote(row["log_path"])
        cwd = shlex.quote(row["cwd"])
        commands.extend([
            f"mkdir -p {shlex.quote(str(PurePosixPath(row['log_path']).parent))}",
            f"(cd {cwd} && env {env} {command}) > {log_path} 2>&1",
            (
                f"test \"$(grep -Fxc {shlex.quote(COMPLETION_MARKER)} "
                f"{log_path})\" -eq 1"
            ),
        ])
    completion = json.dumps({
        "classification": "COMPLETE",
        "case_ids": expected,
    }, sort_keys=True, separators=(",", ":"))
    commands.append(f"printf '%s\\n' {shlex.quote(completion)}")
    return " && ".join(commands)


def _compressed_remote_shell_command(shell, remote_run):
    payload = base64.b64encode(
        gzip.compress(shell.encode("utf-8"), mtime=0)
    ).decode("ascii")
    decoder = (
        "import base64,gzip,subprocess,sys;"
        "script=gzip.decompress(base64.b64decode(sys.argv[1],"
        "validate=True));"
        "raise SystemExit(subprocess.run(['bash','-s'],"
        "input=script).returncode)"
    )
    return [
        REMOTE_PYTHON,
        "-c",
        decoder,
        payload,
        remote_run,
    ]


def _verification_script():
    return "\n".join([
        "import hashlib,importlib.util,json,sys",
        "from pathlib import Path",
        "source=Path(sys.argv[1])",
        "artifact=Path(sys.argv[2])",
        "verifier=source/'tools/verify_qwen35_tp4_hybrid_prefix_benchmark.py'",
        "sys.path.insert(0,str(verifier.parent))",
        "spec=importlib.util.spec_from_file_location('benchmark_verifier',verifier)",
        "module=importlib.util.module_from_spec(spec)",
        "sys.modules[spec.name]=module",
        "spec.loader.exec_module(module)",
        "result=module.verify_run(artifact)",
        "source_manifest=json.loads((artifact/'source_manifest.json').read_text())",
        "result.update({",
        " 'source_tree_sha256':source_manifest['source_tree_sha256'],",
        " 'model_manifest_sha256':source_manifest['model_manifest_sha256'],",
        " 'workload_manifest_sha256':hashlib.sha256((artifact/'workload_manifest.json').read_bytes()).hexdigest(),",
        " 'correctness_prerequisites_sha256':hashlib.sha256((artifact/'correctness_prerequisites.json').read_bytes()).hexdigest(),",
        "})",
        "print(json.dumps(result,sort_keys=True,separators=(',',':')))",
    ])


def _local_verification_script():
    return "\n".join([
        "import hashlib,importlib.util,json,sys,tarfile",
        "from pathlib import Path,PurePosixPath",
        "archive=Path(sys.argv[1])",
        "expected_tar=sys.argv[2]",
        "expected_tree=sys.argv[3]",
        "source=Path(sys.argv[4])",
        "artifact=Path(sys.argv[5])",
        "if source.exists(): raise SystemExit('local verifier source exists')",
        "if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected_tar:",
        " raise SystemExit('local verifier source tar mismatch')",
        "with tarfile.open(archive,'r:') as handle:",
        " members=handle.getmembers()",
        " names=[member.name for member in members]",
        " if names!=sorted(names) or len(names)!=len(set(names)):",
        "  raise SystemExit('local verifier source inventory mismatch')",
        " for member in members:",
        "  path=PurePosixPath(member.name)",
        "  if path.is_absolute() or '..' in path.parts or not member.isfile() or member.issym() or member.islnk():",
        "   raise SystemExit('unsafe local verifier source member')",
        " source.mkdir()",
        " handle.extractall(source,members=members)",
        "digest=hashlib.sha256()",
        "for name in names:",
        " path=source.joinpath(*PurePosixPath(name).parts)",
        " encoded=name.encode('utf-8')",
        " digest.update(len(encoded).to_bytes(8,'big'))",
        " digest.update(encoded)",
        " with path.open('rb') as handle:",
        "  for chunk in iter(lambda:handle.read(1024*1024),b''):",
        "   digest.update(chunk)",
        "if digest.hexdigest()!=expected_tree:",
        " raise SystemExit('local verifier source tree mismatch')",
        "verifier=source/'tools/verify_qwen35_tp4_hybrid_prefix_benchmark.py'",
        "sys.path.insert(0,str(verifier.parent))",
        "spec=importlib.util.spec_from_file_location('benchmark_verifier',verifier)",
        "module=importlib.util.module_from_spec(spec)",
        "sys.modules[spec.name]=module",
        "spec.loader.exec_module(module)",
        "result=module.verify_run(artifact)",
        "source_manifest=json.loads((artifact/'source_manifest.json').read_text())",
        "result.update({",
        " 'source_tree_sha256':source_manifest['source_tree_sha256'],",
        " 'model_manifest_sha256':source_manifest['model_manifest_sha256'],",
        " 'workload_manifest_sha256':hashlib.sha256((artifact/'workload_manifest.json').read_bytes()).hexdigest(),",
        " 'correctness_prerequisites_sha256':hashlib.sha256((artifact/'correctness_prerequisites.json').read_bytes()).hexdigest(),",
        "})",
        "print(json.dumps(result,sort_keys=True,separators=(',',':')))",
    ])


def _extract_command(local_tar, destination):
    script = "\n".join([
        "import sys,tarfile",
        "from pathlib import Path,PurePosixPath",
        "archive=Path(sys.argv[1]); destination=Path(sys.argv[2])",
        "if destination.exists(): raise SystemExit('download destination exists')",
        "with tarfile.open(archive,'r:') as handle:",
        " members=handle.getmembers()",
        " roots=sorted({PurePosixPath(member.name).parts[0] for member in members})",
        " if roots!=['artifact']: raise SystemExit('benchmark package inventory mismatch')",
        " for member in members:",
        "  path=PurePosixPath(member.name)",
        "  if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk():",
        "   raise SystemExit('unsafe benchmark package member')",
        " destination.mkdir()",
        " handle.extractall(destination,members=members)",
    ])
    return [sys.executable, "-c", script, str(local_tar), str(destination)]


def _commands(launch_plan, local_inputs, output_dir):
    worker = launch_plan["worker_authorization"]
    gpu_indices = worker["gpu_indices"]
    resource_policy = launch_plan["resource_policy"]
    maximum_gpu_utilization_percent = launch_plan[
        "maximum_gpu_utilization_percent"
    ]
    remote_run = str(PurePosixPath(launch_plan["remote_source"]).parent)
    remote_tar = launch_plan["remote_source_tar"]
    local_tar = Path(local_inputs["source_tar"])
    local_prerequisites_tar = Path(local_inputs["prerequisites_tar"])
    remote_prerequisites = _remote_prerequisite_path(
        launch_plan["case_commands"]
    )
    remote_prerequisites_tar = (
        f"{remote_run}-correctness-prerequisites.tar"
    )
    remote_assembly = launch_plan["remote_assembly"]
    remote_root = str(PurePosixPath(remote_run).parent)
    reserve = " && ".join([
        "set -eu",
        f"mkdir -p {shlex.quote(remote_root)}",
        f"test ! -e {shlex.quote(remote_run)}",
        f"test ! -e {shlex.quote(remote_tar)}",
        f"test ! -e {shlex.quote(remote_prerequisites_tar)}",
    ])
    stage_shell = " && ".join([
        "set -eu",
        shlex.join(launch_plan["stage_command"]),
        _prerequisite_stage_command(
            remote_tar=remote_prerequisites_tar,
            remote_prerequisite=remote_prerequisites,
            tar_sha256=local_inputs["prerequisites_tar_sha256"],
            prerequisite_sha256=local_inputs["prerequisites_sha256"],
            owned_files=local_inputs["prerequisites_owned_files"],
        ),
        f"mkdir {shlex.quote(launch_plan['remote_output'])}",
        f"mkdir {shlex.quote(launch_plan['remote_cases'])}",
        f"mkdir {shlex.quote(launch_plan['remote_logs'])}",
        *_metadata_stage_commands(
            remote_assembly,
            launch_plan["assembly_metadata"],
        ),
    ])
    if resource_policy == "strict-exclusive":
        resource_argv = engine_plan._resource_guard_command(
            gpu_indices
        )
    else:
        resource_argv = (
            engine_plan._shared_low_utilization_resource_guard_command(
                gpu_indices,
                maximum_gpu_utilization_percent,
            )
        )
    resource = {
        "argv": engine_plan._ssh(resource_argv),
        "gpu_indices": list(gpu_indices),
        "minimum_free_bytes_per_gpu": MIN_GPU_FREE_BYTES,
        "requires_no_active_compute_processes": (
            resource_policy == "strict-exclusive"
        ),
        "resource_policy": resource_policy,
    }
    if maximum_gpu_utilization_percent is not None:
        resource["maximum_gpu_utilization_percent"] = (
            maximum_gpu_utilization_percent
        )
    workers = {
        "argv": engine_plan._ssh(
            _compressed_remote_shell_command(
                _worker_shell(launch_plan["case_commands"]),
                remote_run,
            )
        ),
        "expected_case_ids": [
            row["case_id"] for row in launch_plan["case_commands"]
        ],
    }
    assembly_argv = launch_plan["assembler_command"]["argv"]
    assembly_shell = (
        f"cd {shlex.quote(launch_plan['assembler_command']['cwd'])} && "
        f"exec {shlex.join([str(value) for value in assembly_argv])}"
    )
    remote_verify_argv = [
        REMOTE_PYTHON,
        "-c",
        _verification_script(),
        launch_plan["remote_source"],
        launch_plan["remote_artifact"],
    ]
    package_shell = " && ".join([
        "set -eu",
        f"cd {shlex.quote(remote_run)}",
        "test \"$(find artifact -mindepth 1 | wc -l)\" -gt 0",
        "tar -cf - artifact",
    ])
    package = output_dir / "benchmark_artifact.tar"
    downloaded = output_dir / "downloaded_benchmark"
    local_verifier_source = output_dir / LOCAL_VERIFIER_SOURCE_NAME
    return {
        "reserve_remote": {
            "argv": engine_plan._ssh(["bash", "-lc", reserve]),
        },
        "upload": {
            "argv": [
                engine_plan._scp(local_tar, remote_tar),
                engine_plan._scp(
                    local_prerequisites_tar,
                    remote_prerequisites_tar,
                ),
            ],
        },
        "stage": {
            "argv": engine_plan._ssh(["bash", "-lc", stage_shell]),
        },
        "resource_guard": resource,
        "workers": workers,
        "assembly": {
            "argv": engine_plan._ssh(
                ["bash", "-lc", assembly_shell]
            ),
        },
        "remote_verify": {
            "argv": engine_plan._ssh(remote_verify_argv),
        },
        "final_resource_guard": dict(resource),
        "package_download": {
            "remote_argv": engine_plan._ssh(
                ["bash", "-lc", package_shell]
            ),
            "local_output": str(package),
        },
        "safe_extract": {
            "argv": _extract_command(package, downloaded),
        },
        "local_verify": {
            "argv": [
                sys.executable,
                "-c",
                _local_verification_script(),
                str(local_tar),
                local_inputs["source_tar_sha256"],
                worker["source_tree_sha256"],
                str(local_verifier_source),
                str(downloaded / "artifact"),
            ],
        },
    }


def _validate_launch_plan(launch_plan):
    required = {
        "schema_version",
        "run_tag",
        "worker_authorization",
        "local_source_tar",
        "source_tar_sha256",
        "remote_source_tar",
        "remote_source",
        "remote_output",
        "remote_cases",
        "remote_logs",
        "remote_assembly",
        "remote_artifact",
        "remote_workload_manifest",
        "stage_command",
        "resource_policy",
        "maximum_gpu_utilization_percent",
        "case_commands",
        "assembly_metadata",
        "assembler_command",
    }
    if (
        not isinstance(launch_plan, dict)
        or set(launch_plan) != required
        or launch_plan["schema_version"] != BENCHMARK_SCHEMA_VERSION
        or not isinstance(launch_plan["case_commands"], list)
        or len(launch_plan["case_commands"]) != 70
        or not isinstance(launch_plan["worker_authorization"], dict)
    ):
        raise ValueError("benchmark launch plan schema mismatch")
    resource_policy = launch_plan["resource_policy"]
    maximum_gpu_utilization_percent = launch_plan[
        "maximum_gpu_utilization_percent"
    ]
    if resource_policy not in {
        "strict-exclusive",
        "shared-low-utilization",
    }:
        raise ValueError("benchmark resource policy is invalid")
    if resource_policy == "shared-low-utilization":
        if (
            isinstance(maximum_gpu_utilization_percent, bool)
            or not isinstance(maximum_gpu_utilization_percent, int)
            or not 0 <= maximum_gpu_utilization_percent <= 100
        ):
            raise ValueError(
                "benchmark maximum GPU utilization is invalid"
            )
    elif maximum_gpu_utilization_percent is not None:
        raise ValueError(
            "exclusive benchmark cannot set utilization limit"
        )
    source_tar = _regular_file(
        launch_plan["local_source_tar"],
        "source tar",
    )
    if _sha256(source_tar) != launch_plan["source_tar_sha256"]:
        raise ValueError("source tar identity mismatch")
    if (
        _source_tree_sha256_from_tar(source_tar)
        != launch_plan["worker_authorization"].get(
            "source_tree_sha256"
        )
    ):
        raise ValueError("source tree identity mismatch")
    for name in (
        "remote_source_tar",
        "remote_source",
        "remote_output",
        "remote_cases",
        "remote_logs",
        "remote_assembly",
        "remote_artifact",
        "remote_workload_manifest",
    ):
        _safe_remote(launch_plan[name], name)
    return source_tar


def build_remote_execution_plan(
    *,
    launch_plan,
    output_dir,
    local_prerequisites,
    local_model_manifest,
):
    source_tar = _validate_launch_plan(launch_plan)
    prerequisites = _regular_file(
        local_prerequisites,
        "correctness prerequisites",
    )
    model_manifest = _regular_file(
        local_model_manifest,
        "model manifest",
    )
    worker = launch_plan["worker_authorization"]
    if (
        _sha256(prerequisites) != worker.get("prerequisites_sha256")
        or _sha256(model_manifest)
        != worker.get("model_manifest_sha256")
    ):
        raise ValueError("launch plan local input identity mismatch")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("remote plan output directory already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ))
    try:
        temporary_source_tar = temporary / SOURCE_TAR_NAME
        temporary_prerequisites = temporary / PREREQUISITES_NAME
        temporary_prerequisites_tar = (
            temporary / PREREQUISITES_TAR_NAME
        )
        temporary_model_manifest = temporary / MODEL_MANIFEST_NAME
        shutil.copyfile(source_tar, temporary_source_tar)
        shutil.copyfile(prerequisites, temporary_prerequisites)
        prerequisite_files = _prerequisite_files(prerequisites)
        _write_prerequisite_tar(
            temporary_prerequisites_tar,
            prerequisite_files,
        )
        shutil.copyfile(model_manifest, temporary_model_manifest)
        _canonical_metadata_files(
            temporary,
            launch_plan["assembly_metadata"],
        )
        frozen_launch_plan = {
            **launch_plan,
            "local_source_tar": str(output_dir / SOURCE_TAR_NAME),
        }
        metadata = {
            name: str(output_dir / "assembly_metadata" / name)
            for name in launch_plan["assembly_metadata"]
        }
        local_inputs = {
            "source_tar": str(output_dir / SOURCE_TAR_NAME),
            "source_tar_sha256": _sha256(temporary_source_tar),
            "prerequisites": str(output_dir / PREREQUISITES_NAME),
            "prerequisites_sha256": _sha256(prerequisites),
            "prerequisites_tar": str(
                output_dir / PREREQUISITES_TAR_NAME
            ),
            "prerequisites_tar_sha256": _sha256(
                temporary_prerequisites_tar
            ),
            "prerequisites_owned_files": [
                name for name, _ in prerequisite_files
            ],
            "model_manifest": str(output_dir / MODEL_MANIFEST_NAME),
            "model_manifest_sha256": _sha256(model_manifest),
            "assembly_metadata": metadata,
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_tag": launch_plan["run_tag"],
            "ssh_target": SSH_TARGET,
            "worker_authorization": launch_plan[
                "worker_authorization"
            ],
            "case_commands": launch_plan["case_commands"],
            "launch_plan": frozen_launch_plan,
            "local_inputs": local_inputs,
            "command_order": COMMAND_ORDER,
            "commands": _commands(
                frozen_launch_plan,
                local_inputs,
                output_dir,
            ),
            "execution_performed": False,
            "claim_boundary": (
                "command authorization only; no SSH, GPU, correctness, "
                "performance, cache, memory, compression, or quality claim"
            ),
        }
        _write_json(temporary / PLAN_NAME, payload)
        os.replace(temporary, output_dir)
        return verify_remote_execution_plan(output_dir / PLAN_NAME)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_remote_execution_plan(path):
    path = Path(path)
    payload = _load_json(path, "remote execution plan")
    required = {
        "schema_version",
        "run_tag",
        "ssh_target",
        "worker_authorization",
        "case_commands",
        "launch_plan",
        "local_inputs",
        "command_order",
        "commands",
        "execution_performed",
        "claim_boundary",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["ssh_target"] != SSH_TARGET
        or payload["command_order"] != COMMAND_ORDER
        or payload["execution_performed"] is not False
    ):
        raise ValueError("remote execution plan schema mismatch")
    source_runner.safe_run_tag(payload["run_tag"])
    launch_plan = payload["launch_plan"]
    _validate_launch_plan(launch_plan)
    if (
        payload["run_tag"] != launch_plan["run_tag"]
        or payload["worker_authorization"]
        != launch_plan["worker_authorization"]
        or payload["case_commands"] != launch_plan["case_commands"]
    ):
        raise ValueError("remote execution plan binding mismatch")
    local_inputs = payload["local_inputs"]
    if not isinstance(local_inputs, dict):
        raise ValueError("remote execution plan local inputs mismatch")
    for path_name, sha_name in (
        ("source_tar", "source_tar_sha256"),
        ("prerequisites", "prerequisites_sha256"),
        ("prerequisites_tar", "prerequisites_tar_sha256"),
        ("model_manifest", "model_manifest_sha256"),
    ):
        local_path = _regular_file(local_inputs.get(path_name, ""), path_name)
        if _sha256(local_path) != local_inputs.get(sha_name):
            raise ValueError("remote execution plan input identity mismatch")
    prerequisite_inventory = _prerequisite_tar_inventory(
        local_inputs["prerequisites_tar"],
        local_inputs["prerequisites_sha256"],
    )
    if (
        local_inputs.get("prerequisites_owned_files")
        != prerequisite_inventory
    ):
        raise ValueError(
            "remote execution plan prerequisite inventory mismatch"
        )
    metadata = local_inputs.get("assembly_metadata")
    if not isinstance(metadata, dict) or set(metadata) != set(
        launch_plan["assembly_metadata"]
    ):
        raise ValueError("remote execution plan metadata mismatch")
    for name, local_path in metadata.items():
        if _load_json(local_path, name) != launch_plan[
            "assembly_metadata"
        ][name]:
            raise ValueError("remote execution plan metadata identity mismatch")
    expected = _commands(
        launch_plan,
        local_inputs,
        path.parent.resolve(),
    )
    if payload["commands"] != expected:
        raise ValueError("remote execution plan command mismatch")
    return payload
