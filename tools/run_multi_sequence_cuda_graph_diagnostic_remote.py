"""Run source-bound multi-sequence CUDA Graph diagnostics on the GPU host."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SSH_TARGET = "sitian@10.232.195.203"
SSH_CONTROL_PATH = "/tmp/ssh-sitian-10.232.195.203"
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
)
REMOTE_MODEL = "/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B"
DIST_PORT_ENV = "TINYVLLM_DIST_PORT"
MASTER_PORT_ENV = "MASTER_PORT"
PORT_COLLISION = "EADDRINUSE"
OWNED_ROOTS = ("tinyvllm", "tools")
IGNORED_UNTRACKED_PREFIXES = ("experiments",)
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "cuda_graph"


def _load_tool(module_name: str, filename: str):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load tool module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_tool(
    "multi_sequence_cuda_graph_contract",
    "multi_sequence_cuda_graph_contract.py",
)
source_audit = _load_tool("cuda_graph_source_audit", "source_audit.py")


SSH_OPTIONS = (
    "-S",
    SSH_CONTROL_PATH,
    "-o",
    "ControlMaster=auto",
    "-o",
    "ControlPersist=600",
)


def _canonical_bytes(value: object) -> bytes:
    return contract.canonical_json_bytes(value) + b"\n"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(_canonical_bytes(value))
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as output:
        for row in rows:
            output.write(_canonical_bytes(row))
    temporary.replace(path)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.endswith("\n"):
                raise ValueError(
                    f"{path} line {line_number} is not newline terminated"
                )
            rows.append(json.loads(line))
    return rows


def _artifact_record(root: Path, path: Path) -> dict:
    relative = path.resolve().relative_to(root.resolve())
    return {
        "path": relative.as_posix(),
        "sha256": contract.sha256_file(path),
    }


def allocate_unique_port_pairs(
    count: int,
    allocator: Callable[[], tuple[int, int]],
) -> list[tuple[int, int]]:
    if count < 0:
        raise ValueError("count must be non-negative")
    used_ports = set()
    pairs = []
    for _ in range(count):
        pair = allocator()
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not all(isinstance(port, int) for port in pair)
        ):
            raise ValueError("allocator must return two integer ports")
        dist_port, master_port = pair
        if dist_port == master_port:
            raise ValueError("port pair must contain distinct ports")
        if dist_port in used_ports or master_port in used_ports:
            raise ValueError("duplicate port allocated")
        if not all(1 <= port <= 65535 for port in pair):
            raise ValueError("allocated port is outside the valid range")
        used_ports.update(pair)
        pairs.append(pair)
    return pairs


def reserve_unique_port_pair(
    *,
    used_ports: set[int],
    pair: tuple[int, int],
    owner: str,
) -> None:
    if len(pair) != 2 or not all(isinstance(port, int) for port in pair):
        raise ValueError(f"{owner}: invalid port pair")
    dist_port, master_port = pair
    if dist_port == master_port:
        raise ValueError(f"{owner}: port pair must contain distinct ports")
    reused = sorted(set(pair) & used_ports)
    if reused:
        raise ValueError(f"{owner}: duplicate port allocated: {reused}")
    used_ports.update(pair)


def allocate_fresh_unique_port_pair(
    *,
    used_ports: set[int],
    allocator: Callable[[], tuple[int, int]],
    max_attempts: int = 16,
) -> tuple[int, int]:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    last_error = None
    for _ in range(max_attempts):
        pair = allocate_unique_port_pairs(1, allocator)[0]
        try:
            reserve_unique_port_pair(
                used_ports=used_ports,
                pair=pair,
                owner="new diagnostic process",
            )
        except ValueError as exc:
            if "duplicate port" not in str(exc):
                raise
            last_error = exc
            continue
        return pair
    raise ValueError(
        "failed to allocate globally unique port pair"
    ) from last_error


def is_retryable_port_collision(returncode: int, stderr: str) -> bool:
    return returncode != 0 and PORT_COLLISION in stderr


def read_remote_case_stderr(*, case_dir: Path, fallback: bytes) -> str:
    stderr_parts = []
    for name in ("launcher_stderr.txt", "stderr.txt"):
        path = case_dir / "output" / name
        if path.is_file():
            stderr_parts.append(
                path.read_text(encoding="utf-8", errors="replace")
            )
    if stderr_parts:
        return "\n".join(stderr_parts)
    return fallback.decode("utf-8", errors="replace")


def build_smoke_cases():
    allowed_batches = {2, 3, 4}
    allowed_trajectories = {"uniform-short", "ragged-context"}
    return tuple(
        case
        for case in contract.build_diagnostic_matrix()
        if case.batch_size in allowed_batches
        and case.trajectory in allowed_trajectories
        and case.repetition == 0
    )


def _eager_case_id(case) -> str:
    return (
        f"b{case.batch_size}__{case.trajectory}__"
        f"eager__r{case.repetition}"
    )


def graph_case_ready(case, run_dir: Path) -> bool:
    if case.mode == "eager":
        return True
    reference = (
        Path(run_dir)
        / "cases"
        / _eager_case_id(case)
        / "input"
        / "reference_tokens.json"
    )
    if not reference.is_file():
        return False
    try:
        tokens = _read_json(reference)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    total_steps = contract.WARMUP_STEPS + contract.MEASURED_STEPS
    return (
        isinstance(tokens, list)
        and len(tokens) == total_steps
        and all(
            isinstance(row, list) and len(row) == case.batch_size
            for row in tokens
        )
    )


def completed_case_is_resumable(
    case_dir: Path,
    case,
    source_tree_sha256: str,
    environment_sha256: str,
) -> bool:
    result_path = Path(case_dir) / "case_result.json"
    if not result_path.is_file():
        return False
    try:
        result = _read_json(result_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if result.get("schema_version") != 1 or result.get("status") != "PASS":
        return False
    expected = {
        "case": asdict(case),
        "case_id": case.case_id,
        "source_tree_sha256": source_tree_sha256,
        "environment_sha256": environment_sha256,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        return False
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return False
    for record in artifacts.values():
        if not isinstance(record, dict):
            return False
        relative_value = record.get("path")
        expected_hash = record.get("sha256")
        if not isinstance(relative_value, str) or not isinstance(
            expected_hash,
            str,
        ):
            return False
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            return False
        artifact = (Path(case_dir) / relative).resolve()
        try:
            artifact.relative_to(Path(case_dir).resolve())
        except ValueError:
            return False
        if (
            not artifact.is_file()
            or contract.sha256_file(artifact) != expected_hash
        ):
            return False
    return True


def available_case_artifacts(case_dir: Path) -> list[str]:
    case_dir = Path(case_dir)
    if not case_dir.is_dir():
        return []
    return sorted(
        path.relative_to(case_dir).as_posix()
        for path in case_dir.rglob("*")
        if path.is_file() and not path.name.endswith(".partial")
    )


def _ssh_command(remote_command: str) -> list[str]:
    return [
        "ssh",
        *SSH_OPTIONS,
        SSH_TARGET,
        "bash",
        "-lc",
        shlex.quote(remote_command),
    ]


def _run_remote(
    remote_command: str,
    *,
    input_bytes: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        _ssh_command(remote_command),
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"remote command failed with {result.returncode}: {stderr}"
        )
    return result


def _quote(path: str | Path) -> str:
    return shlex.quote(str(path))


def _make_run_tag(kind: str) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    entropy = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
    return f"{timestamp}-{kind}-{os.getpid()}-{entropy}"


def prepare_run_directory(
    *,
    output_root: Path,
    run_tag: str,
    resume: bool,
) -> Path:
    run_dir = Path(output_root) / run_tag
    if run_dir.exists():
        if not run_dir.is_dir():
            raise ValueError(f"run path is not a directory: {run_dir}")
        if not resume:
            raise ValueError(
                f"run directory already exists; use --resume: {run_dir}"
            )
    else:
        if resume:
            raise ValueError(f"resume run directory does not exist: {run_dir}")
        run_dir.mkdir(parents=True)
    return run_dir


def _remote_root(run_tag: str) -> str:
    if (
        not run_tag
        or any(character not in "abcdefghijklmnopqrstuvwxyz"
               "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in run_tag)
    ):
        raise ValueError("run tag contains unsupported characters")
    return f"/tmp/tllm-cuda-graph-{run_tag}"


def _create_source_snapshot(run_dir: Path) -> tuple[Path, dict]:
    staging = run_dir / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    evidence = source_audit.build_source_evidence(
        ROOT,
        staging,
        owned_roots=OWNED_ROOTS,
        ignored_untracked_prefixes=IGNORED_UNTRACKED_PREFIXES,
    )
    source_audit.validate_source_snapshot(
        staging / "source",
        evidence,
        staging / "source.patch",
        expected_owned_roots=OWNED_ROOTS,
    )
    archive_path = staging / "source_snapshot.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted((staging / "source").rglob("*")):
            if path.is_file():
                archive.add(
                    path,
                    arcname=path.relative_to(staging / "source"),
                    recursive=False,
                )
    return staging, evidence


def _upload_staging(staging: Path, remote_root: str) -> None:
    create = f"mkdir -p {_quote(remote_root)}"
    _run_remote(create)
    tar_process = subprocess.Popen(
        ["tar", "-C", str(staging), "-cf", "-", "."],
        stdout=subprocess.PIPE,
    )
    if tar_process.stdout is None:
        raise RuntimeError("failed to open source archive stream")
    extract = subprocess.run(
        _ssh_command(
            f"tar -C {_quote(remote_root)} -xf -"
        ),
        stdin=tar_process.stdout,
        capture_output=True,
        check=False,
    )
    tar_process.stdout.close()
    tar_returncode = tar_process.wait()
    if tar_returncode != 0 or extract.returncode != 0:
        stderr = extract.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            "source snapshot upload failed: "
            f"tar={tar_returncode}, ssh={extract.returncode}: {stderr}"
        )


def _remote_python_script(
    remote_root: str,
    script: str,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess:
    command = (
        f"cd {_quote(remote_root + '/source')} && "
        "PYTHONDONTWRITEBYTECODE=1 "
        f"PYTHONPATH={_quote(remote_root + '/source')} "
        f"{_quote(REMOTE_PYTHON)} -"
    )
    return _run_remote(
        command,
        input_bytes=script.encode("utf-8"),
        check=check,
    )


def _remote_validate_source(remote_root: str) -> None:
    script = f"""
import json
from pathlib import Path
from tools import source_audit

root = Path({remote_root!r})
evidence = json.loads((root / "source_evidence.json").read_text())
summary = source_audit.validate_source_snapshot(
    root / "source",
    evidence,
    root / "source.patch",
    expected_owned_roots={OWNED_ROOTS!r},
)
print(json.dumps(summary, sort_keys=True))
"""
    _remote_python_script(remote_root, script)


def _collect_environment(remote_root: str, source_hash: str) -> dict:
    script = f"""
import json
import platform
import subprocess
import torch
import transformers

try:
    import flash_attn
    flash_attention = getattr(flash_attn, "__version__", "unknown")
except Exception as exc:
    flash_attention = "unavailable:" + type(exc).__name__

driver = subprocess.run(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    capture_output=True,
    text=True,
    check=False,
).stdout.strip().splitlines()
properties = torch.cuda.get_device_properties(0)
environment = {{
    "schema_version": 1,
    "host": platform.node(),
    "python": platform.python_version(),
    "pytorch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "nvidia_driver": driver[0] if driver else "unknown",
    "gpu_name": properties.name,
    "flash_attention": flash_attention,
    "transformers": transformers.__version__,
    "model_identifier": {REMOTE_MODEL!r},
    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    "source_tree_sha256": {source_hash!r},
}}
print(json.dumps(environment, sort_keys=True))
"""
    result = _remote_python_script(remote_root, script)
    return json.loads(result.stdout.decode("utf-8"))


def _build_prompt_manifest(remote_root: str) -> dict:
    script = f"""
import json
from transformers import AutoTokenizer
from tools import diagnose_multi_sequence_cuda_graph as diagnostic
from tools import multi_sequence_cuda_graph_contract as contract

tokenizer = AutoTokenizer.from_pretrained(
    {REMOTE_MODEL!r},
    trust_remote_code=True,
    local_files_only=True,
)
trajectories = {{}}
for trajectory in contract.DIAGNOSTIC_TRAJECTORIES:
    trajectories[trajectory] = {{}}
    for batch_size in contract.DIAGNOSTIC_BATCH_SIZES:
        plan = diagnostic.build_prompt_plan(tokenizer, batch_size)[trajectory]
        trajectories[trajectory][str(batch_size)] = (
            contract.canonical_json_sha256(plan)
        )
print(json.dumps({{
    "schema_version": 1,
    "trajectories": trajectories,
}}, sort_keys=True))
"""
    result = _remote_python_script(remote_root, script)
    return json.loads(result.stdout.decode("utf-8"))


def _run_remote_preflight(remote_root: str) -> None:
    commands = (
        f"{_quote(REMOTE_PYTHON)} "
        "tools/test_multi_sequence_cuda_graph_gate.py",
        f"{_quote(REMOTE_PYTHON)} tools/test_model_runner_spec_verify.py",
        (
            f"{_quote(REMOTE_PYTHON)} -m py_compile "
            "tools/multi_sequence_cuda_graph_contract.py "
            "tools/diagnose_multi_sequence_cuda_graph.py "
            "tools/verify_multi_sequence_cuda_graph_diagnostic.py "
            "tools/run_multi_sequence_cuda_graph_diagnostic_remote.py"
        ),
    )
    for command in commands:
        _run_remote(
            f"cd {_quote(remote_root + '/source')} && "
            f"PYTHONPATH={_quote(remote_root + '/source')} {command}"
        )


def _remote_allocate_pair(remote_root: str) -> tuple[int, int]:
    script = """
import json
import socket

sockets = []
ports = []
try:
    for _ in range(2):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sockets.append(sock)
        ports.append(sock.getsockname()[1])
    print(json.dumps(ports))
finally:
    for sock in sockets:
        sock.close()
"""
    result = _remote_python_script(remote_root, script)
    ports = json.loads(result.stdout.decode("utf-8"))
    return int(ports[0]), int(ports[1])


def _upload_file(local_path: Path, remote_path: str) -> None:
    payload = local_path.read_bytes()
    parent = str(Path(remote_path).parent)
    command = (
        f"mkdir -p {_quote(parent)} && "
        f"cat > {_quote(remote_path + '.partial')} && "
        f"mv {_quote(remote_path + '.partial')} {_quote(remote_path)}"
    )
    _run_remote(command, input_bytes=payload)


def _download_tree(remote_path: str, local_path: Path) -> None:
    local_path = Path(local_path)
    if local_path.exists():
        raise ValueError(f"download destination exists: {local_path}")
    remote_parent = str(Path(remote_path).parent)
    remote_name = Path(remote_path).name
    size_result = _run_remote(
        f"tar -C {_quote(remote_parent)} -cf - {_quote(remote_name)} "
        "| wc -c"
    )
    expected_size = int(size_result.stdout.decode("utf-8").strip())
    archive_path = local_path.with_name(local_path.name + ".tar.partial")
    download = subprocess.run(
        _ssh_command(
            f"tar -C {_quote(remote_parent)} -cf - {_quote(remote_name)}"
        ),
        stdout=archive_path.open("wb"),
        stderr=subprocess.PIPE,
        check=False,
    )
    if download.returncode != 0:
        archive_path.unlink(missing_ok=True)
        stderr = download.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"artifact download failed: {stderr}")
    if archive_path.stat().st_size != expected_size:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError("artifact download size mismatch")
    partial_dir = local_path.with_name(local_path.name + ".partial")
    partial_dir.mkdir(parents=True)
    with tarfile.open(archive_path, "r:") as archive:
        members = archive.getmembers()
        for member in members:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("remote archive contains unsafe path")
            if member.issym() or member.islnk():
                raise ValueError("remote archive contains a link")
        archive.extractall(partial_dir, members=members)
    extracted = partial_dir / remote_name
    if not extracted.is_dir():
        raise RuntimeError("downloaded artifact tree is incomplete")
    extracted.replace(local_path)
    partial_dir.rmdir()
    archive_path.unlink()


def remove_downloaded_remote_case(remote_case: str) -> None:
    remote_case = str(remote_case)
    expected_prefix = "/tmp/tllm-cuda-graph-"
    if (
        not remote_case.startswith(expected_prefix)
        or "/cases/" not in remote_case
        or remote_case.endswith("/cases")
    ):
        raise ValueError(f"unsafe remote case path: {remote_case}")
    _run_remote(
        f"test -n {_quote(remote_case)} && "
        f"rm -r -- {_quote(remote_case)}"
    )


def _case_spec_path(case_dir: Path) -> Path:
    return case_dir / "input" / "case_spec.json"


def _reference_path(case_dir: Path) -> Path:
    return case_dir / "input" / "reference_tokens.json"


def _prepare_case_input(run_dir: Path, case) -> Path:
    case_dir = run_dir / "cases" / case.case_id
    input_dir = case_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_json(_case_spec_path(case_dir), asdict(case))
    reference = _reference_path(case_dir)
    if case.mode != "eager":
        eager_reference = (
            run_dir
            / "cases"
            / _eager_case_id(case)
            / "input"
            / "reference_tokens.json"
        )
        if not graph_case_ready(case, run_dir):
            raise RuntimeError(
                f"eager reference is not ready for {case.case_id}"
            )
        shutil.copyfile(eager_reference, reference)
    elif not reference.exists():
        _write_json(reference, None)
    return case_dir


def _run_remote_case(
    *,
    remote_root: str,
    run_dir: Path,
    case,
    ports: tuple[int, int],
) -> tuple[int, str]:
    local_case = _prepare_case_input(run_dir, case)
    remote_case = f"{remote_root}/cases/{case.case_id}"
    _run_remote(f"mkdir -p {_quote(remote_case + '/input')}")
    _upload_file(
        _case_spec_path(local_case),
        remote_case + "/input/case_spec.json",
    )
    _upload_file(
        _reference_path(local_case),
        remote_case + "/input/reference_tokens.json",
    )
    dist_port, master_port = ports
    source_root = remote_root + "/source"
    output_dir = remote_case + "/output"
    command = (
        f"cd {_quote(source_root)} && "
        f"mkdir -p {_quote(output_dir)} && "
        f"CUDA_VISIBLE_DEVICES=0 "
        f"{DIST_PORT_ENV}={dist_port} "
        f"{MASTER_PORT_ENV}={master_port} "
        f"PYTHONPATH={_quote(source_root)} "
        f"{_quote(REMOTE_PYTHON)} "
        "tools/diagnose_multi_sequence_cuda_graph.py "
        f"--model {_quote(REMOTE_MODEL)} "
        f"--case-spec {_quote(remote_case + '/input/case_spec.json')} "
        f"--reference-tokens "
        f"{_quote(remote_case + '/input/reference_tokens.json')} "
        f"--output-dir {_quote(output_dir)} "
        f"> {_quote(output_dir + '/stdout.txt')} "
        f"2> {_quote(output_dir + '/launcher_stderr.txt')}"
    )
    result = _run_remote(command, check=False)
    _download_tree(remote_case, local_case.with_name(local_case.name + ".new"))
    shutil.rmtree(local_case)
    local_case.with_name(local_case.name + ".new").replace(local_case)
    remove_downloaded_remote_case(remote_case)
    stderr = read_remote_case_stderr(
        case_dir=local_case,
        fallback=result.stderr,
    )
    return result.returncode, stderr


def _case_artifact_records(case_dir: Path) -> dict:
    output_dir = case_dir / "output"
    producer = _read_json(output_dir / "case_result.json")
    records = {}
    for name, record in producer.get("artifacts", {}).items():
        relative = Path(record["path"])
        artifact = output_dir / relative
        if not artifact.is_file():
            raise ValueError(
                f"{case_dir.name}: missing producer artifact {relative}"
            )
        if contract.sha256_file(artifact) != record["sha256"]:
            raise ValueError(
                f"{case_dir.name}: producer artifact hash mismatch {relative}"
            )
        records[name] = _artifact_record(case_dir, artifact)
    for name in ("stdout.txt", "launcher_stderr.txt", "case_result.json"):
        path = output_dir / name
        if path.is_file():
            records[f"launcher/{name}"] = _artifact_record(case_dir, path)
    return records


def _finalize_case_record(
    *,
    case_dir: Path,
    case,
    source_hash: str,
    environment_hash: str,
    prompt_hash: str,
    ports: tuple[int, int],
) -> dict:
    producer_path = case_dir / "output" / "case_result.json"
    producer = _read_json(producer_path)
    if producer.get("status") != "PASS":
        raise RuntimeError(f"diagnostic case failed: {case.case_id}")
    reference = _reference_path(case_dir)
    reference_tokens = _read_json(reference)
    reference_hash = contract.canonical_json_sha256(reference_tokens)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "case": asdict(case),
        "case_id": case.case_id,
        "source_tree_sha256": source_hash,
        "environment_sha256": environment_hash,
        "prompt_sha256": prompt_hash,
        "reference_token_sha256": reference_hash,
        "tinyvllm_dist_port": ports[0],
        "master_port": ports[1],
        "artifacts": _case_artifact_records(case_dir),
    }
    _write_json(case_dir / "case_result.json", result)
    return result


def _copy_artifact(
    source: Path,
    destination: Path,
) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)
    return _artifact_record(destination.parents[2], destination)


def promote_source_evidence_artifacts(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    for name in ("source.patch", "source_snapshot.tar.gz"):
        source = run_dir / "staging" / name
        if not source.is_file():
            raise ValueError(f"source evidence artifact missing: {name}")
        destination = run_dir / name
        temporary = destination.with_name(destination.name + ".partial")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)


def _merge_cases(
    *,
    run_dir: Path,
    cases,
    source_evidence: dict,
    environment: dict,
    prompt_manifest: dict,
    canonical: bool,
) -> None:
    promote_source_evidence_artifacts(run_dir)
    process_rows = []
    raw_rows = []
    layer_rows = []
    kv_rows = []
    reference_records = {}
    for case in cases:
        case_dir = run_dir / "cases" / case.case_id
        orchestration = _read_json(case_dir / "case_result.json")
        output_dir = case_dir / "output"
        artifacts = {}
        for name, subdirectory in (
            ("logits", "logits"),
            ("layers", "layers"),
            ("kv", "kv"),
        ):
            producer = _read_json(output_dir / "case_result.json")
            source = output_dir / producer["artifacts"][name]["path"]
            destination = (
                run_dir
                / "tensors"
                / subdirectory
                / f"{case.case_id}.pt"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            artifacts[name] = _artifact_record(run_dir, destination)
        raw_rows.extend(_read_jsonl(output_dir / "raw_rows.jsonl"))
        layer_rows.extend(
            _read_jsonl(output_dir / "layer_observations.jsonl")
        )
        kv_rows.extend(_read_jsonl(output_dir / "kv_observations.jsonl"))
        identity = (
            case.batch_size,
            case.trajectory,
            case.repetition,
        )
        reference_destination = (
            run_dir
            / "reference_tokens"
            / (
                f"b{case.batch_size}__{case.trajectory}"
                f"__r{case.repetition}.json"
            )
        )
        if identity not in reference_records:
            reference_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_reference_path(case_dir), reference_destination)
            reference_records[identity] = _artifact_record(
                run_dir,
                reference_destination,
            )
        process_rows.append(
            {
                **asdict(case),
                "case_id": case.case_id,
                "status": orchestration["status"],
                "source_tree_sha256": orchestration[
                    "source_tree_sha256"
                ],
                "environment_sha256": orchestration[
                    "environment_sha256"
                ],
                "prompt_sha256": orchestration["prompt_sha256"],
                "reference_token_sha256": orchestration[
                    "reference_token_sha256"
                ],
                "reference_tokens": reference_records[identity],
                "tinyvllm_dist_port": orchestration[
                    "tinyvllm_dist_port"
                ],
                "master_port": orchestration["master_port"],
                "artifacts": artifacts,
            }
        )

    _write_jsonl(run_dir / "process_rows.jsonl", process_rows)
    _write_jsonl(run_dir / "raw_rows.jsonl", raw_rows)
    _write_jsonl(run_dir / "layer_observations.jsonl", layer_rows)
    _write_jsonl(run_dir / "kv_observations.jsonl", kv_rows)
    _write_json(run_dir / "source_evidence.json", source_evidence)
    _write_json(run_dir / "environment.json", environment)
    _write_json(run_dir / "prompt_manifest.json", prompt_manifest)
    _write_json(
        run_dir / "manifest.json",
        {
            "schema_version": 1,
            "kind": "diagnostic",
            "canonical": canonical,
            "source_tree_sha256": source_evidence["tree_sha256"],
            "environment_sha256": contract.canonical_json_sha256(
                environment
            ),
            "prompt_manifest_sha256": contract.canonical_json_sha256(
                prompt_manifest
            ),
            "case_ids": [case.case_id for case in cases],
            "warmup_steps": contract.WARMUP_STEPS,
            "measured_steps": contract.MEASURED_STEPS,
            "logit_rtol": contract.LOGIT_RTOL,
            "logit_atol": contract.LOGIT_ATOL,
        },
    )
    if canonical:
        classification = "EXACT_REPLAY_CORRECT"
        rounded = "ROUNDED_REPLAY_CORRECT"
    else:
        classification = "NON_AUTHORITATIVE_SMOKE"
        rounded = "NON_AUTHORITATIVE_SMOKE"
    _write_json(
        run_dir / "summary.json",
        {
            "schema_version": 1,
            "classification": classification,
            "rounded_classification": rounded,
            "case_count": len(process_rows),
        },
    )
    report = (
        "# Multi-Sequence CUDA Graph Diagnostic\n\n"
        f"- Canonical: `{str(canonical).lower()}`\n"
        f"- Cases: `{len(process_rows)}`\n"
        f"- Producer classification: `{classification}`\n"
        "- Independent verification is authoritative.\n"
    )
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    hashed_paths = [
        path
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.name != "sha256sums.txt"
        and "independent-verification" not in path.parts
        and "staging" not in path.parts
        and "cases" not in path.parts
    ]
    (run_dir / "sha256sums.txt").write_text(
        "".join(
            f"{contract.sha256_file(path)}  "
            f"{path.relative_to(run_dir).as_posix()}\n"
            for path in sorted(hashed_paths)
        ),
        encoding="utf-8",
    )


def _run_independent_verifier(
    run_dir: Path,
    verifier_python: Path,
) -> int:
    verifier = TOOLS / "verify_multi_sequence_cuda_graph_diagnostic.py"
    capability = subprocess.run(
        [
            str(verifier_python),
            "-c",
            "import torch; print(torch.__version__)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if capability.returncode != 0:
        raise RuntimeError(
            "verifier Python cannot import torch: "
            + capability.stderr.strip()
        )
    result = subprocess.run(
        [
            str(verifier_python),
            str(verifier),
            "--run-dir",
            str(run_dir),
        ],
        cwd=ROOT,
        check=False,
    )
    return result.returncode


def _run_diagnostic(
    *,
    kind: str,
    output_root: Path,
    run_tag: str | None,
    keep_remote: bool,
    resume: bool,
    verifier_python: Path,
) -> tuple[Path, int]:
    canonical = kind == "diagnostic-canonical"
    cases = (
        contract.build_diagnostic_matrix()
        if canonical
        else build_smoke_cases()
    )
    run_tag = run_tag or _make_run_tag(kind)
    run_dir = prepare_run_directory(
        output_root=output_root,
        run_tag=run_tag,
        resume=resume,
    )
    remote_root = _remote_root(run_tag)
    staging, source_evidence = _create_source_snapshot(run_dir)
    source_hash = source_evidence["tree_sha256"]
    remote_created = False
    try:
        _upload_staging(staging, remote_root)
        remote_created = True
        _remote_validate_source(remote_root)
        _run_remote_preflight(remote_root)
        environment = _collect_environment(remote_root, source_hash)
        if environment.get("bf16_supported") is not True:
            raise RuntimeError("remote GPU does not report BF16 support")
        prompt_manifest = _build_prompt_manifest(remote_root)
        _write_json(run_dir / "environment.json", environment)
        _write_json(run_dir / "prompt_manifest.json", prompt_manifest)
        environment_hash = contract.canonical_json_sha256(environment)
        allocated_ports = set()

        def allocate_fresh_pair() -> tuple[int, int]:
            return allocate_fresh_unique_port_pair(
                used_ports=allocated_ports,
                allocator=lambda: _remote_allocate_pair(remote_root),
            )

        for case in cases:
            case_dir = run_dir / "cases" / case.case_id
            if completed_case_is_resumable(
                case_dir,
                case,
                source_hash,
                environment_hash,
            ):
                resumed = _read_json(case_dir / "case_result.json")
                reserve_unique_port_pair(
                    used_ports=allocated_ports,
                    pair=(
                        resumed["tinyvllm_dist_port"],
                        resumed["master_port"],
                    ),
                    owner=f"resumed case {case.case_id}",
                )
                continue
            if case.mode != "eager" and not graph_case_ready(case, run_dir):
                raise RuntimeError(
                    f"graph dependency missing for {case.case_id}"
                )
            attempts = 0
            while True:
                ports = allocate_fresh_pair()
                returncode, stderr = _run_remote_case(
                    remote_root=remote_root,
                    run_dir=run_dir,
                    case=case,
                    ports=ports,
                )
                if returncode == 0:
                    break
                if is_retryable_port_collision(returncode, stderr):
                    attempts += 1
                    if attempts <= 2:
                        shutil.rmtree(case_dir)
                        continue
                _write_json(
                    case_dir / "failure_manifest.json",
                    {
                        "schema_version": 1,
                        "case_id": case.case_id,
                        "returncode": returncode,
                        "available_artifacts": available_case_artifacts(
                            case_dir
                        ),
                    },
                )
                raise RuntimeError(
                    f"remote case failed: {case.case_id}: {stderr[-2000:]}"
                )
            prompt_hash = prompt_manifest["trajectories"][
                case.trajectory
            ][str(case.batch_size)]
            _finalize_case_record(
                case_dir=case_dir,
                case=case,
                source_hash=source_hash,
                environment_hash=environment_hash,
                prompt_hash=prompt_hash,
                ports=ports,
            )

        _merge_cases(
            run_dir=run_dir,
            cases=cases,
            source_evidence=source_evidence,
            environment=environment,
            prompt_manifest=prompt_manifest,
            canonical=canonical,
        )
        if canonical:
            verify_returncode = _run_independent_verifier(
                run_dir,
                verifier_python,
            )
        else:
            _write_json(
                run_dir / "independent-verification-smoke.json",
                {
                    "schema_version": 1,
                    "classification": "NON_AUTHORITATIVE_SMOKE",
                    "case_count": len(cases),
                    "canonical_classification_written": False,
                },
            )
            verify_returncode = 0
        return run_dir, verify_returncode
    finally:
        if remote_created and not keep_remote:
            _run_remote(
                f"test -n {_quote(remote_root)} && "
                f"test {_quote(remote_root)} != /tmp && "
                f"rm -r -- {_quote(remote_root)}",
                check=False,
            )


def _run_preflight(
    *,
    output_root: Path,
    run_tag: str | None,
    keep_remote: bool,
) -> Path:
    run_tag = run_tag or _make_run_tag("preflight")
    run_dir = prepare_run_directory(
        output_root=output_root,
        run_tag=run_tag,
        resume=False,
    )
    remote_root = _remote_root(run_tag)
    staging, evidence = _create_source_snapshot(run_dir)
    remote_created = False
    try:
        _upload_staging(staging, remote_root)
        remote_created = True
        _remote_validate_source(remote_root)
        _run_remote_preflight(remote_root)
        environment = _collect_environment(
            remote_root,
            evidence["tree_sha256"],
        )
        prompt_manifest = _build_prompt_manifest(remote_root)
        _write_json(run_dir / "environment.json", environment)
        _write_json(run_dir / "prompt_manifest.json", prompt_manifest)
        _write_json(
            run_dir / "preflight.json",
            {
                "schema_version": 1,
                "status": "PASS",
                "source_tree_sha256": evidence["tree_sha256"],
                "environment_sha256": contract.canonical_json_sha256(
                    environment
                ),
            },
        )
        return run_dir
    finally:
        if remote_created and not keep_remote:
            _run_remote(
                f"rm -r -- {_quote(remote_root)}",
                check=False,
            )


def _download_only(
    *,
    output_root: Path,
    run_tag: str,
) -> Path:
    destination = Path(output_root) / f"{run_tag}-recovery"
    _download_tree(_remote_root(run_tag), destination)
    return destination


def _verify_only(
    *,
    output_root: Path,
    run_tag: str,
    verifier_python: Path,
) -> tuple[Path, int]:
    run_dir = Path(output_root) / run_tag
    if not run_dir.is_dir():
        raise ValueError(f"run directory does not exist: {run_dir}")
    return run_dir, _run_independent_verifier(
        run_dir,
        verifier_python,
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Source-bound remote CUDA Graph diagnostic orchestrator",
    )
    parser.add_argument(
        "mode",
        choices=(
            "preflight",
            "diagnostic-smoke",
            "diagnostic-canonical",
            "download-only",
            "verify-only",
        ),
    )
    parser.add_argument("--run-tag")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--verifier-python",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument("--keep-remote", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.mode in {"download-only", "verify-only"} and not args.run_tag:
        raise SystemExit(f"{args.mode} requires --run-tag")
    if args.resume and args.mode not in {
        "diagnostic-smoke",
        "diagnostic-canonical",
    }:
        raise SystemExit("--resume is only valid for diagnostic runs")
    if args.mode == "preflight":
        run_dir = _run_preflight(
            output_root=args.output_root,
            run_tag=args.run_tag,
            keep_remote=args.keep_remote,
        )
        print(run_dir)
        return 0
    if args.mode == "download-only":
        run_dir = _download_only(
            output_root=args.output_root,
            run_tag=args.run_tag,
        )
        print(run_dir)
        return 0
    if args.mode == "verify-only":
        run_dir, returncode = _verify_only(
            output_root=args.output_root,
            run_tag=args.run_tag,
            verifier_python=args.verifier_python,
        )
        print(run_dir)
        return returncode
    run_dir, returncode = _run_diagnostic(
        kind=args.mode,
        output_root=args.output_root,
        run_tag=args.run_tag,
        keep_remote=args.keep_remote,
        resume=args.resume,
        verifier_python=args.verifier_python,
    )
    print(run_dir)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
