#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import sys
import tarfile
import traceback


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from autoregressive_draft_cuda_graph_gate import build_worker_command


REMOTE_TARGET = "sitian@10.232.195.203"
REMOTE_TASK_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
REMOTE_CURRENT_SOURCE = f"{REMOTE_TASK_ROOT}/source"
REMOTE_PYTHON = "/data00/home/sitian/tllm/env/bin/python"
REMOTE_PACKAGE_ROOT = (
    "/data00/home/sitian/tllm/env/lib/python3.11/site-packages"
)
DEFAULT_TARGET_MODEL = "/data00/home/sitian/.ms_cache/Qwen/Qwen3-8B"
DEFAULT_DRAFT_MODEL = (
    "/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B"
)
EXPECTED_REMOTE_REF = "origin/feat/kv-sparse-attention"
EXPECTED_KERBEROS_PRINCIPAL = "sitian@BYTEDANCE.COM"
EXPECTED_KERBEROS_TGT = "krbtgt/BYTEDANCE.COM@BYTEDANCE.COM"
MINIMUM_KERBEROS_LIFETIME_SECONDS = 5400
KERBEROS_TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
MAX_IDLE_MEMORY_USED_MIB = 1024
MAX_IDLE_UTILIZATION_PERCENT = 5
RUN_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
SOURCE_PATHS = (
    "tinyvllm/",
    "tools/autoregressive_draft_performance_worker.py",
    "tools/autoregressive_draft_performance_gate.py",
    "tools/autoregressive_draft_cuda_graph_contract.py",
    "tools/autoregressive_draft_cuda_graph_gate.py",
    "tools/autoregressive_draft_command_timeline_diagnostic.py",
    "tools/verify_autoregressive_draft_command_timeline_diagnostic.py",
    "tools/autoregressive_draft_paired_stability_diagnostic.py",
    "tools/autoregressive_draft_instability_telemetry.py",
    "tools/autoregressive_draft_host_semantic_diagnostic.py",
    "tools/autoregressive_draft_host_sampler.py",
    "tools/run_autoregressive_draft_command_timeline_remote.py",
)
RECEIPT_LOCATION_FIELDS = frozenset({
    "verified_at_utc",
    "verification_location",
    "artifact_path",
})
DETACHED_ATTESTATION_PATHS = frozenset({
    "manifest.sha256",
    "verify.command-timeline.remote.json",
    "verify.command-timeline.remote.log",
    "verify.command-timeline.local.json",
    "verify.command-timeline.local.log",
})


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run_tag(value: object) -> str:
    if not isinstance(value, str) or not RUN_TAG_PATTERN.fullmatch(value):
        raise ValueError("run tag must match [A-Za-z0-9_-]+")
    return value


def primary_run_path(run_tag: object) -> str:
    tag = validate_run_tag(run_tag)
    return f"{REMOTE_TASK_ROOT}/runs/{tag}"


def controller_run_path(run_tag: object) -> str:
    tag = validate_run_tag(run_tag)
    return f"{REMOTE_TASK_ROOT}/controller-verification/{tag}"


def build_epoch_schedule() -> list[tuple[str, str, str]]:
    return [
        ("block-0", "eager", "first"),
        ("block-0", "graph", "second"),
        ("block-1", "graph", "first"),
        ("block-1", "eager", "second"),
        ("block-2", "graph", "first"),
        ("block-2", "eager", "second"),
        ("block-3", "eager", "first"),
        ("block-3", "graph", "second"),
    ]


def build_ssh_command(remote_arguments) -> list[str]:
    return [
        "ssh",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        REMOTE_TARGET,
        shlex.join([str(value) for value in remote_arguments]),
    ]


def build_epoch_worker_command(
    *,
    source_root: str,
    output_path: str,
    target_model: str,
    draft_model: str,
    mode: str,
    gpu_indices,
) -> list[str]:
    indices = tuple(gpu_indices)
    if (
        len(indices) != 4
        or len(set(indices)) != 4
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in indices
        )
    ):
        raise ValueError("epoch worker requires four GPU indices")
    return [
        "env",
        "CUDA_VISIBLE_DEVICES="
        + ",".join(str(index) for index in indices),
        f"PYTHONPATH={REMOTE_PACKAGE_ROOT}:{source_root}",
        *build_worker_command(
            python=REMOTE_PYTHON,
            worker_script=(
                f"{source_root}/tools/"
                "autoregressive_draft_performance_worker.py"
            ),
            target_model=target_model,
            draft_model=draft_model,
            mode=mode,
            output_path=output_path,
            warmup_runs=1,
            measured_runs=5,
            command_timeline=True,
        ),
    ]


def _require_safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError("source archive path is unsafe")
    return path


def _lstat_path_without_symlinks(root: Path, relative: PurePosixPath):
    current = root
    root_info = root.lstat()
    if stat.S_ISLNK(root_info.st_mode):
        raise ValueError("source root must not be a symlink")
    if not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("source root must be a directory")
    info = root_info
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except OSError as error:
            raise ValueError(f"missing source path: {relative}") from error
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(
                f"source path contains a symlink: {relative}"
            )
    return current, info


def _source_inventory(repo_root: Path) -> list[tuple[Path, str, bool]]:
    root = Path(repo_root)
    inventory = []
    names = set()
    for configured in SOURCE_PATHS:
        relative_text = configured.rstrip("/")
        relative = _require_safe_relative(relative_text)
        path, info = _lstat_path_without_symlinks(root, relative)
        if stat.S_ISREG(info.st_mode):
            candidates = [(path, relative_text, False)]
        elif stat.S_ISDIR(info.st_mode):
            candidates = [(path, relative_text, True)]
            for directory, directory_names, file_names in os.walk(
                path,
                topdown=True,
                followlinks=False,
            ):
                directory_path = Path(directory)
                for name in sorted(directory_names):
                    child = directory_path / name
                    child_info = child.lstat()
                    child_relative = child.relative_to(root).as_posix()
                    if stat.S_ISLNK(child_info.st_mode):
                        raise ValueError(
                            "source path contains a symlink: "
                            + child_relative
                        )
                    if not stat.S_ISDIR(child_info.st_mode):
                        raise ValueError(
                            "source directory entry is not regular: "
                            + child_relative
                        )
                    if name == "__pycache__":
                        directory_names.remove(name)
                        continue
                    candidates.append((child, child_relative, True))
                for name in sorted(file_names):
                    child = directory_path / name
                    child_info = child.lstat()
                    child_relative = child.relative_to(root).as_posix()
                    if stat.S_ISLNK(child_info.st_mode):
                        raise ValueError(
                            "source path contains a symlink: "
                            + child_relative
                        )
                    if not stat.S_ISREG(child_info.st_mode):
                        raise ValueError(
                            "source archive entry is not regular: "
                            + child_relative
                        )
                    if child.suffix in (".pyc", ".pyo"):
                        continue
                    candidates.append((child, child_relative, False))
        else:
            raise ValueError(
                f"source archive entry is not regular: {relative_text}"
            )
        for candidate, candidate_relative, is_directory in candidates:
            archive_name = f"source/{candidate_relative}"
            _require_safe_relative(archive_name)
            if archive_name in names:
                raise ValueError("source archive path is duplicated")
            names.add(archive_name)
            inventory.append(
                (candidate, archive_name, is_directory)
            )
    return sorted(inventory, key=lambda row: row[1])


def build_source_archive(repo_root: Path, archive_path: Path) -> Path:
    root = Path(repo_root)
    destination = Path(archive_path)
    if destination.exists():
        raise ValueError("source archive already exists")
    inventory = _source_inventory(root)
    if not inventory:
        raise ValueError("source archive inventory is empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:") as archive:
        root_info = tarfile.TarInfo("source")
        root_info.type = tarfile.DIRTYPE
        root_info.mode = 0o755
        archive.addfile(root_info)
        for path, archive_name, is_directory in inventory:
            info = archive.gettarinfo(str(path), arcname=archive_name)
            if info.issym() or info.islnk():
                raise ValueError("source archive must not contain links")
            if is_directory:
                if not info.isdir():
                    raise ValueError(
                        "source archive directory is invalid"
                    )
                archive.addfile(info)
            else:
                if not info.isreg():
                    raise ValueError(
                        "source archive entry is not regular"
                    )
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
    return destination


def extract_source_archive(archive_path: Path, output_root: Path) -> Path:
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=False)
    seen = set()
    with tarfile.open(archive_path, "r:") as archive:
        members = archive.getmembers()
        for member in members:
            relative = _require_safe_relative(member.name)
            if (
                relative.parts[0] != "source"
                or member.name in seen
                or member.issym()
                or member.islnk()
                or not (member.isdir() or member.isreg())
            ):
                raise ValueError("source archive member is unsafe")
            seen.add(member.name)
        archive.extractall(destination, filter="data")
    return destination / "source"


def _kerberos_inconclusive(
    reason: str,
    *,
    minimum: int,
    principal=None,
    tgt_principal=None,
    expires_at=None,
    remaining=None,
) -> dict:
    result = {
        "status": "INCONCLUSIVE_ENVIRONMENT",
        "reason": reason,
        "minimum_required_lifetime_seconds": minimum,
    }
    if principal is not None:
        result["principal"] = principal
    if tgt_principal is not None:
        result["tgt_principal"] = tgt_principal
    if expires_at is not None:
        result["expires_at"] = expires_at
    if remaining is not None:
        result["remaining_lifetime_seconds"] = remaining
    return result


def classify_local_kerberos_payload(
    payload,
    *,
    now: datetime,
    minimum_lifetime_seconds: int = MINIMUM_KERBEROS_LIFETIME_SECONDS,
) -> dict:
    if (
        not isinstance(payload, dict)
        or not isinstance(now, datetime)
        or now.utcoffset() is None
        or isinstance(minimum_lifetime_seconds, bool)
        or not isinstance(minimum_lifetime_seconds, int)
        or minimum_lifetime_seconds <= 0
    ):
        return _kerberos_inconclusive(
            "local Kerberos payload is invalid",
            minimum=minimum_lifetime_seconds,
        )
    principal = payload.get("principal")
    tickets = payload.get("tickets")
    if not isinstance(principal, str) or not isinstance(tickets, list):
        return _kerberos_inconclusive(
            "local Kerberos payload is invalid",
            minimum=minimum_lifetime_seconds,
        )
    if principal != EXPECTED_KERBEROS_PRINCIPAL:
        return _kerberos_inconclusive(
            "local Kerberos principal is unexpected",
            minimum=minimum_lifetime_seconds,
            principal=principal,
        )
    matching = [
        row
        for row in tickets
        if isinstance(row, dict)
        and row.get("Principal") == EXPECTED_KERBEROS_TGT
    ]
    if not matching:
        return _kerberos_inconclusive(
            "local Kerberos TGT is missing",
            minimum=minimum_lifetime_seconds,
            principal=principal,
        )
    expires = matching[0].get("Expires")
    if not isinstance(expires, str):
        return _kerberos_inconclusive(
            "local Kerberos payload is invalid",
            minimum=minimum_lifetime_seconds,
            principal=principal,
            tgt_principal=EXPECTED_KERBEROS_TGT,
        )
    try:
        expires_at = datetime.strptime(
            expires,
            KERBEROS_TIMESTAMP_FORMAT,
        ).replace(tzinfo=now.tzinfo)
    except ValueError:
        return _kerberos_inconclusive(
            "local Kerberos payload is invalid",
            minimum=minimum_lifetime_seconds,
            principal=principal,
            tgt_principal=EXPECTED_KERBEROS_TGT,
        )
    remaining = int((expires_at - now).total_seconds())
    shared = {
        "minimum": minimum_lifetime_seconds,
        "principal": principal,
        "tgt_principal": EXPECTED_KERBEROS_TGT,
        "expires_at": expires_at.isoformat(),
        "remaining": remaining,
    }
    if remaining <= 0:
        return _kerberos_inconclusive(
            "local Kerberos TGT is expired",
            **shared,
        )
    if remaining < minimum_lifetime_seconds:
        return _kerberos_inconclusive(
            "local Kerberos TGT lifetime is insufficient",
            **shared,
        )
    return {
        "status": "READY",
        "principal": principal,
        "tgt_principal": EXPECTED_KERBEROS_TGT,
        "expires_at": expires_at.isoformat(),
        "remaining_lifetime_seconds": remaining,
        "minimum_required_lifetime_seconds": minimum_lifetime_seconds,
    }


def _local_kerberos_preflight(
    *,
    command_runner=subprocess.run,
    now=None,
) -> dict:
    current = datetime.now().astimezone() if now is None else now
    try:
        result = command_runner(
            ["klist", "--json"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return _kerberos_inconclusive(
            "local Kerberos cache is unavailable",
            minimum=MINIMUM_KERBEROS_LIFETIME_SECONDS,
        )
    if result.returncode != 0:
        return _kerberos_inconclusive(
            "local Kerberos cache is unavailable",
            minimum=MINIMUM_KERBEROS_LIFETIME_SECONDS,
        )
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError):
        return _kerberos_inconclusive(
            "local Kerberos payload is invalid",
            minimum=MINIMUM_KERBEROS_LIFETIME_SECONDS,
        )
    return classify_local_kerberos_payload(payload, now=current)


def classify_gpu_preflight(rows) -> dict:
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("GPU preflight requires exactly four rows")
    indices = []
    uuids = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("GPU preflight row must be a mapping")
        index = row.get("index")
        uuid = row.get("uuid")
        memory = row.get("memory_used_mib")
        utilization = row.get("utilization_percent")
        processes = row.get("compute_processes")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or not isinstance(uuid, str)
            or not uuid
            or isinstance(memory, bool)
            or not isinstance(memory, int)
            or memory < 0
            or isinstance(utilization, bool)
            or not isinstance(utilization, int)
            or utilization < 0
            or not isinstance(processes, list)
        ):
            raise ValueError("GPU preflight row is invalid")
        if processes:
            raise ValueError("selected GPU has an unrelated process")
        if (
            memory > MAX_IDLE_MEMORY_USED_MIB
            or utilization > MAX_IDLE_UTILIZATION_PERCENT
        ):
            raise ValueError("selected GPU is not idle")
        indices.append(index)
        uuids.append(uuid)
    if len(set(indices)) != 4 or len(set(uuids)) != 4:
        raise ValueError("GPU preflight inventory is duplicated")
    return {
        "status": "READY",
        "gpu_indices": indices,
        "gpu_uuids": uuids,
    }


def _run_command(
    command,
    *,
    command_runner=subprocess.run,
    context: str,
    allow_failure: bool = False,
    **kwargs,
):
    result = command_runner(command, check=False, **kwargs)
    if result.returncode != 0 and not allow_failure:
        detail = (
            getattr(result, "stderr", "")
            or getattr(result, "stdout", "")
            or ""
        )
        raise RuntimeError(f"{context} failed: {str(detail).strip()}")
    return result


def _local_source_commit(
    *,
    repo_root: Path,
    command_runner=subprocess.run,
) -> str:
    resolved = []
    for revision in ("HEAD", EXPECTED_REMOTE_REF):
        result = _run_command(
            ["git", "rev-parse", revision],
            command_runner=command_runner,
            context=f"resolve {revision}",
            cwd=Path(repo_root),
            text=True,
            capture_output=True,
        )
        value = result.stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError(f"{revision} is not a full commit")
        resolved.append(value)
    if resolved[0] != resolved[1]:
        raise ValueError(
            "local HEAD must equal origin/feat/kv-sparse-attention"
        )
    return resolved[0]


def _local_source_patch(
    *,
    repo_root: Path,
    command_runner=subprocess.run,
) -> bytes:
    result = _run_command(
        ["git", "diff", "--binary", "HEAD", "--", *SOURCE_PATHS],
        command_runner=command_runner,
        context="source patch generation",
        cwd=Path(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if not isinstance(result.stdout, bytes):
        raise ValueError("source patch must be returned as bytes")
    return result.stdout


def _preflight_remote_script(run_tag: str) -> str:
    primary = primary_run_path(run_tag)
    controller = controller_run_path(run_tag)
    return "\n".join((
        "TASK7_ACTION=preflight",
        f"{REMOTE_PYTHON} - <<'PY'",
        "import json,pathlib,subprocess",
        f"primary=pathlib.Path({primary!r})",
        f"controller=pathlib.Path({controller!r})",
        "gpu=subprocess.run([",
        " 'nvidia-smi',",
        " '--query-gpu=index,uuid,memory.used,utilization.gpu',",
        " '--format=csv,noheader,nounits'],",
        " capture_output=True,text=True,check=False)",
        "apps=subprocess.run([",
        " 'nvidia-smi',",
        " '--query-compute-apps=pid,gpu_uuid,process_name',",
        " '--format=csv,noheader,nounits'],",
        " capture_output=True,text=True,check=False)",
        "if gpu.returncode or apps.returncode: raise SystemExit(3)",
        "processes={}",
        "for line in apps.stdout.splitlines():",
        " fields=[part.strip() for part in line.split(',')]",
        " if len(fields)==3:",
        "  processes.setdefault(fields[1],[]).append(",
        "   {'pid':int(fields[0]),'process_name':fields[2]})",
        "rows=[]",
        "for line in gpu.stdout.splitlines():",
        " fields=[part.strip() for part in line.split(',')]",
        " if len(fields)!=4: raise SystemExit(4)",
        " rows.append({'index':int(fields[0]),'uuid':fields[1],",
        "  'memory_used_mib':int(fields[2]),",
        "  'utilization_percent':int(fields[3]),",
        "  'compute_processes':processes.get(fields[1],[])})",
        "idle=[row for row in rows if row['memory_used_mib']<=1024",
        " and row['utilization_percent']<=5",
        " and not row['compute_processes']]",
        "print(json.dumps({'primary_exists':primary.exists(),",
        " 'controller_exists':controller.exists(),",
        " 'gpu_rows':idle},sort_keys=True,separators=(',',':')))",
        "PY",
    ))


def run_preflight(
    *,
    run_tag: str,
    command_runner=subprocess.run,
    now=None,
    repo_root: Path | None = None,
) -> dict:
    tag = validate_run_tag(run_tag)
    kerberos = _local_kerberos_preflight(
        command_runner=command_runner,
        now=now,
    )
    if kerberos["status"] != "READY":
        return kerberos
    root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else Path(repo_root)
    )
    source_commit = _local_source_commit(
        repo_root=root,
        command_runner=command_runner,
    )
    result = _run_command(
        build_ssh_command([
            "bash",
            "-lc",
            _preflight_remote_script(tag),
        ]),
        command_runner=command_runner,
        context="remote read-only preflight",
        text=True,
        capture_output=True,
    )
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("remote preflight returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError("remote preflight payload is invalid")
    if payload.get("primary_exists") is not False:
        raise ValueError("primary remote destination already exists")
    if payload.get("controller_exists") is not False:
        raise ValueError("controller remote destination already exists")
    gpu = classify_gpu_preflight(payload.get("gpu_rows"))
    return {
        **gpu,
        "source_commit": source_commit,
        "local_kerberos": kerberos,
        "primary_run": primary_run_path(tag),
        "controller_run": controller_run_path(tag),
    }


def augment_worker_payload(
    raw_worker: dict,
    *,
    source_commit: str,
    source_tree_sha256: str,
    target_checkpoint_identifier: str,
    draft_checkpoint_identifier: str,
    tokenizer_identifier: str,
    gpu_uuids,
) -> dict:
    if not isinstance(raw_worker, dict):
        raise ValueError("raw worker payload must be a mapping")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source commit is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", source_tree_sha256):
        raise ValueError("source tree digest is invalid")
    identities = tuple(gpu_uuids)
    if (
        len(identities) != 4
        or len(set(identities)) != 4
        or any(not isinstance(value, str) or not value for value in identities)
    ):
        raise ValueError("GPU UUID inventory is invalid")
    worker = copy.deepcopy(raw_worker)
    worker.update({
        "tensor_parallel_size": 4,
        "max_proposal_tokens": 4,
        "requested_output_tokens": 16,
        "request_order": [0, 1, 2, 3],
        "temperature": 0.0,
        "proposal_kv_allocator": "direct",
        "proposal_kv_offload": False,
        "source_commit": source_commit,
        "source_tree_sha256": source_tree_sha256,
        "target_checkpoint_identifier": target_checkpoint_identifier,
        "draft_checkpoint_identifier": draft_checkpoint_identifier,
        "tokenizer_identifier": tokenizer_identifier,
        "gpu_uuids": list(identities),
    })
    return worker


def derive_telemetry_sidecar(epoch_key: str, worker: dict) -> dict:
    if not isinstance(epoch_key, str) or not epoch_key:
        raise ValueError("epoch key is invalid")
    measured = worker.get("measured_runs") if isinstance(worker, dict) else None
    if not isinstance(measured, list) or len(measured) != 5:
        raise ValueError("worker must contain five measured runs")
    rows = []
    for run in measured:
        if not isinstance(run, dict):
            raise ValueError("measured run is invalid")
        try:
            rows.append({
                "repeat": run["repeat"],
                "command_timeline_repeat_index": run[
                    "command_timeline_repeat_index"
                ],
                "telemetry": copy.deepcopy(run["telemetry"]),
            })
        except KeyError as error:
            raise ValueError("measured run telemetry is missing") from error
    return {
        "schema_version": 1,
        "epoch_key": epoch_key,
        "measured_runs": rows,
    }


def normalize_verification_receipt(receipt: dict) -> dict:
    if not isinstance(receipt, dict):
        raise ValueError("verification receipt must be a mapping")
    return {
        key: copy.deepcopy(value)
        for key, value in receipt.items()
        if key not in RECEIPT_LOCATION_FIELDS
    }


def _action_command(action: str, arguments: list[str]) -> list[str]:
    script = " ".join((
        f"TASK7_ACTION={action}",
        "exec",
        shlex.quote(REMOTE_PYTHON),
        shlex.quote(f"{REMOTE_CURRENT_SOURCE}/tools/{Path(__file__).name}"),
        "_remote-action",
        shlex.quote(action),
        *(shlex.quote(value) for value in arguments),
    ))
    return build_ssh_command(["bash", "-lc", script])


def _inventory_rows_from_result(result) -> list[dict]:
    try:
        rows = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("GPU inventory returned invalid JSON") from error
    classify_gpu_preflight(rows)
    return rows


def _same_gpu_inventory(before: list[dict], after: list[dict]) -> bool:
    return [
        (row["index"], row["uuid"]) for row in before
    ] == [
        (row["index"], row["uuid"]) for row in after
    ]


def _matches_frozen_gpu_inventory(
    rows: list[dict],
    *,
    gpu_indices: list[int],
    gpu_uuids: list[str],
) -> bool:
    return [
        (row["index"], row["uuid"]) for row in rows
    ] == list(zip(gpu_indices, gpu_uuids))


def _copy_partial(
    *,
    run_tag: str,
    command_runner=subprocess.run,
) -> None:
    _run_command(
        _action_command("partial-copy", [run_tag]),
        command_runner=command_runner,
        context="partial evidence controller copy",
        text=True,
        capture_output=True,
    )


def _failed_epoch_result(
    *,
    identity: str,
    reason: str,
    completed_epochs: list[str],
    primary: str,
    controller: str,
) -> dict:
    return {
        "status": "FAILED",
        "failed_epoch": identity,
        "reason": reason,
        "completed_epochs": completed_epochs,
        "primary_run": primary,
        "controller_run": controller,
    }


def run_bundle(
    *,
    run_tag: str,
    command_runner=subprocess.run,
    now=None,
    repo_root: Path | None = None,
    target_model: str = DEFAULT_TARGET_MODEL,
    draft_model: str = DEFAULT_DRAFT_MODEL,
) -> dict:
    tag = validate_run_tag(run_tag)
    preflight = run_preflight(
        run_tag=tag,
        command_runner=command_runner,
        now=now,
        repo_root=repo_root,
    )
    if preflight.get("status") != "READY":
        return preflight
    primary = preflight["primary_run"]
    controller = preflight["controller_run"]
    gpu_indices = preflight["gpu_indices"]
    gpu_uuids = preflight["gpu_uuids"]
    root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else Path(repo_root)
    )
    source_patch = _local_source_patch(
        repo_root=root,
        command_runner=command_runner,
    )
    _run_command(
        _action_command(
            "prepare",
            [tag, preflight["source_commit"], json.dumps(preflight)],
        ),
        command_runner=command_runner,
        context="remote bundle preparation",
        input=source_patch,
        capture_output=True,
    )
    completed_epochs = []
    for epoch_index, (block, mode, position) in enumerate(
        build_epoch_schedule()
    ):
        identity = f"{block}:{mode}:{position}"
        before_result = _run_command(
            _action_command(
                "inventory-before",
                [tag, str(epoch_index)],
            ),
            command_runner=command_runner,
            context=f"GPU inventory before {identity}",
            allow_failure=True,
            text=True,
            capture_output=True,
        )
        if before_result.returncode != 0:
            _copy_partial(
                run_tag=tag,
                command_runner=command_runner,
            )
            return _failed_epoch_result(
                identity=identity,
                reason="GPU inventory before epoch failed",
                completed_epochs=completed_epochs,
                primary=primary,
                controller=controller,
            )
        try:
            before = _inventory_rows_from_result(before_result)
        except (RuntimeError, ValueError):
            _copy_partial(
                run_tag=tag,
                command_runner=command_runner,
            )
            return _failed_epoch_result(
                identity=identity,
                reason="GPU inventory before epoch is invalid",
                completed_epochs=completed_epochs,
                primary=primary,
                controller=controller,
            )
        if not _matches_frozen_gpu_inventory(
            before,
            gpu_indices=gpu_indices,
            gpu_uuids=gpu_uuids,
        ):
            _copy_partial(
                run_tag=tag,
                command_runner=command_runner,
            )
            return _failed_epoch_result(
                identity=identity,
                reason="selected GPU inventory changed",
                completed_epochs=completed_epochs,
                primary=primary,
                controller=controller,
            )
        worker_result = _run_command(
            _action_command(
                "epoch",
                [
                    tag,
                    str(epoch_index),
                    mode,
                    position,
                    json.dumps(gpu_indices),
                    json.dumps(gpu_uuids),
                    target_model,
                    draft_model,
                ],
            ),
            command_runner=command_runner,
            context=f"worker epoch {identity}",
            allow_failure=True,
            text=True,
            capture_output=True,
        )
        if worker_result.returncode != 0:
            _copy_partial(
                run_tag=tag,
                command_runner=command_runner,
            )
            return _failed_epoch_result(
                identity=identity,
                reason="worker epoch failed",
                completed_epochs=completed_epochs,
                primary=primary,
                controller=controller,
            )
        after_result = _run_command(
            _action_command(
                "inventory-after",
                [tag, str(epoch_index)],
            ),
            command_runner=command_runner,
            context=f"GPU inventory after {identity}",
            allow_failure=True,
            text=True,
            capture_output=True,
        )
        if after_result.returncode != 0:
            _copy_partial(
                run_tag=tag,
                command_runner=command_runner,
            )
            return _failed_epoch_result(
                identity=identity,
                reason="GPU inventory after epoch failed",
                completed_epochs=completed_epochs,
                primary=primary,
                controller=controller,
            )
        try:
            after = _inventory_rows_from_result(after_result)
        except (RuntimeError, ValueError):
            _copy_partial(
                run_tag=tag,
                command_runner=command_runner,
            )
            return _failed_epoch_result(
                identity=identity,
                reason="GPU inventory after epoch is invalid",
                completed_epochs=completed_epochs,
                primary=primary,
                controller=controller,
            )
        if (
            not _same_gpu_inventory(before, after)
            or not _matches_frozen_gpu_inventory(
                after,
                gpu_indices=gpu_indices,
                gpu_uuids=gpu_uuids,
            )
        ):
            _copy_partial(
                run_tag=tag,
                command_runner=command_runner,
            )
            return _failed_epoch_result(
                identity=identity,
                reason="selected GPU inventory changed",
                completed_epochs=completed_epochs,
                primary=primary,
                controller=controller,
            )
        completed_epochs.append(identity)
    controller_created = False
    for action, context in (
        ("assemble", "canonical artifact assembly"),
        ("pre-manifest-verify", "pre-manifest verification"),
        ("manifest", "checksum manifest creation"),
        ("primary-verify", "primary remote verification"),
        ("controller-copy", "controller no-overwrite copy"),
        ("controller-verify", "controller verification"),
        ("compare-receipts", "verification receipt comparison"),
    ):
        result = _run_command(
            _action_command(action, [tag]),
            command_runner=command_runner,
            context=context,
            allow_failure=True,
            text=True,
            capture_output=True,
        )
        if action == "controller-copy" and result.returncode == 0:
            controller_created = True
        if result.returncode != 0:
            if not controller_created:
                _copy_partial(
                    run_tag=tag,
                    command_runner=command_runner,
                )
            return {
                "status": "FAILED",
                "failed_action": action,
                "completed_epochs": completed_epochs,
                "primary_run": primary,
                "controller_run": controller,
            }
    return {
        "status": "PASS",
        "completed_epochs": completed_epochs,
        "primary_run": primary,
        "controller_run": controller,
    }


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical_json_bytes(payload))


def _source_hashes(source_root: Path) -> dict[str, str]:
    hashes = {}
    for path, archive_name, is_directory in _source_inventory(source_root):
        if is_directory:
            continue
        relative = archive_name.removeprefix("source/")
        hashes[relative] = _sha256_path(path)
    return dict(sorted(hashes.items()))


def _remote_gpu_rows() -> list[dict]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if gpu.returncode != 0 or apps.returncode != 0:
        raise RuntimeError("GPU inventory query failed")
    processes = {}
    for line in apps.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 3:
            processes.setdefault(fields[1], []).append({
                "pid": int(fields[0]),
                "process_name": fields[2],
            })
    rows = []
    for line in gpu.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 4:
            raise ValueError("GPU inventory row is invalid")
        rows.append({
            "index": int(fields[0]),
            "uuid": fields[1],
            "memory_used_mib": int(fields[2]),
            "utilization_percent": int(fields[3]),
            "compute_processes": processes.get(fields[1], []),
        })
    return rows


def _remote_prepare(arguments: list[str]) -> int:
    if len(arguments) != 3:
        raise ValueError("prepare arguments are invalid")
    tag = validate_run_tag(arguments[0])
    source_commit = arguments[1]
    preflight = json.loads(arguments[2])
    source_patch = sys.stdin.buffer.read()
    primary = Path(primary_run_path(tag))
    controller = Path(controller_run_path(tag))
    if primary.exists() or controller.exists():
        raise ValueError("remote destination already exists")
    primary.mkdir(parents=True, exist_ok=False)
    for relative in ("workers", "telemetry", "logs", "status"):
        (primary / relative).mkdir()
    archive_path = primary / "source.tar"
    build_source_archive(Path(REMOTE_CURRENT_SOURCE), archive_path)
    frozen_source = extract_source_archive(
        archive_path,
        primary / "source-extract",
    )
    frozen_source.rename(primary / "source")
    (primary / "source-extract").rmdir()
    source_hashes = _source_hashes(primary / "source")
    source_tree_sha256 = hashlib.sha256(
        _canonical_json_bytes(source_hashes)
    ).hexdigest()
    _write_json_exclusive(primary / "source_manifest.json", source_hashes)
    with (primary / "source.patch").open("xb") as handle:
        handle.write(source_patch)
    metadata = {
        "configuration": {
            "tensor_parallel_size": 4,
            "batch_size": 4,
            "max_proposal_tokens": 4,
            "prompt_tokens": 256,
            "output_tokens": 16,
            "temperature": 0.0,
            "proposal_kv_allocator": "direct",
            "proposal_kv_offload": False,
            "measured_runs_per_epoch": 5,
            "measured_runs_total": 40,
        },
        "provenance": {
            "run_tag": tag,
            "source_commit": source_commit,
            "source_tree_sha256": source_tree_sha256,
            "gpu_uuids": preflight["gpu_uuids"],
        },
    }
    _write_json_exclusive(primary / "metadata.json", metadata)
    _write_json_exclusive(primary / "preflight.json", preflight)
    return 0


def _remote_inventory(arguments: list[str], phase: str) -> int:
    if len(arguments) != 2:
        raise ValueError("inventory arguments are invalid")
    tag = validate_run_tag(arguments[0])
    epoch_index = int(arguments[1])
    if epoch_index < 0 or epoch_index >= len(build_epoch_schedule()):
        raise ValueError("epoch index is invalid")
    status_root = Path(primary_run_path(tag)) / "status"
    stem = status_root / f"epoch-{epoch_index}-{phase}"
    try:
        rows = _remote_gpu_rows()
        selected = [
            row
            for row in rows
            if row["index"] in json.loads(
                (
                    Path(primary_run_path(tag)) / "preflight.json"
                ).read_text()
            )["gpu_indices"]
        ]
        classify_gpu_preflight(selected)
        _write_json_exclusive(
            stem.with_suffix(".json"),
            selected,
        )
        stem.with_suffix(".status").write_text("0\n", encoding="utf-8")
        sys.stdout.write(json.dumps(selected, separators=(",", ":")) + "\n")
        return 0
    except Exception:
        stem.with_suffix(".status").write_text("1\n", encoding="utf-8")
        stem.with_suffix(".stderr.log").write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        stem.with_suffix(".stdout.log").write_text("", encoding="utf-8")
        raise


def _terminate_and_reap(processes) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)


def _start_epoch_samplers(
    *,
    gpu_indices: list[int],
    gpu_path: Path,
    host_path: Path,
) -> tuple[list[subprocess.Popen], list[object]]:
    gpu_script = "\n".join((
        "import json,subprocess,sys,time",
        "indices=json.loads(sys.argv[1])",
        "while True:",
        " unix_ns=time.time_ns()",
        " monotonic_ns=time.monotonic_ns()",
        " result=subprocess.run([",
        "  'nvidia-smi',",
        "  '--query-gpu=index,uuid',",
        "  '--format=csv,noheader,nounits'],",
        "  capture_output=True,text=True,check=False)",
        " if result.returncode:",
        "  raise SystemExit(result.returncode)",
        " for line in result.stdout.splitlines():",
        "  fields=[part.strip() for part in line.split(',')]",
        "  if len(fields)==2 and int(fields[0]) in indices:",
        "   print(json.dumps({",
        "    'sampled_at_unix_ns':unix_ns,",
        "    'sampled_at_monotonic_ns':monotonic_ns,",
        "    'gpu_index':int(fields[0]),",
        "    'gpu_uuid':fields[1]},",
        "    sort_keys=True,separators=(',',':')),flush=True)",
        " time.sleep(0.05)",
    ))
    host_script = "\n".join((
        "import json,time",
        "while True:",
        " print(json.dumps({",
        "  'sampled_at_unix_ns':time.time_ns(),",
        "  'sampled_at_monotonic_ns':time.monotonic_ns()},",
        "  sort_keys=True,separators=(',',':')),flush=True)",
        " time.sleep(0.05)",
    ))
    handles = [
        gpu_path.open("xb"),
        host_path.open("xb"),
        gpu_path.with_name(f"{gpu_path.name}.stderr").open("xb"),
        host_path.with_name(f"{host_path.name}.stderr").open("xb"),
    ]
    processes = []
    try:
        processes.append(subprocess.Popen(
            [sys.executable, "-c", gpu_script, json.dumps(gpu_indices)],
            stdout=handles[0],
            stderr=handles[2],
        ))
        processes.append(subprocess.Popen(
            [sys.executable, "-c", host_script],
            stdout=handles[1],
            stderr=handles[3],
        ))
    except BaseException:
        _terminate_and_reap(processes)
        for handle in handles:
            handle.close()
        raise
    return processes, handles


def _load_json_lines(path: Path, *, name: str) -> list[dict]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"{name} is unreadable") from error
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} contains invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{name} row must be a mapping")
        rows.append(row)
    if not rows:
        raise ValueError(f"{name} is empty")
    return rows


def _attach_epoch_telemetry(
    worker: dict,
    *,
    gpu_path: Path,
    host_path: Path,
    gpu_uuids: list[str],
) -> dict:
    gpu_samples = _load_json_lines(gpu_path, name="GPU telemetry")
    host_samples = _load_json_lines(host_path, name="host telemetry")
    result = copy.deepcopy(worker)
    measured = result.get("measured_runs")
    if not isinstance(measured, list) or len(measured) != 5:
        raise ValueError("worker must contain five measured runs")
    for run in measured:
        interval = run.get("campaign_interval")
        repeat_identity = run.get("command_timeline_repeat_index")
        if not isinstance(interval, dict):
            raise ValueError("worker campaign interval is missing")
        start_unix = interval.get("started_at_unix_ns")
        finish_unix = interval.get("finished_at_unix_ns")
        start_monotonic = interval.get("started_at_monotonic_ns")
        finish_monotonic = interval.get("finished_at_monotonic_ns")
        gpu_rows = [
            {
                "repeat_index": repeat_identity,
                "sampled_at_unix_ns": row["sampled_at_unix_ns"],
                "sampled_at_monotonic_ns": row[
                    "sampled_at_monotonic_ns"
                ],
                "gpu_uuid": row["gpu_uuid"],
            }
            for row in gpu_samples
            if (
                start_unix <= row.get("sampled_at_unix_ns", -1)
                <= finish_unix
                and start_monotonic
                <= row.get("sampled_at_monotonic_ns", -1)
                <= finish_monotonic
            )
        ]
        host_rows = [
            {
                "repeat_index": repeat_identity,
                "sampled_at_unix_ns": row["sampled_at_unix_ns"],
                "sampled_at_monotonic_ns": row[
                    "sampled_at_monotonic_ns"
                ],
            }
            for row in host_samples
            if (
                start_unix <= row.get("sampled_at_unix_ns", -1)
                <= finish_unix
                and start_monotonic
                <= row.get("sampled_at_monotonic_ns", -1)
                <= finish_monotonic
            )
        ]
        if (
            set(row["gpu_uuid"] for row in gpu_rows) != set(gpu_uuids)
            or not host_rows
        ):
            raise ValueError("telemetry coverage is incomplete")
        run["telemetry"] = {
            "gpu_rows": gpu_rows,
            "host_rows": host_rows,
        }
    return result


def _remote_epoch(arguments: list[str]) -> int:
    if len(arguments) != 8:
        raise ValueError("epoch arguments are invalid")
    tag = validate_run_tag(arguments[0])
    epoch_index = int(arguments[1])
    mode = arguments[2]
    position = arguments[3]
    gpu_indices = json.loads(arguments[4])
    gpu_uuids = json.loads(arguments[5])
    target_model = arguments[6]
    draft_model = arguments[7]
    block, expected_mode, expected_position = build_epoch_schedule()[
        epoch_index
    ]
    if (mode, position) != (expected_mode, expected_position):
        raise ValueError("epoch identity is invalid")
    primary = Path(primary_run_path(tag))
    source_root = primary / "source"
    worker_dir = primary / "workers" / block
    telemetry_dir = primary / "telemetry" / block
    worker_dir.mkdir(parents=True, exist_ok=True)
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    raw_path = worker_dir / f"{mode}.raw.json"
    worker_path = worker_dir / f"{mode}.json"
    stdout_path = worker_dir / f"{mode}.stdout.log"
    stderr_path = worker_dir / f"{mode}.stderr.log"
    status_path = worker_dir / f"{mode}.status"
    invariant_status_path = worker_dir / f"{mode}.invariant.status"
    invariant_stderr_path = worker_dir / f"{mode}.invariant.stderr.log"
    gpu_telemetry_path = telemetry_dir / f"{mode}.gpu.jsonl"
    host_telemetry_path = telemetry_dir / f"{mode}.host.jsonl"
    command = build_epoch_worker_command(
        source_root=str(source_root),
        output_path=str(raw_path),
        target_model=target_model,
        draft_model=draft_model,
        mode=mode,
        gpu_indices=gpu_indices,
    )
    samplers = []
    sampler_handles = []
    worker = None
    try:
        invariant_status_path.write_text("125\n", encoding="utf-8")
        samplers, sampler_handles = _start_epoch_samplers(
            gpu_indices=gpu_indices,
            gpu_path=gpu_telemetry_path,
            host_path=host_telemetry_path,
        )
        (worker_dir / f"{mode}.owned-pids").write_text(
            "\n".join(str(process.pid) for process in samplers) + "\n",
            encoding="utf-8",
        )
        with (
            stdout_path.open("xb") as stdout_handle,
            stderr_path.open("xb") as stderr_handle,
        ):
            worker = subprocess.Popen(
                command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            with (worker_dir / f"{mode}.owned-pids").open(
                "a",
                encoding="utf-8",
            ) as pid_handle:
                pid_handle.write(f"{worker.pid}\n")
            returncode = worker.wait()
        status_path.write_text(f"{returncode}\n", encoding="utf-8")
        if returncode != 0:
            return returncode
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        _terminate_and_reap(samplers)
        for handle in sampler_handles:
            handle.close()
        samplers = []
        sampler_handles = []
        raw = _attach_epoch_telemetry(
            raw,
            gpu_path=gpu_telemetry_path,
            host_path=host_telemetry_path,
            gpu_uuids=gpu_uuids,
        )
        metadata = json.loads(
            (primary / "metadata.json").read_text(encoding="utf-8")
        )
        provenance = metadata["provenance"]
        augmented = augment_worker_payload(
            raw,
            source_commit=provenance["source_commit"],
            source_tree_sha256=provenance["source_tree_sha256"],
            target_checkpoint_identifier=Path(target_model).name,
            draft_checkpoint_identifier=Path(draft_model).name,
            tokenizer_identifier=raw["tokenizer_identifier"],
            gpu_uuids=gpu_uuids,
        )
        _write_json_exclusive(worker_path, augmented)
        epoch_key = (
            f"b{block.removeprefix('block-')}-{mode}-{position}"
        )
        _write_json_exclusive(
            telemetry_dir / f"{mode}.json",
            derive_telemetry_sidecar(epoch_key, augmented),
        )
        invariant_status_path.write_text("0\n", encoding="utf-8")
        return 0
    except Exception:
        invariant_status_path.write_text("1\n", encoding="utf-8")
        invariant_stderr_path.write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        if not stdout_path.exists():
            stdout_path.write_text("", encoding="utf-8")
        if not stderr_path.exists():
            stderr_path.write_text("", encoding="utf-8")
        raise
    finally:
        owned = [process for process in (*samplers, worker) if process]
        _terminate_and_reap(owned)
        for handle in sampler_handles:
            handle.close()


def _remote_copy(tag: str, *, partial: bool) -> int:
    primary = Path(primary_run_path(tag))
    controller = Path(controller_run_path(tag))
    if controller.exists():
        raise ValueError("controller destination already exists")
    import shutil

    shutil.copytree(primary, controller, symlinks=False)
    if partial:
        _write_json_exclusive(
            controller / "partial-evidence.json",
            {"partial": True, "source": str(primary)},
        )
    return 0


def _raw_input_files(primary: Path) -> dict[str, dict[str, str]]:
    inventory = {
        "metadata": {
            "path": "metadata.json",
            "sha256": _sha256_path(primary / "metadata.json"),
        },
        "source_manifest": {
            "path": "source_manifest.json",
            "sha256": _sha256_path(primary / "source_manifest.json"),
        },
    }
    for block, mode, position in build_epoch_schedule():
        block_index = int(block.removeprefix("block-"))
        epoch_key = f"b{block_index}-{mode}-{position}"
        worker_relative = f"workers/{block}/{mode}.json"
        telemetry_relative = f"telemetry/{block}/{mode}.json"
        inventory[f"worker:{epoch_key}"] = {
            "path": worker_relative,
            "sha256": _sha256_path(primary / worker_relative),
        }
        inventory[f"telemetry:{epoch_key}"] = {
            "path": telemetry_relative,
            "sha256": _sha256_path(primary / telemetry_relative),
        }
    return inventory


def _remote_assemble(tag: str) -> int:
    from autoregressive_draft_command_timeline_diagnostic import (
        build_command_timeline_artifact,
        canonical_json_bytes,
        expected_epoch_identities,
    )

    primary = Path(primary_run_path(tag))
    metadata = json.loads(
        (primary / "metadata.json").read_text(encoding="utf-8")
    )
    source_files = json.loads(
        (primary / "source_manifest.json").read_text(encoding="utf-8")
    )
    epoch_inputs = {}
    for identity in expected_epoch_identities():
        worker_path = (
            primary
            / "workers"
            / f"block-{identity.block_index}"
            / f"{identity.label}.json"
        )
        telemetry_path = (
            primary
            / "telemetry"
            / f"block-{identity.block_index}"
            / f"{identity.label}.json"
        )
        epoch_inputs[identity.key] = {
            "worker": json.loads(worker_path.read_text(encoding="utf-8")),
            "telemetry": json.loads(
                telemetry_path.read_text(encoding="utf-8")
            ),
        }
    artifact = build_command_timeline_artifact(
        metadata=metadata,
        epoch_raw_inputs=epoch_inputs,
        input_files=_raw_input_files(primary),
        source_files=source_files,
    )
    artifact_path = primary / "command-timeline.json"
    with artifact_path.open("xb") as handle:
        handle.write(canonical_json_bytes(artifact))
    classification = artifact["classification"]
    _write_json_exclusive(primary / "result.json", {
        "artifact_sha256": _sha256_path(artifact_path),
        "classification": classification,
        "localized_boundary": artifact["localized_boundary"],
        "runtime_optimization_authorized": (
            classification == "BOUNDARY_LOCALIZED"
        ),
        "performance_improvement_established": False,
        "phase_1_complete": False,
        "promotion_ready": False,
    })
    return 0


def _verify_bundle(
    *,
    artifact_root: Path,
    source_root: Path,
    manifest: bool,
    receipt_path: Path | None,
    verification_location: str,
) -> dict:
    from verify_autoregressive_draft_command_timeline_diagnostic import (
        verify_command_timeline_diagnostic,
    )

    receipt = verify_command_timeline_diagnostic(
        artifact_path=artifact_root / "command-timeline.json",
        source_root=source_root,
        manifest_path=(
            artifact_root / "manifest.sha256" if manifest else None
        ),
    )
    receipt["verification_location"] = verification_location
    if receipt_path is not None:
        _write_json_exclusive(receipt_path, receipt)
    return receipt


def _remote_manifest(tag: str) -> int:
    primary = Path(primary_run_path(tag))
    rows = []
    for path in sorted(primary.rglob("*")):
        relative = path.relative_to(primary).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("manifest inventory contains a symlink")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("manifest inventory entry is not regular")
        if relative in DETACHED_ATTESTATION_PATHS:
            continue
        rows.append(f"{_sha256_path(path)}  {relative}")
    manifest_path = primary / "manifest.sha256"
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")
    return 0


def _remote_primary_verify(tag: str) -> int:
    primary = Path(primary_run_path(tag))
    _verify_bundle(
        artifact_root=primary,
        source_root=primary / "source",
        manifest=True,
        receipt_path=primary / "verify.command-timeline.remote.json",
        verification_location="remote",
    )
    return 0


def _remote_controller_verify(tag: str) -> int:
    controller = Path(controller_run_path(tag))
    _verify_bundle(
        artifact_root=controller,
        source_root=Path(REMOTE_CURRENT_SOURCE),
        manifest=True,
        receipt_path=controller / "verify.command-timeline.local.json",
        verification_location="local",
    )
    return 0


def _remote_compare_receipts(tag: str) -> int:
    primary = Path(primary_run_path(tag))
    controller = Path(controller_run_path(tag))
    remote = json.loads(
        (primary / "verify.command-timeline.remote.json").read_text(
            encoding="utf-8"
        )
    )
    local = json.loads(
        (controller / "verify.command-timeline.local.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        _canonical_json_bytes(normalize_verification_receipt(remote))
        != _canonical_json_bytes(normalize_verification_receipt(local))
    ):
        raise ValueError("verification receipts differ")
    return 0


def _remote_action(action: str, arguments: list[str]) -> int:
    if action == "prepare":
        return _remote_prepare(arguments)
    if action == "inventory-before":
        return _remote_inventory(arguments, "before")
    if action == "inventory-after":
        return _remote_inventory(arguments, "after")
    if action == "epoch":
        return _remote_epoch(arguments)
    if action == "partial-copy":
        return _remote_copy(validate_run_tag(arguments[0]), partial=True)
    if action == "controller-copy":
        return _remote_copy(validate_run_tag(arguments[0]), partial=False)
    if len(arguments) != 1:
        raise ValueError("remote action arguments are invalid")
    tag = validate_run_tag(arguments[0])
    if action == "assemble":
        return _remote_assemble(tag)
    if action == "pre-manifest-verify":
        primary = Path(primary_run_path(tag))
        _verify_bundle(
            artifact_root=primary,
            source_root=primary / "source",
            manifest=False,
            receipt_path=None,
            verification_location="remote",
        )
        return 0
    if action == "manifest":
        return _remote_manifest(tag)
    if action == "primary-verify":
        return _remote_primary_verify(tag)
    if action == "controller-verify":
        return _remote_controller_verify(tag)
    if action == "compare-receipts":
        return _remote_compare_receipts(tag)
    raise ValueError("remote action is invalid")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "execute", "verify-local"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--run-tag", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments and arguments[0] == "_remote-action":
        if len(arguments) < 2:
            raise ValueError("remote action is missing")
        return _remote_action(arguments[1], arguments[2:])
    args = parse_args(arguments)
    if args.command == "preflight":
        result = run_preflight(run_tag=args.run_tag)
    elif args.command == "execute":
        result = run_bundle(run_tag=args.run_tag)
    else:
        validate_run_tag(args.run_tag)
        result = _run_command(
            _action_command("controller-verify", [args.run_tag]),
            context="controller verification",
            text=True,
            capture_output=True,
        )
        result = {"status": "PASS", "stdout": result.stdout}
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0 if result.get("status") in ("READY", "PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
