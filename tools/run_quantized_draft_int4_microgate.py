#!/usr/bin/env python3
"""Fail-closed local controller for the fused INT4 draft Stage-0 gate."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tarfile
import time

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from tools.run_autoregressive_draft_cuda_graph_gate_remote import (
    _local_kerberos_preflight,
)
from tools.verify_quantized_draft_int4_microgate import verify_bundle


APPROVED_REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
REMOTE_TARGET = "sitian@10.232.195.203"
REMOTE_PYTHON = "/data00/home/sitian/tllm/env/bin/python"
SSH_CONTROL_PATH = "/tmp/ssh-sitian-10.232.195.203"
REMOTE_PACKAGE_ROOT = (
    "/data00/home/sitian/tllm/env/lib/python3.11/site-packages"
)
DRAFT_MODEL = "/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B"
MINIMUM_KERBEROS_LIFETIME_SECONDS = 5400
MAXIMUM_IDLE_MEMORY_MIB = 1024
MAXIMUM_IDLE_UTILIZATION_PERCENT = 5
SOURCE_PATHS = (
    "tinyvllm/layers/fused_int4_linear.py",
    "tinyvllm/layers/quantization.py",
    "tinyvllm/layers/linear.py",
    "tools/quantized_draft_int4_microgate.py",
    "tools/quantized_draft_int4_microgate_worker.py",
    "tools/assemble_quantized_draft_int4_microgate.py",
    "tools/verify_quantized_draft_int4_microgate.py",
)
_RUN_TAG = re.compile(r"^[A-Za-z0-9_-]+$")


def _canonical(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(_canonical(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


def _below(path: PurePosixPath, root: PurePosixPath) -> bool:
    return path != root and root in path.parents


def build_run_plan(
    *,
    run_tag: str,
    source_revision: str,
    remote_root: str = APPROVED_REMOTE_ROOT,
    remote_python: str = REMOTE_PYTHON,
    draft_model: str = DRAFT_MODEL,
) -> dict[str, object]:
    if not _RUN_TAG.fullmatch(str(run_tag)):
        raise ValueError("run_tag is invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", str(source_revision)):
        raise ValueError("source_revision must be a full SHA")
    root = PurePosixPath(remote_root)
    if str(root) != APPROVED_REMOTE_ROOT:
        raise ValueError("remote_root must equal approved remote root")
    run = root / "quantized-draft-int4" / run_tag
    source = run / "source"
    cache = run / "cache"
    for path in (run, source, cache):
        if not _below(path, root):
            raise ValueError("path escapes approved remote root")
    return {
        "schema_version": 1,
        "run_tag": run_tag,
        "source_revision": source_revision,
        "remote_root": str(root),
        "remote_run": str(run),
        "remote_source": str(source),
        "remote_cache": str(cache),
        "remote_raw": str(run / "raw"),
        "remote_final_bundle": str(run / "final_bundle"),
        "remote_controller": str(run / "controller"),
        "remote_python": remote_python,
        "draft_model": draft_model,
        "source_paths": list(SOURCE_PATHS),
    }


def classify_preflight(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {"status": "INCONCLUSIVE_ENVIRONMENT"}
    if (
        payload.get("python_exists") is not True
        or payload.get("draft_model_exists") is not True
        or payload.get("remote_root_exists") is not True
        or payload.get("exact_tag_exists") is not False
        or payload.get("exact_tag_processes") != []
    ):
        return {
            "status": "INCONCLUSIVE_ENVIRONMENT",
            "reason": "remote prerequisite or exact-tag isolation failed",
        }
    gpus = payload.get("gpus")
    if not isinstance(gpus, list):
        return {"status": "INCONCLUSIVE_ENVIRONMENT"}
    eligible = []
    for row in gpus:
        if not isinstance(row, dict):
            continue
        values = (
            row.get("index"),
            row.get("memory_used_mib"),
            row.get("utilization_percent"),
            row.get("compute_process_count"),
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        ):
            continue
        if (
            "A100" in str(row.get("name", ""))
            and row["memory_used_mib"] <= MAXIMUM_IDLE_MEMORY_MIB
            and row["utilization_percent"]
            <= MAXIMUM_IDLE_UTILIZATION_PERCENT
            and row["compute_process_count"] == 0
        ):
            eligible.append(row)
    if not eligible:
        return {
            "status": "WAIT_GPU",
            "reason": "no clean A100 is currently available",
        }
    selected = min(eligible, key=lambda row: row["index"])
    return {
        "status": "READY",
        "selected_gpu_index": selected["index"],
        "selected_gpu_uuid": selected.get("uuid"),
    }


def _ssh(arguments: list[str]) -> list[str]:
    return [
        "ssh",
        "-S",
        SSH_CONTROL_PATH,
        "-o",
        "BatchMode=yes",
        "-o",
        "ProxyCommand=ssh -qW %h:%p jump-proxy-lf",
        REMOTE_TARGET,
        shlex.join(arguments),
    ]


def build_remote_commands(
    plan: dict[str, object],
    *,
    selected_gpu_index: int,
) -> tuple[str, ...]:
    source = str(plan["remote_source"])
    cache = str(plan["remote_cache"])
    python = str(plan["remote_python"])
    environment = [
        "env",
        f"CUDA_VISIBLE_DEVICES={selected_gpu_index}",
        f"PYTHONPATH={source}:{REMOTE_PACKAGE_ROOT}",
        f"TRITON_CACHE_DIR={cache}/triton",
        f"TORCHINDUCTOR_CACHE_DIR={cache}/torchinductor",
        f"XDG_CACHE_HOME={cache}/xdg",
        f"CUDA_CACHE_PATH={cache}/cuda",
    ]
    worker = environment + [
        python,
        f"{source}/tools/quantized_draft_int4_microgate_worker.py",
        "--model-path",
        str(plan["draft_model"]),
        "--output-dir",
        str(plan["remote_raw"]),
        "--approved-remote-root",
        str(plan["remote_root"]),
    ]
    assembler = [
        python,
        f"{source}/tools/assemble_quantized_draft_int4_microgate.py",
        "--raw-dir",
        str(plan["remote_raw"]),
        "--output-dir",
        str(plan["remote_final_bundle"]),
        "--source-revision",
        str(plan["source_revision"]),
        "--run-tag",
        str(plan["run_tag"]),
    ]
    verifier = [
        python,
        f"{source}/tools/verify_quantized_draft_int4_microgate.py",
        str(plan["remote_final_bundle"]),
    ]
    return tuple(shlex.join(command) for command in (
        worker,
        assembler,
        verifier,
    ))


def download_inventory(run_tag: str) -> tuple[str, ...]:
    if not _RUN_TAG.fullmatch(str(run_tag)):
        raise ValueError("run_tag is invalid")
    return ("controller", "final_bundle")


def _validate_local_source(repo_root: Path) -> str:
    def run(*arguments):
        return subprocess.run(
            list(arguments),
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )

    branch = run("git", "branch", "--show-current")
    if branch.returncode or branch.stdout.strip() != "feat/kv-sparse-attention":
        raise ValueError("local branch is not feat/kv-sparse-attention")
    head = run("git", "rev-parse", "HEAD").stdout.strip()
    remote = run(
        "git",
        "ls-remote",
        "origin",
        "refs/heads/feat/kv-sparse-attention",
    )
    remote_sha = remote.stdout.split()[0] if remote.stdout.split() else ""
    if not re.fullmatch(r"[0-9a-f]{40}", head) or head != remote_sha:
        raise ValueError("local and remote source revisions do not match")
    dirty = run("git", "status", "--porcelain", "--", *SOURCE_PATHS)
    if dirty.returncode or dirty.stdout.strip():
        raise ValueError("tracked source paths are not clean")
    return head


def _kerberos_preflight() -> dict[str, object]:
    return _local_kerberos_preflight()


def _remote_preflight(plan: dict[str, object]) -> dict[str, object]:
    script = (
        "import json,os,pathlib,subprocess\n"
        f"run={str(plan['remote_run'])!r}\n"
        "gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,"
        "memory.used,utilization.gpu','--format=csv,noheader,nounits'],"
        "text=True,capture_output=True,check=True).stdout\n"
        "apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid',"
        "'--format=csv,noheader,nounits'],text=True,capture_output=True,"
        "check=False).stdout\n"
        "counts={}\n"
        "for line in apps.splitlines():\n"
        " u=line.strip()\n"
        " if u: counts[u]=counts.get(u,0)+1\n"
        "rows=[]\n"
        "for line in gpu.splitlines():\n"
        " p=[x.strip() for x in line.split(',')]\n"
        " rows.append({'index':int(p[0]),'uuid':p[1],'name':p[2],"
        "'memory_used_mib':int(p[3]),'utilization_percent':int(p[4]),"
        "'compute_process_count':counts.get(p[1],0)})\n"
        "ps=subprocess.run(['ps','-eo','pid=,args='],text=True,"
        "capture_output=True,check=True).stdout\n"
        f"tag={str(plan['run_tag'])!r}\n"
        "owned=[line.strip() for line in ps.splitlines() if tag in line "
        "and 'quantized_draft_int4_microgate_worker.py' in line "
        "and not line.lstrip().startswith(str(os.getpid())+' ')]\n"
        "print(json.dumps({'python_exists':pathlib.Path("
        f"{str(plan['remote_python'])!r}).is_file(),"
        "'draft_model_exists':pathlib.Path("
        f"{str(plan['draft_model'])!r}).is_dir(),"
        "'remote_root_exists':pathlib.Path("
        f"{str(plan['remote_root'])!r}).is_dir(),"
        "'exact_tag_exists':pathlib.Path(run).exists(),"
        "'exact_tag_processes':owned,'gpus':rows},sort_keys=True))\n"
    )
    completed = subprocess.run(
        _ssh([str(plan["remote_python"]), "-c", script]),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "remote preflight failed: "
            + (completed.stderr or completed.stdout).strip()
        )
    payload = json.loads(completed.stdout)
    return {**payload, **classify_preflight(payload)}


def _source_archive(repo_root: Path) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for relative in SOURCE_PATHS:
            path = repo_root / relative
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"invalid source path: {relative}")
            archive.add(path, arcname=relative, recursive=False)
    return stream.getvalue()


def _upload_source(
    plan: dict[str, object],
    repo_root: Path,
) -> None:
    command = _ssh([
        "bash",
        "-lc",
        (
            f"test ! -e {shlex.quote(str(plan['remote_run']))} && "
            f"mkdir -p {shlex.quote(str(plan['remote_source']))} "
            f"{shlex.quote(str(plan['remote_cache']))} "
            f"{shlex.quote(str(plan['remote_controller']))} && "
            f"tar -xf - -C {shlex.quote(str(plan['remote_source']))}"
        ),
    ])
    completed = subprocess.run(
        command,
        input=_source_archive(repo_root),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("remote source upload failed")


def _run_remote(command: str) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        _ssh(["bash", "-lc", command]),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            (completed.stderr or completed.stdout or "").strip()
        )
    return completed


def _download_bundle(plan: dict[str, object], local_root: Path) -> None:
    completed = subprocess.run(
        _ssh([
            "tar",
            "-cf",
            "-",
            "-C",
            str(plan["remote_run"]),
            "final_bundle",
        ]),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("compact bundle download failed")
    local_root.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe download path")
        archive.extractall(local_root, filter="data")


def run_controller(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    source_revision = _validate_local_source(repo_root)
    plan = build_run_plan(
        run_tag=args.run_tag,
        source_revision=source_revision,
    )
    kerberos = _kerberos_preflight()
    if (
        kerberos.get("status") != "READY"
        or kerberos.get("remaining_lifetime_seconds", 0)
        < MINIMUM_KERBEROS_LIFETIME_SECONDS
    ):
        return 2

    deadline = time.monotonic() + max(0, args.max_wait_seconds)
    while True:
        preflight = _remote_preflight(plan)
        if preflight["status"] != "WAIT_GPU":
            break
        if args.dry_run or time.monotonic() >= deadline:
            return 3
        time.sleep(max(1, args.poll_seconds))
    if preflight["status"] != "READY":
        return 4
    if args.dry_run:
        print("READY", flush=True)
        return 0

    local_root = Path(args.local_artifact_root).resolve() / args.run_tag
    if local_root.exists():
        raise ValueError("local exact-tag artifact directory exists")
    _upload_source(plan, repo_root)
    commands = build_remote_commands(
        plan,
        selected_gpu_index=preflight["selected_gpu_index"],
    )
    launch_receipt = {}
    for index, command in enumerate(commands):
        completed = _run_remote(command)
        launch_receipt[f"command_{index}"] = {
            "command": command,
            "stdout": completed.stdout,
        }
    _download_bundle(plan, local_root)
    local_verification = verify_bundle(local_root / "final_bundle")
    controller_dir = local_root / "controller"
    _write_json(controller_dir / "plan.json", plan)
    _write_json(controller_dir / "source_identity.json", {
        "source_revision": source_revision,
        "run_tag": args.run_tag,
    })
    _write_json(controller_dir / "kerberos_preflight.json", kerberos)
    _write_json(controller_dir / "ssh_storage_preflight.json", {
        key: preflight.get(key)
        for key in (
            "python_exists",
            "draft_model_exists",
            "remote_root_exists",
            "exact_tag_exists",
        )
    })
    _write_json(controller_dir / "gpu_admission.json", preflight)
    _write_json(controller_dir / "launch.json", launch_receipt)
    _write_json(
        controller_dir / "remote_verification.json",
        json.loads(launch_receipt["command_2"]["stdout"]),
    )
    _write_json(controller_dir / "download.json", {
        "inventory": list(download_inventory(args.run_tag)),
    })
    _write_json(
        controller_dir / "local_verification.json",
        local_verification,
    )
    scans = []
    for _ in range(3):
        scan = _remote_preflight({
            **plan,
            "remote_run": str(plan["remote_run"]) + "-nonexistent-scan",
        })
        scans.append(scan.get("exact_tag_processes", []))
    _write_json(controller_dir / "remote_cleanup_scan.json", {
        "scans": scans,
        "classification": (
            "CLEAN" if all(not rows for rows in scans) else "DIRTY"
        ),
    })
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_REPOSITORY_ROOT,
    )
    parser.add_argument(
        "--local-artifact-root",
        type=Path,
        default=_REPOSITORY_ROOT / "artifacts" / "quantized_draft_int4",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--max-wait-seconds", type=int, default=7200)
    return parser


def main() -> int:
    return run_controller(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
