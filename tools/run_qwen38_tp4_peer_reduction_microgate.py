#!/usr/bin/env python3
"""Safely orchestrate the TP4 peer-reduction microgate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time

from tools.run_qwen38_tp4_communication_profile import (
    parse_nvidia_smi_inventory,
    query_local_kerberos,
    select_strict_clean_gpus,
    validate_selected_gpu_processes,
    wait_for_strict_clean_gpus,
    write_json_atomic,
)


APPROVED_REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
PLAN_SCHEMA_VERSION = "qwen38.tp4-peer-reduction-plan.v1"
ATTEMPT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_REMOTE_PYTHON = "/data00/home/sitian/tllm/env/bin/python"
DEFAULT_PROXY_HOST = "jump-proxy-lf"
DEFAULT_COMMAND_TIMEOUT_S = 60
DEFAULT_RETRY_COUNT = 3
DEFAULT_GPU_WAIT_TIMEOUT_S = 21_600
DEFAULT_GPU_POLL_INTERVAL_S = 15
DEFAULT_DIST_PORT = 29673
REMOTE_TOOLCHAIN_ROOT = (
    f"{APPROVED_REMOTE_ROOT}/preflight/"
    "20260830-tp4-peer-reduction-compile-r1/toolchain"
)


def _below(path, root):
    candidate = PurePosixPath(path)
    approved = PurePosixPath(root)
    return candidate.is_absolute() and candidate.is_relative_to(approved)


def _validate_remote_path_state(state):
    if (
        not isinstance(state, dict)
        or type(state.get("attempt_exists")) is not bool
        or type(state.get("attempt_parent_is_symlink")) is not bool
        or type(state.get("remote_root_is_symlink")) is not bool
    ):
        raise ValueError("remote path state is invalid")
    if (
        state["attempt_exists"]
        or state["attempt_parent_is_symlink"]
        or state["remote_root_is_symlink"]
    ):
        raise ValueError("attempt path must be fresh and non-symlinked")


def build_attempt_plan(
    *,
    attempt_tag,
    source_revision,
    selected_gpus,
    remote_path_state,
    remote_root=APPROVED_REMOTE_ROOT,
):
    if remote_root != APPROVED_REMOTE_ROOT:
        raise ValueError("remote_root is not approved")
    if (
        not isinstance(attempt_tag, str)
        or not ATTEMPT_PATTERN.fullmatch(attempt_tag)
        or ".." in attempt_tag
    ):
        raise ValueError("attempt tag is invalid")
    if (
        not isinstance(source_revision, str)
        or not REVISION_PATTERN.fullmatch(source_revision)
    ):
        raise ValueError("source revision is invalid")
    _validate_remote_path_state(remote_path_state)
    selected = select_strict_clean_gpus(selected_gpus)
    if len(selected) != 4:
        raise ValueError("four strict-clean GPUs are required")

    attempt_root = f"{remote_root}/attempts/{attempt_tag}"
    runtime_root = f"{attempt_root}/runtime"
    paths = {
        "attempt_root": attempt_root,
        "source_root": f"{attempt_root}/source",
        "raw_root": f"{attempt_root}/raw",
        "bundle_root": f"{attempt_root}/final_bundle",
        "controller_root": f"{attempt_root}/controller",
        "worker_stdout_path": f"{attempt_root}/controller/worker.stdout",
        "worker_stderr_path": f"{attempt_root}/controller/worker.stderr",
    }
    environment = {
        "TMPDIR": f"{runtime_root}/tmp",
        "XDG_CACHE_HOME": f"{runtime_root}/cache/xdg",
        "TORCH_EXTENSIONS_DIR": (
            f"{runtime_root}/cache/torch-extensions"
        ),
        "CUDA_CACHE_PATH": f"{runtime_root}/cache/cuda",
    }
    if not all(
        _below(path, remote_root)
        for path in (*paths.values(), *environment.values())
    ):
        raise ValueError("planned path escapes approved root")
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "attempt_tag": attempt_tag,
        "source_revision": source_revision,
        "remote_root": remote_root,
        **paths,
        "environment": environment,
        "selected_gpus": [dict(row) for row in selected],
        "remote_commands": [
            {
                "purpose": "create attempt directories",
                "argv": [
                    "mkdir",
                    "-p",
                    paths["controller_root"],
                    paths["source_root"],
                    paths["raw_root"],
                    paths["bundle_root"],
                    *environment.values(),
                ],
            },
            {
                "purpose": "run peer reduction workers",
                "argv": [
                    "python3",
                    f"{paths['source_root']}/tools/"
                    "run_qwen38_tp4_peer_reduction_microgate.py",
                    "--remote-supervise",
                    "--attempt",
                    attempt_tag,
                ],
            },
        ],
    }


def build_remote_worker_commands(
    plan,
    *,
    python_path=DEFAULT_REMOTE_PYTHON,
    dist_port=DEFAULT_DIST_PORT,
):
    _validate_plan(plan)
    if (
        not isinstance(python_path, str)
        or not _below(python_path, "/data00/home/sitian")
        or type(dist_port) is not int
        or not 1024 <= dist_port <= 65535
    ):
        raise ValueError("remote worker configuration is invalid")
    worker = (
        f"{plan['source_root']}/tools/"
        "qwen38_tp4_peer_reduction_microgate_worker.py"
    )
    return tuple(
        [
            python_path,
            worker,
            "--attempt",
            plan["attempt_tag"],
            "--source-revision",
            plan["source_revision"],
            "--output-dir",
            plan["raw_root"],
            "--rank",
            str(rank),
            "--world-size",
            "4",
            "--dist-port",
            str(dist_port),
        ]
        for rank in range(4)
    )


def _validate_plan(plan):
    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("remote_root") != APPROVED_REMOTE_ROOT
    ):
        raise ValueError("peer reduction plan is invalid")
    for key in (
        "attempt_root",
        "source_root",
        "raw_root",
        "bundle_root",
        "controller_root",
        "worker_stdout_path",
        "worker_stderr_path",
    ):
        if not _below(plan.get(key, ""), APPROVED_REMOTE_ROOT):
            raise ValueError("peer reduction plan path is invalid")
    selected = select_strict_clean_gpus(plan.get("selected_gpus", ()))
    if len(selected) != 4:
        raise ValueError("four strict-clean GPUs are required")
    return selected


def run_ssh_with_retry(
    argv,
    *,
    retry_count,
    runner=subprocess.run,
    timeout_s=None,
    input_text=None,
):
    if type(retry_count) is not int or retry_count < 0:
        raise ValueError("retry_count is invalid")
    result = None
    for _ in range(retry_count + 1):
        result = runner(
            argv,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
            input=input_text,
        )
        if result.returncode != 255:
            return result
    return result


def _ssh_argv(ssh_target, remote_argv, *, proxy_host):
    if (
        not isinstance(ssh_target, str)
        or not ssh_target
        or not isinstance(remote_argv, (list, tuple))
        or not remote_argv
        or any(
            not isinstance(argument, str)
            or not argument
            or "\0" in argument
            for argument in remote_argv
        )
    ):
        raise ValueError("SSH command is invalid")
    forbidden = {"kinit", "krenew", "kill", "pkill", "killall"}
    if PurePosixPath(remote_argv[0]).name in forbidden:
        raise ValueError("SSH command is forbidden")
    command = f"sh -c {shlex.quote(shlex.join(remote_argv))}"
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        f"ProxyCommand=ssh -qW %h:%p {proxy_host}",
        ssh_target,
        command,
    ]


def _remote_run(
    *,
    ssh_target,
    remote_argv,
    proxy_host,
    retry_count,
    timeout_s,
    input_text=None,
):
    result = run_ssh_with_retry(
        _ssh_argv(
            ssh_target,
            remote_argv,
            proxy_host=proxy_host,
        ),
        retry_count=retry_count,
        timeout_s=timeout_s,
        input_text=input_text,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr or "remote command failed"
        )
    return result


def query_remote_path_state(
    *,
    ssh_target,
    remote_root,
    attempt_tag,
    proxy_host,
    retry_count,
    timeout_s,
):
    if (
        remote_root != APPROVED_REMOTE_ROOT
        or not ATTEMPT_PATTERN.fullmatch(attempt_tag)
        or ".." in attempt_tag
    ):
        raise ValueError("remote path query is invalid")
    attempt_root = f"{remote_root}/attempts/{attempt_tag}"
    script = "\n".join([
        "import json,os,sys",
        "root,attempt=sys.argv[1:]",
        "parent=os.path.dirname(attempt)",
        "print(json.dumps({",
        "'attempt_exists':os.path.lexists(attempt),",
        "'attempt_parent_is_symlink':os.path.islink(parent),",
        "'remote_root_is_symlink':os.path.islink(root),",
        "},sort_keys=True))",
    ])
    result = _remote_run(
        ssh_target=ssh_target,
        remote_argv=[
            "python3",
            "-c",
            script,
            remote_root,
            attempt_root,
        ],
        proxy_host=proxy_host,
        retry_count=retry_count,
        timeout_s=timeout_s,
    )
    payload = json.loads(result.stdout)
    _validate_remote_path_state(payload)
    return payload


def capture_source_identity(*, attempt, source_revision, repo_root):
    root = Path(repo_root).resolve()

    def git(*arguments):
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "git command failed")
        return result.stdout

    if git("rev-parse", "HEAD").strip() != source_revision:
        raise ValueError("source revision does not match local HEAD")
    dirty = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--",
        "tinyvllm",
        "tools",
    )
    if dirty:
        raise ValueError("source archive scope has tracked changes")
    tree = git(
        "ls-tree",
        "-r",
        "--full-tree",
        source_revision,
        "tinyvllm",
        "tools",
    ).encode("utf-8")
    return {
        "schema_version": "qwen38.tp4-peer-reduction-source.v1",
        "attempt": attempt,
        "source_revision": source_revision,
        "source_tree_sha256": hashlib.sha256(tree).hexdigest(),
    }


def _upload_json(
    payload,
    remote_path,
    *,
    ssh_target,
    proxy_host,
    retry_count,
    timeout_s,
):
    script = "\n".join([
        "import os,sys,tempfile",
        "path=sys.argv[1]",
        "payload=sys.stdin.read()",
        "directory=os.path.dirname(path)",
        "fd,temp=tempfile.mkstemp(prefix='.upload.',dir=directory)",
        "with os.fdopen(fd,'w',encoding='utf-8') as handle:",
        "  handle.write(payload)",
        "  handle.flush()",
        "  os.fsync(handle.fileno())",
        "os.replace(temp,path)",
    ])
    _remote_run(
        ssh_target=ssh_target,
        remote_argv=["python3", "-c", script, remote_path],
        proxy_host=proxy_host,
        retry_count=retry_count,
        timeout_s=timeout_s,
        input_text=json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n",
    )


def _stage_committed_source(
    plan,
    *,
    repo_root,
    ssh_target,
    proxy_host,
    timeout_s,
):
    archive = subprocess.Popen(
        [
            "git",
            "-C",
            str(Path(repo_root).resolve()),
            "archive",
            "--format=tar",
            plan["source_revision"],
            "tinyvllm",
            "tools",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert archive.stdout is not None
    receiver = subprocess.run(
        _ssh_argv(
            ssh_target,
            ["tar", "-xf", "-", "-C", plan["source_root"]],
            proxy_host=proxy_host,
        ),
        stdin=archive.stdout,
        text=False,
        capture_output=True,
        check=False,
        timeout=timeout_s,
    )
    archive.stdout.close()
    archive_stderr = archive.stderr.read() if archive.stderr else b""
    archive_returncode = archive.wait()
    if archive_returncode != 0 or receiver.returncode != 0:
        raise RuntimeError(
            archive_stderr.decode(errors="replace")
            or receiver.stderr.decode(errors="replace")
            or "source archive staging failed"
        )


def _remote_inventory():
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    processes = subprocess.run(
        [
            "nvidia-smi",
            (
                "--query-compute-apps="
                "gpu_uuid,pid,process_name,used_memory"
            ),
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return parse_nvidia_smi_inventory(gpu.stdout, processes.stdout)


def _descendant_pids(parents):
    owned = set(parents)
    changed = True
    while changed:
        changed = False
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "stat").read_text(encoding="utf-8")
                fields = raw[raw.rfind(")") + 2:].split()
                pid = int(entry.name)
                parent = int(fields[1])
            except (
                FileNotFoundError,
                PermissionError,
                ProcessLookupError,
                ValueError,
                IndexError,
            ):
                continue
            if parent in owned and pid not in owned:
                owned.add(pid)
                changed = True
    return owned


def _exact_tag_worker_pids(attempt):
    matches = []
    own_pid = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            args = [
                part.decode(errors="replace")
                for part in (entry / "cmdline").read_bytes().split(b"\0")
                if part
            ]
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
        ):
            continue
        basenames = {PurePosixPath(argument).name for argument in args}
        if (
            attempt in args
            and "qwen38_tp4_peer_reduction_microgate_worker.py"
            in basenames
        ):
            matches.append(int(entry.name))
    return sorted(matches)


def supervise_remote_workers(
    plan,
    *,
    python_path=DEFAULT_REMOTE_PYTHON,
    dist_port=DEFAULT_DIST_PORT,
    poll_interval_s=1,
    worker_timeout_s=3600,
):
    selected = _validate_plan(plan)
    attempt_root = Path(plan["attempt_root"])
    raw_root = Path(plan["raw_root"])
    controller_root = Path(plan["controller_root"])
    for path in (
        raw_root,
        controller_root,
        Path(plan["environment"]["TMPDIR"]),
        Path(plan["environment"]["XDG_CACHE_HOME"]),
        Path(plan["environment"]["TORCH_EXTENSIONS_DIR"]),
        Path(plan["environment"]["CUDA_CACHE_PATH"]),
    ):
        path.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(plan["environment"])
    environment.update({
        "CUDA_VISIBLE_DEVICES": ",".join(
            str(row["gpu_index"]) for row in selected
        ),
        "PYTHONPATH": plan["source_root"],
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TORCH_CUDA_ARCH_LIST": "8.0",
        "MAX_JOBS": "8",
        "CC": (
            f"{REMOTE_TOOLCHAIN_ROOT}/bin/"
            "x86_64-conda-linux-gnu-gcc"
        ),
        "CXX": (
            f"{REMOTE_TOOLCHAIN_ROOT}/bin/"
            "x86_64-conda-linux-gnu-g++"
        ),
        "PATH": (
            f"{Path(python_path).parent}:"
            f"{REMOTE_TOOLCHAIN_ROOT}/bin:"
            f"{environment.get('PATH', '')}"
        ),
        "LD_LIBRARY_PATH": (
            "/data00/home/sitian/tllm/miniforge/lib:"
            f"{environment.get('LD_LIBRARY_PATH', '')}"
        ),
    })
    commands = build_remote_worker_commands(
        plan,
        python_path=python_path,
        dist_port=dist_port,
    )
    processes = []
    streams = []
    started = time.monotonic()
    violations = []
    samples = []
    try:
        for rank, command in enumerate(commands):
            stdout = (
                controller_root / f"rank-{rank}.stdout"
            ).open("a", encoding="utf-8")
            stderr = (
                controller_root / f"rank-{rank}.stderr"
            ).open("a", encoding="utf-8")
            streams.extend((stdout, stderr))
            processes.append(subprocess.Popen(
                command,
                cwd=plan["source_root"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=True,
            ))
        timed_out = False
        while any(process.poll() is None for process in processes):
            owned = _descendant_pids(
                {process.pid for process in processes}
            )
            try:
                validate_selected_gpu_processes(
                    selected=tuple(selected),
                    observed=_remote_inventory(),
                    owned_pids=owned,
                )
                samples.append({
                    "captured_at_unix_ns": time.time_ns(),
                    "owned_pids": sorted(owned),
                    "classification": "PASS",
                })
            except Exception as error:
                violations.append(f"{type(error).__name__}: {error}")
            if (
                not timed_out
                and time.monotonic() - started > worker_timeout_s
            ):
                timed_out = True
                violations.append("worker monitoring deadline exceeded")
            time.sleep(poll_interval_s)
        returncodes = [process.wait() for process in processes]
    finally:
        for stream in streams:
            stream.close()

    remaining = sorted(_descendant_pids(
        {process.pid for process in processes}
    ) - {process.pid for process in processes})
    scans = []
    for index in range(3):
        scans.append(_exact_tag_worker_pids(plan["attempt_tag"]))
        if index != 2:
            time.sleep(poll_interval_s)
    cleanup_path = raw_root / "cleanup.json"
    cleanup = (
        json.loads(cleanup_path.read_text(encoding="utf-8"))
        if cleanup_path.is_file()
        else {"classification": "DIRTY", "rank_rows": []}
    )
    cleanup.update({
        "owned_children_remaining": remaining,
        "exact_tag_scans": scans,
    })
    if (
        returncodes != [0, 0, 0, 0]
        or remaining
        or scans != [[], [], []]
    ):
        cleanup["classification"] = "DIRTY"
    write_json_atomic(cleanup_path, cleanup)
    required = (
        "peer_access_matrix.json",
        "ipc_roundtrip.jsonl",
        "microgate_rows.jsonl",
        "memory_summary.json",
        "cleanup.json",
    )
    missing = [
        name for name in required
        if not (raw_root / name).is_file()
    ]
    receipt = {
        "classification": (
            "PASS"
            if (
                returncodes == [0, 0, 0, 0]
                and not violations
                and not missing
                and cleanup.get("classification") == "CLEAN"
            )
            else "FAIL"
        ),
        "attempt": plan["attempt_tag"],
        "source_revision": plan["source_revision"],
        "owned_pids": [process.pid for process in processes],
        "rank_exit_codes": returncodes,
        "resource_snapshot_count": len(samples),
        "violations": violations,
        "missing_artifacts": missing,
        "cleanup": cleanup,
    }
    write_json_atomic(
        controller_root / "supervisor_receipt.json",
        receipt,
    )
    return receipt


def _query_inventory_via_proxy(
    *,
    ssh_target,
    proxy_host,
    retry_count,
    timeout_s,
):
    script = "\n".join([
        "import json,subprocess",
        "gpu=subprocess.run([",
        "'nvidia-smi',",
        "'--query-gpu=index,uuid,memory.used,utilization.gpu',",
        "'--format=csv,noheader,nounits',",
        "],check=True,text=True,capture_output=True)",
        "process=subprocess.run([",
        "'nvidia-smi',",
        (
            "'--query-compute-apps="
            "gpu_uuid,pid,process_name,used_memory',"
        ),
        "'--format=csv,noheader,nounits',",
        "],check=True,text=True,capture_output=True)",
        "print(json.dumps({",
        "'gpu_csv':gpu.stdout,",
        "'process_csv':process.stdout,",
        "},sort_keys=True))",
    ])
    result = _remote_run(
        ssh_target=ssh_target,
        remote_argv=["python3", "-c", script],
        proxy_host=proxy_host,
        retry_count=retry_count,
        timeout_s=timeout_s,
    )
    payload = json.loads(result.stdout)
    return parse_nvidia_smi_inventory(
        payload["gpu_csv"],
        payload["process_csv"],
    )


def _remote_json_command(
    plan,
    argv,
    *,
    ssh_target,
    proxy_host,
    retry_count,
    timeout_s,
):
    result = _remote_run(
        ssh_target=ssh_target,
        remote_argv=argv,
        proxy_host=proxy_host,
        retry_count=retry_count,
        timeout_s=timeout_s,
    )
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError("remote JSON result is invalid") from error
    if not isinstance(payload, dict):
        raise RuntimeError("remote JSON result is invalid")
    return payload


def _create_remote_attempt(
    plan,
    source_identity,
    *,
    repo_root,
    ssh_target,
    proxy_host,
    retry_count,
    timeout_s,
):
    directories = [
        plan["controller_root"],
        plan["source_root"],
        plan["raw_root"],
        *plan["environment"].values(),
    ]
    script = "\n".join([
        "import os,sys",
        "attempt=sys.argv[1]",
        "directories=sys.argv[2:]",
        "os.makedirs(os.path.dirname(attempt),exist_ok=True)",
        "os.mkdir(attempt)",
        "for path in directories: os.makedirs(path,exist_ok=False)",
    ])
    _remote_run(
        ssh_target=ssh_target,
        remote_argv=[
            "python3",
            "-c",
            script,
            plan["attempt_root"],
            *directories,
        ],
        proxy_host=proxy_host,
        retry_count=retry_count,
        timeout_s=timeout_s,
    )
    _stage_committed_source(
        plan,
        repo_root=repo_root,
        ssh_target=ssh_target,
        proxy_host=proxy_host,
        timeout_s=timeout_s,
    )
    _upload_json(
        source_identity,
        f"{plan['controller_root']}/source_identity.json",
        ssh_target=ssh_target,
        proxy_host=proxy_host,
        retry_count=retry_count,
        timeout_s=timeout_s,
    )
    _upload_json(
        plan,
        f"{plan['controller_root']}/plan.json",
        ssh_target=ssh_target,
        proxy_host=proxy_host,
        retry_count=retry_count,
        timeout_s=timeout_s,
    )
    return {"created": True}


def _download_compact_bundle(
    plan,
    *,
    local_attempt_root,
    ssh_target,
    proxy_host,
    timeout_s,
):
    local_root = Path(local_attempt_root).resolve()
    if (local_root / "final_bundle").exists():
        raise ValueError("local final bundle must not already exist")
    local_root.mkdir(parents=True, exist_ok=True)
    sender = subprocess.Popen(
        _ssh_argv(
            ssh_target,
            [
                "tar",
                "-cf",
                "-",
                "-C",
                plan["attempt_root"],
                "final_bundle",
                "controller/source_identity.json",
                "controller/plan.json",
                "controller/supervisor_receipt.json",
            ],
            proxy_host=proxy_host,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert sender.stdout is not None
    receiver = subprocess.run(
        ["tar", "-xf", "-", "-C", str(local_root)],
        stdin=sender.stdout,
        text=False,
        capture_output=True,
        check=False,
        timeout=timeout_s,
    )
    sender.stdout.close()
    sender_stderr = sender.stderr.read() if sender.stderr else b""
    sender_returncode = sender.wait()
    if sender_returncode != 0 or receiver.returncode != 0:
        raise RuntimeError(
            sender_stderr.decode(errors="replace")
            or receiver.stderr.decode(errors="replace")
            or "compact evidence download failed"
        )
    return {"downloaded": True}


def run_attempt(
    plan,
    *,
    plan_only,
    dry_run,
    kerberos_probe=None,
    gpu_probe=None,
    remote_writer=None,
    worker_runner=None,
    assembler=None,
    remote_verifier=None,
    downloader=None,
    local_verifier=None,
):
    selected = _validate_plan(plan)
    if plan_only:
        return {
            "classification": "PLAN_ONLY",
            "worker_started": False,
            "plan": plan,
        }
    if not callable(kerberos_probe):
        raise RuntimeError("Kerberos probe is required")
    kerberos = kerberos_probe()
    if kerberos.get("classification") != "PASS":
        return {
            "classification": "BLOCKED_KERBEROS",
            "worker_started": False,
            "kerberos": kerberos,
        }
    if not callable(gpu_probe):
        raise RuntimeError("GPU probe is required")
    observed = list(gpu_probe())
    select_strict_clean_gpus(observed)
    validate_selected_gpu_processes(
        selected=selected,
        observed=observed,
        owned_pids=set(),
    )
    if dry_run:
        return {
            "classification": "DRY_RUN_READY",
            "worker_started": False,
            "plan": plan,
        }
    required = {
        "remote_writer": remote_writer,
        "worker_runner": worker_runner,
        "assembler": assembler,
        "remote_verifier": remote_verifier,
        "downloader": downloader,
        "local_verifier": local_verifier,
    }
    missing = [
        name for name, callback in required.items()
        if not callable(callback)
    ]
    if missing:
        raise RuntimeError(
            "execution adapters are missing: " + ", ".join(missing)
        )
    receipt = remote_writer(plan)
    if not isinstance(receipt, dict) or receipt.get("created") is not True:
        raise RuntimeError("remote attempt creation failed")

    observed = list(gpu_probe())
    select_strict_clean_gpus(observed)
    validate_selected_gpu_processes(
        selected=selected,
        observed=observed,
        owned_pids=set(),
    )
    worker = worker_runner(plan)
    if (
        not isinstance(worker, dict)
        or worker.get("classification") != "PASS"
    ):
        raise RuntimeError("peer reduction worker failed")
    producer = assembler(plan)
    remote = remote_verifier(plan)
    download = downloader(plan)
    if (
        not isinstance(download, dict)
        or download.get("downloaded") is not True
    ):
        raise RuntimeError("peer reduction download is incomplete")
    local = local_verifier(plan)
    if not all(
        isinstance(row, dict)
        and isinstance(row.get("classification"), str)
        for row in (producer, remote, local)
    ):
        raise RuntimeError("producer/verifier result is invalid")
    classifications = {
        row.get("classification")
        for row in (producer, remote, local)
    }
    if len(classifications) != 1:
        raise RuntimeError("producer/verifier classification disagreement")
    return {
        "classification": classifications.pop(),
        "worker_started": True,
        "producer": producer,
        "remote_verification": remote,
        "local_verification": local,
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--remote-root", default=APPROVED_REMOTE_ROOT)
    parser.add_argument("--ssh-target", default="sitian@10.232.195.203")
    parser.add_argument("--proxy-host", default=DEFAULT_PROXY_HOST)
    parser.add_argument(
        "--command-timeout-s",
        type=int,
        default=DEFAULT_COMMAND_TIMEOUT_S,
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=DEFAULT_RETRY_COUNT,
    )
    parser.add_argument(
        "--gpu-wait-timeout-s",
        type=int,
        default=DEFAULT_GPU_WAIT_TIMEOUT_S,
    )
    parser.add_argument(
        "--gpu-poll-interval-s",
        type=int,
        default=DEFAULT_GPU_POLL_INTERVAL_S,
    )
    parser.add_argument("--remote-python", default=DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--dist-port", type=int, default=DEFAULT_DIST_PORT)
    parser.add_argument("--local-attempt-root", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--remote-supervise", action="store_true")
    parser.add_argument("--selected-gpus-json")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    source_revision = args.source_revision
    if source_revision is None:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "cannot resolve HEAD")
        source_revision = result.stdout.strip()

    if args.remote_supervise:
        if not args.selected_gpus_json:
            raise ValueError("selected GPU inventory is required")
        selected = json.loads(args.selected_gpus_json)
        plan = build_attempt_plan(
            attempt_tag=args.attempt,
            source_revision=source_revision,
            selected_gpus=selected,
            remote_path_state={
                "attempt_exists": False,
                "attempt_parent_is_symlink": False,
                "remote_root_is_symlink": False,
            },
            remote_root=args.remote_root,
        )
        receipt = supervise_remote_workers(
            plan,
            python_path=args.remote_python,
            dist_port=args.dist_port,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0 if receipt["classification"] == "PASS" else 2

    kerberos = query_local_kerberos()
    if kerberos.get("classification") != "READY":
        print(json.dumps({
            "classification": "BLOCKED_KERBEROS",
            "worker_started": False,
            "kerberos": kerberos,
        }, sort_keys=True))
        return 3

    path_state = query_remote_path_state(
        ssh_target=args.ssh_target,
        remote_root=args.remote_root,
        attempt_tag=args.attempt,
        proxy_host=args.proxy_host,
        retry_count=args.retry_count,
        timeout_s=args.command_timeout_s,
    )

    def inventory():
        return _query_inventory_via_proxy(
            ssh_target=args.ssh_target,
            proxy_host=args.proxy_host,
            retry_count=args.retry_count,
            timeout_s=args.command_timeout_s,
        )

    admission = wait_for_strict_clean_gpus(
        query_inventory=inventory,
        timeout_s=args.gpu_wait_timeout_s,
        poll_interval_s=args.gpu_poll_interval_s,
    )
    if admission.get("classification") != "READY":
        print(json.dumps({
            "classification": "BLOCKED_RESOURCES",
            "worker_started": False,
            "admission": admission,
        }, sort_keys=True))
        return 4
    selected = admission["selected_gpus"]
    plan = build_attempt_plan(
        attempt_tag=args.attempt,
        source_revision=source_revision,
        selected_gpus=selected,
        remote_path_state=path_state,
        remote_root=args.remote_root,
    )
    if args.plan_only:
        print(json.dumps({
            "classification": "PLAN_ONLY",
            "worker_started": False,
            "plan": plan,
        }, sort_keys=True))
        return 0

    source_identity = capture_source_identity(
        attempt=args.attempt,
        source_revision=source_revision,
        repo_root=repo_root,
    )
    local_attempt_root = (
        args.local_attempt_root.resolve()
        if args.local_attempt_root is not None
        else (
            repo_root
            / "artifacts"
            / "qwen38_tp4_peer_reduction"
            / args.attempt
        )
    )
    local_controller = local_attempt_root / "controller"
    local_controller.mkdir(parents=True, exist_ok=True)
    write_json_atomic(local_controller / "plan.json", plan)
    write_json_atomic(
        local_controller / "source_identity.json",
        source_identity,
    )
    write_json_atomic(
        local_controller / "launch_admission.json",
        admission,
    )

    def create_remote(current_plan):
        return _create_remote_attempt(
            current_plan,
            source_identity,
            repo_root=repo_root,
            ssh_target=args.ssh_target,
            proxy_host=args.proxy_host,
            retry_count=args.retry_count,
            timeout_s=max(args.command_timeout_s, 600),
        )

    def run_workers(current_plan):
        return _remote_json_command(
            current_plan,
            [
                args.remote_python,
                (
                    f"{current_plan['source_root']}/tools/"
                    "run_qwen38_tp4_peer_reduction_microgate.py"
                ),
                "--remote-supervise",
                "--attempt",
                current_plan["attempt_tag"],
                "--source-revision",
                current_plan["source_revision"],
                "--remote-root",
                current_plan["remote_root"],
                "--remote-python",
                args.remote_python,
                "--dist-port",
                str(args.dist_port),
                "--selected-gpus-json",
                json.dumps(
                    current_plan["selected_gpus"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ],
            ssh_target=args.ssh_target,
            proxy_host=args.proxy_host,
            retry_count=0,
            timeout_s=max(args.command_timeout_s, 7200),
        )

    def assemble_remote(current_plan):
        return _remote_json_command(
            current_plan,
            [
                args.remote_python,
                (
                    f"{current_plan['source_root']}/tools/"
                    "assemble_qwen38_tp4_peer_reduction_microgate.py"
                ),
                "--attempt-root",
                current_plan["attempt_root"],
            ],
            ssh_target=args.ssh_target,
            proxy_host=args.proxy_host,
            retry_count=args.retry_count,
            timeout_s=max(args.command_timeout_s, 300),
        )

    def verify_remote(current_plan):
        payload = _remote_json_command(
            current_plan,
            [
                args.remote_python,
                (
                    f"{current_plan['source_root']}/tools/"
                    "verify_qwen38_tp4_peer_reduction_microgate.py"
                ),
                "--bundle-root",
                current_plan["bundle_root"],
            ],
            ssh_target=args.ssh_target,
            proxy_host=args.proxy_host,
            retry_count=args.retry_count,
            timeout_s=max(args.command_timeout_s, 300),
        )
        return {
            **payload,
            "classification": payload.get(
                "reconstructed_classification"
            ),
        }

    def download(current_plan):
        return _download_compact_bundle(
            current_plan,
            local_attempt_root=local_attempt_root,
            ssh_target=args.ssh_target,
            proxy_host=args.proxy_host,
            timeout_s=max(args.command_timeout_s, 600),
        )

    def verify_local(_current_plan):
        from tools.verify_qwen38_tp4_peer_reduction_microgate import (
            verify_bundle,
        )

        payload = verify_bundle(local_attempt_root / "final_bundle")
        return {
            **payload,
            "classification": payload[
                "reconstructed_classification"
            ],
        }

    def kerberos_probe():
        current = query_local_kerberos()
        return {
            **current,
            "classification": (
                "PASS"
                if current.get("classification") == "READY"
                else "BLOCKED"
            ),
        }

    result = run_attempt(
        plan,
        plan_only=False,
        dry_run=args.dry_run,
        kerberos_probe=kerberos_probe,
        gpu_probe=inventory,
        remote_writer=create_remote,
        worker_runner=run_workers,
        assembler=assemble_remote,
        remote_verifier=verify_remote,
        downloader=download,
        local_verifier=verify_local,
    )
    write_json_atomic(local_controller / "controller_result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["classification"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
