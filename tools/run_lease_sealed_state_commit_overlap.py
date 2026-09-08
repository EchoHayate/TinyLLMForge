#!/usr/bin/env python3
"""Safely orchestrate the TP4 state-commit overlap Stage-0 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import subprocess
import time

if __package__:
    from tools.run_qwen38_tp4_communication_profile import (
        parse_nvidia_smi_inventory,
        query_local_kerberos,
        select_strict_clean_gpus,
        validate_selected_gpu_processes,
        wait_for_strict_clean_gpus,
        write_json_atomic,
    )
else:
    from run_qwen38_tp4_communication_profile import (
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
DEFAULT_REMOTE_PYTHON = "/data00/home/sitian/tllm/env/bin/python"
DEFAULT_PROXY_HOST = "jump-proxy-lf"
DEFAULT_GPU_WAIT_TIMEOUT_S = 21_600
DEFAULT_GPU_POLL_INTERVAL_S = 15
DEFAULT_COMMAND_TIMEOUT_S = 60
DEFAULT_RETRY_COUNT = 3
DEFAULT_DIST_PORT = 29_741
MINIMUM_KERBEROS_LIFETIME_SECONDS = 22_560
PLAN_SCHEMA = "tp4-completion-owned-overlap-plan.v2"
PROTOCOL = "completion-owned-stage01"
ATTEMPT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _below(path, root):
    candidate = PurePosixPath(path)
    approved = PurePosixPath(root)
    return candidate.is_absolute() and candidate.is_relative_to(approved)


def _validate_remote_path_state(state, *, expected_attempt_root):
    if (
        not isinstance(state, dict)
        or type(state.get("attempt_exists")) is not bool
        or type(state.get("attempt_parent_is_symlink")) is not bool
        or type(state.get("remote_root_is_symlink")) is not bool
        or type(state.get("remote_root_exists")) is not bool
        or type(state.get("remote_root_is_directory")) is not bool
        or type(state.get("remote_root_on_distinct_filesystem")) is not bool
        or not isinstance(state.get("resolved_remote_root"), str)
        or not isinstance(state.get("resolved_attempt_root"), str)
    ):
        raise ValueError("remote path state is invalid")
    if (
        not state["remote_root_exists"]
        or not state["remote_root_is_directory"]
        or not state["remote_root_on_distinct_filesystem"]
    ):
        raise ValueError("approved remote root must be a mounted filesystem")
    if (
        state["attempt_exists"]
        or state["attempt_parent_is_symlink"]
        or state["remote_root_is_symlink"]
        or state["resolved_remote_root"] != APPROVED_REMOTE_ROOT
        or state["resolved_attempt_root"] != expected_attempt_root
    ):
        raise ValueError("attempt path must be fresh and non-symlinked")


def build_attempt_plan(
    *,
    attempt_tag,
    source_revision,
    source_tree_sha256,
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
    if not REVISION_PATTERN.fullmatch(str(source_revision)):
        raise ValueError("source revision is invalid")
    if not SHA256_PATTERN.fullmatch(str(source_tree_sha256)):
        raise ValueError("source tree SHA-256 is invalid")
    attempt_root = f"{remote_root}/attempts/{attempt_tag}"
    _validate_remote_path_state(
        remote_path_state,
        expected_attempt_root=attempt_root,
    )
    try:
        selected = select_strict_clean_gpus(list(selected_gpus))
    except ValueError as error:
        raise ValueError("four strict-clean GPUs are required") from error
    if len(selected) != 4:
        raise ValueError("four strict-clean GPUs are required")

    runtime_root = f"{attempt_root}/runtime"
    plan = {
        "schema_version": PLAN_SCHEMA,
        "protocol": PROTOCOL,
        "attempt_tag": attempt_tag,
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "remote_root": remote_root,
        "attempt_root": attempt_root,
        "source_root": f"{attempt_root}/source",
        "raw_root": f"{attempt_root}/raw",
        "bundle_root": f"{attempt_root}/final_bundle",
        "controller_root": f"{attempt_root}/controller",
        "environment": {
            "TMPDIR": f"{runtime_root}/tmp",
            "XDG_CACHE_HOME": f"{runtime_root}/cache/xdg",
            "TORCH_EXTENSIONS_DIR": (
                f"{runtime_root}/cache/torch-extensions"
            ),
            "CUDA_CACHE_PATH": f"{runtime_root}/cache/cuda",
        },
        "selected_gpus": [dict(row) for row in selected],
    }
    planned_paths = [
        plan[name]
        for name in (
            "attempt_root",
            "source_root",
            "raw_root",
            "bundle_root",
            "controller_root",
        )
    ]
    if not all(_below(path, remote_root) for path in planned_paths):
        raise ValueError("planned path escapes approved root")
    if not all(
        _below(path, attempt_root)
        for path in plan["environment"].values()
    ):
        raise ValueError("planned environment path escapes attempt root")
    return plan


def _validate_plan(plan):
    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("protocol") != PROTOCOL
        or plan.get("remote_root") != APPROVED_REMOTE_ROOT
        or not REVISION_PATTERN.fullmatch(
            str(plan.get("source_revision", ""))
        )
        or not SHA256_PATTERN.fullmatch(
            str(plan.get("source_tree_sha256", ""))
        )
    ):
        raise ValueError("state-commit overlap plan is invalid")
    for name in (
        "attempt_root",
        "source_root",
        "raw_root",
        "bundle_root",
        "controller_root",
    ):
        if not _below(plan.get(name, ""), APPROVED_REMOTE_ROOT):
            raise ValueError("state-commit overlap plan path is invalid")
    if not all(
        _below(path, plan["attempt_root"])
        for path in plan.get("environment", {}).values()
    ):
        raise ValueError("state-commit overlap environment path is invalid")
    selected = select_strict_clean_gpus(plan.get("selected_gpus", []))
    if len(selected) != 4:
        raise ValueError("four strict-clean GPUs are required")
    return selected


def build_remote_worker_commands(
    plan,
    *,
    python_path=DEFAULT_REMOTE_PYTHON,
    dist_port=DEFAULT_DIST_PORT,
):
    selected = _validate_plan(plan)
    if (
        not isinstance(python_path, str)
        or not _below(python_path, "/data00/home/sitian")
        or type(dist_port) is not int
        or not 1024 <= dist_port <= 65535
    ):
        raise ValueError("remote worker configuration is invalid")
    worker = (
        f"{plan['source_root']}/tools/"
        "lease_sealed_state_commit_overlap_worker.py"
    )
    visible = ",".join(str(row["gpu_index"]) for row in selected)
    environment = {
        **plan["environment"],
        "CUDA_VISIBLE_DEVICES": visible,
        "PYTHONPATH": plan["source_root"],
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
    }
    prefix = " ".join(
        f"{name}={shlex.quote(value)}"
        for name, value in sorted(environment.items())
    )
    return tuple(
        f"{prefix} "
        + shlex.join([
            python_path,
            worker,
            "--attempt",
            plan["attempt_tag"],
            "--source-revision",
            plan["source_revision"],
            "--source-tree-sha256",
            plan["source_tree_sha256"],
            "--protocol",
            PROTOCOL,
            "--output-dir",
            plan["raw_root"],
            "--rank",
            str(rank),
            "--world-size",
            "4",
            "--dist-port",
            str(dist_port),
        ])
        for rank in range(4)
    )


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


def _ssh_argv(ssh_target, remote_argv, proxy_host):
    if (
        not isinstance(ssh_target, str)
        or not ssh_target
        or not isinstance(remote_argv, (list, tuple))
        or not remote_argv
    ):
        raise ValueError("SSH command is invalid")
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
        _ssh_argv(ssh_target, remote_argv, proxy_host),
        retry_count=retry_count,
        timeout_s=timeout_s,
        input_text=input_text,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "remote command failed")
    return result


def query_remote_path_state(
    *,
    ssh_target,
    attempt_tag,
    proxy_host,
    retry_count,
    timeout_s,
):
    if (
        not ATTEMPT_PATTERN.fullmatch(attempt_tag)
        or ".." in attempt_tag
    ):
        raise ValueError("attempt tag is invalid")
    attempt_root = f"{APPROVED_REMOTE_ROOT}/attempts/{attempt_tag}"
    script = "\n".join([
        "import json,os,sys",
        "root,attempt=sys.argv[1:]",
        "parent=os.path.dirname(attempt)",
        "root_exists=os.path.exists(root)",
        "root_is_directory=os.path.isdir(root)",
        "root_device=os.stat(root).st_dev if root_is_directory else None",
        "filesystem_root_device=os.stat('/').st_dev",
        "print(json.dumps({",
        "'attempt_exists':os.path.lexists(attempt),",
        "'attempt_parent_is_symlink':os.path.islink(parent),",
        "'remote_root_is_symlink':os.path.islink(root),",
        "'remote_root_exists':root_exists,",
        "'remote_root_is_directory':root_is_directory,",
        "'remote_root_on_distinct_filesystem':",
        "root_device is not None and root_device != filesystem_root_device,",
        "'resolved_remote_root':os.path.realpath(root),",
        "'resolved_attempt_root':os.path.realpath(attempt),",
        "},sort_keys=True))",
    ])
    result = _remote_run(
        ssh_target=ssh_target,
        remote_argv=[
            "python3",
            "-c",
            script,
            APPROVED_REMOTE_ROOT,
            attempt_root,
        ],
        proxy_host=proxy_host,
        retry_count=retry_count,
        timeout_s=timeout_s,
    )
    payload = json.loads(result.stdout)
    _validate_remote_path_state(
        payload,
        expected_attempt_root=attempt_root,
    )
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
        "schema_version": "tp4-completion-owned-overlap-source.v2",
        "protocol": PROTOCOL,
        "attempt": attempt,
        "source_revision": source_revision,
        "source_tree_sha256": hashlib.sha256(tree).hexdigest(),
    }


def _query_inventory(
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
        "'--query-compute-apps=gpu_uuid,pid,process_name,used_memory',",
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
        )
        + "\n",
    )


def _stage_committed_source(
    plan,
    *,
    repo_root,
    ssh_target,
    proxy_host,
    retry_count=0,
    timeout_s,
):
    if type(retry_count) is not int or retry_count < 0:
        raise ValueError("retry_count is invalid")
    for attempt in range(retry_count + 1):
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
        try:
            receiver = subprocess.run(
                _ssh_argv(
                    ssh_target,
                    ["tar", "-xf", "-", "-C", plan["source_root"]],
                    proxy_host,
                ),
                stdin=archive.stdout,
                text=False,
                capture_output=True,
                check=False,
                timeout=timeout_s,
            )
        except BaseException:
            archive.stdout.close()
            try:
                archive.terminate()
            except ProcessLookupError:
                pass
            try:
                archive.wait(timeout=5)
            except subprocess.TimeoutExpired:
                archive.kill()
                archive.wait()
            raise
        archive.stdout.close()
        archive_stderr = archive.stderr.read() if archive.stderr else b""
        archive_returncode = archive.wait()
        if archive_returncode == 0 and receiver.returncode == 0:
            return
        if receiver.returncode == 255 and attempt < retry_count:
            continue
        raise RuntimeError(
            archive_stderr.decode(errors="replace")
            or receiver.stderr.decode(errors="replace")
            or "source archive staging failed"
        )


def _create_remote_attempt(
    plan,
    source_identity,
    admission,
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
        "root,attempt=sys.argv[1:3]",
        "directories=sys.argv[3:]",
        "if not os.path.isdir(root): raise RuntimeError('root missing')",
        "if os.path.realpath(root)!=root: raise RuntimeError('root symlinked')",
        "if os.stat(root).st_dev==os.stat('/').st_dev:",
        "  raise RuntimeError('root is not mounted')",
        "parent=os.path.dirname(attempt)",
        "os.makedirs(parent,exist_ok=True)",
        "if os.path.realpath(parent)!=parent:",
        "  raise RuntimeError('attempt parent symlinked')",
        "if os.stat(parent).st_dev!=os.stat(root).st_dev:",
        "  raise RuntimeError('attempt parent changed filesystem')",
        "os.mkdir(attempt)",
        "for path in directories: os.makedirs(path,exist_ok=False)",
    ])
    _remote_run(
        ssh_target=ssh_target,
        remote_argv=[
            "python3",
            "-c",
            script,
            plan["remote_root"],
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
        retry_count=retry_count,
        timeout_s=timeout_s,
    )
    for payload, name in (
        (plan, "plan.json"),
        (source_identity, "source_identity.json"),
        (admission, "launch_admission.json"),
    ):
        _upload_json(
            payload,
            f"{plan['controller_root']}/{name}",
            ssh_target=ssh_target,
            proxy_host=proxy_host,
            retry_count=retry_count,
            timeout_s=timeout_s,
        )
    return {"classification": "PASS", "created": True}


def _remote_inventory_local():
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
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return parse_nvidia_smi_inventory(gpu.stdout, processes.stdout)


def _validate_frozen_gpu_admission(selected, observed):
    current = validate_selected_gpu_processes(
        selected=selected,
        observed=observed,
        owned_pids=set(),
    )
    strict = select_strict_clean_gpus(list(current))
    if tuple(row["gpu_uuid"] for row in strict) != tuple(
        row["gpu_uuid"] for row in current
    ):
        raise ValueError("frozen GPU inventory is not strict-clean")
    return current


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
                fields = raw[raw.rfind(")") + 2 :].split()
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
            arguments = [
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
        if (
            attempt in arguments
            and any(
                PurePosixPath(argument).name
                == "lease_sealed_state_commit_overlap_worker.py"
                for argument in arguments
            )
        ):
            matches.append(int(entry.name))
    return sorted(matches)


def _terminate_owned_process_groups(
    processes,
    *,
    owned_process_groups=None,
    grace_s=5,
    get_process_group=os.getpgid,
    signal_group=os.killpg,
    sleeper=time.sleep,
):
    if owned_process_groups is None:
        process_groups = []
        for process in processes:
            try:
                process_group = get_process_group(process.pid)
            except ProcessLookupError:
                continue
            if process_group != process.pid:
                raise RuntimeError("worker process group ownership is invalid")
            process_groups.append(process_group)
    else:
        process_groups = list(owned_process_groups)
        if (
            len(process_groups) != len(set(process_groups))
            or any(
                type(process_group) is not int or process_group <= 0
                for process_group in process_groups
            )
        ):
            raise RuntimeError("owned worker process groups are invalid")
    for process_group in process_groups:
        try:
            signal_group(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if process_groups:
        sleeper(grace_s)
    for process_group in process_groups:
        try:
            signal_group(process_group, 0)
        except ProcessLookupError:
            continue
        signal_group(process_group, signal.SIGKILL)


def supervise_remote_workers(
    plan,
    *,
    python_path=DEFAULT_REMOTE_PYTHON,
    dist_port=DEFAULT_DIST_PORT,
    poll_interval_s=1,
    worker_timeout_s=3600,
):
    selected = _validate_plan(plan)
    raw_root = Path(plan["raw_root"])
    controller_root = Path(plan["controller_root"])
    for path in (
        raw_root,
        controller_root,
        *(Path(value) for value in plan["environment"].values()),
    ):
        path.mkdir(parents=True, exist_ok=True)
    commands = build_remote_worker_commands(
        plan,
        python_path=python_path,
        dist_port=dist_port,
    )
    processes = []
    owned_process_groups = []
    streams = []
    samples = []
    violations = []
    started = time.monotonic()
    try:
        for rank, command in enumerate(commands):
            stdout = (controller_root / f"rank-{rank}.stdout").open(
                "a",
                encoding="utf-8",
            )
            stderr = (controller_root / f"rank-{rank}.stderr").open(
                "a",
                encoding="utf-8",
            )
            streams.extend((stdout, stderr))
            process = subprocess.Popen(
                command,
                cwd=plan["source_root"],
                shell=True,
                executable="/bin/bash",
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=True,
            )
            processes.append(process)
            owned_process_groups.append(process.pid)
        deadline_recorded = False
        termination_requested = False
        while any(process.poll() is None for process in processes):
            returncode_snapshot = [
                process.poll() for process in processes
            ]
            failed_ranks = [
                rank
                for rank, returncode in enumerate(returncode_snapshot)
                if returncode not in (None, 0)
            ]
            if failed_ranks and not termination_requested:
                termination_requested = True
                violations.append(
                    "worker rank failed before peers completed: "
                    + ",".join(str(rank) for rank in failed_ranks)
                )
                _terminate_owned_process_groups(
                    processes,
                    owned_process_groups=owned_process_groups,
                )
            owned = _descendant_pids(
                {process.pid for process in processes}
            )
            try:
                validate_selected_gpu_processes(
                    selected=selected,
                    observed=_remote_inventory_local(),
                    owned_pids=owned,
                )
                samples.append({
                    "captured_at_unix_ns": time.time_ns(),
                    "owned_pids": sorted(owned),
                    "classification": "PASS",
                })
            except Exception as error:
                violations.append(f"{type(error).__name__}: {error}")
                if not termination_requested:
                    termination_requested = True
                    _terminate_owned_process_groups(
                        processes,
                        owned_process_groups=owned_process_groups,
                    )
            if (
                not deadline_recorded
                and time.monotonic() - started > worker_timeout_s
            ):
                deadline_recorded = True
                violations.append("worker monitoring deadline exceeded")
                if not termination_requested:
                    termination_requested = True
                    _terminate_owned_process_groups(
                        processes,
                        owned_process_groups=owned_process_groups,
                    )
            time.sleep(poll_interval_s)
        returncodes = [process.wait() for process in processes]
    except BaseException:
        _terminate_owned_process_groups(
            processes,
            owned_process_groups=owned_process_groups,
        )
        raise
    finally:
        for stream in streams:
            stream.close()

    remaining = sorted(
        _descendant_pids({process.pid for process in processes})
        - {process.pid for process in processes}
    )
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
        "attempt": plan["attempt_tag"],
        "source_revision": plan["source_revision"],
        "source_tree_sha256": plan["source_tree_sha256"],
        "owned_children_remaining": remaining,
        "exact_tag_scans": scans,
    })
    if (
        returncodes != [0, 0, 0, 0]
        or remaining
        or scans != [[], [], []]
        or violations
    ):
        cleanup["classification"] = "DIRTY"
    write_json_atomic(cleanup_path, cleanup)
    required = (
        "diagnostic_rows.jsonl",
        "measurement_rows.jsonl",
        "memory.json",
        "lifecycle.json",
        "cleanup.json",
        "runtime_capabilities.json",
    )
    missing = [
        name for name in required if not (raw_root / name).is_file()
    ]
    receipt = {
        "protocol": PROTOCOL,
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


def _remote_json_command(
    plan,
    argv,
    *,
    ssh_target,
    proxy_host,
    retry_count,
    timeout_s,
):
    _validate_plan(plan)
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
            ],
            proxy_host,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert sender.stdout is not None
    try:
        receiver = subprocess.run(
            ["tar", "-xf", "-", "-C", str(local_root)],
            stdin=sender.stdout,
            text=False,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except BaseException:
        sender.stdout.close()
        try:
            sender.terminate()
        except ProcessLookupError:
            pass
        try:
            sender.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sender.kill()
            sender.wait()
        raise
    sender.stdout.close()
    sender_stderr = sender.stderr.read() if sender.stderr else b""
    sender_returncode = sender.wait()
    if sender_returncode != 0 or receiver.returncode != 0:
        raise RuntimeError(
            sender_stderr.decode(errors="replace")
            or receiver.stderr.decode(errors="replace")
            or "compact evidence download failed"
        )
    return {"classification": "PASS", "downloaded": True}


def verify_local_downloaded_bundle(bundle_root):
    if __package__:
        from tools.verify_lease_sealed_state_commit_overlap import (
            LOCAL_RECEIPT_NAME,
            verify_bundle,
        )
    else:
        from verify_lease_sealed_state_commit_overlap import (
            LOCAL_RECEIPT_NAME,
            verify_bundle,
        )
    return verify_bundle(
        bundle_root,
        receipt_name=LOCAL_RECEIPT_NAME,
        seal_terminal=True,
    )


def _prepare_local_attempt_root(local_attempt_root):
    local_root = Path(local_attempt_root).resolve()
    if local_root.exists():
        raise ValueError("local attempt path must be fresh")
    controller_root = local_root / "controller"
    controller_root.mkdir(parents=True, exist_ok=False)
    return controller_root


def _run_with_terminal_receipt(receipt_path, operation):
    try:
        result = operation()
    except Exception as error:
        write_json_atomic(
            receipt_path,
            {
                "protocol": PROTOCOL,
                "classification": "CONTROLLER_ERROR",
                "worker_started": None,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    if isinstance(result, dict):
        result = {**result, "protocol": PROTOCOL}
    write_json_atomic(receipt_path, result)
    return result


def run_attempt(
    plan,
    *,
    plan_only=False,
    dry_run=False,
    kerberos_probe=None,
    gpu_probe=None,
    remote_writer=None,
    launch_admission_writer=None,
    worker_runner=None,
    assembler=None,
    remote_verifier=None,
    downloader=None,
    local_verifier=None,
):
    selected = _validate_plan(plan)
    if plan_only:
        return {
            "protocol": PROTOCOL,
            "classification": "PLAN_ONLY",
            "worker_started": False,
            "plan": plan,
        }
    if not callable(kerberos_probe):
        raise RuntimeError("Kerberos probe is required")
    kerberos = kerberos_probe()
    if kerberos.get("classification") not in ("PASS", "READY"):
        return {
            "protocol": PROTOCOL,
            "classification": "BLOCKED_KERBEROS",
            "worker_started": False,
            "kerberos": kerberos,
        }
    if not callable(gpu_probe):
        raise RuntimeError("GPU probe is required")
    observed = list(gpu_probe())
    _validate_frozen_gpu_admission(selected, observed)
    if dry_run:
        return {
            "protocol": PROTOCOL,
            "classification": "DRY_RUN_READY",
            "worker_started": False,
            "plan": plan,
        }
    required = {
        "remote_writer": remote_writer,
        "launch_admission_writer": launch_admission_writer,
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

    created = remote_writer(plan)
    if (
        not isinstance(created, dict)
        or created.get("classification") != "PASS"
    ):
        raise RuntimeError("remote attempt creation failed")
    observed = list(gpu_probe())
    immediate_selected = _validate_frozen_gpu_admission(selected, observed)
    launch_admission = launch_admission_writer(
        plan,
        list(immediate_selected),
    )
    if (
        not isinstance(launch_admission, dict)
        or launch_admission.get("classification") != "PASS"
    ):
        raise RuntimeError("immediate launch admission write failed")
    worker = worker_runner(plan)
    if (
        not isinstance(worker, dict)
        or worker.get("classification") != "PASS"
    ):
        raise RuntimeError("state-commit overlap worker failed")
    producer = assembler(plan)
    remote = remote_verifier(plan)
    downloaded = downloader(plan)
    if (
        not isinstance(downloaded, dict)
        or downloaded.get("classification") != "PASS"
    ):
        raise RuntimeError("compact evidence download is incomplete")
    local = local_verifier(plan)
    if (
        not isinstance(producer, dict)
        or not isinstance(producer.get("classification"), str)
        or not isinstance(remote, dict)
        or remote.get("status") != "PASS"
        or not isinstance(local, dict)
        or local.get("status") != "PASS"
    ):
        raise RuntimeError("producer/verifier result is invalid")
    classifications = {
        producer["classification"],
        remote.get("reconstructed_classification"),
        local.get("reconstructed_classification"),
    }
    if None in classifications or len(classifications) != 1:
        raise RuntimeError("producer/verifier classification disagreement")
    return {
        "protocol": PROTOCOL,
        "classification": classifications.pop(),
        "worker_started": True,
        "producer": producer,
        "remote_verification": remote,
        "local_verification": local,
    }


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-tag", required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--ssh-target", default="10.232.195.203")
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
    parser.add_argument("--source-tree-sha256")
    parser.add_argument("--selected-gpus-json")
    return parser


def _main_unwrapped(argv=None):
    args = build_argument_parser().parse_args(argv)
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
        if not args.selected_gpus_json or not args.source_tree_sha256:
            raise ValueError("selected GPU inventory and tree hash required")
        plan = build_attempt_plan(
            attempt_tag=args.attempt_tag,
            source_revision=source_revision,
            source_tree_sha256=args.source_tree_sha256,
            selected_gpus=json.loads(args.selected_gpus_json),
            remote_path_state={
                "attempt_exists": False,
                "attempt_parent_is_symlink": False,
                "remote_root_is_symlink": False,
                "remote_root_exists": True,
                "remote_root_is_directory": True,
                "remote_root_on_distinct_filesystem": True,
                "resolved_remote_root": APPROVED_REMOTE_ROOT,
                "resolved_attempt_root": (
                    f"{APPROVED_REMOTE_ROOT}/attempts/{args.attempt_tag}"
                ),
            },
        )
        receipt = _run_with_terminal_receipt(
            Path(plan["controller_root"]) / "supervisor_receipt.json",
            lambda: supervise_remote_workers(
                plan,
                python_path=args.remote_python,
                dist_port=args.dist_port,
            ),
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0 if receipt["classification"] == "PASS" else 2

    local_attempt_root = (
        args.local_attempt_root.resolve()
        if args.local_attempt_root is not None
        else (
            repo_root
            / "artifacts"
            / "lease_sealed_state_commit_overlap"
            / args.attempt_tag
        )
    )
    local_controller = _prepare_local_attempt_root(local_attempt_root)
    kerberos = query_local_kerberos(
        minimum_lifetime_seconds=MINIMUM_KERBEROS_LIFETIME_SECONDS
    )
    if kerberos.get("classification") != "READY":
        result = {
            "protocol": PROTOCOL,
            "classification": "BLOCKED_KERBEROS",
            "worker_started": False,
            "kerberos": kerberos,
        }
        write_json_atomic(local_controller / "result.json", result)
        print(json.dumps(result, sort_keys=True))
        return 3
    source = capture_source_identity(
        attempt=args.attempt_tag,
        source_revision=source_revision,
        repo_root=repo_root,
    )
    path_state = query_remote_path_state(
        ssh_target=args.ssh_target,
        attempt_tag=args.attempt_tag,
        proxy_host=args.proxy_host,
        retry_count=args.retry_count,
        timeout_s=args.command_timeout_s,
    )

    def inventory():
        return _query_inventory(
            ssh_target=args.ssh_target,
            proxy_host=args.proxy_host,
            retry_count=args.retry_count,
            timeout_s=args.command_timeout_s,
        )

    admission_wait = wait_for_strict_clean_gpus(
        query_inventory=inventory,
        timeout_s=args.gpu_wait_timeout_s,
        poll_interval_s=args.gpu_poll_interval_s,
    )
    if admission_wait.get("classification") != "READY":
        result = {
            "protocol": PROTOCOL,
            "classification": "BLOCKED_RESOURCES",
            "worker_started": False,
            "admission": admission_wait,
        }
        write_json_atomic(local_controller / "result.json", result)
        print(json.dumps(result, sort_keys=True))
        return 4
    selected = admission_wait["selected_gpus"]
    plan = build_attempt_plan(
        attempt_tag=args.attempt_tag,
        source_revision=source_revision,
        source_tree_sha256=source["source_tree_sha256"],
        selected_gpus=selected,
        remote_path_state=path_state,
    )
    identity_rows = [
        {
            "rank": rank,
            "device_index": row["gpu_index"],
            "device_uuid": row["gpu_uuid"],
        }
        for rank, row in enumerate(selected)
    ]
    admission = {
        "protocol": PROTOCOL,
        "classification": "STRICT_CLEAN",
        "rank_rows": [
            {
                "rank": rank,
                "memory_mib": row["memory_used_mib"],
                "utilization_percent": row["utilization_percent"],
                "compute_processes": row["compute_processes"],
            }
            for rank, row in enumerate(selected)
        ],
    }
    source.update({
        "environment": {
            "ssh_target": args.ssh_target,
            "remote_python": args.remote_python,
        },
        "gpu_rank_rows": identity_rows,
        "admission": admission,
    })
    write_json_atomic(local_controller / "plan.json", plan)
    write_json_atomic(local_controller / "source_identity.json", source)
    write_json_atomic(local_controller / "launch_admission.json", admission)
    if args.plan_only:
        result = run_attempt(plan, plan_only=True)
        write_json_atomic(local_controller / "result.json", result)
        print(json.dumps(result, sort_keys=True))
        return 0

    def kerberos_probe():
        return query_local_kerberos(
            minimum_lifetime_seconds=MINIMUM_KERBEROS_LIFETIME_SECONDS
        )

    def remote_writer(current):
        return _create_remote_attempt(
            current,
            source,
            admission,
            repo_root=repo_root,
            ssh_target=args.ssh_target,
            proxy_host=args.proxy_host,
            retry_count=args.retry_count,
            timeout_s=max(args.command_timeout_s, 600),
        )

    def worker_runner(current):
        return _remote_json_command(
            current,
            [
                args.remote_python,
                (
                    f"{current['source_root']}/tools/"
                    "run_lease_sealed_state_commit_overlap.py"
                ),
                "--remote-supervise",
                "--attempt-tag",
                current["attempt_tag"],
                "--source-revision",
                current["source_revision"],
                "--source-tree-sha256",
                current["source_tree_sha256"],
                "--remote-python",
                args.remote_python,
                "--dist-port",
                str(args.dist_port),
                "--selected-gpus-json",
                json.dumps(
                    current["selected_gpus"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ],
            ssh_target=args.ssh_target,
            proxy_host=args.proxy_host,
            retry_count=0,
            timeout_s=max(args.command_timeout_s, 7200),
        )

    def launch_admission_writer(current, observed):
        by_uuid = {row["gpu_uuid"]: row for row in observed}
        rows = [by_uuid[row["gpu_uuid"]] for row in current["selected_gpus"]]
        immediate = {
            "protocol": PROTOCOL,
            "classification": "STRICT_CLEAN",
            "rank_rows": [
                {
                    "rank": rank,
                    "memory_mib": row["memory_used_mib"],
                    "utilization_percent": row["utilization_percent"],
                    "compute_processes": row["compute_processes"],
                }
                for rank, row in enumerate(rows)
            ],
        }
        write_json_atomic(
            local_controller / "launch_admission.json",
            immediate,
        )
        _upload_json(
            immediate,
            f"{current['controller_root']}/launch_admission.json",
            ssh_target=args.ssh_target,
            proxy_host=args.proxy_host,
            retry_count=args.retry_count,
            timeout_s=args.command_timeout_s,
        )
        return {"classification": "PASS"}

    def assembler(current):
        return _remote_json_command(
            current,
            [
                args.remote_python,
                (
                    f"{current['source_root']}/tools/"
                    "assemble_lease_sealed_state_commit_overlap.py"
                ),
                "--raw-root",
                current["raw_root"],
                "--source-identity",
                f"{current['controller_root']}/source_identity.json",
                "--admission",
                f"{current['controller_root']}/launch_admission.json",
                "--output-root",
                current["bundle_root"],
            ],
            ssh_target=args.ssh_target,
            proxy_host=args.proxy_host,
            retry_count=args.retry_count,
            timeout_s=max(args.command_timeout_s, 300),
        )

    def remote_verifier(current):
        result = _remote_json_command(
            current,
            [
                args.remote_python,
                (
                    f"{current['source_root']}/tools/"
                    "verify_lease_sealed_state_commit_overlap.py"
                ),
                current["bundle_root"],
                "--receipt-name",
                "remote_independent_verification.json",
            ],
            ssh_target=args.ssh_target,
            proxy_host=args.proxy_host,
            retry_count=args.retry_count,
            timeout_s=max(args.command_timeout_s, 300),
        )
        _upload_json(
            result,
            (
                f"{current['controller_root']}/"
                "remote-independent-verification.json"
            ),
            ssh_target=args.ssh_target,
            proxy_host=args.proxy_host,
            retry_count=args.retry_count,
            timeout_s=args.command_timeout_s,
        )
        return result

    def downloader(current):
        return _download_compact_bundle(
            current,
            local_attempt_root=local_attempt_root,
            ssh_target=args.ssh_target,
            proxy_host=args.proxy_host,
            timeout_s=max(args.command_timeout_s, 600),
        )

    def local_verifier(_current):
        result = verify_local_downloaded_bundle(
            local_attempt_root / "final_bundle"
        )
        write_json_atomic(
            (
                local_controller
                / "local-streaming-independent-verification.json"
            ),
            result,
        )
        return result

    result = _run_with_terminal_receipt(
        local_controller / "result.json",
        lambda: run_attempt(
            plan,
            dry_run=args.dry_run,
            kerberos_probe=kerberos_probe,
            gpu_probe=inventory,
            remote_writer=remote_writer,
            launch_admission_writer=launch_admission_writer,
            worker_runner=worker_runner,
            assembler=assembler,
            remote_verifier=remote_verifier,
            downloader=downloader,
            local_verifier=local_verifier,
        ),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    local_attempt_root = (
        args.local_attempt_root.resolve()
        if args.local_attempt_root is not None
        else (
            repo_root
            / "artifacts"
            / "lease_sealed_state_commit_overlap"
            / args.attempt_tag
        )
    )
    local_root_existed = local_attempt_root.exists()
    try:
        return _main_unwrapped(argv)
    except Exception as error:
        if not args.remote_supervise and not local_root_existed:
            controller_root = local_attempt_root / "controller"
            receipt_path = controller_root / "result.json"
            if controller_root.is_dir() and not receipt_path.exists():
                write_json_atomic(
                    receipt_path,
                    {
                        "protocol": PROTOCOL,
                        "classification": "CONTROLLER_ERROR",
                        "worker_started": None,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
