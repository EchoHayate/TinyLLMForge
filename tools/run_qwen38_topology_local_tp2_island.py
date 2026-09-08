#!/usr/bin/env python3
"""Safely orchestrate the topology-local TP2 island Stage-0 microgate."""

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
        MAX_GPU_MEMORY_USED_MIB,
        MAX_GPU_UTILIZATION_PERCENT,
        parse_nvidia_smi_inventory,
        query_local_kerberos,
        select_strict_clean_gpus,
        validate_selected_gpu_processes,
        wait_for_strict_clean_gpus,
        write_json_atomic,
    )
    from tools.verify_qwen38_topology_local_tp2_island import (
        LOCAL_RECEIPT_NAME,
        REMOTE_RECEIPT_NAME,
    )
else:
    from run_qwen38_tp4_communication_profile import (
        MAX_GPU_MEMORY_USED_MIB,
        MAX_GPU_UTILIZATION_PERCENT,
        parse_nvidia_smi_inventory,
        query_local_kerberos,
        select_strict_clean_gpus,
        validate_selected_gpu_processes,
        wait_for_strict_clean_gpus,
        write_json_atomic,
    )
    from verify_qwen38_topology_local_tp2_island import (
        LOCAL_RECEIPT_NAME,
        REMOTE_RECEIPT_NAME,
    )


APPROVED_REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
DEFAULT_REMOTE_PYTHON = "/data00/home/sitian/tllm/env/bin/python"
DEFAULT_MODEL_ROOT = (
    f"{APPROVED_REMOTE_ROOT}/models/Qwen3.8-27B/"
    "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
)
DEFAULT_SSH_TARGET = "sitian@10.232.195.203"
DEFAULT_PROXY_HOST = "jump-proxy-hl"
DEFAULT_DIST_PORT = 29683
DEFAULT_RETRY_COUNT = 3
DEFAULT_COMMAND_TIMEOUT_S = 60
DEFAULT_GPU_WAIT_TIMEOUT_S = 21_600
DEFAULT_GPU_POLL_INTERVAL_S = 15
MINIMUM_KERBEROS_LIFETIME_SECONDS = 10_800
EXPECTED_KERBEROS_PRINCIPAL = "sitian@BYTEDANCE.COM"
EXPECTED_KERBEROS_TGT = "krbtgt/BYTEDANCE.COM@BYTEDANCE.COM"
MODEL_REPOSITORY = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
PLAN_SCHEMA = "qwen38.topology-local-tp2-island-plan.v1"
ATTEMPT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
HEX40_PATTERN = re.compile(r"^[0-9a-f]{40}$")
HEX64_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_LINK_COST = {"PIX": 0, "PXB": 1, "PHB": 2, "NODE": 3, "SYS": 4}


def _select_best_pair_groups(rows):
    directed = {}
    for row in rows:
        key = (row.get("left_rank"), row.get("right_rank"))
        link = row.get("link")
        if (
            key[0] == key[1]
            or key[0] not in range(4)
            or key[1] not in range(4)
            or link not in _LINK_COST
            or key in directed
        ):
            raise ValueError("topology row is invalid")
        directed[key] = _LINK_COST[link]
    costs = {}
    for left in range(4):
        for right in range(left + 1, 4):
            if (
                (left, right) not in directed
                or (right, left) not in directed
                or directed[(left, right)] != directed[(right, left)]
            ):
                raise ValueError("topology rows are incomplete or asymmetric")
            costs[(left, right)] = directed[(left, right)]
    matchings = (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )
    return min(
        matchings,
        key=lambda matching: (
            tuple(sorted(costs[pair] for pair in matching)),
            matching,
        ),
    )


def _below(path, root):
    candidate = PurePosixPath(path)
    parent = PurePosixPath(root)
    return (
        candidate.is_absolute()
        and candidate != parent
        and candidate.is_relative_to(parent)
    )


def _validate_path_state(state):
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


def _validate_topology_rows(rows):
    pair_groups = _select_best_pair_groups(rows)
    return [dict(row) for row in rows], [
        list(group) for group in pair_groups
    ]


def build_attempt_plan(
    *,
    attempt_tag,
    source_revision,
    source_tree_sha256,
    selected_gpus,
    topology_rows,
    remote_path_state,
    dist_port=DEFAULT_DIST_PORT,
    remote_root=APPROVED_REMOTE_ROOT,
    model_root=DEFAULT_MODEL_ROOT,
):
    if remote_root != APPROVED_REMOTE_ROOT:
        raise ValueError("remote root is not approved")
    if (
        not isinstance(attempt_tag, str)
        or not ATTEMPT_PATTERN.fullmatch(attempt_tag)
        or ".." in attempt_tag
    ):
        raise ValueError("attempt tag is invalid")
    if not isinstance(source_revision, str) or not HEX40_PATTERN.fullmatch(
        source_revision
    ):
        raise ValueError("source revision is invalid")
    if (
        not isinstance(source_tree_sha256, str)
        or not HEX64_PATTERN.fullmatch(source_tree_sha256)
    ):
        raise ValueError("source tree digest is invalid")
    if (
        type(dist_port) is not int
        or not 1024 <= dist_port <= 65535
    ):
        raise ValueError("distributed port is invalid")
    if not _below(model_root, "/data00/home/sitian"):
        raise ValueError("model root is invalid")
    _validate_path_state(remote_path_state)
    selected = select_strict_clean_gpus(list(selected_gpus))
    if len(selected) != 4:
        raise ValueError("exactly four strict-clean GPUs are required")
    frozen_topology, pair_groups = _validate_topology_rows(topology_rows)

    attempt_root = f"{remote_root}/attempts/{attempt_tag}"
    runtime_root = f"{attempt_root}/runtime"
    paths = {
        "attempt_root": attempt_root,
        "source_root": f"{attempt_root}/source",
        "raw_root": f"{attempt_root}/raw",
        "bundle_root": f"{attempt_root}/final_bundle",
        "controller_root": f"{attempt_root}/controller",
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
        for path in (*paths.values(), *environment.values(), model_root)
    ):
        raise ValueError("planned path escapes approved root")
    return {
        "schema": PLAN_SCHEMA,
        "attempt_tag": attempt_tag,
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "model_root": model_root,
        "remote_root": remote_root,
        **paths,
        "environment": environment,
        "selected_gpus": [dict(row) for row in selected],
        "topology_rows": frozen_topology,
        "pair_groups": pair_groups,
        "dist_port": dist_port,
        "workload": {
            "active_token_groups": [1, 4, 8],
            "warmup_pairs_per_shape": 2,
            "measured_pairs_per_shape": 15,
            "migration_warmups": 2,
            "migration_measurements": 15,
        },
        "thresholds": {
            "token_1_minimum_speedup": 0.05,
            "token_4_8_minimum_geometric_speedup": 0.05,
            "maximum_p99_regression": 0.03,
            "minimum_improving_pairs": 11,
            "maximum_host_median_regression": 0.10,
            "maximum_break_even_tokens": 32,
            "maximum_steady_increment_bytes": 1920 * 1024 * 1024,
            "maximum_peak_allocated_ratio": 0.98,
        },
    }


def _validate_plan(plan):
    if (
        not isinstance(plan, dict)
        or plan.get("schema") != PLAN_SCHEMA
        or plan.get("remote_root") != APPROVED_REMOTE_ROOT
        or not HEX40_PATTERN.fullmatch(
            str(plan.get("source_revision", ""))
        )
        or not HEX64_PATTERN.fullmatch(
            str(plan.get("source_tree_sha256", ""))
        )
        or plan.get("model_repository") != MODEL_REPOSITORY
        or plan.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("controller plan is invalid")
    for key in (
        "attempt_root",
        "source_root",
        "raw_root",
        "bundle_root",
        "controller_root",
    ):
        if not _below(plan.get(key, ""), APPROVED_REMOTE_ROOT):
            raise ValueError("controller plan path is invalid")
    selected = select_strict_clean_gpus(plan.get("selected_gpus", []))
    if len(selected) != 4:
        raise ValueError("exactly four strict-clean GPUs are required")
    expected_groups = [
        list(group)
        for group in _select_best_pair_groups(plan["topology_rows"])
    ]
    if plan.get("pair_groups") != expected_groups:
        raise ValueError("frozen pair topology is invalid")
    return tuple(selected)


def build_remote_worker_commands(
    plan,
    *,
    python_path=DEFAULT_REMOTE_PYTHON,
):
    selected = _validate_plan(plan)
    if (
        not isinstance(python_path, str)
        or not _below(python_path, "/data00/home/sitian")
    ):
        raise ValueError("remote Python path is invalid")
    worker = (
        f"{plan['source_root']}/tools/"
        "qwen38_topology_local_tp2_island_worker.py"
    )
    visible = ",".join(
        str(row["gpu_index"]) for row in selected
    )
    pair_groups = ";".join(
        ",".join(str(rank) for rank in group)
        for group in plan["pair_groups"]
    )
    commands = []
    for rank in range(4):
        commands.append({
            "argv": [
                python_path,
                worker,
                "--attempt",
                plan["attempt_tag"],
                "--source-revision",
                plan["source_revision"],
                "--model-root",
                plan["model_root"],
                "--output-root",
                plan["raw_root"],
                "--pair-groups",
                pair_groups,
            ],
            "environment": {
                "WORLD_SIZE": "4",
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(plan["dist_port"]),
                "CUDA_VISIBLE_DEVICES": visible,
                "PYTHONPATH": plan["source_root"],
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                **plan["environment"],
            },
        })
    return tuple(commands)


def run_ssh_with_retry(
    argv,
    *,
    retry_count,
    runner=subprocess.run,
    timeout_s=None,
    input_text=None,
):
    if type(retry_count) is not int or retry_count < 0:
        raise ValueError("retry count is invalid")
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
        or any(
            not isinstance(argument, str)
            or not argument
            or "\0" in argument
            for argument in remote_argv
        )
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


def select_owned_process_groups(
    process_rows,
    *,
    attempt_tag,
    registered_pgids,
):
    if (
        not isinstance(process_rows, (list, tuple))
        or not isinstance(registered_pgids, set)
        or any(type(value) is not int or value <= 0 for value in registered_pgids)
    ):
        raise ValueError("owned process inventory is invalid")
    observed = {
        row.get("pgid")
        for row in process_rows
        if (
            isinstance(row, dict)
            and row.get("attempt") == attempt_tag
            and row.get("pgid") in registered_pgids
        )
    }
    return tuple(sorted(observed))


def compact_download_members(plan):
    _validate_plan(plan)
    return (
        "final_bundle",
        "controller/source_identity.json",
        "controller/plan.json",
        "controller/launch_admission.json",
        "controller/supervisor_receipt.json",
    )


def _validate_kerberos(receipt):
    return (
        isinstance(receipt, dict)
        and receipt.get("classification") == "READY"
        and receipt.get("principal") == EXPECTED_KERBEROS_PRINCIPAL
        and receipt.get("tgt_principal") == EXPECTED_KERBEROS_TGT
        and type(receipt.get("remaining_lifetime_seconds")) is int
        and receipt["remaining_lifetime_seconds"]
        >= MINIMUM_KERBEROS_LIFETIME_SECONDS
    )


def _validate_frozen_clean_inventory(selected, observed):
    validated = validate_selected_gpu_processes(
        selected=selected,
        observed=observed,
        owned_pids=set(),
    )
    if any(
        row["memory_used_mib"] > MAX_GPU_MEMORY_USED_MIB
        or row["utilization_percent"] > MAX_GPU_UTILIZATION_PERCENT
        or row["compute_processes"]
        for row in validated
    ):
        raise ValueError("frozen GPU selection is no longer strict-clean")
    return validated


def run_attempt(
    plan,
    *,
    dry_run,
    kerberos_probe,
    gpu_probe,
    remote_writer=None,
    worker_runner=None,
    assembler=None,
    remote_verifier=None,
    downloader=None,
    local_sealer=None,
    local_checker=None,
    terminal_writer=None,
):
    selected = _validate_plan(plan)
    kerberos = kerberos_probe()
    if not _validate_kerberos(kerberos):
        result = {
            "classification": "BLOCKED_KERBEROS",
            "worker_started": False,
            "kerberos": kerberos,
        }
        if callable(terminal_writer):
            terminal_writer(result)
        return result
    observed = list(gpu_probe())
    try:
        _validate_frozen_clean_inventory(selected, observed)
    except Exception as error:
        result = {
            "classification": "BLOCKED_ADMISSION",
            "worker_started": False,
            "error": f"{type(error).__name__}: {error}",
        }
        if callable(terminal_writer):
            terminal_writer(result)
        return result
    if dry_run:
        return {
            "classification": "DRY_RUN_READY",
            "worker_started": False,
            "plan": plan,
        }
    try:
        if not callable(remote_writer):
            raise RuntimeError("remote writer is missing")
        created = remote_writer(plan)
        if not isinstance(created, dict) or created.get("created") is not True:
            raise RuntimeError("remote attempt creation failed")
        _validate_frozen_clean_inventory(
            selected, list(gpu_probe())
        )
        callbacks = (
            ("worker", worker_runner),
            ("assembler", assembler),
            ("remote verifier", remote_verifier),
            ("downloader", downloader),
            ("local sealer", local_sealer),
            ("local checker", local_checker),
        )
        outputs = {}
        for name, callback in callbacks:
            if not callable(callback):
                raise RuntimeError(f"{name} adapter is missing")
            outputs[name] = callback(plan)
            if name == "worker" and (
                not isinstance(outputs[name], dict)
                or outputs[name].get("classification") != "PASS"
            ):
                raise RuntimeError("worker campaign failed")
            if name == "downloader" and (
                not isinstance(outputs[name], dict)
                or outputs[name].get("downloaded") is not True
            ):
                raise RuntimeError("compact download failed")
        classified = (
            outputs["assembler"],
            outputs["remote verifier"],
            outputs["local sealer"],
            outputs["local checker"],
        )
        if not all(
            isinstance(value, dict)
            and isinstance(value.get("classification"), str)
            for value in classified
        ):
            raise RuntimeError("producer/verifier result is invalid")
        classifications = {
            value["classification"] for value in classified
        }
        if len(classifications) != 1:
            raise RuntimeError(
                "producer/verifier classification disagreement"
            )
        result = {
            "classification": classifications.pop(),
            "worker_started": True,
            "producer": outputs["assembler"],
            "remote_verification": outputs["remote verifier"],
            "local_verification": outputs["local sealer"],
            "local_check": outputs["local checker"],
        }
    except Exception as error:
        result = {
            "classification": "FAILED_CONTROLLER",
            "worker_started": "worker" in locals().get("outputs", {}),
            "error": f"{type(error).__name__}: {error}",
        }
    if callable(terminal_writer):
        terminal_writer(result)
    return result


def capture_source_identity(repo_root, attempt):
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

    revision = git("rev-parse", "HEAD").strip()
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
        revision,
        "tinyvllm",
        "tools",
    ).encode("utf-8")
    return {
        "attempt": attempt,
        "source_revision": revision,
        "source_tree_sha256": hashlib.sha256(tree).hexdigest(),
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
    }


def _query_remote_path_state(args):
    attempt_root = f"{args.remote_root}/attempts/{args.attempt}"
    script = "\n".join([
        "import json,os,sys",
        "root,attempt=sys.argv[1:]",
        "print(json.dumps({",
        "'attempt_exists':os.path.lexists(attempt),",
        "'attempt_parent_is_symlink':os.path.islink(os.path.dirname(attempt)),",
        "'remote_root_is_symlink':os.path.islink(root),",
        "},sort_keys=True))",
    ])
    result = _remote_run(
        ssh_target=args.ssh_target,
        remote_argv=[
            "python3", "-c", script, args.remote_root, attempt_root
        ],
        proxy_host=args.proxy_host,
        retry_count=args.retry_count,
        timeout_s=args.command_timeout_s,
    )
    return json.loads(result.stdout)


def _query_remote_inventory(args):
    script = "\n".join([
        "import json,subprocess",
        "gpu=subprocess.run([",
        "'nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',",
        "'--format=csv,noheader,nounits'],check=True,text=True,capture_output=True)",
        "proc=subprocess.run([",
        "'nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory',",
        "'--format=csv,noheader,nounits'],check=True,text=True,capture_output=True)",
        "print(json.dumps({'gpu':gpu.stdout,'proc':proc.stdout},sort_keys=True))",
    ])
    result = _remote_run(
        ssh_target=args.ssh_target,
        remote_argv=["python3", "-c", script],
        proxy_host=args.proxy_host,
        retry_count=args.retry_count,
        timeout_s=args.command_timeout_s,
    )
    payload = json.loads(result.stdout)
    return parse_nvidia_smi_inventory(payload["gpu"], payload["proc"])


def _parse_topology_rows(matrix, selected):
    matrix = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", matrix)
    lines = [line.split() for line in matrix.splitlines() if line.strip()]
    header = next(
        row for row in lines
        if row and all(value.startswith("GPU") for value in row[:4])
    )
    physical = [row["gpu_index"] for row in selected]
    columns = {
        int(name.removeprefix("GPU")): index
        for index, name in enumerate(header)
        if name.startswith("GPU") and name[3:].isdigit()
    }
    table = {
        int(row[0].removeprefix("GPU")): row[1:]
        for row in lines
        if row[0].startswith("GPU") and row[0][3:].isdigit()
    }
    rows = []
    for left_rank, left_gpu in enumerate(physical):
        for right_rank, right_gpu in enumerate(physical):
            if left_rank == right_rank:
                continue
            rows.append({
                "left_rank": left_rank,
                "right_rank": right_rank,
                "link": table[left_gpu][columns[right_gpu]],
            })
    _select_best_pair_groups(rows)
    return rows


def _query_remote_topology(args, selected):
    result = _remote_run(
        ssh_target=args.ssh_target,
        remote_argv=["nvidia-smi", "topo", "-m"],
        proxy_host=args.proxy_host,
        retry_count=args.retry_count,
        timeout_s=args.command_timeout_s,
    )
    return _parse_topology_rows(result.stdout, selected)


def _upload_json(args, payload, remote_path):
    script = "\n".join([
        "import os,sys,tempfile",
        "path=sys.argv[1]",
        "fd,temp=tempfile.mkstemp(prefix='.upload.',dir=os.path.dirname(path))",
        "with os.fdopen(fd,'w',encoding='utf-8') as handle:",
        " handle.write(sys.stdin.read()); handle.flush(); os.fsync(handle.fileno())",
        "os.replace(temp,path)",
    ])
    _remote_run(
        ssh_target=args.ssh_target,
        remote_argv=["python3", "-c", script, remote_path],
        proxy_host=args.proxy_host,
        retry_count=args.retry_count,
        timeout_s=args.command_timeout_s,
        input_text=json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ) + "\n",
    )


def _create_remote_attempt(args, plan, source_identity):
    directories = [
        plan["source_root"],
        plan["raw_root"],
        plan["controller_root"],
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
        ssh_target=args.ssh_target,
        remote_argv=[
            "python3", "-c", script, plan["attempt_root"], *directories
        ],
        proxy_host=args.proxy_host,
        retry_count=args.retry_count,
        timeout_s=args.command_timeout_s,
    )
    archive = subprocess.Popen(
        [
            "git", "-C", str(Path(__file__).resolve().parents[1]),
            "archive", "--format=tar", plan["source_revision"],
            "tinyvllm", "tools",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    receiver = subprocess.run(
        _ssh_argv(
            args.ssh_target,
            ["tar", "-xf", "-", "-C", plan["source_root"]],
            args.proxy_host,
        ),
        stdin=archive.stdout,
        capture_output=True,
        check=False,
        timeout=max(args.command_timeout_s, 600),
    )
    if archive.stdout is not None:
        archive.stdout.close()
    archive_error = archive.stderr.read() if archive.stderr else b""
    archive_code = archive.wait()
    if archive_code != 0 or receiver.returncode != 0:
        raise RuntimeError(
            archive_error.decode(errors="replace")
            or receiver.stderr.decode(errors="replace")
            or "source staging failed"
        )
    _upload_json(
        args,
        source_identity,
        f"{plan['controller_root']}/source_identity.json",
    )
    _upload_json(args, plan, f"{plan['controller_root']}/plan.json")
    return {"created": True}


def _descendant_pids(parents):
    owned = set(parents)
    changed = True
    while changed:
        changed = False
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "stat").read_text()
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


def _enrich(identity, payload):
    return {**identity, **payload}


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def _write_jsonl(path, rows):
    Path(path).write_text("".join(
        json.dumps(
            row, sort_keys=True, separators=(",", ":"), allow_nan=False
        ) + "\n"
        for row in rows
    ))


def _build_parameter_slice_manifest(rows, *, identity):
    by_logical_rank = {}
    for row in rows:
        by_logical_rank.setdefault(
            row["logical_rank"],
            [],
        ).append(row.get("parameter_digests"))
    replica_digest_match = (
        set(by_logical_rank) == {0, 1}
        and all(
            len(values) == 2
            and values[0] == values[1]
            and isinstance(values[0], dict)
            and bool(values[0])
            for values in by_logical_rank.values()
        )
    )
    reconstruction_match = (
        len(rows) == 4
        and all(
            row.get("checkpoint_reconstruction_match") is True
            and isinstance(
                row.get("checkpoint_full_parameter_digests"),
                dict,
            )
            and row.get("checkpoint_full_parameter_digests")
            == row.get("reconstructed_full_parameter_digests")
            for row in rows
        )
    )
    return {
        **identity,
        "layer_index": 0,
        "linear_attention_only": True,
        "full_attention_parameters_changed": False,
        "mlp_parameters_changed": False,
        "replica_digest_match": replica_digest_match,
        "checkpoint_reconstruction_match": reconstruction_match,
        "rank_parameter_evidence": [
            {
                "rank": row["rank"],
                "logical_rank": row["logical_rank"],
                "parameter_digests": row.get("parameter_digests"),
                "checkpoint_full_parameter_digests": row.get(
                    "checkpoint_full_parameter_digests"
                ),
                "reconstructed_full_parameter_digests": row.get(
                    "reconstructed_full_parameter_digests"
                ),
                "checkpoint_reconstruction_match": row.get(
                    "checkpoint_reconstruction_match"
                ),
            }
            for row in sorted(rows, key=lambda value: value["rank"])
        ],
    }


def _finalize_worker_outputs(plan, rank_exit_codes):
    raw = Path(plan["raw_root"])
    identity = {
        "attempt": plan["attempt_tag"],
        "source_revision": plan["source_revision"],
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "pair_groups": plan["pair_groups"],
    }
    timing_rows = []
    migration_rows = []
    memory_rows = []
    cleanup_rows = []
    for rank in range(4):
        timing_rows.extend(
            _enrich(identity, row)
            for row in _read_jsonl(
                raw / f"measurement_rows.rank-{rank}.jsonl"
            )
        )
        migration_rows.extend(
            _enrich(identity, row)
            for row in _read_jsonl(
                raw / f"migration_rows.rank-{rank}.jsonl"
            )
        )
        memory_rows.append(_enrich(
            identity,
            json.loads(
                (raw / f"memory.rank-{rank}.json").read_text()
            ),
        ))
        cleanup_rows.append(_enrich(
            identity,
            json.loads(
                (raw / f"cleanup.rank-{rank}.json").read_text()
            ),
        ))
    timing_rows.sort(
        key=lambda row: (
            row["active_tokens"], row["repetition"], row["rank"]
        )
    )
    migration_rows.sort(
        key=lambda row: (row["repetition"], row["rank"])
    )
    _write_jsonl(raw / "measurement_rows.jsonl", timing_rows)
    _write_jsonl(raw / "migration_rows.jsonl", migration_rows)
    _write_jsonl(raw / "memory_rows.jsonl", memory_rows)
    write_json_atomic(raw / "model_identity.json", {
        **identity,
        "hidden_size": 5120,
        "layer_count": 64,
        "linear_attention_layer_count": 48,
        "full_attention_layer_count": 16,
        "dtype": "bfloat16",
    })
    write_json_atomic(raw / "topology.json", {
        **identity,
        "selection_frozen": True,
        "selected_pair_groups": plan["pair_groups"],
        "rows": plan["topology_rows"],
    })
    write_json_atomic(raw / "workload_manifest.json", {
        **identity,
        **plan["workload"],
    })
    parameter_rows = [
        row
        for row in timing_rows
        if row.get("active_tokens") == 1
        and row.get("repetition") == 0
    ]
    write_json_atomic(
        raw / "parameter_slice_manifest.json",
        _build_parameter_slice_manifest(
            parameter_rows,
            identity=identity,
        ),
    )
    lifecycle_rows = [
        _enrich(
            identity,
            json.loads(
                (raw / f"lifecycle.rank-{rank}.json").read_text()
            ),
        )
        for rank in range(4)
    ]
    _write_jsonl(raw / "lifecycle_rows.jsonl", lifecycle_rows)
    cleanup = {
        **identity,
        "classification": (
            "CLEAN"
            if rank_exit_codes == [0, 0, 0, 0]
            and all(
                row.get("classification") == "CLEAN"
                and row.get("candidate_state_unpublished") is True
                for row in cleanup_rows
            )
            else "DIRTY"
        ),
        "rank_rows": [
            {
                **row,
                "owned_children_remaining": [],
                "task_files_outside_attempt_root": [],
            }
            for row in cleanup_rows
        ],
    }
    write_json_atomic(raw / "cleanup.json", cleanup)
    return {
        "timing_row_count": len(timing_rows),
        "migration_row_count": len(migration_rows),
        "cleanup": cleanup,
    }


def _reap_worker_processes(
    processes,
    *,
    terminate=True,
    terminate_grace_s=10,
    kill_grace_s=10,
):
    if terminate:
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    terminate_deadline = time.monotonic() + terminate_grace_s
    codes = []
    survivors = []
    for process in processes:
        code = process.poll()
        if code is not None:
            codes.append(code)
            continue
        try:
            codes.append(process.wait(timeout=max(
                0,
                terminate_deadline - time.monotonic(),
            )))
        except subprocess.TimeoutExpired:
            codes.append(None)
            survivors.append(process)
    for process in survivors:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    kill_deadline = time.monotonic() + kill_grace_s
    for index, process in enumerate(processes):
        if codes[index] is not None:
            continue
        try:
            codes[index] = process.wait(timeout=max(
                0,
                kill_deadline - time.monotonic(),
            ))
        except subprocess.TimeoutExpired:
            codes[index] = None
    return codes


def _completed_worker_failure(processes):
    return any(
        code is not None and code != 0
        for code in (process.poll() for process in processes)
    )


def supervise_remote_workers(
    plan,
    *,
    python_path=DEFAULT_REMOTE_PYTHON,
    poll_interval_s=1,
    timeout_s=7200,
):
    selected = _validate_plan(plan)
    raw = Path(plan["raw_root"])
    controller = Path(plan["controller_root"])
    for path in (
        raw,
        controller,
        *(Path(value) for value in plan["environment"].values()),
    ):
        path.mkdir(parents=True, exist_ok=True)
    processes = []
    streams = []
    commands = build_remote_worker_commands(
        plan, python_path=python_path
    )
    started = time.monotonic()
    violations = []
    try:
        for rank, command in enumerate(commands):
            stdout = (controller / f"rank-{rank}.stdout").open("a")
            stderr = (controller / f"rank-{rank}.stderr").open("a")
            streams.extend((stdout, stderr))
            environment = dict(os.environ)
            environment.update(command["environment"])
            process = subprocess.Popen(
                command["argv"],
                cwd=plan["source_root"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            processes.append(process)
        stop_requested = False
        while any(process.poll() is None for process in processes):
            if _completed_worker_failure(processes):
                violations.append("worker rank exited non-zero")
                stop_requested = True
                break
            owned = _descendant_pids({process.pid for process in processes})
            try:
                inventory = _local_gpu_inventory()
                validate_selected_gpu_processes(
                    selected=selected,
                    observed=inventory,
                    owned_pids=owned,
                )
            except Exception as error:
                violations.append(f"{type(error).__name__}: {error}")
                stop_requested = True
                break
            if time.monotonic() - started > timeout_s:
                violations.append("worker timeout")
                stop_requested = True
                break
            time.sleep(poll_interval_s)
        codes = _reap_worker_processes(
            processes,
            terminate=stop_requested,
        )
    finally:
        for stream in streams:
            stream.close()
    try:
        finalized = _finalize_worker_outputs(plan, codes)
        finalization = {"complete": True, "error": None}
    except Exception as error:
        violations.append(
            f"worker output finalization failed: "
            f"{type(error).__name__}: {error}"
        )
        finalized = {
            "timing_row_count": 0,
            "migration_row_count": 0,
            "cleanup": {
                "classification": "DIRTY",
                "rank_rows": [],
            },
        }
        finalization = {
            "complete": False,
            "error": f"{type(error).__name__}: {error}",
        }
    receipt = {
        "classification": (
            "PASS"
            if codes == [0, 0, 0, 0]
            and not violations
            and finalized["cleanup"]["classification"] == "CLEAN"
            else "FAIL"
        ),
        "attempt": plan["attempt_tag"],
        "source_revision": plan["source_revision"],
        "owned_pids": [process.pid for process in processes],
        "rank_exit_codes": codes,
        "violations": violations,
        "finalization": finalization,
        **finalized,
    }
    write_json_atomic(
        controller / "supervisor_receipt.json", receipt
    )
    return receipt


def _local_gpu_inventory():
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
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return parse_nvidia_smi_inventory(gpu.stdout, proc.stdout)


def _remote_json(args, argv, timeout_s):
    result = _remote_run(
        ssh_target=args.ssh_target,
        remote_argv=argv,
        proxy_host=args.proxy_host,
        retry_count=args.retry_count,
        timeout_s=timeout_s,
    )
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError("remote JSON result is invalid") from error
    if not isinstance(payload, dict):
        raise RuntimeError("remote JSON result is invalid")
    return payload


def _download_bundle(args, plan, local_attempt_root):
    local_root = Path(local_attempt_root).resolve()
    if (local_root / "final_bundle").exists():
        raise ValueError("local final bundle must not already exist")
    local_root.mkdir(parents=True, exist_ok=True)
    members = compact_download_members(plan)
    sender = subprocess.Popen(
        _ssh_argv(
            args.ssh_target,
            [
                "tar", "-cf", "-", "-C", plan["attempt_root"], *members
            ],
            args.proxy_host,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    receiver = subprocess.run(
        ["tar", "-xf", "-", "-C", str(local_root)],
        stdin=sender.stdout,
        capture_output=True,
        check=False,
        timeout=max(args.command_timeout_s, 600),
    )
    if sender.stdout is not None:
        sender.stdout.close()
    sender_error = sender.stderr.read() if sender.stderr else b""
    sender_code = sender.wait()
    if sender_code != 0 or receiver.returncode != 0:
        raise RuntimeError(
            sender_error.decode(errors="replace")
            or receiver.stderr.decode(errors="replace")
            or "compact download failed"
        )
    return {"downloaded": True}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--remote-root", default=APPROVED_REMOTE_ROOT)
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--remote-python", default=DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--ssh-target", default=DEFAULT_SSH_TARGET)
    parser.add_argument("--proxy-host", default=DEFAULT_PROXY_HOST)
    parser.add_argument("--dist-port", type=int, default=DEFAULT_DIST_PORT)
    parser.add_argument(
        "--retry-count", type=int, default=DEFAULT_RETRY_COUNT
    )
    parser.add_argument(
        "--command-timeout-s",
        type=int,
        default=DEFAULT_COMMAND_TIMEOUT_S,
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
    parser.add_argument("--local-attempt-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--remote-supervise", action="store_true")
    parser.add_argument("--plan-json", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.remote_supervise:
        if args.plan_json is None:
            raise ValueError("remote supervisor requires plan JSON")
        plan = json.loads(args.plan_json.read_text())
        result = supervise_remote_workers(
            plan, python_path=args.remote_python
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if result["classification"] == "PASS" else 2

    repo_root = Path(__file__).resolve().parents[1]
    source = capture_source_identity(repo_root, args.attempt)
    kerberos = query_local_kerberos(
        minimum_lifetime_seconds=MINIMUM_KERBEROS_LIFETIME_SECONDS
    )
    if not _validate_kerberos(kerberos):
        print(json.dumps({
            "classification": "BLOCKED_KERBEROS",
            "worker_started": False,
            "kerberos": kerberos,
        }, sort_keys=True))
        return 3
    path_state = _query_remote_path_state(args)

    def inventory():
        return _query_remote_inventory(args)

    admission = wait_for_strict_clean_gpus(
        query_inventory=inventory,
        timeout_s=args.gpu_wait_timeout_s,
        poll_interval_s=args.gpu_poll_interval_s,
    )
    if admission.get("classification") != "READY":
        print(json.dumps({
            "classification": "BLOCKED_ADMISSION",
            "worker_started": False,
            "admission": admission,
        }, sort_keys=True))
        return 4
    selected = admission["selected_gpus"]
    topology_rows = _query_remote_topology(args, selected)
    plan = build_attempt_plan(
        attempt_tag=args.attempt,
        source_revision=source["source_revision"],
        source_tree_sha256=source["source_tree_sha256"],
        selected_gpus=selected,
        topology_rows=topology_rows,
        remote_path_state=path_state,
        dist_port=args.dist_port,
        remote_root=args.remote_root,
        model_root=args.model_root,
    )
    identity = {
        **source,
        "pair_groups": plan["pair_groups"],
    }
    admission_payload = {
        **identity,
        "classification": "ADMITTED",
        "rank_rows": [
            {
                "rank": rank,
                "device_uuid": row["gpu_uuid"],
                "memory_used_mib": row["memory_used_mib"],
                "utilization_percent": row["utilization_percent"],
                "foreign_compute_processes": row["compute_processes"],
            }
            for rank, row in enumerate(selected)
        ],
    }
    local_attempt = (
        args.local_attempt_root.resolve()
        if args.local_attempt_root is not None
        else repo_root / "artifacts" / (
            "qwen38_topology_local_tp2_islands"
        ) / args.attempt
    )
    local_controller = local_attempt / "controller"
    local_controller.mkdir(parents=True, exist_ok=True)
    write_json_atomic(local_controller / "plan.json", plan)
    write_json_atomic(
        local_controller / "source_identity.json", identity
    )
    write_json_atomic(
        local_controller / "launch_admission.json",
        admission_payload,
    )
    if args.dry_run:
        result = {
            "classification": "DRY_RUN_READY",
            "worker_started": False,
            "plan": plan,
        }
        write_json_atomic(
            local_controller / "controller_result.json", result
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    def create_remote(current):
        receipt = _create_remote_attempt(args, current, identity)
        _upload_json(
            args,
            admission_payload,
            f"{current['controller_root']}/launch_admission.json",
        )
        return receipt

    def run_workers(current):
        return _remote_json(
            args,
            [
                args.remote_python,
                f"{current['source_root']}/tools/"
                "run_qwen38_topology_local_tp2_island.py",
                "--attempt",
                current["attempt_tag"],
                "--remote-root",
                current["remote_root"],
                "--model-root",
                current["model_root"],
                "--remote-python",
                args.remote_python,
                "--remote-supervise",
                "--plan-json",
                f"{current['controller_root']}/plan.json",
            ],
            7200,
        )

    def assemble(current):
        return _remote_json(
            args,
            [
                args.remote_python,
                f"{current['source_root']}/tools/"
                "assemble_qwen38_topology_local_tp2_island.py",
                "--attempt-root",
                current["attempt_root"],
            ],
            600,
        )

    def remote_verify(current):
        return _remote_json(
            args,
            [
                args.remote_python,
                f"{current['source_root']}/tools/"
                "verify_qwen38_topology_local_tp2_island.py",
                current["bundle_root"],
                "--receipt-name",
                REMOTE_RECEIPT_NAME,
            ],
            600,
        )

    def download(current):
        return _download_bundle(args, current, local_attempt)

    def local_seal(_current):
        if __package__:
            from tools.verify_qwen38_topology_local_tp2_island import (
                verify_bundle,
            )
        else:
            from verify_qwen38_topology_local_tp2_island import (
                verify_bundle,
            )
        return verify_bundle(
            local_attempt / "final_bundle",
            receipt_name=LOCAL_RECEIPT_NAME,
            seal_terminal=True,
        )

    def local_check(_current):
        if __package__:
            from tools.verify_qwen38_topology_local_tp2_island import (
                verify_bundle,
            )
        else:
            from verify_qwen38_topology_local_tp2_island import (
                verify_bundle,
            )
        return verify_bundle(
            local_attempt / "final_bundle",
            receipt_name=None,
            check_only=True,
        )

    def terminal_writer(receipt):
        write_json_atomic(
            local_controller / "controller_result.json", receipt
        )

    result = run_attempt(
        plan,
        dry_run=False,
        kerberos_probe=lambda: query_local_kerberos(
            minimum_lifetime_seconds=(
                MINIMUM_KERBEROS_LIFETIME_SECONDS
            )
        ),
        gpu_probe=inventory,
        remote_writer=create_remote,
        worker_runner=run_workers,
        assembler=assemble,
        remote_verifier=remote_verify,
        downloader=download,
        local_sealer=local_seal,
        local_checker=local_check,
        terminal_writer=terminal_writer,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["classification"] not in {
        "FAILED_CONTROLLER",
        "BLOCKED_KERBEROS",
        "BLOCKED_ADMISSION",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
