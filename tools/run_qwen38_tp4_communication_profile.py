from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import statistics
import subprocess
import tempfile
import time
from typing import Callable, Mapping


MAX_GPU_MEMORY_USED_MIB = 1024
MAX_GPU_UTILIZATION_PERCENT = 5
APPROVED_REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
PLAN_SCHEMA_VERSION = "qwen38.tp4-communication-profile-plan.v1"
DEFAULT_SSH_TARGET = "sitian@10.232.195.203"
DEFAULT_COMMAND_TIMEOUT_S = 7200
DEFAULT_RETRY_COUNT = 2
MAX_COMMAND_TIMEOUT_S = 24 * 60 * 60
MAX_RETRY_COUNT = 10
DEFAULT_GPU_WAIT_TIMEOUT_S = 0
DEFAULT_GPU_POLL_INTERVAL_S = 30
MAX_GPU_MONITOR_INTERVAL_S = 24 * 60 * 60
EXPECTED_KERBEROS_PRINCIPAL = "sitian@BYTEDANCE.COM"
EXPECTED_KERBEROS_TGT = "krbtgt/BYTEDANCE.COM@BYTEDANCE.COM"
MINIMUM_KERBEROS_LIFETIME_SECONDS = 5400
KERBEROS_TIMESTAMP_FORMAT = "%Y%m%d%H%M%S"
WORKLOAD_ORDER = ("P0", "P1", "Q0", "Q1", "Q2")
WORKLOADS = {
    "P0": ("causal", 256, 128, 1),
    "P1": ("causal", 2048, 128, 1),
    "Q0": ("online", 256, 128, 4),
    "Q1": ("online", 256, 128, 8),
    "Q2": ("online", 2048, 128, 4),
}


class ResourceIdentityError(ValueError):
    """Raised when the frozen GPU identity or ownership contract drifts."""


class CleanupSafetyError(ValueError):
    """Raised when cleanup reports touching a process it does not own."""


@dataclass(frozen=True)
class ProfileCase:
    case_id: str
    workload: str
    family: str
    prompt_tokens: int
    output_tokens: int
    concurrency: int
    phase: str
    repetition: int
    profiled: bool
    overhead_pair_id: str | None
    representative: bool = False


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_gpu_row(row: object) -> dict:
    required = {
        "gpu_index",
        "gpu_uuid",
        "memory_used_mib",
        "utilization_percent",
        "compute_processes",
    }
    if not isinstance(row, dict) or set(row) != required:
        raise ValueError("GPU telemetry schema mismatch")
    if (
        not _is_int(row["gpu_index"])
        or row["gpu_index"] < 0
        or not isinstance(row["gpu_uuid"], str)
        or not row["gpu_uuid"].startswith("GPU-")
        or not _is_int(row["memory_used_mib"])
        or row["memory_used_mib"] < 0
        or not _is_int(row["utilization_percent"])
        or row["utilization_percent"] < 0
        or not isinstance(row["compute_processes"], list)
    ):
        raise ValueError("GPU telemetry schema mismatch")
    return row


def select_strict_clean_gpus(
    inventory: list[dict],
) -> tuple[dict, ...]:
    if not isinstance(inventory, list):
        raise ValueError("GPU telemetry must be a list")
    validated = [_validate_gpu_row(row) for row in inventory]
    uuids = [row["gpu_uuid"] for row in validated]
    if len(uuids) != len(set(uuids)):
        raise ValueError("duplicate GPU UUID in inventory")
    indices = [row["gpu_index"] for row in validated]
    if len(indices) != len(set(indices)):
        raise ValueError("GPU telemetry contains duplicate index")
    clean = sorted(
        (
            row
            for row in validated
            if row["memory_used_mib"] <= MAX_GPU_MEMORY_USED_MIB
            and row["utilization_percent"]
            <= MAX_GPU_UTILIZATION_PERCENT
            and row["compute_processes"] == []
        ),
        key=lambda row: row["gpu_index"],
    )
    if len(clean) < 4:
        raise ValueError("four strict-clean GPUs are required")
    return tuple(dict(row) for row in clean[:4])


def parse_nvidia_smi_inventory(
    gpu_csv: str,
    process_csv: str,
) -> list[dict]:
    if not isinstance(gpu_csv, str) or not isinstance(process_csv, str):
        raise ValueError("GPU telemetry text is invalid")
    rows = []
    by_uuid = {}
    try:
        for fields in csv.reader(io.StringIO(gpu_csv)):
            if not fields or not any(value.strip() for value in fields):
                continue
            if len(fields) != 4:
                raise ValueError("GPU telemetry schema mismatch")
            index, uuid, memory_used, utilization = (
                value.strip() for value in fields
            )
            row = {
                "gpu_index": int(index),
                "gpu_uuid": uuid,
                "memory_used_mib": int(memory_used),
                "utilization_percent": int(utilization),
                "compute_processes": [],
            }
            _validate_gpu_row(row)
            if uuid in by_uuid:
                raise ValueError("duplicate GPU UUID in inventory")
            by_uuid[uuid] = row
            rows.append(row)
        for fields in csv.reader(io.StringIO(process_csv)):
            if not fields or not any(value.strip() for value in fields):
                continue
            if len(fields) == 1 and fields[0].strip() == (
                "No running processes found"
            ):
                continue
            if len(fields) != 4:
                raise ValueError("GPU process telemetry schema mismatch")
            uuid, pid, process_name, used_memory = (
                value.strip() for value in fields
            )
            if uuid not in by_uuid:
                raise ValueError(
                    "compute process references unknown GPU UUID"
                )
            process = {
                "pid": int(pid),
                "process_name": process_name,
                "used_memory_mib": int(used_memory),
            }
            if (
                process["pid"] <= 0
                or not process["process_name"]
                or process["used_memory_mib"] < 0
            ):
                raise ValueError("GPU process telemetry schema mismatch")
            by_uuid[uuid]["compute_processes"].append(process)
    except (TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            ("GPU ", "compute process")
        ):
            raise
        raise ValueError("GPU telemetry schema mismatch") from error
    rows.sort(key=lambda row: row["gpu_index"])
    _ = [_validate_gpu_row(row) for row in rows]
    return rows


def _validate_ssh_target(ssh_target: object) -> str:
    if (
        not isinstance(ssh_target, str)
        or re.fullmatch(
            (
                r"[A-Za-z0-9][A-Za-z0-9._-]*@"
                r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?"
            ),
            ssh_target,
        )
        is None
    ):
        raise ValueError("SSH target is invalid")
    return ssh_target


def build_ssh_argv(
    *,
    ssh_target: str,
    remote_argv: list[str],
    control_path: str | None = None,
) -> list[str]:
    ssh_target = _validate_ssh_target(ssh_target)
    if (
        not isinstance(remote_argv, list)
        or not remote_argv
        or any(
            not isinstance(argument, str)
            or not argument
            or "\0" in argument
            for argument in remote_argv
        )
    ):
        raise ValueError("remote argv is invalid")
    forbidden_commands = {"kinit", "krenew", "pkill", "killall"}
    if PurePosixPath(remote_argv[0]).name in forbidden_commands:
        raise ValueError("remote argv contains a forbidden command")
    argv = ["ssh"]
    if control_path is not None:
        if (
            not isinstance(control_path, str)
            or not control_path
            or not Path(control_path).is_absolute()
            or any(
                character in control_path
                for character in ("\0", "\n", "\r")
            )
        ):
            raise ValueError("SSH control path is invalid")
        argv.extend(["-S", control_path])
    argv.extend([
        "-o",
        "ControlMaster=no",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        ssh_target,
        "sh",
        "-c",
        shlex.join(remote_argv),
    ])
    return argv


def run_remote_argv(
    *,
    ssh_target: str,
    remote_argv: list[str],
    timeout_s: int,
    retry_count: int,
    control_path: str | None = None,
    command_runner: Callable[..., object] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
):
    timeout_s, retry_count = _validate_execution_policy(
        timeout_s,
        retry_count,
    )
    if not callable(command_runner) or not callable(sleep):
        raise ValueError("remote command dependency is invalid")
    argv = build_ssh_argv(
        ssh_target=ssh_target,
        remote_argv=remote_argv,
        control_path=control_path,
    )
    result = None
    for attempt in range(retry_count):
        result = command_runner(
            argv,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
        returncode = getattr(result, "returncode", None)
        if not _is_int(returncode):
            raise ValueError("remote command result is invalid")
        if returncode != 255 or attempt + 1 == retry_count:
            return result
        sleep(1.0)
    raise AssertionError("remote retry loop is unreachable")


def query_remote_gpu_inventory(
    *,
    ssh_target: str,
    timeout_s: int,
    retry_count: int,
    control_path: str | None = None,
    command_runner: Callable[..., object] = subprocess.run,
) -> list[dict]:
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
    result = run_remote_argv(
        ssh_target=ssh_target,
        remote_argv=["python3", "-c", script],
        control_path=control_path,
        timeout_s=timeout_s,
        retry_count=retry_count,
        command_runner=command_runner,
    )
    if result.returncode != 0:
        raise RuntimeError(
            getattr(result, "stderr", "")
            or "remote GPU inventory failed"
        )
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("remote GPU inventory JSON is invalid") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"gpu_csv", "process_csv"}
    ):
        raise ValueError("remote GPU inventory JSON is invalid")
    return parse_nvidia_smi_inventory(
        payload["gpu_csv"],
        payload["process_csv"],
    )


def _validate_gpu_topology(
    topology: object,
    selected_gpus: tuple[dict, ...],
) -> dict:
    if (
        not isinstance(topology, dict)
        or set(topology) != {"gpu_rows", "interconnect_matrix"}
        or not isinstance(topology["gpu_rows"], list)
        or not isinstance(topology["interconnect_matrix"], str)
        or not topology["interconnect_matrix"].strip()
    ):
        raise ValueError("GPU topology schema mismatch")
    selected = _validate_selected_gpus(selected_gpus)
    rows = []
    for row in topology["gpu_rows"]:
        if (
            not isinstance(row, dict)
            or set(row) != {
                "gpu_index",
                "gpu_uuid",
                "pci_bus_id",
            }
            or not _is_int(row["gpu_index"])
            or row["gpu_index"] < 0
            or not isinstance(row["gpu_uuid"], str)
            or not row["gpu_uuid"].startswith("GPU-")
            or not isinstance(row["pci_bus_id"], str)
            or re.fullmatch(
                r"[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:"
                r"[0-9A-Fa-f]{2}\.[0-7]",
                row["pci_bus_id"],
            )
            is None
        ):
            raise ValueError("GPU topology schema mismatch")
        rows.append(dict(row))
    if len(rows) != 4:
        raise ValueError("GPU topology must cover four selected GPUs")
    expected_identity = {
        (row["gpu_index"], row["gpu_uuid"]) for row in selected
    }
    observed_identity = {
        (row["gpu_index"], row["gpu_uuid"]) for row in rows
    }
    if (
        len(observed_identity) != 4
        or observed_identity != expected_identity
        or len({row["pci_bus_id"].lower() for row in rows}) != 4
        or any(
            f"GPU{row['gpu_index']}"
            not in topology["interconnect_matrix"]
            for row in rows
        )
    ):
        raise ValueError("GPU topology identity mismatch")
    rows.sort(key=lambda row: row["gpu_index"])
    return {
        "gpu_rows": rows,
        "interconnect_matrix": topology["interconnect_matrix"],
    }


def query_remote_gpu_topology(
    *,
    ssh_target: str,
    timeout_s: int,
    retry_count: int,
    control_path: str | None = None,
    command_runner: Callable[..., object] = subprocess.run,
) -> dict:
    script = "\n".join([
        "import json,subprocess",
        "gpu=subprocess.run([",
        "'nvidia-smi',",
        "'--query-gpu=index,uuid,pci.bus_id',",
        "'--format=csv,noheader,nounits',",
        "],check=True,text=True,capture_output=True)",
        "topology=subprocess.run([",
        "'nvidia-smi','topo','-m',",
        "],check=True,text=True,capture_output=True)",
        "print(json.dumps({",
        "'gpu_csv':gpu.stdout,",
        "'topology_matrix':topology.stdout,",
        "},sort_keys=True))",
    ])
    result = run_remote_argv(
        ssh_target=ssh_target,
        remote_argv=["python3", "-c", script],
        control_path=control_path,
        timeout_s=timeout_s,
        retry_count=retry_count,
        command_runner=command_runner,
    )
    if result.returncode != 0:
        raise RuntimeError(
            getattr(result, "stderr", "")
            or "remote GPU topology query failed"
        )
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("remote GPU topology JSON is invalid") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"gpu_csv", "topology_matrix"}
        or not isinstance(payload["gpu_csv"], str)
        or not isinstance(payload["topology_matrix"], str)
        or not payload["topology_matrix"].strip()
    ):
        raise ValueError("remote GPU topology JSON is invalid")
    rows = []
    try:
        for fields in csv.reader(io.StringIO(payload["gpu_csv"])):
            if not fields or not any(value.strip() for value in fields):
                continue
            if len(fields) != 3:
                raise ValueError
            gpu_index, gpu_uuid, pci_bus_id = (
                value.strip() for value in fields
            )
            rows.append({
                "gpu_index": int(gpu_index),
                "gpu_uuid": gpu_uuid,
                "pci_bus_id": pci_bus_id,
            })
    except (TypeError, ValueError) as error:
        raise ValueError("remote GPU topology JSON is invalid") from error
    return {
        "gpu_rows": rows,
        "interconnect_matrix": payload["topology_matrix"],
    }


def query_remote_path_state(
    *,
    ssh_target: str,
    remote_root: str,
    model_root: str,
    attempt_tag: str,
    timeout_s: int,
    retry_count: int,
    control_path: str | None = None,
    command_runner: Callable[..., object] = subprocess.run,
) -> dict:
    remote_root = _validate_remote_root(remote_root)
    model_root = _validate_absolute_remote_path(model_root)
    attempt_tag = _validate_attempt_tag(attempt_tag)
    attempt_root = _validate_absolute_remote_path(
        f"{remote_root}/attempts/{attempt_tag}"
    )
    script = "\n".join([
        "import json,os,sys",
        "remote_root,model_root,attempt_root=sys.argv[1:]",
        "print(json.dumps({",
        "'resolved_paths':{",
        "'remote_root':os.path.realpath(remote_root),",
        "'model_root':os.path.realpath(model_root),",
        "'attempt_root':os.path.realpath(attempt_root),",
        "},",
        "'attempt_exists':os.path.lexists(attempt_root),",
        "},sort_keys=True))",
    ])
    result = run_remote_argv(
        ssh_target=ssh_target,
        remote_argv=[
            "python3",
            "-c",
            script,
            remote_root,
            model_root,
            attempt_root,
        ],
        control_path=control_path,
        timeout_s=timeout_s,
        retry_count=retry_count,
        command_runner=command_runner,
    )
    if result.returncode != 0:
        raise RuntimeError(
            getattr(result, "stderr", "")
            or "remote path preflight failed"
        )
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("remote path preflight JSON is invalid") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"resolved_paths", "attempt_exists"}
        or not isinstance(payload["resolved_paths"], dict)
        or set(payload["resolved_paths"])
        != {"remote_root", "model_root", "attempt_root"}
        or any(
            not isinstance(value, str) or not value
            for value in payload["resolved_paths"].values()
        )
        or not isinstance(payload["attempt_exists"], bool)
    ):
        raise ValueError("remote path preflight JSON is invalid")
    return payload


def _blocked_kerberos(reason: str, minimum: int) -> dict:
    return {
        "classification": "BLOCKED_KERBEROS_TTL",
        "reason": reason,
        "minimum_required_lifetime_seconds": minimum,
    }


def classify_kerberos_ttl(
    payload: object,
    *,
    now: datetime,
    minimum_lifetime_seconds: int = (
        MINIMUM_KERBEROS_LIFETIME_SECONDS
    ),
) -> dict:
    if (
        not isinstance(payload, dict)
        or not isinstance(now, datetime)
        or now.utcoffset() is None
        or not _is_int(minimum_lifetime_seconds)
        or minimum_lifetime_seconds <= 0
    ):
        return _blocked_kerberos(
            "Kerberos payload is invalid",
            minimum_lifetime_seconds,
        )
    if payload.get("principal") != EXPECTED_KERBEROS_PRINCIPAL:
        return _blocked_kerberos(
            "Kerberos principal is unexpected",
            minimum_lifetime_seconds,
        )
    tickets = payload.get("tickets")
    if not isinstance(tickets, list):
        return _blocked_kerberos(
            "Kerberos ticket inventory is invalid",
            minimum_lifetime_seconds,
        )
    matching = [
        row
        for row in tickets
        if isinstance(row, dict)
        and row.get("Principal") == EXPECTED_KERBEROS_TGT
    ]
    if len(matching) != 1 or not isinstance(
        matching[0].get("Expires"),
        str,
    ):
        return _blocked_kerberos(
            "Kerberos TGT is missing or ambiguous",
            minimum_lifetime_seconds,
        )
    try:
        expires = datetime.strptime(
            matching[0]["Expires"],
            KERBEROS_TIMESTAMP_FORMAT,
        ).replace(tzinfo=now.tzinfo)
    except ValueError:
        return _blocked_kerberos(
            "Kerberos expiration is invalid",
            minimum_lifetime_seconds,
        )
    remaining = int((expires - now).total_seconds())
    if remaining < minimum_lifetime_seconds:
        result = _blocked_kerberos(
            "Kerberos TGT lifetime is insufficient",
            minimum_lifetime_seconds,
        )
        result["remaining_lifetime_seconds"] = remaining
        result["expires_at"] = expires.isoformat()
        return result
    return {
        "classification": "READY",
        "principal": EXPECTED_KERBEROS_PRINCIPAL,
        "tgt_principal": EXPECTED_KERBEROS_TGT,
        "expires_at": expires.isoformat(),
        "remaining_lifetime_seconds": remaining,
        "minimum_required_lifetime_seconds": (
            minimum_lifetime_seconds
        ),
    }


def query_local_kerberos(
    *,
    now: datetime | None = None,
    minimum_lifetime_seconds: int = (
        MINIMUM_KERBEROS_LIFETIME_SECONDS
    ),
    command_runner: Callable[..., object] = subprocess.run,
) -> dict:
    if not callable(command_runner):
        raise ValueError("Kerberos command runner is invalid")
    current = datetime.now().astimezone() if now is None else now
    try:
        result = command_runner(
            ["klist", "--json"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return _blocked_kerberos(
            "Kerberos cache is unavailable",
            minimum_lifetime_seconds,
        )
    if getattr(result, "returncode", None) != 0:
        return _blocked_kerberos(
            "Kerberos cache is unavailable",
            minimum_lifetime_seconds,
        )
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError):
        return _blocked_kerberos(
            "Kerberos payload is invalid",
            minimum_lifetime_seconds,
        )
    return classify_kerberos_ttl(
        payload,
        now=current,
        minimum_lifetime_seconds=minimum_lifetime_seconds,
    )


def _validate_owned_pid_set(owned_pids: object) -> frozenset[int]:
    if (
        not isinstance(owned_pids, (set, frozenset))
        or any(not _is_int(pid) or pid <= 0 for pid in owned_pids)
    ):
        raise ValueError("owned PID set is invalid")
    return frozenset(owned_pids)


def validate_selected_gpu_processes(
    *,
    selected: tuple[dict, ...],
    observed: list[dict],
    owned_pids: set[int] | frozenset[int],
) -> tuple[dict, ...]:
    if not isinstance(selected, tuple) or len(selected) != 4:
        raise ResourceIdentityError(
            "selected GPU inventory is invalid"
        )
    try:
        frozen = tuple(_validate_gpu_row(row) for row in selected)
        owned = _validate_owned_pid_set(owned_pids)
    except ValueError as error:
        raise ResourceIdentityError(str(error)) from error
    if not isinstance(observed, list):
        raise ResourceIdentityError("GPU telemetry must be a list")
    try:
        current_rows = [_validate_gpu_row(row) for row in observed]
    except ValueError as error:
        raise ResourceIdentityError(str(error)) from error
    by_uuid = {row["gpu_uuid"]: row for row in current_rows}
    if len(by_uuid) != len(current_rows):
        raise ResourceIdentityError(
            "duplicate GPU UUID in inventory"
        )
    result = []
    for frozen_row in frozen:
        current = by_uuid.get(frozen_row["gpu_uuid"])
        if (
            current is None
            or current["gpu_index"] != frozen_row["gpu_index"]
        ):
            raise ResourceIdentityError("selected GPU identity drift")
        for process in current["compute_processes"]:
            if (
                not isinstance(process, dict)
                or set(process)
                != {"pid", "process_name", "used_memory_mib"}
                or not _is_int(process.get("pid"))
                or process["pid"] <= 0
                or not isinstance(process.get("process_name"), str)
                or not process["process_name"]
                or not _is_int(process.get("used_memory_mib"))
                or process["used_memory_mib"] < 0
            ):
                raise ResourceIdentityError(
                    "GPU process telemetry schema mismatch"
                )
            if process["pid"] not in owned:
                raise ResourceIdentityError(
                    "unrelated GPU process detected"
                )
        result.append(dict(current))
    return tuple(result)


def _case_id(workload: str, phase: str, repetition: int) -> str:
    return f"{workload}__{phase}__r{repetition}"


def _overhead_pair_id(workload: str, repetition: int) -> str:
    return f"{workload}__overhead__r{repetition}"


def build_workload_cases() -> tuple[ProfileCase, ...]:
    cases = []
    for workload in WORKLOAD_ORDER:
        family, prompt_tokens, output_tokens, concurrency = (
            WORKLOADS[workload]
        )
        for repetition in range(2):
            cases.append(ProfileCase(
                case_id=_case_id(workload, "warmup", repetition),
                workload=workload,
                family=family,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                concurrency=concurrency,
                phase="warmup",
                repetition=repetition,
                profiled=False,
                overhead_pair_id=None,
            ))
        for phase, profiled in (
            ("measured", False),
            ("nsys_replay", True),
        ):
            for repetition in range(5):
                cases.append(ProfileCase(
                    case_id=_case_id(
                        workload,
                        phase,
                        repetition,
                    ),
                    workload=workload,
                    family=family,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    concurrency=concurrency,
                    phase=phase,
                    repetition=repetition,
                    profiled=profiled,
                    overhead_pair_id=_overhead_pair_id(
                        workload,
                        repetition,
                    ),
                ))
    return tuple(cases)


def _validate_structured_timings(
    timings: Mapping[str, Mapping[int, float]],
) -> dict[str, dict[int, float]]:
    if not isinstance(timings, Mapping) or set(timings) != set(
        WORKLOAD_ORDER
    ):
        raise ValueError(
            "five structured timings are required for every workload"
        )
    validated = {}
    for workload in WORKLOAD_ORDER:
        rows = timings[workload]
        if not isinstance(rows, Mapping) or set(rows) != set(range(5)):
            raise ValueError(
                "five structured timings are required for every workload"
            )
        current = {}
        for repetition, value in rows.items():
            if (
                isinstance(repetition, bool)
                or not isinstance(repetition, int)
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("structured timing is invalid")
            current[repetition] = float(value)
        validated[workload] = current
    return validated


def mark_representative_replays(
    cases: tuple[ProfileCase, ...],
    structured_decode_times_ms: Mapping[
        str,
        Mapping[int, float],
    ],
) -> tuple[ProfileCase, ...]:
    expected = build_workload_cases()
    if not isinstance(cases, tuple) or cases != expected:
        raise ValueError("workload case inventory is invalid")
    timings = _validate_structured_timings(
        structured_decode_times_ms
    )
    representatives = {}
    for workload in WORKLOAD_ORDER:
        rows = timings[workload]
        median = statistics.median(rows.values())
        representatives[workload] = min(
            rows,
            key=lambda repetition: (
                abs(rows[repetition] - median),
                repetition,
            ),
        )
    return tuple(
        replace(
            case,
            representative=(
                case.phase == "nsys_replay"
                and representatives[case.workload]
                == case.repetition
            ),
        )
        for case in cases
    )


def _path_is_below(path: PurePosixPath, root: PurePosixPath) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _is_valid_nsys_sqlite_path(
    raw_path: object,
    *,
    nsys_root: str,
) -> bool:
    if not isinstance(raw_path, str):
        return False
    path = PurePosixPath(raw_path)
    if ".." in path.parts or path.suffix != ".sqlite":
        return False
    try:
        _validate_absolute_remote_path(raw_path)
    except ValueError:
        return False
    return _path_is_below(path, PurePosixPath(nsys_root))


def _validate_absolute_remote_path(
    raw_path: object,
    *,
    allow_root: bool = False,
) -> str:
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or any(character in raw_path for character in ("\0", "\n", "\r"))
    ):
        raise ValueError("path is outside approved remote root")
    path = PurePosixPath(raw_path)
    approved = PurePosixPath(APPROVED_REMOTE_ROOT)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or (
            path != approved
            and not _path_is_below(path, approved)
        )
        or (path == approved and not allow_root)
        or "adaptive-ngram" in raw_path
    ):
        raise ValueError("path is outside approved remote root")
    return path.as_posix()


def is_path_below_approved_remote_root(path: object) -> bool:
    try:
        _validate_absolute_remote_path(path, allow_root=True)
    except ValueError:
        return False
    return True


def _validate_remote_root(remote_root: object) -> str:
    checked = _validate_absolute_remote_path(
        remote_root,
        allow_root=True,
    )
    if checked != APPROVED_REMOTE_ROOT:
        raise ValueError("remote root must equal approved remote root")
    return checked


def _validate_attempt_tag(attempt_tag: object) -> str:
    if (
        not isinstance(attempt_tag, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", attempt_tag)
        is None
    ):
        raise ValueError("attempt tag is invalid")
    return attempt_tag


def _validate_revision(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be an immutable revision")
    return value


def _validate_execution_policy(
    command_timeout_s: object,
    retry_count: object,
) -> tuple[int, int]:
    if (
        not _is_int(command_timeout_s)
        or not 1 <= command_timeout_s <= MAX_COMMAND_TIMEOUT_S
        or not _is_int(retry_count)
        or not 1 <= retry_count <= MAX_RETRY_COUNT
    ):
        raise ValueError("bounded execution policy is invalid")
    return command_timeout_s, retry_count


def _validate_selected_gpus(
    selected_gpus: object,
) -> tuple[dict, ...]:
    if not isinstance(selected_gpus, tuple) or len(selected_gpus) != 4:
        raise ValueError("four selected GPUs are required")
    validated = tuple(
        _validate_gpu_row(row) for row in selected_gpus
    )
    if len({row["gpu_uuid"] for row in validated}) != 4:
        raise ValueError("duplicate GPU UUID in selected inventory")
    if len({row["gpu_index"] for row in validated}) != 4:
        raise ValueError("duplicate GPU index in selected inventory")
    if any(
        row["memory_used_mib"] > MAX_GPU_MEMORY_USED_MIB
        or row["utilization_percent"]
        > MAX_GPU_UTILIZATION_PERCENT
        or row["compute_processes"]
        for row in validated
    ):
        raise ValueError("selected GPU inventory is not strict-clean")
    return validated


def _attempt_commands(
    *,
    attempt_root: str,
    temporary_root: str,
    artifact_root: str,
    nsys_root: str,
) -> list[dict]:
    return [
        {
            "name": "create-attempt-root",
            "argv": ["mkdir", "--", attempt_root],
        },
        {
            "name": "create-attempt-staging",
            "argv": ["mkdir", "--", temporary_root],
        },
        {
            "name": "create-artifact-root",
            "argv": ["mkdir", "--", artifact_root],
        },
        {
            "name": "create-nsys-root",
            "argv": ["mkdir", "--", nsys_root],
        },
    ]


def build_attempt_plan(
    *,
    ssh_target: str,
    remote_root: str,
    model_root: str,
    attempt_tag: str,
    source_revision: str,
    model_revision: str,
    selected_gpus: tuple[dict, ...],
    gpu_topology: dict,
    command_timeout_s: int = DEFAULT_COMMAND_TIMEOUT_S,
    retry_count: int = DEFAULT_RETRY_COUNT,
    resolve_remote_path: Callable[[str], str] | None = None,
    attempt_exists: Callable[[str], bool] | None = None,
) -> dict:
    remote_root = _validate_remote_root(remote_root)
    if (
        not isinstance(ssh_target, str)
        or re.fullmatch(
            (
                r"[A-Za-z0-9][A-Za-z0-9._-]*@"
                r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?"
            ),
            ssh_target,
        )
        is None
    ):
        raise ValueError("SSH target is invalid")
    attempt_tag = _validate_attempt_tag(attempt_tag)
    model_root = _validate_absolute_remote_path(model_root)
    source_revision = _validate_revision(
        source_revision,
        "source revision",
    )
    model_revision = _validate_revision(
        model_revision,
        "model revision",
    )
    command_timeout_s, retry_count = _validate_execution_policy(
        command_timeout_s,
        retry_count,
    )
    selected_gpus = _validate_selected_gpus(selected_gpus)
    gpu_topology = _validate_gpu_topology(
        gpu_topology,
        selected_gpus,
    )
    attempt_root = _validate_absolute_remote_path(
        f"{remote_root}/attempts/{attempt_tag}"
    )
    if attempt_exists is not None:
        if not callable(attempt_exists):
            raise ValueError("attempt existence check is invalid")
        if attempt_exists(attempt_root):
            raise ValueError("attempt directory already exists")
    if resolve_remote_path is not None:
        if not callable(resolve_remote_path):
            raise ValueError("remote path resolver is invalid")
        for path in (remote_root, model_root, attempt_root):
            resolved = resolve_remote_path(path)
            try:
                _validate_absolute_remote_path(
                    resolved,
                    allow_root=(path == remote_root),
                )
            except ValueError as error:
                raise ValueError("remote symlink escape detected") from error
    temporary_root = _validate_absolute_remote_path(
        f"{attempt_root}/.staging"
    )
    artifact_root = _validate_absolute_remote_path(
        f"{attempt_root}/artifacts"
    )
    nsys_root = _validate_absolute_remote_path(
        f"{attempt_root}/nsys"
    )
    cases = build_workload_cases()
    rank_mapping = [
        {
            "rank": rank,
            "gpu_index": row["gpu_index"],
            "gpu_uuid": row["gpu_uuid"],
        }
        for rank, row in enumerate(selected_gpus)
    ]
    workload_rows = [asdict(case) for case in cases]
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "ssh_target": ssh_target,
        "remote_root": remote_root,
        "model_root": model_root,
        "attempt_tag": attempt_tag,
        "attempt_root": attempt_root,
        "temporary_root": temporary_root,
        "artifact_root": artifact_root,
        "nsys_root": nsys_root,
        "source_revision": source_revision,
        "model_revision": model_revision,
        "command_timeout_s": command_timeout_s,
        "retry_count": retry_count,
        "gpu_rank_mapping": rank_mapping,
        "workload_cases": workload_rows,
        "manifests": {
            "source": {
                "revision": source_revision,
            },
            "model": {
                "root": model_root,
                "revision": model_revision,
            },
            "environment": {
                "dtype": "bfloat16",
                "tensor_parallel_size": 4,
                "decoding": "greedy",
                "temperature": 0.0,
                "fixed_output_tokens": 128,
                "scheduler_policy": "identical",
                "cuda_graph_policy": "identical",
            },
            "workloads": {
                "order": list(WORKLOAD_ORDER),
                "cases": workload_rows,
                "counts": {
                    "warmup": 10,
                    "measured": 25,
                    "nsys_replay": 25,
                    "overhead_pairs": 25,
                },
            },
            "topology": {
                "rank_mapping": rank_mapping,
                "gpu_rows": gpu_topology["gpu_rows"],
                "interconnect_matrix": (
                    gpu_topology["interconnect_matrix"]
                ),
                "strict_clean_limits": {
                    "maximum_memory_used_mib": (
                        MAX_GPU_MEMORY_USED_MIB
                    ),
                    "maximum_utilization_percent": (
                        MAX_GPU_UTILIZATION_PERCENT
                    ),
                    "compute_processes": [],
                },
            },
        },
        "commands": _attempt_commands(
            attempt_root=attempt_root,
            temporary_root=temporary_root,
            artifact_root=artifact_root,
            nsys_root=nsys_root,
        ),
        "benchmark_execution_authorized": False,
    }


def write_json_atomic(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def write_attempt_json_atomic(
    *,
    plan: dict,
    local_attempt_root: Path,
    relative_path: str,
    payload: object,
) -> Path:
    _validate_attempt_plan(plan)
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or any(
            character in relative_path
            for character in ("\0", "\n", "\r")
        )
    ):
        raise ValueError("relative artifact path is invalid")
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != relative_path
    ):
        raise ValueError("relative artifact path is invalid")
    root = Path(local_attempt_root).resolve()
    target = root.joinpath(*relative.parts)
    try:
        target.parent.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError("relative artifact path escapes attempt") from error
    write_json_atomic(target, payload)
    return target


def _plan_selected_gpus(plan: dict) -> tuple[dict, ...]:
    rows = plan.get("gpu_rank_mapping")
    if (
        not isinstance(rows, list)
        or len(rows) != 4
        or [row.get("rank") for row in rows] != list(range(4))
    ):
        raise ValueError("attempt plan GPU mapping is invalid")
    selected = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"rank", "gpu_index", "gpu_uuid"}
        ):
            raise ValueError("attempt plan GPU mapping is invalid")
        selected.append({
            "gpu_index": row["gpu_index"],
            "gpu_uuid": row["gpu_uuid"],
            "memory_used_mib": 0,
            "utilization_percent": 0,
            "compute_processes": [],
        })
    return _validate_selected_gpus(tuple(selected))


def _validate_attempt_plan(plan: object) -> dict:
    expected_keys = {
        "schema_version",
        "ssh_target",
        "remote_root",
        "model_root",
        "attempt_tag",
        "attempt_root",
        "temporary_root",
        "artifact_root",
        "nsys_root",
        "source_revision",
        "model_revision",
        "command_timeout_s",
        "retry_count",
        "gpu_rank_mapping",
        "workload_cases",
        "manifests",
        "commands",
        "benchmark_execution_authorized",
    }
    if (
        not isinstance(plan, dict)
        or set(plan) != expected_keys
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("remote_root") != APPROVED_REMOTE_ROOT
    ):
        raise ValueError("attempt plan is invalid")
    expected_attempt = (
        f"{APPROVED_REMOTE_ROOT}/attempts/"
        f"{_validate_attempt_tag(plan.get('attempt_tag'))}"
    )
    expected_paths = {
        "attempt_root": expected_attempt,
        "temporary_root": f"{expected_attempt}/.staging",
        "artifact_root": f"{expected_attempt}/artifacts",
        "nsys_root": f"{expected_attempt}/nsys",
    }
    if any(plan.get(key) != value for key, value in expected_paths.items()):
        raise ValueError("attempt plan path inventory is invalid")
    try:
        _validate_ssh_target(plan.get("ssh_target"))
    except ValueError as error:
        raise ValueError("attempt plan SSH target is invalid") from error
    for value in expected_paths.values():
        try:
            _validate_absolute_remote_path(value)
        except ValueError as error:
            raise ValueError("attempt plan path is invalid") from error
    _validate_absolute_remote_path(plan.get("model_root"))
    _validate_revision(plan.get("source_revision"), "source revision")
    _validate_revision(plan.get("model_revision"), "model revision")
    _validate_execution_policy(
        plan.get("command_timeout_s"),
        plan.get("retry_count"),
    )
    expected_cases = [asdict(case) for case in build_workload_cases()]
    if plan.get("workload_cases") != expected_cases:
        raise ValueError("attempt plan workload inventory is invalid")
    try:
        selected = _plan_selected_gpus(plan)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "attempt plan GPU mapping is invalid"
        ) from error
    rank_mapping = plan["gpu_rank_mapping"]
    manifests = plan.get("manifests")
    if (
        not isinstance(manifests, dict)
        or set(manifests)
        != {"source", "model", "environment", "workloads", "topology"}
        or not isinstance(manifests.get("topology"), dict)
    ):
        raise ValueError("attempt plan manifest inventory is invalid")
    topology_manifest = manifests["topology"]
    expected_manifests = {
        "source": {
            "revision": plan["source_revision"],
        },
        "model": {
            "root": plan["model_root"],
            "revision": plan["model_revision"],
        },
        "environment": {
            "dtype": "bfloat16",
            "tensor_parallel_size": 4,
            "decoding": "greedy",
            "temperature": 0.0,
            "fixed_output_tokens": 128,
            "scheduler_policy": "identical",
            "cuda_graph_policy": "identical",
        },
        "workloads": {
            "order": list(WORKLOAD_ORDER),
            "cases": expected_cases,
            "counts": {
                "warmup": 10,
                "measured": 25,
                "nsys_replay": 25,
                "overhead_pairs": 25,
            },
        },
        "topology": {
            "rank_mapping": rank_mapping,
            "gpu_rows": topology_manifest.get("gpu_rows"),
            "interconnect_matrix": topology_manifest.get(
                "interconnect_matrix"
            ),
            "strict_clean_limits": {
                "maximum_memory_used_mib": (
                    MAX_GPU_MEMORY_USED_MIB
                ),
                "maximum_utilization_percent": (
                    MAX_GPU_UTILIZATION_PERCENT
                ),
                "compute_processes": [],
            },
        },
    }
    if manifests != expected_manifests:
        raise ValueError("attempt plan manifest inventory is invalid")
    try:
        _validate_gpu_topology(
            {
                "gpu_rows": expected_manifests["topology"][
                    "gpu_rows"
                ],
                "interconnect_matrix": expected_manifests[
                    "topology"
                ]["interconnect_matrix"],
            },
            selected,
        )
    except ValueError as error:
        raise ValueError(
            "attempt plan topology manifest is invalid"
        ) from error
    if len(selected) != 4:
        raise ValueError("attempt plan GPU mapping is invalid")
    expected_commands = _attempt_commands(
        attempt_root=expected_paths["attempt_root"],
        temporary_root=expected_paths["temporary_root"],
        artifact_root=expected_paths["artifact_root"],
        nsys_root=expected_paths["nsys_root"],
    )
    commands = plan.get("commands")
    if commands != expected_commands:
        raise ValueError("attempt plan command inventory is invalid")
    for command in commands:
        if (
            not isinstance(command, dict)
            or set(command) != {"name", "argv"}
            or not isinstance(command["name"], str)
            or not command["name"]
            or not isinstance(command["argv"], list)
            or not command["argv"]
            or any(
                not isinstance(argument, str) or not argument
                for argument in command["argv"]
            )
        ):
            raise ValueError("attempt plan command inventory is invalid")
        serialized = json.dumps(command, sort_keys=True)
        if any(
            forbidden in serialized
            for forbidden in (
                "kinit",
                "krenew",
                "pkill",
                "killall",
                "adaptive-ngram",
                "/private/tmp",
            )
        ):
            raise ValueError("attempt plan contains forbidden command")
    if plan.get("benchmark_execution_authorized") is not False:
        raise ValueError("attempt plan authorization is invalid")
    return plan


def _guard_planned_gpus(
    plan: dict,
    inventory: list[dict],
) -> tuple[dict, ...]:
    expected = _plan_selected_gpus(plan)
    try:
        if not isinstance(inventory, list):
            raise ValueError("GPU telemetry must be a list")
        observed = [_validate_gpu_row(row) for row in inventory]
        uuids = [row["gpu_uuid"] for row in observed]
        indices = [row["gpu_index"] for row in observed]
        if len(uuids) != len(set(uuids)):
            raise ValueError("duplicate GPU UUID in inventory")
        if len(indices) != len(set(indices)):
            raise ValueError("GPU telemetry contains duplicate index")
        by_uuid = {row["gpu_uuid"]: row for row in observed}
        selected = []
        for planned in expected:
            current = by_uuid.get(planned["gpu_uuid"])
            if (
                current is None
                or current["gpu_index"] != planned["gpu_index"]
            ):
                raise ValueError("selected GPU identity drift")
            if (
                current["memory_used_mib"]
                > MAX_GPU_MEMORY_USED_MIB
                or current["utilization_percent"]
                > MAX_GPU_UTILIZATION_PERCENT
                or current["compute_processes"]
            ):
                raise ValueError(
                    "planned GPU inventory is not strict-clean"
                )
            selected.append(dict(current))
        return tuple(selected)
    except ValueError as error:
        raise ResourceIdentityError(str(error)) from error


def wait_for_strict_clean_gpus(
    *,
    query_inventory: Callable[[], list[dict]],
    timeout_s: int,
    poll_interval_s: int,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict:
    if (
        not callable(query_inventory)
        or not callable(sleep)
        or not callable(monotonic)
        or not _is_int(timeout_s)
        or not 0 <= timeout_s <= MAX_GPU_MONITOR_INTERVAL_S
        or not _is_int(poll_interval_s)
        or not 1
        <= poll_interval_s
        <= MAX_GPU_MONITOR_INTERVAL_S
    ):
        raise ValueError("GPU monitor policy is invalid")
    started = monotonic()
    samples = []
    while True:
        try:
            selected = select_strict_clean_gpus(query_inventory())
        except Exception as error:
            samples.append({
                "classification": "BLOCKED_RESOURCES",
                "reason": f"{type(error).__name__}: {error}",
            })
        else:
            samples.append({
                "classification": "READY",
                "gpu_uuids": [
                    row["gpu_uuid"] for row in selected
                ],
            })
            return {
                "classification": "READY",
                "selected_gpus": list(selected),
                "samples": samples,
            }
        elapsed = monotonic() - started
        if elapsed + poll_interval_s > timeout_s:
            return {
                "classification": "BLOCKED_RESOURCES",
                "selected_gpus": [],
                "samples": samples,
                "reason": "strict-clean GPU monitor timed out",
            }
        sleep(poll_interval_s)


def _owned_pids_from_result(result: object) -> frozenset[int]:
    if not isinstance(result, dict):
        raise ValueError("worker result is invalid")
    owned_pids = result.get("owned_pids")
    if (
        not isinstance(owned_pids, list)
        or any(not _is_int(pid) or pid <= 0 for pid in owned_pids)
        or len(owned_pids) != len(set(owned_pids))
    ):
        raise ValueError("worker owned PID inventory is invalid")
    return frozenset(owned_pids)


def _cleanup_and_validate(
    cleanup_owned: Callable[..., object],
    owned_pids: frozenset[int],
) -> dict:
    receipt = cleanup_owned(
        owned_pids=owned_pids,
        normal_control_only=True,
    )
    if not isinstance(receipt, dict):
        raise ValueError("cleanup receipt is invalid")
    signaled = receipt.get("signaled_pids")
    if (
        not isinstance(signaled, list)
        or any(not _is_int(pid) or pid <= 0 for pid in signaled)
    ):
        raise ValueError("cleanup receipt is invalid")
    if not set(signaled).issubset(owned_pids):
        raise CleanupSafetyError("cleanup signaled an unowned PID")
    return receipt


def _safe_cleanup(
    cleanup_owned: Callable[..., object],
    owned_pids: frozenset[int],
) -> dict:
    try:
        return _cleanup_and_validate(cleanup_owned, owned_pids)
    except CleanupSafetyError:
        raise
    except Exception as error:
        return {
            "classification": "FAILED",
            "reason": f"{type(error).__name__}: {error}",
            "signaled_pids": [],
            "owned_children_remaining": sorted(owned_pids),
        }


def _failure_result(
    classification: str,
    *,
    plan: dict,
    correctness: dict | None,
    structured_results: list[dict],
    nsys_results: list[dict],
    owned_pids: frozenset[int],
    cleanup_owned: Callable[..., object],
    reason: str | None = None,
) -> dict:
    cleanup = _safe_cleanup(cleanup_owned, owned_pids)
    cleanup_failed = (
        cleanup.get("classification") != "CLEAN"
        or cleanup.get("owned_children_remaining", []) != []
    )
    result = {
        "classification": (
            "FAILED_CLEANUP" if cleanup_failed else classification
        ),
        "plan": plan,
        "correctness": correctness,
        "structured_results": structured_results,
        "nsys_results": nsys_results,
        "cleanup": cleanup,
        "preserve_attempt": True,
        "benchmark_execution_authorized": False,
    }
    if cleanup_failed:
        result["prior_classification"] = classification
    if reason is not None:
        result["reason"] = reason
    return result


def _validate_worker_result(
    result: object,
    *,
    case: ProfileCase | None,
    selected_gpus: tuple[dict, ...],
) -> tuple[dict, frozenset[int]]:
    owned_pids = _owned_pids_from_result(result)
    assert isinstance(result, dict)
    if not isinstance(result.get("classification"), str):
        raise ValueError("worker result classification is invalid")
    if case is not None and result.get("case_id") != case.case_id:
        raise ValueError("worker case identity drift")
    resource_samples = result.get("resource_samples")
    if (
        not isinstance(resource_samples, list)
        or not resource_samples
    ):
        raise ValueError("worker resource samples are invalid")
    for sample in resource_samples:
        validate_selected_gpu_processes(
            selected=selected_gpus,
            observed=sample,
            owned_pids=owned_pids,
        )
    return dict(result), owned_pids


def _worker_cleanup_complete(result: dict) -> bool:
    remaining = result.get("owned_children_remaining")
    return (
        result.get("process_group_destroyed") is True
        and isinstance(remaining, list)
        and remaining == []
    )


def run_attempt(
    *,
    plan: dict,
    plan_only: bool,
    dry_run: bool = False,
    kerberos_status: dict | None = None,
    query_inventory: Callable[[], list[dict]],
    run_correctness: Callable[..., object],
    run_case: Callable[..., object],
    cleanup_owned: Callable[..., object],
) -> dict:
    plan = _validate_attempt_plan(plan)
    if not isinstance(plan_only, bool) or not isinstance(dry_run, bool):
        raise ValueError("attempt mode is invalid")
    if plan_only:
        return {
            "classification": "PLAN_ONLY",
            "plan": plan,
            "benchmark_execution_authorized": False,
        }
    if (
        not isinstance(kerberos_status, dict)
        or kerberos_status.get("classification") != "READY"
    ):
        return {
            "classification": "BLOCKED_KERBEROS_TTL",
            "plan": plan,
            "kerberos": kerberos_status,
            "preserve_attempt": True,
            "benchmark_execution_authorized": False,
        }
    for callback in (
        query_inventory,
        run_correctness,
        run_case,
        cleanup_owned,
    ):
        if not callable(callback):
            raise ValueError("attempt callback is invalid")
    if dry_run:
        try:
            selected = _guard_planned_gpus(
                plan,
                query_inventory(),
            )
        except ValueError as error:
            return {
                "classification": "BLOCKED_RESOURCES",
                "plan": plan,
                "reason": str(error),
                "preserve_attempt": True,
                "benchmark_execution_authorized": False,
            }
        return {
            "classification": "DRY_RUN_READY",
            "plan": plan,
            "selected_gpus": list(selected),
            "preserve_attempt": True,
            "benchmark_execution_authorized": False,
        }
    planned_cases = build_workload_cases()
    try:
        selected = _guard_planned_gpus(plan, query_inventory())
    except Exception as error:
        return {
            "classification": "BLOCKED_RESOURCES",
            "plan": plan,
            "reason": f"{type(error).__name__}: {error}",
            "preserve_attempt": True,
            "benchmark_execution_authorized": False,
        }
    owned_pids: frozenset[int] = frozenset()
    structured_results: list[dict] = []
    nsys_results: list[dict] = []
    correctness = None

    try:
        _guard_planned_gpus(plan, query_inventory())
        raw_correctness = run_correctness(
            plan=plan,
            selected_gpus=selected,
        )
        current_owned = _owned_pids_from_result(raw_correctness)
        owned_pids = owned_pids.union(current_owned)
        correctness, _ = _validate_worker_result(
            raw_correctness,
            case=None,
            selected_gpus=selected,
        )
        _guard_planned_gpus(plan, query_inventory())
    except ResourceIdentityError as error:
        return _failure_result(
            "INVALID_RESOURCE_IDENTITY",
            plan=plan,
            correctness=correctness,
            structured_results=structured_results,
            nsys_results=nsys_results,
            owned_pids=owned_pids,
            cleanup_owned=cleanup_owned,
            reason=str(error),
        )
    except Exception as error:
        return _failure_result(
            "FAILED_EXECUTION",
            plan=plan,
            correctness=correctness,
            structured_results=structured_results,
            nsys_results=nsys_results,
            owned_pids=owned_pids,
            cleanup_owned=cleanup_owned,
            reason=f"{type(error).__name__}: {error}",
        )
    if not _worker_cleanup_complete(correctness):
        return _failure_result(
            "FAILED_WORKER_CLEANUP",
            plan=plan,
            correctness=correctness,
            structured_results=structured_results,
            nsys_results=nsys_results,
            owned_pids=owned_pids,
            cleanup_owned=cleanup_owned,
        )
    if correctness["classification"] != "PASS":
        return _failure_result(
            correctness["classification"],
            plan=plan,
            correctness=correctness,
            structured_results=structured_results,
            nsys_results=nsys_results,
            owned_pids=owned_pids,
            cleanup_owned=cleanup_owned,
        )

    structured_cases = [
        case
        for case in planned_cases
        if case.phase in {"warmup", "measured"}
    ]
    timings = {workload: {} for workload in WORKLOAD_ORDER}
    measured_by_pair = {}
    for case in structured_cases:
        try:
            _guard_planned_gpus(plan, query_inventory())
            raw_result = run_case(
                plan=plan,
                case=case,
                selected_gpus=selected,
            )
            current_owned = _owned_pids_from_result(raw_result)
            owned_pids = owned_pids.union(current_owned)
            result, _ = _validate_worker_result(
                raw_result,
                case=case,
                selected_gpus=selected,
            )
            _guard_planned_gpus(plan, query_inventory())
        except ResourceIdentityError as error:
            return _failure_result(
                "INVALID_RESOURCE_IDENTITY",
                plan=plan,
                correctness=correctness,
                structured_results=structured_results,
                nsys_results=nsys_results,
                owned_pids=owned_pids,
                cleanup_owned=cleanup_owned,
                reason=str(error),
            )
        except Exception as error:
            return _failure_result(
                "FAILED_EXECUTION",
                plan=plan,
                correctness=correctness,
                structured_results=structured_results,
                nsys_results=nsys_results,
                owned_pids=owned_pids,
                cleanup_owned=cleanup_owned,
                reason=f"{type(error).__name__}: {error}",
            )
        if not _worker_cleanup_complete(result):
            return _failure_result(
                "FAILED_WORKER_CLEANUP",
                plan=plan,
                correctness=correctness,
                structured_results=structured_results,
                nsys_results=nsys_results,
                owned_pids=owned_pids,
                cleanup_owned=cleanup_owned,
            )
        if result["classification"] != "PASS":
            return _failure_result(
                result["classification"],
                plan=plan,
                correctness=correctness,
                structured_results=structured_results,
                nsys_results=nsys_results,
                owned_pids=owned_pids,
                cleanup_owned=cleanup_owned,
            )
        if case.phase == "measured":
            decode_time = result.get("decode_time_ms")
            if (
                isinstance(decode_time, bool)
                or not isinstance(decode_time, (int, float))
                or not math.isfinite(decode_time)
                or decode_time <= 0
            ):
                return _failure_result(
                    "INVALID_STRUCTURED_TIMING",
                    plan=plan,
                    correctness=correctness,
                    structured_results=structured_results,
                    nsys_results=nsys_results,
                    owned_pids=owned_pids,
                    cleanup_owned=cleanup_owned,
                )
            timings[case.workload][case.repetition] = float(
                decode_time
            )
            measured_by_pair[case.overhead_pair_id] = result
        structured_results.append(result)

    marked = mark_representative_replays(planned_cases, timings)
    replay_cases = [
        case for case in marked if case.phase == "nsys_replay"
    ]
    seen_sqlite_paths = set()
    for case in replay_cases:
        try:
            _guard_planned_gpus(plan, query_inventory())
            raw_result = run_case(
                plan=plan,
                case=case,
                selected_gpus=selected,
            )
            current_owned = _owned_pids_from_result(raw_result)
            owned_pids = owned_pids.union(current_owned)
            result, _ = _validate_worker_result(
                raw_result,
                case=case,
                selected_gpus=selected,
            )
            _guard_planned_gpus(plan, query_inventory())
        except ResourceIdentityError as error:
            return _failure_result(
                "INVALID_RESOURCE_IDENTITY",
                plan=plan,
                correctness=correctness,
                structured_results=structured_results,
                nsys_results=nsys_results,
                owned_pids=owned_pids,
                cleanup_owned=cleanup_owned,
                reason=str(error),
            )
        except Exception as error:
            return _failure_result(
                "FAILED_EXECUTION",
                plan=plan,
                correctness=correctness,
                structured_results=structured_results,
                nsys_results=nsys_results,
                owned_pids=owned_pids,
                cleanup_owned=cleanup_owned,
                reason=f"{type(error).__name__}: {error}",
            )
        if not _worker_cleanup_complete(result):
            return _failure_result(
                "FAILED_WORKER_CLEANUP",
                plan=plan,
                correctness=correctness,
                structured_results=structured_results,
                nsys_results=nsys_results,
                owned_pids=owned_pids,
                cleanup_owned=cleanup_owned,
            )
        if result["classification"] != "PASS":
            return _failure_result(
                result["classification"],
                plan=plan,
                correctness=correctness,
                structured_results=structured_results,
                nsys_results=nsys_results,
                owned_pids=owned_pids,
                cleanup_owned=cleanup_owned,
            )
        sqlite_path = result.get("sqlite_path")
        if (
            not _is_valid_nsys_sqlite_path(
                sqlite_path,
                nsys_root=plan["nsys_root"],
            )
            or sqlite_path in seen_sqlite_paths
        ):
            return _failure_result(
                "INCONCLUSIVE_TRACE_COVERAGE",
                plan=plan,
                correctness=correctness,
                structured_results=structured_results,
                nsys_results=nsys_results,
                owned_pids=owned_pids,
                cleanup_owned=cleanup_owned,
                reason="Nsight SQLite export is missing or invalid",
            )
        decode_time = result.get("decode_time_ms")
        if (
            isinstance(decode_time, bool)
            or not isinstance(decode_time, (int, float))
            or not math.isfinite(decode_time)
            or decode_time <= 0
        ):
            return _failure_result(
                "INCONCLUSIVE_TRACE_COVERAGE",
                plan=plan,
                correctness=correctness,
                structured_results=structured_results,
                nsys_results=nsys_results,
                owned_pids=owned_pids,
                cleanup_owned=cleanup_owned,
                reason="profiled timing is missing or invalid",
            )
        seen_sqlite_paths.add(sqlite_path)
        result["representative"] = case.representative
        result["overhead_pair_id"] = case.overhead_pair_id
        nsys_results.append(result)

    overhead_controls = []
    for result in nsys_results:
        pair_id = result["overhead_pair_id"]
        unprofiled = measured_by_pair[pair_id]["decode_time_ms"]
        profiled = result["decode_time_ms"]
        overhead_controls.append({
            "overhead_pair_id": pair_id,
            "unprofiled_decode_time_ms": unprofiled,
            "profiled_decode_time_ms": profiled,
            "relative_overhead": profiled / unprofiled - 1.0,
        })
    cleanup = _safe_cleanup(cleanup_owned, owned_pids)
    cleanup_complete = (
        cleanup.get("classification") == "CLEAN"
        and cleanup.get("owned_children_remaining", []) == []
    )
    classification = "COMPLETE" if cleanup_complete else "FAILED_CLEANUP"
    return {
        "classification": classification,
        "plan": plan,
        "correctness": correctness,
        "structured_results": structured_results,
        "nsys_results": nsys_results,
        "overhead_controls": overhead_controls,
        "owned_pids": sorted(owned_pids),
        "cleanup": cleanup,
        "preserve_attempt": True,
        "benchmark_execution_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or run the strict-clean Qwen3.8-27B TP4 "
            "communication profiling campaign."
        )
    )
    parser.add_argument(
        "--ssh-target",
        default=DEFAULT_SSH_TARGET,
    )
    parser.add_argument(
        "--remote-root",
        default=APPROVED_REMOTE_ROOT,
    )
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--attempt-tag", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--control-path")
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
    )
    return parser


def main(
    argv=None,
    *,
    inventory_query: Callable[..., list[dict]] = (
        query_remote_gpu_inventory
    ),
    topology_query: Callable[..., dict] = query_remote_gpu_topology,
    path_state_query: Callable[..., dict] = query_remote_path_state,
    kerberos_query: Callable[..., dict] = query_local_kerberos,
    gpu_monitor: Callable[..., dict] = wait_for_strict_clean_gpus,
    run_correctness: Callable[..., object] | None = None,
    run_case: Callable[..., object] | None = None,
    cleanup_owned: Callable[..., object] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not all(
        callable(callback)
        for callback in (
            inventory_query,
            topology_query,
            path_state_query,
            kerberos_query,
            gpu_monitor,
        )
    ):
        raise ValueError("CLI query dependency is invalid")

    def query_inventory() -> list[dict]:
        return inventory_query(
            ssh_target=args.ssh_target,
            control_path=args.control_path,
            timeout_s=args.command_timeout_s,
            retry_count=args.retry_count,
        )

    kerberos_status = None
    if not args.plan_only:
        kerberos_status = kerberos_query()
        if (
            not isinstance(kerberos_status, dict)
            or kerberos_status.get("classification") != "READY"
        ):
            result = {
                "classification": "BLOCKED_KERBEROS_TTL",
                "kerberos": kerberos_status,
                "preserve_attempt": True,
                "benchmark_execution_authorized": False,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2

    path_state = path_state_query(
        ssh_target=args.ssh_target,
        remote_root=args.remote_root,
        model_root=args.model_root,
        attempt_tag=args.attempt_tag,
        control_path=args.control_path,
        timeout_s=args.command_timeout_s,
        retry_count=args.retry_count,
    )
    if (
        not isinstance(path_state, dict)
        or not isinstance(path_state.get("resolved_paths"), dict)
        or not isinstance(path_state.get("attempt_exists"), bool)
    ):
        raise ValueError("remote path preflight result is invalid")
    resolved_paths = path_state["resolved_paths"]
    try:
        _validate_absolute_remote_path(
            resolved_paths["remote_root"],
            allow_root=True,
        )
        _validate_absolute_remote_path(resolved_paths["model_root"])
        _validate_absolute_remote_path(resolved_paths["attempt_root"])
    except (KeyError, ValueError) as error:
        raise ValueError("remote symlink escape detected") from error
    if path_state["attempt_exists"]:
        raise ValueError("attempt directory already exists")

    if args.plan_only:
        selected = select_strict_clean_gpus(query_inventory())
    else:
        monitor = gpu_monitor(
            query_inventory=query_inventory,
            timeout_s=args.gpu_wait_timeout_s,
            poll_interval_s=args.gpu_poll_interval_s,
        )
        if (
            not isinstance(monitor, dict)
            or monitor.get("classification") != "READY"
        ):
            result = {
                "classification": "BLOCKED_RESOURCES",
                "monitor": monitor,
                "preserve_attempt": True,
                "benchmark_execution_authorized": False,
            }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2
        selected_rows = monitor.get("selected_gpus")
        if not isinstance(selected_rows, list):
            raise ValueError("GPU monitor result is invalid")
        selected = _validate_selected_gpus(tuple(selected_rows))

    raw_topology = topology_query(
        ssh_target=args.ssh_target,
        control_path=args.control_path,
        timeout_s=args.command_timeout_s,
        retry_count=args.retry_count,
    )
    if not isinstance(raw_topology, dict):
        raise ValueError("GPU topology schema mismatch")
    selected_identities = {
        (row["gpu_index"], row["gpu_uuid"]) for row in selected
    }
    gpu_topology = _validate_gpu_topology(
        {
            "gpu_rows": [
                row
                for row in raw_topology.get("gpu_rows", [])
                if isinstance(row, dict)
                and (row.get("gpu_index"), row.get("gpu_uuid"))
                in selected_identities
            ],
            "interconnect_matrix": raw_topology.get(
                "interconnect_matrix"
            ),
        },
        selected,
    )

    attempt_root = (
        f"{args.remote_root}/attempts/{args.attempt_tag}"
    )
    path_resolution = {
        args.remote_root: resolved_paths["remote_root"],
        args.model_root: resolved_paths["model_root"],
        attempt_root: resolved_paths["attempt_root"],
    }
    plan = build_attempt_plan(
        ssh_target=args.ssh_target,
        remote_root=args.remote_root,
        model_root=args.model_root,
        attempt_tag=args.attempt_tag,
        source_revision=args.source_revision,
        model_revision=args.model_revision,
        selected_gpus=selected,
        gpu_topology=gpu_topology,
        command_timeout_s=args.command_timeout_s,
        retry_count=args.retry_count,
        resolve_remote_path=lambda path: path_resolution[path],
        attempt_exists=lambda path: path_state["attempt_exists"],
    )

    def unavailable(**kwargs):
        del kwargs
        raise RuntimeError(
            "production worker adapter is not configured"
        )

    if not args.plan_only and not args.dry_run and not all(
        callable(callback)
        for callback in (run_correctness, run_case, cleanup_owned)
    ):
        result = {
            "classification": "EXECUTION_ADAPTER_UNAVAILABLE",
            "plan": plan,
            "preserve_attempt": True,
            "benchmark_execution_authorized": False,
        }
    else:
        result = run_attempt(
            plan=plan,
            plan_only=args.plan_only,
            dry_run=args.dry_run,
            kerberos_status=kerberos_status,
            query_inventory=query_inventory,
            run_correctness=run_correctness or unavailable,
            run_case=run_case or unavailable,
            cleanup_owned=cleanup_owned or unavailable,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["classification"] in {
        "PLAN_ONLY",
        "DRY_RUN_READY",
        "COMPLETE",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
