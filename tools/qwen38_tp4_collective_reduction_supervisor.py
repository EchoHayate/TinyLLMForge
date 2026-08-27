#!/usr/bin/env python3
"""Run and observe one TP4 collective-reduction worker without signals."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Callable


SUPERVISOR_SCHEMA = (
    "qwen38.tp4-collective-reduction-supervisor.v1"
)
CLEANUP_SCHEMA = "qwen38.tp4-collective-reduction-cleanup.v1"
WORKER_SCRIPT = "qwen38_tp4_collective_reduction_worker.py"


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _validate_selected_gpus(selected_gpus):
    if not isinstance(selected_gpus, list) or len(selected_gpus) != 4:
        raise ValueError("selected GPU inventory is invalid")
    identities = []
    for row in selected_gpus:
        if (
            not isinstance(row, dict)
            or not _is_int(row.get("gpu_index"))
            or row["gpu_index"] < 0
            or not isinstance(row.get("gpu_uuid"), str)
            or not row["gpu_uuid"].startswith("GPU-")
        ):
            raise ValueError("selected GPU inventory is invalid")
        identities.append((row["gpu_index"], row["gpu_uuid"]))
    if len(set(identities)) != 4:
        raise ValueError("selected GPU inventory is invalid")
    return [dict(row) for row in selected_gpus]


def build_runtime_sample(
    *,
    case_id,
    selected_gpus,
    observed_gpus,
    owned_pids,
    captured_at_unix_ns,
):
    selected = _validate_selected_gpus(selected_gpus)
    if (
        not isinstance(case_id, str)
        or not case_id
        or not isinstance(owned_pids, (set, frozenset))
        or any(not _is_int(pid) or pid <= 0 for pid in owned_pids)
        or not _is_int(captured_at_unix_ns)
        or captured_at_unix_ns <= 0
        or not isinstance(observed_gpus, list)
    ):
        raise ValueError("runtime resource sample is invalid")
    by_uuid = {}
    for row in observed_gpus:
        if (
            not isinstance(row, dict)
            or not _is_int(row.get("gpu_index"))
            or not isinstance(row.get("gpu_uuid"), str)
            or not _is_int(row.get("memory_used_mib"))
            or row["memory_used_mib"] < 0
            or not _is_int(row.get("utilization_percent"))
            or not 0 <= row["utilization_percent"] <= 100
            or not isinstance(row.get("compute_processes"), list)
        ):
            raise ValueError("runtime GPU telemetry is invalid")
        if row["gpu_uuid"] in by_uuid:
            raise ValueError("runtime GPU telemetry is invalid")
        by_uuid[row["gpu_uuid"]] = row
    normalized = []
    for frozen in selected:
        current = by_uuid.get(frozen["gpu_uuid"])
        if (
            current is None
            or current["gpu_index"] != frozen["gpu_index"]
        ):
            raise ValueError("selected GPU identity drift")
        for process in current["compute_processes"]:
            if (
                not isinstance(process, dict)
                or not _is_int(process.get("pid"))
                or process["pid"] <= 0
                or not isinstance(process.get("process_name"), str)
                or not process["process_name"]
                or not _is_int(process.get("used_memory_mib"))
                or process["used_memory_mib"] < 0
            ):
                raise ValueError("runtime GPU process telemetry is invalid")
            if process["pid"] not in owned_pids:
                raise ValueError("foreign GPU process detected")
        normalized.append(dict(current))
    return {
        "case_id": case_id,
        "captured_at_unix_ns": captured_at_unix_ns,
        "owned_pids": sorted(owned_pids),
        "selected_gpus": normalized,
    }


def build_worker_environment(
    *,
    attempt_root,
    selected_gpus,
    base_environment,
    dist_port,
):
    attempt_root = Path(attempt_root).resolve()
    selected = _validate_selected_gpus(selected_gpus)
    if (
        not attempt_root.is_absolute()
        or not isinstance(base_environment, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in base_environment.items()
        )
        or not _is_int(dist_port)
        or not 1024 <= dist_port <= 65535
    ):
        raise ValueError("worker environment is invalid")
    runtime = attempt_root / "runtime"
    paths = {
        "TMPDIR": runtime / "tmp",
        "XDG_CACHE_HOME": runtime / "cache" / "xdg",
        "HF_HOME": runtime / "cache" / "huggingface",
        "TORCH_HOME": runtime / "cache" / "torch",
        "TORCH_EXTENSIONS_DIR": runtime / "cache" / "torch-extensions",
        "CUDA_CACHE_PATH": runtime / "cache" / "cuda",
        "TRITON_CACHE_DIR": runtime / "cache" / "triton",
    }
    existing_pythonpath = base_environment.get("PYTHONPATH")
    pythonpath = str(attempt_root / "source")
    if existing_pythonpath:
        pythonpath = pythonpath + os.pathsep + existing_pythonpath
    environment = dict(base_environment)
    environment.update({
        "CUDA_VISIBLE_DEVICES": ",".join(
            str(row["gpu_index"]) for row in selected
        ),
        "TINYVLLM_DIST_PORT": str(dist_port),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": pythonpath,
        **{name: str(path) for name, path in paths.items()},
    })
    return environment


def query_gpu_inventory():
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
    by_uuid = {}
    rows = []
    for fields in csv.reader(io.StringIO(gpu.stdout)):
        if not fields or not any(value.strip() for value in fields):
            continue
        index, uuid, memory, utilization = (
            value.strip() for value in fields
        )
        row = {
            "gpu_index": int(index),
            "gpu_uuid": uuid,
            "memory_used_mib": int(memory),
            "utilization_percent": int(utilization),
            "compute_processes": [],
        }
        rows.append(row)
        by_uuid[uuid] = row
    for fields in csv.reader(io.StringIO(processes.stdout)):
        if not fields or not any(value.strip() for value in fields):
            continue
        if len(fields) == 1 and fields[0].strip() == (
            "No running processes found"
        ):
            continue
        uuid, pid, process_name, used_memory = (
            value.strip() for value in fields
        )
        if uuid not in by_uuid:
            raise ValueError("GPU process references unknown GPU")
        by_uuid[uuid]["compute_processes"].append({
            "pid": int(pid),
            "process_name": process_name,
            "used_memory_mib": int(used_memory),
        })
    return sorted(rows, key=lambda row: row["gpu_index"])


def process_group_pids(pgid):
    if not _is_int(pgid) or pgid <= 0:
        raise ValueError("process group is invalid")
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            remainder = raw[raw.rfind(")") + 2:].split()
            if int(remainder[2]) == pgid:
                result.append(int(entry.name))
        except (
            FileNotFoundError,
            PermissionError,
            ValueError,
            IndexError,
        ):
            continue
    return sorted(result)


def exact_tag_processes(attempt):
    if not isinstance(attempt, str) or not attempt:
        raise ValueError("attempt is invalid")
    own_pid = os.getpid()
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError):
            continue
        if (
            attempt.encode("utf-8") in command
            and WORKER_SCRIPT.encode("utf-8") in command
        ):
            matches.append(int(entry.name))
    return sorted(matches)


def _load_worker_result(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("worker result is unavailable") from error
    if (
        not isinstance(payload, dict)
        or payload.get("classification") != "PASS"
        or not isinstance(payload.get("cases"), list)
        or not payload["cases"]
    ):
        raise RuntimeError("worker result is invalid")
    case_ids = [row.get("case_id") for row in payload["cases"]]
    if (
        any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
        or len(case_ids) != len(set(case_ids))
    ):
        raise RuntimeError("worker case inventory is invalid")
    return payload, case_ids


def supervise_worker(
    *,
    attempt,
    source_revision,
    attempt_root,
    source_root,
    model_root,
    python_path,
    selected_gpus,
    dist_port,
    poll_interval_s,
    worker_timeout_s,
    launcher=subprocess.Popen,
    inventory_query=query_gpu_inventory,
    pgid_resolver=os.getpgid,
    process_group_pids=process_group_pids,
    exact_tag_scan=exact_tag_processes,
    sleep=time.sleep,
    clock_ns=time.time_ns,
    monotonic=time.monotonic,
):
    attempt_root = Path(attempt_root).resolve()
    source_root = Path(source_root).resolve()
    model_root = Path(model_root).resolve()
    python_path = Path(python_path)
    selected = _validate_selected_gpus(selected_gpus)
    if (
        not isinstance(attempt, str)
        or not attempt
        or not isinstance(source_revision, str)
        or len(source_revision) != 40
        or not source_root.is_relative_to(attempt_root)
        or not _is_int(poll_interval_s)
        or poll_interval_s <= 0
        or not _is_int(worker_timeout_s)
        or worker_timeout_s <= 0
    ):
        raise ValueError("supervisor configuration is invalid")
    controller = attempt_root / "controller"
    cases = attempt_root / "cases"
    worker_output = attempt_root / "worker.json"
    for path in (
        controller,
        cases,
        attempt_root / "runtime" / "tmp",
        attempt_root / "runtime" / "cache" / "xdg",
        attempt_root / "runtime" / "cache" / "huggingface",
        attempt_root / "runtime" / "cache" / "torch",
        attempt_root / "runtime" / "cache" / "torch-extensions",
        attempt_root / "runtime" / "cache" / "cuda",
        attempt_root / "runtime" / "cache" / "triton",
    ):
        path.mkdir(parents=True, exist_ok=True)
    environment = build_worker_environment(
        attempt_root=attempt_root,
        selected_gpus=selected,
        base_environment=dict(os.environ),
        dist_port=dist_port,
    )
    worker_argv = [
        str(python_path),
        str(source_root / "tools" / WORKER_SCRIPT),
        "--attempt",
        attempt,
        "--source-revision",
        source_revision,
        "--model-root",
        str(model_root),
        "--output",
        str(worker_output),
        "--output-dir",
        str(cases),
        "--phase",
        "full",
    ]
    snapshots = []
    violations = []
    timed_out = False
    started_ns = clock_ns()
    started_monotonic = monotonic()
    stdout_path = controller / "worker.stdout"
    stderr_path = controller / "worker.stderr"
    with (
        stdout_path.open("a", encoding="utf-8") as stdout,
        stderr_path.open("a", encoding="utf-8") as stderr,
    ):
        process = launcher(
            worker_argv,
            cwd=source_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        pgid = pgid_resolver(process.pid)
        while process.poll() is None:
            owned_before = set(process_group_pids(pgid))
            try:
                observed = inventory_query()
                owned = owned_before | set(process_group_pids(pgid))
                snapshots.append(build_runtime_sample(
                    case_id="__runtime__",
                    selected_gpus=selected,
                    observed_gpus=observed,
                    owned_pids=owned,
                    captured_at_unix_ns=clock_ns(),
                ))
            except Exception as error:
                violations.append(
                    f"{type(error).__name__}: {error}"
                )
            if (
                not timed_out
                and monotonic() - started_monotonic > worker_timeout_s
            ):
                timed_out = True
                violations.append("worker monitoring deadline exceeded")
            sleep(poll_interval_s)
        returncode = process.wait()
    remaining = process_group_pids(pgid)
    scans = []
    for index in range(3):
        scans.append(list(exact_tag_scan(attempt)))
        if index != 2:
            sleep(poll_interval_s)
    cleanup = {
        "schema_version": CLEANUP_SCHEMA,
        "complete": (
            returncode == 0
            and remaining == []
            and scans == [[], [], []]
        ),
        "process_group_destroyed": remaining == [],
        "owned_children_remaining": list(remaining),
        "exact_tag_scans": scans,
    }
    _write_json_atomic(controller / "cleanup.json", cleanup)
    worker = None
    case_ids = []
    try:
        worker, case_ids = _load_worker_result(worker_output)
    except Exception as error:
        violations.append(f"{type(error).__name__}: {error}")
    resource_samples = []
    if case_ids and snapshots:
        final_sample = snapshots[-1]
        resource_samples = [
            {**final_sample, "case_id": case_id}
            for case_id in case_ids
        ]
    _write_json_atomic(
        controller / "resource_samples.json",
        resource_samples,
    )
    receipt = {
        "schema_version": SUPERVISOR_SCHEMA,
        "classification": (
            "PASS"
            if (
                returncode == 0
                and worker is not None
                and not violations
                and cleanup["complete"]
                and len(resource_samples) == len(case_ids)
            )
            else "FAIL"
        ),
        "attempt": attempt,
        "source_revision": source_revision,
        "worker_pid": process.pid,
        "worker_pgid": pgid,
        "worker_returncode": returncode,
        "started_at_unix_ns": started_ns,
        "finished_at_unix_ns": clock_ns(),
        "resource_snapshot_count": len(snapshots),
        "resource_sample_count": len(resource_samples),
        "violations": violations,
        "process_group_destroyed": cleanup["process_group_destroyed"],
        "owned_children_remaining": list(remaining),
        "exact_tag_scans": scans,
    }
    _write_json_atomic(
        controller / "supervisor_receipt.json",
        receipt,
    )
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--selected-gpus-json", required=True)
    parser.add_argument("--dist-port", type=int, default=29671)
    parser.add_argument("--poll-interval-s", type=int, default=1)
    parser.add_argument("--worker-timeout-s", type=int, default=21600)
    args = parser.parse_args(argv)
    selected = json.loads(args.selected_gpus_json)
    result = supervise_worker(
        attempt=args.attempt,
        source_revision=args.source_revision,
        attempt_root=args.attempt_root,
        source_root=args.source_root,
        model_root=args.model_root,
        python_path=args.python,
        selected_gpus=selected,
        dist_port=args.dist_port,
        poll_interval_s=args.poll_interval_s,
        worker_timeout_s=args.worker_timeout_s,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["classification"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
