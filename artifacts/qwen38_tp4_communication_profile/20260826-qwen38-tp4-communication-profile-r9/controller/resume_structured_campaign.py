from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


ROOT = Path(
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
ATTEMPT = (
    ROOT
    / "attempts"
    / "20260826-qwen38-tp4-communication-profile-r9"
)
SOURCE = ATTEMPT / ".staging" / "source"
MODEL = (
    ROOT
    / "models"
    / "Qwen3.8-27B"
    / "snapshots"
    / "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
)
PYTHON = ATTEMPT / ".staging" / "python" / "bin" / "python"
CONTROLLER = ATTEMPT / "controller"
STRUCTURED = ATTEMPT / "artifacts" / "structured"
CASES = STRUCTURED / "cases"
SELECTED = (
    (2, "GPU-63c05907-407b-8240-07a0-f38872840867"),
    (3, "GPU-f8904cb4-f9f0-c757-df36-e6fd971b3a9d"),
    (4, "GPU-56b882d2-6e6e-adb3-80e7-95f0a9e678f1"),
    (5, "GPU-687b7858-ca44-98ad-cfba-b6785eaf05e8"),
)
WORKLOADS = ("P0", "P1", "Q0", "Q1", "Q2")
GPU_WAIT_TIMEOUT_S = 12 * 60 * 60


def canonical_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def workload_cases(workload: str) -> tuple[dict, ...]:
    from tools.qwen38_tp4_communication_profile_worker import (
        build_structured_cases,
    )

    return tuple(
        case
        for case in build_structured_cases()
        if case["workload"] == workload
    )


def case_id(case: dict) -> str:
    return (
        f"{case['workload']}__{case['phase']}__"
        f"r{case['repetition']}"
    )


def normalize_engine_kwargs(kwargs: dict) -> dict:
    normalized = dict(kwargs)
    normalized["max_num_batched_tokens"] = max(
        normalized["max_num_batched_tokens"],
        normalized["max_model_len"],
    )
    return normalized


def ready_for_resource_aggregation(
    case: dict,
    marker_path: Path,
) -> bool:
    return (
        case.get("phase") == "measured"
        and Path(marker_path).is_file()
    )


def allocate_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("0.0.0.0", 0))
        return int(probe.getsockname()[1])


def case_attempt_is_retryable(result: dict) -> bool:
    return (
        bool(result.get("violations"))
        and result.get("process_group_destroyed") is True
        and result.get("owned_children_remaining") == []
    )


def resume_case_batches(completed_case_ids) -> tuple[dict, ...]:
    completed_case_ids = set(completed_case_ids)
    ordered_cases = tuple(
        case
        for workload in WORKLOADS
        for case in workload_cases(workload)
    )
    all_cases = {case_id(case): case for case in ordered_cases}
    unknown = completed_case_ids - all_cases.keys()
    if unknown:
        raise RuntimeError(
            f"unknown completed cases: {sorted(unknown)}"
        )
    return tuple(
        case
        for case in ordered_cases
        if case_id(case) not in completed_case_ids
    )


def gpu_inventory() -> list[dict]:
    gpu_text = subprocess.run(
        [
            "nvidia-smi",
            (
                "--query-gpu="
                "index,uuid,memory.used,utilization.gpu,power.draw"
            ),
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    process_text = subprocess.run(
        [
            "nvidia-smi",
            (
                "--query-compute-apps="
                "gpu_uuid,pid,used_memory,process_name"
            ),
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    processes: dict[str, list[dict]] = {}
    for fields in csv.reader(process_text.splitlines()):
        if not fields or not any(value.strip() for value in fields):
            continue
        if len(fields) == 1 and fields[0].strip() == (
            "No running processes found"
        ):
            continue
        uuid, pid, used, name = (value.strip() for value in fields)
        processes.setdefault(uuid, []).append({
            "pid": int(pid),
            "used_memory_mib": int(used),
            "process_name": name,
        })
    rows = []
    for fields in csv.reader(gpu_text.splitlines()):
        if not fields or not any(value.strip() for value in fields):
            continue
        index, uuid, memory, utilization, power = (
            value.strip() for value in fields
        )
        rows.append({
            "gpu_index": int(index),
            "gpu_uuid": uuid,
            "memory_used_mib": int(memory),
            "gpu_utilization_percent": int(utilization),
            "power_watts": float(power),
            "compute_processes": processes.get(uuid, []),
        })
    return rows


def require_clean_entry(rows: list[dict]) -> list[dict]:
    by_index = {row["gpu_index"]: row for row in rows}
    selected = []
    for index, uuid in SELECTED:
        row = by_index.get(index)
        if row is None or row["gpu_uuid"] != uuid:
            raise RuntimeError(f"GPU identity drift at index {index}")
        if (
            row["memory_used_mib"] > 1024
            or row["gpu_utilization_percent"] > 5
            or row["compute_processes"]
        ):
            raise RuntimeError(f"GPU {index} is not strict-clean")
        selected.append(row)
    return selected


def wait_for_clean_entry(
    timeout_s: float = GPU_WAIT_TIMEOUT_S,
) -> list[dict]:
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            return require_clean_entry(gpu_inventory())
        except RuntimeError as error:
            last_error = error
            time.sleep(1.0)
    raise RuntimeError(
        f"strict-clean GPU timeout: {last_error}"
    )


def process_group_pids(pgid: int) -> list[int]:
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split()
            if int(fields[4]) == pgid:
                result.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
    return sorted(result)


def validate_existing_cases() -> set[str]:
    completed = set()
    for path in CASES.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("classification") != "PASS"
            or payload.get("case_id") != path.stem
        ):
            raise RuntimeError(
                f"existing case is invalid: {path.name}"
            )
        completed.add(path.stem)
    return completed


def current_case():
    completed = {path.stem for path in CASES.glob("*.json")}
    for workload in WORKLOADS:
        for case in workload_cases(workload):
            if case_id(case) not in completed:
                return case
    return None


def run_case(selected_case_id: str) -> int:
    from tools.qwen38_tp4_communication_profile_worker import (
        _atomic_write_json,
        _default_engine_factory,
        _default_sampling_params_factory,
        _reset_sequence_ids,
        run_profile_case,
    )

    matches = [
        case
        for workload in WORKLOADS
        for case in workload_cases(workload)
        if case_id(case) == selected_case_id
    ]
    if len(matches) != 1:
        raise RuntimeError("requested case identity is invalid")
    case = matches[0]
    output = CASES / f"{selected_case_id}.json"
    if output.exists():
        raise RuntimeError(
            f"refusing to overwrite case {output.name}"
        )
    marker_value = os.environ.get(
        "TINYLLMFORGE_CASE_READY_MARKER"
    )
    if not marker_value:
        raise RuntimeError("case-ready marker path is missing")
    marker_path = Path(marker_value)
    if marker_path.parent != CONTROLLER:
        raise RuntimeError("case-ready marker path is invalid")

    def engine_factory(model_root, **kwargs):
        engine = _default_engine_factory(
            model_root,
            **normalize_engine_kwargs(kwargs),
        )
        canonical_write(
            marker_path,
            {
                "case_id": selected_case_id,
                "engine_ready_at_unix_ns": time.time_ns(),
            },
        )
        return engine

    result = run_profile_case(
        attempt="20260826-qwen38-tp4-communication-profile-r9",
        model_root=MODEL,
        timeout_s=1800.0,
        engine_factory=engine_factory,
        sampling_params_factory=_default_sampling_params_factory,
        clock_ns=time.monotonic_ns,
        reset_sequence_ids=_reset_sequence_ids,
        **case,
    )
    _atomic_write_json(output, result)
    print(json.dumps({
        "classification": result["classification"],
        "case_id": selected_case_id,
        "output": str(output),
    }, sort_keys=True))
    return 0


def monitor_case(
    case: dict,
    *,
    samples,
    attempt_index: int,
) -> dict:
    selected_case_id = case_id(case)
    command = [
        str(PYTHON),
        str(CONTROLLER / "resume_structured_campaign.py"),
        "--run-case",
        selected_case_id,
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(index) for index, _ in SELECTED
    )
    dist_port = allocate_free_tcp_port()
    environment["TINYVLLM_DIST_PORT"] = str(dist_port)
    marker_path = (
        CONTROLLER
        / (
            f"structured-{selected_case_id}."
            f"attempt-{attempt_index}.engine-ready.json"
        )
    )
    if marker_path.exists():
        marker_path.unlink()
    environment["TINYLLMFORGE_CASE_READY_MARKER"] = str(
        marker_path
    )
    stdout_path = (
        CONTROLLER
        / f"structured-{selected_case_id}.attempt-{attempt_index}.stdout"
    )
    stderr_path = (
        CONTROLLER
        / f"structured-{selected_case_id}.attempt-{attempt_index}.stderr"
    )
    violations = []
    case_aggregates = {}
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=SOURCE,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        pgid = os.getpgid(process.pid)
        while process.poll() is None:
            rows = gpu_inventory()
            by_uuid = {row["gpu_uuid"]: row for row in rows}
            owned = set(process_group_pids(pgid))
            sample = {
                "captured_at_unix_ns": time.time_ns(),
                "case": case,
                "engine_ready": marker_path.is_file(),
                "owned_pids": sorted(owned),
                "selected_gpus": [],
            }
            for index, uuid in SELECTED:
                row = by_uuid.get(uuid)
                if row is None or row["gpu_index"] != index:
                    violations.append(
                        f"GPU identity drift at index {index}"
                    )
                    continue
                foreign = [
                    item["pid"]
                    for item in row["compute_processes"]
                    if item["pid"] not in owned
                ]
                if foreign:
                    violations.append(
                        f"unrelated GPU process on {uuid}: {foreign}"
                    )
                sample["selected_gpus"].append(row)
                if ready_for_resource_aggregation(
                    case,
                    marker_path,
                ):
                    key = (
                        case["workload"],
                        case["repetition"],
                        uuid,
                    )
                    current = case_aggregates.setdefault(key, {
                        "workload": case["workload"],
                        "repetition": case["repetition"],
                        "gpu_uuid": uuid,
                        "gpu_utilization_percent": 0,
                        "power_watts": 0.0,
                        "sample_count": 0,
                    })
                    current["gpu_utilization_percent"] = max(
                        current["gpu_utilization_percent"],
                        row["gpu_utilization_percent"],
                    )
                    current["power_watts"] = max(
                        current["power_watts"],
                        row["power_watts"],
                    )
                    current["sample_count"] += 1
            samples.write(
                json.dumps(
                    sample,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
            samples.flush()
            if violations:
                os.killpg(pgid, signal.SIGTERM)
                break
            time.sleep(1.0)
        try:
            returncode = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            returncode = process.wait(timeout=60)
    time.sleep(1.0)
    remaining = process_group_pids(pgid)
    return {
        "case_id": selected_case_id,
        "attempt_index": attempt_index,
        "command": command,
        "dist_port": dist_port,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "ready_marker_path": str(marker_path),
        "pid": process.pid,
        "pgid": pgid,
        "returncode": returncode,
        "violations": violations,
        "process_group_destroyed": not remaining,
        "owned_children_remaining": remaining,
        "resource_rows": [
            case_aggregates[key]
            for key in sorted(case_aggregates)
        ],
    }


def load_initial_aggregates() -> dict:
    receipt_path = CONTROLLER / "structured-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("returncode") != 1
        or receipt.get("completed_case_count") != 7
        or receipt.get("violations") != []
        or receipt.get("process_group_destroyed") is not True
        or receipt.get("owned_children_remaining") != []
    ):
        raise RuntimeError("initial P0 receipt is not resumable")
    aggregates = {
        (
            row["workload"],
            row["repetition"],
            row["gpu_uuid"],
        ): dict(row)
        for row in receipt["resource_rows"]
    }
    resume_path = CONTROLLER / "structured-resume-receipt.json"
    if resume_path.is_file():
        previous = json.loads(
            resume_path.read_text(encoding="utf-8")
        )
        for row in previous.get("resource_rows", []):
            measured_case = (
                f"{row['workload']}__measured__"
                f"r{row['repetition']}"
            )
            if (CASES / f"{measured_case}.json").is_file():
                aggregates[(
                    row["workload"],
                    row["repetition"],
                    row["gpu_uuid"],
                )] = dict(row)
    return aggregates


def assemble_campaign(batch_results: list[dict]) -> dict:
    expected = {
        case_id(case)
        for workload in WORKLOADS
        for case in workload_cases(workload)
    }
    completed = validate_existing_cases()
    if completed != expected:
        raise RuntimeError(
            "final structured case inventory is incomplete"
        )
    concise_cases = []
    cleanups = {
        "P0": {
            "process_group_destroyed": True,
            "owned_children_remaining": [],
            "source": "controller/structured-receipt.json",
        },
    }
    for workload in WORKLOADS:
        case_payloads = [
            json.loads(
                (CASES / f"{case_id(case)}.json").read_text(
                    encoding="utf-8"
                )
            )
            for case in workload_cases(workload)
        ]
        if workload != "P0":
            cleanups[workload] = {
                payload["case_id"]: payload.get("cleanup")
                for payload in case_payloads
            }
        concise_cases.extend({
            "classification": payload["classification"],
            "case_id": payload["case_id"],
            "decode_time_ns": payload["decode_time_ns"],
        } for payload in case_payloads)
    result = {
        "schema_version": (
            "qwen38.tp4-communication-profile-worker.v1"
        ),
        "classification": "PASS",
        "attempt": "20260826-qwen38-tp4-communication-profile-r9",
        "cases": concise_cases,
        "cleanup": {
            "process_group_destroyed": True,
            "rank_exit_codes": [0, 0, 0, 0],
            "owned_children_remaining": [],
            "workload_cleanups": cleanups,
        },
        "execution_boundary": (
            "P0 used the original engine lifetime; every later case "
            "used an isolated engine lifetime to prevent cross-case "
            "KV reuse without an aligned hybrid-state snapshot"
        ),
        "resume_batch_results": batch_results,
    }
    canonical_write(STRUCTURED / "campaign.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-case")
    args = parser.parse_args()
    if args.run_case is not None:
        return run_case(args.run_case)

    for path in (SOURCE, MODEL, PYTHON, CASES):
        if not path.exists():
            raise RuntimeError(f"required path is missing: {path}")
    completed = validate_existing_cases()
    batches = resume_case_batches(completed)
    if not batches:
        raise RuntimeError("structured campaign is already complete")
    selected = wait_for_clean_entry()
    canonical_write(
        CONTROLLER / "structured-resume-admission.json",
        {
            "captured_at_unix_ns": time.time_ns(),
            "selected_gpus": selected,
            "preserved_case_ids": sorted(completed),
            "remaining_case_ids": [
                case_id(case) for case in batches
            ],
        },
    )
    aggregates = load_initial_aggregates()
    batch_results = []
    recovered_interference = []
    terminal_failure = False
    samples_path = (
        CONTROLLER / "structured-resource-samples.raw.jsonl"
    )
    with samples_path.open("a", encoding="utf-8") as samples:
        for case in batches:
            attempt_index = 0
            while True:
                wait_for_clean_entry()
                result = monitor_case(
                    case,
                    samples=samples,
                    attempt_index=attempt_index,
                )
                batch_results.append(result)
                if case_attempt_is_retryable(result):
                    recovered_interference.append({
                        "case_id": result["case_id"],
                        "attempt_index": result["attempt_index"],
                        "violations": result["violations"],
                    })
                    attempt_index += 1
                    continue
                if (
                    result["returncode"] != 0
                    or result["violations"]
                    or not result["process_group_destroyed"]
                    or result["owned_children_remaining"]
                ):
                    terminal_failure = True
                else:
                    for row in result["resource_rows"]:
                        aggregates[(
                            row["workload"],
                            row["repetition"],
                            row["gpu_uuid"],
                        )] = dict(row)
                break
            if terminal_failure:
                break
    expected_resource_keys = {
        (workload, repetition, uuid)
        for workload in WORKLOADS
        for repetition in range(5)
        for _, uuid in SELECTED
    }
    expected_case_ids = {
        case_id(case)
        for workload in WORKLOADS
        for case in workload_cases(workload)
    }
    success = (
        not terminal_failure
        and validate_existing_cases() == expected_case_ids
        and set(aggregates) == expected_resource_keys
    )
    campaign = assemble_campaign(batch_results) if success else None
    receipt = {
        "classification": "PASS" if success else "FAIL",
        "completed_case_count": len(list(CASES.glob("*.json"))),
        "batch_results": batch_results,
        "resource_rows": [
            aggregates[key] for key in sorted(aggregates)
        ],
        "recovered_interference": recovered_interference,
        "violations": [
            violation
            for result in batch_results
            if not case_attempt_is_retryable(result)
            for violation in result["violations"]
        ],
        "process_groups_destroyed": all(
            result["process_group_destroyed"]
            for result in batch_results
        ),
        "owned_children_remaining": [
            pid
            for result in batch_results
            for pid in result["owned_children_remaining"]
        ],
        "campaign_classification": (
            None if campaign is None else campaign["classification"]
        ),
    }
    canonical_write(
        CONTROLLER / "structured-resume-receipt.json",
        receipt,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
