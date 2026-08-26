from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import signal
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


def current_case():
    from tools.qwen38_tp4_communication_profile_worker import (
        build_structured_cases,
    )

    cases = build_structured_cases()
    complete = {
        path.stem for path in CASES.glob("*.json") if path.is_file()
    }
    for case in cases:
        case_id = (
            f"{case['workload']}__{case['phase']}__"
            f"r{case['repetition']}"
        )
        if case_id not in complete:
            return case
    return None


def main() -> int:
    for path in (SOURCE, MODEL, PYTHON):
        if not path.exists():
            raise RuntimeError(f"required path is missing: {path}")
    CASES.mkdir(parents=True, exist_ok=False)
    selected = require_clean_entry(gpu_inventory())
    canonical_write(
        CONTROLLER / "structured-admission.json",
        {
            "captured_at_unix_ns": time.time_ns(),
            "selected_gpus": selected,
        },
    )
    command = [
        str(PYTHON),
        str(SOURCE / "tools/qwen38_tp4_communication_profile_worker.py"),
        "--attempt=20260826-qwen38-tp4-communication-profile-r9",
        "--structured-campaign",
        f"--model-root={MODEL}",
        f"--output={STRUCTURED / 'campaign.json'}",
        f"--output-dir={CASES}",
        "--timeout-s=1800",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(index) for index, _ in SELECTED
    )
    environment["TINYVLLM_DIST_PORT"] = "29669"
    stdout_path = CONTROLLER / "structured.stdout"
    stderr_path = CONTROLLER / "structured.stderr"
    samples_path = CONTROLLER / "structured-resource-samples.raw.jsonl"
    aggregates: dict[tuple[str, int, str], dict] = {}
    violations = []
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
        samples_path.open("w", encoding="utf-8") as samples,
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
        started_ns = time.time_ns()
        while process.poll() is None:
            rows = gpu_inventory()
            by_uuid = {row["gpu_uuid"]: row for row in rows}
            owned = set(process_group_pids(pgid))
            case = current_case()
            sample = {
                "captured_at_unix_ns": time.time_ns(),
                "case": case,
                "owned_pids": sorted(owned),
                "selected_gpus": [],
            }
            for index, uuid in SELECTED:
                row = by_uuid.get(uuid)
                if row is None or row["gpu_index"] != index:
                    violations.append(f"GPU identity drift at index {index}")
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
                if case is not None and case["phase"] == "measured":
                    key = (case["workload"], case["repetition"], uuid)
                    current = aggregates.setdefault(key, {
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
    expected_resource_keys = {
        (workload, repetition, uuid)
        for workload in ("P0", "P1", "Q0", "Q1", "Q2")
        for repetition in range(5)
        for _, uuid in SELECTED
    }
    receipt = {
        "classification": (
            "PASS"
            if (
                returncode == 0
                and not violations
                and not remaining
                and set(aggregates) == expected_resource_keys
            )
            else "FAIL"
        ),
        "command": command,
        "pid": process.pid,
        "pgid": pgid,
        "started_at_unix_ns": started_ns,
        "finished_at_unix_ns": time.time_ns(),
        "returncode": returncode,
        "process_group_destroyed": not remaining,
        "owned_children_remaining": remaining,
        "violations": violations,
        "completed_case_count": len(list(CASES.glob("*.json"))),
        "resource_rows": [
            aggregates[key] for key in sorted(aggregates)
        ],
    }
    canonical_write(CONTROLLER / "structured-receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["classification"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
