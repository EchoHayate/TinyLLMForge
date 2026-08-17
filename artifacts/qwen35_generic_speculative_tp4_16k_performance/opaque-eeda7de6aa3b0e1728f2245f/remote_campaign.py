from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


source_root = Path(os.environ["REMOTE_SOURCE"])
gate_path = (
    source_root
    / "tools"
    / "qwen35_generic_speculative_tp4_16k_performance_gate.py"
)
spec = importlib.util.spec_from_file_location(
    "qwen35_tp4_16k_performance_remote_gate",
    gate_path,
)
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)

gpu_indices = tuple(
    int(item)
    for item in os.environ["SELECTED_GPU_CSV"].split(",")
)
minimum_free = int(os.environ["MIN_FREE_MEMORY_MIB"])
maximum_utilization = int(os.environ["MAX_GPU_UTILIZATION"])
maximum_drift = int(os.environ["MAX_POST_CELL_DRIFT_MIB"])
settle_attempts = int(os.environ["POST_SETTLE_ATTEMPTS"])
settle_interval = int(os.environ["POST_SETTLE_INTERVAL_SECONDS"])


def inventory():
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        rows.append({
            "index": int(fields[0]),
            "memory_free_mib": int(fields[1]),
            "memory_total_mib": int(fields[2]),
            "utilization_gpu_percent": int(fields[3]),
        })
    return rows


def selected(rows):
    by_index = {row["index"]: row for row in rows}
    if set(gpu_indices) - set(by_index):
        raise RuntimeError("selected GPU inventory changed")
    return [by_index[index] for index in gpu_indices]


def require_pre_cell(rows, key):
    for row in rows:
        if row["memory_free_mib"] < minimum_free:
            raise RuntimeError(f"{key} free-memory preflight failed")
        if row["utilization_gpu_percent"] > maximum_utilization:
            raise RuntimeError(f"{key} utilization preflight failed")


environment = {
    "python_version": sys.version.split()[0],
    "torch_version": "loaded-worker-recorded",
    "device_name": "nvidia-smi-inventory",
    "gpu_inventory": {
        "selected_physical_indices": list(gpu_indices),
        "campaign_start": inventory(),
        "pre_cells": {},
        "post_cells": {},
    },
}


def worker_runner(command, *, log_path, cwd):
    policy = command[command.index("--policy") + 1]
    batch_size = command[command.index("--batch-size") + 1]
    key = f"{policy}:b{batch_size}"
    pre_rows = selected(inventory())
    require_pre_cell(pre_rows, key)
    environment["gpu_inventory"]["pre_cells"][key] = pre_rows
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(log_path).open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    post_rows = None
    stable = False
    for attempt in range(settle_attempts):
        if attempt:
            time.sleep(settle_interval)
        candidate = selected(inventory())
        stable = all(
            abs(
                after["memory_free_mib"]
                - before["memory_free_mib"]
            )
            <= maximum_drift
            and after["utilization_gpu_percent"]
            <= maximum_utilization
            for before, after in zip(pre_rows, candidate)
        )
        post_rows = candidate
        if stable:
            break
    environment["gpu_inventory"]["post_cells"][key] = post_rows
    if completed.returncode != 0:
        return int(completed.returncode)
    if not stable:
        return 91
    return 0


gate.run_campaign(
    model_path=os.environ["MODEL_PATH"],
    gpu_indices=gpu_indices,
    output_dir=Path(os.environ["REMOTE_ARTIFACTS"]) / "authority",
    dist_port_base=int(os.environ["DIST_PORT_BASE"]),
    master_port_base=int(os.environ["MASTER_PORT_BASE"]),
    repo_root=source_root,
    worker_script=(
        source_root
        / "tools"
        / "qwen35_generic_speculative_tp4_16k_performance_worker.py"
    ),
    worker_runner=worker_runner,
    python_executable=os.environ["REMOTE_PYTHON"],
    environment=environment,
)
