from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time


def main():
    raw_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    selected = {}
    with raw_path.open(newline="") as handle:
        for row in csv.reader(handle):
            gpu_index = int(row[0].strip())
            if gpu_index in (2, 4, 5, 6):
                selected[gpu_index] = {
                    "gpu_index": gpu_index,
                    "gpu_uuid": row[1].strip(),
                    "free_mib": int(row[2].strip()),
                    "utilization_percent": int(row[3].strip()),
                }
    reasons = []
    for gpu_index in (2, 4, 5, 6):
        row = selected.get(gpu_index)
        if row is None:
            reasons.append(f"GPU {gpu_index} missing")
            continue
        if row["free_mib"] < 25600:
            reasons.append(
                f"GPU {gpu_index} has less than 25 GiB free"
            )
        if row["utilization_percent"] > 10:
            reasons.append(
                f"GPU {gpu_index} utilization exceeds 10 percent"
            )
    payload = {
        "classification": (
            "READY" if not reasons else "BLOCKED_RESOURCES"
        ),
        "resource_policy": "shared-low-utilization",
        "exclusive": False,
        "minimum_gpu_free_mib": 25600,
        "maximum_gpu_utilization_percent": 10,
        "observed_at_ns": time.time_ns(),
        "selected_gpus": [
            selected[gpu_index]
            for gpu_index in (2, 4, 5, 6)
            if gpu_index in selected
        ],
        "reasons": reasons,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True), flush=True)
    if reasons:
        raise SystemExit(42)


if __name__ == "__main__":
    main()
