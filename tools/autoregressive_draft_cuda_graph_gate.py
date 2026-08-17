from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from autoregressive_draft_cuda_graph_contract import (
    build_gate_payload,
    canonical_json_bytes,
)


def build_pair_schedule(
    *,
    warmup_pairs: int,
    measured_pairs: int,
) -> dict:
    for name, value, minimum in (
        ("warmup pairs", warmup_pairs, 2),
        ("measured pairs", measured_pairs, 8),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
            or value % 2
        ):
            raise ValueError(
                f"{name} must be an even integer >= {minimum}"
            )
    return {
        "warmups": [
            (
                "eager_graph"
                if index % 2 == 0
                else "graph_eager"
            )
            for index in range(warmup_pairs)
        ],
        "measured": [
            (
                "eager_graph"
                if index % 2 == 0
                else "graph_eager"
            )
            for index in range(measured_pairs)
        ],
    }


def build_worker_command(
    *,
    python: str,
    worker_script: str,
    target_model: str,
    draft_model: str,
    mode: str,
    output_path: str,
) -> list[str]:
    if mode not in ("eager", "graph"):
        raise ValueError("worker mode is invalid")
    return [
        python,
        worker_script,
        "--target-model",
        target_model,
        "--draft-model",
        draft_model,
        "--policy",
        "learned",
        "--batch-size",
        "4",
        "--cuda-graph-mode",
        mode,
        "--warmup-runs",
        "1",
        "--measured-runs",
        "1",
        "--out",
        output_path,
    ]


def _seconds_to_ns(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive")
    return int(round(float(value) * 1_000_000_000))


def _timing_row(run: dict) -> dict:
    timing = run.get("timing")
    runtime = run.get("runtime")
    if not isinstance(timing, dict) or not isinstance(
        runtime,
        dict,
    ):
        raise ValueError("worker timing evidence is missing")
    per_request = timing.get("per_request")
    if isinstance(per_request, list) and per_request:
        ttft_s = statistics.median(
            row["ttft_s"] for row in per_request
        )
        tpot_s = statistics.median(
            row["tpot_s"] for row in per_request
        )
    else:
        ttft_s = timing.get("ttft_s")
        tpot_s = timing.get("tpot_s")
    e2e_s = timing.get(
        "batch_elapsed_s",
        timing.get("latency_s"),
    )
    throughput = timing.get(
        "batch_token_throughput_tps",
        timing.get("throughput_tokens_per_second"),
    )
    executor_timing = runtime.get(
        "draft_executor_timing",
        {},
    )
    proposal_detail = runtime.get(
        "draft_executor_proposal_detail",
        {},
    )
    proposal_forward_ms = executor_timing.get(
        "max_rank_ms",
        {},
    ).get("proposal_forward")
    detail_ms = proposal_detail.get(
        "critical_rank_ms",
        {},
    )
    if (
        not isinstance(proposal_forward_ms, (int, float))
        or proposal_forward_ms <= 0
    ):
        raise ValueError(
            "worker proposal-forward timing is invalid"
        )
    detail_names = (
        "setup",
        "backend_submit",
        "selection_collective",
        "decode_authority",
        "token_readback",
        "materialize_register",
    )
    return {
        "e2e_ns": _seconds_to_ns(e2e_s, "worker E2E timing"),
        "throughput_tokens_per_second": float(throughput),
        "ttft_ns": _seconds_to_ns(ttft_s, "worker TTFT"),
        "tpot_ns": _seconds_to_ns(tpot_s, "worker TPOT"),
        "proposal_forward_ns": int(round(
            float(proposal_forward_ms) * 1_000_000
        )),
        "proposal_detail_ns": {
            name: int(round(
                float(detail_ms[name]) * 1_000_000
            ))
            for name in detail_names
        },
    }


def mode_row_from_worker(worker: dict, *, mode: str) -> dict:
    if (
        not isinstance(worker, dict)
        or worker.get("policy") != "learned"
        or worker.get("batch_size") != 4
        or worker.get("cuda_graph_mode") != mode
    ):
        raise ValueError("worker identity mismatch")
    warmups = worker.get("warmup_runs")
    measured = worker.get("measured_runs")
    if not isinstance(warmups, list) or len(warmups) != 1:
        raise ValueError(
            "worker must contain one in-process warmup run"
        )
    if not isinstance(measured, list) or len(measured) != 1:
        raise ValueError("worker must contain one measured run")
    warmup = warmups[0]
    run = measured[0]
    warmup_correctness = warmup.get("correctness")
    correctness = run.get("correctness")
    runtime = run.get("runtime")
    memory = run.get("memory")
    outputs = run.get("outputs")
    if (
        not isinstance(warmup_correctness, dict)
        or
        not isinstance(correctness, dict)
        or not isinstance(runtime, dict)
        or not isinstance(memory, dict)
        or not isinstance(outputs, list)
    ):
        raise ValueError("worker raw evidence is missing")
    memory_rows = []
    for row in memory.get("ranks", ()):
        memory_rows.append({
            "rank": row["rank"],
            "peak_allocated_bytes": row.get(
                "peak_allocated_bytes",
                row.get("cuda_peak_allocated_bytes"),
            ),
            "peak_reserved_bytes": row.get(
                "peak_reserved_bytes",
                row.get("cuda_peak_reserved_bytes"),
            ),
        })
    proposed = runtime.get("proposed_tokens")
    accepted = runtime.get("accepted_draft_tokens")
    stage_count = runtime.get(
        "stage_timing",
        {},
    ).get("step_count", 0)
    return {
        "mode": mode,
        "target_token_rows": outputs,
        "proposal_token_rows": correctness[
            "proposal_token_rows"
        ],
        "accepted_prefix_counts": correctness[
            "accepted_prefix_counts"
        ],
        "transaction_digest": correctness[
            "transaction_digest"
        ],
        "active_transaction_count": correctness[
            "active_transaction_count"
        ],
        "warmup_rank_graph_counters": warmup_correctness[
            "rank_graph_counters"
        ],
        "rank_graph_counters": correctness[
            "rank_graph_counters"
        ],
        "warmup_rank_graph_resources": warmup_correctness[
            "rank_graph_resources"
        ],
        "rank_graph_resources": correctness[
            "rank_graph_resources"
        ],
        "rank_memory_rows": memory_rows,
        "timing": _timing_row(run),
        "acceptance": {
            "proposed_tokens": proposed,
            "accepted_tokens": accepted,
            "accepted_tokens_per_target_call": (
                accepted / stage_count if stage_count else 0.0
            ),
            "rate": runtime.get("acceptance_rate"),
        },
    }


def _run_worker(command, output_path: Path) -> dict:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CUDA graph gate worker failed: "
            + completed.stdout[-4000:]
        )
    try:
        return json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "CUDA graph gate worker result is unreadable"
        ) from error


def _live_gpu_snapshot(selected_uuids: list[str]) -> list[dict]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if query.returncode != 0:
        raise RuntimeError(
            "live GPU snapshot failed: "
            + (query.stderr or query.stdout).strip()
        )
    applications = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    process_counts = {}
    if applications.returncode == 0:
        for line in applications.stdout.splitlines():
            gpu_uuid = line.strip()
            if gpu_uuid:
                process_counts[gpu_uuid] = (
                    process_counts.get(gpu_uuid, 0) + 1
                )
    by_uuid = {}
    for line in query.stdout.splitlines():
        index, gpu_uuid, memory, utilization = [
            value.strip() for value in line.split(",")
        ]
        by_uuid[gpu_uuid] = {
            "physical_index": int(index),
            "uuid": gpu_uuid,
            "memory_used_mib": int(memory),
            "utilization_percent": int(utilization),
            "compute_process_count": process_counts.get(
                gpu_uuid,
                0,
            ),
        }
    if any(gpu_uuid not in by_uuid for gpu_uuid in selected_uuids):
        raise RuntimeError("selected GPU disappeared during gate")
    return [
        {"rank": rank, **by_uuid[gpu_uuid]}
        for rank, gpu_uuid in enumerate(selected_uuids)
    ]


def _environment_interference(rows: list[dict]) -> bool:
    return any(
        row["compute_process_count"] != 0
        or row["memory_used_mib"] > 1024
        or row["utilization_percent"] > 5
        for row in rows
    )


def execute_gate(
    *,
    target_model: str,
    draft_model: str,
    provenance: dict,
    environment: dict,
    output_path: Path,
    warmup_pairs: int,
    measured_pairs: int,
    python: str = sys.executable,
    worker_script: Path | None = None,
    live_environment: bool = False,
) -> dict:
    schedule = build_pair_schedule(
        warmup_pairs=warmup_pairs,
        measured_pairs=measured_pairs,
    )
    worker_script = (
        TOOLS_ROOT / "autoregressive_draft_performance_worker.py"
        if worker_script is None
        else Path(worker_script)
    )
    worker_root = Path(output_path).parent / "workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    if live_environment:
        selected_uuids = provenance["gpu_uuids"]
        environment = {
            **environment,
            "gpu_before": _live_gpu_snapshot(
                selected_uuids
            ),
        }

    def execute_pair(index, order, *, warmup):
        rows = {}
        for mode in order.split("_"):
            worker_output = worker_root / (
                f"{'warmup' if warmup else 'pair'}-"
                f"{index}-{mode}.json"
            )
            command = build_worker_command(
                python=python,
                worker_script=str(worker_script),
                target_model=target_model,
                draft_model=draft_model,
                mode=mode,
                output_path=str(worker_output),
            )
            rows[mode] = mode_row_from_worker(
                _run_worker(command, worker_output),
                mode=mode,
            )
        result = {
            (
                "warmup_index"
                if warmup
                else "pair_index"
            ): index,
            "eager": rows["eager"],
            "graph": rows["graph"],
        }
        if not warmup:
            result["order"] = order
        return result

    warmups = [
        execute_pair(index, order, warmup=True)
        for index, order in enumerate(schedule["warmups"])
    ]
    pairs = [
        execute_pair(index, order, warmup=False)
        for index, order in enumerate(schedule["measured"])
    ]
    if live_environment:
        gpu_after = _live_gpu_snapshot(
            provenance["gpu_uuids"]
        )
        environment = {
            **environment,
            "gpu_after": gpu_after,
            "interference_detected": (
                _environment_interference(
                    environment["gpu_before"]
                )
                or _environment_interference(gpu_after)
            ),
        }
    payload = build_gate_payload(
        provenance=provenance,
        environment=environment,
        warmups=warmups,
        pairs=pairs,
    )
    Path(output_path).write_bytes(canonical_json_bytes(payload))
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-proposal-tokens", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--warmup-pairs", type=int, default=2)
    parser.add_argument("--measured-pairs", type=int, default=8)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    exact = (
        args.tensor_parallel_size,
        args.batch_size,
        args.max_proposal_tokens,
        args.prompt_tokens,
        args.output_tokens,
    )
    if exact != (4, 4, 4, 256, 16):
        raise ValueError("gate configuration is not exact TP4/B4/Q4")
    provenance = json.loads(
        Path(args.provenance).read_text(encoding="utf-8")
    )
    environment = json.loads(
        Path(args.environment).read_text(encoding="utf-8")
    )
    execute_gate(
        target_model=args.target_model,
        draft_model=args.draft_model,
        provenance=provenance,
        environment=environment,
        output_path=Path(args.out),
        warmup_pairs=args.warmup_pairs,
        measured_pairs=args.measured_pairs,
        live_environment=bool(
            os.environ.get("TINYLLM_GATE_LIVE_ENVIRONMENT")
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
