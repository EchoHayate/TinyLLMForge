from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time


SCHEMA_VERSION = 1
WORLD_SIZE = 4
EXPECTED_ALL_REDUCE_VALUE = 10.0
EXPECTED_BROADCAST_VALUE = 7.0


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _rank_rows(rows) -> tuple[dict, ...]:
    if (
        not isinstance(rows, (tuple, list))
        or len(rows) != WORLD_SIZE
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError("diagnostic requires four rank rows")
    normalized = tuple(sorted(
        (dict(row) for row in rows),
        key=lambda row: row.get("rank", -1),
    ))
    if tuple(row.get("rank") for row in normalized) != tuple(
        range(WORLD_SIZE)
    ):
        raise ValueError("diagnostic rank inventory mismatch")
    if any(
        row.get("world_size") != WORLD_SIZE
        or row.get("device_index") != row["rank"]
        for row in normalized
    ):
        raise ValueError("diagnostic TP4 device mapping mismatch")
    return normalized


def summarize_rank_rows(rows) -> dict:
    rows = _rank_rows(rows)
    if not all(row.get("capture_completed") is True for row in rows):
        raise ValueError("collective capture did not complete on every rank")
    if not all(row.get("replay_completed") is True for row in rows):
        raise ValueError("collective replay did not complete on every rank")
    all_reduce_values = tuple(
        float(row.get("all_reduce_value")) for row in rows
    )
    if any(
        value != EXPECTED_ALL_REDUCE_VALUE
        for value in all_reduce_values
    ):
        raise ValueError("all-reduce replay parity mismatch")
    broadcast_values = tuple(
        float(row.get("broadcast_value")) for row in rows
    )
    if any(
        value != EXPECTED_BROADCAST_VALUE
        for value in broadcast_values
    ):
        raise ValueError("broadcast replay parity mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "PASS",
        "rank_count": len(rows),
        "capture_completed": True,
        "replay_completed": True,
        "all_reduce_values": all_reduce_values,
        "broadcast_values": broadcast_values,
        "ranks": rows,
    }


def _write_rank_row(output_root: Path, rank: int, row: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".rank-{rank}.json.tmp"
    final = output_root / f"rank-{rank}.json"
    temporary.write_bytes(_canonical_json_bytes(row))
    temporary.replace(final)


def release_captured_graph(graph, *, synchronize) -> None:
    reset = getattr(graph, "reset", None)
    if not callable(reset):
        raise ValueError("captured graph must expose callable reset")
    if not callable(synchronize):
        raise ValueError("synchronize must be callable")
    reset()
    synchronize()


def run_diagnostic(*, output_root: Path) -> dict | None:
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise ValueError("diagnostic requires exact TP4")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    initialized = False
    row = {
        "schema_version": SCHEMA_VERSION,
        "rank": rank,
        "world_size": world_size,
        "device_index": local_rank,
        "capture_completed": False,
        "replay_completed": False,
        "all_reduce_value": None,
        "broadcast_value": None,
        "error_type": None,
        "message": None,
        "torch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda),
        "nccl_version": str(torch.cuda.nccl.version()),
    }
    started_ns = time.perf_counter_ns()
    graph = None
    try:
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=120),
        )
        initialized = True
        graph_pool = torch.cuda.graph_pool_handle()
        capture_stream = torch.cuda.Stream(device=device)
        all_reduce_tensor = torch.tensor(
            [float(rank + 1)],
            dtype=torch.float32,
            device=device,
        )
        broadcast_tensor = torch.tensor(
            [EXPECTED_BROADCAST_VALUE if rank == 0 else 0.0],
            dtype=torch.float32,
            device=device,
        )

        dist.barrier()
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                all_reduce_tensor.fill_(float(rank + 1))
                broadcast_tensor.fill_(
                    EXPECTED_BROADCAST_VALUE if rank == 0 else 0.0
                )
                dist.all_reduce(all_reduce_tensor)
                dist.broadcast(broadcast_tensor, src=0)
        capture_stream.synchronize()
        dist.barrier()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            pool=graph_pool,
            stream=capture_stream,
        ):
            dist.all_reduce(all_reduce_tensor)
            dist.broadcast(broadcast_tensor, src=0)
        torch.cuda.synchronize()
        row["capture_completed"] = True

        all_reduce_tensor.fill_(float(rank + 1))
        broadcast_tensor.fill_(
            EXPECTED_BROADCAST_VALUE if rank == 0 else 0.0
        )
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        row["replay_completed"] = True
        row["all_reduce_value"] = float(all_reduce_tensor.item())
        row["broadcast_value"] = float(broadcast_tensor.item())
        row["elapsed_ns"] = time.perf_counter_ns() - started_ns
        _write_rank_row(output_root, rank, row)

        gathered = [None for _ in range(WORLD_SIZE)]
        dist.all_gather_object(gathered, row)
        if rank != 0:
            return None
        summary = summarize_rank_rows(gathered)
        (output_root / "summary.json").write_bytes(
            _canonical_json_bytes(summary)
        )
        return summary
    except BaseException as error:
        row["error_type"] = type(error).__name__
        row["message"] = str(error)[:4096]
        row["elapsed_ns"] = time.perf_counter_ns() - started_ns
        _write_rank_row(output_root, rank, row)
        raise
    finally:
        if graph is not None:
            release_captured_graph(
                graph,
                synchronize=torch.cuda.synchronize,
            )
        if initialized:
            dist.destroy_process_group()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_diagnostic(output_root=Path(args.out_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
