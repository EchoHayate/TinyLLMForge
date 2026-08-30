#!/usr/bin/env python3
"""Run the isolated four-GPU TP4 peer-reduction microgate."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time


ACTIVE_TOKEN_GROUPS = (1, 4, 8)
WARMUP_PAIR_COUNT = 2
MEASURED_PAIR_COUNT = 200
WORLD_SIZE = 4
HIDDEN_SIZE = 5120
LAYER_COUNT = 64
ARTIFACT_NAMES = (
    "peer_access_matrix.json",
    "ipc_roundtrip.jsonl",
    "microgate_rows.jsonl",
    "memory_summary.json",
    "cleanup.json",
)


def build_workload_schedule():
    groups = []
    for active_tokens in ACTIVE_TOKEN_GROUPS:
        def pair(pair_index):
            return {
                "pair_index": pair_index,
                "arm_order": (
                    ("baseline", "candidate")
                    if pair_index % 2 == 0
                    else ("candidate", "baseline")
                ),
            }

        groups.append({
            "active_tokens": active_tokens,
            "seed": 2026083000 + active_tokens,
            "warmups": tuple(
                pair(pair_index)
                for pair_index in range(WARMUP_PAIR_COUNT)
            ),
            "measurements": tuple(
                pair(pair_index)
                for pair_index in range(MEASURED_PAIR_COUNT)
            ),
        })
    return tuple(groups)


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--dist-port", required=True, type=int)
    return parser


def _finite_nonnegative(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def validate_measurement_row(row):
    if not isinstance(row, dict):
        raise ValueError("measurement row must be an object")
    for field in (
        "baseline_cuda_ns",
        "candidate_cuda_ns",
        "baseline_host_submission_ns",
        "candidate_host_submission_ns",
        "cross_rank_max_abs_error",
        "cross_rank_max_rel_error",
        "baseline_max_abs_error",
        "baseline_max_rel_error",
    ):
        if field not in row or not _finite_nonnegative(row[field]):
            raise ValueError(f"{field} is invalid")
    if row.get("active_tokens") not in ACTIVE_TOKEN_GROUPS:
        raise ValueError("active_tokens is invalid")
    if (
        type(row.get("pair_index")) is not int
        or row["pair_index"] < 0
    ):
        raise ValueError("pair_index is invalid")
    if (
        type(row.get("rank")) is not int
        or row["rank"] not in range(WORLD_SIZE)
    ):
        raise ValueError("rank is invalid")
    if row.get("arm_order") not in (
        ["baseline", "candidate"],
        ["candidate", "baseline"],
    ):
        raise ValueError("arm_order is invalid")
    if type(row.get("timed_out")) is not bool:
        raise ValueError("timed_out is invalid")
    if type(row.get("device_status")) is not int:
        raise ValueError("device_status is invalid")
    return dict(row)


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
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


def _atomic_write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _relative_errors(actual, expected, torch):
    difference = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs().clamp_min(1e-12)
    return (
        float(difference.max().item()),
        float((difference / denominator).max().item()),
    )


def _broadcast_inputs(active_tokens, seed, rank, device, torch, dist):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = torch.empty(
        (active_tokens, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
    )
    residual = torch.empty_like(x)
    if rank == 0:
        x.normal_(generator=generator)
        residual.normal_(generator=generator)
    dist.broadcast(x, src=0)
    dist.broadcast(residual, src=0)

    transfer = torch.empty(
        (HIDDEN_SIZE, HIDDEN_SIZE),
        dtype=torch.float32,
        device=device,
    )
    local_weight = None
    for owner_rank in range(WORLD_SIZE):
        if rank == 0:
            transfer.normal_(generator=generator)
            transfer.mul_(0.01)
        dist.broadcast(transfer, src=0)
        if rank == owner_rank:
            local_weight = transfer.clone()
    return x, local_weight, residual


def _run_arm(
    arm,
    *,
    x,
    local_weight,
    residual,
    group,
    generation,
    torch,
    dist,
):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    submitted_at = time.perf_counter_ns()
    start.record()
    partial = torch.nn.functional.linear(x.float(), local_weight)
    if arm == "baseline":
        dist.all_reduce(partial)
        output = partial.to(torch.bfloat16) + residual
    else:
        output = group.reduce_add_residual(
            layer_index=0,
            generation=generation,
            local_partial=partial,
            residual=residual,
        )
    end.record()
    return {
        "arm": arm,
        "output": output,
        "start": start,
        "end": end,
        "host_submission_ns": time.perf_counter_ns() - submitted_at,
    }


def _peer_rows(rank, torch):
    return [
        {
            "source_rank": source_rank,
            "destination_rank": rank,
            "can_access": bool(
                torch.cuda.can_device_access_peer(rank, source_rank)
            ),
            "ipc_roundtrip": True,
        }
        for source_rank in range(WORLD_SIZE)
        if source_rank != rank
    ]


def run_worker(args):
    import torch
    import torch.distributed as dist

    from tinyvllm.engine.tp4_peer_reduction import (
        TP4PeerReductionGroup,
    )

    if args.world_size != WORLD_SIZE:
        raise ValueError("world_size must be 4")
    if args.rank not in range(WORLD_SIZE):
        raise ValueError("rank is invalid")
    if len(args.source_revision) != 40:
        raise ValueError("source_revision must be a full SHA")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda", args.rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{args.dist_port}",
        rank=args.rank,
        world_size=args.world_size,
    )
    group = None
    local_rows = []
    local_topology = []
    local_memory = {}
    cleanup_row = {
        "rank": args.rank,
        "peer_group_closed": False,
        "timed_out": False,
    }
    try:
        baseline_allocated = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        group = TP4PeerReductionGroup.create(
            rank=args.rank,
            world_size=args.world_size,
            device=device,
            layer_count=LAYER_COUNT,
            max_active_tokens=max(ACTIVE_TOKEN_GROUPS),
            hidden_size=HIDDEN_SIZE,
            distributed=dist,
        )
        after_group_allocated = torch.cuda.memory_allocated(device)
        local_topology = _peer_rows(args.rank, torch)
        generation = 1
        for workload in build_workload_schedule():
            x, local_weight, residual = _broadcast_inputs(
                workload["active_tokens"],
                workload["seed"],
                args.rank,
                device,
                torch,
                dist,
            )
            for pair in workload["warmups"]:
                for arm in pair["arm_order"]:
                    _run_arm(
                        arm,
                        x=x,
                        local_weight=local_weight,
                        residual=residual,
                        group=group,
                        generation=generation,
                        torch=torch,
                        dist=dist,
                    )
                    if arm == "candidate":
                        generation += 1
                torch.cuda.synchronize(device)
                group.check_status()

            for pair in workload["measurements"]:
                results = {}
                for arm in pair["arm_order"]:
                    result = _run_arm(
                        arm,
                        x=x,
                        local_weight=local_weight,
                        residual=residual,
                        group=group,
                        generation=generation,
                        torch=torch,
                        dist=dist,
                    )
                    results[arm] = result
                    if arm == "candidate":
                        generation += 1
                torch.cuda.synchronize(device)
                group.check_status()
                baseline = results["baseline"]["output"]
                candidate = results["candidate"]["output"]
                peers = [torch.empty_like(candidate) for _ in range(4)]
                dist.all_gather(peers, candidate)
                cross_abs = 0.0
                cross_rel = 0.0
                for peer in peers:
                    absolute, relative = _relative_errors(
                        candidate,
                        peer,
                        torch,
                    )
                    cross_abs = max(cross_abs, absolute)
                    cross_rel = max(cross_rel, relative)
                baseline_abs, baseline_rel = _relative_errors(
                    candidate,
                    baseline,
                    torch,
                )
                row = {
                    "attempt": args.attempt,
                    "source_revision": args.source_revision,
                    "active_tokens": workload["active_tokens"],
                    "pair_index": pair["pair_index"],
                    "rank": args.rank,
                    "arm_order": list(pair["arm_order"]),
                    "baseline_cuda_ns": int(
                        results["baseline"]["start"].elapsed_time(
                            results["baseline"]["end"]
                        )
                        * 1_000_000
                    ),
                    "candidate_cuda_ns": int(
                        results["candidate"]["start"].elapsed_time(
                            results["candidate"]["end"]
                        )
                        * 1_000_000
                    ),
                    "baseline_host_submission_ns": results[
                        "baseline"
                    ]["host_submission_ns"],
                    "candidate_host_submission_ns": results[
                        "candidate"
                    ]["host_submission_ns"],
                    "cross_rank_max_abs_error": cross_abs,
                    "cross_rank_max_rel_error": cross_rel,
                    "baseline_max_abs_error": baseline_abs,
                    "baseline_max_rel_error": baseline_rel,
                    "timed_out": False,
                    "device_status": 0,
                }
                local_rows.append(validate_measurement_row(row))
        local_memory = {
            "rank": args.rank,
            "baseline_allocated_bytes": baseline_allocated,
            "after_group_allocated_bytes": after_group_allocated,
            "allocated_delta_bytes": (
                after_group_allocated - baseline_allocated
            ),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                device
            ),
        }
        dist.barrier()
    except RuntimeError as error:
        if "timed out" in str(error):
            cleanup_row["timed_out"] = True
        raise
    finally:
        if group is not None:
            group.close()
            cleanup_row["peer_group_closed"] = True

        gathered_rows = [None] * WORLD_SIZE if args.rank == 0 else None
        gathered_topology = (
            [None] * WORLD_SIZE if args.rank == 0 else None
        )
        gathered_memory = (
            [None] * WORLD_SIZE if args.rank == 0 else None
        )
        gathered_cleanup = (
            [None] * WORLD_SIZE if args.rank == 0 else None
        )
        if dist.is_initialized():
            dist.gather_object(local_rows, gathered_rows, dst=0)
            dist.gather_object(
                local_topology,
                gathered_topology,
                dst=0,
            )
            dist.gather_object(
                local_memory,
                gathered_memory,
                dst=0,
            )
            dist.gather_object(
                cleanup_row,
                gathered_cleanup,
                dst=0,
            )
            if args.rank == 0:
                topology_rows = [
                    row
                    for rank_rows in gathered_topology
                    for row in rank_rows
                ]
                microgate_rows = [
                    row
                    for rank_rows in gathered_rows
                    for row in rank_rows
                ]
                _atomic_write_json(
                    output_dir / ARTIFACT_NAMES[0],
                    {
                        "attempt": args.attempt,
                        "source_revision": args.source_revision,
                        "world_size": WORLD_SIZE,
                        "rows": topology_rows,
                    },
                )
                _atomic_write_jsonl(
                    output_dir / ARTIFACT_NAMES[1],
                    topology_rows,
                )
                _atomic_write_jsonl(
                    output_dir / ARTIFACT_NAMES[2],
                    microgate_rows,
                )
                _atomic_write_json(
                    output_dir / ARTIFACT_NAMES[3],
                    {
                        "maximum_allocated_delta_bytes": max(
                            row["allocated_delta_bytes"]
                            for row in gathered_memory
                        ),
                        "rank_rows": gathered_memory,
                    },
                )
                _atomic_write_json(
                    output_dir / ARTIFACT_NAMES[4],
                    {
                        "classification": (
                            "CLEAN"
                            if all(
                                row["peer_group_closed"]
                                and not row["timed_out"]
                                for row in gathered_cleanup
                            )
                            else "DIRTY"
                        ),
                        "rank_rows": gathered_cleanup,
                    },
                )
            dist.destroy_process_group()


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    run_worker(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
