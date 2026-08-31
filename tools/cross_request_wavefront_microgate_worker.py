#!/usr/bin/env python3
"""Run the isolated four-GPU cross-request wavefront microgate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

from tools.cross_request_wavefront_overlap import (
    ACTIVE_TOKEN_GROUPS,
    MEASURED_PAIR_COUNT,
    WARMUP_PAIR_COUNT,
    WORLD_SIZE,
    build_balanced_cohorts,
    cohort_digest,
    interval_overlap_ns,
    interval_union_ns,
)


LOCAL_INPUT_SIZE = 1536
HIDDEN_SIZE = 5120
ARTIFACT_NAMES = (
    "microgate_rows.jsonl",
    "memory_summary.json",
    "cleanup.json",
    "runtime_capabilities.json",
)
COLLECTIVE_ORDER = ("cohort:0", "cohort:1")
COLLECTIVE_ORDER_DIGEST = hashlib.sha256(
    json.dumps(
        COLLECTIVE_ORDER,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


@dataclass
class WavefrontBuffers:
    compute_streams: tuple
    communication_stream: object
    origin: object
    local_started: tuple
    local_ready: tuple
    communication_started: tuple
    communication_ready: tuple
    dependent_started: tuple
    dependent_ready: tuple
    completed: object
    baseline_started: object
    baseline_completed: object
    local_partials: object
    cast_buffer: object
    output: object
    baseline_partial: object
    baseline_cast: object
    baseline_output: object

    @classmethod
    def create(cls, torch, device, active_tokens):
        return cls(
            compute_streams=(
                torch.cuda.Stream(device=device),
                torch.cuda.Stream(device=device),
            ),
            communication_stream=torch.cuda.Stream(device=device),
            origin=torch.cuda.Event(enable_timing=True),
            local_started=tuple(
                torch.cuda.Event(enable_timing=True) for _ in range(2)
            ),
            local_ready=tuple(
                torch.cuda.Event(enable_timing=True) for _ in range(2)
            ),
            communication_started=tuple(
                torch.cuda.Event(enable_timing=True) for _ in range(2)
            ),
            communication_ready=tuple(
                torch.cuda.Event(enable_timing=True) for _ in range(2)
            ),
            dependent_started=tuple(
                torch.cuda.Event(enable_timing=True) for _ in range(2)
            ),
            dependent_ready=tuple(
                torch.cuda.Event(enable_timing=True) for _ in range(2)
            ),
            completed=torch.cuda.Event(enable_timing=True),
            baseline_started=torch.cuda.Event(enable_timing=True),
            baseline_completed=torch.cuda.Event(enable_timing=True),
            local_partials=torch.empty(
                (active_tokens, HIDDEN_SIZE),
                dtype=torch.float32,
                device=device,
            ),
            cast_buffer=torch.empty(
                (active_tokens, HIDDEN_SIZE),
                dtype=torch.bfloat16,
                device=device,
            ),
            output=torch.empty(
                (active_tokens, HIDDEN_SIZE),
                dtype=torch.bfloat16,
                device=device,
            ),
            baseline_partial=torch.empty(
                (active_tokens, HIDDEN_SIZE),
                dtype=torch.float32,
                device=device,
            ),
            baseline_cast=torch.empty(
                (active_tokens, HIDDEN_SIZE),
                dtype=torch.bfloat16,
                device=device,
            ),
            baseline_output=torch.empty(
                (active_tokens, HIDDEN_SIZE),
                dtype=torch.bfloat16,
                device=device,
            ),
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

        groups.append(
            {
                "active_tokens": active_tokens,
                "seed": 2026083100 + active_tokens,
                "warmups": tuple(
                    pair(pair_index)
                    for pair_index in range(WARMUP_PAIR_COUNT)
                ),
                "measurements": tuple(
                    pair(pair_index)
                    for pair_index in range(MEASURED_PAIR_COUNT)
                ),
            }
        )
    return tuple(groups)


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
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


def _is_hex(value, length):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_measurement_row(row):
    if not isinstance(row, dict):
        raise ValueError("measurement row must be an object")
    for field in (
        "baseline_cuda_ns",
        "candidate_cuda_ns",
        "baseline_host_submission_ns",
        "candidate_host_submission_ns",
        "candidate_communication_union_ns",
        "candidate_realized_overlap_ns",
        "cross_rank_max_abs_error",
        "cross_rank_max_rel_error",
        "baseline_max_abs_error",
        "baseline_max_rel_error",
    ):
        if field not in row or not _finite_nonnegative(row[field]):
            raise ValueError(f"{field} is invalid")
    if not isinstance(row.get("attempt"), str) or not row["attempt"]:
        raise ValueError("attempt is invalid")
    if not _is_hex(row.get("source_revision"), 40):
        raise ValueError("source_revision is invalid")
    if not _is_hex(row.get("source_tree_sha256"), 64):
        raise ValueError("source_tree_sha256 is invalid")
    if (
        type(row.get("active_tokens")) is not int
        or row["active_tokens"] not in ACTIVE_TOKEN_GROUPS
    ):
        raise ValueError("active_tokens is invalid")
    if (
        type(row.get("pair_index")) is not int
        or row["pair_index"] not in range(MEASURED_PAIR_COUNT)
    ):
        raise ValueError("pair_index is invalid")
    if (
        type(row.get("rank")) is not int
        or row["rank"] not in range(WORLD_SIZE)
    ):
        raise ValueError("rank is invalid")
    expected_order = (
        ["baseline", "candidate"]
        if row["pair_index"] % 2 == 0
        else ["candidate", "baseline"]
    )
    if row.get("arm_order") != expected_order:
        raise ValueError("arm_order is invalid")
    for field in (
        "cohort_digest",
        "collective_order_digest",
    ):
        if not _is_hex(row.get(field), 64):
            raise ValueError(f"{field} is invalid")
    for field in ("nan_count", "inf_count"):
        if type(row.get(field)) is not int or row[field] < 0:
            raise ValueError(f"{field} is invalid")
    if type(row.get("timed_out")) is not bool:
        raise ValueError("timed_out is invalid")
    if row["baseline_cuda_ns"] <= 0:
        raise ValueError("baseline_cuda_ns is invalid")
    if row["baseline_host_submission_ns"] <= 0:
        raise ValueError("baseline_host_submission_ns is invalid")
    if row["candidate_communication_union_ns"] <= 0:
        raise ValueError("candidate_communication_union_ns is invalid")
    if (
        row["candidate_realized_overlap_ns"]
        > row["candidate_communication_union_ns"]
    ):
        raise ValueError("candidate_realized_overlap_ns is invalid")
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
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
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


def _run_baseline(
    *,
    x_fp32,
    weight_t,
    residual,
    buffers,
    torch,
    dist,
):
    caller = torch.cuda.current_stream()
    submitted_at = time.perf_counter_ns()
    buffers.baseline_started.record(caller)
    torch.mm(
        x_fp32,
        weight_t,
        out=buffers.baseline_partial,
    )
    dist.all_reduce(buffers.baseline_partial)
    buffers.baseline_cast.copy_(buffers.baseline_partial)
    buffers.baseline_output.copy_(buffers.baseline_cast).add_(residual)
    buffers.baseline_completed.record(caller)
    return {
        "output": buffers.baseline_output,
        "start": buffers.baseline_started,
        "end": buffers.baseline_completed,
        "host_submission_ns": time.perf_counter_ns() - submitted_at,
    }


def _run_candidate(
    *,
    x_cohorts,
    weight_t,
    residual_cohorts,
    partials,
    casts,
    outputs,
    buffers,
    torch,
    dist,
):
    caller = torch.cuda.current_stream()
    submitted_at = time.perf_counter_ns()
    buffers.origin.record(caller)

    with torch.cuda.stream(buffers.compute_streams[0]):
        buffers.compute_streams[0].wait_event(buffers.origin)
        buffers.local_started[0].record()
        torch.mm(x_cohorts[0], weight_t, out=partials[0])
        buffers.local_ready[0].record()

    with torch.cuda.stream(buffers.communication_stream):
        buffers.communication_stream.wait_event(buffers.local_ready[0])
        buffers.communication_started[0].record()
        work0 = dist.all_reduce(partials[0], async_op=True)
        work0.wait()
        buffers.communication_ready[0].record()

    with torch.cuda.stream(buffers.compute_streams[1]):
        buffers.compute_streams[1].wait_event(buffers.origin)
        buffers.local_started[1].record()
        torch.mm(x_cohorts[1], weight_t, out=partials[1])
        buffers.local_ready[1].record()

    with torch.cuda.stream(buffers.compute_streams[0]):
        buffers.compute_streams[0].wait_event(
            buffers.communication_ready[0]
        )
        buffers.dependent_started[0].record()
        casts[0].copy_(partials[0])
        outputs[0].copy_(casts[0]).add_(residual_cohorts[0])
        buffers.dependent_ready[0].record()

    with torch.cuda.stream(buffers.communication_stream):
        buffers.communication_stream.wait_event(buffers.local_ready[1])
        buffers.communication_started[1].record()
        work1 = dist.all_reduce(partials[1], async_op=True)
        work1.wait()
        buffers.communication_ready[1].record()

    with torch.cuda.stream(buffers.compute_streams[1]):
        buffers.compute_streams[1].wait_event(
            buffers.communication_ready[1]
        )
        buffers.dependent_started[1].record()
        casts[1].copy_(partials[1])
        outputs[1].copy_(casts[1]).add_(residual_cohorts[1])
        buffers.dependent_ready[1].record()

    caller.wait_event(buffers.dependent_ready[0])
    caller.wait_event(buffers.dependent_ready[1])
    buffers.completed.record(caller)
    return {
        "output": buffers.output,
        "start": buffers.origin,
        "end": buffers.completed,
        "host_submission_ns": time.perf_counter_ns() - submitted_at,
    }


def _event_interval_ns(origin, started, completed):
    return (
        int(origin.elapsed_time(started) * 1_000_000),
        int(origin.elapsed_time(completed) * 1_000_000),
    )


def _candidate_intervals(buffers):
    communication = tuple(
        _event_interval_ns(
            buffers.origin,
            buffers.communication_started[index],
            buffers.communication_ready[index],
        )
        for index in range(2)
    )
    computation = tuple(
        [
            _event_interval_ns(
                buffers.origin,
                buffers.local_started[index],
                buffers.local_ready[index],
            )
            for index in range(2)
        ]
        + [
            _event_interval_ns(
                buffers.origin,
                buffers.dependent_started[index],
                buffers.dependent_ready[index],
            )
            for index in range(2)
        ]
    )
    return {
        "communication_union_ns": interval_union_ns(communication),
        "realized_overlap_ns": interval_overlap_ns(
            communication,
            computation,
        ),
    }


def _output_digest(output, torch):
    payload = (
        output.detach()
        .contiguous()
        .view(torch.uint8)
        .cpu()
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _build_inputs(active_tokens, seed, rank, device, torch, dist):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + rank)
    x = torch.empty(
        (active_tokens, LOCAL_INPUT_SIZE),
        dtype=torch.bfloat16,
        device=device,
    )
    x.normal_(generator=generator)
    x_fp32 = x.float()
    weight_t = torch.empty(
        (LOCAL_INPUT_SIZE, HIDDEN_SIZE),
        dtype=torch.float32,
        device=device,
    )
    weight_t.normal_(generator=generator).mul_(0.01)

    residual = torch.empty(
        (active_tokens, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
    )
    if rank == 0:
        residual.normal_(generator=generator)
    dist.broadcast(residual, src=0)
    return x_fp32, weight_t.contiguous(), residual


def _cohort_views(active_tokens, x_fp32, residual, buffers):
    cohorts = build_balanced_cohorts(active_tokens)
    slices = tuple(
        slice(
            cohort["request_indices"][0],
            cohort["request_indices"][-1] + 1,
        )
        for cohort in cohorts
    )
    return {
        "cohorts": cohorts,
        "x": tuple(x_fp32[index] for index in slices),
        "residual": tuple(residual[index] for index in slices),
        "partials": tuple(
            buffers.local_partials[index] for index in slices
        ),
        "casts": tuple(buffers.cast_buffer[index] for index in slices),
        "outputs": tuple(buffers.output[index] for index in slices),
    }


def _runtime_capability_row(rank, device, torch, dist):
    properties = torch.cuda.get_device_properties(device)
    return {
        "rank": rank,
        "device_index": rank,
        "device_name": properties.name,
        "device_uuid": str(getattr(properties, "uuid", "")),
        "compute_capability": [
            int(properties.major),
            int(properties.minor),
        ],
        "cuda_version": str(torch.version.cuda),
        "torch_version": str(torch.__version__),
        "nccl_available": bool(dist.is_nccl_available()),
        "world_size": WORLD_SIZE,
        "local_input_size": LOCAL_INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "input_dtype": "bfloat16",
        "accumulation_dtype": "float32",
        "output_dtype": "bfloat16",
    }


def _wait_for_cleanup_rows(output_dir):
    paths = [
        output_dir / f".cleanup-rank-{rank}.json"
        for rank in range(WORLD_SIZE)
    ]
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if all(path.is_file() for path in paths):
            rows = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in paths
            ]
            for path in paths:
                path.unlink()
            return rows
        time.sleep(0.05)
    return []


def run_worker(args):
    import torch
    import torch.distributed as dist

    if args.world_size != WORLD_SIZE:
        raise ValueError("world_size must be 4")
    if args.rank not in range(WORLD_SIZE):
        raise ValueError("rank is invalid")
    if not _is_hex(args.source_revision, 40):
        raise ValueError("source_revision must be a full SHA")
    if not _is_hex(args.source_tree_sha256, 64):
        raise ValueError("source_tree_sha256 must be a SHA-256")

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

    local_rows = []
    local_memory_rows = []
    local_capability = {}
    gathered_rows = None
    gathered_memory = None
    gathered_capabilities = None
    cleanup_row = {
        "rank": args.rank,
        "streams_released": False,
        "events_released": False,
        "timed_out": False,
        "process_group_destroyed": False,
    }
    try:
        local_capability = _runtime_capability_row(
            args.rank,
            device,
            torch,
            dist,
        )
        for workload in build_workload_schedule():
            active_tokens = workload["active_tokens"]
            x_fp32, weight_t, residual = _build_inputs(
                active_tokens,
                workload["seed"],
                args.rank,
                device,
                torch,
                dist,
            )
            before_allocated = torch.cuda.memory_allocated(device)
            before_reserved = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
            buffers = WavefrontBuffers.create(
                torch,
                device,
                active_tokens,
            )
            views = _cohort_views(
                active_tokens,
                x_fp32,
                residual,
                buffers,
            )
            peer_outputs = [
                torch.empty_like(buffers.output)
                for _ in range(WORLD_SIZE)
            ]
            after_allocated = torch.cuda.memory_allocated(device)
            after_reserved = torch.cuda.memory_reserved(device)

            for pair in workload["warmups"]:
                results = {}
                for arm in pair["arm_order"]:
                    if arm == "baseline":
                        results[arm] = _run_baseline(
                            x_fp32=x_fp32,
                            weight_t=weight_t,
                            residual=residual,
                            buffers=buffers,
                            torch=torch,
                            dist=dist,
                        )
                    else:
                        results[arm] = _run_candidate(
                            x_cohorts=views["x"],
                            weight_t=weight_t,
                            residual_cohorts=views["residual"],
                            partials=views["partials"],
                            casts=views["casts"],
                            outputs=views["outputs"],
                            buffers=buffers,
                            torch=torch,
                            dist=dist,
                        )
                results["baseline"]["end"].synchronize()
                results["candidate"]["end"].synchronize()

            for pair in workload["measurements"]:
                results = {}
                for arm in pair["arm_order"]:
                    if arm == "baseline":
                        results[arm] = _run_baseline(
                            x_fp32=x_fp32,
                            weight_t=weight_t,
                            residual=residual,
                            buffers=buffers,
                            torch=torch,
                            dist=dist,
                        )
                    else:
                        results[arm] = _run_candidate(
                            x_cohorts=views["x"],
                            weight_t=weight_t,
                            residual_cohorts=views["residual"],
                            partials=views["partials"],
                            casts=views["casts"],
                            outputs=views["outputs"],
                            buffers=buffers,
                            torch=torch,
                            dist=dist,
                        )
                results["baseline"]["end"].synchronize()
                results["candidate"]["end"].synchronize()

                baseline = results["baseline"]["output"]
                candidate = results["candidate"]["output"]
                dist.all_gather(peer_outputs, candidate)
                cross_abs = 0.0
                cross_rel = 0.0
                for peer in peer_outputs:
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
                digest = _output_digest(candidate, torch)
                rank_digests = [None] * WORLD_SIZE
                dist.all_gather_object(rank_digests, digest)
                intervals = _candidate_intervals(buffers)
                row = {
                    "attempt": args.attempt,
                    "source_revision": args.source_revision,
                    "source_tree_sha256": args.source_tree_sha256,
                    "active_tokens": active_tokens,
                    "pair_index": pair["pair_index"],
                    "rank": args.rank,
                    "arm_order": list(pair["arm_order"]),
                    "cohort_digest": cohort_digest(views["cohorts"]),
                    "collective_order_digest": COLLECTIVE_ORDER_DIGEST,
                    "candidate_output_digest": digest,
                    "rank_output_digests": rank_digests,
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
                    "candidate_communication_union_ns": intervals[
                        "communication_union_ns"
                    ],
                    "candidate_realized_overlap_ns": intervals[
                        "realized_overlap_ns"
                    ],
                    "cross_rank_max_abs_error": cross_abs,
                    "cross_rank_max_rel_error": cross_rel,
                    "baseline_max_abs_error": baseline_abs,
                    "baseline_max_rel_error": baseline_rel,
                    "nan_count": int(
                        torch.count_nonzero(torch.isnan(candidate)).item()
                    ),
                    "inf_count": int(
                        torch.count_nonzero(torch.isinf(candidate)).item()
                    ),
                    "timed_out": False,
                }
                local_rows.append(validate_measurement_row(row))

            local_memory_rows.append(
                {
                    "rank": args.rank,
                    "active_tokens": active_tokens,
                    "before_allocated_bytes": before_allocated,
                    "after_allocated_bytes": after_allocated,
                    "allocated_delta_bytes": (
                        after_allocated - before_allocated
                    ),
                    "before_reserved_bytes": before_reserved,
                    "after_reserved_bytes": after_reserved,
                    "reserved_delta_bytes": (
                        after_reserved - before_reserved
                    ),
                    "peak_allocated_delta_bytes": max(
                        0,
                        torch.cuda.max_memory_allocated(device)
                        - before_allocated,
                    ),
                }
            )
            del peer_outputs
            del views
            del buffers
            del residual
            del weight_t
            del x_fp32
            cleanup_row["streams_released"] = True
            cleanup_row["events_released"] = True
            torch.cuda.empty_cache()

        gathered_rows = [None] * WORLD_SIZE if args.rank == 0 else None
        gathered_memory = [None] * WORLD_SIZE if args.rank == 0 else None
        gathered_capabilities = (
            [None] * WORLD_SIZE if args.rank == 0 else None
        )
        dist.gather_object(local_rows, gathered_rows, dst=0)
        dist.gather_object(
            local_memory_rows,
            gathered_memory,
            dst=0,
        )
        dist.gather_object(
            local_capability,
            gathered_capabilities,
            dst=0,
        )
        dist.barrier()
    except RuntimeError as error:
        if "timed out" in str(error).lower():
            cleanup_row["timed_out"] = True
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
            cleanup_row["process_group_destroyed"] = True
        _atomic_write_json(
            output_dir / f".cleanup-rank-{args.rank}.json",
            cleanup_row,
        )

    if args.rank == 0:
        cleanup_rows = _wait_for_cleanup_rows(output_dir)
        flattened_rows = [
            row for rank_rows in gathered_rows for row in rank_rows
        ]
        flattened_memory = [
            row for rank_rows in gathered_memory for row in rank_rows
        ]
        _atomic_write_jsonl(
            output_dir / ARTIFACT_NAMES[0],
            flattened_rows,
        )
        _atomic_write_json(
            output_dir / ARTIFACT_NAMES[1],
            {
                "maximum_allocated_delta_bytes": max(
                    row["peak_allocated_delta_bytes"]
                    for row in flattened_memory
                ),
                "maximum_reserved_delta_bytes": max(
                    row["reserved_delta_bytes"]
                    for row in flattened_memory
                ),
                "rank_shape_rows": flattened_memory,
            },
        )
        _atomic_write_json(
            output_dir / ARTIFACT_NAMES[2],
            {
                "classification": (
                    "CLEAN"
                    if len(cleanup_rows) == WORLD_SIZE
                    and all(
                        row["streams_released"]
                        and row["events_released"]
                        and not row["timed_out"]
                        and row["process_group_destroyed"]
                        for row in cleanup_rows
                    )
                    else "DIRTY"
                ),
                "rank_rows": cleanup_rows,
            },
        )
        _atomic_write_json(
            output_dir / ARTIFACT_NAMES[3],
            {
                "schema_version": (
                    "cross-request-wavefront-runtime-capabilities.v1"
                ),
                "attempt": args.attempt,
                "source_revision": args.source_revision,
                "source_tree_sha256": args.source_tree_sha256,
                "rank_rows": gathered_capabilities,
            },
        )


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    run_worker(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
