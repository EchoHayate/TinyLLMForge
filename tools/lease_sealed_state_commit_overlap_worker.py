#!/usr/bin/env python3
"""Run the model-neutral TP4 lease-sealed state-commit overlap microgate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

if __package__:
    from tools.lease_sealed_state_commit_overlap import (
        ACTIVE_TOKEN_GROUPS,
        HIDDEN_SIZE,
        LINEAR_LAYER_COUNT,
        MEASURED_PAIR_COUNT,
        STATE_BYTES_PER_TOKEN_PER_LAYER,
        WARMUP_PAIR_COUNT,
        WORLD_SIZE,
        interval_intersection_ns,
        validate_measurement_row,
    )
else:
    from lease_sealed_state_commit_overlap import (
        ACTIVE_TOKEN_GROUPS,
        HIDDEN_SIZE,
        LINEAR_LAYER_COUNT,
        MEASURED_PAIR_COUNT,
        STATE_BYTES_PER_TOKEN_PER_LAYER,
        WARMUP_PAIR_COUNT,
        WORLD_SIZE,
        interval_intersection_ns,
        validate_measurement_row,
    )


SEEDS = {
    1: 2026090701,
    4: 2026090704,
    8: 2026090708,
}


@dataclass
class OverlapBuffers:
    communication_stream: object
    side_effect_stream: object
    producer_ready_event: object
    consumer_ready_event: object
    side_effect_ready_event: object
    baseline_started: object
    baseline_completed: object
    candidate_started: object
    candidate_completed: object
    allreduce_started: object
    allreduce_completed: object
    state_copy_started: object
    state_copy_completed: object
    local_result: object
    baseline_result: object
    candidate_result: object
    baseline_output: object
    candidate_output: object
    side_effect_payload: object
    baseline_shadow: object
    candidate_shadow: object

    @classmethod
    def create(cls, torch, device, active_tokens):
        def event():
            return torch.cuda.Event(enable_timing=True)

        state_elements = STATE_BYTES_PER_TOKEN_PER_LAYER // 2
        return cls(
            communication_stream=torch.cuda.Stream(device=device),
            side_effect_stream=torch.cuda.Stream(device=device),
            producer_ready_event=event(),
            consumer_ready_event=event(),
            side_effect_ready_event=event(),
            baseline_started=event(),
            baseline_completed=event(),
            candidate_started=event(),
            candidate_completed=event(),
            allreduce_started=event(),
            allreduce_completed=event(),
            state_copy_started=event(),
            state_copy_completed=event(),
            local_result=torch.empty(
                (active_tokens, HIDDEN_SIZE),
                dtype=torch.float32,
                device=device,
            ),
            baseline_result=torch.empty(
                (active_tokens, HIDDEN_SIZE),
                dtype=torch.float32,
                device=device,
            ),
            candidate_result=torch.empty(
                (active_tokens, HIDDEN_SIZE),
                dtype=torch.float32,
                device=device,
            ),
            baseline_output=torch.empty(
                (active_tokens, HIDDEN_SIZE),
                dtype=torch.bfloat16,
                device=device,
            ),
            candidate_output=torch.empty(
                (active_tokens, HIDDEN_SIZE),
                dtype=torch.bfloat16,
                device=device,
            ),
            side_effect_payload=torch.empty(
                (active_tokens, state_elements),
                dtype=torch.bfloat16,
                device=device,
            ),
            baseline_shadow=torch.empty(
                (active_tokens, state_elements),
                dtype=torch.bfloat16,
                device=device,
            ),
            candidate_shadow=torch.empty(
                (active_tokens, state_elements),
                dtype=torch.bfloat16,
                device=device,
            ),
        )


def build_workload_schedule():
    def pair(pair_index):
        return {
            "pair_index": pair_index,
            "arm_order": (
                ("baseline", "candidate")
                if pair_index % 2 == 0
                else ("candidate", "baseline")
            ),
        }

    return tuple(
        {
            "active_tokens": active_tokens,
            "seed": SEEDS[active_tokens],
            "warmups": tuple(
                pair(pair_index) for pair_index in range(WARMUP_PAIR_COUNT)
            ),
            "measurements": tuple(
                pair(pair_index) for pair_index in range(MEASURED_PAIR_COUNT)
            ),
        }
        for active_tokens in ACTIVE_TOKEN_GROUPS
    )


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


def _is_hex(value, length):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


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


def _timed_allreduce(tensor, buffers, dist):
    buffers.allreduce_started.record(buffers.communication_stream)
    work = dist.all_reduce(tensor, async_op=True)
    buffers.allreduce_completed.record(buffers.communication_stream)
    return work


def _timed_state_copy(payload, buffers):
    buffers.state_copy_started.record(buffers.side_effect_stream)
    buffers.candidate_shadow.copy_(payload)
    buffers.state_copy_completed.record(buffers.side_effect_stream)


def build_overlap_runtime(*, buffers, torch, dist):
    from tinyvllm.engine.collective_side_effect_overlap import (
        LeaseSealedCollectiveSideEffect,
        OverlapResources,
    )

    return LeaseSealedCollectiveSideEffect(
        resources=OverlapResources(
            communication_stream=buffers.communication_stream,
            side_effect_stream=buffers.side_effect_stream,
            producer_ready_event=buffers.producer_ready_event,
            consumer_ready_event=buffers.consumer_ready_event,
            side_effect_ready_event=buffers.side_effect_ready_event,
        ),
        current_stream=lambda tensor: torch.cuda.current_stream(tensor.device),
        stream_context=torch.cuda.stream,
        collective=lambda tensor: _timed_allreduce(
            tensor,
            buffers,
            dist,
        ),
    )


def _run_baseline(*, buffers, torch, dist):
    stream = torch.cuda.current_stream(buffers.local_result.device)
    submitted = time.perf_counter_ns()
    buffers.baseline_started.record(stream)
    buffers.baseline_result.copy_(buffers.local_result)
    dist.all_reduce(buffers.baseline_result)
    buffers.baseline_output.copy_(buffers.baseline_result)
    buffers.baseline_shadow.copy_(buffers.side_effect_payload)
    buffers.baseline_completed.record(stream)
    return {
        "reduced_result": buffers.baseline_result,
        "final_output": buffers.baseline_output,
        "shadow": buffers.baseline_shadow,
        "started": buffers.baseline_started,
        "completed": buffers.baseline_completed,
        "host_submission_ns": time.perf_counter_ns() - submitted,
    }


def _run_candidate(*, buffers, runtime, torch, commit_identity):
    stream = torch.cuda.current_stream(buffers.local_result.device)
    submitted = time.perf_counter_ns()
    buffers.candidate_started.record(stream)
    buffers.candidate_result.copy_(buffers.local_result)
    ticket = runtime.launch(
        local_result=buffers.candidate_result,
        side_effect_payload=buffers.side_effect_payload,
        materialize_side_effect=lambda payload: _timed_state_copy(
            payload,
            buffers,
        ),
        commit_identity=commit_identity,
    )
    result = runtime.join(ticket)
    buffers.candidate_output.copy_(result)
    runtime.seal(ticket, commit_identity)
    runtime.publish(ticket, commit_identity, lambda: None)
    buffers.candidate_completed.record(stream)
    return {
        "reduced_result": result,
        "final_output": buffers.candidate_output,
        "shadow": buffers.candidate_shadow,
        "started": buffers.candidate_started,
        "completed": buffers.candidate_completed,
        "host_submission_ns": time.perf_counter_ns() - submitted,
    }


def _event_interval_ns(origin, started, completed):
    return [
        int(origin.elapsed_time(started) * 1_000_000),
        int(origin.elapsed_time(completed) * 1_000_000),
    ]


def _tensor_digest(tensor, torch):
    payload = tensor.detach().contiguous().view(torch.uint8)
    return hashlib.sha256(payload.cpu().numpy().tobytes()).hexdigest()


def _initialize_buffers(buffers, seed, rank, torch, dist):
    local_generator = torch.Generator(device=buffers.local_result.device)
    local_generator.manual_seed(seed + rank)
    buffers.local_result.normal_(generator=local_generator)
    if rank == 0:
        payload_generator = torch.Generator(
            device=buffers.side_effect_payload.device
        )
        payload_generator.manual_seed(seed)
        buffers.side_effect_payload.normal_(generator=payload_generator)
    dist.broadcast(buffers.side_effect_payload, src=0)
    buffers.baseline_shadow.zero_()
    buffers.candidate_shadow.zero_()


def _run_lifecycle_probe(
    *,
    buffers,
    runtime,
    torch,
    commit_identity,
):
    stream = torch.cuda.current_stream(buffers.local_result.device)
    buffers.baseline_shadow.zero_()
    old_digest = _tensor_digest(buffers.baseline_shadow, torch)
    buffers.candidate_result.copy_(buffers.local_result)
    ticket = runtime.launch(
        local_result=buffers.candidate_result,
        side_effect_payload=buffers.side_effect_payload,
        materialize_side_effect=lambda payload: buffers.candidate_shadow.copy_(
            payload
        ),
        commit_identity=commit_identity,
    )
    runtime.join(ticket)
    stream.synchronize()
    active_preserved = (
        _tensor_digest(buffers.baseline_shadow, torch) == old_digest
    )
    runtime.seal(ticket, commit_identity)
    runtime.publish(
        ticket,
        commit_identity,
        lambda: buffers.baseline_shadow.copy_(buffers.candidate_shadow),
    )
    stream.synchronize()
    published_exact = bool(
        torch.equal(buffers.baseline_shadow, buffers.side_effect_payload)
    )

    buffers.baseline_shadow.zero_()
    abort_old_digest = _tensor_digest(buffers.baseline_shadow, torch)
    buffers.candidate_result.copy_(buffers.local_result)
    abort_ticket = runtime.launch(
        local_result=buffers.candidate_result,
        side_effect_payload=buffers.side_effect_payload,
        materialize_side_effect=lambda payload: buffers.candidate_shadow.copy_(
            payload
        ),
        commit_identity=f"{commit_identity}:abort",
    )
    runtime.abort(abort_ticket, lambda: buffers.candidate_shadow.zero_())
    abort_preserved = (
        _tensor_digest(buffers.baseline_shadow, torch) == abort_old_digest
    )
    return {
        "active_state_preserved_before_publish": active_preserved,
        "published_state_exact": published_exact,
        "abort_preserved_old_state": abort_preserved,
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
        "hidden_size": HIDDEN_SIZE,
        "collective_dtype": "float32",
        "output_dtype": "bfloat16",
        "state_dtype": "bfloat16",
    }


def _rank_paths(output_dir, rank):
    return {
        "measurements": output_dir / f"measurement_rows.rank-{rank}.jsonl",
        "memory": output_dir / f"memory.rank-{rank}.json",
        "lifecycle": output_dir / f"lifecycle.rank-{rank}.json",
        "cleanup": output_dir / f"cleanup.rank-{rank}.json",
        "capability": output_dir / f"capability.rank-{rank}.json",
    }


def _assert_rank_paths_fresh(paths):
    occupied = [str(path) for path in paths.values() if path.exists()]
    if occupied:
        raise ValueError(f"rank output paths are not fresh: {occupied}")


def _wait_for_rank_files(output_dir, filename, timeout_seconds=60.0):
    paths = [
        output_dir / filename.format(rank=rank)
        for rank in range(WORLD_SIZE)
    ]
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if all(path.is_file() for path in paths):
            return paths
        time.sleep(0.05)
    missing = [str(path) for path in paths if not path.is_file()]
    raise TimeoutError(f"rank artifacts were not sealed: {missing}")


def _merge_rank_artifacts(output_dir):
    measurement_paths = _wait_for_rank_files(
        output_dir,
        "measurement_rows.rank-{rank}.jsonl",
    )
    memory_paths = _wait_for_rank_files(
        output_dir,
        "memory.rank-{rank}.json",
    )
    lifecycle_paths = _wait_for_rank_files(
        output_dir,
        "lifecycle.rank-{rank}.json",
    )
    cleanup_paths = _wait_for_rank_files(
        output_dir,
        "cleanup.rank-{rank}.json",
    )
    capability_paths = _wait_for_rank_files(
        output_dir,
        "capability.rank-{rank}.json",
    )
    rows = []
    for path in measurement_paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    memory_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in memory_paths
    ]
    lifecycle_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in lifecycle_paths
    ]
    cleanup_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in cleanup_paths
    ]
    capability_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in capability_paths
    ]
    _atomic_write_jsonl(output_dir / "measurement_rows.jsonl", rows)
    _atomic_write_json(
        output_dir / "memory.json",
        {"rank_rows": memory_rows},
    )
    _atomic_write_json(
        output_dir / "lifecycle.json",
        {"rank_rows": lifecycle_rows},
    )
    _atomic_write_json(
        output_dir / "cleanup.json",
        {
            "classification": (
                "CLEAN"
                if len(cleanup_rows) == WORLD_SIZE
                and all(
                    row["process_group_destroyed"]
                    and row["streams_released"]
                    and row["events_released"]
                    and not row["timed_out"]
                    for row in cleanup_rows
                )
                else "DIRTY"
            ),
            "rank_rows": cleanup_rows,
        },
    )
    _atomic_write_json(
        output_dir / "runtime_capabilities.json",
        {"rank_rows": capability_rows},
    )


def run_worker(args):
    import torch
    import torch.distributed as dist

    if args.world_size != WORLD_SIZE:
        raise ValueError("world_size must be 4")
    if args.rank not in range(WORLD_SIZE):
        raise ValueError("rank is invalid")
    if not isinstance(args.attempt, str) or not args.attempt:
        raise ValueError("attempt is invalid")
    if not _is_hex(args.source_revision, 40):
        raise ValueError("source_revision must be a full SHA")
    if not _is_hex(args.source_tree_sha256, 64):
        raise ValueError("source_tree_sha256 must be a SHA-256")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _rank_paths(output_dir, args.rank)
    _assert_rank_paths_fresh(paths)
    device = torch.device("cuda", args.rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{args.dist_port}",
        rank=args.rank,
        world_size=args.world_size,
    )

    local_rows = []
    memory_rows = []
    lifecycle_rows = []
    cleanup_row = {
        "rank": args.rank,
        "streams_released": False,
        "events_released": False,
        "timed_out": False,
        "process_group_destroyed": False,
    }
    try:
        _atomic_write_json(
            paths["capability"],
            _runtime_capability_row(args.rank, device, torch, dist),
        )
        for workload in build_workload_schedule():
            active_tokens = workload["active_tokens"]
            before_allocated = torch.cuda.memory_allocated(device)
            before_reserved = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
            buffers = OverlapBuffers.create(torch, device, active_tokens)
            _initialize_buffers(
                buffers,
                workload["seed"],
                args.rank,
                torch,
                dist,
            )
            runtime = build_overlap_runtime(
                buffers=buffers,
                torch=torch,
                dist=dist,
            )
            after_allocated = torch.cuda.memory_allocated(device)
            after_reserved = torch.cuda.memory_reserved(device)
            commit_identity = (
                f"{args.attempt}:{args.source_revision}:"
                f"shape-{active_tokens}"
            )

            for pair in workload["warmups"]:
                results = {}
                for arm in pair["arm_order"]:
                    if arm == "baseline":
                        results[arm] = _run_baseline(
                            buffers=buffers,
                            torch=torch,
                            dist=dist,
                        )
                    else:
                        results[arm] = _run_candidate(
                            buffers=buffers,
                            runtime=runtime,
                            torch=torch,
                            commit_identity=commit_identity,
                        )
                results["baseline"]["completed"].synchronize()
                results["candidate"]["completed"].synchronize()

            lifecycle = _run_lifecycle_probe(
                buffers=buffers,
                runtime=runtime,
                torch=torch,
                commit_identity=f"{commit_identity}:lifecycle",
            )
            gathered_identities = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_identities, commit_identity)
            identity_match = len(set(gathered_identities)) == 1
            lifecycle_rows.append({
                "rank": args.rank,
                "active_tokens": active_tokens,
                "commit_identity": commit_identity,
                "rank_commit_identities": gathered_identities,
                "commit_identity_match": identity_match,
                **lifecycle,
            })

            for pair in workload["measurements"]:
                results = {}
                allocated_before_pair = torch.cuda.memory_allocated(device)
                for arm in pair["arm_order"]:
                    if arm == "baseline":
                        results[arm] = _run_baseline(
                            buffers=buffers,
                            torch=torch,
                            dist=dist,
                        )
                    else:
                        results[arm] = _run_candidate(
                            buffers=buffers,
                            runtime=runtime,
                            torch=torch,
                            commit_identity=commit_identity,
                        )
                results["baseline"]["completed"].synchronize()
                results["candidate"]["completed"].synchronize()
                allocated_after_pair = torch.cuda.memory_allocated(device)

                baseline = results["baseline"]
                candidate = results["candidate"]
                reduced_digest = _tensor_digest(
                    candidate["reduced_result"],
                    torch,
                )
                final_digest = _tensor_digest(
                    candidate["final_output"],
                    torch,
                )
                shadow_digest = _tensor_digest(candidate["shadow"], torch)
                rank_reduced_digests = [None] * WORLD_SIZE
                rank_final_digests = [None] * WORLD_SIZE
                rank_shadow_digests = [None] * WORLD_SIZE
                dist.all_gather_object(
                    rank_reduced_digests,
                    reduced_digest,
                )
                dist.all_gather_object(rank_final_digests, final_digest)
                dist.all_gather_object(rank_shadow_digests, shadow_digest)

                allreduce_interval = _event_interval_ns(
                    candidate["started"],
                    buffers.allreduce_started,
                    buffers.allreduce_completed,
                )
                state_copy_interval = _event_interval_ns(
                    candidate["started"],
                    buffers.state_copy_started,
                    buffers.state_copy_completed,
                )
                row = {
                    "attempt": args.attempt,
                    "source_revision": args.source_revision,
                    "source_tree_sha256": args.source_tree_sha256,
                    "active_tokens": active_tokens,
                    "pair_index": pair["pair_index"],
                    "rank": args.rank,
                    "arm_order": list(pair["arm_order"]),
                    "baseline_critical_ns": int(
                        baseline["started"].elapsed_time(
                            baseline["completed"]
                        )
                        * 1_000_000
                    ),
                    "candidate_critical_ns": int(
                        candidate["started"].elapsed_time(
                            candidate["completed"]
                        )
                        * 1_000_000
                    ),
                    "baseline_host_submission_ns": baseline[
                        "host_submission_ns"
                    ],
                    "candidate_host_submission_ns": candidate[
                        "host_submission_ns"
                    ],
                    "allreduce_interval_ns": allreduce_interval,
                    "state_copy_interval_ns": state_copy_interval,
                    "overlap_intersection_ns": interval_intersection_ns(
                        allreduce_interval,
                        state_copy_interval,
                    ),
                    "reduced_output_exact": bool(
                        torch.equal(
                            candidate["reduced_result"],
                            baseline["reduced_result"],
                        )
                    )
                    and len(set(rank_reduced_digests)) == 1,
                    "final_output_exact": bool(
                        torch.equal(
                            candidate["final_output"],
                            baseline["final_output"],
                        )
                    )
                    and len(set(rank_final_digests)) == 1,
                    "shadow_payload_exact": bool(
                        torch.equal(
                            candidate["shadow"],
                            baseline["shadow"],
                        )
                    )
                    and len(set(rank_shadow_digests)) == 1,
                    **lifecycle,
                    "commit_identity_match": identity_match,
                    "finite_output": bool(
                        torch.isfinite(candidate["final_output"]).all().item()
                    ),
                    "timed_path_allocation_count": int(
                        allocated_after_pair != allocated_before_pair
                    ),
                    "timed_out": False,
                    "reduced_output_digest": reduced_digest,
                    "final_output_digest": final_digest,
                    "shadow_payload_digest": shadow_digest,
                }
                local_rows.append(validate_measurement_row(row))

            memory_rows.append({
                "rank": args.rank,
                "active_tokens": active_tokens,
                "before_allocated_bytes": before_allocated,
                "after_allocated_bytes": after_allocated,
                "allocated_delta_bytes": after_allocated - before_allocated,
                "before_reserved_bytes": before_reserved,
                "after_reserved_bytes": after_reserved,
                "reserved_delta_bytes": after_reserved - before_reserved,
                "peak_allocated_delta_bytes": max(
                    0,
                    torch.cuda.max_memory_allocated(device)
                    - before_allocated,
                ),
                "maximum_theoretical_shadow_bytes": (
                    STATE_BYTES_PER_TOKEN_PER_LAYER
                    * active_tokens
                    * LINEAR_LAYER_COUNT
                ),
            })
            del runtime
            del buffers
            cleanup_row["streams_released"] = True
            cleanup_row["events_released"] = True
            torch.cuda.empty_cache()

        maximum_memory = {
            "rank": args.rank,
            "maximum_allocated_delta_bytes": max(
                row["peak_allocated_delta_bytes"] for row in memory_rows
            ),
            "maximum_reserved_delta_bytes": max(
                row["reserved_delta_bytes"] for row in memory_rows
            ),
            "maximum_theoretical_shadow_bytes": max(
                row["maximum_theoretical_shadow_bytes"]
                for row in memory_rows
            ),
            "shape_rows": memory_rows,
        }
        _atomic_write_jsonl(paths["measurements"], local_rows)
        _atomic_write_json(paths["memory"], maximum_memory)
        _atomic_write_json(
            paths["lifecycle"],
            {
                "rank": args.rank,
                "classification": (
                    "PASS"
                    if all(
                        row["active_state_preserved_before_publish"]
                        and row["published_state_exact"]
                        and row["abort_preserved_old_state"]
                        and row["commit_identity_match"]
                        for row in lifecycle_rows
                    )
                    else "FAIL"
                ),
                "shape_rows": lifecycle_rows,
            },
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
        _atomic_write_json(paths["cleanup"], cleanup_row)

    if args.rank == 0:
        _merge_rank_artifacts(output_dir)


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    run_worker(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
