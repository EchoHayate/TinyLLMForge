#!/usr/bin/env python3
"""TP4 segmented CUDA Graph capture census worker contracts.

The CUDA-backed census entry point is intentionally layered on top of the
dependency-light helpers in this module so plan identity and rank aggregation
can be validated on hosts without a CUDA runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from types import SimpleNamespace


WORLD_SIZE = 4
LAYER_COUNT = 64
CANDIDATE_PLANS = {
    "p2": ((0, 32), (32, 64)),
    "p3": ((0, 22), (22, 43), (43, 64)),
    "p4": ((0, 16), (16, 32), (32, 48), (48, 64)),
}
WORKER_SCHEMA = "tinyllmforge.tp4-segmented-capture-worker.v1"
RUNTIME_ENV = "TINYLLMFORGE_SEGMENTED_CENSUS_RUNTIME"


def _load_segment_contract():
    module_name = "_tinyllmforge_segmented_exact_cuda_graph_worker"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_path = (
        Path(__file__).resolve().parents[1]
        / "tinyvllm"
        / "engine"
        / "segmented_exact_cuda_graph.py"
    )
    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("segmented graph contract cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_segment_plan(plan_id: str):
    try:
        ranges = CANDIDATE_PLANS[plan_id]
    except (KeyError, TypeError) as error:
        raise ValueError("candidate plan is invalid") from error
    contract = _load_segment_contract()
    return contract.ExactGraphSegmentPlan(
        layer_count=LAYER_COUNT,
        segments=tuple(
            contract.ExactGraphSegment(
                start_layer=start_layer,
                end_layer=end_layer,
                include_embedding=ordinal == 0,
                include_final=ordinal == len(ranges) - 1,
                include_commit=ordinal == len(ranges) - 1,
            )
            for ordinal, (start_layer, end_layer) in enumerate(ranges)
        ),
    )


def build_engine_config() -> dict:
    """Return the frozen Q1 TP4 configuration with automatic graphs disabled."""

    return {
        "tensor_parallel_size": WORLD_SIZE,
        "gpu_memory_utilization": 0.84,
        "enforce_eager": True,
        "multi_sequence_cuda_graphs": False,
        "multi_sequence_cuda_graph_dynamic_pool_indices": False,
        "multi_sequence_cuda_graph_batch_allowlist": (2, 4, 8),
        "max_num_seqs": 8,
        "max_model_len": 384,
        "max_num_batched_tokens": 2_048,
    }


def _ranked_results(local_result: object, acknowledgements: object):
    if not isinstance(local_result, dict):
        raise RuntimeError("rank result is missing")
    ranked = [(local_result.get("rank"), local_result)]
    try:
        ranked.extend(
            (acknowledgement.rank, acknowledgement.result)
            for acknowledgement in acknowledgements
        )
    except (AttributeError, TypeError) as error:
        raise RuntimeError("rank inventory is invalid") from error
    ranks = [rank for rank, _ in ranked]
    if (
        len(ranked) != WORLD_SIZE
        or set(ranks) != set(range(WORLD_SIZE))
        or len(set(ranks)) != WORLD_SIZE
    ):
        raise RuntimeError("rank inventory is incomplete")
    return sorted(ranked)


def collect_rank_plan_rows(
    plan_id: str,
    local_result: dict,
    acknowledgements: object,
) -> list[dict]:
    """Validate four rank receipts and return canonical segment rows."""

    plan = build_segment_plan(plan_id)
    ranges = CANDIDATE_PLANS[plan_id]
    ranked = _ranked_results(
        local_result,
        acknowledgements,
    )
    if any(
        not isinstance(result, dict)
        or result.get("rank") != rank
        or result.get("plan_id") != plan_id
        or result.get("plan_sha256") != plan.sha256
        for rank, result in ranked
    ):
        raise RuntimeError("rank plan identity disagrees")
    rows = []
    for rank, result in ranked:
        segment_rows = result.get("segment_rows")
        if (
            not isinstance(segment_rows, list)
            or len(segment_rows) != len(ranges)
        ):
            raise RuntimeError("rank segment inventory is incomplete")
        seen_ordinals = set()
        for raw_row in segment_rows:
            if not isinstance(raw_row, dict):
                raise RuntimeError("rank segment row is invalid")
            ordinal = raw_row.get("segment_ordinal")
            if (
                isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or ordinal < 0
                or ordinal >= len(ranges)
                or ordinal in seen_ordinals
            ):
                raise RuntimeError("rank segment inventory is invalid")
            seen_ordinals.add(ordinal)
            start_layer, end_layer = ranges[ordinal]
            row = dict(raw_row)
            row.update({
                "row_id": (
                    f"{plan_id}:segment-{ordinal}:rank-{rank}"
                ),
                "plan_id": plan_id,
                "plan_sha256": plan.sha256,
                "plan_ranges": [list(value) for value in ranges],
                "segment_ordinal": ordinal,
                "start_layer": start_layer,
                "end_layer": end_layer,
                "rank": rank,
                "world_size": WORLD_SIZE,
            })
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (row["rank"], row["segment_ordinal"]),
    )


def capture_segment_program(
    plan_id: str,
    backend: object,
    *,
    clock_ns,
    lifecycle_started_ns: int | None = None,
    memory_before: dict | None = None,
) -> dict:
    """Execute one plan through eager, capture, replay, and cleanup."""

    if not callable(clock_ns):
        raise ValueError("clock_ns must be callable")
    plan = build_segment_plan(plan_id)
    rank = getattr(backend, "rank", None)
    world_size = getattr(backend, "world_size", None)
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 0
        or world_size != WORLD_SIZE
    ):
        raise ValueError("capture backend topology is invalid")

    if lifecycle_started_ns is None:
        lifecycle_started_ns = int(clock_ns())
    elif (
        isinstance(lifecycle_started_ns, bool)
        or not isinstance(lifecycle_started_ns, int)
        or lifecycle_started_ns < 0
    ):
        raise ValueError("lifecycle_started_ns must be non-negative")
    if memory_before is None:
        memory_before = {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
        }
    if (
        not isinstance(memory_before, dict)
        or any(
            isinstance(memory_before.get(name), bool)
            or not isinstance(memory_before.get(name), int)
            or memory_before[name] < 0
            for name in ("allocated_bytes", "reserved_bytes")
        )
    ):
        raise ValueError("memory baseline is invalid")
    snapshot = backend.snapshot()
    graphs = []
    captures = []
    shared_pool = None
    operation_error = None
    comparison = None
    try:
        expected = backend.run_eager()
        backend.restore(snapshot)
        for ordinal, segment in enumerate(plan.segments):
            capture = backend.capture_segment(
                segment,
                ordinal=ordinal,
                shared_pool=shared_pool,
            )
            graph = getattr(capture, "graph", None)
            if graph is None:
                raise RuntimeError("captured graph is missing")
            graphs.append(graph)
            shared_pool = getattr(capture, "shared_pool", None)
            if shared_pool is None:
                raise RuntimeError("shared graph pool is missing")
            captures.append(capture)
        backend.restore(snapshot)
        actual = backend.replay(tuple(graphs))
        comparison = backend.compare(expected, actual)
        required = (
            "exact_output",
            "selected_state_exact",
            "unselected_state_unchanged",
            "scratch_kv_restored",
        )
        if (
            not isinstance(comparison, dict)
            or any(
                not isinstance(comparison.get(name), bool)
                for name in required
            )
        ):
            raise RuntimeError("capture comparison is incomplete")
    except BaseException as error:
        operation_error = error
    restore_error = None
    try:
        backend.restore(snapshot)
    except BaseException as error:
        restore_error = error
    memory_after = dict(memory_before)
    stable_boundary_buffer_bytes = 0
    memory_error = None
    try:
        memory_snapshot = getattr(backend, "memory_snapshot", None)
        if callable(memory_snapshot):
            memory_after = memory_snapshot()
        stable_bytes = getattr(
            backend,
            "stable_boundary_buffer_bytes",
            None,
        )
        if callable(stable_bytes):
            stable_boundary_buffer_bytes = int(stable_bytes())
    except BaseException as error:
        memory_error = error
    reset_errors = []
    for graph in reversed(graphs):
        try:
            backend.reset_graph(graph)
        except BaseException as error:
            reset_errors.append(error)
    lifecycle_duration_ns = int(clock_ns()) - lifecycle_started_ns
    if lifecycle_duration_ns < 0:
        raise RuntimeError("capture lifecycle clock moved backwards")
    if operation_error is not None:
        if restore_error is not None:
            raise operation_error from restore_error
        if memory_error is not None:
            raise operation_error from memory_error
        if reset_errors:
            raise operation_error from reset_errors[0]
        raise operation_error
    if restore_error is not None:
        if memory_error is not None:
            raise restore_error from memory_error
        if reset_errors:
            raise restore_error from reset_errors[0]
        raise restore_error
    if memory_error is not None:
        if reset_errors:
            raise memory_error from reset_errors[0]
        raise memory_error
    if reset_errors:
        raise reset_errors[0]
    if (
        not isinstance(memory_after, dict)
        or any(
            isinstance(memory_after.get(name), bool)
            or not isinstance(memory_after.get(name), int)
            or memory_after[name] < 0
            for name in ("allocated_bytes", "reserved_bytes")
        )
        or stable_boundary_buffer_bytes < 0
    ):
        raise RuntimeError("capture memory accounting is invalid")
    allocated_delta_bytes = max(
        0,
        memory_after["allocated_bytes"]
        - memory_before["allocated_bytes"],
    )
    reserved_delta_bytes = max(
        0,
        memory_after["reserved_bytes"]
        - memory_before["reserved_bytes"],
    )

    segment_rows = []
    for ordinal, (segment, capture) in enumerate(
        zip(plan.segments, captures, strict=True)
    ):
        segment_rows.append({
            "segment_ordinal": ordinal,
            "start_layer": segment.start_layer,
            "end_layer": segment.end_layer,
            "include_embedding": segment.include_embedding,
            "include_final": segment.include_final,
            "include_commit": segment.include_commit,
            "capture_body_duration_ns": int(
                capture.capture_body_duration_ns
            ),
            "post_capture_sync_duration_ns": int(
                capture.post_capture_sync_duration_ns
            ),
            "segment_capture_duration_ns": int(
                capture.segment_capture_duration_ns
            ),
            "lifecycle_duration_ns": lifecycle_duration_ns,
            "allocated_delta_bytes": allocated_delta_bytes,
            "reserved_delta_bytes": reserved_delta_bytes,
            "stable_boundary_buffer_bytes": (
                stable_boundary_buffer_bytes
            ),
            **comparison,
            "graph_reset": True,
            "complete": True,
        })
    return {
        "rank": rank,
        "plan_id": plan_id,
        "plan_sha256": plan.sha256,
        "lifecycle_duration_ns": lifecycle_duration_ns,
        "segment_rows": segment_rows,
    }


def _state_equal(left, right, *, torch_module) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, torch_module.Tensor):
        return bool(torch_module.equal(left, right))
    if isinstance(left, tuple):
        return len(left) == len(right) and all(
            _state_equal(a, b, torch_module=torch_module)
            for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _state_equal(
                left[key],
                right[key],
                torch_module=torch_module,
            )
            for key in left
        )
    return left == right


@dataclass(frozen=True)
class _CapturedSegment:
    graph: object
    shared_pool: object
    capture_body_duration_ns: int
    post_capture_sync_duration_ns: int
    segment_capture_duration_ns: int


class _CudaSegmentedCaptureBackend:
    def __init__(
        self,
        runner,
        *,
        plan_id: str,
        input_ids,
        positions,
        context,
        torch_module,
        temporary_context,
    ):
        self.runner = runner
        self.rank = int(runner.rank)
        self.world_size = int(runner.world_size)
        self.plan_id = plan_id
        self.plan = build_segment_plan(plan_id)
        self.input_ids = input_ids
        self.positions = positions
        self.context = context
        self.torch = torch_module
        self.temporary_context = temporary_context
        self.model = runner.model
        self.leases = tuple(runner._last_hybrid_state_leases)
        self.token_counts = tuple(
            runner._last_hybrid_state_token_counts
        )
        if (
            len(self.leases) != 8
            or len(self.token_counts) != 8
            or int(input_ids.shape[0]) != 8
        ):
            raise RuntimeError(
                "segmented census requires the frozen Q1 batch of eight"
            )
        manifest = self.model.exact_cuda_graph_lease_manifest(
            self.leases,
            tuple(runner._last_hybrid_state_request_ids),
        )
        self.lease_manifest_sha256 = manifest.sha256
        self.state_slot_ids = torch_module.tensor(
            manifest.slot_ids,
            dtype=torch_module.int64,
            device=runner.kv_cache.device,
        )
        used_blocks = {
            int(block_id)
            for row in context.block_tables.tolist()
            for block_id in row
            if int(block_id) >= 0
        }
        available_blocks = [
            block_id
            for block_id in range(
                int(runner._physical_num_kvcache_blocks)
            )
            if block_id not in used_blocks
        ]
        if len(available_blocks) < 8:
            raise RuntimeError(
                "segmented census has insufficient scratch KV blocks"
            )
        self.scratch_slots = [
            block_id * int(runner.block_size)
            for block_id in available_blocks[:8]
        ]
        self.static_input_ids = input_ids.clone()
        self.static_positions = positions.clone()
        self.static_slot_mapping = torch_module.tensor(
            self.scratch_slots,
            dtype=torch_module.int32,
            device=runner.kv_cache.device,
        )
        self.static_context_lens = context.context_lens.clone()
        self.static_block_tables = context.block_tables.clone()
        self._hidden = None
        self._candidates = []
        self._captured_logits = None
        self._selected_snapshot = None
        self._scratch_snapshot = None
        self._unselected_snapshot = None

    def _context(self):
        return self.temporary_context(
            slot_mapping=self.static_slot_mapping,
            context_lens=self.static_context_lens,
            block_tables=self.static_block_tables,
            flash_attn_num_splits=int(
                self.context.flash_attn_num_splits
            ),
            force_attention_backend=True,
        )

    def _snapshot_unselected(self):
        pool = self.model.layer_stack.state_transaction.pool
        selected = set(int(value) for value in self.state_slot_ids.tolist())
        unselected = [
            slot_id
            for slot_id in range(pool.capacity)
            if slot_id not in selected
        ]
        return {
            key: tensor[unselected].clone()
            for key, tensor in pool._tensors.items()
        }

    def snapshot(self):
        self._selected_snapshot = (
            self.model.snapshot_exact_cuda_graph_state(self.leases)
        )
        self._scratch_snapshot = self.runner.snapshot_kv_slots(
            self.scratch_slots
        )
        self._unselected_snapshot = self._snapshot_unselected()
        return {
            "selected": self._selected_snapshot,
            "scratch": self._scratch_snapshot,
        }

    def _selected_state(self):
        return self.model.snapshot_exact_cuda_graph_state(self.leases)

    def run_eager(self):
        with self._context():
            logits = (
                self.model.run_exact_cuda_graph_step_by_pool_index(
                    self.state_slot_ids,
                    self.token_counts,
                    self.static_input_ids,
                    self.static_positions,
                )
            )
            self.torch.cuda.synchronize()
        return {
            "logits": logits.clone(),
            "selected": self._selected_state(),
        }

    def restore(self, snapshot):
        self.model.restore_exact_cuda_graph_state(
            self.leases,
            snapshot["selected"],
        )
        self.runner.restore_kv_slots(
            self.scratch_slots,
            snapshot["scratch"],
        )

    def capture_segment(
        self,
        segment,
        *,
        ordinal: int,
        shared_pool,
    ) -> _CapturedSegment:
        if ordinal == 0:
            self._hidden = None
            self._candidates = []
            self._captured_logits = None
        graph = self.torch.cuda.CUDAGraph()
        started_ns = time.perf_counter_ns()
        with self._context():
            with self.torch.cuda.graph(graph, pool=shared_pool):
                hidden = self._hidden
                if segment.include_embedding:
                    hidden = self.model.embed_exact_graph_inputs(
                        self.static_input_ids
                    )
                prepared = (
                    self.model.run_exact_cuda_graph_layer_range(
                        state_slot_ids=self.state_slot_ids,
                        token_counts=self.token_counts,
                        position_ids=self.static_positions,
                        hidden_states=hidden,
                        start_layer=segment.start_layer,
                        end_layer=segment.end_layer,
                    )
                )
                self._hidden = prepared.hidden_states
                self._candidates.append(prepared.candidates)
                if segment.include_final:
                    self._captured_logits = (
                        self.model.finalize_exact_cuda_graph_hidden(
                            self._hidden
                        )
                    )
                if segment.include_commit:
                    self.model.commit_exact_cuda_graph_candidates(
                        self.state_slot_ids,
                        tuple(self._candidates),
                    )
        body_completed_ns = time.perf_counter_ns()
        self.torch.cuda.synchronize()
        finished_ns = time.perf_counter_ns()
        resolved_pool = (
            graph.pool() if shared_pool is None else shared_pool
        )
        return _CapturedSegment(
            graph=graph,
            shared_pool=resolved_pool,
            capture_body_duration_ns=body_completed_ns - started_ns,
            post_capture_sync_duration_ns=(
                finished_ns - body_completed_ns
            ),
            segment_capture_duration_ns=finished_ns - started_ns,
        )

    def replay(self, graphs):
        current_manifest = self.model.exact_cuda_graph_lease_manifest(
            tuple(self.runner._last_hybrid_state_leases),
            tuple(self.runner._last_hybrid_state_request_ids),
        )
        if current_manifest.sha256 != self.lease_manifest_sha256:
            raise RuntimeError(
                "segmented census lease manifest drift before replay"
            )
        with self._context():
            for graph in graphs:
                graph.replay()
            self.torch.cuda.synchronize()
        result = {
            "logits": self._captured_logits.clone(),
            "selected": self._selected_state(),
        }
        self.restore({
            "selected": self._selected_snapshot,
            "scratch": self._scratch_snapshot,
        })
        return result

    def compare(self, expected, actual):
        scratch_after = self.runner.snapshot_kv_slots(
            self.scratch_slots
        )
        return {
            "exact_output": bool(
                self.torch.equal(
                    expected["logits"],
                    actual["logits"],
                )
            ),
            "selected_state_exact": _state_equal(
                expected["selected"],
                actual["selected"],
                torch_module=self.torch,
            ),
            "unselected_state_unchanged": _state_equal(
                self._unselected_snapshot,
                self._snapshot_unselected(),
                torch_module=self.torch,
            ),
            "scratch_kv_restored": _state_equal(
                self._scratch_snapshot,
                scratch_after,
                torch_module=self.torch,
            ),
        }

    def memory_snapshot(self):
        device = self.runner.kv_cache.device
        return {
            "allocated_bytes": int(
                self.torch.cuda.memory_allocated(device)
            ),
            "reserved_bytes": int(
                self.torch.cuda.memory_reserved(device)
            ),
        }

    def stable_boundary_buffer_bytes(self):
        seen = set()

        def count(value):
            if isinstance(value, self.torch.Tensor):
                identity = id(value)
                if identity in seen:
                    return 0
                seen.add(identity)
                return int(value.numel()) * int(value.element_size())
            if isinstance(value, dict):
                return sum(count(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return sum(count(item) for item in value)
            values = getattr(value, "__dict__", None)
            if isinstance(values, dict):
                return sum(count(item) for item in values.values())
            return 0

        return count((
            self.static_input_ids,
            self.static_positions,
            self.static_slot_mapping,
            self.static_context_lens,
            self.static_block_tables,
            self._hidden,
            self._candidates,
            self._captured_logits,
        ))

    @staticmethod
    def reset_graph(graph):
        graph.reset()


class _SegmentedCensusModelRunnerMixin:
    def arm_segmented_capture_census(self, plan_id: str) -> dict:
        build_segment_plan(plan_id)
        self._segmented_census_plan_id = plan_id
        self._segmented_census_result = None
        return {"rank": int(self.rank), "armed": True}

    def segmented_capture_census_result(self) -> dict:
        result = getattr(self, "_segmented_census_result", None)
        if not isinstance(result, dict):
            raise RuntimeError("segmented capture census did not execute")
        return dict(result)

    def run_model(self, input_ids, positions, is_prefill, *args, **kwargs):
        plan_id = getattr(self, "_segmented_census_plan_id", None)
        if (
            not is_prefill
            and plan_id is not None
            and getattr(self, "_segmented_census_result", None) is None
        ):
            from tinyvllm.utils.context import (
                get_context,
                temporary_context,
            )

            torch_module = __import__("torch")
            lifecycle_started_ns = time.perf_counter_ns()
            device = input_ids.device
            memory_before = {
                "allocated_bytes": int(
                    torch_module.cuda.memory_allocated(device)
                ),
                "reserved_bytes": int(
                    torch_module.cuda.memory_reserved(device)
                ),
            }
            backend = _CudaSegmentedCaptureBackend(
                self,
                plan_id=plan_id,
                input_ids=input_ids,
                positions=positions,
                context=get_context(),
                torch_module=torch_module,
                temporary_context=temporary_context,
            )
            with torch_module.inference_mode():
                self._segmented_census_result = (
                    capture_segment_program(
                        plan_id,
                        backend,
                        clock_ns=time.perf_counter_ns,
                        lifecycle_started_ns=lifecycle_started_ns,
                        memory_before=memory_before,
                    )
                )
        return super().run_model(
            input_ids,
            positions,
            is_prefill,
            *args,
            **kwargs,
        )


def _install_runtime_model_runner():
    existing = globals().get("SegmentedCensusModelRunner")
    if existing is not None:
        return existing
    from tinyvllm.engine.model_runner import ModelRunner
    import tinyvllm.engine.llm_engine as llm_engine_module

    runtime_class = type(
        "SegmentedCensusModelRunner",
        (_SegmentedCensusModelRunnerMixin, ModelRunner),
        {"__module__": __name__},
    )
    globals()["SegmentedCensusModelRunner"] = runtime_class
    llm_engine_module.ModelRunner = runtime_class
    return runtime_class


def _free_rendezvous_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _rendezvous_address_in_use(error: BaseException) -> bool:
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current)
        if (
            "EADDRINUSE" in message
            or "address already in use" in message.lower()
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _cleanup_failed_engine_children() -> None:
    import multiprocessing

    children = tuple(multiprocessing.active_children())
    for child in children:
        if child.is_alive():
            child.terminate()
    for child in children:
        child.join(timeout=10.0)
    lingering = tuple(child for child in children if child.is_alive())
    for child in lingering:
        child.kill()
    for child in lingering:
        child.join(timeout=10.0)
    if any(child.is_alive() for child in lingering):
        raise RuntimeError(
            "failed engine children remained after rendezvous retry cleanup"
        )


def create_engine_with_rendezvous_retry(
    model_root,
    *,
    engine_config,
    port_factory,
    engine_factory,
    environment=os.environ,
    cleanup_failed_attempt=_cleanup_failed_engine_children,
    sleep=time.sleep,
    maximum_attempts=3,
    retry_delay_s=0.25,
):
    if maximum_attempts <= 0:
        raise ValueError("maximum_attempts must be positive")
    for attempt in range(maximum_attempts):
        port = int(port_factory())
        environment["TINYVLLM_DIST_PORT"] = str(port)
        try:
            return (
                engine_factory(
                    Path(model_root),
                    **dict(engine_config),
                ),
                port,
            )
        except RuntimeError as error:
            if not _rendezvous_address_in_use(error):
                raise
            cleanup_failed_attempt()
            if attempt + 1 == maximum_attempts:
                raise RuntimeError(
                    "rendezvous port retries exhausted"
                ) from error
            sleep(retry_delay_s)
    raise AssertionError("unreachable")


def _default_engine_factory(model_root, **kwargs):
    os.environ[RUNTIME_ENV] = "1"
    _install_runtime_model_runner()
    from tinyvllm.engine.llm_engine import LLMEngine

    engine, _port = create_engine_with_rendezvous_retry(
        model_root,
        engine_config=kwargs,
        port_factory=_free_rendezvous_port,
        engine_factory=lambda root, **config: LLMEngine(
            str(root),
            **config,
        ),
    )
    return engine


def _default_workload_runner(engine) -> None:
    from tinyvllm.sampling_params import SamplingParams

    prompts = [
        [11 + ((position + request_index * 257) % 2000)
         for position in range(256)]
        for request_index in range(8)
    ]
    engine.generate(
        prompts,
        SamplingParams(
            temperature=0.0,
            max_tokens=2,
            ignore_eos=True,
        ),
        use_tqdm=False,
    )


def _validate_engine_cleanup(receipt: object) -> dict:
    if (
        not isinstance(receipt, dict)
        or receipt.get("rank_exit_codes") != [0, 0, 0, 0]
        or receipt.get("process_group_destroyed") is not True
        or receipt.get("owned_children_remaining") != []
    ):
        raise RuntimeError("TP4 worker cleanup receipt is incomplete")
    rank_rows = receipt.get("rank_cleanup_receipts")
    if (
        not isinstance(rank_rows, list)
        or len(rank_rows) != WORLD_SIZE
        or {row.get("rank") for row in rank_rows}
        != set(range(WORLD_SIZE))
        or any(
            row.get("process_group_destroyed") is not True
            for row in rank_rows
        )
    ):
        raise RuntimeError("TP4 worker cleanup receipt is incomplete")
    return dict(receipt)


def run_plan_census(
    *,
    model_root,
    plan_id: str,
    timeout_s: float,
    engine_factory=None,
    workload_runner=None,
) -> dict:
    if plan_id not in CANDIDATE_PLANS:
        raise ValueError("candidate plan is invalid")
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or timeout_s <= 0
    ):
        raise ValueError("timeout_s must be positive")
    if engine_factory is None:
        engine_factory = _default_engine_factory
    if workload_runner is None:
        workload_runner = _default_workload_runner
    if not callable(engine_factory) or not callable(workload_runner):
        raise ValueError("worker dependencies must be callable")
    engine = None
    operation_error = None
    rows = None
    cleanup = None
    try:
        engine = engine_factory(
            model_root,
            **build_engine_config(),
        )
        local, acknowledgements = (
            engine.call_model_runner_acknowledged(
                "arm_segmented_capture_census",
                plan_id,
                timeout_s=float(timeout_s),
            )
        )
        armed = _ranked_results(local, acknowledgements)
        if any(
            not isinstance(result, dict)
            or result.get("rank") != rank
            or result.get("armed") is not True
            for rank, result in armed
        ):
            raise RuntimeError(
                "segmented capture census arming is incomplete"
            )
        workload_runner(engine)
        local, acknowledgements = (
            engine.call_model_runner_acknowledged(
                "segmented_capture_census_result",
                timeout_s=float(timeout_s),
            )
        )
        rows = collect_rank_plan_rows(
            plan_id,
            local,
            acknowledgements,
        )
    except BaseException as error:
        operation_error = error
    cleanup_error = None
    if engine is not None:
        try:
            cleanup = _validate_engine_cleanup(engine.exit())
        except BaseException as error:
            cleanup_error = error
    if operation_error is not None:
        if cleanup_error is not None:
            raise operation_error from cleanup_error
        raise operation_error
    if cleanup_error is not None:
        raise cleanup_error
    return {
        "plan_id": plan_id,
        "plan_sha256": build_segment_plan(plan_id).sha256,
        "rows": rows,
        "cleanup": cleanup,
    }


def run_census(
    *,
    model_root,
    timeout_s: float,
    engine_factory=None,
    workload_runner=None,
    run_tag: str | None = None,
) -> dict:
    rows = []
    process_receipts = {}
    for plan_id in CANDIDATE_PLANS:
        result = run_plan_census(
            model_root=model_root,
            plan_id=plan_id,
            timeout_s=timeout_s,
            engine_factory=engine_factory,
            workload_runner=workload_runner,
        )
        rows.extend(result["rows"])
        process_receipts[plan_id] = result["cleanup"]
    return {
        "schema_version": WORKER_SCHEMA,
        "run_tag": run_tag,
        "rows": rows,
        "process_receipts": process_receipts,
    }


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
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


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
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


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    result = run_census(
        model_root=args.model_root,
        timeout_s=args.timeout_s,
        run_tag=args.run_tag,
    )
    _atomic_write_jsonl(
        args.output_dir / "segment_rows.jsonl",
        result["rows"],
    )
    _atomic_write_json(
        args.output_dir / "process_receipts.json",
        {
            "schema_version": (
                "tinyllmforge.tp4-segmented-capture-process.v1"
            ),
            "run_tag": args.run_tag,
            "plans": result["process_receipts"],
        },
    )
    _atomic_write_json(
        args.output_dir / "worker_result.json",
        result,
    )
    print(json.dumps({
        "schema_version": WORKER_SCHEMA,
        "classification": "WORKER_COMPLETE",
        "row_count": len(result["rows"]),
    }, sort_keys=True))
    return 0


if os.environ.get(RUNTIME_ENV) == "1":
    _install_runtime_model_runner()


if __name__ == "__main__":
    raise SystemExit(main())
