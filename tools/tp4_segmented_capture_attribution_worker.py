#!/usr/bin/env python3
"""Diagnostic primitives for TP4 segmented-capture attribution."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time


WORLD_SIZE = 4
WORKER_SCHEMA = "tinyllmforge.tp4-segmented-attribution-worker.v1"
RUNTIME_ENV = "TINYLLMFORGE_SEGMENTED_ATTRIBUTION_RUNTIME"
SOURCE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class PhaseA1WorkerError(RuntimeError):
    def __init__(self, message: str, *, result: dict):
        super().__init__(message)
        self.result = result


def _load_attribution_contract():
    module_name = "_tinyllmforge_segmented_capture_attribution_worker"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_path = (
        Path(__file__).resolve().parents[1]
        / "tinyvllm"
        / "engine"
        / "segmented_capture_attribution.py"
    )
    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("capture attribution contract cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _sentinel_seed(
    *,
    run_tag: str,
    rank: int,
    selector: str,
    layer: int,
    scratch_slot_ordinal: int,
) -> int:
    identity = "|".join(
        (
            run_tag,
            str(rank),
            selector,
            str(layer),
            str(scratch_slot_ordinal),
        )
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(identity).digest()[:8],
        byteorder="big",
        signed=False,
    )


def fill_scratch_sentinel(
    runner,
    scratch_slots: list[int],
    *,
    run_tag: str,
    rank: int,
    torch_module,
) -> None:
    if not isinstance(run_tag, str) or not run_tag:
        raise ValueError("run_tag must be non-empty")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    if (
        not isinstance(scratch_slots, list)
        or not scratch_slots
        or any(
            isinstance(slot, bool)
            or not isinstance(slot, int)
            or slot < 0
            for slot in scratch_slots
        )
    ):
        raise ValueError("scratch_slots must be non-empty physical slots")
    cache = runner.kv_cache
    for selector_index, selector in enumerate(("key", "value")):
        for layer in range(int(cache.shape[1])):
            for scratch_slot_ordinal, physical_slot in enumerate(
                scratch_slots
            ):
                block_id = physical_slot // int(runner.block_size)
                offset = physical_slot % int(runner.block_size)
                destination = cache[
                    selector_index,
                    layer,
                    block_id,
                    offset,
                ]
                seed = _sentinel_seed(
                    run_tag=run_tag,
                    rank=rank,
                    selector=selector,
                    layer=layer,
                    scratch_slot_ordinal=scratch_slot_ordinal,
                )
                values = (
                    (
                        torch_module.arange(
                            destination.numel(),
                            device=destination.device,
                            dtype=torch_module.int64,
                        )
                        * 1_315_423_911
                        + seed
                    )
                    % 2_047
                    + 1
                )
                if destination.dtype.is_floating_point:
                    values = values.to(dtype=torch_module.float32) / 1_024
                destination.copy_(
                    values.reshape(destination.shape).to(
                        dtype=destination.dtype
                    )
                )
    torch_module.cuda.synchronize()


def _tensor_digest(tensor, *, selector: str, torch_module) -> dict:
    cpu = tensor.detach().cpu().contiguous()
    metadata = {
        "selector": selector,
        "dtype": str(cpu.dtype),
        "shape": tuple(int(value) for value in cpu.shape),
        "byte_count": int(cpu.numel()) * int(cpu.element_size()),
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    digest.update(cpu.view(torch_module.uint8).numpy().tobytes())
    record = _load_attribution_contract().ScratchTensorDigest(
        **metadata,
        sha256=digest.hexdigest(),
    )
    return asdict(record)


def _first_mismatch_location(mask, *, torch_module) -> dict[str, int]:
    coordinate = [
        int(value)
        for value in torch_module.nonzero(
            mask,
            as_tuple=False,
        )[0].tolist()
    ]
    if len(coordinate) < 4:
        raise ValueError("scratch snapshot must expose layer/slot/head")
    trailing_shape = tuple(int(value) for value in mask.shape[3:])
    element_offset = 0
    for index, size in zip(
        coordinate[3:],
        trailing_shape,
        strict=True,
    ):
        element_offset = element_offset * size + index
    return {
        "layer": coordinate[0],
        "scratch_slot_ordinal": coordinate[1],
        "head": coordinate[2],
        "element_offset": element_offset,
    }


def _tensor_diff(current, baseline, *, torch_module) -> dict:
    if current.shape != baseline.shape or current.dtype != baseline.dtype:
        raise ValueError("scratch snapshot dtype or shape changed")
    mismatch = current != baseline
    mismatch_count = int(
        torch_module.count_nonzero(mismatch).item()
    )
    if mismatch_count == 0:
        record = _load_attribution_contract().ScratchDiffSummary(
            equal_to_s0=True,
            mismatching_element_count=0,
            first_mismatch=None,
            max_absolute_difference=0.0,
        )
        return asdict(record)
    maximum = float(
        (
            current.to(dtype=torch_module.float32)
            - baseline.to(dtype=torch_module.float32)
        )
        .abs()
        .max()
        .item()
    )
    record = _load_attribution_contract().ScratchDiffSummary(
        equal_to_s0=False,
        mismatching_element_count=mismatch_count,
        first_mismatch=_first_mismatch_location(
            mismatch,
            torch_module=torch_module,
        ),
        max_absolute_difference=maximum,
    )
    return asdict(record)


def _checkpoint_from_snapshot(
    snapshot: dict,
    *,
    checkpoint: str,
    rank: int,
    s0: dict | None,
    synchronized: bool,
    segment_ordinal: int | None,
    torch_module,
) -> dict:
    baseline = snapshot if s0 is None else s0
    contract = _load_attribution_contract()
    record = contract.ScratchCheckpointRecord(
        checkpoint=checkpoint,
        rank=rank,
        synchronized=synchronized,
        keys=contract.ScratchTensorDigest(
            **_tensor_digest(
                snapshot["keys"],
                selector="key",
                torch_module=torch_module,
            )
        ),
        values=contract.ScratchTensorDigest(
            **_tensor_digest(
                snapshot["values"],
                selector="value",
                torch_module=torch_module,
            )
        ),
        key_diff=contract.ScratchDiffSummary(
            **_tensor_diff(
                snapshot["keys"],
                baseline["keys"],
                torch_module=torch_module,
            )
        ),
        value_diff=contract.ScratchDiffSummary(
            **_tensor_diff(
                snapshot["values"],
                baseline["values"],
                torch_module=torch_module,
            )
        ),
        segment_ordinal=segment_ordinal,
    )
    return asdict(record)


def snapshot_scratch_checkpoint(
    runner,
    scratch_slots: list[int],
    *,
    checkpoint: str,
    rank: int,
    s0: dict | None,
    synchronized: bool,
    segment_ordinal: int | None,
    torch_module,
) -> dict:
    if synchronized is not True:
        raise ValueError("scratch checkpoint must be synchronized")
    torch_module.cuda.synchronize()
    snapshot = runner.snapshot_kv_slots(scratch_slots)
    return _checkpoint_from_snapshot(
        snapshot,
        checkpoint=checkpoint,
        rank=rank,
        s0=s0,
        synchronized=True,
        segment_ordinal=segment_ordinal,
        torch_module=torch_module,
    )


def first_scratch_divergence(rows: list[dict]) -> str | None:
    for row in rows:
        if (
            row.get("key_diff", {}).get("equal_to_s0") is False
            or row.get("value_diff", {}).get("equal_to_s0") is False
        ):
            return row.get("checkpoint")
    return None


def _raise_primary(
    operation_error: BaseException | None,
    cleanup_errors: list[BaseException],
) -> None:
    if operation_error is not None:
        if cleanup_errors:
            raise operation_error from cleanup_errors[0]
        raise operation_error
    if cleanup_errors:
        raise cleanup_errors[0]


def run_scratch_forensic_sequence(
    backend,
    *,
    segment_count: int,
) -> dict:
    if (
        isinstance(segment_count, bool)
        or not isinstance(segment_count, int)
        or segment_count <= 0
    ):
        raise ValueError("segment_count must be positive")
    checkpoint_rows = []
    graphs = []
    restore_round_trip_exact = False
    operation_error = None
    try:
        backend.initialize_sentinel()
        checkpoint_rows.append(
            backend.checkpoint("S0", segment_ordinal=None)
        )
        backend.restore_s0()
        restore_round_trip_exact = bool(
            backend.scratch_equal_to_s0()
        )
        if not restore_round_trip_exact:
            raise RuntimeError("scratch_restore_primitive")
        backend.run_eager()
        checkpoint_rows.append(
            backend.checkpoint("S1", segment_ordinal=None)
        )
        backend.restore_s0()
        checkpoint_rows.append(
            backend.checkpoint("S2", segment_ordinal=None)
        )
        for ordinal in range(segment_count):
            graph = backend.capture_segment(ordinal)
            graphs.append(graph)
            checkpoint_rows.append(
                backend.checkpoint(
                    "S3",
                    segment_ordinal=ordinal,
                )
            )
        backend.restore_s0()
        checkpoint_rows.append(
            backend.checkpoint("S4", segment_ordinal=None)
        )
        backend.replay(tuple(graphs))
        checkpoint_rows.append(
            backend.checkpoint("S5", segment_ordinal=None)
        )
        backend.restore_s0()
        checkpoint_rows.append(
            backend.checkpoint("S6", segment_ordinal=None)
        )
    except BaseException as error:
        operation_error = error

    cleanup_errors = []
    try:
        backend.restore_s0()
    except BaseException as error:
        cleanup_errors.append(error)
    for graph in reversed(graphs):
        try:
            backend.reset_graph(graph)
        except BaseException as error:
            cleanup_errors.append(error)
    try:
        backend.synchronize()
    except BaseException as error:
        cleanup_errors.append(error)
    try:
        checkpoint_rows.append(
            backend.checkpoint("S7", segment_ordinal=None)
        )
    except BaseException as error:
        cleanup_errors.append(error)
    _raise_primary(operation_error, cleanup_errors)
    return {
        "restore_round_trip_exact": restore_round_trip_exact,
        "checkpoint_rows": checkpoint_rows,
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


class _AttributionCudaBackend:
    """Scratch-forensic portion of the diagnostic CUDA backend."""

    def __init__(
        self,
        runner,
        *,
        scratch_slots: list[int],
        run_tag: str,
        rank: int,
        torch_module,
        input_ids=None,
        positions=None,
        state_slot_ids=None,
        token_counts: tuple[int, ...] | None = None,
        runtime_context_factory=None,
        source_revision: str = "",
        plan_sha256: str = "",
        clock_ns=None,
    ):
        self.runner = runner
        self.model = getattr(runner, "model", None)
        self.scratch_slots = list(scratch_slots)
        self.run_tag = run_tag
        self.rank = rank
        self.torch = torch_module
        self.input_ids = (
            None if input_ids is None else input_ids.clone()
        )
        self.positions = (
            None if positions is None else positions.clone()
        )
        self.state_slot_ids = state_slot_ids
        self.token_counts = token_counts
        self.runtime_context_factory = runtime_context_factory
        self.source_revision = source_revision
        self.plan_sha256 = plan_sha256
        self.clock_ns = (
            time.perf_counter_ns if clock_ns is None else clock_ns
        )
        self._s0 = None
        self._hidden = None
        self._candidates = []
        self._captured_logits = None
        self._pool_ordinals = {}
        self._isolated_hidden = None
        self._captured_segments = []
        self._active_graphs = []
        self._reset_graph_ids = set()
        self._graph_reset_durations_ns = {}
        self._post_capture_restore_ns = 0
        self._restore_call_count = 0
        self._synchronization_count = 0
        self._eager_result = None
        self._replay_result = None
        self.leases = tuple(
            getattr(runner, "_last_hybrid_state_leases", ())
        )
        self.request_ids = tuple(
            getattr(runner, "_last_hybrid_state_request_ids", ())
        )
        if not self.leases or len(self.leases) != len(self.request_ids):
            raise RuntimeError(
                "Phase A1 requires aligned hybrid-state leases"
            )
        manifest = self.model.exact_cuda_graph_lease_manifest(
            self.leases,
            self.request_ids,
        )
        self.lease_manifest_sha256 = manifest.sha256
        self._selected_baseline = (
            self.model.snapshot_exact_cuda_graph_state(self.leases)
        )
        self._unselected_baseline = self._snapshot_unselected()

    def _snapshot_unselected(self) -> dict:
        pool = self.model.layer_stack.state_transaction.pool
        selected = {
            int(value) for value in self.state_slot_ids.tolist()
        }
        unselected = [
            slot_id
            for slot_id in range(int(pool.capacity))
            if slot_id not in selected
        ]
        return {
            name: tensor[unselected].clone()
            for name, tensor in pool._tensors.items()
        }

    def _validate_lease_manifest(self) -> None:
        current = self.model.exact_cuda_graph_lease_manifest(
            tuple(
                getattr(
                    self.runner,
                    "_last_hybrid_state_leases",
                    (),
                )
            ),
            tuple(
                getattr(
                    self.runner,
                    "_last_hybrid_state_request_ids",
                    (),
                )
            ),
        )
        if current.sha256 != self.lease_manifest_sha256:
            raise RuntimeError(
                "Phase A1 lease manifest drift before execution"
            )

    def restore_control_baseline(self) -> None:
        self._validate_lease_manifest()
        if not _state_equal(
            self._unselected_baseline,
            self._snapshot_unselected(),
            torch_module=self.torch,
        ):
            raise RuntimeError(
                "Phase A1 unselected state changed"
            )
        self.model.restore_exact_cuda_graph_state(
            self.leases,
            self._selected_baseline,
        )
        if self._s0 is not None:
            self.runner.restore_kv_slots(
                self.scratch_slots,
                self._s0,
            )
            self._synchronization_count += 1
        else:
            self.synchronize()

    def reset_control_graphs(self) -> None:
        errors = []
        for graph in reversed(tuple(self._active_graphs)):
            try:
                self.reset_graph(graph)
            except BaseException as error:
                errors.append(error)
        self._active_graphs = []
        if errors:
            raise errors[0]

    def initialize_sentinel(self) -> None:
        fill_scratch_sentinel(
            self.runner,
            self.scratch_slots,
            run_tag=self.run_tag,
            rank=self.rank,
            torch_module=self.torch,
        )
        self._synchronization_count += 1

    def checkpoint(
        self,
        checkpoint: str,
        *,
        segment_ordinal: int | None,
    ) -> dict:
        self.synchronize()
        started_ns = int(self.clock_ns())
        snapshot = self.runner.snapshot_kv_slots(self.scratch_slots)
        if checkpoint == "S0":
            self._s0 = {
                name: tensor.clone()
                for name, tensor in snapshot.items()
            }
        if self._s0 is None:
            raise RuntimeError("S0 scratch snapshot is missing")
        row = _checkpoint_from_snapshot(
            snapshot,
            checkpoint=checkpoint,
            rank=self.rank,
            s0=self._s0,
            synchronized=True,
            segment_ordinal=segment_ordinal,
            torch_module=self.torch,
        )
        row["scratch_snapshot_cpu_ns"] = (
            int(self.clock_ns()) - started_ns
        )
        return row

    def restore_s0(self) -> None:
        if self._s0 is None:
            raise RuntimeError("S0 scratch snapshot is missing")
        started_ns = int(self.clock_ns())
        self.model.restore_exact_cuda_graph_state(
            self.leases,
            self._selected_baseline,
        )
        self.runner.restore_kv_slots(self.scratch_slots, self._s0)
        self._synchronization_count += 1
        duration_ns = int(self.clock_ns()) - started_ns
        self._restore_call_count += 1
        if self._restore_call_count == 3:
            self._post_capture_restore_ns = duration_ns

    def scratch_equal_to_s0(self) -> bool:
        if self._s0 is None:
            raise RuntimeError("S0 scratch snapshot is missing")
        self.synchronize()
        current = self.runner.snapshot_kv_slots(self.scratch_slots)
        return all(
            bool(self.torch.equal(current[name], self._s0[name]))
            for name in ("keys", "values")
        )

    def synchronize(self) -> None:
        self.torch.cuda.synchronize()
        self._synchronization_count += 1

    def memory_snapshot(self) -> dict[str, int]:
        device = self.runner.kv_cache.device
        return {
            "allocated_bytes": int(
                self.torch.cuda.memory_allocated(device)
            ),
            "reserved_bytes": int(
                self.torch.cuda.memory_reserved(device)
            ),
        }

    def prepare_segment(self, segment, *, ordinal: int) -> None:
        del segment
        if ordinal == 0:
            self._hidden = self._isolated_hidden
            self._candidates = []
            self._captured_logits = None

    def create_graph(self):
        return self.torch.cuda.CUDAGraph()

    @contextmanager
    def capture_context(self, graph, *, shared_pool):
        runtime_context = (
            nullcontext()
            if self.runtime_context_factory is None
            else self.runtime_context_factory()
        )
        with runtime_context:
            with self.torch.cuda.graph(graph, pool=shared_pool):
                yield

    def execute_capture_body(self, segment, *, ordinal: int) -> None:
        del ordinal
        hidden = self._hidden
        if getattr(segment, "include_embedding", False):
            hidden = self.model.embed_exact_graph_inputs(
                self.input_ids
            )
        prepared = self.model.run_exact_cuda_graph_layer_range(
            state_slot_ids=self.state_slot_ids,
            token_counts=self.token_counts,
            position_ids=self.positions,
            hidden_states=hidden,
            start_layer=segment.start_layer,
            end_layer=segment.end_layer,
        )
        self._hidden = prepared.hidden_states
        self._candidates.append(prepared.candidates)
        if getattr(segment, "include_final", False):
            self._captured_logits = (
                self.model.finalize_exact_cuda_graph_hidden(
                    self._hidden
                )
            )
        if getattr(segment, "include_commit", False):
            self.model.commit_exact_cuda_graph_candidates(
                self.state_slot_ids,
                tuple(self._candidates),
            )

    @staticmethod
    def resolve_pool(graph, *, shared_pool):
        return graph.pool() if shared_pool is None else shared_pool

    def pool_identity(self, pool, *, pool_mode: str) -> str:
        object_identity = id(pool)
        if object_identity not in self._pool_ordinals:
            self._pool_ordinals[object_identity] = len(
                self._pool_ordinals
            )
        pool_ordinal = self._pool_ordinals[object_identity]
        return _load_attribution_contract().canonical_sha256({
            "run_tag": self.run_tag,
            "rank": self.rank,
            "pool_mode": pool_mode,
            "pool_ordinal": pool_ordinal,
        })

    def segment_metadata(self, segment, *, ordinal: int) -> dict:
        del ordinal
        candidates = (
            self._candidates[-1] if self._candidates else ()
        )
        return {
            **layer_type_inventory(
                self.model,
                int(segment.start_layer),
                int(segment.end_layer),
            ),
            "candidate_tensor_count": _count_unique_tensors(
                candidates,
                torch_module=self.torch,
            ),
            "candidate_tensor_bytes": count_unique_tensor_bytes(
                candidates,
                torch_module=self.torch,
            ),
            "stable_hidden_candidate_logits_bytes": (
                count_unique_tensor_bytes(
                    (
                        self._hidden,
                        self._candidates,
                        self._captured_logits,
                    ),
                    torch_module=self.torch,
                )
            ),
            "collectives": {
                "available": False,
                "counts": {},
                "unavailable_reason": (
                    "existing_receipt_not_exposed"
                ),
            },
            "cuda_stream_identity": str(
                int(
                    self.torch.cuda.current_stream(
                        self.runner.kv_cache.device
                    ).cuda_stream
                )
            ),
        }

    def embed_inputs(self):
        self._validate_lease_manifest()
        return self.model.embed_exact_graph_inputs(
            self.input_ids
        )

    def run_eager_prefix(
        self,
        hidden,
        *,
        start_layer: int,
        end_layer: int,
    ):
        self._validate_lease_manifest()
        with (
            nullcontext()
            if self.runtime_context_factory is None
            else self.runtime_context_factory()
        ):
            prepared = self.model.run_exact_cuda_graph_layer_range(
                state_slot_ids=self.state_slot_ids,
                token_counts=self.token_counts,
                position_ids=self.positions,
                hidden_states=hidden,
                start_layer=start_layer,
                end_layer=end_layer,
            )
        return prepared.hidden_states

    def install_isolated_hidden(self, hidden) -> None:
        self._isolated_hidden = hidden
        self._hidden = hidden

    def gather_isolated_rows(
        self,
        local_results: list[dict],
    ) -> list[dict]:
        local_rows = [
            dict(row)
            for result in local_results
            for row in result.get("phase_rows", ())
        ]
        distributed = getattr(self.torch, "distributed", None)
        if (
            distributed is None
            or not distributed.is_available()
            or not distributed.is_initialized()
        ):
            raise RuntimeError(
                "Phase A1 TP4 gather is unavailable"
            )
        gathered = [None] * WORLD_SIZE
        distributed.all_gather_object(gathered, local_rows)
        if any(not isinstance(rows, list) for rows in gathered):
            raise RuntimeError(
                "Phase A1 isolated rank gather is incomplete"
            )
        flattened = [
            dict(row)
            for rows in gathered
            for row in rows
            if isinstance(row, dict)
        ]
        if {row.get("rank") for row in flattened} != set(
            range(WORLD_SIZE)
        ):
            raise RuntimeError(
                "Phase A1 isolated rank inventory is incomplete"
            )
        return flattened

    def run_eager(self):
        self._validate_lease_manifest()
        with (
            nullcontext()
            if self.runtime_context_factory is None
            else self.runtime_context_factory()
        ):
            logits = (
                self.model.run_exact_cuda_graph_step_by_pool_index(
                    self.state_slot_ids,
                    self.token_counts,
                    self.input_ids,
                    self.positions,
                )
            )
        self.synchronize()
        self._eager_result = {
            "logits": self.clone_logits(logits),
            "selected": self.model.snapshot_exact_cuda_graph_state(
                self.leases
            ),
        }
        return self._eager_result

    def capture_segment(self, ordinal: int):
        control = self._current_control
        ranges = control["ranges"]
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal < 0
            or ordinal >= len(ranges)
        ):
            raise ValueError("Phase A1 segment ordinal is invalid")
        self._validate_lease_manifest()
        start_layer, end_layer = ranges[ordinal]
        segment = _load_census_worker()._load_segment_contract().ExactGraphSegment(
            start_layer=start_layer,
            end_layer=end_layer,
            include_embedding=(
                control["kind"] == "stitched" and ordinal == 0
            ),
            include_final=(
                control["kind"] == "stitched"
                and ordinal == len(ranges) - 1
            ),
            include_commit=(
                control["kind"] == "stitched"
                and ordinal == len(ranges) - 1
            ),
        )
        shared_pool = None
        if (
            control["pool_mode"] == "shared"
            and self._captured_segments
        ):
            shared_pool = self._captured_segments[0].shared_pool
        captured = capture_attributed_segment(
            self,
            segment,
            ordinal=ordinal,
            control_id=control["control_id"],
            pool_mode=control["pool_mode"],
            shared_pool=shared_pool,
        )
        self._captured_segments.append(captured)
        self._active_graphs.append(captured.graph)
        return captured.graph

    def replay(self, graphs) -> dict:
        self._validate_lease_manifest()
        with (
            nullcontext()
            if self.runtime_context_factory is None
            else self.runtime_context_factory()
        ):
            for graph in graphs:
                graph.replay()
        self.synchronize()
        self._replay_result = {
            "logits": self.clone_logits(self._captured_logits),
            "selected": self.model.snapshot_exact_cuda_graph_state(
                self.leases
            ),
        }
        return self._replay_result

    def control_comparison(self, control: dict) -> dict:
        if self._eager_result is None or self._replay_result is None:
            raise RuntimeError(
                "Phase A1 control comparison is incomplete"
            )
        stitched = control.get("kind") == "stitched"
        eager_logits = self._eager_result["logits"]
        replay_logits = self._replay_result["logits"]
        exact_output = (
            eager_logits is None
            and replay_logits is None
        ) or (
            eager_logits is not None
            and replay_logits is not None
            and bool(self.torch.equal(eager_logits, replay_logits))
        )
        expected_selected = (
            self._eager_result["selected"]
            if stitched
            else self._selected_baseline
        )
        return {
            "exact_output": exact_output if stitched else None,
            "exact_output_applicable": stitched,
            "selected_state_exact": _state_equal(
                expected_selected,
                self._replay_result["selected"],
                torch_module=self.torch,
            ),
            "unselected_state_unchanged": _state_equal(
                self._unselected_baseline,
                self._snapshot_unselected(),
                torch_module=self.torch,
            ),
            "scratch_kv_restored": self.scratch_equal_to_s0(),
            "graph_reset": all(
                id(captured.graph) in self._reset_graph_ids
                for captured in self._captured_segments
            ),
        }

    def run_control(self, control: dict) -> dict:
        if (
            not isinstance(control, dict)
            or not isinstance(control.get("control_id"), str)
            or not isinstance(control.get("ranges"), tuple)
            or not control["ranges"]
        ):
            raise ValueError("Phase A1 control is invalid")
        self._current_control = dict(control)
        self._captured_segments = []
        self._active_graphs = []
        self._reset_graph_ids = set()
        self._graph_reset_durations_ns = {}
        self._post_capture_restore_ns = 0
        self._restore_call_count = 0
        self._synchronization_count = 0
        self._isolated_hidden = getattr(
            self,
            "_isolated_hidden",
            None,
        )
        self._eager_result = None
        self._replay_result = None
        lifecycle_started_ns = int(self.clock_ns())
        self.restore_control_baseline()
        sequence = run_scratch_forensic_sequence(
            self,
            segment_count=len(control["ranges"]),
        )
        comparison = self.control_comparison(control)
        program_lifecycle_ns = (
            int(self.clock_ns()) - lifecycle_started_ns
        )
        phase_rows = []
        for captured_ordinal, captured in enumerate(
            self._captured_segments
        ):
            finalized = finalize_captured_segment(
                captured,
                post_capture_restore_ns=(
                    self._post_capture_restore_ns
                    if captured_ordinal
                    == len(self._captured_segments) - 1
                    else 0
                ),
                graph_reset_ns=self._graph_reset_durations_ns.get(
                    id(captured.graph),
                    0,
                ),
                program_lifecycle_ns=program_lifecycle_ns,
            )
            row = {
                "row_id": (
                    f"{control['control_id']}:"
                    f"segment-{finalized.metadata['segment_ordinal']}:"
                    f"rank-{self.rank}"
                ),
                **finalized.metadata,
                **asdict(finalized.accounting),
                **comparison,
                "formal_route_row": control[
                    "formal_route_row"
                ],
            }
            phase_rows.append(row)
        scratch_rows = []
        for checkpoint_ordinal, row in enumerate(
            sequence["checkpoint_rows"]
        ):
            scratch_rows.append({
                "row_id": (
                    f"{control['control_id']}:"
                    f"checkpoint-{checkpoint_ordinal}:"
                    f"rank-{self.rank}"
                ),
                "control_id": control["control_id"],
                "source_revision": self.source_revision,
                "plan_sha256": self.plan_sha256,
                **row,
            })
        snapshot_cpu_ns = sum(
            int(row.get("scratch_snapshot_cpu_ns", 0))
            for row in scratch_rows
        )
        return {
            "control_id": control["control_id"],
            "ranges": control["ranges"],
            "pool_mode": control["pool_mode"],
            "formal_route_row": control["formal_route_row"],
            "allocated_delta_bytes": max(
                (
                    int(row["allocated_delta_bytes"])
                    for row in phase_rows
                ),
                default=0,
            ),
            "reserved_delta_bytes": max(
                (
                    int(row["reserved_delta_bytes"])
                    for row in phase_rows
                ),
                default=0,
            ),
            "phase_rows": phase_rows,
            "scratch_rows": scratch_rows,
            "benefit": {
                "first_scratch_divergence": (
                    first_scratch_divergence(scratch_rows)
                ),
                "restore_round_trip_exact": sequence[
                    "restore_round_trip_exact"
                ],
            },
            "cost": {
                "diagnostic_capture_count": len(phase_rows),
                "diagnostic_synchronization_count": (
                    self._synchronization_count
                ),
                "scratch_snapshot_cpu_ns": snapshot_cpu_ns,
                "peak_allocated_delta_bytes": max(
                    (
                        int(row["allocated_delta_bytes"])
                        for row in phase_rows
                    ),
                    default=0,
                ),
                "peak_reserved_delta_bytes": max(
                    (
                        int(row["reserved_delta_bytes"])
                        for row in phase_rows
                    ),
                    default=0,
                ),
            },
            **comparison,
        }

    def reset_graph(self, graph) -> None:
        identity = id(graph)
        if identity in self._reset_graph_ids:
            return
        started_ns = int(self.clock_ns())
        graph.reset()
        self._graph_reset_durations_ns[identity] = (
            int(self.clock_ns()) - started_ns
        )
        self._reset_graph_ids.add(identity)
        self._active_graphs = [
            candidate
            for candidate in self._active_graphs
            if candidate is not graph
        ]

    @staticmethod
    def clone_logits(logits):
        return None if logits is None else logits.clone()


@dataclass(frozen=True)
class _CapturedAttributionSegment:
    graph: object
    shared_pool: object
    pool_identity: str
    accounting: object
    metadata: dict


def layer_type_inventory(
    model,
    start_layer: int,
    end_layer: int,
) -> dict[str, int]:
    layers = model.layer_stack.layers
    if (
        isinstance(start_layer, bool)
        or not isinstance(start_layer, int)
        or isinstance(end_layer, bool)
        or not isinstance(end_layer, int)
        or start_layer < 0
        or end_layer <= start_layer
        or end_layer > len(layers)
    ):
        raise ValueError("layer inventory range is invalid")
    counts = {
        "linear_attention_layer_count": 0,
        "full_attention_layer_count": 0,
    }
    for layer in layers[start_layer:end_layer]:
        block_type = getattr(layer, "block_type", None)
        name = f"{block_type}_layer_count"
        if name not in counts:
            raise ValueError("layer inventory block type is invalid")
        counts[name] += 1
    return counts


def count_unique_tensor_bytes(value, *, torch_module) -> int:
    seen = set()

    def count(item) -> int:
        if isinstance(item, torch_module.Tensor):
            identity = id(item)
            if identity in seen:
                return 0
            seen.add(identity)
            return int(item.numel()) * int(item.element_size())
        if isinstance(item, dict):
            return sum(count(nested) for nested in item.values())
        if isinstance(item, (list, tuple)):
            return sum(count(nested) for nested in item)
        values = getattr(item, "__dict__", None)
        if isinstance(values, dict):
            return sum(count(nested) for nested in values.values())
        return 0

    return count(value)


def _count_unique_tensors(value, *, torch_module) -> int:
    seen = set()

    def count(item) -> int:
        if isinstance(item, torch_module.Tensor):
            identity = id(item)
            if identity in seen:
                return 0
            seen.add(identity)
            return 1
        if isinstance(item, dict):
            return sum(count(nested) for nested in item.values())
        if isinstance(item, (list, tuple)):
            return sum(count(nested) for nested in item)
        values = getattr(item, "__dict__", None)
        if isinstance(values, dict):
            return sum(count(nested) for nested in values.values())
        return 0

    return count(value)


def _memory_snapshot(value: object) -> dict[str, int]:
    if (
        not isinstance(value, dict)
        or any(
            isinstance(value.get(name), bool)
            or not isinstance(value.get(name), int)
            or value[name] < 0
            for name in ("allocated_bytes", "reserved_bytes")
        )
    ):
        raise ValueError("capture memory snapshot is invalid")
    return {
        "allocated_bytes": value["allocated_bytes"],
        "reserved_bytes": value["reserved_bytes"],
    }


def capture_attributed_segment(
    backend,
    segment,
    *,
    ordinal: int,
    control_id: str,
    pool_mode: str,
    shared_pool,
) -> _CapturedAttributionSegment:
    if pool_mode not in {"shared", "isolated"}:
        raise ValueError("capture pool mode is invalid")
    memory_before = _memory_snapshot(backend.memory_snapshot())
    segment_started_ns = int(backend.clock_ns())

    backend.prepare_segment(segment, ordinal=ordinal)
    prepared_ns = int(backend.clock_ns())

    graph = backend.create_graph()
    graph_created_ns = int(backend.clock_ns())

    capture_enter_started_ns = int(backend.clock_ns())
    with backend.capture_context(graph, shared_pool=shared_pool):
        capture_body_started_ns = int(backend.clock_ns())
        backend.execute_capture_body(segment, ordinal=ordinal)
        capture_body_finished_ns = int(backend.clock_ns())
    capture_context_exited_ns = int(backend.clock_ns())

    backend.synchronize()
    capture_synchronized_ns = int(backend.clock_ns())
    memory_after = _memory_snapshot(backend.memory_snapshot())
    segment_finished_ns = int(backend.clock_ns())

    resolved_pool = backend.resolve_pool(
        graph,
        shared_pool=shared_pool,
    )
    pool_identity = backend.pool_identity(
        resolved_pool,
        pool_mode=pool_mode,
    )
    accounting = _load_attribution_contract().CapturePhaseAccounting(
        snapshot_and_prepare_ns=prepared_ns - segment_started_ns,
        graph_object_create_ns=graph_created_ns - prepared_ns,
        capture_context_enter_ns=(
            capture_body_started_ns - capture_enter_started_ns
        ),
        capture_body_ns=(
            capture_body_finished_ns - capture_body_started_ns
        ),
        capture_context_exit_and_instantiate_ns=(
            capture_context_exited_ns - capture_body_finished_ns
        ),
        post_capture_synchronize_ns=(
            capture_synchronized_ns - capture_context_exited_ns
        ),
        post_capture_restore_ns=0,
        graph_reset_ns=0,
        segment_total_ns=segment_finished_ns - segment_started_ns,
        program_lifecycle_ns=segment_finished_ns - segment_started_ns,
    )
    metadata = {
        "control_id": control_id,
        "segment_ordinal": ordinal,
        "start_layer": int(segment.start_layer),
        "end_layer": int(segment.end_layer),
        "rank": int(backend.rank),
        "source_revision": backend.source_revision,
        "plan_sha256": backend.plan_sha256,
        "pool_mode": pool_mode,
        "pool_identity": pool_identity,
        "allocated_before_bytes": memory_before["allocated_bytes"],
        "allocated_after_bytes": memory_after["allocated_bytes"],
        "allocated_delta_bytes": max(
            0,
            memory_after["allocated_bytes"]
            - memory_before["allocated_bytes"],
        ),
        "reserved_before_bytes": memory_before["reserved_bytes"],
        "reserved_after_bytes": memory_after["reserved_bytes"],
        "reserved_delta_bytes": max(
            0,
            memory_after["reserved_bytes"]
            - memory_before["reserved_bytes"],
        ),
        **backend.segment_metadata(segment, ordinal=ordinal),
    }
    return _CapturedAttributionSegment(
        graph=graph,
        shared_pool=resolved_pool,
        pool_identity=pool_identity,
        accounting=accounting,
        metadata=metadata,
    )


def finalize_captured_segment(
    captured: _CapturedAttributionSegment,
    *,
    post_capture_restore_ns: int,
    graph_reset_ns: int,
    program_lifecycle_ns: int,
) -> _CapturedAttributionSegment:
    accounting = captured.accounting
    final_accounting = replace(
        accounting,
        post_capture_restore_ns=post_capture_restore_ns,
        graph_reset_ns=graph_reset_ns,
        segment_total_ns=(
            accounting.segment_total_ns
            + post_capture_restore_ns
            + graph_reset_ns
        ),
        program_lifecycle_ns=program_lifecycle_ns,
    )
    return replace(captured, accounting=final_accounting)


_P4_RANGES = (
    (0, 16),
    (16, 32),
    (32, 48),
    (48, 64),
)
MAX_ADDED_MEMORY_BYTES_PER_RANK = 512 * 1024 * 1024


def _control(
    control_id: str,
    ranges: tuple[tuple[int, int], ...],
    *,
    kind: str,
    pool_mode: str,
    formal_route_row: bool,
) -> dict:
    return {
        "control_id": control_id,
        "ranges": ranges,
        "kind": kind,
        "pool_mode": pool_mode,
        "formal_route_row": formal_route_row,
    }


def build_phase_a1_controls() -> tuple[dict, ...]:
    return (
        _control(
            "stitched_p4_repeat_0",
            _P4_RANGES,
            kind="stitched",
            pool_mode="shared",
            formal_route_row=True,
        ),
        _control(
            "stitched_p4_repeat_1",
            _P4_RANGES,
            kind="stitched",
            pool_mode="shared",
            formal_route_row=False,
        ),
        *(
            _control(
                f"isolated_{start}_{end}",
                ((start, end),),
                kind="isolated",
                pool_mode="isolated",
                formal_route_row=False,
            )
            for start, end in _P4_RANGES
        ),
    )


def select_pool_control_ranges(
    isolated_rows: list[dict],
) -> tuple[tuple[int, int], tuple[int, int]]:
    grouped = {}
    for row in isolated_rows:
        identity = (
            row.get("start_layer"),
            row.get("end_layer"),
        )
        grouped.setdefault(identity, []).append(row)
    if set(grouped) != set(_P4_RANGES):
        raise ValueError("isolated range inventory is incomplete")
    contract = _load_attribution_contract()
    maxima = {
        identity: contract.aggregate_tp4_phase_rows(rows)[
            "segment_total_ns"
        ]
        for identity, rows in grouped.items()
    }
    fastest = min(
        maxima,
        key=lambda identity: (maxima[identity], identity),
    )
    slowest = min(
        maxima,
        key=lambda identity: (-maxima[identity], identity),
    )
    return fastest, slowest


def build_pool_controls(
    fastest_range: tuple[int, int],
    slowest_range: tuple[int, int],
) -> tuple[dict, ...]:
    if (
        fastest_range not in _P4_RANGES
        or slowest_range not in _P4_RANGES
    ):
        raise ValueError("pool control range is invalid")
    return (
        _control(
            "pool_fastest_shared",
            (fastest_range,),
            kind="pool_control",
            pool_mode="shared",
            formal_route_row=False,
        ),
        _control(
            "pool_fastest_isolated",
            (fastest_range,),
            kind="pool_control",
            pool_mode="isolated",
            formal_route_row=False,
        ),
        _control(
            "pool_slowest_shared",
            (slowest_range,),
            kind="pool_control",
            pool_mode="shared",
            formal_route_row=False,
        ),
        _control(
            "pool_slowest_isolated",
            (slowest_range,),
            kind="pool_control",
            pool_mode="isolated",
            formal_route_row=False,
        ),
    )


def isolated_pool_memory_gate_pass(rows: list[dict]) -> bool:
    isolated = [
        row for row in rows if row.get("pool_mode") == "isolated"
    ]
    if not isolated:
        return False
    for row in isolated:
        for name in (
            "allocated_delta_bytes",
            "reserved_delta_bytes",
        ):
            value = row.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value > MAX_ADDED_MEMORY_BYTES_PER_RANK
            ):
                return False
    return True


def run_stitched_repeat(backend, *, repeat_ordinal: int) -> dict:
    if repeat_ordinal not in (0, 1):
        raise ValueError("stitched repeat ordinal is invalid")
    if repeat_ordinal == 1:
        backend.restore_control_baseline()
        backend.reset_control_graphs()
        backend.synchronize()
    control = build_phase_a1_controls()[repeat_ordinal]
    result = dict(backend.run_control(control))
    result["formal_route_row"] = control["formal_route_row"]
    return result


def run_isolated_range(
    backend,
    *,
    start_layer: int,
    end_layer: int,
    pool_mode: str,
    control_id: str,
) -> dict:
    if (start_layer, end_layer) not in _P4_RANGES:
        raise ValueError("isolated range is invalid")
    if pool_mode not in {"shared", "isolated"}:
        raise ValueError("isolated pool mode is invalid")
    backend.restore_control_baseline()
    hidden = backend.embed_inputs()
    eager_prefix_prepare_ns = 0
    if start_layer:
        prefix_started_ns = int(backend.clock_ns())
        hidden = backend.run_eager_prefix(
            hidden,
            start_layer=0,
            end_layer=start_layer,
        )
        backend.synchronize()
        eager_prefix_prepare_ns = (
            int(backend.clock_ns()) - prefix_started_ns
        )
    else:
        backend.synchronize()
    backend.install_isolated_hidden(hidden)
    control = _control(
        control_id,
        ((start_layer, end_layer),),
        kind=(
            "pool_control"
            if control_id.startswith("pool_")
            else "isolated"
        ),
        pool_mode=pool_mode,
        formal_route_row=False,
    )
    result = dict(backend.run_control(control))
    result["eager_prefix_prepare_ns"] = eager_prefix_prepare_ns
    return result


def run_phase_a1_matrix(backend) -> dict:
    controls = list(build_phase_a1_controls())
    results = [
        run_stitched_repeat(backend, repeat_ordinal=0),
        run_stitched_repeat(backend, repeat_ordinal=1),
    ]
    isolated_results = []
    for start_layer, end_layer in _P4_RANGES:
        result = run_isolated_range(
            backend,
            start_layer=start_layer,
            end_layer=end_layer,
            pool_mode="isolated",
            control_id=f"isolated_{start_layer}_{end_layer}",
        )
        isolated_results.append(result)
        results.append(result)
    gathered_rows = backend.gather_isolated_rows(isolated_results)
    fastest_range, slowest_range = select_pool_control_ranges(
        gathered_rows
    )
    pool_controls = build_pool_controls(
        fastest_range,
        slowest_range,
    )
    controls.extend(pool_controls)
    for control in pool_controls:
        (start_layer, end_layer), = control["ranges"]
        results.append(
            run_isolated_range(
                backend,
                start_layer=start_layer,
                end_layer=end_layer,
                pool_mode=control["pool_mode"],
                control_id=control["control_id"],
            )
        )
    return {
        "controls": tuple(controls),
        "control_results": results,
        "fastest_isolated_range": fastest_range,
        "slowest_isolated_range": slowest_range,
        "isolated_pool_memory_gate_pass": (
            isolated_pool_memory_gate_pass(results)
        ),
    }


def build_engine_config() -> dict:
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


def _ranked_results(
    local_result: object,
    acknowledgements: object,
) -> list[tuple[int, dict]]:
    if not isinstance(local_result, dict):
        raise RuntimeError("local rank result is missing")
    ranked = [(local_result.get("rank"), local_result)]
    try:
        ranked.extend(
            (ack.rank, ack.result)
            for ack in acknowledgements
        )
    except (AttributeError, TypeError) as error:
        raise RuntimeError("rank acknowledgements are invalid") from error
    if (
        sorted(rank for rank, _result in ranked)
        != list(range(WORLD_SIZE))
        or any(
            not isinstance(result, dict)
            or result.get("rank") != rank
            for rank, result in ranked
        )
    ):
        raise RuntimeError("rank inventory is incomplete")
    return sorted(ranked)


def collect_rank_results(
    local_result: object,
    acknowledgements: object,
) -> dict:
    ranked = _ranked_results(local_result, acknowledgements)
    contract = _load_attribution_contract()
    expected_control_ids = tuple(contract.CONTROL_IDS)
    rank_results = []
    phase_rows = []
    scratch_rows = []
    identities = set()
    phase_inventories = set()
    for rank, result in ranked:
        if (
            result.get("phase") != "A1"
            or result.get("complete") is not True
            or not isinstance(result.get("phase_rows"), list)
            or not isinstance(result.get("scratch_rows"), list)
            or not isinstance(result.get("benefit"), dict)
            or not isinstance(result.get("cost"), dict)
        ):
            raise RuntimeError("rank attribution result is incomplete")
        identity = (
            result.get("run_tag"),
            result.get("source_revision"),
            result.get("plan_sha256"),
        )
        if (
            not isinstance(identity[0], str)
            or not identity[0]
            or not SOURCE_REVISION_PATTERN.fullmatch(
                str(identity[1])
            )
            or not re.fullmatch(r"[0-9a-f]{64}", str(identity[2]))
        ):
            raise RuntimeError("rank attribution identity is invalid")
        identities.add(identity)
        control_ids = tuple(result.get("control_ids", ()))
        if control_ids != expected_control_ids:
            raise RuntimeError("rank control inventory is invalid")
        local_phase_rows = result["phase_rows"]
        local_scratch_rows = result["scratch_rows"]
        if (
            len(local_phase_rows) != 16
            or int(result["benefit"].get("attributed_segments", -1))
            != len(local_phase_rows)
            or int(
                result["cost"].get(
                    "diagnostic_capture_count",
                    -1,
                )
            )
            != len(local_phase_rows)
        ):
            raise RuntimeError("rank control inventory is invalid")
        phase_inventory = tuple(sorted(
            (
                row.get("control_id"),
                row.get("segment_ordinal"),
                row.get("start_layer"),
                row.get("end_layer"),
            )
            for row in local_phase_rows
        ))
        phase_inventories.add(phase_inventory)
        if {
            row.get("control_id") for row in local_phase_rows
        } != set(expected_control_ids):
            raise RuntimeError("rank control inventory is invalid")
        rank_results.append(dict(result))
        for row in local_phase_rows:
            if (
                not isinstance(row, dict)
                or row.get("rank") != rank
                or not isinstance(row.get("row_id"), str)
                or row.get("source_revision") != identity[1]
                or row.get("plan_sha256") != identity[2]
            ):
                raise RuntimeError(
                    "rank phase row identity is invalid"
                )
            phase_rows.append(dict(row))
        grouped_scratch = {
            control_id: [] for control_id in expected_control_ids
        }
        for row in local_scratch_rows:
            if (
                not isinstance(row, dict)
                or row.get("rank") != rank
                or not isinstance(row.get("row_id"), str)
                or row.get("source_revision") != identity[1]
                or row.get("plan_sha256") != identity[2]
                or row.get("control_id") not in grouped_scratch
            ):
                raise RuntimeError(
                    "rank scratch row identity is invalid"
                )
            grouped_scratch[row["control_id"]].append(row)
            scratch_rows.append(dict(row))
        segment_counts = {
            control_id: sum(
                row.get("control_id") == control_id
                for row in local_phase_rows
            )
            for control_id in expected_control_ids
        }
        for control_id, rows in grouped_scratch.items():
            checkpoints = tuple(
                row.get("checkpoint") for row in rows
            )
            expected = (
                "S0",
                "S1",
                "S2",
                *(("S3",) * segment_counts[control_id]),
                "S4",
                "S5",
                "S6",
                "S7",
            )
            ordinals = tuple(
                row.get("segment_ordinal")
                for row in rows
                if row.get("checkpoint") == "S3"
            )
            if (
                checkpoints != expected
                or ordinals
                != tuple(range(segment_counts[control_id]))
            ):
                raise RuntimeError(
                    "rank scratch checkpoint inventory is invalid"
                )
    if len(identities) != 1 or len(phase_inventories) != 1:
        raise RuntimeError("rank attribution identity disagrees")
    row_ids = [
        row["row_id"] for row in phase_rows + scratch_rows
    ]
    if len(row_ids) != len(set(row_ids)):
        raise RuntimeError("worker row identity is duplicated")
    benefit_rows = [result["benefit"] for result in rank_results]
    cost_rows = [result["cost"] for result in rank_results]
    divergence_values = {
        row.get("first_scratch_divergence")
        for row in benefit_rows
    }
    restore_values = {
        row.get("restore_round_trip_exact")
        for row in benefit_rows
    }
    run_tag, source_revision, plan_sha256 = next(iter(identities))
    return {
        "rank_results": rank_results,
        "phase_rows": phase_rows,
        "scratch_rows": scratch_rows,
        "worker_summary": {
            "schema_version": WORKER_SCHEMA,
            "phase": "A1",
            "run_tag": run_tag,
            "source_revision": source_revision,
            "plan_sha256": plan_sha256,
            "control_ids": expected_control_ids,
            "complete": True,
            "benefit": {
                "attributed_segments": max(
                    int(row["attributed_segments"])
                    for row in benefit_rows
                ),
                "first_scratch_divergence": (
                    next(iter(divergence_values))
                    if len(divergence_values) == 1
                    else "RANK_DISAGREEMENT"
                ),
                "restore_round_trip_exact": (
                    next(iter(restore_values))
                    if len(restore_values) == 1
                    else False
                ),
            },
            "cost": {
                name: max(int(row[name]) for row in cost_rows)
                for name in (
                    "diagnostic_capture_count",
                    "diagnostic_synchronization_count",
                    "total_worker_duration_ns",
                    "scratch_snapshot_cpu_ns",
                    "peak_allocated_delta_bytes",
                    "peak_reserved_delta_bytes",
                )
            },
        },
    }


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


def _load_census_worker():
    module_name = "_tinyllmforge_segmented_census_worker_reuse"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    module_path = (
        Path(__file__).resolve().parent
        / "tp4_segmented_capture_census_worker.py"
    )
    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("segmented census worker cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _default_engine_factory(model_root, **kwargs):
    os.environ[RUNTIME_ENV] = "1"
    _install_runtime_model_runner()
    from tinyvllm.engine.llm_engine import LLMEngine

    census = _load_census_worker()
    engine, _port = census.create_engine_with_rendezvous_retry(
        model_root,
        engine_config=kwargs,
        port_factory=census._free_rendezvous_port,
        engine_factory=lambda root, **config: LLMEngine(
            str(root),
            **config,
        ),
    )
    return engine


def _default_workload_runner(engine) -> None:
    _load_census_worker()._default_workload_runner(engine)


def run_phase_a1(
    *,
    model_root,
    run_tag: str,
    source_revision: str,
    timeout_s: float,
    engine_factory=None,
    workload_runner=None,
) -> dict:
    if not isinstance(run_tag, str) or not run_tag:
        raise ValueError("run_tag must be non-empty")
    if not SOURCE_REVISION_PATTERN.fullmatch(source_revision):
        raise ValueError("source_revision must be a committed SHA")
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
    collected = None
    cleanup = None
    try:
        engine = engine_factory(
            model_root,
            **build_engine_config(),
        )
        local, acknowledgements = (
            engine.call_model_runner_acknowledged(
                "arm_segmented_capture_attribution",
                run_tag,
                source_revision,
                timeout_s=float(timeout_s),
            )
        )
        armed = _ranked_results(local, acknowledgements)
        if any(
            result.get("armed") is not True
            for _rank, result in armed
        ):
            raise RuntimeError(
                "segmented capture attribution arming is incomplete"
            )
        workload_runner(engine)
        local, acknowledgements = (
            engine.call_model_runner_acknowledged(
                "segmented_capture_attribution_result",
                timeout_s=float(timeout_s),
            )
        )
        collected = collect_rank_results(local, acknowledgements)
    except BaseException as error:
        operation_error = error

    cleanup_error = None
    if engine is not None:
        try:
            cleanup = _validate_engine_cleanup(engine.exit())
        except BaseException as error:
            cleanup_error = error
    if operation_error is not None or cleanup_error is not None:
        primary = (
            operation_error
            if operation_error is not None
            else cleanup_error
        )
        partial = (
            collected
            if isinstance(collected, dict)
            else {
                "rank_results": [],
                "phase_rows": [],
                "scratch_rows": [],
            }
        )
        result = {
            "schema_version": WORKER_SCHEMA,
            "phase": "A1",
            "run_tag": run_tag,
            **partial,
            "process_receipts": (
                cleanup
                if isinstance(cleanup, dict)
                else {
                    "complete": False,
                    "error": (
                        None
                        if cleanup_error is None
                        else {
                            "type": type(cleanup_error).__name__,
                            "message": str(cleanup_error),
                        }
                    ),
                }
            ),
            "worker_summary": {
                "schema_version": WORKER_SCHEMA,
                "phase": "A1",
                "run_tag": run_tag,
                "source_revision": source_revision,
                "complete": False,
                "first_operational_error": {
                    "type": type(primary).__name__,
                    "message": str(primary),
                },
                "cleanup_error": (
                    None
                    if cleanup_error is None
                    else {
                        "type": type(cleanup_error).__name__,
                        "message": str(cleanup_error),
                    }
                ),
            },
        }
        failure = PhaseA1WorkerError(
            str(primary),
            result=result,
        )
        if operation_error is not None and cleanup_error is not None:
            raise failure from cleanup_error
        raise failure from primary
    return {
        "schema_version": WORKER_SCHEMA,
        "phase": "A1",
        "run_tag": run_tag,
        **collected,
        "process_receipts": cleanup,
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


def write_worker_artifacts(output_root: Path, result: dict) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(
        output_root / "phase_rows.jsonl",
        result["phase_rows"],
    )
    _atomic_write_jsonl(
        output_root / "scratch_rows.jsonl",
        result["scratch_rows"],
    )
    for name in (
        "rank_results",
        "process_receipts",
        "worker_summary",
    ):
        _atomic_write_json(
            output_root / f"{name}.json",
            result[name],
        )


def execute_runtime_phase_a1(
    runner,
    *,
    run_tag: str,
    source_revision: str,
    input_ids,
    positions,
    torch_module=None,
    context=None,
    temporary_context=None,
    backend_factory=None,
    matrix_runner=None,
    clock_ns=None,
) -> dict:
    if torch_module is None:
        torch_module = __import__("torch")
    if context is None or temporary_context is None:
        from tinyvllm.utils.context import (
            get_context,
            temporary_context as runtime_temporary_context,
        )

        if context is None:
            context = get_context()
        if temporary_context is None:
            temporary_context = runtime_temporary_context
    if backend_factory is None:
        backend_factory = _AttributionCudaBackend
    if matrix_runner is None:
        matrix_runner = run_phase_a1_matrix
    if clock_ns is None:
        clock_ns = time.perf_counter_ns
    if (
        int(runner.world_size) != WORLD_SIZE
        or int(input_ids.shape[0]) != 8
    ):
        raise RuntimeError(
            "Phase A1 requires the frozen TP4 batch of eight"
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
            "Phase A1 has insufficient scratch KV blocks"
        )
    scratch_slots = [
        block_id * int(runner.block_size)
        for block_id in available_blocks[:8]
    ]

    leases = tuple(runner._last_hybrid_state_leases)
    request_ids = tuple(runner._last_hybrid_state_request_ids)
    token_counts = tuple(runner._last_hybrid_state_token_counts)
    manifest_builder = getattr(
        runner.model,
        "exact_cuda_graph_lease_manifest",
        None,
    )
    if callable(manifest_builder):
        manifest = manifest_builder(leases, request_ids)
        slot_ids = tuple(int(value) for value in manifest.slot_ids)
    else:
        slot_ids = tuple(int(value) for value in leases)
    state_slot_ids = torch_module.tensor(
        slot_ids,
        dtype=torch_module.int64,
        device=runner.kv_cache.device,
    )

    def runtime_context_factory():
        arguments = {
            "slot_mapping": torch_module.tensor(
                scratch_slots,
                dtype=torch_module.int32,
                device=runner.kv_cache.device,
            ),
        }
        for name in (
            "context_lens",
            "block_tables",
            "flash_attn_num_splits",
        ):
            if hasattr(context, name):
                arguments[name] = getattr(context, name)
        arguments["force_attention_backend"] = True
        return temporary_context(**arguments)

    started_ns = int(clock_ns())
    backend = backend_factory(
        runner,
        scratch_slots=scratch_slots,
        run_tag=run_tag,
        rank=int(runner.rank),
        torch_module=torch_module,
        input_ids=input_ids,
        positions=positions,
        state_slot_ids=state_slot_ids,
        token_counts=token_counts,
        runtime_context_factory=runtime_context_factory,
        source_revision=source_revision,
        plan_sha256=_load_attribution_contract().canonical_sha256(
            build_phase_a1_controls()
        ),
        clock_ns=clock_ns,
    )
    matrix = matrix_runner(backend)
    control_results = matrix["control_results"]
    phase_rows = [
        row
        for result in control_results
        for row in result.get("phase_rows", [])
    ]
    scratch_rows = [
        row
        for result in control_results
        for row in result.get("scratch_rows", [])
    ]
    benefit_rows = [
        result.get("benefit", {}) for result in control_results
    ]
    cost_rows = [
        result.get("cost", {}) for result in control_results
    ]
    divergences = [
        row.get("first_scratch_divergence")
        for row in benefit_rows
        if row.get("first_scratch_divergence") is not None
    ]
    return {
        "phase": "A1",
        "rank": int(runner.rank),
        "run_tag": run_tag,
        "source_revision": source_revision,
        "plan_sha256": backend.plan_sha256,
        "control_ids": tuple(
            control["control_id"]
            for control in matrix["controls"]
        ),
        "complete": True,
        "phase_rows": phase_rows,
        "scratch_rows": scratch_rows,
        "benefit": {
            "attributed_segments": len(phase_rows),
            "first_scratch_divergence": (
                divergences[0] if divergences else None
            ),
            "restore_round_trip_exact": all(
                row.get("restore_round_trip_exact") is True
                for row in benefit_rows
            ),
        },
        "cost": {
            "diagnostic_capture_count": sum(
                int(row.get("diagnostic_capture_count", 0))
                for row in cost_rows
            ),
            "diagnostic_synchronization_count": sum(
                int(row.get("diagnostic_synchronization_count", 0))
                for row in cost_rows
            ),
            "total_worker_duration_ns": (
                int(clock_ns()) - started_ns
            ),
            "scratch_snapshot_cpu_ns": sum(
                int(row.get("scratch_snapshot_cpu_ns", 0))
                for row in cost_rows
            ),
            "peak_allocated_delta_bytes": max(
                (
                    int(row.get("peak_allocated_delta_bytes", 0))
                    for row in cost_rows
                ),
                default=0,
            ),
            "peak_reserved_delta_bytes": max(
                (
                    int(row.get("peak_reserved_delta_bytes", 0))
                    for row in cost_rows
                ),
                default=0,
            ),
        },
        "isolated_pool_memory_gate_pass": matrix[
            "isolated_pool_memory_gate_pass"
        ],
    }


class _SegmentedAttributionModelRunnerMixin:
    def arm_segmented_capture_attribution(
        self,
        run_tag: str,
        source_revision: str,
    ) -> dict:
        if not isinstance(run_tag, str) or not run_tag:
            raise ValueError("run_tag must be non-empty")
        if not SOURCE_REVISION_PATTERN.fullmatch(source_revision):
            raise ValueError("source_revision must be a committed SHA")
        self._segmented_attribution_run_tag = run_tag
        self._segmented_attribution_source_revision = source_revision
        self._segmented_attribution_result = None
        return {"rank": int(self.rank), "armed": True}

    def segmented_capture_attribution_result(self) -> dict:
        result = getattr(
            self,
            "_segmented_attribution_result",
            None,
        )
        if not isinstance(result, dict):
            raise RuntimeError(
                "segmented capture attribution did not execute"
            )
        return dict(result)

    def run_model(
        self,
        input_ids,
        positions,
        is_prefill,
        *args,
        **kwargs,
    ):
        run_tag = getattr(
            self,
            "_segmented_attribution_run_tag",
            None,
        )
        if (
            not is_prefill
            and run_tag is not None
            and getattr(
                self,
                "_segmented_attribution_result",
                None,
            )
            is None
        ):
            torch_module = __import__("torch")
            with torch_module.inference_mode():
                self._segmented_attribution_result = (
                    execute_runtime_phase_a1(
                        self,
                        run_tag=run_tag,
                        source_revision=(
                            self._segmented_attribution_source_revision
                        ),
                        input_ids=input_ids,
                        positions=positions,
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
    existing = globals().get("SegmentedAttributionModelRunner")
    if existing is not None:
        return existing
    from tinyvllm.engine.model_runner import ModelRunner
    import tinyvllm.engine.llm_engine as llm_engine_module

    runtime_class = type(
        "SegmentedAttributionModelRunner",
        (_SegmentedAttributionModelRunnerMixin, ModelRunner),
        {"__module__": __name__},
    )
    globals()["SegmentedAttributionModelRunner"] = runtime_class
    llm_engine_module.ModelRunner = runtime_class
    return runtime_class


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        result = run_phase_a1(
            model_root=args.model_root,
            run_tag=args.run_tag,
            source_revision=args.source_revision,
            timeout_s=args.timeout_s,
        )
    except PhaseA1WorkerError as error:
        write_worker_artifacts(args.output_root, error.result)
        print(json.dumps(
            {
                "schema_version": WORKER_SCHEMA,
                "classification": "WORKER_INCOMPLETE",
                "error": str(error),
            },
            sort_keys=True,
            allow_nan=False,
        ))
        return 1
    write_worker_artifacts(args.output_root, result)
    print(json.dumps(
        {
            "schema_version": WORKER_SCHEMA,
            "classification": "WORKER_COMPLETE",
            "phase_row_count": len(result["phase_rows"]),
            "scratch_row_count": len(result["scratch_rows"]),
        },
        sort_keys=True,
        allow_nan=False,
    ))
    return 0


if os.environ.get(RUNTIME_ENV) == "1":
    _install_runtime_model_runner()


if __name__ == "__main__":
    raise SystemExit(main())
