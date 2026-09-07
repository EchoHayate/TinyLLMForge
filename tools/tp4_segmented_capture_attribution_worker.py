#!/usr/bin/env python3
"""Diagnostic primitives for TP4 segmented-capture attribution."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


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
        self.input_ids = input_ids
        self.positions = positions
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

    def initialize_sentinel(self) -> None:
        fill_scratch_sentinel(
            self.runner,
            self.scratch_slots,
            run_tag=self.run_tag,
            rank=self.rank,
            torch_module=self.torch,
        )

    def checkpoint(
        self,
        checkpoint: str,
        *,
        segment_ordinal: int | None,
    ) -> dict:
        self.torch.cuda.synchronize()
        snapshot = self.runner.snapshot_kv_slots(self.scratch_slots)
        if checkpoint == "S0":
            self._s0 = {
                name: tensor.clone()
                for name, tensor in snapshot.items()
            }
        if self._s0 is None:
            raise RuntimeError("S0 scratch snapshot is missing")
        return _checkpoint_from_snapshot(
            snapshot,
            checkpoint=checkpoint,
            rank=self.rank,
            s0=self._s0,
            synchronized=True,
            segment_ordinal=segment_ordinal,
            torch_module=self.torch,
        )

    def restore_s0(self) -> None:
        if self._s0 is None:
            raise RuntimeError("S0 scratch snapshot is missing")
        self.runner.restore_kv_slots(
            self.scratch_slots,
            self._s0,
        )

    def scratch_equal_to_s0(self) -> bool:
        if self._s0 is None:
            raise RuntimeError("S0 scratch snapshot is missing")
        self.torch.cuda.synchronize()
        current = self.runner.snapshot_kv_slots(self.scratch_slots)
        return all(
            bool(self.torch.equal(current[name], self._s0[name]))
            for name in ("keys", "values")
        )

    def synchronize(self) -> None:
        self.torch.cuda.synchronize()

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
            self._hidden = None
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

    @staticmethod
    def reset_graph(graph) -> None:
        graph.reset()

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
