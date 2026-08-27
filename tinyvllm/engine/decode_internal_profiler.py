from __future__ import annotations

from contextlib import contextmanager, ExitStack, nullcontext
from contextvars import ContextVar
import copy
import sys
import time


_ACTIVE_PROFILER = ContextVar(
    "tinyvllm_decode_internal_profiler",
    default=None,
)

LAYER_ROLES = frozenset({
    "linear_attention",
    "full_attention",
    "mlp",
    "normalization",
    "residual",
    "embedding",
    "output_head",
})

OPERATION_CLASSES = frozenset({
    "gemm",
    "attention",
    "recurrent",
    "collective",
    "memory",
    "other_compute",
})


def active_decode_internal_profiler():
    return _ACTIVE_PROFILER.get()


def _synchronous_collective_census_module():
    return (
        sys.modules.get(
            "tinyvllm.engine.synchronous_collective_census"
        )
        or sys.modules.get("synchronous_collective_census")
    )


def _active_synchronous_collective_census():
    module = _synchronous_collective_census_module()
    if module is None:
        return None
    return module.active_synchronous_collective_census()


def _active_model_runner_command_trace():
    module = sys.modules.get(
        "tinyvllm.engine.model_runner_command_timeline"
    )
    if module is None:
        return None
    return module.active_model_runner_command_trace()


def _active_cuda_stream_identity():
    import torch

    stream = torch.cuda.current_stream()
    device_index = stream.device.index
    return (
        f"cuda:{device_index}:stream:"
        f"{int(stream.cuda_stream)}"
    )


def profile_layer(layer_index, layer_role):
    stack = ExitStack()
    profiler = active_decode_internal_profiler()
    if profiler is not None:
        stack.enter_context(
            profiler.layer(layer_index, layer_role)
        )
    module = _synchronous_collective_census_module()
    if module is not None:
        stack.enter_context(
            module.census_layer(layer_index, layer_role)
        )
    return stack


def profile_operation(operation_class, operation_name, *, tensor=None):
    profiler = active_decode_internal_profiler()
    if profiler is None or profiler._active_layer is None:
        return nullcontext()
    return profiler.operation(
        operation_class,
        operation_name,
        tensor=tensor,
    )


def profile_collective(
    operation,
    tensor,
    call,
    *,
    site_role=None,
    collective_kind=None,
    process_group=None,
    execution_phase="decode",
    async_mode=False,
    source_rank=None,
    destination_rank=None,
    source_stream=None,
    completion_stream=None,
):
    if not isinstance(async_mode, bool) or async_mode:
        raise ValueError("async_mode must be False")
    profiler = active_decode_internal_profiler()

    def profiled_call(value):
        if profiler is None:
            return call(value)
        with profiler.collective(
            operation,
            value,
            collective_kind=collective_kind,
            process_group=process_group,
            async_mode=async_mode,
            source_stream=source_stream,
            completion_stream=completion_stream,
        ):
            return call(value)

    module = _synchronous_collective_census_module()
    if module is None:
        return profiled_call(tensor)
    return module.observe_synchronous_collective(
        site_role=site_role,
        operation=operation,
        tensor=tensor,
        call=profiled_call,
        collective_kind=collective_kind,
        process_group=process_group,
        execution_phase=execution_phase,
        async_mode=async_mode,
        source_rank=source_rank,
        destination_rank=destination_rank,
    )


def run_profiled_step(
    profiler,
    *,
    batch_kind,
    is_decode,
    active_sequence_count,
    request_set_sha256,
    dispatch,
    call,
):
    profiler.begin_step(
        batch_kind=batch_kind,
        is_decode=is_decode,
        active_sequence_count=active_sequence_count,
        request_set_sha256=request_set_sha256,
        dispatch=dispatch,
    )
    census = _active_synchronous_collective_census()
    if census is not None:
        try:
            census.begin_step(
                batch_kind=batch_kind,
                is_decode=is_decode,
                active_sequence_count=active_sequence_count,
                request_set_sha256=request_set_sha256,
                dispatch=dispatch,
            )
        except BaseException:
            profiler.end_step()
            raise
    status = "completed"
    try:
        return call()
    except BaseException:
        status = "failed"
        raise
    finally:
        try:
            if census is not None:
                census.end_step(status=status)
        finally:
            profiler.end_step()


class DecodeInternalProfiler:

    def __init__(
        self,
        *,
        rank,
        clock_ns=time.perf_counter_ns,
        event_factory=None,
        synchronize=None,
        nvtx_range_factory=None,
        stream_resolver=None,
        profile_label=None,
        active_command_trace=None,
        enabled=True,
    ):
        self.rank = int(rank)
        self.enabled = bool(enabled)
        self._clock_ns = clock_ns
        self._event_factory = event_factory
        self._synchronize = synchronize
        self._nvtx_range_factory = (
            nvtx_range_factory
            if callable(nvtx_range_factory)
            else lambda _label: nullcontext()
        )
        self._stream_resolver = (
            stream_resolver
            if callable(stream_resolver)
            else _active_cuda_stream_identity
        )
        self._profile_label = (
            str(profile_label).strip("/")
            if profile_label
            else f"rank={self.rank}"
        )
        self._profile_identity = self._parse_profile_identity(
            self._profile_label
        )
        self._active_command_trace = (
            active_command_trace
            if callable(active_command_trace)
            else _active_model_runner_command_trace
        )
        self._active_step = None
        self._active_layer = None
        self._active_operation = None
        self._active_token = None
        self._pending_steps = []
        self._pending_layers = []
        self._pending_operations = []
        self._pending_collectives = []
        self._layer_identities = set()
        self._operation_ordinal = 0
        self._decode_ordinal = 0
        self._decode_request_set_sha256 = None
        self._finalized = None
        if self.enabled and (
            not callable(self._event_factory)
            or not callable(self._synchronize)
        ):
            raise ValueError(
                "enabled profiler requires CUDA event and synchronize hooks"
            )

    @classmethod
    def disabled(cls, *, rank):
        return cls(
            rank=rank,
            enabled=False,
        )

    def _require_mutable(self):
        if self._finalized is not None:
            raise RuntimeError("decode internal profiler is finalized")

    @staticmethod
    def _parse_profile_identity(profile_label):
        identity = {
            "attempt": None,
            "workload": None,
            "repetition": None,
        }
        for component in profile_label.split("/"):
            key, separator, value = component.partition("=")
            if not separator or key not in identity:
                continue
            if identity[key] is not None:
                raise ValueError(
                    f"profile label has duplicate {key}"
                )
            if not value:
                raise ValueError(
                    f"profile label {key} must be non-empty"
                )
            if key == "repetition":
                try:
                    value = int(value)
                except ValueError as error:
                    raise ValueError(
                        "profile label repetition must be an integer"
                    ) from error
                if value < 0:
                    raise ValueError(
                        "profile label repetition must be non-negative"
                    )
            identity[key] = value
        return identity

    def begin_step(
        self,
        *,
        batch_kind,
        is_decode,
        active_sequence_count,
        request_set_sha256,
        dispatch,
    ):
        if not self.enabled:
            return
        self._require_mutable()
        if self._active_step is not None:
            raise RuntimeError("decode internal profiler step is active")
        if not isinstance(is_decode, bool):
            raise ValueError("is_decode must be a boolean")
        if (
            is_decode
            and request_set_sha256
            != self._decode_request_set_sha256
        ):
            self._decode_ordinal = 0
            self._decode_request_set_sha256 = request_set_sha256
        decode_ordinal = self._decode_ordinal if is_decode else None
        if is_decode:
            self._decode_ordinal += 1
        step_kind = (
            "decode_first"
            if is_decode and decode_ordinal == 0
            else "decode_steady"
            if is_decode
            else "prefill"
        )
        nvtx_context = self._nvtx_range_factory(
            f"decode_internal/{self._profile_label}/{step_kind}"
        )
        nvtx_context.__enter__()
        start_event = self._event_factory()
        start_event.record()
        command_trace = self._active_command_trace()
        command_request_set_sha256 = (
            None
            if command_trace is None
            else command_trace.request_set_sha256
        )
        self._active_step = {
            "rank": self.rank,
            "step_index": len(self._pending_steps),
            "batch_kind": batch_kind,
            "is_decode": is_decode,
            "decode_ordinal": decode_ordinal,
            "active_sequence_count": int(active_sequence_count),
            "request_set_sha256": (
                command_request_set_sha256
                if command_request_set_sha256 is not None
                else request_set_sha256
            ),
            "dispatch": dispatch,
            "command_id": (
                None
                if command_trace is None
                else command_trace.command_id
            ),
            "engine_step_id": (
                None
                if command_trace is None
                else command_trace.engine_step_id
            ),
            "repeat_index": (
                None
                if command_trace is None
                else command_trace.repeat_index
            ),
            "speculative_selected_sequence_ids_sha256": (
                None
                if command_trace is None
                else (
                    command_trace
                    .speculative_selected_sequence_ids_sha256
                )
            ),
            "_wall_start_ns": self._clock_ns(),
            "_cuda_start": start_event,
            "_nvtx_context": nvtx_context,
        }
        self._operation_ordinal = 0
        self._active_token = _ACTIVE_PROFILER.set(self)

    def end_step(self):
        if not self.enabled:
            return
        self._require_mutable()
        if self._active_step is None:
            raise RuntimeError("decode internal profiler has no active step")
        if (
            self._active_layer is not None
            or self._active_operation is not None
        ):
            raise RuntimeError(
                "cannot end step with an open profiler scope"
            )
        end_event = self._event_factory()
        end_event.record()
        row = self._active_step
        row["_wall_end_ns"] = self._clock_ns()
        row["_cuda_end"] = end_event
        self._pending_steps.append(row)
        self._active_step = None
        if self._active_token is not None:
            _ACTIVE_PROFILER.reset(self._active_token)
            self._active_token = None
        row["_nvtx_context"].__exit__(None, None, None)

    @contextmanager
    def layer(self, layer_index, layer_role):
        if (
            not self.enabled
            or self._active_step is None
            or not self._active_step["is_decode"]
        ):
            if self.enabled and self._active_step is None:
                self._require_mutable()
                raise RuntimeError(
                    "layer scope requires an active step"
                )
            yield
            return
        self._require_mutable()
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index < 0
        ):
            raise ValueError(
                "layer index must be a non-negative integer"
            )
        if layer_role not in LAYER_ROLES:
            raise ValueError(
                f"unsupported layer role: {layer_role!r}"
            )
        if self._active_layer is not None:
            raise RuntimeError("a layer scope is active")
        if self._active_operation is not None:
            raise RuntimeError("an operation scope is active")
        identity = (
            self._active_step["step_index"],
            layer_index,
            layer_role,
        )
        if identity in self._layer_identities:
            raise RuntimeError("duplicate layer identity")
        self._layer_identities.add(identity)
        wall_start_ns = self._clock_ns()
        start_event = self._event_factory()
        start_event.record()
        nvtx_context = self._nvtx_range_factory(
            f"decode_internal/{self._profile_label}/"
            f"layer/{layer_index}/{layer_role}"
        )
        nvtx_context.__enter__()
        row = self._step_identity() | {
            "layer_index": layer_index,
            "layer_role": layer_role,
            "_wall_start_ns": wall_start_ns,
            "_cuda_start": start_event,
            "_nvtx_context": nvtx_context,
        }
        self._active_layer = row
        try:
            yield
        finally:
            if self._active_operation is not None:
                raise RuntimeError(
                    "cannot close layer with an open operation scope"
                )
            if self._active_layer is not row:
                raise RuntimeError("mismatched layer scope exit")
            end_event = self._event_factory()
            end_event.record()
            wall_end_ns = self._clock_ns()
            row["_wall_end_ns"] = wall_end_ns
            row["_cuda_end"] = end_event
            self._pending_layers.append(row)
            self._active_layer = None
            nvtx_context.__exit__(None, None, None)

    def _step_identity(self):
        return self._profile_identity | {
            name: self._active_step[name]
            for name in (
                "rank",
                "step_index",
                "decode_ordinal",
                "command_id",
                "engine_step_id",
                "repeat_index",
                "request_set_sha256",
                "speculative_selected_sequence_ids_sha256",
            )
        }

    @staticmethod
    def _tensor_identity(tensor):
        if tensor is None:
            return {
                "tensor_shape": None,
                "tensor_dtype": None,
            }
        return {
            "tensor_shape": [
                int(value) for value in tensor.shape
            ],
            "tensor_dtype": str(tensor.dtype),
        }

    def _start_operation(
        self,
        operation_class,
        operation_name,
        tensor,
        *,
        allow_missing_layer=False,
        source_stream=None,
    ):
        self._require_mutable()
        if self._active_step is None:
            raise RuntimeError(
                "operation scope requires an active step"
            )
        if operation_class not in OPERATION_CLASSES:
            raise ValueError(
                f"unsupported operation class: {operation_class!r}"
            )
        if (
            not isinstance(operation_name, str)
            or not operation_name
        ):
            raise ValueError(
                "operation name must be a non-empty string"
            )
        if self._active_layer is None and not allow_missing_layer:
            raise RuntimeError(
                "operation scope requires an active layer"
            )
        if self._active_operation is not None:
            raise RuntimeError("an operation scope is active")
        operation_ordinal = self._operation_ordinal
        self._operation_ordinal += 1
        wall_start_ns = self._clock_ns()
        start_event = self._event_factory()
        start_event.record()
        resolved_source_stream = (
            self._stream_resolver()
            if source_stream is None
            else source_stream
        )
        nvtx_context = self._nvtx_range_factory(
            f"decode_internal/{self._profile_label}/"
            f"operation/{operation_ordinal}/"
            f"{operation_class}/{operation_name}"
        )
        nvtx_context.__enter__()
        row = self._step_identity() | {
            "layer_index": (
                None
                if self._active_layer is None
                else self._active_layer["layer_index"]
            ),
            "layer_role": (
                None
                if self._active_layer is None
                else self._active_layer["layer_role"]
            ),
            "operation_ordinal": operation_ordinal,
            "operation_class": operation_class,
            "operation_name": operation_name,
            **self._tensor_identity(tensor),
            "source_stream": resolved_source_stream,
            "_wall_start_ns": wall_start_ns,
            "_cuda_start": start_event,
            "_nvtx_context": nvtx_context,
        }
        self._active_operation = row
        return row

    def _finish_operation(self, row, *, completion_stream=None):
        if self._active_operation is not row:
            raise RuntimeError("mismatched operation scope exit")
        end_event = self._event_factory()
        end_event.record()
        wall_end_ns = self._clock_ns()
        row["_wall_end_ns"] = wall_end_ns
        row["_cuda_end"] = end_event
        row["completion_stream"] = (
            self._stream_resolver()
            if completion_stream is None
            else completion_stream
        )
        self._pending_operations.append(row)
        self._active_operation = None
        row["_nvtx_context"].__exit__(None, None, None)

    @contextmanager
    def operation(
        self,
        operation_class,
        operation_name,
        *,
        tensor=None,
    ):
        if (
            not self.enabled
            or (
                self._active_step is not None
                and not self._active_step["is_decode"]
            )
        ):
            yield
            return
        row = self._start_operation(
            operation_class,
            operation_name,
            tensor,
        )
        try:
            yield
        finally:
            self._finish_operation(row)

    @contextmanager
    def collective(
        self,
        operation,
        tensor,
        *,
        collective_kind=None,
        process_group=None,
        async_mode=False,
        source_stream=None,
        completion_stream=None,
    ):
        if not isinstance(async_mode, bool) or async_mode:
            raise ValueError("async_mode must be False")
        if (
            not self.enabled
            or self._active_step is None
            or not self._active_step["is_decode"]
        ):
            yield
            return
        operation_name = str(operation)
        row = self._start_operation(
            "collective",
            operation_name,
            tensor,
            allow_missing_layer=True,
            source_stream=source_stream,
        )
        try:
            yield
        finally:
            self._finish_operation(
                row,
                completion_stream=completion_stream,
            )
            self._pending_collectives.append(
                row | {
                    "operation": operation_name,
                    "collective_kind": (
                        operation_name
                        if collective_kind is None
                        else str(collective_kind)
                    ),
                    "process_group": (
                        "tensor_parallel"
                        if process_group is None
                        else str(process_group)
                    ),
                    "async_mode": False,
                }
            )

    @staticmethod
    def _elapsed_ns(row):
        return round(
            row["_cuda_start"].elapsed_time(
                row["_cuda_end"],
            )
            * 1_000_000
        )

    def finalize(self, *, already_synchronized=False):
        if not isinstance(already_synchronized, bool):
            raise ValueError(
                "already_synchronized must be a bool"
            )
        if self._finalized is not None:
            return copy.deepcopy(self._finalized)
        if not self.enabled:
            self._finalized = {
                "rank": self.rank,
                "enabled": False,
                "finalization_status": "complete",
                "steps": [],
                "layers": [],
                "operations": [],
                "collectives": [],
            }
            return copy.deepcopy(self._finalized)
        if (
            self._active_layer is not None
            or self._active_operation is not None
        ):
            raise RuntimeError(
                "cannot finalize with an open profiler scope"
            )
        if self._active_step is not None:
            raise RuntimeError(
                "cannot finalize with an active profiler step"
            )
        if not already_synchronized:
            self._synchronize()
        steps = []
        for pending in self._pending_steps:
            wall_ns = (
                pending["_wall_end_ns"]
                - pending["_wall_start_ns"]
            )
            cuda_ns = self._elapsed_ns(pending)
            steps.append({
                name: pending[name]
                for name in (
                    "rank",
                    "step_index",
                    "batch_kind",
                    "is_decode",
                    "decode_ordinal",
                    "active_sequence_count",
                    "request_set_sha256",
                    "dispatch",
                    "command_id",
                    "engine_step_id",
                    "repeat_index",
                    "speculative_selected_sequence_ids_sha256",
                )
            } | {
                "wall_ns": wall_ns,
                "cuda_ns": cuda_ns,
                "non_cuda_upper_bound_ns": max(
                    0,
                    wall_ns - cuda_ns,
                ),
            })
        layers = []
        for pending in self._pending_layers:
            layers.append({
                name: pending[name]
                for name in (
                    "rank",
                    "step_index",
                    "decode_ordinal",
                    "command_id",
                    "engine_step_id",
                    "repeat_index",
                    "request_set_sha256",
                    "speculative_selected_sequence_ids_sha256",
                    "attempt",
                    "workload",
                    "repetition",
                    "layer_index",
                    "layer_role",
                )
            } | {
                "wall_start_ns": pending["_wall_start_ns"],
                "wall_end_ns": pending["_wall_end_ns"],
                "wall_ns": (
                    pending["_wall_end_ns"]
                    - pending["_wall_start_ns"]
                ),
                "cuda_ns": self._elapsed_ns(pending),
            })
        operations = []
        for pending in self._pending_operations:
            operations.append({
                name: pending[name]
                for name in (
                    "rank",
                    "step_index",
                    "decode_ordinal",
                    "command_id",
                    "engine_step_id",
                    "repeat_index",
                    "request_set_sha256",
                    "speculative_selected_sequence_ids_sha256",
                    "attempt",
                    "workload",
                    "repetition",
                    "layer_index",
                    "layer_role",
                    "operation_ordinal",
                    "operation_class",
                    "operation_name",
                    "tensor_shape",
                    "tensor_dtype",
                    "source_stream",
                    "completion_stream",
                )
            } | {
                "wall_start_ns": pending["_wall_start_ns"],
                "wall_end_ns": pending["_wall_end_ns"],
                "wall_ns": (
                    pending["_wall_end_ns"]
                    - pending["_wall_start_ns"]
                ),
                "cuda_ns": self._elapsed_ns(pending),
            })
        collectives = []
        for pending in self._pending_collectives:
            collectives.append({
                name: pending[name]
                for name in (
                    "rank",
                    "step_index",
                    "decode_ordinal",
                    "command_id",
                    "engine_step_id",
                    "repeat_index",
                    "request_set_sha256",
                    "speculative_selected_sequence_ids_sha256",
                    "attempt",
                    "workload",
                    "repetition",
                    "layer_index",
                    "layer_role",
                    "operation_ordinal",
                    "operation",
                    "operation_class",
                    "collective_kind",
                    "process_group",
                    "async_mode",
                    "source_stream",
                    "completion_stream",
                    "tensor_shape",
                    "tensor_dtype",
                )
            } | {
                "wall_start_ns": pending["_wall_start_ns"],
                "wall_end_ns": pending["_wall_end_ns"],
                "wall_ns": (
                    pending["_wall_end_ns"]
                    - pending["_wall_start_ns"]
                ),
                "cuda_ns": self._elapsed_ns(pending),
            })
        self._finalized = {
            "rank": self.rank,
            "enabled": True,
            "finalization_status": "complete",
            "steps": steps,
            "layers": layers,
            "operations": operations,
            "collectives": collectives,
            "dropped_steps": 0,
            "dropped_layers": 0,
            "dropped_operations": 0,
            "dropped_collectives": 0,
        }
        return copy.deepcopy(self._finalized)
