from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import copy
import sys
import time


_ACTIVE_PROFILER = ContextVar(
    "tinyvllm_decode_internal_profiler",
    default=None,
)


def active_decode_internal_profiler():
    return _ACTIVE_PROFILER.get()


def _active_model_runner_command_trace():
    module = sys.modules.get(
        "tinyvllm.engine.model_runner_command_timeline"
    )
    if module is None:
        return None
    return module.active_model_runner_command_trace()


def profile_collective(operation, tensor, call):
    profiler = active_decode_internal_profiler()
    if profiler is None:
        return call(tensor)
    with profiler.collective(operation, tensor):
        return call(tensor)


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
    try:
        return call()
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
        self._profile_label = (
            str(profile_label).strip("/")
            if profile_label
            else f"rank={self.rank}"
        )
        self._active_command_trace = (
            active_command_trace
            if callable(active_command_trace)
            else _active_model_runner_command_trace
        )
        self._active_step = None
        self._active_token = None
        self._pending_steps = []
        self._pending_collectives = []
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
        self._active_step = {
            "rank": self.rank,
            "step_index": len(self._pending_steps),
            "batch_kind": batch_kind,
            "is_decode": is_decode,
            "decode_ordinal": decode_ordinal,
            "active_sequence_count": int(active_sequence_count),
            "request_set_sha256": request_set_sha256,
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
            "_wall_start_ns": self._clock_ns(),
            "_cuda_start": start_event,
            "_nvtx_context": nvtx_context,
        }
        self._active_token = _ACTIVE_PROFILER.set(self)

    def end_step(self):
        if not self.enabled:
            return
        self._require_mutable()
        if self._active_step is None:
            raise RuntimeError("decode internal profiler has no active step")
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
    def collective(self, operation, tensor):
        if (
            not self.enabled
            or self._active_step is None
            or not self._active_step["is_decode"]
        ):
            yield
            return
        self._require_mutable()
        start_event = self._event_factory()
        start_event.record()
        wall_start_ns = self._clock_ns()
        nvtx_context = self._nvtx_range_factory(
            f"decode_internal/{self._profile_label}/"
            f"collective/{operation}"
        )
        nvtx_context.__enter__()
        try:
            yield
        finally:
            wall_end_ns = self._clock_ns()
            end_event = self._event_factory()
            end_event.record()
            self._pending_collectives.append({
                "rank": self.rank,
                "step_index": self._active_step["step_index"],
                "decode_ordinal": self._active_step["decode_ordinal"],
                "command_id": self._active_step["command_id"],
                "engine_step_id": self._active_step["engine_step_id"],
                "repeat_index": self._active_step["repeat_index"],
                "operation": str(operation),
                "tensor_shape": [
                    int(value) for value in tensor.shape
                ],
                "tensor_dtype": str(tensor.dtype),
                "_wall_start_ns": wall_start_ns,
                "_wall_end_ns": wall_end_ns,
                "_cuda_start": start_event,
                "_cuda_end": end_event,
            })
            nvtx_context.__exit__(None, None, None)

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
                "collectives": [],
            }
            return copy.deepcopy(self._finalized)
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
                )
            } | {
                "wall_ns": wall_ns,
                "cuda_ns": cuda_ns,
                "non_cuda_upper_bound_ns": max(
                    0,
                    wall_ns - cuda_ns,
                ),
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
                    "operation",
                    "tensor_shape",
                    "tensor_dtype",
                )
            } | {
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
            "collectives": collectives,
        }
        return copy.deepcopy(self._finalized)
