from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import copy
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Callable, Sequence


SCHEMA = "tinyllmforge.synchronous-collective-census.v1"
ALLOWED_SAMPLE_BUDGETS = frozenset({0, 8, 16, 32})

_ACTIVE_CENSUS = ContextVar(
    "tinyvllm_synchronous_collective_census",
    default=None,
)

_SOURCE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_ELEMENT_BYTES = {
    "torch.bool": 1,
    "torch.uint8": 1,
    "torch.int8": 1,
    "torch.float8_e4m3fn": 1,
    "torch.float8_e5m2": 1,
    "torch.int16": 2,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.int32": 4,
    "torch.float32": 4,
    "torch.int64": 8,
    "torch.float64": 8,
    "torch.complex64": 8,
    "torch.complex128": 16,
}


@dataclass(frozen=True)
class CollectiveCensusPolicy:
    sample_budget: int
    cohort_count: int
    expected_collective_count: int
    source_revision: str
    attempt: str
    workload: str
    repetition: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sample_budget, int)
            or isinstance(self.sample_budget, bool)
            or self.sample_budget not in ALLOWED_SAMPLE_BUDGETS
        ):
            raise ValueError(
                "sample_budget must be one of 0, 8, 16, or 32"
            )
        if (
            not isinstance(self.cohort_count, int)
            or isinstance(self.cohort_count, bool)
            or self.cohort_count <= 0
        ):
            raise ValueError("cohort_count must be a positive integer")
        if (
            not isinstance(self.expected_collective_count, int)
            or isinstance(self.expected_collective_count, bool)
            or self.expected_collective_count <= 0
        ):
            raise ValueError(
                "expected_collective_count must be a positive integer"
            )
        if not _SOURCE_REVISION_PATTERN.fullmatch(
            str(self.source_revision)
        ):
            raise ValueError(
                "source_revision must be a 40-character lowercase git SHA"
            )
        if not isinstance(self.attempt, str) or not self.attempt:
            raise ValueError("attempt must be a non-empty string")
        if not isinstance(self.workload, str) or not self.workload:
            raise ValueError("workload must be a non-empty string")
        if (
            not isinstance(self.repetition, int)
            or isinstance(self.repetition, bool)
            or self.repetition < 0
        ):
            raise ValueError("repetition must be a non-negative integer")

    def sampled_ordinals(
        self,
        *,
        decode_ordinal: int,
        collective_count: int,
    ) -> Sequence[int]:
        if (
            not isinstance(decode_ordinal, int)
            or isinstance(decode_ordinal, bool)
            or decode_ordinal < 0
        ):
            raise ValueError(
                "decode_ordinal must be a non-negative integer"
            )
        if collective_count != self.expected_collective_count:
            raise ValueError(
                "collective_count must equal expected_collective_count"
            )
        seed = (
            f"{self.source_revision}\0{self.attempt}\0"
            f"{self.workload}\0{self.repetition}\0{decode_ordinal}"
        ).encode("utf-8")
        cohort = int.from_bytes(
            hashlib.sha256(seed).digest()[:8],
            "big",
        ) % self.cohort_count
        cohort_width = math.ceil(
            self.expected_collective_count / self.cohort_count
        )
        start = (
            cohort * cohort_width
        ) % self.expected_collective_count
        return tuple(
            sorted(
                (start + offset) % self.expected_collective_count
                for offset in range(
                    min(
                        self.sample_budget,
                        self.expected_collective_count,
                    )
                )
            )
        )


def active_synchronous_collective_census():
    return _ACTIVE_CENSUS.get()


@contextmanager
def activate_synchronous_collective_census(census):
    token = _ACTIVE_CENSUS.set(census)
    try:
        yield census
    finally:
        _ACTIVE_CENSUS.reset(token)


def census_layer(layer_index: int, layer_role: str):
    census = active_synchronous_collective_census()
    if census is None:
        return nullcontext()
    return census.layer(layer_index, layer_role)


def run_census_step(
    census,
    *,
    batch_kind: str,
    is_decode: bool,
    active_sequence_count: int,
    request_set_sha256: str,
    dispatch: str,
    call: Callable,
):
    census.begin_step(
        batch_kind=batch_kind,
        is_decode=is_decode,
        active_sequence_count=active_sequence_count,
        request_set_sha256=request_set_sha256,
        dispatch=dispatch,
    )
    try:
        result = call()
    except BaseException:
        census.end_step(status="failed")
        raise
    census.end_step(status="completed")
    return result


def observe_synchronous_collective(
    *,
    site_role: str,
    operation: str,
    tensor,
    call: Callable,
    collective_kind: str,
    process_group: str,
    execution_phase: str,
    async_mode: bool,
    source_rank: int | None,
    destination_rank: int | None,
):
    if not isinstance(async_mode, bool) or async_mode:
        raise ValueError("async_mode must be False")
    census = active_synchronous_collective_census()
    if census is None:
        return call(tensor)
    return census.observe(
        site_role=site_role,
        operation=operation,
        tensor=tensor,
        call=call,
        collective_kind=collective_kind,
        process_group=process_group,
        execution_phase=execution_phase,
        async_mode=async_mode,
        source_rank=source_rank,
        destination_rank=destination_rank,
    )


class SynchronousCollectiveCensus:

    def __init__(
        self,
        *,
        rank: int,
        policy: CollectiveCensusPolicy | None,
        event_factory: Callable | None,
        synchronize: Callable | None,
        stream_resolver: Callable | None,
        enabled: bool = True,
    ):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        if enabled and not isinstance(policy, CollectiveCensusPolicy):
            raise TypeError("policy must be a CollectiveCensusPolicy")
        if enabled and not callable(event_factory):
            raise ValueError("event_factory must be callable")
        if enabled and not callable(synchronize):
            raise ValueError("synchronize must be callable")
        if enabled and not callable(stream_resolver):
            raise ValueError("stream_resolver must be callable")
        self.rank = int(rank)
        self.enabled = enabled
        self.policy = policy
        self._event_factory = event_factory
        self._synchronize = synchronize
        self._stream_resolver = stream_resolver
        self._active_step = None
        self._active_layer = None
        self._steps = []
        self._collectives = []
        self._timed_records = []
        self._decode_ordinal = 0
        self._decode_request_set_sha256 = None
        self._finalized = None

    @classmethod
    def disabled(cls, *, rank: int):
        return cls(
            rank=rank,
            policy=None,
            event_factory=None,
            synchronize=None,
            stream_resolver=None,
            enabled=False,
        )

    def _require_mutable(self) -> None:
        if self._finalized is not None:
            raise RuntimeError("synchronous collective census is finalized")

    def begin_step(
        self,
        *,
        batch_kind: str,
        is_decode: bool,
        active_sequence_count: int,
        request_set_sha256: str,
        dispatch: str,
    ) -> None:
        if not self.enabled:
            return
        self._require_mutable()
        if self._active_step is not None:
            raise RuntimeError("collective census step is active")
        if not isinstance(is_decode, bool):
            raise ValueError("is_decode must be a boolean")
        if (
            not isinstance(active_sequence_count, int)
            or isinstance(active_sequence_count, bool)
            or active_sequence_count <= 0
        ):
            raise ValueError(
                "active_sequence_count must be a positive integer"
            )
        if not _SHA256_PATTERN.fullmatch(str(request_set_sha256)):
            raise ValueError(
                "request_set_sha256 must be a lowercase SHA-256"
            )
        if not isinstance(batch_kind, str) or not batch_kind:
            raise ValueError("batch_kind must be a non-empty string")
        if not isinstance(dispatch, str) or not dispatch:
            raise ValueError("dispatch must be a non-empty string")

        decode_ordinal = None
        sampled_ordinals = ()
        if is_decode:
            if request_set_sha256 != self._decode_request_set_sha256:
                self._decode_ordinal = 0
                self._decode_request_set_sha256 = request_set_sha256
            decode_ordinal = self._decode_ordinal
            self._decode_ordinal += 1
            sampled_ordinals = self.policy.sampled_ordinals(
                decode_ordinal=decode_ordinal,
                collective_count=self.policy.expected_collective_count,
            )
        self._active_step = {
            "batch_kind": batch_kind,
            "is_decode": is_decode,
            "active_sequence_count": active_sequence_count,
            "request_set_sha256": request_set_sha256,
            "dispatch": dispatch,
            "decode_ordinal": decode_ordinal,
            "collective_count": 0,
            "status": "active",
            "_sampled_ordinals": frozenset(sampled_ordinals),
        }

    def end_step(self, *, status: str = "completed") -> None:
        if not self.enabled:
            return
        if self._active_step is None:
            raise RuntimeError("collective census step is not active")
        if status not in {"completed", "failed"}:
            raise ValueError("step status must be completed or failed")
        step = self._active_step
        step["status"] = status
        step.pop("_sampled_ordinals")
        if step["is_decode"]:
            self._steps.append(step)
        self._active_step = None

    @contextmanager
    def layer(self, layer_index: int, layer_role: str):
        if not self.enabled:
            yield
            return
        self._require_mutable()
        if (
            not isinstance(layer_index, int)
            or isinstance(layer_index, bool)
            or layer_index < 0
        ):
            raise ValueError("layer_index must be a non-negative integer")
        if not isinstance(layer_role, str) or not layer_role:
            raise ValueError("layer_role must be a non-empty string")
        previous = self._active_layer
        self._active_layer = {
            "layer_index": int(layer_index),
            "layer_role": str(layer_role),
        }
        try:
            yield
        finally:
            self._active_layer = previous

    def observe(
        self,
        *,
        site_role: str,
        operation: str,
        tensor,
        call: Callable,
        collective_kind: str,
        process_group: str,
        execution_phase: str,
        async_mode: bool,
        source_rank: int | None,
        destination_rank: int | None,
    ):
        if not self.enabled:
            return call(tensor)
        self._require_mutable()
        row = self._prepare_record(
            site_role=site_role,
            operation=operation,
            tensor=tensor,
            collective_kind=collective_kind,
            process_group=process_group,
            execution_phase=execution_phase,
            async_mode=async_mode,
            source_rank=source_rank,
            destination_rank=destination_rank,
        )
        if row is None:
            return call(tensor)
        start_event = (
            self._event_factory()
            if row["event_sampled"]
            else None
        )
        end_event = (
            self._event_factory()
            if row["event_sampled"]
            else None
        )
        if start_event is not None:
            start_event.record()
        try:
            result = call(tensor)
        except BaseException:
            row["status"] = "failed"
            raise
        else:
            row["status"] = "completed"
            return result
        finally:
            if end_event is not None:
                end_event.record()
            self._finish_record(row, start_event, end_event)

    def _prepare_record(
        self,
        *,
        site_role,
        operation,
        tensor,
        collective_kind,
        process_group,
        execution_phase,
        async_mode,
        source_rank,
        destination_rank,
    ):
        if not isinstance(async_mode, bool) or async_mode:
            raise ValueError("async_mode must be False")
        site_id = self._site_id(site_role)
        if self._active_step is None or not self._active_step["is_decode"]:
            return None
        if execution_phase not in {"decode", "decode_or_prefill"}:
            return None
        if not isinstance(operation, str) or not operation:
            raise ValueError("operation must be a non-empty string")
        if not isinstance(collective_kind, str) or not collective_kind:
            raise ValueError(
                "collective_kind must be a non-empty string"
            )
        if not isinstance(process_group, str) or not process_group:
            raise ValueError("process_group must be a non-empty string")

        step = self._active_step
        collective_ordinal = step["collective_count"]
        step["collective_count"] += 1
        shape = [int(dimension) for dimension in tensor.shape]
        dtype = str(tensor.dtype)
        tensor_bytes = math.prod(shape) * self._element_size(tensor, dtype)
        layer = self._active_layer or {
            "layer_index": None,
            "layer_role": None,
        }
        return {
            "attempt": self.policy.attempt,
            "workload": self.policy.workload,
            "repetition": self.policy.repetition,
            "rank": self.rank,
            "batch_kind": step["batch_kind"],
            "request_set_sha256": step["request_set_sha256"],
            "decode_ordinal": step["decode_ordinal"],
            "collective_ordinal": collective_ordinal,
            "site_id": site_id,
            "site_role": site_role,
            "operation": operation,
            "collective_kind": collective_kind,
            "process_group": process_group,
            "execution_phase": "decode",
            "tensor_shape": shape,
            "tensor_dtype": dtype,
            "tensor_bytes": tensor_bytes,
            "source_stream": str(self._stream_resolver()),
            "async_mode": False,
            "source_rank": source_rank,
            "destination_rank": destination_rank,
            "layer_index": layer["layer_index"],
            "layer_role": layer["layer_role"],
            "event_sampled": (
                collective_ordinal in step["_sampled_ordinals"]
            ),
            "cuda_ns": None,
            "status": "pending",
        }

    def _site_id(self, site_role: str) -> str:
        if site_role == "vocab_parallel_embedding":
            return "embedding.input"
        if site_role == "greedy_token_broadcast":
            return "sampling.greedy_token"
        if site_role == "row_parallel_output":
            if self._active_layer is None:
                raise ValueError(
                    "site_role row_parallel_output requires layer context"
                )
            layer_index = self._active_layer["layer_index"]
            layer_role = self._active_layer["layer_role"]
            if layer_role in {"full_attention", "linear_attention"}:
                component = "attention"
            elif layer_role == "mlp":
                component = "mlp"
            else:
                raise ValueError(
                    "site_role row_parallel_output has unknown layer_role"
                )
            return f"layer.{layer_index:03d}.{component}.output"
        if site_role in {
            "row_parallel_prefill_materialization",
            "replicated_weight_input_materialization",
            "lm_head_parameter_materialization",
            "vocab_parallel_logits_materialization",
        }:
            return site_role.replace("_", ".")
        raise ValueError(f"unknown site_role: {site_role}")

    @staticmethod
    def _element_size(tensor, dtype: str) -> int:
        element_size = getattr(tensor, "element_size", None)
        if callable(element_size):
            size = int(element_size())
            if size <= 0:
                raise ValueError("tensor element_size must be positive")
            return size
        try:
            return _ELEMENT_BYTES[dtype]
        except KeyError as error:
            raise ValueError(
                f"unsupported tensor dtype without element_size: {dtype}"
            ) from error

    def _finish_record(self, row, start_event, end_event) -> None:
        if start_event is not None:
            row["_start_event"] = start_event
            row["_end_event"] = end_event
            self._timed_records.append(row)
        self._collectives.append(row)

    def finalize(self, *, already_synchronized: bool = False) -> dict:
        if not isinstance(already_synchronized, bool):
            raise ValueError("already_synchronized must be a bool")
        if not self.enabled:
            return {
                "schema": SCHEMA,
                "rank": self.rank,
                "enabled": False,
                "finalization_status": "complete",
                "steps": [],
                "collectives": [],
            }
        if self._active_step is not None or self._active_layer is not None:
            raise RuntimeError("cannot finalize an open census scope")
        if self._finalized is None:
            if self._timed_records and not already_synchronized:
                self._synchronize()
            for row in self._timed_records:
                start_event = row.pop("_start_event")
                end_event = row.pop("_end_event")
                elapsed_ms = float(start_event.elapsed_time(end_event))
                row["cuda_ns"] = int(round(elapsed_ms * 1_000_000))
            self._finalized = {
                "schema": SCHEMA,
                "rank": self.rank,
                "enabled": True,
                "finalization_status": "complete",
                "source_revision": self.policy.source_revision,
                "attempt": self.policy.attempt,
                "workload": self.policy.workload,
                "repetition": self.policy.repetition,
                "sample_budget": self.policy.sample_budget,
                "cohort_count": self.policy.cohort_count,
                "expected_collective_count": (
                    self.policy.expected_collective_count
                ),
                "steps": copy.deepcopy(self._steps),
                "collectives": copy.deepcopy(self._collectives),
            }
        return copy.deepcopy(self._finalized)
