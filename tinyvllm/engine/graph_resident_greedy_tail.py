"""Generic contracts for a graph-resident greedy decode tail."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Optional


TensorIdentity = tuple[
    int,
    tuple[int, ...],
    tuple[int, ...],
    int,
    str,
    str,
]


def _require_bool(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _require_non_negative_int(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a non-negative integer"
        )
    return value


def _require_positive_int(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_reason(value, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{name} must be a non-empty string"
        )
    return value


def tensor_identity(tensor) -> TensorIdentity:
    """Return storage geometry that survives recreation of a tensor view."""

    data_ptr = tensor.data_ptr()
    shape = tuple(tensor.shape)
    stride = tuple(tensor.stride())
    storage_offset = tensor.storage_offset()
    if (
        isinstance(data_ptr, bool)
        or not isinstance(data_ptr, int)
        or data_ptr < 0
    ):
        raise ValueError(
            "tensor data pointer must be a non-negative integer"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for value in shape
    ):
        raise ValueError(
            "tensor shape must contain non-negative integers"
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        for value in stride
    ):
        raise ValueError(
            "tensor stride must contain integers"
        )
    _require_non_negative_int(
        storage_offset,
        "tensor storage offset",
    )
    return (
        data_ptr,
        shape,
        stride,
        storage_offset,
        str(tensor.dtype),
        str(tensor.device),
    )


@dataclass(frozen=True)
class GraphResidentGreedyTailDecision:
    optimized: bool
    fallback_reason: Optional[str]


@dataclass(frozen=True)
class GraphResidentGreedyTailCaptureReceipt:
    source_identity: TensorIdentity
    graph_generation: int
    rank: int
    capture_duration_ns: int
    allocated_delta_bytes: int
    reserved_delta_bytes: int
    retained_logits_bytes: int
    retained_float32_bytes: int
    retained_token_bytes: int

    def __post_init__(self) -> None:
        _require_positive_int(
            self.graph_generation,
            "graph_generation",
        )
        _require_non_negative_int(self.rank, "rank")
        for field_name in (
            "capture_duration_ns",
            "allocated_delta_bytes",
            "reserved_delta_bytes",
            "retained_logits_bytes",
            "retained_float32_bytes",
            "retained_token_bytes",
        ):
            _require_non_negative_int(
                getattr(self, field_name),
                field_name,
            )


@dataclass(frozen=True)
class GraphResidentGreedyTailReplay:
    logits: object
    token_ids: object


@dataclass
class GraphResidentGreedyTailStats:
    eligible_steps: int = 0
    captured_graphs: int = 0
    replayed_steps: int = 0
    final_token_d2h_calls: int = 0
    avoided_external_compute_logits_calls: int = 0
    avoided_external_float32_conversions: int = 0
    avoided_external_argmax_calls: int = 0
    fallback_counts: dict[str, int] = field(default_factory=dict)
    quarantine_reason: Optional[str] = None
    capture_receipt: Optional[
        GraphResidentGreedyTailCaptureReceipt
    ] = None

    def record_capture(
        self,
        receipt: GraphResidentGreedyTailCaptureReceipt,
    ) -> None:
        if not isinstance(
            receipt,
            GraphResidentGreedyTailCaptureReceipt,
        ):
            raise ValueError(
                "capture receipt has an invalid type"
            )
        if self.capture_receipt is not None:
            raise ValueError("capture receipt is already recorded")
        self.capture_receipt = receipt
        self.captured_graphs += 1

    def record_replay(self) -> None:
        self.eligible_steps += 1
        self.replayed_steps += 1
        self.avoided_external_compute_logits_calls += 1
        self.avoided_external_float32_conversions += 1
        self.avoided_external_argmax_calls += 1

    def record_token_d2h(self) -> None:
        if self.final_token_d2h_calls >= self.replayed_steps:
            raise ValueError(
                "token D2H count cannot exceed replay count"
            )
        self.final_token_d2h_calls += 1

    def record_fallback(self, reason: str) -> None:
        reason = _require_reason(reason, "fallback reason")
        self.fallback_counts[reason] = (
            self.fallback_counts.get(reason, 0) + 1
        )

    def quarantine(self, reason: str) -> None:
        reason = _require_reason(reason, "quarantine reason")
        if self.quarantine_reason is None:
            self.quarantine_reason = reason

    def summary(self) -> dict[str, object]:
        receipt = self.capture_receipt
        receipt_payload = None
        if receipt is not None:
            (
                data_ptr,
                shape,
                stride,
                storage_offset,
                dtype,
                device,
            ) = receipt.source_identity
            retained_static_bytes = (
                receipt.retained_logits_bytes
                + receipt.retained_float32_bytes
                + receipt.retained_token_bytes
            )
            receipt_payload = {
                "source_identity": {
                    "data_ptr": data_ptr,
                    "shape": list(shape),
                    "stride": list(stride),
                    "storage_offset": storage_offset,
                    "dtype": dtype,
                    "device": device,
                },
                "graph_generation": receipt.graph_generation,
                "rank": receipt.rank,
                "capture_duration_ns":
                    receipt.capture_duration_ns,
                "allocated_delta_bytes":
                    receipt.allocated_delta_bytes,
                "reserved_delta_bytes":
                    receipt.reserved_delta_bytes,
                "retained_logits_bytes":
                    receipt.retained_logits_bytes,
                "retained_float32_bytes":
                    receipt.retained_float32_bytes,
                "retained_token_bytes":
                    receipt.retained_token_bytes,
                "retained_static_bytes": retained_static_bytes,
            }
        return {
            "eligible_steps": self.eligible_steps,
            "captured_graphs": self.captured_graphs,
            "replayed_steps": self.replayed_steps,
            "final_token_d2h_calls": self.final_token_d2h_calls,
            "avoided_external_compute_logits_calls":
                self.avoided_external_compute_logits_calls,
            "avoided_external_float32_conversions":
                self.avoided_external_float32_conversions,
            "avoided_external_argmax_calls":
                self.avoided_external_argmax_calls,
            "fallback_counts": dict(
                sorted(self.fallback_counts.items())
            ),
            "quarantine_reason": self.quarantine_reason,
            "capture_receipt": receipt_payload,
        }


def _tensor_bytes(tensor, name: str) -> int:
    try:
        numel = tensor.numel()
        element_size = tensor.element_size()
    except (AttributeError, TypeError) as error:
        raise ValueError(
            f"{name} must expose numel() and element_size()"
        ) from error
    _require_non_negative_int(numel, f"{name} numel")
    _require_non_negative_int(
        element_size,
        f"{name} element_size",
    )
    return numel * element_size


class GraphResidentGreedyTail:
    """Own one captured logits-to-token graph and its static outputs."""

    def __init__(
        self,
        *,
        graph,
        logits,
        float_logits,
        token_ids,
        receipt: GraphResidentGreedyTailCaptureReceipt,
        stats: GraphResidentGreedyTailStats,
    ):
        self.graph = graph
        self.logits = logits
        self.float_logits = float_logits
        self.token_ids = token_ids
        self.receipt = receipt
        self.stats = stats
        self._token_d2h_pending = False

    @classmethod
    def capture(
        cls,
        *,
        static_hidden,
        compute_logits,
        float32_dtype,
        graph_generation: int,
        rank: int,
        graph_factory,
        capture_context_factory,
        synchronize,
        memory_snapshot,
        clock_ns,
        stats: Optional[GraphResidentGreedyTailStats] = None,
    ) -> "GraphResidentGreedyTail":
        source_identity = tensor_identity(static_hidden)
        _require_positive_int(
            graph_generation,
            "graph_generation",
        )
        _require_non_negative_int(rank, "rank")
        for value, name in (
            (compute_logits, "compute_logits"),
            (graph_factory, "graph_factory"),
            (
                capture_context_factory,
                "capture_context_factory",
            ),
            (synchronize, "synchronize"),
            (memory_snapshot, "memory_snapshot"),
            (clock_ns, "clock_ns"),
        ):
            if not callable(value):
                raise ValueError(f"{name} must be callable")
        if stats is None:
            stats = GraphResidentGreedyTailStats()
        if not isinstance(stats, GraphResidentGreedyTailStats):
            raise ValueError(
                "stats must be GraphResidentGreedyTailStats"
            )

        synchronize()
        before_allocated, before_reserved = memory_snapshot()
        _require_non_negative_int(
            before_allocated,
            "allocated memory snapshot",
        )
        _require_non_negative_int(
            before_reserved,
            "reserved memory snapshot",
        )

        warmup_logits = compute_logits(static_hidden[:1])
        warmup_float_logits = warmup_logits.to(float32_dtype)
        warmup_float_logits.argmax(dim=-1)
        synchronize()

        graph = graph_factory()
        start_ns = clock_ns()
        _require_non_negative_int(start_ns, "capture start time")
        with capture_context_factory(graph):
            logits = compute_logits(static_hidden[:1])
            float_logits = logits.to(float32_dtype)
            token_ids = float_logits.argmax(dim=-1)
        synchronize()
        end_ns = clock_ns()
        _require_non_negative_int(end_ns, "capture end time")
        if end_ns < start_ns:
            raise ValueError(
                "capture end time precedes start time"
            )
        after_allocated, after_reserved = memory_snapshot()
        _require_non_negative_int(
            after_allocated,
            "allocated memory snapshot",
        )
        _require_non_negative_int(
            after_reserved,
            "reserved memory snapshot",
        )
        receipt = GraphResidentGreedyTailCaptureReceipt(
            source_identity=source_identity,
            graph_generation=graph_generation,
            rank=rank,
            capture_duration_ns=end_ns - start_ns,
            allocated_delta_bytes=max(
                0,
                after_allocated - before_allocated,
            ),
            reserved_delta_bytes=max(
                0,
                after_reserved - before_reserved,
            ),
            retained_logits_bytes=_tensor_bytes(
                logits,
                "logits",
            ),
            retained_float32_bytes=_tensor_bytes(
                float_logits,
                "float logits",
            ),
            retained_token_bytes=_tensor_bytes(
                token_ids,
                "token ids",
            ),
        )
        stats.record_capture(receipt)
        return cls(
            graph=graph,
            logits=logits,
            float_logits=float_logits,
            token_ids=token_ids,
            receipt=receipt,
            stats=stats,
        )

    def matches(
        self,
        *,
        static_hidden,
        graph_generation: int,
        rank: int,
    ) -> bool:
        try:
            identity = tensor_identity(static_hidden)
        except (AttributeError, TypeError, ValueError):
            return False
        return (
            identity == self.receipt.source_identity
            and graph_generation
            == self.receipt.graph_generation
            and rank == self.receipt.rank
        )

    def replay(
        self,
        *,
        static_hidden,
        graph_generation: int,
        rank: int,
    ) -> GraphResidentGreedyTailReplay:
        if self.stats.quarantine_reason is not None:
            raise RuntimeError(
                "graph-resident greedy tail is quarantined: "
                f"{self.stats.quarantine_reason}"
            )
        if graph_generation != self.receipt.graph_generation:
            raise RuntimeError("graph generation drift")
        if rank != self.receipt.rank:
            raise RuntimeError("rank drift")
        if tensor_identity(
            static_hidden
        ) != self.receipt.source_identity:
            raise RuntimeError("source identity drift")
        if self._token_d2h_pending:
            raise RuntimeError(
                "previous replay token D2H is pending"
            )
        try:
            self.graph.replay()
        except Exception as error:
            self.stats.quarantine(
                "replay_failure:"
                + type(error).__name__
            )
            raise
        self.stats.record_replay()
        self._token_d2h_pending = True
        return GraphResidentGreedyTailReplay(
            logits=self.logits,
            token_ids=self.token_ids,
        )

    def mark_token_d2h(self) -> None:
        if not self._token_d2h_pending:
            raise RuntimeError("no replay token D2H is pending")
        self.stats.record_token_d2h()
        self._token_d2h_pending = False

    def summary(self) -> dict[str, object]:
        return self.stats.summary()


def decide_graph_resident_greedy_tail(
    *,
    enabled: bool,
    rank: int,
    tensor_parallel_size: int,
    is_prefill: bool,
    enforce_eager: bool,
    batch_kind: Optional[str],
    active_batch_size: int,
    selected_graph_batch_size: int,
    do_sample: bool,
    temperatures: tuple[object, ...],
    input_embeds_present: bool,
    return_hidden: bool,
    incompatible_modes: tuple[str, ...],
    capture_available: bool,
    quarantined: bool,
    source_matches: bool,
) -> GraphResidentGreedyTailDecision:
    """Decide whether one ordinary decode step may replay the tail."""

    _require_bool(enabled, "enabled")
    _require_non_negative_int(rank, "rank")
    _require_positive_int(
        tensor_parallel_size,
        "tensor_parallel_size",
    )
    _require_bool(is_prefill, "is_prefill")
    _require_bool(enforce_eager, "enforce_eager")
    if batch_kind is not None and not isinstance(batch_kind, str):
        raise ValueError("batch_kind must be a string or None")
    _require_non_negative_int(
        active_batch_size,
        "active_batch_size",
    )
    _require_positive_int(
        selected_graph_batch_size,
        "selected_graph_batch_size",
    )
    _require_bool(do_sample, "do_sample")
    if not isinstance(temperatures, tuple):
        raise ValueError("temperatures must be a tuple")
    _require_bool(
        input_embeds_present,
        "input_embeds_present",
    )
    _require_bool(return_hidden, "return_hidden")
    if not isinstance(incompatible_modes, tuple):
        raise ValueError("incompatible_modes must be a tuple")
    if any(
        not isinstance(mode, str) or not mode
        for mode in incompatible_modes
    ):
        raise ValueError(
            "incompatible_modes must contain non-empty strings"
        )
    _require_bool(capture_available, "capture_available")
    _require_bool(quarantined, "quarantined")
    _require_bool(source_matches, "source_matches")

    if not enabled:
        return GraphResidentGreedyTailDecision(False, "disabled")
    if rank != 0:
        return GraphResidentGreedyTailDecision(
            False,
            "non_root_rank",
        )
    if tensor_parallel_size != 1:
        return GraphResidentGreedyTailDecision(
            False,
            "tensor_parallel_unsupported",
        )
    if is_prefill:
        return GraphResidentGreedyTailDecision(
            False,
            "prefill_unsupported",
        )
    if enforce_eager:
        return GraphResidentGreedyTailDecision(
            False,
            "eager_unsupported",
        )
    if batch_kind == "mixed":
        return GraphResidentGreedyTailDecision(
            False,
            "mixed_batch_unsupported",
        )
    if active_batch_size != 1 or len(temperatures) != 1:
        return GraphResidentGreedyTailDecision(
            False,
            "batch_size_unsupported",
        )
    if selected_graph_batch_size != 1:
        return GraphResidentGreedyTailDecision(
            False,
            "selected_graph_batch_unsupported",
        )
    if not do_sample:
        return GraphResidentGreedyTailDecision(
            False,
            "sampling_disabled",
        )
    temperature = temperatures[0]
    if isinstance(temperature, bool) or not isinstance(
        temperature,
        Real,
    ):
        return GraphResidentGreedyTailDecision(
            False,
            "temperature_invalid",
        )
    if temperature != 0.0:
        return GraphResidentGreedyTailDecision(
            False,
            "nonzero_temperature",
        )
    if input_embeds_present:
        return GraphResidentGreedyTailDecision(
            False,
            "input_embeds_unsupported",
        )
    if return_hidden:
        return GraphResidentGreedyTailDecision(
            False,
            "return_hidden_unsupported",
        )
    if incompatible_modes:
        return GraphResidentGreedyTailDecision(
            False,
            f"incompatible_mode:{incompatible_modes[0]}",
        )
    if not capture_available:
        return GraphResidentGreedyTailDecision(
            False,
            "capture_unavailable",
        )
    if quarantined:
        return GraphResidentGreedyTailDecision(
            False,
            "quarantined",
        )
    if not source_matches:
        return GraphResidentGreedyTailDecision(
            False,
            "source_identity_drift",
        )
    return GraphResidentGreedyTailDecision(True, None)
