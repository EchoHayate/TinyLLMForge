from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ExactGraphSegment:
    start_layer: int
    end_layer: int
    include_embedding: bool
    include_final: bool
    include_commit: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.start_layer, bool)
            or not isinstance(self.start_layer, int)
            or isinstance(self.end_layer, bool)
            or not isinstance(self.end_layer, int)
            or self.start_layer < 0
            or self.end_layer <= self.start_layer
        ):
            raise ValueError("segment layer range is invalid")
        for name in (
            "include_embedding",
            "include_final",
            "include_commit",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")


@dataclass(frozen=True)
class ExactGraphSegmentPlan:
    layer_count: int
    segments: tuple[ExactGraphSegment, ...]
    schema_version: str = (
        "tinyllmforge.segmented-exact-graph-plan.v1"
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_count, bool)
            or not isinstance(self.layer_count, int)
            or self.layer_count <= 0
            or not isinstance(self.segments, tuple)
            or not self.segments
        ):
            raise ValueError(
                "segment plan must contain positive layers"
            )
        cursor = 0
        for ordinal, segment in enumerate(self.segments):
            if not isinstance(segment, ExactGraphSegment):
                raise ValueError(
                    "segments must contain ExactGraphSegment values"
                )
            if segment.start_layer != cursor:
                if ordinal == 0:
                    raise ValueError(
                        "segment ranges must cover every model layer"
                    )
                raise ValueError(
                    "segment ranges must be contiguous"
                )
            if segment.include_embedding != (ordinal == 0):
                raise ValueError(
                    "embedding must belong to the first segment"
                )
            if segment.include_final != (
                ordinal == len(self.segments) - 1
            ):
                raise ValueError(
                    "final stage must belong to the last segment"
                )
            cursor = segment.end_layer
        if cursor != self.layer_count:
            raise ValueError(
                "segment ranges must cover every model layer"
            )
        if (
            sum(segment.include_commit for segment in self.segments)
            != 1
            or not self.segments[-1].include_commit
        ):
            raise ValueError(
                "state commit must belong to the last segment"
            )

    @property
    def sha256(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class CompositeCaptureAccounting:
    segment_capture_durations_ns: tuple[int, ...]
    lifecycle_duration_ns: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.segment_capture_durations_ns, tuple)
            or not self.segment_capture_durations_ns
        ):
            raise ValueError(
                "segment capture durations must be non-empty"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in self.segment_capture_durations_ns
        ):
            raise ValueError(
                "segment capture durations must be non-negative"
            )
        if (
            isinstance(self.lifecycle_duration_ns, bool)
            or not isinstance(self.lifecycle_duration_ns, int)
            or self.lifecycle_duration_ns < 0
        ):
            raise ValueError(
                "lifecycle duration must be non-negative"
            )
        if self.lifecycle_duration_ns < sum(
            self.segment_capture_durations_ns
        ):
            raise ValueError(
                "lifecycle duration cannot be below segment total"
            )

    @property
    def max_segment_capture_duration_ns(self) -> int:
        return max(self.segment_capture_durations_ns)

    @property
    def total_capture_duration_ns(self) -> int:
        return self.lifecycle_duration_ns


class CompositeExactCudaGraph:

    def __init__(
        self,
        *,
        graphs: tuple[object, ...],
        shared_pool: object,
    ):
        if not isinstance(graphs, tuple) or not graphs:
            raise ValueError(
                "composite graph requires at least one graph"
            )
        if any(
            not callable(getattr(graph, "replay", None))
            or not callable(getattr(graph, "reset", None))
            for graph in graphs
        ):
            raise ValueError(
                "every segment graph must support replay/reset"
            )
        self.graphs = graphs
        self.shared_pool = shared_pool
        self._reset = False

    def replay(self) -> None:
        if self._reset:
            raise RuntimeError("composite graph was reset")
        for graph in self.graphs:
            graph.replay()

    def reset(self) -> None:
        if self._reset:
            return
        for graph in reversed(self.graphs):
            graph.reset()
        self._reset = True

    def pool(self):
        return self.shared_pool
