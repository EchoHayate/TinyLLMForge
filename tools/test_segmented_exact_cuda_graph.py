from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tinyvllm"
    / "engine"
    / "segmented_exact_cuda_graph.py"
)
assert MODULE_PATH.is_file(), "segmented exact graph contract is missing"
SPEC = importlib.util.spec_from_file_location(
    "segmented_exact_cuda_graph_under_test",
    MODULE_PATH,
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

CompositeCaptureAccounting = module.CompositeCaptureAccounting
CompositeExactCudaGraph = module.CompositeExactCudaGraph
ExactGraphSegment = module.ExactGraphSegment
ExactGraphSegmentPlan = module.ExactGraphSegmentPlan


def _plan(*ranges):
    segments = []
    for ordinal, (start, end) in enumerate(ranges):
        segments.append(
            ExactGraphSegment(
                start_layer=start,
                end_layer=end,
                include_embedding=ordinal == 0,
                include_final=ordinal == len(ranges) - 1,
                include_commit=ordinal == len(ranges) - 1,
            )
        )
    return ExactGraphSegmentPlan(
        layer_count=ranges[-1][1],
        segments=tuple(segments),
    )


def test_plan_requires_exact_contiguous_coverage():
    assert _plan((0, 22), (22, 43), (43, 64)).layer_count == 64

    with pytest.raises(ValueError, match="contiguous"):
        _plan((0, 22), (23, 64))

    with pytest.raises(ValueError, match="cover"):
        ExactGraphSegmentPlan(
            layer_count=64,
            segments=(
                ExactGraphSegment(1, 64, True, True, True),
            ),
        )


def test_plan_hash_binds_boundaries_and_stage_owners():
    first = _plan((0, 22), (22, 43), (43, 64))
    moved = _plan((0, 21), (21, 43), (43, 64))
    four_segments = _plan(
        (0, 16),
        (16, 32),
        (32, 48),
        (48, 64),
    )

    assert first.sha256 != moved.sha256
    assert first.sha256 != four_segments.sha256


def test_plan_rejects_invalid_stage_owners():
    canonical = _plan((0, 22), (22, 43), (43, 64))

    with pytest.raises(ValueError, match="embedding"):
        replace(
            canonical,
            segments=(
                replace(
                    canonical.segments[0],
                    include_embedding=False,
                ),
                canonical.segments[1],
                canonical.segments[2],
            ),
        )


def test_composite_replay_and_reset_are_ordered():
    events = []

    class Graph:
        def __init__(self, ordinal):
            self.ordinal = ordinal

        def replay(self):
            events.append(("replay", self.ordinal))

        def reset(self):
            events.append(("reset", self.ordinal))

    composite = CompositeExactCudaGraph(
        graphs=(Graph(0), Graph(1), Graph(2)),
        shared_pool="pool",
    )

    composite.replay()
    composite.reset()
    composite.reset()

    assert events == [
        ("replay", 0),
        ("replay", 1),
        ("replay", 2),
        ("reset", 2),
        ("reset", 1),
        ("reset", 0),
    ]
    assert composite.pool() == "pool"

    with pytest.raises(RuntimeError, match="reset"):
        composite.replay()


def test_composite_requires_graph_protocol():
    with pytest.raises(ValueError, match="at least one"):
        CompositeExactCudaGraph(graphs=(), shared_pool="pool")

    with pytest.raises(ValueError, match="replay/reset"):
        CompositeExactCudaGraph(
            graphs=(object(),),
            shared_pool="pool",
        )


def test_capture_accounting_exposes_single_and_total_views():
    accounting = CompositeCaptureAccounting(
        segment_capture_durations_ns=(1_100, 1_300, 900),
        lifecycle_duration_ns=3_700,
    )

    assert accounting.max_segment_capture_duration_ns == 1_300
    assert accounting.total_capture_duration_ns == 3_700


@pytest.mark.parametrize(
    ("durations", "lifecycle", "message"),
    (
        ((), 0, "non-empty"),
        ((1, -1), 1, "non-negative"),
        ((1, 2), -1, "non-negative"),
        ((1, 2), 2, "below segment total"),
    ),
)
def test_capture_accounting_rejects_invalid_values(
    durations,
    lifecycle,
    message,
):
    with pytest.raises(ValueError, match=message):
        CompositeCaptureAccounting(
            segment_capture_durations_ns=durations,
            lifecycle_duration_ns=lifecycle,
        )
