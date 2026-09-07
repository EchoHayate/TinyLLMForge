from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import tp4_segmented_capture_attribution_worker as worker


SLOTS = [2, 5]


class _CudaFacade:
    def __init__(self):
        self.synchronize_count = 0

    def synchronize(self):
        self.synchronize_count += 1


class _TorchFacade:
    Tensor = torch.Tensor
    uint8 = torch.uint8
    int64 = torch.int64

    def __init__(self):
        self.cuda = _CudaFacade()

    def __getattr__(self, name):
        return getattr(torch, name)


class _FakeRunner:
    block_size = 2

    def __init__(self):
        self.kv_cache = torch.zeros(
            2,
            3,
            4,
            self.block_size,
            2,
            4,
            dtype=torch.bfloat16,
        )

    def snapshot_kv_slots(self, physical_slots):
        block_ids = torch.tensor(
            [slot // self.block_size for slot in physical_slots],
            dtype=torch.long,
        )
        offsets = torch.tensor(
            [slot % self.block_size for slot in physical_slots],
            dtype=torch.long,
        )
        return {
            "keys": (
                self.kv_cache[0, :, block_ids, offsets]
                .detach()
                .cpu()
                .clone()
            ),
            "values": (
                self.kv_cache[1, :, block_ids, offsets]
                .detach()
                .cpu()
                .clone()
            ),
        }

    def restore_kv_slots(self, physical_slots, snapshot):
        for slot_ordinal, physical_slot in enumerate(physical_slots):
            block_id = physical_slot // self.block_size
            offset = physical_slot % self.block_size
            self.kv_cache[0, :, block_id, offset].copy_(
                snapshot["keys"][:, slot_ordinal]
            )
            self.kv_cache[1, :, block_id, offset].copy_(
                snapshot["values"][:, slot_ordinal]
            )


def _snapshots_equal(left, right):
    return all(
        torch.equal(left[name], right[name])
        for name in ("keys", "values")
    )


def test_sentinel_is_nonzero_deterministic_and_rank_sensitive():
    torch_facade = _TorchFacade()
    first = _FakeRunner()
    second = _FakeRunner()
    other_rank = _FakeRunner()

    worker.fill_scratch_sentinel(
        first,
        SLOTS,
        run_tag="a1-r1",
        rank=0,
        torch_module=torch_facade,
    )
    worker.fill_scratch_sentinel(
        second,
        SLOTS,
        run_tag="a1-r1",
        rank=0,
        torch_module=_TorchFacade(),
    )
    worker.fill_scratch_sentinel(
        other_rank,
        SLOTS,
        run_tag="a1-r1",
        rank=1,
        torch_module=_TorchFacade(),
    )

    first_snapshot = first.snapshot_kv_slots(SLOTS)
    second_snapshot = second.snapshot_kv_slots(SLOTS)
    other_snapshot = other_rank.snapshot_kv_slots(SLOTS)
    assert _snapshots_equal(first_snapshot, second_snapshot)
    assert not _snapshots_equal(first_snapshot, other_snapshot)
    assert all(
        bool(torch.count_nonzero(tensor))
        for tensor in first_snapshot.values()
    )
    assert torch_facade.cuda.synchronize_count == 1


def test_sentinel_handles_full_width_unsigned_sha_seed():
    runner = SimpleNamespace(
        block_size=1,
        kv_cache=torch.zeros(
            2,
            64,
            8,
            1,
            1,
            1,
            dtype=torch.bfloat16,
        ),
    )
    worker.fill_scratch_sentinel(
        runner,
        list(range(8)),
        run_tag="phase-a1",
        rank=3,
        torch_module=_TorchFacade(),
    )
    assert bool(torch.count_nonzero(runner.kv_cache))


def test_checkpoint_hashes_canonical_bytes_and_bounds_diff():
    runner = _FakeRunner()
    torch_facade = _TorchFacade()
    worker.fill_scratch_sentinel(
        runner,
        SLOTS,
        run_tag="a1-r2",
        rank=0,
        torch_module=torch_facade,
    )
    s0 = runner.snapshot_kv_slots(SLOTS)
    baseline = worker.snapshot_scratch_checkpoint(
        runner,
        SLOTS,
        checkpoint="S0",
        rank=0,
        s0=None,
        synchronized=True,
        segment_ordinal=None,
        torch_module=torch_facade,
    )
    repeated = worker.snapshot_scratch_checkpoint(
        runner,
        SLOTS,
        checkpoint="S1",
        rank=0,
        s0=s0,
        synchronized=True,
        segment_ordinal=None,
        torch_module=torch_facade,
    )
    assert baseline["keys"]["sha256"] == repeated["keys"]["sha256"]
    assert repeated["key_diff"]["equal_to_s0"] is True

    runner.kv_cache[1, 1, 1, 0, 1, 3] += 0.5
    changed = worker.snapshot_scratch_checkpoint(
        runner,
        SLOTS,
        checkpoint="S3",
        rank=0,
        s0=s0,
        synchronized=True,
        segment_ordinal=0,
        torch_module=torch_facade,
    )
    assert changed["values"]["sha256"] != baseline["values"]["sha256"]
    assert {
        key: value
        for key, value in changed["value_diff"].items()
        if key != "max_absolute_difference"
    } == {
        "equal_to_s0": False,
        "mismatching_element_count": 1,
        "first_mismatch": {
            "layer": 1,
            "scratch_slot_ordinal": 0,
            "head": 1,
            "element_offset": 3,
        },
    }
    assert changed["value_diff"]["max_absolute_difference"] == pytest.approx(
        0.5,
        abs=0.01,
    )
    serialized_names = set(changed)
    assert not serialized_names.intersection(
        {"bytes", "data", "tensor", "base64"}
    )


def test_first_scratch_divergence_uses_checkpoint_order():
    rows = [
        {"checkpoint": "S0", "key_diff": {"equal_to_s0": True},
         "value_diff": {"equal_to_s0": True}},
        {"checkpoint": "S1", "key_diff": {"equal_to_s0": True},
         "value_diff": {"equal_to_s0": True}},
        {"checkpoint": "S2", "key_diff": {"equal_to_s0": True},
         "value_diff": {"equal_to_s0": True}},
        {"checkpoint": "S3", "segment_ordinal": 0,
         "key_diff": {"equal_to_s0": False},
         "value_diff": {"equal_to_s0": True}},
        {"checkpoint": "S4", "key_diff": {"equal_to_s0": True},
         "value_diff": {"equal_to_s0": True}},
    ]
    assert worker.first_scratch_divergence(rows) == "S3"
    assert worker.first_scratch_divergence(rows[:3]) is None


class _SequenceBackend:
    def __init__(
        self,
        *,
        round_trip_exact=True,
        eager_error=None,
        capture_error_ordinal=None,
        restore_error=None,
        reset_error=None,
    ):
        self.events = []
        self.round_trip_exact = round_trip_exact
        self.eager_error = eager_error
        self.capture_error_ordinal = capture_error_ordinal
        self.restore_error = restore_error
        self.reset_error = reset_error
        self.graphs = []

    def initialize_sentinel(self):
        self.events.append("sentinel")

    def checkpoint(self, name, *, segment_ordinal=None):
        self.events.append(("checkpoint", name, segment_ordinal))
        return {
            "checkpoint": name,
            "segment_ordinal": segment_ordinal,
            "key_diff": {"equal_to_s0": True},
            "value_diff": {"equal_to_s0": True},
        }

    def restore_s0(self):
        self.events.append("restore")
        if self.restore_error is not None:
            error, self.restore_error = self.restore_error, None
            raise error

    def scratch_equal_to_s0(self):
        self.events.append("round_trip")
        return self.round_trip_exact

    def run_eager(self):
        self.events.append("eager")
        if self.eager_error is not None:
            raise self.eager_error

    def capture_segment(self, ordinal):
        if ordinal == self.capture_error_ordinal:
            self.events.append(("capture", ordinal))
            raise RuntimeError("capture failed")
        graph = f"graph-{ordinal}"
        self.events.append(("capture", ordinal))
        self.graphs.append(graph)
        return graph

    def replay(self, graphs):
        self.events.append(("replay", tuple(graphs)))

    def reset_graph(self, graph):
        self.events.append(("reset", graph))
        if self.reset_error is not None:
            error, self.reset_error = self.reset_error, None
            raise error

    def synchronize(self):
        self.events.append("synchronize")


def test_forensic_sequence_orders_restore_checkpoints_and_reverse_reset():
    backend = _SequenceBackend()
    result = worker.run_scratch_forensic_sequence(
        backend,
        segment_count=2,
    )
    assert result["restore_round_trip_exact"] is True
    assert [row["checkpoint"] for row in result["checkpoint_rows"]] == [
        "S0",
        "S1",
        "S2",
        "S3",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
    ]
    assert backend.events == [
        "sentinel",
        ("checkpoint", "S0", None),
        "restore",
        "round_trip",
        "eager",
        ("checkpoint", "S1", None),
        "restore",
        ("checkpoint", "S2", None),
        ("capture", 0),
        ("checkpoint", "S3", 0),
        ("capture", 1),
        ("checkpoint", "S3", 1),
        "restore",
        ("checkpoint", "S4", None),
        ("replay", ("graph-0", "graph-1")),
        ("checkpoint", "S5", None),
        "restore",
        ("checkpoint", "S6", None),
        "restore",
        ("reset", "graph-1"),
        ("reset", "graph-0"),
        "synchronize",
        ("checkpoint", "S7", None),
    ]


def test_restore_round_trip_failure_stops_before_eager_or_capture():
    backend = _SequenceBackend(round_trip_exact=False)
    with pytest.raises(RuntimeError, match="scratch_restore_primitive"):
        worker.run_scratch_forensic_sequence(
            backend,
            segment_count=2,
        )
    assert "eager" not in backend.events
    assert not any(
        isinstance(event, tuple) and event[0] == "capture"
        for event in backend.events
    )
    assert ("checkpoint", "S7", None) in backend.events


def test_source_error_remains_primary_when_cleanup_also_fails():
    backend = _SequenceBackend(
        capture_error_ordinal=1,
        reset_error=RuntimeError("reset failed"),
    )
    with pytest.raises(RuntimeError, match="capture failed"):
        worker.run_scratch_forensic_sequence(
            backend,
            segment_count=2,
        )
    assert ("checkpoint", "S7", None) in backend.events


def test_attribution_backend_preserves_none_logits_on_non_root_rank():
    backend = object.__new__(worker._AttributionCudaBackend)
    backend.rank = 3
    assert backend.clone_logits(None) is None


class _FakeGraph:
    def __init__(self, events):
        self.events = events

    def pool(self):
        return "resolved-pool"

    def reset(self):
        self.events.append("graph-reset")


class _PhaseBackend:
    def __init__(self):
        self.events = []
        self._ticks = iter(range(0, 1_000, 10))
        self.source_revision = "1" * 40
        self.plan_sha256 = "2" * 64
        self.rank = 0

    def clock_ns(self):
        return next(self._ticks)

    def memory_snapshot(self):
        self.events.append("memory")
        if self.events.count("memory") == 1:
            return {"allocated_bytes": 100, "reserved_bytes": 200}
        return {"allocated_bytes": 140, "reserved_bytes": 280}

    def prepare_segment(self, segment, *, ordinal):
        self.events.append(("prepare", segment.start_layer, ordinal))

    def create_graph(self):
        self.events.append("graph-create")
        return _FakeGraph(self.events)

    @contextmanager
    def capture_context(self, graph, *, shared_pool):
        self.events.append(("capture-enter", graph, shared_pool))
        yield
        self.events.append("capture-exit")

    def execute_capture_body(self, segment, *, ordinal):
        self.events.append(("body", segment.end_layer, ordinal))

    def synchronize(self):
        self.events.append("capture-sync")

    def resolve_pool(self, graph, *, shared_pool):
        return graph.pool() if shared_pool is None else shared_pool

    def pool_identity(self, pool, *, pool_mode):
        assert pool in {"resolved-pool", "shared-pool"}
        return f"{pool_mode}-pool-digest"

    def segment_metadata(self, segment, *, ordinal):
        return {
            "linear_attention_layer_count": 12,
            "full_attention_layer_count": 4,
            "candidate_tensor_count": 24,
            "candidate_tensor_bytes": 4_096,
            "stable_hidden_candidate_logits_bytes": 8_192,
            "collectives": {
                "available": False,
                "counts": {},
                "unavailable_reason": "existing_receipt_not_exposed",
            },
            "cuda_stream_identity": "stream-0",
        }


def test_capture_phase_timestamps_are_non_overlapping_and_named():
    backend = _PhaseBackend()
    segment = SimpleNamespace(start_layer=0, end_layer=16)
    captured = worker.capture_attributed_segment(
        backend,
        segment,
        ordinal=0,
        control_id="stitched_p4_repeat_0",
        pool_mode="shared",
        shared_pool=None,
    )
    accounting = captured.accounting
    assert accounting.snapshot_and_prepare_ns == 10
    assert accounting.graph_object_create_ns == 10
    assert accounting.capture_context_enter_ns == 10
    assert accounting.capture_body_ns == 10
    assert accounting.capture_context_exit_and_instantiate_ns == 10
    assert accounting.post_capture_synchronize_ns == 10
    assert accounting.post_capture_restore_ns == 0
    assert accounting.graph_reset_ns == 0
    assert accounting.segment_total_ns == 80
    assert accounting.program_lifecycle_ns == 80
    assert captured.pool_identity == "shared-pool-digest"
    assert captured.shared_pool == "resolved-pool"


def test_segment_metadata_counts_layer_types_from_model_modules():
    model = SimpleNamespace(
        layer_stack=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    block_type=(
                        "full_attention"
                        if index % 4 == 3
                        else "linear_attention"
                    )
                )
                for index in range(64)
            ]
        )
    )
    for start, end in ((0, 16), (16, 32), (32, 48), (48, 64)):
        assert worker.layer_type_inventory(model, start, end) == {
            "linear_attention_layer_count": 12,
            "full_attention_layer_count": 4,
        }


def test_unique_tensor_bytes_does_not_double_count_aliases():
    tensor = torch.zeros(4, dtype=torch.float32)
    assert worker.count_unique_tensor_bytes(
        {
            "hidden": tensor,
            "again": tensor,
            "nested": (torch.zeros(2, dtype=torch.int64),),
        },
        torch_module=torch,
    ) == 32


def test_capture_metadata_binds_memory_pool_source_and_plan():
    backend = _PhaseBackend()
    captured = worker.capture_attributed_segment(
        backend,
        SimpleNamespace(start_layer=16, end_layer=32),
        ordinal=1,
        control_id="stitched_p4_repeat_0",
        pool_mode="shared",
        shared_pool="shared-pool",
    )
    assert captured.metadata == {
        "control_id": "stitched_p4_repeat_0",
        "segment_ordinal": 1,
        "start_layer": 16,
        "end_layer": 32,
        "rank": 0,
        "source_revision": "1" * 40,
        "plan_sha256": "2" * 64,
        "pool_mode": "shared",
        "pool_identity": "shared-pool-digest",
        "allocated_before_bytes": 100,
        "allocated_after_bytes": 140,
        "allocated_delta_bytes": 40,
        "reserved_before_bytes": 200,
        "reserved_after_bytes": 280,
        "reserved_delta_bytes": 80,
        "linear_attention_layer_count": 12,
        "full_attention_layer_count": 4,
        "candidate_tensor_count": 24,
        "candidate_tensor_bytes": 4_096,
        "stable_hidden_candidate_logits_bytes": 8_192,
        "collectives": {
            "available": False,
            "counts": {},
            "unavailable_reason": "existing_receipt_not_exposed",
        },
        "cuda_stream_identity": "stream-0",
    }


def test_finalize_capture_accounting_keeps_restore_and_reset_measured():
    captured = worker.capture_attributed_segment(
        _PhaseBackend(),
        SimpleNamespace(start_layer=0, end_layer=16),
        ordinal=0,
        control_id="isolated_0_16",
        pool_mode="isolated",
        shared_pool=None,
    )
    finalized = worker.finalize_captured_segment(
        captured,
        post_capture_restore_ns=5,
        graph_reset_ns=7,
        program_lifecycle_ns=200,
    )
    assert finalized.accounting.post_capture_restore_ns == 5
    assert finalized.accounting.graph_reset_ns == 7
    assert finalized.accounting.segment_total_ns == 92
    assert finalized.accounting.program_lifecycle_ns == 200
    assert finalized.graph is captured.graph


def test_phase_a1_base_matrix_is_fixed_and_bounded():
    controls = worker.build_phase_a1_controls()
    assert tuple(
        (control["control_id"], control["ranges"])
        for control in controls
    ) == (
        (
            "stitched_p4_repeat_0",
            ((0, 16), (16, 32), (32, 48), (48, 64)),
        ),
        (
            "stitched_p4_repeat_1",
            ((0, 16), (16, 32), (32, 48), (48, 64)),
        ),
        ("isolated_0_16", ((0, 16),)),
        ("isolated_16_32", ((16, 32),)),
        ("isolated_32_48", ((32, 48),)),
        ("isolated_48_64", ((48, 64),)),
    )
    assert controls[0]["formal_route_row"] is True
    assert controls[1]["formal_route_row"] is False
    assert all(
        len(control["ranges"]) <= 4
        for control in controls
    )


def _isolated_phase_rows():
    totals = {
        (0, 16): (600, 610, 620, 630),
        (16, 32): (2_400, 2_500, 2_450, 2_550),
        (32, 48): (600, 610, 620, 630),
        (48, 64): (2_300, 2_400, 2_500, 2_520),
    }
    rows = []
    for (start, end), rank_totals in totals.items():
        for rank, total in enumerate(rank_totals):
            row = {
                "rank": rank,
                "control_id": f"isolated_{start}_{end}",
                "segment_ordinal": 0,
                "start_layer": start,
                "end_layer": end,
                "snapshot_and_prepare_ns": 10,
                "graph_object_create_ns": 10,
                "capture_context_enter_ns": 10,
                "capture_body_ns": total - 140,
                "capture_context_exit_and_instantiate_ns": 10,
                "post_capture_synchronize_ns": 10,
                "post_capture_restore_ns": 10,
                "graph_reset_ns": 10,
                "segment_total_ns": total,
                "program_lifecycle_ns": total + 100,
            }
            rows.append(row)
    return rows


def test_pool_controls_select_tp_wide_fastest_and_slowest_ranges():
    fastest, slowest = worker.select_pool_control_ranges(
        _isolated_phase_rows()
    )
    assert fastest == (0, 16)
    assert slowest == (16, 32)
    controls = worker.build_pool_controls(fastest, slowest)
    assert tuple(
        (
            control["control_id"],
            control["ranges"],
            control["pool_mode"],
        )
        for control in controls
    ) == (
        ("pool_fastest_shared", ((0, 16),), "shared"),
        ("pool_fastest_isolated", ((0, 16),), "isolated"),
        ("pool_slowest_shared", ((16, 32),), "shared"),
        ("pool_slowest_isolated", ((16, 32),), "isolated"),
    )


def test_pool_control_ties_choose_lexicographically_smallest_range():
    rows = _isolated_phase_rows()
    for row in rows:
        row["segment_total_ns"] = 1_000
        row["program_lifecycle_ns"] = 1_100
        row["capture_body_ns"] = 900
    assert worker.select_pool_control_ranges(rows) == (
        (0, 16),
        (0, 16),
    )


class _MatrixBackend:
    def __init__(self):
        self.events = []

    def restore_control_baseline(self):
        self.events.append("restore-baseline")

    def reset_control_graphs(self):
        self.events.append("reset-control-graphs")

    def synchronize(self):
        self.events.append("synchronize")

    def embed_inputs(self):
        self.events.append("embed")
        return "embedded"

    def run_eager_prefix(self, hidden, *, start_layer, end_layer):
        self.events.append(
            ("prefix", hidden, start_layer, end_layer)
        )
        return f"hidden-{end_layer}"

    def install_isolated_hidden(self, hidden):
        self.events.append(("install-hidden", hidden))

    def clock_ns(self):
        return len(self.events) * 10

    def run_control(self, control):
        self.events.append(("run-control", control["control_id"]))
        return {
            "control_id": control["control_id"],
            "ranges": control["ranges"],
            "pool_mode": control["pool_mode"],
            "allocated_delta_bytes": (
                200 if control["pool_mode"] == "isolated" else 100
            ),
            "reserved_delta_bytes": (
                300 if control["pool_mode"] == "isolated" else 150
            ),
        }

    def gather_isolated_rows(self, local_results):
        assert len(local_results) == 4
        self.events.append("gather-isolated")
        return _isolated_phase_rows()


def test_isolated_range_prepares_prefix_outside_capture_measurement():
    backend = _MatrixBackend()
    result = worker.run_isolated_range(
        backend,
        start_layer=16,
        end_layer=32,
        pool_mode="isolated",
        control_id="isolated_16_32",
    )
    assert result["eager_prefix_prepare_ns"] > 0
    assert backend.events[:5] == [
        "restore-baseline",
        "embed",
        ("prefix", "embedded", 0, 16),
        "synchronize",
        ("install-hidden", "hidden-16"),
    ]
    assert backend.events[-1] == (
        "run-control",
        "isolated_16_32",
    )


def test_zero_start_isolated_range_has_zero_prefix_duration():
    backend = _MatrixBackend()
    result = worker.run_isolated_range(
        backend,
        start_layer=0,
        end_layer=16,
        pool_mode="isolated",
        control_id="isolated_0_16",
    )
    assert result["eager_prefix_prepare_ns"] == 0
    assert not any(
        isinstance(event, tuple) and event[0] == "prefix"
        for event in backend.events
    )


def test_second_stitched_repeat_has_explicit_reset_boundary():
    backend = _MatrixBackend()
    worker.run_stitched_repeat(backend, repeat_ordinal=0)
    assert backend.events[0] == (
        "run-control",
        "stitched_p4_repeat_0",
    )

    backend = _MatrixBackend()
    result = worker.run_stitched_repeat(
        backend,
        repeat_ordinal=1,
    )
    assert backend.events[:4] == [
        "restore-baseline",
        "reset-control-graphs",
        "synchronize",
        ("run-control", "stitched_p4_repeat_1"),
    ]
    assert result["formal_route_row"] is False


def test_phase_a1_matrix_runs_only_six_base_and_four_pool_controls():
    backend = _MatrixBackend()
    result = worker.run_phase_a1_matrix(backend)
    assert len(result["control_results"]) == 10
    assert sum(
        len(control["ranges"])
        for control in result["controls"]
    ) == 16
    assert result["fastest_isolated_range"] == (0, 16)
    assert result["slowest_isolated_range"] == (16, 32)
    assert result["isolated_pool_memory_gate_pass"] is True
    assert {
        control["control_id"]
        for control in result["controls"]
    } == {
        "stitched_p4_repeat_0",
        "stitched_p4_repeat_1",
        "isolated_0_16",
        "isolated_16_32",
        "isolated_32_48",
        "isolated_48_64",
        "pool_fastest_shared",
        "pool_fastest_isolated",
        "pool_slowest_shared",
        "pool_slowest_isolated",
    }


def test_isolated_pool_memory_gate_uses_frozen_half_gib_limit():
    rows = [
        {
            "pool_mode": "isolated",
            "allocated_delta_bytes": 512 * 1024 * 1024,
            "reserved_delta_bytes": 512 * 1024 * 1024,
        }
    ]
    assert worker.isolated_pool_memory_gate_pass(rows) is True
    rows[0]["reserved_delta_bytes"] += 1
    assert worker.isolated_pool_memory_gate_pass(rows) is False
