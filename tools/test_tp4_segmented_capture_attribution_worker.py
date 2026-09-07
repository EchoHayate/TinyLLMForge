from contextlib import contextmanager, nullcontext
import json
import sys
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


def test_engine_config_is_frozen_qwen38_tp4_manual_capture():
    config = worker.build_engine_config()
    assert config["tensor_parallel_size"] == 4
    assert config["max_num_seqs"] == 8
    assert config["max_model_len"] == 384
    assert config["max_num_batched_tokens"] == 2_048
    assert config["enforce_eager"] is True
    assert config["multi_sequence_cuda_graphs"] is False


def _rank_result(rank):
    controls = (
        ("stitched_p4_repeat_0", ((0, 16), (16, 32), (32, 48), (48, 64))),
        ("stitched_p4_repeat_1", ((0, 16), (16, 32), (32, 48), (48, 64))),
        ("isolated_0_16", ((0, 16),)),
        ("isolated_16_32", ((16, 32),)),
        ("isolated_32_48", ((32, 48),)),
        ("isolated_48_64", ((48, 64),)),
        ("pool_fastest_shared", ((0, 16),)),
        ("pool_fastest_isolated", ((0, 16),)),
        ("pool_slowest_shared", ((16, 32),)),
        ("pool_slowest_isolated", ((16, 32),)),
    )
    phase_rows = []
    scratch_rows = []
    for control_id, ranges in controls:
        for ordinal, (start_layer, end_layer) in enumerate(ranges):
            phase_rows.append({
                "row_id": (
                    f"{control_id}:segment-{ordinal}:rank-{rank}"
                ),
                "rank": rank,
                "control_id": control_id,
                "segment_ordinal": ordinal,
                "start_layer": start_layer,
                "end_layer": end_layer,
                "source_revision": "1" * 40,
                "plan_sha256": "2" * 64,
            })
        checkpoints = (
            ("S0", None),
            ("S1", None),
            ("S2", None),
            *((("S3", ordinal) for ordinal in range(len(ranges)))),
            ("S4", None),
            ("S5", None),
            ("S6", None),
            ("S7", None),
        )
        for checkpoint_ordinal, (checkpoint, segment_ordinal) in enumerate(
            checkpoints
        ):
            scratch_rows.append({
                "row_id": (
                    f"{control_id}:checkpoint-{checkpoint_ordinal}:"
                    f"rank-{rank}"
                ),
                "rank": rank,
                "control_id": control_id,
                "checkpoint": checkpoint,
                "segment_ordinal": segment_ordinal,
                "source_revision": "1" * 40,
                "plan_sha256": "2" * 64,
            })
    return {
        "phase": "A1",
        "rank": rank,
        "run_tag": "a1-r1",
        "source_revision": "1" * 40,
        "plan_sha256": "2" * 64,
        "control_ids": tuple(
            control_id for control_id, _ranges in controls
        ),
        "complete": True,
        "phase_rows": phase_rows,
        "scratch_rows": scratch_rows,
        "benefit": {
            "attributed_segments": 16,
            "first_scratch_divergence": "S4",
            "restore_round_trip_exact": True,
        },
        "cost": {
            "diagnostic_capture_count": 16,
            "diagnostic_synchronization_count": 32,
            "total_worker_duration_ns": 1_000 + rank,
            "scratch_snapshot_cpu_ns": 100 + rank,
            "peak_allocated_delta_bytes": 200 + rank,
            "peak_reserved_delta_bytes": 300 + rank,
        },
    }


class _Engine:
    def __init__(self):
        self.events = []

    def call_model_runner_acknowledged(
        self,
        method_name,
        *args,
        timeout_s,
    ):
        self.events.append((method_name, args, timeout_s))
        if method_name == "arm_segmented_capture_attribution":
            run_tag, source_revision = args
            assert run_tag == "a1-r1"
            assert source_revision == "1" * 40
            return (
                {"rank": 0, "armed": True},
                tuple(
                    SimpleNamespace(
                        rank=rank,
                        result={"rank": rank, "armed": True},
                    )
                    for rank in (1, 2, 3)
                ),
            )
        assert method_name == "segmented_capture_attribution_result"
        return (
            _rank_result(0),
            tuple(
                SimpleNamespace(rank=rank, result=_rank_result(rank))
                for rank in (1, 2, 3)
            ),
        )

    def exit(self):
        self.events.append(("exit", (), None))
        return {
            "rank_exit_codes": [0, 0, 0, 0],
            "process_group_destroyed": True,
            "owned_children_remaining": [],
            "rank_cleanup_receipts": [
                {"rank": rank, "process_group_destroyed": True}
                for rank in range(4)
            ],
        }


def test_run_phase_a1_arms_executes_collects_and_cleans():
    engine = _Engine()
    workloads = []
    result = worker.run_phase_a1(
        model_root="/model",
        run_tag="a1-r1",
        source_revision="1" * 40,
        timeout_s=15.0,
        engine_factory=lambda model_root, **config: engine,
        workload_runner=lambda current: workloads.append(current),
    )
    assert workloads == [engine]
    assert len(result["phase_rows"]) == 64
    assert len(result["scratch_rows"]) == 344
    assert result["process_receipts"]["process_group_destroyed"] is True
    assert result["worker_summary"]["run_tag"] == "a1-r1"
    assert result["worker_summary"]["source_revision"] == "1" * 40
    assert result["worker_summary"]["plan_sha256"] == "2" * 64
    assert result["worker_summary"]["cost"][
        "total_worker_duration_ns"
    ] == 1_003
    assert [event[0] for event in engine.events] == [
        "arm_segmented_capture_attribution",
        "segmented_capture_attribution_result",
        "exit",
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda result: result.update(
                source_revision="3" * 40
            ),
            "identity",
        ),
        (
            lambda result: result["control_ids"].__class__(
                result["control_ids"][:-1]
            ),
            "control",
        ),
        (
            lambda result: result["scratch_rows"].__setitem__(
                0,
                {
                    **result["scratch_rows"][0],
                    "checkpoint": "S7",
                },
            ),
            "checkpoint",
        ),
    ),
)
def test_collect_rank_results_rejects_identity_or_inventory_drift(
    mutation,
    message,
):
    rows = [_rank_result(rank) for rank in range(4)]
    if message == "control":
        rows[3]["control_ids"] = rows[3]["control_ids"][:-1]
    else:
        mutation(rows[3])
    with pytest.raises(RuntimeError, match=message):
        worker.collect_rank_results(
            rows[0],
            tuple(
                SimpleNamespace(rank=rank, result=rows[rank])
                for rank in (1, 2, 3)
            ),
        )


def test_run_phase_a1_always_cleans_after_workload_failure():
    engine = _Engine()

    def fail(_engine):
        raise RuntimeError("workload failed")

    with pytest.raises(RuntimeError, match="workload failed"):
        worker.run_phase_a1(
            model_root="/model",
            run_tag="a1-r1",
            source_revision="1" * 40,
            timeout_s=15.0,
            engine_factory=lambda model_root, **config: engine,
            workload_runner=fail,
        )
    assert engine.events[-1][0] == "exit"


def test_main_persists_incomplete_worker_artifacts_on_failure(
    tmp_path,
    monkeypatch,
):
    failure = {
        "schema_version": worker.WORKER_SCHEMA,
        "phase": "A1",
        "run_tag": "a1-failure",
        "phase_rows": [{"row_id": "completed-phase"}],
        "scratch_rows": [{"row_id": "completed-scratch"}],
        "rank_results": [],
        "process_receipts": {
            "process_group_destroyed": True,
        },
        "worker_summary": {
            "schema_version": worker.WORKER_SCHEMA,
            "phase": "A1",
            "run_tag": "a1-failure",
            "complete": False,
            "first_operational_error": {
                "type": "RuntimeError",
                "message": "capture failed",
            },
        },
    }
    monkeypatch.setattr(
        worker,
        "_parse_args",
        lambda _argv: SimpleNamespace(
            model_root="/model",
            run_tag="a1-failure",
            source_revision="1" * 40,
            output_root=tmp_path,
            timeout_s=1.0,
        ),
    )
    monkeypatch.setattr(
        worker,
        "run_phase_a1",
        lambda **_kwargs: (_ for _ in ()).throw(
            worker.PhaseA1WorkerError(
                "capture failed",
                result=failure,
            )
        ),
    )
    assert worker.main([]) == 1
    assert json.loads(
        (tmp_path / "worker_summary.json").read_text()
    )["first_operational_error"]["message"] == "capture failed"
    assert (
        tmp_path / "phase_rows.jsonl"
    ).read_text().strip() == '{"row_id":"completed-phase"}'


def test_atomic_worker_artifacts_stay_below_output_root(tmp_path):
    result = {
        "schema_version": worker.WORKER_SCHEMA,
        "phase": "A1",
        "run_tag": "a1-r1",
        "phase_rows": [{"row_id": "phase"}],
        "scratch_rows": [{"row_id": "scratch"}],
        "rank_results": [{"rank": rank} for rank in range(4)],
        "process_receipts": {"process_group_destroyed": True},
        "worker_summary": {"complete": True},
    }
    worker.write_worker_artifacts(tmp_path, result)
    assert {
        path.name for path in tmp_path.iterdir()
    } == {
        "phase_rows.jsonl",
        "scratch_rows.jsonl",
        "rank_results.json",
        "process_receipts.json",
        "worker_summary.json",
    }
    assert json.loads(
        (tmp_path / "worker_summary.json").read_text()
    )["complete"] is True


def test_model_runner_mixin_executes_phase_a1_once_under_inference_mode(
    monkeypatch,
):
    state = {"inference": False, "calls": 0}

    class _InferenceMode:
        def __enter__(self):
            state["inference"] = True

        def __exit__(self, *_args):
            state["inference"] = False

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(inference_mode=lambda: _InferenceMode()),
    )

    def execute(runner, **kwargs):
        assert runner.rank == 0
        assert kwargs["run_tag"] == "a1-r1"
        assert kwargs["source_revision"] == "1" * 40
        assert state["inference"] is True
        state["calls"] += 1
        return _rank_result(0)

    monkeypatch.setattr(worker, "execute_runtime_phase_a1", execute)

    class _Base:
        rank = 0

        def run_model(self, *_args, **_kwargs):
            assert state["inference"] is False
            return "downstream"

    class _Runner(worker._SegmentedAttributionModelRunnerMixin, _Base):
        pass

    runner = _Runner()
    assert runner.arm_segmented_capture_attribution(
        "a1-r1",
        "1" * 40,
    ) == {"rank": 0, "armed": True}
    assert runner.run_model("ids", "positions", False) == "downstream"
    assert runner.run_model("ids", "positions", False) == "downstream"
    assert state["calls"] == 1
    assert runner.segmented_capture_attribution_result()["complete"] is True


def test_execute_runtime_phase_a1_allocates_eight_unused_scratch_slots():
    runner = SimpleNamespace(
        rank=2,
        world_size=4,
        block_size=2,
        _physical_num_kvcache_blocks=20,
        _last_hybrid_state_leases=tuple(range(8)),
        _last_hybrid_state_request_ids=tuple(range(100, 108)),
        _last_hybrid_state_token_counts=(1,) * 8,
        kv_cache=torch.zeros(1),
        model=SimpleNamespace(),
    )
    context = SimpleNamespace(
        block_tables=torch.tensor(
            [[0, 1], [2, 3]],
            dtype=torch.int64,
        ),
    )
    captured = {}

    def backend_factory(current_runner, **kwargs):
        assert current_runner is runner
        captured.update(kwargs)
        return SimpleNamespace(plan_sha256=kwargs["plan_sha256"])

    matrix = {
        "controls": (),
        "control_results": [
            {
                "phase_rows": [{"row_id": "phase", "rank": 2}],
                "scratch_rows": [{"row_id": "scratch", "rank": 2}],
                "benefit": {
                    "first_scratch_divergence": "S4",
                    "restore_round_trip_exact": True,
                },
                "cost": {
                    "diagnostic_capture_count": 16,
                    "diagnostic_synchronization_count": 32,
                    "scratch_snapshot_cpu_ns": 100,
                    "peak_allocated_delta_bytes": 200,
                    "peak_reserved_delta_bytes": 300,
                },
            }
        ],
        "isolated_pool_memory_gate_pass": True,
    }
    ticks = iter((100, 1_100))
    result = worker.execute_runtime_phase_a1(
        runner,
        run_tag="a1-r1",
        source_revision="1" * 40,
        input_ids=torch.zeros(8, dtype=torch.int64),
        positions=torch.zeros(8, dtype=torch.int64),
        torch_module=torch,
        context=context,
        temporary_context=lambda **_kwargs: nullcontext(),
        backend_factory=backend_factory,
        matrix_runner=lambda _backend: matrix,
        clock_ns=lambda: next(ticks),
    )
    assert captured["scratch_slots"] == [
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
    ]
    assert result["rank"] == 2
    assert result["phase_rows"][0]["row_id"] == "phase"
    assert result["scratch_rows"][0]["row_id"] == "scratch"
    assert result["cost"]["total_worker_duration_ns"] == 1_000


def test_cuda_backend_run_control_emits_phase_scratch_and_cost_rows():
    contract = worker._load_attribution_contract()

    class Backend:
        rank = 1
        source_revision = "1" * 40
        plan_sha256 = "2" * 64

        def __init__(self):
            self.events = []
            self._ticks = iter(range(0, 10_000, 10))

        def clock_ns(self):
            return next(self._ticks)

        def restore_control_baseline(self):
            self.events.append("restore-baseline")

        def initialize_sentinel(self):
            self.events.append("sentinel")

        def checkpoint(self, name, *, segment_ordinal=None):
            self.events.append(("checkpoint", name, segment_ordinal))
            return {
                "checkpoint": name,
                "segment_ordinal": segment_ordinal,
                "rank": self.rank,
                "key_diff": {"equal_to_s0": True},
                "value_diff": {"equal_to_s0": True},
                "scratch_snapshot_cpu_ns": 2,
            }

        def restore_s0(self):
            self.events.append("restore")

        def scratch_equal_to_s0(self):
            return True

        def run_eager(self):
            self.events.append("eager")

        def capture_segment(self, ordinal):
            graph = _FakeGraph(self.events)
            accounting = contract.CapturePhaseAccounting(
                snapshot_and_prepare_ns=1,
                graph_object_create_ns=1,
                capture_context_enter_ns=1,
                capture_body_ns=1,
                capture_context_exit_and_instantiate_ns=1,
                post_capture_synchronize_ns=1,
                post_capture_restore_ns=0,
                graph_reset_ns=0,
                segment_total_ns=8,
                program_lifecycle_ns=8,
            )
            captured = worker._CapturedAttributionSegment(
                graph=graph,
                shared_pool="pool",
                pool_identity="pool-digest",
                accounting=accounting,
                metadata={
                    "control_id": "stitched_p4_repeat_0",
                    "segment_ordinal": ordinal,
                    "start_layer": ordinal * 16,
                    "end_layer": (ordinal + 1) * 16,
                    "rank": self.rank,
                    "pool_mode": "shared",
                    "allocated_delta_bytes": 10,
                    "reserved_delta_bytes": 20,
                },
            )
            self._captured_segments.append(captured)
            return graph

        def replay(self, graphs):
            self.events.append(("replay", len(graphs)))

        def reset_graph(self, graph):
            graph.reset()

        def synchronize(self):
            self.events.append("synchronize")

        def control_comparison(self, control):
            assert control["kind"] == "stitched"
            return {
                "exact_output": True,
                "exact_output_applicable": True,
                "selected_state_exact": True,
                "unselected_state_unchanged": True,
                "scratch_kv_restored": True,
                "graph_reset": True,
            }

    backend = Backend()
    result = worker._AttributionCudaBackend.run_control(
        backend,
        worker.build_phase_a1_controls()[0],
    )
    assert len(result["phase_rows"]) == 4
    assert len(result["scratch_rows"]) == 11
    assert result["benefit"]["restore_round_trip_exact"] is True
    assert result["cost"]["diagnostic_capture_count"] == 4
    assert result["cost"]["scratch_snapshot_cpu_ns"] == 22
    assert result["allocated_delta_bytes"] == 10
    assert result["reserved_delta_bytes"] == 20
    assert backend.events[-6:] == [
        "graph-reset",
        "graph-reset",
        "graph-reset",
        "graph-reset",
        "synchronize",
        ("checkpoint", "S7", None),
    ]


class _ConcreteAttributionModel:
    def __init__(self):
        self.selected = torch.tensor([1.0, 2.0])
        self.manifest_sha256 = "a" * 64
        self.layer_stack = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    block_type=(
                        "full_attention"
                        if index % 4 == 3
                        else "linear_attention"
                    )
                )
                for index in range(64)
            ],
            state_transaction=SimpleNamespace(
                pool=SimpleNamespace(
                    capacity=4,
                    _tensors={
                        "state": torch.arange(
                            8,
                            dtype=torch.float32,
                        ).reshape(4, 2),
                    },
                )
            ),
        )

    def exact_cuda_graph_lease_manifest(
        self,
        _leases,
        _request_ids,
    ):
        return SimpleNamespace(
            sha256=self.manifest_sha256,
            slot_ids=(0, 2),
        )

    def snapshot_exact_cuda_graph_state(self, _leases):
        return {"selected": self.selected.clone()}

    def restore_exact_cuda_graph_state(self, _leases, snapshot):
        self.selected.copy_(snapshot["selected"])

    def embed_exact_graph_inputs(self, input_ids):
        return input_ids.to(dtype=torch.float32).reshape(-1, 1)

    def run_exact_cuda_graph_layer_range(
        self,
        *,
        hidden_states,
        start_layer,
        end_layer,
        **_kwargs,
    ):
        return SimpleNamespace(
            hidden_states=hidden_states + (end_layer - start_layer),
            candidates=(torch.tensor([float(end_layer)]),),
        )

    def run_exact_cuda_graph_step_by_pool_index(
        self,
        _state_slot_ids,
        _token_counts,
        input_ids,
        positions,
    ):
        self.selected.add_(1)
        return input_ids.to(dtype=torch.float32) + positions


class _ConcreteAttributionRunner(_FakeRunner):
    rank = 0
    world_size = 4

    def __init__(self):
        super().__init__()
        self.model = _ConcreteAttributionModel()
        self._last_hybrid_state_leases = ("lease-0", "lease-2")
        self._last_hybrid_state_request_ids = (10, 12)


def _concrete_backend(*, torch_module=None, clock_ns=None):
    runner = _ConcreteAttributionRunner()
    if torch_module is None:
        torch_module = _TorchFacade()
    backend = worker._AttributionCudaBackend(
        runner,
        scratch_slots=SLOTS,
        run_tag="phase-a1-concrete",
        rank=0,
        torch_module=torch_module,
        input_ids=torch.tensor([3, 4]),
        positions=torch.tensor([7, 8]),
        state_slot_ids=torch.tensor([0, 2]),
        token_counts=(1, 1),
        runtime_context_factory=lambda: nullcontext(),
        source_revision="1" * 40,
        plan_sha256="2" * 64,
        clock_ns=clock_ns,
    )
    return backend, runner


def test_concrete_backend_restores_selected_and_scratch_baselines():
    backend, runner = _concrete_backend()
    backend.initialize_sentinel()
    backend.checkpoint("S0", segment_ordinal=None)
    selected_s0 = runner.model.selected.clone()
    scratch_s0 = runner.snapshot_kv_slots(SLOTS)

    runner.model.selected.add_(10)
    runner.kv_cache.add_(5)
    backend.restore_control_baseline()

    assert torch.equal(runner.model.selected, selected_s0)
    assert _snapshots_equal(
        runner.snapshot_kv_slots(SLOTS),
        scratch_s0,
    )


def test_concrete_backend_synchronizes_initial_selected_restore():
    torch_facade = _TorchFacade()
    backend, _runner = _concrete_backend(
        torch_module=torch_facade,
    )
    backend.restore_control_baseline()
    assert torch_facade.cuda.synchronize_count == 1
    assert backend._synchronization_count == 1


def test_concrete_backend_revalidates_lease_manifest_before_replay():
    backend, runner = _concrete_backend()
    runner.model.manifest_sha256 = "b" * 64
    with pytest.raises(RuntimeError, match="lease manifest drift"):
        backend.replay(())


def test_concrete_backend_prepares_and_installs_isolated_hidden():
    backend, _runner = _concrete_backend()
    hidden = backend.embed_inputs()
    prepared = backend.run_eager_prefix(
        hidden,
        start_layer=0,
        end_layer=16,
    )
    backend.install_isolated_hidden(prepared)
    backend.prepare_segment(
        SimpleNamespace(start_layer=16, end_layer=32),
        ordinal=0,
    )
    assert torch.equal(backend._hidden, prepared)


def test_concrete_backend_uses_shared_pool_only_for_shared_control(
    monkeypatch,
):
    backend, _runner = _concrete_backend()
    calls = []

    def capture(current, segment, **kwargs):
        calls.append((segment.start_layer, kwargs["shared_pool"]))
        return worker._CapturedAttributionSegment(
            graph=f"graph-{len(calls)}",
            shared_pool=f"pool-{len(calls)}",
            pool_identity=f"digest-{len(calls)}",
            accounting=SimpleNamespace(),
            metadata={},
        )

    monkeypatch.setattr(worker, "capture_attributed_segment", capture)
    backend._current_control = worker.build_phase_a1_controls()[0]
    backend._captured_segments = []
    backend.capture_segment(0)
    backend.capture_segment(1)
    assert calls == [(0, None), (16, "pool-1")]

    calls.clear()
    backend._current_control = {
        **worker.build_phase_a1_controls()[2],
        "pool_mode": "isolated",
    }
    backend._captured_segments = []
    backend.capture_segment(0)
    assert calls == [(0, None)]


def test_concrete_backend_gathers_all_four_rank_isolated_rows():
    class _Distributed:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def all_gather_object(output, local):
            for rank in range(4):
                output[rank] = [
                    {
                        **local[0],
                        "rank": rank,
                    }
                ]

    torch_facade = _TorchFacade()
    torch_facade.distributed = _Distributed()
    backend, _runner = _concrete_backend(torch_module=torch_facade)
    rows = backend.gather_isolated_rows([
        {
            "phase_rows": [{
                "rank": 0,
                "start_layer": 0,
                "end_layer": 16,
            }],
        }
    ])
    assert [row["rank"] for row in rows] == [0, 1, 2, 3]


def test_concrete_backend_measures_restore_and_idempotent_reset():
    ticks = iter(range(0, 1_000, 10))
    backend, runner = _concrete_backend(
        clock_ns=lambda: next(ticks),
    )
    backend.initialize_sentinel()
    backend.checkpoint("S0", segment_ordinal=None)
    for _ in range(3):
        backend.restore_s0()
    assert backend._post_capture_restore_ns == 10

    events = []
    graph = _FakeGraph(events)
    backend._active_graphs = [graph]
    backend.reset_graph(graph)
    backend.reset_graph(graph)
    assert events == ["graph-reset"]
    assert backend._graph_reset_durations_ns[id(graph)] == 10
    assert backend._active_graphs == []
    assert backend._synchronization_count >= 5
