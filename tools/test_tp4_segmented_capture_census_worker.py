from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import tp4_segmented_capture_census_worker as worker


def test_candidate_plans_are_canonical_and_source_bound():
    assert worker.CANDIDATE_PLANS == {
        "p2": ((0, 32), (32, 64)),
        "p3": ((0, 22), (22, 43), (43, 64)),
        "p4": ((0, 16), (16, 32), (32, 48), (48, 64)),
    }
    hashes = {
        worker.build_segment_plan(plan_id).sha256
        for plan_id in worker.CANDIDATE_PLANS
    }
    assert len(hashes) == 3


def test_rank_rows_bind_plan_segment_and_rank():
    rows = worker.collect_rank_plan_rows(
        "p2",
        {
            "rank": 0,
            "plan_id": "p2",
            "plan_sha256": worker.build_segment_plan("p2").sha256,
            "segment_rows": [
                {"segment_ordinal": 0},
                {"segment_ordinal": 1},
            ],
        },
        (
            SimpleNamespace(
                rank=1,
                result={
                    "rank": 1,
                    "plan_id": "p2",
                    "plan_sha256": (
                        worker.build_segment_plan("p2").sha256
                    ),
                    "segment_rows": [
                        {"segment_ordinal": 0},
                        {"segment_ordinal": 1},
                    ],
                },
            ),
            SimpleNamespace(
                rank=2,
                result={
                    "rank": 2,
                    "plan_id": "p2",
                    "plan_sha256": (
                        worker.build_segment_plan("p2").sha256
                    ),
                    "segment_rows": [
                        {"segment_ordinal": 0},
                        {"segment_ordinal": 1},
                    ],
                },
            ),
            SimpleNamespace(
                rank=3,
                result={
                    "rank": 3,
                    "plan_id": "p2",
                    "plan_sha256": (
                        worker.build_segment_plan("p2").sha256
                    ),
                    "segment_rows": [
                        {"segment_ordinal": 0},
                        {"segment_ordinal": 1},
                    ],
                },
            ),
        ),
    )

    assert [row["row_id"] for row in rows] == [
        f"p2:segment-{segment}:rank-{rank}"
        for rank in range(4)
        for segment in range(2)
    ]


def test_rank_rows_reject_missing_or_disagreeing_ranks():
    local = {
        "rank": 0,
        "plan_id": "p3",
        "plan_sha256": worker.build_segment_plan("p3").sha256,
        "segment_rows": [],
    }
    with pytest.raises(RuntimeError, match="rank inventory"):
        worker.collect_rank_plan_rows("p3", local, ())
    with pytest.raises(RuntimeError, match="plan identity"):
        worker.collect_rank_plan_rows(
            "p3",
            local,
            tuple(
                SimpleNamespace(
                    rank=rank,
                    result={
                        **local,
                        "rank": rank,
                        "plan_id": "wrong" if rank == 2 else "p3",
                    },
                )
                for rank in (1, 2, 3)
            ),
        )


def test_engine_config_is_exact_q1_tp4_and_manual_capture_only():
    config = worker.build_engine_config()
    assert config["tensor_parallel_size"] == 4
    assert config["max_num_seqs"] == 8
    assert config["max_model_len"] == 384
    assert config["enforce_eager"] is True
    assert config["multi_sequence_cuda_graphs"] is False
    assert config["multi_sequence_cuda_graph_dynamic_pool_indices"] is False


class _Engine:
    def __init__(self, plan_id, *, fail_workload=False):
        self.plan_id = plan_id
        self.fail_workload = fail_workload
        self.events = []

    def call_model_runner_acknowledged(
        self,
        method_name,
        *args,
        timeout_s,
    ):
        self.events.append((method_name, args, timeout_s))
        if method_name == "arm_segmented_capture_census":
            assert args == (self.plan_id,)
            local = {"rank": 0, "armed": True}
            remote = tuple(
                SimpleNamespace(
                    rank=rank,
                    result={"rank": rank, "armed": True},
                )
                for rank in (1, 2, 3)
            )
            return local, remote
        assert method_name == "segmented_capture_census_result"
        plan = worker.build_segment_plan(self.plan_id)

        def result(rank):
            return {
                "rank": rank,
                "plan_id": self.plan_id,
                "plan_sha256": plan.sha256,
                "segment_rows": [
                    {
                        "segment_ordinal": ordinal,
                        "segment_capture_duration_ns": 100 + ordinal,
                    }
                    for ordinal in range(
                        len(worker.CANDIDATE_PLANS[self.plan_id])
                    )
                ],
            }

        return result(0), tuple(
            SimpleNamespace(rank=rank, result=result(rank))
            for rank in (1, 2, 3)
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


def test_run_plan_arms_executes_collects_and_cleans():
    engine = _Engine("p2")
    workloads = []
    result = worker.run_plan_census(
        model_root="/model",
        plan_id="p2",
        timeout_s=15.0,
        engine_factory=lambda model_root, **config: engine,
        workload_runner=lambda current: workloads.append(current),
    )
    assert workloads == [engine]
    assert len(result["rows"]) == 8
    assert result["cleanup"]["process_group_destroyed"] is True
    assert [event[0] for event in engine.events] == [
        "arm_segmented_capture_census",
        "segmented_capture_census_result",
        "exit",
    ]


def test_run_plan_always_cleans_after_workload_failure():
    engine = _Engine("p3")

    def fail(_engine):
        raise RuntimeError("workload failed")

    with pytest.raises(RuntimeError, match="workload failed"):
        worker.run_plan_census(
            model_root="/model",
            plan_id="p3",
            timeout_s=15.0,
            engine_factory=lambda model_root, **config: engine,
            workload_runner=fail,
        )
    assert engine.events[-1][0] == "exit"


def test_run_census_uses_a_fresh_engine_for_every_plan():
    engines = []

    def factory(model_root, **config):
        plan_id = tuple(worker.CANDIDATE_PLANS)[len(engines)]
        engine = _Engine(plan_id)
        engines.append(engine)
        return engine

    result = worker.run_census(
        model_root="/model",
        timeout_s=15.0,
        engine_factory=factory,
        workload_runner=lambda engine: None,
    )
    assert len(engines) == len(worker.CANDIDATE_PLANS)
    assert len({id(engine) for engine in engines}) == 3
    assert len(result["rows"]) == (2 + 3 + 4) * 4
    assert set(result["process_receipts"]) == {"p2", "p3", "p4"}


def test_engine_creation_uses_fresh_ports_and_retries_only_collisions():
    ports = iter((41001, 41002, 41003))
    attempts = []
    cleanup_calls = []
    environment = {}
    expected_engine = object()

    def engine_factory(model_root, **config):
        attempts.append(
            (
                model_root,
                dict(config),
                environment["TINYVLLM_DIST_PORT"],
            )
        )
        if len(attempts) == 1:
            raise RuntimeError(
                "TCPStore failed: EADDRINUSE address already in use"
            )
        return expected_engine

    engine, port = worker.create_engine_with_rendezvous_retry(
        "/model",
        engine_config={"tensor_parallel_size": 4},
        port_factory=lambda: next(ports),
        engine_factory=engine_factory,
        environment=environment,
        cleanup_failed_attempt=lambda: cleanup_calls.append(True),
        sleep=lambda _seconds: None,
        maximum_attempts=3,
    )

    assert engine is expected_engine
    assert port == 41002
    assert [attempt[2] for attempt in attempts] == ["41001", "41002"]
    assert cleanup_calls == [True]

    with pytest.raises(RuntimeError, match="different failure"):
        worker.create_engine_with_rendezvous_retry(
            "/model",
            engine_config={"tensor_parallel_size": 4},
            port_factory=lambda: 41003,
            engine_factory=lambda *_args, **_kwargs: (
                (_ for _ in ()).throw(RuntimeError("different failure"))
            ),
            environment={},
            cleanup_failed_attempt=lambda: cleanup_calls.append(False),
            sleep=lambda _seconds: None,
        )
    assert cleanup_calls == [True]


def test_capture_program_restores_replays_and_resets_in_reverse_order():
    class Backend:
        rank = 2
        world_size = 4

        def __init__(self):
            self.events = []

        def snapshot(self):
            self.events.append("snapshot")
            return "initial"

        def run_eager(self):
            self.events.append("eager")
            return "expected"

        def restore(self, snapshot):
            assert snapshot == "initial"
            self.events.append("restore")

        def capture_segment(self, segment, *, ordinal, shared_pool):
            self.events.append(("capture", ordinal, shared_pool))
            return SimpleNamespace(
                graph=f"graph-{ordinal}",
                shared_pool="pool",
                capture_body_duration_ns=100 + ordinal,
                post_capture_sync_duration_ns=10,
                segment_capture_duration_ns=120 + ordinal,
            )

        def replay(self, graphs):
            self.events.append(("replay", tuple(graphs)))
            return "actual"

        def compare(self, expected, actual):
            assert (expected, actual) == ("expected", "actual")
            self.events.append("compare")
            return {
                "exact_output": True,
                "selected_state_exact": True,
                "unselected_state_unchanged": True,
                "scratch_kv_restored": True,
            }

        def memory_snapshot(self):
            self.events.append("memory")
            return {
                "allocated_bytes": 1_500,
                "reserved_bytes": 2_500,
            }

        def stable_boundary_buffer_bytes(self):
            self.events.append("stable-bytes")
            return 700

        def reset_graph(self, graph):
            self.events.append(("reset", graph))

    ticks = iter((1_000, 5_000))
    backend = Backend()
    result = worker.capture_segment_program(
        "p2",
        backend,
        clock_ns=lambda: next(ticks),
        memory_before={
            "allocated_bytes": 1_000,
            "reserved_bytes": 2_000,
        },
    )
    assert result["lifecycle_duration_ns"] == 4_000
    assert [row["segment_ordinal"] for row in result["segment_rows"]] == [
        0,
        1,
    ]
    assert all(row["graph_reset"] for row in result["segment_rows"])
    assert result["segment_rows"][0]["include_embedding"] is True
    assert result["segment_rows"][0]["include_final"] is False
    assert result["segment_rows"][1]["include_commit"] is True
    assert all(
        row["allocated_delta_bytes"] == 500
        and row["reserved_delta_bytes"] == 500
        and row["stable_boundary_buffer_bytes"] == 700
        for row in result["segment_rows"]
    )
    assert backend.events == [
        "snapshot",
        "eager",
        "restore",
        ("capture", 0, None),
        ("capture", 1, "pool"),
        "restore",
        ("replay", ("graph-0", "graph-1")),
        "compare",
        "restore",
        "memory",
        "stable-bytes",
        ("reset", "graph-1"),
        ("reset", "graph-0"),
    ]


def test_capture_program_can_start_lifecycle_before_backend_allocation():
    class Backend:
        rank = 0
        world_size = 4

        def snapshot(self):
            return "snapshot"

        def run_eager(self):
            return "expected"

        def restore(self, snapshot):
            assert snapshot == "snapshot"

        def capture_segment(self, segment, *, ordinal, shared_pool):
            return SimpleNamespace(
                graph=SimpleNamespace(),
                shared_pool="pool",
                capture_body_duration_ns=100,
                post_capture_sync_duration_ns=10,
                segment_capture_duration_ns=120,
            )

        def replay(self, graphs):
            return "actual"

        def compare(self, expected, actual):
            return {
                "exact_output": True,
                "selected_state_exact": True,
                "unselected_state_unchanged": True,
                "scratch_kv_restored": True,
            }

        def memory_snapshot(self):
            return {
                "allocated_bytes": 1_000,
                "reserved_bytes": 2_000,
            }

        def stable_boundary_buffer_bytes(self):
            return 300

        @staticmethod
        def reset_graph(graph):
            del graph

    result = worker.capture_segment_program(
        "p2",
        Backend(),
        clock_ns=lambda: 5_000,
        lifecycle_started_ns=500,
        memory_before={
            "allocated_bytes": 800,
            "reserved_bytes": 1_700,
        },
    )

    assert result["lifecycle_duration_ns"] == 4_500
    assert all(
        row["allocated_delta_bytes"] == 200
        and row["reserved_delta_bytes"] == 300
        for row in result["segment_rows"]
    )


def test_capture_program_resets_graphs_when_memory_accounting_fails():
    class Backend:
        rank = 0
        world_size = 4

        def __init__(self):
            self.reset = []

        def snapshot(self):
            return "snapshot"

        def run_eager(self):
            return "expected"

        def restore(self, snapshot):
            assert snapshot == "snapshot"

        def capture_segment(self, segment, *, ordinal, shared_pool):
            return SimpleNamespace(
                graph=f"graph-{ordinal}",
                shared_pool="pool",
                capture_body_duration_ns=100,
                post_capture_sync_duration_ns=10,
                segment_capture_duration_ns=120,
            )

        def replay(self, graphs):
            return "actual"

        def compare(self, expected, actual):
            return {
                "exact_output": True,
                "selected_state_exact": True,
                "unselected_state_unchanged": True,
                "scratch_kv_restored": True,
            }

        def memory_snapshot(self):
            raise RuntimeError("memory accounting failed")

        def stable_boundary_buffer_bytes(self):
            return 0

        def reset_graph(self, graph):
            self.reset.append(graph)

    backend = Backend()
    ticks = iter((1_000, 5_000))

    with pytest.raises(RuntimeError, match="memory accounting failed"):
        worker.capture_segment_program(
            "p2",
            backend,
            clock_ns=lambda: next(ticks),
        )

    assert backend.reset == ["graph-1", "graph-0"]


def test_capture_program_resets_graph_returned_without_shared_pool():
    class Backend:
        rank = 0
        world_size = 4

        def __init__(self):
            self.reset = []

        def snapshot(self):
            return "snapshot"

        def run_eager(self):
            return "expected"

        def restore(self, snapshot):
            assert snapshot == "snapshot"

        def capture_segment(self, segment, *, ordinal, shared_pool):
            return SimpleNamespace(
                graph=f"graph-{ordinal}",
                shared_pool=None,
            )

        def reset_graph(self, graph):
            self.reset.append(graph)

    backend = Backend()
    ticks = iter((1_000, 5_000))

    with pytest.raises(RuntimeError, match="shared graph pool"):
        worker.capture_segment_program(
            "p2",
            backend,
            clock_ns=lambda: next(ticks),
        )

    assert backend.reset == ["graph-0"]


def test_cuda_backend_revalidates_lease_identity_before_replay():
    backend = object.__new__(worker._CudaSegmentedCaptureBackend)
    backend.runner = SimpleNamespace(
        _last_hybrid_state_leases=("lease",),
        _last_hybrid_state_request_ids=(17,),
    )
    backend.model = SimpleNamespace(
        exact_cuda_graph_lease_manifest=lambda leases, request_ids: (
            SimpleNamespace(sha256="changed")
        ),
    )
    backend.lease_manifest_sha256 = "captured"
    backend._context = lambda: nullcontext()
    backend.torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: None),
    )
    replayed = []
    graph = SimpleNamespace(replay=lambda: replayed.append(True))

    with pytest.raises(RuntimeError, match="lease manifest drift"):
        backend.replay((graph,))

    assert replayed == []
