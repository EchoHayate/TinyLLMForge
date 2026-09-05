#!/usr/bin/env python3
"""Dependency-light tests for the TP4 decode replay worker."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from itertools import count
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "tools" / "tp4_decode_replay_worker.py"


def _load_worker():
    spec = importlib.util.spec_from_file_location(
        "tp4_decode_replay_worker",
        WORKER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load worker: {WORKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


worker = _load_worker()


@dataclass
class _Ack:
    rank: int
    result: object


def _event(*, arm, rank, mode="decode", dispatch=None):
    graph = arm == "graph" and mode == "decode"
    return {
        "step_id": 2 if mode == "decode" else 1,
        "request_ids_hash": "1" * 64,
        "mode": mode,
        "active_batch_size": 4,
        "page_table_width": 2,
        "effective_num_splits": 1 if mode == "decode" else None,
        "graph_identity_sha256": "a" * 64 if graph else None,
        "feature_enabled": arm == "graph",
        "dispatch": dispatch or ("graph" if graph else "eager"),
        "cache_state": "ready" if graph else "absent",
        "observation_count": 3 if graph else 0,
        "fallback_reason": (
            None
            if graph
            else ("enforce_eager" if arm == "eager" else "unsupported_mode")
        ),
        "capture_attempted": False,
        "capture_duration_ns": 0,
        "capture_static_bytes": 0,
        "capture_allocated_delta_bytes": 0,
        "capture_reserved_delta_bytes": 0,
        "cache_ready_entries": 1 if graph else 0,
        "cache_static_bytes": 1024 if graph else 0,
        "cache_reserved_delta_bytes": 2048 if graph else 0,
        "cache_total_capture_ns": 50_000_000 if graph else 0,
        "source_sha256": "b" * 64,
        "rank": rank,
        "graph_replay_count": 8 if graph else 0,
    }


class _ObservationEngine:
    def __init__(self, rows):
        self.model_runner = SimpleNamespace(rank=0, world_size=4)
        self.rows = rows

    def call_model_runner_acknowledged(
        self,
        method_name,
        *,
        timeout_s,
    ):
        assert method_name == "cuda_graph_dispatch_observation"
        assert timeout_s > 0
        return self.rows[0], tuple(
            _Ack(rank=rank, result=self.rows[rank])
            for rank in (1, 2, 3)
        )


class _Clock:
    def __init__(self):
        self._values = count(start=1_000_000_000, step=1_000_000)

    def __call__(self):
        return next(self._values)


class _FakeEngine:
    def __init__(self, arm, output_tokens, clock, token_id=17):
        self.arm = arm
        self.output_tokens = output_tokens
        self.clock = clock
        self.token_id = token_id
        self.model_runner = SimpleNamespace(rank=0, world_size=4)
        self.last_step_observation = None
        self.exit_calls = 0
        self.reset_profile_calls = 0
        self.clear_prefix_calls = 0
        self._requests = []
        self._step_index = 0
        self._finished = True

    def add_request(self, prompt_token_ids, sampling):
        del sampling
        if self._finished:
            self._requests = []
            self._step_index = 0
            self._finished = False
        self._requests.append(list(prompt_token_ids))

    def is_finished(self):
        return self._finished

    def step(self):
        step_end_ns = self.clock()
        if self._step_index == 0:
            self._step_index += 1
            self.last_step_observation = {
                "step_end_ns": step_end_ns,
                "new_completion_tokens_by_seq": {},
            }
            return [], 0
        tokens = [self.token_id] * self.output_tokens
        deltas = {
            sequence_id: list(tokens)
            for sequence_id in range(len(self._requests))
        }
        outputs = [
            (sequence_id, list(tokens))
            for sequence_id in range(len(self._requests))
        ]
        self.last_step_observation = {
            "step_end_ns": step_end_ns,
            "new_completion_tokens_by_seq": deltas,
        }
        self._finished = True
        self._step_index += 1
        return outputs, len(outputs)

    def call_model_runner_acknowledged(
        self,
        method_name,
        *,
        timeout_s,
    ):
        assert timeout_s > 0
        assert method_name == "cuda_graph_dispatch_observation"
        mode = "prefill" if self._step_index == 1 else "decode"
        rows = [
            _event(arm=self.arm, rank=rank, mode=mode)
            for rank in range(4)
        ]
        return rows[0], tuple(
            _Ack(rank=rank, result=rows[rank])
            for rank in range(1, 4)
        )

    def configure_decode_internal_profile(
        self,
        enabled,
        profile_label,
        *,
        timeout_s,
    ):
        assert enabled is True
        assert profile_label
        assert timeout_s > 0
        return {"enabled": True, "rank_inventory": [0, 1, 2, 3]}

    def reset_decode_internal_profile(self, *, timeout_s):
        assert timeout_s > 0
        assert self._finished is True
        self.reset_profile_calls += 1
        return tuple(
            {
                "rank": rank,
                "enabled": True,
                "profile_label": "measured",
            }
            for rank in range(4)
        )

    def clear_reusable_prefix_cache(self):
        assert self._finished is True
        assert self.reset_profile_calls == 0
        self.clear_prefix_calls += 1
        return 4

    def finalize_decode_internal_profile(self, *, timeout_s):
        assert timeout_s > 0
        return {
            "enabled": True,
            "rank_inventory": [0, 1, 2, 3],
            "ranks": [
                {
                    "rank": rank,
                    "enabled": True,
                    "finalization_status": "complete",
                    "steps": [],
                    "layers": [],
                    "operations": [],
                    "collectives": [
                        {
                            "step_index": 1,
                            "operation_ordinal": 0,
                            "collective_kind": "all_reduce",
                            "tensor_shape": [4, 4096],
                            "tensor_dtype": "torch.bfloat16",
                        }
                    ],
                }
                for rank in range(4)
            ],
        }

    def reset_peak_memory_stats(self, *, timeout_s):
        assert timeout_s > 0
        return tuple({"rank": rank} for rank in range(4))

    def memory_snapshots(self, *, timeout_s):
        assert timeout_s > 0
        return tuple(
            {
                "rank": rank,
                "cuda_peak_allocated_bytes": 70_000_000_000
                + (100_000_000 if self.arm == "graph" else 0),
                "cuda_peak_reserved_bytes": 71_000_000_000
                + (100_000_000 if self.arm == "graph" else 0),
            }
            for rank in range(4)
        )

    def exit(self):
        self.exit_calls += 1
        return {
            "process_group_destroyed": True,
            "rank_exit_codes": [0, 0, 0, 0],
            "owned_children_remaining": [],
            "rank_cleanup_receipts": [
                {
                    "rank": rank,
                    "process_group_destroyed": True,
                }
                for rank in range(4)
            ],
        }


def test_engine_config_differs_only_by_graph_policy():
    eager = worker.build_engine_config(arm="eager", workload="Q1")
    graph = worker.build_engine_config(arm="graph", workload="Q1")
    assert eager | {
        "enforce_eager": False,
        "multi_sequence_cuda_graphs": True,
    } == graph
    assert eager["tensor_parallel_size"] == 4
    assert eager["gpu_memory_utilization"] == 0.84
    assert eager["enforce_eager"] is True
    assert graph["multi_sequence_cuda_graph_batch_allowlist"] == (2, 4, 8)
    assert graph["max_num_seqs"] == 8
    assert graph["max_model_len"] == 384
    assert graph["max_num_batched_tokens"] == 2048


def test_shared_capacity_engine_config_uses_workload_bounded_kv_capacity():
    with mock.patch.dict(
        os.environ,
        {"TINYLLMFORGE_TP4_ADMISSION_MODE": "shared_capacity"},
    ):
        q0 = worker.build_engine_config(arm="eager", workload="Q0")
        q1 = worker.build_engine_config(arm="graph", workload="Q1")
        q2 = worker.build_engine_config(arm="eager", workload="Q2")

    assert q0["gpu_memory_utilization"] == 0.95
    assert q1["gpu_memory_utilization"] == 0.95
    assert q2["gpu_memory_utilization"] == 0.95
    assert q0["num_kvcache_blocks"] == 8
    assert q1["num_kvcache_blocks"] == 16
    assert q2["num_kvcache_blocks"] == 36


def test_engine_config_rejects_unknown_admission_mode():
    with mock.patch.dict(
        os.environ,
        {"TINYLLMFORGE_TP4_ADMISSION_MODE": "best_effort"},
    ):
        try:
            worker.build_engine_config(arm="eager", workload="Q1")
        except ValueError as exc:
            assert "admission mode" in str(exc)
        else:
            raise AssertionError("unknown admission mode was accepted")


def test_engine_creation_retries_only_rendezvous_port_collisions():
    ports = iter((41001, 41002))
    attempts = []
    cleanup_calls = []
    sleeps = []
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
        Path("/model"),
        engine_config={"tensor_parallel_size": 4},
        port_factory=lambda: next(ports),
        engine_factory=engine_factory,
        environment=environment,
        cleanup_failed_attempt=lambda: cleanup_calls.append(True),
        sleep=lambda seconds: sleeps.append(seconds),
        maximum_attempts=3,
        retry_delay_s=0.25,
    )

    assert engine is expected_engine
    assert port == 41002
    assert [attempt[2] for attempt in attempts] == ["41001", "41002"]
    assert cleanup_calls == [True]
    assert sleeps == [0.25]

    try:
        worker.create_engine_with_rendezvous_retry(
            Path("/model"),
            engine_config={"tensor_parallel_size": 4},
            port_factory=lambda: 42001,
            engine_factory=lambda *_args, **_kwargs: (
                (_ for _ in ()).throw(RuntimeError("different failure"))
            ),
            environment={},
            cleanup_failed_attempt=lambda: cleanup_calls.append(False),
            sleep=lambda _seconds: None,
        )
    except RuntimeError as exc:
        assert str(exc) == "different failure"
    else:
        raise AssertionError("non-rendezvous failure was retried")
    assert cleanup_calls == [True]


def test_run_arm_routes_engine_creation_through_rendezvous_retry():
    clock = _Clock()
    engine = _FakeEngine("eager", output_tokens=128, clock=clock)
    retry_calls = []
    original_retry = worker.create_engine_with_rendezvous_retry

    def retry(model_root, *, engine_config, port_factory, engine_factory):
        retry_calls.append({
            "model_root": model_root,
            "engine_config": dict(engine_config),
            "port": int(port_factory()),
        })
        return engine_factory(model_root, **engine_config), 41001

    case = {
        "case_id": "Q0__r0__eager",
        "pair_id": "Q0__r0",
        "workload": "Q0",
        "repetition": 0,
        "arm": "eager",
        "order_index": 0,
        "profile": {
            "prompt_tokens": 256,
            "output_tokens": 128,
            "concurrency": 4,
        },
    }
    worker.create_engine_with_rendezvous_retry = retry
    try:
        with tempfile.TemporaryDirectory() as directory:
            worker.run_arm(
                model_root=Path("/model"),
                case=case,
                output_dir=Path(directory),
                timeout_s=5.0,
                engine_factory=lambda *_args, **_kwargs: engine,
                sampling_params_factory=lambda **kwargs: dict(kwargs),
                clock_ns=clock,
                reset_sequence_ids=lambda: None,
            )
    finally:
        worker.create_engine_with_rendezvous_retry = original_retry

    assert len(retry_calls) == 1
    assert retry_calls[0]["model_root"] == Path("/model")
    assert retry_calls[0]["engine_config"]["tensor_parallel_size"] == 4


def test_collect_rank_graph_observations_preserves_all_ranks():
    engine = _ObservationEngine(
        [_event(arm="graph", rank=rank) for rank in range(4)]
    )
    rows = worker.collect_rank_graph_observations(
        engine,
        case_id="Q0__r0__graph",
        phase="measured",
        step_index=7,
        timeout_s=5.0,
    )
    assert [row["rank"] for row in rows] == [0, 1, 2, 3]
    assert all(row["case_id"] == "Q0__r0__graph" for row in rows)
    assert all(row["phase"] == "measured" for row in rows)
    assert all(row["step_index"] == 7 for row in rows)
    assert all(row["graph_replay_count"] == 8 for row in rows)


def test_collect_rank_graph_observations_rejects_rank_disagreement():
    rows = [_event(arm="graph", rank=rank) for rank in range(4)]
    rows[3]["graph_identity_sha256"] = "c" * 64
    engine = _ObservationEngine(rows)
    try:
        worker.collect_rank_graph_observations(
            engine,
            case_id="Q0__r0__graph",
            phase="measured",
            step_index=7,
            timeout_s=5.0,
        )
    except RuntimeError as exc:
        assert "disagree" in str(exc)
    else:
        raise AssertionError("rank disagreement was accepted")


def test_run_arm_emits_complete_measured_evidence_and_cleanup():
    clock = _Clock()
    engines = []

    class CaptureCostEngine(_FakeEngine):
        def call_model_runner_acknowledged(
            self,
            method_name,
            *,
            timeout_s,
        ):
            local, acknowledgements = super().call_model_runner_acknowledged(
                method_name,
                timeout_s=timeout_s,
            )
            rows = [local] + [ack.result for ack in acknowledgements]
            if rows[0]["mode"] == "decode":
                duration_ns = (
                    20_000_000
                    if self.reset_profile_calls
                    else 10_000_000
                )
                for row in rows:
                    row.update({
                        "capture_attempted": True,
                        "capture_duration_ns": duration_ns,
                        "capture_static_bytes": 1_000_000,
                        "capture_allocated_delta_bytes": 2_000_000,
                        "capture_reserved_delta_bytes": 3_000_000,
                    })
            return local, acknowledgements

    def engine_factory(model_root, **config):
        del model_root
        arm = "eager" if config["enforce_eager"] else "graph"
        engine = CaptureCostEngine(
            arm,
            output_tokens=128,
            clock=clock,
        )
        engines.append(engine)
        return engine

    def sampling_factory(**kwargs):
        return dict(kwargs)

    case = {
        "case_id": "Q0__r0__graph",
        "pair_id": "Q0__r0",
        "workload": "Q0",
        "repetition": 0,
        "arm": "graph",
        "order_index": 1,
        "profile": {
            "prompt_tokens": 256,
            "output_tokens": 128,
            "concurrency": 4,
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        result = worker.run_arm(
            model_root=Path("/model"),
            case=case,
            output_dir=Path(directory),
            timeout_s=5.0,
            engine_factory=engine_factory,
            sampling_params_factory=sampling_factory,
            clock_ns=clock,
            reset_sequence_ids=lambda: None,
        )
    assert result["case_id"] == case["case_id"]
    assert len(result["request_rows"]) == 4
    assert len(result["rank_lifecycle_rows"]) == 4
    assert len(result["memory_rows"]) == 4
    assert len(result["rank_collective_rows"]) == 4
    measured_dispatch = [
        row
        for row in result["rank_dispatch_rows"]
        if row["phase"] == "measured" and row["mode"] == "decode"
    ]
    assert len(measured_dispatch) == 4
    assert all(row["dispatch"] == "graph" for row in measured_dispatch)
    assert result["cleanup"]["rank_exit_codes"] == [0, 0, 0, 0]
    assert {
        row["capture_duration_ns"]
        for row in result["capture_cost_rows"]
    } == {20_000_000}
    assert engines[0].clear_prefix_calls == 1
    assert engines[0].reset_profile_calls == 1
    assert engines[0].exit_calls == 1


def test_run_arm_cleans_up_after_execution_failure():
    clock = _Clock()
    engine = _FakeEngine("graph", output_tokens=128, clock=clock)

    def broken_step():
        raise RuntimeError("synthetic step failure")

    engine.step = broken_step
    case = {
        "case_id": "Q0__r0__graph",
        "pair_id": "Q0__r0",
        "workload": "Q0",
        "repetition": 0,
        "arm": "graph",
        "order_index": 1,
        "profile": {
            "prompt_tokens": 256,
            "output_tokens": 128,
            "concurrency": 4,
        },
    }
    try:
        worker.run_arm(
            model_root=Path("/model"),
            case=case,
            output_dir=Path("/unused"),
            timeout_s=5.0,
            engine_factory=lambda *_args, **_kwargs: engine,
            sampling_params_factory=lambda **kwargs: kwargs,
            clock_ns=clock,
            reset_sequence_ids=lambda: None,
        )
    except RuntimeError as exc:
        assert "synthetic step failure" in str(exc)
    else:
        raise AssertionError("execution failure was swallowed")
    assert engine.exit_calls == 1


def test_run_pair_retains_mismatch_as_correctness_evidence():
    clock = _Clock()

    def engine_factory(model_root, **config):
        del model_root
        arm = "eager" if config["enforce_eager"] else "graph"
        return _FakeEngine(
            arm,
            output_tokens=128,
            clock=clock,
            token_id=17 if arm == "eager" else 19,
        )

    pair_cases = tuple(
        row
        for row in worker.contract.build_case_matrix()
        if row["pair_id"] == "Q0__r0"
    )
    with tempfile.TemporaryDirectory() as directory:
        result = worker.run_pair(
            model_root=Path("/model"),
            pair_cases=pair_cases,
            output_dir=Path(directory),
            timeout_s=5.0,
            engine_factory=engine_factory,
            sampling_params_factory=lambda **kwargs: kwargs,
            clock_ns=clock,
            reset_sequence_ids=lambda: None,
        )
    assert result["correctness_row"]["exact_match"] is False
    assert (
        result["correctness_row"]["eager_outputs"]
        != result["correctness_row"]["graph_outputs"]
    )


def test_capture_cost_rows_keep_case_identity():
    case = next(
        row
        for row in worker.contract.build_case_matrix()
        if row["case_id"] == "Q0__r0__graph"
    )
    dispatch_rows = []
    for rank in range(4):
        row = _event(arm="graph", rank=rank)
        row.update({
            "rank": rank,
            "capture_duration_ns": 50_000_000,
            "capture_static_bytes": 1_000_000,
            "capture_allocated_delta_bytes": 2_000_000,
            "capture_reserved_delta_bytes": 3_000_000,
        })
        dispatch_rows.append(row)
    rows = worker._capture_cost_rows(dispatch_rows, case)
    assert len(rows) == 4
    assert all(row["case_id"] == case["case_id"] for row in rows)
    assert all(row["pair_id"] == case["pair_id"] for row in rows)
    assert all(row["repetition"] == 0 for row in rows)
    assert all(row["arm"] == "graph" for row in rows)


def main() -> None:
    tests = (
        test_engine_config_differs_only_by_graph_policy,
        test_shared_capacity_engine_config_uses_workload_bounded_kv_capacity,
        test_engine_config_rejects_unknown_admission_mode,
        test_engine_creation_retries_only_rendezvous_port_collisions,
        test_run_arm_routes_engine_creation_through_rendezvous_retry,
        test_collect_rank_graph_observations_preserves_all_ranks,
        test_collect_rank_graph_observations_rejects_rank_disagreement,
        test_run_arm_emits_complete_measured_evidence_and_cleanup,
        test_run_arm_cleans_up_after_execution_failure,
        test_run_pair_retains_mismatch_as_correctness_evidence,
        test_capture_cost_rows_keep_case_identity,
    )
    for test in tests:
        test()
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
