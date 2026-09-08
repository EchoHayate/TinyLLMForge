from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "tools/qwen38_topology_local_tp2_whole_model_worker.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "qwen38_topology_local_tp2_whole_model_worker_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_freezes_workloads_epochs_and_checkpoints():
    worker = _load()

    assert worker.WORKLOADS == {
        "P0": ("causal", 256, 128, 1),
        "P1": ("causal", 2048, 128, 1),
        "Q0": ("online", 256, 128, 4),
        "Q1": ("online", 256, 128, 8),
        "Q2": ("online", 2048, 128, 4),
    }
    assert worker.EPOCH_ARMS == (
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    )
    assert worker.MEASURED_REPETITIONS == 5
    assert worker.WARMUP_REPETITIONS == 2
    assert worker.STATE_CHECKPOINTS == (
        "pre_migration",
        "post_migration",
        "token_1",
        "token_4",
        "token_8",
        "token_32",
        "token_128",
    )


def test_request_specs_are_deterministic_exact_and_namespace_disjoint():
    worker = _load()

    first = worker.build_request_specs(256, 128, 4, "timing/P0/r0")
    repeated = worker.build_request_specs(
        256,
        128,
        4,
        "timing/P0/r0",
    )
    correctness = worker.build_request_specs(
        256,
        128,
        4,
        "correctness/P0/r0",
    )

    assert first == repeated
    assert len(first) == 4
    assert all(len(row["prompt_token_ids"]) == 256 for row in first)
    assert all(row["output_tokens"] == 128 for row in first)
    assert len({row["request_id"] for row in first}) == 4
    assert {
        tuple(row["prompt_token_ids"]) for row in first
    }.isdisjoint({
        tuple(row["prompt_token_ids"]) for row in correctness
    })


def test_request_metrics_use_first_to_last_tpot_and_all_token_gaps():
    worker = _load()
    timestamps = tuple(index * index + 100 for index in range(128))

    metrics = worker.reconstruct_request_metrics(
        admitted_ns=10,
        token_timestamps_ns=timestamps,
    )

    assert metrics["ttft_ns"] == 90
    assert metrics["tpot_ns"] == (
        timestamps[-1] - timestamps[0]
    ) / 127
    assert metrics["token_gaps_ns"] == [
        current - previous
        for previous, current in zip(
            timestamps,
            timestamps[1:],
        )
    ]
    assert metrics["e2e_ns"] == timestamps[-1] - 10


def _model_manifest():
    return {
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "pair_local_accumulation_dtype_bytes": 4,
        "full_attention_collective_dtype_bytes": 2,
        "layer_types": [
            (
                "full_attention"
                if index % 4 == 3
                else "linear_attention"
            )
            for index in range(64)
        ],
    }


def _raw_candidate_evidence(worker, *, concurrency=2):
    request_rows = [
        {
            "request_id": f"request-{index}",
            "output_token_ids": list(range(128)),
            "rank_token_agreement": True,
            "finite_logits": True,
            "stop_position": 128,
            "stop_reason": "length",
        }
        for index in range(concurrency)
    ]
    scheduler_step_rows = [{
        "step_index": 0,
        "is_prefill": True,
        "batch_kind": "prefill",
        "request_ids": [
            row["request_id"] for row in request_rows
        ],
    }]
    token_count_rows = []
    collective_rows = []
    for step_index in range(1, 128):
        scheduler_step_rows.append({
            "step_index": step_index,
            "is_prefill": False,
            "batch_kind": "decode",
            "request_ids": [
                row["request_id"] for row in request_rows
            ],
        })
        for request in request_rows:
            token_count_rows.append({
                "step_index": step_index,
                "request_id": request["request_id"],
                "token_count": 1,
            })
        for rank in range(4):
            for layer_index in range(64):
                layer_role = (
                    "full_attention"
                    if layer_index % 4 == 3
                    else "linear_attention"
                )
                scope = (
                    "global_tp4"
                    if layer_role == "full_attention"
                    else "pair_local"
                )
                dtype_bytes = (
                    2 if layer_role == "full_attention" else 4
                )
                collective_rows.append({
                    "rank": rank,
                    "step_index": step_index,
                    "layer_index": layer_index,
                    "layer_role": layer_role,
                    "scope": scope,
                    "calls": concurrency,
                    "bytes": (
                        concurrency
                        * 5120
                        * dtype_bytes
                    ),
                    "sequence": [
                        f"{step_index}:{layer_index}:{request_id}"
                        for request_id in (
                            row["request_id"]
                            for row in request_rows
                        )
                    ],
                })

    expected_segments = concurrency * 127
    before = []
    after = []
    linear_indices = tuple(
        index for index in range(64) if index % 4 != 3
    )
    for rank in range(4):
        pair_id = 0 if rank < 2 else 1
        before.append({
            "rank": rank,
            "enabled": True,
            "pair_id": pair_id,
            "linear_layer_indices": linear_indices,
            "transition_count": 0,
            "fixed_cohort": None,
            "state": {
                "publication_count": 0,
                "temporary_live_tensors": 0,
                "rollback_count": 0,
            },
            "mixers": tuple({
                "tp2_decode_calls": 0,
                "recurrent_token_one_calls": 0,
                "short_chunk_calls": 0,
                "chunk_64_calls": 0,
                "pair_local_all_reduce_calls": 0,
                "global_tp4_decode_all_reduce_calls": 0,
            } for _ in linear_indices),
            "fallback_calls": 0,
            "post_warmup_request_path_allocations": 0,
            "prefix_restore_calls": 0,
            "prefix_publication_calls": 0,
            "retry_after_mutation_calls": 0,
            "duplicate_commit_calls": 0,
            "pair_replica_comparison_failures": 0,
        })
        after.append({
            **before[-1],
            "transition_count": 1,
            "fixed_cohort": tuple(
                (index, 1, index)
                for index in range(concurrency)
            ),
            "state": {
                "publication_count": concurrency,
                "temporary_live_tensors": 0,
                "rollback_count": 0,
            },
            "mixers": tuple({
                "tp2_decode_calls": expected_segments,
                "recurrent_token_one_calls": expected_segments,
                "short_chunk_calls": 0,
                "chunk_64_calls": 0,
                "pair_local_all_reduce_calls": expected_segments,
                "global_tp4_decode_all_reduce_calls": 0,
            } for _ in linear_indices),
        })
    cleanup = {
        "process_group_destroyed": True,
        "rank_exit_codes": [0, 0, 0, 0],
        "owned_children_remaining": [],
        "rank_cleanup_receipts": [{
            "rank": rank,
            "process_group_destroyed": True,
            "qwen38_topology_local_tp2_cleanup": {
                "pair_groups_destroyed": 2,
                "candidate_state_released": True,
                "published_generations_remaining": 0,
                "temporary_live_tensors": 0,
            },
        } for rank in range(4)],
    }
    return {
        "request_rows": request_rows,
        "scheduler_step_rows": scheduler_step_rows,
        "token_count_rows": token_count_rows,
        "collective_rows": collective_rows,
        "model_manifest": _model_manifest(),
        "before_snapshots": tuple(before),
        "after_snapshots": tuple(after),
        "cleanup": cleanup,
    }


def test_candidate_evidence_is_rederived_from_raw_rows():
    worker = _load()
    raw = _raw_candidate_evidence(worker)

    result = worker.validate_candidate_evidence(**raw)

    expected_segments = 2 * 127
    assert result["expected_segments"] == expected_segments
    assert result["tp2_decode_calls"] == expected_segments * 48
    assert (
        result["recurrent_token_one_calls"]
        == expected_segments * 48
    )
    assert result["short_chunk_calls"] == 0
    assert result["ordinary_chunk_calls"] == 0
    assert (
        result["global_tp4_linear_decode_all_reduce_calls"]
        == 0
    )
    assert result["fallback_calls"] == 0
    assert result["migration_publications"] == 2
    assert result["full_attention_layer_count"] == 16
    assert (
        result["full_attention_tp4_collective_calls"]
        == expected_segments * 16
    )
    assert result["pair_local_collective_sequence_match"] is True
    assert result["post_warmup_request_path_allocations"] == 0
    assert result["prefix_restore_calls"] == 0
    assert result["prefix_publication_calls"] == 0
    assert result["retry_after_mutation_calls"] == 0
    assert result["duplicate_commit_calls"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("token_mismatch", "output token"),
        ("changed_cohort", "cohort"),
        ("missing_rank", "rank"),
        ("duplicate_commit", "duplicate commit"),
        ("nonfinite_logits", "finite"),
        ("missing_layer", "layer"),
        ("retained_temporary", "temporary"),
        ("incomplete_cleanup", "cleanup"),
    ),
)
def test_candidate_evidence_rejects_incomplete_or_invalid_proof(
    mutation,
    message,
):
    worker = _load()
    raw = _raw_candidate_evidence(worker)
    if mutation == "token_mismatch":
        raw["request_rows"][0]["output_token_ids"] = list(range(127))
    elif mutation == "changed_cohort":
        raw["after_snapshots"][0]["fixed_cohort"] = ((9, 1, 9),)
    elif mutation == "missing_rank":
        raw["after_snapshots"] = raw["after_snapshots"][:-1]
    elif mutation == "duplicate_commit":
        raw["after_snapshots"][0]["duplicate_commit_calls"] = 1
    elif mutation == "nonfinite_logits":
        raw["request_rows"][0]["finite_logits"] = False
    elif mutation == "missing_layer":
        raw["model_manifest"]["layer_types"].pop()
    elif mutation == "retained_temporary":
        raw["after_snapshots"][0]["state"][
            "temporary_live_tensors"
        ] = 1
    elif mutation == "incomplete_cleanup":
        raw["cleanup"]["rank_cleanup_receipts"].pop()

    with pytest.raises((ValueError, RuntimeError), match=message):
        worker.validate_candidate_evidence(**raw)


class _FakeEngine:

    def __init__(self, request_specs, *, candidate, seq_offset=0):
        self.request_specs = request_specs
        self.candidate = candidate
        self.seq_offset = seq_offset
        self.model_runner = SimpleNamespace(rank=0, world_size=4)
        self.last_step_observation = None
        self._step = 0
        self._finished = False
        self._admitted = []
        self.exit_calls = 0
        self.flush_calls = 0

    def add_request(self, prompt, sampling):
        self._admitted.append((list(prompt), sampling))
        return self.seq_offset + len(self._admitted) - 1

    def flush_pending_hybrid_state_releases(self, *, timeout_s):
        self.flush_calls += 1
        return ()

    def is_finished(self):
        return self._finished

    def step(self):
        tokens = {
            self.seq_offset + index: [1000 + self._step + index]
            for index in range(len(self.request_specs))
        }
        self.last_step_observation = {
            "step_end_ns": 1_000_000 + self._step * 100,
            "is_prefill": self._step == 0,
            "batch_kind": (
                "prefill" if self._step == 0 else "decode"
            ),
            "scheduled": [
                {"seq_id": self.seq_offset + index}
                for index in range(len(self.request_specs))
            ],
            "new_completion_tokens_by_seq": tokens,
            "memory": {},
        }
        self._step += 1
        if self._step == 128:
            self._finished = True
            return [
                (
                    self.seq_offset + index,
                    [
                        1000 + step + index
                        for step in range(128)
                    ],
                )
                for index in range(len(self.request_specs))
            ], 0
        return [], 0

    def qwen38_topology_local_tp2_snapshots(self, timeout_s):
        raw = _raw_candidate_evidence(
            SimpleNamespace(),
            concurrency=len(self.request_specs),
        )
        return (
            raw["after_snapshots"]
            if self._step
            else raw["before_snapshots"]
        )

    def memory_snapshots(self, *, timeout_s):
        return tuple({
            "rank": rank,
            "cuda_peak_allocated_bytes": 100 + rank,
            "cuda_peak_reserved_bytes": 200 + rank,
            "physical_memory_bytes": 80 * 1024**3,
        } for rank in range(4))

    def exit(self):
        self.exit_calls += 1
        return _raw_candidate_evidence(
            SimpleNamespace(),
            concurrency=len(self.request_specs),
        )["cleanup"]


def test_run_engine_case_uses_frozen_engine_shape_and_exact_timing():
    worker = _load()
    specs = worker.build_request_specs(
        256,
        128,
        2,
        "timing/P0/r0",
    )
    factory_calls = []
    fake = _FakeEngine(specs, candidate=True)

    def factory(model_root, **kwargs):
        factory_calls.append((model_root, kwargs))
        return fake

    ticks = iter(range(10, 1000))
    result = worker.run_engine_case(
        model_root=Path("/model"),
        arm="candidate",
        workload_id="P0",
        request_specs=specs,
        warmup=False,
        epoch=1,
        repetition=0,
        engine_factory=factory,
        sampling_params_factory=lambda **kwargs: kwargs,
        clock_ns=lambda: next(ticks),
        timeout_s=30.0,
    )

    assert factory_calls == [(
        Path("/model"),
        {
            "tensor_parallel_size": 4,
            "enforce_eager": True,
            "max_num_seqs": 8,
            "max_model_len": 384,
            "max_num_batched_tokens": 512,
            "qwen38_topology_local_tp2_islands": True,
        },
    )]
    assert len(result["requests"]) == 2
    assert all(
        len(row["output_token_ids"]) == 128
        for row in result["requests"]
    )
    assert all(
        len(row["token_gaps_ns"]) == 127
        for row in result["requests"]
    )
    assert len(result["request_set_digest"]) == 64
    assert result["cohort_makespan_ns"] >= 0
    assert all(
        row["rank_token_agreement"] is True
        and row["finite_logits"] is True
        and row["stop_position"] == 128
        and row["stop_reason"] == "length"
        for row in result["requests"]
    )
    assert result["cleanup"]["process_group_destroyed"] is True
    assert fake.exit_calls == 1


def test_shared_engine_case_flushes_and_uses_returned_sequence_ids():
    worker = _load()
    specs = worker.build_request_specs(
        256,
        128,
        2,
        "timing/P0/r1",
    )
    fake = _FakeEngine(specs, candidate=False, seq_offset=40)

    result = worker.run_engine_case(
        model_root=Path("/model"),
        arm="baseline",
        workload_id="P0",
        request_specs=specs,
        warmup=True,
        epoch=0,
        repetition=1,
        engine=fake,
        close_engine=False,
        sampling_params_factory=lambda **kwargs: kwargs,
        clock_ns=iter(range(10, 1000)).__next__,
        timeout_s=30.0,
    )

    assert fake.flush_calls == 1
    assert fake.exit_calls == 0
    assert result["cleanup"] is None
    assert [row["request_id"] for row in result["requests"]] == [
        row["request_id"] for row in specs
    ]


def test_service_replica_case_uses_a_real_tp2_engine_shape():
    worker = _load()
    specs = worker.build_request_specs(
        256,
        128,
        2,
        "service/Q0/r0",
    )
    fake = _FakeEngine(specs, candidate=False)
    fake.model_runner.world_size = 2
    factory_calls = []

    result = worker.run_service_replica_case(
        model_root=Path("/model"),
        workload_id="Q0",
        request_specs=specs,
        engine_factory=lambda model_root, **kwargs: (
            factory_calls.append((model_root, kwargs)) or fake
        ),
        sampling_params_factory=lambda **kwargs: kwargs,
        clock_ns=iter(range(10, 1000)).__next__,
    )

    assert factory_calls[0][1]["tensor_parallel_size"] == 2
    assert factory_calls[0][1]["qwen38_topology_local_tp2_islands"] is False
    assert result["replica_tensor_parallel_size"] == 2
    assert len(result["requests"]) == 2


def test_performance_epoch_uses_two_warmups_and_five_measurements():
    worker = _load()
    calls = []

    def case_runner(**kwargs):
        calls.append(kwargs)
        return {
            "workload_id": kwargs["workload_id"],
            "warmup": kwargs["warmup"],
            "repetition": kwargs["repetition"],
        }

    result = worker.run_performance_epoch(
        model_root=Path("/model"),
        output_root=Path("/output"),
        epoch=1,
        arm="candidate",
        workload_order=("P0", "Q0"),
        case_runner=case_runner,
        row_sink=lambda _row: None,
    )

    assert len(calls) == 2 * (2 + 5)
    assert [row["workload_id"] for row in calls[:7]] == ["P0"] * 7
    assert sum(not row["warmup"] for row in calls) == 10
    assert result["epoch"] == 1
    assert result["arm"] == "candidate"


def test_performance_epoch_reuses_one_engine_when_factory_is_supplied():
    worker = _load()
    engine = SimpleNamespace(exit=lambda: {"clean": True})
    factory_calls = []
    case_engines = []

    def factory(model_root, **kwargs):
        factory_calls.append((model_root, kwargs))
        return engine

    def case_runner(**kwargs):
        case_engines.append(kwargs["engine"])
        assert kwargs["close_engine"] is False
        return {
            "workload_id": kwargs["workload_id"],
            "warmup": kwargs["warmup"],
            "repetition": kwargs["repetition"],
        }

    result = worker.run_performance_epoch(
        model_root=Path("/model"),
        output_root=Path("/output"),
        epoch=2,
        arm="candidate",
        workload_order=("P0",),
        case_runner=case_runner,
        engine_factory=factory,
        row_sink=lambda _row: None,
    )

    assert len(factory_calls) == 1
    assert factory_calls[0][1] == {
        "tensor_parallel_size": 4,
        "enforce_eager": True,
        "max_num_seqs": 8,
        "max_model_len": 2176,
        "max_num_batched_tokens": 8192,
        "qwen38_topology_local_tp2_islands": True,
    }
    assert case_engines == [engine] * 7
    assert result["cleanup"] == {"clean": True}


def test_worker_cli_dispatches_one_frozen_performance_epoch(tmp_path):
    worker = _load()
    calls = []

    def performance_runner(**kwargs):
        calls.append(kwargs)
        return {
            "schema_version": worker.WORKER_SCHEMA,
            "phase": "performance_epoch",
            "epoch": kwargs["epoch"],
            "arm": kwargs["arm"],
        }

    exit_code = worker.main(
        [
            "performance-epoch",
            "--model-root",
            "/model",
            "--output-root",
            str(tmp_path),
            "--epoch",
            "2",
            "--arm",
            "candidate",
            "--workload-order",
            "P0,P1,Q0,Q1,Q2",
            "--source-revision",
            "a" * 40,
            "--model-revision",
            "b" * 40,
        ],
        performance_runner=performance_runner,
    )

    assert exit_code == 0
    assert calls == [{
        "model_root": Path("/model"),
        "output_root": tmp_path,
        "epoch": 2,
        "arm": "candidate",
        "workload_order": ("P0", "P1", "Q0", "Q1", "Q2"),
        "engine_factory": worker._default_engine_factory,
    }]
    payload = json.loads(
        (tmp_path / "worker-receipt.json").read_text(encoding="utf-8")
    )
    assert payload["phase"] == "performance_epoch"
    assert payload["epoch"] == 2
    artifacts = json.loads(
        (tmp_path / "epoch-2-artifact-rows.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(artifacts) == {
        "request_rows.jsonl",
        "scheduler_step_rows.jsonl",
        "candidate_hit_rows.jsonl",
        "collective_rows.jsonl",
        "migration_rows.jsonl",
        "memory_rows.jsonl",
    }


def test_worker_cli_dispatches_correctness_and_service_control(tmp_path):
    worker = _load()
    calls = []

    def correctness_runner(**kwargs):
        calls.append(("correctness", kwargs))
        return {
            "schema_version": worker.WORKER_SCHEMA,
            "phase": "correctness",
        }

    correctness_root = tmp_path / "correctness"
    assert worker.main(
        [
            "correctness",
            "--model-root",
            "/model",
            "--output-root",
            str(correctness_root),
        ],
        correctness_runner=correctness_runner,
    ) == 0

    def service_runner(**kwargs):
        calls.append(("service", kwargs))
        return {
            "schema_version": worker.WORKER_SCHEMA,
            "arm": "TP2_X2_SERVICE_CONTROL",
        }

    service_root = tmp_path / "service"
    assert worker.main(
        [
            "service-control",
            "--model-root",
            "/model",
            "--output-root",
            str(service_root),
            "--pair-devices",
            "0,1;2,3",
            "--workloads",
            "Q0,Q1,Q2",
        ],
        service_runner=service_runner,
        replica_runner=lambda **_kwargs: {},
    ) == 0

    assert calls[0] == (
        "correctness",
        {
            "model_root": Path("/model"),
            "output_root": correctness_root,
            "engine_factory": worker._default_engine_factory,
        },
    )
    assert calls[1][0] == "service"
    assert calls[1][1]["model_root"] == Path("/model")
    assert calls[1][1]["output_root"] == service_root
    assert calls[1][1]["pair_devices"] == ((0, 1), (2, 3))
    assert calls[1][1]["workloads"] == ("Q0", "Q1", "Q2")
    assert callable(calls[1][1]["replica_runner"])


def test_correctness_campaign_loads_one_engine_per_arm():
    worker = _load()
    engines = {
        "baseline": SimpleNamespace(exit=lambda: {"arm": "baseline"}),
        "candidate": SimpleNamespace(exit=lambda: {"arm": "candidate"}),
    }
    factory_calls = []
    case_calls = []

    def factory(_model_root, **kwargs):
        arm = (
            "candidate"
            if kwargs["qwen38_topology_local_tp2_islands"]
            else "baseline"
        )
        factory_calls.append(arm)
        return engines[arm]

    def case_runner(**kwargs):
        case_calls.append(kwargs)
        return {
            "requests": [{
                "output_token_ids": list(range(128)),
            } for _ in kwargs["request_specs"]],
            "timing_authority": False,
        }

    result = worker.run_correctness_campaign(
        model_root=Path("/model"),
        output_root=Path("/output"),
        engine_factory=factory,
        case_runner=case_runner,
        row_sink=lambda _row: None,
    )

    assert factory_calls == ["baseline", "candidate"]
    assert len(case_calls) == 50
    assert all(call["close_engine"] is False for call in case_calls)
    assert result["cleanup"] == {
        "baseline": {"arm": "baseline"},
        "candidate": {"arm": "candidate"},
    }


def test_service_control_splits_requests_and_is_non_authoritative():
    worker = _load()
    calls = []

    def replica_runner(*, pair_devices, request_specs, **kwargs):
        calls.append((pair_devices, request_specs))
        return {
            "requests": [{
                "request_id": row["request_id"],
                "output_token_ids": list(range(128)),
                "admitted_ns": 100,
                "completion_ns": 200,
            } for row in request_specs],
            "memory": [],
        }

    result = worker.run_service_control(
        model_root=Path("/model"),
        output_root=Path("/output"),
        pair_devices=((0, 1), (2, 3)),
        workloads=("Q0",),
        replica_runner=replica_runner,
        row_sink=lambda _row: None,
    )

    assert len(calls) == 2
    assert {
        int(row["request_id"].rsplit("-", 1)[-1]) % 2
        for row in calls[0][1]
    } == {0}
    assert {
        int(row["request_id"].rsplit("-", 1)[-1]) % 2
        for row in calls[1][1]
    } == {1}
    assert result["arm"] == "TP2_X2_SERVICE_CONTROL"
    assert result["classification_authority"] is False


def test_service_control_can_launch_both_replicas_as_one_parallel_batch():
    worker = _load()
    calls = []

    def parallel_runner(
        *,
        pair_devices,
        request_specs_by_replica,
        **kwargs,
    ):
        calls.append((pair_devices, request_specs_by_replica, kwargs))
        replicas = []
        for replica_specs in request_specs_by_replica:
            replicas.append({
                "requests": [{
                    "request_id": row["request_id"],
                    "output_token_ids": list(range(128)),
                    "admitted_ns": 10,
                    "completion_ns": 110,
                } for row in replica_specs],
            })
        return tuple(replicas)

    result = worker.run_service_control(
        model_root=Path("/model"),
        output_root=Path("/output"),
        pair_devices=((0, 1), (2, 3)),
        workloads=("Q0",),
        parallel_runner=parallel_runner,
        row_sink=lambda _row: None,
    )

    assert len(calls) == 1
    assert calls[0][0] == ((0, 1), (2, 3))
    assert [len(rows) for rows in calls[0][1]] == [2, 2]
    assert result["rows"][0]["request_qps"] == 40_000_000.0


def test_parallel_service_runner_starts_both_children_before_waiting(
    tmp_path,
):
    worker = _load()
    events = []
    processes = []
    specs = (
        worker.build_request_specs(256, 128, 2, "service/Q0/a"),
        worker.build_request_specs(256, 128, 2, "service/Q0/b"),
    )

    class Process:
        returncode = 0

        def __init__(self, argv, env):
            self.argv = argv
            self.env = env
            processes.append(self)
            events.append(("start", env["CUDA_VISIBLE_DEVICES"]))
            output_path = Path(
                argv[argv.index("--output-path") + 1]
            )
            input_path = Path(
                argv[argv.index("--request-specs-path") + 1]
            )
            requests = json.loads(input_path.read_text(encoding="utf-8"))
            output_path.write_text(json.dumps({
                "requests": [{
                    "request_id": row["request_id"],
                    "output_token_ids": list(range(128)),
                    "admitted_ns": 1,
                    "completion_ns": 2,
                } for row in requests],
            }))

        def communicate(self, timeout):
            events.append(("wait", self.env["CUDA_VISIBLE_DEVICES"], timeout))
            return "", ""

    result = worker._run_service_replicas_in_subprocesses(
        model_root=Path("/model"),
        output_root=tmp_path,
        pair_devices=((3, 4), (6, 7)),
        workload_id="Q0",
        request_specs_by_replica=specs,
        shared_start_ns=0,
        popen_factory=lambda argv, **kwargs: Process(argv, kwargs["env"]),
    )

    assert [event[0] for event in events] == [
        "start",
        "start",
        "wait",
        "wait",
    ]
    assert [process.env["CUDA_VISIBLE_DEVICES"] for process in processes] == [
        "3,4",
        "6,7",
    ]
    assert len(result) == 2


def test_performance_artifacts_are_rebuilt_from_measured_raw_case():
    worker = _load()
    specs = worker.build_request_specs(256, 128, 2, "timing/P0/r0")
    fake = _FakeEngine(specs, candidate=True)
    case = worker.run_engine_case(
        model_root=Path("/model"),
        arm="candidate",
        workload_id="P0",
        request_specs=specs,
        warmup=False,
        epoch=1,
        repetition=0,
        engine_factory=lambda *_args, **_kwargs: fake,
        sampling_params_factory=lambda **kwargs: kwargs,
        clock_ns=iter(range(10, 1000)).__next__,
    )
    for snapshot in case["after_snapshots"]:
        snapshot["last_transition_latency_ns"] = 1234
    artifacts = worker.build_performance_artifact_rows(
        {
            "epoch": 1,
            "arm": "candidate",
            "rows": [case],
            "cleanup": case["cleanup"],
        },
        source_revision="a" * 40,
        model_revision="b" * 40,
    )

    assert len(artifacts["request_rows.jsonl"]) == 1
    request = artifacts["request_rows.jsonl"][0]
    assert request["request_set_digest"] == case["request_set_digest"]
    assert request["cohort_makespan_ns"] == case["cohort_makespan_ns"]
    scheduler = artifacts["scheduler_step_rows.jsonl"][0]
    assert scheduler["decode_steps"] == 127
    assert scheduler["token_one_segments"] == 2 * 127
    hit = artifacts["candidate_hit_rows.jsonl"][0]
    assert hit["tp2_decode_calls"] == 2 * 127 * 48
    assert hit["short_chunk_calls"] == 0
    collective = artifacts["collective_rows.jsonl"][0]
    assert collective["pair_local_calls"] == 2 * 127 * 48
    assert collective["pair_local_bytes"] == 2 * 127 * 48 * 5120 * 4
    assert artifacts["migration_rows.jsonl"][0]["latency_ns"] == 1234
    assert len(artifacts["memory_rows.jsonl"]) == 4
