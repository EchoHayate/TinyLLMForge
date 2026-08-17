from __future__ import annotations

import importlib.util
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = (
    ROOT / "tools" / "speculative_runtime_performance_gate.py"
)
WORKER_PATH = (
    ROOT / "tools" / "speculative_runtime_performance_worker.py"
)
VERIFY_PATH = (
    ROOT
    / "tools"
    / "verify_speculative_runtime_performance_gate.py"
)
REMOTE_SCRIPT_PATH = (
    ROOT
    / "tools"
    / "run_speculative_runtime_performance_gate_remote.sh"
)


def _load_module(name: str, path: Path):
    assert path.exists(), f"missing module: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _gate():
    return _load_module(
        "speculative_runtime_performance_gate_test_module",
        GATE_PATH,
    )


def _worker():
    _load_module(
        "speculative_runtime_performance_gate",
        GATE_PATH,
    )
    return _load_module(
        "speculative_runtime_performance_worker_test_module",
        WORKER_PATH,
    )


def _verifier():
    _load_module(
        "speculative_runtime_performance_gate",
        GATE_PATH,
    )
    return _load_module(
        "verify_speculative_runtime_performance_gate_test_module",
        VERIFY_PATH,
    )


class FakeTokenizer:
    def __init__(self, rows):
        self.rows = dict(rows)

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(self.rows[text])


def test_prompt_builder_returns_exact_4096_token_rows():
    tokenizer = FakeTokenizer(
        {
            "alpha": [11, 12, 13, 14],
            "beta": [21, 22, 23, 24],
            "gamma": [31, 32, 33, 34],
            "delta": [41, 42, 43, 44],
        }
    )

    rows = _gate().build_prompt_token_batches(
        tokenizer,
        batch_size=4,
        prompt_tokens=4096,
        seeds=("alpha", "beta", "gamma", "delta"),
    )

    assert len(rows) == 4
    assert all(len(row["token_ids"]) == 4096 for row in rows)
    assert len({row["sha256"] for row in rows}) == 4
    assert [row["prompt_index"] for row in rows] == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "batch_size,prompt_tokens,seeds,match",
    (
        (0, 4096, ("alpha",), "batch_size"),
        (1, 0, ("alpha",), "prompt_tokens"),
        (2, 4096, ("alpha",), "seed"),
    ),
)
def test_prompt_builder_rejects_invalid_shape(
    batch_size,
    prompt_tokens,
    seeds,
    match,
):
    tokenizer = FakeTokenizer({"alpha": [1]})

    with pytest.raises(ValueError, match=match):
        _gate().build_prompt_token_batches(
            tokenizer,
            batch_size=batch_size,
            prompt_tokens=prompt_tokens,
            seeds=seeds,
        )


def test_counter_delta_uses_only_monotonic_real_summaries():
    result = _gate().subtract_counter_summaries(
        {"h2d_copies": 2, "h2d_bytes": 1024},
        {"h2d_copies": 5, "h2d_bytes": 4096},
        keys=("h2d_copies", "h2d_bytes"),
    )

    assert result == {
        "h2d_copies": 3,
        "h2d_bytes": 3072,
    }


@pytest.mark.parametrize(
    "before,after,match",
    (
        (
            {"h2d_copies": 1},
            {},
            "h2d_copies",
        ),
        (
            {"h2d_copies": True},
            {"h2d_copies": 2},
            "integer",
        ),
        (
            {"h2d_copies": 1},
            {"h2d_copies": 0},
            "decreased",
        ),
    ),
)
def test_counter_delta_rejects_non_real_or_decreasing_values(
    before,
    after,
    match,
):
    with pytest.raises(ValueError, match=match):
        _gate().subtract_counter_summaries(
            before,
            after,
            keys=("h2d_copies",),
        )


def test_run_metrics_compute_synchronized_ttft_tpot_and_throughput():
    metrics = _gate().build_run_metrics(
        request_start_ns=1_000_000_000,
        request_finish_ns=3_000_000_000,
        token_events={
            0: [(1_500_000_000, 1), (3_000_000_000, 63)],
            1: [(2_000_000_000, 4), (3_000_000_000, 60)],
        },
        finished_at_ns={
            0: 3_000_000_000,
            1: 3_000_000_000,
        },
        expected_output_tokens=64,
    )

    assert metrics["per_request"][0]["ttft_s"] == 0.5
    assert metrics["per_request"][0]["tpot_s"] == pytest.approx(
        1.5 / 63
    )
    assert metrics["per_request"][1]["ttft_s"] == 1.0
    assert metrics["batch_token_throughput_tps"] == 64.0
    assert metrics["request_throughput_rps"] == 1.0
    assert metrics["batch_elapsed_s"] == 2.0


@pytest.mark.parametrize(
    "token_events,finished_at_ns,match",
    (
        ({0: []}, {0: 2}, "token event"),
        ({0: [(2, 63)]}, {0: 2}, "64"),
        ({0: [(2, 64)]}, {}, "finish"),
    ),
)
def test_run_metrics_rejects_incomplete_request_evidence(
    token_events,
    finished_at_ns,
    match,
):
    with pytest.raises(ValueError, match=match):
        _gate().build_run_metrics(
            request_start_ns=1,
            request_finish_ns=3,
            token_events=token_events,
            finished_at_ns=finished_at_ns,
            expected_output_tokens=64,
        )


def test_direction_requires_tpot_and_throughput_to_agree():
    gate = _gate()
    baseline = {
        "tpot_s": {"median": 0.020},
        "batch_token_throughput_tps": {"median": 50.0},
    }

    assert gate.classify_batch_direction(
        baseline,
        {
            "tpot_s": {"median": 0.015},
            "batch_token_throughput_tps": {"median": 60.0},
        },
    ) == "IMPROVED"
    assert gate.classify_batch_direction(
        baseline,
        {
            "tpot_s": {"median": 0.025},
            "batch_token_throughput_tps": {"median": 40.0},
        },
    ) == "REGRESSED"
    assert gate.classify_batch_direction(
        baseline,
        {
            "tpot_s": {"median": 0.015},
            "batch_token_throughput_tps": {"median": 40.0},
        },
    ) == "MIXED"


def test_aggregate_measurements_reports_raw_distribution():
    summary = _gate().aggregate_measurements(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )

    assert summary == {
        "count": 5,
        "median": 3.0,
        "min": 1.0,
        "max": 5.0,
        "pstdev": pytest.approx(2.0 ** 0.5),
    }


class FakeRequestEngine:
    def __init__(self):
        self.events = []
        self.active = False
        self.step_index = 0
        self.last_step_observation = None
        self.tokenizer = FakeTokenizer(
            {"alpha": [11, 12, 13, 14]}
        )

    def is_finished(self):
        return not self.active

    def clear_reusable_prefix_cache(self):
        self.events.append("clear")
        return ()

    def kv_offload_summaries(self, *, timeout_s):
        assert timeout_s == 60.0
        self.events.append("summary")
        if self.step_index == 0:
            return ({
                "h2d_copies": 2,
                "h2d_bytes": 1024,
                "d2h_copies": 1,
                "d2h_bytes": 512,
                "copy_waits": 0,
                "evictions": 0,
                "evict_clean": 0,
                "speculative_residency_committed_blocks": 0,
                "speculative_residency_rejected_blocks": 0,
                "speculative_residency_rejected_d2h_copies": 0,
            },)
        return ({
            "h2d_copies": 5,
            "h2d_bytes": 4096,
            "d2h_copies": 3,
            "d2h_bytes": 1536,
            "copy_waits": 2,
            "evictions": 1,
            "evict_clean": 1,
            "speculative_residency_committed_blocks": 1,
            "speculative_residency_rejected_blocks": 2,
            "speculative_residency_rejected_d2h_copies": 0,
        },)

    def reset_peak_memory_stats(self, *, timeout_s):
        assert timeout_s == 60.0
        self.events.append("reset")
        return ({
            "rank": 0,
            "cuda_allocated_bytes": 100,
            "cuda_reserved_bytes": 120,
            "cuda_peak_allocated_bytes": 100,
            "cuda_peak_reserved_bytes": 120,
            "kv_capacity_bytes": 64,
        },)

    def memory_snapshots(self, *, timeout_s):
        assert timeout_s == 60.0
        self.events.append("memory")
        return ({
            "rank": 0,
            "cuda_allocated_bytes": 105,
            "cuda_reserved_bytes": 130,
            "cuda_peak_allocated_bytes": 150,
            "cuda_peak_reserved_bytes": 180,
            "kv_capacity_bytes": 64,
        },)

    def add_request(self, prompt, sampling_params):
        assert len(prompt) == 4096
        self.events.append("add")
        self.active = True

    def step(self):
        self.events.append(f"step-{self.step_index}")
        if self.step_index == 0:
            outputs = []
            token_delta = [10]
            finished = []
        else:
            outputs = [(0, list(range(64)))]
            token_delta = list(range(1, 64))
            finished = [0]
            self.active = False
        self.last_step_observation = {
            "is_prefill": self.step_index == 0,
            "new_completion_tokens_by_seq": {
                0: token_delta,
            },
            "finished_seq_ids": finished,
            "speculative_selected_seq_ids": [0],
            "speculative_proposal_token_counts": {0: 3},
            "speculative_proposal_row_count": 1,
            "speculative_accepted_draft_token_counts": {0: 2},
            "speculative_first_target_callback_count": 1,
            "speculative_fixed_q_group_count": 1,
            "speculative_output_token_counts": {
                0: len(token_delta),
            },
            "speculative_runtime_timing_ms": {
                "target_forward_ms": 1.0,
            },
        }
        self.step_index += 1
        return outputs, len(token_delta)


def _prompt_row():
    return {
        "prompt_index": 0,
        "seed": "alpha",
        "token_ids": [11, 12, 13, 14] * 1024,
        "token_count": 4096,
        "sha256": "a" * 64,
    }


def test_worker_records_step_end_token_events_and_counter_deltas():
    engine = FakeRequestEngine()
    clock = iter(
        (1_000_000_000, 1_500_000_000, 3_000_000_000)
    )
    synchronize_calls = []
    evictions = []

    result = _worker().run_request_batch(
        engine=engine,
        prompt_rows=[_prompt_row()],
        sampling_params=object(),
        expected_output_tokens=64,
        synchronize=lambda: synchronize_calls.append("sync"),
        clock_ns=clock.__next__,
        evict_fn=lambda active_engine: (
            evictions.append(active_engine),
            ((3, 7),),
        )[1],
    )

    assert result["movement"]["ranks"][0]["h2d_copies"] == 3
    assert result["movement"]["ranks"][0]["h2d_bytes"] == 3072
    assert result["timing"]["per_request"][0]["ttft_s"] == 0.5
    assert result["outputs"][0] == list(range(64))
    assert result["memory"]["peak_allocated_bytes"] == 150
    assert result["memory"]["peak_allocated_delta_bytes"] == 50
    assert result["runtime"]["accepted_draft_tokens"] == 4
    assert result["evicted_block_identities"] == [[3, 7]]
    assert evictions == [engine]
    assert engine.events == [
        "clear",
        "summary",
        "reset",
        "add",
        "step-0",
        "step-1",
        "summary",
        "memory",
    ]
    assert synchronize_calls == ["sync", "sync", "sync"]


class FakeCampaignEngine:
    def __init__(self):
        self.tokenizer = FakeTokenizer(
            {
                seed: [index + 1, index + 11]
                for index, seed in enumerate(
                    _gate().DEFAULT_PROMPT_SEEDS
                )
            }
        )
        self.activate_calls = []
        self.exit_calls = 0

    def activate_speculative_runtime(self, runtime):
        self.activate_calls.append(runtime)

    def exit(self):
        self.exit_calls += 1


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


class FakeAdapter:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


class FakeRuntime:
    def __init__(self, adapter):
        self.adapter = adapter


def test_candidate_campaign_installs_real_adapter_once_and_runs_1_1_5():
    engine = FakeCampaignEngine()
    calls = []
    factory_kwargs = []

    def run_batch_fn(**kwargs):
        calls.append(kwargs)
        return {
            "outputs": [[1] * 64] * len(kwargs["prompt_rows"]),
        }

    result = _worker().run_policy_campaign(
        model_path="/model",
        policy="ngram",
        batch_size=4,
        engine_factory=lambda *args, **kwargs: (
            factory_kwargs.append(dict(kwargs)),
            engine,
        )[1],
        sampling_params_type=FakeSamplingParams,
        runtime_type=FakeRuntime,
        adapter_type=FakeAdapter,
        synchronize=lambda: None,
        clock_ns=lambda: 0,
        run_batch_fn=run_batch_fn,
    )

    assert len(engine.activate_calls) == 1
    assert factory_kwargs == [{
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "max_model_len": 4352,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 4,
        "max_num_prefill_tokens_per_step": 1024,
        "chunked_prefill_mixed_batch": False,
        "kv_offload_mvp0": True,
        "kv_offload_gpu_blocks": 68,
        "kv_offload_logical_blocks": 128,
        "kv_offload_blockwise_decode": False,
        "kv_offload_blockwise_prefill": False,
        "kv_offload_blockwise_blocks": 1,
    }]
    assert engine.activate_calls[0].adapter.kwargs == {
        "ngram_size": 3,
        "max_proposal_tokens": 4,
    }
    assert len(result["warmup_runs"]) == 1
    assert len(result["parity_runs"]) == 1
    assert len(result["measured_runs"]) == 5
    assert len(calls) == 7
    assert all(
        len(row["token_ids"]) == 4096
        for row in calls[0]["prompt_rows"]
    )
    assert engine.exit_calls == 1


def test_baseline_campaign_does_not_install_runtime_and_exits_on_failure():
    engine = FakeCampaignEngine()

    with pytest.raises(RuntimeError, match="injected"):
        _worker().run_policy_campaign(
            model_path="/model",
            policy="baseline",
            batch_size=1,
            engine_factory=lambda *args, **kwargs: engine,
            sampling_params_type=FakeSamplingParams,
            runtime_type=FakeRuntime,
            adapter_type=FakeAdapter,
            synchronize=lambda: None,
            clock_ns=lambda: 0,
            run_batch_fn=lambda **kwargs: (
                (_ for _ in ()).throw(RuntimeError("injected"))
            ),
        )

    assert engine.activate_calls == []
    assert engine.exit_calls == 1


def _performance_run(
    *,
    policy,
    batch_size,
    tpot_s,
    throughput,
):
    candidate = policy == "ngram"
    per_request = [
        {
            "sequence_id": index,
            "output_tokens": 64,
            "ttft_s": 0.5 + index * 0.01,
            "tpot_s": tpot_s,
            "completion_latency_s": (
                0.5 + index * 0.01 + tpot_s * 63
            ),
        }
        for index in range(batch_size)
    ]
    runtime = {
        "engine_steps": 16,
        "prefill_steps": 4,
        "decode_steps": 12,
        "selected_rows": batch_size * 8 if candidate else 0,
        "proposal_rows": batch_size * 6 if candidate else 0,
        "proposed_tokens": batch_size * 18 if candidate else 0,
        "accepted_draft_tokens": (
            batch_size * 9 if candidate else 0
        ),
        "first_target_callbacks": 8 if candidate else 0,
        "tail_callbacks": 6 if candidate else 0,
        "speculative_output_tokens": (
            batch_size * 20 if candidate else 0
        ),
        "target_callbacks": 14 if candidate else 0,
        "acceptance_rate": 0.5 if candidate else 0.0,
        "timing_ms": (
            {"target_forward_ms": 10.0}
            if candidate
            else {}
        ),
    }
    movement_totals = {
        "h2d_copies": 10,
        "h2d_bytes": 10240,
        "d2h_copies": 8,
        "d2h_bytes": 8192,
        "copy_waits": 4,
        "evictions": 3,
        "evict_clean": 3,
        "speculative_residency_committed_blocks": (
            2 if candidate else 0
        ),
        "speculative_residency_rejected_blocks": (
            1 if candidate else 0
        ),
        "speculative_residency_rejected_d2h_copies": 0,
    }
    return {
        "outputs": [
            list(range(64))
            for _ in range(batch_size)
        ],
        "timing": {
            "request_count": batch_size,
            "total_output_tokens": batch_size * 64,
            "batch_elapsed_s": (
                batch_size * 64 / throughput
            ),
            "batch_token_throughput_tps": throughput,
            "request_throughput_rps": (
                batch_size
                / (batch_size * 64 / throughput)
            ),
            "per_request": per_request,
        },
        "runtime": runtime,
        "movement": {
            "ranks": [
                {"rank": 0, **movement_totals},
            ],
            "totals": movement_totals,
        },
        "memory": {
            "ranks": [{
                "rank": 0,
                "reset": {
                    "rank": 0,
                    "cuda_allocated_bytes": 100,
                    "cuda_reserved_bytes": 120,
                    "cuda_peak_allocated_bytes": 100,
                    "cuda_peak_reserved_bytes": 120,
                    "kv_capacity_bytes": 64,
                },
                "final": {
                    "rank": 0,
                    "cuda_allocated_bytes": 110,
                    "cuda_reserved_bytes": 130,
                    "cuda_peak_allocated_bytes": 180,
                    "cuda_peak_reserved_bytes": 220,
                    "kv_capacity_bytes": 64,
                },
                "peak_allocated_delta_bytes": 80,
                "peak_reserved_delta_bytes": 100,
            }],
            "peak_allocated_bytes": 180,
            "peak_reserved_bytes": 220,
            "peak_allocated_delta_bytes": 80,
            "peak_reserved_delta_bytes": 100,
        },
        "evicted_block_identities": [[0, 1]],
        "observations": [],
    }


def _worker_fixture(policy, batch_size):
    candidate = policy == "ngram"
    tpot_s = 0.015 if candidate else 0.020
    throughput = 60.0 if candidate else 50.0
    run = _performance_run(
        policy=policy,
        batch_size=batch_size,
        tpot_s=tpot_s,
        throughput=throughput,
    )
    return {
        "policy": policy,
        "batch_size": batch_size,
        "prompt_rows": [
            {
                "prompt_index": index,
                "seed": f"seed-{index}",
                "token_ids": [index + 1] * 4096,
                "token_count": 4096,
                "sha256": hashlib.sha256(
                    json.dumps(
                        [index + 1] * 4096,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
            for index in range(batch_size)
        ],
        "warmup_runs": [copy.deepcopy(run)],
        "parity_runs": [copy.deepcopy(run)],
        "measured_runs": [
            copy.deepcopy(run) for _ in range(5)
        ],
        "tokenizer_identifier": "Qwen3-0.6B",
        "dtype": "bfloat16",
    }


def _worker_fixtures():
    return [
        _worker_fixture(policy, batch_size)
        for policy in ("baseline", "ngram")
        for batch_size in (1, 4)
    ]


def _environment():
    return {
        "model_path": "/models/Qwen3-0.6B",
        "model_identifier": "Qwen3-0.6B",
        "tensor_parallel_size": 1,
        "temperature": 0.0,
        "device_name": "A100",
        "python_version": "3.12.1",
        "torch_version": "2.4.1",
        "command": ["python", "gate.py"],
    }


def _artifact():
    return _gate().build_performance_artifact(
        worker_results=_worker_fixtures(),
        environment=_environment(),
        source_files={
            "tinyvllm/engine/llm_engine.py": "a" * 64,
        },
    )


def test_artifact_requires_four_cells_exact_parity_and_real_paths():
    artifact = _artifact()

    result = _gate().validate_performance_artifact(artifact)

    assert artifact["schema_version"] == 1
    assert artifact["classification"] == "NOT_PROMOTABLE"
    assert artifact["direction"] == "POSITIVE"
    assert result["status"] == "PASS"
    assert result["batch_directions"] == {
        "1": "IMPROVED",
        "4": "IMPROVED",
    }


@pytest.mark.parametrize(
    "mutate,match",
    (
        (
            lambda rows: rows.pop(),
            "four",
        ),
        (
            lambda rows: rows[0]["measured_runs"].pop(),
            "five",
        ),
        (
            lambda rows: rows[0]["prompt_rows"][0].__setitem__(
                "token_count",
                4095,
            ),
            "4096",
        ),
        (
            lambda rows: rows[2]["measured_runs"][0][
                "outputs"
            ][0].__setitem__(0, 999),
            "parity",
        ),
        (
            lambda rows: rows[2]["measured_runs"][0][
                "runtime"
            ].__setitem__("accepted_draft_tokens", 0),
            "accepted",
        ),
        (
            lambda rows: rows[0]["measured_runs"][0][
                "movement"
            ]["totals"].__setitem__("h2d_copies", 0),
            "movement",
        ),
        (
            lambda rows: rows[0]["measured_runs"][0].__setitem__(
                "evicted_block_identities",
                [],
            ),
            "evicted",
        ),
    ),
)
def test_artifact_rejects_invalid_worker_evidence(
    mutate,
    match,
):
    rows = _worker_fixtures()
    mutate(rows)

    with pytest.raises(ValueError, match=match):
        _gate().build_performance_artifact(
            worker_results=rows,
            environment=_environment(),
            source_files={
                "tinyvllm/engine/llm_engine.py": "a" * 64,
            },
        )


def test_artifact_validation_recomputes_derived_direction():
    artifact = _artifact()
    artifact["direction"] = "NEGATIVE"

    with pytest.raises(ValueError, match="direction"):
        _gate().validate_performance_artifact(artifact)


def test_artifact_validation_tolerates_cross_python_float_roundoff():
    artifact = _artifact()
    artifact["cells"]["baseline:b1"]["aggregate"][
        "tpot_s"
    ]["pstdev"] += 1e-16

    assert _gate().validate_performance_artifact(
        artifact
    )["status"] == "PASS"


def test_artifact_validation_rejects_material_aggregate_drift():
    artifact = _artifact()
    artifact["cells"]["baseline:b1"]["aggregate"][
        "tpot_s"
    ]["median"] += 1e-3

    with pytest.raises(ValueError, match="aggregate"):
        _gate().validate_performance_artifact(artifact)


def test_source_hash_verifier_rejects_changed_file(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("before\n", encoding="utf-8")
    artifact = _artifact()
    artifact["source_files"] = {
        "source.py": hashlib.sha256(
            source.read_bytes()
        ).hexdigest(),
    }
    artifact_path = tmp_path / "result.json"
    artifact_path.write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )
    source.write_text("after\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash"):
        _verifier().verify_performance_artifact(
            artifact_path,
            tmp_path,
        )


def test_parent_gate_launches_four_isolated_worker_cells(
    tmp_path,
):
    commands = []

    def worker_runner(command, *, log_path, cwd):
        commands.append(list(command))
        policy = command[command.index("--policy") + 1]
        batch_size = int(
            command[command.index("--batch-size") + 1]
        )
        output_path = Path(
            command[command.index("--out") + 1]
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                _worker_fixture(policy, batch_size)
            ),
            encoding="utf-8",
        )
        Path(log_path).write_text(
            f"{policy} b{batch_size}\n",
            encoding="utf-8",
        )
        return 0

    source = tmp_path / "source.py"
    source.write_text("source\n", encoding="utf-8")
    output_path = tmp_path / "result.json"

    artifact = _gate().run_performance_gate(
        model_path="/models/Qwen3-0.6B",
        output_path=output_path,
        repo_root=tmp_path,
        worker_script=WORKER_PATH,
        worker_runner=worker_runner,
        python_executable=sys.executable,
        source_files=("source.py",),
        environment=_environment(),
    )

    assert len(commands) == 4
    assert {
        (
            command[command.index("--policy") + 1],
            int(command[command.index("--batch-size") + 1]),
        )
        for command in commands
    } == {
        ("baseline", 1),
        ("baseline", 4),
        ("ngram", 1),
        ("ngram", 4),
    }
    assert artifact["direction"] == "POSITIVE"
    assert json.loads(
        output_path.read_text(encoding="utf-8")
    )["status"] == "PASS"


def test_remote_runner_uses_fixed_authority_and_verifies_both_sides():
    assert REMOTE_SCRIPT_PATH.exists()
    source = REMOTE_SCRIPT_PATH.read_text(encoding="utf-8")

    for required in (
        "sitian@10.232.195.203",
        "/tmp/ssh-sitian-10.232.195.203",
        "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python",
        "/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0.6B",
        "/data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge",
        'GPU_ID="${GPU_ID:-0}"',
        "tinyvllm/",
        "tools/speculative_runtime_performance_gate.py",
        "tools/speculative_runtime_performance_worker.py",
        "tools/verify_speculative_runtime_performance_gate.py",
        "tools/test_speculative_runtime_performance_gate.py",
        "python3 tools/verify_speculative_runtime_performance_gate.py",
        "verify.remote.json",
        "verify.json",
        "rsync -av",
        "set +e",
        '"${remote_python}" -m py_compile',
    ):
        assert required in source

    assert '"${remote_python}" -m pytest' not in source
    assert (
        'CUDA_VISIBLE_DEVICES="${gpu_id}"'
        in source
    )
