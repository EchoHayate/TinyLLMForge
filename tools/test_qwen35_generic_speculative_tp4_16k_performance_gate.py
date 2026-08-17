from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = (
    ROOT
    / "tools"
    / "qwen35_generic_speculative_tp4_16k_performance_gate.py"
)
WORKER_PATH = (
    ROOT
    / "tools"
    / "qwen35_generic_speculative_tp4_16k_performance_worker.py"
)
VERIFY_PATH = (
    ROOT
    / "tools"
    / "verify_qwen35_generic_speculative_tp4_16k_performance_gate.py"
)
REMOTE_RUNNER_PATH = (
    ROOT
    / "tools"
    / "run_qwen35_generic_speculative_tp4_16k_performance_gate_remote.sh"
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
        "qwen35_generic_speculative_tp4_16k_performance_gate_test_module",
        GATE_PATH,
    )


def _worker():
    _load_module(
        "qwen35_generic_speculative_tp4_16k_performance_gate",
        GATE_PATH,
    )
    return _load_module(
        "qwen35_generic_speculative_tp4_16k_performance_worker_test_module",
        WORKER_PATH,
    )


def _verifier():
    return _load_module(
        "verify_qwen35_generic_speculative_tp4_16k_performance_gate_test_module",
        VERIFY_PATH,
    )


def test_campaign_constants_are_independent_tp4_16k_contract():
    gate = _gate()

    assert gate.SCHEMA_VERSION == (
        "qwen35.generic-speculative-tp4-16k-performance.v1"
    )
    assert gate.CLASSIFICATION == (
        "SECOND_MODEL_TP4_16K_PERFORMANCE_MEASURED"
    )
    assert gate.WORLD_SIZE == 4
    assert gate.PROMPT_TOKENS == 16384
    assert gate.MAX_OUTPUT_TOKENS == 64
    assert gate.BATCH_SIZES == (1, 4)
    assert gate.POLICIES == ("baseline", "ngram")
    assert gate.WARMUP_RUNS == 1
    assert gate.PARITY_RUNS == 1
    assert gate.MEASURED_RUNS == 5
    assert gate.NGRAM_SIZE == 3
    assert gate.MAX_PROPOSAL_TOKENS == 4
    assert "h2d_bytes" in gate.REAL_MOVEMENT_KEYS
    assert "d2h_bytes" in gate.REAL_MOVEMENT_KEYS
    assert (
        "tools/qwen35_generic_speculative_tp4_16k_performance_gate.py"
        in gate.DEFAULT_SOURCE_FILES
    )


def test_cell_key_is_policy_and_batch_specific():
    gate = _gate()

    assert gate.cell_key("baseline", 1) == "baseline:b1"
    assert gate.cell_key("ngram", 4) == "ngram:b4"

    with pytest.raises(ValueError, match="policy"):
        gate.cell_key("unknown", 1)
    with pytest.raises(ValueError, match="batch"):
        gate.cell_key("baseline", 2)


def test_counter_delta_uses_monotonic_real_summaries():
    result = _gate().subtract_counter_summaries(
        {"h2d_copies": 2, "h2d_bytes": 1024},
        {"h2d_copies": 5, "h2d_bytes": 4096},
        keys=("h2d_copies", "h2d_bytes"),
    )

    assert result == {
        "h2d_copies": 3,
        "h2d_bytes": 3072,
    }


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


def _movement_fixture():
    gate = _gate()
    ranks = []
    for rank in range(gate.WORLD_SIZE):
        row = {key: 0 for key in gate.REAL_MOVEMENT_KEYS}
        row.update({
            "rank": rank,
            "h2d_copies": rank + 1,
            "h2d_bytes": (rank + 1) * 1024,
            "d2h_copies": rank + 2,
            "d2h_bytes": (rank + 2) * 2048,
            "copy_waits": rank + 3,
            "evictions": rank + 4,
            "evict_clean": rank + 4,
        })
        ranks.append(row)
    return {
        "ranks": ranks,
        "totals": {
            key: sum(row[key] for row in ranks)
            for key in gate.REAL_MOVEMENT_KEYS
        },
    }


def _memory_snapshot(rank, *, allocated, reserved, peak_allocated, peak_reserved):
    return {
        "rank": rank,
        "cuda_allocated_bytes": allocated,
        "cuda_reserved_bytes": reserved,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "kv_capacity_bytes": 4096,
    }


def _memory_fixture():
    ranks = []
    for rank in range(4):
        reset = _memory_snapshot(
            rank,
            allocated=100 + rank,
            reserved=200 + rank,
            peak_allocated=100 + rank,
            peak_reserved=200 + rank,
        )
        final = _memory_snapshot(
            rank,
            allocated=150 + rank,
            reserved=260 + rank,
            peak_allocated=400 + rank,
            peak_reserved=600 + rank,
        )
        ranks.append({
            "rank": rank,
            "reset": reset,
            "final": final,
            "peak_allocated_delta_bytes": 300,
            "peak_reserved_delta_bytes": 400,
        })
    return {
        "ranks": ranks,
        "peak_allocated_bytes": 403,
        "peak_reserved_bytes": 603,
        "peak_allocated_delta_bytes": 300,
        "peak_reserved_delta_bytes": 400,
    }


def _runtime_fixture(policy):
    if policy == "baseline":
        return {
            "engine_steps": 65,
            "prefill_steps": 1,
            "decode_steps": 64,
            "selected_rows": 0,
            "proposal_rows": 0,
            "proposed_tokens": 0,
            "accepted_draft_tokens": 0,
            "first_target_callbacks": 0,
            "tail_callbacks": 0,
            "speculative_output_tokens": 0,
            "target_callbacks": 0,
            "acceptance_rate": 0.0,
            "timing_ms": {},
        }
    return {
        "engine_steps": 30,
        "prefill_steps": 1,
        "decode_steps": 29,
        "selected_rows": 20,
        "proposal_rows": 20,
        "proposed_tokens": 60,
        "accepted_draft_tokens": 45,
        "first_target_callbacks": 20,
        "tail_callbacks": 19,
        "speculative_output_tokens": 45,
        "target_callbacks": 39,
        "acceptance_rate": 0.75,
        "timing_ms": {"proposal": 1.25},
    }


def _run_fixture(policy="baseline", batch_size=1):
    token_events = {
        sequence_id: [
            (1_500_000_000, 1),
            (3_000_000_000, 63),
        ]
        for sequence_id in range(batch_size)
    }
    return {
        "outputs": [
            list(range(64))
            for _ in range(batch_size)
        ],
        "timing": _gate().build_run_metrics(
            request_start_ns=1_000_000_000,
            request_finish_ns=3_000_000_000,
            token_events=token_events,
            finished_at_ns={
                sequence_id: 3_000_000_000
                for sequence_id in range(batch_size)
            },
            expected_output_tokens=64,
        ),
        "movement": _movement_fixture(),
        "memory": _memory_fixture(),
        "runtime": _runtime_fixture(policy),
        "observations": [],
    }


def test_validate_movement_requires_complete_four_rank_inventory_and_totals():
    gate = _gate()
    fixture = _movement_fixture()

    normalized = gate.validate_movement(fixture)

    assert [row["rank"] for row in normalized["ranks"]] == [0, 1, 2, 3]
    assert normalized["totals"]["h2d_bytes"] == 10 * 1024


@pytest.mark.parametrize(
    "mutate,match",
    (
        (lambda value: value["ranks"].pop(), "four"),
        (
            lambda value: value["ranks"][3].update(rank=2),
            "rank inventory",
        ),
        (
            lambda value: value["ranks"][0].update(h2d_bytes=-1),
            "non-negative",
        ),
        (
            lambda value: value["totals"].update(h2d_bytes=1),
            "totals",
        ),
        (
            lambda value: (
                value["ranks"][0].update(
                    speculative_residency_rejected_d2h_copies=1
                ),
                value["totals"].update(
                    speculative_residency_rejected_d2h_copies=1
                ),
            ),
            "rejected speculative D2H",
        ),
    ),
)
def test_validate_movement_rejects_invalid_rank_evidence(mutate, match):
    fixture = _movement_fixture()
    mutate(fixture)

    with pytest.raises(ValueError, match=match):
        _gate().validate_movement(fixture)


def test_validate_memory_recomputes_four_rank_peaks_and_deltas():
    normalized = _gate().validate_memory(_memory_fixture())

    assert normalized["peak_allocated_bytes"] == 403
    assert normalized["peak_reserved_bytes"] == 603
    assert normalized["peak_allocated_delta_bytes"] == 300
    assert normalized["peak_reserved_delta_bytes"] == 400


def test_validate_runtime_separates_baseline_and_candidate_contracts():
    gate = _gate()

    baseline = gate.validate_runtime(
        _runtime_fixture("baseline"),
        policy="baseline",
    )
    candidate = gate.validate_runtime(
        _runtime_fixture("ngram"),
        policy="ngram",
    )

    assert baseline["proposed_tokens"] == 0
    assert candidate["acceptance_rate"] == 0.75

    invalid_baseline = _runtime_fixture("baseline")
    invalid_baseline["proposed_tokens"] = 1
    with pytest.raises(ValueError, match="baseline speculative"):
        gate.validate_runtime(invalid_baseline, policy="baseline")

    invalid_candidate = _runtime_fixture("ngram")
    invalid_candidate["accepted_draft_tokens"] = 0
    invalid_candidate["acceptance_rate"] = 0.0
    with pytest.raises(ValueError, match="candidate accepted proposal"):
        gate.validate_runtime(invalid_candidate, policy="ngram")


def test_validate_run_accepts_complete_tp4_performance_evidence():
    normalized = _gate().validate_run(
        _run_fixture(policy="ngram", batch_size=4),
        policy="ngram",
        batch_size=4,
    )

    assert len(normalized["outputs"]) == 4
    assert normalized["timing"]["total_output_tokens"] == 256
    assert len(normalized["movement"]["ranks"]) == 4
    assert len(normalized["memory"]["ranks"]) == 4


def test_validate_run_allows_monotonic_engine_sequence_ids_across_runs():
    fixture = _run_fixture(policy="baseline", batch_size=4)
    for offset, row in enumerate(fixture["timing"]["per_request"]):
        row["sequence_id"] = 8 + offset

    normalized = _gate().validate_run(
        fixture,
        policy="baseline",
        batch_size=4,
    )

    assert [
        row["sequence_id"]
        for row in normalized["timing"]["per_request"]
    ] == [8, 9, 10, 11]


@pytest.mark.parametrize(
    "mutate,match",
    (
        (lambda value: value["outputs"].pop(), "4 outputs"),
        (lambda value: value["outputs"][0].pop(), "64 tokens"),
        (lambda value: value.pop("timing"), "timing"),
        (lambda value: value["movement"]["ranks"].pop(), "four"),
        (lambda value: value["memory"]["ranks"].pop(), "four"),
    ),
)
def test_validate_run_rejects_incomplete_evidence(mutate, match):
    fixture = _run_fixture(policy="ngram", batch_size=4)
    mutate(fixture)

    with pytest.raises(ValueError, match=match):
        _gate().validate_run(
            fixture,
            policy="ngram",
            batch_size=4,
        )


def _prompt_rows(batch_size):
    rows = []
    for prompt_index in range(batch_size):
        token_ids = [prompt_index + 1] * 16384
        digest = hashlib.sha256(
            json.dumps(
                token_ids,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        rows.append({
            "prompt_index": prompt_index,
            "token_count": len(token_ids),
            "token_ids": token_ids,
            "sha256": digest,
        })
    return rows


def _measured_run(policy, batch_size, run_index):
    run = _run_fixture(policy=policy, batch_size=batch_size)
    if policy == "baseline":
        ttft = 1.0 + run_index * 0.01
        tpot = 0.020 + run_index * 0.001
        throughput = 50.0 + run_index
    else:
        ttft = 0.9 + run_index * 0.01
        tpot = 0.015 + run_index * 0.001
        throughput = 60.0 + run_index
    completion = ttft + tpot * 63
    elapsed = batch_size * 64 / throughput
    for row in run["timing"]["per_request"]:
        row["ttft_s"] = ttft
        row["tpot_s"] = tpot
        row["completion_latency_s"] = completion
    run["timing"]["batch_elapsed_s"] = elapsed
    run["timing"]["batch_token_throughput_tps"] = throughput
    run["timing"]["request_throughput_rps"] = batch_size / elapsed
    return run


def _worker_fixture(policy, batch_size):
    parity = _measured_run(policy, batch_size, 0)
    return {
        "policy": policy,
        "batch_size": batch_size,
        "prompt_rows": _prompt_rows(batch_size),
        "warmup_runs": [_measured_run(policy, batch_size, 0)],
        "parity_runs": [parity],
        "measured_runs": [
            _measured_run(policy, batch_size, run_index)
            for run_index in range(5)
        ],
        "tokenizer_identifier": "qwen35-test-tokenizer",
        "dtype": "torch.bfloat16",
        "cleanup_receipt": _cleanup_receipt(),
    }


def _worker_results():
    return [
        _worker_fixture("baseline", 1),
        _worker_fixture("ngram", 1),
        _worker_fixture("baseline", 4),
        _worker_fixture("ngram", 4),
    ]


def _cleanup_receipt():
    return {
        "process_group_destroyed": True,
        "rank_exit_codes": [0, 0, 0, 0],
        "owned_children_remaining": [],
        "rank_cleanup_receipts": [
            {
                "rank": rank,
                "worker_exit_code": 0,
                "process_group_initialized": False,
                "engine_exit_called": True,
                "live_lease_count": 0,
                "prepared_transaction_count": 0,
                "runtime_poisoned": False,
            }
            for rank in range(4)
        ],
    }


def test_validate_worker_and_aggregate_preserve_five_run_distributions():
    gate = _gate()
    worker = gate.validate_worker_result(
        _worker_fixture("ngram", 4)
    )

    aggregate = gate.aggregate_worker(worker)

    assert len(worker["warmup_runs"]) == 1
    assert len(worker["parity_runs"]) == 1
    assert len(worker["measured_runs"]) == 5
    assert aggregate["ttft_s"]["count"] == 5
    assert aggregate["tpot_s"]["median"] == pytest.approx(0.017)
    assert aggregate["completion_latency_s"]["count"] == 5
    assert aggregate["batch_token_throughput_tps"]["median"] == 62.0
    assert aggregate["request_throughput_rps"]["count"] == 5
    assert aggregate["peak_allocated_bytes"]["count"] == 5
    assert aggregate["peak_reserved_bytes"]["count"] == 5
    assert aggregate["movement_totals"]["h2d_bytes"] > 0
    assert aggregate["runtime_totals"]["proposed_tokens"] > 0
    assert aggregate["runtime_totals"]["acceptance_rate"] == 0.75


def test_derive_comparison_reports_honest_ratios_and_direction():
    gate = _gate()
    derived = gate.derive_artifact(_worker_results())

    comparison = derived["comparisons"]["4"]

    assert comparison["direction"] == "IMPROVED"
    assert comparison["tpot_ratio"] == pytest.approx(0.017 / 0.022)
    assert comparison["tpot_percent_delta"] == pytest.approx(
        (0.017 / 0.022 - 1.0) * 100.0
    )
    assert comparison["throughput_ratio"] == pytest.approx(62.0 / 52.0)
    assert comparison["ttft_ratio"] == pytest.approx(0.92 / 1.02)
    assert comparison["peak_allocated_ratio"] == 1.0
    assert comparison["h2d_bytes_ratio"] == 1.0
    assert comparison["d2h_bytes_ratio"] == 1.0
    assert derived["campaign_direction"] == "POSITIVE"


@pytest.mark.parametrize(
    "mutate,match",
    (
        (
            lambda workers: workers.pop(),
            "four worker cells",
        ),
        (
            lambda workers: workers.append(copy.deepcopy(workers[0])),
            "four worker cells",
        ),
        (
            lambda workers: workers[0]["measured_runs"].pop(),
            "five runs",
        ),
        (
            lambda workers: workers[1]["prompt_rows"][0].update(
                token_ids=[9] * 16384,
                sha256=hashlib.sha256(
                    json.dumps(
                        [9] * 16384,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            ),
            "prompt parity",
        ),
        (
            lambda workers: workers[1]["measured_runs"][2][
                "outputs"
            ][0].__setitem__(0, 999),
            "exact token parity",
        ),
        (
            lambda workers: workers[1]["measured_runs"][0][
                "runtime"
            ].update(
                proposed_tokens=0,
                accepted_draft_tokens=0,
                acceptance_rate=0.0,
            ),
            "candidate accepted proposal",
        ),
        (
            lambda workers: [
                run["movement"]["totals"].update(
                    h2d_copies=0,
                    h2d_bytes=0,
                    d2h_copies=0,
                    d2h_bytes=0,
                )
                or [
                    row.update(
                        h2d_copies=0,
                        h2d_bytes=0,
                        d2h_copies=0,
                        d2h_bytes=0,
                    )
                    for row in run["movement"]["ranks"]
                ]
                for worker in workers
                if worker["batch_size"] == 4
                for run in worker["measured_runs"]
            ],
            "batch-4.*movement",
        ),
    ),
)
def test_derive_artifact_rejects_incomplete_or_unfair_campaign(
    mutate,
    match,
):
    workers = _worker_results()
    mutate(workers)

    with pytest.raises(ValueError, match=match):
        _gate().derive_artifact(workers)


class _FakeMeasuredEngine:
    def __init__(self):
        self.events = []
        self._finished = True
        self._step_index = 0
        self.last_step_observation = None

    def is_finished(self):
        self.events.append("idle_check")
        return self._finished

    def clear_reusable_prefix_cache(self):
        self.events.append("clear_prefix_cache")

    def kv_offload_summaries(self, timeout_s):
        assert timeout_s == 60.0
        phase = (
            "before_movement"
            if "before_movement" not in self.events
            else "after_movement"
        )
        self.events.append(phase)
        base = 0 if phase == "before_movement" else 10
        rows = []
        for rank in range(4):
            row = {
                key: base
                for key in _gate().REAL_MOVEMENT_KEYS
            }
            row[
                "speculative_residency_rejected_d2h_copies"
            ] = 0
            rows.append(row)
        return tuple(rows)

    def reset_peak_memory_stats(self, timeout_s):
        assert timeout_s == 60.0
        self.events.append("peak_reset")
        return tuple(
            _memory_snapshot(
                rank,
                allocated=100,
                reserved=200,
                peak_allocated=100,
                peak_reserved=200,
            )
            for rank in range(4)
        )

    def add_request(self, token_ids, sampling_params):
        assert len(token_ids) == 16384
        self.events.append("add_request")
        self._finished = False

    def step(self):
        self.events.append("step")
        if self._step_index == 0:
            self.last_step_observation = {
                "new_completion_tokens_by_seq": {0: [1]},
                "finished_seq_ids": [],
                "speculative_selected_seq_ids": [],
                "speculative_proposal_token_counts": {},
                "speculative_accepted_draft_token_counts": {},
                "speculative_output_token_counts": {},
                "speculative_first_target_callbacks": 0,
                "speculative_tail_callbacks": 0,
            }
            outputs = []
        else:
            self.last_step_observation = {
                "new_completion_tokens_by_seq": {0: list(range(63))},
                "finished_seq_ids": [0],
                "speculative_selected_seq_ids": [],
                "speculative_proposal_token_counts": {},
                "speculative_accepted_draft_token_counts": {},
                "speculative_output_token_counts": {},
                "speculative_first_target_callbacks": 0,
                "speculative_tail_callbacks": 0,
            }
            outputs = [(0, list(range(64)))]
            self._finished = True
        self._step_index += 1
        return outputs, 0

    def memory_snapshots(self, timeout_s):
        assert timeout_s == 60.0
        self.events.append("final_memory")
        return tuple(
            _memory_snapshot(
                rank,
                allocated=150,
                reserved=250,
                peak_allocated=400,
                peak_reserved=600,
            )
            for rank in range(4)
        )

    def evict_clean_resident_blocks(self, *args, **kwargs):
        raise AssertionError("manual eviction must not be called")

    def upload(self, *args, **kwargs):
        raise AssertionError("manual upload must not be called")


def test_loaded_worker_run_order_uses_natural_blockwise_movement():
    worker = _worker()
    engine = _FakeMeasuredEngine()
    clock_values = iter(
        [1_000_000_000, 1_500_000_000, 3_000_000_000]
    )

    result = worker.run_request_batch(
        engine=engine,
        prompt_rows=_prompt_rows(1),
        sampling_params=object(),
        expected_output_tokens=64,
        synchronize=lambda: engine.events.append("synchronize"),
        clock_ns=lambda: next(clock_values),
    )

    assert engine.events == [
        "idle_check",
        "clear_prefix_cache",
        "before_movement",
        "peak_reset",
        "synchronize",
        "add_request",
        "idle_check",
        "step",
        "synchronize",
        "idle_check",
        "step",
        "synchronize",
        "idle_check",
        "after_movement",
        "final_memory",
    ]
    assert result["timing"]["per_request"][0]["ttft_s"] == 0.5
    assert result["timing"]["per_request"][0]["tpot_s"] == pytest.approx(
        1.5 / 63
    )
    assert len(result["movement"]["ranks"]) == 4
    assert len(result["memory"]["ranks"]) == 4


class _FakeTokenizer:
    name_or_path = "fake-qwen35"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [1, 2, 3, 4]


class _FakeCampaignEngine:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.config = type(
            "Config",
            (),
            {"dtype": "torch.bfloat16"},
        )()
        self.activated = []
        self.exit_calls = 0

    def activate_speculative_runtime(self, runtime):
        self.activated.append(runtime)

    def exit(self):
        self.exit_calls += 1
        return {"fake": "exit"}


def test_loaded_worker_constructs_tp4_engine_and_runs_1_1_5(monkeypatch):
    worker = _worker()
    created = []
    run_calls = []

    def engine_factory(model_path, **kwargs):
        engine = _FakeCampaignEngine()
        created.append((model_path, kwargs, engine))
        return engine

    monkeypatch.setattr(
        worker,
        "build_prompt_rows",
        lambda tokenizer, batch_size: _prompt_rows(batch_size),
    )

    class Adapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Runtime:
        def __init__(self, adapter):
            self.adapter = adapter

    def run_batch_fn(**kwargs):
        run_calls.append(kwargs)
        return _run_fixture(
            policy="ngram",
            batch_size=len(kwargs["prompt_rows"]),
        )

    result = worker.run_policy_campaign(
        model_path="/checkpoint",
        gpu_indices=(7, 5, 3, 2),
        policy="ngram",
        batch_size=4,
        dist_port=29601,
        master_port=29701,
        engine_factory=engine_factory,
        sampling_params_type=lambda **kwargs: kwargs,
        runtime_type=Runtime,
        adapter_type=Adapter,
        synchronize=lambda: None,
        clock_ns=lambda: 1,
        run_batch_fn=run_batch_fn,
        cleanup_observations_fn=lambda engine: [
            {"rank": rank}
            for rank in range(4)
        ],
        merge_cleanup_fn=lambda receipt, observations, **kwargs: {
            "receipt": receipt,
            "observations": observations,
        },
    )

    _, kwargs, engine = created[0]
    assert kwargs["tensor_parallel_size"] == 4
    assert kwargs["max_model_len"] == 33024
    assert kwargs["max_num_batched_tokens"] == 132096
    assert kwargs["max_num_prefill_tokens_per_step"] == 1024
    assert kwargs["kv_offload_gpu_blocks"] == 48
    assert kwargs["kv_offload_logical_blocks"] == 640
    assert kwargs["kv_offload_blockwise_blocks"] == 8
    assert len(engine.activated) == 1
    assert engine.activated[0].adapter.kwargs == {
        "ngram_size": 3,
        "max_proposal_tokens": 4,
    }
    assert len(run_calls) == 7
    assert len(result["warmup_runs"]) == 1
    assert len(result["parity_runs"]) == 1
    assert len(result["measured_runs"]) == 5
    assert engine.exit_calls == 1
    assert result["cleanup_receipt"]["receipt"] == {"fake": "exit"}


def _artifact_environment():
    return {
        "python_version": "3.11.0",
        "torch_version": "2.7.0",
        "device_name": "NVIDIA H100",
        "gpu_inventory": {
            "selected_physical_indices": [7, 5, 3, 2],
            "campaign_start": [{"index": index} for index in range(8)],
            "pre_cells": {},
            "post_cells": {},
        },
    }


def test_build_and_validate_artifact_recompute_raw_campaign():
    gate = _gate()
    source_files = {
        "source.py": "a" * 64,
    }
    artifact = gate.build_performance_artifact(
        worker_results=_worker_results(),
        environment=_artifact_environment(),
        gpu_indices=(7, 5, 3, 2),
        source_files=source_files,
        source_tree_sha256="b" * 64,
        model_manifest_sha256=gate.MODEL_MANIFEST_SHA256,
    )

    validated = gate.validate_performance_artifact(artifact)

    assert artifact["schema_version"] == gate.SCHEMA_VERSION
    assert artifact["status"] == "PASS"
    assert artifact["classification"] == gate.CLASSIFICATION
    assert artifact["campaign"]["tensor_parallel_size"] == 4
    assert artifact["campaign"]["prompt_tokens"] == 16384
    assert artifact["engine_config"]["kv_offload_gpu_blocks"] == 48
    assert set(artifact["cells"]) == {
        "baseline:b1",
        "ngram:b1",
        "baseline:b4",
        "ngram:b4",
    }
    assert artifact["campaign_direction"] == "POSITIVE"
    assert validated["campaign_direction"] == "POSITIVE"

    mutated = copy.deepcopy(artifact)
    mutated["comparisons"]["4"]["tpot_ratio"] = 999.0
    with pytest.raises(ValueError, match="comparison"):
        gate.validate_performance_artifact(mutated)


def test_run_campaign_orders_cells_and_publishes_atomically(
    tmp_path,
    monkeypatch,
):
    gate = _gate()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "source.py").write_text(
        "print('source')\n",
        encoding="utf-8",
    )
    model_path = tmp_path / "checkpoint" / "model"
    model_path.mkdir(parents=True)
    manifest_path = model_path.parent / "model_manifest.json"
    manifest_path.write_text(
        '{"approved":true}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        gate,
        "MODEL_MANIFEST_SHA256",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    calls = []

    def worker_runner(command, *, log_path, cwd):
        policy = command[command.index("--policy") + 1]
        batch_size = int(
            command[command.index("--batch-size") + 1]
        )
        calls.append(f"{policy}:b{batch_size}")
        out = Path(command[command.index("--out") + 1])
        out.write_text(
            json.dumps(_worker_fixture(policy, batch_size)),
            encoding="utf-8",
        )
        Path(log_path).write_text("ok\n", encoding="utf-8")
        return 0

    output_dir = tmp_path / "authority"
    artifact = gate.run_campaign(
        model_path=str(model_path),
        gpu_indices=(7, 5, 3, 2),
        output_dir=output_dir,
        dist_port_base=29600,
        master_port_base=29700,
        repo_root=repo_root,
        worker_script=repo_root / "worker.py",
        worker_runner=worker_runner,
        source_files=("source.py",),
        environment=_artifact_environment(),
        verifier=lambda authority_path, source_root: {
            "classification": "PASS",
            "failures": [],
        },
    )

    assert calls == [
        "baseline:b1",
        "ngram:b1",
        "baseline:b4",
        "ngram:b4",
    ]
    assert artifact["status"] == "PASS"
    assert output_dir.is_dir()
    assert (output_dir / "result.json").is_file()
    assert (output_dir / "source_manifest.json").is_file()
    assert (output_dir / "verify.json").is_file()
    assert not output_dir.with_name("authority.failed").exists()


def test_run_campaign_retains_partial_failure_atomically(
    tmp_path,
    monkeypatch,
):
    gate = _gate()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "source.py").write_text("x = 1\n", encoding="utf-8")
    model_path = tmp_path / "checkpoint" / "model"
    model_path.mkdir(parents=True)
    manifest_path = model_path.parent / "model_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        gate,
        "MODEL_MANIFEST_SHA256",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    calls = []

    def worker_runner(command, *, log_path, cwd):
        policy = command[command.index("--policy") + 1]
        batch_size = int(
            command[command.index("--batch-size") + 1]
        )
        key = f"{policy}:b{batch_size}"
        calls.append(key)
        Path(log_path).write_text("worker failed\n", encoding="utf-8")
        if key == "ngram:b1":
            return 17
        out = Path(command[command.index("--out") + 1])
        out.write_text(
            json.dumps(_worker_fixture(policy, batch_size)),
            encoding="utf-8",
        )
        return 0

    output_dir = tmp_path / "authority"
    with pytest.raises(RuntimeError, match="failed_artifacts"):
        gate.run_campaign(
            model_path=str(model_path),
            gpu_indices=(7, 5, 3, 2),
            output_dir=output_dir,
            dist_port_base=29600,
            master_port_base=29700,
            repo_root=repo_root,
            worker_script=repo_root / "worker.py",
            worker_runner=worker_runner,
            source_files=("source.py",),
            environment=_artifact_environment(),
            verifier=lambda authority_path, source_root: {
                "classification": "PASS",
                "failures": [],
            },
        )

    assert calls == ["baseline:b1", "ngram:b1"]
    assert not output_dir.exists()
    failed_dir = output_dir.with_name("authority.failed")
    assert failed_dir.is_dir()
    assert (
        failed_dir / "cells" / "ngram:b1.log"
    ).read_text(encoding="utf-8") == "worker failed\n"


def _write_authority(tmp_path):
    gate = _gate()
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "source.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    source_files = ("source.py",)
    artifact = gate.build_performance_artifact(
        worker_results=_worker_results(),
        environment=_artifact_environment(),
        gpu_indices=(7, 5, 3, 2),
        source_files=gate.hash_source_files(
            source_root,
            source_files,
        ),
        source_tree_sha256=gate.source_tree_sha256(
            source_root,
            source_files,
        ),
        model_manifest_sha256=gate.MODEL_MANIFEST_SHA256,
    )
    authority = tmp_path / "authority"
    authority.mkdir()
    gate.write_json_atomic(authority / "result.json", artifact)
    gate.write_json_atomic(
        authority / "source_manifest.json",
        {
            "schema_version": gate.SCHEMA_VERSION,
            "source_tree_sha256": artifact[
                "source_tree_sha256"
            ],
            "model_manifest_sha256": artifact[
                "model_manifest_sha256"
            ],
            "source_files": artifact["source_files"],
            "artifacts": {
                "result.json": gate.sha256_file(
                    authority / "result.json"
                ),
            },
        },
    )
    return gate, authority, source_root


def test_independent_verifier_accepts_valid_authority(tmp_path):
    _, authority, source_root = _write_authority(tmp_path)

    verification = _verifier().verify_run(authority, source_root)

    assert verification == {
        "classification": "PASS",
        "failures": [],
    }


@pytest.mark.parametrize(
    "mutate,expected",
    (
        (
            lambda artifact, manifest: artifact["source_files"].update(
                {"source.py": "0" * 64}
            ),
            "source_files_mismatch",
        ),
        (
            lambda artifact, manifest: artifact.update(
                source_tree_sha256="0" * 64
            ),
            "source_tree_mismatch",
        ),
        (
            lambda artifact, manifest: artifact.update(
                model_manifest_sha256="0" * 64
            ),
            "artifact_validation",
        ),
        (
            lambda artifact, manifest: artifact["cells"][
                "baseline:b1"
            ]["measured_runs"][0]["timing"].update(
                batch_elapsed_s=999.0
            ),
            "artifact_validation",
        ),
        (
            lambda artifact, manifest: artifact["cells"][
                "baseline:b1"
            ]["aggregate"]["tpot_s"].update(median=999.0),
            "artifact_validation",
        ),
        (
            lambda artifact, manifest: artifact["comparisons"][
                "1"
            ].update(tpot_ratio=999.0),
            "artifact_validation",
        ),
        (
            lambda artifact, manifest: artifact["comparisons"][
                "1"
            ].update(direction="REGRESSED"),
            "artifact_validation",
        ),
        (
            lambda artifact, manifest: artifact.update(
                campaign_direction="NEGATIVE"
            ),
            "artifact_validation",
        ),
        (
            lambda artifact, manifest: artifact["cells"][
                "ngram:b1"
            ]["measured_runs"][0]["outputs"][0].__setitem__(0, 999),
            "artifact_validation",
        ),
        (
            lambda artifact, manifest: artifact["cells"][
                "ngram:b4"
            ]["measured_runs"][0]["movement"]["totals"].update(
                h2d_bytes=1
            ),
            "artifact_validation",
        ),
        (
            lambda artifact, manifest: (
                artifact["cells"]["ngram:b4"]["measured_runs"][0][
                    "movement"
                ]["ranks"][0].update(
                    speculative_residency_rejected_d2h_copies=1
                ),
                artifact["cells"]["ngram:b4"]["measured_runs"][0][
                    "movement"
                ]["totals"].update(
                    speculative_residency_rejected_d2h_copies=1
                ),
            ),
            "artifact_validation",
        ),
        (
            lambda artifact, manifest: artifact["cells"][
                "ngram:b4"
            ]["cleanup_receipt"].update(
                process_group_destroyed=False
            ),
            "artifact_validation",
        ),
    ),
)
def test_independent_verifier_rejects_mutated_authority(
    tmp_path,
    mutate,
    expected,
):
    gate, authority, source_root = _write_authority(tmp_path)
    artifact = json.loads(
        (authority / "result.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (authority / "source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    mutate(artifact, manifest)
    gate.write_json_atomic(authority / "result.json", artifact)
    manifest["artifacts"]["result.json"] = gate.sha256_file(
        authority / "result.json"
    )
    gate.write_json_atomic(
        authority / "source_manifest.json",
        manifest,
    )

    verification = _verifier().verify_run(authority, source_root)

    assert verification["classification"] == "FAIL"
    assert any(
        failure.startswith(expected)
        for failure in verification["failures"]
    )


def test_remote_runner_is_bounded_fixed_gpu_and_independently_verified():
    source = REMOTE_RUNNER_PATH.read_text(encoding="utf-8")

    for fragment in (
        "sitian@10.232.195.203",
        "FILE:/Users/bytedance/krb5cc_sitian",
        "ControlMaster=no",
        "ControlPath=none",
        "head -n 4",
        "MIN_FREE_MEMORY_MIB=49152",
        "MAX_GPU_UTILIZATION=10",
        "MAX_POST_CELL_DRIFT_MIB=4096",
        "REMOTE_COMMAND_RETRY_ATTEMPTS",
        "REMOTE_RSYNC_RETRY_ATTEMPTS",
        "MAX_POLL_ATTEMPTS",
        "opaque-",
        "campaign.status",
        "campaign.pid",
        "campaign.exit_code",
        "authority.failed",
        "verify.remote.json",
        "verify.local.json",
        "export PYTHONPATH=",
        "qwen35_generic_speculative_tp4_16k_performance_gate.py",
        "qwen35_generic_speculative_tp4_16k_performance_worker.py",
        "verify_qwen35_generic_speculative_tp4_16k_performance_gate.py",
    ):
        assert fragment in source
    assert "ControlMaster=yes" not in source
    assert "while true" not in source
