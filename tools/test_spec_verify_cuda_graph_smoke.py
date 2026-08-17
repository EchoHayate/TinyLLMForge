from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SMOKE_PATH = ROOT / "tools" / "spec_verify_cuda_graph_smoke.py"


def _load_smoke():
    assert SMOKE_PATH.is_file(), f"missing smoke: {SMOKE_PATH}"
    spec = importlib.util.spec_from_file_location(
        "spec_verify_cuda_graph_smoke_test_module",
        SMOKE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("query_len", "width", "expected"),
    (
        (1, 1, 255),
        (3, 1, 253),
        (1, 2, 256),
        (3, 2, 254),
    ),
)
def test_prompt_length_targets_exact_terminal_width(
    query_len,
    width,
    expected,
):
    smoke = _load_smoke()

    assert smoke.prompt_length_for_family(
        query_len=query_len,
        page_table_width=width,
        block_size=256,
    ) == expected


def test_build_prompt_token_batch_is_exact_and_distinct():
    smoke = _load_smoke()

    prompts = smoke.build_prompt_token_batch(
        seed_token_ids=(11, 12, 13, 14),
        batch_size=4,
        query_len=3,
        page_table_width=2,
        block_size=256,
    )

    assert len(prompts) == 4
    assert {len(prompt) for prompt in prompts} == {254}
    assert len(set(prompts)) == 4
    assert all(prompt[-1] in (11, 12, 13, 14) for prompt in prompts)


def test_oracle_adapter_accepts_q_minus_one_and_rejects_last():
    smoke = _load_smoke()

    class Capability:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Proposal:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    prompt = (11, 12, 13)
    adapter = smoke.OracleMismatchDraftAdapter(
        oracle_tokens_by_prompt={prompt: (21, 22, 23, 24)},
        query_len=3,
        vocab_size=100,
        adapter_types=(Capability, Proposal),
    )
    proposals = adapter.propose_batch((
        SimpleNamespace(
            sequence_id=7,
            token_ids=prompt,
            max_proposal_tokens=3,
            first_target_token=21,
        ),
    ))

    assert adapter.capabilities.max_proposal_tokens == 3
    assert len(proposals) == 1
    assert proposals[0].token_ids[:2] == (21, 22)
    assert proposals[0].token_ids[2] != 23
    assert proposals[0].metadata == {
        "accepted_prefix_length": 2,
        "query_len": 3,
    }


def test_oracle_adapter_q1_rejects_the_only_token():
    smoke = _load_smoke()

    class Capability:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Proposal:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    prompt = (31, 32)
    adapter = smoke.OracleMismatchDraftAdapter(
        oracle_tokens_by_prompt={prompt: (41, 42)},
        query_len=1,
        vocab_size=100,
        adapter_types=(Capability, Proposal),
    )
    proposal = adapter.propose_batch((
        SimpleNamespace(
            sequence_id=9,
            token_ids=prompt,
            max_proposal_tokens=1,
            first_target_token=41,
        ),
    ))[0]

    assert proposal.token_ids != (41,)
    assert proposal.metadata["accepted_prefix_length"] == 0


def test_build_transaction_results_allows_no_unused_blocks():
    smoke = _load_smoke()
    rows = (
        {
            "transaction_state": "committed",
            "materialized_token_count": 1,
            "accepted_token_count": 0,
            "unused_block_ids": (),
            "released_block_ids": (),
        },
    )

    result = smoke.build_transaction_results(
        rows,
        accepted_prefix_kv_parity=True,
    )

    assert result == {
        "accepted_prefix_kv_parity": True,
        "rejected_suffix_released": True,
        "transaction_states": ["committed"],
        "materialized_token_counts": [1],
        "committed_materialized_token_counts": [0],
        "rejected_materialized_token_counts": [1],
        "unused_block_ids": [],
        "released_block_ids": [],
        "all_unused_blocks_released": True,
    }


def test_build_transaction_results_rejects_release_mismatch():
    smoke = _load_smoke()

    with pytest.raises(ValueError, match="released"):
        smoke.build_transaction_results(
            (
                {
                    "transaction_state": "committed",
                    "materialized_token_count": 3,
                    "accepted_token_count": 3,
                    "unused_block_ids": (8,),
                    "released_block_ids": (),
                },
            ),
            accepted_prefix_kv_parity=True,
        )


def test_build_performance_evidence_separates_warmed_and_mixed():
    smoke = _load_smoke()
    rows = (
        {
            "family_id": "b1-q1-w1",
            "batch_size": 1,
            "query_len": 1,
            "prompt_length": 255,
            "accepted_draft_tokens": 0,
            "eager_request_metrics": {
                "elapsed_ns": 10_000,
                "ttft_ns": 4_000,
                "output_token_count": 1,
                "gpu_allocated_bytes": 100,
                "gpu_reserved_bytes": 200,
            },
            "cold_request_metrics": {
                "elapsed_ns": 12_000,
                "ttft_ns": 5_000,
                "output_token_count": 1,
                "gpu_allocated_bytes": 110,
                "gpu_reserved_bytes": 210,
            },
            "capture_request_metrics": {
                "elapsed_ns": 16_000,
                "ttft_ns": 6_000,
                "output_token_count": 1,
                "gpu_allocated_bytes": 120,
                "gpu_reserved_bytes": 220,
            },
            "warmed_request_metrics": {
                "elapsed_ns": 8_000,
                "ttft_ns": 3_000,
                "output_token_count": 1,
                "gpu_allocated_bytes": 115,
                "gpu_reserved_bytes": 215,
            },
            "warmed_verifier_latency_ns": 1_000,
            "capture_duration_ns": 5_000,
            "capture_allocated_delta_bytes": 10,
            "capture_reserved_delta_bytes": 20,
        },
        {
            "family_id": "b4-q3-w2",
            "batch_size": 4,
            "query_len": 3,
            "prompt_length": 254,
            "accepted_draft_tokens": 8,
            "eager_request_metrics": {
                "elapsed_ns": 30_000,
                "ttft_ns": 8_000,
                "output_token_count": 12,
                "gpu_allocated_bytes": 300,
                "gpu_reserved_bytes": 400,
            },
            "cold_request_metrics": {
                "elapsed_ns": 35_000,
                "ttft_ns": 9_000,
                "output_token_count": 12,
                "gpu_allocated_bytes": 310,
                "gpu_reserved_bytes": 410,
            },
            "capture_request_metrics": {
                "elapsed_ns": 40_000,
                "ttft_ns": 10_000,
                "output_token_count": 12,
                "gpu_allocated_bytes": 320,
                "gpu_reserved_bytes": 420,
            },
            "warmed_request_metrics": {
                "elapsed_ns": 25_000,
                "ttft_ns": 7_000,
                "output_token_count": 12,
                "gpu_allocated_bytes": 315,
                "gpu_reserved_bytes": 415,
            },
            "warmed_verifier_latency_ns": 2_000,
            "capture_duration_ns": 6_000,
            "capture_allocated_delta_bytes": 11,
            "capture_reserved_delta_bytes": 21,
        },
    )

    result = smoke.build_performance_evidence(
        rows,
        cache_counts={
            "hits": 2,
            "misses": 4,
            "evictions": 0,
            "quarantines": 0,
        },
    )

    assert result["warmup_count"] == 2
    assert result["measurement_count"] == 2
    assert result["warmed_exact_graph_hits"][
        "latency_ns_by_family"
    ] == {
        "b1-q1-w1": [1_000],
        "b4-q3-w2": [2_000],
    }
    assert result["mixed_hit_rate"]["hit_count"] == 2
    assert result["mixed_hit_rate"]["miss_count"] == 4
    assert result["mixed_hit_rate"]["measurement_count"] == 6
    assert result["acceptance"] == {
        "proposed_tokens": 13,
        "accepted_draft_tokens": 8,
        "acceptance_rate": 8 / 13,
    }
    assert result["eager_baseline"]["gpu_allocated_bytes"] == 300
    assert result["warmed_exact_graph_hits"][
        "gpu_reserved_bytes"
    ] == 415


def test_build_performance_evidence_keeps_repeated_warmed_samples():
    smoke = _load_smoke()
    request_metrics = {
        "elapsed_ns": 8_000,
        "ttft_ns": 3_000,
        "output_token_count": 1,
        "gpu_allocated_bytes": 115,
        "gpu_reserved_bytes": 215,
    }
    row = {
        "family_id": "b1-q1-w1",
        "batch_size": 1,
        "query_len": 1,
        "prompt_length": 255,
        "accepted_draft_tokens": 0,
        "eager_request_metrics": dict(request_metrics),
        "cold_request_metrics": dict(request_metrics),
        "capture_request_metrics": dict(request_metrics),
        "warmed_request_metrics": dict(request_metrics),
        "warmed_request_metrics_rows": (
            dict(request_metrics),
            dict(request_metrics),
        ),
        "warmed_verifier_latency_ns": 1_000,
        "warmed_verifier_latency_ns_rows": (
            1_000,
            1_100,
        ),
        "capture_duration_ns": 5_000,
        "capture_allocated_delta_bytes": 10,
        "capture_reserved_delta_bytes": 20,
    }

    result = smoke.build_performance_evidence(
        (row,),
        cache_counts={
            "hits": 2,
            "misses": 2,
            "evictions": 0,
            "quarantines": 0,
        },
    )

    assert result["measurement_count"] == 2
    assert result["warmed_exact_graph_hits"][
        "measurement_count"
    ] == 2
    assert result["warmed_exact_graph_hits"][
        "latency_ns_by_family"
    ]["b1-q1-w1"] == [1_000, 1_100]
    assert result["mixed_hit_rate"]["hit_count"] == 2
    assert result["mixed_hit_rate"]["measurement_count"] == 4


def test_run_request_batch_records_real_request_metrics():
    smoke = _load_smoke()

    class ModelRunner:
        @staticmethod
        def memory_snapshot():
            return {
                "cuda_allocated_bytes": 123,
                "cuda_reserved_bytes": 456,
            }

    class Engine:
        def __init__(self):
            self.model_runner = ModelRunner()
            self.last_step_observation = None
            self.step_index = 0
            self.requests = []

        def add_request(self, prompt, sampling_params):
            self.requests.append((prompt, sampling_params))

        def is_finished(self):
            return self.step_index >= 2

        def step(self):
            self.step_index += 1
            if self.step_index == 1:
                self.last_step_observation = {
                    "new_completion_tokens_by_seq": {
                        0: [21],
                    },
                    "speculative_selected_seq_ids": [],
                }
                return [], 1
            self.last_step_observation = {
                "new_completion_tokens_by_seq": {
                    0: [22],
                },
                "speculative_selected_seq_ids": [0],
            }
            return [(0, [21, 22])], -1

    clock_values = iter((1_000, 1_100, 1_400))

    outputs, observation, metrics = smoke._run_request_batch(
        Engine(),
        prompts=((11, 12),),
        max_tokens=2,
        clock_ns=lambda: next(clock_values),
        sampling_params_factory=lambda **kwargs: kwargs,
    )

    assert outputs == [[21, 22]]
    assert observation["speculative_selected_seq_ids"] == [0]
    assert metrics == {
        "elapsed_ns": 400,
        "ttft_ns": 100,
        "output_token_count": 2,
        "gpu_allocated_bytes": 123,
        "gpu_reserved_bytes": 456,
    }


def test_parse_args_enables_performance_measurement(tmp_path):
    smoke = _load_smoke()

    arguments = smoke.parse_args([
        "--model",
        "/models/Qwen3-0.6B",
        "--output-json",
        str(tmp_path / "artifact.json"),
        "--measure-performance",
    ])

    assert arguments.measure_performance is True


def test_source_inventory_binds_remote_authority_runner():
    smoke = _load_smoke()

    assert (
        "tools/run_spec_verify_cuda_graph_gate_remote.py"
        in smoke.SOURCE_FILES
    )
