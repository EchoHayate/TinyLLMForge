from __future__ import annotations

import pytest

from tools.cross_engine_k8_worker import (
    EngineResult,
    discover_vllm_public_multi_step,
    run_worker,
)
from tools.cross_engine_k8_workload import REQUIRED_ARMS


class FakeAdapter:
    def __init__(self, token_offset=0):
        self.token_offset = token_offset
        self.calls = []

    def run_case(self, case, arm):
        self.calls.append((case["context"], arm))
        output_tokens = case["output_tokens"]
        token_ids = list(range(output_tokens))
        if self.token_offset:
            token_ids[-1] += self.token_offset
        return EngineResult(
            token_ids=token_ids,
            token_timestamps_ns=list(range(10, 10 + output_tokens)),
            request_start_ns=0,
            request_end_ns=10 + output_tokens,
            engine_metrics={"graph_replays": 16},
            retained_logits={},
            resource_summary={
                "peak_gpu_memory_bytes": 1_000,
                "peak_rss_bytes": 2_000,
            },
        )

    def close(self):
        self.calls.append(("close", "close"))


class FakeSampler:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True
        return {
            "peak_gpu_memory_bytes": 9_000,
            "peak_rss_bytes": 8_000,
        }


def _worker_plan(arm="tinyllmforge_host_greedy"):
    cases = [
        {
            "context": context,
            "prompt_token_ids": list(range(prompt_tokens)),
            "prompt_tokens": prompt_tokens,
            "output_tokens": 128,
        }
        for context, prompt_tokens in (
            ("short", 256),
            ("medium", 2048),
            ("long", 8192),
        )
    ]
    return {
        "schema_version": "cross-engine-k8.worker-plan.v1",
        "run_tag": "20260829-cross-engine-k8-qwen3-06b-r1",
        "source_revision": "a" * 40,
        "arm": arm,
        "gpu_uuid": "GPU-test",
        "repetition": 0,
        "warmups": 2,
        "cases": cases,
        "expected_tokens": {
            case["context"]: list(range(128))
            for case in cases
        },
    }


@pytest.mark.parametrize("arm", REQUIRED_ARMS)
def test_worker_emits_one_terminal_receipt_per_arm(arm):
    adapter = FakeAdapter()

    result = run_worker(
        _worker_plan(arm),
        adapter_factory=lambda _plan: adapter,
        sampler_factory=FakeSampler,
    )

    assert result["terminal"] is True
    assert result["arm"] == arm
    assert result["measured_rows"] == 3
    assert result["correctness_valid"] is True
    assert result["performance_eligible"] is True
    assert adapter.calls[-1] == ("close", "close")


def test_worker_runs_two_warmups_then_three_measured_cases():
    adapter = FakeAdapter()

    run_worker(
        _worker_plan(),
        adapter_factory=lambda _plan: adapter,
        sampler_factory=FakeSampler,
    )

    assert len(adapter.calls[:-1]) == 5
    assert adapter.calls[:2] == [
        ("short", "tinyllmforge_host_greedy"),
        ("short", "tinyllmforge_host_greedy"),
    ]


def test_worker_rejects_engine_tokenizer_substitution():
    plan = _worker_plan()
    plan["cases"][0].pop("prompt_token_ids")

    with pytest.raises(ValueError, match="prompt_token_ids"):
        run_worker(
            plan,
            adapter_factory=lambda _plan: FakeAdapter(),
            sampler_factory=FakeSampler,
        )


def test_cross_engine_mismatch_excludes_performance():
    result = run_worker(
        _worker_plan("vllm_default_greedy"),
        adapter_factory=lambda _plan: FakeAdapter(token_offset=1),
        sampler_factory=FakeSampler,
    )

    assert result["correctness_valid"] is False
    assert result["performance_eligible"] is False
    assert result["case_rows"][0]["performance_eligible"] is False


def test_worker_merges_external_resource_peaks_into_each_case():
    samplers = []

    def sampler_factory():
        sampler = FakeSampler()
        samplers.append(sampler)
        return sampler

    result = run_worker(
        _worker_plan(),
        adapter_factory=lambda _plan: FakeAdapter(),
        sampler_factory=sampler_factory,
    )

    assert len(samplers) == 3
    assert all(sampler.started and sampler.stopped for sampler in samplers)
    assert all(
        row["peak_gpu_memory_bytes"] == 9_000
        and row["peak_rss_bytes"] == 8_000
        for row in result["case_rows"]
    )


def test_engine_result_rejects_nonmonotonic_timestamps():
    result = EngineResult(
        token_ids=[1, 2],
        token_timestamps_ns=[10, 9],
        request_start_ns=0,
        request_end_ns=20,
        engine_metrics={},
        retained_logits={},
        resource_summary={},
    )

    with pytest.raises(ValueError, match="monotonic"):
        result.validate(expected_output_tokens=2)


def test_vllm_multi_step_discovery_uses_only_public_config():
    class EngineArgs:
        __annotations__ = {"num_scheduler_steps": int}

    result = discover_vllm_public_multi_step(EngineArgs)

    assert result == {
        "available": True,
        "parameter": "num_scheduler_steps",
        "value": 8,
    }


def test_vllm_multi_step_discovery_does_not_invent_private_control():
    class EngineArgs:
        __annotations__ = {"_private_multi_step": int}

    result = discover_vllm_public_multi_step(EngineArgs)

    assert result == {
        "available": False,
        "parameter": None,
        "value": None,
    }
