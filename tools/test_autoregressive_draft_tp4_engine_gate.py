from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

import tools.autoregressive_draft_tp4_engine_gate as gate_module
from tools.autoregressive_draft_tp4_engine_gate import (
    _TinyVLLMTP4EngineAdapter,
    _validate_gpu_indices,
    distributed_environment,
    load_prompt_file,
    main,
    run_gate,
    validate_gate_payload,
)


def _identity_provider(target_model, draft_model):
    del target_model, draft_model
    return (
        {
            "target": {
                "composite_sha256": "target-checkpoint",
            },
            "draft": {
                "composite_sha256": "draft-checkpoint",
            },
        },
        {
            "compatible": True,
            "target": {
                "ordered_token_to_id_sha256": "mapping",
                "artifact_sha256": [],
                "composite_sha256": "target-tokenizer",
            },
            "draft": {
                "ordered_token_to_id_sha256": "mapping",
                "artifact_sha256": [],
                "composite_sha256": "draft-tokenizer",
            },
        },
    )


def _rank_snapshot(rank):
    logical_rows = (
        {
            "stage": "proposal_materialized",
            "rows": ({
                "sequence_id": 0,
                "sequence_epoch": 0,
                "exact_q": 4,
                "proposal_token_ids": (71, 81),
                "logical_state": "materialized",
            },),
        },
        {
            "stage": "release_complete",
            "rows": ({
                "sequence_id": 0,
                "sequence_epoch": 0,
                "active_transaction_count": 0,
                "active_ticket_count": 0,
                "committed_logical_entries": 0,
                "live_local_slot_count": 0,
            },),
        },
    )
    return {
        "rank": rank,
        "world_size": 4,
        "registered": True,
        "registration_consensus_sha256": "a" * 64,
        "checkpoint_identity": {
            "target": {"composite_sha256": "b" * 64},
            "draft": {"composite_sha256": "c" * 64},
        },
        "tokenizer_contract": {
            "target": {"composite_sha256": "d" * 64},
            "draft": {"composite_sha256": "e" * 64},
        },
        "executor_descriptor": {
            "executor_id": "autoregressive-draft",
            "capabilities": {
                "source_type": "independent_draft_model",
                "supports_batch": True,
                "requires_target_hidden": False,
                "requires_target_logits": False,
                "max_proposal_tokens": 4,
                "execution_domain": "model_runner",
                "requires_proposal_lifecycle": True,
                "requires_full_token_history": False,
            },
        },
        "registration_error": None,
        "executor": {
            "rank": rank,
            "world_size": 4,
            "backend_identity": "qwen3",
            "logical_authority_rows": logical_rows,
            "logical_authority_digest_count": len(logical_rows),
            "last_logical_authority_sha256": "f" * 64,
            "timing_ms": {
                "prompt_bootstrap": 1.0 + rank,
                "proposal_forward": 2.0 + rank,
                "proposal_finalize": 3.0 + rank,
            },
            "proposal_kv_lifecycle": {
                "active_transaction_count": 0,
                "prepared_ticket_count": 0,
                "proposal_kv_cache": {
                    "owned_entry_count": 0,
                },
            },
            "backend": {
                "backend_identity": "qwen3",
                "local_proposal_kv_bytes": 1024 + rank,
                "local_prefill_forward_count": 1,
                "local_decode_forward_count": 3,
                "proposal_kv_storage_id": f"rank-{rank}-store",
                "proposal_kv_cache": {
                    "entry_allocator": {
                        "allocator_mode": "direct",
                        "accepted_entry_copy_count": 0,
                        "accepted_entry_replay_count": 0,
                        "accepted_entry_rematerialization_count": 0,
                    },
                },
                "physical_store": {
                    "allocated_slot_count": 0,
                },
            },
        },
    }


class _FakeEngine:

    def __init__(self, mode, *, mismatch=False, invalid_rank=False):
        self.mode = mode
        self.mismatch = mismatch
        self.invalid_rank = invalid_rank

    def run_case(self, prompts, *, max_output_tokens):
        outputs = [
            [70 + index, 80 + index][:max_output_tokens]
            for index, _ in enumerate(prompts)
        ]
        if self.mode == "target":
            return {
                "output_token_ids": outputs,
            }
        if self.mismatch:
            outputs[0] = [999]
        snapshots = tuple(
            _rank_snapshot(rank) for rank in range(4)
        )
        if self.invalid_rank:
            snapshots = tuple(
                {
                    **snapshot,
                    "registration_consensus_sha256": (
                        "mismatch"
                        if snapshot["rank"] == 2
                        else snapshot[
                            "registration_consensus_sha256"
                        ]
                    ),
                }
                for snapshot in snapshots
            )
        return {
            "output_token_ids": outputs,
            "acceptance_rows": [
                {
                    "event_index": index,
                    "step_index": 0,
                    "sequence_id": index,
                    "prompt_index": index,
                    "prompt_token_ids": list(prompts[index]),
                    "output_token_count_before_step": 0,
                    "proposal_token_ids": [70 + index, 80 + index],
                    "accepted_prefix_count": 2,
                    "accepted_prefix_token_ids": [
                        70 + index,
                        80 + index,
                    ],
                }
                for index, _ in enumerate(prompts)
            ],
            "rank_snapshots": snapshots,
        }

    def close(self):
        return None


_FACTORY_CALLS = []
_LIFECYCLE = []
_MISMATCH = False
_INVALID_RANK = False


def _engine_factory(mode, **kwargs):
    _FACTORY_CALLS.append((mode, dict(kwargs)))
    _LIFECYCLE.append(f"create:{mode}")
    engine = _FakeEngine(
        mode,
        mismatch=_MISMATCH and mode == "learned",
        invalid_rank=_INVALID_RANK and mode == "learned",
    )
    engine.close = lambda: _LIFECYCLE.append(f"close:{mode}")
    return engine


def _payload():
    global _MISMATCH, _INVALID_RANK
    _MISMATCH = False
    _INVALID_RANK = False
    _FACTORY_CALLS.clear()
    _LIFECYCLE.clear()
    return run_gate(
        target_model="/models/target",
        draft_model="/models/draft",
        prompts=((11,), (12,), (13,), (14,)),
        max_output_tokens=2,
        engine_factory=_engine_factory,
        identity_provider=_identity_provider,
    )


def test_gate_closes_target_before_loading_learned_engine():
    _payload()

    assert _LIFECYCLE == [
        "create:target",
        "close:target",
        "create:learned",
        "close:learned",
    ]


def test_gate_passes_tp4_direct_configuration_to_both_engines():
    payload = _payload()

    target_call, learned_call = _FACTORY_CALLS
    for mode, kwargs in (target_call, learned_call):
        assert kwargs["tensor_parallel_size"] == 4
        assert kwargs["max_num_seqs"] == 4
        assert kwargs["max_model_len"] == 6
        assert kwargs["max_num_batched_tokens"] == 16
        assert kwargs["proposal_slot_capacity"] == 28
        assert kwargs["learned_enabled"] is (mode == "learned")
    assert payload["configuration"] == {
        "tensor_parallel_size": 4,
        "allocator_mode": "direct",
        "dtype": "bfloat16",
        "temperature": 0.0,
        "max_proposal_tokens": 4,
    }


def test_gate_records_batch_parity_acceptance_and_rank_authority():
    payload = _payload()

    assert payload["schema_version"] == 2
    assert payload["gate"] == "autoregressive_draft_tp4_engine"
    for name, count in (("batch_1", 1), ("batch_4", 4)):
        case = payload["cases"][name]
        assert case["exact_output_parity"] is True
        assert len(case["acceptance_rows"]) == count
        for index, row in enumerate(case["acceptance_rows"]):
            assert row == {
                "event_index": index,
                "step_index": 0,
                "sequence_id": index,
                "prompt_index": index,
                "prompt_token_ids": [11 + index],
                "output_token_count_before_step": 0,
                "proposal_token_ids": [
                    70 + index,
                    80 + index,
                ],
                "accepted_prefix_count": 2,
                "accepted_prefix_token_ids": [
                    70 + index,
                    80 + index,
                ],
            }
        assert len(case["rank_snapshots"]) == 4
        assert case["rank_summary"]["rank_count"] == 4
        assert case["rank_summary"]["classification"] == (
            "NOT_PROMOTABLE"
        )
        assert case["rank_summary"]["promotion_boundary"][
            "phase_1"
        ] == "NOT_ACHIEVED"
    assert payload["performance_pass_criterion"] is False
    assert payload["real_proposal_kv_movement"] is False
    assert payload["gate_pass"] is True


def test_gate_fails_closed_on_target_learned_output_mismatch():
    global _MISMATCH
    _MISMATCH = True
    _FACTORY_CALLS.clear()
    _LIFECYCLE.clear()

    with pytest.raises(ValueError, match="exact output parity"):
        run_gate(
            target_model="/models/target",
            draft_model="/models/draft",
            prompts=((11,), (12,), (13,), (14,)),
            max_output_tokens=2,
            engine_factory=_engine_factory,
            identity_provider=_identity_provider,
        )


def test_gate_fails_closed_on_invalid_rank_authority():
    global _INVALID_RANK
    _INVALID_RANK = True
    _FACTORY_CALLS.clear()
    _LIFECYCLE.clear()

    with pytest.raises(ValueError, match="registration consensus"):
        run_gate(
            target_model="/models/target",
            draft_model="/models/draft",
            prompts=((11,), (12,), (13,), (14,)),
            max_output_tokens=2,
            engine_factory=_engine_factory,
            identity_provider=_identity_provider,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda payload: payload.update(
                performance_pass_criterion=True
            ),
            "performance",
        ),
        (
            lambda payload: payload.update(
                real_proposal_kv_movement=True
            ),
            "movement",
        ),
        (
            lambda payload: payload["cases"]["batch_1"].update(
                acceptance_rows=[]
            ),
            "acceptance",
        ),
        (
            lambda payload: payload["configuration"].update(
                allocator_mode="residency"
            ),
            "configuration",
        ),
        (
            lambda payload: payload["cases"]["batch_1"][
                "acceptance_rows"
            ][0].update(
                accepted_prefix_token_ids=[999]
            ),
            "accepted prefix",
        ),
        (
            lambda payload: payload["cases"]["batch_4"][
                "acceptance_rows"
            ][1].update(
                event_index=9
            ),
            "event ordering",
        ),
    ),
)
def test_validate_gate_payload_rejects_overclaim_or_missing_evidence(
    mutate,
    message,
):
    payload = deepcopy(_payload())
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        validate_gate_payload(payload)


def test_validate_gate_payload_accepts_multiple_acceptance_events():
    payload = deepcopy(_payload())
    payload["cases"]["batch_1"]["acceptance_rows"].append({
        "event_index": 1,
        "step_index": 1,
        "sequence_id": 0,
        "prompt_index": 0,
        "prompt_token_ids": [11],
        "output_token_count_before_step": 2,
        "proposal_token_ids": [91],
        "accepted_prefix_count": 1,
        "accepted_prefix_token_ids": [91],
    })

    validate_gate_payload(payload)


@pytest.mark.parametrize(
    "gpu_indices",
    (
        (0, 1, 2),
        (0, 1, 2, 2),
        (0, 1, 2, -1),
        (0, 1, 2, True),
        [0, 1, 2, 3],
    ),
)
def test_gpu_indices_require_four_distinct_nonnegative_integers(
    gpu_indices,
):
    with pytest.raises(ValueError, match="four distinct"):
        _validate_gpu_indices(gpu_indices)


@pytest.mark.parametrize(
    ("dist_port", "master_port", "message"),
    (
        (0, 20002, "distributed port"),
        (20001, -1, "master port"),
        (True, 20002, "distributed port"),
    ),
)
def test_distributed_environment_rejects_invalid_ports(
    dist_port,
    master_port,
    message,
):
    with pytest.raises(ValueError, match=message):
        with distributed_environment(
            gpu_indices=(0, 1, 2, 3),
            dist_port=dist_port,
            master_port=master_port,
        ):
            pass


def test_distributed_environment_restores_previous_values(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.delenv("TINYVLLM_DIST_PORT", raising=False)
    monkeypatch.setenv("MASTER_PORT", "19999")

    with distributed_environment(
        gpu_indices=(0, 1, 2, 3),
        dist_port=20001,
        master_port=20002,
    ):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        assert os.environ["TINYVLLM_DIST_PORT"] == "20001"
        assert os.environ["MASTER_PORT"] == "20002"

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"
    assert "TINYVLLM_DIST_PORT" not in os.environ
    assert os.environ["MASTER_PORT"] == "19999"


class _FakeSamplingParams:

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeRuntime:

    def __init__(self, *, model_runner_executor):
        self.model_runner_executor = model_runner_executor


class _FakeProductionLLM:

    calls = []
    registration_error = None
    descriptor = {"executor_id": "autoregressive-draft"}

    def __init__(self, model, **kwargs):
        self.calls.append((model, dict(kwargs)))
        self.model_runner = type("ModelRunner", (), {
            "autoregressive_draft_executor_descriptor": (
                self.descriptor
                if kwargs["autoregressive_draft_enabled"]
                else None
            ),
            "autoregressive_draft_registration_error": (
                self.registration_error
            ),
        })()
        self.activated_runtime = None
        self.pending = []
        self.last_step_observation = None
        self.flush_timeouts = []
        self.snapshot_timeouts = []
        self.exit_called = False

    def activate_speculative_runtime(self, runtime):
        self.activated_runtime = runtime

    def add_request(self, token_ids, sampling_params):
        self.pending.append(
            (list(token_ids), sampling_params)
        )

    def is_finished(self):
        return not self.pending

    def step(self):
        rows = [
            (index, [70 + index, 80 + index])
            for index, _ in enumerate(self.pending)
        ]
        self.last_step_observation = {
            "speculative_proposal_token_ids_by_seq": {
                index: [70 + index, 80 + index]
                for index, _ in enumerate(self.pending)
            },
            "speculative_accepted_draft_token_counts": {
                index: 2
                for index, _ in enumerate(self.pending)
            },
            "speculative_accepted_draft_token_ids_by_seq": {
                index: [70 + index, 80 + index]
                for index, _ in enumerate(self.pending)
            },
            "new_completion_tokens_by_seq": {
                index: [70 + index, 80 + index]
                for index, _ in enumerate(self.pending)
            },
        }
        self.pending.clear()
        return rows, None

    def flush_pending_hybrid_state_releases(self, *, timeout_s):
        self.flush_timeouts.append(timeout_s)

    def autoregressive_draft_authority_snapshots(
        self,
        *,
        timeout_s,
    ):
        self.snapshot_timeouts.append(timeout_s)
        return tuple(_rank_snapshot(rank) for rank in range(4))

    def exit(self):
        self.exit_called = True


def _production_adapter(mode, **overrides):
    return _TinyVLLMTP4EngineAdapter(
        mode,
        target_model="/models/target",
        draft_model="/models/draft",
        tensor_parallel_size=4,
        max_num_seqs=4,
        max_model_len=6,
        max_num_batched_tokens=16,
        proposal_slot_capacity=28,
        learned_enabled=mode == "learned",
        llm_type=_FakeProductionLLM,
        sampling_params_type=_FakeSamplingParams,
        runtime_type=_FakeRuntime,
        **overrides,
    )


def test_production_adapter_passes_direct_tp4_engine_kwargs():
    _FakeProductionLLM.calls.clear()

    target = _production_adapter("target")
    learned = _production_adapter("learned")

    target_model, target_kwargs = _FakeProductionLLM.calls[0]
    learned_model, learned_kwargs = _FakeProductionLLM.calls[1]
    assert target_model == learned_model == "/models/target"
    assert target_kwargs == {
        "tensor_parallel_size": 4,
        "enforce_eager": True,
        "max_num_seqs": 4,
        "max_model_len": 6,
        "max_num_batched_tokens": 16,
        "autoregressive_draft_enabled": False,
        "autoregressive_draft_model": None,
        "autoregressive_draft_backend": "qwen3",
        "autoregressive_draft_max_proposal_tokens": 4,
        "autoregressive_draft_gpu_slot_capacity": 0,
        "autoregressive_draft_proposal_kv_offload_enabled": False,
        "autoregressive_draft_cuda_graphs": False,
        "autoregressive_draft_logical_entry_capacity": 0,
        "autoregressive_draft_cpu_backing_capacity": 0,
        "proposal_kv_async_copy": True,
        "proposal_kv_batch_copy": True,
    }
    assert learned_kwargs == {
        **target_kwargs,
        "autoregressive_draft_enabled": True,
        "autoregressive_draft_model": "/models/draft",
        "autoregressive_draft_gpu_slot_capacity": 28,
    }
    assert target.engine.activated_runtime is None
    assert isinstance(learned.engine.activated_runtime, _FakeRuntime)
    assert (
        learned.engine.activated_runtime.model_runner_executor
        == _FakeProductionLLM.descriptor
    )


def test_production_adapter_passes_explicit_graph_budget_overrides():
    _FakeProductionLLM.calls.clear()

    adapter = _production_adapter(
        "learned",
        cuda_graph_enabled=True,
        cuda_graph_max_reserved_bytes=4 * 1024 * 1024 * 1024,
        cuda_graph_max_total_capture_ns=120_000_000_000,
        cuda_graph_max_single_capture_ns=60_000_000_000,
    )

    _, kwargs = _FakeProductionLLM.calls[0]
    assert kwargs[
        "autoregressive_draft_cuda_graph_max_reserved_bytes"
    ] == 4 * 1024 * 1024 * 1024
    assert kwargs[
        "autoregressive_draft_cuda_graph_max_total_capture_ns"
    ] == 120_000_000_000
    assert kwargs[
        "autoregressive_draft_cuda_graph_max_single_capture_ns"
    ] == 60_000_000_000
    adapter.close()


def test_production_adapter_fails_closed_on_registration_error():
    original_descriptor = _FakeProductionLLM.descriptor
    original_error = _FakeProductionLLM.registration_error
    _FakeProductionLLM.descriptor = None
    _FakeProductionLLM.registration_error = {"stage": "build"}
    try:
        with pytest.raises(
            RuntimeError,
            match="registration failed",
        ):
            _production_adapter("learned")
    finally:
        _FakeProductionLLM.descriptor = original_descriptor
        _FakeProductionLLM.registration_error = original_error


def test_production_adapter_collects_acceptance_and_rank_authority():
    adapter = _production_adapter("learned")

    result = adapter.run_case(
        ((11,), (12,), (13,), (14,)),
        max_output_tokens=2,
    )

    assert result["output_token_ids"] == [
        [70, 80],
        [71, 81],
        [72, 82],
        [73, 83],
    ]
    assert result["acceptance_rows"] == [
        {
            "event_index": index,
            "step_index": 0,
            "sequence_id": index,
            "prompt_index": index,
            "prompt_token_ids": [11 + index],
            "output_token_count_before_step": 0,
            "proposal_token_ids": [70 + index, 80 + index],
            "accepted_prefix_count": 2,
            "accepted_prefix_token_ids": [
                70 + index,
                80 + index,
            ],
        }
        for index in range(4)
    ]
    assert len(result["rank_snapshots"]) == 4
    assert adapter.engine.flush_timeouts == [60.0]
    assert adapter.engine.snapshot_timeouts == [60.0]
    sampling_params = result["sampling_params"]
    assert sampling_params == {
        "temperature": 0.0,
        "max_tokens": 2,
        "ignore_eos": True,
    }


def test_prompt_file_uses_existing_token_ids_without_tokenizer(
    tmp_path,
):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({
        "prompts": [[11], [12], [13], [14]],
    }))
    tokenizer_calls = []

    prompts = load_prompt_file(
        path,
        target_model="/models/target",
        tokenizer_loader=lambda path: tokenizer_calls.append(path),
    )

    assert prompts == ((11,), (12,), (13,), (14,))
    assert tokenizer_calls == []


def test_main_runs_inside_restored_environment_and_writes_json(
    tmp_path,
    monkeypatch,
):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps({
        "prompts": [[11], [12], [13], [14]],
    }))
    output_path = tmp_path / "result.json"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.delenv("TINYVLLM_DIST_PORT", raising=False)
    monkeypatch.setenv("MASTER_PORT", "19999")
    calls = []

    def fake_run_gate(**kwargs):
        calls.append({
            "kwargs": kwargs,
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "CUDA_VISIBLE_DEVICES",
                    "TINYVLLM_DIST_PORT",
                    "MASTER_PORT",
                )
            },
        })
        return {"gate_pass": True}

    monkeypatch.setattr(gate_module, "run_gate", fake_run_gate)

    assert main([
        "--target-model", "/models/target",
        "--draft-model", "/models/draft",
        "--prompt-file", str(prompt_path),
        "--output", str(output_path),
        "--gpu-indices", "0,1,2,3",
        "--dist-port", "20001",
        "--master-port", "20002",
        "--max-output-tokens", "2",
    ]) == 0

    assert json.loads(output_path.read_text()) == {
        "gate_pass": True,
    }
    assert calls[0]["environment"] == {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "TINYVLLM_DIST_PORT": "20001",
        "MASTER_PORT": "20002",
    }
    assert calls[0]["kwargs"]["prompts"] == (
        (11,),
        (12,),
        (13,),
        (14,),
    )
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"
    assert "TINYVLLM_DIST_PORT" not in os.environ
    assert os.environ["MASTER_PORT"] == "19999"


def test_main_routes_output_directory_to_authority_bundle(
    tmp_path,
    monkeypatch,
):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps({
        "prompts": [[11], [12], [13], [14]],
    }))
    output_dir = tmp_path / "authority"
    payload = {"gate_pass": True}
    published = []
    monkeypatch.setattr(
        gate_module,
        "run_gate",
        lambda **kwargs: payload,
    )
    monkeypatch.setattr(
        gate_module,
        "publish_authority_bundle",
        lambda value, path: published.append(
            (value, Path(path))
        ) or {"classification": "PASS"},
    )

    assert main([
        "--target-model", "/models/target",
        "--draft-model", "/models/draft",
        "--prompt-file", str(prompt_path),
        "--output-dir", str(output_dir),
        "--gpu-indices", "0,1,2,3",
        "--dist-port", "20001",
        "--master-port", "20002",
    ]) == 0

    assert published == [(payload, output_dir)]
    assert not output_dir.exists()


def test_main_refuses_to_replace_existing_output(
    tmp_path,
    monkeypatch,
):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(json.dumps({
        "prompts": [[11], [12], [13], [14]],
    }))
    output_path = tmp_path / "result.json"
    output_path.write_text("keep-me")
    calls = []
    monkeypatch.setattr(
        gate_module,
        "run_gate",
        lambda **kwargs: calls.append(kwargs),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        main([
            "--target-model", "/models/target",
            "--draft-model", "/models/draft",
            "--prompt-file", str(prompt_path),
            "--output", str(output_path),
            "--gpu-indices", "0,1,2,3",
            "--dist-port", "20001",
            "--master-port", "20002",
        ])

    assert output_path.read_text() == "keep-me"
    assert calls == []
