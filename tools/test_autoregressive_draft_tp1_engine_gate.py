from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.autoregressive_draft_tp1_engine_gate import (
    _allocator_evidence_delta,
    _owned_proposal_entry_count,
    load_prompt_file,
    run_gate,
    run_preflight,
    validate_gate_payload,
)


def _identity_provider(target_model, draft_model):
    del target_model, draft_model
    checkpoint = {
        "target": {
            "model_path": "/models/target",
            "config_sha256": "target-config",
            "shard_sha256": [["model.safetensors", "target-shard"]],
            "composite_sha256": "target-composite",
        },
        "draft": {
            "model_path": "/models/draft",
            "config_sha256": "draft-config",
            "shard_sha256": [["model.safetensors", "draft-shard"]],
            "composite_sha256": "draft-composite",
        },
    }
    tokenizer = {
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
    }
    return checkpoint, tokenizer


class _FakeEngine:

    def __init__(self, mode, proposal_kv_configuration=None):
        self.mode = mode
        self.calls = []
        self.proposal_kv_configuration = proposal_kv_configuration

    def run_case(self, prompts, *, max_output_tokens):
        self.calls.append(tuple(prompts))
        outputs = [
            [prompt[0], 70 + index, 80 + index][
                :max_output_tokens
            ]
            for index, prompt in enumerate(prompts)
        ]
        if self.mode == "target":
            return {
                "output_token_ids": outputs,
                "observations": [],
                "timing_ms": {
                    "generation": 4.0,
                },
                "evidence": {},
            }
        proposal_kv = self.proposal_kv_configuration
        offload_enabled = proposal_kv["offload_enabled"]
        return {
            "output_token_ids": outputs,
            "observations": [
                {
                    "proposal_token_ids": [
                        70 + index,
                        80 + index,
                    ],
                    "accepted_prefix_count": 2,
                }
                for index, _ in enumerate(prompts)
            ],
            "timing_ms": {
                "bootstrap": 1.0,
                "proposal": 2.0,
                "verification": 3.0,
                "finalize": 1.0,
                "generation": 7.0,
            },
            "evidence": {
                "real_draft_forward_count": len(prompts) + 1,
                "extra_target_forward_count": 0,
                "proposal_kv_bytes": 4096,
                "target_kv_bytes": 8192,
                "proposal_kv_storage_id": "proposal-storage",
                "target_kv_storage_id": "target-storage",
                "proposal_kv_live_slots_before_release": len(prompts),
                "proposal_kv_live_slots_after_release": 0,
                "allocator_mode": proposal_kv["allocator_mode"],
                "logical_entry_capacity": proposal_kv[
                    "logical_entry_capacity"
                ],
                "gpu_slot_capacity": proposal_kv[
                    "gpu_slot_capacity"
                ],
                "h2d_operation_count": (
                    len(prompts) if offload_enabled else 0
                ),
                "h2d_entry_count": (
                    len(prompts) * 2 if offload_enabled else 0
                ),
                "h2d_bytes": (
                    len(prompts) * 128 if offload_enabled else 0
                ),
                "d2h_operation_count": (
                    len(prompts) if offload_enabled else 0
                ),
                "d2h_entry_count": (
                    len(prompts) if offload_enabled else 0
                ),
                "d2h_bytes": (
                    len(prompts) * 64 if offload_enabled else 0
                ),
                "accepted_entry_copy_count": 0,
                "accepted_entry_replay_count": 0,
                "accepted_entry_rematerialization_count": 0,
            },
        }

    def close(self):
        return None


_ENGINE_FACTORY_CALLS = []
_ENGINE_LIFECYCLE_EVENTS = []


def _engine_factory(mode, **kwargs):
    _ENGINE_FACTORY_CALLS.append((mode, dict(kwargs)))
    _ENGINE_LIFECYCLE_EVENTS.append(f"create:{mode}")
    engine = _FakeEngine(
        mode,
        kwargs.get("proposal_kv_configuration"),
    )
    engine.close = lambda: _ENGINE_LIFECYCLE_EVENTS.append(
        f"close:{mode}"
    )
    return engine


def _payload():
    _ENGINE_FACTORY_CALLS.clear()
    _ENGINE_LIFECYCLE_EVENTS.clear()
    return run_gate(
        target_model="/models/target",
        draft_model="/models/draft",
        prompts=((11,), (12,), (13,), (14,)),
        max_output_tokens=3,
        engine_factory=_engine_factory,
        identity_provider=_identity_provider,
    )


def _offload_payload():
    _ENGINE_FACTORY_CALLS.clear()
    _ENGINE_LIFECYCLE_EVENTS.clear()
    return run_gate(
        target_model="/models/target",
        draft_model="/models/draft",
        prompts=((11,), (12,), (13,), (14,)),
        max_output_tokens=3,
        proposal_kv_offload_enabled=True,
        proposal_kv_gpu_slot_capacity=8,
        engine_factory=_engine_factory,
        identity_provider=_identity_provider,
    )


def _allocator_snapshot(
    *,
    mode,
    logical_capacity,
    gpu_capacity,
    counter,
):
    return {
        "proposal_kv_cache": {
            "entry_allocator": {
                "allocator_mode": mode,
                "logical_entry_capacity": logical_capacity,
                "gpu_slot_capacity": gpu_capacity,
                "h2d_operation_count": counter,
                "h2d_entry_count": counter * 2,
                "h2d_bytes": counter * 128,
                "d2h_operation_count": counter,
                "d2h_entry_count": counter,
                "d2h_bytes": counter * 64,
                "accepted_entry_copy_count": 0,
                "accepted_entry_replay_count": 0,
                "accepted_entry_rematerialization_count": 0,
            },
        },
    }


def test_allocator_evidence_delta_uses_nested_authority_snapshot():
    before = _allocator_snapshot(
        mode="residency",
        logical_capacity=32,
        gpu_capacity=8,
        counter=2,
    )
    after = _allocator_snapshot(
        mode="residency",
        logical_capacity=32,
        gpu_capacity=8,
        counter=5,
    )

    assert _allocator_evidence_delta(before, after) == {
        "allocator_mode": "residency",
        "logical_entry_capacity": 32,
        "gpu_slot_capacity": 8,
        "h2d_operation_count": 3,
        "h2d_entry_count": 6,
        "h2d_bytes": 384,
        "d2h_operation_count": 3,
        "d2h_entry_count": 3,
        "d2h_bytes": 192,
        "accepted_entry_copy_count": 0,
        "accepted_entry_replay_count": 0,
        "accepted_entry_rematerialization_count": 0,
    }


def test_allocator_evidence_delta_rejects_counter_regression():
    before = _allocator_snapshot(
        mode="residency",
        logical_capacity=32,
        gpu_capacity=8,
        counter=5,
    )
    after = _allocator_snapshot(
        mode="residency",
        logical_capacity=32,
        gpu_capacity=8,
        counter=2,
    )

    with pytest.raises(ValueError, match="counter regressed"):
        _allocator_evidence_delta(before, after)


def test_owned_proposal_entry_count_uses_current_cache_snapshot_field():
    snapshot = {
        "proposal_kv_cache": {
            "owned_entry_count": 7,
        },
    }

    assert _owned_proposal_entry_count(snapshot) == 7


def test_owned_proposal_entry_count_rejects_stale_slot_field():
    snapshot = {
        "proposal_kv_cache": {
            "owned_slot_count": 7,
        },
    }

    with pytest.raises(ValueError, match="owned entry count"):
        _owned_proposal_entry_count(snapshot)


def test_gate_merges_offload_movement_across_batch_cases():
    payload = _offload_payload()

    assert payload["evidence"]["allocator_mode"] == "residency"
    assert payload["evidence"]["logical_entry_capacity"] == 32
    assert payload["evidence"]["gpu_slot_capacity"] == 8
    assert payload["evidence"]["h2d_operation_count"] == 5
    assert payload["evidence"]["h2d_entry_count"] == 10
    assert payload["evidence"]["h2d_bytes"] == 640
    assert payload["evidence"]["d2h_operation_count"] == 5
    assert payload["evidence"]["d2h_entry_count"] == 5
    assert payload["evidence"]["d2h_bytes"] == 320
    assert payload["evidence"]["accepted_entry_copy_count"] == 0
    assert payload["evidence"]["accepted_entry_replay_count"] == 0
    assert (
        payload["evidence"][
            "accepted_entry_rematerialization_count"
        ]
        == 0
    )


def test_baseline_engine_is_closed_before_learned_engine_loads():
    _payload()

    assert _ENGINE_LIFECYCLE_EVENTS == [
        "create:target",
        "close:target",
        "create:learned",
        "close:learned",
    ]


def test_preflight_validates_inputs_without_creating_engine():
    payload = run_preflight(
        target_model="/models/target",
        draft_model="/models/draft",
        prompts=((11,), (12,), (13,), (14,)),
        max_output_tokens=3,
        identity_provider=_identity_provider,
    )

    assert payload["gate"] == "autoregressive_draft_tp1_preflight"
    assert payload["input_ready"] is True
    assert payload["correctness_established"] is False
    assert payload["workload"] == {
        "prompt_count": 4,
        "batch_1_prompt_lengths": [1],
        "batch_4_prompt_lengths": [1, 1, 1, 1],
        "max_output_tokens": 3,
        "max_model_len": 7,
        "max_num_batched_tokens": 16,
        "proposal_slot_capacity": 32,
    }


def test_preflight_records_default_direct_proposal_kv_configuration():
    payload = run_preflight(
        target_model="/models/target",
        draft_model="/models/draft",
        prompts=((11,), (12,), (13,), (14,)),
        max_output_tokens=3,
        identity_provider=_identity_provider,
    )

    assert payload["proposal_kv"] == {
        "offload_enabled": False,
        "allocator_mode": "direct",
        "logical_entry_capacity": 32,
        "gpu_slot_capacity": 32,
        "cpu_backing_capacity": 0,
        "async_copy": True,
        "batch_copy": True,
    }


def test_offload_configuration_reaches_only_learned_engine():
    _ENGINE_FACTORY_CALLS.clear()
    _ENGINE_LIFECYCLE_EVENTS.clear()

    payload = run_gate(
        target_model="/models/target",
        draft_model="/models/draft",
        prompts=((11,), (12,), (13,), (14,)),
        max_output_tokens=3,
        proposal_kv_offload_enabled=True,
        proposal_kv_gpu_slot_capacity=8,
        proposal_kv_async_copy=False,
        proposal_kv_batch_copy=False,
        engine_factory=_engine_factory,
        identity_provider=_identity_provider,
    )

    target_call, learned_call = _ENGINE_FACTORY_CALLS
    assert target_call[1]["proposal_kv_configuration"] is None
    assert learned_call[1]["proposal_kv_configuration"] == {
        "offload_enabled": True,
        "allocator_mode": "residency",
        "logical_entry_capacity": 32,
        "gpu_slot_capacity": 8,
        "cpu_backing_capacity": 32,
        "async_copy": False,
        "batch_copy": False,
    }
    assert payload["proposal_kv"] == learned_call[1][
        "proposal_kv_configuration"
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (
            {"proposal_kv_offload_enabled": True},
            "GPU slot capacity is required",
        ),
        (
            {
                "proposal_kv_offload_enabled": True,
                "proposal_kv_gpu_slot_capacity": 32,
            },
            "must be smaller",
        ),
        (
            {"proposal_kv_gpu_slot_capacity": 8},
            "must be omitted",
        ),
        (
            {"proposal_kv_async_copy": 1},
            "async copy",
        ),
        (
            {"proposal_kv_batch_copy": 1},
            "batch copy",
        ),
    ),
)
def test_preflight_rejects_invalid_proposal_kv_configuration(
    kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        run_preflight(
            target_model="/models/target",
            draft_model="/models/draft",
            prompts=((11,), (12,), (13,), (14,)),
            max_output_tokens=3,
            identity_provider=_identity_provider,
            **kwargs,
        )


def test_gate_derives_engine_capacity_from_batch_four_workload():
    _payload()

    target_call, learned_call = _ENGINE_FACTORY_CALLS
    assert target_call[0] == "target"
    assert learned_call[0] == "learned"
    for _, kwargs in (target_call, learned_call):
        assert kwargs["max_num_seqs"] == 4
        assert kwargs["max_model_len"] == 7
        assert kwargs["max_num_batched_tokens"] == 16
        assert kwargs["proposal_slot_capacity"] == 32


def test_gate_records_exact_tp1_parity_and_physical_evidence():
    payload = _payload()

    assert payload["schema_version"] == 2
    assert payload["gate"] == "autoregressive_draft_tp1_engine"
    assert payload["configuration"] == {
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "temperature": 0.0,
        "max_proposal_tokens": 4,
    }
    assert payload["checkpoint_identity"]["draft"][
        "composite_sha256"
    ]
    assert payload["checkpoint_identity"]["target"][
        "composite_sha256"
    ]
    assert payload["tokenizer_contract"]["compatible"] is True
    assert payload["cases"]["batch_1"]["exact_output_parity"] is True
    assert payload["cases"]["batch_4"]["exact_output_parity"] is True
    assert payload["evidence"]["extra_target_forward_count"] == 0
    assert (
        payload["evidence"][
            "proposal_kv_live_slots_after_release"
        ]
        == 0
    )
    assert payload["evidence"]["real_draft_forward_count"] > 0
    assert payload["evidence"]["proposal_kv_bytes"] > 0
    assert payload["evidence"]["target_kv_bytes"] > 0
    assert payload["performance_pass_criterion"] is False
    assert payload["proposal_kv_offload_enabled"] is False
    assert (
        payload["real_proposal_kv_bidirectional_movement"]
        is False
    )
    assert payload["gate_pass"] is True


def test_offload_gate_requires_positive_h2d_and_d2h_for_movement():
    payload = _offload_payload()

    assert payload["proposal_kv_offload_enabled"] is True
    assert (
        payload["real_proposal_kv_bidirectional_movement"]
        is True
    )
    assert payload["gate_pass"] is True


def test_offload_parity_can_pass_without_bidirectional_movement():
    payload = _offload_payload()
    for name in (
        "h2d_operation_count",
        "h2d_entry_count",
        "h2d_bytes",
        "d2h_operation_count",
        "d2h_entry_count",
        "d2h_bytes",
    ):
        payload["evidence"][name] = 0
    payload["real_proposal_kv_bidirectional_movement"] = False

    validate_gate_payload(payload)

    assert payload["gate_pass"] is True
    assert (
        payload["real_proposal_kv_bidirectional_movement"]
        is False
    )


def test_tp1_gate_does_not_claim_tp4_or_phase_1_completion():
    payload = _payload()

    assert payload["gate"] == "autoregressive_draft_tp1_engine"
    assert payload["configuration"]["tensor_parallel_size"] == 1
    assert payload["cases"]["batch_1"]["exact_output_parity"]
    assert payload["cases"]["batch_4"]["exact_output_parity"]
    assert payload["evidence"]["proposal_kv_bytes"] > 0
    assert (
        payload["evidence"]["proposal_kv_storage_id"]
        != payload["evidence"]["target_kv_storage_id"]
    )
    assert payload.get("tp4_contract_established", False) is False
    assert payload.get("phase_1_achieved", False) is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda row: row["cases"]["batch_4"].update(
                learned_output_token_ids=[[999]]
            ),
            "exact output parity",
        ),
        (
            lambda row: row["checkpoint_identity"]["draft"].update(
                composite_sha256=""
            ),
            "checkpoint",
        ),
        (
            lambda row: row["tokenizer_contract"].update(
                compatible=False
            ),
            "tokenizer",
        ),
        (
            lambda row: row["evidence"].update(
                real_draft_forward_count=0
            ),
            "draft forward",
        ),
        (
            lambda row: row["evidence"].update(
                extra_target_forward_count=1
            ),
            "extra target forward",
        ),
        (
            lambda row: row["evidence"].update(
                proposal_kv_live_slots_after_release=1
            ),
            "proposal KV leak",
        ),
        (
            lambda row: row["cases"]["batch_1"].update(
                acceptance_rows=[]
            ),
            "acceptance rows",
        ),
        (
            lambda row: row["evidence"].update(
                proposal_kv_storage_id="same",
                target_kv_storage_id="same",
            ),
            "distinct storage",
        ),
        (
            lambda row: row["evidence"].update(
                accepted_entry_replay_count=1,
            ),
            "accepted proposal KV replay",
        ),
        (
            lambda row: row["evidence"].update(
                accepted_entry_rematerialization_count=1,
            ),
            "accepted proposal KV rematerialization",
        ),
        (
            lambda row: row["evidence"].update(
                allocator_mode="residency",
            ),
            "allocator mode",
        ),
        (
            lambda row: row.update(
                real_proposal_kv_bidirectional_movement=True,
            ),
            "movement classification",
        ),
    ),
)
def test_gate_rejects_missing_correctness_or_evidence(
    mutate,
    message,
):
    payload = deepcopy(_payload())
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        validate_gate_payload(payload)


def test_prompt_file_contract_is_json_serializable(tmp_path):
    payload = _payload()
    output = tmp_path / "gate.json"

    output.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )

    assert json.loads(output.read_text())["gate_pass"] is True


def test_prompt_bank_uses_existing_token_ids_without_tokenizer(
    tmp_path,
):
    path = tmp_path / "prompt_bank.json"
    path.write_text(json.dumps({
        "prompts": [
            {"prompt_token_ids": [11, 12]},
            {"token_ids": [21]},
            [31, 32],
            [41],
        ],
    }))
    tokenizer_calls = []

    prompts = load_prompt_file(
        path,
        target_model="/models/target",
        tokenizer_loader=lambda path: tokenizer_calls.append(path),
    )

    assert prompts == ((11, 12), (21,), (31, 32), (41,))
    assert tokenizer_calls == []


def test_text_target_bank_is_encoded_with_target_tokenizer(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({
        "targets": [
            {"prompt": "alpha"},
            {"text": "beta"},
            "gamma",
            {"prompt": "delta"},
        ],
    }))

    class Tokenizer:
        def encode(self, text):
            return [len(text), ord(text[0])]

    tokenizer_paths = []

    prompts = load_prompt_file(
        path,
        target_model="/models/target",
        tokenizer_loader=lambda model_path: (
            tokenizer_paths.append(model_path) or Tokenizer()
        ),
    )

    assert prompts == (
        (5, ord("a")),
        (4, ord("b")),
        (5, ord("g")),
        (5, ord("d")),
    )
    assert tokenizer_paths == ["/models/target"]
