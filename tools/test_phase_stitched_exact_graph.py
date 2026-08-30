from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "tinyvllm"
    / "engine"
    / "phase_stitched_exact_graph.py"
)
assert MODULE_PATH.is_file(), "phase-stitch contract module is missing"
SPEC = importlib.util.spec_from_file_location(
    "phase_stitched_exact_graph_under_test",
    MODULE_PATH,
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

PhaseStitchPrefixResult = module.PhaseStitchPrefixResult
PhaseStitchSuffixResult = module.PhaseStitchSuffixResult
PhaseStitchTransaction = module.PhaseStitchTransaction
build_phase_stitch_lease = module.build_phase_stitch_lease
decide_phase_stitch_admission = module.decide_phase_stitch_admission
validate_phase_stitch_prefix = module.validate_phase_stitch_prefix
validate_phase_stitch_suffix = module.validate_phase_stitch_suffix

CONFIG_PATH = REPO_ROOT / "tinyvllm" / "config.py"


def _load_config_class():
    module_name = "phase_stitched_config_under_test"
    fake_transformers = types.ModuleType("transformers")

    class FakeAutoConfig:

        @staticmethod
        def from_pretrained(model):
            del model
            return types.SimpleNamespace(
                max_position_embeddings=4096,
                num_hidden_layers=4,
            )

    fake_transformers.AutoConfig = FakeAutoConfig
    original = sys.modules.get("transformers")
    sys.modules["transformers"] = fake_transformers
    try:
        config_module = types.ModuleType(module_name)
        config_module.__file__ = os.fspath(CONFIG_PATH)
        sys.modules[module_name] = config_module
        source = CONFIG_PATH.read_text(encoding="utf-8")
        exec(
            compile(
                "from __future__ import annotations\n" + source,
                os.fspath(CONFIG_PATH),
                "exec",
            ),
            config_module.__dict__,
        )
        return config_module.Config
    finally:
        if original is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = original


def _lease(**overrides):
    values = {
        "sequence_id": 7,
        "sequence_generation": 3,
        "schedule_generation": 11,
        "prefill_graph_identity_sha256": "a" * 64,
        "prefill_graph_generation": 5,
        "decode_graph_identity_sha256": "b" * 64,
        "decode_graph_generation": 13,
        "prompt_token_count": 256,
        "final_prefill_first_position": 0,
        "final_prefill_last_position": 255,
        "initial_completion_count": 0,
        "remaining_output_tokens": 8,
        "decode_first_write_position": 256,
        "decode_last_write_position": 262,
        "decode_first_physical_slot": 1024,
        "decode_last_physical_slot": 1030,
        "block_table_identity": ((64, 9),),
        "completion_only": True,
        "source_identity_sha256": "c" * 64,
    }
    values.update(overrides)
    return build_phase_stitch_lease(**values)


def _admission(**overrides):
    values = {
        "enabled": True,
        "prefill_graph_available": True,
        "decode_graph_available": True,
        "prompt_token_count": 256,
        "prompt_token_allowlist": (256, 2048),
        "sequence_count": 1,
        "waiting_count": 0,
        "prefilling_count": 0,
        "do_sample": True,
        "temperatures": (0.0,),
        "ignore_eos": (True,),
        "completion_only": True,
        "remaining_output_tokens": 8,
        "decode_kv_capacity_tokens": 7,
        "tensor_parallel_size": 1,
        "rank": 0,
        "incompatible_modes": (),
        "pending_lease": False,
        "quarantined": False,
    }
    values.update(overrides)
    return decide_phase_stitch_admission(**values)


def test_phase_stitch_lease_binds_exact_k8_parent_transaction():
    lease = _lease()

    assert lease.parent_token_count == 8
    assert lease.authorized_decode_replay_count == 7
    assert lease.first_token_ordinal == 0
    assert lease.suffix_start_ordinal == 1
    assert len(lease.identity_sha256) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("sequence_generation", 4),
        ("schedule_generation", 12),
        ("prefill_graph_identity_sha256", "d" * 64),
        ("prefill_graph_generation", 6),
        ("decode_graph_identity_sha256", "e" * 64),
        ("decode_graph_generation", 14),
        ("block_table_identity", ((64, 10),)),
        ("remaining_output_tokens", 9),
        ("source_identity_sha256", "f" * 64),
    ),
)
def test_phase_stitch_lease_identity_binds_authoritative_fields(
    field,
    value,
):
    baseline = _lease()
    changed = _lease(**{field: value})

    assert changed.identity_sha256 != baseline.identity_sha256


def test_phase_stitch_lease_identity_binds_decode_physical_interval():
    baseline = _lease()
    changed = _lease(
        decode_first_physical_slot=2048,
        decode_last_physical_slot=2054,
    )

    assert changed.identity_sha256 != baseline.identity_sha256


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("sequence_id", True, "sequence_id"),
        (
            "prefill_graph_identity_sha256",
            "not-a-digest",
            "prefill_graph_identity_sha256",
        ),
        (
            "final_prefill_last_position",
            254,
            "prefill position interval",
        ),
        (
            "decode_last_write_position",
            263,
            "decode write interval",
        ),
        (
            "decode_last_physical_slot",
            1031,
            "decode physical interval",
        ),
        (
            "remaining_output_tokens",
            7,
            "remaining_output_tokens",
        ),
        ("completion_only", False, "completion-only"),
    ),
)
def test_phase_stitch_lease_rejects_invalid_authority(
    field,
    value,
    message,
):
    with pytest.raises(ValueError, match=message):
        _lease(**{field: value})


def test_phase_stitch_prefix_and_suffix_validate_parent_identity():
    lease = _lease()
    prefix = PhaseStitchPrefixResult(
        parent_lease_identity_sha256=lease.identity_sha256,
        token=101,
        token_ordinal=0,
        replay_count=0,
        d2h_calls=1,
        d2h_bytes=8,
    )
    suffix = PhaseStitchSuffixResult(
        parent_lease_identity_sha256=lease.identity_sha256,
        tokens=(102, 103, 104, 105, 106, 107, 108),
        first_token_ordinal=1,
        replay_count=7,
        d2h_calls=1,
        d2h_bytes=56,
    )

    assert validate_phase_stitch_prefix(lease, prefix) is prefix
    assert validate_phase_stitch_suffix(lease, suffix) is suffix

    with pytest.raises(ValueError, match="parent lease identity"):
        validate_phase_stitch_prefix(
            lease,
            replace(
                prefix,
                parent_lease_identity_sha256="d" * 64,
            ),
        )
    with pytest.raises(ValueError, match="parent lease identity"):
        validate_phase_stitch_suffix(
            lease,
            replace(
                suffix,
                parent_lease_identity_sha256="d" * 64,
            ),
        )


def test_phase_stitch_transaction_accepts_only_ordered_two_phase_commit():
    transaction = PhaseStitchTransaction(_lease())

    transaction.mark_replay_started()
    transaction.mark_prefix_ready()
    transaction.mark_prefix_committed()
    transaction.mark_suffix_ready(replay_count=7)
    transaction.mark_suffix_committed()
    transaction.close()

    assert transaction.state == "closed"
    assert transaction.last_authoritative_phase == "suffix_committed"
    assert transaction.completed_decode_replays == 7


def test_phase_stitch_transaction_rejects_duplicate_prefix_commit():
    transaction = PhaseStitchTransaction(_lease())
    transaction.mark_replay_started()
    transaction.mark_prefix_ready()
    transaction.mark_prefix_committed()

    with pytest.raises(ValueError, match="prefix commit"):
        transaction.mark_prefix_committed()


def test_phase_stitch_transaction_distinguishes_failure_visibility():
    before_prefix = PhaseStitchTransaction(_lease())
    before_prefix.mark_replay_started()
    before_prefix.fail("replay_failed")
    assert before_prefix.state == "failed_before_prefix"
    assert before_prefix.partial_visibility is False
    assert before_prefix.failure_reason == "replay_failed"

    after_prefix = PhaseStitchTransaction(_lease())
    after_prefix.mark_replay_started()
    after_prefix.mark_prefix_ready()
    after_prefix.mark_prefix_committed()
    after_prefix.fail("suffix_copy_failed")
    assert after_prefix.state == "failed_after_prefix"
    assert after_prefix.partial_visibility is True
    assert after_prefix.failure_reason == "suffix_copy_failed"


def test_phase_stitch_transaction_cancels_only_before_replay():
    transaction = PhaseStitchTransaction(_lease())
    transaction.cancel("unsupported_request")
    assert transaction.state == "cancelled"

    started = PhaseStitchTransaction(_lease())
    started.mark_replay_started()
    with pytest.raises(ValueError, match="cannot cancel"):
        started.cancel("too_late")


def test_phase_stitch_config_defaults_off_and_rejects_non_bool():
    Config = _load_config_class()
    with tempfile.TemporaryDirectory() as model:
        assert Config(
            model=model
        ).phase_stitched_exact_graph_runtime is False
        with pytest.raises(
            ValueError,
            match="phase_stitched_exact_graph_runtime must be a bool",
        ):
            Config(
                model=model,
                phase_stitched_exact_graph_runtime=1,
            )


def test_phase_stitch_admission_accepts_exact_supported_request():
    decision = _admission()

    assert decision.optimized is True
    assert decision.fallback_reason is None


@pytest.mark.parametrize(
    ("overrides", "reason"),
    (
        ({"enabled": False}, "disabled"),
        (
            {"prefill_graph_available": False},
            "prefill_graph_unavailable",
        ),
        (
            {"decode_graph_available": False},
            "decode_graph_unavailable",
        ),
        (
            {"prompt_token_count": 128},
            "prompt_shape_not_allowlisted",
        ),
        ({"sequence_count": 2}, "sequence_count_unsupported"),
        ({"waiting_count": 1}, "waiting_request_present"),
        (
            {"prefilling_count": 1},
            "prefilling_request_present",
        ),
        ({"do_sample": False}, "sampling_unsupported"),
        ({"temperatures": (0.1,)}, "temperature_nonzero"),
        ({"ignore_eos": (False,)}, "ignore_eos_required"),
        ({"completion_only": False}, "completion_only_required"),
        (
            {"remaining_output_tokens": 7},
            "output_budget_insufficient",
        ),
        (
            {"decode_kv_capacity_tokens": 6},
            "decode_kv_capacity_insufficient",
        ),
        (
            {"tensor_parallel_size": 2},
            "tensor_parallel_unsupported",
        ),
        ({"rank": 1}, "non_root_rank"),
        (
            {"incompatible_modes": ("kv_offload",)},
            "incompatible_mode:kv_offload",
        ),
        ({"pending_lease": True}, "lease_pending"),
        ({"quarantined": True}, "identity_quarantined"),
    ),
)
def test_phase_stitch_admission_matrix(overrides, reason):
    decision = _admission(**overrides)

    assert decision.optimized is False
    assert decision.fallback_reason == reason
