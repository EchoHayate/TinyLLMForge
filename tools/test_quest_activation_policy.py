from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "tinyvllm" / "engine" / "quest_activation.py"
CONFIG_PATH = ROOT / "tinyvllm" / "config.py"


def _load_policy_module():
    spec = importlib.util.spec_from_file_location(
        "tinyvllm_quest_activation_under_test",
        POLICY_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


resolve_quest_activation = _load_policy_module().resolve_quest_activation


def _load_config_class():
    module_name = "tinyvllm_quest_activation_config_under_test"
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
        module = types.ModuleType(module_name)
        module.__file__ = os.fspath(CONFIG_PATH)
        sys.modules[module_name] = module
        source = CONFIG_PATH.read_text(encoding="utf-8")
        exec(
            compile(
                "from __future__ import annotations\n" + source,
                os.fspath(CONFIG_PATH),
                "exec",
            ),
            module.__dict__,
        )
        return module.Config
    finally:
        sys.modules.pop(module_name, None)
        if original is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = original


def _resolve(
    *,
    batch: int = 8,
    sequence_length: int = 8192,
    block_count: int = 32,
    requested_top_k: int = 16,
    min_seq_len: int = 512,
    min_saved_blocks: int = 128,
    incompatible_feature: bool = False,
):
    return resolve_quest_activation(
        requested_top_k=requested_top_k,
        min_seq_len=min_seq_len,
        min_saved_blocks=min_saved_blocks,
        block_size=256,
        sequence_lengths=[sequence_length] * batch,
        sequence_block_counts=[block_count] * batch,
        incompatible_feature=incompatible_feature,
    )


@pytest.mark.parametrize(
    ("batch", "saved_blocks"),
    ((4, 64), (6, 96)),
)
def test_saved_block_threshold_falls_back_below_boundary(
    batch,
    saved_blocks,
):
    decision = _resolve(batch=batch)

    assert decision.resolved_top_k == -1
    assert decision.saved_blocks == saved_blocks
    assert decision.reason == "below_saved_blocks"


def test_saved_block_threshold_activates_at_boundary():
    decision = _resolve(batch=8)

    assert decision.requested_top_k == 16
    assert decision.resolved_top_k == 16
    assert decision.min_seq_len == 512
    assert decision.min_saved_blocks == 128
    assert decision.saved_blocks == 128
    assert decision.batch_size == 8
    assert decision.reason == "active"


def test_zero_threshold_preserves_existing_activation():
    decision = _resolve(batch=1, min_saved_blocks=0)

    assert decision.resolved_top_k == 16
    assert decision.saved_blocks == 16
    assert decision.reason == "active"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    (
        ({"requested_top_k": -1}, "disabled"),
        ({"incompatible_feature": True}, "incompatible_feature"),
        ({"sequence_length": 511}, "below_min_seq_len"),
        ({"block_count": 16}, "insufficient_blocks"),
        (
            {
                "batch": 1,
                "sequence_length": 5120,
                "block_count": 20,
                "min_saved_blocks": 0,
            },
            "insufficient_pruning",
        ),
    ),
)
def test_existing_eligibility_reasons_take_precedence(
    overrides,
    reason,
):
    decision = _resolve(**overrides)

    assert decision.resolved_top_k == -1
    assert decision.reason == reason


def test_policy_rejects_mismatched_sequence_metadata():
    with pytest.raises(
        ValueError,
        match="sequence lengths and block counts must have equal size",
    ):
        resolve_quest_activation(
            requested_top_k=16,
            min_seq_len=512,
            min_saved_blocks=128,
            block_size=256,
            sequence_lengths=[8192, 8192],
            sequence_block_counts=[32],
            incompatible_feature=False,
        )


def test_config_defaults_adaptive_quest_activation_off():
    Config = _load_config_class()
    with tempfile.TemporaryDirectory() as model:
        config = Config(model=model)

    assert config.quest_min_saved_blocks == 0


def test_config_accepts_non_negative_saved_block_threshold():
    Config = _load_config_class()
    with tempfile.TemporaryDirectory() as model:
        config = Config(
            model=model,
            quest_min_saved_blocks=128,
        )

    assert config.quest_min_saved_blocks == 128


@pytest.mark.parametrize("value", (-1, True, 1.5, "128"))
def test_config_rejects_invalid_saved_block_threshold(value):
    Config = _load_config_class()
    with tempfile.TemporaryDirectory() as model:
        with pytest.raises(
            ValueError,
            match=(
                "quest_min_saved_blocks must be a "
                "non-negative integer"
            ),
        ):
            Config(
                model=model,
                quest_min_saved_blocks=value,
            )
