from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tinyvllm/models/qwen38_text_adopter.py"


def _load():
    assert MODULE_PATH.is_file(), "Qwen3.8 text adopter is not implemented"
    spec = importlib.util.spec_from_file_location(
        "qwen38_text_adopter_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


adopter = _load()
QWEN38_ARCHITECTURE = adopter.QWEN38_ARCHITECTURE
QWEN38_REPOSITORY = adopter.QWEN38_REPOSITORY
adopt_qwen38_text_config = adopter.adopt_qwen38_text_config
is_qwen38_text_checkpoint = adopter.is_qwen38_text_checkpoint
reject_qwen38_multimodal_inputs = (
    adopter.reject_qwen38_multimodal_inputs
)
validate_qwen38_sequence_batch = (
    adopter.validate_qwen38_sequence_batch
)


REVISION = "1" * 40


def _official_config(**overrides):
    layer_types = tuple(
        "full_attention" if (index + 1) % 4 == 0
        else "linear_attention"
        for index in range(64)
    )
    text_values = {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "layer_types": layer_types,
        "dtype": "bfloat16",
        "vocab_size": 248320,
        "image_token_id": 248056,
        "video_token_id": 248057,
    }
    text_values.update(overrides.pop("text_overrides", {}))
    values = {
        "_name_or_path": QWEN38_REPOSITORY,
        "_commit_hash": REVISION,
        "model_type": "qwen3_5",
        "architectures": [QWEN38_ARCHITECTURE],
        "language_model_only": False,
        "text_config": SimpleNamespace(**text_values),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_adopts_official_qwen38_text_topology():
    profile = adopt_qwen38_text_config(_official_config())

    assert profile.repository == QWEN38_REPOSITORY
    assert profile.revision == REVISION
    assert profile.architecture == QWEN38_ARCHITECTURE
    assert profile.text_model_type == "qwen3_5_text"
    assert profile.num_hidden_layers == 64
    assert profile.hidden_size == 5120
    assert profile.intermediate_size == 17408
    assert profile.layer_types[0] == "linear_attention"
    assert profile.layer_types[3] == "full_attention"
    assert profile.dtype == "bfloat16"
    assert profile.vocab_size == 248320
    assert profile.language_model_only is False
    assert profile.multimodal_token_ids == (248056, 248057)

    with pytest.raises(FrozenInstanceError):
        profile.hidden_size = 1


def test_qwen38_detection_requires_architecture_and_exact_topology():
    assert is_qwen38_text_checkpoint(_official_config()) is True
    assert is_qwen38_text_checkpoint(
        _official_config(architectures=["SomeOtherArchitecture"])
    ) is False
    assert is_qwen38_text_checkpoint(
        _official_config(text_overrides={"hidden_size": 4096})
    ) is False


@pytest.mark.parametrize(
    "config,match",
    (
        (
            _official_config(_name_or_path="Qwen/another-model"),
            "repository",
        ),
        (
            _official_config(_commit_hash="main"),
            "immutable revision",
        ),
        (
            _official_config(architectures=["SomeOtherArchitecture"]),
            "architecture",
        ),
        (
            SimpleNamespace(
                _name_or_path=QWEN38_REPOSITORY,
                _commit_hash=REVISION,
                model_type="qwen3_5",
                architectures=[QWEN38_ARCHITECTURE],
            ),
            "text_config",
        ),
        (
            _official_config(
                text_overrides={"model_type": "qwen3_5"}
            ),
            "text model type",
        ),
        (
            _official_config(
                text_overrides={"num_hidden_layers": 63}
            ),
            "num_hidden_layers",
        ),
        (
            _official_config(
                text_overrides={"hidden_size": 4096}
            ),
            "hidden_size",
        ),
        (
            _official_config(
                text_overrides={"intermediate_size": 16384}
            ),
            "intermediate_size",
        ),
        (
            _official_config(
                text_overrides={
                    "layer_types": ("linear_attention",) * 64
                }
            ),
            "layer cadence",
        ),
        (
            _official_config(text_overrides={"dtype": "float16"}),
            "dtype",
        ),
    ),
)
def test_rejects_identity_or_topology_drift(config, match):
    with pytest.raises(ValueError, match=match):
        adopt_qwen38_text_config(config)


def test_rejects_multimodal_payload_or_special_tokens():
    profile = adopt_qwen38_text_config(_official_config())

    with pytest.raises(ValueError, match="text-only"):
        reject_qwen38_multimodal_inputs(
            profile,
            input_ids=[1, 248056, 2],
        )
    with pytest.raises(ValueError, match="text-only"):
        reject_qwen38_multimodal_inputs(
            profile,
            input_ids=[1, 2],
            multimodal_inputs={"image": object()},
        )


def test_accepts_plain_python_text_token_ids_without_tensor_sync():
    profile = adopt_qwen38_text_config(_official_config())

    reject_qwen38_multimodal_inputs(
        profile,
        input_ids=[1, 2, 3],
        multimodal_inputs=None,
    )


def test_sequence_batch_checks_prompt_tokens_before_prefill():
    profile = adopt_qwen38_text_config(_official_config())
    seqs = [
        SimpleNamespace(prompt_token_ids=[1, 2, 3], last_token=3),
        SimpleNamespace(prompt_token_ids=[4, 248057], last_token=248057),
    ]

    with pytest.raises(ValueError, match="text-only"):
        validate_qwen38_sequence_batch(
            profile,
            seqs,
            is_prefill=True,
        )


def test_sequence_batch_checks_only_latest_token_during_decode():
    profile = adopt_qwen38_text_config(_official_config())
    seqs = [
        SimpleNamespace(
            prompt_token_ids=[248056],
            last_token=7,
        )
    ]

    validate_qwen38_sequence_batch(
        profile,
        seqs,
        is_prefill=False,
    )
    seqs[0].last_token = 248056
    with pytest.raises(ValueError, match="text-only"):
        validate_qwen38_sequence_batch(
            profile,
            seqs,
            is_prefill=False,
        )
