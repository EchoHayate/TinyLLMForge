"""Dependency-light tests for speculative verifier CUDA Graph config."""

from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "tinyvllm" / "config.py"


def load_real_config_class():
    module_name = "tinyvllm_spec_verify_graph_config_under_test"
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
        if original is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = original


def test_spec_verify_cuda_graph_config_defaults_fail_closed():
    Config = load_real_config_class()
    with tempfile.TemporaryDirectory() as model:
        config = Config(model=model)

    assert config.spec_verify_cuda_graphs is False
    assert config.spec_verify_cuda_graph_batch_allowlist == (1, 4)
    assert config.spec_verify_cuda_graph_query_len_allowlist == ()
    assert config.spec_verify_cuda_graph_min_observations == 2
    assert config.spec_verify_cuda_graph_max_entries == 8
    assert (
        config.spec_verify_cuda_graph_max_static_bytes
        == 64 * 1024 * 1024
    )
    assert (
        config.spec_verify_cuda_graph_max_reserved_bytes
        == 512 * 1024 * 1024
    )
    assert (
        config.spec_verify_cuda_graph_max_total_capture_ns
        == 5_000_000_000
    )
    assert (
        config.spec_verify_cuda_graph_max_single_capture_ns
        == 2_000_000_000
    )


def test_spec_verify_cuda_graph_allowlists_are_independently_canonical():
    Config = load_real_config_class()
    with tempfile.TemporaryDirectory() as model:
        config = Config(
            model=model,
            spec_verify_cuda_graph_batch_allowlist=[4, 1, 4],
            spec_verify_cuda_graph_query_len_allowlist=[8, 1, 8, 3],
        )

    assert config.spec_verify_cuda_graph_batch_allowlist == (1, 4)
    assert config.spec_verify_cuda_graph_query_len_allowlist == (1, 3, 8)
    assert config.multi_sequence_cuda_graph_batch_allowlist == (2, 4, 8)


@pytest.mark.parametrize(
    "overrides",
    (
        {"spec_verify_cuda_graphs": 1},
        {"spec_verify_cuda_graph_batch_allowlist": ()},
        {"spec_verify_cuda_graph_batch_allowlist": "1,4"},
        {"spec_verify_cuda_graph_batch_allowlist": (0, 1)},
        {"spec_verify_cuda_graph_batch_allowlist": (1, True)},
        {"spec_verify_cuda_graph_batch_allowlist": (1, 4.0)},
        {"spec_verify_cuda_graph_query_len_allowlist": "1,3"},
        {"spec_verify_cuda_graph_query_len_allowlist": (0,)},
        {"spec_verify_cuda_graph_query_len_allowlist": (1, True)},
        {"spec_verify_cuda_graph_query_len_allowlist": (1, 3.0)},
        {"spec_verify_cuda_graph_min_observations": 0},
        {"spec_verify_cuda_graph_min_observations": True},
        {"spec_verify_cuda_graph_max_entries": 0},
        {"spec_verify_cuda_graph_max_static_bytes": 0},
        {"spec_verify_cuda_graph_max_reserved_bytes": 0},
        {"spec_verify_cuda_graph_max_total_capture_ns": 0},
        {"spec_verify_cuda_graph_max_single_capture_ns": 0},
    ),
)
def test_spec_verify_cuda_graph_config_rejects_invalid_controls(
    overrides,
):
    Config = load_real_config_class()
    with tempfile.TemporaryDirectory() as model:
        with pytest.raises((AssertionError, ValueError, TypeError)):
            Config(model=model, **overrides)


def test_decode_graph_defaults_and_validator_remain_unchanged():
    Config = load_real_config_class()
    with tempfile.TemporaryDirectory() as model:
        config = Config(model=model)
        assert config.multi_sequence_cuda_graphs is False
        assert config.multi_sequence_cuda_graph_batch_allowlist == (2, 4, 8)
        with pytest.raises(AssertionError):
            Config(
                model=model,
                multi_sequence_cuda_graph_batch_allowlist=(1, 2),
            )
