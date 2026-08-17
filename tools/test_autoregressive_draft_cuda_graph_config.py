from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_config_class():
    module_name = "tinyvllm_config_autoregressive_graph_under_test"
    module_path = ROOT / "tinyvllm" / "config.py"
    fake_transformers = ModuleType("transformers")

    class FakeAutoConfig:

        @staticmethod
        def from_pretrained(model):
            del model
            return SimpleNamespace(num_hidden_layers=4)

    fake_transformers.AutoConfig = FakeAutoConfig
    original_transformers = sys.modules.get("transformers")
    original_module = sys.modules.get(module_name)
    sys.modules["transformers"] = fake_transformers
    try:
        module = ModuleType(module_name)
        module.__file__ = os.fspath(module_path)
        sys.modules[module_name] = module
        source = module_path.read_text(encoding="utf-8")
        exec(
            compile(
                "from __future__ import annotations\n" + source,
                os.fspath(module_path),
                "exec",
            ),
            module.__dict__,
        )
        return module.Config
    finally:
        if original_transformers is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = original_transformers
        if original_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_module


def _config(**overrides):
    Config = _load_config_class()
    model = tempfile.TemporaryDirectory()
    values = {
        "model": model.name,
        "max_num_batched_tokens": 4096,
        "max_model_len": 4096,
        "kvcache_block_size": 256,
    }
    values.update(overrides)
    try:
        return model, Config(**values)
    except BaseException:
        model.cleanup()
        raise


def _enabled_values():
    return {
        "tensor_parallel_size": 4,
        "autoregressive_draft_enabled": True,
        "autoregressive_draft_model": "/models/qwen3-draft",
        "autoregressive_draft_max_proposal_tokens": 4,
        "autoregressive_draft_gpu_slot_capacity": 4096,
        "autoregressive_draft_cuda_graphs": True,
    }


def test_autoregressive_draft_graph_defaults_fail_closed():
    model, config = _config()
    try:
        assert config.autoregressive_draft_cuda_graphs is False
        assert (
            config.autoregressive_draft_cuda_graph_q_allowlist
            == (4,)
        )
        assert (
            config.autoregressive_draft_cuda_graph_batch_allowlist
            == (4,)
        )
        assert (
            config.autoregressive_draft_cuda_graph_min_observations
            == 2
        )
        assert (
            config.autoregressive_draft_cuda_graph_max_entries
            == 4
        )
        assert (
            config.autoregressive_draft_cuda_graph_max_static_bytes
            == 64 * 1024 * 1024
        )
        assert (
            config.autoregressive_draft_cuda_graph_max_reserved_bytes
            == 512 * 1024 * 1024
        )
        assert (
            config.autoregressive_draft_cuda_graph_max_total_capture_ns
            == 5_000_000_000
        )
        assert (
            config.autoregressive_draft_cuda_graph_max_single_capture_ns
            == 4_000_000_000
        )
    finally:
        model.cleanup()


def test_autoregressive_draft_graph_allowlists_are_canonical():
    model, config = _config(
        autoregressive_draft_cuda_graph_q_allowlist=[4, 2, 4],
        autoregressive_draft_cuda_graph_batch_allowlist=[4, 1, 4],
    )
    try:
        assert (
            config.autoregressive_draft_cuda_graph_q_allowlist
            == (2, 4)
        )
        assert (
            config.autoregressive_draft_cuda_graph_batch_allowlist
            == (1, 4)
        )
    finally:
        model.cleanup()


def test_enabled_first_slice_accepts_only_tp4_b4_q4_dense_mode():
    model, config = _config(**_enabled_values())
    try:
        assert config.autoregressive_draft_cuda_graphs is True
        assert config.tensor_parallel_size == 4
        assert (
            config.autoregressive_draft_proposal_kv_offload_enabled
            is False
        )
    finally:
        model.cleanup()


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        (
            {"autoregressive_draft_cuda_graphs": 1},
            "autoregressive_draft_cuda_graphs must be a bool",
        ),
        (
            {
                "autoregressive_draft_cuda_graph_q_allowlist": (),
            },
            "q_allowlist",
        ),
        (
            {
                "autoregressive_draft_cuda_graph_q_allowlist": (
                    1,
                    4,
                ),
            },
            "at least two",
        ),
        (
            {
                "autoregressive_draft_cuda_graph_batch_allowlist": (),
            },
            "batch_allowlist",
        ),
        (
            {
                "autoregressive_draft_cuda_graph_min_observations": 0,
            },
            "positive integer",
        ),
        (
            {
                "autoregressive_draft_cuda_graph_max_entries": True,
            },
            "positive integer",
        ),
        (
            {
                "autoregressive_draft_cuda_graph_max_static_bytes": 0,
            },
            "positive integer",
        ),
        (
            {
                "autoregressive_draft_cuda_graph_max_reserved_bytes": 0,
            },
            "positive integer",
        ),
        (
            {
                "autoregressive_draft_cuda_graph_max_total_capture_ns": 0,
            },
            "positive integer",
        ),
        (
            {
                "autoregressive_draft_cuda_graph_max_single_capture_ns": 0,
            },
            "positive integer",
        ),
    ),
)
def test_autoregressive_draft_graph_controls_reject_invalid_values(
    overrides,
    match,
):
    with pytest.raises((AssertionError, ValueError), match=match):
        _config(**overrides)


@pytest.mark.parametrize(
    ("changes", "match"),
    (
        (
            {
                "autoregressive_draft_enabled": False,
                "autoregressive_draft_model": None,
                "autoregressive_draft_gpu_slot_capacity": 0,
            },
            "requires autoregressive draft",
        ),
        (
            {"tensor_parallel_size": 1},
            "requires tensor_parallel_size 4",
        ),
        (
            {
                "autoregressive_draft_proposal_kv_offload_enabled": (
                    True
                ),
                "autoregressive_draft_logical_entry_capacity": 8192,
                "autoregressive_draft_cpu_backing_capacity": 8192,
            },
            "incompatible with proposal KV offload",
        ),
        (
            {
                "autoregressive_draft_cuda_graph_q_allowlist": (
                    3,
                    4,
                ),
            },
            "Q4 only",
        ),
        (
            {
                "autoregressive_draft_cuda_graph_batch_allowlist": (
                    1,
                    4,
                ),
            },
            "batch size 4 only",
        ),
        (
            {
                "autoregressive_draft_max_proposal_tokens": 3,
            },
            "max proposal tokens 4",
        ),
    ),
)
def test_enabled_graph_rejects_unsupported_first_slice(
    changes,
    match,
):
    values = _enabled_values()
    values.update(changes)
    with pytest.raises(ValueError, match=match):
        _config(**values)
