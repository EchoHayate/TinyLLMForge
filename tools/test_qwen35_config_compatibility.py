import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_config_class(hf_config):
    module_name = "tinyvllm_config_qwen35_compatibility_under_test"
    module_path = REPO_ROOT / "tinyvllm" / "config.py"
    fake_transformers = ModuleType("transformers")

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model):
            del model
            return hf_config

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


def test_config_keeps_explicit_limit_when_hf_config_has_no_position_limit():
    Config = _load_config_class(SimpleNamespace(num_hidden_layers=4))
    with tempfile.TemporaryDirectory() as model:
        config = Config(
            model=model,
            max_num_batched_tokens=4096,
            max_model_len=4096,
            kvcache_block_size=256,
        )
    assert config.max_model_len == 4096


def _config(**overrides):
    Config = _load_config_class(
        SimpleNamespace(num_hidden_layers=4)
    )
    model = tempfile.TemporaryDirectory()
    values = {
        "model": model.name,
        "max_num_batched_tokens": 4096,
        "max_model_len": 4096,
        "kvcache_block_size": 256,
    }
    values.update(overrides)
    return model, Config(**values)


def test_proposal_kv_offload_defaults_are_strictly_disabled():
    model, config = _config()
    try:
        assert config.proposal_kv_offload_enabled is False
        assert config.proposal_kv_logical_entry_capacity == 0
        assert config.proposal_kv_gpu_slot_capacity == 0
        assert config.proposal_kv_cpu_backing_capacity == 0
        assert config.proposal_kv_async_copy is True
        assert config.proposal_kv_batch_copy is True
    finally:
        model.cleanup()


def test_proposal_kv_offload_accepts_fixed_v1_capacity_contract():
    model, config = _config(
        qwen35_mtp_enabled=True,
        qwen35_mtp_cuda_graphs=False,
        proposal_kv_offload_enabled=True,
        proposal_kv_logical_entry_capacity=16,
        proposal_kv_gpu_slot_capacity=4,
        proposal_kv_cpu_backing_capacity=16,
    )
    try:
        assert config.proposal_kv_offload_enabled is True
    finally:
        model.cleanup()


@pytest.mark.parametrize(
    "overrides,match",
    (
        (
            {
                "qwen35_mtp_enabled": True,
                "proposal_kv_offload_enabled": True,
                "proposal_kv_logical_entry_capacity": 16,
                "proposal_kv_gpu_slot_capacity": 4,
                "proposal_kv_cpu_backing_capacity": 15,
            },
            "logical == cpu > gpu > 0",
        ),
        (
            {
                "qwen35_mtp_enabled": True,
                "proposal_kv_offload_enabled": True,
                "proposal_kv_logical_entry_capacity": 4,
                "proposal_kv_gpu_slot_capacity": 4,
                "proposal_kv_cpu_backing_capacity": 4,
            },
            "logical == cpu > gpu > 0",
        ),
        (
            {
                "qwen35_mtp_enabled": True,
                "proposal_kv_offload_enabled": True,
                "proposal_kv_logical_entry_capacity": 16,
                "proposal_kv_gpu_slot_capacity": 0,
                "proposal_kv_cpu_backing_capacity": 16,
            },
            "logical == cpu > gpu > 0",
        ),
        (
            {
                "proposal_kv_offload_enabled": True,
                "proposal_kv_logical_entry_capacity": 16,
                "proposal_kv_gpu_slot_capacity": 4,
                "proposal_kv_cpu_backing_capacity": 16,
            },
            "Qwen3.5 MTP",
        ),
        (
            {
                "qwen35_mtp_enabled": True,
                "qwen35_mtp_cuda_graphs": True,
                "proposal_kv_offload_enabled": True,
                "proposal_kv_logical_entry_capacity": 16,
                "proposal_kv_gpu_slot_capacity": 4,
                "proposal_kv_cpu_backing_capacity": 16,
            },
            "CUDA graphs",
        ),
        (
            {
                "proposal_kv_logical_entry_capacity": True,
            },
            "nonnegative integer",
        ),
    ),
)
def test_proposal_kv_offload_rejects_invalid_v1_configuration(
    overrides,
    match,
):
    Config = _load_config_class(
        SimpleNamespace(num_hidden_layers=4)
    )
    with tempfile.TemporaryDirectory() as model:
        with pytest.raises(ValueError, match=match):
            Config(
                model=model,
                max_num_batched_tokens=4096,
                max_model_len=4096,
                kvcache_block_size=256,
                **overrides,
            )
