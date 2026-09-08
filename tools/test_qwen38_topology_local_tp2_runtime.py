import __future__
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "tinyvllm" / "config.py"

for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)

fake_torch = types.ModuleType("torch")
fake_torch.Tensor = object
fake_torch.cat = lambda tensors, dim=0: tuple(tensors)
sys.modules.setdefault("torch", fake_torch)


def _load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        code = compile(
            path.read_text(),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        )
        exec(code, module.__dict__)
        return module
    except Exception:
        sys.modules.pop(name, None)
        raise


def _load_real_config_class():
    module_name = "tinyvllm_config_topology_local_tp2_under_test"
    fake_transformers = types.ModuleType("transformers")

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model):
            del model
            return SimpleNamespace(
                max_position_embeddings=4096,
                num_hidden_layers=64,
            )

    fake_transformers.AutoConfig = FakeAutoConfig
    original_transformers = sys.modules.get("transformers")
    try:
        sys.modules["transformers"] = fake_transformers
        return _load_source_module(module_name, CONFIG_PATH).Config
    finally:
        if original_transformers is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = original_transformers
        sys.modules.pop(module_name, None)


def _build_config(**overrides):
    Config = _load_real_config_class()
    model_directory = tempfile.TemporaryDirectory()
    config = Config(
        model=model_directory.name,
        max_num_batched_tokens=4096,
        **overrides,
    )
    config._test_model_directory = model_directory
    return config


def test_topology_local_tp2_config_is_default_off_and_strict_bool():
    Config = _load_real_config_class()

    assert (
        Config.__dataclass_fields__[
            "qwen38_topology_local_tp2_islands"
        ].default
        is False
    )
    with pytest.raises(
        ValueError,
        match=(
            "^qwen38_topology_local_tp2_islands must be a bool$"
        ),
    ):
        _build_config(qwen38_topology_local_tp2_islands=1)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        (
            {"tensor_parallel_size": 2, "enforce_eager": True},
            "requires tensor_parallel_size 4",
        ),
        (
            {"tensor_parallel_size": 4, "enforce_eager": False},
            "requires eager execution",
        ),
    ),
)
def test_topology_local_tp2_config_rejects_wrong_execution_contract(
    overrides,
    message,
):
    with pytest.raises(ValueError, match=message):
        _build_config(
            qwen38_topology_local_tp2_islands=True,
            **overrides,
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"qwen35_mtp_enabled": True},
        {"autoregressive_draft_enabled": True},
        {"multi_sequence_cuda_graphs": True},
        {"prefill_cuda_graphs": True},
        {"cpu_offload": True},
        {"kv_offload_mvp0": True},
        {"exact_greedy_decode_burst": True},
        {"graph_resident_greedy_tail": True},
        {"quantization": "int8"},
        {"kv_quant_bits": 8},
        {"act_quant_bits": 8},
        {"smoothquant_scale_path": "non-null"},
        {"quest_top_k_blocks": 1},
        {"kv_cartridge_blocks": 1},
        {"am_compact_blocks": 1},
    ),
)
def test_topology_local_tp2_config_rejects_incompatible_modes(
    overrides,
):
    with pytest.raises(ValueError, match="incompatible"):
        _build_config(
            tensor_parallel_size=4,
            enforce_eager=True,
            qwen38_topology_local_tp2_islands=True,
            **overrides,
        )


class FakeDistributed:
    def __init__(self):
        self.calls = []

    def new_group(self, *, ranks):
        group = ("group", tuple(ranks))
        self.calls.append(tuple(ranks))
        return group


def test_pair_context_creates_all_groups_in_global_order():
    from tinyvllm.engine.topology_local_tp2_island import (
        TopologyLocalTP2PairMap,
        create_topology_local_tp2_pair_context,
    )

    pair_map = TopologyLocalTP2PairMap(((0, 1), (2, 3)))
    for global_rank in range(4):
        distributed = FakeDistributed()
        context = create_topology_local_tp2_pair_context(
            distributed,
            global_rank=global_rank,
            world_size=4,
            pair_map=pair_map,
        )

        assert distributed.calls == [(0, 1), (2, 3)]
        assert context.identity.global_rank == global_rank
        assert context.identity.pair_id == global_rank // 2
        assert context.identity.logical_rank == global_rank % 2
        assert context.pair_group == (
            "group",
            pair_map.pair_groups[global_rank // 2],
        )
        assert context.all_pair_groups == (
            ("group", (0, 1)),
            ("group", (2, 3)),
        )


@pytest.mark.parametrize("world_size", (1, 2, 3, 5, True))
def test_pair_context_rejects_non_tp4_world_size(world_size):
    from tinyvllm.engine.topology_local_tp2_island import (
        TopologyLocalTP2PairMap,
        create_topology_local_tp2_pair_context,
    )

    with pytest.raises(
        ValueError,
        match="requires world_size 4",
    ):
        create_topology_local_tp2_pair_context(
            FakeDistributed(),
            global_rank=0,
            world_size=world_size,
            pair_map=TopologyLocalTP2PairMap(((0, 1), (2, 3))),
        )
