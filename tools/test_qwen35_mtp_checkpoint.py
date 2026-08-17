from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import types

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
for package_name in (
    "tinyvllm",
    "tinyvllm.models",
):
    package = types.ModuleType(package_name)
    package.__path__ = [
        str(_REPO_ROOT / package_name.replace(".", "/"))
    ]
    sys.modules[package_name] = package

from tinyvllm.models.qwen35_checkpoint import (
    Qwen35CheckpointSkip,
    Qwen35CheckpointSource,
    Qwen35CheckpointWeightPlan,
    qwen35_mtp_skip_source_names,
)
from tinyvllm.models.qwen35_mtp_checkpoint import (
    Qwen35MTPCheckpointPlan,
    bind_qwen35_mtp_checkpoint,
    build_qwen35_mtp_checkpoint_plan,
)


EXPECTED_MTP = {
    "mtp.fc.weight": ("BF16", (2048, 4096)),
    "mtp.layers.0.input_layernorm.weight": (
        "BF16",
        (2048,),
    ),
    "mtp.layers.0.self_attn.q_proj.weight": (
        "BF16",
        (4096, 2048),
    ),
    "mtp.layers.0.self_attn.k_proj.weight": (
        "BF16",
        (512, 2048),
    ),
    "mtp.layers.0.self_attn.v_proj.weight": (
        "BF16",
        (512, 2048),
    ),
    "mtp.layers.0.self_attn.o_proj.weight": (
        "BF16",
        (2048, 2048),
    ),
    "mtp.layers.0.self_attn.q_norm.weight": (
        "BF16",
        (256,),
    ),
    "mtp.layers.0.self_attn.k_norm.weight": (
        "BF16",
        (256,),
    ),
    "mtp.layers.0.post_attention_layernorm.weight": (
        "BF16",
        (2048,),
    ),
    "mtp.layers.0.mlp.gate_proj.weight": (
        "BF16",
        (6144, 2048),
    ),
    "mtp.layers.0.mlp.up_proj.weight": (
        "BF16",
        (6144, 2048),
    ),
    "mtp.layers.0.mlp.down_proj.weight": (
        "BF16",
        (2048, 6144),
    ),
    "mtp.norm.weight": ("BF16", (2048,)),
    "mtp.pre_fc_norm_embedding.weight": (
        "BF16",
        (2048,),
    ),
    "mtp.pre_fc_norm_hidden.weight": (
        "BF16",
        (2048,),
    ),
}


def _config(**overrides):
    values = {
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "tie_word_embeddings": True,
        "dtype": "bfloat16",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _fixture():
    weight_map = {}
    shard_headers = {
        "model-00001-of-00002.safetensors": {},
        "model-00002-of-00002.safetensors": {},
    }
    offsets = {
        shard_name: 0 for shard_name in shard_headers
    }
    for index, source_name in enumerate(sorted(EXPECTED_MTP)):
        shard_name = tuple(shard_headers)[index % 2]
        dtype, shape = EXPECTED_MTP[source_name]
        elements = 1
        for dimension in shape:
            elements *= dimension
        byte_count = elements * 2
        start = offsets[shard_name]
        end = start + byte_count
        offsets[shard_name] = end
        weight_map[source_name] = shard_name
        shard_headers[shard_name][source_name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, end],
        }
    return (
        {
            "metadata": {
                "total_size": sum(offsets.values()),
            },
            "weight_map": weight_map,
        },
        shard_headers,
    )


def _plan():
    index_payload, shard_headers = _fixture()
    return build_qwen35_mtp_checkpoint_plan(
        _config(),
        index_payload,
        shard_headers,
    )


def test_builds_exact_immutable_mtp_tensor_plan():
    plan = _plan()

    assert type(plan) is Qwen35MTPCheckpointPlan
    assert tuple(
        tensor.source_name for tensor in plan.tensors
    ) == tuple(sorted(EXPECTED_MTP))
    assert {
        tensor.source_name: (
            tensor.dtype,
            tensor.shape,
        )
        for tensor in plan.tensors
    } == EXPECTED_MTP
    assert plan.payload_bytes == sum(
        tensor.data_offsets[1] - tensor.data_offsets[0]
        for tensor in plan.tensors
    )
    with pytest.raises(Exception):
        plan.tensors[0].shape += (1,)


def test_plan_maps_separate_attention_and_packed_mlp_slots():
    by_source = {
        tensor.source_name: tensor
        for tensor in _plan().tensors
    }

    assert (
        by_source[
            "mtp.layers.0.self_attn.q_proj.weight"
        ].destination_path
        == "layer.decoder_layer.full_attention.q_projection.weight"
    )
    assert (
        by_source[
            "mtp.layers.0.self_attn.q_proj.weight"
        ].packed_slot
        is None
    )
    assert (
        by_source[
            "mtp.layers.0.self_attn.k_proj.weight"
        ].packed_slot
        is None
    )
    assert (
        by_source[
            "mtp.layers.0.self_attn.v_proj.weight"
        ].packed_slot
        is None
    )
    assert (
        by_source[
            "mtp.layers.0.mlp.gate_proj.weight"
        ].destination_path
        == "layer.decoder_layer.mlp.gate_up_proj.weight"
    )
    assert (
        by_source[
            "mtp.layers.0.mlp.gate_proj.weight"
        ].packed_slot
        == 0
    )
    assert (
        by_source[
            "mtp.layers.0.mlp.up_proj.weight"
        ].packed_slot
        == 1
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"mtp_num_hidden_layers": 2}, "mtp_num_hidden_layers"),
        (
            {"mtp_use_dedicated_embeddings": True},
            "dedicated",
        ),
        ({"tie_word_embeddings": False}, "tie_word_embeddings"),
        ({"dtype": "float32"}, "bfloat16"),
    ],
)
def test_rejects_unsupported_mtp_config(overrides, match):
    index_payload, shard_headers = _fixture()

    with pytest.raises(ValueError, match=match):
        build_qwen35_mtp_checkpoint_plan(
            _config(**overrides),
            index_payload,
            shard_headers,
        )


@pytest.mark.parametrize("mutation,match", [
    ("missing", "missing"),
    ("unexpected", "unexpected"),
    ("dtype", "dtype"),
    ("shape", "shape"),
    ("wrong_shard", "shard"),
])
def test_rejects_invalid_mtp_source_contract(mutation, match):
    index_payload, shard_headers = _fixture()
    source_name = sorted(EXPECTED_MTP)[0]
    shard_name = index_payload["weight_map"][source_name]
    if mutation == "missing":
        del index_payload["weight_map"][source_name]
        del shard_headers[shard_name][source_name]
    elif mutation == "unexpected":
        index_payload["weight_map"]["mtp.extra.weight"] = shard_name
        shard_headers[shard_name]["mtp.extra.weight"] = {
            "dtype": "BF16",
            "shape": [1],
            "data_offsets": [0, 2],
        }
    elif mutation == "dtype":
        metadata = shard_headers[shard_name][source_name]
        metadata["dtype"] = "F32"
        start = metadata["data_offsets"][0]
        metadata["data_offsets"][1] = start + (
            metadata["shape"][0]
            * metadata["shape"][1]
            * 4
        )
    elif mutation == "shape":
        metadata = shard_headers[shard_name][source_name]
        metadata["shape"][0] += 1
        start = metadata["data_offsets"][0]
        metadata["data_offsets"][1] = start + (
            metadata["shape"][0]
            * metadata["shape"][1]
            * 2
        )
    else:
        other_shard = next(
            name
            for name in shard_headers
            if name != shard_name
        )
        shard_headers[other_shard][source_name] = (
            shard_headers[shard_name].pop(source_name)
        )

    with pytest.raises(ValueError, match=match):
        build_qwen35_mtp_checkpoint_plan(
            _config(),
            index_payload,
            shard_headers,
        )


def test_target_plan_exposes_exact_sorted_mtp_skips():
    plan = Qwen35CheckpointWeightPlan(
        loads=(),
        skips=(
            Qwen35CheckpointSkip(
                Qwen35CheckpointSource(
                    "model.visual.weight",
                    "model.safetensors",
                ),
                "visual",
            ),
            Qwen35CheckpointSkip(
                Qwen35CheckpointSource(
                    "mtp.z.weight",
                    "model.safetensors",
                ),
                "mtp",
            ),
            Qwen35CheckpointSkip(
                Qwen35CheckpointSource(
                    "mtp.a.weight",
                    "model.safetensors",
                ),
                "mtp",
            ),
        ),
        shards=("model.safetensors",),
    )

    assert qwen35_mtp_skip_source_names(plan) == (
        "mtp.a.weight",
        "mtp.z.weight",
    )


class _FakeSource:
    def __init__(self, tensor, *, wrong_shape=False):
        self.source_name = tensor.source_name
        self.shape = (
            (1,)
            if wrong_shape
            else tensor.shape
        )
        self.dtype = "torch.bfloat16"
        self.payload = tensor.source_name


class _FakeParameter:
    def __init__(self, *, fail_source=None):
        self.state = {}
        self.fail_source = fail_source
        self.data = self
        self.weight_loader = self._weight_loader

    def detach(self):
        return self

    def clone(self):
        snapshot = _FakeParameter()
        snapshot.state = deepcopy(self.state)
        return snapshot

    def copy_(self, other):
        self.state = deepcopy(other.state)
        return self

    def _weight_loader(self, parameter, source, packed_slot=None):
        if source.source_name == self.fail_source:
            raise RuntimeError("assignment failed")
        key = (
            "value"
            if packed_slot is None
            else packed_slot
        )
        parameter.state[key] = source.payload


class _FakePlainParameter(_FakeParameter):
    def __init__(self):
        super().__init__()
        del self.weight_loader

    def clone(self):
        snapshot = _FakePlainParameter()
        snapshot.state = deepcopy(self.state)
        return snapshot

    def copy_(self, other):
        if isinstance(other, _FakeSource):
            self.state["value"] = other.payload
        else:
            self.state = deepcopy(other.state)
        return self


def _module_for_plan(plan, *, fail_source=None):
    root = SimpleNamespace()
    destinations = {}
    for tensor in plan.tensors:
        path = tensor.destination_path.split(".")
        parent = root
        for part in path[:-1]:
            if not hasattr(parent, part):
                setattr(parent, part, SimpleNamespace())
            parent = getattr(parent, part)
        destination = destinations.get(
            tensor.destination_path
        )
        if destination is None:
            destination = _FakeParameter(
                fail_source=fail_source,
            )
            destinations[tensor.destination_path] = destination
        setattr(parent, path[-1], destination)
    return root, destinations


def _replace_destination(module, path, destination):
    parts = path.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], destination)


def test_binding_resolves_all_destinations_before_reading():
    plan = _plan()
    module, _ = _module_for_plan(plan)
    first_path = plan.tensors[0].destination_path.split(".")
    parent = module
    for part in first_path[:-1]:
        parent = getattr(parent, part)
    delattr(parent, first_path[-1])
    reads = []

    with pytest.raises(ValueError, match="destination"):
        bind_qwen35_mtp_checkpoint(
            module,
            plan,
            lambda tensor: reads.append(tensor.source_name),
        )

    assert reads == []


def test_binding_uses_default_copy_for_unpacked_plain_parameter():
    plan = _plan()
    module, _ = _module_for_plan(plan)
    tensor = next(
        tensor
        for tensor in plan.tensors
        if tensor.packed_slot is None
    )
    destination = _FakePlainParameter()
    _replace_destination(
        module,
        tensor.destination_path,
        destination,
    )

    bind_qwen35_mtp_checkpoint(
        module,
        plan,
        lambda row: _FakeSource(row),
    )

    assert destination.state == {
        "value": tensor.source_name,
    }


def test_binding_requires_custom_loader_for_packed_parameter():
    plan = _plan()
    module, _ = _module_for_plan(plan)
    tensor = next(
        tensor
        for tensor in plan.tensors
        if tensor.packed_slot is not None
    )
    _replace_destination(
        module,
        tensor.destination_path,
        _FakePlainParameter(),
    )

    with pytest.raises(ValueError, match="weight_loader"):
        bind_qwen35_mtp_checkpoint(
            module,
            plan,
            lambda row: _FakeSource(row),
        )


def test_binding_reads_all_sources_before_assignment():
    plan = _plan()
    module, destinations = _module_for_plan(plan)
    reads = []
    fail_name = plan.tensors[1].source_name

    def reader(tensor):
        reads.append(tensor.source_name)
        if tensor.source_name == fail_name:
            raise RuntimeError("read failed")
        return _FakeSource(tensor)

    with pytest.raises(RuntimeError, match="read failed"):
        bind_qwen35_mtp_checkpoint(module, plan, reader)

    assert all(not destination.state for destination in destinations.values())


def test_binding_rolls_back_every_destination_on_assignment_failure():
    plan = _plan()
    fail_name = plan.tensors[-1].source_name
    module, destinations = _module_for_plan(
        plan,
        fail_source=fail_name,
    )
    for destination in destinations.values():
        destination.state = {"original": True}

    with pytest.raises(RuntimeError, match="assignment failed"):
        bind_qwen35_mtp_checkpoint(
            module,
            plan,
            lambda tensor: _FakeSource(tensor),
        )

    assert all(
        destination.state == {"original": True}
        for destination in destinations.values()
    )


def test_binding_returns_exact_sorted_loaded_source_set():
    plan = _plan()
    module, destinations = _module_for_plan(plan)

    loaded = bind_qwen35_mtp_checkpoint(
        module,
        plan,
        lambda tensor: _FakeSource(tensor),
    )

    assert loaded == tuple(sorted(EXPECTED_MTP))
    assert any(
        destination.state
        for destination in destinations.values()
    )


def test_binding_rejects_source_shape_before_assignment():
    plan = _plan()
    module, destinations = _module_for_plan(plan)
    wrong_name = plan.tensors[3].source_name

    with pytest.raises(ValueError, match="shape"):
        bind_qwen35_mtp_checkpoint(
            module,
            plan,
            lambda tensor: _FakeSource(
                tensor,
                wrong_shape=(
                    tensor.source_name == wrong_name
                ),
            ),
        )

    assert all(not destination.state for destination in destinations.values())
