"""CPU-only unit tests for Qwen3.5 hybrid-state normalization helpers.

Run: python3 tools/test_qwen35_hybrid_state_probe.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, os.fspath(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_module(
    "qwen35_hybrid_state_contract_for_probe_tests",
    THIS_DIR / "qwen35_hybrid_state_contract.py",
)
probe = _load_module(
    "qwen35_hybrid_state_probe_under_test",
    THIS_DIR / "qwen35_hybrid_state_probe.py",
)


def _expect_value_error(callable_, message_fragment):
    try:
        callable_()
    except ValueError as exc:
        assert message_fragment in str(exc)
    else:
        raise AssertionError("expected ValueError")


@dataclass
class _DataclassState:
    recurrent_state: torch.Tensor
    convolution_state: torch.Tensor


_NamedTupleState = namedtuple(
    "_NamedTupleState",
    ("key_cache", "value_cache"),
)


class _AdapterState:
    def __init__(self, hidden, ignored):
        self.hidden = hidden
        self.ignored = ignored


def test_walk_tensor_leaves_preserves_explicit_paths_and_aliases():
    storage = torch.arange(8, dtype=torch.float32)
    state = {"layers": [{"key": storage[:4], "value": storage[4:]}]}
    leaves = list(probe.walk_tensor_leaves(state))
    assert [path for path, _ in leaves] == [
        "layers[0].key",
        "layers[0].value",
    ]
    assert leaves[0][1].untyped_storage().data_ptr() == (
        leaves[1][1].untyped_storage().data_ptr()
    )


def test_walk_tensor_leaves_uses_frozen_container_order():
    state = {
        "z": torch.tensor([6]),
        "a": _DataclassState(
            recurrent_state=torch.tensor([1]),
            convolution_state=torch.tensor([2]),
        ),
        "named": _NamedTupleState(
            key_cache=torch.tensor([3]),
            value_cache=torch.tensor([4]),
        ),
        "sequence": [torch.tensor([5])],
    }
    paths = [path for path, _ in probe.walk_tensor_leaves(state)]
    assert paths == [
        "a.recurrent_state",
        "a.convolution_state",
        "named.key_cache",
        "named.value_cache",
        "sequence[0]",
        "z",
    ]


def test_walk_tensor_leaves_rejects_arbitrary_object_attributes():
    state = _AdapterState(
        hidden=torch.tensor([1]),
        ignored=torch.tensor([2]),
    )
    _expect_value_error(
        lambda: list(probe.walk_tensor_leaves(state)),
        "adapter",
    )
    leaves = list(probe.walk_tensor_leaves(
        state,
        adapter_registry={
            _AdapterState: lambda value: {"hidden": value.hidden},
        },
    ))
    assert [path for path, _ in leaves] == ["hidden"]


def test_classify_state_role_covers_frozen_role_domain():
    cases = (
        (
            "layers[0].key_cache",
            "full_attention",
            None,
            "full_attention_key",
        ),
        (
            "layers[0].value_cache",
            "full_attention",
            None,
            "full_attention_value",
        ),
        (
            "layers[0].recurrent_state",
            "linear_attention",
            None,
            "linear_recurrent_state",
        ),
        (
            "layers[0].convolution_state",
            "linear_attention",
            None,
            "linear_convolution_state",
        ),
        (
            "cache_position",
            "metadata",
            None,
            "position_or_sequence_metadata",
        ),
        (
            "layers[0].opaque",
            "linear_attention",
            "adapter_hidden",
            "other_persistent_state",
        ),
    )
    observed = {
        probe.classify_state_role(
            path,
            declared_layer_type=layer_type,
            component_name=component_name,
        )
        for path, layer_type, component_name, _ in cases
    }
    assert observed == set(contract.STATE_ROLES)
    for path, layer_type, component_name, expected in cases:
        assert probe.classify_state_role(
            path,
            declared_layer_type=layer_type,
            component_name=component_name,
        ) == expected


def test_normalization_assigns_request_generation_and_storage_identity():
    tensor = torch.zeros((1, 2, 3), dtype=torch.float32)
    rows = probe.normalize_state_components(
        state={"recurrent_state": tensor},
        request_id="request-a",
        request_generation=2,
        sequence_length=17,
        lifetime_epoch=3,
        layer_schedule={0: "linear_attention"},
    )
    assert rows[0]["request_generation"] == 2
    assert rows[0]["layer_index"] == 0
    assert rows[0]["state_role"] == "linear_recurrent_state"
    assert rows[0]["logical_bytes"] == tensor.numel() * tensor.element_size()
    assert rows[0]["storage_identity"]
    assert len(rows[0]["content_sha256"]) == 64
    assert set(rows[0]) == {
        field.name for field in contract.StateComponent.__dataclass_fields__.values()
    }


def test_normalization_preserves_alias_storage_and_unknown_roles():
    storage = torch.arange(8, dtype=torch.float32)
    rows = probe.normalize_state_components(
        state={
            "layers": [{
                "key_cache": storage[:4],
                "mystery": storage[4:],
            }],
        },
        request_id="request-a",
        request_generation=0,
        sequence_length=4,
        lifetime_epoch=1,
        layer_schedule={0: "full_attention"},
    )
    assert rows[0]["state_role"] == "full_attention_key"
    assert rows[1]["state_role"] == "other_persistent_state"
    assert rows[0]["storage_identity"] == rows[1]["storage_identity"]
    assert rows[0]["storage_nbytes"] == rows[1]["storage_nbytes"]


def _component(
    *,
    role,
    path,
    shape,
    storage_identity,
    content_sha256,
    storage_offset=0,
    storage_nbytes=64,
    generation=0,
):
    dtype = "float32"
    return {
        "request_id": "request-a",
        "request_generation": generation,
        "layer_index": 0,
        "declared_layer_type": "linear_attention",
        "state_role": role,
        "tensor_path": path,
        "shape": list(shape),
        "stride": [1] * len(shape),
        "dtype": dtype,
        "device": "cpu",
        "requires_grad": False,
        "logical_numel": 1 if not shape else int(torch.tensor(shape).prod()),
        "logical_bytes": contract.logical_bytes(tuple(shape), dtype),
        "storage_data_ptr": 1,
        "storage_offset": storage_offset,
        "storage_nbytes": storage_nbytes,
        "storage_identity": storage_identity,
        "lifetime_epoch": 1,
        "sequence_length": 17,
        "update_kind": "created",
        "content_sha256": content_sha256,
    }


def _synthetic_components():
    return [
        _component(
            role="full_attention_key",
            path="layers[0].key_cache",
            shape=(1, 4),
            storage_identity="key-storage",
            content_sha256="a" * 64,
        ),
        _component(
            role="linear_recurrent_state",
            path="layers[0].recurrent_state",
            shape=(1, 4),
            storage_identity="recurrent-storage",
            content_sha256="b" * 64,
        ),
        _component(
            role="linear_convolution_state",
            path="layers[0].convolution_state",
            shape=(1, 4),
            storage_identity="convolution-storage",
            content_sha256="c" * 64,
        ),
        _component(
            role="position_or_sequence_metadata",
            path="sequence_length",
            shape=(1,),
            storage_identity="metadata-storage",
            content_sha256="d" * 64,
        ),
    ]


def test_snapshot_comparison_distinguishes_growth_replacement_and_in_place():
    previous = _synthetic_components()
    current = [dict(row) for row in previous]
    current[0]["shape"] = [1, 5]
    current[0]["logical_numel"] = 5
    current[0]["logical_bytes"] = 20
    current[0]["content_sha256"] = "e" * 64
    current[1]["content_sha256"] = "f" * 64
    current[2]["storage_identity"] = "new-convolution-storage"
    current[2]["storage_data_ptr"] = 2
    current[2]["content_sha256"] = "0" * 64
    transitions = probe.compare_state_snapshots(previous, current)
    assert transitions["full_attention_key"] == "grown"
    assert transitions["linear_recurrent_state"] == "mutated_in_place"
    assert transitions["linear_convolution_state"] == "replaced"
    assert transitions["position_or_sequence_metadata"] == "unchanged"


def test_snapshot_comparison_emits_created_and_released():
    previous = _synthetic_components()
    current = [dict(row) for row in previous[1:]]
    current.append(_component(
        role="full_attention_value",
        path="layers[0].value_cache",
        shape=(1, 4),
        storage_identity="value-storage",
        content_sha256="1" * 64,
    ))
    transitions = probe.compare_state_snapshots(previous, current)
    assert transitions["full_attention_key"] == "released"
    assert transitions["full_attention_value"] == "created"


def test_snapshot_comparison_rejects_generation_aliasing():
    previous = _synthetic_components()
    current = [dict(row) for row in previous]
    current.append(dict(current[0], request_generation=1))
    _expect_value_error(
        lambda: probe.compare_state_snapshots(previous, current),
        "request generation",
    )


def test_export_import_round_trip_is_ordered_by_request_layer_and_role():
    components = [
        dict(_synthetic_components()[2], layer_index=2),
        dict(_synthetic_components()[0], layer_index=0),
        dict(_synthetic_components()[1], layer_index=1),
    ]
    payload = probe.export_normalized_state(components)
    restored = probe.import_normalized_state(payload)
    assert [item["layer_index"] for item in restored] == sorted(
        item["layer_index"] for item in restored
    )
    assert contract.canonical_json_sha256(restored) == (
        contract.canonical_json_sha256(
            probe.export_normalized_state(restored)["components"]
        )
    )


def test_export_import_rejects_wrong_schema_and_duplicate_keys():
    payload = probe.export_normalized_state(_synthetic_components())
    _expect_value_error(
        lambda: probe.import_normalized_state(
            dict(payload, schema_version=999)
        ),
        "schema_version",
    )
    duplicated = list(payload["components"])
    duplicated.append(dict(duplicated[0]))
    _expect_value_error(
        lambda: probe.import_normalized_state(
            dict(payload, components=duplicated)
        ),
        "duplicate",
    )


def test_atomic_json_and_jsonl_writers_leave_no_partial_files():
    component = _synthetic_components()[0]
    snapshot = {
        "snapshot_id": "snapshot-0",
        "request_id": "request-a",
        "request_generation": 0,
        "lifetime_epoch": 1,
        "sequence_length": 17,
        "component_count": 1,
        "component_sha256": contract.canonical_json_sha256([component]),
    }
    memory = {
        "snapshot_id": "memory-0",
        "phase": "after_prefill",
        "request_id": "request-a",
        "request_generation": 0,
        "cuda_allocated_bytes": 0,
        "cuda_reserved_bytes": 0,
        "logical_state_bytes": component["logical_bytes"],
        "unique_storage_bytes": component["storage_nbytes"],
    }
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary)
        probe.write_json_atomic(
            run_dir / "summary.json",
            {"schema_version": contract.SCHEMA_VERSION},
        )
        probe.write_jsonl_atomic(
            run_dir / "state_snapshots.jsonl",
            [snapshot],
            required_fields=probe.STATE_SNAPSHOT_FIELDS,
        )
        probe.write_jsonl_atomic(
            run_dir / "state_components.jsonl",
            [component],
            required_fields=tuple(
                contract.StateComponent.__dataclass_fields__
            ),
        )
        probe.write_jsonl_atomic(
            run_dir / "memory_snapshots.jsonl",
            [memory],
            required_fields=probe.MEMORY_SNAPSHOT_FIELDS,
        )
        assert not list(run_dir.glob("*.partial"))
        assert json.loads((run_dir / "summary.json").read_text()) == {
            "schema_version": contract.SCHEMA_VERSION,
        }
        for filename, fields in (
            ("state_snapshots.jsonl", probe.STATE_SNAPSHOT_FIELDS),
            (
                "state_components.jsonl",
                tuple(contract.StateComponent.__dataclass_fields__),
            ),
            ("memory_snapshots.jsonl", probe.MEMORY_SNAPSHOT_FIELDS),
        ):
            lines = (run_dir / filename).read_text().splitlines()
            assert lines
            for line in lines:
                assert set(json.loads(line)) == set(fields)


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print("qwen35 hybrid-state probe unit tests passed")


if __name__ == "__main__":
    main()
