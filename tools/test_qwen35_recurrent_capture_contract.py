from copy import deepcopy
from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT / "tinyvllm/engine/qwen35_recurrent_capture_contract.py"
)
SPEC = importlib.util.spec_from_file_location(
    "qwen35_recurrent_capture_contract",
    CONTRACT_PATH,
)
contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contract
SPEC.loader.exec_module(contract)


def valid_identity_dict():
    return {
        "schema_version": contract.CAPTURE_IDENTITY_SCHEMA_VERSION,
        "model_manifest_sha256": "a" * 64,
        "source_tree_sha256": "b" * 64,
        "workload_manifest_sha256": "c" * 64,
        "world_size": 2,
        "workload_ids": ["w0", "w1"],
        "linear_layer_indices": [0, 2],
    }


def valid_tensor_record_dict():
    return {
        "tensor_id": "rank0:w0:layer0:linear_recurrent",
        "rank": 0,
        "workload_id": "w0",
        "layer_index": 0,
        "relative_path": "rank0/tensors/w0/layer0.pt",
        "sha256": "d" * 64,
        "shape": [2, 3, 8],
        "dtype": "float32",
        "logical_bytes": 192,
    }


def valid_rank_manifest_dict():
    identity = valid_identity_dict()
    tensors = []
    for workload_id in ("w0", "w1"):
        for layer_index in (0, 2):
            tensors.append({
                "tensor_id": (
                    f"rank0:{workload_id}:layer{layer_index}:"
                    "linear_recurrent"
                ),
                "rank": 0,
                "workload_id": workload_id,
                "layer_index": layer_index,
                "relative_path": (
                    f"rank0/tensors/{workload_id}/layer{layer_index}.pt"
                ),
                "sha256": "d" * 64,
                "shape": [2, 3, 8],
                "dtype": "float32",
                "logical_bytes": 192,
            })
    return {
        "schema_version": contract.RANK_CAPTURE_MANIFEST_SCHEMA_VERSION,
        "identity": identity,
        "rank": 0,
        "tensors": tensors,
    }


def test_run_identity_and_expected_tensor_ids_are_exact():
    identity = contract.validate_run_identity(valid_identity_dict())
    assert identity.world_size == 2
    assert identity.payload() == valid_identity_dict()
    assert contract.expected_tensor_ids(
        world_size=2,
        workload_ids=("w0", "w1"),
        linear_layer_indices=(0, 2),
    ) == (
        "rank0:w0:layer0:linear_recurrent",
        "rank0:w0:layer2:linear_recurrent",
        "rank0:w1:layer0:linear_recurrent",
        "rank0:w1:layer2:linear_recurrent",
        "rank1:w0:layer0:linear_recurrent",
        "rank1:w0:layer2:linear_recurrent",
        "rank1:w1:layer0:linear_recurrent",
        "rank1:w1:layer2:linear_recurrent",
    )
    with pytest.raises(FrozenInstanceError):
        identity.world_size = 4


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(extra=True),
    lambda value: value.update(world_size=True),
    lambda value: value.update(workload_ids=["w0", "w0"]),
    lambda value: value.update(linear_layer_indices=[2, 0]),
    lambda value: value.update(source_tree_sha256="not-a-hash"),
])
def test_run_identity_rejects_unknown_or_noncanonical_fields(mutation):
    value = valid_identity_dict()
    mutation(value)
    with pytest.raises(ValueError):
        contract.validate_run_identity(value)


def test_canonical_json_is_stable_and_strict():
    left = {"z": [2, 1], "a": "value"}
    right = {"a": "value", "z": [2, 1]}
    assert contract.canonical_json_bytes(left) == (
        b'{"a":"value","z":[2,1]}'
    )
    assert (
        contract.canonical_json_sha256(left)
        == contract.canonical_json_sha256(right)
    )
    with pytest.raises(ValueError):
        contract.canonical_json_bytes({"value": float("nan")})


def test_tensor_record_requires_exact_canonical_metadata():
    identity = contract.validate_run_identity(valid_identity_dict())
    record = contract.validate_tensor_record(
        valid_tensor_record_dict(),
        identity=identity,
        expected_rank=0,
    )
    assert record.payload() == valid_tensor_record_dict()
    assert record.shape == (2, 3, 8)
    with pytest.raises(FrozenInstanceError):
        record.logical_bytes = 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tensor_id", "rank0:w0:layer2:linear_recurrent"),
        ("rank", True),
        ("rank", 2),
        ("workload_id", "undeclared"),
        ("layer_index", 1),
        ("relative_path", "/rank0/tensors/w0/layer0.pt"),
        ("relative_path", "rank0/tensors/w0/../layer0.pt"),
        ("relative_path", "rank0//tensors/w0/layer0.pt"),
        ("relative_path", "rank0\\tensors\\w0\\layer0.pt"),
        ("relative_path", "rank1/tensors/w0/layer0.pt"),
        ("sha256", "not-a-hash"),
        ("shape", [2, 3]),
        ("shape", [2, True, 8]),
        ("dtype", "float16"),
        ("logical_bytes", 191),
    ],
)
def test_tensor_record_rejects_inconsistent_or_noncanonical_fields(
    field,
    value,
):
    identity = contract.validate_run_identity(valid_identity_dict())
    tensor = valid_tensor_record_dict()
    tensor[field] = value
    with pytest.raises(ValueError):
        contract.validate_tensor_record(
            tensor,
            identity=identity,
            expected_rank=0,
        )


def test_tensor_record_rejects_unknown_fields():
    identity = contract.validate_run_identity(valid_identity_dict())
    tensor = valid_tensor_record_dict()
    tensor["extra"] = True
    with pytest.raises(ValueError):
        contract.validate_tensor_record(tensor, identity=identity)


def test_rank_manifest_is_exact_ordered_and_identity_bound():
    payload = valid_rank_manifest_dict()
    manifest = contract.validate_rank_manifest(payload)
    assert manifest.payload() == payload
    assert manifest.rank == 0
    assert manifest.identity.payload() == valid_identity_dict()
    assert tuple(row.tensor_id for row in manifest.tensors) == (
        "rank0:w0:layer0:linear_recurrent",
        "rank0:w0:layer2:linear_recurrent",
        "rank0:w1:layer0:linear_recurrent",
        "rank0:w1:layer2:linear_recurrent",
    )
    with pytest.raises(FrozenInstanceError):
        manifest.rank = 1

    expected_identity = contract.validate_run_identity(
        valid_identity_dict()
    )
    assert contract.validate_rank_manifest(
        payload,
        expected_identity=expected_identity,
    ) == manifest


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra=True),
        lambda value: value.update(rank=True),
        lambda value: value.update(rank=2),
        lambda value: value.update(tensors=[]),
        lambda value: value["tensors"].append(
            deepcopy(value["tensors"][0])
        ),
        lambda value: value["tensors"].reverse(),
    ],
)
def test_rank_manifest_rejects_unknown_or_noncanonical_fields(mutation):
    value = valid_rank_manifest_dict()
    mutation(value)
    with pytest.raises(ValueError):
        contract.validate_rank_manifest(value)


def test_rank_manifest_rejects_mismatched_expected_identity():
    value = valid_rank_manifest_dict()
    expected = valid_identity_dict()
    expected["source_tree_sha256"] = "e" * 64
    with pytest.raises(ValueError):
        contract.validate_rank_manifest(
            value,
            expected_identity=contract.validate_run_identity(expected),
        )
