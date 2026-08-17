from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tools/qwen35_real_checkpoint_heterogeneous_two_layer_preflight.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_heterogeneous_two_layer_preflight_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expect_error(callback, message):
    try:
        callback()
    except (TypeError, ValueError, RuntimeError, OSError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def _memory(tp_size):
    if tp_size == 1:
        points = (24000, 369000, 500000, 510000, 4192000, 4490000)
    else:
        points = (24000, 369000, 500000, 505000, 2348000, 2580000)
    names = (
        "before",
        "after_torch",
        "after_metadata",
        "after_pool",
        "after_target",
        "after_payload",
    )
    return {
        name: {"vmrss_kib": value, "vmhwm_kib": value}
        for name, value in zip(names, points, strict=True)
    }


def _valid_binding_result(module, index, tp_size):
    digest = f"{index:064x}"[-64:]
    contract = module.binding_contract(index, tp_size)
    return {
        **{
            name: contract[name]
            for name in (
                "binding_index",
                "layer_index",
                "source_name",
                "target",
                "kind",
                "dtype",
                "local_shape",
                "destination_slice",
                "tile_count",
                "range_count",
                "byte_count",
            )
        },
        "production_sha256": digest,
        "verifier_sha256": digest,
        "source_tensor_sha256": digest,
        "destination_sha256": digest,
        "coverage_complete": True,
    }


def _valid_row(module, tp_size, tp_rank, process_id):
    contract = module.TWO_LAYER_CONTRACTS[(tp_size, tp_rank)]
    memory = _memory(tp_size)
    digest = "f" * 64
    return {
        "schema_version": module.ROW_SCHEMA_VERSION,
        "status": "PASS",
        "tp_size": tp_size,
        "tp_rank": tp_rank,
        "process_id": process_id,
        "observed_user": "sitian",
        "observed_hostname": "n232-195-203",
        "checkpoint_dir": module.APPROVED_MODEL_DIR,
        "config_sha256": module.APPROVED_CONFIG_SHA256,
        "index_sha256": module.APPROVED_INDEX_SHA256,
        "config_index_header_sha256": module.APPROVED_COMPOSITE_SHA256,
        "metadata_bytes_read": 144024,
        "selected_layer_indices": [0, 3],
        "selected_layer_types": ["linear_attention", "full_attention"],
        "selected_binding_indices": list(
            module.SELECTED_BINDING_INDICES
        ),
        "selected_binding_count": 25,
        "unique_destination_count": 23,
        "alias_groups": [[12, 13], [229, 230]],
        "tile_count": contract["tile_count"],
        "kind_counts": contract["kind_counts"],
        "ranges_per_pass": contract["ranges_per_pass"],
        "production_payload_bytes_read": contract["bytes_per_pass"],
        "verifier_payload_bytes_read": contract["bytes_per_pass"],
        "logical_payload_bytes_read": contract["logical_bytes"],
        "open_count": 2,
        "pread_count": contract["pread_count"],
        "binding_results": [
            _valid_binding_result(module, index, tp_size)
            for index in module.SELECTED_BINDING_INDICES
        ],
        "layer_results": [
            {
                "layer_index": layer_index,
                "layer_type": layer_type,
                "binding_indices": list(binding_indices),
                "binding_count": len(binding_indices),
                "tile_count": contract["layers"][layer_index][
                    "tile_count"
                ],
                "range_count": contract["layers"][layer_index][
                    "range_count"
                ],
                "byte_count": contract["layers"][layer_index][
                    "byte_count"
                ],
                "production_sha256": digest,
                "verifier_sha256": digest,
                "destination_sha256": digest,
                "coverage_complete": True,
            }
            for layer_index, layer_type, binding_indices in (
                (0, "linear_attention", tuple(range(1, 15))),
                (3, "full_attention", tuple(range(227, 238))),
            )
        ],
        "layer_completion_order": [0, 3],
        "layer0_changed_before_layer3": True,
        "layer3_zero_before_first_copy": True,
        "aggregate_source_sha256": digest,
        "aggregate_destination_sha256": digest,
        "selected_destinations_changed": True,
        "non_selected_tensors_remained_zero": True,
        "rollback_binding_order": list(
            reversed(module.UNIQUE_BINDING_ORDER)
        ),
        "all_selected_snapshots_restored": True,
        "all_unique_tensors_zero_after_rollback": True,
        "loader_call_count": 0,
        "assignment_call_count": 0,
        "target_take_count": 0,
        "model_forward_count": 0,
        "attention_forward_count": 0,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "memory": memory,
        "total_vmhwm_increment_kib": (
            memory["after_payload"]["vmhwm_kib"]
            - memory["before"]["vmhwm_kib"]
        ),
        "post_torch_vmhwm_increment_kib": (
            memory["after_payload"]["vmhwm_kib"]
            - memory["after_torch"]["vmhwm_kib"]
        ),
        "post_metadata_vmhwm_increment_kib": (
            memory["after_payload"]["vmhwm_kib"]
            - memory["after_metadata"]["vmhwm_kib"]
        ),
    }


def _valid_record(module):
    hashes = {name: "a" * 64 for name in module.SOURCE_FILES}
    return {
        "schema_version": module.SCHEMA_VERSION,
        "status": "PASS",
        "remote_target": module.REMOTE_TARGET,
        "remote_python": module.REMOTE_PYTHON,
        "checkpoint_dir": module.APPROVED_MODEL_DIR,
        "model_manifest_sha256": module.APPROVED_MODEL_MANIFEST_SHA256,
        "config_sha256": module.APPROVED_CONFIG_SHA256,
        "index_sha256": module.APPROVED_INDEX_SHA256,
        "config_index_header_sha256": module.APPROVED_COMPOSITE_SHA256,
        "payload_identity_source": "retained_approved_manifest",
        "fresh_process_per_rank": True,
        "source_file_sha256": hashes,
        "source_tree_sha256": module._source_tree_sha256(hashes),
        "rows": [
            _valid_row(module, 1, 0, 401),
            _valid_row(module, 2, 0, 402),
            _valid_row(module, 2, 1, 403),
        ],
    }


def _tile(module, binding_index, layer_index, destination, slices):
    shape = tuple(
        item.stop - item.start
        for item in slices
    )
    return module.Qwen35CheckpointTile(
        binding_index,
        f"source-{binding_index}",
        module.APPROVED_SHARD_NAME,
        shape,
        tuple(slice(0, size) for size in shape),
        shape,
        destination,
        slices,
        shape,
        destination.dtype,
        destination.element_size()
        * int(torch.tensor(shape).prod().item()),
        f"layers.{layer_index}.target",
        "axis0",
    )


def test_frozen_two_layer_contracts_and_record_validation():
    module = _load_module()
    assert len(module.SOURCE_FILES) == 41
    assert len(set(module.SOURCE_FILES)) == 41
    assert module.SELECTED_BINDING_INDICES == (
        *range(1, 15),
        *range(227, 238),
    )
    assert module.UNIQUE_BINDING_ORDER == (
        *range(1, 13),
        14,
        227,
        228,
        229,
        *range(231, 238),
    )
    expected = {
        (1, 0): (3456, 222496352, 3456, 6912, 444992704),
        (2, 0): (1734, 111257136, 9388, 18776, 222514272),
        (2, 1): (1734, 111257136, 9388, 18776, 222514272),
    }
    for tp, values in expected.items():
        contract = module.TWO_LAYER_CONTRACTS[tp]
        assert (
            contract["tile_count"],
            contract["bytes_per_pass"],
            contract["ranges_per_pass"],
            contract["pread_count"],
            contract["logical_bytes"],
        ) == values
    assert module.validate_heterogeneous_two_layer_preflight(
        _valid_record(module)
    )["status"] == "PASS"
    invalid = _valid_record(module)
    invalid["rows"][0]["selected_layer_types"] = [
        "linear_attention",
        "linear_attention",
    ]
    _expect_error(
        lambda: module.validate_heterogeneous_two_layer_preflight(invalid),
        "layer types",
    )


def test_binding_contracts_cover_full_attention_and_shared_slices():
    module = _load_module()
    assert module.binding_contract(232, 1)["target"] == (
        "layers.3.full_attention.k_norm.weight"
    )
    assert module.binding_contract(234, 2)["kind"] == "axis1"
    assert module.binding_contract(236, 2)["local_shape"] == [2048, 2048]
    assert module.binding_contract(229, 1)["destination_slice"] == [0, 6144]
    assert module.binding_contract(230, 1)["destination_slice"] == [
        6144,
        6144,
    ]
    assert module.binding_contract(229, 2)["destination_slice"] == [0, 3072]
    assert module.binding_contract(230, 2)["destination_slice"] == [
        3072,
        3072,
    ]
    _expect_error(lambda: module.binding_contract(15, 1), "binding")


def test_selected_binding_indices_use_exact_layer_prefixes():
    module = _load_module()
    bindings = tuple(
        SimpleNamespace(
            load=SimpleNamespace(
                weight=SimpleNamespace(target=target),
            ),
        )
        for target in (
            "embed_tokens.weight",
            "layers.0.input_layernorm.weight",
            "layers.1.input_layernorm.weight",
            "layers.3.full_attention.q_projection.weight",
            "final_norm.weight",
        )
    )
    assert module._selected_binding_indices(bindings) == (1, 3)


def test_two_layer_isolation_aliases_and_reverse_rollback():
    module = _load_module()
    layer0 = torch.zeros((2,), dtype=torch.bfloat16)
    shared0 = torch.zeros((4,), dtype=torch.bfloat16)
    layer3 = torch.zeros((2,), dtype=torch.bfloat16)
    shared3 = torch.zeros((4,), dtype=torch.bfloat16)
    other = torch.zeros((3,), dtype=torch.float32)
    tiles = (
        _tile(module, 1, 0, layer0, (slice(0, 2),)),
        _tile(module, 12, 0, shared0, (slice(0, 2),)),
        _tile(module, 13, 0, shared0, (slice(2, 4),)),
        _tile(module, 227, 3, layer3, (slice(0, 2),)),
        _tile(module, 229, 3, shared3, (slice(0, 2),)),
        _tile(module, 230, 3, shared3, (slice(2, 4),)),
    )
    payloads = tuple(
        torch.full(tile.tile_shape, index + 1, dtype=tile.dtype)
        for index, tile in enumerate(tiles)
    )
    result = module.apply_verify_and_rollback_two_layer_tiles(
        tiles,
        payloads,
        binding_order=(1, 12, 13, 227, 229, 230),
        binding_layer={
            1: 0, 12: 0, 13: 0, 227: 3, 229: 3, 230: 3,
        },
        unique_tensors=(layer0, shared0, layer3, shared3, other),
        selected_destination_ids={
            id(layer0), id(shared0), id(layer3), id(shared3),
        },
    )
    assert result == {
        "unique_destination_count": 4,
        "alias_groups": [[12, 13], [229, 230]],
        "layer_completion_order": [0, 3],
        "layer0_changed_before_layer3": True,
        "layer3_zero_before_first_copy": True,
        "non_selected_tensors_remained_zero": True,
        "rollback_binding_order": [229, 227, 12, 1],
        "all_selected_snapshots_restored": True,
        "all_unique_tensors_zero_after_rollback": True,
    }
    _expect_error(
        lambda: module.apply_verify_and_rollback_two_layer_tiles(
            (tiles[0], tiles[3], tiles[1], tiles[2], tiles[4], tiles[5]),
            (payloads[0], payloads[3], payloads[1], payloads[2],
             payloads[4], payloads[5]),
            binding_order=(1, 227, 12, 13, 229, 230),
            binding_layer={
                1: 0, 12: 0, 13: 0, 227: 3, 229: 3, 230: 3,
            },
            unique_tensors=(layer0, shared0, layer3, shared3, other),
            selected_destination_ids={
                id(layer0), id(shared0), id(layer3), id(shared3),
            },
        ),
        "layer order",
    )


def test_row_validation_rejects_layer_isolation_and_rollback_drift():
    module = _load_module()
    row = _valid_row(module, 2, 1, 403)
    assert module.validate_heterogeneous_two_layer_row(row) == row
    cases = (
        ({"selected_binding_count": 24}, "binding"),
        ({"unique_destination_count": 24}, "destination"),
        ({"alias_groups": [[12, 13]]}, "alias"),
        ({"layer_completion_order": [3, 0]}, "layer order"),
        ({"layer0_changed_before_layer3": False}, "layer 0"),
        ({"layer3_zero_before_first_copy": False}, "layer 3"),
        ({"non_selected_tensors_remained_zero": False}, "non-selected"),
        ({"rollback_binding_order": [1]}, "rollback"),
        ({"all_selected_snapshots_restored": False}, "snapshots"),
        ({"target_take_count": 1}, "target.take"),
        ({"cuda_initialized_after": True}, "CUDA"),
    )
    for update, message in cases:
        invalid = dict(row)
        invalid.update(update)
        _expect_error(
            lambda invalid=invalid: (
                module.validate_heterogeneous_two_layer_row(invalid)
            ),
            message,
        )


def test_source_tar_is_exact_41_file_closure():
    module = _load_module()
    payload = module.build_source_tar(ROOT)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        assert tuple(archive.getnames()) == module.SOURCE_FILES
        assert len(archive.getmembers()) == 41
        assert all(member.mtime == 0 for member in archive.getmembers())


def test_remote_orchestration_and_partial_failure_are_atomic():
    module = _load_module()
    record = _valid_record(module)
    staged = {
        "remote_source_dir": "/remote/run/source",
        "local_file_sha256": record["source_file_sha256"],
        "remote_file_sha256": record["source_file_sha256"],
        "source_tree_sha256": record["source_tree_sha256"],
    }
    rows = iter(record["rows"])
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        text = command[-1]
        if "internal-rank-worker" in text:
            return subprocess.CompletedProcess(
                command, 0, json.dumps(next(rows)), ""
            )
        if "internal-finalize" in text:
            return subprocess.CompletedProcess(
                command, 0, json.dumps(record), ""
            )
        payload = json.loads(kwargs["input"])
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({
                "heterogeneous_two_layer_preflight": record,
                "source_manifest": payload["source_manifest"],
            }),
            "",
        )

    with tempfile.TemporaryDirectory() as temporary:
        result = module.run_remote_heterogeneous_two_layer_preflight(
            ROOT,
            "two-layer-test",
            staged=staged,
            local_run_root=Path(temporary),
            command_runner=runner,
        )
        destination = Path(temporary) / "two-layer-test"
        assert result == record
        assert {path.name for path in destination.iterdir()} == {
            "heterogeneous_two_layer_preflight.json",
            "source_manifest.json",
        }
    worker_commands = [
        command[-1]
        for command, _ in calls
        if "internal-rank-worker" in command[-1]
    ]
    assert len(worker_commands) == 3
    assert all("CUDA_VISIBLE_DEVICES=" in text for text in worker_commands)

    def fail_runner(command, **kwargs):
        if "internal-rank-worker" in command[-1]:
            return subprocess.CompletedProcess(command, 1, "", "injected")
        raise AssertionError("unexpected command after worker failure")

    with tempfile.TemporaryDirectory() as temporary:
        _expect_error(
            lambda: module.run_remote_heterogeneous_two_layer_preflight(
                ROOT,
                "two-layer-fail",
                staged=staged,
                local_run_root=Path(temporary),
                command_runner=fail_runner,
            ),
            "injected",
        )
        assert not (Path(temporary) / "two-layer-fail").exists()


def main():
    tests = (
        test_frozen_two_layer_contracts_and_record_validation,
        test_binding_contracts_cover_full_attention_and_shared_slices,
        test_selected_binding_indices_use_exact_layer_prefixes,
        test_two_layer_isolation_aliases_and_reverse_rollback,
        test_row_validation_rejects_layer_isolation_and_rollback_drift,
        test_source_tar_is_exact_41_file_closure,
        test_remote_orchestration_and_partial_failure_are_atomic,
    )
    for test in tests:
        test()
    print(
        "qwen35 heterogeneous two-layer preflight tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
