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
    ROOT / "tools/qwen35_real_checkpoint_layer0_transaction_preflight.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_layer0_transaction_preflight_under_test",
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
        points = (23000, 368000, 499000, 509000, 4191000, 4360000)
    else:
        points = (23000, 368000, 499000, 504000, 2347000, 2440000)
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
        "binding_index": index,
        "source_name": contract["source_name"],
        "target": contract["target"],
        "kind": contract["kind"],
        "dtype": contract["dtype"],
        "local_shape": contract["local_shape"],
        "destination_slice": contract["destination_slice"],
        "tile_count": contract["tile_count"],
        "range_count": contract["range_count"],
        "byte_count": contract["byte_count"],
        "production_sha256": digest,
        "verifier_sha256": digest,
        "source_tensor_sha256": digest,
        "destination_sha256": digest,
        "coverage_complete": True,
    }


def _valid_row(module, tp_size, tp_rank, process_id):
    contract = module.LAYER_CONTRACTS[(tp_size, tp_rank)]
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
        "selected_binding_indices": list(module.SELECTED_BINDING_INDICES),
        "selected_binding_count": 14,
        "unique_destination_count": 13,
        "alias_groups": [[12, 13]],
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
        "aggregate_source_sha256": digest,
        "aggregate_destination_sha256": digest,
        "layer_destinations_changed": True,
        "non_layer_tensors_remained_zero": True,
        "rollback_binding_order": list(
            reversed(module.ROLLBACK_BINDING_ORDER)
        ),
        "all_layer_snapshots_restored": True,
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
            _valid_row(module, 1, 0, 301),
            _valid_row(module, 2, 0, 302),
            _valid_row(module, 2, 1, 303),
        ],
    }


def _fake_binding(
    shape,
    offsets,
    dtype,
    *,
    local_shape=None,
    destination_slice=None,
):
    return SimpleNamespace(
        load=SimpleNamespace(
            metadata=SimpleNamespace(
                shape=tuple(shape),
                data_offsets=tuple(offsets),
                dtype=dtype,
            ),
        ),
        local_shape=tuple(local_shape or shape),
        destination_slice=destination_slice,
    )


def _fake_tile(
    source_shape,
    source_slices,
    tile_shape,
    byte_count,
    kind,
):
    return SimpleNamespace(
        source_tensor_shape=tuple(source_shape),
        source_slices=tuple(source_slices),
        tile_shape=tuple(tile_shape),
        byte_count=byte_count,
        kind=kind,
    )


def test_frozen_layer_contracts_and_record_validation():
    module = _load_module()
    assert len(module.SOURCE_FILES) == 40
    assert len(set(module.SOURCE_FILES)) == 40
    assert module.SELECTED_BINDING_INDICES == tuple(range(1, 15))
    assert module.ROLLBACK_BINDING_ORDER == tuple(range(1, 13)) + (14,)
    expected = {
        (1, 0): (1826, 117629536, 1826, 3652, 235259072),
        (2, 0): (917, 58819120, 4744, 9488, 117638240),
        (2, 1): (917, 58819120, 4744, 9488, 117638240),
    }
    for tp, values in expected.items():
        contract = module.LAYER_CONTRACTS[tp]
        assert (
            contract["tile_count"],
            contract["bytes_per_pass"],
            contract["ranges_per_pass"],
            contract["pread_count"],
            contract["logical_bytes"],
        ) == values
    assert module.BINDING_CONTRACTS[12]["destination_slice_by_tp"] == {
        1: [0, 6144],
        2: [0, 3072],
    }
    assert module.BINDING_CONTRACTS[13]["destination_slice_by_tp"] == {
        1: [6144, 6144],
        2: [3072, 3072],
    }
    record = _valid_record(module)
    assert module.validate_layer0_transaction_preflight(record) == record
    duplicate = dict(record)
    duplicate["rows"] = [
        record["rows"][0],
        record["rows"][1],
        dict(record["rows"][2], process_id=302),
    ]
    _expect_error(
        lambda: module.validate_layer0_transaction_preflight(duplicate),
        "process IDs",
    )


def test_generic_tile_range_derivation_for_all_layouts():
    module = _load_module()
    data_start = 100
    cases = (
        (
            _fake_binding((8,), (20, 36), "BF16"),
            _fake_tile((8,), (slice(2, 6),), (4,), 8, "axis0"),
            ((124, 132),),
        ),
        (
            _fake_binding((4, 6), (40, 88), "BF16"),
            _fake_tile(
                (4, 6),
                (slice(1, 3), slice(0, 6)),
                (2, 6),
                24,
                "axis0",
            ),
            ((152, 176),),
        ),
        (
            _fake_binding((4, 6), (40, 88), "BF16"),
            _fake_tile(
                (4, 6),
                (slice(1, 3), slice(2, 5)),
                (2, 3),
                12,
                "axis1",
            ),
            ((156, 162), (168, 174)),
        ),
        (
            _fake_binding((4, 1, 3), (90, 114), "BF16"),
            _fake_tile(
                (4, 1, 3),
                (slice(1, 3), 0, slice(0, 3)),
                (2, 3),
                12,
                "squeeze_axis0",
            ),
            ((196, 208),),
        ),
    )
    for binding, tile, expected in cases:
        assert module.derive_tile_ranges(
            binding,
            tile,
            data_start=data_start,
        ) == expected
    _expect_error(
        lambda: module.derive_tile_ranges(
            _fake_binding((4, 6), (40, 88), "BF16"),
            _fake_tile(
                (4, 6),
                (slice(1, 3), slice(5, 2)),
                (2, 3),
                12,
                "axis1",
            ),
            data_start=data_start,
        ),
        "slice",
    )
    _expect_error(
        lambda: module.derive_tile_ranges(
            _fake_binding((4, 6), (40, 88), "BF16"),
            _fake_tile(
                (4, 6),
                (slice(1, 3), slice(2, 5)),
                (2, 3),
                10,
                "axis1",
            ),
            data_start=data_start,
        ),
        "byte count",
    )


def test_streaming_layer_transaction_covers_shared_destination_and_rolls_back():
    module = _load_module()
    destination_a = torch.zeros((4,), dtype=torch.bfloat16)
    destination_shared = torch.zeros((4,), dtype=torch.bfloat16)
    destination_b = torch.zeros((2, 2), dtype=torch.float32)
    other = torch.zeros((3,), dtype=torch.float32)
    tiles = (
        module.Qwen35CheckpointTile(
            1, "a", module.APPROVED_SHARD_NAME, (4,),
            (slice(0, 4),), (4,), destination_a,
            (slice(0, 4),), (4,), torch.bfloat16, 8, "a", "replicated",
        ),
        module.Qwen35CheckpointTile(
            12, "gate", module.APPROVED_SHARD_NAME, (2,),
            (slice(0, 2),), (2,), destination_shared,
            (slice(0, 2),), (2,), torch.bfloat16, 4,
            "shared", "axis0",
        ),
        module.Qwen35CheckpointTile(
            13, "up", module.APPROVED_SHARD_NAME, (2,),
            (slice(0, 2),), (2,), destination_shared,
            (slice(2, 4),), (2,), torch.bfloat16, 4,
            "shared", "axis0",
        ),
        module.Qwen35CheckpointTile(
            14, "b", module.APPROVED_SHARD_NAME, (2, 2),
            (slice(0, 2), slice(0, 2)), (2, 2), destination_b,
            (slice(0, 2), slice(0, 2)), (2, 2), torch.float32,
            16, "b", "axis0",
        ),
    )
    payloads = (
        torch.arange(4, dtype=torch.float32).to(torch.bfloat16),
        torch.tensor([5, 6], dtype=torch.bfloat16),
        torch.tensor([7, 8], dtype=torch.bfloat16),
        torch.arange(4, dtype=torch.float32).reshape(2, 2),
    )
    result = module.apply_verify_and_rollback_layer_tiles(
        tiles,
        payloads,
        binding_order=(1, 12, 13, 14),
        unique_tensors=(
            destination_a,
            destination_shared,
            destination_b,
            other,
        ),
        layer_destination_ids={
            id(destination_a),
            id(destination_shared),
            id(destination_b),
        },
    )
    assert result["unique_destination_count"] == 3
    assert result["alias_groups"] == [[12, 13]]
    assert result["non_layer_tensors_remained_zero"] is True
    assert result["rollback_binding_order"] == [14, 12, 1]
    assert result["all_layer_snapshots_restored"] is True
    assert result["all_unique_tensors_zero_after_rollback"] is True
    assert all(
        torch.count_nonzero(tensor).item() == 0
        for tensor in (destination_a, destination_shared, destination_b, other)
    )
    _expect_error(
        lambda: module.apply_verify_and_rollback_layer_tiles(
            (tiles[0], tiles[2], tiles[1], tiles[3]),
            (payloads[0], payloads[2], payloads[1], payloads[3]),
            binding_order=(1, 12, 13, 14),
            unique_tensors=(
                destination_a,
                destination_shared,
                destination_b,
                other,
            ),
            layer_destination_ids={
                id(destination_a),
                id(destination_shared),
                id(destination_b),
            },
        ),
        "order",
    )


def test_row_validation_rejects_coverage_alias_and_rollback_regressions():
    module = _load_module()
    row = _valid_row(module, 2, 1, 303)
    assert module.validate_layer0_transaction_row(row) == row
    assert module.validate_layer0_transaction_row(
        json.loads(json.dumps(row, sort_keys=True))
    )
    cases = (
        ({"selected_binding_indices": [1]}, "binding"),
        ({"unique_destination_count": 14}, "destination"),
        ({"alias_groups": []}, "alias"),
        ({"tile_count": 1}, "tile count"),
        ({"pread_count": 1}, "pread"),
        ({"non_layer_tensors_remained_zero": False}, "non-layer"),
        ({"rollback_binding_order": [1]}, "rollback"),
        ({"all_layer_snapshots_restored": False}, "snapshots"),
        ({"loader_call_count": 1}, "loader"),
        ({"cuda_initialized_after": True}, "CUDA"),
    )
    for updates, message in cases:
        invalid = dict(row)
        invalid.update(updates)
        _expect_error(
            lambda invalid=invalid: (
                module.validate_layer0_transaction_row(invalid)
            ),
            message,
        )
    invalid = json.loads(json.dumps(row))
    invalid["binding_results"][0]["coverage_complete"] = False
    _expect_error(
        lambda: module.validate_layer0_transaction_row(invalid),
        "coverage",
    )


def test_source_tar_is_exact_40_file_closure():
    module = _load_module()
    payload = module.build_source_tar(ROOT)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        assert tuple(archive.getnames()) == module.SOURCE_FILES
        assert len(archive.getmembers()) == 40
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
                "layer0_transaction_preflight": record,
                "source_manifest": payload["source_manifest"],
            }),
            "",
        )

    with tempfile.TemporaryDirectory() as temporary:
        result = module.run_remote_layer0_transaction_preflight(
            ROOT,
            "layer0-test",
            staged=staged,
            local_run_root=Path(temporary),
            command_runner=runner,
        )
        destination = Path(temporary) / "layer0-test"
        assert result == record
        assert {path.name for path in destination.iterdir()} == {
            "layer0_transaction_preflight.json",
            "source_manifest.json",
        }
    worker_commands = [
        command[-1]
        for command, _ in calls
        if "internal-rank-worker" in command[-1]
    ]
    assert len(worker_commands) == 3
    assert all("CUDA_VISIBLE_DEVICES=" in text for text in worker_commands)

    count = 0

    def fail_runner(command, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            return subprocess.CompletedProcess(
                command, 0, json.dumps(record["rows"][0]), ""
            )
        return subprocess.CompletedProcess(command, 1, "", "rank failed")

    with tempfile.TemporaryDirectory() as temporary:
        _expect_error(
            lambda: module.run_remote_layer0_transaction_preflight(
                ROOT,
                "layer0-failed",
                staged=staged,
                local_run_root=Path(temporary),
                command_runner=fail_runner,
            ),
            "rank worker",
        )
        assert not (Path(temporary) / "layer0-failed").exists()


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 layer0 transaction preflight tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
