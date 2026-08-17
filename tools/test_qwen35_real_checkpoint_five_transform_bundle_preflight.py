from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tools/qwen35_real_checkpoint_five_transform_bundle_preflight.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_five_transform_bundle_preflight_under_test",
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
        points = (23000, 367000, 498000, 508000, 4190000, 4240000)
    else:
        points = (23000, 368000, 499000, 504000, 2347000, 2380000)
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


def _valid_tile_result(contract):
    digest = "a" * 64
    return {
        "binding_index": contract["binding_index"],
        "source_name": contract["source_name"],
        "target": contract["target"],
        "transform": contract["transform"],
        "kind": contract["kind"],
        "dtype": contract["dtype"],
        "source_shape": contract["source_shape"],
        "tile_shape": contract["tile_shape"],
        "source_slices": contract["source_slices"],
        "destination_slices": contract["destination_slices"],
        "ranges": contract["ranges"],
        "range_count": len(contract["ranges"]),
        "tile_bytes": contract["byte_count"],
        "production_sha256": digest,
        "verifier_sha256": digest,
        "source_tensor_sha256": digest,
        "destination_sha256": digest,
        "destination_initially_zero": True,
        "destination_changed_after_copy": True,
        "not_yet_selected_destinations_remained_zero": True,
    }


def _valid_row(module, tp_size, tp_rank, process_id):
    contract = module.BUNDLE_CONTRACTS[(tp_size, tp_rank)]
    memory = _memory(tp_size)
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
        "selected_tile_count": 5,
        "selected_source_count": 5,
        "selected_shard_count": 1,
        "production_payload_bytes_read": contract["bytes_per_pass"],
        "verifier_payload_bytes_read": contract["bytes_per_pass"],
        "logical_payload_bytes_read": contract["logical_bytes"],
        "payload_hashes_recomputed": False,
        "plan_loads": 320,
        "plan_skips": 312,
        "plan_payload_bytes": 4548144832,
        "tile_results": [
            _valid_tile_result(tile)
            for tile in contract["tiles"]
        ],
        "selected_destinations_distinct": True,
        "non_selected_tensors_remained_zero": True,
        "rollback_order": list(reversed(module.SELECTED_BINDING_INDICES)),
        "all_selected_snapshots_restored": True,
        "all_unique_tensors_zero_after_rollback": True,
        "open_count": 2,
        "pread_count": contract["pread_count"],
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
            _valid_row(module, 1, 0, 201),
            _valid_row(module, 2, 0, 202),
            _valid_row(module, 2, 1, 203),
        ],
    }


def test_frozen_bundle_contracts_and_record_validation():
    module = _load_module()
    assert len(module.SOURCE_FILES) == 39
    assert len(set(module.SOURCE_FILES)) == 39
    assert module.SELECTED_BINDING_INDICES == (3, 4, 7, 9, 11)
    expected = {
        (1, 0): (176672, 5, 353344, 10),
        (2, 0): (152080, 14, 304160, 28),
        (2, 1): (152080, 14, 304160, 28),
    }
    for tp, values in expected.items():
        contract = module.BUNDLE_CONTRACTS[tp]
        assert (
            contract["bytes_per_pass"],
            contract["ranges_per_pass"],
            contract["logical_bytes"],
            contract["pread_count"],
        ) == values
        assert [
            tile["binding_index"] for tile in contract["tiles"]
        ] == [3, 4, 7, 9, 11]
    record = _valid_record(module)
    assert module.validate_five_transform_bundle_preflight(record) == record
    duplicate = dict(record)
    duplicate["rows"] = [
        record["rows"][0],
        record["rows"][1],
        dict(record["rows"][2], process_id=202),
    ]
    _expect_error(
        lambda: module.validate_five_transform_bundle_preflight(duplicate),
        "process IDs",
    )


def test_row_validation_rejects_range_isolation_and_rollback_regressions():
    module = _load_module()
    row = _valid_row(module, 2, 1, 203)
    assert module.validate_five_transform_bundle_row(row) == row
    assert module.validate_five_transform_bundle_row(
        json.loads(json.dumps(row, sort_keys=True))
    )
    cases = (
        ({"selected_binding_indices": [3]}, "binding"),
        ({"logical_payload_bytes_read": 1}, "payload bytes"),
        ({"pread_count": 1}, "pread"),
        ({"selected_destinations_distinct": False}, "distinct"),
        ({"non_selected_tensors_remained_zero": False}, "non-selected"),
        ({"rollback_order": [3, 4, 7, 9, 11]}, "rollback order"),
        ({"all_selected_snapshots_restored": False}, "snapshots"),
        ({"loader_call_count": 1}, "loader"),
        ({"cuda_initialized_after": True}, "CUDA"),
    )
    for updates, message in cases:
        invalid = dict(row)
        invalid.update(updates)
        _expect_error(
            lambda invalid=invalid: (
                module.validate_five_transform_bundle_row(invalid)
            ),
            message,
        )
    invalid = json.loads(json.dumps(row))
    invalid["tile_results"][4]["ranges"][0][0] += 2
    _expect_error(
        lambda: module.validate_five_transform_bundle_row(invalid),
        "ranges",
    )


def test_exact_double_descriptor_multi_range_reads_fail_closed():
    module = _load_module()
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "shard.safetensors"
        path.write_bytes(bytes(range(128)))
        tile_ranges = (
            ((4, 12),),
            ((20, 24), (30, 35), (50, 53)),
        )
        result = module.read_and_verify_exact_ranges(path, tile_ranges)
        assert result["production_tiles"] == (
            bytes(range(4, 12)),
            bytes(range(20, 24))
            + bytes(range(30, 35))
            + bytes(range(50, 53)),
        )
        assert result["production_tiles"] == result["verifier_tiles"]
        assert result["open_count"] == 2
        assert result["pread_count"] == 8
        assert result["production_bytes_read"] == 20
        assert result["verifier_bytes_read"] == 20
        _expect_error(
            lambda: module.read_and_verify_exact_ranges(
                path,
                (((120, 140),),),
            ),
            "short payload read",
        )
        _expect_error(
            lambda: module.read_and_verify_exact_ranges(
                path,
                (((20, 30), (25, 35)),),
            ),
            "overlap",
        )
        _expect_error(
            lambda: module.read_and_verify_exact_ranges(
                path,
                (((30, 35), (20, 25)),),
            ),
            "sorted",
        )


def test_five_tile_copy_isolated_and_reverse_rollback_exact():
    module = _load_module()
    destinations = (
        torch.zeros((4, 2), dtype=torch.bfloat16),
        torch.zeros((4,), dtype=torch.bfloat16),
        torch.zeros((2, 4), dtype=torch.bfloat16),
        torch.zeros((3,), dtype=torch.float32),
        torch.zeros((2, 3), dtype=torch.bfloat16),
    )
    source_tensors = (
        torch.arange(8, dtype=torch.float32).to(torch.bfloat16).reshape(4, 2),
        torch.arange(4, dtype=torch.float32).to(torch.bfloat16),
        torch.arange(8, dtype=torch.float32).to(torch.bfloat16).reshape(2, 4),
        torch.arange(3, dtype=torch.float32),
        torch.arange(6, dtype=torch.float32).to(torch.bfloat16).reshape(2, 3),
    )
    shapes = ((4, 2), (4,), (2, 4), (3,), (2, 3))
    kinds = (
        "squeeze_axis0",
        "axis0",
        "segmented_axis0",
        "replicated",
        "axis1",
    )
    tiles = []
    for index, (destination, shape, kind) in enumerate(
        zip(destinations, shapes, kinds, strict=True)
    ):
        slices = tuple(slice(0, dimension) for dimension in shape)
        tiles.append(module.Qwen35CheckpointTile(
            binding_index=module.SELECTED_BINDING_INDICES[index],
            source_name=f"source-{index}",
            shard=module.APPROVED_SHARD_NAME,
            source_tensor_shape=shape,
            source_slices=slices,
            tile_shape=shape,
            destination=destination,
            destination_slices=slices,
            destination_shape=shape,
            dtype=destination.dtype,
            byte_count=destination.numel() * destination.element_size(),
            target=f"target-{index}",
            kind=kind,
        ))
    other = torch.zeros((7,), dtype=torch.float32)
    result = module.copy_verify_and_reverse_rollback_bundle(
        tuple(tiles),
        source_tensors,
        unique_tensors=(*destinations, other),
    )
    assert result["selected_destinations_distinct"] is True
    assert result["non_selected_tensors_remained_zero"] is True
    assert result["rollback_order"] == [11, 9, 7, 4, 3]
    assert result["all_selected_snapshots_restored"] is True
    assert result["all_unique_tensors_zero_after_rollback"] is True
    assert all(torch.count_nonzero(tensor).item() == 0 for tensor in destinations)
    duplicate_tiles = (*tiles[:4], module.Qwen35CheckpointTile(
        **{
            **tiles[4].__dict__,
            "destination": destinations[0],
            "destination_slices": (
                slice(0, 2),
                slice(0, 2),
            ),
            "destination_shape": (2, 2),
            "tile_shape": (2, 2),
            "byte_count": 8,
        }
    ))
    _expect_error(
        lambda: module.copy_verify_and_reverse_rollback_bundle(
            duplicate_tiles,
            (*source_tensors[:4], source_tensors[4][:, :2]),
            unique_tensors=(*destinations, other),
        ),
        "distinct",
    )


def test_source_tar_is_exact_39_file_closure():
    module = _load_module()
    payload = module.build_source_tar(ROOT)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        assert tuple(archive.getnames()) == module.SOURCE_FILES
        assert len(archive.getmembers()) == 39
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
                "five_transform_bundle_preflight": record,
                "source_manifest": payload["source_manifest"],
            }),
            "",
        )

    with tempfile.TemporaryDirectory() as temporary:
        result = module.run_remote_five_transform_bundle_preflight(
            ROOT,
            "five-transform-test",
            staged=staged,
            local_run_root=Path(temporary),
            command_runner=runner,
        )
        destination = Path(temporary) / "five-transform-test"
        assert result == record
        assert {path.name for path in destination.iterdir()} == {
            "five_transform_bundle_preflight.json",
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
            lambda: module.run_remote_five_transform_bundle_preflight(
                ROOT,
                "five-transform-failed",
                staged=staged,
                local_run_root=Path(temporary),
                command_runner=fail_runner,
            ),
            "rank worker",
        )
        assert not (Path(temporary) / "five-transform-failed").exists()


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 five-transform bundle preflight tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
