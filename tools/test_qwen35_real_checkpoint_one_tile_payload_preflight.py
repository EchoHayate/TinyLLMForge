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
    ROOT / "tools/qwen35_real_checkpoint_one_tile_payload_preflight.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_one_tile_payload_preflight_under_test",
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
        points = (22000, 366000, 497000, 507000, 4188000, 4189000)
    else:
        points = (22000, 367000, 497000, 503000, 2345000, 2346000)
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


def _valid_row(module, tp_size, tp_rank, process_id):
    contract = module.TILE_CONTRACTS[(tp_size, tp_rank)]
    digest = "a" * 64
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
        "production_payload_bytes_read": contract["byte_count"],
        "verifier_payload_bytes_read": contract["byte_count"],
        "logical_payload_bytes_read": contract["byte_count"] * 2,
        "payload_hashes_recomputed": False,
        "plan_loads": 320,
        "plan_skips": 312,
        "plan_payload_bytes": 4548144832,
        "selected_binding_index": 3,
        "selected_tile_count": 1,
        "selected_source_count": 1,
        "selected_shard_count": 1,
        "selected_source_name": module.SELECTED_SOURCE_NAME,
        "selected_target": module.SELECTED_TARGET,
        "selected_transform": "squeeze_conv_channel",
        "selected_kind": "squeeze_axis0",
        "selected_dtype": "torch.bfloat16",
        "selected_source_shape": [6144, 1, 4],
        "selected_tile_shape": contract["tile_shape"],
        "selected_source_slices": contract["source_slices"],
        "selected_destination_slices": contract[
            "destination_slices"
        ],
        "selected_payload_relative_range": [
            contract["payload_relative_start"],
            contract["payload_relative_end"],
        ],
        "selected_absolute_file_range": [
            contract["absolute_start"],
            contract["absolute_end"],
        ],
        "selected_tile_bytes": contract["byte_count"],
        "production_sha256": digest,
        "verifier_sha256": digest,
        "source_tensor_sha256": digest,
        "destination_sha256": digest,
        "destination_initially_zero": True,
        "destination_changed_after_copy": True,
        "non_selected_tensors_remained_zero": True,
        "rollback_restored_selected_destination": True,
        "all_unique_tensors_zero_after_rollback": True,
        "open_count": 2,
        "pread_count": 2,
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
            _valid_row(module, 1, 0, 101),
            _valid_row(module, 2, 0, 102),
            _valid_row(module, 2, 1, 103),
        ],
    }


def test_fixed_real_tile_contracts_and_record_validation():
    module = _load_module()
    assert len(module.SOURCE_FILES) == 38
    assert module.SELECTED_BINDING_INDEX == 3
    assert module.SELECTED_SOURCE_NAME == (
        "model.language_model.layers.0.linear_attn.conv1d.weight"
    )
    assert module.TILE_CONTRACTS[(1, 0)]["byte_count"] == 49152
    assert module.TILE_CONTRACTS[(2, 0)]["absolute_start"] == 1017209840
    assert module.TILE_CONTRACTS[(2, 1)]["absolute_start"] == 1017234416
    record = _valid_record(module)
    assert module.validate_one_tile_payload_preflight(record) == record
    duplicate = dict(record)
    duplicate["rows"] = [
        record["rows"][0],
        record["rows"][1],
        dict(record["rows"][2], process_id=102),
    ]
    _expect_error(
        lambda: module.validate_one_tile_payload_preflight(duplicate),
        "process IDs",
    )


def test_row_contract_rejects_range_hash_and_rollback_regressions():
    module = _load_module()
    row = _valid_row(module, 2, 1, 103)
    assert module.validate_one_tile_payload_row(row) == row
    assert module.validate_one_tile_payload_row(
        json.loads(json.dumps(row, sort_keys=True))
    )
    cases = (
        ({"selected_binding_index": 4}, "binding"),
        ({"logical_payload_bytes_read": 1}, "payload bytes"),
        ({"verifier_sha256": "b" * 64}, "hash"),
        ({"non_selected_tensors_remained_zero": False}, "non-selected"),
        ({"rollback_restored_selected_destination": False}, "rollback"),
        ({"loader_call_count": 1}, "loader"),
        ({"cuda_initialized_after": True}, "CUDA"),
    )
    for updates, message in cases:
        invalid = dict(row)
        invalid.update(updates)
        _expect_error(
            lambda invalid=invalid: (
                module.validate_one_tile_payload_row(invalid)
            ),
            message,
        )


def test_exact_double_pread_and_short_read_failure():
    module = _load_module()
    payload = bytes(range(64))
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "shard.safetensors"
        path.write_bytes(b"prefix" + payload + b"suffix")
        result = module.read_and_verify_exact_range(
            path,
            absolute_start=6,
            byte_count=len(payload),
        )
        assert result["production_bytes"] == payload
        assert result["verifier_bytes"] == payload
        assert result["production_sha256"] == result["verifier_sha256"]
        assert result["open_count"] == 2
        assert result["pread_count"] == 2
        _expect_error(
            lambda: module.read_and_verify_exact_range(
                path,
                absolute_start=6,
                byte_count=len(payload) + 10,
            ),
            "short payload read",
        )


def test_synthetic_tile_copy_isolated_and_rollback_exact():
    module = _load_module()
    source = torch.arange(
        24,
        dtype=torch.float32,
    ).to(torch.bfloat16).reshape(6, 4)
    destination = torch.zeros((6, 4), dtype=torch.bfloat16)
    other = torch.zeros((3,), dtype=torch.float32)
    tile = module.Qwen35CheckpointTile(
        binding_index=3,
        source_name=module.SELECTED_SOURCE_NAME,
        shard=module.APPROVED_SHARD_NAME,
        source_tensor_shape=(6, 1, 4),
        source_slices=(slice(0, 6), 0, slice(0, 4)),
        tile_shape=(6, 4),
        destination=destination,
        destination_slices=(slice(0, 6), slice(0, 4)),
        destination_shape=(6, 4),
        dtype=torch.bfloat16,
        byte_count=48,
        target=module.SELECTED_TARGET,
        kind="squeeze_axis0",
    )
    result = module.copy_verify_and_rollback_tile(
        tile,
        source,
        unique_tensors=(destination, other),
    )
    assert result["destination_initially_zero"] is True
    assert result["destination_changed_after_copy"] is True
    assert result["non_selected_tensors_remained_zero"] is True
    assert result["rollback_restored_selected_destination"] is True
    assert result["all_unique_tensors_zero_after_rollback"] is True
    assert torch.count_nonzero(destination).item() == 0
    assert torch.count_nonzero(other).item() == 0


def test_source_tar_is_exact_38_file_closure():
    module = _load_module()
    payload = module.build_source_tar(ROOT)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        assert tuple(archive.getnames()) == module.SOURCE_FILES
        assert len(archive.getmembers()) == 38
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
                "one_tile_payload_preflight": record,
                "source_manifest": payload["source_manifest"],
            }),
            "",
        )

    with tempfile.TemporaryDirectory() as temporary:
        result = module.run_remote_one_tile_payload_preflight(
            ROOT,
            "one-tile-test",
            staged=staged,
            local_run_root=Path(temporary),
            command_runner=runner,
        )
        destination = Path(temporary) / "one-tile-test"
        assert result == record
        assert {path.name for path in destination.iterdir()} == {
            "one_tile_payload_preflight.json",
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
            lambda: module.run_remote_one_tile_payload_preflight(
                ROOT,
                "one-tile-failed",
                staged=staged,
                local_run_root=Path(temporary),
                command_runner=fail_runner,
            ),
            "rank worker",
        )
        assert not (Path(temporary) / "one-tile-failed").exists()


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 one-tile payload preflight tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
