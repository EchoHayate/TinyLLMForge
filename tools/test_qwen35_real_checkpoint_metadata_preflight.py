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


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/qwen35_real_checkpoint_metadata_preflight.py"
CONFIG_SNAPSHOT = Path("/tmp/qwen35-2b-15852e8-config.json")
INDEX_SNAPSHOT = Path(
    "/tmp/qwen35-2b-15852e8-model.safetensors.index.json"
)
HEADER_SNAPSHOT = Path("/tmp/qwen35-safetensors-header.json")


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_real_checkpoint_metadata_preflight_under_test",
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


def _write_sparse_checkpoint(module, directory):
    for snapshot, name in (
        (CONFIG_SNAPSHOT, "config.json"),
        (INDEX_SNAPSHOT, "model.safetensors.index.json"),
    ):
        if not snapshot.is_file():
            raise AssertionError(f"missing local metadata snapshot: {snapshot}")
        (directory / name).write_bytes(snapshot.read_bytes())
    if not HEADER_SNAPSHOT.is_file():
        raise AssertionError(
            f"missing local metadata snapshot: {HEADER_SNAPSHOT}"
        )
    header = HEADER_SNAPSHOT.read_bytes()
    shard = directory / module.APPROVED_SHARD_NAME
    with shard.open("wb") as handle:
        handle.write(len(header).to_bytes(8, "little"))
        handle.write(header)
        handle.truncate(module.APPROVED_SHARD_SIZE)
    return header


def _valid_record(module):
    source_hashes = {
        name: "a" * 64 for name in module.SOURCE_FILES
    }
    return {
        "schema_version": module.SCHEMA_VERSION,
        "status": "PASS",
        "remote_target": module.REMOTE_TARGET,
        "remote_python": module.REMOTE_PYTHON,
        "observed_user": "sitian",
        "observed_hostname": "n232-195-203",
        "checkpoint_dir": module.APPROVED_MODEL_DIR,
        "model_manifest_sha256": module.APPROVED_MODEL_MANIFEST_SHA256,
        "config_sha256": module.APPROVED_CONFIG_SHA256,
        "index_sha256": module.APPROVED_INDEX_SHA256,
        "config_index_header_sha256": module.APPROVED_COMPOSITE_SHA256,
        "shards": [{
            "name": module.APPROVED_SHARD_NAME,
            "size": module.APPROVED_SHARD_SIZE,
            "sha256": module.APPROVED_SHARD_SHA256,
        }],
        "source_file_sha256": source_hashes,
        "source_tree_sha256": module._source_tree_sha256(source_hashes),
        "metadata_bytes_read": 144024,
        "payload_bytes_read": 0,
        "payload_hashes_recomputed": False,
        "payload_identity_source": "retained_approved_manifest",
        "layer_count": 24,
        "linear_attention_layer_count": 18,
        "full_attention_layer_count": 6,
        "index_weight_count": 632,
        "header_tensor_count": 632,
        "load_count": 320,
        "skip_count": 312,
        "plan_payload_bytes": 4548144832,
        "index_total_size": 4548144832,
        "shard_count": 1,
    }


def test_exact_record_contract_is_fail_closed():
    module = _load_module()
    record = _valid_record(module)
    assert module.validate_metadata_preflight(record) == record
    cases = (
        ("status", "FAIL", "status"),
        ("config_sha256", "f" * 64, "config_sha256"),
        ("metadata_bytes_read", 0, "metadata_bytes_read"),
        ("payload_bytes_read", 1, "payload_bytes_read"),
        ("payload_hashes_recomputed", True, "payload hashes"),
        ("layer_count", 23, "layer counts"),
        ("header_tensor_count", 631, "tensor counts"),
        ("plan_payload_bytes", 1, "payload total"),
    )
    for key, value, message in cases:
        invalid = dict(record)
        invalid[key] = value
        _expect_error(
            lambda invalid=invalid: module.validate_metadata_preflight(
                invalid
            ),
            message,
        )


def test_local_snapshot_worker_builds_exact_plan_without_payload_reads():
    module = _load_module()
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint_dir = Path(temporary) / "checkpoint"
        checkpoint_dir.mkdir()
        header = _write_sparse_checkpoint(module, checkpoint_dir)
        record = module.run_metadata_worker(
            checkpoint_dir=checkpoint_dir,
            source_root=ROOT,
            observed_user="sitian",
            observed_hostname="n232-195-203",
        )

    assert record["checkpoint_dir"] == str(checkpoint_dir.resolve())
    assert record["metadata_bytes_read"] == (
        CONFIG_SNAPSHOT.stat().st_size
        + INDEX_SNAPSHOT.stat().st_size
        + 8
        + len(header)
    )
    assert record["payload_bytes_read"] == 0
    assert record["layer_count"] == 24
    assert record["linear_attention_layer_count"] == 18
    assert record["full_attention_layer_count"] == 6
    assert record["index_weight_count"] == 632
    assert record["header_tensor_count"] == 632
    assert record["load_count"] == 320
    assert record["skip_count"] == 312
    assert record["plan_payload_bytes"] == 4548144832
    assert record["index_total_size"] == 4548144832


def test_worker_rejects_modified_config_before_plan_construction():
    module = _load_module()
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint_dir = Path(temporary) / "checkpoint"
        checkpoint_dir.mkdir()
        _write_sparse_checkpoint(module, checkpoint_dir)
        (checkpoint_dir / "config.json").write_text(
            json.dumps({"invalid": True}),
            encoding="utf-8",
        )
        _expect_error(
            lambda: module.run_metadata_worker(
                checkpoint_dir=checkpoint_dir,
                source_root=ROOT,
                observed_user="sitian",
                observed_hostname="n232-195-203",
            ),
            "config SHA256",
        )


def test_source_staging_is_deterministic_and_bound_to_fixed_ssh():
    module = _load_module()
    payload = module.build_source_tar(ROOT)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        assert tuple(archive.getnames()) == module.SOURCE_FILES
        for member in archive.getmembers():
            assert member.uid == 0
            assert member.gid == 0
            assert member.mtime == 0

    local_hashes = module._source_hashes(ROOT)
    calls = []

    def command_runner(command, **kwargs):
        calls.append((command, kwargs))
        if kwargs.get("input") is not None:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(local_hashes),
            "",
        )

    staged = module.stage_source(
        ROOT,
        "metadata-preflight-stage-test",
        command_runner=command_runner,
    )
    assert staged["local_file_sha256"] == local_hashes
    assert staged["remote_file_sha256"] == local_hashes
    assert staged["source_tree_sha256"] == module._source_tree_sha256(
        local_hashes
    )
    assert len(calls) == 2
    assert calls[0][0][:4] == [
        "ssh",
        "-S",
        module.SSH_CONTROL_PATH,
        "-o",
    ]
    assert calls[0][0][-2] == module.REMOTE_TARGET
    assert calls[0][1]["input"] == payload
    assert "test ! -e" in calls[0][0][-1]


def test_remote_orchestration_publishes_only_two_verified_artifacts():
    module = _load_module()
    record = _valid_record(module)
    staged = {
        "remote_source_dir": (
            f"{module.REMOTE_RUN_ROOT}/metadata-preflight-run/source"
        ),
        "local_file_sha256": record["source_file_sha256"],
        "remote_file_sha256": record["source_file_sha256"],
        "source_tree_sha256": record["source_tree_sha256"],
    }
    calls = []

    def command_runner(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(record),
                "",
            )
        payload = json.loads(kwargs["input"])
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(payload),
            "",
        )

    with tempfile.TemporaryDirectory() as temporary:
        local_root = Path(temporary)
        result = module.run_remote_metadata_preflight(
            ROOT,
            "metadata-preflight-run",
            staged=staged,
            local_run_root=local_root,
            command_runner=command_runner,
        )
        destination = local_root / "metadata-preflight-run"
        assert result == record
        assert sorted(path.name for path in destination.iterdir()) == [
            "metadata_preflight.json",
            "source_manifest.json",
        ]
        assert json.loads(
            (destination / "metadata_preflight.json").read_text()
        ) == record
        source_manifest = json.loads(
            (destination / "source_manifest.json").read_text()
        )
        assert source_manifest["source_tree_sha256"] == (
            staged["source_tree_sha256"]
        )
        _expect_error(
            lambda: module.run_remote_metadata_preflight(
                ROOT,
                "metadata-preflight-run",
                staged=staged,
                local_run_root=local_root,
                command_runner=command_runner,
            ),
            "already exists",
        )

    assert len(calls) == 2
    assert "CUDA_VISIBLE_DEVICES=" in calls[0][0][-1]
    assert "PYTHONDONTWRITEBYTECODE=1" in calls[0][0][-1]
    assert "qwen35_real_checkpoint_metadata_preflight.py" in (
        calls[0][0][-1]
    )
    assert "qwen35_real_checkpoint_load_worker.py" not in calls[0][0][-1]
    assert "metadata_preflight.json" in calls[1][0][-1]
    assert "source_manifest.json" in calls[1][0][-1]


def test_internal_worker_rejects_unapproved_path_before_reader_invocation():
    module = _load_module()
    events = []
    module.run_metadata_worker = lambda **kwargs: events.append(kwargs)
    with tempfile.TemporaryDirectory() as temporary:
        arguments = SimpleNamespace(
            checkpoint_dir=temporary,
            source_root=str(ROOT),
            output=str(Path(temporary) / "metadata.json"),
        )
        _expect_error(
            lambda: module._worker_main(arguments),
            "approved model",
        )
    assert events == []


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 real checkpoint metadata preflight tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
