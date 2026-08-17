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
MODULE_PATH = (
    ROOT / "tools/qwen35_real_checkpoint_loader_construction_preflight.py"
)
CONFIG_SNAPSHOT = Path("/tmp/qwen35-2b-15852e8-config.json")
INDEX_SNAPSHOT = Path(
    "/tmp/qwen35-2b-15852e8-model.safetensors.index.json"
)
HEADER_SNAPSHOT = Path("/tmp/qwen35-safetensors-header.json")


EXPECTED_PRODUCTION_CLOSURE = (
    "tinyvllm/engine/hybrid_state.py",
    "tinyvllm/engine/qwen35_hybrid_model_owner.py",
    "tinyvllm/engine/qwen35_hybrid_prefix_runtime_identity.py",
    "tinyvllm/engine/qwen35_hybrid_state.py",
    "tinyvllm/engine/qwen35_layer_state.py",
    "tinyvllm/engine/qwen35_state_transaction.py",
    "tinyvllm/layers/embed_head.py",
    "tinyvllm/layers/gated_delta.py",
    "tinyvllm/layers/linear.py",
    "tinyvllm/layers/quantization.py",
    "tinyvllm/layers/qwen35_decoder_layer.py",
    "tinyvllm/layers/qwen35_full_attention.py",
    "tinyvllm/layers/qwen35_linear_attention.py",
    "tinyvllm/layers/qwen35_packed_layer_stack.py",
    "tinyvllm/layers/qwen35_packed_stateful_decoder_layer.py",
    "tinyvllm/layers/qwen35_primitives.py",
    "tinyvllm/layers/qwen35_rotary_embedding.py",
    "tinyvllm/models/qwen35_checkpoint.py",
    "tinyvllm/models/qwen35_checkpoint_assignment.py",
    "tinyvllm/models/qwen35_checkpoint_binding.py",
    "tinyvllm/models/qwen35_checkpoint_candidate_factory.py",
    "tinyvllm/models/qwen35_checkpoint_candidate_loader.py",
    "tinyvllm/models/qwen35_checkpoint_loader_configuration.py",
    "tinyvllm/models/qwen35_checkpoint_metadata.py",
    "tinyvllm/models/qwen35_checkpoint_streaming.py",
    "tinyvllm/models/qwen35_checkpoint_worker.py",
    "tinyvllm/models/qwen35_components.py",
    "tinyvllm/models/qwen35_factory.py",
    "tinyvllm/models/qwen35_packed.py",
    "tinyvllm/speculative/verifier.py",
    "tinyvllm/utils/context.py",
    "tools/qwen35_real_checkpoint_load_worker.py",
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_loader_construction_preflight_under_test",
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
            raise AssertionError(f"missing metadata snapshot: {snapshot}")
        (directory / name).write_bytes(snapshot.read_bytes())
    header = HEADER_SNAPSHOT.read_bytes()
    with (directory / module.APPROVED_SHARD_NAME).open("wb") as handle:
        handle.write(len(header).to_bytes(8, "little"))
        handle.write(header)
        handle.truncate(module.APPROVED_SHARD_SIZE)


def _valid_record(module):
    source_hashes = {
        name: "a" * 64 for name in module.SOURCE_FILES
    }
    rows = [{
        "tp_size": size,
        "tp_rank": rank,
        "loader_type": (
            "Qwen35ManifestBoundCheckpointCandidateLoader"
        ),
        "configuration_type": (
            "Qwen35RankCheckpointLoaderConfiguration"
        ),
        "manifest_dir": module.APPROVED_MODEL_DIR,
        "plan_loads": 320,
        "plan_skips": 312,
        "plan_payload_bytes": 4548144832,
    } for size, rank in ((1, 0), (2, 0), (2, 1))]
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
        "source_file_sha256": source_hashes,
        "source_tree_sha256": module._source_tree_sha256(source_hashes),
        "metadata_bytes_read": 144024,
        "payload_bytes_read": 0,
        "payload_hashes_recomputed": False,
        "provider_events": [],
        "loader_call_count": 0,
        "pool_create_count": 0,
        "backend_create_count": 0,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "vmrss_before_kib": 300000,
        "vmrss_after_torch_kib": 360000,
        "vmrss_after_kib": 420000,
        "vmhwm_before_kib": 300000,
        "vmhwm_after_torch_kib": 360000,
        "vmhwm_after_kib": 430000,
        "total_vmhwm_increment_kib": 130000,
        "construction_vmhwm_increment_kib": 70000,
        "rows": rows,
    }


def test_exact_source_closure_and_record_contract_are_frozen():
    module = _load_module()
    assert module.PRODUCTION_SOURCE_FILES == EXPECTED_PRODUCTION_CLOSURE
    assert module.SOURCE_FILES == (
        *EXPECTED_PRODUCTION_CLOSURE,
        "tools/qwen35_real_checkpoint_loader_construction_preflight.py",
    )
    record = _valid_record(module)
    assert module.validate_construction_preflight(record) == record
    cases = (
        ({"payload_bytes_read": 1}, "payload_bytes_read"),
        ({"provider_events": ["pool"]}, "provider"),
        ({"cuda_initialized_after": True}, "CUDA"),
        ({
            "vmrss_after_kib": (
                record["vmhwm_before_kib"]
                + module.MAX_TOTAL_VMHWM_INCREMENT_KIB
                + 1
            ),
            "vmhwm_after_kib": (
                record["vmhwm_before_kib"]
                + module.MAX_TOTAL_VMHWM_INCREMENT_KIB
                + 1
            ),
            "total_vmhwm_increment_kib": (
                module.MAX_TOTAL_VMHWM_INCREMENT_KIB + 1
            ),
            "construction_vmhwm_increment_kib": (
                record["vmhwm_before_kib"]
                + module.MAX_TOTAL_VMHWM_INCREMENT_KIB
                + 1
                - record["vmhwm_after_torch_kib"]
            ),
        }, "total VmHWM"),
        ({
            "vmrss_after_kib": (
                record["vmhwm_after_torch_kib"]
                + module.MAX_CONSTRUCTION_VMHWM_INCREMENT_KIB
                + 1
            ),
            "vmhwm_after_kib": (
                record["vmhwm_after_torch_kib"]
                + module.MAX_CONSTRUCTION_VMHWM_INCREMENT_KIB
                + 1
            ),
            "total_vmhwm_increment_kib": (
                record["vmhwm_after_torch_kib"]
                + module.MAX_CONSTRUCTION_VMHWM_INCREMENT_KIB
                + 1
                - record["vmhwm_before_kib"]
            ),
            "construction_vmhwm_increment_kib": (
                module.MAX_CONSTRUCTION_VMHWM_INCREMENT_KIB + 1
            ),
        }, "construction phase VmHWM"),
        ({"rows": record["rows"][:2]}, "TP rows"),
    )
    for updates, message in cases:
        invalid = dict(record)
        invalid.update(updates)
        _expect_error(
            lambda invalid=invalid: module.validate_construction_preflight(
                invalid
            ),
            message,
        )


def test_local_sparse_worker_constructs_all_rank_loaders_only():
    module = _load_module()
    statuses = iter((
        {"VmRSS": 300000, "VmHWM": 300000},
        {"VmRSS": 360000, "VmHWM": 360000},
        {"VmRSS": 420000, "VmHWM": 430000},
    ))
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint_dir = Path(temporary) / "checkpoint"
        checkpoint_dir.mkdir()
        _write_sparse_checkpoint(module, checkpoint_dir)
        record = module.run_construction_worker(
            checkpoint_dir=checkpoint_dir,
            source_root=ROOT,
            observed_user="sitian",
            observed_hostname="n232-195-203",
            status_reader=lambda: next(statuses),
        )

    assert record["checkpoint_dir"] == str(checkpoint_dir.resolve())
    assert record["payload_bytes_read"] == 0
    assert record["provider_events"] == []
    assert record["loader_call_count"] == 0
    assert record["pool_create_count"] == 0
    assert record["backend_create_count"] == 0
    assert record["cuda_initialized_before"] is False
    assert record["cuda_initialized_after"] is False
    assert record["total_vmhwm_increment_kib"] == 130000
    assert record["construction_vmhwm_increment_kib"] == 70000
    assert [
        (row["tp_size"], row["tp_rank"])
        for row in record["rows"]
    ] == [(1, 0), (2, 0), (2, 1)]
    assert all(row["plan_loads"] == 320 for row in record["rows"])
    assert all(row["plan_skips"] == 312 for row in record["rows"])


def test_internal_worker_rejects_unapproved_path_before_construction():
    module = _load_module()
    events = []
    module.run_construction_worker = lambda **kwargs: events.append(kwargs)
    with tempfile.TemporaryDirectory() as temporary:
        arguments = SimpleNamespace(
            checkpoint_dir=temporary,
            source_root=str(ROOT),
            output=str(Path(temporary) / "construction.json"),
        )
        _expect_error(
            lambda: module._worker_main(arguments),
            "approved model",
        )
    assert events == []


def test_source_tar_and_staging_are_exact_and_no_bytecode():
    module = _load_module()
    payload = module.build_source_tar(ROOT)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        assert tuple(archive.getnames()) == module.SOURCE_FILES
        for member in archive.getmembers():
            assert member.uid == 0
            assert member.gid == 0
            assert member.mtime == 0
    hashes = module._source_hashes(ROOT)
    calls = []

    def command_runner(command, **kwargs):
        calls.append((command, kwargs))
        if kwargs.get("input") is not None:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(hashes),
            "",
        )

    staged = module.stage_source(
        ROOT,
        "construction-stage-test",
        command_runner=command_runner,
    )
    assert staged["local_file_sha256"] == hashes
    assert staged["remote_file_sha256"] == hashes
    assert staged["source_tree_sha256"] == module._source_tree_sha256(
        hashes
    )
    assert len(calls) == 2
    assert calls[0][0][:4] == [
        "ssh",
        "-S",
        module.SSH_CONTROL_PATH,
        "-o",
    ]
    assert calls[0][0][-2] == module.REMOTE_TARGET
    assert "test ! -e" in calls[0][0][-1]
    assert calls[0][1]["input"] == payload


def test_remote_orchestration_publishes_only_two_artifacts():
    module = _load_module()
    record = _valid_record(module)
    staged = {
        "remote_source_dir": (
            f"{module.REMOTE_RUN_ROOT}/construction-run/source"
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
        result = module.run_remote_construction_preflight(
            ROOT,
            "construction-run",
            staged=staged,
            local_run_root=local_root,
            command_runner=command_runner,
        )
        destination = local_root / "construction-run"
        assert result == record
        assert sorted(path.name for path in destination.iterdir()) == [
            "loader_construction_preflight.json",
            "source_manifest.json",
        ]
        _expect_error(
            lambda: module.run_remote_construction_preflight(
                ROOT,
                "construction-run",
                staged=staged,
                local_run_root=local_root,
                command_runner=command_runner,
            ),
            "already exists",
        )

    assert len(calls) == 2
    command = calls[0][0][-1]
    assert "CUDA_VISIBLE_DEVICES=" in command
    assert "PYTHONDONTWRITEBYTECODE=1" in command
    assert " -B " in f" {command} "
    assert "internal-worker" in command
    assert "qwen35_real_checkpoint_load_worker.py" not in command
    assert "loader_construction_preflight.json" in calls[1][0][-1]
    assert "source_manifest.json" in calls[1][0][-1]


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 loader construction preflight tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
