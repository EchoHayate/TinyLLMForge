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
    ROOT
    / "tools/qwen35_real_checkpoint_target_preparation_preflight.py"
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
        "qwen35_target_preparation_preflight_under_test",
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


def _memory_points():
    return {
        "before": {"vmrss_kib": 13000, "vmhwm_kib": 13000},
        "after_torch": {"vmrss_kib": 365000, "vmhwm_kib": 365000},
        "after_metadata": {"vmrss_kib": 495000, "vmhwm_kib": 495000},
        "after_pool": {"vmrss_kib": 505000, "vmhwm_kib": 505000},
        "after_target": {"vmrss_kib": 509000, "vmhwm_kib": 509000},
        "after_release": {"vmrss_kib": 509000, "vmhwm_kib": 509000},
    }


def _valid_row(module, tp_size, tp_rank, process_id):
    pool_bytes = 10321920 if tp_size == 1 else 5160960
    local_query_heads = 8 if tp_size == 1 else 4
    local_kv_heads = 2 if tp_size == 1 else 1
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
        "payload_bytes_read": 0,
        "payload_hashes_recomputed": False,
        "plan_loads": 320,
        "plan_skips": 312,
        "plan_payload_bytes": 4548144832,
        "pool_capacity": 1,
        "pool_device": "cpu",
        "pool_component_count": 36,
        "pool_binding_count": 0,
        "pool_logical_bytes": pool_bytes,
        "pool_physical_bytes": pool_bytes,
        "pool_nonzero_count": 0,
        "pool_unchanged": True,
        "layer_count": 24,
        "linear_adapter_count": 18,
        "backend_calls": [
            [layer_index, local_query_heads, local_kv_heads, 256]
            for layer_index in (3, 7, 11, 15, 19, 23)
        ],
        "binding_count": 320,
        "shared_binding_count": 2,
        "linear_binding_count": 252,
        "full_binding_count": 66,
        "buffer_binding_count": 72,
        "float32_binding_count": 36,
        "all_binding_destinations_meta": True,
        "registered_parameter_count": 225,
        "registered_buffer_count": 78,
        "unexpected_non_meta_registrations": [],
        "loader_call_count": 0,
        "assignment_call_count": 0,
        "model_forward_count": 0,
        "attention_forward_count": 0,
        "pool_create_count": 1,
        "backend_create_count": 6,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "memory": _memory_points(),
        "total_vmhwm_increment_kib": 496000,
        "post_torch_vmhwm_increment_kib": 144000,
        "post_metadata_vmhwm_increment_kib": 14000,
    }


def _valid_record(module):
    source_hashes = {
        name: "a" * 64 for name in module.SOURCE_FILES
    }
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
        "source_file_sha256": source_hashes,
        "source_tree_sha256": module._source_tree_sha256(source_hashes),
        "rows": [
            _valid_row(module, 1, 0, 101),
            _valid_row(module, 2, 0, 102),
            _valid_row(module, 2, 1, 103),
        ],
    }


def test_exact_source_closure_and_record_contract_are_frozen():
    module = _load_module()
    assert module.PRODUCTION_SOURCE_FILES == EXPECTED_PRODUCTION_CLOSURE
    assert module.SOURCE_FILES == (
        *EXPECTED_PRODUCTION_CLOSURE,
        "tools/qwen35_real_checkpoint_target_preparation_preflight.py",
    )
    record = _valid_record(module)
    assert module.validate_target_preparation_preflight(record) == record
    invalid_cases = (
        ({"fresh_process_per_rank": False}, "fresh process"),
        ({
            "rows": [
                record["rows"][0],
                record["rows"][1],
                dict(record["rows"][2], process_id=102),
            ],
        }, "process IDs"),
        ({"rows": record["rows"][:2]}, "TP rows"),
    )
    for updates, message in invalid_cases:
        invalid = dict(record)
        invalid.update(updates)
        _expect_error(
            lambda invalid=invalid: (
                module.validate_target_preparation_preflight(invalid)
            ),
            message,
        )


def test_row_contract_rejects_execution_device_and_memory_regressions():
    module = _load_module()
    row = _valid_row(module, 1, 0, 101)
    assert module.validate_target_preparation_row(row) == row
    round_tripped = json.loads(json.dumps(
        row,
        sort_keys=True,
        separators=(",", ":"),
    ))
    assert (
        module.validate_target_preparation_row(round_tripped)
        == round_tripped
    )
    total_memory = _memory_points()
    total_memory["after_release"] = {
        "vmrss_kib": 537289,
        "vmhwm_kib": 537289,
    }
    post_torch_memory = _memory_points()
    post_torch_memory["before"] = {
        "vmrss_kib": 100000,
        "vmhwm_kib": 100000,
    }
    post_torch_memory["after_metadata"] = {
        "vmrss_kib": 550000,
        "vmhwm_kib": 550000,
    }
    post_torch_memory["after_pool"] = {
        "vmrss_kib": 558000,
        "vmhwm_kib": 558000,
    }
    post_torch_memory["after_target"] = {
        "vmrss_kib": 561609,
        "vmhwm_kib": 561609,
    }
    post_torch_memory["after_release"] = {
        "vmrss_kib": 561609,
        "vmhwm_kib": 561609,
    }
    post_metadata_memory = _memory_points()
    post_metadata_memory["after_metadata"] = {
        "vmrss_kib": 497231,
        "vmhwm_kib": 497231,
    }
    post_metadata_memory["after_pool"] = {
        "vmrss_kib": 525000,
        "vmhwm_kib": 525000,
    }
    post_metadata_memory["after_target"] = {
        "vmrss_kib": 530000,
        "vmhwm_kib": 530000,
    }
    post_metadata_memory["after_release"] = {
        "vmrss_kib": 530000,
        "vmhwm_kib": 530000,
    }
    cases = (
        ({"payload_bytes_read": 1}, "payload"),
        ({"loader_call_count": 1}, "loader"),
        ({"assignment_call_count": 1}, "assignment"),
        ({"model_forward_count": 1}, "forward"),
        ({"attention_forward_count": 1}, "forward"),
        ({"cuda_initialized_after": True}, "CUDA"),
        ({"unexpected_non_meta_registrations": ["weight"]}, "non-meta"),
        ({"pool_physical_bytes": 1}, "pool bytes"),
        ({
            "memory": total_memory,
            "total_vmhwm_increment_kib": (
                module.MAX_TOTAL_VMHWM_INCREMENT_KIB + 1
            ),
            "post_torch_vmhwm_increment_kib": 172289,
            "post_metadata_vmhwm_increment_kib": 42289,
        }, "total VmHWM"),
        ({
            "memory": post_torch_memory,
            "total_vmhwm_increment_kib": 461609,
            "post_torch_vmhwm_increment_kib": (
                module.MAX_POST_TORCH_VMHWM_INCREMENT_KIB + 1
            ),
            "post_metadata_vmhwm_increment_kib": 11609,
        }, "post-Torch VmHWM"),
        ({
            "memory": post_metadata_memory,
            "total_vmhwm_increment_kib": 517000,
            "post_torch_vmhwm_increment_kib": 165000,
            "post_metadata_vmhwm_increment_kib": (
                module.MAX_POST_METADATA_VMHWM_INCREMENT_KIB + 1
            ),
        }, "post-metadata VmHWM"),
    )
    for updates, message in cases:
        invalid = dict(row)
        invalid.update(updates)
        _expect_error(
            lambda invalid=invalid: (
                module.validate_target_preparation_row(invalid)
            ),
            message,
        )


def test_local_sparse_rank_worker_prepares_exact_meta_target_only():
    module = _load_module()
    statuses = iter((
        {"VmRSS": 13000, "VmHWM": 13000},
        {"VmRSS": 365000, "VmHWM": 365000},
        {"VmRSS": 495000, "VmHWM": 495000},
        {"VmRSS": 505000, "VmHWM": 505000},
        {"VmRSS": 509000, "VmHWM": 509000},
        {"VmRSS": 509000, "VmHWM": 509000},
    ))
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint_dir = Path(temporary) / "checkpoint"
        checkpoint_dir.mkdir()
        _write_sparse_checkpoint(module, checkpoint_dir)
        row = module.run_target_preparation_rank_worker(
            checkpoint_dir=checkpoint_dir,
            source_root=ROOT,
            tensor_parallel_size=1,
            tensor_parallel_rank=0,
            observed_user="sitian",
            observed_hostname="n232-195-203",
            process_id=101,
            status_reader=lambda: next(statuses),
        )
    assert row["payload_bytes_read"] == 0
    assert row["pool_logical_bytes"] == 10321920
    assert row["pool_physical_bytes"] == 10321920
    assert row["pool_component_count"] == 36
    assert row["pool_nonzero_count"] == 0
    assert row["pool_unchanged"] is True
    assert row["layer_count"] == 24
    assert row["linear_adapter_count"] == 18
    assert len(row["backend_calls"]) == 6
    assert row["binding_count"] == 320
    assert row["all_binding_destinations_meta"] is True
    assert row["unexpected_non_meta_registrations"] == []
    assert row["loader_call_count"] == 0
    assert row["assignment_call_count"] == 0
    assert row["model_forward_count"] == 0
    assert row["attention_forward_count"] == 0
    assert row["cuda_initialized_before"] is False
    assert row["cuda_initialized_after"] is False
    assert row["memory"]["after_release"]["vmhwm_kib"] == 509000


def test_internal_rank_worker_rejects_unapproved_path_before_imports():
    module = _load_module()
    events = []
    module.run_target_preparation_rank_worker = (
        lambda **kwargs: events.append(kwargs)
    )
    with tempfile.TemporaryDirectory() as temporary:
        arguments = SimpleNamespace(
            checkpoint_dir=temporary,
            source_root=str(ROOT),
            tp_size=1,
            tp_rank=0,
        )
        _expect_error(
            lambda: module._rank_worker_main(arguments),
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
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        hashes = module._source_hashes(ROOT)
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(hashes),
            "",
        )

    staged = module.stage_source(
        ROOT,
        "target-preflight-test",
        command_runner=runner,
    )
    assert staged["local_file_sha256"] == staged["remote_file_sha256"]
    assert calls[0][0][0] == "ssh"
    assert calls[0][1]["input"] == payload
    assert "PYTHONDONTWRITEBYTECODE=1" in calls[1][0][-1]
    assert " -B " in calls[1][0][-1]


def test_remote_orchestration_uses_three_workers_and_atomic_publication():
    module = _load_module()
    record = _valid_record(module)
    source_hashes = record["source_file_sha256"]
    staged = {
        "remote_source_dir": "/remote/run/source",
        "local_file_sha256": source_hashes,
        "remote_file_sha256": source_hashes,
        "source_tree_sha256": record["source_tree_sha256"],
    }
    commands = []
    worker_rows = iter(record["rows"])

    def runner(command, **kwargs):
        commands.append((command, kwargs))
        text = command[-1]
        if "internal-rank-worker" in text:
            row = next(worker_rows)
            return subprocess.CompletedProcess(
                command, 0, json.dumps(row), ""
            )
        if "internal-finalize" in text:
            return subprocess.CompletedProcess(
                command, 0, json.dumps(record), ""
            )
        assert "source_manifest" in kwargs["input"]
        manifest = kwargs["input"]
        payload = json.loads(manifest)
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps({
                "target_preparation_preflight": record,
                "source_manifest": payload["source_manifest"],
            }),
            "",
        )

    with tempfile.TemporaryDirectory() as temporary:
        result = module.run_remote_target_preparation_preflight(
            ROOT,
            "target-preflight-test",
            staged=staged,
            local_run_root=Path(temporary),
            command_runner=runner,
        )
        destination = Path(temporary) / "target-preflight-test"
        assert result == record
        assert tuple(path.name for path in destination.iterdir()) == (
            "target_preparation_preflight.json",
            "source_manifest.json",
        )
    worker_commands = [
        command[-1]
        for command, _ in commands
        if "internal-rank-worker" in command[-1]
    ]
    assert len(worker_commands) == 3
    assert all("CUDA_VISIBLE_DEVICES=" in text for text in worker_commands)
    assert all("PYTHONDONTWRITEBYTECODE=1" in text for text in worker_commands)
    assert all(" -B " in text for text in worker_commands)


def test_partial_remote_failure_publishes_no_local_directory():
    module = _load_module()
    record = _valid_record(module)
    staged = {
        "remote_source_dir": "/remote/run/source",
        "local_file_sha256": record["source_file_sha256"],
        "remote_file_sha256": record["source_file_sha256"],
        "source_tree_sha256": record["source_tree_sha256"],
    }
    calls = 0

    def runner(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(
                command, 0, json.dumps(record["rows"][0]), ""
            )
        return subprocess.CompletedProcess(command, 1, "", "rank failed")

    with tempfile.TemporaryDirectory() as temporary:
        _expect_error(
            lambda: module.run_remote_target_preparation_preflight(
                ROOT,
                "target-preflight-failed",
                staged=staged,
                local_run_root=Path(temporary),
                command_runner=runner,
            ),
            "rank worker",
        )
        assert not (
            Path(temporary) / "target-preflight-failed"
        ).exists()


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 target preparation preflight tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
