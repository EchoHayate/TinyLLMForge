from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import types

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tools/qwen35_real_checkpoint_cpu_materialization_preflight.py"
)
TARGET_GATE = (
    "tools/qwen35_real_checkpoint_target_preparation_preflight.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_cpu_materialization_preflight_under_test",
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
        points = (20000, 365000, 496000, 506000, 4186000, 4187000)
    else:
        points = (20000, 365000, 496000, 502000, 2343000, 2344000)
    names = (
        "before",
        "after_torch",
        "after_metadata",
        "after_pool",
        "after_target",
        "after_touch",
    )
    return {
        name: {"vmrss_kib": value, "vmhwm_kib": value}
        for name, value in zip(names, points, strict=True)
    }


def _rotary_buffers():
    return [
        {
            "name": (
                f"layer_stack.layers.{layer_index}."
                "full_attention.rotary.inv_freq"
            ),
            "shape": [32],
            "dtype": "torch.float32",
            "bytes": 128,
        }
        for layer_index in (3, 7, 11, 15, 19, 23)
    ]


def _valid_row(module, tp_size, tp_rank, process_id):
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
        "payload_bytes_read": 0,
        "payload_hashes_recomputed": False,
        "plan_loads": 320,
        "plan_skips": 312,
        "plan_payload_bytes": 4548144832,
        "pool_capacity": 1,
        "pool_device": "cpu",
        "pool_component_count": 36,
        "pool_binding_count": 0,
        "pool_logical_bytes": (
            10321920 if tp_size == 1 else 5160960
        ),
        "pool_physical_bytes": (
            10321920 if tp_size == 1 else 5160960
        ),
        "pool_nonzero_count": 0,
        "pool_unchanged": True,
        "layer_count": 24,
        "linear_adapter_count": 18,
        "backend_calls": [
            [
                layer_index,
                8 // tp_size,
                2 // tp_size,
                256,
            ]
            for layer_index in (3, 7, 11, 15, 19, 23)
        ],
        "binding_count": 320,
        "shared_binding_count": 2,
        "linear_binding_count": 252,
        "full_binding_count": 66,
        "buffer_binding_count": 72,
        "float32_binding_count": 36,
        "registered_entry_count": 303,
        "unique_registered_tensor_count": 302,
        "unique_registered_bytes": (
            3763656128 if tp_size == 1 else 1881936480
        ),
        "unique_binding_tensor_count": 296,
        "unique_binding_bytes": (
            3763655360 if tp_size == 1 else 1881935712
        ),
        "unbound_registered": _rotary_buffers(),
        "tied_embedding_same_object": True,
        "all_registrations_cpu": True,
        "all_bindings_cpu": True,
        "all_binding_destinations_registered": True,
        "all_unique_tensors_zero_after_touch": True,
        "loader_call_count": 0,
        "assignment_call_count": 0,
        "model_forward_count": 0,
        "attention_forward_count": 0,
        "pool_create_count": 1,
        "backend_create_count": 6,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "memory": memory,
        "total_vmhwm_increment_kib": (
            memory["after_touch"]["vmhwm_kib"]
            - memory["before"]["vmhwm_kib"]
        ),
        "post_torch_vmhwm_increment_kib": (
            memory["after_touch"]["vmhwm_kib"]
            - memory["after_torch"]["vmhwm_kib"]
        ),
        "post_metadata_vmhwm_increment_kib": (
            memory["after_touch"]["vmhwm_kib"]
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
        "process_exit_is_release_boundary": True,
        "source_file_sha256": hashes,
        "source_tree_sha256": module._source_tree_sha256(hashes),
        "rows": [
            _valid_row(module, 1, 0, 101),
            _valid_row(module, 2, 0, 102),
            _valid_row(module, 2, 1, 103),
        ],
    }


def _compact_target():
    factory_test_path = (
        ROOT / "tools/test_qwen35_checkpoint_candidate_factory.py"
    )
    spec = importlib.util.spec_from_file_location(
        "candidate_factory_fixture",
        factory_test_path,
    )
    fixture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture
    spec.loader.exec_module(fixture)
    config = fixture._config()
    pool = fixture._pool(config, 1)
    target = fixture.prepare_qwen35_checkpoint_candidate_target(
        config,
        fixture._tensor_plan(),
        pool=pool,
        tensor_parallel_size=1,
        tensor_parallel_rank=0,
        build_attention_backend=fixture._build_backend,
        parameter_device="cpu",
    )
    return target


def test_exact_source_closure_and_record_contract():
    module = _load_module()
    assert module.PRODUCTION_SOURCE_FILES == module.base.PRODUCTION_SOURCE_FILES
    assert module.SOURCE_FILES == (
        *module.PRODUCTION_SOURCE_FILES,
        TARGET_GATE,
        "tools/qwen35_real_checkpoint_cpu_materialization_preflight.py",
    )
    record = _valid_record(module)
    assert module.validate_cpu_materialization_preflight(record) == record
    duplicate_pid = dict(record)
    duplicate_pid["rows"] = [
        record["rows"][0],
        record["rows"][1],
        dict(record["rows"][2], process_id=102),
    ]
    _expect_error(
        lambda: module.validate_cpu_materialization_preflight(
            duplicate_pid
        ),
        "process IDs",
    )


def test_row_contract_rejects_bytes_execution_and_memory_regressions():
    module = _load_module()
    row = _valid_row(module, 1, 0, 101)
    assert module.validate_cpu_materialization_row(row) == row
    round_tripped = json.loads(json.dumps(row, sort_keys=True))
    assert module.validate_cpu_materialization_row(round_tripped)
    cases = (
        ({"unique_registered_bytes": 1}, "registered bytes"),
        ({"unbound_registered": []}, "rotary"),
        ({"all_unique_tensors_zero_after_touch": False}, "zero"),
        ({"payload_bytes_read": 1}, "payload"),
        ({"loader_call_count": 1}, "loader"),
        ({"cuda_initialized_after": True}, "CUDA"),
    )
    for updates, message in cases:
        invalid = dict(row)
        invalid.update(updates)
        _expect_error(
            lambda invalid=invalid: (
                module.validate_cpu_materialization_row(invalid)
            ),
            message,
        )


def test_compact_cpu_target_touch_is_exact_and_identity_preserving():
    module = _load_module()
    target = _compact_target()
    model = target.assembly.packed.model
    before = {
        binding.destination_name: id(binding.destination)
        for binding in target.binding_plan.bindings
    }
    result = module.inspect_and_touch_cpu_target(target)
    assert result["all_registrations_cpu"] is True
    assert result["all_bindings_cpu"] is True
    assert result["all_binding_destinations_registered"] is True
    assert result["all_unique_tensors_zero_after_touch"] is True
    assert result["tied_embedding_same_object"] is True
    assert {
        binding.destination_name: id(binding.destination)
        for binding in target.binding_plan.bindings
    } == before
    assert model.embed_tokens.weight is model.lm_head.weight


def test_source_tar_is_deterministic_and_contains_34_files():
    module = _load_module()
    payload = module.build_source_tar(ROOT)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        assert tuple(archive.getnames()) == module.SOURCE_FILES
        assert len(archive.getmembers()) == 34
        assert all(member.mtime == 0 for member in archive.getmembers())


def test_remote_orchestration_is_three_processes_and_atomic():
    module = _load_module()
    record = _valid_record(module)
    staged = {
        "remote_source_dir": "/remote/run/source",
        "local_file_sha256": record["source_file_sha256"],
        "remote_file_sha256": record["source_file_sha256"],
        "source_tree_sha256": record["source_tree_sha256"],
    }
    rows = iter(record["rows"])
    commands = []

    def runner(command, **kwargs):
        commands.append((command, kwargs))
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
                "cpu_materialization_preflight": record,
                "source_manifest": payload["source_manifest"],
            }),
            "",
        )

    with tempfile.TemporaryDirectory() as temporary:
        result = module.run_remote_cpu_materialization_preflight(
            ROOT,
            "cpu-materialization-test",
            staged=staged,
            local_run_root=Path(temporary),
            command_runner=runner,
        )
        destination = Path(temporary) / "cpu-materialization-test"
        assert result == record
        assert {path.name for path in destination.iterdir()} == {
            "cpu_materialization_preflight.json",
            "source_manifest.json",
        }
    worker_commands = [
        command[-1]
        for command, _ in commands
        if "internal-rank-worker" in command[-1]
    ]
    assert len(worker_commands) == 3
    for text in worker_commands:
        assert "CUDA_VISIBLE_DEVICES=" in text
        assert "OMP_NUM_THREADS=8" in text
        assert "MKL_NUM_THREADS=8" in text
        assert "PYTHONDONTWRITEBYTECODE=1" in text
        assert " -B " in text


def test_partial_failure_publishes_no_local_artifact():
    module = _load_module()
    record = _valid_record(module)
    staged = {
        "remote_source_dir": "/remote/run/source",
        "local_file_sha256": record["source_file_sha256"],
        "remote_file_sha256": record["source_file_sha256"],
        "source_tree_sha256": record["source_tree_sha256"],
    }
    count = 0

    def runner(command, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            return subprocess.CompletedProcess(
                command, 0, json.dumps(record["rows"][0]), ""
            )
        return subprocess.CompletedProcess(command, 1, "", "rank failed")

    with tempfile.TemporaryDirectory() as temporary:
        _expect_error(
            lambda: module.run_remote_cpu_materialization_preflight(
                ROOT,
                "cpu-materialization-failed",
                staged=staged,
                local_run_root=Path(temporary),
                command_runner=runner,
            ),
            "rank worker",
        )
        assert not (
            Path(temporary) / "cpu-materialization-failed"
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
        "qwen35 CPU materialization preflight tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
