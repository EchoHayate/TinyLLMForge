from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys
import tarfile
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tools/qwen35_real_checkpoint_tiled_loader_core_preflight.py"
)
PREREQUISITE_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-complete-checkpoint-20260728-065128/"
    "complete_checkpoint_transaction_preflight.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_tiled_loader_core_preflight_under_test",
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


def _digest(tensor):
    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _synthetic_target():
    selected = torch.full((4,), 9, dtype=torch.bfloat16)
    other = torch.tensor((0.5, 0.25), dtype=torch.float32)
    bindings = (
        SimpleNamespace(
            destination=selected,
            destination_slice=(0, 2),
        ),
        SimpleNamespace(
            destination=selected,
            destination_slice=(2, 2),
        ),
    )
    model = SimpleNamespace(
        named_parameters=lambda remove_duplicate=False: (
            ("selected", selected),
        ),
        named_buffers=lambda remove_duplicate=False: (
            ("other", other),
        ),
    )
    binding_plan = SimpleNamespace(
        bindings=bindings,
        tensor_parallel_size=1,
        tensor_parallel_rank=0,
    )
    target = SimpleNamespace(
        assembly=SimpleNamespace(
            packed=SimpleNamespace(model=model),
        ),
        binding_plan=binding_plan,
        pool=SimpleNamespace(marker="pool"),
        _consumed=False,
    )
    tile_plan = SimpleNamespace(marker="tiles")
    expected = torch.tensor(
        [1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16
    )
    oracle = {
        "tp_size": 1,
        "tp_rank": 0,
        "binding_results": [
            {
                "binding_index": 0,
                "phase_name": "left",
                "destination_sha256": _digest(expected[:2]),
            },
            {
                "binding_index": 1,
                "phase_name": "right",
                "destination_sha256": _digest(expected[2:]),
            },
        ],
        "phase_results": [
            {
                "phase_name": "left",
                "destination_sha256": _digest(expected[:2]),
            },
            {
                "phase_name": "right",
                "destination_sha256": _digest(expected[2:]),
            },
        ],
        "aggregate_destination_sha256": _digest(expected),
    }
    return target, tile_plan, expected, oracle, selected, other


def test_exact_prerequisite_oracle_binding():
    module = _load_module()
    oracle = module.load_complete_gate_oracle(PREREQUISITE_ARTIFACT)
    assert oracle["source_tree_sha256"] == (
        "da665b2de0aaa6533e55be0469c76ed39d92e817aabf80618f07b7efa7ef7042"
    )
    assert [row["process_id"] for row in oracle["rows"]] == [
        3946836,
        3960911,
        3966499,
    ]
    assert module.select_oracle_row(oracle, 2, 1)["process_id"] == 3966499
    with tempfile.TemporaryDirectory() as temporary:
        copied = Path(temporary) / "oracle.json"
        payload = json.loads(PREREQUISITE_ARTIFACT.read_text())
        payload["rows"][0]["process_id"] = 1
        copied.write_text(json.dumps(payload))
        _expect_error(
            lambda: module.load_complete_gate_oracle(copied),
            "prerequisite artifact hash",
        )


def test_success_and_failure_clear_private_target():
    module = _load_module()
    target, tile_plan, expected, oracle, selected, other = (
        _synthetic_target()
    )

    def successful_loader(
        model,
        binding_plan,
        observed_tile_plan,
        checkpoint_dir,
        model_fingerprint,
    ):
        assert model is target.assembly.packed.model
        assert binding_plan is target.binding_plan
        assert observed_tile_plan is tile_plan
        assert checkpoint_dir == "/checkpoint"
        assert model_fingerprint == "a" * 64
        selected.copy_(expected)
        return SimpleNamespace(
            owner=SimpleNamespace(model=model),
            binding_plan=binding_plan,
            tile_plan=tile_plan,
            model_fingerprint=model_fingerprint,
            stats=SimpleNamespace(
                assigned_bindings=2,
                source_tensors=2,
                shard_count=1,
                tile_count=2,
                destination_bytes=8,
                materialized_bytes=8,
                peak_tile_bytes=4,
            ),
        )

    result = module.execute_and_clear_tiled_loader_core(
        target=target,
        tile_plan=tile_plan,
        checkpoint_dir="/checkpoint",
        model_fingerprint="a" * 64,
        oracle_row=oracle,
        load_core=successful_loader,
    )
    assert result["loaded_state_verified"] is True
    assert result["target_consumed_before"] is False
    assert result["target_consumed_after"] is False
    assert result["selected_destinations_initialized_zero"] is True
    assert result["all_selected_destinations_zero_after_clear"] is True
    assert result["non_selected_tensors_unchanged"] is True
    assert not int(selected.count_nonzero().item())
    torch.testing.assert_close(
        other, torch.tensor((0.5, 0.25), dtype=torch.float32)
    )

    calls = 0

    def failing_loader(*_args):
        nonlocal calls
        calls += 1
        selected[:2].fill_(7)
        raise RuntimeError("injected tiled loader failure")

    _expect_error(
        lambda: module.execute_and_clear_tiled_loader_core(
            target=target,
            tile_plan=tile_plan,
            checkpoint_dir="/checkpoint",
            model_fingerprint="a" * 64,
            oracle_row=oracle,
            load_core=failing_loader,
        ),
        "injected tiled loader failure",
    )
    assert calls == 1
    assert target._consumed is False
    assert not int(selected.count_nonzero().item())
    torch.testing.assert_close(
        other, torch.tensor((0.5, 0.25), dtype=torch.float32)
    )


def _memory(tp_size):
    values = (
        (25000, 370000, 500000, 510000, 4192000, 6200000)
        if tp_size == 1 else
        (25000, 370000, 500000, 505000, 2348000, 3500000)
    )
    return {
        name: {"vmrss_kib": value, "vmhwm_kib": value}
        for name, value in zip((
            "before", "after_torch", "after_metadata",
            "after_pool", "after_target", "after_load_and_clear",
        ), values, strict=True)
    }


def _row(module, tp_size, tp_rank, pid):
    contract = module.LOADER_CORE_CONTRACTS[(tp_size, tp_rank)]
    memory = _memory(tp_size)
    return {
        "schema_version": module.ROW_SCHEMA_VERSION,
        "status": "PASS",
        "tp_size": tp_size,
        "tp_rank": tp_rank,
        "process_id": pid,
        "observed_user": "sitian",
        "observed_hostname": "n232-195-203",
        "checkpoint_dir": module.APPROVED_MODEL_DIR,
        "prerequisite_artifact_sha256": (
            module.PREREQUISITE_ARTIFACT_SHA256
        ),
        "prerequisite_source_tree_sha256": (
            module.PREREQUISITE_SOURCE_TREE_SHA256
        ),
        "config_sha256": module.APPROVED_CONFIG_SHA256,
        "index_sha256": module.APPROVED_INDEX_SHA256,
        "config_index_header_sha256": module.APPROVED_COMPOSITE_SHA256,
        "metadata_bytes_read": 144024,
        "selected_binding_count": 320,
        "unique_destination_count": 296,
        "alias_groups": module.ALIAS_GROUPS,
        "binding_hash_count": 320,
        "phase_hash_count": 26,
        "aggregate_hash_verified": True,
        "loaded_state_verified": True,
        "loader_core_call_count": 1,
        "loader_stats": contract,
        "target_consumed_before": False,
        "target_consumed_after": False,
        "selected_destinations_initialized_zero": True,
        "all_selected_destinations_zero_after_clear": True,
        "tensor_identity_preserved": True,
        "pool_unchanged": True,
        "non_selected_tensors_unchanged": True,
        "model_forward_count": 0,
        "attention_forward_count": 0,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "memory": memory,
        "total_vmhwm_increment_kib": (
            memory["after_load_and_clear"]["vmhwm_kib"]
            - memory["before"]["vmhwm_kib"]
        ),
        "post_torch_vmhwm_increment_kib": (
            memory["after_load_and_clear"]["vmhwm_kib"]
            - memory["after_torch"]["vmhwm_kib"]
        ),
        "post_metadata_vmhwm_increment_kib": (
            memory["after_load_and_clear"]["vmhwm_kib"]
            - memory["after_metadata"]["vmhwm_kib"]
        ),
    }


def test_worker_row_contract_and_source_closure():
    module = _load_module()
    assert len(module.SOURCE_FILES) == len(set(module.SOURCE_FILES)) == 44
    assert module.MEMORY_CEILINGS_KIB == {
        1: {
            "total": 8388608,
            "post_torch": 7864320,
            "post_metadata": 7864320,
        },
        2: {
            "total": 4980736,
            "post_torch": 4718592,
            "post_metadata": 4456448,
        },
    }
    assert module.LOADER_CORE_CONTRACTS[(1, 0)] == {
        "assigned_bindings": 320,
        "source_tensors": 320,
        "shard_count": 1,
        "tile_count": 58169,
        "destination_bytes": 3763655360,
        "materialized_bytes": 3763655360,
        "peak_tile_bytes": 65536,
    }
    assert module.LOADER_CORE_CONTRACTS[(2, 1)]["tile_count"] == 29169
    row = _row(module, 2, 1, 701)
    assert module.validate_tiled_loader_core_row(row) == row
    invalid = dict(row)
    invalid["target_consumed_after"] = True
    _expect_error(
        lambda: module.validate_tiled_loader_core_row(invalid),
        "consumed",
    )
    invalid = dict(row)
    invalid["loader_core_call_count"] = 2
    _expect_error(
        lambda: module.validate_tiled_loader_core_row(invalid),
        "call count",
    )


def test_memory_ceiling_failure_reports_observed_and_allowed_deltas():
    module = _load_module()
    row = _row(module, 1, 0, 701)
    ceilings = module.MEMORY_CEILINGS_KIB[1]
    increments = {
        "total_vmhwm_increment_kib": ceilings["total"] + 3,
        "post_torch_vmhwm_increment_kib": ceilings["post_torch"] + 2,
        "post_metadata_vmhwm_increment_kib": (
            ceilings["post_metadata"] + 1
        ),
    }
    row.update(increments)
    final_vmhwm_kib = row["memory"]["after_load_and_clear"]["vmhwm_kib"]
    row["memory"]["before"]["vmhwm_kib"] = (
        final_vmhwm_kib - increments["total_vmhwm_increment_kib"]
    )
    row["memory"]["after_torch"]["vmhwm_kib"] = (
        final_vmhwm_kib - increments["post_torch_vmhwm_increment_kib"]
    )
    row["memory"]["after_metadata"]["vmhwm_kib"] = (
        final_vmhwm_kib
        - increments["post_metadata_vmhwm_increment_kib"]
    )

    expected = (
        "loader-core memory ceiling exceeded: "
        f"total={ceilings['total'] + 3}/{ceilings['total']} KiB, "
        f"post_torch={ceilings['post_torch'] + 2}/"
        f"{ceilings['post_torch']} KiB, "
        f"post_metadata={ceilings['post_metadata'] + 1}/"
        f"{ceilings['post_metadata']} KiB"
    )
    _expect_error(
        lambda: module.validate_tiled_loader_core_row(row),
        expected,
    )


def test_module_supports_local_python39_orchestration():
    source = MODULE_PATH.read_text()
    assert "strict=True" not in source


def _record(module):
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
        "prerequisite_artifact_sha256": (
            module.PREREQUISITE_ARTIFACT_SHA256
        ),
        "prerequisite_source_tree_sha256": (
            module.PREREQUISITE_SOURCE_TREE_SHA256
        ),
        "fresh_process_per_rank": True,
        "source_file_sha256": hashes,
        "source_tree_sha256": module._source_tree_sha256(hashes),
        "rows": [
            _row(module, 1, 0, 701),
            _row(module, 2, 0, 702),
            _row(module, 2, 1, 703),
        ],
    }


def test_source_staging_remote_atomicity_and_cli():
    module = _load_module()
    payload = module.build_source_tar(ROOT)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        assert tuple(archive.getnames()) == module.SOURCE_FILES
        assert len(archive.getmembers()) == 44
    record = _record(module)
    staged = {
        "remote_source_dir": "/remote/source",
        "remote_prerequisite_artifact": "/remote/prerequisite.json",
        "local_file_sha256": record["source_file_sha256"],
        "remote_file_sha256": record["source_file_sha256"],
        "source_tree_sha256": record["source_tree_sha256"],
        "prerequisite_artifact_sha256": (
            module.PREREQUISITE_ARTIFACT_SHA256
        ),
    }
    rows = iter(record["rows"])
    observed = []

    def runner(command, **kwargs):
        text = command[-1]
        observed.append(text)
        if "internal-rank-worker" in text:
            assert "--source-root /remote/source" in text
            assert (
                "--prerequisite-artifact /remote/prerequisite.json"
                in text
            )
            assert (
                f"--checkpoint-dir {module.APPROVED_MODEL_DIR}" in text
            )
            assert "CUDA_VISIBLE_DEVICES=" in text
            return subprocess.CompletedProcess(
                command, 0, json.dumps(next(rows)), ""
            )
        if "internal-finalize" in text:
            assert "--source-root /remote/source" in text
            assert (
                "--output "
                f"{module.REMOTE_RUN_ROOT}/loader-core-test/"
                "tiled_loader_core_preflight.json"
            ) in text
            return subprocess.CompletedProcess(
                command, 0, json.dumps(record), ""
            )
        manifest = json.loads(kwargs["input"])["source_manifest"]
        return subprocess.CompletedProcess(command, 0, json.dumps({
            "tiled_loader_core_preflight": record,
            "source_manifest": manifest,
        }), "")

    with tempfile.TemporaryDirectory() as temporary:
        result = module.run_remote_tiled_loader_core_preflight(
            ROOT,
            "loader-core-test",
            staged=staged,
            local_run_root=Path(temporary),
            command_runner=runner,
        )
        assert result == record
        assert len(observed) == 5
        assert {
            path.name
            for path in (
                Path(temporary) / "loader-core-test"
            ).iterdir()
        } == {
            "tiled_loader_core_preflight.json",
            "source_manifest.json",
        }

    calls = []

    def stage(
        source_root,
        run_tag,
        *,
        prerequisite_artifact,
        command_runner,
    ):
        calls.append(("stage", Path(source_root), run_tag))
        return staged

    def run(
        source_root,
        run_tag,
        *,
        staged,
        local_run_root,
        command_runner,
    ):
        calls.append(("run", Path(source_root), run_tag))
        return record

    original_stage = module.stage_source_and_prerequisite
    original_run = module.run_remote_tiled_loader_core_preflight
    module.stage_source_and_prerequisite = stage
    module.run_remote_tiled_loader_core_preflight = run
    try:
        result = module.execute_remote_tiled_loader_core_preflight(
            ROOT,
            "loader-core-execute",
            prerequisite_artifact=PREREQUISITE_ARTIFACT,
            local_run_root=ROOT / "experiments",
            command_runner=object(),
        )
    finally:
        module.stage_source_and_prerequisite = original_stage
        module.run_remote_tiled_loader_core_preflight = original_run
    assert result == record
    assert calls == [
        ("stage", ROOT, "loader-core-execute"),
        ("run", ROOT, "loader-core-execute"),
    ]
    try:
        module.main(["--help"])
    except SystemExit as error:
        assert error.code == 0
    else:
        raise AssertionError("CLI help must exit zero")


def main():
    tests = (
        test_exact_prerequisite_oracle_binding,
        test_success_and_failure_clear_private_target,
        test_worker_row_contract_and_source_closure,
        test_memory_ceiling_failure_reports_observed_and_allowed_deltas,
        test_module_supports_local_python39_orchestration,
        test_source_staging_remote_atomicity_and_cli,
    )
    for test in tests:
        test()
    print(
        "qwen35 real checkpoint tiled loader-core tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
