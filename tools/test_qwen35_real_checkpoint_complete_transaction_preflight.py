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
    ROOT / "tools/qwen35_real_checkpoint_complete_transaction_preflight.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_complete_transaction_preflight_under_test",
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
    values = (
        (25000, 370000, 500000, 510000, 4192000, 8500000)
        if tp_size == 1 else
        (25000, 370000, 500000, 505000, 2348000, 4900000)
    )
    return {
        name: {"vmrss_kib": value, "vmhwm_kib": value}
        for name, value in zip((
            "before", "after_torch", "after_metadata",
            "after_pool", "after_target", "after_payload",
        ), values, strict=True)
    }


def _binding_result(module, index, tp_size):
    contract = module.binding_contract(index, tp_size)
    digest = f"{index + 1:064x}"[-64:]
    return {
        **contract,
        "production_sha256": digest,
        "verifier_sha256": digest,
        "source_tensor_sha256": digest,
        "destination_sha256": digest,
        "coverage_complete": True,
    }


def _row(module, tp_size, tp_rank, pid):
    contract = module.COMPLETE_TRANSACTION_CONTRACTS[(tp_size, tp_rank)]
    memory = _memory(tp_size)
    digest = "f" * 64
    return {
        "schema_version": module.ROW_SCHEMA_VERSION,
        "status": "PASS",
        "tp_size": tp_size,
        "tp_rank": tp_rank,
        "process_id": pid,
        "observed_user": "sitian",
        "observed_hostname": "n232-195-203",
        "checkpoint_dir": module.APPROVED_MODEL_DIR,
        "config_sha256": module.APPROVED_CONFIG_SHA256,
        "index_sha256": module.APPROVED_INDEX_SHA256,
        "config_index_header_sha256": module.APPROVED_COMPOSITE_SHA256,
        "metadata_bytes_read": 144024,
        "layer_schedule": list(module.LAYER_SCHEDULE),
        "phase_names": list(module.PHASE_NAMES),
        "phase_binding_runs": [
            [name, list(indices)]
            for name, indices in module.PHASE_BINDING_RUNS
        ],
        "selected_binding_indices": list(range(320)),
        "selected_binding_count": 320,
        "unique_destination_count": 296,
        "alias_groups": module.ALIAS_GROUPS,
        "tile_count": contract["tile_count"],
        "kind_counts": contract["kind_counts"],
        "ranges_per_pass": contract["ranges_per_pass"],
        "production_payload_bytes_read": contract["bytes_per_pass"],
        "verifier_payload_bytes_read": contract["bytes_per_pass"],
        "logical_payload_bytes_read": contract["logical_bytes"],
        "open_count": 2,
        "pread_count": contract["pread_count"],
        "binding_results": [
            _binding_result(module, index, tp_size)
            for index in range(320)
        ],
        "phase_results": [
            {
                "phase_name": name,
                "binding_indices": list(indices),
                "binding_count": len(indices),
                **contract["phases"][name],
                "production_sha256": digest,
                "verifier_sha256": digest,
                "destination_sha256": digest,
                "coverage_complete": True,
            }
            for name, indices in module.PHASE_BINDING_RUNS
        ],
        "phase_completion_order": list(module.PHASE_NAMES),
        "transition_checks": [
            {
                "next_phase": name,
                "completed_phases_changed": True,
                "future_phases_zero": True,
            }
            for name in module.PHASE_NAMES[1:]
        ],
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
        "payload_identity_source": "retained_approved_manifest",
        "fresh_process_per_rank": True,
        "source_file_sha256": hashes,
        "source_tree_sha256": module._source_tree_sha256(hashes),
        "rows": [
            _row(module, 1, 0, 601),
            _row(module, 2, 0, 602),
            _row(module, 2, 1, 603),
        ],
    }


def _tile(module, binding, phase, destination, destination_slice):
    shape = (destination_slice.stop - destination_slice.start,)
    return module.Qwen35CheckpointTile(
        binding, f"source-{binding}", module.APPROVED_SHARD_NAME,
        shape, (slice(0, shape[0]),), shape, destination,
        (destination_slice,), shape, destination.dtype,
        shape[0] * destination.element_size(),
        phase, "axis0",
    )


def test_frozen_complete_contract_and_record():
    module = _load_module()
    assert len(module.SOURCE_FILES) == len(set(module.SOURCE_FILES)) == 43
    assert len(module.PHASE_BINDING_RUNS) == 26
    assert module.PHASE_NAMES[:4] == (
        "embed_tokens", "layer_0", "layer_1", "layer_10",
    )
    assert module.PHASE_NAMES[-2:] == ("layer_9", "final_norm")
    assert tuple(
        index
        for _, indices in module.PHASE_BINDING_RUNS
        for index in indices
    ) == tuple(range(320))
    assert len(module.UNIQUE_BINDING_ORDER) == 296
    assert len(module.ALIAS_GROUPS) == 24
    expected = {
        (1, 0): (58169, 3763655360, 58169, 116338, 7527310720),
        (2, 0): (29169, 1881935712, 121017, 242034, 3763871424),
        (2, 1): (29169, 1881935712, 121017, 242034, 3763871424),
    }
    for key, values in expected.items():
        contract = module.COMPLETE_TRANSACTION_CONTRACTS[key]
        assert (
            contract["tile_count"], contract["bytes_per_pass"],
            contract["ranges_per_pass"], contract["pread_count"],
            contract["logical_bytes"],
        ) == values
    assert module.validate_complete_checkpoint_preflight(
        _record(module)
    )["status"] == "PASS"


def test_root_and_non_numeric_layer_binding_contracts():
    module = _load_module()
    assert module.binding_contract(0, 1) == {
        "binding_index": 0,
        "phase_name": "embed_tokens",
        "source_name": "model.language_model.embed_tokens.weight",
        "target": "embed_tokens.weight",
        "kind": "axis0",
        "dtype": "torch.bfloat16",
        "local_shape": [248320, 2048],
        "destination_slice": None,
        "tile_count": 15520,
        "range_count": 15520,
        "byte_count": 1017118720,
    }
    assert module.binding_contract(0, 2)["local_shape"] == [124160, 2048]
    assert module.binding_contract(29, 1)["phase_name"] == "layer_10"
    assert module.binding_contract(160, 1)["phase_name"] == "layer_2"
    assert module.binding_contract(319, 2)["target"] == "final_norm.weight"
    _expect_error(lambda: module.binding_contract(320, 1), "binding")


def test_phase_selector_preserves_real_binding_order():
    module = _load_module()
    targets = (
        "embed_tokens.weight",
        "layers.0.input_layernorm.weight",
        "layers.10.input_layernorm.weight",
        "layers.2.input_layernorm.weight",
        "final_norm.weight",
    )
    bindings = tuple(
        SimpleNamespace(
            load=SimpleNamespace(
                weight=SimpleNamespace(target=target),
            ),
        )
        for target in targets
    )
    assert module._phase_names_for_bindings(bindings) == (
        "embed_tokens", "layer_0", "layer_10", "layer_2", "final_norm",
    )


def test_complete_phase_transitions_aliases_and_rollback():
    module = _load_module()
    destinations = [
        torch.zeros((2,), dtype=torch.bfloat16)
        for _ in module.PHASE_NAMES
    ]
    shared = torch.zeros((4,), dtype=torch.bfloat16)
    other = torch.zeros((3,), dtype=torch.float32)
    tiles = []
    payloads = []
    binding_phase = {}
    binding_order = []
    selected_ids = set()
    for offset, phase in enumerate(module.PHASE_NAMES):
        binding = offset + 1000
        tile = _tile(
            module, binding, phase, destinations[offset], slice(0, 2)
        )
        tiles.append(tile)
        payloads.append(torch.ones(tile.tile_shape, dtype=tile.dtype))
        binding_phase[binding] = phase
        binding_order.append(binding)
        selected_ids.add(id(tile.destination))
    left = _tile(module, 2000, module.PHASE_NAMES[1], shared, slice(0, 2))
    right = _tile(module, 2001, module.PHASE_NAMES[1], shared, slice(2, 4))
    tiles[2:2] = (left, right)
    payloads[2:2] = (
        torch.ones(left.tile_shape, dtype=left.dtype),
        torch.ones(right.tile_shape, dtype=right.dtype),
    )
    binding_phase[2000] = binding_phase[2001] = module.PHASE_NAMES[1]
    binding_order[2:2] = (2000, 2001)
    selected_ids.add(id(shared))
    result = module.apply_verify_and_rollback_complete_tiles(
        tiles,
        payloads,
        binding_order=tuple(binding_order),
        binding_phase=binding_phase,
        expected_phase_order=module.PHASE_NAMES,
        expected_alias_groups=[[2000, 2001]],
        unique_tensors=(*destinations, shared, other),
        selected_destination_ids=selected_ids,
    )
    assert result["phase_completion_order"] == list(module.PHASE_NAMES)
    assert len(result["transition_checks"]) == 25
    assert result["all_unique_tensors_zero_after_rollback"] is True
    gap = _tile(
        module, 2001, module.PHASE_NAMES[1], shared, slice(3, 4)
    )
    invalid_tiles = (*tiles[:3], gap, *tiles[4:])
    _expect_error(
        lambda: module.apply_verify_and_rollback_complete_tiles(
            invalid_tiles,
            payloads,
            binding_order=tuple(binding_order),
            binding_phase=binding_phase,
            expected_phase_order=module.PHASE_NAMES,
            expected_alias_groups=[[2000, 2001]],
            unique_tensors=(*destinations, shared, other),
            selected_destination_ids=selected_ids,
        ),
        "alias partition",
    )


def test_row_validation_rejects_phase_root_and_safety_drift():
    module = _load_module()
    row = _row(module, 2, 1, 603)
    assert module.validate_complete_checkpoint_row(row) == row
    cases = (
        ({"selected_binding_count": 319}, "binding"),
        ({"unique_destination_count": 295}, "destination"),
        ({"alias_groups": []}, "alias"),
        ({"phase_completion_order": sorted(module.PHASE_NAMES)}, "phase order"),
        ({"transition_checks": []}, "transition"),
        ({"non_selected_tensors_remained_zero": False}, "non-selected"),
        ({"rollback_binding_order": [0]}, "rollback"),
        ({"target_take_count": 1}, "target.take"),
        ({"cuda_initialized_after": True}, "CUDA"),
    )
    for update, message in cases:
        invalid = dict(row)
        invalid.update(update)
        _expect_error(
            lambda invalid=invalid: module.validate_complete_checkpoint_row(
                invalid
            ),
            message,
        )


def test_source_closure_and_remote_atomicity():
    module = _load_module()
    payload = module.build_source_tar(ROOT)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        assert tuple(archive.getnames()) == module.SOURCE_FILES
        assert len(archive.getmembers()) == 43
    record = _record(module)
    staged = {
        "remote_source_dir": "/remote/source",
        "local_file_sha256": record["source_file_sha256"],
        "remote_file_sha256": record["source_file_sha256"],
        "source_tree_sha256": record["source_tree_sha256"],
    }
    rows = iter(record["rows"])
    observed_commands = []

    def runner(command, **kwargs):
        text = command[-1]
        observed_commands.append((text, kwargs))
        if "internal-rank-worker" in text:
            assert "--source-root /remote/source" in text
            assert (
                f"--checkpoint-dir {module.APPROVED_MODEL_DIR}" in text
            )
            assert "CUDA_VISIBLE_DEVICES=" in text
            assert "OMP_NUM_THREADS=8" in text
            assert "MKL_NUM_THREADS=8" in text
            return subprocess.CompletedProcess(
                command, 0, json.dumps(next(rows)), ""
            )
        if "internal-finalize" in text:
            assert "--source-root /remote/source" in text
            assert (
                "--output "
                f"{module.REMOTE_RUN_ROOT}/complete-test/"
                "complete_checkpoint_transaction_preflight.json"
            ) in text
            return subprocess.CompletedProcess(
                command, 0, json.dumps(record), ""
            )
        assert "round-trip" not in text
        assert "complete_checkpoint_transaction_preflight.json" in text
        assert "source_manifest.json" in text
        manifest = json.loads(kwargs["input"])["source_manifest"]
        return subprocess.CompletedProcess(command, 0, json.dumps({
            "complete_checkpoint_transaction_preflight": record,
            "source_manifest": manifest,
        }), "")

    with tempfile.TemporaryDirectory() as temporary:
        result = module.run_remote_complete_checkpoint_preflight(
            ROOT, "complete-test", staged=staged,
            local_run_root=Path(temporary), command_runner=runner,
        )
        assert result == record
        assert {path.name for path in (
            Path(temporary) / "complete-test"
        ).iterdir()} == {
            "complete_checkpoint_transaction_preflight.json",
            "source_manifest.json",
        }
        assert len(observed_commands) == 5

    def fail_runner(command, **kwargs):
        if "internal-rank-worker" in command[-1]:
            return subprocess.CompletedProcess(command, 1, "", "injected")
        raise AssertionError("unexpected command after worker failure")

    with tempfile.TemporaryDirectory() as temporary:
        _expect_error(
            lambda: module.run_remote_complete_checkpoint_preflight(
                ROOT, "complete-fail", staged=staged,
                local_run_root=Path(temporary),
                command_runner=fail_runner,
            ),
            "injected",
        )
        assert not (Path(temporary) / "complete-fail").exists()


def test_execute_and_cli_interfaces():
    module = _load_module()
    assert callable(module.execute_remote_complete_checkpoint_preflight)
    assert callable(module._rank_worker_main)
    assert callable(module._finalize_main)

    calls = []

    def stage(source_root, run_tag, *, command_runner):
        calls.append(("stage", Path(source_root), run_tag, command_runner))
        return {"remote_source_dir": "/remote/source"}

    def run(
        source_root,
        run_tag,
        *,
        staged,
        local_run_root,
        command_runner,
    ):
        calls.append((
            "run",
            Path(source_root),
            run_tag,
            staged,
            Path(local_run_root),
            command_runner,
        ))
        return {"status": "PASS"}

    original_stage = module.stage_source
    original_run = module.run_remote_complete_checkpoint_preflight
    module.stage_source = stage
    module.run_remote_complete_checkpoint_preflight = run
    try:
        runner = object()
        result = module.execute_remote_complete_checkpoint_preflight(
            ROOT,
            "complete-execute",
            local_run_root=ROOT / "experiments",
            command_runner=runner,
        )
    finally:
        module.stage_source = original_stage
        module.run_remote_complete_checkpoint_preflight = original_run
    assert result == {"status": "PASS"}
    assert calls == [
        ("stage", ROOT, "complete-execute", runner),
        (
            "run",
            ROOT,
            "complete-execute",
            {"remote_source_dir": "/remote/source"},
            ROOT / "experiments",
            runner,
        ),
    ]

    try:
        module.main(["--help"])
    except SystemExit as error:
        assert error.code == 0
    else:
        raise AssertionError("CLI help must exit with status zero")


def main():
    tests = (
        test_frozen_complete_contract_and_record,
        test_root_and_non_numeric_layer_binding_contracts,
        test_phase_selector_preserves_real_binding_order,
        test_complete_phase_transitions_aliases_and_rollback,
        test_row_validation_rejects_phase_root_and_safety_drift,
        test_source_closure_and_remote_atomicity,
        test_execute_and_cli_interfaces,
    )
    for test in tests:
        test()
    print(
        "qwen35 complete checkpoint transaction tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
