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
    ROOT / "tools/qwen35_real_checkpoint_four_layer_cadence_preflight.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_four_layer_cadence_preflight_under_test",
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
        (25000, 370000, 500000, 510000, 4192000, 4700000)
        if tp_size == 1 else
        (25000, 370000, 500000, 505000, 2348000, 2700000)
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
    digest = f"{index:064x}"[-64:]
    return {
        **contract,
        "production_sha256": digest,
        "verifier_sha256": digest,
        "source_tensor_sha256": digest,
        "destination_sha256": digest,
        "coverage_complete": True,
    }


def _row(module, tp_size, tp_rank, pid):
    contract = module.FOUR_LAYER_CONTRACTS[(tp_size, tp_rank)]
    memory = _memory(tp_size)
    digest = "f" * 64
    layer_bindings = module.LAYER_BINDING_INDICES
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
        "selected_layer_indices": [0, 1, 2, 3],
        "selected_layer_types": [
            "linear_attention", "linear_attention",
            "linear_attention", "full_attention",
        ],
        "selected_binding_indices": list(module.SELECTED_BINDING_INDICES),
        "selected_binding_count": 53,
        "unique_destination_count": 49,
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
            for index in module.SELECTED_BINDING_INDICES
        ],
        "layer_results": [
            {
                "layer_index": layer,
                "layer_type": module.SELECTED_LAYER_TYPES[layer],
                "binding_indices": list(layer_bindings[layer]),
                "binding_count": len(layer_bindings[layer]),
                **contract["layers"][layer],
                "production_sha256": digest,
                "verifier_sha256": digest,
                "destination_sha256": digest,
                "coverage_complete": True,
            }
            for layer in range(4)
        ],
        "layer_completion_order": [0, 1, 2, 3],
        "transition_checks": [
            {
                "next_layer": layer,
                "completed_layers_changed": True,
                "future_layers_zero": True,
            }
            for layer in (1, 2, 3)
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
            _row(module, 1, 0, 501),
            _row(module, 2, 0, 502),
            _row(module, 2, 1, 503),
        ],
    }


def _tile(module, binding, layer, destination, destination_slice):
    shape = (destination_slice.stop - destination_slice.start,)
    return module.Qwen35CheckpointTile(
        binding, f"source-{binding}", module.APPROVED_SHARD_NAME,
        shape, (slice(0, shape[0]),), shape, destination,
        (destination_slice,), shape, destination.dtype,
        shape[0] * destination.element_size(),
        f"layers.{layer}.target", "axis0",
    )


def test_frozen_four_layer_contract_and_record():
    module = _load_module()
    assert len(module.SOURCE_FILES) == 42
    assert module.SELECTED_BINDING_INDICES == (
        *range(1, 29), *range(160, 174), *range(227, 238),
    )
    assert module.ALIAS_GROUPS == [
        [12, 13], [26, 27], [171, 172], [229, 230],
    ]
    expected = {
        (1, 0): (7108, 457755424, 7108, 14216, 915510848),
        (2, 0): (3568, 228895376, 18876, 37752, 457790752),
        (2, 1): (3568, 228895376, 18876, 37752, 457790752),
    }
    for key, values in expected.items():
        contract = module.FOUR_LAYER_CONTRACTS[key]
        assert (
            contract["tile_count"], contract["bytes_per_pass"],
            contract["ranges_per_pass"], contract["pread_count"],
            contract["logical_bytes"],
        ) == values
    assert module.validate_four_layer_cadence_preflight(
        _record(module)
    )["status"] == "PASS"


def test_binding_contract_mapping_for_non_contiguous_layers():
    module = _load_module()
    assert module.binding_contract(15, 1)["layer_index"] == 1
    assert module.binding_contract(160, 1)["layer_index"] == 2
    assert module.binding_contract(227, 1)["layer_index"] == 3
    assert module.binding_contract(171, 2)["destination_slice"] == [0, 3072]
    assert module.binding_contract(172, 2)["destination_slice"] == [
        3072, 3072,
    ]
    _expect_error(lambda: module.binding_contract(29, 1), "binding")


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
            "layers.1.linear_attention.A_log",
            "layers.2.mlp.down_proj.weight",
            "layers.3.full_attention.q_projection.weight",
            "layers.4.input_layernorm.weight",
            "final_norm.weight",
        )
    )
    assert module._selected_binding_indices(bindings) == (1, 2, 3, 4)


def test_four_layer_transitions_and_reverse_rollback():
    module = _load_module()
    destinations = [torch.zeros((2,), dtype=torch.bfloat16) for _ in range(4)]
    shared = [torch.zeros((4,), dtype=torch.bfloat16) for _ in range(4)]
    other = torch.zeros((3,), dtype=torch.float32)
    bindings = ((1, 12, 13), (15, 26, 27),
                (160, 171, 172), (227, 229, 230))
    tiles = []
    payloads = []
    binding_layer = {}
    binding_order = []
    selected_ids = set()
    for layer, (plain, left, right) in enumerate(bindings):
        group = (
            _tile(module, plain, layer, destinations[layer], slice(0, 2)),
            _tile(module, left, layer, shared[layer], slice(0, 2)),
            _tile(module, right, layer, shared[layer], slice(2, 4)),
        )
        for tile in group:
            tiles.append(tile)
            payloads.append(torch.ones(tile.tile_shape, dtype=tile.dtype))
            binding_layer[tile.binding_index] = layer
            binding_order.append(tile.binding_index)
            selected_ids.add(id(tile.destination))
    result = module.apply_verify_and_rollback_four_layer_tiles(
        tiles,
        payloads,
        binding_order=tuple(binding_order),
        binding_layer=binding_layer,
        unique_tensors=(*destinations, *shared, other),
        selected_destination_ids=selected_ids,
    )
    assert result["layer_completion_order"] == [0, 1, 2, 3]
    assert result["transition_checks"] == [
        {
            "next_layer": layer,
            "completed_layers_changed": True,
            "future_layers_zero": True,
        }
        for layer in (1, 2, 3)
    ]
    assert result["rollback_binding_order"] == [
        229, 227, 171, 160, 26, 15, 12, 1,
    ]
    assert result["all_unique_tensors_zero_after_rollback"] is True
    _expect_error(
        lambda: module.apply_verify_and_rollback_four_layer_tiles(
            (tiles[0], tiles[3], tiles[1], *tiles[2:3], *tiles[4:]),
            (payloads[0], payloads[3], payloads[1],
             *payloads[2:3], *payloads[4:]),
            binding_order=(
                bindings[0][0], bindings[1][0], bindings[0][1],
                bindings[0][2], *binding_order[4:],
            ),
            binding_layer=binding_layer,
            unique_tensors=(*destinations, *shared, other),
            selected_destination_ids=selected_ids,
        ),
        "layer order",
    )
    gap_tile = _tile(
        module,
        bindings[0][2],
        0,
        shared[0],
        slice(3, 4),
    )
    gap_tiles = (tiles[0], tiles[1], gap_tile, *tiles[3:])
    gap_payloads = (
        payloads[0],
        payloads[1],
        torch.ones(gap_tile.tile_shape, dtype=gap_tile.dtype),
        *payloads[3:],
    )
    _expect_error(
        lambda: module.apply_verify_and_rollback_four_layer_tiles(
            gap_tiles,
            gap_payloads,
            binding_order=tuple(binding_order),
            binding_layer=binding_layer,
            unique_tensors=(*destinations, *shared, other),
            selected_destination_ids=selected_ids,
        ),
        "alias partition",
    )


def test_row_validation_rejects_transition_and_safety_drift():
    module = _load_module()
    row = _row(module, 2, 1, 503)
    assert module.validate_four_layer_cadence_row(row) == row
    cases = (
        ({"selected_binding_count": 52}, "binding"),
        ({"unique_destination_count": 50}, "destination"),
        ({"alias_groups": [[12, 13]]}, "alias"),
        ({"layer_completion_order": [0, 2, 1, 3]}, "layer order"),
        ({"transition_checks": []}, "transition"),
        ({"non_selected_tensors_remained_zero": False}, "non-selected"),
        ({"rollback_binding_order": [1]}, "rollback"),
        ({"target_take_count": 1}, "target.take"),
        ({"cuda_initialized_after": True}, "CUDA"),
    )
    for update, message in cases:
        invalid = dict(row)
        invalid.update(update)
        _expect_error(
            lambda invalid=invalid: module.validate_four_layer_cadence_row(
                invalid
            ),
            message,
        )


def test_source_closure_and_remote_atomicity():
    module = _load_module()
    payload = module.build_source_tar(ROOT)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        assert tuple(archive.getnames()) == module.SOURCE_FILES
        assert len(archive.getmembers()) == 42
    record = _record(module)
    staged = {
        "remote_source_dir": "/remote/source",
        "local_file_sha256": record["source_file_sha256"],
        "remote_file_sha256": record["source_file_sha256"],
        "source_tree_sha256": record["source_tree_sha256"],
    }
    rows = iter(record["rows"])

    def runner(command, **kwargs):
        text = command[-1]
        if "internal-rank-worker" in text:
            return subprocess.CompletedProcess(
                command, 0, json.dumps(next(rows)), ""
            )
        if "internal-finalize" in text:
            return subprocess.CompletedProcess(
                command, 0, json.dumps(record), ""
            )
        manifest = json.loads(kwargs["input"])["source_manifest"]
        return subprocess.CompletedProcess(command, 0, json.dumps({
            "four_layer_cadence_preflight": record,
            "source_manifest": manifest,
        }), "")

    with tempfile.TemporaryDirectory() as temporary:
        result = module.run_remote_four_layer_cadence_preflight(
            ROOT, "four-layer-test", staged=staged,
            local_run_root=Path(temporary), command_runner=runner,
        )
        assert result == record
        assert {p.name for p in (
            Path(temporary) / "four-layer-test"
        ).iterdir()} == {
            "four_layer_cadence_preflight.json", "source_manifest.json",
        }
    def fail_runner(command, **kwargs):
        if "internal-rank-worker" in command[-1]:
            return subprocess.CompletedProcess(command, 1, "", "injected")
        raise AssertionError("unexpected command after worker failure")

    with tempfile.TemporaryDirectory() as temporary:
        _expect_error(
            lambda: module.run_remote_four_layer_cadence_preflight(
                ROOT,
                "four-layer-fail",
                staged=staged,
                local_run_root=Path(temporary),
                command_runner=fail_runner,
            ),
            "injected",
        )
        assert not (Path(temporary) / "four-layer-fail").exists()


def main():
    tests = (
        test_frozen_four_layer_contract_and_record,
        test_binding_contract_mapping_for_non_contiguous_layers,
        test_selected_binding_indices_use_exact_layer_prefixes,
        test_four_layer_transitions_and_reverse_rollback,
        test_row_validation_rejects_transition_and_safety_drift,
        test_source_closure_and_remote_atomicity,
    )
    for test in tests:
        test()
    print(
        "qwen35 four-layer cadence preflight tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
