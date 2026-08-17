from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tools/"
    "qwen35_real_checkpoint_private_candidate_ownership_preflight.py"
)
COMPLETE_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-complete-checkpoint-20260728-065128/"
    "complete_checkpoint_transaction_preflight.json"
)
LOADER_CORE_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-tiled-loader-core-20260728-075700/"
    "tiled_loader_core_preflight.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_private_candidate_ownership_preflight_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_prerequisite_oracles_and_tp_selection():
    module = _load_module()
    assert module.STREAMED_STATS == {
        (1, 0): {
            "assigned_bindings": 320,
            "source_tensors": 320,
            "shard_count": 1,
            "loaded_bytes": 3763655360,
            "peak_source_bytes": 1017118720,
        },
        (2, 0): {
            "assigned_bindings": 320,
            "source_tensors": 320,
            "shard_count": 1,
            "loaded_bytes": 3763655360,
            "peak_source_bytes": 1017118720,
        },
        (2, 1): {
            "assigned_bindings": 320,
            "source_tensors": 320,
            "shard_count": 1,
            "loaded_bytes": 3763655360,
            "peak_source_bytes": 1017118720,
        },
    }
    prerequisites = module.load_private_ownership_prerequisites(
        COMPLETE_ARTIFACT,
        LOADER_CORE_ARTIFACT,
    )

    assert prerequisites.complete_artifact_sha256 == (
        "7ed84bd1e4e894e9ea990ff2dc631db849f805f4468a9b266bd308d690acb176"
    )
    assert prerequisites.loader_core_artifact_sha256 == (
        "58df3dfa9fec11d1fd079c9473766413232bd3f928f537ac87e047e13ef65aae"
    )
    assert prerequisites.complete_source_tree_sha256 == (
        "da665b2de0aaa6533e55be0469c76ed39d92e817aabf80618f07b7efa7ef7042"
    )
    assert prerequisites.loader_core_source_tree_sha256 == (
        "c84eb9252bb5294d0fe00a4c48769e659274eb0a2d8c4548c25fb1ecdaf6869b"
    )
    assert tuple(prerequisites.complete_rows) == ((1, 0), (2, 0), (2, 1))
    assert tuple(prerequisites.loader_core_rows) == (
        (1, 0),
        (2, 0),
        (2, 1),
    )
    assert len(prerequisites.loader_core_source_file_sha256) == 44

    for tp_size, tp_rank in ((1, 0), (2, 0), (2, 1)):
        complete_row, loader_core_row = (
            module.select_private_ownership_prerequisite_rows(
                prerequisites,
                tp_size,
                tp_rank,
            )
        )
        assert (complete_row["tp_size"], complete_row["tp_rank"]) == (
            tp_size,
            tp_rank,
        )
        assert (
            loader_core_row["tp_size"],
            loader_core_row["tp_rank"],
        ) == (tp_size, tp_rank)
        assert len(complete_row["binding_results"]) == 320
        assert len(complete_row["phase_results"]) == 26
        assert len(loader_core_row["alias_groups"]) == 24
        assert loader_core_row["binding_hash_count"] == 320
        assert loader_core_row["phase_hash_count"] == 26
        assert loader_core_row["loaded_state_verified"] is True
        assert loader_core_row[
            "all_selected_destinations_zero_after_clear"
        ] is True
        assert loader_core_row["cuda_initialized_before"] is False
        assert loader_core_row["cuda_initialized_after"] is False


def _digest(tensor):
    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _synthetic_target():
    selected = torch.full((4,), 9, dtype=torch.bfloat16)
    rotary = torch.tensor((0.5, 0.25), dtype=torch.float32)
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
            ("rotary", rotary),
        ),
    )
    binding_plan = SimpleNamespace(
        bindings=bindings,
        tensor_parallel_size=1,
        tensor_parallel_rank=0,
    )
    pool = SimpleNamespace(marker="pool")
    target = SimpleNamespace(
        assembly=SimpleNamespace(
            packed=SimpleNamespace(model=model),
        ),
        binding_plan=binding_plan,
        pool=pool,
        _consumed=False,
    )
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
    return target, expected, oracle, selected, rotary


def _expect_error(callback, message):
    try:
        callback()
    except (TypeError, ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_success_and_failure_clear_consumed_private_target():
    module = _load_module()
    target, expected, oracle, selected, rotary = _synthetic_target()
    candidate = SimpleNamespace(
        owner=SimpleNamespace(
            model=target.assembly.packed.model,
            pool=target.pool,
        ),
        binding_plan=target.binding_plan,
        model_fingerprint="a" * 64,
        stats=SimpleNamespace(
            assigned_bindings=2,
            source_tensors=2,
            shard_count=1,
            loaded_bytes=8,
            peak_source_bytes=4,
        ),
    )

    def successful_adapter_call():
        assert target._consumed is False
        target._consumed = True
        selected.copy_(expected)
        return candidate

    result = module.execute_and_clear_private_candidate_ownership(
        target=target,
        model_fingerprint="a" * 64,
        oracle_row=oracle,
        adapter_call=successful_adapter_call,
    )
    assert result["loaded_state_verified"] is True
    assert result["binding_destination_sha256"] == [
        _digest(expected[:2]),
        _digest(expected[2:]),
    ]
    assert result["phase_destination_sha256"] == {
        "left": _digest(expected[:2]),
        "right": _digest(expected[2:]),
    }
    assert result["aggregate_destination_sha256"] == _digest(expected)
    assert result["target_consumed_before"] is False
    assert result["target_consumed_after"] is True
    assert result["candidate_returned"] is True
    assert result["all_selected_destinations_zero_after_clear"] is True
    assert result["non_selected_tensors_unchanged"] is True
    assert result["loader_stats"]["assigned_bindings"] == 2
    assert not int(selected.count_nonzero().item())
    torch.testing.assert_close(
        rotary,
        torch.tensor((0.5, 0.25), dtype=torch.float32),
    )

    target, _expected, oracle, selected, rotary = _synthetic_target()

    def failing_adapter_call():
        target._consumed = True
        selected[:2].fill_(7)
        raise RuntimeError(
            "injected ownership-transfer assignment failure"
        )

    _expect_error(
        lambda: module.execute_and_clear_private_candidate_ownership(
            target=target,
            model_fingerprint="a" * 64,
            oracle_row=oracle,
            adapter_call=failing_adapter_call,
        ),
        "injected ownership-transfer assignment failure",
    )
    assert target._consumed is True
    assert not int(selected.count_nonzero().item())
    torch.testing.assert_close(
        rotary,
        torch.tensor((0.5, 0.25), dtype=torch.float32),
    )

    target, _expected, oracle, selected, rotary = _synthetic_target()
    failure_evidence = {
        "assignment_call_count": 1,
        "first_source_name": "first.weight",
        "first_source_binding_count": 1,
        "first_source_binding_indices": [0],
        "first_source_binding_sha256": [
            _digest(torch.full((2,), 7, dtype=torch.bfloat16))
        ],
        "first_source_hashes_verified": True,
    }

    def expected_failing_adapter_call():
        target._consumed = True
        selected[:2].fill_(7)
        raise RuntimeError(
            "injected ownership-transfer assignment failure"
        )

    result = module.execute_and_clear_private_candidate_ownership(
        target=target,
        model_fingerprint="a" * 64,
        oracle_row=oracle,
        adapter_call=expected_failing_adapter_call,
        expected_error_message=(
            "injected ownership-transfer assignment failure"
        ),
        failure_evidence=failure_evidence,
    )
    assert result["expected_failure_observed"] is True
    assert result["candidate_returned"] is False
    assert result["target_consumed_after"] is True
    assert result["assignment_call_count"] == 1
    assert result["first_source_name"] == "first.weight"
    assert result["first_source_binding_indices"] == [0]
    assert result["first_source_binding_sha256"] == [
        _digest(torch.full((2,), 7, dtype=torch.bfloat16))
    ]
    assert result["all_selected_destinations_zero_after_clear"] is True
    assert not int(selected.count_nonzero().item())


def main():
    tests = (
        test_exact_prerequisite_oracles_and_tp_selection,
        test_success_and_failure_clear_consumed_private_target,
    )
    for test in tests:
        test()
    print(
        "qwen35 private candidate ownership tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
