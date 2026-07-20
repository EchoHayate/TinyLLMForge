"""Dependency-light tests for the multi-sequence CUDA Graph gate contract."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "tools" / "multi_sequence_cuda_graph_contract.py"
DIAGNOSTIC_PATH = ROOT / "tools" / "diagnose_multi_sequence_cuda_graph.py"


def load_contract():
    spec = importlib.util.spec_from_file_location(
        "cuda_graph_contract",
        CONTRACT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = load_contract()


def load_diagnostic_module_without_gpu():
    spec = importlib.util.spec_from_file_location(
        "cuda_graph_diagnostic",
        DIAGNOSTIC_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    def encode(self, text):
        return [ord(character) for character in text]


def test_diagnostic_matrix_is_exact_and_unique():
    matrix = contract.build_diagnostic_matrix()
    assert len(matrix) == 189
    assert len({case.case_id for case in matrix}) == 189
    assert {case.batch_size for case in matrix} == {2, 3, 4, 5, 8, 9, 16}
    assert {case.trajectory for case in matrix} == {
        "uniform-short",
        "ragged-context",
        "duplicate-and-distinct",
    }
    assert {case.mode for case in matrix} == {
        "eager",
        "exact_graph",
        "rounded_graph",
    }
    assert {case.repetition for case in matrix} == {0, 1, 2}


def test_exact_and_rounded_graph_sizes_are_frozen():
    for batch_size in (2, 3, 4, 5, 8, 9, 16):
        assert contract.diagnostic_graph_size(batch_size, "eager") == batch_size
        assert (
            contract.diagnostic_graph_size(batch_size, "exact_graph")
            == batch_size
        )
    assert contract.ROUNDED_GRAPH_SIZE == {
        2: 4,
        3: 4,
        4: 8,
        5: 8,
        8: 16,
        9: 16,
        16: 32,
    }


def test_canonical_json_and_file_hashes_are_stable_and_strict():
    first = {"unicode": "图", "nested": {"z": 1, "a": [2, 3]}}
    second = {"nested": {"a": [2, 3], "z": 1}, "unicode": "图"}
    expected_bytes = (
        '{"nested":{"a":[2,3],"z":1},"unicode":"图"}'.encode("utf-8")
    )
    assert contract.canonical_json_bytes(first) == expected_bytes
    assert contract.canonical_json_bytes(second) == expected_bytes
    assert contract.canonical_json_sha256(first) == hashlib.sha256(
        expected_bytes
    ).hexdigest()

    with tempfile.TemporaryDirectory() as temporary_directory:
        artifact = Path(temporary_directory) / "artifact.bin"
        artifact.write_bytes(b"cuda-graph-evidence")
        assert contract.sha256_file(artifact) == hashlib.sha256(
            b"cuda-graph-evidence"
        ).hexdigest()

    try:
        contract.canonical_json_bytes({"invalid": float("nan")})
    except ValueError:
        pass
    else:
        raise AssertionError("canonical JSON accepted NaN")


def test_tensor_metadata_hashes_contiguous_bytes_and_reports_nonfinite():
    import torch

    source = torch.tensor(
        [[1.0, 2.0], [3.0, float("inf")]],
        dtype=torch.float32,
    ).transpose(0, 1)
    metadata = contract.tensor_metadata(source)
    contiguous = source.detach().cpu().contiguous().view(torch.uint8)
    assert metadata == {
        "dtype": "torch.float32",
        "shape": [2, 2],
        "finite": False,
        "sha256": hashlib.sha256(contiguous.numpy().tobytes()).hexdigest(),
    }


def test_graph_size_contract_rejects_unknown_inputs():
    for batch_size, mode, expected_message in (
        (1, "eager", "unsupported batch size"),
        (2, "larger_graph", "unsupported mode"),
    ):
        try:
            contract.diagnostic_graph_size(batch_size, mode)
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError((batch_size, mode))


def test_tensor_comparison_requires_finite_close_and_equal_argmax():
    import torch

    eager = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
    close = eager + torch.tensor([[0.001, -0.001], [0.001, 0.0]])
    result = contract.compare_tensor_pair(eager, close)
    assert result["finite"] is True
    assert result["argmax_equal"] is True
    assert result["close"] is True

    wrong_argmax = torch.tensor([[2.1, 2.0], [3.0, 1.0]])
    assert (
        contract.compare_tensor_pair(eager, wrong_argmax)["argmax_equal"]
        is False
    )

    nonfinite = eager.clone()
    nonfinite[0, 0] = float("nan")
    assert contract.compare_tensor_pair(eager, nonfinite)["finite"] is False


def make_complete_diagnostic_evidence():
    matrix_rows = []
    logit_results = []
    layer_results = []
    kv_results = []
    for case in contract.build_diagnostic_matrix():
        matrix_rows.append(
            {
                "case_id": case.case_id,
                "batch_size": case.batch_size,
                "trajectory": case.trajectory,
                "mode": case.mode,
                "repetition": case.repetition,
                "graph_size": case.graph_size,
                "status": "PASS",
            }
        )
        if case.mode == "eager":
            continue
        common = {
            "case_id": case.case_id,
            "mode": case.mode,
            "batch_size": case.batch_size,
            "graph_size": case.graph_size,
        }
        logit_results.append(
            {
                **common,
                "finite": True,
                "argmax_equal": True,
                "close": True,
            }
        )
        layer_results.append(
            {
                **common,
                "required_layer_count": 4,
                "observed_layer_count": 4,
                "finite": True,
                "close": True,
            }
        )
        kv_results.append(
            {
                **common,
                "active_slots_equal": True,
                "unexpected_slot_mutations": [],
            }
        )
    return {
        "matrix_rows": matrix_rows,
        "logit_results": logit_results,
        "layer_results": layer_results,
        "kv_results": kv_results,
    }


def _first_result_for_mode(rows, mode):
    return next(row for row in rows if row["mode"] == mode)


def test_diagnostic_classification_separates_exact_and_rounded():
    complete = make_complete_diagnostic_evidence()
    result = contract.classify_diagnostic(**complete)
    assert result["classification"] == "EXACT_REPLAY_CORRECT"
    assert result["rounded_classification"] == "ROUNDED_REPLAY_CORRECT"

    rounded_bad = make_complete_diagnostic_evidence()
    rounded_kv = _first_result_for_mode(
        rounded_bad["kv_results"],
        "rounded_graph",
    )
    rounded_kv["unexpected_slot_mutations"] = [0]
    result = contract.classify_diagnostic(**rounded_bad)
    assert result["classification"] == "EXACT_REPLAY_CORRECT"
    assert result["rounded_classification"] == "ROUNDED_REPLAY_CORRUPT"

    exact_bad = make_complete_diagnostic_evidence()
    exact_logits = _first_result_for_mode(
        exact_bad["logit_results"],
        "exact_graph",
    )
    exact_logits["close"] = False
    result = contract.classify_diagnostic(**exact_bad)
    assert result["classification"] == "EXACT_REPLAY_CORRUPT"

    incomplete = make_complete_diagnostic_evidence()
    incomplete["matrix_rows"].pop()
    result = contract.classify_diagnostic(**incomplete)
    assert result["classification"] == "INCOMPLETE"


def test_diagnostic_classification_rejects_duplicate_evidence():
    duplicate = make_complete_diagnostic_evidence()
    duplicate["logit_results"].append(dict(duplicate["logit_results"][0]))
    result = contract.classify_diagnostic(**duplicate)
    assert result["classification"] == "INCOMPLETE"


def make_complete_production_rows(**overrides):
    values = {
        "aggregate_decode_ratio": 1.15,
        "stable_decode_ratio": 1.25,
        "minimum_request_ratio": 0.95,
        "maximum_p95_itl_ratio": 1.05,
        "maximum_p99_itl_ratio": 1.10,
        "peak_reserved_ratio": 1.02,
        "initialization_ratio": 1.05,
        "stable_graph_hit_rate": 0.60,
    }
    values.update(overrides)
    return [
        {
            **values,
            "structural_failures": [],
            "correctness_failures": [],
            "measured_repetitions_complete": True,
        }
    ]


def test_production_gate_frozen_boundaries():
    rows = make_complete_production_rows()
    assert contract.classify_production_gate(rows)["classification"] == "GO"

    failing_values = {
        "aggregate_decode_ratio": 1.15 - 1e-6,
        "stable_decode_ratio": 1.25 - 1e-6,
        "minimum_request_ratio": 0.95 - 1e-6,
        "maximum_p95_itl_ratio": 1.05 + 1e-6,
        "maximum_p99_itl_ratio": 1.10 + 1e-6,
        "peak_reserved_ratio": 1.02 + 1e-6,
        "initialization_ratio": 1.05 + 1e-6,
        "stable_graph_hit_rate": 0.60 - 1e-6,
    }
    for field, failing_value in failing_values.items():
        rows = make_complete_production_rows(**{field: failing_value})
        result = contract.classify_production_gate(rows)
        assert result["classification"] == "NO_GO", field


def test_production_gate_fails_closed_on_structure_and_correctness():
    rows = make_complete_production_rows()
    rows[0]["structural_failures"] = ["missing graph evidence"]
    assert contract.classify_production_gate(rows)["classification"] == "NO_GO"

    rows = make_complete_production_rows()
    rows[0]["correctness_failures"] = ["token mismatch"]
    assert contract.classify_production_gate(rows)["classification"] == "NO_GO"

    rows = make_complete_production_rows()
    rows[0]["measured_repetitions_complete"] = False
    assert contract.classify_production_gate(rows)["classification"] == "NO_GO"


def test_prompt_plan_is_deterministic_and_covers_required_trajectories():
    diagnostic = load_diagnostic_module_without_gpu()
    tokenizer = FakeTokenizer()
    first = diagnostic.build_prompt_plan(tokenizer=tokenizer, batch_size=16)
    second = diagnostic.build_prompt_plan(tokenizer=tokenizer, batch_size=16)
    assert first == second
    assert set(first) == {
        "uniform-short",
        "ragged-context",
        "duplicate-and-distinct",
    }
    assert all(len(rows) == 16 for rows in first.values())
    ragged = first["ragged-context"]
    assert min(len(row) for row in ragged) < 256
    assert max(len(row) for row in ragged) > 256
    duplicate = first["duplicate-and-distinct"]
    assert duplicate[0] == duplicate[1]
    assert duplicate[0] != duplicate[2]


def test_kv_observation_plan_covers_active_zero_inactive_and_sentinels():
    diagnostic = load_diagnostic_module_without_gpu()
    plan = diagnostic.build_kv_observation_plan(
        active_slots=(300, 557, 814),
        graph_size=4,
        inactive_slots=(0,),
        total_slots=4096,
    )
    assert plan["active_write_slots"] == [300, 557, 814]
    assert plan["slot_zero"] == 0
    assert plan["inactive_declared_slots"] == [0]
    assert len(plan["sentinel_slots"]) >= 3
    assert set(plan["sentinel_slots"]).isdisjoint({0, 300, 557, 814})


def test_teacher_forcing_records_observed_and_reference_tokens_separately():
    diagnostic = load_diagnostic_module_without_gpu()
    row = diagnostic.build_step_row(
        observed_argmax_token_ids=[7, 8],
        reference_next_input_token_ids=[7, 9],
    )
    assert row["observed_argmax_token_ids"] == [7, 8]
    assert row["reference_next_input_token_ids"] == [7, 9]
    assert row["teacher_forcing_diverged"] is True


def test_tensor_shard_schema_rejects_missing_order_fields():
    diagnostic = load_diagnostic_module_without_gpu()
    shard = {
        "schema_version": 1,
        "case_id": "b2__uniform-short__eager__r0",
        "tensor": object(),
    }
    try:
        diagnostic.validate_tensor_shard(shard)
    except ValueError as exc:
        assert "step_ids" in str(exc)
    else:
        raise AssertionError("missing step_ids accepted")


def test_tensor_shard_schema_accepts_ordered_complete_metadata():
    diagnostic = load_diagnostic_module_without_gpu()
    shard = {
        "schema_version": 1,
        "case_id": "b2__uniform-short__eager__r0",
        "dtype": "torch.float32",
        "shape": [2, 2],
        "step_ids": [0, 1],
        "row_ids": [0, 1],
        "tensor": object(),
    }
    assert diagnostic.validate_tensor_shard(shard) is None


def test_direct_model_forward_disables_autograd():
    import torch

    diagnostic = load_diagnostic_module_without_gpu()

    class GradObservingModel:
        def __call__(self, input_ids, positions):
            del input_ids, positions
            return torch.tensor([torch.is_grad_enabled()])

    observed = diagnostic._forward_without_autograd(
        GradObservingModel(),
        torch.tensor([1]),
        torch.tensor([0]),
    )
    assert observed.tolist() == [False]


def test_direct_model_forward_and_logits_disable_autograd():
    import torch

    diagnostic = load_diagnostic_module_without_gpu()

    class GradObservingModel:
        def __call__(self, input_ids, positions):
            del input_ids, positions
            return torch.tensor([torch.is_grad_enabled()])

        def compute_logits(self, hidden_states):
            return torch.tensor(
                [
                    bool(hidden_states.item()),
                    torch.is_grad_enabled(),
                ]
            )

    observed = diagnostic._forward_and_logits_without_autograd(
        GradObservingModel(),
        torch.tensor([1]),
        torch.tensor([0]),
    )
    assert observed.tolist() == [False, False]


def test_artifact_record_path_is_relative_and_resolves_after_relocation():
    diagnostic = load_diagnostic_module_without_gpu()

    with tempfile.TemporaryDirectory() as temporary_directory:
        output_dir = Path(temporary_directory) / "remote-case-output"
        artifact = (
            output_dir
            / "tensors"
            / "logits"
            / "b2__uniform-short__eager__r0.pt"
        )
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"portable cuda graph artifact")

        record = diagnostic._artifact_record(
            output_dir=output_dir,
            path=artifact,
        )

        recorded_path = Path(record["path"])
        assert recorded_path == Path(
            "tensors/logits/b2__uniform-short__eager__r0.pt"
        )
        assert recorded_path.is_absolute() is False

        relocated_output_dir = (
            Path(temporary_directory) / "downloaded-case-output"
        )
        relocated_artifact = relocated_output_dir / recorded_path
        relocated_artifact.parent.mkdir(parents=True)
        relocated_artifact.write_bytes(artifact.read_bytes())

        assert relocated_artifact.is_file()
        assert record["sha256"] == contract.sha256_file(relocated_artifact)


if __name__ == "__main__":
    tests = [
        test_diagnostic_matrix_is_exact_and_unique,
        test_exact_and_rounded_graph_sizes_are_frozen,
        test_canonical_json_and_file_hashes_are_stable_and_strict,
        test_tensor_metadata_hashes_contiguous_bytes_and_reports_nonfinite,
        test_graph_size_contract_rejects_unknown_inputs,
        test_tensor_comparison_requires_finite_close_and_equal_argmax,
        test_diagnostic_classification_separates_exact_and_rounded,
        test_diagnostic_classification_rejects_duplicate_evidence,
        test_production_gate_frozen_boundaries,
        test_production_gate_fails_closed_on_structure_and_correctness,
        test_prompt_plan_is_deterministic_and_covers_required_trajectories,
        test_kv_observation_plan_covers_active_zero_inactive_and_sentinels,
        test_teacher_forcing_records_observed_and_reference_tokens_separately,
        test_tensor_shard_schema_rejects_missing_order_fields,
        test_tensor_shard_schema_accepts_ordered_complete_metadata,
        test_direct_model_forward_disables_autograd,
        test_direct_model_forward_and_logits_disable_autograd,
        test_artifact_record_path_is_relative_and_resolves_after_relocation,
    ]
    for test in tests:
        test()
    print("multi-sequence cuda graph gate tests passed")
