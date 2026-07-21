"""Dependency-light tests for the multi-sequence CUDA Graph gate contract."""

from __future__ import annotations

import collections
import copy
import hashlib
import importlib.util
import json
import shlex
import sys
import tempfile
import types
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "tools" / "multi_sequence_cuda_graph_contract.py"
DIAGNOSTIC_PATH = ROOT / "tools" / "diagnose_multi_sequence_cuda_graph.py"
VERIFIER_PATH = (
    ROOT / "tools" / "verify_multi_sequence_cuda_graph_diagnostic.py"
)
REMOTE_RUNNER_PATH = (
    ROOT / "tools" / "run_multi_sequence_cuda_graph_diagnostic_remote.py"
)
SPLIT_POLICY_PATH = (
    ROOT / "tinyvllm" / "engine" / "flash_attn_split_policy.py"
)


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


def load_split_policy():
    spec = importlib.util.spec_from_file_location(
        "flash_attn_split_policy",
        SPLIT_POLICY_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_diagnostic_module_without_gpu():
    spec = importlib.util.spec_from_file_location(
        "cuda_graph_diagnostic",
        DIAGNOSTIC_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install_fake_context_module():
    module_name = "tinyvllm.utils.context"
    tinyvllm_package = types.ModuleType("tinyvllm")
    tinyvllm_package.__path__ = [str(ROOT / "tinyvllm")]
    utils_package = types.ModuleType("tinyvllm.utils")
    utils_package.__path__ = [str(ROOT / "tinyvllm" / "utils")]
    context_module = types.ModuleType(module_name)
    state = types.SimpleNamespace(flash_attn_num_splits=0)

    def get_context():
        return state

    def reset_context():
        state.flash_attn_num_splits = 0

    @contextmanager
    def temporary_flash_attn_num_splits(num_splits):
        previous = state.flash_attn_num_splits
        state.flash_attn_num_splits = int(num_splits)
        try:
            yield state
        finally:
            state.flash_attn_num_splits = previous

    context_module.get_context = get_context
    context_module.reset_context = reset_context
    context_module.temporary_flash_attn_num_splits = (
        temporary_flash_attn_num_splits
    )
    sys.modules["tinyvllm"] = tinyvllm_package
    sys.modules["tinyvllm.utils"] = utils_package
    sys.modules[module_name] = context_module
    return context_module


def load_verifier():
    spec = importlib.util.spec_from_file_location(
        "cuda_graph_independent_verifier",
        VERIFIER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_remote_runner():
    spec = importlib.util.spec_from_file_location(
        "cuda_graph_remote_runner",
        REMOTE_RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    def encode(self, text):
        return [ord(character) for character in text]


class FakeSizedTensor:
    def __init__(self, shape, *, device="cuda:0"):
        self._shape = tuple(shape)
        self.device = device

    def size(self, dimension=None):
        if dimension is None:
            return self._shape
        return self._shape[dimension]


def make_fake_policy_runner():
    hf_config = types.SimpleNamespace(
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
    )
    return types.SimpleNamespace(
        config=types.SimpleNamespace(hf_config=hf_config),
        block_size=256,
        kv_cache=FakeSizedTensor((2, 28, 64, 256, 128)),
    )


@contextmanager
def fake_cuda_device_properties(multi_processor_count=108):
    import torch

    original = torch.cuda.get_device_properties
    torch.cuda.get_device_properties = lambda device: types.SimpleNamespace(
        multi_processor_count=multi_processor_count,
    )
    try:
        yield
    finally:
        torch.cuda.get_device_properties = original


def test_flash_attn_263_known_qwen3_a100_vectors():
    split_policy = load_split_policy()
    vectors = {
        (2, 1): 2,
        (3, 1): 2,
        (4, 1): 2,
        (5, 1): 2,
        (8, 2): 2,
        (9, 2): 2,
        (16, 3): 3,
    }
    for (batch_size, page_table_width), expected in vectors.items():
        inputs = split_policy.FlashAttentionSplitInputs(
            batch_size=batch_size,
            num_query_heads=16,
            num_kv_heads=8,
            head_dim=128,
            page_block_size=256,
            page_table_width=page_table_width,
            max_seqlen_q=1,
            multi_processor_count=108,
        )
        assert (
            split_policy.flash_attn_263_decode_num_splits(inputs)
            == expected
        )


def test_flash_attn_263_early_return_and_graph_identity():
    split_policy = load_split_policy()
    inputs = split_policy.FlashAttentionSplitInputs(
        batch_size=22,
        num_query_heads=16,
        num_kv_heads=8,
        head_dim=128,
        page_block_size=256,
        page_table_width=3,
        max_seqlen_q=1,
        multi_processor_count=108,
    )
    assert split_policy.flash_attn_263_decode_num_splits(inputs) == 1

    first = split_policy.build_flash_attn_263_graph_identity(
        graph_batch_size=22,
        inputs=inputs,
        flash_attn_version="2.6.3",
    )
    second = split_policy.build_flash_attn_263_graph_identity(
        graph_batch_size=22,
        inputs=inputs,
        flash_attn_version="2.6.3",
    )
    assert first == second
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64

    wider = split_policy.build_flash_attn_263_graph_identity(
        graph_batch_size=22,
        inputs=split_policy.FlashAttentionSplitInputs(
            **{
                **asdict(inputs),
                "page_table_width": 4,
            }
        ),
        flash_attn_version="2.6.3",
    )
    rounded = split_policy.build_flash_attn_263_graph_identity(
        graph_batch_size=32,
        inputs=inputs,
        flash_attn_version="2.6.3",
    )
    assert first.sha256 != wider.sha256
    assert first.sha256 != rounded.sha256


def test_flash_attn_263_rejects_unsupported_inputs():
    split_policy = load_split_policy()
    valid = {
        "batch_size": 8,
        "num_query_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "page_block_size": 256,
        "page_table_width": 2,
        "max_seqlen_q": 1,
        "multi_processor_count": 108,
    }
    invalid = (
        ("batch_size", 0),
        ("num_query_heads", 0),
        ("num_kv_heads", 0),
        ("head_dim", 0),
        ("head_dim", 257),
        ("page_block_size", 128),
        ("page_table_width", 0),
        ("max_seqlen_q", 2),
        ("multi_processor_count", 0),
        ("num_query_heads", 15),
    )
    for field, value in invalid:
        values = dict(valid)
        values[field] = value
        inputs = split_policy.FlashAttentionSplitInputs(**values)
        try:
            split_policy.flash_attn_263_decode_num_splits(inputs)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"unsupported split input accepted: {field}={value}"
            )

    try:
        split_policy.build_flash_attn_263_graph_identity(
            graph_batch_size=8,
            inputs=split_policy.FlashAttentionSplitInputs(**valid),
            flash_attn_version="2.6.4",
        )
    except ValueError as exc:
        assert "2.6.3" in str(exc)
    else:
        raise AssertionError("unsupported FlashAttention version accepted")


def test_same_policy_matrix_is_exact_policy_aware_and_unique():
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
        "candidate_eager_heuristic",
        "exact_graph_heuristic",
        "rounded_graph_heuristic",
    }
    assert {case.repetition for case in matrix} == {0, 1, 2}
    assert {case.split_policy_name for case in matrix} == {
        contract.HEURISTIC_POLICY_NAME
    }
    assert all(case.flash_attn_num_splits is None for case in matrix)
    assert all("fa2-263-exact-width" in case.case_id for case in matrix)
    assert all("-s16" not in case.case_id for case in matrix)


def test_legacy_compatibility_matrix_is_63_pairs_126_processes():
    matrix = contract.build_legacy_compatibility_matrix()
    assert len(matrix) == 126
    assert len({case.case_id for case in matrix}) == 126
    pair_counts = collections.Counter(case.pair_id for case in matrix)
    assert len(pair_counts) == 63
    assert set(pair_counts.values()) == {2}
    assert {
        (case.policy, case.split_policy_name, case.flash_attn_num_splits)
        for case in matrix
    } == {
        ("legacy_eager_auto", "auto", 0),
        (
            "candidate_eager_heuristic",
            contract.HEURISTIC_POLICY_NAME,
            None,
        ),
    }


def test_case_ids_bind_split_policy_identity():
    case = contract.build_diagnostic_matrix()[0]
    assert "__fa2-263-exact-width__r" in case.case_id
    assert "fa2-263-exact-width-s" not in case.case_id
    compatibility = contract.build_legacy_compatibility_matrix()
    assert any("auto-s0" in case.case_id for case in compatibility)
    assert any(
        "__fa2-263-exact-width" in case.case_id
        and "fa2-263-exact-width-s" not in case.case_id
        for case in compatibility
        if case.policy == "candidate_eager_heuristic"
    )


def test_gate_matrix_cardinality_and_frozen_thresholds():
    diagnostic = contract.build_diagnostic_matrix()
    compatibility = contract.build_legacy_compatibility_matrix()
    all_cases = diagnostic + compatibility

    assert len(diagnostic) == 189
    assert len(compatibility) == 126
    assert len(all_cases) == 315
    assert len({case.case_id for case in all_cases}) == 315
    assert {
        case.batch_size for case in all_cases
    } == {2, 3, 4, 5, 8, 9, 16}
    assert {
        case.trajectory for case in all_cases
    } == {
        "uniform-short",
        "ragged-context",
        "duplicate-and-distinct",
    }
    assert {case.repetition for case in all_cases} == {0, 1, 2}
    assert contract.WARMUP_STEPS == 2
    assert contract.MEASURED_STEPS == 16
    assert contract.LOGIT_RTOL == 1e-3
    assert contract.LOGIT_ATOL == 1e-2


def test_exact_and_rounded_graph_sizes_are_frozen():
    for batch_size in (2, 3, 4, 5, 8, 9, 16):
        assert (
            contract.diagnostic_graph_size(
                batch_size,
                "candidate_eager_heuristic",
            )
            == batch_size
        )
        assert (
            contract.diagnostic_graph_size(
                batch_size,
                "exact_graph_heuristic",
            )
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
        (
            1,
            "candidate_eager_heuristic",
            "unsupported batch size",
        ),
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
        if case.mode == "candidate_eager_heuristic":
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
        "rounded_graph_heuristic",
    )
    rounded_kv["unexpected_slot_mutations"] = [0]
    result = contract.classify_diagnostic(**rounded_bad)
    assert result["classification"] == "EXACT_REPLAY_CORRECT"
    assert result["rounded_classification"] == "ROUNDED_REPLAY_CORRUPT"

    exact_bad = make_complete_diagnostic_evidence()
    exact_logits = _first_result_for_mode(
        exact_bad["logit_results"],
        "exact_graph_heuristic",
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


def make_complete_legacy_compatibility_evidence():
    process_rows = []
    logit_results = []
    kv_results = []
    token_results = []
    for case in contract.build_legacy_compatibility_matrix():
        process_rows.append(
            {
                "case_id": case.case_id,
                "pair_id": case.pair_id,
                "batch_size": case.batch_size,
                "trajectory": case.trajectory,
                "policy": case.policy,
                "repetition": case.repetition,
                "split_policy_name": case.split_policy_name,
                "flash_attn_num_splits": case.flash_attn_num_splits,
                "comparison_policy_name": "legacy_auto_vs_heuristic",
                "status": "PASS",
            }
        )
        if case.policy != "candidate_eager_heuristic":
            continue
        common = {
            "pair_id": case.pair_id,
            "batch_size": case.batch_size,
            "trajectory": case.trajectory,
            "repetition": case.repetition,
            "comparison_policy_name": "legacy_auto_vs_heuristic",
        }
        logit_results.append(
            {
                **common,
                "finite": True,
                "argmax_equal": True,
                "close": True,
            }
        )
        kv_results.append(
            {
                **common,
                "touched_slot_sets_equal": True,
                "unexpected_slot_mutations": [],
            }
        )
        token_results.append(
            {
                **common,
                "tokens_equal": True,
            }
        )
    return {
        "process_rows": process_rows,
        "logit_results": logit_results,
        "kv_results": kv_results,
        "token_results": token_results,
    }


def test_legacy_compatibility_requires_tokens_close_logits_and_kv_ownership():
    complete = make_complete_legacy_compatibility_evidence()
    assert (
        contract.classify_legacy_compatibility(**complete)["classification"]
        == "LEGACY_COMPATIBLE"
    )

    token_bad = copy.deepcopy(complete)
    token_bad["token_results"][0]["tokens_equal"] = False
    assert (
        contract.classify_legacy_compatibility(**token_bad)[
            "classification"
        ]
        == "LEGACY_INCOMPATIBLE"
    )

    close_bad = copy.deepcopy(complete)
    close_bad["logit_results"][0]["close"] = False
    assert (
        contract.classify_legacy_compatibility(**close_bad)[
            "classification"
        ]
        == "LEGACY_INCOMPATIBLE"
    )

    kv_bad = copy.deepcopy(complete)
    kv_bad["kv_results"][0]["touched_slot_sets_equal"] = False
    assert (
        contract.classify_legacy_compatibility(**kv_bad)["classification"]
        == "LEGACY_INCOMPATIBLE"
    )


def test_legacy_compatibility_missing_or_mixed_policy_is_incomplete():
    incomplete = make_complete_legacy_compatibility_evidence()
    incomplete["process_rows"].pop()
    result = contract.classify_legacy_compatibility(**incomplete)
    assert result["classification"] == "INCOMPLETE"

    mixed = make_complete_legacy_compatibility_evidence()
    mixed["process_rows"][0]["flash_attn_num_splits"] = 16
    result = contract.classify_legacy_compatibility(**mixed)
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


def test_build_step_split_policy_uses_exact_runtime_width_and_batch_identity():
    diagnostic = load_diagnostic_module_without_gpu()
    runner = make_fake_policy_runner()
    dynamic_context = {"block_tables": FakeSizedTensor((8, 2))}

    with fake_cuda_device_properties():
        policy = diagnostic.build_step_split_policy(
            runner=runner,
            dynamic_context=dynamic_context,
            active_batch_size=8,
            graph_batch_size=16,
            flash_attn_version="2.6.3",
        )

    assert policy.inputs.page_table_width == 2
    assert policy.inputs.batch_size == 8
    assert policy.identity.active_batch_size == 8
    assert policy.identity.graph_batch_size == 16
    assert policy.identity.page_table_width == 2
    assert policy.effective_num_splits == 2


def test_build_step_split_policy_matches_qwen3_a100_vectors():
    diagnostic = load_diagnostic_module_without_gpu()
    runner = make_fake_policy_runner()
    vectors = (
        (2, 1, 2),
        (8, 2, 2),
        (9, 2, 2),
        (16, 3, 3),
    )

    with fake_cuda_device_properties():
        for active_batch_size, page_table_width, expected in vectors:
            policy = diagnostic.build_step_split_policy(
                runner=runner,
                dynamic_context={
                    "block_tables": FakeSizedTensor(
                        (active_batch_size, page_table_width)
                    )
                },
                active_batch_size=active_batch_size,
                graph_batch_size=active_batch_size,
                flash_attn_version="2.6.3",
            )
            assert policy.effective_num_splits == expected


def test_build_step_split_policy_rejects_version_and_missing_inputs():
    diagnostic = load_diagnostic_module_without_gpu()
    runner = make_fake_policy_runner()
    dynamic_context = {"block_tables": FakeSizedTensor((8, 2))}

    with fake_cuda_device_properties():
        try:
            diagnostic.build_step_split_policy(
                runner=runner,
                dynamic_context=dynamic_context,
                active_batch_size=8,
                graph_batch_size=8,
                flash_attn_version="2.6.4",
            )
        except ValueError as exc:
            assert "2.6.3" in str(exc)
        else:
            raise AssertionError("unsupported FlashAttention version accepted")

        for field in (
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        ):
            broken = make_fake_policy_runner()
            delattr(broken.config.hf_config, field)
            try:
                diagnostic.build_step_split_policy(
                    runner=broken,
                    dynamic_context=dynamic_context,
                    active_batch_size=8,
                    graph_batch_size=8,
                    flash_attn_version="2.6.3",
                )
            except ValueError as exc:
                assert field in str(exc)
            else:
                raise AssertionError(f"missing {field} accepted")

        broken = make_fake_policy_runner()
        delattr(broken, "block_size")
        try:
            diagnostic.build_step_split_policy(
                runner=broken,
                dynamic_context=dynamic_context,
                active_batch_size=8,
                graph_batch_size=8,
                flash_attn_version="2.6.3",
            )
        except ValueError as exc:
            assert "block_size" in str(exc)
        else:
            raise AssertionError("missing block_size accepted")

    import torch

    original = torch.cuda.get_device_properties
    torch.cuda.get_device_properties = lambda device: types.SimpleNamespace()
    try:
        try:
            diagnostic.build_step_split_policy(
                runner=runner,
                dynamic_context=dynamic_context,
                active_batch_size=8,
                graph_batch_size=8,
                flash_attn_version="2.6.3",
            )
        except ValueError as exc:
            assert "multi_processor_count" in str(exc)
        else:
            raise AssertionError("missing multi_processor_count accepted")
    finally:
        torch.cuda.get_device_properties = original


def make_graph_identity(
    *,
    active_batch_size=8,
    graph_batch_size=8,
    page_table_width=2,
    effective_num_splits=2,
):
    split_policy = load_split_policy()
    return split_policy.FlashAttentionGraphIdentity(
        graph_batch_size=graph_batch_size,
        active_batch_size=active_batch_size,
        page_table_width=page_table_width,
        effective_num_splits=effective_num_splits,
        flash_attn_version="2.6.3",
        multi_processor_count=108,
        num_query_heads=16,
        num_kv_heads=8,
        head_dim=128,
        page_block_size=256,
        max_seqlen_q=1,
    )


def test_decode_graph_allocations_use_exact_identity_width():
    diagnostic = load_diagnostic_module_without_gpu()
    runner = make_fake_policy_runner()
    runner.config.hf_config.hidden_size = 1024
    runner.config.hf_config.torch_dtype = "bf16"
    runner.config.hf_config.num_hidden_layers = 28
    identity = make_graph_identity(
        active_batch_size=8,
        graph_batch_size=16,
        page_table_width=3,
    )
    allocations = []

    class FakeTorch:
        int64 = "int64"
        int32 = "int32"

        @staticmethod
        def zeros(*shape, dtype, device):
            allocations.append((shape, dtype, device))
            return FakeSizedTensor(shape, device=device)

    tensors = diagnostic._allocate_decode_graph_tensors(
        runner,
        identity,
        torch_module=FakeTorch,
    )

    assert tensors["block_tables"].size() == (16, 3)
    assert ((16, 3), "int32", "cuda:0") in allocations
    assert all(
        shape != (16, 4)
        for shape, _dtype, _device in allocations
    )


def test_graph_cache_reuses_only_identical_identity():
    diagnostic = load_diagnostic_module_without_gpu()
    graph_cache = {}
    captures = []

    def capture(identity):
        captures.append(identity)
        return types.SimpleNamespace(identity=identity)

    first_identity = make_graph_identity()
    first = diagnostic._get_or_capture_decode_graph(
        graph_cache,
        first_identity,
        capture,
    )
    reused = diagnostic._get_or_capture_decode_graph(
        graph_cache,
        first_identity,
        capture,
    )
    wider_identity = make_graph_identity(page_table_width=3)
    wider = diagnostic._get_or_capture_decode_graph(
        graph_cache,
        wider_identity,
        capture,
    )
    split_identity = make_graph_identity(effective_num_splits=3)
    split = diagnostic._get_or_capture_decode_graph(
        graph_cache,
        split_identity,
        capture,
    )

    assert first is reused
    assert wider is not first
    assert split is not first
    assert captures == [first_identity, wider_identity, split_identity]
    assert len(graph_cache) == 3


def test_graph_cache_and_replay_reject_identity_mismatch():
    diagnostic = load_diagnostic_module_without_gpu()
    expected = make_graph_identity()
    mismatched = make_graph_identity(page_table_width=3)

    try:
        diagnostic._get_or_capture_decode_graph(
            {},
            expected,
            lambda identity: types.SimpleNamespace(identity=mismatched),
        )
    except ValueError as exc:
        assert "identity" in str(exc)
    else:
        raise AssertionError("mismatched captured graph was cached")

    try:
        diagnostic._validate_graph_replay_identity(
            types.SimpleNamespace(identity=expected),
            mismatched,
        )
    except ValueError as exc:
        assert "identity" in str(exc)
    else:
        raise AssertionError("mismatched graph replay was accepted")


def test_capture_operation_restores_all_active_write_slots_on_failure():
    diagnostic = load_diagnostic_module_without_gpu()
    runner = types.SimpleNamespace()
    restored = []
    snapshot = {"keys": object(), "values": object()}

    def snapshot_kv_slots(slots):
        assert slots == [7, 11, 19]
        return snapshot

    runner.snapshot_kv_slots = snapshot_kv_slots
    original_restore = diagnostic._restore_kv_slots
    diagnostic._restore_kv_slots = (
        lambda observed_runner, slots, observed_snapshot: restored.append(
            (observed_runner, slots, observed_snapshot)
        )
    )
    try:
        try:
            diagnostic._run_with_kv_slot_restore(
                runner,
                [19, 7, 11, 7],
                lambda: (_ for _ in ()).throw(RuntimeError("capture failed")),
            )
        except RuntimeError as exc:
            assert str(exc) == "capture failed"
        else:
            raise AssertionError("capture failure was swallowed")
    finally:
        diagnostic._restore_kv_slots = original_restore

    assert restored == [(runner, [7, 11, 19], snapshot)]


def test_step_policy_evidence_is_complete_and_stable():
    diagnostic = load_diagnostic_module_without_gpu()
    identity = make_graph_identity(
        active_batch_size=8,
        graph_batch_size=16,
        page_table_width=3,
        effective_num_splits=2,
    )
    policy = diagnostic.StepSplitPolicy(
        inputs=load_split_policy().FlashAttentionSplitInputs(
            batch_size=8,
            num_query_heads=16,
            num_kv_heads=8,
            head_dim=128,
            page_block_size=256,
            page_table_width=3,
            max_seqlen_q=1,
            multi_processor_count=108,
        ),
        identity=identity,
    )
    expected = {
        "split_policy_name": "fa2_263_heuristic_exact_width",
        "flash_attn_version": "2.6.3",
        "page_table_width": 3,
        "effective_num_splits": 2,
        "heuristic_batch_size": 8,
        "heuristic_num_query_heads": 16,
        "heuristic_num_kv_heads": 8,
        "heuristic_head_dim": 128,
        "heuristic_page_block_size": 256,
        "heuristic_max_seqlen_q": 1,
        "heuristic_multi_processor_count": 108,
        "graph_batch_size": 16,
        "graph_identity_sha256": identity.sha256,
    }

    assert diagnostic.step_policy_evidence(policy) == expected
    assert diagnostic.step_policy_evidence(policy) == expected


def test_step_policy_evidence_matches_raw_layer_and_kv_rows():
    diagnostic = load_diagnostic_module_without_gpu()
    identity = make_graph_identity()
    policy = diagnostic.StepSplitPolicy(
        inputs=load_split_policy().FlashAttentionSplitInputs(
            batch_size=8,
            num_query_heads=16,
            num_kv_heads=8,
            head_dim=128,
            page_block_size=256,
            page_table_width=2,
            max_seqlen_q=1,
            multi_processor_count=108,
        ),
        identity=identity,
    )
    rows = diagnostic.build_step_policy_rows(
        policy,
        raw={"kind": "raw"},
        layer={"kind": "layer"},
        kv={"kind": "kv"},
    )
    evidence = diagnostic.step_policy_evidence(policy)

    assert rows["raw"] == {"kind": "raw", **evidence}
    assert rows["layer"] == {"kind": "layer", **evidence}
    assert rows["kv"] == {"kind": "kv", **evidence}


def test_graph_identity_summary_is_ordered_unique_and_complete():
    diagnostic = load_diagnostic_module_without_gpu()
    first = make_graph_identity()
    wider = make_graph_identity(page_table_width=3)
    split = make_graph_identity(effective_num_splits=3)

    assert diagnostic.graph_identity_summary(
        [first, first, wider, split, wider]
    ) == [
        {
            "sha256": first.sha256,
            "page_table_width": 2,
            "effective_num_splits": 2,
            "graph_batch_size": 8,
        },
        {
            "sha256": wider.sha256,
            "page_table_width": 3,
            "effective_num_splits": 2,
            "graph_batch_size": 8,
        },
        {
            "sha256": split.sha256,
            "page_table_width": 2,
            "effective_num_splits": 3,
            "graph_batch_size": 8,
        },
    ]


def make_complete_policy_integrity_fixture():
    identity = make_graph_identity()
    evidence = {
        "split_policy_name": "fa2_263_heuristic_exact_width",
        "flash_attn_version": "2.6.3",
        "page_table_width": 2,
        "effective_num_splits": 2,
        "heuristic_batch_size": 8,
        "heuristic_num_query_heads": 16,
        "heuristic_num_kv_heads": 8,
        "heuristic_head_dim": 128,
        "heuristic_page_block_size": 256,
        "heuristic_max_seqlen_q": 1,
        "heuristic_multi_processor_count": 108,
        "graph_batch_size": 8,
        "graph_identity_sha256": identity.sha256,
    }
    case_id = (
        "b8__ragged-context__exact_graph_heuristic"
        "__fa2-263-exact-width__r0"
    )
    row = {"case_id": case_id, "step_id": 0, **evidence}
    return {
        "raw_rows": [dict(row)],
        "layer_rows": [dict(row)],
        "kv_rows": [dict(row)],
        "process_rows": {
            case_id: {
                "case_id": case_id,
                "mode": "exact_graph_heuristic",
                "graph_identities": [
                    {
                        "sha256": identity.sha256,
                        "page_table_width": 2,
                        "effective_num_splits": 2,
                        "graph_batch_size": 8,
                    }
                ],
            }
        },
    }


def test_verifier_recomputes_exact_policy_integrity():
    verifier = load_verifier()
    fixture = make_complete_policy_integrity_fixture()
    result = verifier.verify_policy_integrity(**fixture)
    assert result == {"classification": "POLICY_EXACT", "failures": []}


def test_verifier_rejects_policy_integrity_mutations():
    verifier = load_verifier()
    mutations = (
        ("missing width", "raw_rows", "page_table_width", None),
        ("wrong split", "raw_rows", "effective_num_splits", 1),
        (
            "wrong sm count",
            "raw_rows",
            "heuristic_multi_processor_count",
            107,
        ),
        (
            "wrong identity hash",
            "raw_rows",
            "graph_identity_sha256",
            "f" * 64,
        ),
        (
            "row disagreement",
            "layer_rows",
            "page_table_width",
            3,
        ),
        (
            "wrong version",
            "raw_rows",
            "flash_attn_version",
            "2.6.4",
        ),
        (
            "graph auto split",
            "raw_rows",
            "effective_num_splits",
            0,
        ),
    )
    for name, collection, field, value in mutations:
        fixture = make_complete_policy_integrity_fixture()
        if value is None:
            fixture[collection][0].pop(field)
        else:
            fixture[collection][0][field] = value
        result = verifier.verify_policy_integrity(**fixture)
        assert result["classification"] in {"POLICY_DRIFT", "INCOMPLETE"}, (
            name,
            result,
        )

    split_mismatch = make_complete_policy_integrity_fixture()
    eager = copy.deepcopy(split_mismatch["raw_rows"][0])
    eager["case_id"] = (
        "b8__ragged-context__candidate_eager_heuristic"
        "__fa2-263-exact-width__r0"
    )
    eager["effective_num_splits"] = 3
    split_mismatch["raw_rows"].append(eager)
    split_mismatch["layer_rows"].append(copy.deepcopy(eager))
    split_mismatch["kv_rows"].append(copy.deepcopy(eager))
    split_mismatch["process_rows"][eager["case_id"]] = {
        "case_id": eager["case_id"],
        "mode": "candidate_eager_heuristic",
        "graph_identities": [],
    }
    result = verifier.verify_policy_integrity(**split_mismatch)
    assert result["classification"] == "POLICY_DRIFT"

    omitted = make_complete_policy_integrity_fixture()
    next(iter(omitted["process_rows"].values()))["graph_identities"] = []
    result = verifier.verify_policy_integrity(**omitted)
    assert result["classification"] == "POLICY_DRIFT"


def test_diagnostic_case_parser_rejects_split_policy_drift():
    diagnostic = load_diagnostic_module_without_gpu()
    case = contract.build_diagnostic_matrix()[0]
    case_spec = {"case_id": case.case_id, **asdict(case)}
    case_spec["flash_attn_num_splits"] = 0

    try:
        diagnostic._parse_case(case_spec)
    except ValueError as exc:
        assert "drift" in str(exc) or "outside frozen" in str(exc)
    else:
        raise AssertionError("split policy drift was accepted")


def test_execution_policy_rejects_case_level_heuristic_split():
    diagnostic = load_diagnostic_module_without_gpu()

    for case in contract.build_diagnostic_matrix():
        try:
            diagnostic.execution_split_count(case)
        except ValueError as exc:
            assert "per step" in str(exc)
        else:
            raise AssertionError("heuristic case exposed a frozen split")
    for case in contract.build_legacy_compatibility_matrix():
        if case.policy == "legacy_eager_auto":
            assert diagnostic.execution_split_count(case) == 0
        else:
            try:
                diagnostic.execution_split_count(case)
            except ValueError as exc:
                assert "per step" in str(exc)
            else:
                raise AssertionError("heuristic case exposed a frozen split")


def test_candidate_eager_forward_observes_step_split_and_restores_auto():
    diagnostic = load_diagnostic_module_without_gpu()
    context = install_fake_context_module()
    context.reset_context()
    seen = []
    try:
        result = diagnostic._run_with_split_policy(
            2,
            lambda: seen.append(
                context.get_context().flash_attn_num_splits
            ),
        )
        assert result is None
        assert seen == [2]
        assert context.get_context().flash_attn_num_splits == 0
    finally:
        context.reset_context()


def test_legacy_eager_forward_observes_auto():
    diagnostic = load_diagnostic_module_without_gpu()
    context = install_fake_context_module()
    context.reset_context()
    seen = []
    try:
        diagnostic._run_with_split_policy(
            0,
            lambda: seen.append(
                context.get_context().flash_attn_num_splits
            ),
        )
        assert seen == [0]
        assert context.get_context().flash_attn_num_splits == 0
    finally:
        context.reset_context()


def test_policy_evidence_distinguishes_same_policy_and_legacy_comparison():
    diagnostic = load_diagnostic_module_without_gpu()
    diagnostic_case = contract.build_diagnostic_matrix()[0]
    compatibility_cases = contract.build_legacy_compatibility_matrix()
    legacy_case = next(
        case
        for case in compatibility_cases
        if case.policy == "legacy_eager_auto"
    )

    assert diagnostic.policy_evidence(
        diagnostic_case,
        "2.6.3",
    ) == {
        "flash_attn_version": "2.6.3",
        "split_policy_name": contract.HEURISTIC_POLICY_NAME,
        "flash_attn_num_splits": None,
        "comparison_policy_name": "same_policy_heuristic_exact_width",
    }
    assert diagnostic.policy_evidence(legacy_case, "2.6.3") == {
        "flash_attn_version": "2.6.3",
        "split_policy_name": "auto",
        "flash_attn_num_splits": 0,
        "comparison_policy_name": "legacy_auto_vs_heuristic",
    }


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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contract.canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(
            contract.canonical_json_bytes(row) + b"\n" for row in rows
        )
    )


def _artifact_record(run_dir: Path, path: Path) -> dict:
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "sha256": contract.sha256_file(path),
    }


def _case_identity(case) -> tuple[int, str, int]:
    return case.batch_size, case.trajectory, case.repetition


def _comparison_policy_name(case) -> str:
    return (
        "same_policy_heuristic_exact_width"
        if hasattr(case, "mode")
        else "legacy_auto_vs_heuristic"
    )


def _case_graph_size(case) -> int:
    return getattr(case, "graph_size", case.batch_size)


def _is_reference_case(case) -> bool:
    return (
        getattr(case, "mode", None) == "candidate_eager_heuristic"
        or getattr(case, "policy", None) == "legacy_eager_auto"
    )


def _reference_name(case) -> str:
    return (
        f"b{case.batch_size}__{case.trajectory}__"
        f"r{case.repetition}.json"
    )


def _refresh_sha256sums(run_dir: Path) -> None:
    hashed_paths = [
        path
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.name != "sha256sums.txt"
        and "independent-verification" not in path.parts
    ]
    (run_dir / "sha256sums.txt").write_text(
        "".join(
            f"{contract.sha256_file(path)}  "
            f"{path.relative_to(run_dir).as_posix()}\n"
            for path in sorted(hashed_paths)
        ),
        encoding="utf-8",
    )


def write_complete_diagnostic_fixture(root: Path) -> Path:
    import torch

    run_dir = root / "canonical-diagnostic"
    run_dir.mkdir()
    source_tree_sha256 = "a" * 64
    environment = {
        "schema_version": 1,
        "host": "synthetic-gpu-host",
        "python": "3.11.synthetic",
        "pytorch": "2.synthetic",
        "cuda_runtime": "12.synthetic",
        "nvidia_driver": "synthetic",
        "gpu_name": "Synthetic GPU",
        "flash_attention": "2.6.3",
        "transformers": "synthetic",
        "model_identifier": "Qwen3-0.6B-synthetic",
        "bf16_supported": True,
        "source_tree_sha256": source_tree_sha256,
    }
    environment_sha256 = contract.canonical_json_sha256(environment)
    source_evidence = {
        "schema_version": 1,
        "base_commit": "b" * 40,
        "dirty": False,
        "tree_sha256": source_tree_sha256,
        "files": [],
    }
    prompt_manifest = {
        "schema_version": 1,
        "trajectories": {
            trajectory: {
                str(batch_size): contract.canonical_json_sha256(
                    {
                        "trajectory": trajectory,
                        "batch_size": batch_size,
                    }
                )
                for batch_size in contract.DIAGNOSTIC_BATCH_SIZES
            }
            for trajectory in contract.DIAGNOSTIC_TRAJECTORIES
        },
    }
    prompt_manifest_sha256 = contract.canonical_json_sha256(prompt_manifest)
    manifest = {
        "schema_version": 1,
        "kind": "heuristic_exact_width_recovery",
        "canonical": True,
        "source_tree_sha256": source_tree_sha256,
        "environment_sha256": environment_sha256,
        "prompt_manifest_sha256": prompt_manifest_sha256,
        "case_ids": [
            case.case_id
            for case in (
                contract.build_diagnostic_matrix()
                + contract.build_legacy_compatibility_matrix()
            )
        ],
        "same_policy_case_ids": [
            case.case_id for case in contract.build_diagnostic_matrix()
        ],
        "compatibility_case_ids": [
            case.case_id
            for case in contract.build_legacy_compatibility_matrix()
        ],
        "legacy_compatibility_case_ids": [
            case.case_id
            for case in contract.build_legacy_compatibility_matrix()
        ],
        "same_policy_process_count": 189,
        "compatibility_process_count": 126,
        "compatibility_pair_count": 63,
        "flash_attn_version": environment["flash_attention"],
        "policy_name": contract.HEURISTIC_POLICY_NAME,
        "auto_split_count": contract.AUTO_FLASH_ATTN_NUM_SPLITS,
        "warmup_steps": contract.WARMUP_STEPS,
        "measured_steps": contract.MEASURED_STEPS,
        "logit_rtol": contract.LOGIT_RTOL,
        "logit_atol": contract.LOGIT_ATOL,
    }
    _write_json(run_dir / "source_evidence.json", source_evidence)
    _write_json(run_dir / "environment.json", environment)
    _write_json(run_dir / "prompt_manifest.json", prompt_manifest)
    _write_json(run_dir / "manifest.json", manifest)

    reference_tensors = {}
    reference_hashes = {}
    process_rows = []
    raw_rows = []
    layer_rows = []
    kv_rows = []
    used_ports = set()
    all_cases = (
        contract.build_diagnostic_matrix()
        + contract.build_legacy_compatibility_matrix()
    )
    for case_index, case in enumerate(all_cases):
        identity = _case_identity(case)
        comparison_policy_name = _comparison_policy_name(case)
        policy_evidence = {
            "flash_attn_version": "2.6.3",
            "split_policy_name": case.split_policy_name,
            "flash_attn_num_splits": case.flash_attn_num_splits,
            "comparison_policy_name": comparison_policy_name,
        }
        graph_identities = []
        step_policy_evidence = {}
        if case.flash_attn_num_splits is None:
            page_table_width = (
                1
                if case.batch_size <= 5
                else 2
                if case.batch_size <= 9
                else 3
            )
            split_inputs = load_split_policy().FlashAttentionSplitInputs(
                batch_size=case.batch_size,
                num_query_heads=16,
                num_kv_heads=8,
                head_dim=128,
                page_block_size=256,
                page_table_width=page_table_width,
                max_seqlen_q=1,
                multi_processor_count=108,
            )
            graph_identity = (
                load_split_policy().build_flash_attn_263_graph_identity(
                    graph_batch_size=_case_graph_size(case),
                    inputs=split_inputs,
                    flash_attn_version="2.6.3",
                )
            )
            step_policy_evidence = {
                "split_policy_name": contract.HEURISTIC_POLICY_NAME,
                "flash_attn_version": "2.6.3",
                "page_table_width": page_table_width,
                "effective_num_splits": (
                    graph_identity.effective_num_splits
                ),
                "heuristic_batch_size": case.batch_size,
                "heuristic_num_query_heads": 16,
                "heuristic_num_kv_heads": 8,
                "heuristic_head_dim": 128,
                "heuristic_page_block_size": 256,
                "heuristic_max_seqlen_q": 1,
                "heuristic_multi_processor_count": 108,
                "graph_batch_size": _case_graph_size(case),
                "graph_identity_sha256": graph_identity.sha256,
            }
            graph_identities = [
                {
                    "sha256": graph_identity.sha256,
                    "page_table_width": page_table_width,
                    "effective_num_splits": (
                        graph_identity.effective_num_splits
                    ),
                    "graph_batch_size": _case_graph_size(case),
                }
            ]
        reference_key = (
            comparison_policy_name,
            identity,
        )
        base = torch.arange(
            contract.MEASURED_STEPS * case.batch_size * 3,
            dtype=torch.float32,
        ).reshape(contract.MEASURED_STEPS, case.batch_size, 3)
        logits = base / 100.0
        layers = torch.stack(
            (logits[..., :2], logits[..., 1:3]),
            dim=1,
        ).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        active_slots = list(range(10, 10 + case.batch_size))
        observed_slots = active_slots + [0, 1000, 2000, 3000]
        slot_count = len(observed_slots)
        keys_before = torch.zeros(
            contract.MEASURED_STEPS,
            2,
            slot_count,
            1,
            dtype=torch.float32,
        )
        keys_after = keys_before.clone()
        keys_after[:, :, :case.batch_size] = 1.0
        values_before = keys_before.clone()
        values_after = keys_after.clone()
        reference_tokens = [
            [step + row for row in range(case.batch_size)]
            for step in range(
                contract.WARMUP_STEPS + contract.MEASURED_STEPS
            )
        ]
        reference_sha256 = contract.canonical_json_sha256(reference_tokens)
        reference_path = (
            run_dir / "reference_tokens" / _reference_name(case)
        )
        if _is_reference_case(case):
            _write_json(reference_path, reference_tokens)
        prompt_sha256 = prompt_manifest["trajectories"][case.trajectory][
            str(case.batch_size)
        ]
        if _is_reference_case(case):
            reference_tensors[reference_key] = {
                "logits": logits.clone(),
                "layers": layers.clone(),
                "keys_before": keys_before.clone(),
                "keys_after": keys_after.clone(),
                "values_before": values_before.clone(),
                "values_after": values_after.clone(),
                "observed_slots": list(observed_slots),
            }
            reference_hashes[reference_key] = reference_sha256
        else:
            reference = reference_tensors[reference_key]
            logits = reference["logits"].clone()
            layers = reference["layers"].clone()
            keys_before = reference["keys_before"].clone()
            keys_after = reference["keys_after"].clone()
            values_before = reference["values_before"].clone()
            values_after = reference["values_after"].clone()
            observed_slots = list(reference["observed_slots"])
            reference_sha256 = reference_hashes[reference_key]

        logits_path = run_dir / "tensors" / "logits" / f"{case.case_id}.pt"
        layers_path = run_dir / "tensors" / "layers" / f"{case.case_id}.pt"
        kv_path = run_dir / "tensors" / "kv" / f"{case.case_id}.pt"
        logits_path.parent.mkdir(parents=True, exist_ok=True)
        layers_path.parent.mkdir(parents=True, exist_ok=True)
        kv_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "case_id": case.case_id,
                **policy_evidence,
                "dtype": str(logits.dtype),
                "shape": list(logits.shape),
                "step_ids": list(range(contract.MEASURED_STEPS)),
                "row_ids": list(range(case.batch_size)),
                "tensor": logits,
            },
            logits_path,
        )
        torch.save(
            {
                "schema_version": 1,
                "case_id": case.case_id,
                **policy_evidence,
                "dtype": str(layers.dtype),
                "shape": list(layers.shape),
                "step_ids": list(range(contract.MEASURED_STEPS)),
                "row_ids": list(range(case.batch_size)),
                "layer_ids": [0, 1],
                "component_ids": ["hidden_states", "residual"],
                "tensor": layers,
            },
            layers_path,
        )
        torch.save(
            {
                "schema_version": 1,
                "case_id": case.case_id,
                **policy_evidence,
                "step_ids": list(range(contract.MEASURED_STEPS)),
                "row_ids": list(range(case.batch_size)),
                "slot_ids": [
                    list(observed_slots)
                    for _ in range(contract.MEASURED_STEPS)
                ],
                "plans": [
                    {
                        "active_write_slots": list(active_slots),
                        "slot_zero": 0,
                        "inactive_declared_slots": (
                            [0]
                            if _case_graph_size(case) > case.batch_size
                            else []
                        ),
                        "sentinel_slots": [1000, 2000, 3000],
                    }
                    for _ in range(contract.MEASURED_STEPS)
                ],
                "keys_before": keys_before,
                "values_before": values_before,
                "keys_after": keys_after,
                "values_after": values_after,
            },
            kv_path,
        )
        tiny_port = 20000 + case_index * 2
        master_port = tiny_port + 1
        assert tiny_port not in used_ports
        assert master_port not in used_ports
        used_ports.update((tiny_port, master_port))
        process_rows.append(
            {
                **asdict(case),
                **(
                    {"pair_id": case.pair_id}
                    if hasattr(case, "pair_id")
                    else {}
                ),
                **policy_evidence,
                "case_id": case.case_id,
                "status": "PASS",
                "source_tree_sha256": source_tree_sha256,
                "environment_sha256": environment_sha256,
                "prompt_sha256": prompt_sha256,
                "reference_token_sha256": reference_sha256,
                "reference_tokens": _artifact_record(
                    run_dir,
                    reference_path,
                ),
                "tinyvllm_dist_port": tiny_port,
                "master_port": master_port,
                "graph_identities": graph_identities,
                "artifacts": {
                    "logits": _artifact_record(run_dir, logits_path),
                    "layers": _artifact_record(run_dir, layers_path),
                    "kv": _artifact_record(run_dir, kv_path),
                },
            }
        )
        for step_id in range(contract.MEASURED_STEPS):
            raw_rows.append(
                {
                    **asdict(case),
                    **policy_evidence,
                    **step_policy_evidence,
                    "case_id": case.case_id,
                    "step_id": step_id,
                    "observed_argmax_token_ids": torch.argmax(
                        logits[step_id],
                        dim=-1,
                    ).tolist(),
                    "reference_next_input_token_ids": reference_tokens[
                        contract.WARMUP_STEPS + step_id
                    ],
                }
            )
            layer_rows.append(
                {
                    **policy_evidence,
                    **step_policy_evidence,
                    "case_id": case.case_id,
                    "step_id": step_id,
                    "required_layer_count": 2,
                    "observed_layer_count": 2,
                    "layer_ids": [0, 1],
                    "finite": True,
                }
            )
            kv_rows.append(
                {
                    **policy_evidence,
                    **step_policy_evidence,
                    "case_id": case.case_id,
                    "step_id": step_id,
                    "active_write_slots": list(active_slots),
                    "slot_zero": 0,
                    "inactive_declared_slots": (
                        [0]
                        if _case_graph_size(case) > case.batch_size
                        else []
                    ),
                    "sentinel_slots": [1000, 2000, 3000],
                    "observed_slot_ids": list(observed_slots),
                }
            )

    _write_jsonl(run_dir / "process_rows.jsonl", process_rows)
    _write_jsonl(run_dir / "raw_rows.jsonl", raw_rows)
    _write_jsonl(run_dir / "layer_observations.jsonl", layer_rows)
    _write_jsonl(run_dir / "kv_observations.jsonl", kv_rows)
    producer_summary = {
        "schema_version": 1,
        "classification": "EXACT_REPLAY_CORRECT",
        "rounded_classification": "ROUNDED_REPLAY_CORRECT",
        "legacy_compatibility": "LEGACY_COMPATIBLE",
        "policy_integrity": "POLICY_EXACT",
        "case_count": len(process_rows),
        "same_policy_case_count": 189,
        "compatibility_process_count": 126,
        "compatibility_pair_count": 63,
    }
    _write_json(run_dir / "summary.json", producer_summary)
    _refresh_sha256sums(run_dir)
    return run_dir


def _rewrite_jsonl(path: Path, rows: list[dict]) -> None:
    _write_jsonl(path, rows)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_verifier_reconstructs_complete_diagnostic():
    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "EXACT_REPLAY_CORRECT"
        assert (
            summary["rounded_classification"]
            == "ROUNDED_REPLAY_CORRECT"
        )
        assert summary["legacy_compatibility"] == "LEGACY_COMPATIBLE"
        assert summary["same_policy_case_count"] == 189
        assert summary["compatibility_process_count"] == 126
        assert summary["compatibility_pair_count"] == 63


def test_verifier_rejects_heuristic_manifest_contract_drift():
    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        manifest = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
        source_evidence = json.loads(
            (run_dir / "source_evidence.json").read_text(encoding="utf-8")
        )
        environment = json.loads(
            (run_dir / "environment.json").read_text(encoding="utf-8")
        )
        prompt_manifest = json.loads(
            (run_dir / "prompt_manifest.json").read_text(encoding="utf-8")
        )
        mutations = (
            ("legacy_compatibility_case_ids", []),
            ("same_policy_process_count", 188),
            ("compatibility_process_count", 125),
            ("compatibility_pair_count", 62),
            ("flash_attn_version", "different"),
            ("policy_name", "different_policy"),
            ("auto_split_count", 1),
        )
        for field, value in mutations:
            mutated = copy.deepcopy(manifest)
            mutated[field] = value
            failures = verifier._validate_manifest(
                mutated,
                source_evidence,
                environment,
                prompt_manifest,
            )
            assert any(f"manifest {field}=" in failure for failure in failures)


def test_verifier_rejects_missing_split_identity():
    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        process_path = run_dir / "process_rows.jsonl"
        rows = _read_jsonl(process_path)
        rows[0].pop("flash_attn_num_splits")
        _rewrite_jsonl(process_path, rows)
        _refresh_sha256sums(run_dir)
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "INCOMPLETE"


def test_verifier_rejects_auto_graph_as_fixed16_evidence():
    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        process_path = run_dir / "process_rows.jsonl"
        rows = _read_jsonl(process_path)
        graph = next(
            row
            for row in rows
            if row.get("mode") == "exact_graph_heuristic"
        )
        graph["split_policy_name"] = "auto"
        graph["flash_attn_num_splits"] = 0
        _rewrite_jsonl(process_path, rows)
        _refresh_sha256sums(run_dir)
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "INCOMPLETE"


def test_verifier_rejects_step_policy_identity_drift():
    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        raw_path = run_dir / "raw_rows.jsonl"
        rows = _read_jsonl(raw_path)
        rows[0]["comparison_policy_name"] = "legacy_auto_vs_heuristic"
        _rewrite_jsonl(raw_path, rows)
        _refresh_sha256sums(run_dir)
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "INCOMPLETE"


def test_verifier_reports_fixed_vs_auto_token_mismatch_as_legacy_incompatible():
    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        raw_path = run_dir / "raw_rows.jsonl"
        rows = _read_jsonl(raw_path)
        target = next(
            row
            for row in rows
            if row.get("policy") == "candidate_eager_heuristic"
        )
        target["observed_argmax_token_ids"][0] += 1
        _rewrite_jsonl(raw_path, rows)
        _refresh_sha256sums(run_dir)
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "EXACT_REPLAY_CORRECT"
        assert summary["legacy_compatibility"] == "LEGACY_INCOMPATIBLE"


def test_verifier_is_independent_from_diagnostic_producer():
    source = VERIFIER_PATH.read_text(encoding="utf-8")
    assert "diagnose_multi_sequence_cuda_graph" not in source


def test_verifier_rejects_missing_matrix_case():
    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        process_path = run_dir / "process_rows.jsonl"
        rows = _read_jsonl(process_path)
        _rewrite_jsonl(process_path, rows[:-1])
        _refresh_sha256sums(run_dir)
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "INCOMPLETE"
        assert any("missing" in failure for failure in summary["failures"])


def test_verifier_detects_rehashed_exact_logit_mutation():
    import torch

    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        process_path = run_dir / "process_rows.jsonl"
        rows = _read_jsonl(process_path)
        target = next(row for row in rows if row["mode"] == "exact_graph_heuristic")
        artifact = run_dir / target["artifacts"]["logits"]["path"]
        shard = torch.load(artifact, weights_only=False)
        shard["tensor"][0, 0, 0] += 10.0
        torch.save(shard, artifact)
        target["artifacts"]["logits"]["sha256"] = contract.sha256_file(
            artifact
        )
        _rewrite_jsonl(process_path, rows)
        _refresh_sha256sums(run_dir)
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "EXACT_REPLAY_CORRUPT"
        assert target["case_id"] in summary["corrupt_exact_case_ids"]


def test_verifier_rejects_source_and_environment_drift():
    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        process_path = run_dir / "process_rows.jsonl"
        rows = _read_jsonl(process_path)
        rows[0]["source_tree_sha256"] = "f" * 64
        rows[1]["environment_sha256"] = "e" * 64
        _rewrite_jsonl(process_path, rows)
        _refresh_sha256sums(run_dir)
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "INCOMPLETE"
        assert any(
            "source_tree_sha256" in failure
            for failure in summary["failures"]
        )
        assert any(
            "environment_sha256" in failure
            for failure in summary["failures"]
        )


def test_verifier_rejects_duplicate_matrix_key_and_port():
    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        process_path = run_dir / "process_rows.jsonl"
        rows = _read_jsonl(process_path)
        duplicate = dict(rows[0])
        duplicate["tinyvllm_dist_port"] = rows[1]["tinyvllm_dist_port"]
        rows.append(duplicate)
        _rewrite_jsonl(process_path, rows)
        _refresh_sha256sums(run_dir)
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "INCOMPLETE"
        assert any("duplicate case_id" in item for item in summary["failures"])
        assert any("reused port" in item for item in summary["failures"])


def test_verifier_rejects_graph_size_mismatch():
    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        process_path = run_dir / "process_rows.jsonl"
        rows = _read_jsonl(process_path)
        rows[0]["graph_size"] += 1
        _rewrite_jsonl(process_path, rows)
        _refresh_sha256sums(run_dir)
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "INCOMPLETE"
        assert any("graph_size" in item for item in summary["failures"])


def test_verifier_rejects_prompt_and_reference_hash_drift():
    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        process_path = run_dir / "process_rows.jsonl"
        rows = _read_jsonl(process_path)
        rows[0]["prompt_sha256"] = "1" * 64
        rows[1]["reference_token_sha256"] = "2" * 64
        _rewrite_jsonl(process_path, rows)
        _refresh_sha256sums(run_dir)
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "INCOMPLETE"
        assert any("prompt_sha256" in item for item in summary["failures"])
        assert any(
            "reference_token_sha256" in item
            for item in summary["failures"]
        )


def test_verifier_rejects_truncated_raw_jsonl():
    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        raw_path = run_dir / "raw_rows.jsonl"
        rows = _read_jsonl(raw_path)
        _rewrite_jsonl(raw_path, rows[:-1])
        _refresh_sha256sums(run_dir)
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "INCOMPLETE"
        assert any("raw_rows" in item for item in summary["failures"])


def test_verifier_detects_nonfinite_exact_logit():
    import torch

    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        process_path = run_dir / "process_rows.jsonl"
        rows = _read_jsonl(process_path)
        target = next(row for row in rows if row["mode"] == "exact_graph_heuristic")
        artifact = run_dir / target["artifacts"]["logits"]["path"]
        shard = torch.load(artifact, weights_only=False)
        shard["tensor"][0, 0, 0] = float("nan")
        torch.save(shard, artifact)
        target["artifacts"]["logits"]["sha256"] = contract.sha256_file(
            artifact
        )
        _rewrite_jsonl(process_path, rows)
        _refresh_sha256sums(run_dir)
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "EXACT_REPLAY_CORRUPT"
        detail = summary["first_divergence"]
        assert detail["evidence"] == "logits"
        assert detail["case_id"] == target["case_id"]
        assert detail["step_id"] == 0
        assert detail["row_id"] == 0


def test_verifier_detects_exact_argmax_mismatch():
    import torch

    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        process_path = run_dir / "process_rows.jsonl"
        rows = _read_jsonl(process_path)
        target = next(row for row in rows if row["mode"] == "exact_graph_heuristic")
        artifact = run_dir / target["artifacts"]["logits"]["path"]
        shard = torch.load(artifact, weights_only=False)
        shard["tensor"][0, 0] = torch.tensor([100.0, 0.0, 0.0])
        torch.save(shard, artifact)
        target["artifacts"]["logits"]["sha256"] = contract.sha256_file(
            artifact
        )
        _rewrite_jsonl(process_path, rows)
        _refresh_sha256sums(run_dir)
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "EXACT_REPLAY_CORRUPT"
        assert summary["first_divergence"]["kind"] == "argmax_mismatch"


def test_verifier_detects_exact_close_threshold_failure():
    import torch

    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        process_path = run_dir / "process_rows.jsonl"
        rows = _read_jsonl(process_path)
        target = next(row for row in rows if row["mode"] == "exact_graph_heuristic")
        artifact = run_dir / target["artifacts"]["logits"]["path"]
        shard = torch.load(artifact, weights_only=False)
        shard["tensor"][0, 0] += torch.tensor([0.1, 0.1, 0.1])
        torch.save(shard, artifact)
        target["artifacts"]["logits"]["sha256"] = contract.sha256_file(
            artifact
        )
        _rewrite_jsonl(process_path, rows)
        _refresh_sha256sums(run_dir)
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "EXACT_REPLAY_CORRUPT"
        assert summary["first_divergence"]["kind"] == "close_failure"


def test_verifier_rejects_missing_layer_index():
    import torch

    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        process_path = run_dir / "process_rows.jsonl"
        rows = _read_jsonl(process_path)
        target = next(row for row in rows if row["mode"] == "exact_graph_heuristic")
        artifact = run_dir / target["artifacts"]["layers"]["path"]
        shard = torch.load(artifact, weights_only=False)
        shard["layer_ids"] = [0]
        shard["tensor"] = shard["tensor"][:, :, :1]
        shard["shape"] = list(shard["tensor"].shape)
        torch.save(shard, artifact)
        target["artifacts"]["layers"]["sha256"] = contract.sha256_file(
            artifact
        )
        _rewrite_jsonl(process_path, rows)
        _refresh_sha256sums(run_dir)
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "INCOMPLETE"
        assert any("layer_ids" in item for item in summary["failures"])


def test_verifier_detects_rehashed_layer_tensor_mutation():
    import torch

    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        process_path = run_dir / "process_rows.jsonl"
        rows = _read_jsonl(process_path)
        target = next(row for row in rows if row["mode"] == "exact_graph_heuristic")
        artifact = run_dir / target["artifacts"]["layers"]["path"]
        shard = torch.load(artifact, weights_only=False)
        shard["tensor"][0, 0, 0, 0, 0] += 1.0
        torch.save(shard, artifact)
        target["artifacts"]["layers"]["sha256"] = contract.sha256_file(
            artifact
        )
        _rewrite_jsonl(process_path, rows)
        _refresh_sha256sums(run_dir)
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "EXACT_REPLAY_CORRUPT"
        detail = summary["first_divergence"]
        assert detail["evidence"] == "layers"
        assert detail["layer_id"] == 0


def _mutate_kv_evidence(run_dir: Path, mutation: str) -> str:
    import torch

    process_path = run_dir / "process_rows.jsonl"
    rows = _read_jsonl(process_path)
    target = next(row for row in rows if row["mode"] == "exact_graph_heuristic")
    artifact = run_dir / target["artifacts"]["kv"]["path"]
    shard = torch.load(artifact, weights_only=False)
    if mutation == "active":
        shard["keys_after"][0, 0, 0, 0] += 1.0
    elif mutation == "slot_zero":
        slot_index = shard["slot_ids"][0].index(0)
        shard["keys_after"][0, 0, slot_index, 0] += 1.0
    elif mutation == "sentinel":
        sentinel = shard["plans"][0]["sentinel_slots"][0]
        slot_index = shard["slot_ids"][0].index(sentinel)
        shard["keys_after"][0, 0, slot_index, 0] += 1.0
    else:
        raise AssertionError(mutation)
    torch.save(shard, artifact)
    target["artifacts"]["kv"]["sha256"] = contract.sha256_file(artifact)
    _rewrite_jsonl(process_path, rows)
    _refresh_sha256sums(run_dir)
    return target["case_id"]


def test_verifier_detects_active_kv_mismatch():
    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        target_case_id = _mutate_kv_evidence(run_dir, "active")
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "EXACT_REPLAY_CORRUPT"
        assert summary["first_divergence"]["evidence"] == "kv"
        assert summary["first_divergence"]["case_id"] == target_case_id
        assert summary["first_divergence"]["kind"] == "active_kv_mismatch"


def test_verifier_detects_unexpected_slot_zero_mutation():
    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        _mutate_kv_evidence(run_dir, "slot_zero")
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "EXACT_REPLAY_CORRUPT"
        assert summary["first_divergence"]["slot_id"] == 0


def test_verifier_detects_unexpected_sentinel_mutation():
    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        _mutate_kv_evidence(run_dir, "sentinel")
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "EXACT_REPLAY_CORRUPT"
        assert summary["first_divergence"]["kind"] == "sentinel_mutation"


def test_verifier_rejects_producer_classification_tamper():
    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        summary_path = run_dir / "summary.json"
        producer = json.loads(summary_path.read_text(encoding="utf-8"))
        producer["classification"] = "EXACT_REPLAY_CORRUPT"
        _write_json(summary_path, producer)
        _refresh_sha256sums(run_dir)
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "INCOMPLETE"
        assert any(
            "producer classification" in item
            for item in summary["failures"]
        )


def test_verifier_rejects_missing_artifact_hash():
    verifier = load_verifier()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = write_complete_diagnostic_fixture(
            Path(temporary_directory)
        )
        process_path = run_dir / "process_rows.jsonl"
        rows = _read_jsonl(process_path)
        del rows[0]["artifacts"]["logits"]["sha256"]
        _rewrite_jsonl(process_path, rows)
        _refresh_sha256sums(run_dir)
        summary = verifier.verify_diagnostic(run_dir)
        assert summary["classification"] == "INCOMPLETE"
        assert any("missing artifact hash" in item for item in summary["failures"])


def test_remote_runner_has_frozen_transport_and_safety_contract():
    source = REMOTE_RUNNER_PATH.read_text(encoding="utf-8")
    for required in (
        "sitian@10.232.195.203",
        "/tmp/ssh-sitian-10.232.195.203",
        "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python",
        "/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B",
        "TINYVLLM_DIST_PORT",
        "MASTER_PORT",
        "EADDRINUSE",
        "source_audit.build_source_evidence",
        "source_audit.validate_source_snapshot",
        "heuristic-exact-width-preflight",
        "heuristic-exact-width-smoke",
        "heuristic-exact-width-canonical",
        "heuristic_exact_width_recovery",
        "flash_attn_split_policy.py",
        "download-only",
        "verify-only",
    ):
        assert required in source, required
    for forbidden in (
        "rsync",
        "pkill",
        "killall",
        "rm -rf /tmp",
        "git checkout",
        "git reset",
        "git clean",
        "git add -A",
    ):
        assert forbidden not in source, forbidden


def test_remote_runner_allocates_globally_unique_port_pairs():
    runner = load_remote_runner()
    allocated = iter(
        [
            (21000, 21001),
            (21002, 21003),
            (21004, 21005),
        ]
    )
    pairs = runner.allocate_unique_port_pairs(
        count=3,
        allocator=lambda: next(allocated),
    )
    assert pairs == [
        (21000, 21001),
        (21002, 21003),
        (21004, 21005),
    ]
    assert len({port for pair in pairs for port in pair}) == 6


def test_remote_runner_rejects_duplicate_or_equal_ports():
    runner = load_remote_runner()
    duplicate = iter([(22000, 22001), (22001, 22002)])
    try:
        runner.allocate_unique_port_pairs(
            count=2,
            allocator=lambda: next(duplicate),
        )
    except ValueError as exc:
        assert "duplicate port" in str(exc)
    else:
        raise AssertionError("duplicate port pair accepted")

    try:
        runner.allocate_unique_port_pairs(
            count=1,
            allocator=lambda: (23000, 23000),
        )
    except ValueError as exc:
        assert "distinct" in str(exc)
    else:
        raise AssertionError("equal port pair accepted")


def test_remote_runner_retries_only_eaddrinuse():
    runner = load_remote_runner()
    assert runner.is_retryable_port_collision(
        returncode=1,
        stderr="RuntimeError: server failed to listen: EADDRINUSE",
    )
    assert not runner.is_retryable_port_collision(
        returncode=0,
        stderr="EADDRINUSE",
    )
    assert not runner.is_retryable_port_collision(
        returncode=1,
        stderr="CUDA out of memory",
    )


def test_remote_runner_reads_diagnostic_stderr_for_port_collision():
    runner = load_remote_runner()
    with tempfile.TemporaryDirectory() as temporary_directory:
        case_dir = Path(temporary_directory)
        output_dir = case_dir / "output"
        output_dir.mkdir()
        (output_dir / "launcher_stderr.txt").write_text(
            "[transformers] torch_dtype is deprecated\n",
            encoding="utf-8",
        )
        (output_dir / "stderr.txt").write_text(
            "RuntimeError: address already in use: EADDRINUSE\n",
            encoding="utf-8",
        )
        stderr = runner.read_remote_case_stderr(
            case_dir=case_dir,
            fallback=b"",
        )
        assert "torch_dtype is deprecated" in stderr
        assert "EADDRINUSE" in stderr
        assert runner.is_retryable_port_collision(
            returncode=1,
            stderr=stderr,
        )


def test_remote_runner_preserves_remote_shell_command_as_one_argument():
    runner = load_remote_runner()
    remote_command = "cd /tmp/example && printf 'OK\\n'"
    command = runner._ssh_command(remote_command)
    assert command[-3:] == [
        "bash",
        "-lc",
        shlex.quote(remote_command),
    ]


def test_remote_runner_disables_bytecode_during_source_validation():
    source = REMOTE_RUNNER_PATH.read_text(encoding="utf-8")
    start = source.index("def _remote_python_script(")
    end = source.index("\ndef _remote_validate_source(", start)
    helper = source[start:end]
    assert "PYTHONDONTWRITEBYTECODE=1" in helper


def test_remote_runner_requires_explicit_resume_for_existing_run():
    runner = load_remote_runner()
    with tempfile.TemporaryDirectory() as temporary_directory:
        output_root = Path(temporary_directory)
        existing = output_root / "existing-run"
        existing.mkdir()
        try:
            runner.prepare_run_directory(
                output_root=output_root,
                run_tag="existing-run",
                resume=False,
            )
        except ValueError as exc:
            assert "already exists" in str(exc)
        else:
            raise AssertionError("existing run accepted without --resume")
        assert runner.prepare_run_directory(
            output_root=output_root,
            run_tag="existing-run",
            resume=True,
        ) == existing


def test_remote_runner_reserves_resumed_ports_globally():
    runner = load_remote_runner()
    used_ports = set()
    runner.reserve_unique_port_pair(
        used_ports=used_ports,
        pair=(24000, 24001),
        owner="resumed-case",
    )
    assert used_ports == {24000, 24001}
    try:
        runner.reserve_unique_port_pair(
            used_ports=used_ports,
            pair=(24001, 24002),
            owner="new-case",
        )
    except ValueError as exc:
        assert "duplicate port" in str(exc)
    else:
        raise AssertionError("resumed port was reused")


def test_remote_runner_reallocates_duplicate_ephemeral_ports():
    runner = load_remote_runner()
    allocated = iter(
        [
            (25000, 25001),
            (25001, 25002),
            (25003, 25004),
        ]
    )
    used_ports = {25000, 25001}
    pair = runner.allocate_fresh_unique_port_pair(
        used_ports=used_ports,
        allocator=lambda: next(allocated),
        max_attempts=3,
    )
    assert pair == (25003, 25004)
    assert used_ports == {25000, 25001, 25003, 25004}


def test_remote_runner_exposes_resume_and_verifier_python_options():
    runner = load_remote_runner()
    args = runner._parse_args(
        [
            "heuristic-exact-width-canonical",
            "--run-tag",
            "resume-run",
            "--resume",
            "--verifier-python",
            "/tmp/verifier-python",
        ]
    )
    assert args.resume is True
    assert args.verifier_python == Path("/tmp/verifier-python")


def test_remote_runner_promotes_source_evidence_artifacts():
    runner = load_remote_runner()
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = Path(temporary_directory)
        staging = run_dir / "staging"
        staging.mkdir()
        (staging / "source.patch").write_bytes(b"source patch")
        (staging / "source_snapshot.tar.gz").write_bytes(b"snapshot")
        runner.promote_source_evidence_artifacts(run_dir)
        assert (run_dir / "source.patch").read_bytes() == b"source patch"
        assert (
            run_dir / "source_snapshot.tar.gz"
        ).read_bytes() == b"snapshot"


def test_remote_runner_removes_only_downloaded_remote_case():
    runner = load_remote_runner()
    commands = []
    original = runner._run_remote
    try:
        runner._run_remote = lambda command, **kwargs: commands.append(
            (command, kwargs)
        )
        runner.remove_downloaded_remote_case(
            "/tmp/tllm-cuda-graph-run/cases/b2__uniform-short__eager__r0"
        )
    finally:
        runner._run_remote = original
    assert len(commands) == 1
    command, kwargs = commands[0]
    assert kwargs == {}
    assert command == (
        "test -n "
        "/tmp/tllm-cuda-graph-run/cases/b2__uniform-short__eager__r0 "
        "&& rm -r -- "
        "/tmp/tllm-cuda-graph-run/cases/b2__uniform-short__eager__r0"
    )
    assert command != "rm -r -- /tmp"


def test_remote_runner_builds_heuristic_smoke_for_both_gates():
    runner = load_remote_runner()
    same_policy, compatibility = runner.build_smoke_cases()
    assert same_policy
    assert compatibility
    assert len(same_policy) == 18
    assert len(compatibility) == 12
    assert all(case.repetition == 0 for case in same_policy)
    assert all(case.repetition == 0 for case in compatibility)
    assert {case.mode for case in same_policy} == {
        "candidate_eager_heuristic",
        "exact_graph_heuristic",
        "rounded_graph_heuristic",
    }
    assert {case.policy for case in compatibility} == {
        "legacy_eager_auto",
        "candidate_eager_heuristic",
    }


def test_remote_runner_orders_each_reference_before_candidates():
    runner = load_remote_runner()
    cases = runner.ordered_canonical_cases()
    assert len(cases) == 315
    allocated = runner.allocate_unique_port_pairs(
        count=len(cases),
        allocator=iter(
            (30000 + 2 * index, 30001 + 2 * index)
            for index in range(len(cases))
        ).__next__,
    )
    assert len({port for pair in allocated for port in pair}) == 630
    positions = {case.case_id: index for index, case in enumerate(cases)}
    for case in contract.build_diagnostic_matrix():
        if case.mode == "candidate_eager_heuristic":
            continue
        reference = runner.same_policy_reference_case(case)
        assert reference.mode == "candidate_eager_heuristic"
        assert positions[reference.case_id] < positions[case.case_id]
    for case in contract.build_legacy_compatibility_matrix():
        if case.policy == "legacy_eager_auto":
            continue
        reference = runner.compatibility_reference_case(case)
        assert reference.policy == "legacy_eager_auto"
        assert positions[reference.case_id] < positions[case.case_id]


def test_remote_runner_requires_paired_reference_before_candidate():
    runner = load_remote_runner()
    candidates = (
        next(
            case
            for case in contract.build_diagnostic_matrix()
            if case.mode == "exact_graph_heuristic"
        ),
        next(
            case
            for case in contract.build_legacy_compatibility_matrix()
            if case.policy == "candidate_eager_heuristic"
        ),
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        run_dir = Path(temporary_directory)
        for case in candidates:
            assert runner.candidate_case_ready(case, run_dir) is False
            reference_case = runner.reference_case(case)
            reference = (
                run_dir
                / "cases"
                / reference_case.case_id
                / "input"
                / "reference_tokens.json"
            )
            reference.parent.mkdir(parents=True)
            _write_json(
                reference,
                [
                    [step + row for row in range(case.batch_size)]
                    for step in range(
                        contract.WARMUP_STEPS + contract.MEASURED_STEPS
                    )
                ],
            )
            assert runner.candidate_case_ready(case, run_dir) is True


def test_remote_runner_resume_requires_policy_identity_and_artifact_hashes():
    runner = load_remote_runner()
    case = next(
        case
        for case in contract.build_legacy_compatibility_matrix()
        if case.policy == "candidate_eager_heuristic"
    )
    source_tree_sha256 = "a" * 64
    environment_sha256 = "b" * 64
    flash_attn_version = "2.6.3"
    with tempfile.TemporaryDirectory() as temporary_directory:
        case_dir = Path(temporary_directory) / case.case_id
        artifact = case_dir / "output" / "artifact.bin"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"completed case")
        reference = case_dir / "input" / "reference_tokens.json"
        _write_json(
            reference,
            [
                [step + row for row in range(case.batch_size)]
                for step in range(
                    contract.WARMUP_STEPS + contract.MEASURED_STEPS
                )
            ],
        )
        reference_token_sha256 = contract.canonical_json_sha256(
            json.loads(reference.read_text(encoding="utf-8"))
        )
        _write_json(
            case_dir / "case_result.json",
            {
                "schema_version": 1,
                "status": "PASS",
                "artifact_kind": "heuristic_exact_width_recovery",
                "case": asdict(case),
                "case_id": case.case_id,
                "flash_attn_version": flash_attn_version,
                "split_policy_name": case.split_policy_name,
                "flash_attn_num_splits": case.flash_attn_num_splits,
                "comparison_policy_name": "legacy_auto_vs_heuristic",
                "graph_identities": [],
                "source_tree_sha256": source_tree_sha256,
                "environment_sha256": environment_sha256,
                "reference_token_sha256": reference_token_sha256,
                "artifacts": {
                    "payload": _artifact_record(case_dir, artifact),
                },
            },
        )
        assert runner.completed_case_is_resumable(
            case_dir=case_dir,
            case=case,
            source_tree_sha256=source_tree_sha256,
            environment_sha256=environment_sha256,
            flash_attn_version=flash_attn_version,
        )
        baseline = json.loads(
            (case_dir / "case_result.json").read_text(encoding="utf-8")
        )
        mutations = (
            ("flash_attn_version", "2.6.4"),
            ("split_policy_name", "auto"),
            ("flash_attn_num_splits", 0),
            (
                "comparison_policy_name",
                "same_policy_heuristic_exact_width",
            ),
            ("artifact_kind", "fixed_split_recovery"),
            ("source_tree_sha256", "c" * 64),
            ("environment_sha256", "d" * 64),
        )
        for field, value in mutations:
            mutated = copy.deepcopy(baseline)
            mutated[field] = value
            _write_json(case_dir / "case_result.json", mutated)
            assert not runner.completed_case_is_resumable(
                case_dir=case_dir,
                case=case,
                source_tree_sha256=source_tree_sha256,
                environment_sha256=environment_sha256,
                flash_attn_version=flash_attn_version,
            ), field
        _write_json(case_dir / "case_result.json", baseline)
        artifact.write_bytes(b"tampered")
        assert not runner.completed_case_is_resumable(
            case_dir=case_dir,
            case=case,
            source_tree_sha256=source_tree_sha256,
            environment_sha256=environment_sha256,
            flash_attn_version=flash_attn_version,
        )


def test_remote_runner_exposes_heuristic_cli_modes():
    runner = load_remote_runner()
    for mode in (
        "heuristic-exact-width-preflight",
        "heuristic-exact-width-smoke",
        "heuristic-exact-width-canonical",
        "download-only",
        "verify-only",
    ):
        args = [mode]
        if mode in {"download-only", "verify-only"}:
            args.extend(["--run-tag", "existing-run"])
        assert runner._parse_args(args).mode == mode


def test_remote_runner_failed_case_preservation_manifest():
    runner = load_remote_runner()
    with tempfile.TemporaryDirectory() as temporary_directory:
        case_dir = Path(temporary_directory) / "failed-case"
        output_dir = case_dir / "output"
        output_dir.mkdir(parents=True)
        (output_dir / "stdout.txt").write_text("partial stdout\n")
        (output_dir / "stderr.txt").write_text("partial stderr\n")
        _write_json(output_dir / "case_result.json", {"status": "FAIL"})
        manifest = runner.available_case_artifacts(case_dir)
        assert manifest == [
            "output/case_result.json",
            "output/stderr.txt",
            "output/stdout.txt",
        ]


if __name__ == "__main__":
    tests = [
        test_flash_attn_263_known_qwen3_a100_vectors,
        test_flash_attn_263_early_return_and_graph_identity,
        test_flash_attn_263_rejects_unsupported_inputs,
        test_same_policy_matrix_is_exact_policy_aware_and_unique,
        test_legacy_compatibility_matrix_is_63_pairs_126_processes,
        test_case_ids_bind_split_policy_identity,
        test_gate_matrix_cardinality_and_frozen_thresholds,
        test_exact_and_rounded_graph_sizes_are_frozen,
        test_canonical_json_and_file_hashes_are_stable_and_strict,
        test_tensor_metadata_hashes_contiguous_bytes_and_reports_nonfinite,
        test_graph_size_contract_rejects_unknown_inputs,
        test_tensor_comparison_requires_finite_close_and_equal_argmax,
        test_diagnostic_classification_separates_exact_and_rounded,
        test_diagnostic_classification_rejects_duplicate_evidence,
        test_legacy_compatibility_requires_tokens_close_logits_and_kv_ownership,
        test_legacy_compatibility_missing_or_mixed_policy_is_incomplete,
        test_production_gate_frozen_boundaries,
        test_production_gate_fails_closed_on_structure_and_correctness,
        test_prompt_plan_is_deterministic_and_covers_required_trajectories,
        test_kv_observation_plan_covers_active_zero_inactive_and_sentinels,
        test_teacher_forcing_records_observed_and_reference_tokens_separately,
        test_tensor_shard_schema_rejects_missing_order_fields,
        test_tensor_shard_schema_accepts_ordered_complete_metadata,
        test_direct_model_forward_disables_autograd,
        test_direct_model_forward_and_logits_disable_autograd,
        test_build_step_split_policy_uses_exact_runtime_width_and_batch_identity,
        test_build_step_split_policy_matches_qwen3_a100_vectors,
        test_build_step_split_policy_rejects_version_and_missing_inputs,
        test_decode_graph_allocations_use_exact_identity_width,
        test_graph_cache_reuses_only_identical_identity,
        test_graph_cache_and_replay_reject_identity_mismatch,
        test_capture_operation_restores_all_active_write_slots_on_failure,
        test_step_policy_evidence_is_complete_and_stable,
        test_step_policy_evidence_matches_raw_layer_and_kv_rows,
        test_graph_identity_summary_is_ordered_unique_and_complete,
        test_verifier_recomputes_exact_policy_integrity,
        test_verifier_rejects_policy_integrity_mutations,
        test_diagnostic_case_parser_rejects_split_policy_drift,
        test_execution_policy_rejects_case_level_heuristic_split,
        test_candidate_eager_forward_observes_step_split_and_restores_auto,
        test_legacy_eager_forward_observes_auto,
        test_policy_evidence_distinguishes_same_policy_and_legacy_comparison,
        test_artifact_record_path_is_relative_and_resolves_after_relocation,
        test_verifier_reconstructs_complete_diagnostic,
        test_verifier_rejects_heuristic_manifest_contract_drift,
        test_verifier_rejects_missing_split_identity,
        test_verifier_rejects_auto_graph_as_fixed16_evidence,
        test_verifier_rejects_step_policy_identity_drift,
        test_verifier_reports_fixed_vs_auto_token_mismatch_as_legacy_incompatible,
        test_verifier_is_independent_from_diagnostic_producer,
        test_verifier_rejects_missing_matrix_case,
        test_verifier_detects_rehashed_exact_logit_mutation,
        test_verifier_rejects_source_and_environment_drift,
        test_verifier_rejects_duplicate_matrix_key_and_port,
        test_verifier_rejects_graph_size_mismatch,
        test_verifier_rejects_prompt_and_reference_hash_drift,
        test_verifier_rejects_truncated_raw_jsonl,
        test_verifier_detects_nonfinite_exact_logit,
        test_verifier_detects_exact_argmax_mismatch,
        test_verifier_detects_exact_close_threshold_failure,
        test_verifier_rejects_missing_layer_index,
        test_verifier_detects_rehashed_layer_tensor_mutation,
        test_verifier_detects_active_kv_mismatch,
        test_verifier_detects_unexpected_slot_zero_mutation,
        test_verifier_detects_unexpected_sentinel_mutation,
        test_verifier_rejects_producer_classification_tamper,
        test_verifier_rejects_missing_artifact_hash,
        test_remote_runner_has_frozen_transport_and_safety_contract,
        test_remote_runner_allocates_globally_unique_port_pairs,
        test_remote_runner_rejects_duplicate_or_equal_ports,
        test_remote_runner_retries_only_eaddrinuse,
        test_remote_runner_reads_diagnostic_stderr_for_port_collision,
        test_remote_runner_preserves_remote_shell_command_as_one_argument,
        test_remote_runner_disables_bytecode_during_source_validation,
        test_remote_runner_requires_explicit_resume_for_existing_run,
        test_remote_runner_reserves_resumed_ports_globally,
        test_remote_runner_reallocates_duplicate_ephemeral_ports,
        test_remote_runner_exposes_resume_and_verifier_python_options,
        test_remote_runner_promotes_source_evidence_artifacts,
        test_remote_runner_removes_only_downloaded_remote_case,
        test_remote_runner_builds_heuristic_smoke_for_both_gates,
        test_remote_runner_orders_each_reference_before_candidates,
        test_remote_runner_requires_paired_reference_before_candidate,
        test_remote_runner_resume_requires_policy_identity_and_artifact_hashes,
        test_remote_runner_exposes_heuristic_cli_modes,
        test_remote_runner_failed_case_preservation_manifest,
    ]
    for test in tests:
        test()
    print("multi-sequence cuda graph gate tests passed")
