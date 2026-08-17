from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import pytest
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "tools/qwen35_tp4_hybrid_prefix_benchmark_contract.py"
)


def _load_contract():
    spec = importlib.util.spec_from_file_location(
        "qwen35_tp4_hybrid_prefix_benchmark_contract",
        CONTRACT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract()
BUILDER_TEST_PATH = (
    ROOT / "tools/test_build_qwen35_tp4_performance_prerequisites.py"
)


def _load_builder_fixture():
    spec = importlib.util.spec_from_file_location(
        "qwen35_prerequisite_builder_fixture",
        BUILDER_TEST_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder_fixture = _load_builder_fixture()


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path):
    return contract.sha256_file(path)


def _authority(root, name, *, classification="PASS", source_tree=None):
    authority_dir = root / name
    authority_dir.mkdir()
    artifact = authority_dir / "artifact.json"
    verification = authority_dir / "independent_verification.json"
    provenance = authority_dir / "provenance.json"
    source_tree_sha256 = (
        contract.TP4_ROOT_SOURCE_TREE_SHA256
        if source_tree is None
        else source_tree
    )
    if name == "root-logit":
        artifact_payload, verification_payload = (
            builder_fixture._root_payloads()
        )
    elif name == "cached-continuation":
        artifact_payload, verification_payload = (
            builder_fixture._cached_payloads(source_tree_sha256)
        )
    else:
        artifact_payload, verification_payload = (
            builder_fixture._engine_payloads(source_tree_sha256)
        )
    if classification != "PASS":
        artifact_payload["classification"] = classification
        verification_payload["classification"] = classification
    _write_json(artifact, artifact_payload)
    _write_json(verification, verification_payload)
    canonical_name = {
        "root-logit": "tp4_root_logit",
        "cached-continuation": "cached_continuation",
        "engine-correctness": "engine_correctness",
    }[name]
    evidence = {}
    for filename, kind in (
        ("execution_plan.json", "plan"),
        ("consumed_authorization.json", "authorization"),
        ("execution_receipt.json", "receipt"),
    ):
        path = authority_dir / filename
        _write_json(path, {"kind": kind, "authority": canonical_name})
        evidence[filename] = _sha256(path)
    _write_json(provenance, {
        "schema_version": (
            contract.PREREQUISITE_PROVENANCE_SCHEMA_VERSION
        ),
        "authority_name": canonical_name,
        "run_tag": name,
        "binding_kind": "remote_execution_receipt",
        "source_tree_sha256": source_tree_sha256,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "root_logit_receipt_gap": False,
        "plan_path": "execution_plan.json",
        "plan_sha256": evidence["execution_plan.json"],
        "authorization_path": "consumed_authorization.json",
        "authorization_sha256": evidence[
            "consumed_authorization.json"
        ],
        "receipt_path": "execution_receipt.json",
        "receipt_sha256": evidence["execution_receipt.json"],
    })
    return {
        "run_tag": name,
        "source_tree_sha256": (
            source_tree_sha256
        ),
        "artifact_path": artifact.relative_to(root).as_posix(),
        "artifact_sha256": _sha256(artifact),
        "independent_verification_path": (
            verification.relative_to(root).as_posix()
        ),
        "independent_verification_sha256": _sha256(verification),
        "provenance_path": provenance.relative_to(root).as_posix(),
        "provenance_sha256": _sha256(provenance),
        "classification": classification,
    }


def _complete_prerequisite_fixture(root):
    payload = {
        "schema_version": contract.PREREQUISITE_SCHEMA_VERSION,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "tp4_root_logit": _authority(root, "root-logit"),
        "cached_continuation": _authority(
            root,
            "cached-continuation",
        ),
        "engine_correctness": _authority(
            root,
            "engine-correctness",
        ),
    }
    path = root / "correctness_prerequisites.json"
    _write_json(path, payload)
    return path, payload


def _passing_metrics():
    return {
        "evidence_complete": True,
        "measured_matrix_complete": True,
        "correctness_pass": True,
        "eligible_gpu_count": 4,
        "workloads": {
            "w0_short_control": {
                "median_e2e_ratio": 1.0,
                "median_decode_ratio": 1.0,
            },
            "w1_medium_reuse": {
                "median_ttft_ratio": 0.80,
                "max_repetition_ttft_ratio": 1.0,
                "median_decode_ratio": 1.0,
            },
            "w2_long_reuse": {
                "median_ttft_ratio": 0.70,
                "max_repetition_ttft_ratio": 1.0,
                "median_decode_ratio": 1.0,
            },
            "w3_batched_fanout": {
                "throughput_ratio": 1.20,
                "max_repetition_ttft_ratio": 1.0,
                "median_decode_ratio": 1.0,
            },
            "w4_miss_invalidation": {
                "median_e2e_ratio": 1.0,
                "median_decode_ratio": 1.0,
            },
        },
        "initialization_ratio": 1.0,
        "peak_cuda_reserved_ratio": 1.05,
        "scheduler_visible_kv_capacity_equal": True,
        "kv_capacity_bytes_equal": True,
        "cache_accounting_valid": True,
        "cache_within_limits": True,
        "no_required_workload_evictions": True,
    }


def test_contract_freezes_identity_policies_and_thresholds():
    assert contract.SCHEMA_VERSION == (
        "qwen35.tp4-hybrid-prefix-performance-cache.v1"
    )
    assert contract.PREREQUISITE_SCHEMA_VERSION == (
        "qwen35.tp4-performance-prerequisites.v2"
    )
    assert contract.PREREQUISITE_PROVENANCE_SCHEMA_VERSION == (
        "qwen35.tp4-performance-prerequisite-provenance.v1"
    )
    assert contract.POLICIES == ("recompute", "exact_restore")
    assert contract.WORKLOADS == (
        "w0_short_control",
        "w1_medium_reuse",
        "w2_long_reuse",
        "w3_batched_fanout",
        "w4_miss_invalidation",
    )
    assert contract.WARMUP_REPETITIONS == 1
    assert contract.CORRECTNESS_REPETITIONS == 1
    assert contract.MEASURED_REPETITIONS == 5
    assert contract.WORLD_SIZE == 4
    assert contract.MIN_GPU_FREE_BYTES == 24 * 1024**3
    assert contract.MAX_MODEL_LEN == 4096
    assert contract.MODEL_MANIFEST_SHA256 == (
        "3e650a908234771c3cf1ac4e20c4d38f"
        "e69982efedaf4a3e631ad0b14aad7dd0"
    )
    assert contract.TP4_ROOT_SOURCE_TREE_SHA256 == (
        "37135279047a569df8e0d26c6e396472"
        "02b27aca758ac8ac135bdab25612f20a"
    )
    assert contract.ENGINE_CORRECTNESS_SCENARIOS == {
        "construct_and_bind": (0, 0, 0, 0, 0, 0, 0, 0),
        "publish_source": (1, 1, 1, 1, 0, 0, 1, 1),
        "restore_w1": (64, 64, 64, 0, 1, 0, 0, 1),
        "miss_w4_token": (33, 33, 32, 1, 0, 0, 0, 2),
        "miss_w4_stale": (33, 33, 32, 1, 0, 1, 0, 1),
        "miss_w4_clear": (33, 33, 32, 1, 0, 1, 0, 1),
    }
    assert contract.THRESHOLDS == {
        "w1_ttft_max_ratio": 0.85,
        "w2_ttft_max_ratio": 0.75,
        "w3_throughput_min_ratio": 1.15,
        "per_repetition_ttft_max_ratio": 1.05,
        "decode_latency_max_ratio": 1.02,
        "initialization_max_ratio": 1.10,
        "control_e2e_max_ratio": 1.05,
        "peak_cuda_reserved_max_ratio": 1.10,
    }


def test_workload_contract_is_exact_and_compression_free():
    assert contract.WORKLOAD_SPECS == {
        "w0_short_control": {
            "shared_prefix_tokens": 256,
            "suffix_tokens": 32,
            "continuations": 1,
            "generated_tokens": 32,
            "kind": "reuse",
        },
        "w1_medium_reuse": {
            "shared_prefix_tokens": 1024,
            "suffix_tokens": 64,
            "continuations": 4,
            "generated_tokens": 64,
            "kind": "reuse",
        },
        "w2_long_reuse": {
            "shared_prefix_tokens": 3840,
            "suffix_tokens": 64,
            "continuations": 4,
            "generated_tokens": 64,
            "kind": "reuse",
        },
        "w3_batched_fanout": {
            "shared_prefix_tokens": 2048,
            "suffix_tokens": 64,
            "continuations": 8,
            "generated_tokens": 32,
            "kind": "batched_reuse",
        },
        "w4_miss_invalidation": {
            "shared_prefix_tokens": 1024,
            "suffix_tokens": 64,
            "continuations": 3,
            "generated_tokens": 32,
            "kind": "miss_control",
        },
    }
    assert contract.EXCLUDED_CANDIDATES == (
        "int4_state",
        "token_sparse_state",
        "low_rank_state",
        "gist_layer_share",
    )


def test_workload_requests_fit_engine_max_model_len():
    assert contract.MAX_MODEL_LEN == 4096

    for workload, spec in contract.WORKLOAD_SPECS.items():
        source_seed_tokens = (
            spec["shared_prefix_tokens"]
            + spec["suffix_tokens"]
            + 1
        )
        continuation_tokens = (
            spec["shared_prefix_tokens"]
            + spec["suffix_tokens"]
            + spec["generated_tokens"]
        )

        assert source_seed_tokens <= contract.MAX_MODEL_LEN, (
            f"{workload} source seed exceeds max_model_len: "
            f"{source_seed_tokens} > {contract.MAX_MODEL_LEN}"
        )
        assert continuation_tokens <= contract.MAX_MODEL_LEN, (
            f"{workload} continuation exceeds max_model_len: "
            f"{continuation_tokens} > {contract.MAX_MODEL_LEN}"
        )


def test_workload_payload_rejects_requests_over_max_model_len():
    original = contract.WORKLOAD_SPECS["w2_long_reuse"]
    contract.WORKLOAD_SPECS["w2_long_reuse"] = {
        **original,
        "shared_prefix_tokens": contract.MAX_MODEL_LEN,
    }
    try:
        with pytest.raises(
            ValueError,
            match=(
                "w2_long_reuse source seed exceeds max_model_len: "
                "4161 > 4096"
            ),
        ):
            contract.workload_payload("w2_long_reuse")
    finally:
        contract.WORKLOAD_SPECS["w2_long_reuse"] = original


def test_workload_manifest_freezes_reconstructable_token_ids():
    manifest = contract.workload_manifest_payload()

    assert manifest["schema_version"] == contract.SCHEMA_VERSION
    assert tuple(manifest["workloads"]) == contract.WORKLOADS
    for workload_index, workload in enumerate(contract.WORKLOADS):
        spec = contract.WORKLOAD_SPECS[workload]
        payload = manifest["workloads"][workload]
        assert payload["spec"] == spec
        assert len(payload["shared_prefix_token_ids"]) == (
            spec["shared_prefix_tokens"]
        )
        assert len(payload["source_suffix_token_ids"]) == (
            spec["suffix_tokens"]
        )
        assert len(payload["continuations"]) == spec["continuations"]
        assert all(
            len(row["suffix_token_ids"]) == spec["suffix_tokens"]
            for row in payload["continuations"]
        )
        assert len({
            tuple(row["suffix_token_ids"])
            for row in payload["continuations"]
        }) == spec["continuations"]
        assert all(
            isinstance(token_id, int)
            and not isinstance(token_id, bool)
            and 0 <= token_id < contract.TOKEN_ID_UPPER_BOUND
            for token_id in (
                payload["shared_prefix_token_ids"]
                + payload["source_suffix_token_ids"]
                + [
                    token_id
                    for row in payload["continuations"]
                    for token_id in row["suffix_token_ids"]
                ]
            )
        )
        assert payload["token_seed"] == 2026072900 + workload_index


def test_w4_manifest_freezes_three_distinct_miss_controls():
    payload = contract.workload_manifest_payload()["workloads"][
        "w4_miss_invalidation"
    ]
    rows = payload["continuations"]

    assert [row["invalidation"] for row in rows] == [
        {
            "kind": "token_mismatch",
            "prefix_index": 512,
            "replacement_token_id": rows[0]["prefix_overrides"][0][1],
        },
        {"kind": "stale_block_generation"},
        {"kind": "cache_clear"},
    ]
    assert rows[0]["prefix_overrides"] == [[
        512,
        rows[0]["invalidation"]["replacement_token_id"],
    ]]
    assert (
        rows[0]["invalidation"]["replacement_token_id"]
        != payload["shared_prefix_token_ids"][512]
    )
    assert rows[1]["prefix_overrides"] == []
    assert rows[2]["prefix_overrides"] == []


def test_case_matrix_is_complete_unique_and_deterministic():
    first = contract.build_case_matrix()
    second = contract.build_case_matrix()

    assert first == second
    assert len(first) == (
        len(contract.WORKLOADS)
        * len(contract.POLICIES)
        * (
            contract.WARMUP_REPETITIONS
            + contract.CORRECTNESS_REPETITIONS
            + contract.MEASURED_REPETITIONS
        )
    )
    assert len({case.case_id for case in first}) == len(first)
    assert {
        case.phase for case in first
    } == {"warmup", "correctness", "measured"}
    for workload in contract.WORKLOADS:
        for policy in contract.POLICIES:
            rows = [
                case
                for case in first
                if case.workload == workload
                and case.policy == policy
            ]
            assert sum(case.phase == "warmup" for case in rows) == 1
            assert sum(case.phase == "correctness" for case in rows) == 1
            assert [
                case.repetition
                for case in rows
                if case.phase == "measured"
            ] == list(range(contract.MEASURED_REPETITIONS))


def test_pair_order_alternates_without_changing_policy_set():
    assert contract.pair_order(0) == ("recompute", "exact_restore")
    assert contract.pair_order(1) == ("exact_restore", "recompute")
    assert contract.pair_order(4) == ("recompute", "exact_restore")
    assert contract.pair_order(5) == ("exact_restore", "recompute")


def test_artifact_and_row_schemas_are_closed():
    assert contract.TOP_LEVEL_ARTIFACTS == (
        "correctness_prerequisites.json",
        "workload_manifest.json",
        "source_manifest.json",
        "environment.json",
        "gpu_assignments.json",
        "commands.json",
        "case_rows.jsonl",
        "process_rows.jsonl",
        "logits_manifest.json",
        "worker_logs_manifest.json",
        "summary.json",
        "artifact_manifest.json",
        "independent_verification.json",
        "report.md",
    )
    assert contract.ARTIFACT_MANIFEST_HASH_DOMAIN == (
        "correctness_prerequisites.json",
        "workload_manifest.json",
        "source_manifest.json",
        "environment.json",
        "gpu_assignments.json",
        "commands.json",
        "case_rows.jsonl",
        "process_rows.jsonl",
        "logits_manifest.json",
        "worker_logs_manifest.json",
        "summary.json",
    )
    assert contract.NESTED_ARTIFACT_DIRECTORIES == (
        "prerequisites",
        "logits",
        "logs",
    )
    assert "output_token_ids" in contract.CASE_ROW_FIELDS
    assert "final_logits_sha256" in contract.CASE_ROW_FIELDS
    assert "hybrid_cache_current_bytes" in contract.PROCESS_ROW_FIELDS
    assert "hybrid_cache_current_logical_bytes" in (
        contract.PROCESS_ROW_FIELDS
    )
    assert "kv_capacity_bytes" in contract.PROCESS_ROW_FIELDS


def test_missing_prerequisite_file_blocks_without_launch_authorization():
    with tempfile.TemporaryDirectory() as temporary:
        result = contract.validate_prerequisites(
            Path(temporary) / "missing.json"
        )

    assert result.classification == "BLOCKED_CORRECTNESS"
    assert result.authorized is False
    assert "missing" in result.reasons[0]


def test_complete_prerequisite_fixture_authorizes_benchmark():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, _ = _complete_prerequisite_fixture(root)

        result = contract.validate_prerequisites(path)

    assert result.classification == "PASS"
    assert result.authorized is True
    assert result.reasons == ()


def test_legacy_root_directory_only_provenance_blocks_benchmark():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, payload = _complete_prerequisite_fixture(root)
        provenance_path = (
            root / payload["tp4_root_logit"]["provenance_path"]
        )
        provenance = json.loads(provenance_path.read_text())
        provenance.update({
            "binding_kind": "complete_directory_only",
            "root_logit_receipt_gap": True,
            "plan_path": None,
            "plan_sha256": None,
            "authorization_path": None,
            "authorization_sha256": None,
            "receipt_path": None,
            "receipt_sha256": None,
        })
        _write_json(provenance_path, provenance)
        payload["tp4_root_logit"]["provenance_sha256"] = _sha256(
            provenance_path
        )
        _write_json(path, payload)

        result = contract.validate_prerequisites(path)

    assert result.classification == "BLOCKED_CORRECTNESS"
    assert result.authorized is False
    assert "receipt provenance" in " ".join(result.reasons)


def test_non_pass_and_hash_tamper_block_prerequisites():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, payload = _complete_prerequisite_fixture(root)
        payload["cached_continuation"]["classification"] = "NO_GO"
        _write_json(path, payload)
        result = contract.validate_prerequisites(path)
        assert result.classification == "BLOCKED_CORRECTNESS"
        assert "cached_continuation" in " ".join(result.reasons)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, payload = _complete_prerequisite_fixture(root)
        payload["engine_correctness"]["artifact_sha256"] = "0" * 64
        _write_json(path, payload)
        result = contract.validate_prerequisites(path)
        assert result.classification == "BLOCKED_CORRECTNESS"
        assert "artifact SHA" in " ".join(result.reasons)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, payload = _complete_prerequisite_fixture(root)
        payload["cached_continuation"]["provenance_sha256"] = "0" * 64
        _write_json(path, payload)
        result = contract.validate_prerequisites(path)
        assert result.classification == "BLOCKED_CORRECTNESS"
        assert "provenance SHA" in " ".join(result.reasons)


def test_forged_two_field_pass_documents_block_prerequisites():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        payload = {
            "schema_version": contract.PREREQUISITE_SCHEMA_VERSION,
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        }
        for key, directory in (
            ("tp4_root_logit", "root-logit"),
            ("cached_continuation", "cached-continuation"),
            ("engine_correctness", "engine-correctness"),
        ):
            row = _authority(root, directory)
            for field in (
                "artifact_path",
                "independent_verification_path",
            ):
                path = root / row[field]
                _write_json(path, {
                    "classification": "PASS",
                    "model_manifest_sha256": (
                        contract.MODEL_MANIFEST_SHA256
                    ),
                })
                row[
                    field.replace("_path", "_sha256")
                ] = _sha256(path)
            payload[key] = row
        path = root / "correctness_prerequisites.json"
        _write_json(path, payload)

        result = contract.validate_prerequisites(path)

    assert result.classification == "BLOCKED_CORRECTNESS"
    assert result.authorized is False
    assert "schema" in " ".join(result.reasons)


def test_wrong_model_or_root_source_blocks_prerequisites():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, payload = _complete_prerequisite_fixture(root)
        payload["model_manifest_sha256"] = "f" * 64
        _write_json(path, payload)
        result = contract.validate_prerequisites(path)
        assert result.classification == "BLOCKED_CORRECTNESS"
        assert "model manifest" in " ".join(result.reasons)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, payload = _complete_prerequisite_fixture(root)
        payload["tp4_root_logit"]["source_tree_sha256"] = "e" * 64
        _write_json(path, payload)
        result = contract.validate_prerequisites(path)
        assert result.classification == "BLOCKED_CORRECTNESS"
        assert "root-logit source tree" in " ".join(result.reasons)


def test_classifier_distinguishes_blocked_invalid_no_go_and_go():
    passing = _passing_metrics()
    assert contract.classify_run(passing) == "GO"

    blocked_correctness = dict(passing)
    blocked_correctness["prerequisites_pass"] = False
    assert (
        contract.classify_run(blocked_correctness)
        == "BLOCKED_CORRECTNESS"
    )

    blocked_resources = dict(passing)
    blocked_resources["eligible_gpu_count"] = 3
    assert (
        contract.classify_run(blocked_resources)
        == "BLOCKED_RESOURCES"
    )

    invalid = dict(passing)
    invalid["correctness_pass"] = False
    assert contract.classify_run(invalid) == "INVALID"

    no_go = _passing_metrics()
    no_go["workloads"] = dict(no_go["workloads"])
    no_go["workloads"]["w2_long_reuse"] = dict(
        no_go["workloads"]["w2_long_reuse"]
    )
    no_go["workloads"]["w2_long_reuse"][
        "median_ttft_ratio"
    ] = 0.80
    assert contract.classify_run(no_go) == "NO_GO"


def test_classifier_rejects_cache_accounting_as_invalid():
    metrics = _passing_metrics()
    metrics["cache_accounting_valid"] = False

    assert contract.classify_run(metrics) == "INVALID"


def test_classifier_rejects_missing_measured_repetition_as_invalid():
    metrics = _passing_metrics()
    metrics["measured_matrix_complete"] = False

    assert contract.classify_run(metrics) == "INVALID"


def test_classifier_treats_capacity_mismatch_as_no_go():
    metrics = _passing_metrics()
    metrics["scheduler_visible_kv_capacity_equal"] = False

    assert contract.classify_run(metrics) == "NO_GO"


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 hybrid-prefix benchmark contract tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
