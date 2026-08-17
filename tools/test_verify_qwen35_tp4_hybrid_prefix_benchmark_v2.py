from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name, filename):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract_for_verifier_test",
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py",
)
assembler_test = _load(
    "test_qwen35_tp4_hybrid_prefix_benchmark_v2_assembler_for_verifier",
    "test_qwen35_tp4_hybrid_prefix_benchmark_v2_assembler.py",
)
verifier = _load(
    "verify_qwen35_tp4_hybrid_prefix_benchmark_v2",
    "verify_qwen35_tp4_hybrid_prefix_benchmark_v2.py",
)


LOGIT_ATOL = 2e-5
LOGIT_RTOL = 0.0
FORBIDDEN_COUNTERS = (
    "hybrid_cache_evictions",
    "hybrid_cache_validation_failures",
    "hybrid_cache_failed_restores",
    "hybrid_cache_quarantines",
    "hybrid_cache_failed_rollbacks",
    "hybrid_cache_corruption_events",
    "hybrid_cache_partial_restore_attempts",
    "hybrid_cache_fallbacks",
    "hybrid_cache_mixed_representation_events",
    "hybrid_cache_missing_layer_events",
    "oom_events",
    "undeclared_eviction_events",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_canonical_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contract.canonical_json_bytes(value) + b"\n")


def _write_jsonl(path, rows):
    Path(path).write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _set_exact_recurrent_elements(raw_root, elements):
    for case in contract.build_case_matrix():
        if case.profile == "recompute":
            continue
        case_dir = (
            Path(raw_root)
            / "profiles"
            / case.profile
            / "cases"
            / case.case_id
        )
        process_path = case_dir / "process_rows.jsonl"
        process_rows = _read_jsonl(process_path)
        for process_row in process_rows:
            tensor_path = (
                case_dir
                / "tensor-inventories"
                / f"rank-{process_row['rank']}.json"
            )
            evidence = _read_json(tensor_path)
            for observation in evidence["observations"]:
                observation["cuda_allocated_bytes"] = 800
                observation["cuda_reserved_bytes"] = 1000
            recurrent_storage_ids = set()
            if case.profile == contract.P1_REFERENCE_PROFILE:
                for snapshot in evidence["snapshots"]:
                    for reference in snapshot["tensor_references"]:
                        if reference["semantic_role"] != "recurrent_values":
                            continue
                        reference["logical_shape"] = [1, 1, elements]
                        reference["resident_shape"] = [1, 1, elements]
                        reference["storage_length_bytes"] = elements * 4
                        recurrent_storage_ids.add(reference["storage_id"])
                for storage in evidence["storages"]:
                    if storage["storage_id"] in recurrent_storage_ids:
                        storage["storage_nbytes"] = elements * 4
            _write_json(tensor_path, evidence)
            process_row.update(
                contract.recompute_tensor_storage_accounting(evidence)
            )
        _write_jsonl(process_path, process_rows)


def _complete_assembled_fixture(root):
    root = Path(root)
    raw_root = assembler_test._canonical_raw_bundle(root)
    _set_exact_recurrent_elements(raw_root, 4)
    run_dir = root / "assembled"
    assembler_test._assemble(raw_root, run_dir)
    return run_dir


def _assert_contract_valid_artifact(run_dir):
    manifests = {
        kind: _read_json(
            run_dir / contract.NESTED_MANIFEST_ARTIFACT_PATHS[kind]
        )
        for kind in contract.NESTED_MANIFEST_KINDS
    }
    file_inventory = sorted(
        [
            file_row
            for kind in contract.NESTED_MANIFEST_KINDS
            for file_row in manifests[kind]["files"]
        ],
        key=lambda row: row["path"],
    )
    contract.validate_artifact_evidence(
        _read_jsonl(run_dir / "case_rows.jsonl"),
        _read_jsonl(run_dir / "process_rows.jsonl"),
        manifests,
        file_inventory,
        _read_json(run_dir / "artifact_manifest.json"),
    )


def _resign_artifact_manifest(run_dir):
    entries = []
    for relative in contract.ARTIFACT_MANIFEST_HASH_DOMAIN:
        path = run_dir / relative
        entries.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "producer": "task8-assembler",
                "trust_domain": "producer",
            }
        )
    _write_json(
        run_dir / "artifact_manifest.json",
        {
            "schema_version": contract.ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "hash_domain": list(contract.ARTIFACT_MANIFEST_HASH_DOMAIN),
            "entries": entries,
            "excluded_verifier_outputs": list(contract.VERIFIER_TRUST_DOMAIN),
        },
    )


def _replace_first_logit(run_dir, profile, value):
    rows = _read_jsonl(run_dir / "case_rows.jsonl")
    row = next(row for row in rows if row["profile"] == profile)
    path = run_dir / row["final_logits_path"]
    data = bytearray(path.read_bytes())
    struct.pack_into("<f", data, 0, value)
    path.write_bytes(data)
    row["final_logits_sha256"] = _sha256(path)
    _write_jsonl(run_dir / "case_rows.jsonl", rows)
    _refresh_nested_manifest_file(run_dir, "logits_manifest.json", path)
    _resign_artifact_manifest(run_dir)


def _replace_first_output_token(run_dir, profile, value):
    rows = _read_jsonl(run_dir / "case_rows.jsonl")
    row = next(row for row in rows if row["profile"] == profile)
    path = run_dir / row["output_token_ids_path"]
    values = _read_json(path)
    values[0] = value
    _write_json(path, values)
    row["output_token_ids_sha256"] = _sha256(path)
    _write_jsonl(run_dir / "case_rows.jsonl", rows)
    _refresh_nested_manifest_file(run_dir, "token_manifest.json", path)
    _resign_artifact_manifest(run_dir)


def _replace_process_metric(run_dir, profile, field, value):
    if field in contract.TENSOR_STORAGE_ACCOUNTING_FIELDS:
        def mutate(evidence):
            if field == "cuda_peak_reserved_bytes":
                for observation in evidence["observations"]:
                    observation["cuda_reserved_bytes"] = value
                return
            raise AssertionError(f"unsupported accounting mutation: {field}")

        _replace_tensor_evidence(run_dir, profile, mutate)
        return
    rows = _read_jsonl(run_dir / "process_rows.jsonl")
    matching_rows = [row for row in rows if row["profile"] == profile]
    assert matching_rows
    for row in matching_rows:
        row[field] = value
    _write_jsonl(run_dir / "process_rows.jsonl", rows)
    _resign_artifact_manifest(run_dir)


def _replace_process_storage_metric(
    run_dir,
    profile,
    process_field,
    evidence_field,
    value,
):
    assert process_field == "hybrid_cache_current_unique_physical_bytes"
    assert evidence_field == "physical_bytes"

    def expand_recurrent_storage(evidence):
        recurrent_storage_ids = set()
        for snapshot in evidence["snapshots"]:
            for reference in snapshot["tensor_references"]:
                if reference["semantic_role"] != "recurrent_values":
                    continue
                reference["logical_shape"] = [1, 1, 3]
                reference["resident_shape"] = [1, 1, 3]
                reference["storage_length_bytes"] = 3
                recurrent_storage_ids.add(reference["storage_id"])
            snapshot["codec_metadata"]["layers"] = [
                {
                    **layer,
                    "source_shape": [1, 1, 3],
                }
                for layer in snapshot["codec_metadata"]["layers"]
            ]
        for storage in evidence["storages"]:
            if storage["storage_id"] in recurrent_storage_ids:
                storage["storage_nbytes"] = 3

    _replace_tensor_evidence(run_dir, profile, expand_recurrent_storage)


def _replace_tensor_evidence(run_dir, profile, mutate):
    rows = _read_jsonl(run_dir / "process_rows.jsonl")
    matching_rows = [row for row in rows if row["profile"] == profile]
    assert matching_rows
    for row in matching_rows:
        tensor_path = (
            run_dir
            / "snapshots"
            / row["case_id"]
            / f"rank-{row['rank']}.tensor-inventory.json"
        )
        evidence = _read_json(tensor_path)
        mutate(evidence)
        _write_canonical_json(tensor_path, evidence)
        accounting = contract.recompute_tensor_storage_accounting(evidence)
        row.update(accounting)
        tensor_manifest = _read_json(
            run_dir / "tensor_inventory_manifest.json"
        )
        tensor_row = next(
            candidate
            for candidate in tensor_manifest["rows"]
            if candidate["case_id"] == row["case_id"]
            and candidate["rank"] == row["rank"]
        )
        updated_file = {
            "path": tensor_path.relative_to(run_dir).as_posix(),
            "sha256": _sha256(tensor_path),
            "bytes": tensor_path.stat().st_size,
            "type": "regular_file",
        }
        next(
            candidate
            for candidate in tensor_manifest["files"]
            if candidate["path"] == updated_file["path"]
        ).update(updated_file)
        tensor_row.update(
            {
                "evidence_schema_version": evidence["schema_version"],
                "snapshot_count": len(evidence["snapshots"]),
                "storage_count": len(evidence["storages"]),
                "reference_count": sum(
                    len(snapshot["tensor_references"])
                    for snapshot in evidence["snapshots"]
                ),
                "observation_count": len(evidence["observations"]),
                "evidence": evidence,
                "file": updated_file,
            }
        )
        _write_canonical_json(
            run_dir / "tensor_inventory_manifest.json",
            tensor_manifest,
        )
        snapshot_manifest = _read_json(run_dir / "snapshot_manifest.json")
        snapshot_row = next(
            candidate
            for candidate in snapshot_manifest["rows"]
            if candidate["case_id"] == row["case_id"]
            and candidate["rank"] == row["rank"]
        )
        snapshot_row["tensor_inventory_file"] = updated_file
        for field in contract.TENSOR_STORAGE_ACCOUNTING_FIELDS:
            snapshot_row[field] = accounting[field]
        snapshot_row.update(
            {
                "full_fidelity_logical_bytes": accounting[
                    "hybrid_cache_current_logical_referenced_bytes"
                ],
                "encoded_physical_bytes": accounting[
                    "hybrid_cache_current_unique_physical_bytes"
                ],
                "codec_metadata_bytes": accounting[
                    "hybrid_cache_current_metadata_bytes"
                ],
                "temporary_encode_workspace_bytes": accounting[
                    "encode_workspace_peak_allocated_bytes"
                ],
                "temporary_decode_workspace_bytes": accounting[
                    "decode_workspace_peak_allocated_bytes"
                ],
            }
        )
        _write_canonical_json(
            run_dir / "snapshot_manifest.json",
            snapshot_manifest,
        )
    _write_jsonl(run_dir / "process_rows.jsonl", rows)
    _resign_artifact_manifest(run_dir)


def _assert_legitimate_classification(run_dir, expected):
    _assert_contract_valid_artifact(run_dir)
    result = verifier.verify_run(run_dir)

    assert result["classification"] == expected
    assert _read_json(run_dir / "independent_verification.json") == result
    assert f"Classification: `{expected}`" in (
        run_dir / "report.md"
    ).read_text(encoding="utf-8")
    return result


def test_canonical_assembler_artifact_is_accepted_directly():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _complete_assembled_fixture(temporary)
        _assert_contract_valid_artifact(run_dir)

        result = verifier.verify_run(run_dir)

        assert result["classification"] == "GO"
        assert _read_json(run_dir / "independent_verification.json") == result


def test_legacy_private_manifest_dialect_is_invalid_artifact():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _complete_assembled_fixture(temporary)
        token_manifest = _read_json(run_dir / "token_manifest.json")
        _write_json(
            run_dir / "token_manifest.json",
            {"files": token_manifest["files"]},
        )
        _resign_artifact_manifest(run_dir)

        try:
            verifier.verify_run(run_dir)
        except verifier.VerificationError as error:
            assert error.classification == "INVALID_ARTIFACT"
        else:
            raise AssertionError("legacy private manifest dialect was accepted")

        assert not (run_dir / "independent_verification.json").exists()
        assert not (run_dir / "report.md").exists()


def test_verify_run_accepts_float32_logits_within_tolerance_via_torch_assert_close():
    import torch

    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _complete_assembled_fixture(temporary)
        _replace_first_logit(run_dir, contract.P2_PROFILE, 1e-5)
        calls = []
        original_assert_close = torch.testing.assert_close

        def spy_assert_close(actual, expected, *, atol, rtol):
            calls.append((actual, expected, atol, rtol))
            return original_assert_close(
                actual,
                expected,
                atol=atol,
                rtol=rtol,
            )

        torch.testing.assert_close = spy_assert_close
        try:
            result = verifier.verify_run(run_dir)
        finally:
            torch.testing.assert_close = original_assert_close

        assert result["classification"] == "GO"
        assert calls
        assert all(
            isinstance(candidate, torch.Tensor)
            and isinstance(exact, torch.Tensor)
            and candidate.dtype == torch.float32
            and exact.dtype == torch.float32
            and atol == 2e-5
            and rtol == 0.0
            for candidate, exact, atol, rtol in calls
        )


def test_verify_run_classifies_valid_token_mismatch_as_no_go_correctness():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _complete_assembled_fixture(temporary)
        _replace_first_output_token(run_dir, contract.P2_PROFILE, 9999)

        result = _assert_legitimate_classification(
            run_dir,
            "NO_GO_CORRECTNESS",
        )

        assert result["correctness"]["token_ids_equal"] is False
        assert result["correctness"]["logits_assert_close"] is True


def test_verify_run_classifies_valid_logit_mismatch_as_no_go_correctness():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _complete_assembled_fixture(temporary)
        _replace_first_logit(run_dir, contract.P2_PROFILE, 3e-5)

        result = _assert_legitimate_classification(
            run_dir,
            "NO_GO_CORRECTNESS",
        )

        assert result["correctness"]["token_ids_equal"] is True
        assert result["correctness"]["logits_assert_close"] is False


def test_verify_run_classifies_valid_safety_counter_as_no_go_runtime_safety():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _complete_assembled_fixture(temporary)
        _replace_process_metric(
            run_dir,
            contract.P2_PROFILE,
            "oom_events",
            1,
        )

        result = _assert_legitimate_classification(
            run_dir,
            "NO_GO_RUNTIME_SAFETY",
        )

        assert result["safety"]["forbidden_event_counts"]["oom_events"] > 0


def test_verify_run_classifies_valid_cache_miss_as_no_go_cache():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _complete_assembled_fixture(temporary)
        _replace_process_storage_metric(
            run_dir,
            contract.P2_PROFILE,
            "hybrid_cache_current_unique_physical_bytes",
            "physical_bytes",
            500,
        )

        result = _assert_legitimate_classification(run_dir, "NO_GO_CACHE")

        assert (
            result["cache"]["int8_to_exact_unique_physical_bytes_ratio"]
            == 0.5
        )


def test_verify_run_classifies_valid_capacity_miss_as_no_go_cache():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _complete_assembled_fixture(temporary)
        _replace_process_metric(
            run_dir,
            contract.P2_PROFILE,
            "same_budget_entry_capacity",
            20,
        )

        result = _assert_legitimate_classification(run_dir, "NO_GO_CACHE")

        assert (
            result["capacity"]["int8_to_exact_same_budget_capacity_ratio"]
            == 2.0
        )


def test_verify_run_classifies_valid_performance_miss_as_no_go_performance():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _complete_assembled_fixture(temporary)
        rows = _read_jsonl(run_dir / "case_rows.jsonl")
        for row in rows:
            if (
                row["phase"] == "measured"
                and row["profile"] == contract.P2_PROFILE
                and row["workload"] == "w1_medium_reuse"
            ):
                row["ttft_ns"] = 840
        _write_jsonl(run_dir / "case_rows.jsonl", rows)
        _resign_artifact_manifest(run_dir)

        result = _assert_legitimate_classification(
            run_dir,
            "NO_GO_PERFORMANCE",
        )

        assert (
            result["performance"]["w1"][
                "int8_to_exact_median_ttft_ratio"
            ]
            == 1.05
        )


def test_verify_run_enforces_w2_median_ttft_relative_to_exact_restore():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _complete_assembled_fixture(temporary)
        rows = _read_jsonl(run_dir / "case_rows.jsonl")
        for row in rows:
            if (
                row["phase"] == "measured"
                and row["profile"] == contract.P2_PROFILE
                and row["workload"] == "w2_long_reuse"
            ):
                row["ttft_ns"] = 735
        _write_jsonl(run_dir / "case_rows.jsonl", rows)
        _resign_artifact_manifest(run_dir)

        result = _assert_legitimate_classification(
            run_dir,
            "NO_GO_PERFORMANCE",
        )

        assert (
            result["performance"]["w2"][
                "int8_to_exact_median_ttft_ratio"
            ]
            == 1.05
        )


def test_verify_run_enforces_every_w2_ttft_relative_to_exact_restore():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _complete_assembled_fixture(temporary)
        rows = _read_jsonl(run_dir / "case_rows.jsonl")
        for row in rows:
            if (
                row["phase"] == "measured"
                and row["repetition"] == 0
                and row["profile"] == contract.P2_PROFILE
                and row["workload"] == "w2_long_reuse"
            ):
                row["ttft_ns"] = 742
        _write_jsonl(run_dir / "case_rows.jsonl", rows)
        _resign_artifact_manifest(run_dir)

        result = _assert_legitimate_classification(
            run_dir,
            "NO_GO_PERFORMANCE",
        )

        assert (
            result["performance"]["w2"][
                "int8_to_exact_median_ttft_ratio"
            ]
            == 1.0
        )
        assert (
            result["performance"]["w2"][
                "int8_to_exact_every_ttft_max_ratio"
            ]
            == 1.06
        )


def test_verify_run_enforces_w1_median_ttft_relative_to_recompute():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _complete_assembled_fixture(temporary)
        rows = _read_jsonl(run_dir / "case_rows.jsonl")
        for row in rows:
            if (
                row["phase"] == "measured"
                and row["workload"] == "w1_medium_reuse"
            ):
                if row["profile"] == contract.P1_REFERENCE_PROFILE:
                    row["ttft_ns"] = 900
                elif row["profile"] == contract.P2_PROFILE:
                    row["ttft_ns"] = 880
        _write_jsonl(run_dir / "case_rows.jsonl", rows)
        _resign_artifact_manifest(run_dir)

        result = _assert_legitimate_classification(
            run_dir,
            "NO_GO_PERFORMANCE",
        )

        assert (
            result["performance"]["w1"][
                "int8_to_exact_median_ttft_ratio"
            ]
            == 880 / 900
        )
        assert (
            result["performance"]["w1"][
                "int8_to_recompute_median_ttft_ratio"
            ]
            == 0.88
        )


def test_verify_run_classifies_valid_memory_miss_as_no_go_performance():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _complete_assembled_fixture(temporary)
        _replace_process_metric(
            run_dir,
            contract.P2_PROFILE,
            "cuda_peak_reserved_bytes",
            1100,
        )

        result = _assert_legitimate_classification(
            run_dir,
            "NO_GO_PERFORMANCE",
        )

        assert (
            result["memory"]["int8_to_exact_peak_cuda_reserved_ratio"]
            == 1.1
        )
        report = (run_dir / "report.md").read_text(encoding="utf-8")
        assert (
            "`int8_to_exact_peak_cuda_reserved_max_ratio` | `1.100000` "
            "| `<=` | `1.05` | `FAIL`"
        ) in report


def test_verify_run_recomputes_all_gates_and_ignores_lying_producer_summary():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _complete_assembled_fixture(temporary)

        result = verifier.verify_run(run_dir)

        assert result["classification"] == "GO"
        assert _read_json(run_dir / "independent_verification.json") == result
        assert "Classification: `GO`" in (
            run_dir / "report.md"
        ).read_text(encoding="utf-8")
        assert result["logit_tolerance"] == {"atol": 2e-5, "rtol": 0.0}
        assert result["correctness"]["token_ids_equal"] is True
        assert result["correctness"]["logits_assert_close"] is True
        assert (
            result["cache"]["int8_to_exact_unique_physical_bytes_ratio"]
            == 7 / 18
        )
        assert result["capacity"]["int8_to_exact_same_budget_capacity_ratio"] == 2.5
        assert result["performance"]["w1"]["int8_to_exact_median_ttft_ratio"] == 1.0
        assert result["performance"]["w1"]["int8_to_exact_every_ttft_max_ratio"] == 1.0
        assert result["performance"]["w2"]["int8_to_recompute_median_ttft_ratio"] == 0.7
        assert result["performance"]["w3"]["int8_to_exact_throughput_ratio"] == 1.0
        assert result["performance"]["w3"]["int8_to_recompute_throughput_ratio"] == 1.25
        assert result["performance"]["int8_to_recompute_decode_latency_ratio"] == 1.0
        assert result["memory"]["int8_to_exact_peak_cuda_reserved_ratio"] == 1.0
        assert result["safety"]["forbidden_event_counts"] == {
            field: 0 for field in FORBIDDEN_COUNTERS
        }
        assert (run_dir / "independent_verification.json").is_file()
        assert (run_dir / "report.md").is_file()
        report = (run_dir / "report.md").read_text(encoding="utf-8")
        for label in (
            "W1 int8/exact median TTFT ratio",
            "W1 int8/exact every-repetition TTFT max ratio",
            "W1 int8/recompute median TTFT ratio",
            "W2 int8/exact median TTFT ratio",
            "W2 int8/exact every-repetition TTFT max ratio",
            "W2 int8/recompute median TTFT ratio",
            "W3 int8/exact concurrent E2E proxy ratio",
            "W3 int8/recompute concurrent E2E proxy ratio",
            "Int8/recompute decode latency ratio",
            "Int8/exact peak CUDA reserved ratio",
            "Int8/exact unique physical cache bytes ratio",
            "Int8/exact same-budget capacity ratio",
        ):
            assert label in report
        for threshold_name, threshold in contract.THRESHOLDS.items():
            assert f"`{threshold_name}`: `{threshold}`" in report
        assert "W3 measured concurrency: `8`" in report
        assert (
            "not sustained serving QPS, tokens/s, arrival-rate saturation, "
            "or batch makespan"
        ) in report
        source = _read_json(run_dir / "source_manifest.json")
        assert f"Source tree SHA256: `{source['source_tree_sha256']}`" in report
        assert (
            f"Model manifest SHA256: `{contract.MODEL_MANIFEST_SHA256}`"
            in report
        )
        assert (
            f"Workload manifest SHA256: `{_sha256(run_dir / 'workload_manifest.json')}`"
            in report
        )
        assert "TP world size: `4`" in report
        assert "Assigned GPU indices: `[2, 4, 5, 6]`" in report
        assert (
            "Claim boundary: this classification applies only to the bound "
            "model, source, workload, configuration, thresholds, and "
            "artifact hashes listed below."
        ) in report
        for entry in _read_json(run_dir / "artifact_manifest.json")["entries"]:
            assert f"`{entry['path']}`: `{entry['sha256']}`" in report


def test_run_local_receipt_or_authority_is_rejected_as_unknown_inventory():
    injected_documents = {
        "execution_receipt.json": {"classification": "PASS"},
        "verification_authority.json": {"classification": "BOUND"},
    }
    for name, payload in injected_documents.items():
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _complete_assembled_fixture(temporary)
            _write_json(run_dir / name, payload)

            try:
                verifier.verify_run(run_dir)
            except verifier.VerificationError as error:
                assert error.classification == "INVALID_ARTIFACT"
                assert "unexpected top-level artifact" in str(error)
            else:
                raise AssertionError(f"run-local {name} was accepted")

            assert not (run_dir / "independent_verification.json").exists()
            assert not (run_dir / "report.md").exists()


def test_second_final_replace_failure_leaves_neither_verifier_output():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _complete_assembled_fixture(temporary)
        (run_dir / "independent_verification.json").write_text(
            '{"stale":true}\n',
            encoding="utf-8",
        )
        (run_dir / "report.md").write_text(
            "# stale verifier report\n",
            encoding="utf-8",
        )
        original_replace = Path.replace
        final_replaces = 0

        def fail_second_final_replace(path, target):
            nonlocal final_replaces
            target = Path(target)
            if target.parent == run_dir and target.name in {
                "independent_verification.json",
                "report.md",
            }:
                final_replaces += 1
                if final_replaces == 2:
                    raise OSError("injected second final replace failure")
            return original_replace(path, target)

        Path.replace = fail_second_final_replace
        try:
            try:
                verifier.verify_run(run_dir)
            except OSError as error:
                assert "second final replace" in str(error)
            else:
                raise AssertionError("second final replace failure was ignored")
        finally:
            Path.replace = original_replace

        assert not (run_dir / "independent_verification.json").exists()
        assert not (run_dir / "report.md").exists()


def test_failed_reverification_removes_stale_verifier_outputs():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _complete_assembled_fixture(temporary)
        first_result = verifier.verify_run(run_dir)
        assert first_result["classification"] == "GO"

        _mutate_json(
            run_dir / "artifact_manifest.json",
            lambda payload: payload["entries"][0].update(
                {"sha256": "0" * 64}
            ),
        )

        try:
            verifier.verify_run(run_dir)
        except verifier.VerificationError as error:
            assert error.classification == "INVALID_ARTIFACT"
        else:
            raise AssertionError("invalid artifact was accepted")

        assert not (run_dir / "independent_verification.json").exists()
        assert not (run_dir / "report.md").exists()


def _mutate_json(path, callback):
    payload = _read_json(path)
    callback(payload)
    _write_json(path, payload)


def _mutate_jsonl(path, callback):
    rows = _read_jsonl(path)
    callback(rows)
    _write_jsonl(path, rows)


def _tamper(run_dir, name):
    if name == "artifact manifest":
        _mutate_json(
            run_dir / "artifact_manifest.json",
            lambda payload: payload["entries"][0].update({"sha256": "0" * 64}),
        )
        return
    if name == "source manifest":
        _mutate_json(
            run_dir / "source_manifest.json",
            lambda payload: payload.update({"source_tree_sha256": "f" * 64}),
        )
    elif name == "prerequisite hash":
        _mutate_jsonl(
            run_dir / "case_rows.jsonl",
            lambda rows: rows[0].update(
                {"correctness_prerequisites_sha256": "f" * 64}
            ),
        )
    elif name == "calibration hash":
        _mutate_jsonl(
            run_dir / "case_rows.jsonl",
            lambda rows: rows[0].update({"calibration_artifact_sha256": "f" * 64}),
        )
    elif name == "P1 authority hash":
        _mutate_jsonl(
            run_dir / "case_rows.jsonl",
            lambda rows: rows[0].update({"p1_authority_artifact_sha256": "f" * 64}),
        )
    elif name == "authorization":
        _mutate_json(
            run_dir / "consumed_authorization.json",
            lambda payload: payload.update({"consumed": False}),
        )
    elif name == "tensor inventory":
        path = next(
            (run_dir / "snapshots").rglob("*.tensor-inventory.json")
        )
        evidence = _read_json(path)
        evidence["storages"][0]["storage_nbytes"] += 1
        _write_canonical_json(path, evidence)
        _refresh_nested_manifest_file(
            run_dir,
            "tensor_inventory_manifest.json",
            path,
        )
    elif name == "snapshot binding":
        manifest = _read_json(run_dir / "snapshot_manifest.json")
        row = next(
            row
            for row in manifest["rows"]
            if row["profile"] != "recompute"
        )
        row["encoded_physical_bytes"] += 1
        _write_canonical_json(run_dir / "snapshot_manifest.json", manifest)
    elif name == "prerequisite evidence":
        manifest = _read_json(run_dir / "correctness_prerequisites.json")
        manifest["rows"][0]["role"] = manifest["rows"][1]["role"]
        _write_canonical_json(
            run_dir / "correctness_prerequisites.json",
            manifest,
        )
    elif name == "token IDs":
        path = next((run_dir / "tokens").glob("*.output.json"))
        values = _read_json(path)
        values[0] += 1
        _write_json(path, values)
        _refresh_nested_manifest_file(run_dir, "token_manifest.json", path)
    elif name == "logit bytes":
        path = next((run_dir / "logits").glob("*.float32.bin"))
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
        _refresh_nested_manifest_file(run_dir, "logits_manifest.json", path)
    elif name == "raw timing":
        def tamper_measured_int8_timing(rows):
            row = next(
                row
                for row in rows
                if row["phase"] == "measured"
                and row["profile"] == contract.P2_PROFILE
                and row["workload"] == "w1_medium_reuse"
            )
            row["ttft_ns"] = row["e2e_ns"]

        _mutate_jsonl(
            run_dir / "case_rows.jsonl",
            tamper_measured_int8_timing,
        )
        return
    elif name == "raw cache bytes":
        _mutate_jsonl(
            run_dir / "process_rows.jsonl",
            lambda rows: rows[0].update(
                {
                    "hybrid_cache_current_unique_physical_bytes": (
                        rows[0]["hybrid_cache_current_unique_physical_bytes"] + 1
                    )
                }
            ),
        )
    elif name == "raw capacity":
        _mutate_jsonl(
            run_dir / "process_rows.jsonl",
            lambda rows: rows[0].update(
                {"same_budget_entry_capacity": rows[0]["same_budget_entry_capacity"] + 1}
            ),
        )
    elif name == "raw CUDA memory":
        _mutate_jsonl(
            run_dir / "process_rows.jsonl",
            lambda rows: rows[0].update(
                {"cuda_peak_reserved_bytes": rows[0]["cuda_peak_reserved_bytes"] + 1}
            ),
        )
    elif name == "safety counters":
        _mutate_jsonl(
            run_dir / "process_rows.jsonl",
            lambda rows: rows[0].update({"oom_events": 1}),
        )
    elif name == "producer summary":
        _mutate_json(
            run_dir / "summary.json",
            lambda payload: payload.update({"producer_claim": "GO"}),
        )
    elif name == "thresholds":
        _mutate_json(
            run_dir / "summary.json",
            lambda payload: payload["thresholds"].update(
                {"w3_int8_to_recompute_throughput_min_ratio": 0.1}
            ),
        )
    else:
        raise AssertionError(f"unknown tamper: {name}")
    _resign_artifact_manifest(run_dir)


def _refresh_nested_manifest_file(run_dir, manifest_name, path):
    manifest = _read_json(run_dir / manifest_name)
    relative = path.relative_to(run_dir).as_posix()
    updated = {
        "path": relative,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "type": "regular_file",
    }
    file_row = next(
        item for item in manifest["files"] if item["path"] == relative
    )
    file_row.update(updated)
    for row in manifest["rows"]:
        for field in ("file", "snapshot_file", "tensor_inventory_file"):
            if (
                isinstance(row.get(field), dict)
                and row[field].get("path") == relative
            ):
                row[field].update(updated)
    _write_canonical_json(run_dir / manifest_name, manifest)
    if manifest_name == "tensor_inventory_manifest.json":
        snapshot_manifest = _read_json(run_dir / "snapshot_manifest.json")
        for row in snapshot_manifest["rows"]:
            if (
                isinstance(row.get("tensor_inventory_file"), dict)
                and row["tensor_inventory_file"].get("path") == relative
            ):
                row["tensor_inventory_file"].update(updated)
        _write_canonical_json(
            run_dir / "snapshot_manifest.json",
            snapshot_manifest,
        )


def test_every_integrity_tamper_is_invalid_artifact_and_publishes_no_outputs():
    tamper_classes = (
        "artifact manifest",
        "source manifest",
        "prerequisite hash",
        "calibration hash",
        "P1 authority hash",
        "authorization",
        "tensor inventory",
        "snapshot binding",
        "prerequisite evidence",
        "token IDs",
        "logit bytes",
        "raw timing",
        "raw cache bytes",
        "raw capacity",
        "raw CUDA memory",
        "safety counters",
        "producer summary",
        "thresholds",
    )
    for tamper_class in tamper_classes:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = _complete_assembled_fixture(temporary)
            _tamper(run_dir, tamper_class)
            try:
                result = verifier.verify_run(run_dir)
            except verifier.VerificationError as error:
                assert error.classification == "INVALID_ARTIFACT", str(error)
            else:
                assert result["classification"] == "INVALID_ARTIFACT", (
                    tamper_class,
                    result,
                )
            assert not (run_dir / "independent_verification.json").exists()
            assert not (run_dir / "report.md").exists()


def test_tolerance_contract_is_exact_and_verifier_uses_torch():
    assert contract.LOGIT_TOLERANCE == {
        "atol": LOGIT_ATOL,
        "rtol": LOGIT_RTOL,
    }
    assert LOGIT_ATOL == 2e-5
    assert LOGIT_RTOL == 0.0
    assert verifier.torch is sys.modules["torch"]
    assert math.isclose(LOGIT_ATOL, 0.00002, rel_tol=0.0, abs_tol=0.0)


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 hybrid-prefix benchmark v2 verifier tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
