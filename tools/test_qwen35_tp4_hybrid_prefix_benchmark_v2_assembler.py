from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
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
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract_for_assembler_test",
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py",
)
contract_fixture = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract_fixture_for_assembler",
    "test_qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py",
)
assembler = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_v2_assembler",
    "qwen35_tp4_hybrid_prefix_benchmark_v2_assembler.py",
)
verifier = _load(
    "verify_qwen35_tp4_hybrid_prefix_benchmark_v2_for_assembler_test",
    "verify_qwen35_tp4_hybrid_prefix_benchmark_v2.py",
)


COMPLETION_MARKER = "QWEN35_TP4_BENCHMARK_V2_WORKER_COMPLETE"


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


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
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


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _complete_raw_bundle(root):
    raw_root = Path(root) / "raw"
    rows_by_case = {}
    for row in contract_fixture._canonical_case_rows():
        copied = copy.deepcopy(row)
        copied["ttft_ns"] = {
            "recompute": 1000,
            "exact_restore": (
                800 if row["workload"] == "w1_medium_reuse" else 700
            ),
            contract.P2_PROFILE: (
                800 if row["workload"] == "w1_medium_reuse" else 700
            ),
        }[row["profile"]]
        if row["workload"] not in {"w1_medium_reuse", "w2_long_reuse"}:
            copied["ttft_ns"] = 1000
        copied["e2e_ns"] = (
            1600
            if row["workload"] == "w3_batched_fanout"
            and row["profile"] != "recompute"
            else 2000
        )
        copied["decode_step_ns"] = 100
        rows_by_case.setdefault(row["case_id"], []).append(copied)
    processes_by_case = {}
    for row in contract_fixture._canonical_process_rows():
        copied = copy.deepcopy(row)
        if row["profile"] == "exact_restore":
            copied.update(
                {
                    "hybrid_cache_current_entries": 1,
                    "hybrid_cache_current_unique_physical_bytes": 1000,
                    "hybrid_cache_current_logical_referenced_bytes": 1000,
                    "hybrid_cache_current_metadata_bytes": 0,
                    "hybrid_cache_deduplicated_bytes": 0,
                    "hybrid_cache_peak_entries": 1,
                    "hybrid_cache_peak_unique_physical_bytes": 1000,
                    "hybrid_cache_peak_logical_referenced_bytes": 1000,
                    "hybrid_cache_peak_metadata_bytes": 0,
                    "same_budget_entry_capacity": 10,
                }
            )
        elif row["profile"] == contract.P2_PROFILE:
            copied.update(
                {
                    "hybrid_cache_current_entries": 1,
                    "hybrid_cache_current_unique_physical_bytes": 400,
                    "hybrid_cache_current_logical_referenced_bytes": 1000,
                    "hybrid_cache_current_metadata_bytes": 40,
                    "hybrid_cache_deduplicated_bytes": 600,
                    "hybrid_cache_peak_entries": 1,
                    "hybrid_cache_peak_unique_physical_bytes": 400,
                    "hybrid_cache_peak_logical_referenced_bytes": 1000,
                    "hybrid_cache_peak_metadata_bytes": 40,
                    "same_budget_entry_capacity": 25,
                }
            )
        processes_by_case.setdefault(row["case_id"], []).append(copied)

    common = raw_root / "common"
    common_documents = {
        "correctness_prerequisites.json": {
            "schema_version": contract.PREREQUISITE_SCHEMA_VERSION,
            "classification": "PASS",
            "artifact_sha256": "1" * 64,
        },
        "calibration_binding.json": contract_fixture._calibration_binding(),
        "p1_authority_binding.json": contract_fixture._p1_authority_binding(),
        "source_manifest.json": contract_fixture._source_manifest(),
        "gate1_audit.json": {
            "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS["gate1_audit"],
            "classification": "PASS",
            "source_tree_sha256": "a" * 64,
        },
        "consumed_authorization.json": {
            "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[
                "consumed_authorization"
            ],
            "classification": "AUTHORIZED",
            "consumed": True,
            "run_tag": "run-1",
            "nonce": "nonce-1",
        },
        "workload_manifest.json": contract.workload_manifest_payload(),
    }
    for name, payload in common_documents.items():
        _write_json(common / name, payload)

    case_matrix = {case.case_id: case for case in contract.build_case_matrix()}
    for case_id, case in case_matrix.items():
        case_dir = (
            raw_root / "profiles" / case.profile / "cases" / case.case_id
        )
        case_rows = rows_by_case[case_id]
        process_rows = processes_by_case[case_id]
        for row in case_rows:
            prompt = list(
                contract.workload_payload(case.workload)[
                    "shared_prefix_token_ids"
                ]
            )
            prompt.extend(
                contract.workload_payload(case.workload)[
                    "continuations"
                ][int(row["request_id"].split("-")[-1])][
                    "suffix_token_ids"
                ]
            )
            output = [
                1000 + index
                for index in range(row["generated_tokens"])
            ]
            prompt_path = (
                case_dir / "tokens" / f"{row['row_id']}.prompt.json"
            )
            output_path = (
                case_dir / "tokens" / f"{row['row_id']}.output.json"
            )
            logits_path = (
                case_dir / "logits" / f"{row['row_id']}.float32.bin"
            )
            _write_json(prompt_path, prompt)
            _write_json(output_path, output)
            logits_path.parent.mkdir(parents=True, exist_ok=True)
            logits_path.write_bytes(b"\x00" * (contract.MODEL_VOCAB_SIZE * 4))
            row.update(
                {
                    "prompt_token_ids_path": (
                        prompt_path.relative_to(case_dir).as_posix()
                    ),
                    "prompt_token_ids_sha256": _sha256(prompt_path),
                    "output_token_ids_path": (
                        output_path.relative_to(case_dir).as_posix()
                    ),
                    "output_token_ids_sha256": _sha256(output_path),
                    "final_logits_path": (
                        logits_path.relative_to(case_dir).as_posix()
                    ),
                    "final_logits_sha256": _sha256(logits_path),
                    "final_logits_shape": [contract.MODEL_VOCAB_SIZE],
                    "final_logits_dtype": "float32",
                }
            )
        _write_jsonl(case_dir / "case_rows.jsonl", case_rows)
        _write_jsonl(case_dir / "process_rows.jsonl", process_rows)
        _write_json(
            case_dir / "summary.json",
            {
                "schema_version": contract.SCHEMA_VERSION,
                "complete": True,
                "case_id": case_id,
                "case_rows": len(case_rows),
                "process_rows": len(process_rows),
            },
        )
        _write_json(
            case_dir / "execution_receipt.json",
            {
                "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[
                    "execution_receipt"
                ],
                "case_id": case_id,
                "run_tag": "run-1",
                "nonce": "nonce-1",
                "complete": True,
            },
        )
        for process_row in process_rows:
            rank = process_row["rank"]
            log = case_dir / "logs" / f"rank-{rank}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(COMPLETION_MARKER + "\n", encoding="utf-8")
            tensor_inventory = (
                case_dir / "tensor-inventories" / f"rank-{rank}.json"
            )
            _write_json(
                tensor_inventory,
                {
                    "schema_version": (
                        "qwen35.tp4-hybrid-prefix-tensor-storage-evidence.v1"
                    ),
                    "case_id": case_id,
                    "profile": case.profile,
                    "rank": rank,
                    "cache": {
                        key: process_row[key]
                        for key in contract.PROCESS_ROW_FIELDS
                        if key.startswith("hybrid_cache_")
                    },
                    "workspace": {
                        key: process_row[key]
                        for key in contract.PROCESS_ROW_FIELDS
                        if "workspace" in key
                    },
                },
            )
    return raw_root


def _upgrade_to_canonical_raw_evidence(raw_root):
    common = raw_root / "common"
    contract_fixture._complete_prerequisite_fixture(common)
    expected_tensor_evidence = {}
    for case in contract.build_case_matrix():
        case_dir = (
            raw_root / "profiles" / case.profile / "cases" / case.case_id
        )
        process_path = case_dir / "process_rows.jsonl"
        process_rows = _read_jsonl(process_path)
        for process_row in process_rows:
            if case.profile == "recompute":
                continue
            rank = process_row["rank"]
            evidence = contract_fixture._canonical_tensor_storage_evidence(
                case.profile,
                case_id=case.case_id,
                rank=rank,
            )
            accounting = contract.recompute_tensor_storage_accounting(evidence)
            process_row.update(accounting)
            snapshot_path = (
                case_dir
                / "snapshots"
                / case.case_id
                / f"rank-{rank}.snapshot"
            )
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_bytes(
                f"canonical-snapshot-{case.case_id}-{rank}\n".encode("utf-8")
            )
            tensor_path = (
                case_dir / "tensor-inventories" / f"rank-{rank}.json"
            )
            _write_json(tensor_path, evidence)
            expected_tensor_evidence[(case.case_id, rank)] = evidence
        _write_jsonl(process_path, process_rows)
    return expected_tensor_evidence


def _canonical_raw_bundle(root):
    raw_root = _complete_raw_bundle(root)
    _upgrade_to_canonical_raw_evidence(raw_root)
    return raw_root


def _load_nested_artifact_evidence(output_dir):
    manifests = {
        kind: json.loads(
            (
                output_dir / contract.NESTED_MANIFEST_ARTIFACT_PATHS[kind]
            ).read_text(encoding="utf-8")
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
    artifact_manifest = json.loads(
        (output_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    return manifests, file_inventory, artifact_manifest


def _assemble(raw_root, output_dir):
    return assembler.assemble_benchmark_run(
        raw_bundle_dir=raw_root,
        output_dir=output_dir,
    )


def _assert_rejected_without_publication(mutator, fragment):
    with tempfile.TemporaryDirectory() as temporary:
        raw_root = _canonical_raw_bundle(temporary)
        output_dir = Path(temporary) / "final-run"
        mutator(raw_root)
        try:
            _assemble(raw_root, output_dir)
        except (ValueError, assembler.AssemblyError) as error:
            assert fragment in str(error).lower(), str(error)
        else:
            raise AssertionError(f"invalid raw bundle accepted: {fragment}")
        assert not output_dir.exists()
        assert not list(Path(temporary).glob(".final-run.*"))


def test_complete_task6_task7_bundle_is_published_atomically():
    with tempfile.TemporaryDirectory() as temporary:
        raw_root = _canonical_raw_bundle(temporary)
        output_dir = Path(temporary) / "final-run"

        result = _assemble(raw_root, output_dir)

        assert result["classification"] == "ASSEMBLED"
        assert output_dir.is_dir()
        assert (output_dir / "artifact_manifest.json").is_file()
        assert not (output_dir / "independent_verification.json").exists()
        assert not (output_dir / "report.md").exists()


def test_real_rank_evidence_publishes_task6_canonical_nested_artifact():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        raw_root = _complete_raw_bundle(root)
        expected_tensor_evidence = _upgrade_to_canonical_raw_evidence(
            raw_root
        )
        output_dir = root / "canonical-run"

        _assemble(raw_root, output_dir)

        case_rows = _read_jsonl(output_dir / "case_rows.jsonl")
        process_rows = _read_jsonl(output_dir / "process_rows.jsonl")
        manifests, file_inventory, artifact_manifest = (
            _load_nested_artifact_evidence(output_dir)
        )
        contract.validate_artifact_evidence(
            case_rows,
            process_rows,
            manifests,
            file_inventory,
            artifact_manifest,
        )
        actual_tensor_evidence = {
            (row["case_id"], row["rank"]): row["evidence"]
            for row in manifests["tensor_inventories"]["rows"]
        }
        assert actual_tensor_evidence == expected_tensor_evidence


def test_legacy_private_evidence_dialect_is_rejected():
    with tempfile.TemporaryDirectory() as temporary:
        raw_root = _complete_raw_bundle(temporary)
        output_dir = Path(temporary) / "legacy-run"

        try:
            _assemble(raw_root, output_dir)
        except (ValueError, assembler.AssemblyError) as error:
            assert "canonical" in str(error).lower()
        else:
            raise AssertionError("legacy private evidence dialect was accepted")
        assert not output_dir.exists()
        assert not list(Path(temporary).glob(".legacy-run.*"))


def test_missing_evidence_classes_prevent_final_directory():
    cases = contract.build_case_matrix()
    mutations = {
        "profile": lambda root: shutil.rmtree(
            root / "profiles" / contract.P2_PROFILE
        ),
        "case": lambda root: shutil.rmtree(
            root
            / "profiles"
            / cases[0].profile
            / "cases"
            / cases[0].case_id
        ),
        "rank": lambda root: _remove_last_jsonl_row(
            root
            / "profiles"
            / cases[0].profile
            / "cases"
            / cases[0].case_id
            / "process_rows.jsonl"
        ),
        "logit": lambda root: next(root.rglob("*.float32.bin")).unlink(),
        "token": lambda root: next(root.rglob("*.prompt.json")).unlink(),
        "cache": lambda root: _drop_process_field(root, "hybrid_cache_hits"),
        "workspace": lambda root: _drop_process_field(
            root, "decode_workspace_peak_reserved_bytes"
        ),
        "receipt": lambda root: next(
            root.rglob("cases/*/execution_receipt.json")
        ).unlink(),
    }
    for evidence_class, mutate in mutations.items():
        _assert_rejected_without_publication(mutate, evidence_class)


def _remove_last_jsonl_row(path):
    rows = _read_jsonl(path)
    _write_jsonl(path, rows[:-1])


def _drop_process_field(root, field):
    path = next(root.rglob("process_rows.jsonl"))
    rows = _read_jsonl(path)
    rows[0].pop(field)
    _write_jsonl(path, rows)


def test_untrusted_or_noncanonical_inputs_leave_no_partial_publication():
    cases = contract.build_case_matrix()
    first_case = (
        Path("profiles")
        / cases[0].profile
        / "cases"
        / cases[0].case_id
    )

    def traceback(root):
        next((root / first_case / "logs").glob("*.log")).write_text(
            "Traceback (most recent call last):\nboom\n",
            encoding="utf-8",
        )

    def unknown(root):
        (root / first_case / "unknown.bin").write_bytes(b"x")

    def symlink(root):
        target = root / first_case / "summary.json"
        (root / first_case / "summary-link.json").symlink_to(target)

    def provenance(root):
        path = root / first_case / "case_rows.jsonl"
        rows = _read_jsonl(path)
        rows[0]["source_tree_sha256"] = "f" * 64
        _write_jsonl(path, rows)

    def duplicate(root):
        path = root / first_case / "case_rows.jsonl"
        rows = _read_jsonl(path)
        _write_jsonl(path, [*rows, copy.deepcopy(rows[0])])

    def producer_classification(root):
        path = root / first_case / "summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["classification"] = "GO"
        _write_json(path, payload)

    for fragment, mutate in (
        ("traceback", traceback),
        ("unknown", unknown),
        ("symlink", symlink),
        ("provenance", provenance),
        ("duplicate", duplicate),
        ("classification", producer_classification),
    ):
        _assert_rejected_without_publication(mutate, fragment)


def test_real_assembler_output_is_accepted_directly_by_real_verifier():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        raw_root = _canonical_raw_bundle(root)
        output_dir = root / "assembled"

        _assemble(raw_root, output_dir)
        result = verifier.verify_run(output_dir)

        assert result["classification"] == "NO_GO_CACHE"
        assert (output_dir / "independent_verification.json").is_file()
        assert (output_dir / "report.md").is_file()


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 hybrid-prefix benchmark v2 assembler tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
