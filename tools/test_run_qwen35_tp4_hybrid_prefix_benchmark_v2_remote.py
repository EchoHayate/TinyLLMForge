from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
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


remote = _load(
    "run_qwen35_tp4_hybrid_prefix_benchmark_v2_remote",
    "run_qwen35_tp4_hybrid_prefix_benchmark_v2_remote.py",
)
contract_fixture = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract_fixture_for_remote",
    "test_qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py",
)


def _sha(character):
    return character * 64


def _write_document(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(remote.contract.canonical_json_bytes(payload) + b"\n")
    return {
        "classification": payload["classification"],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "path": str(path),
    }


def _prerequisites(root=None):
    prerequisites = {
        "correctness_prerequisites": {
            "classification": "PASS",
            "sha256": _sha("1"),
        },
        "calibration": {
            "classification": "PASS",
            "sha256": _sha("2"),
        },
        "p1_authority": {
            "classification": "GO",
            "sha256": _sha("3"),
        },
        "gate1_audit": {
            "classification": "PASS",
            "sha256": _sha("4"),
        },
        "model_manifest_sha256": _sha("5"),
        "workload_manifest_sha256": _sha("6"),
        "source_tree_sha256": _sha("7"),
    }
    for index, field in enumerate(
        remote.contract.EXECUTION_PROVENANCE_FIELDS,
        start=1,
    ):
        prerequisites.setdefault(field, f"{index:064x}")
    prerequisites["model_manifest_sha256"] = (
        remote.contract.MODEL_MANIFEST_SHA256
    )
    if root is None:
        return prerequisites
    authority_root = Path(root) / "prerequisites"
    correctness_path, _ = contract_fixture._complete_prerequisite_fixture(
        authority_root / "correctness"
    )
    prerequisites["correctness_prerequisites"] = {
        "classification": "PASS",
        "sha256": hashlib.sha256(
            correctness_path.read_bytes()
        ).hexdigest(),
        "path": str(correctness_path),
    }
    prerequisites["correctness_prerequisites_sha256"] = prerequisites[
        "correctness_prerequisites"
    ]["sha256"]
    calibration = contract_fixture._calibration_binding()
    calibration_artifact = authority_root / calibration["artifact_path"]
    calibration_artifact.parent.mkdir(parents=True, exist_ok=True)
    calibration_artifact.write_bytes(b"calibration-authority\n")
    calibration.update({
        "source_tree_sha256": prerequisites["source_tree_sha256"],
        "model_manifest_sha256": prerequisites["model_manifest_sha256"],
        "workload_manifest_sha256": prerequisites[
            "workload_manifest_sha256"
        ],
        "artifact_sha256": hashlib.sha256(
            calibration_artifact.read_bytes()
        ).hexdigest(),
    })
    prerequisites["calibration_artifact_sha256"] = calibration[
        "artifact_sha256"
    ]
    prerequisites["calibration"] = _write_document(
        authority_root / "calibration.json",
        calibration,
    )
    p1_authority = contract_fixture._p1_authority_binding()
    p1_artifact = authority_root / p1_authority["artifact_path"]
    p1_artifact.parent.mkdir(parents=True, exist_ok=True)
    p1_artifact.write_bytes(b"p1-authority\n")
    p1_verification = (
        authority_root / p1_authority["independent_verification_path"]
    )
    p1_verification.parent.mkdir(parents=True, exist_ok=True)
    p1_verification.write_bytes(b"p1-independent-verification\n")
    p1_authority.update({
        "source_tree_sha256": prerequisites["source_tree_sha256"],
        "model_manifest_sha256": prerequisites["model_manifest_sha256"],
        "workload_manifest_sha256": prerequisites[
            "workload_manifest_sha256"
        ],
        "artifact_sha256": hashlib.sha256(
            p1_artifact.read_bytes()
        ).hexdigest(),
        "independent_verification_sha256": hashlib.sha256(
            p1_verification.read_bytes()
        ).hexdigest(),
    })
    prerequisites["p1_authority_artifact_sha256"] = p1_authority[
        "artifact_sha256"
    ]
    prerequisites["p1_authority"] = _write_document(
        authority_root / "p1_authority.json",
        p1_authority,
    )
    gate1_audit = contract_fixture._closed_evidence_document("gate1_audit")
    gate1_audit.update({
        "source_tree_sha256": prerequisites["source_tree_sha256"],
        "gate1_audit_sha256": prerequisites["gate1_audit_sha256"],
    })
    prerequisites["gate1_audit"] = _write_document(
        authority_root / "gate1_audit.json",
        gate1_audit,
    )
    return prerequisites


def _gpu_rows(free_bytes=None):
    available = (
        remote.MIN_GPU_FREE_BYTES + 1024
        if free_bytes is None
        else free_bytes
    )
    return [
        {
            "gpu_index": index,
            "gpu_uuid": f"GPU-{index}",
            "free_bytes": available,
            "compute_processes": [],
        }
        for index in remote.REQUIRED_GPU_INDICES
    ]


def _real_source_bundle(root, *, run_tag, nonce, prerequisites):
    output_dir = Path(root)
    source_dir = output_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    member = source_dir / "worker.py"
    member.write_bytes(b"print('worker')\n")
    tar_path = source_dir / "qwen35-v2-source.tar"
    with tarfile.open(tar_path, "w") as archive:
        archive.add(member, arcname="tools/worker.py")
    tar_sha256 = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    prerequisites["source_bundle_sha256"] = tar_sha256
    inventory = [
        {
            "path": "tools/worker.py",
            "sha256": hashlib.sha256(member.read_bytes()).hexdigest(),
            "bytes": member.stat().st_size,
            "type": "file",
        }
    ]
    inventory_path = source_dir / "source_inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    payload = {
        "schema_version": remote.contract.EVIDENCE_SCHEMA_VERSIONS[
            "source_bundle"
        ],
        "run_tag": run_tag,
        "nonce": nonce,
        **{
            field: prerequisites[field]
            for field in remote.contract.EXECUTION_PROVENANCE_FIELDS
        },
        "dirty_tree_policy": "reject_dirty",
        "path": tar_path.relative_to(output_dir).as_posix(),
        "sha256": tar_sha256,
        "inventory_path": inventory_path.relative_to(output_dir).as_posix(),
        "inventory_sha256": remote.contract.canonical_json_sha256(
            inventory
        ),
        "inventory": inventory,
    }
    return payload


def test_remote_identity_and_fixed_resources_are_frozen():
    assert remote.SSH_TARGET == "sitian@10.232.195.203"
    assert remote.REQUIRED_GPU_INDICES == (2, 4, 5, 6)
    assert remote.MIN_GPU_FREE_BYTES == 25769803776
    assert remote.KRB5CCNAME == "FILE:/Users/bytedance/krb5cc_sitian"
    assert remote.SSH_OPTIONS == (
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "ConnectTimeout=20",
    )


def test_blocked_prerequisite_preflight_has_no_side_effects():
    for name, classification in (
        ("correctness_prerequisites", "INVALID_ARTIFACT"),
        ("calibration", "INVALID_ARTIFACT"),
        ("p1_authority", "NO_GO_CACHE"),
        ("gate1_audit", "FAIL"),
    ):
        prerequisites = _prerequisites()
        prerequisites[name]["classification"] = classification
        events = []
        try:
            remote.run_preflight(
                run_tag="v2-preflight-blocked",
                nonce="nonce-blocked",
                prerequisites=prerequisites,
                gpu_query=lambda: events.append("gpu_query"),
                source_bundle_builder=lambda **kwargs: events.append(
                    "source_bundle"
                ),
                remote_path_creator=lambda **kwargs: events.append(
                    "remote_path"
                ),
                process_launcher=lambda **kwargs: events.append("launch"),
            )
        except ValueError as error:
            assert "prerequisite" in str(error).lower(), str(error)
        else:
            raise AssertionError("invalid prerequisites returned ad hoc evidence")
        assert events == []


def test_preflight_rejects_asserted_prerequisite_summaries_without_documents():
    events = []
    try:
        remote.run_preflight(
            run_tag="v2-preflight-fabricated",
            nonce="nonce-fabricated",
            prerequisites=_prerequisites(),
            gpu_query=lambda: events.append("gpu_query") or _gpu_rows(),
            source_bundle_builder=lambda **kwargs: events.append(
                "source_bundle"
            ),
        )
    except ValueError as error:
        assert "prerequisite" in str(error).lower(), str(error)
    else:
        raise AssertionError(
            "asserted prerequisite summaries were treated as authority"
        )
    assert events == []


def test_preflight_rejects_missing_binding_referenced_authority_files():
    with tempfile.TemporaryDirectory() as temporary:
        prerequisites = _prerequisites(temporary)
        calibration_path = Path(prerequisites["calibration"]["path"])
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        referenced = calibration_path.parent / calibration["artifact_path"]
        referenced.unlink()
        events = []
        try:
            remote.run_preflight(
                run_tag="v2-preflight-missing-reference",
                nonce="nonce-missing-reference",
                prerequisites=prerequisites,
                gpu_query=lambda: events.append("gpu_query"),
                source_bundle_builder=lambda **kwargs: events.append(
                    "source_bundle"
                ),
            )
        except ValueError as error:
            assert "artifact" in str(error).lower() or (
                "authority" in str(error).lower()
            ), str(error)
        else:
            raise AssertionError("missing referenced authority was accepted")
        assert events == []


def test_preflight_prerequisite_hash_and_parse_use_same_file_identity():
    with tempfile.TemporaryDirectory() as temporary:
        prerequisites = _prerequisites(temporary)
        gate_path = Path(prerequisites["gate1_audit"]["path"])
        replacement = json.loads(gate_path.read_text(encoding="utf-8"))
        replacement["checks"] += 1
        replacement_path = gate_path.with_name("gate1-replacement.json")
        replacement_path.write_bytes(
            remote.contract.canonical_json_bytes(replacement) + b"\n"
        )
        original_sha256_file = remote._sha256_file
        swapped = False

        def swap_after_hash(path):
            nonlocal swapped
            digest = original_sha256_file(path)
            if Path(path) == gate_path and not swapped:
                os.replace(replacement_path, gate_path)
                swapped = True
            return digest

        remote._sha256_file = swap_after_hash
        try:
            remote._validate_authoritative_prerequisites(prerequisites)
        finally:
            remote._sha256_file = original_sha256_file
        assert hashlib.sha256(gate_path.read_bytes()).hexdigest() == (
            prerequisites["gate1_audit"]["sha256"]
        )


def test_blocked_resource_preflight_queries_only_and_creates_no_path():
    events = []
    rows = _gpu_rows(remote.MIN_GPU_FREE_BYTES - 1)
    with tempfile.TemporaryDirectory() as temporary:
        result = remote.run_preflight(
            run_tag="v2-preflight-resource-blocked",
            nonce="nonce-resource-blocked",
            prerequisites=_prerequisites(temporary),
            gpu_query=lambda: events.append("gpu_query") or rows,
            source_bundle_builder=lambda **kwargs: events.append(
                "source_bundle"
            ),
            remote_path_creator=lambda **kwargs: events.append("remote_path"),
            process_launcher=lambda **kwargs: events.append("launch"),
        )
    assert result["classification"] == "BLOCKED_RESOURCES"
    assert result["gpu_query_rows"] == rows
    assert result["worker_authorized"] is False
    assert result["remote_path_created"] is False
    assert result["source_staged"] is False
    assert result["worker_launched"] is False
    assert events == ["gpu_query"]


def test_ready_preflight_is_read_only_and_source_bound():
    events = []
    with tempfile.TemporaryDirectory() as temporary:
        prerequisites = _prerequisites(temporary)
        source_bundle = _real_source_bundle(
            temporary,
            run_tag="v2-preflight-ready",
            nonce="nonce-ready",
            prerequisites=prerequisites,
        )
        result = remote.run_preflight(
            run_tag="v2-preflight-ready",
            nonce="nonce-ready",
            prerequisites=prerequisites,
            gpu_query=lambda: events.append("gpu_query") or _gpu_rows(),
            source_bundle_builder=lambda **kwargs: (
                events.append(("source_bundle", kwargs))
                or source_bundle
            ),
            source_bundle_output_dir=Path(temporary),
            remote_path_creator=lambda **kwargs: events.append(
                "remote_path"
            ),
            process_launcher=lambda **kwargs: events.append("launch"),
        )
    assert result["classification"] == "READY"
    assert "source_bundle" not in result
    assert result["worker_authorized"] is True
    assert result["remote_path_created"] is False
    assert result["source_staged"] is False
    assert result["worker_launched"] is False
    assert events[0] == "gpu_query"
    assert events[1][0] == "source_bundle"
    assert events[1][1]["output_dir"] == Path(temporary)
    assert len(events) == 2


def test_ready_preflight_is_a_closed_contract_document():
    with tempfile.TemporaryDirectory() as temporary:
        prerequisites = _prerequisites(temporary)
        source_bundle = _real_source_bundle(
            temporary,
            run_tag="v2-preflight-contract",
            nonce="nonce-contract",
            prerequisites=prerequisites,
        )
        result = remote.run_preflight(
            run_tag="v2-preflight-contract",
            nonce="nonce-contract",
            prerequisites=prerequisites,
            gpu_query=_gpu_rows,
            source_bundle_builder=lambda **kwargs: source_bundle,
            source_bundle_output_dir=Path(temporary),
        )
    remote.contract.validate_evidence_document("preflight", result)
    assert result["world_size"] == remote.contract.WORLD_SIZE
    assert result["worker_authorized"] is True
    for field in remote.contract.EXECUTION_PROVENANCE_FIELDS:
        assert result[field] == prerequisites[field]


def test_ready_preflight_rejects_source_bundle_hash_drift():
    with tempfile.TemporaryDirectory() as temporary:
        prerequisites = _prerequisites(temporary)
        source_bundle = {
            "schema_version": remote.contract.EVIDENCE_SCHEMA_VERSIONS[
                "source_bundle"
            ],
            "run_tag": "v2-preflight-source-drift",
            "nonce": "nonce-source-drift",
            **{
                field: prerequisites[field]
                for field in remote.contract.EXECUTION_PROVENANCE_FIELDS
            },
            "dirty_tree_policy": "reject_dirty",
            "path": "source/qwen35-v2-source.tar",
            "sha256": _sha("f"),
            "inventory_path": "source/source_inventory.json",
            "inventory": [
                {
                    "path": "tools/worker.py",
                    "sha256": _sha("a"),
                    "bytes": 1,
                    "type": "file",
                }
            ],
        }
        source_bundle["inventory_sha256"] = (
            remote.contract.canonical_json_sha256(source_bundle["inventory"])
        )
        try:
            remote.run_preflight(
                run_tag="v2-preflight-source-drift",
                nonce="nonce-source-drift",
                prerequisites=prerequisites,
                gpu_query=_gpu_rows,
                source_bundle_builder=lambda **kwargs: source_bundle,
            )
        except ValueError as error:
            assert "source bundle" in str(error).lower(), str(error)
        else:
            raise AssertionError("source bundle hash drift was authorized")


def test_ready_preflight_rejects_nonexistent_source_tar():
    with tempfile.TemporaryDirectory() as temporary:
        prerequisites = _prerequisites(temporary)
        source_bundle = {
            "schema_version": remote.contract.EVIDENCE_SCHEMA_VERSIONS[
                "source_bundle"
            ],
            "run_tag": "v2-preflight-missing-tar",
            "nonce": "nonce-missing-tar",
            **{
                field: prerequisites[field]
                for field in remote.contract.EXECUTION_PROVENANCE_FIELDS
            },
            "dirty_tree_policy": "reject_dirty",
            "path": "source/does-not-exist.tar",
            "sha256": prerequisites["source_bundle_sha256"],
            "inventory_path": "source/does-not-exist.json",
            "inventory_sha256": remote.contract.canonical_json_sha256(
                [
                    {
                        "path": "tools/worker.py",
                        "sha256": _sha("a"),
                        "bytes": 1,
                        "type": "file",
                    }
                ]
            ),
            "inventory": [
                {
                    "path": "tools/worker.py",
                    "sha256": _sha("a"),
                    "bytes": 1,
                    "type": "file",
                }
            ],
        }
        try:
            remote.run_preflight(
                run_tag="v2-preflight-missing-tar",
                nonce="nonce-missing-tar",
                prerequisites=prerequisites,
                gpu_query=_gpu_rows,
                source_bundle_builder=lambda **kwargs: source_bundle,
                source_bundle_output_dir=Path(temporary),
            )
        except ValueError as error:
            assert "source" in str(error).lower(), str(error)
        else:
            raise AssertionError("nonexistent source tar was authorized")


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"qwen35 v2 remote preflight tests passed ({len(tests)} tests)")


if __name__ == "__main__":
    _run()
