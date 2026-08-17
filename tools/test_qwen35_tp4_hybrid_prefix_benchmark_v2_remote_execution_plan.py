from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
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
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract_for_plan_test",
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py",
)
plan_module = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_plan",
    "qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_plan.py",
)
PHYSICAL_ROOTS = Path(tempfile.mkdtemp(prefix="qwen35-v2-plan-roots-"))
(PHYSICAL_ROOTS / "authority").mkdir()
(PHYSICAL_ROOTS / "artifacts").mkdir()


def _sha(character):
    return character * 64


def _preflight():
    return {
        "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS["preflight"],
        "classification": "READY",
        "run_tag": "v2-plan-r1",
        "nonce": "nonce-v2-plan-r1",
        "source_tree_sha256": _sha("1"),
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "workload_manifest_sha256": _sha("3"),
        "correctness_prerequisites_sha256": _sha("4"),
        "calibration_artifact_sha256": _sha("5"),
        "p1_authority_artifact_sha256": _sha("6"),
        "gate1_audit_sha256": _sha("7"),
        "source_bundle_sha256": _sha("8"),
        "source_package_sha256": _sha("9"),
        "producer_source_sha256": _sha("a"),
        "producer_version_sha256": _sha("b"),
        "verifier_source_sha256": _sha("c"),
        "verifier_version_sha256": _sha("d"),
        "command_manifest_sha256": _sha("e"),
        "required_gpu_indices": list(contract.REQUIRED_GPU_INDICES),
        "world_size": contract.WORLD_SIZE,
        "minimum_free_bytes_per_gpu": contract.MIN_GPU_FREE_BYTES,
        "gpu_query_rows": [
            {
                "gpu_index": gpu_index,
                "gpu_uuid": f"GPU-{gpu_index}",
                "free_bytes": contract.MIN_GPU_FREE_BYTES,
                "compute_processes": [],
            }
            for gpu_index in contract.REQUIRED_GPU_INDICES
        ],
        "blocking_reasons": [],
        "worker_authorized": True,
        "remote_path_created": False,
        "source_staged": False,
        "worker_launched": False,
    }


def _port_pairs():
    return [
        {
            "case_id": case.case_id,
            "tinyvllm_dist_port": 24000 + index * 2,
            "master_port": 24001 + index * 2,
        }
        for index, case in enumerate(contract.build_case_matrix())
    ]


def _build_plan(*, preflight=None, ports=None):
    return plan_module.build_remote_execution_plan(
        preflight=_preflight() if preflight is None else preflight,
        case_port_pairs=_port_pairs() if ports is None else ports,
        artifact_root="artifacts/v2-plan-r1",
        authority_root=PHYSICAL_ROOTS / "authority",
        physical_artifact_root=PHYSICAL_ROOTS / "artifacts",
    )


def test_plan_is_deterministic_command_only_and_covers_every_case():
    preflight = _preflight()
    ports = _port_pairs()
    first = _build_plan(preflight=preflight, ports=ports)
    second = _build_plan(
        preflight=copy.deepcopy(preflight),
        ports=copy.deepcopy(ports),
    )
    assert first == second
    assert first["command_order"] == list(contract.EXECUTION_COMMAND_ORDER)
    assert first["case_port_pairs"] == ports
    assert len(first["case_port_pairs"]) == len(contract.build_case_matrix())
    assert first["required_gpu_indices"] == list(
        contract.REQUIRED_GPU_INDICES
    )
    assert first["world_size"] == contract.WORLD_SIZE
    assert not any(
        token in {"kill", "pkill", "killall"}
        for command in contract.canonical_execution_commands(first).values()
        for token in repr(command).split()
    )
    contract.validate_evidence_document("execution_plan", first)


def test_plan_rejects_duplicate_ports_or_non_ready_preflight():
    ports = _port_pairs()
    ports[1]["tinyvllm_dist_port"] = ports[0][
        "tinyvllm_dist_port"
    ]
    try:
        _build_plan(ports=ports)
    except ValueError as error:
        assert "port" in str(error).lower(), str(error)
    else:
        raise AssertionError("duplicate port was accepted")

    blocked = _preflight()
    blocked["classification"] = "BLOCKED_RESOURCES"
    try:
        _build_plan(preflight=blocked)
    except ValueError as error:
        assert "READY" in str(error), str(error)
    else:
        raise AssertionError("blocked preflight was accepted")


def test_plan_verifier_rejects_identity_and_command_tamper():
    plan = _build_plan()
    for field, value in (
        ("source_tree_sha256", _sha("f")),
        ("required_gpu_indices", [0, 1, 2, 3]),
        ("command_order", list(reversed(contract.EXECUTION_COMMAND_ORDER))),
    ):
        changed = copy.deepcopy(plan)
        changed[field] = value
        try:
            plan_module.verify_remote_execution_plan(changed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"tampered {field} was accepted")


def test_plan_binds_physical_authority_and_artifact_roots():
    with tempfile.TemporaryDirectory() as temporary:
        authority_root = Path(temporary) / "authority"
        physical_artifact_root = Path(temporary) / "artifacts"
        authority_root.mkdir()
        physical_artifact_root.mkdir()
        plan = plan_module.build_remote_execution_plan(
            preflight=_preflight(),
            case_port_pairs=_port_pairs(),
            artifact_root="artifacts/v2-plan-r1",
            authority_root=authority_root,
            physical_artifact_root=physical_artifact_root,
        )
        expected_authority = contract.physical_directory_sha256(
            authority_root
        )
        expected_artifact = contract.physical_directory_sha256(
            physical_artifact_root
        )
    assert plan["authority_root_sha256"] == expected_authority
    assert plan["physical_artifact_root_sha256"] == expected_artifact
    plan_module.verify_remote_execution_plan(plan)


def test_plan_physical_root_identity_changes_after_directory_replacement():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        authority_root = root / "authority"
        physical_artifact_root = root / "artifacts"
        authority_root.mkdir()
        physical_artifact_root.mkdir()
        first = plan_module.build_remote_execution_plan(
            preflight=_preflight(),
            case_port_pairs=_port_pairs(),
            artifact_root="artifacts/v2-plan-r1",
            authority_root=authority_root,
            physical_artifact_root=physical_artifact_root,
        )
        authority_root.rename(root / "authority-old")
        physical_artifact_root.rename(root / "artifacts-old")
        authority_root.mkdir()
        physical_artifact_root.mkdir()
        second = plan_module.build_remote_execution_plan(
            preflight=_preflight(),
            case_port_pairs=_port_pairs(),
            artifact_root="artifacts/v2-plan-r1",
            authority_root=authority_root,
            physical_artifact_root=physical_artifact_root,
        )

    assert first["authority_root_sha256"] != (
        second["authority_root_sha256"]
    )
    assert first["physical_artifact_root_sha256"] != (
        second["physical_artifact_root_sha256"]
    )


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"qwen35 v2 remote plan tests passed ({len(tests)} tests)")


if __name__ == "__main__":
    _run()
