from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
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
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract_for_auth_test",
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py",
)
authorization = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_authorization",
    "qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_authorization.py",
)


def _sha(character):
    return character * 64


def _plan():
    return {
        "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS["execution_plan"],
        "run_tag": "v2-auth-r1",
        "nonce": "nonce-v2-auth-r1",
        **{
            field: _sha(hex(index % 16)[2:])
            for index, field in enumerate(
                contract.EXECUTION_PROVENANCE_FIELDS, start=1
            )
        },
        "required_gpu_indices": list(contract.REQUIRED_GPU_INDICES),
        "world_size": contract.WORLD_SIZE,
        "gpu_assignments": [
            {
                "rank": rank,
                "gpu_index": gpu,
                "cuda_visible_device": str(rank),
            }
            for rank, gpu in enumerate(contract.REQUIRED_GPU_INDICES)
        ],
        "case_port_pairs": [
            {
                "case_id": case.case_id,
                "tinyvllm_dist_port": 25000 + index * 2,
                "master_port": 25001 + index * 2,
            }
            for index, case in enumerate(contract.build_case_matrix())
        ],
        "artifact_paths": {
            "remote_run": "remote/v2-auth-r1/run",
            "remote_artifact": "remote/v2-auth-r1/run/artifact",
            "package": "local/v2-auth-r1/package.tar",
            "local_extract": "local/v2-auth-r1/run",
        },
        "command_order": list(contract.EXECUTION_COMMAND_ORDER),
        "authority_root_sha256": _sha("e"),
        "physical_artifact_root_sha256": _sha("f"),
    }


def _bind_authority_root(plan, root):
    Path(root).mkdir(parents=True, exist_ok=True)
    plan["authority_root_sha256"] = contract.physical_directory_sha256(root)
    return plan


def test_authorization_binds_plan_nonce_ports_gpus_and_provenance():
    plan = _plan()
    payload = authorization.build_authorization(
        plan=plan,
        authorization_id="auth-v2-r1",
        active_path="authority/authorization.active.json",
        consumed_path="authority/authorization.consumed.json",
    )
    authorization.validate_authorization(plan=plan, payload=payload)
    assert payload["execution_plan_sha256"] == (
        contract.canonical_json_sha256(plan)
    )
    assert payload["nonce"] == plan["nonce"]
    assert payload["case_port_pairs"] == plan["case_port_pairs"]
    assert payload["gpu_assignments"] == plan["gpu_assignments"]
    assert payload["consumed"] is False
    assert payload["consumed_once"] is False


def test_authorization_rejects_plan_identity_or_nonce_tamper():
    plan = _plan()
    payload = authorization.build_authorization(
        plan=plan,
        authorization_id="auth-v2-r1",
        active_path="authority/authorization.active.json",
        consumed_path="authority/authorization.consumed.json",
    )
    for field, value in (
        ("nonce", "different"),
        ("source_tree_sha256", _sha("f")),
        ("case_port_pairs", payload["case_port_pairs"][:-1]),
    ):
        changed = copy.deepcopy(payload)
        changed[field] = value
        try:
            authorization.validate_authorization(
                plan=plan,
                payload=changed,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"tampered {field} was accepted")


def test_authorization_consumption_is_atomic_and_single_use():
    plan = _plan()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _bind_authority_root(plan, root)
        active = root / "authority" / "authorization.active.json"
        consumed = root / "authority" / "authorization.consumed.json"
        authorization.produce_authorization(
            plan=plan,
            output_path=active,
            authorization_id="auth-v2-r1",
            consumed_path=consumed,
            authority_root=root,
        )
        record = authorization.consume_authorization(
            plan=plan,
            active_path=active,
            consumed_path=consumed,
            active_record_path="authority/authorization.active.json",
            consumed_record_path=(
                "authority/authorization.consumed.json"
            ),
            authority_root=root,
        )
        assert not active.exists()
        assert consumed.exists()
        assert record["consumed"] is True
        assert record["consumed_once"] is True
        contract.validate_evidence_document(
            "consumed_authorization",
            record,
        )
        assert record["active_path"] == (
            "authority/authorization.active.json"
        )
        assert record["consumed_path"] == (
            "authority/authorization.consumed.json"
        )
        assert json.loads(consumed.read_text(encoding="utf-8")) == record
        try:
            authorization.consume_authorization(
                plan=plan,
                active_path=active,
                consumed_path=consumed,
                active_record_path=(
                    "authority/authorization.active.json"
                ),
                consumed_record_path=(
                    "authority/authorization.consumed.json"
                ),
                authority_root=root,
            )
        except ValueError as error:
            assert "consum" in str(error).lower(), str(error)
        else:
            raise AssertionError("authorization was consumed twice")


def test_authorization_cannot_change_bound_consumed_record_path():
    plan = _plan()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _bind_authority_root(plan, root)
        active = root / "authority" / "authorization.active.json"
        consumed = root / "authority" / "authorization.consumed.json"
        authorization.produce_authorization(
            plan=plan,
            output_path=active,
            authorization_id="auth-v2-r1",
            consumed_path="authority/authorization.consumed.json",
            authority_root=root,
        )
        try:
            authorization.consume_authorization(
                plan=plan,
                active_path=active,
                consumed_path=consumed,
                active_record_path="authority/authorization.active.json",
                consumed_record_path="authority/other.consumed.json",
                authority_root=root,
            )
        except ValueError as error:
            assert "path" in str(error).lower(), str(error)
        else:
            raise AssertionError("authorization destination binding changed")
        assert active.exists()
        assert not consumed.exists()


def test_authorization_recovers_existing_consuming_claim():
    plan = _plan()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _bind_authority_root(plan, root)
        active = root / "authority" / "authorization.active.json"
        consumed = root / "authority" / "authorization.consumed.json"
        authorization.produce_authorization(
            plan=plan,
            output_path=active,
            authorization_id="auth-v2-r1",
            consumed_path="authority/authorization.consumed.json",
            authority_root=root,
        )
        claim = active.with_name(f".{active.name}.consuming")
        active.replace(claim)
        record = authorization.consume_authorization(
            plan=plan,
            active_path=active,
            consumed_path=consumed,
            active_record_path="authority/authorization.active.json",
            consumed_record_path=(
                "authority/authorization.consumed.json"
            ),
            authority_root=root,
        )
        assert record["consumed"] is True
        assert consumed.exists()
        assert not claim.exists()


def test_copied_authorization_outside_bound_root_cannot_be_consumed():
    plan = _plan()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "canonical"
        copied_root = Path(temporary) / "copied"
        active = root / "authority" / "authorization.active.json"
        consumed = root / "authority" / "authorization.consumed.json"
        copied_active = (
            copied_root / "authority" / "authorization.active.json"
        )
        copied_consumed = (
            copied_root / "authority" / "authorization.consumed.json"
        )
        _bind_authority_root(plan, root)
        authorization.produce_authorization(
            plan=plan,
            output_path=active,
            authorization_id="auth-v2-r1",
            consumed_path="authority/authorization.consumed.json",
            authority_root=root,
        )
        copied_active.parent.mkdir(parents=True)
        copied_active.write_bytes(active.read_bytes())
        authorization.consume_authorization(
            plan=plan,
            active_path=active,
            consumed_path=consumed,
            active_record_path="authority/authorization.active.json",
            consumed_record_path=(
                "authority/authorization.consumed.json"
            ),
            authority_root=root,
        )
        try:
            authorization.consume_authorization(
                plan=plan,
                active_path=copied_active,
                consumed_path=copied_consumed,
                active_record_path=(
                    "authority/authorization.active.json"
                ),
                consumed_record_path=(
                    "authority/authorization.consumed.json"
                ),
                authority_root=root,
            )
        except ValueError as error:
            assert "root" in str(error).lower() or (
                "path" in str(error).lower()
            ), str(error)
        else:
            raise AssertionError("copied authorization was consumed twice")
        assert not copied_consumed.exists()


def test_copied_authorization_cannot_select_a_new_authority_root():
    plan = _plan()
    with tempfile.TemporaryDirectory() as temporary:
        canonical_root = Path(temporary) / "canonical"
        copied_root = Path(temporary) / "copied"
        active = canonical_root / "authority" / "authorization.active.json"
        copied_active = (
            copied_root / "authority" / "authorization.active.json"
        )
        copied_consumed = (
            copied_root / "authority" / "authorization.consumed.json"
        )
        _bind_authority_root(plan, canonical_root)
        authorization.produce_authorization(
            plan=plan,
            output_path=active,
            authorization_id="auth-v2-r1",
            consumed_path="authority/authorization.consumed.json",
            authority_root=canonical_root,
        )
        copied_active.parent.mkdir(parents=True)
        copied_active.write_bytes(active.read_bytes())
        try:
            authorization.consume_authorization(
                plan=plan,
                active_path=copied_active,
                consumed_path=copied_consumed,
                active_record_path=(
                    "authority/authorization.active.json"
                ),
                consumed_record_path=(
                    "authority/authorization.consumed.json"
                ),
                authority_root=copied_root,
            )
        except ValueError as error:
            assert "root" in str(error).lower() or (
                "identity" in str(error).lower()
            ), str(error)
        else:
            raise AssertionError(
                "copied authorization selected a second authority root"
            )
        assert not copied_consumed.exists()


def test_copied_authorization_cannot_rewrite_root_suffix_and_double_consume():
    plan = _plan()
    with tempfile.TemporaryDirectory() as temporary:
        canonical_root = Path(temporary) / "canonical"
        copied_root = Path(temporary) / "copied"
        active = canonical_root / "authority" / "authorization.active.json"
        consumed = (
            canonical_root / "authority" / "authorization.consumed.json"
        )
        copied_active = (
            copied_root / "authority" / "authorization.active.json"
        )
        copied_consumed = (
            copied_root / "authority" / "authorization.consumed.json"
        )
        _bind_authority_root(plan, canonical_root)
        payload = authorization.produce_authorization(
            plan=plan,
            output_path=active,
            authorization_id="auth-v2-r1",
            consumed_path="authority/authorization.consumed.json",
            authority_root=canonical_root,
        )
        copied_payload = copy.deepcopy(payload)
        copied_root.mkdir(parents=True)
        copied_digest = contract.physical_directory_sha256(copied_root)
        copied_payload["authorization_id"] = (
            copied_payload["authorization_id"].rsplit("-root-", 1)[0]
            + f"-root-{copied_digest}"
        )
        copied_active.parent.mkdir(parents=True, exist_ok=True)
        copied_active.write_bytes(
            contract.canonical_json_bytes(copied_payload) + b"\n"
        )
        authorization.consume_authorization(
            plan=plan,
            active_path=active,
            consumed_path=consumed,
            active_record_path="authority/authorization.active.json",
            consumed_record_path=(
                "authority/authorization.consumed.json"
            ),
            authority_root=canonical_root,
        )
        try:
            authorization.consume_authorization(
                plan=plan,
                active_path=copied_active,
                consumed_path=copied_consumed,
                active_record_path=(
                    "authority/authorization.active.json"
                ),
                consumed_record_path=(
                    "authority/authorization.consumed.json"
                ),
                authority_root=copied_root,
            )
        except ValueError as error:
            assert "root" in str(error).lower() or (
                "plan" in str(error).lower()
            ), str(error)
        else:
            raise AssertionError(
                "rewritten copied authorization was consumed twice"
            )
        assert consumed.exists()
        assert not copied_consumed.exists()


def test_authorization_produce_rejects_intermediate_directory_symlink():
    plan = _plan()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "canonical"
        outside = Path(temporary) / "outside"
        root.mkdir()
        outside.mkdir()
        (root / "authority").symlink_to(
            outside,
            target_is_directory=True,
        )
        _bind_authority_root(plan, root)
        active = root / "authority" / "authorization.active.json"
        consumed = root / "authority" / "authorization.consumed.json"

        try:
            authorization.produce_authorization(
                plan=plan,
                output_path=active,
                authorization_id="auth-v2-r1",
                consumed_path=consumed,
                authority_root=root,
            )
        except ValueError as error:
            assert "root" in str(error).lower() or (
                "symlink" in str(error).lower()
            ), str(error)
        else:
            raise AssertionError(
                "intermediate authority symlink escaped the bound root"
            )
        assert not (outside / active.name).exists()


def test_authorization_consume_rejects_intermediate_directory_symlink():
    plan = _plan()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "canonical"
        outside = Path(temporary) / "outside"
        root.mkdir()
        outside.mkdir()
        (root / "authority").symlink_to(
            outside,
            target_is_directory=True,
        )
        _bind_authority_root(plan, root)
        active = root / "authority" / "authorization.active.json"
        consumed = root / "authority" / "authorization.consumed.json"
        payload = authorization.build_authorization(
            plan=plan,
            authorization_id="auth-v2-r1",
            active_path="authority/authorization.active.json",
            consumed_path="authority/authorization.consumed.json",
        )
        (outside / active.name).write_bytes(
            contract.canonical_json_bytes(payload) + b"\n"
        )

        try:
            authorization.consume_authorization(
                plan=plan,
                active_path=active,
                consumed_path=consumed,
                active_record_path="authority/authorization.active.json",
                consumed_record_path=(
                    "authority/authorization.consumed.json"
                ),
                authority_root=root,
            )
        except ValueError as error:
            assert "root" in str(error).lower() or (
                "symlink" in str(error).lower()
            ), str(error)
        else:
            raise AssertionError(
                "intermediate authority symlink was consumed outside root"
            )
        assert not (outside / consumed.name).exists()


def test_authorization_rejects_same_path_root_directory_replacement():
    plan = _plan()
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        root = base / "canonical"
        root.mkdir()
        _bind_authority_root(plan, root)
        root.rename(base / "canonical-old")
        root.mkdir()
        active = root / "authority" / "authorization.active.json"
        consumed = root / "authority" / "authorization.consumed.json"

        try:
            authorization.produce_authorization(
                plan=plan,
                output_path=active,
                authorization_id="auth-v2-r1",
                consumed_path=consumed,
                authority_root=root,
            )
        except ValueError as error:
            assert "root" in str(error).lower() or (
                "identity" in str(error).lower()
            ), str(error)
        else:
            raise AssertionError(
                "same-path replacement preserved authority root identity"
            )
        assert not active.exists()


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 v2 remote authorization tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
