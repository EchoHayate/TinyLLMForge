from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
AUTHORIZATION_TEXT = (
    "允许只运行一个 source-bound focused-H2D four-cell campaign"
)


def _load(name: str, filename: str):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


source_bundle = _load(
    "focused_h2d_source_bundle",
    "qwen35_tp4_32k_h2d_slot_reuse_source_bundle.py",
)
campaign = _load(
    "focused_h2d_campaign",
    "qwen35_tp4_32k_h2d_slot_reuse_campaign.py",
)
authorization = _load(
    "focused_h2d_campaign_authorization",
    "qwen35_tp4_32k_h2d_slot_reuse_campaign_authorization.py",
)
executor = _load(
    "focused_h2d_campaign_executor",
    "qwen35_tp4_32k_h2d_slot_reuse_campaign_executor.py",
)


def _checkpoint_manifest(path: Path) -> str:
    path.write_text(
        json.dumps(
            {
                "schema": "focused-h2d-test-checkpoint.v1",
                "model": "Qwen3.5-test",
                "files": [{"path": "model.safetensors", "sha256": "a" * 64}],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return source_bundle.sha256_file(path)


def _prepared_campaign(root: Path, *, repetitions_per_cell: int = 2):
    checkpoint_manifest = root / "checkpoint_manifest.json"
    checkpoint_sha256 = _checkpoint_manifest(checkpoint_manifest)
    prepared = campaign.prepare_local_campaign(
        repo_root=ROOT,
        output_dir=root / "prepared",
        run_tag="focused-h2d-local-r1",
        checkpoint_manifest_path=checkpoint_manifest,
        remote_python=(
            "/data00/home/sitian/miniconda3/envs/py311/bin/python"
        ),
        remote_model_dir=(
            "/dev/shm/sitian/"
            "tllm-qwen35-target-qwen3-draft-20260815/target"
        ),
        repetitions_per_cell=repetitions_per_cell,
        gpu_indices=(0, 1, 2, 3),
        dist_port=29681,
        master_port=29781,
    )
    return prepared, checkpoint_sha256


def test_source_bundle_is_deterministic_and_covers_dynamic_producer_closure():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = source_bundle.build_source_bundle(
            repo_root=ROOT,
            output_dir=root / "first",
        )
        second = source_bundle.build_source_bundle(
            repo_root=ROOT,
            output_dir=root / "second",
        )
        assert first["source_tree_sha256"] == second["source_tree_sha256"]
        assert first["source_tar_sha256"] == second["source_tar_sha256"]
        assert first["files"] == second["files"]
        owned = [row["path"] for row in first["files"]]
        assert owned == sorted(owned)
        assert len(owned) == len(set(owned))
        required = {
            "tools/qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic_gate.py",
            "tools/qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic_worker.py",
            "tools/verify_qwen35_tp4_32k_h2d_slot_reuse_causal_diagnostic.py",
            "tools/qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py",
            "tools/qwen35_native_mtp_tp4_16k_target_kv_offload_worker.py",
            "tools/qwen35_native_mtp_tp4_4k_engine_gate.py",
            "tools/qwen35_tp4_32k_h2d_slot_reuse_source_bundle.py",
            "tools/qwen35_tp4_32k_h2d_slot_reuse_campaign.py",
            (
                "tools/qwen35_tp4_32k_h2d_slot_reuse_"
                "campaign_authorization.py"
            ),
            (
                "tools/qwen35_tp4_32k_h2d_slot_reuse_"
                "campaign_executor.py"
            ),
        }
        assert required.issubset(owned)
        expected_tinyvllm = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "tinyvllm").rglob("*.py")
            if path.is_file() and not path.is_symlink()
        }
        assert expected_tinyvllm.issubset(owned)
        with tarfile.open(first["source_tar"], "r:") as archive:
            members = archive.getmembers()
        assert [member.name for member in members] == owned
        assert all(
            member.isfile() and not member.issym() and not member.islnk()
            for member in members
        )
        assert source_bundle.validate_source_bundle(
            inventory_path=first["source_inventory"],
            tar_path=first["source_tar"],
        )["source_tree_sha256"] == first["source_tree_sha256"]


def test_source_bundle_rejects_symlinked_required_source():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "tinyvllm").mkdir()
        (root / "tools").mkdir()
        target = root / "real.py"
        target.write_text("VALUE = 1\n", encoding="utf-8")
        (root / "tinyvllm" / "__init__.py").symlink_to(target)
        with pytest.raises(ValueError, match="symlink"):
            source_bundle.collect_source_files(root)


def test_campaign_plan_freezes_source_workload_and_non_execution_boundary():
    with tempfile.TemporaryDirectory() as temporary:
        prepared, checkpoint_sha256 = _prepared_campaign(Path(temporary))
        plan = campaign.validate_campaign_plan(prepared["plan"])
        assert plan["schema"] == (
            "qwen35.tp4-32k-h2d-source-bound-campaign-plan.v1"
        )
        assert plan["authorization_text"] == AUTHORIZATION_TEXT
        assert plan["ssh_target"] == "sitian@10.232.195.203"
        assert plan["remote_root"] == (
            "/dev/shm/sitian/"
            "tllm-qwen35-target-qwen3-draft-20260815"
        )
        assert plan["cells"] == [
            "observe:b1",
            "observe:b4",
            "control:b1",
            "control:b4",
        ]
        assert plan["repetitions_per_cell"] == 2
        assert plan["gpu_indices"] == [0, 1, 2, 3]
        assert plan["ports"] == {
            "dist_port": 29681,
            "master_port": 29781,
        }
        assert plan["checkpoint_manifest_sha256"] == checkpoint_sha256
        assert plan["source_tree_sha256"] == prepared[
            "source_bundle"
        ]["source_tree_sha256"]
        assert plan["source_tar_sha256"] == prepared[
            "source_bundle"
        ]["source_tar_sha256"]
        assert plan["execution_boundary"] == {
            "ssh_authorized": False,
            "remote_write_authorized": False,
            "gpu_authorized": False,
            "cuda_authorized": False,
            "nccl_authorized": False,
            "campaign_authorized": False,
        }
        assert plan["commands"] == {
            "worker_argv": [
                "python",
                "tools/qwen35_tp4_32k_h2d_slot_reuse_"
                "causal_diagnostic_worker.py",
            ],
            "verifier_argv": [
                "python",
                "tools/verify_qwen35_tp4_32k_h2d_slot_reuse_"
                "causal_diagnostic.py",
            ],
        }
        assert not any(
            callable(value)
            for value in plan["commands"].values()
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("authorization_text", "批准"),
        ("ssh_target", "other@10.232.195.203"),
        ("remote_root", "/data00/new-experiment"),
        ("cells", ["observe:b1"]),
        ("repetitions_per_cell", 0),
        ("gpu_indices", [0, 1, 2, 4]),
        ("source_tree_sha256", "0" * 64),
        ("checkpoint_manifest_sha256", "0" * 64),
        ("commands", {"worker_argv": ["ssh"]}),
        (
            "execution_boundary",
            {
                "ssh_authorized": True,
                "remote_write_authorized": False,
                "gpu_authorized": False,
                "cuda_authorized": False,
                "nccl_authorized": False,
                "campaign_authorized": False,
            },
        ),
    ),
)
def test_campaign_plan_rejects_tampering(field, value):
    with tempfile.TemporaryDirectory() as temporary:
        prepared, _ = _prepared_campaign(Path(temporary))
        changed = copy.deepcopy(prepared["plan"])
        changed[field] = value
        with pytest.raises(ValueError):
            campaign.validate_campaign_plan(changed)


def test_campaign_module_has_no_execution_capable_import_or_call():
    source = (
        TOOLS / "qwen35_tp4_32k_h2d_slot_reuse_campaign.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert not imported.intersection(
        {"subprocess", "socket", "paramiko", "torch"}
    )
    call_names = {
        (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else node.func.id
            if isinstance(node.func, ast.Name)
            else ""
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not call_names.intersection(
        {"run", "Popen", "system", "execv", "execve"}
    )


def test_campaign_prepare_cli_routes_frozen_local_inputs(capsys):
    calls = []

    def prepare_fn(**kwargs):
        calls.append(kwargs)
        return {
            "plan_path": "/tmp/prepared/campaign_plan.json",
            "plan_sha256": "a" * 64,
            "source_bundle": {
                "source_tree_sha256": "b" * 64,
                "source_tar_sha256": "c" * 64,
            },
        }

    result = campaign.main(
        [
            "prepare",
            "--repo-root",
            "/repo",
            "--output-dir",
            "/out",
            "--run-tag",
            "focused-h2d-r1",
            "--checkpoint-manifest",
            "/model/model_manifest.json",
            "--remote-python",
            "/data00/python",
            "--remote-model-dir",
            "/data00/model",
            "--repetitions",
            "3",
            "--dist-port",
            "29681",
            "--master-port",
            "29781",
        ],
        prepare_fn=prepare_fn,
    )
    assert result == 0
    assert calls == [{
        "repo_root": Path("/repo"),
        "output_dir": Path("/out"),
        "run_tag": "focused-h2d-r1",
        "checkpoint_manifest_path": Path(
            "/model/model_manifest.json"
        ),
        "remote_python": "/data00/python",
        "remote_model_dir": "/data00/model",
        "repetitions_per_cell": 3,
        "gpu_indices": (0, 1, 2, 3),
        "dist_port": 29681,
        "master_port": 29781,
    }]
    assert json.loads(capsys.readouterr().out) == {
        "classification": "PREPARED_LOCAL_ONLY",
        "plan_path": "/tmp/prepared/campaign_plan.json",
        "plan_sha256": "a" * 64,
        "source_tree_sha256": "b" * 64,
        "source_tar_sha256": "c" * 64,
    }


def test_campaign_validate_cli_does_not_prepare_or_authorize(capsys):
    calls = []

    def validate_fn(plan):
        calls.append(plan)
        return plan

    with tempfile.TemporaryDirectory() as temporary:
        plan_path = Path(temporary) / "plan.json"
        plan_path.write_text('{"schema":"test"}\n', encoding="utf-8")
        result = campaign.main(
            ["validate", "--plan", str(plan_path)],
            validate_fn=validate_fn,
        )
    assert result == 0
    assert calls == [{"schema": "test"}]
    assert json.loads(capsys.readouterr().out) == {
        "classification": "VALID_LOCAL_PLAN",
        "plan_sha256": campaign.canonical_sha256(
            {"schema": "test"}
        ),
    }


def test_authorization_is_exact_plan_bound_and_single_use():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        prepared, _ = _prepared_campaign(root)
        plan = prepared["plan"]
        active = root / "authorization.json"
        consumed = root / "authorization.consumed.json"
        payload = authorization.produce_authorization(
            plan=plan,
            authorization_text=AUTHORIZATION_TEXT,
            nonce="focused-h2d-r1",
            output_path=active,
        )
        assert payload["classification"] == "AUTHORIZED"
        assert payload["plan_sha256"] == campaign.canonical_sha256(plan)
        assert payload["source_tree_sha256"] == plan[
            "source_tree_sha256"
        ]
        assert payload["source_tar_sha256"] == plan["source_tar_sha256"]
        assert payload["cells"] == plan["cells"]
        assert payload["repetitions_per_cell"] == 2
        result = authorization.consume_authorization(
            plan=plan,
            authorization_path=active,
            consumed_path=consumed,
        )
        assert result["consumed"] is True
        assert not active.exists()
        assert json.loads(consumed.read_text(encoding="utf-8")) == result
        with pytest.raises(ValueError, match="authorization"):
            authorization.consume_authorization(
                plan=plan,
                authorization_path=active,
                consumed_path=root / "again.json",
            )


def test_authorization_rejects_wrong_text_unsafe_nonce_and_tampering():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        prepared, _ = _prepared_campaign(root)
        plan = prepared["plan"]
        for text in ("批准", AUTHORIZATION_TEXT + " replication"):
            with pytest.raises(ValueError, match="authorization text"):
                authorization.produce_authorization(
                    plan=plan,
                    authorization_text=text,
                    nonce="focused-h2d-r1",
                    output_path=root / f"wrong-{len(text)}.json",
                )
        for nonce in ("", "../escape", "with space", "semi;colon"):
            with pytest.raises(ValueError, match="nonce"):
                authorization.produce_authorization(
                    plan=plan,
                    authorization_text=AUTHORIZATION_TEXT,
                    nonce=nonce,
                    output_path=root / f"nonce-{len(nonce)}.json",
                )
        payload = authorization.build_authorization(
            plan=plan,
            authorization_text=AUTHORIZATION_TEXT,
            nonce="focused-h2d-r2",
        )
        for field, value in (
            ("plan_sha256", "0" * 64),
            ("source_tree_sha256", "0" * 64),
            ("cells", ["observe:b1"]),
            ("gpu_indices", [4, 5, 6, 7]),
            ("consumed", True),
        ):
            changed = copy.deepcopy(payload)
            changed[field] = value
            with pytest.raises(ValueError, match="authorization"):
                authorization.validate_authorization(plan, changed)


def test_executor_consumes_authorization_before_invoking_injected_runner():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        prepared, _ = _prepared_campaign(root)
        plan = prepared["plan"]
        active = root / "authorization.json"
        consumed = root / "runtime" / "consumed.json"
        authorization.produce_authorization(
            plan=plan,
            authorization_text=AUTHORIZATION_TEXT,
            nonce="focused-h2d-execute-r1",
            output_path=active,
        )
        calls = []

        def command_runner(received_plan, authorization_record):
            assert received_plan == plan
            assert not active.exists()
            assert consumed.is_file()
            assert authorization_record["consumed"] is True
            calls.append(authorization_record["nonce"])
            return {
                "classification": "DRY_CALLBACK_COMPLETE",
                "remote_command_count": 0,
            }

        result = executor.execute_authorized_campaign(
            plan=plan,
            authorization_path=active,
            consumed_authorization_path=consumed,
            command_runner=command_runner,
        )
        assert calls == ["focused-h2d-execute-r1"]
        assert result == {
            "schema": (
                "qwen35.tp4-32k-h2d-authorized-executor-result.v1"
            ),
            "classification": "DRY_CALLBACK_COMPLETE",
            "plan_sha256": campaign.canonical_sha256(plan),
            "authorization_nonce": "focused-h2d-execute-r1",
            "remote_command_count": 0,
        }


def test_executor_fails_closed_before_callback_for_invalid_authorization():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        prepared, _ = _prepared_campaign(root)
        plan = prepared["plan"]
        active = root / "authorization.json"
        active.write_text("{}\n", encoding="utf-8")
        calls = []
        with pytest.raises(ValueError, match="authorization"):
            executor.execute_authorized_campaign(
                plan=plan,
                authorization_path=active,
                consumed_authorization_path=(
                    root / "runtime" / "consumed.json"
                ),
                command_runner=lambda *args: calls.append(args),
            )
        assert calls == []
        assert active.is_file()


def test_executor_does_not_restore_consumed_authorization_after_callback_error():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        prepared, _ = _prepared_campaign(root)
        plan = prepared["plan"]
        active = root / "authorization.json"
        consumed = root / "runtime" / "consumed.json"
        authorization.produce_authorization(
            plan=plan,
            authorization_text=AUTHORIZATION_TEXT,
            nonce="focused-h2d-failing-callback-r1",
            output_path=active,
        )

        def fail_after_consumption(*_args):
            raise RuntimeError("injected runner failed")

        with pytest.raises(RuntimeError, match="injected runner failed"):
            executor.execute_authorized_campaign(
                plan=plan,
                authorization_path=active,
                consumed_authorization_path=consumed,
                command_runner=fail_after_consumption,
            )
        assert not active.exists()
        assert json.loads(
            consumed.read_text(encoding="utf-8")
        )["consumed"] is True


def test_executor_has_no_builtin_transport_or_gpu_import():
    source = (
        TOOLS / "qwen35_tp4_32k_h2d_slot_reuse_campaign_executor.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert not imported.intersection(
        {"subprocess", "socket", "paramiko", "torch"}
    )
