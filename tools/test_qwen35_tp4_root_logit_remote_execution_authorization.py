from __future__ import annotations

import copy
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


planner = _load(
    "qwen35_tp4_root_logit_remote_execution_plan_for_auth_test",
    "qwen35_tp4_root_logit_remote_execution_plan.py",
)
authorization = _load(
    "qwen35_tp4_root_logit_remote_execution_authorization",
    "qwen35_tp4_root_logit_remote_execution_authorization.py",
)


def _plan(root):
    return planner.build_remote_execution_plan(
        repo_root=root / "repo",
        output_dir=root / "plan",
        run_tag="root-logit-auth-r1",
    )


def _baseline(root):
    path = root / "resource_baseline.json"
    path.write_text(json.dumps({
        "schema_version": (
            "qwen35.tp4-controlled-shared-resource-baseline.v1"
        ),
        "classification": "READY",
        "ssh_target": "sitian@10.232.195.203",
        "captured_at": "2026-07-29T15:00:00+08:00",
        "gpu_indices": [2, 4, 5, 6],
        "selected": [
            {
                "gpu_index": index,
                "gpu_uuid": f"GPU-{index}",
                "free_bytes": 64 * 1024**3,
                "compute_processes": [{
                    "pid": 1000 + index,
                    "process_name": "python3",
                    "used_memory_mib": 436,
                    "start_time_ticks": 2000 + index,
                }],
            }
            for index in [2, 4, 5, 6]
        ],
        "minimum_free_bytes_per_gpu": 24 * 1024**3,
        "benchmark_execution_authorized": False,
    }) + "\n")
    return path


def test_authorization_binds_complete_root_plan():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = _plan(root)
        path = root / "authorization.json"
        payload = authorization.produce_authorization(
            plan=plan,
            output_path=path,
            nonce="root-logit-nonce-1",
        )

        assert payload["classification"] == "AUTHORIZED"
        assert payload["run_tag"] == plan["run_tag"]
        assert payload["ssh_target"] == plan["ssh_target"]
        assert payload["frozen_source_tree_sha256"] == (
            plan["frozen_source_tree_sha256"]
        )
        assert payload["model_manifest_sha256"] == (
            plan["model_manifest_sha256"]
        )
        assert payload["stage_order"] == plan["stage_order"]
        assert payload["nonce"] == "root-logit-nonce-1"
        assert payload["consumed"] is False
        assert authorization.validate_authorization(
            plan,
            payload,
        )["classification"] == "AUTHORIZED"


def test_authorization_binds_controlled_shared_policy_and_baseline():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = planner.build_remote_execution_plan(
            repo_root=root / "repo",
            output_dir=root / "plan",
            run_tag="root-logit-shared-auth-r1",
            resource_policy="controlled_shared",
            resource_baseline_path=_baseline(root),
        )
        payload = authorization.produce_authorization(
            plan=plan,
            output_path=root / "authorization.json",
            nonce="root-shared-r1",
        )
        assert payload["resource_policy"] == "controlled_shared"
        assert payload["resource_baseline_sha256"] == plan[
            "resource_baseline_sha256"
        ]
        changed = copy.deepcopy(payload)
        changed["resource_baseline_sha256"] = "0" * 64
        try:
            authorization.validate_authorization(plan, changed)
        except ValueError as error:
            assert "authorization" in str(error), str(error)
        else:
            raise AssertionError("baseline drift was accepted")


def test_authorization_rejects_nonce_or_plan_drift():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = _plan(root)
        for nonce in ("", "../escape", "a b", "中文"):
            try:
                authorization.produce_authorization(
                    plan=plan,
                    output_path=root / f"auth-{len(nonce)}.json",
                    nonce=nonce,
                )
            except ValueError as error:
                assert "nonce" in str(error), str(error)
            else:
                raise AssertionError("unsafe nonce was accepted")

        path = root / "authorization.json"
        payload = authorization.produce_authorization(
            plan=plan,
            output_path=path,
            nonce="root-logit-nonce-2",
        )
        for field, value in (
            ("plan_sha256", "0" * 64),
            ("frozen_source_tree_sha256", "1" * 64),
            ("model_manifest_sha256", "2" * 64),
            ("stage_order", list(reversed(plan["stage_order"]))),
        ):
            changed = copy.deepcopy(payload)
            changed[field] = value
            try:
                authorization.validate_authorization(plan, changed)
            except ValueError as error:
                assert "authorization" in str(error), str(error)
            else:
                raise AssertionError(
                    f"authorization {field} drift was accepted"
                )


def test_authorization_is_consumed_once_by_atomic_claim():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan = _plan(root)
        active = root / "authorization.json"
        consumed = root / "consumed_authorization.json"
        authorization.produce_authorization(
            plan=plan,
            output_path=active,
            nonce="root-logit-nonce-3",
        )
        payload = authorization.consume_authorization(
            plan=plan,
            authorization_path=active,
            consumed_path=consumed,
        )

        assert not active.exists()
        assert consumed.is_file()
        assert payload["consumed"] is True
        assert json.loads(consumed.read_text()) == payload
        try:
            authorization.consume_authorization(
                plan=plan,
                authorization_path=active,
                consumed_path=consumed,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("consumed authorization was reused")

        active = root / "children" / "authorization-2.json"
        consumed = root / "runtime" / "root" / "consumed.json"
        authorization.produce_authorization(
            plan=plan,
            output_path=active,
            nonce="root-logit-nonce-4",
        )
        payload = authorization.consume_authorization(
            plan=plan,
            authorization_path=active,
            consumed_path=consumed,
        )
        assert payload["consumed"] is True
        assert not active.exists()
        assert json.loads(consumed.read_text()) == payload


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 root-logit remote execution authorization "
        f"tests passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
