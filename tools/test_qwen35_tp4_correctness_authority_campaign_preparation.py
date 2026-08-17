from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
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


preparation = _load(
    "qwen35_tp4_correctness_authority_campaign_preparation",
    "qwen35_tp4_correctness_authority_campaign_preparation.py",
)


def test_preparation_freezes_current_root_source_identity():
    assert preparation.TP4_ROOT_SOURCE_TREE_SHA256 == (
        "ec19a8fa68abfba72e9594bdd1e05428"
        "b0add9169d3dbdde24190686c013411f"
    )


def _canonical_bytes(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload) + b"\n")


def _fake_dependencies(call_log, *, fail_at=None):
    source_sha = "a" * 64
    model_sha = preparation.MODEL_MANIFEST_SHA256
    workload_sha = "b" * 64

    def build_child(kind):
        def build(**kwargs):
            call_log.append(f"{kind}.build")
            if fail_at == f"{kind}.build":
                raise RuntimeError(f"injected {kind} build failure")
            output = Path(kwargs["output_dir"])
            payload = {
                "kind": kind,
                "run_tag": kwargs["run_tag"],
                "ssh_target": preparation.SSH_TARGET,
                "execution_env": dict(preparation.EXECUTION_ENV),
                "source_tree_sha256": (
                    preparation.TP4_ROOT_SOURCE_TREE_SHA256
                    if kind == "root"
                    else source_sha
                ),
                "model_manifest_sha256": model_sha,
                "workload_manifest_sha256": (
                    None if kind == "root" else workload_sha
                ),
                "remote_model_dir": kwargs.get("remote_model_dir"),
                "remote_model_manifest": kwargs.get(
                    "remote_model_manifest"
                ),
            }
            if kwargs.get("resource_policy") is not None:
                baseline = Path(kwargs["resource_baseline_path"])
                payload.update({
                    "resource_policy": kwargs["resource_policy"],
                    "resource_baseline_path": str(baseline.resolve()),
                    "resource_baseline_sha256": _sha256(baseline),
                    "gpu_indices": [2, 4, 5, 6],
                    "gpu_uuids": [
                        "GPU-2",
                        "GPU-4",
                        "GPU-5",
                        "GPU-6",
                    ],
                    "benchmark_execution_authorized": False,
                })
            if kind == "root":
                payload["frozen_source_tree_sha256"] = payload.pop(
                    "source_tree_sha256"
                )
                payload["stage_inputs"] = {
                    "verify": {
                        "local_artifact_dir": str(
                            Path(kwargs["repo_root"])
                            / "experiments"
                            / kwargs["run_tag"]
                            / "artifacts"
                        ),
                    },
                }
            else:
                downloaded = output / f"downloaded_{kind}_authority"
                payload["local_inputs"] = {
                    "configuration": str(
                        output / "remote_executor_configuration.json"
                    ),
                    "source_inventory": str(
                        kwargs["source_inventory_path"]
                    ),
                    "workload_manifest_sha256": workload_sha,
                }
                payload["gpu_indices"] = payload.get(
                    "gpu_indices",
                    [0, 1, 2, 3],
                )
                if kwargs.get("resource_policy") is not None:
                    payload["local_inputs"].update({
                        "resource_baseline": str(
                            Path(
                                kwargs["resource_baseline_path"]
                            ).resolve()
                        ),
                        "resource_baseline_sha256": payload[
                            "resource_baseline_sha256"
                        ],
                    })
                payload["ports"] = {
                    "dist_port": 31001,
                    "master_port": 31002,
                }
                payload["commands"] = {
                    "local_verify": {
                        "argv": (
                            [
                                "python3",
                                "verify.py",
                                str(
                                    downloaded
                                    / "cached_continuation_authority"
                                ),
                                str(
                                    downloaded
                                    / (
                                        "cached_continuation_"
                                        "independent_verification.json"
                                    )
                                ),
                            ]
                            if kind == "cached"
                            else [
                                "python3",
                                "verify.py",
                                str(downloaded),
                            ]
                        ),
                    },
                }
            _write(output / "remote_execution_plan.json", payload)
            return payload

        return build

    def verify_child(kind):
        def verify(path):
            call_log.append(f"{kind}.verify")
            payload = json.loads(Path(path).read_text())
            assert payload["kind"] == kind
            return payload

        return verify

    def authorization_module(kind):
        def produce_authorization(*, plan, output_path, nonce):
            call_log.append(f"{kind}.authorize")
            if fail_at == f"{kind}.authorize":
                raise RuntimeError(f"injected {kind} authorization failure")
            payload = {
                "kind": kind,
                "plan_sha256": hashlib.sha256(
                    _canonical_bytes(plan)
                ).hexdigest(),
                "nonce": nonce,
                "consumed": False,
            }
            _write(output_path, payload)
            return payload

        def validate_authorization(plan, payload):
            assert payload == {
                "kind": kind,
                "plan_sha256": hashlib.sha256(
                    _canonical_bytes(plan)
                ).hexdigest(),
                "nonce": payload["nonce"],
                "consumed": False,
            }
            return payload

        return SimpleNamespace(
            produce_authorization=produce_authorization,
            validate_authorization=validate_authorization,
        )

    class CampaignChild:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def build_campaign_plan(**kwargs):
        call_log.append("campaign.build")
        output = Path(kwargs["output_dir"])
        children = kwargs["children"]
        rows = []
        for child in children:
            child_plan = kwargs["child_plan_verifiers"][
                child.name
            ](child.plan_path)
            rows.append({
                "name": child.name,
                "run_tag": child_plan["run_tag"],
                "plan_path": str(Path(child.plan_path).resolve()),
                "plan_sha256": _sha256(child.plan_path),
                "source_tree_sha256": child_plan.get(
                    "frozen_source_tree_sha256",
                    child_plan.get("source_tree_sha256"),
                ),
                "model_manifest_sha256": model_sha,
                "authority_dir": str(
                    Path(child.authority_dir).resolve()
                ),
                "authorization_path": str(
                    Path(child.authorization_path).resolve()
                ),
                "consumed_authorization_path": str(
                    Path(child.consumed_authorization_path).resolve()
                ),
                "receipt_path": str(Path(child.receipt_path).resolve()),
                "failure_path": str(Path(child.failure_path).resolve()),
                **({
                    "resource_policy": child_plan["resource_policy"],
                    "resource_baseline_sha256": child_plan[
                        "resource_baseline_sha256"
                    ],
                } if child_plan.get("resource_policy") else {}),
            })
        payload = {
            "campaign_tag": kwargs["campaign_tag"],
            "ssh_target": preparation.SSH_TARGET,
            "execution_env": dict(preparation.EXECUTION_ENV),
            "child_order": list(preparation.CHILD_ORDER),
            "stage_order": list(preparation.STAGE_ORDER),
            "children": rows,
            "adapter_output_dir": str(
                Path(kwargs["adapter_output_dir"]).resolve()
            ),
            "bundle_output_dir": str(
                Path(kwargs["bundle_output_dir"]).resolve()
            ),
            "prerequisite_path": str(
                Path(kwargs["bundle_output_dir"]).resolve()
                / "correctness_prerequisites.json"
            ),
            "benchmark_execution_authorized": False,
            "execution_performed": False,
        }
        policies = {
            row.get("resource_policy") for row in rows
        }
        if policies != {None}:
            payload.update({
                "resource_policy": policies.pop(),
                "resource_baseline_sha256": rows[0][
                    "resource_baseline_sha256"
                ],
            })
        _write(output / "campaign_plan.json", payload)
        return payload

    def verify_campaign_plan(path, *, child_plan_verifiers):
        call_log.append("campaign.verify")
        payload = json.loads(Path(path).read_text())
        for row in payload["children"]:
            child_plan_verifiers[row["name"]](row["plan_path"])
        return payload

    return {
        "root_plan": SimpleNamespace(
            PLAN_NAME="remote_execution_plan.json",
            build_remote_execution_plan=build_child("root"),
            verify_remote_execution_plan=verify_child("root"),
        ),
        "root_authorization": authorization_module("root"),
        "cached_plan": SimpleNamespace(
            PLAN_NAME="remote_execution_plan.json",
            DOWNLOADED_AUTHORITY_NAME="downloaded_cached_authority",
            build_remote_execution_plan=build_child("cached"),
            verify_remote_execution_plan=verify_child("cached"),
        ),
        "cached_authorization": authorization_module("cached"),
        "engine_plan": SimpleNamespace(
            PLAN_NAME="remote_execution_plan.json",
            DOWNLOADED_AUTHORITY_NAME="downloaded_engine_authority",
            build_remote_execution_plan=build_child("engine"),
            verify_remote_execution_plan=verify_child("engine"),
        ),
        "engine_authorization": authorization_module("engine"),
        "campaign_plan": SimpleNamespace(
            PLAN_NAME="campaign_plan.json",
            CampaignChild=CampaignChild,
            build_campaign_plan=build_campaign_plan,
            verify_campaign_plan=verify_campaign_plan,
        ),
        "campaign_authorization": authorization_module("campaign"),
    }


def _inputs(root):
    configuration = root / "configuration.json"
    source_inventory = root / "source_inventory.json"
    _write(configuration, {
        "source_tree_sha256": "a" * 64,
        "model_manifest_sha256": preparation.MODEL_MANIFEST_SHA256,
        "workload_manifest_sha256": "b" * 64,
        "gpu_indices": [0, 1, 2, 3],
        "dist_port": 31001,
        "master_port": 31002,
    })
    _write(source_inventory, {
        "owned_files": ["tinyvllm/example.py"],
        "source_tree_sha256": "a" * 64,
    })
    return {
        "repo_root": root,
        "output_dir": root / "prepared",
        "campaign_tag": "campaign-20260729",
        "root_run_tag": "root-20260729",
        "cached_run_tag": "cached-20260729",
        "engine_run_tag": "engine-20260729",
        "configuration_path": configuration,
        "source_inventory_path": source_inventory,
        "remote_model_dir": "/remote/models/qwen35-m8",
        "remote_model_manifest": "/remote/models/qwen35-m8/manifest.json",
        "root_authorization_nonce": "root-nonce",
        "cached_authorization_nonce": "cached-nonce",
        "engine_authorization_nonce": "engine-nonce",
        "campaign_authorization_nonce": "campaign-nonce",
    }


def _baseline(root):
    path = root / "resource_baseline.json"
    _write(path, {
        "schema_version": (
            "qwen35.tp4-controlled-shared-resource-baseline.v1"
        ),
        "classification": "READY",
        "ssh_target": preparation.SSH_TARGET,
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
    })
    return path


def test_builder_publishes_closed_ready_bundle_and_reopens_every_artifact():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        calls = []
        dependencies = _fake_dependencies(calls)
        result = preparation.prepare_campaign_bundle(
            **_inputs(root),
            dependencies=dependencies,
        )
        manifest_path = root / "prepared" / preparation.MANIFEST_NAME
        assert result == preparation.verify_preparation_bundle(
            manifest_path,
            dependencies=dependencies,
        )
        assert result["classification"] == "READY"
        assert result["child_order"] == list(preparation.CHILD_ORDER)
        assert result["stage_order"] == list(preparation.STAGE_ORDER)
        assert result["execution_performed"] is False
        assert result["benchmark_execution_authorized"] is False
        assert Path(result["children"][0]["authority_dir"]) == (
            root / "experiments" / "root-20260729" / "artifacts"
        ).resolve()
        assert Path(result["children"][1]["authority_dir"]).name == (
            "downloaded_cached_authority"
        )
        assert Path(result["children"][2]["authority_dir"]).name == (
            "downloaded_engine_authority"
        )
        assert Path(
            result["inputs"]["configuration_path"]
        ).parent == (root / "prepared" / "inputs").resolve()
        assert Path(
            result["inputs"]["source_inventory_path"]
        ).parent == (root / "prepared" / "inputs").resolve()
        original_configuration = root / "configuration.json"
        original_inventory = root / "source_inventory.json"
        original_configuration.unlink()
        original_inventory.unlink()
        assert preparation.verify_preparation_bundle(
            manifest_path,
            dependencies=dependencies,
        )["classification"] == "READY"
        assert calls[:11] == [
            "root.build",
            "root.verify",
            "root.authorize",
            "cached.build",
            "cached.verify",
            "cached.authorize",
            "engine.build",
            "engine.verify",
            "engine.authorize",
            "campaign.build",
            "root.verify",
        ]
        assert manifest_path.is_file()
        assert not any(
            Path(path).exists()
            for row in result["children"]
            for path in (
                row["consumed_authorization_path"],
                row["receipt_path"],
                row["failure_path"],
                row["authority_dir"],
            )
        )


def test_builder_propagates_one_controlled_shared_baseline_to_all_children():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        inputs = _inputs(root)
        baseline = _baseline(root)
        configuration = json.loads(
            Path(inputs["configuration_path"]).read_text()
        )
        configuration["gpu_indices"] = [2, 4, 5, 6]
        _write(inputs["configuration_path"], configuration)
        result = preparation.prepare_campaign_bundle(
            **inputs,
            resource_policy="controlled_shared",
            resource_baseline_path=baseline,
            dependencies=_fake_dependencies([]),
        )
        copied = (
            root / "prepared" / "inputs" / "resource_baseline.json"
        ).resolve()
        assert result["resource_policy"] == "controlled_shared"
        assert result["resource_baseline_path"] == str(copied)
        assert result["resource_baseline_sha256"] == _sha256(copied)
        assert result["inputs"]["resource_baseline_path"] == str(copied)
        assert all(
            row["resource_policy"] == "controlled_shared"
            and row["resource_baseline_sha256"]
            == result["resource_baseline_sha256"]
            for row in result["children"]
        )
        campaign = json.loads(
            Path(result["campaign"]["plan_path"]).read_text()
        )
        assert campaign["resource_policy"] == "controlled_shared"
        assert campaign["resource_baseline_sha256"] == result[
            "resource_baseline_sha256"
        ]
        assert campaign["benchmark_execution_authorized"] is False


def test_builder_rejects_duplicate_tags_and_nonces_without_output():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        inputs = _inputs(root)
        inputs["cached_run_tag"] = inputs["root_run_tag"]
        try:
            preparation.prepare_campaign_bundle(
                **inputs,
                dependencies=_fake_dependencies([]),
            )
        except ValueError as error:
            assert "run tags" in str(error)
        else:
            raise AssertionError("duplicate run tags were accepted")
        assert not inputs["output_dir"].exists()

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        inputs = _inputs(root)
        inputs["campaign_authorization_nonce"] = inputs[
            "engine_authorization_nonce"
        ]
        try:
            preparation.prepare_campaign_bundle(
                **inputs,
                dependencies=_fake_dependencies([]),
            )
        except ValueError as error:
            assert "nonces" in str(error)
        else:
            raise AssertionError("duplicate nonces were accepted")
        assert not inputs["output_dir"].exists()


def test_verifier_rejects_plan_drift_and_future_runtime_output():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        dependencies = _fake_dependencies([])
        result = preparation.prepare_campaign_bundle(
            **_inputs(root),
            dependencies=dependencies,
        )
        manifest_path = root / "prepared" / preparation.MANIFEST_NAME
        root_plan = Path(result["children"][0]["plan_path"])
        root_plan.write_text(root_plan.read_text() + " ")
        try:
            preparation.verify_preparation_bundle(
                manifest_path,
                dependencies=dependencies,
            )
        except ValueError as error:
            assert "SHA" in str(error)
        else:
            raise AssertionError("child plan drift was accepted")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        dependencies = _fake_dependencies([])
        result = preparation.prepare_campaign_bundle(
            **_inputs(root),
            dependencies=dependencies,
        )
        manifest_path = root / "prepared" / preparation.MANIFEST_NAME
        future = Path(result["children"][0]["receipt_path"])
        future.parent.mkdir(parents=True)
        future.write_text("{}\n")
        try:
            preparation.verify_preparation_bundle(
                manifest_path,
                dependencies=dependencies,
            )
        except ValueError as error:
            assert "future output" in str(error)
        else:
            raise AssertionError("pre-existing runtime output was accepted")


def test_injected_mid_build_failure_removes_entire_output():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        inputs = _inputs(root)
        try:
            preparation.prepare_campaign_bundle(
                **inputs,
                dependencies=_fake_dependencies(
                    [],
                    fail_at="engine.authorize",
                ),
            )
        except RuntimeError as error:
            assert "injected engine authorization failure" in str(error)
        else:
            raise AssertionError("injected failure was swallowed")
        assert not inputs["output_dir"].exists()


def test_builder_rejects_configuration_identity_drift():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        inputs = _inputs(root)
        configuration = json.loads(
            Path(inputs["configuration_path"]).read_text()
        )
        configuration["source_tree_sha256"] = "c" * 64
        _write(inputs["configuration_path"], configuration)
        try:
            preparation.prepare_campaign_bundle(
                **inputs,
                dependencies=_fake_dependencies([]),
            )
        except ValueError as error:
            assert "configuration identity" in str(error)
        else:
            raise AssertionError(
                "configuration/child plan drift was accepted"
            )
        assert not inputs["output_dir"].exists()


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 correctness campaign preparation tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
