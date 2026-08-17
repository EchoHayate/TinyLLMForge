from __future__ import annotations

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
    "qwen35_tp4_engine_remote_execution_plan",
    "qwen35_tp4_engine_remote_execution_plan.py",
)
receipt = _load(
    "qwen35_tp4_engine_remote_execution_receipt_for_plan_test",
    "qwen35_tp4_engine_remote_execution_receipt.py",
)


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _expect_value_error(function, fragment):
    try:
        function()
    except ValueError as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(
            f"expected ValueError containing {fragment!r}"
        )


def _without_control_path_none(value):
    if not isinstance(value, list):
        return value
    result = []
    index = 0
    while index < len(value):
        if value[index:index + 2] == ["-o", "ControlPath=none"]:
            index += 2
            continue
        item = value[index]
        result.append(
            _without_control_path_none(item)
            if isinstance(item, list)
            else item
        )
        index += 1
    return result


def _fixture(root):
    repo = root / "repo"
    (repo / "tools").mkdir(parents=True)
    (repo / "tinyvllm").mkdir()
    (repo / "tools" / "driver.py").write_text("print('driver')\n")
    (
        repo / "tools" / "run_qwen35_tp4_engine_correctness_authority.py"
    ).write_text("print('authority')\n")
    (
        repo
        / "tools"
        / "verify_qwen35_tp4_engine_correctness_authority.py"
    ).write_text("print('verify')\n")
    (
        repo
        / "tools"
        / "qwen35_tp4_engine_remote_execution_receipt.py"
    ).write_text("print('receipt')\n")
    (
        repo
        / "tools"
        / "qwen35_tp4_engine_remote_execution_plan.py"
    ).write_text("print('plan')\n")
    (
        repo
        / "tools"
        / "qwen35_tp4_engine_remote_execution_executor.py"
    ).write_text("print('executor')\n")
    (
        repo
        / "tools"
        / "qwen35_tp4_engine_remote_execution_source_contract.py"
    ).write_text("print('source contract')\n")
    (
        repo
        / "tools"
        / "qwen35_tp4_engine_remote_execution_authorization.py"
    ).write_text("print('authorization')\n")
    (
        repo
        / "tools"
        / "qwen35_tp4_engine_remote_subprocess_adapter.py"
    ).write_text("print('adapter')\n")
    (repo / "tinyvllm" / "engine.py").write_text("VALUE = 1\n")
    model_manifest = root / "model_manifest.json"
    model_manifest.write_text('{"files":{}}\n')
    workload = root / "workload_manifest.json"
    workload.write_text('{"workloads":[]}\n')
    source_files = sorted([
        "tinyvllm/engine.py",
        "tools/driver.py",
        "tools/run_qwen35_tp4_engine_correctness_authority.py",
        "tools/verify_qwen35_tp4_engine_correctness_authority.py",
        "tools/qwen35_tp4_engine_remote_execution_receipt.py",
        "tools/qwen35_tp4_engine_remote_execution_plan.py",
        "tools/qwen35_tp4_engine_remote_execution_executor.py",
        "tools/qwen35_tp4_engine_remote_execution_source_contract.py",
        "tools/qwen35_tp4_engine_remote_execution_authorization.py",
        "tools/qwen35_tp4_engine_remote_subprocess_adapter.py",
    ])
    source_tree = planner.source_runner._source_tree_sha256([
        (name, repo / name) for name in source_files
    ])
    configuration = {
        "model_dir": "/local/model",
        "model_manifest_path": str(model_manifest),
        "model_manifest_sha256": planner._sha256(model_manifest),
        "source_tree_sha256": source_tree,
        "workload_manifest_path": str(workload),
        "workload_manifest_sha256": planner._sha256(workload),
        "model_fingerprint": "qwen35-m8-authority",
        "gpu_indices": [2, 3, 4, 5],
        "world_size": 4,
        "dist_port": 32101,
        "master_port": 32102,
        "max_cache_entries": 8,
        "max_cache_bytes": 1 << 30,
        "timeout_s": 600.0,
    }
    configuration_path = root / "executor_configuration.json"
    inventory_path = root / "source_inventory.json"
    _write_json(configuration_path, configuration)
    _write_json(inventory_path, {
        "owned_files": source_files,
        "source_tree_sha256": source_tree,
    })
    return repo, configuration_path, inventory_path


def _baseline(root):
    path = root / "resource_baseline.json"
    _write_json(path, {
        "schema_version": (
            "qwen35.tp4-controlled-shared-resource-baseline.v1"
        ),
        "classification": "READY",
        "ssh_target": "sitian@10.232.195.203",
        "captured_at": "2026-07-29T15:00:00+08:00",
        "gpu_indices": [2, 3, 4, 5],
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
            for index in [2, 3, 4, 5]
        ],
        "minimum_free_bytes_per_gpu": 24 * 1024**3,
        "benchmark_execution_authorized": False,
    })
    return path


def test_plan_binds_bundle_resources_commands_and_exact_verification():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo, configuration, inventory = _fixture(root)
        output = root / "remote-plan"
        payload = planner.build_remote_execution_plan(
            repo_root=repo,
            configuration_path=configuration,
            source_inventory_path=inventory,
            output_dir=output,
            run_tag="qwen35-engine-authority-20260729-r1",
            remote_model_dir="/remote/models/qwen35",
            remote_model_manifest=(
                "/remote/models/qwen35/model_manifest.json"
            ),
        )

        assert {path.name for path in output.iterdir()} == {
            planner.PLAN_NAME,
            planner.REMOTE_CONFIGURATION_NAME,
            planner.SOURCE_TAR_NAME,
        }
        assert payload["schema_version"] == (
            "qwen35.tp4-engine-remote-execution-plan.v1"
        )
        assert payload["ssh_target"] == "sitian@10.232.195.203"
        assert payload["run_tag"].endswith("-r1")
        assert payload["gpu_indices"] == [2, 3, 4, 5]
        assert payload["ports"] == {
            "dist_port": 32101,
            "master_port": 32102,
        }
        assert payload["source_tree_sha256"] == json.loads(
            inventory.read_text()
        )["source_tree_sha256"]
        assert payload["model_manifest_sha256"] == (
            planner._sha256(root / "model_manifest.json")
        )
        assert payload["local_inputs"]["source_tar_sha256"] == (
            planner._sha256(output / planner.SOURCE_TAR_NAME)
        )
        assert payload["command_order"] == [
            "reserve_remote",
            "upload",
            "stage",
            "resource_guard",
            "guarded_authority",
            "package_download",
            "safe_extract",
            "prepare_local_verifier",
            "local_verify",
        ]
        reserve_text = " ".join(
            payload["commands"]["reserve_remote"]["argv"]
        )
        assert "test ! -e" in reserve_text
        assert "mkdir -p" in reserve_text
        stage_text = " ".join(payload["commands"]["stage"]["argv"])
        for name in (
            "configuration_sha256",
            "source_inventory_sha256",
            "source_tar_sha256",
            "workload_manifest_sha256",
        ):
            assert payload["local_inputs"][name] in stage_text
        assert payload["local_inputs"]["configuration"] == str(
            (output / planner.REMOTE_CONFIGURATION_NAME).resolve()
        )
        assert payload["local_inputs"]["source_tar"] == str(
            (output / planner.SOURCE_TAR_NAME).resolve()
        )
        remote_configuration = json.loads(
            (output / planner.REMOTE_CONFIGURATION_NAME).read_text()
        )
        assert remote_configuration["model_dir"] == (
            "/remote/models/qwen35"
        )
        assert remote_configuration["model_manifest_path"] == (
            "/remote/models/qwen35/model_manifest.json"
        )
        assert remote_configuration["workload_manifest_path"].startswith(
            payload["remote_run_root"]
        )
        guarded = payload["commands"]["guarded_authority"]
        assert "TORCH_COMPILE_DISABLE=1" in guarded["authority_argv"]
        assert guarded["authority_argv"][-6:] == [
            "--configuration",
            payload["remote_inputs"]["configuration"],
            "--source-inventory",
            payload["remote_inputs"]["source_inventory"],
            "--output-root",
            payload["remote_authority_root"],
        ]
        guarded_text = " ".join(guarded["ssh_argv"])
        assert "nvidia-smi" in guarded_text
        assert "QWEN35_FINAL_RESOURCE_JSON=" in guarded_text
        assert (
            "run_qwen35_tp4_engine_correctness_authority.py"
            in guarded_text
        )
        assert guarded["final_resource_recheck"] is True
        guard = payload["commands"]["resource_guard"]
        assert guard["requires_no_active_compute_processes"] is True
        assert guard["minimum_free_bytes_per_gpu"] == 24 * 1024**3
        assert guard["gpu_indices"] == [2, 3, 4, 5]
        assert "compute_processes" in " ".join(guard["argv"])
        assert "nvidia-smi" in " ".join(guard["argv"])
        assert "import torch" not in " ".join(guard["argv"])
        assert payload["command_order"][-2:] == [
            "prepare_local_verifier",
            "local_verify",
        ]
        prepare_local = payload["commands"]["prepare_local_verifier"]
        assert prepare_local["source_tar"] == str(
            (output / planner.SOURCE_TAR_NAME).resolve()
        )
        assert prepare_local["source_tree_sha256"] == (
            payload["source_tree_sha256"]
        )
        assert payload["commands"]["local_verify"]["argv"] == [
            sys.executable,
            str(
                (
                    output
                    / "local_verifier_source"
                    / "tools"
                    / "verify_qwen35_tp4_engine_correctness_authority.py"
                ).resolve()
            ),
            str((output / "downloaded_authority").resolve()),
        ]
        assert planner.verify_remote_execution_plan(
            output / planner.PLAN_NAME
        ) == payload


def test_plan_binds_controlled_shared_baseline_without_configuration_drift():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo, configuration, inventory = _fixture(root)
        baseline = _baseline(root)
        output = root / "remote-plan"
        payload = planner.build_remote_execution_plan(
            repo_root=repo,
            configuration_path=configuration,
            source_inventory_path=inventory,
            output_dir=output,
            run_tag="qwen35-engine-shared-r1",
            remote_model_dir="/remote/models/qwen35",
            remote_model_manifest=(
                "/remote/models/qwen35/model_manifest.json"
            ),
            resource_policy="controlled_shared",
            resource_baseline_path=baseline,
        )
        assert payload["resource_policy"] == "controlled_shared"
        assert payload["resource_baseline_sha256"] == planner._sha256(
            baseline
        )
        assert payload["local_inputs"]["resource_baseline"] == str(
            (output / planner.RESOURCE_BASELINE_NAME).resolve()
        )
        guard = payload["commands"]["resource_guard"]
        assert guard["requires_no_active_compute_processes"] is False
        assert guard["resource_policy"] == "controlled_shared"
        assert guard["resource_baseline_sha256"] == payload[
            "resource_baseline_sha256"
        ]
        guard_text = " ".join(guard["argv"])
        assert "controlled_shared" in guard_text
        assert payload["remote_inputs"]["resource_baseline"] in guard_text
        remote_configuration = json.loads(
            (output / planner.REMOTE_CONFIGURATION_NAME).read_text()
        )
        assert "resource_policy" not in remote_configuration
        assert "resource_baseline_sha256" not in remote_configuration


def test_plan_rejects_unsafe_or_existing_destinations_and_identity_drift():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo, configuration, inventory = _fixture(root)
        for tag in ("../escape", "space tag", "semi;colon"):
            _expect_value_error(
                lambda tag=tag: planner.build_remote_execution_plan(
                    repo_root=repo,
                    configuration_path=configuration,
                    source_inventory_path=inventory,
                    output_dir=root / "plan",
                    run_tag=tag,
                    remote_model_dir="/remote/model",
                    remote_model_manifest="/remote/model/manifest.json",
                ),
                "run tag",
            )
        existing = root / "existing"
        existing.mkdir()
        _expect_value_error(
            lambda: planner.build_remote_execution_plan(
                repo_root=repo,
                configuration_path=configuration,
                source_inventory_path=inventory,
                output_dir=existing,
                run_tag="safe-r1",
                remote_model_dir="/remote/model",
                remote_model_manifest="/remote/model/manifest.json",
            ),
            "already exists",
        )
        (repo / "tools" / "driver.py").write_text("drift = True\n")
        _expect_value_error(
            lambda: planner.build_remote_execution_plan(
                repo_root=repo,
                configuration_path=configuration,
                source_inventory_path=inventory,
                output_dir=root / "drift-plan",
                run_tag="safe-r2",
                remote_model_dir="/remote/model",
                remote_model_manifest="/remote/model/manifest.json",
            ),
            "source tree",
        )
        assert not (root / "drift-plan").exists()


def test_plan_requires_authority_driver_and_verifier_sources():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo, configuration, inventory = _fixture(root)
        inventory_payload = json.loads(inventory.read_text())
        inventory_payload["owned_files"].remove(
            "tools/verify_qwen35_tp4_engine_correctness_authority.py"
        )
        inventory_payload["source_tree_sha256"] = (
            planner.source_runner._source_tree_sha256([
                (name, repo / name)
                for name in inventory_payload["owned_files"]
            ])
        )
        _write_json(inventory, inventory_payload)
        configuration_payload = json.loads(configuration.read_text())
        configuration_payload["source_tree_sha256"] = (
            inventory_payload["source_tree_sha256"]
        )
        _write_json(configuration, configuration_payload)
        _expect_value_error(
            lambda: planner.build_remote_execution_plan(
                repo_root=repo,
                configuration_path=configuration,
                source_inventory_path=inventory,
                output_dir=root / "missing-authority-source",
                run_tag="safe-r5",
                remote_model_dir="/remote/model",
                remote_model_manifest="/remote/model/manifest.json",
            ),
            "authority source inventory",
        )


def test_verifier_rejects_tamper_and_module_has_no_execution_surface():
    source = (
        TOOLS / "qwen35_tp4_engine_remote_execution_plan.py"
    ).read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "subprocess." not in source
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo, configuration, inventory = _fixture(root)
        output = root / "plan"
        planner.build_remote_execution_plan(
            repo_root=repo,
            configuration_path=configuration,
            source_inventory_path=inventory,
            output_dir=output,
            run_tag="safe-r3",
            remote_model_dir="/remote/model",
            remote_model_manifest="/remote/model/manifest.json",
        )
        plan_path = output / planner.PLAN_NAME
        payload = json.loads(plan_path.read_text())
        payload["gpu_indices"] = [0, 1, 2, 3]
        _write_json(plan_path, payload)
        _expect_value_error(
            lambda: planner.verify_remote_execution_plan(plan_path),
            "plan",
        )

        planner.build_remote_execution_plan(
            repo_root=repo,
            configuration_path=configuration,
            source_inventory_path=inventory,
            output_dir=root / "plan-command",
            run_tag="safe-r4",
            remote_model_dir="/remote/model",
            remote_model_manifest="/remote/model/manifest.json",
        )
        command_plan = root / "plan-command" / planner.PLAN_NAME
        payload = json.loads(command_plan.read_text())
        payload["commands"]["stage"]["argv"][-1] = "true"
        _write_json(command_plan, payload)
        _expect_value_error(
            lambda: planner.verify_remote_execution_plan(command_plan),
            "command",
        )

        payload = json.loads(
            (
                root / "plan-command" / planner.PLAN_NAME
            ).read_text()
        )
        payload["commands"]["guarded_authority"][
            "final_resource_recheck"
        ] = False
        _write_json(command_plan, payload)
        _expect_value_error(
            lambda: planner.verify_remote_execution_plan(command_plan),
            "command",
        )

        payload = json.loads(
            (
                root / "plan-command" / planner.PLAN_NAME
            ).read_text()
        )
        payload["commands"]["local_verify"]["argv"][1] = (
            "/tmp/unbound-verifier.py"
        )
        _write_json(command_plan, payload)
        _expect_value_error(
            lambda: planner.verify_remote_execution_plan(command_plan),
            "command",
        )


def test_verifier_accepts_frozen_legacy_transport_and_python():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo, configuration, inventory = _fixture(root)
        output = root / "plan"
        planner.build_remote_execution_plan(
            repo_root=repo,
            configuration_path=configuration,
            source_inventory_path=inventory,
            output_dir=output,
            run_tag="safe-legacy-r1",
            remote_model_dir="/remote/model",
            remote_model_manifest="/remote/model/manifest.json",
        )
        plan_path = output / planner.PLAN_NAME
        payload = json.loads(plan_path.read_text())
        for command in payload["commands"].values():
            for name, value in tuple(command.items()):
                command[name] = _without_control_path_none(value)
        for name in (
            "safe_extract",
            "prepare_local_verifier",
            "local_verify",
        ):
            payload["commands"][name]["argv"][0] = "/usr/bin/python3"
        _write_json(plan_path, payload)

        assert planner.verify_remote_execution_plan(plan_path) == payload


def test_verifier_rejects_non_python_frozen_local_interpreter():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo, configuration, inventory = _fixture(root)
        output = root / "plan"
        planner.build_remote_execution_plan(
            repo_root=repo,
            configuration_path=configuration,
            source_inventory_path=inventory,
            output_dir=output,
            run_tag="safe-invalid-interpreter-r1",
            remote_model_dir="/remote/model",
            remote_model_manifest="/remote/model/manifest.json",
        )
        plan_path = output / planner.PLAN_NAME
        payload = json.loads(plan_path.read_text())
        payload["commands"]["local_verify"]["argv"][0] = "/bin/sh"
        _write_json(plan_path, payload)

        _expect_value_error(
            lambda: planner.verify_remote_execution_plan(plan_path),
            "command",
        )


def test_complete_authority_verifier_exposes_read_only_cli():
    verifier = (
        TOOLS / "verify_qwen35_tp4_engine_correctness_authority.py"
    ).read_text(encoding="utf-8")
    assert "def main(argv=None):" in verifier
    assert 'parser.add_argument("run_dir")' in verifier


def test_real_plan_is_compatible_with_execution_receipt_contract():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo, configuration, inventory = _fixture(root)
        output = root / "plan"
        plan = planner.build_remote_execution_plan(
            repo_root=repo,
            configuration_path=configuration,
            source_inventory_path=inventory,
            output_dir=output,
            run_tag="receipt-compatible-r1",
            remote_model_dir="/remote/model",
            remote_model_manifest="/remote/model/manifest.json",
        )
        remote_configuration = json.loads(
            (
                output / planner.REMOTE_CONFIGURATION_NAME
            ).read_text()
        )
        pass_payload = {
            "classification": "PASS",
            "model_manifest_sha256": remote_configuration[
                "model_manifest_sha256"
            ],
            "source_tree_sha256": plan["source_tree_sha256"],
            "workload_manifest_sha256": remote_configuration[
                "workload_manifest_sha256"
            ],
            "reference_classification": "PASS",
            "engine_classification": "PASS",
        }
        resource = {
            "classification": "READY",
            "selected": [
                {
                    "gpu_index": index,
                    "gpu_uuid": f"GPU-{index}",
                    "free_bytes": 25 * 1024**3,
                    "compute_processes": [],
                }
                for index in plan["gpu_indices"]
            ],
        }
        steps = []
        for name in plan["command_order"]:
            stdout = ""
            if name == "resource_guard":
                stdout = json.dumps(resource)
            elif name == "guarded_authority":
                stdout = "\n".join([
                    "QWEN35_FINAL_RESOURCE_JSON="
                    + json.dumps(resource),
                    json.dumps(pass_payload),
                ])
            elif name == "local_verify":
                stdout = json.dumps(pass_payload)
            step = {
                "name": name,
                "command_sha256": receipt._canonical_sha(
                    plan["commands"][name]
                ),
                "returncode": 0,
                "stdout": stdout,
                "stderr": "",
            }
            if name == "package_download":
                step.update({
                    "output_sha256": "d" * 64,
                    "output_size": 1,
                })
            steps.append(step)
        summary = receipt.produce_execution_receipt(
            plan=plan,
            step_results=steps,
            output_path=output / "execution_receipt.json",
            authorization_record={
                "schema_version": (
                    "qwen35.tp4-engine-remote-execution-authorization.v1"
                ),
                "classification": "AUTHORIZED",
                "plan_sha256": receipt._canonical_sha(plan),
                "run_tag": plan["run_tag"],
                "source_tree_sha256": plan[
                    "source_tree_sha256"
                ],
                "model_manifest_sha256": plan[
                    "model_manifest_sha256"
                ],
                "workload_manifest_sha256": remote_configuration[
                    "workload_manifest_sha256"
                ],
                "gpu_indices": plan["gpu_indices"],
                "ports": plan["ports"],
                "nonce": "integration-r1",
                "consumed": True,
            },
        )
        assert summary["classification"] == "PASS"
        assert summary["step_count"] == len(plan["command_order"])


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 Engine remote execution plan tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
