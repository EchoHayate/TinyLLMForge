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
    "qwen35_tp4_cached_continuation_remote_execution_plan",
    "qwen35_tp4_cached_continuation_remote_execution_plan.py",
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
    source_files = sorted([
        "tinyvllm/engine.py",
        "tools/qwen35_tp4_cached_continuation_remote_execution_plan.py",
        "tools/qwen35_tp4_cached_continuation_remote_execution_receipt.py",
        "tools/qwen35_tp4_cached_continuation_remote_execution_executor.py",
        "tools/qwen35_tp4_engine_remote_execution_authorization.py",
        "tools/qwen35_tp4_engine_remote_execution_source_contract.py",
        "tools/qwen35_tp4_engine_remote_subprocess_adapter.py",
        "tools/run_qwen35_tp4_cached_continuation_authority.py",
        "tools/verify_qwen35_tp4_cached_continuation_correctness_gate.py",
    ])
    for name in source_files:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"VALUE = {name!r}\n", encoding="utf-8")
    model_manifest = root / "model_manifest.json"
    model_manifest.write_text('{"files":{}}\n', encoding="utf-8")
    workload = root / "workload_manifest.json"
    workload.write_text('{"workloads":[]}\n', encoding="utf-8")
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


def test_plan_binds_cached_driver_guard_package_and_local_verifier():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo, configuration, inventory = _fixture(root)
        output = root / "remote-plan"
        payload = planner.build_remote_execution_plan(
            repo_root=repo,
            configuration_path=configuration,
            source_inventory_path=inventory,
            output_dir=output,
            run_tag="qwen35-cached-authority-20260729-r1",
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
            "qwen35.tp4-cached-continuation-remote-execution-plan.v1"
        )
        assert payload["ssh_target"] == "sitian@10.232.195.203"
        assert payload["gpu_indices"] == [2, 3, 4, 5]
        assert payload["ports"] == {
            "dist_port": 32101,
            "master_port": 32102,
        }
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
        guard = payload["commands"]["resource_guard"]
        assert guard["requires_no_active_compute_processes"] is True
        assert guard["minimum_free_bytes_per_gpu"] == 24 * 1024**3
        assert "compute_processes" in " ".join(guard["argv"])
        assert "nvidia-smi" in " ".join(guard["argv"])
        guarded = payload["commands"]["guarded_authority"]
        assert "TORCH_COMPILE_DISABLE=1" in guarded["authority_argv"]
        assert guarded["authority_argv"][-8:] == [
            "--configuration",
            payload["remote_inputs"]["configuration"],
            "--source-inventory",
            payload["remote_inputs"]["source_inventory"],
            "--output-dir",
            payload["remote_cached_authority_dir"],
            "--verification-path",
            payload["remote_cached_verification_path"],
        ]
        guarded_text = " ".join(guarded["ssh_argv"])
        assert (
            "run_qwen35_tp4_cached_continuation_authority.py"
            in guarded_text
        )
        assert "QWEN35_FINAL_RESOURCE_JSON=" in guarded_text
        package_text = " ".join(
            payload["commands"]["package_download"]["remote_argv"]
        )
        assert "cached_continuation_authority" in package_text
        assert (
            "cached_continuation_independent_verification.json"
            in package_text
        )
        assert "authority_summary.json" not in package_text
        local_verify = payload["commands"]["local_verify"]["argv"]
        assert local_verify[:2] == [sys.executable, "-c"]
        assert (
            "verify_qwen35_tp4_cached_continuation_correctness_gate.py"
            in local_verify[3]
        )
        assert local_verify[-2:] == [
            str(
                (
                    output
                    / planner.DOWNLOADED_AUTHORITY_NAME
                    / "cached_continuation_authority"
                ).resolve()
            ),
            str(
                (
                    output
                    / planner.DOWNLOADED_AUTHORITY_NAME
                    / "cached_continuation_independent_verification.json"
                ).resolve()
            ),
        ]
        assert planner.verify_remote_execution_plan(
            output / planner.PLAN_NAME
        ) == payload


def test_plan_binds_controlled_shared_baseline():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo, configuration, inventory = _fixture(root)
        baseline = _baseline(root)
        output = root / "cached-plan"
        payload = planner.build_remote_execution_plan(
            repo_root=repo,
            configuration_path=configuration,
            source_inventory_path=inventory,
            output_dir=output,
            run_tag="qwen35-cached-shared-r1",
            remote_model_dir="/remote/models/qwen35",
            remote_model_manifest=(
                "/remote/models/qwen35/model_manifest.json"
            ),
            resource_policy="controlled_shared",
            resource_baseline_path=baseline,
        )
        assert payload["resource_policy"] == "controlled_shared"
        assert payload["commands"]["resource_guard"][
            "requires_no_active_compute_processes"
        ] is False
        assert payload["remote_inputs"]["resource_baseline"] in " ".join(
            payload["commands"]["resource_guard"]["argv"]
        )


def test_plan_rejects_unsafe_destinations_and_source_identity_drift():
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
        (repo / "tinyvllm" / "engine.py").write_text(
            "DRIFT = True\n",
            encoding="utf-8",
        )
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


def test_plan_requires_cached_protocol_sources():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo, configuration, inventory = _fixture(root)
        inventory_payload = json.loads(inventory.read_text())
        inventory_payload["owned_files"].remove(
            "tools/verify_qwen35_tp4_cached_continuation_correctness_gate.py"
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
                output_dir=root / "missing-source",
                run_tag="safe-r3",
                remote_model_dir="/remote/model",
                remote_model_manifest="/remote/model/manifest.json",
            ),
            "cached authority source inventory",
        )


def test_verifier_rejects_plan_command_tamper_and_has_no_execution_surface():
    source = (
        TOOLS
        / "qwen35_tp4_cached_continuation_remote_execution_plan.py"
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
            run_tag="safe-r4",
            remote_model_dir="/remote/model",
            remote_model_manifest="/remote/model/manifest.json",
        )
        plan_path = output / planner.PLAN_NAME
        payload = json.loads(plan_path.read_text())
        payload["commands"]["package_download"]["remote_argv"][-1] = (
            "true"
        )
        _write_json(plan_path, payload)
        _expect_value_error(
            lambda: planner.verify_remote_execution_plan(plan_path),
            "command",
        )


def test_verifier_accepts_legacy_commands_without_control_path_none():
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


def test_verifier_rejects_non_python_legacy_local_interpreter():
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
        payload["commands"]["safe_extract"]["argv"][0] = "/bin/sh"
        _write_json(plan_path, payload)

        _expect_value_error(
            lambda: planner.verify_remote_execution_plan(plan_path),
            "command",
        )


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 cached-continuation remote execution plan tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
