from __future__ import annotations

import copy
import base64
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
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


planner = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_remote_execution_plan",
    (
        "qwen35_tp4_hybrid_prefix_benchmark_"
        "remote_execution_plan.py"
    ),
)


def _write(path, payload):
    path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_tree_sha256(files):
    digest = hashlib.sha256()
    for name, content in files:
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(content)
    return digest.hexdigest()


def _launch_plan(
    source_tar,
    prerequisites,
    model_manifest,
    *,
    source_tree_sha256,
    resource_policy="strict-exclusive",
    maximum_gpu_utilization_percent=None,
):
    remote_root = (
        "/data00/home/sitian/sitian-workspace01/tllm/"
        "qwen35-tp4-hybrid-prefix-benchmark-runs"
    )
    run_tag = "benchmark-plan-r1"
    remote_run = f"{remote_root}/{run_tag}"
    case_commands = []
    for index in range(70):
        case_id = f"case-{index:02d}"
        case_commands.append({
            "case_id": case_id,
            "policy": "recompute" if index % 2 == 0 else "exact_restore",
            "workload": f"w{index % 5}",
            "phase": "measured",
            "repetition": index % 5,
            "dist_port": 22000 + index * 2,
            "master_port": 22001 + index * 2,
            "cwd": f"{remote_run}/source",
            "env": {
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "TINYVLLM_DIST_PORT": str(22000 + index * 2),
                "MASTER_PORT": str(22001 + index * 2),
                "PYTHONPATH": f"{remote_run}/source",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            "argv": [
                planner.REMOTE_PYTHON,
                "tools/qwen35_tp4_hybrid_prefix_benchmark_worker.py",
                "--case",
                case_id,
                "--correctness-prerequisites",
                f"{remote_run}/correctness_prerequisites.json",
            ],
            "log_path": f"{remote_run}/logs/{case_id}.log",
        })
    metadata = {
        "source_manifest.json": {
            "schema_version": planner.BENCHMARK_SCHEMA_VERSION,
            "source_tree_sha256": source_tree_sha256,
            "model_manifest_sha256": "c" * 64,
        },
        "environment.json": {
            "schema_version": planner.BENCHMARK_SCHEMA_VERSION,
            "world_size": 4,
            "python": planner.REMOTE_PYTHON,
        },
        "gpu_assignments.json": {
            "schema_version": planner.BENCHMARK_SCHEMA_VERSION,
            "assignments": [
                {
                    "rank": index,
                    "gpu_index": index,
                    "gpu_uuid": f"GPU-{index}",
                    "free_bytes": 25 * 1024**3,
                    "compute_processes": [],
                }
                for index in range(4)
            ],
        },
        "commands.json": {
            "schema_version": planner.BENCHMARK_SCHEMA_VERSION,
            "commands": [
                {
                    key: row[key]
                    for key in (
                        "case_id",
                        "policy",
                        "workload",
                        "phase",
                        "repetition",
                        "dist_port",
                        "master_port",
                    )
                }
                for row in case_commands
            ],
        },
        "worker_logs.json": {
            "schema_version": planner.BENCHMARK_SCHEMA_VERSION,
            "worker_logs": {
                row["case_id"]: row["log_path"]
                for row in case_commands
            },
        },
    }
    return {
        "schema_version": planner.BENCHMARK_SCHEMA_VERSION,
        "run_tag": run_tag,
        "local_source_tar": str(source_tar),
        "source_tar_sha256": _sha256(source_tar),
        "remote_source_tar": f"{remote_root}/{run_tag}-source.tar",
        "remote_source": f"{remote_run}/source",
        "remote_output": f"{remote_run}/output",
        "remote_cases": f"{remote_run}/output/cases",
        "remote_logs": f"{remote_run}/logs",
        "remote_assembly": f"{remote_run}/assembly",
        "remote_artifact": f"{remote_run}/artifact",
        "remote_workload_manifest": (
            f"{remote_run}/workload_manifest.json"
        ),
        "stage_command": ["bash", "-lc", "stage-frozen"],
        "resource_policy": resource_policy,
        "maximum_gpu_utilization_percent": (
            maximum_gpu_utilization_percent
        ),
        "case_commands": case_commands,
        "assembly_metadata": metadata,
        "assembler_command": {
            "cwd": f"{remote_run}/source",
            "argv": [
                planner.REMOTE_PYTHON,
                (
                    "tools/"
                    "qwen35_tp4_hybrid_prefix_benchmark_assembler.py"
                ),
                "--output-dir",
                f"{remote_run}/artifact",
            ],
        },
        "worker_authorization": {
            "prerequisites_sha256": _sha256(prerequisites),
            "source_tree_sha256": source_tree_sha256,
            "model_manifest_sha256": _sha256(model_manifest),
            "workload_manifest_sha256": "d" * 64,
            "gpu_indices": [0, 1, 2, 3],
        },
    }


def _inputs(root):
    source_tar = root / "source.tar"
    prerequisites = root / "prerequisites.json"
    model_manifest = root / "model_manifest.json"
    source_files = [
        (
            "tools/verify_qwen35_tp4_hybrid_prefix_benchmark.py",
            b"def verify_run(path): return {'classification':'GO'}\n",
        ),
    ]
    source_root = root / "source"
    for name, content in source_files:
        path = source_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    with tarfile.open(source_tar, "w:") as handle:
        for name, _ in source_files:
            handle.add(source_root / name, arcname=name, recursive=False)
    prerequisites.write_bytes(b"prerequisites")
    model_manifest.write_bytes(b"model")
    return (
        source_tar,
        prerequisites,
        model_manifest,
        _source_tree_sha256(source_files),
    )


def _prerequisite_bundle(root):
    bundle = root / "prerequisite-bundle"
    bundle.mkdir()
    prerequisite = bundle / "correctness_prerequisites.json"
    prerequisite.write_bytes(b"prerequisites")
    nested = bundle / "prerequisites/cached_continuation"
    nested.mkdir(parents=True)
    (nested / "provenance.json").write_bytes(b"provenance")
    (nested / "execution_receipt.json").write_bytes(b"receipt")
    return prerequisite


def test_builder_emits_exact_11_step_plan_and_local_inputs():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_tar, _, model_manifest, source_tree = _inputs(root)
        prerequisites = _prerequisite_bundle(root)
        output = root / "plan-output"
        result = planner.build_remote_execution_plan(
            launch_plan=_launch_plan(
                source_tar,
                prerequisites,
                model_manifest,
                source_tree_sha256=source_tree,
            ),
            output_dir=output,
            local_prerequisites=prerequisites,
            local_model_manifest=model_manifest,
        )

        assert result["command_order"] == planner.COMMAND_ORDER
        assert set(result["commands"]) == set(planner.COMMAND_ORDER)
        assert len(result["case_commands"]) == 70
        assert result["execution_performed"] is False
        assert Path(result["local_inputs"]["source_tar"]).is_file()
        assert Path(result["local_inputs"]["prerequisites"]).is_file()
        assert Path(result["local_inputs"]["model_manifest"]).is_file()
        assert set(result["local_inputs"]["assembly_metadata"]) == {
            "source_manifest.json",
            "environment.json",
            "gpu_assignments.json",
            "commands.json",
            "worker_logs.json",
        }
        assert result["commands"]["workers"][
            "expected_case_ids"
        ] == [f"case-{index:02d}" for index in range(70)]
        assert result["commands"]["package_download"][
            "local_output"
        ].endswith("benchmark_artifact.tar")
        upload = result["commands"]["upload"]["argv"]
        assert len(upload) == 2
        assert result["local_inputs"]["prerequisites_tar"] in upload[1]
        stage = " ".join(
            str(value)
            for value in result["commands"]["stage"]["argv"]
        )
        assert result["local_inputs"][
            "prerequisites_tar_sha256"
        ] in stage
        assert "unsafe prerequisite bundle member" in stage
        assert "correctness_prerequisites.json" in stage


def test_reserve_remote_bootstraps_trusted_remote_root_before_checks():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_tar, _, model_manifest, source_tree = _inputs(root)
        prerequisites = _prerequisite_bundle(root)
        result = planner.build_remote_execution_plan(
            launch_plan=_launch_plan(
                source_tar,
                prerequisites,
                model_manifest,
                source_tree_sha256=source_tree,
            ),
            output_dir=root / "plan-output",
            local_prerequisites=prerequisites,
            local_model_manifest=model_manifest,
        )

        reserve = result["commands"]["reserve_remote"]["argv"][-1]
        remote_root = (
            "/data00/home/sitian/sitian-workspace01/tllm/"
            "qwen35-tp4-hybrid-prefix-benchmark-runs"
        )
        mkdir = f"mkdir -p {remote_root}"
        first_check = (
            f"test ! -e {remote_root}/benchmark-plan-r1"
        )
        assert mkdir in reserve
        assert reserve.index(mkdir) < reserve.index(first_check)


def test_workers_command_stays_below_remote_argument_limit():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_tar, _, model_manifest, source_tree = _inputs(root)
        prerequisites = _prerequisite_bundle(root)
        result = planner.build_remote_execution_plan(
            launch_plan=_launch_plan(
                source_tar,
                prerequisites,
                model_manifest,
                source_tree_sha256=source_tree,
            ),
            output_dir=root / "plan-output",
            local_prerequisites=prerequisites,
            local_model_manifest=model_manifest,
        )

        workers = result["commands"]["workers"]["argv"]
        remote_run = (
            "/data00/home/sitian/sitian-workspace01/tllm/"
            "qwen35-tp4-hybrid-prefix-benchmark-runs/"
            "benchmark-plan-r1"
        )
        worker_shell = planner._worker_shell(result["case_commands"])
        compressed = planner._compressed_remote_shell_command(
            worker_shell,
            remote_run,
        )
        payload = compressed[-2]
        decoded = gzip.decompress(
            base64.b64decode(payload, validate=True)
        ).decode("utf-8")
        assert max(len(value.encode("utf-8")) for value in workers) < (
            128 * 1024
        )
        assert remote_run in workers[-1]
        assert "case-00" in decoded
        assert "case-69" in decoded
        assert planner.COMPLETION_MARKER in decoded


def test_compressed_workers_transport_has_deterministic_gzip_timestamp():
    command = planner._compressed_remote_shell_command(
        "printf deterministic",
        "/remote/run",
    )
    compressed = base64.b64decode(command[-2], validate=True)

    assert compressed[4:8] == b"\0\0\0\0"


def test_plan_commands_preserve_resource_guard_and_safe_local_verify():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_tar, _, model_manifest, source_tree = _inputs(root)
        prerequisites = _prerequisite_bundle(root)
        result = planner.build_remote_execution_plan(
            launch_plan=_launch_plan(
                source_tar,
                prerequisites,
                model_manifest,
                source_tree_sha256=source_tree,
            ),
            output_dir=root / "plan-output",
            local_prerequisites=prerequisites,
            local_model_manifest=model_manifest,
        )

        guard = result["commands"]["resource_guard"]
        final = result["commands"]["final_resource_guard"]
        assert guard["gpu_indices"] == [0, 1, 2, 3]
        assert guard["minimum_free_bytes_per_gpu"] == 24 * 1024**3
        assert guard["requires_no_active_compute_processes"] is True
        assert final == guard
        assert result["commands"]["safe_extract"]["argv"][0] == (
            sys.executable
        )
        assert result["commands"]["local_verify"]["argv"][0] == (
            sys.executable
        )
        assert result["local_inputs"]["source_tar"] in (
            result["commands"]["local_verify"]["argv"]
        )
        assert result["worker_authorization"][
            "source_tree_sha256"
        ] in result["commands"]["local_verify"]["argv"]
        assert str(ROOT) not in result["commands"]["local_verify"]["argv"]
        assert result["commands"]["package_download"][
            "remote_argv"
        ][0] == "ssh"
        remote_verify_script = " ".join(
            result["commands"]["remote_verify"]["argv"]
        )
        local_verify_script = result["commands"]["local_verify"][
            "argv"
        ][2]
        expected_import_path = (
            "sys.path.insert(0,str(verifier.parent))"
        )
        assert expected_import_path in remote_verify_script
        assert expected_import_path in local_verify_script


def test_plan_emits_shared_low_utilization_resource_guard():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_tar, _, model_manifest, source_tree = _inputs(root)
        prerequisites = _prerequisite_bundle(root)
        result = planner.build_remote_execution_plan(
            launch_plan=_launch_plan(
                source_tar,
                prerequisites,
                model_manifest,
                source_tree_sha256=source_tree,
                resource_policy="shared-low-utilization",
                maximum_gpu_utilization_percent=10,
            ),
            output_dir=root / "plan-output",
            local_prerequisites=prerequisites,
            local_model_manifest=model_manifest,
        )

        guard = result["commands"]["resource_guard"]
        assert guard["resource_policy"] == "shared-low-utilization"
        assert guard["requires_no_active_compute_processes"] is False
        assert guard["maximum_gpu_utilization_percent"] == 10
        assert "utilization.gpu" in " ".join(guard["argv"])
        assert result["commands"]["final_resource_guard"] == guard


def test_built_plan_is_self_contained_after_original_inputs_are_removed():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_tar, _, model_manifest, source_tree = _inputs(root)
        prerequisites = _prerequisite_bundle(root)
        output = root / "plan-output"
        planner.build_remote_execution_plan(
            launch_plan=_launch_plan(
                source_tar,
                prerequisites,
                model_manifest,
                source_tree_sha256=source_tree,
            ),
            output_dir=output,
            local_prerequisites=prerequisites,
            local_model_manifest=model_manifest,
        )

        source_tar.unlink()
        shutil.rmtree(prerequisites.parent)
        model_manifest.unlink()
        result = planner.verify_remote_execution_plan(
            output / planner.PLAN_NAME
        )
        assert Path(result["local_inputs"]["source_tar"]).is_file()
        assert Path(result["local_inputs"]["prerequisites"]).is_file()
        assert Path(
            result["local_inputs"]["prerequisites_tar"]
        ).is_file()
        assert Path(result["local_inputs"]["model_manifest"]).is_file()


def test_builder_rejects_source_tree_identity_mismatch():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_tar, prerequisites, model_manifest, _ = _inputs(root)
        launch = _launch_plan(
            source_tar,
            prerequisites,
            model_manifest,
            source_tree_sha256="f" * 64,
        )
        try:
            planner.build_remote_execution_plan(
                launch_plan=launch,
                output_dir=root / "plan-output",
                local_prerequisites=prerequisites,
                local_model_manifest=model_manifest,
            )
        except ValueError as error:
            assert "source tree" in str(error), str(error)
        else:
            raise AssertionError("source tree identity mismatch was accepted")


def test_verifier_rejects_command_or_identity_tamper():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_tar, _, model_manifest, source_tree = _inputs(root)
        prerequisites = _prerequisite_bundle(root)
        output = root / "plan-output"
        planner.build_remote_execution_plan(
            launch_plan=_launch_plan(
                source_tar,
                prerequisites,
                model_manifest,
                source_tree_sha256=source_tree,
            ),
            output_dir=output,
            local_prerequisites=prerequisites,
            local_model_manifest=model_manifest,
        )
        plan_path = output / planner.PLAN_NAME
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        for mutate, fragment in (
            (
                lambda value: value["commands"]["workers"].update(
                    {"expected_case_ids": ["tampered"]}
                ),
                "command",
            ),
            (
                lambda value: value["worker_authorization"].update(
                    {"source_tree_sha256": "0" * 64}
                ),
                "binding",
            ),
        ):
            changed = copy.deepcopy(payload)
            mutate(changed)
            _write(plan_path, changed)
            try:
                planner.verify_remote_execution_plan(plan_path)
            except ValueError as error:
                assert fragment in str(error), str(error)
            else:
                raise AssertionError("tampered plan was accepted")
            _write(plan_path, payload)


def test_verifier_rejects_prerequisite_owned_file_inventory_drift():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_tar, _, model_manifest, source_tree = _inputs(root)
        prerequisites = _prerequisite_bundle(root)
        output = root / "plan-output"
        planner.build_remote_execution_plan(
            launch_plan=_launch_plan(
                source_tar,
                prerequisites,
                model_manifest,
                source_tree_sha256=source_tree,
            ),
            output_dir=output,
            local_prerequisites=prerequisites,
            local_model_manifest=model_manifest,
        )
        plan_path = output / planner.PLAN_NAME
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        original_owned_files = payload["local_inputs"][
            "prerequisites_owned_files"
        ]
        tampered_owned_files = [planner.PREREQUISITES_NAME]
        payload["local_inputs"]["prerequisites_owned_files"] = [
            planner.PREREQUISITES_NAME,
        ]
        stage_command = payload["commands"]["stage"]["argv"][-1]
        payload["commands"]["stage"]["argv"][-1] = stage_command.replace(
            json.dumps(original_owned_files, separators=(",", ":")),
            json.dumps(tampered_owned_files, separators=(",", ":")),
        )
        _write(plan_path, payload)

        try:
            planner.verify_remote_execution_plan(plan_path)
        except ValueError as error:
            assert "inventory" in str(error), str(error)
        else:
            raise AssertionError(
                "prerequisite owned-file inventory drift was accepted"
            )


def test_builder_rejects_existing_output_or_input_identity_drift():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_tar, _, model_manifest, source_tree = _inputs(root)
        prerequisites = _prerequisite_bundle(root)
        launch = _launch_plan(
            source_tar,
            prerequisites,
            model_manifest,
            source_tree_sha256=source_tree,
        )
        output = root / "plan-output"
        output.mkdir()
        try:
            planner.build_remote_execution_plan(
                launch_plan=launch,
                output_dir=output,
                local_prerequisites=prerequisites,
                local_model_manifest=model_manifest,
            )
        except ValueError as error:
            assert "exists" in str(error), str(error)
        else:
            raise AssertionError("existing output was accepted")

        output.rmdir()
        launch["local_source_tar"] = str(root / "missing.tar")
        try:
            planner.build_remote_execution_plan(
                launch_plan=launch,
                output_dir=output,
                local_prerequisites=prerequisites,
                local_model_manifest=model_manifest,
            )
        except ValueError as error:
            assert "source tar" in str(error), str(error)
        else:
            raise AssertionError("missing source tar was accepted")


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 hybrid-prefix benchmark remote execution "
        f"plan tests passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
