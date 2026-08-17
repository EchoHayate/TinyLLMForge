from __future__ import annotations

import importlib.util
import json
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


contract = _load(
    "qwen35_tp4_hybrid_prefix_contract_for_runner_test",
    "qwen35_tp4_hybrid_prefix_benchmark_contract.py",
)
runner = _load(
    "run_qwen35_tp4_hybrid_prefix_benchmark_remote",
    "run_qwen35_tp4_hybrid_prefix_benchmark_remote.py",
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


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_runner_identity_and_modes_are_frozen():
    assert runner.SSH_TARGET == "sitian@10.232.195.203"
    assert runner.MIN_GPU_FREE_BYTES == 24 * 1024**3
    assert runner.REQUIRED_GPU_INDICES == (2, 4, 5, 6)
    assert runner.MODES == (
        "preflight",
        "smoke",
        "canonical",
        "download-only",
        "verify-only",
    )
    assert runner.REMOTE_PYTHON == (
        "/data00/home/sitian/sitian-workspace01/tllm/"
        "env/bin/python"
    )
    ssh_argv = runner._ssh_argv(["true"])
    assert "KRB5CCNAME" not in ssh_argv
    assert "ControlMaster=no" in ssh_argv
    assert "ControlPath=none" in ssh_argv
    assert {
        (
            "tools/qwen35_tp4_hybrid_prefix_benchmark_"
            "remote_execution_authorization.py"
        ),
        (
            "tools/qwen35_tp4_hybrid_prefix_benchmark_"
            "remote_execution_receipt.py"
        ),
        (
            "tools/qwen35_tp4_hybrid_prefix_benchmark_"
            "remote_execution_executor.py"
        ),
        (
            "tools/qwen35_tp4_hybrid_prefix_benchmark_"
            "remote_execution_plan.py"
        ),
        "tools/qwen35_tp4_engine_remote_execution_source_contract.py",
        "tools/qwen35_tp4_engine_remote_subprocess_adapter.py",
    }.issubset(set(runner.BENCHMARK_OWNED_SOURCE_PATHS))
    assert (
        "tools/qwen35_tp4_real_prerequisite_authority_adapter.py"
        not in runner.BENCHMARK_OWNED_SOURCE_PATHS
    )


def test_gpu_selector_requires_exact_strict_p1_gpu_identity():
    rows = [
        {
            "gpu_index": index,
            "gpu_uuid": f"GPU-{index}",
            "free_bytes": runner.MIN_GPU_FREE_BYTES,
            "compute_processes": [],
        }
        for index in (0, 1, 2, 4, 5, 6)
    ]
    selected = runner._select_tp4_gpu_resources(rows)
    assert [row["gpu_index"] for row in selected] == [2, 4, 5, 6]

    missing = [row for row in rows if row["gpu_index"] != 5]
    _expect_value_error(
        lambda: runner._select_tp4_gpu_resources(missing),
        "2,4,5,6",
    )


def test_run_preflight_accepts_explicit_shared_resource_selector():
    calls = []
    with tempfile.TemporaryDirectory() as temporary:
        local_root = Path(temporary)
        prerequisites = local_root / "prerequisites.json"
        prerequisites.write_text("{}\n", encoding="utf-8")
        rows = [
            {
                "gpu_index": index,
                "gpu_uuid": f"GPU-{index}",
                "free_bytes": runner.MIN_GPU_FREE_BYTES,
                "compute_processes": [{"pid": 1000 + index}],
                "utilization_percent": 10,
            }
            for index in runner.REQUIRED_GPU_INDICES
        ]

        original = runner.contract.validate_prerequisites
        runner.contract.validate_prerequisites = lambda path: (
            contract.PrerequisiteStatus(
                classification="PASS",
                authorized=True,
                reasons=(),
            )
        )
        try:
            result = runner.run_preflight(
                run_tag="shared-resources",
                prerequisites_path=prerequisites,
                output_root=local_root,
                source_bundle_builder=lambda *, output_dir: {
                    "source_tree_sha256": "c" * 64,
                },
                remote_query=lambda: {"gpus": rows},
                resource_selector=lambda value: (
                    calls.append(value) or value
                ),
            )
        finally:
            runner.contract.validate_prerequisites = original

    assert result["classification"] == "READY"
    assert calls == [rows]
    assert result["selected_gpus"] == rows


def test_safe_run_tags_reject_shell_and_path_characters():
    assert runner.safe_run_tag("qwen35-bench_20260729") == (
        "qwen35-bench_20260729"
    )
    for value in (
        "",
        "../escape",
        "with space",
        "semi;colon",
        "slash/name",
        "$(command)",
    ):
        _expect_value_error(
            lambda value=value: runner.safe_run_tag(value),
            "run tag",
        )


def test_allocate_unique_port_pairs_rejects_duplicates():
    values = iter(((22000, 22001), (22002, 22003)))
    assert runner.allocate_unique_port_pairs(
        2,
        allocator=lambda: next(values),
    ) == [(22000, 22001), (22002, 22003)]

    duplicate = iter(((22000, 22001), (22001, 22002)))
    _expect_value_error(
        lambda: runner.allocate_unique_port_pairs(
            2,
            allocator=lambda: next(duplicate),
        ),
        "duplicate",
    )


def test_remote_port_pairs_use_one_non_ephemeral_remote_probe():
    calls = []
    expected = [(22000, 22001), (22002, 22003)]

    def command_runner(**kwargs):
        calls.append(kwargs)
        return {
            "returncode": 0,
            "stdout": json.dumps({
                "ephemeral_start": 32768,
                "pairs": expected,
            }),
            "stderr": "",
        }

    assert runner.allocate_remote_port_pairs(
        2,
        command_runner=command_runner,
        execution_env={"KRB5CCNAME": "test-cache"},
    ) == expected
    assert len(calls) == 1
    assert calls[0]["name"] == "remote_port_inventory"
    assert calls[0]["argv"][0] == "ssh"
    assert calls[0]["stdout_path"] is None
    assert calls[0]["env"] == {"KRB5CCNAME": "test-cache"}
    remote_command = " ".join(calls[0]["argv"])
    assert "/proc/sys/net/ipv4/ip_local_port_range" in remote_command
    assert "bind(('127.0.0.1',0))" not in remote_command


def test_remote_port_pairs_reject_ephemeral_range_results():
    def command_runner(**kwargs):
        return {
            "returncode": 0,
            "stdout": json.dumps({
                "ephemeral_start": 22001,
                "pairs": [(22000, 22001)],
            }),
            "stderr": "",
        }

    _expect_value_error(
        lambda: runner.allocate_remote_port_pairs(
            1,
            command_runner=command_runner,
            execution_env={"KRB5CCNAME": "test-cache"},
        ),
        "non-ephemeral",
    )


def test_case_commands_preserve_alternating_pair_order_and_unique_ports():
    runtime_artifacts = runner.WorkerRuntimeArtifacts(
        model_dir="/remote/model",
        model_manifest_path="/remote/model_manifest.json",
        correctness_prerequisites_path=(
            "/remote/correctness_prerequisites.json"
        ),
        workload_manifest_path="/remote/workload_manifest.json",
    )
    commands = runner.build_case_commands(
        remote_source="/remote/source",
        remote_output="/remote/output",
        ports=[
            (22000 + index * 2, 22001 + index * 2)
            for index in range(
                len(contract.build_case_matrix())
            )
        ],
        authorization=runner.WorkerAuthorization(
            prerequisites_sha256="a" * 64,
            source_tree_sha256=(
                contract.TP4_ROOT_SOURCE_TREE_SHA256
            ),
            model_manifest_sha256=(
                contract.MODEL_MANIFEST_SHA256
            ),
            workload_manifest_sha256=(
                contract.canonical_json_file_sha256(
                    contract.workload_manifest_payload()
                )
            ),
            gpu_indices=(0, 1, 2, 3),
        ),
        runtime_artifacts=runtime_artifacts,
    )

    matrix = contract.build_case_matrix()
    assert len(commands) == len(matrix)
    assert [row["case_id"] for row in commands] == [
        case.case_id for case in matrix
    ]
    assert commands[0]["policy"] == "recompute"
    measured_r1 = next(
        row
        for row in commands
        if row["phase"] == "measured"
        and row["repetition"] == 1
        and row["workload"] == "w0_short_control"
    )
    assert measured_r1["policy"] == "exact_restore"
    assert "--model-dir" in commands[0]["argv"]
    assert "/remote/model" in commands[0]["argv"]
    assert "--model-manifest" in commands[0]["argv"]
    assert "--correctness-prerequisites" in commands[0]["argv"]
    assert "--workload-manifest" in commands[0]["argv"]
    assert "/remote/workload_manifest.json" in commands[0]["argv"]
    all_ports = [
        port
        for row in commands
        for port in (row["dist_port"], row["master_port"])
    ]
    assert len(all_ports) == len(set(all_ports))


def test_worker_command_requires_explicit_authorization():
    case = contract.build_case_matrix()[0]
    _expect_value_error(
        lambda: runner.build_worker_command(
            case=case,
            remote_source="/remote/source",
            case_output_dir="/remote/output/case",
            dist_port=22000,
            master_port=22001,
            authorization=None,
        ),
        "authorization",
    )


def test_worker_command_requires_explicit_runtime_artifacts():
    case = contract.build_case_matrix()[0]
    authorization = runner.WorkerAuthorization(
        prerequisites_sha256="a" * 64,
        source_tree_sha256="c" * 64,
        model_manifest_sha256=contract.MODEL_MANIFEST_SHA256,
        workload_manifest_sha256=(
            contract.canonical_json_file_sha256(
                contract.workload_manifest_payload()
            )
        ),
        gpu_indices=(0, 1, 2, 3),
    )
    _expect_value_error(
        lambda: runner.build_worker_command(
            case=case,
            remote_source="/remote/source",
            case_output_dir="/remote/output/case",
            dist_port=22000,
            master_port=22001,
            authorization=authorization,
            runtime_artifacts=None,
        ),
        "runtime artifacts",
    )


def test_worker_command_accepts_field_equivalent_canonical_case():
    canonical = contract.build_case_matrix()[0]

    class EquivalentCase:
        case_id = canonical.case_id
        policy = canonical.policy
        workload = canonical.workload
        phase = canonical.phase
        repetition = canonical.repetition

    command = runner.build_worker_command(
        case=EquivalentCase(),
        remote_source="/remote/source",
        case_output_dir="/remote/output/case",
        dist_port=22000,
        master_port=22001,
        authorization=runner.WorkerAuthorization(
            prerequisites_sha256="a" * 64,
            source_tree_sha256="c" * 64,
            model_manifest_sha256=contract.MODEL_MANIFEST_SHA256,
            workload_manifest_sha256=(
                contract.canonical_json_file_sha256(
                    contract.workload_manifest_payload()
                )
            ),
            gpu_indices=(0, 1, 2, 3),
        ),
        runtime_artifacts=runner.WorkerRuntimeArtifacts(
            model_dir="/remote/model",
            model_manifest_path="/remote/model_manifest.json",
            correctness_prerequisites_path=(
                "/remote/correctness_prerequisites.json"
            ),
            workload_manifest_path="/remote/workload_manifest.json",
        ),
    )

    assert command["case_id"] == canonical.case_id


def test_worker_authorization_separates_benchmark_and_root_sources():
    benchmark_source = "c" * 64
    authorization = runner.WorkerAuthorization(
        prerequisites_sha256="a" * 64,
        source_tree_sha256=benchmark_source,
        model_manifest_sha256=contract.MODEL_MANIFEST_SHA256,
        workload_manifest_sha256=(
            contract.canonical_json_file_sha256(
                contract.workload_manifest_payload()
            )
        ),
        gpu_indices=(0, 1, 2, 3),
    )

    assert runner.validate_worker_authorization(
        authorization
    ).source_tree_sha256 == benchmark_source
    assert benchmark_source != contract.TP4_ROOT_SOURCE_TREE_SHA256


def test_blocked_correctness_returns_before_ssh_or_remote_path_creation():
    calls = []
    source_calls = []
    with tempfile.TemporaryDirectory() as temporary:
        local_root = Path(temporary)
        result = runner.run_preflight(
            run_tag="blocked-correctness",
            prerequisites_path=local_root / "missing.json",
            output_root=local_root,
            remote_query=lambda: calls.append("ssh"),
            source_bundle_builder=lambda *, output_dir: (
                source_calls.append(output_dir)
            ),
        )

        assert result["classification"] == "BLOCKED_CORRECTNESS"
        assert result["authorized"] is False
        assert calls == []
        assert source_calls == []
        output_dir = local_root / "blocked-correctness"
        assert sorted(path.name for path in output_dir.iterdir()) == [
            "benchmark_preflight.json"
        ]
        assert not (
            output_dir / "remote_path_created.json"
        ).exists()


def test_blocked_resources_records_query_without_worker_authorization():
    calls = []
    source_calls = []
    with tempfile.TemporaryDirectory() as temporary:
        local_root = Path(temporary)
        prerequisites = local_root / "prerequisites.json"
        prerequisites.write_text("{}\n", encoding="utf-8")

        original = runner.contract.validate_prerequisites
        runner.contract.validate_prerequisites = lambda path: (
            contract.PrerequisiteStatus(
                classification="PASS",
                authorized=True,
                reasons=(),
            )
        )
        try:
            result = runner.run_preflight(
                run_tag="blocked-resources",
                prerequisites_path=prerequisites,
                output_root=local_root,
                source_bundle_builder=lambda *, output_dir: (
                    source_calls.append(output_dir)
                    or {
                        "owned_files": ["tinyvllm/runtime.py"],
                        "source_tree_sha256": "c" * 64,
                        "tar_sha256": "d" * 64,
                        "tar_path": str(output_dir / "source.tar"),
                    }
                ),
                remote_query=lambda: (
                    calls.append("ssh")
                    or {
                        "gpus": [
                            {
                                "gpu_index": index,
                                "gpu_uuid": (
                                    "GPU-00000000-0000-0000-0000-"
                                    f"00000000000{index}"
                                ),
                                "free_bytes": (
                                    contract.MIN_GPU_FREE_BYTES
                                ),
                                "compute_processes": (
                                    [{"pid": 100 + index}]
                                    if index == 0
                                    else []
                                ),
                            }
                            for index in range(4)
                        ]
                    }
                ),
            )
        finally:
            runner.contract.validate_prerequisites = original

        assert result["classification"] == "BLOCKED_RESOURCES"
        assert result["authorized"] is False
        assert calls == ["ssh"]
        assert source_calls == [local_root / "blocked-resources"]
        assert result["source_bundle"]["source_tree_sha256"] == (
            "c" * 64
        )
        assert "worker_authorization" not in result


def test_preflight_calls_source_bundle_builder_with_named_output_dir():
    source_calls = []
    with tempfile.TemporaryDirectory() as temporary:
        local_root = Path(temporary)
        prerequisites = local_root / "prerequisites.json"
        prerequisites.write_text("valid\n", encoding="utf-8")
        original = runner.contract.validate_prerequisites
        runner.contract.validate_prerequisites = lambda path: (
            contract.PrerequisiteStatus(
                classification="PASS",
                authorized=True,
                reasons=(),
            )
        )
        try:
            result = runner.run_preflight(
                run_tag="named-source-output",
                prerequisites_path=prerequisites,
                output_root=local_root,
                source_bundle_builder=lambda *, output_dir: (
                    source_calls.append(output_dir)
                    or {
                        "owned_files": ["tinyvllm/runtime.py"],
                        "source_tree_sha256": "c" * 64,
                        "tar_sha256": "d" * 64,
                        "tar_path": str(output_dir / "source.tar"),
                    }
                ),
                remote_query=lambda: {
                    "gpus": [
                        {
                            "gpu_index": index,
                            "gpu_uuid": (
                                "GPU-00000000-0000-0000-0000-"
                                f"00000000000{index}"
                            ),
                            "free_bytes": (
                                contract.MIN_GPU_FREE_BYTES
                                + 1024
                            ),
                            "compute_processes": [],
                        }
                        for index in runner.REQUIRED_GPU_INDICES
                    ]
                },
            )
        finally:
            runner.contract.validate_prerequisites = original

        assert result["classification"] == "READY"
        assert source_calls == [local_root / "named-source-output"]


def test_ready_preflight_binds_source_model_prerequisite_and_gpu_identity():
    with tempfile.TemporaryDirectory() as temporary:
        local_root = Path(temporary)
        prerequisites = local_root / "prerequisites.json"
        prerequisites.write_text("valid\n", encoding="utf-8")
        original = runner.contract.validate_prerequisites
        runner.contract.validate_prerequisites = lambda path: (
            contract.PrerequisiteStatus(
                classification="PASS",
                authorized=True,
                reasons=(),
            )
        )
        try:
            result = runner.run_preflight(
                run_tag="ready",
                prerequisites_path=prerequisites,
                output_root=local_root,
                benchmark_source_tree_sha256="c" * 64,
                remote_query=lambda: {
                    "gpus": [
                        {
                            "gpu_index": index,
                            "gpu_uuid": (
                                "GPU-00000000-0000-0000-0000-"
                                f"00000000000{index}"
                            ),
                            "free_bytes": (
                                contract.MIN_GPU_FREE_BYTES
                                + 1024
                            ),
                            "compute_processes": [],
                        }
                        for index in runner.REQUIRED_GPU_INDICES
                    ]
                },
            )
        finally:
            runner.contract.validate_prerequisites = original

        assert result["classification"] == "READY"
        assert result["authorized"] is True
        authorization = runner.WorkerAuthorization.from_dict(
            result["worker_authorization"]
        )
        assert authorization.source_tree_sha256 == "c" * 64
        assert authorization.model_manifest_sha256 == (
            contract.MODEL_MANIFEST_SHA256
        )
        assert authorization.gpu_indices == runner.REQUIRED_GPU_INDICES


def test_launch_plan_requires_ready_and_binds_all_canonical_commands():
    ports = iter(
        (22000 + index * 2, 22001 + index * 2)
        for index in range(len(contract.build_case_matrix()))
    )
    preflight = {
        "classification": "READY",
        "authorized": True,
        "worker_authorization": {
            "prerequisites_sha256": "a" * 64,
            "source_tree_sha256": "c" * 64,
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
            "workload_manifest_sha256": (
                contract.canonical_json_file_sha256(
                    contract.workload_manifest_payload()
                )
            ),
            "gpu_indices": [0, 1, 2, 3],
        },
        "source_bundle": {
            "source_tree_sha256": "c" * 64,
            "tar_sha256": "d" * 64,
            "tar_path": "/local/benchmark_source.tar",
        },
        "selected_gpus": [
            {
                "gpu_index": index,
                "gpu_uuid": f"GPU-{index}",
                "free_bytes": contract.MIN_GPU_FREE_BYTES,
                "compute_processes": [],
            }
            for index in range(contract.WORLD_SIZE)
        ],
    }
    plan = runner.build_authorized_launch_plan(
        run_tag="ready-plan",
        preflight=preflight,
        remote_model_dir="/remote/model",
        remote_model_manifest="/remote/model_manifest.json",
        remote_prerequisites="/remote/correctness_prerequisites.json",
        port_allocator=lambda: next(ports),
    )

    assert plan["run_tag"] == "ready-plan"
    assert plan["worker_authorization"] == (
        preflight["worker_authorization"]
    )
    assert plan["source_tar_sha256"] == "d" * 64
    assert plan["remote_source"].endswith("/ready-plan/source")
    assert plan["remote_output"].endswith("/ready-plan/output")
    assert len(plan["case_commands"]) == len(
        contract.build_case_matrix()
    )
    assert plan["remote_artifact"].endswith("/ready-plan/artifact")
    assert (
        "tools/qwen35_tp4_hybrid_prefix_benchmark_assembler.py"
        in plan["assembler_command"]["argv"]
    )
    assert set(plan["assembly_metadata"]) == {
        "source_manifest.json",
        "environment.json",
        "gpu_assignments.json",
        "commands.json",
        "worker_logs.json",
    }
    assert len(
        plan["assembly_metadata"]["worker_logs.json"]["worker_logs"]
    ) == len(contract.build_case_matrix())
    assert len({
        row["log_path"] for row in plan["case_commands"]
    }) == len(contract.build_case_matrix())
    assert all(
        row["log_path"].startswith(plan["remote_logs"] + "/")
        for row in plan["case_commands"]
    )
    assert plan["stage_command"] == runner.build_remote_stage_command(
        run_tag="ready-plan",
        source_tree_sha256="c" * 64,
        tar_sha256="d" * 64,
    )
    assert plan["case_commands"][0]["argv"][-6:] == [
        "--correctness-prerequisites",
        "/remote/correctness_prerequisites.json",
        "--workload-manifest",
        plan["remote_workload_manifest"],
        "--workload-manifest-sha256",
        contract.canonical_json_file_sha256(
            contract.workload_manifest_payload()
        ),
    ]


def test_launch_plan_rejects_blocked_or_source_mismatched_preflight():
    blocked = {
        "classification": "BLOCKED_RESOURCES",
        "authorized": False,
    }
    _expect_value_error(
        lambda: runner.build_authorized_launch_plan(
            run_tag="blocked-plan",
            preflight=blocked,
            remote_model_dir="/remote/model",
            remote_model_manifest="/remote/model_manifest.json",
            remote_prerequisites="/remote/prerequisites.json",
        ),
        "READY",
    )
    mismatched = {
        "classification": "READY",
        "authorized": True,
        "worker_authorization": {
            "prerequisites_sha256": "a" * 64,
            "source_tree_sha256": "c" * 64,
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
            "workload_manifest_sha256": (
                contract.canonical_json_file_sha256(
                    contract.workload_manifest_payload()
                )
            ),
            "gpu_indices": [0, 1, 2, 3],
        },
        "source_bundle": {
            "source_tree_sha256": "e" * 64,
            "tar_sha256": "d" * 64,
            "tar_path": "/local/source.tar",
        },
    }
    _expect_value_error(
        lambda: runner.build_authorized_launch_plan(
            run_tag="mismatch-plan",
            preflight=mismatched,
            remote_model_dir="/remote/model",
            remote_model_manifest="/remote/model_manifest.json",
            remote_prerequisites="/remote/prerequisites.json",
        ),
        "source",
    )


def test_prepare_benchmark_source_bundle_uses_frozen_owned_inventory():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        (root / "tinyvllm").mkdir(parents=True)
        (root / "tinyvllm" / "runtime.py").write_text(
            "VALUE = 1\n",
            encoding="utf-8",
        )
        (root / "tinyvllm" / "__pycache__").mkdir()
        (root / "tinyvllm" / "__pycache__" / "runtime.pyc").write_bytes(
            b"interpreter-specific"
        )
        (root / "tools").mkdir()
        for path in runner.BENCHMARK_OWNED_SOURCE_PATHS[1:]:
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                f"# {path}\n",
                encoding="utf-8",
            )
        output_dir = Path(temporary) / "output"
        result = runner.prepare_benchmark_source_bundle(
            repo_root=root,
            output_dir=output_dir,
        )

        assert result["tar_path"] == str(
            output_dir / "benchmark_source.tar"
        )
        assert result["owned_paths"] == list(
            runner.BENCHMARK_OWNED_SOURCE_PATHS
        )
        assert not any(
            "__pycache__" in path or path.endswith(".pyc")
            for path in result["owned_files"]
        )
        assert (output_dir / "benchmark_source.tar").is_file()


def test_safe_download_member_rejects_links_and_traversal():
    assert runner.safe_download_member(
        "logs/worker.log",
        is_file=True,
        is_link=False,
    ) == "logs/worker.log"
    for name, is_file, is_link in (
        ("../escape", True, False),
        ("/absolute", True, False),
        ("logs/link", True, True),
        ("logs/socket", False, False),
    ):
        _expect_value_error(
            lambda name=name, is_file=is_file, is_link=is_link: (
                runner.safe_download_member(
                    name,
                    is_file=is_file,
                    is_link=is_link,
                )
            ),
            "download",
        )


def test_deterministic_source_bundle_has_explicit_owned_inventory():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        root.mkdir()
        (root / "tinyvllm").mkdir()
        (root / "tinyvllm" / "runtime.py").write_text(
            "VALUE = 1\n",
            encoding="utf-8",
        )
        (root / "tools").mkdir()
        (root / "tools" / "worker.py").write_text(
            "print('worker')\n",
            encoding="utf-8",
        )
        first_tar = Path(temporary) / "first.tar"
        second_tar = Path(temporary) / "second.tar"
        owned = ("tinyvllm", "tools/worker.py")

        first = runner.build_deterministic_source_bundle(
            repo_root=root,
            owned_paths=owned,
            output_tar=first_tar,
        )
        second = runner.build_deterministic_source_bundle(
            repo_root=root,
            owned_paths=owned,
            output_tar=second_tar,
        )

        assert first == second
        assert first["source_tree_sha256"] != (
            contract.TP4_ROOT_SOURCE_TREE_SHA256
        )
        assert first["tar_sha256"] == second["tar_sha256"]
        assert first_tar.read_bytes() == second_tar.read_bytes()
        with tarfile.open(first_tar, "r") as archive:
            assert archive.getnames() == [
                "tinyvllm/runtime.py",
                "tools/worker.py",
            ]


def test_source_bundle_rejects_missing_or_unsafe_owned_paths():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        outside = root.parent / "outside-owned-source"
        outside.mkdir(exist_ok=True)
        (outside / "escaped.py").write_text(
            "ESCAPED = True\n",
            encoding="utf-8",
        )
        (root / "linked").symlink_to(outside, target_is_directory=True)
        _expect_value_error(
            lambda: runner.build_deterministic_source_bundle(
                repo_root=root,
                owned_paths=("missing.py",),
                output_tar=root / "bundle.tar",
            ),
            "owned source",
        )
        _expect_value_error(
            lambda: runner.build_deterministic_source_bundle(
                repo_root=root,
                owned_paths=("linked/escaped.py",),
                output_tar=root / "bundle.tar",
            ),
            "owned source",
        )
        _expect_value_error(
            lambda: runner.build_deterministic_source_bundle(
                repo_root=root,
                owned_paths=("../escape",),
                output_tar=root / "bundle.tar",
            ),
            "owned source",
        )


def test_source_bundle_rejects_output_inside_owned_inventory():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "tools").mkdir()
        (root / "tools" / "worker.py").write_text(
            "print('worker')\n",
            encoding="utf-8",
        )
        _expect_value_error(
            lambda: runner.build_deterministic_source_bundle(
                repo_root=root,
                owned_paths=("tools",),
                output_tar=root / "tools" / "bundle.tar",
            ),
            "output tar",
        )


def test_remote_stage_command_is_unique_source_bound_and_non_destructive():
    command = runner.build_remote_stage_command(
        run_tag="stage-test",
        source_tree_sha256="c" * 64,
        tar_sha256="d" * 64,
    )
    joined = " ".join(command)

    assert "stage-test" in joined
    assert "c" * 64 in joined
    assert "d" * 64 in joined
    assert "mkdir" in joined
    assert "test ! -e" in joined
    assert runner.REMOTE_PYTHON in joined
    assert "source tree mismatch" in joined
    assert "unsafe source tar member" in joined
    assert "issym" in joined
    assert "islnk" in joined
    assert "workload_manifest.json" in joined
    assert (
        contract.canonical_json_file_sha256(
            contract.workload_manifest_payload()
        )
        in joined
    )
    assert (
        f"mkdir {runner.REMOTE_ROOT}/stage-test"
        in joined
    )
    assert "rm -rf" not in joined
    assert "git " not in joined


def test_download_inventory_requires_exact_top_level_and_nested_dirs():
    runner.validate_download_inventory(
        top_level_files=contract.TOP_LEVEL_ARTIFACTS,
        top_level_directories=contract.NESTED_ARTIFACT_DIRECTORIES,
    )
    _expect_value_error(
        lambda: runner.validate_download_inventory(
            top_level_files=(
                *contract.TOP_LEVEL_ARTIFACTS,
                "extra.txt",
            ),
            top_level_directories=(
                contract.NESTED_ARTIFACT_DIRECTORIES
            ),
        ),
        "download inventory",
    )


def test_execute_benchmark_launch_wires_plan_authorization_and_executor():
    events = []
    ready = {
        "classification": "READY",
        "authorized": True,
    }
    verified_plan = {
        "run_tag": "strict-p1-smoke",
        "commands": {},
    }

    class PlanModule:
        PLAN_NAME = "remote_execution_plan.json"

        @staticmethod
        def build_remote_execution_plan(
            *,
            launch_plan,
            output_dir,
            local_prerequisites,
            local_model_manifest,
        ):
            events.append((
                "build_plan",
                launch_plan,
                Path(output_dir),
                Path(local_prerequisites),
                Path(local_model_manifest),
            ))
            Path(output_dir).mkdir()
            _write_json(
                Path(output_dir) / PlanModule.PLAN_NAME,
                verified_plan,
            )
            return verified_plan

        @staticmethod
        def verify_remote_execution_plan(path):
            events.append(("verify_plan", Path(path)))
            return verified_plan

    class AuthorizationModule:
        @staticmethod
        def produce_authorization(*, plan, output_path, nonce):
            events.append((
                "authorize",
                plan,
                Path(output_path),
                nonce,
            ))
            _write_json(Path(output_path), {"nonce": nonce})
            return {"nonce": nonce}

    class ExecutorModule:
        REQUIRED_EXECUTION_ENV = {
            "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian",
        }

        @staticmethod
        def execute_verified_plan_file(**kwargs):
            events.append(("execute", kwargs))
            return {
                "classification": "PASS",
                "run_tag": "strict-p1-smoke",
            }

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        prerequisites = root / "correctness_prerequisites.json"
        model_manifest = root / "model_manifest.json"
        prerequisites.write_text("{}\n", encoding="utf-8")
        model_manifest.write_text("{}\n", encoding="utf-8")

        result = runner.execute_benchmark_launch(
            mode="smoke",
            run_tag="strict-p1-smoke",
            prerequisites_path=prerequisites,
            local_model_manifest=model_manifest,
            remote_model_dir="/remote/model",
            remote_model_manifest="/remote/model_manifest.json",
            authorization_nonce="strict-p1-smoke-nonce",
            output_root=root / "runs",
            preflight_runner=lambda **kwargs: (
                events.append(("preflight", kwargs)) or ready
            ),
            launch_plan_builder=lambda **kwargs: (
                events.append(("launch_plan", kwargs))
                or {"run_tag": "strict-p1-smoke"}
            ),
            plan_module=PlanModule,
            authorization_module=AuthorizationModule,
            executor_module=ExecutorModule,
            command_runner=lambda **kwargs: None,
        )

        run_dir = root / "runs" / "strict-p1-smoke"
        assert result == {
            "classification": "PASS",
            "run_tag": "strict-p1-smoke",
        }
        assert [event[0] for event in events] == [
            "preflight",
            "launch_plan",
            "build_plan",
            "authorize",
            "execute",
        ]
        launch_kwargs = events[1][1]
        assert launch_kwargs["remote_prerequisites"] == (
            f"{runner.REMOTE_ROOT}/strict-p1-smoke/"
            "prerequisites/correctness_prerequisites.json"
        )
        execute_kwargs = events[-1][1]
        assert execute_kwargs["plan_path"] == (
            run_dir / "plan" / PlanModule.PLAN_NAME
        )
        assert execute_kwargs["authorization_path"] == (
            run_dir / "runtime" / "authorization.json"
        )
        assert execute_kwargs["consumed_authorization_path"] == (
            run_dir / "runtime" / "consumed_authorization.json"
        )
        assert execute_kwargs["output_path"] == (
            run_dir / "runtime" / "execution_receipt.json"
        )
        assert execute_kwargs["failure_path"] == (
            run_dir / "runtime" / "execution_failure.json"
        )
        assert execute_kwargs["execution_env"] == (
            ExecutorModule.REQUIRED_EXECUTION_ENV
        )


def test_main_launch_modes_require_explicit_runtime_identity():
    try:
        runner.main([
            "smoke",
            "--run-tag",
            "missing-runtime-identity",
        ])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("smoke accepted missing runtime identity")


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 hybrid-prefix remote runner tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
