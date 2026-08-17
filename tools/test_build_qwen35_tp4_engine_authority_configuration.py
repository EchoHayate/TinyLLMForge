from __future__ import annotations

from contextlib import redirect_stderr
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
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


builder = _load(
    "build_qwen35_tp4_engine_authority_configuration",
    "build_qwen35_tp4_engine_authority_configuration.py",
)
executor = _load(
    "qwen35_tp4_engine_executor_for_configuration_builder_test",
    "qwen35_tp4_engine_correctness_executor.py",
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_authority_owned_source_includes_cached_first_divergence_probe():
    assert (
        "tools/qwen35_tp4_cached_first_divergence_probe.py"
        in builder.AUTHORITY_OWNED_SOURCE_PATHS
    )


def test_builder_writes_workload_and_strict_configuration_atomically():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("print('a')\n")
        model_dir = root / "model"
        model_dir.mkdir()
        model_manifest = root / "model_manifest.json"
        model_manifest.write_text('{"files":{}}\n')
        output = root / "configuration"

        original_paths = builder.AUTHORITY_OWNED_SOURCE_PATHS
        builder.AUTHORITY_OWNED_SOURCE_PATHS = ("a.py",)
        try:
            result = builder.build_configuration(
                repo_root=repo,
                output_dir=output,
                model_dir=model_dir,
                model_manifest_path=model_manifest,
                model_fingerprint="qwen35-m8-authority",
                gpu_indices=(0, 1, 2, 3),
                dist_port=31001,
                master_port=31002,
                max_cache_entries=8,
                max_cache_bytes=1 << 30,
                timeout_s=600.0,
            )
        finally:
            builder.AUTHORITY_OWNED_SOURCE_PATHS = original_paths

        assert {path.name for path in output.iterdir()} == {
            builder.CONFIGURATION_NAME,
            builder.WORKLOAD_MANIFEST_NAME,
            builder.SOURCE_INVENTORY_NAME,
        }
        payload = json.loads(
            (output / builder.CONFIGURATION_NAME).read_text()
        )
        configuration = executor.ExecutorConfiguration(
            **{
                **{
                    name: value
                    for name, value in payload.items()
                    if name != "world_size"
                },
                "gpu_indices": tuple(payload["gpu_indices"]),
            }
        )
        assert configuration.to_payload() == payload
        assert payload["model_manifest_sha256"] == _sha256(
            model_manifest
        )
        assert payload["workload_manifest_sha256"] == _sha256(
            output / builder.WORKLOAD_MANIFEST_NAME
        )
        inventory = json.loads(
            (output / builder.SOURCE_INVENTORY_NAME).read_text()
        )
        assert inventory == {
            "owned_files": ["a.py"],
            "source_tree_sha256": payload["source_tree_sha256"],
        }
        assert result == payload


def test_remote_builder_uses_manifest_bound_model_without_local_weights():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("print('a')\n")
        remote_model_dir = "/remote/models/qwen35-2b/model"
        model_manifest = root / "model_manifest.json"
        model_manifest.write_text(
            json.dumps({
                "remote_model_dir": remote_model_dir,
                "files": {},
            })
            + "\n"
        )
        output = root / "configuration"

        original_paths = builder.AUTHORITY_OWNED_SOURCE_PATHS
        builder.AUTHORITY_OWNED_SOURCE_PATHS = ("a.py",)
        try:
            result = builder.build_remote_configuration(
                repo_root=repo,
                output_dir=output,
                model_manifest_path=model_manifest,
                remote_model_dir=remote_model_dir,
                model_fingerprint="remote-manifest-authority",
                gpu_indices=(0, 1, 2, 3),
                dist_port=31001,
                master_port=31002,
                max_cache_entries=8,
                max_cache_bytes=1 << 30,
                timeout_s=600.0,
            )
        finally:
            builder.AUTHORITY_OWNED_SOURCE_PATHS = original_paths

        payload = json.loads(
            (output / builder.CONFIGURATION_NAME).read_text()
        )
        assert result == payload
        assert payload["model_dir"] == remote_model_dir
        assert payload["model_manifest_path"] == str(
            model_manifest.resolve()
        )
        assert payload["model_manifest_sha256"] == _sha256(
            model_manifest
        )
        assert payload["model_fingerprint"] == (
            "remote-manifest-authority"
        )
        assert payload["gpu_indices"] == [0, 1, 2, 3]
        assert payload["dist_port"] == 31001
        assert payload["master_port"] == 31002
        assert payload["max_cache_entries"] == 8
        assert payload["max_cache_bytes"] == 1 << 30
        assert payload["timeout_s"] == 600.0
        assert payload["workload_manifest_sha256"] == _sha256(
            output / builder.WORKLOAD_MANIFEST_NAME
        )
        inventory = json.loads(
            (output / builder.SOURCE_INVENTORY_NAME).read_text()
        )
        assert inventory["source_tree_sha256"] == payload[
            "source_tree_sha256"
        ]


def test_builder_rejects_existing_target_and_invalid_model_inputs():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "configuration"
        output.mkdir()
        try:
            builder.build_configuration(
                repo_root=root,
                output_dir=output,
                model_dir=root / "model",
                model_manifest_path=root / "manifest.json",
                model_fingerprint="x",
                gpu_indices=(0, 1, 2, 3),
                dist_port=31001,
                master_port=31002,
                max_cache_entries=8,
                max_cache_bytes=1024,
                timeout_s=10.0,
            )
        except ValueError as error:
            assert "already exists" in str(error)
        else:
            raise AssertionError("existing configuration was overwritten")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("x\n")
        manifest = root / "manifest.json"
        manifest.write_text("{}\n")
        original_paths = builder.AUTHORITY_OWNED_SOURCE_PATHS
        builder.AUTHORITY_OWNED_SOURCE_PATHS = ("a.py",)
        try:
            try:
                builder.build_configuration(
                    repo_root=repo,
                    output_dir=root / "output",
                    model_dir=root / "missing-model",
                    model_manifest_path=manifest,
                    model_fingerprint="x",
                    gpu_indices=(0, 1, 2, 3),
                    dist_port=31001,
                    master_port=31002,
                    max_cache_entries=8,
                    max_cache_bytes=1024,
                    timeout_s=10.0,
                )
            except ValueError as error:
                assert "model directory" in str(error)
            else:
                raise AssertionError("missing model directory was accepted")
        finally:
            builder.AUTHORITY_OWNED_SOURCE_PATHS = original_paths


def _expect_remote_builder_failure(
    *,
    manifest_text,
    remote_model_dir,
    symlink_manifest=False,
    source_paths=("a.py",),
):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("x\n")
        manifest_target = root / "manifest-target.json"
        manifest_target.write_text(manifest_text)
        manifest = root / "manifest.json"
        if symlink_manifest:
            manifest.symlink_to(manifest_target)
        else:
            manifest = manifest_target
        output = root / "output"
        original_paths = builder.AUTHORITY_OWNED_SOURCE_PATHS
        builder.AUTHORITY_OWNED_SOURCE_PATHS = source_paths
        try:
            try:
                builder.build_remote_configuration(
                    repo_root=repo,
                    output_dir=output,
                    model_manifest_path=manifest,
                    remote_model_dir=remote_model_dir,
                    model_fingerprint="x",
                    gpu_indices=(0, 1, 2, 3),
                    dist_port=31001,
                    master_port=31002,
                    max_cache_entries=8,
                    max_cache_bytes=1024,
                    timeout_s=10.0,
                )
            except ValueError:
                pass
            else:
                raise AssertionError("invalid remote configuration accepted")
        finally:
            builder.AUTHORITY_OWNED_SOURCE_PATHS = original_paths
        assert not output.exists()


def test_remote_builder_rejects_invalid_manifest_paths_without_output():
    valid_remote = "/remote/models/qwen35/model"
    cases = (
        ("not-json\n", valid_remote),
        ('{"files":{}}\n', valid_remote),
        ('{"remote_model_dir":""}\n', valid_remote),
        ('{"remote_model_dir":"relative/model"}\n', "relative/model"),
        (
            '{"remote_model_dir":"/remote/models/other"}\n',
            valid_remote,
        ),
        ('{"remote_model_dir":7}\n', valid_remote),
    )
    for manifest_text, remote_model_dir in cases:
        _expect_remote_builder_failure(
            manifest_text=manifest_text,
            remote_model_dir=remote_model_dir,
        )
    _expect_remote_builder_failure(
        manifest_text=json.dumps({
            "remote_model_dir": valid_remote,
        }),
        remote_model_dir=valid_remote,
        symlink_manifest=True,
    )


def test_remote_builder_failure_removes_partial_output():
    remote_model_dir = "/remote/models/qwen35/model"
    _expect_remote_builder_failure(
        manifest_text=json.dumps({
            "remote_model_dir": remote_model_dir,
        }),
        remote_model_dir=remote_model_dir,
        source_paths=("missing.py",),
    )


def _cli_arguments(root, output, manifest):
    return [
        "--repo-root",
        str(root / "repo"),
        "--output-dir",
        str(output),
        "--model-manifest",
        str(manifest),
        "--model-fingerprint",
        "remote-manifest-authority",
        "--gpu-indices",
        "0,1,2,3",
        "--dist-port",
        "31001",
        "--master-port",
        "31002",
        "--max-cache-entries",
        "8",
        "--max-cache-bytes",
        str(1 << 30),
        "--timeout-s",
        "600",
    ]


def test_cli_selects_exactly_one_local_or_remote_model_mode():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "repo"
        repo.mkdir()
        (repo / "a.py").write_text("x\n")
        remote_model_dir = "/remote/models/qwen35/model"
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "remote_model_dir": remote_model_dir,
        }))
        original_paths = builder.AUTHORITY_OWNED_SOURCE_PATHS
        builder.AUTHORITY_OWNED_SOURCE_PATHS = ("a.py",)
        try:
            output = root / "remote-output"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                assert builder.main(
                    _cli_arguments(root, output, manifest)
                    + ["--remote-model-dir", remote_model_dir]
                ) == 0
            assert json.loads(stdout.getvalue())["model_dir"] == (
                remote_model_dir
            )

            local_output = root / "local-output"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                assert builder.main(
                    _cli_arguments(root, local_output, manifest)
                    + ["--model-dir", str(root)]
                ) == 0
            assert json.loads(stdout.getvalue())["model_dir"] == str(
                root.resolve()
            )

            invalid_cases = (
                [],
                [
                    "--model-dir",
                    str(root),
                    "--remote-model-dir",
                    remote_model_dir,
                ],
            )
            for extra in invalid_cases:
                with redirect_stderr(io.StringIO()):
                    try:
                        builder.main(
                            _cli_arguments(
                                root,
                                root / f"invalid-{len(extra)}",
                                manifest,
                            )
                            + extra
                        )
                    except SystemExit as error:
                        assert error.code == 2
                    else:
                        raise AssertionError(
                            "ambiguous model mode was accepted"
                        )
        finally:
            builder.AUTHORITY_OWNED_SOURCE_PATHS = original_paths


def test_builder_failure_leaves_no_partial_output():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo = root / "repo"
        repo.mkdir()
        manifest = root / "manifest.json"
        manifest.write_text("{}\n")
        output = root / "output"
        original_paths = builder.AUTHORITY_OWNED_SOURCE_PATHS
        builder.AUTHORITY_OWNED_SOURCE_PATHS = ("missing.py",)
        try:
            try:
                builder.build_configuration(
                    repo_root=repo,
                    output_dir=output,
                    model_dir=root,
                    model_manifest_path=manifest,
                    model_fingerprint="x",
                    gpu_indices=(0, 1, 2, 3),
                    dist_port=31001,
                    master_port=31002,
                    max_cache_entries=8,
                    max_cache_bytes=1024,
                    timeout_s=10.0,
                )
            except ValueError:
                pass
            else:
                raise AssertionError("missing source file was accepted")
        finally:
            builder.AUTHORITY_OWNED_SOURCE_PATHS = original_paths
        assert not output.exists()


def test_authority_source_inventory_contains_remote_driver_dependencies():
    required = {
        "tools/run_qwen35_tp4_engine_correctness_authority.py",
        "tools/verify_qwen35_tp4_engine_correctness_authority.py",
        "tools/qwen35_tp4_engine_correctness_executor.py",
        "tools/qwen35_tp4_engine_correctness_producer.py",
        "tools/qwen35_tp4_engine_backend_session.py",
        "tools/qwen35_tp4_engine_reference_tokens_producer.py",
        "tools/qwen35_tp4_engine_official_reference_executor.py",
        "tools/qwen35_tp4_cached_continuation_correctness_contract.py",
        "tools/qwen35_tp4_cached_continuation_correctness_executor.py",
        "tools/qwen35_tp4_cached_continuation_backend_session.py",
        "tools/qwen35_tp4_cached_partition_diagnostic.py",
        "tools/qwen35_tp4_cached_continuation_correctness_producer.py",
        "tools/run_qwen35_tp4_cached_continuation_authority.py",
        "tools/verify_qwen35_tp4_cached_continuation_correctness_gate.py",
        "tools/qwen35_tp4_engine_remote_execution_plan.py",
        "tools/qwen35_tp4_engine_remote_execution_receipt.py",
        "tools/qwen35_tp4_engine_remote_execution_executor.py",
        "tools/qwen35_tp4_engine_remote_execution_source_contract.py",
        "tools/qwen35_tp4_engine_remote_execution_authorization.py",
        "tools/qwen35_tp4_engine_remote_subprocess_adapter.py",
        "tools/qwen35_tp4_cached_continuation_remote_execution_plan.py",
        "tools/qwen35_tp4_cached_continuation_remote_execution_receipt.py",
        "tools/qwen35_tp4_cached_continuation_remote_execution_executor.py",
    }
    assert required.issubset(set(builder.AUTHORITY_OWNED_SOURCE_PATHS))


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 Engine configuration builder tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
