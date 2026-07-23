"""Tests for the non-destructive Qwen3.5 hybrid-state remote runner.

Run: python3 tools/test_run_qwen35_hybrid_state_gate_remote.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
RUNNER_PATH = THIS_DIR / "run_qwen35_hybrid_state_gate_remote.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "qwen35_hybrid_state_remote_runner_under_test",
        os.fspath(RUNNER_PATH),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expect_value_error(callable_, message):
    try:
        callable_()
    except ValueError as exc:
        assert message in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_runner_binds_exact_remote_identity_and_modes():
    source = RUNNER_PATH.read_text()
    for required in (
        "sitian@10.232.195.203",
        "/tmp/ssh-sitian-10.232.195.203",
        "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python",
        "CUDA_VISIBLE_DEVICES",
        "qwen35-hybrid-state-runs",
    ):
        assert required in source
    for mode in (
        "preflight",
        "acquire",
        "smoke",
        "canonical",
        "download-only",
        "verify-only",
    ):
        assert f'"{mode}"' in source


def test_runner_forbids_remote_mutation_and_cleanup():
    source = RUNNER_PATH.read_text()
    for forbidden in (
        "rsync ",
        "pkill",
        "killall",
        "rm -rf",
        "git checkout",
        "git reset",
        "git clean",
    ):
        assert forbidden not in source


def test_owned_source_files_are_exact_and_run_tags_are_safe():
    runner = _load_runner()
    assert runner.OWNED_SOURCE_FILES == (
        "tools/qwen35_hybrid_state_contract.py",
        "tools/qwen35_hybrid_state_probe.py",
        "tools/verify_qwen35_hybrid_state_gate.py",
        "tools/run_qwen35_hybrid_state_gate_remote.py",
        "tools/test_qwen35_hybrid_state_contract.py",
        "tools/test_qwen35_hybrid_state_probe.py",
        "tools/test_verify_qwen35_hybrid_state_gate.py",
        "tools/test_run_qwen35_hybrid_state_gate_remote.py",
    )
    assert runner.validate_run_tag("qwen35_20260723-a") == (
        "qwen35_20260723-a"
    )
    for invalid in ("", "../escape", "a/b", "a b", "a.b", "中文"):
        _expect_value_error(
            lambda value=invalid: runner.validate_run_tag(value),
            "run tag",
        )


def test_ssh_command_uses_control_master_and_batch_mode():
    runner = _load_runner()
    command = runner.build_ssh_command(["python3", "-V"])
    assert command[:2] == ["ssh", "-S"]
    assert runner.SSH_CONTROL_PATH in command
    assert "BatchMode=yes" in command
    assert command[-2] == runner.REMOTE_TARGET
    assert command[-1] == "python3 -V"


def test_preflight_is_read_only_and_computes_frozen_peak_bytes():
    runner = _load_runner()
    result = runner.evaluate_disk_preflight(
        declared_model_file_bytes=4 * runner.GIB,
        free_bytes=11 * runner.GIB,
    )
    assert result["required_bytes"] == (
        (4 * runner.GIB * 2)
        + (512 * runner.MIB)
        + (2 * runner.GIB)
    )
    assert result["can_acquire"] is True
    assert result["classification_detail"] is None
    assert set(result) == {
        "declared_model_file_bytes",
        "free_bytes",
        "required_bytes",
        "can_acquire",
        "classification_detail",
    }


def test_preflight_summary_has_closed_status_and_task6_fields():
    runner = _load_runner()
    payload = {
        "resolved_revision": "a" * 40,
        "declared_model_file_bytes": 4 * runner.GIB,
        "disk_preflight": {
            "free_bytes": 11 * runner.GIB,
            "required_bytes": 10 * runner.GIB + 512 * runner.MIB,
            "can_acquire": True,
            "classification_detail": None,
        },
        "packages": {
            "torch": "2.4",
            "transformers": "5.8",
            "huggingface_hub": "0.34",
        },
        "gpu_processes": [],
        "checked_cache_roots": ["/cache/hub"],
        "candidate_snapshots": [],
    }
    summary = runner.build_preflight_summary(payload)
    assert summary["status"] == "READY_TO_ACQUIRE"
    assert summary["required_acquisition_peak_bytes"] == (
        payload["disk_preflight"]["required_bytes"]
    )
    assert summary["runtime"]["packages"] == payload["packages"]
    assert summary["gpu_processes"] == []
    assert summary["checked_cache_roots"] == ["/cache/hub"]
    assert summary["candidate_snapshots"] == []
    blocked = dict(payload)
    blocked["disk_preflight"] = {
        **payload["disk_preflight"],
        "can_acquire": False,
        "classification_detail": "INCOMPLETE_RESOURCE_BLOCKED",
    }
    assert runner.build_preflight_summary(blocked)["status"] == (
        "INCOMPLETE_RESOURCE_BLOCKED"
    )


def test_model_metadata_timeout_preserves_fail_closed_preflight_fields():
    runner = _load_runner()
    remote_payload = {
        "resolved_revision": None,
        "siblings": [],
        "metadata_error": "ConnectTimeout: [Errno 110] timed out",
        "free_bytes": 12 * runner.GIB,
        "packages": {
            "torch": "2.4",
            "transformers": "5.8",
            "huggingface_hub": "0.34",
        },
        "gpu_processes": ["1234, python, 1024"],
        "cuda_visible_devices": None,
        "host": "10.232.195.203",
        "observed_hostname": "remote-host",
        "user": "sitian",
        "python_version": "3.11.0",
        "gpu_name": "GPU",
        "gpu_uuid": "GPU-uuid",
        "driver_version": "550",
        "cuda_runtime_version": "12.4",
        "checked_cache_roots": ["/cache/hub"],
        "candidate_snapshots": ["/cache/hub/snapshot"],
    }

    def command_runner(_command, **_kwargs):
        return type("Result", (), {
            "returncode": 0,
            "stdout": json.dumps(remote_payload),
            "stderr": "",
        })()

    payload = runner.run_remote_preflight(
        "qwen35-metadata-timeout",
        command_runner=command_runner,
    )
    summary = runner.build_preflight_summary(payload)
    assert summary == {
        "status": "INCOMPLETE_MODEL_METADATA",
        "resolved_revision": None,
        "declared_model_file_bytes": 0,
        "free_bytes": 12 * runner.GIB,
        "required_acquisition_peak_bytes": None,
        "runtime": {
            "python_executable": runner.REMOTE_PYTHON,
            "packages": remote_payload["packages"],
        },
        "gpu_processes": remote_payload["gpu_processes"],
        "checked_cache_roots": remote_payload["checked_cache_roots"],
        "candidate_snapshots": remote_payload["candidate_snapshots"],
        "failure_detail": remote_payload["metadata_error"],
    }


def test_insufficient_disk_stops_before_download_or_gpu():
    runner = _load_runner()
    result = runner.evaluate_disk_preflight(
        declared_model_file_bytes=4 * runner.GIB,
        free_bytes=8 * runner.GIB,
    )
    assert result["can_acquire"] is False
    assert result["classification_detail"] == (
        "INCOMPLETE_RESOURCE_BLOCKED"
    )


def test_revision_and_model_allow_list_are_immutable_and_bounded():
    runner = _load_runner()
    revision = "a" * 40
    assert runner.validate_resolved_revision(revision) == revision
    _expect_value_error(
        lambda: runner.validate_resolved_revision("main"),
        "40-hex",
    )
    siblings = [
        {"rfilename": "config.json", "size": 10},
        {"rfilename": "tokenizer.json", "size": 20},
        {"rfilename": "model-00001-of-00002.safetensors", "size": 30},
        {"rfilename": "model-00002-of-00002.safetensors", "size": 40},
        {"rfilename": "model.safetensors.index.json", "size": 5},
        {"rfilename": "modeling_qwen3_5.py", "size": 7},
        {"rfilename": "README.md", "size": 999},
    ]
    inventory = runner.build_model_file_inventory(siblings)
    assert inventory["declared_model_file_bytes"] == 112
    assert inventory["allow_patterns"] == [
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "modeling_qwen3_5.py",
        "tokenizer.json",
    ]
    assert "README.md" not in inventory["allow_patterns"]


def test_snapshot_download_script_uses_exact_immutable_arguments():
    runner = _load_runner()
    script = runner.build_snapshot_download_script(
        resolved_revision="a" * 40,
        remote_run_dir="/safe/run",
        allow_patterns=["config.json", "model.safetensors"],
    )
    for required in (
        "snapshot_download(",
        'repo_id="Qwen/Qwen3.5-2B"',
        'revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"',
        'local_dir="/safe/run/model"',
        "local_dir_use_symlinks=False",
        'allow_patterns=["config.json","model.safetensors"]',
    ):
        assert required in script


def test_acquired_inventory_requires_indexed_shards_and_hashes():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temporary:
        model_dir = Path(temporary)
        (model_dir / "config.json").write_text("{}")
        (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"a")
        (model_dir / "model-00002-of-00002.safetensors").write_bytes(b"b")
        (model_dir / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            },
        }))
        result = runner.hash_and_validate_model_files(
            model_dir,
            [
                "config.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model.safetensors.index.json",
            ],
        )
        assert set(result) == {
            "config.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
        }
        assert all(len(entry["sha256"]) == 64 for entry in result.values())
        (model_dir / "model-00002-of-00002.safetensors").unlink()
        _expect_value_error(
            lambda: runner.hash_and_validate_model_files(
                model_dir,
                [
                    "config.json",
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                    "model.safetensors.index.json",
                ],
            ),
            "missing model file",
        )


def test_remote_model_hash_script_is_revision_bound_and_rejects_extras():
    runner = _load_runner()
    script = runner.build_remote_model_hash_script(
        remote_model_dir="/safe/model",
        expected_files=["config.json", "model.safetensors"],
    )
    assert "unexpected model file" in script
    assert "missing model file" in script
    assert 'root=pathlib.Path("/safe/model")' in script
    assert 'expected=["config.json","model.safetensors"]' in script
    assert ".cache/huggingface" in script
    indexed = runner.build_remote_model_hash_script(
        remote_model_dir="/safe/model",
        expected_files=[
            "model.safetensors.index.json",
            "model-00001-of-00001.safetensors",
        ],
    )
    assert "missing indexed shard" in indexed
    assert "weight_map" in indexed


def test_model_manifest_binds_remote_path_revision_and_file_hashes():
    runner = _load_runner()
    files = {
        "config.json": {"size": 2, "sha256": "c" * 64},
        "model.safetensors": {"size": 3, "sha256": "d" * 64},
    }
    manifest = runner.build_model_manifest(
        resolved_revision="b" * 40,
        remote_model_dir="/safe/model",
        files=files,
    )
    assert manifest == {
        "schema_version": 1,
        "repository": runner.MODEL_REPOSITORY,
        "resolved_revision": "b" * 40,
        "local_path": "/safe/model",
        "remote_model_dir": "/safe/model",
        "files": files,
        "total_weight_bytes": 3,
        "trust_remote_code": False,
    }


def test_acquisition_failure_is_preserved_as_incomplete_manifest():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary)

        def acquire():
            raise RuntimeError("download failed")

        result = runner.run_acquisition_mode(
            run_dir=run_dir,
            source_commit="a" * 40,
            resolved_revision="b" * 40,
            acquire=acquire,
        )
        assert result["classification"] == "INCOMPLETE"
        assert result["failure_kind"] == "INCOMPLETE_ACQUISITION_FAILURE"
        assert "download failed" in result["failure_detail"]
        assert (run_dir / "manifest.json").is_file()


def test_model_manifest_is_enriched_from_observed_architecture():
    runner = _load_runner()
    base = runner.build_model_manifest(
        resolved_revision="b" * 40,
        remote_model_dir="/safe/model",
        files={"model.safetensors": {"size": 3, "sha256": "d" * 64}},
    )
    architecture = {
        "config_class": "Qwen3_5Config",
        "model_class": "Qwen3_5ForCausalLM",
        "tokenizer_class": "Qwen2TokenizerFast",
        "tokenizer_vocab_size": 151936,
        "parameter_dtypes": {"bfloat16": 2_000_000_000},
    }
    enriched = runner.enrich_model_manifest(base, architecture)
    for key, value in architecture.items():
        assert enriched[key] == value
    assert enriched["requested_dtype"] == "auto"


def test_environment_manifest_uses_exact_worker_ports_and_remote_identity():
    runner = _load_runner()
    runtime = {
        "host": "10.232.195.203",
        "user": "sitian",
        "gpu_name": "NVIDIA H100",
        "gpu_uuid": "GPU-1",
        "driver_version": "550",
        "cuda_runtime_version": "12.1",
        "python_version": "3.11",
        "packages": {"torch": "2.4", "transformers": "5.8"},
        "gpu_processes": [],
    }
    attempt = {
        "tinyvllm_dist_port": 40101,
        "master_port": 40102,
    }
    environment = runner.build_environment_manifest(runtime, attempt)
    assert environment["host"] == "10.232.195.203"
    assert environment["user"] == "sitian"
    assert environment["python_executable"] == runner.REMOTE_PYTHON
    assert environment["environment"] == {
        "CUDA_VISIBLE_DEVICES": "0",
        "TINYVLLM_DIST_PORT": "40101",
        "MASTER_PORT": "40102",
    }
    assert environment["torch_version"] == "2.4"
    assert environment["transformers_version"] == "5.8"
    _expect_value_error(
        lambda: runner.build_environment_manifest(
            {**runtime, "user": "other"},
            attempt,
        ),
        "remote runtime user",
    )


def test_port_pairs_are_globally_unique_and_distinct():
    runner = _load_runner()
    values = iter((40101, 40102, 40103, 40104, 40105, 40106))
    pairs = runner.allocate_unique_port_pairs(
        3,
        allocator=lambda: next(values),
    )
    assert len({port for pair in pairs for port in pair}) == 6
    assert all(dist != master for dist, master in pairs)


def test_port_allocator_rejects_duplicates_and_invalid_ports():
    runner = _load_runner()
    duplicate_values = iter((40101, 40101))
    _expect_value_error(
        lambda: runner.allocate_unique_port_pairs(
            1,
            allocator=lambda: next(duplicate_values),
        ),
        "unique ports",
    )
    _expect_value_error(
        lambda: runner.allocate_unique_port_pairs(
            1,
            allocator=lambda: 70000,
        ),
        "valid port",
    )


def test_only_exact_eaddrinuse_is_retryable_and_attempts_are_capped():
    runner = _load_runner()
    assert runner.is_retryable_port_collision(1, "EADDRINUSE") is True
    assert (
        runner.is_retryable_port_collision(
            1,
            "prefix EADDRINUSE suffix",
        )
        is True
    )
    assert (
        runner.is_retryable_port_collision(
            1,
            "Address already in use",
        )
        is False
    )
    assert runner.is_retryable_port_collision(0, "EADDRINUSE") is False
    assert runner.is_retryable_port_collision(3, "EADDRINUSE") is False
    assert runner.MAX_PORT_ATTEMPTS == 3


def test_safe_artifact_paths_and_chunk_ranges_cover_zero_byte_files():
    runner = _load_runner()
    assert runner.validate_artifact_path("stdout/worker.log") == (
        "stdout/worker.log"
    )
    for invalid in ("", "/absolute", "../escape", "a/../../b"):
        _expect_value_error(
            lambda value=invalid: runner.validate_artifact_path(value),
            "artifact path",
        )
    assert list(runner.iter_download_ranges(0, chunk_size=4)) == []
    assert list(runner.iter_download_ranges(10, chunk_size=4)) == [
        (0, 4),
        (4, 4),
        (8, 2),
    ]
    assert runner.download_artifacts.__kwdefaults__["remote_subdir"] == (
        "artifacts"
    )


def test_mode_policies_isolate_verify_and_download_only():
    runner = _load_runner()
    assert runner.mode_policy("verify-only") == {
        "uses_ssh": False,
        "stages_source": False,
        "launches_process": False,
        "downloads": False,
        "verifies": True,
    }
    assert runner.mode_policy("download-only") == {
        "uses_ssh": True,
        "stages_source": False,
        "launches_process": False,
        "downloads": True,
        "verifies": False,
    }
    assert runner.mode_policy("preflight")["launches_process"] is False
    assert runner.mode_policy("acquire")["launches_process"] is False
    assert runner.mode_policy("canonical")["launches_process"] is True


def test_local_verifier_selects_smoke_or_canonical_domain():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temporary:
        repo_root = Path(temporary)
        verifier = repo_root / "tools" / (
            "verify_qwen35_hybrid_state_gate.py"
        )
        verifier.parent.mkdir()
        verifier.write_text("")
        run_dir = repo_root / "run"
        commands = []

        def command_runner(command, **_kwargs):
            commands.append(command)
            return type("Result", (), {
                "returncode": 0,
                "stdout": "{}",
                "stderr": "",
            })()

        runner.run_local_verifier(
            repo_root,
            run_dir,
            domain="smoke",
            command_runner=command_runner,
        )
        assert commands[0][-2:] == ["--domain", "smoke"]
        commands.clear()
        runner.run_local_verifier(
            repo_root,
            run_dir,
            domain="canonical",
            command_runner=command_runner,
        )
        assert "--domain" not in commands[0]


def test_process_retry_uses_fresh_pairs_and_preserves_attempts():
    runner = _load_runner()
    pairs = iter(((40101, 40102), (40103, 40104)))
    calls = []

    def launch(attempt, tiny_port, master_port):
        calls.append((attempt, tiny_port, master_port))
        if attempt == 1:
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": "EADDRINUSE",
            }
        return {"exit_code": 0, "stdout": "ok", "stderr": ""}

    result = runner.run_with_port_retries(
        "canonical",
        launch=launch,
        pair_allocator=lambda: next(pairs),
    )
    assert result["success"] is True
    assert calls == [(1, 40101, 40102), (2, 40103, 40104)]
    assert len(result["attempts"]) == 2


def test_non_collision_failure_is_not_retried():
    runner = _load_runner()
    calls = []

    def launch(attempt, tiny_port, master_port):
        calls.append(attempt)
        return {
            "exit_code": 1,
            "stdout": "",
            "stderr": "model load failed",
        }

    result = runner.run_with_port_retries(
        "canonical",
        launch=launch,
        pair_allocator=lambda: (40101, 40102),
    )
    assert result["success"] is False
    assert calls == [1]


def test_partial_manifest_preserves_available_files():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary)
        (run_dir / "stdout").mkdir()
        (run_dir / "stdout" / "worker.log").write_text("partial\n")
        manifest = runner.write_incomplete_manifest(
            run_dir,
            source_commit="a" * 40,
            model_revision="b" * 40,
            failure_kind="INCOMPLETE_WORKER_FAILURE",
            failure_detail="worker exited 1",
        )
        assert manifest["classification"] == "INCOMPLETE"
        assert manifest["failure_kind"] == "INCOMPLETE_WORKER_FAILURE"
        assert [entry["path"] for entry in manifest["artifacts"]] == [
            "stdout/worker.log",
        ]
        assert json.loads((run_dir / "manifest.json").read_text()) == manifest


def test_source_clean_check_uses_exact_owned_paths():
    runner = _load_runner()
    commands = []

    def command_runner(command, **_kwargs):
        commands.append(command)
        return type("Result", (), {
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        })()

    result = runner.require_clean_owned_source(
        Path("/repo"),
        command_runner=command_runner,
    )
    assert result == "HEAD"
    assert commands[0][:4] == [
        "git",
        "status",
        "--porcelain",
        "--",
    ]
    assert tuple(commands[0][4:]) == runner.OWNED_SOURCE_FILES
    assert commands[1] == ["git", "rev-parse", "HEAD"]


def test_smoke_domain_is_explicit_and_canonical_requires_bound_smoke():
    runner = _load_runner()
    assert runner.SMOKE_CASE_IDS == (
        "environment_preflight",
        "architecture_verification",
        "same_path_repeatability__cached_repeatability__p17__r0__c17",
        "same_path_repeatability__cached_repeatability__p17__r1__c17",
        "one_shot_vs_cached__one_shot_vs_cached__p17__r0__c17",
        "state_export_import__state_export_import__p17__r0__c17",
        "post_run_audit",
    )
    arguments = runner.parse_arguments([
        "canonical",
        "--run-tag",
        "canonical-a",
        "--resolved-revision",
        "a" * 40,
        "--smoke-run-tag",
        "smoke-a",
    ])
    assert arguments.smoke_run_tag == "smoke-a"
    _expect_value_error(
        lambda: runner.validate_mode_arguments(
            mode="canonical",
            resolved_revision="a" * 40,
            smoke_run_tag=None,
        ),
        "smoke-run-tag",
    )


def test_smoke_probe_script_filters_exact_cases_without_changing_probe():
    runner = _load_runner()
    script = runner.build_smoke_probe_script(
        remote_source_dir="/safe/run/source",
        remote_model_dir="/safe/model",
        remote_artifact_dir="/safe/run/artifacts",
        contract_sha256="c" * 64,
    )
    assert "original_build_case_matrix" in script
    assert json.dumps(
        list(runner.SMOKE_CASE_IDS),
        separators=(",", ":"),
    ) in script
    assert "probe.main([" in script
    assert '"run-canonical"' in script


def test_smoke_probe_script_patches_the_contract_instance_used_by_probe():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        tools = root / "tools"
        tools.mkdir()
        (tools / "qwen35_hybrid_state_contract.py").write_text(
            "class Case:\n"
            " def __init__(self,case_id):self.case_id=case_id\n"
            "def build_case_matrix():\n"
            " return tuple(Case(x) for x in "
            f"{list(runner.SMOKE_CASE_IDS) + ['unexpected']!r})\n"
        )
        output = root / "observed.json"
        (tools / "qwen35_hybrid_state_probe.py").write_text(
            "import importlib.util,json,pathlib,sys\n"
            "p=pathlib.Path(__file__).with_name("
            "'qwen35_hybrid_state_contract.py')\n"
            "s=importlib.util.spec_from_file_location("
            "'qwen35_hybrid_state_contract_for_probe',p)\n"
            "contract=importlib.util.module_from_spec(s)\n"
            "sys.modules[s.name]=contract\n"
            "s.loader.exec_module(contract)\n"
            "def main(argv):\n"
            f" pathlib.Path({os.fspath(output)!r}).write_text("
            "json.dumps([x.case_id for x in "
            "contract.build_case_matrix()]))\n"
            " return 0\n"
        )
        script = runner.build_smoke_probe_script(
            remote_source_dir=os.fspath(root),
            remote_model_dir="/safe/model",
            remote_artifact_dir="/safe/artifacts",
            contract_sha256="c" * 64,
        )
        try:
            exec(compile(script, "<smoke-wrapper>", "exec"), {})
        except SystemExit as exc:
            assert exc.code == 0
        assert json.loads(output.read_text()) == list(
            runner.SMOKE_CASE_IDS
        )


def test_smoke_audit_rejects_missing_cases_and_emits_only_smoke_pass():
    runner = _load_runner()
    rows = [
        {
            "case_id": case_id,
            "complete": True,
            "failure_kind": None,
        }
        for case_id in runner.SMOKE_CASE_IDS
    ]
    result = runner.audit_smoke_case_rows(rows)
    assert result == {
        "classification": "SMOKE_PASS",
        "case_ids": list(runner.SMOKE_CASE_IDS),
        "claim_boundary": (
            "Smoke compatibility prerequisite only; not canonical GO."
        ),
    }
    _expect_value_error(
        lambda: runner.audit_smoke_case_rows(rows[:-1]),
        "smoke case domain",
    )
    tampered = [dict(row) for row in rows]
    tampered[0]["complete"] = False
    _expect_value_error(
        lambda: runner.audit_smoke_case_rows(tampered),
        "incomplete smoke case",
    )


def test_model_snapshot_selection_is_revision_and_hash_bound():
    runner = _load_runner()
    revision = "b" * 40
    files = {"config.json": {"size": 2, "sha256": "c" * 64}}
    candidates = [
        {
            "run_tag": "wrong",
            "repository": runner.MODEL_REPOSITORY,
            "resolved_revision": "d" * 40,
            "files": files,
            "remote_model_dir": "/runs/wrong/model",
        },
        {
            "run_tag": "acquired",
            "repository": runner.MODEL_REPOSITORY,
            "resolved_revision": revision,
            "files": files,
            "remote_model_dir": "/runs/acquired/model",
        },
    ]
    selected = runner.select_verified_model_snapshot(
        candidates,
        resolved_revision=revision,
    )
    assert selected["run_tag"] == "acquired"
    _expect_value_error(
        lambda: runner.select_verified_model_snapshot(
            candidates[:1],
            resolved_revision=revision,
        ),
        "verified model snapshot",
    )


def test_remote_model_snapshot_is_rehashed_before_gpu_launch():
    runner = _load_runner()
    files = {
        "config.json": {"size": 2, "sha256": "c" * 64},
        "model.safetensors": {"size": 3, "sha256": "d" * 64},
    }
    snapshot = {
        "remote_model_dir": "/safe/model",
        "files": files,
    }

    def command_runner(_command, **_kwargs):
        return type("Result", (), {
            "returncode": 0,
            "stdout": json.dumps(files),
            "stderr": "",
        })()

    assert runner.verify_remote_model_snapshot(
        snapshot,
        command_runner=command_runner,
    ) == files

    def drifted(_command, **_kwargs):
        changed = dict(files)
        changed["config.json"] = {
            "size": 2,
            "sha256": "e" * 64,
        }
        return type("Result", (), {
            "returncode": 0,
            "stdout": json.dumps(changed),
            "stderr": "",
        })()

    _expect_value_error(
        lambda: runner.verify_remote_model_snapshot(
            snapshot,
            command_runner=drifted,
        ),
        "model snapshot hash mismatch",
    )


def test_remote_source_tests_run_before_worker_with_no_gpu():
    runner = _load_runner()
    canonical = runner.build_remote_execution_plan(
        mode="canonical",
        remote_source_dir="/safe/run/source",
        remote_model_dir="/safe/model",
        remote_artifact_dir="/safe/run/artifacts",
        contract_sha256="c" * 64,
    )
    assert [step["name"] for step in canonical["source_tests"]] == [
        "contract",
        "probe",
        "verifier",
        "runner",
    ]
    assert all(
        step["environment"]["CUDA_VISIBLE_DEVICES"] == ""
        for step in canonical["source_tests"]
    )
    assert canonical["worker"]["command"] == [
        runner.REMOTE_PYTHON,
        "/safe/run/source/tools/qwen35_hybrid_state_probe.py",
        "run-canonical",
        "--model-dir",
        "/safe/model",
        "--run-dir",
        "/safe/run/artifacts",
        "--contract-sha256",
        "c" * 64,
    ]
    smoke = runner.build_remote_execution_plan(
        mode="smoke",
        remote_source_dir="/safe/run/source",
        remote_model_dir="/safe/model",
        remote_artifact_dir="/safe/run/artifacts",
        contract_sha256="c" * 64,
    )
    assert smoke["worker"]["command"] == [
        runner.REMOTE_PYTHON,
        "-c",
        smoke["worker"]["command"][2],
    ]
    assert "probe.main([" in smoke["worker"]["command"][2]
    assert canonical["worker"]["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert "TINYVLLM_DIST_PORT" not in canonical["worker"]["environment"]
    assert "MASTER_PORT" not in canonical["worker"]["environment"]


def test_worker_launch_records_exact_environment_command_and_fresh_ports():
    runner = _load_runner()
    commands = []

    def command_runner(command, **kwargs):
        commands.append((command, kwargs))
        return type("Result", (), {
            "returncode": 0,
            "stdout": "worker summary\n",
            "stderr": "",
        })()

    result = runner.launch_remote_worker(
        name="canonical",
        command=["python", "probe.py", "run-canonical"],
        base_environment={"CUDA_VISIBLE_DEVICES": "0"},
        pair_allocator=lambda: (40101, 40102),
        command_runner=command_runner,
    )
    assert result["success"] is True
    assert result["attempts"][0]["command"] == [
        "python",
        "probe.py",
        "run-canonical",
    ]
    assert result["attempts"][0]["environment"] == {
        "CUDA_VISIBLE_DEVICES": "0",
        "MASTER_PORT": "40102",
        "TINYVLLM_DIST_PORT": "40101",
    }
    remote_command = commands[0][0][-1]
    assert "CUDA_VISIBLE_DEVICES=0" in remote_command
    assert "TINYVLLM_DIST_PORT=40101" in remote_command
    assert "MASTER_PORT=40102" in remote_command


def test_canonical_smoke_binding_requires_source_model_and_pass_label():
    runner = _load_runner()
    source_hashes = {"tools/a.py": "c" * 64}
    model_files = {"config.json": {"size": 1, "sha256": "d" * 64}}
    smoke = {
        "classification": "SMOKE_PASS",
        "source_commit": "a" * 40,
        "source_file_sha256": source_hashes,
        "model_resolved_revision": "b" * 40,
        "model_files": model_files,
    }
    assert runner.validate_smoke_binding(
        smoke,
        source_commit="a" * 40,
        source_file_sha256=source_hashes,
        model_resolved_revision="b" * 40,
        model_files=model_files,
    ) is True
    for field, replacement in (
        ("classification", "GO"),
        ("source_commit", "e" * 40),
        ("model_resolved_revision", "f" * 40),
        ("source_file_sha256", {"tools/a.py": "0" * 64}),
        (
            "model_files",
            {"config.json": {"size": 1, "sha256": "1" * 64}},
        ),
    ):
        tampered = dict(smoke)
        tampered[field] = replacement
        _expect_value_error(
            lambda payload=tampered: runner.validate_smoke_binding(
                payload,
                source_commit="a" * 40,
                source_file_sha256=source_hashes,
                model_resolved_revision="b" * 40,
                model_files=model_files,
            ),
            "smoke",
        )


def test_failed_execution_downloads_partial_artifacts_and_writes_incomplete():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary)
        calls = []

        def download():
            calls.append("download")
            (run_dir / "stdout").mkdir(parents=True)
            (run_dir / "stdout" / "canonical.log").write_text("partial\n")

        manifest = runner.preserve_failed_execution(
            run_dir,
            source_commit="a" * 40,
            model_revision="b" * 40,
            failure_kind="INCOMPLETE_WORKER_FAILURE",
            failure_detail="worker exited 1",
            download=download,
        )
        assert calls == ["download"]
        assert manifest["classification"] == "INCOMPLETE"
        assert [row["path"] for row in manifest["artifacts"]] == [
            "stdout/canonical.log",
        ]


def test_remote_source_tests_stop_before_worker_and_record_outputs():
    runner = _load_runner()
    calls = []
    plan = {
        "source_tests": [
            {
                "name": "contract",
                "command": ["python", "test_contract.py"],
                "environment": {"CUDA_VISIBLE_DEVICES": ""},
            },
            {
                "name": "probe",
                "command": ["python", "test_probe.py"],
                "environment": {"CUDA_VISIBLE_DEVICES": ""},
            },
        ],
    }

    def command_runner(command, **_kwargs):
        calls.append(command)
        return type("Result", (), {
            "returncode": 0,
            "stdout": "passed\n",
            "stderr": "",
        })()

    rows = runner.run_remote_source_tests(
        plan,
        command_runner=command_runner,
    )
    assert [row["name"] for row in rows] == ["contract", "probe"]
    assert all(row["exit_code"] == 0 for row in rows)
    assert all("CUDA_VISIBLE_DEVICES=" in call[-1] for call in calls)


def test_complete_manifest_inventory_excludes_itself_and_verifier_outputs():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary)
        (run_dir / "stdout").mkdir()
        (run_dir / "stderr").mkdir()
        (run_dir / "summary.json").write_text("{}\n")
        (run_dir / "independent_verification.json").write_text("{}\n")
        (run_dir / "report.md").write_text("# report\n")
        manifest = runner.write_complete_manifest(
            run_dir,
            source_commit="a" * 40,
            model_revision="b" * 40,
        )
        assert manifest["classification"] is None
        assert [row["path"] for row in manifest["artifacts"]] == [
            "summary.json",
        ]
        assert json.loads((run_dir / "manifest.json").read_text()) == (
            manifest
        )


def test_discover_model_snapshots_reads_only_matching_manifests():
    runner = _load_runner()
    with tempfile.TemporaryDirectory() as temporary:
        repo_root = Path(temporary)
        root = repo_root / runner.LOCAL_RUN_ROOT
        good = root / "acquire-good"
        bad = root / "acquire-bad"
        good.mkdir(parents=True)
        bad.mkdir(parents=True)
        payload = {
            "repository": runner.MODEL_REPOSITORY,
            "resolved_revision": "b" * 40,
            "files": {
                "config.json": {"size": 2, "sha256": "c" * 64},
            },
            "remote_model_dir": "/remote/good/model",
        }
        (good / "model_manifest.json").write_text(json.dumps(payload))
        (bad / "model_manifest.json").write_text("not json")
        candidates = runner.discover_model_snapshots(
            repo_root,
            resolved_revision="b" * 40,
        )
        assert len(candidates) == 1
        assert candidates[0]["run_tag"] == "acquire-good"


def test_process_and_port_manifests_preserve_all_attempts():
    runner = _load_runner()
    execution = {
        "attempts": [
            {
                "name": "canonical",
                "attempt": 1,
                "command": ["python", "probe.py"],
                "environment": {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "TINYVLLM_DIST_PORT": "40101",
                    "MASTER_PORT": "40102",
                },
                "tinyvllm_dist_port": 40101,
                "master_port": 40102,
                "exit_code": 1,
                "stdout": "",
                "stderr": "EADDRINUSE",
            },
            {
                "name": "canonical",
                "attempt": 2,
                "command": ["python", "probe.py"],
                "environment": {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "TINYVLLM_DIST_PORT": "40103",
                    "MASTER_PORT": "40104",
                },
                "tinyvllm_dist_port": 40103,
                "master_port": 40104,
                "exit_code": 0,
                "stdout": "ok",
                "stderr": "",
            },
        ],
    }
    processes, ports, attempts = runner.build_process_manifests(execution)
    assert len(processes["processes"]) == 1
    assert len(ports["pairs"]) == 1
    assert len(attempts["attempts"]) == 2
    assert ports["pairs"][0] == {
        "process": "canonical",
        "attempt": 2,
        "tinyvllm_dist_port": 40103,
        "master_port": 40104,
    }
    assert attempts["attempts"][0]["stderr"] == "EADDRINUSE"


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print("qwen35 hybrid-state remote runner tests passed")


if __name__ == "__main__":
    main()
