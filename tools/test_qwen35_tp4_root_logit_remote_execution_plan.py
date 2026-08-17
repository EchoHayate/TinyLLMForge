from __future__ import annotations

import copy
import hashlib
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


planner = _load(
    "qwen35_tp4_root_logit_remote_execution_plan",
    "qwen35_tp4_root_logit_remote_execution_plan.py",
)
runner = planner.runner
contract = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_contract_for_root_plan_test",
    "qwen35_tp4_hybrid_prefix_benchmark_contract.py",
)


def _write(path, payload):
    path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prepare_source(root):
    source_tag = planner.runner.FROZEN_SOURCE_TAG
    source_dir = (
        root
        / "repo"
        / planner.runner.LOCAL_RUN_ROOT
        / source_tag
    )
    source_dir.mkdir(parents=True)
    source_tar = source_dir / "source.tar.gz"
    source_bytes = b"VALUE = 1\n"
    source_hashes = {
        "tinyvllm/example.py": hashlib.sha256(
            source_bytes
        ).hexdigest(),
    }
    source_tree_sha256 = hashlib.sha256(
        json.dumps(
            source_hashes,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_bytes = json.dumps(
        {
            "source_file_sha256": source_hashes,
            "source_tree_sha256": source_tree_sha256,
        },
        sort_keys=True,
    ).encode("utf-8")
    with tarfile.open(source_tar, "w:gz") as archive:
        for name, payload in (
            ("source/source_manifest.input.json", manifest_bytes),
            ("source/tinyvllm/example.py", source_bytes),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(
                info,
                __import__("io").BytesIO(payload),
            )
    planner.runner.FROZEN_SOURCE_TREE_SHA256 = source_tree_sha256
    _write(source_dir / "source_preparation.json", {
        "source_tag": source_tag,
        "root_source_tree_sha256": source_tree_sha256,
        "source_tar_sha256": planner._sha256(source_tar),
    })
    return source_tree_sha256


def _build(root):
    _prepare_source(root)
    return planner.build_remote_execution_plan(
        repo_root=root / "repo",
        output_dir=root / "plan",
        run_tag="root-logit-receipt-r1",
    )


def test_verifier_uses_frozen_source_bundle_not_live_runner_constants():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        original_tag = planner.runner.FROZEN_SOURCE_TAG
        original_sha = planner.runner.FROZEN_SOURCE_TREE_SHA256
        try:
            result = _build(root)
            path = root / "plan" / planner.PLAN_NAME
            planner.runner.FROZEN_SOURCE_TAG = "newer-source-tag"
            planner.runner.FROZEN_SOURCE_TREE_SHA256 = "f" * 64

            verified = planner.verify_remote_execution_plan(path)

            assert verified == result
        finally:
            planner.runner.FROZEN_SOURCE_TAG = original_tag
            planner.runner.FROZEN_SOURCE_TREE_SHA256 = original_sha


def _baseline(root):
    path = root / "resource_baseline.json"
    _write(path, {
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
    })
    return path


def test_builder_freezes_exact_root_authority():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        result = _build(root)

        assert result["stage_order"] == [
            "preflight",
            "run",
            "download",
            "verify",
        ]
        assert result["ssh_target"] == "sitian@10.232.195.203"
        assert result["frozen_source_tag"] == runner.FROZEN_SOURCE_TAG
        assert result["frozen_source_tree_sha256"] == (
            runner.FROZEN_SOURCE_TREE_SHA256
        )
        assert result["model_manifest_sha256"] == (
            contract.MODEL_MANIFEST_SHA256
        )
        assert result["exact_artifact_names"] == sorted(
            runner.EXACT_ARTIFACT_NAMES
        )
        assert result["minimum_free_bytes_per_gpu"] == 24 * 1024**3
        assert result["requires_no_active_compute_processes"] is True
        assert result["execution_performed"] is False
        assert result["remote_run_dir"] == runner.remote_run_dir(
            result["run_tag"]
        )
        assert result["local_run_dir"] == str(
            Path(result["repo_root"])
            / runner.LOCAL_RUN_ROOT
            / result["run_tag"]
        )
        assert set(result["stage_inputs"]) == set(
            result["stage_order"]
        )


def test_builder_binds_controlled_shared_baseline_sidecar():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _prepare_source(root)
        baseline = _baseline(root)
        result = planner.build_remote_execution_plan(
            repo_root=root / "repo",
            output_dir=root / "plan",
            run_tag="root-logit-shared-r1",
            resource_policy="controlled_shared",
            resource_baseline_path=baseline,
        )
        copied = (
            root
            / "plan"
            / planner.RESOURCE_BASELINE_NAME
        ).resolve()
        assert result["resource_policy"] == "controlled_shared"
        assert result["resource_baseline_path"] == str(copied)
        assert result["resource_baseline_sha256"] == planner._sha256(
            copied
        )
        assert result["gpu_indices"] == [2, 4, 5, 6]
        assert result["gpu_uuids"] == [
            "GPU-2",
            "GPU-4",
            "GPU-5",
            "GPU-6",
        ]
        assert result["requires_no_active_compute_processes"] is False
        assert result["benchmark_execution_authorized"] is False
        for stage in ("preflight", "run"):
            assert result["stage_inputs"][stage][
                "resource_baseline_sha256"
            ] == result["resource_baseline_sha256"]


def test_verifier_rejects_tampered_identity_or_stage():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _build(root)
        path = root / "plan" / planner.PLAN_NAME
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases = (
            (
                lambda value: value.update(
                    {"frozen_source_tree_sha256": "0" * 64}
                ),
                "source",
            ),
            (
                lambda value: value.update(
                    {"model_manifest_sha256": "1" * 64}
                ),
                "model",
            ),
            (
                lambda value: value["stage_order"].reverse(),
                "stage",
            ),
            (
                lambda value: value["stage_inputs"]["run"].update(
                    {"remote_run_dir": "/tmp/drift"}
                ),
                "stage",
            ),
        )
        for mutate, fragment in cases:
            changed = copy.deepcopy(payload)
            mutate(changed)
            _write(path, changed)
            try:
                planner.verify_remote_execution_plan(path)
            except ValueError as error:
                assert fragment in str(error), str(error)
            else:
                raise AssertionError("tampered root plan was accepted")
            _write(path, payload)


def test_builder_rejects_unsafe_or_existing_targets():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for run_tag in ("", "../escape", "a/b", "a b"):
            try:
                planner.build_remote_execution_plan(
                    repo_root=root / "repo",
                    output_dir=root / f"plan-{len(run_tag)}",
                    run_tag=run_tag,
                )
            except ValueError as error:
                assert "run tag" in str(error), str(error)
            else:
                raise AssertionError("unsafe run tag was accepted")

        output = root / "plan"
        output.mkdir()
        try:
            planner.build_remote_execution_plan(
                repo_root=root / "repo",
                output_dir=output,
                run_tag="root-logit-receipt-r1",
            )
        except ValueError as error:
            assert "output" in str(error), str(error)
        else:
            raise AssertionError("existing plan output was accepted")

        output.rmdir()
        local_run = (
            root
            / "repo"
            / runner.LOCAL_RUN_ROOT
            / "root-logit-receipt-r1"
        )
        local_run.mkdir(parents=True)
        try:
            planner.build_remote_execution_plan(
                repo_root=root / "repo",
                output_dir=output,
                run_tag="root-logit-receipt-r1",
            )
        except ValueError as error:
            assert "local run" in str(error), str(error)
        else:
            raise AssertionError("existing local run was accepted")


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 root-logit remote execution plan tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
