from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
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
    "qwen35_tp4_root_logit_remote_execution_plan_for_receipt_test",
    "qwen35_tp4_root_logit_remote_execution_plan.py",
)
authorization = _load(
    "qwen35_tp4_root_logit_remote_execution_authorization_for_receipt_test",
    "qwen35_tp4_root_logit_remote_execution_authorization.py",
)
receipt = _load(
    "qwen35_tp4_root_logit_remote_execution_receipt",
    "qwen35_tp4_root_logit_remote_execution_receipt.py",
)
builder_fixture = _load(
    "qwen35_tp4_prerequisite_builder_fixture_for_root_receipt",
    "test_build_qwen35_tp4_performance_prerequisites.py",
)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
                "gpu_uuid": f"GPU-root-{index}",
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
            archive.addfile(info, io.BytesIO(payload))
    planner.runner.FROZEN_SOURCE_TREE_SHA256 = source_tree_sha256
    _write(source_dir / "source_preparation.json", {
        "source_tag": source_tag,
        "root_source_tree_sha256": source_tree_sha256,
        "source_tar_sha256": planner._sha256(source_tar),
    })


def _fixture(root, *, controlled_shared=False):
    _prepare_source(root)
    plan_kwargs = {}
    if controlled_shared:
        plan_kwargs = {
            "resource_policy": "controlled_shared",
            "resource_baseline_path": _baseline(root),
        }
    plan = planner.build_remote_execution_plan(
        repo_root=root / "repo",
        output_dir=root / "plan",
        run_tag="root-logit-receipt-r1",
        **plan_kwargs,
    )
    active = root / "authorization.json"
    consumed = root / "consumed_authorization.json"
    authorization.produce_authorization(
        plan=plan,
        output_path=active,
        nonce="root-logit-receipt-nonce",
    )
    authorization_record = authorization.consume_authorization(
        plan=plan,
        authorization_path=active,
        consumed_path=consumed,
    )
    local_run = Path(plan["local_run_dir"])
    artifacts = local_run / "artifacts"
    artifacts.mkdir(parents=True)
    artifact, verification = builder_fixture._root_payloads()
    _write(
        artifacts / "tp4_real_root_logit_correctness.json",
        artifact,
    )
    _write(artifacts / "rank_evidence.json", [{"rank": rank} for rank in range(4)])
    _write(artifacts / "source_manifest.json", {
        "source_tree_sha256": plan["frozen_source_tree_sha256"],
        "model_manifest_sha256": plan["model_manifest_sha256"],
    })
    (artifacts / "reference_logits.pt").write_bytes(b"reference")
    (artifacts / "native_rank0_logits.pt").write_bytes(b"native")
    preflight = {
        "run_tag": plan["run_tag"],
        "frozen_source_tag": plan["frozen_source_tag"],
        "frozen_source_tree_sha256": (
            plan["frozen_source_tree_sha256"]
        ),
        "status": "READY",
        "source_tree_sha256": plan["frozen_source_tree_sha256"],
        "selected": [
            {
                "rank": rank,
                "world_size": 4,
                "gpu_index": (
                    plan.get("gpu_indices", [0, 1, 2, 3])[rank]
                ),
                "gpu_uuid": (
                    plan.get(
                        "gpu_uuids",
                        [f"GPU-root-{value}" for value in range(4)],
                    )[rank]
                ),
                "free_bytes": 25 * 1024**3,
                "compute_processes": (
                    [{
                        "pid": 1000 + plan["gpu_indices"][rank],
                        "process_name": "python3",
                        "used_memory_mib": 436,
                        "start_time_ticks": (
                            2000 + plan["gpu_indices"][rank]
                        ),
                    }]
                    if controlled_shared
                    else []
                ),
            }
            for rank in range(4)
        ],
        "rows": [],
    }
    if controlled_shared:
        preflight.update({
            "resource_policy": "controlled_shared",
            "baseline_sha256": plan["resource_baseline_sha256"],
            "benchmark_execution_authorized": False,
        })
    run = {
        "status": "REMOTE_PASS",
        "run_tag": plan["run_tag"],
        "remote_run_dir": plan["remote_run_dir"],
        "artifact_names": plan["exact_artifact_names"],
    }
    if controlled_shared:
        run["final_resource"] = {
            "classification": "READY",
            "resource_policy": "controlled_shared",
            "baseline_sha256": plan["resource_baseline_sha256"],
            "benchmark_execution_authorized": False,
            "selected": [
                copy.deepcopy({
                    key: row[key]
                    for key in (
                        "gpu_index",
                        "gpu_uuid",
                        "free_bytes",
                        "compute_processes",
                    )
                })
                for row in preflight["selected"]
            ],
        }
    download = {
        "status": "DOWNLOADED",
        "artifact_names": plan["exact_artifact_names"],
    }
    _write(local_run / "remote_resource_preflight.json", preflight)
    _write(local_run / "remote_run.json", run)
    _write(local_run / "download.json", download)
    _write(local_run / "independent_verification.json", verification)
    stage_results = [
        {"name": "preflight", "result": preflight},
        {"name": "run", "result": run},
        {"name": "download", "result": download},
        {"name": "verify", "result": verification},
    ]
    for row in stage_results:
        row["result_sha256"] = receipt._canonical_sha(row["result"])
    return plan, authorization_record, consumed, stage_results


def test_receipt_accepts_controlled_shared_preflight_and_rejects_new_pid():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan, authorization_record, _, stages = _fixture(
            root,
            controlled_shared=True,
        )
        summary = receipt.produce_execution_receipt(
            plan=plan,
            stage_results=stages,
            output_path=root / "execution_receipt.json",
            authorization_record=authorization_record,
            root_verifier=lambda _path: stages[-1]["result"],
        )
        assert summary["classification"] == "PASS"

        changed = copy.deepcopy(stages)
        changed[0]["result"]["selected"][0][
            "compute_processes"
        ][0]["pid"] = 9000
        for row in changed:
            row["result_sha256"] = receipt._canonical_sha(
                row["result"]
            )
        try:
            receipt.produce_execution_receipt(
                plan=plan,
                stage_results=changed,
                output_path=root / "new-pid-receipt.json",
                authorization_record=authorization_record,
                root_verifier=lambda _path: changed[-1]["result"],
            )
        except ValueError as error:
            assert "process" in str(error), str(error)
        else:
            raise AssertionError("new shared PID was accepted")

        changed = copy.deepcopy(stages)
        changed[1]["result"]["final_resource"]["selected"][0][
            "compute_processes"
        ][0]["pid"] = 9000
        for row in changed:
            row["result_sha256"] = receipt._canonical_sha(
                row["result"]
            )
        _write(
            Path(plan["local_run_dir"]) / "remote_run.json",
            changed[1]["result"],
        )
        try:
            receipt.produce_execution_receipt(
                plan=plan,
                stage_results=changed,
                output_path=root / "final-new-pid-receipt.json",
                authorization_record=authorization_record,
                root_verifier=lambda _path: changed[-1]["result"],
            )
        except ValueError as error:
            assert "process" in str(error), str(error)
        else:
            raise AssertionError("new final shared PID was accepted")


def test_receipt_accepts_complete_semantic_authority():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan, authorization_record, _, stages = _fixture(root)
        output = root / "execution_receipt.json"
        summary = receipt.produce_execution_receipt(
            plan=plan,
            stage_results=stages,
            output_path=output,
            authorization_record=authorization_record,
            root_verifier=lambda _path: stages[-1]["result"],
        )

        assert summary["classification"] == "PASS"
        assert summary["run_tag"] == plan["run_tag"]
        assert summary["source_tree_sha256"] == (
            plan["frozen_source_tree_sha256"]
        )
        assert summary["model_manifest_sha256"] == (
            plan["model_manifest_sha256"]
        )
        assert summary["case_ids"] == ["p17", "p65", "synthetic"]
        assert summary["ranks"] == [0, 1, 2, 3]
        assert summary["checks"] > 0
        assert summary["artifact_names"] == (
            plan["exact_artifact_names"]
        )
        assert summary["stage_count"] == 4
        assert output.is_file()


def test_receipt_accepts_independent_verifier_check_count_drift():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan, authorization_record, _, stages = _fixture(root)
        independently_verified = copy.deepcopy(stages[-1]["result"])
        independently_verified["checks"] -= 1
        summary = receipt.produce_execution_receipt(
            plan=plan,
            stage_results=stages,
            output_path=root / "execution_receipt.json",
            authorization_record=authorization_record,
            root_verifier=lambda _path: independently_verified,
        )
        assert summary["classification"] == "PASS"
        assert summary["checks"] == stages[-1]["result"]["checks"]


def test_receipt_rejects_stage_or_artifact_tamper():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan, authorization_record, _, stages = _fixture(root)
        cases = (
            (
                lambda values: values[0]["result"].update(
                    {"status": "BLOCKED"}
                ),
                "preflight",
            ),
            (
                lambda values: values[0]["result"]["selected"][3].update(
                    {"gpu_uuid": "GPU-root-2"}
                ),
                "GPU",
            ),
            (
                lambda values: values[1]["result"].update(
                    {"artifact_names": ["missing"]}
                ),
                "inventory",
            ),
            (
                lambda values: values[3]["result"].update(
                    {"checks": 0}
                ),
                "disk",
            ),
        )
        for mutate, fragment in cases:
            changed = copy.deepcopy(stages)
            mutate(changed)
            for row in changed:
                row["result_sha256"] = receipt._canonical_sha(
                    row["result"]
                )
            try:
                receipt.produce_execution_receipt(
                    plan=plan,
                    stage_results=changed,
                    output_path=root / f"receipt-{fragment}.json",
                    authorization_record=authorization_record,
                    root_verifier=lambda _path, value=changed: (
                        value[-1]["result"]
                    ),
                )
            except ValueError as error:
                assert fragment.lower() in str(error).lower(), str(error)
            else:
                raise AssertionError(f"{fragment} tamper was accepted")

        artifact_dir = Path(plan["local_run_dir"]) / "artifacts"
        (artifact_dir / "extra.txt").write_text("x")
        try:
            receipt.produce_execution_receipt(
                plan=plan,
                stage_results=stages,
                output_path=root / "receipt-extra.json",
                authorization_record=authorization_record,
                root_verifier=lambda _path: stages[-1]["result"],
            )
        except ValueError as error:
            assert "inventory" in str(error), str(error)
        else:
            raise AssertionError("extra artifact was accepted")


def test_receipt_files_bind_plan_authorization_and_disk_evidence():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        plan, authorization_record, consumed, stages = _fixture(root)
        plan_path = root / "plan" / planner.PLAN_NAME
        receipt_path = root / "execution_receipt.json"
        receipt.produce_execution_receipt(
            plan=plan,
            stage_results=stages,
            output_path=receipt_path,
            authorization_record=authorization_record,
            root_verifier=lambda _path: stages[-1]["result"],
        )
        summary = receipt.verify_receipt_files(
            plan_path=plan_path,
            receipt_path=receipt_path,
            authorization_path=consumed,
            plan_verifier=planner.verify_remote_execution_plan,
            root_verifier=lambda _path: stages[-1]["result"],
        )
        assert summary["classification"] == "PASS"

        evidence = (
            Path(plan["local_run_dir"])
            / "independent_verification.json"
        )
        changed = json.loads(evidence.read_text())
        changed["checks"] += 1
        _write(evidence, changed)
        try:
            receipt.verify_receipt_files(
                plan_path=plan_path,
                receipt_path=receipt_path,
                authorization_path=consumed,
                plan_verifier=planner.verify_remote_execution_plan,
                root_verifier=lambda _path: stages[-1]["result"],
            )
        except ValueError as error:
            assert "disk" in str(error), str(error)
        else:
            raise AssertionError("changed disk evidence was accepted")


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 root-logit remote execution receipt tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
