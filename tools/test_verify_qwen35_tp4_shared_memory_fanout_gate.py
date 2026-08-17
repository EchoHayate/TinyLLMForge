from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = (
    ROOT / "tools/qwen35_tp4_shared_memory_fanout_preflight.py"
)
VERIFIER_PATH = (
    ROOT / "tools/verify_qwen35_tp4_shared_memory_fanout_gate.py"
)
PREREQUISITE = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-live-shm-engine-ack-20260728-110846/"
    "live_shared_memory_engine_ack_dispatch_preflight.json"
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _build_run(directory):
    preflight = _load(
        PREFLIGHT_PATH,
        "qwen35_tp4_preflight_for_verifier_test",
    )
    rows = []
    for index, mode in enumerate(preflight.ATTEMPT_MODES):
        attempt = preflight.execute_tp4_shared_memory_fanout_attempt(
            source_root=ROOT,
            prerequisite_artifact=PREREQUISITE,
            mode=mode,
            timeout_s=4.0,
            name_prefix=f"qwen35-verifier-{mode}",
        )
        rows.append({
            "schema_version": preflight.ROW_SCHEMA_VERSION,
            "observed_user": "sitian",
            "observed_hostname": "verifier-test-host",
            "prerequisite_artifact_sha256": (
                preflight.PREREQUISITE_ARTIFACT_SHA256
            ),
            "llm_engine_file_sha256": (
                preflight.LLM_ENGINE_FILE_SHA256
            ),
            "model_runner_file_sha256": (
                preflight.MODEL_RUNNER_FILE_SHA256
            ),
            "ack_file_sha256": preflight.ACK_FILE_SHA256,
            "method_source_sha256": dict(
                preflight.METHOD_SOURCE_SHA256
            ),
            **attempt,
            "process_id": 10_000_000 + index,
        })
    record = preflight._aggregate(rows, ROOT)
    run_dir = Path(directory) / "qwen35-tp4-verifier-test"
    run_dir.mkdir()
    artifact = (
        run_dir / "tp4_shared_memory_fanout_preflight.json"
    )
    artifact.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    manifest = {
        "schema_version": preflight.SCHEMA_VERSION,
        "run_tag": run_dir.name,
        "remote_target": preflight.REMOTE_TARGET,
        "remote_source_dir": "/remote/test/source",
        "remote_prerequisite_artifact": "/remote/test/prerequisite.json",
        "prerequisite_artifact_sha256": (
            preflight.PREREQUISITE_ARTIFACT_SHA256
        ),
        "method_source_sha256": dict(
            preflight.METHOD_SOURCE_SHA256
        ),
        "source_tree_sha256": record["source_tree_sha256"],
        "local_file_sha256": record["source_file_sha256"],
        "remote_file_sha256": record["source_file_sha256"],
    }
    (run_dir / "source_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    return run_dir


def test_generated_artifact_passes_independent_verification():
    verifier = _load(
        VERIFIER_PATH,
        "qwen35_tp4_verifier_under_test",
    )
    with tempfile.TemporaryDirectory() as directory:
        run_dir = _build_run(directory)
        result = verifier.verify_run(
            run_dir,
            source_root=ROOT,
            prerequisite_artifact=PREREQUISITE,
        )
    assert result["status"] == "PASS"
    assert result["checks"] >= 350
    assert result["row_count"] == 4
    assert result["unique_process_count"] == 4
    assert result["unique_child_process_count"] == 12
    assert result["unique_shared_memory_count"] == 4
    assert result["source_file_count"] == 56


def _tampered_result(mutator):
    verifier = _load(
        VERIFIER_PATH,
        "qwen35_tp4_verifier_tamper_test",
    )
    with tempfile.TemporaryDirectory() as directory:
        run_dir = _build_run(directory)
        tamper_root = Path(directory) / "tampered"
        tamper_root.mkdir()
        destination = tamper_root / run_dir.name
        shutil.copytree(run_dir, destination)
        path = (
            destination / "tp4_shared_memory_fanout_preflight.json"
        )
        record = json.loads(path.read_text())
        mutator(record)
        path.write_text(json.dumps(record, sort_keys=True) + "\n")
        return verifier.verify_run(
            destination,
            source_root=ROOT,
            prerequisite_artifact=PREREQUISITE,
        )


def test_tampered_collector_return_order_is_rejected():
    def mutate(record):
        record["rows"][0]["collector_return_order"] = [3, 2, 1]

    result = _tampered_result(mutate)
    assert result["status"] == "FAIL"
    assert "collector_return_order" in result["detail"]


def test_tampered_rank2_exit_code_is_rejected():
    def mutate(record):
        row = next(
            row
            for row in record["rows"]
            if row["mode"] == "tp4_fanout_rank2_exit_without_ack"
        )
        row["child_exitcodes"]["2"] = 0

    result = _tampered_result(mutate)
    assert result["status"] == "FAIL"
    assert "child_exitcodes" in result["detail"]


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 shared-memory fan-out verifier tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
