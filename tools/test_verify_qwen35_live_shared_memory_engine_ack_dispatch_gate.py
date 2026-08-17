from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = (
    ROOT
    / "tools/"
    "verify_qwen35_live_shared_memory_engine_ack_dispatch_gate.py"
)
RUN_DIR = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-live-shm-engine-ack-20260728-110846"
)
PREREQUISITE = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-engine-ack-transport-20260728-102828/"
    "engine_ack_transport_preflight.json"
)


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "qwen35_live_shm_engine_ack_verifier_under_test",
        VERIFIER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_artifact_passes_independent_verification():
    verifier = _load_verifier()
    result = verifier.verify_run(
        RUN_DIR,
        source_root=ROOT,
        prerequisite_artifact=PREREQUISITE,
    )
    assert result["status"] == "PASS"
    assert result["checks"] >= 350
    assert result["row_count"] == 4
    assert result["unique_process_count"] == 4
    assert result["unique_child_process_count"] == 4
    assert result["unique_shared_memory_count"] == 4
    assert result["source_file_count"] == 55


def _tampered_result(field, value):
    verifier = _load_verifier()
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / RUN_DIR.name
        shutil.copytree(RUN_DIR, destination)
        path = (
            destination
            / "live_shared_memory_engine_ack_dispatch_preflight.json"
        )
        record = json.loads(path.read_text())
        row = next(
            row
            for row in record["rows"]
            if row["mode"] == "tp2_shm_worker_binding_error"
        )
        row[field] = value
        path.write_text(json.dumps(record, sort_keys=True) + "\n")
        return verifier.verify_run(
            destination,
            source_root=ROOT,
            prerequisite_artifact=PREREQUISITE,
        )


def test_tampered_shared_memory_name_is_rejected():
    result = _tampered_result(
        "shared_memory_name",
        "tinyvllm",
    )
    assert result["status"] == "FAIL"
    assert "shared_memory_name" in result["detail"]


def test_tampered_acknowledgement_status_is_rejected():
    result = _tampered_result(
        "acknowledgement_status",
        "error",
    )
    assert result["status"] == "FAIL"
    assert "acknowledgement_status" in result["detail"]


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 live shared-memory Engine ack verifier tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
