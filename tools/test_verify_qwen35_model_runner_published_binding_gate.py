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
    / "tools/verify_qwen35_model_runner_published_binding_gate.py"
)
RUN_DIR = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-model-runner-published-binding-20260728-100419"
)
PREREQUISITE = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-model-runner-load-publish-20260728-092500/"
    "model_runner_load_and_publish_preflight.json"
)


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "qwen35_published_binding_verifier_under_test",
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
    assert result["checks"] >= 300
    assert result["row_count"] == 6
    assert result["unique_process_count"] == 6
    assert result["source_file_count"] == 51


def test_tampered_value_hash_fails_against_prerequisite():
    verifier = _load_verifier()
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / "run"
        shutil.copytree(RUN_DIR, destination)
        path = (
            destination
            / "model_runner_published_candidate_binding_preflight.json"
        )
        record = json.loads(path.read_text())
        record["rows"][0]["aggregate_destination_sha256"] = "0" * 64
        path.write_text(json.dumps(record, sort_keys=True) + "\n")
        result = verifier.verify_run(
            destination,
            source_root=ROOT,
            prerequisite_artifact=PREREQUISITE,
        )
    assert result["status"] == "FAIL"
    assert "aggregate prerequisite value mismatch" in result["detail"]


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 ModelRunner published binding verifier tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
