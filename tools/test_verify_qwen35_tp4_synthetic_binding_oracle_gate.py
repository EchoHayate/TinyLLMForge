from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = (
    ROOT / "tools/qwen35_tp4_synthetic_binding_oracle_preflight.py"
)
VERIFIER_PATH = (
    ROOT / "tools/verify_qwen35_tp4_synthetic_binding_oracle_gate.py"
)
TP4_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-tp4-shm-fanout-20260728-115046/"
    "tp4_shared_memory_fanout_preflight.json"
)
ORACLE_ARTIFACT = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-tp4-synthetic-binding-oracle-v1/"
    "synthetic_binding_oracle.json"
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _build_run(directory):
    module = _load(PREFLIGHT_PATH, "synthetic_binding_fixture")
    rows = []
    for index, mode in enumerate(module.ATTEMPT_MODES):
        attempt = module.execute_tp4_synthetic_binding_attempt(
            source_root=ROOT,
            tp4_artifact=TP4_ARTIFACT,
            oracle_artifact=ORACLE_ARTIFACT,
            mode=mode,
            timeout_s=4.0,
            name_prefix=f"qwen35-synthetic-verifier-{mode}",
        )
        row = {
            "schema_version": module.ROW_SCHEMA_VERSION,
            "observed_user": "sitian",
            "observed_hostname": "verifier-test-host",
            "tp4_artifact_sha256": module.TP4_ARTIFACT_SHA256,
            "oracle_artifact_sha256": module.ORACLE_ARTIFACT_SHA256,
            "oracle_provenance": "synthetic-construction-free-oracle",
            "oracle_claim_boundary": "not-real-checkpoint-binding",
            "oracle_tensor_payload": "absent",
            "method_source_sha256": dict(
                module.METHOD_SOURCE_SHA256
            ),
            **attempt,
            "process_id": 20_000_000 + index,
        }
        module.validate_synthetic_binding_attempt_row(row)
        rows.append(row)
    record = module._aggregate(rows, ROOT)
    run_dir = Path(directory) / "qwen35-synthetic-binding-verifier-test"
    run_dir.mkdir()
    (run_dir / "tp4_synthetic_binding_oracle_preflight.json").write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    )
    manifest = {
        "schema_version": module.SCHEMA_VERSION,
        "run_tag": run_dir.name,
        "remote_target": module.REMOTE_TARGET,
        "remote_source_dir": "/remote/test/source",
        "remote_tp4_artifact": "/remote/test/tp4.json",
        "remote_oracle_artifact": "/remote/test/oracle.json",
        "tp4_artifact_sha256": module.TP4_ARTIFACT_SHA256,
        "oracle_artifact_sha256": module.ORACLE_ARTIFACT_SHA256,
        "method_source_sha256": dict(module.METHOD_SOURCE_SHA256),
        "source_tree_sha256": record["source_tree_sha256"],
        "local_file_sha256": record["source_file_sha256"],
        "remote_file_sha256": record["source_file_sha256"],
    }
    (run_dir / "source_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return run_dir


def test_generated_artifact_passes_independent_verification():
    verifier = _load(VERIFIER_PATH, "synthetic_binding_verifier")
    with tempfile.TemporaryDirectory() as directory:
        run_dir = _build_run(directory)
        result = verifier.verify_run(
            run_dir,
            source_root=ROOT,
            tp4_artifact=TP4_ARTIFACT,
            oracle_artifact=ORACLE_ARTIFACT,
        )
    assert result["status"] == "PASS"
    assert result["checks"] >= 400
    assert result["row_count"] == 4
    assert result["unique_process_count"] == 4
    assert result["unique_child_process_count"] == 12
    assert result["source_file_count"] == 57


def _tamper(mutator):
    verifier = _load(
        VERIFIER_PATH,
        "synthetic_binding_verifier_tamper",
    )
    with tempfile.TemporaryDirectory() as directory:
        run_dir = _build_run(directory)
        tamper_root = Path(directory) / "tampered"
        tamper_root.mkdir()
        destination = tamper_root / run_dir.name
        shutil.copytree(run_dir, destination)
        artifact = (
            destination
            / "tp4_synthetic_binding_oracle_preflight.json"
        )
        record = json.loads(artifact.read_text())
        mutator(record)
        artifact.write_text(json.dumps(record, sort_keys=True) + "\n")
        return verifier.verify_run(
            destination,
            source_root=ROOT,
            tp4_artifact=TP4_ARTIFACT,
            oracle_artifact=ORACLE_ARTIFACT,
        )


def test_modified_oracle_provenance_is_rejected():
    result = _tamper(
        lambda record: record.__setitem__(
            "oracle_provenance",
            "real-checkpoint",
        )
    )
    assert result["status"] == "FAIL"
    assert "oracle_provenance" in result["detail"]


def test_rank2_unauthorized_second_field_change_is_rejected():
    def mutate(record):
        row = next(
            row
            for row in record["rows"]
            if row["mode"] == "tp4_synthetic_rank2_model_mismatch"
        )
        row["oracle_rows"][2]["dtype"] = "float16"

    result = _tamper(mutate)
    assert result["status"] == "FAIL"
    assert "oracle_rows" in result["detail"]


def test_resigned_preflight_with_production_engine_import_is_rejected():
    verifier = _load(
        VERIFIER_PATH,
        "synthetic_binding_verifier_source_tamper",
    )
    preflight = "tools/qwen35_tp4_synthetic_binding_oracle_preflight.py"
    with tempfile.TemporaryDirectory() as directory:
        run_dir = _build_run(directory)
        source_root = Path(directory) / "source"
        source_root.mkdir()
        record_path = (
            run_dir / "tp4_synthetic_binding_oracle_preflight.json"
        )
        manifest_path = run_dir / "source_manifest.json"
        record = json.loads(record_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        for name in record["source_file_sha256"]:
            destination = source_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        target = source_root / preflight
        target.write_text(
            target.read_text()
            + "\nfrom tinyvllm.engine.llm_engine import LLMEngine\n"
        )
        tampered_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        record["source_file_sha256"][preflight] = tampered_hash
        manifest["local_file_sha256"][preflight] = tampered_hash
        manifest["remote_file_sha256"][preflight] = tampered_hash
        tree = verifier._tree_hash(record["source_file_sha256"])
        record["source_tree_sha256"] = tree
        manifest["source_tree_sha256"] = tree
        record_path.write_text(json.dumps(record, sort_keys=True) + "\n")
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        result = verifier.verify_run(
            run_dir,
            source_root=source_root,
            tp4_artifact=TP4_ARTIFACT,
            oracle_artifact=ORACLE_ARTIFACT,
        )
    assert result["status"] == "FAIL"
    assert "import" in result["detail"].lower()


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 synthetic binding oracle verifier tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
