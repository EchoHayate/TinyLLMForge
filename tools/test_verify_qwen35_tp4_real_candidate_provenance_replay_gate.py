from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = (
    ROOT
    / "tools/verify_qwen35_tp4_real_candidate_provenance_replay_gate.py"
)
RUN_DIR = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "qwen35-tp4-real-candidate-replay-20260728-145713"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "qwen35_tp4_real_candidate_independent_verifier",
        VERIFIER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_authoritative_run_passes_independent_verification():
    verifier = _load()
    result = verifier.verify_run(RUN_DIR, source_root=ROOT)
    assert result["status"] == "PASS"
    assert result["checks"] >= 500
    assert result["producer_count"] == 4
    assert result["replay_count"] == 4
    assert result["source_file_count"] == 58
    assert result["unique_process_count"] == 20


def _tamper(filename, mutator):
    verifier = _load()
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / RUN_DIR.name
        shutil.copytree(RUN_DIR, destination)
        path = destination / filename
        record = json.loads(path.read_text())
        mutator(record)
        path.write_text(
            json.dumps(record, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        return verifier.verify_run(destination, source_root=ROOT)


def test_producer_participant_rank_mismatch_is_rejected():
    result = _tamper(
        "tp4_real_candidate_provenance_oracle.json",
        lambda record: record["producer_rows"][2].__setitem__(
            "method_row",
            {
                **record["producer_rows"][2]["method_row"],
                "participant_id": 1,
            },
        ),
    )
    assert result["status"] == "FAIL"
    assert "participant" in result["detail"]


def test_rank2_unauthorized_second_replay_field_change_is_rejected():
    def mutate(record):
        row = next(
            row
            for row in record["replay_rows"]
            if row["mode"] == "tp4_real_replay_rank2_model_mismatch"
        )
        row["oracle_rows"][2]["dtype"] = "float32"

    result = _tamper(
        "tp4_real_candidate_provenance_replay_preflight.json",
        mutate,
    )
    assert result["status"] == "FAIL"
    assert "oracle" in result["detail"]


def test_producer_and_replay_pid_overlap_is_rejected():
    def mutate(record):
        record["replay_outer_process_ids"][0] = (
            record["producer_process_ids"][0]
        )
        record["replay_rows"][0]["process_id"] = (
            record["producer_process_ids"][0]
        )

    result = _tamper(
        "tp4_real_candidate_provenance_replay_preflight.json",
        mutate,
    )
    assert result["status"] == "FAIL"
    assert "process" in result["detail"].lower()


def test_resigned_preflight_with_production_engine_import_is_rejected():
    verifier = _load()
    preflight = (
        "tools/qwen35_tp4_real_candidate_provenance_replay_preflight.py"
    )
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / RUN_DIR.name
        shutil.copytree(RUN_DIR, destination)
        source_root = Path(directory) / "source"
        source_root.mkdir()
        oracle_path = (
            destination / "tp4_real_candidate_provenance_oracle.json"
        )
        result_path = (
            destination
            / "tp4_real_candidate_provenance_replay_preflight.json"
        )
        manifest_path = destination / "source_manifest.json"
        oracle = json.loads(oracle_path.read_text())
        result = json.loads(result_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        for name in oracle["source_file_sha256"]:
            target = source_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        target = source_root / preflight
        target.write_text(
            target.read_text()
            + "\nfrom tinyvllm.engine.llm_engine import LLMEngine\n"
        )
        tampered_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        for record in (oracle, result):
            record["source_file_sha256"][preflight] = tampered_hash
        manifest["local_file_sha256"][preflight] = tampered_hash
        tree = verifier._tree_hash(oracle["source_file_sha256"])
        oracle["source_tree_sha256"] = tree
        result["source_tree_sha256"] = tree
        manifest["source_tree_sha256"] = tree
        oracle_path.write_text(json.dumps(oracle, sort_keys=True) + "\n")
        result_path.write_text(json.dumps(result, sort_keys=True) + "\n")
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        verification = verifier.verify_run(
            destination,
            source_root=source_root,
        )
    assert verification["status"] == "FAIL"
    assert "import" in verification["detail"].lower()


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 real-candidate independent verifier tests "
        f"passed ({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
