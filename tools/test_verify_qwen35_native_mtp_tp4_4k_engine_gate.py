from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str, filename: str):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate_tests = _load(
    "qwen35_native_mtp_tp4_gate_test_fixtures",
    "test_qwen35_native_mtp_tp4_4k_engine_gate.py",
)
gate = _load(
    "qwen35_native_mtp_tp4_gate_for_verifier",
    "qwen35_native_mtp_tp4_4k_engine_gate.py",
)
verifier = _load(
    "verify_qwen35_native_mtp_tp4_gate",
    "verify_qwen35_native_mtp_tp4_4k_engine_gate.py",
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _published_run(tmp_path: Path) -> Path:
    result = gate.validate_result(gate_tests._result())
    source_files = {
        name: "b" * 64
        for name in gate.DEFAULT_SOURCE_FILES
    }
    result["source_tree_sha256"] = gate.source_hashes_sha256(
        source_files
    )
    run_dir = tmp_path / "authority"
    gate.publish_authority(
        run_dir,
        result,
        source_files=source_files,
    )
    return run_dir


def test_complete_published_run_passes(tmp_path):
    run_dir = _published_run(tmp_path)

    assert verifier.verify_run(run_dir) == {
        "classification": "PASS",
        "failures": [],
    }


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda run_dir: (run_dir / "status.json").unlink(),
            "status",
        ),
        (
            lambda run_dir: _write_json(
                run_dir / "status.json",
                {"status": "FAIL"},
            ),
            "status",
        ),
        (
            lambda run_dir: _write_json(
                run_dir / "source_manifest.json",
                {},
            ),
            "source manifest",
        ),
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: result["rank_inventory"].pop(),
            ),
            "rank inventory",
        ),
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: result["cells"][
                    "native_mtp:b1"
                ]["rank_snapshots"][2]["executor"][
                    "proposal_transactions"
                ].pop(),
            ),
            "transaction",
        ),
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: result["cells"][
                    "native_mtp:b1"
                ]["rank_snapshots"][3]["executor"].update(
                    allocated_physical_slots=1
                ),
            ),
            "slot leak",
        ),
    ),
)
def test_mutated_published_run_fails(tmp_path, mutate, match):
    run_dir = _published_run(tmp_path)
    mutate(run_dir)

    result = verifier.verify_run(run_dir)

    assert result["classification"] == "FAIL"
    assert any(
        match.lower() in failure.lower()
        for failure in result["failures"]
    )


def _mutate_result(run_dir: Path, mutate) -> None:
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    mutate(result)
    _write_json(result_path, result)


def test_source_root_recomputation_rejects_changed_source(
    tmp_path,
):
    run_dir = _published_run(tmp_path)
    source_root = tmp_path / "source"
    for name in gate.DEFAULT_SOURCE_FILES:
        path = source_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source\n", encoding="utf-8")
    manifest_path = run_dir / "source_manifest.json"
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest["source_files"] = gate.hash_source_files(
        source_root,
        gate.DEFAULT_SOURCE_FILES,
    )
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["source_tree_sha256"] = gate.source_hashes_sha256(
        manifest["source_files"]
    )
    _write_json(result_path, result)
    manifest["source_tree_sha256"] = result[
        "source_tree_sha256"
    ]
    manifest["artifacts"]["result.json"] = gate.sha256_file(
        result_path
    )
    _write_json(manifest_path, manifest)

    changed = source_root / gate.DEFAULT_SOURCE_FILES[0]
    changed.write_text("changed\n", encoding="utf-8")

    verified = verifier.verify_run(
        run_dir,
        source_root=source_root,
    )
    assert verified["classification"] == "FAIL"
    assert any(
        "source file digest mismatch" in failure
        for failure in verified["failures"]
    )
