from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


def _load_contract():
    module_name = "qwen35_tp4_engine_correctness_contract"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = (
        Path(__file__).resolve().parent
        / "qwen35_tp4_engine_correctness_contract.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract()


class VerificationError(RuntimeError):
    pass


def _fail(message):
    raise VerificationError(message)


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"invalid JSON artifact: {path.name}: {error}")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_inventory(run_dir):
    entries = list(run_dir.iterdir())
    if (
        any(entry.is_symlink() for entry in entries)
        or any(not entry.is_file() for entry in entries)
    ):
        _fail("every artifact must be a regular file")
    if {entry.name for entry in entries} != set(contract.ARTIFACT_NAMES):
        _fail("artifact inventory mismatch")


def _verify_manifest(run_dir):
    manifest = _load_json(run_dir / "source_manifest.json")
    required = {
        "schema_version",
        "source_tree_sha256",
        "model_manifest_sha256",
        "files",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        _fail("source manifest schema mismatch")
    if manifest["schema_version"] != contract.SCHEMA_VERSION:
        _fail("source manifest schema version mismatch")
    for name in ("source_tree_sha256", "model_manifest_sha256"):
        value = manifest[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            _fail(f"{name} is invalid")
    if (
        not isinstance(manifest["files"], dict)
        or set(manifest["files"]) != set(contract.ARTIFACT_NAMES[:-1])
    ):
        _fail("source manifest file inventory mismatch")
    for name, digest in manifest["files"].items():
        if _sha256(run_dir / name) != digest:
            _fail(f"artifact hash mismatch: {name}")
    return manifest


def _verify_cross_files(run_dir, rows):
    scheduler = _load_json(
        run_dir / "scheduler_observations.json"
    )
    ranks = _load_json(run_dir / "rank_events.json")
    expected_scheduler = [{
        "scenario": row["scenario"],
        "scheduler_steps": row["scheduler_steps"],
        "model_runner_calls": row["model_runner_calls"],
        "output_token_ids": row["output_token_ids"],
    } for row in rows]
    expected_ranks = [{
        "scenario": row["scenario"],
        "rank_inventory": row["rank_inventory"],
        "ack_ranks": row["ack_ranks"],
        "process_group_destroyed": row["process_group_destroyed"],
        "rank_exit_codes": row["rank_exit_codes"],
        "owned_children_remaining": row["owned_children_remaining"],
    } for row in rows]
    if scheduler != expected_scheduler:
        _fail("scheduler observation mismatch")
    if ranks != expected_ranks:
        _fail("rank event mismatch")


def verify_run(run_dir):
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        _fail("run directory is missing")
    _verify_inventory(run_dir)
    manifest = _verify_manifest(run_dir)
    result = _load_json(run_dir / "engine_correctness.json")
    required = {
        "schema_version",
        "classification",
        "model_manifest_sha256",
        "rows",
    }
    if not isinstance(result, dict) or set(result) != required:
        _fail("result schema mismatch")
    if result["schema_version"] != contract.SCHEMA_VERSION:
        _fail("result schema version mismatch")
    if (
        result["model_manifest_sha256"]
        != manifest["model_manifest_sha256"]
    ):
        _fail("model manifest identity mismatch")
    classification = contract.classify_rows(result["rows"])
    if classification["classification"] != "PASS":
        _fail(
            "independent classification failed: "
            + "; ".join(classification["failures"])
        )
    if result["classification"] != "PASS":
        _fail("producer classification is not PASS")
    _verify_cross_files(run_dir, result["rows"])
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "classification": "PASS",
        "source_tree_sha256": manifest["source_tree_sha256"],
        "model_manifest_sha256": manifest["model_manifest_sha256"],
        "checks": classification["checks"],
    }


def verify_and_write(run_dir, *, output_path):
    run_dir = Path(run_dir).resolve()
    output_path = Path(output_path).resolve()
    try:
        output_path.relative_to(run_dir)
    except ValueError:
        pass
    else:
        raise ValueError(
            "independent verification output must be outside exact-four run"
        )
    if output_path.exists():
        raise ValueError(
            "independent verification output already exists"
        )
    result = verify_run(run_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            result,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return result
