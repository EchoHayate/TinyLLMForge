from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


def _load_contract():
    module_name = "qwen35_tp4_cached_continuation_correctness_contract"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = (
        Path(__file__).resolve().parent
        / "qwen35_tp4_cached_continuation_correctness_contract.py"
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


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"invalid JSON artifact: {path.name}: {error}")


def _verify_inventory(run_dir):
    entries = list(run_dir.iterdir())
    if (
        any(entry.is_symlink() for entry in entries)
        or any(not entry.is_file() for entry in entries)
    ):
        _fail("every artifact must be a regular file")
    if {entry.name for entry in entries} != set(contract.ARTIFACT_NAMES):
        _fail("artifact inventory mismatch")


def _verify_source_manifest(run_dir):
    payload = _load_json(run_dir / "source_manifest.json")
    required = {
        "schema_version",
        "source_tree_sha256",
        "model_manifest_sha256",
        "workload_manifest_sha256",
        "files",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        _fail("source manifest schema mismatch")
    if payload["schema_version"] != contract.SCHEMA_VERSION:
        _fail("source manifest schema version mismatch")
    for name in ("source_tree_sha256", "model_manifest_sha256"):
        value = payload[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            _fail(f"{name} is invalid")
    if (
        payload["workload_manifest_sha256"]
        != contract.WORKLOAD_MANIFEST_SHA256
    ):
        _fail("workload manifest identity mismatch")
    expected_files = set(contract.ARTIFACT_NAMES[:-1])
    if (
        not isinstance(payload["files"], dict)
        or set(payload["files"]) != expected_files
    ):
        _fail("source manifest file inventory mismatch")
    for name, expected_sha256 in payload["files"].items():
        if _sha256(run_dir / name) != expected_sha256:
            _fail(f"artifact hash mismatch: {name}")
    return payload


def _key(row):
    return f"{row['workload']}:{row['request_index']}"


def _verify_cross_artifacts(run_dir, rows):
    reference = _load_json(run_dir / "reference_outputs.json")
    restored = _load_json(run_dir / "restored_outputs.json")
    logits = _load_json(run_dir / "registered_logits.json")
    expected_keys = [_key(row) for row in rows]
    if (
        not isinstance(reference, dict)
        or not isinstance(restored, dict)
        or list(reference) != expected_keys
        or list(restored) != expected_keys
    ):
        _fail("output artifact inventory mismatch")
    for row in rows:
        key = _key(row)
        if (
            reference[key] != row["reference_output_token_ids"]
            or restored[key] != row["output_token_ids"]
        ):
            _fail("output artifact content mismatch")
    expected_logits = [{
        "workload": row["workload"],
        "request_index": row["request_index"],
        "max_abs_diff": row["logits_max_abs_diff"],
        "allclose": row["logits_allclose"],
    } for row in rows]
    if logits != expected_logits:
        _fail("registered logits artifact mismatch")


def verify_run(run_dir):
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        _fail("run directory is missing")
    _verify_inventory(run_dir)
    source = _verify_source_manifest(run_dir)
    result = _load_json(
        run_dir / "cached_continuation_correctness.json"
    )
    required = {
        "schema_version",
        "classification",
        "model_manifest_sha256",
        "workload_manifest_sha256",
        "rows",
    }
    if not isinstance(result, dict) or set(result) != required:
        _fail("result schema mismatch")
    if result["schema_version"] != contract.SCHEMA_VERSION:
        _fail("result schema version mismatch")
    if (
        result["model_manifest_sha256"]
        != source["model_manifest_sha256"]
    ):
        _fail("model manifest identity mismatch")
    if (
        result["workload_manifest_sha256"]
        != contract.WORKLOAD_MANIFEST_SHA256
    ):
        _fail("workload manifest identity mismatch")
    rows = result["rows"]
    classification = contract.classify_rows(rows)
    if classification["classification"] != "PASS":
        _fail(
            "independent classification failed: "
            + "; ".join(classification["failures"])
        )
    if result["classification"] != "PASS":
        _fail("producer classification is not PASS")
    _verify_cross_artifacts(run_dir, rows)
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "classification": "PASS",
        "source_tree_sha256": source["source_tree_sha256"],
        "model_manifest_sha256": source["model_manifest_sha256"],
        "workload_manifest_sha256": (
            source["workload_manifest_sha256"]
        ),
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
            "independent verification output must be outside exact-five run"
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
