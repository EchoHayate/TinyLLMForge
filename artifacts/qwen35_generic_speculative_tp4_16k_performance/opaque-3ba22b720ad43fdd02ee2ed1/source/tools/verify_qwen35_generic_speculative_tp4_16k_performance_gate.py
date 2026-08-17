from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


TOOLS = Path(__file__).resolve().parent


def _load_gate():
    path = (
        TOOLS
        / "qwen35_generic_speculative_tp4_16k_performance_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_qwen35_tp4_16k_performance_verifier_gate",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load gate from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def _read_json(path: Path, name: str, failures: list[str]):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        failures.append(f"{name}_read:{error}")
        return None


def verify_run(
    authority_path: Path,
    source_root: Path,
) -> dict:
    authority_path = Path(authority_path)
    source_root = Path(source_root)
    failures = []
    result_path = authority_path / "result.json"
    manifest_path = authority_path / "source_manifest.json"
    artifact = _read_json(
        result_path,
        "result",
        failures,
    )
    manifest = _read_json(
        manifest_path,
        "source_manifest",
        failures,
    )
    if artifact is not None:
        try:
            gate.validate_performance_artifact(artifact)
        except Exception as error:
            failures.append(f"artifact_validation:{error}")
    if not isinstance(manifest, dict):
        if manifest is not None:
            failures.append("source_manifest_validation:not_mapping")
    else:
        if manifest.get("schema_version") != gate.SCHEMA_VERSION:
            failures.append(
                "source_manifest_validation:schema_mismatch"
            )
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            failures.append(
                "source_manifest_validation:artifact_inventory"
            )
        elif result_path.is_file():
            expected_result_hash = artifacts.get("result.json")
            actual_result_hash = gate.sha256_file(result_path)
            if expected_result_hash != actual_result_hash:
                failures.append("result_digest_mismatch")
    source_inventory = None
    if isinstance(artifact, dict) and isinstance(
        artifact.get("source_files"),
        dict,
    ):
        source_inventory = artifact["source_files"]
    elif isinstance(manifest, dict) and isinstance(
        manifest.get("source_files"),
        dict,
    ):
        source_inventory = manifest["source_files"]
    if source_inventory:
        source_files = tuple(source_inventory)
        try:
            actual_sources = gate.hash_source_files(
                source_root,
                source_files,
            )
            actual_tree = gate.source_tree_sha256(
                source_root,
                source_files,
            )
        except Exception as error:
            failures.append(f"source_recompute:{error}")
        else:
            if source_inventory != actual_sources:
                failures.append("source_files_mismatch")
            if (
                isinstance(manifest, dict)
                and manifest.get("source_files")
                != actual_sources
            ):
                failures.append(
                    "source_manifest_files_mismatch"
                )
            if (
                not isinstance(artifact, dict)
                or artifact.get("source_tree_sha256")
                != actual_tree
            ):
                failures.append("source_tree_mismatch")
            if (
                isinstance(manifest, dict)
                and manifest.get("source_tree_sha256")
                != actual_tree
            ):
                failures.append(
                    "source_manifest_tree_mismatch"
                )
    else:
        failures.append("source_inventory_missing")
    if isinstance(artifact, dict):
        if (
            artifact.get("model_manifest_sha256")
            != gate.MODEL_MANIFEST_SHA256
        ):
            failures.append("model_manifest_mismatch")
    if isinstance(manifest, dict):
        if (
            manifest.get("model_manifest_sha256")
            != gate.MODEL_MANIFEST_SHA256
        ):
            failures.append(
                "source_manifest_model_mismatch"
            )
    failures = sorted(set(failures))
    return {
        "classification": "PASS" if not failures else "FAIL",
        "failures": failures,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", required=True)
    parser.add_argument("--source-root", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    verification = verify_run(
        Path(args.authority),
        Path(args.source_root),
    )
    print(
        json.dumps(
            verification,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if verification["classification"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
