from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def _load_artifact_validator():
    gate_path = (
        Path(__file__).resolve().parent
        / "speculative_residency_boundary_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_speculative_residency_boundary_gate_validation",
        gate_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "boundary gate validation module is unavailable"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.validate_boundary_artifact


def verify_boundary_artifact(
    artifact_path,
    repo_root,
) -> dict:
    artifact_path = Path(artifact_path)
    repo_root = Path(repo_root)
    artifact = json.loads(artifact_path.read_text())
    validate_boundary_artifact = (
        _load_artifact_validator()
    )
    validate_boundary_artifact(artifact)
    verified_source_files = []
    for relative_path, expected_hash in sorted(
        artifact["source_hashes"].items()
    ):
        source_path = repo_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(
                f"source file is missing: {relative_path}"
            )
        actual_hash = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"source hash mismatch: {relative_path}"
            )
        verified_source_files.append(relative_path)
    return {
        "schema_version": 1,
        "status": "PASS",
        "artifact_sha256": hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest(),
        "verified_source_files": verified_source_files,
        "classification": "NOT_PROMOTABLE",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_path")
    parser.add_argument("repo_root")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = verify_boundary_artifact(
        args.artifact_path,
        args.repo_root,
    )
    payload = json.dumps(
        result,
        indent=2,
        sort_keys=True,
    )
    if args.output:
        Path(args.output).write_text(payload + "\n")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
