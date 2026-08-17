from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from blockwise_speculative_verifier_gate import (
    CLASSIFICATION,
    SCHEMA_VERSION,
    atomic_write_json,
    sha256_file,
    validate_artifact,
)


def verify_artifact(
    artifact_path: Path,
    repo_root: Path,
    *,
    output_path: Path | None = None,
) -> dict:
    artifact_path = Path(artifact_path)
    repo_root = Path(repo_root)
    artifact = json.loads(
        artifact_path.read_text(encoding="utf-8")
    )
    validated = validate_artifact(artifact)
    for relative_path, expected_digest in validated[
        "source_files"
    ].items():
        source_path = repo_root / relative_path
        if not source_path.is_file():
            raise ValueError(
                f"source file is missing: {relative_path}"
            )
        actual_digest = sha256_file(source_path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"source hash drift: {relative_path}"
            )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "classification": CLASSIFICATION,
        "artifact_sha256": sha256_file(artifact_path),
        "cells": {
            key: value["status"]
            for key, value in validated["parity"].items()
        },
    }
    if output_path is not None:
        atomic_write_json(Path(output_path), receipt)
    return receipt


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument("repo_root")
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    verify_artifact(
        Path(args.artifact),
        Path(args.repo_root),
        output_path=(
            Path(args.output)
            if args.output is not None
            else None
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
