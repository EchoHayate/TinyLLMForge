#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from autoregressive_draft_instability_telemetry import (
    validate_instability_telemetry_artifact,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_instability_telemetry(
    artifact_path: Path,
    repo_root: Path,
) -> dict:
    artifact_path = Path(artifact_path)
    repo_root = Path(repo_root)
    try:
        artifact = json.loads(
            artifact_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "telemetry artifact is unreadable"
        ) from error
    receipt = validate_instability_telemetry_artifact(
        artifact
    )
    source_files_verified = 0
    for relative_path, expected_hash in artifact[
        "source_files"
    ].items():
        source_path = repo_root / relative_path
        if not source_path.is_file():
            raise ValueError(
                f"source file is missing: {relative_path}"
            )
        if _sha256(source_path) != expected_hash:
            raise ValueError(
                f"source hash mismatch: {relative_path}"
            )
        source_files_verified += 1
    host_files_verified = 0
    for name, row in artifact["host_files"].items():
        host_path = artifact_path.parent / row["path"]
        if not host_path.is_file():
            raise ValueError(
                f"host file is missing: {name}"
            )
        if _sha256(host_path) != row["sha256"]:
            raise ValueError(
                f"host file hash mismatch: {name}"
            )
        host_files_verified += 1
    return {
        **receipt,
        "source_files_verified": source_files_verified,
        "host_files_verified": host_files_verified,
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--receipt")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    receipt = verify_instability_telemetry(
        Path(args.artifact),
        Path(args.repo_root),
    )
    if args.receipt:
        _write_json_atomic(Path(args.receipt), receipt)
    else:
        print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
