from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from speculative_runtime_performance_gate import (
    validate_performance_artifact,
)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
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


def verify_performance_artifact(
    artifact_path: Path,
    repo_root: Path,
) -> dict:
    artifact_path = Path(artifact_path)
    repo_root = Path(repo_root)
    artifact = json.loads(
        artifact_path.read_text(encoding="utf-8")
    )
    result = validate_performance_artifact(artifact)
    for relative_path, expected_digest in artifact[
        "source_files"
    ].items():
        source_path = repo_root / relative_path
        if not source_path.is_file():
            raise ValueError(
                f"source file is missing: {relative_path}"
            )
        actual_digest = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
        if actual_digest != expected_digest:
            raise ValueError(
                f"source hash mismatch: {relative_path}"
            )
    return {
        **result,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest(),
        "schema_version": artifact["schema_version"],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument("repo_root")
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    receipt = verify_performance_artifact(
        Path(args.artifact),
        Path(args.repo_root),
    )
    if args.output:
        _write_json_atomic(Path(args.output), receipt)
    else:
        print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
