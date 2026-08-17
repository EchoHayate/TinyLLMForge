from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from speculative_tp1_parity_gate import (  # noqa: E402
    validate_parity_artifact,
)


class VerificationError(RuntimeError):
    pass


def verify_artifact(
    *,
    artifact_path: Path,
    repo_root: Path,
) -> dict:
    artifact_path = Path(artifact_path)
    repo_root = Path(repo_root)
    try:
        payload = json.loads(
            artifact_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise VerificationError(
            f"cannot read artifact: {artifact_path}"
        ) from exc
    try:
        result = validate_parity_artifact(payload)
    except ValueError as exc:
        raise VerificationError(str(exc)) from exc
    for relative_path, expected_sha256 in (
        payload["source_files"].items()
    ):
        path = repo_root / relative_path
        if not path.is_file():
            raise VerificationError(
                f"source file is missing: {relative_path}"
            )
        actual_sha256 = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        if actual_sha256 != expected_sha256:
            raise VerificationError(
                f"source hash mismatch: {relative_path}"
            )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = verify_artifact(
        artifact_path=args.artifact,
        repo_root=args.repo_root,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
