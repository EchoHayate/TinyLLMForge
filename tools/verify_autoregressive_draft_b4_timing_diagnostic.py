from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from autoregressive_draft_b4_timing_diagnostic import (
    validate_b4_timing_diagnostic,
)
from autoregressive_draft_performance_gate import write_json_atomic


def verify_b4_timing_diagnostic(
    artifact_path: Path,
    repo_root: Path,
) -> dict:
    try:
        artifact = json.loads(
            Path(artifact_path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("diagnostic artifact is unreadable") from error
    receipt = validate_b4_timing_diagnostic(artifact)
    verified = 0
    for relative_path, expected_hash in artifact["source_files"].items():
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("source path is unsafe")
        source_path = Path(repo_root) / path
        if not source_path.is_file():
            raise ValueError(
                f"source file is missing: {relative_path}"
            )
        actual_hash = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"source hash mismatch: {relative_path}"
            )
        verified += 1
    return {
        **receipt,
        "source_files_verified": verified,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--receipt")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    receipt = verify_b4_timing_diagnostic(
        Path(args.artifact),
        Path(args.repo_root),
    )
    if args.receipt:
        write_json_atomic(Path(args.receipt), receipt)
    else:
        print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
