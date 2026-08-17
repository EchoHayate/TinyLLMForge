from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from autoregressive_draft_cuda_graph_contract import (
    SCHEMA_VERSION,
    canonical_json_bytes,
    canonical_json_sha256,
    validate_gate_payload,
)


def _load_json(path: Path, name: str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ValueError(f"source file is unreadable: {path}") from error
    return digest.hexdigest()


def _safe_relative_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{name} is unsafe")
    return path


def verify_gate_bundle(
    *,
    payload_path: Path,
    source_root: Path,
    source_patch_path: Path,
    source_manifest_path: Path,
) -> dict:
    payload = _load_json(payload_path, "gate payload")
    summary = validate_gate_payload(payload)
    manifest = _load_json(
        source_manifest_path,
        "source manifest",
    )
    if manifest.get("schema_version") != 1:
        raise ValueError("source manifest schema mismatch")
    payload_sha256 = canonical_json_sha256(payload)
    if manifest.get("payload_sha256") != payload_sha256:
        raise ValueError("payload hash mismatch")
    source_patch_name = _safe_relative_path(
        manifest.get("source_patch"),
        "source patch path",
    )
    if source_patch_name.name != Path(source_patch_path).name:
        raise ValueError("source patch path mismatch")
    patch_sha256 = _sha256_file(source_patch_path)
    if (
        payload["provenance"]["source_patch_sha256"]
        != patch_sha256
    ):
        raise ValueError("source patch hash mismatch")

    source_files = manifest.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("source manifest files are missing")
    normalized_hashes = {}
    for relative_name, expected_sha256 in sorted(
        source_files.items()
    ):
        relative_path = _safe_relative_path(
            relative_name,
            "source file path",
        )
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_sha256
            )
        ):
            raise ValueError("source file digest is invalid")
        actual_sha256 = _sha256_file(
            Path(source_root) / relative_path
        )
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"source hash mismatch: {relative_name}"
            )
        normalized_hashes[relative_name] = actual_sha256
    source_tree_sha256 = canonical_json_sha256(
        normalized_hashes
    )
    if (
        payload["provenance"]["source_tree_sha256"]
        != source_tree_sha256
    ):
        raise ValueError("source tree hash mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": summary["classification"],
        "correctness_passed": summary[
            "correctness_passed"
        ],
        "every_rank_replayed": summary[
            "every_rank_replayed"
        ],
        "payload_sha256": payload_sha256,
        "source_patch_sha256": patch_sha256,
        "source_tree_sha256": source_tree_sha256,
        "source_files_verified": len(normalized_hashes),
        "summary": summary,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-patch", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--receipt")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    receipt = verify_gate_bundle(
        payload_path=Path(args.payload),
        source_root=Path(args.source_root),
        source_patch_path=Path(args.source_patch),
        source_manifest_path=Path(args.source_manifest),
    )
    output = canonical_json_bytes(receipt)
    if args.receipt:
        Path(args.receipt).write_bytes(output)
    else:
        sys.stdout.buffer.write(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
