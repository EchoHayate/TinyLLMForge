#!/usr/bin/env python3
"""Independent verifier for the autoregressive-draft source-pair gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from autoregressive_draft_source_pair_gate import (
    canonical_json_bytes,
    validate_source_pair_artifact,
)


DETACHED_ATTESTATION_PATHS = {
    "manifest.sha256",
    "verify.source-pair.remote.json",
    "verify.source-pair.remote.log",
    "verify.source-pair.local.json",
    "verify.source-pair.local.log",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, name: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _receipt_mapping(paths: object, *, source: str) -> dict:
    if not isinstance(paths, dict) or tuple(paths) != ("remote", "local"):
        raise ValueError(
            f"{source} receipt paths require remote and local entries"
        )
    return {
        location: _load_json(
            Path(paths[location]),
            name=f"{source} {location} receipt",
        )
        for location in ("remote", "local")
    }


def _safe_manifest_path(root: Path, value: str) -> Path:
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or ".." in pure.parts
        or value in DETACHED_ATTESTATION_PATHS
    ):
        raise ValueError("manifest contains an unsafe path")
    path = root.joinpath(*pure.parts)
    current = root
    if current.is_symlink():
        raise ValueError("manifest root must not be a symlink")
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("manifest path contains a symlink")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("manifest path escapes artifact root") from error
    return path


def verify_manifest(manifest_path: Path, root: Path) -> dict:
    manifest_path = Path(manifest_path)
    root = Path(root)
    if manifest_path != root / "manifest.sha256":
        raise ValueError("parent manifest must be manifest.sha256")
    if manifest_path.is_symlink():
        raise ValueError("parent manifest must not be a symlink")
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError("parent manifest is unreadable") from error
    if not lines:
        raise ValueError("parent manifest is empty")
    rows = {}
    for line in lines:
        if "  " not in line:
            raise ValueError("parent manifest row is invalid")
        digest, relative = line.split("  ", 1)
        if (
            len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
            or relative in rows
        ):
            raise ValueError("parent manifest row is invalid")
        path = _safe_manifest_path(root, relative)
        if not path.is_file():
            raise ValueError("parent manifest bound file is missing")
        if _sha256(path) != digest:
            raise ValueError("parent manifest digest mismatch")
        rows[relative] = digest
    expected = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("artifact inventory contains a symlink")
        relative = path.relative_to(root).as_posix()
        if (
            path.is_file()
            and relative not in DETACHED_ATTESTATION_PATHS
        ):
            expected.add(relative)
    if set(rows) != expected:
        raise ValueError(
            "parent manifest does not match complete file inventory"
        )
    return {
        "verified": True,
        "sha256": _sha256(manifest_path),
        "file_count": len(rows),
    }


def verify_source_pair_gate(
    *,
    artifact_path: Path,
    baseline_artifact_path: Path,
    candidate_artifact_path: Path,
    baseline_manifest_path: Path,
    candidate_manifest_path: Path,
    baseline_receipt_paths: dict,
    candidate_receipt_paths: dict,
    manifest_path: Path | None = None,
) -> dict:
    artifact_path = Path(artifact_path)
    baseline_artifact_path = Path(baseline_artifact_path)
    candidate_artifact_path = Path(candidate_artifact_path)
    baseline_manifest_path = Path(baseline_manifest_path)
    candidate_manifest_path = Path(candidate_manifest_path)
    artifact = _load_json(
        artifact_path,
        name="source-pair artifact",
    )
    baseline_artifact = _load_json(
        baseline_artifact_path,
        name="baseline command-timeline artifact",
    )
    candidate_artifact = _load_json(
        candidate_artifact_path,
        name="candidate command-timeline artifact",
    )
    baseline_receipts = _receipt_mapping(
        baseline_receipt_paths,
        source="baseline",
    )
    candidate_receipts = _receipt_mapping(
        candidate_receipt_paths,
        source="candidate",
    )
    validated = validate_source_pair_artifact(
        artifact,
        baseline_artifact=baseline_artifact,
        candidate_artifact=candidate_artifact,
        baseline_verifier_receipts=baseline_receipts,
        candidate_verifier_receipts=candidate_receipts,
    )
    if canonical_json_bytes(validated) != artifact_path.read_bytes():
        raise ValueError("source-pair artifact is not canonical")
    sources = validated["sources"]
    if sources["baseline"]["artifact_sha256"] != _sha256(
        baseline_artifact_path
    ):
        raise ValueError("baseline artifact file hash mismatch")
    if sources["candidate"]["artifact_sha256"] != _sha256(
        candidate_artifact_path
    ):
        raise ValueError("candidate artifact file hash mismatch")
    if sources["baseline"]["manifest_sha256"] != _sha256(
        baseline_manifest_path
    ):
        raise ValueError("baseline manifest file hash mismatch")
    if sources["candidate"]["manifest_sha256"] != _sha256(
        candidate_manifest_path
    ):
        raise ValueError("candidate manifest file hash mismatch")
    manifest = (
        verify_manifest(Path(manifest_path), artifact_path.parent)
        if manifest_path is not None
        else {
            "verified": False,
            "sha256": None,
            "file_count": 0,
        }
    )
    return {
        "schema_version": 1,
        "verified": True,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_location": "unspecified",
        "artifact_path": str(artifact_path),
        "artifact_sha256": _sha256(artifact_path),
        "classification": validated["classification"],
        "performance_improvement_established": validated[
            "performance_improvement_established"
        ],
        "baseline_artifact_sha256": sources["baseline"][
            "artifact_sha256"
        ],
        "candidate_artifact_sha256": sources["candidate"][
            "artifact_sha256"
        ],
        "manifest_verified": manifest["verified"],
        "manifest_sha256": manifest["sha256"],
        "manifest_file_count": manifest["file_count"],
        "verifier_source_sha256": _sha256(Path(__file__)),
    }


def _write_json_exclusive(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--baseline-artifact", required=True)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--baseline-remote-receipt", required=True)
    parser.add_argument("--baseline-local-receipt", required=True)
    parser.add_argument("--candidate-remote-receipt", required=True)
    parser.add_argument("--candidate-local-receipt", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--receipt")
    parser.add_argument(
        "--verification-location",
        choices=("remote", "local"),
        default="local",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    receipt = verify_source_pair_gate(
        artifact_path=Path(args.artifact),
        baseline_artifact_path=Path(args.baseline_artifact),
        candidate_artifact_path=Path(args.candidate_artifact),
        baseline_manifest_path=Path(args.baseline_manifest),
        candidate_manifest_path=Path(args.candidate_manifest),
        baseline_receipt_paths={
            "remote": Path(args.baseline_remote_receipt),
            "local": Path(args.baseline_local_receipt),
        },
        candidate_receipt_paths={
            "remote": Path(args.candidate_remote_receipt),
            "local": Path(args.candidate_local_receipt),
        },
        manifest_path=(
            None if args.manifest is None else Path(args.manifest)
        ),
    )
    receipt["verification_location"] = args.verification_location
    if args.receipt:
        _write_json_exclusive(Path(args.receipt), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
