#!/usr/bin/env python3

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

from autoregressive_draft_paired_stability_diagnostic import (
    build_paired_stability_artifact,
    load_bound_bundle_inputs,
    validate_paired_stability_artifact,
)


DETACHED_ATTESTATION_PATHS = {
    "manifest.sha256",
    "verify.paired-stability.remote.json",
    "verify.paired-stability.remote.log",
    "verify.paired-stability.local.json",
    "verify.paired-stability.local.log",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_bound_path(
    root: Path,
    relative_path: object,
    *,
    name: str,
) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{name} path must be a relative path")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"{name} path must be a safe relative path")
    path = root / Path(*pure.parts)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            f"{name} path escapes the artifact root"
        ) from error
    if not path.is_file():
        raise ValueError(f"bound file is missing: {name}")
    return path


def verify_raw_input_bindings(
    artifact: dict,
    artifact_root: Path,
) -> dict[str, Path]:
    inventory = artifact.get("raw_input_files")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("raw input inventory is invalid")
    verified = {}
    for name, row in inventory.items():
        if not isinstance(row, dict):
            raise ValueError("raw input binding row is invalid")
        path = _resolve_bound_path(
            artifact_root,
            row.get("path"),
            name=f"raw input {name}",
        )
        if _sha256(path) != row.get("sha256"):
            raise ValueError(f"raw input hash mismatch: {name}")
        verified[name] = path
    return verified


def verify_source_bindings(
    artifact: dict,
    repo_root: Path,
) -> int:
    inventory = artifact.get("source_files")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("source inventory is invalid")
    for relative, expected_digest in inventory.items():
        path = _resolve_bound_path(
            repo_root,
            relative,
            name=f"source {relative}",
        )
        if _sha256(path) != expected_digest:
            raise ValueError(f"source hash mismatch: {relative}")
    return len(inventory)


def verify_manifest(
    manifest_path: Path,
    artifact_root: Path,
) -> dict:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise ValueError("manifest is missing")
    rows = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise ValueError(
                f"manifest line {line_number} is invalid"
            )
        digest, relative = parts
        if (
            len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise ValueError("manifest digest is invalid")
        pure = PurePosixPath(relative)
        if (
            not relative
            or pure.is_absolute()
            or ".." in pure.parts
        ):
            raise ValueError("manifest path must be a safe relative path")
        normalized = pure.as_posix()
        if normalized in rows:
            raise ValueError("manifest path is duplicated")
        path = _resolve_bound_path(
            artifact_root,
            normalized,
            name=f"manifest file {normalized}",
        )
        if _sha256(path) != digest:
            raise ValueError(f"manifest hash mismatch: {normalized}")
        rows[normalized] = digest

    actual = {
        path.relative_to(artifact_root).as_posix()
        for path in artifact_root.rglob("*")
        if path.is_file()
        and path.relative_to(artifact_root).as_posix()
        not in DETACHED_ATTESTATION_PATHS
    }
    if set(rows) != actual:
        raise ValueError(
            "manifest inventory does not cover every authoritative file"
        )
    return {
        "verified": True,
        "sha256": _sha256(manifest_path),
        "file_count": len(rows),
    }


def verify_paired_stability_diagnostic(
    *,
    artifact_path: Path,
    repo_root: Path,
    manifest_path: Path | None = None,
) -> dict:
    artifact_path = Path(artifact_path)
    repo_root = Path(repo_root)
    try:
        artifact = json.loads(
            artifact_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("canonical artifact is invalid") from error
    validate_paired_stability_artifact(artifact)
    artifact_root = artifact_path.parent
    verified_inputs = verify_raw_input_bindings(
        artifact,
        artifact_root,
    )
    verified_sources = verify_source_bindings(artifact, repo_root)
    rebuilt_inputs = load_bound_bundle_inputs(
        artifact_root=artifact_root,
        artifact=artifact,
        verified_inputs=verified_inputs,
    )
    rebuilt = build_paired_stability_artifact(**rebuilt_inputs)
    if rebuilt != artifact:
        raise ValueError(
            "canonical artifact mismatch after recomputation"
        )
    manifest = (
        verify_manifest(manifest_path, artifact_root)
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
        "artifact_path": artifact_path.name,
        "artifact_sha256": _sha256(artifact_path),
        "classification": artifact["classification"],
        "candidate_process_boundary_effect": artifact[
            "candidate_process_boundary_effect"
        ],
        "process_boundary_effect_established": False,
        "source_file_count": verified_sources,
        "source_inventory_sha256": _canonical_json_sha256(
            artifact["source_files"]
        ),
        "raw_input_file_count": len(verified_inputs),
        "raw_input_inventory_sha256": _canonical_json_sha256(
            artifact["raw_input_sha256"]
        ),
        "manifest_verified": manifest["verified"],
        "manifest_sha256": manifest["sha256"],
        "manifest_file_count": manifest["file_count"],
        "verifier_source_sha256": _sha256(Path(__file__)),
    }


def _write_json_exclusive(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--repo-root", required=True)
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
    receipt = verify_paired_stability_diagnostic(
        artifact_path=Path(args.artifact),
        repo_root=Path(args.repo_root),
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
