#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from autoregressive_draft_command_timeline_diagnostic import (
    ENGINE_STEP_PHASES,
    TOP_LEVEL_KEYS,
    build_command_timeline_artifact,
    canonical_json_bytes,
    canonical_json_sha256,
    expected_epoch_identities,
)


DETACHED_ATTESTATION_PATHS = {
    "manifest.sha256",
    "verify.command-timeline.remote.json",
    "verify.command-timeline.remote.log",
    "verify.command-timeline.local.json",
    "verify.command-timeline.local.log",
}
FALSE_CLAIM_FIELDS = (
    "performance_improvement_established",
    "phase_1_complete",
    "promotion_ready",
)


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


def _restore_protocol_mapping_order(value: object) -> object:
    if isinstance(value, list):
        return [
            _restore_protocol_mapping_order(item)
            for item in value
        ]
    if not isinstance(value, dict):
        return value
    restored = {
        key: _restore_protocol_mapping_order(item)
        for key, item in value.items()
    }
    if set(restored) == set(ENGINE_STEP_PHASES):
        return {
            key: restored[key]
            for key in ENGINE_STEP_PHASES
        }
    if set(restored) == {"numerator", "denominator"}:
        return {
            key: restored[key]
            for key in ("numerator", "denominator")
        }
    return restored


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
        raise ValueError(f"{name} path escapes its root") from error
    if not path.is_file():
        raise ValueError(f"bound file is missing: {name}")
    return path


def _expected_raw_input_paths() -> dict[str, str]:
    expected = {
        "metadata": "metadata.json",
        "source_manifest": "source_manifest.json",
    }
    for identity in expected_epoch_identities():
        expected[f"worker:{identity.key}"] = (
            f"workers/block-{identity.block_index}/"
            f"{identity.label}.json"
        )
        expected[f"telemetry:{identity.key}"] = (
            f"telemetry/block-{identity.block_index}/"
            f"{identity.label}.json"
        )
    return expected


def verify_raw_input_bindings(
    artifact: dict,
    artifact_root: Path,
) -> dict[str, Path]:
    inventory = artifact.get("raw_input_files")
    expected = _expected_raw_input_paths()
    if not isinstance(inventory, dict) or set(inventory) != set(expected):
        raise ValueError("raw input inventory is incomplete")
    verified = {}
    for name, relative in expected.items():
        row = inventory.get(name)
        if not isinstance(row, dict):
            raise ValueError("raw input binding row is invalid")
        if row.get("path") != relative:
            raise ValueError(f"raw input path mismatch: {name}")
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
    source_root: Path,
) -> int:
    inventory = artifact.get("source_files")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("source inventory is invalid")
    for relative, expected_digest in inventory.items():
        path = _resolve_bound_path(
            source_root,
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
    artifact_root = Path(artifact_root)
    expected_manifest = artifact_root / "manifest.sha256"
    try:
        is_expected_path = (
            manifest_path.resolve() == expected_manifest.resolve()
        )
    except OSError as error:
        raise ValueError("manifest is missing") from error
    if not is_expected_path or not manifest_path.is_file():
        raise ValueError("manifest must be manifest.sha256")

    rows = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise ValueError(f"manifest line {line_number} is invalid")
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
        if normalized in DETACHED_ATTESTATION_PATHS:
            raise ValueError("manifest must exclude detached attestations")
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
        if (
            path.is_file()
            and path.relative_to(artifact_root).as_posix()
            not in DETACHED_ATTESTATION_PATHS
        )
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


def _require_authoritative_layout(
    artifact_path: Path,
) -> dict[str, Path]:
    artifact_root = artifact_path.parent
    canonical_path = artifact_root / "command-timeline.json"
    if artifact_path.resolve() != canonical_path.resolve():
        raise ValueError(
            "canonical artifact must be command-timeline.json"
        )
    required = {
        "artifact": canonical_path,
        "result": artifact_root / "result.json",
        "metadata": artifact_root / "metadata.json",
        "source_manifest": artifact_root / "source_manifest.json",
    }
    for identity in expected_epoch_identities():
        required[f"worker:{identity.key}"] = (
            artifact_root
            / "workers"
            / f"block-{identity.block_index}"
            / f"{identity.label}.json"
        )
        required[f"telemetry:{identity.key}"] = (
            artifact_root
            / "telemetry"
            / f"block-{identity.block_index}"
            / f"{identity.label}.json"
        )
    for name, path in required.items():
        if not path.is_file():
            raise ValueError(f"authoritative file is missing: {name}")
    return required


def _load_bound_bundle_inputs(
    *,
    artifact: dict,
    verified_inputs: dict[str, Path],
) -> dict:
    expected = _expected_raw_input_paths()
    if set(verified_inputs) != set(expected):
        raise ValueError("verified raw input inventory is incomplete")
    metadata = _load_json(
        verified_inputs["metadata"],
        name="bound metadata",
    )
    source_files = _load_json(
        verified_inputs["source_manifest"],
        name="bound source manifest",
    )
    if canonical_json_bytes(source_files) != canonical_json_bytes(
        artifact["source_files"]
    ):
        raise ValueError("bound source manifest mismatch")
    epoch_raw_inputs = {}
    for identity in expected_epoch_identities():
        epoch_raw_inputs[identity.key] = {
            "worker": _restore_protocol_mapping_order(
                _load_json(
                    verified_inputs[f"worker:{identity.key}"],
                    name=f"bound worker {identity.key}",
                )
            ),
            "telemetry": _load_json(
                verified_inputs[f"telemetry:{identity.key}"],
                name=f"bound telemetry {identity.key}",
            ),
        }
    return {
        "metadata": metadata,
        "epoch_raw_inputs": epoch_raw_inputs,
        "input_files": copy.deepcopy(artifact["raw_input_files"]),
        "source_files": source_files,
    }


def _expected_result_summary(
    artifact_path: Path,
    artifact: dict,
) -> dict:
    classification = artifact["classification"]
    return {
        "artifact_sha256": _sha256(artifact_path),
        "classification": classification,
        "localized_boundary": artifact["localized_boundary"],
        "runtime_optimization_authorized": (
            classification == "BOUNDARY_LOCALIZED"
        ),
        "performance_improvement_established": False,
        "phase_1_complete": False,
        "promotion_ready": False,
    }


def _verify_result_summary(
    result_path: Path,
    artifact_path: Path,
    artifact: dict,
) -> None:
    result = _load_json(result_path, name="authoritative result")
    expected = _expected_result_summary(artifact_path, artifact)
    if canonical_json_bytes(result) != canonical_json_bytes(expected):
        raise ValueError("authoritative result summary mismatch")


def verify_command_timeline_diagnostic(
    *,
    artifact_path: Path,
    source_root: Path,
    manifest_path: Path | None = None,
) -> dict:
    artifact_path = Path(artifact_path)
    source_root = Path(source_root)
    authoritative = _require_authoritative_layout(artifact_path)
    artifact = _load_json(
        artifact_path,
        name="canonical artifact",
    )
    if set(artifact) != set(TOP_LEVEL_KEYS):
        raise ValueError("canonical artifact keys are invalid")
    artifact = {
        key: artifact[key]
        for key in TOP_LEVEL_KEYS
    }
    canonical_json_bytes(artifact)
    for field in FALSE_CLAIM_FIELDS:
        if artifact.get(field) is not False:
            raise ValueError(f"{field} must remain false")

    artifact_root = artifact_path.parent
    verified_inputs = verify_raw_input_bindings(
        artifact,
        artifact_root,
    )
    verified_sources = verify_source_bindings(
        artifact,
        source_root,
    )
    rebuilt_inputs = _load_bound_bundle_inputs(
        artifact=artifact,
        verified_inputs=verified_inputs,
    )
    rebuilt = build_command_timeline_artifact(**rebuilt_inputs)
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(artifact):
        raise ValueError(
            "canonical artifact mismatch after recomputation"
        )
    _verify_result_summary(
        authoritative["result"],
        artifact_path,
        artifact,
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
    classification = artifact["classification"]
    return {
        "schema_version": 1,
        "verified": True,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_location": "unspecified",
        "artifact_path": str(artifact_path),
        "artifact_sha256": _sha256(artifact_path),
        "classification": classification,
        "localized_boundary": artifact["localized_boundary"],
        "runtime_optimization_authorized": (
            classification == "BOUNDARY_LOCALIZED"
        ),
        "performance_improvement_established": False,
        "phase_1_complete": False,
        "promotion_ready": False,
        "source_file_count": verified_sources,
        "source_inventory_sha256": canonical_json_sha256(
            artifact["source_files"]
        ),
        "raw_input_file_count": len(verified_inputs),
        "raw_input_inventory_sha256": canonical_json_sha256(
            artifact["raw_input_files"]
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
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--source-root", required=True)
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
    receipt = verify_command_timeline_diagnostic(
        artifact_path=Path(args.artifact),
        source_root=Path(args.source_root),
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
