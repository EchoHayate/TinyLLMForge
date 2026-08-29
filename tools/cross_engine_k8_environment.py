from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence


ENVIRONMENT_SCHEMA_VERSION = "cross-engine-k8.environment.v1"
MODEL_INVENTORY_SCHEMA_VERSION = "cross-engine-k8.model-inventory.v1"
_STABLE_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")
_SOURCE_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class StableRelease:
    version: str
    requirement: str


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def candidate_releases(index_json: Mapping) -> tuple[StableRelease, ...]:
    releases = index_json.get("releases")
    if not isinstance(releases, Mapping):
        raise ValueError("release index does not contain releases")
    versions = []
    for version, files in releases.items():
        if not isinstance(version, str) or _STABLE_VERSION.fullmatch(
            version
        ) is None:
            continue
        if not isinstance(files, list) or not files:
            continue
        if all(bool(item.get("yanked")) for item in files):
            continue
        versions.append(version)
    versions.sort(key=_version_key, reverse=True)
    return tuple(
        StableRelease(
            version=version,
            requirement=f"vllm=={version}",
        )
        for version in versions
    )


def compatibility_decision(probes: Sequence[Mapping]) -> dict:
    normalized = [dict(probe) for probe in probes]
    for probe in normalized:
        if (
            probe.get("compatible") is True
            and probe.get("smoke_output_tokens") == 128
        ):
            public_multi_step = probe.get("public_multi_step") is True
            return {
                "classification": "COMPATIBLE",
                "selected_version": probe.get("version"),
                "multi_step_status": (
                    "VLLM_PUBLIC_MULTI_STEP_AVAILABLE"
                    if public_multi_step
                    else "VLLM_MULTI_STEP_NOT_PUBLICLY_AVAILABLE"
                ),
                "source_patch_allowed": False,
                "probes": normalized,
            }
    return {
        "classification": "INCOMPLETE_VLLM_COMPATIBILITY",
        "selected_version": None,
        "multi_step_status": "UNKNOWN",
        "source_patch_allowed": False,
        "probes": normalized,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Mapping) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_model_inventory(
    model_root: Path,
    *,
    expected_root: Path,
) -> dict:
    root = Path(model_root)
    expected = Path(expected_root)
    if root.resolve() != expected:
        raise ValueError("model root does not match expected canonical path")
    if not root.is_dir():
        raise ValueError("model root must be a directory")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("model inventory may not contain symlinks")
        if not path.is_file():
            continue
        stat = path.stat()
        files.append({
            "path": path.relative_to(root).as_posix(),
            "logical_bytes": stat.st_size,
            "allocated_bytes": stat.st_blocks * 512,
            "sha256": _sha256_file(path),
        })
    body = {
        "schema_version": MODEL_INVENTORY_SCHEMA_VERSION,
        "canonical_root": root.resolve().as_posix(),
        "logical_bytes": sum(item["logical_bytes"] for item in files),
        "allocated_bytes": sum(item["allocated_bytes"] for item in files),
        "files": files,
    }
    body["inventory_sha256"] = _canonical_json_sha256(body)
    return body


def build_environment_manifest(
    *,
    tinyllmforge_probe: Mapping,
    vllm_probe: Mapping,
    model_inventory: Mapping,
    source_revision: str,
) -> dict:
    if _SOURCE_REVISION.fullmatch(source_revision) is None:
        raise ValueError("source_revision is invalid")
    inventory_digest = model_inventory.get("inventory_sha256")
    if (
        not isinstance(inventory_digest, str)
        or _SHA256.fullmatch(inventory_digest) is None
    ):
        raise ValueError("model inventory digest is invalid")
    body = {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "source_revision": source_revision,
        "model_inventory_sha256": inventory_digest,
        "tinyllmforge": dict(tinyllmforge_probe),
        "vllm": dict(vllm_probe),
    }
    body["manifest_sha256"] = _canonical_json_sha256(body)
    return body
