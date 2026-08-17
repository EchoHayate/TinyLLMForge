#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from autoregressive_draft_host_semantic_diagnostic import (
    build_host_semantic_artifact,
    build_host_semantic_comparison,
    parse_host_jsonl,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, *, name: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} is invalid")
    return value


def _resolve_relative(root: Path, value: str, *, name: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"{name} path is invalid")
    path = root / Path(*pure.parts)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{name} path is invalid") from error
    return path


def verify_host_semantic_artifact(
    artifact_path: Path,
    repo_root: Path,
) -> dict:
    artifact_path = Path(artifact_path)
    repo_root = Path(repo_root)
    artifact = _read_json(
        artifact_path,
        name="host semantic artifact",
    )
    if artifact.get("classification") != "ALIGNED_CAMPAIGN":
        raise ValueError("campaign classification is invalid")
    source_files_verified = 0
    for relative_path, expected_hash in artifact["source_files"].items():
        source_path = _resolve_relative(
            repo_root,
            relative_path,
            name="source",
        )
        if not source_path.is_file():
            raise ValueError(f"source file is missing: {relative_path}")
        if _sha256(source_path) != expected_hash:
            raise ValueError(f"source hash mismatch: {relative_path}")
        source_files_verified += 1
    resolved = {}
    input_files_verified = 0
    for name, row in artifact["input_files"].items():
        path = _resolve_relative(
            artifact_path.parent,
            row["path"],
            name=name,
        )
        if not path.is_file():
            raise ValueError(f"input file is missing: {name}")
        if _sha256(path) != row["sha256"]:
            raise ValueError(f"input hash mismatch: {name}")
        resolved[name] = path
        input_files_verified += 1
    expected = build_host_semantic_artifact(
        timing_artifact=_read_json(
            resolved["timing_artifact"],
            name="timing artifact",
        ),
        gpu_telemetry_artifact=_read_json(
            resolved["gpu_telemetry_artifact"],
            name="GPU telemetry artifact",
        ),
        target_worker=_read_json(
            resolved["target_worker"],
            name="target worker",
        ),
        learned_worker=_read_json(
            resolved["learned_worker"],
            name="learned worker",
        ),
        target_samples=parse_host_jsonl(
            resolved["target_host_jsonl"].read_text(encoding="utf-8")
        ),
        learned_samples=parse_host_jsonl(
            resolved["learned_host_jsonl"].read_text(encoding="utf-8")
        ),
        policy_order=artifact["policy_order"],
        prime_each_policy=artifact["prime_each_policy"],
        source_files=artifact["source_files"],
        input_files=artifact["input_files"],
    )
    if artifact != expected:
        raise ValueError("host semantic artifact recomputation mismatch")
    return {
        "status": "PASS",
        "schema_version": 1,
        "classification": artifact["classification"],
        "source_files_verified": source_files_verified,
        "input_files_verified": input_files_verified,
        "policy_repeat_coverage": {
            policy: len(artifact["policies"][policy]["measured_runs"])
            for policy in ("target", "learned")
        },
    }


def verify_host_semantic_comparison(
    artifact_path: Path,
    repo_root: Path,
) -> dict:
    artifact_path = Path(artifact_path)
    artifact = _read_json(
        artifact_path,
        name="host semantic comparison",
    )
    loaded = {}
    references = {}
    for role in ("learned_first", "learned_second"):
        row = artifact["campaign_artifacts"][role]
        if row.get("role") != role:
            raise ValueError(f"campaign artifact role mismatch: {role}")
        path = _resolve_relative(
            artifact_path.parent,
            row["path"],
            name=role,
        )
        if not path.is_file() or _sha256(path) != row["sha256"]:
            raise ValueError(
                f"campaign artifact hash mismatch: {role}"
            )
        verify_host_semantic_artifact(path, repo_root)
        loaded[role] = _read_json(path, name=f"{role} campaign")
        references[role] = row
    expected = build_host_semantic_comparison(
        first_artifact=loaded["learned_first"],
        second_artifact=loaded["learned_second"],
        first_reference=references["learned_first"],
        second_reference=references["learned_second"],
    )
    if artifact != expected:
        raise ValueError("host comparison recomputation mismatch")
    return {
        "status": "PASS",
        "schema_version": 1,
        "classification": artifact["classification"],
        "campaign_artifacts_verified": 2,
        "source_files_verified_per_campaign": len(
            artifact["source_identity"]
        ),
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
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
    artifact_path = Path(args.artifact)
    artifact = _read_json(artifact_path, name="artifact")
    if artifact.get("classification") == "ALIGNED_CAMPAIGN":
        receipt = verify_host_semantic_artifact(
            artifact_path,
            Path(args.repo_root),
        )
    else:
        receipt = verify_host_semantic_comparison(
            artifact_path,
            Path(args.repo_root),
        )
    if args.receipt:
        _write_json_atomic(Path(args.receipt), receipt)
    else:
        print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
