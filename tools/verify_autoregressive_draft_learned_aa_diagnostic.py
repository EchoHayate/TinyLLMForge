#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import sys


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from autoregressive_draft_host_semantic_diagnostic import (
    parse_host_jsonl,
)
from autoregressive_draft_instability_telemetry import (
    parse_gpu_telemetry,
)
from autoregressive_draft_learned_aa_diagnostic import (
    EPOCH_ORDER,
    build_learned_aa_artifact,
    validate_learned_aa_artifact,
)


EXPECTED_INPUT_FILES = {
    "learned_a_prime_worker",
    "learned_b_prime_worker",
    "learned_a_worker",
    "learned_b_worker",
    "learned_a_gpu_csv",
    "learned_b_gpu_csv",
    "learned_a_host_jsonl",
    "learned_b_host_jsonl",
    "epoch_order",
    "prime_each_epoch",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, *, name: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


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
            f"{name} path must be a safe relative path"
        ) from error
    if not path.is_file():
        raise ValueError(f"bound file is missing: {name}")
    return path


def _verified_inputs(
    artifact: dict,
    *,
    artifact_directory: Path,
) -> dict[str, Path]:
    input_files = artifact.get("input_files")
    if (
        not isinstance(input_files, dict)
        or set(input_files) != EXPECTED_INPUT_FILES
    ):
        raise ValueError("learned A/A input inventory mismatch")
    resolved = {}
    for name, row in input_files.items():
        if not isinstance(row, dict):
            raise ValueError(f"input binding is invalid: {name}")
        path = _resolve_bound_path(
            artifact_directory,
            row.get("path"),
            name=name,
        )
        if _sha256(path) != row.get("sha256"):
            raise ValueError(f"input hash mismatch: {name}")
        resolved[name] = path
    return resolved


def _verify_sources(artifact: dict, repo_root: Path) -> int:
    source_files = artifact.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("learned A/A source inventory is invalid")
    verified = 0
    for relative_path, expected_hash in source_files.items():
        path = _resolve_bound_path(
            repo_root,
            relative_path,
            name="source",
        )
        if _sha256(path) != expected_hash:
            raise ValueError(
                f"source hash mismatch: {relative_path}"
            )
        verified += 1
    return verified


def verify_learned_aa_diagnostic(
    artifact_path: Path,
    repo_root: Path,
) -> dict:
    artifact_path = Path(artifact_path)
    repo_root = Path(repo_root)
    artifact = _read_json(
        artifact_path,
        name="learned A/A artifact",
    )
    validate_learned_aa_artifact(artifact)
    resolved = _verified_inputs(
        artifact,
        artifact_directory=artifact_path.parent,
    )
    source_files_verified = _verify_sources(
        artifact,
        repo_root,
    )
    epoch_order = (
        resolved["epoch_order"]
        .read_text(encoding="utf-8")
        .strip()
        .split(",")
    )
    prime_each_epoch = (
        resolved["prime_each_epoch"]
        .read_text(encoding="utf-8")
        .strip()
    )
    if prime_each_epoch != "1":
        raise ValueError("prime-each-epoch file must contain 1")
    expected = build_learned_aa_artifact(
        prime_workers={
            "learned_a": _read_json(
                resolved["learned_a_prime_worker"],
                name="learned A prime worker",
            ),
            "learned_b": _read_json(
                resolved["learned_b_prime_worker"],
                name="learned B prime worker",
            ),
        },
        workers={
            "learned_a": _read_json(
                resolved["learned_a_worker"],
                name="learned A worker",
            ),
            "learned_b": _read_json(
                resolved["learned_b_worker"],
                name="learned B worker",
            ),
        },
        gpu_samples={
            "learned_a": parse_gpu_telemetry(
                resolved["learned_a_gpu_csv"].read_text(
                    encoding="utf-8"
                )
            ),
            "learned_b": parse_gpu_telemetry(
                resolved["learned_b_gpu_csv"].read_text(
                    encoding="utf-8"
                )
            ),
        },
        host_samples={
            "learned_a": parse_host_jsonl(
                resolved["learned_a_host_jsonl"].read_text(
                    encoding="utf-8"
                )
            ),
            "learned_b": parse_host_jsonl(
                resolved["learned_b_host_jsonl"].read_text(
                    encoding="utf-8"
                )
            ),
        },
        epoch_order=epoch_order,
        prime_each_epoch=True,
        bundle_role=artifact["bundle_role"],
        input_files=artifact["input_files"],
        source_files=artifact["source_files"],
    )
    if artifact != expected:
        raise ValueError(
            "learned A/A artifact recomputation mismatch"
        )
    return {
        "status": "PASS",
        "schema_version": artifact["schema_version"],
        "classification": artifact["classification"],
        "exact_parity": True,
        "epoch_order": list(EPOCH_ORDER),
        "measured_runs_per_epoch": {
            epoch: len(artifact["epochs"][epoch]["measured_runs"])
            for epoch in EPOCH_ORDER
        },
        "coverage": {
            epoch: artifact["epochs"][epoch]["coverage"]["status"]
            for epoch in EPOCH_ORDER
        },
        "input_files_verified": len(resolved),
        "source_files_verified": source_files_verified,
        "candidate_process_boundary_effect": artifact[
            "claim_state"
        ]["candidate_process_boundary_effect"],
        "process_boundary_effect_established": False,
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
    receipt = verify_learned_aa_diagnostic(
        Path(args.artifact),
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
