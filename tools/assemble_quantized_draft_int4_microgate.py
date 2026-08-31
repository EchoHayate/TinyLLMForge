#!/usr/bin/env python3
"""Assemble the compact fused INT4 draft Stage-0 evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from tools.quantized_draft_int4_microgate import (
    classify_int4_microgate,
    validate_shape_manifest,
)
from tools.verify_quantized_draft_int4_microgate import (
    recompute_bundle_evidence,
)


RAW_FILES = (
    "environment.json",
    "shape_manifest.json",
    "microgate_rows.jsonl",
    "memory.json",
    "graph.json",
    "cleanup.json",
)


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON value: {value}")


def _load_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )


def _load_rows(path: Path) -> list[object]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(f"blank JSONL row at line {line_number}")
        rows.append(json.loads(line, parse_constant=_reject_constant))
    return rows


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _write_json(path: Path, payload: object) -> None:
    _atomic_write(path, _canonical_json(payload))


def _validate_identity(source_revision: str, run_tag: str) -> None:
    if (
        not isinstance(source_revision, str)
        or len(source_revision) != 40
        or any(
            character not in "0123456789abcdef"
            for character in source_revision
        )
    ):
        raise ValueError("source_revision must be a full lowercase SHA")
    if (
        not isinstance(run_tag, str)
        or not run_tag
        or "/" in run_tag
        or run_tag in {".", ".."}
    ):
        raise ValueError("run_tag is invalid")


def _validate_raw_inventory(raw_dir: Path) -> None:
    if not raw_dir.is_dir():
        raise ValueError("raw evidence directory does not exist")
    actual = {path.name for path in raw_dir.iterdir()}
    if actual != set(RAW_FILES):
        raise ValueError("raw evidence inventory mismatch")
    for name in RAW_FILES:
        path = raw_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("raw evidence contains a symlink or non-file")
        if path.resolve().parent != raw_dir:
            raise ValueError("raw evidence path escapes its directory")


def _validate_unique_row_identities(rows: list[object]) -> None:
    identities = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("microgate row must be an object")
        identity = (row.get("shape_id"), row.get("pair_index"))
        if identity in identities:
            raise ValueError("duplicate microgate row identity")
        identities.add(identity)


def _write_manifest(output_dir: Path) -> None:
    lines = []
    for path in sorted(output_dir.iterdir()):
        if path.name == "manifest.sha256":
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError("final bundle contains a symlink or non-file")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    _atomic_write(
        output_dir / "manifest.sha256",
        ("\n".join(lines) + "\n").encode("utf-8"),
    )


def assemble_bundle(
    *,
    raw_dir: Path,
    output_dir: Path,
    source_revision: str,
    run_tag: str,
) -> dict[str, object]:
    _validate_identity(source_revision, run_tag)
    raw_dir = Path(raw_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    _validate_raw_inventory(raw_dir)
    if output_dir.exists():
        raise ValueError("output_dir already exists")
    output_dir.mkdir(parents=True)
    try:
        for name in RAW_FILES:
            shutil.copyfile(raw_dir / name, output_dir / name)

        shapes = validate_shape_manifest(
            _load_json(output_dir / "shape_manifest.json")
        )
        rows = _load_rows(output_dir / "microgate_rows.jsonl")
        _validate_unique_row_identities(rows)
        result = classify_int4_microgate(
            shapes=shapes,
            rows=rows,
            memory=_load_json(output_dir / "memory.json"),
            graph=_load_json(output_dir / "graph.json"),
            cleanup=_load_json(output_dir / "cleanup.json"),
        )
        identity = {
            "schema_version": 1,
            "source_revision": source_revision,
            "run_tag": run_tag,
        }
        _write_json(output_dir / "source_identity.json", identity)
        _write_json(
            output_dir / "summary.json",
            {**identity, **result},
        )
        _write_json(
            output_dir / "classification.json",
            {
                **identity,
                "classification": result["classification"],
                "stop_before_distillation": (
                    result["classification"]
                    != "GO_FUSED_INT4_DRAFT_KERNEL"
                ),
            },
        )
        independent = recompute_bundle_evidence(
            output_dir,
            verify_manifest=False,
            require_receipt=False,
        )
        _write_json(
            output_dir / "independent_verification.json",
            independent,
        )
        _write_manifest(output_dir)
        return {**identity, **result}
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    result = assemble_bundle(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        source_revision=args.source_revision,
        run_tag=args.run_tag,
    )
    print(
        json.dumps(result, sort_keys=True, allow_nan=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
