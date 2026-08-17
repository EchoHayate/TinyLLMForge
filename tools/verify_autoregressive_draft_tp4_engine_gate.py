from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


RECEIPT_SCHEMA_VERSION = 1
MAX_FAILURES = 8
MAX_FAILURE_LENGTH = 500


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8")
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError(
            f"invalid JSON artifact: {Path(path).name}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(
            f"JSON artifact must be an object: {Path(path).name}"
        )
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _load_gate(source_root: Path):
    source_root = Path(source_root).resolve()
    path = (
        source_root
        / "tools"
        / "autoregressive_draft_tp4_engine_gate.py"
    )
    if not path.is_file() or path.is_symlink():
        raise ValueError("archived gate source is missing")
    previous = list(sys.path)
    try:
        sys.path.insert(0, os.fspath(source_root))
        spec = importlib.util.spec_from_file_location(
            "_archived_autoregressive_draft_tp4_engine_gate",
            path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous


def _validate_manifest(value: object, gate) -> dict:
    if not isinstance(value, dict):
        raise ValueError("source manifest must be a mapping")
    if value.get("schema_version") != gate.SCHEMA_VERSION:
        raise ValueError("source manifest schema mismatch")
    source_files = value.get("source_files")
    if (
        not isinstance(source_files, dict)
        or tuple(source_files) != gate.DEFAULT_SOURCE_FILES
    ):
        raise ValueError("source manifest inventory mismatch")
    normalized_files = {
        name: _sha256(
            digest,
            f"source file {name}",
        )
        for name, digest in source_files.items()
    }
    artifacts = value.get("artifacts")
    if set(artifacts or {}) != {"result.json", "source.tar"}:
        raise ValueError("source manifest artifact inventory mismatch")
    return {
        "schema_version": gate.SCHEMA_VERSION,
        "source_tree_sha256": _sha256(
            value.get("source_tree_sha256"),
            "source tree",
        ),
        "source_files": normalized_files,
        "artifacts": {
            name: _sha256(
                artifacts[name],
                f"artifact {name}",
            )
            for name in ("result.json", "source.tar")
        },
    }


def _bounded_failure(error: BaseException) -> str:
    text = str(error).strip() or type(error).__name__
    return text[:MAX_FAILURE_LENGTH]


def verify_run(
    run_dir: Path,
    source_root: Path,
) -> dict:
    failures = []
    result_sha256 = None
    source_digest = None
    try:
        run_dir = Path(run_dir)
        source_root = Path(source_root)
        gate = _load_gate(source_root)
        result_path = run_dir / "result.json"
        result = _read_json(result_path)
        gate.validate_gate_payload(result)
        if result_path.read_bytes() != _canonical_json_bytes(result):
            raise ValueError("result artifact is not canonical JSON")
        manifest = _validate_manifest(
            _read_json(run_dir / "source_manifest.json"),
            gate,
        )
        result_sha256 = _sha256_file(result_path)
        if (
            result_sha256
            != manifest["artifacts"]["result.json"]
        ):
            raise ValueError(
                "result artifact SHA-256 mismatch"
            )
        if (
            _sha256_file(run_dir / "source.tar")
            != manifest["artifacts"]["source.tar"]
        ):
            raise ValueError(
                "source archive SHA-256 mismatch"
            )
        if (
            gate.hash_source_files(
                source_root,
                gate.DEFAULT_SOURCE_FILES,
            )
            != manifest["source_files"]
        ):
            raise ValueError("source file identity mismatch")
        source_digest = gate.source_tree_sha256(
            source_root,
            gate.DEFAULT_SOURCE_FILES,
        )
        if source_digest != manifest["source_tree_sha256"]:
            raise ValueError("source tree identity mismatch")
    except BaseException as error:
        failures.append(_bounded_failure(error))
    failures = failures[:MAX_FAILURES]
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "classification": (
            "PASS" if not failures else "FAIL"
        ),
        "failures": failures,
        "result_sha256": result_sha256,
        "source_tree_sha256": source_digest,
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--source-root", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    receipt = verify_run(
        Path(args.run_dir),
        Path(args.source_root),
    )
    sys.stdout.buffer.write(_canonical_json_bytes(receipt))
    return 0 if receipt["classification"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
