from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def _load_gate_module():
    module_name = "generic_speculative_tp4_gate"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = (
        Path(__file__).resolve().parent
        / "generic_speculative_tp4_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate_module()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError(
            f"invalid JSON artifact: {path.name}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(
            f"JSON artifact must be an object: {path.name}"
        )
    return value


def _validate_manifest(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError(
            "source manifest must be a mapping"
        )
    if value.get("schema_version") != gate.SCHEMA_VERSION:
        raise ValueError(
            "source manifest schema version mismatch"
        )
    source_digest = gate._validate_sha256(
        value.get("source_tree_sha256"),
        "manifest source tree",
    )
    model_digest = gate._validate_sha256(
        value.get("model_manifest_sha256"),
        "manifest model",
    )
    source_files = value.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError(
            "source manifest file inventory is invalid"
        )
    normalized_sources = {}
    for name, digest in source_files.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                "source manifest path is invalid"
            )
        normalized_sources[name] = gate._validate_sha256(
            digest,
            f"manifest source file {name}",
        )
    artifacts = value.get("artifacts")
    if set(artifacts or {}) != {"result.json"}:
        raise ValueError(
            "source manifest artifact inventory mismatch"
        )
    result_digest = gate._validate_sha256(
        artifacts["result.json"],
        "manifest result artifact",
    )
    return {
        "schema_version": gate.SCHEMA_VERSION,
        "source_tree_sha256": source_digest,
        "model_manifest_sha256": model_digest,
        "source_files": normalized_sources,
        "artifacts": {
            "result.json": result_digest,
        },
    }


def verify_run(
    run_dir: Path,
    source_root: Path | None = None,
) -> dict:
    run_dir = Path(run_dir)
    failures = []
    try:
        result = gate.validate_result(
            _read_json(run_dir / "result.json")
        )
        manifest = _validate_manifest(
            _read_json(
                run_dir / "source_manifest.json"
            )
        )
        if (
            result["source_tree_sha256"]
            != manifest["source_tree_sha256"]
        ):
            raise ValueError(
                "source tree identity mismatch"
            )
        if (
            result["model_manifest_sha256"]
            != manifest["model_manifest_sha256"]
        ):
            raise ValueError(
                "model manifest identity mismatch"
            )
        if (
            gate.sha256_file(run_dir / "result.json")
            != manifest["artifacts"]["result.json"]
        ):
            raise ValueError(
                "result artifact SHA-256 mismatch"
            )
        if source_root is not None:
            source_root = Path(source_root)
            source_files = tuple(
                manifest["source_files"]
            )
            current_files = gate.hash_source_files(
                source_root,
                source_files,
            )
            if current_files != manifest["source_files"]:
                raise ValueError(
                    "current source file identity mismatch"
                )
            if (
                gate.source_tree_sha256(
                    source_root,
                    source_files,
                )
                != manifest["source_tree_sha256"]
            ):
                raise ValueError(
                    "current source tree identity mismatch"
                )
    except Exception as error:
        failures.append(str(error))
    return {
        "classification": (
            "PASS" if not failures else "FAIL"
        ),
        "failures": failures,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--source-root")
    parser.add_argument("--out")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = verify_run(
        Path(args.run_dir),
        (
            None
            if args.source_root is None
            else Path(args.source_root)
        ),
    )
    payload = (
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    if args.out is None:
        sys.stdout.write(payload)
    else:
        Path(args.out).write_text(
            payload,
            encoding="utf-8",
        )
    return 0 if result["classification"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
