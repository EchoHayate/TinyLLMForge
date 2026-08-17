from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def _load_gate_module():
    path = (
        Path(__file__).resolve().parent
        / (
            "qwen35_native_mtp_tp4_16k_"
            "target_kv_offload_gate.py"
        )
    )
    spec = importlib.util.spec_from_file_location(
        (
            "qwen35_native_mtp_tp4_16k_"
            "target_kv_offload_gate"
        ),
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate_module()


def _load_json(path: Path, name: str):
    if not path.is_file():
        raise ValueError(f"{name} is missing")
    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except (
        OSError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError(
            f"{name} is invalid: {error}"
        ) from error


def verify_run(
    run_dir: Path,
    source_root: Path | None = None,
) -> dict:
    failures = []
    try:
        run_dir = Path(run_dir)
        result_path = run_dir / "result.json"
        status_path = run_dir / "status.json"
        manifest_path = (
            run_dir / "source_manifest.json"
        )
        raw_result = _load_json(
            result_path,
            "result",
        )
        result = gate.validate_result(raw_result)
        if raw_result != result:
            raise ValueError(
                "result is not canonical"
            )
        status = _load_json(
            status_path,
            "status",
        )
        if status != {
            "schema_version": gate.SCHEMA_VERSION,
            "status": "PASS",
            "classification": gate.CLASSIFICATION,
            "promotion_classification": (
                gate.PROMOTION_CLASSIFICATION
            ),
        }:
            raise ValueError(
                "status receipt mismatch"
            )
        manifest = _load_json(
            manifest_path,
            "source manifest",
        )
        expected_manifest_fields = {
            "schema_version",
            "source_tree_sha256",
            "target_model_manifest_sha256",
            "mtp_checkpoint_manifest_sha256",
            "source_files",
            "artifacts",
        }
        if (
            not isinstance(manifest, dict)
            or set(manifest)
            != expected_manifest_fields
        ):
            raise ValueError(
                "source manifest is not canonical"
            )
        if (
            manifest["schema_version"]
            != gate.SCHEMA_VERSION
        ):
            raise ValueError(
                "source manifest schema mismatch"
            )
        for field, name in (
            (
                "source_tree_sha256",
                "source tree digest",
            ),
            (
                "target_model_manifest_sha256",
                "target model manifest",
            ),
            (
                "mtp_checkpoint_manifest_sha256",
                "MTP checkpoint manifest",
            ),
        ):
            if manifest[field] != result[field]:
                raise ValueError(
                    f"source manifest {name} mismatch"
                )
        artifacts = manifest["artifacts"]
        if (
            not isinstance(artifacts, dict)
            or set(artifacts)
            != {"result.json", "status.json"}
        ):
            raise ValueError(
                "source manifest artifact "
                "inventory mismatch"
            )
        for name in (
            "result.json",
            "status.json",
        ):
            if artifacts[name] != (
                gate.sha256_file(run_dir / name)
            ):
                raise ValueError(
                    "artifact digest mismatch: "
                    f"{name}"
                )
        source_files = manifest["source_files"]
        if (
            not isinstance(source_files, dict)
            or tuple(sorted(source_files))
            != gate.DEFAULT_SOURCE_FILES
        ):
            raise ValueError(
                "source file inventory mismatch"
            )
        for name, digest in source_files.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "source file name is invalid"
                )
            gate._sha256(
                digest,
                "source file digest",
            )
        if gate.source_hashes_sha256(
            source_files
        ) != result["source_tree_sha256"]:
            raise ValueError(
                "source tree digest mismatch"
            )
        if source_root is not None:
            source_root = Path(source_root)
            for name, expected in source_files.items():
                path = source_root / name
                if not path.is_file():
                    raise ValueError(
                        "source file is missing: "
                        f"{name}"
                    )
                actual = gate.sha256_file(path)
                if actual != expected:
                    raise ValueError(
                        "source file digest mismatch: "
                        f"{name}"
                    )
    except Exception as error:
        failures.append(str(error))
    return {
        "classification": (
            "PASS"
            if not failures
            else "FAIL"
        ),
        "failures": failures,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--source-root")
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
    print(json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
    ))
    return (
        0
        if result["classification"] == "PASS"
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
