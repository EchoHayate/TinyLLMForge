from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def _load_module(module_name, filename):
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


driver = _load_module(
    "run_qwen35_tp4_engine_correctness_authority",
    "run_qwen35_tp4_engine_correctness_authority.py",
)
reference_verifier = _load_module(
    "verify_qwen35_tp4_engine_reference_tokens",
    "verify_qwen35_tp4_engine_reference_tokens.py",
)
engine_verifier = _load_module(
    "verify_qwen35_tp4_engine_correctness_gate",
    "verify_qwen35_tp4_engine_correctness_gate.py",
)


class VerificationError(RuntimeError):
    pass


def _fail(message):
    raise VerificationError(message)


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"invalid authority summary: {error}")


def verify_run(run_dir):
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        _fail("authority root is missing")
    entries = list(run_dir.iterdir())
    expected = {
        driver.REFERENCE_DIR_NAME,
        driver.REFERENCE_VERIFICATION_NAME,
        driver.ENGINE_DIR_NAME,
        driver.RUN_SUMMARY_NAME,
    }
    if (
        any(entry.is_symlink() for entry in entries)
        or {entry.name for entry in entries} != expected
        or not (run_dir / driver.REFERENCE_DIR_NAME).is_dir()
        or not (run_dir / driver.ENGINE_DIR_NAME).is_dir()
        or not (
            run_dir / driver.REFERENCE_VERIFICATION_NAME
        ).is_file()
        or not (run_dir / driver.RUN_SUMMARY_NAME).is_file()
    ):
        _fail("authority root inventory mismatch")
    summary = _load_json(run_dir / driver.RUN_SUMMARY_NAME)
    required = {
        "classification",
        "model_manifest_sha256",
        "source_tree_sha256",
        "workload_manifest_sha256",
        "reference_classification",
        "engine_classification",
        "inventory",
    }
    if (
        not isinstance(summary, dict)
        or set(summary) != required
        or summary["classification"] != "PASS"
        or summary["reference_classification"] != "PASS"
        or summary["engine_classification"] != "PASS"
        or summary["inventory"] != [
            driver.REFERENCE_DIR_NAME,
            driver.REFERENCE_VERIFICATION_NAME,
            driver.ENGINE_DIR_NAME,
            driver.RUN_SUMMARY_NAME,
        ]
    ):
        _fail("authority summary schema or classification mismatch")
    reference = reference_verifier.verify_run(
        run_dir / driver.REFERENCE_DIR_NAME
    )
    external_reference = _load_json(
        run_dir / driver.REFERENCE_VERIFICATION_NAME
    )
    if external_reference != reference:
        _fail("external reference verification mismatch")
    engine = engine_verifier.verify_run(
        run_dir / driver.ENGINE_DIR_NAME
    )
    identities = {
        "model_manifest_sha256": summary["model_manifest_sha256"],
        "source_tree_sha256": summary["source_tree_sha256"],
        "workload_manifest_sha256": (
            summary["workload_manifest_sha256"]
        ),
    }
    if (
        reference.get("classification") != "PASS"
        or engine.get("classification") != "PASS"
        or any(
            reference.get(name) != value
            for name, value in identities.items()
        )
        or any(
            engine.get(name) != value
            for name, value in identities.items()
            if name != "workload_manifest_sha256"
        )
    ):
        _fail("authority cross-phase identity mismatch")
    return {
        "classification": "PASS",
        **identities,
        "reference_classification": "PASS",
        "engine_classification": "PASS",
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args(argv)
    print(json.dumps(
        verify_run(args.run_dir),
        sort_keys=True,
        separators=(",", ":"),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
