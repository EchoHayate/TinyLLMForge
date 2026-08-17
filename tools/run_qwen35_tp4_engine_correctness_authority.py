from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


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


executor_module = _load_module(
    "qwen35_tp4_engine_correctness_executor",
    "qwen35_tp4_engine_correctness_executor.py",
)
engine_producer = _load_module(
    "qwen35_tp4_engine_correctness_producer",
    "qwen35_tp4_engine_correctness_producer.py",
)
backend_session = _load_module(
    "qwen35_tp4_engine_backend_session",
    "qwen35_tp4_engine_backend_session.py",
)
reference_producer = _load_module(
    "qwen35_tp4_engine_reference_tokens_producer",
    "qwen35_tp4_engine_reference_tokens_producer.py",
)
official = _load_module(
    "qwen35_tp4_engine_official_reference_executor",
    "qwen35_tp4_engine_official_reference_executor.py",
)


REFERENCE_DIR_NAME = "reference_authority"
REFERENCE_VERIFICATION_NAME = "reference_independent_verification.json"
ENGINE_DIR_NAME = "engine_authority"
RUN_SUMMARY_NAME = "authority_summary.json"


def _write_json(path, payload):
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _verify_complete_authority(path):
    verifier = _load_module(
        "verify_qwen35_tp4_engine_correctness_authority",
        "verify_qwen35_tp4_engine_correctness_authority.py",
    )
    return verifier.verify_run(path)


def _verify_manifest(path, expected_sha256, label):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")


def load_configuration(path, *, source_inventory_path=None):
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid executor configuration JSON") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != set(executor_module.CONFIGURATION_FIELDS)
    ):
        raise ValueError("executor configuration schema mismatch")
    if payload.get("world_size") != 4:
        raise ValueError("executor configuration world_size mismatch")
    values = dict(payload)
    values.pop("world_size")
    if isinstance(values.get("gpu_indices"), list):
        values["gpu_indices"] = tuple(values["gpu_indices"])
    configuration = executor_module.ExecutorConfiguration(**values)
    _verify_manifest(
        configuration.model_manifest_path,
        configuration.model_manifest_sha256,
        "model manifest",
    )
    _verify_manifest(
        configuration.workload_manifest_path,
        configuration.workload_manifest_sha256,
        "workload manifest",
    )
    if source_inventory_path is not None:
        source_inventory_path = Path(source_inventory_path)
        if (
            not source_inventory_path.is_file()
            or source_inventory_path.is_symlink()
        ):
            raise ValueError("source inventory is missing")
        try:
            inventory = json.loads(
                source_inventory_path.read_text(encoding="utf-8")
            )
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as error:
            raise ValueError("source inventory is invalid") from error
        if (
            not isinstance(inventory, dict)
            or set(inventory)
            != {"owned_files", "source_tree_sha256"}
            or inventory["source_tree_sha256"]
            != configuration.source_tree_sha256
            or not isinstance(inventory["owned_files"], list)
            or not inventory["owned_files"]
            or len(set(inventory["owned_files"]))
            != len(inventory["owned_files"])
            or any(
                not isinstance(name, str)
                or not name
                or name.startswith("/")
                or "\\" in name
                or ".." in Path(name).parts
                for name in inventory["owned_files"]
            )
        ):
            raise ValueError("source inventory mismatch")
    return configuration


def run_authority(
    *,
    output_root,
    configuration,
    engine_factory=backend_session._default_engine_factory,
):
    output_root = Path(output_root)
    if output_root.exists():
        raise ValueError("authority output root already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(
        prefix=f".{output_root.name}.",
        dir=output_root.parent,
    ))
    try:
        reference_dir = temporary_root / REFERENCE_DIR_NAME
        reference_verification = (
            temporary_root / REFERENCE_VERIFICATION_NAME
        )
        engine_dir = temporary_root / ENGINE_DIR_NAME
        reference_executor_factory = (
            official.build_official_reference_executor_factory(
                configuration
            )
        )
        reference_result = (
            reference_producer.produce_reference_authority(
                output_dir=reference_dir,
                verification_path=reference_verification,
                configuration=configuration,
                executor_factory=reference_executor_factory,
            )
        )
        backend_factory = (
            engine_producer.build_source_bound_real_backend_factory(
                configuration,
                authority_dir=reference_dir,
                verification_path=reference_verification,
                engine_factory=engine_factory,
            )
        )
        engine_executor_factory = (
            engine_producer.build_audited_executor_factory(
                configuration,
                backend_factory=backend_factory,
            )
        )
        engine_result = engine_producer.produce_authority(
            output_dir=engine_dir,
            source_tree_sha256=configuration.source_tree_sha256,
            model_manifest_sha256=(
                configuration.model_manifest_sha256
            ),
            executor_factory=engine_executor_factory,
        )
        summary = {
            "classification": "PASS",
            "model_manifest_sha256": (
                configuration.model_manifest_sha256
            ),
            "source_tree_sha256": configuration.source_tree_sha256,
            "workload_manifest_sha256": (
                configuration.workload_manifest_sha256
            ),
            "reference_classification": (
                reference_result["classification"]
            ),
            "engine_classification": engine_result["classification"],
            "inventory": [
                REFERENCE_DIR_NAME,
                REFERENCE_VERIFICATION_NAME,
                ENGINE_DIR_NAME,
                RUN_SUMMARY_NAME,
            ],
        }
        _write_json(temporary_root / RUN_SUMMARY_NAME, summary)
        verification = _verify_complete_authority(temporary_root)
        if verification.get("classification") != "PASS":
            raise ValueError(
                "complete Engine authority verification failed"
            )
        os.replace(temporary_root, output_root)
        return summary
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--source-inventory", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    result = run_authority(
        output_root=args.output_root,
        configuration=load_configuration(
            args.configuration,
            source_inventory_path=args.source_inventory,
        ),
    )
    print(json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
