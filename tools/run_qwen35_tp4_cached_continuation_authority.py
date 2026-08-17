from __future__ import annotations

import argparse
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


contract = _load_module(
    "qwen35_tp4_cached_continuation_correctness_contract",
    "qwen35_tp4_cached_continuation_correctness_contract.py",
)
producer = _load_module(
    "qwen35_tp4_cached_continuation_correctness_producer",
    "qwen35_tp4_cached_continuation_correctness_producer.py",
)
verifier = _load_module(
    "verify_qwen35_tp4_cached_continuation_correctness_gate",
    "verify_qwen35_tp4_cached_continuation_correctness_gate.py",
)
engine_driver = _load_module(
    "run_qwen35_tp4_engine_correctness_authority",
    "run_qwen35_tp4_engine_correctness_authority.py",
)
backend_session = _load_module(
    "qwen35_tp4_cached_continuation_backend_session",
    "qwen35_tp4_cached_continuation_backend_session.py",
)


def load_configuration(path, *, source_inventory_path):
    configuration = engine_driver.load_configuration(
        path,
        source_inventory_path=source_inventory_path,
    )
    if (
        configuration.workload_manifest_sha256
        != contract.WORKLOAD_MANIFEST_SHA256
    ):
        raise ValueError(
            "cached-continuation workload manifest SHA mismatch"
        )
    return configuration


def run_authority(
    *,
    output_dir,
    verification_path,
    configuration,
    engine_factory=backend_session._default_engine_factory,
    reference_executor_factory=None,
):
    output_dir = Path(output_dir)
    verification_path = Path(verification_path)
    if output_dir.exists():
        raise ValueError("cached authority output already exists")
    if verification_path.exists():
        raise ValueError(
            "cached authority verification already exists"
        )
    if output_dir.resolve() == verification_path.resolve():
        raise ValueError(
            "cached authority outputs must be distinct"
        )
    try:
        verification_path.resolve().relative_to(
            output_dir.resolve()
        )
    except ValueError:
        pass
    else:
        raise ValueError(
            "cached authority verification must be outside exact-five run"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    verification_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ))
    temporary_output = temporary_root / "exact_five"
    temporary_verification = (
        verification_path.parent
        / f".{verification_path.name}.{temporary_root.name}"
    )
    published_verification = False
    try:
        kwargs = {
            "output_dir": temporary_output,
            "configuration": configuration,
            "engine_factory": engine_factory,
        }
        if reference_executor_factory is not None:
            kwargs["reference_executor_factory"] = (
                reference_executor_factory
            )
        result = producer.produce_configured_authority(**kwargs)
        if result.get("classification") != "PASS":
            raise ValueError(
                "cached authority producer did not return PASS"
            )
        verification = verifier.verify_and_write(
            temporary_output,
            output_path=temporary_verification,
        )
        if verification.get("classification") != "PASS":
            raise ValueError(
                "cached authority independent verification failed"
            )
        os.replace(temporary_verification, verification_path)
        published_verification = True
        os.replace(temporary_output, output_dir)
        return verification
    except BaseException:
        if published_verification and verification_path.exists():
            verification_path.unlink()
        raise
    finally:
        if temporary_verification.exists():
            temporary_verification.unlink()
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--source-inventory", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verification-path", required=True)
    arguments = parser.parse_args(argv)
    result = run_authority(
        output_dir=arguments.output_dir,
        verification_path=arguments.verification_path,
        configuration=load_configuration(
            arguments.configuration,
            source_inventory_path=arguments.source_inventory,
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
