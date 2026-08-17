from __future__ import annotations

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


contract = _load_module(
    "qwen35_tp4_cached_continuation_correctness_contract",
    "qwen35_tp4_cached_continuation_correctness_contract.py",
)


def build_configured_executor_factory(
    configuration,
    *,
    engine_factory,
    reference_executor_factory,
    session_factory_type=None,
):
    executor_module = _load_module(
        "qwen35_tp4_cached_continuation_correctness_executor",
        "qwen35_tp4_cached_continuation_correctness_executor.py",
    )
    backend_module = _load_module(
        "qwen35_tp4_cached_continuation_backend_session",
        "qwen35_tp4_cached_continuation_backend_session.py",
    )
    if session_factory_type is None:
        session_factory_type = (
            backend_module.CachedContinuationSessionFactory
        )
    for value, label in (
        (engine_factory, "engine_factory"),
        (reference_executor_factory, "reference_executor_factory"),
        (session_factory_type, "session_factory_type"),
    ):
        if not callable(value):
            raise TypeError(f"{label} must be callable")
    to_payload = getattr(configuration, "to_payload", None)
    if not callable(to_payload):
        raise TypeError(
            "configuration must provide a canonical payload"
        )
    payload = dict(to_payload())
    executor_configuration = (
        executor_module._configuration_from_payload(payload)
    )

    def factory():
        session_factory = session_factory_type(
            executor_configuration,
            engine_factory=engine_factory,
            reference_executor_factory=reference_executor_factory,
        )
        return executor_module.CachedContinuationExecutor(
            executor_configuration,
            session_factory=session_factory,
        )

    return factory


def produce_configured_authority(
    *,
    output_dir,
    configuration,
    engine_factory,
    reference_executor_factory=None,
):
    if reference_executor_factory is None:
        official = _load_module(
            "qwen35_tp4_engine_official_reference_executor",
            "qwen35_tp4_engine_official_reference_executor.py",
        )
        reference_executor_factory = (
            official.build_official_reference_executor_factory(
                configuration
            )
        )
    executor_factory = build_configured_executor_factory(
        configuration,
        engine_factory=engine_factory,
        reference_executor_factory=reference_executor_factory,
    )
    return produce_authority(
        output_dir=output_dir,
        source_tree_sha256=configuration.source_tree_sha256,
        model_manifest_sha256=configuration.model_manifest_sha256,
        executor_factory=executor_factory,
    )


def _default_executor_factory():
    raise RuntimeError(
        "real Qwen3.5 TP4 cached-continuation executor is not implemented"
    )


def _require_sha256(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _write_json(path, value):
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_verifier():
    return _load_module(
        "verify_qwen35_tp4_cached_continuation_correctness_gate",
        "verify_qwen35_tp4_cached_continuation_correctness_gate.py",
    )


def produce_authority(
    *,
    output_dir,
    source_tree_sha256,
    model_manifest_sha256,
    executor_factory=_default_executor_factory,
):
    output_dir = Path(output_dir)
    source_tree_sha256 = _require_sha256(
        source_tree_sha256,
        "source tree",
    )
    model_manifest_sha256 = _require_sha256(
        model_manifest_sha256,
        "model manifest",
    )
    if output_dir.exists():
        raise ValueError("output directory already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ))
    executor = None
    try:
        executor = executor_factory()
        rows = []
        for workload in contract.WORKLOADS:
            payload = contract.workload_payload(workload)
            continuation_count = payload["spec"]["continuations"]
            for request_index in range(continuation_count):
                rows.append(executor.run_continuation(
                    workload=workload,
                    request_index=request_index,
                    payload=payload,
                ))
        executor.close()
        executor = None
        classification = contract.classify_rows(rows)
        if classification["classification"] != "PASS":
            raise ValueError(
                "cached-continuation classification failed: "
                + "; ".join(classification["failures"])
            )
        result = {
            "schema_version": contract.SCHEMA_VERSION,
            "classification": "PASS",
            "model_manifest_sha256": model_manifest_sha256,
            "workload_manifest_sha256": (
                contract.WORKLOAD_MANIFEST_SHA256
            ),
            "rows": rows,
        }
        reference = {
            f"{row['workload']}:{row['request_index']}": (
                row["reference_output_token_ids"]
            )
            for row in rows
        }
        restored = {
            f"{row['workload']}:{row['request_index']}": (
                row["output_token_ids"]
            )
            for row in rows
        }
        logits = [{
            "workload": row["workload"],
            "request_index": row["request_index"],
            "max_abs_diff": row["logits_max_abs_diff"],
            "allclose": row["logits_allclose"],
        } for row in rows]
        _write_json(
            temporary_dir / "cached_continuation_correctness.json",
            result,
        )
        _write_json(
            temporary_dir / "reference_outputs.json",
            reference,
        )
        _write_json(
            temporary_dir / "restored_outputs.json",
            restored,
        )
        _write_json(
            temporary_dir / "registered_logits.json",
            logits,
        )
        _write_json(
            temporary_dir / "source_manifest.json",
            {
                "schema_version": contract.SCHEMA_VERSION,
                "source_tree_sha256": source_tree_sha256,
                "model_manifest_sha256": model_manifest_sha256,
                "workload_manifest_sha256": (
                    contract.WORKLOAD_MANIFEST_SHA256
                ),
                "files": {
                    name: _sha256(temporary_dir / name)
                    for name in contract.ARTIFACT_NAMES[:-1]
                },
            },
        )
        verification = _load_verifier().verify_run(temporary_dir)
        if verification["classification"] != "PASS":
            raise ValueError(
                "cached-continuation independent verification failed"
            )
        os.replace(temporary_dir, output_dir)
        return result
    finally:
        if executor is not None:
            executor.close()
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
