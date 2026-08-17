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
    "qwen35_tp4_engine_correctness_contract",
    "qwen35_tp4_engine_correctness_contract.py",
)


def _default_executor_factory():
    raise RuntimeError(
        "real Qwen3.5 TP4 Engine correctness executor is not implemented"
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
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_verifier():
    return _load_module(
        "verify_qwen35_tp4_engine_correctness_gate",
        "verify_qwen35_tp4_engine_correctness_gate.py",
    )


def build_configured_executor_factory(
    configuration,
    *,
    runtime_factory=None,
):
    executor_module = _load_module(
        "qwen35_tp4_engine_correctness_executor",
        "qwen35_tp4_engine_correctness_executor.py",
    )
    configuration = _normalize_executor_configuration(
        executor_module,
        configuration,
    )
    if runtime_factory is None:
        return executor_module.build_executor_factory(configuration)
    return executor_module.build_executor_factory(
        configuration,
        runtime_factory=runtime_factory,
    )


def _normalize_executor_configuration(
    executor_module,
    configuration,
):
    if not isinstance(
        configuration,
        executor_module.ExecutorConfiguration,
    ):
        to_payload = getattr(configuration, "to_payload", None)
        if not callable(to_payload):
            raise TypeError(
                "configuration must provide a canonical payload"
            )
        payload = to_payload()
        if not isinstance(payload, dict):
            raise TypeError(
                "configuration payload must be an object"
            )
        payload = dict(payload)
        if payload.pop("world_size", None) != contract.WORLD_SIZE:
            raise ValueError(
                "configuration world_size mismatch"
            )
        gpu_indices = payload.get("gpu_indices")
        if isinstance(gpu_indices, list):
            payload["gpu_indices"] = tuple(gpu_indices)
        configuration = executor_module.ExecutorConfiguration(
            **payload
        )
    return configuration


def build_audited_executor_factory(
    configuration,
    *,
    backend_factory=None,
):
    executor_module = _load_module(
        "qwen35_tp4_engine_correctness_executor",
        "qwen35_tp4_engine_correctness_executor.py",
    )
    configuration = _normalize_executor_configuration(
        executor_module,
        configuration,
    )

    def runtime_factory(runtime_configuration):
        if backend_factory is None:
            return executor_module.AuditedScenarioRuntime(
                runtime_configuration
            )
        return executor_module.AuditedScenarioRuntime(
            runtime_configuration,
            backend_factory=backend_factory,
        )

    return executor_module.build_executor_factory(
        configuration,
        runtime_factory=runtime_factory,
    )


def build_real_backend_factory(
    *,
    engine_factory=None,
    reference_token_provider,
):
    backend_module = _load_module(
        "qwen35_tp4_engine_backend_session",
        "qwen35_tp4_engine_backend_session.py",
    )
    if not callable(reference_token_provider):
        raise TypeError(
            "reference_token_provider must be callable"
        )

    def backend_factory(configuration, *, scenario, expected):
        kwargs = {
            "scenario": scenario,
            "expected": expected,
            "reference_token_provider": reference_token_provider,
        }
        if engine_factory is not None:
            if not callable(engine_factory):
                raise TypeError("engine_factory must be callable")
            kwargs["engine_factory"] = engine_factory
        return backend_module.EngineBackendSession(
            configuration,
            **kwargs,
        )

    return backend_factory


def build_source_bound_real_backend_factory(
    configuration,
    *,
    authority_dir,
    verification_path,
    engine_factory=None,
):
    executor_module = _load_module(
        "qwen35_tp4_engine_correctness_executor",
        "qwen35_tp4_engine_correctness_executor.py",
    )
    configuration = _normalize_executor_configuration(
        executor_module,
        configuration,
    )
    reference_module = _load_module(
        "qwen35_tp4_engine_reference_tokens",
        "qwen35_tp4_engine_reference_tokens.py",
    )
    reference_provider = (
        reference_module.build_reference_token_provider(
            authority_dir=authority_dir,
            verification_path=verification_path,
            configuration=configuration,
        )
    )
    return build_real_backend_factory(
        engine_factory=engine_factory,
        reference_token_provider=reference_provider,
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
        rows = [
            executor.run_scenario(
                scenario=scenario,
                expected=expected,
            )
            for scenario, expected in contract.SCENARIOS.items()
        ]
        classification = contract.classify_rows(rows)
        if classification["classification"] != "PASS":
            raise ValueError(
                "Engine correctness classification failed: "
                + "; ".join(classification["failures"])
            )
        result = {
            "schema_version": contract.SCHEMA_VERSION,
            "classification": "PASS",
            "model_manifest_sha256": model_manifest_sha256,
            "rows": rows,
        }
        scheduler_observations = [{
            "scenario": row["scenario"],
            "scheduler_steps": row["scheduler_steps"],
            "model_runner_calls": row["model_runner_calls"],
            "output_token_ids": row["output_token_ids"],
        } for row in rows]
        rank_events = [{
            "scenario": row["scenario"],
            "rank_inventory": row["rank_inventory"],
            "ack_ranks": row["ack_ranks"],
            "process_group_destroyed": row["process_group_destroyed"],
            "rank_exit_codes": row["rank_exit_codes"],
            "owned_children_remaining": row["owned_children_remaining"],
        } for row in rows]
        _write_json(
            temporary_dir / "engine_correctness.json",
            result,
        )
        _write_json(
            temporary_dir / "scheduler_observations.json",
            scheduler_observations,
        )
        _write_json(
            temporary_dir / "rank_events.json",
            rank_events,
        )
        _write_json(
            temporary_dir / "source_manifest.json",
            {
                "schema_version": contract.SCHEMA_VERSION,
                "source_tree_sha256": source_tree_sha256,
                "model_manifest_sha256": model_manifest_sha256,
                "files": {
                    name: _sha256(temporary_dir / name)
                    for name in contract.ARTIFACT_NAMES[:-1]
                },
            },
        )
        verification = _load_verifier().verify_run(temporary_dir)
        if verification["classification"] != "PASS":
            raise ValueError(
                "Engine correctness independent verification failed"
            )
        os.replace(temporary_dir, output_dir)
        return result
    finally:
        if executor is not None:
            executor.close()
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
