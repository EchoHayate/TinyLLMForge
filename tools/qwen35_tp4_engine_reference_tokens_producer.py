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


reference = _load_module(
    "qwen35_tp4_engine_reference_tokens",
    "qwen35_tp4_engine_reference_tokens.py",
)
executor_module = _load_module(
    "qwen35_tp4_engine_correctness_executor",
    "qwen35_tp4_engine_correctness_executor.py",
)
verifier = _load_module(
    "verify_qwen35_tp4_engine_reference_tokens",
    "verify_qwen35_tp4_engine_reference_tokens.py",
)


def _default_executor_factory():
    raise RuntimeError(
        "official Qwen3.5 reference executor is not implemented"
    )


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


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_sha256(tokens):
    return hashlib.sha256(
        json.dumps(
            list(tokens),
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _normalize_configuration(configuration):
    if isinstance(
        configuration,
        executor_module.ExecutorConfiguration,
    ):
        return configuration
    to_payload = getattr(configuration, "to_payload", None)
    if not callable(to_payload):
        raise TypeError(
            "configuration must provide a canonical payload"
        )
    payload = dict(to_payload())
    if payload.pop("world_size", None) != 4:
        raise ValueError("configuration world_size mismatch")
    if isinstance(payload.get("gpu_indices"), list):
        payload["gpu_indices"] = tuple(payload["gpu_indices"])
    return executor_module.ExecutorConfiguration(**payload)


def produce_reference_authority(
    *,
    output_dir,
    verification_path,
    configuration,
    executor_factory=_default_executor_factory,
):
    output_dir = Path(output_dir)
    verification_path = Path(verification_path)
    configuration = _normalize_configuration(configuration)
    if output_dir.exists():
        raise ValueError("output directory already exists")
    if verification_path.exists():
        raise ValueError(
            "independent verification output already exists"
        )
    try:
        verification_path.resolve().relative_to(
            output_dir.resolve()
        )
    except ValueError:
        pass
    else:
        raise ValueError(
            "independent verification output must be outside reference run"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    verification_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ))
    temporary_verification = verification_path.parent / (
        f".{verification_path.name}.{os.getpid()}.tmp"
    )
    executor = None
    try:
        executor = executor_factory()
        payloads = executor_module.build_scenario_payloads()
        rows = []
        for scenario in reference.REFERENCE_SCENARIOS:
            payload = payloads[scenario]
            prompt = (
                payload["source_prompt_token_ids"]
                if scenario == "publish_source"
                else payload["request_prompt_token_ids"]
            )
            generated_tokens = payload["generated_tokens"]
            output = executor.generate_reference(
                scenario=scenario,
                prompt_token_ids=list(prompt),
                generated_tokens=generated_tokens,
                generation_policy=dict(
                    reference.GENERATION_POLICY
                ),
            )
            if (
                not isinstance(output, (list, tuple))
                or len(output) != generated_tokens
                or any(
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or token_id < 0
                    for token_id in output
                )
            ):
                raise ValueError(
                    f"reference output is invalid: {scenario}"
                )
            rows.append({
                "scenario": scenario,
                "prompt_token_count": len(prompt),
                "prompt_token_ids_sha256": _token_sha256(prompt),
                "generated_tokens": generated_tokens,
                "output_token_ids": list(output),
            })
        result = {
            "schema_version": reference.SCHEMA_VERSION,
            "classification": "PASS",
            "reference_backend": reference.REFERENCE_BACKEND,
            "generation_policy": dict(
                reference.GENERATION_POLICY
            ),
            "model_manifest_sha256": (
                configuration.model_manifest_sha256
            ),
            "source_tree_sha256": (
                configuration.source_tree_sha256
            ),
            "workload_manifest_sha256": (
                configuration.workload_manifest_sha256
            ),
            "rows": rows,
        }
        _write_json(
            temporary_dir / "reference_tokens.json",
            result,
        )
        _write_json(
            temporary_dir / "source_manifest.json",
            {
                "schema_version": reference.SCHEMA_VERSION,
                "model_manifest_sha256": (
                    configuration.model_manifest_sha256
                ),
                "source_tree_sha256": (
                    configuration.source_tree_sha256
                ),
                "workload_manifest_sha256": (
                    configuration.workload_manifest_sha256
                ),
                "files": {
                    "reference_tokens.json": _sha256(
                        temporary_dir / "reference_tokens.json"
                    ),
                },
            },
        )
        verification = verifier.verify_and_write(
            temporary_dir,
            output_path=temporary_verification,
        )
        os.replace(temporary_dir, output_dir)
        os.replace(
            temporary_verification,
            verification_path,
        )
        return {
            **result,
            "independent_verification": verification,
        }
    finally:
        if executor is not None:
            executor.close()
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        if temporary_verification.exists():
            temporary_verification.unlink()
