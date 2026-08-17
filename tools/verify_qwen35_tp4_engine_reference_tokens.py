from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
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
executor = _load_module(
    "qwen35_tp4_engine_correctness_executor",
    "qwen35_tp4_engine_correctness_executor.py",
)


class VerificationError(RuntimeError):
    pass


def _fail(message):
    raise VerificationError(message)


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"invalid JSON artifact: {path.name}: {error}")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_sha256(tokens):
    return hashlib.sha256(
        json.dumps(
            list(tokens),
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(
            character in "0123456789abcdef"
            for character in value
        )
    )


def verify_run(run_dir):
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        _fail("reference authority directory is missing")
    entries = list(run_dir.iterdir())
    if (
        any(entry.is_symlink() or not entry.is_file() for entry in entries)
        or {entry.name for entry in entries}
        != set(reference.ARTIFACT_NAMES)
    ):
        _fail("reference artifact inventory mismatch")

    result_path = run_dir / "reference_tokens.json"
    manifest_path = run_dir / "source_manifest.json"
    result = _load_json(result_path)
    manifest = _load_json(manifest_path)
    identity_fields = (
        "model_manifest_sha256",
        "source_tree_sha256",
        "workload_manifest_sha256",
    )
    result_required = {
        "schema_version",
        "classification",
        "reference_backend",
        "generation_policy",
        *identity_fields,
        "rows",
    }
    if (
        not isinstance(result, dict)
        or set(result) != result_required
        or result["schema_version"] != reference.SCHEMA_VERSION
        or result["classification"] != "PASS"
        or result["reference_backend"]
        != reference.REFERENCE_BACKEND
        or result["generation_policy"]
        != reference.GENERATION_POLICY
    ):
        _fail("reference result schema or semantics mismatch")

    manifest_required = {
        "schema_version",
        *identity_fields,
        "files",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != manifest_required
        or manifest["schema_version"] != reference.SCHEMA_VERSION
        or manifest["files"]
        != {"reference_tokens.json": _sha256(result_path)}
    ):
        _fail("reference source manifest mismatch")
    for name in identity_fields:
        if (
            not _valid_sha256(result[name])
            or result[name] != manifest[name]
        ):
            _fail(f"reference {name} mismatch")

    rows = result["rows"]
    if (
        not isinstance(rows, list)
        or len(rows) != len(reference.REFERENCE_SCENARIOS)
    ):
        _fail("reference scenario inventory mismatch")
    payloads = executor.build_scenario_payloads()
    row_fields = {
        "scenario",
        "prompt_token_count",
        "prompt_token_ids_sha256",
        "generated_tokens",
        "output_token_ids",
    }
    for scenario, row in zip(reference.REFERENCE_SCENARIOS, rows):
        if (
            not isinstance(row, dict)
            or set(row) != row_fields
            or row["scenario"] != scenario
        ):
            _fail("reference scenario row mismatch")
        payload = payloads[scenario]
        prompt = (
            payload["source_prompt_token_ids"]
            if scenario == "publish_source"
            else payload["request_prompt_token_ids"]
        )
        generated_tokens = payload["generated_tokens"]
        if (
            row["prompt_token_count"] != len(prompt)
            or row["prompt_token_ids_sha256"]
            != _token_sha256(prompt)
        ):
            _fail(f"reference prompt mismatch: {scenario}")
        output = row["output_token_ids"]
        if (
            row["generated_tokens"] != generated_tokens
            or not isinstance(output, list)
            or len(output) != generated_tokens
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in output
            )
        ):
            _fail(f"reference output mismatch: {scenario}")

    return {
        "schema_version": reference.SCHEMA_VERSION,
        "classification": "PASS",
        **{
            name: result[name]
            for name in identity_fields
        },
        "reference_tokens_sha256": _sha256(result_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "scenario_count": len(rows),
    }


def verify_and_write(run_dir, *, output_path):
    run_dir = Path(run_dir).resolve()
    output_path = Path(output_path).resolve()
    try:
        output_path.relative_to(run_dir)
    except ValueError:
        pass
    else:
        raise ValueError(
            "independent verification output must be outside reference run"
        )
    if output_path.exists():
        raise ValueError(
            "independent verification output already exists"
        )
    result = verify_run(run_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            result,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return result
