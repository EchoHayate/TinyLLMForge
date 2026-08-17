from __future__ import annotations

import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = "qwen35.tp4-engine-reference-tokens.v1"
REFERENCE_BACKEND = "official_huggingface_qwen35_greedy"
GENERATION_POLICY = {
    "temperature": 0.0,
    "ignore_eos": True,
}
REFERENCE_SCENARIOS = (
    "publish_source",
    "restore_w1",
    "miss_w4_token",
    "miss_w4_stale",
    "miss_w4_clear",
)
ARTIFACT_NAMES = (
    "reference_tokens.json",
    "source_manifest.json",
)


def _load_json(path, label):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_sha256(tokens):
    return hashlib.sha256(
        json.dumps(
            list(tokens),
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


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


def _validate_rows(rows):
    if not isinstance(rows, list) or len(rows) != len(
        REFERENCE_SCENARIOS
    ):
        raise ValueError("reference scenario inventory mismatch")
    normalized = {}
    required = {
        "scenario",
        "prompt_token_count",
        "prompt_token_ids_sha256",
        "generated_tokens",
        "output_token_ids",
    }
    for expected_scenario, row in zip(REFERENCE_SCENARIOS, rows):
        if (
            not isinstance(row, dict)
            or set(row) != required
            or row["scenario"] != expected_scenario
        ):
            raise ValueError("reference scenario row mismatch")
        prompt_count = row["prompt_token_count"]
        generated_tokens = row["generated_tokens"]
        output = row["output_token_ids"]
        if (
            isinstance(prompt_count, bool)
            or not isinstance(prompt_count, int)
            or prompt_count <= 0
        ):
            raise ValueError("reference prompt count is invalid")
        _require_sha256(
            row["prompt_token_ids_sha256"],
            "prompt token identity",
        )
        if (
            isinstance(generated_tokens, bool)
            or not isinstance(generated_tokens, int)
            or generated_tokens <= 0
            or not isinstance(output, list)
            or len(output) != generated_tokens
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in output
            )
        ):
            raise ValueError("reference generated token evidence is invalid")
        normalized[expected_scenario] = {
            **row,
            "output_token_ids": list(output),
        }
    return normalized


def build_reference_token_provider(
    *,
    authority_dir,
    verification_path,
    configuration,
):
    authority_dir = Path(authority_dir)
    verification_path = Path(verification_path)
    if not authority_dir.is_dir():
        raise ValueError("reference authority directory is missing")
    entries = list(authority_dir.iterdir())
    if (
        any(entry.is_symlink() or not entry.is_file() for entry in entries)
        or {entry.name for entry in entries} != set(ARTIFACT_NAMES)
    ):
        raise ValueError("reference authority inventory mismatch")
    if (
        not verification_path.is_file()
        or verification_path.is_symlink()
    ):
        raise ValueError(
            "independent reference verification is missing"
        )

    result_path = authority_dir / "reference_tokens.json"
    manifest_path = authority_dir / "source_manifest.json"
    result = _load_json(result_path, "reference tokens")
    manifest = _load_json(manifest_path, "source manifest")
    verification = _load_json(
        verification_path,
        "independent verification",
    )

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
    if not isinstance(result, dict) or set(result) != result_required:
        raise ValueError("reference token result schema mismatch")
    if (
        result["schema_version"] != SCHEMA_VERSION
        or result["classification"] != "PASS"
        or result["reference_backend"] != REFERENCE_BACKEND
        or result["generation_policy"] != GENERATION_POLICY
    ):
        raise ValueError("reference token result semantics mismatch")

    manifest_required = {
        "schema_version",
        *identity_fields,
        "files",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != manifest_required
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["files"]
        != {"reference_tokens.json": _sha256(result_path)}
    ):
        raise ValueError("reference source manifest hash mismatch")

    verification_required = {
        "schema_version",
        "classification",
        *identity_fields,
        "reference_tokens_sha256",
        "source_manifest_sha256",
        "scenario_count",
    }
    if (
        not isinstance(verification, dict)
        or set(verification) != verification_required
        or verification["schema_version"] != SCHEMA_VERSION
        or verification["classification"] != "PASS"
        or verification["reference_tokens_sha256"]
        != _sha256(result_path)
        or verification["source_manifest_sha256"]
        != _sha256(manifest_path)
        or verification["scenario_count"]
        != len(REFERENCE_SCENARIOS)
    ):
        raise ValueError(
            "independent reference verification hash mismatch"
        )

    expected_identities = {
        "model_manifest_sha256": (
            configuration.model_manifest_sha256
        ),
        "source_tree_sha256": configuration.source_tree_sha256,
        "workload_manifest_sha256": (
            configuration.workload_manifest_sha256
        ),
    }
    labels = {
        "model_manifest_sha256": "model manifest",
        "source_tree_sha256": "source tree",
        "workload_manifest_sha256": "workload manifest",
    }
    for name, expected in expected_identities.items():
        _require_sha256(expected, labels[name])
        if (
            result[name] != expected
            or manifest[name] != expected
            or verification[name] != expected
        ):
            raise ValueError(
                f"reference {labels[name]} identity mismatch"
            )

    rows = _validate_rows(result["rows"])

    def provider(*, scenario, prompt_token_ids, generated_tokens):
        if scenario not in rows:
            raise ValueError(
                f"unsupported reference scenario: {scenario}"
            )
        if not isinstance(prompt_token_ids, (list, tuple)):
            raise ValueError("reference prompt token IDs are invalid")
        row = rows[scenario]
        if (
            len(prompt_token_ids) != row["prompt_token_count"]
            or _token_sha256(prompt_token_ids)
            != row["prompt_token_ids_sha256"]
        ):
            raise ValueError("reference prompt identity mismatch")
        if generated_tokens != row["generated_tokens"]:
            raise ValueError("reference generated token count mismatch")
        return list(row["output_token_ids"])

    return provider
