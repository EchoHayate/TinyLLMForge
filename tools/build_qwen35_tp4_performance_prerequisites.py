from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile

import qwen35_tp4_hybrid_prefix_benchmark_contract as contract


AUTHORITY_NAMES = (
    "tp4_root_logit",
    "cached_continuation",
    "engine_correctness",
)


@dataclass(frozen=True)
class AuthorityInput:
    name: str
    run_tag: str
    source_tree_sha256: str
    artifact_path: Path
    artifact_sha256: str
    independent_verification_path: Path
    independent_verification_sha256: str
    provenance_path: Path
    provenance_sha256: str


def _require_sha256(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} SHA is invalid")
    return value


def _regular_file(path, label):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file")
    return path


def _load_json(path, label):
    path = _regular_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} JSON is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} schema is invalid")
    return path, payload


def _atomic_write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def build_prerequisite_bundle(*, output_dir, authorities):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("prerequisite output already exists")
    if not isinstance(authorities, (tuple, list)):
        raise ValueError("authority inventory is invalid")
    by_name = {}
    for authority in authorities:
        if not isinstance(authority, AuthorityInput):
            raise ValueError("authority inventory is invalid")
        if authority.name in by_name:
            raise ValueError("authority inventory is invalid")
        by_name[authority.name] = authority
    if set(by_name) != set(AUTHORITY_NAMES):
        raise ValueError("authority inventory is invalid")

    prepared = {}
    for name in AUTHORITY_NAMES:
        authority = by_name[name]
        if (
            not isinstance(authority.run_tag, str)
            or not authority.run_tag
        ):
            raise ValueError(f"{name} run tag is invalid")
        source_tree_sha256 = _require_sha256(
            authority.source_tree_sha256,
            f"{name} source tree",
        )
        if (
            name == "tp4_root_logit"
            and source_tree_sha256
            != contract.TP4_ROOT_SOURCE_TREE_SHA256
        ):
            raise ValueError("root-logit source tree mismatch")
        artifact, artifact_payload = _load_json(
            authority.artifact_path,
            f"{name} artifact",
        )
        artifact_sha256 = _require_sha256(
            authority.artifact_sha256,
            f"{name} artifact",
        )
        if contract.sha256_file(artifact) != artifact_sha256:
            raise ValueError(f"{name} artifact SHA mismatch")
        verification, verification_payload = _load_json(
            authority.independent_verification_path,
            f"{name} independent verification",
        )
        verification_sha256 = _require_sha256(
            authority.independent_verification_sha256,
            f"{name} independent verification",
        )
        if contract.sha256_file(verification) != verification_sha256:
            raise ValueError(
                f"{name} independent verification SHA mismatch"
            )
        provenance, provenance_payload = _load_json(
            authority.provenance_path,
            f"{name} provenance",
        )
        provenance_sha256 = _require_sha256(
            authority.provenance_sha256,
            f"{name} provenance",
        )
        if contract.sha256_file(provenance) != provenance_sha256:
            raise ValueError(f"{name} provenance SHA mismatch")
        contract.validate_authority_documents(
            name,
            artifact_payload,
            verification_payload,
            source_tree_sha256,
        )
        contract._validate_prerequisite_provenance(
            name,
            provenance_payload,
            run_tag=authority.run_tag,
            source_tree_sha256=source_tree_sha256,
        )
        provenance_evidence = {}
        for path_field, sha_field, label in (
            ("plan_path", "plan_sha256", "execution plan"),
            (
                "authorization_path",
                "authorization_sha256",
                "consumed authorization",
            ),
            (
                "receipt_path",
                "receipt_sha256",
                "execution receipt",
            ),
        ):
            relative = provenance_payload[path_field]
            evidence = _regular_file(
                provenance.parent / relative,
                f"{name} {label}",
            )
            if (
                contract.sha256_file(evidence)
                != provenance_payload[sha_field]
            ):
                raise ValueError(f"{name} {label} SHA mismatch")
            provenance_evidence[relative] = evidence
        prepared[name] = (
            authority,
            artifact,
            verification,
            provenance,
            provenance_evidence,
        )

    output_dir.mkdir(parents=True)
    try:
        rows = {}
        for name in AUTHORITY_NAMES:
            (
                authority,
                artifact,
                verification,
                provenance,
                provenance_evidence,
            ) = prepared[name]
            destination = output_dir / "prerequisites" / name
            destination.mkdir(parents=True)
            copied_artifact = destination / "artifact.json"
            copied_verification = (
                destination / "independent_verification.json"
            )
            copied_provenance = destination / "provenance.json"
            shutil.copyfile(artifact, copied_artifact)
            shutil.copyfile(verification, copied_verification)
            shutil.copyfile(provenance, copied_provenance)
            for relative, evidence in provenance_evidence.items():
                shutil.copyfile(evidence, destination / relative)
            rows[name] = {
                "run_tag": authority.run_tag,
                "source_tree_sha256": authority.source_tree_sha256,
                "artifact_path": (
                    copied_artifact.relative_to(output_dir).as_posix()
                ),
                "artifact_sha256": contract.sha256_file(
                    copied_artifact
                ),
                "independent_verification_path": (
                    copied_verification
                    .relative_to(output_dir)
                    .as_posix()
                ),
                "independent_verification_sha256": (
                    contract.sha256_file(copied_verification)
                ),
                "provenance_path": (
                    copied_provenance
                    .relative_to(output_dir)
                    .as_posix()
                ),
                "provenance_sha256": contract.sha256_file(
                    copied_provenance
                ),
                "classification": "PASS",
            }
        prerequisite_path = (
            output_dir / "correctness_prerequisites.json"
        )
        _atomic_write_json(
            prerequisite_path,
            {
                "schema_version": contract.PREREQUISITE_SCHEMA_VERSION,
                "model_manifest_sha256": (
                    contract.MODEL_MANIFEST_SHA256
                ),
                **rows,
            },
        )
        status = contract.validate_prerequisites(prerequisite_path)
        if not status.authorized:
            raise RuntimeError(
                "built prerequisite bundle failed validation: "
                + "; ".join(status.reasons)
            )
        return {
            "classification": "PASS",
            "correctness_prerequisites_sha256": (
                contract.sha256_file(prerequisite_path)
            ),
        }
    except BaseException:
        shutil.rmtree(output_dir)
        raise
