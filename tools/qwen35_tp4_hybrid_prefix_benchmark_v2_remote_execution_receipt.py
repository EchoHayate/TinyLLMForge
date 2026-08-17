from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile


def _load_contract():
    name = "qwen35_tp4_hybrid_prefix_benchmark_v2_contract_for_receipt"
    module = sys.modules.get(name)
    if module is not None:
        return module
    path = Path(__file__).resolve().parent / (
        "qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract()


def _physical_root_sha256(path):
    return contract.physical_directory_sha256(path)


def _validate_inventory(rows, expected_paths, label):
    if not isinstance(rows, list):
        raise ValueError(f"{label} inventory is invalid")
    paths = [row.get("path") for row in rows]
    if paths != sorted(expected_paths):
        raise ValueError(f"{label} inventory domain is invalid")
    for row in rows:
        if set(row) != {"path", "sha256", "bytes", "type"}:
            raise ValueError(f"{label} inventory row is invalid")
        if (
            row["type"] != "file"
            or
            not isinstance(row["sha256"], str)
            or len(row["sha256"]) != 64
            or type(row["bytes"]) is not int
            or row["bytes"] < 0
        ):
            raise ValueError(f"{label} inventory row is invalid")


def validate_detached_inventory_domains(
    payload,
    *,
    detached_receipt_path,
):
    package = payload.get("package_inventory")
    final = payload.get("final_inventory")
    _validate_inventory(
        package,
        contract.ARTIFACT_MANIFEST_HASH_DOMAIN,
        "package",
    )
    _validate_inventory(final, contract.PRODUCER_TRUST_DOMAIN, "final")
    detached = str(detached_receipt_path)
    if (
        not detached
        or detached in {row["path"] for row in package}
        or detached in {row["path"] for row in final}
        or detached == "artifact_manifest.json"
        or detached in contract.VERIFIER_TRUST_DOMAIN
    ):
        raise ValueError("detached receipt path creates a hash cycle")
    if set(contract.VERIFIER_TRUST_DOMAIN) & {
        row["path"] for row in package + final
    }:
        raise ValueError("producer inventory contains verifier output")
    final_without_manifest = [
        row for row in final if row["path"] != "artifact_manifest.json"
    ]
    if package != final_without_manifest:
        raise ValueError("package/final inventory equality mismatch")
    if payload.get("package_inventory_sha256") != (
        contract.canonical_json_sha256(package)
    ):
        raise ValueError("package inventory hash is invalid")
    if payload.get("final_inventory_sha256") != (
        contract.canonical_json_sha256(final)
    ):
        raise ValueError("final inventory hash is invalid")
    return payload


def publish_detached_execution_receipt(*, payload, run_dir, output_path):
    del payload, run_dir, output_path
    raise ValueError(
        "private detached receipt publisher is disabled; "
        "publish a complete execution evidence bundle"
    )


def _validate_detached_output(
    *,
    bundle,
    artifact_root,
    run_dir,
    output_path,
):
    run = Path(run_dir).resolve()
    raw_output = Path(output_path)
    if raw_output.is_symlink():
        raise ValueError("detached evidence output must not be a symlink")
    output = raw_output.resolve()
    artifact_root = Path(artifact_root).resolve()
    artifact_paths = bundle["execution_plan"]["artifact_paths"]
    if bundle["execution_plan"].get(
        "physical_artifact_root_sha256"
    ) != _physical_root_sha256(artifact_root):
        raise ValueError(
            "artifact root does not match execution plan identity"
        )
    bound_run = (
        artifact_root / Path(artifact_paths["local_extract"])
    ).resolve()
    if run != bound_run:
        raise ValueError("artifact root and run directory identity mismatch")
    try:
        output.relative_to(run)
    except ValueError:
        pass
    else:
        raise ValueError("detached evidence must be outside run directory")
    producer_and_verifier = {
        *contract.PRODUCER_TRUST_DOMAIN,
        *contract.VERIFIER_TRUST_DOMAIN,
    }
    if output.name in producer_and_verifier:
        raise ValueError(
            "detached evidence must be outside producer/verifier domains"
        )
    for name, value in artifact_paths.items():
        declared = Path(value)
        resolved = (
            declared.resolve()
            if declared.is_absolute()
            else (artifact_root / declared).resolve()
        )
        exclusions = [resolved]
        if name == "package":
            exclusions.append(resolved.parent)
        for excluded in exclusions:
            if output == excluded:
                raise ValueError(
                    "detached evidence overlaps an artifact path"
                )
            if name != "package" or excluded == resolved.parent:
                try:
                    output.relative_to(excluded)
                except ValueError:
                    pass
                else:
                    raise ValueError(
                        "detached evidence is inside an artifact path"
                    )
    if output.exists() or output.is_symlink():
        raise ValueError("detached evidence already exists")
    return output


def publish_execution_evidence_bundle(
    *,
    bundle,
    artifact_root,
    run_dir,
    output_path,
):
    contract.validate_execution_evidence_bundle(bundle)
    output = _validate_detached_output(
        bundle=bundle,
        artifact_root=artifact_root,
        run_dir=run_dir,
        output_path=output_path,
    )
    validate_detached_inventory_domains(
        bundle["execution_receipt"],
        detached_receipt_path=str(output),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    data = contract.canonical_json_bytes(bundle) + b"\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise ValueError(
                "detached evidence already exists"
            ) from error
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return bundle
