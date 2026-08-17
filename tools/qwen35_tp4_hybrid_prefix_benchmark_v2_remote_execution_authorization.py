from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys


def _load_contract():
    name = "qwen35_tp4_hybrid_prefix_benchmark_v2_contract_for_remote_auth"
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


def _validate_declared_path(root, path, logical_path, label):
    root = Path(os.path.abspath(os.fspath(root)))
    logical = Path(logical_path)
    if logical.is_absolute() or ".." in logical.parts:
        raise ValueError(f"{label} logical path is invalid")
    actual = Path(os.path.abspath(os.fspath(path)))
    expected = Path(os.path.abspath(os.fspath(root / logical)))
    if actual != expected:
        raise ValueError(f"{label} path is outside authority root")
    return logical


def _open_rooted_parent(root_fd, logical_path, *, create):
    current = os.dup(root_fd)
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        parts = Path(logical_path).parts
        for part in parts[:-1]:
            try:
                child = os.open(part, flags, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, dir_fd=current)
                except FileExistsError:
                    pass
                child = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = child
        return current, parts[-1]
    except (OSError, ValueError) as error:
        os.close(current)
        raise ValueError(
            "authorization path contains an invalid authority root directory"
        ) from error


def _lstat_at(parent_fd, name):
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _regular_file_exists_at(parent_fd, name):
    metadata = _lstat_at(parent_fd, name)
    return metadata is not None and stat.S_ISREG(metadata.st_mode)


def _read_json_at(parent_fd, name, label):
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError(f"{label} is invalid") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} is invalid")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    try:
        return json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error


def _write_temp_at(parent_fd, name, payload):
    data = contract.canonical_json_bytes(payload) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    for _ in range(128):
        temporary = f".{name}.{secrets.token_hex(16)}"
        try:
            descriptor = os.open(
                temporary,
                flags,
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            continue
        try:
            offset = 0
            while offset < len(data):
                offset += os.write(descriptor, data[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return temporary
    raise ValueError("authorization temporary output cannot be created")


def _atomic_write_at(parent_fd, name, payload):
    if _lstat_at(parent_fd, name) is not None:
        raise ValueError("authorization output already exists")
    temporary = _write_temp_at(parent_fd, name, payload)
    try:
        try:
            os.link(
                temporary,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise ValueError(
                "authorization output already exists"
            ) from error
        os.fsync(parent_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass


def _authorization_tombstone(authority_root, plan):
    value = (
        f"{contract.canonical_json_sha256(plan)}-"
        f"{hashlib.sha256(str(plan['nonce']).encode()).hexdigest()}"
    )
    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789-_"
    )
    if not value or any(character not in allowed for character in value):
        raise ValueError("authorization id is invalid")
    return Path(".consumed") / f"{value}.json"


def _bound_authorization_id(authorization_id, root_digest):
    value = str(authorization_id)
    suffix = root_digest
    if value.endswith(f"-root-{suffix}"):
        return value
    return f"{value}-root-{suffix}"


def _validate_authority_root_identity(plan, root_fd, resolved_root):
    actual = contract.physical_directory_fd_sha256(root_fd, resolved_root)
    if plan.get("authority_root_sha256") != actual:
        raise ValueError("authorization root does not match execution plan")
    return actual


def _logical_record_path(path):
    value = Path(path)
    if value.is_absolute():
        return f"authority/{value.name}"
    return value.as_posix()


def build_authorization(
    *,
    plan,
    authorization_id,
    active_path,
    consumed_path,
):
    payload = {
        "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[
            "consumed_authorization"
        ],
        "run_tag": plan["run_tag"],
        "nonce": plan["nonce"],
        **{
            field: plan[field]
            for field in contract.EXECUTION_PROVENANCE_FIELDS
        },
        "execution_plan_sha256": contract.canonical_json_sha256(plan),
        "required_gpu_indices": list(plan["required_gpu_indices"]),
        "world_size": plan["world_size"],
        "gpu_assignments": list(plan["gpu_assignments"]),
        "case_port_pairs": list(plan["case_port_pairs"]),
        "artifact_paths": dict(plan["artifact_paths"]),
        "authorization_id": authorization_id,
        "active_path": _logical_record_path(active_path),
        "consumed_path": _logical_record_path(consumed_path),
        "consumed": False,
        "consumed_once": False,
    }
    validate_authorization(plan=plan, payload=payload)
    return payload


def validate_authorization(*, plan, payload):
    expected = build_authorization.__wrapped__(
        plan=plan,
        authorization_id=payload.get("authorization_id"),
        active_path=payload.get("active_path"),
        consumed_path=payload.get("consumed_path"),
    )
    if payload != expected:
        raise ValueError("authorization does not match execution plan")
    return payload


def _build_without_validation(
    *,
    plan,
    authorization_id,
    active_path,
    consumed_path,
):
    return {
        "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[
            "consumed_authorization"
        ],
        "run_tag": plan["run_tag"],
        "nonce": plan["nonce"],
        **{
            field: plan[field]
            for field in contract.EXECUTION_PROVENANCE_FIELDS
        },
        "execution_plan_sha256": contract.canonical_json_sha256(plan),
        "required_gpu_indices": list(plan["required_gpu_indices"]),
        "world_size": plan["world_size"],
        "gpu_assignments": list(plan["gpu_assignments"]),
        "case_port_pairs": list(plan["case_port_pairs"]),
        "artifact_paths": dict(plan["artifact_paths"]),
        "authorization_id": authorization_id,
        "active_path": _logical_record_path(active_path),
        "consumed_path": _logical_record_path(consumed_path),
        "consumed": False,
        "consumed_once": False,
    }


build_authorization.__wrapped__ = _build_without_validation


def produce_authorization(
    *,
    plan,
    output_path,
    authorization_id,
    consumed_path,
    authority_root,
):
    logical_active = _logical_record_path(output_path)
    logical_consumed = _logical_record_path(consumed_path)
    active_logical = _validate_declared_path(
        authority_root,
        output_path,
        logical_active,
        "active authorization",
    )
    root_fd, resolved_root = contract.open_physical_directory(
        authority_root
    )
    try:
        root_digest = _validate_authority_root_identity(
            plan,
            root_fd,
            resolved_root,
        )
        bound_authorization_id = _bound_authorization_id(
            authorization_id,
            root_digest,
        )
        tombstone_logical = _authorization_tombstone(
            authority_root,
            plan,
        )
        tombstone_parent_fd, tombstone_name = _open_rooted_parent(
            root_fd,
            tombstone_logical,
            create=True,
        )
        try:
            if _lstat_at(tombstone_parent_fd, tombstone_name) is not None:
                raise ValueError("authorization was already consumed")
        finally:
            os.close(tombstone_parent_fd)
        payload = build_authorization(
            plan=plan,
            authorization_id=bound_authorization_id,
            active_path=logical_active,
            consumed_path=logical_consumed,
        )
        active_parent_fd, active_name = _open_rooted_parent(
            root_fd,
            active_logical,
            create=True,
        )
        try:
            _atomic_write_at(active_parent_fd, active_name, payload)
        finally:
            os.close(active_parent_fd)
        return payload
    finally:
        os.close(root_fd)


def consume_authorization(
    *,
    plan,
    active_path,
    consumed_path,
    active_record_path,
    consumed_record_path,
    authority_root,
):
    active_logical = _validate_declared_path(
        authority_root,
        active_path,
        active_record_path,
        "active authorization",
    )
    consumed_logical = _validate_declared_path(
        authority_root,
        consumed_path,
        consumed_record_path,
        "consumed authorization",
    )
    root_fd, resolved_root = contract.open_physical_directory(
        authority_root,
    )
    active_parent_fd = None
    consumed_parent_fd = None
    tombstone_parent_fd = None
    try:
        _validate_authority_root_identity(plan, root_fd, resolved_root)
        active_parent_fd, active_name = _open_rooted_parent(
            root_fd,
            active_logical,
            create=False,
        )
        consumed_parent_fd, consumed_name = _open_rooted_parent(
            root_fd,
            consumed_logical,
            create=True,
        )
        claim_name = f".{active_name}.consuming"
        tombstone_logical = _authorization_tombstone(
            authority_root,
            plan,
        )
        tombstone_parent_fd, tombstone_name = _open_rooted_parent(
            root_fd,
            tombstone_logical,
            create=True,
        )
        if _lstat_at(consumed_parent_fd, consumed_name) is not None:
            raise ValueError("authorization cannot be consumed")
        if _regular_file_exists_at(active_parent_fd, active_name):
            source_name = active_name
        elif _regular_file_exists_at(active_parent_fd, claim_name):
            source_name = claim_name
        else:
            raise ValueError("authorization cannot be consumed")
        payload = _read_json_at(
            active_parent_fd,
            source_name,
            "authorization cannot be consumed",
        )
        validate_authorization(plan=plan, payload=payload)
        if _lstat_at(tombstone_parent_fd, tombstone_name) is not None:
            recovered = _read_json_at(
                tombstone_parent_fd,
                tombstone_name,
                "authorization tombstone",
            )
            contract.validate_evidence_document(
                "consumed_authorization",
                recovered,
            )
            if recovered != {
                **payload,
                "consumed": True,
                "consumed_once": True,
            }:
                raise ValueError(
                    "authorization tombstone identity mismatch"
                )
            try:
                os.link(
                    tombstone_name,
                    consumed_name,
                    src_dir_fd=tombstone_parent_fd,
                    dst_dir_fd=consumed_parent_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as error:
                raise ValueError(
                    "authorization was already consumed"
                ) from error
            if _lstat_at(active_parent_fd, claim_name) is not None:
                os.unlink(claim_name, dir_fd=active_parent_fd)
            return recovered
        active_parent_stat = os.fstat(active_parent_fd)
        consumed_parent_stat = os.fstat(consumed_parent_fd)
        if (
            active_parent_stat.st_dev,
            active_parent_stat.st_ino,
        ) != (
            consumed_parent_stat.st_dev,
            consumed_parent_stat.st_ino,
        ):
            raise ValueError(
                "authorization consume paths must share a directory"
            )
        if (
            str(active_record_path) != payload["active_path"]
            or str(consumed_record_path) != payload["consumed_path"]
            or active_name != Path(payload["active_path"]).name
            or consumed_name != Path(payload["consumed_path"]).name
        ):
            raise ValueError("authorization record path binding mismatch")
        record = {
            **payload,
            "consumed": True,
            "consumed_once": True,
        }
        contract.validate_evidence_document(
            "consumed_authorization",
            record,
        )
        installed = False
        temporary = None
        try:
            if source_name == active_name:
                if _lstat_at(active_parent_fd, claim_name) is not None:
                    raise ValueError(
                        "authorization is already being consumed"
                    )
                os.replace(
                    active_name,
                    claim_name,
                    src_dir_fd=active_parent_fd,
                    dst_dir_fd=active_parent_fd,
                )
            temporary = _write_temp_at(
                consumed_parent_fd,
                consumed_name,
                record,
            )
            try:
                os.link(
                    temporary,
                    tombstone_name,
                    src_dir_fd=consumed_parent_fd,
                    dst_dir_fd=tombstone_parent_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as error:
                raise ValueError(
                    "authorization was already consumed"
                ) from error
            try:
                os.link(
                    tombstone_name,
                    consumed_name,
                    src_dir_fd=tombstone_parent_fd,
                    dst_dir_fd=consumed_parent_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as error:
                raise ValueError(
                    "authorization was already consumed"
                ) from error
            os.fsync(tombstone_parent_fd)
            os.fsync(consumed_parent_fd)
            installed = True
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary, dir_fd=consumed_parent_fd)
                except FileNotFoundError:
                    pass
            if _lstat_at(active_parent_fd, claim_name) is not None:
                if installed or (
                    _lstat_at(consumed_parent_fd, consumed_name) is not None
                ):
                    os.unlink(claim_name, dir_fd=active_parent_fd)
                elif _lstat_at(active_parent_fd, active_name) is None:
                    os.replace(
                        claim_name,
                        active_name,
                        src_dir_fd=active_parent_fd,
                        dst_dir_fd=active_parent_fd,
                    )
        return record
    finally:
        for descriptor in (
            tombstone_parent_fd,
            consumed_parent_fd,
            active_parent_fd,
            root_fd,
        ):
            if descriptor is not None:
                os.close(descriptor)
