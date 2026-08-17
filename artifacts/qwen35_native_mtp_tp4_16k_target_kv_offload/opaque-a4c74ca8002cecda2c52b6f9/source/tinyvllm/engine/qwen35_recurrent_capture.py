from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import uuid

import torch

from tinyvllm.engine.qwen35_recurrent_capture_contract import (
    CaptureRunIdentity,
    CapturedTensorRecord,
    canonical_json_bytes,
    validate_run_identity,
    validate_tensor_record,
)


def save_tensor(tensor, path):
    torch.save(tensor, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_regular_file(path):
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _new_temporary_path(final_path):
    return final_path.with_name(
        f".{final_path.name}.tmp-{uuid.uuid4().hex}"
    )


def _write_atomic_bytes(final_path, payload):
    if final_path.exists() or final_path.is_symlink():
        raise ValueError(f"pre-existing final path: {final_path}")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _new_temporary_path(final_path)
    try:
        temporary_path.write_bytes(payload)
        _fsync_regular_file(temporary_path)
        _publish_no_clobber(temporary_path, final_path)
    finally:
        if temporary_path.exists() or temporary_path.is_symlink():
            temporary_path.unlink()


def _publish_no_clobber(temporary_path, final_path):
    library = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(temporary_path)
    target = os.fsencode(final_path)
    if sys.platform == "darwin":
        rename = library.renamex_np
        rename.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        result = rename(source, target, 0x00000004)
    elif sys.platform.startswith("linux"):
        rename = library.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        result = rename(-100, source, -100, target, 1)
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace rename is unsupported",
            final_path,
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            final_path,
        )


def _validate_workload_component(workload_id):
    if (
        not isinstance(workload_id, str)
        or not workload_id
        or "\0" in workload_id
        or "\\" in workload_id
    ):
        raise ValueError("workload_id must be a safe path component")
    path = PurePosixPath(workload_id)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or path.parts[0] in {".", ".."}
        or path.as_posix() != workload_id
    ):
        raise ValueError("workload_id must be a safe path component")


def _validate_capture_root(capture_root):
    if str(capture_root) == "":
        raise ValueError("capture root must be a non-empty path")
    if "\0" in str(capture_root):
        raise ValueError("capture root must not contain NUL")
    if ".." in capture_root.parts:
        raise ValueError("capture root must not contain traversal")
    absolute_root = capture_root.absolute()
    current = Path(absolute_root.anchor)
    for component in absolute_root.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("capture root must not contain symlinks")


def _is_generated_temporary_path(path):
    return re.fullmatch(
        r"\..+\.tmp-[0-9a-f]{32}",
        path.name,
    ) is not None


class Qwen35RecurrentCaptureSession:
    def __init__(self, *, run_identity, rank, staging_dir=None):
        if isinstance(run_identity, CaptureRunIdentity):
            identity = validate_run_identity(run_identity.payload())
        else:
            identity = validate_run_identity(run_identity)
        for workload_id in identity.workload_ids:
            _validate_workload_component(workload_id)
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
            or rank >= identity.world_size
        ):
            raise ValueError("rank must be below world_size")

        self.run_identity = identity
        self.rank = rank
        if staging_dir == "":
            raise ValueError("capture root must be a non-empty path")
        self.capture_root = Path(
            staging_dir if staging_dir is not None else "."
        )
        _validate_capture_root(self.capture_root)
        self.rank_root = self.capture_root / f"rank{rank}"
        self.tensor_root = self.rank_root / "tensors"
        self.identity_path = self.rank_root / "capture_identity.json"
        self.records = {}
        self._completed_workloads = set()

        if self.rank_root.is_symlink():
            raise ValueError("rank root must not be a symlink")
        self.rank_root.mkdir(parents=True, exist_ok=True)
        if self.tensor_root.is_symlink():
            raise ValueError("tensor root must not be a symlink")
        self.tensor_root.mkdir(parents=True, exist_ok=True)
        if not self.tensor_root.is_dir():
            raise ValueError("tensor root must be a directory")
        self._publish_or_validate_identity()
        self._scan_existing_rank_root()

    def _publish_or_validate_identity(self):
        expected = canonical_json_bytes(
            self.run_identity.payload()
        ) + b"\n"
        if self.identity_path.is_symlink():
            raise ValueError("identity path must not be a symlink")
        if self.identity_path.exists():
            if (
                not self.identity_path.is_file()
                or self.identity_path.read_bytes() != expected
            ):
                raise ValueError("capture identity mismatch")
            return
        _write_atomic_bytes(self.identity_path, expected)

    def _scan_existing_rank_root(self):
        receipt_root = self.rank_root / "workloads"
        receipt_paths = (
            sorted(receipt_root.glob("*.complete.json"))
            if receipt_root.exists()
            else []
        )
        allowed_paths = {
            self.rank_root,
            self.tensor_root,
            self.identity_path,
        }
        if receipt_root.exists():
            if receipt_root.is_symlink() or not receipt_root.is_dir():
                raise ValueError("untracked path in rank root")
            allowed_paths.add(receipt_root)

        for receipt_path in receipt_paths:
            if receipt_path.is_symlink() or not receipt_path.is_file():
                raise ValueError("untracked path in rank root")
            receipt = self._load_receipt(receipt_path)
            workload_id = receipt["workload_id"]
            expected_receipt_path = (
                receipt_root / f"{workload_id}.complete.json"
            )
            if receipt_path != expected_receipt_path:
                raise ValueError("workload receipt path mismatch")
            if workload_id in self._completed_workloads:
                raise ValueError("duplicate completed workload")
            tensors = receipt["tensors"]
            expected_layers = self.run_identity.linear_layer_indices
            if tuple(
                tensor["layer_index"] for tensor in tensors
            ) != expected_layers:
                raise ValueError("completed workload layer mismatch")
            for tensor_payload in tensors:
                record = validate_tensor_record(
                    tensor_payload,
                    identity=self.run_identity,
                    expected_rank=self.rank,
                )
                if record.workload_id != workload_id:
                    raise ValueError("receipt workload mismatch")
                tensor_path = self.capture_root / record.relative_path
                expected_path = (
                    self.tensor_root
                    / workload_id
                    / f"layer{record.layer_index}.pt"
                )
                if tensor_path != expected_path:
                    raise ValueError("receipt tensor path mismatch")
                if (
                    tensor_path.parent.is_symlink()
                    or tensor_path.is_symlink()
                ):
                    raise ValueError("receipt tensor path must not be a symlink")
                if (
                    not tensor_path.is_file()
                    or sha256_file(tensor_path) != record.sha256
                ):
                    raise ValueError("receipt tensor evidence mismatch")
                if record.tensor_id in self.records:
                    raise ValueError("duplicate tensor inventory")
                self.records[record.tensor_id] = record
                allowed_paths.add(tensor_path)
                allowed_paths.add(tensor_path.parent)
            self._completed_workloads.add(workload_id)
            allowed_paths.add(receipt_path)

        for path in self.rank_root.rglob("*"):
            if path not in allowed_paths:
                if _is_generated_temporary_path(path):
                    raise ValueError("temporary file remains in rank root")
                if (
                    path.is_dir()
                    and path.parent == self.tensor_root
                    and path.name in self.run_identity.workload_ids
                    and any(
                        child.is_file() and child.suffix == ".pt"
                        for child in path.iterdir()
                    )
                ):
                    raise ValueError("partial workload in rank root")
                if path.is_file() and path.suffix == ".pt":
                    raise ValueError("partial workload in rank root")
                raise ValueError("untracked path in rank root")

    def _load_receipt(self, receipt_path):
        try:
            receipt = json.loads(receipt_path.read_text("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid workload receipt") from error
        if set(receipt) != {"rank", "workload_id", "tensors"}:
            raise ValueError("invalid workload receipt")
        if receipt["rank"] != self.rank:
            raise ValueError("workload receipt rank mismatch")
        workload_id = receipt["workload_id"]
        if workload_id not in self.run_identity.workload_ids:
            raise ValueError("workload receipt is not declared")
        if not isinstance(receipt["tensors"], list):
            raise ValueError("invalid workload receipt")
        return receipt

    def _tensor_id(self, workload_id, layer_index):
        if workload_id not in self.run_identity.workload_ids:
            raise ValueError("workload_id is not declared")
        _validate_workload_component(workload_id)
        if layer_index not in self.run_identity.linear_layer_indices:
            raise ValueError("layer_index is not declared")
        return (
            f"rank{self.rank}:{workload_id}:layer{layer_index}:"
            "linear_recurrent"
        )

    def capture_layer(self, *, workload_id, layer_index, tensor):
        tensor_id = self._tensor_id(workload_id, layer_index)
        if workload_id in self._completed_workloads:
            raise ValueError("completed workload cannot be reopened")
        if tensor_id in self.records:
            raise ValueError(f"duplicate tensor: {tensor_id}")

        cpu_tensor = (
            tensor.detach()
            .to(dtype=torch.float32)
            .contiguous()
            .to(device="cpu")
        )
        if (
            len(cpu_tensor.shape) != 3
            or any(dimension <= 0 for dimension in cpu_tensor.shape)
        ):
            raise ValueError("shape must contain three positive integers")

        final_path = (
            self.tensor_root
            / workload_id
            / f"layer{layer_index}.pt"
        )
        if final_path.exists() or final_path.is_symlink():
            raise ValueError(f"pre-existing final path: {final_path}")
        if final_path.parent.is_symlink():
            raise ValueError("workload tensor directory must not be a symlink")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            final_path.parent.is_symlink()
            or not final_path.parent.is_dir()
        ):
            raise ValueError("workload tensor directory is invalid")

        temporary_path = None
        try:
            temporary_path = _new_temporary_path(final_path)
            save_tensor(cpu_tensor, temporary_path)
            _fsync_regular_file(temporary_path)
            payload_sha256 = sha256_file(temporary_path)
            _publish_no_clobber(temporary_path, final_path)
            record = CapturedTensorRecord(
                tensor_id=tensor_id,
                rank=self.rank,
                workload_id=workload_id,
                layer_index=layer_index,
                relative_path=final_path.relative_to(
                    self.capture_root
                ).as_posix(),
                sha256=payload_sha256,
                shape=tuple(cpu_tensor.shape),
                dtype="float32",
                logical_bytes=(
                    cpu_tensor.numel() * cpu_tensor.element_size()
                ),
            )
            self.records[tensor_id] = record
            return record
        finally:
            if (
                temporary_path is not None
                and (
                    temporary_path.exists()
                    or temporary_path.is_symlink()
                )
            ):
                temporary_path.unlink()
            del cpu_tensor

    def finish_workload(self, workload_id):
        if workload_id not in self.run_identity.workload_ids:
            raise ValueError("workload_id is not declared")
        if workload_id in self._completed_workloads:
            raise ValueError("completed workload cannot be reopened")
        records = [
            self.records.get(
                self._tensor_id(workload_id, layer_index)
            )
            for layer_index in self.run_identity.linear_layer_indices
        ]
        if any(record is None for record in records):
            raise ValueError("missing layers for workload")
        for record in records:
            tensor_path = self.capture_root / record.relative_path
            expected_path = (
                self.tensor_root
                / workload_id
                / f"layer{record.layer_index}.pt"
            )
            if (
                tensor_path != expected_path
                or tensor_path.parent.is_symlink()
                or tensor_path.is_symlink()
                or not tensor_path.is_file()
                or sha256_file(tensor_path) != record.sha256
            ):
                raise ValueError("workload tensor evidence mismatch")

        receipt = {
            "rank": self.rank,
            "workload_id": workload_id,
            "tensors": [record.payload() for record in records],
        }
        receipt_path = (
            self.rank_root
            / "workloads"
            / f"{workload_id}.complete.json"
        )
        if receipt_path.parent.is_symlink():
            raise ValueError("workload receipt root must not be a symlink")
        _write_atomic_bytes(
            receipt_path,
            canonical_json_bytes(receipt) + b"\n",
        )
        self._completed_workloads.add(workload_id)
        return receipt


def capture_recurrent_state(
    *,
    run_identity,
    rank,
    workload_id,
    layer_index,
    tensor,
    staging_dir=None,
):
    session = Qwen35RecurrentCaptureSession(
        run_identity=run_identity,
        rank=rank,
        staging_dir=staging_dir,
    )
    return session.capture_layer(
        workload_id=workload_id,
        layer_index=layer_index,
        tensor=tensor,
    )
