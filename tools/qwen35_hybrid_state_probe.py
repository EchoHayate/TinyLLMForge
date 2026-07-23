"""State normalization helpers for the Qwen3.5 compatibility gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent


def _load_contract():
    module_name = "qwen35_hybrid_state_contract_for_probe"
    spec = importlib.util.spec_from_file_location(
        module_name,
        os.fspath(THIS_DIR / "qwen35_hybrid_state_contract.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract()

STATE_SNAPSHOT_FIELDS = (
    "snapshot_id",
    "request_id",
    "request_generation",
    "lifetime_epoch",
    "sequence_length",
    "component_count",
    "component_sha256",
)
MEMORY_SNAPSHOT_FIELDS = (
    "snapshot_id",
    "phase",
    "request_id",
    "request_generation",
    "cuda_allocated_bytes",
    "cuda_reserved_bytes",
    "logical_state_bytes",
    "unique_storage_bytes",
)
EXPORT_SCHEMA = "qwen35_hybrid_state_components"


def _join_path(prefix, component):
    if not prefix:
        return component
    if component.startswith("["):
        return f"{prefix}{component}"
    return f"{prefix}.{component}"


def walk_tensor_leaves(state, *, adapter_registry=None):
    """Yield ``(path, tensor)`` leaves using only frozen container rules."""
    registry = adapter_registry or {}

    def walk(value, path):
        if isinstance(value, torch.Tensor):
            yield path, value
            return
        adapter = registry.get(type(value))
        if adapter is not None:
            yield from walk(adapter(value), path)
            return
        if is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                yield from walk(
                    getattr(value, field.name),
                    _join_path(path, field.name),
                )
            return
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            for name in value._fields:
                yield from walk(
                    getattr(value, name),
                    _join_path(path, name),
                )
            return
        if isinstance(value, Mapping):
            for key in sorted(value, key=lambda item: str(item)):
                yield from walk(
                    value[key],
                    _join_path(path, str(key)),
                )
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, item in enumerate(value):
                yield from walk(item, _join_path(path, f"[{index}]"))
            return
        if value is None or isinstance(
            value, (str, bytes, bytearray, bool, int, float)
        ):
            return
        raise ValueError(
            f"unsupported state object {type(value).__name__}; "
            "register an explicit adapter"
        )

    yield from walk(state, "")


def _path_tokens(tensor_path):
    return (
        tensor_path.lower()
        .replace("[", ".")
        .replace("]", "")
        .replace("-", "_")
        .split(".")
    )


def classify_state_role(
    tensor_path,
    *,
    declared_layer_type,
    component_name=None,
):
    text = "_".join(
        token
        for token in _path_tokens(component_name or tensor_path)
        if token
    )
    layer_type = str(declared_layer_type).lower()
    if any(name in text for name in ("position", "sequence", "cache_position")):
        return "position_or_sequence_metadata"
    if "conv" in text:
        return "linear_convolution_state"
    if any(name in text for name in ("recurrent", "ssm", "state_cache")):
        return "linear_recurrent_state"
    if any(name in text for name in ("value_cache", "value_state", "values")):
        return "full_attention_value"
    if any(name in text for name in ("key_cache", "key_state", "keys")):
        return "full_attention_key"
    if "full" in layer_type or "attention" in layer_type:
        if text.endswith("_key") or text == "key":
            return "full_attention_key"
        if text.endswith("_value") or text == "value":
            return "full_attention_value"
    return "other_persistent_state"


def _dtype_name(dtype):
    name = str(dtype)
    return name.removeprefix("torch.")


def _tensor_content_sha256(tensor):
    value = tensor.detach().contiguous().cpu()
    try:
        content = value.numpy().tobytes(order="C")
    except (TypeError, RuntimeError):
        content = value.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(content).hexdigest()


def _layer_index(tensor_path, layer_schedule):
    tokens = _path_tokens(tensor_path)
    if "layers" in tokens:
        position = tokens.index("layers")
        if position + 1 < len(tokens):
            try:
                return int(tokens[position + 1])
            except ValueError:
                pass
    if len(layer_schedule) == 1:
        return next(iter(layer_schedule))
    return -1


def normalize_state_components(
    *,
    state,
    request_id,
    request_generation,
    sequence_length,
    lifetime_epoch,
    layer_schedule,
    adapter_registry=None,
    component_names=None,
):
    rows = []
    names = component_names or {}
    for tensor_path, tensor in walk_tensor_leaves(
        state,
        adapter_registry=adapter_registry,
    ):
        layer_index = _layer_index(tensor_path, layer_schedule)
        declared_layer_type = layer_schedule.get(layer_index, "unknown")
        storage = tensor.untyped_storage()
        device = str(tensor.device)
        storage_data_ptr = int(storage.data_ptr())
        storage_nbytes = int(storage.nbytes())
        storage_identity = contract.canonical_json_sha256({
            "device": device,
            "storage_data_ptr": storage_data_ptr,
            "storage_nbytes": storage_nbytes,
        })
        dtype = _dtype_name(tensor.dtype)
        component = contract.StateComponent(
            request_id=str(request_id),
            request_generation=int(request_generation),
            layer_index=int(layer_index),
            declared_layer_type=str(declared_layer_type),
            state_role=classify_state_role(
                tensor_path,
                declared_layer_type=declared_layer_type,
                component_name=names.get(tensor_path),
            ),
            tensor_path=tensor_path,
            shape=tuple(int(value) for value in tensor.shape),
            stride=tuple(int(value) for value in tensor.stride()),
            dtype=dtype,
            device=device,
            requires_grad=bool(tensor.requires_grad),
            logical_numel=int(tensor.numel()),
            logical_bytes=int(tensor.numel() * tensor.element_size()),
            storage_data_ptr=storage_data_ptr,
            storage_offset=int(tensor.storage_offset()),
            storage_nbytes=storage_nbytes,
            storage_identity=storage_identity,
            lifetime_epoch=int(lifetime_epoch),
            sequence_length=int(sequence_length),
            update_kind="created",
            content_sha256=_tensor_content_sha256(tensor),
        )
        rows.append(asdict(component))
    return sorted(rows, key=_component_sort_key)


def _component_key(component):
    return (
        component["request_id"],
        int(component["request_generation"]),
        int(component["layer_index"]),
        component["state_role"],
        component["tensor_path"],
    )


def _component_sort_key(component):
    return _component_key(component)


def _validate_generation_ownership(components):
    owners = {}
    for component in components:
        identity = (
            component["request_id"],
            int(component["layer_index"]),
            component["state_role"],
            component["tensor_path"],
        )
        generation = int(component["request_generation"])
        previous = owners.setdefault(identity, generation)
        if previous != generation:
            raise ValueError(
                "component maps to more than one request generation"
            )


def _transition_kind(previous, current):
    if current["storage_identity"] != previous["storage_identity"]:
        return "replaced"
    if (
        current["shape"] != previous["shape"]
        or current["storage_offset"] != previous["storage_offset"]
        or current["storage_nbytes"] != previous["storage_nbytes"]
        or current["logical_bytes"] != previous["logical_bytes"]
    ):
        return "grown"
    if current["content_sha256"] != previous["content_sha256"]:
        return "mutated_in_place"
    return "unchanged"


def compare_state_snapshots(previous, current):
    _validate_generation_ownership([*previous, *current])
    previous_by_key = {_component_key(row): row for row in previous}
    current_by_key = {_component_key(row): row for row in current}
    if len(previous_by_key) != len(previous) or len(current_by_key) != len(current):
        raise ValueError("duplicate component key")
    transitions = {}
    for key in sorted(set(previous_by_key) | set(current_by_key)):
        before = previous_by_key.get(key)
        after = current_by_key.get(key)
        role = key[3]
        if before is None:
            kind = "created"
        elif after is None:
            kind = "released"
        else:
            kind = _transition_kind(before, after)
        if role in transitions and transitions[role] != kind:
            raise ValueError(
                f"multiple transition kinds for state role {role}"
            )
        transitions[role] = kind
    return transitions


def export_normalized_state(components):
    ordered = sorted((dict(row) for row in components), key=_component_sort_key)
    _validate_component_rows(ordered)
    return {
        "schema": EXPORT_SCHEMA,
        "schema_version": contract.SCHEMA_VERSION,
        "components": ordered,
        "components_sha256": contract.canonical_json_sha256(ordered),
    }


def import_normalized_state(payload):
    if payload.get("schema") != EXPORT_SCHEMA:
        raise ValueError("unsupported schema")
    if payload.get("schema_version") != contract.SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    components = [dict(row) for row in payload.get("components", ())]
    _validate_component_rows(components)
    expected = payload.get("components_sha256")
    actual = contract.canonical_json_sha256(components)
    if expected != actual:
        raise ValueError("components_sha256 mismatch")
    return sorted(components, key=_component_sort_key)


def _validate_component_rows(components):
    expected_fields = set(contract.StateComponent.__dataclass_fields__)
    seen = set()
    for component in components:
        if set(component) != expected_fields:
            raise ValueError("state component fields do not match contract")
        key = _component_key(component)
        if key in seen:
            raise ValueError("duplicate component key")
        seen.add(key)
    _validate_generation_ownership(components)


def _write_atomic(path, content):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()


def write_json_atomic(path, payload):
    content = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    _write_atomic(path, f"{content}\n")


def write_jsonl_atomic(path, rows, *, required_fields):
    expected = set(required_fields)
    lines = []
    for row in rows:
        if set(row) != expected:
            raise ValueError("JSONL row fields do not match required_fields")
        lines.append(json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ))
    _write_atomic(path, "".join(f"{line}\n" for line in lines))
