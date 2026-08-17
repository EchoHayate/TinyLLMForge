from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import builtins
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import MappingProxyType, SimpleNamespace


open = builtins.open


def _sha256(value, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _positive_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _safe_shard_name(value) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("shard name must be a non-empty string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or value != str(path)
        or not value.endswith(".safetensors")
    ):
        raise ValueError(
            "shard name must be a safe relative .safetensors path"
        )
    return value


def _checkpoint_directory(value) -> Path:
    try:
        path = Path(value)
    except TypeError as error:
        raise ValueError(
            "checkpoint_dir must be an existing directory"
        ) from error
    if not path.is_dir():
        raise ValueError(
            "checkpoint_dir must be an existing directory"
        )
    return path.resolve()


def _canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _recursive_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{
            key: _recursive_namespace(item)
            for key, item in value.items()
        })
    if isinstance(value, list):
        return tuple(_recursive_namespace(item) for item in value)
    return value


def _read_bounded_json_object(
    path: Path,
    *,
    max_bytes: int,
    expected_sha256: str,
    label: str,
) -> tuple[dict, int]:
    if not path.is_file():
        raise ValueError(f"missing {label} file")
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError(f"{label} bytes exceed configured budget")
    with open(path, "rb") as handle:
        payload = handle.read(size)
    if len(payload) != size:
        raise ValueError(f"short {label} read")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{label} SHA256 does not match")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON must be an object")
    return value, size


@dataclass(frozen=True)
class Qwen35CheckpointShardIdentity:
    name: str
    size: int
    sha256: str

    def __post_init__(self):
        _safe_shard_name(self.name)
        _positive_integer(self.size, "shard size")
        _sha256(self.sha256, "shard sha256")


@dataclass(frozen=True)
class Qwen35CheckpointMetadataBundle:
    hf_config: object
    index_payload: Mapping[str, object]
    shard_headers: Mapping[str, Mapping[str, object]]
    config_sha256: str
    index_sha256: str
    config_index_header_sha256: str
    metadata_bytes_read: int
    payload_bytes_read: int


def read_qwen35_checkpoint_metadata(
    checkpoint_dir,
    *,
    shards,
    expected_config_sha256,
    expected_index_sha256,
    expected_config_index_header_sha256,
    max_config_bytes=1 << 20,
    max_index_bytes=16 << 20,
    max_header_bytes=64 << 20,
) -> Qwen35CheckpointMetadataBundle:
    directory = _checkpoint_directory(checkpoint_dir)
    expected_config_sha256 = _sha256(
        expected_config_sha256,
        "expected_config_sha256",
    )
    expected_index_sha256 = _sha256(
        expected_index_sha256,
        "expected_index_sha256",
    )
    expected_composite = _sha256(
        expected_config_index_header_sha256,
        "expected_config_index_header_sha256",
    )
    max_config_bytes = _positive_integer(
        max_config_bytes,
        "max_config_bytes",
    )
    max_index_bytes = _positive_integer(
        max_index_bytes,
        "max_index_bytes",
    )
    max_header_bytes = _positive_integer(
        max_header_bytes,
        "max_header_bytes",
    )
    if not isinstance(shards, (tuple, list)) or not shards:
        raise ValueError("shards must be a non-empty tuple or list")

    identities = []
    names = set()
    shard_paths = {}
    for shard in shards:
        if type(shard) is not Qwen35CheckpointShardIdentity:
            raise ValueError(
                "shards must contain exact "
                "Qwen35CheckpointShardIdentity values"
            )
        if shard.name in names:
            raise ValueError(f"duplicate shard identity: {shard.name}")
        names.add(shard.name)
        identities.append(shard)
        relative = PurePosixPath(shard.name)
        path = (directory / Path(*relative.parts)).resolve()
        if directory not in path.parents or not path.is_file():
            raise ValueError(f"missing checkpoint shard: {shard.name}")
        if path.stat().st_size != shard.size:
            raise ValueError(f"checkpoint shard size mismatch: {shard.name}")
        shard_paths[shard.name] = path

    identity_payload = {
        "config_sha256": expected_config_sha256,
        "index_sha256": expected_index_sha256,
        "shards": {
            shard.name: {
                "sha256": shard.sha256,
                "size": shard.size,
            }
            for shard in identities
        },
    }
    actual_composite = hashlib.sha256(
        _canonical_json_bytes(identity_payload)
    ).hexdigest()
    if actual_composite != expected_composite:
        raise ValueError(
            "config/index/shard composite identity does not match"
        )

    config_payload, config_bytes = _read_bounded_json_object(
        directory / "config.json",
        max_bytes=max_config_bytes,
        expected_sha256=expected_config_sha256,
        label="config",
    )
    index_payload, index_bytes = _read_bounded_json_object(
        directory / "model.safetensors.index.json",
        max_bytes=max_index_bytes,
        expected_sha256=expected_index_sha256,
        label="index",
    )
    weight_map = index_payload.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("index weight_map must be a non-empty mapping")
    declared_shards = set(weight_map.values())
    if declared_shards != names:
        raise ValueError(
            "index shard set must match retained shard identities"
        )

    headers = {}
    metadata_bytes = config_bytes + index_bytes
    for shard in identities:
        path = shard_paths[shard.name]
        with open(path, "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(
                    f"short safetensors header prefix: {shard.name}"
                )
            header_length = int.from_bytes(prefix, "little")
            if header_length <= 0:
                raise ValueError(
                    f"safetensors header length is invalid: {shard.name}"
                )
            if header_length > max_header_bytes:
                raise ValueError(
                    "safetensors header exceeds max_header_bytes: "
                    f"{shard.name}"
                )
            if 8 + header_length > shard.size:
                raise ValueError(
                    f"safetensors header exceeds shard size: {shard.name}"
                )
            header_bytes = handle.read(header_length)
            if len(header_bytes) != header_length:
                raise ValueError(
                    f"short safetensors header read: {shard.name}"
                )
        try:
            header = json.loads(header_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid safetensors header JSON: {shard.name}"
            ) from error
        if not isinstance(header, dict):
            raise ValueError(
                f"safetensors header JSON must be an object: {shard.name}"
            )
        headers[shard.name] = MappingProxyType(header)
        metadata_bytes += 8 + header_length

    return Qwen35CheckpointMetadataBundle(
        hf_config=_recursive_namespace(config_payload),
        index_payload=MappingProxyType(index_payload),
        shard_headers=MappingProxyType(headers),
        config_sha256=expected_config_sha256,
        index_sha256=expected_index_sha256,
        config_index_header_sha256=actual_composite,
        metadata_bytes_read=metadata_bytes,
        payload_bytes_read=0,
    )
