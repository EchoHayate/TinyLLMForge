from __future__ import annotations

import builtins
import hashlib
import json
import operator
from pathlib import Path
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for package_name in ("tinyvllm", "tinyvllm.models"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules[package_name] = package

import tinyvllm.models.qwen35_checkpoint_metadata as metadata_module
from tinyvllm.models.qwen35_checkpoint_metadata import (
    Qwen35CheckpointMetadataBundle,
    Qwen35CheckpointShardIdentity,
    read_qwen35_checkpoint_metadata,
)


def _canonical_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _identity(config_sha256, index_sha256, shards):
    payload = {
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "shards": {
            shard.name: {
                "sha256": shard.sha256,
                "size": shard.size,
            }
            for shard in shards
        },
    }
    return _sha256(_canonical_bytes(payload))


def _write_fixture(directory, *, shard_count=2):
    config_payload = {
        "dtype": "bfloat16",
        "text_config": {
            "hidden_size": 8,
            "layer_types": ["linear_attention", "full_attention"],
        },
    }
    shard_headers = {}
    shard_payloads = {}
    weight_map = {}
    for index in range(shard_count):
        shard_name = f"model-{index + 1:05d}-of-{shard_count:05d}.safetensors"
        source_name = f"tensor.{index}"
        header = {
            source_name: {
                "dtype": "BF16",
                "shape": [2],
                "data_offsets": [0, 4],
            },
            "__metadata__": {"format": "pt"},
        }
        header_bytes = _canonical_bytes(header)
        payload = bytes([65 + index]) * 4
        shard_bytes = (
            len(header_bytes).to_bytes(8, "little")
            + header_bytes
            + payload
        )
        (directory / shard_name).write_bytes(shard_bytes)
        shard_headers[shard_name] = header
        shard_payloads[shard_name] = payload
        weight_map[source_name] = shard_name
    index_payload = {
        "metadata": {"total_size": 4 * shard_count},
        "weight_map": weight_map,
    }
    config_bytes = _canonical_bytes(config_payload)
    index_bytes = _canonical_bytes(index_payload)
    (directory / "config.json").write_bytes(config_bytes)
    (directory / "model.safetensors.index.json").write_bytes(index_bytes)
    shards = tuple(
        Qwen35CheckpointShardIdentity(
            name=shard_name,
            size=(directory / shard_name).stat().st_size,
            sha256=chr(97 + index) * 64,
        )
        for index, shard_name in enumerate(sorted(shard_headers))
    )
    return {
        "config_payload": config_payload,
        "index_payload": index_payload,
        "shard_headers": shard_headers,
        "shard_payloads": shard_payloads,
        "config_bytes": config_bytes,
        "index_bytes": index_bytes,
        "config_sha256": _sha256(config_bytes),
        "index_sha256": _sha256(index_bytes),
        "shards": shards,
        "composite": _identity(
            _sha256(config_bytes),
            _sha256(index_bytes),
            shards,
        ),
    }


class _TrackedFile:

    def __init__(self, handle, path, observations):
        self._handle = handle
        self._path = path
        self._observations = observations

    def read(self, size=-1):
        before = self._handle.tell()
        value = self._handle.read(size)
        after = self._handle.tell()
        self._observations.setdefault(self._path, []).append(
            (before, size, after, len(value))
        )
        return value

    def __enter__(self):
        self._handle.__enter__()
        return self

    def __exit__(self, *args):
        return self._handle.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self._handle, name)


class _ShortHeaderFile(_TrackedFile):

    def __init__(self, handle, path, observations):
        super().__init__(handle, path, observations)
        self._read_count = 0

    def read(self, size=-1):
        self._read_count += 1
        if self._read_count == 2 and size > 0:
            size -= 1
        return super().read(size)


def _tracking_open(observations):
    real_open = builtins.open

    def tracked(file, *args, **kwargs):
        handle = real_open(file, *args, **kwargs)
        path = str(file)
        if path.endswith(".safetensors"):
            return _TrackedFile(handle, path, observations)
        return handle

    return tracked


def _short_header_open(observations):
    real_open = builtins.open

    def tracked(file, *args, **kwargs):
        handle = real_open(file, *args, **kwargs)
        path = str(file)
        if path.endswith(".safetensors"):
            return _ShortHeaderFile(handle, path, observations)
        return handle

    return tracked


def _read(directory, fixture, **overrides):
    values = {
        "checkpoint_dir": directory,
        "shards": fixture["shards"],
        "expected_config_sha256": fixture["config_sha256"],
        "expected_index_sha256": fixture["index_sha256"],
        "expected_config_index_header_sha256": fixture["composite"],
    }
    values.update(overrides)
    return read_qwen35_checkpoint_metadata(**values)


def _expect_error(callback, message):
    try:
        callback()
    except (TypeError, ValueError, OSError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_reads_exact_metadata_and_stops_before_payload():
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        fixture = _write_fixture(directory)
        observations = {}
        original_open = metadata_module.open
        metadata_module.open = _tracking_open(observations)
        try:
            result = _read(directory, fixture)
        finally:
            metadata_module.open = original_open

        assert type(result) is Qwen35CheckpointMetadataBundle
        assert result.config_sha256 == fixture["config_sha256"]
        assert result.index_sha256 == fixture["index_sha256"]
        assert result.config_index_header_sha256 == fixture["composite"]
        assert result.payload_bytes_read == 0
        expected_metadata_bytes = (
            len(fixture["config_bytes"])
            + len(fixture["index_bytes"])
            + sum(
                8 + len(_canonical_bytes(header))
                for header in fixture["shard_headers"].values()
            )
        )
        assert result.metadata_bytes_read == expected_metadata_bytes
        assert result.hf_config.text_config.hidden_size == 8
        assert result.hf_config.text_config.layer_types == (
            "linear_attention",
            "full_attention",
        )
        assert dict(result.index_payload) == fixture["index_payload"]
        assert {
            name: dict(header)
            for name, header in result.shard_headers.items()
        } == fixture["shard_headers"]

        for shard in fixture["shards"]:
            path = str((directory / shard.name).resolve())
            header_length = len(
                _canonical_bytes(fixture["shard_headers"][shard.name])
            )
            assert observations[path] == [
                (0, 8, 8, 8),
                (8, header_length, 8 + header_length, header_length),
            ]
            assert max(row[2] for row in observations[path]) < shard.size

        _expect_error(
            lambda: operator.setitem(result.index_payload, "x", 1),
            "item assignment",
        )
        _expect_error(
            lambda: operator.setitem(result.shard_headers, "x", {}),
            "item assignment",
        )


def test_identity_path_budget_and_shard_contracts_fail_closed():
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        fixture = _write_fixture(directory)
        mismatched_config = "f" * 64
        mismatched_index = "f" * 64
        mismatched_size_shards = (
            Qwen35CheckpointShardIdentity(
                name=fixture["shards"][0].name,
                size=fixture["shards"][0].size + 1,
                sha256=fixture["shards"][0].sha256,
            ),
            fixture["shards"][1],
        )
        cases = (
            ({"checkpoint_dir": directory / "missing"}, "directory"),
            ({
                "expected_config_sha256": mismatched_config,
                "expected_config_index_header_sha256": _identity(
                    mismatched_config,
                    fixture["index_sha256"],
                    fixture["shards"],
                ),
            }, "config SHA256"),
            ({
                "expected_index_sha256": mismatched_index,
                "expected_config_index_header_sha256": _identity(
                    fixture["config_sha256"],
                    mismatched_index,
                    fixture["shards"],
                ),
            }, "index SHA256"),
            (
                {"expected_config_index_header_sha256": "f" * 64},
                "composite",
            ),
            ({"max_config_bytes": len(fixture["config_bytes"]) - 1}, "config"),
            ({"max_index_bytes": len(fixture["index_bytes"]) - 1}, "index"),
            ({"max_header_bytes": 1}, "header"),
            ({"shards": (*fixture["shards"], fixture["shards"][0])}, "duplicate"),
            (
                {
                    "shards": mismatched_size_shards,
                    "expected_config_index_header_sha256": _identity(
                        fixture["config_sha256"],
                        fixture["index_sha256"],
                        mismatched_size_shards,
                    )
                },
                "size",
            ),
        )
        for overrides, message in cases:
            _expect_error(
                lambda overrides=overrides: _read(
                    directory,
                    fixture,
                    **overrides,
                ),
                message,
            )
        _expect_error(
            lambda: Qwen35CheckpointShardIdentity(
                name="../escape.safetensors",
                size=12,
                sha256="a" * 64,
            ),
            "safe relative",
        )


def test_short_invalid_and_oversized_headers_never_read_payload():
    mutations = (
        (b"\x01\x02", 64, "prefix"),
        ((100).to_bytes(8, "little") + b"{}", 128, "shard size"),
        ((1000).to_bytes(8, "little") + b"{}", 16, "max_header_bytes"),
        ((1).to_bytes(8, "little") + b"{" + b"PAYLOAD-SENTINEL", 64, "JSON"),
        ((2).to_bytes(8, "little") + b"[]" + b"PAYLOAD-SENTINEL", 64, "object"),
    )
    for shard_bytes, max_header_bytes, message in mutations:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fixture = _write_fixture(directory, shard_count=1)
            shard = fixture["shards"][0]
            path = directory / shard.name
            path.write_bytes(shard_bytes)
            mutated = Qwen35CheckpointShardIdentity(
                name=shard.name,
                size=path.stat().st_size,
                sha256=shard.sha256,
            )
            composite = _identity(
                fixture["config_sha256"],
                fixture["index_sha256"],
                (mutated,),
            )
            observations = {}
            original_open = metadata_module.open
            metadata_module.open = _tracking_open(observations)
            try:
                _expect_error(
                    lambda: read_qwen35_checkpoint_metadata(
                        directory,
                        shards=(mutated,),
                        expected_config_sha256=fixture["config_sha256"],
                        expected_index_sha256=fixture["index_sha256"],
                        expected_config_index_header_sha256=composite,
                        max_header_bytes=max_header_bytes,
                    ),
                    message,
                )
            finally:
                metadata_module.open = original_open
            assert all(
                after <= 8 + max_header_bytes
                for rows in observations.values()
                for _, _, after, _ in rows
            )

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        fixture = _write_fixture(directory, shard_count=1)
        observations = {}
        original_open = metadata_module.open
        metadata_module.open = _short_header_open(observations)
        try:
            _expect_error(
                lambda: _read(directory, fixture),
                "short safetensors header read",
            )
        finally:
            metadata_module.open = original_open
        assert all(
            after < fixture["shards"][0].size
            for rows in observations.values()
            for _, _, after, _ in rows
        )


def test_config_and_index_json_must_be_objects():
    for name, message in (
        ("config.json", "config JSON"),
        ("model.safetensors.index.json", "index JSON"),
    ):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fixture = _write_fixture(directory)
            bad = _canonical_bytes([])
            (directory / name).write_bytes(bad)
            overrides = {
                (
                    "expected_config_sha256"
                    if name == "config.json"
                    else "expected_index_sha256"
                ): _sha256(bad)
            }
            shards = fixture["shards"]
            overrides["expected_config_index_header_sha256"] = _identity(
                overrides.get(
                    "expected_config_sha256",
                    fixture["config_sha256"],
                ),
                overrides.get(
                    "expected_index_sha256",
                    fixture["index_sha256"],
                ),
                shards,
            )
            _expect_error(
                lambda: _read(
                    directory,
                    fixture,
                    **overrides,
                ),
                message,
            )


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 checkpoint metadata tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
