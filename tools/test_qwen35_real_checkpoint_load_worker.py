from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "tools/qwen35_real_checkpoint_load_worker.py"


def _load_worker():
    spec = importlib.util.spec_from_file_location(
        "qwen35_real_checkpoint_load_worker_under_test",
        WORKER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expect_error(callback, message):
    try:
        callback()
    except (TypeError, ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_worker_builds_exact_manifest_bound_rank_loader_without_providers():
    worker = _load_worker()
    events = []
    tensor_plan = object()
    manifest = object()
    loader = object()
    hf_config = object()
    index_payload = object()
    shard_headers = object()
    create_pool = lambda: events.append("pool")
    build_backend = lambda *args: events.append(("backend", args))

    def build_tensor_plan(config, index, headers):
        events.append(("plan", config, index, headers))
        return tensor_plan

    class Manifest:

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            events.append(("manifest", kwargs))

    class Configuration:

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            events.append(("configuration", kwargs))

        def build_loader(self):
            events.append("build_loader")
            return loader

    worker.build_qwen35_checkpoint_tensor_plan = build_tensor_plan
    worker.Qwen35CheckpointManifestIdentity = Manifest
    worker.Qwen35RankCheckpointLoaderConfiguration = Configuration

    result = worker.build_qwen35_real_checkpoint_rank_loader(
        hf_config,
        index_payload,
        shard_headers,
        checkpoint_dir="/approved/model",
        model_manifest_sha256="a" * 64,
        config_sha256="b" * 64,
        index_sha256="c" * 64,
        config_index_header_sha256="d" * 64,
        tensor_parallel_size=2,
        tensor_parallel_rank=1,
        create_pool=create_pool,
        build_attention_backend=build_backend,
        authorization_sha256="e" * 64,
    )

    assert result is loader
    assert events[0] == (
        "plan",
        hf_config,
        index_payload,
        shard_headers,
    )
    assert events[1] == (
        "manifest",
        {
            "checkpoint_dir": "/approved/model",
            "model_manifest_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "index_sha256": "c" * 64,
            "config_index_header_sha256": "d" * 64,
        },
    )
    configuration_kwargs = events[2][1]
    assert configuration_kwargs["manifest"].kwargs == events[1][1]
    assert configuration_kwargs["hf_config"] is hf_config
    assert configuration_kwargs["tensor_plan"] is tensor_plan
    assert configuration_kwargs["tensor_parallel_size"] == 2
    assert configuration_kwargs["tensor_parallel_rank"] == 1
    assert configuration_kwargs["create_pool"] is create_pool
    assert configuration_kwargs["build_attention_backend"] is build_backend
    assert configuration_kwargs["authorization_sha256"] == "e" * 64
    assert events[-1] == "build_loader"
    assert "pool" not in events
    assert not any(
        isinstance(event, tuple) and event[0] == "backend"
        for event in events
    )


def test_worker_propagates_metadata_failure_without_configuration():
    worker = _load_worker()
    events = []

    def fail_plan(*_):
        raise ValueError("injected metadata failure")

    class Forbidden:

        def __init__(self, **kwargs):
            events.append(kwargs)

    worker.build_qwen35_checkpoint_tensor_plan = fail_plan
    worker.Qwen35CheckpointManifestIdentity = Forbidden
    worker.Qwen35RankCheckpointLoaderConfiguration = Forbidden

    _expect_error(
        lambda: worker.build_qwen35_real_checkpoint_rank_loader(
            object(),
            object(),
            object(),
            checkpoint_dir="/approved/model",
            model_manifest_sha256="a" * 64,
            config_sha256="b" * 64,
            index_sha256="c" * 64,
            config_index_header_sha256="d" * 64,
            tensor_parallel_size=1,
            tensor_parallel_rank=0,
            create_pool=lambda: object(),
            build_attention_backend=lambda *args: args,
            authorization_sha256="e" * 64,
        ),
        "injected metadata failure",
    )
    assert events == []


def test_worker_builds_rank_loader_from_exact_metadata_bundle():
    worker = _load_worker()
    events = []
    loader = object()
    hf_config = object()
    index_payload = object()
    shard_headers = object()
    create_pool = lambda: object()
    build_backend = lambda *args: args

    @dataclass(frozen=True)
    class MetadataBundle:
        hf_config: object
        index_payload: object
        shard_headers: object
        config_sha256: str
        index_sha256: str
        config_index_header_sha256: str
        metadata_bytes_read: int
        payload_bytes_read: int

    metadata = MetadataBundle(
        hf_config=hf_config,
        index_payload=index_payload,
        shard_headers=shard_headers,
        config_sha256="b" * 64,
        index_sha256="c" * 64,
        config_index_header_sha256="d" * 64,
        metadata_bytes_read=123,
        payload_bytes_read=0,
    )

    def build_rank_loader(*args, **kwargs):
        events.append((args, kwargs))
        return loader

    worker.Qwen35CheckpointMetadataBundle = MetadataBundle
    worker.build_qwen35_real_checkpoint_rank_loader = build_rank_loader

    result = worker.build_qwen35_real_checkpoint_rank_loader_from_metadata(
        metadata,
        checkpoint_dir="/approved/model",
        model_manifest_sha256="a" * 64,
        tensor_parallel_size=2,
        tensor_parallel_rank=1,
        create_pool=create_pool,
        build_attention_backend=build_backend,
        authorization_sha256="e" * 64,
    )

    assert result is loader
    assert events == [(
        (hf_config, index_payload, shard_headers),
        {
            "checkpoint_dir": "/approved/model",
            "model_manifest_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "index_sha256": "c" * 64,
            "config_index_header_sha256": "d" * 64,
            "tensor_parallel_size": 2,
            "tensor_parallel_rank": 1,
            "create_pool": create_pool,
            "build_attention_backend": build_backend,
            "authorization_sha256": "e" * 64,
        },
    )]


def test_worker_rejects_non_exact_metadata_bundle_before_construction():
    worker = _load_worker()
    events = []

    class MetadataBundle:
        pass

    class DerivedMetadataBundle(MetadataBundle):
        pass

    worker.Qwen35CheckpointMetadataBundle = MetadataBundle
    worker.build_qwen35_real_checkpoint_rank_loader = (
        lambda *args, **kwargs: events.append((args, kwargs))
    )

    _expect_error(
        lambda: worker.build_qwen35_real_checkpoint_rank_loader_from_metadata(
            DerivedMetadataBundle(),
            checkpoint_dir="/approved/model",
            model_manifest_sha256="a" * 64,
            tensor_parallel_size=1,
            tensor_parallel_rank=0,
            create_pool=lambda: object(),
            build_attention_backend=lambda *args: args,
            authorization_sha256="e" * 64,
        ),
        "exact Qwen35CheckpointMetadataBundle",
    )
    assert events == []


def test_worker_rejects_metadata_bundle_with_payload_bytes_read():
    worker = _load_worker()
    events = []

    @dataclass(frozen=True)
    class MetadataBundle:
        hf_config: object
        index_payload: object
        shard_headers: object
        config_sha256: str
        index_sha256: str
        config_index_header_sha256: str
        metadata_bytes_read: int
        payload_bytes_read: int

    worker.Qwen35CheckpointMetadataBundle = MetadataBundle
    worker.build_qwen35_real_checkpoint_rank_loader = (
        lambda *args, **kwargs: events.append((args, kwargs))
    )
    metadata = MetadataBundle(
        hf_config=object(),
        index_payload=object(),
        shard_headers=object(),
        config_sha256="b" * 64,
        index_sha256="c" * 64,
        config_index_header_sha256="d" * 64,
        metadata_bytes_read=123,
        payload_bytes_read=1,
    )

    _expect_error(
        lambda: worker.build_qwen35_real_checkpoint_rank_loader_from_metadata(
            metadata,
            checkpoint_dir="/approved/model",
            model_manifest_sha256="a" * 64,
            tensor_parallel_size=1,
            tensor_parallel_rank=0,
            create_pool=lambda: object(),
            build_attention_backend=lambda *args: args,
            authorization_sha256="e" * 64,
        ),
        "payload bytes",
    )
    assert events == []


def test_main_remains_hard_execution_rejection():
    worker = _load_worker()
    _expect_error(worker.main, "execution is not implemented")


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 real checkpoint load worker tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
