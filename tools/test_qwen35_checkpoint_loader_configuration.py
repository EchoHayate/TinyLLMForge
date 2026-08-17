from __future__ import annotations

from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for package_name in (
    "tinyvllm",
    "tinyvllm.engine",
    "tinyvllm.layers",
    "tinyvllm.models",
    "tinyvllm.utils",
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules[package_name] = package

import tinyvllm.models.qwen35_checkpoint_loader_configuration as config_module
from tinyvllm.engine.hybrid_state import HybridStateTensorPool
from tinyvllm.models.qwen35_checkpoint import Qwen35CheckpointTensorPlan
from tinyvllm.models.qwen35_checkpoint_loader_configuration import (
    Qwen35CheckpointManifestIdentity,
    Qwen35ManifestBoundCheckpointCandidateLoader,
    Qwen35RankCheckpointLoaderConfiguration,
)
from tinyvllm.models.qwen35_checkpoint_worker import (
    Qwen35CheckpointCandidateLoadRequest,
)


CHECKPOINT_DIR = "/approved/model"
MODEL_FINGERPRINT = "a" * 64
CONFIG_SHA256 = "b" * 64
INDEX_SHA256 = "c" * 64
COMPOSITE_SHA256 = "d" * 64
AUTHORIZATION_SHA256 = "e" * 64


class _Pool:
    pass


class _Target:
    pass


class _Candidate:
    pass


def _manifest(**overrides):
    values = {
        "checkpoint_dir": CHECKPOINT_DIR,
        "model_manifest_sha256": MODEL_FINGERPRINT,
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
        "config_index_header_sha256": COMPOSITE_SHA256,
    }
    values.update(overrides)
    return Qwen35CheckpointManifestIdentity(**values)


def _plan():
    return Qwen35CheckpointTensorPlan(
        loads=(),
        skips=(),
        payload_bytes=0,
    )


def _request(**overrides):
    values = {
        "checkpoint_dir": CHECKPOINT_DIR,
        "model_fingerprint": MODEL_FINGERPRINT,
        "max_tensor_bytes": 8 << 20,
        "authorization_sha256": AUTHORIZATION_SHA256,
    }
    values.update(overrides)
    return Qwen35CheckpointCandidateLoadRequest(**values)


def _configuration(**overrides):
    values = {
        "manifest": _manifest(),
        "hf_config": object(),
        "tensor_plan": _plan(),
        "tensor_parallel_size": 2,
        "tensor_parallel_rank": 1,
        "create_pool": lambda: _Pool(),
        "build_attention_backend": lambda *args: args,
        "authorization_sha256": AUTHORIZATION_SHA256,
    }
    values.update(overrides)
    return Qwen35RankCheckpointLoaderConfiguration(**values)


def _expect_error(callback, message):
    try:
        callback()
    except (TypeError, ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_manifest_and_configuration_validation_is_allocation_free():
    for overrides, message in (
        ({"checkpoint_dir": "relative"}, "absolute"),
        ({"checkpoint_dir": "/approved/../model"}, "normalized"),
        ({"model_manifest_sha256": "bad"}, "model_manifest_sha256"),
        ({"config_sha256": "bad"}, "config_sha256"),
        ({"index_sha256": "bad"}, "index_sha256"),
        (
            {"config_index_header_sha256": "bad"},
            "config_index_header_sha256",
        ),
    ):
        _expect_error(lambda overrides=overrides: _manifest(**overrides), message)

    pool_calls = []
    configuration = _configuration(
        create_pool=lambda: pool_calls.append("pool"),
    )
    loader = configuration.build_loader()

    assert type(configuration) is Qwen35RankCheckpointLoaderConfiguration
    assert type(loader) is Qwen35ManifestBoundCheckpointCandidateLoader
    assert loader.configuration is configuration
    assert pool_calls == []

    for overrides, message in (
        ({"manifest": object()}, "manifest"),
        ({"tensor_plan": object()}, "tensor_plan"),
        ({"tensor_parallel_size": 0}, "tensor_parallel_size"),
        ({"tensor_parallel_rank": 2}, "tensor_parallel_rank"),
        ({"create_pool": object()}, "create_pool"),
        ({"build_attention_backend": object()}, "build_attention_backend"),
        ({"authorization_sha256": "bad"}, "authorization_sha256"),
    ):
        _expect_error(
            lambda overrides=overrides: _configuration(**overrides),
            message,
        )


def test_request_conflicts_fail_before_pool_creation():
    pool_calls = []
    loader = _configuration(
        create_pool=lambda: pool_calls.append("pool"),
    ).build_loader()

    for request, message in (
        (object(), "exact Qwen35CheckpointCandidateLoadRequest"),
        (_request(checkpoint_dir="/other/model"), "checkpoint_dir"),
        (_request(model_fingerprint="f" * 64), "model_fingerprint"),
        (_request(authorization_sha256="f" * 64), "authorization"),
    ):
        _expect_error(lambda request=request: loader(request), message)
    assert pool_calls == []


def test_each_invocation_forwards_exact_rank_configuration_and_fresh_pool():
    pools = []
    prepared = []
    candidates = []
    original_pool_type = config_module.HybridStateTensorPool
    original_prepare = (
        config_module.prepare_qwen35_checkpoint_candidate_target
    )
    original_build_adapter = (
        config_module.build_qwen35_authorized_checkpoint_candidate_loader
    )
    config_module.HybridStateTensorPool = _Pool

    def create_pool():
        pool = _Pool()
        pools.append(pool)
        return pool

    def prepare_target(hf_config, tensor_plan, **kwargs):
        target = _Target()
        prepared.append((hf_config, tensor_plan, kwargs, target))
        return target

    def build_adapter(prepare_target, *, authorization_sha256):
        def adapter(request):
            target = prepare_target()
            candidate = _Candidate()
            candidates.append((
                request,
                target,
                authorization_sha256,
                candidate,
            ))
            return candidate

        return adapter

    config_module.prepare_qwen35_checkpoint_candidate_target = prepare_target
    config_module.build_qwen35_authorized_checkpoint_candidate_loader = (
        build_adapter
    )
    hf_config = object()
    tensor_plan = _plan()
    backend = lambda *args: args
    try:
        loader = _configuration(
            hf_config=hf_config,
            tensor_plan=tensor_plan,
            create_pool=create_pool,
            build_attention_backend=backend,
        ).build_loader()
        first = loader(_request())
        second = loader(_request())
    finally:
        config_module.HybridStateTensorPool = original_pool_type
        config_module.prepare_qwen35_checkpoint_candidate_target = (
            original_prepare
        )
        config_module.build_qwen35_authorized_checkpoint_candidate_loader = (
            original_build_adapter
        )

    assert type(first) is _Candidate
    assert type(second) is _Candidate
    assert first is not second
    assert len(pools) == 2
    assert pools[0] is not pools[1]
    assert len(prepared) == 2
    for index, (actual_config, actual_plan, kwargs, target) in enumerate(
        prepared
    ):
        assert actual_config is hf_config
        assert actual_plan is tensor_plan
        assert kwargs == {
            "pool": pools[index],
            "tensor_parallel_size": 2,
            "tensor_parallel_rank": 1,
            "build_attention_backend": backend,
            "parameter_device": "cpu",
        }
        assert candidates[index][1] is target
        assert candidates[index][2] == AUTHORIZATION_SHA256


def test_invalid_pool_and_retry_after_prepare_failure_are_fresh():
    original_pool_type = config_module.HybridStateTensorPool
    original_prepare = (
        config_module.prepare_qwen35_checkpoint_candidate_target
    )
    config_module.HybridStateTensorPool = _Pool
    pool_calls = []
    prepare_calls = []

    def invalid_pool():
        pool_calls.append(object())
        return object()

    loader = _configuration(create_pool=invalid_pool).build_loader()
    _expect_error(lambda: loader(_request()), "HybridStateTensorPool")
    assert len(pool_calls) == 1

    def create_pool():
        pool = _Pool()
        pool_calls.append(pool)
        return pool

    def fail_prepare(*args, **kwargs):
        prepare_calls.append(kwargs["pool"])
        raise RuntimeError("injected prepare failure")

    config_module.prepare_qwen35_checkpoint_candidate_target = fail_prepare
    try:
        loader = _configuration(create_pool=create_pool).build_loader()
        _expect_error(lambda: loader(_request()), "injected prepare failure")
        _expect_error(lambda: loader(_request()), "injected prepare failure")
    finally:
        config_module.HybridStateTensorPool = original_pool_type
        config_module.prepare_qwen35_checkpoint_candidate_target = (
            original_prepare
        )

    assert len(prepare_calls) == 2
    assert prepare_calls[0] is not prepare_calls[1]


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 checkpoint loader configuration tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
