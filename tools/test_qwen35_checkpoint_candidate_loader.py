from __future__ import annotations

from pathlib import Path
import sys
import types

import torch

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

import tinyvllm.models.qwen35_checkpoint_candidate_loader as loader_module
from tinyvllm.models.qwen35_checkpoint_candidate_factory import (
    Qwen35PreparedCheckpointCandidateTarget,
)
from tinyvllm.models.qwen35_checkpoint_candidate_loader import (
    Qwen35AuthorizedCheckpointCandidateLoader,
    build_qwen35_authorized_checkpoint_candidate_loader,
)
from tinyvllm.models.qwen35_checkpoint_worker import (
    Qwen35CheckpointCandidateLoadRequest,
)


MODEL_FINGERPRINT = "a" * 64
AUTHORIZATION_SHA256 = "b" * 64


class _Assembly:

    def __init__(self, device):
        self.parameter_device = torch.device(device)


class _PreparedTarget:

    def __init__(self, device="cpu"):
        self.assembly = _Assembly(device)
        self.take_calls = 0
        self.model = object()
        self.binding_plan = object()

    def take(self):
        self.take_calls += 1
        if self.take_calls != 1:
            raise RuntimeError("already consumed")
        return self.model, self.binding_plan


class _LoadedCandidate:
    pass


def _request(**overrides):
    values = {
        "checkpoint_dir": "/approved/model",
        "model_fingerprint": MODEL_FINGERPRINT,
        "max_tensor_bytes": 8 << 20,
        "authorization_sha256": AUTHORIZATION_SHA256,
    }
    values.update(overrides)
    return Qwen35CheckpointCandidateLoadRequest(**values)


def _expect_error(callback, message):
    try:
        callback()
    except (TypeError, ValueError, RuntimeError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def _install_fakes(targets, events, *, result=None, error=None):
    original_target_type = (
        loader_module.Qwen35PreparedCheckpointCandidateTarget
    )
    original_streamed_loader = (
        loader_module.load_qwen35_fresh_checkpoint_candidate
    )
    loader_module.Qwen35PreparedCheckpointCandidateTarget = _PreparedTarget

    def prepare_target():
        events.append(("prepare", len(targets)))
        target = _PreparedTarget()
        targets.append(target)
        return target

    def streamed_loader(
        candidate_factory,
        checkpoint_dir,
        *,
        max_tensor_bytes,
        model_fingerprint,
    ):
        events.append((
            "stream",
            checkpoint_dir,
            max_tensor_bytes,
            model_fingerprint,
        ))
        candidate = candidate_factory()
        events.append(("candidate", candidate))
        if error is not None:
            raise error
        return result

    loader_module.load_qwen35_fresh_checkpoint_candidate = streamed_loader
    return (
        prepare_target,
        original_target_type,
        original_streamed_loader,
    )


def _restore_fakes(original_target_type, original_streamed_loader):
    loader_module.Qwen35PreparedCheckpointCandidateTarget = (
        original_target_type
    )
    loader_module.load_qwen35_fresh_checkpoint_candidate = (
        original_streamed_loader
    )


def test_builder_validates_provider_and_authorization():
    provider = lambda: None
    loader = build_qwen35_authorized_checkpoint_candidate_loader(
        provider,
        authorization_sha256=AUTHORIZATION_SHA256,
    )

    assert type(loader) is Qwen35AuthorizedCheckpointCandidateLoader
    assert loader.prepare_target is provider
    assert loader.authorization_sha256 == AUTHORIZATION_SHA256
    _expect_error(
        lambda: build_qwen35_authorized_checkpoint_candidate_loader(
            object(),
            authorization_sha256=AUTHORIZATION_SHA256,
        ),
        "callable",
    )
    _expect_error(
        lambda: build_qwen35_authorized_checkpoint_candidate_loader(
            provider,
            authorization_sha256="bad",
        ),
        "authorization_sha256",
    )


def test_request_and_authorization_fail_before_provider_or_streamed_loader():
    events = []
    targets = []
    provider, original_type, original_loader = _install_fakes(
        targets,
        events,
    )
    try:
        loader = build_qwen35_authorized_checkpoint_candidate_loader(
            provider,
            authorization_sha256=AUTHORIZATION_SHA256,
        )
        _expect_error(
            lambda: loader(object()),
            "exact Qwen35CheckpointCandidateLoadRequest",
        )
        _expect_error(
            lambda: loader(_request(authorization_sha256="c" * 64)),
            "authorization",
        )
    finally:
        _restore_fakes(original_type, original_loader)
    assert events == []
    assert targets == []


def test_exact_cpu_target_is_taken_once_and_request_is_forwarded():
    events = []
    targets = []
    loaded = _LoadedCandidate()
    provider, original_type, original_loader = _install_fakes(
        targets,
        events,
        result=loaded,
    )
    try:
        loader = build_qwen35_authorized_checkpoint_candidate_loader(
            provider,
            authorization_sha256=AUTHORIZATION_SHA256,
        )
        result = loader(_request())
    finally:
        _restore_fakes(original_type, original_loader)

    assert result is loaded
    assert len(targets) == 1
    assert targets[0].take_calls == 1
    assert events[:2] == [
        ("prepare", 0),
        (
            "stream",
            "/approved/model",
            8 << 20,
            MODEL_FINGERPRINT,
        ),
    ]
    assert events[2] == (
        "candidate",
        (targets[0].model, targets[0].binding_plan),
    )


def test_invalid_or_meta_target_fails_before_streamed_loader_delegation():
    for output, message in (
        (object(), "exact Qwen35PreparedCheckpointCandidateTarget"),
        (_PreparedTarget("meta"), "CPU"),
    ):
        events = []
        original_type = (
            loader_module.Qwen35PreparedCheckpointCandidateTarget
        )
        original_loader = (
            loader_module.load_qwen35_fresh_checkpoint_candidate
        )
        loader_module.Qwen35PreparedCheckpointCandidateTarget = (
            _PreparedTarget
        )
        loader_module.load_qwen35_fresh_checkpoint_candidate = (
            lambda *args, **kwargs: events.append((args, kwargs))
        )
        try:
            loader = build_qwen35_authorized_checkpoint_candidate_loader(
                lambda output=output: output,
                authorization_sha256=AUTHORIZATION_SHA256,
            )
            _expect_error(lambda: loader(_request()), message)
        finally:
            _restore_fakes(original_type, original_loader)
        assert events == []


def test_delegated_failure_retries_with_a_fresh_target():
    events = []
    targets = []
    provider, original_type, original_loader = _install_fakes(
        targets,
        events,
        error=RuntimeError("injected streamed failure"),
    )
    try:
        loader = build_qwen35_authorized_checkpoint_candidate_loader(
            provider,
            authorization_sha256=AUTHORIZATION_SHA256,
        )
        _expect_error(
            lambda: loader(_request()),
            "injected streamed failure",
        )
        _expect_error(
            lambda: loader(_request()),
            "injected streamed failure",
        )
    finally:
        _restore_fakes(original_type, original_loader)

    assert len(targets) == 2
    assert targets[0] is not targets[1]
    assert [target.take_calls for target in targets] == [1, 1]


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 checkpoint candidate loader tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
