from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tinyvllm/models/qwen35_checkpoint_worker.py"
MODEL_FINGERPRINT = "a" * 64
AUTHORIZATION_SHA256 = "b" * 64
sys.path.insert(0, str(ROOT))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "qwen35_checkpoint_worker_request_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _request(module, **overrides):
    arguments = {
        "checkpoint_dir": "/approved/model",
        "model_fingerprint": MODEL_FINGERPRINT,
        "max_tensor_bytes": 8 << 20,
        "authorization_sha256": AUTHORIZATION_SHA256,
    }
    arguments.update(overrides)
    return module.Qwen35CheckpointCandidateLoadRequest(**arguments)


def _expect_error(callback, message):
    try:
        callback()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_request_accepts_exact_bounded_metadata():
    module = _load_module()

    request = _request(module)

    assert request.checkpoint_dir == "/approved/model"
    assert request.model_fingerprint == MODEL_FINGERPRINT
    assert request.max_tensor_bytes == 8 << 20
    assert request.authorization_sha256 == AUTHORIZATION_SHA256
    assert module.validate_qwen35_checkpoint_candidate_load_request(
        request
    ) is request


def test_request_rejects_unsafe_or_unbounded_path():
    module = _load_module()
    cases = (
        ("", "checkpoint_dir"),
        ("relative/model", "absolute"),
        ("/approved/../model", "normalized"),
        ("/approved/\x00model", "NUL"),
        ("/" + "x" * 4097, "4096"),
        (object(), "checkpoint_dir"),
    )
    for checkpoint_dir, message in cases:
        _expect_error(
            lambda checkpoint_dir=checkpoint_dir: _request(
                module,
                checkpoint_dir=checkpoint_dir,
            ),
            message,
        )


def test_request_rejects_invalid_fingerprints_and_budget():
    module = _load_module()
    cases = (
        ({"model_fingerprint": "A" * 64}, "model_fingerprint"),
        (
            {"authorization_sha256": "not-a-sha256"},
            "authorization_sha256",
        ),
        ({"max_tensor_bytes": True}, "max_tensor_bytes"),
        ({"max_tensor_bytes": 0}, "max_tensor_bytes"),
    )
    for overrides, message in cases:
        _expect_error(
            lambda overrides=overrides: _request(
                module,
                **overrides,
            ),
            message,
        )


def test_validator_rejects_subclass_and_other_values():
    module = _load_module()

    class Derived(module.Qwen35CheckpointCandidateLoadRequest):
        pass

    derived = Derived(
        "/approved/model",
        MODEL_FINGERPRINT,
        8 << 20,
        AUTHORIZATION_SHA256,
    )
    for value in (derived, object()):
        _expect_error(
            lambda value=value: (
                module.validate_qwen35_checkpoint_candidate_load_request(
                    value
                )
            ),
            "exact",
        )


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 checkpoint worker request tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
