"""Dependency-light tests for explicit attention modes."""

from __future__ import annotations

import __future__
import importlib.util
import os
import sys
import types

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_VERIFIER_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "speculative",
    "verifier.py",
)
_CONTEXT_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "utils",
    "context.py",
)


def _load_context_module():
    class TensorAnnotation:
        def __or__(self, other):
            return object

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = TensorAnnotation()
    sys.modules.setdefault("torch", torch_stub)

    tinyvllm_package = types.ModuleType("tinyvllm")
    tinyvllm_package.__path__ = []
    speculative_package = types.ModuleType("tinyvllm.speculative")
    speculative_package.__path__ = []
    sys.modules["tinyvllm"] = tinyvllm_package
    sys.modules["tinyvllm.speculative"] = speculative_package

    verifier_spec = importlib.util.spec_from_file_location(
        "tinyvllm.speculative.verifier",
        _VERIFIER_PATH,
    )
    verifier_module = importlib.util.module_from_spec(verifier_spec)
    sys.modules["tinyvllm.speculative.verifier"] = verifier_module
    verifier_spec.loader.exec_module(verifier_module)

    context_module = types.ModuleType("context_modes_under_test")
    context_module.__file__ = _CONTEXT_PATH
    sys.modules["context_modes_under_test"] = context_module
    source = open(_CONTEXT_PATH).read()
    code = compile(
        source,
        _CONTEXT_PATH,
        "exec",
        flags=__future__.annotations.compiler_flag,
    )
    exec(code, context_module.__dict__)
    return context_module


context = _load_context_module()


def test_explicit_modes_are_preserved():
    for mode, expected_prefill in (
        ("prefill", True),
        ("decode", False),
        ("spec_verify", False),
    ):
        context.set_context(mode=mode)
        current = context.get_context()
        assert current.mode == mode
        assert current.is_prefill is expected_prefill


def test_legacy_boolean_callers_keep_current_behavior():
    context.set_context(True)
    assert context.get_context().mode == "prefill"
    context.set_context(False)
    assert context.get_context().mode == "decode"


def test_conflicting_mode_and_boolean_fail():
    for is_prefill, mode in (
        (True, "decode"),
        (False, "prefill"),
        (True, "spec_verify"),
    ):
        try:
            context.resolve_attention_mode(is_prefill, mode)
        except ValueError as exc:
            assert "conflicting attention mode" in str(exc)
        else:
            raise AssertionError((is_prefill, mode))


def test_reset_context_returns_decode_default():
    context.set_context(mode="spec_verify")
    context.reset_context()
    assert context.get_context().mode == "decode"
    assert context.get_context().is_prefill is False


def test_temporary_flash_attn_split_restores_previous_context():
    context.set_context(mode="decode", flash_attn_num_splits=0)
    original = context.get_context()
    with context.temporary_flash_attn_num_splits(16):
        assert context.get_context() is not original
        assert context.get_context().flash_attn_num_splits == 16
    assert context.get_context() is original
    assert context.get_context().flash_attn_num_splits == 0


def test_temporary_flash_attn_split_restores_after_exception():
    context.set_context(mode="decode", flash_attn_num_splits=0)
    original = context.get_context()
    try:
        with context.temporary_flash_attn_num_splits(16):
            raise RuntimeError("capture failed")
    except RuntimeError:
        pass
    assert context.get_context() is original
    assert context.get_context().flash_attn_num_splits == 0


def test_temporary_flash_attn_split_supports_nested_scopes():
    context.set_context(mode="decode", flash_attn_num_splits=0)
    with context.temporary_flash_attn_num_splits(16):
        with context.temporary_flash_attn_num_splits(1):
            assert context.get_context().flash_attn_num_splits == 1
        assert context.get_context().flash_attn_num_splits == 16
    assert context.get_context().flash_attn_num_splits == 0


def main():
    test_explicit_modes_are_preserved()
    test_legacy_boolean_callers_keep_current_behavior()
    test_conflicting_mode_and_boolean_fail()
    test_reset_context_returns_decode_default()
    test_temporary_flash_attn_split_restores_previous_context()
    test_temporary_flash_attn_split_restores_after_exception()
    test_temporary_flash_attn_split_supports_nested_scopes()
    print("context mode tests passed")


if __name__ == "__main__":
    main()
