from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "tinyvllm/utils/torch_compile.py"
COMPILED_LAYER_PATHS = (
    ROOT / "tinyvllm/layers/activation.py",
    ROOT / "tinyvllm/layers/layernorm.py",
    ROOT / "tinyvllm/layers/rotary_embedding.py",
)
LINEAR_PATH = ROOT / "tinyvllm/layers/linear.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _without_modules(*names):
    previous = {name: sys.modules.pop(name, None) for name in names}

    def restore():
        for name in names:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]

    return restore


def test_disabled_policy_never_calls_torch_compile():
    calls = []
    original_disable = os.environ.get("TORCH_COMPILE_DISABLE")
    fake_torch = ModuleType("torch")

    def forbidden_compile(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("torch.compile was called while disabled")

    fake_torch.compile = forbidden_compile
    restore_modules = _without_modules(
        "torch",
        "tinyvllm.utils.torch_compile",
    )
    try:
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        sys.modules["torch"] = fake_torch
        policy = _load_module(
            "tinyvllm.utils.torch_compile",
            POLICY_PATH,
        )

        @policy.compile_if_enabled(dynamic=True)
        def function(value):
            return value + 1

        assert function(2) == 3
        assert calls == []
    finally:
        if original_disable is None:
            os.environ.pop("TORCH_COMPILE_DISABLE", None)
        else:
            os.environ["TORCH_COMPILE_DISABLE"] = original_disable
        restore_modules()


def test_enabled_policy_preserves_dynamic_compile_decorator():
    calls = []
    original_disable = os.environ.pop("TORCH_COMPILE_DISABLE", None)
    fake_torch = ModuleType("torch")

    def fake_compile(*args, **kwargs):
        calls.append((args, kwargs))

        def decorate(function):
            return function

        return decorate

    fake_torch.compile = fake_compile
    restore_modules = _without_modules(
        "torch",
        "tinyvllm.utils.torch_compile",
    )
    try:
        sys.modules["torch"] = fake_torch
        policy = _load_module(
            "tinyvllm.utils.torch_compile",
            POLICY_PATH,
        )

        @policy.compile_if_enabled(dynamic=True)
        def function(value):
            return value + 1

        assert function(2) == 3
        assert calls == [((), {"dynamic": True})]
    finally:
        if original_disable is not None:
            os.environ["TORCH_COMPILE_DISABLE"] = original_disable
        restore_modules()


def test_compiled_layers_use_policy_not_torch_compile_directly():
    for path in COMPILED_LAYER_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = False
        decorators = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if (
                    node.module == "tinyvllm.utils.torch_compile"
                    and any(
                        alias.name == "compile_if_enabled"
                        for alias in node.names
                    )
                ):
                    imported = True
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                decorators.extend(node.decorator_list)
        assert imported, path
        assert any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "compile_if_enabled"
            for decorator in decorators
        ), path
        assert not any(
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and isinstance(decorator.func.value, ast.Name)
            and decorator.func.value.id == "torch"
            and decorator.func.attr == "compile"
            for decorator in decorators
        ), path


def test_bitsandbytes_is_loaded_only_inside_lazy_helper():
    tree = ast.parse(LINEAR_PATH.read_text(encoding="utf-8"))
    top_level_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not any(
        (
            isinstance(node, ast.Import)
            and any(
                alias.name == "bitsandbytes"
                or alias.name.startswith("bitsandbytes.")
                for alias in node.names
            )
        )
        or (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (
                node.module == "bitsandbytes"
                or node.module.startswith("bitsandbytes.")
            )
        )
        for node in top_level_imports
    )
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_load_bitsandbytes_functional"
    ]
    assert len(helpers) == 1
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "importlib"
        and node.func.attr == "import_module"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "bitsandbytes.functional"
        for node in ast.walk(helpers[0])
    )


def test_lazy_bitsandbytes_loader_caches_success_and_failure():
    source = LINEAR_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_load_bitsandbytes_functional"
    )
    helper_module = ast.Module(body=[helper], type_ignores=[])

    class FakeImportlib:

        def __init__(self, result=None, error=None):
            self.result = result
            self.error = error
            self.calls = []

        def import_module(self, name):
            self.calls.append(name)
            if self.error is not None:
                raise self.error
            return self.result

    success = object()
    success_importlib = FakeImportlib(result=success)
    namespace = {
        "importlib": success_importlib,
        "_BNB_FUNCTIONAL": None,
        "_BNB_IMPORT_ATTEMPTED": False,
    }
    exec(compile(helper_module, str(LINEAR_PATH), "exec"), namespace)
    assert namespace["_load_bitsandbytes_functional"]() is success
    assert namespace["_load_bitsandbytes_functional"]() is success
    assert success_importlib.calls == ["bitsandbytes.functional"]

    failure_importlib = FakeImportlib(error=ImportError("missing"))
    namespace = {
        "importlib": failure_importlib,
        "_BNB_FUNCTIONAL": None,
        "_BNB_IMPORT_ATTEMPTED": False,
    }
    exec(compile(helper_module, str(LINEAR_PATH), "exec"), namespace)
    assert namespace["_load_bitsandbytes_functional"]() is None
    assert namespace["_load_bitsandbytes_functional"]() is None
    assert failure_importlib.calls == ["bitsandbytes.functional"]


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "tinyvllm torch compile policy tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    main()
