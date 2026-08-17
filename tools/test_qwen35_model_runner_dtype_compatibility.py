import ast
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_resolver():
    path = REPO_ROOT / "tinyvllm" / "engine" / "model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_hf_model_dtype"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_resolve_hf_model_dtype"]


def _load_linear_execution_configurator():
    path = REPO_ROOT / "tinyvllm" / "engine" / "model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_configure_qwen35_linear_execution"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_configure_qwen35_linear_execution"]


def test_qwen35_missing_transformers_dtype_defaults_to_bfloat16():
    resolver = _load_resolver()
    torch_module = SimpleNamespace(
        float16=object(),
        bfloat16=object(),
        float32=object(),
        float64=object(),
        get_default_dtype=lambda: object(),
    )
    hf_config = SimpleNamespace(
        model_type="qwen3_5",
        dtype=None,
        torch_dtype=None,
    )

    resolved = resolver(hf_config, torch_module)

    assert resolved is torch_module.bfloat16
    assert hf_config.torch_dtype is torch_module.bfloat16


def test_transformers_dtype_field_takes_precedence_over_legacy_field():
    resolver = _load_resolver()
    torch_module = SimpleNamespace(
        float16=object(),
        bfloat16=object(),
        float32=object(),
        float64=object(),
        get_default_dtype=lambda: object(),
    )
    hf_config = SimpleNamespace(
        model_type="qwen3_5",
        dtype=torch_module.float16,
        torch_dtype=torch_module.float32,
    )

    resolved = resolver(hf_config, torch_module)

    assert resolved is torch_module.float16
    assert hf_config.torch_dtype is torch_module.float16


def test_qwen35_linear_execution_uses_1024_fixed_rows():
    configure = _load_linear_execution_configurator()
    model = object()
    calls = []

    result = configure(
        model,
        configure_rows=lambda received, rows: calls.append(
            (received, rows)
        ),
    )

    assert result is model
    assert calls == [(model, 1024)]


