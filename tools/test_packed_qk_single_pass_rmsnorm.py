from __future__ import annotations

import __future__
import ast
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "tinyvllm" / "config.py"
MODEL_RUNNER_PATH = ROOT / "tinyvllm" / "engine" / "model_runner.py"
QWEN3_PATH = ROOT / "tinyvllm" / "models" / "qwen3.py"


def _load_config_module(monkeypatch):
    transformers = types.ModuleType("transformers")

    class AutoConfig:
        @staticmethod
        def from_pretrained(_model):
            return types.SimpleNamespace(
                model_type="qwen3",
                num_hidden_layers=28,
                max_position_embeddings=32768,
            )

    transformers.AutoConfig = AutoConfig
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    module = types.ModuleType("packed_qk_config_under_test")
    module.__file__ = str(CONFIG_PATH)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = CONFIG_PATH.read_text(encoding="utf-8")
    code = compile(
        source,
        str(CONFIG_PATH),
        "exec",
        flags=__future__.annotations.compiler_flag,
    )
    exec(code, module.__dict__)
    return module


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _init_keywords(class_node: ast.ClassDef, callee: str) -> set[str]:
    init = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    return {
        keyword.arg
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and (
            (
                isinstance(node.func, ast.Name)
                and node.func.id == callee
            )
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == callee
            )
        )
        for keyword in node.keywords
        if keyword.arg is not None
    }


def test_packed_qk_config_is_strict_and_default_off(
    monkeypatch,
    tmp_path,
):
    config_module = _load_config_module(monkeypatch)
    config_type = config_module.Config

    assert (
        config_type.__dataclass_fields__[
            "packed_qk_single_pass_rmsnorm"
        ].default
        is False
    )
    for invalid in (None, 0, 1, "true"):
        with pytest.raises(
            ValueError,
            match=(
                "^packed_qk_single_pass_rmsnorm must be a bool$"
            ),
        ):
            config_type(
                model=str(tmp_path),
                packed_qk_single_pass_rmsnorm=invalid,
            )


def test_model_runner_passes_packed_qk_mode_to_qwen3():
    tree = ast.parse(MODEL_RUNNER_PATH.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Qwen3ForCausalLM"
    ]

    assert len(calls) == 1
    keyword = next(
        item
        for item in calls[0].keywords
        if item.arg == "packed_qk_single_pass_rmsnorm"
    )
    assert ast.unparse(keyword.value) == (
        "runner_config.packed_qk_single_pass_rmsnorm"
    )


def test_qwen3_constructors_propagate_packed_qk_mode():
    tree = ast.parse(QWEN3_PATH.read_text(encoding="utf-8"))

    causal_lm = _class(tree, "Qwen3ForCausalLM")
    model = _class(tree, "Qwen3Model")
    decoder = _class(tree, "Qwen3DecoderLayer")
    attention = _class(tree, "QWen3Attention")

    for class_node in (causal_lm, model, decoder, attention):
        init = next(
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "__init__"
        )
        argument_names = {
            argument.arg
            for argument in (
                list(init.args.args)
                + list(init.args.kwonlyargs)
            )
        }
        assert "packed_qk_single_pass_rmsnorm" in argument_names

    assert "packed_qk_single_pass_rmsnorm" in _init_keywords(
        causal_lm,
        "Qwen3Model",
    )
    assert "packed_qk_single_pass_rmsnorm" in _init_keywords(
        model,
        "Qwen3DecoderLayer",
    )
    assert "packed_qk_single_pass_rmsnorm" in _init_keywords(
        decoder,
        "QWen3Attention",
    )
