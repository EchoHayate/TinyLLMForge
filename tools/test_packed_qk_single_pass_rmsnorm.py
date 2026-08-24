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


def _method(
    class_node: ast.ClassDef,
    name: str,
) -> ast.FunctionDef:
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_names(node: ast.AST) -> list[str]:
    names = []
    for call in (
        item for item in ast.walk(node) if isinstance(item, ast.Call)
    ):
        if isinstance(call.func, ast.Name):
            names.append(call.func.id)
        elif isinstance(call.func, ast.Attribute):
            names.append(call.func.attr)
    return names


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


def test_attention_exposes_packed_qk_receipt_contract():
    tree = ast.parse(QWEN3_PATH.read_text(encoding="utf-8"))
    attention = _class(tree, "QWen3Attention")
    receipt = _method(
        attention,
        "packed_qk_single_pass_rmsnorm_receipt",
    )
    returned = next(
        node.value
        for node in ast.walk(receipt)
        if isinstance(node, ast.Return)
    )

    assert isinstance(returned, ast.Dict)
    keys = {
        key.value
        for key in returned.keys
        if isinstance(key, ast.Constant)
    }
    assert keys == {
        "packed_qk_single_pass_rmsnorm_enabled",
        "q_heads",
        "kv_heads",
        "head_dim",
    }


def test_attention_has_compiled_packed_qk_helper():
    tree = ast.parse(QWEN3_PATH.read_text(encoding="utf-8"))
    attention = _class(tree, "QWen3Attention")
    helper = _method(attention, "_packed_qk_rmsnorm")

    assert any(
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Name)
        and decorator.func.id == "compile_if_enabled"
        and any(
            keyword.arg == "dynamic"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in decorator.keywords
        )
        for decorator in helper.decorator_list
    )
    calls = _call_names(helper)
    assert "cat" in calls
    assert "rsqrt" in calls
    assert "expand" not in calls
    assignments = {
        target.id
        for node in ast.walk(helper)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "q_normalized" in assignments
    assert "k_normalized" in assignments
    assert "weights" not in assignments


def test_normalize_qk_routes_disabled_and_enabled_paths():
    tree = ast.parse(QWEN3_PATH.read_text(encoding="utf-8"))
    attention = _class(tree, "QWen3Attention")
    normalize = _method(attention, "_normalize_qk")
    branches = [
        node
        for node in ast.walk(normalize)
        if isinstance(node, ast.If)
    ]

    assert len(branches) == 1
    branch = branches[0]
    assert ast.unparse(branch.test) == (
        "self.packed_qk_single_pass_rmsnorm"
    )
    assert "_packed_qk_rmsnorm" in _call_names(
        ast.Module(body=branch.body, type_ignores=[]),
    )
    disabled_calls = _call_names(
        ast.Module(body=branch.orelse, type_ignores=[]),
    )
    assert disabled_calls.count("q_norm") == 1
    assert disabled_calls.count("k_norm") == 1


@pytest.mark.parametrize("token_count", [1, 4, 17])
@pytest.mark.parametrize(
    ("q_heads", "kv_heads"),
    [(16, 8), (8, 8)],
)
def test_packed_qk_matches_separate_bf16_rmsnorm(
    monkeypatch,
    token_count,
    q_heads,
    kv_heads,
):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")

    from tinyvllm.layers.layernorm import RMSNorm
    from tinyvllm.models.qwen3 import QWen3Attention

    head_dim = 128
    attention = QWen3Attention.__new__(QWen3Attention)
    torch.nn.Module.__init__(attention)
    attention.num_heads = q_heads
    attention.num_kv_heads = kv_heads
    attention.head_dim = head_dim
    attention.q_size = q_heads * head_dim
    attention.kv_size = kv_heads * head_dim
    attention.packed_qk_single_pass_rmsnorm = True
    attention.q_norm = RMSNorm(head_dim, eps=1e-6)
    attention.k_norm = RMSNorm(head_dim, eps=1e-6)

    with torch.no_grad():
        attention.q_norm.weight.copy_(
            torch.linspace(
                0.75,
                1.25,
                head_dim,
                dtype=torch.float32,
            ),
        )
        attention.k_norm.weight.copy_(
            torch.linspace(
                1.5,
                0.5,
                head_dim,
                dtype=torch.float32,
            ),
        )

    generator = torch.Generator().manual_seed(
        token_count * 1000 + q_heads * 10 + kv_heads,
    )
    qkv = torch.randn(
        token_count,
        (q_heads + 2 * kv_heads) * head_dim,
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    q, k, _ = qkv.split(
        [attention.q_size, attention.kv_size, attention.kv_size],
        dim=-1,
    )
    expected_q = attention.q_norm(
        q.view(token_count, q_heads, head_dim),
    ).view(q.shape)
    expected_k = attention.k_norm(
        k.view(token_count, kv_heads, head_dim),
    ).view(k.shape)

    actual_q, actual_k = attention._normalize_qk(qkv)

    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)
