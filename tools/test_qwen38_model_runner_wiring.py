from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODEL_RUNNER_PATH = ROOT / "tinyvllm/engine/model_runner.py"


def _load_function(name):
    tree = ast.parse(
        MODEL_RUNNER_PATH.read_text(encoding="utf-8"),
        filename=str(MODEL_RUNNER_PATH),
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(MODEL_RUNNER_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def test_resolves_qwen38_profile_only_for_official_architecture():
    resolve = _load_function("_resolve_qwen38_text_profile")
    calls = []
    profile = object()
    qwen38 = SimpleNamespace(
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=SimpleNamespace(
            num_hidden_layers=64,
            hidden_size=5120,
            intermediate_size=17408,
        ),
    )

    assert resolve(
        qwen38,
        adopt_qwen38_text=lambda value: (
            calls.append(value) or profile
        ),
    ) is profile
    assert calls == [qwen38]
    assert resolve(
        SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
        adopt_qwen38_text=lambda _value: calls.append("unexpected"),
    ) is None
    assert calls == [qwen38]


def test_same_architecture_with_other_topology_stays_qwen35():
    resolve = _load_function("_resolve_qwen38_text_profile")
    calls = []
    hf_config = SimpleNamespace(
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=SimpleNamespace(
            num_hidden_layers=36,
            hidden_size=2048,
            intermediate_size=6144,
        ),
    )

    assert resolve(
        hf_config,
        adopt_qwen38_text=lambda value: calls.append(value),
    ) is None
    assert calls == []


def test_model_runner_validates_before_distributed_and_checks_each_batch():
    source = MODEL_RUNNER_PATH.read_text(encoding="utf-8")
    init_body = source[
        source.index("class ModelRunner:"):
        source.index("    def bind_kv_block_identity_rows")
    ]
    run_body = source[
        source.index("    def run(self, seqs:"):
        source.index("    @torch.inference_mode()", source.index(
            "    def run(self, seqs:"
        ))
    ]

    assert init_body.index(
        "self.qwen38_text_profile ="
    ) < init_body.index("dist.init_process_group(")
    assert "validate_qwen38_sequence_batch(" in run_body
    assert run_body.index(
        "validate_qwen38_sequence_batch("
    ) < run_body.index("self.bind_kv_block_identity_rows(")
