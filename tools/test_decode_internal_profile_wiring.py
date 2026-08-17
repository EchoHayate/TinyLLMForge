from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_RUNNER = ROOT / "tinyvllm/engine/model_runner.py"
LLM_ENGINE = ROOT / "tinyvllm/engine/llm_engine.py"
LINEAR = ROOT / "tinyvllm/layers/linear.py"
EMBED_HEAD = ROOT / "tinyvllm/layers/embed_head.py"
QWEN35_COMPONENTS = ROOT / "tinyvllm/models/qwen35_components.py"


def _class_method(path, class_name, method_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == method_name
    )


def _called_names(node):
    names = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            names.append(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            names.append(child.func.attr)
    return names


def test_model_runner_exposes_profile_lifecycle_and_wraps_run():
    configure = _class_method(
        MODEL_RUNNER,
        "ModelRunner",
        "configure_decode_internal_profile",
    )
    finalize = _class_method(
        MODEL_RUNNER,
        "ModelRunner",
        "finalize_decode_internal_profile",
    )
    run = _class_method(MODEL_RUNNER, "ModelRunner", "run")

    assert "DecodeInternalProfiler" in _called_names(configure)
    assert "finalize" in _called_names(finalize)
    assert "run_profiled_step" in _called_names(run)
    assert "_run_model_step" in _called_names(run)


def test_llm_engine_exposes_acknowledged_profile_lifecycle():
    configure = _class_method(
        LLM_ENGINE,
        "LLMEngine",
        "configure_decode_internal_profile",
    )
    finalize = _class_method(
        LLM_ENGINE,
        "LLMEngine",
        "finalize_decode_internal_profile",
    )

    assert "call_model_runner_acknowledged" in _called_names(configure)
    assert "call_model_runner_acknowledged" in _called_names(finalize)


def test_collective_call_sites_use_profile_helper():
    linear_source = LINEAR.read_text(encoding="utf-8")
    embed_source = EMBED_HEAD.read_text(encoding="utf-8")

    assert linear_source.count(
        'profile_collective(\n                        "row_parallel_all_reduce"'
    ) == 1
    assert linear_source.count(
        'profile_collective(\n                "row_parallel_all_reduce"'
    ) == 1
    assert (
        'profile_collective(\n'
        '                "vocab_parallel_embedding_all_reduce"'
        in embed_source
    )
    assert (
        'profile_collective(\n'
        '                "replicated_weight_row_parallel_all_gather"'
        in linear_source
    )


def test_qwen35_output_projections_use_true_row_parallel_layout():
    components_source = QWEN35_COMPONENTS.read_text(encoding="utf-8")

    assert "ReplicatedWeightRowParallelLinear(" not in components_source
    assert components_source.count("RowParallelLinear(") >= 2
