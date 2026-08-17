from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tinyvllm/engine/model_runner.py"


def test_model_runner_starts_peak_epoch_before_model_loading():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    init_method = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    source = ast.unparse(init_method)
    set_device = source.index("torch.cuda.set_device(rank)")
    reset_peak = source.index(
        "torch.cuda.reset_peak_memory_stats()",
        set_device,
    )
    load_model = source.index("_initialize_model_runner_model(")
    allocate_cache = source.index("self.allocate_kv_cache()")

    assert set_device < reset_peak < load_model < allocate_cache


def _run():
    test_model_runner_starts_peak_epoch_before_model_loading()
    print("qwen35 ModelRunner peak epoch source contract tests passed (1 test)")


if __name__ == "__main__":
    _run()
