from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tinyvllm/engine/llm_engine.py"


def test_engine_exit_reclaims_rank0_cuda_allocator_before_receipt():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imports_gc = any(
        isinstance(node, ast.Import)
        and any(alias.name == "gc" for alias in node.names)
        for node in tree.body
    )
    engine = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LLMEngine"
    )
    exit_method = next(
        node
        for node in engine.body
        if isinstance(node, ast.FunctionDef) and node.name == "exit"
    )
    source = ast.unparse(exit_method)
    release_model_runner = source.index("del self.model_runner")
    collect_cycles = source.index("gc.collect()")
    release_allocator = source.index("torch.cuda.empty_cache()")
    publish_receipt = source.index("self._exit_receipt = dict(receipt)")

    assert imports_gc
    assert (
        release_model_runner
        < collect_cycles
        < release_allocator
        < publish_receipt
    )


def test_engine_explicit_exit_unregisters_atexit_handler_once():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    engine = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LLMEngine"
    )
    init_method = next(
        node
        for node in engine.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    exit_method = next(
        node
        for node in engine.body
        if isinstance(node, ast.FunctionDef) and node.name == "exit"
    )
    init_source = ast.unparse(init_method)
    exit_source = ast.unparse(exit_method)

    assert init_source.count("atexit.register(self.exit)") == 1
    assert exit_source.count("atexit.unregister(self.exit)") == 1
    return_existing_receipt = exit_source.index(
        "if existing_receipt is not None:\n        return dict(existing_receipt)"
    )
    unregister_handler = exit_source.index(
        "atexit.unregister(self.exit)"
    )
    collect_cycles = exit_source.index("gc.collect()")

    assert return_existing_receipt < unregister_handler < collect_cycles


def _run():
    test_engine_exit_reclaims_rank0_cuda_allocator_before_receipt()
    test_engine_explicit_exit_unregisters_atexit_handler_once()
    print("qwen35 TP4 Engine exit cleanup source contract tests passed (2 tests)")


if __name__ == "__main__":
    _run()
