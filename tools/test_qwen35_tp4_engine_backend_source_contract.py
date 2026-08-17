from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name, filename):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


source_contract = _load(
    "qwen35_tp4_engine_backend_source_contract",
    "qwen35_tp4_engine_backend_source_contract.py",
)


def test_current_source_satisfies_backend_api_contract():
    result = source_contract.inspect_source(ROOT)
    assert result["classification"] == "PASS"
    assert result["missing_methods"] == []
    assert result["signature_mismatches"] == []
    assert result["action_coverage"] == {
        "construct_engine": "LLMEngine.__init__",
        "configure_exact_restore": (
            "LLMEngine.configure_qwen35_hybrid_prefix_publication_runtime"
        ),
        "verify_rank_bindings": (
            "LLMEngine.qwen35_hybrid_prefix_authority_snapshots"
        ),
        "submit_source_request": "LLMEngine.add_request",
        "submit_cached_continuation": "LLMEngine.add_request",
        "submit_token_mismatch": "LLMEngine.add_request",
        "run_to_completion": "LLMEngine.step",
        "drain_release_events": (
            "LLMEngine.hybrid_state_release_event_count"
        ),
        "snapshot_cache": (
            "LLMEngine.qwen35_hybrid_prefix_authority_snapshots"
        ),
        "clear_reusable_cache": (
            "LLMEngine.clear_qwen35_hybrid_prefix_caches"
        ),
        "invalidate_block_generation": (
            "LLMEngine.invalidate_qwen35_hybrid_prefix_blocks"
        ),
        "close_engine": "LLMEngine.exit",
    }


def test_required_signatures_are_exact():
    result = source_contract.inspect_source(ROOT)
    assert result["signatures"][
        "LLMEngine.qwen35_hybrid_prefix_authority_snapshots"
    ] == "(self, *, timeout_s)"
    assert result["signatures"][
        "LLMEngine.clear_qwen35_hybrid_prefix_caches"
    ] == "(self, *, timeout_s)"
    assert result["signatures"][
        "LLMEngine.invalidate_qwen35_hybrid_prefix_blocks"
    ] == "(self, block_identities, *, timeout_s)"
    assert result["signatures"][
        "LLMEngine.hybrid_state_release_event_count"
    ] == "(self)"
    assert result["signatures"][
        "ModelRunner.clear_qwen35_hybrid_prefix_cache"
    ] == "(self)"
    assert result["signatures"][
        "ModelRunner.invalidate_qwen35_hybrid_prefix_blocks"
    ] == "(self, block_identities)"


def test_tampered_source_reports_missing_method():
    with tempfile.TemporaryDirectory() as temporary:
        source_root = Path(temporary) / "source"
        engine_dir = source_root / "tinyvllm/engine"
        engine_dir.mkdir(parents=True)
        for name in ("llm_engine.py", "model_runner.py"):
            source = (
                ROOT / "tinyvllm/engine" / name
            ).read_text(encoding="utf-8")
            if name == "llm_engine.py":
                source = source.replace(
                    "    def clear_qwen35_hybrid_prefix_caches(",
                    "    def removed_clear_qwen35_hybrid_prefix_caches(",
                    1,
                )
            (engine_dir / name).write_text(
                source,
                encoding="utf-8",
            )

        result = source_contract.inspect_source(source_root)
        assert result["classification"] == "FAIL"
        assert (
            "LLMEngine.clear_qwen35_hybrid_prefix_caches"
            in result["missing_methods"]
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
        "qwen35 TP4 Engine backend source contract tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
