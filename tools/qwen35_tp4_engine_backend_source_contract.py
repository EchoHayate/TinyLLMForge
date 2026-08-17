from __future__ import annotations

import ast
from pathlib import Path


METHOD_SIGNATURES = {
    "LLMEngine.__init__": "(self, model, **kwargs)",
    "LLMEngine.configure_qwen35_hybrid_prefix_publication_runtime": (
        "(self, *, model_fingerprint, max_entries, max_bytes, timeout_s)"
    ),
    "LLMEngine.qwen35_hybrid_prefix_cache_snapshots": (
        "(self, *, timeout_s)"
    ),
    "LLMEngine.qwen35_hybrid_prefix_authority_snapshots": (
        "(self, *, timeout_s)"
    ),
    "LLMEngine.add_request": "(self, prompt, sampling_params)",
    "LLMEngine.step": "(self)",
    "LLMEngine.hybrid_state_release_event_count": "(self)",
    "LLMEngine.clear_qwen35_hybrid_prefix_caches": (
        "(self, *, timeout_s)"
    ),
    "LLMEngine.invalidate_qwen35_hybrid_prefix_blocks": (
        "(self, block_identities, *, timeout_s)"
    ),
    "LLMEngine.exit": "(self)",
    "ModelRunner.qwen35_hybrid_prefix_cache_snapshot": "(self)",
    "ModelRunner.clear_qwen35_hybrid_prefix_cache": "(self)",
    "ModelRunner.invalidate_qwen35_hybrid_prefix_blocks": (
        "(self, block_identities)"
    ),
}

ACTION_COVERAGE = {
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

_CLASS_FILES = {
    "LLMEngine": "tinyvllm/engine/llm_engine.py",
    "ModelRunner": "tinyvllm/engine/model_runner.py",
}


def _format_arguments(arguments):
    positional = list(arguments.posonlyargs) + list(arguments.args)
    parts = [argument.arg for argument in positional]
    if arguments.vararg is not None:
        parts.append(f"*{arguments.vararg.arg}")
    elif arguments.kwonlyargs:
        parts.append("*")
    parts.extend(argument.arg for argument in arguments.kwonlyargs)
    if arguments.kwarg is not None:
        parts.append(f"**{arguments.kwarg.arg}")
    return "(" + ", ".join(parts) + ")"


def _class_methods(path, class_name):
    try:
        tree = ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
        )
    except (OSError, UnicodeDecodeError, SyntaxError):
        return {}
    class_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == class_name
        ),
        None,
    )
    if class_node is None:
        return {}
    return {
        node.name: _format_arguments(node.args)
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def inspect_source(source_root):
    source_root = Path(source_root)
    discovered = {}
    for class_name, relative_path in _CLASS_FILES.items():
        methods = _class_methods(
            source_root / relative_path,
            class_name,
        )
        discovered.update({
            f"{class_name}.{method_name}": signature
            for method_name, signature in methods.items()
        })
    missing = sorted(
        name for name in METHOD_SIGNATURES
        if name not in discovered
    )
    mismatches = sorted(
        name
        for name, expected in METHOD_SIGNATURES.items()
        if name in discovered and discovered[name] != expected
    )
    return {
        "classification": (
            "PASS" if not missing and not mismatches else "FAIL"
        ),
        "missing_methods": missing,
        "signature_mismatches": mismatches,
        "signatures": {
            name: discovered.get(name)
            for name in METHOD_SIGNATURES
        },
        "action_coverage": dict(ACTION_COVERAGE),
    }
