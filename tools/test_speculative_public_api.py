from __future__ import annotations

import importlib.util
import os
import sys
import types


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_PACKAGE_DIR = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "speculative",
)


def _load_public_package():
    tinyvllm_package = types.ModuleType("tinyvllm")
    tinyvllm_package.__path__ = [
        os.path.join(_REPO_ROOT, "tinyvllm")
    ]
    sys.modules["tinyvllm"] = tinyvllm_package
    for module_name in tuple(sys.modules):
        if module_name.startswith("tinyvllm.speculative"):
            sys.modules.pop(module_name)
    spec = importlib.util.spec_from_file_location(
        "tinyvllm.speculative",
        os.path.join(_PACKAGE_DIR, "__init__.py"),
        submodule_search_locations=[_PACKAGE_DIR],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["tinyvllm.speculative"] = module
    spec.loader.exec_module(module)
    return module


def test_public_api_exports_adapter_and_batch_runtime_contracts():
    module = _load_public_package()
    expected = {
        "DraftAdapter",
        "DraftCapabilities",
        "DraftContext",
        "DraftProposal",
        "validate_draft_capabilities",
        "validate_draft_adapter_batch",
        "NGramDraftAdapter",
        "SAMDraftAdapter",
        "FirstTargetResult",
        "FirstTargetProposalResult",
        "TailBatchItem",
        "TailBatchResult",
        "NativeSpeculativeSequenceResult",
        "NativeSpeculativeBatchResult",
        "NativeSpeculativeBatchError",
        "PreparedNativeSpeculativeSequence",
        "PreparedNativeSpeculativeBatch",
        "prepare_native_speculative_batch",
        "commit_prepared_native_speculative_batch",
        "rollback_prepared_native_speculative_batch",
        "execute_native_speculative_batch",
    }

    assert expected <= set(module.__all__)
    for name in expected:
        assert getattr(module, name) is not None
