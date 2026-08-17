from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXECUTOR = (
    ROOT / "tinyvllm/engine/autoregressive_draft_executor.py"
)
BACKEND = ROOT / "tinyvllm/engine/qwen3_draft_backend.py"
REGISTRATION = (
    ROOT / "tinyvllm/engine/autoregressive_draft_registration.py"
)
STORAGE = ROOT / "tinyvllm/engine/qwen3_draft_proposal_kv.py"
MODEL_RUNNER = ROOT / "tinyvllm/engine/model_runner.py"
CONFIG = ROOT / "tinyvllm/config.py"


TERMINAL_CLASSIFICATIONS = {
    "AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_ALLOCATOR_REUSE": (
        "ESTABLISHED"
    ),
    "AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_OFFLOAD_RUNTIME_WIRING": (
        "ESTABLISHED_LOCAL"
    ),
    "AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_OFFLOAD_DEFAULT": (
        "DISABLED"
    ),
    "REAL_PROPOSAL_KV_MOVEMENT": "NOT_ESTABLISHED",
    "LEARNED_DRAFTER_LOADED_PARITY": "NOT_ESTABLISHED",
    "PERFORMANCE_IMPROVEMENT": "NOT_ESTABLISHED",
    "PHASE_1": "NOT_ACHIEVED",
    "PROMOTION": "NOT_PROMOTABLE",
}


def _source(path):
    return path.read_text(encoding="utf-8")


def test_registration_uses_storage_aware_allocator_builder():
    registration = _source(REGISTRATION)
    model_runner = _source(MODEL_RUNNER)

    assert "build_qwen3_draft_proposal_kv_allocator" in registration
    assert "build_proposal_kv_allocator" in registration
    assert model_runner.index(
        'stage = "build_proposal_kv_allocator"'
    ) < model_runner.index(
        'stage = "build_proposal_kv_cache"',
        model_runner.index(
            'stage = "build_proposal_kv_allocator"'
        ),
    )
    assert (
        "dependencies.build_proposal_kv_allocator("
        in model_runner
    )
    assert "dependencies.build_proposal_kv_cache(\n" in model_runner
    assert "                    entry_allocator\n" in model_runner
    assert '"storage"' in model_runner
    assert '"physical_store"' in model_runner


def test_executor_owns_lease_mapping_and_completion():
    executor = _source(EXECUTOR)

    for required in (
        ".ensure_writable(",
        ".ensure_readable(",
        ".record_write_complete(",
        ".record_read_complete(",
        "physical_slot_ids",
        "writable_physical_slot_id",
        "visible_physical_slot_ids",
    ):
        assert required in executor


def test_backend_consumes_only_ephemeral_physical_mappings():
    backend = _source(BACKEND)

    assert "row.physical_slot_ids" in backend
    assert "row.writable_physical_slot_id" in backend
    assert "row.visible_physical_slot_ids" in backend
    assert "proposal_kv_cache.entry_allocator" in backend


def test_removed_physical_slot_cache_apis_do_not_return():
    production = "\n".join(
        _source(path)
        for path in (
            EXECUTOR,
            BACKEND,
            REGISTRATION,
            MODEL_RUNNER,
        )
    )

    for removed in (
        "staged_slot_ids",
        "committed_slot_ids",
        "proposal_kv_cache.physical_store",
        "ProposalKVCache.physical_store",
    ):
        assert removed not in production


def test_default_direct_mode_and_proposal_ceiling_remain_closed():
    registration = _source(REGISTRATION)
    storage = _source(STORAGE)
    config = _source(CONFIG)
    executor = _source(EXECUTOR)

    assert (
        "build_qwen3_draft_proposal_kv_allocator"
        in registration
    )
    assert "if not offload_enabled:" in storage
    assert "DirectProposalKVAllocator(storage)" in storage
    assert "ProposalKVResidencyManager(" in storage
    assert "autoregressive_draft_max_proposal_tokens: int = 4" in config
    assert (
        "autoregressive_draft_proposal_kv_offload_enabled: bool = False"
        in config
    )
    assert "max_proposal_tokens > 4" in executor


def test_terminal_classification_remains_fail_closed():
    assert TERMINAL_CLASSIFICATIONS == {
        "AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_ALLOCATOR_REUSE": (
            "ESTABLISHED"
        ),
        "AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_OFFLOAD_RUNTIME_WIRING": (
            "ESTABLISHED_LOCAL"
        ),
        "AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_OFFLOAD_DEFAULT": (
            "DISABLED"
        ),
        "REAL_PROPOSAL_KV_MOVEMENT": "NOT_ESTABLISHED",
        "LEARNED_DRAFTER_LOADED_PARITY": "NOT_ESTABLISHED",
        "PERFORMANCE_IMPROVEMENT": "NOT_ESTABLISHED",
        "PHASE_1": "NOT_ACHIEVED",
        "PROMOTION": "NOT_PROMOTABLE",
    }
