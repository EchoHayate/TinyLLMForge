from __future__ import annotations

import pytest

from tools.autoregressive_draft_tp4_local_gate import (
    validate_autoregressive_draft_tp4_local_evidence,
)


def _rank_snapshot(rank):
    logical_rows = (
        {
            "stage": "proposal_materialized",
            "rows": ({
                "sequence_id": 7,
                "sequence_epoch": 0,
                "exact_q": 4,
                "proposal_token_ids": (11, 12, 13, 14),
                "logical_state": "materialized",
            },),
        },
        {
            "stage": "release_complete",
            "rows": ({
                "sequence_id": 7,
                "sequence_epoch": 0,
                "active_transaction_count": 0,
                "active_ticket_count": 0,
                "committed_logical_entries": 0,
                "live_local_slot_count": 0,
            },),
        },
    )
    return {
        "rank": rank,
        "world_size": 4,
        "registered": True,
        "registration_consensus_sha256": "a" * 64,
        "checkpoint_identity": {
            "target": {"composite_sha256": "b" * 64},
            "draft": {"composite_sha256": "c" * 64},
        },
        "tokenizer_contract": {
            "target": {"composite_sha256": "d" * 64},
            "draft": {"composite_sha256": "e" * 64},
        },
        "executor_descriptor": {
            "executor_id": "autoregressive-draft",
            "capabilities": {
                "source_type": "independent_draft_model",
                "supports_batch": True,
                "requires_target_hidden": False,
                "requires_target_logits": False,
                "max_proposal_tokens": 4,
                "execution_domain": "model_runner",
                "requires_proposal_lifecycle": True,
                "requires_full_token_history": False,
            },
        },
        "registration_error": None,
        "executor": {
            "rank": rank,
            "world_size": 4,
            "backend_identity": "qwen3",
            "logical_authority_rows": logical_rows,
            "logical_authority_digest_count": len(logical_rows),
            "last_logical_authority_sha256": "f" * 64,
            "timing_ms": {
                "prompt_bootstrap": 1.0 + rank,
                "proposal_forward": 2.0 + rank,
                "proposal_finalize": 3.0 + rank,
            },
            "proposal_kv_lifecycle": {
                "active_transaction_count": 0,
                "prepared_ticket_count": 0,
                "proposal_kv_cache": {
                    "owned_entry_count": 0,
                },
            },
            "backend": {
                "backend_identity": "qwen3",
                "local_proposal_kv_bytes": 1024 + rank,
                "local_prefill_forward_count": 1,
                "local_decode_forward_count": 3,
                "proposal_kv_storage_id": f"rank-{rank}-store",
                "proposal_kv_cache": {
                    "entry_allocator": {
                        "allocator_mode": "direct",
                        "accepted_entry_copy_count": 0,
                        "accepted_entry_replay_count": 0,
                        "accepted_entry_rematerialization_count": 0,
                    },
                },
                "physical_store": {
                    "allocated_slot_count": 0,
                },
            },
        },
    }


def test_local_gate_requires_exactly_four_distinct_rank_snapshots():
    with pytest.raises(ValueError, match="exactly four"):
        validate_autoregressive_draft_tp4_local_evidence(
            tuple(_rank_snapshot(rank) for rank in range(3))
        )


def test_local_gate_accepts_rank_local_physical_identity_differences():
    snapshots = tuple(_rank_snapshot(rank) for rank in range(4))

    aggregate = validate_autoregressive_draft_tp4_local_evidence(
        snapshots
    )

    assert aggregate["rank_count"] == 4
    assert aggregate["total_proposal_kv_bytes"] == sum(
        row["executor"]["backend"]["local_proposal_kv_bytes"]
        for row in snapshots
    )
    assert aggregate["max_rank_timing_ms"] == {
        "prompt_bootstrap": 4.0,
        "proposal_forward": 5.0,
        "proposal_finalize": 6.0,
    }
    assert aggregate["classification"] == "NOT_PROMOTABLE"
    assert aggregate["promotion_boundary"]["phase_1"] == (
        "NOT_ACHIEVED"
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda rows: rows[2].update(
                registration_consensus_sha256="mismatch"
            ),
            "registration consensus",
        ),
        (
            lambda rows: rows[2]["executor"]["backend"].update(
                local_prefill_forward_count=0
            ),
            "prefill",
        ),
        (
            lambda rows: rows[2]["executor"].update(
                logical_authority_rows=(),
            ),
            "logical authority",
        ),
        (
            lambda rows: rows[2]["executor"][
                "proposal_kv_lifecycle"
            ].update(active_transaction_count=1),
            "active transaction",
        ),
        (
            lambda rows: rows[2].update(registered=False),
            "registered",
        ),
        (
            lambda rows: rows[2].update(
                registration_error={"stage": "build"}
            ),
            "registration error",
        ),
        (
            lambda rows: rows[2]["executor"]["backend"][
                "proposal_kv_cache"
            ]["entry_allocator"].update(
                allocator_mode="residency"
            ),
            "allocator mode",
        ),
        (
            lambda rows: rows[2]["executor"]["backend"][
                "proposal_kv_cache"
            ]["entry_allocator"].update(
                accepted_entry_replay_count=1
            ),
            "accepted proposal KV replay",
        ),
    ),
)
def test_local_gate_rejects_incomplete_rank_evidence(
    mutate,
    message,
):
    snapshots = [
        _rank_snapshot(rank) for rank in range(4)
    ]
    mutate(snapshots)

    with pytest.raises(ValueError, match=message):
        validate_autoregressive_draft_tp4_local_evidence(
            tuple(snapshots)
        )
