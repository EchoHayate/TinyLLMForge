from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


TOOLS = Path(__file__).resolve().parent
MODEL_RUNNER_PATH = (
    TOOLS.parent / "tinyvllm" / "engine" / "model_runner.py"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_model_runner_method(name: str):
    tree = ast.parse(
        MODEL_RUNNER_PATH.read_text(encoding="utf-8"),
        filename=str(MODEL_RUNNER_PATH),
    )
    model_runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ModelRunner"
    )
    method = next(
        node
        for node in model_runner.body
        if isinstance(node, ast.FunctionDef)
        and node.name == name
    )
    function = ast.FunctionDef(
        name=method.name,
        args=method.args,
        body=method.body,
        decorator_list=[],
        returns=method.returns,
        type_comment=method.type_comment,
    )
    namespace = {}
    module = ast.fix_missing_locations(
        ast.Module(
            body=[function],
            type_ignores=[],
        )
    )
    exec(
        compile(
            module,
            str(MODEL_RUNNER_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


gate = _load_module(
    "qwen35_generic_speculative_tp4_gate",
    TOOLS / "qwen35_generic_speculative_tp4_gate.py",
)


def test_contract_constants_are_frozen():
    assert gate.SCHEMA_VERSION == (
        "qwen35.generic-speculative-tp4-"
        "transactional-correctness.v1"
    )
    assert gate.CLASSIFICATION == (
        "SECOND_MODEL_TP4_4K_ESTABLISHED"
    )
    assert gate.CLAIM_SCOPE == "second_model_tp4_4k_only"
    assert gate.WORLD_SIZE == 4
    assert gate.BATCH_SIZES == (1, 4)
    assert gate.POLICIES == ("baseline", "ngram")
    assert gate.CONTEXT_TOKENS == 4096
    assert gate.NGRAM_SIZE == 3
    assert gate.MAX_PROPOSAL_TOKENS == 4
    assert gate.MAX_OUTPUT_TOKENS == 8
    assert gate.MODEL_MANIFEST_SHA256 == (
        "3e650a908234771c3cf1ac4e20c4d38fe"
        "69982efedaf4a3e631ad0b14aad7dd0"
    )
    assert "phase1_not_promotable" in gate.LIMITATIONS
    assert gate.cell_key("baseline", 1) == "baseline:b1"
    assert gate.cell_key("ngram", 4) == "ngram:b4"


def test_model_identity_requires_real_qwen35_hybrid_shape():
    identity = gate._validate_model_identity({
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_layer_count": 24,
        "linear_layer_count": 18,
        "full_attention_layer_count": 6,
    })
    assert identity == {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_layer_count": 24,
        "linear_layer_count": 18,
        "full_attention_layer_count": 6,
    }


def test_consumed_input_mapping_is_canonical():
    row = gate._validate_mapping({
        "sequence_id": 9,
        "transaction_ordinal": 0,
        "proposal_token_count": 4,
        "accepted_draft_count": 2,
        "verify_input_count": 3,
        "committed_tail_input_count": 2,
        "committed_input_count": 3,
    })
    assert row["committed_input_count"] == 3


def test_consumed_input_mapping_rejects_output_length_inference():
    with pytest.raises(
        ValueError,
        match="committed input count mismatch",
    ):
        gate._validate_mapping({
            "sequence_id": 9,
            "transaction_ordinal": 0,
            "proposal_token_count": 4,
            "accepted_draft_count": 2,
            "verify_input_count": 3,
            "committed_tail_input_count": 2,
            "committed_input_count": 4,
        })


def _side_state_receipts(
    rank: int,
    handle_id: str,
    sequence_id: int,
) -> list[dict]:
    return [
        {
            "rank": rank,
            "handle_id": handle_id,
            "sequence_id": sequence_id,
            "operation": operation,
            "state": state,
        }
        for operation, state in (
            ("prepare", "prepared"),
            ("select", "selected"),
            ("apply", "applied"),
            ("seal", "sealed"),
        )
    ]


def _sequence_transaction(
    rank: int,
    *,
    physical_slot_id: int,
) -> dict:
    return {
        "rank": rank,
        "cell_key": "ngram:b1",
        "sequence_id": 7,
        "transaction_ordinal": 0,
        "proposal_token_ids": [11, 12, 13, 14],
        "acceptance_mask": [True, True, False, False],
        "proposal_token_count": 4,
        "accepted_draft_count": 2,
        "verify_input_count": 3,
        "committed_tail_input_count": 2,
        "committed_input_count": 3,
        "kv_decision": "commit_prefix_3_rollback_suffix",
        "selected_checkpoint_id": "checkpoint:3",
        "physical_slot_id": physical_slot_id,
    }


def test_side_state_lifecycle_is_rank_handle_sequence_scoped():
    receipts = (
        _side_state_receipts(0, "handle-0", 7)
        + _side_state_receipts(1, "handle-0", 7)
    )
    normalized = gate._validate_side_state(
        receipts,
        [],
        rank=0,
    )
    assert {
        row["rank"]
        for row in normalized
    } == {0, 1}


def test_side_state_lifecycle_rejects_missing_select():
    receipts = _side_state_receipts(0, "handle-0", 7)
    receipts = [
        row
        for row in receipts
        if row["operation"] != "select"
    ]
    with pytest.raises(
        ValueError,
        match="side-state lifecycle receipts are incomplete",
    ):
        gate._validate_side_state(receipts, [], rank=0)


def test_transaction_semantic_digest_ignores_physical_slots():
    first = gate._transaction_semantic_digest(
        _sequence_transaction(0, physical_slot_id=17)
    )
    second = gate._transaction_semantic_digest(
        _sequence_transaction(0, physical_slot_id=91)
    )
    assert first == second


def test_sequence_transaction_rejects_rank_mismatch():
    with pytest.raises(
        ValueError,
        match="transaction rank mismatch",
    ):
        gate._validate_sequence_transaction(
            _sequence_transaction(
                1,
                physical_slot_id=17,
            ),
            rank=0,
        )


def _profile_step(
    rank: int,
    step_index: int,
    batch_kind: str,
) -> dict:
    return {
        "rank": rank,
        "step_index": step_index,
        "batch_kind": batch_kind,
        "is_decode": True,
        "decode_ordinal": step_index,
        "active_sequence_count": 2,
        "request_set_sha256": "a" * 64,
        "dispatch": "eager",
        "wall_ns": 1000 + rank,
        "cuda_ns": 800 + rank,
        "non_cuda_upper_bound_ns": 200,
    }


def _profile_collective(
    rank: int,
    step_index: int,
) -> dict:
    return {
        "rank": rank,
        "step_index": step_index,
        "decode_ordinal": step_index,
        "operation": "row_parallel_all_reduce",
        "tensor_shape": [2, 151936],
        "tensor_dtype": "torch.float16",
        "wall_ns": 100 + rank,
        "cuda_ns": 80 + rank,
    }


def _valid_rank_profile(policy: str = "ngram") -> dict:
    rows = []
    for rank in range(4):
        steps = []
        collectives = []
        if policy == "ngram":
            steps = [
                _profile_step(rank, 0, "spec_first_target"),
                _profile_step(rank, 1, "spec_verify"),
            ]
            collectives = [
                _profile_collective(rank, 0),
                _profile_collective(rank, 1),
            ]
        rows.append({
            "rank": rank,
            "enabled": True,
            "finalization_status": "complete",
            "steps": steps,
            "collectives": collectives,
        })
    return {
        "enabled": True,
        "rank_inventory": [0, 1, 2, 3],
        "ranks": rows,
    }


def _residency_phase(operation: str) -> dict:
    status = {
        "prepare": "prepared",
        "precommit": "precommitted",
        "seal": "sealed",
    }[operation]
    return {
        "ticket_id": 1,
        "operation": operation,
        "status": status,
        "rows": [
            {
                "ticket_id": 1,
                "participant_id": rank,
                "operation": operation,
                "status": status,
                "sequence_ids": [7],
                "committed_block_identities": (
                    [] if operation == "prepare" else [[12, 3]]
                ),
                "rejected_block_identities": (
                    [] if operation == "prepare" else [[13, 4]]
                ),
                "detail": "",
            }
            for rank in range(4)
        ],
    }


def _valid_residency_phases() -> list[dict]:
    return [
        _residency_phase("prepare"),
        _residency_phase("precommit"),
        _residency_phase("seal"),
    ]


def _valid_kv_rank_deltas() -> list[dict]:
    return [
        {
            "rank": rank,
            "provenance": "engine.kv_offload_summaries",
            "h2d_copies": 4 + rank,
            "h2d_bytes": 4096 + rank,
            "d2h_copies": 1,
            "d2h_bytes": 1024,
            "copy_waits": 2,
            "evictions": 1,
            "evict_clean": 1,
            "speculative_residency_committed_blocks": 1,
            "speculative_residency_rejected_blocks": 1,
            "speculative_residency_rejected_d2h_copies": 0,
        }
        for rank in range(4)
    ]


def _valid_cleanup_receipt() -> dict:
    return {
        "process_group_destroyed": True,
        "rank_exit_codes": [0, 0, 0, 0],
        "owned_children_remaining": [],
        "rank_cleanup_receipts": [
            {
                "rank": rank,
                "worker_exit_code": 0,
                "process_group_initialized": False,
                "engine_exit_called": True,
                "live_lease_count": 0,
                "prepared_transaction_count": 0,
                "runtime_poisoned": False,
            }
            for rank in range(4)
        ],
    }


def test_validate_rank_profile_requires_all_rank_callbacks():
    normalized = gate.validate_rank_profile(
        _valid_rank_profile(),
        policy="ngram",
    )
    assert normalized["rank_inventory"] == [0, 1, 2, 3]


def test_validate_rank_profile_rejects_missing_collective():
    profile = _valid_rank_profile()
    profile["ranks"][2]["collectives"].pop()
    with pytest.raises(ValueError, match="collective"):
        gate.validate_rank_profile(profile, policy="ngram")


def test_validate_residency_phases_rejects_wrong_order():
    phases = _valid_residency_phases()
    phases[1], phases[2] = phases[2], phases[1]
    with pytest.raises(
        ValueError,
        match="residency phase order",
    ):
        gate.validate_residency_phases(phases)


def test_validate_residency_phases_accepts_production_tuple_receipts():
    phases = _valid_residency_phases()
    for phase in phases:
        for row in phase["rows"]:
            row["sequence_ids"] = tuple(
                row["sequence_ids"]
            )
            row["committed_block_identities"] = tuple(
                tuple(identity)
                for identity in row[
                    "committed_block_identities"
                ]
            )
            row["rejected_block_identities"] = tuple(
                tuple(identity)
                for identity in row[
                    "rejected_block_identities"
                ]
            )

    normalized = gate.validate_residency_phases(phases)

    assert isinstance(
        normalized[0]["rows"][0]["sequence_ids"],
        list,
    )
    assert isinstance(
        normalized[1]["rows"][0][
            "committed_block_identities"
        ],
        list,
    )


def test_validate_kv_rank_deltas_requires_production_provenance():
    rows = _valid_kv_rank_deltas()
    rows[3]["provenance"] = "synthetic_tensor_copy"
    with pytest.raises(
        ValueError,
        match="movement provenance",
    ):
        gate._validate_kv_rank_deltas(rows)


def test_validate_cleanup_rejects_live_rank_state():
    receipt = copy.deepcopy(_valid_cleanup_receipt())
    receipt["rank_cleanup_receipts"][1][
        "live_lease_count"
    ] = 1
    with pytest.raises(
        ValueError,
        match="rank cleanup receipt",
    ):
        gate.validate_cleanup_receipt(receipt)


def _token_row(
    prompt_index: int,
    token_ids: list[int],
) -> dict:
    return {
        "prompt_index": prompt_index,
        "token_count": len(token_ids),
        "token_ids": list(token_ids),
        "sha256": hashlib.sha256(
            json.dumps(
                token_ids,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _prompt_rows(batch_size: int) -> list[dict]:
    return [
        _token_row(
            prompt_index,
            [prompt_index + 1] * gate.CONTEXT_TOKENS,
        )
        for prompt_index in range(batch_size)
    ]


def _output_rows(batch_size: int) -> list[dict]:
    return [
        _token_row(
            prompt_index,
            [
                1000 + prompt_index * 10 + offset
                for offset in range(gate.MAX_OUTPUT_TOKENS)
            ],
        )
        for prompt_index in range(batch_size)
    ]


def _runtime(policy: str, batch_size: int) -> dict:
    if policy == "baseline":
        return {
            "proposal_rows": 0,
            "proposed_tokens": 0,
            "accepted_draft_tokens": 0,
            "rejected_draft_tokens": 0,
            "first_target_callbacks": 0,
            "verify_callbacks": 0,
            "accepted_prefix_replays": 0,
            "consumed_input_mappings": [],
        }
    return {
        "proposal_rows": batch_size,
        "proposed_tokens": batch_size * 4,
        "accepted_draft_tokens": batch_size * 2,
        "rejected_draft_tokens": batch_size * 2,
        "first_target_callbacks": 1,
        "verify_callbacks": 1,
        "accepted_prefix_replays": 0,
        "consumed_input_mappings": [
            {
                "sequence_id": sequence_id,
                "transaction_ordinal": 0,
                "proposal_token_count": 4,
                "accepted_draft_count": 2,
                "verify_input_count": 3,
                "committed_tail_input_count": 2,
                "committed_input_count": 3,
            }
            for sequence_id in range(batch_size)
        ],
    }


def _rank_evidence(
    policy: str,
    batch_size: int,
) -> list[dict]:
    rows = []
    for rank in range(4):
        transactions = []
        receipts = []
        if policy == "ngram":
            for sequence_id in range(batch_size):
                transaction = _sequence_transaction(
                    rank,
                    physical_slot_id=rank * 100 + sequence_id,
                )
                transaction["cell_key"] = (
                    f"{policy}:b{batch_size}"
                )
                transaction["sequence_id"] = sequence_id
                transactions.append(transaction)
                receipts.extend(
                    _side_state_receipts(
                        rank,
                        f"rank-{rank}-sequence-{sequence_id}",
                        sequence_id,
                    )
                )
        rows.append({
            "rank": rank,
            "checkpoint_loaded": True,
            "transactions": transactions,
            "side_state_receipts": receipts,
            "failure_path_rollbacks": [],
        })
    return rows


def _valid_cell(
    policy: str = "ngram",
    batch_size: int = 1,
) -> dict:
    return {
        "schema_version": gate.SCHEMA_VERSION,
        "classification": gate.CLASSIFICATION,
        "policy": policy,
        "context_tokens": gate.CONTEXT_TOKENS,
        "batch_size": batch_size,
        "world_size": gate.WORLD_SIZE,
        "model_identity": {
            "model_type": "qwen3_5",
            "architectures": [
                "Qwen3_5ForConditionalGeneration"
            ],
            "text_layer_count": 24,
            "linear_layer_count": 18,
            "full_attention_layer_count": 6,
        },
        "prompt_rows": _prompt_rows(batch_size),
        "output_rows": _output_rows(batch_size),
        "runtime": _runtime(policy, batch_size),
        "rank_evidence": _rank_evidence(
            policy,
            batch_size,
        ),
        "profile": _valid_rank_profile(policy),
        "kv_rank_deltas": _valid_kv_rank_deltas(),
        "residency_phases": (
            _valid_residency_phases()
            if policy == "ngram"
            else []
        ),
        "cleanup_receipt": _valid_cleanup_receipt(),
    }


def _valid_result() -> dict:
    cells = {
        gate.cell_key(policy, batch_size): _valid_cell(
            policy,
            batch_size,
        )
        for batch_size in gate.BATCH_SIZES
        for policy in gate.POLICIES
    }
    return {
        "schema_version": gate.SCHEMA_VERSION,
        "classification": gate.CLASSIFICATION,
        "claim_scope": gate.CLAIM_SCOPE,
        "limitations": list(gate.LIMITATIONS),
        "source_tree_sha256": "a" * 64,
        "model_manifest_sha256": (
            gate.MODEL_MANIFEST_SHA256
        ),
        "world_size": gate.WORLD_SIZE,
        "gpu_indices": [0, 1, 2, 3],
        "cells": cells,
        "parity": {"b1": True, "b4": True},
    }


def test_validate_result_accepts_complete_authority():
    normalized = gate.validate_result(_valid_result())
    assert normalized["classification"] == (
        "SECOND_MODEL_TP4_4K_ESTABLISHED"
    )


def test_validate_result_rejects_output_parity_mismatch():
    result = _valid_result()
    result["cells"]["ngram:b4"]["output_rows"][0][
        "token_ids"
    ][0] += 1
    result["cells"]["ngram:b4"]["output_rows"][0][
        "sha256"
    ] = gate._json_sha256(
        result["cells"]["ngram:b4"]["output_rows"][0][
            "token_ids"
        ]
    )
    with pytest.raises(
        ValueError,
        match="output parity mismatch",
    ):
        gate.validate_result(result)


def test_validate_cell_rejects_zero_rejected_drafts():
    cell = _valid_cell()
    cell["runtime"]["rejected_draft_tokens"] = 0
    with pytest.raises(
        ValueError,
        match="rejected draft tokens must be positive",
    ):
        gate.validate_cell_result(cell)


def test_validate_cell_rejects_cross_rank_transaction_mismatch():
    cell = _valid_cell()
    cell["rank_evidence"][3]["transactions"][0][
        "selected_checkpoint_id"
    ] = "checkpoint:tampered"
    with pytest.raises(
        ValueError,
        match="cross-rank transaction digest mismatch",
    ):
        gate.validate_cell_result(cell)


def test_validate_result_rejects_broader_classification():
    result = _valid_result()
    result["classification"] = "PHASE1_COMPLETE"
    with pytest.raises(
        ValueError,
        match="result classification mismatch",
    ):
        gate.validate_result(result)


def test_validate_cell_accepts_multiple_transactions_per_sequence():
    cell = _valid_cell(policy="ngram", batch_size=1)
    second_mapping = {
        "sequence_id": 0,
        "transaction_ordinal": 1,
        "proposal_token_count": 4,
        "accepted_draft_count": 2,
        "verify_input_count": 3,
        "committed_tail_input_count": 2,
        "committed_input_count": 3,
    }
    cell["runtime"]["proposal_rows"] = 2
    cell["runtime"]["proposed_tokens"] = 8
    cell["runtime"]["accepted_draft_tokens"] = 4
    cell["runtime"]["rejected_draft_tokens"] = 4
    cell["runtime"]["consumed_input_mappings"].append(
        second_mapping
    )
    for rank_row in cell["rank_evidence"]:
        rank = rank_row["rank"]
        transaction = _sequence_transaction(
            rank,
            physical_slot_id=rank * 100 + 1,
        )
        transaction["cell_key"] = "ngram:b1"
        transaction["sequence_id"] = 0
        transaction["transaction_ordinal"] = 1
        transaction["proposal_token_ids"] = [21, 22, 23, 24]
        rank_row["transactions"].append(transaction)
        rank_row["side_state_receipts"].extend(
            _side_state_receipts(
                rank,
                f"rank-{rank}-sequence-0-transaction-1",
                0,
            )
        )

    normalized = gate.validate_cell_result(cell)

    assert len(
        normalized["runtime"]["consumed_input_mappings"]
    ) == 2
    assert all(
        len(row["transactions"]) == 2
        for row in normalized["rank_evidence"]
    )


def _load_worker():
    return _load_module(
        "qwen35_generic_speculative_tp4_worker",
        TOOLS / "qwen35_generic_speculative_tp4_worker.py",
    )


def _raw_side_state_receipt(
    operation: str,
    status: str,
    sequence_ids: tuple[int, ...] = (0,),
) -> dict:
    receipt = {
        "operation": operation,
        "status": status,
        "transaction_id": "transaction-1",
        "sequence_ids": list(sequence_ids),
    }
    if operation == "select":
        receipt["rows"] = [
            {
                "sequence_id": sequence_id,
                "committed_input_count": 3,
                "checkpoint_index": 3,
            }
            for sequence_id in sequence_ids
        ]
    return receipt


def test_worker_normalizes_ranked_side_state_receipts():
    worker = _load_worker()
    normalized = worker.normalize_side_state_receipts(
        [
            _raw_side_state_receipt(
                "prepare",
                "prepared",
                (0, 1),
            )
        ],
        rank=2,
    )
    assert normalized == [
        {
            "rank": 2,
            "sequence_id": 0,
            "handle_id": "transaction-1",
            "operation": "prepare",
            "state": "prepared",
        },
        {
            "rank": 2,
            "sequence_id": 1,
            "handle_id": "transaction-1",
            "operation": "prepare",
            "state": "prepared",
        },
    ]


def test_worker_capture_acknowledges_all_side_state_ranks():
    worker = _load_worker()

    class FakeModelRunner:
        def __init__(self):
            self.call = lambda method_name, *args: {
                "unhandled": method_name,
            }

    class FakeEngine:
        def __init__(self):
            self.model_runner = FakeModelRunner()

        def call_model_runner_acknowledged(
            self,
            method_name,
            *args,
            timeout_s,
        ):
            operation = method_name.split("_", 1)[0]
            local = _raw_side_state_receipt(
                operation,
                f"{operation}d",
            )
            workers = tuple(
                SimpleNamespace(
                    rank=rank,
                    result=dict(local),
                )
                for rank in (1, 2, 3)
            )
            return local, workers

    engine = FakeEngine()
    with worker.capture_rank_side_state_receipts(
        engine
    ) as ranked:
        result = engine.model_runner.call(
            "prepare_speculative_side_state_batch",
            ("sequence",),
        )
    assert result["operation"] == "prepare"
    assert sorted(ranked) == [0, 1, 2, 3]
    assert all(
        rows[0]["operation"] == "prepare"
        for rows in ranked.values()
    )


def test_worker_summarizes_mappings_and_requires_all_ranks():
    worker = _load_worker()
    observations = [{
        "speculative_proposal_row_count": 1,
        "speculative_first_target_callback_count": 1,
        "speculative_fixed_q_group_count": 1,
        "speculative_proposal_token_counts": {
            0: 4,
        },
        "speculative_accepted_draft_token_counts": {
            0: 2,
        },
        "speculative_proposal_token_ids_by_seq": {
            0: [11, 12, 13, 14],
        },
        "speculative_accepted_draft_token_ids_by_seq": {
            0: [11, 12],
        },
    }]
    rank_receipts = {
        rank: [
            _raw_side_state_receipt(
                operation,
                status,
            )
            for operation, status in (
                ("prepare", "prepared"),
                ("select", "selected"),
                ("apply", "applied"),
                ("seal", "sealed"),
            )
        ]
        for rank in range(4)
    }
    summary = worker.summarize_step_observations(
        observations,
        rank_receipts=rank_receipts,
        policy="ngram",
        batch_size=1,
    )
    assert summary["runtime"]["proposed_tokens"] == 4
    assert summary["runtime"][
        "accepted_draft_tokens"
    ] == 2
    assert summary["runtime"][
        "rejected_draft_tokens"
    ] == 2
    assert {
        row["committed_input_count"]
        for row in summary["runtime"][
            "consumed_input_mappings"
        ]
    } == {3}
    assert {
        row["rank"]
        for row in summary["rank_side_state_receipts"]
    } == {0, 1, 2, 3}
    assert all(
        len(row["transactions"]) == 1
        for row in summary["rank_evidence"]
    )
    assert {
        row["transactions"][0]["semantic_digest"]
        for row in summary["rank_evidence"]
    } == {
        summary["rank_evidence"][0][
            "transactions"
        ][0]["semantic_digest"]
    }

    del rank_receipts[3]
    with pytest.raises(
        ValueError,
        match="rank receipt inventory mismatch",
    ):
        worker.summarize_step_observations(
            observations,
            rank_receipts=rank_receipts,
            policy="ngram",
            batch_size=1,
        )


def test_model_runner_reports_speculative_cleanup_observation():
    method = _load_model_runner_method(
        "speculative_cleanup_observation"
    )
    runner = SimpleNamespace(
        rank=2,
        qwen35_speculative_state_owner=SimpleNamespace(
            active=True,
        ),
        _speculative_side_state_leases_by_sequence={
            7: object(),
            9: object(),
        },
    )

    assert method(runner) == {
        "rank": 2,
        "active_transaction_count": 1,
        "live_lease_count": 2,
    }


class _FakeTokenizer:
    name_or_path = "qwen35-fixture"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [1, 2, 3, 4]


class _FakeModelRunner:
    def __init__(self):
        self.world_size = 4
        self.call = lambda method_name, *args: {
            "unhandled": method_name,
        }


class _FakeTP4Engine:
    def __init__(self, batch_size: int):
        layer_types = (
            ["linear_attention"] * 3
            + ["full_attention"]
        ) * 6
        self.config = SimpleNamespace(
            dtype="torch.float16",
            hf_config=SimpleNamespace(
                model_type="qwen3_5",
                architectures=[
                    "Qwen3_5ForConditionalGeneration"
                ],
                text_config=SimpleNamespace(
                    layer_types=layer_types,
                ),
            ),
        )
        self.tokenizer = _FakeTokenizer()
        self.model_runner = _FakeModelRunner()
        self.batch_size = batch_size
        self.speculative_runtime_poisoned = False
        self.activated_runtime = None
        self._movement_call_count = 0
        self._call_speculative_residency_phase = (
            self._residency_phase
        )

    def activate_speculative_runtime(self, runtime):
        self.activated_runtime = runtime

    def configure_decode_internal_profile(
        self,
        enabled,
        label,
        *,
        timeout_s,
    ):
        assert enabled is True
        assert timeout_s == 60.0
        assert label
        return {"rank_inventory": [0, 1, 2, 3]}

    def clear_reusable_prefix_cache(self):
        return None

    def kv_offload_summaries(self, *, timeout_s):
        assert timeout_s == 60.0
        offset = self._movement_call_count
        self._movement_call_count += 1
        return tuple(
            {
                name: (
                    0
                    if name
                    == "speculative_residency_rejected_d2h_copies"
                    else offset
                )
                for name in gate.MOVEMENT_KEYS
            }
            for _ in range(4)
        )

    def finalize_decode_internal_profile(
        self,
        *,
        timeout_s,
    ):
        assert timeout_s == 60.0
        return _valid_rank_profile("ngram")

    def flush_pending_hybrid_state_releases(
        self,
        *,
        timeout_s,
    ):
        assert timeout_s == 60.0
        return ()

    def _residency_phase(
        self,
        method_name,
        ticket_id,
        *args,
        **kwargs,
    ):
        operation = kwargs["expected_operation"]
        status = kwargs["expected_status"]
        return tuple(
            {
                "ticket_id": ticket_id,
                "participant_id": rank,
                "operation": operation,
                "status": status,
                "sequence_ids": list(
                    kwargs["expected_sequence_ids"]
                ),
                "committed_block_identities": [
                    list(row)
                    for row in kwargs[
                        "expected_committed_block_identities"
                    ]
                ],
                "rejected_block_identities": [
                    list(row)
                    for row in kwargs[
                        "expected_rejected_block_identities"
                    ]
                ],
                "detail": "",
            }
            for rank in range(4)
        )

    def call_model_runner_acknowledged(
        self,
        method_name,
        *args,
        timeout_s,
    ):
        assert timeout_s == 60.0
        if method_name == "speculative_cleanup_observation":
            rows = [
                {
                    "rank": rank,
                    "active_transaction_count": 0,
                    "live_lease_count": 0,
                }
                for rank in range(4)
            ]
        else:
            operation = method_name.split("_", 1)[0]
            rows = [
                _raw_side_state_receipt(
                    operation,
                    {
                        "prepare": "prepared",
                        "select": "selected",
                        "apply": "applied",
                        "seal": "sealed",
                    }[operation],
                )
                for _ in range(4)
            ]
        return (
            rows[0],
            tuple(
                SimpleNamespace(
                    rank=rank,
                    result=rows[rank],
                )
                for rank in (1, 2, 3)
            ),
        )

    def exit(self):
        return {
            "process_group_destroyed": True,
            "rank_exit_codes": [0, 0, 0, 0],
            "owned_children_remaining": [],
            "rank_cleanup_receipts": [
                {
                    "rank": rank,
                    "process_group_destroyed": True,
                }
                for rank in range(4)
            ],
        }


def _fake_generation_runner(
    *,
    engine,
    prompt_rows,
    sampling_params,
    expected_output_tokens,
    synchronize,
):
    del sampling_params
    assert expected_output_tokens == gate.MAX_OUTPUT_TOKENS
    synchronize()
    if engine.activated_runtime is not None:
        for method_name in (
            "prepare_speculative_side_state_batch",
            "select_speculative_side_state_batch",
            "apply_speculative_side_state_batch",
            "seal_speculative_side_state_batch",
        ):
            engine.model_runner.call(method_name)
        for operation, status in (
            ("prepare", "prepared"),
            ("precommit", "precommitted"),
            ("seal", "sealed"),
        ):
            engine._call_speculative_residency_phase(
                f"{operation}_speculative_residency_batch",
                1,
                expected_operation=operation,
                expected_status=status,
                expected_sequence_ids=(0,),
                expected_committed_block_identities=(
                    ()
                    if operation == "prepare"
                    else ((12, 3),)
                ),
                expected_rejected_block_identities=(
                    ()
                    if operation == "prepare"
                    else ((13, 4),)
                ),
            )
        observations = [{
            "speculative_proposal_row_count": 1,
            "speculative_first_target_callback_count": 1,
            "speculative_fixed_q_group_count": 1,
            "speculative_proposal_token_counts": {0: 4},
            "speculative_accepted_draft_token_counts": {
                0: 2
            },
            "speculative_proposal_token_ids_by_seq": {
                0: [11, 12, 13, 14]
            },
            "speculative_accepted_draft_token_ids_by_seq": {
                0: [11, 12]
            },
        }]
    else:
        observations = [{
            "speculative_proposal_row_count": 0,
            "speculative_first_target_callback_count": 0,
            "speculative_fixed_q_group_count": 0,
            "speculative_proposal_token_counts": {},
            "speculative_accepted_draft_token_counts": {},
            "speculative_proposal_token_ids_by_seq": {},
            "speculative_accepted_draft_token_ids_by_seq": {},
        }]
    return {
        "output_rows": [
            _token_row(
                index,
                list(range(gate.MAX_OUTPUT_TOKENS)),
            )
            for index, _ in enumerate(prompt_rows)
        ],
        "observations": observations,
    }


def test_worker_builds_complete_tp4_candidate_cell():
    worker = _load_worker()
    factory_calls = []

    def engine_factory(model_path, **kwargs):
        factory_calls.append((model_path, dict(kwargs)))
        return _FakeTP4Engine(kwargs["max_num_seqs"])

    class Runtime:
        def __init__(self, adapter):
            self.adapter = adapter

    class Adapter:
        def __init__(
            self,
            *,
            ngram_size,
            max_proposal_tokens,
        ):
            self.ngram_size = ngram_size
            self.max_proposal_tokens = max_proposal_tokens

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    cell = worker.run_policy_cell(
        model_path="/model",
        gpu_indices=(0, 1, 2, 3),
        policy="ngram",
        batch_size=1,
        dist_port=31001,
        master_port=32001,
        engine_factory=engine_factory,
        sampling_params_type=SamplingParams,
        runtime_type=Runtime,
        adapter_type=Adapter,
        synchronize=lambda: None,
        run_generation_fn=_fake_generation_runner,
    )

    assert factory_calls[0][1][
        "tensor_parallel_size"
    ] == 4
    assert cell["runtime"]["accepted_prefix_replays"] == 0
    assert len(cell["rank_evidence"]) == 4
    assert all(
        row["transactions"]
        for row in cell["rank_evidence"]
    )
    assert cell["cleanup_receipt"][
        "process_group_destroyed"
    ] is True


def test_worker_source_contract_uses_real_tp4_dependencies_without_replay():
    source = (
        Path(__file__).resolve().parent
        / "qwen35_generic_speculative_tp4_worker.py"
    ).read_text(encoding="utf-8")

    assert "from tinyvllm import LLM" in source
    assert (
        "from tinyvllm.engine.speculative_runtime import "
        "EngineSpeculativeRuntime"
    ) in source
    assert (
        "from tinyvllm.speculative.ngram_adapter import "
        "NGramDraftAdapter"
    ) in source
    assert (
        "from tinyvllm.sampling_params import SamplingParams"
        in source
    )
    assert "tensor_parallel_size=gate.WORLD_SIZE" in source
    assert "enforce_eager=True" in source
    assert "replay_accepted_prefix" not in source
    run_generation_source = source[
        source.index("def run_generation("):
        source.index("\ndef _movement_delta(")
    ]
    assert run_generation_source.count("engine.step()") == 1


def _fake_campaign_subprocess(calls):
    def run(command, **kwargs):
        policy = command[command.index("--policy") + 1]
        batch_size = int(
            command[command.index("--batch-size") + 1]
        )
        output_path = Path(
            command[command.index("--out") + 1]
        )
        calls.append({
            "cell": gate.cell_key(policy, batch_size),
            "dist_port": int(
                command[
                    command.index("--dist-port") + 1
                ]
            ),
            "master_port": int(
                command[
                    command.index("--master-port") + 1
                ]
            ),
            "output_path": output_path,
            "kwargs": kwargs,
        })
        gate.atomic_write_json(
            output_path,
            _valid_cell(policy, batch_size),
        )
        return SimpleNamespace(returncode=0)

    return run


def test_campaign_runs_four_fresh_cells_and_publishes_after_verification(
    tmp_path,
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        _fake_campaign_subprocess(calls),
    )
    monkeypatch.setattr(
        gate,
        "model_manifest_sha256",
        lambda model_path: gate.MODEL_MANIFEST_SHA256,
    )
    verification_calls = []

    def verify(run_dir, source_root):
        verification_calls.append(
            (Path(run_dir), Path(source_root))
        )
        assert not (tmp_path / "authority").exists()
        assert not (Path(run_dir) / "verify.json").exists()
        return {"classification": "PASS", "failures": []}

    result = gate.run_campaign(
        model_path=str(tmp_path / "model"),
        gpu_indices=(0, 1, 2, 3),
        output_dir=tmp_path / "authority",
        dist_port_base=31000,
        master_port_base=32000,
        repo_root=TOOLS.parent,
        source_files=(
            "tools/qwen35_generic_speculative_tp4_gate.py",
        ),
        verifier=verify,
    )

    assert [call["cell"] for call in calls] == [
        "baseline:b1",
        "ngram:b1",
        "baseline:b4",
        "ngram:b4",
    ]
    assert {
        call["dist_port"] for call in calls
    } == {31000, 31001, 31002, 31003}
    assert {
        call["master_port"] for call in calls
    } == {32000, 32001, 32002, 32003}
    assert len({
        call["output_path"].parent.parent
        for call in calls
    }) == 1
    assert verification_calls
    assert result["classification"] == gate.CLASSIFICATION
    assert (
        tmp_path / "authority" / "verify.json"
    ).is_file()

    with pytest.raises(
        ValueError,
        match="already exists",
    ):
        gate.run_campaign(
            model_path=str(tmp_path / "model"),
            gpu_indices=(0, 1, 2, 3),
            output_dir=tmp_path / "authority",
            dist_port_base=33000,
            master_port_base=34000,
            repo_root=TOOLS.parent,
            source_files=(
                "tools/qwen35_generic_speculative_tp4_gate.py",
            ),
            verifier=verify,
        )


def test_campaign_preserves_failed_verification_artifacts(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        _fake_campaign_subprocess([]),
    )
    monkeypatch.setattr(
        gate,
        "model_manifest_sha256",
        lambda model_path: gate.MODEL_MANIFEST_SHA256,
    )

    with pytest.raises(
        RuntimeError,
        match="sentinel verification failure",
    ):
        gate.run_campaign(
            model_path=str(tmp_path / "model"),
            gpu_indices=(0, 1, 2, 3),
            output_dir=tmp_path / "authority",
            dist_port_base=35000,
            master_port_base=36000,
            repo_root=TOOLS.parent,
            source_files=(
                "tools/qwen35_generic_speculative_tp4_gate.py",
            ),
            verifier=lambda run_dir, source_root: {
                "classification": "FAIL",
                "failures": [
                    "sentinel verification failure"
                ],
            },
        )

    assert not (tmp_path / "authority").exists()
    assert (
        tmp_path / "authority.failed" / "verify.json"
    ).is_file()


def _load_verifier():
    return _load_module(
        "verify_qwen35_generic_speculative_tp4_gate_test",
        TOOLS
        / "verify_qwen35_generic_speculative_tp4_gate.py",
    )


def _write_valid_verifier_run(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "bound.py"
    source_path.write_text("BOUND = True\n", encoding="utf-8")
    source_files = ("bound.py",)
    result = _valid_result()
    result["source_tree_sha256"] = gate.source_tree_sha256(
        source_root,
        source_files,
    )
    run_dir = tmp_path / "authority"
    run_dir.mkdir()
    gate.atomic_write_json(run_dir / "result.json", result)
    gate.atomic_write_json(
        run_dir / "source_manifest.json",
        {
            "schema_version": gate.SCHEMA_VERSION,
            "source_tree_sha256": result[
                "source_tree_sha256"
            ],
            "model_manifest_sha256": result[
                "model_manifest_sha256"
            ],
            "source_files": gate.hash_source_files(
                source_root,
                source_files,
            ),
            "artifacts": {
                "result.json": gate.sha256_file(
                    run_dir / "result.json"
                ),
            },
        },
    )
    return run_dir, source_root


def _rewrite_result_and_manifest(run_dir, result):
    gate.atomic_write_json(run_dir / "result.json", result)
    manifest_path = run_dir / "source_manifest.json"
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest["artifacts"]["result.json"] = gate.sha256_file(
        run_dir / "result.json"
    )
    gate.atomic_write_json(manifest_path, manifest)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda result: result["cells"]["ngram:b1"][
                "rank_evidence"
            ][3]["transactions"][0].update(
                selected_checkpoint_id="checkpoint:tampered"
            ),
            "cross-rank transaction digest mismatch",
        ),
        (
            lambda result: result["cells"]["ngram:b1"][
                "rank_evidence"
            ][2]["side_state_receipts"][0].update(
                state="tampered"
            ),
            "side-state lifecycle state mismatch",
        ),
        (
            lambda result: result["cells"]["ngram:b1"][
                "kv_rank_deltas"
            ][1].update(provenance="synthetic"),
            "KV movement provenance is invalid",
        ),
        (
            lambda result: result["cells"]["ngram:b1"][
                "cleanup_receipt"
            ]["rank_cleanup_receipts"][0].update(
                live_lease_count=1
            ),
            "cleanup",
        ),
    ],
)
def test_verifier_rejects_semantic_tampering(
    tmp_path,
    mutate,
    match,
):
    run_dir, _ = _write_valid_verifier_run(tmp_path)
    result = json.loads(
        (run_dir / "result.json").read_text(
            encoding="utf-8"
        )
    )
    mutate(result)
    _rewrite_result_and_manifest(run_dir, result)

    verification = _load_verifier().verify_run(run_dir)

    assert verification["classification"] == "FAIL"
    assert match in verification["failures"][0]


def test_verifier_rejects_result_hash_tampering(tmp_path):
    run_dir, _ = _write_valid_verifier_run(tmp_path)
    result = json.loads(
        (run_dir / "result.json").read_text(
            encoding="utf-8"
        )
    )
    result["unbound_field"] = True
    gate.atomic_write_json(run_dir / "result.json", result)

    verification = _load_verifier().verify_run(run_dir)

    assert verification["classification"] == "FAIL"
    assert "result artifact SHA-256 mismatch" in (
        verification["failures"][0]
    )


def test_verifier_rejects_bound_source_tampering(tmp_path):
    run_dir, source_root = _write_valid_verifier_run(tmp_path)
    (source_root / "bound.py").write_text(
        "BOUND = False\n",
        encoding="utf-8",
    )

    verification = _load_verifier().verify_run(
        run_dir,
        source_root,
    )

    assert verification["classification"] == "FAIL"
    assert "current source file identity mismatch" in (
        verification["failures"][0]
    )


def test_verifier_rejects_approved_model_manifest_tampering(
    tmp_path,
):
    run_dir, _ = _write_valid_verifier_run(tmp_path)
    manifest_path = run_dir / "source_manifest.json"
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest["model_manifest_sha256"] = "0" * 64
    gate.atomic_write_json(manifest_path, manifest)

    verification = _load_verifier().verify_run(run_dir)

    assert verification["classification"] == "FAIL"
    assert "model manifest identity mismatch" in (
        verification["failures"][0]
    )


def test_remote_runner_source_contract():
    runner_path = (
        TOOLS
        / "run_qwen35_generic_speculative_tp4_gate_remote.sh"
    )
    text = runner_path.read_text(encoding="utf-8")

    for required in (
        "sitian@10.232.195.203",
        "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python",
        "/data00/home/sitian/sitian-workspace01/tllm/"
        "qwen35-hybrid-state-runs/"
        "qwen35-2b-hybrid-acquire-20260723-222004/model",
        "FILE:/Users/bytedance/krb5cc_sitian",
        "ControlMaster=no",
        "ControlPath=none",
        "campaign.status",
        "campaign.pid",
        "campaign.exit_code",
        "selected_gpu_indices.txt",
        "selected_ports.txt",
        "--gpu-indices",
        "--dist-port-base",
        "--master-port-base",
        "verify_qwen35_generic_speculative_tp4_gate.py",
        "REMOTE_COMMAND_RETRY_ATTEMPTS",
        "POLL_INTERVAL_SECONDS",
    ):
        assert required in text
    assert "head -n 4" in text
    assert "campaign already terminal" in text
    assert "campaign already running" in text
    assert "authority.failed" in text
    assert (
        "attempt <= REMOTE_COMMAND_RETRY_ATTEMPTS"
        in text
    )
    assert (
        "attempt <= REMOTE_RSYNC_RETRY_ATTEMPTS"
        in text
    )
    assert "while" in text
    assert "rsync -a" in text
