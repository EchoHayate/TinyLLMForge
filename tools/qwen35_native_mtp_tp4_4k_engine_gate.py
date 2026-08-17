from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


SCHEMA_VERSION = (
    "qwen35.native-mtp-tp4-4k-engine-"
    "transactional-correctness.v1"
)
CLASSIFICATION = (
    "QWEN35_NATIVE_MTP_TP4_4K_ENGINE_ESTABLISHED"
)
PROMOTION_CLASSIFICATION = "NOT_PROMOTABLE"
POLICIES = ("baseline", "native_mtp")
BATCH_SIZES = (1, 4)
PROMPT_TOKENS = 4096
MAX_OUTPUT_TOKENS = 32
MAX_PROPOSAL_TOKENS = 4
WORLD_SIZE = 4
RANKS = (0, 1, 2, 3)
WORKER_RANKS = (1, 2, 3)
TARGET_MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38fe"
    "69982efedaf4a3e631ad0b14aad7dd0"
)
MTP_CHECKPOINT_MANIFEST_SHA256 = (
    "9a975bdcf0383774183cae560594dd60"
    "b522b83fe9c4cd595c47c12e2403702b"
)
TP1_AUTHORITY_SHA256 = (
    "f267e49281cc12e64c176fc2294f594e7"
    "b2118897092708ce1ece3bd3b9ee9ac"
)
REQUIRED_LIMITATIONS = (
    "TP4 only",
    "4K prompt only",
    "KV offload disabled",
    "eager native MTP only",
    "not production ready",
)

_TOOLS = Path(__file__).resolve().parent
_ROOT = _TOOLS.parent
_WORKER = _TOOLS / "qwen35_native_mtp_tp4_4k_engine_worker.py"
_TP1_GATE = _TOOLS / "qwen35_native_mtp_tp1_4k_engine_gate.py"
_TP1_WORKER = (
    _TOOLS / "qwen35_native_mtp_tp1_4k_engine_worker.py"
)
_VERIFIER = (
    _TOOLS / "verify_qwen35_native_mtp_tp4_4k_engine_gate.py"
)
DEFAULT_SOURCE_FILES = tuple(sorted(
    [
        str(path.relative_to(_ROOT))
        for path in (_ROOT / "tinyvllm").rglob("*.py")
    ]
    + [
        str(Path(__file__).resolve().relative_to(_ROOT)),
        str(_WORKER.relative_to(_ROOT)),
        str(_TP1_GATE.relative_to(_ROOT)),
        str(_TP1_WORKER.relative_to(_ROOT)),
        str(_VERIFIER.relative_to(_ROOT)),
    ]
))


def cell_key(policy: str, batch_size: int) -> str:
    if policy not in POLICIES:
        raise ValueError("cell policy is invalid")
    if batch_size not in BATCH_SIZES:
        raise ValueError("cell batch size is invalid")
    return f"{policy}:b{batch_size}"


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    value = _integer(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} SHA-256 is invalid")
    return value


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _exact_keys(
    value: object,
    expected: set[str],
    name: str,
) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} fields mismatch")
    return value


def _validate_token_rows(
    value: object,
    *,
    batch_size: int,
    token_count: int,
    name: str,
) -> list[dict]:
    if not isinstance(value, list) or len(value) != batch_size:
        raise ValueError(f"{name} row inventory mismatch")
    rows = []
    for prompt_index, row in enumerate(value):
        _exact_keys(
            row,
            {"prompt_index", "token_count", "token_ids", "sha256"},
            f"{name} row",
        )
        if row["prompt_index"] != prompt_index:
            raise ValueError(f"{name} prompt index mismatch")
        if row["token_count"] != token_count:
            raise ValueError(f"{name} token count mismatch")
        token_ids = row["token_ids"]
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != token_count
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in token_ids
            )
        ):
            raise ValueError(f"{name} token rows are invalid")
        digest = _sha256(row["sha256"], f"{name} row")
        if digest != _json_sha256(token_ids):
            raise ValueError(f"{name} token digest mismatch")
        rows.append({
            "prompt_index": prompt_index,
            "token_count": token_count,
            "token_ids": list(token_ids),
            "sha256": digest,
        })
    return rows


def _validate_model_identity(value: object) -> dict:
    _exact_keys(
        value,
        {
            "model_type",
            "architectures",
            "target_model_manifest_sha256",
            "mtp_checkpoint_manifest_sha256",
        },
        "model identity",
    )
    if value["model_type"] != "qwen3_5":
        raise ValueError("model type mismatch")
    architectures = ["Qwen3_5ForConditionalGeneration"]
    if value["architectures"] != architectures:
        raise ValueError("model architecture mismatch")
    target = _sha256(
        value["target_model_manifest_sha256"],
        "target model manifest",
    )
    mtp = _sha256(
        value["mtp_checkpoint_manifest_sha256"],
        "MTP checkpoint manifest",
    )
    if target != TARGET_MODEL_MANIFEST_SHA256:
        raise ValueError("target model manifest does not match authority")
    if mtp != MTP_CHECKPOINT_MANIFEST_SHA256:
        raise ValueError("MTP checkpoint manifest does not match authority")
    return {
        "model_type": "qwen3_5",
        "architectures": architectures,
        "target_model_manifest_sha256": target,
        "mtp_checkpoint_manifest_sha256": mtp,
    }


def _validate_receipts(
    value: object,
    *,
    batch_size: int,
    operations: list[str],
    name: str,
) -> list[dict]:
    if not isinstance(value, list) or len(value) != batch_size:
        raise ValueError(f"{name} receipt inventory mismatch")
    rows = []
    for sequence_id, row in enumerate(value):
        _exact_keys(
            row,
            {"sequence_id", "operations"},
            f"{name} receipt",
        )
        if (
            row["sequence_id"] != sequence_id
            or row["operations"] != operations
        ):
            raise ValueError(f"{name} lifecycle mismatch")
        rows.append({
            "sequence_id": sequence_id,
            "operations": list(operations),
        })
    return rows


def _validate_transaction_rows(
    value: object,
    *,
    batch_size: int,
    expected_count: int,
) -> list[dict]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError("transaction inventory mismatch")
    rows = []
    for row in value:
        _exact_keys(
            row,
            {
                "sequence_id",
                "sequence_epoch",
                "transaction_id",
                "exact_q",
                "token_ids",
                "staged_entry_count",
                "accepted_proposal_tokens",
                "rejected_proposal_tokens",
                "finalize_ticket_id",
                "state",
            },
            "proposal transaction",
        )
        token_ids = row["token_ids"]
        if (
            isinstance(row["sequence_id"], bool)
            or not isinstance(row["sequence_id"], int)
            or row["sequence_id"] < 0
            or row["sequence_id"] >= batch_size
            or row["sequence_epoch"] != 0
            or isinstance(row["exact_q"], bool)
            or not isinstance(row["exact_q"], int)
            or row["exact_q"] <= 0
            or row["exact_q"] > MAX_PROPOSAL_TOKENS
            or not isinstance(token_ids, list)
            or len(token_ids) != row["exact_q"]
            or row["staged_entry_count"] != row["exact_q"] - 1
            or row["accepted_proposal_tokens"] < 0
            or row["rejected_proposal_tokens"] < 0
            or (
                row["accepted_proposal_tokens"]
                + row["rejected_proposal_tokens"]
                != row["exact_q"]
            )
            or not isinstance(row["transaction_id"], str)
            or not row["transaction_id"]
            or not isinstance(row["finalize_ticket_id"], str)
            or not row["finalize_ticket_id"]
            or row["state"] != "committed"
        ):
            raise ValueError("proposal transaction is invalid")
        rows.append({
            "sequence_id": row["sequence_id"],
            "sequence_epoch": 0,
            "transaction_id": row["transaction_id"],
            "exact_q": row["exact_q"],
            "token_ids": list(token_ids),
            "staged_entry_count": row["exact_q"] - 1,
            "accepted_proposal_tokens": row[
                "accepted_proposal_tokens"
            ],
            "rejected_proposal_tokens": row[
                "rejected_proposal_tokens"
            ],
            "finalize_ticket_id": row["finalize_ticket_id"],
            "state": "committed",
        })
    return rows


def _validate_selected_tokens(
    value: object,
    *,
    transactions: list[dict],
) -> list[dict]:
    expected = []
    for transaction in transactions:
        for step, token_id in enumerate(
            transaction["token_ids"][1:]
        ):
            expected.append({
                "sequence_id": transaction["sequence_id"],
                "transaction_id": transaction["transaction_id"],
                "step": step,
                "token_id": token_id,
            })
    if value != expected:
        raise ValueError("selected token rows mismatch")
    return [dict(row) for row in expected]


def _validate_cache_snapshot(
    value: object,
    *,
    transactions: list[dict],
    batch_size: int,
) -> dict:
    _exact_keys(
        value,
        {
            "active_sequence_count",
            "active_transaction_count",
            "prepared_ticket_count",
            "owned_slot_count",
            "transactions",
            "tickets",
        },
        "proposal KV cache",
    )
    for field, message in (
        ("active_sequence_count", "sequence leak"),
        ("active_transaction_count", "transaction leak"),
        ("prepared_ticket_count", "ticket leak"),
        ("owned_slot_count", "slot leak"),
    ):
        if value[field] != 0:
            raise ValueError(message)
    cache_transactions = value["transactions"]
    expected_count = len(transactions) + batch_size
    if (
        not isinstance(cache_transactions, list)
        or len(cache_transactions) != expected_count
    ):
        raise ValueError("proposal KV transaction inventory mismatch")
    proposals_by_id = {
        row["transaction_id"]: row
        for row in transactions
    }
    if len(proposals_by_id) != len(transactions):
        raise ValueError("proposal KV transaction inventory mismatch")
    bootstrap_sequence_ids = set()
    proposal_transaction_ids = set()
    committed_lengths = {
        sequence_id: 0
        for sequence_id in range(batch_size)
    }
    normalized_transactions = []
    transaction_expectations = {}
    for row in cache_transactions:
        _exact_keys(
            row,
            {
                "transaction_id",
                "sequence_id",
                "sequence_epoch",
                "original_committed_length",
                "staged_entry_count",
                "materialized_entry_count",
                "state",
            },
            "proposal KV transaction",
        )
        transaction_id = row["transaction_id"]
        sequence_id = row["sequence_id"]
        expected = proposals_by_id.get(transaction_id)
        if expected is None:
            if (
                sequence_id not in committed_lengths
                or sequence_id in bootstrap_sequence_ids
                or row["sequence_epoch"] != 0
                or row["original_committed_length"] != 0
                or row["staged_entry_count"] != PROMPT_TOKENS
                or row["materialized_entry_count"] != PROMPT_TOKENS
                or row["state"] != "committed"
            ):
                raise ValueError(
                    "proposal KV bootstrap transaction mismatch"
                )
            bootstrap_sequence_ids.add(sequence_id)
            committed_lengths[sequence_id] = PROMPT_TOKENS
            commit_entry_count = PROMPT_TOKENS
            release_entry_count = 0
        else:
            if (
                transaction_id in proposal_transaction_ids
                or sequence_id != expected["sequence_id"]
                or row["sequence_epoch"] != 0
                or row["original_committed_length"]
                != committed_lengths.get(sequence_id)
                or row["staged_entry_count"]
                != expected["staged_entry_count"]
                or row["materialized_entry_count"]
                != expected["staged_entry_count"]
                or row["state"] != "committed"
            ):
                raise ValueError("proposal KV transaction mismatch")
            proposal_transaction_ids.add(transaction_id)
            commit_entry_count = max(
                expected["accepted_proposal_tokens"] - 1,
                0,
            )
            release_entry_count = (
                expected["staged_entry_count"]
                - commit_entry_count
            )
            committed_lengths[sequence_id] += commit_entry_count
        if (
            not isinstance(transaction_id, str)
            or not transaction_id
        ):
            raise ValueError("proposal KV transaction mismatch")
        transaction_expectations[transaction_id] = (
            commit_entry_count,
            release_entry_count,
        )
        normalized_transactions.append(dict(row))
    if (
        bootstrap_sequence_ids != set(range(batch_size))
        or proposal_transaction_ids != set(proposals_by_id)
    ):
        raise ValueError("proposal KV transaction inventory mismatch")
    tickets = value["tickets"]
    if not isinstance(tickets, list) or len(tickets) != expected_count:
        raise ValueError("proposal KV ticket inventory mismatch")
    normalized_tickets = []
    ticket_transaction_ids = set()
    for ticket in tickets:
        _exact_keys(
            ticket,
            {
                "ticket_id",
                "transaction_id",
                "commit_entry_count",
                "release_entry_count",
                "state",
            },
            "proposal KV ticket",
        )
        transaction_id = ticket["transaction_id"]
        expected = transaction_expectations.get(transaction_id)
        if (
            not isinstance(ticket["ticket_id"], str)
            or not ticket["ticket_id"]
            or expected is None
            or transaction_id in ticket_transaction_ids
            or ticket["commit_entry_count"] != expected[0]
            or ticket["release_entry_count"] != expected[1]
            or ticket["state"] != "committed"
        ):
            raise ValueError("proposal KV ticket mismatch")
        ticket_transaction_ids.add(transaction_id)
        normalized_tickets.append(dict(ticket))
    if ticket_transaction_ids != set(transaction_expectations):
        raise ValueError("proposal KV ticket inventory mismatch")
    return {
        "active_sequence_count": 0,
        "active_transaction_count": 0,
        "prepared_ticket_count": 0,
        "owned_slot_count": 0,
        "transactions": normalized_transactions,
        "tickets": normalized_tickets,
    }


def _validate_native_rank(
    value: object,
    *,
    rank: int,
    batch_size: int,
) -> dict:
    fields = {
        "rank",
        "world_size",
        "registered",
        "module_type",
        "physical_store_type",
        "shared_embed_tokens",
        "shared_lm_head",
        "local_query_heads",
        "local_kv_heads",
        "target_prefill_observations",
        "bootstrap_rows",
        "proposal_rows",
        "proposed_tokens",
        "accepted_draft_tokens",
        "rejected_draft_tokens",
        "first_target_callbacks",
        "verify_callbacks",
        "first_target_target_forwards",
        "verify_target_forwards",
        "accepted_prefix_target_replays",
        "lm_head_logits_rows",
        "token_broadcasts",
        "token_broadcast_shape",
        "token_broadcast_dtype",
        "token_broadcast_source_rank",
        "selected_tokens_sha256",
        "finalize_ack_ranks",
        "release_ack_ranks",
        "executor",
    }
    _exact_keys(value, fields, "rank snapshot")
    if (
        value["rank"] != rank
        or value["world_size"] != WORLD_SIZE
        or value["registered"] is not True
        or value["module_type"] != "Qwen35NativeMTP"
        or value["physical_store_type"]
        != "Qwen35MTPPhysicalSlotStore"
        or value["shared_embed_tokens"] is not True
        or value["shared_lm_head"] is not True
        or value["local_query_heads"] <= 0
        or value["local_kv_heads"] <= 0
        or value["target_prefill_observations"] != batch_size
        or value["bootstrap_rows"] != batch_size
        or value["proposal_rows"] <= 0
        or value["proposed_tokens"] <= 0
        or value["accepted_draft_tokens"] <= 0
        or value["rejected_draft_tokens"] <= 0
        or value["first_target_callbacks"] <= 0
        or value["verify_callbacks"] <= 0
        or value["first_target_target_forwards"]
        != value["first_target_callbacks"]
        or value["verify_target_forwards"]
        != value["verify_callbacks"]
    ):
        raise ValueError("native rank callback counters mismatch")
    if value["accepted_prefix_target_replays"] != 0:
        raise ValueError("accepted-prefix target replay detected")
    if (
        value["finalize_ack_ranks"] != list(WORKER_RANKS)
        or value["release_ack_ranks"] != list(WORKER_RANKS)
    ):
        raise ValueError("command acknowledgement rank mismatch")
    executor = value["executor"]
    _exact_keys(
        executor,
        {
            "tensor_parallel_rank",
            "tensor_parallel_size",
            "proposal_transactions",
            "selected_tokens",
            "release_rows",
            "active_transactions",
            "prepared_tickets",
            "pending_sequences",
            "bootstrapped_sequences",
            "allocated_physical_slots",
            "proposal_kv_cache",
        },
        "executor snapshot",
    )
    if (
        executor["tensor_parallel_rank"] != rank
        or executor["tensor_parallel_size"] != WORLD_SIZE
    ):
        raise ValueError("executor rank topology mismatch")
    transactions = _validate_transaction_rows(
        executor["proposal_transactions"],
        batch_size=batch_size,
        expected_count=value["proposal_rows"],
    )
    selected_tokens = _validate_selected_tokens(
        executor["selected_tokens"],
        transactions=transactions,
    )
    expected_selected = len(selected_tokens)
    if value["lm_head_logits_rows"] != (
        expected_selected if rank == 0 else 0
    ):
        raise ValueError("rank-0 logits authority mismatch")
    if (
        value["token_broadcasts"] != expected_selected
        or value["token_broadcast_shape"] != [1]
        or value["token_broadcast_dtype"] != "torch.int64"
        or value["token_broadcast_source_rank"] != 0
    ):
        raise ValueError("token broadcast evidence mismatch")
    if (
        _sha256(
            value["selected_tokens_sha256"],
            "selected token",
        )
        != _json_sha256(selected_tokens)
    ):
        raise ValueError("selected token digest mismatch")
    release_rows = [
        {"sequence_id": sequence_id, "sequence_epoch": 0}
        for sequence_id in range(batch_size)
    ]
    if executor["release_rows"] != release_rows:
        raise ValueError("release row inventory mismatch")
    for field, message in (
        ("active_transactions", "transaction leak"),
        ("prepared_tickets", "ticket leak"),
        ("pending_sequences", "sequence leak"),
        ("bootstrapped_sequences", "sequence leak"),
        ("allocated_physical_slots", "slot leak"),
    ):
        if executor[field] != 0:
            raise ValueError(message)
    cache = _validate_cache_snapshot(
        executor["proposal_kv_cache"],
        transactions=transactions,
        batch_size=batch_size,
    )
    return {
        **{
            name: value[name]
            for name in fields
            if name != "executor"
        },
        "selected_tokens_sha256": _json_sha256(selected_tokens),
        "executor": {
            "tensor_parallel_rank": rank,
            "tensor_parallel_size": WORLD_SIZE,
            "proposal_transactions": transactions,
            "selected_tokens": selected_tokens,
            "release_rows": release_rows,
            "active_transactions": 0,
            "prepared_tickets": 0,
            "pending_sequences": 0,
            "bootstrapped_sequences": 0,
            "allocated_physical_slots": 0,
            "proposal_kv_cache": cache,
        },
    }


def _validate_baseline_rank(value: object, *, rank: int) -> dict:
    if not isinstance(value, dict):
        raise ValueError("rank snapshot must be a mapping")
    expected = {
        "rank": rank,
        "world_size": WORLD_SIZE,
        "registered": False,
        "module_type": None,
        "physical_store_type": None,
        "shared_embed_tokens": False,
        "shared_lm_head": False,
        "local_query_heads": 0,
        "local_kv_heads": 0,
        "target_prefill_observations": 0,
        "bootstrap_rows": 0,
        "proposal_rows": 0,
        "proposed_tokens": 0,
        "accepted_draft_tokens": 0,
        "rejected_draft_tokens": 0,
        "first_target_callbacks": 0,
        "verify_callbacks": 0,
        "first_target_target_forwards": 0,
        "verify_target_forwards": 0,
        "accepted_prefix_target_replays": 0,
        "lm_head_logits_rows": 0,
        "token_broadcasts": 0,
        "token_broadcast_shape": [],
        "token_broadcast_dtype": None,
        "token_broadcast_source_rank": None,
        "selected_tokens_sha256": _json_sha256([]),
        "finalize_ack_ranks": [],
        "release_ack_ranks": [],
        "executor": None,
    }
    if value != expected:
        raise ValueError("baseline speculative activity detected")
    return dict(expected)


def _validate_rank_snapshots(
    value: object,
    *,
    policy: str,
    batch_size: int,
) -> list[dict]:
    if not isinstance(value, list) or len(value) != WORLD_SIZE:
        raise ValueError("rank snapshot inventory mismatch")
    if [row.get("rank") for row in value if isinstance(row, dict)] != list(
        RANKS
    ):
        raise ValueError("rank snapshot inventory mismatch")
    if policy == "native_mtp":
        try:
            reference_transactions = value[0]["executor"][
                "proposal_transactions"
            ]
            reference_transaction_ids = [
                row["transaction_id"]
                for row in reference_transactions
            ]
            reference_ticket_ids = [
                row["finalize_ticket_id"]
                for row in reference_transactions
            ]
            for row in value[1:]:
                transactions = row["executor"][
                    "proposal_transactions"
                ]
                if [
                    transaction["transaction_id"]
                    for transaction in transactions
                ] != reference_transaction_ids:
                    raise ValueError("transaction parity mismatch")
                if [
                    transaction["finalize_ticket_id"]
                    for transaction in transactions
                ] != reference_ticket_ids:
                    raise ValueError("ticket parity mismatch")
        except (KeyError, TypeError):
            raise ValueError(
                "rank snapshot transaction inventory mismatch"
            ) from None
    rows = [
        (
            _validate_baseline_rank(row, rank=rank)
            if policy == "baseline"
            else _validate_native_rank(
                row,
                rank=rank,
                batch_size=batch_size,
            )
        )
        for rank, row in enumerate(value)
    ]
    if policy == "native_mtp":
        root = rows[0]["executor"]
        for row in rows[1:]:
            executor = row["executor"]
            if (
                executor["proposal_transactions"]
                != root["proposal_transactions"]
            ):
                raise ValueError("transaction parity mismatch")
            if any(
                left["finalize_ticket_id"]
                != right["finalize_ticket_id"]
                for left, right in zip(
                    executor["proposal_transactions"],
                    root["proposal_transactions"],
                )
            ):
                raise ValueError("ticket parity mismatch")
            if executor["selected_tokens"] != root["selected_tokens"]:
                raise ValueError("selected token parity mismatch")
    return rows


def _validate_cleanup(value: object) -> dict:
    _exact_keys(
        value,
        {
            "rank_exit_codes",
            "process_group_destroyed",
            "shared_memory_released",
            "owned_children_remaining",
            "engine_exit_called",
        },
        "cleanup",
    )
    if value["rank_exit_codes"] != [0, 0, 0, 0]:
        raise ValueError("rank exit cleanup mismatch")
    if (
        value["process_group_destroyed"] is not True
        or value["shared_memory_released"] is not True
        or value["owned_children_remaining"] != []
        or value["engine_exit_called"] is not True
    ):
        raise ValueError("cleanup invariant mismatch")
    return {
        "rank_exit_codes": [0, 0, 0, 0],
        "process_group_destroyed": True,
        "shared_memory_released": True,
        "owned_children_remaining": [],
        "engine_exit_called": True,
    }


def validate_cell_result(value: object) -> dict:
    fields = {
        "schema_version",
        "policy",
        "batch_size",
        "world_size",
        "rank_inventory",
        "gpu_indices",
        "prompt_token_count",
        "max_output_tokens",
        "max_proposal_tokens",
        "model_identity",
        "prompt_rows",
        "output_rows",
        "tp1_output_rows",
        "rank_snapshots",
        "side_state_receipts",
        "target_kv_receipts",
        "runtime_poisoned",
        "cleanup",
    }
    _exact_keys(value, fields, "cell")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("cell schema version mismatch")
    policy = value["policy"]
    batch_size = value["batch_size"]
    cell_key(policy, batch_size)
    if (
        value["world_size"] != WORLD_SIZE
        or value["rank_inventory"] != list(RANKS)
    ):
        raise ValueError("cell rank inventory mismatch")
    if (
        not isinstance(value["gpu_indices"], list)
        or len(value["gpu_indices"]) != WORLD_SIZE
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in value["gpu_indices"]
        )
        or len(set(value["gpu_indices"])) != WORLD_SIZE
    ):
        raise ValueError("cell GPU inventory mismatch")
    if value["prompt_token_count"] != PROMPT_TOKENS:
        raise ValueError("cell prompt token count mismatch")
    if value["max_output_tokens"] != MAX_OUTPUT_TOKENS:
        raise ValueError("cell output token count mismatch")
    if value["max_proposal_tokens"] != MAX_PROPOSAL_TOKENS:
        raise ValueError("cell proposal token count mismatch")
    model_identity = _validate_model_identity(value["model_identity"])
    prompt_rows = _validate_token_rows(
        value["prompt_rows"],
        batch_size=batch_size,
        token_count=PROMPT_TOKENS,
        name="prompt",
    )
    output_rows = _validate_token_rows(
        value["output_rows"],
        batch_size=batch_size,
        token_count=MAX_OUTPUT_TOKENS,
        name="output",
    )
    tp1_output_rows = value["tp1_output_rows"]
    if policy == "baseline":
        if tp1_output_rows is not None:
            raise ValueError("baseline TP1 output rows must be absent")
        normalized_tp1 = None
    else:
        normalized_tp1 = _validate_token_rows(
            tp1_output_rows,
            batch_size=batch_size,
            token_count=MAX_OUTPUT_TOKENS,
            name="TP1 output",
        )
    rank_snapshots = _validate_rank_snapshots(
        value["rank_snapshots"],
        policy=policy,
        batch_size=batch_size,
    )
    if policy == "baseline":
        if value["side_state_receipts"] != []:
            raise ValueError("baseline side-state activity detected")
        if value["target_kv_receipts"] != []:
            raise ValueError("baseline target KV activity detected")
        side_state = []
        target_kv = []
    else:
        side_state = _validate_receipts(
            value["side_state_receipts"],
            batch_size=batch_size,
            operations=["prepare", "select", "apply", "seal"],
            name="side-state",
        )
        target_kv = _validate_receipts(
            value["target_kv_receipts"],
            batch_size=batch_size,
            operations=["prepare", "commit"],
            name="target KV",
        )
    if value["runtime_poisoned"] is not False:
        raise ValueError("runtime is poisoned")
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": policy,
        "batch_size": batch_size,
        "world_size": WORLD_SIZE,
        "rank_inventory": list(RANKS),
        "gpu_indices": list(value["gpu_indices"]),
        "prompt_token_count": PROMPT_TOKENS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_proposal_tokens": MAX_PROPOSAL_TOKENS,
        "model_identity": model_identity,
        "prompt_rows": prompt_rows,
        "output_rows": output_rows,
        "tp1_output_rows": normalized_tp1,
        "rank_snapshots": rank_snapshots,
        "side_state_receipts": side_state,
        "target_kv_receipts": target_kv,
        "runtime_poisoned": False,
        "cleanup": _validate_cleanup(value["cleanup"]),
    }


def validate_result(value: object) -> dict:
    fields = {
        "schema_version",
        "classification",
        "promotion_classification",
        "target_model_manifest_sha256",
        "mtp_checkpoint_manifest_sha256",
        "tp1_authority_sha256",
        "source_tree_sha256",
        "world_size",
        "rank_inventory",
        "gpu_indices",
        "gpu_process_inventory_before",
        "gpu_process_inventory_after",
        "cells",
        "parity",
        "limitations",
    }
    _exact_keys(value, fields, "result")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("result schema version mismatch")
    if value["classification"] != CLASSIFICATION:
        raise ValueError("result classification mismatch")
    if value["promotion_classification"] != PROMOTION_CLASSIFICATION:
        raise ValueError("result promotion classification mismatch")
    target = _sha256(
        value["target_model_manifest_sha256"],
        "target model manifest",
    )
    mtp = _sha256(
        value["mtp_checkpoint_manifest_sha256"],
        "MTP checkpoint manifest",
    )
    tp1 = _sha256(
        value["tp1_authority_sha256"],
        "TP1 authority",
    )
    source = _sha256(value["source_tree_sha256"], "source tree")
    if target != TARGET_MODEL_MANIFEST_SHA256:
        raise ValueError("target model manifest does not match authority")
    if mtp != MTP_CHECKPOINT_MANIFEST_SHA256:
        raise ValueError("MTP checkpoint manifest does not match authority")
    if tp1 != TP1_AUTHORITY_SHA256:
        raise ValueError("TP1 authority does not match frozen authority")
    if (
        value["world_size"] != WORLD_SIZE
        or value["rank_inventory"] != list(RANKS)
    ):
        raise ValueError("result rank inventory mismatch")
    gpu_indices = value["gpu_indices"]
    if (
        not isinstance(gpu_indices, list)
        or len(gpu_indices) != WORLD_SIZE
        or len(set(gpu_indices)) != WORLD_SIZE
    ):
        raise ValueError("result GPU inventory mismatch")
    before = value["gpu_process_inventory_before"]
    after = value["gpu_process_inventory_after"]
    if not isinstance(before, list) or not isinstance(after, list):
        raise ValueError("GPU process inventory is invalid")
    if before != after:
        raise ValueError("GPU process inventory changed")
    expected_cells = {
        cell_key(policy, batch_size)
        for policy in POLICIES
        for batch_size in BATCH_SIZES
    }
    cells = value["cells"]
    if not isinstance(cells, dict) or set(cells) != expected_cells:
        raise ValueError("result cell inventory mismatch")
    normalized_cells = {
        key: validate_cell_result(cells[key])
        for key in sorted(cells)
    }
    parity = {
        "baseline_native": {},
        "tp1_tp4_native": {},
    }
    for batch_size in BATCH_SIZES:
        baseline = normalized_cells[cell_key("baseline", batch_size)]
        native = normalized_cells[cell_key("native_mtp", batch_size)]
        if baseline["gpu_indices"] != gpu_indices:
            raise ValueError("cell GPU inventory mismatch")
        if native["gpu_indices"] != gpu_indices:
            raise ValueError("cell GPU inventory mismatch")
        if (
            baseline["prompt_rows"] != native["prompt_rows"]
            or baseline["output_rows"] != native["output_rows"]
        ):
            raise ValueError(
                f"baseline/native output parity mismatch for batch {batch_size}"
            )
        if native["output_rows"] != native["tp1_output_rows"]:
            raise ValueError(
                f"TP1/TP4 output parity mismatch for batch {batch_size}"
            )
        parity["baseline_native"][f"b{batch_size}"] = True
        parity["tp1_tp4_native"][f"b{batch_size}"] = True
    if value["parity"] != parity:
        raise ValueError("parity summary mismatch")
    if value["limitations"] != list(REQUIRED_LIMITATIONS):
        raise ValueError("authority limitations mismatch")
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "promotion_classification": PROMOTION_CLASSIFICATION,
        "target_model_manifest_sha256": target,
        "mtp_checkpoint_manifest_sha256": mtp,
        "tp1_authority_sha256": tp1,
        "source_tree_sha256": source,
        "world_size": WORLD_SIZE,
        "rank_inventory": list(RANKS),
        "gpu_indices": list(gpu_indices),
        "gpu_process_inventory_before": list(before),
        "gpu_process_inventory_after": list(after),
        "cells": normalized_cells,
        "parity": parity,
        "limitations": list(REQUIRED_LIMITATIONS),
    }


def assemble_authority(
    cells: dict,
    *,
    source_tree_sha256: str,
    target_model_manifest_sha256: str,
    mtp_checkpoint_manifest_sha256: str,
    tp1_authority_sha256: str,
    gpu_indices: list[int],
    gpu_process_inventory_before: list,
    gpu_process_inventory_after: list,
    limitations: list[str],
) -> dict:
    return validate_result({
        "schema_version": SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "promotion_classification": PROMOTION_CLASSIFICATION,
        "target_model_manifest_sha256": target_model_manifest_sha256,
        "mtp_checkpoint_manifest_sha256": mtp_checkpoint_manifest_sha256,
        "tp1_authority_sha256": tp1_authority_sha256,
        "source_tree_sha256": source_tree_sha256,
        "world_size": WORLD_SIZE,
        "rank_inventory": list(RANKS),
        "gpu_indices": list(gpu_indices),
        "gpu_process_inventory_before": list(
            gpu_process_inventory_before
        ),
        "gpu_process_inventory_after": list(
            gpu_process_inventory_after
        ),
        "cells": cells,
        "parity": {
            "baseline_native": {
                f"b{batch_size}": True
                for batch_size in BATCH_SIZES
            },
            "tp1_tp4_native": {
                f"b{batch_size}": True
                for batch_size in BATCH_SIZES
            },
        },
        "limitations": limitations,
    })


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_source_files(
    root: Path,
    files: tuple[str, ...],
) -> dict[str, str]:
    root = Path(root)
    return {
        name: sha256_file(root / name)
        for name in files
    }


def source_hashes_sha256(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(hashes):
        name_bytes = name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(bytes.fromhex(hashes[name]))
    return digest.hexdigest()


def source_tree_sha256(
    root: Path,
    files: tuple[str, ...],
) -> str:
    return source_hashes_sha256(hash_source_files(root, files))


def atomic_write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def publish_authority(
    output_dir: Path,
    result: dict,
    *,
    source_files: dict[str, str],
) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("authority output path already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.",
        dir=output_dir.parent,
    ))
    try:
        normalized = validate_result(result)
        normalized_sources = {
            name: _sha256(digest, "source file digest")
            for name, digest in source_files.items()
        }
        atomic_write_json(temporary / "result.json", normalized)
        status = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "classification": CLASSIFICATION,
            "promotion_classification": PROMOTION_CLASSIFICATION,
        }
        atomic_write_json(temporary / "status.json", status)
        atomic_write_json(
            temporary / "source_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "source_tree_sha256": normalized[
                    "source_tree_sha256"
                ],
                "target_model_manifest_sha256": normalized[
                    "target_model_manifest_sha256"
                ],
                "mtp_checkpoint_manifest_sha256": normalized[
                    "mtp_checkpoint_manifest_sha256"
                ],
                "tp1_authority_sha256": normalized[
                    "tp1_authority_sha256"
                ],
                "source_files": normalized_sources,
                "artifacts": {
                    "result.json": sha256_file(
                        temporary / "result.json"
                    ),
                    "status.json": sha256_file(
                        temporary / "status.json"
                    ),
                },
            },
        )
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            for path in temporary.iterdir():
                path.unlink()
            temporary.rmdir()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_default_verifier():
    module = _load_module(
        "verify_qwen35_native_mtp_tp4_4k_engine_gate",
        _VERIFIER,
    )
    return module.verify_run


def _default_gpu_process_inventory(
    gpu_indices: tuple[int, ...],
) -> list[str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            ",".join(str(index) for index in gpu_indices),
            (
                "--query-compute-apps="
                "gpu_uuid,pid,process_name,used_gpu_memory"
            ),
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "GPU process inventory failed: "
            + completed.stderr.strip()
        )
    return sorted(
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    )


def run_campaign(
    *,
    model_path: str,
    gpu_indices: tuple[int, ...],
    tp1_result_path: Path,
    output_dir: Path,
    repo_root: Path | None = None,
    worker_script: Path | None = None,
    python_executable: str = sys.executable,
    source_files: tuple[str, ...] = DEFAULT_SOURCE_FILES,
    gpu_process_inventory=_default_gpu_process_inventory,
    verifier=None,
    dist_port_base: int = 29640,
    master_port_base: int = 29740,
) -> dict:
    if (
        not isinstance(gpu_indices, tuple)
        or len(gpu_indices) != WORLD_SIZE
        or len(set(gpu_indices)) != WORLD_SIZE
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in gpu_indices
        )
    ):
        raise ValueError("campaign GPU indices are invalid")
    tp1_result_path = Path(tp1_result_path)
    if (
        not tp1_result_path.is_file()
        or sha256_file(tp1_result_path) != TP1_AUTHORITY_SHA256
    ):
        raise ValueError("TP1 authority digest mismatch")
    repo_root = (
        _ROOT if repo_root is None else Path(repo_root)
    )
    worker_script = (
        _WORKER
        if worker_script is None
        else Path(worker_script)
    )
    output_dir = Path(output_dir)
    failed_dir = output_dir.with_name(
        f"{output_dir.name}.failed"
    )
    if output_dir.exists() or failed_dir.exists():
        raise ValueError("authority output path already exists")
    source_hashes = hash_source_files(
        repo_root,
        source_files,
    )
    source_digest = source_hashes_sha256(source_hashes)
    before = gpu_process_inventory(gpu_indices)
    if not isinstance(before, list):
        raise ValueError("GPU process inventory is invalid")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.campaign.",
        dir=output_dir.parent,
    ))
    try:
        cell_dir = temporary_root / "cells"
        cell_dir.mkdir()
        cells = {}
        ordinal = 0
        for batch_size in BATCH_SIZES:
            for policy in POLICIES:
                key = cell_key(policy, batch_size)
                cell_path = cell_dir / f"{key}.json"
                log_path = cell_dir / f"{key}.log"
                command = [
                    python_executable,
                    str(worker_script),
                    "--model",
                    model_path,
                    "--gpu-indices",
                    ",".join(str(index) for index in gpu_indices),
                    "--policy",
                    policy,
                    "--batch-size",
                    str(batch_size),
                    "--dist-port",
                    str(dist_port_base + ordinal),
                    "--master-port",
                    str(master_port_base + ordinal),
                    "--tp1-result",
                    str(tp1_result_path),
                    "--out",
                    str(cell_path),
                ]
                ordinal += 1
                with log_path.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        command,
                        cwd=repo_root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"worker failed for {key}; log={log_path}"
                    )
                cells[key] = validate_cell_result(
                    json.loads(
                        cell_path.read_text(encoding="utf-8")
                    )
                )
        worker_module = _load_module(
            "qwen35_native_mtp_tp4_4k_engine_worker",
            worker_script,
        )
        target_digest = (
            worker_module.target_model_manifest_sha256(
                model_path
            )
        )
        mtp_digest = (
            worker_module.mtp_checkpoint_manifest_sha256(
                model_path
            )
        )
        final_source_hashes = hash_source_files(
            repo_root,
            source_files,
        )
        if final_source_hashes != source_hashes:
            raise RuntimeError("source changed during campaign")
        after = gpu_process_inventory(gpu_indices)
        if not isinstance(after, list):
            raise ValueError("GPU process inventory is invalid")
        result = assemble_authority(
            cells,
            source_tree_sha256=source_digest,
            target_model_manifest_sha256=target_digest,
            mtp_checkpoint_manifest_sha256=mtp_digest,
            tp1_authority_sha256=TP1_AUTHORITY_SHA256,
            gpu_indices=list(gpu_indices),
            gpu_process_inventory_before=before,
            gpu_process_inventory_after=after,
            limitations=list(REQUIRED_LIMITATIONS),
        )
        authority_dir = temporary_root / "authority"
        publish_authority(
            authority_dir,
            result,
            source_files=source_hashes,
        )
        shutil.copytree(
            cell_dir,
            authority_dir / "cells",
        )
        verify = (
            _load_default_verifier()
            if verifier is None
            else verifier
        )
        verification = verify(authority_dir, repo_root)
        atomic_write_json(
            authority_dir / "verify.json",
            verification,
        )
        if verification != {
            "classification": "PASS",
            "failures": [],
        }:
            raise RuntimeError(
                "independent verification failed: "
                + json.dumps(
                    verification.get("failures", []),
                    sort_keys=True,
                )
            )
        os.replace(authority_dir, output_dir)
        shutil.rmtree(temporary_root)
        return result
    except Exception as error:
        if temporary_root.exists():
            os.replace(temporary_root, failed_dir)
        raise RuntimeError(
            f"{error}; failed_artifacts={failed_dir}"
        ) from error
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def _gpu_indices_argument(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "GPU indices must be comma-separated integers"
        ) from error
    if (
        len(parsed) != WORLD_SIZE
        or len(set(parsed)) != WORLD_SIZE
        or any(index < 0 for index in parsed)
    ):
        raise argparse.ArgumentTypeError(
            "GPU indices must contain four distinct integers"
        )
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--gpu-indices",
        required=True,
        type=_gpu_indices_argument,
    )
    parser.add_argument("--tp1-result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dist-port-base",
        type=int,
        default=29640,
    )
    parser.add_argument(
        "--master-port-base",
        type=int,
        default=29740,
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_campaign(
        model_path=args.model,
        gpu_indices=args.gpu_indices,
        tp1_result_path=Path(args.tp1_result),
        output_dir=Path(args.output_dir),
        dist_port_base=args.dist_port_base,
        master_port_base=args.master_port_base,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

