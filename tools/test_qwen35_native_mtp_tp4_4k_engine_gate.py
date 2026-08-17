from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = (
    ROOT
    / "tools"
    / "qwen35_native_mtp_tp4_4k_engine_gate.py"
)
WORKER_PATH = (
    ROOT
    / "tools"
    / "qwen35_native_mtp_tp4_4k_engine_worker.py"
)
TP1_GATE_PATH = (
    ROOT
    / "tools"
    / "qwen35_native_mtp_tp1_4k_engine_gate.py"
)
TP1_WORKER_PATH = (
    ROOT
    / "tools"
    / "qwen35_native_mtp_tp1_4k_engine_worker.py"
)
VERIFIER_PATH = (
    ROOT
    / "tools"
    / "verify_qwen35_native_mtp_tp4_4k_engine_gate.py"
)
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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_gate(name: str):
    return _load_module(name, GATE_PATH)


def _load_worker(name: str):
    return _load_module(name, WORKER_PATH)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _token_rows(
    batch_size: int,
    *,
    token_count: int,
    start: int,
) -> list[dict]:
    rows = []
    for prompt_index in range(batch_size):
        token_ids = [
            start + prompt_index + offset % 7
            for offset in range(token_count)
        ]
        rows.append({
            "prompt_index": prompt_index,
            "token_count": token_count,
            "token_ids": token_ids,
            "sha256": _digest(token_ids),
        })
    return rows


def _transaction_rows(batch_size: int) -> list[dict]:
    return [
        {
            "sequence_id": sequence_id,
            "sequence_epoch": 0,
            "transaction_id": (
                f"proposal-kv-transaction-{2 * sequence_id + 2}"
            ),
            "exact_q": 4,
            "token_ids": [
                300 + sequence_id,
                301 + sequence_id,
                302 + sequence_id,
                303 + sequence_id,
            ],
            "staged_entry_count": 3,
            "accepted_proposal_tokens": 2,
            "rejected_proposal_tokens": 2,
            "finalize_ticket_id": (
                f"qwen35-mtp-finalize-{sequence_id + 1}"
            ),
            "state": "committed",
        }
        for sequence_id in range(batch_size)
    ]


def _selected_tokens(batch_size: int) -> list[dict]:
    rows = []
    for transaction in _transaction_rows(batch_size):
        for step, token_id in enumerate(
            transaction["token_ids"][1:]
        ):
            rows.append({
                "sequence_id": transaction["sequence_id"],
                "transaction_id": transaction[
                    "transaction_id"
                ],
                "step": step,
                "token_id": token_id,
            })
    return rows


def _rank_snapshot(rank: int, batch_size: int) -> dict:
    transactions = _transaction_rows(batch_size)
    cache_transactions = []
    tickets = []
    for sequence_id, row in enumerate(transactions):
        bootstrap_transaction_id = (
            f"proposal-kv-transaction-{2 * sequence_id + 1}"
        )
        cache_transactions.extend((
            {
                "transaction_id": bootstrap_transaction_id,
                "sequence_id": sequence_id,
                "sequence_epoch": 0,
                "original_committed_length": 0,
                "staged_entry_count": 4096,
                "materialized_entry_count": 4096,
                "state": "committed",
            },
            {
                "transaction_id": row["transaction_id"],
                "sequence_id": row["sequence_id"],
                "sequence_epoch": row["sequence_epoch"],
                "original_committed_length": 4096,
                "staged_entry_count": row["staged_entry_count"],
                "materialized_entry_count": row[
                    "staged_entry_count"
                ],
                "state": "committed",
            },
        ))
        tickets.extend((
            {
                "ticket_id": (
                    f"proposal-kv-ticket-{2 * sequence_id + 1}"
                ),
                "transaction_id": bootstrap_transaction_id,
                "commit_entry_count": 4096,
                "release_entry_count": 0,
                "state": "committed",
            },
            {
                "ticket_id": (
                    f"proposal-kv-ticket-{2 * sequence_id + 2}"
                ),
                "transaction_id": row["transaction_id"],
                "commit_entry_count": 1,
                "release_entry_count": 2,
                "state": "committed",
            },
        ))
    return {
        "rank": rank,
        "world_size": 4,
        "registered": True,
        "module_type": "Qwen35NativeMTP",
        "physical_store_type": "Qwen35MTPPhysicalSlotStore",
        "shared_embed_tokens": True,
        "shared_lm_head": True,
        "local_query_heads": 4,
        "local_kv_heads": 1,
        "target_prefill_observations": batch_size,
        "bootstrap_rows": batch_size,
        "proposal_rows": batch_size,
        "proposed_tokens": batch_size * 4,
        "accepted_draft_tokens": batch_size * 2,
        "rejected_draft_tokens": batch_size * 2,
        "first_target_callbacks": 1,
        "verify_callbacks": 1,
        "first_target_target_forwards": 1,
        "verify_target_forwards": 1,
        "accepted_prefix_target_replays": 0,
        "lm_head_logits_rows": batch_size * 3 if rank == 0 else 0,
        "token_broadcasts": batch_size * 3,
        "token_broadcast_shape": [1],
        "token_broadcast_dtype": "torch.int64",
        "token_broadcast_source_rank": 0,
        "selected_tokens_sha256": _digest(
            _selected_tokens(batch_size)
        ),
        "finalize_ack_ranks": [1, 2, 3],
        "release_ack_ranks": [1, 2, 3],
        "executor": {
            "tensor_parallel_rank": rank,
            "tensor_parallel_size": 4,
            "proposal_transactions": transactions,
            "selected_tokens": _selected_tokens(batch_size),
            "release_rows": [
                {
                    "sequence_id": sequence_id,
                    "sequence_epoch": 0,
                }
                for sequence_id in range(batch_size)
            ],
            "active_transactions": 0,
            "prepared_tickets": 0,
            "pending_sequences": 0,
            "bootstrapped_sequences": 0,
            "allocated_physical_slots": 0,
            "proposal_kv_cache": {
                "active_sequence_count": 0,
                "active_transaction_count": 0,
                "prepared_ticket_count": 0,
                "owned_slot_count": 0,
                "transactions": cache_transactions,
                "tickets": tickets,
            },
        },
    }


def _baseline_rank_snapshot(rank: int) -> dict:
    return {
        "rank": rank,
        "world_size": 4,
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
        "selected_tokens_sha256": _digest([]),
        "finalize_ack_ranks": [],
        "release_ack_ranks": [],
        "executor": None,
    }


def _cell(policy: str, batch_size: int) -> dict:
    output_rows = _token_rows(
        batch_size,
        token_count=32,
        start=101,
    )
    return {
        "schema_version": (
            "qwen35.native-mtp-tp4-4k-engine-"
            "transactional-correctness.v1"
        ),
        "policy": policy,
        "batch_size": batch_size,
        "world_size": 4,
        "rank_inventory": [0, 1, 2, 3],
        "gpu_indices": [0, 1, 2, 3],
        "prompt_token_count": 4096,
        "max_output_tokens": 32,
        "max_proposal_tokens": 4,
        "model_identity": {
            "model_type": "qwen3_5",
            "architectures": [
                "Qwen3_5ForConditionalGeneration"
            ],
            "target_model_manifest_sha256": (
                TARGET_MODEL_MANIFEST_SHA256
            ),
            "mtp_checkpoint_manifest_sha256": (
                MTP_CHECKPOINT_MANIFEST_SHA256
            ),
        },
        "prompt_rows": _token_rows(
            batch_size,
            token_count=4096,
            start=17,
        ),
        "output_rows": output_rows,
        "tp1_output_rows": (
            deepcopy(output_rows)
            if policy == "native_mtp"
            else None
        ),
        "rank_snapshots": [
            (
                _baseline_rank_snapshot(rank)
                if policy == "baseline"
                else _rank_snapshot(rank, batch_size)
            )
            for rank in range(4)
        ],
        "side_state_receipts": (
            []
            if policy == "baseline"
            else [
                {
                    "sequence_id": sequence_id,
                    "operations": [
                        "prepare",
                        "select",
                        "apply",
                        "seal",
                    ],
                }
                for sequence_id in range(batch_size)
            ]
        ),
        "target_kv_receipts": (
            []
            if policy == "baseline"
            else [
                {
                    "sequence_id": sequence_id,
                    "operations": ["prepare", "commit"],
                }
                for sequence_id in range(batch_size)
            ]
        ),
        "runtime_poisoned": False,
        "cleanup": {
            "rank_exit_codes": [0, 0, 0, 0],
            "process_group_destroyed": True,
            "shared_memory_released": True,
            "owned_children_remaining": [],
            "engine_exit_called": True,
        },
    }


def _result() -> dict:
    cells = {
        f"{policy}:b{batch_size}": _cell(
            policy,
            batch_size,
        )
        for batch_size in (1, 4)
        for policy in ("baseline", "native_mtp")
    }
    return {
        "schema_version": (
            "qwen35.native-mtp-tp4-4k-engine-"
            "transactional-correctness.v1"
        ),
        "classification": (
            "QWEN35_NATIVE_MTP_TP4_4K_ENGINE_ESTABLISHED"
        ),
        "promotion_classification": "NOT_PROMOTABLE",
        "target_model_manifest_sha256": (
            TARGET_MODEL_MANIFEST_SHA256
        ),
        "mtp_checkpoint_manifest_sha256": (
            MTP_CHECKPOINT_MANIFEST_SHA256
        ),
        "tp1_authority_sha256": TP1_AUTHORITY_SHA256,
        "source_tree_sha256": "a" * 64,
        "world_size": 4,
        "rank_inventory": [0, 1, 2, 3],
        "gpu_indices": [0, 1, 2, 3],
        "gpu_process_inventory_before": ["pid:17"],
        "gpu_process_inventory_after": ["pid:17"],
        "cells": cells,
        "parity": {
            "baseline_native": {"b1": True, "b4": True},
            "tp1_tp4_native": {"b1": True, "b4": True},
        },
        "limitations": [
            "TP4 only",
            "4K prompt only",
            "KV offload disabled",
            "eager native MTP only",
            "not production ready",
        ],
    }


def test_contract_constants_are_frozen():
    gate = _load_gate("native_mtp_tp4_gate_constants")

    assert gate.SCHEMA_VERSION == (
        "qwen35.native-mtp-tp4-4k-engine-"
        "transactional-correctness.v1"
    )
    assert gate.CLASSIFICATION == (
        "QWEN35_NATIVE_MTP_TP4_4K_ENGINE_ESTABLISHED"
    )
    assert gate.PROMOTION_CLASSIFICATION == "NOT_PROMOTABLE"
    assert gate.WORLD_SIZE == 4
    assert gate.RANKS == (0, 1, 2, 3)
    assert gate.WORKER_RANKS == (1, 2, 3)
    assert gate.POLICIES == ("baseline", "native_mtp")
    assert gate.BATCH_SIZES == (1, 4)
    assert gate.PROMPT_TOKENS == 4096
    assert gate.MAX_OUTPUT_TOKENS == 32
    assert gate.MAX_PROPOSAL_TOKENS == 4
    assert gate.TP1_AUTHORITY_SHA256 == TP1_AUTHORITY_SHA256


def test_complete_authority_is_accepted_and_canonical():
    gate = _load_gate("native_mtp_tp4_gate_accept")

    normalized = gate.validate_result(_result())

    assert normalized == _result()


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda result: result["rank_inventory"].pop(),
            "rank inventory",
        ),
        (
            lambda result: result["cells"].pop("baseline:b1"),
            "cell inventory",
        ),
        (
            lambda result: result.update(
                tp1_authority_sha256="0" * 64
            ),
            "TP1 authority",
        ),
        (
            lambda result: result.update(
                gpu_process_inventory_after=["pid:18"]
            ),
            "GPU process inventory",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "prompt_rows"
            ][0]["token_ids"].pop(),
            "prompt",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "output_rows"
            ][0].update(sha256="0" * 64),
            "output",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "rank_snapshots"
            ].pop(),
            "rank snapshot",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "rank_snapshots"
            ][1].update(lm_head_logits_rows=1),
            "rank-0 logits",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "rank_snapshots"
            ][0].update(token_broadcasts=2),
            "token broadcast",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "rank_snapshots"
            ][2].update(selected_tokens_sha256="0" * 64),
            "selected token",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "rank_snapshots"
            ][3]["executor"]["proposal_transactions"][0].update(
                transaction_id="other"
            ),
            "transaction parity",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "rank_snapshots"
            ][2]["executor"]["proposal_transactions"][0].update(
                finalize_ticket_id="other"
            ),
            "ticket parity",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "rank_snapshots"
            ][1]["executor"]["release_rows"].pop(),
            "release",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "rank_snapshots"
            ][0].update(accepted_prefix_target_replays=1),
            "accepted-prefix",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "rank_snapshots"
            ][0]["executor"].update(active_transactions=1),
            "transaction leak",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "rank_snapshots"
            ][0]["executor"].update(prepared_tickets=1),
            "ticket leak",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "rank_snapshots"
            ][0]["executor"].update(pending_sequences=1),
            "sequence leak",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "rank_snapshots"
            ][0]["executor"].update(allocated_physical_slots=1),
            "slot leak",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "side_state_receipts"
            ][0]["operations"].pop(),
            "side-state",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "target_kv_receipts"
            ][0]["operations"].pop(),
            "target KV",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "cleanup"
            ].update(rank_exit_codes=[0, 0, 0, 1]),
            "rank exit",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "output_rows"
            ].__setitem__(
                0,
                _token_rows(
                    1,
                    token_count=32,
                    start=777,
                )[0],
            ),
            "baseline/native",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "tp1_output_rows"
            ].__setitem__(
                0,
                _token_rows(
                    1,
                    token_count=32,
                    start=888,
                )[0],
            ),
            "TP1/TP4",
        ),
        (
            lambda result: result["parity"][
                "tp1_tp4_native"
            ].update(b1=False),
            "parity summary",
        ),
    ),
)
def test_invalid_authority_is_rejected(mutate, match):
    gate = _load_gate("native_mtp_tp4_gate_reject")
    result = _result()
    mutate(result)

    with pytest.raises(ValueError, match=match):
        gate.validate_result(result)


def test_assemble_authority_recomputes_parity():
    gate = _load_gate("native_mtp_tp4_gate_assemble")
    result = _result()

    assembled = gate.assemble_authority(
        result["cells"],
        source_tree_sha256=result["source_tree_sha256"],
        target_model_manifest_sha256=(
            result["target_model_manifest_sha256"]
        ),
        mtp_checkpoint_manifest_sha256=(
            result["mtp_checkpoint_manifest_sha256"]
        ),
        tp1_authority_sha256=result[
            "tp1_authority_sha256"
        ],
        gpu_indices=result["gpu_indices"],
        gpu_process_inventory_before=result[
            "gpu_process_inventory_before"
        ],
        gpu_process_inventory_after=result[
            "gpu_process_inventory_after"
        ],
        limitations=result["limitations"],
    )

    assert assembled == gate.validate_result(result)


def test_default_source_inventory_covers_remote_python_bundle():
    gate = _load_gate("native_mtp_tp4_source_inventory")
    expected = tuple(sorted(
        [
            str(path.relative_to(ROOT))
            for path in (ROOT / "tinyvllm").rglob("*.py")
        ]
        + [
            str(GATE_PATH.relative_to(ROOT)),
            str(WORKER_PATH.relative_to(ROOT)),
            str(TP1_GATE_PATH.relative_to(ROOT)),
            str(TP1_WORKER_PATH.relative_to(ROOT)),
            str(VERIFIER_PATH.relative_to(ROOT)),
        ]
    ))

    assert gate.DEFAULT_SOURCE_FILES == expected


def test_publish_authority_is_atomic_and_complete(tmp_path):
    gate = _load_gate("native_mtp_tp4_publish")
    output_dir = tmp_path / "authority"
    result = gate.validate_result(deepcopy(_result()))

    gate.publish_authority(
        output_dir,
        result,
        source_files={"tinyvllm/engine/llm_engine.py": "b" * 64},
    )

    assert json.loads(
        (output_dir / "result.json").read_text()
    ) == result
    assert not tuple(tmp_path.glob(".authority.*"))


def _raw_native_snapshot(rank: int, batch_size: int) -> dict:
    row = _rank_snapshot(rank, batch_size)
    return {
        key: deepcopy(value)
        for key, value in row.items()
        if key
        in {
            "rank",
            "world_size",
            "registered",
            "module_type",
            "physical_store_type",
            "shared_embed_tokens",
            "shared_lm_head",
            "local_query_heads",
            "local_kv_heads",
            "executor",
        }
    }


def test_worker_normalizes_direct_four_rank_snapshots():
    worker = _load_worker("native_mtp_tp4_worker_normalize")
    runtime = {
        "target_prefill_observations": 4,
        "proposal_rows": 4,
        "proposed_tokens": 16,
        "accepted_draft_tokens": 8,
        "rejected_draft_tokens": 8,
        "first_target_callbacks": 1,
        "verify_callbacks": 1,
        "first_target_target_forwards": 1,
        "verify_target_forwards": 1,
        "accepted_prefix_target_replays": 0,
    }

    normalized = worker.normalize_rank_snapshots(
        tuple(
            _raw_native_snapshot(rank, 4)
            for rank in range(4)
        ),
        policy="native_mtp",
        batch_size=4,
        runtime=runtime,
        finalize_ack_ranks=(1, 2, 3),
        release_ack_ranks=(1, 2, 3),
    )

    assert normalized == [
        _rank_snapshot(rank, 4)
        for rank in range(4)
    ]


def test_gate_accepts_bootstrap_and_proposal_cache_history():
    gate = _load_gate("native_mtp_tp4_gate_cache_history")
    snapshot = _rank_snapshot(0, 1)

    normalized = gate._validate_rank_snapshots(
        [snapshot] + [
            _rank_snapshot(rank, 1)
            for rank in range(1, 4)
        ],
        policy="native_mtp",
        batch_size=1,
    )

    assert normalized[0]["executor"]["proposal_kv_cache"] == (
        snapshot["executor"]["proposal_kv_cache"]
    )


def test_worker_normalizes_baseline_as_zero_activity():
    worker = _load_worker("native_mtp_tp4_worker_baseline")

    normalized = worker.normalize_rank_snapshots(
        tuple(
            {
                "rank": rank,
                "world_size": 4,
                "registered": False,
                "executor": None,
            }
            for rank in range(4)
        ),
        policy="baseline",
        batch_size=1,
        runtime=None,
        finalize_ack_ranks=(),
        release_ack_ranks=(),
    )

    assert normalized == [
        _baseline_rank_snapshot(rank)
        for rank in range(4)
    ]


def test_worker_loads_frozen_tp1_outputs_by_digest(tmp_path):
    worker = _load_worker("native_mtp_tp4_worker_tp1")
    result = {
        "cells": {
            "native_mtp:b1": {
                "output_rows": _token_rows(
                    1,
                    token_count=32,
                    start=101,
                )
            },
            "native_mtp:b4": {
                "output_rows": _token_rows(
                    4,
                    token_count=32,
                    start=101,
                )
            },
        }
    }
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert worker.load_tp1_output_rows(
        path,
        batch_size=4,
        expected_sha256=digest,
    ) == result["cells"]["native_mtp:b4"]["output_rows"]
    with pytest.raises(ValueError, match="TP1 authority"):
        worker.load_tp1_output_rows(
            path,
            batch_size=4,
            expected_sha256="0" * 64,
        )


def test_gpu_process_inventory_is_scoped_to_selected_gpus(
    monkeypatch,
):
    gate = _load_gate("native_mtp_tp4_gpu_inventory")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout="gpu-b, 2, python, 20\ngpu-a, 1, python, 10\n",
            stderr="",
        )

    monkeypatch.setattr(gate.subprocess, "run", fake_run)

    inventory = gate._default_gpu_process_inventory((2, 1, 0, 6))

    assert inventory == [
        "gpu-a, 1, python, 10",
        "gpu-b, 2, python, 20",
    ]
    assert calls[0][0][:3] == [
        "nvidia-smi",
        "-i",
        "2,1,0,6",
    ]


def test_campaign_runs_four_cells_and_binds_inventories(
    tmp_path,
    monkeypatch,
):
    gate = _load_gate("native_mtp_tp4_campaign")
    repo_root = tmp_path / "source"
    repo_root.mkdir()
    source_path = repo_root / "engine.py"
    source_path.write_text("engine\n", encoding="utf-8")
    worker_script = repo_root / "worker.py"
    worker_script.write_text("", encoding="utf-8")
    tp1_result = repo_root / "tp1-result.json"
    tp1_result.write_text("{}\n", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        policy = command[command.index("--policy") + 1]
        batch_size = int(
            command[command.index("--batch-size") + 1]
        )
        output_path = Path(
            command[command.index("--out") + 1]
        )
        gate.atomic_write_json(
            output_path,
            _cell(policy, batch_size),
        )
        calls.append((policy, batch_size))
        return SimpleNamespace(returncode=0)

    real_sha256_file = gate.sha256_file

    def fake_sha256_file(path):
        if Path(path) == tp1_result:
            return gate.TP1_AUTHORITY_SHA256
        return real_sha256_file(path)

    worker_module = SimpleNamespace(
        target_model_manifest_sha256=(
            lambda _model_path: TARGET_MODEL_MANIFEST_SHA256
        ),
        mtp_checkpoint_manifest_sha256=(
            lambda _model_path: MTP_CHECKPOINT_MANIFEST_SHA256
        ),
    )
    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    monkeypatch.setattr(
        gate,
        "sha256_file",
        fake_sha256_file,
    )
    monkeypatch.setattr(
        gate,
        "_load_module",
        lambda _name, _path: worker_module,
    )

    result = gate.run_campaign(
        model_path="/checkpoint",
        gpu_indices=(0, 1, 2, 3),
        tp1_result_path=tp1_result,
        output_dir=tmp_path / "authority",
        repo_root=repo_root,
        worker_script=worker_script,
        source_files=("engine.py",),
        gpu_process_inventory=lambda _indices: ["pid:17"],
        verifier=lambda _run_dir, _source_root: {
            "classification": "PASS",
            "failures": [],
        },
    )

    assert calls == [
        ("baseline", 1),
        ("native_mtp", 1),
        ("baseline", 4),
        ("native_mtp", 4),
    ]
    assert result["gpu_process_inventory_before"] == [
        "pid:17"
    ]
    assert result["gpu_process_inventory_after"] == [
        "pid:17"
    ]

