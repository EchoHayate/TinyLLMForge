from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


TOOLS = Path(__file__).resolve().parent
FROZEN_GATE_PATH = (
    TOOLS / "qwen35_native_mtp_tp4_4k_engine_gate.py"
)
GATE_PATH = (
    TOOLS
    / "qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py"
)
WORKER_PATH = (
    TOOLS
    / "qwen35_native_mtp_tp4_16k_target_kv_offload_worker.py"
)
VERIFIER_PATH = (
    TOOLS
    / "verify_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py"
)
REMOTE_RUNNER_PATH = (
    TOOLS
    / "run_qwen35_native_mtp_tp4_16k_target_kv_offload_remote.sh"
)
FROZEN_TEST_PATH = (
    TOOLS / "test_qwen35_native_mtp_tp4_4k_engine_gate.py"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_worker_compacts_target_logits_with_stable_topk_and_margin():
    worker = _load_module(
        "qwen35_native_mtp_tp4_16k_logits_fixture",
        WORKER_PATH,
    )

    rows = worker.compact_target_logits(
        SimpleNamespace(
            tolist=lambda: [
                [0.0, 4.0, 3.5, -1.0],
                [8.0, 1.0, 8.5, 7.5],
            ],
        ),
        sequence_ids=(9, 3),
        top_k=3,
    )

    assert rows == [
        {
            "sequence_id": 9,
            "top_tokens": [1, 2, 0],
            "top_logits": [4.0, 3.5, 0.0],
            "top1_margin": 0.5,
        },
        {
            "sequence_id": 3,
            "top_tokens": [2, 0, 3],
            "top_logits": [8.5, 8.0, 7.5],
            "top1_margin": 0.5,
        },
    ]


def test_worker_generation_logs_per_step_target_logits(capsys):
    worker = _load_module(
        "qwen35_native_mtp_tp4_16k_generation_logits_fixture",
        WORKER_PATH,
    )

    class ModelRunner:
        def __init__(self):
            self.enabled = []

        def enable_step_logits_recording(self, enabled):
            self.enabled.append(enabled)

        def last_step_logits(self):
            return SimpleNamespace(
                tolist=lambda: [
                    [0.0, 4.0, 3.5],
                    [8.0, 1.0, 8.5],
                ],
            )

    target_forward_capture = {
        "ordinary_decode_target_forward_calls": 0,
    }

    class Engine:
        def __init__(self):
            self.model_runner = ModelRunner()
            self.last_step_observation = None
            self.finished = False
            self.requests = []

        def add_request(self, token_ids, sampling_params):
            self.requests.append((token_ids, sampling_params))

        def is_finished(self):
            return self.finished

        def step(self):
            target_forward_capture[
                "ordinary_decode_target_forward_calls"
            ] += 1
            self.finished = True
            self.last_step_observation = {
                "new_completion_tokens_by_seq": {
                    9: [1],
                    3: [2],
                },
            }
            return (
                [
                    (9, list(range(worker.gate.MAX_OUTPUT_TOKENS))),
                    (3, list(range(worker.gate.MAX_OUTPUT_TOKENS))),
                ],
                2,
            )

    engine = Engine()
    output_rows, observations = (
        worker.run_generation_with_target_logit_diagnostics(
            engine=engine,
            prompt_rows=[
                {"token_ids": [11]},
                {"token_ids": [22]},
            ],
            sampling_params=object(),
            synchronize=lambda: None,
            target_forward_capture=target_forward_capture,
        )
    )

    assert engine.model_runner.enabled == [True, False]
    assert len(output_rows) == 2
    assert observations[0][
        "authority_normal_decode_target_forward_calls"
    ] == 1
    assert observations[0]["authority_target_logits"][0][
        "top_tokens"
    ] == [1, 2, 0]
    logged = capsys.readouterr().out
    assert "AUTHORITY_TARGET_LOGITS " in logged
    assert '"sequence_id":9' in logged


def test_worker_generation_skips_target_logits_for_prefill_only_steps(
    capsys,
):
    worker = _load_module(
        "qwen35_native_mtp_tp4_16k_prefill_logits_fixture",
        WORKER_PATH,
    )

    class ModelRunner:
        def __init__(self):
            self.enabled = []
            self.reads = 0

        def enable_step_logits_recording(self, enabled):
            self.enabled.append(enabled)

        def last_step_logits(self):
            self.reads += 1
            return SimpleNamespace(
                tolist=lambda: [[0.0, 4.0, 3.5]],
            )

    class Engine:
        def __init__(self):
            self.model_runner = ModelRunner()
            self.last_step_observation = None
            self.step_index = 0

        def add_request(self, token_ids, sampling_params):
            del token_ids, sampling_params

        def is_finished(self):
            return self.step_index == 2

        def step(self):
            self.step_index += 1
            if self.step_index == 1:
                self.last_step_observation = {
                    "new_completion_tokens_by_seq": {9: []},
                }
                return ([], 0)
            self.last_step_observation = {
                "new_completion_tokens_by_seq": {9: [1]},
            }
            return (
                [(
                    9,
                    list(range(worker.gate.MAX_OUTPUT_TOKENS)),
                )],
                1,
            )

    engine = Engine()
    output_rows, observations = (
        worker.run_generation_with_target_logit_diagnostics(
            engine=engine,
            prompt_rows=[{"token_ids": [11]}],
            sampling_params=object(),
            synchronize=lambda: None,
        )
    )

    assert engine.model_runner.enabled == [True, False]
    assert engine.model_runner.reads == 1
    assert len(output_rows) == 1
    assert len(observations) == 2
    assert "authority_target_logits" not in observations[0]
    assert observations[1]["authority_target_logits"][0][
        "sequence_id"
    ] == 9
    logged = capsys.readouterr().out
    assert logged.count("AUTHORITY_TARGET_LOGITS ") == 1


def test_worker_generation_maps_mixed_step_logits_to_speculative_rows(
    capsys,
):
    worker = _load_module(
        "qwen35_native_mtp_tp4_16k_mixed_logits_fixture",
        WORKER_PATH,
    )

    class ModelRunner:
        def enable_step_logits_recording(self, enabled):
            del enabled

        def last_step_logits(self):
            return SimpleNamespace(
                tolist=lambda: [
                    [0.0, 4.0, 3.5],
                    [8.0, 1.0, 8.5],
                ],
            )

    class Engine:
        def __init__(self):
            self.model_runner = ModelRunner()
            self.last_step_observation = None
            self.finished = False

        def add_request(self, token_ids, sampling_params):
            del token_ids, sampling_params

        def is_finished(self):
            return self.finished

        def step(self):
            self.finished = True
            self.last_step_observation = {
                "speculative_selected_seq_ids": [2, 3],
                "speculative_suppressed_seq_ids": [0, 1],
                "new_completion_tokens_by_seq": {
                    0: [15],
                    1: [16],
                    2: [17, 17],
                    3: [18],
                },
            }
            return (
                [
                    (
                        sequence_id,
                        list(range(worker.gate.MAX_OUTPUT_TOKENS)),
                    )
                    for sequence_id in range(4)
                ],
                5,
            )

    _, observations = (
        worker.run_generation_with_target_logit_diagnostics(
            engine=Engine(),
            prompt_rows=[
                {"token_ids": [sequence_id]}
                for sequence_id in range(4)
            ],
            sampling_params=object(),
            synchronize=lambda: None,
        )
    )

    compact = observations[0]["authority_target_logits"]
    assert [row["sequence_id"] for row in compact] == [2, 3]
    logged = capsys.readouterr().out
    assert '"sequence_id":2' in logged
    assert '"sequence_id":3' in logged
    assert '"sequence_id":0' not in logged


def _token_rows(
    batch_size: int,
    *,
    token_count: int,
    start: int,
) -> list[dict]:
    return [
        {
            "prompt_index": prompt_index,
            "token_count": token_count,
            "token_ids": [
                start + prompt_index + offset % 7
                for offset in range(token_count)
            ],
            "sha256": _digest([
                start + prompt_index + offset % 7
                for offset in range(token_count)
            ]),
        }
        for prompt_index in range(batch_size)
    ]


def _rank_snapshot(rank: int, batch_size: int) -> dict:
    frozen_test = _load_module(
        "qwen35_native_mtp_tp4_4k_fixture",
        FROZEN_TEST_PATH,
    )
    row = deepcopy(
        frozen_test._rank_snapshot(rank, batch_size)
    )
    for transaction in row["executor"][
        "proposal_kv_cache"
    ]["transactions"]:
        if transaction["original_committed_length"] == 4096:
            transaction["original_committed_length"] = 16384
        if transaction["staged_entry_count"] == 4096:
            transaction["staged_entry_count"] = 16384
            transaction["materialized_entry_count"] = 16384
    for ticket in row["executor"][
        "proposal_kv_cache"
    ]["tickets"]:
        if ticket["commit_entry_count"] == 4096:
            ticket["commit_entry_count"] = 16384
    return row


def _baseline_rank_snapshot(rank: int) -> dict:
    frozen_test = _load_module(
        "qwen35_native_mtp_tp4_4k_baseline_fixture",
        FROZEN_TEST_PATH,
    )
    return deepcopy(
        frozen_test._baseline_rank_snapshot(rank)
    )


def _raw_native_snapshot(
    rank: int,
    batch_size: int,
) -> dict:
    row = _rank_snapshot(rank, batch_size)
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
        "executor",
    }
    return {
        name: deepcopy(row[name])
        for name in fields
    }


def _engine_config(batch_size: int, native: bool) -> dict:
    return {
        "tensor_parallel_size": 4,
        "enforce_eager": True,
        "max_model_len": 33024,
        "max_num_batched_tokens": 132096,
        "max_num_prefill_tokens_per_step": 1024,
        "max_num_seqs": batch_size,
        "kvcache_block_size": 256,
        "chunked_prefill_decode_first": False,
        "chunked_prefill_mixed_batch": False,
        "kv_offload_mvp0": True,
        "kv_offload_gpu_blocks": 68,
        "kv_offload_logical_blocks": 640,
        "kv_offload_blockwise_prefill": True,
        "kv_offload_blockwise_decode": True,
        "kv_offload_blockwise_blocks": 8,
        "qwen35_mtp_enabled": native,
        "qwen35_mtp_cuda_graphs": False,
        "qwen35_mtp_max_proposal_tokens": 4,
    }


def _residency_phases(batch_size: int) -> list[dict]:
    sequence_ids = list(range(batch_size))
    committed = [
        [100 + sequence_id, 0]
        for sequence_id in sequence_ids
    ]
    rejected = [
        [200 + sequence_id, 0]
        for sequence_id in sequence_ids
    ]
    result = []
    for operation, status, accepted, discarded in (
        ("prepare", "prepared", [], []),
        ("precommit", "precommitted", committed, rejected),
        ("seal", "sealed", committed, rejected),
    ):
        result.append({
            "ticket_id": 17,
            "operation": operation,
            "status": status,
            "rows": [
                {
                    "ticket_id": 17,
                    "participant_id": rank,
                    "operation": operation,
                    "status": status,
                    "sequence_ids": sequence_ids,
                    "committed_block_identities": accepted,
                    "rejected_block_identities": discarded,
                    "detail": "",
                }
                for rank in range(4)
            ],
        })
    return result


def _movement_rows(*, positive: bool) -> list[dict]:
    movement = 1 if positive else 0
    return [
        {
            "rank": rank,
            "provenance": "engine.kv_offload_summaries",
            "h2d_copies": movement,
            "h2d_bytes": movement * 1024,
            "d2h_copies": movement,
            "d2h_bytes": movement * 1024,
            "copy_waits": movement,
            "evictions": movement,
            "evict_clean": 0,
            "speculative_residency_committed_blocks": (
                movement
            ),
            "speculative_residency_rejected_blocks": (
                movement
            ),
            "speculative_residency_rejected_d2h_copies": 0,
        }
        for rank in range(4)
    ]


def _capacity_rows() -> list[dict]:
    return [
        {
            "rank": rank,
            "provenance": "engine.kv_offload_summaries",
            "gpu_blocks": 68,
            "logical_blocks": 640,
            "resident_blocks": 64,
            "peak_resident_blocks": 68,
        }
        for rank in range(4)
    ]


def _cell(policy: str, batch_size: int) -> dict:
    native = policy == "native_mtp"
    return {
        "schema_version": (
            "qwen35.native-mtp-tp4-16k-target-kv-offload.v1"
        ),
        "policy": policy,
        "batch_size": batch_size,
        "world_size": 4,
        "rank_inventory": [0, 1, 2, 3],
        "gpu_indices": [0, 1, 2, 3],
        "prompt_token_count": 16384,
        "max_output_tokens": 8,
        "max_proposal_tokens": 4,
        "model_identity": {
            "model_type": "qwen3_5",
            "architectures": [
                "Qwen3_5ForConditionalGeneration"
            ],
            "target_model_manifest_sha256": (
                "3e650a908234771c3cf1ac4e20c4d38fe"
                "69982efedaf4a3e631ad0b14aad7dd0"
            ),
            "mtp_checkpoint_manifest_sha256": (
                "9a975bdcf0383774183cae560594dd60"
                "b522b83fe9c4cd595c47c12e2403702b"
            ),
        },
        "engine_config": _engine_config(batch_size, native),
        "prompt_rows": _token_rows(
            batch_size,
            token_count=16384,
            start=17,
        ),
        "output_rows": _token_rows(
            batch_size,
            token_count=8,
            start=101,
        ),
        "rank_snapshots": [
            (
                _rank_snapshot(rank, batch_size)
                if native
                else _baseline_rank_snapshot(rank)
            )
            for rank in range(4)
        ],
        "side_state_receipts": (
            [
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
            if native
            else []
        ),
        "target_kv_receipts": (
            [
                {
                    "sequence_id": sequence_id,
                    "operations": ["prepare", "commit"],
                }
                for sequence_id in range(batch_size)
            ]
            if native
            else []
        ),
        "residency_phases": (
            _residency_phases(batch_size)
            if native
            else []
        ),
        "kv_rank_deltas": _movement_rows(
            positive=native and batch_size == 4
        ),
        "kv_capacity_rows": _capacity_rows(),
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
    return {
        "schema_version": (
            "qwen35.native-mtp-tp4-16k-target-kv-offload.v1"
        ),
        "classification": (
            "QWEN35_NATIVE_MTP_TP4_16K_"
            "TARGET_KV_OFFLOAD_ESTABLISHED"
        ),
        "promotion_classification": "NOT_PROMOTABLE",
        "target_model_manifest_sha256": (
            "3e650a908234771c3cf1ac4e20c4d38fe"
            "69982efedaf4a3e631ad0b14aad7dd0"
        ),
        "mtp_checkpoint_manifest_sha256": (
            "9a975bdcf0383774183cae560594dd60"
            "b522b83fe9c4cd595c47c12e2403702b"
        ),
        "source_tree_sha256": "a" * 64,
        "world_size": 4,
        "rank_inventory": [0, 1, 2, 3],
        "gpu_indices": [0, 1, 2, 3],
        "gpu_process_inventory_before": ["pid:17"],
        "gpu_process_inventory_after": ["pid:17"],
        "cells": {
            f"{policy}:b{batch_size}": _cell(
                policy,
                batch_size,
            )
            for batch_size in (1, 4)
            for policy in ("baseline", "native_mtp")
        },
        "parity": {
            "baseline_native": {
                "b1": True,
                "b4": True,
            },
        },
        "limitations": [
            "phase1_not_promotable",
            "proposal_kv_offload_not_established",
            "tp1_16k_not_established",
            "context_32k_not_established",
            "performance_not_established",
            "kv_quantization_not_established",
            "second_learned_structure_not_established",
        ],
    }


def _make_native_cell_all_accepted(cell: dict) -> None:
    for snapshot in cell["rank_snapshots"]:
        snapshot["accepted_draft_tokens"] = snapshot[
            "proposed_tokens"
        ]
        snapshot["rejected_draft_tokens"] = 0
        transactions = snapshot["executor"][
            "proposal_transactions"
        ]
        tickets = {
            ticket["transaction_id"]: ticket
            for ticket in snapshot["executor"][
                "proposal_kv_cache"
            ]["tickets"]
        }
        for transaction in transactions:
            transaction["accepted_proposal_tokens"] = transaction[
                "exact_q"
            ]
            transaction["rejected_proposal_tokens"] = 0
            ticket = tickets[transaction["transaction_id"]]
            ticket["commit_entry_count"] = transaction[
                "staged_entry_count"
            ]
            ticket["release_entry_count"] = 0


def test_contract_constants_are_frozen():
    gate = _load_module(
        "qwen35_native_mtp_tp4_16k_target_kv_offload_gate",
        GATE_PATH,
    )

    assert gate.SCHEMA_VERSION == (
        "qwen35.native-mtp-tp4-16k-target-kv-offload.v1"
    )
    assert gate.CLASSIFICATION == (
        "QWEN35_NATIVE_MTP_TP4_16K_TARGET_KV_OFFLOAD_ESTABLISHED"
    )
    assert gate.PROMOTION_CLASSIFICATION == "NOT_PROMOTABLE"
    assert gate.POLICIES == ("baseline", "native_mtp")
    assert gate.BATCH_SIZES == (1, 4)
    assert gate.PROMPT_TOKENS == 16384
    assert gate.MAX_OUTPUT_TOKENS == 8
    assert gate.MAX_PROPOSAL_TOKENS == 4
    assert gate.WORLD_SIZE == 4
    assert gate.KV_OFFLOAD_GPU_BLOCKS == 68
    assert gate.KV_OFFLOAD_LOGICAL_BLOCKS == 640
    assert gate.KV_OFFLOAD_BLOCKWISE_BLOCKS == 8
    assert gate.BLOCK_SIZE == 256
    assert "tp1_authority_sha256" not in gate.RESULT_FIELDS
    assert "tp1_output_rows" not in gate.CELL_FIELDS


def test_gate_exposes_strict_integer_helper_for_worker_reuse():
    gate = _load_module(
        "qwen35_native_mtp_tp4_16k_integer_gate",
        GATE_PATH,
    )

    assert gate._integer(7, "value") == 7
    with pytest.raises(ValueError, match="must be an integer"):
        gate._integer(True, "value")
    with pytest.raises(ValueError, match="must be an integer"):
        gate._integer(7.0, "value")


def test_loading_16k_gate_does_not_mutate_frozen_4k_source():
    before = _sha256(FROZEN_GATE_PATH)

    _load_module(
        "qwen35_native_mtp_tp4_16k_target_kv_offload_gate_isolation",
        GATE_PATH,
    )

    assert _sha256(FROZEN_GATE_PATH) == before


@pytest.mark.parametrize(
    ("policy", "batch_size", "native"),
    (
        ("baseline", 1, False),
        ("native_mtp", 4, True),
    ),
)
def test_worker_uses_frozen_long_context_configuration(
    policy,
    batch_size,
    native,
):
    worker = _load_module(
        "qwen35_native_mtp_tp4_16k_worker_config",
        WORKER_PATH,
    )

    kwargs = worker.engine_kwargs(
        policy=policy,
        batch_size=batch_size,
    )

    assert kwargs == _engine_config(batch_size, native)


def test_worker_builds_deterministic_challenge_tail_prompts():
    worker = _load_module(
        "qwen35_native_mtp_tp4_16k_worker_prompts",
        WORKER_PATH,
    )

    class FakeTokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            seed = sum(ord(character) for character in text)
            return [
                10 + (seed + 37 * offset) % 997
                for offset in range(16)
            ]

    tokenizer = FakeTokenizer()
    first = worker.build_prompt_rows(tokenizer, 4)
    second = worker.build_prompt_rows(tokenizer, 4)
    frozen = worker.tp1_worker.build_prompt_rows(tokenizer, 4)

    assert first == second
    assert worker.CHALLENGE_TAIL_TOKENS == 1024
    assert len({row["sha256"] for row in first}) == 4
    for prompt_index, row in enumerate(first):
        assert row["prompt_index"] == prompt_index
        assert row["token_count"] == 16384
        assert len(row["token_ids"]) == 16384
        assert row["sha256"] == _digest(row["token_ids"])
        assert row["token_ids"][-1024:] != frozen[
            prompt_index
        ]["token_ids"][-1024:]
        assert len(set(row["token_ids"][-1024:])) > 8
    assert worker.run_policy_cell.__kwdefaults__[
        "prompt_builder"
    ] is worker.build_prompt_rows


def test_worker_projects_production_movement_and_capacity():
    worker = _load_module(
        "qwen35_native_mtp_tp4_16k_worker_movement",
        WORKER_PATH,
    )
    before = tuple(
        {
            **{
                name: 0
                for name in worker.gate.MOVEMENT_KEYS
            },
            "gpu_blocks": 68,
            "logical_blocks": 640,
            "resident_blocks": 0,
            "peak_resident_blocks": 0,
        }
        for _ in range(4)
    )
    after = tuple(
        {
            **{
                name: rank + 1
                for name in worker.gate.MOVEMENT_KEYS
            },
            "speculative_residency_rejected_d2h_copies": 0,
            "gpu_blocks": 68,
            "logical_blocks": 640,
            "resident_blocks": 64,
            "peak_resident_blocks": 68,
        }
        for rank in range(4)
    )

    movement = worker.movement_delta(before, after)
    capacity = worker.capacity_rows(after)

    assert movement[3]["h2d_copies"] == 4
    assert movement[0]["provenance"] == (
        "engine.kv_offload_summaries"
    )
    assert capacity[0] == {
        "rank": 0,
        "provenance": "engine.kv_offload_summaries",
        "gpu_blocks": 68,
        "logical_blocks": 640,
        "resident_blocks": 64,
        "peak_resident_blocks": 68,
    }


def test_worker_cli_has_no_tp1_authority_argument():
    worker = _load_module(
        "qwen35_native_mtp_tp4_16k_worker_cli",
        WORKER_PATH,
    )

    args = worker.parse_args([
        "--model",
        "/checkpoint",
        "--gpu-indices",
        "0,1,2,3",
        "--policy",
        "native_mtp",
        "--batch-size",
        "4",
        "--dist-port",
        "29640",
        "--master-port",
        "29740",
        "--out",
        "/tmp/result.json",
    ])

    assert not hasattr(args, "tp1_result")


def test_native_rank_normalization_failure_reports_runtime_counters():
    worker = _load_module(
        "qwen35_native_mtp_tp4_16k_worker_diagnostics",
        WORKER_PATH,
    )
    runtime = {
        "target_prefill_observations": 1,
        "proposal_rows": 4,
        "proposed_tokens": 7,
        "accepted_draft_tokens": 5,
        "rejected_draft_tokens": 2,
        "first_target_callbacks": 3,
        "verify_callbacks": 2,
        "first_target_target_forwards": 4,
        "verify_target_forwards": 2,
        "accepted_prefix_target_replays": 0,
    }

    with pytest.raises(ValueError) as error:
        worker.normalize_rank_snapshots(
            (),
            policy="native_mtp",
            batch_size=1,
            runtime=runtime,
            finalize_ack_ranks=(1, 2, 3),
            release_ack_ranks=(1, 2, 3),
        )

    message = str(error.value)
    for fragment in (
        "proposal_rows=4",
        "proposed_tokens=7",
        "accepted_draft_tokens=5",
        "rejected_draft_tokens=2",
        "first_target_callbacks=3",
        "verify_callbacks=2",
        "first_target_target_forwards=4",
        "verify_target_forwards=2",
        "accepted_prefix_target_replays=0",
    ):
        assert fragment in message


def test_gate_accepts_batch_native_fixed_q_verifier_splits():
    gate = _load_module(
        "qwen35_native_mtp_tp4_16k_fixed_q_splits",
        GATE_PATH,
    )
    rows = [
        _rank_snapshot(rank, 4)
        for rank in range(4)
    ]
    for row in rows:
        row["verify_callbacks"] = 2
        row["verify_target_forwards"] = 4

    normalized = gate._validate_rank_snapshots(
        rows,
        policy="native_mtp",
        batch_size=4,
    )

    assert [
        row["verify_target_forwards"]
        for row in normalized
    ] == [4, 4, 4, 4]


def test_gate_canonicalizes_release_rows_by_sequence_id():
    gate = _load_module(
        "qwen35_native_mtp_tp4_16k_release_order",
        GATE_PATH,
    )
    rows = [
        _rank_snapshot(rank, 4)
        for rank in range(4)
    ]
    for row in rows:
        row["executor"]["release_rows"].reverse()

    normalized = gate._validate_rank_snapshots(
        rows,
        policy="native_mtp",
        batch_size=4,
    )

    assert normalized[0]["executor"]["release_rows"] == [
        {
            "sequence_id": sequence_id,
            "sequence_epoch": 0,
        }
        for sequence_id in range(4)
    ]


def test_worker_does_not_count_suppressed_decode_as_prefix_replay(
    monkeypatch,
):
    worker = _load_module(
        "qwen35_native_mtp_tp4_16k_replay_accounting",
        WORKER_PATH,
    )
    monkeypatch.setattr(
        worker.tp1_worker,
        "summarize_runtime",
        lambda _observations, **_kwargs: {
            "accepted_prefix_target_replays": 1,
        },
    )
    observations = [
        {
            "speculative_selected_seq_ids": [0],
            "speculative_suppressed_seq_ids": [1],
            "speculative_accepted_draft_token_counts": {
                0: 2,
            },
            "authority_normal_decode_target_forward_calls": 1,
        },
    ]

    runtime = worker.summarize_runtime(
        observations,
        capture={},
        native_binding={"registered": True},
    )

    assert runtime["accepted_prefix_target_replays"] == 0


def test_worker_runs_baseline_cell_with_real_summary_interfaces():
    worker = _load_module(
        "qwen35_native_mtp_tp4_16k_worker_baseline_cell",
        WORKER_PATH,
    )
    captured_kwargs = {}

    class FakeEngine:
        def __init__(self):
            self.tokenizer = object()
            self.speculative_runtime_poisoned = False
            self._summary_calls = 0
            self._call_speculative_residency_phase = (
                lambda *args, **kwargs: ()
            )

        def kv_offload_summaries(self, timeout_s):
            assert timeout_s == 60.0
            self._summary_calls += 1
            resident = 0 if self._summary_calls == 1 else 64
            peak = 0 if self._summary_calls == 1 else 68
            return tuple(
                {
                    **{
                        name: 0
                        for name in worker.gate.MOVEMENT_KEYS
                    },
                    "gpu_blocks": 68,
                    "logical_blocks": 640,
                    "resident_blocks": resident,
                    "peak_resident_blocks": peak,
                }
                for _ in range(4)
            )

        def flush_pending_hybrid_state_releases(
            self,
            timeout_s,
        ):
            assert timeout_s == 60.0

        def exit(self):
            return {
                "rank_exit_codes": [0, 0, 0, 0],
                "process_group_destroyed": True,
                "owned_children_remaining": [],
            }

    def engine_factory(model_path, **kwargs):
        assert model_path == "/checkpoint"
        captured_kwargs.update(kwargs)
        return FakeEngine()

    def run_generation_fn(**kwargs):
        assert len(kwargs["prompt_rows"][0]["token_ids"]) == 16384
        return (
            _token_rows(1, token_count=8, start=101),
            [],
        )

    cell = worker.run_policy_cell(
        model_path="/checkpoint",
        gpu_indices=(0, 1, 2, 3),
        policy="baseline",
        batch_size=1,
        dist_port=29640,
        master_port=29740,
        engine_factory=engine_factory,
        sampling_params_type=lambda **kwargs: SimpleNamespace(
            **kwargs
        ),
        runtime_type=object,
        synchronize=lambda: None,
        run_generation_fn=run_generation_fn,
        target_manifest_resolver=lambda _path: (
            worker.gate.TARGET_MODEL_MANIFEST_SHA256
        ),
        mtp_manifest_resolver=lambda _path: (
            worker.gate.MTP_CHECKPOINT_MANIFEST_SHA256
        ),
        prompt_builder=lambda _tokenizer, batch_size: (
            _token_rows(
                batch_size,
                token_count=16384,
                start=17,
            )
        ),
        rank_snapshot_collector=lambda _engine: tuple(
            {
                "rank": rank,
                "world_size": 4,
                "registered": False,
                "executor": None,
            }
            for rank in range(4)
        ),
        model_identity_fn=lambda _engine, **kwargs: {
            "model_type": "qwen3_5",
            "architectures": [
                "Qwen3_5ForConditionalGeneration"
            ],
            "target_model_manifest_sha256": kwargs[
                "target_digest"
            ],
            "mtp_checkpoint_manifest_sha256": kwargs[
                "mtp_digest"
            ],
        },
    )

    assert captured_kwargs == _engine_config(1, False)
    assert cell["policy"] == "baseline"
    assert cell["kv_capacity_rows"][0][
        "peak_resident_blocks"
    ] == 68
    assert cell["cleanup"]["engine_exit_called"] is True


def test_worker_runs_native_cell_with_residency_and_movement(
    monkeypatch,
):
    worker = _load_module(
        "qwen35_native_mtp_tp4_16k_worker_native_cell",
        WORKER_PATH,
    )
    phases = _residency_phases(4)
    phase_by_operation = {
        phase["operation"]: phase
        for phase in phases
    }

    class FakeEngine:
        def __init__(self):
            self.tokenizer = object()
            self.model_runner = object()
            self.speculative_runtime_poisoned = False
            self.speculative_proposal_lifecycle_ack_rows = []
            self._summary_calls = 0
            self.runtime = None

        def _call_speculative_residency_phase(
            self,
            _method_name,
            _ticket_id,
            *,
            expected_operation,
            **_kwargs,
        ):
            return phase_by_operation[
                expected_operation
            ]["rows"]

        def activate_speculative_runtime(self, runtime):
            self.runtime = runtime

        def kv_offload_summaries(self, timeout_s):
            assert timeout_s == 60.0
            self._summary_calls += 1
            movement = 0 if self._summary_calls == 1 else 1
            return tuple(
                {
                    **{
                        name: movement
                        for name in worker.gate.MOVEMENT_KEYS
                    },
                    "speculative_residency_rejected_d2h_copies": 0,
                    "gpu_blocks": 68,
                    "logical_blocks": 640,
                    "resident_blocks": (
                        0 if movement == 0 else 64
                    ),
                    "peak_resident_blocks": (
                        0 if movement == 0 else 68
                    ),
                }
                for _ in range(4)
            )

        def flush_pending_hybrid_state_releases(
            self,
            timeout_s,
        ):
            assert timeout_s == 60.0

        def exit(self):
            return {
                "rank_exit_codes": [0, 0, 0, 0],
                "process_group_destroyed": True,
                "owned_children_remaining": [],
            }

    @contextmanager
    def fake_capture(_engine, _executor):
        receipts = []
        for sequence_id in range(4):
            for operation in (
                "prepare",
                "select",
                "apply",
                "seal",
            ):
                receipts.append({
                    "sequence_id": sequence_id,
                    "operation": operation,
                })
        yield {
            "method_names": [],
            "ordinary_decode_target_forward_calls": 0,
            "proposal_finalize_receipts": [],
            "side_state_receipts": receipts,
            "proposal_kv_receipts": [],
            "lifecycle_events": [],
        }

    monkeypatch.setattr(
        worker.tp1_worker,
        "validate_native_registration",
        lambda _model_runner: (
            "descriptor",
            object(),
            "executor",
            object(),
        ),
    )
    monkeypatch.setattr(
        worker.tp1_worker,
        "capture_runtime_receipts",
        fake_capture,
    )
    monkeypatch.setattr(
        worker.tp1_worker,
        "summarize_runtime",
        lambda _observations, **_kwargs: {
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
        },
    )
    monkeypatch.setattr(
        worker,
        "_ack_ranks",
        lambda _engine, _fragment: (1, 2, 3),
    )

    engine = FakeEngine()

    def run_generation_fn(**kwargs):
        for phase in phases:
            kwargs["engine"]._call_speculative_residency_phase(
                f"{phase['operation']}_speculative_residency_batch",
                phase["ticket_id"],
                expected_operation=phase["operation"],
                expected_status=phase["status"],
            )
        return (
            _token_rows(4, token_count=8, start=101),
            [],
        )

    cell = worker.run_policy_cell(
        model_path="/checkpoint",
        gpu_indices=(0, 1, 2, 3),
        policy="native_mtp",
        batch_size=4,
        dist_port=29640,
        master_port=29740,
        engine_factory=lambda _path, **_kwargs: engine,
        sampling_params_type=lambda **kwargs: SimpleNamespace(
            **kwargs
        ),
        runtime_type=lambda **kwargs: SimpleNamespace(**kwargs),
        synchronize=lambda: None,
        run_generation_fn=run_generation_fn,
        target_manifest_resolver=lambda _path: (
            worker.gate.TARGET_MODEL_MANIFEST_SHA256
        ),
        mtp_manifest_resolver=lambda _path: (
            worker.gate.MTP_CHECKPOINT_MANIFEST_SHA256
        ),
        prompt_builder=lambda _tokenizer, batch_size: (
            _token_rows(
                batch_size,
                token_count=16384,
                start=17,
            )
        ),
        rank_snapshot_collector=lambda _engine: tuple(
            _raw_native_snapshot(rank, 4)
            for rank in range(4)
        ),
        model_identity_fn=lambda _engine, **kwargs: {
            "model_type": "qwen3_5",
            "architectures": [
                "Qwen3_5ForConditionalGeneration"
            ],
            "target_model_manifest_sha256": kwargs[
                "target_digest"
            ],
            "mtp_checkpoint_manifest_sha256": kwargs[
                "mtp_digest"
            ],
        },
    )

    assert engine.runtime.model_runner_executor == "descriptor"
    assert [
        phase["operation"]
        for phase in cell["residency_phases"]
    ] == ["prepare", "precommit", "seal"]
    assert sum(
        row["h2d_copies"]
        for row in cell["kv_rank_deltas"]
    ) == 4


def test_complete_authority_is_accepted_and_canonical():
    gate = _load_module(
        "qwen35_native_mtp_tp4_16k_accept",
        GATE_PATH,
    )
    result = _result()

    assert gate.validate_result(result) == result


def test_native_batch1_all_accepted_is_valid_when_campaign_has_rejection():
    gate = _load_module(
        "qwen35_native_mtp_tp4_16k_all_accepted_b1",
        GATE_PATH,
    )
    result = _result()
    _make_native_cell_all_accepted(
        result["cells"]["native_mtp:b1"]
    )

    normalized = gate.validate_result(result)

    assert normalized["cells"]["native_mtp:b1"][
        "rank_snapshots"
    ][0]["rejected_draft_tokens"] == 0
    assert normalized["cells"]["native_mtp:b4"][
        "rank_snapshots"
    ][0]["rejected_draft_tokens"] > 0


def test_native_campaign_rejects_zero_rejected_draft_tokens():
    gate = _load_module(
        "qwen35_native_mtp_tp4_16k_no_rejection",
        GATE_PATH,
    )
    result = _result()
    _make_native_cell_all_accepted(
        result["cells"]["native_mtp:b1"]
    )
    _make_native_cell_all_accepted(
        result["cells"]["native_mtp:b4"]
    )

    with pytest.raises(
        ValueError,
        match="native campaign requires rejected draft tokens",
    ):
        gate.validate_result(result)


def _zero_native_b4_counter(result: dict, name: str) -> None:
    for row in result["cells"]["native_mtp:b4"][
        "kv_rank_deltas"
    ]:
        row[name] = 0


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda result: result["cells"]["native_mtp:b4"][
                "kv_rank_deltas"
            ][0].update(provenance="synthetic_copy"),
            "KV movement provenance is invalid",
        ),
        (
            lambda result: _zero_native_b4_counter(
                result,
                "d2h_copies",
            ),
            "native batch-4 requires real target-KV D2H copies",
        ),
        (
            lambda result: _zero_native_b4_counter(
                result,
                "d2h_bytes",
            ),
            "native batch-4 requires real target-KV D2H bytes",
        ),
        (
            lambda result: _zero_native_b4_counter(
                result,
                "h2d_copies",
            ),
            "native batch-4 requires real target-KV H2D copies",
        ),
        (
            lambda result: _zero_native_b4_counter(
                result,
                "h2d_bytes",
            ),
            "native batch-4 requires real target-KV H2D bytes",
        ),
        (
            lambda result: result["cells"]["native_mtp:b4"][
                "kv_capacity_rows"
            ][0].update(gpu_blocks=69),
            "target-KV GPU block capacity mismatch",
        ),
        (
            lambda result: result["cells"]["native_mtp:b4"][
                "kv_capacity_rows"
            ][0].update(logical_blocks=639),
            "target-KV logical block capacity mismatch",
        ),
        (
            lambda result: result["cells"]["native_mtp:b4"][
                "kv_capacity_rows"
            ][0].update(peak_resident_blocks=69),
            "target-KV peak residency exceeds GPU capacity",
        ),
        (
            lambda result: result["cells"]["native_mtp:b4"][
                "kv_capacity_rows"
            ][0].update(resident_blocks=69),
            "target-KV resident blocks exceed GPU capacity",
        ),
        (
            lambda result: result.update(
                tp1_authority_sha256="b" * 64
            ),
            "result fields mismatch",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"].update(
                tp1_output_rows=[]
            ),
            "cell fields mismatch",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "prompt_rows"
            ].__setitem__(
                0,
                _token_rows(
                    1,
                    token_count=16384,
                    start=777,
                )[0],
            ),
            "baseline/native output parity mismatch",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "output_rows"
            ].__setitem__(
                0,
                _token_rows(
                    1,
                    token_count=8,
                    start=888,
                )[0],
            ),
            "baseline/native output parity mismatch",
        ),
        (
            lambda result: result["parity"].update(
                tp1_tp4_native={"b1": True, "b4": True}
            ),
            "parity summary mismatch",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "rank_snapshots"
            ][0]["executor"]["proposal_kv_cache"][
                "transactions"
            ][0].update(staged_entry_count=4096),
            "proposal KV bootstrap transaction mismatch",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "rank_snapshots"
            ][0].update(accepted_prefix_target_replays=1),
            "accepted-prefix target replay detected",
        ),
        (
            lambda result: result["cells"]["native_mtp:b1"][
                "residency_phases"
            ][1].update(operation="seal"),
            "residency phase order mismatch",
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
    ),
)
def test_invalid_authority_is_rejected(mutate, match):
    gate = _load_module(
        "qwen35_native_mtp_tp4_16k_reject",
        GATE_PATH,
    )
    result = _result()
    mutate(result)

    with pytest.raises(ValueError, match=match):
        gate.validate_result(result)


def test_default_source_inventory_binds_all_runtime_helpers():
    gate = _load_module(
        "qwen35_native_mtp_tp4_16k_source_inventory",
        GATE_PATH,
    )
    expected = tuple(sorted(
        [
            str(path.relative_to(TOOLS.parent))
            for path in (
                TOOLS.parent / "tinyvllm"
            ).rglob("*.py")
        ]
        + [
            "tools/qwen35_generic_speculative_tp4_gate.py",
            "tools/qwen35_native_mtp_tp4_4k_engine_gate.py",
            "tools/qwen35_native_mtp_tp4_4k_engine_worker.py",
            "tools/qwen35_native_mtp_tp1_4k_engine_gate.py",
            "tools/qwen35_native_mtp_tp1_4k_engine_worker.py",
            (
                "tools/qwen35_native_mtp_tp4_16k_"
                "target_kv_offload_gate.py"
            ),
            (
                "tools/qwen35_native_mtp_tp4_16k_"
                "target_kv_offload_worker.py"
            ),
            (
                "tools/verify_qwen35_native_mtp_tp4_16k_"
                "target_kv_offload_gate.py"
            ),
        ]
    ))

    assert gate.DEFAULT_SOURCE_FILES == expected


def test_gpu_process_inventory_is_scoped_to_selected_gpus(
    monkeypatch,
):
    gate = _load_module(
        "qwen35_native_mtp_tp4_16k_gpu_inventory",
        GATE_PATH,
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "gpu-b, 2, python, 20\n"
                "gpu-a, 1, python, 10\n"
            ),
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


def test_campaign_runs_four_cells_without_tp1_and_binds_sources(
    tmp_path,
    monkeypatch,
):
    gate = _load_module(
        "qwen35_native_mtp_tp4_16k_campaign",
        GATE_PATH,
    )
    repo_root = tmp_path / "source"
    repo_root.mkdir()
    source_path = repo_root / "engine.py"
    source_path.write_text("engine\n", encoding="utf-8")
    worker_script = repo_root / "worker.py"
    worker_script.write_text("", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        assert "--tp1-result" not in command
        policy = command[command.index("--policy") + 1]
        batch_size = int(
            command[command.index("--batch-size") + 1]
        )
        output_path = Path(
            command[command.index("--out") + 1]
        )
        assert (
            output_path.parents[1] / "source_manifest.json"
        ).is_file()
        gate.atomic_write_json(
            output_path,
            _cell(policy, batch_size),
        )
        calls.append((policy, batch_size))
        return SimpleNamespace(returncode=0)

    worker_module = SimpleNamespace(
        target_model_manifest_sha256=(
            lambda _model_path: (
                gate.TARGET_MODEL_MANIFEST_SHA256
            )
        ),
        mtp_checkpoint_manifest_sha256=(
            lambda _model_path: (
                gate.MTP_CHECKPOINT_MANIFEST_SHA256
            )
        ),
    )
    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    monkeypatch.setattr(
        gate,
        "_load_module",
        lambda _name, _path: worker_module,
    )
    inventory_calls = []

    def inventory(indices):
        inventory_calls.append(indices)
        return ["pid:17"]

    output_dir = tmp_path / "authority"
    result = gate.run_campaign(
        model_path="/checkpoint",
        gpu_indices=(0, 1, 2, 3),
        output_dir=output_dir,
        repo_root=repo_root,
        worker_script=worker_script,
        source_files=("engine.py",),
        gpu_process_inventory=inventory,
        verifier=lambda run_dir, source_root: (
            {
                "classification": "PASS",
                "failures": [],
            }
            if not output_dir.exists()
            and source_root == repo_root
            and (run_dir / "source_manifest.json").is_file()
            else {
                "classification": "FAIL",
                "failures": ["premature publication"],
            }
        ),
        dist_port_base=29640,
        master_port_base=29740,
    )

    assert calls == [
        ("baseline", 1),
        ("native_mtp", 1),
        ("baseline", 4),
        ("native_mtp", 4),
    ]
    assert inventory_calls == [
        (0, 1, 2, 3),
        (0, 1, 2, 3),
    ]
    assert result["target_model_manifest_sha256"] == (
        gate.TARGET_MODEL_MANIFEST_SHA256
    )
    assert result["mtp_checkpoint_manifest_sha256"] == (
        gate.MTP_CHECKPOINT_MANIFEST_SHA256
    )
    assert output_dir.is_dir()


def test_campaign_rejects_inventory_change_and_existing_output(
    tmp_path,
    monkeypatch,
):
    gate = _load_module(
        "qwen35_native_mtp_tp4_16k_campaign_reject",
        GATE_PATH,
    )
    repo_root = tmp_path / "source"
    repo_root.mkdir()
    source_path = repo_root / "engine.py"
    source_path.write_text("engine\n", encoding="utf-8")
    worker_script = repo_root / "worker.py"
    worker_script.write_text("", encoding="utf-8")

    def fake_run(command, **_kwargs):
        policy = command[command.index("--policy") + 1]
        batch_size = int(
            command[command.index("--batch-size") + 1]
        )
        gate.atomic_write_json(
            Path(command[command.index("--out") + 1]),
            _cell(policy, batch_size),
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    monkeypatch.setattr(
        gate,
        "_load_module",
        lambda _name, _path: SimpleNamespace(
            target_model_manifest_sha256=(
                lambda _model_path: (
                    gate.TARGET_MODEL_MANIFEST_SHA256
                )
            ),
            mtp_checkpoint_manifest_sha256=(
                lambda _model_path: (
                    gate.MTP_CHECKPOINT_MANIFEST_SHA256
                )
            ),
        ),
    )
    inventories = iter((["selected:17"], ["selected:18"]))
    output_dir = tmp_path / "authority"

    with pytest.raises(
        RuntimeError,
        match="GPU process inventory changed",
    ):
        gate.run_campaign(
            model_path="/checkpoint",
            gpu_indices=(0, 1, 2, 3),
            output_dir=output_dir,
            repo_root=repo_root,
            worker_script=worker_script,
            source_files=("engine.py",),
            gpu_process_inventory=lambda _indices: next(
                inventories
            ),
            verifier=lambda _run_dir, _source_root: {
                "classification": "PASS",
                "failures": [],
            },
            dist_port_base=29640,
            master_port_base=29740,
        )

    assert not output_dir.exists()
    assert output_dir.with_name("authority.failed").is_dir()

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(
        ValueError,
        match="authority output path already exists",
    ):
        gate.run_campaign(
            model_path="/checkpoint",
            gpu_indices=(0, 1, 2, 3),
            output_dir=existing,
            repo_root=repo_root,
            worker_script=worker_script,
            source_files=("engine.py",),
            dist_port_base=29640,
            master_port_base=29740,
        )


def test_campaign_does_not_publish_before_verifier_pass(
    tmp_path,
    monkeypatch,
):
    gate = _load_module(
        "qwen35_native_mtp_tp4_16k_campaign_verify",
        GATE_PATH,
    )
    repo_root = tmp_path / "source"
    repo_root.mkdir()
    source_path = repo_root / "engine.py"
    source_path.write_text("engine\n", encoding="utf-8")
    worker_script = repo_root / "worker.py"
    worker_script.write_text("", encoding="utf-8")

    def fake_run(command, **_kwargs):
        policy = command[command.index("--policy") + 1]
        batch_size = int(
            command[command.index("--batch-size") + 1]
        )
        gate.atomic_write_json(
            Path(command[command.index("--out") + 1]),
            _cell(policy, batch_size),
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    monkeypatch.setattr(
        gate,
        "_load_module",
        lambda _name, _path: SimpleNamespace(
            target_model_manifest_sha256=(
                lambda _model_path: (
                    gate.TARGET_MODEL_MANIFEST_SHA256
                )
            ),
            mtp_checkpoint_manifest_sha256=(
                lambda _model_path: (
                    gate.MTP_CHECKPOINT_MANIFEST_SHA256
                )
            ),
        ),
    )
    output_dir = tmp_path / "authority"

    with pytest.raises(
        RuntimeError,
        match="independent verification failed",
    ):
        gate.run_campaign(
            model_path="/checkpoint",
            gpu_indices=(0, 1, 2, 3),
            output_dir=output_dir,
            repo_root=repo_root,
            worker_script=worker_script,
            source_files=("engine.py",),
            gpu_process_inventory=lambda _indices: [],
            verifier=lambda run_dir, _source_root: (
                {
                    "classification": "FAIL",
                    "failures": ["tampered"],
                }
                if not output_dir.exists()
                and run_dir.is_dir()
                else {
                    "classification": "FAIL",
                    "failures": ["premature publication"],
                }
            ),
            dist_port_base=29640,
            master_port_base=29740,
        )

    assert not output_dir.exists()
    assert output_dir.with_name("authority.failed").is_dir()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _published_run(tmp_path: Path):
    gate = _load_module(
        "qwen35_native_mtp_tp4_16k_publish_fixture",
        GATE_PATH,
    )
    result = gate.validate_result(deepcopy(_result()))
    source_files = {
        name: "b" * 64
        for name in gate.DEFAULT_SOURCE_FILES
    }
    result["source_tree_sha256"] = gate.source_hashes_sha256(
        source_files
    )
    run_dir = tmp_path / "authority"
    gate.publish_authority(
        run_dir,
        result,
        source_files=source_files,
    )
    verifier = _load_module(
        "verify_qwen35_native_mtp_tp4_16k_fixture",
        VERIFIER_PATH,
    )
    return gate, verifier, run_dir


def _mutate_result(run_dir: Path, mutate) -> None:
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    mutate(result)
    _write_json(result_path, result)


def test_complete_published_run_verifier_passes(tmp_path):
    _gate, verifier, run_dir = _published_run(tmp_path)

    assert verifier.verify_run(run_dir) == {
        "classification": "PASS",
        "failures": [],
    }


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda run_dir: _write_json(
                run_dir / "result.json",
                {},
            ),
            "result",
        ),
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: result.update(
                    target_model_manifest_sha256="0" * 64
                ),
            ),
            "target model manifest",
        ),
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: result.update(
                    mtp_checkpoint_manifest_sha256="0" * 64
                ),
            ),
            "MTP checkpoint manifest",
        ),
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: result.update(
                    source_tree_sha256="0" * 64
                ),
            ),
            "source tree",
        ),
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: result.update(
                    schema_version="tampered"
                ),
            ),
            "schema",
        ),
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: result.update(
                    classification="tampered"
                ),
            ),
            "classification",
        ),
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: result["cells"][
                    "native_mtp:b4"
                ]["engine_config"].update(
                    kv_offload_gpu_blocks=69
                ),
            ),
            "engine configuration",
        ),
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: result["cells"][
                    "native_mtp:b1"
                ]["output_rows"][0].update(
                    token_ids=[7] * 8,
                    sha256=_digest([7] * 8),
                ),
            ),
            "baseline/native output parity",
        ),
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: result["cells"][
                    "native_mtp:b4"
                ]["kv_rank_deltas"][0].update(
                    provenance="synthetic_copy"
                ),
            ),
            "movement provenance",
        ),
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: _zero_native_b4_counter(
                    result,
                    "d2h_copies",
                ),
            ),
            "D2H copies",
        ),
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: _zero_native_b4_counter(
                    result,
                    "h2d_bytes",
                ),
            ),
            "H2D bytes",
        ),
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: result["cells"][
                    "native_mtp:b4"
                ]["kv_capacity_rows"][0].update(
                    peak_resident_blocks=69
                ),
            ),
            "peak residency",
        ),
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: result["cells"][
                    "native_mtp:b1"
                ]["rank_snapshots"][0]["executor"][
                    "proposal_transactions"
                ].pop(),
            ),
            "transaction",
        ),
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: result["cells"][
                    "native_mtp:b1"
                ]["residency_phases"][1].update(
                    operation="seal"
                ),
            ),
            "residency",
        ),
        (
            lambda run_dir: _mutate_result(
                run_dir,
                lambda result: result["cells"][
                    "native_mtp:b1"
                ]["cleanup"].update(
                    rank_exit_codes=[0, 0, 0, 1]
                ),
            ),
            "rank exit",
        ),
    ),
)
def test_published_run_tamper_fails_closed(
    tmp_path,
    mutate,
    match,
):
    _gate, verifier, run_dir = _published_run(tmp_path)
    mutate(run_dir)

    verified = verifier.verify_run(run_dir)

    assert verified["classification"] == "FAIL"
    assert any(
        match.lower() in failure.lower()
        for failure in verified["failures"]
    )


def test_verifier_rejects_changed_bound_source(tmp_path):
    gate, verifier, run_dir = _published_run(tmp_path)
    source_root = tmp_path / "source"
    for name in gate.DEFAULT_SOURCE_FILES:
        path = source_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source\n", encoding="utf-8")
    source_files = gate.hash_source_files(
        source_root,
        gate.DEFAULT_SOURCE_FILES,
    )
    manifest_path = run_dir / "source_manifest.json"
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    manifest["source_files"] = source_files
    manifest["source_tree_sha256"] = (
        gate.source_hashes_sha256(source_files)
    )
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["source_tree_sha256"] = manifest[
        "source_tree_sha256"
    ]
    _write_json(result_path, result)
    manifest["artifacts"]["result.json"] = gate.sha256_file(
        result_path
    )
    _write_json(manifest_path, manifest)
    changed = source_root / gate.DEFAULT_SOURCE_FILES[0]
    changed.write_text("changed\n", encoding="utf-8")

    verified = verifier.verify_run(
        run_dir,
        source_root=source_root,
    )

    assert verified["classification"] == "FAIL"
    assert any(
        "source file digest mismatch" in failure
        for failure in verified["failures"]
    )


def test_remote_runner_is_bounded_and_source_bound():
    text = REMOTE_RUNNER_PATH.read_text(
        encoding="utf-8"
    )

    for required in (
        "sitian@10.232.195.203",
        "FILE:/Users/bytedance/krb5cc_sitian",
        "ControlMaster=no",
        "ControlPath=none",
        (
            "qwen35_native_mtp_tp4_16k_"
            "target_kv_offload_gate.py"
        ),
        "qwen35_generic_speculative_tp4_gate.py",
        (
            "qwen35_native_mtp_tp4_16k_"
            "target_kv_offload_worker.py"
        ),
        (
            "verify_qwen35_native_mtp_tp4_16k_"
            "target_kv_offload_gate.py"
        ),
        "qwen35_native_mtp_tp4_16k_target_kv_offload",
        "campaign.status",
        "campaign.pid",
        "campaign.exit_code",
        "authority.failed",
        "REMOTE_COMMAND_RETRY_ATTEMPTS",
        "REMOTE_RSYNC_RETRY_ATTEMPTS",
        "POLL_INTERVAL_SECONDS",
        "MAX_POLL_ATTEMPTS",
    ):
        assert required in text
    for forbidden in (
        "pkill",
        "killall",
        "nvidia-smi --gpu-reset",
        "git clean",
        "while true",
        "ControlMaster=yes",
    ):
        assert forbidden not in text
    assert "head -n 4" in text
    assert "campaign already terminal" in text
    assert "campaign already running" in text
    assert "verify.local.json" in text


def test_remote_runner_freezes_gpu_selection_after_remote_preflight():
    text = REMOTE_RUNNER_PATH.read_text(
        encoding="utf-8"
    )

    source_upload = text.index(
        'retry_remote_rsync \\\n'
        '  "${source_tar}"'
    )
    source_unpack = text.index(
        "tar -xf '${REMOTE_RUN}/source.tar'"
    )
    manifest_preflight = text.index(
        "target_actual = "
        "worker.target_model_manifest_sha256"
    )
    gpu_inventory = text.index('gpu_inventory="$(')
    selected_gpu_record = text.index(
        '> "${LOCAL_RUN}/selected_gpu_indices.txt"'
    )
    campaign_generation = text.index(
        'cat > "${LOCAL_RUN}/campaign.sh"'
    )

    assert text.count(
        "nvidia-smi --query-gpu="
        "index,memory.free,memory.total,utilization.gpu"
    ) == 1
    assert (
        source_upload
        < source_unpack
        < manifest_preflight
        < gpu_inventory
        < selected_gpu_record
        < campaign_generation
    )
