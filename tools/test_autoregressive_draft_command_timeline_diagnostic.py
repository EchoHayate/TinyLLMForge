from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tools"
    / "autoregressive_draft_command_timeline_diagnostic.py"
)
assert MODULE_PATH.exists(), f"missing module: {MODULE_PATH}"
SPEC = importlib.util.spec_from_file_location(
    "autoregressive_draft_command_timeline_diagnostic_test_module",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnostic
SPEC.loader.exec_module(diagnostic)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _clock() -> dict:
    return {
        "boot_id": "boot-task5",
        "implementation": "clock_gettime(CLOCK_MONOTONIC)",
        "resolution_s": 1e-9,
        "monotonic": True,
        "adjustable": False,
        "captured_at_unix_ns": 1_800_000_000_000_000_000,
    }


def _phases(*, start_ns: int, total_ns: int = 100_000_000) -> dict:
    rows = {
        name: {
            "executed": False,
            "started_monotonic_ns": None,
            "finished_monotonic_ns": None,
            "duration_ns": 0,
        }
        for name in diagnostic.ENGINE_STEP_PHASES
    }
    cursor = start_ns
    for name, duration in (
        ("scheduler_schedule", 40_000_000),
        ("scheduler_prepare_postprocess", 30_000_000),
        (
            "scheduler_commit_postprocess",
            total_ns - 70_000_000,
        ),
    ):
        rows[name] = {
            "executed": True,
            "started_monotonic_ns": cursor,
            "finished_monotonic_ns": cursor + duration,
            "duration_ns": duration,
        }
        cursor += duration
    return rows


def _rank_command_rows(
    rank: int,
    *,
    repeat_identity: int,
    request_sha: str,
    selected_sha: str,
    queue_debt_ns: int,
    cuda_ns: int,
) -> list[dict]:
    prior_finished = 40_000_000 + (
        queue_debt_ns if rank == 3 else 0
    )
    current_started = 40_000_000 + (
        queue_debt_ns if rank == 3 else 0
    )
    current_wall = cuda_ns if rank == 3 else (rank + 1) * 100_000_000
    current_finished = current_started + current_wall
    common = {
        "repeat_index": repeat_identity,
        "request_set_sha256": request_sha,
        "batch_kind": "speculative",
        "speculative_selected_sequence_ids_sha256": selected_sha,
        "dispatch_started_monotonic_ns": 0,
        "status": "ok",
        "error_type": "",
        "error_detail": "",
        "terminal_error_monotonic_ns": None,
    }
    prior = {
        "rank": rank,
        "command_id": 10,
        "method_name": "prepare",
        "requires_ack": False,
        "engine_step_id": None,
        "dispatch_published_monotonic_ns": 0,
        **common,
    }
    current = {
        "rank": rank,
        "command_id": 11,
        "method_name": "step",
        "requires_ack": True,
        "engine_step_id": 20,
        "dispatch_published_monotonic_ns": 40_000_000,
        **common,
    }
    if rank == 0:
        local_finished = 40_000_000 + cuda_ns
        ack_started = 40_000_000 + queue_debt_ns + cuda_ns
        prior.update({
            "event_woken_monotonic_ns": None,
            "envelope_read_monotonic_ns": None,
            "method_started_monotonic_ns": None,
            "method_finished_monotonic_ns": None,
            "local_method_started_monotonic_ns": 0,
            "local_method_finished_monotonic_ns": 40_000_000,
            "ack_send_started_monotonic_ns": None,
            "ack_send_finished_monotonic_ns": None,
            "ack_wait_started_monotonic_ns": None,
            "ack_wait_finished_monotonic_ns": None,
        })
        current.update({
            "event_woken_monotonic_ns": None,
            "envelope_read_monotonic_ns": None,
            "method_started_monotonic_ns": None,
            "method_finished_monotonic_ns": None,
            "local_method_started_monotonic_ns": 40_000_000,
            "local_method_finished_monotonic_ns": local_finished,
            "ack_send_started_monotonic_ns": None,
            "ack_send_finished_monotonic_ns": None,
            "ack_wait_started_monotonic_ns": ack_started,
            "ack_wait_finished_monotonic_ns": (
                ack_started + 20_000_000
            ),
        })
    else:
        prior.update({
            "event_woken_monotonic_ns": 0,
            "envelope_read_monotonic_ns": 0,
            "method_started_monotonic_ns": 0,
            "method_finished_monotonic_ns": prior_finished,
            "local_method_started_monotonic_ns": None,
            "local_method_finished_monotonic_ns": None,
            "ack_send_started_monotonic_ns": None,
            "ack_send_finished_monotonic_ns": None,
            "ack_wait_started_monotonic_ns": None,
            "ack_wait_finished_monotonic_ns": None,
        })
        current.update({
            "event_woken_monotonic_ns": 40_000_000,
            "envelope_read_monotonic_ns": 40_000_000,
            "method_started_monotonic_ns": current_started,
            "method_finished_monotonic_ns": current_finished,
            "local_method_started_monotonic_ns": None,
            "local_method_finished_monotonic_ns": None,
            "ack_send_started_monotonic_ns": current_finished,
            "ack_send_finished_monotonic_ns": current_finished,
            "ack_wait_started_monotonic_ns": None,
            "ack_wait_finished_monotonic_ns": None,
        })
    return [prior, current]


def _timeline(
    *,
    repeat_identity: int,
    request_sha: str,
    queue_debt_ns: int,
    cuda_ns: int,
) -> dict:
    selected_sha = "b" * 64
    step_finish = (
        40_000_000
        + queue_debt_ns
        + cuda_ns
        + 20_000_000
        + 100_000_000
    )
    phase_start = step_finish - 100_000_000
    return {
        "schema_version": 1,
        "rank_snapshots": [
            {
                "schema_version": 1,
                "rank": rank,
                "enabled": True,
                "clock": _clock(),
                "rows": _rank_command_rows(
                    rank,
                    repeat_identity=repeat_identity,
                    request_sha=request_sha,
                    selected_sha=selected_sha,
                    queue_debt_ns=queue_debt_ns,
                    cuda_ns=cuda_ns,
                ),
                "dropped_rows": 0,
            }
            for rank in range(4)
        ],
        "cuda_rank_snapshots": [
            {
                "rank": rank,
                "enabled": True,
                "finalization_status": "complete",
                "steps": [{
                    "rank": rank,
                    "step_index": 0,
                    "batch_kind": "speculative",
                    "is_decode": True,
                    "decode_ordinal": 0,
                    "active_sequence_count": 4,
                    "request_set_sha256": request_sha,
                    "dispatch": "step",
                    "command_id": 11,
                    "engine_step_id": 20,
                    "repeat_index": repeat_identity,
                    "speculative_selected_sequence_ids_sha256": (
                        selected_sha
                    ),
                    "wall_ns": (
                        cuda_ns if rank == 3 else (rank + 1) * 100_000_000
                    ),
                    "cuda_ns": (
                        cuda_ns if rank == 3 else (rank + 1) * 100_000_000
                    ),
                    "non_cuda_upper_bound_ns": 0,
                }],
                "collectives": [],
                "dropped_steps": 0,
                "dropped_collectives": 0,
            }
            for rank in range(4)
        ],
        "engine_steps": [{
            "engine_step_id": 20,
            "repeat_index": repeat_identity,
            "request_set_sha256": request_sha,
            "batch_kind": "speculative",
            "speculative_selected_sequence_ids_sha256": selected_sha,
            "command_ids": [11],
            "started_monotonic_ns": 40_000_000,
            "finished_monotonic_ns": step_finish,
            "step_wall_ns": step_finish - 40_000_000,
            "phases": _phases(start_ns=phase_start),
            "status": "ok",
            "error_type": "",
            "detail": "",
        }],
        "engine_dropped_steps": 0,
    }


def _parity(mode: str, *, repeat_index: int) -> dict:
    graph = mode == "graph"
    proposal_rows = [
        [[31, 32, 33, 34] for _ in range(4)],
        [[41, 42], [43], [44, 45, 46], [47, 48, 49, 50]],
    ]
    accepted_prefixes = [[2, 1, 3, 4], [1, 1, 2, 2]]
    accepted_rows = [
        [
            row[:accepted]
            for row, accepted in zip(call, prefixes)
        ]
        for call, prefixes in zip(
            proposal_rows,
            accepted_prefixes,
        )
    ]
    proposed = sum(
        len(row) for call in proposal_rows for row in call
    )
    accepted = sum(
        len(row) for call in accepted_rows for row in call
    )
    return {
        "target_token_rows": [
            list(range(request, request + 16))
            for request in range(4)
        ],
        "proposal_token_rows": proposal_rows,
        "proposal_row_lengths": [
            [len(row) for row in call] for call in proposal_rows
        ],
        "accepted_prefix_counts": accepted_prefixes,
        "accepted_token_rows": accepted_rows,
        "transaction_digest": "c" * 64,
        "active_transaction_count": 0,
        "acceptance": {
            "proposed_tokens": proposed,
            "accepted_tokens": accepted,
            "rate": accepted / proposed,
        },
        "rank_graph_counters": [
            {
                "rank": rank,
                "capture_attempts": 1 if graph else 0,
                "captures": 1 if graph else 0,
                "replays": (repeat_index + 2) * 4 if graph else 0,
                "quarantines": 0,
                "fallback_pre_replay": 0,
            }
            for rank in range(4)
        ],
        "rank_graph_resources": [
            {
                "rank": rank,
                "ready_entry_count": 1 if graph else 0,
                "static_bytes": 55_000 if graph else 0,
                "reserved_bytes": 700_000_000 if graph else 0,
                "total_capture_ns": 1_500_000_000 if graph else 0,
            }
            for rank in range(4)
        ],
        "rank_graph_identities": [
            {
                "rank": rank,
                "sha256": (f"{rank + 1:x}" * 64)[:64] if graph else None,
            }
            for rank in range(4)
        ],
    }


def _run(
    mode: str,
    repeat: int,
    *,
    queue_debt_ns: int,
    cuda_ns: int,
    request_sha: str,
) -> dict:
    timeline_repeat = repeat + 1
    e2e_ns = 1_100_000_000 if mode == "graph" else 1_000_000_000
    return {
        "repeat": repeat,
        "command_timeline_repeat_index": timeline_repeat,
        "campaign_interval": {
            "started_at_unix_ns": 1_800_000_000_000_000_000,
            "finished_at_unix_ns": 1_800_000_001_000_000_000,
            "started_at_monotonic_ns": 0,
            "finished_at_monotonic_ns": 1_000_000_000,
        },
        "outputs": [
            list(range(request, request + 16))
            for request in range(4)
        ],
        "timing": {
            "request_count": 4,
            "total_output_tokens": 64,
            "batch_elapsed_ns": e2e_ns,
            "per_request": [
                {
                    "sequence_id": request,
                    "output_tokens": 16,
                    "ttft_ns": 200_000_000,
                    "tpot_ns": 60_000_000,
                    "completion_latency_ns": e2e_ns,
                }
                for request in range(4)
            ],
        },
        "correctness": _parity(mode, repeat_index=repeat),
        "runtime": {
            "command_timeline": _timeline(
                repeat_identity=timeline_repeat,
                request_sha=request_sha,
                queue_debt_ns=queue_debt_ns,
                cuda_ns=cuda_ns,
            ),
        },
        "telemetry": {
            "gpu_rows": [
                {
                    "repeat_index": timeline_repeat,
                    "sampled_at_unix_ns": (
                        1_800_000_000_500_000_000 + rank
                    ),
                    "sampled_at_monotonic_ns": 500_000_000 + rank,
                    "gpu_uuid": f"GPU-{rank}",
                }
                for rank in range(4)
            ],
            "host_rows": [{
                "repeat_index": timeline_repeat,
                "sampled_at_unix_ns": 1_800_000_000_500_000_000,
                "sampled_at_monotonic_ns": 500_000_000,
            }],
        },
    }


def _worker(mode: str) -> dict:
    queue_debt_ns = 60_000_000 if mode == "graph" else 0
    cuda_ns = 400_000_000 if mode == "graph" else 370_000_000
    prompt_rows = [
        {
            "prompt_index": index,
            "token_ids": [index + 1] * 256,
            "token_count": 256,
            "sha256": _sha([index + 1] * 256),
        }
        for index in range(4)
    ]
    prompt_sha = _sha([row["token_ids"] for row in prompt_rows])
    warmup = _run(
        mode,
        -1,
        queue_debt_ns=queue_debt_ns,
        cuda_ns=cuda_ns,
        request_sha=prompt_sha,
    )
    warmup["correctness"]["rank_graph_counters"] = [
        {
            **row,
            "replays": 4 if mode == "graph" else 0,
        }
        for row in warmup["correctness"]["rank_graph_counters"]
    ]
    return {
        "policy": "learned",
        "tensor_parallel_size": 4,
        "batch_size": 4,
        "max_proposal_tokens": 4,
        "prompt_rows": prompt_rows,
        "prompt_sha256": prompt_sha,
        "requested_output_tokens": 16,
        "request_order": [0, 1, 2, 3],
        "temperature": 0.0,
        "proposal_kv_allocator": "direct",
        "proposal_kv_offload": False,
        "source_commit": "d" * 40,
        "source_tree_sha256": "e" * 64,
        "target_checkpoint_identifier": "Qwen3-1.7B",
        "draft_checkpoint_identifier": "Qwen3-0.6B",
        "tokenizer_identifier": "Qwen3-1.7B",
        "gpu_uuids": [f"GPU-{rank}" for rank in range(4)],
        "cuda_graph_mode": mode,
        "warmup_runs": [warmup],
        "measured_runs": [
            _run(
                mode,
                repeat,
                queue_debt_ns=queue_debt_ns,
                cuda_ns=cuda_ns,
                request_sha=prompt_sha,
            )
            for repeat in range(5)
        ],
    }


def _raw_epochs() -> dict[str, dict]:
    epochs = {}
    for identity in diagnostic.expected_epoch_identities():
        worker = _worker(identity.label)
        if identity.block_index == 1 and identity.label == "eager":
            request_sha = worker["prompt_sha256"]
            worker["measured_runs"] = [
                _run(
                    "eager",
                    repeat,
                    queue_debt_ns=180_000_000,
                    cuda_ns=370_000_000,
                    request_sha=request_sha,
                )
                for repeat in range(5)
            ]
        epochs[identity.key] = {"worker": worker}
    return epochs


def _artifact() -> dict:
    return diagnostic.build_command_timeline_artifact(
        metadata={
            "configuration": copy.deepcopy(
                diagnostic.EXACT_CONFIGURATION
            ),
            "provenance": {
                "run_tag": "task5-local",
                "captured_at_unix_ns": 1_800_000_000_000_000_000,
            },
        },
        epoch_raw_inputs=_raw_epochs(),
        input_files={
            "epoch_inputs": {
                "path": "workers/epochs.json",
                "sha256": "1" * 64,
            },
        },
        source_files={
            "tools/source.py": "2" * 64,
        },
    )


def test_schedule_constants_and_epoch_identities_are_exact():
    assert diagnostic.SCHEMA_VERSION == 1
    assert diagnostic.BLOCK_SCHEDULE == (
        ("eager", "graph"),
        ("graph", "eager"),
        ("graph", "eager"),
        ("eager", "graph"),
    )
    identities = diagnostic.expected_epoch_identities()
    assert len(identities) == 8
    assert [
        (
            row.block_index,
            row.order,
            row.label,
            row.position,
            row.epoch_index,
            row.key,
        )
        for row in identities
    ] == [
        (0, "eager_graph", "eager", "first", 0, "b0-eager-first"),
        (0, "eager_graph", "graph", "second", 1, "b0-graph-second"),
        (1, "graph_eager", "graph", "first", 2, "b1-graph-first"),
        (1, "graph_eager", "eager", "second", 3, "b1-eager-second"),
        (2, "graph_eager", "graph", "first", 4, "b2-graph-first"),
        (2, "graph_eager", "eager", "second", 5, "b2-eager-second"),
        (3, "eager_graph", "eager", "first", 6, "b3-eager-first"),
        (3, "eager_graph", "graph", "second", 7, "b3-graph-second"),
    ]


@pytest.mark.parametrize(
    "replacement",
    [
        diagnostic.EpochIdentity(
            0,
            "eager_graph",
            "graph",
            "first",
            0,
        ),
        diagnostic.EpochIdentity(
            0,
            "eager_graph",
            "eager",
            "second",
            0,
        ),
    ],
)
def test_identity_rejects_wrong_schedule_label_or_position(replacement):
    with pytest.raises(ValueError, match="epoch identity"):
        diagnostic.validate_epoch_worker(_worker("eager"), replacement)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("policy", "target", "policy"),
        ("tensor_parallel_size", 1, "tensor parallel"),
        ("batch_size", 1, "batch size"),
        ("max_proposal_tokens", 3, "proposal"),
        ("requested_output_tokens", 15, "output"),
        ("request_order", [1, 0, 2, 3], "request order"),
        ("temperature", 0.1, "temperature"),
        ("proposal_kv_allocator", "pooled", "allocator"),
        ("proposal_kv_offload", True, "offload"),
        ("cuda_graph_mode", "graph", "mode"),
    ],
)
def test_identity_rejects_fixed_configuration_drift(
    field,
    value,
    message,
):
    worker = _worker("eager")
    worker[field] = value
    with pytest.raises(ValueError, match=message):
        diagnostic.validate_epoch_worker(
            worker,
            diagnostic.expected_epoch_identities()[0],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda worker: worker["prompt_rows"][0]["token_ids"].pop(),
            "prompt",
        ),
        (
            lambda worker: worker["prompt_rows"][0].__setitem__(
                "sha256",
                "0" * 64,
            ),
            "prompt",
        ),
        (
            lambda worker: worker.__setitem__(
                "prompt_sha256",
                "0" * 64,
            ),
            "prompt",
        ),
    ],
)
def test_identity_rejects_prompt_length_or_digest_drift(
    mutation,
    message,
):
    worker = _worker("eager")
    mutation(worker)
    with pytest.raises(ValueError, match=message):
        diagnostic.validate_epoch_worker(
            worker,
            diagnostic.expected_epoch_identities()[0],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_commit", "f" * 40, "source"),
        ("source_tree_sha256", "f" * 64, "source"),
        (
            "target_checkpoint_identifier",
            "other-target",
            "target checkpoint",
        ),
        (
            "draft_checkpoint_identifier",
            "other-draft",
            "draft checkpoint",
        ),
        ("tokenizer_identifier", "other-tokenizer", "tokenizer"),
        ("gpu_uuids", ["GPU-9", "GPU-1", "GPU-2", "GPU-3"], "GPU"),
    ],
)
def test_artifact_rejects_cross_epoch_provenance_drift(
    field,
    value,
    message,
):
    raw = _raw_epochs()
    graph_key = diagnostic.expected_epoch_identities()[1].key
    raw[graph_key]["worker"][field] = value
    with pytest.raises(ValueError, match=message):
        diagnostic.build_command_timeline_artifact(
            metadata={
                "configuration": diagnostic.EXACT_CONFIGURATION,
                "provenance": {"run_tag": "drift"},
            },
            epoch_raw_inputs=raw,
            input_files={
                "epochs": {
                    "path": "workers/epochs.json",
                    "sha256": "1" * 64,
                },
            },
            source_files={"tools/source.py": "2" * 64},
        )


def test_artifact_rejects_cross_graph_epoch_identity_drift():
    raw = _raw_epochs()
    graph_keys = [
        identity.key
        for identity in diagnostic.expected_epoch_identities()
        if identity.label == "graph"
    ]
    raw[graph_keys[-1]]["worker"]["warmup_runs"][0]["correctness"][
        "rank_graph_identities"
    ][3]["sha256"] = "f" * 64
    for run in raw[graph_keys[-1]]["worker"]["measured_runs"]:
        run["correctness"]["rank_graph_identities"][3]["sha256"] = (
            "f" * 64
        )
    with pytest.raises(ValueError, match="graph identity"):
        diagnostic.build_command_timeline_artifact(
            metadata={
                "configuration": diagnostic.EXACT_CONFIGURATION,
                "provenance": {"run_tag": "graph-identity-drift"},
            },
            epoch_raw_inputs=raw,
            input_files={
                "epochs": {
                    "path": "workers/epochs.json",
                    "sha256": "1" * 64,
                },
            },
            source_files={"tools/source.py": "2" * 64},
        )


@pytest.mark.parametrize(
    ("mode", "mutation", "message"),
    [
        (
            "graph",
            lambda worker: worker["measured_runs"][2]["correctness"][
                "rank_graph_counters"
            ][0].__setitem__("captures", 2),
            "capture",
        ),
        (
            "graph",
            lambda worker: worker["measured_runs"][2]["correctness"][
                "rank_graph_counters"
            ][0].__setitem__("replays", 4),
            "replay",
        ),
        (
            "graph",
            lambda worker: worker["measured_runs"][2]["correctness"][
                "rank_graph_resources"
            ][0].__setitem__("static_bytes", 55_001),
            "resource",
        ),
        (
            "eager",
            lambda worker: worker["measured_runs"][2]["correctness"][
                "rank_graph_counters"
            ][0].__setitem__("captures", 1),
            "eager",
        ),
        (
            "eager",
            lambda worker: worker["measured_runs"][2]["correctness"][
                "rank_graph_counters"
            ][0].__setitem__("replays", 1),
            "eager",
        ),
        (
            "eager",
            lambda worker: worker["measured_runs"][2]["correctness"][
                "rank_graph_resources"
            ][0].__setitem__("ready_entry_count", 1),
            "eager",
        ),
    ],
)
def test_identity_rejects_graph_or_eager_counter_drift(
    mode,
    mutation,
    message,
):
    worker = _worker(mode)
    mutation(worker)
    identity = next(
        row
        for row in diagnostic.expected_epoch_identities()
        if row.label == mode
    )
    with pytest.raises(ValueError, match=message):
        diagnostic.validate_epoch_worker(worker, identity)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda row: row["target_token_rows"][0].__setitem__(0, 999),
            "parity",
        ),
        (
            lambda row: row["proposal_token_rows"][0][0].__setitem__(
                0,
                999,
            ),
            "parity",
        ),
        (
            lambda row: row["accepted_prefix_counts"][0].__setitem__(
                0,
                1,
            ),
            "accepted",
        ),
        (
            lambda row: row["accepted_token_rows"][0][0].__setitem__(
                0,
                999,
            ),
            "accepted",
        ),
        (
            lambda row: row.__setitem__(
                "transaction_digest",
                "f" * 64,
            ),
            "transaction",
        ),
        (
            lambda row: row["acceptance"].__setitem__(
                "accepted_tokens",
                row["acceptance"]["accepted_tokens"] - 1,
            ),
            "acceptance",
        ),
        (
            lambda row: row.__setitem__("active_transaction_count", 1),
            "active transaction",
        ),
    ],
)
def test_parity_rejects_one_mutated_repeat(mutation, message):
    worker = _worker("graph")
    mutation(worker["measured_runs"][3]["correctness"])
    with pytest.raises(ValueError, match=message):
        diagnostic.validate_epoch_worker(
            worker,
            diagnostic.expected_epoch_identities()[1],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda row: row["proposal_token_rows"][1][1].append(0),
            "padding",
        ),
        (
            lambda row: row["proposal_token_rows"][0][0].append(35),
            "Q4",
        ),
    ],
)
def test_parity_rejects_padded_or_oversized_logical_rows(
    mutation,
    message,
):
    worker = _worker("graph")
    mutation(worker["measured_runs"][0]["correctness"])
    with pytest.raises(ValueError, match=message):
        diagnostic.validate_epoch_worker(
            worker,
            diagnostic.expected_epoch_identities()[1],
        )


def test_timeline_join_computes_exact_components_and_conservation():
    repeat = diagnostic.join_repeat_timeline(_worker("graph"), 0)
    assert repeat["critical_rank"] == 3
    assert repeat["components_ns"] == {
        "worker_queue_debt": 60_000_000,
        "worker_cuda_execution": 400_000_000,
        "ack_wait": 20_000_000,
        "scheduler_postprocess": 100_000_000,
    }
    assert repeat["conservation"]["passed"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda run: run["runtime"]["command_timeline"][
                "rank_snapshots"
            ][3]["clock"].__setitem__("boot_id", "other-boot"),
            "boot",
        ),
        (
            lambda run: run["runtime"]["command_timeline"][
                "rank_snapshots"
            ][3]["clock"].__setitem__("implementation", "other-clock"),
            "clock",
        ),
        (
            lambda run: run["runtime"]["command_timeline"][
                "rank_snapshots"
            ][0]["rows"].append(
                copy.deepcopy(
                    run["runtime"]["command_timeline"][
                        "rank_snapshots"
                    ][0]["rows"][-1]
                )
            ),
            "duplicate",
        ),
        (
            lambda run: run["runtime"]["command_timeline"][
                "rank_snapshots"
            ][1]["rows"].pop(),
            "missing",
        ),
        (
            lambda run: run["runtime"]["command_timeline"][
                "rank_snapshots"
            ][2]["rows"].reverse(),
            "order",
        ),
        (
            lambda run: run["runtime"]["command_timeline"][
                "cuda_rank_snapshots"
            ][0]["steps"][0].__setitem__("command_id", 999),
            "unknown command",
        ),
        (
            lambda run: run["runtime"]["command_timeline"][
                "engine_steps"
            ][0].__setitem__("command_ids", [999]),
            "unknown command",
        ),
        (
            lambda run: run["runtime"]["command_timeline"][
                "rank_snapshots"
            ][3]["rows"][1].__setitem__(
                "method_started_monotonic_ns",
                39_999_999,
            ),
            "queue",
        ),
        (
            lambda run: run["runtime"]["command_timeline"][
                "rank_snapshots"
            ][0]["rows"][1].__setitem__(
                "ack_wait_finished_monotonic_ns",
                499_999_999,
            ),
            "ack",
        ),
        (
            lambda run: run["runtime"]["command_timeline"][
                "cuda_rank_snapshots"
            ][3]["steps"][0].__setitem__("cuda_ns", -1),
            "CUDA",
        ),
        (
            lambda run: run["runtime"]["command_timeline"][
                "engine_steps"
            ][0]["phases"]["scheduler_schedule"].__setitem__(
                "duration_ns",
                -1,
            ),
            "phase",
        ),
        (
            lambda run: run["runtime"]["command_timeline"][
                "cuda_rank_snapshots"
            ][3]["steps"][0].__setitem__(
                "cuda_ns",
                400_000_001,
            ),
            "wall",
        ),
        (
            lambda run: run["runtime"]["command_timeline"][
                "rank_snapshots"
            ][3]["rows"][0].__setitem__(
                "method_finished_monotonic_ns",
                100_000_001,
            ),
            "overlap",
        ),
        (
            lambda run: run["runtime"]["command_timeline"][
                "rank_snapshots"
            ][0]["rows"][0].__setitem__(
                "ack_wait_started_monotonic_ns",
                1,
            ),
            "non-ack",
        ),
        (
            lambda run: run["runtime"]["command_timeline"][
                "rank_snapshots"
            ][0]["rows"][1].__setitem__(
                "ack_wait_finished_monotonic_ns",
                None,
            ),
            "missing ack",
        ),
        (
            lambda run: run["runtime"]["command_timeline"][
                "engine_steps"
            ][0].update(
                finished_monotonic_ns=630_000_000,
                step_wall_ns=590_000_000,
            ),
            "conservation",
        ),
        (
            lambda run: run["runtime"]["command_timeline"][
                "rank_snapshots"
            ][0]["rows"][0].__setitem__(
                "dispatch_started_monotonic_ns",
                1_000_000_001,
            ),
            "campaign interval",
        ),
    ],
)
def test_timeline_join_rejects_one_invalid_mutation(
    mutation,
    message,
):
    worker = _worker("graph")
    mutation(worker["measured_runs"][0])
    with pytest.raises(ValueError, match=message):
        diagnostic.join_repeat_timeline(worker, 0)


@pytest.mark.parametrize(
    "field",
    ["rank", "command_id", "engine_step_id", "repeat_index"],
)
def test_timeline_identities_reject_boolean_integer_aliases(field):
    worker = _worker("graph")
    row = worker["measured_runs"][0]["runtime"]["command_timeline"][
        "rank_snapshots"
    ][0]["rows"][1]
    row[field] = True
    with pytest.raises(ValueError, match="integer|identity|rank"):
        diagnostic.join_repeat_timeline(worker, 0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda timeline: timeline["cuda_rank_snapshots"][2][
                "steps"
            ][0].__setitem__(
                "speculative_selected_sequence_ids_sha256",
                "f" * 64,
            ),
            "selected-sequence",
        ),
        (
            lambda timeline: timeline["rank_snapshots"][3]["rows"][
                1
            ].__setitem__(
                "ack_send_finished_monotonic_ns",
                520_000_001,
            ),
            "ack",
        ),
        (
            lambda timeline: timeline["engine_steps"][0].__setitem__(
                "status",
                "error",
            ),
            "engine status",
        ),
        (
            lambda timeline: timeline["engine_steps"][0]["phases"][
                "scheduler_schedule"
            ].update(
                started_monotonic_ns=430_000_000,
                finished_monotonic_ns=470_000_000,
            ),
            "overlap",
        ),
    ],
)
def test_timeline_join_rejects_cross_surface_semantic_errors(
    mutation,
    message,
):
    worker = _worker("graph")
    mutation(
        worker["measured_runs"][0]["runtime"]["command_timeline"]
    )
    with pytest.raises(ValueError, match=message):
        diagnostic.join_repeat_timeline(worker, 0)


def test_stationarity_thresholds_are_inclusive():
    assert diagnostic.stationarity_for_values(
        [100.0, 100.0, 100.0, 100.0, 110.0]
    )["robust_dispersion_passed"] is True
    exact = diagnostic.stationarity_for_values(
        [90.0, 90.0, 100.0, 100.0, 110.0]
    )
    assert exact["robust_dispersion"] == pytest.approx(0.10)
    assert exact["half_drift"] == pytest.approx(0.15)
    assert exact["passed"] is True


@pytest.mark.parametrize(
    "values",
    [
        [89.0, 89.0, 100.0, 100.0, 110.0],
        [90.0, 90.0, 100.0, 100.0, 111.0],
    ],
)
def test_stationarity_rejects_one_unit_beyond_a_threshold(values):
    assert diagnostic.stationarity_for_values(values)["passed"] is False


def test_b4_request_timing_uses_the_exact_even_cardinality_midpoint():
    identity = diagnostic.expected_epoch_identities()[0]
    worker = _worker(identity.label)
    request_values = [1, 2, 100, 101]
    for run in worker["measured_runs"]:
        for row, value in zip(
            run["timing"]["per_request"],
            request_values,
        ):
            row["completion_latency_ns"] = value
    admission = diagnostic.build_epoch_admission(
        identity,
        {"worker": worker},
    )
    assert admission["metrics"]["e2e"] == [51] * 5
    assert admission["stationarity"]["e2e"]["values"] == [51] * 5
    assert admission["stationarity"]["e2e"]["median"] == 51
    assert admission["stationarity"]["e2e"]["passed"] is True


def test_exact_half_ns_midpoint_reaches_localization_without_float_drift():
    midpoint = diagnostic._median(
        [1, 2, 2_000_000_000_000_001, 2_000_000_000_000_002],
        "B4 request timing",
    )
    assert midpoint == {
        "numerator": 2_000_000_000_000_003,
        "denominator": 2,
    }
    blocks = _classification_blocks(same_sign_blocks=4)
    for block in blocks:
        block["median_e2e_pair_ns"] = [midpoint, midpoint]
        block["absolute_unexplained_ns"] = 100_000_000_000_000
    exact = diagnostic.summarize_boundary_effects(blocks)
    assert exact["median_unexplained_ratio_passed"] is True

    blocks[2]["absolute_unexplained_ns"] += 1
    blocks[3]["absolute_unexplained_ns"] += 1
    beyond = diagnostic.summarize_boundary_effects(blocks)
    assert beyond["median_unexplained_ratio_passed"] is False


def _classification_effects(
    *,
    explained_ns: int = 60,
    qualifying_blocks: int = 3,
    same_sign_blocks: int = 3,
    unexplained_ns: int = 10,
) -> dict:
    return diagnostic.summarize_boundary_effects(
        _classification_blocks(
            explained_ns=explained_ns,
            qualifying_blocks=qualifying_blocks,
            same_sign_blocks=same_sign_blocks,
            unexplained_ns=unexplained_ns,
        )
    )


def _classification_blocks(
    *,
    explained_ns: int = 60,
    qualifying_blocks: int = 3,
    same_sign_blocks: int = 3,
    unexplained_ns: int = 10,
) -> list[dict]:
    blocks = []
    for block_index in range(4):
        queue = (
            explained_ns
            if block_index < same_sign_blocks
            else 0
        )
        if block_index >= qualifying_blocks and queue != 0:
            queue = 50 if queue > 0 else -50
        other = 100 - queue - unexplained_ns
        blocks.append({
            "block_index": block_index,
            "order": (
                "eager_graph"
                if block_index in (0, 3)
                else "graph_eager"
            ),
            "e2e_delta_ns": 100,
            "component_deltas_ns": {
                "worker_queue_debt": queue,
                "worker_cuda_execution": other,
                "ack_wait": 0,
                "scheduler_postprocess": 0,
            },
            "absolute_unexplained_ns": unexplained_ns,
            "median_e2e_ns": 100,
        })
    return blocks


def _admission(**overrides) -> dict:
    row = {
        "identity_correctness_passed": True,
        "timeline_conservation_passed": True,
        "stationarity_passed": True,
        "passed": True,
    }
    row.update(overrides)
    return row


def _position_balanced_localization_effects() -> dict:
    blocks = _classification_blocks(same_sign_blocks=4)
    blocks[1]["component_deltas_ns"].update({
        "worker_queue_debt": -120,
        "worker_cuda_execution": 210,
    })
    return diagnostic.summarize_boundary_effects(blocks)


def test_classification_localizes_exact_inclusive_boundaries():
    result = diagnostic.classify_boundary(
        _admission(),
        _position_balanced_localization_effects(),
    )
    assert result == {
        "classification": "BOUNDARY_LOCALIZED",
        "localized_boundary": "worker_queue_debt",
        "stable_but_unlocalized": False,
        "runtime_optimization_authorized": True,
        "performance_improvement_established": False,
        "phase_1_complete": False,
        "promotion_ready": False,
    }


def test_classification_rejects_true_chronological_order_crossover():
    blocks = _classification_blocks(same_sign_blocks=4)
    assert all(block["e2e_delta_ns"] > 0 for block in blocks)
    chronological_second_minus_first = [
        (
            block["e2e_delta_ns"]
            if block["order"] == "eager_graph"
            else -block["e2e_delta_ns"]
        )
        for block in blocks
    ]
    assert chronological_second_minus_first == [100, -100, -100, 100]

    effects = diagnostic.summarize_boundary_effects(blocks)
    result = diagnostic.classify_boundary(_admission(), effects)
    assert effects["boundaries"]["worker_queue_debt"][
        "position_balance_consistent"
    ] is False
    assert effects["boundaries"]["worker_queue_debt"][
        "sequence_interaction_consistent"
    ] is False
    assert result["classification"] == "PAIRED_PROTOCOL_UNSTABLE"
    assert result["localized_boundary"] is None
    assert result["stable_but_unlocalized"] is True
    assert result["runtime_optimization_authorized"] is False


def test_classification_rejects_mixed_signs_within_one_order_group():
    blocks = _classification_blocks(same_sign_blocks=4)
    blocks[3]["component_deltas_ns"].update({
        "worker_queue_debt": -60,
        "worker_cuda_execution": 150,
    })
    result = diagnostic.classify_boundary(
        _admission(),
        diagnostic.summarize_boundary_effects(blocks),
    )
    assert result["classification"] == "PAIRED_PROTOCOL_UNSTABLE"
    assert result["localized_boundary"] is None
    assert result["stable_but_unlocalized"] is True
    assert result["runtime_optimization_authorized"] is False


def test_classification_rejects_order_reversal_sequence_interaction():
    blocks = _classification_blocks(same_sign_blocks=4)
    for block_index in (1, 2, 3):
        blocks[block_index].update({
            "e2e_delta_ns": -100,
            "component_deltas_ns": {
                "worker_queue_debt": -60,
                "worker_cuda_execution": -50,
                "ack_wait": 0,
                "scheduler_postprocess": 0,
            },
        })
    result = diagnostic.classify_boundary(
        _admission(),
        diagnostic.summarize_boundary_effects(blocks),
    )
    assert result["classification"] == "PAIRED_PROTOCOL_UNSTABLE"
    assert result["localized_boundary"] is None
    assert result["stable_but_unlocalized"] is True
    assert result["runtime_optimization_authorized"] is False


def test_classification_rejects_multiple_localized_boundaries():
    blocks = _classification_blocks(same_sign_blocks=4)
    for block_index, block in enumerate(blocks):
        block["component_deltas_ns"] = {
            "worker_queue_debt": (
                -120 if block_index == 1 else 60
            ),
            "worker_cuda_execution": (
                -120 if block_index == 1 else 60
            ),
            "ack_wait": 330 if block_index == 1 else -30,
            "scheduler_postprocess": 0,
        }
    effects = diagnostic.summarize_boundary_effects(blocks)
    assert effects["boundaries"]["worker_queue_debt"]["localized"] is True
    assert effects["boundaries"]["worker_cuda_execution"]["localized"] is True
    result = diagnostic.classify_boundary(_admission(), effects)
    assert result["classification"] == "PAIRED_PROTOCOL_UNSTABLE"
    assert result["localized_boundary"] is None
    assert result["stable_but_unlocalized"] is True
    assert result["runtime_optimization_authorized"] is False


def _large_integer_effects(
    *,
    third_explained_ns: int = 600_000_000_000_000_000,
    unexplained_ns: int = 100_000_000_000_000_000,
) -> dict:
    e2e_ns = 1_000_000_000_000_000_000
    blocks = []
    for block_index, queue_ns in enumerate((
        600_000_000_000_000_000,
        -700_000_000_000_000_000,
        third_explained_ns,
        500_000_000_000_000_000,
    )):
        blocks.append({
            "block_index": block_index,
            "order": (
                "eager_graph"
                if block_index in (0, 3)
                else "graph_eager"
            ),
            "e2e_delta_ns": e2e_ns,
            "component_deltas_ns": {
                "worker_queue_debt": queue_ns,
                "worker_cuda_execution": (
                    e2e_ns - queue_ns - unexplained_ns
                ),
                "ack_wait": 0,
                "scheduler_postprocess": 0,
            },
            "absolute_unexplained_ns": unexplained_ns,
            "median_e2e_ns": e2e_ns,
        })
    return diagnostic.summarize_boundary_effects(blocks)


def test_integer_threshold_accepts_exact_sixty_percent_at_1e18():
    effects = _large_integer_effects()
    queue = effects["boundaries"]["worker_queue_debt"]
    assert queue["qualifying_block_count"] == 3
    assert isinstance(effects["blocks"][0]["e2e_delta_ns"], int)
    assert isinstance(
        effects["blocks"][0]["component_deltas_ns"][
            "worker_queue_debt"
        ],
        int,
    )
    assert diagnostic.classify_boundary(
        _admission(),
        effects,
    )["runtime_optimization_authorized"] is True


def test_integer_threshold_rejects_one_ns_below_sixty_percent_at_1e18():
    effects = _large_integer_effects(
        third_explained_ns=599_999_999_999_999_999,
    )
    assert effects["boundaries"]["worker_queue_debt"][
        "qualifying_block_count"
    ] == 2
    result = diagnostic.classify_boundary(_admission(), effects)
    assert result["stable_but_unlocalized"] is True
    assert result["runtime_optimization_authorized"] is False


def test_integer_residual_accepts_exact_ten_percent_at_1e18():
    result = diagnostic.classify_boundary(
        _admission(),
        _large_integer_effects(),
    )
    assert result["runtime_optimization_authorized"] is True


def test_integer_residual_rejects_one_ns_above_ten_percent_at_1e18():
    result = diagnostic.classify_boundary(
        _admission(),
        _large_integer_effects(
            unexplained_ns=100_000_000_000_000_001,
        ),
    )
    assert result["stable_but_unlocalized"] is True
    assert result["runtime_optimization_authorized"] is False


def test_zero_e2e_delta_with_nonzero_component_is_finite_and_unlocalized():
    blocks = _classification_blocks(same_sign_blocks=4)
    blocks[3].update({
        "e2e_delta_ns": 0,
        "component_deltas_ns": {
            "worker_queue_debt": 60,
            "worker_cuda_execution": -50,
            "ack_wait": 0,
            "scheduler_postprocess": 0,
        },
    })
    effects = diagnostic.summarize_boundary_effects(blocks)
    zero_block = effects["blocks"][3]
    assert zero_block["explanation_ratios"]["worker_queue_debt"] is None
    assert zero_block["explanation_ratio_defined"][
        "worker_queue_debt"
    ] is False
    diagnostic.canonical_json_bytes(effects)
    result = diagnostic.classify_boundary(_admission(), effects)
    assert result["stable_but_unlocalized"] is True
    assert result["runtime_optimization_authorized"] is False


@pytest.mark.parametrize(
    "effects",
    [
        _classification_effects(explained_ns=59),
        _classification_effects(same_sign_blocks=2),
        _classification_effects(unexplained_ns=11),
    ],
)
def test_classification_rejects_one_unit_beyond_localization_threshold(
    effects,
):
    result = diagnostic.classify_boundary(_admission(), effects)
    assert result["classification"] == "PAIRED_PROTOCOL_UNSTABLE"
    assert result["stable_but_unlocalized"] is True
    assert result["runtime_optimization_authorized"] is False


@pytest.mark.parametrize(
    ("admission", "expected"),
    [
        (
            _admission(
                identity_correctness_passed=False,
                timeline_conservation_passed=False,
                stationarity_passed=False,
                passed=False,
            ),
            "INVALID_IDENTITY_OR_CORRECTNESS",
        ),
        (
            _admission(
                timeline_conservation_passed=False,
                stationarity_passed=False,
                passed=False,
            ),
            "TIMELINE_INCOMPLETE_OR_NONCONSERVING",
        ),
        (
            _admission(
                stationarity_passed=False,
                passed=False,
            ),
            "PAIRED_PROTOCOL_UNSTABLE",
        ),
    ],
)
def test_classification_precedence_is_fail_closed(admission, expected):
    assert diagnostic.classify_boundary(
        admission,
        _classification_effects(),
    )["classification"] == expected


def test_classification_rejects_inconsistent_admission_summary():
    with pytest.raises(ValueError, match="admission"):
        diagnostic.classify_boundary(
            _admission(passed=False),
            _classification_effects(),
        )


@pytest.mark.parametrize(
    ("snapshot", "value"),
    [
        ("timeline", True),
        ("timeline", "1"),
        ("rank", True),
        ("rank", "1"),
    ],
)
def test_timeline_schema_versions_require_strict_integers(
    snapshot,
    value,
):
    worker = _worker("graph")
    timeline = worker["measured_runs"][0]["runtime"]["command_timeline"]
    if snapshot == "timeline":
        timeline["schema_version"] = value
    else:
        timeline["rank_snapshots"][0]["schema_version"] = value
    with pytest.raises(ValueError, match="schema version|integer"):
        diagnostic.join_repeat_timeline(worker, 0)


def test_artifact_has_exact_keys_and_recomputes_all_derived_fields():
    artifact = _artifact()
    assert tuple(artifact) == diagnostic.TOP_LEVEL_KEYS
    assert artifact["classification"] == "BOUNDARY_LOCALIZED"
    assert artifact["runtime_optimization_authorized"] is True
    assert artifact["performance_improvement_established"] is False
    assert diagnostic.validate_command_timeline_artifact(
        artifact
    ) == artifact


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("blocks", 0, "passed"), False),
        (("admission", "passed"), False),
        (
            (
                "effects",
                "blocks",
                0,
                "component_deltas_ns",
                "worker_queue_debt",
            ),
            59_999_999,
        ),
        (("classification",), "PAIRED_PROTOCOL_UNSTABLE"),
        (("localized_boundary",), None),
        (("runtime_optimization_authorized",), False),
        (("performance_improvement_established",), True),
    ],
)
def test_artifact_validation_rejects_derived_field_tampering(
    path,
    value,
):
    artifact = _artifact()
    target = artifact
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="recomputation|canonical"):
        diagnostic.validate_command_timeline_artifact(artifact)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda artifact: artifact.__setitem__("extra", True),
        lambda artifact: artifact["configuration"].__setitem__(
            "temperature",
            float("nan"),
        ),
        lambda artifact: artifact["epochs"][
            next(iter(artifact["epochs"]))
        ]["worker"].__setitem__("batch_size", True),
        lambda artifact: artifact["raw_input_files"][
            "epoch_inputs"
        ].__setitem__("path", "../escape.json"),
        lambda artifact: artifact["source_files"].__setitem__(
            "/absolute.py",
            "2" * 64,
        ),
    ],
)
def test_artifact_validation_rejects_noncanonical_or_unbounded_data(
    mutation,
):
    artifact = _artifact()
    mutation(artifact)
    with pytest.raises(ValueError):
        diagnostic.validate_command_timeline_artifact(artifact)


def test_canonical_json_and_hash_are_deterministic():
    left = {"b": [2, 1], "a": {"x": 3}}
    right = {"a": {"x": 3}, "b": [2, 1]}
    assert diagnostic.canonical_json_bytes(left) == (
        b'{"a":{"x":3},"b":[2,1]}\n'
    )
    assert diagnostic.canonical_json_sha256(left) == (
        diagnostic.canonical_json_sha256(right)
    )


def test_canonical_json_rejects_oversized_integer_data():
    with pytest.raises(ValueError, match="integer|bounded"):
        diagnostic.canonical_json_bytes({
            "timestamp": 1 << 80,
        })
