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
VERIFIER_PATH = (
    ROOT
    / "tools"
    / "verify_autoregressive_draft_command_timeline_diagnostic.py"
)
RUNNER_PATH = (
    ROOT
    / "tools"
    / "run_autoregressive_draft_command_timeline_remote.py"
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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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
        local_finished = 40_000_000 + queue_debt_ns + cuda_ns
        ack_started = local_finished
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
        epochs[identity.key] = {"worker": worker}
    return epochs


def _raw_epochs_with_half_integer_e2e() -> dict[str, dict]:
    epochs = _raw_epochs()
    for identity in diagnostic.expected_epoch_identities():
        base = (
            1_100_000_000
            if identity.label == "graph"
            else 1_000_000_000
        )
        worker = epochs[identity.key]["worker"]
        for run in worker["measured_runs"]:
            for offset, row in enumerate(run["timing"]["per_request"]):
                row["completion_latency_ns"] = base + offset
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


def test_artifact_accepts_cross_graph_epoch_capture_time_variation():
    raw = _raw_epochs()
    graph_keys = [
        identity.key
        for identity in diagnostic.expected_epoch_identities()
        if identity.label == "graph"
    ]
    worker = raw[graph_keys[-1]]["worker"]
    for run in worker["warmup_runs"] + worker["measured_runs"]:
        for resource in run["correctness"]["rank_graph_resources"]:
            resource["total_capture_ns"] += 1_000_000 + resource["rank"]

    artifact = diagnostic.build_command_timeline_artifact(
        metadata={
            "configuration": diagnostic.EXACT_CONFIGURATION,
            "provenance": {"run_tag": "graph-capture-time-variation"},
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

    assert artifact["schema_version"] == diagnostic.SCHEMA_VERSION


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


def test_timeline_join_accepts_json_sorted_engine_phase_keys():
    worker = _worker("graph")
    for step in worker["measured_runs"][0]["runtime"][
        "command_timeline"
    ]["engine_steps"]:
        step["phases"] = dict(sorted(step["phases"].items()))

    repeat = diagnostic.join_repeat_timeline(worker, 0)

    assert list(repeat["engine_steps"][0]["phases"]) == list(
        diagnostic.ENGINE_STEP_PHASES
    )


def test_timeline_join_accepts_async_cuda_duration_beyond_cpu_wall():
    worker = _worker("graph")
    row = worker["measured_runs"][0]["runtime"]["command_timeline"][
        "cuda_rank_snapshots"
    ][0]["steps"][0]
    row["cuda_ns"] = row["wall_ns"] + 500_000_000
    row["non_cuda_upper_bound_ns"] = 0

    repeat = diagnostic.join_repeat_timeline(worker, 0)

    assert repeat["cuda_steps"][0]["cuda_ns"] == row["cuda_ns"]
    assert repeat["cuda_steps"][0]["attributed_cuda_ns"] == row["wall_ns"]
    assert repeat["cuda_steps"][0]["non_cuda_upper_bound_ns"] == 0


def test_timeline_join_accepts_spec_verify_cuda_kind_with_decode_step():
    worker = _worker("graph")
    timeline = worker["measured_runs"][0]["runtime"]["command_timeline"]
    for snapshot in timeline["rank_snapshots"]:
        command = snapshot["rows"][1]
        command["method_name"] = "run_spec_verify_batch"
        command["batch_kind"] = "decode"
    timeline["engine_steps"][0]["batch_kind"] = "decode"
    for snapshot in timeline["cuda_rank_snapshots"]:
        snapshot["steps"][0]["batch_kind"] = "spec_verify"

    repeat = diagnostic.join_repeat_timeline(worker, 0)

    assert repeat["cuda_steps"][0]["batch_kind"] == "spec_verify"
    assert repeat["engine_steps"][0]["batch_kind"] == "decode"


def test_timeline_conservation_uses_command_wall_not_cuda_attribution():
    worker = _worker("eager")
    timeline = worker["measured_runs"][0]["runtime"]["command_timeline"]
    for snapshot in timeline["cuda_rank_snapshots"]:
        row = snapshot["steps"][0]
        row["cuda_ns"] -= 50_000_000
        row["non_cuda_upper_bound_ns"] = row["wall_ns"] - row["cuda_ns"]

    repeat = diagnostic.join_repeat_timeline(worker, 0)

    assert repeat["components_ns"]["worker_cuda_execution"] == 320_000_000
    assert repeat["conservation"] == {
        "step_wall_ns": 490_000_000,
        "attributed_ns": 490_000_000,
        "residual_ns": 0,
        "tolerance_ns": 4_900_000,
        "passed": True,
    }


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
                "non_cuda_upper_bound_ns",
                1,
            ),
            "upper bound",
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
        (
            lambda timeline: timeline["cuda_rank_snapshots"][0][
                "steps"
            ][0].__setitem__("batch_kind", "unknown-callback"),
            "batch kind",
        ),
        (
            lambda timeline: timeline["engine_steps"][0].__setitem__(
                "batch_kind",
                "other-scheduler-kind",
            ),
            "batch kind",
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


def test_zero_median_nonzero_half_drift_is_json_safe_and_fails():
    result = diagnostic.stationarity_for_values(
        [0, 0, 0, 26_430, 0]
    )

    assert result["robust_dispersion"] == 0.0
    assert result["robust_dispersion_passed"] is True
    assert result["half_drift"] is None
    assert result["half_drift_passed"] is False
    assert result["passed"] is False
    diagnostic.canonical_json_bytes(result)


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


def test_half_integer_b4_median_survives_the_complete_artifact_path():
    raw_epochs = _raw_epochs_with_half_integer_e2e()
    identities = diagnostic.expected_epoch_identities()
    eager_half = {
        "numerator": 2_000_000_003,
        "denominator": 2,
    }
    graph_half = {
        "numerator": 2_200_000_003,
        "denominator": 2,
    }
    pair_half = {
        "numerator": 2_100_000_003,
        "denominator": 2,
    }

    first_admission = diagnostic.build_epoch_admission(
        identities[0],
        raw_epochs[identities[0].key],
    )
    assert first_admission["metrics"]["e2e"] == [eager_half] * 5
    assert first_admission["stationarity"]["e2e"]["values"] == (
        [eager_half] * 5
    )
    assert first_admission["stationarity"]["e2e"]["median"] == eager_half
    assert diagnostic.stationarity_for_values(
        first_admission["metrics"]["e2e"]
    )["median"] == eager_half

    epochs = {
        identity.key: diagnostic.build_epoch_admission(
            identity,
            raw_epochs[identity.key],
        )
        for identity in identities
    }
    effects = diagnostic.compute_paired_boundary_effects(epochs)
    assert effects["blocks"][0]["e2e_delta_ns"] == 100_000_000
    assert effects["blocks"][0]["median_e2e_ns"] == pair_half

    artifact = diagnostic.build_command_timeline_artifact(
        metadata={
            "configuration": copy.deepcopy(
                diagnostic.EXACT_CONFIGURATION
            ),
            "provenance": {
                "run_tag": "task5-half-integer",
                "captured_at_unix_ns": 1_800_000_000_000_000_000,
            },
        },
        epoch_raw_inputs=raw_epochs,
        input_files={
            "epoch_inputs": {
                "path": "workers/half-integer-epochs.json",
                "sha256": "3" * 64,
            },
        },
        source_files={
            "tools/source.py": "4" * 64,
        },
    )
    assert artifact["epochs"][identities[0].key]["metrics"]["e2e"] == (
        [eager_half] * 5
    )
    assert artifact["epochs"][identities[1].key]["metrics"]["e2e"] == (
        [graph_half] * 5
    )
    assert artifact["effects"]["blocks"][0]["median_e2e_ns"] == (
        pair_half
    )
    assert diagnostic.validate_command_timeline_artifact(
        artifact
    ) == artifact


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


def test_classification_localizes_exact_inclusive_boundaries():
    result = diagnostic.classify_boundary(
        _admission(),
        _classification_effects(same_sign_blocks=4),
    )
    assert result == {
        "classification": "BOUNDARY_LOCALIZED",
        "localized_boundary": "worker_queue_debt",
        "stable_but_unlocalized": False,
        "runtime_optimization_authorized": False,
        "performance_improvement_established": False,
        "phase_1_complete": False,
        "promotion_ready": False,
    }


def test_stable_graph_minus_eager_label_effect_is_not_a_crossover():
    blocks = _classification_blocks(
        qualifying_blocks=4,
        same_sign_blocks=4,
    )
    assert all(block["e2e_delta_ns"] > 0 for block in blocks)

    effects = diagnostic.summarize_boundary_effects(blocks)
    boundary = effects["boundaries"]["worker_queue_debt"]
    result = diagnostic.classify_boundary(_admission(), effects)
    assert boundary["block_effects"] == [
        {
            "block_index": 0,
            "order": "eager_graph",
            "label_effect_ns": 60,
            "position_effect_ns": 60,
        },
        {
            "block_index": 1,
            "order": "graph_eager",
            "label_effect_ns": 60,
            "position_effect_ns": -60,
        },
        {
            "block_index": 2,
            "order": "graph_eager",
            "label_effect_ns": 60,
            "position_effect_ns": -60,
        },
        {
            "block_index": 3,
            "order": "eager_graph",
            "label_effect_ns": 60,
            "position_effect_ns": 60,
        },
    ]
    assert boundary["aggregate_label_effect_ns"] == 60
    assert boundary["aggregate_position_effect_ns"] == 0
    assert boundary["eager_graph_position_effect_ns"] == 60
    assert boundary["graph_eager_position_effect_ns"] == -60
    assert boundary["order_interaction_ns"] == 0
    assert boundary["sequence_interaction_ns"] == 120
    order_checks = {
        order: {
            key: value
            for key, value in check.items()
            if key in {
                "aggregate_label_effect_ns",
                "aggregate_position_effect_ns",
                "supports_aggregate_label",
                "has_qualifying_block",
                "label_reversal_block_indices",
                "no_label_reversal",
                "passed",
            }
        }
        for order, check in boundary["order_group_checks"].items()
    }
    assert order_checks == {
        "eager_graph": {
            "aggregate_label_effect_ns": 60,
            "aggregate_position_effect_ns": 60,
            "supports_aggregate_label": True,
            "has_qualifying_block": True,
            "label_reversal_block_indices": [],
            "no_label_reversal": True,
            "passed": True,
        },
        "graph_eager": {
            "aggregate_label_effect_ns": 60,
            "aggregate_position_effect_ns": -60,
            "supports_aggregate_label": True,
            "has_qualifying_block": True,
            "label_reversal_block_indices": [],
            "no_label_reversal": True,
            "passed": True,
        },
    }
    assert boundary["position_balance_consistent"] is True
    assert boundary["order_interaction_below_label"] is True
    assert boundary["sequence_interaction_consistent"] is True
    assert result["classification"] == "BOUNDARY_LOCALIZED"
    assert result["localized_boundary"] == "worker_queue_debt"
    assert result["runtime_optimization_authorized"] is False


def test_position_driven_order_confound_is_not_a_label_effect():
    blocks = _classification_blocks(
        qualifying_blocks=4,
        same_sign_blocks=4,
    )
    for block_index in (1, 2):
        blocks[block_index]["e2e_delta_ns"] *= -1
        blocks[block_index]["component_deltas_ns"] = {
            name: -value
            for name, value in blocks[block_index][
                "component_deltas_ns"
            ].items()
        }

    effects = diagnostic.summarize_boundary_effects(blocks)
    boundary = effects["boundaries"]["worker_queue_debt"]
    result = diagnostic.classify_boundary(_admission(), effects)
    assert [
        row["label_effect_ns"] for row in boundary["block_effects"]
    ] == [60, -60, -60, 60]
    assert [
        row["position_effect_ns"] for row in boundary["block_effects"]
    ] == [60, 60, 60, 60]
    assert boundary["aggregate_label_effect_ns"] == 0
    assert boundary["aggregate_position_effect_ns"] == 60
    assert boundary["eager_graph_position_effect_ns"] == 60
    assert boundary["graph_eager_position_effect_ns"] == 60
    assert boundary["order_interaction_ns"] == 120
    assert boundary["sequence_interaction_ns"] == 0
    assert all(
        check["supports_aggregate_label"] is False
        for check in boundary["order_group_checks"].values()
    )
    assert boundary["position_balance_consistent"] is False
    assert boundary["order_interaction_below_label"] is False
    assert boundary["sequence_interaction_consistent"] is False
    assert result["classification"] == "PAIRED_PROTOCOL_UNSTABLE"
    assert result["localized_boundary"] is None
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
    effects = diagnostic.summarize_boundary_effects(blocks)
    boundary = effects["boundaries"]["worker_queue_debt"]
    result = diagnostic.classify_boundary(_admission(), effects)
    assert boundary["eager_graph_position_effect_ns"] == 0
    assert boundary["graph_eager_position_effect_ns"] == 60
    assert boundary["order_interaction_ns"] == 60
    assert boundary["sequence_interaction_ns"] == -60
    assert boundary["order_group_checks"]["eager_graph"][
        "label_reversal_block_indices"
    ] == [0]
    assert boundary["order_group_checks"]["eager_graph"][
        "no_label_reversal"
    ] is False
    assert boundary["order_group_checks"]["eager_graph"]["passed"] is False
    assert boundary["position_balance_consistent"] is False
    assert boundary["sequence_interaction_consistent"] is False
    assert result["classification"] == "PAIRED_PROTOCOL_UNSTABLE"
    assert result["localized_boundary"] is None
    assert result["stable_but_unlocalized"] is True
    assert result["runtime_optimization_authorized"] is False


def test_classification_rejects_multiple_localized_boundaries():
    blocks = _classification_blocks(same_sign_blocks=4)
    for block in blocks:
        block["component_deltas_ns"] = {
            "worker_queue_debt": 60,
            "worker_cuda_execution": 60,
            "ack_wait": -30,
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
        700_000_000_000_000_000,
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
    )["runtime_optimization_authorized"] is False


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
    assert result["runtime_optimization_authorized"] is False


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
    assert artifact["runtime_optimization_authorized"] is False
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
        (("runtime_optimization_authorized",), True),
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


DETACHED_COMMAND_TIMELINE_PATHS = {
    "manifest.sha256",
    "verify.command-timeline.remote.json",
    "verify.command-timeline.remote.log",
    "verify.command-timeline.local.json",
    "verify.command-timeline.local.log",
}


def _sha_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(diagnostic.canonical_json_bytes(payload))


def _telemetry_sidecar(identity, worker: dict) -> dict:
    return {
        "schema_version": 1,
        "epoch_key": identity.key,
        "measured_runs": [
            {
                "repeat": run["repeat"],
                "command_timeline_repeat_index": run[
                    "command_timeline_repeat_index"
                ],
                "telemetry": copy.deepcopy(run["telemetry"]),
            }
            for run in worker["measured_runs"]
        ],
    }


def _result_summary(artifact_path: Path, artifact: dict) -> dict:
    return {
        "artifact_sha256": _sha_path(artifact_path),
        "classification": artifact["classification"],
        "localized_boundary": artifact["localized_boundary"],
        "runtime_optimization_authorized": False,
        "performance_improvement_established": False,
        "phase_1_complete": False,
        "promotion_ready": False,
    }


def _write_manifest(bundle_root: Path) -> Path:
    manifest_path = bundle_root / "manifest.sha256"
    rows = []
    for path in sorted(
        candidate
        for candidate in bundle_root.rglob("*")
        if (
            candidate.is_file()
            and candidate.relative_to(bundle_root).as_posix()
            not in DETACHED_COMMAND_TIMELINE_PATHS
        )
    ):
        relative = path.relative_to(bundle_root).as_posix()
        rows.append(f"{_sha_path(path)}  {relative}")
    manifest_path.write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _refresh_artifact_result_and_manifest(
    bundle_root: Path,
    artifact: dict,
) -> None:
    artifact_path = bundle_root / "command-timeline.json"
    _write_json(artifact_path, artifact)
    _write_json(
        bundle_root / "result.json",
        _result_summary(artifact_path, artifact),
    )
    _write_manifest(bundle_root)


def _replace_with_symlink(
    path: Path,
    *,
    root: Path,
    destination: str,
    in_root_target: Path | None = None,
) -> None:
    if destination == "in_root":
        target = in_root_target or (
            root
            / ".symlink-targets"
            / path.relative_to(root)
        )
    elif destination == "escaping":
        target = (
            root.parent
            / f"{root.name}-escaping-symlink-targets"
            / path.relative_to(root)
        )
    else:
        raise AssertionError(f"unexpected destination: {destination}")
    target.parent.mkdir(parents=True, exist_ok=True)
    path.rename(target)
    path.symlink_to(target, target_is_directory=target.is_dir())


@pytest.fixture
def command_timeline_bundle(tmp_path) -> dict:
    bundle_root = tmp_path / "bundle"
    source_root = tmp_path / "source-root"
    bundle_root.mkdir()
    source_path = source_root / "tools" / "source.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("BOUND_SOURCE = True\n", encoding="utf-8")
    source_files = {
        "tools/source.py": _sha_path(source_path),
    }
    metadata = {
        "configuration": copy.deepcopy(
            diagnostic.EXACT_CONFIGURATION
        ),
        "provenance": {
            "run_tag": "task6-source-bound",
            "captured_at_unix_ns": 1_800_000_000_000_000_000,
        },
    }
    metadata_path = bundle_root / "metadata.json"
    source_manifest_path = bundle_root / "source_manifest.json"
    _write_json(metadata_path, metadata)
    _write_json(source_manifest_path, source_files)

    raw_epochs = _raw_epochs()
    input_files = {
        "metadata": {
            "path": "metadata.json",
            "sha256": _sha_path(metadata_path),
        },
        "source_manifest": {
            "path": "source_manifest.json",
            "sha256": _sha_path(source_manifest_path),
        },
    }
    for identity in diagnostic.expected_epoch_identities():
        worker_path = (
            bundle_root
            / "workers"
            / f"block-{identity.block_index}"
            / f"{identity.label}.json"
        )
        telemetry_path = (
            bundle_root
            / "telemetry"
            / f"block-{identity.block_index}"
            / f"{identity.label}.json"
        )
        telemetry = _telemetry_sidecar(
            identity,
            raw_epochs[identity.key]["worker"],
        )
        _write_json(
            worker_path,
            raw_epochs[identity.key]["worker"],
        )
        _write_json(telemetry_path, telemetry)
        raw_epochs[identity.key]["telemetry"] = telemetry
        input_files[f"worker:{identity.key}"] = {
            "path": worker_path.relative_to(bundle_root).as_posix(),
            "sha256": _sha_path(worker_path),
        }
        input_files[f"telemetry:{identity.key}"] = {
            "path": telemetry_path.relative_to(
                bundle_root
            ).as_posix(),
            "sha256": _sha_path(telemetry_path),
        }

    artifact = diagnostic.build_command_timeline_artifact(
        metadata=metadata,
        epoch_raw_inputs=raw_epochs,
        input_files=input_files,
        source_files=source_files,
    )
    artifact_path = bundle_root / "command-timeline.json"
    _write_json(artifact_path, artifact)
    _write_json(
        bundle_root / "result.json",
        _result_summary(artifact_path, artifact),
    )
    (bundle_root / "source.patch").write_bytes(b"")
    manifest_path = _write_manifest(bundle_root)
    return {
        "artifact": artifact,
        "artifact_path": artifact_path,
        "bundle_root": bundle_root,
        "manifest_path": manifest_path,
        "source_root": source_root,
    }


def test_verifier_recomputes_complete_source_bound_bundle(
    command_timeline_bundle,
):
    verifier = load_module(VERIFIER_PATH, "command_timeline_verifier")
    receipt = verifier.verify_command_timeline_diagnostic(
        artifact_path=command_timeline_bundle["artifact_path"],
        source_root=command_timeline_bundle["source_root"],
        manifest_path=command_timeline_bundle["manifest_path"],
    )
    artifact = command_timeline_bundle["artifact"]
    assert receipt["verified"] is True
    assert receipt["artifact_sha256"] == _sha_path(
        command_timeline_bundle["artifact_path"]
    )
    assert receipt["classification"] == artifact["classification"]
    assert receipt["localized_boundary"] == artifact[
        "localized_boundary"
    ]
    assert receipt["runtime_optimization_authorized"] is False
    assert receipt["performance_improvement_established"] is False
    assert receipt["phase_1_complete"] is False
    assert receipt["promotion_ready"] is False
    assert receipt["source_file_count"] == 1
    assert receipt["raw_input_file_count"] == 18
    assert receipt["manifest_verified"] is True
    assert receipt["manifest_file_count"] == 21
    assert receipt["source_inventory_sha256"] == (
        diagnostic.canonical_json_sha256(artifact["source_files"])
    )
    assert receipt["raw_input_inventory_sha256"] == (
        diagnostic.canonical_json_sha256(
            artifact["raw_input_files"]
        )
    )
    assert receipt["manifest_sha256"] == _sha_path(
        command_timeline_bundle["manifest_path"]
    )
    assert receipt["verifier_source_sha256"] == _sha_path(
        VERIFIER_PATH
    )


def test_verifier_rejects_worker_byte_tamper(
    command_timeline_bundle,
):
    worker = (
        command_timeline_bundle["bundle_root"]
        / "workers"
        / "block-0"
        / "eager.json"
    )
    worker.write_bytes(worker.read_bytes() + b"\n")
    verifier = load_module(
        VERIFIER_PATH,
        "command_timeline_worker_byte_tamper",
    )
    with pytest.raises(ValueError, match="raw input hash mismatch"):
        verifier.verify_command_timeline_diagnostic(
            artifact_path=command_timeline_bundle["artifact_path"],
            source_root=command_timeline_bundle["source_root"],
            manifest_path=command_timeline_bundle["manifest_path"],
        )


def test_verifier_rejects_timeline_row_raw_tamper_after_rebinding(
    command_timeline_bundle,
):
    bundle_root = command_timeline_bundle["bundle_root"]
    worker_path = bundle_root / "workers" / "block-0" / "eager.json"
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    for snapshot in worker["measured_runs"][0]["runtime"][
        "command_timeline"
    ]["rank_snapshots"]:
        snapshot["rows"][0]["method_name"] = "prepare-tampered"
    _write_json(worker_path, worker)
    artifact = copy.deepcopy(command_timeline_bundle["artifact"])
    first_key = diagnostic.expected_epoch_identities()[0].key
    artifact["raw_input_files"][f"worker:{first_key}"]["sha256"] = (
        _sha_path(worker_path)
    )
    _refresh_artifact_result_and_manifest(bundle_root, artifact)
    verifier = load_module(
        VERIFIER_PATH,
        "command_timeline_timeline_row_raw_tamper",
    )
    with pytest.raises(
        ValueError,
        match="canonical artifact mismatch after recomputation",
    ):
        verifier.verify_command_timeline_diagnostic(
            artifact_path=command_timeline_bundle["artifact_path"],
            source_root=command_timeline_bundle["source_root"],
            manifest_path=command_timeline_bundle["manifest_path"],
        )


def test_verifier_rejects_telemetry_tamper_after_complete_rebinding(
    command_timeline_bundle,
):
    bundle_root = command_timeline_bundle["bundle_root"]
    identity = diagnostic.expected_epoch_identities()[0]
    telemetry_path = (
        bundle_root
        / "telemetry"
        / "block-0"
        / "eager.json"
    )
    telemetry = json.loads(
        telemetry_path.read_text(encoding="utf-8")
    )
    telemetry["measured_runs"][0]["telemetry"]["host_rows"][0][
        "sampled_at_monotonic_ns"
    ] += 1
    _write_json(telemetry_path, telemetry)
    artifact = copy.deepcopy(command_timeline_bundle["artifact"])
    artifact["raw_input_files"][f"telemetry:{identity.key}"][
        "sha256"
    ] = _sha_path(telemetry_path)
    _refresh_artifact_result_and_manifest(bundle_root, artifact)
    verifier = load_module(
        VERIFIER_PATH,
        "command_timeline_telemetry_semantic_tamper",
    )
    with pytest.raises(ValueError, match="telemetry sidecar mismatch"):
        verifier.verify_command_timeline_diagnostic(
            artifact_path=command_timeline_bundle["artifact_path"],
            source_root=command_timeline_bundle["source_root"],
            manifest_path=command_timeline_bundle["manifest_path"],
        )


def test_verifier_rejects_source_file_tamper(
    command_timeline_bundle,
):
    source = (
        command_timeline_bundle["source_root"]
        / "tools"
        / "source.py"
    )
    source.write_text("BOUND_SOURCE = False\n", encoding="utf-8")
    verifier = load_module(
        VERIFIER_PATH,
        "command_timeline_source_tamper",
    )
    with pytest.raises(ValueError, match="source hash mismatch"):
        verifier.verify_command_timeline_diagnostic(
            artifact_path=command_timeline_bundle["artifact_path"],
            source_root=command_timeline_bundle["source_root"],
            manifest_path=command_timeline_bundle["manifest_path"],
        )


@pytest.mark.parametrize(
    ("authoritative_name", "relative"),
    [
        ("artifact", "command-timeline.json"),
        ("result", "result.json"),
        ("metadata", "metadata.json"),
        ("worker", "workers/block-0"),
        ("telemetry", "telemetry/block-0"),
    ],
)
@pytest.mark.parametrize("destination", ["escaping", "in_root"])
@pytest.mark.parametrize("with_manifest", [False, True])
def test_verifier_rejects_authoritative_symlinks(
    command_timeline_bundle,
    authoritative_name,
    relative,
    destination,
    with_manifest,
):
    bundle_root = command_timeline_bundle["bundle_root"]
    _replace_with_symlink(
        bundle_root / relative,
        root=bundle_root,
        destination=destination,
    )
    manifest_path = None
    if with_manifest:
        manifest_path = _write_manifest(bundle_root)
    verifier = load_module(
        VERIFIER_PATH,
        "command_timeline_authoritative_symlink_"
        f"{authoritative_name}_{destination}_{with_manifest}",
    )
    with pytest.raises(ValueError, match="symlink"):
        verifier.verify_command_timeline_diagnostic(
            artifact_path=command_timeline_bundle["artifact_path"],
            source_root=command_timeline_bundle["source_root"],
            manifest_path=manifest_path,
        )


@pytest.mark.parametrize("destination", ["escaping", "in_root"])
@pytest.mark.parametrize("with_manifest", [False, True])
def test_verifier_rejects_source_binding_symlinks(
    command_timeline_bundle,
    destination,
    with_manifest,
):
    source_root = command_timeline_bundle["source_root"]
    _replace_with_symlink(
        source_root / "tools",
        root=source_root,
        destination=destination,
    )
    verifier = load_module(
        VERIFIER_PATH,
        "command_timeline_source_symlink_"
        f"{destination}_{with_manifest}",
    )
    with pytest.raises(ValueError, match="symlink"):
        verifier.verify_command_timeline_diagnostic(
            artifact_path=command_timeline_bundle["artifact_path"],
            source_root=source_root,
            manifest_path=(
                command_timeline_bundle["manifest_path"]
                if with_manifest
                else None
            ),
        )


@pytest.mark.parametrize("destination", ["escaping", "in_root"])
def test_manifest_rejects_manifest_path_symlink(
    command_timeline_bundle,
    destination,
):
    bundle_root = command_timeline_bundle["bundle_root"]
    manifest_path = command_timeline_bundle["manifest_path"]
    _replace_with_symlink(
        manifest_path,
        root=bundle_root,
        destination=destination,
        in_root_target=(
            bundle_root / "verify.command-timeline.local.log"
            if destination == "in_root"
            else None
        ),
    )
    verifier = load_module(
        VERIFIER_PATH,
        f"command_timeline_manifest_symlink_{destination}",
    )
    with pytest.raises(ValueError, match="symlink"):
        verifier.verify_manifest(manifest_path, bundle_root)


@pytest.mark.parametrize("destination", ["escaping", "in_root"])
def test_manifest_inventory_rejects_unlisted_symlink(
    command_timeline_bundle,
    destination,
):
    bundle_root = command_timeline_bundle["bundle_root"]
    unlisted = bundle_root / "unlisted.json"
    unlisted.write_text("{}\n", encoding="utf-8")
    _replace_with_symlink(
        unlisted,
        root=bundle_root,
        destination=destination,
    )
    manifest_path = _write_manifest(bundle_root)
    verifier = load_module(
        VERIFIER_PATH,
        f"command_timeline_inventory_symlink_{destination}",
    )
    with pytest.raises(ValueError, match="symlink"):
        verifier.verify_manifest(manifest_path, bundle_root)


@pytest.mark.parametrize(
    "unsafe_path",
    ["/absolute.json", "../escape.json"],
)
def test_manifest_rejects_unsafe_absolute_or_parent_path(
    command_timeline_bundle,
    unsafe_path,
):
    manifest = command_timeline_bundle["manifest_path"]
    rows = manifest.read_text(encoding="utf-8").splitlines()
    digest = rows[0].split("  ", 1)[0]
    rows[0] = f"{digest}  {unsafe_path}"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    verifier = load_module(
        VERIFIER_PATH,
        "command_timeline_manifest_unsafe",
    )
    with pytest.raises(ValueError, match="safe relative"):
        verifier.verify_manifest(
            manifest,
            command_timeline_bundle["bundle_root"],
        )


def test_manifest_rejects_duplicate_path(command_timeline_bundle):
    manifest = command_timeline_bundle["manifest_path"]
    rows = manifest.read_text(encoding="utf-8").splitlines()
    manifest.write_text(
        "\n".join([*rows, rows[0]]) + "\n",
        encoding="utf-8",
    )
    verifier = load_module(
        VERIFIER_PATH,
        "command_timeline_manifest_duplicate",
    )
    with pytest.raises(ValueError, match="duplicated"):
        verifier.verify_manifest(
            manifest,
            command_timeline_bundle["bundle_root"],
        )


def test_verifier_rejects_missing_authoritative_result(
    command_timeline_bundle,
):
    bundle_root = command_timeline_bundle["bundle_root"]
    (bundle_root / "result.json").unlink()
    _write_manifest(bundle_root)
    verifier = load_module(
        VERIFIER_PATH,
        "command_timeline_missing_authoritative",
    )
    with pytest.raises(ValueError, match="authoritative file is missing"):
        verifier.verify_command_timeline_diagnostic(
            artifact_path=command_timeline_bundle["artifact_path"],
            source_root=command_timeline_bundle["source_root"],
            manifest_path=command_timeline_bundle["manifest_path"],
        )


def test_manifest_rejects_extra_unlisted_authoritative_file(
    command_timeline_bundle,
):
    (
        command_timeline_bundle["bundle_root"] / "unlisted.json"
    ).write_text("{}\n", encoding="utf-8")
    verifier = load_module(
        VERIFIER_PATH,
        "command_timeline_manifest_extra",
    )
    with pytest.raises(ValueError, match="manifest inventory"):
        verifier.verify_manifest(
            command_timeline_bundle["manifest_path"],
            command_timeline_bundle["bundle_root"],
        )


def test_verifier_rejects_result_summary_mismatch(
    command_timeline_bundle,
):
    bundle_root = command_timeline_bundle["bundle_root"]
    result_path = bundle_root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["localized_boundary"] = "ack_wait"
    _write_json(result_path, result)
    _write_manifest(bundle_root)
    verifier = load_module(
        VERIFIER_PATH,
        "command_timeline_result_summary_tamper",
    )
    with pytest.raises(ValueError, match="result summary mismatch"):
        verifier.verify_command_timeline_diagnostic(
            artifact_path=command_timeline_bundle["artifact_path"],
            source_root=command_timeline_bundle["source_root"],
            manifest_path=command_timeline_bundle["manifest_path"],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("classification", "PAIRED_PROTOCOL_UNSTABLE"),
        ("runtime_optimization_authorized", True),
        ("performance_improvement_established", True),
        ("phase_1_complete", True),
        ("promotion_ready", True),
    ],
)
def test_verifier_rejects_classification_or_false_claim_tamper(
    command_timeline_bundle,
    field,
    value,
):
    bundle_root = command_timeline_bundle["bundle_root"]
    artifact = copy.deepcopy(command_timeline_bundle["artifact"])
    artifact[field] = value
    if field == "classification":
        artifact["runtime_optimization_authorized"] = False
    _refresh_artifact_result_and_manifest(bundle_root, artifact)
    verifier = load_module(
        VERIFIER_PATH,
        f"command_timeline_claim_tamper_{field}",
    )
    with pytest.raises(
        ValueError,
        match="recomputation|must remain false|canonical",
    ):
        verifier.verify_command_timeline_diagnostic(
            artifact_path=command_timeline_bundle["artifact_path"],
            source_root=command_timeline_bundle["source_root"],
            manifest_path=command_timeline_bundle["manifest_path"],
        )


def test_remote_and_local_receipts_are_canonically_equivalent(
    command_timeline_bundle,
    capsys,
):
    verifier = load_module(
        VERIFIER_PATH,
        "command_timeline_receipt_equivalence",
    )
    bundle_root = command_timeline_bundle["bundle_root"]
    remote_path = (
        bundle_root / "verify.command-timeline.remote.json"
    )
    local_path = bundle_root / "verify.command-timeline.local.json"
    common = [
        "--artifact",
        str(command_timeline_bundle["artifact_path"]),
        "--source-root",
        str(command_timeline_bundle["source_root"]),
        "--manifest",
        str(command_timeline_bundle["manifest_path"]),
    ]
    assert verifier.main([
        *common,
        "--receipt",
        str(remote_path),
        "--verification-location",
        "remote",
    ]) == 0
    capsys.readouterr()
    assert verifier.main([
        *common,
        "--receipt",
        str(local_path),
        "--verification-location",
        "local",
    ]) == 0
    capsys.readouterr()
    remote = json.loads(remote_path.read_text(encoding="utf-8"))
    local = json.loads(local_path.read_text(encoding="utf-8"))
    for receipt in (remote, local):
        assert receipt["verified_at_utc"]
        assert receipt["verification_location"] in {"remote", "local"}
        assert receipt["artifact_path"]
        for field in (
            "verified_at_utc",
            "verification_location",
            "artifact_path",
        ):
            receipt.pop(field)
    assert diagnostic.canonical_json_bytes(remote) == (
        diagnostic.canonical_json_bytes(local)
    )


def test_runner_augments_raw_worker_into_exact_task5_identity_schema():
    runner = load_module(RUNNER_PATH, "command_timeline_runner_augment")
    raw = _worker("graph")
    for field in (
        "source_commit",
        "source_tree_sha256",
        "target_checkpoint_identifier",
        "draft_checkpoint_identifier",
        "tokenizer_identifier",
        "gpu_uuids",
    ):
        raw.pop(field)

    augmented = runner.augment_worker_payload(
        raw,
        source_commit="a" * 40,
        source_tree_sha256="b" * 64,
        target_checkpoint_identifier="Qwen3-8B",
        draft_checkpoint_identifier="Qwen3-0.6B",
        tokenizer_identifier="Qwen3-8B",
        gpu_uuids=["GPU-0", "GPU-1", "GPU-2", "GPU-3"],
    )

    assert augmented["source_commit"] == "a" * 40
    assert augmented["source_tree_sha256"] == "b" * 64
    assert augmented["target_checkpoint_identifier"] == "Qwen3-8B"
    assert augmented["draft_checkpoint_identifier"] == "Qwen3-0.6B"
    assert augmented["tokenizer_identifier"] == "Qwen3-8B"
    assert augmented["gpu_uuids"] == [
        "GPU-0",
        "GPU-1",
        "GPU-2",
        "GPU-3",
    ]
    identity = diagnostic.expected_epoch_identities()[1]
    assert diagnostic.validate_epoch_worker(augmented, identity)


def test_runner_derives_exact_task6_telemetry_projection():
    runner = load_module(RUNNER_PATH, "command_timeline_runner_sidecar")
    worker = _worker("eager")
    identity = diagnostic.expected_epoch_identities()[0]

    sidecar = runner.derive_telemetry_sidecar(identity.key, worker)

    assert sidecar == _telemetry_sidecar(identity, worker)


def test_runner_receipt_normalization_removes_only_location_fields():
    runner = load_module(
        RUNNER_PATH,
        "command_timeline_runner_receipt_normalization",
    )
    receipt = {
        "verified": True,
        "verified_at_utc": "2026-08-18T00:00:00Z",
        "verification_location": "primary",
        "artifact_path": "/remote/primary/command-timeline.json",
        "classification": "BOUNDARY_LOCALIZED",
        "nested": {"artifact_path": "must-remain"},
    }

    assert runner.normalize_verification_receipt(receipt) == {
        "verified": True,
        "classification": "BOUNDARY_LOCALIZED",
        "nested": {"artifact_path": "must-remain"},
    }
