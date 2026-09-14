from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from tools import run_slo_cohort_burst_remote as producer
from tools import slo_cohort_burst_verify as verifier


ROOT = Path(__file__).resolve().parents[1]
REAL_STAGE0_ROOT = (
    ROOT
    / "artifacts"
    / "slo_cohort_burst_ceiling"
    / "20260913-slo-cohort-ceiling-r5-a4d2bdc1"
)
SOURCE_FILE = "tools/slo_cohort_burst_gate.py"
SOURCE_COMMIT = "a" * 40
GRAPH_SHA = "b" * 64
MODEL_SHA = "c" * 64
CONFIG_SHA = "d" * 64
WORKLOADS = ("decode_heavy", "mixed", "bursty_eos")
LOADS = ("low", "medium", "high")
ARMS = ("baseline", "candidate")
REPETITIONS = 5
REQUESTS_PER_REPETITION = 26


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(_canonical_bytes(row) for row in rows)


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _artifact_sha(bundle: dict, relative: str) -> str:
    key = verifier.ARTIFACT_KEYS[relative]
    payload = bundle[key]
    encoded = (
        _jsonl_bytes(payload)
        if relative.endswith(".jsonl")
        else _canonical_bytes(payload)
    )
    return hashlib.sha256(encoded).hexdigest()


def _canonical_graph_sha(repetition: int) -> str:
    return f"{repetition + 1:064x}"


def _canonical_graph_identities() -> dict:
    shape_keys = {
        (
            f"b{batch_size}-w{block_table_width}"
            "-trace0"
        )
        for batch_size in range(1, 9)
        for block_table_width in (2, 9, 33)
    }
    return {
        "schema_version": (
            "slo-cohort-burst.canonical-graph-identities.v1"
        ),
        "source_commit": SOURCE_COMMIT,
        "graph_identity_sha256_by_repetition": {
            str(repetition): {
                shape: _canonical_graph_sha(repetition)
                for shape in sorted(shape_keys)
            }
            for repetition in range(REPETITIONS)
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(_canonical_bytes(payload))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(_jsonl_bytes(rows))


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_legacy_single_bucket_stage0_is_rejected() -> None:
    cost_table = json.loads(
        (REAL_STAGE0_ROOT / "cost_table.json").read_text(encoding="utf-8")
    )
    raw_rows = _load_jsonl(REAL_STAGE0_ROOT / "raw_rows.jsonl")

    with pytest.raises(ValueError, match="frozen profile inventory"):
        verifier.verify_cost_table_against_profile_rows(
            cost_table,
            cost_table["source_identity"],
            raw_rows,
        )


def test_stage0_cost_table_rejects_mutated_raw_profile_row() -> None:
    source_identity = {
        "source_commit": SOURCE_COMMIT,
        "source_patch_sha256": verifier._source_tree_sha256(ROOT),
        "model": "Qwen3-0.6B",
        "checkpoint_sha256": MODEL_SHA,
        "gpu_uuid": "GPU-test",
        "gpu_name": "NVIDIA A100 80GB PCIe",
        "tensor_parallel_size": 1,
        "dtype": "torch.bfloat16",
        "config_sha256": CONFIG_SHA,
    }
    raw_rows = _cost_profile_rows(source_identity)
    cost_table = _cost_table(source_identity, raw_rows)
    raw_rows[0]["component_ns"]["graph_launch_gap"] += 1
    raw_rows[0]["wall_ns"] += 1

    with pytest.raises(ValueError, match="cost raw sample identity"):
        verifier.verify_cost_table_against_profile_rows(
            cost_table,
            cost_table["source_identity"],
            raw_rows,
        )


def test_stage0_cost_table_rejects_incomplete_frozen_profile() -> None:
    source_identity = {
        "source_commit": SOURCE_COMMIT,
        "source_patch_sha256": verifier._source_tree_sha256(ROOT),
        "model": "Qwen3-0.6B",
        "checkpoint_sha256": MODEL_SHA,
        "gpu_uuid": "GPU-test",
        "gpu_name": "NVIDIA A100 80GB PCIe",
        "tensor_parallel_size": 1,
        "dtype": "torch.bfloat16",
        "config_sha256": CONFIG_SHA,
    }
    complete_rows = _cost_profile_rows(source_identity)
    cost_table = _cost_table(source_identity, complete_rows)
    rows = [
        row
        for row in complete_rows
        if row["batch_size"] in (2, 8)
    ]
    cost_table["entries"] = {
        name: entry
        for name, entry in cost_table["entries"].items()
        if entry["batch_size"] in (2, 8)
    }
    cost_table["table_sha256"] = _sha({
        "schema_version": cost_table["schema_version"],
        "source_identity": cost_table["source_identity"],
        "entries": cost_table["entries"],
    })

    with pytest.raises(ValueError, match="frozen profile inventory"):
        verifier.verify_cost_table_against_profile_rows(
            cost_table,
            source_identity,
            rows,
        )


def test_context_prediction_uses_smallest_conservative_profile_bucket() -> None:
    predictions = {
        (8, 512, 4): 100,
        (8, 4096, 4): 200,
        (8, 16384, 4): 300,
    }

    assert verifier._context_prediction(
        predictions,
        batch_size=8,
        contexts=[256, 2048, 8192],
        width=4,
    ) == 300
    assert verifier._context_prediction(
        predictions,
        batch_size=8,
        contexts=[16_385],
        width=4,
    ) is None


def test_source_identity_rejects_wrong_source_tree_digest() -> None:
    bundle = complete_synthetic_bundle()
    bundle["source_manifest"]["source_identity"] = dict(
        bundle["source_manifest"]["source_identity"]
    )
    bundle["source_manifest"]["source_identity"][
        "source_patch_sha256"
    ] = "f" * 64

    with pytest.raises(ValueError, match="source tree digest"):
        verifier._validate_source_and_environment(bundle, ROOT)


def test_decision_reconstruction_uses_only_state_visible_at_decision() -> None:
    sequence_ids = [41, 42]
    case = {
        "workload": "decode_heavy",
        "load": "low",
        "repetition": 0,
        "arm": "candidate",
    }
    decision = _decision_row(
        case=case,
        sequence_ids=sequence_ids,
        context_buckets=[257, 257],
        cost_table_sha256="f" * 64,
        arrival_ns_by_sequence={
            sequence_id: 1_000_000
            for sequence_id in sequence_ids
        },
    )
    decision["decision"]["decision_now_ns"] = 1_000_050
    decision["decision"]["global_slack_ns"] = 870
    decision["decision"]["predicted_cost_ns_by_width"] = {
        "8": 804,
        "4": 404,
        "2": 204,
    }
    decision["decision"]["structural_eligibility_by_width"] = {
        "8": False,
        "4": False,
        "2": True,
    }
    decision["decision"]["selected_width"] = 2
    for context in decision["decision"]["context_buckets"]:
        context["remaining_output_tokens"] = 3
    for protected in decision["decision"]["protected_requests"]:
        protected["age_ns"] = 30
        protected["slack_ns"] = 870
    requests = {}
    for sequence_id in sequence_ids:
        request = _request_contract(
            request_id=f"request-{sequence_id}",
            sequence_id=sequence_id,
            arm="candidate",
            eos=False,
        )
        request["token_visible_ns"] = [
            1_000_020,
            1_000_080,
            1_000_120,
            1_000_160,
        ]
        request["completion_ns"] = request["token_visible_ns"][-1]
        request["output_token_ids"] = [11, 12, 13, 14]
        requests[(
            "decode_heavy",
            "low",
            0,
            "candidate",
            sequence_id,
        )] = {
            "maximum_output_tokens": 4,
            "_prompt_tokens": 256,
            "request": request,
        }
    predictions = {
        (2, context_bucket, width): 100 * width + 4
        for context_bucket in (256, 2048, 8192)
        for width in (2, 4, 8)
    }

    rows, indexed = verifier._validate_decisions(
        [decision],
        environment={
            "target_itl_ns": 1_000,
            "target_ttft_ns": 1_000,
            "reserve_ns": 100,
        },
        cost_table_sha256="f" * 64,
        predictions=predictions,
        request_by_sequence=requests,
    )

    assert rows == [decision]
    assert len(indexed) == 1


def test_publication_matches_its_request_output_segment() -> None:
    assert verifier._publication_matches_request_segment(
        request_output_token_ids=[7, 11, 12, 13],
        initial_completion_count=1,
        commit_tokens=[11, 12],
    )
    assert not verifier._publication_matches_request_segment(
        request_output_token_ids=[7, 11, 99, 13],
        initial_completion_count=1,
        commit_tokens=[11, 12],
    )


def test_execution_inventory_may_be_empty_when_all_decisions_are_k1() -> None:
    decision_key = ("decode_heavy", "low", 0, 1, (41, 42))

    rows, wasted, total_slots = verifier._validate_executions(
        [],
        decisions={decision_key: {"selected_width": 1}},
        environment={},
        canonical_graph_identities={},
        cost_table_sha256="f" * 64,
        request_by_sequence={},
    )

    assert rows == []
    assert wasted == 0
    assert total_slots == 0


def test_frozen_arrival_schedule_rejects_shifted_request() -> None:
    group_key = ("decode_heavy", "low", 0, "candidate")
    grouped = {
        group_key: [
            {"request": {"request_id": "q0", "arrival_ns": 100}},
            {"request": {"request_id": "q1", "arrival_ns": 201}},
        ],
    }
    traces = {
        group_key[:3]: {
            "q0": {"arrival_offset_ns": 0},
            "q1": {"arrival_offset_ns": 100},
        },
    }

    with pytest.raises(ValueError, match="arrival schedule"):
        verifier._validate_frozen_arrival_schedule(grouped, traces)


def test_terminal_request_budget_is_fail_closed() -> None:
    request = {
        "terminal_reason": "length",
        "output_token_ids": [11, 12, 13],
    }
    with pytest.raises(ValueError, match="output budget"):
        verifier._validate_terminal_request_budget(
            request,
            maximum_output_tokens=4,
            ignore_eos=True,
            eos_token_id=2,
        )


def _cost_profile_rows(source_identity: dict) -> list[dict]:
    rows = []
    for batch_size in (1, 2, 4, 8):
        for context_bucket in (512, 4096, 16384):
            for load_index, load in enumerate(LOADS):
                arrival_gap_ns = {
                    "low": 4_000_000,
                    "medium": 1_000_000,
                    "high": 0,
                }[load]
                for sample_index in range(16):
                    rows.append({
                        "schema_version": (
                            "slo-cohort-burst.ceiling-profile-row.v1"
                        ),
                        "case_id": (
                            f"{load}-b{batch_size}-c{context_bucket}"
                            f"-r0-s{sample_index + 1}"
                        ),
                        "load": load,
                        "batch_size": batch_size,
                        "context_bucket": context_bucket,
                        "burst_width": 1,
                        "source_commit": source_identity["source_commit"],
                        "offered_arrival_offsets_ns": [
                            index * arrival_gap_ns
                            for index in range(batch_size)
                        ],
                        "component_ns": {
                            "target_cuda": 90,
                            "graph_launch_gap": 2,
                            "scheduler": 10,
                            "token_d2h_publication": 1,
                            "batch_binding": 1,
                            "unattributed": 0,
                        },
                        "wall_ns": 104,
                        "committed_tokens": batch_size,
                        "cuda_reserved_bytes": 1_000,
                    })
    return rows


def _cost_table(source_identity: dict, profile_rows: list[dict]) -> dict:
    samples_by_key = verifier._cost_samples_from_profile_rows(
        profile_rows,
        source_commit=source_identity["source_commit"],
    )
    entries = {}
    for (batch_size, context_bucket, width), raw_samples in (
        samples_by_key.items()
    ):
        samples = sorted(raw_samples)
        name = f"b{batch_size}-c{context_bucket}-k{width}"
        entries[name] = {
            "batch_size": batch_size,
            "context_bucket": context_bucket,
            "burst_width": width,
            "sample_count": len(samples),
            "raw_sample_sha256": _sha(samples),
            "p50_ns": samples[1],
            "p95_ns": samples[-1],
            "p99_ns": samples[-1],
        }
    payload = {
        "schema_version": "slo-cohort-burst.cost-table.v1",
        "source_identity": source_identity,
        "entries": entries,
    }
    return {**payload, "table_sha256": _sha(payload)}


def _request_contract(
    *,
    request_id: str,
    sequence_id: int,
    arm: str,
    eos: bool,
    maximum_output_tokens: int = 8,
    arrival_ns: int = 1_000_000,
) -> dict:
    token_gap_ns = 1_000_000 if arm == "baseline" else 800_000
    output_ids = (
        [2]
        if eos
        else [
            11 + (index % 1000)
            for index in range(maximum_output_tokens)
        ]
    )
    token_times = (
        [arrival_ns + token_gap_ns]
        if eos
        else [
            arrival_ns + token_gap_ns * (index + 1)
            for index in range(len(output_ids))
        ]
    )
    return {
        "schema_version": "slo-cohort-burst.request.v1",
        "request_id": request_id,
        "sequence_id": sequence_id,
        "service_class": "default",
        "arrival_ns": arrival_ns,
        "prefill_start_ns": arrival_ns + 5,
        "prefill_complete_ns": arrival_ns + 10,
        "first_token_visible_ns": token_times[0],
        "token_visible_ns": token_times,
        "completion_ns": token_times[-1],
        "output_token_ids": output_ids,
        "output_text_sha256": _sha(output_ids),
        "terminal_reason": "eos" if eos else "length",
    }


def _lease_payload(
    *,
    sequence_ids: list[int],
    context_buckets: list[int],
    remaining_output_tokens: list[int],
    cost_table_sha256: str,
    decision_now_ns: int,
    global_slack_ns: int,
) -> dict:
    rows = []
    for (
        row_index,
        sequence_id,
        context_bucket,
        remaining_tokens,
    ) in zip(
        range(len(sequence_ids)),
        sequence_ids,
        context_buckets,
        remaining_output_tokens,
    ):
        block_table_width = context_bucket // 256 + 1
        block_table = [
            [row_index * 100 + block_index, 1]
            for block_index in range(block_table_width)
        ]
        first_slot = row_index * 16
        rows.append({
            "sequence_id": sequence_id,
            "sequence_generation": 1,
            "block_table_identity": block_table,
            "writable_block_identities": [block_table[0]],
            "first_write_position": context_bucket - 1,
            "last_write_position": context_bucket + 6,
            "first_physical_slot": first_slot,
            "last_physical_slot": first_slot + 7,
            "initial_completion_count": 0,
            "initial_sequence_length": context_bucket,
            "remaining_output_tokens": remaining_tokens,
        })
    return {
        "schema_version": "exact-greedy-cohort-burst.lease.v1",
        "schedule_generation": 1,
        "graph_generation": 1,
        "graph_identity_sha256": GRAPH_SHA,
        "ordered_sequence_ids": sequence_ids,
        "requested_width": 8,
        "authorized_width": 8,
        "decision_now_ns": decision_now_ns,
        "cost_table_sha256": cost_table_sha256,
        "predicted_duration_ns": 804,
        "global_slack_ns": global_slack_ns,
        "rows": rows,
    }


def _decision_row(
    *,
    case: dict,
    sequence_ids: list[int],
    context_buckets: list[int],
    cost_table_sha256: str,
    arrival_ns_by_sequence: dict[int, int],
    remaining_output_tokens: list[int] | None = None,
) -> dict:
    remaining = (
        [8] * len(sequence_ids)
        if remaining_output_tokens is None
        else remaining_output_tokens
    )
    decision_now_ns = max(arrival_ns_by_sequence.values()) + 10
    protected = [{
        "sequence_id": sequence_id,
        "category": "cohort",
        "service_class": "default",
        "age_ns": decision_now_ns - arrival_ns_by_sequence[sequence_id],
        "slack_ns": (
            1_000_000_000
            - (decision_now_ns - arrival_ns_by_sequence[sequence_id])
            - 2_000_000
        ),
    } for sequence_id in sequence_ids]
    global_slack_ns = min(row["slack_ns"] for row in protected)
    return {
        "schema_version": "slo-cohort-burst.decision-evidence.v1",
        "case": case,
        "decision": {
            "schema_version": "slo-cohort-burst.decision.v1",
            "decision_now_ns": decision_now_ns,
            "schedule_generation": 1,
            "batch_size": 2,
            "ordered_cohort_sequence_ids": sequence_ids,
            "queue_depths": {
                "waiting": 0,
                "prefilling": 0,
                "running": 2,
            },
            "context_buckets": [{
                "sequence_id": sequence_id,
                "context_bucket": context_bucket,
                "remaining_output_tokens": remaining_tokens,
                "writable_tokens": 8,
            } for (
                sequence_id,
                context_bucket,
                remaining_tokens,
            ) in zip(
                sequence_ids,
                context_buckets,
                remaining,
            )],
            "protected_requests": protected,
            "global_slack_ns": global_slack_ns,
            "predicted_cost_ns_by_width": {
                "8": 804,
                "4": 404,
                "2": 204,
            },
            "structural_eligibility_by_width": {
                "8": True,
                "4": True,
                "2": True,
            },
            "selected_width": 8,
            "reason": "selected",
            "cost_table_sha256": cost_table_sha256,
        },
    }


def _execution_row(
    *,
    case: dict,
    sequence_ids: list[int],
    context_buckets: list[int],
    remaining_output_tokens: list[int],
    eos_sequence_id: int | None,
    cost_table_sha256: str,
    decision_now_ns: int,
    global_slack_ns: int,
) -> dict:
    lease = _lease_payload(
        sequence_ids=sequence_ids,
        context_buckets=context_buckets,
        remaining_output_tokens=remaining_output_tokens,
        cost_table_sha256=cost_table_sha256,
        decision_now_ns=decision_now_ns,
        global_slack_ns=global_slack_ns,
    )
    lease_identity = _sha(lease)
    result_rows = []
    publication_rows = []
    generated = {}
    committed = {}
    discarded = {}
    wasted = 0
    for sequence_id in sequence_ids:
        tokens = (
            [2, *([99] * 7)]
            if sequence_id == eos_sequence_id
            else list(range(11, 19))
        )
        commit_tokens = [2] if sequence_id == eos_sequence_id else tokens
        result_rows.append({
            "sequence_id": sequence_id,
            "sequence_generation": 1,
            "tokens": tokens,
            "final_position": (
                context_buckets[sequence_ids.index(sequence_id)] + 7
            ),
            "final_context_length": (
                context_buckets[sequence_ids.index(sequence_id)] + 8
            ),
            "final_physical_slot": (
                sequence_ids.index(sequence_id) * 16 + 8
            ),
            "sampled_logits": [],
        })
        publication_rows.append({
            "sequence_id": sequence_id,
            "commit_tokens": commit_tokens,
        })
        generated[str(sequence_id)] = len(tokens)
        committed[str(sequence_id)] = len(commit_tokens)
        discarded[str(sequence_id)] = len(tokens) - len(commit_tokens)
        wasted += len(tokens) - len(commit_tokens)
    result = {
        "schema_version": (
            "exact-greedy-cohort-burst.result-identity.v1"
        ),
        "lease_identity_sha256": lease_identity,
        "graph_identity_sha256": GRAPH_SHA,
        "graph_generation": 1,
        "replay_count": 8,
        "rows": result_rows,
        "token_d2h_calls": 1,
        "sampled_logit_d2h_calls": 0,
    }
    result_identity = _sha(result)
    return {
        "schema_version": "slo-cohort-burst.execution-evidence.v1",
        "case": case,
        "lease": lease,
        "lease_identity_sha256": lease_identity,
        "result": result,
        "result_identity_sha256": result_identity,
        "publication": {
            "ordered_sequence_ids": sequence_ids,
            "rows": publication_rows,
        },
        "execution": {
            "schema_version": (
                "exact-greedy-cohort-burst.execution.v1"
            ),
            "lease_identity_sha256": lease_identity,
            "result_identity_sha256": result_identity,
            "graph_identity_sha256": GRAPH_SHA,
            "requested_width": 8,
            "authorized_width": 8,
            "completed_replay_count": 8,
            "predicted_duration_ns": 804,
            "actual_duration_ns": 6_400_000,
            "host_visible_publication_gap_ns": 6_400_000,
            "token_d2h_calls": 1,
            "token_d2h_bytes": len(sequence_ids) * 8 * 8,
            "sampled_logit_d2h_calls": 0,
            "generated_token_counts": generated,
            "committed_token_counts": committed,
            "eos_discarded_token_counts": discarded,
            "post_eos_wasted_tokens": wasted,
            "post_eos_wasted_forwards": wasted,
            "post_eos_wasted_forward_fraction": wasted / 16,
            "fallback_reason": None,
            "failure_reason": None,
            "rollback_reason": None,
            "quarantined": False,
            "quarantine_reason": None,
            "pending_inventory": {"leases": 0, "transactions": 0},
        },
    }


def _correctness_rows(*, cost_table_sha256: str) -> list[dict]:
    rows = []
    for batch_size in (1, 2, 4, 8):
        for width in (1, 2, 4, 8):
            row_results = []
            for row_index in range(batch_size):
                tokens = [1] * width
                logits = [0.1, 0.9]
                logits_sha256 = _sha([logits] * width)
                row_results.append({
                    "row_index": row_index,
                    "baseline_output_token_ids": tokens,
                    "candidate_output_token_ids": tokens,
                    "baseline_output_text_sha256": _sha(tokens),
                    "candidate_output_text_sha256": _sha(tokens),
                    "baseline_sampled_logits_sha256": logits_sha256,
                    "candidate_sampled_logits_sha256": logits_sha256,
                    "baseline_argmax_token_ids": [1] * width,
                    "candidate_argmax_token_ids": [1] * width,
                })
            rows.append({
                "schema_version": (
                    "slo-cohort-burst.correctness-case.v2"
                ),
                "batch_size": batch_size,
                "burst_width": width,
                "rows": row_results,
                "candidate_execution": (
                    None
                    if width == 1
                    else _correctness_execution_row(
                        batch_size=batch_size,
                        burst_width=width,
                        cost_table_sha256=cost_table_sha256,
                    )
                ),
                "duplicate_forwards": 0,
                "duplicate_commits": 0,
                "unauthorized_kv_publications": 0,
                "pending_leases_after_case": 0,
            })
    return rows


def _correctness_execution_row(
    *,
    batch_size: int,
    burst_width: int,
    cost_table_sha256: str,
) -> dict:
    sequence_ids = list(range(10_000, 10_000 + batch_size))
    case = {
        "workload": "correctness",
        "load": "correctness",
        "repetition": 0,
        "arm": "candidate",
        "batch_size": batch_size,
        "burst_width": burst_width,
    }
    wrapper = _execution_row(
        case=case,
        sequence_ids=sequence_ids,
        context_buckets=[256] * batch_size,
        remaining_output_tokens=[burst_width] * batch_size,
        eos_sequence_id=None,
        cost_table_sha256=cost_table_sha256,
        decision_now_ns=100,
        global_slack_ns=1_000_000_000,
    )
    wrapper["lease"]["requested_width"] = burst_width
    wrapper["lease"]["authorized_width"] = burst_width
    wrapper["result"]["replay_count"] = burst_width
    wrapper["result"]["sampled_logit_d2h_calls"] = 1
    wrapper["execution"]["requested_width"] = burst_width
    wrapper["execution"]["authorized_width"] = burst_width
    wrapper["execution"]["completed_replay_count"] = burst_width
    wrapper["execution"]["sampled_logit_d2h_calls"] = 1
    wrapper["execution"]["generated_token_counts"] = {
        str(sequence_id): burst_width
        for sequence_id in sequence_ids
    }
    wrapper["execution"]["committed_token_counts"] = dict(
        wrapper["execution"]["generated_token_counts"]
    )
    wrapper["execution"]["token_d2h_bytes"] = (
        batch_size * burst_width * 8
    )
    for lease_row in wrapper["lease"]["rows"]:
        lease_row["last_write_position"] = (
            lease_row["first_write_position"] + burst_width - 1
        )
        lease_row["last_physical_slot"] = (
            lease_row["first_physical_slot"] + burst_width - 1
        )
    for row_index, (result_row, publication_row) in enumerate(zip(
        wrapper["result"]["rows"],
        wrapper["publication"]["rows"],
    )):
        tokens = [1] * burst_width
        result_row["tokens"] = tokens
        result_row["sampled_logits"] = [[0.1, 0.9]] * burst_width
        result_row["final_position"] = 256 + burst_width - 1
        result_row["final_context_length"] = 256 + burst_width
        result_row["final_physical_slot"] = (
            row_index * 16 + burst_width
        )
        publication_row["commit_tokens"] = tokens
    graph_identity = GRAPH_SHA
    wrapper["lease"]["graph_identity_sha256"] = graph_identity
    wrapper["result"]["graph_identity_sha256"] = graph_identity
    wrapper["execution"]["graph_identity_sha256"] = graph_identity
    _refresh_execution_identities(wrapper)
    return wrapper


def complete_synthetic_bundle() -> dict:
    source_identity = {
        "source_commit": SOURCE_COMMIT,
        "source_patch_sha256": verifier._source_tree_sha256(ROOT),
        "model": "Qwen3-0.6B",
        "checkpoint_sha256": MODEL_SHA,
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
        "gpu_name": "NVIDIA A100 80GB PCIe",
        "tensor_parallel_size": 1,
        "dtype": "torch.bfloat16",
        "config_sha256": CONFIG_SHA,
    }
    source_manifest = {
        "schema_version": "slo-cohort-burst.source-manifest.v1",
        "source_identity": source_identity,
        "dirty": False,
        "source_sha256": {
            relative: hashlib.sha256(
                (ROOT / relative).read_bytes()
            ).hexdigest()
            for relative in verifier.QUALIFICATION_SOURCE_PATHS
        },
    }
    environment = {
        "schema_version": "slo-cohort-burst.environment.v2",
        "source_commit": SOURCE_COMMIT,
        "model": "Qwen3-0.6B",
        "checkpoint_sha256": MODEL_SHA,
        "gpu_uuid": source_identity["gpu_uuid"],
        "gpu_name": source_identity["gpu_name"],
        "tensor_parallel_size": 1,
        "temperature": 0.0,
        "completion_only": True,
        "eos_token_id": 2,
        "target_itl_ns": 40_000_000,
        "target_ttft_ns": 1_000_000_000,
        "reserve_ns": 2_000_000,
        "graph_identity_sha256_by_shape": {
            (
                f"b{batch_size}-w{block_table_width}"
                "-trace0"
            ): GRAPH_SHA
            for batch_size in range(1, 9)
            for block_table_width in (2, 9, 33)
        } | {
            f"b{batch_size}-w2-trace1": GRAPH_SHA
            for batch_size in (1, 2, 4, 8)
        },
    }
    cost_profile_rows = _cost_profile_rows(source_identity)
    cost_table = _cost_table(source_identity, cost_profile_rows)
    arrival_traces = producer.build_frozen_arrival_traces(
        source_commit=SOURCE_COMMIT,
        saturation_rps_by_workload=producer._saturation_rates(
            cost_table
        ),
    )
    arrival_cases = arrival_traces["cases"]
    arrivals_by_case = {
        (
            case["workload"],
            case["load"],
            case["repetition"],
        ): case["requests"]
        for case in arrival_cases
    }
    request_rows = []
    decision_rows = []
    execution_rows = []
    sequence_id = 0
    for workload in WORKLOADS:
        for load in LOADS:
            for repetition in range(REPETITIONS):
                requests = arrivals_by_case[(
                    workload,
                    load,
                    repetition,
                )]
                for arm in ARMS:
                    case = {
                        "workload": workload,
                        "load": load,
                        "repetition": repetition,
                        "arm": arm,
                    }
                    current_sequences = []
                    arrival_ns_by_sequence = {}
                    for request_index, request in enumerate(requests):
                        sequence_id += 1
                        current_sequences.append(sequence_id)
                        arrival_ns_by_sequence[sequence_id] = (
                            1_000_000
                            + request["arrival_offset_ns"]
                        )
                        eos = (
                            workload == "bursty_eos"
                            and request_index == 0
                        )
                        request_rows.append({
                            "schema_version": (
                                "slo-cohort-burst.request-evidence.v1"
                            ),
                            "case": case,
                            "prompt_sha256": request["prompt_sha256"],
                            "maximum_output_tokens": request[
                                "maximum_output_tokens"
                            ],
                            "ignore_eos": workload != "bursty_eos",
                            "peak_cuda_reserved_bytes": (
                                1_000 if arm == "baseline" else 1_040
                            ),
                            "request": _request_contract(
                                request_id=request["request_id"],
                                sequence_id=sequence_id,
                                arm=arm,
                                eos=eos,
                                maximum_output_tokens=request[
                                    "maximum_output_tokens"
                                ],
                                arrival_ns=arrival_ns_by_sequence[
                                    sequence_id
                                ],
                            ),
                        })
                    if arm == "candidate":
                        for offset in range(0, len(current_sequences), 2):
                            pair = current_sequences[offset:offset + 2]
                            decision_row = _decision_row(
                                case=case,
                                sequence_ids=pair,
                                context_buckets=[
                                    requests[index][
                                        "prompt_tokens"
                                    ]
                                    for index in range(
                                        offset,
                                        offset + len(pair),
                                    )
                                ],
                                cost_table_sha256=(
                                    cost_table["table_sha256"]
                                ),
                                arrival_ns_by_sequence={
                                    sequence_id: (
                                        arrival_ns_by_sequence[sequence_id]
                                    )
                                    for sequence_id in pair
                                },
                                remaining_output_tokens=[
                                    requests[index][
                                        "maximum_output_tokens"
                                    ]
                                    for index in range(
                                        offset,
                                        offset + len(pair),
                                    )
                                ],
                            )
                            decision_rows.append(decision_row)
                            execution_row = _execution_row(
                                case=case,
                                sequence_ids=pair,
                                context_buckets=[
                                    requests[index]["prompt_tokens"]
                                    for index in range(
                                        offset,
                                        offset + len(pair),
                                    )
                                ],
                                remaining_output_tokens=[
                                    requests[index][
                                        "maximum_output_tokens"
                                    ]
                                    for index in range(
                                        offset,
                                        offset + len(pair),
                                    )
                                ],
                                eos_sequence_id=(
                                    pair[0]
                                    if workload == "bursty_eos"
                                    and offset == 0
                                    else None
                                ),
                                cost_table_sha256=(
                                    cost_table["table_sha256"]
                                ),
                                decision_now_ns=decision_row[
                                    "decision"
                                ]["decision_now_ns"],
                                global_slack_ns=decision_row[
                                    "decision"
                                ]["global_slack_ns"],
                            )
                            graph_identity = _canonical_graph_sha(
                                repetition
                            )
                            execution_row["lease"][
                                "graph_identity_sha256"
                            ] = graph_identity
                            execution_row["result"][
                                "graph_identity_sha256"
                            ] = graph_identity
                            execution_row["execution"][
                                "graph_identity_sha256"
                            ] = graph_identity
                            _refresh_execution_identities(execution_row)
                            execution_rows.append(execution_row)
    fallback = deepcopy(decision_rows[0])
    fallback["decision"]["schedule_generation"] = 999_999
    fallback["decision"]["selected_width"] = 1
    fallback["decision"]["reason"] = "kv_block_boundary"
    fallback["decision"]["global_slack_ns"] = 0
    fallback["decision"]["context_buckets"][0][
        "writable_tokens"
    ] = 1
    fallback["decision"]["structural_eligibility_by_width"] = {
        "8": False,
        "4": False,
        "2": False,
    }
    decision_rows.append(fallback)
    grouped_requests = {}
    for wrapper in request_rows:
        case = wrapper["case"]
        grouped_requests.setdefault((
            case["workload"],
            case["load"],
            case["repetition"],
            case["arm"],
        ), []).append(wrapper)
    summary = verifier._reconstruct_summary(
        grouped_requests=grouped_requests,
        correctness_passed=True,
        lifecycle_closed=True,
        wasted_forwards=sum(
            row["execution"]["post_eos_wasted_forwards"]
            for row in execution_rows
        ),
        total_forward_slots=sum(
            row["execution"]["completed_replay_count"]
            * len(row["execution"]["generated_token_counts"])
            for row in execution_rows
        ),
    )
    bundle = {
        "source_manifest": source_manifest,
        "environment": environment,
        "cost_profile_rows": cost_profile_rows,
        "arrival_traces": arrival_traces,
        "cost_table": cost_table,
        "decision_rows": decision_rows,
        "execution_rows": execution_rows,
        "request_rows": request_rows,
        "correctness_rows": _correctness_rows(
            cost_table_sha256=cost_table["table_sha256"],
        ),
        "canonical_graph_identities": (
            _canonical_graph_identities()
        ),
        "summary": summary,
    }
    bundle["manifest"] = {
        "schema_version": "slo-cohort-burst.manifest.v1",
        "artifact_sha256": {
            relative: _artifact_sha(bundle, relative)
            for relative in verifier.AUTHORITATIVE_ARTIFACTS
        },
    }
    return bundle


def _refresh_artifact_hash(bundle: dict, relative: str) -> None:
    bundle["manifest"]["artifact_sha256"][relative] = (
        _artifact_sha(bundle, relative)
    )


def _refresh_execution_identities(wrapper: dict) -> None:
    lease_identity = _sha(wrapper["lease"])
    wrapper["lease_identity_sha256"] = lease_identity
    wrapper["result"]["lease_identity_sha256"] = lease_identity
    wrapper["execution"]["lease_identity_sha256"] = lease_identity
    result_identity = _sha(wrapper["result"])
    wrapper["result_identity_sha256"] = result_identity
    wrapper["execution"]["result_identity_sha256"] = result_identity


def _mutate(bundle: dict, mutation: str) -> None:
    if mutation == "decision_width":
        bundle["decision_rows"][0]["decision"]["selected_width"] = 4
        _refresh_artifact_hash(bundle, "decision_rows.jsonl")
    elif mutation == "protected_request_slack":
        bundle["decision_rows"][0]["decision"][
            "protected_requests"
        ][0]["slack_ns"] += 1
        _refresh_artifact_hash(bundle, "decision_rows.jsonl")
    elif mutation == "lease_identity":
        bundle["execution_rows"][0]["lease_identity_sha256"] = "e" * 64
        _refresh_artifact_hash(bundle, "execution_rows.jsonl")
    elif mutation == "row_order":
        bundle["execution_rows"][0]["result"]["rows"].reverse()
        bundle["execution_rows"][0]["result_identity_sha256"] = _sha(
            bundle["execution_rows"][0]["result"]
        )
        bundle["execution_rows"][0]["execution"][
            "result_identity_sha256"
        ] = bundle["execution_rows"][0]["result_identity_sha256"]
        _refresh_artifact_hash(bundle, "execution_rows.jsonl")
    elif mutation == "eos_prefix":
        eos_row = next(
            row
            for row in bundle["execution_rows"]
            if row["execution"]["post_eos_wasted_tokens"]
        )
        eos_row["publication"]["rows"][0]["commit_tokens"].append(99)
        _refresh_artifact_hash(bundle, "execution_rows.jsonl")
    elif mutation == "request_timestamp":
        row = bundle["request_rows"][0]["request"]
        row["first_token_visible_ns"] += 1
        _refresh_artifact_hash(bundle, "request_rows.jsonl")
    elif mutation == "reserved_memory":
        bundle["request_rows"][0]["peak_cuda_reserved_bytes"] = -1
        _refresh_artifact_hash(bundle, "request_rows.jsonl")
    elif mutation == "late_arm_order":
        bundle["arrival_traces"]["arm_order_by_repetition"][4] = [
            "candidate",
            "baseline",
        ]
        _refresh_artifact_hash(bundle, "arrival_traces.json")
    elif mutation == "arrival_offset":
        bundle["arrival_traces"]["cases"][0]["requests"][1][
            "arrival_offset_ns"
        ] += 1
        _refresh_artifact_hash(bundle, "arrival_traces.json")
    elif mutation == "arrival_schedule":
        for wrapper in bundle["request_rows"]:
            if (
                wrapper["case"]["workload"] == "decode_heavy"
                and wrapper["case"]["load"] == "low"
                and wrapper["case"]["repetition"] == 0
                and wrapper["request"]["request_id"]
                == "decode_heavy-low-r0-q1"
            ):
                request = wrapper["request"]
                for field in (
                    "arrival_ns",
                    "prefill_start_ns",
                    "prefill_complete_ns",
                    "first_token_visible_ns",
                    "completion_ns",
                ):
                    request[field] += 1
                request["token_visible_ns"] = [
                    timestamp + 1
                    for timestamp in request["token_visible_ns"]
                ]
        _refresh_artifact_hash(bundle, "request_rows.jsonl")
    elif mutation == "target_itl":
        bundle["environment"]["target_itl_ns"] = 41_000_000
        _refresh_artifact_hash(bundle, "environment.json")
    elif mutation == "queue_depth":
        bundle["decision_rows"][0]["decision"]["queue_depths"] = {
            "waiting": 999,
            "prefilling": 999,
            "running": 999,
        }
        _refresh_artifact_hash(bundle, "decision_rows.jsonl")
    elif mutation == "writable_capacity":
        bundle["decision_rows"][-1]["decision"]["context_buckets"][0][
            "writable_tokens"
        ] = 8
        _refresh_artifact_hash(bundle, "decision_rows.jsonl")
    elif mutation == "source_inventory":
        bundle["source_manifest"]["source_sha256"].pop(
            verifier.QUALIFICATION_SOURCE_PATHS[-1]
        )
        _refresh_artifact_hash(bundle, "source_manifest.json")
    elif mutation == "source_patch":
        bundle["source_manifest"]["source_identity"] = dict(
            bundle["source_manifest"]["source_identity"]
        )
        bundle["source_manifest"]["source_identity"][
            "source_patch_sha256"
        ] = "f" * 64
        _refresh_artifact_hash(bundle, "source_manifest.json")
    elif mutation == "fallback_reason":
        bundle["decision_rows"][-1]["decision"]["reason"] = (
            "graph_unavailable"
        )
        _refresh_artifact_hash(bundle, "decision_rows.jsonl")
    elif mutation == "missing_execution":
        bundle["execution_rows"].pop()
        _refresh_artifact_hash(bundle, "execution_rows.jsonl")
    elif mutation == "result_sequence_generation":
        row = bundle["execution_rows"][0]
        row["result"]["rows"][0]["sequence_generation"] += 1
        _refresh_execution_identities(row)
        _refresh_artifact_hash(bundle, "execution_rows.jsonl")
    elif mutation == "lease_initial_sequence_length":
        row = bundle["execution_rows"][0]
        authority = row["lease"]["rows"][0]
        authority["initial_sequence_length"] += 1
        authority["first_write_position"] += 1
        authority["last_write_position"] += 1
        _refresh_execution_identities(row)
        _refresh_artifact_hash(bundle, "execution_rows.jsonl")
    elif mutation == "lease_remaining_output_tokens":
        row = bundle["execution_rows"][0]
        row["lease"]["rows"][0]["remaining_output_tokens"] += 1
        _refresh_execution_identities(row)
        _refresh_artifact_hash(bundle, "execution_rows.jsonl")
    elif mutation == "result_final_position":
        row = bundle["execution_rows"][0]
        row["result"]["rows"][0]["final_position"] += 1
        _refresh_execution_identities(row)
        _refresh_artifact_hash(bundle, "execution_rows.jsonl")
    elif mutation == "result_final_context_length":
        row = bundle["execution_rows"][0]
        row["result"]["rows"][0]["final_context_length"] += 1
        _refresh_execution_identities(row)
        _refresh_artifact_hash(bundle, "execution_rows.jsonl")
    elif mutation == "result_final_physical_slot":
        row = bundle["execution_rows"][0]
        row["result"]["rows"][0]["final_physical_slot"] += 1
        _refresh_execution_identities(row)
        _refresh_artifact_hash(bundle, "execution_rows.jsonl")
    elif mutation == "result_sampled_logits":
        row = bundle["execution_rows"][0]
        row["result"]["rows"][0]["sampled_logits"] = [[0.0]]
        _refresh_execution_identities(row)
        _refresh_artifact_hash(bundle, "execution_rows.jsonl")
    elif mutation == "token_d2h_bytes":
        bundle["execution_rows"][0]["execution"]["token_d2h_bytes"] += 8
        _refresh_artifact_hash(bundle, "execution_rows.jsonl")
    elif mutation == "execution_duration_disagreement":
        bundle["execution_rows"][0]["execution"][
            "host_visible_publication_gap_ns"
        ] += 1
        _refresh_artifact_hash(bundle, "execution_rows.jsonl")
    elif mutation == "correctness_boolean_row_index":
        bundle["correctness_rows"][0]["rows"][0]["row_index"] = False
        _refresh_artifact_hash(bundle, "correctness_rows.jsonl")
    elif mutation == "correctness_output_text_digest":
        row = bundle["correctness_rows"][0]["rows"][0]
        row["baseline_output_text_sha256"] = "not-a-digest"
        row["candidate_output_text_sha256"] = "not-a-digest"
        _refresh_artifact_hash(bundle, "correctness_rows.jsonl")
    elif mutation == "correctness_graph_identity":
        row = next(
            row
            for row in bundle["correctness_rows"]
            if row["candidate_execution"] is not None
        )
        wrapper = row["candidate_execution"]
        wrapper["lease"]["graph_identity_sha256"] = "e" * 64
        wrapper["result"]["graph_identity_sha256"] = "e" * 64
        wrapper["execution"]["graph_identity_sha256"] = "e" * 64
        _refresh_execution_identities(wrapper)
        _refresh_artifact_hash(bundle, "correctness_rows.jsonl")
    elif mutation == "graph_shape_inventory":
        bundle["environment"][
            "graph_identity_sha256_by_shape"
        ].pop("b8-w33-trace0")
        _refresh_artifact_hash(bundle, "environment.json")
    elif mutation == "graph_shape_identity":
        bundle["canonical_graph_identities"][
            "graph_identity_sha256_by_repetition"
        ]["0"]["b2-w2-trace0"] = "e" * 64
        _refresh_artifact_hash(
            bundle,
            "canonical_graph_identities.json",
        )
    elif mutation == "duplicate_lease_block_identity":
        row = bundle["execution_rows"][0]
        authority = row["lease"]["rows"][0]
        authority["block_table_identity"].append(
            list(authority["block_table_identity"][0])
        )
        _refresh_execution_identities(row)
        _refresh_artifact_hash(bundle, "execution_rows.jsonl")
    elif mutation == "source_sha":
        bundle["source_manifest"]["source_sha256"][SOURCE_FILE] = "f" * 64
        _refresh_artifact_hash(bundle, "source_manifest.json")
    elif mutation == "artifact_hash":
        bundle["manifest"]["artifact_sha256"][
            "request_rows.jsonl"
        ] = "f" * 64
    else:
        raise AssertionError(f"unknown mutation: {mutation}")


@pytest.mark.parametrize(
    "mutation",
    (
        "decision_width",
        "protected_request_slack",
        "lease_identity",
        "row_order",
        "eos_prefix",
        "request_timestamp",
        "reserved_memory",
        "late_arm_order",
        "arrival_offset",
        "arrival_schedule",
        "target_itl",
        "queue_depth",
        "writable_capacity",
        "source_inventory",
        "source_patch",
        "fallback_reason",
        "missing_execution",
        "result_sequence_generation",
        "lease_initial_sequence_length",
        "lease_remaining_output_tokens",
        "result_final_position",
        "result_final_context_length",
        "result_final_physical_slot",
        "result_sampled_logits",
        "token_d2h_bytes",
        "execution_duration_disagreement",
        "correctness_boolean_row_index",
        "correctness_output_text_digest",
        "correctness_graph_identity",
        "graph_shape_inventory",
        "graph_shape_identity",
        "duplicate_lease_block_identity",
        "source_sha",
        "artifact_hash",
    ),
)
def test_verifier_rejects_authoritative_mutation(mutation: str) -> None:
    bundle = complete_synthetic_bundle()
    _mutate(bundle, mutation)
    with pytest.raises(ValueError):
        verifier.verify_slo_cohort_burst_bundle(
            bundle,
            source_root=ROOT,
        )


def test_verifier_is_independent_and_reconstructs_complete_bundle() -> None:
    source = Path(verifier.__file__).read_text(encoding="utf-8")
    assert "slo_cohort_burst_gate import" not in source
    result = verifier.verify_slo_cohort_burst_bundle(
        complete_synthetic_bundle(),
        source_root=ROOT,
    )
    assert result["verified"] is True
    assert result["classification"] == (
        "GO_SLO_AWARE_COHORT_DECODE_BURST"
    )
    assert result["request_row_count"] == 2_340
    assert result["decision_row_count"] == 586
    assert result["execution_row_count"] == 585
    assert result["correctness_case_count"] == 16


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_repetition",
        "extra_repetition",
        "missing_shape",
        "wrong_execution_identity",
    ),
)
def test_verifier_rejects_unclosed_canonical_graph_identity_inventory(
    mutation: str,
) -> None:
    bundle = complete_synthetic_bundle()
    identities = bundle["canonical_graph_identities"][
        "graph_identity_sha256_by_repetition"
    ]
    if mutation == "missing_repetition":
        identities.pop("4")
    elif mutation == "extra_repetition":
        identities["5"] = dict(identities["4"])
    elif mutation == "missing_shape":
        identities["0"].pop("b2-w2-trace0")
    elif mutation == "wrong_execution_identity":
        row = next(
            row
            for row in bundle["execution_rows"]
            if row["case"]["repetition"] == 0
        )
        wrong = "f" * 64
        row["lease"]["graph_identity_sha256"] = wrong
        row["result"]["graph_identity_sha256"] = wrong
        row["execution"]["graph_identity_sha256"] = wrong
        _refresh_execution_identities(row)
        _refresh_artifact_hash(bundle, "execution_rows.jsonl")
    else:
        raise AssertionError(mutation)
    if mutation != "wrong_execution_identity":
        _refresh_artifact_hash(
            bundle,
            "canonical_graph_identities.json",
        )

    with pytest.raises(ValueError):
        verifier.verify_slo_cohort_burst_bundle(
            bundle,
            source_root=ROOT,
        )


def test_directory_verifier_writes_requested_receipt(
    tmp_path: Path,
) -> None:
    bundle = complete_synthetic_bundle()
    for relative, key in verifier.ARTIFACT_KEYS.items():
        path = tmp_path / relative
        if relative.endswith(".jsonl"):
            _write_jsonl(path, bundle[key])
        else:
            _write_json(path, bundle[key])
    output = tmp_path / "local_verify.json"

    result = verifier.verify_artifact_directory(
        tmp_path,
        source_root=ROOT,
        output=output,
    )

    assert json.loads(output.read_text()) == result
    assert result["manifest_sha256"] == hashlib.sha256(
        (tmp_path / "manifest.json").read_bytes()
    ).hexdigest()


def test_correctness_stage_verifies_closed_bxk_matrix(
    tmp_path: Path,
) -> None:
    complete = complete_synthetic_bundle()
    bundle = {
        key: complete[key]
        for key in (
            "source_manifest",
            "environment",
            "cost_profile_rows",
            "cost_table",
            "correctness_rows",
        )
    }
    bundle["manifest"] = {
        "schema_version": verifier.MANIFEST_SCHEMA_VERSION,
        "artifact_sha256": {
            relative: hashlib.sha256(
                _artifact_bytes_for_key(
                    relative,
                    bundle[verifier.CORRECTNESS_ARTIFACT_KEYS[relative]],
                )
            ).hexdigest()
            for relative in verifier.CORRECTNESS_AUTHORITATIVE_ARTIFACTS
        },
    }
    for relative, key in verifier.CORRECTNESS_ARTIFACT_KEYS.items():
        path = tmp_path / relative
        if relative.endswith(".jsonl"):
            _write_jsonl(path, bundle[key])
        else:
            _write_json(path, bundle[key])

    result = verifier.verify_artifact_directory(
        tmp_path,
        source_root=ROOT,
        stage="correctness",
    )

    assert result["verified"] is True
    assert result["classification"] == (
        "PASS_CORRECTNESS_AND_LIFECYCLE"
    )
    assert result["correctness_case_count"] == 16


def _artifact_bytes_for_key(relative: str, payload: object) -> bytes:
    return (
        _jsonl_bytes(payload)
        if relative.endswith(".jsonl")
        else _canonical_bytes(payload)
    )
