from __future__ import annotations

import hashlib
import json
from pathlib import Path

import qwen35_tp4_hybrid_prefix_benchmark_v2_contract as contract


BYTE_FIELDS = (
    "hybrid_cache_current_unique_physical_bytes",
    "hybrid_cache_current_logical_referenced_bytes",
    "hybrid_cache_current_metadata_bytes",
    "hybrid_cache_deduplicated_bytes",
    "hybrid_cache_peak_unique_physical_bytes",
    "hybrid_cache_peak_logical_referenced_bytes",
    "hybrid_cache_peak_metadata_bytes",
)
TRANSACTION_COUNTER_FIELDS = (
    "hybrid_cache_hits",
    "hybrid_cache_misses",
    "hybrid_cache_evictions",
    "hybrid_cache_validation_failures",
    "hybrid_cache_failed_restores",
    "hybrid_cache_quarantines",
    "hybrid_cache_rollbacks",
    "hybrid_cache_failed_rollbacks",
    "hybrid_cache_corruption_events",
    "hybrid_cache_partial_restore_attempts",
    "hybrid_cache_fallbacks",
    "hybrid_cache_mixed_representation_events",
    "hybrid_cache_missing_layer_events",
    "oom_events",
    "undeclared_eviction_events",
)
RANK_LOCAL_OBSERVATION_FIELDS = (
    "cuda_allocated_bytes",
    "cuda_reserved_bytes",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
    "encode_workspace_peak_allocated_bytes",
    "encode_workspace_peak_reserved_bytes",
    "decode_workspace_peak_allocated_bytes",
    "decode_workspace_peak_reserved_bytes",
)
SNAPSHOT_FILE_FIELDS = ("path", "sha256", "bytes", "type")
TENSOR_IDENTITY_FIELDS = (
    "case_id",
    "profile",
    "representation",
    "representation_version",
    "codec",
    "rank",
    "world_size",
)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _write_bytes(output_dir, relative_path, data):
    path = Path(output_dir) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return relative_path, _sha256(data)


def _token_bytes(token_ids):
    return (
        json.dumps(
            list(token_ids),
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _profile_evidence(profile):
    if profile == contract.P2_PROFILE:
        return (
            contract.P2_REPRESENTATION,
            contract.P2_REPRESENTATION_VERSION,
            contract.P2_CODEC_ID,
        )
    if profile == contract.P1_REFERENCE_PROFILE:
        return (
            "exact_restore",
            contract.P1_REPRESENTATION_VERSION,
            None,
        )
    return None, None, None


def build_case_execution(case, *, workload_payload):
    canonical_payload = contract.workload_payload(case.workload)
    if workload_payload != canonical_payload:
        raise ValueError("workload payload does not match case")
    concurrency = (
        contract.CORRECTNESS_CONCURRENCY
        if case.phase == "correctness"
        else case.concurrency
    )
    return {
        "case_id": case.case_id,
        "profile": case.profile,
        "workload": case.workload,
        "phase": case.phase,
        "repetition": case.repetition,
        "workload_payload": canonical_payload,
        "concurrency": concurrency,
        "serial_correctness": case.phase == "correctness",
    }


def aggregate_process_rows(rows):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("rank rows must be a non-empty list")
    for row in rows:
        contract.validate_process_row(row)
    by_rank = {}
    for row in rows:
        rank = row.get("rank")
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank in by_rank
        ):
            raise ValueError("rank inventory is invalid")
        by_rank[rank] = row
    world_sizes = {row.get("world_size") for row in rows}
    if len(world_sizes) != 1:
        raise ValueError("rank world size is inconsistent")
    world_size = next(iter(world_sizes))
    rank_inventory = sorted(by_rank)
    if rank_inventory != list(range(world_size)):
        raise ValueError("rank inventory is incomplete")

    reference = by_rank[0]
    for field in TRANSACTION_COUNTER_FIELDS:
        if any(
            by_rank[rank][field] != reference[field]
            for rank in rank_inventory[1:]
        ):
            raise ValueError(
                f"transaction counter parity mismatch: {field}"
            )
    capacities = {
        row["same_budget_entry_capacity"] for row in rows
    }
    if len(capacities) != 1:
        raise ValueError("same-budget capacity parity mismatch")

    aggregate = {
        "rank_inventory": rank_inventory,
        "same_budget_entry_capacity": capacities.pop(),
        "rank_observations": [
            {
                "rank": row["rank"],
                **{
                    field: row[field]
                    for field in RANK_LOCAL_OBSERVATION_FIELDS
                },
            }
            for row in sorted(rows, key=lambda item: item["rank"])
        ],
    }
    aggregate.update({
        field: sum(row[field] for row in rows)
        for field in BYTE_FIELDS
    })
    aggregate.update({
        field: reference[field]
        for field in TRANSACTION_COUNTER_FIELDS
    })
    return aggregate


def _validate_request_observations(observations, workload_payload):
    if not isinstance(observations, (list, tuple)):
        raise ValueError("request observations must be a list")
    expected_request_ids = [
        f"request-{continuation['request_index']}"
        for continuation in workload_payload["continuations"]
    ]
    actual_request_ids = [
        observation.get("request_id")
        if isinstance(observation, dict)
        else None
        for observation in observations
    ]
    if actual_request_ids != expected_request_ids:
        raise ValueError(
            "request observations do not exactly cover canonical "
            "request IDs in order"
        )


def _validate_process_rows_for_execution(
    rows,
    *,
    case,
    execution,
    evidence_bindings,
    process_identity,
):
    representation, version, codec = _profile_evidence(case.profile)
    expected_fields = {
        "case_id": case.case_id,
        "profile": case.profile,
        "representation": representation,
        "representation_version": version,
        "codec": codec,
        "workload": case.workload,
        "phase": case.phase,
        "repetition": case.repetition,
        "sampling_temperature": contract.SAMPLING_TEMPERATURE,
        "sampling_max_tokens": execution["workload_payload"]["spec"][
            "generated_tokens"
        ],
        "sampling_ignore_eos": contract.SAMPLING_IGNORE_EOS,
        "sampling_seed": contract.workload_sampling_seed(case.workload),
        "concurrency": execution["concurrency"],
        "world_size": contract.WORLD_SIZE,
        **process_identity,
    }
    for row in rows:
        for field, expected in expected_fields.items():
            if row[field] != expected:
                raise ValueError(
                    "process row does not match current case execution: "
                    f"{field}"
                )
    ranks = [row["rank"] for row in rows]
    if ranks != list(range(contract.WORLD_SIZE)):
        raise ValueError(
            "process row does not match current case execution: rank"
        )
    for field in set(evidence_bindings) & set(
        contract.PROCESS_ROW_FIELDS
    ):
        expected = evidence_bindings[field]
        if any(row[field] != expected for row in rows):
            raise ValueError(
                f"process row evidence binding mismatch: {field}"
            )


def _rank_observer_rows(raw_rows):
    if not isinstance(raw_rows, (list, tuple)) or not raw_rows:
        raise ValueError("rank observer rows must be a non-empty list")
    process_fields = set(contract.PROCESS_ROW_FIELDS)
    canonical_fields = process_fields | {
        "snapshot_file",
        "tensor_storage_evidence",
    }
    field_sets = [
        set(row) if isinstance(row, dict) else set()
        for row in raw_rows
    ]
    if all(fields == process_fields for fields in field_sets):
        canonical = False
    elif all(fields == canonical_fields for fields in field_sets):
        canonical = True
    else:
        raise ValueError(
            "rank observer rows must use one closed evidence shape"
        )
    process_rows = [
        {
            field: row[field]
            for field in contract.PROCESS_ROW_FIELDS
        }
        for row in raw_rows
    ]
    return process_rows, canonical


def _validate_snapshot_file(snapshot_file, *, output_dir, case_id, rank):
    if (
        not isinstance(snapshot_file, dict)
        or set(snapshot_file) != set(SNAPSHOT_FILE_FIELDS)
    ):
        raise ValueError("snapshot file schema fields are invalid")
    expected_path = f"snapshots/{case_id}/rank-{rank}.snapshot"
    if snapshot_file["path"] != expected_path:
        raise ValueError("snapshot file path does not match rank")
    if snapshot_file["type"] != "regular_file":
        raise ValueError("snapshot file type is invalid")
    if (
        isinstance(snapshot_file["bytes"], bool)
        or not isinstance(snapshot_file["bytes"], int)
        or snapshot_file["bytes"] < 0
    ):
        raise ValueError("snapshot file bytes are invalid")
    if (
        not isinstance(snapshot_file["sha256"], str)
        or len(snapshot_file["sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in snapshot_file["sha256"]
        )
    ):
        raise ValueError("snapshot file sha256 is invalid")

    output_path = Path(output_dir).resolve()
    path = Path(output_dir) / expected_path
    try:
        path.resolve().relative_to(output_path)
    except ValueError as error:
        raise ValueError("snapshot file path is unsafe") from error
    if not path.is_file() or path.is_symlink():
        raise ValueError("snapshot file is missing")
    payload = path.read_bytes()
    if len(payload) != snapshot_file["bytes"]:
        raise ValueError("snapshot file byte length is invalid")
    if _sha256(payload) != snapshot_file["sha256"]:
        raise ValueError("snapshot file sha256 mismatch")


def _build_rank_evidence(raw_rows, process_rows, *, output_dir):
    rank_evidence = []
    for raw_row, process_row in zip(raw_rows, process_rows):
        snapshot_file = raw_row["snapshot_file"]
        evidence = raw_row["tensor_storage_evidence"]
        contract.validate_tensor_storage_evidence(evidence)
        for field in TENSOR_IDENTITY_FIELDS:
            if evidence[field] != process_row[field]:
                raise ValueError(
                    "tensor storage evidence identity mismatch: "
                    f"{field}"
                )
        _validate_snapshot_file(
            snapshot_file,
            output_dir=output_dir,
            case_id=process_row["case_id"],
            rank=process_row["rank"],
        )
        rank_evidence.append({
            "process_row": process_row,
            "snapshot_file": snapshot_file,
            "tensor_storage_evidence": evidence,
        })
    return rank_evidence


def _output_inventory(output_dir):
    output_path = Path(output_dir)
    if not output_path.exists():
        return set()
    return {
        path.relative_to(output_path)
        for path in output_path.rglob("*")
    }


def _cleanup_new_output_paths(output_dir, original_inventory):
    output_path = Path(output_dir)
    if not output_path.exists():
        return
    current_paths = sorted(
        output_path.rglob("*"),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in current_paths:
        if path.relative_to(output_path) in original_inventory:
            continue
        if path.is_symlink() or not path.is_dir():
            path.unlink(missing_ok=True)
        else:
            try:
                path.rmdir()
            except OSError:
                pass


def _case_row(
    *,
    case,
    observation,
    request_index,
    output_dir,
    evidence_bindings,
    execution,
):
    representation, version, codec = _profile_evidence(case.profile)
    token_path, token_sha256 = _write_bytes(
        output_dir,
        f"tokens/{case.case_id}-{request_index}.json",
        _token_bytes(observation["continuation_token_ids"]),
    )
    prompt_path, prompt_sha256 = _write_bytes(
        output_dir,
        f"tokens/{case.case_id}-{request_index}-prompt.json",
        _token_bytes(observation["prompt_token_ids"]),
    )
    logits_path, logits_sha256 = _write_bytes(
        output_dir,
        f"logits/{case.case_id}-{request_index}.bin",
        observation["final_logits_bytes"],
    )
    row = {
        "row_id": f"{case.case_id}__{observation['request_id']}",
        "case_id": case.case_id,
        "profile": case.profile,
        "representation": representation,
        "representation_version": version,
        "codec": codec,
        "workload": case.workload,
        "phase": case.phase,
        "repetition": case.repetition,
        "request_id": observation["request_id"],
        "sampling_temperature": contract.SAMPLING_TEMPERATURE,
        "sampling_max_tokens": execution["workload_payload"]["spec"][
            "generated_tokens"
        ],
        "sampling_ignore_eos": contract.SAMPLING_IGNORE_EOS,
        "sampling_seed": contract.workload_sampling_seed(case.workload),
        "concurrency": execution["concurrency"],
        "tp_world_size": contract.WORLD_SIZE,
        "gpu_indices": list(contract.REQUIRED_GPU_INDICES),
        "kv_capacity_bytes": 64 * 256,
        "hybrid_prefix_max_entries": contract.HYBRID_PREFIX_MAX_ENTRIES,
        "hybrid_prefix_max_bytes": contract.HYBRID_PREFIX_MAX_BYTES,
        "dirty_tree_policy": "reject_dirty",
        **evidence_bindings,
        "prompt_token_ids_path": prompt_path,
        "prompt_token_ids_sha256": prompt_sha256,
        "prompt_tokens": observation["prompt_tokens"],
        "reused_kv_tokens": observation["reused_kv_tokens"],
        "restored_hybrid_state": observation[
            "restored_hybrid_state"
        ],
        "executed_prefill_tokens": observation[
            "executed_prefill_tokens"
        ],
        "generated_tokens": observation["generated_tokens"],
        "ttft_ns": observation["ttft_ns"],
        "e2e_ns": observation["e2e_ns"],
        "decode_step_ns": observation["decode_step_ns"],
        "output_token_ids_path": token_path,
        "output_token_ids_sha256": token_sha256,
        "final_logits_path": logits_path,
        "final_logits_sha256": logits_sha256,
        "final_logits_shape": observation["final_logits_shape"],
        "final_logits_dtype": observation["final_logits_dtype"],
    }
    return {field: row[field] for field in contract.CASE_ROW_FIELDS}


def run_benchmark_case(
    *,
    case,
    output_dir,
    engine_factory,
    request_runner,
    rank_observer,
    evidence_bindings,
    process_identity,
):
    original_inventory = _output_inventory(output_dir)
    try:
        payload = contract.workload_payload(case.workload)
        execution = build_case_execution(
            case,
            workload_payload=payload,
        )
        configuration = {
            **execution,
            **process_identity,
        }
        engine = engine_factory(configuration)
        observations = request_runner(
            engine,
            case=case,
            configuration=configuration,
            workload_payload=payload,
        )
        raw_process_rows = rank_observer(
            engine,
            case=case,
            configuration=configuration,
            workload_payload=payload,
        )
        _validate_request_observations(observations, payload)
        process_rows, has_rank_evidence = _rank_observer_rows(
            raw_process_rows
        )
        for row in process_rows:
            contract.validate_process_row(row)
        _validate_process_rows_for_execution(
            process_rows,
            case=case,
            execution=execution,
            evidence_bindings=evidence_bindings,
            process_identity=process_identity,
        )
        process_aggregate_raw_evidence = aggregate_process_rows(
            process_rows
        )
        rank_evidence = None
        if has_rank_evidence:
            rank_evidence = _build_rank_evidence(
                raw_process_rows,
                process_rows,
                output_dir=output_dir,
            )
        case_rows = [
            _case_row(
                case=case,
                observation=observation,
                request_index=index,
                output_dir=output_dir,
                evidence_bindings=evidence_bindings,
                execution=execution,
            )
            for index, observation in enumerate(observations)
        ]
        for row in case_rows:
            contract.validate_case_row(row)
        result = {
            "schema_version": contract.SCHEMA_VERSION,
            "complete": True,
            "case_id": case.case_id,
            "case_rows": case_rows,
            "process_rows": process_rows,
            "process_aggregate_raw_evidence": (
                process_aggregate_raw_evidence
            ),
        }
        if rank_evidence is not None:
            result["rank_evidence"] = rank_evidence
        return result
    except Exception:
        _cleanup_new_output_paths(output_dir, original_inventory)
        raise
