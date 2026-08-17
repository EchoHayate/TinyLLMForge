from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name, filename):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract_for_worker_test",
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py",
)
worker = _load(
    "qwen35_tp4_hybrid_prefix_benchmark_v2_worker",
    "qwen35_tp4_hybrid_prefix_benchmark_v2_worker.py",
)


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
PROCESS_RAW_EVIDENCE_FIELDS = {
    *BYTE_FIELDS,
    *TRANSACTION_COUNTER_FIELDS,
    *RANK_LOCAL_OBSERVATION_FIELDS,
    "same_budget_entry_capacity",
    "pid",
    "hostname",
    "gpu_uuid",
    "cuda_visible_device",
    "master_addr",
    "master_port",
    "tinyvllm_dist_port",
    "nonce",
    "run_tag",
}


def _case(
    *,
    workload="w1_medium_reuse",
    phase="correctness",
    profile="recurrent_int8_per_row",
):
    return next(
        case
        for case in contract.build_case_matrix()
        if (
            case.profile == profile
            and case.workload == workload
            and case.phase == phase
            and case.repetition == 0
        )
    )


def _evidence_bindings():
    return {
        "source_tree_sha256": "a" * 64,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "tokenizer_manifest_sha256": "b" * 64,
        "workload_manifest_sha256": "c" * 64,
        "correctness_prerequisites_sha256": "d" * 64,
        "calibration_artifact_sha256": "e" * 64,
        "p1_authority_artifact_sha256": "f" * 64,
        "preflight_receipt_sha256": "1" * 64,
        "authorization_receipt_sha256": "2" * 64,
        "execution_receipt_sha256": "3" * 64,
        "gate1_audit_sha256": "4" * 64,
        "execution_plan_sha256": "5" * 64,
        "source_bundle_sha256": "6" * 64,
        "source_package_sha256": "7" * 64,
        "producer_source_sha256": "8" * 64,
        "producer_version_sha256": "9" * 64,
        "verifier_source_sha256": "a" * 64,
        "verifier_version_sha256": "b" * 64,
    }


def _request_observation(request_index=0):
    return {
        "request_id": f"request-{request_index}",
        "prompt_token_ids": [11, 12, 13],
        "continuation_token_ids": [101, 102],
        "prompt_tokens": 1088,
        "reused_kv_tokens": 1024,
        "restored_hybrid_state": True,
        "executed_prefill_tokens": 64,
        "generated_tokens": 64,
        "ttft_ns": 100,
        "e2e_ns": 200,
        "decode_step_ns": 10,
        "final_logits_bytes": b"\x00\x00\x80?",
        "final_logits_shape": [1],
        "final_logits_dtype": "float32",
    }


def _rank_row(rank, **overrides):
    row = {
        "case_id": _case().case_id,
        "profile": "recurrent_int8_per_row",
        "representation": "recurrent_int8_per_row",
        "representation_version": contract.P2_REPRESENTATION_VERSION,
        "codec": contract.P2_CODEC_ID,
        "workload": "w1_medium_reuse",
        "phase": "correctness",
        "repetition": 0,
        "rank": rank,
        "world_size": contract.WORLD_SIZE,
        "pid": 1000 + rank,
        "hostname": "tp4-host",
        "gpu_uuid": f"GPU-{rank}",
        "cuda_visible_device": str(rank),
        "master_addr": "127.0.0.1",
        "master_port": 29500,
        "tinyvllm_dist_port": 29501,
        "nonce": "nonce-1",
        "run_tag": "run-1",
        "artifact_path": "runs/run-1",
        "sampling_temperature": contract.SAMPLING_TEMPERATURE,
        "sampling_max_tokens": 64,
        "sampling_ignore_eos": contract.SAMPLING_IGNORE_EOS,
        "sampling_seed": contract.workload_sampling_seed(
            "w1_medium_reuse"
        ),
        "concurrency": 1,
        "gpu_indices": list(contract.REQUIRED_GPU_INDICES),
        "hybrid_prefix_max_entries": contract.HYBRID_PREFIX_MAX_ENTRIES,
        "hybrid_prefix_max_bytes": contract.HYBRID_PREFIX_MAX_BYTES,
        "dirty_tree_policy": "reject_dirty",
        **{
            field: value
            for field, value in _evidence_bindings().items()
            if field in contract.PROCESS_ROW_FIELDS
        },
        "initialization_ns": 10,
        "cuda_allocated_bytes": 100 + rank,
        "cuda_reserved_bytes": 200 + rank,
        "cuda_peak_allocated_bytes": 300 + rank,
        "cuda_peak_reserved_bytes": 400 + rank,
        "encode_workspace_peak_allocated_bytes": 10 + rank,
        "encode_workspace_peak_reserved_bytes": 20 + rank,
        "decode_workspace_peak_allocated_bytes": 30 + rank,
        "decode_workspace_peak_reserved_bytes": 40 + rank,
        "kv_capacity_bytes": 8192,
        "scheduler_visible_kv_blocks": 64,
        "hybrid_cache_current_entries": 2,
        "hybrid_cache_current_unique_physical_bytes": 1000 + rank,
        "hybrid_cache_current_logical_referenced_bytes": 3000 + rank,
        "hybrid_cache_current_metadata_bytes": 100 + rank,
        "hybrid_cache_deduplicated_bytes": 2000,
        "hybrid_cache_peak_entries": 3,
        "hybrid_cache_peak_unique_physical_bytes": 1500 + rank,
        "hybrid_cache_peak_logical_referenced_bytes": 3500 + rank,
        "hybrid_cache_peak_metadata_bytes": 150 + rank,
        "same_budget_entry_capacity": 32,
    }
    for field in TRANSACTION_COUNTER_FIELDS:
        row[field] = 7
    row.update(overrides)
    return row


def _tensor_reference(
    *,
    reference_id,
    layer_index,
    semantic_role,
    logical_dtype,
    logical_shape,
    resident_dtype,
    resident_shape,
    storage_id,
    storage_length_bytes,
):
    return {
        "reference_id": reference_id,
        "layer_index": layer_index,
        "semantic_role": semantic_role,
        "logical_dtype": logical_dtype,
        "logical_shape": logical_shape,
        "resident_dtype": resident_dtype,
        "resident_shape": resident_shape,
        "storage_id": storage_id,
        "storage_offset_bytes": 0,
        "storage_length_bytes": storage_length_bytes,
    }


def _canonical_tensor_storage_evidence(rank):
    references = []
    storages = []
    for layer_index in range(18):
        convolution_storage_id = f"resident-convolution-{layer_index}"
        recurrent_storage_id = f"resident-recurrent-{layer_index}"
        scale_storage_id = f"resident-scale-{layer_index}"
        storages.extend([
            {
                "storage_id": convolution_storage_id,
                "kind": "resident",
                "storage_nbytes": 2,
                "content_sha256": f"{layer_index + 1:064x}",
            },
            {
                "storage_id": recurrent_storage_id,
                "kind": "resident",
                "storage_nbytes": 1,
                "content_sha256": f"{layer_index + 101:064x}",
            },
            {
                "storage_id": scale_storage_id,
                "kind": "resident",
                "storage_nbytes": 4,
                "content_sha256": f"{layer_index + 201:064x}",
            },
        ])
        references.extend([
            _tensor_reference(
                reference_id=f"layer-{layer_index}-convolution",
                layer_index=layer_index,
                semantic_role="convolution",
                logical_dtype="bfloat16",
                logical_shape=[1, 1],
                resident_dtype="bfloat16",
                resident_shape=[1, 1],
                storage_id=convolution_storage_id,
                storage_length_bytes=2,
            ),
            _tensor_reference(
                reference_id=f"layer-{layer_index}-recurrent-values",
                layer_index=layer_index,
                semantic_role="recurrent_values",
                logical_dtype="float32",
                logical_shape=[1, 1, 1],
                resident_dtype="int8",
                resident_shape=[1, 1, 1],
                storage_id=recurrent_storage_id,
                storage_length_bytes=1,
            ),
            _tensor_reference(
                reference_id=f"layer-{layer_index}-recurrent-scales",
                layer_index=layer_index,
                semantic_role="recurrent_scales",
                logical_dtype=None,
                logical_shape=None,
                resident_dtype="float32",
                resident_shape=[1, 1],
                storage_id=scale_storage_id,
                storage_length_bytes=4,
            ),
        ])
    return {
        "schema_version": contract.TENSOR_STORAGE_EVIDENCE_SCHEMA_VERSION,
        "case_id": _case().case_id,
        "profile": contract.P2_PROFILE,
        "representation": contract.P2_REPRESENTATION,
        "representation_version": contract.P2_REPRESENTATION_VERSION,
        "codec": contract.P2_CODEC_ID,
        "rank": rank,
        "world_size": contract.WORLD_SIZE,
        "snapshots": [{
            "snapshot_id": f"snapshot-rank-{rank}",
            "tensor_references": references,
            "codec_metadata": {
                "codec": contract.P2_CODEC_ID,
                "layers": [
                    {
                        "codec": contract.P2_CODEC_ID,
                        "layer_index": layer_index,
                        "source_dtype": "torch.float32",
                        "source_shape": [1, 1, 1],
                    }
                    for layer_index in range(18)
                ],
                "representation": contract.P2_REPRESENTATION,
                "version": contract.P2_REPRESENTATION_VERSION,
            },
        }],
        "storages": storages,
        "observations": [{
            "ordinal": 0,
            "event": "final",
            "active_snapshot_ids": [f"snapshot-rank-{rank}"],
            "live_workspace_storage_ids": [],
            "encode_workspace_reserved_bytes": 0,
            "decode_workspace_reserved_bytes": 0,
            "cuda_allocated_bytes": 256 + rank,
            "cuda_reserved_bytes": 512 + rank,
        }],
    }


def _rank_observation(output_dir, rank):
    evidence = _canonical_tensor_storage_evidence(rank)
    accounting = contract.recompute_tensor_storage_accounting(evidence)
    payload = f"canonical-snapshot-rank-{rank}\n".encode("utf-8")
    relative_path = f"snapshots/{_case().case_id}/rank-{rank}.snapshot"
    snapshot_path = Path(output_dir) / relative_path
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(payload)
    return {
        **_rank_row(rank, **accounting),
        "snapshot_file": {
            "path": relative_path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "type": "regular_file",
        },
        "tensor_storage_evidence": evidence,
    }


def _assert_no_classification_field(value):
    if isinstance(value, dict):
        assert not any(
            "go" == key.lower() or "classification" in key.lower()
            for key in value
        )
        for item in value.values():
            _assert_no_classification_field(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_classification_field(item)


def test_worker_public_interface_writes_raw_tokens_logits_and_process_rows(
    tmp_path,
):
    result = worker.run_benchmark_case(
        case=_case(),
        output_dir=tmp_path,
        engine_factory=lambda configuration: object(),
        request_runner=lambda engine, **kwargs: [
            _request_observation(index)
            for index in range(4)
        ],
        rank_observer=lambda engine, **kwargs: [
            _rank_row(rank) for rank in range(contract.WORLD_SIZE)
        ],
        evidence_bindings=_evidence_bindings(),
        process_identity={
            "master_addr": "127.0.0.1",
            "master_port": 29500,
            "tinyvllm_dist_port": 29501,
            "nonce": "nonce-1",
            "run_tag": "run-1",
        },
    )

    assert set(result) == {
        "schema_version",
        "complete",
        "case_id",
        "case_rows",
        "process_rows",
        "process_aggregate_raw_evidence",
    }
    _assert_no_classification_field(result)
    assert len(result["case_rows"]) == 4
    assert len(result["process_rows"]) == contract.WORLD_SIZE
    assert result["process_aggregate_raw_evidence"] == (
        worker.aggregate_process_rows(result["process_rows"])
    )
    for row in result["case_rows"]:
        assert set(row) == set(contract.CASE_ROW_FIELDS)
        assert row["kv_capacity_bytes"] == 64 * 256
        assert all(
            row[field] == value
            for field, value in _evidence_bindings().items()
            if field in contract.CASE_ROW_FIELDS
        )
        assert row["output_token_ids_path"].startswith("tokens/")
        assert len(row["output_token_ids_sha256"]) == 64
        assert row["final_logits_path"].startswith("logits/")
        assert len(row["final_logits_sha256"]) == 64
        assert row["final_logits_shape"] == [1]
        assert row["final_logits_dtype"] == "float32"
        assert (tmp_path / row["output_token_ids_path"]).is_file()
        assert (tmp_path / row["final_logits_path"]).read_bytes() == (
            b"\x00\x00\x80?"
        )
    for row in result["process_rows"]:
        assert set(row) == set(contract.PROCESS_ROW_FIELDS)
        assert PROCESS_RAW_EVIDENCE_FIELDS <= set(row)
        assert row["pid"] > 0
        assert row["master_port"] == 29500
        assert row["tinyvllm_dist_port"] == 29501
        assert row["nonce"] == "nonce-1"
        assert all(
            len(row[field]) == 64
            for field in (
                "source_tree_sha256",
                "gate1_audit_sha256",
                "execution_plan_sha256",
                "source_bundle_sha256",
                "source_package_sha256",
            )
        )


def test_worker_preserves_canonical_rank_snapshot_and_tensor_evidence(
    tmp_path,
):
    raw_rank_observations = []

    def observe_ranks(engine, **kwargs):
        del engine, kwargs
        raw_rank_observations.extend(
            _rank_observation(tmp_path, rank)
            for rank in range(contract.WORLD_SIZE)
        )
        return raw_rank_observations

    result = worker.run_benchmark_case(
        case=_case(),
        output_dir=tmp_path,
        engine_factory=lambda configuration: object(),
        request_runner=lambda engine, **kwargs: [
            _request_observation(index)
            for index in range(4)
        ],
        rank_observer=observe_ranks,
        evidence_bindings=_evidence_bindings(),
        process_identity={
            "master_addr": "127.0.0.1",
            "master_port": 29500,
            "tinyvllm_dist_port": 29501,
            "nonce": "nonce-1",
            "run_tag": "run-1",
        },
    )

    assert [item["process_row"]["rank"] for item in result["rank_evidence"]] == (
        list(range(contract.WORLD_SIZE))
    )
    assert result["rank_evidence"] == [
        {
            "process_row": {
                field: observation[field]
                for field in contract.PROCESS_ROW_FIELDS
            },
            "snapshot_file": observation["snapshot_file"],
            "tensor_storage_evidence": observation[
                "tensor_storage_evidence"
            ],
        }
        for observation in raw_rank_observations
    ]
    for item in result["rank_evidence"]:
        evidence = item["tensor_storage_evidence"]
        snapshot_file = item["snapshot_file"]
        contract.validate_tensor_storage_evidence(evidence)
        assert evidence["case_id"] == result["case_id"]
        assert evidence["rank"] == item["process_row"]["rank"]
        assert snapshot_file["path"].startswith(
            f"snapshots/{result['case_id']}/rank-"
        )
        payload = (tmp_path / snapshot_file["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == snapshot_file["sha256"]
        assert len(payload) == snapshot_file["bytes"]


def test_worker_rejects_mismatched_tensor_evidence_and_cleans_snapshot(
    tmp_path,
):
    def observe_ranks(engine, **kwargs):
        del engine, kwargs
        raw_rank_observations = [
            _rank_observation(tmp_path, rank)
            for rank in range(contract.WORLD_SIZE)
        ]
        raw_rank_observations[2]["tensor_storage_evidence"]["rank"] = 3
        return raw_rank_observations

    try:
        worker.run_benchmark_case(
            case=_case(),
            output_dir=tmp_path,
            engine_factory=lambda configuration: object(),
            request_runner=lambda engine, **kwargs: [
                _request_observation(index)
                for index in range(4)
            ],
            rank_observer=observe_ranks,
            evidence_bindings=_evidence_bindings(),
            process_identity={
                "master_addr": "127.0.0.1",
                "master_port": 29500,
                "tinyvllm_dist_port": 29501,
                "nonce": "nonce-1",
                "run_tag": "run-1",
            },
        )
    except ValueError as error:
        assert "tensor" in str(error).lower()
        assert "rank" in str(error).lower()
    else:
        raise AssertionError("mismatched tensor evidence rank was accepted")
    assert not any(tmp_path.rglob("*.snapshot"))


def test_worker_rejects_noncanonical_request_id_coverage(tmp_path):
    canonical = [
        _request_observation(index)
        for index in range(4)
    ]
    variants = {
        "missing": canonical[:-1],
        "duplicate": [
            canonical[0],
            canonical[1],
            canonical[1],
            canonical[3],
        ],
        "wrong": [
            canonical[0],
            canonical[1],
            {
                **canonical[2],
                "request_id": "request-20",
            },
            canonical[3],
        ],
        "extra": [
            *canonical,
            _request_observation(4),
        ],
    }

    for name, observations in variants.items():
        output_dir = tmp_path / name
        try:
            worker.run_benchmark_case(
                case=_case(),
                output_dir=output_dir,
                engine_factory=lambda configuration: object(),
                request_runner=lambda engine, **kwargs: observations,
                rank_observer=lambda engine, **kwargs: [
                    _rank_row(rank)
                    for rank in range(contract.WORLD_SIZE)
                ],
                evidence_bindings=_evidence_bindings(),
                process_identity={
                    "master_addr": "127.0.0.1",
                    "master_port": 29500,
                    "tinyvllm_dist_port": 29501,
                    "nonce": "nonce-1",
                    "run_tag": "run-1",
                },
            )
        except ValueError as error:
            assert "request" in str(error).lower()
        else:
            raise AssertionError(
                f"{name} request ID coverage was accepted"
            )
        assert not output_dir.exists()


def test_worker_binds_process_rows_to_current_case_execution(tmp_path):
    mismatch_values = {
        "case_id": "another-case",
        "profile": contract.P1_REFERENCE_PROFILE,
        "representation": "exact_restore",
        "representation_version": contract.P1_REPRESENTATION_VERSION,
        "codec": None,
        "workload": "w2_long_reuse",
        "phase": "measured",
        "repetition": 1,
        "sampling_temperature": (
            contract.SAMPLING_TEMPERATURE + 0.5
        ),
        "sampling_max_tokens": 65,
        "sampling_ignore_eos": not contract.SAMPLING_IGNORE_EOS,
        "sampling_seed": (
            contract.workload_sampling_seed("w1_medium_reuse") + 1
        ),
        "concurrency": 2,
        "world_size": contract.WORLD_SIZE + 1,
        "master_addr": "127.0.0.2",
        "master_port": 29510,
        "tinyvllm_dist_port": 29511,
        "nonce": "nonce-2",
        "run_tag": "run-2",
    }

    for field, value in mismatch_values.items():
        rows = [
            _rank_row(rank)
            for rank in range(contract.WORLD_SIZE)
        ]
        rows[2][field] = value
        output_dir = tmp_path / field
        try:
            worker.run_benchmark_case(
                case=_case(),
                output_dir=output_dir,
                engine_factory=lambda configuration: object(),
                request_runner=lambda engine, **kwargs: [
                    _request_observation(index)
                    for index in range(4)
                ],
                rank_observer=lambda engine, **kwargs: rows,
                evidence_bindings=_evidence_bindings(),
                process_identity={
                    "master_addr": "127.0.0.1",
                    "master_port": 29500,
                    "tinyvllm_dist_port": 29501,
                    "nonce": "nonce-1",
                    "run_tag": "run-1",
                },
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"process row {field} mismatch was accepted"
            )
        assert not output_dir.exists()

    rows = [
        _rank_row(rank)
        for rank in range(contract.WORLD_SIZE)
    ]
    rows[3]["rank"] = contract.WORLD_SIZE
    try:
        worker.run_benchmark_case(
            case=_case(),
            output_dir=tmp_path / "rank",
            engine_factory=lambda configuration: object(),
            request_runner=lambda engine, **kwargs: [
                _request_observation(index)
                for index in range(4)
            ],
            rank_observer=lambda engine, **kwargs: rows,
            evidence_bindings=_evidence_bindings(),
            process_identity={
                "master_addr": "127.0.0.1",
                "master_port": 29500,
                "tinyvllm_dist_port": 29501,
                "nonce": "nonce-1",
                "run_tag": "run-1",
            },
        )
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range process rank was accepted")
    assert not (tmp_path / "rank").exists()


def test_worker_binds_process_source_identity_to_evidence_bindings(
    tmp_path,
):
    evidence_bindings = _evidence_bindings()
    overlap = sorted(
        set(evidence_bindings) & set(contract.PROCESS_ROW_FIELDS)
    )
    assert {
        "source_tree_sha256",
        "gate1_audit_sha256",
        "execution_plan_sha256",
        "source_bundle_sha256",
        "source_package_sha256",
        "producer_source_sha256",
        "producer_version_sha256",
        "verifier_source_sha256",
        "verifier_version_sha256",
    } <= set(overlap)

    for field in overlap:
        rows = [
            _rank_row(rank)
            for rank in range(contract.WORLD_SIZE)
        ]
        rows[1][field] = "0" * 64
        output_dir = tmp_path / field
        try:
            worker.run_benchmark_case(
                case=_case(),
                output_dir=output_dir,
                engine_factory=lambda configuration: object(),
                request_runner=lambda engine, **kwargs: [
                    _request_observation(index)
                    for index in range(4)
                ],
                rank_observer=lambda engine, **kwargs: rows,
                evidence_bindings=evidence_bindings,
                process_identity={
                    "master_addr": "127.0.0.1",
                    "master_port": 29500,
                    "tinyvllm_dist_port": 29501,
                    "nonce": "nonce-1",
                    "run_tag": "run-1",
                },
            )
        except ValueError as error:
            assert "evidence binding" in str(error).lower()
        else:
            raise AssertionError(
                f"process row {field} evidence mismatch was accepted"
            )
        assert not output_dir.exists()


def test_rank_aggregation_requires_all_ranks_and_sums_byte_metrics():
    rows = [_rank_row(rank) for rank in range(contract.WORLD_SIZE)]

    aggregate = worker.aggregate_process_rows(rows)

    assert aggregate["rank_inventory"] == [0, 1, 2, 3]
    assert {
        field: aggregate[field]
        for field in BYTE_FIELDS
    } == {
        field: sum(row[field] for row in rows)
        for field in BYTE_FIELDS
    }
    assert aggregate["same_budget_entry_capacity"] == 32

    try:
        worker.aggregate_process_rows(rows[:-1])
    except ValueError as error:
        assert "rank" in str(error).lower()
    else:
        raise AssertionError("missing TP rank was accepted")


def test_distributed_transaction_counters_require_rank_parity():
    rows = [_rank_row(rank) for rank in range(contract.WORLD_SIZE)]
    rows[3]["hybrid_cache_hits"] += 1

    try:
        worker.aggregate_process_rows(rows)
    except ValueError as error:
        assert "parity" in str(error).lower()
    else:
        raise AssertionError("divergent distributed counter was accepted")


def test_rank_local_allocator_and_workspace_observations_are_retained():
    rows = [_rank_row(rank) for rank in range(contract.WORLD_SIZE)]

    aggregate = worker.aggregate_process_rows(rows)

    assert aggregate["rank_observations"] == [
        {
            "rank": row["rank"],
            **{
                field: row[field]
                for field in RANK_LOCAL_OBSERVATION_FIELDS
            },
        }
        for row in rows
    ]


def test_correctness_runs_are_serial_and_w3_uses_frozen_fanout_only():
    correctness = worker.build_case_execution(
        _case(phase="correctness"),
        workload_payload=contract.workload_payload(
            "w1_medium_reuse"
        ),
    )
    w3 = worker.build_case_execution(
        _case(workload="w3_batched_fanout", phase="measured"),
        workload_payload=contract.workload_payload(
            "w3_batched_fanout"
        ),
    )

    assert correctness["concurrency"] == 1
    assert correctness["serial_correctness"] is True
    assert w3["workload"] == "w3_batched_fanout"
    assert w3["workload_payload"] == contract.workload_payload(
        "w3_batched_fanout"
    )
    assert w3["concurrency"] == 8
    assert len(w3["workload_payload"]["continuations"]) == 8


def test_case_execution_rejects_mutated_w3_batched_fanout_payload():
    case = _case(workload="w3_batched_fanout", phase="measured")
    payload = copy.deepcopy(contract.workload_payload(case.workload))
    payload["continuations"][0]["suffix_token_ids"][0] += 1
    assert len(payload["continuations"]) == 8

    try:
        worker.build_case_execution(case, workload_payload=payload)
    except ValueError as error:
        assert "workload" in str(error).lower()
    else:
        raise AssertionError("mutated W3 workload payload was accepted")


def test_worker_rejects_task6_invalid_case_row(tmp_path):
    observation = _request_observation()
    observation["restored_hybrid_state"] = 1

    try:
        worker.run_benchmark_case(
            case=_case(),
            output_dir=tmp_path,
            engine_factory=lambda configuration: object(),
            request_runner=lambda engine, **kwargs: [
                observation,
                _request_observation(1),
                _request_observation(2),
                _request_observation(3),
            ],
            rank_observer=lambda engine, **kwargs: [
                _rank_row(rank) for rank in range(contract.WORLD_SIZE)
            ],
            evidence_bindings=_evidence_bindings(),
            process_identity={
                "master_addr": "127.0.0.1",
                "master_port": 29500,
                "tinyvllm_dist_port": 29501,
                "nonce": "nonce-1",
                "run_tag": "run-1",
            },
        )
    except ValueError as error:
        assert "restored hybrid state" in str(error).lower()
    else:
        raise AssertionError("Task 6-invalid case row was emitted")
    assert tmp_path.is_dir()
    assert not any(tmp_path.iterdir())


def test_worker_rejects_task6_invalid_process_row(tmp_path):
    rows = [_rank_row(rank) for rank in range(contract.WORLD_SIZE)]
    rows[2]["master_port"] = 70000

    try:
        worker.run_benchmark_case(
            case=_case(),
            output_dir=tmp_path,
            engine_factory=lambda configuration: object(),
            request_runner=lambda engine, **kwargs: [
                _request_observation(index)
                for index in range(4)
            ],
            rank_observer=lambda engine, **kwargs: rows,
            evidence_bindings=_evidence_bindings(),
            process_identity={
                "master_addr": "127.0.0.1",
                "master_port": 29500,
                "tinyvllm_dist_port": 29501,
                "nonce": "nonce-1",
                "run_tag": "run-1",
            },
        )
    except ValueError as error:
        assert "master port" in str(error).lower()
    else:
        raise AssertionError("Task 6-invalid process row was emitted")


def test_rank_aggregation_rejects_task6_invalid_complete_process_row():
    rows = [_rank_row(rank) for rank in range(contract.WORLD_SIZE)]
    rows[1]["sampling_ignore_eos"] = 1

    try:
        worker.aggregate_process_rows(rows)
    except ValueError as error:
        assert "sampling ignore eos" in str(error).lower()
    else:
        raise AssertionError("Task 6-invalid process row was aggregated")


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        if "tmp_path" in test.__code__.co_varnames:
            import tempfile

            with tempfile.TemporaryDirectory() as temporary:
                test(Path(temporary))
        else:
            test()
    print(
        "qwen35 TP4 hybrid-prefix benchmark v2 worker tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
