from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "tools/qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py"
)
RUNTIME_REPRESENTATION_PATH = (
    ROOT / "tinyvllm/engine/qwen35_hybrid_prefix_representation.py"
)
RUNTIME_CONFIG_PATH = ROOT / "tinyvllm/config.py"
TP4_ROOT_LOGIT_CONTRACT_PATH = (
    ROOT / "tools/qwen35_tp4_real_root_logit_correctness_contract.py"
)
SCHEMA_V1_CONTRACT_PATH = (
    ROOT / "tools/qwen35_tp4_hybrid_prefix_benchmark_contract.py"
)
ROOT_RECEIPT_PATH = (
    ROOT / "tools/qwen35_tp4_root_logit_remote_execution_receipt.py"
)
NATIVE_ENGINE_PLAN_TEST_PATH = (
    ROOT / "tools/test_qwen35_tp4_engine_remote_execution_plan.py"
)
NATIVE_CACHED_PLAN_TEST_PATH = (
    ROOT
    / "tools/test_qwen35_tp4_cached_continuation_remote_execution_plan.py"
)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_module(
    CONTRACT_PATH,
    "qwen35_tp4_hybrid_prefix_benchmark_v2_contract",
)
native_root_receipt = _load_module(
    ROOT_RECEIPT_PATH,
    "qwen35_tp4_root_logit_remote_execution_receipt_for_v2_contract",
)
native_engine_plan_test = _load_module(
    NATIVE_ENGINE_PLAN_TEST_PATH,
    "test_qwen35_tp4_engine_remote_execution_plan_for_v2_contract",
)
native_cached_plan_test = _load_module(
    NATIVE_CACHED_PLAN_TEST_PATH,
    "test_qwen35_tp4_cached_continuation_remote_execution_plan_for_v2_contract",
)


def _literal_assignment(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = (node.target,)
        else:
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal assignment: {name}")


def _class_literal_assignment(path, class_name, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for statement in node.body:
            if (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == name
            ):
                return ast.literal_eval(statement.value)
    raise AssertionError(
        f"missing class literal assignment: {class_name}.{name}"
    )


EXPECTED_WORKLOAD_SPECS = {
    "w0_short_control": {
        "shared_prefix_tokens": 256,
        "suffix_tokens": 32,
        "continuations": 1,
        "generated_tokens": 32,
        "kind": "reuse",
    },
    "w1_medium_reuse": {
        "shared_prefix_tokens": 1024,
        "suffix_tokens": 64,
        "continuations": 4,
        "generated_tokens": 64,
        "kind": "reuse",
    },
    "w2_long_reuse": {
        "shared_prefix_tokens": 3840,
        "suffix_tokens": 64,
        "continuations": 4,
        "generated_tokens": 64,
        "kind": "reuse",
    },
    "w3_batched_fanout": {
        "shared_prefix_tokens": 2048,
        "suffix_tokens": 64,
        "continuations": 8,
        "generated_tokens": 32,
        "kind": "batched_reuse",
    },
    "w4_miss_invalidation": {
        "shared_prefix_tokens": 1024,
        "suffix_tokens": 64,
        "continuations": 3,
        "generated_tokens": 32,
        "kind": "miss_control",
    },
}
APPROVED_WORKLOAD_CORPUS_SHA256 = (
    "d8bbfb7f114e5f8a8e109ff402289ec4"
    "14b43fa5f8b83c245ef8d47e854a2acc"
)
APPROVED_WORKLOAD_PAYLOAD_SHA256 = {
    "w0_short_control": (
        "189b098fd1f11bdac07ba5e5f0b27654"
        "a96696ec436b5b86e63d01aa179e4736"
    ),
    "w1_medium_reuse": (
        "969def502895116d0db873277d0e34ad5"
        "8a1579e30df152860f0af09e681d298"
    ),
    "w2_long_reuse": (
        "d238eda649b85e4c2b6c57f135810242"
        "e19071b15f9d9107999506c82459fe3d"
    ),
    "w3_batched_fanout": (
        "2fdfcaabcf9dc98ef89946d26a60d054"
        "edf599dcfb45e6fec1e5abd36241993d"
    ),
    "w4_miss_invalidation": (
        "dcdab4100e9d9fdd34bc64ec8d47eb3"
        "b4c469925813de102643d07fab8fb80dd"
    ),
}

EXPECTED_CASE_ROW_FIELDS = (
    "row_id",
    "case_id",
    "profile",
    "representation",
    "representation_version",
    "codec",
    "workload",
    "phase",
    "repetition",
    "request_id",
    "sampling_temperature",
    "sampling_max_tokens",
    "sampling_ignore_eos",
    "sampling_seed",
    "concurrency",
    "tp_world_size",
    "gpu_indices",
    "kv_capacity_bytes",
    "hybrid_prefix_max_entries",
    "hybrid_prefix_max_bytes",
    "dirty_tree_policy",
    "source_tree_sha256",
    "model_manifest_sha256",
    "tokenizer_manifest_sha256",
    "workload_manifest_sha256",
    "correctness_prerequisites_sha256",
    "calibration_artifact_sha256",
    "p1_authority_artifact_sha256",
    "preflight_receipt_sha256",
    "authorization_receipt_sha256",
    "gate1_audit_sha256",
    "execution_plan_sha256",
    "source_bundle_sha256",
    "source_package_sha256",
    "producer_source_sha256",
    "producer_version_sha256",
    "verifier_source_sha256",
    "verifier_version_sha256",
    "prompt_token_ids_path",
    "prompt_token_ids_sha256",
    "prompt_tokens",
    "reused_kv_tokens",
    "restored_hybrid_state",
    "executed_prefill_tokens",
    "generated_tokens",
    "ttft_ns",
    "e2e_ns",
    "decode_step_ns",
    "output_token_ids_path",
    "output_token_ids_sha256",
    "final_logits_path",
    "final_logits_sha256",
    "final_logits_shape",
    "final_logits_dtype",
)

EXPECTED_PROCESS_ROW_FIELDS = (
    "case_id",
    "profile",
    "representation",
    "representation_version",
    "codec",
    "workload",
    "phase",
    "repetition",
    "rank",
    "world_size",
    "pid",
    "hostname",
    "gpu_uuid",
    "cuda_visible_device",
    "master_addr",
    "master_port",
    "tinyvllm_dist_port",
    "nonce",
    "run_tag",
    "artifact_path",
    "sampling_temperature",
    "sampling_max_tokens",
    "sampling_ignore_eos",
    "sampling_seed",
    "concurrency",
    "gpu_indices",
    "hybrid_prefix_max_entries",
    "hybrid_prefix_max_bytes",
    "dirty_tree_policy",
    "source_tree_sha256",
    "gate1_audit_sha256",
    "execution_plan_sha256",
    "source_bundle_sha256",
    "source_package_sha256",
    "producer_source_sha256",
    "producer_version_sha256",
    "verifier_source_sha256",
    "verifier_version_sha256",
    "initialization_ns",
    "cuda_allocated_bytes",
    "cuda_reserved_bytes",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
    "encode_workspace_peak_allocated_bytes",
    "encode_workspace_peak_reserved_bytes",
    "decode_workspace_peak_allocated_bytes",
    "decode_workspace_peak_reserved_bytes",
    "kv_capacity_bytes",
    "scheduler_visible_kv_blocks",
    "hybrid_cache_current_entries",
    "hybrid_cache_current_unique_physical_bytes",
    "hybrid_cache_current_logical_referenced_bytes",
    "hybrid_cache_current_metadata_bytes",
    "hybrid_cache_deduplicated_bytes",
    "hybrid_cache_peak_entries",
    "hybrid_cache_peak_unique_physical_bytes",
    "hybrid_cache_peak_logical_referenced_bytes",
    "hybrid_cache_peak_metadata_bytes",
    "same_budget_entry_capacity",
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


def _case_row():
    return {
        "row_id": "row-1",
        "case_id": (
            "w1_medium_reuse__correctness__r0__"
            "recurrent_int8_per_row"
        ),
        "profile": "recurrent_int8_per_row",
        "representation": "recurrent_int8_per_row",
        "representation_version": contract.P2_REPRESENTATION_VERSION,
        "codec": "qwen35_recurrent_symmetric_int8_per_row_v1",
        "workload": "w1_medium_reuse",
        "phase": "correctness",
        "repetition": 0,
        "request_id": "request-1",
        "sampling_temperature": 0.0,
        "sampling_max_tokens": 64,
        "sampling_ignore_eos": True,
        "sampling_seed": 2026072901,
        "concurrency": 1,
        "tp_world_size": 4,
        "gpu_indices": [2, 4, 5, 6],
        "kv_capacity_bytes": 1,
        "hybrid_prefix_max_entries": 16,
        "hybrid_prefix_max_bytes": 2 * 1024**3,
        "dirty_tree_policy": "reject_dirty",
        "source_tree_sha256": "a" * 64,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "tokenizer_manifest_sha256": "b" * 64,
        "workload_manifest_sha256": "c" * 64,
        "correctness_prerequisites_sha256": "d" * 64,
        "calibration_artifact_sha256": "e" * 64,
        "p1_authority_artifact_sha256": "f" * 64,
        "preflight_receipt_sha256": "1" * 64,
        "authorization_receipt_sha256": "2" * 64,
        "gate1_audit_sha256": "7" * 64,
        "execution_plan_sha256": "8" * 64,
        "source_bundle_sha256": "9" * 64,
        "source_package_sha256": "a" * 64,
        "producer_source_sha256": "b" * 64,
        "producer_version_sha256": "c" * 64,
        "verifier_source_sha256": "d" * 64,
        "verifier_version_sha256": "e" * 64,
        "prompt_token_ids_path": "tokens/prompt-row-1.json",
        "prompt_token_ids_sha256": "4" * 64,
        "prompt_tokens": 1088,
        "reused_kv_tokens": 1024,
        "restored_hybrid_state": True,
        "executed_prefill_tokens": 64,
        "generated_tokens": 64,
        "ttft_ns": 1,
        "e2e_ns": 2,
        "decode_step_ns": 1,
        "output_token_ids_path": "tokens/output-row-1.json",
        "output_token_ids_sha256": "5" * 64,
        "final_logits_path": "logits/final-row-1.bin",
        "final_logits_sha256": "6" * 64,
        "final_logits_shape": [32000],
        "final_logits_dtype": "float32",
    }


def _canonical_case_rows():
    rows = []
    for case in contract.build_case_matrix():
        spec = contract.WORKLOAD_SPECS[case.workload]
        prompt_tokens = (
            spec["shared_prefix_tokens"] + spec["suffix_tokens"]
        )
        restore_hit = (
            case.profile != "recompute"
            and case.workload != "w4_miss_invalidation"
        )
        for request_index in range(spec["continuations"]):
            row = _case_row()
            row.update({
                "row_id": f"{case.case_id}__request-{request_index}",
                "case_id": case.case_id,
                "profile": case.profile,
                "workload": case.workload,
                "phase": case.phase,
                "repetition": case.repetition,
                "request_id": f"request-{request_index}",
                "sampling_max_tokens": spec["generated_tokens"],
                "sampling_seed": contract.workload_sampling_seed(
                    case.workload
                ),
                "concurrency": case.concurrency,
                "prompt_tokens": prompt_tokens,
                "reused_kv_tokens": (
                    spec["shared_prefix_tokens"] if restore_hit else 0
                ),
                "restored_hybrid_state": restore_hit,
                "executed_prefill_tokens": (
                    spec["suffix_tokens"]
                    if restore_hit
                    else prompt_tokens
                ),
                "generated_tokens": spec["generated_tokens"],
            })
            if case.profile == "recompute":
                row.update({
                    "representation": None,
                    "representation_version": None,
                    "codec": None,
                })
            elif case.profile == "exact_restore":
                row.update({
                    "representation": "exact_restore",
                    "representation_version": (
                        contract.P1_REPRESENTATION_VERSION
                    ),
                    "codec": None,
                })
            rows.append(row)
    return rows


def _process_row():
    row = {
        "case_id": (
            "w1_medium_reuse__correctness__r0__"
            "recurrent_int8_per_row"
        ),
        "profile": "recurrent_int8_per_row",
        "representation": "recurrent_int8_per_row",
        "representation_version": contract.P2_REPRESENTATION_VERSION,
        "codec": "qwen35_recurrent_symmetric_int8_per_row_v1",
        "workload": "w1_medium_reuse",
        "phase": "correctness",
        "repetition": 0,
        "rank": 0,
        "world_size": 4,
        "pid": 1234,
        "hostname": "tp4-host",
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
        "cuda_visible_device": "0",
        "master_addr": "127.0.0.1",
        "master_port": 29500,
        "tinyvllm_dist_port": 29501,
        "nonce": "nonce-1",
        "run_tag": "run-1",
        "artifact_path": "runs/run-1",
        "sampling_temperature": 0.0,
        "sampling_max_tokens": 64,
        "sampling_ignore_eos": True,
        "sampling_seed": 2026072901,
        "concurrency": 1,
        "gpu_indices": [2, 4, 5, 6],
        "hybrid_prefix_max_entries": 16,
        "hybrid_prefix_max_bytes": 2 * 1024**3,
        "dirty_tree_policy": "reject_dirty",
        "source_tree_sha256": "a" * 64,
        "gate1_audit_sha256": "7" * 64,
        "execution_plan_sha256": "8" * 64,
        "source_bundle_sha256": "9" * 64,
        "source_package_sha256": "a" * 64,
        "producer_source_sha256": "b" * 64,
        "producer_version_sha256": "c" * 64,
        "verifier_source_sha256": "d" * 64,
        "verifier_version_sha256": "e" * 64,
        "initialization_ns": 1,
        "cuda_allocated_bytes": 1,
        "cuda_reserved_bytes": 1,
        "cuda_peak_allocated_bytes": 1,
        "cuda_peak_reserved_bytes": 1,
        "encode_workspace_peak_allocated_bytes": 1,
        "encode_workspace_peak_reserved_bytes": 1,
        "decode_workspace_peak_allocated_bytes": 1,
        "decode_workspace_peak_reserved_bytes": 1,
        "kv_capacity_bytes": 1,
        "scheduler_visible_kv_blocks": 1,
        "hybrid_cache_current_entries": 1,
        "hybrid_cache_current_unique_physical_bytes": 1,
        "hybrid_cache_current_logical_referenced_bytes": 1,
        "hybrid_cache_current_metadata_bytes": 1,
        "hybrid_cache_deduplicated_bytes": 0,
        "hybrid_cache_peak_entries": 1,
        "hybrid_cache_peak_unique_physical_bytes": 4,
        "hybrid_cache_peak_logical_referenced_bytes": 4,
        "hybrid_cache_peak_metadata_bytes": 1,
        "same_budget_entry_capacity": 1,
    }
    for field in contract.PROCESS_ROW_FIELDS:
        if field not in row:
            row[field] = 0
    return row


def _canonical_process_rows():
    rows = []
    for case in contract.build_case_matrix():
        spec = contract.WORKLOAD_SPECS[case.workload]
        for rank in range(4):
            row = _process_row()
            row.update({
                "case_id": case.case_id,
                "profile": case.profile,
                "workload": case.workload,
                "phase": case.phase,
                "repetition": case.repetition,
                "rank": rank,
                "cuda_visible_device": str(rank),
                "sampling_max_tokens": spec["generated_tokens"],
                "sampling_seed": contract.workload_sampling_seed(
                    case.workload
                ),
                "concurrency": case.concurrency,
            })
            if case.profile == "recompute":
                row.update({
                    "representation": None,
                    "representation_version": None,
                    "codec": None,
                })
            elif case.profile == "exact_restore":
                row.update({
                    "representation": "exact_restore",
                    "representation_version": (
                        contract.P1_REPRESENTATION_VERSION
                    ),
                    "codec": None,
                })
            row.update({
                "hybrid_cache_current_logical_referenced_bytes": (
                    0 if case.profile == "recompute" else 4
                ),
                "hybrid_cache_current_unique_physical_bytes": (
                    0
                    if case.profile == "recompute"
                    else 2
                    if case.profile == contract.P2_PROFILE
                    else 4
                ),
                "hybrid_cache_current_metadata_bytes": (
                    1 if case.profile == contract.P2_PROFILE else 0
                ),
                "encode_workspace_peak_allocated_bytes": 0,
                "decode_workspace_peak_allocated_bytes": 0,
            })
            rows.append(row)
    return rows


def _calibration_binding():
    return {
        "schema_version": contract.CALIBRATION_SCHEMA_VERSION,
        "codec": contract.P2_CODEC_ID,
        "representation": contract.P2_REPRESENTATION,
        "representation_version": contract.P2_REPRESENTATION_VERSION,
        "source_tree_sha256": "a" * 64,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "workload_manifest_sha256": "b" * 64,
        "artifact_path": "calibration/calibration.json",
        "artifact_sha256": "c" * 64,
        "classification": "PASS",
    }


def _p1_authority_binding():
    return {
        "schema_version": contract.P1_AUTHORITY_SCHEMA_VERSION,
        "profile": contract.P1_REFERENCE_PROFILE,
        "representation": contract.P1_REFERENCE_PROFILE,
        "source_tree_sha256": "a" * 64,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "workload_manifest_sha256": "b" * 64,
        "artifact_path": "p1/artifact.json",
        "artifact_sha256": "c" * 64,
        "independent_verification_path": (
            "p1/independent_verification.json"
        ),
        "independent_verification_sha256": "d" * 64,
        "classification": "GO",
    }


def _snapshot_inventory():
    return {
        "schema_version": contract.SNAPSHOT_INVENTORY_SCHEMA_VERSION,
        "case_id": _case_row()["case_id"],
        "profile": contract.P2_PROFILE,
        "representation": contract.P2_REPRESENTATION,
        "representation_version": contract.P2_REPRESENTATION_VERSION,
        "codec": contract.P2_CODEC_ID,
        "rank": 0,
        "world_size": 4,
        "snapshot_path": "snapshots/rank-0.snapshot",
        "snapshot_sha256": "a" * 64,
        "tensor_inventory_path": "snapshots/rank-0-tensors.json",
        "tensor_inventory_sha256": "b" * 64,
        "full_fidelity_logical_bytes": 4,
        "encoded_physical_bytes": 2,
        "codec_metadata_bytes": 1,
        "temporary_encode_workspace_bytes": 1,
        "temporary_decode_workspace_bytes": 1,
    }


def _tensor_reference(
    reference_id,
    layer_index,
    semantic_role,
    logical_dtype,
    logical_shape,
    resident_dtype,
    resident_shape,
    storage_id,
    storage_offset_bytes,
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
        "storage_offset_bytes": storage_offset_bytes,
        "storage_length_bytes": storage_length_bytes,
    }


def _canonical_tensor_storage_evidence(
    profile=contract.P1_REFERENCE_PROFILE,
    *,
    case_id="w1_medium_reuse__correctness__r0__exact_restore",
    rank=0,
    alias_recurrent=False,
    observations=None,
):
    if profile == contract.P1_REFERENCE_PROFILE:
        representation = contract.P1_REFERENCE_PROFILE
        representation_version = contract.P1_REPRESENTATION_VERSION
        codec = None
    elif profile == contract.P2_PROFILE:
        representation = contract.P2_REPRESENTATION
        representation_version = contract.P2_REPRESENTATION_VERSION
        codec = contract.P2_CODEC_ID
        case_id = case_id.replace(
            contract.P1_REFERENCE_PROFILE,
            contract.P2_PROFILE,
        )
    else:
        raise AssertionError(f"unsupported fixture profile: {profile}")

    references = []
    storages = []
    for layer_index in range(18):
        convolution_storage_id = f"resident-convolution-{layer_index}"
        storages.append({
            "storage_id": convolution_storage_id,
            "kind": "resident",
            "storage_nbytes": 2,
            "content_sha256": f"{layer_index + 1:064x}",
        })
        references.append(_tensor_reference(
            f"layer-{layer_index}-convolution",
            layer_index,
            "convolution",
            "bfloat16",
            [1, 1],
            "bfloat16",
            [1, 1],
            convolution_storage_id,
            0,
            2,
        ))
        recurrent_storage_id = (
            "resident-recurrent-shared"
            if alias_recurrent
            else f"resident-recurrent-{layer_index}"
        )
        if not alias_recurrent or layer_index == 0:
            storages.append({
                "storage_id": recurrent_storage_id,
                "kind": "resident",
                "storage_nbytes": 4 if profile == contract.P1_REFERENCE_PROFILE else 1,
                "content_sha256": (
                    "f" * 64
                    if alias_recurrent
                    else f"{layer_index + 101:064x}"
                ),
            })
        references.append(_tensor_reference(
            f"layer-{layer_index}-recurrent-values",
            layer_index,
            "recurrent_values",
            "float32",
            [1, 1, 1],
            (
                "float32"
                if profile == contract.P1_REFERENCE_PROFILE
                else "int8"
            ),
            [1, 1, 1],
            recurrent_storage_id,
            0,
            4 if profile == contract.P1_REFERENCE_PROFILE else 1,
        ))
        if profile == contract.P2_PROFILE:
            scale_storage_id = f"resident-scale-{layer_index}"
            storages.append({
                "storage_id": scale_storage_id,
                "kind": "resident",
                "storage_nbytes": 4,
                "content_sha256": f"{layer_index + 201:064x}",
            })
            references.append(_tensor_reference(
                f"layer-{layer_index}-recurrent-scales",
                layer_index,
                "recurrent_scales",
                None,
                None,
                "float32",
                [1, 1],
                scale_storage_id,
                0,
                4,
            ))

    snapshot = {
        "snapshot_id": "snapshot-0",
        "tensor_references": references,
        "codec_metadata": (
            None
            if profile == contract.P1_REFERENCE_PROFILE
            else {
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
            }
        ),
    }
    if observations is None:
        observations = [{
            "ordinal": 0,
            "event": "final",
            "active_snapshot_ids": ["snapshot-0"],
            "live_workspace_storage_ids": [],
            "encode_workspace_reserved_bytes": 0,
            "decode_workspace_reserved_bytes": 0,
            "cuda_allocated_bytes": 256,
            "cuda_reserved_bytes": 512,
        }]
    return {
        "schema_version": (
            "qwen35.tp4-hybrid-prefix-tensor-storage-evidence.v1"
        ),
        "case_id": case_id,
        "profile": profile,
        "representation": representation,
        "representation_version": representation_version,
        "codec": codec,
        "rank": rank,
        "world_size": contract.WORLD_SIZE,
        "snapshots": [snapshot],
        "storages": storages,
        "observations": observations,
    }


def _append_tensor_storage_snapshot(evidence, snapshot_id):
    snapshot = copy.deepcopy(evidence["snapshots"][0])
    snapshot["snapshot_id"] = snapshot_id
    evidence["snapshots"].append(snapshot)


def _append_workspace_storage(evidence, storage_id, kind, storage_nbytes):
    evidence["storages"].append({
        "storage_id": storage_id,
        "kind": kind,
        "storage_nbytes": storage_nbytes,
        "content_sha256": None,
    })


def _prepend_tensor_storage_observation(evidence, event):
    evidence["observations"][0]["ordinal"] = 1
    evidence["observations"].insert(0, {
        "ordinal": 0,
        "event": event,
        "active_snapshot_ids": [],
        "live_workspace_storage_ids": [],
        "encode_workspace_reserved_bytes": 0,
        "decode_workspace_reserved_bytes": 0,
        "cuda_allocated_bytes": 0,
        "cuda_reserved_bytes": 0,
    })


def _manifest_entry():
    return {
        "path": "snapshots/rank-0.snapshot",
        "sha256": "a" * 64,
        "bytes": 1,
        "producer": "task7-worker",
        "trust_domain": "producer",
    }


def _receipt_binding():
    return {
        "schema_version": contract.RECEIPT_BINDING_SCHEMA_VERSION,
        "run_tag": "run-1",
        "nonce": "nonce-1",
        "artifact_path": "runs/run-1",
        "gate1_audit_sha256": "a" * 64,
        "preflight_sha256": "b" * 64,
        "execution_plan_sha256": "c" * 64,
        "consumed_authorization_sha256": "d" * 64,
        "source_bundle_sha256": "f" * 64,
        "source_package_sha256": "1" * 64,
        "resource_guards_sha256": "2" * 64,
    }


def _source_manifest():
    return {
        "schema_version": contract.SOURCE_MANIFEST_SCHEMA_VERSION,
        "source_tree_sha256": "a" * 64,
        "dirty_tree_policy": "reject_dirty",
        "dirty_tree": False,
        "gate1_audit_sha256": "7" * 64,
        "execution_plan_sha256": "8" * 64,
        "source_bundle_sha256": "9" * 64,
        "source_package_sha256": "a" * 64,
        "producer_source_sha256": "b" * 64,
        "producer_version_sha256": "c" * 64,
        "verifier_source_sha256": "d" * 64,
        "verifier_version_sha256": "e" * 64,
    }


def _matched_configuration():
    return {
        "schema_version": contract.MATCHED_CONFIGURATION_SCHEMA_VERSION,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "tokenizer_manifest_sha256": "b" * 64,
        "workload_manifest_sha256": "c" * 64,
        "sampling_temperature": 0.0,
        "sampling_max_tokens": 64,
        "sampling_ignore_eos": True,
        "sampling_seed": 2026072901,
        "concurrency": 1,
        "tp_world_size": 4,
        "gpu_indices": [2, 4, 5, 6],
        "kv_capacity_bytes": 1,
        "hybrid_prefix_max_entries": 16,
        "hybrid_prefix_max_bytes": 2 * 1024**3,
    }


def _closed_evidence_document(kind):
    fields = contract.EVIDENCE_DOCUMENT_FIELDS[kind]
    defaults = {
        "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[kind],
        "classification": "PASS",
        "run_tag": "run-1",
        "nonce": "nonce-1",
        "source_tree_sha256": "a" * 64,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "workload_manifest_sha256": "b" * 64,
        "calibration_artifact_sha256": "c" * 64,
        "p1_authority_artifact_sha256": "d" * 64,
        "gate1_audit_sha256": "e" * 64,
        "execution_plan_sha256": "f" * 64,
        "consumed_authorization_sha256": "1" * 64,
        "execution_receipt_sha256": "2" * 64,
        "source_bundle_sha256": "3" * 64,
        "source_package_sha256": "4" * 64,
        "resource_guards_sha256": "5" * 64,
        "producer_source_sha256": "6" * 64,
        "producer_version_sha256": "7" * 64,
        "verifier_source_sha256": "8" * 64,
        "verifier_version_sha256": "9" * 64,
        "artifact_manifest_sha256": "a" * 64,
        "path": f"receipts/{kind}.json",
        "sha256": "b" * 64,
        "inventory_path": f"source/{kind}-inventory.json",
        "inventory_sha256": "c" * 64,
        "command_manifest_sha256": "d" * 64,
        "resource_guard_before_sha256": "e" * 64,
        "resource_guard_after_sha256": "f" * 64,
        "local_verifier_sha256": "1" * 64,
        "remote_verifier_sha256": "2" * 64,
        "required_gpu_indices": [2, 4, 5, 6],
        "minimum_free_bytes_per_gpu": contract.MIN_GPU_FREE_BYTES,
        "world_size": 4,
        "rank": 0,
        "phase": "before",
        "role": "local",
        "local_verifier_role": "local",
        "remote_verifier_role": "remote",
        "checks": 1,
        "consumed": True,
        "dirty_tree_policy": "reject_dirty",
    }
    document = {field: defaults[field] for field in fields}
    if kind == "preflight":
        document["classification"] = "READY"
    elif kind == "verifier_output":
        document["classification"] = "GO"
    elif kind == "independent_verification":
        document["classification"] = "GO"
    return document


def _execution_evidence_bundle():
    case_port_pairs = [
        {
            "case_id": case.case_id,
            "tinyvllm_dist_port": 22000 + index * 2,
            "master_port": 22001 + index * 2,
        }
        for index, case in enumerate(contract.build_case_matrix())
    ]
    command_order = [
        "reserve_remote",
        "upload",
        "stage",
        "resource_guard",
        "workers",
        "assembly",
        "remote_verify",
        "final_resource_guard",
        "package_download",
        "safe_extract",
        "local_verify",
    ]
    hashes = {
        "source_tree_sha256": "a" * 64,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "workload_manifest_sha256": "b" * 64,
        "correctness_prerequisites_sha256": "c" * 64,
        "calibration_artifact_sha256": "d" * 64,
        "p1_authority_artifact_sha256": "e" * 64,
        "gate1_audit_sha256": "f" * 64,
        "source_bundle_sha256": "1" * 64,
        "source_package_sha256": "2" * 64,
        "command_manifest_sha256": "3" * 64,
        "execution_plan_sha256": "4" * 64,
        "consumed_authorization_sha256": "5" * 64,
        "package_inventory_sha256": "6" * 64,
        "final_inventory_sha256": "7" * 64,
        "resource_guard_before_sha256": "8" * 64,
        "resource_guard_after_sha256": "9" * 64,
        "artifact_manifest_sha256": "a" * 64,
        "local_verifier_sha256": "b" * 64,
        "remote_verifier_sha256": "c" * 64,
        "producer_source_sha256": "d" * 64,
        "producer_version_sha256": "e" * 64,
        "verifier_source_sha256": "f" * 64,
        "verifier_version_sha256": "1" * 64,
    }
    producer_identity = {
        "run_tag": "run-1",
        "nonce": "nonce-1",
        **{
            name: value
            for name, value in hashes.items()
            if name
            not in {
                "execution_plan_sha256",
                "consumed_authorization_sha256",
                "artifact_manifest_sha256",
                "local_verifier_sha256",
                "remote_verifier_sha256",
                "package_inventory_sha256",
                "final_inventory_sha256",
                "resource_guard_before_sha256",
                "resource_guard_after_sha256",
            }
        },
    }
    gpu_rows = [
        {
            "rank": rank,
            "gpu_index": gpu_index,
            "cuda_visible_device": str(rank),
        }
        for rank, gpu_index in enumerate(contract.REQUIRED_GPU_INDICES)
    ]
    resource_rows = [
        {
            "gpu_index": gpu_index,
            "gpu_uuid": f"GPU-{gpu_index}",
            "free_bytes": contract.MIN_GPU_FREE_BYTES,
            "compute_processes": [],
        }
        for gpu_index in contract.REQUIRED_GPU_INDICES
    ]
    artifact_paths = {
        "remote_run": "runs/run-1",
        "remote_artifact": "runs/run-1/artifact",
        "package": "receipts/benchmark-artifact.tar",
        "local_extract": "downloads/run-1/artifact",
    }
    source_inventory = [
        {
            "path": "tools/worker.py",
            "sha256": "2" * 64,
            "bytes": 10,
            "type": "file",
        }
    ]
    package_inventory = [
        {
            "path": path,
            "sha256": f"{index + 32:064x}",
            "bytes": index + 1,
            "type": "file",
        }
        for index, path in enumerate(
            sorted(
                set(contract.ARTIFACT_MANIFEST_HASH_DOMAIN)
                - {"execution_receipt.json"}
            )
        )
    ]
    final_inventory = sorted(
        [
            {
                "path": "artifact_manifest.json",
                "sha256": hashes["artifact_manifest_sha256"],
                "bytes": 10,
                "type": "file",
            },
            *copy.deepcopy(package_inventory),
        ],
        key=lambda row: row["path"],
    )
    command_results = [
        {
            "name": name,
            "command_sha256": f"{index + 1:064x}",
            "outcome": "attempted",
            "returncode": 0,
            "stdout": "ok",
            "stderr": "",
            "stdout_truncated": False,
            "stderr_truncated": False,
        }
        for index, name in enumerate(command_order)
    ]
    hashes["command_manifest_sha256"] = contract.canonical_json_sha256(
        [
            {
                "name": row["name"],
                "command_sha256": row["command_sha256"],
            }
            for row in command_results
        ]
    )
    hashes["package_inventory_sha256"] = (
        contract.canonical_json_sha256(package_inventory)
    )
    hashes["final_inventory_sha256"] = (
        contract.canonical_json_sha256(final_inventory)
    )
    producer_identity["command_manifest_sha256"] = hashes[
        "command_manifest_sha256"
    ]
    common_plan = {
        "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[
            "execution_plan"
        ],
        **producer_identity,
        "authority_root_sha256": "a" * 64,
        "physical_artifact_root_sha256": "b" * 64,
        "required_gpu_indices": list(contract.REQUIRED_GPU_INDICES),
        "world_size": contract.WORLD_SIZE,
        "gpu_assignments": gpu_rows,
        "case_port_pairs": case_port_pairs,
        "artifact_paths": artifact_paths,
        "command_order": command_order,
    }
    canonical_commands = contract.canonical_execution_commands(common_plan)
    for row in command_results:
        row["command_sha256"] = contract.execution_command_sha256(
            canonical_commands[row["name"]]
        )
    hashes["command_manifest_sha256"] = contract.canonical_json_sha256(
        [
            {
                "name": row["name"],
                "command_sha256": row["command_sha256"],
            }
            for row in command_results
        ]
    )
    producer_identity["command_manifest_sha256"] = hashes[
        "command_manifest_sha256"
    ]
    common_plan["command_manifest_sha256"] = hashes[
        "command_manifest_sha256"
    ]
    bundle = {
        "lifecycle_state": "execution_success",
        "environment": {
            "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[
                "environment"
            ],
            **producer_identity,
            "dirty_tree_policy": "reject_dirty",
        },
        "gpu_assignments": {
            "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[
                "gpu_assignments"
            ],
            "run_tag": "run-1",
            "required_gpu_indices": list(contract.REQUIRED_GPU_INDICES),
            "world_size": contract.WORLD_SIZE,
            "assignments": gpu_rows,
        },
        "commands": {
            "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[
                "commands"
            ],
            "run_tag": "run-1",
            "nonce": "nonce-1",
            "execution_plan_sha256": hashes["execution_plan_sha256"],
            "command_manifest_sha256": hashes[
                "command_manifest_sha256"
            ],
            "command_order": command_order,
            "commands": [
                {
                    "name": row["name"],
                    "command_sha256": row["command_sha256"],
                }
                for row in command_results
            ],
        },
        "preflight": {
            "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[
                "preflight"
            ],
            "classification": "READY",
            **producer_identity,
            "required_gpu_indices": list(contract.REQUIRED_GPU_INDICES),
            "world_size": contract.WORLD_SIZE,
            "minimum_free_bytes_per_gpu": contract.MIN_GPU_FREE_BYTES,
            "gpu_query_rows": resource_rows,
            "blocking_reasons": [],
            "worker_authorized": True,
            "remote_path_created": False,
            "source_staged": False,
            "worker_launched": False,
        },
        "execution_plan": dict(common_plan),
        "consumed_authorization": {
            "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[
                "consumed_authorization"
            ],
            **producer_identity,
            "execution_plan_sha256": hashes["execution_plan_sha256"],
            "required_gpu_indices": list(contract.REQUIRED_GPU_INDICES),
            "world_size": contract.WORLD_SIZE,
            "gpu_assignments": gpu_rows,
            "case_port_pairs": case_port_pairs,
            "artifact_paths": artifact_paths,
            "authorization_id": "authorization-1",
            "active_path": "receipts/authorization.json",
            "consumed_path": "receipts/authorization.consumed.json",
            "consumed": True,
            "consumed_once": True,
        },
        "source_bundle": {
            "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[
                "source_bundle"
            ],
            **producer_identity,
            "dirty_tree_policy": "reject_dirty",
            "path": "source/source-bundle.tar",
            "sha256": hashes["source_bundle_sha256"],
            "inventory_path": "source/source-bundle-inventory.json",
            "inventory_sha256": contract.canonical_json_sha256(
                source_inventory
            ),
            "inventory": source_inventory,
        },
        "source_package": {
            "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[
                "source_package"
            ],
            **producer_identity,
            "path": "source/source-package.tar",
            "sha256": hashes["source_package_sha256"],
            "inventory_path": "source/source-package-inventory.json",
            "inventory_sha256": contract.canonical_json_sha256(
                source_inventory
            ),
            "inventory": source_inventory,
        },
        "resource_guard_before": {
            "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[
                "resource_guard"
            ],
            "run_tag": "run-1",
            "phase": "before",
            "required_gpu_indices": list(contract.REQUIRED_GPU_INDICES),
            "minimum_free_bytes_per_gpu": contract.MIN_GPU_FREE_BYTES,
            "sha256": hashes["resource_guard_before_sha256"],
            "gpu_query_rows": resource_rows,
            "side_effects_observed": False,
        },
        "resource_guard_after": {
            "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[
                "resource_guard"
            ],
            "run_tag": "run-1",
            "phase": "after",
            "required_gpu_indices": list(contract.REQUIRED_GPU_INDICES),
            "minimum_free_bytes_per_gpu": contract.MIN_GPU_FREE_BYTES,
            "sha256": hashes["resource_guard_after_sha256"],
            "gpu_query_rows": resource_rows,
            "side_effects_observed": False,
        },
        "execution_receipt": {
            "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[
                "execution_receipt"
            ],
            "classification": "PASS",
            **producer_identity,
            "execution_plan_sha256": hashes["execution_plan_sha256"],
            "consumed_authorization_sha256": hashes[
                "consumed_authorization_sha256"
            ],
            "authorization_id": "authorization-1",
            "command_order": command_order,
            "command_results": command_results,
            "artifact_paths": artifact_paths,
            "source_inventory": source_inventory,
            "package_inventory": package_inventory,
            "final_inventory": final_inventory,
            "package_inventory_sha256": hashes[
                "package_inventory_sha256"
            ],
            "final_inventory_sha256": hashes[
                "final_inventory_sha256"
            ],
            "resource_guard_before_sha256": hashes[
                "resource_guard_before_sha256"
            ],
            "resource_guard_after_sha256": hashes[
                "resource_guard_after_sha256"
            ],
            "remote_path_created": True,
            "source_staged": True,
            "worker_launched": True,
            "cleanup_complete": True,
        },
        "local_verifier_output": {
            "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[
                "verifier_output"
            ],
            "classification": "GO",
            "role": "local",
            "artifact_manifest_sha256": hashes[
                "artifact_manifest_sha256"
            ],
            "verifier_source_sha256": hashes[
                "verifier_source_sha256"
            ],
            "verifier_version_sha256": hashes[
                "verifier_version_sha256"
            ],
            "checks": 1,
        },
        "remote_verifier_output": {
            "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[
                "verifier_output"
            ],
            "classification": "GO",
            "role": "remote",
            "artifact_manifest_sha256": hashes[
                "artifact_manifest_sha256"
            ],
            "verifier_source_sha256": hashes[
                "verifier_source_sha256"
            ],
            "verifier_version_sha256": hashes[
                "verifier_version_sha256"
            ],
            "checks": 1,
        },
        "independent_verification": {
            "schema_version": contract.EVIDENCE_SCHEMA_VERSIONS[
                "independent_verification"
            ],
            "classification": "GO",
            "artifact_manifest_sha256": hashes[
                "artifact_manifest_sha256"
            ],
            "local_verifier_sha256": hashes[
                "local_verifier_sha256"
            ],
            "remote_verifier_sha256": hashes[
                "remote_verifier_sha256"
            ],
            "local_verifier_role": "local",
            "remote_verifier_role": "remote",
            "checks": 1,
        },
    }
    plan_sha256 = contract.canonical_json_sha256(
        bundle["execution_plan"]
    )
    bundle["commands"]["execution_plan_sha256"] = plan_sha256
    bundle["consumed_authorization"][
        "execution_plan_sha256"
    ] = plan_sha256
    bundle["execution_receipt"]["execution_plan_sha256"] = plan_sha256
    bundle["execution_receipt"][
        "consumed_authorization_sha256"
    ] = contract.canonical_json_sha256(
        bundle["consumed_authorization"]
    )
    bundle["independent_verification"][
        "local_verifier_sha256"
    ] = contract.canonical_json_sha256(
        bundle["local_verifier_output"]
    )
    bundle["independent_verification"][
        "remote_verifier_sha256"
    ] = contract.canonical_json_sha256(
        bundle["remote_verifier_output"]
    )
    for name in ("resource_guard_before", "resource_guard_after"):
        bundle[name]["sha256"] = contract.resource_guard_sha256(
            bundle[name]
        )
    bundle["execution_receipt"][
        "resource_guard_before_sha256"
    ] = bundle["resource_guard_before"]["sha256"]
    bundle["execution_receipt"][
        "resource_guard_after_sha256"
    ] = bundle["resource_guard_after"]["sha256"]
    return bundle


def _bind_execution_roots(bundle, *, authority_root, artifact_root):
    plan = bundle["execution_plan"]
    Path(authority_root).mkdir(parents=True, exist_ok=True)
    Path(artifact_root).mkdir(parents=True, exist_ok=True)
    plan["authority_root_sha256"] = contract.physical_directory_sha256(
        authority_root
    )
    plan["physical_artifact_root_sha256"] = (
        contract.physical_directory_sha256(artifact_root)
    )
    plan_sha256 = contract.canonical_json_sha256(plan)
    bundle["commands"]["execution_plan_sha256"] = plan_sha256
    bundle["consumed_authorization"][
        "execution_plan_sha256"
    ] = plan_sha256
    bundle["execution_receipt"]["execution_plan_sha256"] = plan_sha256
    bundle["execution_receipt"][
        "consumed_authorization_sha256"
    ] = contract.canonical_json_sha256(
        bundle["consumed_authorization"]
    )
    return bundle


def test_slice_b_canonical_command_reconstruction_is_complete_and_deterministic():
    bundle = _execution_evidence_bundle()
    first = contract.canonical_execution_commands(bundle["execution_plan"])
    second = contract.canonical_execution_commands(
        copy.deepcopy(bundle["execution_plan"])
    )

    assert list(first) == list(contract.EXECUTION_COMMAND_ORDER)
    assert first == second
    assert first["upload"]["argv"] != first["upload"]["argv"][::-1]
    assert set(first["workers"]) >= {
        "argv",
        "worker_commands",
        "timeout_seconds",
        "stdout_limit_bytes",
        "stderr_limit_bytes",
    }
    assert first["package_download"]["local_output"] == (
        bundle["execution_plan"]["artifact_paths"]["package"]
    )
    assert {
        name: contract.execution_command_sha256(command)
        for name, command in first.items()
    } == {
        name: contract.execution_command_sha256(command)
        for name, command in second.items()
    }


def test_slice_b2_canonical_execution_commands_pass_semantic_validation():
    bundle = _execution_evidence_bundle()
    commands = contract.canonical_execution_commands(
        bundle["execution_plan"]
    )

    contract.validate_execution_command_semantics(
        commands,
        expected_order=contract.EXECUTION_COMMAND_ORDER,
    )


@pytest.mark.parametrize("forbidden_basename", ["kill", "pkill", "killall"])
@pytest.mark.parametrize("wrapped", [False, True], ids=["direct", "wrapped"])
def test_slice_b2_forbidden_process_commands_are_rejected(
    forbidden_basename,
    wrapped,
):
    bundle = _execution_evidence_bundle()
    commands = contract.canonical_execution_commands(
        bundle["execution_plan"]
    )
    commands["reserve_remote"]["argv"] = (
        ["bash", "-lc", f"sh -c '{forbidden_basename} 1234'"]
        if wrapped
        else [forbidden_basename, "1234"]
    )

    with pytest.raises(ValueError, match="forbidden|command"):
        contract.validate_execution_command_semantics(
            commands,
            expected_order=contract.EXECUTION_COMMAND_ORDER,
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["bash", "-lc", "kill>/tmp/x"],
        ["bash", "-lc", "(kill 1234)"],
        ["bash", "-lc", "env kill 1234"],
        ["bash", "-lc", "command /usr/bin/pkill 1234"],
        ["env", "killall", "target"],
    ],
)
def test_slice_b2_shell_and_delegate_wrappers_reject_forbidden_commands(argv):
    bundle = _execution_evidence_bundle()
    commands = contract.canonical_execution_commands(
        bundle["execution_plan"]
    )
    commands["reserve_remote"]["argv"] = argv

    with pytest.raises(ValueError, match="forbidden|command|shell"):
        contract.validate_execution_command_semantics(
            commands,
            expected_order=contract.EXECUTION_COMMAND_ORDER,
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["bash", "-lc", "true\nkill 1234"],
        ["bash", "-lc", "true |& kill 1234"],
        ["bash", "-lc", "SAFE=1 /usr/bin/pkill 1234"],
        ["bash", "-lc", "if true; then kill 1234; fi"],
        ["exec", "/usr/bin/pkill", "1234"],
        ["nohup", "/usr/bin/pkill", "1234"],
        ["timeout", "1", "/usr/bin/pkill", "1234"],
    ],
)
def test_slice_b2_remaining_shell_and_delegate_bypasses_are_rejected(argv):
    bundle = _execution_evidence_bundle()
    commands = contract.canonical_execution_commands(
        bundle["execution_plan"]
    )
    commands["reserve_remote"]["argv"] = argv

    with pytest.raises(ValueError, match="forbidden|command|shell"):
        contract.validate_execution_command_semantics(
            commands,
            expected_order=contract.EXECUTION_COMMAND_ORDER,
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["bash", "--norc", "-c", "kill 999999"],
        ["bash", "--rcfile", "/dev/null", "-c", "/usr/bin/pkill 999999"],
        ["env", "SAFE=1", "bash", "--norc", "-c", "killall target"],
        [
            "timeout",
            "1",
            "bash",
            "--rcfile",
            "/dev/null",
            "-c",
            "kill 999999",
        ],
        ["bash", "-c", "builtin kill 999999"],
        ["bash", "-c", "eval 'kill 999999'"],
        ["bash", "-c", "printf '999999\\n' | xargs kill"],
        ["bash", "-c", "printf 'kill 999999\\n' | sh"],
    ],
)
def test_slice_b2_shell_option_and_evaluator_bypasses_are_rejected(argv):
    bundle = _execution_evidence_bundle()
    commands = contract.canonical_execution_commands(
        bundle["execution_plan"]
    )
    commands["reserve_remote"]["argv"] = argv

    with pytest.raises(ValueError, match="forbidden|command|shell"):
        contract.validate_execution_command_semantics(
            commands,
            expected_order=contract.EXECUTION_COMMAND_ORDER,
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["xargs", "kill"],
        ["find", ".", "-exec", "/usr/bin/pkill", "{}", ";"],
        ["sudo", "killall", "target"],
        ["nice", "-n", "5", "/bin/kill", "1234"],
    ],
)
def test_slice_b2_forbidden_basename_anywhere_in_argv_is_rejected(argv):
    bundle = _execution_evidence_bundle()
    commands = contract.canonical_execution_commands(
        bundle["execution_plan"]
    )
    commands["reserve_remote"]["argv"] = argv

    with pytest.raises(ValueError, match="forbidden|command"):
        contract.validate_execution_command_semantics(
            commands,
            expected_order=contract.EXECUTION_COMMAND_ORDER,
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["nice", "/bin/sh", "-c", "kill 1234"],
        ["nice", "-n", "5", "/bin/bash", "-c", "/usr/bin/pkill 1234"],
        ["setsid", "/bin/sh", "-c", "killall target"],
        ["sudo", "/bin/sh", "-c", "kill 1234"],
    ],
)
def test_slice_b2_shell_basename_anywhere_in_argv_is_rejected(argv):
    bundle = _execution_evidence_bundle()
    commands = contract.canonical_execution_commands(
        bundle["execution_plan"]
    )
    commands["reserve_remote"]["argv"] = argv

    with pytest.raises(ValueError, match="shell|command|forbidden"):
        contract.validate_execution_command_semantics(
            commands,
            expected_order=contract.EXECUTION_COMMAND_ORDER,
        )


def _refresh_execution_command_trust_bindings(bundle, commands):
    command_order = list(commands)
    command_manifest = [
        {
            "name": name,
            "command_sha256": contract.execution_command_sha256(
                commands[name]
            ),
        }
        for name in command_order
    ]
    command_manifest_sha256 = contract.canonical_json_sha256(
        command_manifest
    )
    result_by_name = {
        row["name"]: row
        for row in bundle["execution_receipt"]["command_results"]
    }
    command_results = []
    for manifest_row in command_manifest:
        result = result_by_name[manifest_row["name"]]
        result["command_sha256"] = manifest_row["command_sha256"]
        command_results.append(result)

    bundle["execution_plan"]["command_order"] = command_order
    bundle["execution_plan"][
        "command_manifest_sha256"
    ] = command_manifest_sha256
    bundle["commands"].update({
        "command_order": command_order,
        "commands": command_manifest,
        "command_manifest_sha256": command_manifest_sha256,
    })
    bundle["execution_receipt"]["command_order"] = command_order
    bundle["execution_receipt"]["command_results"] = command_results
    for document in (
        bundle["environment"],
        bundle["preflight"],
        bundle["consumed_authorization"],
        bundle["source_bundle"],
        bundle["source_package"],
        bundle["execution_receipt"],
    ):
        document["command_manifest_sha256"] = command_manifest_sha256

    execution_plan_sha256 = contract.canonical_json_sha256(
        bundle["execution_plan"]
    )
    bundle["commands"]["execution_plan_sha256"] = execution_plan_sha256
    bundle["consumed_authorization"][
        "execution_plan_sha256"
    ] = execution_plan_sha256
    bundle["execution_receipt"][
        "execution_plan_sha256"
    ] = execution_plan_sha256
    consumed_authorization_sha256 = contract.canonical_json_sha256(
        bundle["consumed_authorization"]
    )
    bundle["execution_receipt"][
        "consumed_authorization_sha256"
    ] = consumed_authorization_sha256
    execution_receipt_sha256 = contract.canonical_json_sha256(
        bundle["execution_receipt"]
    )
    return {
        "command_manifest_sha256": command_manifest_sha256,
        "execution_plan_sha256": execution_plan_sha256,
        "consumed_authorization_sha256": (
            consumed_authorization_sha256
        ),
        "execution_receipt_sha256": execution_receipt_sha256,
    }


def _mutate_slice_b2_execution_commands(case, bundle, commands):
    worker = commands["workers"]["worker_commands"][0]
    if case == "argv_action":
        commands["reserve_remote"]["argv"][0] = "reserve_remote_attacker"
    elif case == "argv_path":
        commands["safe_extract"]["argv"][1] = "receipts/attacker.tar"
    elif case == "worker_env":
        worker["env"]["PYTHONPATH"] = "runs/attacker/source"
    elif case == "worker_cwd":
        worker["cwd"] = "runs/attacker/source"
    elif case == "timeout":
        commands["workers"]["timeout_seconds"] += 1
    elif case == "stdout_bound":
        commands["assembly"]["stdout_limit_bytes"] += 1
    elif case == "stderr_bound":
        commands["assembly"]["stderr_limit_bytes"] += 1
    elif case == "ssh_target":
        commands["upload"]["ssh_target"] = "attacker@example.invalid"
    elif case == "ssh_options":
        commands["upload"]["ssh_options"][-1] = "ConnectTimeout=21"
    elif case == "krb5_execution_env":
        commands["upload"]["execution_env"]["KRB5CCNAME"] = (
            "FILE:/tmp/attacker-krb5cc"
        )
    elif case == "remote_python":
        commands["upload"]["remote_python"] = "/tmp/attacker-python"
    elif case == "gpu_assignment":
        worker["gpu_assignments"][0]["gpu_index"] = 7
    elif case == "visible_gpu_list":
        worker["env"]["CUDA_VISIBLE_DEVICES"] = "2,4,5,7"
    elif case == "per_case_ports":
        worker["tinyvllm_dist_port"] += 1000
        worker["master_port"] += 1000
        worker["env"]["TINYVLLM_DIST_PORT"] = str(
            worker["tinyvllm_dist_port"]
        )
        worker["env"]["MASTER_PORT"] = str(worker["master_port"])
    elif case == "worker_provenance_identity":
        worker["provenance"]["producer_source_sha256"] = "9" * 64
    elif case == "command_order":
        mutated_order = list(commands)
        mutated_order[0], mutated_order[1] = (
            mutated_order[1],
            mutated_order[0],
        )
        commands.clear()
        commands.update({
            name: contract.canonical_execution_commands(
                bundle["execution_plan"]
            )[name]
            for name in mutated_order
        })
    else:
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    (
        "argv_action",
        "argv_path",
        "worker_env",
        "worker_cwd",
        "timeout",
        "stdout_bound",
        "stderr_bound",
        "ssh_target",
        "ssh_options",
        "krb5_execution_env",
        "remote_python",
        "gpu_assignment",
        "visible_gpu_list",
        "per_case_ports",
        "worker_provenance_identity",
        "command_order",
    ),
)
def test_slice_b2_refreshed_semantic_mutation_rejects_noncanonical_commands(
    case,
):
    bundle = _execution_evidence_bundle()
    commands = contract.canonical_execution_commands(
        bundle["execution_plan"]
    )
    _mutate_slice_b2_execution_commands(case, bundle, commands)
    refreshed = _refresh_execution_command_trust_bindings(
        bundle,
        commands,
    )

    contract.validate_execution_command_semantics(
        commands,
        expected_order=bundle["execution_plan"]["command_order"],
    )
    assert bundle["commands"]["command_manifest_sha256"] == (
        refreshed["command_manifest_sha256"]
    )
    assert bundle["commands"]["execution_plan_sha256"] == (
        refreshed["execution_plan_sha256"]
    )
    assert bundle["execution_receipt"][
        "consumed_authorization_sha256"
    ] == refreshed["consumed_authorization_sha256"]
    assert contract.canonical_json_sha256(
        bundle["execution_receipt"]
    ) == refreshed["execution_receipt_sha256"]

    with pytest.raises(ValueError, match="canonical|command|semantic"):
        contract.validate_execution_command_semantics(
            commands,
            expected_order=bundle["execution_plan"]["command_order"],
            execution_plan=bundle["execution_plan"],
        )


def test_slice_b2_root_logit_stage_order_passes_semantic_validation():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _, payload = _complete_prerequisite_fixture(root)
        plan, _, _ = _semantic_bundle(root, payload, "tp4_root_logit")

        contract.validate_execution_command_semantics(
            plan["stage_inputs"],
            expected_order=plan["stage_order"],
        )


def test_slice_b_canonical_commands_freeze_execution_timeouts_and_remote_runtime():
    bundle = _execution_evidence_bundle()
    commands = contract.canonical_execution_commands(
        bundle["execution_plan"]
    )

    assert [
        commands[name]["timeout_seconds"]
        for name in contract.EXECUTION_COMMAND_ORDER
    ] == [60, 300, 600, 120, 3600, 600, 600, 120, 1800, 1800, 600]

    remote_names = contract.EXECUTION_COMMAND_ORDER[:-2]
    expected_ssh_options = [
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "ConnectTimeout=20",
    ]
    expected_execution_env = {
        "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian",
    }
    expected_remote_python = (
        "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
    )
    for name in remote_names:
        assert commands[name]["ssh_target"] == "sitian@10.232.195.203"
        assert commands[name]["ssh_options"] == expected_ssh_options
        assert commands[name]["execution_env"] == expected_execution_env
        assert commands[name]["remote_python"] == expected_remote_python

    remote_run = bundle["execution_plan"]["artifact_paths"]["remote_run"]
    expected_worker_env = {
        "CUDA_VISIBLE_DEVICES": "2,4,5,6",
        "PYTHONPATH": f"{remote_run}/source",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for worker, port_pair in zip(
        commands["workers"]["worker_commands"],
        bundle["execution_plan"]["case_port_pairs"],
    ):
        assert worker["cwd"] == f"{remote_run}/source"
        assert worker["env"] == {
            **expected_worker_env,
            "TINYVLLM_DIST_PORT": str(
                port_pair["tinyvllm_dist_port"]
            ),
            "MASTER_PORT": str(port_pair["master_port"]),
        }

    assert commands["final_resource_guard"] == commands["resource_guard"]
    for name in ("safe_extract", "local_verify"):
        assert "ssh_target" not in commands[name]
        assert "ssh_options" not in commands[name]
        assert "execution_env" not in commands[name]
        assert "remote_python" not in commands[name]


def test_slice_b_rejects_arbitrary_refreshed_command_hash_manifest():
    bundle = _execution_evidence_bundle()
    for index, row in enumerate(bundle["commands"]["commands"]):
        row["command_sha256"] = f"{index + 100:064x}"
        bundle["execution_receipt"]["command_results"][index][
            "command_sha256"
        ] = row["command_sha256"]
    command_manifest_sha256 = contract.canonical_json_sha256(
        bundle["commands"]["commands"]
    )
    for document in (
        bundle["environment"],
        bundle["preflight"],
        bundle["execution_plan"],
        bundle["consumed_authorization"],
        bundle["source_bundle"],
        bundle["source_package"],
        bundle["execution_receipt"],
    ):
        document["command_manifest_sha256"] = command_manifest_sha256
    bundle["commands"][
        "command_manifest_sha256"
    ] = command_manifest_sha256
    plan_sha256 = contract.canonical_json_sha256(bundle["execution_plan"])
    bundle["commands"]["execution_plan_sha256"] = plan_sha256
    bundle["consumed_authorization"][
        "execution_plan_sha256"
    ] = plan_sha256
    bundle["execution_receipt"]["execution_plan_sha256"] = plan_sha256
    bundle["execution_receipt"][
        "consumed_authorization_sha256"
    ] = contract.canonical_json_sha256(bundle["consumed_authorization"])

    with pytest.raises(ValueError, match="command|canonical|semantic"):
        contract.validate_execution_evidence_bundle(bundle)


def _blocked_execution_evidence_bundle():
    success = _execution_evidence_bundle()
    blocked = {
        "lifecycle_state": "preflight_blocked",
        "environment": success["environment"],
        "gpu_assignments": success["gpu_assignments"],
        "preflight": success["preflight"],
    }
    blocked["preflight"]["classification"] = "BLOCKED_RESOURCES"
    blocked["preflight"]["gpu_query_rows"][0]["free_bytes"] = (
        contract.MIN_GPU_FREE_BYTES - 1
    )
    blocked["preflight"]["blocking_reasons"] = [
        "GPU 2 free bytes below minimum"
    ]
    blocked["preflight"]["worker_authorized"] = False
    return blocked


def _failed_execution_evidence_bundle():
    failed = _execution_evidence_bundle()
    failed["lifecycle_state"] = "execution_failed"
    failed.pop("local_verifier_output")
    failed.pop("remote_verifier_output")
    failed.pop("independent_verification")
    command_results = failed["execution_receipt"]["command_results"]
    failure_index = contract.EXECUTION_COMMAND_ORDER.index("workers")
    for index, row in enumerate(command_results):
        if index < failure_index:
            continue
        if index == failure_index:
            row["returncode"] = 17
            row["stderr"] = "worker launch failed"
            continue
        row["outcome"] = "skipped"
        row["returncode"] = None
        row["stdout"] = ""
        row["stderr"] = ""
    failed["execution_receipt"]["classification"] = "INVALID_ARTIFACT"
    failed["execution_receipt"]["worker_launched"] = False
    failed["execution_receipt"]["cleanup_complete"] = True
    failed["execution_receipt"]["resource_guard_after_sha256"] = None
    failed["execution_receipt"]["package_inventory"] = []
    failed["execution_receipt"]["final_inventory"] = []
    failed["execution_receipt"][
        "package_inventory_sha256"
    ] = contract.canonical_json_sha256([])
    failed["execution_receipt"][
        "final_inventory_sha256"
    ] = contract.canonical_json_sha256([])
    failed.pop("resource_guard_after")
    return failed


def _execution_failure_at(failed_command):
    bundle = _failed_execution_evidence_bundle()
    success_receipt = _execution_evidence_bundle()["execution_receipt"]
    bundle["resource_guard_after"] = _execution_evidence_bundle()[
        "resource_guard_after"
    ]
    command_results = bundle["execution_receipt"]["command_results"]
    failure_index = contract.EXECUTION_COMMAND_ORDER.index(failed_command)
    for index, row in enumerate(command_results):
        if index < failure_index:
            row["outcome"] = "attempted"
            row["returncode"] = 0
            row["stdout"] = "ok"
            row["stderr"] = ""
        elif index == failure_index:
            row["outcome"] = "attempted"
            row["returncode"] = 17
            row["stdout"] = ""
            row["stderr"] = f"{failed_command} failed"
        else:
            row["outcome"] = "skipped"
            row["returncode"] = None
            row["stdout"] = ""
            row["stderr"] = ""
    completed = set(contract.EXECUTION_COMMAND_ORDER[:failure_index])
    bundle["execution_receipt"]["remote_path_created"] = (
        "reserve_remote" in completed
    )
    bundle["execution_receipt"]["source_staged"] = "stage" in completed
    bundle["execution_receipt"]["worker_launched"] = (
        "workers" in completed
    )
    bundle["execution_receipt"]["cleanup_complete"] = any(
        bundle["execution_receipt"][field]
        for field in (
            "remote_path_created",
            "source_staged",
            "worker_launched",
        )
    )
    bundle["execution_receipt"]["package_inventory"] = (
        []
        if failure_index
        <= contract.EXECUTION_COMMAND_ORDER.index("assembly")
        else json.loads(
            json.dumps(success_receipt["final_inventory"])
        )
    )
    bundle["execution_receipt"]["final_inventory"] = (
        []
        if failure_index
        <= contract.EXECUTION_COMMAND_ORDER.index("safe_extract")
        else json.loads(
            json.dumps(success_receipt["final_inventory"])
        )
    )
    bundle["execution_receipt"][
        "package_inventory_sha256"
    ] = contract.canonical_json_sha256(
        bundle["execution_receipt"]["package_inventory"]
    )
    bundle["execution_receipt"][
        "final_inventory_sha256"
    ] = contract.canonical_json_sha256(
        bundle["execution_receipt"]["final_inventory"]
    )
    if failure_index <= contract.EXECUTION_COMMAND_ORDER.index(
        "resource_guard"
    ):
        bundle["execution_receipt"]["resource_guard_before_sha256"] = None
        bundle["execution_receipt"]["resource_guard_after_sha256"] = None
    elif failed_command == "workers":
        bundle["execution_receipt"]["resource_guard_after_sha256"] = None
    return bundle


def _nested_evidence_bundle():
    manifests = {
        kind: {
            "schema_version": contract.NESTED_MANIFEST_SCHEMA_VERSIONS[
                kind
            ],
            "kind": kind,
            "files": [],
            "rows": [],
        }
        for kind in contract.NESTED_MANIFEST_KINDS
    }

    def add_file(kind, path, seed):
        row = {
            "path": path,
            "sha256": f"{seed:064x}",
            "bytes": seed + 1,
            "type": "regular_file",
        }
        manifests[kind]["files"].append(row)
        return row

    seed = 1
    for name in contract.PREREQUISITE_NAMES:
        for role in ("artifact", "independent_verification", "provenance"):
            file_row = add_file(
                "prerequisites",
                f"prerequisites/{name}/{role}.json",
                seed,
            )
            seed += 1
            manifests["prerequisites"]["rows"].append({
                "name": name,
                "role": role,
                "file": file_row,
            })

    for case in contract.build_case_matrix():
        continuations = contract.WORKLOAD_SPECS[
            case.workload
        ]["continuations"]
        for request_index in range(continuations):
            request_id = f"request-{request_index}"
            prompt = add_file(
                "tokens",
                f"tokens/{case.case_id}/{request_id}-prompt.json",
                seed,
            )
            seed += 1
            output = add_file(
                "tokens",
                f"tokens/{case.case_id}/{request_id}-output.json",
                seed,
            )
            seed += 1
            for role, file_row, token_count in (
                (
                    "prompt",
                    prompt,
                    contract.WORKLOAD_SPECS[case.workload][
                        "shared_prefix_tokens"
                    ]
                    + contract.WORKLOAD_SPECS[case.workload][
                        "suffix_tokens"
                    ],
                ),
                (
                    "output",
                    output,
                    contract.WORKLOAD_SPECS[case.workload][
                        "generated_tokens"
                    ],
                ),
            ):
                manifests["tokens"]["rows"].append({
                    "case_id": case.case_id,
                    "request_id": request_id,
                    "role": role,
                    "token_count": token_count,
                    "file": file_row,
                })
            logit = add_file(
                "logits",
                f"logits/{case.case_id}/{request_id}.bin",
                seed,
            )
            seed += 1
            manifests["logits"]["rows"].append({
                "case_id": case.case_id,
                "request_id": request_id,
                "shape": [248320],
                "dtype": "float32",
                "file": logit,
            })

        for rank in range(contract.WORLD_SIZE):
            log_file = add_file(
                "logs",
                f"logs/{case.case_id}/rank-{rank}.log",
                seed,
            )
            seed += 1
            manifests["logs"]["rows"].append({
                "case_id": case.case_id,
                "rank": rank,
                "world_size": contract.WORLD_SIZE,
                "completion_marker": True,
                "traceback_present": False,
                "file": log_file,
            })
            evidence_kind = (
                "accounting_only"
                if case.profile == "recompute"
                else "snapshot"
            )
            snapshot_file = None
            tensor_inventory_file = None
            accounting = {
                field: 0
                for field in contract.TENSOR_STORAGE_ACCOUNTING_FIELDS
            }
            if evidence_kind == "snapshot":
                snapshot_file = add_file(
                    "snapshots",
                    (
                        f"snapshots/{case.case_id}/rank-{rank}."
                        "snapshot"
                    ),
                    seed,
                )
                seed += 1
                tensor_inventory_file = add_file(
                    "tensor_inventories",
                    (
                        f"snapshots/{case.case_id}/rank-{rank}-"
                        "tensors.json"
                    ),
                    seed,
                )
                seed += 1
                evidence = _canonical_tensor_storage_evidence(
                    case.profile,
                    case_id=case.case_id,
                    rank=rank,
                )
                accounting = contract.recompute_tensor_storage_accounting(
                    evidence
                )
                tensor_inventory_file["sha256"] = (
                    contract.canonical_json_file_sha256(evidence)
                )
                tensor_inventory_file["bytes"] = (
                    len(contract.canonical_json_bytes(evidence)) + 1
                )
                manifests["tensor_inventories"]["rows"].append({
                    "case_id": evidence["case_id"],
                    "profile": evidence["profile"],
                    "representation": evidence["representation"],
                    "representation_version": evidence[
                        "representation_version"
                    ],
                    "codec": evidence["codec"],
                    "rank": evidence["rank"],
                    "world_size": evidence["world_size"],
                    "evidence_schema_version": evidence[
                        "schema_version"
                    ],
                    "snapshot_count": len(evidence["snapshots"]),
                    "storage_count": len(evidence["storages"]),
                    "reference_count": sum(
                        len(snapshot["tensor_references"])
                        for snapshot in evidence["snapshots"]
                    ),
                    "observation_count": len(evidence["observations"]),
                    "evidence": evidence,
                    "file": tensor_inventory_file,
                })
            manifests["snapshots"]["rows"].append({
                "case_id": case.case_id,
                "profile": case.profile,
                "rank": rank,
                "world_size": contract.WORLD_SIZE,
                "evidence_kind": evidence_kind,
                "snapshot_file": snapshot_file,
                "tensor_inventory_file": tensor_inventory_file,
                "full_fidelity_logical_bytes": accounting[
                    "hybrid_cache_current_logical_referenced_bytes"
                ],
                "encoded_physical_bytes": accounting[
                    "hybrid_cache_current_unique_physical_bytes"
                ],
                "codec_metadata_bytes": accounting[
                    "hybrid_cache_current_metadata_bytes"
                ],
                "temporary_encode_workspace_bytes": accounting[
                    "encode_workspace_peak_allocated_bytes"
                ],
                "temporary_decode_workspace_bytes": accounting[
                    "decode_workspace_peak_allocated_bytes"
                ],
                **accounting,
            })

    for manifest in manifests.values():
        manifest["files"].sort(key=lambda row: row["path"])
    file_inventory = sorted(
        [
            row
            for manifest in manifests.values()
            for row in manifest["files"]
        ],
        key=lambda row: row["path"],
    )
    artifact_manifest = {
        "schema_version": contract.ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "hash_domain": list(contract.ARTIFACT_MANIFEST_HASH_DOMAIN),
        "entries": [
            {
                "path": path,
                "sha256": "f" * 64,
                "bytes": 1,
                "producer": "task8-assembler",
                "trust_domain": "producer",
            }
            for path in contract.ARTIFACT_MANIFEST_HASH_DOMAIN
        ],
        "excluded_verifier_outputs": list(
            contract.VERIFIER_TRUST_DOMAIN
        ),
    }
    artifact_by_kind = {
        "prerequisites": "correctness_prerequisites.json",
        "tokens": "token_manifest.json",
        "logits": "logits_manifest.json",
        "logs": "worker_logs_manifest.json",
        "snapshots": "snapshot_manifest.json",
        "tensor_inventories": "tensor_inventory_manifest.json",
    }
    entries = {
        row["path"]: row for row in artifact_manifest["entries"]
    }
    for kind, path in artifact_by_kind.items():
        entries[path]["sha256"] = contract.canonical_json_file_sha256(
            manifests[kind]
        )
        entries[path]["bytes"] = len(
            contract.canonical_json_bytes(manifests[kind])
        ) + 1
    return manifests, file_inventory, artifact_manifest


def _joint_case_row_nested_evidence():
    rows = _canonical_case_rows()
    manifests, file_inventory, artifact_manifest = (
        _nested_evidence_bundle()
    )
    token_rows = {
        (row["case_id"], row["request_id"], row["role"]): row
        for row in manifests["tokens"]["rows"]
    }
    logit_rows = {
        (row["case_id"], row["request_id"]): row
        for row in manifests["logits"]["rows"]
    }
    for row in rows:
        key = (row["case_id"], row["request_id"])
        prompt = token_rows[key + ("prompt",)]["file"]
        output = token_rows[key + ("output",)]["file"]
        logit = logit_rows[key]
        row.update({
            "prompt_token_ids_path": prompt["path"],
            "prompt_token_ids_sha256": prompt["sha256"],
            "output_token_ids_path": output["path"],
            "output_token_ids_sha256": output["sha256"],
            "final_logits_path": logit["file"]["path"],
            "final_logits_sha256": logit["file"]["sha256"],
            "final_logits_shape": list(logit["shape"]),
            "final_logits_dtype": logit["dtype"],
        })
    return rows, manifests, file_inventory, artifact_manifest


def _joint_process_row_nested_worker_logs():
    process_rows = _canonical_process_rows()
    manifests, file_inventory, artifact_manifest = (
        _nested_evidence_bundle()
    )
    return process_rows, manifests, file_inventory, artifact_manifest


def _joint_process_row_nested_snapshots():
    process_rows = _canonical_process_rows()
    manifests, file_inventory, artifact_manifest = (
        _nested_evidence_bundle()
    )
    for process_row, snapshot_row in zip(
        process_rows,
        manifests["snapshots"]["rows"],
    ):
        process_row.update({
            field: snapshot_row[field]
            for field in contract.TENSOR_STORAGE_ACCOUNTING_FIELDS
        })
    return process_rows, manifests, file_inventory, artifact_manifest


def _evidence_backed_snapshot_collection():
    process_rows, manifests, file_inventory, artifact_manifest = (
        _joint_process_row_nested_snapshots()
    )
    inventory_by_path = {
        row["path"]: row for row in file_inventory
    }
    tensor_rows = {
        (
            row["case_id"],
            row["profile"],
            row["rank"],
        ): row
        for row in manifests["tensor_inventories"]["rows"]
    }
    accounting_fields = (
        "cuda_allocated_bytes",
        "cuda_reserved_bytes",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "encode_workspace_peak_allocated_bytes",
        "encode_workspace_peak_reserved_bytes",
        "decode_workspace_peak_allocated_bytes",
        "decode_workspace_peak_reserved_bytes",
        "hybrid_cache_current_entries",
        "hybrid_cache_current_unique_physical_bytes",
        "hybrid_cache_current_logical_referenced_bytes",
        "hybrid_cache_current_metadata_bytes",
        "hybrid_cache_deduplicated_bytes",
        "hybrid_cache_peak_entries",
        "hybrid_cache_peak_unique_physical_bytes",
        "hybrid_cache_peak_logical_referenced_bytes",
        "hybrid_cache_peak_metadata_bytes",
    )
    for process_row, snapshot_row in zip(
        process_rows,
        manifests["snapshots"]["rows"],
    ):
        if process_row["profile"] == "recompute":
            for field in accounting_fields:
                process_row[field] = 0
            continue
        evidence = _canonical_tensor_storage_evidence(
            process_row["profile"],
            case_id=process_row["case_id"],
            rank=process_row["rank"],
        )
        accounting = contract.recompute_tensor_storage_accounting(evidence)
        process_row.update(accounting)
        snapshot_row.update(accounting)
        snapshot_row.update({
            "full_fidelity_logical_bytes": accounting[
                "hybrid_cache_current_logical_referenced_bytes"
            ],
            "encoded_physical_bytes": accounting[
                "hybrid_cache_current_unique_physical_bytes"
            ],
            "codec_metadata_bytes": accounting[
                "hybrid_cache_current_metadata_bytes"
            ],
            "temporary_encode_workspace_bytes": accounting[
                "encode_workspace_peak_allocated_bytes"
            ],
            "temporary_decode_workspace_bytes": accounting[
                "decode_workspace_peak_allocated_bytes"
            ],
        })
        tensor_row = tensor_rows[(
            process_row["case_id"],
            process_row["profile"],
            process_row["rank"],
        )]
        tensor_row.clear()
        tensor_row.update({
            "case_id": evidence["case_id"],
            "profile": evidence["profile"],
            "representation": evidence["representation"],
            "representation_version": evidence[
                "representation_version"
            ],
            "codec": evidence["codec"],
            "rank": evidence["rank"],
            "world_size": evidence["world_size"],
            "evidence_schema_version": evidence["schema_version"],
            "snapshot_count": len(evidence["snapshots"]),
            "storage_count": len(evidence["storages"]),
            "reference_count": sum(
                len(snapshot["tensor_references"])
                for snapshot in evidence["snapshots"]
            ),
            "observation_count": len(evidence["observations"]),
            "evidence": evidence,
            "file": snapshot_row["tensor_inventory_file"],
        })
        file_row = tensor_row["file"]
        file_row["sha256"] = contract.canonical_json_file_sha256(
            evidence
        )
        file_row["bytes"] = len(contract.canonical_json_bytes(evidence)) + 1
        inventory_by_path[file_row["path"]].update(file_row)

    snapshot_manifest_entry = next(
        row
        for row in artifact_manifest["entries"]
        if row["path"] == "snapshot_manifest.json"
    )
    snapshot_manifest_entry["sha256"] = (
        contract.canonical_json_file_sha256(manifests["snapshots"])
    )
    snapshot_manifest_entry["bytes"] = (
        len(contract.canonical_json_bytes(manifests["snapshots"])) + 1
    )
    tensor_manifest_entry = next(
        row
        for row in artifact_manifest["entries"]
        if row["path"] == "tensor_inventory_manifest.json"
    )
    tensor_manifest_entry["sha256"] = (
        contract.canonical_json_file_sha256(
            manifests["tensor_inventories"]
        )
    )
    tensor_manifest_entry["bytes"] = (
        len(
            contract.canonical_json_bytes(
                manifests["tensor_inventories"]
            )
        )
        + 1
    )
    return process_rows, manifests, file_inventory, artifact_manifest


def _canonical_artifact_evidence():
    case_rows, manifests, file_inventory, artifact_manifest = (
        _joint_case_row_nested_evidence()
    )
    process_rows = _canonical_process_rows()
    for process_row, snapshot_row in zip(
        process_rows,
        manifests["snapshots"]["rows"],
    ):
        process_row.update({
            field: snapshot_row[field]
            for field in contract.TENSOR_STORAGE_ACCOUNTING_FIELDS
        })
    return (
        case_rows,
        process_rows,
        manifests,
        file_inventory,
        artifact_manifest,
    )


def _assert_case_and_nested_planes_valid(
    rows,
    manifests,
    file_inventory,
    artifact_manifest,
):
    contract.validate_case_rows(rows)
    contract.validate_nested_evidence_bundle(
        manifests,
        file_inventory,
        artifact_manifest,
    )


def _assert_rejects_unknown_field(validator, row):
    row["unexpected"] = True
    with pytest.raises(ValueError, match="unknown|schema|fields"):
        validator(row)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _root_authority_payloads():
    comparisons = []
    for index, case_id in enumerate(contract.TP4_ROOT_CASE_IDS):
        winner = 100 + index
        runner_up = 200 + index
        comparisons.append({
            "case_id": case_id,
            "native_winner_token_id": winner,
            "native_runner_up_token_id": runner_up,
            "native_winner_margin": 1.0,
            "official_winner_token_id": winner,
            "official_runner_up_token_id": runner_up,
            "official_winner_margin": 1.0,
            "native_topk_token_ids": [winner, runner_up],
            "official_topk_token_ids": [winner, runner_up],
        })
    return (
        {
            "schema_version": contract.TP4_ROOT_CORRECTNESS_SCHEMA_VERSION,
            "run_tag": "tp4-root-logit-run",
            "classification": "PASS",
            "comparison_policy": "registered_logits_strict_allclose",
            "tolerance": {"atol": 2e-5, "rtol": 0.0},
            "prompts": [
                {"case_id": case_id}
                for case_id in contract.TP4_ROOT_CASE_IDS
            ],
            "reference_process": {
                "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
            },
            "comparisons": comparisons,
            "forbidden_counters": {
                "engine": 0,
                "generation": 0,
                "model_runner": 0,
                "sampler": 0,
                "scheduler": 0,
            },
            "claim_boundary": (
                "TP4 root-logit correctness only; no cached decode"
            ),
        },
        {
            "classification": "PASS",
            "case_ids": list(contract.TP4_ROOT_CASE_IDS),
            "ranks": [0, 1, 2, 3],
            "checks": 100,
        },
    )


def _cached_authority_payloads(source_tree_sha256):
    rows = []
    for workload in contract.WORKLOADS[1:]:
        spec = contract.WORKLOAD_SPECS[workload]
        for request_index in range(spec["continuations"]):
            restore_hit = workload != "w4_miss_invalidation"
            rows.append({
                "workload": workload,
                "request_index": request_index,
                "outcome": "continuation",
                "restore_hit": restore_hit,
                "restore_reason": (
                    "exact_hit"
                    if restore_hit
                    else (
                        "token_mismatch",
                        "stale_block_generation",
                        "cache_clear",
                    )[request_index]
                ),
                "prompt_tokens": (
                    spec["shared_prefix_tokens"] + spec["suffix_tokens"]
                ),
                "reused_tokens": (
                    spec["shared_prefix_tokens"] if restore_hit else 0
                ),
                "executed_prefill_tokens": (
                    spec["suffix_tokens"]
                    if restore_hit
                    else (
                        spec["shared_prefix_tokens"]
                        + spec["suffix_tokens"]
                    )
                ),
                "output_token_ids": [7] * spec["generated_tokens"],
                "reference_output_token_ids": (
                    [7] * spec["generated_tokens"]
                ),
                "logits_max_abs_diff": 0.0,
                "logits_allclose": True,
                "cache_identity_match": True,
                "rank_inventory": [0, 1, 2, 3],
                "process_group_destroyed": True,
                "owned_children_remaining": [],
            })
    checks = {
        "row_count": len(rows),
        "restore_hits": sum(row["restore_hit"] for row in rows),
        "w4_misses": sum(not row["restore_hit"] for row in rows),
    }
    workload_sha256 = contract.APPROVED_V1_WORKLOAD_MANIFEST_SHA256
    return (
        {
            "schema_version": contract.CACHED_CONTINUATION_SCHEMA_VERSION,
            "classification": "PASS",
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
            "workload_manifest_sha256": workload_sha256,
            "rows": rows,
        },
        {
            "schema_version": contract.CACHED_CONTINUATION_SCHEMA_VERSION,
            "classification": "PASS",
            "source_tree_sha256": source_tree_sha256,
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
            "workload_manifest_sha256": workload_sha256,
            "checks": checks,
        },
    )


def _engine_authority_payloads(source_tree_sha256):
    rows = []
    for scenario, expected in contract.ENGINE_CORRECTNESS_SCENARIOS.items():
        (
            scheduler_steps,
            model_runner_calls,
            generated_tokens,
            publication_commits,
            restore_hits,
            restore_misses,
            release_events,
            cache_entries_after,
        ) = expected
        rows.append({
            "scenario": scenario,
            "engine_class": "tinyvllm.engine.llm_engine.LLMEngine",
            "model_runner_class": (
                "tinyvllm.engine.model_runner.ModelRunner"
            ),
            "rank_inventory": [0, 1, 2, 3],
            "ack_ranks": [1, 2, 3],
            "scheduler_steps": scheduler_steps,
            "model_runner_calls": model_runner_calls,
            "output_token_ids": [7] * generated_tokens,
            "reference_output_token_ids": [7] * generated_tokens,
            "publication_commits": publication_commits,
            "restore_hits": restore_hits,
            "restore_misses": restore_misses,
            "release_events": release_events,
            "cache_entries_after": cache_entries_after,
            "cache_identity_match": True,
            "process_group_destroyed": True,
            "rank_exit_codes": [0, 0, 0, 0],
            "owned_children_remaining": [],
        })
    checks = {
        "scenario_count": len(rows),
        "restore_hits": sum(row["restore_hits"] for row in rows),
        "restore_misses": sum(row["restore_misses"] for row in rows),
    }
    return (
        {
            "schema_version": contract.ENGINE_CORRECTNESS_SCHEMA_VERSION,
            "classification": "PASS",
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
            "rows": rows,
        },
        {
            "schema_version": contract.ENGINE_CORRECTNESS_SCHEMA_VERSION,
            "classification": "PASS",
            "source_tree_sha256": source_tree_sha256,
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
            "checks": checks,
        },
    )


def _canonical_sha(value):
    return contract.canonical_json_sha256(value)


def _command_rows(names):
    return {
        name: {"argv": [name, "--frozen"]}
        for name in names
    }


def _root_prerequisite_documents(
    name,
    source_tree_sha256,
    artifact_payload,
    verification_payload,
    *,
    controlled_shared=False,
):
    stage_order = ["preflight", "run", "download", "verify"]
    repo_root = "/frozen/repo"
    local_run_dir = (
        f"{repo_root}/experiments/qwen35_hybrid_state/{name}"
    )
    remote_run_dir = (
        "/data00/home/sitian/sitian-workspace01/tllm/"
        f"qwen35-tp4-root-logit-tests/{name}"
    )
    exact_artifact_names = [
        "native_rank0_logits.pt",
        "rank_evidence.json",
        "reference_logits.pt",
        "source_manifest.json",
        "tp4_real_root_logit_correctness.json",
    ]
    stage_inputs = {
        "preflight": {
            "run_tag": name,
            "repo_root": repo_root,
        },
        "run": {
            "run_tag": name,
            "repo_root": repo_root,
            "remote_run_dir": remote_run_dir,
            "frozen_source_tree_sha256": source_tree_sha256,
        },
        "download": {
            "run_tag": name,
            "repo_root": repo_root,
            "remote_run_dir": remote_run_dir,
            "local_artifact_dir": f"{local_run_dir}/artifacts",
            "exact_artifact_names": exact_artifact_names,
        },
        "verify": {
            "run_tag": name,
            "repo_root": repo_root,
            "local_artifact_dir": f"{local_run_dir}/artifacts",
            "independent_verification_path": (
                f"{local_run_dir}/independent_verification.json"
            ),
            "frozen_source_tree_sha256": source_tree_sha256,
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        },
    }
    plan = {
        "schema_version": (
            "qwen35.tp4-root-logit-remote-execution-plan.v1"
        ),
        "run_tag": name,
        "repo_root": repo_root,
        "local_run_dir": local_run_dir,
        "ssh_target": "sitian@10.232.195.203",
        "remote_run_dir": remote_run_dir,
        "frozen_source_tag": "qwen35-tp4-source-prep-20260729-170818",
        "frozen_source_tree_sha256": source_tree_sha256,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "exact_artifact_names": exact_artifact_names,
        "minimum_free_bytes_per_gpu": 24 * 1024**3,
        "requires_no_active_compute_processes": not controlled_shared,
        "stage_order": stage_order,
        "stage_inputs": stage_inputs,
        "execution_performed": False,
        "claim_boundary": (
            "execution authorization only; no SSH, GPU, correctness, "
            "performance, cache, memory, compression, or quality claim"
        ),
        "plan_output_dir": "/frozen/plan",
    }
    gpu_indices = [2, 4, 5, 6]
    gpu_uuids = [f"GPU-root-{index}" for index in gpu_indices]
    baseline_sha256 = "b" * 64
    if controlled_shared:
        resource_binding = {
            "resource_policy": "controlled_shared",
            "resource_baseline_path": (
                "/frozen/plan/resource_baseline.json"
            ),
            "resource_baseline_sha256": baseline_sha256,
            "gpu_indices": gpu_indices,
            "gpu_uuids": gpu_uuids,
            "benchmark_execution_authorized": False,
        }
        plan.update(resource_binding)
        for stage in ("preflight", "run"):
            stage_inputs[stage].update(resource_binding)
    authorization = {
        "schema_version": (
            "qwen35.tp4-root-logit-remote-execution-authorization.v1"
        ),
        "classification": "AUTHORIZED",
        "plan_sha256": _canonical_sha(plan),
        "run_tag": name,
        "ssh_target": plan["ssh_target"],
        "frozen_source_tree_sha256": source_tree_sha256,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "stage_order": stage_order,
        "nonce": f"{name}-nonce",
        "consumed": True,
    }
    if controlled_shared:
        authorization.update({
            "resource_policy": "controlled_shared",
            "resource_baseline_sha256": baseline_sha256,
        })
    selected = []
    query_rows = []
    for rank in range(4):
        gpu_index = gpu_indices[rank] if controlled_shared else rank
        resource_row = {
            "gpu_index": gpu_index,
            "gpu_uuid": (
                gpu_uuids[rank]
                if controlled_shared
                else f"GPU-{rank}"
            ),
            "free_bytes": 25 * 1024**3,
            "compute_processes": (
                [{
                    "pid": 1000 + gpu_index,
                    "process_name": "python3",
                    "used_memory_mib": 436,
                    "start_time_ticks": 2000 + gpu_index,
                }]
                if controlled_shared
                else []
            ),
        }
        if controlled_shared:
            selected.append({
                **resource_row,
                "rank": rank,
                "world_size": 4,
                "minimum_free_bytes": 24 * 1024**3,
            })
        else:
            query_row = {
                **resource_row,
                "gpu_name": f"GPU model {rank}",
                "total_bytes": 80 * 1024**3,
            }
            query_rows.append(query_row)
            selected.append({
                **query_row,
                "rank": rank,
                "world_size": 4,
                "minimum_free_bytes": 24 * 1024**3,
            })
    stage_results = {
        "preflight": {
            "status": "READY",
            "run_tag": name,
            "frozen_source_tag": plan["frozen_source_tag"],
            "frozen_source_tree_sha256": source_tree_sha256,
            "source_tree_sha256": source_tree_sha256,
            "selected": selected,
            "rows": (
                copy.deepcopy(selected)
                if controlled_shared
                else query_rows
            ),
        },
        "run": {
            "status": "REMOTE_PASS",
            "run_tag": name,
            "remote_run_dir": plan["remote_run_dir"],
            "artifact_names": plan["exact_artifact_names"],
        },
        "download": {
            "status": "DOWNLOADED",
            "artifact_names": plan["exact_artifact_names"],
        },
        "verify": verification_payload,
    }
    if controlled_shared:
        stage_results["preflight"].update({
            "resource_policy": "controlled_shared",
            "baseline_sha256": baseline_sha256,
            "benchmark_execution_authorized": False,
        })
        stage_results["run"]["final_resource"] = {
            "classification": "READY",
            "resource_policy": "controlled_shared",
            "baseline_sha256": baseline_sha256,
            "selected": [
                {
                    key: copy.deepcopy(row[key])
                    for key in (
                        "gpu_index",
                        "gpu_uuid",
                        "free_bytes",
                        "compute_processes",
                    )
                }
                for row in selected
            ],
            "benchmark_execution_authorized": False,
        }
    receipt = {
        "schema_version": (
            "qwen35.tp4-root-logit-remote-execution-receipt.v1"
        ),
        "classification": "PASS",
        "plan_sha256": _canonical_sha(plan),
        "authorization_sha256": _canonical_sha(authorization),
        "authorization_nonce": authorization["nonce"],
        "run_tag": name,
        "stages": [
            {
                "name": stage,
                "result_sha256": _canonical_sha(stage_results[stage]),
                "result": stage_results[stage],
            }
            for stage in stage_order
        ],
    }
    return plan, authorization, receipt


def _workload_prerequisite_documents(
    name,
    source_tree_sha256,
    artifact_payload,
    verification_payload,
):
    command_order = [
        "reserve_remote",
        "upload",
        "stage",
        "resource_guard",
        "guarded_authority",
        "package_download",
        "safe_extract",
        "prepare_local_verifier",
        "local_verify",
    ]
    ssh_target = "sitian@10.232.195.203"
    is_cached = name == "cached_continuation"
    planner = (
        native_cached_plan_test.planner
        if is_cached
        else native_engine_plan_test.planner
    )
    remote_run_root = f"{planner.REMOTE_ROOT}/{name}"
    remote_inputs_root = f"{remote_run_root}/inputs"
    remote_source_root = f"{remote_run_root}/source"
    remote_authority_root = f"{remote_run_root}/authority"
    local_output_root = Path(f"/frozen/{name}-plan")
    local_source_inventory = str(
        local_output_root / "source_inventory.json"
    )
    local_source_tar = str(
        local_output_root / planner.SOURCE_TAR_NAME
    )
    plan = {
        "schema_version": (
            "qwen35.tp4-cached-continuation-remote-execution-plan.v1"
            if is_cached
            else "qwen35.tp4-engine-remote-execution-plan.v1"
        ),
        "run_tag": name,
        "ssh_target": ssh_target,
        "remote_run_root": remote_run_root,
        "remote_source_root": remote_source_root,
        "remote_authority_root": remote_authority_root,
        "gpu_indices": [0, 1, 2, 3],
        "ports": {"dist_port": 32001, "master_port": 32002},
        "source_tree_sha256": source_tree_sha256,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "local_inputs": {
            "configuration": str(
                local_output_root / planner.REMOTE_CONFIGURATION_NAME
            ),
            "configuration_sha256": "1" * 64,
            "source_inventory": local_source_inventory,
            "source_inventory_sha256": "2" * 64,
            "source_tar": local_source_tar,
            "source_tar_sha256": "3" * 64,
            "workload_manifest": str(
                local_output_root / "workload_manifest.json"
            ),
            "workload_manifest_sha256": (
                contract.APPROVED_V1_WORKLOAD_MANIFEST_SHA256
            ),
        },
        "remote_inputs": {
            "configuration": (
                f"{remote_inputs_root}/"
                f"{planner.REMOTE_CONFIGURATION_NAME}"
            ),
            "source_inventory": (
                f"{remote_inputs_root}/source_inventory.json"
            ),
            "source_tar": (
                f"{remote_inputs_root}/{planner.SOURCE_TAR_NAME}"
            ),
            "workload_manifest": (
                f"{remote_inputs_root}/workload_manifest.json"
            ),
        },
        "command_order": command_order,
        "commands": {},
        "execution_performed": False,
        "claim_boundary": (
            "command authorization only; no correctness claim"
        ),
    }
    if is_cached:
        plan.update({
            "remote_cached_authority_dir": (
                f"{remote_authority_root}/"
                "cached_continuation_authority"
            ),
            "remote_cached_verification_path": (
                f"{remote_authority_root}/"
                "cached_continuation_independent_verification.json"
            ),
        })
    configuration = SimpleNamespace(
        gpu_indices=tuple(plan["gpu_indices"]),
        dist_port=plan["ports"]["dist_port"],
        master_port=plan["ports"]["master_port"],
        source_tree_sha256=source_tree_sha256,
    )
    identities = {
        "configuration_sha256": plan["local_inputs"][
            "configuration_sha256"
        ],
        "source_inventory_sha256": plan["local_inputs"][
            "source_inventory_sha256"
        ],
        "source_tar_sha256": plan["local_inputs"][
            "source_tar_sha256"
        ],
        "source_tree_sha256": source_tree_sha256,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "workload_manifest_sha256": (
            contract.APPROVED_V1_WORKLOAD_MANIFEST_SHA256
        ),
    }
    if is_cached:
        paths = {
            "remote_run": remote_run_root,
            "remote_inputs_root": remote_inputs_root,
            "remote_source": remote_source_root,
            "remote_authority_root": remote_authority_root,
            "remote_cached_authority_dir": plan[
                "remote_cached_authority_dir"
            ],
            "remote_cached_verification_path": plan[
                "remote_cached_verification_path"
            ],
            "authority_tar": (
                local_output_root / "cached_authority.tar"
            ),
            "downloaded": (
                local_output_root / planner.DOWNLOADED_AUTHORITY_NAME
            ),
            "verifier_source": (
                local_output_root / planner.LOCAL_VERIFIER_SOURCE_NAME
            ),
        }
        commands = planner._commands(
            configuration=configuration,
            paths=paths,
            remote_inputs=plan["remote_inputs"],
            identities=identities,
            local_inputs=plan["local_inputs"],
            resource_policy="strict_exclusive",
            resource_baseline_sha256=None,
        )
    else:
        authority_tar = local_output_root / "authority.tar"
        downloaded = (
            local_output_root / planner.DOWNLOADED_AUTHORITY_NAME
        )
        verifier_source = (
            local_output_root / planner.LOCAL_VERIFIER_SOURCE_NAME
        )
        authority_argv = [
            "env",
            f"PYTHONPATH={remote_source_root}",
            "PYTHONDONTWRITEBYTECODE=1",
            "TORCH_COMPILE_DISABLE=1",
            "CUDA_VISIBLE_DEVICES=0,1,2,3",
            "TINYVLLM_DIST_PORT=32001",
            "MASTER_PORT=32002",
            planner.REMOTE_PYTHON,
            (
                f"{remote_source_root}/tools/"
                "run_qwen35_tp4_engine_correctness_authority.py"
            ),
            "--configuration",
            plan["remote_inputs"]["configuration"],
            "--source-inventory",
            plan["remote_inputs"]["source_inventory"],
            "--output-root",
            remote_authority_root,
        ]
        upload_argv = [
            planner._scp(
                Path(plan["local_inputs"][key]),
                plan["remote_inputs"][key],
            )
            for key in (
                "configuration",
                "source_inventory",
                "source_tar",
                "workload_manifest",
            )
        ]
        commands = {
            "reserve_remote": {
                "argv": planner._ssh([
                    "bash",
                    "-lc",
                    " && ".join([
                        "set -eu",
                        f"test ! -e {remote_run_root}",
                        f"mkdir -p {remote_run_root}",
                        f"mkdir {remote_inputs_root}",
                    ]),
                ]),
            },
            "upload": {"argv": upload_argv},
            "stage": {
                "argv": planner._ssh([
                    "bash",
                    "-lc",
                    planner._stage_script(
                        remote_source_root,
                        remote_inputs_root,
                        identities,
                    ),
                ]),
            },
            "resource_guard": {
                "argv": planner._ssh(
                    planner._resource_guard_command(
                        configuration.gpu_indices
                    )
                ),
                "gpu_indices": list(configuration.gpu_indices),
                "minimum_free_bytes_per_gpu": (
                    planner.MIN_GPU_FREE_BYTES
                ),
                "requires_no_active_compute_processes": True,
            },
            "guarded_authority": {
                "authority_argv": authority_argv,
                "ssh_argv": planner._ssh(
                    planner._guarded_authority_command(
                        configuration.gpu_indices,
                        authority_argv,
                    )
                ),
                "final_resource_recheck": True,
            },
            "package_download": {
                "remote_argv": planner._ssh([
                    "bash",
                    "-lc",
                    planner._package_script(remote_authority_root),
                ]),
                "local_output": str(authority_tar),
            },
            "safe_extract": {
                "argv": planner._extract_command(
                    authority_tar,
                    downloaded,
                ),
            },
            "prepare_local_verifier": {
                "argv": planner._prepare_local_verifier_command(
                    Path(local_source_tar),
                    Path(local_source_inventory),
                    source_tree_sha256,
                    verifier_source,
                ),
                "source_tar": local_source_tar,
                "source_inventory": local_source_inventory,
                "source_tree_sha256": source_tree_sha256,
            },
            "local_verify": {
                "argv": [
                    sys.executable,
                    str(
                        verifier_source
                        / "tools"
                        / "verify_qwen35_tp4_engine_correctness_authority.py"
                    ),
                    str(downloaded),
                ],
            },
        }
    plan["commands"] = commands
    authorization = {
        "schema_version": (
            "qwen35.tp4-engine-remote-execution-authorization.v1"
        ),
        "classification": "AUTHORIZED",
        "plan_sha256": _canonical_sha(plan),
        "run_tag": name,
        "source_tree_sha256": source_tree_sha256,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "workload_manifest_sha256": (
            contract.APPROVED_V1_WORKLOAD_MANIFEST_SHA256
        ),
        "gpu_indices": plan["gpu_indices"],
        "ports": plan["ports"],
        "nonce": f"{name}-nonce",
        "consumed": True,
    }
    resource = {
        "classification": "READY",
        "selected": [
            {
                "gpu_index": index,
                "gpu_uuid": f"GPU-{index}",
                "free_bytes": 25 * 1024**3,
                "compute_processes": [],
            }
            for index in plan["gpu_indices"]
        ],
    }
    pass_payload = (
        {
            "classification": "PASS",
            "schema_version": contract.CACHED_CONTINUATION_SCHEMA_VERSION,
            "source_tree_sha256": source_tree_sha256,
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
            "workload_manifest_sha256": (
                contract.APPROVED_V1_WORKLOAD_MANIFEST_SHA256
            ),
            "checks": verification_payload["checks"],
        }
        if is_cached
        else {
            "classification": "PASS",
            "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
            "source_tree_sha256": source_tree_sha256,
            "workload_manifest_sha256": (
                contract.APPROVED_V1_WORKLOAD_MANIFEST_SHA256
            ),
            "reference_classification": "PASS",
            "engine_classification": "PASS",
        }
    )
    authority_payload = (
        pass_payload
        if is_cached
        else {
            **pass_payload,
            "inventory": [
                "reference_authority",
                "reference_independent_verification.json",
                "engine_authority",
                "authority_summary.json",
            ],
        }
    )
    steps = []
    for command_name in command_order:
        stdout = ""
        if command_name == "resource_guard":
            stdout = json.dumps(resource, sort_keys=True)
        elif command_name == "guarded_authority":
            stdout = "\n".join([
                "QWEN35_FINAL_RESOURCE_JSON="
                + json.dumps(resource, sort_keys=True),
                json.dumps(authority_payload, sort_keys=True),
            ])
        elif command_name == "local_verify":
            stdout = json.dumps(pass_payload, sort_keys=True)
        step = {
            "name": command_name,
            "command_sha256": _canonical_sha(commands[command_name]),
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
        }
        if command_name == "package_download":
            step.update({
                "output_sha256": "4" * 64,
                "output_size": 4096,
            })
        steps.append(step)
    receipt = {
        "schema_version": (
            "qwen35.tp4-cached-continuation-remote-execution-receipt.v1"
            if is_cached
            else "qwen35.tp4-engine-remote-execution-receipt.v1"
        ),
        "plan_sha256": _canonical_sha(plan),
        "authorization_sha256": _canonical_sha(authorization),
        "authorization_nonce": authorization["nonce"],
        "run_tag": name,
        "steps": steps,
        "classification": "PASS",
    }
    return plan, authorization, receipt


def _prerequisite_documents(
    name,
    source_tree_sha256,
    artifact_payload,
    verification_payload,
):
    if name == "tp4_root_logit":
        return _root_prerequisite_documents(
            name,
            source_tree_sha256,
            artifact_payload,
            verification_payload,
        )
    return _workload_prerequisite_documents(
        name,
        source_tree_sha256,
        artifact_payload,
        verification_payload,
    )


def _prerequisite_authority(root, name):
    authority_dir = root / name
    source_tree_sha256 = (
        contract.TP4_ROOT_SOURCE_TREE_SHA256
        if name == "tp4_root_logit"
        else "a" * 64
    )
    if name == "tp4_root_logit":
        artifact_payload, verification_payload = (
            _root_authority_payloads()
        )
    elif name == "cached_continuation":
        artifact_payload, verification_payload = (
            _cached_authority_payloads(source_tree_sha256)
        )
    else:
        artifact_payload, verification_payload = (
            _engine_authority_payloads(source_tree_sha256)
        )
    artifact = authority_dir / "artifact.json"
    verification = authority_dir / "independent_verification.json"
    provenance = authority_dir / "provenance.json"
    _write_json(artifact, artifact_payload)
    _write_json(verification, verification_payload)
    plan_payload, authorization_payload, receipt_payload = (
        _prerequisite_documents(
            name,
            source_tree_sha256,
            artifact_payload,
            verification_payload,
        )
    )
    evidence = {}
    for filename, document in (
        ("execution_plan.json", plan_payload),
        ("consumed_authorization.json", authorization_payload),
        ("execution_receipt.json", receipt_payload),
    ):
        path = authority_dir / filename
        _write_json(path, document)
        evidence[filename] = contract.sha256_file(path)
    _write_json(provenance, {
        "schema_version": contract.PREREQUISITE_PROVENANCE_SCHEMA_VERSION,
        "authority_name": name,
        "run_tag": name,
        "binding_kind": "remote_execution_receipt",
        "source_tree_sha256": source_tree_sha256,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "root_logit_receipt_gap": False,
        "plan_path": "execution_plan.json",
        "plan_sha256": evidence["execution_plan.json"],
        "authorization_path": "consumed_authorization.json",
        "authorization_sha256": evidence[
            "consumed_authorization.json"
        ],
        "receipt_path": "execution_receipt.json",
        "receipt_sha256": evidence["execution_receipt.json"],
    })
    return {
        "run_tag": name,
        "source_tree_sha256": source_tree_sha256,
        "artifact_path": artifact.relative_to(root).as_posix(),
        "artifact_sha256": contract.sha256_file(artifact),
        "independent_verification_path": (
            verification.relative_to(root).as_posix()
        ),
        "independent_verification_sha256": contract.sha256_file(
            verification
        ),
        "provenance_path": provenance.relative_to(root).as_posix(),
        "provenance_sha256": contract.sha256_file(provenance),
        "classification": "PASS",
    }


def _complete_prerequisite_fixture(root):
    payload = {
        "schema_version": contract.PREREQUISITE_SCHEMA_VERSION,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "tp4_root_logit": _prerequisite_authority(
            root,
            "tp4_root_logit",
        ),
        "cached_continuation": _prerequisite_authority(
            root,
            "cached_continuation",
        ),
        "engine_correctness": _prerequisite_authority(
            root,
            "engine_correctness",
        ),
    }
    path = root / "correctness_prerequisites.json"
    _write_json(path, payload)
    return path, payload


def _refresh_prerequisite_evidence(
    root,
    path,
    payload,
    name,
    filename,
    document,
):
    row = payload[name]
    provenance_path = root / row["provenance_path"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    evidence_path = provenance_path.parent / filename
    _write_json(evidence_path, document)
    sha_field = {
        "execution_plan.json": "plan_sha256",
        "consumed_authorization.json": "authorization_sha256",
        "execution_receipt.json": "receipt_sha256",
    }[filename]
    provenance[sha_field] = contract.sha256_file(evidence_path)
    _write_json(provenance_path, provenance)
    row["provenance_sha256"] = contract.sha256_file(provenance_path)
    _write_json(path, payload)


def _refresh_raw_prerequisite_evidence(
    root,
    path,
    payload,
    name,
    filename,
    content,
):
    row = payload[name]
    provenance_path = root / row["provenance_path"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    evidence_path = provenance_path.parent / filename
    evidence_path.write_text(content, encoding="utf-8")
    sha_field = {
        "execution_plan.json": "plan_sha256",
        "consumed_authorization.json": "authorization_sha256",
        "execution_receipt.json": "receipt_sha256",
    }[filename]
    provenance[sha_field] = contract.sha256_file(evidence_path)
    _write_json(provenance_path, provenance)
    row["provenance_sha256"] = contract.sha256_file(provenance_path)
    _write_json(path, payload)


def _load_prerequisite_evidence(root, payload, name, filename):
    provenance_path = root / payload[name]["provenance_path"]
    return json.loads(
        (provenance_path.parent / filename).read_text(encoding="utf-8")
    )


def _write_prerequisite_bundle(
    root,
    path,
    payload,
    name,
    plan,
    authorization,
    receipt,
):
    row = payload[name]
    provenance_path = root / row["provenance_path"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for filename, document, sha_field in (
        ("execution_plan.json", plan, "plan_sha256"),
        (
            "consumed_authorization.json",
            authorization,
            "authorization_sha256",
        ),
        ("execution_receipt.json", receipt, "receipt_sha256"),
    ):
        evidence_path = provenance_path.parent / filename
        _write_json(evidence_path, document)
        provenance[sha_field] = contract.sha256_file(evidence_path)
    _write_json(provenance_path, provenance)
    row["provenance_sha256"] = contract.sha256_file(provenance_path)
    _write_json(path, payload)


def _semantic_bundle(root, payload, name):
    return tuple(
        _load_prerequisite_evidence(root, payload, name, filename)
        for filename in (
            "execution_plan.json",
            "consumed_authorization.json",
            "execution_receipt.json",
        )
    )


def _refresh_semantic_bindings(plan, authorization, receipt):
    authorization["plan_sha256"] = _canonical_sha(plan)
    receipt["plan_sha256"] = _canonical_sha(plan)
    receipt["authorization_sha256"] = _canonical_sha(authorization)
    receipt["authorization_nonce"] = authorization["nonce"]
    if "steps" in receipt:
        for step in receipt["steps"]:
            step["command_sha256"] = _canonical_sha(
                plan["commands"][step["name"]]
            )
    if "stages" in receipt:
        for stage in receipt["stages"]:
            stage["result_sha256"] = _canonical_sha(stage["result"])


def _raise_if_prerequisites_are_rejected(path):
    result = contract.validate_prerequisites(path)
    if not result.authorized:
        raise ValueError(" ".join(result.reasons))


def _assert_prerequisite_mutation_rejected(
    name,
    mutate,
    expected_reason,
):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, payload = _complete_prerequisite_fixture(root)
        plan, authorization, receipt = (
            copy.deepcopy(document)
            for document in _semantic_bundle(root, payload, name)
        )
        mutate(plan, authorization, receipt, payload[name])
        _write_prerequisite_bundle(
            root,
            path,
            payload,
            name,
            plan,
            authorization,
            receipt,
        )

        result = contract.validate_prerequisites(path)

        assert result.authorized is False
        assert expected_reason in " ".join(result.reasons)


def _replace_nested_string(value, old, new):
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [
            _replace_nested_string(item, old, new)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _replace_nested_string(item, old, new)
            for key, item in value.items()
        }
    return value


@pytest.mark.parametrize(
    "name",
    ("cached_continuation", "engine_correctness"),
)
def test_prerequisites_reject_non_env_authority_substitution(name):
    def mutate(plan, authorization, receipt, _row):
        authority = [
            "/opt/attacker/python3",
            "/opt/attacker/authority.py",
            "--accepted-by-shape-only-validation",
        ]
        guarded = plan["commands"]["guarded_authority"]
        guarded["authority_argv"] = authority
        guarded["ssh_argv"] = (
            native_engine_plan_test.planner._ssh(
                native_engine_plan_test.planner._guarded_authority_command(
                    plan["gpu_indices"],
                    authority,
                )
            )
        )
        _refresh_semantic_bindings(plan, authorization, receipt)

    _assert_prerequisite_mutation_rejected(
        name,
        mutate,
        "command mapping",
    )


@pytest.mark.parametrize(
    ("name", "command_name", "replacement"),
    [
        (
            "engine_correctness",
            "stage",
            ["bash", "-lc", "mkdir -p /tmp/rebound-stage"],
        ),
        (
            "cached_continuation",
            "resource_guard",
            ["bash", "-lc", "printf '{\"classification\":\"READY\"}'"],
        ),
    ],
)
def test_prerequisites_reject_rebound_remote_command_mapping(
    name,
    command_name,
    replacement,
):
    def mutate(plan, authorization, receipt, _row):
        plan["commands"][command_name]["argv"] = (
            native_engine_plan_test.planner._ssh(replacement)
        )
        _refresh_semantic_bindings(plan, authorization, receipt)

    _assert_prerequisite_mutation_rejected(
        name,
        mutate,
        "command mapping",
    )


@pytest.mark.parametrize(
    "name",
    ("cached_continuation", "engine_correctness"),
)
def test_prerequisites_reject_upload_row_path_substitution(name):
    def mutate(plan, authorization, receipt, _row):
        upload = plan["commands"]["upload"]["argv"][1]
        upload[-2] = "/frozen/attacker-source-inventory.json"
        upload[-1] = (
            f"{plan['ssh_target']}:"
            f"{plan['remote_run_root']}/inputs/attacker.json"
        )
        _refresh_semantic_bindings(plan, authorization, receipt)

    _assert_prerequisite_mutation_rejected(
        name,
        mutate,
        "command mapping",
    )


@pytest.mark.parametrize(
    "name",
    ("cached_continuation", "engine_correctness"),
)
def test_prerequisites_reject_local_verifier_script_or_arg_substitution(
    name,
):
    def mutate(plan, authorization, receipt, _row):
        plan["commands"]["local_verify"]["argv"][-1] = (
            "/frozen/attacker-authority"
        )
        _refresh_semantic_bindings(plan, authorization, receipt)

    _assert_prerequisite_mutation_rejected(
        name,
        mutate,
        "command mapping",
    )


@pytest.mark.parametrize(
    "name",
    ("cached_continuation", "engine_correctness"),
)
def test_prerequisites_reject_run_root_and_native_path_rebinding(name):
    def mutate(plan, authorization, receipt, _row):
        old_root = plan["remote_run_root"]
        new_root = old_root + "-rebound"
        rebound = _replace_nested_string(plan, old_root, new_root)
        plan.clear()
        plan.update(rebound)
        _refresh_semantic_bindings(plan, authorization, receipt)

    _assert_prerequisite_mutation_rejected(
        name,
        mutate,
        "native layout",
    )


@pytest.mark.parametrize(
    "name",
    ("cached_continuation", "engine_correctness"),
)
def test_prerequisites_reject_source_hash_and_local_input_rebinding(name):
    def mutate(plan, authorization, receipt, row):
        rebound_sha = "9" * 64
        plan["source_tree_sha256"] = rebound_sha
        plan["local_inputs"]["source_tar"] = (
            f"/frozen/{name}-plan/rebound-source.tar"
        )
        prepare = plan["commands"]["prepare_local_verifier"]
        prepare["source_tar"] = plan["local_inputs"]["source_tar"]
        prepare["source_tree_sha256"] = rebound_sha
        prepare["argv"][-4] = plan["local_inputs"]["source_tar"]
        prepare["argv"][-2] = rebound_sha
        authorization["source_tree_sha256"] = rebound_sha
        row["source_tree_sha256"] = rebound_sha
        _refresh_semantic_bindings(plan, authorization, receipt)

    _assert_prerequisite_mutation_rejected(
        name,
        mutate,
        "independent verification",
    )


@pytest.mark.parametrize(
    "name",
    ("cached_continuation", "engine_correctness"),
)
def test_prerequisites_reject_package_local_output_rebinding(name):
    def mutate(plan, authorization, receipt, _row):
        plan["commands"]["package_download"]["local_output"] = (
            f"/frozen/{name}-plan/rebound-package.tar"
        )
        _refresh_semantic_bindings(plan, authorization, receipt)

    _assert_prerequisite_mutation_rejected(
        name,
        mutate,
        "command",
    )


@pytest.mark.parametrize(
    ("name", "module", "run_tag"),
    [
        (
            "engine_correctness",
            native_engine_plan_test,
            "engine-python-compat",
        ),
        (
            "cached_continuation",
            native_cached_plan_test,
            "cached-python-compat",
        ),
    ],
)
def test_workload_command_validator_accepts_shared_versioned_python(
    name,
    module,
    run_tag,
):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo, configuration, inventory = module._fixture(root)
        output = root / "native-plan"
        plan = module.planner.build_remote_execution_plan(
            repo_root=repo,
            configuration_path=configuration,
            source_inventory_path=inventory,
            output_dir=output,
            run_tag=run_tag,
            remote_model_dir="/remote/models/qwen35",
            remote_model_manifest=(
                "/remote/models/qwen35/model_manifest.json"
            ),
        )
        plan = copy.deepcopy(plan)
        for command_name in (
            "safe_extract",
            "prepare_local_verifier",
            "local_verify",
        ):
            plan["commands"][command_name]["argv"][0] = (
                "/opt/python/3.12/bin/python3.12"
            )

        contract._validate_workload_prerequisite_commands(name, plan)


@pytest.mark.parametrize(
    ("name", "module", "run_tag"),
    [
        (
            "engine_correctness",
            native_engine_plan_test,
            "engine-legacy-control-path",
        ),
        (
            "cached_continuation",
            native_cached_plan_test,
            "cached-legacy-control-path",
        ),
    ],
)
def test_workload_command_validator_accepts_only_legacy_control_path_omission(
    name,
    module,
    run_tag,
):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo, configuration, inventory = module._fixture(root)
        output = root / "native-plan"
        plan = module.planner.build_remote_execution_plan(
            repo_root=repo,
            configuration_path=configuration,
            source_inventory_path=inventory,
            output_dir=output,
            run_tag=run_tag,
            remote_model_dir="/remote/models/qwen35",
            remote_model_manifest=(
                "/remote/models/qwen35/model_manifest.json"
            ),
        )
        plan = copy.deepcopy(plan)
        plan["commands"] = module.planner._legacy_commands_without_control_path_none(
            plan["commands"]
        )

        contract._validate_workload_prerequisite_commands(name, plan)


@pytest.mark.parametrize(
    ("interpreters", "reason"),
    [
        (
            (
                "/opt/python/3.12/bin/python3.12",
                "/opt/python/3.12/bin/python3.11",
                "/opt/python/3.12/bin/python3.12",
            ),
            "shared",
        ),
        (("python3", "python3", "python3"), "absolute"),
        (
            (
                "/opt/python/3.12/bin/python3.12",
                "/opt/python/3.12/bin/python3.12",
                "/opt/python/3.12/bin/pypy3",
            ),
            "Python",
        ),
    ],
)
def test_workload_command_validator_rejects_invalid_local_interpreters(
    interpreters,
    reason,
):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo, configuration, inventory = native_engine_plan_test._fixture(
            root
        )
        output = root / "native-plan"
        plan = native_engine_plan_test.planner.build_remote_execution_plan(
            repo_root=repo,
            configuration_path=configuration,
            source_inventory_path=inventory,
            output_dir=output,
            run_tag="engine-python-negative",
            remote_model_dir="/remote/models/qwen35",
            remote_model_manifest=(
                "/remote/models/qwen35/model_manifest.json"
            ),
        )
        plan = copy.deepcopy(plan)
        for command_name, interpreter in zip(
            (
                "safe_extract",
                "prepare_local_verifier",
                "local_verify",
            ),
            interpreters,
        ):
            plan["commands"][command_name]["argv"][0] = interpreter

        with pytest.raises(ValueError, match=reason):
            contract._validate_workload_prerequisite_commands(
                "engine_correctness",
                plan,
            )


@pytest.mark.parametrize(
    ("name", "module", "run_tag"),
    [
        (
            "engine_correctness",
            native_engine_plan_test,
            "engine-controlled-shared",
        ),
        (
            "cached_continuation",
            native_cached_plan_test,
            "cached-controlled-shared",
        ),
    ],
)
def test_workload_command_validator_requires_controlled_shared_upload_row(
    name,
    module,
    run_tag,
):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo, configuration, inventory = module._fixture(root)
        baseline = module._baseline(root)
        output = root / "native-plan"
        plan = module.planner.build_remote_execution_plan(
            repo_root=repo,
            configuration_path=configuration,
            source_inventory_path=inventory,
            output_dir=output,
            run_tag=run_tag,
            remote_model_dir="/remote/models/qwen35",
            remote_model_manifest=(
                "/remote/models/qwen35/model_manifest.json"
            ),
            resource_policy="controlled_shared",
            resource_baseline_path=baseline,
        )
        plan = copy.deepcopy(plan)
        plan["commands"]["upload"]["argv"].pop()

        with pytest.raises(ValueError, match="command"):
            contract._validate_workload_prerequisite_commands(name, plan)


@pytest.mark.parametrize(
    "name",
    ("cached_continuation", "engine_correctness"),
)
def test_prerequisites_reject_semantically_rebound_reserve_command(name):
    def mutate(plan, authorization, receipt, _row):
        command = copy.deepcopy(plan["commands"]["reserve_remote"])
        command["argv"][-1] = "mkdir -p /tmp/attacker-controlled"
        plan["commands"]["reserve_remote"] = command
        _refresh_semantic_bindings(plan, authorization, receipt)

    _assert_prerequisite_mutation_rejected(name, mutate, "command")


@pytest.mark.parametrize(
    "name",
    ("cached_continuation", "engine_correctness"),
)
def test_prerequisites_reject_rebound_guarded_authority_ssh_payload(name):
    def mutate(plan, authorization, receipt, _row):
        guarded_authority = plan["commands"]["guarded_authority"]
        ssh_argv = copy.deepcopy(guarded_authority["ssh_argv"])
        ssh_argv[-1] = shlex.quote(
            shlex.join(["bash", "-lc", "exec python3 -c 'print(1)'"])
        )
        guarded_authority["ssh_argv"] = ssh_argv
        _refresh_semantic_bindings(plan, authorization, receipt)

    _assert_prerequisite_mutation_rejected(name, mutate, "command mapping")


@pytest.mark.parametrize(
    "name",
    ("cached_continuation", "engine_correctness"),
)
def test_prerequisites_reject_rebound_workload_ssh_target(name):
    def mutate(plan, authorization, receipt, _row):
        old = plan["ssh_target"]
        new = "attacker@example.invalid"
        plan["ssh_target"] = new
        for command_name in (
            "reserve_remote",
            "stage",
            "resource_guard",
        ):
            command = plan["commands"][command_name]
            command["argv"] = [
                new if token == old else token
                for token in command["argv"]
            ]
        upload = plan["commands"]["upload"]
        for argv in upload["argv"]:
            argv[:] = [
                new + token[len(old):]
                if token.startswith(old + ":")
                else token
                for token in argv
            ]
        guarded_authority = plan["commands"]["guarded_authority"]
        guarded_authority["ssh_argv"] = [
            new if token == old else token
            for token in guarded_authority["ssh_argv"]
        ]
        package_download = plan["commands"]["package_download"]
        package_download["remote_argv"] = [
            new if token == old else token
            for token in package_download["remote_argv"]
        ]
        _refresh_semantic_bindings(plan, authorization, receipt)

    _assert_prerequisite_mutation_rejected(name, mutate, "native layout")


def test_v2_profiles_thresholds_and_result_vocabulary_are_frozen():
    assert contract.SCHEMA_VERSION == (
        "qwen35.tp4-hybrid-prefix-performance-cache.v2"
    )
    assert contract.PROFILES == (
        "recompute",
        "exact_restore",
        "recurrent_int8_per_row",
    )
    assert contract.LOGIT_TOLERANCE == {"atol": 2e-5, "rtol": 0.0}
    assert contract.THRESHOLDS == {
        "int8_to_exact_unique_physical_bytes_max_ratio": 0.40,
        "int8_to_exact_same_budget_capacity_min_ratio": 2.5,
        "w1_int8_to_exact_median_ttft_max_ratio": 1.03,
        "w1_int8_to_exact_every_ttft_max_ratio": 1.05,
        "w2_int8_to_exact_median_ttft_max_ratio": 1.03,
        "w2_int8_to_exact_every_ttft_max_ratio": 1.05,
        "w3_int8_to_exact_throughput_min_ratio": 0.98,
        "int8_to_exact_peak_cuda_reserved_max_ratio": 1.05,
        "w1_int8_to_recompute_median_ttft_max_ratio": 0.85,
        "w2_int8_to_recompute_median_ttft_max_ratio": 0.75,
        "w3_int8_to_recompute_throughput_min_ratio": 1.15,
        "int8_to_recompute_decode_latency_max_ratio": 1.02,
    }
    assert contract.RESULTS == (
        "GO",
        "NO_GO_CORRECTNESS",
        "NO_GO_RUNTIME_SAFETY",
        "NO_GO_CACHE",
        "NO_GO_PERFORMANCE",
        "BLOCKED_RESOURCES",
        "INVALID_ARTIFACT",
    )


def test_v2_copies_exact_v1_workload_identities_without_importing_v1_state():
    assert contract.WORKLOADS == (
        "w0_short_control",
        "w1_medium_reuse",
        "w2_long_reuse",
        "w3_batched_fanout",
        "w4_miss_invalidation",
    )
    assert contract.WORKLOAD_SPECS == EXPECTED_WORKLOAD_SPECS
    assert contract.WARMUP_REPETITIONS == 1
    assert contract.CORRECTNESS_REPETITIONS == 1
    assert contract.MEASURED_REPETITIONS == 5
    assert contract.MAX_MODEL_LEN == 4096
    assert contract.TOKEN_ID_UPPER_BOUND == 32000

    imports = {
        alias.name
        for node in ast.walk(
            ast.parse(CONTRACT_PATH.read_text(encoding="utf-8"))
        )
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(
            ast.parse(CONTRACT_PATH.read_text(encoding="utf-8"))
        )
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert (
        "qwen35_tp4_hybrid_prefix_benchmark_contract" not in imports
    )


def test_model_vocab_size_matches_tp4_root_logit_contract_literal():
    assert contract.MODEL_VOCAB_SIZE == 248320
    assert (
        _literal_assignment(
            TP4_ROOT_LOGIT_CONTRACT_PATH,
            "MODEL_VOCAB_SIZE",
        )
        == contract.MODEL_VOCAB_SIZE
    )


def test_v2_workload_payloads_freeze_w0_through_w4_token_identities():
    manifest = contract.workload_manifest_payload()

    assert manifest["schema_version"] == contract.SCHEMA_VERSION
    assert tuple(manifest["workloads"]) == contract.WORKLOADS
    for workload_index, workload in enumerate(contract.WORKLOADS):
        payload = manifest["workloads"][workload]
        spec = EXPECTED_WORKLOAD_SPECS[workload]
        assert payload["spec"] == spec
        assert payload["token_seed"] == 2026072900 + workload_index
        assert len(payload["shared_prefix_token_ids"]) == (
            spec["shared_prefix_tokens"]
        )
        assert len(payload["source_suffix_token_ids"]) == (
            spec["suffix_tokens"]
        )
        assert len(payload["continuations"]) == spec["continuations"]
        assert all(
            len(row["suffix_token_ids"]) == spec["suffix_tokens"]
            for row in payload["continuations"]
        )

    w4_rows = manifest["workloads"]["w4_miss_invalidation"][
        "continuations"
    ]
    assert [row["invalidation"]["kind"] for row in w4_rows] == [
        "token_mismatch",
        "stale_block_generation",
        "cache_clear",
    ]
    assert w4_rows[0]["invalidation"]["prefix_index"] == 512
    assert w4_rows[1]["prefix_overrides"] == []
    assert w4_rows[2]["prefix_overrides"] == []


def test_v2_workload_payloads_match_immutable_approved_v1_hashes_and_vectors():
    workloads = {
        workload: contract.workload_payload(workload)
        for workload in contract.WORKLOADS
    }

    assert contract.APPROVED_WORKLOAD_CORPUS_SHA256 == (
        APPROVED_WORKLOAD_CORPUS_SHA256
    )
    assert contract.APPROVED_WORKLOAD_PAYLOAD_SHA256 == (
        APPROVED_WORKLOAD_PAYLOAD_SHA256
    )
    assert contract.canonical_json_sha256(workloads) == (
        APPROVED_WORKLOAD_CORPUS_SHA256
    )
    assert {
        workload: contract.canonical_json_sha256(payload)
        for workload, payload in workloads.items()
    } == APPROVED_WORKLOAD_PAYLOAD_SHA256

    assert workloads["w0_short_control"][
        "shared_prefix_token_ids"
    ][:16] == [
        7469, 10850, 17907, 26800, 23081, 23982, 11599, 17884,
        29669, 8634, 20587, 13000, 19553, 13190, 23879, 24436,
    ]
    assert workloads["w0_short_control"][
        "source_suffix_token_ids"
    ][:16] == [
        2714, 29131, 18600, 24513, 29542, 1959, 28756, 29693,
        9458, 23619, 2752, 10745, 23102, 8863, 18156, 25781,
    ]
    w4 = workloads["w4_miss_invalidation"]
    assert w4["shared_prefix_token_ids"][508:517] == [
        28261, 26170, 12523, 26184, 7905, 11270, 13255, 24308, 8989,
    ]
    assert w4["continuations"][0]["suffix_token_ids"][:16] == [
        9845, 20746, 17531, 31896, 6129, 31958, 16215, 6212,
        1069, 10338, 29427, 6064, 4905, 16302, 21583, 14044,
    ]
    assert w4["continuations"][0]["prefix_overrides"] == [[512, 7906]]
    assert w4["continuations"][0]["invalidation"] == {
        "kind": "token_mismatch",
        "prefix_index": 512,
        "replacement_token_id": 7906,
    }


def test_v2_prerequisite_authority_and_provenance_are_frozen_and_fail_closed():
    assert contract.PREREQUISITE_NAMES == (
        "tp4_root_logit",
        "cached_continuation",
        "engine_correctness",
    )
    assert contract.PREREQUISITE_ROW_FIELDS == (
        "run_tag",
        "source_tree_sha256",
        "artifact_path",
        "artifact_sha256",
        "independent_verification_path",
        "independent_verification_sha256",
        "provenance_path",
        "provenance_sha256",
        "classification",
    )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, payload = _complete_prerequisite_fixture(root)

        result = contract.validate_prerequisites(path)

        assert isinstance(result, contract.PrerequisiteStatus)
        assert result == contract.PrerequisiteStatus(
            classification="PASS",
            authorized=True,
            reasons=(),
        )

        payload["cached_continuation"]["artifact_path"] = "../artifact.json"
        _write_json(path, payload)
        result = contract.validate_prerequisites(path)
        assert result.authorized is False
        assert "unsafe" in " ".join(result.reasons)

    mutations = (
        ("schema_version", "wrong.version", "schema version"),
        ("model_manifest_sha256", "f" * 64, "model manifest"),
    )
    for field, value, expected_reason in mutations:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, payload = _complete_prerequisite_fixture(root)
            payload[field] = value
            _write_json(path, payload)

            result = contract.validate_prerequisites(path)

            assert result.authorized is False
            assert expected_reason in " ".join(result.reasons)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, payload = _complete_prerequisite_fixture(root)
        row = payload["engine_correctness"]
        row["classification"] = "NO_GO"
        row["artifact_sha256"] = "ABC"
        _write_json(path, payload)

        result = contract.validate_prerequisites(path)

        assert result.authorized is False
        reasons = " ".join(result.reasons)
        assert "classification is not PASS" in reasons
        assert "artifact SHA is invalid" in reasons

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, payload = _complete_prerequisite_fixture(root)
        row = payload["tp4_root_logit"]
        provenance_path = root / row["provenance_path"]
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["binding_kind"] = "complete_directory_only"
        provenance["root_logit_receipt_gap"] = True
        _write_json(provenance_path, provenance)
        row["provenance_sha256"] = contract.sha256_file(provenance_path)
        _write_json(path, payload)

        result = contract.validate_prerequisites(path)

        assert result.authorized is False
        assert "receipt provenance" in " ".join(result.reasons)


@pytest.mark.parametrize(
    "filename",
    (
        "execution_plan.json",
        "consumed_authorization.json",
        "execution_receipt.json",
    ),
)
def test_validate_prerequisites_rejects_semantically_opaque_documents(
    filename,
):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, payload = _complete_prerequisite_fixture(root)
        _refresh_prerequisite_evidence(
            root,
            path,
            payload,
            "engine_correctness",
            filename,
            {"arbitrary": "json"},
        )

        with pytest.raises(ValueError):
            _raise_if_prerequisites_are_rejected(path)


@pytest.mark.parametrize(
    "filename",
    (
        "execution_plan.json",
        "consumed_authorization.json",
        "execution_receipt.json",
    ),
)
@pytest.mark.parametrize(
    "content",
    (
        "[]\n",
        "{not-json\n",
    ),
    ids=("non-object", "malformed-json"),
)
def test_validate_prerequisites_rejects_invalid_execution_document_json(
    filename,
    content,
):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, payload = _complete_prerequisite_fixture(root)
        _refresh_raw_prerequisite_evidence(
            root,
            path,
            payload,
            "engine_correctness",
            filename,
            content,
        )

        result = contract.validate_prerequisites(path)

        assert result.authorized is False
        assert "JSON" in " ".join(result.reasons)


def test_validate_prerequisites_accepts_root_and_workload_family_union():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, payload = _complete_prerequisite_fixture(root)
        root_plan, root_authorization, _ = _semantic_bundle(
            root,
            payload,
            "tp4_root_logit",
        )
        cached_plan, cached_authorization, _ = _semantic_bundle(
            root,
            payload,
            "cached_continuation",
        )

        result = contract.validate_prerequisites(path)

        assert result.authorized is True
        assert "workload_manifest_sha256" not in root_plan
        assert "workload_manifest_sha256" not in root_authorization
        assert cached_plan["local_inputs"]["workload_manifest_sha256"] == (
            contract.APPROVED_V1_WORKLOAD_MANIFEST_SHA256
        )
        assert cached_authorization["workload_manifest_sha256"] == (
            contract.APPROVED_V1_WORKLOAD_MANIFEST_SHA256
        )


def test_slice_a_p1_root_plan_literals_match_real_v1_authority_sources():
    runner_path = (
        Path(__file__).resolve().parent
        / "run_qwen35_tp4_real_root_logit_gate_remote.py"
    )
    plan_path = (
        Path(__file__).resolve().parent
        / "qwen35_tp4_root_logit_remote_execution_plan.py"
    )

    assert _literal_assignment(runner_path, "REMOTE_TARGET") == (
        "sitian@10.232.195.203"
    )
    assert _literal_assignment(runner_path, "FROZEN_SOURCE_TAG") == (
        "qwen35-tp4-source-prep-20260729-170818"
    )
    assert _literal_assignment(runner_path, "EXACT_ARTIFACT_NAMES") == {
        "native_rank0_logits.pt",
        "rank_evidence.json",
        "reference_logits.pt",
        "source_manifest.json",
        "tp4_real_root_logit_correctness.json",
    }
    assert _literal_assignment(plan_path, "STAGE_ORDER") == [
        "preflight",
        "run",
        "download",
        "verify",
    ]
    assert _literal_assignment(plan_path, "RESOURCE_BASELINE_NAME") == (
        "resource_baseline.json"
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "ssh_target",
        "exact_artifact_names",
        "frozen_source_tag",
        "remote_run_dir",
        "local_run_dir",
        "stage_inputs",
    ),
)
def test_slice_a_p1_rejects_attacker_selected_root_plan_identity(mutation):
    def mutate(plan, authorization, receipt, _row):
        if mutation == "ssh_target":
            plan["ssh_target"] = "attacker@example.invalid"
            authorization["ssh_target"] = plan["ssh_target"]
        elif mutation == "exact_artifact_names":
            plan["exact_artifact_names"] = ["attacker-output.json"]
            receipt["stages"][1]["result"]["artifact_names"] = (
                plan["exact_artifact_names"]
            )
            receipt["stages"][2]["result"]["artifact_names"] = (
                plan["exact_artifact_names"]
            )
        elif mutation == "frozen_source_tag":
            plan["frozen_source_tag"] = "attacker-source"
        elif mutation == "remote_run_dir":
            plan["remote_run_dir"] = "/tmp/attacker-run"
            plan["stage_inputs"]["run"]["remote_run_dir"] = (
                plan["remote_run_dir"]
            )
            plan["stage_inputs"]["download"]["remote_run_dir"] = (
                plan["remote_run_dir"]
            )
            receipt["stages"][1]["result"]["remote_run_dir"] = (
                plan["remote_run_dir"]
            )
        elif mutation == "local_run_dir":
            plan["local_run_dir"] = "/tmp/attacker-local"
            plan["stage_inputs"]["download"]["local_artifact_dir"] = (
                "/tmp/attacker-local/artifacts"
            )
            plan["stage_inputs"]["verify"]["local_artifact_dir"] = (
                "/tmp/attacker-local/artifacts"
            )
            plan["stage_inputs"]["verify"][
                "independent_verification_path"
            ] = "/tmp/attacker-local/independent_verification.json"
        elif mutation == "stage_inputs":
            plan["stage_inputs"]["preflight"]["repo_root"] = (
                "/tmp/attacker-repo"
            )
        else:
            raise AssertionError(mutation)
        _refresh_semantic_bindings(plan, authorization, receipt)

    _assert_prerequisite_mutation_rejected(
        "tp4_root_logit",
        mutate,
        "execution plan",
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_gpu_index",
        "missing_compute_processes",
        "missing_minimum_free_bytes",
        "missing_frozen_source_tag",
        "missing_rows",
        "unknown_preflight_field",
        "mixed_controlled_shared_field",
        "malformed_raw_process",
        "duplicate_gpu_index",
        "duplicate_gpu_uuid",
        "wrong_rank_order",
        "wrong_world_size",
        "malformed_compute_process",
        "active_compute_process",
        "insufficient_free_bytes",
    ),
)
def test_slice_a_p1_rejects_incomplete_duplicate_or_unsafe_root_gpu_identity(
    mutation,
):
    def mutate(_plan, _authorization, receipt, _row):
        selected = receipt["stages"][0]["result"]["selected"]
        if mutation == "missing_gpu_index":
            del selected[0]["gpu_index"]
        elif mutation == "missing_compute_processes":
            del selected[0]["compute_processes"]
        elif mutation == "missing_minimum_free_bytes":
            del selected[0]["minimum_free_bytes"]
        elif mutation == "missing_frozen_source_tag":
            del receipt["stages"][0]["result"]["frozen_source_tag"]
        elif mutation == "missing_rows":
            del receipt["stages"][0]["result"]["rows"]
        elif mutation == "unknown_preflight_field":
            receipt["stages"][0]["result"]["unexpected"] = True
        elif mutation == "mixed_controlled_shared_field":
            receipt["stages"][0]["result"]["resource_policy"] = (
                "controlled_shared"
            )
        elif mutation == "malformed_raw_process":
            receipt["stages"][0]["result"]["rows"][0][
                "compute_processes"
            ] = [{
                "pid": 1000,
                "process_name": "python3",
                "used_memory_mib": 436,
                "start_time_ticks": 2000,
            }]
        elif mutation == "duplicate_gpu_index":
            selected[1]["gpu_index"] = selected[0]["gpu_index"]
        elif mutation == "duplicate_gpu_uuid":
            selected[1]["gpu_uuid"] = selected[0]["gpu_uuid"]
        elif mutation == "wrong_rank_order":
            selected[0]["rank"], selected[1]["rank"] = (
                selected[1]["rank"],
                selected[0]["rank"],
            )
        elif mutation == "wrong_world_size":
            selected[0]["world_size"] = 8
        elif mutation == "malformed_compute_process":
            selected[0]["compute_processes"] = [{"pid": 1}]
        elif mutation == "active_compute_process":
            selected[0]["compute_processes"] = [{
                "pid": 1000,
                "process_name": "python3",
                "used_memory_mib": 436,
                "start_time_ticks": 2000,
            }]
        elif mutation == "insufficient_free_bytes":
            selected[0]["free_bytes"] = 24 * 1024**3 - 1
        else:
            raise AssertionError(mutation)
        _refresh_semantic_bindings(
            _plan,
            _authorization,
            receipt,
        )

    _assert_prerequisite_mutation_rejected(
        "tp4_root_logit",
        mutate,
        "preflight",
    )


def _controlled_shared_root_bundle(root, payload):
    artifact, verification = _root_authority_payloads()
    plan, authorization, receipt = _root_prerequisite_documents(
        "tp4_root_logit",
        contract.TP4_ROOT_SOURCE_TREE_SHA256,
        artifact,
        verification,
        controlled_shared=True,
    )
    baseline = _write_controlled_shared_baseline(root, plan)
    authorization["resource_baseline_sha256"] = (
        plan["resource_baseline_sha256"]
    )
    receipt["stages"][0]["result"] = (
        _actual_controlled_shared_root_preflight(plan, baseline)
    )
    receipt["stages"][1]["result"]["final_resource"] = {
        "classification": "READY",
        "resource_policy": "controlled_shared",
        "baseline_sha256": plan["resource_baseline_sha256"],
        "selected": copy.deepcopy(baseline["selected"]),
        "benchmark_execution_authorized": False,
    }
    _refresh_semantic_bindings(plan, authorization, receipt)
    path = root / "correctness_prerequisites.json"
    _write_prerequisite_bundle(
        root,
        path,
        payload,
        "tp4_root_logit",
        plan,
        authorization,
        receipt,
    )
    return path, plan, authorization, receipt


def _actual_standard_root_preflight(plan, receipt):
    preflight = receipt["stages"][0]["result"]
    raw_rows = []
    selected = []
    for rank, row in enumerate(preflight["selected"]):
        raw = {
            "gpu_index": row["gpu_index"],
            "gpu_uuid": row["gpu_uuid"],
            "gpu_name": f"GPU model {rank}",
            "total_bytes": 80 * 1024**3,
            "free_bytes": row["free_bytes"],
            "compute_processes": [],
        }
        raw_rows.append(raw)
        selected.append({
            **raw,
            "rank": rank,
            "world_size": 4,
            "minimum_free_bytes": 24 * 1024**3,
        })
    return {
        "run_tag": plan["run_tag"],
        "frozen_source_tag": plan["frozen_source_tag"],
        "frozen_source_tree_sha256": (
            plan["frozen_source_tree_sha256"]
        ),
        "status": "READY",
        "source_tree_sha256": plan["frozen_source_tree_sha256"],
        "selected": selected,
        "rows": raw_rows,
    }


def _write_controlled_shared_baseline(root, plan):
    baseline_path = root / "tp4_root_logit" / "resource_baseline.json"
    baseline = {
        "schema_version": (
            "qwen35.tp4-controlled-shared-resource-baseline.v1"
        ),
        "classification": "READY",
        "ssh_target": plan["ssh_target"],
        "captured_at": "2026-07-29T15:00:00+08:00",
        "gpu_indices": plan["gpu_indices"],
        "selected": [
            {
                "gpu_index": gpu_index,
                "gpu_uuid": gpu_uuid,
                "free_bytes": 64 * 1024**3,
                "compute_processes": [{
                    "pid": 1000 + gpu_index,
                    "process_name": "python3",
                    "used_memory_mib": 436,
                    "start_time_ticks": 2000 + gpu_index,
                }],
            }
            for gpu_index, gpu_uuid in zip(
                plan["gpu_indices"],
                plan["gpu_uuids"],
            )
        ],
        "minimum_free_bytes_per_gpu": 24 * 1024**3,
        "benchmark_execution_authorized": False,
    }
    _write_json(baseline_path, baseline)
    plan["plan_output_dir"] = str(baseline_path.parent)
    plan["resource_baseline_path"] = str(baseline_path)
    plan["resource_baseline_sha256"] = hashlib.sha256(
        baseline_path.read_bytes()
    ).hexdigest()
    for stage in ("preflight", "run"):
        plan["stage_inputs"][stage]["resource_baseline_path"] = (
            plan["resource_baseline_path"]
        )
        plan["stage_inputs"][stage]["resource_baseline_sha256"] = (
            plan["resource_baseline_sha256"]
        )
    return baseline


def _actual_controlled_shared_root_preflight(plan, baseline):
    selected = []
    for rank, row in enumerate(baseline["selected"]):
        selected.append({
            **copy.deepcopy(row),
            "rank": rank,
            "world_size": 4,
            "minimum_free_bytes": 24 * 1024**3,
        })
    return {
        "run_tag": plan["run_tag"],
        "frozen_source_tag": plan["frozen_source_tag"],
        "frozen_source_tree_sha256": (
            plan["frozen_source_tree_sha256"]
        ),
        "status": "READY",
        "source_tree_sha256": plan["frozen_source_tree_sha256"],
        "selected": selected,
        "rows": copy.deepcopy(selected),
        "resource_policy": "controlled_shared",
        "baseline_sha256": plan["resource_baseline_sha256"],
        "benchmark_execution_authorized": False,
    }


def _refresh_root_bundle(
    root,
    path,
    payload,
    plan,
    authorization,
    receipt,
):
    _refresh_semantic_bindings(plan, authorization, receipt)
    _write_prerequisite_bundle(
        root,
        path,
        payload,
        "tp4_root_logit",
        plan,
        authorization,
        receipt,
    )


def test_slice_a_p1_native_v1_accepts_actual_standard_preflight_shape():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, payload = _complete_prerequisite_fixture(root)
        plan, authorization, receipt = (
            copy.deepcopy(document)
            for document in _semantic_bundle(
                root,
                payload,
                "tp4_root_logit",
            )
        )
        preflight = _actual_standard_root_preflight(plan, receipt)
        receipt["stages"][0]["result"] = preflight

        native_root_receipt._validate_preflight(plan, preflight)
        _refresh_root_bundle(
            root,
            path,
            payload,
            plan,
            authorization,
            receipt,
        )

        result = contract.validate_prerequisites(path)

        assert result.authorized is True, result.reasons


def test_slice_a_p1_native_v1_accepts_baseline_process_reappearance():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, payload = _complete_prerequisite_fixture(root)
        path, plan, authorization, receipt = (
            _controlled_shared_root_bundle(root, payload)
        )
        baseline = _write_controlled_shared_baseline(root, plan)
        authorization["resource_baseline_sha256"] = (
            plan["resource_baseline_sha256"]
        )
        preflight = _actual_controlled_shared_root_preflight(
            plan,
            baseline,
        )
        preflight["selected"][0]["compute_processes"] = []
        preflight["rows"][0]["compute_processes"] = []
        receipt["stages"][0]["result"] = preflight
        receipt["stages"][1]["result"]["final_resource"] = {
            "classification": "READY",
            "resource_policy": "controlled_shared",
            "baseline_sha256": plan["resource_baseline_sha256"],
            "selected": copy.deepcopy(baseline["selected"]),
            "benchmark_execution_authorized": False,
        }

        native_root_receipt._validate_plan(plan)
        native_root_receipt._validate_preflight(plan, preflight)
        native_root_receipt._validate_run(
            plan,
            receipt["stages"][1]["result"],
            preflight,
        )
        _refresh_root_bundle(
            root,
            path,
            payload,
            plan,
            authorization,
            receipt,
        )

        result = contract.validate_prerequisites(path)

        assert result.authorized is True, result.reasons


@pytest.mark.parametrize(
    "mutation",
    (
        "preflight_process_not_in_baseline",
        "final_process_not_in_baseline",
        "malformed_final_process",
        "duplicate_final_process",
        "final_gpu_drift",
        "wrong_final_baseline_hash",
        "wrong_final_policy",
        "baseline_file_tamper",
    ),
)
def test_slice_a_p1_controlled_shared_observations_bind_frozen_baseline(
    mutation,
):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, payload = _complete_prerequisite_fixture(root)
        path, plan, authorization, receipt = (
            _controlled_shared_root_bundle(root, payload)
        )
        preflight = receipt["stages"][0]["result"]
        final_resource = receipt["stages"][1]["result"][
            "final_resource"
        ]
        if mutation == "preflight_process_not_in_baseline":
            preflight["selected"][0]["compute_processes"][0][
                "pid"
            ] = 9000
            preflight["rows"][0]["compute_processes"][0]["pid"] = 9000
        elif mutation == "final_process_not_in_baseline":
            final_resource["selected"][0]["compute_processes"][0][
                "pid"
            ] = 9000
        elif mutation == "malformed_final_process":
            del final_resource["selected"][0]["compute_processes"][0][
                "start_time_ticks"
            ]
        elif mutation == "duplicate_final_process":
            final_resource["selected"][0]["compute_processes"].append(
                copy.deepcopy(
                    final_resource["selected"][0][
                        "compute_processes"
                    ][0]
                )
            )
        elif mutation == "final_gpu_drift":
            final_resource["selected"][0]["gpu_uuid"] = "GPU-attacker"
        elif mutation == "wrong_final_baseline_hash":
            final_resource["baseline_sha256"] = "c" * 64
        elif mutation == "wrong_final_policy":
            final_resource["resource_policy"] = "strict_exclusive"
        elif mutation == "baseline_file_tamper":
            baseline_path = Path(plan["resource_baseline_path"])
            baseline = json.loads(
                baseline_path.read_text(encoding="utf-8")
            )
            baseline["captured_at"] = "2026-07-29T15:00:01+08:00"
            _write_json(baseline_path, baseline)
        else:
            raise AssertionError(mutation)
        _refresh_root_bundle(
            root,
            path,
            payload,
            plan,
            authorization,
            receipt,
        )

        result = contract.validate_prerequisites(path)

        assert result.authorized is False


def test_slice_a_p1_accepts_valid_controlled_shared_root_variant():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, payload = _complete_prerequisite_fixture(root)
        path, plan, authorization, receipt = (
            _controlled_shared_root_bundle(root, payload)
        )

        result = contract.validate_prerequisites(path)

        assert result.authorized is True
        assert plan["resource_policy"] == "controlled_shared"
        assert authorization["resource_baseline_sha256"] == (
            plan["resource_baseline_sha256"]
        )
        assert receipt["stages"][1]["result"]["final_resource"][
            "baseline_sha256"
        ] == plan["resource_baseline_sha256"]


def test_slice_a_p1_accepts_source_bound_dynamic_root_source_tag():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, payload = _complete_prerequisite_fixture(root)
        plan, authorization, receipt = (
            copy.deepcopy(document)
            for document in _semantic_bundle(
                root,
                payload,
                "tp4_root_logit",
            )
        )
        source_tag = (
            "qwen35-tp4-source-prep-20260804-160931-"
            "attempt67-qwen35-dual-receipt-schema-retry2-r537"
        )
        plan["frozen_source_tag"] = source_tag
        receipt["stages"][0]["result"]["frozen_source_tag"] = source_tag
        _refresh_semantic_bindings(plan, authorization, receipt)
        _write_prerequisite_bundle(
            root,
            path,
            payload,
            "tp4_root_logit",
            plan,
            authorization,
            receipt,
        )

        result = contract.validate_prerequisites(path)

        assert result.authorized is True, result.reasons


def test_slice_a_p1_accepts_distinct_root_receipt_and_artifact_check_counts():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, payload = _complete_prerequisite_fixture(root)
        plan, authorization, receipt = (
            copy.deepcopy(document)
            for document in _semantic_bundle(
                root,
                payload,
                "tp4_root_logit",
            )
        )
        receipt["stages"][-1]["result"]["checks"] += 225
        _refresh_semantic_bindings(plan, authorization, receipt)
        _write_prerequisite_bundle(
            root,
            path,
            payload,
            "tp4_root_logit",
            plan,
            authorization,
            receipt,
        )

        result = contract.validate_prerequisites(path)

        assert result.authorized is True, result.reasons


def test_slice_a_p1_accepts_controlled_shared_workload_resource_guard():
    gpu_indices = [2, 4, 5, 6]
    baseline_sha256 = "b" * 64
    baseline = {
        "selected": [
            {
                "gpu_index": gpu_index,
                "gpu_uuid": f"GPU-{gpu_index}",
                "free_bytes": 64 * 1024**3,
                "compute_processes": [{
                    "pid": 1000 + gpu_index,
                    "process_name": "python3",
                    "used_memory_mib": 436,
                    "start_time_ticks": 2000 + gpu_index,
                }],
            }
            for gpu_index in gpu_indices
        ],
    }
    payload = {
        "classification": "READY",
        "resource_policy": "controlled_shared",
        "baseline_sha256": baseline_sha256,
        "selected": copy.deepcopy(baseline["selected"]),
        "benchmark_execution_authorized": False,
    }

    identities = contract._validate_resource_receipt(
        payload,
        gpu_indices,
        "controlled shared resource guard",
        resource_policy="controlled_shared",
        baseline=baseline,
        baseline_sha256=baseline_sha256,
    )

    assert identities == tuple(
        (gpu_index, f"GPU-{gpu_index}")
        for gpu_index in gpu_indices
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "partial_plan_variant",
        "partial_authorization_variant",
        "wrong_final_resource_baseline",
        "wrong_final_resource_gpu",
        "wrong_preflight_gpu_order",
    ),
)
def test_slice_a_p1_rejects_mixed_or_unbound_controlled_shared_root_variant(
    mutation,
):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path, payload = _complete_prerequisite_fixture(root)
        path, plan, authorization, receipt = (
            _controlled_shared_root_bundle(root, payload)
        )
        if mutation == "partial_plan_variant":
            del plan["gpu_uuids"]
        elif mutation == "partial_authorization_variant":
            del authorization["resource_baseline_sha256"]
        elif mutation == "wrong_final_resource_baseline":
            receipt["stages"][1]["result"]["final_resource"][
                "baseline_sha256"
            ] = "c" * 64
        elif mutation == "wrong_final_resource_gpu":
            receipt["stages"][1]["result"]["final_resource"]["selected"][0][
                "gpu_uuid"
            ] = "GPU-attacker"
        elif mutation == "wrong_preflight_gpu_order":
            selected = receipt["stages"][0]["result"]["selected"]
            selected[0]["gpu_index"], selected[1]["gpu_index"] = (
                selected[1]["gpu_index"],
                selected[0]["gpu_index"],
            )
        else:
            raise AssertionError(mutation)
        _refresh_semantic_bindings(plan, authorization, receipt)
        _write_prerequisite_bundle(
            root,
            path,
            payload,
            "tp4_root_logit",
            plan,
            authorization,
            receipt,
        )

        result = contract.validate_prerequisites(path)

        assert result.authorized is False


@pytest.mark.parametrize(
    "name,mutation,expected_reason",
    (
        ("tp4_root_logit", "unknown_plan_schema", "execution plan"),
        ("engine_correctness", "unknown_authorization_schema", "authorization"),
        ("cached_continuation", "unknown_receipt_schema", "receipt"),
        ("engine_correctness", "unknown_plan_field", "execution plan"),
        ("engine_correctness", "missing_authorization_field", "authorization"),
        ("tp4_root_logit", "root_fabricated_workload", "workload"),
        ("cached_continuation", "missing_workload", "local inputs"),
        ("engine_correctness", "wrong_workload", "command mapping"),
        ("engine_correctness", "plan_run_tag_mismatch", "native layout"),
        ("cached_continuation", "plan_source_mismatch", "command mapping"),
        ("engine_correctness", "plan_model_mismatch", "model"),
        ("engine_correctness", "authorization_wrong_plan_hash", "authorization"),
        ("engine_correctness", "authorization_wrong_nonce", "authorization"),
        ("engine_correctness", "authorization_not_consumed", "consumed"),
        ("engine_correctness", "authorization_reuse_marker", "authorization"),
        ("engine_correctness", "receipt_not_pass", "receipt"),
        ("engine_correctness", "receipt_failed_command", "returncode"),
        ("engine_correctness", "receipt_wrong_authorization", "authorization"),
        ("engine_correctness", "receipt_wrong_artifact_binding", "artifact"),
        ("engine_correctness", "receipt_wrong_verifier_binding", "verification"),
    ),
)
def test_validate_prerequisites_rejects_semantic_family_mutations(
    name,
    mutation,
    expected_reason,
):
    def mutate(plan, authorization, receipt, row):
        if mutation == "unknown_plan_schema":
            plan["schema_version"] = "unknown.plan.v1"
            _refresh_semantic_bindings(plan, authorization, receipt)
        elif mutation == "unknown_authorization_schema":
            authorization["schema_version"] = "unknown.authorization.v1"
            _refresh_semantic_bindings(plan, authorization, receipt)
        elif mutation == "unknown_receipt_schema":
            receipt["schema_version"] = "unknown.receipt.v1"
        elif mutation == "unknown_plan_field":
            plan["unexpected"] = True
            _refresh_semantic_bindings(plan, authorization, receipt)
        elif mutation == "missing_authorization_field":
            del authorization["ports"]
            _refresh_semantic_bindings(plan, authorization, receipt)
        elif mutation == "root_fabricated_workload":
            plan["workload_manifest_sha256"] = (
                contract.APPROVED_V1_WORKLOAD_MANIFEST_SHA256
            )
            authorization["workload_manifest_sha256"] = (
                contract.APPROVED_V1_WORKLOAD_MANIFEST_SHA256
            )
            _refresh_semantic_bindings(plan, authorization, receipt)
        elif mutation == "missing_workload":
            del plan["local_inputs"]["workload_manifest_sha256"]
            del authorization["workload_manifest_sha256"]
            _refresh_semantic_bindings(plan, authorization, receipt)
        elif mutation == "wrong_workload":
            plan["local_inputs"]["workload_manifest_sha256"] = "9" * 64
            authorization["workload_manifest_sha256"] = "9" * 64
            for step in receipt["steps"]:
                if step["name"] in {"guarded_authority", "local_verify"}:
                    step["stdout"] = step["stdout"].replace(
                        contract.APPROVED_V1_WORKLOAD_MANIFEST_SHA256,
                        "9" * 64,
                    )
            _refresh_semantic_bindings(plan, authorization, receipt)
        elif mutation == "plan_run_tag_mismatch":
            plan["run_tag"] = "different-run-tag"
            authorization["run_tag"] = "different-run-tag"
            receipt["run_tag"] = "different-run-tag"
            _refresh_semantic_bindings(plan, authorization, receipt)
        elif mutation == "plan_source_mismatch":
            plan["source_tree_sha256"] = "9" * 64
            authorization["source_tree_sha256"] = "9" * 64
            for step in receipt["steps"]:
                if step["name"] in {"guarded_authority", "local_verify"}:
                    step["stdout"] = step["stdout"].replace(
                        row["source_tree_sha256"],
                        "9" * 64,
                    )
            _refresh_semantic_bindings(plan, authorization, receipt)
        elif mutation == "plan_model_mismatch":
            plan["model_manifest_sha256"] = "9" * 64
            authorization["model_manifest_sha256"] = "9" * 64
            for step in receipt["steps"]:
                if step["name"] in {"guarded_authority", "local_verify"}:
                    step["stdout"] = step["stdout"].replace(
                        contract.MODEL_MANIFEST_SHA256,
                        "9" * 64,
                    )
            _refresh_semantic_bindings(plan, authorization, receipt)
        elif mutation == "authorization_wrong_plan_hash":
            authorization["plan_sha256"] = "9" * 64
            receipt["authorization_sha256"] = _canonical_sha(authorization)
        elif mutation == "authorization_wrong_nonce":
            authorization["nonce"] = "other-nonce"
            receipt["authorization_sha256"] = _canonical_sha(authorization)
        elif mutation == "authorization_not_consumed":
            authorization["consumed"] = False
            receipt["authorization_sha256"] = _canonical_sha(authorization)
        elif mutation == "authorization_reuse_marker":
            authorization["consumed_once"] = False
            receipt["authorization_sha256"] = _canonical_sha(authorization)
        elif mutation == "receipt_not_pass":
            receipt["classification"] = "FAILED"
        elif mutation == "receipt_failed_command":
            receipt["steps"][0]["returncode"] = 1
        elif mutation == "receipt_wrong_authorization":
            receipt["authorization_sha256"] = "9" * 64
        elif mutation == "receipt_wrong_artifact_binding":
            guarded = next(
                step
                for step in receipt["steps"]
                if step["name"] == "guarded_authority"
            )
            guarded["stdout"] = guarded["stdout"].replace(
                contract.MODEL_MANIFEST_SHA256,
                "9" * 64,
            )
        elif mutation == "receipt_wrong_verifier_binding":
            local = next(
                step
                for step in receipt["steps"]
                if step["name"] == "local_verify"
            )
            local["stdout"] = local["stdout"].replace(
                contract.MODEL_MANIFEST_SHA256,
                "9" * 64,
            )
        else:
            raise AssertionError(f"unknown mutation: {mutation}")

    _assert_prerequisite_mutation_rejected(
        name,
        mutate,
        expected_reason,
    )


def test_triple_profile_order_is_deterministic_and_rotates_by_repetition():
    assert contract.profile_order(0) == (
        "recompute",
        "exact_restore",
        "recurrent_int8_per_row",
    )
    assert contract.profile_order(1) == (
        "exact_restore",
        "recurrent_int8_per_row",
        "recompute",
    )
    assert contract.profile_order(2) == (
        "recurrent_int8_per_row",
        "recompute",
        "exact_restore",
    )
    assert contract.profile_order(3) == contract.profile_order(0)
    assert contract.profile_order(4) == contract.profile_order(1)


def test_case_matrix_has_exact_counts_unique_ids_and_deterministic_order():
    first = contract.build_case_matrix()
    second = contract.build_case_matrix()

    assert first == second
    assert len(first) == 105
    assert len({case.case_id for case in first}) == 105
    assert sum(case.phase == "warmup" for case in first) == 15
    assert sum(case.phase == "correctness" for case in first) == 15
    assert sum(case.phase == "measured" for case in first) == 75
    for workload in contract.WORKLOADS:
        for profile in contract.PROFILES:
            rows = [
                case
                for case in first
                if case.workload == workload
                and case.profile == profile
            ]
            assert len(rows) == 7
            assert sum(case.phase == "warmup" for case in rows) == 1
            assert sum(case.phase == "correctness" for case in rows) == 1
            assert [
                case.repetition
                for case in rows
                if case.phase == "measured"
            ] == [0, 1, 2, 3, 4]


def test_correctness_cases_are_explicitly_strict_and_serial():
    correctness = [
        case
        for case in contract.build_case_matrix()
        if case.phase == "correctness"
    ]

    assert contract.CORRECTNESS_CONCURRENCY == 1
    assert len(correctness) == 15
    assert all(case.concurrency == 1 for case in correctness)
    assert all(case.strict_correctness is True for case in correctness)
    assert [
        (case.workload, case.profile)
        for case in correctness
    ] == [
        (workload, profile)
        for workload in contract.WORKLOADS
        for profile in contract.PROFILES
    ]


def test_sampling_seed_max_tokens_and_effective_concurrency_are_workload_bound():
    matrix = contract.build_case_matrix()
    assert all(
        case.concurrency == 1
        for case in matrix
        if case.phase == "correctness"
    )
    assert all(
        case.concurrency == 8
        for case in matrix
        if case.workload == "w3_batched_fanout"
        and case.phase in {"warmup", "measured"}
    )
    assert all(
        case.concurrency == 1
        for case in matrix
        if case.workload != "w3_batched_fanout"
        or case.phase == "correctness"
    )

    row = _case_row()
    contract.validate_case_row(row)
    for field, value in {
        "sampling_max_tokens": 32,
        "sampling_seed": 2026072900,
        "concurrency": 8,
    }.items():
        mutated = dict(row)
        mutated[field] = value
        with pytest.raises(ValueError, match=field.replace("_", " ")):
            contract.validate_case_row(mutated)

    process = _process_row()
    contract.validate_process_row(process)
    for field, value in {
        "sampling_max_tokens": 32,
        "sampling_seed": 2026072900,
        "concurrency": 8,
    }.items():
        mutated = dict(process)
        mutated[field] = value
        with pytest.raises(ValueError, match=field.replace("_", " ")):
            contract.validate_process_row(mutated)


def test_p2_identity_version_calibration_and_p1_authority_are_bound():
    runtime_version = _literal_assignment(
        RUNTIME_REPRESENTATION_PATH,
        "QWEN35_HYBRID_PREFIX_INT8_VERSION",
    )
    assert runtime_version == "qwen35_hybrid_prefix_recurrent_int8_v1"
    assert contract.P2_REPRESENTATION_VERSION == runtime_version
    assert contract.P2_PROFILE == "recurrent_int8_per_row"
    assert contract.P2_REPRESENTATION == "recurrent_int8_per_row"
    assert contract.P2_REPRESENTATION_VERSION == (
        "qwen35_hybrid_prefix_recurrent_int8_v1"
    )
    assert contract.P2_CODEC_ID == (
        "qwen35_recurrent_symmetric_int8_per_row_v1"
    )
    assert contract.P1_REFERENCE_PROFILE == "exact_restore"
    assert contract.P2_NUMERICAL_REFERENCE_PROFILE == "exact_restore"
    assert contract.P2_CACHE_COST_REFERENCE_PROFILE == "exact_restore"
    assert contract.CALIBRATION_SCHEMA_VERSION == (
        "qwen35.recurrent-int8-calibration.v1"
    )
    assert contract.P1_AUTHORITY_SCHEMA_VERSION == (
        "qwen35.tp4-hybrid-prefix-performance-cache.v1"
    )
    assert contract.P2_REQUIRED_BINDINGS == (
        "codec",
        "representation",
        "representation_version",
        "calibration_artifact_sha256",
        "p1_authority_artifact_sha256",
    )


def test_top_level_artifact_inventory_is_exact_and_closed():
    assert contract.TOP_LEVEL_ARTIFACTS == (
        "correctness_prerequisites.json",
        "calibration_binding.json",
        "p1_authority_binding.json",
        "workload_manifest.json",
        "source_manifest.json",
        "environment.json",
        "gpu_assignments.json",
        "commands.json",
        "gate1_audit.json",
        "preflight.json",
        "execution_plan.json",
        "consumed_authorization.json",
        "source_bundle_manifest.json",
        "source_package_manifest.json",
        "resource_guards.json",
        "snapshot_manifest.json",
        "tensor_inventory_manifest.json",
        "case_rows.jsonl",
        "process_rows.jsonl",
        "token_manifest.json",
        "logits_manifest.json",
        "worker_logs_manifest.json",
        "summary.json",
        "artifact_manifest.json",
        "local_verifier_output.json",
        "remote_verifier_output.json",
        "independent_verification.json",
        "report.md",
    )
    assert contract.NESTED_ARTIFACT_DIRECTORIES == (
        "prerequisites",
        "snapshots",
        "receipts",
        "source",
        "tokens",
        "logits",
        "logs",
        "verifier",
    )
    assert contract.ARTIFACT_MANIFEST_HASH_DOMAIN == (
        "correctness_prerequisites.json",
        "calibration_binding.json",
        "p1_authority_binding.json",
        "workload_manifest.json",
        "source_manifest.json",
        "environment.json",
        "gpu_assignments.json",
        "commands.json",
        "gate1_audit.json",
        "preflight.json",
        "execution_plan.json",
        "consumed_authorization.json",
        "source_bundle_manifest.json",
        "source_package_manifest.json",
        "resource_guards.json",
        "snapshot_manifest.json",
        "tensor_inventory_manifest.json",
        "case_rows.jsonl",
        "process_rows.jsonl",
        "token_manifest.json",
        "logits_manifest.json",
        "worker_logs_manifest.json",
        "summary.json",
    )
    assert contract.PRODUCER_TRUST_DOMAIN == (
        "artifact_manifest.json",
        *contract.ARTIFACT_MANIFEST_HASH_DOMAIN,
    )
    assert contract.VERIFIER_TRUST_DOMAIN == (
        "local_verifier_output.json",
        "remote_verifier_output.json",
        "independent_verification.json",
        "report.md",
    )
    assert contract.MANIFEST_ENTRY_FIELDS == (
        "path",
        "sha256",
        "bytes",
        "producer",
        "trust_domain",
    )
    assert contract.SNAPSHOT_INVENTORY_FIELDS == (
        "schema_version",
        "case_id",
        "profile",
        "representation",
        "representation_version",
        "codec",
        "rank",
        "world_size",
        "snapshot_path",
        "snapshot_sha256",
        "tensor_inventory_path",
        "tensor_inventory_sha256",
        "full_fidelity_logical_bytes",
        "encoded_physical_bytes",
        "codec_metadata_bytes",
        "temporary_encode_workspace_bytes",
        "temporary_decode_workspace_bytes",
    )
    assert contract.RECEIPT_BINDING_FIELDS == (
        "schema_version",
        "run_tag",
        "nonce",
        "artifact_path",
        "gate1_audit_sha256",
        "preflight_sha256",
        "execution_plan_sha256",
        "consumed_authorization_sha256",
        "source_bundle_sha256",
        "source_package_sha256",
        "resource_guards_sha256",
    )


def test_raw_case_row_schema_is_exact_closed_and_carries_token_logit_paths():
    assert contract.CASE_ROW_FIELDS == EXPECTED_CASE_ROW_FIELDS
    assert "execution_receipt_sha256" not in contract.CASE_ROW_FIELDS
    assert "execution_receipt_sha256" not in (
        contract.RECEIPT_BINDING_FIELDS
    )
    assert {
        "prompt_token_ids_path",
        "prompt_token_ids_sha256",
        "output_token_ids_path",
        "output_token_ids_sha256",
        "final_logits_path",
        "final_logits_sha256",
        "final_logits_shape",
        "final_logits_dtype",
    } <= set(contract.CASE_ROW_FIELDS)
    contract.validate_case_row(_case_row())
    _assert_rejects_unknown_field(
        contract.validate_case_row,
        _case_row(),
    )
    with pytest.raises(ValueError):
        contract.validate_case_row({
            field: None for field in contract.CASE_ROW_FIELDS
        })

    mutations = {
        "profile": "wrong-profile",
        "representation": "exact_restore",
        "codec": "wrong-codec",
        "workload": "wrong-workload",
        "phase": "wrong-phase",
        "repetition": True,
        "source_tree_sha256": "A" * 64,
        "prompt_token_ids_path": "../prompt.json",
        "prompt_tokens": -1,
        "reused_kv_tokens": True,
        "restored_hybrid_state": 1,
        "executed_prefill_tokens": -1,
        "generated_tokens": 63,
        "ttft_ns": float("inf"),
        "e2e_ns": -1,
        "decode_step_ns": float("nan"),
        "final_logits_shape": [0],
        "final_logits_dtype": "float16",
    }
    for field, value in mutations.items():
        row = _case_row()
        row[field] = value
        with pytest.raises(ValueError, match=field.replace("_", " ")):
            contract.validate_case_row(row)

    row = _case_row()
    row["executed_prefill_tokens"] = 65
    with pytest.raises(ValueError, match="token accounting"):
        contract.validate_case_row(row)

    row = _case_row()
    row["e2e_ns"] = 0
    with pytest.raises(ValueError, match="timing"):
        contract.validate_case_row(row)


def test_validate_case_rows_accepts_frozen_ordered_canonical_collection():
    rows = _canonical_case_rows()
    request_cardinality = {
        "w0_short_control": 1,
        "w1_medium_reuse": 4,
        "w2_long_reuse": 4,
        "w3_batched_fanout": 8,
        "w4_miss_invalidation": 3,
    }
    ordered_case_ids = []
    rows_by_case = {}
    for row in rows:
        case_id = row["case_id"]
        if case_id not in rows_by_case:
            ordered_case_ids.append(case_id)
            rows_by_case[case_id] = []
        rows_by_case[case_id].append(row)

    assert len(ordered_case_ids) == 105
    assert len(rows) == 420
    for case_id in ordered_case_ids:
        case_rows = rows_by_case[case_id]
        workload = case_rows[0]["workload"]
        cardinality = request_cardinality[workload]
        assert [row["request_id"] for row in case_rows] == [
            f"request-{request_index}"
            for request_index in range(cardinality)
        ]
        assert [row["row_id"] for row in case_rows] == [
            f"{case_id}__request-{request_index}"
            for request_index in range(cardinality)
        ]

    frozen_case_identities = {
        0: "w0_short_control__warmup__r0__recompute",
        3: "w0_short_control__correctness__r0__recompute",
        6: "w0_short_control__measured__r0__recompute",
        21: "w1_medium_reuse__warmup__r0__recompute",
        24: "w1_medium_reuse__correctness__r0__recompute",
        27: "w1_medium_reuse__measured__r0__recompute",
        42: "w2_long_reuse__warmup__r0__recompute",
        45: "w2_long_reuse__correctness__r0__recompute",
        48: "w2_long_reuse__measured__r0__recompute",
        63: "w3_batched_fanout__warmup__r0__recompute",
        66: "w3_batched_fanout__correctness__r0__recompute",
        69: "w3_batched_fanout__measured__r0__recompute",
        84: "w4_miss_invalidation__warmup__r0__recompute",
        87: "w4_miss_invalidation__correctness__r0__recompute",
        90: "w4_miss_invalidation__measured__r0__recompute",
        104: "w4_miss_invalidation__measured__r4__recompute",
    }
    assert {
        index: ordered_case_ids[index]
        for index in frozen_case_identities
    } == frozen_case_identities

    semantic_fields = (
        "row_id",
        "case_id",
        "request_id",
        "profile",
        "workload",
        "phase",
        "repetition",
        "prompt_tokens",
        "reused_kv_tokens",
        "restored_hybrid_state",
        "executed_prefill_tokens",
        "generated_tokens",
    )
    representative_rows = {
        0: (
            "w0_short_control__warmup__r0__recompute__request-0",
            "w0_short_control__warmup__r0__recompute",
            "request-0",
            "recompute",
            "w0_short_control",
            "warmup",
            0,
            288,
            0,
            False,
            288,
            32,
        ),
        1: (
            "w0_short_control__warmup__r0__exact_restore__request-0",
            "w0_short_control__warmup__r0__exact_restore",
            "request-0",
            "exact_restore",
            "w0_short_control",
            "warmup",
            0,
            288,
            256,
            True,
            32,
            32,
        ),
        360: (
            "w4_miss_invalidation__warmup__r0__exact_restore__request-0",
            "w4_miss_invalidation__warmup__r0__exact_restore",
            "request-0",
            "exact_restore",
            "w4_miss_invalidation",
            "warmup",
            0,
            1088,
            0,
            False,
            1088,
            32,
        ),
        419: (
            "w4_miss_invalidation__measured__r4__recompute__request-2",
            "w4_miss_invalidation__measured__r4__recompute",
            "request-2",
            "recompute",
            "w4_miss_invalidation",
            "measured",
            4,
            1088,
            0,
            False,
            1088,
            32,
        ),
    }
    assert {
        index: tuple(rows[index][field] for field in semantic_fields)
        for index in representative_rows
    } == representative_rows

    contract.validate_case_rows(rows)


def test_validate_case_rows_rejects_case_id_workload_mismatch():
    rows = _canonical_case_rows()
    rows[0]["workload"] = "w1_medium_reuse"
    rows[0]["sampling_max_tokens"] = 64
    rows[0]["sampling_seed"] = contract.workload_sampling_seed(
        "w1_medium_reuse"
    )
    rows[0]["prompt_tokens"] = 1088
    rows[0]["reused_kv_tokens"] = 0
    rows[0]["executed_prefill_tokens"] = 1088
    rows[0]["generated_tokens"] = 64

    with pytest.raises(ValueError, match="canonical case row"):
        contract.validate_case_rows(rows)


def test_validate_case_rows_rejects_unknown_request_id():
    rows = _canonical_case_rows()
    rows[0]["request_id"] = "request-999"

    with pytest.raises(ValueError, match="canonical case row"):
        contract.validate_case_rows(rows)


def test_validate_case_rows_rejects_recompute_restored_state():
    rows = _canonical_case_rows()
    rows[0]["restored_hybrid_state"] = True

    with pytest.raises(ValueError, match="canonical case row"):
        contract.validate_case_rows(rows)


def test_validate_case_rows_rejects_wrong_prompt_token_accounting():
    rows = _canonical_case_rows()
    rows[0]["prompt_tokens"] += 1
    rows[0]["executed_prefill_tokens"] += 1

    with pytest.raises(ValueError, match="canonical case row"):
        contract.validate_case_rows(rows)


def test_validate_case_rows_rejects_duplicate_canonical_row():
    rows = _canonical_case_rows()
    rows.insert(1, dict(rows[0]))

    with pytest.raises(ValueError, match="canonical case row"):
        contract.validate_case_rows(rows)


def test_validate_case_rows_rejects_missing_canonical_row():
    rows = _canonical_case_rows()
    rows.pop()

    with pytest.raises(ValueError, match="canonical case row"):
        contract.validate_case_rows(rows)


def test_validate_case_rows_rejects_reordered_canonical_rows():
    rows = _canonical_case_rows()
    rows[0], rows[1] = rows[1], rows[0]

    with pytest.raises(ValueError, match="canonical case row"):
        contract.validate_case_rows(rows)


def test_raw_process_row_schema_is_exact_closed_and_rank_complete():
    assert contract.PROCESS_ROW_FIELDS == EXPECTED_PROCESS_ROW_FIELDS
    assert {
        "rank",
        "world_size",
        "hybrid_cache_current_unique_physical_bytes",
        "hybrid_cache_current_logical_referenced_bytes",
        "hybrid_cache_current_metadata_bytes",
        "encode_workspace_peak_allocated_bytes",
        "encode_workspace_peak_reserved_bytes",
        "decode_workspace_peak_allocated_bytes",
        "decode_workspace_peak_reserved_bytes",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "same_budget_entry_capacity",
        "oom_events",
        "undeclared_eviction_events",
        "hybrid_cache_corruption_events",
        "hybrid_cache_fallbacks",
        "hybrid_cache_partial_restore_attempts",
        "hybrid_cache_mixed_representation_events",
        "hybrid_cache_missing_layer_events",
        "hybrid_cache_failed_rollbacks",
    } <= set(contract.PROCESS_ROW_FIELDS)
    contract.validate_process_row(_process_row())
    _assert_rejects_unknown_field(
        contract.validate_process_row,
        _process_row(),
    )
    with pytest.raises(ValueError):
        contract.validate_process_row({
            field: None for field in contract.PROCESS_ROW_FIELDS
        })

    mutations = {
        "profile": "wrong-profile",
        "representation": "exact_restore",
        "codec": "wrong-codec",
        "workload": "wrong-workload",
        "phase": "wrong-phase",
        "repetition": True,
        "rank": 4,
        "world_size": 3,
        "pid": True,
        "hostname": "",
        "master_port": 0,
        "tinyvllm_dist_port": 65536,
        "artifact_path": "../run",
        "initialization_ns": float("inf"),
        "cuda_allocated_bytes": -1,
        "kv_capacity_bytes": 0,
        "same_budget_entry_capacity": 0,
        "hybrid_cache_hits": True,
        "oom_events": -1,
    }
    for field, value in mutations.items():
        row = _process_row()
        row[field] = value
        with pytest.raises(ValueError, match=field.replace("_", " ")):
            contract.validate_process_row(row)

    row = _process_row()
    row["cuda_allocated_bytes"] = 2
    row["cuda_peak_allocated_bytes"] = 1
    with pytest.raises(ValueError, match="CUDA memory"):
        contract.validate_process_row(row)

    row = _process_row()
    row["hybrid_cache_current_unique_physical_bytes"] = 2
    row["hybrid_cache_current_logical_referenced_bytes"] = 1
    row["hybrid_cache_peak_unique_physical_bytes"] = 2
    row["hybrid_cache_peak_logical_referenced_bytes"] = 1
    row["hybrid_cache_deduplicated_bytes"] = 0
    contract.validate_process_row(row)


def test_validate_process_rows_accepts_frozen_ordered_canonical_collection():
    rows = _canonical_process_rows()
    identities = (
        "case_id",
        "workload",
        "profile",
        "phase",
        "repetition",
        "rank",
        "cuda_visible_device",
    )

    assert len(contract.build_case_matrix()) == 105
    assert len(rows) == 420
    assert all(
        [row["rank"] for row in rows[offset:offset + 4]]
        == [0, 1, 2, 3]
        for offset in range(0, len(rows), 4)
    )
    assert {
        index: tuple(rows[index][field] for field in identities)
        for index in (0, 3, 12, 24, 84, 208, 336, 416, 419)
    } == {
        0: (
            "w0_short_control__warmup__r0__recompute",
            "w0_short_control",
            "recompute",
            "warmup",
            0,
            0,
            "0",
        ),
        3: (
            "w0_short_control__warmup__r0__recompute",
            "w0_short_control",
            "recompute",
            "warmup",
            0,
            3,
            "3",
        ),
        12: (
            "w0_short_control__correctness__r0__recompute",
            "w0_short_control",
            "recompute",
            "correctness",
            0,
            0,
            "0",
        ),
        24: (
            "w0_short_control__measured__r0__recompute",
            "w0_short_control",
            "recompute",
            "measured",
            0,
            0,
            "0",
        ),
        84: (
            "w1_medium_reuse__warmup__r0__recompute",
            "w1_medium_reuse",
            "recompute",
            "warmup",
            0,
            0,
            "0",
        ),
        208: (
            "w2_long_reuse__measured__r1__recurrent_int8_per_row",
            "w2_long_reuse",
            "recurrent_int8_per_row",
            "measured",
            1,
            0,
            "0",
        ),
        336: (
            "w4_miss_invalidation__warmup__r0__recompute",
            "w4_miss_invalidation",
            "recompute",
            "warmup",
            0,
            0,
            "0",
        ),
        416: (
            "w4_miss_invalidation__measured__r4__recompute",
            "w4_miss_invalidation",
            "recompute",
            "measured",
            4,
            0,
            "0",
        ),
        419: (
            "w4_miss_invalidation__measured__r4__recompute",
            "w4_miss_invalidation",
            "recompute",
            "measured",
            4,
            3,
            "3",
        ),
    }
    assert [
        rows[rank]["gpu_indices"][rows[rank]["rank"]]
        for rank in range(4)
    ] == [2, 4, 5, 6]
    for row in rows:
        contract.validate_process_row(row)

    contract.validate_process_rows(rows)


def test_validate_process_rows_rejects_case_id_workload_mismatch():
    rows = _canonical_process_rows()
    rows[0]["case_id"] = rows[84]["case_id"]
    contract.validate_process_row(rows[0])

    with pytest.raises(ValueError, match="canonical process row"):
        contract.validate_process_rows(rows)


def test_validate_process_rows_rejects_phase_profile_mismatch():
    rows = _canonical_process_rows()
    rows[12]["case_id"] = rows[16]["case_id"]
    contract.validate_process_row(rows[12])

    with pytest.raises(ValueError, match="canonical process row"):
        contract.validate_process_rows(rows)


def test_validate_process_rows_rejects_rank_specific_row_identity_mismatch():
    rows = _canonical_process_rows()
    rows[0]["rank"], rows[1]["rank"] = rows[1]["rank"], rows[0]["rank"]
    rows[0]["cuda_visible_device"] = str(rows[0]["rank"])
    rows[1]["cuda_visible_device"] = str(rows[1]["rank"])
    contract.validate_process_row(rows[0])
    contract.validate_process_row(rows[1])

    with pytest.raises(ValueError, match="canonical process row"):
        contract.validate_process_rows(rows)


def test_validate_process_rows_rejects_duplicate_canonical_row():
    rows = _canonical_process_rows()
    rows.insert(1, dict(rows[0]))

    with pytest.raises(ValueError, match="canonical process row"):
        contract.validate_process_rows(rows)


def test_validate_process_rows_rejects_missing_canonical_row():
    rows = _canonical_process_rows()
    rows.pop()

    with pytest.raises(ValueError, match="canonical process row"):
        contract.validate_process_rows(rows)


def test_validate_process_rows_rejects_reordered_canonical_rows():
    rows = _canonical_process_rows()
    rows[0], rows[1] = rows[1], rows[0]

    with pytest.raises(ValueError, match="canonical process row"):
        contract.validate_process_rows(rows)


def test_binding_schemas_are_closed_and_reject_unknown_fields():
    assert contract.CALIBRATION_BINDING_FIELDS == (
        "schema_version",
        "codec",
        "representation",
        "representation_version",
        "source_tree_sha256",
        "model_manifest_sha256",
        "workload_manifest_sha256",
        "artifact_path",
        "artifact_sha256",
        "classification",
    )
    assert contract.P1_AUTHORITY_BINDING_FIELDS == (
        "schema_version",
        "profile",
        "representation",
        "source_tree_sha256",
        "model_manifest_sha256",
        "workload_manifest_sha256",
        "artifact_path",
        "artifact_sha256",
        "independent_verification_path",
        "independent_verification_sha256",
        "classification",
    )
    calibration = _calibration_binding()
    p1_authority = _p1_authority_binding()
    contract.validate_calibration_binding(calibration)
    contract.validate_p1_authority_binding(p1_authority)
    _assert_rejects_unknown_field(
        contract.validate_calibration_binding,
        calibration,
    )
    _assert_rejects_unknown_field(
        contract.validate_p1_authority_binding,
        p1_authority,
    )

    for field, value in {
        "schema_version": "wrong.version",
        "codec": "wrong-codec",
        "representation": "wrong-representation",
        "representation_version": "wrong-version",
        "source_tree_sha256": "A" * 64,
        "artifact_path": "../calibration.json",
        "classification": "NO_GO",
    }.items():
        binding = _calibration_binding()
        binding[field] = value
        with pytest.raises(ValueError, match=field.replace("_", " ")):
            contract.validate_calibration_binding(binding)

    for field, value in {
        "schema_version": "wrong.version",
        "profile": "recompute",
        "representation": "recompute",
        "artifact_sha256": "ABC",
        "independent_verification_path": "../verification.json",
        "classification": "PASS",
    }.items():
        binding = _p1_authority_binding()
        binding[field] = value
        with pytest.raises(ValueError, match=field.replace("_", " ")):
            contract.validate_p1_authority_binding(binding)


def test_snapshot_manifest_and_receipt_schemas_are_semantic_and_closed():
    assert contract.SNAPSHOT_INVENTORY_SCHEMA_VERSION == (
        "qwen35.tp4-hybrid-prefix-snapshot-inventory.v1"
    )
    assert contract.RECEIPT_BINDING_SCHEMA_VERSION == (
        "qwen35.tp4-hybrid-prefix-remote-receipt-binding.v1"
    )
    contract.validate_snapshot_inventory(_snapshot_inventory())
    contract.validate_manifest_entry(_manifest_entry())
    contract.validate_receipt_binding(_receipt_binding())
    producer_receipt_fields = set(
        contract.EVIDENCE_DOCUMENT_FIELDS["execution_receipt"]
    )
    assert "local_verifier_sha256" not in producer_receipt_fields
    assert "remote_verifier_sha256" not in producer_receipt_fields
    assert "local_verifier_sha256" not in contract.RECEIPT_BINDING_FIELDS
    assert "remote_verifier_sha256" not in contract.RECEIPT_BINDING_FIELDS
    assert "execution_receipt_sha256" not in (
        contract.RECEIPT_BINDING_FIELDS
    )
    assert "artifact_manifest.json" not in (
        contract.ARTIFACT_MANIFEST_HASH_DOMAIN
    )
    final_fields = contract.EVIDENCE_DOCUMENT_FIELDS[
        "independent_verification"
    ]
    assert final_fields == (
        "schema_version",
        "classification",
        "artifact_manifest_sha256",
        "local_verifier_sha256",
        "remote_verifier_sha256",
        "local_verifier_role",
        "remote_verifier_role",
        "checks",
    )
    final_verification = _closed_evidence_document(
        "independent_verification"
    )
    contract.validate_evidence_document(
        "independent_verification",
        final_verification,
    )

    for field, value in {
        "schema_version": "wrong.version",
        "representation_version": "wrong-version",
        "rank": 4,
        "world_size": 3,
        "snapshot_path": "../snapshot.bin",
        "snapshot_sha256": "A" * 64,
        "encoded_physical_bytes": -1,
        "temporary_decode_workspace_bytes": True,
    }.items():
        inventory = _snapshot_inventory()
        inventory[field] = value
        with pytest.raises(ValueError, match=field.replace("_", " ")):
            contract.validate_snapshot_inventory(inventory)

    entry_mutations = {
        "path": "../snapshot.bin",
        "sha256": "ABC",
        "bytes": True,
        "producer": "",
        "trust_domain": "verifier",
    }
    for field, value in entry_mutations.items():
        entry = _manifest_entry()
        entry[field] = value
        with pytest.raises(ValueError, match=field.replace("_", " ")):
            contract.validate_manifest_entry(entry)

    for field, value in {
        "schema_version": "wrong.version",
        "run_tag": "",
        "artifact_path": "/absolute/run",
        "gate1_audit_sha256": "ABC",
        "resource_guards_sha256": None,
    }.items():
        receipt = _receipt_binding()
        receipt[field] = value
        with pytest.raises(ValueError, match=field.replace("_", " ")):
            contract.validate_receipt_binding(receipt)


def test_matched_configuration_and_provenance_identities_are_bound_cross_file():
    assert contract.SAMPLING_TEMPERATURE == 0.0
    assert contract.SAMPLING_IGNORE_EOS is True
    assert contract.HYBRID_PREFIX_MAX_ENTRIES == 16
    assert contract.HYBRID_PREFIX_MAX_BYTES == 2 * 1024**3
    assert contract.REQUIRED_GPU_INDICES == (2, 4, 5, 6)
    assert contract.DIRTY_TREE_POLICIES == ("reject_dirty",)
    assert contract.SOURCE_MANIFEST_FIELDS == (
        "schema_version",
        "source_tree_sha256",
        "dirty_tree_policy",
        "dirty_tree",
        "gate1_audit_sha256",
        "execution_plan_sha256",
        "source_bundle_sha256",
        "source_package_sha256",
        "producer_source_sha256",
        "producer_version_sha256",
        "verifier_source_sha256",
        "verifier_version_sha256",
    )
    assert contract.MATCHED_CONFIGURATION_FIELDS == (
        "schema_version",
        "model_manifest_sha256",
        "tokenizer_manifest_sha256",
        "workload_manifest_sha256",
        "sampling_temperature",
        "sampling_max_tokens",
        "sampling_ignore_eos",
        "sampling_seed",
        "concurrency",
        "tp_world_size",
        "gpu_indices",
        "kv_capacity_bytes",
        "hybrid_prefix_max_entries",
        "hybrid_prefix_max_bytes",
    )

    source = _source_manifest()
    configuration = _matched_configuration()
    case = _case_row()
    process = _process_row()
    contract.validate_source_manifest(source)
    contract.validate_matched_configuration(configuration)
    contract.validate_row_bindings(
        case,
        process,
        source,
        configuration,
    )

    for field, value in {
        "dirty_tree_policy": "allow_dirty",
        "dirty_tree": True,
        "producer_version_sha256": "ABC",
        "source_bundle_sha256": None,
    }.items():
        mutated = _source_manifest()
        mutated[field] = value
        with pytest.raises(ValueError, match=field.replace("_", " ")):
            contract.validate_source_manifest(mutated)

    for field, value in {
        "sampling_temperature": 0.1,
        "sampling_max_tokens": True,
        "sampling_ignore_eos": 1,
        "sampling_seed": -1,
        "concurrency": 0,
        "tp_world_size": 3,
        "gpu_indices": [2, 4, 5, 5],
        "kv_capacity_bytes": 0,
        "hybrid_prefix_max_entries": 0,
        "hybrid_prefix_max_bytes": -1,
    }.items():
        mutated = _matched_configuration()
        mutated[field] = value
        with pytest.raises(ValueError, match=field.replace("_", " ")):
            contract.validate_matched_configuration(mutated)

    drift_cases = (
        ("case", "sampling_seed", 1, "sampling seed"),
        (
            "case",
            "representation_version",
            "wrong-version",
            "representation version",
        ),
        (
            "case",
            "gate1_audit_sha256",
            "f" * 64,
            "binding mismatch",
        ),
        ("process", "gpu_indices", [2, 4, 5, 7], "gpu indices"),
        (
            "process",
            "source_package_sha256",
            "f" * 64,
            "binding mismatch",
        ),
        ("process", "kv_capacity_bytes", 2, "binding mismatch"),
    )
    for target, field, value, expected_error in drift_cases:
        case = _case_row()
        process = _process_row()
        row = case if target == "case" else process
        row[field] = value
        with pytest.raises(ValueError, match=expected_error):
            contract.validate_row_bindings(
                case,
                process,
                _source_manifest(),
                _matched_configuration(),
            )


def test_all_named_evidence_documents_and_artifact_manifest_are_closed():
    assert tuple(contract.EVIDENCE_DOCUMENT_FIELDS) == (
        "environment",
        "gpu_assignments",
        "commands",
        "gate1_audit",
        "preflight",
        "execution_plan",
        "consumed_authorization",
        "execution_receipt",
        "source_bundle",
        "source_package",
        "resource_guard",
        "verifier_output",
        "independent_verification",
    )
    bundle = _execution_evidence_bundle()
    fixture_by_kind = {
        "environment": bundle["environment"],
        "gpu_assignments": bundle["gpu_assignments"],
        "commands": bundle["commands"],
        "gate1_audit": _closed_evidence_document("gate1_audit"),
        "preflight": bundle["preflight"],
        "execution_plan": bundle["execution_plan"],
        "consumed_authorization": bundle["consumed_authorization"],
        "execution_receipt": bundle["execution_receipt"],
        "source_bundle": bundle["source_bundle"],
        "source_package": bundle["source_package"],
        "resource_guard": bundle["resource_guard_before"],
        "verifier_output": bundle["local_verifier_output"],
        "independent_verification": bundle[
            "independent_verification"
        ],
    }
    for kind in contract.EVIDENCE_DOCUMENT_FIELDS:
        document = fixture_by_kind[kind]
        contract.validate_evidence_document(kind, document)

        unknown = dict(document)
        unknown["unexpected"] = True
        with pytest.raises(ValueError, match="fields"):
            contract.validate_evidence_document(kind, unknown)

        wrong_version = dict(document)
        wrong_version["schema_version"] = "wrong.version"
        with pytest.raises(ValueError, match="schema version"):
            contract.validate_evidence_document(kind, wrong_version)

    manifest = {
        "schema_version": contract.ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "hash_domain": list(contract.ARTIFACT_MANIFEST_HASH_DOMAIN),
        "entries": [
            {
                "path": path,
                "sha256": f"{index:064x}",
                "bytes": index + 1,
                "producer": "task8-assembler",
                "trust_domain": "producer",
            }
            for index, path in enumerate(
                contract.ARTIFACT_MANIFEST_HASH_DOMAIN
            )
        ],
        "excluded_verifier_outputs": list(
            contract.VERIFIER_TRUST_DOMAIN
        ),
    }
    contract.validate_artifact_manifest(manifest)

    receipt_entry = {
        "path": "execution_receipt.json",
        "sha256": "f" * 64,
        "bytes": 1,
        "producer": "task9-executor",
        "trust_domain": "producer",
    }
    tampered = copy.deepcopy(manifest)
    tampered["hash_domain"].append("execution_receipt.json")
    tampered["entries"].append(receipt_entry)
    with pytest.raises(ValueError, match="hash domain|entries"):
        contract.validate_artifact_manifest(tampered)

    tampered = dict(manifest)
    tampered["hash_domain"] = tampered["hash_domain"][:-1]
    with pytest.raises(ValueError, match="hash domain"):
        contract.validate_artifact_manifest(tampered)

    tampered = dict(manifest)
    tampered["entries"] = list(tampered["entries"])
    tampered["entries"][0] = {
        **tampered["entries"][0],
        "trust_domain": "verifier",
    }
    with pytest.raises(ValueError, match="trust domain"):
        contract.validate_artifact_manifest(tampered)


def test_execution_evidence_bundle_is_closed_cross_bound_and_role_exact():
    bundle = _execution_evidence_bundle()
    contract.validate_execution_evidence_bundle(bundle)

    mutations = (
        ("gpu_assignments", "world_size", 3, "world size"),
        ("execution_plan", "nonce", "other", "nonce"),
        (
            "consumed_authorization",
            "authorization_id",
            "other",
            "authorization",
        ),
        (
            "execution_receipt",
            "package_inventory_sha256",
            "f" * 64,
            "package inventory",
        ),
        (
            "local_verifier_output",
            "role",
            "remote",
            "local verifier role",
        ),
    )
    for document_name, field, value, message in mutations:
        malformed = json.loads(json.dumps(bundle))
        malformed[document_name][field] = value
        with pytest.raises(ValueError, match=message):
            contract.validate_execution_evidence_bundle(malformed)

    malformed = json.loads(json.dumps(bundle))
    malformed["execution_plan"]["case_port_pairs"][1][
        "master_port"
    ] = malformed["execution_plan"]["case_port_pairs"][0][
        "tinyvllm_dist_port"
    ]
    with pytest.raises(ValueError, match="port"):
        contract.validate_execution_evidence_bundle(malformed)

    malformed = json.loads(json.dumps(bundle))
    malformed["execution_receipt"]["command_results"].reverse()
    with pytest.raises(ValueError, match="command order"):
        contract.validate_execution_evidence_bundle(malformed)

    malformed = json.loads(json.dumps(bundle))
    malformed["resource_guard_after"]["side_effects_observed"] = True
    with pytest.raises(ValueError, match="side effect"):
        contract.validate_execution_evidence_bundle(malformed)


def test_preflight_blocked_bundle_is_closed_and_has_no_execution_documents():
    assert contract.EXECUTION_LIFECYCLE_STATES == (
        "preflight_blocked",
        "execution_success",
        "execution_failed",
    )
    assert contract.EXECUTION_BUNDLE_DOCUMENTS["preflight_blocked"] == (
        "lifecycle_state",
        "environment",
        "gpu_assignments",
        "preflight",
    )
    bundle = _blocked_execution_evidence_bundle()
    contract.validate_execution_evidence_bundle(bundle)

    malformed = json.loads(json.dumps(bundle))
    malformed["execution_plan"] = _execution_evidence_bundle()[
        "execution_plan"
    ]
    with pytest.raises(ValueError, match="execution evidence bundle"):
        contract.validate_execution_evidence_bundle(malformed)

    malformed = json.loads(json.dumps(bundle))
    malformed["preflight"]["worker_authorized"] = True
    with pytest.raises(ValueError, match="worker authorization"):
        contract.validate_execution_evidence_bundle(malformed)

    for field in (
        "remote_path_created",
        "source_staged",
        "worker_launched",
    ):
        malformed = json.loads(json.dumps(bundle))
        malformed["preflight"][field] = True
        with pytest.raises(ValueError, match="side effect"):
            contract.validate_execution_evidence_bundle(malformed)


def test_failed_execution_bundle_tracks_attempted_and_skipped_commands():
    bundle = _failed_execution_evidence_bundle()
    contract.validate_execution_evidence_bundle(bundle)

    for command_name, expected_side_effects in (
        (
            "reserve_remote",
            {
                "remote_path_created": False,
                "source_staged": False,
                "worker_launched": False,
            },
        ),
        (
            "stage",
            {
                "remote_path_created": True,
                "source_staged": False,
                "worker_launched": False,
            },
        ),
    ):
        failed_at_stage = _failed_execution_evidence_bundle()
        command_results = failed_at_stage["execution_receipt"][
            "command_results"
        ]
        failure_index = contract.EXECUTION_COMMAND_ORDER.index(
            command_name
        )
        for index, row in enumerate(command_results):
            if index < failure_index:
                row["outcome"] = "attempted"
                row["returncode"] = 0
                row["stdout"] = "ok"
                row["stderr"] = ""
            elif index == failure_index:
                row["outcome"] = "attempted"
                row["returncode"] = 17
                row["stdout"] = ""
                row["stderr"] = f"{command_name} failed"
            else:
                row["outcome"] = "skipped"
                row["returncode"] = None
                row["stdout"] = ""
                row["stderr"] = ""
        for field, value in expected_side_effects.items():
            failed_at_stage["execution_receipt"][field] = value
        failed_at_stage["execution_receipt"]["cleanup_complete"] = any(
            expected_side_effects.values()
        )
        failed_at_stage["execution_receipt"][
            "resource_guard_before_sha256"
        ] = None
        failed_at_stage.pop("resource_guard_before")
        contract.validate_execution_evidence_bundle(failed_at_stage)

    malformed = json.loads(json.dumps(bundle))
    malformed["execution_receipt"]["command_results"][-1][
        "outcome"
    ] = "attempted"
    malformed["execution_receipt"]["command_results"][-1][
        "returncode"
    ] = 0
    with pytest.raises(ValueError, match="skipped|failure"):
        contract.validate_execution_evidence_bundle(malformed)

    malformed = json.loads(json.dumps(bundle))
    failed_row = malformed["execution_receipt"]["command_results"][
        contract.EXECUTION_COMMAND_ORDER.index("workers")
    ]
    failed_row["returncode"] = 0
    with pytest.raises(ValueError, match="nonzero|failure"):
        contract.validate_execution_evidence_bundle(malformed)

    malformed = json.loads(json.dumps(bundle))
    malformed["execution_receipt"]["classification"] = "GO"
    with pytest.raises(ValueError, match="classification"):
        contract.validate_execution_evidence_bundle(malformed)

    malformed = json.loads(json.dumps(bundle))
    malformed["execution_receipt"]["worker_launched"] = True
    malformed["execution_receipt"]["source_staged"] = False
    with pytest.raises(ValueError, match="source staged|worker launched"):
        contract.validate_execution_evidence_bundle(malformed)


def test_reserve_remote_failure_accepts_truthful_pre_guard_evidence_bundle():
    bundle = _failed_execution_evidence_bundle()
    command_results = bundle["execution_receipt"]["command_results"]
    for index, row in enumerate(command_results):
        if index == 0:
            row["outcome"] = "attempted"
            row["returncode"] = 17
            row["stdout"] = ""
            row["stderr"] = "reserve_remote failed"
        else:
            row["outcome"] = "skipped"
            row["returncode"] = None
            row["stdout"] = ""
            row["stderr"] = ""
    bundle["execution_receipt"]["remote_path_created"] = False
    bundle["execution_receipt"]["source_staged"] = False
    bundle["execution_receipt"]["worker_launched"] = False
    bundle["execution_receipt"]["cleanup_complete"] = False
    bundle["execution_receipt"]["resource_guard_before_sha256"] = None
    bundle.pop("resource_guard_before")
    bundle.pop("resource_guard_after", None)

    contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    ("failed_command", "expected_side_effects"),
    (
        (
            "upload",
            {
                "remote_path_created": True,
                "source_staged": False,
                "worker_launched": False,
            },
        ),
        (
            "stage",
            {
                "remote_path_created": True,
                "source_staged": False,
                "worker_launched": False,
            },
        ),
    ),
)
def test_pre_guard_failure_accepts_truthful_no_guard_evidence_bundle(
    failed_command,
    expected_side_effects,
):
    bundle = _failed_execution_evidence_bundle()
    command_results = bundle["execution_receipt"]["command_results"]
    failure_index = contract.EXECUTION_COMMAND_ORDER.index(failed_command)
    for index, row in enumerate(command_results):
        if index < failure_index:
            row["outcome"] = "attempted"
            row["returncode"] = 0
            row["stdout"] = "ok"
            row["stderr"] = ""
        elif index == failure_index:
            row["outcome"] = "attempted"
            row["returncode"] = 17
            row["stdout"] = ""
            row["stderr"] = f"{failed_command} failed"
        else:
            row["outcome"] = "skipped"
            row["returncode"] = None
            row["stdout"] = ""
            row["stderr"] = ""
    for field, value in expected_side_effects.items():
        bundle["execution_receipt"][field] = value
    bundle["execution_receipt"]["resource_guard_before_sha256"] = None
    bundle.pop("resource_guard_before")
    bundle.pop("resource_guard_after", None)

    contract.validate_execution_evidence_bundle(bundle)


def test_workers_failure_rejects_no_guard_shape_relaxation():
    bundle = _failed_execution_evidence_bundle()
    bundle.pop("resource_guard_before")

    with pytest.raises(ValueError, match="resource guard|required"):
        contract.validate_execution_evidence_bundle(bundle)


def test_resource_guard_failure_accepts_no_guard_documents():
    bundle = _execution_failure_at("resource_guard")
    bundle.pop("resource_guard_before")
    bundle.pop("resource_guard_after")

    contract.validate_execution_evidence_bundle(bundle)


def test_workers_failure_accepts_only_before_guard_document():
    bundle = _execution_failure_at("workers")
    bundle.pop("resource_guard_after")

    contract.validate_execution_evidence_bundle(bundle)


def test_workers_failure_rejects_missing_before_guard_document():
    bundle = _execution_failure_at("workers")
    bundle.pop("resource_guard_before")

    with pytest.raises(
        ValueError,
        match="resource_guard_before|resource guard|required",
    ):
        contract.validate_execution_evidence_bundle(bundle)


def test_resource_guard_failure_rejects_fabricated_guard_document():
    bundle = _execution_failure_at("resource_guard")
    bundle.pop("resource_guard_after")

    with pytest.raises(
        ValueError,
        match="resource_guard_after|resource guard|required",
    ):
        contract.validate_execution_evidence_bundle(bundle)


def test_resource_guard_failure_requires_null_unproduced_guard_hashes():
    bundle = _execution_failure_at("resource_guard")
    bundle.pop("resource_guard_before")
    bundle.pop("resource_guard_after")
    receipt = bundle["execution_receipt"]
    receipt["resource_guard_before_sha256"] = None
    receipt["resource_guard_after_sha256"] = None

    contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "field",
    (
        "resource_guard_before_sha256",
        "resource_guard_after_sha256",
    ),
)
def test_resource_guard_failure_rejects_fabricated_guard_hash(field):
    bundle = _execution_failure_at("resource_guard")
    bundle.pop("resource_guard_before")
    bundle.pop("resource_guard_after")
    receipt = bundle["execution_receipt"]
    receipt["resource_guard_before_sha256"] = None
    receipt["resource_guard_after_sha256"] = None
    receipt[field] = "f" * 64

    with pytest.raises(ValueError, match=field.replace("_", " ")):
        contract.validate_execution_evidence_bundle(bundle)


def test_workers_failure_requires_matching_before_and_null_after_hash():
    bundle = _execution_failure_at("workers")
    bundle.pop("resource_guard_after")
    receipt = bundle["execution_receipt"]
    receipt["resource_guard_after_sha256"] = None

    contract.validate_execution_evidence_bundle(bundle)


def test_workers_failure_rejects_fabricated_after_guard_hash():
    bundle = _execution_failure_at("workers")
    bundle.pop("resource_guard_after")
    bundle["execution_receipt"]["resource_guard_after_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="resource guard after sha256"):
        contract.validate_execution_evidence_bundle(bundle)


def test_assembly_failure_accepts_truthful_before_guard_and_side_effects():
    bundle = _execution_failure_at("assembly")
    bundle.pop("resource_guard_after")
    receipt = bundle["execution_receipt"]
    receipt["resource_guard_after_sha256"] = None

    assert receipt["remote_path_created"] is True
    assert receipt["source_staged"] is True
    assert receipt["worker_launched"] is True
    contract.validate_execution_evidence_bundle(bundle)


def test_assembly_failure_rejects_missing_before_guard_document():
    bundle = _execution_failure_at("assembly")
    bundle.pop("resource_guard_before")
    bundle.pop("resource_guard_after")
    bundle["execution_receipt"]["resource_guard_after_sha256"] = None

    with pytest.raises(
        ValueError,
        match="resource_guard_before|resource guard|required",
    ):
        contract.validate_execution_evidence_bundle(bundle)


def test_assembly_failure_rejects_fabricated_after_guard_document():
    bundle = _execution_failure_at("assembly")
    bundle["execution_receipt"]["resource_guard_after_sha256"] = None

    with pytest.raises(
        ValueError,
        match="resource_guard_after|resource guard|required",
    ):
        contract.validate_execution_evidence_bundle(bundle)


def test_assembly_failure_rejects_fabricated_after_guard_hash():
    bundle = _execution_failure_at("assembly")
    bundle.pop("resource_guard_after")
    bundle["execution_receipt"]["resource_guard_after_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="resource guard after sha256"):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize("field", ("source_staged", "worker_launched"))
def test_assembly_failure_rejects_false_completed_side_effect_flag(field):
    bundle = _execution_failure_at("assembly")
    bundle.pop("resource_guard_after")
    receipt = bundle["execution_receipt"]
    receipt["resource_guard_after_sha256"] = None
    receipt[field] = False

    with pytest.raises(
        ValueError,
        match="remote path created|source staged|worker launched",
    ):
        contract.validate_execution_evidence_bundle(bundle)


def _remote_verify_failure_evidence_bundle():
    bundle = _execution_failure_at("remote_verify")
    bundle.pop("resource_guard_after")
    receipt = bundle["execution_receipt"]
    receipt["resource_guard_after_sha256"] = None
    return bundle


def test_remote_verify_failure_accepts_truthful_pre_verify_evidence():
    bundle = _remote_verify_failure_evidence_bundle()
    receipt = bundle["execution_receipt"]
    failure_index = contract.EXECUTION_COMMAND_ORDER.index("remote_verify")

    assert all(
        row["outcome"] == "attempted" and row["returncode"] == 0
        for row in receipt["command_results"][:failure_index]
    )
    assert receipt["command_results"][failure_index]["returncode"] != 0
    assert all(
        row["outcome"] == "skipped"
        for row in receipt["command_results"][failure_index + 1 :]
    )
    assert bundle["resource_guard_before"]["sha256"] == receipt[
        "resource_guard_before_sha256"
    ]
    assert "resource_guard_after" not in bundle
    assert receipt["resource_guard_after_sha256"] is None
    assert "remote_verifier_output" not in bundle
    assert "local_verifier_output" not in bundle
    assert "independent_verification" not in bundle
    assert receipt["remote_path_created"] is True
    assert receipt["source_staged"] is True
    assert receipt["worker_launched"] is True

    contract.validate_execution_evidence_bundle(bundle)


def test_remote_verify_failure_rejects_fabricated_verifier_output():
    bundle = _remote_verify_failure_evidence_bundle()
    bundle["remote_verifier_output"] = _execution_evidence_bundle()[
        "remote_verifier_output"
    ]

    with pytest.raises(ValueError, match="execution evidence bundle"):
        contract.validate_execution_evidence_bundle(bundle)


def test_remote_verify_failure_rejects_missing_before_guard():
    bundle = _remote_verify_failure_evidence_bundle()
    bundle.pop("resource_guard_before")

    with pytest.raises(
        ValueError,
        match="resource_guard_before|resource guard|required",
    ):
        contract.validate_execution_evidence_bundle(bundle)


def test_remote_verify_failure_rejects_fabricated_after_guard_document():
    bundle = _remote_verify_failure_evidence_bundle()
    bundle["resource_guard_after"] = _execution_evidence_bundle()[
        "resource_guard_after"
    ]

    with pytest.raises(
        ValueError,
        match="resource_guard_after|resource guard|required",
    ):
        contract.validate_execution_evidence_bundle(bundle)


def test_remote_verify_failure_rejects_fabricated_after_guard_hash():
    bundle = _remote_verify_failure_evidence_bundle()
    bundle["execution_receipt"]["resource_guard_after_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="resource guard after sha256"):
        contract.validate_execution_evidence_bundle(bundle)


def test_remote_verify_failure_rejects_incomplete_success_prefix():
    bundle = _remote_verify_failure_evidence_bundle()
    assembly = bundle["execution_receipt"]["command_results"][
        contract.EXECUTION_COMMAND_ORDER.index("assembly")
    ]
    assembly["outcome"] = "skipped"
    assembly["returncode"] = None
    assembly["stdout"] = ""
    assembly["stderr"] = ""

    with pytest.raises(ValueError, match="attempted command ordering"):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "field",
    ("remote_path_created", "source_staged", "worker_launched"),
)
def test_remote_verify_failure_rejects_false_prior_side_effect_flag(field):
    bundle = _remote_verify_failure_evidence_bundle()
    bundle["execution_receipt"][field] = False

    with pytest.raises(ValueError, match=field.replace("_", " ")):
        contract.validate_execution_evidence_bundle(bundle)


def _final_resource_guard_failure_evidence_bundle():
    bundle = _execution_failure_at("final_resource_guard")
    bundle.pop("resource_guard_after")
    bundle["execution_receipt"]["resource_guard_after_sha256"] = None
    bundle["remote_verifier_output"] = _execution_evidence_bundle()[
        "remote_verifier_output"
    ]
    return bundle


def test_final_resource_guard_failure_accepts_truthful_remote_verifier():
    bundle = _final_resource_guard_failure_evidence_bundle()
    receipt = bundle["execution_receipt"]
    failure_index = contract.EXECUTION_COMMAND_ORDER.index(
        "final_resource_guard"
    )

    assert all(
        row["outcome"] == "attempted" and row["returncode"] == 0
        for row in receipt["command_results"][:failure_index]
    )
    assert receipt["command_results"][failure_index]["returncode"] != 0
    assert all(
        row["outcome"] == "skipped"
        for row in receipt["command_results"][failure_index + 1 :]
    )
    assert bundle["resource_guard_before"]["sha256"] == receipt[
        "resource_guard_before_sha256"
    ]
    assert "resource_guard_after" not in bundle
    assert receipt["resource_guard_after_sha256"] is None
    assert bundle["remote_verifier_output"]["role"] == "remote"
    assert "local_verifier_output" not in bundle
    assert "independent_verification" not in bundle

    contract.validate_execution_evidence_bundle(bundle)


def test_final_resource_guard_failure_rejects_missing_remote_output():
    bundle = _final_resource_guard_failure_evidence_bundle()
    bundle.pop("remote_verifier_output")

    with pytest.raises(
        ValueError,
        match="remote_verifier_output|remote verifier|required",
    ):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("role", "local"),
        ("artifact_manifest_sha256", "0" * 64),
        ("verifier_source_sha256", "0" * 64),
        ("verifier_version_sha256", "0" * 64),
    ),
)
def test_final_resource_guard_failure_rejects_fabricated_remote_output(
    field,
    value,
):
    bundle = _final_resource_guard_failure_evidence_bundle()
    bundle["remote_verifier_output"][field] = value

    with pytest.raises(ValueError):
        contract.validate_execution_evidence_bundle(bundle)


def test_final_resource_guard_failure_rejects_after_guard_document():
    bundle = _final_resource_guard_failure_evidence_bundle()
    bundle["resource_guard_after"] = _execution_evidence_bundle()[
        "resource_guard_after"
    ]

    with pytest.raises(
        ValueError,
        match="resource_guard_after|resource guard|required",
    ):
        contract.validate_execution_evidence_bundle(bundle)


def test_final_resource_guard_failure_rejects_after_guard_hash():
    bundle = _final_resource_guard_failure_evidence_bundle()
    bundle["execution_receipt"]["resource_guard_after_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="resource guard after sha256"):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "document",
    ("local_verifier_output", "independent_verification"),
)
def test_final_resource_guard_failure_rejects_later_verifier_document(
    document,
):
    bundle = _final_resource_guard_failure_evidence_bundle()
    bundle[document] = _execution_evidence_bundle()[document]

    with pytest.raises(ValueError, match="execution evidence bundle"):
        contract.validate_execution_evidence_bundle(bundle)


def test_final_resource_guard_failure_rejects_incomplete_success_prefix():
    bundle = _final_resource_guard_failure_evidence_bundle()
    remote_verify = bundle["execution_receipt"]["command_results"][
        contract.EXECUTION_COMMAND_ORDER.index("remote_verify")
    ]
    remote_verify["outcome"] = "skipped"
    remote_verify["returncode"] = None
    remote_verify["stdout"] = ""
    remote_verify["stderr"] = ""

    with pytest.raises(ValueError, match="attempted command ordering"):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "field",
    ("remote_path_created", "source_staged", "worker_launched"),
)
def test_final_resource_guard_failure_rejects_false_prior_side_effect_flag(
    field,
):
    bundle = _final_resource_guard_failure_evidence_bundle()
    bundle["execution_receipt"][field] = False

    with pytest.raises(ValueError, match=field.replace("_", " ")):
        contract.validate_execution_evidence_bundle(bundle)


def _post_guard_failure_evidence_bundle(failed_command):
    bundle = _execution_failure_at(failed_command)
    bundle["execution_receipt"]["resource_guard_after_sha256"] = bundle[
        "resource_guard_after"
    ]["sha256"]
    bundle["remote_verifier_output"] = _execution_evidence_bundle()[
        "remote_verifier_output"
    ]
    return bundle


@pytest.mark.parametrize(
    "failed_command",
    ("package_download", "safe_extract"),
)
def test_post_guard_failure_accepts_truthful_remote_verified_evidence(
    failed_command,
):
    bundle = _post_guard_failure_evidence_bundle(failed_command)
    receipt = bundle["execution_receipt"]
    failure_index = contract.EXECUTION_COMMAND_ORDER.index(failed_command)

    assert all(
        row["outcome"] == "attempted" and row["returncode"] == 0
        for row in receipt["command_results"][:failure_index]
    )
    assert receipt["command_results"][failure_index]["returncode"] != 0
    assert all(
        row["outcome"] == "skipped"
        for row in receipt["command_results"][failure_index + 1 :]
    )
    assert bundle["resource_guard_before"]["sha256"] == receipt[
        "resource_guard_before_sha256"
    ]
    assert bundle["resource_guard_after"]["sha256"] == receipt[
        "resource_guard_after_sha256"
    ]
    assert bundle["remote_verifier_output"]["role"] == "remote"
    assert "local_verifier_output" not in bundle
    assert "independent_verification" not in bundle
    assert receipt["remote_path_created"] is True
    assert receipt["source_staged"] is True
    assert receipt["worker_launched"] is True

    contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "failed_command",
    ("package_download", "safe_extract"),
)
def test_post_guard_failure_rejects_missing_remote_output(failed_command):
    bundle = _post_guard_failure_evidence_bundle(failed_command)
    bundle.pop("remote_verifier_output")

    with pytest.raises(
        ValueError,
        match="remote_verifier_output|remote verifier|required",
    ):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "failed_command",
    ("package_download", "safe_extract"),
)
def test_post_guard_failure_rejects_missing_after_guard(failed_command):
    bundle = _post_guard_failure_evidence_bundle(failed_command)
    bundle.pop("resource_guard_after")

    with pytest.raises(
        ValueError,
        match="resource_guard_after|resource guard|required",
    ):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "failed_command",
    ("package_download", "safe_extract"),
)
def test_post_guard_failure_rejects_missing_before_guard(failed_command):
    bundle = _post_guard_failure_evidence_bundle(failed_command)
    bundle.pop("resource_guard_before")

    with pytest.raises(
        ValueError,
        match="resource_guard_before|resource guard|required",
    ):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "failed_command",
    ("package_download", "safe_extract"),
)
def test_post_guard_failure_rejects_mismatching_before_guard_hash(
    failed_command,
):
    bundle = _post_guard_failure_evidence_bundle(failed_command)
    bundle["execution_receipt"]["resource_guard_before_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="resource guard before sha256"):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "failed_command",
    ("package_download", "safe_extract"),
)
def test_post_guard_failure_rejects_mismatching_after_guard_hash(
    failed_command,
):
    bundle = _post_guard_failure_evidence_bundle(failed_command)
    bundle["execution_receipt"]["resource_guard_after_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="resource guard after sha256"):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    ("failed_command", "document"),
    (
        ("package_download", "local_verifier_output"),
        ("package_download", "independent_verification"),
        ("safe_extract", "local_verifier_output"),
        ("safe_extract", "independent_verification"),
    ),
)
def test_post_guard_failure_rejects_later_verifier_document(
    failed_command,
    document,
):
    bundle = _post_guard_failure_evidence_bundle(failed_command)
    bundle[document] = _execution_evidence_bundle()[document]

    with pytest.raises(ValueError, match="execution evidence bundle"):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "failed_command",
    ("package_download", "safe_extract"),
)
def test_post_guard_failure_rejects_incomplete_success_prefix(
    failed_command,
):
    bundle = _post_guard_failure_evidence_bundle(failed_command)
    final_guard = bundle["execution_receipt"]["command_results"][
        contract.EXECUTION_COMMAND_ORDER.index("final_resource_guard")
    ]
    final_guard["outcome"] = "skipped"
    final_guard["returncode"] = None
    final_guard["stdout"] = ""
    final_guard["stderr"] = ""

    with pytest.raises(ValueError, match="attempted command ordering"):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    ("failed_command", "field"),
    (
        ("package_download", "remote_path_created"),
        ("package_download", "source_staged"),
        ("package_download", "worker_launched"),
        ("safe_extract", "remote_path_created"),
        ("safe_extract", "source_staged"),
        ("safe_extract", "worker_launched"),
    ),
)
def test_post_guard_failure_rejects_false_prior_side_effect_flag(
    failed_command,
    field,
):
    bundle = _post_guard_failure_evidence_bundle(failed_command)
    bundle["execution_receipt"][field] = False

    with pytest.raises(ValueError, match=field.replace("_", " ")):
        contract.validate_execution_evidence_bundle(bundle)


def _local_verify_failure_evidence_bundle():
    bundle = _post_guard_failure_evidence_bundle("local_verify")
    bundle["execution_receipt"]["package_inventory"] = [
        row
        for row in bundle["execution_receipt"]["package_inventory"]
        if row["path"] != "artifact_manifest.json"
    ]
    bundle["execution_receipt"][
        "package_inventory_sha256"
    ] = contract.canonical_json_sha256(
        bundle["execution_receipt"]["package_inventory"]
    )
    return bundle


def _stage_failure_evidence_bundle(failed_command):
    failure_index = contract.EXECUTION_COMMAND_ORDER.index(failed_command)
    if failed_command == "final_resource_guard":
        return _final_resource_guard_failure_evidence_bundle()
    if failed_command in {
        "package_download",
        "safe_extract",
    }:
        return _post_guard_failure_evidence_bundle(failed_command)
    if failed_command == "local_verify":
        return _local_verify_failure_evidence_bundle()

    bundle = _execution_failure_at(failed_command)
    if failure_index <= contract.EXECUTION_COMMAND_ORDER.index(
        "resource_guard"
    ):
        bundle.pop("resource_guard_before")
        bundle.pop("resource_guard_after")
        bundle["execution_receipt"]["resource_guard_before_sha256"] = None
        bundle["execution_receipt"]["resource_guard_after_sha256"] = None
    else:
        bundle.pop("resource_guard_after")
        bundle["execution_receipt"]["resource_guard_after_sha256"] = None
    return bundle


def _set_success_classification(bundle, classification):
    bundle["local_verifier_output"]["classification"] = classification
    bundle["remote_verifier_output"]["classification"] = classification
    bundle["independent_verification"]["classification"] = classification
    bundle["independent_verification"][
        "local_verifier_sha256"
    ] = contract.canonical_json_sha256(bundle["local_verifier_output"])
    bundle["independent_verification"][
        "remote_verifier_sha256"
    ] = contract.canonical_json_sha256(bundle["remote_verifier_output"])


def _set_failure_remote_classification(bundle, classification):
    bundle["remote_verifier_output"]["classification"] = classification


def _set_success_verifier_artifact_manifest_sha256(bundle, sha256):
    bundle["local_verifier_output"]["artifact_manifest_sha256"] = sha256
    bundle["remote_verifier_output"]["artifact_manifest_sha256"] = sha256
    bundle["independent_verification"]["artifact_manifest_sha256"] = sha256
    bundle["independent_verification"][
        "local_verifier_sha256"
    ] = contract.canonical_json_sha256(bundle["local_verifier_output"])
    bundle["independent_verification"][
        "remote_verifier_sha256"
    ] = contract.canonical_json_sha256(bundle["remote_verifier_output"])


def _refresh_final_inventory_sha256(bundle):
    bundle["execution_receipt"][
        "final_inventory_sha256"
    ] = contract.canonical_json_sha256(
        bundle["execution_receipt"]["final_inventory"]
    )


def _refresh_package_inventory_sha256(bundle):
    bundle["execution_receipt"][
        "package_inventory_sha256"
    ] = contract.canonical_json_sha256(
        bundle["execution_receipt"]["package_inventory"]
    )


def _truthful_inventory_failure_bundle(failed_command):
    bundle = _stage_failure_evidence_bundle(failed_command)
    receipt = bundle["execution_receipt"]
    success_receipt = _execution_evidence_bundle()["execution_receipt"]
    failure_index = contract.EXECUTION_COMMAND_ORDER.index(failed_command)
    assembly_index = contract.EXECUTION_COMMAND_ORDER.index("assembly")
    safe_extract_index = contract.EXECUTION_COMMAND_ORDER.index(
        "safe_extract"
    )

    receipt["final_inventory"] = (
        []
        if failure_index <= safe_extract_index
        else json.loads(
            json.dumps(success_receipt["final_inventory"])
        )
    )
    receipt["package_inventory"] = (
        []
        if failure_index <= assembly_index
        else json.loads(
            json.dumps(
                success_receipt[
                    "package_inventory"
                    if failed_command == "local_verify"
                    else "final_inventory"
                ]
            )
        )
    )
    _refresh_package_inventory_sha256(bundle)
    _refresh_final_inventory_sha256(bundle)
    return bundle


@pytest.mark.parametrize(
    "failed_command",
    contract.EXECUTION_COMMAND_ORDER,
)
def test_execution_failure_accepts_truthful_stage_inventory_shape(
    failed_command,
):
    bundle = _truthful_inventory_failure_bundle(failed_command)
    receipt = bundle["execution_receipt"]
    failure_index = contract.EXECUTION_COMMAND_ORDER.index(failed_command)

    assert bool(receipt["package_inventory"]) is (failure_index >= 6)
    assert bool(receipt["final_inventory"]) is (failure_index == 10)
    contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "failed_command",
    contract.EXECUTION_COMMAND_ORDER[:6],
)
def test_execution_failure_rejects_unproduced_package_inventory(
    failed_command,
):
    bundle = _truthful_inventory_failure_bundle(failed_command)
    bundle["execution_receipt"]["package_inventory"] = json.loads(
        json.dumps(
            _execution_evidence_bundle()["execution_receipt"][
                "final_inventory"
            ]
        )
    )
    _refresh_package_inventory_sha256(bundle)

    with pytest.raises(ValueError, match="package inventory"):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "failed_command",
    contract.EXECUTION_COMMAND_ORDER[:10],
)
def test_execution_failure_rejects_unproduced_final_inventory(
    failed_command,
):
    bundle = _truthful_inventory_failure_bundle(failed_command)
    bundle["execution_receipt"]["final_inventory"] = json.loads(
        json.dumps(
            _execution_evidence_bundle()["execution_receipt"][
                "final_inventory"
            ]
        )
    )
    _refresh_final_inventory_sha256(bundle)

    with pytest.raises(ValueError, match="final inventory"):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    ("failed_command", "field"),
    (
        ("assembly", "package_inventory_sha256"),
        ("safe_extract", "final_inventory_sha256"),
    ),
)
def test_execution_failure_rejects_incorrect_empty_inventory_hash(
    failed_command,
    field,
):
    bundle = _truthful_inventory_failure_bundle(failed_command)
    bundle["execution_receipt"][field] = "f" * 64

    with pytest.raises(
        ValueError,
        match=field.replace("_sha256", "").replace("_", " "),
    ):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "failed_command",
    contract.EXECUTION_COMMAND_ORDER[7:],
)
def test_remote_verifier_failure_rejects_missing_authority_manifest_row(
    failed_command,
):
    bundle = _truthful_inventory_failure_bundle(failed_command)
    receipt = bundle["execution_receipt"]
    authority_field = (
        "final_inventory"
        if failed_command == "local_verify"
        else "package_inventory"
    )
    receipt[authority_field] = [
        row
        for row in receipt[authority_field]
        if row["path"] != "artifact_manifest.json"
    ]
    if authority_field == "final_inventory":
        _refresh_final_inventory_sha256(bundle)
    else:
        _refresh_package_inventory_sha256(bundle)

    with pytest.raises(
        ValueError,
        match="artifact manifest verifier binding",
    ):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "failed_command",
    contract.EXECUTION_COMMAND_ORDER[7:],
)
def test_remote_verifier_failure_rejects_duplicate_authority_manifest_row(
    failed_command,
):
    bundle = _truthful_inventory_failure_bundle(failed_command)
    receipt = bundle["execution_receipt"]
    authority_field = (
        "final_inventory"
        if failed_command == "local_verify"
        else "package_inventory"
    )
    manifest_row = next(
        row
        for row in receipt[authority_field]
        if row["path"] == "artifact_manifest.json"
    )
    receipt[authority_field].insert(1, dict(manifest_row))
    if authority_field == "final_inventory":
        _refresh_final_inventory_sha256(bundle)
    else:
        _refresh_package_inventory_sha256(bundle)

    with pytest.raises(
        ValueError,
        match="inventory paths|artifact manifest verifier binding",
    ):
        contract.validate_execution_evidence_bundle(bundle)


def test_local_verify_failure_rejects_verifier_binding_to_package_inventory():
    bundle = _truthful_inventory_failure_bundle("local_verify")
    receipt = bundle["execution_receipt"]
    package_row = next(
        row
        for row in receipt["package_inventory"]
        if row["path"] != "artifact_manifest.json"
    )
    final_manifest = next(
        row
        for row in receipt["final_inventory"]
        if row["path"] == "artifact_manifest.json"
    )
    package_row["sha256"] = "0" * 64
    assert package_row["sha256"] != final_manifest["sha256"]
    _refresh_package_inventory_sha256(bundle)
    bundle["remote_verifier_output"][
        "artifact_manifest_sha256"
    ] = package_row["sha256"]

    with pytest.raises(
        ValueError,
        match="artifact manifest verifier binding",
    ):
        contract.validate_execution_evidence_bundle(bundle)


def test_local_verify_failure_accepts_truthful_remote_verified_evidence():
    bundle = _local_verify_failure_evidence_bundle()
    receipt = bundle["execution_receipt"]
    failure_index = contract.EXECUTION_COMMAND_ORDER.index("local_verify")

    assert all(
        row["outcome"] == "attempted" and row["returncode"] == 0
        for row in receipt["command_results"][:failure_index]
    )
    assert receipt["command_results"][failure_index]["outcome"] == "attempted"
    assert receipt["command_results"][failure_index]["returncode"] != 0
    assert receipt["command_results"][failure_index + 1 :] == []
    assert bundle["resource_guard_before"]["sha256"] == receipt[
        "resource_guard_before_sha256"
    ]
    assert bundle["resource_guard_after"]["sha256"] == receipt[
        "resource_guard_after_sha256"
    ]
    assert bundle["remote_verifier_output"]["role"] == "remote"
    assert "local_verifier_output" not in bundle
    assert "independent_verification" not in bundle
    assert receipt["remote_path_created"] is True
    assert receipt["source_staged"] is True
    assert receipt["worker_launched"] is True

    contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "document",
    ("local_verifier_output", "independent_verification"),
)
def test_local_verify_failure_rejects_fabricated_later_verifier_document(
    document,
):
    bundle = _local_verify_failure_evidence_bundle()
    bundle[document] = _execution_evidence_bundle()[document]

    with pytest.raises(ValueError, match="execution evidence bundle"):
        contract.validate_execution_evidence_bundle(bundle)


def test_local_verify_failure_rejects_missing_remote_output():
    bundle = _local_verify_failure_evidence_bundle()
    bundle.pop("remote_verifier_output")

    with pytest.raises(
        ValueError,
        match="remote_verifier_output|remote verifier|required",
    ):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "document",
    ("resource_guard_before", "resource_guard_after"),
)
def test_local_verify_failure_rejects_missing_guard(document):
    bundle = _local_verify_failure_evidence_bundle()
    bundle.pop(document)

    with pytest.raises(
        ValueError,
        match=f"{document}|resource guard|required",
    ):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "field",
    (
        "resource_guard_before_sha256",
        "resource_guard_after_sha256",
    ),
)
def test_local_verify_failure_rejects_mismatching_guard_hash(field):
    bundle = _local_verify_failure_evidence_bundle()
    bundle["execution_receipt"][field] = "f" * 64

    with pytest.raises(ValueError, match=field.replace("_", " ")):
        contract.validate_execution_evidence_bundle(bundle)


def test_local_verify_failure_rejects_incomplete_success_prefix():
    bundle = _local_verify_failure_evidence_bundle()
    safe_extract = bundle["execution_receipt"]["command_results"][
        contract.EXECUTION_COMMAND_ORDER.index("safe_extract")
    ]
    safe_extract["outcome"] = "skipped"
    safe_extract["returncode"] = None
    safe_extract["stdout"] = ""
    safe_extract["stderr"] = ""

    with pytest.raises(ValueError, match="attempted command ordering"):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "field",
    ("remote_path_created", "source_staged", "worker_launched"),
)
def test_local_verify_failure_rejects_false_prior_side_effect_flag(field):
    bundle = _local_verify_failure_evidence_bundle()
    bundle["execution_receipt"][field] = False

    with pytest.raises(ValueError, match=field.replace("_", " ")):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "failed_command",
    ("reserve_remote", "upload", "stage"),
)
def test_pre_guard_failure_requires_null_guard_hashes(failed_command):
    bundle = _execution_failure_at(failed_command)
    bundle.pop("resource_guard_before")
    bundle.pop("resource_guard_after")
    receipt = bundle["execution_receipt"]
    receipt["resource_guard_before_sha256"] = None
    receipt["resource_guard_after_sha256"] = None

    contract.validate_execution_evidence_bundle(bundle)


def test_execution_success_keeps_matching_guard_hashes_required():
    bundle = _execution_evidence_bundle()
    contract.validate_execution_evidence_bundle(bundle)

    for field in (
        "resource_guard_before_sha256",
        "resource_guard_after_sha256",
    ):
        malformed = json.loads(json.dumps(bundle))
        malformed["execution_receipt"][field] = None
        with pytest.raises(ValueError, match=field.replace("_", " ")):
            contract.validate_execution_evidence_bundle(malformed)


def test_execution_success_accepts_canonical_inventory_binding():
    bundle = _execution_evidence_bundle()

    receipt = bundle["execution_receipt"]
    assert {
        row["path"] for row in receipt["package_inventory"]
    } == set(contract.ARTIFACT_MANIFEST_HASH_DOMAIN)
    assert {
        row["path"] for row in receipt["final_inventory"]
    } == set(contract.PRODUCER_TRUST_DOMAIN)
    assert "execution_receipt.json" not in {
        row["path"] for row in receipt["package_inventory"]
    }
    assert "execution_receipt.json" not in {
        row["path"] for row in receipt["final_inventory"]
    }
    contract.validate_execution_evidence_bundle(bundle)


def _execution_success_with_full_producer_domain():
    bundle = _execution_evidence_bundle()
    receipt = bundle["execution_receipt"]
    manifest_row = next(
        copy.deepcopy(row)
        for row in receipt["final_inventory"]
        if row["path"] == "artifact_manifest.json"
    )
    receipt["package_inventory"] = [
        {
            "path": path,
            "sha256": f"{index + 1:064x}",
            "bytes": index + 1,
            "type": "file",
        }
        for index, path in enumerate(
            sorted(contract.ARTIFACT_MANIFEST_HASH_DOMAIN)
        )
    ]
    receipt["final_inventory"] = sorted(
        [*copy.deepcopy(receipt["package_inventory"]), manifest_row],
        key=lambda row: row["path"],
    )
    _refresh_package_inventory_sha256(bundle)
    _refresh_final_inventory_sha256(bundle)
    return bundle


def test_execution_success_producer_trust_domain_accepts_full_domain():
    bundle = _execution_success_with_full_producer_domain()

    contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    ("inventory_name", "forbidden_path"),
    (
        ("package_inventory", "execution_receipt.json"),
        ("package_inventory", "artifact_manifest.json"),
        ("final_inventory", "execution_receipt.json"),
    ),
)
def test_detached_execution_receipt_rejects_inventory_cycles(
    inventory_name,
    forbidden_path,
):
    bundle = _execution_success_with_full_producer_domain()
    receipt = bundle["execution_receipt"]
    inventory = copy.deepcopy(receipt[inventory_name])
    inventory.append({
        "path": forbidden_path,
        "sha256": contract.canonical_json_sha256(receipt),
        "bytes": 1,
        "type": "file",
    })
    inventory.sort(key=lambda row: row["path"])
    receipt[inventory_name] = inventory
    if inventory_name == "package_inventory":
        _refresh_package_inventory_sha256(bundle)
    else:
        _refresh_final_inventory_sha256(bundle)

    with pytest.raises(
        ValueError,
        match="package|final|inventory|domain|receipt|cycle",
    ):
        contract.validate_execution_evidence_bundle(bundle)


def test_detached_execution_failure_inventory_excludes_receipt():
    bundle = _truthful_inventory_failure_bundle("local_verify")
    receipt = bundle["execution_receipt"]

    assert "execution_receipt.json" not in {
        row["path"] for row in receipt["package_inventory"]
    }
    assert "execution_receipt.json" not in {
        row["path"] for row in receipt["final_inventory"]
    }
    contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "mutation",
    ("missing_required_path", "substituted_safe_path"),
)
def test_execution_success_producer_trust_domain_rejects_drift(mutation):
    bundle = _execution_success_with_full_producer_domain()
    receipt = bundle["execution_receipt"]
    manifest_row = next(
        copy.deepcopy(row)
        for row in receipt["final_inventory"]
        if row["path"] == "artifact_manifest.json"
    )
    package_inventory = copy.deepcopy(receipt["package_inventory"])

    if mutation == "missing_required_path":
        package_inventory.pop()
    else:
        package_inventory[-1]["path"] = "safe/extra-producer-output.json"
        package_inventory.sort(key=lambda row: row["path"])

    assert {row["path"] for row in package_inventory} != set(
        contract.ARTIFACT_MANIFEST_HASH_DOMAIN
    )
    receipt["package_inventory"] = package_inventory
    receipt["final_inventory"] = sorted(
        [*copy.deepcopy(package_inventory), manifest_row],
        key=lambda row: row["path"],
    )
    _refresh_package_inventory_sha256(bundle)
    _refresh_final_inventory_sha256(bundle)

    with pytest.raises(
        ValueError,
        match="package|final|inventory|domain",
    ):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "drift",
    ("path", "sha256", "bytes", "delete"),
)
def test_slice_c_execution_success_rejects_package_final_inventory_drift(
    drift,
):
    bundle = _execution_evidence_bundle()
    receipt = bundle["execution_receipt"]
    package_inventory = copy.deepcopy(receipt["package_inventory"])
    receipt["final_inventory"] = copy.deepcopy(receipt["final_inventory"])
    row = next(
        row
        for row in receipt["final_inventory"]
        if row["path"] != "artifact_manifest.json"
    )

    if drift == "path":
        row["path"] = "tools/worker-renamed.py"
        receipt["final_inventory"].sort(key=lambda item: item["path"])
    elif drift == "sha256":
        row["sha256"] = "0" * 64
    elif drift == "bytes":
        row["bytes"] += 1
    else:
        receipt["final_inventory"].remove(row)
    _refresh_final_inventory_sha256(bundle)

    assert receipt["package_inventory"] == package_inventory
    with pytest.raises(
        ValueError,
        match="package|final|inventory|extract|equality",
    ):
        contract.validate_execution_evidence_bundle(bundle)


def test_execution_success_rejects_unanimous_absent_artifact_manifest_hash():
    bundle = _execution_evidence_bundle()
    contract.validate_execution_evidence_bundle(bundle)
    inventory_sha256 = next(
        row["sha256"]
        for row in bundle["execution_receipt"]["final_inventory"]
        if row["path"] == "artifact_manifest.json"
    )
    absent_sha256 = "0" * 64
    assert absent_sha256 != inventory_sha256
    assert all(
        row["sha256"] != absent_sha256
        for row in bundle["execution_receipt"]["final_inventory"]
    )
    _set_success_verifier_artifact_manifest_sha256(
        bundle,
        absent_sha256,
    )

    with pytest.raises(
        ValueError,
        match="artifact manifest verifier binding",
    ):
        contract.validate_execution_evidence_bundle(bundle)


def test_execution_success_rejects_missing_artifact_manifest_inventory_row():
    bundle = _execution_evidence_bundle()
    bundle["execution_receipt"]["final_inventory"] = [
        row
        for row in bundle["execution_receipt"]["final_inventory"]
        if row["path"] != "artifact_manifest.json"
    ]
    _refresh_final_inventory_sha256(bundle)

    with pytest.raises(
        ValueError,
        match="artifact manifest verifier binding",
    ):
        contract.validate_execution_evidence_bundle(bundle)


def test_execution_success_rejects_duplicate_artifact_manifest_inventory_rows():
    bundle = _execution_evidence_bundle()
    manifest_row = next(
        row
        for row in bundle["execution_receipt"]["final_inventory"]
        if row["path"] == "artifact_manifest.json"
    )
    bundle["execution_receipt"]["final_inventory"].insert(
        1,
        dict(manifest_row),
    )
    _refresh_final_inventory_sha256(bundle)

    with pytest.raises(
        ValueError,
        match="final inventory paths|artifact manifest verifier binding",
    ):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "classification",
    (
        "GO",
        "NO_GO_CORRECTNESS",
        "NO_GO_RUNTIME_SAFETY",
        "NO_GO_CACHE",
        "NO_GO_PERFORMANCE",
    ),
)
def test_execution_success_accepts_each_allowed_final_classification(
    classification,
):
    bundle = _execution_evidence_bundle()
    _set_success_classification(bundle, classification)

    contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "failed_command",
    contract.EXECUTION_COMMAND_ORDER[7:],
)
def test_execution_failed_completed_remote_verifier_rejects_blocked_resources(
    failed_command,
):
    bundle = _truthful_inventory_failure_bundle(failed_command)
    _set_failure_remote_classification(bundle, "BLOCKED_RESOURCES")

    with pytest.raises(ValueError, match="classification"):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "failed_command",
    contract.EXECUTION_COMMAND_ORDER[7:],
)
@pytest.mark.parametrize(
    "classification",
    (
        "GO",
        "NO_GO_CORRECTNESS",
        "NO_GO_RUNTIME_SAFETY",
        "NO_GO_CACHE",
        "NO_GO_PERFORMANCE",
        "INVALID_ARTIFACT",
    ),
)
def test_execution_failed_completed_remote_verifier_accepts_executed_outcome(
    failed_command,
    classification,
):
    bundle = _truthful_inventory_failure_bundle(failed_command)
    _set_failure_remote_classification(bundle, classification)

    contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "failed_command",
    contract.EXECUTION_COMMAND_ORDER[7:],
)
def test_execution_failed_completed_remote_verifier_rejects_unknown_classification(
    failed_command,
):
    bundle = _truthful_inventory_failure_bundle(failed_command)
    _set_failure_remote_classification(bundle, "UNKNOWN")

    with pytest.raises(ValueError, match="classification"):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    ("failed_command", "document"),
    tuple(
        (failed_command, document)
        for failed_command in contract.EXECUTION_COMMAND_ORDER[:4]
        for document in (
            "resource_guard_before",
            "resource_guard_after",
            "remote_verifier_output",
            "local_verifier_output",
            "independent_verification",
        )
    )
    + tuple(
        (failed_command, document)
        for failed_command in contract.EXECUTION_COMMAND_ORDER[4:7]
        for document in (
            "resource_guard_after",
            "remote_verifier_output",
            "local_verifier_output",
            "independent_verification",
        )
    )
    + tuple(
        ("final_resource_guard", document)
        for document in (
            "resource_guard_after",
            "local_verifier_output",
            "independent_verification",
        )
    )
    + tuple(
        (failed_command, document)
        for failed_command in contract.EXECUTION_COMMAND_ORDER[8:]
        for document in (
            "local_verifier_output",
            "independent_verification",
        )
    ),
)
def test_each_failure_stage_rejects_forbidden_trusted_document(
    failed_command,
    document,
):
    bundle = _stage_failure_evidence_bundle(failed_command)
    bundle[document] = _execution_evidence_bundle()[document]

    with pytest.raises(
        ValueError,
        match="execution evidence bundle|resource guard documents",
    ):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    ("failed_command", "field"),
    tuple(
        (failed_command, field)
        for failed_command in contract.EXECUTION_COMMAND_ORDER[:4]
        for field in (
            "resource_guard_before_sha256",
            "resource_guard_after_sha256",
        )
    )
    + tuple(
        (failed_command, "resource_guard_after_sha256")
        for failed_command in contract.EXECUTION_COMMAND_ORDER[4:8]
    ),
)
def test_each_absent_failure_guard_hash_rejects_non_null_value(
    failed_command,
    field,
):
    bundle = _stage_failure_evidence_bundle(failed_command)
    assert bundle["execution_receipt"][field] is None
    bundle["execution_receipt"][field] = "f" * 64

    with pytest.raises(ValueError, match=field.replace("_", " ")):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    ("failed_command", "field"),
    tuple(
        (failed_command, "resource_guard_before_sha256")
        for failed_command in contract.EXECUTION_COMMAND_ORDER[4:]
    )
    + tuple(
        (failed_command, "resource_guard_after_sha256")
        for failed_command in contract.EXECUTION_COMMAND_ORDER[8:]
    ),
)
def test_each_present_failure_guard_hash_rejects_independent_corruption(
    failed_command,
    field,
):
    bundle = _stage_failure_evidence_bundle(failed_command)
    assert bundle["execution_receipt"][field] is not None
    bundle["execution_receipt"][field] = "f" * 64

    with pytest.raises(ValueError, match=field.replace("_", " ")):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "failed_command",
    contract.EXECUTION_COMMAND_ORDER[1:],
)
def test_each_noninitial_failure_rejects_incomplete_earlier_success_prefix(
    failed_command,
):
    bundle = _stage_failure_evidence_bundle(failed_command)
    failure_index = contract.EXECUTION_COMMAND_ORDER.index(failed_command)
    earlier = bundle["execution_receipt"]["command_results"][
        failure_index - 1
    ]
    earlier["outcome"] = "skipped"
    earlier["returncode"] = None
    earlier["stdout"] = ""
    earlier["stderr"] = ""

    with pytest.raises(ValueError, match="attempted command ordering"):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "failed_command",
    contract.EXECUTION_COMMAND_ORDER[:-1],
)
def test_each_nonfinal_failure_rejects_later_attempted_command(
    failed_command,
):
    bundle = _stage_failure_evidence_bundle(failed_command)
    failure_index = contract.EXECUTION_COMMAND_ORDER.index(failed_command)
    later = bundle["execution_receipt"]["command_results"][
        failure_index + 1
    ]
    later["outcome"] = "attempted"
    later["returncode"] = 0
    later["stdout"] = "unexpected"

    with pytest.raises(ValueError, match="skipped command result"):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "failed_command",
    contract.EXECUTION_COMMAND_ORDER,
)
def test_each_failure_stage_rejects_success_receipt_classification(
    failed_command,
):
    bundle = _stage_failure_evidence_bundle(failed_command)
    bundle["execution_receipt"]["classification"] = "PASS"

    with pytest.raises(
        ValueError,
        match="execution receipt classification",
    ):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    ("failed_command", "field"),
    (
        ("upload", "remote_path_created"),
        ("resource_guard", "source_staged"),
        ("assembly", "worker_launched"),
        ("remote_verify", "worker_launched"),
        ("final_resource_guard", "worker_launched"),
        ("package_download", "worker_launched"),
        ("safe_extract", "worker_launched"),
        ("local_verify", "worker_launched"),
    ),
)
def test_completed_producer_lower_bounds_reject_false_flags(
    failed_command,
    field,
):
    bundle = _stage_failure_evidence_bundle(failed_command)
    bundle["execution_receipt"][field] = False

    with pytest.raises(ValueError, match=field.replace("_", " ")):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    ("failed_command", "field"),
    (
        ("reserve_remote", "source_staged"),
        ("reserve_remote", "worker_launched"),
        ("upload", "source_staged"),
        ("upload", "worker_launched"),
        ("stage", "worker_launched"),
        ("resource_guard", "worker_launched"),
    ),
)
def test_nonproducer_failure_rejects_future_side_effect(
    failed_command,
    field,
):
    bundle = _stage_failure_evidence_bundle(failed_command)
    bundle["execution_receipt"][field] = True
    if field == "source_staged":
        bundle["execution_receipt"]["remote_path_created"] = True
    elif field == "worker_launched":
        bundle["execution_receipt"]["remote_path_created"] = True
        bundle["execution_receipt"]["source_staged"] = True
    bundle["execution_receipt"]["cleanup_complete"] = True

    with pytest.raises(
        ValueError,
        match="remote path created|source staged|worker launched",
    ):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    ("failed_command", "side_effects"),
    (
        (
            "reserve_remote",
            {
                "remote_path_created": True,
                "source_staged": False,
                "worker_launched": False,
            },
        ),
        (
            "stage",
            {
                "remote_path_created": True,
                "source_staged": True,
                "worker_launched": False,
            },
        ),
        (
            "workers",
            {
                "remote_path_created": True,
                "source_staged": True,
                "worker_launched": True,
            },
        ),
    ),
)
def test_failing_producer_accepts_truthful_partial_side_effect(
    failed_command,
    side_effects,
):
    bundle = _stage_failure_evidence_bundle(failed_command)
    bundle["execution_receipt"].update(side_effects)
    bundle["execution_receipt"]["cleanup_complete"] = True

    contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    ("failed_command", "field"),
    (
        ("reserve_remote", "remote_path_created"),
        ("stage", "source_staged"),
        ("workers", "worker_launched"),
    ),
)
@pytest.mark.parametrize(
    "malformed",
    (None, 0, 1, "", "nonempty", [], {}),
    ids=(
        "none",
        "zero",
        "one",
        "empty-string",
        "nonempty-string",
        "list",
        "dict",
    ),
)
def test_failing_producer_rejects_non_boolean_side_effect_receipt(
    failed_command,
    field,
    malformed,
):
    bundle = _stage_failure_evidence_bundle(failed_command)
    receipt = bundle["execution_receipt"]
    receipt[field] = malformed
    receipt["cleanup_complete"] = any(
        receipt[side_effect]
        for side_effect in (
            "remote_path_created",
            "source_staged",
            "worker_launched",
        )
    )

    with pytest.raises(ValueError, match=field.replace("_", " ")):
        contract.validate_execution_evidence_bundle(bundle)


def test_reserve_remote_failure_without_side_effect_has_no_cleanup_obligation():
    bundle = _stage_failure_evidence_bundle("reserve_remote")
    receipt = bundle["execution_receipt"]
    receipt["remote_path_created"] = False
    receipt["source_staged"] = False
    receipt["worker_launched"] = False
    receipt["cleanup_complete"] = False

    contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    ("failed_command", "cleanup_complete"),
    (
        ("reserve_remote", True),
        ("upload", False),
    ),
)
def test_execution_failed_rejects_cleanup_obligation_mismatch(
    failed_command,
    cleanup_complete,
):
    bundle = _stage_failure_evidence_bundle(failed_command)
    bundle["execution_receipt"]["cleanup_complete"] = cleanup_complete

    with pytest.raises(ValueError, match="cleanup complete"):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    ("failed_command", "side_effects"),
    (
        (
            "reserve_remote",
            {
                "remote_path_created": False,
                "source_staged": True,
                "worker_launched": False,
            },
        ),
        (
            "stage",
            {
                "remote_path_created": True,
                "source_staged": False,
                "worker_launched": True,
            },
        ),
        (
            "workers",
            {
                "remote_path_created": False,
                "source_staged": True,
                "worker_launched": True,
            },
        ),
    ),
)
def test_execution_failed_rejects_side_effect_dependency_violation(
    failed_command,
    side_effects,
):
    bundle = _stage_failure_evidence_bundle(failed_command)
    bundle["execution_receipt"].update(side_effects)
    bundle["execution_receipt"]["cleanup_complete"] = True

    with pytest.raises(ValueError, match="side effect|created|staged|launched"):
        contract.validate_execution_evidence_bundle(bundle)


@pytest.mark.parametrize(
    "document",
    ("package_download", "safe_extract"),
)
def test_execution_failed_rejects_invented_package_or_extraction_document(
    document,
):
    bundle = _stage_failure_evidence_bundle("local_verify")
    bundle[document] = {"classification": "PASS"}

    with pytest.raises(ValueError, match="execution evidence bundle"):
        contract.validate_execution_evidence_bundle(bundle)


def test_verifier_identity_manifest_and_classification_must_agree():
    bundle = _execution_evidence_bundle()
    for document_name, field, value in (
        ("remote_verifier_output", "verifier_source_sha256", "0" * 64),
        ("remote_verifier_output", "verifier_version_sha256", "0" * 64),
        ("remote_verifier_output", "artifact_manifest_sha256", "0" * 64),
        ("remote_verifier_output", "classification", "NO_GO_CACHE"),
        (
            "independent_verification",
            "classification",
            "NO_GO_PERFORMANCE",
        ),
    ):
        malformed = json.loads(json.dumps(bundle))
        malformed[document_name][field] = value
        if document_name == "remote_verifier_output":
            malformed["independent_verification"][
                "remote_verifier_sha256"
            ] = contract.canonical_json_sha256(
                malformed["remote_verifier_output"]
            )
        with pytest.raises(
            ValueError,
            match="verifier|classification|artifact manifest",
        ):
            contract.validate_execution_evidence_bundle(malformed)


def test_execution_success_rejects_unanimous_blocked_resources_verdict():
    bundle = _execution_evidence_bundle()
    assert bundle["execution_receipt"]["classification"] == "PASS"
    contract.validate_execution_evidence_bundle(bundle)

    malformed = json.loads(json.dumps(bundle))
    malformed["local_verifier_output"][
        "classification"
    ] = "BLOCKED_RESOURCES"
    malformed["remote_verifier_output"][
        "classification"
    ] = "BLOCKED_RESOURCES"
    malformed["independent_verification"][
        "local_verifier_sha256"
    ] = contract.canonical_json_sha256(
        malformed["local_verifier_output"]
    )
    malformed["independent_verification"][
        "remote_verifier_sha256"
    ] = contract.canonical_json_sha256(
        malformed["remote_verifier_output"]
    )
    malformed["independent_verification"][
        "classification"
    ] = "BLOCKED_RESOURCES"

    with pytest.raises(
        ValueError,
        match="classification|lifecycle|success|blocked",
    ):
        contract.validate_execution_evidence_bundle(malformed)


def test_execution_success_rejects_unanimous_invalid_artifact_verdict():
    bundle = _execution_evidence_bundle()
    assert bundle["execution_receipt"]["classification"] == "PASS"
    contract.validate_execution_evidence_bundle(bundle)

    malformed = json.loads(json.dumps(bundle))
    malformed["local_verifier_output"][
        "classification"
    ] = "INVALID_ARTIFACT"
    malformed["remote_verifier_output"][
        "classification"
    ] = "INVALID_ARTIFACT"
    malformed["independent_verification"][
        "local_verifier_sha256"
    ] = contract.canonical_json_sha256(
        malformed["local_verifier_output"]
    )
    malformed["independent_verification"][
        "remote_verifier_sha256"
    ] = contract.canonical_json_sha256(
        malformed["remote_verifier_output"]
    )
    malformed["independent_verification"][
        "classification"
    ] = "INVALID_ARTIFACT"

    with pytest.raises(
        ValueError,
        match="classification|lifecycle|success|invalid",
    ):
        contract.validate_execution_evidence_bundle(malformed)


def test_resource_guard_hash_rejects_content_tamper():
    bundle = _execution_evidence_bundle()
    malformed = json.loads(json.dumps(bundle))
    malformed["resource_guard_before"]["gpu_query_rows"][0][
        "free_bytes"
    ] += 1
    with pytest.raises(ValueError, match="resource guard.*sha256"):
        contract.validate_execution_evidence_bundle(malformed)


def test_equal_resource_observations_and_guard_hashes_are_valid():
    bundle = _execution_evidence_bundle()
    assert (
        bundle["resource_guard_before"]["gpu_query_rows"]
        == bundle["resource_guard_after"]["gpu_query_rows"]
    )
    assert (
        bundle["resource_guard_before"]["sha256"]
        == bundle["resource_guard_after"]["sha256"]
    )
    contract.validate_execution_evidence_bundle(bundle)


def test_nested_evidence_manifests_are_closed_complete_and_hash_bound():
    manifests, file_inventory, artifact_manifest = (
        _nested_evidence_bundle()
    )
    contract.validate_nested_evidence_bundle(
        manifests,
        file_inventory,
        artifact_manifest,
    )

    malformed = json.loads(json.dumps(manifests))
    malformed["tokens"]["rows"].pop()
    with pytest.raises(
        ValueError,
        match="token coverage|file inventory",
    ):
        contract.validate_nested_evidence_bundle(
            malformed,
            file_inventory,
            artifact_manifest,
        )

    malformed = json.loads(json.dumps(manifests))
    malformed["logs"]["rows"].append(
        json.loads(json.dumps(malformed["logs"]["rows"][0]))
    )
    with pytest.raises(
        ValueError,
        match="duplicate|log coverage|file inventory",
    ):
        contract.validate_nested_evidence_bundle(
            malformed,
            file_inventory,
            artifact_manifest,
        )

    malformed_inventory = json.loads(json.dumps(file_inventory))
    malformed_inventory[0]["type"] = "symlink"
    with pytest.raises(ValueError, match="type|symlink"):
        contract.validate_nested_evidence_bundle(
            manifests,
            malformed_inventory,
            artifact_manifest,
        )

    malformed_inventory = json.loads(json.dumps(file_inventory))
    malformed_inventory.append({
        "path": "tokens/extra.json",
        "sha256": "a" * 64,
        "bytes": 1,
        "type": "regular_file",
    })
    with pytest.raises(ValueError, match="file inventory"):
        contract.validate_nested_evidence_bundle(
            manifests,
            malformed_inventory,
            artifact_manifest,
        )

    malformed_manifest = json.loads(json.dumps(artifact_manifest))
    for entry in malformed_manifest["entries"]:
        if entry["path"] == "token_manifest.json":
            entry["sha256"] = "0" * 64
            break
    with pytest.raises(ValueError, match="nested manifest hash"):
        contract.validate_nested_evidence_bundle(
            manifests,
            file_inventory,
            malformed_manifest,
        )

    malformed = json.loads(json.dumps(manifests))
    p0_row = next(
        row
        for row in malformed["snapshots"]["rows"]
        if row["profile"] == "recompute"
    )
    p0_row["evidence_kind"] = "snapshot"
    with pytest.raises(ValueError, match="profile evidence"):
        contract.validate_nested_evidence_bundle(
            malformed,
            file_inventory,
            artifact_manifest,
        )


def test_nested_final_logit_shape_uses_model_vocabulary_size():
    manifests, file_inventory, artifact_manifest = (
        _nested_evidence_bundle()
    )
    contract.validate_nested_evidence_bundle(
        manifests,
        file_inventory,
        artifact_manifest,
    )

    malformed = json.loads(json.dumps(manifests))
    for row in malformed["logits"]["rows"]:
        row["shape"] = [contract.TOKEN_ID_UPPER_BOUND]
    with pytest.raises(ValueError, match="logit shape"):
        contract.validate_nested_evidence_bundle(
            malformed,
            file_inventory,
            artifact_manifest,
        )


def test_case_row_nested_evidence_bindings_accept_canonical_joint_collection():
    rows, manifests, file_inventory, artifact_manifest = (
        _joint_case_row_nested_evidence()
    )
    _assert_case_and_nested_planes_valid(
        rows,
        manifests,
        file_inventory,
        artifact_manifest,
    )
    assert len(rows) == len(manifests["logits"]["rows"]) == 420
    assert len(manifests["tokens"]["rows"]) == 840

    contract.validate_case_row_nested_evidence_bindings(
        rows,
        manifests,
    )


def test_case_row_nested_evidence_bindings_reject_prompt_path_mismatch():
    rows, manifests, file_inventory, artifact_manifest = (
        _joint_case_row_nested_evidence()
    )
    rows[len(rows) // 2]["prompt_token_ids_path"] = (
        "tokens/independently-valid-prompt.json"
    )
    _assert_case_and_nested_planes_valid(
        rows,
        manifests,
        file_inventory,
        artifact_manifest,
    )

    with pytest.raises(ValueError, match="prompt|binding|evidence"):
        contract.validate_case_row_nested_evidence_bindings(
            rows,
            manifests,
        )


def test_case_row_nested_evidence_bindings_reject_prompt_hash_mismatch():
    rows, manifests, file_inventory, artifact_manifest = (
        _joint_case_row_nested_evidence()
    )
    rows[0]["prompt_token_ids_sha256"] = "0" * 64
    _assert_case_and_nested_planes_valid(
        rows,
        manifests,
        file_inventory,
        artifact_manifest,
    )

    with pytest.raises(ValueError, match="prompt|binding|evidence"):
        contract.validate_case_row_nested_evidence_bindings(
            rows,
            manifests,
        )


def test_case_row_nested_evidence_bindings_reject_output_path_mismatch():
    rows, manifests, file_inventory, artifact_manifest = (
        _joint_case_row_nested_evidence()
    )
    rows[1]["output_token_ids_path"] = (
        "tokens/independently-valid-output.json"
    )
    _assert_case_and_nested_planes_valid(
        rows,
        manifests,
        file_inventory,
        artifact_manifest,
    )

    with pytest.raises(ValueError, match="output|binding|evidence"):
        contract.validate_case_row_nested_evidence_bindings(
            rows,
            manifests,
        )


def test_case_row_nested_evidence_bindings_reject_output_hash_mismatch():
    rows, manifests, file_inventory, artifact_manifest = (
        _joint_case_row_nested_evidence()
    )
    rows[1]["output_token_ids_sha256"] = "0" * 64
    _assert_case_and_nested_planes_valid(
        rows,
        manifests,
        file_inventory,
        artifact_manifest,
    )

    with pytest.raises(ValueError, match="output|binding|evidence"):
        contract.validate_case_row_nested_evidence_bindings(
            rows,
            manifests,
        )


def test_case_row_nested_evidence_bindings_reject_final_logit_path_mismatch():
    rows, manifests, file_inventory, artifact_manifest = (
        _joint_case_row_nested_evidence()
    )
    rows[2]["final_logits_path"] = (
        "logits/independently-valid-final.bin"
    )
    _assert_case_and_nested_planes_valid(
        rows,
        manifests,
        file_inventory,
        artifact_manifest,
    )

    with pytest.raises(ValueError, match="logit|binding|evidence"):
        contract.validate_case_row_nested_evidence_bindings(
            rows,
            manifests,
        )


def test_case_row_nested_evidence_bindings_reject_final_logit_hash_mismatch():
    rows, manifests, file_inventory, artifact_manifest = (
        _joint_case_row_nested_evidence()
    )
    rows[2]["final_logits_sha256"] = "0" * 64
    _assert_case_and_nested_planes_valid(
        rows,
        manifests,
        file_inventory,
        artifact_manifest,
    )

    with pytest.raises(ValueError, match="logit|binding|evidence"):
        contract.validate_case_row_nested_evidence_bindings(
            rows,
            manifests,
        )


def test_case_row_nested_evidence_bindings_reject_final_logit_shape_mismatch():
    rows, manifests, file_inventory, artifact_manifest = (
        _joint_case_row_nested_evidence()
    )
    rows[2]["final_logits_shape"] = [1]
    _assert_case_and_nested_planes_valid(
        rows,
        manifests,
        file_inventory,
        artifact_manifest,
    )

    with pytest.raises(ValueError, match="logit|shape|binding|evidence"):
        contract.validate_case_row_nested_evidence_bindings(
            rows,
            manifests,
        )


def test_case_row_nested_evidence_bindings_reject_cross_request_key_binding():
    rows, manifests, file_inventory, artifact_manifest = (
        _joint_case_row_nested_evidence()
    )
    evidence_fields = (
        "prompt_token_ids_path",
        "prompt_token_ids_sha256",
        "output_token_ids_path",
        "output_token_ids_sha256",
        "final_logits_path",
        "final_logits_sha256",
        "final_logits_shape",
        "final_logits_dtype",
    )
    first = {field: rows[0][field] for field in evidence_fields}
    second = {field: rows[1][field] for field in evidence_fields}
    rows[0].update(second)
    rows[1].update(first)
    _assert_case_and_nested_planes_valid(
        rows,
        manifests,
        file_inventory,
        artifact_manifest,
    )

    with pytest.raises(ValueError, match="case|request|binding|evidence"):
        contract.validate_case_row_nested_evidence_bindings(
            rows,
            manifests,
        )


def test_case_row_nested_evidence_bindings_reject_missing_token_manifest_row():
    rows, manifests, file_inventory, artifact_manifest = (
        _joint_case_row_nested_evidence()
    )
    _assert_case_and_nested_planes_valid(
        rows,
        manifests,
        file_inventory,
        artifact_manifest,
    )
    manifests["tokens"]["rows"].pop()

    with pytest.raises(ValueError, match="token|coverage|binding|evidence"):
        contract.validate_case_row_nested_evidence_bindings(
            rows,
            manifests,
        )


def test_case_row_nested_evidence_bindings_reject_duplicate_logit_manifest_row():
    rows, manifests, file_inventory, artifact_manifest = (
        _joint_case_row_nested_evidence()
    )
    _assert_case_and_nested_planes_valid(
        rows,
        manifests,
        file_inventory,
        artifact_manifest,
    )
    manifests["logits"]["rows"].append(
        json.loads(json.dumps(manifests["logits"]["rows"][-1]))
    )

    with pytest.raises(ValueError, match="logit|coverage|binding|evidence"):
        contract.validate_case_row_nested_evidence_bindings(
            rows,
            manifests,
        )


def test_case_row_nested_evidence_bindings_cover_last_canonical_request():
    rows, manifests, file_inventory, artifact_manifest = (
        _joint_case_row_nested_evidence()
    )
    assert rows[-1]["row_id"] == (
        "w4_miss_invalidation__measured__r4__"
        "recompute__request-2"
    )
    rows[-1]["final_logits_sha256"] = "0" * 64
    _assert_case_and_nested_planes_valid(
        rows,
        manifests,
        file_inventory,
        artifact_manifest,
    )

    with pytest.raises(ValueError, match="logit|binding|evidence"):
        contract.validate_case_row_nested_evidence_bindings(
            rows,
            manifests,
        )


def test_process_row_nested_worker_log_bindings_accept_canonical_collection():
    process_rows, manifests, file_inventory, artifact_manifest = (
        _joint_process_row_nested_worker_logs()
    )
    contract.validate_process_rows(process_rows)
    contract.validate_nested_evidence_bundle(
        manifests,
        file_inventory,
        artifact_manifest,
    )
    assert len(process_rows) == len(manifests["logs"]["rows"]) == 420
    assert [
        (row["case_id"], row["rank"], row["world_size"])
        for row in process_rows
    ] == [
        (row["case_id"], row["rank"], row["world_size"])
        for row in manifests["logs"]["rows"]
    ]

    contract.validate_process_row_nested_worker_log_bindings(
        process_rows,
        manifests,
    )


def test_process_row_nested_worker_log_bindings_reject_missing_log_row():
    process_rows, manifests, file_inventory, artifact_manifest = (
        _joint_process_row_nested_worker_logs()
    )
    contract.validate_process_rows(process_rows)
    contract.validate_nested_evidence_bundle(
        manifests,
        file_inventory,
        artifact_manifest,
    )
    manifests["logs"]["rows"].pop()

    with pytest.raises(ValueError, match="log|coverage|binding|evidence"):
        contract.validate_process_row_nested_worker_log_bindings(
            process_rows,
            manifests,
        )


def test_process_row_nested_worker_log_bindings_reject_duplicate_log_row():
    process_rows, manifests, file_inventory, artifact_manifest = (
        _joint_process_row_nested_worker_logs()
    )
    contract.validate_process_rows(process_rows)
    contract.validate_nested_evidence_bundle(
        manifests,
        file_inventory,
        artifact_manifest,
    )
    manifests["logs"]["rows"].append(
        json.loads(json.dumps(manifests["logs"]["rows"][-1]))
    )

    with pytest.raises(ValueError, match="log|duplicate|binding|evidence"):
        contract.validate_process_row_nested_worker_log_bindings(
            process_rows,
            manifests,
        )


def test_process_row_nested_worker_log_bindings_reject_cross_key_binding():
    process_rows, manifests, file_inventory, artifact_manifest = (
        _joint_process_row_nested_worker_logs()
    )
    contract.validate_process_rows(process_rows)
    contract.validate_nested_evidence_bundle(
        manifests,
        file_inventory,
        artifact_manifest,
    )
    first = manifests["logs"]["rows"][0]
    second = manifests["logs"]["rows"][4]
    first["case_id"], second["case_id"] = (
        second["case_id"],
        first["case_id"],
    )

    with pytest.raises(ValueError, match="case|rank|log|binding|evidence"):
        contract.validate_process_row_nested_worker_log_bindings(
            process_rows,
            manifests,
        )


def test_process_row_nested_worker_log_bindings_cover_last_rank():
    process_rows, manifests, file_inventory, artifact_manifest = (
        _joint_process_row_nested_worker_logs()
    )
    contract.validate_process_rows(process_rows)
    contract.validate_nested_evidence_bundle(
        manifests,
        file_inventory,
        artifact_manifest,
    )
    assert (
        process_rows[-1]["case_id"],
        process_rows[-1]["rank"],
        process_rows[-1]["world_size"],
    ) == (
        "w4_miss_invalidation__measured__r4__recompute",
        3,
        4,
    )
    manifests["logs"]["rows"][-1]["rank"] = 2

    with pytest.raises(ValueError, match="rank|log|binding|evidence"):
        contract.validate_process_row_nested_worker_log_bindings(
            process_rows,
            manifests,
        )


def test_process_row_nested_snapshot_bindings_accept_canonical_collection():
    process_rows, manifests, file_inventory, artifact_manifest = (
        _joint_process_row_nested_snapshots()
    )
    contract.validate_process_rows(process_rows)
    contract.validate_nested_evidence_bundle(
        manifests,
        file_inventory,
        artifact_manifest,
    )
    assert len(process_rows) == len(manifests["snapshots"]["rows"]) == 420
    assert [
        (
            row["case_id"],
            row["profile"],
            row["rank"],
            row["world_size"],
        )
        for row in process_rows
    ] == [
        (
            row["case_id"],
            row["profile"],
            row["rank"],
            row["world_size"],
        )
        for row in manifests["snapshots"]["rows"]
    ]

    contract.validate_process_row_nested_snapshot_bindings(
        process_rows,
        manifests,
    )


def _aligned_recurrent_snapshot_accounting_pair():
    process_rows, manifests, file_inventory, artifact_manifest = (
        _joint_process_row_nested_snapshots()
    )
    accounting_fields = (
        (
            "hybrid_cache_current_logical_referenced_bytes",
            "full_fidelity_logical_bytes",
        ),
        (
            "hybrid_cache_current_unique_physical_bytes",
            "encoded_physical_bytes",
        ),
        (
            "hybrid_cache_current_metadata_bytes",
            "codec_metadata_bytes",
        ),
        (
            "encode_workspace_peak_allocated_bytes",
            "temporary_encode_workspace_bytes",
        ),
        (
            "decode_workspace_peak_allocated_bytes",
            "temporary_decode_workspace_bytes",
        ),
    )
    recurrent_pair = None
    for process_row, snapshot_row in zip(
        process_rows,
        manifests["snapshots"]["rows"],
    ):
        assert all(
            process_row[process_field] == snapshot_row[snapshot_field]
            for process_field, snapshot_field in accounting_fields
        )
        if (
            recurrent_pair is None
            and process_row["profile"] == contract.P2_PROFILE
        ):
            recurrent_pair = (process_row, snapshot_row)
    if recurrent_pair is None:
        raise AssertionError("missing recurrent_int8_per_row process/snapshot pair")
    process_row, snapshot_row = recurrent_pair
    return (
        process_rows,
        manifests,
        file_inventory,
        artifact_manifest,
        process_row,
        snapshot_row,
        accounting_fields,
    )


def test_process_row_nested_snapshot_accounting_bindings_accept_exact_values():
    (
        process_rows,
        manifests,
        _file_inventory,
        _artifact_manifest,
        process_row,
        snapshot_row,
        accounting_fields,
    ) = _aligned_recurrent_snapshot_accounting_pair()
    contract.validate_process_rows(process_rows)
    contract._validate_nested_manifest(
        "snapshots",
        manifests["snapshots"],
    )
    contract._validate_snapshot_manifests(
        manifests["snapshots"],
        manifests["tensor_inventories"],
    )
    assert all(
        process_row[process_field] == snapshot_row[snapshot_field]
        for process_field, snapshot_field in accounting_fields
    )

    contract.validate_process_row_nested_snapshot_bindings(
        process_rows,
        manifests,
    )


@pytest.mark.parametrize(
    ("process_field", "snapshot_field"),
    [
        pytest.param(
            "hybrid_cache_current_logical_referenced_bytes",
            "full_fidelity_logical_bytes",
            id="logical-referenced-bytes",
        ),
        pytest.param(
            "hybrid_cache_current_unique_physical_bytes",
            "encoded_physical_bytes",
            id="unique-physical-bytes",
        ),
        pytest.param(
            "hybrid_cache_current_metadata_bytes",
            "codec_metadata_bytes",
            id="metadata-bytes",
        ),
        pytest.param(
            "encode_workspace_peak_allocated_bytes",
            "temporary_encode_workspace_bytes",
            id="encode-workspace-bytes",
        ),
        pytest.param(
            "decode_workspace_peak_allocated_bytes",
            "temporary_decode_workspace_bytes",
            id="decode-workspace-bytes",
        ),
    ],
)
def test_process_row_nested_snapshot_accounting_bindings_reject_mismatch(
    process_field,
    snapshot_field,
):
    (
        process_rows,
        manifests,
        _file_inventory,
        _artifact_manifest,
        process_row,
        snapshot_row,
        _accounting_fields,
    ) = _aligned_recurrent_snapshot_accounting_pair()
    assert process_row[process_field] == snapshot_row[snapshot_field]
    snapshot_row[snapshot_field] += 1
    contract._validate_nested_manifest(
        "snapshots",
        manifests["snapshots"],
    )
    contract._validate_snapshot_manifests(
        manifests["snapshots"],
        manifests["tensor_inventories"],
    )

    with pytest.raises(ValueError, match="accounting|binding|field"):
        contract.validate_process_row_nested_snapshot_bindings(
            process_rows,
            manifests,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        pytest.param(
            lambda evidence: evidence.update({"unexpected": True}),
            "fields",
            id="unknown-document-field",
        ),
        pytest.param(
            lambda evidence: evidence["observations"][0].update(
                {"ordinal": True}
            ),
            "ordinal",
            id="bool-as-int",
        ),
        pytest.param(
            lambda evidence: evidence["snapshots"][0][
                "tensor_references"
            ][0].update({"resident_dtype": "float64"}),
            "dtype",
            id="bad-dtype",
        ),
        pytest.param(
            lambda evidence: evidence["snapshots"][0][
                "tensor_references"
            ].reverse(),
            "order",
            id="bad-reference-order",
        ),
        pytest.param(
            lambda evidence: evidence["snapshots"][0][
                "tensor_references"
            ][0].update({"layer_index": 18}),
            "layer",
            id="bad-layer",
        ),
    ],
)
def test_tensor_storage_evidence_schema_is_closed_and_semantic(
    mutation,
    error,
):
    evidence = _canonical_tensor_storage_evidence()
    mutation(evidence)

    with pytest.raises(ValueError, match=error):
        contract.validate_tensor_storage_evidence(evidence)


def test_tensor_storage_evidence_accepts_exact_restore_and_p2_semantics():
    exact_restore = _canonical_tensor_storage_evidence()
    recurrent_int8 = _canonical_tensor_storage_evidence(
        contract.P2_PROFILE
    )

    contract.validate_tensor_storage_evidence(exact_restore)
    contract.validate_tensor_storage_evidence(recurrent_int8)

    assert exact_restore["snapshots"][0]["codec_metadata"] is None
    assert recurrent_int8["snapshots"][0]["codec_metadata"]["layers"][0] == {
        "codec": contract.P2_CODEC_ID,
        "layer_index": 0,
        "source_dtype": "torch.float32",
        "source_shape": [1, 1, 1],
    }


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        pytest.param(
            lambda evidence: _prepend_tensor_storage_observation(
                evidence,
                "unknown",
            ),
            "event",
            id="unknown-event",
        ),
        pytest.param(
            lambda evidence: (
                _append_tensor_storage_snapshot(evidence, "snapshot-1"),
                evidence["observations"][0].update({
                    "active_snapshot_ids": ["snapshot-1", "snapshot-0"]
                }),
            ),
            "snapshot",
            id="unsorted-active-snapshot-ids",
        ),
        pytest.param(
            lambda evidence: (
                _append_workspace_storage(
                    evidence,
                    "encode-workspace",
                    "encode_workspace",
                    11,
                ),
                _append_workspace_storage(
                    evidence,
                    "decode-workspace",
                    "decode_workspace",
                    13,
                ),
                evidence["observations"][0].update({
                    "live_workspace_storage_ids": [
                        "encode-workspace",
                        "decode-workspace",
                    ],
                    "encode_workspace_reserved_bytes": 11,
                    "decode_workspace_reserved_bytes": 13,
                }),
            ),
            "workspace",
            id="unsorted-live-workspace-storage-ids",
        ),
        pytest.param(
            lambda evidence: evidence["observations"][0].update({
                "cuda_allocated_bytes": 513,
                "cuda_reserved_bytes": 512,
            }),
            "cuda",
            id="allocated-exceeds-reserved",
        ),
        pytest.param(
            lambda evidence: (
                _append_workspace_storage(
                    evidence,
                    "encode-workspace",
                    "encode_workspace",
                    11,
                ),
                evidence["observations"][0].update({
                    "live_workspace_storage_ids": ["encode-workspace"],
                    "encode_workspace_reserved_bytes": 10,
                }),
            ),
            "encode workspace reserved",
            id="encode-reserved-below-live-bytes",
        ),
        pytest.param(
            lambda evidence: (
                _append_workspace_storage(
                    evidence,
                    "decode-workspace",
                    "decode_workspace",
                    13,
                ),
                evidence["observations"][0].update({
                    "live_workspace_storage_ids": ["decode-workspace"],
                    "decode_workspace_reserved_bytes": 12,
                }),
            ),
            "decode workspace reserved",
            id="decode-reserved-below-live-bytes",
        ),
    ],
)
def test_tensor_storage_observations_enforce_closed_semantics(
    mutation,
    error,
):
    evidence = _canonical_tensor_storage_evidence()
    mutation(evidence)

    with pytest.raises(ValueError, match=error):
        contract.validate_tensor_storage_evidence(evidence)


def test_tensor_storage_evidence_accepts_real_shapes_and_binds_codec_shapes():
    exact_restore = _canonical_tensor_storage_evidence()
    recurrent_int8 = _canonical_tensor_storage_evidence(
        contract.P2_PROFILE
    )
    for evidence in (exact_restore, recurrent_int8):
        references = evidence["snapshots"][0]["tensor_references"]
        storages = {
            storage["storage_id"]: storage
            for storage in evidence["storages"]
        }
        for reference in references:
            if reference["semantic_role"] == "convolution":
                reference["logical_shape"] = [2, 3]
                reference["resident_shape"] = [2, 3]
                reference["storage_length_bytes"] = 12
            elif reference["semantic_role"] == "recurrent_values":
                reference["logical_shape"] = [2, 3, 4]
                reference["resident_shape"] = [2, 3, 4]
                reference["storage_length_bytes"] = (
                    96
                    if evidence["profile"] == contract.P1_REFERENCE_PROFILE
                    else 24
                )
            else:
                reference["resident_shape"] = [2, 3]
                reference["storage_length_bytes"] = 24
            storages[reference["storage_id"]]["storage_nbytes"] = reference[
                "storage_length_bytes"
            ]
        if evidence["profile"] == contract.P2_PROFILE:
            for layer in evidence["snapshots"][0]["codec_metadata"]["layers"]:
                layer["source_shape"] = [2, 3, 4]

    contract.validate_tensor_storage_evidence(exact_restore)
    contract.validate_tensor_storage_evidence(recurrent_int8)

    malformed = copy.deepcopy(recurrent_int8)
    scale = next(
        reference
        for reference in malformed["snapshots"][0]["tensor_references"]
        if reference["semantic_role"] == "recurrent_scales"
    )
    scale["resident_shape"] = [2, 3, 1]
    with pytest.raises(ValueError, match="shape"):
        contract.validate_tensor_storage_evidence(malformed)


def test_tensor_storage_metadata_uses_canonical_runtime_object_not_byte_count():
    evidence = _canonical_tensor_storage_evidence(contract.P2_PROFILE)
    metadata = evidence["snapshots"][0]["codec_metadata"]

    accounting = contract.recompute_tensor_storage_accounting(evidence)

    assert accounting["hybrid_cache_current_metadata_bytes"] == len(
        contract.canonical_json_bytes(metadata)
    )
    malicious = copy.deepcopy(evidence)
    malicious["snapshots"][0]["codec_metadata"]["byte_count"] = 1
    with pytest.raises(ValueError, match="codec metadata"):
        contract.validate_tensor_storage_evidence(malicious)


def test_tensor_storage_accounting_recomputes_logical_physical_and_metadata():
    exact_restore = _canonical_tensor_storage_evidence(
        alias_recurrent=True
    )
    recurrent_int8 = _canonical_tensor_storage_evidence(
        contract.P2_PROFILE
    )
    recurrent_int8["storages"][2]["content_sha256"] = (
        recurrent_int8["storages"][0]["content_sha256"]
    )

    exact_accounting = contract.recompute_tensor_storage_accounting(
        exact_restore
    )
    p2_accounting = contract.recompute_tensor_storage_accounting(
        recurrent_int8
    )

    assert exact_accounting[
        "hybrid_cache_current_logical_referenced_bytes"
    ] == 108
    assert exact_accounting[
        "hybrid_cache_current_unique_physical_bytes"
    ] == 40
    assert exact_accounting["hybrid_cache_current_metadata_bytes"] == 0
    assert exact_accounting["hybrid_cache_deduplicated_bytes"] == 68
    assert p2_accounting[
        "hybrid_cache_current_logical_referenced_bytes"
    ] == 108
    assert p2_accounting[
        "hybrid_cache_current_unique_physical_bytes"
    ] == 126
    assert p2_accounting["hybrid_cache_current_metadata_bytes"] == len(
        contract.canonical_json_bytes(
            recurrent_int8["snapshots"][0]["codec_metadata"]
        )
    )
    assert p2_accounting["hybrid_cache_deduplicated_bytes"] == 0


def test_tensor_storage_accounting_recomputes_middle_observation_peaks():
    evidence = _canonical_tensor_storage_evidence()
    evidence["storages"].extend([
        {
            "storage_id": "encode-workspace",
            "kind": "encode_workspace",
            "storage_nbytes": 11,
            "content_sha256": None,
        },
        {
            "storage_id": "decode-workspace",
            "kind": "decode_workspace",
            "storage_nbytes": 13,
            "content_sha256": None,
        },
    ])
    evidence["observations"] = [
        {
            "ordinal": 0,
            "event": "baseline",
            "active_snapshot_ids": [],
            "live_workspace_storage_ids": [],
            "encode_workspace_reserved_bytes": 0,
            "decode_workspace_reserved_bytes": 0,
            "cuda_allocated_bytes": 100,
            "cuda_reserved_bytes": 200,
        },
        {
            "ordinal": 1,
            "event": "encode",
            "active_snapshot_ids": ["snapshot-0"],
            "live_workspace_storage_ids": [
                "decode-workspace",
                "encode-workspace",
            ],
            "encode_workspace_reserved_bytes": 17,
            "decode_workspace_reserved_bytes": 19,
            "cuda_allocated_bytes": 900,
            "cuda_reserved_bytes": 1000,
        },
        {
            "ordinal": 2,
            "event": "final",
            "active_snapshot_ids": ["snapshot-0"],
            "live_workspace_storage_ids": [],
            "encode_workspace_reserved_bytes": 0,
            "decode_workspace_reserved_bytes": 0,
            "cuda_allocated_bytes": 300,
            "cuda_reserved_bytes": 400,
        },
    ]

    accounting = contract.recompute_tensor_storage_accounting(evidence)

    assert accounting["encode_workspace_peak_allocated_bytes"] == 11
    assert accounting["encode_workspace_peak_reserved_bytes"] == 17
    assert accounting["decode_workspace_peak_allocated_bytes"] == 13
    assert accounting["decode_workspace_peak_reserved_bytes"] == 19
    assert accounting["cuda_allocated_bytes"] == 300
    assert accounting["cuda_reserved_bytes"] == 400
    assert accounting["cuda_peak_allocated_bytes"] == 900
    assert accounting["cuda_peak_reserved_bytes"] == 1000
    assert accounting["hybrid_cache_peak_entries"] == 1
    assert accounting[
        "hybrid_cache_peak_logical_referenced_bytes"
    ] == 108


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        pytest.param(
            lambda evidence: evidence["snapshots"][0][
                "tensor_references"
            ][0].update({"storage_offset_bytes": 2}),
            "range",
            id="out-of-bounds-range",
        ),
        pytest.param(
            lambda evidence: evidence["snapshots"][0][
                "tensor_references"
            ][0].update({"storage_length_bytes": 1}),
            "cover",
            id="resident-gap",
        ),
        pytest.param(
            lambda evidence: evidence["storages"][0].update(
                {"kind": "encode_workspace", "content_sha256": None}
            ),
            "workspace",
            id="snapshot-references-workspace",
        ),
        pytest.param(
            lambda evidence: evidence["observations"][0].update({
                "live_workspace_storage_ids": [
                    evidence["storages"][0]["storage_id"]
                ]
            }),
            "workspace",
            id="observation-references-resident",
        ),
    ],
)
def test_tensor_storage_evidence_rejects_ranges_gaps_and_kind_misuse(
    mutation,
    error,
):
    evidence = _canonical_tensor_storage_evidence()
    mutation(evidence)

    with pytest.raises(ValueError, match=error):
        contract.validate_tensor_storage_evidence(evidence)


def _rehash_evidence_backed_manifests(
    manifests,
    file_inventory,
    artifact_manifest,
):
    inventory_by_path = {row["path"]: row for row in file_inventory}
    for tensor_row in manifests["tensor_inventories"]["rows"]:
        evidence = tensor_row["evidence"]
        file_row = tensor_row["file"]
        file_row["sha256"] = contract.canonical_json_file_sha256(evidence)
        file_row["bytes"] = len(contract.canonical_json_bytes(evidence)) + 1
        inventory_by_path[file_row["path"]].update(file_row)
    for kind in ("snapshots", "tensor_inventories"):
        path = contract.NESTED_MANIFEST_ARTIFACT_PATHS[kind]
        entry = next(
            row
            for row in artifact_manifest["entries"]
            if row["path"] == path
        )
        entry["sha256"] = contract.canonical_json_file_sha256(
            manifests[kind]
        )
        entry["bytes"] = len(
            contract.canonical_json_bytes(manifests[kind])
        ) + 1


def test_coordinated_current_total_rewrite_and_full_rehash_is_rejected():
    process_rows, manifests, file_inventory, artifact_manifest = (
        _evidence_backed_snapshot_collection()
    )
    contract.validate_process_row_nested_snapshot_bindings(
        process_rows,
        manifests,
    )
    target_index = next(
        index
        for index, row in enumerate(process_rows)
        if row["profile"] == contract.P2_PROFILE
    )
    process_row = process_rows[target_index]
    snapshot_row = manifests["snapshots"]["rows"][target_index]
    process_row["hybrid_cache_current_logical_referenced_bytes"] += 1
    process_row["hybrid_cache_current_unique_physical_bytes"] += 1
    process_row["hybrid_cache_peak_logical_referenced_bytes"] += 1
    process_row["hybrid_cache_peak_unique_physical_bytes"] += 1
    snapshot_row["hybrid_cache_current_logical_referenced_bytes"] += 1
    snapshot_row["hybrid_cache_current_unique_physical_bytes"] += 1
    snapshot_row["hybrid_cache_peak_logical_referenced_bytes"] += 1
    snapshot_row["hybrid_cache_peak_unique_physical_bytes"] += 1
    snapshot_row["full_fidelity_logical_bytes"] = snapshot_row[
        "hybrid_cache_current_logical_referenced_bytes"
    ]
    snapshot_row["encoded_physical_bytes"] = snapshot_row[
        "hybrid_cache_current_unique_physical_bytes"
    ]
    _rehash_evidence_backed_manifests(
        manifests,
        file_inventory,
        artifact_manifest,
    )

    with pytest.raises(ValueError, match="accounting|evidence|recomputed"):
        contract.validate_process_row_nested_snapshot_bindings(
            process_rows,
            manifests,
        )


def test_coordinated_peak_workspace_cuda_rewrite_and_full_rehash_is_rejected():
    process_rows, manifests, file_inventory, artifact_manifest = (
        _evidence_backed_snapshot_collection()
    )
    contract.validate_process_row_nested_snapshot_bindings(
        process_rows,
        manifests,
    )
    target_index = next(
        index
        for index, row in enumerate(process_rows)
        if row["profile"] == contract.P1_REFERENCE_PROFILE
    )
    process_row = process_rows[target_index]
    snapshot_row = manifests["snapshots"]["rows"][target_index]
    for field in (
        "hybrid_cache_peak_logical_referenced_bytes",
        "encode_workspace_peak_reserved_bytes",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
    ):
        process_row[field] += 1
        snapshot_row[field] += 1
    _rehash_evidence_backed_manifests(
        manifests,
        file_inventory,
        artifact_manifest,
    )

    with pytest.raises(ValueError, match="accounting|evidence|recomputed"):
        contract.validate_process_row_nested_snapshot_bindings(
            process_rows,
            manifests,
        )


def test_process_row_nested_snapshot_bindings_reject_profile_mismatch():
    process_rows, manifests, file_inventory, artifact_manifest = (
        _joint_process_row_nested_snapshots()
    )
    contract.validate_process_rows(process_rows)
    contract.validate_nested_evidence_bundle(
        manifests,
        file_inventory,
        artifact_manifest,
    )
    manifests["snapshots"]["rows"][0]["profile"] = "exact_restore"

    with pytest.raises(
        ValueError,
        match="profile|snapshot|binding|evidence",
    ):
        contract.validate_process_row_nested_snapshot_bindings(
            process_rows,
            manifests,
        )


def test_process_row_nested_snapshot_bindings_reject_missing_row():
    process_rows, manifests, file_inventory, artifact_manifest = (
        _joint_process_row_nested_snapshots()
    )
    contract.validate_process_rows(process_rows)
    contract.validate_nested_evidence_bundle(
        manifests,
        file_inventory,
        artifact_manifest,
    )
    manifests["snapshots"]["rows"].pop()

    with pytest.raises(
        ValueError,
        match="snapshot|coverage|binding|evidence",
    ):
        contract.validate_process_row_nested_snapshot_bindings(
            process_rows,
            manifests,
        )


def test_process_row_nested_snapshot_bindings_reject_duplicate_row():
    process_rows, manifests, file_inventory, artifact_manifest = (
        _joint_process_row_nested_snapshots()
    )
    contract.validate_process_rows(process_rows)
    contract.validate_nested_evidence_bundle(
        manifests,
        file_inventory,
        artifact_manifest,
    )
    manifests["snapshots"]["rows"].append(
        json.loads(json.dumps(manifests["snapshots"]["rows"][-1]))
    )

    with pytest.raises(
        ValueError,
        match="snapshot|duplicate|binding|evidence",
    ):
        contract.validate_process_row_nested_snapshot_bindings(
            process_rows,
            manifests,
        )


def test_process_row_nested_snapshot_bindings_reject_cross_key_reorder():
    process_rows, manifests, file_inventory, artifact_manifest = (
        _joint_process_row_nested_snapshots()
    )
    contract.validate_process_rows(process_rows)
    contract.validate_nested_evidence_bundle(
        manifests,
        file_inventory,
        artifact_manifest,
    )
    snapshots = manifests["snapshots"]["rows"]
    snapshots[0], snapshots[4] = snapshots[4], snapshots[0]

    with pytest.raises(
        ValueError,
        match="case|profile|rank|snapshot|binding|evidence",
    ):
        contract.validate_process_row_nested_snapshot_bindings(
            process_rows,
            manifests,
        )


def test_process_row_nested_snapshot_bindings_cover_last_rank():
    process_rows, manifests, file_inventory, artifact_manifest = (
        _joint_process_row_nested_snapshots()
    )
    contract.validate_process_rows(process_rows)
    contract.validate_nested_evidence_bundle(
        manifests,
        file_inventory,
        artifact_manifest,
    )
    assert (
        process_rows[-1]["case_id"],
        process_rows[-1]["profile"],
        process_rows[-1]["rank"],
        process_rows[-1]["world_size"],
    ) == (
        "w4_miss_invalidation__measured__r4__recompute",
        "recompute",
        3,
        4,
    )
    manifests["snapshots"]["rows"][-1]["rank"] = 2

    with pytest.raises(
        ValueError,
        match="rank|snapshot|binding|evidence",
    ):
        contract.validate_process_row_nested_snapshot_bindings(
            process_rows,
            manifests,
        )


def test_validate_artifact_evidence_accepts_canonical_composition():
    (
        case_rows,
        process_rows,
        manifests,
        file_inventory,
        artifact_manifest,
    ) = _canonical_artifact_evidence()
    contract.validate_case_row_nested_evidence_bindings(
        case_rows,
        manifests,
    )
    contract.validate_process_row_nested_worker_log_bindings(
        process_rows,
        manifests,
    )
    contract.validate_process_row_nested_snapshot_bindings(
        process_rows,
        manifests,
    )
    contract.validate_nested_evidence_bundle(
        manifests,
        file_inventory,
        artifact_manifest,
    )

    contract.validate_artifact_evidence(
        case_rows,
        process_rows,
        manifests,
        file_inventory,
        artifact_manifest,
    )


def test_validate_artifact_evidence_accepts_canonical_process_binding_oracle():
    (
        case_rows,
        process_rows,
        manifests,
        file_inventory,
        artifact_manifest,
    ) = _canonical_artifact_evidence()

    contract.validate_artifact_evidence(
        case_rows,
        process_rows,
        manifests,
        file_inventory,
        artifact_manifest,
    )


@pytest.mark.parametrize(
    ("field", "mutated_value"),
    [
        pytest.param("kv_capacity_bytes", 2, id="kv-capacity-bytes"),
        pytest.param("source_tree_sha256", "f" * 64, id="source-tree"),
        pytest.param("gate1_audit_sha256", "f" * 64, id="gate1-audit"),
        pytest.param(
            "execution_plan_sha256",
            "f" * 64,
            id="execution-plan",
        ),
        pytest.param(
            "source_bundle_sha256",
            "f" * 64,
            id="source-bundle",
        ),
        pytest.param(
            "source_package_sha256",
            "f" * 64,
            id="source-package",
        ),
        pytest.param(
            "producer_source_sha256",
            "f" * 64,
            id="producer-source",
        ),
        pytest.param(
            "producer_version_sha256",
            "f" * 64,
            id="producer-version",
        ),
        pytest.param(
            "verifier_source_sha256",
            "f" * 64,
            id="verifier-source",
        ),
        pytest.param(
            "verifier_version_sha256",
            "f" * 64,
            id="verifier-version",
        ),
    ],
)
def test_validate_artifact_evidence_rejects_process_side_binding_drift(
    field,
    mutated_value,
):
    (
        case_rows,
        process_rows,
        manifests,
        file_inventory,
        artifact_manifest,
    ) = _canonical_artifact_evidence()
    process_rows[0][field] = mutated_value

    contract.validate_process_rows(process_rows)

    with pytest.raises(
        ValueError,
        match=rf"case|process|binding|{field}",
    ):
        contract.validate_artifact_evidence(
            case_rows,
            process_rows,
            manifests,
            file_inventory,
            artifact_manifest,
        )


def test_validate_artifact_evidence_rejects_raw_case_prompt_hash_mismatch():
    (
        case_rows,
        process_rows,
        manifests,
        file_inventory,
        artifact_manifest,
    ) = _canonical_artifact_evidence()
    contract.validate_case_rows(case_rows)
    contract.validate_process_rows(process_rows)
    contract.validate_nested_evidence_bundle(
        manifests,
        file_inventory,
        artifact_manifest,
    )
    case_rows[0]["prompt_token_ids_sha256"] = "0" * 64
    contract.validate_case_rows(case_rows)

    with pytest.raises(ValueError, match="prompt|binding|evidence"):
        contract.validate_artifact_evidence(
            case_rows,
            process_rows,
            manifests,
            file_inventory,
            artifact_manifest,
        )


def test_validate_artifact_evidence_rejects_process_snapshot_identity_mismatch():
    (
        case_rows,
        process_rows,
        manifests,
        file_inventory,
        artifact_manifest,
    ) = _canonical_artifact_evidence()
    contract.validate_case_row_nested_evidence_bindings(
        case_rows,
        manifests,
    )
    contract.validate_process_row_nested_snapshot_bindings(
        process_rows,
        manifests,
    )
    contract.validate_nested_evidence_bundle(
        manifests,
        file_inventory,
        artifact_manifest,
    )
    manifests["snapshots"]["rows"][0]["profile"] = "exact_restore"

    with pytest.raises(
        ValueError,
        match="snapshot|profile|binding|evidence",
    ):
        contract.validate_artifact_evidence(
            case_rows,
            process_rows,
            manifests,
            file_inventory,
            artifact_manifest,
        )


def test_validate_artifact_evidence_composes_nested_inventory_rejection():
    (
        case_rows,
        process_rows,
        manifests,
        file_inventory,
        artifact_manifest,
    ) = _canonical_artifact_evidence()
    contract.validate_case_row_nested_evidence_bindings(
        case_rows,
        manifests,
    )
    contract.validate_process_row_nested_worker_log_bindings(
        process_rows,
        manifests,
    )
    contract.validate_process_row_nested_snapshot_bindings(
        process_rows,
        manifests,
    )
    contract.validate_nested_evidence_bundle(
        manifests,
        file_inventory,
        artifact_manifest,
    )
    file_inventory[0]["type"] = "symlink"
    with pytest.raises(ValueError, match="type|symlink"):
        contract.validate_nested_evidence_bundle(
            manifests,
            file_inventory,
            artifact_manifest,
        )

    with pytest.raises(ValueError, match="type|symlink|inventory|evidence"):
        contract.validate_artifact_evidence(
            case_rows,
            process_rows,
            manifests,
            file_inventory,
            artifact_manifest,
        )


def test_classify_run_uses_exact_fail_closed_precedence():
    passing = {
        "artifact_invalid": False,
        "resources_blocked": False,
        "correctness_pass": True,
        "runtime_safety_pass": True,
        "cache_pass": True,
        "performance_pass": True,
    }
    assert contract.classify_run(passing) == "GO"

    expected = (
        ("artifact_invalid", "INVALID_ARTIFACT"),
        ("resources_blocked", "BLOCKED_RESOURCES"),
        ("correctness_pass", "NO_GO_CORRECTNESS"),
        ("runtime_safety_pass", "NO_GO_RUNTIME_SAFETY"),
        ("cache_pass", "NO_GO_CACHE"),
        ("performance_pass", "NO_GO_PERFORMANCE"),
    )
    for field, result in expected:
        metrics = dict(passing)
        metrics[field] = (
            True
            if field in {
                "artifact_invalid",
                "resources_blocked",
            }
            else False
        )
        assert contract.classify_run(metrics) == result

    metrics = {
        field: (
            True
            if field in {
                "artifact_invalid",
                "resources_blocked",
            }
            else False
        )
        for field in passing
    }
    assert contract.classify_run(metrics) == "INVALID_ARTIFACT"

    metrics = dict(passing)
    metrics.update(
        {
            "resources_blocked": True,
            "correctness_pass": False,
            "runtime_safety_pass": False,
            "cache_pass": False,
            "performance_pass": False,
        }
    )
    assert contract.classify_run(metrics) == "BLOCKED_RESOURCES"

    metrics = dict(passing)
    metrics.update(
        {
            "correctness_pass": False,
            "runtime_safety_pass": False,
            "cache_pass": False,
            "performance_pass": False,
        }
    )
    assert contract.classify_run(metrics) == "NO_GO_CORRECTNESS"

    metrics = dict(passing)
    metrics.update(
        {
            "runtime_safety_pass": False,
            "cache_pass": False,
            "performance_pass": False,
        }
    )
    assert contract.classify_run(metrics) == "NO_GO_RUNTIME_SAFETY"


def test_p2_is_default_off_and_schema_v1_profiles_remain_unchanged():
    assert _class_literal_assignment(
        RUNTIME_CONFIG_PATH,
        "Config",
        "qwen35_hybrid_prefix_representation",
    ) == "exact_restore"
    assert _literal_assignment(
        SCHEMA_V1_CONTRACT_PATH,
        "POLICIES",
    ) == ("recompute", "exact_restore")


def test_classify_run_rejects_missing_unknown_and_non_boolean_metrics():
    passing = {
        "artifact_invalid": False,
        "resources_blocked": False,
        "correctness_pass": True,
        "runtime_safety_pass": True,
        "cache_pass": True,
        "performance_pass": True,
    }
    for field in tuple(passing):
        malformed = dict(passing)
        malformed.pop(field)
        assert contract.classify_run(malformed) == "INVALID_ARTIFACT"

    malformed = dict(passing)
    malformed["unexpected"] = True
    assert contract.classify_run(malformed) == "INVALID_ARTIFACT"

    for field in tuple(passing):
        malformed = dict(passing)
        malformed[field] = 1
        assert contract.classify_run(malformed) == "INVALID_ARTIFACT"


@pytest.mark.parametrize(
    ("name", "module", "run_tag"),
    [
        (
            "engine_correctness",
            native_engine_plan_test,
            "engine-run-tag",
        ),
        (
            "cached_continuation",
            native_cached_plan_test,
            "cached-run-tag",
        ),
    ],
)
def test_workload_command_validator_accepts_native_v1_plan(
    name,
    module,
    run_tag,
):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        repo, configuration, inventory = module._fixture(root)
        output = root / "native-plan"
        plan = module.planner.build_remote_execution_plan(
            repo_root=repo,
            configuration_path=configuration,
            source_inventory_path=inventory,
            output_dir=output,
            run_tag=run_tag,
            remote_model_dir="/remote/models/qwen35",
            remote_model_manifest=(
                "/remote/models/qwen35/model_manifest.json"
            ),
        )
        assert module.planner.verify_remote_execution_plan(
            output / module.planner.PLAN_NAME
        ) == plan
        contract._validate_workload_prerequisite_commands(name, plan)


def _run():
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(Path(__file__).resolve()),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(_run())
