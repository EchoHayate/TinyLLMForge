from __future__ import annotations

import hashlib
from dataclasses import dataclass
import json
import math
import os
import shlex
import stat
from collections.abc import Mapping
from pathlib import Path


SCHEMA_VERSION = "qwen35.tp4-hybrid-prefix-performance-cache.v2"
PREREQUISITE_SCHEMA_VERSION = (
    "qwen35.tp4-performance-prerequisites.v2"
)
PREREQUISITE_PROVENANCE_SCHEMA_VERSION = (
    "qwen35.tp4-performance-prerequisite-provenance.v1"
)
MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38f"
    "e69982efedaf4a3e631ad0b14aad7dd0"
)
TP4_ROOT_SOURCE_TREE_SHA256 = (
    "37135279047a569df8e0d26c6e396472"
    "02b27aca758ac8ac135bdab25612f20a"
)
TP4_ROOT_CORRECTNESS_SCHEMA_VERSION = (
    "qwen35.tp4-real-root-logit-correctness.v1"
)
CACHED_CONTINUATION_SCHEMA_VERSION = (
    "qwen35.tp4-cached-continuation-correctness.v1"
)
ENGINE_CORRECTNESS_SCHEMA_VERSION = (
    "qwen35.tp4-engine-model-runner-correctness.v1"
)
TP4_ROOT_CASE_IDS = ("p17", "p65", "synthetic")
ENGINE_CORRECTNESS_SCENARIOS = {
    "construct_and_bind": (0, 0, 0, 0, 0, 0, 0, 0),
    "publish_source": (1, 1, 1, 1, 0, 0, 1, 1),
    "restore_w1": (64, 64, 64, 0, 1, 0, 0, 1),
    "miss_w4_token": (33, 33, 32, 1, 0, 0, 0, 2),
    "miss_w4_stale": (33, 33, 32, 1, 0, 1, 0, 1),
    "miss_w4_clear": (33, 33, 32, 1, 0, 1, 0, 1),
}
APPROVED_V1_WORKLOAD_MANIFEST_SHA256 = (
    "71909b825d1a8d162604f6cc3d34ad4"
    "13b2af6c191425ec007859715a4d084e3"
)
PREREQUISITE_NAMES = (
    "tp4_root_logit",
    "cached_continuation",
    "engine_correctness",
)
PREREQUISITE_ROW_FIELDS = (
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
PROFILES = (
    "recompute",
    "exact_restore",
    "recurrent_int8_per_row",
)
RESULTS = (
    "GO",
    "NO_GO_CORRECTNESS",
    "NO_GO_RUNTIME_SAFETY",
    "NO_GO_CACHE",
    "NO_GO_PERFORMANCE",
    "BLOCKED_RESOURCES",
    "INVALID_ARTIFACT",
)
LOGIT_TOLERANCE = {"atol": 2e-5, "rtol": 0.0}
THRESHOLDS = {
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

WORKLOADS = (
    "w0_short_control",
    "w1_medium_reuse",
    "w2_long_reuse",
    "w3_batched_fanout",
    "w4_miss_invalidation",
)
WORKLOAD_SPECS = {
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
WARMUP_REPETITIONS = 1
CORRECTNESS_REPETITIONS = 1
MEASURED_REPETITIONS = 5
CORRECTNESS_CONCURRENCY = 1
WORLD_SIZE = 4
MIN_GPU_FREE_BYTES = 24 * 1024**3
MAX_MODEL_LEN = 4096
TOKEN_ID_UPPER_BOUND = 32000
MODEL_VOCAB_SIZE = 248320
SAMPLING_TEMPERATURE = 0.0
SAMPLING_IGNORE_EOS = True
HYBRID_PREFIX_MAX_ENTRIES = 16
HYBRID_PREFIX_MAX_BYTES = 2 * 1024**3
REQUIRED_GPU_INDICES = (2, 4, 5, 6)
DIRTY_TREE_POLICIES = ("reject_dirty",)

P2_PROFILE = "recurrent_int8_per_row"
P2_REPRESENTATION = "recurrent_int8_per_row"
P2_REPRESENTATION_VERSION = "qwen35_hybrid_prefix_recurrent_int8_v1"
P2_CODEC_ID = "qwen35_recurrent_symmetric_int8_per_row_v1"
P1_REFERENCE_PROFILE = "exact_restore"
P1_REPRESENTATION_VERSION = "qwen35_hybrid_prefix_exact_v1"
P2_NUMERICAL_REFERENCE_PROFILE = P1_REFERENCE_PROFILE
P2_CACHE_COST_REFERENCE_PROFILE = P1_REFERENCE_PROFILE
CALIBRATION_SCHEMA_VERSION = "qwen35.recurrent-int8-calibration.v1"
P1_AUTHORITY_SCHEMA_VERSION = (
    "qwen35.tp4-hybrid-prefix-performance-cache.v1"
)
P2_REQUIRED_BINDINGS = (
    "codec",
    "representation",
    "representation_version",
    "calibration_artifact_sha256",
    "p1_authority_artifact_sha256",
)

TOP_LEVEL_ARTIFACTS = (
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
NESTED_ARTIFACT_DIRECTORIES = (
    "prerequisites",
    "snapshots",
    "receipts",
    "source",
    "tokens",
    "logits",
    "logs",
    "verifier",
)
ARTIFACT_MANIFEST_HASH_DOMAIN = (
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
PRODUCER_TRUST_DOMAIN = (
    "artifact_manifest.json",
    *ARTIFACT_MANIFEST_HASH_DOMAIN,
)
VERIFIER_TRUST_DOMAIN = (
    "local_verifier_output.json",
    "remote_verifier_output.json",
    "independent_verification.json",
    "report.md",
)
MANIFEST_ENTRY_FIELDS = (
    "path",
    "sha256",
    "bytes",
    "producer",
    "trust_domain",
)
SNAPSHOT_INVENTORY_FIELDS = (
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
RECEIPT_BINDING_FIELDS = (
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
SNAPSHOT_INVENTORY_SCHEMA_VERSION = (
    "qwen35.tp4-hybrid-prefix-snapshot-inventory.v1"
)
RECEIPT_BINDING_SCHEMA_VERSION = (
    "qwen35.tp4-hybrid-prefix-remote-receipt-binding.v1"
)
SOURCE_MANIFEST_SCHEMA_VERSION = (
    "qwen35.tp4-hybrid-prefix-source-manifest.v1"
)
MATCHED_CONFIGURATION_SCHEMA_VERSION = (
    "qwen35.tp4-hybrid-prefix-matched-configuration.v1"
)
SOURCE_MANIFEST_FIELDS = (
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
MATCHED_CONFIGURATION_FIELDS = (
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
ARTIFACT_MANIFEST_SCHEMA_VERSION = (
    "qwen35.tp4-hybrid-prefix-artifact-manifest.v2"
)
EXECUTION_COMMAND_ORDER = (
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
)
MAX_BOUNDED_OUTPUT_BYTES = 64 * 1024
RESERVE_REMOTE_TIMEOUT_SECONDS = 60
UPLOAD_TIMEOUT_SECONDS = 300
STAGE_TIMEOUT_SECONDS = 600
RESOURCE_GUARD_TIMEOUT_SECONDS = 120
WORKERS_TIMEOUT_SECONDS = 3600
ASSEMBLY_TIMEOUT_SECONDS = 600
REMOTE_VERIFY_TIMEOUT_SECONDS = 600
FINAL_RESOURCE_GUARD_TIMEOUT_SECONDS = 120
PACKAGE_DOWNLOAD_TIMEOUT_SECONDS = 1800
SAFE_EXTRACT_TIMEOUT_SECONDS = 1800
LOCAL_VERIFY_TIMEOUT_SECONDS = 600
EXECUTION_SSH_TARGET = "sitian@10.232.195.203"
EXECUTION_SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ControlMaster=no",
    "-o",
    "ControlPath=none",
    "-o",
    "ConnectTimeout=20",
)
EXECUTION_ENV = {
    "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian",
}
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
)
EXECUTION_COMMAND_TIMEOUT_SECONDS = {
    "reserve_remote": RESERVE_REMOTE_TIMEOUT_SECONDS,
    "upload": UPLOAD_TIMEOUT_SECONDS,
    "stage": STAGE_TIMEOUT_SECONDS,
    "resource_guard": RESOURCE_GUARD_TIMEOUT_SECONDS,
    "workers": WORKERS_TIMEOUT_SECONDS,
    "assembly": ASSEMBLY_TIMEOUT_SECONDS,
    "remote_verify": REMOTE_VERIFY_TIMEOUT_SECONDS,
    "final_resource_guard": FINAL_RESOURCE_GUARD_TIMEOUT_SECONDS,
    "package_download": PACKAGE_DOWNLOAD_TIMEOUT_SECONDS,
    "safe_extract": SAFE_EXTRACT_TIMEOUT_SECONDS,
    "local_verify": LOCAL_VERIFY_TIMEOUT_SECONDS,
}
EXECUTION_PROVENANCE_FIELDS = (
    "source_tree_sha256",
    "model_manifest_sha256",
    "workload_manifest_sha256",
    "correctness_prerequisites_sha256",
    "calibration_artifact_sha256",
    "p1_authority_artifact_sha256",
    "gate1_audit_sha256",
    "source_bundle_sha256",
    "source_package_sha256",
    "command_manifest_sha256",
    "producer_source_sha256",
    "producer_version_sha256",
    "verifier_source_sha256",
    "verifier_version_sha256",
)
GPU_ASSIGNMENT_FIELDS = (
    "rank",
    "gpu_index",
    "cuda_visible_device",
)
CASE_PORT_PAIR_FIELDS = (
    "case_id",
    "tinyvllm_dist_port",
    "master_port",
)
ARTIFACT_PATH_FIELDS = (
    "remote_run",
    "remote_artifact",
    "package",
    "local_extract",
)
COMMAND_MANIFEST_ROW_FIELDS = (
    "name",
    "command_sha256",
)
COMMAND_RESULT_FIELDS = (
    "name",
    "command_sha256",
    "outcome",
    "returncode",
    "stdout",
    "stderr",
    "stdout_truncated",
    "stderr_truncated",
)
SOURCE_INVENTORY_ROW_FIELDS = (
    "path",
    "sha256",
    "bytes",
    "type",
)
GPU_RESOURCE_ROW_FIELDS = (
    "gpu_index",
    "gpu_uuid",
    "free_bytes",
    "compute_processes",
)
EXECUTION_LIFECYCLE_STATES = (
    "preflight_blocked",
    "execution_success",
    "execution_failed",
)
EXECUTION_BUNDLE_DOCUMENTS = {
    "preflight_blocked": (
        "lifecycle_state",
        "environment",
        "gpu_assignments",
        "preflight",
    ),
    "execution_success": (
        "lifecycle_state",
        "environment",
        "gpu_assignments",
        "commands",
        "preflight",
        "execution_plan",
        "consumed_authorization",
        "source_bundle",
        "source_package",
        "resource_guard_before",
        "resource_guard_after",
        "execution_receipt",
        "local_verifier_output",
        "remote_verifier_output",
        "independent_verification",
    ),
    "execution_failed": (
        "lifecycle_state",
        "environment",
        "gpu_assignments",
        "commands",
        "preflight",
        "execution_plan",
        "consumed_authorization",
        "source_bundle",
        "source_package",
        "resource_guard_before",
        "resource_guard_after",
        "execution_receipt",
    ),
}
NESTED_MANIFEST_KINDS = (
    "prerequisites",
    "tokens",
    "logits",
    "logs",
    "snapshots",
    "tensor_inventories",
)
NESTED_MANIFEST_FIELDS = (
    "schema_version",
    "kind",
    "files",
    "rows",
)
NESTED_FILE_FIELDS = (
    "path",
    "sha256",
    "bytes",
    "type",
)
NESTED_ROW_FIELDS = {
    "prerequisites": (
        "name",
        "role",
        "file",
    ),
    "tokens": (
        "case_id",
        "request_id",
        "role",
        "token_count",
        "file",
    ),
    "logits": (
        "case_id",
        "request_id",
        "shape",
        "dtype",
        "file",
    ),
    "logs": (
        "case_id",
        "rank",
        "world_size",
        "completion_marker",
        "traceback_present",
        "file",
    ),
    "snapshots": (
        "case_id",
        "profile",
        "rank",
        "world_size",
        "evidence_kind",
        "snapshot_file",
        "tensor_inventory_file",
        "full_fidelity_logical_bytes",
        "encoded_physical_bytes",
        "codec_metadata_bytes",
        "temporary_encode_workspace_bytes",
        "temporary_decode_workspace_bytes",
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
    ),
    "tensor_inventories": (
        "case_id",
        "profile",
        "representation",
        "representation_version",
        "codec",
        "rank",
        "world_size",
        "evidence_schema_version",
        "snapshot_count",
        "storage_count",
        "reference_count",
        "observation_count",
        "evidence",
        "file",
    ),
}
TENSOR_STORAGE_EVIDENCE_SCHEMA_VERSION = (
    "qwen35.tp4-hybrid-prefix-tensor-storage-evidence.v1"
)
TENSOR_STORAGE_EVIDENCE_FIELDS = (
    "schema_version",
    "case_id",
    "profile",
    "representation",
    "representation_version",
    "codec",
    "rank",
    "world_size",
    "snapshots",
    "storages",
    "observations",
)
TENSOR_STORAGE_SNAPSHOT_FIELDS = (
    "snapshot_id",
    "tensor_references",
    "codec_metadata",
)
TENSOR_STORAGE_REFERENCE_FIELDS = (
    "reference_id",
    "layer_index",
    "semantic_role",
    "logical_dtype",
    "logical_shape",
    "resident_dtype",
    "resident_shape",
    "storage_id",
    "storage_offset_bytes",
    "storage_length_bytes",
)
TENSOR_STORAGE_STORAGE_FIELDS = (
    "storage_id",
    "kind",
    "storage_nbytes",
    "content_sha256",
)
TENSOR_STORAGE_OBSERVATION_FIELDS = (
    "ordinal",
    "event",
    "active_snapshot_ids",
    "live_workspace_storage_ids",
    "encode_workspace_reserved_bytes",
    "decode_workspace_reserved_bytes",
    "cuda_allocated_bytes",
    "cuda_reserved_bytes",
)
TENSOR_STORAGE_ACCOUNTING_FIELDS = (
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
NESTED_MANIFEST_SCHEMA_VERSIONS = {
    kind: (
        "qwen35.tp4-hybrid-prefix-"
        f"{kind.replace('_', '-')}-manifest.v1"
    )
    for kind in NESTED_MANIFEST_KINDS
}
NESTED_MANIFEST_ARTIFACT_PATHS = {
    "prerequisites": "correctness_prerequisites.json",
    "tokens": "token_manifest.json",
    "logits": "logits_manifest.json",
    "logs": "worker_logs_manifest.json",
    "snapshots": "snapshot_manifest.json",
    "tensor_inventories": "tensor_inventory_manifest.json",
}
EVIDENCE_DOCUMENT_FIELDS = {
    "environment": (
        "schema_version",
        "run_tag",
        "nonce",
        *EXECUTION_PROVENANCE_FIELDS,
        "dirty_tree_policy",
    ),
    "gpu_assignments": (
        "schema_version",
        "run_tag",
        "required_gpu_indices",
        "world_size",
        "assignments",
    ),
    "commands": (
        "schema_version",
        "run_tag",
        "nonce",
        "execution_plan_sha256",
        "command_manifest_sha256",
        "command_order",
        "commands",
    ),
    "gate1_audit": (
        "schema_version",
        "classification",
        "source_tree_sha256",
        "gate1_audit_sha256",
        "checks",
    ),
    "preflight": (
        "schema_version",
        "classification",
        "run_tag",
        "nonce",
        *EXECUTION_PROVENANCE_FIELDS,
        "required_gpu_indices",
        "world_size",
        "minimum_free_bytes_per_gpu",
        "gpu_query_rows",
        "blocking_reasons",
        "worker_authorized",
        "remote_path_created",
        "source_staged",
        "worker_launched",
    ),
    "execution_plan": (
        "schema_version",
        "run_tag",
        "nonce",
        *EXECUTION_PROVENANCE_FIELDS,
        "authority_root_sha256",
        "physical_artifact_root_sha256",
        "required_gpu_indices",
        "world_size",
        "gpu_assignments",
        "case_port_pairs",
        "artifact_paths",
        "command_order",
    ),
    "consumed_authorization": (
        "schema_version",
        "run_tag",
        "nonce",
        *EXECUTION_PROVENANCE_FIELDS,
        "execution_plan_sha256",
        "required_gpu_indices",
        "world_size",
        "gpu_assignments",
        "case_port_pairs",
        "artifact_paths",
        "authorization_id",
        "active_path",
        "consumed_path",
        "consumed",
        "consumed_once",
    ),
    "execution_receipt": (
        "schema_version",
        "classification",
        "run_tag",
        "nonce",
        *EXECUTION_PROVENANCE_FIELDS,
        "execution_plan_sha256",
        "consumed_authorization_sha256",
        "authorization_id",
        "command_order",
        "command_results",
        "artifact_paths",
        "source_inventory",
        "package_inventory",
        "final_inventory",
        "package_inventory_sha256",
        "final_inventory_sha256",
        "resource_guard_before_sha256",
        "resource_guard_after_sha256",
        "remote_path_created",
        "source_staged",
        "worker_launched",
        "cleanup_complete",
    ),
    "source_bundle": (
        "schema_version",
        "run_tag",
        "nonce",
        *EXECUTION_PROVENANCE_FIELDS,
        "dirty_tree_policy",
        "path",
        "sha256",
        "inventory_path",
        "inventory_sha256",
        "inventory",
    ),
    "source_package": (
        "schema_version",
        "run_tag",
        "nonce",
        *EXECUTION_PROVENANCE_FIELDS,
        "path",
        "sha256",
        "inventory_path",
        "inventory_sha256",
        "inventory",
    ),
    "resource_guard": (
        "schema_version",
        "run_tag",
        "phase",
        "required_gpu_indices",
        "minimum_free_bytes_per_gpu",
        "sha256",
        "gpu_query_rows",
        "side_effects_observed",
    ),
    "verifier_output": (
        "schema_version",
        "classification",
        "role",
        "artifact_manifest_sha256",
        "verifier_source_sha256",
        "verifier_version_sha256",
        "checks",
    ),
    "independent_verification": (
        "schema_version",
        "classification",
        "artifact_manifest_sha256",
        "local_verifier_sha256",
        "remote_verifier_sha256",
        "local_verifier_role",
        "remote_verifier_role",
        "checks",
    ),
}
EVIDENCE_SCHEMA_VERSIONS = {
    kind: f"qwen35.tp4-hybrid-prefix-{kind.replace('_', '-')}.v1"
    for kind in EVIDENCE_DOCUMENT_FIELDS
}

CASE_ROW_FIELDS = (
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
PROCESS_ROW_FIELDS = (
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
SHARED_CASE_PROCESS_FIELDS = (
    "case_id",
    "profile",
    "representation",
    "representation_version",
    "codec",
    "workload",
    "phase",
    "repetition",
    "sampling_temperature",
    "sampling_max_tokens",
    "sampling_ignore_eos",
    "sampling_seed",
    "concurrency",
    "gpu_indices",
    "kv_capacity_bytes",
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
)
CALIBRATION_BINDING_FIELDS = (
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
P1_AUTHORITY_BINDING_FIELDS = (
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
CLASSIFICATION_FIELDS = (
    "artifact_invalid",
    "resources_blocked",
    "correctness_pass",
    "runtime_safety_pass",
    "cache_pass",
    "performance_pass",
)


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    workload: str
    profile: str
    phase: str
    repetition: int
    concurrency: int
    strict_correctness: bool


@dataclass(frozen=True)
class PrerequisiteStatus:
    classification: str
    authorized: bool
    reasons: tuple[str, ...]


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def open_physical_directory(path: object) -> tuple[int, Path]:
    value = Path(path)
    if value.is_symlink():
        raise ValueError("physical directory must not be a symlink")
    try:
        resolved = value.resolve(strict=True)
    except OSError as error:
        raise ValueError("physical directory is invalid") from error
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(resolved, flags)
    except OSError as error:
        raise ValueError("physical directory is invalid") from error
    try:
        metadata = os.fstat(descriptor)
        current = resolved.stat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino)
            != (current.st_dev, current.st_ino)
        ):
            raise ValueError("physical directory is invalid")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, resolved


def physical_directory_fd_sha256(
    descriptor: int,
    resolved_path: object,
) -> str:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("physical directory is invalid")
    return canonical_json_sha256(
        {
            "resolved_path": str(Path(resolved_path)),
            "st_dev": metadata.st_dev,
            "st_ino": metadata.st_ino,
        }
    )


def physical_directory_sha256(path: object) -> str:
    descriptor, resolved = open_physical_directory(path)
    try:
        return physical_directory_fd_sha256(descriptor, resolved)
    finally:
        os.close(descriptor)


def execution_command_sha256(command: object) -> str:
    return canonical_json_sha256(command)


def canonical_execution_commands(
    execution_plan: object,
) -> dict[str, dict[str, object]]:
    if not isinstance(execution_plan, Mapping):
        raise ValueError("execution plan must be a mapping")
    artifact_paths = execution_plan["artifact_paths"]
    gpu_assignments = execution_plan["gpu_assignments"]
    case_port_pairs = execution_plan["case_port_pairs"]
    provenance = {
        field: execution_plan[field]
        for field in EXECUTION_PROVENANCE_FIELDS
        if field != "command_manifest_sha256"
    }
    common = {
        "timeout_seconds": None,
        "stdout_limit_bytes": MAX_BOUNDED_OUTPUT_BYTES,
        "stderr_limit_bytes": MAX_BOUNDED_OUTPUT_BYTES,
    }

    def command(name: str, **payload: object) -> dict[str, object]:
        return {
            **payload,
            **common,
            "timeout_seconds": EXECUTION_COMMAND_TIMEOUT_SECONDS[name],
        }

    remote_run = artifact_paths["remote_run"]
    remote_source = f"{remote_run}/source"
    remote_common = {
        "ssh_target": EXECUTION_SSH_TARGET,
        "ssh_options": list(EXECUTION_SSH_OPTIONS),
        "execution_env": dict(EXECUTION_ENV),
        "remote_python": REMOTE_PYTHON,
    }

    def remote_command(
        name: str,
        **payload: object,
    ) -> dict[str, object]:
        return command(name, **payload, **remote_common)

    worker_commands = [
        {
            "case_id": row["case_id"],
            "tinyvllm_dist_port": row["tinyvllm_dist_port"],
            "master_port": row["master_port"],
            "cwd": remote_source,
            "env": {
                "CUDA_VISIBLE_DEVICES": ",".join(
                    str(index)
                    for index in execution_plan["required_gpu_indices"]
                ),
                "TINYVLLM_DIST_PORT": str(
                    row["tinyvllm_dist_port"]
                ),
                "MASTER_PORT": str(row["master_port"]),
                "PYTHONPATH": remote_source,
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            "gpu_assignments": [
                dict(assignment)
                for assignment in gpu_assignments
            ],
            "provenance": provenance,
        }
        for row in case_port_pairs
    ]
    remote_artifact = artifact_paths["remote_artifact"]
    package = artifact_paths["package"]
    local_extract = artifact_paths["local_extract"]
    commands = {
        "reserve_remote": remote_command(
            "reserve_remote",
            argv=["reserve_remote", remote_run],
        ),
        "upload": remote_command(
            "upload",
            argv=[
                ["upload", "source_bundle", remote_run],
                ["upload", "source_package", remote_run],
            ],
            provenance=provenance,
        ),
        "stage": remote_command(
            "stage",
            argv=["stage", remote_run, remote_artifact],
            provenance=provenance,
        ),
        "resource_guard": remote_command(
            "resource_guard",
            argv=[
                "resource_guard",
                *execution_plan["required_gpu_indices"],
            ],
            minimum_free_bytes_per_gpu=MIN_GPU_FREE_BYTES,
            requires_no_active_compute_processes=True,
        ),
        "workers": remote_command(
            "workers",
            argv=["workers", remote_run],
            worker_commands=worker_commands,
        ),
        "assembly": remote_command(
            "assembly",
            argv=["assembly", remote_run, remote_artifact],
            provenance=provenance,
        ),
        "remote_verify": remote_command(
            "remote_verify",
            argv=["remote_verify", remote_artifact],
            provenance=provenance,
        ),
        "package_download": remote_command(
            "package_download",
            remote_argv=["package_download", remote_artifact],
            local_output=package,
        ),
        "safe_extract": command(
            "safe_extract",
            argv=["safe_extract", package, local_extract],
        ),
        "local_verify": command(
            "local_verify",
            argv=["local_verify", local_extract],
            provenance=provenance,
        ),
    }
    commands["final_resource_guard"] = dict(commands["resource_guard"])
    return {
        name: commands[name]
        for name in EXECUTION_COMMAND_ORDER
    }


_FORBIDDEN_EXECUTABLE_BASENAMES = {"kill", "pkill", "killall"}
_SHELL_EXECUTABLE_BASENAMES = {"bash", "dash", "ksh", "sh", "zsh"}
_ENV_OPTIONS_WITH_ARGUMENT = {"-u", "--unset", "-C", "--chdir", "--argv0"}
_ENV_OPTIONS_WITHOUT_ARGUMENT = {
    "-i",
    "--ignore-environment",
    "-0",
    "--null",
}
_ENV_LONG_OPTIONS_WITH_VALUE = {"--unset", "--chdir", "--argv0"}
_TIMEOUT_OPTIONS_WITH_ARGUMENT = {"-k", "--kill-after", "-s", "--signal"}
_TIMEOUT_OPTIONS_WITHOUT_ARGUMENT = {
    "--foreground",
    "--preserve-status",
    "-v",
    "--verbose",
}
_TIMEOUT_LONG_OPTIONS_WITH_VALUE = {"--kill-after", "--signal"}


def _executable_basename(value: str) -> str:
    return value.rstrip("/").rsplit("/", 1)[-1]


def _unwrap_env_argv(argv: list[object]) -> list[object]:
    index = 1
    while index < len(argv):
        argument = argv[index]
        if not isinstance(argument, str) or not argument:
            raise ValueError("env command argv is malformed")
        if argument == "--":
            index += 1
            break
        if argument in _ENV_OPTIONS_WITH_ARGUMENT:
            index += 2
            if index > len(argv):
                raise ValueError("env command argv is malformed")
            continue
        if argument in _ENV_OPTIONS_WITHOUT_ARGUMENT:
            index += 1
            continue
        if argument.startswith("--") and "=" in argument:
            option = argument.split("=", 1)[0]
            if option not in _ENV_LONG_OPTIONS_WITH_VALUE:
                raise ValueError("env command argv is malformed")
            index += 1
            continue
        if argument.startswith("-"):
            raise ValueError("env command argv is malformed")
        if "=" in argument:
            name, _value = argument.split("=", 1)
            if not name or not name.replace("_", "a").isalnum():
                raise ValueError("env command argv is malformed")
            index += 1
            continue
        break
    if index >= len(argv):
        raise ValueError("env command argv is malformed")
    return argv[index:]


def _unwrap_command_argv(argv: list[object]) -> list[object]:
    index = 1
    while index < len(argv):
        argument = argv[index]
        if not isinstance(argument, str) or not argument:
            raise ValueError("command argv is malformed")
        if argument == "--":
            index += 1
            break
        if argument == "-p":
            index += 1
            continue
        if argument.startswith("-"):
            raise ValueError("command argv is malformed")
        break
    if index >= len(argv):
        raise ValueError("command argv is malformed")
    return argv[index:]


def _unwrap_simple_delegate_argv(
    argv: list[object],
    *,
    command_name: str,
) -> list[object]:
    if len(argv) < 2:
        raise ValueError(f"{command_name} command argv is malformed")
    if not isinstance(argv[1], str) or not argv[1] or argv[1].startswith("-"):
        raise ValueError(f"{command_name} command argv is malformed")
    return argv[1:]


def _unwrap_timeout_argv(argv: list[object]) -> list[object]:
    index = 1
    while index < len(argv):
        argument = argv[index]
        if not isinstance(argument, str) or not argument:
            raise ValueError("timeout command argv is malformed")
        if argument == "--":
            index += 1
            break
        if argument in _TIMEOUT_OPTIONS_WITH_ARGUMENT:
            index += 2
            if index > len(argv):
                raise ValueError("timeout command argv is malformed")
            continue
        if argument in _TIMEOUT_OPTIONS_WITHOUT_ARGUMENT:
            index += 1
            continue
        if argument.startswith("--") and "=" in argument:
            option = argument.split("=", 1)[0]
            if option not in _TIMEOUT_LONG_OPTIONS_WITH_VALUE:
                raise ValueError("timeout command argv is malformed")
            index += 1
            continue
        if argument.startswith("-"):
            raise ValueError("timeout command argv is malformed")
        break
    if index >= len(argv):
        raise ValueError("timeout command argv is malformed")
    duration = argv[index]
    if (
        not isinstance(duration, str)
        or not duration
        or duration.startswith("-")
    ):
        raise ValueError("timeout command argv is malformed")
    index += 1
    if index >= len(argv):
        raise ValueError("timeout command argv is malformed")
    return argv[index:]


def _validate_executable_argv(argv: object) -> None:
    if not isinstance(argv, (list, tuple)) or not argv:
        raise ValueError("command argv structure is malformed")
    if not isinstance(argv[0], str) or not argv[0]:
        raise ValueError("command argv executable is malformed")
    if any(
        isinstance(value, (Mapping, list, tuple))
        for value in argv[1:]
    ):
        raise ValueError("command argv arguments are malformed")
    for value in argv:
        if not isinstance(value, str):
            continue
        value_basename = _executable_basename(value)
        if value_basename in _FORBIDDEN_EXECUTABLE_BASENAMES:
            raise ValueError(
                f"forbidden command executable: "
                f"{value_basename}"
            )
        if value_basename in _SHELL_EXECUTABLE_BASENAMES:
            raise ValueError(
                f"shell command executable is forbidden: {value_basename}"
            )
    basename = _executable_basename(argv[0])
    if basename == "env":
        _validate_executable_argv(_unwrap_env_argv(list(argv)))
        return
    if basename == "command":
        _validate_executable_argv(_unwrap_command_argv(list(argv)))
        return
    if basename in {"exec", "nohup"}:
        _validate_executable_argv(
            _unwrap_simple_delegate_argv(
                list(argv),
                command_name=basename,
            )
        )
        return
    if basename == "timeout":
        _validate_executable_argv(_unwrap_timeout_argv(list(argv)))
        return


def _validate_command_payload(value: object, *, argv_context: bool) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _validate_command_payload(
                nested,
                argv_context=(
                    isinstance(key, str)
                    and (key == "argv" or key.endswith("_argv"))
                ),
            )
        return
    if isinstance(value, (list, tuple)):
        if argv_context:
            if not value:
                raise ValueError("command argv structure is malformed")
            if isinstance(value[0], (list, tuple)):
                for nested in value:
                    _validate_executable_argv(nested)
            else:
                _validate_executable_argv(value)
        elif value and isinstance(value[0], str) and not value[0].startswith(
            "-"
        ):
            _validate_executable_argv(value)
        for nested in value:
            _validate_command_payload(nested, argv_context=False)
        return
    if argv_context:
        raise ValueError("command argv structure is malformed")


def _contains_command_argv(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            (
                isinstance(key, str)
                and (key == "argv" or key.endswith("_argv"))
            )
            or _contains_command_argv(nested)
            for key, nested in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_command_argv(nested) for nested in value)
    return False


def validate_execution_command_semantics(
    commands: object,
    *,
    expected_order: object,
    execution_plan: object = None,
) -> None:
    if (
        not isinstance(expected_order, (list, tuple))
        or not expected_order
        or any(
            not isinstance(name, str) or not name
            for name in expected_order
        )
        or len(set(expected_order)) != len(expected_order)
    ):
        raise ValueError("expected command order is invalid")
    if not isinstance(commands, Mapping):
        raise ValueError("commands must be a mapping")
    if set(commands) != set(expected_order):
        raise ValueError("command entries are missing, extra, or reordered")
    if _contains_command_argv(commands) and list(commands) != list(
        expected_order
    ):
        raise ValueError("command entries are missing, extra, or reordered")
    if execution_plan is not None:
        if (
            not isinstance(execution_plan, Mapping)
            or execution_plan.get("command_order")
            != list(EXECUTION_COMMAND_ORDER)
        ):
            raise ValueError("execution plan command order is not canonical")
        canonical_commands = canonical_execution_commands(execution_plan)
        if list(commands) != list(EXECUTION_COMMAND_ORDER):
            raise ValueError("command order is not canonical")
        if commands != canonical_commands:
            raise ValueError("command semantics are not canonical")
    for name, command in commands.items():
        if not isinstance(command, Mapping):
            raise ValueError(f"command object is malformed: {name}")
        _validate_command_payload(command, argv_context=False)


def canonical_json_file_sha256(value: object) -> str:
    return hashlib.sha256(
        canonical_json_bytes(value) + b"\n"
    ).hexdigest()


def resource_guard_sha256(document: object) -> str:
    if not isinstance(document, Mapping):
        raise ValueError("resource guard document is invalid")
    hash_domain = {
        key: value
        for key, value in document.items()
        if key not in {"sha256", "phase"}
    }
    return canonical_json_sha256(hash_domain)


def sha256_file(path: object) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_regular_file_once(path: object, label: str) -> bytes:
    value = Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(value, flags)
    except OSError as error:
        raise ValueError(f"{label} file is invalid") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} file is invalid")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_json_file_once(path: object, label: str) -> tuple[bytes, object]:
    data = read_regular_file_once(path, label)
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} JSON is invalid") from error
    return data, document


def _deterministic_token_ids(seed: int, count: int) -> list[int]:
    state = seed
    token_ids = []
    for _ in range(count):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        token_ids.append(1024 + state % (TOKEN_ID_UPPER_BOUND - 1024))
    return token_ids


def _validate_workload_admission(
    workload: str,
    spec: Mapping[str, object],
) -> None:
    source_seed_tokens = (
        spec["shared_prefix_tokens"] + spec["suffix_tokens"] + 1
    )
    continuation_tokens = (
        spec["shared_prefix_tokens"]
        + spec["suffix_tokens"]
        + spec["generated_tokens"]
    )
    if source_seed_tokens > MAX_MODEL_LEN:
        raise ValueError(f"{workload} source seed exceeds max_model_len")
    if continuation_tokens > MAX_MODEL_LEN:
        raise ValueError(f"{workload} continuation exceeds max_model_len")


def _build_workload_payload(workload: str) -> dict[str, object]:
    if workload not in WORKLOAD_SPECS:
        raise ValueError(f"unsupported workload: {workload}")
    spec = dict(WORKLOAD_SPECS[workload])
    _validate_workload_admission(workload, spec)
    token_seed = 2026072900 + WORKLOADS.index(workload)
    shared_prefix = _deterministic_token_ids(
        token_seed,
        spec["shared_prefix_tokens"],
    )
    continuations = [
        {
            "request_index": continuation_index,
            "suffix_token_ids": _deterministic_token_ids(
                token_seed + 100 + continuation_index,
                spec["suffix_tokens"],
            ),
            "prefix_overrides": [],
            "invalidation": {"kind": "none"},
        }
        for continuation_index in range(spec["continuations"])
    ]
    if workload == "w4_miss_invalidation":
        mismatch_index = spec["shared_prefix_tokens"] // 2
        replacement = (
            shared_prefix[mismatch_index] + 1
        ) % TOKEN_ID_UPPER_BOUND
        if replacement < 1024:
            replacement += 1024
        continuations[0]["prefix_overrides"] = [[
            mismatch_index,
            replacement,
        ]]
        continuations[0]["invalidation"] = {
            "kind": "token_mismatch",
            "prefix_index": mismatch_index,
            "replacement_token_id": replacement,
        }
        continuations[1]["invalidation"] = {
            "kind": "stale_block_generation",
        }
        continuations[2]["invalidation"] = {"kind": "cache_clear"}
    return {
        "spec": spec,
        "token_seed": token_seed,
        "shared_prefix_token_ids": shared_prefix,
        "source_suffix_token_ids": _deterministic_token_ids(
            token_seed + 1,
            spec["suffix_tokens"],
        ),
        "continuations": continuations,
    }


def workload_payload(workload: str) -> dict[str, object]:
    return json.loads(json.dumps(
        _build_workload_payload(workload),
        sort_keys=True,
        separators=(",", ":"),
    ))


def workload_manifest_payload() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "workloads": {
            workload: workload_payload(workload)
            for workload in WORKLOADS
        },
    }


def workload_sampling_seed(workload: object) -> int:
    if workload not in WORKLOADS:
        raise ValueError("workload is invalid")
    return 2026072900 + WORKLOADS.index(workload)


def effective_concurrency(workload: object, phase: object) -> int:
    if workload not in WORKLOADS:
        raise ValueError("workload is invalid")
    if phase not in {"warmup", "correctness", "measured"}:
        raise ValueError("phase is invalid")
    if phase == "correctness":
        return CORRECTNESS_CONCURRENCY
    if workload == "w3_batched_fanout":
        return WORKLOAD_SPECS[workload]["continuations"]
    return 1


def profile_order(repetition: int) -> tuple[str, ...]:
    if (
        isinstance(repetition, bool)
        or not isinstance(repetition, int)
        or repetition < 0
    ):
        raise ValueError("repetition must be a non-negative integer")
    offset = repetition % len(PROFILES)
    return PROFILES[offset:] + PROFILES[:offset]


def build_case_matrix() -> tuple[BenchmarkCase, ...]:
    cases = []
    phases = (
        ("warmup", WARMUP_REPETITIONS),
        ("correctness", CORRECTNESS_REPETITIONS),
        ("measured", MEASURED_REPETITIONS),
    )
    for workload in WORKLOADS:
        for phase, repetitions in phases:
            for repetition in range(repetitions):
                for profile in profile_order(repetition):
                    cases.append(BenchmarkCase(
                        case_id=(
                            f"{workload}__{phase}__r{repetition}__"
                            f"{profile}"
                        ),
                        workload=workload,
                        profile=profile,
                        phase=phase,
                        repetition=repetition,
                        concurrency=effective_concurrency(workload, phase),
                        strict_correctness=phase == "correctness",
                    ))
    return tuple(cases)


def _blocked(*reasons: str) -> PrerequisiteStatus:
    return PrerequisiteStatus(
        classification="BLOCKED_CORRECTNESS",
        authorized=False,
        reasons=tuple(reasons),
    )


def _safe_relative_file(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path is unsafe")
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} file is missing")
    return path


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _nonnegative_integer(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value >= 0
    )


def _validate_root_logit_documents(
    artifact: Mapping[str, object],
    verification: Mapping[str, object],
    source_tree_sha256: object,
) -> None:
    required = {
        "schema_version",
        "run_tag",
        "classification",
        "comparison_policy",
        "tolerance",
        "prompts",
        "reference_process",
        "comparisons",
        "forbidden_counters",
        "claim_boundary",
    }
    prompts = artifact.get("prompts")
    reference_process = artifact.get("reference_process")
    comparisons = artifact.get("comparisons")
    if (
        set(artifact) != required
        or artifact["schema_version"]
        != TP4_ROOT_CORRECTNESS_SCHEMA_VERSION
        or artifact["classification"] != "PASS"
        or artifact["comparison_policy"]
        != "registered_logits_strict_allclose"
        or artifact["tolerance"] != {"atol": 2e-5, "rtol": 0.0}
        or not isinstance(prompts, list)
        or [row.get("case_id") for row in prompts]
        != list(TP4_ROOT_CASE_IDS)
        or not isinstance(reference_process, Mapping)
        or reference_process.get("model_manifest_sha256")
        != MODEL_MANIFEST_SHA256
        or artifact["forbidden_counters"] != {
            "engine": 0,
            "generation": 0,
            "model_runner": 0,
            "sampler": 0,
            "scheduler": 0,
        }
        or not isinstance(artifact["claim_boundary"], str)
        or "no cached decode" not in artifact["claim_boundary"]
        or not isinstance(comparisons, list)
        or [row.get("case_id") for row in comparisons]
        != list(TP4_ROOT_CASE_IDS)
    ):
        raise ValueError("tp4_root_logit artifact schema is invalid")
    for row in comparisons:
        fields = {
            "native_winner_token_id",
            "native_runner_up_token_id",
            "native_winner_margin",
            "official_winner_token_id",
            "official_runner_up_token_id",
            "official_winner_margin",
            "native_topk_token_ids",
            "official_topk_token_ids",
        }
        if not isinstance(row, Mapping) or not fields.issubset(row):
            raise ValueError(
                "tp4_root_logit comparison schema is invalid"
            )
        if (
            row["native_winner_token_id"]
            != row["official_winner_token_id"]
            or row["official_winner_token_id"]
            not in row["native_topk_token_ids"]
            or row["native_winner_token_id"]
            not in row["official_topk_token_ids"]
            or (
                row["official_winner_margin"] > 0
                and row["native_winner_margin"] <= 0
            )
            or (
                row["official_winner_margin"] == 0
                and (
                    row["native_winner_margin"] != 0
                    or row["native_runner_up_token_id"]
                    != row["official_runner_up_token_id"]
                )
            )
        ):
            raise ValueError(
                "tp4_root_logit artifact does not prove PASS"
            )
    if (
        set(verification)
        != {"classification", "case_ids", "ranks", "checks"}
        or verification["classification"] != "PASS"
        or verification["case_ids"] != list(TP4_ROOT_CASE_IDS)
        or verification["ranks"] != [0, 1, 2, 3]
        or not _nonnegative_integer(verification["checks"])
        or verification["checks"] == 0
        or not _valid_sha256(source_tree_sha256)
    ):
        raise ValueError(
            "tp4_root_logit independent verification schema is invalid"
        )


def _cached_expected_keys() -> tuple[tuple[str, int], ...]:
    return tuple(
        (workload, request_index)
        for workload in WORKLOADS[1:]
        for request_index in range(
            WORKLOAD_SPECS[workload]["continuations"]
        )
    )


def _validate_cached_documents(
    artifact: Mapping[str, object],
    verification: Mapping[str, object],
    source_tree_sha256: object,
) -> None:
    required = {
        "schema_version",
        "classification",
        "model_manifest_sha256",
        "workload_manifest_sha256",
        "rows",
    }
    rows = artifact.get("rows")
    if (
        set(artifact) != required
        or artifact["schema_version"]
        != CACHED_CONTINUATION_SCHEMA_VERSION
        or artifact["classification"] != "PASS"
        or artifact["model_manifest_sha256"] != MODEL_MANIFEST_SHA256
        or artifact["workload_manifest_sha256"]
        != APPROVED_V1_WORKLOAD_MANIFEST_SHA256
        or not isinstance(rows, list)
        or tuple(
            (row.get("workload"), row.get("request_index"))
            for row in rows
        )
        != _cached_expected_keys()
    ):
        raise ValueError(
            "cached_continuation artifact schema is invalid"
        )
    restore_hits = 0
    w4_misses = 0
    for row in rows:
        workload = row["workload"]
        request_index = row["request_index"]
        spec = WORKLOAD_SPECS[workload]
        expected_hit = workload != "w4_miss_invalidation"
        expected_reason = (
            "exact_hit"
            if expected_hit
            else (
                "token_mismatch",
                "stale_block_generation",
                "cache_clear",
            )[request_index]
        )
        if (
            row.get("outcome") != "continuation"
            or row.get("restore_hit") is not expected_hit
            or row.get("restore_reason") != expected_reason
            or row.get("prompt_tokens")
            != spec["shared_prefix_tokens"] + spec["suffix_tokens"]
            or row.get("reused_tokens")
            != (spec["shared_prefix_tokens"] if expected_hit else 0)
            or row.get("executed_prefill_tokens")
            != (
                spec["suffix_tokens"]
                if expected_hit
                else spec["shared_prefix_tokens"] + spec["suffix_tokens"]
            )
            or row.get("output_token_ids")
            != row.get("reference_output_token_ids")
            or len(row.get("output_token_ids", ()))
            != spec["generated_tokens"]
            or row.get("logits_allclose") is not True
            or not isinstance(row.get("logits_max_abs_diff"), (int, float))
            or isinstance(row.get("logits_max_abs_diff"), bool)
            or not math.isfinite(row["logits_max_abs_diff"])
            or row["logits_max_abs_diff"] < 0
            or row["logits_max_abs_diff"] > 2e-5
            or row.get("cache_identity_match") is not True
            or row.get("rank_inventory") != [0, 1, 2, 3]
            or row.get("process_group_destroyed") is not True
            or row.get("owned_children_remaining") != []
        ):
            raise ValueError(
                "cached_continuation artifact does not prove PASS"
            )
        restore_hits += int(expected_hit)
        w4_misses += int(not expected_hit)
    checks = {
        "row_count": len(rows),
        "restore_hits": restore_hits,
        "w4_misses": w4_misses,
    }
    if (
        set(verification)
        != {
            "schema_version",
            "classification",
            "source_tree_sha256",
            "model_manifest_sha256",
            "workload_manifest_sha256",
            "checks",
        }
        or verification["schema_version"]
        != CACHED_CONTINUATION_SCHEMA_VERSION
        or verification["classification"] != "PASS"
        or verification["source_tree_sha256"] != source_tree_sha256
        or verification["model_manifest_sha256"]
        != MODEL_MANIFEST_SHA256
        or verification["workload_manifest_sha256"]
        != APPROVED_V1_WORKLOAD_MANIFEST_SHA256
        or verification["checks"] != checks
    ):
        raise ValueError(
            "cached_continuation independent verification schema is invalid"
        )


def _validate_engine_documents(
    artifact: Mapping[str, object],
    verification: Mapping[str, object],
    source_tree_sha256: object,
) -> None:
    required = {
        "schema_version",
        "classification",
        "model_manifest_sha256",
        "rows",
    }
    rows = artifact.get("rows")
    if (
        set(artifact) != required
        or artifact["schema_version"] != ENGINE_CORRECTNESS_SCHEMA_VERSION
        or artifact["classification"] != "PASS"
        or artifact["model_manifest_sha256"] != MODEL_MANIFEST_SHA256
        or not isinstance(rows, list)
        or [row.get("scenario") for row in rows]
        != list(ENGINE_CORRECTNESS_SCENARIOS)
    ):
        raise ValueError("engine_correctness artifact schema is invalid")
    restore_hits = 0
    restore_misses = 0
    for row in rows:
        expected = ENGINE_CORRECTNESS_SCENARIOS[row["scenario"]]
        values = (
            row.get("scheduler_steps"),
            row.get("model_runner_calls"),
            len(row.get("output_token_ids", ())),
            row.get("publication_commits"),
            row.get("restore_hits"),
            row.get("restore_misses"),
            row.get("release_events"),
            row.get("cache_entries_after"),
        )
        if (
            values != expected
            or row.get("engine_class")
            != "tinyvllm.engine.llm_engine.LLMEngine"
            or row.get("model_runner_class")
            != "tinyvllm.engine.model_runner.ModelRunner"
            or row.get("rank_inventory") != [0, 1, 2, 3]
            or row.get("ack_ranks") != [1, 2, 3]
            or row.get("output_token_ids")
            != row.get("reference_output_token_ids")
            or row.get("cache_identity_match") is not True
            or row.get("process_group_destroyed") is not True
            or row.get("rank_exit_codes") != [0, 0, 0, 0]
            or row.get("owned_children_remaining") != []
        ):
            raise ValueError(
                "engine_correctness artifact does not prove PASS"
            )
        restore_hits += row["restore_hits"]
        restore_misses += row["restore_misses"]
    checks = {
        "scenario_count": len(rows),
        "restore_hits": restore_hits,
        "restore_misses": restore_misses,
    }
    if (
        set(verification)
        != {
            "schema_version",
            "classification",
            "source_tree_sha256",
            "model_manifest_sha256",
            "checks",
        }
        or verification["schema_version"]
        != ENGINE_CORRECTNESS_SCHEMA_VERSION
        or verification["classification"] != "PASS"
        or verification["source_tree_sha256"] != source_tree_sha256
        or verification["model_manifest_sha256"]
        != MODEL_MANIFEST_SHA256
        or verification["checks"] != checks
    ):
        raise ValueError(
            "engine_correctness independent verification schema is invalid"
        )


def validate_authority_documents(
    name: object,
    artifact: object,
    verification: object,
    source_tree_sha256: object,
) -> None:
    if not isinstance(artifact, Mapping) or not isinstance(
        verification,
        Mapping,
    ):
        raise ValueError(f"{name} authority schema is invalid")
    if name == "tp4_root_logit":
        _validate_root_logit_documents(
            artifact,
            verification,
            source_tree_sha256,
        )
    elif name == "cached_continuation":
        _validate_cached_documents(
            artifact,
            verification,
            source_tree_sha256,
        )
    elif name == "engine_correctness":
        _validate_engine_documents(
            artifact,
            verification,
            source_tree_sha256,
        )
    else:
        raise ValueError("authority name is invalid")


def validate_prerequisite_provenance(
    name: object,
    provenance: object,
    *,
    run_tag: object,
    source_tree_sha256: object,
) -> None:
    required = {
        "schema_version",
        "authority_name",
        "run_tag",
        "binding_kind",
        "source_tree_sha256",
        "model_manifest_sha256",
        "root_logit_receipt_gap",
        "plan_path",
        "plan_sha256",
        "authorization_path",
        "authorization_sha256",
        "receipt_path",
        "receipt_sha256",
    }
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != required
        or provenance["schema_version"]
        != PREREQUISITE_PROVENANCE_SCHEMA_VERSION
        or provenance["authority_name"] != name
        or provenance["run_tag"] != run_tag
        or provenance["source_tree_sha256"] != source_tree_sha256
        or provenance["model_manifest_sha256"]
        != MODEL_MANIFEST_SHA256
    ):
        raise ValueError(f"{name} provenance schema is invalid")
    if (
        provenance["binding_kind"] != "remote_execution_receipt"
        or provenance["root_logit_receipt_gap"] is not False
    ):
        raise ValueError(f"{name} receipt provenance is invalid")
    for path_field, sha_field in (
        ("plan_path", "plan_sha256"),
        ("authorization_path", "authorization_sha256"),
        ("receipt_path", "receipt_sha256"),
    ):
        relative = provenance[path_field]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not _valid_sha256(provenance[sha_field])
        ):
            raise ValueError(f"{name} receipt provenance is invalid")


_ROOT_PLAN_SCHEMA = (
    "qwen35.tp4-root-logit-remote-execution-plan.v1"
)
_ROOT_AUTHORIZATION_SCHEMA = (
    "qwen35.tp4-root-logit-remote-execution-authorization.v1"
)
_ROOT_RECEIPT_SCHEMA = (
    "qwen35.tp4-root-logit-remote-execution-receipt.v1"
)
_ROOT_REMOTE_TARGET = "sitian@10.232.195.203"
_ROOT_FROZEN_SOURCE_TAG_PREFIX = "qwen35-tp4-source-prep-"
_ROOT_LOCAL_RUN_ROOT = "experiments/qwen35_hybrid_state"
_ROOT_REMOTE_GATE_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-tp4-root-logit-tests"
)
_ROOT_EXACT_ARTIFACT_NAMES = (
    "native_rank0_logits.pt",
    "rank_evidence.json",
    "reference_logits.pt",
    "source_manifest.json",
    "tp4_real_root_logit_correctness.json",
)
_ROOT_STAGE_ORDER = ("preflight", "run", "download", "verify")
_ROOT_RESOURCE_POLICY = "controlled_shared"
_ROOT_RESOURCE_BASELINE_NAME = "resource_baseline.json"
_ROOT_MIN_GPU_FREE_BYTES = 24 * 1024**3
_ROOT_CLAIM_BOUNDARY = (
    "execution authorization only; no SSH, GPU, correctness, "
    "performance, cache, memory, compression, or quality claim"
)
_CACHED_PLAN_SCHEMA = (
    "qwen35.tp4-cached-continuation-remote-execution-plan.v1"
)
_CACHED_RECEIPT_SCHEMA = (
    "qwen35.tp4-cached-continuation-remote-execution-receipt.v1"
)
_ENGINE_PLAN_SCHEMA = "qwen35.tp4-engine-remote-execution-plan.v1"
_ENGINE_AUTHORIZATION_SCHEMA = (
    "qwen35.tp4-engine-remote-execution-authorization.v1"
)
_ENGINE_RECEIPT_SCHEMA = (
    "qwen35.tp4-engine-remote-execution-receipt.v1"
)
_WORKLOAD_COMMAND_ORDER = (
    "reserve_remote",
    "upload",
    "stage",
    "resource_guard",
    "guarded_authority",
    "package_download",
    "safe_extract",
    "prepare_local_verifier",
    "local_verify",
)
_WORKLOAD_SSH_TARGET = "sitian@10.232.195.203"
_WORKLOAD_FORBIDDEN_EXECUTABLE_BASENAMES = {
    "kill",
    "pkill",
    "killall",
}
_WORKLOAD_SHELL_OR_DELEGATOR_BASENAMES = {
    "bash",
    "command",
    "dash",
    "env",
    "exec",
    "ksh",
    "nohup",
    "sh",
    "timeout",
    "zsh",
}


def _require_exact_fields(
    value: object,
    fields: set[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} schema fields are invalid")
    return value


def _require_safe_nonce(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(
            character not in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789-_"
            )
            for character in value
        )
    ):
        raise ValueError(f"{label} nonce is invalid")
    return value


def _require_gpu_indices(value: object, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or len(set(value)) != 4
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            for index in value
        )
    ):
        raise ValueError(f"{label} GPU identity is invalid")
    return value


def _require_ports(value: object, label: str) -> Mapping[str, object]:
    ports = _require_exact_fields(
        value,
        {"dist_port", "master_port"},
        label,
    )
    values = (ports["dist_port"], ports["master_port"])
    if (
        any(
            isinstance(port, bool)
            or not isinstance(port, int)
            or port <= 0
            or port > 65535
            for port in values
        )
        or values[0] == values[1]
    ):
        raise ValueError(f"{label} port identity is invalid")
    return ports


def _workload_command_basename(value: str) -> str:
    return value.rstrip("/").rsplit("/", 1)[-1]


def _require_workload_argv(value: object, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(token, str) or not token for token in value)
    ):
        raise ValueError(f"{label} command argv is invalid")
    for token in value:
        if (
            _workload_command_basename(token)
            in _WORKLOAD_FORBIDDEN_EXECUTABLE_BASENAMES
        ):
            raise ValueError(f"{label} command executable is forbidden")
    return value


def _validate_workload_non_shell_argv(
    value: object,
    label: str,
) -> list[str]:
    argv = _require_workload_argv(value, label)
    if any(
        _workload_command_basename(token)
        in _WORKLOAD_SHELL_OR_DELEGATOR_BASENAMES
        for token in argv
    ):
        raise ValueError(f"{label} shell or delegator command is forbidden")
    return argv


def _validate_workload_ssh_argv(
    value: object,
    *,
    ssh_target: object,
    label: str,
) -> list[str]:
    argv = _require_workload_argv(value, label)
    if (
        not isinstance(ssh_target, str)
        or not ssh_target
        or _workload_command_basename(argv[0]) != "ssh"
        or argv.count(ssh_target) != 1
    ):
        raise ValueError(f"{label} ssh command is invalid")
    target_index = argv.index(ssh_target)
    remote_argv = argv[target_index + 1:]
    if (
        len(remote_argv) != 3
        or _workload_command_basename(remote_argv[0]) != "bash"
        or remote_argv[1] != "-lc"
    ):
        raise ValueError(f"{label} remote command is invalid")
    for index, token in enumerate(argv):
        basename = _workload_command_basename(token)
        if (
            basename in _WORKLOAD_SHELL_OR_DELEGATOR_BASENAMES
            and index != target_index + 1
        ):
            raise ValueError(
                f"{label} shell or delegator command is forbidden"
            )
    return argv


def _normalize_reserve_remote_script(
    value: object,
    label: str,
) -> str:
    argv = _require_workload_argv(value, label)
    direct_prefix = ["ssh", _WORKLOAD_SSH_TARGET, "bash", "-lc"]
    native_prefix = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "ConnectTimeout=20",
        _WORKLOAD_SSH_TARGET,
        "bash",
        "-lc",
    ]
    legacy_prefix = native_prefix[:6] + native_prefix[8:]
    if len(argv) == len(direct_prefix) + 1 and argv[:-1] == direct_prefix:
        return argv[-1]
    if len(argv) == len(native_prefix) + 1 and argv[:-1] == native_prefix:
        wrapped_tokens = shlex.split(argv[-1])
    elif len(argv) == len(legacy_prefix) + 1 and argv[:-1] == legacy_prefix:
        wrapped_tokens = shlex.split(argv[-1])
    else:
        raise ValueError(f"{label} command is invalid")
    if (
        len(wrapped_tokens) != 1
        or shlex.quote(wrapped_tokens[0]) != argv[-1]
    ):
        raise ValueError(f"{label} command is invalid")
    wrapped_argv = shlex.split(wrapped_tokens[0])
    if len(wrapped_argv) != 3 or wrapped_argv[:2] != ["bash", "-lc"]:
        raise ValueError(f"{label} command is invalid")
    if shlex.join(wrapped_argv) != wrapped_tokens[0]:
        raise ValueError(f"{label} command is invalid")
    return wrapped_argv[2]


def _validate_workload_python_argv(
    value: object,
    label: str,
) -> list[str]:
    argv = _validate_workload_non_shell_argv(value, label)
    if _workload_command_basename(argv[0]) not in {"python", "python3"}:
        raise ValueError(f"{label} Python command is invalid")
    return argv


def _validate_workload_authority_argv(
    name: str,
    plan: Mapping[str, object],
    value: object,
    label: str,
) -> list[str]:
    argv = _require_workload_argv(value, label)
    if _workload_command_basename(argv[0]) != "env":
        return _validate_workload_non_shell_argv(argv, label)
    remote_inputs = plan.get("remote_inputs")
    ports = plan.get("ports")
    gpu_indices = plan.get("gpu_indices")
    remote_source = plan.get("remote_source_root")
    if (
        not isinstance(remote_inputs, Mapping)
        or not isinstance(ports, Mapping)
        or not isinstance(gpu_indices, list)
        or not isinstance(remote_source, str)
    ):
        raise ValueError(f"{label} command argv is invalid")
    expected = [
        "env",
        f"PYTHONPATH={remote_source}",
        "PYTHONDONTWRITEBYTECODE=1",
        "TORCH_COMPILE_DISABLE=1",
        "CUDA_VISIBLE_DEVICES="
        + ",".join(str(value) for value in gpu_indices),
        f"TINYVLLM_DIST_PORT={ports.get('dist_port')}",
        f"MASTER_PORT={ports.get('master_port')}",
        REMOTE_PYTHON,
    ]
    if name == "engine_correctness":
        expected.extend([
            f"{remote_source}/tools/"
            "run_qwen35_tp4_engine_correctness_authority.py",
            "--configuration",
            remote_inputs.get("configuration"),
            "--source-inventory",
            remote_inputs.get("source_inventory"),
            "--output-root",
            plan.get("remote_authority_root"),
        ])
    elif name == "cached_continuation":
        expected.extend([
            f"{remote_source}/tools/"
            "run_qwen35_tp4_cached_continuation_authority.py",
            "--configuration",
            remote_inputs.get("configuration"),
            "--source-inventory",
            remote_inputs.get("source_inventory"),
            "--output-dir",
            plan.get("remote_cached_authority_dir"),
            "--verification-path",
            plan.get("remote_cached_verification_path"),
        ])
    else:
        raise ValueError(f"{label} command argv is invalid")
    if argv != expected:
        raise ValueError(f"{label} command argv is invalid")
    return argv


def _strict_workload_resource_guard_shell(
    gpu_indices: list[int],
) -> str:
    parse_script = "\n".join([
        "import json,sys",
        "indices=[int(value) for value in sys.argv[1].split(',')]",
        "minimum=int(sys.argv[2])",
        "gpu_text,process_text=sys.stdin.read().split('\\n---PROCESSES---\\n',1)",
        "rows=[]",
        "for line in gpu_text.splitlines():",
        " parts=[value.strip() for value in line.split(',')]",
        " if len(parts)!=3: raise SystemExit('invalid GPU inventory')",
        " rows.append({'gpu_index':int(parts[0]),'gpu_uuid':parts[1],'free_bytes':int(parts[2])*1024*1024,'compute_processes':[]})",
        "by_uuid={row['gpu_uuid']:row for row in rows}",
        "for line in process_text.splitlines():",
        " if not line.strip() or line.strip()=='No running processes found': continue",
        " parts=[value.strip() for value in line.split(',',3)]",
        " if len(parts)!=4 or parts[0] not in by_uuid: raise SystemExit('invalid compute process inventory')",
        " by_uuid[parts[0]]['compute_processes'].append({'pid':int(parts[1]),'process_name':parts[2],'used_memory_mib':int(parts[3])})",
        "selected=[row for row in rows if row['gpu_index'] in indices]",
        "selected.sort(key=lambda row:indices.index(row['gpu_index']))",
        "if len(selected)!=4 or len({row['gpu_uuid'] for row in selected})!=4:",
        " raise SystemExit('four unique configured GPUs are required')",
        "if any(row['free_bytes']<minimum for row in selected):",
        " raise SystemExit('configured GPU free memory is insufficient')",
        "if any(row['compute_processes']!=[] for row in selected):",
        " raise SystemExit('configured GPU has active compute processes')",
        "print(json.dumps({'classification':'READY','selected':selected},sort_keys=True,separators=(',',':')))",
    ])
    gpu_query = (
        "nvidia-smi "
        "--query-gpu=index,uuid,memory.free "
        "--format=csv,noheader,nounits"
    )
    process_query = (
        "nvidia-smi "
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory "
        "--format=csv,noheader,nounits"
    )
    return " && ".join([
        "set -eu",
        f"gpu_rows=\"$({gpu_query})\"",
        f"process_rows=\"$({process_query})\"",
        (
            "printf '%s\\n---PROCESSES---\\n%s\\n' "
            "\"$gpu_rows\" \"$process_rows\" | "
            f"{shlex.quote(REMOTE_PYTHON)} -c "
            f"{shlex.quote(parse_script)} "
            f"{shlex.quote(','.join(str(value) for value in gpu_indices))} "
            f"{MIN_GPU_FREE_BYTES}"
        ),
    ])


def _controlled_shared_workload_resource_guard_shell(
    plan: Mapping[str, object],
    gpu_indices: list[int],
) -> str:
    script = "\n".join([
        "import hashlib,json,sys",
        "from datetime import datetime,timezone",
        "from pathlib import Path",
        "indices=[int(value) for value in sys.argv[1].split(',')]",
        "minimum=int(sys.argv[2])",
        "target=sys.argv[3]",
        "gpu_text,process_text=sys.stdin.read().split('\\n---PROCESSES---\\n',1)",
        "rows=[]",
        "for line in gpu_text.splitlines():",
        " parts=[value.strip() for value in line.split(',')]",
        " if len(parts)!=3: raise SystemExit('invalid GPU inventory')",
        " rows.append({'gpu_index':int(parts[0]),'gpu_uuid':parts[1],'free_bytes':int(parts[2])*1024*1024,'compute_processes':[]})",
        "by_uuid={row['gpu_uuid']:row for row in rows}",
        "for line in process_text.splitlines():",
        " if not line.strip() or line.strip()=='No running processes found': continue",
        " parts=[value.strip() for value in line.split(',',3)]",
        " if len(parts)!=4 or parts[0] not in by_uuid: raise SystemExit('invalid compute process inventory')",
        " pid=int(parts[1])",
        " stat=Path(f'/proc/{pid}/stat').read_text(encoding='utf-8')",
        " close=stat.rfind(')')",
        " if close<0: raise SystemExit('invalid process stat')",
        " fields=stat[close+2:].split()",
        " if len(fields)<=19: raise SystemExit('invalid process stat')",
        " by_uuid[parts[0]]['compute_processes'].append({'pid':pid,'process_name':parts[2],'used_memory_mib':int(parts[3]),'start_time_ticks':int(fields[19])})",
        "selected=[row for row in rows if row['gpu_index'] in indices]",
        "selected.sort(key=lambda row:indices.index(row['gpu_index']))",
        "if len(selected)!=4 or len({row['gpu_uuid'] for row in selected})!=4: raise SystemExit('four unique configured GPUs are required')",
        "if any(row['free_bytes']<minimum for row in selected): raise SystemExit('configured GPU free memory is insufficient')",
        "for row in selected: row['compute_processes'].sort(key=lambda process:(process['pid'],process['process_name'],process['start_time_ticks']))",
        "baseline_path=Path(sys.argv[4])",
        "expected_sha=sys.argv[5]",
        "raw=baseline_path.read_bytes()",
        "if hashlib.sha256(raw).hexdigest()!=expected_sha: raise SystemExit('resource baseline SHA mismatch')",
        "baseline=json.loads(raw)",
        "if baseline.get('ssh_target')!=target or baseline.get('gpu_indices')!=indices: raise SystemExit('resource baseline binding mismatch')",
        "frozen=baseline.get('selected')",
        "if not isinstance(frozen,list) or len(frozen)!=4: raise SystemExit('resource baseline inventory mismatch')",
        "for current,original in zip(selected,frozen):",
        " if current['gpu_index']!=original.get('gpu_index') or current['gpu_uuid']!=original.get('gpu_uuid'): raise SystemExit('resource GPU drift')",
        " allowed={(process.get('pid'),process.get('process_name'),process.get('start_time_ticks')) for process in original.get('compute_processes',[])}",
        " observed={(process['pid'],process['process_name'],process['start_time_ticks']) for process in current['compute_processes']}",
        " if not observed.issubset(allowed): raise SystemExit('resource process drift')",
        "payload={'classification':'READY','resource_policy':'controlled_shared','baseline_sha256':expected_sha,'selected':selected,'benchmark_execution_authorized':False}",
        "print(json.dumps(payload,sort_keys=True,separators=(',',':')))",
    ])
    remote_inputs = plan["remote_inputs"]
    arguments = [
        ",".join(str(value) for value in gpu_indices),
        MIN_GPU_FREE_BYTES,
        plan["ssh_target"],
        remote_inputs["resource_baseline"],
        plan["resource_baseline_sha256"],
    ]
    return " && ".join([
        "set -eu",
        (
            "gpu_rows=\"$(nvidia-smi "
            "--query-gpu=index,uuid,memory.free "
            "--format=csv,noheader,nounits)\""
        ),
        (
            "process_rows=\"$(nvidia-smi "
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory "
            "--format=csv,noheader,nounits)\""
        ),
        (
            "printf '%s\\n---PROCESSES---\\n%s\\n' "
            "\"$gpu_rows\" \"$process_rows\" | "
            f"python3 -c {shlex.quote(script)} "
            + " ".join(shlex.quote(str(value)) for value in arguments)
        ),
    ])


def _expected_guarded_authority_ssh_argv(
    plan: Mapping[str, object],
    authority_argv: list[str],
) -> tuple[list[str], list[str]]:
    gpu_indices = plan["gpu_indices"]
    if plan.get("resource_policy", "strict_exclusive") == (
        "strict_exclusive"
    ):
        guard_shell = _strict_workload_resource_guard_shell(gpu_indices)
    else:
        guard_shell = _controlled_shared_workload_resource_guard_shell(
            plan,
            gpu_indices,
        )
    guarded_shell = (
        f"final_resource=\"$({guard_shell})\" && "
        "printf 'QWEN35_FINAL_RESOURCE_JSON=%s\\n' "
        "\"$final_resource\" && exec "
        f"{shlex.join(authority_argv)}"
    )
    remote_argv = ["bash", "-lc", guarded_shell]
    expected = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "ConnectTimeout=20",
        plan["ssh_target"],
        "bash",
        "-lc",
        shlex.quote(shlex.join(remote_argv)),
    ]
    legacy = expected[:6] + expected[8:]
    return expected, legacy


def _require_safe_workload_output_path(
    value: object,
    label: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or Path(value) == Path(".")
        or ".." in Path(value).parts
    ):
        raise ValueError(f"{label} path is unsafe")
    return value


_WORKLOAD_SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ControlMaster=no",
    "-o",
    "ControlPath=none",
    "-o",
    "ConnectTimeout=20",
)
_WORKLOAD_REMOTE_ROOTS = {
    "engine_correctness": (
        "/data00/home/sitian/sitian-workspace01/tllm/"
        "qwen35-tp4-engine-authority-runs"
    ),
    "cached_continuation": (
        "/data00/home/sitian/sitian-workspace01/tllm/"
        "qwen35-tp4-cached-continuation-authority-runs"
    ),
}
_WORKLOAD_REMOTE_CONFIGURATION_NAME = (
    "remote_executor_configuration.json"
)
_WORKLOAD_SOURCE_TAR_NAME = "authority_source.tar"
_WORKLOAD_RESOURCE_BASELINE_NAME = "resource_baseline.json"
_WORKLOAD_LOCAL_VERIFIER_SOURCE_NAME = "local_verifier_source"
_WORKLOAD_DOWNLOADED_AUTHORITY_NAMES = {
    "engine_correctness": "downloaded_authority",
    "cached_continuation": "downloaded_cached_authority",
}
_WORKLOAD_PACKAGE_TAR_NAMES = {
    "engine_correctness": "authority.tar",
    "cached_continuation": "cached_authority.tar",
}
_WORKLOAD_ENGINE_PACKAGE_ENTRIES = (
    "reference_authority",
    "reference_independent_verification.json",
    "engine_authority",
    "authority_summary.json",
)
_WORKLOAD_CACHED_PACKAGE_ENTRIES = (
    "cached_continuation_authority",
    "cached_continuation_independent_verification.json",
)


def _workload_native_ssh_argv(
    remote_argv: list[str],
) -> list[str]:
    remote_command = " ".join(
        shlex.quote(str(value)) for value in remote_argv
    )
    return [
        "ssh",
        *_WORKLOAD_SSH_OPTIONS,
        _WORKLOAD_SSH_TARGET,
        "bash",
        "-lc",
        shlex.quote(remote_command),
    ]


def _workload_native_scp_argv(
    local_path: str,
    remote_path: str,
) -> list[str]:
    return [
        "scp",
        *_WORKLOAD_SSH_OPTIONS,
        local_path,
        f"{_WORKLOAD_SSH_TARGET}:{remote_path}",
    ]


def _workload_stage_script(
    remote_source: str,
    remote_inputs_root: str,
    identities: Mapping[str, str],
    *,
    resource_baseline_name: str | None,
) -> str:
    verify_script = "\n".join([
        "import hashlib,json,sys,tarfile",
        "from pathlib import Path,PurePosixPath",
        "archive=Path(sys.argv[1])",
        "destination=Path(sys.argv[2])",
        "inventory_path=Path(sys.argv[3])",
        "expected_tree=sys.argv[4]",
        "expected_tar=sys.argv[5]",
        "if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected_tar:",
        " raise SystemExit('source tar SHA mismatch')",
        "inventory=json.loads(inventory_path.read_text())",
        "with tarfile.open(archive,'r:') as handle:",
        " members=handle.getmembers()",
        " names=[member.name for member in members]",
        " if names!=inventory['owned_files']:",
        "  raise SystemExit('source tar inventory mismatch')",
        " for member in members:",
        "  path=PurePosixPath(member.name)",
        "  if path.is_absolute() or '..' in path.parts or not member.isfile() or member.issym() or member.islnk():",
        "   raise SystemExit('unsafe source tar member')",
        " handle.extractall(destination,members=members)",
        "digest=hashlib.sha256()",
        "for name in inventory['owned_files']:",
        " path=destination.joinpath(*PurePosixPath(name).parts)",
        " encoded=name.encode('utf-8')",
        " digest.update(len(encoded).to_bytes(8,'big'))",
        " digest.update(encoded)",
        " with path.open('rb') as source:",
        "  for chunk in iter(lambda:source.read(1024*1024),b''):",
        "   digest.update(chunk)",
        "if digest.hexdigest()!=expected_tree:",
        " raise SystemExit('source tree SHA mismatch')",
    ])
    identity_rows = [
        (
            _WORKLOAD_REMOTE_CONFIGURATION_NAME,
            "configuration_sha256",
        ),
        ("source_inventory.json", "source_inventory_sha256"),
        (_WORKLOAD_SOURCE_TAR_NAME, "source_tar_sha256"),
        ("workload_manifest.json", "workload_manifest_sha256"),
    ]
    if resource_baseline_name is not None:
        identity_rows.append((
            resource_baseline_name,
            "resource_baseline_sha256",
        ))
    commands = [
        "set -eu",
        *[
            (
                f"test \"$(sha256sum "
                f"{shlex.quote(remote_inputs_root + '/' + filename)} "
                "| awk '{print $1}')\" = "
                f"{shlex.quote(identities[identity_name])}"
            )
            for filename, identity_name in identity_rows
        ],
        f"mkdir {shlex.quote(remote_source)}",
        (
            f"{shlex.quote(REMOTE_PYTHON)} -c "
            f"{shlex.quote(verify_script)} "
            f"{shlex.quote(remote_inputs_root + '/' + _WORKLOAD_SOURCE_TAR_NAME)} "
            f"{shlex.quote(remote_source)} "
            f"{shlex.quote(remote_inputs_root + '/source_inventory.json')} "
            f"{shlex.quote(identities['source_tree_sha256'])} "
            f"{shlex.quote(identities['source_tar_sha256'])}"
        ),
    ]
    return " && ".join(commands)


def _workload_package_script(
    name: str,
    remote_authority_root: str,
) -> str:
    if name == "engine_correctness":
        entries = _WORKLOAD_ENGINE_PACKAGE_ENTRIES
        return " && ".join([
            "set -eu",
            f"cd {shlex.quote(remote_authority_root)}",
            "test \"$(find . -mindepth 1 -maxdepth 1 | wc -l)\" -eq 4",
            *[
                f"test -e {shlex.quote(entry)}"
                for entry in entries
            ],
            f"tar -cf - {' '.join(entries)}",
        ])
    entries = _WORKLOAD_CACHED_PACKAGE_ENTRIES
    return " && ".join([
        "set -eu",
        f"cd {shlex.quote(remote_authority_root)}",
        "test \"$(find . -mindepth 1 -maxdepth 1 | wc -l)\" -eq 2",
        "test -d cached_continuation_authority",
        "test -f cached_continuation_independent_verification.json",
        f"tar -cf - {' '.join(entries)}",
    ])


def _workload_extract_argv(
    name: str,
    interpreter: str,
    local_tar: Path,
    destination: Path,
) -> list[str]:
    entries = (
        _WORKLOAD_ENGINE_PACKAGE_ENTRIES
        if name == "engine_correctness"
        else _WORKLOAD_CACHED_PACKAGE_ENTRIES
    )
    mismatch = (
        "authority inventory mismatch"
        if name == "engine_correctness"
        else "cached authority inventory mismatch"
    )
    unsafe = (
        "unsafe authority tar member"
        if name == "engine_correctness"
        else "unsafe cached authority tar member"
    )
    script = "\n".join([
        "import sys,tarfile",
        "from pathlib import Path,PurePosixPath",
        "archive=Path(sys.argv[1])",
        "destination=Path(sys.argv[2])",
        f"expected={list(entries)!r}",
        "if destination.exists(): raise SystemExit('download destination exists')",
        "with tarfile.open(archive,'r:') as handle:",
        " members=handle.getmembers()",
        " roots=sorted({PurePosixPath(member.name).parts[0] for member in members})",
        f" if roots!=sorted(expected): raise SystemExit({mismatch!r})",
        " for member in members:",
        "  path=PurePosixPath(member.name)",
        "  if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk():",
        f"   raise SystemExit({unsafe!r})",
        " destination.mkdir()",
        " handle.extractall(destination,members=members)",
    ])
    return [
        interpreter,
        "-c",
        script,
        str(local_tar),
        str(destination),
    ]


def _workload_prepare_local_verifier_argv(
    interpreter: str,
    source_tar: Path,
    source_inventory: Path,
    source_tree_sha256: str,
    destination: Path,
) -> list[str]:
    script = "\n".join([
        "import hashlib,json,sys,tarfile",
        "from pathlib import Path,PurePosixPath",
        "archive=Path(sys.argv[1])",
        "inventory=json.loads(Path(sys.argv[2]).read_text())",
        "expected=sys.argv[3]",
        "destination=Path(sys.argv[4])",
        "if destination.exists(): raise SystemExit('local verifier source exists')",
        "with tarfile.open(archive,'r:') as handle:",
        " members=handle.getmembers()",
        " if [member.name for member in members]!=inventory['owned_files']:",
        "  raise SystemExit('local verifier source inventory mismatch')",
        " for member in members:",
        "  path=PurePosixPath(member.name)",
        "  if path.is_absolute() or '..' in path.parts or not member.isfile() or member.issym() or member.islnk():",
        "   raise SystemExit('unsafe local verifier source member')",
        " destination.mkdir()",
        " handle.extractall(destination,members=members)",
        "digest=hashlib.sha256()",
        "for name in inventory['owned_files']:",
        " path=destination.joinpath(*PurePosixPath(name).parts)",
        " encoded=name.encode('utf-8')",
        " digest.update(len(encoded).to_bytes(8,'big'))",
        " digest.update(encoded)",
        " with path.open('rb') as source:",
        "  for chunk in iter(lambda:source.read(1024*1024),b''):",
        "   digest.update(chunk)",
        "if digest.hexdigest()!=expected:",
        " raise SystemExit('local verifier source tree mismatch')",
    ])
    return [
        interpreter,
        "-c",
        script,
        str(source_tar),
        str(source_inventory),
        source_tree_sha256,
        str(destination),
    ]


def _workload_local_verify_argv(
    name: str,
    interpreter: str,
    verifier_source: Path,
    downloaded_authority: Path,
) -> list[str]:
    if name == "engine_correctness":
        return [
            interpreter,
            str(
                verifier_source
                / "tools"
                / "verify_qwen35_tp4_engine_correctness_authority.py"
            ),
            str(downloaded_authority),
        ]
    script = "\n".join([
        "import importlib.util,json,sys",
        "from pathlib import Path",
        "source=Path(sys.argv[1])",
        "authority=Path(sys.argv[2])",
        "remote_path=Path(sys.argv[3])",
        "spec=importlib.util.spec_from_file_location('cached_verifier',source)",
        "module=importlib.util.module_from_spec(spec)",
        "sys.modules[spec.name]=module",
        "spec.loader.exec_module(module)",
        "local=module.verify_run(authority)",
        "remote=json.loads(remote_path.read_text(encoding='utf-8'))",
        "if local!=remote: raise SystemExit('cached verification payload mismatch')",
        "print(json.dumps(local,sort_keys=True,separators=(',',':')))",
    ])
    return [
        interpreter,
        "-c",
        script,
        str(
            verifier_source
            / "tools"
            / "verify_qwen35_tp4_cached_continuation_correctness_gate.py"
        ),
        str(
            downloaded_authority
            / "cached_continuation_authority"
        ),
        str(
            downloaded_authority
            / "cached_continuation_independent_verification.json"
        ),
    ]


def _workload_without_control_path_none(value: object) -> object:
    if not isinstance(value, list):
        if isinstance(value, Mapping):
            return {
                key: _workload_without_control_path_none(item)
                for key, item in value.items()
            }
        return value
    result = []
    index = 0
    while index < len(value):
        if value[index:index + 2] == ["-o", "ControlPath=none"]:
            index += 2
            continue
        result.append(
            _workload_without_control_path_none(value[index])
        )
        index += 1
    return result


def _workload_shared_local_interpreter(
    commands: Mapping[str, object],
    name: str,
) -> str:
    interpreters = []
    for command_name in (
        "safe_extract",
        "prepare_local_verifier",
        "local_verify",
    ):
        command = commands.get(command_name)
        argv = command.get("argv") if isinstance(command, Mapping) else None
        if not isinstance(argv, list) or not argv:
            raise ValueError(
                f"{name} execution plan local Python command is invalid"
            )
        interpreter = argv[0]
        if (
            not isinstance(interpreter, str)
            or not Path(interpreter).is_absolute()
            or not Path(interpreter).name.startswith("python")
        ):
            raise ValueError(
                f"{name} execution plan local Python must be absolute"
            )
        interpreters.append(interpreter)
    if len(set(interpreters)) != 1:
        raise ValueError(
            f"{name} execution plan local Python must be shared"
        )
    return interpreters[0]


def _expected_workload_commands(
    name: str,
    plan: Mapping[str, object],
    interpreter: str,
) -> dict[str, object]:
    run_tag = _require_safe_nonce(
        plan.get("run_tag"),
        f"{name} execution plan",
    )
    gpu_indices = _require_gpu_indices(
        plan.get("gpu_indices"),
        f"{name} execution plan",
    )
    ports = _require_ports(
        plan.get("ports"),
        f"{name} execution plan",
    )
    local_inputs = _require_exact_fields(
        plan.get("local_inputs"),
        {
            "configuration",
            "configuration_sha256",
            "source_inventory",
            "source_inventory_sha256",
            "source_tar",
            "source_tar_sha256",
            "workload_manifest",
            "workload_manifest_sha256",
            *(
                {
                    "resource_baseline",
                    "resource_baseline_sha256",
                }
                if "resource_policy" in plan
                else set()
            ),
        },
        f"{name} execution plan local inputs",
    )
    output_dir = Path(
        _require_nonempty_string(
            local_inputs["configuration"],
            f"{name} execution plan configuration",
        )
    ).parent
    if not output_dir.is_absolute():
        raise ValueError(
            f"{name} execution plan output directory is invalid"
        )
    expected_local_paths = {
        "configuration": (
            output_dir / _WORKLOAD_REMOTE_CONFIGURATION_NAME
        ),
        "source_tar": output_dir / _WORKLOAD_SOURCE_TAR_NAME,
    }
    if "resource_policy" in plan:
        expected_local_paths["resource_baseline"] = (
            output_dir / _WORKLOAD_RESOURCE_BASELINE_NAME
        )
    if any(
        Path(local_inputs[key]) != expected_path
        for key, expected_path in expected_local_paths.items()
    ):
        raise ValueError(
            f"{name} execution plan local input layout is invalid"
        )
    remote_run_root = (
        f"{_WORKLOAD_REMOTE_ROOTS[name]}/{run_tag}"
    )
    remote_inputs_root = f"{remote_run_root}/inputs"
    remote_source_root = f"{remote_run_root}/source"
    remote_authority_root = f"{remote_run_root}/authority"
    expected_remote_inputs = {
        "configuration": (
            f"{remote_inputs_root}/"
            f"{_WORKLOAD_REMOTE_CONFIGURATION_NAME}"
        ),
        "source_inventory": (
            f"{remote_inputs_root}/source_inventory.json"
        ),
        "source_tar": (
            f"{remote_inputs_root}/{_WORKLOAD_SOURCE_TAR_NAME}"
        ),
        "workload_manifest": (
            f"{remote_inputs_root}/workload_manifest.json"
        ),
    }
    controlled_shared = "resource_policy" in plan
    if controlled_shared:
        expected_remote_inputs["resource_baseline"] = (
            f"{remote_inputs_root}/"
            f"{_WORKLOAD_RESOURCE_BASELINE_NAME}"
        )
    if (
        plan.get("ssh_target") != _WORKLOAD_SSH_TARGET
        or plan.get("remote_run_root") != remote_run_root
        or plan.get("remote_source_root") != remote_source_root
        or plan.get("remote_authority_root")
        != remote_authority_root
        or plan.get("remote_inputs") != expected_remote_inputs
    ):
        raise ValueError(
            f"{name} execution plan native layout is invalid"
        )
    if name == "cached_continuation":
        if (
            plan.get("remote_cached_authority_dir")
            != (
                f"{remote_authority_root}/"
                "cached_continuation_authority"
            )
            or plan.get("remote_cached_verification_path")
            != (
                f"{remote_authority_root}/"
                "cached_continuation_independent_verification.json"
            )
        ):
            raise ValueError(
                f"{name} execution plan native layout is invalid"
            )
    identities = {
        key: _require_sha256(
            local_inputs[key],
            f"{name} execution plan {key}",
        )
        for key in (
            "configuration_sha256",
            "source_inventory_sha256",
            "source_tar_sha256",
            "workload_manifest_sha256",
        )
    }
    identities["source_tree_sha256"] = _require_sha256(
        plan.get("source_tree_sha256"),
        f"{name} execution plan source tree",
    )
    if controlled_shared:
        resource_policy = plan.get("resource_policy")
        resource_baseline_sha256 = _require_sha256(
            plan.get("resource_baseline_sha256"),
            f"{name} execution plan resource baseline",
        )
        if (
            resource_policy != "controlled_shared"
            or local_inputs["resource_baseline_sha256"]
            != resource_baseline_sha256
        ):
            raise ValueError(
                f"{name} execution plan resource policy is invalid"
            )
        identities["resource_baseline_sha256"] = (
            resource_baseline_sha256
        )
    else:
        resource_policy = "strict_exclusive"
        resource_baseline_sha256 = None
    authority_argv = [
        "env",
        f"PYTHONPATH={remote_source_root}",
        "PYTHONDONTWRITEBYTECODE=1",
        "TORCH_COMPILE_DISABLE=1",
        "CUDA_VISIBLE_DEVICES="
        + ",".join(str(value) for value in gpu_indices),
        f"TINYVLLM_DIST_PORT={ports['dist_port']}",
        f"MASTER_PORT={ports['master_port']}",
        REMOTE_PYTHON,
        (
            f"{remote_source_root}/tools/"
            + (
                "run_qwen35_tp4_cached_continuation_authority.py"
                if name == "cached_continuation"
                else "run_qwen35_tp4_engine_correctness_authority.py"
            )
        ),
        "--configuration",
        expected_remote_inputs["configuration"],
        "--source-inventory",
        expected_remote_inputs["source_inventory"],
    ]
    if name == "cached_continuation":
        authority_argv.extend([
            "--output-dir",
            plan["remote_cached_authority_dir"],
            "--verification-path",
            plan["remote_cached_verification_path"],
        ])
    else:
        authority_argv.extend([
            "--output-root",
            remote_authority_root,
        ])
    reserve_parts = [
        "set -eu",
        f"test ! -e {shlex.quote(remote_run_root)}",
        f"mkdir -p {shlex.quote(remote_run_root)}",
        f"mkdir {shlex.quote(remote_inputs_root)}",
    ]
    if name == "cached_continuation":
        reserve_parts.append(
            f"mkdir {shlex.quote(remote_authority_root)}"
        )
    upload_keys = [
        "configuration",
        "source_inventory",
        "source_tar",
        "workload_manifest",
    ]
    if controlled_shared:
        upload_keys.append("resource_baseline")
    if resource_policy == "strict_exclusive":
        resource_shell = _strict_workload_resource_guard_shell(
            gpu_indices
        )
    else:
        resource_shell = (
            _controlled_shared_workload_resource_guard_shell(
                plan,
                gpu_indices,
            )
        )
    guarded_shell = (
        f"final_resource=\"$({resource_shell})\" && "
        "printf 'QWEN35_FINAL_RESOURCE_JSON=%s\\n' "
        "\"$final_resource\" && exec "
        f"{shlex.join(authority_argv)}"
    )
    local_tar = (
        output_dir / _WORKLOAD_PACKAGE_TAR_NAMES[name]
    )
    downloaded = (
        output_dir / _WORKLOAD_DOWNLOADED_AUTHORITY_NAMES[name]
    )
    verifier_source = (
        output_dir / _WORKLOAD_LOCAL_VERIFIER_SOURCE_NAME
    )
    resource_guard = {
        "argv": _workload_native_ssh_argv([
            "bash",
            "-lc",
            resource_shell,
        ]),
        "gpu_indices": gpu_indices,
        "minimum_free_bytes_per_gpu": MIN_GPU_FREE_BYTES,
        "requires_no_active_compute_processes": (
            resource_policy == "strict_exclusive"
        ),
    }
    if controlled_shared:
        resource_guard.update({
            "resource_policy": resource_policy,
            "resource_baseline_sha256": (
                resource_baseline_sha256
            ),
        })
    return {
        "reserve_remote": {
            "argv": _workload_native_ssh_argv([
                "bash",
                "-lc",
                " && ".join(reserve_parts),
            ]),
        },
        "upload": {
            "argv": [
                _workload_native_scp_argv(
                    local_inputs[key],
                    expected_remote_inputs[key],
                )
                for key in upload_keys
            ],
        },
        "stage": {
            "argv": _workload_native_ssh_argv([
                "bash",
                "-lc",
                _workload_stage_script(
                    remote_source_root,
                    remote_inputs_root,
                    identities,
                    resource_baseline_name=(
                        _WORKLOAD_RESOURCE_BASELINE_NAME
                        if controlled_shared
                        else None
                    ),
                ),
            ]),
        },
        "resource_guard": resource_guard,
        "guarded_authority": {
            "authority_argv": authority_argv,
            "ssh_argv": _workload_native_ssh_argv([
                "bash",
                "-lc",
                guarded_shell,
            ]),
            "final_resource_recheck": True,
        },
        "package_download": {
            "remote_argv": _workload_native_ssh_argv([
                "bash",
                "-lc",
                _workload_package_script(
                    name,
                    remote_authority_root,
                ),
            ]),
            "local_output": str(local_tar),
        },
        "safe_extract": {
            "argv": _workload_extract_argv(
                name,
                interpreter,
                local_tar,
                downloaded,
            ),
        },
        "prepare_local_verifier": {
            "argv": _workload_prepare_local_verifier_argv(
                interpreter,
                Path(local_inputs["source_tar"]),
                Path(local_inputs["source_inventory"]),
                identities["source_tree_sha256"],
                verifier_source,
            ),
            "source_tar": local_inputs["source_tar"],
            "source_inventory": local_inputs["source_inventory"],
            "source_tree_sha256": identities[
                "source_tree_sha256"
            ],
        },
        "local_verify": {
            "argv": _workload_local_verify_argv(
                name,
                interpreter,
                verifier_source,
                downloaded,
            ),
        },
    }


def _validate_workload_prerequisite_commands(
    name: str,
    plan: Mapping[str, object],
) -> None:
    if name not in _WORKLOAD_REMOTE_ROOTS:
        raise ValueError(f"{name} execution plan command is invalid")
    commands = plan.get("commands")
    if not isinstance(commands, Mapping):
        raise ValueError(f"{name} execution plan command is invalid")
    interpreter = _workload_shared_local_interpreter(commands, name)
    expected = _expected_workload_commands(
        name,
        plan,
        interpreter,
    )
    if commands not in (
        expected,
        _workload_without_control_path_none(expected),
    ):
        raise ValueError(
            f"{name} execution plan command mapping is invalid"
        )


def _parse_prerequisite_json_log(
    value: object,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} JSON is invalid") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} JSON must be an object")
    return payload


def _validate_resource_receipt(
    value: object,
    gpu_indices: list[int],
    label: str,
    *,
    resource_policy: str = "strict_exclusive",
    baseline: Mapping[str, object] | None = None,
    baseline_sha256: str | None = None,
) -> tuple[tuple[int, str], ...]:
    payload = (
        _parse_prerequisite_json_log(value, label)
        if isinstance(value, str)
        else value
    )
    controlled_shared = resource_policy == _ROOT_RESOURCE_POLICY
    resource = _require_exact_fields(
        payload,
        (
            {
                "classification",
                "resource_policy",
                "baseline_sha256",
                "selected",
                "benchmark_execution_authorized",
            }
            if controlled_shared
            else {"classification", "selected"}
        ),
        label,
    )
    selected = resource["selected"]
    if (
        resource["classification"] != "READY"
        or (
            controlled_shared
            and (
                resource["resource_policy"] != _ROOT_RESOURCE_POLICY
                or resource["baseline_sha256"] != baseline_sha256
                or resource["benchmark_execution_authorized"] is not False
                or not isinstance(baseline, Mapping)
            )
        )
        or not isinstance(selected, list)
        or len(selected) != 4
        or [row.get("gpu_index") for row in selected] != gpu_indices
    ):
        raise ValueError(f"{label} GPU identity is invalid")
    identities = []
    for row in selected:
        fields = {
            "gpu_index",
            "gpu_uuid",
            "free_bytes",
            "compute_processes",
        }
        if (
            not isinstance(row, Mapping)
            or set(row) != fields
            or not isinstance(row["gpu_uuid"], str)
            or not row["gpu_uuid"]
            or isinstance(row["free_bytes"], bool)
            or not isinstance(row["free_bytes"], int)
            or row["free_bytes"] < 24 * 1024**3
            or not isinstance(row["compute_processes"], list)
            or (
                not controlled_shared
                and row["compute_processes"] != []
            )
        ):
            raise ValueError(f"{label} resource result is invalid")
        processes = []
        for process in row["compute_processes"]:
            processes.append(
                _validate_root_process(process, f"{label} process")
            )
        if len(processes) != len(set(processes)):
            raise ValueError(f"{label} process inventory is duplicated")
        identities.append((row["gpu_index"], row["gpu_uuid"]))
    if len({uuid for _, uuid in identities}) != 4:
        raise ValueError(f"{label} GPU identity is invalid")
    if controlled_shared:
        frozen = baseline.get("selected")
        if (
            not isinstance(frozen, list)
            or len(frozen) != 4
            or [row.get("gpu_index") for row in frozen] != gpu_indices
        ):
            raise ValueError(f"{label} baseline GPU identity is invalid")
        for observed_row, frozen_row in zip(selected, frozen):
            if (
                not isinstance(frozen_row, Mapping)
                or frozen_row.get("gpu_uuid") != observed_row["gpu_uuid"]
                or not isinstance(
                    frozen_row.get("compute_processes"),
                    list,
                )
            ):
                raise ValueError(f"{label} baseline GPU identity is invalid")
            allowed = {
                _validate_root_process(
                    process,
                    f"{label} baseline process",
                )
                for process in frozen_row["compute_processes"]
            }
            observed = {
                _validate_root_process(process, f"{label} process")
                for process in observed_row["compute_processes"]
            }
            if not observed.issubset(allowed):
                raise ValueError(f"{label} process drift")
    return tuple(identities)


def _root_expected_stage_inputs(
    plan: Mapping[str, object],
    *,
    resource_binding: Mapping[str, object] | None,
) -> dict[str, dict[str, object]]:
    common = {
        "run_tag": plan["run_tag"],
        "repo_root": plan["repo_root"],
    }
    local_artifact_dir = f"{plan['local_run_dir']}/artifacts"
    result = {
        "preflight": dict(common),
        "run": {
            **common,
            "remote_run_dir": plan["remote_run_dir"],
            "frozen_source_tree_sha256": (
                plan["frozen_source_tree_sha256"]
            ),
        },
        "download": {
            **common,
            "remote_run_dir": plan["remote_run_dir"],
            "local_artifact_dir": local_artifact_dir,
            "exact_artifact_names": list(
                _ROOT_EXACT_ARTIFACT_NAMES
            ),
        },
        "verify": {
            **common,
            "local_artifact_dir": local_artifact_dir,
            "independent_verification_path": (
                f"{plan['local_run_dir']}/independent_verification.json"
            ),
            "frozen_source_tree_sha256": (
                plan["frozen_source_tree_sha256"]
            ),
            "model_manifest_sha256": MODEL_MANIFEST_SHA256,
        },
    }
    if resource_binding is not None:
        for stage in ("preflight", "run"):
            result[stage].update(resource_binding)
    return result


def _validate_root_process(
    value: object,
    label: str,
) -> tuple[int, str, int]:
    process = _require_exact_fields(
        value,
        {
            "pid",
            "process_name",
            "used_memory_mib",
            "start_time_ticks",
        },
        label,
    )
    if (
        isinstance(process["pid"], bool)
        or not isinstance(process["pid"], int)
        or process["pid"] <= 0
        or not isinstance(process["process_name"], str)
        or not process["process_name"]
        or isinstance(process["used_memory_mib"], bool)
        or not isinstance(process["used_memory_mib"], int)
        or process["used_memory_mib"] < 0
        or isinstance(process["start_time_ticks"], bool)
        or not isinstance(process["start_time_ticks"], int)
        or process["start_time_ticks"] <= 0
    ):
        raise ValueError(f"{label} schema is invalid")
    return (
        process["pid"],
        process["process_name"],
        process["start_time_ticks"],
    )


def _validate_root_query_process(
    value: object,
    label: str,
) -> tuple[int, str]:
    process = _require_exact_fields(
        value,
        {
            "pid",
            "process_name",
            "used_bytes",
        },
        label,
    )
    if (
        isinstance(process["pid"], bool)
        or not isinstance(process["pid"], int)
        or process["pid"] <= 0
        or not isinstance(process["process_name"], str)
        or not process["process_name"]
        or isinstance(process["used_bytes"], bool)
        or not isinstance(process["used_bytes"], int)
        or process["used_bytes"] < 0
    ):
        raise ValueError(f"{label} schema is invalid")
    return process["pid"], process["process_name"]


def _validate_root_selected_rows(
    value: object,
    *,
    controlled_shared: bool,
    gpu_indices: list[int] | None,
    gpu_uuids: list[str] | None,
    label: str,
) -> tuple[
    tuple[int, str],
    tuple[frozenset[tuple[int, str, int]], ...],
]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label} GPU identity is invalid")
    expected_indices = gpu_indices
    expected_uuids = gpu_uuids
    identities = []
    process_inventories = []
    for rank, value_row in enumerate(value):
        resource_fields = {
            "rank",
            "world_size",
            "gpu_index",
            "gpu_uuid",
            "free_bytes",
            "compute_processes",
            "minimum_free_bytes",
        }
        if not controlled_shared:
            resource_fields |= {
                "gpu_name",
                "total_bytes",
            }
        row = _require_exact_fields(
            value_row,
            resource_fields,
            label,
        )
        if (
            row["rank"] != rank
            or row["world_size"] != 4
            or row["minimum_free_bytes"] != _ROOT_MIN_GPU_FREE_BYTES
            or isinstance(row["gpu_index"], bool)
            or not isinstance(row["gpu_index"], int)
            or row["gpu_index"] < 0
            or not isinstance(row["gpu_uuid"], str)
            or not row["gpu_uuid"]
            or isinstance(row["free_bytes"], bool)
            or not isinstance(row["free_bytes"], int)
            or row["free_bytes"] < _ROOT_MIN_GPU_FREE_BYTES
            or not isinstance(row["compute_processes"], list)
        ):
            raise ValueError(f"{label} resource result is invalid")
        if not controlled_shared and (
            not isinstance(row["gpu_name"], str)
            or not row["gpu_name"]
            or isinstance(row["total_bytes"], bool)
            or not isinstance(row["total_bytes"], int)
            or row["total_bytes"] < row["free_bytes"]
        ):
            raise ValueError(f"{label} resource result is invalid")
        if (
            expected_indices is not None
            and row["gpu_index"] != expected_indices[rank]
        ):
            raise ValueError(f"{label} GPU identity is invalid")
        if (
            expected_uuids is not None
            and row["gpu_uuid"] != expected_uuids[rank]
        ):
            raise ValueError(f"{label} GPU identity is invalid")
        processes = tuple(
            _validate_root_process(
                process,
                f"{label} compute process",
            )
            for process in row["compute_processes"]
        )
        if len(processes) != len(set(processes)):
            raise ValueError(f"{label} compute process is duplicated")
        if not controlled_shared and processes:
            raise ValueError(f"{label} has active compute processes")
        identities.append((row["gpu_index"], row["gpu_uuid"]))
        process_inventories.append(frozenset(processes))
    if (
        len({index for index, _ in identities}) != 4
        or len({uuid for _, uuid in identities}) != 4
    ):
        raise ValueError(f"{label} GPU identity is invalid")
    return tuple(identities), tuple(process_inventories)


def _validate_root_preflight_rows(
    value: object,
    *,
    selected: list[object],
    controlled_shared: bool,
    label: str,
) -> None:
    if controlled_shared:
        if value != selected:
            raise ValueError(f"{label} row identity is invalid")
        return
    if not isinstance(value, list) or len(value) < 4:
        raise ValueError(f"{label} row identity is invalid")
    raw_by_identity = {}
    for value_row in value:
        row = _require_exact_fields(
            value_row,
            {
                "gpu_index",
                "gpu_uuid",
                "gpu_name",
                "total_bytes",
                "free_bytes",
                "compute_processes",
            },
            label,
        )
        if (
            isinstance(row["gpu_index"], bool)
            or not isinstance(row["gpu_index"], int)
            or row["gpu_index"] < 0
            or not isinstance(row["gpu_uuid"], str)
            or not row["gpu_uuid"]
            or not isinstance(row["gpu_name"], str)
            or not row["gpu_name"]
            or isinstance(row["total_bytes"], bool)
            or not isinstance(row["total_bytes"], int)
            or isinstance(row["free_bytes"], bool)
            or not isinstance(row["free_bytes"], int)
            or row["total_bytes"] < row["free_bytes"]
            or not isinstance(row["compute_processes"], list)
        ):
            raise ValueError(f"{label} resource result is invalid")
        processes = tuple(
            _validate_root_query_process(
                process,
                f"{label} compute process",
            )
            for process in row["compute_processes"]
        )
        if len(processes) != len(set(processes)):
            raise ValueError(f"{label} compute process is duplicated")
        identity = (row["gpu_index"], row["gpu_uuid"])
        if identity in raw_by_identity:
            raise ValueError(f"{label} GPU identity is invalid")
        raw_by_identity[identity] = row
    for selected_row in selected:
        raw = raw_by_identity.get(
            (selected_row["gpu_index"], selected_row["gpu_uuid"])
        )
        if (
            raw is None
            or any(
                selected_row[field] != raw[field]
                for field in (
                    "gpu_name",
                    "total_bytes",
                    "free_bytes",
                    "compute_processes",
                )
            )
        ):
            raise ValueError(f"{label} selected identity is invalid")


def _load_root_resource_baseline(
    plan: Mapping[str, object],
    *,
    evidence_root: Path,
) -> tuple[frozenset[tuple[int, str, int]], ...]:
    baseline_path = Path(plan["resource_baseline_path"])
    if not baseline_path.is_absolute():
        baseline_path = evidence_root / baseline_path
    if (
        not baseline_path.is_file()
        or baseline_path.is_symlink()
        or sha256_file(baseline_path)
        != plan["resource_baseline_sha256"]
    ):
        raise ValueError(
            "tp4_root_logit resource baseline SHA mismatch"
        )
    try:
        baseline = json.loads(
            baseline_path.read_text(encoding="utf-8")
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError(
            "tp4_root_logit resource baseline JSON is invalid"
        ) from error
    baseline = _require_exact_fields(
        baseline,
        {
            "schema_version",
            "classification",
            "ssh_target",
            "captured_at",
            "gpu_indices",
            "selected",
            "minimum_free_bytes_per_gpu",
            "benchmark_execution_authorized",
        },
        "tp4_root_logit resource baseline",
    )
    if (
        baseline["schema_version"]
        != "qwen35.tp4-controlled-shared-resource-baseline.v1"
        or baseline["classification"] != "READY"
        or baseline["ssh_target"] != plan["ssh_target"]
        or not isinstance(baseline["captured_at"], str)
        or not baseline["captured_at"]
        or baseline["gpu_indices"] != plan["gpu_indices"]
        or baseline["minimum_free_bytes_per_gpu"]
        != _ROOT_MIN_GPU_FREE_BYTES
        or baseline["benchmark_execution_authorized"] is not False
        or not isinstance(baseline["selected"], list)
        or len(baseline["selected"]) != 4
    ):
        raise ValueError(
            "tp4_root_logit resource baseline binding is invalid"
        )
    inventories = []
    for rank, value_row in enumerate(baseline["selected"]):
        row = _require_exact_fields(
            value_row,
            {
                "gpu_index",
                "gpu_uuid",
                "free_bytes",
                "compute_processes",
            },
            "tp4_root_logit resource baseline row",
        )
        if (
            row["gpu_index"] != plan["gpu_indices"][rank]
            or row["gpu_uuid"] != plan["gpu_uuids"][rank]
            or isinstance(row["free_bytes"], bool)
            or not isinstance(row["free_bytes"], int)
            or row["free_bytes"] < _ROOT_MIN_GPU_FREE_BYTES
            or not isinstance(row["compute_processes"], list)
        ):
            raise ValueError(
                "tp4_root_logit resource baseline binding is invalid"
            )
        processes = tuple(
            _validate_root_process(
                process,
                "tp4_root_logit resource baseline process",
            )
            for process in row["compute_processes"]
        )
        if len(processes) != len(set(processes)):
            raise ValueError(
                "tp4_root_logit resource baseline process is duplicated"
            )
        inventories.append(frozenset(processes))
    return tuple(inventories)


def _validate_root_final_resource(
    value: object,
    *,
    plan: Mapping[str, object],
    preflight_identities: tuple[tuple[int, str], ...],
    baseline_processes: tuple[
        frozenset[tuple[int, str, int]],
        ...,
    ],
) -> None:
    resource = _require_exact_fields(
        value,
        {
            "classification",
            "resource_policy",
            "baseline_sha256",
            "selected",
            "benchmark_execution_authorized",
        },
        "tp4_root_logit final resource",
    )
    if (
        resource["classification"] != "READY"
        or resource["resource_policy"] != _ROOT_RESOURCE_POLICY
        or resource["baseline_sha256"]
        != plan["resource_baseline_sha256"]
        or resource["benchmark_execution_authorized"] is not False
        or not isinstance(resource["selected"], list)
        or len(resource["selected"]) != 4
    ):
        raise ValueError(
            "tp4_root_logit final resource binding is invalid"
        )
    final_identities = []
    for rank, value_row in enumerate(resource["selected"]):
        row = _require_exact_fields(
            value_row,
            {
                "gpu_index",
                "gpu_uuid",
                "free_bytes",
                "compute_processes",
            },
            "tp4_root_logit final resource row",
        )
        if (
            (row["gpu_index"], row["gpu_uuid"])
            != preflight_identities[rank]
            or isinstance(row["free_bytes"], bool)
            or not isinstance(row["free_bytes"], int)
            or row["free_bytes"] < _ROOT_MIN_GPU_FREE_BYTES
            or not isinstance(row["compute_processes"], list)
        ):
            raise ValueError(
                "tp4_root_logit final resource binding is invalid"
            )
        processes = tuple(
            _validate_root_process(
                process,
                "tp4_root_logit final resource compute process",
            )
            for process in row["compute_processes"]
        )
        if (
            len(processes) != len(set(processes))
            or not set(processes).issubset(
                baseline_processes[rank]
            )
        ):
            raise ValueError(
                "tp4_root_logit final resource process is invalid"
            )
        final_identities.append(
            (row["gpu_index"], row["gpu_uuid"])
        )
    if tuple(final_identities) != preflight_identities:
        raise ValueError(
            "tp4_root_logit final resource GPU drift"
        )


def _validate_root_prerequisite_documents(
    plan: object,
    authorization: object,
    receipt: object,
    *,
    run_tag: object,
    source_tree_sha256: object,
    verification: Mapping[str, object],
    evidence_root: Path,
) -> None:
    plan_fields = {
        "schema_version",
        "run_tag",
        "repo_root",
        "local_run_dir",
        "ssh_target",
        "remote_run_dir",
        "frozen_source_tag",
        "frozen_source_tree_sha256",
        "model_manifest_sha256",
        "exact_artifact_names",
        "minimum_free_bytes_per_gpu",
        "requires_no_active_compute_processes",
        "stage_order",
        "stage_inputs",
        "execution_performed",
        "claim_boundary",
        "plan_output_dir",
    }
    optional_resource_fields = {
        "resource_policy",
        "resource_baseline_path",
        "resource_baseline_sha256",
        "gpu_indices",
        "gpu_uuids",
        "benchmark_execution_authorized",
    }
    actual_plan = (
        set(plan) if isinstance(plan, Mapping) else set()
    )
    if "workload_manifest_sha256" in actual_plan:
        raise ValueError(
            "tp4_root_logit workload identity is forbidden"
        )
    controlled_shared = "resource_policy" in actual_plan
    expected_plan_fields = (
        plan_fields | optional_resource_fields
        if controlled_shared
        else plan_fields
    )
    plan = _require_exact_fields(
        plan,
        expected_plan_fields,
        "tp4_root_logit execution plan",
    )
    stage_order = list(_ROOT_STAGE_ORDER)
    expected_local_run_dir = (
        f"{str(plan['repo_root']).rstrip('/')}/"
        f"{_ROOT_LOCAL_RUN_ROOT}/{run_tag}"
    )
    expected_remote_run_dir = f"{_ROOT_REMOTE_GATE_ROOT}/{run_tag}"
    if (
        plan["schema_version"] != _ROOT_PLAN_SCHEMA
        or plan["run_tag"] != run_tag
        or plan["local_run_dir"] != expected_local_run_dir
        or plan["ssh_target"] != _ROOT_REMOTE_TARGET
        or plan["remote_run_dir"] != expected_remote_run_dir
        or not isinstance(plan["frozen_source_tag"], str)
        or not plan["frozen_source_tag"].startswith(
            _ROOT_FROZEN_SOURCE_TAG_PREFIX
        )
        or any(
            character not in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789-_."
            )
            for character in plan["frozen_source_tag"]
        )
        or plan["frozen_source_tree_sha256"] != source_tree_sha256
        or plan["model_manifest_sha256"] != MODEL_MANIFEST_SHA256
        or plan["exact_artifact_names"]
        != list(_ROOT_EXACT_ARTIFACT_NAMES)
        or plan["stage_order"] != stage_order
        or not isinstance(plan["stage_inputs"], Mapping)
        or set(plan["stage_inputs"]) != set(stage_order)
        or any(
            not isinstance(plan["stage_inputs"][stage], Mapping)
            for stage in stage_order
        )
        or plan["minimum_free_bytes_per_gpu"]
        != _ROOT_MIN_GPU_FREE_BYTES
        or plan["execution_performed"] is not False
        or plan["claim_boundary"] != _ROOT_CLAIM_BOUNDARY
    ):
        raise ValueError("tp4_root_logit execution plan identity is invalid")
    resource_binding = None
    if controlled_shared:
        expected_baseline_path = (
            f"{str(plan['plan_output_dir']).rstrip('/')}/"
            f"{_ROOT_RESOURCE_BASELINE_NAME}"
        )
        if (
            plan["resource_policy"] != _ROOT_RESOURCE_POLICY
            or plan["resource_baseline_path"]
            != expected_baseline_path
            or not _valid_sha256(plan["resource_baseline_sha256"])
            or plan["benchmark_execution_authorized"] is not False
            or plan["requires_no_active_compute_processes"] is not False
        ):
            raise ValueError(
                "tp4_root_logit execution plan resource identity is invalid"
            )
        _require_gpu_indices(
            plan["gpu_indices"],
            "tp4_root_logit execution plan",
        )
        if (
            not isinstance(plan["gpu_uuids"], list)
            or len(plan["gpu_uuids"]) != 4
            or len(set(plan["gpu_uuids"])) != 4
        ):
            raise ValueError(
                "tp4_root_logit execution plan GPU identity is invalid"
            )
        if any(
            not isinstance(uuid, str) or not uuid
            for uuid in plan["gpu_uuids"]
        ):
            raise ValueError(
                "tp4_root_logit execution plan GPU identity is invalid"
            )
        resource_binding = {
            "resource_policy": plan["resource_policy"],
            "resource_baseline_path": plan["resource_baseline_path"],
            "resource_baseline_sha256": (
                plan["resource_baseline_sha256"]
            ),
            "gpu_indices": plan["gpu_indices"],
            "gpu_uuids": plan["gpu_uuids"],
            "benchmark_execution_authorized": (
                plan["benchmark_execution_authorized"]
            ),
        }
    elif plan["requires_no_active_compute_processes"] is not True:
        raise ValueError(
            "tp4_root_logit execution plan resource identity is invalid"
        )
    if plan["stage_inputs"] != _root_expected_stage_inputs(
        plan,
        resource_binding=resource_binding,
    ):
        raise ValueError(
            "tp4_root_logit execution plan stage identity is invalid"
        )

    authorization_fields = {
        "schema_version",
        "classification",
        "plan_sha256",
        "run_tag",
        "ssh_target",
        "frozen_source_tree_sha256",
        "model_manifest_sha256",
        "stage_order",
        "nonce",
        "consumed",
    }
    if controlled_shared:
        authorization_fields |= {
            "resource_policy",
            "resource_baseline_sha256",
        }
    authorization = _require_exact_fields(
        authorization,
        authorization_fields,
        "tp4_root_logit consumed authorization",
    )
    if (
        authorization["schema_version"] != _ROOT_AUTHORIZATION_SCHEMA
        or authorization["classification"] != "AUTHORIZED"
        or authorization["plan_sha256"] != canonical_json_sha256(plan)
        or authorization["run_tag"] != run_tag
        or authorization["ssh_target"] != plan["ssh_target"]
        or authorization["frozen_source_tree_sha256"]
        != source_tree_sha256
        or authorization["model_manifest_sha256"]
        != MODEL_MANIFEST_SHA256
        or authorization["stage_order"] != stage_order
        or authorization["consumed"] is not True
        or "workload_manifest_sha256" in authorization
    ):
        raise ValueError(
            "tp4_root_logit consumed authorization binding is invalid"
        )
    _require_safe_nonce(
        authorization["nonce"],
        "tp4_root_logit consumed authorization",
    )
    if controlled_shared and (
        authorization["resource_policy"] != plan["resource_policy"]
        or authorization["resource_baseline_sha256"]
        != plan["resource_baseline_sha256"]
    ):
        raise ValueError(
            "tp4_root_logit consumed authorization resource binding is invalid"
        )

    receipt = _require_exact_fields(
        receipt,
        {
            "schema_version",
            "classification",
            "plan_sha256",
            "authorization_sha256",
            "authorization_nonce",
            "run_tag",
            "stages",
        },
        "tp4_root_logit execution receipt",
    )
    if (
        receipt["schema_version"] != _ROOT_RECEIPT_SCHEMA
        or receipt["classification"] != "PASS"
        or receipt["plan_sha256"] != canonical_json_sha256(plan)
        or receipt["authorization_sha256"]
        != canonical_json_sha256(authorization)
        or receipt["authorization_nonce"] != authorization["nonce"]
        or receipt["run_tag"] != run_tag
    ):
        raise ValueError(
            "tp4_root_logit execution receipt authorization binding is invalid"
        )
    stages = receipt["stages"]
    if (
        not isinstance(stages, list)
        or [row.get("name") for row in stages] != stage_order
    ):
        raise ValueError(
            "tp4_root_logit execution receipt stage order is invalid"
        )
    results = {}
    for row in stages:
        row = _require_exact_fields(
            row,
            {"name", "result_sha256", "result"},
            "tp4_root_logit execution receipt stage",
        )
        if (
            not isinstance(row["result"], Mapping)
            or row["result_sha256"]
            != canonical_json_sha256(row["result"])
        ):
            raise ValueError(
                "tp4_root_logit execution receipt stage result is invalid"
            )
        results[row["name"]] = row["result"]
    preflight_fields = {
        "status",
        "run_tag",
        "frozen_source_tag",
        "frozen_source_tree_sha256",
        "source_tree_sha256",
        "selected",
        "rows",
    }
    if controlled_shared:
        preflight_fields |= {
            "resource_policy",
            "baseline_sha256",
            "benchmark_execution_authorized",
        }
    preflight = _require_exact_fields(
        results["preflight"],
        preflight_fields,
        "tp4_root_logit execution receipt preflight",
    )
    selected = preflight.get("selected")
    if (
        preflight.get("status") != "READY"
        or preflight.get("run_tag") != run_tag
        or preflight.get("frozen_source_tag")
        != plan["frozen_source_tag"]
        or preflight.get("frozen_source_tree_sha256")
        != source_tree_sha256
        or preflight.get("source_tree_sha256") != source_tree_sha256
        or not isinstance(selected, list)
    ):
        raise ValueError(
            "tp4_root_logit execution receipt preflight is invalid"
        )
    if controlled_shared and (
        preflight["resource_policy"] != _ROOT_RESOURCE_POLICY
        or preflight["baseline_sha256"]
        != plan["resource_baseline_sha256"]
        or preflight["benchmark_execution_authorized"] is not False
    ):
        raise ValueError(
            "tp4_root_logit execution receipt preflight is invalid"
        )
    preflight_identities, preflight_processes = (
        _validate_root_selected_rows(
            selected,
            controlled_shared=controlled_shared,
            gpu_indices=(
                plan["gpu_indices"] if controlled_shared else None
            ),
            gpu_uuids=(
                plan["gpu_uuids"] if controlled_shared else None
            ),
            label="tp4_root_logit execution receipt preflight",
        )
    )
    _validate_root_preflight_rows(
        preflight["rows"],
        selected=selected,
        controlled_shared=controlled_shared,
        label="tp4_root_logit execution receipt preflight",
    )
    baseline_processes = preflight_processes
    if controlled_shared:
        baseline_processes = _load_root_resource_baseline(
            plan,
            evidence_root=evidence_root,
        )
        for observed, allowed in zip(
            preflight_processes,
            baseline_processes,
        ):
            if not observed.issubset(allowed):
                raise ValueError(
                    "tp4_root_logit execution receipt preflight "
                    "process is invalid"
                )
    expected_run = {
        "status": "REMOTE_PASS",
        "run_tag": run_tag,
        "remote_run_dir": plan["remote_run_dir"],
        "artifact_names": plan["exact_artifact_names"],
    }
    run_result = results["run"]
    if controlled_shared:
        run = _require_exact_fields(
            run_result,
            set(expected_run) | {"final_resource"},
            "tp4_root_logit execution receipt run",
        )
        if any(run[field] != value for field, value in expected_run.items()):
            raise ValueError(
                "tp4_root_logit execution receipt artifact binding is invalid"
            )
        _validate_root_final_resource(
            run["final_resource"],
            plan=plan,
            preflight_identities=preflight_identities,
            baseline_processes=baseline_processes,
        )
    elif run_result != expected_run:
        raise ValueError(
            "tp4_root_logit execution receipt artifact binding is invalid"
        )
    if results["download"] != {
        "status": "DOWNLOADED",
        "artifact_names": plan["exact_artifact_names"],
    }:
        raise ValueError(
            "tp4_root_logit execution receipt inventory is invalid"
        )
    receipt_verification = results["verify"]
    if (
        not isinstance(receipt_verification, Mapping)
        or set(receipt_verification)
        != {"classification", "case_ids", "ranks", "checks"}
        or any(
            receipt_verification.get(field) != verification.get(field)
            for field in ("classification", "case_ids", "ranks")
        )
        or isinstance(receipt_verification.get("checks"), bool)
        or not isinstance(receipt_verification.get("checks"), int)
        or receipt_verification["checks"] <= 0
        or isinstance(verification.get("checks"), bool)
        or not isinstance(verification.get("checks"), int)
        or verification["checks"] <= 0
    ):
        raise ValueError(
            "tp4_root_logit execution receipt verification binding is invalid"
        )


def _validate_workload_prerequisite_documents(
    name: str,
    plan: object,
    authorization: object,
    receipt: object,
    *,
    run_tag: object,
    source_tree_sha256: object,
    artifact: Mapping[str, object],
    verification: Mapping[str, object],
) -> None:
    is_cached = name == "cached_continuation"
    plan_fields = {
        "schema_version",
        "run_tag",
        "ssh_target",
        "remote_run_root",
        "remote_source_root",
        "remote_authority_root",
        "gpu_indices",
        "ports",
        "source_tree_sha256",
        "model_manifest_sha256",
        "local_inputs",
        "remote_inputs",
        "command_order",
        "commands",
        "execution_performed",
        "claim_boundary",
    }
    if is_cached:
        plan_fields |= {
            "remote_cached_authority_dir",
            "remote_cached_verification_path",
        }
    controlled_shared = (
        isinstance(plan, Mapping) and "resource_policy" in plan
    )
    if controlled_shared:
        plan_fields |= {
            "resource_policy",
            "resource_baseline_sha256",
        }
    plan = _require_exact_fields(
        plan,
        plan_fields,
        f"{name} execution plan",
    )
    expected_plan_schema = (
        _CACHED_PLAN_SCHEMA if is_cached else _ENGINE_PLAN_SCHEMA
    )
    if (
        plan["schema_version"] != expected_plan_schema
        or plan["command_order"] != list(_WORKLOAD_COMMAND_ORDER)
        or not isinstance(plan["commands"], Mapping)
        or set(plan["commands"]) != set(_WORKLOAD_COMMAND_ORDER)
        or any(
            not isinstance(plan["commands"][command], Mapping)
            for command in _WORKLOAD_COMMAND_ORDER
        )
        or plan["execution_performed"] is not False
    ):
        raise ValueError(f"{name} execution plan identity is invalid")
    _validate_workload_prerequisite_commands(name, plan)
    if plan["run_tag"] != run_tag:
        raise ValueError(f"{name} execution plan run tag is invalid")
    if plan["source_tree_sha256"] != source_tree_sha256:
        raise ValueError(f"{name} execution plan source identity is invalid")
    if plan["model_manifest_sha256"] != MODEL_MANIFEST_SHA256:
        raise ValueError(f"{name} execution plan model identity is invalid")
    gpu_indices = _require_gpu_indices(
        plan["gpu_indices"],
        f"{name} execution plan",
    )
    ports = _require_ports(plan["ports"], f"{name} execution plan")
    local_fields = {
        "configuration",
        "configuration_sha256",
        "source_inventory",
        "source_inventory_sha256",
        "source_tar",
        "source_tar_sha256",
        "workload_manifest",
        "workload_manifest_sha256",
    }
    if controlled_shared:
        local_fields |= {
            "resource_baseline",
            "resource_baseline_sha256",
        }
    raw_local_inputs = plan["local_inputs"]
    if (
        isinstance(raw_local_inputs, Mapping)
        and "workload_manifest_sha256" not in raw_local_inputs
    ):
        raise ValueError(f"{name} workload identity is missing")
    local_inputs = _require_exact_fields(
        raw_local_inputs,
        local_fields,
        f"{name} execution plan local inputs",
    )
    for field in (
        "configuration_sha256",
        "source_inventory_sha256",
        "source_tar_sha256",
    ):
        if not _valid_sha256(local_inputs[field]):
            raise ValueError(
                f"{name} execution plan local input SHA is invalid"
            )
    workload_sha256 = local_inputs["workload_manifest_sha256"]
    if workload_sha256 != APPROVED_V1_WORKLOAD_MANIFEST_SHA256:
        raise ValueError(f"{name} workload identity is invalid")
    remote_fields = {
        "configuration",
        "source_inventory",
        "source_tar",
        "workload_manifest",
    }
    if controlled_shared:
        remote_fields.add("resource_baseline")
    _require_exact_fields(
        plan["remote_inputs"],
        remote_fields,
        f"{name} execution plan remote inputs",
    )
    if controlled_shared and (
        plan["resource_policy"] != "controlled_shared"
        or not _valid_sha256(plan["resource_baseline_sha256"])
        or local_inputs["resource_baseline_sha256"]
        != plan["resource_baseline_sha256"]
    ):
        raise ValueError(f"{name} execution plan resource binding is invalid")
    resource_baseline = None
    if controlled_shared:
        try:
            baseline_bytes = read_regular_file_once(
                Path(local_inputs["resource_baseline"]),
                f"{name} resource baseline",
            )
            if (
                hashlib.sha256(baseline_bytes).hexdigest()
                != plan["resource_baseline_sha256"]
            ):
                raise ValueError(
                    f"{name} resource baseline SHA mismatch"
                )
            resource_baseline = json.loads(
                baseline_bytes.decode("utf-8")
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            raise ValueError(
                f"{name} resource baseline is invalid"
            ) from error
        if not isinstance(resource_baseline, Mapping):
            raise ValueError(f"{name} resource baseline is invalid")

    authorization_fields = {
        "schema_version",
        "classification",
        "plan_sha256",
        "run_tag",
        "source_tree_sha256",
        "model_manifest_sha256",
        "workload_manifest_sha256",
        "gpu_indices",
        "ports",
        "nonce",
        "consumed",
    }
    if controlled_shared:
        authorization_fields |= {
            "resource_policy",
            "resource_baseline_sha256",
        }
    authorization = _require_exact_fields(
        authorization,
        authorization_fields,
        f"{name} consumed authorization",
    )
    if (
        authorization["schema_version"] != _ENGINE_AUTHORIZATION_SCHEMA
        or authorization["classification"] != "AUTHORIZED"
        or authorization["plan_sha256"] != canonical_json_sha256(plan)
        or authorization["run_tag"] != run_tag
        or authorization["source_tree_sha256"] != source_tree_sha256
        or authorization["model_manifest_sha256"]
        != MODEL_MANIFEST_SHA256
        or authorization["workload_manifest_sha256"] != workload_sha256
        or authorization["gpu_indices"] != gpu_indices
        or authorization["ports"] != ports
        or authorization["consumed"] is not True
    ):
        raise ValueError(f"{name} consumed authorization binding is invalid")
    _require_safe_nonce(
        authorization["nonce"],
        f"{name} consumed authorization",
    )
    if controlled_shared and (
        authorization["resource_policy"] != plan["resource_policy"]
        or authorization["resource_baseline_sha256"]
        != plan["resource_baseline_sha256"]
    ):
        raise ValueError(
            f"{name} consumed authorization resource binding is invalid"
        )

    receipt = _require_exact_fields(
        receipt,
        {
            "schema_version",
            "plan_sha256",
            "authorization_sha256",
            "authorization_nonce",
            "run_tag",
            "steps",
            "classification",
        },
        f"{name} execution receipt",
    )
    expected_receipt_schema = (
        _CACHED_RECEIPT_SCHEMA if is_cached else _ENGINE_RECEIPT_SCHEMA
    )
    if (
        receipt["schema_version"] != expected_receipt_schema
        or receipt["classification"] != "PASS"
        or receipt["plan_sha256"] != canonical_json_sha256(plan)
        or receipt["authorization_sha256"]
        != canonical_json_sha256(authorization)
        or receipt["authorization_nonce"] != authorization["nonce"]
        or receipt["run_tag"] != run_tag
    ):
        raise ValueError(
            f"{name} execution receipt authorization binding is invalid"
        )
    steps = receipt["steps"]
    if (
        not isinstance(steps, list)
        or [step.get("name") for step in steps]
        != list(_WORKLOAD_COMMAND_ORDER)
    ):
        raise ValueError(f"{name} execution receipt step order is invalid")
    by_name = {}
    for step in steps:
        command_name = step.get("name")
        fields = {
            "name",
            "command_sha256",
            "returncode",
            "stdout",
            "stderr",
        }
        if command_name == "package_download":
            fields |= {"output_sha256", "output_size"}
        step = _require_exact_fields(
            step,
            fields,
            f"{name} execution receipt command",
        )
        if (
            step["command_sha256"]
            != canonical_json_sha256(plan["commands"][command_name])
            or isinstance(step["returncode"], bool)
            or not isinstance(step["returncode"], int)
            or step["returncode"] != 0
            or not isinstance(step["stdout"], str)
            or not isinstance(step["stderr"], str)
            or len(step["stdout"].encode("utf-8")) > 64 * 1024
            or len(step["stderr"].encode("utf-8")) > 64 * 1024
        ):
            raise ValueError(
                f"{name} execution receipt command returncode is invalid"
            )
        if command_name == "package_download" and (
            not _valid_sha256(step["output_sha256"])
            or isinstance(step["output_size"], bool)
            or not isinstance(step["output_size"], int)
            or step["output_size"] <= 0
        ):
            raise ValueError(
                f"{name} execution receipt package output is invalid"
            )
        by_name[command_name] = step
    preflight = _validate_resource_receipt(
        by_name["resource_guard"]["stdout"],
        gpu_indices,
        f"{name} execution receipt resource guard",
        resource_policy=(
            plan["resource_policy"]
            if controlled_shared
            else "strict_exclusive"
        ),
        baseline=resource_baseline,
        baseline_sha256=(
            plan["resource_baseline_sha256"]
            if controlled_shared
            else None
        ),
    )
    guarded_lines = by_name["guarded_authority"]["stdout"].splitlines()
    marker = "QWEN35_FINAL_RESOURCE_JSON="
    final_lines = [
        line[len(marker):]
        for line in guarded_lines
        if line.startswith(marker)
    ]
    if len(final_lines) != 1:
        raise ValueError(
            f"{name} execution receipt final resource is invalid"
        )
    final_resource = _validate_resource_receipt(
        final_lines[0],
        gpu_indices,
        f"{name} execution receipt final resource",
        resource_policy=(
            plan["resource_policy"]
            if controlled_shared
            else "strict_exclusive"
        ),
        baseline=resource_baseline,
        baseline_sha256=(
            plan["resource_baseline_sha256"]
            if controlled_shared
            else None
        ),
    )
    if final_resource != preflight:
        raise ValueError(
            f"{name} execution receipt GPU identity drifted"
        )
    artifact_payload = None
    for line in reversed(guarded_lines):
        if not line or line.startswith(marker):
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, Mapping):
            artifact_payload = candidate
            break
    if artifact_payload is None:
        raise ValueError(
            f"{name} execution receipt artifact result is invalid"
        )
    verifier_payload = _parse_prerequisite_json_log(
        by_name["local_verify"]["stdout"],
        f"{name} execution receipt verification",
    )
    common_identity = {
        "source_tree_sha256": source_tree_sha256,
        "model_manifest_sha256": MODEL_MANIFEST_SHA256,
        "workload_manifest_sha256": workload_sha256,
    }
    if is_cached:
        expected_pass = {
            "classification": "PASS",
            "schema_version": CACHED_CONTINUATION_SCHEMA_VERSION,
            **common_identity,
            "checks": verification["checks"],
        }
        if artifact_payload != expected_pass:
            raise ValueError(
                f"{name} execution receipt artifact binding is invalid"
            )
        if verifier_payload != expected_pass:
            raise ValueError(
                f"{name} execution receipt verification binding is invalid"
            )
    else:
        expected_verifier = {
            "classification": "PASS",
            "model_manifest_sha256": MODEL_MANIFEST_SHA256,
            "source_tree_sha256": source_tree_sha256,
            "workload_manifest_sha256": workload_sha256,
            "reference_classification": artifact["classification"],
            "engine_classification": artifact["classification"],
        }
        expected_artifact = {
            **expected_verifier,
            "inventory": [
                "reference_authority",
                "reference_independent_verification.json",
                "engine_authority",
                "authority_summary.json",
            ],
        }
        if artifact_payload != expected_artifact:
            raise ValueError(
                f"{name} execution receipt artifact binding is invalid"
            )
        if verifier_payload != expected_verifier:
            raise ValueError(
                f"{name} execution receipt verification binding is invalid"
            )


def validate_prerequisite_execution_documents(
    name: object,
    plan: object,
    authorization: object,
    receipt: object,
    *,
    run_tag: object,
    source_tree_sha256: object,
    artifact: object,
    verification: object,
    evidence_root: Path,
) -> None:
    if (
        not isinstance(artifact, Mapping)
        or not isinstance(verification, Mapping)
    ):
        raise ValueError(f"{name} prerequisite authority is invalid")
    if name == "tp4_root_logit":
        _validate_root_prerequisite_documents(
            plan,
            authorization,
            receipt,
            run_tag=run_tag,
            source_tree_sha256=source_tree_sha256,
            verification=verification,
            evidence_root=evidence_root,
        )
    elif name in {"cached_continuation", "engine_correctness"}:
        _validate_workload_prerequisite_documents(
            name,
            plan,
            authorization,
            receipt,
            run_tag=run_tag,
            source_tree_sha256=source_tree_sha256,
            artifact=artifact,
            verification=verification,
        )
    else:
        raise ValueError("prerequisite authority family is invalid")


def validate_prerequisites(
    path: object,
    *,
    file_bytes: bytes | None = None,
) -> PrerequisiteStatus:
    prerequisite_path = Path(path)
    try:
        if file_bytes is None:
            file_bytes = read_regular_file_once(
                prerequisite_path,
                "correctness prerequisite",
            )
        payload = json.loads(file_bytes.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return _blocked(f"correctness prerequisite file is invalid: {error}")
    if not isinstance(payload, Mapping):
        return _blocked("correctness prerequisite payload is invalid")
    expected_top_level = {
        "schema_version",
        "model_manifest_sha256",
        *PREREQUISITE_NAMES,
    }
    if set(payload) != expected_top_level:
        return _blocked("correctness prerequisite schema is invalid")
    reasons = []
    if payload.get("schema_version") != PREREQUISITE_SCHEMA_VERSION:
        reasons.append("correctness prerequisite schema version mismatch")
    if payload.get("model_manifest_sha256") != MODEL_MANIFEST_SHA256:
        reasons.append("correctness prerequisite model manifest mismatch")
    root = prerequisite_path.parent
    for name in PREREQUISITE_NAMES:
        row = payload.get(name)
        if not isinstance(row, Mapping) or set(row) != set(
            PREREQUISITE_ROW_FIELDS
        ):
            reasons.append(f"{name} prerequisite row is invalid")
            continue
        if row.get("classification") != "PASS":
            reasons.append(f"{name} classification is not PASS")
        source_tree = row.get("source_tree_sha256")
        if not _valid_sha256(source_tree):
            reasons.append(f"{name} source tree SHA is invalid")
        if (
            name == "tp4_root_logit"
            and source_tree != TP4_ROOT_SOURCE_TREE_SHA256
        ):
            reasons.append("root-logit source tree mismatch")
        documents = {}
        for path_field, sha_field, label in (
            ("artifact_path", "artifact_sha256", f"{name} artifact"),
            (
                "independent_verification_path",
                "independent_verification_sha256",
                f"{name} independent verification",
            ),
            (
                "provenance_path",
                "provenance_sha256",
                f"{name} provenance",
            ),
        ):
            expected_sha = row.get(sha_field)
            if not _valid_sha256(expected_sha):
                reasons.append(f"{label} SHA is invalid")
                continue
            try:
                document_path = _safe_relative_file(
                    root,
                    row.get(path_field),
                    label,
                )
            except ValueError as error:
                reasons.append(str(error))
                continue
            try:
                document_bytes, document = load_json_file_once(
                    document_path,
                    label,
                )
            except ValueError:
                reasons.append(f"{label} payload is invalid")
                continue
            if hashlib.sha256(document_bytes).hexdigest() != expected_sha:
                reasons.append(f"{label} SHA mismatch")
                continue
            documents[path_field] = document
        if set(documents) != {
            "artifact_path",
            "independent_verification_path",
            "provenance_path",
        }:
            continue
        try:
            validate_authority_documents(
                name,
                documents["artifact_path"],
                documents["independent_verification_path"],
                source_tree,
            )
            provenance = documents["provenance_path"]
            validate_prerequisite_provenance(
                name,
                provenance,
                run_tag=row.get("run_tag"),
                source_tree_sha256=source_tree,
            )
            provenance_root = (
                prerequisite_path.parent / row["provenance_path"]
            ).parent
            execution_documents = {}
            for path_field, sha_field, label in (
                ("plan_path", "plan_sha256", "execution plan"),
                (
                    "authorization_path",
                    "authorization_sha256",
                    "consumed authorization",
                ),
                (
                    "receipt_path",
                    "receipt_sha256",
                    "execution receipt",
                ),
            ):
                evidence = _safe_relative_file(
                    provenance_root,
                    provenance[path_field],
                    f"{name} {label}",
                )
                try:
                    evidence_bytes, document = load_json_file_once(
                        evidence,
                        f"{name} {label}",
                    )
                except ValueError as error:
                    raise ValueError(f"{name} {label} JSON is invalid") from error
                if hashlib.sha256(evidence_bytes).hexdigest() != (
                    provenance[sha_field]
                ):
                    raise ValueError(f"{name} {label} SHA mismatch")
                if not isinstance(document, Mapping):
                    raise ValueError(
                        f"{name} {label} JSON must be an object"
                    )
                execution_documents[path_field] = document
            validate_prerequisite_execution_documents(
                name,
                execution_documents["plan_path"],
                execution_documents["authorization_path"],
                execution_documents["receipt_path"],
                run_tag=row.get("run_tag"),
                source_tree_sha256=source_tree,
                artifact=documents["artifact_path"],
                verification=documents[
                    "independent_verification_path"
                ],
                evidence_root=provenance_root,
            )
        except ValueError as error:
            reasons.append(str(error))
    if reasons:
        return _blocked(*reasons)
    return PrerequisiteStatus(
        classification="PASS",
        authorized=True,
        reasons=(),
    )


def _validate_closed_mapping(
    value: object,
    fields: tuple[str, ...],
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} schema requires a mapping")
    actual = set(value)
    expected = set(fields)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{label} schema fields invalid: "
            f"missing={missing}, unknown={unknown}"
        )


def _require_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is invalid")
    return value


def _require_safe_relative_path(value: object, label: str) -> str:
    path = Path(_require_nonempty_string(value, label))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} is unsafe")
    return path.as_posix()


def _require_sha256(value: object, label: str) -> str:
    if not _valid_sha256(value):
        raise ValueError(f"{label} is invalid")
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if not _nonnegative_integer(value):
        raise ValueError(f"{label} is invalid")
    return value


def _require_positive_int(value: object, label: str) -> int:
    result = _require_nonnegative_int(value, label)
    if result == 0:
        raise ValueError(f"{label} is invalid")
    return result


def _tensor_nbytes(
    dtype: object,
    shape: object,
    label: str,
) -> int:
    dtype_sizes = {
        "bfloat16": 2,
        "float32": 4,
        "int8": 1,
    }
    if dtype not in dtype_sizes:
        raise ValueError(f"{label} dtype is invalid")
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            type(dimension) is not int or dimension <= 0
            for dimension in shape
        )
    ):
        raise ValueError(f"{label} shape is invalid")
    return dtype_sizes[dtype] * math.prod(shape)


def validate_tensor_storage_evidence(evidence: object) -> None:
    _validate_closed_mapping(
        evidence,
        TENSOR_STORAGE_EVIDENCE_FIELDS,
        "tensor storage evidence",
    )
    if (
        evidence["schema_version"]
        != TENSOR_STORAGE_EVIDENCE_SCHEMA_VERSION
    ):
        raise ValueError("tensor storage evidence schema version is invalid")
    _require_nonempty_string(evidence["case_id"], "case id")
    profile = evidence["profile"]
    if profile == P1_REFERENCE_PROFILE:
        expected_identity = (
            P1_REFERENCE_PROFILE,
            P1_REPRESENTATION_VERSION,
            None,
        )
    elif profile == P2_PROFILE:
        expected_identity = (
            P2_REPRESENTATION,
            P2_REPRESENTATION_VERSION,
            P2_CODEC_ID,
        )
    else:
        raise ValueError("tensor storage evidence profile is invalid")
    if (
        evidence["representation"],
        evidence["representation_version"],
        evidence["codec"],
    ) != expected_identity:
        raise ValueError("tensor storage evidence representation is invalid")
    rank = _require_nonnegative_int(evidence["rank"], "rank")
    if evidence["world_size"] != WORLD_SIZE or rank >= WORLD_SIZE:
        raise ValueError("tensor storage evidence world size is invalid")

    storages = evidence["storages"]
    snapshots = evidence["snapshots"]
    observations = evidence["observations"]
    if (
        not isinstance(storages, list)
        or not isinstance(snapshots, list)
        or not isinstance(observations, list)
        or not snapshots
        or not observations
    ):
        raise ValueError("tensor storage evidence collections are invalid")

    storage_by_id = {}
    for storage in storages:
        _validate_closed_mapping(
            storage,
            TENSOR_STORAGE_STORAGE_FIELDS,
            "tensor storage",
        )
        storage_id = _require_nonempty_string(
            storage["storage_id"],
            "storage id",
        )
        if storage_id in storage_by_id:
            raise ValueError("tensor storage id is duplicated")
        kind = storage["kind"]
        if kind not in {
            "resident",
            "encode_workspace",
            "decode_workspace",
        }:
            raise ValueError("tensor storage kind is invalid")
        _require_positive_int(storage["storage_nbytes"], "storage nbytes")
        if kind == "resident":
            _require_sha256(storage["content_sha256"], "content sha256")
        elif storage["content_sha256"] is not None:
            raise ValueError("workspace storage content sha256 is invalid")
        storage_by_id[storage_id] = storage

    snapshot_by_id = {}
    covered_ranges = {
        storage_id: []
        for storage_id, storage in storage_by_id.items()
        if storage["kind"] == "resident"
    }
    role_order = {
        "convolution": 0,
        "recurrent_values": 1,
        "recurrent_scales": 2,
    }
    for snapshot in snapshots:
        _validate_closed_mapping(
            snapshot,
            TENSOR_STORAGE_SNAPSHOT_FIELDS,
            "tensor storage snapshot",
        )
        snapshot_id = _require_nonempty_string(
            snapshot["snapshot_id"],
            "snapshot id",
        )
        if snapshot_id in snapshot_by_id:
            raise ValueError("tensor storage snapshot id is duplicated")
        references = snapshot["tensor_references"]
        if not isinstance(references, list):
            raise ValueError("tensor reference collection is invalid")
        expected_reference_order = []
        reference_ids = set()
        roles_by_layer = {layer_index: [] for layer_index in range(18)}
        references_by_layer = {
            layer_index: {}
            for layer_index in range(18)
        }
        for reference in references:
            _validate_closed_mapping(
                reference,
                TENSOR_STORAGE_REFERENCE_FIELDS,
                "tensor reference",
            )
            reference_id = _require_nonempty_string(
                reference["reference_id"],
                "reference id",
            )
            if reference_id in reference_ids:
                raise ValueError("tensor reference id is duplicated")
            reference_ids.add(reference_id)
            layer_index = _require_nonnegative_int(
                reference["layer_index"],
                "layer index",
            )
            if layer_index >= 18:
                raise ValueError("tensor reference layer is invalid")
            role = reference["semantic_role"]
            if role not in role_order:
                raise ValueError("tensor reference semantic role is invalid")
            expected_reference_order.append(
                (layer_index, role_order[role])
            )
            roles_by_layer[layer_index].append(role)
            references_by_layer[layer_index][role] = reference
            storage_id = reference["storage_id"]
            storage = storage_by_id.get(storage_id)
            if storage is None:
                raise ValueError("tensor reference storage is invalid")
            if storage["kind"] != "resident":
                raise ValueError(
                    "tensor reference cannot reference workspace storage"
                )
            offset = _require_nonnegative_int(
                reference["storage_offset_bytes"],
                "storage offset bytes",
            )
            length = _require_positive_int(
                reference["storage_length_bytes"],
                "storage length bytes",
            )
            if offset + length > storage["storage_nbytes"]:
                raise ValueError("tensor reference storage range is invalid")
            if (
                _tensor_nbytes(
                    reference["resident_dtype"],
                    reference["resident_shape"],
                    "resident tensor",
                )
                != length
            ):
                    raise ValueError(
                        "resident tensor byte range cover is invalid"
                    )
            covered_ranges[storage_id].append((offset, offset + length))

            if role == "convolution":
                valid = (
                    reference["logical_dtype"] == "bfloat16"
                    and reference["resident_dtype"] == "bfloat16"
                    and reference["logical_shape"]
                    == reference["resident_shape"]
                )
            elif role == "recurrent_values":
                valid = (
                    reference["logical_dtype"] == "float32"
                    and reference["resident_dtype"]
                    == (
                        "float32"
                        if profile == P1_REFERENCE_PROFILE
                        else "int8"
                    )
                    and reference["logical_shape"]
                    == reference["resident_shape"]
                )
            else:
                valid = (
                    reference["logical_dtype"] is None
                    and reference["logical_shape"] is None
                    and reference["resident_dtype"] == "float32"
                )
            if not valid:
                raise ValueError("tensor reference dtype or shape is invalid")
        if expected_reference_order != sorted(expected_reference_order):
            raise ValueError("tensor reference order is invalid")
        expected_roles = (
            ["convolution", "recurrent_values"]
            if profile == P1_REFERENCE_PROFILE
            else ["convolution", "recurrent_values", "recurrent_scales"]
        )
        if any(roles != expected_roles for roles in roles_by_layer.values()):
            raise ValueError("tensor reference layer coverage is invalid")
        metadata = snapshot["codec_metadata"]
        if profile == P1_REFERENCE_PROFILE:
            if metadata is not None:
                raise ValueError("exact restore codec metadata is invalid")
        else:
            _validate_closed_mapping(
                metadata,
                (
                    "codec",
                    "layers",
                    "representation",
                    "version",
                ),
                "codec metadata",
            )
            if (
                metadata["codec"] != P2_CODEC_ID
                or metadata["representation"] != P2_REPRESENTATION
                or metadata["version"] != P2_REPRESENTATION_VERSION
                or not isinstance(metadata["layers"], list)
                or len(metadata["layers"]) != 18
            ):
                raise ValueError("codec metadata is invalid")
            for layer_index, layer in enumerate(metadata["layers"]):
                _validate_closed_mapping(
                    layer,
                    (
                        "codec",
                        "layer_index",
                        "source_dtype",
                        "source_shape",
                    ),
                    "codec metadata layer",
                )
                source_shape = layer["source_shape"]
                recurrent_values = references_by_layer[layer_index][
                    "recurrent_values"
                ]
                recurrent_scales = references_by_layer[layer_index][
                    "recurrent_scales"
                ]
                if (
                    layer["codec"] != P2_CODEC_ID
                    or layer["layer_index"] != layer_index
                    or layer["source_dtype"] != "torch.float32"
                ):
                    raise ValueError("codec metadata layer is invalid")
                if (
                    recurrent_values["logical_shape"] != source_shape
                    or recurrent_values["resident_shape"] != source_shape
                    or recurrent_scales["resident_shape"]
                    != source_shape[:-1]
                ):
                    raise ValueError(
                        "codec metadata layer shape is invalid"
                    )
        snapshot_by_id[snapshot_id] = snapshot

    for storage_id, ranges in covered_ranges.items():
        if not ranges:
            raise ValueError("resident storage interval cover is invalid")
        merged_end = 0
        for start, end in sorted(ranges):
            if start > merged_end:
                raise ValueError("resident storage interval cover has a gap")
            merged_end = max(merged_end, end)
        if merged_end != storage_by_id[storage_id]["storage_nbytes"]:
            raise ValueError("resident storage interval cover is incomplete")

    for expected_ordinal, observation in enumerate(observations):
        _validate_closed_mapping(
            observation,
            TENSOR_STORAGE_OBSERVATION_FIELDS,
            "tensor storage observation",
        )
        ordinal = _require_nonnegative_int(
            observation["ordinal"],
            "observation ordinal",
        )
        if ordinal != expected_ordinal:
            raise ValueError("observation ordinal is not contiguous")
        if observation["event"] not in {
            "baseline",
            "encode",
            "publish",
            "decode",
            "steady_state",
            "final",
        }:
            raise ValueError("observation event is invalid")
        active_snapshot_ids = observation["active_snapshot_ids"]
        live_workspace_ids = observation["live_workspace_storage_ids"]
        if (
            not isinstance(active_snapshot_ids, list)
            or active_snapshot_ids != sorted(set(active_snapshot_ids))
            or any(
                snapshot_id not in snapshot_by_id
                for snapshot_id in active_snapshot_ids
            )
        ):
            raise ValueError("observation snapshot binding is invalid")
        if (
            not isinstance(live_workspace_ids, list)
            or live_workspace_ids != sorted(set(live_workspace_ids))
        ):
            raise ValueError("observation workspace binding is invalid")
        live_workspace_bytes = {
            "encode_workspace": 0,
            "decode_workspace": 0,
        }
        for storage_id in live_workspace_ids:
            storage = storage_by_id.get(storage_id)
            if storage is None or storage["kind"] == "resident":
                raise ValueError(
                    "observation workspace storage binding is invalid"
                )
            live_workspace_bytes[storage["kind"]] += storage[
                "storage_nbytes"
            ]
        for field in TENSOR_STORAGE_OBSERVATION_FIELDS[4:]:
            _require_nonnegative_int(
                observation[field],
                field.replace("_", " "),
            )
        if (
            observation["cuda_allocated_bytes"]
            > observation["cuda_reserved_bytes"]
        ):
            raise ValueError("observation cuda allocation is invalid")
        if (
            observation["encode_workspace_reserved_bytes"]
            < live_workspace_bytes["encode_workspace"]
        ):
            raise ValueError(
                "observation encode workspace reserved bytes are invalid"
            )
        if (
            observation["decode_workspace_reserved_bytes"]
            < live_workspace_bytes["decode_workspace"]
        ):
            raise ValueError(
                "observation decode workspace reserved bytes are invalid"
            )
    if observations[-1]["event"] != "final":
        raise ValueError("final observation must be last")


def recompute_tensor_storage_accounting(evidence: object) -> dict:
    validate_tensor_storage_evidence(evidence)
    storage_by_id = {
        storage["storage_id"]: storage
        for storage in evidence["storages"]
    }
    snapshot_by_id = {
        snapshot["snapshot_id"]: snapshot
        for snapshot in evidence["snapshots"]
    }
    observations = []
    for observation in evidence["observations"]:
        logical_bytes = 0
        metadata_bytes = 0
        resident_storage_ids = set()
        for snapshot_id in observation["active_snapshot_ids"]:
            snapshot = snapshot_by_id[snapshot_id]
            metadata = snapshot["codec_metadata"]
            if metadata is not None:
                metadata_bytes += len(canonical_json_bytes(metadata))
            for reference in snapshot["tensor_references"]:
                if reference["logical_dtype"] is not None:
                    logical_bytes += _tensor_nbytes(
                        reference["logical_dtype"],
                        reference["logical_shape"],
                        "logical tensor",
                    )
                resident_storage_ids.add(reference["storage_id"])
        unique_physical_bytes = sum(
            storage_by_id[storage_id]["storage_nbytes"]
            for storage_id in resident_storage_ids
        )
        encode_allocated = 0
        decode_allocated = 0
        for storage_id in observation["live_workspace_storage_ids"]:
            storage = storage_by_id[storage_id]
            if storage["kind"] == "encode_workspace":
                encode_allocated += storage["storage_nbytes"]
            else:
                decode_allocated += storage["storage_nbytes"]
        observations.append({
            "hybrid_cache_entries": len(
                observation["active_snapshot_ids"]
            ),
            "hybrid_cache_unique_physical_bytes": unique_physical_bytes,
            "hybrid_cache_logical_referenced_bytes": logical_bytes,
            "hybrid_cache_metadata_bytes": metadata_bytes,
            "encode_workspace_allocated_bytes": encode_allocated,
            "encode_workspace_reserved_bytes": observation[
                "encode_workspace_reserved_bytes"
            ],
            "decode_workspace_allocated_bytes": decode_allocated,
            "decode_workspace_reserved_bytes": observation[
                "decode_workspace_reserved_bytes"
            ],
            "cuda_allocated_bytes": observation["cuda_allocated_bytes"],
            "cuda_reserved_bytes": observation["cuda_reserved_bytes"],
        })
    final = observations[-1]
    result = {
        "cuda_allocated_bytes": final["cuda_allocated_bytes"],
        "cuda_reserved_bytes": final["cuda_reserved_bytes"],
        "cuda_peak_allocated_bytes": max(
            row["cuda_allocated_bytes"] for row in observations
        ),
        "cuda_peak_reserved_bytes": max(
            row["cuda_reserved_bytes"] for row in observations
        ),
        "encode_workspace_peak_allocated_bytes": max(
            row["encode_workspace_allocated_bytes"]
            for row in observations
        ),
        "encode_workspace_peak_reserved_bytes": max(
            row["encode_workspace_reserved_bytes"]
            for row in observations
        ),
        "decode_workspace_peak_allocated_bytes": max(
            row["decode_workspace_allocated_bytes"]
            for row in observations
        ),
        "decode_workspace_peak_reserved_bytes": max(
            row["decode_workspace_reserved_bytes"]
            for row in observations
        ),
        "hybrid_cache_current_entries": final["hybrid_cache_entries"],
        "hybrid_cache_current_unique_physical_bytes": final[
            "hybrid_cache_unique_physical_bytes"
        ],
        "hybrid_cache_current_logical_referenced_bytes": final[
            "hybrid_cache_logical_referenced_bytes"
        ],
        "hybrid_cache_current_metadata_bytes": final[
            "hybrid_cache_metadata_bytes"
        ],
        "hybrid_cache_peak_entries": max(
            row["hybrid_cache_entries"] for row in observations
        ),
        "hybrid_cache_peak_unique_physical_bytes": max(
            row["hybrid_cache_unique_physical_bytes"]
            for row in observations
        ),
        "hybrid_cache_peak_logical_referenced_bytes": max(
            row["hybrid_cache_logical_referenced_bytes"]
            for row in observations
        ),
        "hybrid_cache_peak_metadata_bytes": max(
            row["hybrid_cache_metadata_bytes"]
            for row in observations
        ),
    }
    result["hybrid_cache_deduplicated_bytes"] = (
        max(
            0,
            result["hybrid_cache_current_logical_referenced_bytes"]
            - result["hybrid_cache_current_unique_physical_bytes"],
        )
    )
    return result


def _require_nonnegative_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} is invalid")
    return float(value)


def _validate_profile_identity(
    profile: object,
    representation: object,
    representation_version: object,
    codec: object,
) -> None:
    if profile not in PROFILES:
        raise ValueError("profile is invalid")
    expected = {
        "recompute": (None, None, None),
        "exact_restore": (
            "exact_restore",
            P1_REPRESENTATION_VERSION,
            None,
        ),
        P2_PROFILE: (
            P2_REPRESENTATION,
            P2_REPRESENTATION_VERSION,
            P2_CODEC_ID,
        ),
    }[profile]
    if representation != expected[0]:
        raise ValueError("representation is invalid")
    if representation_version != expected[1]:
        raise ValueError("representation version is invalid")
    if codec != expected[2]:
        raise ValueError("codec is invalid")


def validate_case_row(row: object) -> None:
    _validate_closed_mapping(row, CASE_ROW_FIELDS, "case row")
    _validate_profile_identity(
        row["profile"],
        row["representation"],
        row["representation_version"],
        row["codec"],
    )
    for field in ("row_id", "case_id", "request_id"):
        _require_nonempty_string(row[field], field.replace("_", " "))
    if row["workload"] not in WORKLOADS:
        raise ValueError("workload is invalid")
    if row["phase"] not in {"warmup", "correctness", "measured"}:
        raise ValueError("phase is invalid")
    _require_nonnegative_int(row["repetition"], "repetition")
    _require_nonnegative_number(
        row["sampling_temperature"],
        "sampling temperature",
    )
    if row["sampling_temperature"] != SAMPLING_TEMPERATURE:
        raise ValueError("sampling temperature is invalid")
    _require_positive_int(row["sampling_max_tokens"], "sampling max tokens")
    if (
        row["sampling_max_tokens"]
        != WORKLOAD_SPECS[row["workload"]]["generated_tokens"]
    ):
        raise ValueError("sampling max tokens is invalid")
    if type(row["sampling_ignore_eos"]) is not bool:
        raise ValueError("sampling ignore eos is invalid")
    if row["sampling_ignore_eos"] is not SAMPLING_IGNORE_EOS:
        raise ValueError("sampling ignore eos is invalid")
    _require_nonnegative_int(row["sampling_seed"], "sampling seed")
    if row["sampling_seed"] != workload_sampling_seed(row["workload"]):
        raise ValueError("sampling seed is invalid")
    _require_positive_int(row["concurrency"], "concurrency")
    if row["concurrency"] != effective_concurrency(
        row["workload"],
        row["phase"],
    ):
        raise ValueError("concurrency is invalid")
    if _require_positive_int(row["tp_world_size"], "tp world size") != WORLD_SIZE:
        raise ValueError("tp world size is invalid")
    if row["gpu_indices"] != list(REQUIRED_GPU_INDICES):
        raise ValueError("gpu indices is invalid")
    _require_positive_int(row["kv_capacity_bytes"], "kv capacity bytes")
    if (
        _require_positive_int(
            row["hybrid_prefix_max_entries"],
            "hybrid prefix max entries",
        )
        != HYBRID_PREFIX_MAX_ENTRIES
    ):
        raise ValueError("hybrid prefix max entries is invalid")
    if (
        _require_positive_int(
            row["hybrid_prefix_max_bytes"],
            "hybrid prefix max bytes",
        )
        != HYBRID_PREFIX_MAX_BYTES
    ):
        raise ValueError("hybrid prefix max bytes is invalid")
    if row["dirty_tree_policy"] not in DIRTY_TREE_POLICIES:
        raise ValueError("dirty tree policy is invalid")
    for field in (
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
        "prompt_token_ids_sha256",
        "output_token_ids_sha256",
        "final_logits_sha256",
    ):
        _require_sha256(row[field], field.replace("_", " "))
    if row["model_manifest_sha256"] != MODEL_MANIFEST_SHA256:
        raise ValueError("model manifest sha256 is invalid")
    for field in (
        "prompt_token_ids_path",
        "output_token_ids_path",
        "final_logits_path",
    ):
        _require_safe_relative_path(row[field], field.replace("_", " "))
    for field in (
        "prompt_tokens",
        "reused_kv_tokens",
        "executed_prefill_tokens",
        "generated_tokens",
    ):
        _require_nonnegative_int(row[field], field.replace("_", " "))
    if type(row["restored_hybrid_state"]) is not bool:
        raise ValueError("restored hybrid state is invalid")
    expected_generated = WORKLOAD_SPECS[row["workload"]][
        "generated_tokens"
    ]
    if row["generated_tokens"] != expected_generated:
        raise ValueError("generated tokens is invalid")
    if (
        row["prompt_tokens"]
        != row["reused_kv_tokens"] + row["executed_prefill_tokens"]
    ):
        raise ValueError("token accounting is invalid")
    for field in ("ttft_ns", "e2e_ns", "decode_step_ns"):
        _require_nonnegative_number(row[field], field.replace("_", " "))
    if (
        row["e2e_ns"] < row["ttft_ns"]
        or row["decode_step_ns"] > row["e2e_ns"]
    ):
        raise ValueError("timing is invalid")
    shape = row["final_logits_shape"]
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in shape
        )
    ):
        raise ValueError("final logits shape is invalid")
    if row["final_logits_dtype"] != "float32":
        raise ValueError("final logits dtype is invalid")


def validate_case_rows(rows: object) -> None:
    if not isinstance(rows, list):
        raise ValueError("canonical case row collection is invalid")
    expected_rows = []
    for case in build_case_matrix():
        spec = WORKLOAD_SPECS[case.workload]
        prompt_tokens = (
            spec["shared_prefix_tokens"] + spec["suffix_tokens"]
        )
        restore_hit = (
            case.profile != "recompute"
            and case.workload != "w4_miss_invalidation"
        )
        for request_index in range(spec["continuations"]):
            expected_rows.append({
                "row_id": (
                    f"{case.case_id}__request-{request_index}"
                ),
                "case_id": case.case_id,
                "profile": case.profile,
                "workload": case.workload,
                "phase": case.phase,
                "repetition": case.repetition,
                "request_id": f"request-{request_index}",
                "sampling_max_tokens": spec["generated_tokens"],
                "sampling_seed": workload_sampling_seed(case.workload),
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
    if len(rows) != len(expected_rows):
        raise ValueError("canonical case row collection is invalid")
    semantic_fields = tuple(expected_rows[0])
    for row, expected in zip(rows, expected_rows):
        try:
            validate_case_row(row)
        except ValueError as error:
            raise ValueError("canonical case row is invalid") from error
        if any(row[field] != expected[field] for field in semantic_fields):
            raise ValueError("canonical case row is invalid")


def validate_process_row(row: object) -> None:
    _validate_closed_mapping(row, PROCESS_ROW_FIELDS, "process row")
    _validate_profile_identity(
        row["profile"],
        row["representation"],
        row["representation_version"],
        row["codec"],
    )
    _require_nonempty_string(row["case_id"], "case id")
    if row["workload"] not in WORKLOADS:
        raise ValueError("workload is invalid")
    if row["phase"] not in {"warmup", "correctness", "measured"}:
        raise ValueError("phase is invalid")
    _require_nonnegative_int(row["repetition"], "repetition")
    rank = _require_nonnegative_int(row["rank"], "rank")
    world_size = _require_positive_int(row["world_size"], "world size")
    if world_size != WORLD_SIZE:
        raise ValueError("world size is invalid")
    if rank >= world_size:
        raise ValueError("rank is invalid")
    _require_positive_int(row["pid"], "pid")
    for field in (
        "hostname",
        "gpu_uuid",
        "cuda_visible_device",
        "master_addr",
        "nonce",
        "run_tag",
    ):
        _require_nonempty_string(row[field], field.replace("_", " "))
    for field in ("master_port", "tinyvllm_dist_port"):
        port = _require_positive_int(row[field], field.replace("_", " "))
        if port > 65535:
            raise ValueError(f"{field.replace('_', ' ')} is invalid")
    _require_safe_relative_path(row["artifact_path"], "artifact path")
    _require_nonnegative_number(
        row["sampling_temperature"],
        "sampling temperature",
    )
    if row["sampling_temperature"] != SAMPLING_TEMPERATURE:
        raise ValueError("sampling temperature is invalid")
    _require_positive_int(row["sampling_max_tokens"], "sampling max tokens")
    if (
        row["sampling_max_tokens"]
        != WORKLOAD_SPECS[row["workload"]]["generated_tokens"]
    ):
        raise ValueError("sampling max tokens is invalid")
    if type(row["sampling_ignore_eos"]) is not bool:
        raise ValueError("sampling ignore eos is invalid")
    if row["sampling_ignore_eos"] is not SAMPLING_IGNORE_EOS:
        raise ValueError("sampling ignore eos is invalid")
    _require_nonnegative_int(row["sampling_seed"], "sampling seed")
    if row["sampling_seed"] != workload_sampling_seed(row["workload"]):
        raise ValueError("sampling seed is invalid")
    _require_positive_int(row["concurrency"], "concurrency")
    if row["concurrency"] != effective_concurrency(
        row["workload"],
        row["phase"],
    ):
        raise ValueError("concurrency is invalid")
    if row["gpu_indices"] != list(REQUIRED_GPU_INDICES):
        raise ValueError("gpu indices is invalid")
    if (
        _require_positive_int(
            row["hybrid_prefix_max_entries"],
            "hybrid prefix max entries",
        )
        != HYBRID_PREFIX_MAX_ENTRIES
    ):
        raise ValueError("hybrid prefix max entries is invalid")
    if (
        _require_positive_int(
            row["hybrid_prefix_max_bytes"],
            "hybrid prefix max bytes",
        )
        != HYBRID_PREFIX_MAX_BYTES
    ):
        raise ValueError("hybrid prefix max bytes is invalid")
    if row["dirty_tree_policy"] not in DIRTY_TREE_POLICIES:
        raise ValueError("dirty tree policy is invalid")
    for field in (
        "source_tree_sha256",
        "gate1_audit_sha256",
        "execution_plan_sha256",
        "source_bundle_sha256",
        "source_package_sha256",
        "producer_source_sha256",
        "producer_version_sha256",
        "verifier_source_sha256",
        "verifier_version_sha256",
    ):
        _require_sha256(row[field], field.replace("_", " "))
    positive_fields = {
        "kv_capacity_bytes",
        "same_budget_entry_capacity",
    }
    numeric_fields = PROCESS_ROW_FIELDS[
        PROCESS_ROW_FIELDS.index("initialization_ns"):
    ]
    for field in numeric_fields:
        if field in positive_fields:
            _require_positive_int(row[field], field.replace("_", " "))
        else:
            _require_nonnegative_int(row[field], field.replace("_", " "))
    if (
        row["cuda_allocated_bytes"] > row["cuda_peak_allocated_bytes"]
        or row["cuda_reserved_bytes"] > row["cuda_peak_reserved_bytes"]
        or row["cuda_peak_allocated_bytes"]
        > row["cuda_peak_reserved_bytes"]
    ):
        raise ValueError("CUDA memory accounting is invalid")
    if (
        row["hybrid_cache_current_entries"]
        > row["hybrid_cache_peak_entries"]
        or row["hybrid_cache_current_unique_physical_bytes"]
        > row["hybrid_cache_peak_unique_physical_bytes"]
        or row["hybrid_cache_current_logical_referenced_bytes"]
        > row["hybrid_cache_peak_logical_referenced_bytes"]
        or row["hybrid_cache_current_metadata_bytes"]
        > row["hybrid_cache_peak_metadata_bytes"]
    ):
        raise ValueError("cache peak accounting is invalid")


def validate_process_rows(rows: object) -> None:
    expected_rows = []
    for case in build_case_matrix():
        spec = WORKLOAD_SPECS[case.workload]
        for rank, gpu_index in enumerate(REQUIRED_GPU_INDICES):
            expected_rows.append({
                "case_id": case.case_id,
                "profile": case.profile,
                "workload": case.workload,
                "phase": case.phase,
                "repetition": case.repetition,
                "rank": rank,
                "world_size": WORLD_SIZE,
                "cuda_visible_device": str(rank),
                "gpu_index": gpu_index,
                "sampling_temperature": SAMPLING_TEMPERATURE,
                "sampling_max_tokens": spec["generated_tokens"],
                "sampling_ignore_eos": SAMPLING_IGNORE_EOS,
                "sampling_seed": workload_sampling_seed(case.workload),
                "concurrency": case.concurrency,
            })
    if not isinstance(rows, list) or len(rows) != len(expected_rows):
        raise ValueError("canonical process row collection is invalid")
    for row, expected in zip(rows, expected_rows):
        try:
            validate_process_row(row)
        except ValueError as error:
            raise ValueError("canonical process row is invalid") from error
        if any(
            row[field] != expected[field]
            for field in expected
            if field != "gpu_index"
        ):
            raise ValueError("canonical process row is invalid")
        if row["gpu_indices"][row["rank"]] != expected["gpu_index"]:
            raise ValueError("canonical process row is invalid")


def validate_case_process_row_bindings(
    case_rows: object,
    process_rows: object,
) -> None:
    validate_case_rows(case_rows)
    validate_process_rows(process_rows)
    case_rows_by_id = {}
    for case_row in case_rows:
        case_rows_by_id.setdefault(case_row["case_id"], []).append(case_row)
    process_rows_by_id = {}
    for process_row in process_rows:
        process_rows_by_id.setdefault(
            process_row["case_id"],
            [],
        ).append(process_row)
    for case in build_case_matrix():
        for case_row in case_rows_by_id[case.case_id]:
            for process_row in process_rows_by_id[case.case_id]:
                for field in SHARED_CASE_PROCESS_FIELDS:
                    if case_row[field] != process_row[field]:
                        raise ValueError(
                            f"case/process binding mismatch: {field}"
                        )
                if case_row["tp_world_size"] != process_row["world_size"]:
                    raise ValueError(
                        "case/process binding mismatch: tp_world_size"
                    )


def validate_calibration_binding(binding: object) -> None:
    _validate_closed_mapping(
        binding,
        CALIBRATION_BINDING_FIELDS,
        "calibration binding",
    )
    exact_values = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "codec": P2_CODEC_ID,
        "representation": P2_REPRESENTATION,
        "representation_version": P2_REPRESENTATION_VERSION,
        "model_manifest_sha256": MODEL_MANIFEST_SHA256,
        "classification": "PASS",
    }
    for field, expected in exact_values.items():
        if binding[field] != expected:
            raise ValueError(f"{field.replace('_', ' ')} is invalid")
    for field in (
        "source_tree_sha256",
        "model_manifest_sha256",
        "workload_manifest_sha256",
        "artifact_sha256",
    ):
        _require_sha256(binding[field], field.replace("_", " "))
    _require_safe_relative_path(binding["artifact_path"], "artifact path")


def validate_p1_authority_binding(binding: object) -> None:
    _validate_closed_mapping(
        binding,
        P1_AUTHORITY_BINDING_FIELDS,
        "P1 authority binding",
    )
    exact_values = {
        "schema_version": P1_AUTHORITY_SCHEMA_VERSION,
        "profile": P1_REFERENCE_PROFILE,
        "representation": P1_REFERENCE_PROFILE,
        "model_manifest_sha256": MODEL_MANIFEST_SHA256,
        "classification": "GO",
    }
    for field, expected in exact_values.items():
        if binding[field] != expected:
            raise ValueError(f"{field.replace('_', ' ')} is invalid")
    for field in (
        "source_tree_sha256",
        "model_manifest_sha256",
        "workload_manifest_sha256",
        "artifact_sha256",
        "independent_verification_sha256",
    ):
        _require_sha256(binding[field], field.replace("_", " "))
    for field in ("artifact_path", "independent_verification_path"):
        _require_safe_relative_path(binding[field], field.replace("_", " "))


def validate_snapshot_inventory(inventory: object) -> None:
    _validate_closed_mapping(
        inventory,
        SNAPSHOT_INVENTORY_FIELDS,
        "snapshot inventory",
    )
    exact_values = {
        "schema_version": SNAPSHOT_INVENTORY_SCHEMA_VERSION,
        "profile": P2_PROFILE,
        "representation": P2_REPRESENTATION,
        "representation_version": P2_REPRESENTATION_VERSION,
        "codec": P2_CODEC_ID,
    }
    for field, expected in exact_values.items():
        if inventory[field] != expected:
            raise ValueError(f"{field.replace('_', ' ')} is invalid")
    _require_nonempty_string(inventory["case_id"], "case id")
    rank = _require_nonnegative_int(inventory["rank"], "rank")
    world_size = _require_positive_int(inventory["world_size"], "world size")
    if world_size != WORLD_SIZE:
        raise ValueError("world size is invalid")
    if rank >= world_size:
        raise ValueError("rank is invalid")
    for field in ("snapshot_path", "tensor_inventory_path"):
        _require_safe_relative_path(
            inventory[field],
            field.replace("_", " "),
        )
    for field in ("snapshot_sha256", "tensor_inventory_sha256"):
        _require_sha256(inventory[field], field.replace("_", " "))
    for field in SNAPSHOT_INVENTORY_FIELDS[
        SNAPSHOT_INVENTORY_FIELDS.index("full_fidelity_logical_bytes"):
    ]:
        _require_nonnegative_int(
            inventory[field],
            field.replace("_", " "),
        )


def validate_manifest_entry(entry: object) -> None:
    _validate_closed_mapping(entry, MANIFEST_ENTRY_FIELDS, "manifest entry")
    _require_safe_relative_path(entry["path"], "path")
    _require_sha256(entry["sha256"], "sha256")
    _require_nonnegative_int(entry["bytes"], "bytes")
    _require_nonempty_string(entry["producer"], "producer")
    if entry["trust_domain"] != "producer":
        raise ValueError("trust domain is invalid")


def validate_receipt_binding(binding: object) -> None:
    _validate_closed_mapping(
        binding,
        RECEIPT_BINDING_FIELDS,
        "receipt binding",
    )
    if binding["schema_version"] != RECEIPT_BINDING_SCHEMA_VERSION:
        raise ValueError("schema version is invalid")
    for field in ("run_tag", "nonce"):
        _require_nonempty_string(binding[field], field.replace("_", " "))
    _require_safe_relative_path(binding["artifact_path"], "artifact path")
    for field in RECEIPT_BINDING_FIELDS[
        RECEIPT_BINDING_FIELDS.index("gate1_audit_sha256"):
    ]:
        _require_sha256(binding[field], field.replace("_", " "))


def validate_source_manifest(manifest: object) -> None:
    _validate_closed_mapping(
        manifest,
        SOURCE_MANIFEST_FIELDS,
        "source manifest",
    )
    if manifest["schema_version"] != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("schema version is invalid")
    if manifest["dirty_tree_policy"] not in DIRTY_TREE_POLICIES:
        raise ValueError("dirty tree policy is invalid")
    if type(manifest["dirty_tree"]) is not bool:
        raise ValueError("dirty tree is invalid")
    if manifest["dirty_tree"]:
        raise ValueError("dirty tree is invalid")
    for field in SOURCE_MANIFEST_FIELDS[
        SOURCE_MANIFEST_FIELDS.index("source_tree_sha256"):
    ]:
        if field in {"dirty_tree_policy", "dirty_tree"}:
            continue
        _require_sha256(manifest[field], field.replace("_", " "))


def validate_matched_configuration(configuration: object) -> None:
    _validate_closed_mapping(
        configuration,
        MATCHED_CONFIGURATION_FIELDS,
        "matched configuration",
    )
    if (
        configuration["schema_version"]
        != MATCHED_CONFIGURATION_SCHEMA_VERSION
    ):
        raise ValueError("schema version is invalid")
    for field in (
        "model_manifest_sha256",
        "tokenizer_manifest_sha256",
        "workload_manifest_sha256",
    ):
        _require_sha256(configuration[field], field.replace("_", " "))
    if configuration["model_manifest_sha256"] != MODEL_MANIFEST_SHA256:
        raise ValueError("model manifest sha256 is invalid")
    _require_nonnegative_number(
        configuration["sampling_temperature"],
        "sampling temperature",
    )
    if configuration["sampling_temperature"] != SAMPLING_TEMPERATURE:
        raise ValueError("sampling temperature is invalid")
    _require_positive_int(
        configuration["sampling_max_tokens"],
        "sampling max tokens",
    )
    if type(configuration["sampling_ignore_eos"]) is not bool:
        raise ValueError("sampling ignore eos is invalid")
    if configuration["sampling_ignore_eos"] is not SAMPLING_IGNORE_EOS:
        raise ValueError("sampling ignore eos is invalid")
    _require_nonnegative_int(
        configuration["sampling_seed"],
        "sampling seed",
    )
    _require_positive_int(configuration["concurrency"], "concurrency")
    if (
        _require_positive_int(
            configuration["tp_world_size"],
            "tp world size",
        )
        != WORLD_SIZE
    ):
        raise ValueError("tp world size is invalid")
    if configuration["gpu_indices"] != list(REQUIRED_GPU_INDICES):
        raise ValueError("gpu indices is invalid")
    _require_positive_int(
        configuration["kv_capacity_bytes"],
        "kv capacity bytes",
    )
    if (
        _require_positive_int(
            configuration["hybrid_prefix_max_entries"],
            "hybrid prefix max entries",
        )
        != HYBRID_PREFIX_MAX_ENTRIES
    ):
        raise ValueError("hybrid prefix max entries is invalid")
    if (
        _require_positive_int(
            configuration["hybrid_prefix_max_bytes"],
            "hybrid prefix max bytes",
        )
        != HYBRID_PREFIX_MAX_BYTES
    ):
        raise ValueError("hybrid prefix max bytes is invalid")


def validate_row_bindings(
    case_row: object,
    process_row: object,
    source_manifest: object,
    matched_configuration: object,
) -> None:
    validate_case_row(case_row)
    validate_process_row(process_row)
    validate_source_manifest(source_manifest)
    validate_matched_configuration(matched_configuration)
    for field in SHARED_CASE_PROCESS_FIELDS:
        if case_row[field] != process_row[field]:
            raise ValueError(f"binding mismatch: {field}")
    source_fields = (
        "source_tree_sha256",
        "dirty_tree_policy",
        "gate1_audit_sha256",
        "execution_plan_sha256",
        "source_bundle_sha256",
        "source_package_sha256",
        "producer_source_sha256",
        "producer_version_sha256",
        "verifier_source_sha256",
        "verifier_version_sha256",
    )
    for field in source_fields:
        if case_row[field] != source_manifest[field]:
            raise ValueError(f"binding mismatch: {field}")
    configuration_fields = (
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
    for field in configuration_fields:
        if case_row[field] != matched_configuration[field]:
            raise ValueError(f"binding mismatch: {field}")
    if process_row["world_size"] != matched_configuration["tp_world_size"]:
        raise ValueError("binding mismatch: world_size")


def _validate_gpu_assignments(rows: object) -> None:
    if not isinstance(rows, list) or len(rows) != WORLD_SIZE:
        raise ValueError("GPU assignments are invalid")
    expected = [
        {
            "rank": rank,
            "gpu_index": gpu_index,
            "cuda_visible_device": str(rank),
        }
        for rank, gpu_index in enumerate(REQUIRED_GPU_INDICES)
    ]
    for row in rows:
        _validate_closed_mapping(
            row,
            GPU_ASSIGNMENT_FIELDS,
            "GPU assignment row",
        )
    if rows != expected:
        raise ValueError("GPU assignments are invalid")


def _validate_case_port_pairs(rows: object) -> None:
    expected_case_ids = [
        case.case_id for case in build_case_matrix()
    ]
    if not isinstance(rows, list) or len(rows) != len(expected_case_ids):
        raise ValueError("case port pairs are invalid")
    seen_ports = set()
    actual_case_ids = []
    for row in rows:
        _validate_closed_mapping(
            row,
            CASE_PORT_PAIR_FIELDS,
            "case port pair",
        )
        actual_case_ids.append(
            _require_nonempty_string(row["case_id"], "case id")
        )
        for field in ("tinyvllm_dist_port", "master_port"):
            port = _require_positive_int(
                row[field],
                field.replace("_", " "),
            )
            if port > 65535 or port in seen_ports:
                raise ValueError("case port pairs are invalid")
            seen_ports.add(port)
        if row["tinyvllm_dist_port"] == row["master_port"]:
            raise ValueError("case port pairs are invalid")
    if actual_case_ids != expected_case_ids:
        raise ValueError("case port pairs are invalid")


def _validate_artifact_paths(paths: object) -> None:
    _validate_closed_mapping(
        paths,
        ARTIFACT_PATH_FIELDS,
        "artifact paths",
    )
    for field in ARTIFACT_PATH_FIELDS:
        _require_safe_relative_path(
            paths[field],
            field.replace("_", " "),
        )


def _validate_source_inventory(rows: object, label: str) -> None:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} is invalid")
    paths = []
    for row in rows:
        _validate_closed_mapping(
            row,
            SOURCE_INVENTORY_ROW_FIELDS,
            f"{label} row",
        )
        path = _require_safe_relative_path(row["path"], "path")
        if row["type"] != "file":
            raise ValueError(f"{label} type is invalid")
        _require_sha256(row["sha256"], "sha256")
        _require_nonnegative_int(row["bytes"], "bytes")
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError(f"{label} paths are invalid")


def _validate_gpu_resource_rows(
    rows: object,
    label: str,
    *,
    require_available: bool,
) -> None:
    if not isinstance(rows, list) or len(rows) != WORLD_SIZE:
        raise ValueError(f"{label} is invalid")
    if [row.get("gpu_index") for row in rows] != list(
        REQUIRED_GPU_INDICES
    ):
        raise ValueError(f"{label} GPU indices are invalid")
    uuids = []
    for row in rows:
        _validate_closed_mapping(
            row,
            GPU_RESOURCE_ROW_FIELDS,
            f"{label} row",
        )
        _require_nonempty_string(row["gpu_uuid"], "gpu uuid")
        free_bytes = _require_nonnegative_int(
            row["free_bytes"],
            "free bytes",
        )
        if require_available and free_bytes < MIN_GPU_FREE_BYTES:
            raise ValueError(f"{label} free bytes are invalid")
        if (
            not isinstance(row["compute_processes"], list)
            or any(
                not isinstance(process, str) or not process
                for process in row["compute_processes"]
            )
        ):
            raise ValueError(f"{label} compute processes are invalid")
        if require_available and row["compute_processes"] != []:
            raise ValueError(f"{label} compute processes are invalid")
        uuids.append(row["gpu_uuid"])
    if len(uuids) != len(set(uuids)):
        raise ValueError(f"{label} GPU UUIDs are invalid")


def _validate_command_order(value: object) -> None:
    if value != list(EXECUTION_COMMAND_ORDER):
        raise ValueError("command order is invalid")


def _validate_command_manifest(rows: object) -> None:
    if not isinstance(rows, list) or len(rows) != len(
        EXECUTION_COMMAND_ORDER
    ):
        raise ValueError("command manifest is invalid")
    for row in rows:
        _validate_closed_mapping(
            row,
            COMMAND_MANIFEST_ROW_FIELDS,
            "command manifest row",
        )
        _require_sha256(row["command_sha256"], "command sha256")
    if [row["name"] for row in rows] != list(EXECUTION_COMMAND_ORDER):
        raise ValueError("command order is invalid")


def _validate_command_results(
    rows: object,
    command_manifest: object,
    *,
    lifecycle_state: str,
) -> int | None:
    if not isinstance(rows, list) or len(rows) != len(
        EXECUTION_COMMAND_ORDER
    ):
        raise ValueError("command results are invalid")
    expected_hashes = {
        row["name"]: row["command_sha256"]
        for row in command_manifest
    }
    for row in rows:
        _validate_closed_mapping(
            row,
            COMMAND_RESULT_FIELDS,
            "command result row",
        )
        if row["name"] not in expected_hashes:
            raise ValueError("command order is invalid")
        if row["command_sha256"] != expected_hashes[row["name"]]:
            raise ValueError("command hash is invalid")
        if row["outcome"] not in {"attempted", "skipped"}:
            raise ValueError("command outcome is invalid")
        if row["outcome"] == "attempted":
            if type(row["returncode"]) is not int:
                raise ValueError("command returncode is invalid")
        elif row["returncode"] is not None:
            raise ValueError("skipped command returncode is invalid")
        for field in ("stdout", "stderr"):
            if (
                not isinstance(row[field], str)
                or len(row[field].encode("utf-8"))
                > MAX_BOUNDED_OUTPUT_BYTES
            ):
                raise ValueError("command output is not bounded")
        for field in ("stdout_truncated", "stderr_truncated"):
            if type(row[field]) is not bool:
                raise ValueError("command truncation marker is invalid")
    if [row["name"] for row in rows] != list(EXECUTION_COMMAND_ORDER):
        raise ValueError("command order is invalid")
    if lifecycle_state == "execution_success":
        if any(
            row["outcome"] != "attempted" or row["returncode"] != 0
            for row in rows
        ):
            raise ValueError("successful command result is invalid")
        return None
    failure_indices = [
        index
        for index, row in enumerate(rows)
        if row["outcome"] == "attempted" and row["returncode"] != 0
    ]
    if not failure_indices:
        raise ValueError("execution failure requires a nonzero returncode")
    failure_index = failure_indices[0]
    for index, row in enumerate(rows):
        if index <= failure_index:
            if row["outcome"] != "attempted":
                raise ValueError("attempted command ordering is invalid")
        elif (
            row["outcome"] != "skipped"
            or row["returncode"] is not None
            or row["stdout"] != ""
            or row["stderr"] != ""
            or row["stdout_truncated"]
            or row["stderr_truncated"]
        ):
            raise ValueError("skipped command result is invalid")
    return failure_index


def _validate_execution_provenance(document: object) -> None:
    for field in EXECUTION_PROVENANCE_FIELDS:
        _require_sha256(document[field], field.replace("_", " "))
    if document["model_manifest_sha256"] != MODEL_MANIFEST_SHA256:
        raise ValueError("model manifest sha256 is invalid")


def validate_evidence_document(kind: object, document: object) -> None:
    if kind not in EVIDENCE_DOCUMENT_FIELDS:
        raise ValueError("evidence document kind is invalid")
    fields = EVIDENCE_DOCUMENT_FIELDS[kind]
    _validate_closed_mapping(document, fields, f"{kind} document")
    if document["schema_version"] != EVIDENCE_SCHEMA_VERSIONS[kind]:
        raise ValueError("schema version is invalid")
    for field in fields:
        value = document[field]
        label = field.replace("_", " ")
        if field.endswith("_sha256") or field == "sha256":
            nullable_guard_hash = (
                kind == "execution_receipt"
                and field
                in {
                    "resource_guard_before_sha256",
                    "resource_guard_after_sha256",
                }
            )
            if value is not None or not nullable_guard_hash:
                _require_sha256(value, label)
        elif field in {"path", "inventory_path"}:
            _require_safe_relative_path(value, label)
        elif field in {"run_tag", "nonce"}:
            _require_nonempty_string(value, label)
        elif field in {"checks", "world_size"}:
            _require_positive_int(value, label)
        elif field == "minimum_free_bytes_per_gpu":
            _require_positive_int(value, label)
            if value != MIN_GPU_FREE_BYTES:
                raise ValueError(f"{label} is invalid")
        elif field == "required_gpu_indices":
            if value != list(REQUIRED_GPU_INDICES):
                raise ValueError(f"{label} is invalid")
        elif field == "dirty_tree_policy":
            if value not in DIRTY_TREE_POLICIES:
                raise ValueError(f"{label} is invalid")
        elif field == "classification":
            accepted = {
                "gate1_audit": {"PASS"},
                "preflight": {"READY", "BLOCKED_RESOURCES"},
                "execution_receipt": {
                    "PASS",
                    "BLOCKED_RESOURCES",
                    "INVALID_ARTIFACT",
                },
                "verifier_output": set(RESULTS),
                "independent_verification": set(RESULTS),
            }[kind]
            if value not in accepted:
                raise ValueError(f"{label} is invalid")
        elif field == "consumed":
            if value is not True:
                raise ValueError(f"{label} is invalid")
        elif field == "phase":
            if value not in {"before", "after"}:
                raise ValueError(f"{label} is invalid")
        elif field == "role":
            if value not in {"local", "remote"}:
                raise ValueError(f"{label} is invalid")
        elif field == "local_verifier_role":
            if value != "local":
                raise ValueError(f"{label} is invalid")
        elif field == "remote_verifier_role":
            if value != "remote":
                raise ValueError(f"{label} is invalid")


def validate_execution_evidence_bundle(bundle: object) -> None:
    if not isinstance(bundle, Mapping):
        raise ValueError("execution evidence bundle must be a mapping")
    lifecycle_state = bundle.get("lifecycle_state")
    if lifecycle_state not in EXECUTION_LIFECYCLE_STATES:
        raise ValueError("execution lifecycle state is invalid")
    required = EXECUTION_BUNDLE_DOCUMENTS[lifecycle_state]
    failure_index_hint = None
    if lifecycle_state == "execution_failed":
        receipt_hint = bundle.get("execution_receipt")
        if isinstance(receipt_hint, Mapping):
            results_hint = receipt_hint.get("command_results")
            if isinstance(results_hint, list):
                failure_indices = [
                    index
                    for index, row in enumerate(results_hint)
                    if isinstance(row, Mapping)
                    and row.get("outcome") == "attempted"
                    and type(row.get("returncode")) is int
                    and row["returncode"] != 0
                ]
                if failure_indices:
                    failure_index_hint = failure_indices[0]
    guard_documents = {
        name
        for name in ("resource_guard_before", "resource_guard_after")
        if name in bundle
    }
    if lifecycle_state == "execution_failed" and guard_documents in (
        set(),
        {"resource_guard_before"},
    ):
        required = tuple(
            name
            for name in required
            if name not in {
                "resource_guard_before",
                "resource_guard_after",
            }
            or name in guard_documents
        )
    remote_output_failure_indices = {
        EXECUTION_COMMAND_ORDER.index("final_resource_guard"),
        EXECUTION_COMMAND_ORDER.index("package_download"),
        EXECUTION_COMMAND_ORDER.index("safe_extract"),
        EXECUTION_COMMAND_ORDER.index("local_verify"),
    }
    if failure_index_hint in remote_output_failure_indices:
        required = (*required, "remote_verifier_output")
    _validate_closed_mapping(
        bundle,
        required,
        "execution evidence bundle",
    )
    kind_by_name = {
        "environment": "environment",
        "gpu_assignments": "gpu_assignments",
        "commands": "commands",
        "preflight": "preflight",
        "execution_plan": "execution_plan",
        "consumed_authorization": "consumed_authorization",
        "source_bundle": "source_bundle",
        "source_package": "source_package",
        "resource_guard_before": "resource_guard",
        "resource_guard_after": "resource_guard",
        "execution_receipt": "execution_receipt",
        "local_verifier_output": "verifier_output",
        "remote_verifier_output": "verifier_output",
        "independent_verification": "independent_verification",
    }
    for name in required:
        if name == "lifecycle_state":
            continue
        kind = kind_by_name[name]
        validate_evidence_document(kind, bundle[name])

    preflight = bundle["preflight"]
    environment = bundle["environment"]
    gpu_assignments = bundle["gpu_assignments"]

    for document in (environment, preflight):
        _validate_execution_provenance(document)
    shared_identity = ("run_tag", "nonce", *EXECUTION_PROVENANCE_FIELDS)
    for document in (preflight,):
        for field in shared_identity:
            if document[field] != environment[field]:
                raise ValueError(f"binding mismatch: {field}")

    _validate_gpu_assignments(gpu_assignments["assignments"])
    if gpu_assignments["world_size"] != WORLD_SIZE:
        raise ValueError("world size is invalid")
    if gpu_assignments["required_gpu_indices"] != list(
        REQUIRED_GPU_INDICES
    ):
        raise ValueError("required gpu indices are invalid")
    if gpu_assignments["run_tag"] != environment["run_tag"]:
        raise ValueError("GPU assignments run tag mismatch")
    if preflight["world_size"] != WORLD_SIZE:
        raise ValueError("world size is invalid")
    if preflight["required_gpu_indices"] != list(REQUIRED_GPU_INDICES):
        raise ValueError("required gpu indices are invalid")
    if preflight["minimum_free_bytes_per_gpu"] != MIN_GPU_FREE_BYTES:
        raise ValueError("minimum free bytes per gpu is invalid")

    if lifecycle_state == "preflight_blocked":
        _validate_gpu_resource_rows(
            preflight["gpu_query_rows"],
            "preflight GPU query rows",
            require_available=False,
        )
        resource_blocked = any(
            row["free_bytes"] < MIN_GPU_FREE_BYTES
            or row["compute_processes"] != []
            for row in preflight["gpu_query_rows"]
        )
        if (
            preflight["classification"] != "BLOCKED_RESOURCES"
            or not resource_blocked
        ):
            raise ValueError("blocked resource classification is invalid")
        if (
            not isinstance(preflight["blocking_reasons"], list)
            or not preflight["blocking_reasons"]
            or any(
                not isinstance(reason, str) or not reason
                for reason in preflight["blocking_reasons"]
            )
        ):
            raise ValueError("preflight blocking reasons are invalid")
        if preflight["worker_authorized"] is not False:
            raise ValueError("preflight worker authorization is invalid")
        for field in (
            "remote_path_created",
            "source_staged",
            "worker_launched",
        ):
            if preflight[field] is not False:
                raise ValueError("preflight side effect is invalid")
        return

    plan = bundle["execution_plan"]
    authorization = bundle["consumed_authorization"]
    receipt = bundle["execution_receipt"]
    commands = bundle["commands"]
    source_bundle = bundle["source_bundle"]
    source_package = bundle["source_package"]
    before = bundle.get("resource_guard_before")
    after = bundle.get("resource_guard_after")
    for field in (
        "remote_path_created",
        "source_staged",
        "worker_launched",
    ):
        if type(receipt[field]) is not bool:
            raise ValueError(f"{field.replace('_', ' ')} is invalid")

    for document in (
        plan,
        authorization,
        source_bundle,
        source_package,
        receipt,
    ):
        _validate_execution_provenance(document)
        for field in shared_identity:
            if document[field] != environment[field]:
                raise ValueError(f"binding mismatch: {field}")
    for document in (preflight, plan, authorization):
        if document["world_size"] != WORLD_SIZE:
            raise ValueError("world size is invalid")
        if document["required_gpu_indices"] != list(
            REQUIRED_GPU_INDICES
        ):
            raise ValueError("required gpu indices are invalid")
    for document in (plan, authorization):
        _validate_gpu_assignments(document["gpu_assignments"])
        if document["gpu_assignments"] != gpu_assignments["assignments"]:
            raise ValueError("GPU assignments binding mismatch")
        _validate_case_port_pairs(document["case_port_pairs"])
        _validate_artifact_paths(document["artifact_paths"])
    if authorization["case_port_pairs"] != plan["case_port_pairs"]:
        raise ValueError("case port pair binding mismatch")
    if authorization["artifact_paths"] != plan["artifact_paths"]:
        raise ValueError("artifact path binding mismatch")

    _validate_command_order(plan["command_order"])
    _validate_command_order(commands["command_order"])
    _validate_command_order(receipt["command_order"])
    canonical_commands = canonical_execution_commands(plan)
    validate_execution_command_semantics(
        canonical_commands,
        expected_order=EXECUTION_COMMAND_ORDER,
        execution_plan=plan,
    )
    _validate_command_manifest(commands["commands"])
    expected_command_manifest = [
        {
            "name": name,
            "command_sha256": execution_command_sha256(
                canonical_commands[name]
            ),
        }
        for name in EXECUTION_COMMAND_ORDER
    ]
    if commands["commands"] != expected_command_manifest:
        raise ValueError("command manifest is not canonical")
    failure_index = _validate_command_results(
        receipt["command_results"],
        commands["commands"],
        lifecycle_state=lifecycle_state,
    )
    resource_guard_index = EXECUTION_COMMAND_ORDER.index("resource_guard")
    workers_index = EXECUTION_COMMAND_ORDER.index("workers")
    assembly_index = EXECUTION_COMMAND_ORDER.index("assembly")
    remote_verify_index = EXECUTION_COMMAND_ORDER.index("remote_verify")
    final_resource_guard_index = EXECUTION_COMMAND_ORDER.index(
        "final_resource_guard"
    )
    if lifecycle_state != "execution_failed":
        expected_guard_documents = {
            "resource_guard_before",
            "resource_guard_after",
        }
    elif failure_index <= resource_guard_index:
        expected_guard_documents = set()
    elif failure_index in (
        workers_index,
        assembly_index,
        remote_verify_index,
        final_resource_guard_index,
    ):
        expected_guard_documents = {"resource_guard_before"}
    else:
        expected_guard_documents = {
            "resource_guard_before",
            "resource_guard_after",
        }
    if guard_documents != expected_guard_documents:
        raise ValueError(
            "resource guard documents do not match the failed command"
        )
    if commands["command_manifest_sha256"] != (
        environment["command_manifest_sha256"]
    ):
        raise ValueError("command manifest binding mismatch")
    if commands["command_manifest_sha256"] != canonical_json_sha256(
        commands["commands"]
    ):
        raise ValueError("command manifest sha256 is invalid")
    if authorization["execution_plan_sha256"] != canonical_json_sha256(
        plan
    ):
        raise ValueError("execution plan sha256 is invalid")
    if commands["execution_plan_sha256"] != (
        authorization["execution_plan_sha256"]
    ):
        raise ValueError("execution plan binding mismatch")
    if receipt["execution_plan_sha256"] != (
        authorization["execution_plan_sha256"]
    ):
        raise ValueError("execution plan binding mismatch")
    if receipt["consumed_authorization_sha256"] != canonical_json_sha256(
        authorization
    ):
        raise ValueError("consumed authorization sha256 is invalid")

    _require_nonempty_string(
        authorization["authorization_id"],
        "authorization id",
    )
    if receipt["authorization_id"] != authorization["authorization_id"]:
        raise ValueError("authorization identity mismatch")
    for field in ("active_path", "consumed_path"):
        _require_safe_relative_path(
            authorization[field],
            field.replace("_", " "),
        )
    if authorization["active_path"] == authorization["consumed_path"]:
        raise ValueError("authorization consumption paths are invalid")
    if (
        authorization["consumed"] is not True
        or authorization["consumed_once"] is not True
    ):
        raise ValueError("authorization consumption identity is invalid")

    _validate_source_inventory(
        source_bundle["inventory"],
        "source bundle inventory",
    )
    _validate_source_inventory(
        source_package["inventory"],
        "source package inventory",
    )
    _validate_source_inventory(
        receipt["source_inventory"],
        "execution receipt source inventory",
    )
    if not (
        source_bundle["inventory"]
        == source_package["inventory"]
        == receipt["source_inventory"]
    ):
        raise ValueError("source inventory binding mismatch")
    expected_source_inventory_sha = canonical_json_sha256(
        source_package["inventory"]
    )
    if (
        source_bundle["inventory_sha256"]
        != expected_source_inventory_sha
        or source_package["inventory_sha256"]
        != expected_source_inventory_sha
    ):
        raise ValueError("source inventory sha256 is invalid")
    if source_bundle["sha256"] != environment["source_bundle_sha256"]:
        raise ValueError("source bundle sha256 binding mismatch")
    if source_package["sha256"] != environment["source_package_sha256"]:
        raise ValueError("source package sha256 binding mismatch")
    package_inventory_produced = (
        lifecycle_state == "execution_success"
        or failure_index > assembly_index
    )
    final_inventory_produced = (
        lifecycle_state == "execution_success"
        or failure_index
        > EXECUTION_COMMAND_ORDER.index("safe_extract")
    )
    if package_inventory_produced:
        _validate_source_inventory(
            receipt["package_inventory"],
            "package inventory",
        )
    elif receipt["package_inventory"] != []:
        raise ValueError("package inventory is invalid")
    if final_inventory_produced:
        _validate_source_inventory(
            receipt["final_inventory"],
            "final inventory",
        )
    elif receipt["final_inventory"] != []:
        raise ValueError("final inventory is invalid")
    if final_inventory_produced:
        artifact_manifest_rows = [
            row
            for row in receipt["final_inventory"]
            if row["path"] == "artifact_manifest.json"
        ]
        if len(artifact_manifest_rows) != 1:
            raise ValueError("artifact manifest verifier binding mismatch")
        if [
            row["path"] for row in receipt["package_inventory"]
        ] != sorted(ARTIFACT_MANIFEST_HASH_DOMAIN):
            raise ValueError("package inventory producer domain mismatch")
        if [
            row["path"] for row in receipt["final_inventory"]
        ] != sorted(PRODUCER_TRUST_DOMAIN):
            raise ValueError("final inventory producer trust domain mismatch")
        final_producer_inventory = [
            row
            for row in receipt["final_inventory"]
            if row["path"] != "artifact_manifest.json"
        ]
        if receipt["package_inventory"] != final_producer_inventory:
            raise ValueError(
                "package/final inventory equality and artifact manifest "
                "verifier binding mismatch"
            )
    artifact_manifest_sha256 = None
    if lifecycle_state == "execution_success":
        artifact_manifest_rows = [
            row
            for row in receipt["final_inventory"]
            if isinstance(row, Mapping)
            and row.get("path") == "artifact_manifest.json"
        ]
        if len(artifact_manifest_rows) != 1:
            raise ValueError("artifact manifest verifier binding mismatch")
        artifact_manifest_sha256 = artifact_manifest_rows[0].get("sha256")
    if receipt["package_inventory_sha256"] != canonical_json_sha256(
        receipt["package_inventory"]
    ):
        raise ValueError("package inventory sha256 is invalid")
    if receipt["final_inventory_sha256"] != canonical_json_sha256(
        receipt["final_inventory"]
    ):
        raise ValueError("final inventory sha256 is invalid")

    _validate_gpu_resource_rows(
        preflight["gpu_query_rows"],
        "preflight GPU query rows",
        require_available=True,
    )
    if (
        preflight["classification"] != "READY"
        or preflight["blocking_reasons"] != []
    ):
        raise ValueError("preflight blocking reasons are invalid")
    if preflight["worker_authorized"] is not True:
        raise ValueError("preflight worker authorization is invalid")
    for field in (
        "remote_path_created",
        "source_staged",
        "worker_launched",
    ):
        if preflight[field] is not False:
            raise ValueError("preflight side effect is invalid")

    if guard_documents:
        for name, document, phase in (
            ("resource_guard_before", before, "before"),
            ("resource_guard_after", after, "after"),
        ):
            if name not in guard_documents:
                continue
            if document["run_tag"] != environment["run_tag"]:
                raise ValueError("resource guard run tag mismatch")
            if document["phase"] != phase:
                raise ValueError("resource guard phase is invalid")
            _validate_gpu_resource_rows(
                document["gpu_query_rows"],
                f"{phase} resource guard rows",
                require_available=True,
            )
            if document["side_effects_observed"] is not False:
                raise ValueError("resource guard side effect is invalid")
            if document["sha256"] != resource_guard_sha256(document):
                raise ValueError(f"{phase} resource guard sha256 is invalid")
        if guard_documents == {
            "resource_guard_before",
            "resource_guard_after",
        } and [
            (row["gpu_index"], row["gpu_uuid"])
            for row in before["gpu_query_rows"]
        ] != [
            (row["gpu_index"], row["gpu_uuid"])
            for row in after["gpu_query_rows"]
        ]:
            raise ValueError("resource guard GPU identity drift")
    for name, document in (
        ("resource_guard_before", before),
        ("resource_guard_after", after),
    ):
        field = f"{name}_sha256"
        expected = (
            resource_guard_sha256(document)
            if name in guard_documents
            else None
        )
        if receipt[field] != expected:
            raise ValueError(f"{field.replace('_', ' ')} is invalid")

    _validate_artifact_paths(receipt["artifact_paths"])
    if receipt["artifact_paths"] != plan["artifact_paths"]:
        raise ValueError("artifact path binding mismatch")
    if lifecycle_state == "execution_success":
        if receipt["classification"] != "PASS":
            raise ValueError("execution receipt classification is invalid")
        for field in (
            "remote_path_created",
            "source_staged",
            "worker_launched",
            "cleanup_complete",
        ):
            if receipt[field] is not True:
                raise ValueError(f"{field.replace('_', ' ')} is invalid")
    else:
        if receipt["classification"] != "INVALID_ARTIFACT":
            raise ValueError("execution receipt classification is invalid")
        producer_indices = {
            "remote_path_created": EXECUTION_COMMAND_ORDER.index(
                "reserve_remote"
            ),
            "source_staged": EXECUTION_COMMAND_ORDER.index("stage"),
            "worker_launched": EXECUTION_COMMAND_ORDER.index("workers"),
        }
        for field, producer_index in producer_indices.items():
            if (
                failure_index > producer_index
                and receipt[field] is not True
            ) or (
                failure_index < producer_index
                and receipt[field] is not False
            ):
                raise ValueError(
                    f"{field.replace('_', ' ')} is invalid"
                )
        if (
            receipt["source_staged"]
            and not receipt["remote_path_created"]
        ):
            raise ValueError("source staged side effect is invalid")
        if receipt["worker_launched"] and (
            not receipt["source_staged"]
            or not receipt["remote_path_created"]
        ):
            raise ValueError("worker launched side effect is invalid")
        cleanup_obligation = any(
            receipt[field] for field in producer_indices
        )
        if receipt["cleanup_complete"] is not cleanup_obligation:
            raise ValueError("cleanup complete is invalid")
        if failure_index not in remote_output_failure_indices:
            return
        remote = bundle["remote_verifier_output"]
        if remote["role"] != "remote":
            raise ValueError("remote verifier role is invalid")
        if (
            remote["verifier_source_sha256"]
            != environment["verifier_source_sha256"]
            or remote["verifier_version_sha256"]
            != environment["verifier_version_sha256"]
        ):
            raise ValueError("verifier identity mismatch")
        authority_inventory = (
            receipt["final_inventory"]
            if failure_index
            > EXECUTION_COMMAND_ORDER.index("safe_extract")
            else receipt["package_inventory"]
        )
        artifact_manifest_rows = [
            row
            for row in authority_inventory
            if row["path"] == "artifact_manifest.json"
        ]
        if (
            len(artifact_manifest_rows) != 1
            or artifact_manifest_rows[0]["sha256"]
            != remote["artifact_manifest_sha256"]
        ):
            raise ValueError("artifact manifest verifier binding mismatch")
        if remote["classification"] not in (
            "GO",
            "NO_GO_CORRECTNESS",
            "NO_GO_RUNTIME_SAFETY",
            "NO_GO_CACHE",
            "NO_GO_PERFORMANCE",
            "INVALID_ARTIFACT",
        ):
            raise ValueError(
                "execution failed remote verifier classification is invalid"
            )
        return

    local = bundle["local_verifier_output"]
    remote = bundle["remote_verifier_output"]
    final = bundle["independent_verification"]
    if local["role"] != "local":
        raise ValueError("local verifier role is invalid")
    if remote["role"] != "remote":
        raise ValueError("remote verifier role is invalid")
    for verifier in (local, remote):
        if (
            verifier["verifier_source_sha256"]
            != environment["verifier_source_sha256"]
            or verifier["verifier_version_sha256"]
            != environment["verifier_version_sha256"]
        ):
            raise ValueError("verifier identity mismatch")
    if (
        local["artifact_manifest_sha256"]
        != remote["artifact_manifest_sha256"]
        or final["artifact_manifest_sha256"]
        != local["artifact_manifest_sha256"]
        or local["artifact_manifest_sha256"]
        != artifact_manifest_sha256
    ):
        raise ValueError("artifact manifest verifier binding mismatch")
    if final["local_verifier_sha256"] != canonical_json_sha256(local):
        raise ValueError("local verifier sha256 is invalid")
    if final["remote_verifier_sha256"] != canonical_json_sha256(remote):
        raise ValueError("remote verifier sha256 is invalid")
    if not (
        local["classification"]
        == remote["classification"]
        == final["classification"]
    ):
        raise ValueError(
            "verifier classification disagreement is INVALID_ARTIFACT"
        )
    if (
        lifecycle_state == "execution_success"
        and final["classification"]
        not in (
            "GO",
            "NO_GO_CORRECTNESS",
            "NO_GO_RUNTIME_SAFETY",
            "NO_GO_CACHE",
            "NO_GO_PERFORMANCE",
        )
    ):
        raise ValueError(
            "execution success classification is invalid"
        )


def _validate_nested_file(row: object) -> None:
    _validate_closed_mapping(row, NESTED_FILE_FIELDS, "nested file")
    _require_safe_relative_path(row["path"], "path")
    _require_sha256(row["sha256"], "sha256")
    _require_nonnegative_int(row["bytes"], "bytes")
    if row["type"] != "regular_file":
        raise ValueError("nested file type is invalid")


def _nested_request_keys():
    return [
        (case.case_id, f"request-{request_index}")
        for case in build_case_matrix()
        for request_index in range(
            WORKLOAD_SPECS[case.workload]["continuations"]
        )
    ]


def _nested_rank_keys():
    return [
        (case.case_id, rank)
        for case in build_case_matrix()
        for rank in range(WORLD_SIZE)
    ]


def _validate_nested_manifest(kind: str, manifest: object) -> None:
    _validate_closed_mapping(
        manifest,
        NESTED_MANIFEST_FIELDS,
        f"{kind} manifest",
    )
    if manifest["schema_version"] != NESTED_MANIFEST_SCHEMA_VERSIONS[
        kind
    ]:
        raise ValueError("nested manifest schema version is invalid")
    if manifest["kind"] != kind:
        raise ValueError("nested manifest kind is invalid")
    files = manifest["files"]
    rows = manifest["rows"]
    if not isinstance(files, list) or not isinstance(rows, list):
        raise ValueError("nested manifest rows are invalid")
    file_paths = []
    for file_row in files:
        _validate_nested_file(file_row)
        file_paths.append(file_row["path"])
    if file_paths != sorted(file_paths) or len(file_paths) != len(
        set(file_paths)
    ):
        raise ValueError("nested manifest file inventory is invalid")
    file_by_path = {row["path"]: row for row in files}
    local_file_fields = {
        "snapshots": ("snapshot_file",),
    }.get(kind, ("file",))
    for row in rows:
        _validate_closed_mapping(
            row,
            NESTED_ROW_FIELDS[kind],
            f"{kind} row",
        )
        for field in local_file_fields:
            if field not in row or row[field] is None:
                continue
            _validate_nested_file(row[field])
            if file_by_path.get(row[field]["path"]) != row[field]:
                raise ValueError(
                    "nested manifest row file binding is invalid"
                )
    referenced = []
    for row in rows:
        for field in local_file_fields:
            if field in row and row[field] is not None:
                referenced.append(row[field]["path"])
    if sorted(referenced) != file_paths:
        raise ValueError("nested manifest file inventory is invalid")


def _validate_prerequisite_manifest(manifest: object) -> None:
    expected = [
        (name, role)
        for name in PREREQUISITE_NAMES
        for role in (
            "artifact",
            "independent_verification",
            "provenance",
        )
    ]
    actual = []
    for row in manifest["rows"]:
        if row["name"] not in PREREQUISITE_NAMES:
            raise ValueError("prerequisite name is invalid")
        if row["role"] not in {
            "artifact",
            "independent_verification",
            "provenance",
        }:
            raise ValueError("prerequisite role is invalid")
        actual.append((row["name"], row["role"]))
    if actual != expected:
        raise ValueError("prerequisite coverage is invalid")


def _validate_token_manifest(manifest: object) -> None:
    expected = [
        (case_id, request_id, role)
        for case_id, request_id in _nested_request_keys()
        for role in ("prompt", "output")
    ]
    actual = []
    case_by_id = {
        case.case_id: case for case in build_case_matrix()
    }
    for row in manifest["rows"]:
        case = case_by_id.get(row["case_id"])
        if case is None:
            raise ValueError("token coverage is invalid")
        if row["role"] not in {"prompt", "output"}:
            raise ValueError("token role is invalid")
        expected_count = (
            WORKLOAD_SPECS[case.workload]["shared_prefix_tokens"]
            + WORKLOAD_SPECS[case.workload]["suffix_tokens"]
            if row["role"] == "prompt"
            else WORKLOAD_SPECS[case.workload]["generated_tokens"]
        )
        if (
            _require_positive_int(row["token_count"], "token count")
            != expected_count
        ):
            raise ValueError("token count is invalid")
        actual.append(
            (row["case_id"], row["request_id"], row["role"])
        )
    if actual != expected or len(actual) != len(set(actual)):
        raise ValueError("token coverage is invalid")


def _validate_logit_manifest(manifest: object) -> None:
    expected = _nested_request_keys()
    actual = []
    for row in manifest["rows"]:
        shape = row["shape"]
        if (
            not isinstance(shape, list)
            or shape != [MODEL_VOCAB_SIZE]
        ):
            raise ValueError("logit shape is invalid")
        if row["dtype"] != "float32":
            raise ValueError("logit dtype is invalid")
        actual.append((row["case_id"], row["request_id"]))
    if actual != expected or len(actual) != len(set(actual)):
        raise ValueError("logit coverage is invalid")


def validate_case_row_nested_evidence_bindings(
    case_rows: object,
    manifests: object,
) -> None:
    validate_case_rows(case_rows)
    if not isinstance(manifests, dict):
        raise ValueError("nested evidence manifests are invalid")
    try:
        token_manifest = manifests["tokens"]
        logit_manifest = manifests["logits"]
    except KeyError as error:
        raise ValueError("nested evidence manifests are invalid") from error
    try:
        _validate_nested_manifest("tokens", token_manifest)
        _validate_token_manifest(token_manifest)
    except ValueError as error:
        raise ValueError("token evidence binding is invalid") from error
    try:
        _validate_nested_manifest("logits", logit_manifest)
        _validate_logit_manifest(logit_manifest)
    except ValueError as error:
        raise ValueError("logit evidence binding is invalid") from error

    token_rows = {
        (row["case_id"], row["request_id"], row["role"]): row
        for row in token_manifest["rows"]
    }
    logit_rows = {
        (row["case_id"], row["request_id"]): row
        for row in logit_manifest["rows"]
    }
    for case_row in case_rows:
        key = (case_row["case_id"], case_row["request_id"])
        prompt = token_rows.get(key + ("prompt",))
        output = token_rows.get(key + ("output",))
        logit = logit_rows.get(key)
        if prompt is None or output is None or logit is None:
            raise ValueError("case request evidence binding is invalid")
        bindings = (
            (
                case_row["prompt_token_ids_path"],
                prompt["file"]["path"],
                "prompt evidence binding is invalid",
            ),
            (
                case_row["prompt_token_ids_sha256"],
                prompt["file"]["sha256"],
                "prompt evidence binding is invalid",
            ),
            (
                case_row["output_token_ids_path"],
                output["file"]["path"],
                "output evidence binding is invalid",
            ),
            (
                case_row["output_token_ids_sha256"],
                output["file"]["sha256"],
                "output evidence binding is invalid",
            ),
            (
                case_row["final_logits_path"],
                logit["file"]["path"],
                "logit evidence binding is invalid",
            ),
            (
                case_row["final_logits_sha256"],
                logit["file"]["sha256"],
                "logit evidence binding is invalid",
            ),
            (
                case_row["final_logits_shape"],
                logit["shape"],
                "logit shape binding is invalid",
            ),
            (
                case_row["final_logits_dtype"],
                logit["dtype"],
                "logit dtype binding is invalid",
            ),
        )
        for actual, expected, message in bindings:
            if actual != expected:
                raise ValueError(message)


def _validate_log_manifest(manifest: object) -> None:
    expected = _nested_rank_keys()
    actual = []
    for row in manifest["rows"]:
        rank = _require_nonnegative_int(row["rank"], "rank")
        if row["world_size"] != WORLD_SIZE or rank >= WORLD_SIZE:
            raise ValueError("log world size is invalid")
        if row["completion_marker"] is not True:
            raise ValueError("log completion marker is invalid")
        if row["traceback_present"] is not False:
            raise ValueError("log traceback marker is invalid")
        actual.append((row["case_id"], rank))
    if actual != expected or len(actual) != len(set(actual)):
        raise ValueError("log coverage or duplicate row is invalid")


def validate_process_row_nested_worker_log_bindings(
    process_rows: object,
    manifests: object,
) -> None:
    validate_process_rows(process_rows)
    if not isinstance(manifests, dict) or "logs" not in manifests:
        raise ValueError("worker log evidence binding is invalid")
    log_manifest = manifests["logs"]
    try:
        _validate_nested_manifest("logs", log_manifest)
        _validate_log_manifest(log_manifest)
    except ValueError as error:
        raise ValueError("worker log evidence binding is invalid") from error

    process_keys = [
        (row["case_id"], row["rank"], row["world_size"])
        for row in process_rows
    ]
    log_keys = [
        (row["case_id"], row["rank"], row["world_size"])
        for row in log_manifest["rows"]
    ]
    if process_keys != log_keys:
        raise ValueError("worker log evidence binding is invalid")


def _validate_snapshot_manifests(
    snapshot_manifest: object,
    tensor_manifest: object,
) -> None:
    expected = _nested_rank_keys()
    actual = []
    tensor_keys = []
    tensor_files = {
        row["file"]["path"]: row["file"]
        for row in tensor_manifest["rows"]
    }
    case_by_id = {
        case.case_id: case for case in build_case_matrix()
    }
    for row in snapshot_manifest["rows"]:
        case = case_by_id.get(row["case_id"])
        if case is None or row["profile"] != case.profile:
            raise ValueError("snapshot profile evidence is invalid")
        rank = _require_nonnegative_int(row["rank"], "rank")
        if row["world_size"] != WORLD_SIZE or rank >= WORLD_SIZE:
            raise ValueError("snapshot world size is invalid")
        numeric_fields = NESTED_ROW_FIELDS["snapshots"][
            NESTED_ROW_FIELDS["snapshots"].index(
                "full_fidelity_logical_bytes"
            ):
        ]
        for field in numeric_fields:
            _require_nonnegative_int(
                row[field],
                field.replace("_", " "),
            )
        if case.profile == "recompute":
            if (
                row["evidence_kind"] != "accounting_only"
                or row["snapshot_file"] is not None
                or row["tensor_inventory_file"] is not None
                or any(row[field] != 0 for field in numeric_fields)
            ):
                raise ValueError("snapshot profile evidence is invalid")
        else:
            if (
                row["evidence_kind"] != "snapshot"
                or row["snapshot_file"] is None
                or row["tensor_inventory_file"] is None
                or row["full_fidelity_logical_bytes"] <= 0
                or row["encoded_physical_bytes"] <= 0
            ):
                raise ValueError("snapshot profile evidence is invalid")
            if tensor_files.get(
                row["tensor_inventory_file"]["path"]
            ) != row["tensor_inventory_file"]:
                raise ValueError(
                    "snapshot tensor inventory binding is invalid"
                )
            if (
                case.profile == P1_REFERENCE_PROFILE
                and row["codec_metadata_bytes"] != 0
            ):
                raise ValueError("snapshot profile evidence is invalid")
            if (
                case.profile == P2_PROFILE
                and row["codec_metadata_bytes"] <= 0
            ):
                raise ValueError("snapshot profile evidence is invalid")
            tensor_keys.append((case.case_id, case.profile, rank))
        actual.append((case.case_id, rank))
    if actual != expected or len(actual) != len(set(actual)):
        raise ValueError("snapshot coverage is invalid")

    actual_tensor_keys = []
    for row in tensor_manifest["rows"]:
        rank = _require_nonnegative_int(row["rank"], "rank")
        if row["world_size"] != WORLD_SIZE or rank >= WORLD_SIZE:
            raise ValueError("tensor inventory world size is invalid")
        if row["profile"] == "recompute":
            raise ValueError("tensor inventory profile is invalid")
        evidence = row["evidence"]
        validate_tensor_storage_evidence(evidence)
        for field in (
            "case_id",
            "profile",
            "representation",
            "representation_version",
            "codec",
            "rank",
            "world_size",
        ):
            if row[field] != evidence[field]:
                raise ValueError(
                    "tensor inventory evidence identity binding is invalid"
                )
        expected_counts = {
            "evidence_schema_version": evidence["schema_version"],
            "snapshot_count": len(evidence["snapshots"]),
            "storage_count": len(evidence["storages"]),
            "reference_count": sum(
                len(snapshot["tensor_references"])
                for snapshot in evidence["snapshots"]
            ),
            "observation_count": len(evidence["observations"]),
        }
        if any(row[field] != value for field, value in expected_counts.items()):
            raise ValueError("tensor inventory evidence count binding is invalid")
        if (
            row["file"]["sha256"]
            != canonical_json_file_sha256(evidence)
            or row["file"]["bytes"]
            != len(canonical_json_bytes(evidence)) + 1
        ):
            raise ValueError("tensor inventory evidence file binding is invalid")
        actual_tensor_keys.append(
            (row["case_id"], row["profile"], rank)
        )
    if (
        actual_tensor_keys != tensor_keys
        or len(actual_tensor_keys) != len(set(actual_tensor_keys))
    ):
        raise ValueError("tensor inventory coverage is invalid")


def validate_process_row_nested_snapshot_bindings(
    process_rows: object,
    manifests: object,
) -> None:
    validate_process_rows(process_rows)
    if (
        not isinstance(manifests, dict)
        or "snapshots" not in manifests
        or "tensor_inventories" not in manifests
    ):
        raise ValueError("snapshot evidence binding is invalid")
    snapshot_manifest = manifests["snapshots"]
    tensor_manifest = manifests["tensor_inventories"]
    try:
        _validate_nested_manifest("snapshots", snapshot_manifest)
        _validate_nested_manifest("tensor_inventories", tensor_manifest)
        _validate_snapshot_manifests(
            snapshot_manifest,
            tensor_manifest,
        )
    except ValueError as error:
        raise ValueError("snapshot evidence binding is invalid") from error

    process_keys = [
        (
            row["case_id"],
            row["profile"],
            row["rank"],
            row["world_size"],
        )
        for row in process_rows
    ]
    snapshot_keys = [
        (
            row["case_id"],
            row["profile"],
            row["rank"],
            row["world_size"],
        )
        for row in snapshot_manifest["rows"]
    ]
    if process_keys != snapshot_keys:
        raise ValueError("snapshot evidence binding is invalid")
    accounting_bindings = (
        *(
            (field, field)
            for field in TENSOR_STORAGE_ACCOUNTING_FIELDS
        ),
        ("hybrid_cache_current_logical_referenced_bytes", "full_fidelity_logical_bytes"),
        ("hybrid_cache_current_unique_physical_bytes", "encoded_physical_bytes"),
        ("hybrid_cache_current_metadata_bytes", "codec_metadata_bytes"),
        ("encode_workspace_peak_allocated_bytes", "temporary_encode_workspace_bytes"),
        ("decode_workspace_peak_allocated_bytes", "temporary_decode_workspace_bytes"),
    )
    tensor_rows = {
        (row["case_id"], row["profile"], row["rank"]): row
        for row in tensor_manifest["rows"]
    }
    for process_row, snapshot_row in zip(
        process_rows,
        snapshot_manifest["rows"],
    ):
        for process_field, snapshot_field in accounting_bindings:
            if process_row[process_field] != snapshot_row[snapshot_field]:
                raise ValueError(
                    "snapshot accounting binding mismatch: "
                    f"{process_field} != {snapshot_field}"
                )
        if process_row["profile"] == "recompute":
            if any(
                process_row[field] != 0
                for field in TENSOR_STORAGE_ACCOUNTING_FIELDS
            ):
                raise ValueError("recompute accounting evidence is invalid")
            continue
        tensor_row = tensor_rows.get((
            process_row["case_id"],
            process_row["profile"],
            process_row["rank"],
        ))
        if tensor_row is None:
            raise ValueError("tensor accounting evidence is missing")
        recomputed = recompute_tensor_storage_accounting(
            tensor_row["evidence"]
        )
        for field in TENSOR_STORAGE_ACCOUNTING_FIELDS:
            if (
                process_row[field] != recomputed[field]
                or snapshot_row[field] != recomputed[field]
            ):
                raise ValueError(
                    f"tensor accounting evidence recomputed {field} mismatch"
                )


def validate_nested_evidence_bundle(
    manifests: object,
    file_inventory: object,
    artifact_manifest: object,
) -> None:
    _validate_closed_mapping(
        manifests,
        NESTED_MANIFEST_KINDS,
        "nested evidence manifests",
    )
    for kind in NESTED_MANIFEST_KINDS:
        _validate_nested_manifest(kind, manifests[kind])
    _validate_prerequisite_manifest(manifests["prerequisites"])
    _validate_token_manifest(manifests["tokens"])
    _validate_logit_manifest(manifests["logits"])
    _validate_log_manifest(manifests["logs"])
    _validate_snapshot_manifests(
        manifests["snapshots"],
        manifests["tensor_inventories"],
    )

    if not isinstance(file_inventory, list):
        raise ValueError("nested file inventory is invalid")
    inventory_paths = []
    for row in file_inventory:
        _validate_nested_file(row)
        inventory_paths.append(row["path"])
    if (
        inventory_paths != sorted(inventory_paths)
        or len(inventory_paths) != len(set(inventory_paths))
    ):
        raise ValueError("nested file inventory is invalid")
    expected_inventory = sorted(
        [
            row
            for kind in NESTED_MANIFEST_KINDS
            for row in manifests[kind]["files"]
        ],
        key=lambda row: row["path"],
    )
    if file_inventory != expected_inventory:
        raise ValueError("nested file inventory is invalid")

    validate_artifact_manifest(artifact_manifest)
    entries = {
        row["path"]: row for row in artifact_manifest["entries"]
    }
    for kind, path in NESTED_MANIFEST_ARTIFACT_PATHS.items():
        expected_sha = canonical_json_file_sha256(manifests[kind])
        expected_bytes = len(canonical_json_bytes(manifests[kind])) + 1
        entry = entries[path]
        if (
            entry["sha256"] != expected_sha
            or entry["bytes"] != expected_bytes
        ):
            raise ValueError("nested manifest hash binding is invalid")


def validate_artifact_evidence(
    case_rows: object,
    process_rows: object,
    manifests: object,
    file_inventory: object,
    artifact_manifest: object,
) -> None:
    validate_case_rows(case_rows)
    validate_process_rows(process_rows)
    validate_case_process_row_bindings(case_rows, process_rows)
    validate_case_row_nested_evidence_bindings(
        case_rows,
        manifests,
    )
    validate_process_row_nested_worker_log_bindings(
        process_rows,
        manifests,
    )
    validate_process_row_nested_snapshot_bindings(
        process_rows,
        manifests,
    )
    validate_nested_evidence_bundle(
        manifests,
        file_inventory,
        artifact_manifest,
    )


def validate_artifact_manifest(manifest: object) -> None:
    fields = (
        "schema_version",
        "hash_domain",
        "entries",
        "excluded_verifier_outputs",
    )
    _validate_closed_mapping(manifest, fields, "artifact manifest")
    if manifest["schema_version"] != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("schema version is invalid")
    if manifest["hash_domain"] != list(ARTIFACT_MANIFEST_HASH_DOMAIN):
        raise ValueError("hash domain is invalid")
    if manifest["excluded_verifier_outputs"] != list(
        VERIFIER_TRUST_DOMAIN
    ):
        raise ValueError("verifier trust domain is invalid")
    entries = manifest["entries"]
    if not isinstance(entries, list):
        raise ValueError("entries are invalid")
    if [entry.get("path") for entry in entries] != list(
        ARTIFACT_MANIFEST_HASH_DOMAIN
    ):
        raise ValueError("hash domain entries are invalid")
    for entry in entries:
        validate_manifest_entry(entry)


def classify_run(metrics: object) -> str:
    try:
        _validate_closed_mapping(
            metrics,
            CLASSIFICATION_FIELDS,
            "classification metrics",
        )
    except ValueError:
        return "INVALID_ARTIFACT"
    if any(type(metrics[field]) is not bool for field in CLASSIFICATION_FIELDS):
        return "INVALID_ARTIFACT"
    if metrics["artifact_invalid"]:
        return "INVALID_ARTIFACT"
    if metrics["resources_blocked"]:
        return "BLOCKED_RESOURCES"
    if not metrics["correctness_pass"]:
        return "NO_GO_CORRECTNESS"
    if not metrics["runtime_safety_pass"]:
        return "NO_GO_RUNTIME_SAFETY"
    if not metrics["cache_pass"]:
        return "NO_GO_CACHE"
    if not metrics["performance_pass"]:
        return "NO_GO_PERFORMANCE"
    return "GO"
