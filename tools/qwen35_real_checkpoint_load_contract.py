from __future__ import annotations

import math
import json
import statistics


SCHEMA_VERSION = "qwen35.real-checkpoint-load-safety.v1"
REMOTE_TARGET = "sitian@10.232.195.203"
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
)
MODEL_REPOSITORY = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
APPROVED_MODEL_MANIFEST_PATH = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-hybrid-state-runs/"
    "qwen35-2b-hybrid-acquire-20260723-222004/model_manifest.json"
)
APPROVED_MODEL_DIR = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-hybrid-state-runs/"
    "qwen35-2b-hybrid-acquire-20260723-222004/model"
)
APPROVED_MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0"
)
APPROVED_CONFIG_SHA256 = (
    "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4"
)
APPROVED_INDEX_SHA256 = (
    "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9"
)
APPROVED_SHARD_NAME = (
    "model.safetensors-00001-of-00001.safetensors"
)
APPROVED_SHARD_SIZE = 4548221488
APPROVED_SHARD_SHA256 = (
    "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
)
APPROVED_MODEL_FILES = {
    "config.json": {
        "sha256": APPROVED_CONFIG_SHA256,
        "size": 2908,
    },
    "merges.txt": {
        "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
        "size": 3353259,
    },
    APPROVED_SHARD_NAME: {
        "sha256": APPROVED_SHARD_SHA256,
        "size": APPROVED_SHARD_SIZE,
    },
    "model.safetensors.index.json": {
        "sha256": APPROVED_INDEX_SHA256,
        "size": 64460,
    },
    "tokenizer.json": {
        "sha256": (
            "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a652"
            "2d1b9f64c4bb81cb42"
        ),
        "size": 12807982,
    },
    "tokenizer_config.json": {
        "sha256": (
            "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613"
            "995e06557f843500c"
        ),
        "size": 16709,
    },
    "vocab.json": {
        "sha256": (
            "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebd"
            "e25853284c9dfa003"
        ),
        "size": 6722759,
    },
}
APPROVED_MODEL_MANIFEST = {
    "files": APPROVED_MODEL_FILES,
    "local_path": APPROVED_MODEL_DIR,
    "remote_model_dir": APPROVED_MODEL_DIR,
    "repository": MODEL_REPOSITORY,
    "resolved_revision": MODEL_REVISION,
    "schema_version": 1,
    "total_weight_bytes": APPROVED_SHARD_SIZE,
    "trust_remote_code": False,
}
REQUIRED_PREFLIGHT_PACKAGES = ("torch", "safetensors", "transformers")
PREFLIGHT_ARTIFACT_ALLOWANCE_BYTES = 512 << 20
CASE_ORDER_MIB = (8, 16, 16, 8, 8, 16)
MEASURED_REPEATS_PER_BUDGET = 3
MIN_WALL_TIME_IMPROVEMENT_FRACTION = 0.05
MAX_VMHWM_REGRESSION_BYTES = 16 << 20
DESTINATION_BYTES = 1881935712
REQUIRED_ARTIFACTS = (
    "manifest.json",
    "source_manifest.json",
    "preflight.json",
    "environment.json",
    "model_manifest.json",
    "processes.json",
    "gpu_processes.json",
    "case_rows.jsonl",
    "telemetry.jsonl",
    "summary.json",
    "independent_verification.json",
    "report.md",
    "stdout/worker.log",
    "stderr/worker.log",
)
CLASSIFICATIONS = ("READY", "INCOMPLETE", "NO_GO", "GO")
RESTORE_STATUSES = (
    "RESTORED",
    "ALREADY_PRESENT",
    "INCOMPLETE",
    "CONFLICT",
)


def canonical_approved_model_manifest_bytes():
    return (
        json.dumps(
            APPROVED_MODEL_MANIFEST,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _require_dictionary(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dictionary")
    return value


def _require_non_negative_integer(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_positive_finite(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _require_sha256(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def validate_preflight(value):
    record = _require_dictionary(value, "preflight")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("preflight schema_version is invalid")
    status = record.get("status")
    if status not in ("READY", "INCOMPLETE"):
        raise ValueError("preflight status is invalid")
    checks = _require_dictionary(record.get("checks"), "preflight checks")
    if not checks or any(
        not isinstance(value, bool) for value in checks.values()
    ):
        raise ValueError("preflight checks must be non-empty booleans")
    reasons = record.get("failure_reasons")
    if (
        not isinstance(reasons, list)
        or any(not isinstance(reason, str) or not reason for reason in reasons)
    ):
        raise ValueError("preflight failure reasons are invalid")
    if status == "READY":
        if reasons or not all(checks.values()):
            raise ValueError("READY preflight must pass every check")
    elif not reasons or all(checks.values()):
        raise ValueError("INCOMPLETE preflight must name failed checks")
    if record.get("remote_target") != REMOTE_TARGET:
        raise ValueError("preflight remote target is invalid")
    if record.get("remote_python") != REMOTE_PYTHON:
        raise ValueError("preflight remote Python is invalid")
    if record.get("observed_user") != "sitian":
        raise ValueError("preflight remote user is invalid")
    if not isinstance(record.get("observed_hostname"), str):
        raise ValueError("preflight remote hostname is invalid")
    if not isinstance(record.get("python_version"), str):
        raise ValueError("preflight Python version is invalid")
    packages = _require_dictionary(
        record.get("packages"),
        "preflight packages",
    )
    for name in REQUIRED_PREFLIGHT_PACKAGES:
        version = packages.get(name)
        if version is not None and not isinstance(version, str):
            raise ValueError(f"preflight package {name} is invalid")
    if record.get("model_repository") != MODEL_REPOSITORY:
        raise ValueError("preflight model repository is invalid")
    if record.get("model_revision") != MODEL_REVISION:
        raise ValueError("preflight model revision is invalid")
    if (
        record.get("approved_model_manifest_path")
        != APPROVED_MODEL_MANIFEST_PATH
    ):
        raise ValueError("preflight model manifest path is invalid")
    if record.get("approved_model_dir") != APPROVED_MODEL_DIR:
        raise ValueError("preflight model directory is invalid")
    if record.get("cuda_visible_devices") != "":
        raise ValueError("preflight CUDA_VISIBLE_DEVICES must be empty")
    if record.get("cuda_initialized") is not False:
        raise ValueError("preflight CUDA must remain uninitialized")
    gpu_processes = record.get("gpu_processes")
    if not isinstance(gpu_processes, list):
        raise ValueError("preflight GPU process list is invalid")
    if status == "READY" and gpu_processes:
        raise ValueError("preflight unrelated GPU process list must be empty")
    if _require_non_negative_integer(
        record.get("payload_open_count"),
        "payload_open_count",
    ):
        raise ValueError("preflight payload open count must be zero")
    if _require_non_negative_integer(
        record.get("payload_bytes_read"),
        "payload_bytes_read",
    ):
        raise ValueError("preflight payload bytes read must be zero")
    if record.get("payload_hashes_recomputed") is not False:
        raise ValueError("preflight payload hashes must not be recomputed")
    if (
        record.get("payload_identity_source")
        != "approved_model_manifest"
    ):
        raise ValueError("preflight payload identity source is invalid")
    if (
        status == "READY"
        and record.get("proc_telemetry_available") is not True
    ):
        raise ValueError("preflight /proc telemetry must be available")
    _require_sha256(record.get("source_tree_sha256"), "source_tree_sha256")
    identity_available = checks.get("model_identity") is True
    for name in (
        "model_manifest_sha256",
        "config_index_header_sha256",
        "config_sha256",
        "index_sha256",
    ):
        if not identity_available and record.get(name) is None:
            continue
        _require_sha256(record.get(name), name)
    if (
        status == "READY"
        and record.get("model_manifest_sha256")
        != APPROVED_MODEL_MANIFEST_SHA256
    ):
        raise ValueError("preflight model manifest SHA256 is invalid")
    if (
        status == "READY"
        and record.get("config_sha256") != APPROVED_CONFIG_SHA256
    ):
        raise ValueError("preflight config SHA256 is invalid")
    if (
        status == "READY"
        and record.get("index_sha256") != APPROVED_INDEX_SHA256
    ):
        raise ValueError("preflight index SHA256 is invalid")
    local_hashes = _require_dictionary(
        record.get("source_file_sha256"),
        "source file SHA256",
    )
    remote_hashes = _require_dictionary(
        record.get("remote_source_file_sha256"),
        "remote source file SHA256",
    )
    if status == "READY" and local_hashes != remote_hashes:
        raise ValueError("preflight source SHA256 maps differ")
    for name, digest in local_hashes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("preflight source path is invalid")
        _require_sha256(digest, f"source SHA256 for {name}")
    shards = record.get("shards")
    if not isinstance(shards, list):
        raise ValueError("preflight shard inventory is invalid")
    if not shards and status == "INCOMPLETE" and not checks.get(
        "model_files"
    ):
        shards = []
    elif len(shards) != 1:
        raise ValueError("preflight shard inventory is invalid")
    if not shards:
        shard = None
    else:
        shard = _require_dictionary(shards[0], "preflight shard")
    if shard is not None and status == "READY" and shard.get(
        "name"
    ) != APPROVED_SHARD_NAME:
        raise ValueError("preflight shard name is invalid")
    if (
        shard is not None
        and
        status == "READY"
        and shard.get("expected_size") != APPROVED_SHARD_SIZE
    ):
        raise ValueError("preflight expected shard size is invalid")
    if (
        shard is not None
        and
        status == "READY"
        and shard.get("expected_sha256") != APPROVED_SHARD_SHA256
    ):
        raise ValueError("preflight expected shard SHA256 is invalid")
    if shard is not None:
        _require_non_negative_integer(
            shard.get("observed_size"),
            "shard size",
        )
        for name in ("inode", "device", "mode"):
            _require_non_negative_integer(shard.get(name), f"shard {name}")
        if not isinstance(shard.get("resolved_path"), str):
            raise ValueError("preflight shard resolved path is invalid")
    for name in (
        "proc_meminfo",
        "proc_status_fields",
        "run_root_filesystem",
        "model_root_filesystem",
    ):
        _require_dictionary(record.get(name), name)
    free_bytes = _require_non_negative_integer(
        record.get("free_run_root_bytes"),
        "free_run_root_bytes",
    )
    required_bytes = _require_non_negative_integer(
        record.get("required_run_root_bytes"),
        "required_run_root_bytes",
    )
    if status == "READY" and free_bytes < required_bytes:
        raise ValueError("preflight run root free space is insufficient")
    return record


def validate_restore_record(value):
    record = _require_dictionary(value, "restore record")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("restore schema_version is invalid")
    status = record.get("status")
    if status not in RESTORE_STATUSES:
        raise ValueError("restore status is invalid")
    checks = _require_dictionary(record.get("checks"), "restore checks")
    if not checks or any(
        not isinstance(passed, bool) for passed in checks.values()
    ):
        raise ValueError("restore checks must be non-empty booleans")
    reasons = record.get("failure_reasons")
    if (
        not isinstance(reasons, list)
        or any(not isinstance(reason, str) or not reason for reason in reasons)
    ):
        raise ValueError("restore failure reasons are invalid")
    successful = status in ("RESTORED", "ALREADY_PRESENT")
    if successful and (reasons or not all(checks.values())):
        raise ValueError("successful restore must pass every check")
    if not successful and (not reasons or all(checks.values())):
        raise ValueError("failed restore must name failed checks")
    if record.get("remote_target") != REMOTE_TARGET:
        raise ValueError("restore remote target is invalid")
    if (
        record.get("approved_model_manifest_path")
        != APPROVED_MODEL_MANIFEST_PATH
    ):
        raise ValueError("restore model manifest path is invalid")
    if record.get("approved_model_dir") != APPROVED_MODEL_DIR:
        raise ValueError("restore model directory is invalid")
    if (
        record.get("approved_manifest_sha256")
        != APPROVED_MODEL_MANIFEST_SHA256
    ):
        raise ValueError("restore approved manifest SHA256 is invalid")
    observed = record.get("observed_manifest_sha256")
    if successful:
        if observed != APPROVED_MODEL_MANIFEST_SHA256:
            raise ValueError("restore observed manifest SHA256 is invalid")
    elif status == "CONFLICT" and observed is not None:
        if observed == APPROVED_MODEL_MANIFEST_SHA256:
            raise ValueError("restore conflict digest must differ")
        _require_sha256(observed, "restore conflict SHA256")
    elif observed is not None:
        _require_sha256(observed, "restore observed manifest SHA256")
    files = _require_dictionary(
        record.get("non_payload_files"),
        "restore non-payload files",
    )
    expected_non_payload = {
        name: entry
        for name, entry in APPROVED_MODEL_FILES.items()
        if not name.endswith(".safetensors")
    }
    if successful and files != expected_non_payload:
        raise ValueError("restore non-payload file identity is invalid")
    if record.get("config_model_type") != "qwen3_5":
        raise ValueError("restore config model type is invalid")
    if record.get("index_shard_names") != [APPROVED_SHARD_NAME]:
        raise ValueError("restore index shard names are invalid")
    if record.get("observed_shard_names") != [APPROVED_SHARD_NAME]:
        raise ValueError("restore observed shard names are invalid")
    shard = _require_dictionary(record.get("shard"), "restore shard")
    if shard.get("name") != APPROVED_SHARD_NAME:
        raise ValueError("restore shard name is invalid")
    if shard.get("expected_size") != APPROVED_SHARD_SIZE:
        raise ValueError("restore expected shard size is invalid")
    if shard.get("observed_size") != APPROVED_SHARD_SIZE:
        raise ValueError("restore observed shard size is invalid")
    for name in ("inode", "device", "mode"):
        _require_non_negative_integer(shard.get(name), f"restore shard {name}")
    if _require_non_negative_integer(
        record.get("payload_open_count"),
        "restore payload open count",
    ):
        raise ValueError("restore payload open count must be zero")
    if _require_non_negative_integer(
        record.get("payload_bytes_read"),
        "restore payload bytes read",
    ):
        raise ValueError("restore payload bytes read must be zero")
    if record.get("payload_hashes_recomputed") is not False:
        raise ValueError("restore payload hashes must not be recomputed")
    write_performed = record.get("write_performed")
    if not isinstance(write_performed, bool):
        raise ValueError("restore write flag is invalid")
    if status == "RESTORED" and not write_performed:
        raise ValueError("RESTORED record must report a write")
    if status != "RESTORED" and write_performed:
        raise ValueError("non-RESTORED record must not report a write")
    return record


def _validate_case_row(row, expected_budget_mib, expected_repeat, order):
    row = _require_dictionary(row, "case row")
    budget_bytes = expected_budget_mib << 20
    if row.get("budget_bytes") != budget_bytes:
        raise ValueError("case budget order is invalid")
    if row.get("repeat_index") != expected_repeat:
        raise ValueError("case repeat index is invalid")
    if row.get("order_index") != order:
        raise ValueError("case order index is invalid")
    if row.get("warmup") is not False:
        raise ValueError("measured case row must not be warmup")
    if (
        row.get("tensor_parallel_size") != 2
        or row.get("tensor_parallel_rank") != 0
    ):
        raise ValueError("case tensor parallel context is invalid")
    if row.get("assigned_bindings") != 320:
        raise ValueError("case assigned binding count is invalid")
    if row.get("source_tensors") != 320:
        raise ValueError("case source tensor count is invalid")
    _require_non_negative_integer(row.get("tile_count"), "tile_count")
    if row.get("destination_bytes") != DESTINATION_BYTES:
        raise ValueError("case destination byte count is invalid")
    peak = _require_non_negative_integer(
        row.get("peak_tile_bytes"),
        "peak_tile_bytes",
    )
    if peak > budget_bytes:
        raise ValueError("case peak tile bytes exceed budget")
    actual_digest = _require_sha256(
        row.get("assignment_digest"),
        "assignment_digest",
    )
    expected_digest = _require_sha256(
        row.get("expected_assignment_digest"),
        "expected_assignment_digest",
    )
    if actual_digest != expected_digest:
        raise ValueError("case assignment digest mismatch")
    if row.get("all_handles_closed") is not True:
        raise ValueError("case shard handles must all be closed")
    if row.get("cuda_initialized") is not False:
        raise ValueError("case CUDA must remain uninitialized")
    if _require_non_negative_integer(
        row.get("cuda_allocated_bytes"),
        "cuda_allocated_bytes",
    ):
        raise ValueError("case CUDA allocation must remain zero")
    for name in (
        "wall_seconds",
        "user_cpu_seconds",
        "system_cpu_seconds",
    ):
        _require_positive_finite(row.get(name), name)
    for name in (
        "minor_faults",
        "major_faults",
        "vmrss_bytes",
        "vmhwm_bytes",
        "voluntary_context_switches",
        "involuntary_context_switches",
    ):
        _require_non_negative_integer(row.get(name), name)
    if row.get("returncode") != 0:
        raise ValueError("case process returncode must be zero")
    return row


def classify_case_rows(rows):
    if not isinstance(rows, list) or len(rows) != len(CASE_ORDER_MIB):
        return {
            "classification": "INCOMPLETE",
            "reason": "case row count is incomplete",
            "correctness_passed": False,
            "wall_time_improvement_fraction": None,
            "vmhwm_regression_bytes": None,
        }
    validated = []
    counts = {8: 0, 16: 0}
    try:
        for order, (row, budget_mib) in enumerate(
            zip(rows, CASE_ORDER_MIB)
        ):
            repeat_index = counts[budget_mib]
            counts[budget_mib] += 1
            validated.append(
                _validate_case_row(
                    row,
                    budget_mib,
                    repeat_index,
                    order,
                )
            )
    except (TypeError, ValueError) as error:
        return {
            "classification": "INCOMPLETE",
            "reason": str(error),
            "correctness_passed": False,
            "wall_time_improvement_fraction": None,
            "vmhwm_regression_bytes": None,
        }
    grouped = {
        budget_mib: [
            row for row in validated
            if row["budget_bytes"] == budget_mib << 20
        ]
        for budget_mib in (8, 16)
    }
    if any(
        len(grouped[budget_mib]) != MEASURED_REPEATS_PER_BUDGET
        for budget_mib in grouped
    ):
        return {
            "classification": "INCOMPLETE",
            "reason": "measured repeat coverage is incomplete",
            "correctness_passed": False,
            "wall_time_improvement_fraction": None,
            "vmhwm_regression_bytes": None,
        }
    median_wall = {
        budget_mib: statistics.median(
            row["wall_seconds"] for row in grouped[budget_mib]
        )
        for budget_mib in grouped
    }
    median_vmhwm = {
        budget_mib: statistics.median(
            row["vmhwm_bytes"] for row in grouped[budget_mib]
        )
        for budget_mib in grouped
    }
    improvement = (
        median_wall[8] - median_wall[16]
    ) / median_wall[8]
    vmhwm_regression = int(median_vmhwm[16] - median_vmhwm[8])
    go = (
        improvement >= MIN_WALL_TIME_IMPROVEMENT_FRACTION
        and vmhwm_regression <= MAX_VMHWM_REGRESSION_BYTES
    )
    return {
        "classification": "GO" if go else "NO_GO",
        "reason": (
            None
            if go
            else "performance or resource promotion threshold not met"
        ),
        "correctness_passed": True,
        "wall_time_improvement_fraction": improvement,
        "vmhwm_regression_bytes": vmhwm_regression,
        "median_wall_seconds": median_wall,
        "median_vmhwm_bytes": median_vmhwm,
    }

