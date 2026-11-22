import hashlib
import re

from pilot_protocol import FLAGS, fields, require
from recalibration_data_contract import FILE_PINS, _plain
from recalibration_validation import WORKER_SHA256, build_batches
from validation_contract import digest, encoded


BACKEND_SHA256 = "9be98748dfd2bdcdd132c6ed4c04c3dd67f11d49d6049fcbe7960bde622690da"
SOURCE_PINS = {
    "pilot_protocol.py": "5074980a2b5b9d8ec0a52288d7cb62cbc847688f094cd8ba189a7135ba315c97",
    "validation_contract.py": "a4d6de138fa4817b23fc6e28442785b0d68a6b8f7360fb110582752d3463299d",
    "validation_report.py": "2e6bfdd61b60a08afdac436e2a2d83cc0b6878914a8d1aeb3a2a9a1da2e28a1e",
    "validation_real_report.py": "55d7de49c81236e60712bf0f578a152f9807f758feaca99a09b15e1d4579909c",
    "recalibration_data_contract.py": "52695a9e2fb7e90b04f4fc7c95dcdc8b0310bbb0c269e6779890b9dff989ed02",
    "recalibration_validation.py": "a65fe7813b9ffd9db20dd232350737a528716d03583255fa907706be6f9076ad",
}
MISSING_BINDINGS = (
    "complete_execution_sources", "live_preflight", "launch_configuration", "one_time_scope",
)
FALSE_FLAGS = (
    *FLAGS, "launch_ready", "run_authorized", "dispatch_authorized",
    "execution_source_manifest_complete", "execution_provenance_verified",
    "trusted_collection_accepted", "dynamic_checks_passed", "full_dynamic_validation_complete",
)


def _pinned_bytes(raw, expected):
    require(type(raw) is bytes and len(raw) <= 1048576, "source bytes")
    require(hashlib.sha256(raw).hexdigest() == expected, "source identity")


def build_preparation(raw_files, worker_raw, backend_raw, source_raw, *, stage_id, run_ids):
    require(type(stage_id) is str and re.fullmatch("[0-9a-f]{32}", stage_id) is not None,
            "stage ID")
    fields(source_raw, SOURCE_PINS)
    require(all(type(name) is str for name in source_raw), "source key type")
    for name, expected in SOURCE_PINS.items():
        _pinned_bytes(source_raw[name], expected)
    _pinned_bytes(backend_raw, BACKEND_SHA256)
    batches = build_batches(raw_files, worker_raw, run_ids)
    require(stage_id not in run_ids, "stage and batch IDs must differ")
    manifest = {
        "kind": "LORA_RECALIBRATION_EXECUTION_PREPARATION", "version": 1,
        "data_version": "recalibration_v2_r1", "status": "PREPARATION_ONLY",
        "stage_id": stage_id, "run_ids": list(run_ids),
        "file_sha256": dict(FILE_PINS), "dependency_source_sha256": dict(SOURCE_PINS),
        "backend_sha256": BACKEND_SHA256, "worker_sha256": WORKER_SHA256,
        "plan_sha256": batches[0]["plan_sha256"],
        "counts": {"scheduled": 480, "batches": 15, "references": 192, "witnesses": 36},
        "batches": [
            {**{name: value for name, value in batch.items() if name != "bundle_raw"},
             "bundle_bytes": len(batch["bundle_raw"])}
            for batch in batches
        ],
        "missing_bindings": list(MISSING_BINDINGS),
        **{flag: False for flag in FALSE_FLAGS},
    }
    return {
        "public": {**manifest, "preparation_sha256": digest(manifest)},
        "private": {"bundles_hex": [batch["bundle_raw"].hex() for batch in batches]},
    }


def validate_preparation(preparation, raw_files, worker_raw, backend_raw, source_raw, *,
                         stage_id, run_ids):
    _plain(preparation)
    expected = build_preparation(
        raw_files, worker_raw, backend_raw, source_raw, stage_id=stage_id, run_ids=run_ids)
    require(encoded(preparation) == encoded(expected), "preparation reconstruction differs")
