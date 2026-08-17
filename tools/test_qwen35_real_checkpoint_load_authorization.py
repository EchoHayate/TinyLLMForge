"""Tests for the local real-checkpoint-load authorization gate.

Run: python3 tools/test_qwen35_real_checkpoint_load_authorization.py
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


THIS_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = THIS_DIR / "qwen35_real_checkpoint_load_contract.py"
AUTHORIZATION_PATH = (
    THIS_DIR / "qwen35_real_checkpoint_load_authorization.py"
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.fspath(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load(
    "qwen35_real_checkpoint_load_contract_authorization_test",
    CONTRACT_PATH,
)
authorization = _load(
    "qwen35_real_checkpoint_load_authorization_under_test",
    AUTHORIZATION_PATH,
)


OWNED = (
    "tools/qwen35_real_checkpoint_load_contract.py",
    "tools/qwen35_real_checkpoint_load_worker.py",
)
SOURCE_HASHES = {name: "1" * 64 for name in OWNED}
SOURCE_TREE = hashlib.sha256(
    json.dumps(
        SOURCE_HASHES,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
BRANCH = "feat/adaptive-ngram-speculation"
COMMIT = "3" * 40


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _ready_preflight():
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "status": "READY",
        "checks": {
            "source_identity": True,
            "remote_identity": True,
            "runtime_dependencies": True,
            "model_identity": True,
            "model_files": True,
            "proc_telemetry": True,
            "run_root_space": True,
            "cuda_disabled": True,
            "gpu0_idle": True,
            "payload_zero": True,
        },
        "failure_reasons": [],
        "remote_target": contract.REMOTE_TARGET,
        "remote_python": contract.REMOTE_PYTHON,
        "observed_user": "sitian",
        "observed_hostname": "approved-host",
        "python_version": "3.11.15",
        "packages": {
            "torch": "2.4.1",
            "safetensors": "0.7.0",
            "transformers": "5.8.1",
        },
        "model_repository": contract.MODEL_REPOSITORY,
        "model_revision": contract.MODEL_REVISION,
        "approved_model_manifest_path": (
            contract.APPROVED_MODEL_MANIFEST_PATH
        ),
        "approved_model_dir": contract.APPROVED_MODEL_DIR,
        "cuda_visible_devices": "",
        "cuda_initialized": False,
        "gpu_processes": [],
        "payload_open_count": 0,
        "payload_bytes_read": 0,
        "payload_hashes_recomputed": False,
        "payload_identity_source": "approved_model_manifest",
        "proc_telemetry_available": True,
        "source_tree_sha256": SOURCE_TREE,
        "model_manifest_sha256": (
            contract.APPROVED_MODEL_MANIFEST_SHA256
        ),
        "config_index_header_sha256": "4" * 64,
        "config_sha256": contract.APPROVED_CONFIG_SHA256,
        "index_sha256": contract.APPROVED_INDEX_SHA256,
        "source_file_sha256": SOURCE_HASHES,
        "remote_source_file_sha256": SOURCE_HASHES,
        "shards": [{
            "name": contract.APPROVED_SHARD_NAME,
            "expected_size": contract.APPROVED_SHARD_SIZE,
            "expected_sha256": contract.APPROVED_SHARD_SHA256,
            "observed_size": contract.APPROVED_SHARD_SIZE,
            "inode": 1,
            "device": 2,
            "mode": 33188,
            "resolved_path": (
                f"{contract.APPROVED_MODEL_DIR}/"
                f"{contract.APPROVED_SHARD_NAME}"
            ),
        }],
        "proc_meminfo": {"MemAvailable": 1 << 30},
        "proc_status_fields": {"VmRSS": True, "VmHWM": True},
        "run_root_filesystem": {"type": "ext4"},
        "model_root_filesystem": {"type": "ext4"},
        "free_run_root_bytes": 2 << 30,
        "required_run_root_bytes": 1 << 30,
    }


def _source_manifest():
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "branch": BRANCH,
        "commit": COMMIT,
        "remote_target": contract.REMOTE_TARGET,
        "remote_source_dir": "/remote/source",
        "source_tree_sha256": SOURCE_TREE,
        "local_file_sha256": SOURCE_HASHES,
        "remote_file_sha256": SOURCE_HASHES,
    }


def _write_evidence(root, preflight=None, source_manifest=None):
    run_dir = root / "experiments/qwen35_hybrid_state/auth-test"
    _write_json(
        run_dir / "preflight.json",
        preflight if preflight is not None else _ready_preflight(),
    )
    _write_json(
        run_dir / "source_manifest.json",
        (
            source_manifest
            if source_manifest is not None
            else _source_manifest()
        ),
    )
    return run_dir


def _clean_git_state(**overrides):
    state = {
        "branch": BRANCH,
        "commit": COMMIT,
        "tracked": set(OWNED),
        "staged": set(),
        "unstaged": set(),
        "untracked": set(),
    }
    state.update(overrides)
    return state


def _authorize(
    run_dir,
    *,
    current_hashes=None,
    git_state=None,
):
    return authorization.authorize_run(
        run_dir,
        owned_source_files=OWNED,
        current_hash_function=lambda _root, _owned: (
            dict(SOURCE_HASHES)
            if current_hashes is None
            else dict(current_hashes)
        ),
        git_state_function=lambda _root, _owned: (
            _clean_git_state()
            if git_state is None
            else git_state
        ),
    )


def test_ready_clean_exact_source_is_authorized():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = _write_evidence(Path(temporary))
        result = _authorize(run_dir)
        assert result["decision"] == "AUTHORIZED"
        assert result["worker_implementation_authorized"] is True
        assert result["worker_execution_authorized"] is False
        assert result["reasons"] == []


def test_incomplete_or_stale_source_is_blocked():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        preflight = _ready_preflight()
        preflight["status"] = "INCOMPLETE"
        preflight["checks"]["gpu0_idle"] = False
        preflight["failure_reasons"] = [
            "GPU0 has active compute processes"
        ]
        preflight["gpu_processes"] = ["123, python3, 100"]
        incomplete = _authorize(
            _write_evidence(root / "incomplete", preflight=preflight)
        )
        assert incomplete["decision"] == "BLOCKED"
        assert "preflight is not READY" in incomplete["reasons"]

        stale_hashes = dict(SOURCE_HASHES)
        stale_hashes[OWNED[0]] = "f" * 64
        stale = _authorize(
            _write_evidence(root / "stale"),
            current_hashes=stale_hashes,
        )
        assert stale["decision"] == "BLOCKED"
        assert "current owned source hashes differ" in stale["reasons"]


def test_provenance_and_git_cleanliness_fail_closed():
    cases = (
        (
            _clean_git_state(branch="other"),
            "current branch differs",
        ),
        (
            _clean_git_state(commit="4" * 40),
            "current commit differs",
        ),
        (
            _clean_git_state(tracked={OWNED[0]}),
            "owned source is not tracked",
        ),
        (
            _clean_git_state(staged={OWNED[0]}),
            "owned source has staged changes",
        ),
        (
            _clean_git_state(unstaged={OWNED[0]}),
            "owned source has unstaged changes",
        ),
        (
            _clean_git_state(untracked={OWNED[0]}),
            "owned source is untracked",
        ),
    )
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        for index, (git_state, reason) in enumerate(cases):
            result = _authorize(
                _write_evidence(base / str(index)),
                git_state=git_state,
            )
            assert result["decision"] == "BLOCKED"
            assert reason in result["reasons"]


def test_source_map_disagreement_and_missing_artifacts_are_blocked():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = _source_manifest()
        source["remote_file_sha256"] = dict(SOURCE_HASHES)
        source["remote_file_sha256"][OWNED[0]] = "e" * 64
        mismatch = _authorize(
            _write_evidence(root / "mismatch", source_manifest=source)
        )
        assert mismatch["decision"] == "BLOCKED"
        assert "source hash maps disagree" in mismatch["reasons"]

        missing = authorization.authorize_run(
            root / "missing",
            owned_source_files=OWNED,
            current_hash_function=lambda *_args: dict(SOURCE_HASHES),
            git_state_function=lambda *_args: _clean_git_state(),
        )
        assert missing["decision"] == "BLOCKED"
        assert any(
            "preflight.json" in reason for reason in missing["reasons"]
        )


def test_malformed_preflight_and_source_manifest_are_blocked():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        malformed = _write_evidence(root / "malformed")
        (malformed / "preflight.json").write_text(
            "{not-json\n",
            encoding="utf-8",
        )
        result = _authorize(malformed)
        assert result["decision"] == "BLOCKED"
        assert any(
            "invalid preflight.json" in reason
            for reason in result["reasons"]
        )

        invalid_source = _source_manifest()
        invalid_source["schema_version"] = "wrong"
        result = _authorize(
            _write_evidence(
                root / "invalid-source",
                source_manifest=invalid_source,
            )
        )
        assert result["decision"] == "BLOCKED"
        assert "source manifest schema mismatch" in result["reasons"]


if __name__ == "__main__":
    test_ready_clean_exact_source_is_authorized()
    test_incomplete_or_stale_source_is_blocked()
    test_provenance_and_git_cleanliness_fail_closed()
    test_source_map_disagreement_and_missing_artifacts_are_blocked()
    test_malformed_preflight_and_source_manifest_are_blocked()
    print("qwen35 real checkpoint load authorization tests passed")
