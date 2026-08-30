from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.cross_engine_k8_contract import LOCAL_ALLOWLIST
from tools.stream_verify_cross_engine_k8_remote import (
    StreamVerificationConfig,
    stream_verify,
)


def test_script_entrypoint_can_import_tools_package():
    script = Path(__file__).with_name(
        "stream_verify_cross_engine_k8_remote.py"
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


class FakeRemote:
    def __init__(self, files):
        self.files = dict(files)
        self.reads = []

    def list_files(self, _remote_root):
        return {
            name: len(payload)
            for name, payload in self.files.items()
        }

    def read_bytes(self, _remote_root, name):
        self.reads.append(name)
        return self.files[name]


def _remote_files():
    files = {}
    for name in LOCAL_ALLOWLIST:
        if name == "local_verification.json":
            continue
        if name == "remote_verification.json":
            payload = {"valid": True, "recomputed_classification": "INCOMPLETE"}
        else:
            payload = {}
        files[name] = (
            json.dumps(payload, sort_keys=True) + "\n"
        ).encode("utf-8")
    return files


def _config(tmp_path):
    return StreamVerificationConfig(
        remote_run_tag="20260829-cross-engine-k8-qwen3-06b-r1",
        local_root=tmp_path,
        expected_source="a" * 40,
    )


def _verifier(_root, *, expected_source):
    assert expected_source == "a" * 40
    return {
        "valid": True,
        "recomputed_classification": "INCOMPLETE",
    }


def test_streaming_verifier_does_not_retain_non_allowlisted_inputs(tmp_path):
    files = _remote_files()
    files["worker.log"] = b"large log"
    remote = FakeRemote(files)

    result = stream_verify(_config(tmp_path), remote, _verifier)

    assert result["valid"] is True
    assert not (tmp_path / "worker.log").exists()
    assert set(path.name for path in tmp_path.iterdir()) == set(
        LOCAL_ALLOWLIST
    )


def test_streaming_verifier_rejects_fifty_mib_boundary(tmp_path):
    remote = FakeRemote(_remote_files())
    sizes = remote.list_files("")
    sizes["case_rows.jsonl"] = 50 * 1024**2 + 1
    remote.list_files = lambda _root: sizes

    with pytest.raises(RuntimeError, match="LOCAL_STORAGE_HARD_STOP"):
        stream_verify(_config(tmp_path), remote, _verifier)


def test_streaming_verifier_removes_temporary_files_on_failure(tmp_path):
    remote = FakeRemote(_remote_files())

    def failing_verifier(_root, *, expected_source):
        raise ValueError(expected_source)

    with pytest.raises(ValueError):
        stream_verify(_config(tmp_path), remote, failing_verifier)

    assert list(tmp_path.rglob("*.streaming-tmp")) == []


def test_streaming_verifier_requires_remote_and_local_agreement(tmp_path):
    files = _remote_files()
    files["remote_verification.json"] = json.dumps({
        "valid": True,
        "recomputed_classification": "GO_CROSS_ENGINE_ADVANTAGE",
    }).encode("utf-8")
    remote = FakeRemote(files)

    with pytest.raises(RuntimeError, match="VERIFIER_DISAGREEMENT"):
        stream_verify(_config(tmp_path), remote, _verifier)
