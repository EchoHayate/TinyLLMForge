"""Dependency-light contract tests for the remote router gate runner."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "tools/run_speculation_router_gate_remote.sh"
GATE_PATH = REPO_ROOT / "tools/speculation_router_gate.py"


def test_remote_runner_contract():
    runner = RUNNER_PATH.read_text()
    gate_source = GATE_PATH.read_text()

    for required in (
        "sitian@10.232.195.203",
        "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python",
        "/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B",
        "snapshot-source",
        "source_evidence.json",
        "source.patch",
        "source_snapshot.tar.gz",
        "source_preflight.json",
        "nohup",
        "remote_exitcode",
        "runner.log",
        "verify --out-dir",
    ):
        assert required in runner, required
    assert "TINYVLLM_DIST_PORT" in gate_source
    assert "MASTER_PORT" in gate_source
    assert "rm -rf /tmp" not in runner
    assert "pkill" not in runner
    assert "killall" not in runner


def test_runner_modes_and_owned_source_boundary():
    runner = RUNNER_PATH.read_text()
    gate_source = GATE_PATH.read_text()

    for mode in (
        "preflight",
        "controlled-smoke",
        "controlled",
        "real-smoke",
        "real",
    ):
        assert mode in runner
    for owned in (
        "tinyvllm",
        "tools/profile_ngram_commit.py",
        "tools/source_audit.py",
        "tools/speculation_router_gate.py",
        "tools/test_speculation_router.py",
        "tools/test_speculation_router_gate.py",
        "tools/native_verifier_oracle.py",
        "tools/test_native_verifier_oracle.py",
        "tools/run_speculation_router_gate_remote.sh",
        "tools/test_run_speculation_router_gate_remote.py",
    ):
        assert f'"{owned}"' in gate_source, owned


def test_resume_cannot_poll_a_stale_exitcode():
    runner = RUNNER_PATH.read_text()
    move_old = runner.index(
        'mv "${REMOTE_DIR}/artifacts" '
        '"${REMOTE_DIR}/artifacts.previous"'
    )
    launch = runner.index("nohup bash -c")
    poll = runner.index(
        "test -f '${REMOTE_DIR}/artifacts/remote_exitcode'"
    )
    assert move_old < launch < poll


def test_remote_exitcode_is_published_atomically():
    runner = RUNNER_PATH.read_text()

    assert "remote_exitcode.tmp" in runner
    assert (
        "mv '${REMOTE_DIR}/artifacts.work/remote_exitcode.tmp' "
        "'${REMOTE_DIR}/artifacts.work/remote_exitcode'"
    ) in runner
    assert (
        ">'${REMOTE_DIR}/artifacts.work/remote_exitcode'"
        not in runner
    )


def test_success_artifacts_are_downloaded_individually():
    runner = RUNNER_PATH.read_text()

    assert "SUCCESS_ARTIFACTS=(" in runner
    assert "for artifact_name in \"${SUCCESS_ARTIFACTS[@]}\"" in runner
    assert (
        '"${REMOTE_HOST}:${REMOTE_DIR}/artifacts/" "${LOCAL_OUT}/"'
        not in runner
    )
    assert "raw_payload_sha256" in runner


def test_nonzero_remote_exit_preserves_available_artifacts():
    runner = RUNNER_PATH.read_text()

    read_exit = runner.index(
        'REMOTE_RC="$(cat "${LOCAL_OUT}/remote_exitcode")"'
    )
    failure_branch = runner.index(
        'if [[ "${REMOTE_RC}" -ne 0 ]]; then'
    )
    preserve = runner.index(
        "download_available_artifacts",
        failure_branch,
    )
    success_download = runner.index(
        'for artifact_name in "${SUCCESS_ARTIFACTS[@]}"'
    )
    assert read_exit < failure_branch < preserve < success_download
    assert "runner.log" in runner
    assert "preserved at ${LOCAL_OUT}" in runner


def test_only_immutable_staging_is_uploaded():
    runner = RUNNER_PATH.read_text()

    snapshot = runner.index("snapshot-source")
    upload = runner.index('tar -C "${STAGING_DIR}" -cf - .')
    preflight = runner.index("REMOTE_PREFLIGHT")
    launch = runner.index("nohup bash -c")
    assert snapshot < upload < preflight < launch
    assert "staging.upload" in runner
    assert "SSH_STREAM" in runner
    assert (
        '"${REPO_ROOT}/" "${REMOTE_HOST}:${REMOTE_DIR}/staging/"'
        not in runner
    )
    assert "rsync " not in runner


def test_preflight_download_uses_chunk_transport():
    runner = RUNNER_PATH.read_text()

    assert '"${REMOTE_DIR}/preflight-artifacts/capability.json"' in runner
    assert '"${LOCAL_OUT}/capability.json"' in runner
    assert (
        '"${REMOTE_DIR}/preflight-artifacts/source_preflight.json"'
        in runner
    )
    assert '"${LOCAL_OUT}/source_preflight.json"' in runner


def test_ssh_chunk_download_recovers_from_transport_disconnects():
    runner = RUNNER_PATH.read_text()

    assert "ControlMaster=auto" in runner
    assert "ControlPersist=600" in runner
    assert "DOWNLOAD_BLOCK_BYTES" in runner
    assert "DOWNLOAD_RETRIES" in runner
    assert "iflag=fullblock" in runner
    assert "status=none" in runner
    assert "block_path" in runner
    assert "actual_block_bytes" in runner
    assert "expected_block_bytes" in runner
    assert "cat \"${block_path}\" >> \"${partial_path}\"" in runner
    assert "aligned_size=$((\n" in runner
    assert "--append" not in runner


def test_raw_payload_download_uses_canonical_json_hash():
    runner = RUNNER_PATH.read_text()

    assert "canonical_raw_sha256" in runner
    assert "sort_keys=True" in runner
    assert 'separators=(\",\", \":\")' in runner
    assert (
        'shasum -a 256 "${LOCAL_OUT}/${raw_name}"'
        not in runner
    )


def test_transport_never_consumes_download_manifest_stdin():
    runner = RUNNER_PATH.read_text()

    assert "  -n\n  -o BatchMode=yes" in runner
    stream = runner[
        runner.index("SSH_STREAM=("):
        runner.index("SUCCESS_ARTIFACTS=(")
    ]
    assert "\n  -n\n" not in stream


def main():
    test_remote_runner_contract()
    test_runner_modes_and_owned_source_boundary()
    test_resume_cannot_poll_a_stale_exitcode()
    test_remote_exitcode_is_published_atomically()
    test_success_artifacts_are_downloaded_individually()
    test_nonzero_remote_exit_preserves_available_artifacts()
    test_only_immutable_staging_is_uploaded()
    test_preflight_download_uses_chunk_transport()
    test_ssh_chunk_download_recovers_from_transport_disconnects()
    test_raw_payload_download_uses_canonical_json_hash()
    test_transport_never_consumes_download_manifest_stdin()
    print("speculation router remote runner tests passed")


if __name__ == "__main__":
    main()
