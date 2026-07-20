"""Dependency-light contracts for the arrival-load remote runner."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "tools" / "run_arrival_load_gate_remote.sh"
GATE_PATH = REPO_ROOT / "tools" / "arrival_load_gate.py"


def _runner() -> str:
    return RUNNER_PATH.read_text()


def test_remote_runner_has_exact_host_runtime_and_modes():
    runner = _runner()
    for required in (
        "sitian@10.232.195.203",
        "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python",
        "/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B",
        "/data00/home/sitian/sitian-workspace01/tllm/arrival-load-runs/${RUN_TAG}",
        "experiments/arrival_load/${RUN_TAG}",
    ):
        assert required in runner, required
    for mode in (
        "preflight",
        "smoke",
        "cost-calibration",
        "workload-calibration",
        "canonical",
        "download-only",
        "verify-only",
    ):
        assert mode in runner, mode


def test_only_immutable_staging_is_uploaded_before_preflight():
    runner = _runner()
    staging = runner.index('STAGING_DIR="${LOCAL_OUT}.staging"')
    snapshot = runner.index("snapshot-source")
    upload = runner.index('tar -C "${STAGING_DIR}" -cf - .')
    upload_target = runner.index("staging.upload", upload)
    atomic_rename = runner.index(
        "mv '${REMOTE_DIR}/staging.upload' '${REMOTE_DIR}/staging'"
    )
    source_preflight = runner.index("source_preflight.json")
    assert staging < snapshot < upload < upload_target
    assert upload_target < atomic_rename < source_preflight
    assert "SSH_STREAM" in runner
    assert "rsync " not in runner


def test_preflight_is_run_local_and_dependency_light():
    runner = _runner()
    assert "export TMPDIR=\"${REMOTE_DIR}/tmp\"" in runner
    assert "source_audit.validate_source_snapshot" in runner
    assert "capability.json" in runner
    for test_file in (
        "tools/test_arrival_load_cost_calibration.py",
        "tools/test_arrival_load_gate.py",
        "tools/test_arrival_load_driver.py",
        "tools/test_arrival_load_verify.py",
        "tools/test_chunked_prefill.py",
    ):
        assert f'"${{REMOTE_PYTHON}}" {test_file}' in runner


def test_model_modes_are_detached_and_publish_atomic_exitcode():
    runner = _runner()
    assert "nohup bash -c" in runner
    assert "runner.log" in runner
    assert "remote_exitcode.tmp" in runner
    assert (
        "mv '${REMOTE_DIR}/artifacts.work/remote_exitcode.tmp' "
        "'${REMOTE_DIR}/artifacts.work/remote_exitcode'"
    ) in runner
    assert (
        ">'${REMOTE_DIR}/artifacts.work/remote_exitcode'"
        not in runner
    )
    launch = runner.index("nohup bash -c")
    poll = runner.index(
        "test -f '${REMOTE_DIR}/artifacts/remote_exitcode'"
    )
    assert launch < poll


def test_failure_downloads_available_artifacts_before_returning():
    runner = _runner()
    read_exit = runner.index(
        'REMOTE_RC="$(cat "${LOCAL_OUT}/remote_exitcode")"'
    )
    failure = runner.index(
        'if [[ "${REMOTE_RC}" -ne 0 ]]; then'
    )
    preserve = runner.index(
        "download_available_artifacts",
        failure,
    )
    assert read_exit < failure < preserve
    assert "preserved at ${LOCAL_OUT}" in runner


def test_download_transport_is_chunked_safe_and_zero_byte_capable():
    runner = _runner()
    for required in (
        "DOWNLOAD_BLOCK_BYTES",
        "DOWNLOAD_RETRIES",
        "iflag=fullblock",
        "status=none",
        "block_path",
        "actual_block_bytes",
        "expected_block_bytes",
        'cat "${block_path}" >> "${partial_path}"',
        'if [[ ! -f "${partial_path}" ]]; then',
        ': > "${partial_path}"',
        "while (( offset < remote_size )); do",
    ):
        assert required in runner, required
    assert "find '${REMOTE_DIR}/artifacts' -type f" in runner
    assert "unsafe remote artifact path" in runner
    assert "--append" not in runner


def test_download_only_exits_before_snapshot_upload_or_launch():
    runner = _runner()
    branch = runner.index('if [[ "${MODE}" == download-only ]]')
    explicit_tag = runner.index(
        "download-only requires RUN_TAG",
        branch,
    )
    download = runner.index("download_available_artifacts", explicit_tag)
    early_exit = runner.index("exit 0", download)
    snapshot = runner.index("snapshot-source")
    upload = runner.index('tar -C "${STAGING_DIR}" -cf - .')
    launch = runner.index("nohup bash -c")
    assert branch < explicit_tag < download < early_exit
    assert early_exit < snapshot < upload < launch


def test_verify_only_exits_before_ssh_snapshot_or_launch():
    runner = _runner()
    branch = runner.index('if [[ "${MODE}" == verify-only ]]')
    explicit_tag = runner.index(
        "verify-only requires RUN_TAG",
        branch,
    )
    verifier = runner.index(
        'python3 "${REPO_ROOT}/tools/arrival_load_verify.py"',
        explicit_tag,
    )
    early_exit = runner.index("exit 0", verifier)
    snapshot = runner.index("snapshot-source")
    first_ssh = runner.index('"${SSH[@]}"', snapshot)
    launch = runner.index("nohup bash -c")
    assert branch < explicit_tag < verifier < early_exit
    assert early_exit < snapshot < first_ssh < launch


def test_success_runs_independent_local_verifier_and_checks_exitcode():
    runner = _runner()
    assert 'python3 "${REPO_ROOT}/tools/arrival_load_verify.py"' in runner
    assert '--run-dir "${LOCAL_OUT}"' in runner
    assert (
        '"${LOCAL_OUT}/independent-verify/verify.exitcode"'
        in runner
    )
    assert "artifact_hashes.json" in runner


def test_ports_are_allocated_only_by_python_orchestrator():
    runner = _runner()
    gate_source = GATE_PATH.read_text()
    assert "TINYVLLM_DIST_PORT=" not in runner
    assert "MASTER_PORT=" not in runner
    assert 'environment["TINYVLLM_DIST_PORT"]' in gate_source
    assert 'environment["MASTER_PORT"]' in gate_source
    assert "allocate_port_pair()" in gate_source


def test_canonical_validates_current_source_and_environment_identity():
    runner = _runner()
    canonical = runner.index("run-canonical")
    assert "--source-evidence" in runner[canonical:]
    assert "--environment-evidence" in runner[canonical:]


def test_p5_chain_requires_explicit_predecessor_run_tags():
    runner = _runner()
    cost = runner.index("run-cost-calibration-remote")
    workload = runner.index("run-workload-calibration-remote")
    canonical = runner.index("run-canonical")
    assert "cost-calibration requires SMOKE_RUN_TAG" in runner
    assert "workload-calibration requires SMOKE_RUN_TAG" in runner
    assert (
        "workload-calibration requires COST_CALIBRATION_RUN_TAG"
        in runner
    )
    assert "canonical requires SMOKE_RUN_TAG" in runner
    assert (
        "canonical requires COST_CALIBRATION_RUN_TAG"
        in runner
    )
    assert (
        "canonical requires WORKLOAD_CALIBRATION_RUN_TAG"
        in runner
    )
    assert "--smoke-run-dir" in runner[cost:]
    assert "--smoke-run-dir" in runner[workload:]
    assert "--cost-calibration-run-dir" in runner[workload:]
    assert "--run-tag" in runner[canonical:]
    assert "--smoke-run-dir" in runner[canonical:]
    assert "--cost-calibration-run-dir" in runner[canonical:]
    assert "--workload-calibration-run-dir" in runner[canonical:]


def test_success_verifies_smoke_and_canonical_only():
    runner = _runner()
    assert (
        'if [[ "${MODE}" == canonical || "${MODE}" == smoke ]]'
        in runner
    )
    assert "verify_local_artifacts" in runner


def test_runner_forbids_shared_or_checkout_mutation():
    runner = _runner()
    for forbidden in (
        "pkill",
        "killall",
        "rm -rf /tmp",
        "git checkout",
        "git reset",
        "git clean",
        "git add -A",
        "rsync ",
    ):
        assert forbidden not in runner


def main():
    test_remote_runner_has_exact_host_runtime_and_modes()
    test_only_immutable_staging_is_uploaded_before_preflight()
    test_preflight_is_run_local_and_dependency_light()
    test_model_modes_are_detached_and_publish_atomic_exitcode()
    test_failure_downloads_available_artifacts_before_returning()
    test_download_transport_is_chunked_safe_and_zero_byte_capable()
    test_download_only_exits_before_snapshot_upload_or_launch()
    test_verify_only_exits_before_ssh_snapshot_or_launch()
    test_success_runs_independent_local_verifier_and_checks_exitcode()
    test_ports_are_allocated_only_by_python_orchestrator()
    test_canonical_validates_current_source_and_environment_identity()
    test_p5_chain_requires_explicit_predecessor_run_tags()
    test_success_verifies_smoke_and_canonical_only()
    test_runner_forbids_shared_or_checkout_mutation()
    print("arrival load remote runner tests passed")


if __name__ == "__main__":
    main()
