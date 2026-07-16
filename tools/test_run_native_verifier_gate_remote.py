"""Dependency-light contract tests for the native verifier remote runner."""

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parent
    / "run_native_verifier_gate_remote.sh"
).read_text()


def test_remote_runner_uses_run_local_tmpdir():
    remote_tmp = "${REMOTE_DIR}/tmp"
    mkdir_line = next(
        line for line in SCRIPT.splitlines()
        if "mkdir -p artifacts logs" in line
    )
    assert f"'{remote_tmp}'" in mkdir_line
    assert SCRIPT.count(f"TMPDIR='{remote_tmp}'") >= 2


def test_remote_runner_uses_scp_control_path_option():
    assert 'SCP_ARGS+=(-o "ControlPath=${CONTROL_SOCKET}")' in SCRIPT
    assert SCRIPT.count('scp "${SCP_ARGS[@]}"') >= 2


def main():
    test_remote_runner_uses_run_local_tmpdir()
    test_remote_runner_uses_scp_control_path_option()
    print("native verifier remote runner tests passed")


if __name__ == "__main__":
    main()
