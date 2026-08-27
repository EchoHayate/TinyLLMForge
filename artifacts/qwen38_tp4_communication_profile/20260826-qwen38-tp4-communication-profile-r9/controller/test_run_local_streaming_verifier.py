import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("run_local_streaming_verifier.py")
SPEC = importlib.util.spec_from_file_location(
    "run_local_streaming_verifier",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_load_resume_records_requires_matching_index_path_and_hash(tmp_path):
    resume_log = tmp_path / "part-1.stdout"
    resume_log.write_text(
        "\n".join(
            [
                "TRACE_VERIFY_RESUME_PASS 1/2 nsys/P0-r0.sqlite aaa",
                "TRACE_VERIFY_PASS 2/2 nsys/P0-r1.sqlite bbb",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    trace_names = {"nsys/P0-r0.sqlite", "nsys/P0-r1.sqlite"}
    recorded = {
        "nsys/P0-r0.sqlite": "aaa",
        "nsys/P0-r1.sqlite": "bbb",
    }

    assert module._load_resume_records(
        resume_log,
        recorded=recorded,
        ordered_trace_names=sorted(trace_names),
    ) == {
        "nsys/P0-r0.sqlite": "aaa",
        "nsys/P0-r1.sqlite": "bbb",
    }

    resume_log.write_text(
        "TRACE_VERIFY_PASS 1/2 nsys/P0-r0.sqlite wrong\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="resume digest mismatch"):
        module._load_resume_records(
            resume_log,
            recorded=recorded,
            ordered_trace_names=sorted(trace_names),
        )


def test_copy_trace_retries_and_reuses_control_master(tmp_path, monkeypatch):
    counter = tmp_path / "attempts"
    argv_log = tmp_path / "argv"
    fake_rclone = tmp_path / "rclone"
    fake_rclone.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> '{argv_log}'\n"
        f"count=$(cat '{counter}' 2>/dev/null || echo 0)\n"
        "count=$((count + 1))\n"
        f"printf '%s' \"$count\" > '{counter}'\n"
        "[ \"$count\" -ge 3 ] || exit 1\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_rclone.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{__import__('os').environ['PATH']}")

    module._copy_trace(
        ssh_target="sitian@example",
        ssh_control_path="/tmp/test-control",
        remote_trace_root="/remote/nsys",
        relative="nsys/P0-r0.sqlite",
        destination=tmp_path / "P0-r0.sqlite",
        max_attempts=3,
        retry_delay_seconds=0,
    )

    assert counter.read_text(encoding="utf-8") == "3"
    invocations = argv_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 3
    assert all("-S /tmp/test-control" in line for line in invocations)
    assert all("--multi-thread-streams 2" in line for line in invocations)
