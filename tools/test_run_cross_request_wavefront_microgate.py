from __future__ import annotations

from pathlib import Path, PurePosixPath
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.run_cross_request_wavefront_microgate import (
    APPROVED_REMOTE_ROOT,
    build_attempt_plan,
    build_remote_worker_commands,
    run_attempt,
    run_ssh_with_retry,
)


def _gpu(index, *, memory=0, utilization=0, processes=()):
    return {
        "gpu_index": index,
        "gpu_uuid": f"GPU-{index}",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": list(processes),
    }


def _plan(**overrides):
    arguments = {
        "attempt_tag": "20260831-cross-request-wavefront-stage0-r1",
        "source_revision": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "selected_gpus": [_gpu(index) for index in range(4)],
        "remote_path_state": {
            "attempt_exists": False,
            "attempt_parent_is_symlink": False,
            "remote_root_is_symlink": False,
        },
    }
    arguments.update(overrides)
    return build_attempt_plan(**arguments)


def _absolute_paths(value):
    paths = []
    if isinstance(value, dict):
        for child in value.values():
            paths.extend(_absolute_paths(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            paths.extend(_absolute_paths(child))
    elif isinstance(value, str) and value.startswith("/"):
        paths.append(value)
    return paths


def test_plan_keeps_every_remote_path_attempt_local():
    plan = _plan()
    attempt_root = PurePosixPath(plan["attempt_root"])

    assert all(
        PurePosixPath(path).is_relative_to(
            PurePosixPath(APPROVED_REMOTE_ROOT)
        )
        for path in _absolute_paths(plan)
    )
    for value in plan["environment"].values():
        assert PurePosixPath(value).is_relative_to(attempt_root)


@pytest.mark.parametrize(
    "remote_path_state",
    (
        {
            "attempt_exists": True,
            "attempt_parent_is_symlink": False,
            "remote_root_is_symlink": False,
        },
        {
            "attempt_exists": False,
            "attempt_parent_is_symlink": True,
            "remote_root_is_symlink": False,
        },
        {
            "attempt_exists": False,
            "attempt_parent_is_symlink": False,
            "remote_root_is_symlink": True,
        },
    ),
)
def test_plan_rejects_existing_or_symlinked_attempt(remote_path_state):
    with pytest.raises(ValueError, match="attempt"):
        _plan(remote_path_state=remote_path_state)


def test_plan_requires_four_strict_clean_gpus():
    with pytest.raises(ValueError, match="four strict-clean GPUs"):
        _plan(
            selected_gpus=[
                _gpu(0, processes=[{"pid": 9}]),
                _gpu(1),
                _gpu(2),
                _gpu(3),
            ]
        )


def test_plan_contains_no_auth_or_signal_commands():
    encoded = repr(_plan()).lower()

    for forbidden in ("kinit", "krenew", "kill", "pkill", "killall"):
        assert forbidden not in encoded


def test_remote_worker_commands_launch_exactly_four_ranks():
    plan = _plan()

    commands = build_remote_worker_commands(
        plan,
        python_path="/data00/home/sitian/tllm/env/bin/python",
        dist_port=29673,
    )

    assert len(commands) == 4
    assert [
        command[command.index("--rank") + 1] for command in commands
    ] == ["0", "1", "2", "3"]
    assert all(
        command[command.index("--world-size") + 1] == "4"
        for command in commands
    )
    assert all(
        command[command.index("--source-tree-sha256") + 1] == "b" * 64
        for command in commands
    )
    assert all(plan["raw_root"] in command for command in commands)


def test_remote_worker_commands_reject_python_outside_mounted_home():
    with pytest.raises(ValueError, match="remote worker configuration"):
        build_remote_worker_commands(_plan(), python_path="/usr/bin/python3")


def test_controller_supports_direct_script_execution():
    script = Path(__file__).with_name(
        "run_cross_request_wavefront_microgate.py"
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_expired_kerberos_stops_before_any_remote_access():
    events = []

    result = run_attempt(
        _plan(),
        kerberos_probe=lambda: events.append("kerberos")
        or {"classification": "BLOCKED"},
        gpu_probe=lambda: events.append("gpu") or [],
        remote_writer=lambda _plan: events.append("write"),
        worker_runner=lambda _plan: events.append("worker"),
    )

    assert result["classification"] == "BLOCKED_KERBEROS"
    assert result["worker_started"] is False
    assert events == ["kerberos"]


def test_controller_checks_kerberos_then_gpu_twice():
    events = []
    clean = [_gpu(index) for index in range(4)]

    result = run_attempt(
        _plan(),
        kerberos_probe=lambda: events.append("kerberos")
        or {"classification": "PASS"},
        gpu_probe=lambda: events.append("gpu") or clean,
        remote_writer=lambda plan: events.append("write")
        or {"created": True},
        worker_runner=lambda plan: events.append("worker")
        or {"classification": "PASS"},
        assembler=lambda plan: events.append("assemble")
        or {"classification": "GO_WAVEFRONT_MICROGATE"},
        remote_verifier=lambda plan: events.append("remote_verify")
        or {
            "status": "PASS",
            "reconstructed_classification": "GO_WAVEFRONT_MICROGATE",
        },
        downloader=lambda plan: events.append("download")
        or {"downloaded": True},
        local_verifier=lambda plan: events.append("local_verify")
        or {
            "status": "PASS",
            "reconstructed_classification": "GO_WAVEFRONT_MICROGATE",
        },
    )

    assert result["classification"] == "GO_WAVEFRONT_MICROGATE"
    assert events == [
        "kerberos",
        "gpu",
        "write",
        "gpu",
        "worker",
        "assemble",
        "remote_verify",
        "download",
        "local_verify",
    ]


def test_run_attempt_rejects_missing_or_disagreeing_verifier_result():
    clean = [_gpu(index) for index in range(4)]
    common = {
        "kerberos_probe": lambda: {"classification": "PASS"},
        "gpu_probe": lambda: clean,
        "remote_writer": lambda plan: {"created": True},
        "worker_runner": lambda plan: {"classification": "PASS"},
        "assembler": lambda plan: {
            "classification": "GO_WAVEFRONT_MICROGATE"
        },
        "downloader": lambda plan: {"downloaded": True},
    }
    with pytest.raises(RuntimeError, match="producer/verifier result"):
        run_attempt(
            _plan(),
            remote_verifier=lambda plan: {"status": "PASS"},
            local_verifier=lambda plan: None,
            **common,
        )
    with pytest.raises(RuntimeError, match="classification disagreement"):
        run_attempt(
            _plan(),
            remote_verifier=lambda plan: {
                "status": "PASS",
                "reconstructed_classification": (
                    "GO_WAVEFRONT_MICROGATE"
                ),
            },
            local_verifier=lambda plan: {
                "status": "PASS",
                "reconstructed_classification": "NO_GO_PERFORMANCE",
            },
            **common,
        )


def test_ssh_255_retries_only_within_fixed_budget():
    returncodes = iter((255, 255, 0))
    calls = []

    result = run_ssh_with_retry(
        ["ssh", "host", "true"],
        retry_count=2,
        runner=lambda argv, **kwargs: (
            calls.append(list(argv))
            or SimpleNamespace(
                returncode=next(returncodes),
                stdout="",
                stderr="",
            )
        ),
    )

    assert result.returncode == 0
    assert len(calls) == 3

    calls.clear()
    result = run_ssh_with_retry(
        ["ssh", "host", "false"],
        retry_count=5,
        runner=lambda argv, **kwargs: (
            calls.append(list(argv))
            or SimpleNamespace(returncode=7, stdout="", stderr="")
        ),
    )
    assert result.returncode == 7
    assert len(calls) == 1
