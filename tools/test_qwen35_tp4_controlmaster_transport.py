from __future__ import annotations

import importlib.util
from pathlib import Path
import stat
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load_transport():
    path = TOOLS / "qwen35_tp4_controlmaster_transport.py"
    spec = importlib.util.spec_from_file_location(
        "qwen35_tp4_controlmaster_transport",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_injects_existing_control_socket_without_mutating_input():
    transport = _load_transport()
    events = []
    with tempfile.TemporaryDirectory() as temporary:
        socket_path = Path(temporary) / "master.sock"
        socket_path.touch()
        original_stat = Path.stat

        def fake_stat(path):
            result = original_stat(path)
            values = list(result)
            values[0] = stat.S_IFSOCK | 0o600
            return type(result)(values)

        runner = transport.controlmaster_command_runner(
            base_runner=lambda **kwargs: events.append(kwargs) or {
                "returncode": 0,
                "stdout": "",
                "stderr": "",
            },
            control_path=socket_path,
            stat_fn=fake_stat,
        )
        ssh_argv = [
            "ssh",
            "-o",
            "ControlMaster=no",
            "sitian@10.232.195.203",
            "true",
        ]
        scp_argv = [
            "scp",
            "-q",
            "/tmp/source",
            "sitian@10.232.195.203:/tmp/destination",
        ]

        runner(
            name="reserve_remote",
            argv=ssh_argv,
            stdout_path=None,
            env={"KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian"},
        )
        runner(
            name="upload",
            argv=scp_argv,
            stdout_path=None,
            env={"KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian"},
        )

        assert ssh_argv[1:1] == []
        assert events[0]["argv"] == [
            "ssh",
            "-S",
            str(socket_path),
            "-o",
            "ControlMaster=no",
            "sitian@10.232.195.203",
            "true",
        ]
        assert events[1]["argv"] == [
            "scp",
            "-o",
            f"ControlPath={socket_path}",
            "-q",
            "/tmp/source",
            "sitian@10.232.195.203:/tmp/destination",
        ]


def test_runner_replaces_neutral_control_path_none_with_master_socket():
    transport = _load_transport()
    calls = []
    with tempfile.TemporaryDirectory() as temporary:
        socket_path = Path(temporary) / "master.sock"
        socket_path.touch()
        original_stat = Path.stat

        def fake_stat(path):
            result = original_stat(path)
            values = list(result)
            values[0] = stat.S_IFSOCK | 0o600
            return type(result)(values)

        runner = transport.controlmaster_command_runner(
            base_runner=lambda **kwargs: calls.append(kwargs) or {
                "returncode": 0,
                "stdout": "",
                "stderr": "",
            },
            control_path=socket_path,
            stat_fn=fake_stat,
        )
        ssh_argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ControlPath=none",
            "host",
            "true",
        ]
        scp_argv = [
            "scp",
            "-oControlPath=none",
            "source",
            "host:destination",
        ]

        runner(
            name="ssh",
            argv=ssh_argv,
            stdout_path=None,
            env={"KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian"},
        )
        runner(
            name="scp",
            argv=scp_argv,
            stdout_path=None,
            env={"KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian"},
        )

    assert calls[0]["argv"] == [
        "ssh",
        "-S",
        str(socket_path),
        "-o",
        "BatchMode=yes",
        "host",
        "true",
    ]
    assert calls[1]["argv"] == [
        "scp",
        "-o",
        f"ControlPath={socket_path}",
        "source",
        "host:destination",
    ]
    assert "ControlPath=none" in ssh_argv
    assert "-oControlPath=none" in scp_argv


def test_runner_rejects_missing_non_socket_and_duplicate_transport_options():
    transport = _load_transport()
    calls = []
    with tempfile.TemporaryDirectory() as temporary:
        missing = Path(temporary) / "missing.sock"
        for path, expected in [
            (missing, "missing"),
            (Path(temporary) / "regular", "socket"),
        ]:
            if expected == "socket":
                path.write_text("not a socket\n", encoding="utf-8")
            try:
                transport.controlmaster_command_runner(
                    base_runner=lambda **kwargs: calls.append(kwargs),
                    control_path=path,
                )
            except ValueError as error:
                assert expected in str(error)
            else:
                raise AssertionError("invalid control path was accepted")

        socket_path = Path(temporary) / "master.sock"
        socket_path.touch()
        original_stat = Path.stat

        def fake_stat(path):
            result = original_stat(path)
            values = list(result)
            values[0] = stat.S_IFSOCK | 0o600
            return type(result)(values)

        runner = transport.controlmaster_command_runner(
            base_runner=lambda **kwargs: calls.append(kwargs),
            control_path=socket_path,
            stat_fn=fake_stat,
        )
        for argv in [
            ["ssh", "-S", "/tmp/other", "host", "true"],
            ["scp", "-o", "ControlPath=/tmp/other", "a", "host:b"],
        ]:
            try:
                runner(
                    name="transport",
                    argv=argv,
                    stdout_path=None,
                    env={
                        "KRB5CCNAME": (
                            "FILE:/Users/bytedance/krb5cc_sitian"
                        ),
                    },
                )
            except ValueError as error:
                assert "already" in str(error)
            else:
                raise AssertionError("duplicate transport option accepted")
        assert calls == []


def test_runner_passes_local_python_command_through_exactly():
    transport = _load_transport()
    calls = []
    with tempfile.TemporaryDirectory() as temporary:
        socket_path = Path(temporary) / "master.sock"
        socket_path.touch()
        original_stat = Path.stat

        def fake_stat(path):
            result = original_stat(path)
            values = list(result)
            values[0] = stat.S_IFSOCK | 0o600
            return type(result)(values)

        runner = transport.controlmaster_command_runner(
            base_runner=lambda **kwargs: calls.append(kwargs) or {
                "returncode": 0,
                "stdout": "ok\n",
                "stderr": "",
            },
            control_path=socket_path,
            stat_fn=fake_stat,
        )
        argv = [sys.executable, "-c", "print('ok')"]
        result = runner(
            name="local_verify",
            argv=argv,
            stdout_path=None,
            env={"KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian"},
        )

        assert calls[0]["argv"] == argv
        assert result["stdout"] == "ok\n"


def test_runner_applies_exact_named_timeout_without_overriding_explicit_value():
    transport = _load_transport()
    calls = []
    with tempfile.TemporaryDirectory() as temporary:
        socket_path = Path(temporary) / "master.sock"
        socket_path.touch()
        original_stat = Path.stat

        def fake_stat(path):
            result = original_stat(path)
            values = list(result)
            values[0] = stat.S_IFSOCK | 0o600
            return type(result)(values)

        runner = transport.controlmaster_command_runner(
            base_runner=lambda **kwargs: calls.append(kwargs) or {
                "returncode": 0,
                "stdout": "",
                "stderr": "",
            },
            control_path=socket_path,
            stat_fn=fake_stat,
            command_timeouts_s={"workers": 12 * 60 * 60},
        )
        common = {
            "argv": ["ssh", "host", "true"],
            "stdout_path": None,
            "env": {
                "KRB5CCNAME": (
                    "FILE:/Users/bytedance/krb5cc_sitian"
                ),
            },
        }
        runner(name="workers", **common)
        runner(
            name="workers",
            command_timeout_s=123,
            **common,
        )
        runner(name="resource_guard", **common)

        assert calls[0]["command_timeout_s"] == 12 * 60 * 60
        assert calls[1]["command_timeout_s"] == 123
        assert "command_timeout_s" not in calls[2]


def test_runner_rejects_invalid_timeout_policy():
    transport = _load_transport()
    with tempfile.TemporaryDirectory() as temporary:
        socket_path = Path(temporary) / "master.sock"
        socket_path.touch()
        original_stat = Path.stat

        def fake_stat(path):
            result = original_stat(path)
            values = list(result)
            values[0] = stat.S_IFSOCK | 0o600
            return type(result)(values)

        for policy in [
            {"": 10},
            {"workers": 0},
            {"workers": True},
            {1: 10},
        ]:
            try:
                transport.controlmaster_command_runner(
                    base_runner=lambda **kwargs: None,
                    control_path=socket_path,
                    stat_fn=fake_stat,
                    command_timeouts_s=policy,
                )
            except ValueError as error:
                assert "timeout" in str(error)
            else:
                raise AssertionError("invalid timeout policy accepted")


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 ControlMaster transport tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
