from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load_adapter():
    path = TOOLS / "qwen35_tp4_engine_remote_subprocess_adapter.py"
    spec = importlib.util.spec_from_file_location(
        "qwen35_tp4_engine_remote_subprocess_adapter",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeProcess:
    def __init__(
        self,
        argv,
        *,
        stdin,
        stdout,
        stderr,
        env,
        shell,
        returncode=0,
        stdout_bytes=b"",
        stderr_bytes=b"",
        calls,
    ):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        calls.append({
            "argv": argv,
            "stdin": stdin,
            "stdout": stdout,
            "stderr": stderr,
            "env": env,
            "shell": shell,
        })
        stdout.write(stdout_bytes)
        stdout.flush()
        stderr.write(stderr_bytes)
        stderr.flush()

    def wait(self, timeout=None):
        return self.returncode


def _factory(
    *,
    calls,
    returncode=0,
    stdout_bytes=b"",
    stderr_bytes=b"",
):
    def factory(argv, **kwargs):
        return FakeProcess(
            argv,
            returncode=returncode,
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            calls=calls,
            **kwargs,
        )

    return factory


def test_normal_command_uses_exact_argv_environment_and_shell_false():
    adapter = _load_adapter()
    calls = []
    result = adapter.run_command(
        name="reserve_remote",
        argv=["ssh", "-o", "BatchMode=yes", "sitian@10.232.195.203", "true"],
        stdout_path=None,
        env={"KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian"},
        popen_factory=_factory(
            calls=calls,
            stdout_bytes=b"reserved\n",
            stderr_bytes=b"",
        ),
        base_environment={"PATH": "/usr/bin", "LANG": "C"},
    )
    assert result == {
        "returncode": 0,
        "stdout": "reserved\n",
        "stderr": "",
    }
    assert len(calls) == 1
    assert calls[0]["argv"] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "sitian@10.232.195.203",
        "true",
    ]
    assert calls[0]["stdin"] is adapter.DEVNULL
    assert calls[0]["shell"] is False
    assert calls[0]["env"] == {
        "PATH": "/usr/bin",
        "LANG": "C",
        "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian",
    }


def test_ssh_transport_retries_returncode_255():
    adapter = _load_adapter()
    calls = []
    sleeps = []
    returncodes = iter((255, 0))

    def factory(argv, **kwargs):
        return FakeProcess(
            argv,
            returncode=next(returncodes),
            stdout_bytes=b"ready\n",
            stderr_bytes=b"",
            calls=calls,
            **kwargs,
        )

    result = adapter.run_command(
        name="resource_guard",
        argv=["ssh", "sitian@10.232.195.203", "true"],
        stdout_path=None,
        env={"KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian"},
        popen_factory=factory,
        base_environment={"PATH": "/usr/bin"},
        sleep_fn=sleeps.append,
    )

    assert len(calls) == 2
    assert sleeps == [1.0]
    assert result == {
        "returncode": 0,
        "stdout": "ready\n",
        "stderr": "",
    }


def test_package_download_streams_binary_and_reports_identity():
    adapter = _load_adapter()
    calls = []
    payload = b"\x00authority-tar\xff" * 1024
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "authority.tar"
        result = adapter.run_command(
            name="package_download",
            argv=["ssh", "sitian@10.232.195.203", "tar -cf - authority"],
            stdout_path=output,
            env={
                "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian",
            },
            popen_factory=_factory(
                calls=calls,
                stdout_bytes=payload,
                stderr_bytes=b"package log\n",
            ),
            base_environment={"PATH": "/usr/bin"},
        )
        assert output.read_bytes() == payload
        assert result == {
            "returncode": 0,
            "stdout": "",
            "stderr": "package log\n",
            "output_sha256": hashlib.sha256(payload).hexdigest(),
            "output_size": len(payload),
        }
        assert calls[0]["stdout"].name == str(output)


def test_rejects_unapproved_execution_inputs_before_process_creation():
    adapter = _load_adapter()
    cases = [
        (
            "bad_executable",
            {
            "name": "reserve_remote",
            "argv": ["bash", "-lc", "true"],
            "stdout_path": None,
            "env": {
                "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian",
            },
            },
        ),
        (
            "bad_env",
            {
            "name": "reserve_remote",
            "argv": ["ssh", "sitian@10.232.195.203", "true"],
            "stdout_path": None,
            "env": {
                "KRB5CCNAME": "FILE:/tmp/wrong",
            },
            },
        ),
        (
            "extra_env",
            {
            "name": "reserve_remote",
            "argv": ["ssh", "sitian@10.232.195.203", "true"],
            "stdout_path": None,
            "env": {
                "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian",
                "CUDA_VISIBLE_DEVICES": "0",
            },
            },
        ),
        (
            "package_without_path",
            {
            "name": "package_download",
            "argv": ["ssh", "sitian@10.232.195.203", "tar -cf - authority"],
            "stdout_path": None,
            "env": {
                "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian",
            },
            },
        ),
    ]
    for label, case in cases:
        calls = []
        try:
            adapter.run_command(
                **case,
                popen_factory=_factory(calls=calls),
                base_environment={"PATH": "/usr/bin"},
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"{label} was accepted")
        assert calls == []


def test_rejects_existing_or_empty_package_output():
    adapter = _load_adapter()
    with tempfile.TemporaryDirectory() as temporary:
        existing = Path(temporary) / "existing.tar"
        existing.write_bytes(b"keep")
        calls = []
        try:
            adapter.run_command(
                name="package_download",
                argv=["ssh", "sitian@10.232.195.203", "tar -cf - authority"],
                stdout_path=existing,
                env={
                    "KRB5CCNAME": (
                        "FILE:/Users/bytedance/krb5cc_sitian"
                    ),
                },
                popen_factory=_factory(calls=calls),
                base_environment={"PATH": "/usr/bin"},
            )
        except ValueError as error:
            assert "exists" in str(error)
        else:
            raise AssertionError("existing package output was overwritten")
        assert existing.read_bytes() == b"keep"
        assert calls == []

        empty = Path(temporary) / "empty.tar"
        try:
            adapter.run_command(
                name="package_download",
                argv=["ssh", "sitian@10.232.195.203", "tar -cf - authority"],
                stdout_path=empty,
                env={
                    "KRB5CCNAME": (
                        "FILE:/Users/bytedance/krb5cc_sitian"
                    ),
                },
                popen_factory=_factory(calls=[], stdout_bytes=b""),
                base_environment={"PATH": "/usr/bin"},
            )
        except ValueError as error:
            assert "empty" in str(error)
        else:
            raise AssertionError("empty package output was accepted")
        assert not empty.exists()


def test_rejects_oversized_or_non_utf8_logs_and_removes_failed_package():
    adapter = _load_adapter()
    cases = [
        (b"x" * (adapter.MAX_LOG_BYTES + 1), b"", "bounded"),
        (b"\xff", b"", "UTF-8"),
    ]
    for stdout_bytes, stderr_bytes, expected in cases:
        try:
            adapter.run_command(
                name="local_verify",
                argv=[sys.executable, "-c", "print('ok')"],
                stdout_path=None,
                env={
                    "KRB5CCNAME": (
                        "FILE:/Users/bytedance/krb5cc_sitian"
                    ),
                },
                popen_factory=_factory(
                    calls=[],
                    stdout_bytes=stdout_bytes,
                    stderr_bytes=stderr_bytes,
                ),
                base_environment={"PATH": "/usr/bin"},
            )
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError("invalid logs were accepted")

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "failed.tar"
        result = adapter.run_command(
            name="package_download",
            argv=["ssh", "sitian@10.232.195.203", "tar -cf - authority"],
            stdout_path=output,
            env={
                "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian",
            },
            popen_factory=_factory(
                calls=[],
                returncode=7,
                stdout_bytes=b"partial",
                stderr_bytes=b"failed\n",
            ),
            base_environment={"PATH": "/usr/bin"},
        )
        assert result["returncode"] == 7
        assert result["stderr"] == "failed\n"
        assert not output.exists()
        assert "output_sha256" not in result

    with tempfile.TemporaryDirectory() as temporary:
        for stderr_bytes, expected in [
            (b"\xff", "UTF-8"),
            (b"x" * (adapter.MAX_LOG_BYTES + 1), "bounded"),
        ]:
            output = Path(temporary) / f"{expected}.tar"
            try:
                adapter.run_command(
                    name="package_download",
                    argv=[
                        "ssh",
                        "sitian@10.232.195.203",
                        "tar -cf - authority",
                    ],
                    stdout_path=output,
                    env={
                        "KRB5CCNAME": (
                            "FILE:/Users/bytedance/krb5cc_sitian"
                        ),
                    },
                    popen_factory=_factory(
                        calls=[],
                        stdout_bytes=b"partial",
                        stderr_bytes=stderr_bytes,
                    ),
                    base_environment={"PATH": "/usr/bin"},
                )
            except ValueError as error:
                assert expected in str(error)
            else:
                raise AssertionError("invalid package log was accepted")
            assert not output.exists()


def test_process_creation_failure_removes_new_package_output():
    adapter = _load_adapter()

    def fail_to_start(argv, **kwargs):
        kwargs["stdout"].write(b"partial")
        kwargs["stdout"].flush()
        raise OSError("process start failed")

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "authority.tar"
        try:
            adapter.run_command(
                name="package_download",
                argv=["ssh", "sitian@10.232.195.203", "tar -cf - authority"],
                stdout_path=output,
                env={
                    "KRB5CCNAME": (
                        "FILE:/Users/bytedance/krb5cc_sitian"
                    ),
                },
                popen_factory=fail_to_start,
                base_environment={"PATH": "/usr/bin"},
            )
        except OSError as error:
            assert "process start failed" in str(error)
        else:
            raise AssertionError("process creation failure was hidden")
        assert not output.exists()


def test_guarded_authority_timeout_returns_bounded_failure_and_reaps_local_client():
    adapter = _load_adapter()
    calls = []

    class TimedOutProcess(FakeProcess):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.wait_timeouts = []
            self.terminated = False

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            if not self.terminated:
                raise adapter.subprocess.TimeoutExpired(
                    self._argv,
                    timeout,
                )
            return 143

        def terminate(self):
            self.terminated = True

    processes = []

    def factory(argv, **kwargs):
        process = TimedOutProcess(
            argv,
            returncode=0,
            stdout_bytes=b"",
            stderr_bytes=b"compiling\n",
            calls=calls,
            **kwargs,
        )
        process._argv = argv
        processes.append(process)
        return process

    result = adapter.run_command(
        name="guarded_authority",
        argv=["ssh", "sitian@10.232.195.203", "run-authority"],
        stdout_path=None,
        env={"KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian"},
        popen_factory=factory,
        base_environment={"PATH": "/usr/bin"},
        command_timeout_s=12.5,
    )

    assert result == {
        "returncode": 124,
        "stdout": "",
        "stderr": "compiling\ncommand timed out after 12.5 seconds\n",
    }
    assert len(processes) == 1
    assert processes[0].terminated is True
    assert processes[0].wait_timeouts == [12.5, 5.0]


def test_guarded_authority_uses_default_wall_clock_timeout():
    adapter = _load_adapter()
    calls = []
    process = None

    class RecordingProcess(FakeProcess):
        def wait(self, timeout=None):
            self.timeout = timeout
            return self.returncode

    def factory(argv, **kwargs):
        nonlocal process
        process = RecordingProcess(
            argv,
            returncode=0,
            stdout_bytes=b"done\n",
            stderr_bytes=b"",
            calls=calls,
            **kwargs,
        )
        return process

    result = adapter.run_command(
        name="guarded_authority",
        argv=["ssh", "sitian@10.232.195.203", "run-authority"],
        stdout_path=None,
        env={"KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian"},
        popen_factory=factory,
        base_environment={"PATH": "/usr/bin"},
    )

    assert result["returncode"] == 0
    assert process.timeout == 3600.0


def test_source_has_no_cli_or_automatic_execution_surface():
    source = (
        TOOLS / "qwen35_tp4_engine_remote_subprocess_adapter.py"
    ).read_text(encoding="utf-8")
    assert "def main(" not in source
    assert "__main__" not in source
    assert "shell=True" not in source
    assert "subprocess.run(" not in source
    assert "subprocess.call(" not in source
    assert "subprocess.check_" not in source
    assert "os.system(" not in source
    assert "os.popen(" not in source


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(
        "qwen35 TP4 Engine remote subprocess adapter tests passed "
        f"({len(tests)} tests)"
    )


if __name__ == "__main__":
    _run()
