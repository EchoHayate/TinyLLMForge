from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


MAX_LOG_BYTES = 1024 * 1024
DEFAULT_COMMAND_TIMEOUTS = {
    "guarded_authority": 3600.0,
}
DEVNULL = subprocess.DEVNULL
REQUIRED_EXECUTION_ENV = {
    "KRB5CCNAME": "FILE:/Users/bytedance/krb5cc_sitian",
}
ALLOWED_EXECUTABLES = {
    "ssh",
    "scp",
    sys.executable,
}


def _validate_inputs(*, name, argv, stdout_path, env):
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(argv, list)
        or not argv
        or any(not isinstance(value, str) or not value for value in argv)
    ):
        raise ValueError("subprocess command input is invalid")
    if argv[0] not in ALLOWED_EXECUTABLES:
        raise ValueError("subprocess executable is not approved")
    if env != REQUIRED_EXECUTION_ENV:
        raise ValueError("exact Kerberos execution environment is required")
    if name == "package_download":
        if stdout_path is None:
            raise ValueError("package output path is required")
    elif stdout_path is not None:
        raise ValueError("stdout path is reserved for package download")


def _decode_log(path, label):
    with Path(path).open("rb") as handle:
        payload = handle.read(MAX_LOG_BYTES + 1)
    if len(payload) > MAX_LOG_BYTES:
        raise ValueError(f"{label} log is not bounded")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} log is not UTF-8") from error


def _environment(base_environment, execution_env):
    if not isinstance(base_environment, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in base_environment.items()
    ):
        raise ValueError("base environment is invalid")
    merged = dict(base_environment)
    merged.update(execution_env)
    return merged


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(
    *,
    name,
    argv,
    stdout_path,
    env,
    popen_factory=subprocess.Popen,
    base_environment=None,
    sleep_fn=time.sleep,
    command_timeout_s=None,
):
    _validate_inputs(
        name=name,
        argv=argv,
        stdout_path=stdout_path,
        env=env,
    )
    if not callable(popen_factory):
        raise ValueError("Popen factory is invalid")
    if (
        command_timeout_s is not None
        and (
            isinstance(command_timeout_s, bool)
            or not isinstance(command_timeout_s, (int, float))
            or command_timeout_s <= 0
        )
    ):
        raise ValueError("command timeout must be positive")
    if command_timeout_s is None:
        command_timeout_s = DEFAULT_COMMAND_TIMEOUTS.get(name)
    execution_environment = _environment(
        dict(os.environ) if base_environment is None else base_environment,
        env,
    )
    package_path = (
        Path(stdout_path) if stdout_path is not None else None
    )
    if package_path is not None:
        if package_path.exists() or package_path.is_symlink():
            raise ValueError("package output already exists")
        package_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        attempts = 3 if argv[0] in {"ssh", "scp"} else 1
        for attempt in range(attempts):
            with tempfile.NamedTemporaryFile(mode="w+b") as normal_stdout:
                with tempfile.NamedTemporaryFile(mode="w+b") as stderr:
                    output_handle = normal_stdout
                    if package_path is not None:
                        output_handle = package_path.open("xb")
                    try:
                        process = popen_factory(
                            argv,
                            stdin=DEVNULL,
                            stdout=output_handle,
                            stderr=stderr,
                            env=execution_environment,
                            shell=False,
                        )
                        try:
                            returncode = process.wait(
                                timeout=command_timeout_s
                            )
                        except subprocess.TimeoutExpired:
                            process.terminate()
                            try:
                                returncode = process.wait(timeout=5.0)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                                returncode = 137
                            stderr.seek(0, os.SEEK_END)
                            stderr.write(
                                (
                                    "command timed out after "
                                    f"{command_timeout_s:g} seconds\n"
                                ).encode("utf-8")
                            )
                            stderr.flush()
                            returncode = 124
                    finally:
                        if package_path is not None:
                            output_handle.close()

                    stderr_text = _decode_log(stderr.name, "stderr")
                    if package_path is None:
                        stdout_text = _decode_log(
                            normal_stdout.name,
                            "stdout",
                        )
                        if returncode == 255 and attempt + 1 < attempts:
                            sleep_fn(1.0)
                            continue
                        return {
                            "returncode": returncode,
                            "stdout": stdout_text,
                            "stderr": stderr_text,
                        }
                    if returncode != 0:
                        package_path.unlink(missing_ok=True)
                        if returncode == 255 and attempt + 1 < attempts:
                            sleep_fn(1.0)
                            continue
                        return {
                            "returncode": returncode,
                            "stdout": "",
                            "stderr": stderr_text,
                        }
                    size = package_path.stat().st_size
                    if size <= 0:
                        raise ValueError("package output is empty")
                    return {
                        "returncode": returncode,
                        "stdout": "",
                        "stderr": stderr_text,
                        "output_sha256": _sha256(package_path),
                        "output_size": size,
                    }
    except BaseException:
        if package_path is not None:
            package_path.unlink(missing_ok=True)
        raise
