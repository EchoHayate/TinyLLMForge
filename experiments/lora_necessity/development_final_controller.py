import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

from candidate_execution_protocol import InfrastructureError, READY, make_receipt
from development_final_protocol import prepare_request


def _save(path, value):
    with path.open("x") as output:
        json.dump(value, output, sort_keys=True, indent=2, allow_nan=False)


def _terminate(process, timeout):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise InfrastructureError("owned process could not be reaped") from error


def run_request(request, argv, *, artifact_dir, cwd, env,
                startup_timeout=10.0, candidate_timeout=10.0,
                output_limit=65536, result_limit=16384, reap_timeout=2.0):
    request_sha256, payload = prepare_request(request)
    for value in (startup_timeout, candidate_timeout, reap_timeout):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("invalid deadline")
    for value, maximum in ((output_limit, 65536), (result_limit, 16384)):
        if type(value) is not int or not 0 < value <= maximum:
            raise ValueError("invalid byte limit")
    if (type(argv) not in (list, tuple) or not argv
            or any(type(value) is not str or "\0" in value for value in argv)
            or sum(value.count("{control_fd}") for value in argv) != 1):
        raise ValueError("trusted argv requires exactly one control descriptor token")
    if (type(env) is not dict or any(type(name) is not str or type(value) is not str
            or "\0" in name + value or "=" in name for name, value in env.items())):
        raise ValueError("explicit environment required")
    root = Path(artifact_dir)
    if not root.is_dir() or not Path(cwd).is_dir():
        raise ValueError("existing artifact and work directories required")
    _save(root / "claim.json", {
        "request_sha256": request_sha256, "argv": list(argv),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "startup_timeout": startup_timeout, "candidate_timeout": candidate_timeout,
        "output_limit": output_limit, "result_limit": result_limit,
    })
    started = time.monotonic()
    deadline = started + startup_timeout
    process = None
    control_read = control_write = None
    selector = selectors.DefaultSelector()
    buffers = {name: bytearray() for name in ("stdout", "stderr", "control")}
    ready = False
    sent = 0
    total = 0
    reaped = False
    reap_attempted = False
    send_started = None
    eof = False
    termination = None
    failure = None
    ready_size = 0
    readers = set()
    try:
        control_read, control_write = os.pipe()
        actual_argv = [value.replace("{control_fd}", str(control_write)) for value in argv]
        process = subprocess.Popen(
            actual_argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, pass_fds=(control_write,), close_fds=True,
            cwd=cwd, env=env, start_new_session=True, bufsize=0)
        os.close(control_write)
        control_write = None
        for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr"),
                             (control_read, "control")):
            descriptor = stream if type(stream) is int else stream.fileno()
            os.set_blocking(descriptor, False)
            selector.register(stream, selectors.EVENT_READ, name)
            readers.add(descriptor)
        os.set_blocking(process.stdin.fileno(), False)
        while True:
            now = time.monotonic()
            if now >= deadline:
                if send_started is None:
                    raise InfrastructureError("initialization deadline exceeded")
                termination = "timeout"
                break
            if not readers and process.poll() is not None:
                eof = True
                break
            events = selector.select(min(0.05, deadline - now))
            for key, mask in events:
                if key.data == "input":
                    if send_started is None:
                        send_started = time.monotonic()
                        deadline = send_started + candidate_timeout
                    try:
                        written = os.write(key.fd, payload[sent:sent + 8192])
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        raise InfrastructureError("request pipe closed before complete input")
                    sent += written
                    if sent == len(payload):
                        selector.unregister(process.stdin)
                        process.stdin.close()
                    continue
                try:
                    chunk = os.read(key.fd, min(8192, output_limit + 1 - total))
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    readers.discard(key.fd)
                    continue
                total += len(chunk)
                buffers[key.data].extend(chunk)
                if total > output_limit:
                    if not ready:
                        raise InfrastructureError("initialization output limit exceeded")
                    termination = "output_limit"
                    break
                if key.data == "control":
                    control = buffers["control"]
                    if not ready:
                        if len(control) > 4096 or not READY.startswith(control):
                            raise InfrastructureError("invalid initialization handshake")
                        if control == READY:
                            ready = True
                            ready_size = len(READY)
                            selector.register(process.stdin, selectors.EVENT_WRITE, "input")
                    elif len(control) - ready_size > result_limit:
                        termination = "output_limit"
                        break
            if termination:
                break
        if termination:
            reap_attempted = True
            _terminate(process, reap_timeout)
        else:
            process.wait(timeout=reap_timeout)
        reaped = True
    except BaseException as error:
        failure = error
    finally:
        if process is not None:
            if not reaped and not reap_attempted:
                try:
                    reap_attempted = True
                    _terminate(process, reap_timeout)
                    reaped = True
                except BaseException as error:
                    failure = error
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        selector.close()
        for descriptor in (control_read, control_write):
            if descriptor is not None:
                os.close(descriptor)
    audit = {
        "request_sha256": request_sha256,
        "ready": ready, "input_complete": sent == len(payload),
        "input_bytes_sent": sent, "reaped": reaped, "eof": eof,
        "returncode": None if process is None else process.returncode,
        "termination": termination, "elapsed_seconds": time.monotonic() - started,
        "send_started_seconds": None if send_started is None else send_started - started,
        "output_bytes_observed": total,
        "failure": None if failure is None else type(failure).__name__ + ": " + str(failure),
        "stream_sha256": {name: hashlib.sha256(raw).hexdigest()
                          for name, raw in buffers.items()},
    }
    for name, raw in buffers.items():
        with (root / (name + ".bin")).open("xb") as output:
            output.write(raw)
    _save(root / "observation.json", audit)
    if failure is not None:
        if not isinstance(failure, Exception):
            raise failure
        raise InfrastructureError("executor infrastructure failure; see observation") from failure
    receipt = make_receipt(
        request_sha256, returncode=audit["returncode"], ready=ready,
        input_complete=audit["input_complete"], reaped=reaped, eof=eof,
        termination=termination, result_raw=bytes(buffers["control"][ready_size:]))
    _save(root / "receipt.json", receipt)
    return {"receipt": receipt, "audit": audit}
