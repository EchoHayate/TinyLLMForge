import base64
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
import selectors
import shlex
import signal
import subprocess
import time

from tokenizer_host_remote import (
    DIAGNOSTIC_LIMIT, FrameDecoder, WIRE_LIMIT, encoded, private_directory,
    sha256, strict_json, validate_bundle, validate_work_timeout, write_private,
)
from tokenizer_launch_window import LaunchWindow


@dataclass(frozen=True)
class FrozenLaunch:
    argv: tuple
    config_json: bytes
    payload: bytes = field(repr=False)

    def manifest(self):
        return {
            "kind": "FROZEN_EXISTING_HOST_TOKENIZER_LAUNCH",
            "argv": list(self.argv), "config": strict_json(self.config_json),
            "command_sha256": sha256(self.argv[-1].encode("utf-8")),
            "config_sha256": sha256(self.config_json),
            "payload_sha256": sha256(self.payload), "payload_bytes": len(self.payload),
        }


def build_payload(sources, inventory_raw, profile_raw):
    return encoded({
        "sources": {name: base64.b64encode(raw).decode("ascii")
                    for name, raw in sources.items()},
        "inventory": base64.b64encode(inventory_raw).decode("ascii"),
        "profile": base64.b64encode(profile_raw).decode("ascii"),
    })


def build_launch(*, ssh_prefix, interpreter, hostname, uid, site_packages, source,
                 parent, run_id, payload, sources_sha256, inventory_sha256, profile_sha256,
                 work_timeout_seconds=540):
    validate_work_timeout(work_timeout_seconds)
    if (not ssh_prefix or any(type(argument) is not str or not argument or "\0" in argument
                              for argument in ssh_prefix)
            or type(uid) is not int or uid < 0
            or type(hostname) is not str or not re.fullmatch("[a-zA-Z0-9.-]+", hostname)
            or type(run_id) is not str or not re.fullmatch("[0-9a-f]{32}", run_id)):
        raise ValueError("invalid launch configuration")
    for path in (interpreter, site_packages, source, parent):
        if (type(path) is not str or not path.startswith("/") or "\0" in path
                or ".." in Path(path).parts or str(Path(path)) != path):
            raise ValueError("absolute normalized path required")
    for expected in (*sources_sha256.values(), inventory_sha256, profile_sha256):
        if type(expected) is not str or not re.fullmatch("[0-9a-f]{64}", expected):
            raise ValueError("invalid digest")
    bootstrap_raw = Path(__file__).with_name("tokenizer_host_remote.py").read_bytes()
    config = {
        "hostname": hostname, "uid": uid, "interpreter": interpreter,
        "site_packages": site_packages, "source": source, "parent": parent,
        "stage": str(Path(parent) / ("bootstrap-" + run_id)), "run_id": run_id,
        "sources_sha256": dict(sources_sha256), "inventory_sha256": inventory_sha256,
        "profile_sha256": profile_sha256, "payload_sha256": sha256(payload),
        "payload_bytes": len(payload), "bootstrap_sha256": sha256(bootstrap_raw),
        "launcher_sha256": sha256(Path(__file__).read_bytes()),
        "work_timeout_seconds": work_timeout_seconds,
    }
    validate_bundle(payload, config)
    config_raw = encoded(config)
    program = (
        "import base64,hashlib,json,sys\n"
        f"source=base64.b64decode({base64.b64encode(bootstrap_raw).decode('ascii')!r},validate=True)\n"
        f"config=json.loads({config_raw!r})\n"
        "if hashlib.sha256(source).hexdigest()!=config['bootstrap_sha256']:\n"
        "    sys.exit(78)\n"
        "namespace={'__name__':'tokenizer_host_bootstrap'}\n"
        "try:\n"
        "    exec(compile(source,'<tokenizer_host_remote>','exec'),namespace)\n"
        "    result=namespace['bootstrap'](config)\n"
        "except Exception:\n"
        "    sys.stderr.write('TOKENIZER_BOOTSTRAP_FAILED\\n')\n"
        "    result=78\n"
        "sys.exit(result)\n")
    command = "exec " + shlex.join((
        "env", "-u", "LD_PRELOAD", "-u", "LD_LIBRARY_PATH", "-u", "PYTHONHOME",
        "-u", "PYTHONPATH", interpreter, "-I", "-S", "-B", "-c", program))
    return FrozenLaunch(tuple(ssh_prefix) + (command,), config_raw, payload)


def _validate_ready(raw, config):
    ready = strict_json(raw)
    return (
        ready.get("config_sha256") == sha256(encoded(config))
        and ready.get("run_id") == config["run_id"]
        and ready.get("hostname") == config["hostname"]
        and type(ready.get("uid")) is int and ready["uid"] == config["uid"]
        and ready.get("stage") == config["stage"]
        and ready.get("core_limit") == [0, 0]
        and ready.get("protection") == {
            "threads": 1, "fork_denied": True, "thread_denied": True,
            "denied_syscalls": [56, 57, 58, 435], "non_x86_and_x32": "KILL_PROCESS"})


def collect_once(argv, payload, config_json, attempt_directory, *, timeout_seconds=540,
                 ready_validator):
    """One owned subprocess; returned metadata never contains diagnostic or result bodies."""
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 1200):
        raise ValueError("invalid timeout")
    directory = Path(attempt_directory)
    private_directory(directory.parent)
    directory.mkdir(mode=0o700)
    private_directory(directory)
    write_private(directory / "command.json", encoded(list(argv)))
    write_private(directory / "config.json", config_json)
    write_private(directory / "input-binding.json", encoded({
        "payload_sha256": sha256(payload), "payload_bytes": len(payload)}))
    window = LaunchWindow(work_timeout_seconds=math.ceil(timeout_seconds))
    window.claim_connection()
    started = time.monotonic()
    deadline = started + timeout_seconds
    process = None
    parser = FrameDecoder()
    wire = bytearray()
    transport = bytearray()
    pending = None
    sent = 0
    grant_size = 0
    ready_validated = False
    failure = None
    selector = selectors.DefaultSelector()
    try:
        process = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, bufsize=0, close_fds=True)
        for stream in (process.stdin, process.stdout, process.stderr):
            os.set_blocking(stream.fileno(), False)
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            remaining = min(deadline - time.monotonic(), window.work_remaining_ns() / 1e9)
            if remaining <= 0:
                failure = "TIMEOUT"
                raise TimeoutError()
            for key, mask in selector.select(min(remaining, 0.05)):
                if key.data == "stdin":
                    written = os.write(key.fd, pending[sent:sent + 65536])
                    sent += written
                    if sent == len(pending):
                        selector.unregister(process.stdin)
                        process.stdin.close()
                    continue
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                output = wire if key.data == "stdout" else transport
                limit = WIRE_LIMIT if key.data == "stdout" else DIAGNOSTIC_LIMIT
                available = limit - len(output)
                output.extend(chunk[:available])
                if len(chunk) > available:
                    failure = "TRANSPORT_OVERFLOW"
                    raise ValueError()
                if key.data == "stdout":
                    parser.feed(chunk)
                    if "RESULT" in parser.frames and (pending is None or sent != len(pending)):
                        failure = "RESULT_BEFORE_INPUT"
                        raise ValueError()
                    if not ready_validated and "READY" in parser.frames:
                        if ready_validator(parser.frames["READY"]) is not True:
                            failure = "READY_MISMATCH"
                            raise ValueError()
                        if set(parser.frames) != {"READY"}:
                            raise ValueError("result before input")
                        window.claim_audit()
                        remaining_ns = min(
                            window.work_remaining_ns(),
                            int((deadline - time.monotonic()) * 1_000_000_000))
                        remaining_ns -= 1_000_000_000
                        if remaining_ns <= 0:
                            failure = "TIMEOUT"
                            raise TimeoutError()
                        grant = encoded({"remaining_ns": remaining_ns}) + b"\n"
                        grant_size = len(grant)
                        pending = grant + payload
                        ready_validated = True
                        selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
            if process.poll() is not None and not ready_validated:
                raise ValueError("exit before READY")
        frames = parser.finish()
        if not ready_validated or pending is None or sent != len(pending):
            raise ValueError("incomplete input")
        remaining = min(deadline - time.monotonic(), window.work_remaining_ns() / 1e9)
        if remaining <= 0:
            failure = "TIMEOUT"
            raise TimeoutError()
        process.wait(timeout=remaining)
        if process.returncode != 0:
            failure = "NONZERO_EXIT"
    except BaseException as error:
        failure = failure or (
            "PROTOCOL_OR_PROCESS_ERROR" if isinstance(error, Exception) else "CANCELLED")
    finally:
        window.begin_cleanup("AUDIT_FAILED" if failure else "AUDIT_COMPLETE")
        if process is not None:
            if failure:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    if process.poll() is None:
                        failure = "LOCAL_GROUP_SIGNAL_FAILED"
                        process.kill()
            try:
                process.wait(timeout=max(0.001, min(5, window.cleanup_remaining_ns() / 1e9)))
            except subprocess.TimeoutExpired:
                failure = "LOCAL_REAP_UNVERIFIED"
            for stream in (process.stdin, process.stdout, process.stderr):
                if not stream.closed:
                    stream.close()
        selector.close()
        window.close()
    write_private(directory / "wire.stdout", bytes(wire))
    write_private(directory / "transport.stderr", bytes(transport))
    for channel, raw in parser.frames.items():
        write_private(directory / (channel.lower() + ".bin"), raw)
    result = {
        "kind": "LOCAL_EXISTING_HOST_TOKENIZER_TRANSPORT",
        "status": "FAILED" if failure else "TRANSPORT_COMPLETE",
        "error": failure, "returncode": process.returncode if process else None,
        "pid": process.pid if process else None,
        "reaped": process is not None and process.returncode is not None,
        "elapsed_seconds": time.monotonic() - started,
        "input_bytes_sent": max(0, sent - grant_size),
        "wire_stdout_bytes": len(wire), "transport_stderr_bytes": len(transport),
        "wire_stdout_sha256": sha256(wire), "transport_stderr_sha256": sha256(transport),
        "payload_sha256": sha256(payload), "config_sha256": sha256(config_json),
        "command_sha256": sha256(encoded(list(argv))), "window": window.snapshot(),
        "remote_state": "UNKNOWN", "audit_pass": False,
    }
    write_private(directory / "transport.json", encoded(result))
    return result


def execute_once(spec, attempt_directory, *, approved_manifest_sha256):
    manifest = spec.manifest()
    if sha256(encoded(manifest)) != approved_manifest_sha256:
        raise ValueError("launch approval mismatch")
    config = strict_json(spec.config_json)
    work_timeout_seconds = validate_work_timeout(config.get("work_timeout_seconds", 540))
    if (config.get("launcher_sha256") != sha256(Path(__file__).read_bytes())
            or config.get("bootstrap_sha256") != sha256(
                Path(__file__).with_name("tokenizer_host_remote.py").read_bytes())):
        raise ValueError("local launch source drift")
    validate_bundle(spec.payload, config)
    result = collect_once(
        spec.argv, spec.payload, spec.config_json, attempt_directory,
        timeout_seconds=work_timeout_seconds,
        ready_validator=lambda raw: _validate_ready(raw, config))
    if result["status"] != "TRANSPORT_COMPLETE":
        return result
    directory = Path(attempt_directory)
    try:
        result_raw = (directory / "result.bin").read_bytes()
        envelope = strict_json(result_raw)
        done = strict_json((directory / "done.bin").read_bytes())
        if (not _validate_ready(encoded(envelope["identity"]), config)
                or envelope["kind"] != "EXISTING_HOST_TOKENIZER_RESULT"
                or envelope["inventory_sha256"] != config["inventory_sha256"]
                or envelope["profile_sha256"] != config["profile_sha256"]
                or done != {"config_sha256": sha256(spec.config_json),
                            "result_sha256": sha256(result_raw)}):
            raise ValueError()
        report = envelope["report"]
        if envelope["bootstrap_failure"] is not None or type(report) is not dict:
            raise ValueError()
        public = report["public"]
        if type(public) is not dict:
            raise ValueError()
        if (public["run_id"] != config["run_id"]
                or public["inventory_sha256"] != config["inventory_sha256"]
                or public["profile_sha256"] != config["profile_sha256"]):
            raise ValueError()
        result["status"] = "AUDIT_EVIDENCE_RETURNED"
        result["audit_failure"] = public["failure"] is not None
        if result["audit_failure"]:
            result["status"] = "AUDIT_FAILED"
        elif public.get("stop_reason") is not None:
            result["status"] = "AUDIT_STOPPED"
    except (KeyError, TypeError, ValueError):
        result["status"] = "FAILED"
        result["error"] = "RESULT_BINDING_OR_BOOTSTRAP_FAILURE"
    write_private(directory / "execution.json", encoded(result))
    return result
