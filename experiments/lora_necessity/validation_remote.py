import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import selectors
import signal
import subprocess
import sys
import time


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def collect_process(argv, input_raw, *, timeout_seconds=2, stdout_limit=65536,
                    stderr_limit=4096, pass_fds=(), environment=None,
                    stderr_is_diagnostic=False):
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 120 or type(input_raw) is not bytes
            or type(stderr_is_diagnostic) is not bool
            or any(type(limit) is not int or not 0 < limit <= 16777216
                   for limit in (stdout_limit, stderr_limit))):
        raise ValueError("collector bounds")
    started = time.monotonic_ns()
    deadline = time.monotonic() + timeout_seconds
    stdout = bytearray()
    stderr = bytearray()
    sent = 0
    result = {
        "status": "launch_failed", "returncode": None, "stdout_hex": "",
        "stderr_hex": "", "elapsed_ns": 0, "process_reaped": False,
        "stdin_complete": False,
    }
    try:
        process = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd="/", env=environment, close_fds=True, pass_fds=pass_fds,
            start_new_session=True,
        )
    except OSError:
        result["elapsed_ns"] = time.monotonic_ns() - started
        return result
    status = None
    with selectors.DefaultSelector() as selector:
        for stream, label in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, label)
        if input_raw:
            os.set_blocking(process.stdin.fileno(), False)
            selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        else:
            process.stdin.close()
        try:
            while selector.get_map() or process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status = "timeout"
                    break
                events = selector.select(min(remaining, 0.05))
                for key, _mask in events:
                    if key.data == "stdin":
                        try:
                            sent += os.write(key.fd, input_raw[sent:sent + 4096])
                        except BlockingIOError:
                            continue
                        except BrokenPipeError:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                            continue
                        if sent == len(input_raw):
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                        continue
                    output, limit = ((stdout, stdout_limit) if key.data == "stdout"
                                     else (stderr, stderr_limit))
                    try:
                        chunk = os.read(key.fd, min(4096, limit + 1 - len(output)))
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    else:
                        output.extend(chunk)
                        if len(output) > limit:
                            status = "output_limit"
                            break
                if status is not None:
                    break
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                result["returncode"] = process.wait(timeout=2)
                result["process_reaped"] = True
            except subprocess.TimeoutExpired:
                result["process_reaped"] = False
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
    result.update(
        status=status or (
            "completed" if result["returncode"] == 0 and (not stderr or stderr_is_diagnostic)
            and sent == len(input_raw) and result["process_reaped"] else "program_error"),
        stdout_hex=stdout.hex(), stderr_hex=stderr.hex(),
        elapsed_ns=time.monotonic_ns() - started, stdin_complete=sent == len(input_raw),
    )
    return result


def sandbox_argv(python_root, worker_source, seccomp_fd):
    return [
        "/usr/bin/bwrap", "--unshare-all", "--die-with-parent", "--new-session",
        "--ro-bind", "/usr/lib", "/usr/lib",
        "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", python_root, "/runtime",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--dir", "/work", "--chdir", "/work", "--seccomp", str(seccomp_fd),
        "/runtime/bin/python3.12", "-I", "-S", "-B", "-c", worker_source,
    ]


def syscall_number(name, resolved, native_architecture):
    if resolved >= 0:
        return resolved
    if name == "clone3" and native_architecture == 0xC000003E:
        return 435
    raise RuntimeError("unsupported syscall: " + name)


def seccomp_filter():
    library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
    ]
    library.seccomp_rule_add.restype = ctypes.c_int
    library.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.seccomp_export_bpf.restype = ctypes.c_int
    library.seccomp_arch_native.argtypes = []
    library.seccomp_arch_native.restype = ctypes.c_uint32
    context = library.seccomp_init(0x7FFF0000)
    if not context:
        raise RuntimeError("seccomp initialization")
    descriptor = os.memfd_create("tlf-validation-seccomp", os.MFD_CLOEXEC)
    try:
        for name in (
            "socket", "fork", "vfork", "clone", "clone3", "unshare", "setns",
            "ptrace", "process_vm_readv", "process_vm_writev",
        ):
            number = syscall_number(
                name, library.seccomp_syscall_resolve_name(name.encode("ascii")),
                library.seccomp_arch_native(),
            )
            if library.seccomp_rule_add(context, 0x00050000 | errno.EPERM, number, 0) != 0:
                raise RuntimeError("seccomp rule: " + name)
        if library.seccomp_export_bpf(context, descriptor) != 0:
            raise RuntimeError("seccomp export")
        os.lseek(descriptor, 0, os.SEEK_SET)
        raw = os.read(descriptor, 65536)
        if not raw or len(raw) >= 65536:
            raise RuntimeError("seccomp size")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor, hashlib.sha256(raw).hexdigest()
    except BaseException:
        os.close(descriptor)
        raise
    finally:
        library.seccomp_release(context)


def no_new_privileges():
    library = ctypes.CDLL(None, use_errno=True)
    library.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                             ctypes.c_ulong, ctypes.c_ulong]
    library.prctl.restype = ctypes.c_int
    if library.prctl(38, 1, 0, 0, 0) != 0:
        raise RuntimeError("no_new_privs")
    if "NoNewPrivs:\t1" not in Path("/proc/self/status").read_text():
        raise RuntimeError("no_new_privs observation")


def unique_pairs(entries):
    result = {}
    for name, value in entries:
        if name in result:
            raise ValueError("duplicate key")
        result[name] = value
    return result


def response_is_valid(raw):
    def valid_value(value, depth=0):
        if depth > 16:
            return False
        if value is None or type(value) in (bool, int):
            return True
        if type(value) is str:
            return value.isascii()
        if type(value) is list:
            return len(value) <= 4096 and all(valid_value(item, depth + 1) for item in value)
        if type(value) is dict:
            return len(value) <= 64 and all(
                type(name) is str and name.isascii() and valid_value(item, depth + 1)
                for name, item in value.items())
        return False

    try:
        if len(raw) > 65536:
            return False
        outcome = json.loads(raw, object_pairs_hook=unique_pairs)
        if type(outcome) is not dict:
            return False
        if outcome.get("kind") == "return":
            return set(outcome) == {"kind", "value"} and valid_value(outcome["value"])
        return outcome == {"kind": "raises", "type": "ValueError"}
    except (ValueError, RecursionError):
        return False


def verify_runtime(mode, observed, expected):
    if mode in ("d01", "matrix") and expected is None:
        raise ValueError("task execution requires approved runtime")
    if expected is not None and encoded(observed) != encoded(expected):
        raise ValueError("runtime differs from preflight")


def run_bundle(bundle_raw, *, expected_runtime=None):
    if type(bundle_raw) is not bytes or len(bundle_raw) > 131072:
        raise ValueError("bundle bytes")
    bundle = json.loads(bundle_raw, object_pairs_hook=unique_pairs)
    if type(bundle) is not dict or set(bundle) != {
        "kind", "mode", "run_id", "worker_source", "requests",
    }:
        raise ValueError("bundle fields")
    run_id = bundle["run_id"]
    if (bundle["kind"] != "LORA_REMOTE_VALIDATION_BUNDLE"
            or bundle["mode"] not in ("probe", "d01", "matrix") or type(run_id) is not str
            or len(run_id) != 32 or any(letter not in "0123456789abcdef" for letter in run_id)
            or type(bundle["worker_source"]) is not str
            or len(bundle["worker_source"].encode()) > 16384
            or type(bundle["requests"]) is not list
            or not 1 <= len(bundle["requests"]) <= 32
            or (bundle["mode"] == "d01" and len(bundle["requests"]) != 32)):
        raise ValueError("bundle values")
    for request in bundle["requests"]:
        if type(request) is not dict or set(request) != {
            "request_id", "source", "entry_point", "args",
        } or len(encoded(request)) > 8192:
            raise ValueError("request projection")
    if sys.platform != "linux":
        raise RuntimeError("Linux backend only")
    started = time.monotonic_ns()
    deadline = time.monotonic() + 90
    no_new_privileges()
    descriptor, filter_sha256 = seccomp_filter()
    try:
        python_root = str(Path(sys.base_prefix).resolve())
        command = sandbox_argv(python_root, bundle["worker_source"], descriptor)
        runtime = {
            "hostname": platform.node(), "uid": os.getuid(),
            "python": platform.python_version(), "python_root": python_root,
            "python_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
            "bwrap_sha256": hashlib.sha256(Path("/usr/bin/bwrap").read_bytes()).hexdigest(),
            "seccomp_sha256": filter_sha256, "argv_prefix": command[:-1],
            "no_new_privs": True,
            "host_namespaces": {
                name: os.readlink("/proc/self/ns/" + name)
                for name in ("mnt", "pid", "net", "user")
            },
        }
        verify_runtime(bundle["mode"], runtime, expected_runtime)
        print(encoded({
            "kind": "ready", "run_id": run_id,
            "bundle_sha256": hashlib.sha256(bundle_raw).hexdigest(),
            "worker_sha256": hashlib.sha256(bundle["worker_source"].encode()).hexdigest(),
            "runtime": runtime,
        }).decode(), flush=True)
        received = 0
        stopped = False
        for ordinal, request in enumerate(bundle["requests"]):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stopped = True
                break
            os.lseek(descriptor, 0, os.SEEK_SET)
            result = collect_process(
                command, encoded(request), timeout_seconds=min(2, remaining),
                pass_fds=(descriptor,), environment={"PATH": "/runtime/bin", "HOME": "/nonexistent"},
            )
            if result["status"] == "completed" and not response_is_valid(
                    bytes.fromhex(result["stdout_hex"])):
                result["status"] = "program_error"
            print(encoded({
                "kind": "row", "run_id": run_id, "ordinal": ordinal,
                "request_sha256": digest(request), "result": result,
            }).decode(), flush=True)
            received += 1
            if result["status"] != "completed":
                stopped = True
                break
        print(encoded({
            "kind": "done", "run_id": run_id, "received": received, "stopped": stopped,
            "elapsed_ns": time.monotonic_ns() - started,
        }).decode(), flush=True)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    run_bundle(sys.stdin.buffer.read(131073),
               expected_runtime=globals().get("EXPECTED_RUNTIME"))
