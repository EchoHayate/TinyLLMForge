import errno
import hashlib
import json
import math
import os
from os import read as _io_read
from os import write as _io_write
from pathlib import Path
import platform
import re
import selectors
import signal
import stat
import struct
import subprocess
import sys
import time


STDIN_MAGIC = b"LRB01"
WIRE_MAGIC = b"LRV01"
FRAME_HEADER = struct.Struct("!5sBQ")
CONTROL_LIMITS = {1: 65536, 2: 65536, 3: 4096, 4: 131072}
WIRE_FRAME_LIMITS = {1: 4096, 2: 8388608, 3: 131072, 4: 65536, 5: 4096}
WIRE_TOTAL_LIMIT = 8388608
BACKEND_SHA256 = "9be98748dfd2bdcdd132c6ed4c04c3dd67f11d49d6049fcbe7960bde622690da"
WORKER_SHA256 = "d2c8095b4d1e854a2f497b084226a92b47aa8b083d59c8ab272b088544fedca4"
READY_DEADLINE_NS = 10_000_000_000
MAX_WORK_NS = 95_000_000_000
CLEANUP_LIMIT_NS = 2_000_000_000

BACKEND_EXEC_SNIPPET = (
    "import json, pathlib\n"
    "expected_rt = json.loads(pathlib.Path('expected_runtime.json').read_text('utf-8'))\n"
    "backend_bytes = pathlib.Path('backend.py').read_bytes()\n"
    "namespace = {'__name__': '__main__', 'EXPECTED_RUNTIME': expected_rt}\n"
    "exec(compile(backend_bytes, 'validation_remote.py', 'exec'), namespace)\n"
)

STAGE_CONFIG_FIELDS = (
    "kind", "version", "stage_id", "preparation_sha256", "plan_sha256",
    "backend_sha256", "worker_sha256", "remote_supervisor_sha256",
    "ssh_argv_prefix", "remote_python", "local_stage_root", "remote_stage_root",
    "preflight_receipt_sha256", "expected_runtime", "batches", "limits",
)
RUNTIME_FIELDS = (
    "hostname", "uid", "python", "python_root", "python_sha256", "bwrap_sha256",
    "seccomp_sha256", "argv_prefix", "no_new_privs", "host_namespaces",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _stdin_fd():
    return sys.stdin.fileno()


def _stdout_fd():
    return sys.stdout.fileno()


def _current_executable():
    return os.path.realpath(sys.executable)


def _canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _digest_obj(value):
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_hex(raw):
    return hashlib.sha256(raw).hexdigest()


def _strict_plain(value, depth=0):
    _require(depth <= 32, "JSON nesting depth")
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is list:
        for element in value:
            _strict_plain(element, depth + 1)
        return
    if type(value) is dict:
        for key, item in value.items():
            _require(type(key) is str, "JSON key type")
            _strict_plain(item, depth + 1)
        return
    raise ValueError("unsupported JSON value type")


def _parse_strict_json(raw, *, limit):
    _require(type(raw) is bytes and len(raw) <= limit, "JSON byte limit")

    def check_pairs(entries):
        mapping = {}
        for key, item in entries:
            _require(key not in mapping, "duplicate JSON key")
            mapping[key] = item
        return mapping

    def reject_constant(token):
        raise ValueError("nonfinite JSON constant")

    def reject_float(token):
        parsed = float(token)
        _require(math.isfinite(parsed), "nonfinite JSON float")
        raise ValueError("floating point disallowed")

    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=check_pairs,
            parse_constant=reject_constant,
            parse_float=reject_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as parse_err:
        raise ValueError("invalid strict JSON") from parse_err
    _strict_plain(decoded)
    return decoded


def _check_fields(mapping, expected_names):
    _require(type(mapping) is dict and set(mapping) == set(expected_names), "field mismatch")


def _check_hex(text, length):
    _require(
        type(text) is str and re.fullmatch("[0-9a-f]{%d}" % length, text) is not None,
        "invalid hex digest",
    )


def _check_stage_root_syntax(stage_root, stage_id):
    _require(
        type(stage_root) is str and len(stage_root) <= 256 and stage_root.startswith("/"),
        "stage root syntax",
    )
    parts = stage_root[1:].split("/")
    _require(
        len(parts) >= 2
        and all(
            re.fullmatch("[A-Za-z0-9_.-]+", segment) is not None
            and segment not in (".", "..")
            for segment in parts
        )
        and parts[-1] == stage_id,
        "canonical stage root components",
    )


def encode_control_frame(channel, payload_raw):
    _require(type(channel) is int and channel in CONTROL_LIMITS, "control channel")
    _require(
        type(payload_raw) is bytes and len(payload_raw) <= CONTROL_LIMITS[channel],
        "control payload size",
    )
    return FRAME_HEADER.pack(STDIN_MAGIC, channel, len(payload_raw)) + payload_raw


def decode_control_frame(frame_raw, *, expected_channel):
    _require(
        type(expected_channel) is int and expected_channel in CONTROL_LIMITS,
        "expected control channel",
    )
    _require(
        type(frame_raw) is bytes and len(frame_raw) >= FRAME_HEADER.size,
        "short control frame",
    )
    magic, channel, size = FRAME_HEADER.unpack_from(frame_raw, 0)
    _require(magic == STDIN_MAGIC, "control magic")
    _require(channel == expected_channel, "control channel order")
    _require(size <= CONTROL_LIMITS[channel], "control payload limit")
    _require(len(frame_raw) == FRAME_HEADER.size + size, "control frame length")
    return frame_raw[FRAME_HEADER.size:]


def _pack_wire_frame(channel, payload_raw):
    _require(
        type(payload_raw) is bytes and len(payload_raw) <= WIRE_FRAME_LIMITS[channel],
        "wire frame payload limit",
    )
    return FRAME_HEADER.pack(WIRE_MAGIC, channel, len(payload_raw)) + payload_raw


class _FileErrors:
    def __init__(self):
        self.primary = None
        self.traceback = None
        self.phase = None
        self.secondary = []

    def capture(self, error, phase):
        self.secondary.extend(getattr(error, "_remote_secondary_errors", []))
        if self.primary is None:
            self.primary, self.traceback, self.phase = error, error.__traceback__, phase
        elif not isinstance(error, Exception) and isinstance(self.primary, Exception):
            self.secondary.append({"phase": self.phase, "type": type(self.primary).__name__})
            self.primary, self.traceback, self.phase = error, error.__traceback__, phase
        else:
            self.secondary.append({"phase": phase, "type": type(error).__name__})

    def close(self, descriptor, phase):
        try:
            os.close(descriptor)
        except BaseException as close_err:
            self.capture(close_err, phase)

    def raise_if_any(self):
        if self.primary is not None:
            self.primary._remote_secondary_errors = self.secondary
            raise self.primary.with_traceback(self.traceback)


def _write_exclusive(directory_fd, filename, raw):
    descriptor = os.open(
        filename,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    failures = _FileErrors()
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        view = memoryview(raw)
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("nonpositive file write")
            offset += written
        os.fsync(descriptor)
        os.fsync(directory_fd)
    except BaseException as write_err:
        failures.capture(write_err, "write_body")
    finally:
        failures.close(descriptor, "write_close")
    failures.raise_if_any()


def _read_regular_at(directory_fd, filename, limit):
    descriptor = os.open(
        filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd
    )
    failures = _FileErrors()
    collected = bytearray()
    try:
        metadata = os.fstat(descriptor)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.getuid()
            and stat.S_IMODE(metadata.st_mode) == 0o600
            and metadata.st_size <= limit,
            "regular private file metadata",
        )
        while len(collected) <= limit:
            chunk = os.read(descriptor, min(65536, limit + 1 - len(collected)))
            if not chunk:
                break
            collected.extend(chunk)
    except BaseException as read_err:
        failures.capture(read_err, "read_body")
    finally:
        failures.close(descriptor, "read_close")
    failures.raise_if_any()
    return bytes(collected)


def _verify_private_dir(descriptor):
    metadata = os.fstat(descriptor)
    _require(
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and stat.S_IMODE(metadata.st_mode) == 0o700,
        "private directory ownership or mode",
    )


def _claim_remote_batch_root(stage_root, stage_id, batch_index, batch_name, stage_claim_obj):
    parent_path = str(Path(stage_root).parent)
    _require(os.path.realpath(parent_path) == parent_path, "canonical stage parent")
    parent_fd = os.open(parent_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    stage_fd = None
    batch_fd = None
    failures = _FileErrors()
    try:
        parent_meta = os.fstat(parent_fd)
        _require(stat.S_ISDIR(parent_meta.st_mode), "stage parent directory")
        if batch_index == 0:
            os.mkdir(stage_id, 0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
            stage_fd = os.open(
                stage_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd
            )
            os.fchmod(stage_fd, 0o700)
            _verify_private_dir(stage_fd)
            _write_exclusive(stage_fd, "stage-claim.json", _canonical_bytes(stage_claim_obj))
        else:
            stage_fd = os.open(
                stage_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd
            )
            _verify_private_dir(stage_fd)
            existing_raw = _read_regular_at(stage_fd, "stage-claim.json", 4096)
            _require(
                existing_raw == _canonical_bytes(stage_claim_obj),
                "stage claim mismatch across batches",
            )
        _require(os.path.realpath(stage_root) == stage_root, "canonical stage root")
        os.mkdir(batch_name, 0o700, dir_fd=stage_fd)
        os.fsync(stage_fd)
        batch_fd = os.open(
            batch_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=stage_fd
        )
        os.fchmod(batch_fd, 0o700)
        _verify_private_dir(batch_fd)
    except BaseException as claim_err:
        failures.capture(claim_err, "claim_batch_root")
    finally:
        if stage_fd is not None:
            failures.close(stage_fd, "close_stage_fd")
        failures.close(parent_fd, "close_parent_fd")
    if failures.primary is not None and batch_fd is not None:
        failures.close(batch_fd, "close_batch_fd_on_error")
    failures.raise_if_any()
    return batch_fd


class _MonotonicClock:
    def __init__(self):
        self.last_ns = None
        self.invalid = False

    def now_ns(self):
        current = time.monotonic_ns()
        if (
            self.invalid
            or type(current) is not int
            or current < 0
            or (self.last_ns is not None and current < self.last_ns)
        ):
            self.invalid = True
            raise RuntimeError("monotonic clock invalid")
        self.last_ns = current
        return current

    def check_deadline(self, deadline_ns):
        current = self.now_ns()
        if current >= deadline_ns:
            raise TimeoutError("deadline exceeded")
        return current


def _write_all_fd(descriptor, raw, clock, deadline_ns):
    offset = 0
    view = memoryview(raw)
    while offset < len(view):
        clock.check_deadline(deadline_ns)
        written = _io_write(descriptor, view[offset:])
        if written <= 0:
            raise OSError("nonpositive stdout write")
        offset += written


class _ControlReader:
    def __init__(self, descriptor, clock):
        self.descriptor = descriptor
        self.clock = clock
        self.buffer = bytearray()
        self.eof = False
        os.set_blocking(descriptor, False)

    def _fill(self, needed, deadline_ns):
        with selectors.DefaultSelector() as selector:
            selector.register(self.descriptor, selectors.EVENT_READ, "stdin")
            while len(self.buffer) < needed and not self.eof:
                now_ns = self.clock.check_deadline(deadline_ns)
                remaining_sec = (deadline_ns - now_ns) / 1_000_000_000.0
                events = selector.select(min(0.05, remaining_sec))
                self.clock.check_deadline(deadline_ns)
                if not events:
                    continue
                try:
                    chunk = _io_read(self.descriptor, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    self.eof = True
                    break
                self.buffer.extend(chunk)
                if len(self.buffer) > 262144:
                    raise ValueError("control stream buffer limit exceeded")

    def read_frame(self, expected_channel, deadline_ns):
        self._fill(FRAME_HEADER.size, deadline_ns)
        _require(len(self.buffer) >= FRAME_HEADER.size, "unexpected EOF before control header")
        magic, channel, size = FRAME_HEADER.unpack_from(self.buffer, 0)
        _require(magic == STDIN_MAGIC, "control magic")
        _require(channel == expected_channel, "control channel order")
        _require(size <= CONTROL_LIMITS[expected_channel], "control payload limit")
        total = FRAME_HEADER.size + size
        self._fill(total, deadline_ns)
        _require(len(self.buffer) >= total, "unexpected EOF in control payload")
        payload = bytes(self.buffer[FRAME_HEADER.size:total])
        del self.buffer[:total]
        return payload

    def require_eof(self, deadline_ns):
        self._fill(1, deadline_ns)
        _require(len(self.buffer) == 0 and self.eof, "trailing stdin bytes")


def _validate_launch_config(launch_config, expected_config_sha256, expected_batch_index):
    _check_fields(launch_config, STAGE_CONFIG_FIELDS)
    _require(
        launch_config["kind"] == "LORA_RECALIBRATION_STAGE_CONFIG"
        and type(launch_config["version"]) is int
        and launch_config["version"] == 1,
        "stage config kind/version",
    )
    _require(_digest_obj(launch_config) == expected_config_sha256, "config digest mismatch")
    stage_id = launch_config["stage_id"]
    _check_hex(stage_id, 32)
    for digest_key in (
        "preparation_sha256", "plan_sha256", "backend_sha256", "worker_sha256",
        "remote_supervisor_sha256", "preflight_receipt_sha256",
    ):
        _check_hex(launch_config[digest_key], 64)
    _require(
        launch_config["backend_sha256"] == BACKEND_SHA256
        and launch_config["worker_sha256"] == WORKER_SHA256,
        "pinned backend/worker digest",
    )
    runtime = launch_config["expected_runtime"]
    _check_fields(runtime, RUNTIME_FIELDS)
    _require(
        runtime["hostname"] == platform.node() and runtime["uid"] == os.getuid(),
        "remote host/UID mismatch",
    )
    _require(
        type(launch_config["remote_python"]) is str
        and launch_config["remote_python"] == _current_executable(),
        "remote interpreter mismatch",
    )
    stage_root = launch_config["remote_stage_root"]
    _check_stage_root_syntax(stage_root, stage_id)
    batches = launch_config["batches"]
    _require(type(batches) is list and len(batches) == 15, "batches length")
    _require(0 <= expected_batch_index < 15, "batch index bounds")
    batch_entry = batches[expected_batch_index]
    _check_fields(
        batch_entry,
        ("batch_index", "run_id", "bundle_sha256", "bundle_bytes", "batch_root"),
    )
    _require(batch_entry["batch_index"] == expected_batch_index, "batch index order")
    _check_hex(batch_entry["run_id"], 32)
    _require(batch_entry["run_id"] != stage_id, "distinct stage/run ID")
    _check_hex(batch_entry["bundle_sha256"], 64)
    _require(
        type(batch_entry["bundle_bytes"]) is int
        and 1 <= batch_entry["bundle_bytes"] <= 131072,
        "bundle byte bounds",
    )
    expected_batch_root = (
        stage_root + ("/batch-%02d-" % expected_batch_index) + batch_entry["run_id"]
    )
    _require(batch_entry["batch_root"] == expected_batch_root, "batch root binding")
    return batch_entry


def _safe_raw_ns():
    try:
        val = time.monotonic_ns()
        if type(val) is int and val >= 0:
            return val
    except BaseException:
        return None
    return None


def _cleanup_backend(process, constructor_completed, clock, cancellations):
    start_raw = _safe_raw_ns()
    clock_ok = not clock.invalid and start_raw is not None
    last_raw = start_raw
    deadline_raw = (start_raw + CLEANUP_LIMIT_NS) if start_raw is not None else None
    child_reaped = False
    returncode = None
    group_absent = None
    pid = process.pid

    def sample_raw():
        nonlocal clock_ok, last_raw
        cur = _safe_raw_ns()
        if cur is None or last_raw is None or cur < last_raw:
            clock_ok = False
            return None
        last_raw = cur
        return cur

    def capture_cancel(err):
        if not isinstance(err, Exception) and not cancellations:
            cancellations.append((err, err.__traceback__))

    def probe_group():
        nonlocal group_absent
        try:
            os.killpg(pid, 0)
            group_absent = False
            return True
        except ProcessLookupError:
            group_absent = True
            return False
        except BaseException as probe_err:
            group_absent = None
            capture_cancel(probe_err)
            return None

    def signal_group(signum):
        try:
            os.killpg(pid, signum)
        except ProcessLookupError:
            pass
        except BaseException as sig_err:
            capture_cancel(sig_err)

    def signal_direct(signum):
        if constructor_completed or child_reaped:
            return
        try:
            process.send_signal(signum)
        except ProcessLookupError:
            pass
        except BaseException as dir_err:
            capture_cancel(dir_err)

    def reap_child(wait_sec):
        nonlocal child_reaped, returncode
        if child_reaped:
            return
        try:
            waited = process.wait(timeout=max(0.0, wait_sec))
            if type(waited) is int:
                returncode = waited
                child_reaped = True
        except subprocess.TimeoutExpired:
            pass
        except BaseException as wait_err:
            capture_cancel(wait_err)

    sent_kill = False
    try:
        alive = probe_group()
        if alive is not False:
            signal_group(signal.SIGTERM)
        signal_direct(signal.SIGTERM)
        cur_ns = sample_raw()
        initial_wait = (
            min(0.2, max(0.0, (deadline_raw - cur_ns) / 1_000_000_000.0))
            if cur_ns is not None and deadline_raw is not None
            else 0.05
        )
        reap_child(initial_wait)
        for step_idx in range(110):
            observed_ns = sample_raw()
            alive = probe_group()
            reap_child(0.0)
            if alive is False and child_reaped:
                break
            elapsed_ok = (
                (observed_ns - start_raw >= 200_000_000)
                if (observed_ns is not None and start_raw is not None)
                else (step_idx >= 5)
            )
            if not sent_kill and elapsed_ok:
                if alive is not False:
                    signal_group(signal.SIGKILL)
                signal_direct(signal.SIGKILL)
                sent_kill = True
                reap_child(0.0)
                alive = probe_group()
                if alive is False and child_reaped:
                    break
            if observed_ns is not None and deadline_raw is not None and observed_ns >= deadline_raw:
                break
            after_ns = sample_raw()
            if after_ns is not None and deadline_raw is not None:
                rem_ns = deadline_raw - after_ns
                if rem_ns <= 0:
                    break
                sleep_sec = min(0.02, rem_ns / 1_000_000_000.0)
            else:
                sleep_sec = 0.01
            time.sleep(sleep_sec)
    except BaseException as loop_err:
        clock_ok = False
        capture_cancel(loop_err)
        signal_group(signal.SIGKILL)
        signal_direct(signal.SIGKILL)
        reap_child(0.0)
        probe_group()

    end_raw = sample_raw()
    if clock_ok and start_raw is not None and end_raw is not None and deadline_raw is not None:
        elapsed_ns = end_raw - start_raw
        bounded = end_raw <= deadline_raw
    else:
        elapsed_ns = CLEANUP_LIMIT_NS + 1
        bounded = False
    return {
        "child_reaped": child_reaped,
        "returncode": returncode if child_reaped else None,
        "group_absent": group_absent is True,
        "cleanup_elapsed_ns": elapsed_ns,
        "cleanup_bounded": bounded,
    }


def _run_backend_child(remote_python, batch_root, bundle_raw, clock, work_deadline_ns):
    backend_started_ns = clock.now_ns()
    process = object.__new__(subprocess.Popen)
    constructor_completed = False
    stdout_buf = bytearray()
    stderr_buf = bytearray()
    sent_offset = 0
    stdin_closed = False
    stdout_eof = False
    stderr_eof = False
    run_status = None
    cancellations = []
    selector = None
    try:
        try:
            subprocess.Popen.__init__(
                process,
                [remote_python, "-I", "-S", "-B", "-c", BACKEND_EXEC_SNIPPET],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=batch_root,
                env={
                    "PATH": "/usr/bin:/bin",
                    "LANG": "C",
                    "LC_ALL": "C",
                    "HOME": "/nonexistent",
                },
                close_fds=True,
                pass_fds=(),
                start_new_session=True,
                shell=False,
            )
            constructor_completed = True
        except BaseException as spawn_err:
            if not isinstance(spawn_err, Exception):
                cancellations.append((spawn_err, spawn_err.__traceback__))
            child_pid = getattr(process, "pid", None)
            if type(child_pid) is not int or child_pid <= 1:
                if cancellations:
                    err_obj, err_tb = cancellations[0]
                    raise err_obj.with_traceback(err_tb)
                return None, None, b"", b"", 0, "launch_failed"
            run_status = "program_error"

        if run_status is None:
            selector = selectors.DefaultSelector()
            for stream_obj, stream_label in (
                (process.stdout, "stdout"),
                (process.stderr, "stderr"),
            ):
                os.set_blocking(stream_obj.fileno(), False)
                selector.register(stream_obj, selectors.EVENT_READ, stream_label)
            os.set_blocking(process.stdin.fileno(), False)
            selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")

            while True:
                now_ns = clock.now_ns()
                if now_ns >= work_deadline_ns:
                    run_status = "timeout"
                    break
                polled = process.poll()
                if not selector.get_map() and polled is not None:
                    break
                remaining_sec = (work_deadline_ns - now_ns) / 1_000_000_000.0
                if not selector.get_map():
                    time.sleep(min(0.02, remaining_sec))
                    continue
                events = selector.select(min(0.05, remaining_sec))
                if clock.now_ns() >= work_deadline_ns:
                    run_status = "timeout"
                    break
                for key, _mask in events:
                    if key.data == "stdin":
                        if stdin_closed:
                            continue
                        try:
                            written = _io_write(
                                key.fd, bundle_raw[sent_offset:sent_offset + 16384]
                            )
                        except BlockingIOError:
                            continue
                        except BrokenPipeError:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                            stdin_closed = True
                            continue
                        if written > 0:
                            sent_offset += written
                        if sent_offset >= len(bundle_raw):
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                            stdin_closed = True
                        continue
                    target_buf, stream_cap = (
                        (stdout_buf, WIRE_FRAME_LIMITS[2])
                        if key.data == "stdout"
                        else (stderr_buf, WIRE_FRAME_LIMITS[3])
                    )
                    remaining_cap = stream_cap - len(target_buf)
                    try:
                        chunk = _io_read(key.fd, min(65536, remaining_cap + 1))
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        if key.data == "stdout":
                            stdout_eof = True
                        else:
                            stderr_eof = True
                    else:
                        target_buf.extend(chunk[:remaining_cap])
                        if len(chunk) > remaining_cap:
                            run_status = "output_limit"
                            break
                if run_status is not None:
                    break
    except BaseException as body_err:
        if not isinstance(body_err, Exception):
            cancellations.append((body_err, body_err.__traceback__))
        run_status = "program_error"
    finally:
        cleanup_info = None
        child_pid = getattr(process, "pid", None)
        if type(child_pid) is int and child_pid > 1:
            try:
                cleanup_info = _cleanup_backend(
                    process, constructor_completed, clock, cancellations
                )
            except BaseException as clean_unhandled:
                if not isinstance(clean_unhandled, Exception) and not cancellations:
                    cancellations.append((clean_unhandled, clean_unhandled.__traceback__))
                run_status = run_status or "program_error"
        for resource in (
            selector,
            None if stdin_closed else getattr(process, "stdin", None),
            getattr(process, "stdout", None),
            getattr(process, "stderr", None),
        ):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as close_err:
                    if not isinstance(close_err, Exception) and not cancellations:
                        cancellations.append((close_err, close_err.__traceback__))
                    run_status = run_status or "program_error"

    if cancellations:
        err_obj, err_tb = cancellations[0]
        raise err_obj.with_traceback(err_tb)
    if clock.invalid:
        raise RuntimeError("monotonic clock invalid")

    backend_ended_ns = clock.now_ns()
    backend_elapsed_ns = backend_ended_ns - backend_started_ns
    stdin_complete = sent_offset == len(bundle_raw)
    reaped = bool(cleanup_info and cleanup_info["child_reaped"])
    returncode = cleanup_info["returncode"] if reaped else None
    group_absent = bool(cleanup_info and cleanup_info["group_absent"])
    cleanup_bounded = bool(cleanup_info and cleanup_info["cleanup_bounded"])
    cleanup_elapsed_ns = cleanup_info["cleanup_elapsed_ns"] if cleanup_info else 0

    if run_status is None:
        if (
            reaped
            and returncode == 0
            and stdin_complete
            and stdout_eof
            and stderr_eof
            and group_absent
            and cleanup_bounded
            and backend_ended_ns <= work_deadline_ns + CLEANUP_LIMIT_NS
        ):
            run_status = "completed"
        else:
            run_status = "program_error"

    backend_record = {
        "status": run_status,
        "returncode": returncode,
        "elapsed_ns": backend_elapsed_ns,
        "process_reaped": reaped,
        "stdin_complete": stdin_complete,
    }
    return (
        process.pid,
        backend_record,
        bytes(stdout_buf),
        bytes(stderr_buf),
        cleanup_elapsed_ns,
        run_status,
    )


def supervise_batch(expected_config_sha256, expected_batch_index):
    _check_hex(expected_config_sha256, 64)
    _require(
        type(expected_batch_index) is int and 0 <= expected_batch_index < 15,
        "expected batch index",
    )
    clock = _MonotonicClock()
    anchor_ns = clock.now_ns()
    ready_deadline_ns = anchor_ns + READY_DEADLINE_NS
    work_deadline_ns = anchor_ns + MAX_WORK_NS

    reader = _ControlReader(_stdin_fd(), clock)
    bootstrap_raw = reader.read_frame(1, ready_deadline_ns)
    backend_raw = reader.read_frame(2, ready_deadline_ns)
    _require(_sha256_hex(backend_raw) == BACKEND_SHA256, "backend source digest")

    bootstrap = _parse_strict_json(bootstrap_raw, limit=CONTROL_LIMITS[1])
    _check_fields(
        bootstrap,
        ("kind", "version", "package_sha256", "config_sha256", "batch_index", "launch_config"),
    )
    _require(
        bootstrap["kind"] == "LORA_RECALIBRATION_BOOTSTRAP"
        and type(bootstrap["version"]) is int
        and bootstrap["version"] == 1
        and bootstrap["config_sha256"] == expected_config_sha256
        and bootstrap["batch_index"] == expected_batch_index,
        "bootstrap header binding",
    )
    _check_hex(bootstrap["package_sha256"], 64)
    launch_config = bootstrap["launch_config"]
    batch_entry = _validate_launch_config(
        launch_config, expected_config_sha256, expected_batch_index
    )

    stage_id = launch_config["stage_id"]
    stage_root = launch_config["remote_stage_root"]
    batch_name = ("batch-%02d-" % expected_batch_index) + batch_entry["run_id"]
    stage_claim = {
        "kind": "LORA_RECALIBRATION_REMOTE_STAGE_CLAIM",
        "stage_id": stage_id,
        "package_sha256": bootstrap["package_sha256"],
        "config_sha256": expected_config_sha256,
    }
    batch_fd = _claim_remote_batch_root(
        stage_root, stage_id, expected_batch_index, batch_name, stage_claim
    )
    batch_failures = _FileErrors()
    exit_status = 0

    def try_save_abort(reason_text):
        try:
            _write_exclusive(
                batch_fd,
                "supervisor-abort.json",
                _canonical_bytes({"reason": reason_text}),
            )
        except BaseException as abort_err:
            batch_failures.capture(abort_err, "persist_supervisor_abort")

    try:
        _write_exclusive(batch_fd, "bootstrap.json", bootstrap_raw)
        _write_exclusive(batch_fd, "backend.py", backend_raw)
        _write_exclusive(
            batch_fd,
            "expected_runtime.json",
            _canonical_bytes(launch_config["expected_runtime"]),
        )
        ready_now_ns = clock.check_deadline(ready_deadline_ns)
        ready_elapsed_ns = ready_now_ns - anchor_ns
        _require(ready_elapsed_ns <= READY_DEADLINE_NS, "READY deadline")

        ready_obj = {
            "kind": "LORA_RECALIBRATION_READY",
            "version": 1,
            "stage_id": stage_id,
            "batch_index": expected_batch_index,
            "run_id": batch_entry["run_id"],
            "preparation_sha256": launch_config["preparation_sha256"],
            "package_sha256": bootstrap["package_sha256"],
            "config_sha256": expected_config_sha256,
            "bundle_sha256": batch_entry["bundle_sha256"],
            "backend_sha256": BACKEND_SHA256,
            "worker_sha256": WORKER_SHA256,
            "hostname": launch_config["expected_runtime"]["hostname"],
            "uid": launch_config["expected_runtime"]["uid"],
            "stage_root": stage_root,
            "batch_root": batch_entry["batch_root"],
            "expected_runtime_sha256": _digest_obj(launch_config["expected_runtime"]),
            "ready_elapsed_ns": ready_elapsed_ns,
        }
        ready_raw = _canonical_bytes(ready_obj)
        ready_sha256 = _sha256_hex(ready_raw)
        ready_frame = _pack_wire_frame(1, ready_raw)
        _write_all_fd(_stdout_fd(), ready_frame, clock, ready_deadline_ns)

        payload_received = False
        bundle_raw = None
        done_status = None
        diagnostic_note = "ok"

        try:
            grant_raw = reader.read_frame(3, work_deadline_ns)
            grant = _parse_strict_json(grant_raw, limit=CONTROL_LIMITS[3])
            _check_fields(
                grant,
                ("kind", "version", "ready_sha256", "granted_work_ns", "bundle_sha256", "bundle_bytes"),
            )
            _require(
                grant["kind"] == "LORA_RECALIBRATION_GRANT"
                and type(grant["version"]) is int
                and grant["version"] == 1
                and grant["ready_sha256"] == ready_sha256
                and grant["bundle_sha256"] == batch_entry["bundle_sha256"]
                and grant["bundle_bytes"] == batch_entry["bundle_bytes"]
                and type(grant["granted_work_ns"]) is int
                and 1 <= grant["granted_work_ns"] <= MAX_WORK_NS,
                "grant validation failed",
            )
            work_deadline_ns = min(work_deadline_ns, anchor_ns + grant["granted_work_ns"])
            clock.check_deadline(work_deadline_ns)
            candidate_bundle = reader.read_frame(4, work_deadline_ns)
            _require(
                len(candidate_bundle) == batch_entry["bundle_bytes"]
                and _sha256_hex(candidate_bundle) == batch_entry["bundle_sha256"],
                "bundle digest or length mismatch",
            )
            reader.require_eof(work_deadline_ns)
            _write_exclusive(
                batch_fd,
                "payload-received.json",
                _canonical_bytes({"bundle_sha256": batch_entry["bundle_sha256"]}),
            )
            bundle_raw = candidate_bundle
            payload_received = True
        except TimeoutError:
            done_status = "timeout"
            diagnostic_note = "input_timeout"
        except ValueError:
            done_status = "input_error"
            diagnostic_note = "input_validation_failed"
        except BaseException as cancel_err:
            if not isinstance(cancel_err, Exception):
                done_status = "cancelled"
                diagnostic_note = "cancelled_before_backend"
                batch_failures.capture(cancel_err, "pre_backend_cancel")
            else:
                done_status = "input_error"
                diagnostic_note = "input_exception"

        backend_pid = None
        backend_record = None
        backend_stdout_raw = b""
        backend_stderr_raw = b""
        cleanup_elapsed_ns = 0
        aborted = False

        try:
            if payload_received and done_status is None:
                (
                    backend_pid,
                    backend_record,
                    backend_stdout_raw,
                    backend_stderr_raw,
                    cleanup_elapsed_ns,
                    done_status,
                ) = _run_backend_child(
                    launch_config["remote_python"],
                    batch_entry["batch_root"],
                    bundle_raw,
                    clock,
                    work_deadline_ns,
                )
            work_end_ns = clock.now_ns()
        except RuntimeError:
            try_save_abort("clock_invalid")
            exit_status = 2
            aborted = True
        except BaseException as run_cancel:
            batch_failures.capture(run_cancel, "run_backend")
            try_save_abort(type(run_cancel).__name__)
            aborted = True

        if not aborted:
            total_elapsed_ns = max(0, work_end_ns - anchor_ns)
            work_elapsed_ns = max(0, total_elapsed_ns - cleanup_elapsed_ns)
            if backend_record is not None and backend_record["elapsed_ns"] > work_elapsed_ns + cleanup_elapsed_ns:
                work_elapsed_ns = backend_record["elapsed_ns"]
            if done_status == "completed" and (
                work_elapsed_ns > MAX_WORK_NS
                or work_end_ns > work_deadline_ns + CLEANUP_LIMIT_NS
                or cleanup_elapsed_ns > CLEANUP_LIMIT_NS
            ):
                done_status = "timeout"
                if backend_record is not None:
                    backend_record["status"] = "timeout"

            diag_raw = _canonical_bytes({
                "kind": "LORA_RECALIBRATION_DIAGNOSTIC",
                "note": diagnostic_note,
                "status": done_status,
            })
            frame_two = _pack_wire_frame(2, backend_stdout_raw)
            frame_three = _pack_wire_frame(3, backend_stderr_raw)
            frame_four = _pack_wire_frame(4, diag_raw)

            prior_digests = [
                _sha256_hex(ready_frame),
                _sha256_hex(frame_two),
                _sha256_hex(frame_three),
                _sha256_hex(frame_four),
            ]
            done_obj = {
                "kind": "LORA_RECALIBRATION_DONE",
                "version": 1,
                "stage_id": stage_id,
                "batch_index": expected_batch_index,
                "run_id": batch_entry["run_id"],
                "package_sha256": bootstrap["package_sha256"],
                "config_sha256": expected_config_sha256,
                "ready_sha256": ready_sha256,
                "payload_sha256": batch_entry["bundle_sha256"] if payload_received else None,
                "prior_frames_sha256": prior_digests,
                "status": done_status,
                "payload_received": payload_received,
                "backend_pid": backend_pid,
                "backend": backend_record,
                "work_elapsed_ns": work_elapsed_ns,
                "cleanup_elapsed_ns": cleanup_elapsed_ns,
            }
            done_raw = _canonical_bytes(done_obj)
            frame_five = _pack_wire_frame(5, done_raw)
            trailing_wire = frame_two + frame_three + frame_four + frame_five
            if len(ready_frame) + len(trailing_wire) > WIRE_TOTAL_LIMIT:
                try_save_abort("aggregate_wire_limit_exceeded")
                exit_status = 3
            else:
                _write_exclusive(batch_fd, "done.json", done_raw)
                emit_deadline_ns = work_end_ns + 5_000_000_000
                _write_all_fd(_stdout_fd(), trailing_wire, clock, emit_deadline_ns)
    except BaseException as outer_err:
        batch_failures.capture(outer_err, "supervise_batch_body")
    finally:
        batch_failures.close(batch_fd, "close_batch_fd")
    batch_failures.raise_if_any()
    return exit_status


if __name__ == "__main__":
    if len(sys.argv) != 3 or not re.fullmatch(r"(0|[1-9][0-9]*)", sys.argv[2]):
        raise SystemExit(2)
    raise SystemExit(supervise_batch(sys.argv[1], int(sys.argv[2])))
