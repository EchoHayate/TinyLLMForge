import base64
import copy
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import signal
import socket
import stat
import struct
import sys
import threading
import time


SESSION_SOURCE_NAMES = frozenset({
    "pilot_protocol.py",
    "validation_contract.py",
    "tokenizer_measurement.py",
    "tokenizer_runtime_guard.py",
    "tokenizer_runtime_profile.py",
    "tokenizer_snapshot.py",
    "tokenizer_session.py",
})
CHANNELS = ("READY", "RESULT", "STDOUT", "STDERR", "DONE")
RESULT_LIMIT = 67_108_864
DIAGNOSTIC_LIMIT = 1_048_576
CHANNEL_LIMITS = (4096, RESULT_LIMIT, DIAGNOSTIC_LIMIT, DIAGNOSTIC_LIMIT, 4096)
HEADER = struct.Struct("!5sBQ")
WIRE_LIMIT = sum(CHANNEL_LIMITS) + HEADER.size * len(CHANNELS)
PAYLOAD_LIMIT = 16 * 1024 * 1024
ENVIRONMENT = {
    "USE_TORCH": "0",
    "USE_TF": "0",
    "USE_FLAX": "0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONDONTWRITEBYTECODE": "1",
}


def encoded(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def sha256_hex(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def pairs(entries):
        mapping = {}
        for name, item in entries:
            if name in mapping:
                raise ValueError("duplicate JSON key")
            mapping[name] = item
        return mapping

    def invalid_constant(token):
        raise ValueError("nonfinite JSON value")

    def reject_float(token):
        parsed = float(token)
        if not math.isfinite(parsed):
            raise ValueError("nonfinite JSON float")
        raise ValueError("float disallowed")

    try:
        text = raw.decode("utf-8") if type(raw) is bytes else raw
        return json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
            parse_float=reject_float,
        )
    except (TypeError, UnicodeError, ValueError, RecursionError):
        raise ValueError("invalid strict JSON") from None


def encode_frame(channel, raw):
    if channel not in CHANNELS or type(raw) is not bytes:
        raise ValueError("invalid frame")
    ordinal = CHANNELS.index(channel)
    if len(raw) > CHANNEL_LIMITS[ordinal]:
        raise ValueError("frame overflow")
    return HEADER.pack(b"TLFH1", ordinal + 1, len(raw)) + raw


class FrameDecoder:
    def __init__(self):
        self.frames = {}
        self._buffer = bytearray()
        self._size = None
        self._closed = False
        self._failed = False

    def feed(self, raw):
        if self._closed or self._failed:
            raise ValueError("closed frame decoder")
        try:
            self._buffer.extend(raw)
            while self._buffer or self._size == 0:
                ordinal = len(self.frames)
                if ordinal == len(CHANNELS):
                    raise ValueError("trailing frame data")
                if self._size is None:
                    if len(self._buffer) < HEADER.size:
                        if not b"TLFH1".startswith(bytes(self._buffer[:5])):
                            raise ValueError("invalid frame magic")
                        return
                    magic, channel_num, size = HEADER.unpack(self._buffer[:HEADER.size])
                    if magic != b"TLFH1" or channel_num != ordinal + 1:
                        raise ValueError("invalid or duplicate channel")
                    if size > CHANNEL_LIMITS[ordinal]:
                        raise ValueError("frame overflow")
                    del self._buffer[:HEADER.size]
                    self._size = size
                if len(self._buffer) < self._size:
                    return
                self.frames[CHANNELS[ordinal]] = bytes(self._buffer[:self._size])
                del self._buffer[:self._size]
                self._size = None
        except BaseException:
            self._failed = True
            raise

    def finish(self):
        if self._failed or self._buffer or self._size is not None or len(self.frames) != 5:
            raise ValueError("truncated frame stream")
        self._closed = True
        return dict(self.frames)


def decode_frames(raw):
    decoder = FrameDecoder()
    for offset in range(0, len(raw), 65536):
        decoder.feed(raw[offset:offset + 65536])
    return decoder.finish()


def _validate_projection(projection_obj):
    if (
        type(projection_obj) is not dict
        or set(projection_obj) != {"kind", "version", "rows"}
        or projection_obj["kind"] != "LORA_RECALIBRATION_COUNTING_PROJECTION"
        or type(projection_obj["version"]) is not int
        or projection_obj["version"] != 1
        or type(projection_obj["rows"]) is not list
        or not 1 <= len(projection_obj["rows"]) <= 120
    ):
        raise ValueError("invalid counting projection")
    for expected_ord, row in enumerate(projection_obj["rows"]):
        if (
            type(row) is not dict
            or set(row) != {"ordinal", "messages", "answer", "row_sha256"}
            or type(row["ordinal"]) is not int
            or row["ordinal"] != expected_ord
            or type(row["messages"]) is not list
            or not row["messages"]
            or (row["answer"] is not None and type(row["answer"]) is not str)
            or type(row["row_sha256"]) is not str
            or re.fullmatch("[0-9a-f]{64}", row["row_sha256"]) is None
        ):
            raise ValueError("invalid counting projection row")
        for msg in row["messages"]:
            if (
                type(msg) is not dict
                or set(msg) != {"role", "content"}
                or type(msg["role"]) is not str
                or type(msg["content"]) is not str
            ):
                raise ValueError("invalid message entry")
    return projection_obj


def validate_bundle(raw, config):
    try:
        if (
            type(raw) is not bytes
            or len(raw) > PAYLOAD_LIMIT
            or len(raw) != config["payload_bytes"]
            or sha256_hex(raw) != config["payload_sha256"]
        ):
            raise ValueError()
        bundle = strict_json(raw)
        if set(bundle) != {"sources", "projection", "profile"}:
            raise ValueError()
        if (
            set(bundle["sources"]) != SESSION_SOURCE_NAMES
            or set(config["sources_sha256"]) != SESSION_SOURCE_NAMES
        ):
            raise ValueError()
        sources = {}
        for name in sorted(SESSION_SOURCE_NAMES):
            content = base64.b64decode(bundle["sources"][name], validate=True)
            if len(content) > DIAGNOSTIC_LIMIT or sha256_hex(content) != config["sources_sha256"][name]:
                raise ValueError()
            sources[name] = content
        projection_raw = base64.b64decode(bundle["projection"], validate=True)
        profile_raw = base64.b64decode(bundle["profile"], validate=True)
        if (
            sha256_hex(projection_raw) != config["projection_sha256"]
            or sha256_hex(profile_raw) != config["profile_sha256"]
            or strict_json(profile_raw)["run_id"] != config["run_id"]
        ):
            raise ValueError()
        projection_obj = _validate_projection(strict_json(projection_raw))
        return sources, projection_obj, profile_raw
    except (KeyError, TypeError, ValueError):
        raise ValueError("input bundle invalid") from None


def private_directory(path):
    path_obj = Path(path)
    if not path_obj.is_absolute() or path_obj != path_obj.resolve(strict=True):
        raise ValueError("noncanonical directory")
    info = path_obj.stat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError("private owned directory required")
    return path_obj


def prepare_stage(parent, run_id):
    if type(run_id) is not str or re.fullmatch("[0-9a-f]{32}", run_id) is None:
        raise ValueError("invalid run ID")
    parent_path = Path(parent)
    if not parent_path.is_absolute() or parent_path != parent_path.resolve():
        raise ValueError("noncanonical parent")
    if not parent_path.exists():
        parent_path.mkdir(mode=0o700)
    private_directory(parent_path)
    if os.path.lexists(parent_path / run_id):
        raise FileExistsError("run already exists")
    stage = parent_path / ("bootstrap-" + run_id)
    stage.mkdir(mode=0o700)
    return private_directory(stage)


def write_private(target_path, raw):
    descriptor = os.open(
        target_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def validate_work_timeout(value):
    if type(value) is not int or not 0 < value <= 1200:
        raise ValueError("invalid work timeout")
    return value


def remaining_seconds(anchor, remaining_ns, now, *, work_timeout_seconds=1200):
    maximum_ns = validate_work_timeout(work_timeout_seconds) * 1_000_000_000
    if type(remaining_ns) is not int or not 0 < remaining_ns <= maximum_ns:
        raise ValueError("invalid remaining deadline")
    seconds = anchor + remaining_ns / 1_000_000_000.0 - now
    if seconds <= 0:
        raise TimeoutError("work deadline")
    return seconds


def task_filter_instructions():
    instructions = [
        (0x20, 0, 0, 4),
        (0x15, 1, 0, 0xC000003E),
        (0x06, 0, 0, 0x80000000),
        (0x20, 0, 0, 0),
        (0x45, 0, 1, 0x40000000),
        (0x06, 0, 0, 0x80000000),
    ]
    for syscall_number in (56, 57, 58, 435):
        instructions.extend((
            (0x15, 0, 1, syscall_number),
            (0x06, 0, 0, 0x00050000 | errno.EPERM),
        ))
    instructions.append((0x06, 0, 0, 0x7FFF0000))
    return instructions


def install_task_denial():
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise ValueError("Linux x86_64 required")
    if len(os.listdir("/proc/self/task")) != 1 or threading.active_count() != 1:
        raise ValueError("single thread required")

    class SockFilter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jump_true", ctypes.c_ubyte),
            ("jump_false", ctypes.c_ubyte),
            ("constant", ctypes.c_uint32),
        ]

    class SockProgram(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ushort),
            ("filter", ctypes.POINTER(SockFilter)),
        ]

    instructions = task_filter_instructions()
    filters = (SockFilter * len(instructions))(
        *(SockFilter(*inst) for inst in instructions)
    )
    program = SockProgram(len(instructions), filters)
    library = ctypes.CDLL(None, use_errno=True)
    library.prctl.restype = ctypes.c_int
    if library.prctl(38, 1, 0, 0, 0) != 0:
        raise ValueError("no_new_privs failed")
    if library.prctl(22, 2, ctypes.byref(program), 0, 0) != 0:
        raise ValueError("seccomp failed")
    if library.prctl(39, 0, 0, 0, 0) != 1 or library.prctl(21, 0, 0, 0, 0) != 2:
        raise ValueError("seccomp verification failed")
    try:
        child = os.fork()
    except OSError as err:
        if err.errno != errno.EPERM:
            raise ValueError("fork denial unverified") from None
    else:
        if child == 0:
            os._exit(79)
        os.waitpid(child, 0)
        raise ValueError("fork unexpectedly allowed")
    worker_thread = threading.Thread(target=lambda: None)
    try:
        worker_thread.start()
    except RuntimeError:
        pass
    else:
        worker_thread.join(timeout=0.1)
        raise ValueError("thread unexpectedly allowed")
    if len(os.listdir("/proc/self/task")) != 1:
        raise ValueError("task count changed")
    return {
        "threads": 1,
        "fork_denied": True,
        "thread_denied": True,
        "denied_syscalls": [56, 57, 58, 435],
        "non_x86_and_x32": "KILL_PROCESS",
    }


def limit_diagnostics():
    resource.setrlimit(resource.RLIMIT_FSIZE, (DIAGNOSTIC_LIMIT, DIAGNOSTIC_LIMIT))
    signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGXFSZ})


def redirect_diagnostics(stage, *, enforce_limit=True):
    saved_stdout = os.dup(1)
    os.set_inheritable(saved_stdout, False)
    if enforce_limit:
        limit_diagnostics()
    for descriptor, name in ((1, "native.stdout"), (2, "native.stderr")):
        logfile = os.open(
            Path(stage) / name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        try:
            os.fchmod(logfile, 0o600)
            os.dup2(logfile, descriptor)
        finally:
            os.close(logfile)
    return saved_stdout


def read_diagnostics(stage):
    diagnostics = {}
    for channel_name, filename in (("STDOUT", "native.stdout"), ("STDERR", "native.stderr")):
        with open(Path(stage) / filename, "rb") as stream:
            diagnostics[channel_name] = stream.read(DIAGNOSTIC_LIMIT + 1)
        if len(diagnostics[channel_name]) >= DIAGNOSTIC_LIMIT:
            raise ValueError("diagnostic saturated or overflowed")
    return diagnostics


def _write_frame(descriptor, channel_name, raw):
    framed = encode_frame(channel_name, raw)
    view = memoryview(framed)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("frame write failed")
        view = view[written:]


def _read_exact(stream, size):
    content = bytearray()
    while len(content) < size:
        chunk = stream.read(min(size - len(content), 65536))
        if not chunk:
            raise ValueError("truncated input")
        content.extend(chunk)
    return bytes(content)


def execute_projection_on_session(rows, session_opener):
    session = None
    phase = "OPEN"
    ordinal = None
    records = []
    try:
        session = session_opener()
        for row in rows:
            phase = "MEASURE"
            ordinal = row["ordinal"]
            copied_messages = copy.deepcopy(row["messages"])
            record = session.measure(copied_messages, row["answer"])
            records.append(record)
            if record["measurement"]["status"] != "OK":
                break
        phase = "FINALIZE"
        ordinal = None
        finalization = session.finish()
        return {
            "records": records,
            "finalization": finalization,
            "failure": None,
            "abort_failed": False,
        }
    except BaseException as err:
        abort_failed = False
        if session is not None:
            try:
                session.abort()
            except BaseException as abort_err:
                if isinstance(err, Exception) and not isinstance(abort_err, Exception):
                    raise
                abort_failed = True
        if not isinstance(err, Exception):
            raise
        return {
            "records": records,
            "finalization": None,
            "failure": {"phase": phase, "ordinal": ordinal},
            "abort_failed": abort_failed,
        }


def bootstrap(config):
    work_timeout_seconds = validate_work_timeout(config.get("work_timeout_seconds", 1200))
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGALRM})
    signal.setitimer(signal.ITIMER_REAL, 30)
    anchor = time.monotonic()
    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise ValueError("isolated bootstrap required")
    if socket.gethostname().split(".")[0] != config["hostname"] or os.getuid() != config["uid"]:
        raise ValueError("host identity mismatch")
    if (
        sys.executable != config["interpreter"]
        and os.path.realpath(sys.executable) != config["resolved_interpreter"]
    ):
        raise ValueError("interpreter mismatch")
    for path_key in ("site_packages", "source"):
        path_obj = Path(config[path_key])
        if not path_obj.is_absolute() or path_obj != path_obj.resolve(strict=True) or not path_obj.is_dir():
            raise ValueError("noncanonical runtime path")
    stage = prepare_stage(config["parent"], config["run_id"])
    if str(stage) != config["stage"]:
        raise ValueError("stage binding mismatch")
    os.chdir(stage)
    os.environ.update(ENVIRONMENT)
    os.environ.update({
        "HOME": str(stage),
        "TMPDIR": str(stage),
        "HF_HOME": str(stage / "hf-cache"),
    })
    protection = install_task_denial()
    identity = {
        "run_id": config["run_id"],
        "config_sha256": sha256_hex(encoded(config)),
        "hostname": config["hostname"],
        "uid": os.getuid(),
        "stage": str(stage),
        "protection": protection,
        "core_limit": list(resource.getrlimit(resource.RLIMIT_CORE)),
    }
    _write_frame(1, "READY", encoded(identity))
    output_descriptor = 1
    phase = "INPUT"
    diagnostics = {"STDOUT": b"", "STDERR": b""}
    session_output = None
    bootstrap_failure = None
    try:
        grant_raw = sys.stdin.buffer.readline(256)
        if not grant_raw.endswith(b"\n"):
            raise ValueError("invalid grant")
        grant = strict_json(grant_raw)
        if set(grant) != {"remaining_ns"}:
            raise ValueError("invalid grant")
        signal.setitimer(
            signal.ITIMER_REAL,
            remaining_seconds(
                anchor,
                grant["remaining_ns"],
                time.monotonic(),
                work_timeout_seconds=work_timeout_seconds,
            ),
        )
        raw_payload = _read_exact(sys.stdin.buffer, config["payload_bytes"])
        if sys.stdin.buffer.read(1):
            raise ValueError("trailing input")
        remaining_seconds(
            anchor,
            grant["remaining_ns"],
            time.monotonic(),
            work_timeout_seconds=work_timeout_seconds,
        )
        sources, projection_obj, profile_raw = validate_bundle(raw_payload, config)
        for name, content in sources.items():
            write_private(stage / name, content)
        write_private(stage / "projection.json", encoded(projection_obj))
        write_private(stage / "profile.json", profile_raw)
        output_descriptor = redirect_diagnostics(stage, enforce_limit=False)
        sys.path.extend([str(stage), config["site_packages"]])
        phase = "VALIDATE"
        from tokenizer_runtime_profile import validate_profile
        from tokenizer_session import open_session
        from tokenizer_snapshot import create_snapshot

        validate_profile(profile_raw, config["profile_sha256"])
        phase = "SNAPSHOT"
        create_snapshot(
            config["source"], config["parent"], profile_raw, config["profile_sha256"]
        )
        phase = "SESSION"
        limit_diagnostics()
        run_root = Path(config["parent"]) / config["run_id"]
        session_output = execute_projection_on_session(
            projection_obj["rows"],
            lambda: open_session(run_root, profile_raw, config["profile_sha256"]),
        )
    except Exception:
        bootstrap_failure = {"phase": phase}
    if output_descriptor != 1:
        sys.stdout.flush()
        sys.stderr.flush()
        diagnostics = read_diagnostics(stage)
    result_bytes = encoded({
        "kind": "LORA_RECALIBRATION_TOKENIZER_REMOTE_RESULT",
        "version": 1,
        "identity": identity,
        "projection_sha256": config["projection_sha256"],
        "profile_sha256": config["profile_sha256"],
        "bootstrap_failure": bootstrap_failure,
        "session_output": session_output,
    })
    _write_frame(output_descriptor, "RESULT", result_bytes)
    _write_frame(output_descriptor, "STDOUT", diagnostics["STDOUT"])
    _write_frame(output_descriptor, "STDERR", diagnostics["STDERR"])
    _write_frame(
        output_descriptor,
        "DONE",
        encoded({
            "config_sha256": identity["config_sha256"],
            "result_sha256": sha256_hex(result_bytes),
        }),
    )
    return 0 if bootstrap_failure is None else 1
