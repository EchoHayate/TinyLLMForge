import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import signal
import stat
import struct
import subprocess
import tempfile
import tokenize
import types
import unittest
from unittest import mock

from recalibration_execution_evidence import decode_receipt
from recalibration_execution_package import BACKEND_SHA256
from recalibration_execution_wire import FRAME_NAMES, HEADER, MAGIC, decode_wire
from recalibration_validation import WORKER_SHA256
from validation_contract import digest, encoded


REPO_ROOT = Path(__file__).resolve().parent
BACKEND_RAW = (REPO_ROOT / "validation_remote.py").read_bytes()
assert hashlib.sha256(BACKEND_RAW).hexdigest() == BACKEND_SHA256

with mock.patch("subprocess.Popen") as import_spawn, mock.patch("os.killpg") as import_killpg:
    try:
        remote_module = importlib.import_module("recalibration_execution_remote")
    except ModuleNotFoundError as import_error:
        if import_error.name != "recalibration_execution_remote":
            raise
        remote_module = None
    assert not import_spawn.called
    assert not import_killpg.called


STAGE_ID = "a" * 32
RUN_IDS = [("%02x" % index) + ("b" * 30) for index in range(15)]
PACKAGE_SHA256 = "c" * 64
PREPARATION_SHA256 = "d" * 64
PLAN_SHA256 = "e" * 64
PREFLIGHT_SHA256 = "f" * 64
REMOTE_SUPERVISOR_SHA256 = "1" * 64


def sample_runtime(hostname="target-node", uid=1054, python_root="/runtime/env"):
    return {
        "hostname": hostname,
        "uid": uid,
        "python": "3.12.13",
        "python_root": python_root,
        "python_sha256": "2" * 64,
        "bwrap_sha256": "3" * 64,
        "seccomp_sha256": "4" * 64,
        "argv_prefix": [
            "/usr/bin/bwrap", "--unshare-all", "--seccomp", "3",
            "/runtime/env/bin/python3.12", "-I", "-S", "-B", "-c",
        ],
        "no_new_privs": True,
        "host_namespaces": {
            "mnt": "mnt:[1]", "pid": "pid:[2]", "net": "net:[3]", "user": "user:[4]",
        },
    }


def sample_bundle_raw(run_id=RUN_IDS[0]):
    return encoded({
        "kind": "LORA_REMOTE_VALIDATION_BUNDLE",
        "mode": "matrix",
        "run_id": run_id,
        "worker_source": "x = 1\n",
        "requests": [{
            "request_id": "request-0000",
            "source": "def solve(val):\n    return val + 1\n",
            "entry_point": "solve",
            "args": [2],
        }],
    })


def sample_launch_config(stage_root, *, hostname="target-node", uid=1054,
                         remote_python="/runtime/env/bin/python3.12",
                         bundle_raw=None):
    actual_bundle = sample_bundle_raw() if bundle_raw is None else bundle_raw
    bundle_sha256 = hashlib.sha256(actual_bundle).hexdigest()
    runtime = sample_runtime(hostname=hostname, uid=uid)
    batches = []
    for index, run_id in enumerate(RUN_IDS):
        batches.append({
            "batch_index": index,
            "run_id": run_id,
            "bundle_sha256": bundle_sha256 if index == 0 else ("%02x" % index) + ("9" * 62),
            "bundle_bytes": len(actual_bundle),
            "batch_root": stage_root + ("/batch-%02d-" % index) + run_id,
        })
    return {
        "kind": "LORA_RECALIBRATION_STAGE_CONFIG",
        "version": 1,
        "stage_id": STAGE_ID,
        "preparation_sha256": PREPARATION_SHA256,
        "plan_sha256": PLAN_SHA256,
        "backend_sha256": BACKEND_SHA256,
        "worker_sha256": WORKER_SHA256,
        "remote_supervisor_sha256": REMOTE_SUPERVISOR_SHA256,
        "ssh_argv_prefix": ["/usr/bin/ssh", "rl-server"],
        "remote_python": remote_python,
        "local_stage_root": "/private/tmp/local-" + STAGE_ID,
        "remote_stage_root": stage_root,
        "preflight_receipt_sha256": PREFLIGHT_SHA256,
        "expected_runtime": runtime,
        "batches": batches,
        "limits": {
            "worker_seconds": 2,
            "backend_seconds": 90,
            "ready_seconds": 10,
            "supervisor_work_seconds": 95,
            "supervisor_reap_seconds": 2,
            "ssh_work_seconds": 110,
            "ssh_reap_seconds": 2,
            "grant_reserve_seconds": 5,
            "stage_work_seconds": 1800,
            "stage_finalize_seconds": 5,
            "wire_bytes": 8388608,
            "ssh_stderr_bytes": 131072,
        },
    }


class ArtificialClock:
    def __init__(self):
        self.nanoseconds = 1_000_000_000_000

    def monotonic_ns(self):
        self.nanoseconds += 1_000_000
        return self.nanoseconds

    def monotonic(self):
        return self.monotonic_ns() / 1_000_000_000.0

    def sleep(self, seconds):
        self.nanoseconds += int(max(0.0, seconds) * 1_000_000_000)


class ArtificialPipe:
    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.closed = False

    def fileno(self):
        if self.closed:
            raise ValueError("I/O operation on closed pipe")
        return self.descriptor

    def close(self):
        self.closed = True


class ArtificialBackendProcess:
    def __init__(self):
        self.pid = 432100
        self.stdin = ArtificialPipe(200)
        self.stdout = ArtificialPipe(201)
        self.stderr = ArtificialPipe(202)
        self.returncode = 0
        self.wait_error = None
        self.written = bytearray()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.wait_error is not None:
            raise self.wait_error
        return self.returncode

    def send_signal(self, signum):
        if self.returncode is None:
            self.returncode = -signum


class ArtificialSelector:
    def __init__(self, clock):
        self.clock = clock
        self.registered = {}
        self.idle = False
        self.error = None
        self.closed = False

    def register(self, fileobj, events, data=None):
        descriptor = fileobj.fileno() if hasattr(fileobj, "fileno") else int(fileobj)
        entry = types.SimpleNamespace(fileobj=fileobj, fd=descriptor, events=events, data=data)
        self.registered[descriptor] = entry
        return entry

    def unregister(self, fileobj):
        descriptor = fileobj.fileno() if hasattr(fileobj, "fileno") else int(fileobj)
        return self.registered.pop(descriptor)

    def get_map(self):
        return self.registered

    def select(self, timeout=None):
        if timeout is not None:
            self.clock.sleep(min(0.02, max(0.0, timeout)))
        if self.error is not None:
            raise self.error
        if self.idle:
            return []
        return [(entry, entry.events) for entry in list(self.registered.values())]

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class RemoteSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.spawn_mock = mock.patch("subprocess.Popen").start()
        self.killpg_mock = mock.patch("os.killpg").start()
        self.addCleanup(mock.patch.stopall)
        self.assertIsNotNone(remote_module, "remote supervisor implementation is absent")
        self.tempdir = tempfile.TemporaryDirectory(prefix="tlf-remote-sup-")
        self.addCleanup(self.tempdir.cleanup)
        self.base_dir = Path(self.tempdir.name).resolve()
        self.stage_root = str(self.base_dir / STAGE_ID)
        self.clock = ArtificialClock()
        self.backend_proc = ArtificialBackendProcess()
        self.selector = ArtificialSelector(self.clock)
        self.bundle_raw = sample_bundle_raw()
        self.launch_config = sample_launch_config(
            self.stage_root, uid=os.getuid(), bundle_raw=self.bundle_raw
        )
        self.config_sha256 = digest(self.launch_config)
        self.group_alive = False
        self.group_survives = False
        self.killpg_mock.side_effect = self._handle_killpg
        self.spawn_handler = self._default_spawn
        owner = self

        class OwnedArtificialPopen(ArtificialBackendProcess):
            def __init__(instance, *args, **kwargs):
                instance.pid = None
                instance.stdin = None
                instance.stdout = None
                instance.stderr = None
                supplied = owner.spawn_handler(*args, **kwargs)
                instance.__dict__.update(supplied.__dict__)

        mock.patch.object(remote_module.subprocess, "Popen", OwnedArtificialPopen).start()
        mock.patch.object(remote_module.time, "monotonic_ns", self.clock.monotonic_ns).start()
        mock.patch.object(remote_module.time, "monotonic", self.clock.monotonic).start()
        mock.patch.object(remote_module.time, "sleep", self.clock.sleep).start()
        mock.patch.object(remote_module.selectors, "DefaultSelector", side_effect=lambda: ArtificialSelector(self.clock)).start()
        mock.patch.object(remote_module.os, "set_blocking").start()
        mock.patch.object(remote_module.platform, "node", return_value="target-node").start()
        mock.patch.object(
            remote_module, "_current_executable",
            return_value="/runtime/env/bin/python3.12",
        ).start()
        self.backend_chunks = {
            201: [b'{"kind":"ready"}\n{"kind":"done"}\n', b""],
            202: [b"", b""],
        }
        self.stdin_chunks = []
        self.stdout_written = bytearray()
        self.ready_callback = None
        mock.patch.object(remote_module, "_stdin_fd", return_value=190).start()
        mock.patch.object(remote_module, "_stdout_fd", return_value=191).start()
        mock.patch.object(remote_module, "_io_read", side_effect=self._io_read).start()
        mock.patch.object(remote_module, "_io_write", side_effect=self._io_write).start()

    def _default_spawn(self, *args, **kwargs):
        self.spawn_mock(*args, **kwargs)
        return self.backend_proc

    def _handle_killpg(self, pgid, signum):
        self.assertEqual(pgid, self.backend_proc.pid)
        if not self.group_alive:
            raise ProcessLookupError()
        if signum != 0 and not self.group_survives:
            self.group_alive = False

    def _io_read(self, descriptor, count):
        if descriptor == 190:
            if not self.stdin_chunks:
                return b""
            item = self.stdin_chunks.pop(0)
            if isinstance(item, BaseException):
                raise item
            if len(item) > count:
                self.stdin_chunks.insert(0, item[count:])
            return item[:count]
        stream_items = self.backend_chunks[descriptor]
        if not stream_items:
            return b""
        item = stream_items.pop(0)
        if isinstance(item, BaseException):
            raise item
        if len(item) > count:
            stream_items.insert(0, item[count:])
        return item[:count]

    def _io_write(self, descriptor, raw):
        if descriptor == 191:
            self.stdout_written.extend(raw)
            if (
                self.ready_callback is not None
                and len(self.stdout_written) >= HEADER.size
            ):
                magic, channel, size = HEADER.unpack_from(self.stdout_written, 0)
                if magic == MAGIC and channel == 1 and len(self.stdout_written) >= HEADER.size + size:
                    callback = self.ready_callback
                    self.ready_callback = None
                    callback(bytes(self.stdout_written[HEADER.size:HEADER.size + size]))
            return len(raw)
        if descriptor == 200:
            self.backend_proc.written.extend(raw)
            return len(raw)
        raise AssertionError("unexpected write descriptor: %r" % (descriptor,))

    def _bootstrap_frames(self, *, batch_index=0, config_override=None, backend_raw=BACKEND_RAW):
        active_config = self.launch_config if config_override is None else config_override
        active_digest = digest(active_config)
        bootstrap = {
            "kind": "LORA_RECALIBRATION_BOOTSTRAP",
            "version": 1,
            "package_sha256": PACKAGE_SHA256,
            "config_sha256": active_digest,
            "batch_index": batch_index,
            "launch_config": active_config,
        }
        return (
            remote_module.encode_control_frame(1, encoded(bootstrap))
            + remote_module.encode_control_frame(2, backend_raw),
            active_digest,
        )

    def _grant_frames(self, ready_raw, *, granted_work_ns=95_000_000_000, bundle_raw=None):
        actual_bundle = self.bundle_raw if bundle_raw is None else bundle_raw
        grant = {
            "kind": "LORA_RECALIBRATION_GRANT",
            "version": 1,
            "ready_sha256": hashlib.sha256(ready_raw).hexdigest(),
            "granted_work_ns": granted_work_ns,
            "bundle_sha256": hashlib.sha256(actual_bundle).hexdigest(),
            "bundle_bytes": len(actual_bundle),
        }
        return (
            remote_module.encode_control_frame(3, encoded(grant))
            + remote_module.encode_control_frame(4, actual_bundle)
        )

    def _expected_ready(self, batch_index=0):
        batch = self.launch_config["batches"][batch_index]
        return {
            "kind": "LORA_RECALIBRATION_READY",
            "version": 1,
            "stage_id": STAGE_ID,
            "batch_index": batch_index,
            "run_id": batch["run_id"],
            "preparation_sha256": PREPARATION_SHA256,
            "package_sha256": PACKAGE_SHA256,
            "config_sha256": self.config_sha256,
            "bundle_sha256": batch["bundle_sha256"],
            "backend_sha256": BACKEND_SHA256,
            "worker_sha256": WORKER_SHA256,
            "hostname": "target-node",
            "uid": os.getuid(),
            "stage_root": self.stage_root,
            "batch_root": batch["batch_root"],
            "expected_runtime_sha256": digest(self.launch_config["expected_runtime"]),
        }

    def test_pure_stdlib_imports_and_style_contract(self):
        import ast
        for filename in (
            "recalibration_execution_remote.py",
            "test_recalibration_execution_remote.py",
        ):
            raw = (REPO_ROOT / filename).read_bytes()
            tree = ast.parse(raw)
            single_names = [
                node.id for node in ast.walk(tree)
                if isinstance(node, ast.Name) and len(node.id) == 1
            ]
            self.assertEqual(single_names, [], filename)
            comments = [
                tok for tok in tokenize.tokenize(io.BytesIO(raw).readline)
                if tok.type == tokenize.COMMENT
            ]
            self.assertEqual(comments, [], filename)
        tree = ast.parse((REPO_ROOT / "recalibration_execution_remote.py").read_bytes())
        imported = {
            alias.name for node in tree.body
            if isinstance(node, ast.Import) for alias in node.names
        } | {
            node.module for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        allowed = {
            "errno", "hashlib", "json", "math", "os", "pathlib", "platform",
            "re", "selectors", "signal", "stat", "struct", "subprocess", "sys", "time",
        }
        self.assertTrue(imported.issubset(allowed), imported - allowed)

    def test_control_framing_roundtrip_and_rejections(self):
        frame = remote_module.encode_control_frame(1, b'{"ok":true}')
        self.assertEqual(
            remote_module.decode_control_frame(frame, expected_channel=1),
            b'{"ok":true}',
        )
        for bad in (b"short", b"WRONG" + frame[5:], frame + b"x"):
            with self.assertRaises(ValueError):
                remote_module.decode_control_frame(bad, expected_channel=1)
        with self.assertRaises(ValueError):
            remote_module.decode_control_frame(frame, expected_channel=2)

    def test_full_five_frame_success_decodes_via_evidence_module(self):
        boot_raw, cfg_digest = self._bootstrap_frames()
        self.stdin_chunks = [boot_raw]
        self.ready_callback = lambda ready_raw: self.stdin_chunks.append(
            self._grant_frames(ready_raw)
        )
        exit_code = remote_module.supervise_batch(cfg_digest, 0)
        self.assertEqual(exit_code, 0)
        wire_raw = bytes(self.stdout_written)
        wire_sha = hashlib.sha256(wire_raw).hexdigest()
        decoded_receipt = decode_receipt(
            wire_raw, expected_sha256=wire_sha, expected=self._expected_ready(0)
        )
        self.assertTrue(decoded_receipt["receipt_binding_consistent"])
        self.assertTrue(decoded_receipt["backend_lifecycle_complete"])
        self.assertTrue(decoded_receipt["payload_received"])
        self.assertTrue(decoded_receipt["backend_started"])
        self.assertEqual(decoded_receipt["done"]["status"], "completed")
        self.assertEqual(bytes(self.backend_proc.written), self.bundle_raw)
        self.spawn_mock.assert_called_once()
        spawn_args, spawn_kwargs = self.spawn_mock.call_args
        self.assertEqual(spawn_args[0][:4], ["/runtime/env/bin/python3.12", "-I", "-S", "-B"])
        self.assertEqual(spawn_kwargs["close_fds"], True)
        self.assertEqual(spawn_kwargs.get("pass_fds", ()), ())
        self.assertEqual(spawn_kwargs["start_new_session"], True)
        batch_dir = Path(self.launch_config["batches"][0]["batch_root"])
        self.assertEqual(stat.S_IMODE(Path(self.stage_root).stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(batch_dir.stat().st_mode), 0o700)
        self.assertEqual((batch_dir / "backend.py").read_bytes(), BACKEND_RAW)

    def test_exclusive_batch_root_blocks_replay(self):
        boot_raw, cfg_digest = self._bootstrap_frames()
        self.stdin_chunks = [boot_raw]
        self.ready_callback = lambda ready_raw: self.stdin_chunks.append(
            self._grant_frames(ready_raw)
        )
        self.assertEqual(remote_module.supervise_batch(cfg_digest, 0), 0)
        self.stdout_written.clear()
        self.stdin_chunks = [boot_raw]
        with self.assertRaises(FileExistsError):
            remote_module.supervise_batch(cfg_digest, 0)
        self.assertEqual(self.spawn_mock.call_count, 1)

    def test_host_uid_or_config_mismatch_fails_before_ready(self):
        boot_raw, cfg_digest = self._bootstrap_frames()
        with mock.patch.object(remote_module.platform, "node", return_value="wrong-host"):
            self.stdin_chunks = [boot_raw]
            with self.assertRaises(ValueError):
                remote_module.supervise_batch(cfg_digest, 0)
        self.assertEqual(bytes(self.stdout_written), b"")
        self.spawn_mock.assert_not_called()

    def test_eof_after_ready_emits_valid_input_error_without_backend_launch(self):
        boot_raw, cfg_digest = self._bootstrap_frames()
        self.stdin_chunks = [boot_raw]
        exit_code = remote_module.supervise_batch(cfg_digest, 0)
        self.assertEqual(exit_code, 0)
        self.spawn_mock.assert_not_called()
        wire_raw = bytes(self.stdout_written)
        receipt = decode_receipt(
            wire_raw,
            expected_sha256=hashlib.sha256(wire_raw).hexdigest(),
            expected=self._expected_ready(0),
        )
        self.assertEqual(receipt["done"]["status"], "input_error")
        self.assertFalse(receipt["payload_received"])
        self.assertIsNone(receipt["done"]["backend"])
        self.assertIsNone(receipt["done"]["backend_pid"])

    def test_trailing_stdin_bytes_rejected_before_backend_launch(self):
        boot_raw, cfg_digest = self._bootstrap_frames()
        self.stdin_chunks = [boot_raw]
        self.ready_callback = lambda ready_raw: self.stdin_chunks.append(
            self._grant_frames(ready_raw) + b"extra"
        )
        remote_module.supervise_batch(cfg_digest, 0)
        self.spawn_mock.assert_not_called()
        wire_raw = bytes(self.stdout_written)
        receipt = decode_receipt(
            wire_raw,
            expected_sha256=hashlib.sha256(wire_raw).hexdigest(),
            expected=self._expected_ready(0),
        )
        self.assertEqual(receipt["done"]["status"], "input_error")
        self.assertFalse(receipt["payload_received"])

    def test_grant_cannot_extend_anchor_and_tightened_grant_times_out(self):
        boot_raw, cfg_digest = self._bootstrap_frames()
        self.stdin_chunks = [boot_raw]
        self.backend_chunks[201] = []
        self.backend_proc.returncode = None

        def kill_sets_rc(pgid, signum):
            if signum != 0:
                self.backend_proc.returncode = -9
            raise ProcessLookupError()

        self.killpg_mock.side_effect = kill_sets_rc
        self.ready_callback = lambda ready_raw: self.stdin_chunks.append(
            self._grant_frames(ready_raw, granted_work_ns=500_000_000)
        )
        remote_module.supervise_batch(cfg_digest, 0)
        wire_raw = bytes(self.stdout_written)
        receipt = decode_receipt(
            wire_raw,
            expected_sha256=hashlib.sha256(wire_raw).hexdigest(),
            expected=self._expected_ready(0),
        )
        self.assertEqual(receipt["done"]["status"], "timeout")
        self.assertEqual(receipt["done"]["backend"]["status"], "timeout")

    def test_backend_launch_oserror_records_launch_failed_with_payload_received(self):
        boot_raw, cfg_digest = self._bootstrap_frames()
        self.stdin_chunks = [boot_raw]
        self.ready_callback = lambda ready_raw: self.stdin_chunks.append(
            self._grant_frames(ready_raw)
        )

        def fail_spawn(*args, **kwargs):
            raise OSError("execve failed")

        self.spawn_handler = fail_spawn
        remote_module.supervise_batch(cfg_digest, 0)
        wire_raw = bytes(self.stdout_written)
        receipt = decode_receipt(
            wire_raw,
            expected_sha256=hashlib.sha256(wire_raw).hexdigest(),
            expected=self._expected_ready(0),
        )
        self.assertEqual(receipt["done"]["status"], "launch_failed")
        self.assertTrue(receipt["payload_received"])
        self.assertIsNone(receipt["done"]["backend"])
        self.assertIsNone(receipt["done"]["backend_pid"])

    def test_backend_stderr_overflow_sets_output_limit_within_frame_cap(self):
        boot_raw, cfg_digest = self._bootstrap_frames()
        self.stdin_chunks = [boot_raw]
        self.backend_chunks[202] = [b"e" * 131073, b""]
        self.ready_callback = lambda ready_raw: self.stdin_chunks.append(
            self._grant_frames(ready_raw)
        )
        remote_module.supervise_batch(cfg_digest, 0)
        wire_raw = bytes(self.stdout_written)
        decoded_frames = decode_wire(
            wire_raw, expected_sha256=hashlib.sha256(wire_raw).hexdigest()
        )
        self.assertEqual(len(decoded_frames["backend_stderr_raw"]), 131072)
        receipt = decode_receipt(
            wire_raw,
            expected_sha256=hashlib.sha256(wire_raw).hexdigest(),
            expected=self._expected_ready(0),
        )
        self.assertEqual(receipt["done"]["status"], "output_limit")

    def test_surviving_backend_group_prevents_completed_status(self):
        boot_raw, cfg_digest = self._bootstrap_frames()
        self.stdin_chunks = [boot_raw]
        self.group_alive = True
        self.group_survives = True
        self.ready_callback = lambda ready_raw: self.stdin_chunks.append(
            self._grant_frames(ready_raw)
        )
        remote_module.supervise_batch(cfg_digest, 0)
        wire_raw = bytes(self.stdout_written)
        receipt = decode_receipt(
            wire_raw,
            expected_sha256=hashlib.sha256(wire_raw).hexdigest(),
            expected=self._expected_ready(0),
        )
        self.assertEqual(receipt["done"]["status"], "program_error")

    def test_cancellation_during_backend_reaps_and_reraises_without_forged_done(self):
        boot_raw, cfg_digest = self._bootstrap_frames()
        self.stdin_chunks = [boot_raw]
        interruption = KeyboardInterrupt("remote sigint")
        self.backend_chunks[201] = [interruption]
        self.group_alive = True
        self.ready_callback = lambda ready_raw: self.stdin_chunks.append(
            self._grant_frames(ready_raw)
        )
        with self.assertRaises(KeyboardInterrupt) as caught:
            remote_module.supervise_batch(cfg_digest, 0)
        self.assertIs(caught.exception, interruption)
        self.assertFalse(self.group_alive)
        with self.assertRaises(ValueError):
            decode_wire(
                bytes(self.stdout_written),
                expected_sha256=hashlib.sha256(bytes(self.stdout_written)).hexdigest(),
            )

    def test_cancellation_before_backend_emits_cancelled_done_and_reraises(self):
        boot_raw, cfg_digest = self._bootstrap_frames()
        interruption = KeyboardInterrupt("before payload")
        self.stdin_chunks = [boot_raw]
        self.ready_callback = lambda ready_raw: self.stdin_chunks.append(interruption)
        with self.assertRaises(KeyboardInterrupt) as caught:
            remote_module.supervise_batch(cfg_digest, 0)
        self.assertIs(caught.exception, interruption)
        self.spawn_mock.assert_not_called()
        wire_raw = bytes(self.stdout_written)
        receipt = decode_receipt(
            wire_raw,
            expected_sha256=hashlib.sha256(wire_raw).hexdigest(),
            expected=self._expected_ready(0),
        )
        self.assertEqual(receipt["done"]["status"], "cancelled")
        self.assertIsNone(receipt["done"]["backend"])

    def test_ready_deadline_exceeded_fails_before_emitting_ready(self):
        boot_raw, cfg_digest = self._bootstrap_frames()

        def slow_read(descriptor, count):
            if descriptor == 190 and self.stdin_chunks:
                self.clock.sleep(11.0)
            return self._io_read(descriptor, count)

        self.stdin_chunks = [boot_raw]
        with mock.patch.object(remote_module, "_io_read", side_effect=slow_read):
            with self.assertRaises(TimeoutError):
                remote_module.supervise_batch(cfg_digest, 0)
        self.assertEqual(bytes(self.stdout_written), b"")
        self.spawn_mock.assert_not_called()

    def test_second_batch_reuses_owned_stage_root_and_rejects_config_drift(self):
        bundle_one = sample_bundle_raw(run_id=RUN_IDS[1])
        self.launch_config["batches"][1]["bundle_sha256"] = hashlib.sha256(bundle_one).hexdigest()
        self.launch_config["batches"][1]["bundle_bytes"] = len(bundle_one)
        self.config_sha256 = digest(self.launch_config)

        boot_zero, cfg_digest = self._bootstrap_frames(batch_index=0)
        self.stdin_chunks = [boot_zero]
        self.ready_callback = lambda ready_raw: self.stdin_chunks.append(
            self._grant_frames(ready_raw)
        )
        self.assertEqual(remote_module.supervise_batch(cfg_digest, 0), 0)

        self.stdout_written.clear()
        self.backend_proc = ArtificialBackendProcess()
        self.backend_chunks = {
            201: [b'{"kind":"ready"}\n{"kind":"done"}\n', b""],
            202: [b"", b""],
        }
        boot_one, cfg_digest_one = self._bootstrap_frames(batch_index=1)
        self.stdin_chunks = [boot_one]
        self.ready_callback = lambda ready_raw: self.stdin_chunks.append(
            self._grant_frames(ready_raw, bundle_raw=bundle_one)
        )
        self.assertEqual(remote_module.supervise_batch(cfg_digest_one, 1), 0)
        wire_raw = bytes(self.stdout_written)
        receipt = decode_receipt(
            wire_raw,
            expected_sha256=hashlib.sha256(wire_raw).hexdigest(),
            expected=self._expected_ready(1),
        )
        self.assertEqual(receipt["ready"]["batch_index"], 1)
        self.assertEqual(receipt["done"]["status"], "completed")

        drifted = dict(self.launch_config)
        drifted["preflight_receipt_sha256"] = "8" * 64
        boot_two, drifted_digest = self._bootstrap_frames(
            batch_index=2, config_override=drifted
        )
        self.stdout_written.clear()
        self.stdin_chunks = [boot_two]
        with self.assertRaises(ValueError):
            remote_module.supervise_batch(drifted_digest, 2)

    def test_symlink_parent_or_stage_root_rejected_before_ready(self):
        real_dir = self.base_dir / "real"
        real_dir.mkdir(mode=0o700)
        link_parent = self.base_dir / "link"
        link_parent.symlink_to(real_dir, target_is_directory=True)
        bad_stage_root = str(link_parent / STAGE_ID)
        bad_config = sample_launch_config(
            bad_stage_root, uid=os.getuid(), bundle_raw=self.bundle_raw
        )
        boot_raw, bad_digest = self._bootstrap_frames(config_override=bad_config)
        self.stdin_chunks = [boot_raw]
        with self.assertRaises((OSError, ValueError)):
            remote_module.supervise_batch(bad_digest, 0)
        self.assertEqual(bytes(self.stdout_written), b"")
        self.spawn_mock.assert_not_called()

    def test_unreaped_backend_child_sets_returncode_none_and_noncompleted(self):
        boot_raw, cfg_digest = self._bootstrap_frames()
        self.stdin_chunks = [boot_raw]
        self.backend_proc.returncode = 0
        self.backend_proc.wait_error = subprocess.TimeoutExpired("backend", 0.2)
        self.ready_callback = lambda ready_raw: self.stdin_chunks.append(
            self._grant_frames(ready_raw)
        )
        remote_module.supervise_batch(cfg_digest, 0)
        wire_raw = bytes(self.stdout_written)
        receipt = decode_receipt(
            wire_raw,
            expected_sha256=hashlib.sha256(wire_raw).hexdigest(),
            expected=self._expected_ready(0),
        )
        self.assertFalse(receipt["done"]["backend"]["process_reaped"])
        self.assertIsNone(receipt["done"]["backend"]["returncode"])
        self.assertEqual(receipt["done"]["status"], "program_error")

    def test_clock_rollback_fails_closed(self):
        boot_raw, cfg_digest = self._bootstrap_frames()
        self.stdin_chunks = [boot_raw]

        def rollback_read(descriptor, count):
            result = self._io_read(descriptor, count)
            if descriptor == 201 and result:
                self.clock.nanoseconds -= 5_000_000_000
            return result

        self.ready_callback = lambda ready_raw: self.stdin_chunks.append(
            self._grant_frames(ready_raw)
        )
        with mock.patch.object(remote_module, "_io_read", side_effect=rollback_read):
            exit_code = remote_module.supervise_batch(cfg_digest, 0)
        self.assertNotEqual(exit_code, 0)

    def test_aggregate_wire_budget_exceeded_refuses_truncated_done(self):
        boot_raw, cfg_digest = self._bootstrap_frames()
        self.stdin_chunks = [boot_raw]
        self.backend_chunks[201] = [b"a" * 8388590, b""]
        self.backend_chunks[202] = [b"b" * 1024, b""]
        self.ready_callback = lambda ready_raw: self.stdin_chunks.append(
            self._grant_frames(ready_raw)
        )
        exit_code = remote_module.supervise_batch(cfg_digest, 0)
        self.assertNotEqual(exit_code, 0)
        with self.assertRaises(ValueError):
            decode_wire(
                bytes(self.stdout_written),
                expected_sha256=hashlib.sha256(bytes(self.stdout_written)).hexdigest(),
            )

    def test_clock_rollback_with_live_backend_still_kills_reaps_and_closes_streams(self):
        boot_raw, cfg_digest = self._bootstrap_frames()
        self.stdin_chunks = [boot_raw]
        self.group_alive = True
        self.backend_proc.returncode = None

        def kill_reaps(pgid, signum):
            if not self.group_alive:
                raise ProcessLookupError()
            if signum != 0:
                self.group_alive = False
                self.backend_proc.returncode = -signum

        self.killpg_mock.side_effect = kill_reaps

        def rollback_read(descriptor, count):
            result = self._io_read(descriptor, count)
            if descriptor == 201 and result:
                self.clock.nanoseconds -= 5_000_000_000
            return result

        self.ready_callback = lambda ready_raw: self.stdin_chunks.append(
            self._grant_frames(ready_raw)
        )
        with mock.patch.object(remote_module, "_io_read", side_effect=rollback_read):
            exit_code = remote_module.supervise_batch(cfg_digest, 0)
        self.assertEqual(exit_code, 2)
        self.assertFalse(self.group_alive)
        self.assertTrue(self.backend_proc.stdout.closed)
        self.assertTrue(self.backend_proc.stderr.closed)

    def test_cancellation_during_cleanup_sleep_escalates_sigkill_and_closes_streams(self):
        boot_raw, cfg_digest = self._bootstrap_frames()
        self.stdin_chunks = [boot_raw]
        self.group_alive = True
        self.backend_proc.returncode = None
        cleanup_interrupt = KeyboardInterrupt("interrupt inside cleanup")
        signals_sent = []

        def stubborn_group(pgid, signum):
            if signum != 0:
                signals_sent.append(signum)
            if signum == signal.SIGKILL:
                self.group_alive = False
                self.backend_proc.returncode = -9
            if not self.group_alive:
                raise ProcessLookupError()

        def interrupt_first_sleep(seconds):
            self.clock.sleep(seconds)
            if signal.SIGTERM in signals_sent and signal.SIGKILL not in signals_sent:
                raise cleanup_interrupt

        self.killpg_mock.side_effect = stubborn_group
        self.backend_chunks[201] = [OSError("read failure")]
        self.ready_callback = lambda ready_raw: self.stdin_chunks.append(
            self._grant_frames(ready_raw)
        )
        with mock.patch.object(remote_module.time, "sleep", side_effect=interrupt_first_sleep):
            with self.assertRaises(KeyboardInterrupt) as caught:
                remote_module.supervise_batch(cfg_digest, 0)
        self.assertIs(caught.exception, cleanup_interrupt)
        self.assertIn(signal.SIGKILL, signals_sent)
        self.assertFalse(self.group_alive)
        self.assertTrue(self.backend_proc.stdout.closed)
        self.assertTrue(self.backend_proc.stderr.closed)

    def test_abort_write_or_batch_fd_close_failure_preserves_primary_cancellation(self):
        boot_raw, cfg_digest = self._bootstrap_frames()
        self.stdin_chunks = [boot_raw]
        primary_cancel = KeyboardInterrupt("primary remote cancel")
        self.backend_chunks[201] = [primary_cancel]
        self.ready_callback = lambda ready_raw: self.stdin_chunks.append(
            self._grant_frames(ready_raw)
        )
        original_write = remote_module._write_exclusive
        original_close = remote_module.os.close

        abort_attempted = []
        closed_once = []

        def fail_abort_write(directory_fd, filename, raw):
            if filename == "supervisor-abort.json":
                abort_attempted.append(filename)
                raise OSError("disk full on abort")
            return original_write(directory_fd, filename, raw)

        def fail_batch_close(descriptor):
            is_dir = stat.S_ISDIR(os.fstat(descriptor).st_mode)
            original_close(descriptor)
            if abort_attempted and is_dir and not closed_once:
                closed_once.append(descriptor)
                raise OSError("batch_fd close error")

        with mock.patch.object(remote_module, "_write_exclusive", side_effect=fail_abort_write):
            with mock.patch.object(remote_module.os, "close", side_effect=fail_batch_close):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    remote_module.supervise_batch(cfg_digest, 0)
        self.assertIs(caught.exception, primary_cancel)

    def test_initial_reap_stall_past_deadline_still_sends_sigkill(self):
        boot_raw, cfg_digest = self._bootstrap_frames()
        self.stdin_chunks = [boot_raw]
        self.group_alive = True
        self.backend_proc.returncode = None
        signals_sent = []

        def stubborn_group(pgid, signum):
            if signum != 0:
                signals_sent.append(signum)
            if signum == signal.SIGKILL:
                self.group_alive = False
                self.backend_proc.returncode = -9
            if not self.group_alive:
                raise ProcessLookupError()

        wait_calls = []

        def slow_initial_wait(timeout=None):
            wait_calls.append(timeout)
            if len(wait_calls) == 1:
                self.clock.sleep(2.5)
                raise subprocess.TimeoutExpired("backend", timeout)
            return self.backend_proc.returncode

        self.killpg_mock.side_effect = stubborn_group
        self.backend_proc.wait = slow_initial_wait
        self.backend_chunks[201] = [OSError("trigger cleanup")]
        self.ready_callback = lambda ready_raw: self.stdin_chunks.append(
            self._grant_frames(ready_raw)
        )
        remote_module.supervise_batch(cfg_digest, 0)
        self.assertIn(signal.SIGKILL, signals_sent)
        self.assertFalse(self.group_alive)

    def test_pre_backend_cancel_preserved_when_done_write_or_stdout_fails(self):
        boot_raw, cfg_digest = self._bootstrap_frames()
        primary_cancel = KeyboardInterrupt("cancel during grant wait")
        self.stdin_chunks = [boot_raw]
        self.ready_callback = lambda ready_raw: self.stdin_chunks.append(primary_cancel)
        original_io_write = self._io_write
        write_calls = []

        def fail_trailing_stdout(descriptor, raw):
            write_calls.append(descriptor)
            if descriptor == 191 and len(write_calls) >= 2:
                raise BrokenPipeError("ssh stdout closed")
            return original_io_write(descriptor, raw)

        with mock.patch.object(remote_module, "_io_write", side_effect=fail_trailing_stdout):
            with self.assertRaises(KeyboardInterrupt) as caught:
                remote_module.supervise_batch(cfg_digest, 0)
        self.assertIs(caught.exception, primary_cancel)


if __name__ == "__main__":
    unittest.main()
