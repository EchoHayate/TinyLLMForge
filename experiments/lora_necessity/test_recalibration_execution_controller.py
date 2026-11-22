import ast
import copy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tempfile
import tokenize
import types
import unittest
from unittest import mock

from recalibration_data_contract import FILE_PINS
from recalibration_execution_package import BACKEND_SHA256, SOURCE_PINS
from recalibration_execution_remote import (
    FRAME_HEADER,
    STDIN_MAGIC,
    WIRE_MAGIC,
    decode_control_frame,
    supervise_batch,
)
from recalibration_validation import WORKER_SHA256, build_batches, build_schedule
from validation_contract import digest, encoded


REPO_ROOT = Path(__file__).resolve().parent
PEER_NAMES = (
    *SOURCE_PINS.keys(),
    "recalibration_execution_package.py",
    "recalibration_execution_wire.py",
    "recalibration_execution_evidence.py",
    "recalibration_execution_remote.py",
)

with mock.patch("subprocess.Popen") as import_spawn, mock.patch("os.killpg") as import_killpg:
    try:
        controller_module = importlib.import_module("recalibration_execution_controller")
    except ModuleNotFoundError as import_error:
        if import_error.name != "recalibration_execution_controller":
            raise
        controller_module = None
    assert not import_spawn.called
    assert not import_killpg.called


class ArtificialClock:
    def __init__(self):
        self.nanoseconds = 2_000_000_000_000

    def monotonic_ns(self):
        self.nanoseconds += 1_000_000
        return self.nanoseconds

    def monotonic(self):
        return self.monotonic_ns() / 1_000_000_000.0

    def sleep(self, seconds):
        self.nanoseconds += int(max(0.0, seconds) * 1_000_000_000)


class ArtificialPipe:
    def __init__(self, descriptor, on_close=None):
        self.descriptor = descriptor
        self.closed = False
        self.on_close = on_close

    def fileno(self):
        if self.closed:
            raise ValueError("I/O operation on closed pipe")
        return self.descriptor

    def close(self):
        if not self.closed:
            self.closed = True
            if self.on_close is not None:
                self.on_close()


class ArtificialSshProcess:
    def __init__(self, pid=777000):
        self.pid = pid
        self.stdin = None
        self.stdout = ArtificialPipe(301)
        self.stderr = ArtificialPipe(302)
        self.returncode = 0
        self.wait_error = None

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
        self.error = None
        self.idle = False

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
        self.registered.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_files = {
            name: (REPO_ROOT / "data" / (
                name if name == "supervision_v1.json"
                else "recalibration_v2_r1/" + name
            )).read_bytes()
            for name in FILE_PINS
        }
        cls.worker_raw = (REPO_ROOT / "validation_task_worker.py").read_bytes()
        cls.backend_raw = (REPO_ROOT / "validation_remote.py").read_bytes()
        cls.full_source_raw = {name: (REPO_ROOT / name).read_bytes() for name in PEER_NAMES}
        cls.plan = build_schedule(cls.raw_files)
        cls.stage_id = "ab" * 16
        cls.run_ids = [("%02x" % index) + ("cd" * 15) for index in range(15)]
        cls.batches = build_batches(cls.raw_files, cls.worker_raw, cls.run_ids)

    def setUp(self):
        self.spawn_mock = mock.patch("subprocess.Popen").start()
        self.killpg_mock = mock.patch("os.killpg").start()
        self.addCleanup(mock.patch.stopall)
        self.assertIsNotNone(controller_module, "controller implementation is absent")
        self.tempdir = tempfile.TemporaryDirectory(prefix="tlf-ctrl-test-")
        self.addCleanup(self.tempdir.cleanup)
        self.base_dir = Path(self.tempdir.name).resolve()
        self.local_parent = self.base_dir / "local"
        self.remote_parent = self.base_dir / "remote"
        self.local_parent.mkdir(mode=0o700)
        self.remote_parent.mkdir(mode=0o700)
        self.local_stage_root = str(self.local_parent / self.stage_id)
        self.remote_stage_root = str(self.remote_parent / self.stage_id)
        self.ssh_argv_prefix = ["/usr/bin/ssh", "-T", "rl-server"]
        self.ssh_environment = {
            "HOME": "/Users/bytedance",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        }
        self.runtime = {
            "hostname": "target-node",
            "uid": os.getuid(),
            "python": "3.12.13",
            "python_root": "/runtime/env",
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
        self.preflight_raw = encoded({
            "kind": "LORA_RECALIBRATION_LIVE_PREFLIGHT_RECEIPT",
            "version": 1,
            "status": "PREFLIGHT_PASSED",
            "probes_passed": True,
            "remote_python": "/runtime/env/bin/python3.12",
            "expected_runtime": self.runtime,
        })
        self.scope_raw = encoded({
            "kind": "LORA_RECALIBRATION_STAGE_EXECUTION_SCOPE",
            "version": 1,
            "stage_id": self.stage_id,
            "run_ids": list(self.run_ids),
            "local_stage_root": self.local_stage_root,
            "remote_stage_root": self.remote_stage_root,
            "ssh_argv_prefix_sha256": digest(self.ssh_argv_prefix),
            "ssh_environment_sha256": digest(self.ssh_environment),
            "preflight_receipt_sha256": hashlib.sha256(self.preflight_raw).hexdigest(),
        })
        self.clock = ArtificialClock()
        self.group_alive = False
        self.killpg_mock.side_effect = self._handle_killpg
        self.active_batch_index = None
        self.stdin_stream = bytearray()
        self.stdout_chunks = []
        self.stderr_chunks = []
        self.current_ssh_proc = None
        self.dispatched_bundles = []
        self.custom_batch_mode = {}
        self.ready_delay_sec = 0.0
        owner = self

        class OwnedArtificialSshPopen(ArtificialSshProcess):
            def __init__(instance, *args, **kwargs):
                instance.pid = None
                instance.stdin = None
                instance.stdout = None
                instance.stderr = None
                supplied = owner._spawn_ssh(*args, **kwargs)
                instance.__dict__.update(supplied.__dict__)

        mock.patch.object(controller_module.subprocess, "Popen", OwnedArtificialSshPopen).start()
        mock.patch.object(controller_module.time, "monotonic_ns", self.clock.monotonic_ns).start()
        mock.patch.object(controller_module.time, "monotonic", self.clock.monotonic).start()
        mock.patch.object(controller_module.time, "sleep", self.clock.sleep).start()
        mock.patch.object(
            controller_module.selectors,
            "DefaultSelector",
            side_effect=lambda: ArtificialSelector(self.clock),
        ).start()
        mock.patch.object(controller_module.os, "set_blocking").start()
        mock.patch.object(controller_module, "_io_read", side_effect=self._io_read).start()
        mock.patch.object(controller_module, "_io_write", side_effect=self._io_write).start()

    def _handle_killpg(self, pgid, signum):
        if not self.group_alive:
            raise ProcessLookupError()
        if signum != 0:
            self.group_alive = False
            if self.current_ssh_proc is not None and self.current_ssh_proc.returncode is None:
                self.current_ssh_proc.returncode = -9

    def _batch_backend_jsonl(self, batch_index, *, mismatch_first_ref=False, row_error=False):
        batch = self.batches[batch_index]
        events = [{
            "kind": "ready",
            "run_id": batch["run_id"],
            "bundle_sha256": batch["bundle_sha256"],
            "worker_sha256": batch["worker_sha256"],
            "runtime": copy.deepcopy(self.runtime),
        }]
        rows_slice = self.plan["rows"][batch["start"]:batch["stop"]]
        for ordinal, row in enumerate(rows_slice):
            if row_error and ordinal == 0:
                events.append({
                    "kind": "row",
                    "run_id": batch["run_id"],
                    "ordinal": ordinal,
                    "request_sha256": row["request_sha256"],
                    "result": {
                        "status": "program_error",
                        "returncode": 1,
                        "stdout_hex": "",
                        "stderr_hex": b"err".hex(),
                        "elapsed_ns": 50,
                        "process_reaped": True,
                        "stdin_complete": True,
                    },
                })
                events.append({
                    "kind": "done",
                    "run_id": batch["run_id"],
                    "received": 1,
                    "stopped": True,
                    "elapsed_ns": 80,
                })
                return b"".join(encoded(ev) + b"\n" for ev in events)
            outcome = row["witness_prediction"]
            if outcome is None:
                outcome = row["expected"]
                if mismatch_first_ref and ordinal == 0:
                    outcome = {"kind": "return", "value": "__unexpected_mismatch__"}
            events.append({
                "kind": "row",
                "run_id": batch["run_id"],
                "ordinal": ordinal,
                "request_sha256": row["request_sha256"],
                "result": {
                    "status": "completed",
                    "returncode": 0,
                    "stdout_hex": encoded(outcome).hex(),
                    "stderr_hex": "",
                    "elapsed_ns": 50,
                    "process_reaped": True,
                    "stdin_complete": True,
                },
            })
        events.append({
            "kind": "done",
            "run_id": batch["run_id"],
            "received": 32,
            "stopped": False,
            "elapsed_ns": 100,
        })
        return b"".join(encoded(ev) + b"\n" for ev in events)

    def _parse_frames_so_far(self):
        parsed = []
        offset = 0
        while len(self.stdin_stream) - offset >= FRAME_HEADER.size:
            magic, channel, size = FRAME_HEADER.unpack_from(self.stdin_stream, offset)
            if magic != STDIN_MAGIC or len(self.stdin_stream) < offset + FRAME_HEADER.size + size:
                break
            payload = bytes(
                self.stdin_stream[offset + FRAME_HEADER.size:offset + FRAME_HEADER.size + size]
            )
            parsed.append((channel, payload))
            offset += FRAME_HEADER.size + size
        return parsed, offset == len(self.stdin_stream)

    def _emit_ready_from_bootstrap(self):
        frames, complete = self._parse_frames_so_far()
        if len(frames) < 2 or not complete:
            return
        if self.ready_emitted:
            return
        self.ready_emitted = True
        if self.ready_delay_sec > 0:
            self.clock.sleep(self.ready_delay_sec)
        bootstrap = json.loads(frames[0][1].decode("utf-8"))
        launch_config = bootstrap["launch_config"]
        batch_index = bootstrap["batch_index"]
        batch_meta = launch_config["batches"][batch_index]
        mode = self.custom_batch_mode.get(batch_index, "ok")
        ready_obj = {
            "kind": "LORA_RECALIBRATION_READY",
            "version": 1,
            "stage_id": launch_config["stage_id"],
            "batch_index": batch_index,
            "run_id": "0" * 32 if mode == "bad_ready" else batch_meta["run_id"],
            "preparation_sha256": launch_config["preparation_sha256"],
            "package_sha256": bootstrap["package_sha256"],
            "config_sha256": bootstrap["config_sha256"],
            "bundle_sha256": batch_meta["bundle_sha256"],
            "backend_sha256": BACKEND_SHA256,
            "worker_sha256": WORKER_SHA256,
            "hostname": self.runtime["hostname"],
            "uid": self.runtime["uid"],
            "stage_root": launch_config["remote_stage_root"],
            "batch_root": batch_meta["batch_root"],
            "expected_runtime_sha256": digest(self.runtime),
            "ready_elapsed_ns": 5_000_000,
        }
        self.last_ready_raw = encoded(ready_obj)
        self.last_ready_frame = (
            FRAME_HEADER.pack(WIRE_MAGIC, 1, len(self.last_ready_raw)) + self.last_ready_raw
        )
        self.stdout_chunks.append(self.last_ready_frame)

    def _on_ssh_stdin_closed(self):
        frames, complete = self._parse_frames_so_far()
        batch_index = self.active_batch_index
        mode = self.custom_batch_mode.get(batch_index, "ok")
        bootstrap = json.loads(frames[0][1].decode("utf-8"))
        launch_config = bootstrap["launch_config"]
        batch_meta = launch_config["batches"][batch_index]

        if len(frames) == 4 and complete:
            self.dispatched_bundles.append((batch_index, frames[3][1]))

        if mode == "missing_done":
            self.stdout_chunks.append(b"")
            self.stderr_chunks.append(b"")
            return
        if mode == "wire_overflow":
            self.stdout_chunks.append(b"x" * 8388609)
            self.stdout_chunks.append(b"")
            self.stderr_chunks.append(b"")
            return
        if mode == "cancel":
            self.group_alive = True
            self.current_ssh_proc.returncode = None
            self.stdout_chunks.append(KeyboardInterrupt("ctrl-c during batch"))
            return

        if len(frames) < 4 or not complete:
            backend_stdout = b""
            backend_stderr = b""
            done_status = "input_error"
            payload_received = False
            backend_pid = None
            backend_obj = None
        else:
            payload_received = True
            if mode == "supervisor_timeout":
                backend_stdout = b""
                backend_stderr = b""
                done_status = "timeout"
                backend_pid = 555000 + batch_index
                backend_obj = {
                    "status": "timeout",
                    "returncode": -9,
                    "elapsed_ns": 95_000_000_000,
                    "process_reaped": True,
                    "stdin_complete": True,
                }
            else:
                backend_stdout = self._batch_backend_jsonl(
                    batch_index,
                    mismatch_first_ref=(mode == "semantic_mismatch"),
                    row_error=(mode == "row_program_error"),
                )
                backend_stderr = b""
                done_status = "completed"
                backend_pid = 555000 + batch_index
                backend_obj = {
                    "status": "completed",
                    "returncode": 0,
                    "elapsed_ns": 200,
                    "process_reaped": True,
                    "stdin_complete": True,
                }

        diag_raw = encoded({"kind": "LORA_RECALIBRATION_DIAGNOSTIC", "status": done_status})
        frame_two = FRAME_HEADER.pack(WIRE_MAGIC, 2, len(backend_stdout)) + backend_stdout
        frame_three = FRAME_HEADER.pack(WIRE_MAGIC, 3, len(backend_stderr)) + backend_stderr
        frame_four = FRAME_HEADER.pack(WIRE_MAGIC, 4, len(diag_raw)) + diag_raw
        done_obj = {
            "kind": "LORA_RECALIBRATION_DONE",
            "version": 1,
            "stage_id": launch_config["stage_id"],
            "batch_index": batch_index,
            "run_id": self.last_ready_raw and json.loads(self.last_ready_raw)["run_id"],
            "package_sha256": bootstrap["package_sha256"],
            "config_sha256": bootstrap["config_sha256"],
            "ready_sha256": hashlib.sha256(self.last_ready_raw).hexdigest(),
            "payload_sha256": batch_meta["bundle_sha256"] if payload_received else None,
            "prior_frames_sha256": [
                hashlib.sha256(raw_f).hexdigest()
                for raw_f in (self.last_ready_frame, frame_two, frame_three, frame_four)
            ],
            "status": done_status,
            "payload_received": payload_received,
            "backend_pid": backend_pid,
            "backend": backend_obj,
            "work_elapsed_ns": 500,
            "cleanup_elapsed_ns": 20,
        }
        done_raw = encoded(done_obj)
        frame_five = FRAME_HEADER.pack(WIRE_MAGIC, 5, len(done_raw)) + done_raw
        self.stdout_chunks.extend([frame_two + frame_three + frame_four + frame_five, b""])
        self.stderr_chunks.append(b"")

    def _spawn_ssh(self, argv, **kwargs):
        self.spawn_mock(argv, **kwargs)
        batch_index = int(argv[-1].rsplit(" ", 1)[-1])
        self.active_batch_index = batch_index
        self.stdin_stream = bytearray()
        self.stdout_chunks = []
        self.stderr_chunks = []
        self.ready_emitted = False
        self.last_ready_raw = None
        self.last_ready_frame = None
        proc = ArtificialSshProcess(pid=777000 + batch_index)
        mode_now = self.custom_batch_mode.get(batch_index)
        if mode_now == "early_eof_with_stderr":
            proc.stdin = ArtificialPipe(300, on_close=None)
            proc.returncode = 255
            self.stdout_chunks = [b""]
            self.stderr_chunks = [b"Permission denied\n", b""]
        elif mode_now == "broken_pipe_banner":
            proc.stdin = ArtificialPipe(300, on_close=None)
            proc.returncode = 255
            self.stdout_chunks = [b"NOT_LRV01_BANNER_HEADER_BYTES\n", b""]
            self.stderr_chunks = [b"auth error\n", b""]
        else:
            proc.stdin = ArtificialPipe(300, on_close=self._on_ssh_stdin_closed)
            if mode_now == "ssh_exit_nonzero":
                proc.returncode = 255
        self.current_ssh_proc = proc
        return proc

    def _io_write(self, descriptor, raw):
        self.assertEqual(descriptor, 300)
        if self.current_ssh_proc is not None and self.current_ssh_proc.stdin.closed:
            import errno
            raise OSError(errno.EBADF, "Bad file descriptor")
        if self.custom_batch_mode.get(self.active_batch_index) == "broken_pipe_banner":
            raise BrokenPipeError("ssh closed stdin")
        self.stdin_stream.extend(raw)
        self._emit_ready_from_bootstrap()
        return len(raw)

    def _io_read(self, descriptor, count):
        source = self.stdout_chunks if descriptor == 301 else self.stderr_chunks
        if not source:
            raise BlockingIOError()
        item = source[0]
        if isinstance(item, BaseException):
            source.pop(0)
            raise item
        if item == b"":
            source.pop(0)
            return b""
        if len(item) > count:
            source[0] = item[count:]
            return item[:count]
        source.pop(0)
        return item

    def _build_pkg(self):
        return controller_module.build_launch_package(
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            self.full_source_raw,
            self.preflight_raw,
            self.scope_raw,
            stage_id=self.stage_id,
            run_ids=self.run_ids,
            ssh_argv_prefix=self.ssh_argv_prefix,
            ssh_environment=self.ssh_environment,
            local_stage_root=self.local_stage_root,
            remote_stage_root=self.remote_stage_root,
        )

    def test_style_and_forbidden_imports(self):
        for filename in (
            "recalibration_execution_controller.py",
            "test_recalibration_execution_controller.py",
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
        self.assertNotIn("validation_remote", sys.modules)
        self.assertNotIn("validation_task_worker", sys.modules)

    def test_launch_package_determinism_no_secret_leak_and_false_flags(self):
        pkg_one = self._build_pkg()
        pkg_two = self._build_pkg()
        self.assertEqual(encoded(pkg_one), encoded(pkg_two))
        validated = controller_module.validate_launch_package(
            pkg_one,
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            self.full_source_raw,
            self.preflight_raw,
            self.scope_raw,
        )
        self.assertEqual(
            validated["public"]["package_sha256"],
            pkg_one["public"]["package_sha256"],
        )
        for flag in (
            "run_authorized",
            "trusted_collection_accepted",
            "dynamic_checks_passed",
            "benchmark_eligible",
            "formal_evaluation_ready",
            "net_benefit_established",
            "isolated",
        ):
            self.assertIs(pkg_one["public"][flag], False)
        config_text = json.dumps(pkg_one["public"]["launch_config"])
        self.assertNotIn("witness_prediction", config_text)
        self.assertNotIn("reference_source", config_text)
        for argv_entry in pkg_one["private"]["ssh_argvs"]:
            joined = " ".join(argv_entry)
            self.assertNotIn("LORA_REMOTE_VALIDATION_BUNDLE", joined)
        with self.assertRaises(TypeError):
            controller_module.run_stage(
                pkg_one,
                self.raw_files,
                self.worker_raw,
                self.backend_raw,
                self.full_source_raw,
                self.preflight_raw,
                self.scope_raw,
                approved=True,
            )

    def test_package_rejects_preflight_scope_or_source_tampering(self):
        bad_preflight = encoded({
            **json.loads(self.preflight_raw.decode("utf-8")),
            "probes_passed": False,
        })
        bad_scope = encoded({
            **json.loads(self.scope_raw.decode("utf-8")),
            "preflight_receipt_sha256": hashlib.sha256(bad_preflight).hexdigest(),
        })
        with self.assertRaises(ValueError):
            controller_module.build_launch_package(
                self.raw_files,
                self.worker_raw,
                self.backend_raw,
                self.full_source_raw,
                bad_preflight,
                bad_scope,
                stage_id=self.stage_id,
                run_ids=self.run_ids,
                ssh_argv_prefix=self.ssh_argv_prefix,
                ssh_environment=self.ssh_environment,
                local_stage_root=self.local_stage_root,
                remote_stage_root=self.remote_stage_root,
            )
        tampered_sources = dict(self.full_source_raw)
        tampered_sources["pilot_protocol.py"] = b"tampered"
        with self.assertRaises(ValueError):
            controller_module.build_launch_package(
                self.raw_files,
                self.worker_raw,
                self.backend_raw,
                tampered_sources,
                self.preflight_raw,
                self.scope_raw,
                stage_id=self.stage_id,
                run_ids=self.run_ids,
                ssh_argv_prefix=self.ssh_argv_prefix,
                ssh_environment=self.ssh_environment,
                local_stage_root=self.local_stage_root,
                remote_stage_root=self.remote_stage_root,
            )

    def test_full_fifteen_batch_stage_completes_and_persists_all_480_items(self):
        pkg = self._build_pkg()
        result = controller_module.run_stage(
            pkg,
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            self.full_source_raw,
            self.preflight_raw,
            self.scope_raw,
        )
        self.assertEqual(self.spawn_mock.call_count, 15)
        self.assertEqual(len(self.dispatched_bundles), 15)
        self.assertTrue(result["binding_complete"])
        self.assertTrue(result["collection_complete"])
        self.assertIs(result["trusted_collection_accepted"], False)
        self.assertIs(result["dynamic_checks_passed"], False)
        self.assertIs(result["run_authorized"], False)
        evidence = result["evidence"]
        self.assertTrue(evidence["collection_complete"])
        self.assertTrue(evidence["receipt_binding_consistent"])
        self.assertTrue(evidence["content_report"]["matrix_complete"])
        self.assertTrue(evidence["content_report"]["data_checks_satisfied"])
        self.assertEqual(evidence["content_report"]["observed"], 480)
        self.assertEqual(evidence["content_report"]["completed"], 480)
        self.assertEqual(evidence["content_report"]["unknown_after_dispatch"], 0)
        self.assertEqual(evidence["content_report"]["not_dispatched"], 0)
        stage_dir = Path(self.local_stage_root)
        self.assertEqual(stat.S_IMODE(stage_dir.stat().st_mode), 0o700)
        self.assertTrue((stage_dir / "execution-evidence.json").is_file())
        self.assertTrue((stage_dir / "controller-summary.json").is_file())
        batch_zero_dir = stage_dir / ("batch-00-" + self.run_ids[0])
        self.assertEqual(stat.S_IMODE(batch_zero_dir.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((batch_zero_dir / "wire.bin").stat().st_mode), 0o600)

    def test_semantic_mismatch_continues_all_15_batches_but_fails_data_checks(self):
        self.custom_batch_mode[0] = "semantic_mismatch"
        pkg = self._build_pkg()
        result = controller_module.run_stage(
            pkg,
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            self.full_source_raw,
            self.preflight_raw,
            self.scope_raw,
        )
        self.assertEqual(self.spawn_mock.call_count, 15)
        self.assertTrue(result["collection_complete"])
        self.assertTrue(result["evidence"]["content_report"]["matrix_complete"])
        self.assertFalse(result["evidence"]["content_report"]["data_checks_satisfied"])

    def test_infrastructure_program_error_stops_immediately_at_batch_zero(self):
        self.custom_batch_mode[0] = "row_program_error"
        pkg = self._build_pkg()
        result = controller_module.run_stage(
            pkg,
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            self.full_source_raw,
            self.preflight_raw,
            self.scope_raw,
        )
        self.assertEqual(self.spawn_mock.call_count, 1)
        self.assertFalse(result["collection_complete"])
        self.assertEqual(result["evidence"]["first_stop"]["batch_index"], 0)
        self.assertEqual(result["evidence"]["first_stop"]["reason"], "worker_program_error")
        self.assertEqual(result["evidence"]["content_report"]["not_dispatched"], 448)

    def test_missing_done_preserves_unknown_after_dispatch_and_stops(self):
        self.custom_batch_mode[0] = "missing_done"
        pkg = self._build_pkg()
        result = controller_module.run_stage(
            pkg,
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            self.full_source_raw,
            self.preflight_raw,
            self.scope_raw,
        )
        self.assertEqual(self.spawn_mock.call_count, 1)
        self.assertFalse(result["collection_complete"])
        self.assertEqual(result["evidence"]["first_stop"]["reason"], "receipt_error")
        self.assertEqual(
            result["evidence"]["content_report"]["batches"][0]["stop_reason"],
            "missing_transport",
        )
        self.assertEqual(result["evidence"]["content_report"]["unknown_after_dispatch"], 32)
        self.assertEqual(result["evidence"]["content_report"]["not_dispatched"], 448)

    def test_invalid_ready_withholds_bundle_payload_and_stops(self):
        self.custom_batch_mode[0] = "bad_ready"
        pkg = self._build_pkg()
        result = controller_module.run_stage(
            pkg,
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            self.full_source_raw,
            self.preflight_raw,
            self.scope_raw,
        )
        self.assertEqual(self.spawn_mock.call_count, 1)
        self.assertEqual(self.dispatched_bundles, [])
        self.assertFalse(result["collection_complete"])
        self.assertEqual(result["evidence"]["first_stop"]["reason"], "receipt_error")

    def test_insufficient_stage_budget_after_ready_withholds_payload(self):
        self.ready_delay_sec = 1710.0
        pkg = self._build_pkg()
        result = controller_module.run_stage(
            pkg,
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            self.full_source_raw,
            self.preflight_raw,
            self.scope_raw,
        )
        self.assertEqual(self.spawn_mock.call_count, 1)
        self.assertEqual(self.dispatched_bundles, [])
        self.assertFalse(result["collection_complete"])

    def test_exclusive_stage_claim_blocks_replay(self):
        pkg = self._build_pkg()
        controller_module.run_stage(
            pkg,
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            self.full_source_raw,
            self.preflight_raw,
            self.scope_raw,
        )
        with self.assertRaises(FileExistsError):
            controller_module.run_stage(
                pkg,
                self.raw_files,
                self.worker_raw,
                self.backend_raw,
                self.full_source_raw,
                self.preflight_raw,
                self.scope_raw,
            )
        self.assertEqual(self.spawn_mock.call_count, 15)

    def test_batch_persistence_failure_prevents_subsequent_batch_dispatch(self):
        pkg = self._build_pkg()
        original_write = controller_module._write_exclusive

        def fail_wire_save(directory_fd, filename, raw):
            if filename == "wire.bin":
                raise OSError("disk full")
            return original_write(directory_fd, filename, raw)

        with mock.patch.object(controller_module, "_write_exclusive", side_effect=fail_wire_save):
            with self.assertRaises(OSError):
                controller_module.run_stage(
                    pkg,
                    self.raw_files,
                    self.worker_raw,
                    self.backend_raw,
                    self.full_source_raw,
                    self.preflight_raw,
                    self.scope_raw,
                )
        self.assertEqual(self.spawn_mock.call_count, 1)

    def test_cancellation_cleans_up_ssh_group_persists_and_reraises_exact_exception(self):
        self.custom_batch_mode[0] = "cancel"
        pkg = self._build_pkg()
        with self.assertRaises(KeyboardInterrupt) as caught:
            controller_module.run_stage(
                pkg,
                self.raw_files,
                self.worker_raw,
                self.backend_raw,
                self.full_source_raw,
                self.preflight_raw,
                self.scope_raw,
            )
        self.assertEqual(str(caught.exception), "ctrl-c during batch")
        self.assertFalse(self.group_alive)
        self.assertEqual(self.spawn_mock.call_count, 1)
        self.assertTrue((Path(self.local_stage_root) / "controller-summary.json").is_file())

    def test_wire_overflow_and_ssh_nonzero_exit_stop_at_batch_zero(self):
        for mode_name in ("wire_overflow", "ssh_exit_nonzero"):
            with self.subTest(mode_name=mode_name):
                sub_dir = self.base_dir / ("sub-" + mode_name)
                sub_dir.mkdir(mode=0o700)
                local_root = str(sub_dir / self.stage_id)
                scope_raw = encoded({
                    **json.loads(self.scope_raw.decode("utf-8")),
                    "local_stage_root": local_root,
                })
                pkg = controller_module.build_launch_package(
                    self.raw_files,
                    self.worker_raw,
                    self.backend_raw,
                    self.full_source_raw,
                    self.preflight_raw,
                    scope_raw,
                    stage_id=self.stage_id,
                    run_ids=self.run_ids,
                    ssh_argv_prefix=self.ssh_argv_prefix,
                    ssh_environment=self.ssh_environment,
                    local_stage_root=local_root,
                    remote_stage_root=self.remote_stage_root,
                )
                self.spawn_mock.reset_mock()
                self.custom_batch_mode = {0: mode_name}
                result = controller_module.run_stage(
                    pkg,
                    self.raw_files,
                    self.worker_raw,
                    self.backend_raw,
                    self.full_source_raw,
                    self.preflight_raw,
                    scope_raw,
                )
                self.assertEqual(self.spawn_mock.call_count, 1)
                self.assertFalse(result["collection_complete"])

    def test_live_handshake_compatibility_with_real_remote_supervise_batch(self):
        import recalibration_execution_remote as remote_mod
        pkg = self._build_pkg()
        config_sha = pkg["public"]["config_sha256"]
        launch_cfg = pkg["public"]["launch_config"]
        boot_frames = (
            controller_module._encode_bootstrap_frames(
                pkg["public"]["package_sha256"],
                config_sha,
                0,
                launch_cfg,
                self.backend_raw,
            )
        )
        remote_stdin = [boot_frames]
        remote_stdout = bytearray()

        class FakeBackend:
            pid = 888111
            stdin = ArtificialPipe(400)
            stdout = ArtificialPipe(401)
            stderr = ArtificialPipe(402)
            returncode = 0

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

            def send_signal(self, signum):
                pass

        backend_out = [self._batch_backend_jsonl(0), b""]
        backend_err = [b""]

        def remote_read(descriptor, count):
            if descriptor == 490:
                if not remote_stdin:
                    return b""
                chunk = remote_stdin.pop(0)
                if len(chunk) > count:
                    remote_stdin.insert(0, chunk[count:])
                return chunk[:count]
            target = backend_out if descriptor == 401 else backend_err
            if not target:
                return b""
            chunk = target.pop(0)
            if len(chunk) > count:
                target.insert(0, chunk[count:])
            return chunk[:count]

        def remote_write(descriptor, raw):
            if descriptor == 491:
                remote_stdout.extend(raw)
                if len(remote_stdin) == 0 and len(remote_stdout) >= FRAME_HEADER.size:
                    magic, channel, size = FRAME_HEADER.unpack_from(remote_stdout, 0)
                    if magic == WIRE_MAGIC and channel == 1 and len(remote_stdout) == FRAME_HEADER.size + size:
                        ready_payload = bytes(remote_stdout[FRAME_HEADER.size:])
                        bundle_zero = bytes.fromhex(pkg["private"]["bundles_hex"][0])
                        remote_stdin.append(
                            controller_module._encode_grant_frames(
                                ready_payload,
                                95_000_000_000,
                                bundle_zero,
                            )
                        )
                return len(raw)
            return len(raw)

        class FakeBackendPopen(FakeBackend):
            def __init__(instance, *args, **kwargs):
                pass

        with mock.patch.object(remote_mod.subprocess, "Popen", FakeBackendPopen), \
             mock.patch.object(remote_mod.os, "killpg", side_effect=ProcessLookupError), \
             mock.patch.object(remote_mod.os, "set_blocking"), \
             mock.patch.object(remote_mod.platform, "node", return_value="target-node"), \
             mock.patch.object(remote_mod, "_current_executable", return_value="/runtime/env/bin/python3.12"), \
             mock.patch.object(remote_mod, "_stdin_fd", return_value=490), \
             mock.patch.object(remote_mod, "_stdout_fd", return_value=491), \
             mock.patch.object(remote_mod, "_io_read", side_effect=remote_read), \
             mock.patch.object(remote_mod, "_io_write", side_effect=remote_write), \
             mock.patch.object(remote_mod.selectors, "DefaultSelector", side_effect=lambda: ArtificialSelector(self.clock)):
            rc = supervise_batch(config_sha, 0)
        self.assertEqual(rc, 0)
        from recalibration_execution_evidence import decode_receipt
        wire_bytes = bytes(remote_stdout)
        decoded = decode_receipt(
            wire_bytes,
            expected_sha256=hashlib.sha256(wire_bytes).hexdigest(),
            expected={
                "kind": "LORA_RECALIBRATION_READY",
                "version": 1,
                "stage_id": self.stage_id,
                "batch_index": 0,
                "run_id": self.run_ids[0],
                "preparation_sha256": pkg["public"]["preparation_sha256"],
                "package_sha256": pkg["public"]["package_sha256"],
                "config_sha256": config_sha,
                "bundle_sha256": self.batches[0]["bundle_sha256"],
                "backend_sha256": BACKEND_SHA256,
                "worker_sha256": WORKER_SHA256,
                "hostname": "target-node",
                "uid": os.getuid(),
                "stage_root": self.remote_stage_root,
                "batch_root": self.remote_stage_root + "/batch-00-" + self.run_ids[0],
                "expected_runtime_sha256": digest(self.runtime),
            },
        )
        self.assertEqual(decoded["done"]["status"], "completed")

    def test_early_ssh_stdout_close_while_stdin_writable_no_ebadf_and_drains_stderr(self):
        self.custom_batch_mode[0] = "early_eof_with_stderr"
        pkg = self._build_pkg()
        result = controller_module.run_stage(
            pkg,
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            self.full_source_raw,
            self.preflight_raw,
            self.scope_raw,
        )
        self.assertEqual(result["secondary_errors"], [])
        self.assertTrue(result["binding_complete"])
        batch_zero = Path(self.local_stage_root) / ("batch-00-" + self.run_ids[0])
        self.assertEqual((batch_zero / "ssh-stderr.bin").read_bytes(), b"Permission denied\n")

    def test_broken_pipe_then_invalid_ready_bytes_no_closed_fileno_error(self):
        self.custom_batch_mode[0] = "broken_pipe_banner"
        pkg = self._build_pkg()
        result = controller_module.run_stage(
            pkg,
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            self.full_source_raw,
            self.preflight_raw,
            self.scope_raw,
        )
        self.assertEqual(result["secondary_errors"], [])
        self.assertTrue(result["binding_complete"])

    def test_clock_rollback_and_cleanup_cancel_still_kill_reap_close_and_persist(self):
        pkg = self._build_pkg()

        def rollback_on_read(descriptor, count):
            if descriptor == 301:
                self.group_alive = True
                self.current_ssh_proc.returncode = None
                self.clock.nanoseconds -= 9_000_000_000
            return self._io_read(descriptor, count)

        with mock.patch.object(controller_module, "_io_read", side_effect=rollback_on_read):
            result = controller_module.run_stage(
                pkg,
                self.raw_files,
                self.worker_raw,
                self.backend_raw,
                self.full_source_raw,
                self.preflight_raw,
                self.scope_raw,
            )
        self.assertFalse(self.group_alive)
        self.assertTrue(self.current_ssh_proc.stdout.closed)
        self.assertTrue(self.current_ssh_proc.stderr.closed)
        self.assertFalse(result["binding_complete"])
        self.assertTrue((Path(self.local_stage_root) / "controller-summary.json").is_file())

        sub_dir = self.base_dir / "cleanup-cancel"
        sub_dir.mkdir(mode=0o700)
        local_two = str(sub_dir / self.stage_id)
        scope_two = encoded({
            **json.loads(self.scope_raw.decode("utf-8")),
            "local_stage_root": local_two,
        })
        pkg_two = controller_module.build_launch_package(
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            self.full_source_raw,
            self.preflight_raw,
            scope_two,
            stage_id=self.stage_id,
            run_ids=self.run_ids,
            ssh_argv_prefix=self.ssh_argv_prefix,
            ssh_environment=self.ssh_environment,
            local_stage_root=local_two,
            remote_stage_root=self.remote_stage_root,
        )
        cleanup_cancel = KeyboardInterrupt("cancel in ssh cleanup")
        signals_sent = []

        def stubborn_ssh_group(pgid, signum):
            if signum != 0:
                signals_sent.append(signum)
            if signum == controller_module.signal.SIGKILL:
                self.group_alive = False
                self.current_ssh_proc.returncode = -9
            if not self.group_alive:
                raise ProcessLookupError()

        def interrupt_sleep(seconds):
            self.clock.sleep(seconds)
            if controller_module.signal.SIGTERM in signals_sent and controller_module.signal.SIGKILL not in signals_sent:
                raise cleanup_cancel

        def read_triggers_cleanup(descriptor, count):
            if descriptor == 301:
                self.group_alive = True
                self.current_ssh_proc.returncode = None
                raise OSError("ssh pipe error")
            return self._io_read(descriptor, count)

        self.killpg_mock.side_effect = stubborn_ssh_group
        with mock.patch.object(controller_module, "_io_read", side_effect=read_triggers_cleanup):
            with mock.patch.object(controller_module.time, "sleep", side_effect=interrupt_sleep):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    controller_module.run_stage(
                        pkg_two,
                        self.raw_files,
                        self.worker_raw,
                        self.backend_raw,
                        self.full_source_raw,
                        self.preflight_raw,
                        scope_two,
                    )
        self.assertIs(caught.exception, cleanup_cancel)
        self.assertIn(controller_module.signal.SIGKILL, signals_sent)
        self.assertTrue((Path(local_two) / "controller-summary.json").is_file())

    def test_batch_dir_claim_close_failure_preserves_primary_cancellation(self):
        pkg = self._build_pkg()
        primary_cancel = KeyboardInterrupt("batch dir verify cancel")
        original_verify = controller_module._verify_private_dir
        original_close = controller_module.os.close
        verify_calls = []
        closed_batch = []

        def verify_hook(descriptor):
            verify_calls.append(descriptor)
            if len(verify_calls) == 3:
                raise primary_cancel
            return original_verify(descriptor)

        def close_hook(descriptor):
            original_close(descriptor)
            if len(verify_calls) >= 3 and not closed_batch:
                closed_batch.append(descriptor)
                raise OSError("batch_fd close failed")

        with mock.patch.object(controller_module, "_verify_private_dir", side_effect=verify_hook):
            with mock.patch.object(controller_module.os, "close", side_effect=close_hook):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    controller_module.run_stage(
                        pkg,
                        self.raw_files,
                        self.worker_raw,
                        self.backend_raw,
                        self.full_source_raw,
                        self.preflight_raw,
                        self.scope_raw,
                    )
        self.assertIs(caught.exception, primary_cancel)
        self.spawn_mock.assert_not_called()

    def test_batch_one_cancel_rebuilds_evidence_with_unknown_after_dispatch(self):
        self.custom_batch_mode[1] = "cancel"
        pkg = self._build_pkg()
        with self.assertRaises(KeyboardInterrupt):
            controller_module.run_stage(
                pkg,
                self.raw_files,
                self.worker_raw,
                self.backend_raw,
                self.full_source_raw,
                self.preflight_raw,
                self.scope_raw,
            )
        self.assertEqual(self.spawn_mock.call_count, 2)
        saved_ev = json.loads(
            (Path(self.local_stage_root) / "execution-evidence.json").read_bytes()
        )
        self.assertEqual(len(saved_ev["batches"]), 2)
        self.assertEqual(saved_ev["content_report"]["observed"], 32)
        self.assertEqual(saved_ev["content_report"]["unknown_after_dispatch"], 32)
        self.assertEqual(saved_ev["content_report"]["not_dispatched"], 416)

    def test_clock_rollback_at_batch_start_and_initial_reap_stall_sigkill(self):
        pkg = self._build_pkg()
        original_encode_boot = controller_module._encode_bootstrap_frames

        def rollback_after_claim(*args, **kwargs):
            res = original_encode_boot(*args, **kwargs)
            self.clock.nanoseconds -= 9_000_000_000
            return res

        with mock.patch.object(
            controller_module, "_encode_bootstrap_frames", side_effect=rollback_after_claim
        ):
            result = controller_module.run_stage(
                pkg,
                self.raw_files,
                self.worker_raw,
                self.backend_raw,
                self.full_source_raw,
                self.preflight_raw,
                self.scope_raw,
            )
        self.assertFalse(result["binding_complete"])
        self.assertTrue((Path(self.local_stage_root) / "execution-evidence.json").is_file())
        self.assertTrue((Path(self.local_stage_root) / "controller-summary.json").is_file())
        self.assertEqual(result["evidence"]["content_report"]["unknown_after_dispatch"], 32)

        sub_dir = self.base_dir / "reap-stall"
        sub_dir.mkdir(mode=0o700)
        local_two = str(sub_dir / self.stage_id)
        scope_two = encoded({
            **json.loads(self.scope_raw.decode("utf-8")),
            "local_stage_root": local_two,
        })
        pkg_two = controller_module.build_launch_package(
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            self.full_source_raw,
            self.preflight_raw,
            scope_two,
            stage_id=self.stage_id,
            run_ids=self.run_ids,
            ssh_argv_prefix=self.ssh_argv_prefix,
            ssh_environment=self.ssh_environment,
            local_stage_root=local_two,
            remote_stage_root=self.remote_stage_root,
        )
        signals_sent = []

        def stubborn_group(pgid, signum):
            if signum != 0:
                signals_sent.append(signum)
            if signum == controller_module.signal.SIGKILL:
                self.group_alive = False
                self.current_ssh_proc.returncode = -9
            if not self.group_alive:
                raise ProcessLookupError()

        wait_calls = []
        original_spawn = self._spawn_ssh

        def spawn_with_slow_wait(argv, **kwargs):
            proc = original_spawn(argv, **kwargs)

            def slow_wait(timeout=None):
                wait_calls.append(timeout)
                if len(wait_calls) == 1:
                    self.clock.sleep(2.5)
                    raise subprocess.TimeoutExpired("ssh", timeout)
                return proc.returncode

            proc.wait = slow_wait
            return proc

        def fail_first_read(descriptor, count):
            if descriptor == 301:
                self.group_alive = True
                self.current_ssh_proc.returncode = None
                raise OSError("ssh broken")
            return self._io_read(descriptor, count)

        self.killpg_mock.side_effect = stubborn_group
        with mock.patch.object(self, "_spawn_ssh", side_effect=spawn_with_slow_wait):
            with mock.patch.object(controller_module, "_io_read", side_effect=fail_first_read):
                controller_module.run_stage(
                    pkg_two,
                    self.raw_files,
                    self.worker_raw,
                    self.backend_raw,
                    self.full_source_raw,
                    self.preflight_raw,
                    scope_two,
                )
        self.assertIn(controller_module.signal.SIGKILL, signals_sent)


if __name__ == "__main__":
    unittest.main()
