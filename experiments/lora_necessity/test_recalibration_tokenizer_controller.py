import ast
import errno
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import tokenize
import types
import unittest
from unittest import mock

import test_recalibration_tokenizer_inventory as inventory_fixture
from test_tokenizer_audit_report import synthetic_profile, synthetic_transcript
from recalibration_tokenizer_inventory import build_synthetic_inventory
from recalibration_tokenizer_remote import SESSION_SOURCE_NAMES, encode_frame
from validation_contract import digest, encoded


REPO_LORA = Path(__file__).resolve().parent

with mock.patch("subprocess.Popen") as import_spawn, mock.patch("os.killpg") as import_killpg:
    try:
        tok_ctrl = importlib.import_module("recalibration_tokenizer_controller")
    except ModuleNotFoundError as import_err:
        if import_err.name != "recalibration_tokenizer_controller":
            raise
        tok_ctrl = None
    assert not import_spawn.called
    assert not import_killpg.called


class ArtificialClock:
    def __init__(self):
        self.nanoseconds = 5_000_000_000_000

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
    def __init__(self):
        self.pid = 999111
        self.stdin = None
        self.stdout = ArtificialPipe(601)
        self.stderr = ArtificialPipe(602)
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
        descriptor = fileobj.fileno()
        entry = types.SimpleNamespace(fileobj=fileobj, fd=descriptor, events=events, data=data)
        self.registered[descriptor] = entry
        return entry

    def unregister(self, fileobj):
        descriptor = fileobj.fileno()
        return self.registered.pop(descriptor)

    def get_map(self):
        return self.registered

    def select(self, timeout=None):
        if timeout is not None:
            self.clock.sleep(min(0.05, max(0.0, timeout)))
        if self.error is not None:
            raise self.error
        if self.idle:
            return []
        return [(entry, entry.events) for entry in list(self.registered.values())]

    def close(self):
        self.registered.clear()


class TokenizerControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = inventory_fixture.RecalibrationInventoryTests
        fixture.setUpClass()
        cls.validation_inputs = {
            "raw_files": fixture.raw,
            "worker_raw": fixture.worker,
            "run_ids": fixture.run_ids,
            "records": fixture.records,
            "rendering_raw": fixture.renderer,
            "expected_runtime": fixture.runtime,
        }
        cls.inventory = build_synthetic_inventory(**cls.validation_inputs)
        cls.profile = synthetic_profile()
        cls.profile_raw = encoded(cls.profile)
        cls.profile_sha256 = hashlib.sha256(cls.profile_raw).hexdigest()
        cls.records, cls.finalization = synthetic_transcript(
            cls.inventory, cls.profile, count=120
        )
        cls.session_sources = {
            name: (REPO_LORA / name).read_bytes() for name in SESSION_SOURCE_NAMES
        }

    def setUp(self):
        self.spawn_mock = mock.patch("subprocess.Popen").start()
        self.killpg_mock = mock.patch("os.killpg").start()
        self.addCleanup(mock.patch.stopall)
        self.assertIsNotNone(tok_ctrl, "recalibration_tokenizer_controller is absent")
        self.tempdir = tempfile.TemporaryDirectory(prefix="tlf-tok-ctrl-")
        self.addCleanup(self.tempdir.cleanup)
        self.base_dir = Path(self.tempdir.name).resolve()
        self.local_parent = self.base_dir / "local"
        self.local_parent.mkdir(mode=0o700)
        self.local_run_root = str(self.local_parent / self.profile["run_id"])
        self.ssh_prefix = ["/usr/bin/ssh", "-T", "rl-server"]
        self.ssh_env = {"PATH": "/usr/bin:/bin", "LANG": "C"}
        self.clock = ArtificialClock()
        self.group_alive = False
        self.zombie_eperm_once = False
        self.killpg_mock.side_effect = self._killpg
        self.stdin_received = bytearray()
        self.stdout_chunks = []
        self.stderr_chunks = []
        self.mode = "ok"
        self.current_proc = None
        owner = self

        class OwnedPopen(ArtificialSshProcess):
            def __init__(instance, *args, **kwargs):
                supplied = owner._spawn_ssh(*args, **kwargs)
                instance.__dict__.update(supplied.__dict__)

        mock.patch.object(tok_ctrl.subprocess, "Popen", OwnedPopen).start()
        mock.patch.object(tok_ctrl.time, "monotonic_ns", self.clock.monotonic_ns).start()
        mock.patch.object(tok_ctrl.time, "monotonic", self.clock.monotonic).start()
        mock.patch.object(tok_ctrl.time, "sleep", self.clock.sleep).start()
        mock.patch.object(
            tok_ctrl.selectors,
            "DefaultSelector",
            side_effect=lambda: ArtificialSelector(self.clock),
        ).start()
        mock.patch.object(tok_ctrl.os, "set_blocking").start()
        mock.patch.object(tok_ctrl, "_io_read", side_effect=self._io_read).start()
        mock.patch.object(tok_ctrl, "_io_write", side_effect=self._io_write).start()

    def _killpg(self, pgid, signum):
        if self.zombie_eperm_once and signum == 0:
            self.zombie_eperm_once = False
            raise PermissionError("macOS XNU zombie EPERM")
        if not self.group_alive:
            raise ProcessLookupError()
        if signum != 0:
            self.group_alive = False
            if self.current_proc is not None and self.current_proc.returncode is None:
                self.current_proc.returncode = -signum

    def _spawn_ssh(self, argv, **kwargs):
        self.spawn_mock(argv, **kwargs)
        self.stdin_received = bytearray()
        proc = ArtificialSshProcess()
        proc.stdin = ArtificialPipe(600, on_close=self._on_stdin_closed)
        cfg = self.current_pkg["public"]["bootstrap_config"]
        cfg_sha = self.current_pkg["public"]["config_sha256"]
        self.identity = {
            "run_id": "bad" if self.mode == "bad_ready" else cfg["run_id"],
            "config_sha256": cfg_sha,
            "hostname": cfg["hostname"],
            "uid": cfg["uid"],
            "stage": cfg["stage"],
            "protection": {
                "threads": 1,
                "fork_denied": True,
                "thread_denied": True,
                "denied_syscalls": [56, 57, 58, 435],
                "non_x86_and_x32": "KILL_PROCESS",
            },
            "core_limit": [0, 0],
        }
        self.stdout_chunks = [encode_frame("READY", encoded(self.identity))]
        self.stderr_chunks = []
        self.current_proc = proc
        return proc

    def _on_stdin_closed(self):
        if self.mode == "bad_ready":
            self.stdout_chunks.append(b"")
            self.stderr_chunks.append(b"")
            return
        if self.mode == "cancel":
            self.group_alive = True
            self.current_proc.returncode = None
            self.stdout_chunks.append(KeyboardInterrupt("tok ctrl-c"))
            return
        cfg = self.current_pkg["public"]["bootstrap_config"]
        cfg_sha = self.current_pkg["public"]["config_sha256"]
        result_bytes = encoded({
            "kind": "LORA_RECALIBRATION_TOKENIZER_REMOTE_RESULT",
            "version": 1,
            "identity": self.identity,
            "projection_sha256": cfg["projection_sha256"],
            "profile_sha256": cfg["profile_sha256"],
            "bootstrap_failure": None,
            "session_output": {
                "records": self.records,
                "finalization": self.finalization,
                "failure": None,
                "abort_failed": False,
            },
        })
        done_bytes = encoded({
            "config_sha256": cfg_sha,
            "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        })
        self.stdout_chunks.extend([
            encode_frame("RESULT", result_bytes)
            + encode_frame("STDOUT", b"")
            + encode_frame("STDERR", b"")
            + encode_frame("DONE", done_bytes),
            b"",
        ])
        self.stderr_chunks.append(b"")

    def _io_write(self, descriptor, raw):
        self.assertEqual(descriptor, 600)
        if self.current_proc.stdin.closed:
            raise OSError(errno.EBADF, "closed pipe")
        self.stdin_received.extend(raw)
        return len(raw)

    def _io_read(self, descriptor, count):
        items = self.stdout_chunks if descriptor == 601 else self.stderr_chunks
        if not items:
            raise BlockingIOError()
        item = items[0]
        if isinstance(item, BaseException):
            items.pop(0)
            raise item
        if item == b"":
            items.pop(0)
            return b""
        if len(item) > count:
            items[0] = item[count:]
            return item[:count]
        items.pop(0)
        return item

    def _build_pkg(self, local_root=None):
        actual_local = self.local_run_root if local_root is None else local_root
        scope_raw = encoded({
            "kind": "LORA_RECALIBRATION_TOKENIZER_STAGE_SCOPE",
            "version": 1,
            "run_id": self.profile["run_id"],
            "profile_sha256": self.profile_sha256,
            "local_run_root": actual_local,
            "remote_parent": "/tmp/tok-runs",
            "ssh_argv_prefix_sha256": digest(self.ssh_prefix),
            "ssh_environment_sha256": digest(self.ssh_env),
        })
        pkg = tok_ctrl.build_tokenizer_launch_package(
            self.validation_inputs,
            self.profile_raw,
            self.profile_sha256,
            self.session_sources,
            scope_raw,
            ssh_argv_prefix=self.ssh_prefix,
            ssh_environment=self.ssh_env,
            hostname="target-node",
            uid=os.getuid(),
            interpreter="/env/bin/python",
            resolved_interpreter="/base/bin/python3.12",
            site_packages="/env/lib/python3.12/site-packages",
            checkpoint_source="/models/Qwen3-8B",
            remote_parent="/tmp/tok-runs",
            local_run_root=actual_local,
            work_timeout_seconds=1200,
        )
        self.current_pkg = pkg
        self.current_scope_raw = scope_raw
        return pkg, scope_raw

    def test_style_and_no_forbidden_imports(self):
        for fname in (
            "recalibration_tokenizer_controller.py",
            "test_recalibration_tokenizer_controller.py",
        ):
            raw = (REPO_LORA / fname).read_bytes()
            tree = ast.parse(raw)
            self.assertEqual(
                [node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and len(node.id) == 1],
                [],
                fname,
            )
            self.assertEqual(
                [tok for tok in tokenize.tokenize(io.BytesIO(raw).readline) if tok.type == tokenize.COMMENT],
                [],
                fname,
            )
        for forbidden in ("transformers", "torch", "validation_remote", "validation_task_worker"):
            self.assertNotIn(forbidden, sys.modules)

    def test_package_no_oracle_leak_and_full_120_row_handshake_with_zombie_eperm(self):
        pkg, scope_raw = self._build_pkg()
        payload_text = bytes.fromhex(pkg["private"]["payload_hex"]).decode("ascii")
        self.assertNotIn("witness_prediction", payload_text)
        self.zombie_eperm_once = True
        res = tok_ctrl.run_tokenizer_stage(
            pkg,
            self.validation_inputs,
            self.profile_raw,
            self.profile_sha256,
            self.session_sources,
            scope_raw,
        )
        self.assertTrue(res["binding_complete"])
        self.assertTrue(res["collection_complete"])
        self.assertTrue(res["all_120_rows_within_budget"])
        self.assertTrue(res["session_finalized"])
        self.assertIs(res["trusted_collection_accepted"], False)
        self.assertIs(res["run_authorized"], False)
        self.assertEqual(res["tokenizer_report"]["public"]["checked"], 120)
        self.assertEqual(res["secondary_errors"], [])

    def test_bad_ready_withholds_payload_and_exclusive_claim_blocks_replay(self):
        self.mode = "bad_ready"
        pkg, scope_raw = self._build_pkg()
        res = tok_ctrl.run_tokenizer_stage(
            pkg,
            self.validation_inputs,
            self.profile_raw,
            self.profile_sha256,
            self.session_sources,
            scope_raw,
        )
        self.assertFalse(res["collection_complete"])
        self.assertEqual(bytes(self.stdin_received), b"")
        with self.assertRaises(FileExistsError):
            tok_ctrl.run_tokenizer_stage(
                pkg,
                self.validation_inputs,
                self.profile_raw,
                self.profile_sha256,
                self.session_sources,
                scope_raw,
            )
        self.assertEqual(self.spawn_mock.call_count, 1)

    def test_cancellation_cleans_up_group_persists_and_reraises(self):
        self.mode = "cancel"
        pkg, scope_raw = self._build_pkg()
        with self.assertRaises(KeyboardInterrupt) as caught:
            tok_ctrl.run_tokenizer_stage(
                pkg,
                self.validation_inputs,
                self.profile_raw,
                self.profile_sha256,
                self.session_sources,
                scope_raw,
            )
        self.assertEqual(str(caught.exception), "tok ctrl-c")
        self.assertFalse(self.group_alive)
        self.assertTrue((Path(self.local_run_root) / "tokenizer-controller-summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
