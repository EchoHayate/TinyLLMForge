import ast
import copy
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

from recalibration_data_contract import FILE_PINS, TASK_IDS, load_dataset
from recalibration_diagnostic_scoring import CANDIDATE_ROLES, build_generation_schedule
from validation_contract import digest, encoded


REPO_LORA = Path(__file__).resolve().parent

with mock.patch("subprocess.Popen") as import_spawn, mock.patch("os.killpg") as import_killpg:
    try:
        diag_remote = importlib.import_module("recalibration_diagnostic_remote")
    except ModuleNotFoundError as import_err:
        if import_err.name != "recalibration_diagnostic_remote":
            raise
        diag_remote = None
    try:
        diag_ctrl = importlib.import_module("recalibration_diagnostic_controller")
    except ModuleNotFoundError as import_err_two:
        if import_err_two.name != "recalibration_diagnostic_controller":
            raise
        diag_ctrl = None
    assert not import_spawn.called
    assert not import_killpg.called


class ArtificialClock:
    def __init__(self):
        self.nanoseconds = 8_000_000_000_000

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
            raise ValueError("closed pipe")
        return self.descriptor

    def close(self):
        if not self.closed:
            self.closed = True
            if self.on_close is not None:
                self.on_close()


class ArtificialProcess:
    def __init__(self, pid=333000):
        self.pid = pid
        self.stdin = None
        self.stdout = ArtificialPipe(701)
        self.stderr = ArtificialPipe(702)
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

    def register(self, fileobj, events, data=None):
        descriptor = fileobj.fileno()
        entry = types.SimpleNamespace(fileobj=fileobj, fd=descriptor, events=events, data=data)
        self.registered[descriptor] = entry
        return entry

    def unregister(self, fileobj):
        return self.registered.pop(fileobj.fileno())

    def get_map(self):
        return self.registered

    def select(self, timeout=None):
        if timeout is not None:
            self.clock.sleep(min(0.02, max(0.0, timeout)))
        return [(entry, entry.events) for entry in list(self.registered.values())]

    def close(self):
        self.registered.clear()


class DiagnosticControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_files = {
            fname: (REPO_LORA / "data" / (
                fname if fname == "supervision_v1.json" else "recalibration_v2_r1/" + fname
            )).read_bytes()
            for fname in FILE_PINS
        }
        cls.worker_raw = (REPO_LORA / "validation_task_worker.py").read_bytes()
        cls.backend_raw = (REPO_LORA / "validation_remote.py").read_bytes()
        cls.dataset = load_dataset(cls.raw_files)
        cls.oracles_by_id = {item["id"]: item for item in cls.dataset["oracles"]}
        cls.prompts_by_id = {item["id"]: item for item in cls.dataset["prompts"]}
        cls.expected_runtime = {
            "hostname": "target-node",
            "uid": 1054,
            "python": "3.12.13",
            "python_root": "/opt/cpython",
            "python_sha256": "2" * 64,
            "bwrap_sha256": "3" * 64,
            "seccomp_sha256": "4" * 64,
            "argv_prefix": ["/usr/bin/bwrap", "--seccomp", "3", "/runtime/bin/python3.12"],
            "no_new_privs": True,
            "host_namespaces": {"mnt": "m", "pid": "p", "net": "n", "user": "u"},
        }
        cls.preflight_raw = encoded({
            "kind": "LORA_RECALIBRATION_LIVE_PREFLIGHT_RECEIPT",
            "version": 1,
            "status": "PREFLIGHT_PASSED",
            "probes_passed": True,
            "remote_python": "/opt/cpython/bin/python3.12",
            "expected_runtime": cls.expected_runtime,
        })

    def setUp(self):
        self.spawn_mock = mock.patch("subprocess.Popen").start()
        self.killpg_mock = mock.patch("os.killpg").start()
        self.addCleanup(mock.patch.stopall)
        self.assertIsNotNone(diag_remote, "recalibration_diagnostic_remote is absent")
        self.assertIsNotNone(diag_ctrl, "recalibration_diagnostic_controller is absent")
        self.tempdir = tempfile.TemporaryDirectory(prefix="tlf-diag-ctrl-")
        self.addCleanup(self.tempdir.cleanup)
        self.base_dir = Path(self.tempdir.name).resolve()
        self.local_parent = self.base_dir / "local"
        self.local_parent.mkdir(mode=0o700)
        self.run_id = "34" * 16
        self.local_run_root = str(self.local_parent / self.run_id)
        self.ssh_prefix = ["/usr/bin/ssh", "-T", "rl-server"]
        self.ssh_env = {"PATH": "/usr/bin:/bin", "LANG": "C"}
        self.clock = ArtificialClock()
        self.killpg_mock.side_effect = ProcessLookupError
        self.stdout_chunks = []
        self.stderr_chunks = []
        self.stdin_buf = bytearray()
        self.call_sequence = []
        self.inject_p2_stop_once = False
        self.inject_p2_unreaped_once = False
        owner = self

        class OwnedPopen(ArtificialProcess):
            def __init__(instance, *args, **kwargs):
                supplied = owner._handle_spawn(*args, **kwargs)
                instance.__dict__.update(supplied.__dict__)

        mock.patch.object(diag_ctrl.subprocess, "Popen", OwnedPopen).start()
        mock.patch.object(diag_ctrl.time, "monotonic_ns", self.clock.monotonic_ns).start()
        mock.patch.object(diag_ctrl.time, "monotonic", self.clock.monotonic).start()
        mock.patch.object(diag_ctrl.time, "sleep", self.clock.sleep).start()
        mock.patch.object(
            diag_ctrl.selectors,
            "DefaultSelector",
            side_effect=lambda: ArtificialSelector(self.clock),
        ).start()
        mock.patch.object(diag_ctrl.os, "set_blocking").start()
        mock.patch.object(diag_ctrl, "_io_read", side_effect=self._io_read).start()
        mock.patch.object(diag_ctrl, "_io_write", side_effect=self._io_write).start()

    def _io_write(self, descriptor, raw):
        self.stdin_buf.extend(raw)
        return len(raw)

    def _io_read(self, descriptor, count):
        items = self.stdout_chunks if descriptor == 701 else self.stderr_chunks
        if not items:
            raise BlockingIOError()
        item = items[0]
        if item == b"":
            items.pop(0)
            return b""
        if len(item) > count:
            items[0] = item[count:]
            return item[:count]
        items.pop(0)
        return item

    def _handle_spawn(self, argv, **kwargs):
        self.spawn_mock(argv, **kwargs)
        self.stdin_buf = bytearray()
        proc = ArtificialProcess(pid=333000 + self.spawn_mock.call_count)
        if self.spawn_mock.call_count == 1:
            self.call_sequence.append("phase_one_gpu")
            cfg = self.current_pkg["public"]["phase_one_config"]
            cfg_sha = self.current_pkg["public"]["phase_one_config_sha256"]
            self.p1_identity = {
                "run_id": cfg["run_id"],
                "config_sha256": cfg_sha,
                "hostname": cfg["hostname"],
                "uid": cfg["uid"],
                "stage": cfg["stage"],
            }
            self.stdout_chunks = [diag_remote.encode_frame("READY", encoded(self.p1_identity))]
            self.stderr_chunks = []
            proc.stdin = ArtificialPipe(700, on_close=self._on_p1_stdin_closed)
        else:
            batch_idx = self.spawn_mock.call_count - 2
            self.call_sequence.append("phase_two_batch_%02d" % batch_idx)
            self.active_scoring_batch = batch_idx
            self.stdout_chunks = []
            self.stderr_chunks = []
            proc.stdin = ArtificialPipe(700, on_close=self._on_p2_stdin_closed)
        return proc

    def _on_p1_stdin_closed(self):
        sched = build_generation_schedule(
            self.raw_files["prompts.json"],
            self.raw_files["supervision_v1.json"],
            run_id=self.run_id,
        )
        phase_one_records = []
        for ep in sched["episodes"]:
            tid = ep["task_id"]
            task_idx = TASK_IDS.index(tid)
            oracle = self.oracles_by_id[tid]
            ref_src = oracle["references"][0]["source"]
            init_def = oracle["incorrect_programs"][0]["source"]
            priv_def = oracle["incorrect_programs"][1]["source"]
            if task_idx < 6:
                init_c, fb_c, rt_c = ref_src, ref_src, ref_src
                pub_fb = {"kind": "visible_tests", "cases": [{"id": "P01", "status": "pass"}, {"id": "P02", "status": "pass"}]}
            elif task_idx < 10:
                init_c, fb_c, rt_c = init_def, ref_src, init_def
                wit_cid = oracle["incorrect_programs"][0]["witness"]["case_id"]
                pub_fb = {
                    "kind": "visible_tests",
                    "cases": [
                        {"id": "P01", "status": "mismatch" if wit_cid == "P01" else "pass"},
                        {"id": "P02", "status": "mismatch" if wit_cid == "P02" else "pass"},
                    ],
                }
            else:
                init_c, fb_c, rt_c = priv_def, priv_def, priv_def
                pub_fb = {"kind": "visible_tests", "cases": [{"id": "P01", "status": "pass"}, {"id": "P02", "status": "pass"}]}
            phase_one_records.append({
                "episode_index": ep["episode_index"],
                "task_id": tid,
                "arm": ep["arm"],
                "candidates": {"initial": init_c, "feedback": fb_c, "retry": rt_c},
                "public_feedback": pub_fb,
                "generation_costs": {
                    role_name: {
                        "input_tokens": 600,
                        "output_tokens": 70,
                        "forward_calls": 70,
                        "wall_ns": 2000,
                        "peak_allocated_bytes": 16_000_000_000,
                        "peak_reserved_bytes": 17_000_000_000,
                    }
                    for role_name in CANDIDATE_ROLES
                },
                "public_execution_ns": 300,
            })
        self.simulated_phase_one = phase_one_records
        cfg_sha = self.current_pkg["public"]["phase_one_config_sha256"]
        res_bytes = encoded({
            "kind": "LORA_RECALIBRATION_DIAGNOSTIC_PHASE1_RESULT",
            "version": 1,
            "identity": self.p1_identity,
            "bootstrap_failure": None,
            "gpu_selected": {"index": 7, "uuid": "GPU-synthetic-idle"},
            "integrity": {
                "initial_base_sha256": "b" * 64,
                "final_base_sha256": "b" * 64,
                "rng_unchanged": True,
            },
            "shared_costs": {
                "model_load_ns": 10_000_000_000,
                "phase_one_wall_ns": 50_000_000_000,
            },
            "phase_one_records": phase_one_records,
        })
        done_bytes = encoded({
            "config_sha256": cfg_sha,
            "result_sha256": hashlib.sha256(res_bytes).hexdigest(),
        })
        self.stdout_chunks.extend([
            diag_remote.encode_frame("RESULT", res_bytes)
            + diag_remote.encode_frame("STDOUT", b"")
            + diag_remote.encode_frame("STDERR", b"")
            + diag_remote.encode_frame("DONE", done_bytes),
            b"",
        ])
        self.stderr_chunks.append(b"")

    def _on_p2_stdin_closed(self):
        bundle_raw = bytes(self.stdin_buf)
        bundle_obj = json.loads(bundle_raw.decode("utf-8"))
        b_run_id = bundle_obj["run_id"]
        events = [{
            "kind": "ready",
            "run_id": b_run_id,
            "bundle_sha256": hashlib.sha256(bundle_raw).hexdigest(),
            "worker_sha256": hashlib.sha256(self.worker_raw).hexdigest(),
            "runtime": copy.deepcopy(self.expected_runtime),
        }]
        for ord_idx, req in enumerate(bundle_obj["requests"]):
            matched_task = next(
                item for item in self.dataset["oracles"] if item["entry_point"] == req["entry_point"]
            )
            case_obj = next(
                c_item for c_item in matched_task["cases"]
                if encoded(c_item["args"]) == encoded(req["args"])
            )
            ref_src = matched_task["references"][0]["source"]
            if req["source"] == ref_src:
                out_obj = case_obj["expected"]
            else:
                prog = next(
                    (p_item for p_item in matched_task["incorrect_programs"] if p_item["source"] == req["source"]),
                    None,
                )
                if prog and prog["witness"]["case_id"] == case_obj["id"]:
                    out_obj = prog["witness"]["predicted"]
                else:
                    out_obj = case_obj["expected"]
            if self.inject_p2_unreaped_once and ord_idx == 0:
                self.inject_p2_unreaped_once = False
                events.append({
                    "kind": "row",
                    "run_id": b_run_id,
                    "ordinal": 0,
                    "request_sha256": digest(req),
                    "result": {
                        "status": "timeout",
                        "returncode": None,
                        "stdout_hex": "",
                        "stderr_hex": "",
                        "elapsed_ns": 2_000_000_000,
                        "process_reaped": False,
                        "stdin_complete": True,
                    },
                })
                events.append({
                    "kind": "done",
                    "run_id": b_run_id,
                    "received": 1,
                    "stopped": True,
                    "elapsed_ns": 2_000_000_100,
                })
                self.stdout_chunks.extend([b"".join(encoded(ev) + b"\n" for ev in events), b""])
                self.stderr_chunks.append(b"")
                return
            events.append({
                "kind": "row",
                "run_id": b_run_id,
                "ordinal": ord_idx,
                "request_sha256": digest(req),
                "result": {
                    "status": "program_error" if (self.inject_p2_stop_once and ord_idx == 0) else "completed",
                    "returncode": 20 if (self.inject_p2_stop_once and ord_idx == 0) else 0,
                    "stdout_hex": "" if (self.inject_p2_stop_once and ord_idx == 0) else encoded(out_obj).hex(),
                    "stderr_hex": "",
                    "elapsed_ns": 40,
                    "process_reaped": True,
                    "stdin_complete": True,
                },
            })
            if self.inject_p2_stop_once and ord_idx == 0:
                self.inject_p2_stop_once = False
                events.append({
                    "kind": "done",
                    "run_id": b_run_id,
                    "received": 1,
                    "stopped": True,
                    "elapsed_ns": 60,
                })
                self.stdout_chunks.extend([b"".join(encoded(ev) + b"\n" for ev in events), b""])
                self.stderr_chunks.append(b"")
                return
        events.append({
            "kind": "done",
            "run_id": b_run_id,
            "received": len(bundle_obj["requests"]),
            "stopped": False,
            "elapsed_ns": 400,
        })
        self.stdout_chunks.extend([b"".join(encoded(ev) + b"\n" for ev in events), b""])
        self.stderr_chunks.append(b"")

    def test_style_and_idle_gpu_selector(self):
        for fname in (
            "recalibration_diagnostic_remote.py",
            "recalibration_diagnostic_controller.py",
            "test_recalibration_diagnostic_controller.py",
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
        gpu_csv = "0, GPU-busy, NVIDIA A100 80GB PCIe, 10000, 70000\n3, GPU-idle, NVIDIA A100 80GB PCIe, 81000, 4\n"
        apps_csv = "GPU-busy, 12345, 69000\n"
        chosen = diag_remote.select_idle_a100_gpu(gpu_csv, apps_csv)
        self.assertEqual(chosen["uuid"], "GPU-idle")
        with self.assertRaises(ValueError):
            diag_remote.select_idle_a100_gpu(
                "0, GPU-busy, NVIDIA A100 80GB PCIe, 10000, 70000\n", apps_csv
            )

    def test_zero_private_oracle_leak_in_phase_one_and_two_phase_end_to_end(self):
        dep_sources = {
            fname: (REPO_LORA / fname).read_bytes()
            for fname in diag_remote.PHASE1_SOURCE_NAMES
        }
        scope_raw = encoded({
            "kind": "LORA_RECALIBRATION_DIAGNOSTIC_STAGE_SCOPE",
            "version": 1,
            "run_id": self.run_id,
            "local_run_root": self.local_run_root,
            "remote_parent": "/tmp/diag-runs",
            "ssh_argv_prefix_sha256": digest(self.ssh_prefix),
            "ssh_environment_sha256": digest(self.ssh_env),
            "preflight_receipt_sha256": hashlib.sha256(self.preflight_raw).hexdigest(),
        })
        pkg = diag_ctrl.build_diagnostic_package(
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            dep_sources,
            self.preflight_raw,
            scope_raw,
            run_id=self.run_id,
            scoring_run_ids=[("%02x" % idx) + ("99" * 15) for idx in range(18)],
            ssh_argv_prefix=self.ssh_prefix,
            ssh_environment=self.ssh_env,
            interpreter="/env/bin/python",
            resolved_interpreter="/opt/cpython/bin/python3.12",
            site_packages="/env/lib/python3.12/site-packages",
            checkpoint_root="/models/Qwen3-8B",
            expected_tokenizer_sha256="7" * 64,
            remote_parent="/tmp/diag-runs",
            local_run_root=self.local_run_root,
        )
        self.current_pkg = pkg
        p1_payload_str = bytes.fromhex(pkg["private"]["phase_one_payload_hex"]).decode("ascii")
        self.assertNotIn("H01", p1_payload_str)
        self.assertNotIn("incorrect_programs", p1_payload_str)

        out = diag_ctrl.run_diagnostic_stage(
            pkg,
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            dep_sources,
            self.preflight_raw,
            scope_raw,
        )
        self.assertEqual(self.call_sequence[0], "phase_one_gpu")
        self.assertEqual(len(self.call_sequence), 19)
        self.assertTrue(out["binding_complete"])
        self.assertTrue(out["collection_complete"])
        self.assertEqual(
            out["diagnostic_report"]["calibration"]["status"],
            "DEVELOPMENT_DIAGNOSTIC_USABLE",
        )
        self.assertIs(out["trusted_collection_accepted"], False)
        self.assertIs(out["run_authorized"], False)

    def test_controller_phase_two_continuation_when_bundle_stops_early(self):
        self.inject_p2_stop_once = True
        dep_sources = {
            fname: (REPO_LORA / fname).read_bytes()
            for fname in diag_remote.PHASE1_SOURCE_NAMES
        }
        scope_raw = encoded({
            "kind": "LORA_RECALIBRATION_DIAGNOSTIC_STAGE_SCOPE",
            "version": 1,
            "run_id": self.run_id,
            "local_run_root": self.local_run_root,
            "remote_parent": "/tmp/diag-runs",
            "ssh_argv_prefix_sha256": digest(self.ssh_prefix),
            "ssh_environment_sha256": digest(self.ssh_env),
            "preflight_receipt_sha256": hashlib.sha256(self.preflight_raw).hexdigest(),
        })
        pkg = diag_ctrl.build_diagnostic_package(
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            dep_sources,
            self.preflight_raw,
            scope_raw,
            run_id=self.run_id,
            scoring_run_ids=[("%02x" % idx) + ("88" * 15) for idx in range(18)],
            ssh_argv_prefix=self.ssh_prefix,
            ssh_environment=self.ssh_env,
            interpreter="/env/bin/python",
            resolved_interpreter="/opt/cpython/bin/python3.12",
            site_packages="/env/lib/python3.12/site-packages",
            checkpoint_root="/models/Qwen3-8B",
            expected_tokenizer_sha256="7" * 64,
            remote_parent="/tmp/diag-runs",
            local_run_root=self.local_run_root,
        )
        self.current_pkg = pkg
        out = diag_ctrl.run_diagnostic_stage(
            pkg,
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            dep_sources,
            self.preflight_raw,
            scope_raw,
        )
        self.assertEqual(len(self.call_sequence), 20)
        self.assertEqual(out["phase_two_primary_batches"], 18)
        self.assertEqual(out["phase_two_continuation_batches"], 1)
        self.assertTrue(out["binding_complete"])
        self.assertTrue(out["collection_complete"])
        self.assertEqual(out["diagnostic_report"]["episodes"][0]["roles"]["initial"]["state"], "EXECUTION_FAIL")

    def test_controller_aborts_remaining_scoring_batches_on_unreaped_worker(self):
        self.inject_p2_unreaped_once = True
        dep_sources = {
            fname: (REPO_LORA / fname).read_bytes()
            for fname in diag_remote.PHASE1_SOURCE_NAMES
        }
        scope_raw = encoded({
            "kind": "LORA_RECALIBRATION_DIAGNOSTIC_STAGE_SCOPE",
            "version": 1,
            "run_id": self.run_id,
            "local_run_root": self.local_run_root,
            "remote_parent": "/tmp/diag-runs",
            "ssh_argv_prefix_sha256": digest(self.ssh_prefix),
            "ssh_environment_sha256": digest(self.ssh_env),
            "preflight_receipt_sha256": hashlib.sha256(self.preflight_raw).hexdigest(),
        })
        pkg = diag_ctrl.build_diagnostic_package(
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            dep_sources,
            self.preflight_raw,
            scope_raw,
            run_id=self.run_id,
            scoring_run_ids=[("%02x" % idx) + ("77" * 15) for idx in range(18)],
            ssh_argv_prefix=self.ssh_prefix,
            ssh_environment=self.ssh_env,
            interpreter="/env/bin/python",
            resolved_interpreter="/opt/cpython/bin/python3.12",
            site_packages="/env/lib/python3.12/site-packages",
            checkpoint_root="/models/Qwen3-8B",
            expected_tokenizer_sha256="7" * 64,
            remote_parent="/tmp/diag-runs",
            local_run_root=self.local_run_root,
        )
        self.current_pkg = pkg
        out = diag_ctrl.run_diagnostic_stage(
            pkg,
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            dep_sources,
            self.preflight_raw,
            scope_raw,
        )
        self.assertEqual(len(self.call_sequence), 2)
        self.assertEqual(out["phase_two_primary_batches"], 1)
        self.assertEqual(out["phase_two_continuation_batches"], 0)
        self.assertFalse(out["collection_complete"])
        self.assertEqual(out["calibration_status"], "INCONCLUSIVE_INCOMPLETE")

    def test_remote_cuda_init_runs_before_reset_peak_memory_stats(self):
        cuda_state = {"initialized": False, "seeded_after_init": False, "synced_before_reset": False}

        class FakeRngTensor:
            def clone(self):
                return self

        class FakeCudaModule:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 1

            @staticmethod
            def init():
                cuda_state["initialized"] = True

            @staticmethod
            def get_device_properties(dev_obj):
                if not cuda_state["initialized"]:
                    raise RuntimeError("CUDA not initialized before get_device_properties")
                return types.SimpleNamespace(major=8, minor=0)

            @staticmethod
            def manual_seed_all(seed_val):
                if not cuda_state["initialized"]:
                    raise RuntimeError("CUDA not initialized before manual_seed_all")
                cuda_state["seeded_after_init"] = (seed_val == 42)

            @staticmethod
            def synchronize(dev_obj):
                if not cuda_state["initialized"]:
                    raise RuntimeError("CUDA not initialized before synchronize")
                cuda_state["synced"] = True

            @staticmethod
            def reset_peak_memory_stats(dev_obj):
                if not cuda_state["initialized"]:
                    raise RuntimeError("Invalid device argument ")
                cuda_state["synced_before_reset"] = bool(cuda_state.get("synced"))

            @staticmethod
            def max_memory_allocated(dev_obj):
                if not cuda_state["initialized"]:
                    raise RuntimeError("Invalid device argument ")
                return 16_106_127_360

            @staticmethod
            def max_memory_reserved(dev_obj):
                if not cuda_state["initialized"]:
                    raise RuntimeError("Invalid device argument ")
                return 17_179_869_184

            @staticmethod
            def get_rng_state(dev_obj):
                if not cuda_state["initialized"]:
                    raise RuntimeError("Invalid device argument ")
                return FakeRngTensor()

        class FakeModel:
            def __init__(self):
                self.config = types.SimpleNamespace(use_cache=True)

            def to(self, dev_obj):
                return self

            def eval(self):
                return self

            def requires_grad_(self, flag_val):
                return self

        fake_torch = types.ModuleType("torch")
        fake_torch.bfloat16 = "bf16"
        fake_torch.cuda = FakeCudaModule()
        fake_torch.backends = types.SimpleNamespace(
            cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),
            cudnn=types.SimpleNamespace(allow_tf32=True),
        )
        fake_torch.random = types.SimpleNamespace(get_rng_state=lambda: FakeRngTensor())
        fake_torch.set_num_threads = lambda num_t: None
        fake_torch.set_num_interop_threads = lambda num_t: None
        fake_torch.manual_seed = lambda seed_val: None
        fake_torch.use_deterministic_algorithms = lambda flag_val: None
        fake_torch.device = lambda dev_name: dev_name

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda *args_in, **kw_in: "fake-tok"
        )
        fake_transformers.AutoModelForCausalLM = types.SimpleNamespace(
            from_pretrained=lambda *args_in, **kw_in: FakeModel()
        )

        shared_costs = {}
        cfg_stub = {
            "checkpoint_root": "/models/Qwen3-8B",
            "expected_tokenizer_sha256": digest({"tok": 1}),
        }
        saved_mods = {
            mod_key: sys.modules.get(mod_key)
            for mod_key in ("torch", "transformers", "training_worker", "tokenizer_measurement")
        }
        try:
            sys.modules["torch"] = fake_torch
            sys.modules["transformers"] = fake_transformers
            fake_tw = types.ModuleType("training_worker")
            fake_tw._state = lambda mdl: {}
            fake_tw._tensor_digest = lambda st_map: "f" * 64
            sys.modules["training_worker"] = fake_tw
            fake_tm = types.ModuleType("tokenizer_measurement")
            fake_tm._state = lambda tok_obj: ({"tok": 1}, {})
            sys.modules["tokenizer_measurement"] = fake_tm
            loaded = diag_remote._initialize_cuda_and_load_model(cfg_stub, shared_costs)
        finally:
            for mod_key, orig_mod in saved_mods.items():
                if orig_mod is None:
                    sys.modules.pop(mod_key, None)
                else:
                    sys.modules[mod_key] = orig_mod

        self.assertTrue(cuda_state["initialized"])
        self.assertTrue(cuda_state["seeded_after_init"])
        self.assertTrue(cuda_state["synced_before_reset"])
        self.assertEqual(loaded["init_base_sha"], "f" * 64)
        self.assertEqual(shared_costs["model_load_peak_allocated_bytes"], 16_106_127_360)

    def test_phase_two_spawns_close_fds_child_and_phase_two_only_stage_scores_sealed_phase_one(self):
        dep_sources = {
            fname: (REPO_LORA / fname).read_bytes()
            for fname in diag_remote.PHASE1_SOURCE_NAMES
        }
        scope_raw = encoded({
            "kind": "LORA_RECALIBRATION_DIAGNOSTIC_STAGE_SCOPE",
            "version": 1,
            "run_id": self.run_id,
            "local_run_root": self.local_run_root,
            "remote_parent": "/tmp/diag-runs",
            "ssh_argv_prefix_sha256": digest(self.ssh_prefix),
            "ssh_environment_sha256": digest(self.ssh_env),
            "preflight_receipt_sha256": hashlib.sha256(self.preflight_raw).hexdigest(),
        })
        full_pkg = diag_ctrl.build_diagnostic_package(
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            dep_sources,
            self.preflight_raw,
            scope_raw,
            run_id=self.run_id,
            scoring_run_ids=[("%02x" % idx) + ("66" * 15) for idx in range(18)],
            ssh_argv_prefix=self.ssh_prefix,
            ssh_environment=self.ssh_env,
            interpreter="/env/bin/python",
            resolved_interpreter="/opt/cpython/bin/python3.12",
            site_packages="/env/lib/python3.12/site-packages",
            checkpoint_root="/models/Qwen3-8B",
            expected_tokenizer_sha256="7" * 64,
            remote_parent="/tmp/diag-runs",
            local_run_root=self.local_run_root,
        )
        p2_cmd = full_pkg["private"]["phase_two_ssh_argv"][-1]
        self.assertIn("subprocess.Popen", p2_cmd)
        self.assertIn("close_fds=True", p2_cmd)

        self.current_pkg = full_pkg
        self._on_p1_stdin_closed = self._on_p1_stdin_closed
        p1_cfg = full_pkg["public"]["phase_one_config"]
        p1_cfg_sha = full_pkg["public"]["phase_one_config_sha256"]
        self.p1_identity = {
            "run_id": p1_cfg["run_id"],
            "config_sha256": p1_cfg_sha,
            "hostname": p1_cfg["hostname"],
            "uid": p1_cfg["uid"],
            "stage": p1_cfg["stage"],
        }
        self.stdout_chunks = [diag_remote.encode_frame("READY", encoded(self.p1_identity))]
        self._on_p1_stdin_closed()
        sealed_wire = b"".join(chunk for chunk in self.stdout_chunks if chunk)
        self.stdout_chunks = []
        self.stderr_chunks = []

        p2_stage_id = "ab" * 16
        p2_local_root = str(self.local_parent / p2_stage_id)
        p2_scope_raw = encoded({
            "kind": "LORA_RECALIBRATION_DIAGNOSTIC_PHASE2_SCOPE",
            "version": 1,
            "stage_id": p2_stage_id,
            "source_run_id": self.run_id,
            "phase_one_wire_sha256": hashlib.sha256(sealed_wire).hexdigest(),
            "local_run_root": p2_local_root,
            "ssh_argv_prefix_sha256": digest(self.ssh_prefix),
            "ssh_environment_sha256": digest(self.ssh_env),
            "preflight_receipt_sha256": hashlib.sha256(self.preflight_raw).hexdigest(),
        })
        p2_pkg = diag_ctrl.build_phase_two_scoring_package(
            self.raw_files,
            self.worker_raw,
            self.backend_raw,
            self.preflight_raw,
            p2_scope_raw,
            sealed_phase_one_wire=sealed_wire,
            sealed_phase_one_config=p1_cfg,
            stage_id=p2_stage_id,
            scoring_run_ids=[("%02x" % idx) + ("55" * 15) for idx in range(18)],
            ssh_argv_prefix=self.ssh_prefix,
            ssh_environment=self.ssh_env,
            resolved_interpreter="/opt/cpython/bin/python3.12",
            local_run_root=p2_local_root,
        )

        orig_handle = self._handle_spawn

        def p2_only_spawn(argv, **kwargs):
            self.spawn_mock(argv, **kwargs)
            self.stdin_buf = bytearray()
            proc = ArtificialProcess(pid=444000 + self.spawn_mock.call_count)
            self.call_sequence.append("p2_only_%02d" % (self.spawn_mock.call_count - 1))
            self.stdout_chunks = []
            self.stderr_chunks = []
            proc.stdin = ArtificialPipe(700, on_close=self._on_p2_stdin_closed)
            return proc

        self._handle_spawn = p2_only_spawn
        try:
            p2_out = diag_ctrl.run_phase_two_scoring_stage(
                p2_pkg,
                self.raw_files,
                self.worker_raw,
                self.backend_raw,
                self.preflight_raw,
                p2_scope_raw,
                sealed_phase_one_wire=sealed_wire,
                sealed_phase_one_config=p1_cfg,
            )
        finally:
            self._handle_spawn = orig_handle

        self.assertEqual(len(self.call_sequence), 18)
        self.assertTrue(p2_out["binding_complete"])
        self.assertTrue(p2_out["collection_complete"])
        self.assertEqual(
            p2_out["diagnostic_report"]["calibration"]["status"],
            "DEVELOPMENT_DIAGNOSTIC_USABLE",
        )


if __name__ == "__main__":
    unittest.main()
