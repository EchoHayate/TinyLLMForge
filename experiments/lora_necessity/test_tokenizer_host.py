import base64
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


MODULE_ROOT = Path(__file__).resolve().parent
EVIDENCE_ROOT = Path("/tmp/tlf-tokenizer-host-launch-20261113-tilybrwi")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def hashed(raw):
    return hashlib.sha256(raw).hexdigest()


class HostTests(unittest.TestCase):
    def setUp(self):
        for name in ("tokenizer_host_remote", "tokenizer_host_launch"):
            self.assertTrue((MODULE_ROOT / (name + ".py")).exists(), name + " missing")
        self.remote = importlib.import_module("tokenizer_host_remote")
        self.launch = importlib.import_module("tokenizer_host_launch")
        self.temporary = tempfile.TemporaryDirectory(
            prefix="task-1-test-", dir=EVIDENCE_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def frames(self, report=None):
        report = report or {"failure": None, "stop_reason": None}
        return b"".join(self.remote.encode_frame(channel, data) for channel, data in (
            ("READY", b"{}"), ("RESULT", encoded(report)),
            ("STDOUT", b"hello"), ("STDERR", b"warning"), ("DONE", b"{}")))

    def test_fragmented_frames(self):
        parser = self.remote.FrameDecoder()
        for value in self.frames():
            parser.feed(bytes([value]))
        self.assertEqual(parser.finish()["STDOUT"], b"hello")
        self.assertEqual(self.remote.decode_frames(self.frames())["STDERR"], b"warning")

    def test_invalid_framing(self):
        complete = self.frames()
        malformed = (
            b"not a framed result", complete + b"x", complete[:-1], b"",
            complete + self.remote.encode_frame("DONE", b"{}"),
            self.remote.encode_frame("RESULT", b"{}"),
            self.remote.encode_frame("READY", b"{}") * 2,
            b"TLFH1" + bytes([99]) + struct.pack("!Q", 0),
            b"TLFH1" + bytes([1]) + struct.pack("!Q", 4097),
        )
        for raw in malformed:
            with self.subTest(size=len(raw)), self.assertRaises(ValueError):
                self.remote.decode_frames(raw)

    def test_declaration_rejected_before_body(self):
        parser = self.remote.FrameDecoder()
        parser.feed(self.remote.encode_frame("READY", b"{}"))
        with self.assertRaises(ValueError):
            parser.feed(b"TLFH1" + bytes([2]) + struct.pack("!Q", 67108865))
        with self.assertRaises(ValueError):
            self.remote.encode_frame("STDERR", b"x" * 1048577)

    def bundle(self):
        sources = {name: b"pass\n" for name in self.remote.SOURCE_NAMES}
        inventory = b'{"synthetic":true}'
        profile = encoded({"run_id": "a" * 32})
        payload = encoded({
            "sources": {name: base64.b64encode(raw).decode("ascii")
                        for name, raw in sources.items()},
            "inventory": base64.b64encode(inventory).decode("ascii"),
            "profile": base64.b64encode(profile).decode("ascii"),
        })
        config = {
            "sources_sha256": {name: hashed(raw) for name, raw in sources.items()},
            "inventory_sha256": hashed(inventory), "profile_sha256": hashed(profile),
            "payload_sha256": hashed(payload), "payload_bytes": len(payload),
            "run_id": "a" * 32,
        }
        return payload, config

    def test_hash_bound_bundle(self):
        payload, config = self.bundle()
        sources, inventory, profile = self.remote.validate_bundle(payload, config)
        self.assertEqual(set(sources), set(self.remote.SOURCE_NAMES))
        self.assertEqual(inventory, b'{"synthetic":true}')
        self.assertEqual(json.loads(profile)["run_id"], config["run_id"])
        for field in ("payload_sha256", "inventory_sha256", "profile_sha256"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.remote.validate_bundle(payload, {**config, field: "0" * 64})
        with self.assertRaises(ValueError):
            self.remote.validate_bundle(payload + b" ", config)

    def test_bundle_rejects_unsafe_names_base64_and_duplicate_keys(self):
        payload, config = self.bundle()
        for name in ("../escape.py", "/tmp/escape.py", "sitecustomize.py", "a/b.py"):
            value = json.loads(payload)
            value["sources"][name] = "cGFzcwo="
            changed = encoded(value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.remote.validate_bundle(changed, {
                    **config, "payload_sha256": hashed(changed), "payload_bytes": len(changed)})
        value = json.loads(payload)
        value["inventory"] += "\n"
        for changed in (encoded(value), b'{"sources":{},"sources":{}}'):
            with self.assertRaises(ValueError):
                self.remote.validate_bundle(changed, {
                    **config, "payload_sha256": hashed(changed), "payload_bytes": len(changed)})

    def test_private_paths_and_exclusive_files(self):
        parent = self.root / "parent"
        stage = self.remote.prepare_stage(str(parent), "a" * 32)
        self.assertEqual(stage, parent / ("bootstrap-" + "a" * 32))
        self.assertFalse((parent / ("a" * 32)).exists())
        self.assertEqual(parent.stat().st_mode & 0o777, 0o700)
        self.assertEqual(stage.stat().st_mode & 0o777, 0o700)
        self.remote.write_private(stage / "input", b"synthetic")
        self.assertEqual((stage / "input").stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            self.remote.write_private(stage / "input", b"replacement")
        with self.assertRaises(FileExistsError):
            self.remote.prepare_stage(str(parent), "a" * 32)
        alias = self.root / "alias"
        alias.symlink_to(parent)
        with self.assertRaises(ValueError):
            self.remote.prepare_stage(str(alias), "b" * 32)
        parent.chmod(0o755)
        with self.assertRaises(ValueError):
            self.remote.prepare_stage(str(parent), "b" * 32)

    def child(self, body):
        return (
            sys.executable, "-I", "-S", "-B", "-c",
            "import os,sys,time,signal,json\n"
            f"sys.path.insert(0, {str(MODULE_ROOT)!r})\n"
            "from tokenizer_host_remote import encode_frame\n" + body)

    def collect(self, body, timeout=2, ready=None):
        return self.launch.collect_once(
            self.child(body), b"private-synthetic-input",
            b"{}", self.root / "attempt",
            timeout_seconds=timeout,
            ready_validator=ready or (lambda raw: raw == b"{}"))

    def success_body(self):
        return (
            "sys.stdout.buffer.write(encode_frame('READY', b'{}')); sys.stdout.buffer.flush()\n"
            "grant = sys.stdin.buffer.readline()\n"
            "payload = sys.stdin.buffer.read()\n"
            "assert payload == b'private-synthetic-input'\n"
            "assert 0 < json.loads(grant)['remaining_ns'] < 540000000000\n"
            f"sys.stdout.buffer.write({self.frames()[len(self.remote.encode_frame('READY', b'{}')):]!r})\n"
            "sys.stdout.buffer.flush()\n")

    def test_local_real_subprocess_success_and_exclusive_attempt(self):
        result = self.collect(self.success_body())
        self.assertEqual(result["status"], "TRANSPORT_COMPLETE")
        self.assertEqual(result["returncode"], 0)
        self.assertTrue(result["reaped"])
        self.assertEqual(result["window"]["connection_attempts"], 1)
        self.assertEqual(result["window"]["audit_attempts"], 1)
        self.assertFalse(result["window"]["benchmark_eligible"])
        for path in (self.root / "attempt").iterdir():
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertNotIn("private-synthetic-input", json.dumps(result))
        with self.assertRaises(FileExistsError):
            self.collect(self.success_body())

    def test_private_input_not_sent_before_valid_ready(self):
        body = (
            "import select\n"
            "assert not select.select([sys.stdin], [], [], 0.1)[0]\n"
            "sys.stdout.buffer.write(encode_frame('READY', b'{\"wrong\":true}'))\n"
            "sys.stdout.buffer.flush()\n"
            "time.sleep(20)\n")
        result = self.collect(body)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["input_bytes_sent"], 0)
        self.assertTrue(result["reaped"])

    def test_timeout_reaps_owned_process(self):
        started = time.monotonic()
        result = self.collect("time.sleep(20)\n", timeout=0.2)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error"], "TIMEOUT")
        self.assertTrue(result["reaped"])
        self.assertLess(time.monotonic() - started, 3)
        with self.assertRaises(ProcessLookupError):
            os.kill(result["pid"], 0)
        self.assertEqual(result["remote_state"], "UNKNOWN")

    def test_stdout_stderr_flood_bounded(self):
        for descriptor in (1, 2):
            with self.subTest(descriptor=descriptor):
                attempt = self.root / "attempt"
                if attempt.exists():
                    attempt.rename(self.root / "first")
                result = self.collect(f"while True: os.write({descriptor}, b'x' * 65536)\n")
                self.assertEqual(result["status"], "FAILED")
                self.assertTrue(result["reaped"])
                self.assertLessEqual(result["transport_stderr_bytes"], 1048576)
                self.assertLessEqual(result["wire_stdout_bytes"], self.remote.WIRE_LIMIT)

    def test_nonzero_exit_truncation_and_trailing_bytes(self):
        for suffix in ("sys.exit(9)\n", "os.write(1, b'extra')\n"):
            attempt = self.root / "attempt"
            if attempt.exists():
                attempt.rename(self.root / "first")
            result = self.collect(self.success_body() + suffix)
            self.assertEqual(result["status"], "FAILED")
        result = self.launch.collect_once(
            self.child("os.write(1, b'TLF')\n"), b"x", b"{}",
            self.root / "truncated", timeout_seconds=2,
            ready_validator=lambda raw: True)
        self.assertEqual(result["status"], "FAILED")

    def test_audit_failure_never_becomes_pass(self):
        frames = self.frames({"failure": {"phase": "OPEN"}, "stop_reason": "SESSION_ERROR"})
        body = (
            "os.write(1, encode_frame('READY', b'{}'))\n"
            "sys.stdin.buffer.read()\n"
            f"os.write(1, {frames[len(self.remote.encode_frame('READY', b'{}')):]!r})\n")
        result = self.collect(body)
        self.assertEqual(result["status"], "TRANSPORT_COMPLETE")
        self.assertFalse(result["audit_pass"])

    def test_deadline_anchored_before_ready_not_after_input(self):
        self.assertEqual(self.remote.remaining_seconds(100, 2_000_000_000, 101.5), 0.5)
        with self.assertRaises(TimeoutError):
            self.remote.remaining_seconds(100, 2_000_000_000, 102)
        for remaining in (0, -1, 540_000_000_001, True):
            with self.assertRaises(ValueError):
                self.remote.remaining_seconds(100, remaining, 100)

    def test_explicit_remote_deadline_does_not_reset_or_exceed_bound(self):
        self.assertEqual(self.remote.remaining_seconds(
            100, 1_200_000_000_000, 700, work_timeout_seconds=1200), 600)
        with self.assertRaises(TimeoutError):
            self.remote.remaining_seconds(
                100, 1_200_000_000_000, 1300, work_timeout_seconds=1200)
        for budget, grant in ((1201, 1), (True, 1), (1200.0, 1),
                              (0, 1), (1200, 1_200_000_000_001)):
            with self.subTest(budget=budget, grant=grant), self.assertRaises(ValueError):
                self.remote.remaining_seconds(100, grant, 100, work_timeout_seconds=budget)

    def test_explicit_work_timeout_is_manifest_bound_and_collected(self):
        payload, partial = self.bundle()
        arguments = {
            "ssh_prefix": ("/usr/bin/ssh", "-T", "synthetic-host"),
            "interpreter": "/synthetic/python", "hostname": "synthetic-host", "uid": 1234,
            "site_packages": "/synthetic/packages", "source": "/synthetic/model",
            "parent": "/synthetic/audit", "run_id": partial["run_id"], "payload": payload,
            "sources_sha256": partial["sources_sha256"],
            "inventory_sha256": partial["inventory_sha256"],
            "profile_sha256": partial["profile_sha256"],
        }
        default = self.launch.build_launch(**arguments)
        self.assertEqual(json.loads(default.config_json)["work_timeout_seconds"], 540)
        spec = self.launch.build_launch(**arguments, work_timeout_seconds=1200)
        self.assertEqual(json.loads(spec.config_json)["work_timeout_seconds"], 1200)
        self.assertNotEqual(default.manifest()["config_sha256"], spec.manifest()["config_sha256"])
        with patch.object(self.launch, "collect_once", return_value={"status": "FAILED"}) as collect:
            self.launch.execute_once(
                spec, self.root / "explicit", approved_manifest_sha256=hashed(encoded(spec.manifest())))
        self.assertEqual(collect.call_args.kwargs["timeout_seconds"], 1200)
        for invalid in (0, -1, 1201, True, 1200.0, "1200", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.launch.build_launch(**arguments, work_timeout_seconds=invalid)

    def test_collector_accepts_explicit_long_deadline_without_waiting_for_it(self):
        body = (
            "os.write(1, encode_frame('READY', b'{}'))\n"
            "grant = json.loads(sys.stdin.buffer.readline())\n"
            "assert 540000000000 < grant['remaining_ns'] < 1200000000000\n"
            "sys.stdin.buffer.read()\n"
            f"os.write(1, {self.frames()[len(self.remote.encode_frame('READY', b'{}')):]!r})\n")
        result = self.collect(body, timeout=1200)
        self.assertEqual(result["status"], "TRANSPORT_COMPLETE")
        window = result["window"]
        self.assertEqual(window["work_deadline_ns"] - window["started_ns"], 1_200_000_000_000)

    def test_collector_short_integer_budget_matches_window_accounting(self):
        result = self.collect(self.success_body(), timeout=2)
        self.assertEqual(result["status"], "TRANSPORT_COMPLETE")
        window = result["window"]
        self.assertEqual(window["work_deadline_ns"] - window["started_ns"], 2_000_000_000)
        self.assertEqual(window["total_deadline_ns"] - window["started_ns"], 62_000_000_000)

    def test_frozen_command_is_inspectable_and_import_is_inert(self):
        payload, partial = self.bundle()
        source = self.remote.__file__
        spec = self.launch.build_launch(
            ssh_prefix=("/usr/bin/ssh", "-T", "synthetic-host"),
            interpreter="/synthetic/env/bin/python",
            hostname="synthetic-host", uid=1234,
            site_packages="/synthetic/env/lib/python3.12/site-packages",
            source="/synthetic/model", parent="/synthetic/audit",
            run_id=partial["run_id"], payload=payload,
            sources_sha256=partial["sources_sha256"],
            inventory_sha256=partial["inventory_sha256"],
            profile_sha256=partial["profile_sha256"])
        self.assertEqual(spec.argv[:-1], ("/usr/bin/ssh", "-T", "synthetic-host"))
        self.assertIn("-I -S -B", spec.argv[-1])
        self.assertNotIn("private-synthetic-input", spec.argv[-1])
        self.assertNotIn("payload=", repr(spec))
        config = json.loads(spec.config_json)
        self.assertEqual(config["payload_sha256"], hashed(payload))
        self.assertEqual(config["bootstrap_sha256"], hashed(Path(source).read_bytes()))
        self.assertEqual(config["launcher_sha256"], hashed(Path(self.launch.__file__).read_bytes()))
        self.assertEqual(spec.manifest()["command_sha256"], hashed(spec.argv[-1].encode()))
        with self.assertRaises((AttributeError, TypeError)):
            spec.argv = ("changed",)

    @unittest.skipUnless(sys.platform == "linux" and platform.machine() == "x86_64",
                         "Linux x86_64 seccomp unverified on this host")
    def test_linux_task_denial_in_disposable_process(self):
        result = subprocess.run(
            self.child("from tokenizer_host_remote import install_task_denial\n"
                       "print(json.dumps(install_task_denial()))\n"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["threads"], 1)
        self.assertTrue(json.loads(result.stdout)["fork_denied"])
        self.assertTrue(json.loads(result.stdout)["thread_denied"])

    def test_native_diagnostic_overflow_is_fatal(self):
        result = subprocess.run(
            self.child(
                "from tokenizer_host_remote import redirect_diagnostics\n"
                f"redirect_diagnostics({str(self.root)!r})\n"
                "while True: os.write(1, b'x' * 65536)\n"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        self.assertEqual(result.returncode, -signal.SIGXFSZ)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"")
        self.assertEqual((self.root / "native.stdout").stat().st_size, 1048576)

    def test_cancellation_reaps_and_preserves_evidence(self):
        def cancel(raw):
            raise SystemExit()

        result = self.collect(
            "os.write(1, encode_frame('READY', b'{}'))\ntime.sleep(0.5)\n", ready=cancel)
        self.assertEqual(result["error"], "CANCELLED")
        self.assertTrue(result["reaped"])
        self.assertTrue((self.root / "attempt" / "transport.json").is_file())

    def test_bundle_rejects_non_string_base64(self):
        payload, config = self.bundle()
        bundle = json.loads(payload)
        bundle["inventory"] = ["invalid"]
        raw = encoded(bundle)
        with self.assertRaises(ValueError):
            self.remote.validate_bundle(raw, {
                **config, "payload_sha256": hashed(raw), "payload_bytes": len(raw)})

    def test_result_cannot_arrive_while_input_is_unsent(self):
        frames = self.frames()
        body = (
            "os.write(1, encode_frame('READY', b'{}'))\n"
            "time.sleep(0.1)\n"
            f"os.write(1, {frames[len(self.remote.encode_frame('READY', b'{}')):]!r})\n"
            "time.sleep(20)\n")
        result = self.launch.collect_once(
            self.child(body), b"synthetic" * 131072, b"{}", self.root / "early",
            timeout_seconds=2, ready_validator=lambda raw: True)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error"], "RESULT_BEFORE_INPUT")

    def test_redirect_can_precede_snapshot_without_file_limit(self):
        result = subprocess.run(
            self.child(
                "import resource\n"
                "from tokenizer_host_remote import redirect_diagnostics, limit_diagnostics\n"
                f"saved = redirect_diagnostics({str(self.root)!r}, enforce_limit=False)\n"
                f"with open({str(self.root / 'snapshot')!r}, 'wb') as stream:\n"
                "    stream.write(b'x' * 1048577)\n"
                "limit_diagnostics()\n"
                "os.write(1, b'private-out'); os.write(2, b'private-err')\n"
                "os.write(saved, b'framed-only')\n"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b"framed-only")
        self.assertEqual(result.stderr, b"")
        self.assertEqual((self.root / "native.stdout").read_bytes(), b"private-out")
        self.assertEqual((self.root / "native.stderr").read_bytes(), b"private-err")
        self.assertEqual((self.root / "snapshot").stat().st_size, 1048577)

    def test_seccomp_filter_denies_all_creation_and_alternate_abis(self):
        def evaluate(architecture, syscall_number):
            accumulator = 0
            position = 0
            instructions = self.remote.task_filter_instructions()
            while position < len(instructions):
                opcode, jump_true, jump_false, constant = instructions[position]
                position += 1
                if opcode == 0x20:
                    accumulator = {0: syscall_number, 4: architecture}[constant]
                elif opcode == 0x15:
                    position += jump_true if accumulator == constant else jump_false
                elif opcode == 0x45:
                    position += jump_true if accumulator & constant else jump_false
                elif opcode == 0x06:
                    return constant
                else:
                    self.fail("unexpected filter instruction")
            self.fail("missing filter return")

        for syscall_number in (56, 57, 58, 435):
            self.assertEqual(evaluate(0xC000003E, syscall_number), 0x00050001)
        for architecture in (0, 0x40000003, 0xC00000B7):
            self.assertEqual(evaluate(architecture, 57), 0x80000000)
        self.assertEqual(evaluate(0xC000003E, 0x40000000), 0x80000000)
        self.assertEqual(evaluate(0xC000003E, 0x40000000 | 435), 0x80000000)
        self.assertEqual(evaluate(0xC000003E, 0), 0x7FFF0000)

    def accepted_report(self, *, stop=None, open_failure=False):
        fixtures = importlib.import_module("test_tokenizer_audit_report")
        inventory = fixtures.frozen_inventory()
        profile = fixtures.synthetic_profile()
        inventory_raw, profile_raw = encoded(inventory), encoded(profile)
        payload, config = self.bundle()
        sources, _, _ = self.remote.validate_bundle(payload, config)
        payload = self.launch.build_payload(sources, inventory_raw, profile_raw)
        config.update({
            "inventory_sha256": hashed(inventory_raw),
            "profile_sha256": hashed(profile_raw),
            "payload_sha256": hashed(payload), "payload_bytes": len(payload),
        })
        if open_failure:
            audit = importlib.import_module("tokenizer_audit")
            with patch.object(audit.tokenizer_session, "open_session",
                              side_effect=ValueError("private synthetic OPEN failure")) as opening:
                report = audit.run_audit(
                    self.root / "unopened", profile_raw, config["profile_sha256"],
                    inventory_raw, config["inventory_sha256"])
            opening.assert_called_once_with(
                self.root / "unopened", profile_raw, config["profile_sha256"])
        else:
            records, finalization = fixtures.synthetic_transcript(
                inventory, profile, count=1 if stop else 324, stop=stop)
            builder = importlib.import_module("tokenizer_audit_report")
            report = builder.build_report(
                inventory_raw, config["inventory_sha256"], profile_raw,
                config["profile_sha256"], records, finalization)
        self.assertEqual(set(report), {"public", "private"})
        self.assertEqual(len(report["public"]["rows"]), 324)
        self.assertEqual(report["public"]["cleanup_status"], "CLEANUP_UNVERIFIED")
        for flag in (*importlib.import_module("pilot_protocol").FLAGS,
                     "checked_inventory_within_budget", "all_possible_second_rounds_validated"):
            self.assertIs(report["public"][flag], False)
        return report, payload, config

    def execute_report(self, report, payload, config, *, name="attempt", done_fault=False):
        config = {
            **config, "hostname": "synthetic", "uid": 1234, "stage": "/synthetic/stage",
            "launcher_sha256": hashed(Path(self.launch.__file__).read_bytes()),
            "bootstrap_sha256": hashed(Path(self.remote.__file__).read_bytes()),
        }
        identity = {
            "run_id": config["run_id"], "config_sha256": hashed(encoded(config)),
            "hostname": "synthetic", "uid": 1234, "stage": "/synthetic/stage",
            "core_limit": [0, 0],
            "protection": {
                "threads": 1, "fork_denied": True, "thread_denied": True,
                "denied_syscalls": [56, 57, 58, 435], "non_x86_and_x32": "KILL_PROCESS"},
        }
        envelope = encoded({
            "kind": "EXISTING_HOST_TOKENIZER_RESULT", "identity": identity,
            "inventory_sha256": config["inventory_sha256"],
            "profile_sha256": config["profile_sha256"],
            "report": report, "bootstrap_failure": None})
        fixture = self.root / (name + "-result.bin")
        self.remote.write_private(fixture, envelope)
        done = encoded({
            "config_sha256": hashed(encoded(config)),
            "result_sha256": "0" * 64 if done_fault else hashed(envelope)})
        body = (
            f"os.write(1, encode_frame('READY', {encoded(identity)!r}))\n"
            "sys.stdin.buffer.read()\n"
            f"with open({str(fixture)!r}, 'rb') as stream:\n"
            "    sys.stdout.buffer.write(encode_frame('RESULT', stream.read()))\n"
            "sys.stdout.buffer.write(encode_frame('STDOUT', b''))\n"
            "sys.stdout.buffer.write(encode_frame('STDERR', b''))\n"
            f"sys.stdout.buffer.write(encode_frame('DONE', {done!r}))\n"
            "sys.stdout.buffer.flush()\n")
        spec = self.launch.FrozenLaunch(self.child(body), encoded(config), payload)
        attempt = self.root / name
        with self.assertRaises(ValueError):
            self.launch.execute_once(spec, attempt, approved_manifest_sha256="0" * 64)
        self.assertFalse(attempt.exists())
        result = self.launch.execute_once(
            spec, attempt, approved_manifest_sha256=hashed(encoded(spec.manifest())))
        self.assertEqual(result["returncode"], 0)
        self.assertTrue(result["reaped"])
        self.assertEqual(result["window"]["connection_attempts"], 1)
        self.assertEqual(result["window"]["audit_attempts"], 1)
        self.assertFalse(result["window"]["benchmark_eligible"])
        self.assertFalse(result["audit_pass"])
        self.assertEqual(result["remote_state"], "UNKNOWN")
        self.assertTrue((attempt / "result.bin").read_bytes() == envelope,
                        "full private result envelope changed")
        self.assertEqual((attempt / "result.bin").stat().st_mode & 0o777, 0o600)
        self.assertEqual(json.loads((attempt / "execution.json").read_bytes()), result)
        for private_key in ("report", "public", "private", "inventory", "records", "finalization"):
            self.assertNotIn(private_key, result)
        self.assertNotIn(b"private synthetic", encoded(result))
        return result

    def test_execute_once_preserves_completed_accepted_report(self):
        report, payload, config = self.accepted_report()
        self.assertTrue(report["public"]["all_rows_within_budget"])
        self.assertIsNone(report["public"]["failure"])
        self.assertIsNone(report["public"]["stop_reason"])
        result = self.execute_report(report, payload, config)
        self.assertEqual(result["status"], "AUDIT_EVIDENCE_RETURNED")
        self.assertFalse(result["audit_failure"])
        self.assertIsNone(result["error"])

    def test_execute_once_preserves_stopped_accepted_report(self):
        for stop in ("INPUT_LIMIT", "OUTPUT_LIMIT"):
            with self.subTest(stop=stop):
                report, payload, config = self.accepted_report(stop=stop)
                self.assertEqual(report["public"]["stop_reason"], stop)
                self.assertIsNone(report["public"]["failure"])
                result = self.execute_report(report, payload, config, name=stop)
                self.assertEqual(result["status"], "AUDIT_STOPPED")
                self.assertFalse(result["audit_failure"])
                self.assertIsNone(result["error"])

    def test_execute_once_checks_completion_binding_and_preserves_failed_audit(self):
        for fault in ("none", "done_hash"):
            with self.subTest(fault=fault):
                report, payload, config = self.accepted_report(open_failure=True)
                self.assertEqual(report["public"]["failure"],
                                 {"phase": "OPEN", "ordinal": None, "reason": "OPEN_FAILED"})
                self.assertEqual(report["public"]["stop_reason"], "OPEN_FAILED")
                result = self.execute_report(
                    report, payload, config, name=fault, done_fault=fault == "done_hash")
                if fault == "none":
                    self.assertEqual(result["status"], "AUDIT_FAILED")
                    self.assertTrue(result["audit_failure"])
                    self.assertIsNone(result["error"])
                else:
                    self.assertEqual(result["status"], "FAILED")
                    self.assertEqual(result["error"], "RESULT_BINDING_OR_BOOTSTRAP_FAILURE")

    def test_execute_once_rejects_public_binding_mismatch(self):
        for field in ("run_id", "inventory_sha256", "profile_sha256"):
            with self.subTest(field=field):
                report, payload, config = self.accepted_report()
                report["public"][field] = "b" * len(config[field])
                result = self.execute_report(report, payload, config, name=field)
                self.assertEqual(result["status"], "FAILED")
                self.assertEqual(result["error"], "RESULT_BINDING_OR_BOOTSTRAP_FAILURE")

    def test_execute_once_rejects_flat_accepted_public_report(self):
        report, payload, config = self.accepted_report()
        result = self.execute_report(report["public"], payload, config)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["error"], "RESULT_BINDING_OR_BOOTSTRAP_FAILURE")

    def test_execute_once_rejects_malformed_public_report(self):
        for ordinal, public in enumerate((None, [], "invalid", {})):
            with self.subTest(ordinal=ordinal):
                report, payload, config = self.accepted_report()
                report = {**report["public"], **report, "public": public}
                result = self.execute_report(report, payload, config, name=str(ordinal))
                self.assertEqual(result["status"], "FAILED")
                self.assertEqual(result["error"], "RESULT_BINDING_OR_BOOTSTRAP_FAILURE")

    def test_public_api_builds_payload_and_rejects_unsafe_config(self):
        payload, config = self.bundle()
        sources, inventory, profile = self.remote.validate_bundle(payload, config)
        self.assertEqual(self.launch.build_payload(sources, inventory, profile), payload)
        arguments = {
            "ssh_prefix": ("/usr/bin/ssh", "-T", "synthetic-host"),
            "interpreter": "/synthetic/python", "hostname": "synthetic-host", "uid": 1234,
            "site_packages": "/synthetic/packages", "source": "/synthetic/model",
            "parent": "/synthetic/audit", "run_id": config["run_id"], "payload": payload,
            "sources_sha256": config["sources_sha256"],
            "inventory_sha256": config["inventory_sha256"],
            "profile_sha256": config["profile_sha256"],
        }
        for key, value in (("parent", "/bad/../path"), ("run_id", "../escape"),
                           ("uid", True), ("inventory_sha256", "invalid")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.launch.build_launch(**{**arguments, key: value})

    def test_saturated_diagnostic_is_not_accepted_as_untruncated(self):
        for name in ("native.stdout", "native.stderr"):
            self.remote.write_private(self.root / name, b"small")
        self.assertEqual(self.remote.read_diagnostics(self.root),
                         {"STDOUT": b"small", "STDERR": b"small"})
        with (self.root / "native.stdout").open("wb") as stream:
            stream.write(b"x" * self.remote.DIAGNOSTIC_LIMIT)
        with self.assertRaises(ValueError):
            self.remote.read_diagnostics(self.root)

    def test_bootstrap_identity_failure_has_no_ready_or_staging(self):
        payload, config = self.bundle()
        spec = self.launch.build_launch(
            ssh_prefix=("/bin/sh", "-c"), interpreter=sys.executable,
            hostname="not-the-local-host", uid=os.getuid(),
            site_packages=str(self.root), source=str(self.root), parent=str(self.root / "audit"),
            run_id=config["run_id"], payload=payload,
            sources_sha256=config["sources_sha256"],
            inventory_sha256=config["inventory_sha256"], profile_sha256=config["profile_sha256"])
        process = subprocess.run(
            spec.argv, input=b"", stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        self.assertEqual(process.returncode, 78)
        self.assertEqual(process.stdout, b"")
        self.assertEqual(process.stderr, b"TOKENIZER_BOOTSTRAP_FAILED\n")
        self.assertFalse((self.root / "audit").exists())

    def test_execute_rejects_local_source_drift_before_attempt_creation(self):
        payload, config = self.bundle()
        config.update({
            "launcher_sha256": "0" * 64,
            "bootstrap_sha256": hashed(Path(self.remote.__file__).read_bytes()),
        })
        spec = self.launch.FrozenLaunch(("/nonexistent-command",), encoded(config), payload)
        with self.assertRaises(ValueError):
            self.launch.execute_once(
                spec, self.root / "drift",
                approved_manifest_sha256=hashed(encoded(spec.manifest())))
        self.assertFalse((self.root / "drift").exists())

    @unittest.skipUnless(sys.platform == "linux" and platform.machine() == "x86_64",
                         "Full Linux bootstrap ordering unverified on this host")
    def test_linux_bootstrap_runs_synthetic_api_order_once(self):
        inventory = b'{"synthetic":true}'
        profile = encoded({"run_id": "a" * 32})
        sources = {name: b"pass\n" for name in self.remote.SOURCE_NAMES}
        sources["tokenizer_audit_report.py"] = (
            b"def validate_inventory(raw, approved):\n"
            b"    assert raw == b'{\"synthetic\":true}'\n")
        sources["tokenizer_runtime_profile.py"] = (
            b"def validate_profile(raw, approved):\n"
            b"    return {}\n")
        sources["tokenizer_snapshot.py"] = (
            b"import json\nfrom pathlib import Path\n"
            b"def create_snapshot(source, parent, raw, approved):\n"
            b"    root = Path(parent) / json.loads(raw)['run_id']\n"
            b"    root.mkdir(mode=0o700)\n"
            b"    with (root / 'large-snapshot').open('xb') as stream:\n"
            b"        stream.write(b'x' * 1048577)\n")
        sources["tokenizer_audit.py"] = (
            b"import json, os, resource\n"
            b"def run_audit(root, profile_raw, profile_hash, inventory_raw, inventory_hash):\n"
            b"    assert (root / 'large-snapshot').stat().st_size == 1048577\n"
            b"    assert resource.getrlimit(resource.RLIMIT_FSIZE) == (1048576, 1048576)\n"
            b"    assert len(os.listdir('/proc/self/task')) == 1\n"
            b"    os.write(1,b'native-out'); os.write(2,b'native-err')\n"
            b"    return {'public':{'run_id':json.loads(profile_raw)['run_id'],"
            b"'inventory_sha256':inventory_hash,'profile_sha256':profile_hash,"
            b"'failure':None,'stop_reason':None,'cleanup_status':'CLEANUP_UNVERIFIED'},"
            b"'private':{'synthetic':True}}\n")
        package = self.root / "site-packages"
        package.mkdir(mode=0o700)
        (package / "evil.pth").write_text("raise RuntimeError('must not run')\n")
        payload = self.launch.build_payload(sources, inventory, profile)
        spec = self.launch.build_launch(
            ssh_prefix=("/bin/sh", "-c"), interpreter=sys.executable,
            hostname=platform.node().split(".")[0], uid=os.getuid(),
            site_packages=str(package), source=str(self.root), parent=str(self.root / "audit"),
            run_id="a" * 32, payload=payload,
            sources_sha256={name: hashed(raw) for name, raw in sources.items()},
            inventory_sha256=hashed(inventory), profile_sha256=hashed(profile))
        result = self.launch.execute_once(
            spec, self.root / "launch",
            approved_manifest_sha256=hashed(encoded(spec.manifest())))
        self.assertEqual(result["status"], "AUDIT_EVIDENCE_RETURNED")
        self.assertFalse(result["audit_pass"])
        report = json.loads((self.root / "launch" / "result.bin").read_bytes())["report"]
        self.assertEqual(set(report), {"public", "private"})
        self.assertEqual(report["public"]["run_id"], "a" * 32)
        self.assertEqual(report["private"], {"synthetic": True})
        self.assertEqual((self.root / "launch" / "stdout.bin").read_bytes(), b"native-out")
        self.assertEqual((self.root / "launch" / "stderr.bin").read_bytes(), b"native-err")


if __name__ == "__main__":
    unittest.main()
