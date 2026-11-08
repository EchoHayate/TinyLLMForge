import importlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import development_cpu
import development_protocol as protocol


class FixedModel:
    evidence_kind = "test_double"

    def __init__(self, fail_at=None):
        self.requests = []
        self.fail_at = fail_at

    def start(self):
        return {"event": "ready", "test_double": True}

    def __call__(self, request):
        self.requests.append(request)
        if len(self.requests) == self.fail_at:
            raise ConnectionError("test transport failure")
        action = "VERIFY" if "Return exactly A, B, C, or VERIFY." in request["messages"][0]["content"] else "A"
        return {
            "event": "output", "id": request["id"], "request_sha256": protocol.checksum(request),
            "text": action, "terminated": True, "input_ids": [11, 12],
            "output_ids": [13, 14], "generation_ns": 1, "call_ns": 2,
            "prompt_sha256": "a" * 64,
        }

    def finish(self):
        return {"event": "finished", "calls": len(self.requests)}

    def close(self):
        pass


class CampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.online = importlib.import_module("development_online")
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = pathlib.Path(cls.directory.name)
        audit_root = cls.root / "audit"
        development_cpu.run(audit_root, "audit")
        cls.audit = {
            "manifest": json.loads((audit_root / "manifest.json").read_text()),
            "observation": json.loads((audit_root / "stdout.json").read_text()),
        }
        cls.manifest = protocol.build_manifest("7" * 32)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_all_methods_online_probe_counts_and_seal(self):
        model = FixedModel()
        target = self.root / "full"
        result = self.online.run_campaign(target, self.manifest, model, self.audit)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["score"]["complete_records"], 120)
        self.assertEqual(len(model.requests), 144)
        self.assertEqual(len({request["id"] for request in model.requests}), 144)
        self.assertEqual(len(list(target.glob("probe-*"))), 48)
        self.assertEqual(result["score"]["by_method"]["always_verify"]["cost"]["model_calls"]["total"], 24)
        self.assertEqual(result["score"]["by_method"]["active_verify"]["cost"]["executed_cases"]["total"], 48)
        self.assertEqual(result["model_origin"], "test_double")
        self.assertEqual(protocol.checksum(json.loads((target / "records.json").read_text())),
                         result["records_sha256"])
        lines = [json.loads(line) for line in (target / "journal.jsonl").read_text().splitlines()]
        seal_position = next(index for index, event in enumerate(lines) if event["event"] == "sealed")
        self.assertEqual(lines[seal_position + 1]["event"], "scored")
        self.assertTrue((target / "source" / "development_online.py").exists())
        self.assertTrue(all(record["episode_wall_ns"] > 0 for record in result["records"]))

    def test_failure_stops_without_retry_and_retains_missing(self):
        model = FixedModel(fail_at=2)
        target = self.root / "failure"
        result = self.online.run_campaign(target, self.manifest, model, self.audit)
        self.assertEqual(result["status"], "aborted")
        self.assertEqual(len(model.requests), 2)
        self.assertGreater(sum(summary["missing"] for summary in result["score"]["by_method"].values()), 0)
        self.assertIn("ConnectionError", result["error"])
        self.assertTrue((target / "records.json").exists())
        self.assertIsNone(result["score"]["by_method"]["rethink"]["cost"]["input_tokens"]["total"])

    def test_existing_directory_rejects_before_any_call(self):
        model = FixedModel()
        target = self.root / "existing"
        target.mkdir()
        with self.assertRaises(FileExistsError):
            self.online.run_campaign(target, self.manifest, model, self.audit)
        self.assertEqual(model.requests, [])

    def test_wrong_request_response_is_retained_but_rejected(self):
        class WrongModel(FixedModel):
            def __call__(self, request):
                return {**super().__call__(request), "request_sha256": "0" * 64}
        result = self.online.run_campaign(self.root / "wrong", self.manifest, WrongModel(), self.audit)
        self.assertEqual(result["status"], "aborted")
        self.assertEqual(result["score"]["by_method"]["direct"]["selection_pass_count"], 0)
        self.assertIn("response binding", result["error"])

    def test_cleanup_failure_does_not_erase_partial_records(self):
        class CleanupFailure(FixedModel):
            def close(self):
                raise OSError("test cleanup failure")
        target = self.root / "cleanup"
        result = self.online.run_campaign(
            target, self.manifest, CleanupFailure(fail_at=1), self.audit
        )
        self.assertEqual(result["status"], "aborted")
        self.assertIn("cleanup", result["cleanup_error"])
        self.assertTrue((target / "records.json").exists())

    def test_readiness_failure_keeps_all_records_missing(self):
        class NotReady(FixedModel):
            def start(self):
                raise ValueError("not admitted")
        model = NotReady()
        result = self.online.run_campaign(self.root / "not-ready", self.manifest, model, self.audit)
        self.assertEqual(result["status"], "aborted")
        self.assertEqual(model.requests, [])
        self.assertEqual(result["score"]["complete_records"], 0)
        self.assertEqual(sum(row["missing"] for row in result["score"]["by_method"].values()), 120)


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.online = importlib.import_module("development_online")

    def test_real_local_pipe_ready_echo_finished(self):
        code = (
            "import json,sys\n"
            "print(json.dumps({'event':'ready'}),flush=True)\n"
            "count=0\n"
            "for line in sys.stdin:\n"
            " value=json.loads(line); count+=1\n"
            " print(json.dumps({'event':'output','id':value['id']}),flush=True)\n"
            "print(json.dumps({'event':'finished','calls':count}),flush=True)\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            model = self.online.ModelProcess(
                [sys.executable, "-I", "-B", "-u", "-c", code],
                pathlib.Path(directory) / "transport", timeout_seconds=5,
            )
            try:
                self.assertEqual(model.start()["event"], "ready")
                self.assertEqual(model({"id": "1" * 32})["id"], "1" * 32)
                self.assertEqual(model.finish()["calls"], 1)
                self.assertEqual(model.process.returncode, 0)
            finally:
                model.close()

    def test_eof_oversize_timeout_and_extra_output_reject(self):
        programs = (
            ("pass", EOFError),
            ("print('x'*1048577,flush=True)", ValueError),
            ("import time; time.sleep(2)", TimeoutError),
            ("print('{\"event\":\"ready\"}\\n{\"event\":\"ready\"}',flush=True)", ValueError),
        )
        for code, error in programs:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                model = self.online.ModelProcess(
                    [sys.executable, "-I", "-B", "-u", "-c", code],
                    pathlib.Path(directory) / "transport", timeout_seconds=0.3,
                )
                try:
                    with self.assertRaises(error):
                        model.start()
                        model.finish()
                finally:
                    model.close()
                self.assertIsNotNone(model.process.poll())

    def test_stderr_tail_after_stdout_eof_is_fully_captured(self):
        code = (
            "import os,sys,time\n"
            "print('{\"event\":\"ready\"}',flush=True)\n"
            "sys.stdin.read()\n"
            "print('{\"event\":\"finished\",\"calls\":0}',flush=True)\n"
            "os.close(1)\n"
            "time.sleep(.05)\n"
            "sys.stderr.write('z'*2097152); sys.stderr.flush()\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / "transport"
            model = self.online.ModelProcess(
                [sys.executable, "-I", "-B", "-u", "-c", code], root, timeout_seconds=2
            )
            try:
                model.start()
                model.finish()
            finally:
                model.close()
            self.assertEqual((root / "stderr.raw").stat().st_size, 2097152)

    def test_cleanup_preserves_termination_tail(self):
        code = (
            "import sys,signal,time\n"
            "def stop(*args):\n"
            " sys.stderr.write('cleanup-tail'); sys.stderr.flush()\n"
            " sys.stdout.write('partial'); sys.stdout.flush()\n"
            " raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM,stop)\n"
            "print('{\"event\":\"ready\"}',flush=True)\n"
            "time.sleep(5)\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / "transport"
            model = self.online.ModelProcess(
                [sys.executable, "-I", "-B", "-u", "-c", code], root, timeout_seconds=2
            )
            model.start()
            model.close()
            self.assertEqual((root / "stderr.raw").read_bytes(), b"cleanup-tail")
            self.assertTrue((root / "stdout.raw").read_bytes().endswith(b"partial"))

    def test_disappearing_process_group_still_finalizes_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self.online.ModelProcess(
                [sys.executable, "-I", "-B", "-u", "-c",
                 "import time; print('{\"event\":\"ready\"}',flush=True); time.sleep(5)"],
                pathlib.Path(directory) / "transport", timeout_seconds=2,
            )
            model.start()
            original_kill = self.online.os.killpg

            def disappeared(pid, sig):
                original_kill(pid, sig)
                model.process.wait(timeout=1)
                raise ProcessLookupError("group exited")

            try:
                with mock.patch.object(self.online.os, "killpg", side_effect=disappeared):
                    model.close()
            finally:
                if not all(stream.closed for stream in model.raw.values()):
                    model.close()
            self.assertTrue(all(stream.closed for stream in model.raw.values()))

    def test_cleanup_allows_normal_exit_after_streams_close(self):
        code = (
            "import os,signal,time\n"
            "def stop(*args):\n"
            " os.close(1); os.close(2); time.sleep(.05); raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM,stop)\n"
            "print('{\"event\":\"ready\"}',flush=True)\n"
            "time.sleep(5)\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            model = self.online.ModelProcess(
                [sys.executable, "-I", "-B", "-u", "-c", code],
                pathlib.Path(directory) / "transport", timeout_seconds=2,
            )
            model.start()
            model.close()
            self.assertEqual(model.process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
