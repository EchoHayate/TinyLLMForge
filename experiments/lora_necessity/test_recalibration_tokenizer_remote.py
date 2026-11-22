import ast
import base64
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import tokenize
import unittest
from unittest import mock

from validation_contract import digest, encoded


REPO_LORA = Path(__file__).resolve().parent
SESSION_MODULES = (
    "pilot_protocol.py",
    "validation_contract.py",
    "tokenizer_measurement.py",
    "tokenizer_runtime_guard.py",
    "tokenizer_runtime_profile.py",
    "tokenizer_snapshot.py",
    "tokenizer_session.py",
)

with mock.patch("subprocess.Popen") as import_spawn, mock.patch("os.killpg") as import_killpg:
    try:
        tok_remote = importlib.import_module("recalibration_tokenizer_remote")
    except ModuleNotFoundError as import_err:
        if import_err.name != "recalibration_tokenizer_remote":
            raise
        tok_remote = None
    assert not import_spawn.called
    assert not import_killpg.called


class TokenizerRemoteTests(unittest.TestCase):
    def setUp(self):
        self.spawn_mock = mock.patch("subprocess.Popen").start()
        self.killpg_mock = mock.patch("os.killpg").start()
        self.addCleanup(mock.patch.stopall)
        self.assertIsNotNone(tok_remote, "recalibration_tokenizer_remote is absent")
        self.tempdir = tempfile.TemporaryDirectory(prefix="tlf-tok-rem-")
        self.addCleanup(self.tempdir.cleanup)
        self.base_dir = Path(self.tempdir.name).resolve()
        self.parent_dir = self.base_dir / "runs"
        self.parent_dir.mkdir(mode=0o700)
        self.site_dir = self.base_dir / "site-packages"
        self.site_dir.mkdir(mode=0o700)
        self.source_dir = self.base_dir / "Qwen3-8B"
        self.source_dir.mkdir(mode=0o700)
        self.run_id = "ab" * 16
        self.sources_raw = {name: (REPO_LORA / name).read_bytes() for name in SESSION_MODULES}
        self.sources_sha = {
            name: hashlib.sha256(raw).hexdigest() for name, raw in self.sources_raw.items()
        }
        self.projection = {
            "kind": "LORA_RECALIBRATION_COUNTING_PROJECTION",
            "version": 1,
            "rows": [
                {
                    "ordinal": 0,
                    "messages": [{"role": "user", "content": "prompt zero"}],
                    "answer": None,
                    "row_sha256": "1" * 64,
                },
                {
                    "ordinal": 1,
                    "messages": [{"role": "user", "content": "prompt one"}],
                    "answer": "def fn():\n    return 1\n",
                    "row_sha256": "2" * 64,
                },
            ],
        }
        self.projection_raw = encoded(self.projection)
        self.profile_obj = {
            "kind": "TOKENIZER_RUNTIME_PROFILE",
            "version": 1,
            "run_id": self.run_id,
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "file_sha256": {
                "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
                "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
                "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
                "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
                "merges.txt": "8831e4f1a0444713000000000000000000000000000000000000000000000000"[:64],
            },
            "versions": {
                "python": "3.12.13",
                "transformers": "4.57.1",
                "tokenizers": "0.22.2",
                "huggingface-hub": "0.36.2",
                "jinja2": "3.1.6",
            },
        }
        from tokenizer_runtime_profile import FILE_PINS as REAL_TOK_PINS
        self.profile_obj["file_sha256"] = dict(REAL_TOK_PINS)
        self.profile_raw = encoded(self.profile_obj)
        self.profile_sha = hashlib.sha256(self.profile_raw).hexdigest()
        self.payload_raw = encoded({
            "sources": {
                name: base64.b64encode(raw).decode("ascii")
                for name, raw in self.sources_raw.items()
            },
            "projection": base64.b64encode(self.projection_raw).decode("ascii"),
            "profile": base64.b64encode(self.profile_raw).decode("ascii"),
        })
        self.config = {
            "kind": "LORA_RECALIBRATION_TOKENIZER_BOOTSTRAP_CONFIG",
            "version": 1,
            "hostname": "target-node",
            "uid": os.getuid(),
            "interpreter": "/env/bin/python",
            "resolved_interpreter": "/base/bin/python3.12",
            "site_packages": str(self.site_dir),
            "source": str(self.source_dir),
            "parent": str(self.parent_dir),
            "stage": str(self.parent_dir / ("bootstrap-" + self.run_id)),
            "run_id": self.run_id,
            "sources_sha256": self.sources_sha,
            "projection_sha256": hashlib.sha256(self.projection_raw).hexdigest(),
            "profile_sha256": self.profile_sha,
            "payload_sha256": hashlib.sha256(self.payload_raw).hexdigest(),
            "payload_bytes": len(self.payload_raw),
            "work_timeout_seconds": 1200,
        }

    def test_pure_stdlib_and_style(self):
        for fname in (
            "recalibration_tokenizer_remote.py",
            "test_recalibration_tokenizer_remote.py",
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

    def test_frame_codec_and_bundle_validation(self):
        framed = tok_remote.encode_frame("READY", b'{"ok":true}')
        full_wire = (
            framed
            + tok_remote.encode_frame("RESULT", b"{}")
            + tok_remote.encode_frame("STDOUT", b"")
            + tok_remote.encode_frame("STDERR", b"")
            + tok_remote.encode_frame("DONE", b"{}")
        )
        decoded = tok_remote.decode_frames(full_wire)
        self.assertEqual(decoded["READY"], b'{"ok":true}')
        sources, proj, prof = tok_remote.validate_bundle(self.payload_raw, self.config)
        self.assertEqual(proj["rows"][1]["ordinal"], 1)
        self.assertEqual(hashlib.sha256(prof).hexdigest(), self.profile_sha)
        with self.assertRaises(ValueError):
            tok_remote.validate_bundle(self.payload_raw + b"x", self.config)

    def test_run_session_loop_stops_on_over_limit_and_aborts_on_exception(self):
        measured_calls = []

        class FakeSession:
            def __init__(self, status_seq, fail_measure=False):
                self.status_seq = list(status_seq)
                self.fail_measure = fail_measure
                self.aborted = False
                self.finished = False

            def measure(self, messages, answer=None):
                measured_calls.append((messages, answer))
                if self.fail_measure:
                    raise ValueError("session measure failed")
                status_val = self.status_seq.pop(0)
                return {
                    "kind": "TOKENIZER_SESSION_MEASUREMENT_PROVISIONAL",
                    "ordinal": len(measured_calls) - 1,
                    "measurement": {"status": status_val},
                }

            def finish(self):
                self.finished = True
                return {"kind": "TOKENIZER_SESSION_FINALIZATION", "measurements": len(measured_calls)}

            def abort(self):
                self.aborted = True

        sess_ok = FakeSession(["OK", "OK"])
        out_ok = tok_remote.execute_projection_on_session(
            self.projection["rows"], lambda: sess_ok
        )
        self.assertIsNone(out_ok["failure"])
        self.assertFalse(out_ok["abort_failed"])
        self.assertEqual(len(out_ok["records"]), 2)
        self.assertTrue(sess_ok.finished)

        measured_calls.clear()
        sess_stop = FakeSession(["INPUT_LIMIT", "OK"])
        out_stop = tok_remote.execute_projection_on_session(
            self.projection["rows"], lambda: sess_stop
        )
        self.assertEqual(len(out_stop["records"]), 1)
        self.assertTrue(sess_stop.finished)

        measured_calls.clear()
        sess_err = FakeSession(["OK"], fail_measure=True)
        out_err = tok_remote.execute_projection_on_session(
            self.projection["rows"], lambda: sess_err
        )
        self.assertEqual(out_err["failure"], {"phase": "MEASURE", "ordinal": 0})
        self.assertIsNone(out_err["finalization"])
        self.assertTrue(sess_err.aborted)


if __name__ == "__main__":
    unittest.main()
