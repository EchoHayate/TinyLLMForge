import ast
import copy
import hashlib
import importlib
import io
import json
from pathlib import Path
import tokenize
import types
import unittest
from unittest import mock

from confirmatory_v2_contract import CONFIRMATORY_FILE_PINS, load_confirmatory_dataset
from validation_contract import digest, encoded


REPO_LORA = Path(__file__).resolve().parent

with mock.patch("subprocess.Popen") as import_spawn, mock.patch("os.killpg") as import_killpg:
    try:
        gate_a = importlib.import_module("confirmatory_v2_gate_a")
    except ModuleNotFoundError as import_err:
        if import_err.name != "confirmatory_v2_gate_a":
            raise
        gate_a = None
    assert not import_spawn.called
    assert not import_killpg.called


class ConfirmatoryGateATests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(gate_a, "confirmatory_v2_gate_a is absent")

    def test_style_and_no_single_letter_identifiers(self):
        for fname in ("confirmatory_v2_gate_a.py", "test_confirmatory_v2_gate_a.py"):
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

    def test_gate_a_package_and_simulated_1280_dynamic_plus_200_tokenizer_scoring(self):
        data_dir = REPO_LORA / "data/confirmatory_v2_r1"
        raw_files = {
            fname: (data_dir / fname).read_bytes()
            for fname in CONFIRMATORY_FILE_PINS
        }
        worker_raw = (REPO_LORA / "validation_task_worker.py").read_bytes()
        backend_raw = (REPO_LORA / "validation_remote.py").read_bytes()
        expected_runtime = {
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
        preflight_raw = encoded({
            "kind": "LORA_RECALIBRATION_LIVE_PREFLIGHT_RECEIPT",
            "version": 1,
            "status": "PREFLIGHT_PASSED",
            "probes_passed": True,
            "remote_python": "/opt/cpython/bin/python3.12",
            "expected_runtime": expected_runtime,
        })
        tok_sources = {
            fname: (REPO_LORA / fname).read_bytes()
            for fname in gate_a.TOKENIZER_SOURCE_PINS
        }
        pkg = gate_a.build_gate_a_package(
            raw_files,
            worker_raw,
            backend_raw,
            tok_sources,
            preflight_raw,
            stage_id="aa" * 16,
            dynamic_run_ids=[("%02x" % idx) + ("bb" * 15) for idx in range(40)],
            ssh_argv_prefix=["/usr/bin/ssh", "-T", "rl-server"],
            ssh_environment={"PATH": "/usr/bin:/bin", "LANG": "C"},
            interpreter="/env/bin/python",
            resolved_interpreter="/opt/cpython/bin/python3.12",
            site_packages="/env/lib/python3.12/site-packages",
            checkpoint_root="/models/Qwen3-8B",
            expected_tokenizer_sha256="7" * 64,
            remote_parent="/tmp/g4-a",
            local_run_root="/tmp/local/" + ("aa" * 16),
        )
        self.assertEqual(pkg["public"]["planned_dynamic_batches"], 40)
        self.assertEqual(pkg["public"]["planned_tokenizer_rows"], 200)
        self.assertIn("close_fds=True", pkg["private"]["dynamic_ssh_argv"][-1])

        tok_summary = gate_a.summarize_tokenizer_audit_records(
            raw_files,
            [
                {
                    "ordinal": idx,
                    "status": "OK",
                    "state_sha256": "7" * 64,
                    "input_tokens": 3100 if idx >= 8 else 520,
                    "target_tokens": 95 if idx < 8 or (56 <= idx < 152) else None,
                }
                for idx in range(200)
            ],
            expected_tokenizer_sha256="7" * 64,
        )
        self.assertTrue(tok_summary["all_200_rows_within_budget"])
        self.assertEqual(tok_summary["max_input_tokens"], 3100)
        self.assertEqual(tok_summary["max_target_tokens"], 95)


if __name__ == "__main__":
    unittest.main()
