import hashlib
import importlib
import json
from pathlib import Path
import unittest
from unittest import mock

import test_candidate_execution_controller as original_tests
from validation_contract import digest


class FinalControllerTests(original_tests.ControllerTests):
    def setUp(self):
        super().setUp()
        self.assertTrue(Path(__file__).with_name("development_final_controller.py").is_file(),
                        "final controller implementation missing")
        self.module = importlib.import_module("development_final_controller")
        self.request.update({
            "kind": "LORA_DEVELOPMENT_FINAL_REQUEST", "round": 2,
            "run_id": "a" * 32, "attempt_id": "b" * 32,
            "generation_call_id": "c" * 32, "generation_sha256": "d" * 64,
            "case_set_sha256": "e" * 64, "case_group": "scoring", "case_id": "C01",
            "entry_point": "toy", "source": "def toy(value):\n    return value\n",
        })

    def test_complete_controller_is_exact_sealed_import_derivative(self):
        path = Path(__file__).with_name("candidate_execution_controller.py")
        original_bytes = path.read_bytes()
        original = original_bytes.decode()
        self.assertEqual(hashlib.sha256(original_bytes).hexdigest(),
                         "a9c72892b97eb4cb69492cabd53cff3be2822a4837291fc59709d0be61ee4bc0")
        old = (
            "from candidate_execution_protocol import (\n"
            "    InfrastructureError, READY, make_receipt, prepare_request,\n"
            ")\n"
        )
        new = (
            "from candidate_execution_protocol import InfrastructureError, READY, make_receipt\n"
            "from development_final_protocol import prepare_request\n"
        )
        self.assertEqual(original.count(old), 1)
        self.assertEqual(Path(self.module.__file__).read_bytes(), original.replace(old, new).encode())
        protocol = importlib.import_module("development_final_protocol")
        self.assertIs(self.module.prepare_request, protocol.prepare_request)
        self.assertEqual(hashlib.sha256(Path(protocol.__file__).read_bytes()).hexdigest(),
                         "9315bfec3ed19604c21d8ec6a944d264828f8e92618416748d377d77f36ad141")

    def test_old_public_envelopes_and_invalid_final_requests_never_launch(self):
        old = {name: self.request[name] for name in (
            "version", "round", "attempt_id", "task_id", "arm",
            "case_id", "entry_point", "source", "args")}
        old.update(round=1, case_id="P01")
        for request in (
            old, {**self.request, "round": 1},
            {**self.request, "case_group": "public"},
            {**self.request, "kind": "COMPUTATION_AST_ADMISSION"},
        ):
            with self.subTest(keys=sorted(request)), \
                    mock.patch.object(self.module.subprocess, "Popen") as launch:
                with self.assertRaises(ValueError):
                    self.module.run_request(request, [], artifact_dir=self.root,
                                            cwd=self.root, env={})
                launch.assert_not_called()
            self.assertEqual(list(self.root.iterdir()), [])

    def test_final_request_is_not_replayed_through_old_controller(self):
        old = importlib.import_module("candidate_execution_controller")
        for request in (self.request, {**self.request, "case_group": "public", "case_id": "P01"}):
            with mock.patch.object(old.subprocess, "Popen") as launch:
                with self.assertRaises(ValueError):
                    old.run_request(request, [], artifact_dir=self.root, cwd=self.root, env={})
                launch.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_final_metadata_changes_bind_receipt_claim_and_audit(self):
        baseline = dict(self.request)
        base_root = self.root
        baseline_digest = digest(baseline)
        payload = self.module.prepare_request(baseline)[1]
        changes = (
            {"run_id": "f" * 32}, {"attempt_id": "f" * 32},
            {"generation_call_id": "f" * 32}, {"generation_sha256": "f" * 64},
            {"case_set_sha256": "f" * 64}, {"task_id": "D04"},
            {"arm": "text"}, {"case_id": "C02"},
            {"case_group": "public", "case_id": "P01"},
        )
        for ordinal, changed in enumerate(changes):
            with self.subTest(changed=changed):
                self.root = base_root / str(ordinal)
                self.root.mkdir()
                self.request = {**baseline, **changed}
                result = self.run_helper()
                expected = digest(self.request)
                self.assertNotEqual(expected, baseline_digest)
                claim = json.loads((self.root / "claim.json").read_bytes())
                saved = json.loads((self.root / "observation.json").read_bytes())
                self.assertEqual(saved, result["audit"])
                for record in (claim, result["receipt"], result["audit"]):
                    self.assertEqual(record["request_sha256"], expected)
                self.assertEqual(claim["payload_sha256"], hashlib.sha256(payload).hexdigest())
                self.assertEqual(result["audit"]["input_bytes_sent"], len(payload))
                for name in ("stdout", "stderr", "control"):
                    self.assertEqual(
                        result["audit"]["stream_sha256"][name],
                        hashlib.sha256((self.root / (name + ".bin")).read_bytes()).hexdigest())

    def test_trusted_helper_reads_only_three_fields_and_never_executes_source(self):
        self.request["source"] = "def toy(value):\n    raise RuntimeError('must not execute')\n"
        helper = original_tests.PREFIX + (
            "decoded = json.loads(payload)\n"
            "assert set(decoded) == {'source', 'entry_point', 'args'}\n"
            "assert decoded['entry_point'] == 'toy'\n"
            "assert decoded['args'] == [7]\n"
            "os.write(1, payload)\n"
        ) + original_tests.RETURN
        result = self.run_helper(helper)
        self.assertEqual(result["receipt"]["outcome"], {"kind": "return", "value": 7})
        expected = self.module.prepare_request(self.request)[1]
        self.assertEqual((self.root / "stdout.bin").read_bytes(), expected)
        self.assertEqual(set(json.loads(expected)), {"source", "entry_point", "args"})
        self.assertEqual(result["receipt"]["request_sha256"], digest(self.request))

    def test_result_then_output_flood_cannot_report_success(self):
        base_root = self.root
        for descriptor in (1, 2):
            with self.subTest(descriptor=descriptor):
                self.root = base_root / str(descriptor)
                self.root.mkdir()
                helper = original_tests.PREFIX + original_tests.RETURN + (
                    "while True:\n    os.write(" + str(descriptor) + ", b'x' * 8192)\n"
                )
                result = self.run_helper(helper, output_limit=1024)
                self.assertEqual(result["receipt"]["status"], "output_limit")
                self.assertEqual(result["audit"]["termination"], "output_limit")
                self.assertTrue(result["audit"]["reaped"])
                self.assertLessEqual(
                    sum(path.stat().st_size for path in self.root.glob("*.bin")), 1025)


if __name__ == "__main__":
    unittest.main()
