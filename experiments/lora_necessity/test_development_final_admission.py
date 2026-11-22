import ast
import copy
import errno
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from validation_contract import digest


class FinalAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("development_final_admission.py").is_file(),
                        "final admission implementation missing")
        self.module = importlib.import_module("development_final_admission")
        self.controller = importlib.import_module("development_final_controller")
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.request = {
            "kind": "LORA_DEVELOPMENT_FINAL_REQUEST", "version": 1, "round": 2,
            "run_id": "a" * 32, "attempt_id": "b" * 32,
            "generation_call_id": "c" * 32, "generation_sha256": "d" * 64,
            "case_set_sha256": "e" * 64, "task_id": "D01", "arm": "base",
            "case_group": "scoring", "case_id": "C01", "entry_point": "toy",
            "source": "def toy(value):\n    return value\n", "args": [7],
        }

    def request_body(self, body):
        return {**self.request, "source": "def toy(value):\n    " + body + "\n"}

    def run_request(self, request=None, **limits):
        return self.module.run_candidate_request(
            self.request if request is None else request, [],
            artifact_dir=self.root, cwd=self.root, env={}, **limits)

    def test_admission_body_and_policy_equal_sealed_source_except_kind(self):
        original = importlib.import_module("candidate_execution_admission")
        protocol = importlib.import_module("development_final_protocol")
        self.assertEqual(hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
                         "61fd51d9fda1552d443c3b0e715436d9a05eafdeda284a7cc22bee3ad3902999")
        source = inspect.getsource(original.admit_request)
        self.assertEqual(source.count('"COMPUTATION_AST_ADMISSION"'), 1)
        expected = source.replace('"COMPUTATION_AST_ADMISSION"',
                                  '"DEVELOPMENT_FINAL_AST_ADMISSION"')
        actual = inspect.getsource(self.module.admit_request)
        self.assertEqual(actual, expected)
        self.assertEqual(ast.dump(ast.parse(actual), include_attributes=False),
                         ast.dump(ast.parse(expected), include_attributes=False))
        for name in ("CALLS", "METHODS", "NODE_NAMES", "POLICY_VERSION", "POLICY_SHA256",
                     "CandidateAdmissionError", "_require", "_plain_json", "_bindings",
                     "_local_bindings", "_check_scope"):
            self.assertIs(getattr(self.module, name), getattr(original, name))
        self.assertIs(self.module.prepare_request, protocol.prepare_request)
        with mock.patch.object(original, "prepare_request",
                               side_effect=AssertionError("old request conversion")):
            self.assertEqual(self.module.admit_request(self.request)["request_sha256"],
                             digest(self.request))

    def test_exact_final_admission_policy_and_complete_request_binding(self):
        expected = {
            "kind": "DEVELOPMENT_FINAL_AST_ADMISSION", "version": 2,
            "policy_sha256": "16fb67e48d8adba83ad0cc832053d65d2b4d71a6b0075407d1c4a0e006410859",
            "request_sha256": digest(self.request),
        }
        self.assertEqual(self.module.admit_request(self.request), expected)
        for changes in (
            {"run_id": "f" * 32}, {"attempt_id": "f" * 32},
            {"generation_call_id": "f" * 32}, {"generation_sha256": "f" * 64},
            {"case_set_sha256": "f" * 64}, {"task_id": "D04"}, {"arm": "text"},
            {"case_id": "C02"}, {"case_group": "public", "case_id": "P01"},
            {"source": self.request["source"] + "\n"}, {"args": [8]},
        ):
            with self.subTest(changes=changes):
                request = {**self.request, **changes}
                record = self.module.admit_request(request)
                self.assertEqual(record, {**expected, "request_sha256": digest(request)})
                self.assertNotEqual(record["request_sha256"], expected["request_sha256"])

    def test_ordinary_computations_are_only_statically_admitted(self):
        for body in (
            "return abs(value) if value < 0 else value",
            "result = []\n    for item in value:\n        result.append(item + 1)\n    return result",
            "return [item * 2 for item in value if item > 0]",
            "return {item: item + 1 for item in value}",
            "return sum(item for item in value)",
            "return value.strip().lower().replace('a', 'b')",
            "return toy(value - 1) if value > 0 else 0",
            "return [[item for item in outer] for outer in value]",
            "return [eval for eval in value]",
            "return sorted(value, key=abs)",
            "raise RuntimeError('must not execute')",
        ):
            with self.subTest(body=body), \
                    mock.patch("builtins.exec", side_effect=AssertionError("candidate exec")), \
                    mock.patch("builtins.eval", side_effect=AssertionError("candidate eval")):
                self.assertEqual(self.module.admit_request(self.request_body(body))["kind"],
                                 "DEVELOPMENT_FINAL_AST_ADMISSION")

    def test_unsafe_calls_and_scope_bypasses_deny_without_artifacts_or_launch(self):
        denied = (
            "return __import__('os').getpid()",
            "return open('/tmp/x').read()",
            "return value.__class__",
            "return map(eval, value)",
            "items = [eval for eval in value]\n    return map(eval, value)",
            "len = value\n    return len(value)",
            "return [eval for eval in eval]",
            "helper = abs\n    return helper(value)",
            "return value[0](1)",
            "method = value.append\n    return method(1)",
            "return '{}'.format(value)",
            "return [len for len in value]",
        )
        for body in denied:
            request = self.request_body(body)
            with self.subTest(body=body), \
                    mock.patch.object(self.controller, "run_request") as launch:
                with self.assertRaises(self.module.CandidateAdmissionError):
                    self.module.admit_request(request)
                with self.assertRaises(self.module.CandidateAdmissionError):
                    self.run_request(request)
                launch.assert_not_called()
                self.assertEqual(list(self.root.iterdir()), [])

    def test_legacy_public_envelopes_never_admit_or_launch_in_final_domain(self):
        old = {name: self.request[name] for name in (
            "version", "round", "attempt_id", "task_id", "arm",
            "case_id", "entry_point", "source", "args")}
        old.update(round=1, case_id="P01")
        requests = (old, {**self.request, "round": 1}, {**self.request, "round": True},
                    {**self.request, "case_group": "scoring", "case_id": "P01"})
        for request in requests:
            with self.subTest(keys=sorted(request)), \
                    mock.patch.object(self.controller, "run_request") as launch:
                with self.assertRaises(self.module.CandidateAdmissionError):
                    self.module.admit_request(request)
                with self.assertRaises(self.module.CandidateAdmissionError):
                    self.run_request(request)
                launch.assert_not_called()
                self.assertEqual(list(self.root.iterdir()), [])
        original = importlib.import_module("candidate_execution_admission")
        with self.assertRaises(original.CandidateAdmissionError):
            original.admit_request(self.request)

    def test_hostile_builtin_subclasses_are_rejected_without_hooks(self):
        hooks = []

        def hostile(*arguments, **keywords):
            hooks.append("called")
            raise AssertionError("hostile hook called")

        overrides = {name: hostile for name in (
            "__iter__", "__getitem__", "__len__", "__repr__", "__str__", "__eq__",
            "__int__", "__float__", "items", "keys", "encode",
        )}
        hostile_values = [type("Hostile" + base.__name__, (base,), overrides)(value)
                          for base, value in ((dict, {}), (list, []), (int, 1),
                                              (float, 1.0), (str, "value"))]
        hostile_key = type("HostileKey", (str,), {**overrides, "__hash__": str.__hash__})
        requests = [{hostile_key("extra"): 1}]
        for value in hostile_values:
            requests.extend((value, {**self.request, "args": [{"nested": [value]}]},
                             {**self.request, "generation_sha256": value},
                             {**self.request, "source": value}))
        for ordinal, request in enumerate(requests):
            with self.subTest(ordinal=ordinal), \
                    mock.patch.object(self.controller, "run_request") as launch:
                with self.assertRaises(self.module.CandidateAdmissionError):
                    self.module.admit_request(request)
                with self.assertRaises(self.module.CandidateAdmissionError):
                    self.run_request(request)
                launch.assert_not_called()
                self.assertEqual(hooks, [])
                self.assertEqual(list(self.root.iterdir()), [])

    def test_persistence_closes_before_launch_and_snapshot_is_independent(self):
        self.request["args"] = [{"nested": [{"values": [7]}]}]
        original = copy.deepcopy(self.request)
        expected = self.module.admit_request(original)
        streams = []
        original_fdopen = os.fdopen
        limits = {"startup_timeout": 1.0, "candidate_timeout": 0.5,
                  "output_limit": 4096, "result_limit": 1024, "reap_timeout": 0.5}

        def opened(descriptor, mode):
            stream = original_fdopen(descriptor, mode)
            streams.append(stream)
            return stream

        def launched(snapshot, argv, **options):
            self.assertTrue(streams and all(stream.closed for stream in streams))
            self.assertEqual(json.loads((self.root / "admission.json").read_bytes()), expected)
            self.assertEqual((self.root / "admission.json").stat().st_mode & 0o777, 0o600)
            self.assertEqual(options, {"artifact_dir": self.root, "cwd": self.root,
                                       "env": {}, **limits})
            self.assertEqual(argv, [])
            self.assertIsNot(snapshot, self.request)
            self.assertIsNot(snapshot["args"][0]["nested"][0]["values"],
                             self.request["args"][0]["nested"][0]["values"])
            self.request["args"][0]["nested"][0]["values"].append(99)
            self.request["source"] += "\n"
            self.request["generation_sha256"] = "f" * 64
            self.assertEqual(snapshot, original)
            return {"receipt": {"request_sha256": digest(snapshot), "status": "program_error"},
                    "audit": {"synthetic": True}}

        with mock.patch.object(self.module.os, "fdopen", side_effect=opened), \
                mock.patch.object(self.controller, "run_request", side_effect=launched) as launch:
            result = self.run_request(**limits)
        launch.assert_called_once()
        self.assertEqual(result["admission"], expected)
        self.assertEqual(result["audit"], {"synthetic": True})

    def test_exclusive_duplicate_admission_never_launches_or_overwrites(self):
        path = self.root / "admission.json"
        path.write_bytes(b"existing admission marker")
        with mock.patch.object(self.controller, "run_request") as launch:
            with self.assertRaises(self.module.CandidateAdmissionError) as raised:
                self.run_request()
            launch.assert_not_called()
        self.assertIsInstance(raised.exception.__cause__, FileExistsError)
        self.assertEqual(path.read_bytes(), b"existing admission marker")
        self.assertEqual(sorted(item.name for item in self.root.iterdir()), ["admission.json"])

    def test_open_refusal_never_launches_or_creates_claim(self):
        failure = PermissionError("synthetic persistence refusal")
        with mock.patch.object(self.module.os, "open", side_effect=failure), \
                mock.patch.object(self.controller, "run_request") as launch:
            with self.assertRaises(self.module.CandidateAdmissionError) as raised:
                self.run_request()
            launch.assert_not_called()
        self.assertIs(raised.exception.__cause__, failure)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_fdopen_failure_closes_descriptor_and_retains_claimed_admission(self):
        original_open = os.open
        for ordinal, error_type in enumerate((OSError, ValueError)):
            folder = self.root / str(ordinal)
            folder.mkdir()
            descriptors = []

            def opened(*arguments, **keywords):
                descriptor = original_open(*arguments, **keywords)
                descriptors.append(descriptor)
                return descriptor

            failure = error_type("synthetic fdopen failure")
            with self.subTest(error_type=error_type.__name__), \
                    mock.patch.object(self.module.os, "open", side_effect=opened), \
                    mock.patch.object(self.module.os, "fdopen", side_effect=failure), \
                    mock.patch.object(self.controller, "run_request") as launch:
                with self.assertRaises(self.module.CandidateAdmissionError) as raised:
                    self.module.run_candidate_request(
                        self.request, [], artifact_dir=folder, cwd=folder, env={})
                launch.assert_not_called()
            self.assertIs(raised.exception.__cause__, failure)
            self.assertEqual(len(descriptors), 1)
            with self.assertRaises(OSError) as closed:
                os.fstat(descriptors[0])
            self.assertEqual(closed.exception.errno, errno.EBADF)
            self.assertEqual(sorted(item.name for item in folder.iterdir()), ["admission.json"])

    def test_write_short_write_and_close_failures_never_launch(self):
        original_fdopen = os.fdopen
        for phase in ("write", "short_write", "close"):
            for error_type in (OSError, ValueError):
                folder = self.root / (phase + error_type.__name__)
                folder.mkdir()
                failure = error_type("synthetic " + phase + " failure")
                streams = []

                def opened(descriptor, mode):
                    stream = original_fdopen(descriptor, mode)
                    streams.append(stream)
                    output = mock.MagicMock(spec=type(stream), wraps=stream)
                    output.__enter__.return_value = output
                    if phase == "write":
                        output.write.side_effect = failure
                    elif phase == "short_write":
                        output.write.return_value = 0

                    def closed(*arguments):
                        stream.close()
                        if phase == "close":
                            raise failure
                        return False

                    output.__exit__.side_effect = closed
                    return output

                with self.subTest(phase=phase, error_type=error_type.__name__), \
                        mock.patch.object(self.module.os, "fdopen", side_effect=opened), \
                        mock.patch.object(self.controller, "run_request") as launch:
                    with self.assertRaises(self.module.CandidateAdmissionError) as raised:
                        self.module.run_candidate_request(
                            self.request, [], artifact_dir=folder, cwd=folder, env={})
                    launch.assert_not_called()
                self.assertTrue(streams and all(stream.closed for stream in streams))
                self.assertEqual(sorted(item.name for item in folder.iterdir()), ["admission.json"])
                if phase != "short_write":
                    self.assertIs(raised.exception.__cause__, failure)

    def test_controller_failure_retains_admission_and_is_not_retried(self):
        protocol = importlib.import_module("candidate_execution_protocol")
        failure = protocol.InfrastructureError("synthetic controller failure")
        expected = self.module.admit_request(self.request)
        with mock.patch.object(self.controller, "run_request", side_effect=failure) as launch:
            with self.assertRaises(protocol.InfrastructureError) as raised:
                self.run_request()
            self.assertIs(raised.exception, failure)
            launch.assert_called_once()
        self.assertEqual(json.loads((self.root / "admission.json").read_bytes()), expected)
        self.assertEqual(sorted(item.name for item in self.root.iterdir()), ["admission.json"])

    def test_controller_receipt_must_match_admitted_final_request(self):
        with mock.patch.object(self.controller, "run_request", return_value={
                "receipt": {"request_sha256": "0" * 64}, "audit": {}}) as launch:
            with self.assertRaisesRegex(self.module.CandidateAdmissionError,
                                        "admitted request changed"):
                self.run_request()
            launch.assert_called_once()
        self.assertEqual(json.loads((self.root / "admission.json").read_bytes()),
                         self.module.admit_request(self.request))

    def test_trusted_literal_helper_integration_never_executes_payload_source(self):
        self.request["source"] = "def toy(value):\n    raise RuntimeError('must not execute')\n"
        helper = (
            "import json,os,sys\n"
            "control=int(sys.argv[1])\n"
            "os.write(control,b'TLF_CANDIDATE_READY_V1\\n')\n"
            "payload=sys.stdin.buffer.read()\n"
            "decoded=json.loads(payload)\n"
            "assert set(decoded)=={'source','entry_point','args'}\n"
            "assert decoded['entry_point']=='toy' and decoded['args']==[7]\n"
            "os.write(1,payload)\n"
            "os.write(control,b'{\"kind\":\"return\",\"value\":7}\\n')\n"
        )
        result = self.module.run_candidate_request(
            self.request, [sys.executable, "-I", "-S", "-B", "-c", helper, "{control_fd}"],
            artifact_dir=self.root, cwd=self.root, env={"PATH": os.defpath},
            startup_timeout=2.0, candidate_timeout=0.5)
        self.assertEqual(result["receipt"]["outcome"], {"kind": "return", "value": 7})
        self.assertEqual(result["receipt"]["status"], "completed")
        self.assertEqual(result["receipt"]["request_sha256"], digest(self.request))
        for name in ("admission", "receipt"):
            self.assertEqual(json.loads((self.root / (name + ".json")).read_bytes()), result[name])
        self.assertEqual(json.loads((self.root / "stdout.bin").read_bytes()),
                         {name: self.request[name] for name in ("source", "entry_point", "args")})
        self.assertEqual(result["admission"]["kind"], "DEVELOPMENT_FINAL_AST_ADMISSION")
        self.assertTrue(result["audit"]["ready"] and result["audit"]["input_complete"]
                        and result["audit"]["reaped"])


if __name__ == "__main__":
    unittest.main()
