import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest

from recalibration_data_contract import FILE_PINS
from recalibration_execution_package import SOURCE_PINS, build_preparation
from recalibration_validation import build_batches, build_schedule
from validation_contract import digest, encoded

if importlib.util.find_spec("recalibration_execution_evidence") is not None:
    import recalibration_execution_evidence as evidence
else:
    evidence = None


ROOT = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def frame(channel, raw):
    return b"LRV01" + bytes([channel]) + len(raw).to_bytes(8, "big") + raw


def process(raw=b"", status="completed"):
    return {
        "status": status, "returncode": 0 if status == "completed" else -9,
        "stdout_hex": raw.hex(), "stderr_hex": "", "elapsed_ns": 100,
        "process_reaped": True, "stdin_complete": True,
    }


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_files = {
            name: (ROOT / "data" / (
                name if name == "supervision_v1.json"
                else "recalibration_v2_r1/" + name)).read_bytes()
            for name in FILE_PINS
        }
        cls.worker_raw = (ROOT / "validation_task_worker.py").read_bytes()
        cls.backend_raw = (ROOT / "validation_remote.py").read_bytes()
        cls.source_raw = {name: (ROOT / name).read_bytes() for name in SOURCE_PINS}
        cls.stage_id = "f" * 32
        cls.run_ids = ["%032x" % index for index in range(15)]
        cls.preparation = build_preparation(
            cls.raw_files, cls.worker_raw, cls.backend_raw, cls.source_raw,
            stage_id=cls.stage_id, run_ids=cls.run_ids)
        cls.plan = build_schedule(cls.raw_files)
        cls.batches = build_batches(cls.raw_files, cls.worker_raw, cls.run_ids)
        cls.runtime = {
            "hostname": "synthetic-only", "uid": 123, "python": "3.12.8",
            "python_root": "/synthetic/python", "python_sha256": "1" * 64,
            "bwrap_sha256": "2" * 64, "seccomp_sha256": "3" * 64,
            "argv_prefix": ["/usr/bin/bwrap", "--seccomp", "3", "/runtime/bin/python3.12"],
            "no_new_privs": True,
            "host_namespaces": {name: name + ":[123]"
                                for name in ("mnt", "pid", "net", "user")},
        }
        cls.context = {
            "package_sha256": "a" * 64, "config_sha256": "b" * 64,
            "stage_root": "/synthetic/" + cls.stage_id,
            "expected_runtime": cls.runtime,
        }

    def setUp(self):
        self.assertIsNotNone(evidence, "semantic evidence not implemented")

    def expected(self, index=0):
        public = self.preparation["public"]
        return {
            "kind": "LORA_RECALIBRATION_READY", "version": 1,
            "stage_id": self.stage_id, "batch_index": index,
            "run_id": self.run_ids[index],
            "preparation_sha256": public["preparation_sha256"],
            "package_sha256": self.context["package_sha256"],
            "config_sha256": self.context["config_sha256"],
            "bundle_sha256": self.batches[index]["bundle_sha256"],
            "backend_sha256": public["backend_sha256"],
            "worker_sha256": public["worker_sha256"],
            "hostname": self.runtime["hostname"], "uid": self.runtime["uid"],
            "stage_root": self.context["stage_root"],
            "batch_root": self.context["stage_root"] + "/batch-%02d-" % index
            + self.run_ids[index],
            "expected_runtime_sha256": digest(self.runtime),
        }

    def events(self, index):
        batch = self.batches[index]
        result = [{
            "kind": "ready", "run_id": batch["run_id"],
            "bundle_sha256": batch["bundle_sha256"],
            "worker_sha256": batch["worker_sha256"], "runtime": copy.deepcopy(self.runtime),
        }]
        for ordinal, row in enumerate(self.plan["rows"][batch["start"]:batch["stop"]]):
            outcome = row["witness_prediction"]
            if outcome is None:
                outcome = row["expected"]
            result.append({
                "kind": "row", "run_id": batch["run_id"], "ordinal": ordinal,
                "request_sha256": row["request_sha256"],
                "result": process(encoded(outcome)),
            })
        result.append({
            "kind": "done", "run_id": batch["run_id"], "received": 32,
            "elapsed_ns": 80, "stopped": False,
        })
        return result

    def payloads(self, index=0, events=None):
        expected = self.expected(index)
        ready = encoded({**expected, "ready_elapsed_ns": 10})
        stdout = b"".join(encoded(event) + b"\n" for event in (
            self.events(index) if events is None else events))
        payloads = [ready, stdout, b"backend diagnostic", b"\xffprivate diagnostic"]
        backend = {name: value for name, value in process().items()
                   if name not in ("stdout_hex", "stderr_hex")}
        done = {
            "kind": "LORA_RECALIBRATION_DONE", "version": 1,
            **{name: expected[name] for name in (
                "stage_id", "batch_index", "run_id", "package_sha256", "config_sha256")},
            "ready_sha256": sha(ready), "payload_sha256": expected["bundle_sha256"],
            "prior_frames_sha256": [sha(frame(channel, raw))
                                    for channel, raw in enumerate(payloads, 1)],
            "status": "completed", "payload_received": True, "backend_pid": 321,
            "backend": backend, "work_elapsed_ns": 1000, "cleanup_elapsed_ns": 0,
        }
        return payloads, done

    def wire(self, payloads, done):
        return b"".join(frame(channel, raw) for channel, raw in enumerate(
            payloads + [encoded(done)], 1))

    def record(self, index=0, events=None):
        payloads, done = self.payloads(index, events)
        wire_raw = self.wire(payloads, done)
        return {
            "batch_index": index, "bundle_raw": self.batches[index]["bundle_raw"],
            "ssh": process(wire_raw), "wire_sha256": sha(wire_raw),
        }

    def record_wire(self, wire_raw):
        record = self.record()
        record["ssh"]["stdout_hex"] = wire_raw.hex()
        record["wire_sha256"] = sha(wire_raw)
        return record

    def decode(self, payloads, done, expected=None):
        raw = self.wire(payloads, done)
        return evidence.decode_receipt(
            raw, expected_sha256=sha(raw),
            expected=self.expected() if expected is None else expected)

    def report(self, records, **overrides):
        arguments = {
            "preparation": self.preparation, "raw_files": self.raw_files,
            "worker_raw": self.worker_raw, "backend_raw": self.backend_raw,
            "source_raw": self.source_raw, "records": records,
            "stage_id": self.stage_id, "run_ids": self.run_ids, "context": self.context,
        }
        arguments.update(overrides)
        return evidence.build_evidence(**arguments)

    def test_ready_returns_detached_exact_observation(self):
        expected = self.expected()
        observed = {**expected, "ready_elapsed_ns": 10_000_000_000}
        self.assertEqual(evidence.validate_ready(encoded(observed), expected=expected),
                         observed)
        self.assertNotIn("ready_elapsed_ns", expected)

    def test_ready_rejects_every_identity_change(self):
        expected = self.expected()
        for name, value in expected.items():
            changed = {**expected, "ready_elapsed_ns": 10}
            changed[name] = value + 1 if type(value) is int else value + "x"
            with self.subTest(name=name), self.assertRaises(ValueError):
                evidence.validate_ready(encoded(changed), expected=expected)

    def test_ready_strict_json_and_timing(self):
        expected = self.expected()
        valid = encoded({**expected, "ready_elapsed_ns": 0})
        malformed = [
            valid[:-1] + b',"version":1}', valid[:-1] + b',"extra":0}',
            valid.replace(b'"version":1', b'"version":true'),
            valid.replace(b'"uid":123', b'"uid":123.0'),
            valid.replace(b'"ready_elapsed_ns":0', b'"ready_elapsed_ns":NaN'),
            b"\xff", b"{}", b"[]" , b" " * 4097,
        ]
        malformed.extend(encoded({**expected, "ready_elapsed_ns": value})
                         for value in (-1, True, 10_000_000_001, None, "0"))
        for raw in malformed:
            with self.subTest(raw=raw[:70]), self.assertRaises(ValueError):
                evidence.validate_ready(raw, expected=expected)

    def test_ready_rejects_invalid_expected_even_if_matching(self):
        for name, value in (
                ("version", True), ("uid", True), ("batch_index", -1),
                ("batch_index", 15), ("package_sha256", "A" * 64),
                ("config_sha256", "b" * 64 + "\n"), ("stage_id", "bad"),
                ("run_id", self.stage_id), ("stage_root", "/tmp/../" + self.stage_id),
                ("batch_root", "/other"), ("hostname", ""),
                ("backend_sha256", "0" * 64), ("worker_sha256", "0" * 64)):
            expected = {**self.expected(), name: value}
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                evidence.validate_ready(
                    encoded({**expected, "ready_elapsed_ns": 0}), expected=expected)

    def test_ready_rejects_subclasses(self):
        class Text(str):
            pass

        class Bytes(bytes):
            pass

        expected = self.expected()
        raw = encoded({**expected, "ready_elapsed_ns": 0})
        with self.assertRaises(ValueError):
            evidence.validate_ready(Bytes(raw), expected=expected)
        expected["hostname"] = Text("synthetic-only")
        with self.assertRaises(ValueError):
            evidence.validate_ready(raw, expected=expected)

    def test_receipt_reconstructs_exact_raw_process_bytes(self):
        payloads, done = self.payloads()
        result = self.decode(payloads, done)
        self.assertEqual(result["transport"]["stdout_hex"], payloads[1].hex())
        self.assertEqual(result["transport"]["stderr_hex"], payloads[2].hex())
        self.assertTrue(result["receipt_binding_consistent"])
        self.assertTrue(result["backend_lifecycle_complete"])
        self.assertTrue(result["backend_started"])
        self.assertEqual(result["done"], done)
        self.assertEqual(set(result), {
            "ready", "done", "transport", "receipt_binding_consistent",
            "backend_lifecycle_complete", "payload_received", "backend_started",
        })

    def test_receipt_rejects_stale_digest_before_semantic_parse(self):
        with self.assertRaisesRegex(ValueError, "wire digest"):
            evidence.decode_receipt(b"not a wire", expected_sha256="0" * 64,
                                    expected=self.expected())

    def test_done_rejects_identity_and_frame_digest_changes(self):
        payloads, done = self.payloads()
        changes = [
            ("kind", "wrong"), ("version", True), ("stage_id", "0" * 32),
            ("batch_index", True), ("run_id", "e" * 32),
            ("package_sha256", "c" * 64), ("config_sha256", "d" * 64),
            ("ready_sha256", "0" * 64), ("payload_sha256", "0" * 64),
            ("prior_frames_sha256", []),
            ("prior_frames_sha256", [sha(raw) for raw in payloads]),
        ]
        for name, value in changes:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.decode(payloads, {**done, name: value})
        for index in range(4):
            changed = list(payloads)
            changed[index] += b" "
            with self.subTest(frame=index), self.assertRaises(ValueError):
                self.decode(changed, done)

    def test_done_rejects_json_duplicate_extra_and_nonfinite(self):
        payloads, done = self.payloads()
        prefix = b"".join(frame(channel, raw)
                          for channel, raw in enumerate(payloads, 1))
        raw_done = encoded(done)
        for ending in (b',"version":1}', b',"approved":true}',
                       b',"elapsed":Infinity}'):
            raw = prefix + frame(5, raw_done[:-1] + ending)
            with self.subTest(ending=ending), self.assertRaises(ValueError):
                evidence.decode_receipt(raw, expected_sha256=sha(raw),
                                        expected=self.expected())

    def test_done_rejects_lifecycle_contradictions(self):
        payloads, done = self.payloads()
        for name, value in (
                ("returncode", 1), ("returncode", True), ("returncode", None),
                ("process_reaped", False), ("stdin_complete", False),
                ("elapsed_ns", 1001), ("elapsed_ns", True),
                ("status", "timeout"), ("extra", True)):
            changed = copy.deepcopy(done)
            changed["backend"][name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.decode(payloads, changed)
        for name, value in (
                ("backend_pid", True), ("backend_pid", None), ("backend_pid", 0),
                ("payload_received", False), ("payload_received", 1),
                ("work_elapsed_ns", -1), ("cleanup_elapsed_ns", True),
                ("work_elapsed_ns", 95_000_000_001),
                ("cleanup_elapsed_ns", 2_000_000_001)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.decode(payloads, {**done, name: value})

    def test_failed_backend_retains_unreaped_observation(self):
        payloads, done = self.payloads()
        done.update(status="timeout", work_elapsed_ns=95_000_000_010)
        done["backend"].update(status="timeout", returncode=None, process_reaped=False)
        result = self.decode(payloads, done)
        self.assertTrue(result["receipt_binding_consistent"])
        self.assertFalse(result["backend_lifecycle_complete"])
        self.assertIsNone(result["transport"]["returncode"])

    def test_absent_backend_does_not_fabricate_transport(self):
        payloads, done = self.payloads()
        payloads[1:3] = [b"", b""]
        done.update(status="input_error", payload_received=False, payload_sha256=None,
                    backend=None, backend_pid=None)
        done["prior_frames_sha256"] = [sha(frame(channel, raw))
                                       for channel, raw in enumerate(payloads, 1)]
        result = self.decode(payloads, done)
        self.assertIsNone(result["transport"])
        self.assertFalse(result["backend_started"])
        self.assertFalse(result["backend_lifecycle_complete"])
        for name, value in (("status", "completed"), ("status", "launch_failed"),
                            ("backend_pid", 123), ("payload_sha256", "0" * 64)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.decode(payloads, {**done, name: value})

    def test_complete_synthetic_stage_never_accepts_real_execution(self):
        report = self.report([self.record(index) for index in range(15)])
        self.assertEqual(report["kind"], "LORA_RECALIBRATION_EXECUTION_EVIDENCE")
        self.assertTrue(report["receipt_binding_consistent"])
        self.assertTrue(report["collection_complete"])
        self.assertIsNone(report["first_stop"])
        content = report["content_report"]
        self.assertTrue(content["matrix_complete"])
        self.assertTrue(content["data_checks_satisfied"])
        self.assertEqual(content["completed"], 480)
        self.assertEqual(content["reference_counts"]["matched"], 192)
        self.assertEqual(content["witness_counts"]["matched"], 36)
        for name in ("binding_complete", "trusted_collection_accepted",
                     "dynamic_checks_passed", "launch_ready", "run_authorized",
                     "tokenizer_length_validated", "net_benefit_established"):
            self.assertIs(report[name], False, name)
        self.assertFalse(content["execution_provenance_verified"])
        self.assertTrue(report["missing_bindings"])

    def test_empty_prefix_is_not_consistent_collection(self):
        report = self.report([])
        self.assertFalse(report["receipt_binding_consistent"])
        self.assertFalse(report["collection_complete"])
        self.assertEqual(report["content_report"]["not_dispatched"], 480)
        self.assertEqual(report["first_stop"]["reason"], "records_exhausted")

    def test_raw_backend_report_is_reconstructed(self):
        report = self.report([self.record()])
        self.assertEqual(report["content_report"]["completed"], 32)
        self.assertEqual(report["content_report"]["not_dispatched"], 448)
        self.assertTrue(report["batches"][0]["can_continue"])
        self.assertFalse(report["binding_complete"])
        self.assertFalse(report["trusted_collection_accepted"])

    def test_semantic_reference_mismatch_does_not_stop(self):
        events = self.events(0)
        events[1]["result"] = process(encoded({"kind": "return", "value": -99999}))
        records = [self.record(0, events)] + [self.record(index) for index in range(1, 15)]
        report = self.report(records)
        self.assertTrue(report["collection_complete"])
        self.assertTrue(report["content_report"]["matrix_complete"])
        self.assertFalse(report["content_report"]["data_checks_satisfied"])
        self.assertEqual(report["content_report"]["reference_counts"]["matched"], 191)

    def test_semantic_witness_mismatch_does_not_stop(self):
        selected = next(row for row in self.plan["rows"][:32]
                        if row["witness_prediction"] is not None)
        events = self.events(0)
        events[selected["ordinal"] + 1]["result"] = process(encoded(selected["expected"]))
        records = [self.record(0, events)] + [self.record(index) for index in range(1, 15)]
        report = self.report(records)
        self.assertTrue(report["collection_complete"])
        self.assertEqual(report["content_report"]["witness_counts"]["matched"], 35)
        self.assertFalse(report["content_report"]["data_checks_satisfied"])

    def test_ssh_failure_preserves_completed_backend_but_stops(self):
        record = self.record()
        record["ssh"].update(status="program_error", returncode=255)
        report = self.report([record])
        self.assertEqual(report["content_report"]["completed"], 32)
        self.assertTrue(report["batches"][0]["backend_lifecycle_complete"])
        self.assertTrue(report["batches"][0]["ssh_lifecycle_complete"])
        self.assertFalse(report["batches"][0]["can_continue"])
        self.assertEqual(report["first_stop"]["reason"], "ssh_program_error")
        with self.assertRaises(ValueError):
            self.report([record, self.record(1)])

    def test_ssh_unreaped_or_late_success_cannot_continue(self):
        for changes in (
                {"status": "timeout", "returncode": None, "process_reaped": False},
                {"elapsed_ns": 112_000_000_001}):
            record = self.record()
            record["ssh"].update(changes)
            report = self.report([record])
            self.assertFalse(report["batches"][0]["can_continue"])
            self.assertEqual(report["content_report"]["completed"], 32)
            with self.assertRaises(ValueError):
                self.report([record, self.record(1)])

    def test_incomplete_wire_is_unknown_not_model_failure(self):
        record = self.record()
        raw = bytes.fromhex(record["ssh"]["stdout_hex"])
        for bad_raw in (raw[:-1], raw[:20], b"", raw + b"x"):
            report = self.report([self.record_wire(bad_raw)])
            self.assertEqual(report["content_report"]["unknown_after_dispatch"], 32)
            self.assertEqual(report["content_report"]["not_dispatched"], 448)
            self.assertEqual(report["content_report"]["observed"], 0)
            self.assertFalse(report["receipt_binding_consistent"])
            self.assertEqual(report["first_stop"]["reason"], "receipt_error")

    def test_stale_wire_or_bundle_cannot_bind(self):
        for name, value in (("wire_sha256", "0" * 64), ("bundle_raw", b"{}")):
            record = self.record()
            record[name] = value
            report = self.report([record])
            self.assertFalse(report["receipt_binding_consistent"])
            self.assertEqual(report["content_report"]["unknown_after_dispatch"], 32)

    def test_worker_failure_preserves_observed_and_blocks_next(self):
        events = self.events(0)
        events[1]["result"] = process(b"", "timeout")
        events = [events[0], events[1], {**events[-1], "received": 1, "stopped": True}]
        record = self.record(0, events)
        report = self.report([record])
        self.assertEqual(report["content_report"]["observed"], 1)
        self.assertEqual(report["content_report"]["unknown_after_dispatch"], 31)
        self.assertEqual(report["first_stop"]["reason"], "worker_timeout")
        with self.assertRaises(ValueError):
            self.report([record, self.record(1)])

    def test_backend_runtime_fd_drift_is_not_normalized(self):
        events = self.events(0)
        events[0]["runtime"]["argv_prefix"][2] = "7"
        report = self.report([self.record(0, events)])
        self.assertEqual(report["content_report"]["observed"], 0)
        self.assertEqual(report["first_stop"]["reason"], "evidence_error")
        self.assertFalse(report["batches"][0]["can_continue"])

    def test_rejects_record_gaps_types_and_extra_fields(self):
        for records in (
                None, (), [self.record(1)],
                [{**self.record(), "batch_index": False}],
                [{**self.record(), "real": True}],
                [{**self.record(), "ssh": None}],
                [{**self.record(), "wire_sha256": "A" * 64}]):
            with self.subTest(records=str(records)[:50]), self.assertRaises(ValueError):
                self.report(records)

    def test_rejects_context_not_full_backend_shape(self):
        for changes in (
                {"real": True}, {"package_sha256": "bad"}, {"config_sha256": True},
                {"stage_root": "/tmp//" + self.stage_id},
                {"stage_root": "/tmp/" + "a" * 32},
                {"stage_root": "/tmp/../" + self.stage_id},
                {"expected_runtime": {"uid": 123}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.report([], context={**self.context, **changes})
        for name, value in (
                ("uid", True), ("no_new_privs", 1), ("hostname", ""),
                ("argv_prefix", []), ("argv_prefix", ["bwrap", 3]),
                ("host_namespaces", {}), ("python_sha256", "A" * 64)):
            context = copy.deepcopy(self.context)
            context["expected_runtime"][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.report([], context=context)

    def test_reconstructs_preparation_before_accepting_records(self):
        preparation = copy.deepcopy(self.preparation)
        preparation["public"]["launch_ready"] = True
        with self.assertRaises(ValueError):
            self.report([], preparation=preparation)
        with self.assertRaises(ValueError):
            self.report([], backend_raw=self.backend_raw + b"\n")
        with self.assertRaises(ValueError):
            self.report([], source_raw={**self.source_raw, "unverified.py": b""})

    def test_forbids_caller_approval_switch(self):
        with self.assertRaises(TypeError):
            self.report([], real=True)
        with self.assertRaises(TypeError):
            self.report([], trusted_collection_accepted=True)

    def test_inputs_and_results_do_not_alias(self):
        records = [self.record()]
        snapshot = copy.deepcopy((records, self.context, self.preparation))
        result = self.report(records)
        result["batches"][0]["ready"]["hostname"] = "changed"
        result["content_report"]["rows"][0]["result"]["status"] = "changed"
        self.assertEqual((records, self.context, self.preparation), snapshot)
        self.assertEqual(self.report(records)["content_report"]["completed"], 32)

    def test_imports_only_pure_helpers(self):
        tree = ast.parse((ROOT / "recalibration_execution_evidence.py").read_text())
        allowed = {
            "copy", "hashlib", "re", "pilot_protocol", "recalibration_data_contract",
            "recalibration_execution_package", "recalibration_execution_wire",
            "recalibration_validation", "validation_contract", "validation_real_report",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name in allowed for alias in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, ("open", "exec", "eval", "__import__"))


if __name__ == "__main__":
    unittest.main()
