import copy
import hashlib
import importlib
import json
from pathlib import Path
import unittest

from validation_contract import build_schedule, digest, encoded, worker_requests


def synthetic_result(raw):
    return {
        "status": "completed", "returncode": 0,
        "stdout_hex": raw.hex(), "stderr_hex": "", "elapsed_ns": 1000,
        "process_reaped": True, "stdin_complete": True,
    }


def synthetic_events(plan, bundle_raw, bundle):
    events = [{
        "kind": "ready", "run_id": bundle["run_id"],
        "bundle_sha256": hashlib.sha256(bundle_raw).hexdigest(),
        "worker_sha256": hashlib.sha256(
            bundle["worker_source"].encode("utf-8")).hexdigest(),
        "runtime": {"host": "synthetic-fixture", "uid": 1054,
                    "no_new_privs": True, "argv_prefix": ["/synthetic/runtime"]},
    }]
    for planned in plan["rows"][:32]:
        outcome = copy.deepcopy(planned["expected"])
        if planned["role"] == "initial" and planned["group"] == "scoring":
            outcome = {"kind": "return", "value": "__synthetic_wrong__"}
        if planned["witness_prediction"] is not None:
            outcome = copy.deepcopy(planned["witness_prediction"])
        events.append({
            "kind": "row", "run_id": bundle["run_id"],
            "ordinal": planned["ordinal"],
            "request_sha256": planned["request_sha256"],
            "result": synthetic_result(encoded(outcome)),
        })
    events.append({
        "kind": "done", "run_id": bundle["run_id"],
        "received": 32, "elapsed_ns": 32000, "stopped": False,
    })
    return events


def jsonl(events):
    return b"".join(encoded(event) + b"\n" for event in events)


class ValidationRealReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = Path(__file__).with_name("data")
        cls.source = (data / "task_sources_v1.json").read_bytes()
        cls.manifest = (data / "manifest_v1.json").read_bytes()
        cls.plan = build_schedule(cls.source, cls.manifest)
        cls.requests = worker_requests(cls.source, cls.manifest)[:32]

    def setUp(self):
        self.assertTrue(Path(__file__).with_name("validation_real_report.py").is_file(),
                        "real validation report not implemented")
        self.module = importlib.import_module("validation_real_report")
        self.bundle = {
            "kind": "LORA_REMOTE_VALIDATION_BUNDLE", "mode": "d01",
            "run_id": "0123456789abcdef" * 2,
            "worker_source": "raise RuntimeError('synthetic fixture only')\n",
            "requests": copy.deepcopy(self.requests),
        }
        self.bundle_raw = encoded(self.bundle)
        self.events = synthetic_events(self.plan, self.bundle_raw, self.bundle)

    def report(self, events=None, transport=None, bundle_raw=None):
        if transport is None:
            transport = synthetic_result(jsonl(self.events if events is None else events))
        return self.module.build_d01_report(
            self.source, self.manifest,
            self.bundle_raw if bundle_raw is None else bundle_raw, transport)

    def test_synthetic_full_batch_has_local_checks_only(self):
        report = self.report()
        self.assertEqual(report["kind"], "LORA_D01_EXECUTION_REPORT")
        self.assertEqual(report["full_scheduled"], 1416)
        self.assertEqual(report["batch_scheduled"], 32)
        self.assertEqual(report["received"], 32)
        self.assertEqual(report["remaining_full"], 1384)
        self.assertEqual(len(report["rows"]), 32)
        for name in ("reference_ok", "initial_scoring_rejected", "witnesses_ok",
                     "batch_complete", "batch_checks_satisfied"):
            self.assertIs(report[name], True, name)
        for name in ("full_dynamic_validation_complete", "benchmark_eligible",
                     "formal_evaluation_ready", "net_benefit_established", "isolated",
                     "generated_candidate_backend_ready"):
            self.assertIs(report[name], False, name)
        for planned, row in zip(self.plan["rows"][:32], report["rows"]):
            for name in ("ordinal", "task_id", "role", "group", "case_id",
                         "task_sha256", "source_sha256", "input_sha256",
                         "expected_sha256", "request_sha256"):
                self.assertEqual(row[name], planned[name])
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["outcome"]["kind"], "return")
            if planned["witness_prediction"] is not None:
                self.assertIs(row["prediction_match"], True)
                self.assertIs(row["expected_match"], False)
            else:
                self.assertIsNone(row["prediction_match"])

    def test_report_binds_raw_inputs_and_preserves_runtime(self):
        transport = synthetic_result(jsonl(self.events))
        report = self.report(transport=transport)
        self.assertEqual(report["source_sha256"], hashlib.sha256(self.source).hexdigest())
        self.assertEqual(report["manifest_sha256"],
                         hashlib.sha256(self.manifest).hexdigest())
        self.assertEqual(report["plan_sha256"], digest(self.plan))
        self.assertEqual(report["bundle_sha256"], hashlib.sha256(self.bundle_raw).hexdigest())
        self.assertEqual(report["worker_sha256"], self.events[0]["worker_sha256"])
        self.assertEqual(report["runtime"], self.events[0]["runtime"])
        self.assertEqual(report["runtime_sha256"], digest(self.events[0]["runtime"]))
        self.assertEqual(report["transport_sha256"], digest(transport))
        self.assertEqual(report["transport"], transport)
        self.assertEqual(report["stdout_sha256"], hashlib.sha256(jsonl(self.events)).hexdigest())
        self.assertEqual(report["done"], self.events[-1])
        report["runtime"]["argv_prefix"].append("changed")
        self.assertEqual(self.events[0]["runtime"]["argv_prefix"], ["/synthetic/runtime"])
        self.assertEqual(transport["stdout_hex"], jsonl(self.events).hex())

    def test_only_one_observation_preserves_missing_31(self):
        report = self.report(self.events[:2])
        self.assertEqual(report["received"], 1)
        self.assertEqual(report["remaining_full"], 1415)
        self.assertEqual(len(report["rows"]), 32)
        self.assertIs(report["batch_complete"], False)
        self.assertIs(report["batch_checks_satisfied"], False)
        for row in report["rows"][1:]:
            self.assertEqual(row["status"], "missing")
            self.assertIsNone(row["outcome"])
            self.assertIsNone(row["expected_match"])
            self.assertIsNone(row["prediction_match"])

    def test_empty_and_ready_only_observations_are_incomplete(self):
        for events in ([], self.events[:1]):
            with self.subTest(events=len(events)):
                report = self.report(events)
                self.assertEqual(report["received"], 0)
                self.assertEqual(report["remaining_full"], 1416)
                self.assertIs(report["batch_complete"], False)
                self.assertIsNone(report["done"])
        self.assertIsNone(self.report([])["runtime"])
        self.assertIsNone(self.report([])["runtime_sha256"])

    def test_missing_done_and_stopped_done_are_incomplete(self):
        self.assertIs(self.report(self.events[:-1])["batch_complete"], False)
        self.events[-1]["stopped"] = True
        report = self.report()
        self.assertIs(report["batch_complete"], False)
        self.assertIs(report["batch_checks_satisfied"], False)

    def test_partial_done_preserves_missing_rows(self):
        done = {**self.events[-1], "received": 31, "stopped": True}
        report = self.report(self.events[:32] + [done])
        self.assertEqual(report["received"], 31)
        self.assertEqual(report["rows"][-1]["status"], "missing")
        self.assertIs(report["batch_complete"], False)

    def test_reference_failure_is_a_semantic_result_not_a_stop(self):
        self.events[1]["result"]["stdout_hex"] = encoded(
            {"kind": "return", "value": "wrong"}).hex()
        report = self.report()
        self.assertIs(report["reference_ok"], False)
        self.assertIs(report["batch_complete"], True)
        self.assertIs(report["batch_checks_satisfied"], False)

    def test_each_frozen_witness_must_match_prediction_and_disagree_with_expected(self):
        witnesses = [row for row in self.plan["rows"][:32]
                     if row["witness_prediction"] is not None]
        self.assertEqual(len(witnesses), 3)
        for planned in witnesses:
            events = copy.deepcopy(self.events)
            events[planned["ordinal"] + 1]["result"]["stdout_hex"] = encoded(
                planned["expected"]).hex()
            with self.subTest(ordinal=planned["ordinal"]):
                report = self.report(events)
                self.assertIs(report["witnesses_ok"], False)
                self.assertIs(report["batch_complete"], True)
                self.assertIs(report["batch_checks_satisfied"], False)

    def test_initial_requires_at_least_one_scoring_rejection(self):
        for planned in self.plan["rows"][:32]:
            if planned["role"] == "initial" and planned["group"] == "scoring":
                self.events[planned["ordinal"] + 1]["result"]["stdout_hex"] = encoded(
                    planned["expected"]).hex()
        report = self.report()
        self.assertIs(report["initial_scoring_rejected"], False)
        self.assertIs(report["batch_checks_satisfied"], False)

    def test_bool_and_int_outcomes_are_not_equal(self):
        self.events[1]["result"]["stdout_hex"] = encoded(
            {"kind": "return", "value": False}).hex()
        report = self.report()
        self.assertIs(report["rows"][0]["expected_match"], False)
        self.assertIs(report["reference_ok"], False)

    def test_allowed_exception_outcome_is_preserved(self):
        self.events[1]["result"]["stdout_hex"] = encoded(
            {"kind": "raises", "type": "ValueError"}).hex()
        report = self.report()
        self.assertEqual(report["rows"][0]["outcome"],
                         {"kind": "raises", "type": "ValueError"})
        self.assertIs(report["rows"][0]["expected_match"], False)

    def test_frozen_inputs_are_revalidated(self):
        transport = synthetic_result(jsonl(self.events))
        for source, manifest in ((self.source + b"\n", self.manifest),
                                 (self.source, self.manifest + b"\n")):
            with self.subTest(source_length=len(source)), self.assertRaises(ValueError):
                self.module.build_d01_report(source, manifest, self.bundle_raw, transport)

    def test_bundle_envelope_and_request_projection_are_strict(self):
        changes = [
            ("kind", "synthetic"), ("mode", "probe"), ("run_id", "A" * 32),
            ("run_id", "a" * 31), ("run_id", 1), ("worker_source", ""),
            ("worker_source", None), ("requests", self.requests[:31]),
            ("requests", self.requests + self.requests[:1]), ("requests", {}),
            ("extra", True),
        ]
        for name, value in changes:
            bundle = copy.deepcopy(self.bundle)
            bundle[name] = value
            with self.subTest(name=name, value=str(value)[:30]), self.assertRaises(ValueError):
                self.report(bundle_raw=encoded(bundle))
        for name, value in (("expected", {}), ("role", "reference"),
                            ("witness_prediction", {}), ("request_id", "request-0001"),
                            ("args", []), ("source", "pass"), ("entry_point", "other")):
            bundle = copy.deepcopy(self.bundle)
            bundle["requests"][0][name] = value
            raw = encoded(bundle)
            events = copy.deepcopy(self.events)
            events[0]["bundle_sha256"] = hashlib.sha256(raw).hexdigest()
            with self.subTest(request_field=name), self.assertRaises(ValueError):
                self.report(events, bundle_raw=raw)

    def test_bundle_uses_raw_byte_digest_not_reformatted_digest(self):
        raw = json.dumps(self.bundle, indent=2).encode("utf-8")
        with self.assertRaises(ValueError):
            self.report(bundle_raw=raw)
        self.events[0]["bundle_sha256"] = hashlib.sha256(raw).hexdigest()
        self.assertIs(self.report(bundle_raw=raw)["batch_complete"], True)

    def test_strict_json_for_bundle_and_complete_event_lines(self):
        malformed = [
            b'{"kind":"ready","kind":"ready"}', b"NaN", b"Infinity", b"1e999",
            b"{}", b"[]", b"null", b"\xff", b'{"kind":', b"", b"{} {}",
        ]
        for raw in malformed:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.report(transport=synthetic_result(raw + b"\n"))
            with self.subTest(bundle_raw=raw), self.assertRaises(ValueError):
                self.report(bundle_raw=raw)

    def test_event_fields_bindings_and_types_are_strict(self):
        changes = [
            (0, "run_id", "f" * 32), (0, "bundle_sha256", "0" * 64),
            (0, "worker_sha256", "0" * 64), (0, "runtime", {}),
            (0, "runtime", []), (0, "extra", True),
            (1, "run_id", "f" * 32), (1, "ordinal", True),
            (1, "ordinal", 0.0), (1, "ordinal", -1), (1, "ordinal", 32),
            (1, "request_sha256", "0" * 64), (1, "extra", True),
            (-1, "run_id", "f" * 32), (-1, "received", True),
            (-1, "received", 32.0), (-1, "received", 31),
            (-1, "elapsed_ns", True), (-1, "elapsed_ns", -1),
            (-1, "stopped", 0), (-1, "extra", True),
        ]
        for position, name, value in changes:
            events = copy.deepcopy(self.events)
            events[position][name] = value
            with self.subTest(position=position, name=name, value=value):
                with self.assertRaises(ValueError):
                    self.report(events)

    def test_runtime_is_nonempty_metadata_not_an_invented_admission_gate(self):
        runtime = {"recorded": ["synthetic", {"arbitrary_metadata": 1.5}]}
        self.events[0]["runtime"] = runtime
        report = self.report()
        self.assertEqual(report["runtime"], runtime)
        self.assertEqual(report["runtime_sha256"], digest(runtime))
        self.assertIs(report["batch_complete"], True)

    def test_event_order_duplicates_and_unknown_kinds_are_rejected(self):
        invalid = [
            self.events[1:],
            [self.events[0], self.events[0]],
            self.events[:2] + self.events[1:],
            self.events[:1] + [self.events[2], self.events[1]],
            self.events[:2] + [self.events[3]],
            self.events + [self.events[-1]],
            self.events + [self.events[1]],
            self.events[:1] + [{"kind": "unknown", "run_id": self.bundle["run_id"]}],
        ]
        for events in invalid:
            with self.subTest(kinds=[event["kind"] for event in events][:4]):
                with self.assertRaises(ValueError):
                    self.report(events)

    def test_result_schema_and_exact_types_apply_to_transport_and_rows(self):
        invalid = [
            ("status", "protocol_error"), ("status", []),
            ("returncode", True), ("returncode", 0.0),
            ("elapsed_ns", True), ("elapsed_ns", -1), ("elapsed_ns", 1.0),
            ("stdin_complete", 1), ("process_reaped", 1),
            ("stdout_hex", "00 zz"), ("stdout_hex", "0"),
            ("stdout_hex", 0), ("stderr_hex", "\xff"), ("extra", True),
        ]
        for name, value in invalid:
            events = copy.deepcopy(self.events)
            events[1]["result"][name] = value
            transport = synthetic_result(jsonl(self.events))
            transport[name] = value
            with self.subTest(name=name, target="row"), self.assertRaises(ValueError):
                self.report(events)
            with self.subTest(name=name, target="transport"), self.assertRaises(ValueError):
                self.report(transport=transport)

    def test_completed_requires_exit_zero_complete_stdin_and_reap(self):
        for name, value in (("returncode", 1), ("returncode", None),
                            ("stdin_complete", False), ("process_reaped", False)):
            events = copy.deepcopy(self.events)
            events[1]["result"][name] = value
            transport = synthetic_result(jsonl(self.events))
            transport[name] = value
            with self.subTest(name=name, target="row"), self.assertRaises(ValueError):
                self.report(events)
            with self.subTest(name=name, target="transport"), self.assertRaises(ValueError):
                self.report(transport=transport)

    def test_only_outer_completed_transport_permits_preserved_stderr_diagnostics(self):
        diagnostics = b"debug1: Authentication succeeded\n\xff"
        transport = synthetic_result(jsonl(self.events))
        transport["stderr_hex"] = diagnostics.hex()
        report = self.report(transport=transport)
        self.assertIs(report["batch_complete"], True)
        self.assertEqual(report["transport"]["stderr_hex"], diagnostics.hex())
        self.assertEqual(report["transport_sha256"], digest(transport))
        self.events[1]["result"]["stderr_hex"] = diagnostics.hex()
        with self.assertRaises(ValueError):
            self.report()

    def test_completed_payload_must_be_exact_strict_outcome(self):
        for raw in (b"", b"\xff", b'{"kind":"return","value":NaN}',
                    b'{"kind":"return","value":1.0}',
                    b'{"kind":"return","value":0,"passed":true}',
                    b'{"kind":"return","value":0,"value":1}',
                    b'{"kind":"raises","type":"RuntimeError"}',
                    b'noise\n{"kind":"return","value":0}',
                    b'{"kind":"return","value":0}\n{}', b" " * 65537):
            events = copy.deepcopy(self.events)
            events[1]["result"]["stdout_hex"] = raw.hex()
            with self.subTest(raw=raw[:70]), self.assertRaises(ValueError):
                self.report(events)

    def test_failed_transport_never_promotes_complete_remote_events(self):
        for status in ("program_error", "timeout", "output_limit", "launch_failed"):
            transport = synthetic_result(jsonl(self.events))
            transport.update(status=status, returncode=None, process_reaped=False,
                             stdin_complete=False, stderr_hex=b"\xff".hex())
            with self.subTest(status=status):
                report = self.report(transport=transport)
                self.assertEqual(report["received"], 32)
                self.assertIs(report["batch_complete"], False)
                self.assertIs(report["batch_checks_satisfied"], False)

    def test_failed_rows_do_not_compare_success_shaped_payloads_or_allow_later_rows(self):
        for status in ("program_error", "timeout", "output_limit", "launch_failed"):
            events = copy.deepcopy(self.events)
            events[1]["result"].update(status=status, returncode=-9)
            with self.subTest(status=status):
                with self.assertRaises(ValueError):
                    self.report(events)
                done = {**events[-1], "received": 1, "stopped": True}
                report = self.report(events[:2] + [done])
                self.assertEqual(report["rows"][0]["status"], status)
                self.assertIsNone(report["rows"][0]["outcome"])
                self.assertIsNone(report["rows"][0]["expected_match"])
                self.assertIs(report["batch_complete"], False)
                self.assertEqual(report["stop_ordinal"], 0)
                done["stopped"] = False
                with self.assertRaises(ValueError):
                    self.report(events[:2] + [done])

    def test_failed_row_keeps_raw_binary_and_allows_missing_done(self):
        self.events[1]["result"].update(
            status="program_error", returncode=1, stdout_hex="ff",
            stderr_hex="fe", stdin_complete=False, process_reaped=False)
        report = self.report(self.events[:2])
        self.assertEqual(report["rows"][0]["result"], self.events[1]["result"])
        self.assertIsNone(report["rows"][0]["outcome"])
        self.assertIs(report["batch_complete"], False)

    def test_outer_timeout_preserves_complete_lines_before_truncated_tail(self):
        for tail in (b'{"kind":"row","run_id":"', b"\xff", b'{"kind":'):
            transport = synthetic_result(jsonl(self.events[:3]) + tail)
            transport.update(status="timeout", returncode=-9)
            with self.subTest(tail=tail):
                report = self.report(transport=transport)
                self.assertEqual(report["received"], 2)
                self.assertIs(report["truncated_final_line"], True)
                self.assertIs(report["batch_complete"], False)
                self.assertEqual(report["rows"][2]["status"], "missing")

    def test_outer_timeout_does_not_hide_malformed_complete_lines_or_valid_bad_events(self):
        for tail in (b'{"kind":\n', b"\xff\n", b"\n",
                     b'{"kind":"ready","kind":"ready"}',
                     b'{"kind":"unknown"}', b'{"kind":"done","received":NaN}'):
            transport = synthetic_result(jsonl(self.events[:2]) + tail)
            transport.update(status="timeout", returncode=-9)
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                self.report(transport=transport)

    def test_normal_or_non_timeout_transport_rejects_incomplete_tail(self):
        for status in ("completed", "program_error", "output_limit", "launch_failed"):
            transport = synthetic_result(jsonl(self.events[:2]) + b'{"kind":')
            transport["status"] = status
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.report(transport=transport)

    def test_complete_final_record_without_newline_is_valid(self):
        transport = synthetic_result(jsonl(self.events).rstrip(b"\n"))
        report = self.report(transport=transport)
        self.assertIs(report["batch_complete"], True)
        self.assertIs(report["truncated_final_line"], False)
        transport["status"] = "timeout"
        report = self.report(transport=transport)
        self.assertEqual(report["received"], 32)
        self.assertIs(report["batch_complete"], False)
        self.assertIs(report["truncated_final_line"], False)

    def test_output_limit_rows_preserve_one_overflow_byte_without_comparisons(self):
        for name, limit in (("stdout_hex", 65536), ("stderr_hex", 4096)):
            events = copy.deepcopy(self.events[:2])
            events[1]["result"].update(status="output_limit", returncode=-9)
            events[1]["result"][name] = (b"x" * (limit + 1)).hex()
            with self.subTest(name=name):
                report = self.report(events)
                self.assertEqual(report["rows"][0]["result"], events[1]["result"])
                self.assertIsNone(report["rows"][0]["expected_match"])
                self.assertIsNone(report["rows"][0]["prediction_match"])
                events[1]["result"][name] += "00"
                with self.assertRaises(ValueError):
                    self.report(events)

    def test_outer_output_limit_preserves_one_overflow_byte(self):
        raw = jsonl(self.events)
        transport = synthetic_result(b" " * (8388609 - len(raw)) + raw)
        transport.update(status="output_limit", returncode=-9,
                         stderr_hex=(b"x" * 131073).hex())
        report = self.report(transport=transport)
        self.assertEqual(report["received"], 32)
        self.assertIs(report["batch_complete"], False)
        for name in ("stdout_hex", "stderr_hex"):
            oversized = copy.deepcopy(transport)
            oversized[name] += "00"
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.report(transport=oversized)

    def test_completed_stdout_cannot_exceed_worker_or_transport_limit(self):
        payload = encoded(self.plan["rows"][0]["expected"])
        self.events[1]["result"]["stdout_hex"] = (
            b" " * (65537 - len(payload)) + payload).hex()
        with self.assertRaises(ValueError):
            self.report()
        raw = jsonl(self.events[:1])
        transport = synthetic_result(b" " * (8388609 - len(raw)) + raw)
        with self.assertRaises(ValueError):
            self.report(transport=transport)
