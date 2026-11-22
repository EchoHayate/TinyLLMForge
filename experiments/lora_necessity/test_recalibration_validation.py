import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest

from pilot_protocol import FLAGS
from validation_contract import digest, encoded, parse_json
from recalibration_data_contract import FILE_PINS

if importlib.util.find_spec("recalibration_validation") is not None:
    import recalibration_validation as adapter
else:
    adapter = None


ROOT = Path(__file__).resolve().parent


def process(raw=b"", status="completed"):
    return {
        "status": status, "returncode": 0 if status == "completed" else -9,
        "stdout_hex": raw.hex(), "stderr_hex": "", "elapsed_ns": 1,
        "process_reaped": True, "stdin_complete": True,
    }


class RecalibrationValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = {
            name: (ROOT / "data" / (
                name if name == "supervision_v1.json"
                else "recalibration_v2_r1/" + name)).read_bytes()
            for name in FILE_PINS
        }
        cls.worker = (ROOT / "validation_task_worker.py").read_bytes()
        cls.run_ids = ["%032x" % position for position in range(15)]
        cls.runtime = {"host": "synthetic-only", "uid": 123}

    def setUp(self):
        self.assertIsNotNone(adapter, "recalibration validation not implemented")
        self.plan = adapter.build_schedule(self.raw)
        self.batches = adapter.build_batches(self.raw, self.worker, self.run_ids)

    def events(self, batch_index):
        batch = self.batches[batch_index]
        events = [{
            "kind": "ready", "run_id": batch["run_id"],
            "bundle_sha256": batch["bundle_sha256"],
            "worker_sha256": batch["worker_sha256"], "runtime": self.runtime,
        }]
        for local, row in enumerate(self.plan["rows"][batch["start"]:batch["stop"]]):
            outcome = row["witness_prediction"] or row["expected"]
            events.append({
                "kind": "row", "run_id": batch["run_id"], "ordinal": local,
                "request_sha256": row["request_sha256"],
                "result": process(encoded(outcome)),
            })
        events.append({
            "kind": "done", "run_id": batch["run_id"], "received": 32,
            "elapsed_ns": 1, "stopped": False,
        })
        return copy.deepcopy(events)

    def record(self, batch_index, events=None, status="completed"):
        selected = self.events(batch_index) if events is None else events
        return {
            "batch_index": batch_index,
            "bundle_raw": self.batches[batch_index]["bundle_raw"],
            "transport": process(b"".join(encoded(event) + b"\n"
                                          for event in selected), status),
        }

    def report(self, records):
        return adapter.build_report(self.raw, self.worker, self.run_ids, records,
                                    expected_runtime=self.runtime)

    def test_schedule_identity(self):
        self.assertEqual(self.plan, adapter.build_schedule(self.raw))
        self.assertEqual(self.plan["kind"], "LORA_RECALIBRATION_VALIDATION_PLAN")
        self.assertEqual(len(self.plan["rows"]), 480)
        expected = [(task, role, case)
                    for task in ("C%02d" % value for value in range(1, 13))
                    for role in ("reference_a", "reference_b", "initial",
                                 "variant_1", "variant_2")
                    for case in ("P01", "P02", "H01", "H02", "H03", "H04", "H05", "H06")]
        self.assertEqual([(row["task_id"], row["role"], row["case_id"])
                          for row in self.plan["rows"]], expected)
        for ordinal, row in enumerate(self.plan["rows"]):
            self.assertEqual(row["ordinal"], ordinal)
            self.assertEqual(row["row_sha256"], digest({
                name: value for name, value in row.items() if name != "row_sha256"}))
        self.assertEqual(sum(row["witness_prediction"] is not None
                             for row in self.plan["rows"]), 36)

    def test_minimal_packets(self):
        self.assertEqual(len(self.batches), 15)
        requests = []
        for index, batch in enumerate(self.batches):
            self.assertEqual((batch["batch_index"], batch["start"], batch["stop"]),
                             (index, index * 32, (index + 1) * 32))
            self.assertEqual(batch["plan_sha256"], digest(self.plan))
            self.assertEqual(batch["bundle_sha256"],
                             hashlib.sha256(batch["bundle_raw"]).hexdigest())
            self.assertLessEqual(len(batch["bundle_raw"]), 131072)
            bundle = parse_json(batch["bundle_raw"])
            self.assertEqual(bundle["mode"], "matrix")
            self.assertEqual(len(bundle["requests"]), 32)
            for request in bundle["requests"]:
                self.assertEqual(set(request), {"request_id", "source", "entry_point", "args"})
                self.assertLessEqual(len(encoded(request)), 8192)
            requests.extend(bundle["requests"])
        self.assertEqual([request["request_id"] for request in requests],
                         ["request-%04d" % ordinal for ordinal in range(480)])

    def test_reject_changed_pins_worker_and_run_ids(self):
        changed = dict(self.raw)
        changed["oracles.json"] += b" "
        with self.assertRaises(ValueError):
            adapter.build_schedule(changed)
        for worker in (self.worker + b"\n", b"x" * 16385, "not bytes"):
            with self.assertRaises(ValueError):
                adapter.build_batches(self.raw, worker, self.run_ids)
        for case, identifiers in (
                ("missing", self.run_ids[:-1]), ("duplicate", ["0" * 32] * 15),
                ("invalid_syntax", ["X" * 32] + self.run_ids[1:]),
                ("wrong_container", tuple(self.run_ids))):
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, "batch run IDs"):
                    adapter.build_batches(self.raw, self.worker, identifiers)

    def test_complete_synthetic_checks_without_admission(self):
        report = self.report([self.record(index) for index in range(15)])
        self.assertTrue(report["matrix_complete"])
        self.assertTrue(report["data_checks_satisfied"])
        self.assertEqual(report["reference_counts"],
                         {"planned": 192, "observed": 192, "matched": 192})
        self.assertEqual(report["witness_counts"],
                         {"planned": 36, "observed": 36, "matched": 36})
        self.assertEqual(len(report["rows"]), 480)
        self.assertIsNone(report["first_stop"])
        for name in (*FLAGS, "execution_provenance_verified", "run_authorized",
                     "full_dynamic_validation_complete"):
            self.assertIs(report[name], False)
        self.assertTrue(all(task["completed"] == 40 for task in report["tasks"]))

    def test_semantic_mismatch_continues_and_bool_is_not_int(self):
        expected = self.plan["rows"][0]["expected"]
        outcome = copy.deepcopy(expected)
        self.assertIs(outcome["value"][0], True)
        outcome["value"][0] = 1
        self.assertEqual(outcome, expected)
        self.assertNotEqual(encoded(outcome), encoded(expected))
        records = [self.record(index) for index in range(15)]
        events = self.events(0)
        events[1]["result"] = process(encoded(outcome))
        records[0] = self.record(0, events)
        report = self.report(records)
        row = report["rows"][0]
        self.assertEqual(row["dispatch_state"], "observed")
        self.assertEqual(row["status"], "completed")
        self.assertIs(row["expected_match"], False)
        self.assertTrue(report["matrix_complete"])
        self.assertFalse(report["data_checks_satisfied"])
        self.assertEqual(report["completed"], 480)
        self.assertIsNone(report["first_stop"])

    def test_witness_prediction_bool_is_not_int(self):
        target = next(row for row in self.plan["rows"]
                      if (row["task_id"], row["role"], row["case_id"])
                      == ("C01", "initial", "P01"))
        prediction = target["witness_prediction"]
        outcome = copy.deepcopy(prediction)
        self.assertIs(outcome["value"][0], True)
        outcome["value"][0] = 1
        self.assertEqual(outcome, prediction)
        self.assertNotEqual(encoded(outcome), encoded(prediction))
        records = [self.record(index) for index in range(15)]
        batch_index, local = divmod(target["ordinal"], 32)
        events = self.events(batch_index)
        events[local + 1]["result"] = process(encoded(outcome))
        records[batch_index] = self.record(batch_index, events)
        report = self.report(records)
        row = report["rows"][target["ordinal"]]
        self.assertEqual(row["dispatch_state"], "observed")
        self.assertEqual(row["status"], "completed")
        self.assertIs(row["prediction_match"], False)
        self.assertIs(row["expected_match"], False)
        self.assertTrue(report["matrix_complete"])
        self.assertFalse(report["data_checks_satisfied"])
        self.assertEqual(report["witness_counts"]["matched"], 35)
        self.assertEqual(report["completed"], 480)
        self.assertIsNone(report["first_stop"])

    def test_wrong_witness_prediction_is_data_failure_only(self):
        target = next(row for row in self.plan["rows"]
                      if row["witness_prediction"] is not None)
        records = [self.record(index) for index in range(15)]
        batch_index, local = divmod(target["ordinal"], 32)
        events = self.events(batch_index)
        events[local + 1]["result"] = process(encoded(target["expected"]))
        records[batch_index] = self.record(batch_index, events)
        report = self.report(records)
        self.assertTrue(report["matrix_complete"])
        self.assertFalse(report["data_checks_satisfied"])
        self.assertEqual(report["witness_counts"]["matched"], 35)

    def test_unknown_and_not_dispatched_differ(self):
        record = self.record(0)
        record["transport"] = None
        report = self.report([record])
        self.assertEqual(report["unknown_after_dispatch"], 32)
        self.assertEqual(report["not_dispatched"], 448)
        self.assertFalse(report["matrix_complete"])
        self.assertEqual(self.report([])["not_dispatched"], 480)

    def test_observed_failure_and_unknown_tail(self):
        events = self.events(0)
        events[3]["result"] = process(status="timeout")
        done = {**events[-1], "received": 3, "stopped": True}
        report = self.report([self.record(0, events[:4] + [done])])
        self.assertEqual(report["rows"][2]["status"], "timeout")
        self.assertEqual(report["rows"][2]["dispatch_state"], "observed")
        self.assertEqual(report["unknown_after_dispatch"], 29)
        self.assertEqual(report["not_dispatched"], 448)
        self.assertEqual(report["first_stop"]["ordinal"], 2)

    def test_short_batch_requires_stop(self):
        events = self.events(0)
        done = {**events[-1], "received": 2, "stopped": True}
        report = self.report([self.record(0, events[:3] + [done])])
        self.assertEqual(report["completed"], 2)
        self.assertEqual(report["unknown_after_dispatch"], 30)
        done["stopped"] = False
        report = self.report([self.record(0, events[:3] + [done])])
        self.assertEqual(report["batches"][0]["stop_reason"], "evidence_error")
        self.assertEqual(report["unknown_after_dispatch"], 32)

    def test_malformed_event_identities_quarantine_batch(self):
        mutations = [
            (0, "run_id", "f" * 32), (0, "worker_sha256", "0" * 64),
            (0, "bundle_sha256", "0" * 64), (0, "runtime", {"host": "drift"}),
            (1, "ordinal", True), (1, "ordinal", 32),
            (1, "request_sha256", "0" * 64), (-1, "received", True),
            (-1, "received", 31), (-1, "stopped", 0),
        ]
        for position, name, value in mutations:
            with self.subTest(position=position, name=name, value=value):
                events = self.events(0)
                events[position][name] = value
                report = self.report([self.record(0, events)])
                self.assertEqual(report["batches"][0]["stop_reason"], "evidence_error")
                self.assertEqual(report["unknown_after_dispatch"], 32)
                self.assertFalse(report["data_checks_satisfied"])

    def test_bad_process_or_outcome_quarantines_batch(self):
        for name, value in (("returncode", 1), ("stdin_complete", False),
                            ("process_reaped", False), ("stderr_hex", "61"),
                            ("stdout_hex", encoded({"kind": "return", "value": 1.0}).hex()),
                            ("stdout_hex", b'{"kind":"return","value":NaN}'.hex()),
                            ("stdout_hex", b'{"kind":"return","value":1,"value":2}'.hex())):
            with self.subTest(name=name, value=value):
                events = self.events(0)
                events[1]["result"][name] = value
                report = self.report([self.record(0, events)])
                self.assertEqual(report["batches"][0]["stop_reason"], "evidence_error")

    def test_event_order_and_extra_fields(self):
        original = self.events(0)
        for events in (original[1:], [original[0]] + original,
                       original + [original[-1]],
                       original[:2] + [original[1]] + original[2:],
                       [original[0], original[2], original[1]] + original[3:],
                       [{**original[0], "extra": 1}] + original[1:]):
            with self.subTest(events=len(events)):
                report = self.report([self.record(0, events)])
                self.assertEqual(report["batches"][0]["stop_reason"], "evidence_error")

    def test_missing_done_and_truncated_timeout(self):
        report = self.report([self.record(0, self.events(0)[:-1])])
        self.assertFalse(report["matrix_complete"])
        self.assertEqual(report["batches"][0]["stop_reason"], "missing_done")
        record = self.record(0, self.events(0)[:2], status="timeout")
        record["transport"]["stdout_hex"] += b'{"kind":'.hex()
        report = self.report([record])
        self.assertTrue(report["batches"][0]["truncated_final_line"])
        self.assertEqual(report["completed"], 1)
        self.assertEqual(report["unknown_after_dispatch"], 31)

    def test_batch_order_and_record_after_stop_rejected(self):
        for records in ([self.record(1)], [self.record(0), self.record(0)],
                        [self.record(0, []), self.record(1)],
                        [{**self.record(0), "batch_index": False}],
                        [{**self.record(0), "extra": 1}]):
            with self.subTest(records=len(records)):
                with self.assertRaises(ValueError):
                    self.report(records)

    def test_bundle_tampering_and_complete_prefix(self):
        record = self.record(0)
        bundle = parse_json(record["bundle_raw"])
        bundle["requests"][0]["args"] = []
        record["bundle_raw"] = encoded(bundle)
        report = self.report([record])
        self.assertEqual(report["unknown_after_dispatch"], 32)
        prefix = self.report([self.record(0)])
        self.assertEqual(prefix["completed"], 32)
        self.assertEqual(prefix["not_dispatched"], 448)
        self.assertEqual(prefix["first_stop"]["reason"], "records_exhausted")

    def test_packet_individual_and_total_limits(self):
        rows = copy.deepcopy(self.plan["rows"][:32])
        rows[0]["source"] = "x" * 4097
        with self.assertRaisesRegex(ValueError, "source byte limit"):
            adapter._packet(rows, self.worker, self.run_ids[0])
        rows[0]["source"] = "x" * 4096
        rows[0]["args"] = ["x" * 8192]
        with self.assertRaisesRegex(ValueError, "request byte limit"):
            adapter._packet(rows, self.worker, self.run_ids[0])
        for row in rows:
            row["source"] = "x" * 4096
            row["args"] = ["x" * 100]
        with self.assertRaisesRegex(ValueError, "bundle byte limit"):
            adapter._packet(rows, self.worker, self.run_ids[0])
        with self.assertRaisesRegex(ValueError, "worker byte limit"):
            adapter._packet(self.plan["rows"][:32], b"x" * 16385, self.run_ids[0])

    def test_json_values_and_outer_lifecycle(self):
        record = self.record(0)
        record["transport"]["returncode"] = True
        report = self.report([record])
        self.assertEqual(report["batches"][0]["stop_reason"], "evidence_error")
        for runtime in ({}, {"uid": float("inf")}, {"host": ("tuple",)}):
            with self.assertRaises(ValueError):
                adapter.build_report(self.raw, self.worker, self.run_ids, [],
                                     expected_runtime=runtime)
        for name in ("stdin_complete", "process_reaped"):
            record = self.record(0)
            record["transport"][name] = False
            report = self.report([record])
            self.assertEqual(report["batches"][0]["stop_reason"], "evidence_error")

    def test_observation_after_worker_stop_is_invalid(self):
        events = self.events(0)
        events[1]["result"] = process(status="program_error")
        report = self.report([self.record(0, events)])
        self.assertEqual(report["batches"][0]["stop_reason"], "evidence_error")

    def test_no_execution_imports_or_calls(self):
        import ast
        tree = ast.parse((ROOT / "recalibration_validation.py").read_text())
        forbidden = {"subprocess", "socket", "paramiko", "transformers", "torch",
                     "validation_remote", "validation_task_worker", "tokenizer_session"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name.split(".")[0] in forbidden
                                     for alias in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module.split(".")[0], forbidden)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"exec", "eval", "compile", "open", "__import__"})

    def test_full_identity_hashes_and_packet_binding(self):
        for row in self.plan["rows"]:
            self.assertEqual(row["file_sha256"], FILE_PINS)
            self.assertEqual(row["source_sha256"],
                             hashlib.sha256(row["source"].encode("utf-8")).hexdigest())
            self.assertEqual(row["input_sha256"], digest(row["args"]))
            self.assertEqual(row["expected_sha256"], digest(row["expected"]))
            prediction = row["witness_prediction"]
            self.assertEqual(row["witness_prediction_sha256"],
                             None if prediction is None else digest(prediction))
            bundle = parse_json(self.batches[row["ordinal"] // 32]["bundle_raw"])
            self.assertEqual(bundle["kind"], "LORA_REMOTE_VALIDATION_BUNDLE")
            self.assertEqual(bundle["run_id"], self.run_ids[row["ordinal"] // 32])
            self.assertEqual(bundle["worker_source"].encode("utf-8"), self.worker)
            request = bundle["requests"][row["ordinal"] % 32]
            self.assertEqual(row["request_sha256"], digest(request))
            self.assertEqual(request["entry_point"], row["entry_point"])
            self.assertEqual(request["source"], row["source"])
            self.assertEqual(encoded(request["args"]), encoded(row["args"]))

    def test_quarantine_later_batch_preserves_earlier_evidence(self):
        events = self.events(1)
        events[-1]["run_id"] = self.run_ids[0]
        records = [self.record(0), self.record(1, events)]
        report = self.report(records)
        self.assertEqual(report["completed"], 32)
        self.assertEqual(report["observed"], 32)
        self.assertEqual(report["unknown_after_dispatch"], 32)
        self.assertEqual(report["not_dispatched"], 416)
        self.assertTrue(report["batches"][0]["batch_complete"])
        self.assertEqual(report["batches"][1]["received"], 0)
        self.assertEqual(report["first_stop"]["batch_index"], 1)
        self.assertEqual(report["first_stop"]["reason"], "evidence_error")
        with self.assertRaisesRegex(ValueError, "record after incomplete batch"):
            self.report(records + [self.record(2)])

    def test_report_digests_and_input_detachment(self):
        records = [self.record(0)]
        original = copy.deepcopy(records)
        report = self.report(records)
        batch = report["batches"][0]
        self.assertEqual(records, original)
        self.assertEqual(batch["actual_bundle_sha256"],
                         hashlib.sha256(records[0]["bundle_raw"]).hexdigest())
        self.assertEqual(batch["transport_sha256"], digest(records[0]["transport"]))
        self.assertEqual(batch["stdout_sha256"], hashlib.sha256(
            bytes.fromhex(records[0]["transport"]["stdout_hex"])).hexdigest())
        self.assertEqual(batch["runtime_sha256"], digest(self.runtime))
        self.assertEqual(report["expected_runtime_sha256"], digest(self.runtime))
        self.assertEqual(report["evidence_sha256"], digest([{
            "batch_index": 0, "bundle_sha256": batch["actual_bundle_sha256"],
            "transport_sha256": batch["transport_sha256"],
        }]))
        batch["runtime"]["uid"] = -1
        batch["transport"]["stdout_hex"] = ""
        report["expected_runtime"]["uid"] = -1
        report["run_ids"].clear()
        self.assertEqual(records, original)
        self.assertEqual(self.runtime["uid"], 123)
        self.assertEqual(len(self.run_ids), 15)
        self.plan["rows"][0]["args"].append("changed")
        self.assertNotEqual(self.plan, adapter.build_schedule(self.raw))

    def test_local_record_shape_and_non_json_values_raise(self):
        for records in (None, (), {}, [None], [self.record(0)] * 16):
            with self.subTest(records=type(records).__name__):
                with self.assertRaises(ValueError):
                    self.report(records)
        for value in ("bytes", bytearray(b"{}"), b"x" * 131073):
            with self.assertRaises(ValueError):
                self.report([{**self.record(0), "bundle_raw": value}])
        for transport in ([], {"bad": object()}, {"bad": (1,)},
                          {"bad": float("nan")}, {1: "bad"}):
            with self.assertRaises(ValueError):
                self.report([{**self.record(0), "transport": transport}])

    def test_partial_transport_keeps_observed_prefix_and_global_stop(self):
        events = self.events(1)
        events[2]["result"] = process(status="output_limit")
        report = self.report([
            self.record(0), self.record(1, events[:3], status="timeout")])
        self.assertEqual(report["completed"], 33)
        self.assertEqual(report["observed"], 34)
        self.assertEqual(report["unknown_after_dispatch"], 30)
        self.assertEqual(report["not_dispatched"], 416)
        self.assertEqual(report["first_stop"],
                         {"batch_index": 1, "ordinal": 33, "reason": "worker_output_limit"})
        report = self.report([self.record(0, [], status="launch_failed")])
        self.assertEqual(report["first_stop"]["reason"], "transport_launch_failed")
        self.assertEqual(report["unknown_after_dispatch"], 32)

    def test_runtime_binding_is_strictly_typed(self):
        for value in (True, 123.0):
            events = self.events(0)
            events[0]["runtime"]["uid"] = value
            report = self.report([self.record(0, events)])
            self.assertEqual(report["batches"][0]["stop_reason"], "evidence_error")
            self.assertEqual(report["unknown_after_dispatch"], 32)

    def test_record_keys_must_be_exact_builtin_strings(self):
        class StringSubclass(str):
            pass

        record = self.record(0)
        for name in record:
            with self.subTest(name=name):
                changed = {
                    StringSubclass(key) if key == name else key: value
                    for key, value in record.items()
                }
                with self.assertRaises(ValueError):
                    self.report([changed])


if __name__ == "__main__":
    unittest.main()
