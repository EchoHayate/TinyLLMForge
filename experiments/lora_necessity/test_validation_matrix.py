import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
import unittest

from pilot_protocol import FLAGS
from validation_contract import build_schedule, digest, encoded, worker_requests


ROOT = Path(__file__).parent


def synthetic_result(raw):
    return {
        "status": "completed", "returncode": 0,
        "stdout_hex": raw.hex(), "stderr_hex": "", "elapsed_ns": 1000,
        "process_reaped": True, "stdin_complete": True,
    }


def jsonl(events):
    return b"".join(encoded(event) + b"\n" for event in events)


class ValidationMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / "data/task_sources_v1.json").read_bytes()
        cls.manifest = (ROOT / "data/manifest_v1.json").read_bytes()
        cls.plan = build_schedule(cls.source, cls.manifest)
        cls.requests = worker_requests(cls.source, cls.manifest)
        cls.runtime = {"host": "synthetic-only", "backend_sha256": "a" * 64,
                       "no_new_privs": True, "argv": ["/synthetic/new"]}
        cls.d01_runtime = {**cls.runtime, "backend_sha256": "b" * 64}
        cls.records = []
        for start in range(0, 1416, 32):
            stop = min(start + 32, 1416)
            bundle = {
                "kind": "LORA_REMOTE_VALIDATION_BUNDLE",
                "mode": "d01" if start == 0 else "matrix",
                "run_id": f"{start + 1:032x}",
                "worker_source": "raise RuntimeError('synthetic only')\n",
                "requests": cls.requests[start:stop],
            }
            bundle_raw = encoded(bundle)
            events = [{
                "kind": "ready", "run_id": bundle["run_id"],
                "bundle_sha256": hashlib.sha256(bundle_raw).hexdigest(),
                "worker_sha256": hashlib.sha256(
                    bundle["worker_source"].encode()).hexdigest(),
                "runtime": cls.d01_runtime if start == 0 else cls.runtime,
            }]
            for ordinal, planned in enumerate(cls.plan["rows"][start:stop]):
                outcome = planned["expected"]
                if planned["role"] == "initial" and planned["group"] == "scoring":
                    outcome = {"kind": "return", "value": "__synthetic_wrong__"}
                if planned["witness_prediction"] is not None:
                    outcome = planned["witness_prediction"]
                events.append({
                    "kind": "row", "run_id": bundle["run_id"], "ordinal": ordinal,
                    "request_sha256": planned["request_sha256"],
                    "result": synthetic_result(encoded(outcome)),
                })
            events.append({
                "kind": "done", "run_id": bundle["run_id"],
                "received": stop - start, "elapsed_ns": 1000, "stopped": False,
            })
            record = {"start": start, "bundle_raw": bundle_raw,
                      "transport": synthetic_result(jsonl(events))}
            if start == 0:
                cls.d01 = record
            else:
                cls.records.append(record)

    def setUp(self):
        self.assertTrue((ROOT / "validation_matrix.py").is_file())
        self.module = importlib.import_module("validation_matrix")

    def events(self, record):
        return [json.loads(line) for line in
                bytes.fromhex(record["transport"]["stdout_hex"]).splitlines()]

    def with_events(self, record, events):
        return {**record, "transport": synthetic_result(jsonl(events))}

    def batch(self, record=None, **kwargs):
        record = self.records[0] if record is None else record
        return self.module.build_batch_report(
            self.source, self.manifest, record["bundle_raw"], record["transport"],
            start=record["start"], expected_runtime=kwargs.get("runtime", self.runtime))

    def aggregate(self, records=None, d01=None, **kwargs):
        d01 = self.d01 if d01 is None else d01
        return self.module.aggregate_matrix(
            self.source, self.manifest, d01["bundle_raw"], d01["transport"],
            self.records if records is None else records,
            expected_runtime=kwargs.get("runtime", self.runtime),
            d01_expected_runtime=kwargs.get("d01_runtime", self.d01_runtime))

    def assert_admission_false(self, report):
        for name in (*FLAGS, "generated_candidate_backend_ready"):
            self.assertIs(report[name], False, name)

    def test_reporter_exists(self):
        self.assertTrue((ROOT / "validation_matrix.py").is_file())

    def test_reporter_imports_no_execution_modules(self):
        tree = ast.parse((ROOT / "validation_matrix.py").read_text())
        allowed = {"copy", "hashlib", "re", "pilot_protocol", "validation_contract",
                   "validation_report", "validation_real_report"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name in allowed for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)

    def test_frozen_44_ranges_and_fresh_result(self):
        expected = [(start, min(start + 32, 1416)) for start in range(32, 1416, 32)]
        self.assertEqual(self.module.batch_ranges(), expected)
        self.assertEqual(len(expected), 44)
        self.assertEqual(expected[-1], (1408, 1416))
        self.module.batch_ranges().clear()
        self.assertEqual(self.module.batch_ranges(), expected)

    def test_batch_32_and_final_8_keep_global_ordinals_and_digests(self):
        for record in (self.records[0], self.records[-1]):
            with self.subTest(start=record["start"]):
                report = self.batch(record)
                count = min(32, 1416 - record["start"])
                self.assertEqual(report["kind"], "LORA_MATRIX_BATCH_REPORT")
                self.assertEqual(report["start"], record["start"])
                self.assertEqual(report["stop"], record["start"] + count)
                self.assertEqual(report["received"], count)
                self.assertEqual([row["ordinal"] for row in report["rows"]],
                                 list(range(report["start"], report["stop"])))
                self.assertIs(report["batch_complete"], True)
                self.assertEqual(report["bundle_sha256"],
                                 hashlib.sha256(record["bundle_raw"]).hexdigest())
                self.assertEqual(report["transport_sha256"], digest(record["transport"]))
                self.assertEqual(report["runtime"], self.runtime)
                self.assert_admission_false(report)
                self.assertIs(report["full_dynamic_validation_complete"], False)
                for row in report["rows"]:
                    planned = self.plan["rows"][row["ordinal"]]
                    self.assertEqual(row["request_sha256"], planned["request_sha256"])
                    self.assertEqual(row["expected_match"],
                                     encoded(row["outcome"]) == encoded(planned["expected"]))
                    if planned["witness_prediction"] is not None:
                        self.assertIs(row["prediction_match"], True)

    def test_missing_streams_keep_all_batch_rows(self):
        for record in (self.records[0], self.records[-1]):
            events = self.events(record)
            for received in (0, 1, 3):
                with self.subTest(start=record["start"], received=received):
                    report = self.batch(self.with_events(record, events[:received + 1]))
                    self.assertEqual(report["received"], received)
                    self.assertEqual(len(report["rows"]), len(events) - 2)
                    self.assertIs(report["batch_complete"], False)
                    for row in report["rows"][received:]:
                        self.assertEqual(row["status"], "missing")
                        self.assertIsNone(row["result"])
                        self.assertIsNone(row["expected_match"])
        report = self.batch(self.with_events(self.records[0], []))
        self.assertIsNone(report["runtime"])

    def test_scheduled_start_and_exact_answer_free_projection(self):
        for start in (0, 1, 33, -32, 1416, True, 32.0):
            with self.subTest(start=start), self.assertRaises(ValueError):
                self.batch({**self.records[0], "start": start})
        original = json.loads(self.records[0]["bundle_raw"])
        for name, value in (("mode", "d01"), ("kind", "wrong"),
                            ("run_id", "A" * 32), ("worker_source", ""),
                            ("extra", True), ("requests", original["requests"][:-1])):
            bundle = {**original, name: value}
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.batch({**self.records[0], "bundle_raw": encoded(bundle)})
        for name, value in (("expected", {}), ("witness_prediction", {}),
                            ("source", "pass"), ("args", []),
                            ("entry_point", "other"), ("request_id", "request-0000")):
            bundle = copy.deepcopy(original)
            bundle["requests"][0][name] = value
            raw = encoded(bundle)
            events = self.events(self.records[0])
            events[0]["bundle_sha256"] = hashlib.sha256(raw).hexdigest()
            with self.subTest(request_field=name), self.assertRaises(ValueError):
                self.batch(self.with_events({**self.records[0], "bundle_raw": raw}, events))
        with self.assertRaises(ValueError):
            self.batch({**self.records[-1], "bundle_raw": self.records[0]["bundle_raw"]})

    def test_raw_formatted_bundle_digest(self):
        record = self.records[0]
        raw = json.dumps(json.loads(record["bundle_raw"]), indent=2).encode()
        changed = {**record, "bundle_raw": raw}
        with self.assertRaises(ValueError):
            self.batch(changed)
        events = self.events(record)
        events[0]["bundle_sha256"] = hashlib.sha256(raw).hexdigest()
        self.assertIs(self.batch(self.with_events(changed, events))["batch_complete"], True)

    def test_runtime_required_and_exact_for_new_batches(self):
        for runtime in (None, {}, [], self.d01_runtime,
                        {**self.runtime, "no_new_privs": 1},
                        {**self.runtime, "extra": True}):
            with self.subTest(runtime=runtime), self.assertRaises(ValueError):
                self.batch(runtime=runtime)

    def test_event_hashes_fields_order_duplicates_and_final_excess(self):
        for record in (self.records[0], self.records[-1]):
            base = self.events(record)
            for position, name, value in (
                (0, "run_id", "f" * 32), (0, "bundle_sha256", "0" * 64),
                (0, "worker_sha256", "0" * 64), (0, "extra", True),
                (1, "request_sha256", "0" * 64), (1, "ordinal", True),
                (1, "ordinal", record["start"]), (1, "extra", True),
                (-1, "received", True), (-1, "received", 0),
                (-1, "stopped", 0), (-1, "extra", True),
            ):
                events = copy.deepcopy(base)
                events[position][name] = value
                with self.subTest(start=record["start"], name=name, position=position):
                    with self.assertRaises(ValueError):
                        self.batch(self.with_events(record, events))
            excess = {**base[-2], "ordinal": len(base) - 2}
            for events in (base[1:], [base[0], base[0]], base[:2] + base[1:],
                           base[:1] + [base[2], base[1]], base + [base[-1]],
                           base[:-1] + [excess, {**base[-1], "received": len(base) - 1}]):
                with self.subTest(start=record["start"]), self.assertRaises(ValueError):
                    self.batch(self.with_events(record, events))

    def test_failed_row_stops_and_missing_tail_survives(self):
        record = self.records[0]
        base = self.events(record)
        for status in ("program_error", "timeout", "output_limit", "launch_failed"):
            events = copy.deepcopy(base[:3])
            events[-1]["result"].update(status=status, returncode=None, stdout_hex="")
            done = {**base[-1], "received": 2, "stopped": True}
            report = self.batch(self.with_events(record, events + [done]))
            self.assertEqual(report["stop_ordinal"], record["start"] + 1)
            self.assertEqual(report["received"], 2)
            self.assertIs(report["batch_complete"], False)
            self.assertEqual(report["rows"][2]["status"], "missing")
            for tail in ([{**done, "stopped": False}], [base[3], done]):
                with self.subTest(status=status), self.assertRaises(ValueError):
                    self.batch(self.with_events(record, events + tail))

    def test_done_short_stop_missing_done_and_full_stopped(self):
        for record in (self.records[0], self.records[-1]):
            events = self.events(record)
            for stream in (events[:-1], events[:-2] + [
                    {**events[-1], "received": len(events) - 3, "stopped": True}],
                    events[:-1] + [{**events[-1], "stopped": True}]):
                self.assertIs(self.batch(self.with_events(record, stream))["batch_complete"],
                              False)
            with self.assertRaises(ValueError):
                self.batch(self.with_events(record, events[:2] + [
                    {**events[-1], "received": 1, "stopped": False}]))

    def test_failed_outer_transport_cannot_complete_even_with_all_rows(self):
        for status in ("timeout", "program_error", "output_limit", "launch_failed"):
            record = copy.deepcopy(self.records[0])
            record["transport"].update(status=status, returncode=None)
            report = self.batch(record)
            self.assertEqual(report["received"], 32)
            self.assertIs(report["batch_complete"], False)

    def test_timeout_partial_tail_and_strict_json(self):
        record = copy.deepcopy(self.records[0])
        prefix = jsonl(self.events(record)[:2])
        for tail in (b'{"kind":', b"\xff"):
            record["transport"] = synthetic_result(prefix + tail)
            with self.assertRaises(ValueError):
                self.batch(record)
            record["transport"].update(status="timeout", returncode=None)
            report = self.batch(record)
            self.assertEqual(report["received"], 1)
            self.assertIs(report["truncated_final_line"], True)
        for raw in (b'{"kind":"ready","kind":"ready"}\n', b"NaN\n", b"{}\n"):
            record["transport"] = synthetic_result(raw)
            with self.assertRaises(ValueError):
                self.batch(record)

    def test_normal_receipt_flags_and_outcome_are_strict(self):
        record = self.records[0]
        for name, value in (("stdin_complete", False), ("process_reaped", False),
                            ("returncode", 1), ("elapsed_ns", True), ("extra", True)):
            events = self.events(record)
            events[1]["result"][name] = value
            with self.subTest(name=name, target="row"), self.assertRaises(ValueError):
                self.batch(self.with_events(record, events))
            changed = copy.deepcopy(record)
            changed["transport"][name] = value
            with self.subTest(name=name, target="transport"), self.assertRaises(ValueError):
                self.batch(changed)
        for raw in (b"", b'{"kind":"return","value":1.0}', b'{"kind":"return","value":NaN}',
                    b'{"kind":"return","value":1,"passed":true}',
                    b'{"kind":"raises","type":"RuntimeError"}'):
            events = self.events(record)
            events[1]["result"]["stdout_hex"] = raw.hex()
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.batch(self.with_events(record, events))

    def test_full_synthetic_matrix_all_tasks_and_observation_counts(self):
        report = self.aggregate()
        self.assertEqual(report["kind"], "LORA_MATRIX_EXECUTION_REPORT")
        self.assertEqual((report["scheduled"], report["received"], report["remaining"]),
                         (1416, 1416, 0))
        self.assertEqual(len(report["rows"]), 1416)
        self.assertEqual(len(report["tasks"]), 44)
        self.assertEqual(report["reference_count"], 354)
        self.assertEqual(report["initial_witness_count"], 44)
        self.assertEqual(report["mutant_witness_count"], 88)
        self.assertEqual(report["witness_count"], 132)
        for name in ("matrix_complete", "data_checks_satisfied",
                     "full_dynamic_validation_complete"):
            self.assertIs(report[name], True, name)
        self.assert_admission_false(report)
        crossing = [task for task in report["tasks"]
                    if sum(row["task_id"] == task["task_id"]
                           for row in report["rows"]) == 40]
        self.assertEqual(len(crossing), 1)
        for task in report["tasks"]:
            for name in ("reference_ok", "initial_scoring_rejected", "witnesses_ok"):
                self.assertIs(task[name], True, (task["task_id"], name))

    def test_absent_batches_and_partial_final_batch_keep_1416_denominator(self):
        for records, received in (([], 32), (self.records[:1], 64),
                                  (self.records[:-1], 1408)):
            report = self.aggregate(records)
            self.assertEqual(report["received"], received)
            self.assertEqual(report["remaining"], 1416 - received)
            self.assertEqual(len(report["rows"]), 1416)
            self.assertEqual(len(report["tasks"]), 44)
            self.assertIs(report["matrix_complete"], False)
            self.assertIs(report["data_checks_satisfied"], False)
        final = self.records[-1]
        events = self.events(final)
        partial = self.with_events(final, events[:5] + [
            {**events[-1], "received": 4, "stopped": True}])
        report = self.aggregate(self.records[:-1] + [partial])
        self.assertEqual(report["received"], 1412)
        self.assertEqual(report["remaining"], 4)
        self.assertEqual([row["status"] for row in report["rows"][-4:]], ["missing"] * 4)
        self.assertIs(report["full_dynamic_validation_complete"], False)

    def test_aggregate_rejects_gaps_reordering_duplicates_and_report_fields(self):
        invalid = [
            [self.records[1]], [self.records[0], self.records[2]],
            [self.records[1], self.records[0]], [self.records[0], self.records[0]],
            [{**self.records[0], "batch_complete": True}],
            [{name: value for name, value in self.records[0].items() if name != "transport"}],
        ]
        for records in invalid:
            with self.subTest(starts=[record.get("start") for record in records]):
                with self.assertRaises(ValueError):
                    self.aggregate(records)

    def test_aggregate_rejects_reused_run_ids_including_d01(self):
        for prior in (self.d01, self.records[0]):
            record = self.records[1]
            bundle = json.loads(record["bundle_raw"])
            bundle["run_id"] = json.loads(prior["bundle_raw"])["run_id"]
            raw = encoded(bundle)
            events = self.events(record)
            for event in events:
                event["run_id"] = bundle["run_id"]
            events[0]["bundle_sha256"] = hashlib.sha256(raw).hexdigest()
            changed = self.with_events({**record, "bundle_raw": raw}, events)
            self.assertIs(self.batch(changed)["batch_complete"], True)
            with self.assertRaises(ValueError):
                self.aggregate([self.records[0], changed])

    def test_aggregate_prohibits_records_after_incomplete_batch(self):
        record = self.records[0]
        variants = [self.with_events(record, self.events(record)[:3]),
                    {**record, "transport": {**record["transport"], "status": "timeout",
                                             "returncode": None}}]
        for changed in variants:
            report = self.aggregate([changed])
            self.assertIs(report["matrix_complete"], False)
            with self.assertRaises(ValueError):
                self.aggregate([changed, self.records[1]])

    def test_final_transport_failure_keeps_data_checks_separate_from_completion(self):
        record = self.records[-1]
        variants = [self.with_events(record, self.events(record)[:-1]),
                    {**record, "transport": {**record["transport"], "status": "timeout",
                                             "returncode": None}}]
        for changed in variants:
            report = self.aggregate(self.records[:-1] + [changed])
            self.assertEqual(report["received"], 1416)
            self.assertEqual(report["remaining"], 0)
            self.assertIs(report["matrix_complete"], False)
            self.assertIs(report["data_checks_satisfied"], True)
            self.assertIs(report["full_dynamic_validation_complete"], False)
            self.assert_admission_false(report)

    def test_cross_chunk_task_witness_failure_is_computed_over_the_whole_task(self):
        task_sizes = {}
        for planned in self.plan["rows"]:
            task_sizes[planned["task_id"]] = task_sizes.get(planned["task_id"], 0) + 1
        task_id = next(task_id for task_id, size in task_sizes.items() if size == 40)
        planned = next(planned for planned in reversed(self.plan["rows"])
                       if planned["task_id"] == task_id
                       and planned["witness_prediction"] is not None)
        start = planned["ordinal"] // 32 * 32
        position = start // 32 - 1
        records = list(self.records)
        record = records[position]
        events = self.events(record)
        events[planned["ordinal"] - start + 1]["result"]["stdout_hex"] = encoded(
            planned["expected"]).hex()
        records[position] = self.with_events(record, events)
        report = self.aggregate(records)
        task = next(task for task in report["tasks"] if task["task_id"] == task_id)
        self.assertIs(task["witnesses_ok"], False)
        self.assertIs(task["reference_ok"], True)
        self.assertIs(task["initial_scoring_rejected"], True)
        self.assertIs(report["matrix_complete"], True)
        self.assertIs(report["data_checks_satisfied"], False)

    def test_d01_revalidated_requires_completion_checks_and_own_runtime(self):
        for runtime in (self.runtime, None, {}):
            with self.subTest(runtime=runtime), self.assertRaises(ValueError):
                self.aggregate([], d01_runtime=runtime)
        with self.assertRaises(ValueError):
            self.aggregate([], runtime=None)
        events = self.events(self.d01)
        with self.assertRaises(ValueError):
            self.aggregate([], d01=self.with_events(self.d01, events[:-1]))
        for role in ("reference", "initial", "witness"):
            changed = copy.deepcopy(events)
            for planned in self.plan["rows"][:32]:
                if ((role == "reference" and planned["role"] == "reference")
                        or (role == "initial" and planned["role"] == "initial"
                            and planned["group"] == "scoring")
                        or (role == "witness" and planned["witness_prediction"] is not None)):
                    outcome = ({"kind": "return", "value": "__wrong_reference__"}
                               if role == "reference" else planned["expected"])
                    changed[planned["ordinal"] + 1]["result"]["stdout_hex"] = encoded(outcome).hex()
            with self.subTest(role=role), self.assertRaises(ValueError):
                self.aggregate([], d01=self.with_events(self.d01, changed))

    def test_reference_witness_and_initial_scoring_fail_independently_without_stop(self):
        task_id = self.plan["rows"][32]["task_id"]
        for failing_check in ("reference_ok", "witnesses_ok", "initial_scoring_rejected"):
            records = list(self.records)
            record = records[0]
            events = self.events(record)
            for planned in self.plan["rows"][32:64]:
                if planned["task_id"] != task_id:
                    continue
                selected = (
                    (failing_check == "reference_ok" and planned["role"] == "reference")
                    or (failing_check == "witnesses_ok"
                        and planned["witness_prediction"] is not None)
                    or (failing_check == "initial_scoring_rejected"
                        and planned["role"] == "initial" and planned["group"] == "scoring"))
                if selected:
                    outcome = ({"kind": "return", "value": "__wrong_reference__"}
                               if failing_check == "reference_ok" else planned["expected"])
                    events[planned["ordinal"] - 32 + 1]["result"]["stdout_hex"] = encoded(outcome).hex()
            records[0] = self.with_events(record, events)
            report = self.aggregate(records)
            self.assertIs(report["matrix_complete"], True)
            self.assertIs(report["data_checks_satisfied"], False)
            self.assertIs(report["full_dynamic_validation_complete"], False)
            task = next(task for task in report["tasks"] if task["task_id"] == task_id)
            for name in ("reference_ok", "witnesses_ok", "initial_scoring_rejected"):
                self.assertIs(task[name], name != failing_check, name)
            self.assert_admission_false(report)

    def test_frozen_inputs_and_returned_mutation_do_not_bypass_revalidation(self):
        for source, manifest in ((self.source + b"\n", self.manifest),
                                 (self.source, self.manifest + b"\n")):
            with self.assertRaises(ValueError):
                self.module.build_batch_report(
                    source, manifest, self.records[0]["bundle_raw"],
                    self.records[0]["transport"], start=32, expected_runtime=self.runtime)
        report = self.batch()
        report["runtime"]["argv"].append("changed")
        report["rows"][0]["result"]["status"] = "changed"
        self.assertEqual(self.runtime["argv"], ["/synthetic/new"])
        self.assertIs(self.batch()["batch_complete"], True)


if __name__ == "__main__":
    unittest.main()
