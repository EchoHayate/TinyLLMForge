import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
import unittest

from pilot_protocol import FLAGS
import test_validation_matrix as matrix_tests
from validation_contract import digest, encoded
from validation_matrix import aggregate_matrix


ROOT = Path(__file__).parent
JUMP_REJECTION = (
    b"debug1: Authenticating to jump-proxy.example:22 as 'synthetic'\n"
    b"debug1: No more authentication methods to try.\n"
    b"synthetic@jump-proxy.example: Permission denied (gssapi-with-mic).\n"
    b"kex_exchange_identification: Connection closed by remote host\n"
)


class ValidationTailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        matrix_tests.ValidationMatrixTests.setUpClass.__func__(cls)
        failed = copy.deepcopy(cls.records[33])
        bundle = json.loads(failed["bundle_raw"])
        bundle["run_id"] = "f" * 32
        failed["bundle_raw"] = encoded(bundle)
        failed["transport"] = {
            **matrix_tests.synthetic_result(b""),
            "status": "program_error", "returncode": 255,
            "stdin_complete": False, "stderr_hex": JUMP_REJECTION.hex(),
        }
        cls.previous = copy.deepcopy(cls.records[:33]) + [failed]
        cls.tail = copy.deepcopy(cls.records[33:])
        cls.authorization = {
            "kind": "LORA_TAIL_AUTHORIZATION",
            "prior_stage_id": "e" * 32, "stage_id": "d" * 32,
            "prior_seal_sha256": "a" * 64,
            "failed_run_id": bundle["run_id"],
            "failed_transport_sha256": digest(failed["transport"]),
            "resubmit_start": 1088, "resubmit_stop": 1120, "tail_stop": 1416,
            "user_approved": True,
        }

    def setUp(self):
        self.assertTrue((ROOT / "validation_tail.py").is_file())
        self.module = importlib.import_module("validation_tail")

    events = matrix_tests.ValidationMatrixTests.events
    with_events = matrix_tests.ValidationMatrixTests.with_events

    def aggregate(self, *, previous=None, tail=None, authorization=None, **kwargs):
        d01 = kwargs.get("d01", self.d01)
        return self.module.aggregate_tail(
            kwargs.get("source", self.source), kwargs.get("manifest", self.manifest),
            d01["bundle_raw"], d01["transport"],
            self.previous if previous is None else previous,
            self.tail if tail is None else tail,
            expected_runtime=kwargs.get("runtime", self.runtime),
            d01_expected_runtime=kwargs.get("d01_runtime", self.d01_runtime),
            authorization=self.authorization if authorization is None else authorization)

    def reidentify(self, record, run_id):
        changed = copy.deepcopy(record)
        bundle = json.loads(changed["bundle_raw"])
        bundle["run_id"] = run_id
        changed["bundle_raw"] = encoded(bundle)
        events = self.events(changed)
        for event in events:
            event["run_id"] = run_id
        if events:
            events[0]["bundle_sha256"] = hashlib.sha256(changed["bundle_raw"]).hexdigest()
            changed["transport"]["stdout_hex"] = matrix_tests.jsonl(events).hex()
        return changed

    def rebound_authorization(self, failed):
        return {
            **self.authorization,
            "failed_run_id": json.loads(failed["bundle_raw"])["run_id"],
            "failed_transport_sha256": digest(failed["transport"]),
        }

    def assert_admission_false(self, report):
        for name in (*FLAGS, "generated_candidate_backend_ready"):
            self.assertIs(report[name], False, name)

    def test_reporter_exists(self):
        self.assertTrue((ROOT / "validation_tail.py").is_file())
        self.assertTrue(callable(
            importlib.import_module("validation_tail").aggregate_tail))

    def test_reporter_imports_only_pure_modules(self):
        tree = ast.parse((ROOT / "validation_tail.py").read_text())
        allowed = {"copy", "re", "pilot_protocol", "validation_contract",
                   "validation_matrix"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name in allowed for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, ("exec", "eval", "compile", "open"))

    def test_complete_retains_prior_failure_authority_and_exact_selected_batches(self):
        report = self.aggregate()
        self.assertEqual(set(report), {
            "kind", "authorization", "prior", "combined", "selected_batches"})
        self.assertEqual(report["kind"], "LORA_MATRIX_CONTINUATION_REPORT")
        self.assertEqual(report["authorization"], self.authorization)
        expected_prior = aggregate_matrix(
            self.source, self.manifest, self.d01["bundle_raw"], self.d01["transport"],
            self.previous, expected_runtime=self.runtime,
            d01_expected_runtime=self.d01_runtime)
        self.assertEqual(report["prior"], expected_prior)
        self.assertEqual((report["prior"]["received"], report["prior"]["remaining"]),
                         (1088, 328))
        self.assertIs(report["prior"]["matrix_complete"], False)
        failed = report["prior"]["batches"][-1]
        self.assertEqual(failed["run_id"], self.authorization["failed_run_id"])
        self.assertEqual(failed["transport"], self.previous[-1]["transport"])
        self.assertEqual(failed["transport_sha256"],
                         self.authorization["failed_transport_sha256"])
        self.assertEqual(failed["received"], 0)
        self.assertIsNone(failed["runtime"])
        self.assertIsNone(failed["done"])
        combined = report["combined"]
        expected_combined = aggregate_matrix(
            self.source, self.manifest, self.d01["bundle_raw"], self.d01["transport"],
            self.previous[:-1] + self.tail, expected_runtime=self.runtime,
            d01_expected_runtime=self.d01_runtime)
        expected_combined["kind"] = "LORA_MATRIX_CROSS_STAGE_OBSERVATIONS"
        self.assertEqual(combined, expected_combined)
        self.assertEqual((combined["scheduled"], combined["received"],
                          combined["remaining"]), (1416, 1416, 0))
        self.assertEqual(len(combined["tasks"]), 44)
        self.assertEqual(len(combined["rows"]), 1416)
        self.assertEqual(combined["reference_count"], 354)
        self.assertEqual(combined["witness_count"], 132)
        self.assertIs(combined["full_dynamic_validation_complete"], True)
        self.assertEqual(report["selected_batches"], [
            {"stage_id": self.authorization[stage], "start": record["start"],
             "run_id": json.loads(record["bundle_raw"])["run_id"]}
            for stage, records in (("prior_stage_id", self.previous[:-1]),
                                   ("stage_id", self.tail))
            for record in records
        ])
        self.assertNotIn(failed["run_id"],
                         [batch["run_id"] for batch in report["selected_batches"]])
        for aggregate in (report["prior"], combined):
            self.assert_admission_false(aggregate)
            for batch in [aggregate["d01"], *aggregate["batches"]]:
                self.assert_admission_false(batch)

    def test_empty_and_partial_prefixes_keep_full_denominator_and_selected_attempts(self):
        for count in (0, 1, 10):
            with self.subTest(count=count):
                report = self.aggregate(tail=self.tail[:count])
                combined = report["combined"]
                self.assertEqual(combined["received"], 1088 + 32 * count)
                self.assertEqual(combined["remaining"], 328 - 32 * count)
                self.assertEqual(len(combined["rows"]), 1416)
                self.assertEqual(len(combined["tasks"]), 44)
                self.assertEqual(len(report["selected_batches"]), 33 + count)
                self.assertIs(combined["full_dynamic_validation_complete"], False)
                self.assert_admission_false(combined)
        events = self.events(self.tail[0])
        for received in (0, 1, 31, 32):
            with self.subTest(received=received):
                partial = self.with_events(self.tail[0], events[:received + 1])
                report = self.aggregate(tail=[partial])
                self.assertEqual(report["combined"]["received"], 1088 + received)
                self.assertEqual(len(report["selected_batches"]), 34)
                self.assertIs(report["combined"]["matrix_complete"], False)
                self.assertIs(report["combined"]["full_dynamic_validation_complete"], False)

    def test_final_eight_partial_missing_done_and_outer_failure(self):
        record = self.tail[-1]
        events = self.events(record)
        partial = self.with_events(record, events[:5] + [
            {**events[-1], "received": 4, "stopped": True}])
        report = self.aggregate(tail=self.tail[:-1] + [partial])["combined"]
        self.assertEqual(report["received"], 1412)
        self.assertEqual(report["remaining"], 4)
        self.assertEqual([row["status"] for row in report["rows"][-4:]], ["missing"] * 4)
        self.assertIs(report["matrix_complete"], False)
        variants = [
            self.with_events(record, events[:-1]),
            {**record, "transport": {**record["transport"],
                                     "status": "timeout", "returncode": None}},
        ]
        for final in variants:
            with self.subTest(transport=final["transport"]["status"]):
                report = self.aggregate(tail=self.tail[:-1] + [final])["combined"]
                self.assertEqual(report["received"], 1416)
                self.assertEqual(report["remaining"], 0)
                self.assertIs(report["matrix_complete"], False)
                self.assertIs(report["data_checks_satisfied"], True)
                self.assertIs(report["full_dynamic_validation_complete"], False)
                self.assertEqual(report["batches"][-1]["batch_scheduled"], 8)

    def test_authorization_exact_fields_types_and_bounds(self):
        invalid = [[], {}, {**self.authorization, "extra": True}]
        invalid.extend({name: value for name, value in self.authorization.items()
                        if name != missing} for missing in self.authorization)
        for name, values in {
            "kind": ("other", None, 1),
            "user_approved": (False, 1, "true", None),
            "resubmit_start": (1088.0, True, 1056, "1088"),
            "resubmit_stop": (1120.0, True, 1152, "1120"),
            "tail_stop": (1416.0, True, 1408, "1416"),
            "prior_stage_id": ("A" * 32, "a" * 31, "a" * 33, "z" * 32, None, 1),
            "stage_id": ("A" * 32, "a" * 31, "a" * 33, "z" * 32, None, []),
            "failed_run_id": ("F" * 32, "f" * 31, "f" * 33, "z" * 32, None),
            "prior_seal_sha256": ("A" * 64, "a" * 63, "a" * 65, "z" * 64, None),
            "failed_transport_sha256": ("A" * 64, "a" * 63, "a" * 65, "z" * 64, None),
        }.items():
            invalid.extend({**self.authorization, name: value} for value in values)
        invalid.append({**self.authorization, "stage_id":
                        self.authorization["prior_stage_id"]})
        for authorization in invalid:
            with self.subTest(authorization=authorization), self.assertRaises(ValueError):
                self.aggregate(tail=[], authorization=authorization)

    def test_authorization_kind_requires_plain_string_and_containers_are_required(self):
        class StringSubclass(str):
            pass

        with self.assertRaises(ValueError):
            self.aggregate(tail=[], authorization={
                **self.authorization, "kind": StringSubclass("LORA_TAIL_AUTHORIZATION")})
        for name in ("previous_records", "tail_records", "authorization"):
            for value in (None, (), ""):
                arguments = {
                    "previous_records": self.previous,
                    "tail_records": [], "authorization": self.authorization,
                    name: value,
                }
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.module.aggregate_tail(
                        self.source, self.manifest, self.d01["bundle_raw"],
                        self.d01["transport"], expected_runtime=self.runtime,
                        d01_expected_runtime=self.d01_runtime, **arguments)

    def test_failed_run_and_raw_transport_hash_are_bound(self):
        for name, value in (("failed_run_id", "b" * 32),
                            ("failed_transport_sha256", "b" * 64)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.aggregate(authorization={**self.authorization, name: value})
        previous = copy.deepcopy(self.previous)
        previous[-1]["transport"]["elapsed_ns"] += 1
        with self.assertRaises(ValueError):
            self.aggregate(previous=previous)
        report = self.aggregate(
            previous=previous, authorization=self.rebound_authorization(previous[-1]))
        self.assertEqual(report["prior"]["batches"][-1]["transport"],
                         previous[-1]["transport"])
        report = self.aggregate(authorization={
            **self.authorization, "prior_seal_sha256": "b" * 64})
        self.assertEqual(report["authorization"]["prior_seal_sha256"], "b" * 64)

    def test_prior_requires_exact_count_order_and_completed_prefix(self):
        variants = [
            [], self.previous[:-1], self.previous + [self.tail[0]],
            tuple(self.previous),
            self.previous[:-1] + [{**self.previous[-1], "start": 1056}],
            self.previous[:-1] + [{**self.previous[-1], "start": 1088.0}],
            self.previous[:-1] + [{**self.previous[-1], "start": True}],
            self.previous[:1] + self.previous[:1] + self.previous[2:],
        ]
        partial = self.with_events(self.previous[0], self.events(self.previous[0])[:-1])
        variants.append([partial] + self.previous[1:])
        for previous in variants:
            with self.subTest(count=len(previous)), self.assertRaises(ValueError):
                self.aggregate(previous=previous)

    def test_prior_failure_must_be_zero_receipt_program_error_255(self):
        failed = self.previous[-1]
        variants = []
        for name, value in (
                ("status", "completed"), ("status", "timeout"),
                ("status", "launch_failed"), ("returncode", 1),
                ("returncode", None), ("returncode", 255.0),
                ("stdout_hex", b"\n".hex()), ("stderr_hex", "zz")):
            variants.append({**failed, "transport": {**failed["transport"], name: value}})
        success = self.reidentify(self.tail[0], self.authorization["failed_run_id"])
        for count in (1, 2, 34):
            changed = self.with_events(success, self.events(success)[:count])
            changed["transport"].update(status="program_error", returncode=255,
                                        stderr_hex=JUMP_REJECTION.hex())
            variants.append(changed)
        for changed in variants:
            with self.subTest(transport=changed["transport"]), self.assertRaises(ValueError):
                self.aggregate(previous=self.previous[:-1] + [changed],
                               authorization=self.rebound_authorization(changed))

    def test_stderr_requires_jump_rejection_and_no_target_launch(self):
        invalid = [
            b"", b"Permission denied (gssapi-with-mic).\n",
            b"user@target.example: Permission denied (publickey).\n",
            b"jump-proxy.example connection timed out\n",
            b"user@target-jump-proxy.example: Permission denied (publickey).\n",
        ]
        invalid.extend(JUMP_REJECTION + marker for marker in (
            b"debug1: Sending command: python3 -u -\n",
            b"debug1: Authenticated to target.example using gssapi-with-mic.\n",
            b"debug1: Authentication succeeded (publickey).\n",
            b"debug2: channel 0: request exec confirm 1\n",
            b"debug2: exec request accepted on channel 0\n",
            b"debug1: Entering interactive session.\n",
            b"debug1: Sending subsystem: sftp\n",
        ))
        for stderr in invalid:
            failed = copy.deepcopy(self.previous[-1])
            failed["transport"]["stderr_hex"] = stderr.hex()
            with self.subTest(stderr=stderr), self.assertRaises(ValueError):
                self.aggregate(previous=self.previous[:-1] + [failed],
                               authorization=self.rebound_authorization(failed))

    def test_all_run_ids_unique_including_removed_failure_d01_and_stage_ids(self):
        collisions = [
            self.authorization["failed_run_id"],
            json.loads(self.d01["bundle_raw"])["run_id"],
            json.loads(self.previous[0]["bundle_raw"])["run_id"],
            self.authorization["stage_id"], self.authorization["prior_stage_id"],
        ]
        for run_id in collisions:
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                self.aggregate(tail=[self.reidentify(self.tail[0], run_id)])
        duplicate = self.reidentify(
            self.tail[1], json.loads(self.tail[0]["bundle_raw"])["run_id"])
        with self.assertRaises(ValueError):
            self.aggregate(tail=[self.tail[0], duplicate])
        for record in (self.d01, self.previous[0], self.previous[-1]):
            for name in ("stage_id", "prior_stage_id"):
                authorization = {**self.authorization,
                                 name: json.loads(record["bundle_raw"])["run_id"]}
                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.aggregate(tail=[], authorization=authorization)
        for record in (self.d01, self.previous[0]):
            failed = self.reidentify(
                self.previous[-1], json.loads(record["bundle_raw"])["run_id"])
            with self.assertRaises(ValueError):
                self.aggregate(previous=self.previous[:-1] + [failed],
                               authorization=self.rebound_authorization(failed))

    def test_tail_exact_record_schema_order_bounds_and_count(self):
        invalid = [
            tuple(self.tail), self.tail + [self.tail[-1]], [self.tail[1]],
            [self.tail[0], self.tail[0]], [self.tail[0], self.tail[2]],
            [{**self.tail[0], "start": 1088.0}],
            [{**self.tail[0], "start": True}],
            [{**self.tail[0], "start": 1056}],
            [{**self.tail[0], "start": 1416}],
            [{**self.tail[0], "batch_complete": True}],
            [{name: value for name, value in self.tail[0].items()
              if name != "transport"}],
            [None],
        ]
        for tail in invalid:
            with self.subTest(tail_count=len(tail)), self.assertRaises(ValueError):
                self.aggregate(tail=tail)

    def test_no_tail_record_after_incomplete_batch_or_failed_row(self):
        first = self.tail[0]
        events = self.events(first)
        failed_events = copy.deepcopy(events[:3])
        failed_events[-1]["result"].update(
            status="timeout", returncode=None, stdout_hex="")
        failed_events.append({**events[-1], "received": 2, "stopped": True})
        variants = [
            self.with_events(first, events[:-1]),
            self.with_events(first, events[:3]),
            self.with_events(first, failed_events),
            {**first, "transport": {**first["transport"],
                                    "status": "program_error", "returncode": 255}},
            {**first, "transport": {**first["transport"],
                                    "status": "program_error", "returncode": 255,
                                    "stdout_hex": ""}},
        ]
        for changed in variants:
            with self.subTest(transport=changed["transport"]["status"]):
                report = self.aggregate(tail=[changed])
                self.assertIs(report["combined"]["matrix_complete"], False)
                self.assertEqual(len(report["selected_batches"]), 34)
                with self.assertRaises(ValueError):
                    self.aggregate(tail=[changed, self.tail[1]])

    def test_runtime_and_frozen_inputs_revalidated_in_both_stages_and_d01(self):
        for kwargs in (
                {"runtime": {}}, {"runtime": None}, {"runtime": self.d01_runtime},
                {"runtime": {**self.runtime, "no_new_privs": 1}},
                {"d01_runtime": self.runtime}, {"d01_runtime": None},
                {"source": self.source + b"\n"}, {"manifest": self.manifest + b"\n"}):
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                self.aggregate(**kwargs)
        for name, records in (("previous", self.previous), ("tail", self.tail)):
            events = self.events(records[0])
            events[0]["runtime"] = {**self.runtime, "no_new_privs": 1}
            changed = [self.with_events(records[0], events)] + records[1:]
            with self.subTest(stage=name), self.assertRaises(ValueError):
                self.aggregate(**{name: changed})
        d01 = self.with_events(self.d01, self.events(self.d01)[:-1])
        with self.assertRaises(ValueError):
            self.aggregate(d01=d01)

    def test_projection_and_stream_binding_revalidated_even_for_removed_failure(self):
        for stage, records, position in (
                ("previous", self.previous, 0), ("previous", self.previous, -1),
                ("tail", self.tail, 0), ("tail", self.tail, -1)):
            for field, value in (("source", "pass"), ("expected", {}),
                                 ("args", []), ("request_id", "request-0000")):
                changed = copy.deepcopy(records)
                record = changed[position]
                bundle = json.loads(record["bundle_raw"])
                bundle["requests"][0][field] = value
                record["bundle_raw"] = encoded(bundle)
                events = self.events(record)
                if events:
                    events[0]["bundle_sha256"] = hashlib.sha256(
                        record["bundle_raw"]).hexdigest()
                    record["transport"]["stdout_hex"] = matrix_tests.jsonl(events).hex()
                with self.subTest(stage=stage, field=field, position=position):
                    with self.assertRaises(ValueError):
                        self.aggregate(**{stage: changed})
        events = self.events(self.tail[0])
        for malformed in (events[1:], events[:2] + events[1:], events + [events[-1]],
                          [events[0], {**events[1], "request_sha256": "0" * 64}]):
            with self.assertRaises(ValueError):
                self.aggregate(tail=[self.with_events(self.tail[0], malformed)])
        invalid = {**self.tail[0], "transport": {
            **self.tail[0]["transport"], "stdout_hex": b"not-json\n".hex()}}
        with self.assertRaises(ValueError):
            self.aggregate(tail=[invalid])

    def test_t06_task_checks_span_stages_and_negative_outcomes_do_not_stop(self):
        t06_rows = [row for row in self.plan["rows"] if row["task_id"] == "T06"]
        self.assertTrue(any(row["ordinal"] < 1088 for row in t06_rows))
        self.assertTrue(any(row["ordinal"] >= 1088 for row in t06_rows))
        full = self.aggregate()
        task = next(task for task in full["combined"]["tasks"] if task["task_id"] == "T06")
        self.assertEqual(task, {"task_id": "T06", "reference_ok": True,
                                "initial_scoring_rejected": True, "witnesses_ok": True})
        for failing_check in ("reference_ok", "witnesses_ok", "initial_scoring_rejected"):
            previous, tail = copy.deepcopy(self.previous), copy.deepcopy(self.tail)
            changed_ordinals = []
            for planned in t06_rows:
                selected = (
                    failing_check == "reference_ok" and planned["role"] == "reference"
                    or failing_check == "witnesses_ok"
                    and planned["witness_prediction"] is not None
                    or failing_check == "initial_scoring_rejected"
                    and planned["role"] == "initial" and planned["group"] == "scoring")
                if not selected:
                    continue
                ordinal = planned["ordinal"]
                records = previous if ordinal < 1088 else tail
                start = ordinal // 32 * 32
                position = next(index for index, record in enumerate(records)
                                if record["start"] == start)
                record = records[position]
                events = self.events(record)
                outcome = ({"kind": "return", "value": "__wrong_reference__"}
                           if failing_check == "reference_ok" else planned["expected"])
                events[ordinal - start + 1]["result"]["stdout_hex"] = encoded(outcome).hex()
                records[position] = self.with_events(record, events)
                changed_ordinals.append(ordinal)
            with self.subTest(check=failing_check):
                self.assertTrue(changed_ordinals)
                report = self.aggregate(previous=previous, tail=tail)
                combined = report["combined"]
                self.assertEqual(combined["received"], 1416)
                self.assertIs(combined["matrix_complete"], True)
                self.assertIs(combined["data_checks_satisfied"], False)
                self.assertIs(combined["full_dynamic_validation_complete"], False)
                self.assertEqual(len(report["selected_batches"]), 44)
                self.assert_admission_false(combined)
                task = next(task for task in combined["tasks"] if task["task_id"] == "T06")
                for name in ("reference_ok", "witnesses_ok", "initial_scoring_rejected"):
                    self.assertIs(task[name], name != failing_check)

    def test_result_mutation_cannot_change_inputs_or_other_report_sections(self):
        previous, tail = copy.deepcopy(self.previous), copy.deepcopy(self.tail)
        authorization = dict(self.authorization)
        report = self.aggregate(previous=previous, tail=tail, authorization=authorization)
        report["authorization"]["user_approved"] = False
        report["prior"]["batches"][0]["transport"]["stdout_hex"] = ""
        report["combined"]["expected_runtime"]["argv"].append("changed")
        report["selected_batches"][0]["stage_id"] = "changed"
        self.assertEqual(previous, self.previous)
        self.assertEqual(tail, self.tail)
        self.assertEqual(authorization, self.authorization)
        self.assertEqual(report["combined"]["batches"][0]["transport"],
                         self.previous[0]["transport"])
        self.assertEqual(report["prior"]["expected_runtime"], self.runtime)
        self.assertIs(self.aggregate()["combined"]["full_dynamic_validation_complete"], True)


if __name__ == "__main__":
    unittest.main()
