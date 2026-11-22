import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import traceback
import unittest

from pilot_protocol import FLAGS
import test_recalibration_tokenizer_inventory as inventory_fixture
from test_tokenizer_audit_report import synthetic_profile, synthetic_transcript, rechain
from validation_contract import digest, encoded

if importlib.util.find_spec("recalibration_tokenizer_report") is not None:
    import recalibration_tokenizer_report as report_module
else:
    report_module = None


REAL_FLAGS = (*FLAGS, "all_possible_second_rounds_validated", "tokenizer_length_validated",
              "execution_provenance_verified", "real_dynamic_validation_complete",
              "run_authorized", "tokenizer_dispatch_authorized")
CATEGORY_TOTALS = {"first": 24, "references": 48, "sampled_second": 48}
IDENTITY_FIELDS = ("ordinal", "category", "task_id", "arm", "reference_id", "branch",
                   "row_sha256")


class StringSubclass(str):
    pass


class IntegerSubclass(int):
    pass


class ListSubclass(list):
    pass


class DictSubclass(dict):
    pass


def replace(value, path, replacement):
    for name in path[:-1]:
        value = value[name]
    value[path[-1]] = replacement


def repartition(measurement):
    target = measurement["target"]
    target["target_start"] = measurement["input"]["input_tokens"]
    target["target_tokens"] = len(target["target_ids"])
    target["full_sequence_ids"] = (measurement["input"]["input_ids"]
                                    + target["target_ids"] + target["template_tail_ids"])
    target["full_sequence_tokens"] = len(target["full_sequence_ids"])


def fixture_inputs():
    fixture = inventory_fixture.RecalibrationInventoryTests
    fixture.setUpClass()
    return {
        "raw_files": fixture.raw, "worker_raw": fixture.worker,
        "run_ids": fixture.run_ids, "records": fixture.records,
        "rendering_raw": fixture.renderer, "expected_runtime": fixture.runtime,
    }


class RecalibrationTokenizerReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from recalibration_tokenizer_inventory import build_synthetic_inventory
        cls.inputs = fixture_inputs()
        cls.inventory = build_synthetic_inventory(**cls.inputs)
        cls.profile = synthetic_profile()
        cls.profile_raw = encoded(cls.profile)
        cls.profile_sha = digest(cls.profile)
        cls.records, cls.finalization = synthetic_transcript(cls.inventory, cls.profile, count=120)

    def setUp(self):
        self.assertIsNotNone(report_module, "recalibration tokenizer report not implemented")

    def report(self, records=None, finalization=None, **options):
        return report_module.build_synthetic_report(
            self.inputs, self.profile_raw, self.profile_sha,
            self.records if records is None else records,
            self.finalization if finalization is None else finalization, **options)

    def invalid(self):
        return self.assertRaisesRegex(ValueError, "^tokenizer audit evidence invalid$")

    def failed_report(self, records, phase, ordinal, abort_failed=False):
        return report_module.build_synthetic_report(
            self.inputs, self.profile_raw, self.profile_sha, records, None,
            failure={"phase": phase, "ordinal": ordinal}, abort_failed=abort_failed)

    def test_complete_synthetic_report_and_public_boundary(self):
        actual = self.report()
        public = actual["public"]
        self.assertEqual(public["kind"], "LORA_RECALIBRATION_TOKENIZER_REPORT")
        self.assertEqual(public["status"], "SYNTHETIC_COMPLETE")
        self.assertEqual((public["planned"], public["checked"], public["not_checked"]), (120, 120, 0))
        self.assertTrue(public["synthetic_all_rows_within_budget"])
        self.assertEqual({name: count["scheduled"] for name, count in public["by_category"].items()},
                         {"first": 24, "references": 48, "sampled_second": 48})
        self.assertEqual(len(public["rows"]), 120)
        self.assertNotIn("private synthetic", encoded(public).decode())
        for flag in (*FLAGS, "all_possible_second_rounds_validated", "tokenizer_length_validated",
                     "execution_provenance_verified", "real_dynamic_validation_complete",
                     "run_authorized", "tokenizer_dispatch_authorized"):
            self.assertIs(public[flag], False)
        actual["private"]["records"][0]["measurement"]["input"]["input_ids"][0] = 99
        self.assertNotEqual(actual, self.report())

    def test_dependency_failure_and_no_prebuilt_inventory(self):
        inputs = copy.deepcopy(self.inputs)
        inputs["records"] = []
        actual = report_module.build_synthetic_report(
            inputs, self.profile_raw, self.profile_sha, [], None)
        self.assertEqual(actual["public"]["status"], "DEPENDENCY_UNSATISFIED")
        self.assertEqual(actual["public"]["not_checked"], 120)
        self.assertEqual(actual["public"]["rows"], [])
        with self.assertRaises(ValueError):
            report_module.build_synthetic_report(
                self.inventory, self.profile_raw, self.profile_sha, [], None)
        with self.assertRaises(ValueError):
            report_module.build_synthetic_report(
                inputs, self.profile_raw, self.profile_sha, self.records, self.finalization)

    def test_input_and_answer_over_limit_stop(self):
        for count, stop in ((1, "INPUT_LIMIT"), (25, "OUTPUT_LIMIT")):
            records, final = synthetic_transcript(self.inventory, self.profile, count=count, stop=stop)
            public = self.report(records, final)["public"]
            self.assertEqual(public["stop_reason"], stop)
            self.assertEqual(public["checked"], count)
            self.assertEqual(public["not_checked"], 120 - count)
            self.assertFalse(public["synthetic_all_rows_within_budget"])
            with self.assertRaises(ValueError):
                self.report(records + [self.records[count]], final)

    def test_exact_budget_boundaries(self):
        records, final = copy.deepcopy((self.records, self.finalization))
        observed = records[0]["measurement"]["input"]
        observed.update(input_ids=[11] * 4096, input_tokens=4096, remaining_input_tokens=0)
        target = records[24]["measurement"]["target"]
        target["target_ids"] = [12] * 255 + [2]
        target["target_tokens"] = 256
        target["full_sequence_ids"] = (records[24]["measurement"]["input"]["input_ids"]
                                       + target["target_ids"] + target["template_tail_ids"])
        target["full_sequence_tokens"] = len(target["full_sequence_ids"])
        rechain(records, final)
        self.assertTrue(self.report(records, final)["public"]["synthetic_all_rows_within_budget"])

    def test_binding_and_chain_tampering(self):
        for path, replacement in (
            (("run_id",), "b" * 32), (("ordinal",), True),
            (("profile_sha256",), "0" * 64), (("identity_sha256",), "0" * 64),
            (("measurement", "state_sha256"), "0" * 64),
            (("measurement", "input", "messages_sha256"), "0" * 64),
            (("measurement", "input", "input_tokens"), True),
            (("measurement", "status"), "INPUT_LIMIT"),
        ):
            records, final = copy.deepcopy((self.records, self.finalization))
            changed = records[1]
            for name in path[:-1]:
                changed = changed[name]
            changed[path[-1]] = replacement
            rechain(records, final)
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.report(records, final)
        records = copy.deepcopy(self.records)
        records[0]["chain_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.report(records)

    def test_finalization_eos_missing_and_failure(self):
        for records, final in ((self.records, None), (self.records[:1], self.finalization)):
            with self.assertRaises(ValueError):
                report_module.build_synthetic_report(
                    self.inputs, self.profile_raw, self.profile_sha, records, final)
        records, final = copy.deepcopy((self.records, self.finalization))
        target = records[24]["measurement"]["target"]
        target["target_ids"][-1] = 12
        target["full_sequence_ids"] = [11] * 3 + target["target_ids"] + [13]
        rechain(records, final)
        with self.assertRaises(ValueError):
            self.report(records, final)
        actual = report_module.build_synthetic_report(
            self.inputs, self.profile_raw, self.profile_sha, self.records, None,
            failure={"phase": "FINALIZE", "ordinal": None}, abort_failed=True)
        self.assertEqual(actual["public"]["status"], "SYNTHETIC_FAILED")
        self.assertFalse(actual["public"]["synthetic_all_rows_within_budget"])
        self.assertTrue(actual["public"]["abort_failed"])

    def test_no_execution_imports(self):
        tree = ast.parse(Path(report_module.__file__).read_text())
        forbidden = {"tokenizer_session", "transformers", "torch", "subprocess", "socket",
                     "tokenizers", "huggingface_hub", "tokenizer_snapshot"}
        forbidden_calls = {"eval", "exec", "compile", "__import__", "open", "load",
                           "runtime_inventory", "inspect_runtime", "from_pretrained",
                           "validate_inventory", "validate_synthetic_inventory"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse(any(alias.name.split(".")[0] in forbidden for alias in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module.split(".")[0], forbidden)
                if node.module == "tokenizer_audit_report":
                    self.assertFalse(any(alias.name in ("_row", "_ORDER", "validate_record",
                                                        "build_report", "validate_inventory")
                                         for alias in node.names))
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, forbidden_calls)
                elif isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, forbidden_calls)

    def test_prepare_reconstructs_inventory_and_returns_detached_values(self):
        inputs = copy.deepcopy(self.inputs)
        inventory, profile = report_module.prepare_synthetic_inputs(
            inputs, self.profile_raw, self.profile_sha)
        self.assertEqual(inventory, self.inventory)
        self.assertEqual(profile, self.profile)
        inventory["rows"][0]["messages"][0]["content"] = "changed"
        inventory["run_ids"][0] = "changed"
        profile["versions"]["python"] = "changed"
        self.assertEqual(inputs, self.inputs)
        self.assertEqual(report_module.prepare_synthetic_inputs(
            self.inputs, self.profile_raw, self.profile_sha), (self.inventory, self.profile))

    def test_rejects_prebuilt_reports_and_nonexact_validation_inputs(self):
        variants = [self.inventory, {"status": "PASS"}, {"public": {}, "private": {}},
                    DictSubclass(self.inputs), list(self.inputs)]
        for name in self.inputs:
            changed = dict(self.inputs)
            del changed[name]
            variants.append(changed)
        variants.append({**self.inputs, "inventory": self.inventory})
        variants.append({StringSubclass(name): value for name, value in self.inputs.items()})
        for inputs in variants:
            with self.subTest(keys=list(inputs)), self.invalid():
                report_module.prepare_synthetic_inputs(inputs, self.profile_raw, self.profile_sha)

    def test_prepare_rejects_changed_frozen_dependencies(self):
        for path in (("worker_raw",), ("raw_files", "prompts.json"),
                     ("rendering_raw", "recalibration_rendering.py"),
                     ("rendering_raw", "prompt_rendering.py")):
            inputs = copy.deepcopy(self.inputs)
            replace(inputs, path, b"private invalid stored source")
            with self.subTest(path=path), self.invalid():
                report_module.prepare_synthetic_inputs(inputs, self.profile_raw, self.profile_sha)

    def test_invalid_profiles_are_rejected_without_runtime_inspection(self):
        for path, replacement in (
            (("kind",), "PRIVATE"), (("version",), True), (("version",), 1.0),
            (("run_id",), "b" * 31), (("revision",), "b" * 40),
            (("file_sha256", "tokenizer.json"), "0" * 64),
            (("versions",), {}), (("versions", "python"), "3.12.12"),
            (("versions", "transformers"), "5.0.0"), (("versions", "tokenizers"), True),
            (("versions", "jinja2"), ""), (("versions", "huggingface-hub"), "has space"),
            (("extra",), "private"),
        ):
            profile = copy.deepcopy(self.profile)
            replace(profile, path, replacement)
            with self.subTest(path=path), self.invalid():
                report_module.prepare_synthetic_inputs(self.inputs, encoded(profile), digest(profile))
        for raw, approved in (
            (self.profile_raw, "0" * 64), (self.profile_raw, StringSubclass(self.profile_sha)),
            (self.profile_raw.decode(), self.profile_sha),
            (b"\xff", hashlib.sha256(b"\xff").hexdigest()),
            (b'{"kind":1,"kind":2}', hashlib.sha256(b'{"kind":1,"kind":2}').hexdigest()),
            (b"NaN", hashlib.sha256(b"NaN").hexdigest()),
        ):
            with self.subTest(raw_type=type(raw), approved=approved), self.invalid():
                report_module.prepare_synthetic_inputs(self.inputs, raw, approved)
        with self.invalid():
            report_module.build_synthetic_report(
                self.inputs, self.profile_raw, "0" * 64, self.records, self.finalization)

    def test_dependency_failure_has_exact_planned_counts_and_rejects_all_evidence(self):
        inputs = copy.deepcopy(self.inputs)
        inputs["records"] = []
        result = report_module.build_synthetic_report(
            inputs, self.profile_raw, self.profile_sha, [], None)
        public = result["public"]
        self.assertEqual((public["planned"], public["checked"], public["not_checked"],
                          public["over_limit"]), (120, 0, 120, 0))
        self.assertEqual(public["by_category"], {
            name: {"scheduled": total, "checked": 0, "not_checked": total, "over_limit": 0}
            for name, total in CATEGORY_TOTALS.items()})
        self.assertFalse(public["session_finalized"])
        self.assertFalse(public["synthetic_all_rows_within_budget"])
        self.assertEqual(result["private"]["records"], [])
        self.assertIsNone(result["private"]["finalization"])
        for records, finalization, failure, abort_failed in (
            (self.records[:1], None, None, False),
            ([], self.finalization, None, False),
            ([], None, {"phase": "OPEN", "ordinal": None}, False),
            ([], None, None, True), ([], None, None, 0),
        ):
            with self.subTest(count=len(records), failure=failure, abort=abort_failed), self.invalid():
                report_module.build_synthetic_report(
                    inputs, self.profile_raw, self.profile_sha, records, finalization,
                    failure=failure, abort_failed=abort_failed)

    def test_public_whitelist_exact_identities_digests_and_counts(self):
        result = self.report()
        public = result["public"]
        self.assertEqual(set(result), {"public", "private"})
        self.assertEqual(set(result["private"]), {"inventory", "records", "finalization"})
        self.assertEqual(set(public), {
            "kind", "version", "data_version", "status", "synthetic_only", "evidence_kind",
            "run_id", "profile_sha256", "inventory_sha256", "validation_evidence_sha256",
            "planned", "checked", "not_checked", "over_limit", "rows", "by_category",
            "stop_reason", "stop_ordinal", "failure", "abort_failed", "session_finalized",
            "synthetic_all_rows_within_budget", *REAL_FLAGS})
        self.assertIs(public["synthetic_only"], True)
        self.assertEqual(public["evidence_kind"], "synthetic")
        self.assertEqual(public["data_version"], "recalibration_v2_r1")
        self.assertEqual(public["version"], 1)
        self.assertEqual(public["run_id"], self.profile["run_id"])
        self.assertEqual(public["profile_sha256"], self.profile_sha)
        self.assertEqual(public["inventory_sha256"], self.inventory["inventory_sha256"])
        self.assertEqual(public["validation_evidence_sha256"],
                         self.inventory["dynamic_evidence_sha256"])
        self.assertEqual(public["by_category"], {
            name: {"scheduled": total, "checked": total, "not_checked": 0, "over_limit": 0}
            for name, total in CATEGORY_TOTALS.items()})
        self.assertIsNone(public["failure"])
        self.assertIsNone(public["stop_reason"])
        self.assertIsNone(public["stop_ordinal"])
        self.assertIs(public["session_finalized"], True)
        for row, expected in zip(public["rows"], self.inventory["rows"]):
            self.assertEqual(set(row), {*IDENTITY_FIELDS, "status", "input_tokens",
                                        "target_tokens", "full_sequence_tokens"})
            self.assertEqual({name: row[name] for name in IDENTITY_FIELDS},
                             {name: expected[name] for name in IDENTITY_FIELDS})
            self.assertEqual(row["status"], "OK")
            self.assertEqual(row["input_tokens"], 3)
            self.assertEqual(row["target_tokens"], 2 if expected["answer"] is not None else None)
            self.assertEqual(row["full_sequence_tokens"],
                             6 if expected["answer"] is not None else None)
        for text in ("private synthetic", "3.12.13", "4.57.1", "[EOS]",
                     self.inventory["rows"][0]["messages"][0]["content"],
                     self.inventory["rows"][24]["answer"]):
            self.assertNotIn(text, encoded(public).decode())

    def test_category_edge_stops_preserve_all_identities_and_denominators(self):
        for count, stop in ((1, "INPUT_LIMIT"), (24, "INPUT_LIMIT"), (25, "OUTPUT_LIMIT"),
                            (72, "OUTPUT_LIMIT"), (73, "INPUT_LIMIT"), (120, "INPUT_LIMIT")):
            records, finalization = synthetic_transcript(
                self.inventory, self.profile, count=count, stop=stop)
            public = self.report(records, finalization)["public"]
            with self.subTest(count=count, stop=stop):
                self.assertEqual(public["status"], "SYNTHETIC_STOPPED")
                self.assertEqual((public["checked"], public["not_checked"],
                                  public["over_limit"]), (count, 120 - count, 1))
                self.assertEqual((public["stop_reason"], public["stop_ordinal"]), (stop, count - 1))
                self.assertIs(public["session_finalized"], True)
                self.assertIs(public["synthetic_all_rows_within_budget"], False)
                self.assertEqual(len(public["rows"]), 120)
                for row, planned in zip(public["rows"], self.inventory["rows"]):
                    self.assertEqual({name: row[name] for name in IDENTITY_FIELDS},
                                     {name: planned[name] for name in IDENTITY_FIELDS})
                    if row["ordinal"] >= count:
                        self.assertEqual(row["status"], "NOT_RUN_AFTER_STOP")
                        self.assertIsNone(row["input_tokens"])
                        self.assertIsNone(row["target_tokens"])
                for category, total in CATEGORY_TOTALS.items():
                    checked = sum(row["category"] == category
                                  for row in self.inventory["rows"][:count])
                    self.assertEqual(public["by_category"][category], {
                        "scheduled": total, "checked": checked, "not_checked": total - checked,
                        "over_limit": int(self.inventory["rows"][count - 1]["category"] == category)})
                self.assertTrue(all(public[flag] is False for flag in REAL_FLAGS))

    def test_exact_limits_on_same_reference_and_empty_template_tail(self):
        records, finalization = copy.deepcopy((self.records, self.finalization))
        measurement = records[24]["measurement"]
        measurement["input"].update(input_ids=[11] * 4096, input_tokens=4096,
                                    remaining_input_tokens=0)
        measurement["target"]["target_ids"] = [12] * 255 + [2]
        measurement["target"]["template_tail_ids"] = []
        repartition(measurement)
        rechain(records, finalization)
        public = self.report(records, finalization)["public"]
        self.assertTrue(public["synthetic_all_rows_within_budget"])
        self.assertEqual(public["rows"][24]["full_sequence_tokens"], 4352)

    def test_record_schema_types_counts_and_partition(self):
        changes = (
            (("kind",), "PRIVATE"), (("extra",), "private"),
            (("ordinal",), 24.0), (("ordinal",), IntegerSubclass(24)),
            (("identity_sha256",), "bad"),
            (("measurement", "kind"), "PRIVATE"),
            (("measurement", "state_sha256"), "bad"),
            (("measurement", "extra"), False),
            (("measurement", "input", "prompt"), "changed private prompt"),
            (("measurement", "input", "prompt"), StringSubclass("private synthetic prompt")),
            (("measurement", "input", "prompt_sha256"), "0" * 64),
            (("measurement", "input", "extra"), None),
            (("measurement", "input", "input_ids"), []),
            (("measurement", "input", "input_ids"), [True, 11, 11]),
            (("measurement", "input", "input_ids"), [11, -1, 11]),
            (("measurement", "input", "input_ids"), [11.0, 11, 11]),
            (("measurement", "input", "input_ids"), (11, 11, 11)),
            (("measurement", "input", "input_ids"), ListSubclass([11, 11, 11])),
            (("measurement", "input", "input_tokens"), 3.0),
            (("measurement", "input", "input_tokens"), 2),
            (("measurement", "input", "remaining_input_tokens"), 4093.0),
            (("measurement", "input", "remaining_input_tokens"), 4092),
            (("measurement", "target", "answer_sha256"), "0" * 64),
            (("measurement", "target", "full_prompt_sha256"), "BAD"),
            (("measurement", "target", "extra"), "private"),
            (("measurement", "target", "target_start"), 2),
            (("measurement", "target", "target_start"), True),
            (("measurement", "target", "target_tokens"), 1),
            (("measurement", "target", "target_tokens"), 2.0),
            (("measurement", "target", "target_ids"), []),
            (("measurement", "target", "target_ids"), [True, 2]),
            (("measurement", "target", "template_tail_ids"), [False]),
            (("measurement", "target", "full_sequence_ids"), []),
            (("measurement", "target", "full_sequence_ids"), [12, 11, 11, 12, 2, 13]),
            (("measurement", "target", "full_sequence_ids"), [11, 11, 11, 13, 12, 2]),
            (("measurement", "target", "full_sequence_tokens"), 5),
            (("measurement", "target", "full_sequence_tokens"), 6.0),
        )
        for path, value in changes:
            records, finalization = copy.deepcopy((self.records, self.finalization))
            replace(records[24], path, value)
            rechain(records, finalization)
            with self.subTest(path=path, value_type=type(value)), self.invalid():
                self.report(records, finalization)

    def test_target_presence_and_status_consistency(self):
        for ordinal, target in ((0, self.records[24]["measurement"]["target"]), (24, None)):
            records, finalization = copy.deepcopy((self.records, self.finalization))
            records[ordinal]["measurement"]["target"] = copy.deepcopy(target)
            rechain(records, finalization)
            with self.subTest(ordinal=ordinal), self.invalid():
                self.report(records, finalization)
        for count, stop in ((25, "INPUT_LIMIT"), (25, "OUTPUT_LIMIT")):
            records, finalization = synthetic_transcript(
                self.inventory, self.profile, count=count, stop=stop)
            records[-1]["measurement"]["status"] = "OK"
            rechain(records, finalization)
            with self.subTest(stop=stop), self.invalid():
                self.report(records, finalization)
        records, finalization = synthetic_transcript(
            self.inventory, self.profile, count=25, stop="INPUT_LIMIT")
        records[-1]["measurement"]["target"] = copy.deepcopy(self.records[24]["measurement"]["target"])
        rechain(records, finalization)
        with self.invalid():
            self.report(records, finalization)

    def test_duplicate_reordered_missing_and_excess_records(self):
        for mode in ("duplicate", "reordered", "missing", "excess"):
            records, finalization = copy.deepcopy((self.records, self.finalization))
            if mode == "duplicate":
                records[1] = copy.deepcopy(records[0])
            elif mode == "reordered":
                records[0], records[1] = records[1], records[0]
            elif mode == "missing":
                del records[1]
            else:
                records.append(copy.deepcopy(records[-1]))
            finalization["measurements"] = len(records)
            rechain(records, finalization)
            with self.subTest(mode=mode), self.invalid():
                self.report(records, finalization)

    def test_every_chain_link_is_verified(self):
        for ordinal in range(120):
            records = copy.deepcopy(self.records)
            records[ordinal]["chain_sha256"] = "0" * 64
            with self.subTest(ordinal=ordinal), self.invalid():
                self.report(records)

    def test_identity_and_state_continuity_before_failed_finalization(self):
        for field in ("identity", "state"):
            records = copy.deepcopy(self.records[:2])
            if field == "identity":
                records[1]["identity_sha256"] = "0" * 64
            else:
                records[1]["measurement"]["state_sha256"] = "0" * 64
            rechain(records)
            with self.subTest(field=field), self.invalid():
                self.failed_report(records, "MEASURE", 2)

    def test_validated_record_is_detached_and_binds_internal_arguments(self):
        row = self.inventory["rows"][1]
        record = self.records[1]
        arguments = (row, record, self.profile["run_id"], self.profile_sha,
                     self.records[0]["chain_sha256"])
        identity = record["identity_sha256"]
        state = record["measurement"]["state_sha256"]
        actual = report_module._validate_record(*arguments, identity, state)
        self.assertEqual(actual, record)
        actual["measurement"]["input"]["input_ids"].clear()
        self.assertEqual(record["measurement"]["input"]["input_ids"], [11, 11, 11])
        for changed_identity, changed_state in (("0" * 64, state), (identity, "0" * 64)):
            with self.subTest(identity=changed_identity, state=changed_state), self.invalid():
                report_module._validate_record(*arguments, changed_identity, changed_state)

    def test_continuity_arguments_require_builtin_digest_strings(self):
        record = self.records[1]
        arguments = (self.inventory["rows"][1], record, self.profile["run_id"],
                     self.profile_sha, self.records[0]["chain_sha256"])
        identity = record["identity_sha256"]
        state = record["measurement"]["state_sha256"]
        for changed_identity, changed_state in (
            (StringSubclass(identity), state), (identity, StringSubclass(state)),
        ):
            with self.subTest(identity_type=type(changed_identity),
                              state_type=type(changed_state)), self.invalid():
                report_module._validate_record(*arguments, changed_identity, changed_state)

    def test_finalization_schema_bindings_flags_runtime_and_snapshot(self):
        changes = [
            (("kind",), "PRIVATE"), (("extra",), "private"), (("run_id",), "b" * 32),
            (("profile_sha256",), "0" * 64), (("measurements",), 119),
            (("measurements",), True), (("measurements",), 120.0),
            (("chain_sha256",), "0" * 64), (("stop_reason",), "INPUT_LIMIT"),
            (("tokenizer_loaded",), False),
            (("runtime", "kind"), "PRIVATE"), (("runtime", "run_id"), "b" * 32),
            (("runtime", "profile_sha256"), "0" * 64),
            (("runtime", "versions", "python"), "3.12.12"),
            (("runtime", "versions", "extra"), "fixture"),
            (("runtime", "tokenizer_loaded"), True), (("runtime", "extra"), "private"),
            (("snapshot", "kind"), "PRIVATE"), (("snapshot", "run_id"), "b" * 32),
            (("snapshot", "profile_sha256"), "0" * 64),
            (("snapshot", "revision_declaration"), "0" * 40),
            (("snapshot", "file_sha256", "tokenizer.json"), "0" * 64),
            (("snapshot", "file_bytes"), {}),
            (("snapshot", "file_bytes", "config.json"), True),
            (("snapshot", "file_bytes", "config.json"), -1),
            (("snapshot", "file_bytes", "config.json"), 128 * 1024 * 1024),
            (("snapshot", "tokenizer_loaded"), True), (("snapshot", "extra"), "private"),
        ]
        changes.extend(((*prefix, flag), value)
                       for prefix in ((), ("runtime",), ("snapshot",))
                       for flag in (*FLAGS, "all_possible_second_rounds_validated")
                       for value in (True, 0))
        for path, value in changes:
            finalization = copy.deepcopy(self.finalization)
            replace(finalization, path, value)
            with self.subTest(path=path, value=value), self.invalid():
                self.report(self.records, finalization)
        for name in self.finalization:
            finalization = copy.deepcopy(self.finalization)
            del finalization[name]
            with self.subTest(missing=name), self.invalid():
                self.report(self.records, finalization)

    def test_finalization_identity_structure_and_rebound_state(self):
        changes = (
            (("kind",), "PRIVATE"), (("class",), ""), (("is_fast",), 1),
            (("chat_template",), ""), (("template_sha256",), "0" * 64),
            (("vocabulary_sha256",), "bad"), (("backend_sha256",), "bad"),
            (("eos_token_id",), True), (("pad_token_id",), -1),
            (("measurement_state_sha256",), "0" * 64),
            (("special_tokens_map", "eos_token"), []),
            (("special_token_ids", "[EOS]"), 3),
            (("settings", "padding_side"), "middle"),
            (("settings", "model_max_length"), 0),
            (("settings", "split_special_tokens"), 0),
            (("settings", "extra"), "private"),
        )
        for path, value in changes:
            records, finalization = copy.deepcopy((self.records, self.finalization))
            replace(finalization["identity"], path, value)
            for record in records:
                record["identity_sha256"] = digest(finalization["identity"])
                record["measurement"]["state_sha256"] = finalization["identity"]["measurement_state_sha256"]
            rechain(records, finalization)
            with self.subTest(path=path), self.invalid():
                self.report(records, finalization)
        for path in (("identity_sha256",), ("measurement", "state_sha256")):
            records, finalization = copy.deepcopy((self.records, self.finalization))
            for record in records:
                replace(record, path, "0" * 64)
            rechain(records, finalization)
            with self.subTest(binding=path), self.invalid():
                self.report(records, finalization)

    def test_eos_unique_terminal_and_absent_from_template_tail(self):
        for target_ids, tail in (([12, 12], [13]), ([2, 12], [13]), ([2, 2], [13]),
                                 ([12, 2], [2]), ([12, 2, 12], [13])):
            records, finalization = copy.deepcopy((self.records, self.finalization))
            measurement = records[24]["measurement"]
            measurement["target"]["target_ids"] = target_ids
            measurement["target"]["template_tail_ids"] = tail
            repartition(measurement)
            rechain(records, finalization)
            with self.subTest(target_ids=target_ids, tail=tail), self.invalid():
                self.report(records, finalization)

    def test_strict_nested_builtin_containers_keys_and_non_json_values(self):
        records, finalization = copy.deepcopy((self.records, self.finalization))
        for records_value in (tuple(records), ListSubclass(records), None, {}, records * 2):
            with self.subTest(container=type(records_value)), self.invalid():
                report_module.build_synthetic_report(
                    self.inputs, self.profile_raw, self.profile_sha, records_value, finalization)
        for path, value in (
            (("measurement",), DictSubclass(records[0]["measurement"])),
            (("measurement", "input", "prompt"), b"private"),
            (("measurement", "input", "input_tokens"), float("nan")),
            (("measurement", "input", "input_tokens"), float("inf")),
            (("measurement", "input"), {
                StringSubclass(name): value for name, value in records[0]["measurement"]["input"].items()}),
        ):
            changed = copy.deepcopy(records)
            replace(changed[0], path, value)
            with self.subTest(path=path, value_type=type(value)), self.invalid():
                self.report(changed, finalization)
        for path, value in (
            (("identity", "class"), StringSubclass("synthetic.Tokenizer")),
            (("identity", "settings"), DictSubclass(finalization["identity"]["settings"])),
            (("runtime", "versions", "python"), StringSubclass("3.12.13")),
            (("snapshot", "file_bytes", "config.json"), IntegerSubclass(1)),
        ):
            changed = copy.deepcopy(finalization)
            replace(changed, path, value)
            with self.subTest(path=path), self.invalid():
                self.report(records, changed)

    def test_missing_and_premature_finalization_are_rejected(self):
        for count in (0, 1, 24, 72, 119):
            records, finalization = synthetic_transcript(self.inventory, self.profile, count=count)
            with self.subTest(count=count), self.invalid():
                self.report(records, finalization)
        for count, stop in ((120, None), (1, "INPUT_LIMIT"), (25, "OUTPUT_LIMIT")):
            records, unused_finalization = synthetic_transcript(
                self.inventory, self.profile, count=count, stop=stop)
            with self.subTest(count=count, stop=stop), self.invalid():
                report_module.build_synthetic_report(
                    self.inputs, self.profile_raw, self.profile_sha, records, None)

    def test_all_legal_failure_phases_and_abort_results(self):
        for phase, count, reason in (
            ("OPEN", 0, "OPEN_FAILED"), ("MEASURE", 0, "MEASUREMENT_FAILED"),
            ("MEASURE", 24, "MEASUREMENT_FAILED"), ("MEASURE", 119, "MEASUREMENT_FAILED"),
            ("RECEIPT", 0, "RECEIPT_INVALID"), ("RECEIPT", 72, "RECEIPT_INVALID"),
            ("RECEIPT", 119, "RECEIPT_INVALID"), ("FINALIZE", 120, "FINALIZATION_FAILED"),
        ):
            ordinal = count if phase in ("MEASURE", "RECEIPT") else None
            for abort_failed in ((False,) if phase == "OPEN" else (False, True)):
                with self.subTest(phase=phase, count=count, abort=abort_failed):
                    result = self.failed_report(self.records[:count], phase, ordinal, abort_failed)
                    public = result["public"]
                    self.assertEqual(public["status"], "SYNTHETIC_FAILED")
                    self.assertEqual(public["failure"],
                                     {"phase": phase, "ordinal": ordinal, "reason": reason})
                    self.assertEqual((public["stop_reason"], public["stop_ordinal"]), (reason, ordinal))
                    self.assertEqual((public["checked"], public["not_checked"]), (count, 120 - count))
                    self.assertEqual(len(public["rows"]), 120)
                    self.assertIs(public["abort_failed"], abort_failed)
                    self.assertIs(public["session_finalized"], False)
                    self.assertIs(public["synthetic_all_rows_within_budget"], False)
                    self.assertTrue(all(public[flag] is False for flag in REAL_FLAGS))
                    self.assertEqual(result["private"]["records"], self.records[:count])
                    self.assertIsNone(result["private"]["finalization"])
                    if ordinal is not None:
                        self.assertEqual(public["rows"][ordinal]["status"], reason)
                        self.assertIsNone(public["rows"][ordinal]["input_tokens"])
                    for category, total in CATEGORY_TOTALS.items():
                        checked = sum(row["category"] == category
                                      for row in self.inventory["rows"][:count])
                        self.assertEqual(public["by_category"][category], {
                            "scheduled": total, "checked": checked,
                            "not_checked": total - checked, "over_limit": 0})

    def test_finalize_failure_preserves_normal_over_limit_stop(self):
        for count, stop in ((1, "INPUT_LIMIT"), (25, "OUTPUT_LIMIT"), (120, "INPUT_LIMIT")):
            records, unused_finalization = synthetic_transcript(
                self.inventory, self.profile, count=count, stop=stop)
            for abort_failed in (False, True):
                public = self.failed_report(records, "FINALIZE", None, abort_failed)["public"]
                with self.subTest(count=count, stop=stop, abort=abort_failed):
                    self.assertEqual(public["status"], "SYNTHETIC_FAILED")
                    self.assertEqual((public["stop_reason"], public["stop_ordinal"]), (stop, count - 1))
                    self.assertEqual(public["failure"]["reason"], "FINALIZATION_FAILED")
                    self.assertEqual(public["over_limit"], 1)
                    self.assertFalse(public["synthetic_all_rows_within_budget"])

    def test_illegal_failure_phases_ordinals_finalization_and_abort(self):
        cases = [
            ([], None, {"phase": "OPEN", "ordinal": 0}, False),
            (self.records[:1], None, {"phase": "OPEN", "ordinal": None}, False),
            ([], None, {"phase": "OPEN", "ordinal": None}, True),
            ([], None, {"phase": "FINALIZE", "ordinal": None}, False),
            (self.records[:119], None, {"phase": "FINALIZE", "ordinal": None}, False),
            (self.records, None, {"phase": "FINALIZE", "ordinal": 120}, False),
            ([], None, {"phase": "private", "ordinal": None}, False),
            ([], None, {"phase": [], "ordinal": None}, False),
            ([], None, {"phase": "OPEN"}, False),
            ([], None, {"phase": "OPEN", "ordinal": None, "message": "private"}, False),
            ([], None, {"phase": StringSubclass("OPEN"), "ordinal": None}, False),
            ([], None, DictSubclass(phase="OPEN", ordinal=None), False),
        ]
        for phase in ("MEASURE", "RECEIPT"):
            for ordinal in (None, True, 1.0, -1, 0, 2):
                cases.append((self.records[:1], None, {"phase": phase, "ordinal": ordinal}, False))
            cases.append((self.records, None, {"phase": phase, "ordinal": 120}, False))
            stopped, finalization = synthetic_transcript(
                self.inventory, self.profile, count=1, stop="INPUT_LIMIT")
            cases.append((stopped, None, {"phase": phase, "ordinal": 1}, False))
        for phase, count in (("OPEN", 0), ("MEASURE", 1), ("RECEIPT", 1), ("FINALIZE", 120)):
            ordinal = count if phase in ("MEASURE", "RECEIPT") else None
            cases.append((self.records[:count], self.finalization,
                          {"phase": phase, "ordinal": ordinal}, False))
        for abort_failed in (0, 1, None, "private", True):
            cases.append((self.records, self.finalization, None, abort_failed))
        for records, finalization, failure, abort_failed in cases:
            with self.subTest(count=len(records), failure=failure, abort=abort_failed), self.invalid():
                report_module.build_synthetic_report(
                    self.inputs, self.profile_raw, self.profile_sha, records, finalization,
                    failure=failure, abort_failed=abort_failed)

    def test_exceptions_are_bounded_without_private_text(self):
        records = copy.deepcopy(self.records)
        records[0]["measurement"]["input"]["prompt"] = "\ud800PRIVATE_SENTINEL"
        for arguments, options in (
            ((self.inputs, self.profile_raw, self.profile_sha, records, self.finalization), {}),
            ((self.inputs, b"PRIVATE_SENTINEL", self.profile_sha, [], None), {}),
            ((self.inputs, self.profile_raw, self.profile_sha, [], None),
             {"failure": {"phase": "PRIVATE_SENTINEL", "ordinal": None}}),
        ):
            with self.subTest(options=options):
                try:
                    report_module.build_synthetic_report(*arguments, **options)
                except ValueError as error:
                    self.assertEqual(str(error), "tokenizer audit evidence invalid")
                    self.assertNotIn("PRIVATE_SENTINEL", "".join(traceback.format_exception(
                        type(error), error, error.__traceback__)))
                    self.assertTrue(error.__suppress_context__)
                else:
                    self.fail("invalid private evidence accepted")

    def test_returned_public_private_and_caller_values_are_detached(self):
        inputs, records, finalization = copy.deepcopy((self.inputs, self.records, self.finalization))
        result = report_module.build_synthetic_report(
            inputs, self.profile_raw, self.profile_sha, records, finalization)
        expected = copy.deepcopy(result)
        inputs["run_ids"][0] = "changed"
        records[0]["measurement"]["input"]["input_ids"].clear()
        finalization["identity"]["settings"].clear()
        self.assertEqual(result, expected)
        result["private"]["inventory"]["rows"][0]["messages"].clear()
        result["private"]["inventory"]["rows"][0]["row_sha256"] = "changed"
        result["private"]["records"][0]["measurement"]["input"]["input_ids"].clear()
        result["private"]["finalization"]["identity"]["settings"].clear()
        self.assertEqual(result["public"], expected["public"])
        result["public"]["rows"][1]["row_sha256"] = "changed"
        self.assertEqual(result["private"]["inventory"]["rows"][1],
                         expected["private"]["inventory"]["rows"][1])
        self.assertEqual(self.report(), expected)


if __name__ == "__main__":
    unittest.main()
