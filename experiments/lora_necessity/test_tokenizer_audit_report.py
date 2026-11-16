import builtins
import copy
import hashlib
import importlib
from pathlib import Path
import unittest
from unittest.mock import patch

from pilot_protocol import FLAGS
from tokenizer_inventory import build_inventory
from tokenizer_runtime_profile import FILE_PINS, REVISION
from validation_contract import digest, encoded


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def frozen_inventory():
    directory = Path(__file__).with_name("data")
    return build_inventory(*((directory / name).read_bytes() for name in (
        "task_sources_v1.json", "manifest_v1.json",
        "supervision_v1.json", "prompts_v1.json",
    )))


def synthetic_profile():
    return {
        "kind": "TOKENIZER_RUNTIME_PROFILE", "version": 1, "run_id": "a" * 32,
        "revision": REVISION, "file_sha256": dict(FILE_PINS),
        "versions": {"python": "3.12.13", "transformers": "4.57.1",
                     "tokenizers": "fixture", "huggingface-hub": "fixture",
                     "jinja2": "fixture"},
    }


def synthetic_identity():
    state = {
        "template_sha256": text_hash("private synthetic template"),
        "vocabulary_sha256": digest({"[PAD]": 0, "[EOS]": 2, "input": 11,
                                     "answer": 12, "tail": 13}),
        "eos_token_id": 2, "pad_token_id": 0,
    }
    return {
        "kind": "TOKENIZER_OBSERVED_STATE_UNATTESTED",
        "class": "synthetic.Tokenizer", "is_fast": True,
        "chat_template": "private synthetic template", **state,
        "measurement_state_sha256": digest(state),
        "special_tokens_map": {"eos_token": "[EOS]", "pad_token": "[PAD]"},
        "special_token_ids": {"[EOS]": 2, "[PAD]": 0},
        "backend_sha256": text_hash("private synthetic backend"),
        "settings": {"padding_side": "right", "truncation_side": "right",
                     "model_max_length": 4096, "split_special_tokens": False},
    }


def rechain(records, finalization=None):
    previous = digest([])
    for record in records:
        unsigned = {name: value for name, value in record.items() if name != "chain_sha256"}
        record["chain_sha256"] = digest({"previous": previous, "record": unsigned})
        previous = record["chain_sha256"]
    if finalization is not None:
        finalization["chain_sha256"] = previous


def synthetic_transcript(inventory, profile, count=324, stop=None):
    identity = synthetic_identity()
    approved = digest(profile)
    flags = {name: False for name in (*FLAGS, "all_possible_second_rounds_validated")}
    records = []
    for row in inventory["rows"][:count]:
        status = stop if row["ordinal"] == count - 1 and stop else "OK"
        input_ids = [11] * (4097 if status == "INPUT_LIMIT" else 3)
        prompt = "private synthetic prompt"
        target = None
        if row["answer"] is not None and status != "INPUT_LIMIT":
            target_ids = [12] * (256 if status == "OUTPUT_LIMIT" else 1) + [2]
            full_ids = input_ids + target_ids + [13]
            target = {
                "answer_sha256": text_hash(row["answer"]),
                "full_prompt_sha256": text_hash(prompt + row["answer"] + "[EOS]"),
                "full_sequence_ids": full_ids, "full_sequence_tokens": len(full_ids),
                "target_start": len(input_ids), "target_ids": target_ids,
                "target_tokens": len(target_ids), "template_tail_ids": [13],
            }
        records.append({
            "kind": "TOKENIZER_SESSION_MEASUREMENT_PROVISIONAL",
            "run_id": profile["run_id"], "profile_sha256": approved,
            "identity_sha256": digest(identity), "ordinal": row["ordinal"],
            "measurement": {
                "kind": "TOKENIZER_MEASUREMENT_UNATTESTED",
                "state_sha256": identity["measurement_state_sha256"], "status": status,
                "input": {"messages_sha256": digest(row["messages"]), "prompt": prompt,
                          "prompt_sha256": text_hash(prompt), "input_ids": input_ids,
                          "input_tokens": len(input_ids),
                          "remaining_input_tokens": 4096 - len(input_ids)},
                "target": target,
            },
        })
    finalization = {
        "kind": "TOKENIZER_SESSION_FINALIZATION", "run_id": profile["run_id"],
        "profile_sha256": approved, "identity": identity,
        "runtime": {
            "kind": "TOKENIZER_RUNTIME_METADATA_ONLY", "run_id": profile["run_id"],
            "profile_sha256": approved, "versions": copy.deepcopy(profile["versions"]),
            "tokenizer_loaded": False, **flags,
        },
        "snapshot": {
            "kind": "TOKENIZER_FILE_SNAPSHOT_ONLY", "run_id": profile["run_id"],
            "profile_sha256": approved, "revision_declaration": profile["revision"],
            "file_sha256": dict(FILE_PINS), "file_bytes": {name: 1 for name in FILE_PINS},
            "tokenizer_loaded": False, **flags,
        },
        "measurements": count, "stop_reason": stop, "tokenizer_loaded": True, **flags,
    }
    rechain(records, finalization)
    return records, finalization


def replace(value, path, replacement):
    for name in path[:-1]:
        value = value[name]
    value[path[-1]] = replacement


class TokenizerAuditReportTests(unittest.TestCase):
    def test_open_diagnostic_is_strict_and_copied(self):
        diagnostic = {
            "stage": "TOKENIZER_LOAD", "error_type": "ImportError", "guard_failed": True}
        result = self.report([], None, failure={"phase": "OPEN", "ordinal": None},
                             open_diagnostic=diagnostic)
        self.assertEqual(result["public"]["open_diagnostic"], diagnostic)
        diagnostic["stage"] = "PRIVATE"
        self.assertEqual(result["public"]["open_diagnostic"]["stage"], "TOKENIZER_LOAD")
        for changed in (
            {"stage": "PRIVATE"}, {"error_type": "PRIVATE"}, {"guard_failed": 1},
            {"message": "PRIVATE"}, {"stage": None},
        ):
            candidate = {"stage": "TOKENIZER_LOAD", "error_type": "OTHER",
                         "guard_failed": False, **changed}
            with self.subTest(keys=tuple(changed)), self.assertRaises(ValueError):
                self.report([], None, failure={"phase": "OPEN", "ordinal": None},
                            open_diagnostic=candidate)

    def test_diagnostic_rejected_outside_open_failure(self):
        diagnostic = {
            "stage": "TOKENIZER_LOAD", "error_type": "OTHER", "guard_failed": False}
        with self.assertRaises(ValueError):
            self.report([], None, failure={"phase": "MEASURE", "ordinal": 0},
                        open_diagnostic=diagnostic)

    @classmethod
    def setUpClass(cls):
        cls.inventory = frozen_inventory()
        cls.inventory_raw = encoded(cls.inventory)
        cls.inventory_sha256 = digest(cls.inventory)
        cls.profile = synthetic_profile()
        cls.profile_raw = encoded(cls.profile)
        cls.profile_sha256 = digest(cls.profile)

    def setUp(self):
        self.assertTrue(Path(__file__).with_name("tokenizer_audit_report.py").is_file(),
                        "tokenizer audit report not implemented")
        self.module = importlib.import_module("tokenizer_audit_report")

    def invalid(self):
        return self.assertRaisesRegex(ValueError, "^tokenizer audit evidence invalid$")

    def report(self, records, finalization, **options):
        return self.module.build_report(
            self.inventory_raw, self.inventory_sha256, self.profile_raw,
            self.profile_sha256, records, finalization, **options)

    def validate_record(self, record, ordinal=0, previous=None, identity=None):
        return self.module.validate_record(
            self.inventory["rows"][ordinal], record, self.profile["run_id"],
            self.profile_sha256, digest([]) if previous is None else previous, identity)

    def test_complete_inventory(self):
        actual = self.module.validate_inventory(self.inventory_raw, self.inventory_sha256)
        self.assertEqual(actual, self.inventory)
        self.assertEqual(len(actual["rows"]), 324)

    def test_inventory_raw_approval_canonical_and_duplicate_keys(self):
        for raw, approved in (
            (self.inventory_raw, "0" * 64),
            (self.inventory_raw + b"\n", self.inventory_sha256),
            (self.inventory_raw + b"\n", hashlib.sha256(self.inventory_raw + b"\n").hexdigest()),
            (b'{"kind":1,"kind":2}', text_hash('{"kind":1,"kind":2}')),
            (b"\xff", hashlib.sha256(b"\xff").hexdigest()),
            (b"NaN", text_hash("NaN")),
            (self.inventory_raw.decode(), self.inventory_sha256),
            (self.inventory_raw, self.inventory_sha256.upper()),
        ):
            with self.subTest(raw_type=type(raw), approved=approved), self.invalid():
                self.module.validate_inventory(raw, approved)

    def test_inventory_strict_structure_and_pins(self):
        changes = [
            (("kind",), "OTHER"), (("version",), True), (("version",), 1.0),
            (("extra",), "private"), (("rows",), []),
            (("references_sha256",), "0" * 64),
            (("data_sha256", "prompts_v1.json"), "0" * 64),
        ]
        changes.extend(((name,), value) for name in (*FLAGS, "all_possible_second_rounds_validated")
                       for value in (True, 0, None))
        for path, value in changes:
            changed = copy.deepcopy(self.inventory)
            replace(changed, path, value)
            with self.subTest(path=path, value=value), self.invalid():
                self.module.validate_inventory(encoded(changed), digest(changed))

    def test_inventory_rows_category_shape_and_primitive_safety(self):
        changes = [
            (0, ("ordinal",), True), (0, ("ordinal",), 1), (0, ("category",), "first"),
            (0, ("task_id",), "private\nleak"), (0, ("task_id",), {}),
            (0, ("arm",), "base"), (0, ("answer",), None), (0, ("answer",), ""),
            (0, ("sample",), {}), (0, ("messages",), []),
            (0, ("messages", 0, "role"), "system"),
            (0, ("messages", 0, "content"), 1), (0, ("extra",), "private"),
            (8, ("arm",), None), (8, ("arm",), []), (8, ("answer",), "extra"),
            (188, ("sample",), None), (188, ("sample", "extra"), "private"),
            (188, ("sample", "kind"), "OTHER"),
            (188, ("sample", "candidate_sha256"), "BAD"),
            (188, ("sample", "feedback_sha256"), False),
            (188, ("messages",), [{"role": "user", "content": "short"}]),
        ]
        for ordinal, path, value in changes:
            changed = copy.deepcopy(self.inventory)
            row = changed["rows"][ordinal]
            replace(row, path, value)
            row["row_sha256"] = digest({name: value for name, value in row.items()
                                        if name != "row_sha256"})
            with self.subTest(ordinal=ordinal, path=path), self.invalid():
                self.module.validate_inventory(encoded(changed), digest(changed))

    def test_inventory_reference_projection_row_hash_and_order(self):
        for mode in ("reference", "row_hash", "order", "duplicate"):
            changed = copy.deepcopy(self.inventory)
            if mode == "reference":
                row = changed["rows"][144]
                row["answer"] += "\n"
                row["row_sha256"] = digest({name: value for name, value in row.items()
                                            if name != "row_sha256"})
            elif mode == "row_hash":
                changed["rows"][0]["messages"][0]["content"] += "changed"
            elif mode == "order":
                changed["rows"][0], changed["rows"][1] = changed["rows"][1], changed["rows"][0]
            else:
                changed["rows"][1] = copy.deepcopy(changed["rows"][0])
            with self.subTest(mode=mode), self.invalid():
                self.module.validate_inventory(encoded(changed), digest(changed))

    def test_inventory_independent_copies(self):
        first = self.module.validate_inventory(self.inventory_raw, self.inventory_sha256)
        second = self.module.validate_inventory(self.inventory_raw, self.inventory_sha256)
        first["rows"][0]["messages"].clear()
        first["data_sha256"].clear()
        self.assertEqual(second, self.inventory)

    def test_record_success_and_independent_copy(self):
        records, finalization = synthetic_transcript(self.inventory, self.profile, count=2)
        actual = self.validate_record(records[0])
        self.assertEqual(actual, records[0])
        self.assertEqual(self.validate_record(
            records[1], 1, records[0]["chain_sha256"], digest(finalization["identity"])),
            records[1])
        actual["measurement"]["input"]["input_ids"].clear()
        self.assertEqual(records[0]["measurement"]["input"]["input_ids"], [11] * 3)

    def test_record_strict_fields_bindings_and_chain(self):
        changes = [
            (("kind",), "TOKENIZER_MEASUREMENT_UNATTESTED"),
            (("extra",), "private"), (("ordinal",), False), (("ordinal",), 1),
            (("run_id",), "b" * 32), (("profile_sha256",), "0" * 64),
            (("identity_sha256",), "bad"), (("chain_sha256",), "0" * 64),
            (("measurement", "kind"), "SIMULATED"),
            (("measurement", "extra"), "private"),
            (("measurement", "state_sha256"), "bad"),
            (("measurement", "status"), "PASS"),
            (("measurement", "input", "messages_sha256"), "0" * 64),
            (("measurement", "input", "prompt"), "changed"),
            (("measurement", "input", "prompt_sha256"), "0" * 64),
            (("measurement", "input", "extra"), "private"),
        ]
        for path, value in changes:
            records, _ = synthetic_transcript(self.inventory, self.profile, count=1)
            replace(records[0], path, value)
            if path != ("chain_sha256",):
                rechain(records)
            with self.subTest(path=path), self.invalid():
                self.validate_record(records[0])
        records, _ = synthetic_transcript(self.inventory, self.profile, count=1)
        for previous, identity in (("1" * 64, None), (digest([]), "1" * 64)):
            with self.subTest(previous=previous, identity=identity), self.invalid():
                self.validate_record(records[0], previous=previous, identity=identity)

    def test_record_integer_lists_and_arithmetic(self):
        changes = [
            (("input", "input_ids"), []), (("input", "input_ids"), [True, 11, 11]),
            (("input", "input_ids"), [11, -1, 11]),
            (("input", "input_ids"), [11.0, 11, 11]),
            (("input", "input_ids"), (11, 11, 11)),
            (("input", "input_tokens"), 3.0), (("input", "input_tokens"), 2),
            (("input", "remaining_input_tokens"), 4092),
            (("input", "remaining_input_tokens"), 4093.0),
            (("target", "answer_sha256"), "0" * 64),
            (("target", "full_prompt_sha256"), "bad"),
            (("target", "target_start"), 2), (("target", "target_start"), 3.0),
            (("target", "target_ids"), []), (("target", "target_ids"), [True, 2]),
            (("target", "target_tokens"), 1), (("target", "target_tokens"), 2.0),
            (("target", "template_tail_ids"), [False]),
            (("target", "full_sequence_ids"), []),
            (("target", "full_sequence_ids"), [12, 11, 11, 12, 2, 13]),
            (("target", "full_sequence_tokens"), 5),
            (("target", "full_sequence_tokens"), 6.0),
            (("target", "extra"), "private"),
        ]
        for path, value in changes:
            records, _ = synthetic_transcript(self.inventory, self.profile, count=1)
            replace(records[0]["measurement"], path, value)
            rechain(records)
            with self.subTest(path=path, value=value), self.invalid():
                self.validate_record(records[0])

    def test_record_status_target_consistency(self):
        for count, stop in ((1, None), (1, "INPUT_LIMIT"), (1, "OUTPUT_LIMIT"), (9, None)):
            records, _ = synthetic_transcript(self.inventory, self.profile, count, stop)
            ordinal = count - 1
            previous = digest([]) if count == 1 else records[-2]["chain_sha256"]
            self.assertEqual(self.validate_record(records[-1], ordinal, previous), records[-1])
            changed = copy.deepcopy(records)
            changed[-1]["measurement"]["status"] = "OK" if stop else "INPUT_LIMIT"
            rechain(changed)
            with self.subTest(count=count, stop=stop), self.invalid():
                self.validate_record(changed[-1], ordinal, previous)
        records, _ = synthetic_transcript(self.inventory, self.profile, 9)
        records[-1]["measurement"]["target"] = copy.deepcopy(records[0]["measurement"]["target"])
        rechain(records)
        with self.invalid():
            self.validate_record(records[-1], 8, records[-2]["chain_sha256"])

    def test_exact_input_and_target_limits_and_empty_tail(self):
        records, finalization = synthetic_transcript(self.inventory, self.profile)
        measurement = records[0]["measurement"]
        measurement["input"].update(input_ids=[11] * 4096, input_tokens=4096,
                                    remaining_input_tokens=0)
        measurement["target"].update(
            target_ids=[12] * 255 + [2], target_tokens=256, target_start=4096,
            template_tail_ids=[], full_sequence_ids=[11] * 4096 + [12] * 255 + [2],
            full_sequence_tokens=4352)
        rechain(records, finalization)
        public = self.report(records, finalization)["public"]
        self.assertTrue(public["all_rows_within_budget"])
        self.assertEqual(public["rows"][0]["target_tokens"], 256)
        self.assertEqual(public["rows"][0]["full_sequence_tokens"], 4352)

    def test_target_presence_matches_answer_and_input_limit(self):
        baseline, _ = synthetic_transcript(self.inventory, self.profile, 1)
        for stop in (None, "INPUT_LIMIT"):
            records, _ = synthetic_transcript(self.inventory, self.profile, 1, stop)
            records[0]["measurement"]["target"] = (
                copy.deepcopy(baseline[0]["measurement"]["target"]) if stop else None)
            rechain(records)
            with self.subTest(stop=stop), self.invalid():
                self.validate_record(records[0])

    def test_receipt_scalar_subclasses_are_not_primitive_evidence(self):
        class PrivateString(str):
            pass

        for path in (("kind",), ("run_id",), ("profile_sha256",),
                     ("measurement", "kind"), ("measurement", "status"),
                     ("measurement", "input", "messages_sha256")):
            records, _ = synthetic_transcript(self.inventory, self.profile, 1)
            value = records[0]
            for name in path:
                value = value[name]
            replace(records[0], path, PrivateString(value))
            rechain(records)
            with self.subTest(path=path), self.invalid():
                self.validate_record(records[0])

    def test_finalization_scalar_subclasses_are_not_primitive_evidence(self):
        class PrivateString(str):
            pass

        records, finalization = synthetic_transcript(self.inventory, self.profile)
        finalization["runtime"]["versions"]["jinja2"] = PrivateString("fixture")
        with self.invalid():
            self.report(records, finalization)

    def test_complete_report_pending_cleanup_and_public_whitelist(self):
        records, finalization = synthetic_transcript(self.inventory, self.profile)
        report = self.report(records, finalization)
        self.assertEqual(set(report), {"public", "private"})
        public = report["public"]
        self.assertEqual(set(public), {
            "kind", "run_id", "profile_sha256", "inventory_sha256", "rows", "by_category",
            "stop_reason", "stop_ordinal", "failure", "abort_failed", "session_finalized",
            "all_rows_within_budget", "checked_inventory_within_budget", "cleanup_status",
            *FLAGS, "all_possible_second_rounds_validated",
        })
        self.assertEqual(public["kind"], "LORA_TOKENIZER_AUDIT_PENDING_CLEANUP")
        self.assertTrue(public["all_rows_within_budget"])
        self.assertTrue(public["session_finalized"])
        self.assertIsNone(public["failure"])
        self.assertIsNone(public["stop_reason"])
        self.assertIsNone(public["stop_ordinal"])
        self.assertEqual(public["run_id"], self.profile["run_id"])
        self.assertEqual(public["profile_sha256"], self.profile_sha256)
        self.assertEqual(public["inventory_sha256"], self.inventory_sha256)
        self.assertFalse(public["abort_failed"])
        self.assertFalse(public["checked_inventory_within_budget"])
        self.assertEqual(public["cleanup_status"], "CLEANUP_UNVERIFIED")
        for name in (*FLAGS, "all_possible_second_rounds_validated"):
            self.assertIs(public[name], False)
        for category, count in (("training", 8), ("first", 136), ("references", 44),
                                ("sampled_second", 136)):
            self.assertEqual(public["by_category"][category],
                             {"scheduled": count, "checked": count,
                              "not_checked": 0, "over_limit": 0})
        for planned, actual in zip(self.inventory["rows"], public["rows"]):
            self.assertEqual(set(actual), {"ordinal", "category", "task_id", "arm",
                                          "row_sha256", "status", "input_tokens",
                                          "target_tokens", "full_sequence_tokens"})
            self.assertEqual(actual, {
                **{name: planned[name] for name in
                   ("ordinal", "category", "task_id", "arm", "row_sha256")},
                "status": "OK", "input_tokens": 3,
                "target_tokens": 2 if planned["answer"] is not None else None,
                "full_sequence_tokens": 6 if planned["answer"] is not None else None,
            })
        self.assertEqual(report["private"], {"inventory": self.inventory,
                                            "records": records, "finalization": finalization})
        for secret in (b"private synthetic", b"input_ids", b"messages", b"answer",
                       b"identity", b"special_tokens"):
            self.assertNotIn(secret, encoded(public))

    def test_category_edge_limits_keep_denominator(self):
        for ordinal in (0, 7, 8, 143, 144, 187, 188, 323):
            for stop in ("INPUT_LIMIT", "OUTPUT_LIMIT"):
                if stop == "OUTPUT_LIMIT" and self.inventory["rows"][ordinal]["answer"] is None:
                    continue
                records, finalization = synthetic_transcript(
                    self.inventory, self.profile, ordinal + 1, stop)
                public = self.report(records, finalization)["public"]
                with self.subTest(ordinal=ordinal, stop=stop):
                    self.assertEqual(len(public["rows"]), 324)
                    self.assertEqual(public["stop_ordinal"], ordinal)
                    self.assertEqual(public["stop_reason"], stop)
                    self.assertFalse(public["all_rows_within_budget"])
                    self.assertTrue(public["session_finalized"])
                    self.assertEqual(public["rows"][ordinal]["status"], stop)
                    self.assertEqual(sum(count["checked"] for count in
                                         public["by_category"].values()), ordinal + 1)
                    self.assertEqual(sum(count["over_limit"] for count in
                                         public["by_category"].values()), 1)
                    for row in public["rows"][ordinal + 1:]:
                        self.assertEqual(row["status"], "NOT_RUN_AFTER_STOP")
                        self.assertIsNone(row["input_tokens"])
                        self.assertIsNone(row["target_tokens"])
                        self.assertIsNone(row["full_sequence_tokens"])

    def test_missing_duplicate_reordered_and_after_stop_records(self):
        for mode in ("missing", "duplicate", "reordered", "after_stop", "empty"):
            records, finalization = synthetic_transcript(self.inventory, self.profile)
            if mode == "missing":
                del records[1]
            elif mode == "duplicate":
                records[1] = copy.deepcopy(records[0])
            elif mode == "reordered":
                records[0], records[1] = records[1], records[0]
            elif mode == "after_stop":
                limited, _ = synthetic_transcript(self.inventory, self.profile, 1, "INPUT_LIMIT")
                records[0] = limited[0]
            else:
                records.clear()
            rechain(records, finalization)
            finalization["measurements"] = len(records)
            with self.subTest(mode=mode), self.invalid():
                self.report(records, finalization)

    def test_finalization_exact_bindings_schema_and_flags(self):
        changes = [
            (("kind",), "OTHER"), (("extra",), "private"),
            (("run_id",), "b" * 32), (("profile_sha256",), "0" * 64),
            (("measurements",), 1), (("measurements",), 324.0),
            (("chain_sha256",), "0" * 64), (("stop_reason",), "INPUT_LIMIT"),
            (("tokenizer_loaded",), 1),
            (("runtime", "kind"), "OTHER"), (("runtime", "run_id"), "b" * 32),
            (("runtime", "profile_sha256"), "0" * 64),
            (("runtime", "versions", "jinja2"), "changed"),
            (("runtime", "tokenizer_loaded"), True), (("runtime", "extra"), "private"),
            (("snapshot", "kind"), "OTHER"), (("snapshot", "run_id"), "b" * 32),
            (("snapshot", "profile_sha256"), "0" * 64),
            (("snapshot", "revision_declaration"), "0" * 40),
            (("snapshot", "file_sha256", "config.json"), "0" * 64),
            (("snapshot", "file_bytes", "extra"), 1),
            (("snapshot", "file_bytes", "config.json"), True),
            (("snapshot", "file_bytes", "config.json"), -1),
            (("snapshot", "file_bytes", "config.json"), 128 * 1024 * 1024 + 1),
            (("snapshot", "tokenizer_loaded"), True),
        ]
        changes.extend((prefix + (flag,), value)
                       for prefix in ((), ("runtime",), ("snapshot",))
                       for flag in (*FLAGS, "all_possible_second_rounds_validated")
                       for value in (True, 0))
        records, baseline = synthetic_transcript(self.inventory, self.profile)
        for path, value in changes:
            finalization = copy.deepcopy(baseline)
            replace(finalization, path, value)
            with self.subTest(path=path, value=value), self.invalid():
                self.report(records, finalization)

    def test_identity_and_state_binding_even_with_recomputed_chains(self):
        for mode in ("record_identity", "record_state", "final_identity"):
            records, finalization = synthetic_transcript(self.inventory, self.profile)
            if mode == "record_identity":
                records[1]["identity_sha256"] = "0" * 64
            elif mode == "record_state":
                records[1]["measurement"]["state_sha256"] = "0" * 64
            else:
                finalization["identity"]["class"] = "changed.Class"
            rechain(records, finalization)
            with self.subTest(mode=mode), self.invalid():
                self.report(records, finalization)

    def test_identity_structure_and_measurement_state_hash(self):
        changes = [
            (("kind",), "OTHER"), (("extra",), "private"),
            (("is_fast",), 1), (("class",), ""), (("chat_template",), "changed"),
            (("template_sha256",), "bad"), (("vocabulary_sha256",), "bad"),
            (("measurement_state_sha256",), "0" * 64),
            (("eos_token_id",), True), (("pad_token_id",), -1),
            (("backend_sha256",), "bad"),
            (("special_token_ids", "[EOS]"), 3),
            (("special_tokens_map", "eos_token"), "[MISSING]"),
            (("settings", "model_max_length"), True),
            (("settings", "padding_side"), "middle"),
            (("settings", "split_special_tokens"), 0),
        ]
        for path, value in changes:
            records, finalization = synthetic_transcript(self.inventory, self.profile)
            replace(finalization["identity"], path, value)
            for record in records:
                record["identity_sha256"] = digest(finalization["identity"])
            rechain(records, finalization)
            with self.subTest(path=path), self.invalid():
                self.report(records, finalization)

    def test_unique_eos_at_target_end_and_not_in_tail(self):
        for target_ids, tail_ids in (([12, 13], []), ([2, 12], []),
                                     ([2, 2], []), ([12, 2], [2])):
            records, finalization = synthetic_transcript(self.inventory, self.profile)
            target = records[0]["measurement"]["target"]
            target["target_ids"] = target_ids
            target["template_tail_ids"] = tail_ids
            target["full_sequence_ids"] = [11] * 3 + target_ids + tail_ids
            target["full_sequence_tokens"] = len(target["full_sequence_ids"])
            rechain(records, finalization)
            with self.subTest(target_ids=target_ids, tail_ids=tail_ids), self.invalid():
                self.report(records, finalization)

    def test_infrastructure_failures_keep_checked_rows_and_sanitize(self):
        reasons = {"OPEN": "OPEN_FAILED", "MEASURE": "MEASUREMENT_FAILED",
                   "RECEIPT": "RECEIPT_INVALID", "FINALIZE": "FINALIZATION_FAILED"}
        for phase, count in (("OPEN", 0), ("MEASURE", 5), ("RECEIPT", 5), ("FINALIZE", 324)):
            records, _ = synthetic_transcript(self.inventory, self.profile, count)
            ordinal = count if phase in ("MEASURE", "RECEIPT") else None
            for abort_failed in (False, True):
                if phase == "OPEN" and abort_failed:
                    continue
                public = self.report(records, None, failure={"phase": phase, "ordinal": ordinal},
                                     abort_failed=abort_failed)["public"]
                with self.subTest(phase=phase, abort_failed=abort_failed):
                    self.assertEqual(public["failure"], {
                        "phase": phase, "ordinal": ordinal, "reason": reasons[phase]})
                    self.assertEqual(public["stop_reason"], reasons[phase])
                    self.assertEqual(public["stop_ordinal"], ordinal)
                    self.assertEqual(public["abort_failed"], abort_failed)
                    self.assertFalse(public["session_finalized"])
                    self.assertFalse(public["all_rows_within_budget"])
                    self.assertFalse(public["checked_inventory_within_budget"])
                    self.assertEqual(sum(value["checked"] for value in
                                         public["by_category"].values()), count)
                    if ordinal is not None:
                        self.assertEqual(public["rows"][ordinal]["status"], reasons[phase])
                        self.assertIsNone(public["rows"][ordinal]["input_tokens"])

    def test_finalize_failure_preserves_earlier_limit(self):
        records, _ = synthetic_transcript(self.inventory, self.profile, 8, "INPUT_LIMIT")
        public = self.report(records, None, failure={"phase": "FINALIZE", "ordinal": None})["public"]
        self.assertEqual(public["stop_reason"], "INPUT_LIMIT")
        self.assertEqual(public["stop_ordinal"], 7)
        self.assertEqual(public["failure"]["reason"], "FINALIZATION_FAILED")
        self.assertFalse(public["session_finalized"])

    def test_failure_combinations_and_missing_finalization_rejected(self):
        cases = [
            (0, None, None, None, False),
            (2, None, None, None, False),
            (324, None, None, None, False),
            (2, None, {"phase": "OPEN", "ordinal": None}, None, False),
            (0, None, {"phase": "OPEN", "ordinal": 0}, None, False),
            (2, None, {"phase": "MEASURE", "ordinal": 1}, None, False),
            (2, None, {"phase": "RECEIPT", "ordinal": 2.0}, None, False),
            (324, None, {"phase": "MEASURE", "ordinal": 324}, None, False),
            (2, "INPUT_LIMIT", {"phase": "MEASURE", "ordinal": 2}, None, False),
            (2, None, {"phase": "FINALIZE", "ordinal": None}, None, False),
            (324, None, {"phase": "FINALIZE", "ordinal": 0}, None, False),
            (324, None, {"phase": "FINALIZE", "ordinal": None}, True, False),
            (0, None, {"phase": "SECRET", "ordinal": None}, None, False),
            (0, None, {"phase": "OPEN", "ordinal": None, "error": "private"}, None, False),
            (324, None, None, True, True),
            (324, None, None, True, 0),
        ]
        for count, stop, failure, finalized, abort_failed in cases:
            records, finalization = synthetic_transcript(self.inventory, self.profile, count, stop)
            with self.subTest(count=count, failure=failure, stop=stop), self.invalid():
                self.report(records, finalization if finalized else None,
                            failure=failure, abort_failed=abort_failed)

    def test_report_revalidates_inputs_and_profile(self):
        records, finalization = synthetic_transcript(self.inventory, self.profile)
        with self.invalid():
            self.module.build_report(self.inventory_raw, "0" * 64, self.profile_raw,
                                     self.profile_sha256, records, finalization)
        for raw, approved in ((self.profile_raw, "0" * 64), (b"{}", text_hash("{}"))):
            with self.invalid():
                self.module.build_report(self.inventory_raw, self.inventory_sha256,
                                         raw, approved, records, finalization)
        with self.invalid():
            self.report(tuple(records), finalization)

    def test_report_private_and_public_do_not_alias_callers_or_each_other(self):
        records, finalization = synthetic_transcript(self.inventory, self.profile)
        report = self.report(records, finalization)
        baseline = copy.deepcopy(report)
        records[0]["measurement"]["input"]["input_ids"].clear()
        finalization["identity"]["settings"].clear()
        self.assertEqual(report, baseline)
        report["private"]["inventory"]["rows"][0]["task_id"] = "changed"
        report["private"]["records"][0]["measurement"]["input"]["input_tokens"] = 99
        self.assertEqual(report["public"], baseline["public"])

    def test_no_runtime_discovery_loading_or_io_during_validation(self):
        records, finalization = synthetic_transcript(self.inventory, self.profile)
        with patch("tokenizer_runtime_profile.runtime_inventory",
                   side_effect=AssertionError("runtime forbidden")) as runtime:
            with patch("builtins.open", side_effect=AssertionError("I/O forbidden")) as opening:
                self.assertTrue(self.report(records, finalization)["public"]["all_rows_within_budget"])
            opening.assert_not_called()
        runtime.assert_not_called()

    def test_import_does_not_load_session_snapshot_or_runtime_libraries(self):
        original_import = builtins.__import__
        attempts = []

        def guarded_import(name, *arguments, **keywords):
            if name.split(".")[0] in {
                "transformers", "tokenizers", "torch", "tokenizer_session", "tokenizer_snapshot",
            }:
                attempts.append(name)
                raise AssertionError("runtime import forbidden")
            return original_import(name, *arguments, **keywords)

        with patch("builtins.__import__", side_effect=guarded_import):
            with patch("tokenizer_runtime_profile.runtime_inventory",
                       side_effect=AssertionError("metadata forbidden")) as metadata:
                importlib.reload(self.module)
            metadata.assert_not_called()
        self.assertEqual(attempts, [])


if __name__ == "__main__":
    unittest.main()
