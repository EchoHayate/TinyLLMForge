import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest

from pilot_protocol import FLAGS
from recalibration_data_contract import FILE_PINS
from recalibration_validation import build_batches, build_report, build_schedule
from validation_contract import digest, encoded, parse_json

if importlib.util.find_spec("recalibration_tokenizer_inventory") is not None:
    import recalibration_tokenizer_inventory as inventory_module
else:
    inventory_module = None


ROOT = Path(__file__).resolve().parent


def process(raw):
    return {"status": "completed", "returncode": 0, "stdout_hex": raw.hex(),
            "stderr_hex": "", "elapsed_ns": 1, "process_reaped": True,
            "stdin_complete": True}


class RecalibrationInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = {name: (ROOT / "data" / (
            name if name == "supervision_v1.json" else "recalibration_v2_r1/" + name
        )).read_bytes() for name in FILE_PINS}
        cls.worker = (ROOT / "validation_task_worker.py").read_bytes()
        cls.renderer = {name: (ROOT / name).read_bytes()
                        for name in ("recalibration_rendering.py", "prompt_rendering.py")}
        cls.run_ids = ["%032x" % position for position in range(15)]
        cls.runtime = {"host": "synthetic-inventory-only", "uid": 123}
        cls.plan = build_schedule(cls.raw)
        cls.batches = build_batches(cls.raw, cls.worker, cls.run_ids)
        cls.records = []
        for batch in cls.batches:
            events = [{
                "kind": "ready", "run_id": batch["run_id"],
                "bundle_sha256": batch["bundle_sha256"], "worker_sha256": batch["worker_sha256"],
                "runtime": cls.runtime,
            }]
            for local, row in enumerate(cls.plan["rows"][batch["start"]:batch["stop"]]):
                events.append({
                    "kind": "row", "run_id": batch["run_id"], "ordinal": local,
                    "request_sha256": row["request_sha256"],
                    "result": process(encoded(row["witness_prediction"] or row["expected"])),
                })
            events.append({"kind": "done", "run_id": batch["run_id"], "received": 32,
                           "elapsed_ns": 1, "stopped": False})
            cls.records.append({
                "batch_index": batch["batch_index"], "bundle_raw": batch["bundle_raw"],
                "transport": process(b"".join(encoded(event) + b"\n" for event in events)),
            })

    def setUp(self):
        self.assertIsNotNone(inventory_module, "recalibration inventory not implemented")

    def build(self, records=None, renderer=None):
        return inventory_module.build_synthetic_inventory(
            self.raw, self.worker, self.run_ids,
            self.records if records is None else records,
            self.renderer if renderer is None else renderer, expected_runtime=self.runtime)

    def validate(self, inventory, records=None):
        return inventory_module.validate_synthetic_inventory(
            inventory, self.raw, self.worker, self.run_ids,
            self.records if records is None else records,
            self.renderer, expected_runtime=self.runtime)

    def changed_outcome(self, ordinal, outcome):
        records = copy.deepcopy(self.records)
        batch_index, local = divmod(ordinal, 32)
        events = [parse_json(line) for line in bytes.fromhex(
            records[batch_index]["transport"]["stdout_hex"]).splitlines()]
        events[local + 1]["result"] = process(encoded(outcome))
        records[batch_index]["transport"] = process(
            b"".join(encoded(event) + b"\n" for event in events))
        return records

    def rehash(self, inventory):
        for row in inventory["rows"]:
            row["messages_sha256"] = digest(row["messages"])
            row["answer_sha256"] = (None if row["answer"] is None else
                                    hashlib.sha256(row["answer"].encode("utf-8")).hexdigest())
            row["row_sha256"] = digest({
                name: value for name, value in row.items() if name != "row_sha256"})
        inventory["inventory_sha256"] = digest({
            name: value for name, value in inventory.items() if name != "inventory_sha256"})

    def test_exact_order_counts_and_determinism(self):
        actual = self.build()
        self.assertEqual(actual, self.build())
        self.assertEqual(actual["status"], "SYNTHETIC_ONLY")
        self.assertEqual(actual["kind"], "LORA_RECALIBRATION_TOKENIZER_INVENTORY")
        self.assertEqual(actual["version"], 1)
        self.assertEqual(actual["data_version"], "recalibration_v2_r1")
        self.assertEqual(actual["constructed_counts"], {"first": 24, "references": 48,
                                                       "sampled_second": 48})
        self.assertEqual(actual["planned_counts"], actual["constructed_counts"])
        self.assertEqual(actual["planned"], 120)
        self.assertEqual(len(actual["rows"]), 120)
        identities = [[row[name] for name in (
            "ordinal", "category", "task_id", "arm", "reference_id", "branch")]
                      for row in actual["rows"]]
        self.assertEqual(digest(identities),
                         "18dffaeabdda4ffa3633d9c3f2e1cb1026c3dd7e2d82cd2de929f9e8ccc769df")
        self.assertEqual(actual["limits"],
                         {"input_tokens": 4096, "answer_tokens_including_eos": 256,
                          "enable_thinking": False})

    def test_reference_answer_and_public_information_boundaries(self):
        actual = self.build()
        tasks = {task["id"]: task for task in parse_json(self.raw["prompts.json"])}
        oracles = {task["id"]: task for task in parse_json(self.raw["oracles.json"])}
        for row in actual["rows"]:
            first = row["messages"][0]["content"]
            public = parse_json(first.split("\n\nCurrent task:\n")[1].encode("ascii"))
            self.assertEqual(public, tasks[row["task_id"]])
            self.assertEqual(row["initial_source_sha256"], hashlib.sha256(
                tasks[row["task_id"]]["initial_source"].encode("utf-8")).hexdigest())
            self.assertEqual(row["examples_sha256"],
                             actual["examples_sha256"] if row["arm"] == "text" else None)
            if row["category"] == "references":
                expected = next(reference["source"] for reference in
                                oracles[row["task_id"]]["references"]
                                if reference["id"] == row["reference_id"])
                self.assertEqual(row["answer"], expected)
                self.assertEqual(len(row["messages"]), 1)
            else:
                self.assertIsNone(row["answer"])
                self.assertIsNone(row["reference_id"])
            if row["category"] == "sampled_second":
                self.assertEqual(row["messages"][1]["content"],
                                 tasks[row["task_id"]]["initial_source"])
                payload = parse_json(row["messages"][-1]["content"].split("\n", 1)[1].encode("ascii"))
                if row["branch"] == "retry":
                    self.assertEqual(payload, {"kind": "results_withheld"})
                    self.assertIsNone(row["public_feedback_sha256"])
                else:
                    self.assertEqual(set(payload), {"kind", "cases"})
                    self.assertEqual([case["id"] for case in payload["cases"]], ["P01", "P02"])
                    self.assertTrue(all(set(case) == {"id", "status"} for case in payload["cases"]))
                    self.assertEqual(row["public_feedback_sha256"], digest(payload))
            else:
                self.assertIsNone(row["branch"])
                self.assertIsNone(row["public_feedback_sha256"])

    def test_feedback_is_rebuilt_from_observed_public_rows(self):
        original = self.build()
        target = next(row for row in self.plan["rows"]
                      if (row["task_id"], row["role"], row["case_id"]) == ("C01", "initial", "P02"))
        self.assertIsNone(target["witness_prediction"])
        changed = self.build(self.changed_outcome(target["ordinal"],
                                                 {"kind": "return", "value": "wrong"}))
        self.assertEqual(changed["status"], "SYNTHETIC_ONLY")
        for previous, current in zip(original["rows"], changed["rows"]):
            if (current["task_id"] == "C01" and current["category"] == "sampled_second"
                    and current["branch"] == "feedback"):
                self.assertNotEqual(previous["messages"], current["messages"])
                self.assertIn('"id":"P02","status":"mismatch"', current["messages"][-1]["content"])
            else:
                self.assertEqual(previous, current)
        self.assertNotEqual(original["dynamic_evidence_sha256"], changed["dynamic_evidence_sha256"])
        with self.assertRaises(ValueError):
            self.validate(changed)
        hidden = next(row for row in self.plan["rows"]
                      if row["role"] == "initial" and row["case_id"].startswith("H")
                      and row["witness_prediction"] is None)
        hidden_records = self.changed_outcome(hidden["ordinal"], {"kind": "return", "value": "wrong"})
        hidden_changed = self.build(hidden_records)
        self.assertEqual(hidden_changed["status"], "SYNTHETIC_ONLY")
        self.assertEqual(original["rows"], hidden_changed["rows"])
        self.assertNotEqual(original["dynamic_evidence_sha256"],
                            hidden_changed["dynamic_evidence_sha256"])
        self.assertIsNone(self.validate(hidden_changed, hidden_records))

    def test_incomplete_and_failed_data_dependency_construct_no_rows(self):
        witness = next(row for row in self.plan["rows"] if row["witness_prediction"] is not None)
        for records in ([], self.records[:1], self.changed_outcome(
                0, {"kind": "return", "value": "wrong"}),
                self.changed_outcome(witness["ordinal"], witness["expected"])):
            with self.subTest(batches=len(records)):
                actual = self.build(records)
                report = build_report(self.raw, self.worker, self.run_ids, records,
                                      expected_runtime=self.runtime)
                self.assertEqual(actual["status"], "DEPENDENCY_UNSATISFIED")
                self.assertEqual(actual["planned"], 120)
                self.assertEqual(actual["rows"], [])
                self.assertEqual(actual["constructed_counts"],
                                 {"first": 0, "references": 0, "sampled_second": 0})
                self.assertIs(actual["synthetic_data_checks_satisfied"], False)
                self.assertEqual(actual["validation_report_sha256"], digest(report))
                self.assertEqual(actual["dynamic_evidence_sha256"], report["evidence_sha256"])
                self.assertEqual(actual["dependency_first_stop"], report["first_stop"])
                self.assertIsNone(self.validate(actual, records))
                for name in (*FLAGS, "run_authorized", "tokenizer_dispatch_authorized",
                             "execution_provenance_verified", "real_dynamic_validation_complete",
                             "tokenizer_length_validated", "all_possible_second_rounds_validated"):
                    self.assertIs(actual[name], False)

    def test_identity_hashes_and_input_detachment(self):
        inputs = copy.deepcopy((self.raw, self.worker, self.run_ids,
                                self.records, self.renderer, self.runtime))
        actual = self.build()
        self.assertEqual(inputs, (self.raw, self.worker, self.run_ids,
                                  self.records, self.renderer, self.runtime))
        report = build_report(self.raw, self.worker, self.run_ids, self.records,
                              expected_runtime=self.runtime)
        self.assertEqual(actual["validation_report_sha256"], digest(report))
        self.assertEqual(actual["validation_plan_sha256"], report["plan_sha256"])
        self.assertEqual(actual["dynamic_evidence_sha256"], report["evidence_sha256"])
        self.assertEqual(actual["expected_runtime_sha256"], digest(self.runtime))
        self.assertEqual(actual["worker_sha256"], hashlib.sha256(self.worker).hexdigest())
        self.assertEqual(actual["file_sha256"], FILE_PINS)
        self.assertEqual(actual["run_ids"], self.run_ids)
        self.assertIsNot(actual["run_ids"], self.run_ids)
        self.assertEqual(actual["rendering_sha256"], {
            name: hashlib.sha256(raw).hexdigest() for name, raw in self.renderer.items()})
        for row in actual["rows"]:
            self.assertEqual(row["row_sha256"], digest({
                name: value for name, value in row.items() if name != "row_sha256"}))
            self.assertEqual(row["messages_sha256"], digest(row["messages"]))
            self.assertEqual(row["answer_sha256"], None if row["answer"] is None
                             else hashlib.sha256(row["answer"].encode("utf-8")).hexdigest())
        self.assertEqual(actual["inventory_sha256"], digest({
            name: value for name, value in actual.items() if name != "inventory_sha256"}))
        original = copy.deepcopy(actual["rows"][0]["messages"])
        actual["rows"][24]["messages"][0]["content"] = "changed"
        self.assertEqual(actual["rows"][0]["messages"], original)
        self.assertEqual(self.build()["rows"][24]["messages"], original)

    def test_synthetic_flags_and_tampering_rejected_after_rehash(self):
        actual = self.build()
        self.assertIs(actual["synthetic_only"], True)
        self.assertEqual(actual["evidence_kind"], "synthetic")
        flags = (*FLAGS, "run_authorized", "tokenizer_dispatch_authorized",
                 "execution_provenance_verified", "real_dynamic_validation_complete",
                 "tokenizer_length_validated", "all_possible_second_rounds_validated")
        for name in flags:
            self.assertIs(actual[name], False)
            changed = copy.deepcopy(actual)
            changed[name] = True
            self.rehash(changed)
            with self.subTest(flag=name), self.assertRaises(ValueError):
                self.validate(changed)
        mutations = [
            (0, "messages", [{"role": "user", "content": "changed"}]),
            (24, "answer", "def changed():\n    return 1\n"),
            (72, "public_feedback_sha256", "0" * 64),
            (0, "initial_source_sha256", "0" * 64),
            (1, "examples_sha256", "0" * 64),
        ]
        for ordinal, name, value in mutations:
            changed = copy.deepcopy(actual)
            changed["rows"][ordinal][name] = value
            self.rehash(changed)
            with self.subTest(row=ordinal, field=name), self.assertRaises(ValueError):
                self.validate(changed)
        self.assertIsNone(self.validate(actual))

    def test_old_inventories_reordering_and_prebuilt_pass_rejected(self):
        actual = self.build()
        mutations = [
            {**actual, "kind": "LORA_TOKENIZER_INVENTORY"},
            {**actual, "rows": actual["rows"] * 3},
            {**actual, "rows": list(reversed(actual["rows"]))},
            {**actual, "version": True},
            {**actual, "evidence_kind": "real"},
            {**actual, "synthetic_only": False},
            {**actual, "extra": 1},
        ]
        for changed in mutations:
            self.rehash(changed)
            with self.subTest(kind=changed["kind"]), self.assertRaises(ValueError):
                self.validate(changed)
        for records in ({"kind": "LORA_RECALIBRATION_VALIDATION_REPORT",
                         "matrix_complete": True, "data_checks_satisfied": True},
                        {"kind": "LORA_REAL_VALIDATION_REPORT", "status": "PASS"}):
            with self.subTest(records=records), self.assertRaises(ValueError):
                self.build(records)

    def test_renderer_bytes_and_strict_local_types(self):
        class StringSubclass(str):
            pass

        class DictSubclass(dict):
            pass

        class ListSubclass(list):
            pass

        changed = dict(self.renderer)
        changed["recalibration_rendering.py"] += b"\n"
        renderers = [
            changed, {}, {**self.renderer, "extra.py": b""}, DictSubclass(self.renderer),
            {StringSubclass(name): value for name, value in self.renderer.items()},
            {name: bytearray(value) for name, value in self.renderer.items()},
        ]
        for renderer in renderers:
            with self.subTest(keys=list(renderer)), self.assertRaises(ValueError):
                self.build(renderer=renderer)
        actual = self.build()
        for invalid in (float("nan"), float("inf"), StringSubclass("4096"), (4096,)):
            changed = copy.deepcopy(actual)
            changed["limits"]["input_tokens"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.validate(changed)
        for changed in (DictSubclass(actual), {**actual, "rows": ListSubclass(actual["rows"])}):
            with self.assertRaises(ValueError):
                self.validate(changed)

    def test_no_execution_or_tokenizer_imports(self):
        tree = ast.parse((ROOT / "recalibration_tokenizer_inventory.py").read_text())
        allowed = {"copy", "hashlib", "pilot_protocol", "recalibration_data_contract",
                   "recalibration_rendering", "recalibration_validation", "validation_contract"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name in allowed for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"exec", "eval", "compile", "open", "__import__"})


if __name__ == "__main__":
    unittest.main()
