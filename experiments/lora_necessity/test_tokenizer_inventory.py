import copy
import importlib
from pathlib import Path
import unittest

from prompt_rendering import episode_jobs, second_round, training_jobs
from validation_contract import digest, encoded, load_frozen, reference_jobs


class TokenizerInventoryTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("tokenizer_inventory.py").is_file(),
                        "tokenizer inventory not implemented")
        self.module = importlib.import_module("tokenizer_inventory")
        directory = Path(__file__).with_name("data")
        self.inputs = tuple((directory / name).read_bytes() for name in (
            "task_sources_v1.json", "manifest_v1.json",
            "supervision_v1.json", "prompts_v1.json",
        ))

    def test_complete_order_and_content_hashes(self):
        inventory = self.module.build_inventory(*self.inputs)
        rows = inventory["rows"]
        self.assertEqual(len(rows), 324)
        self.assertEqual([row["ordinal"] for row in rows], list(range(324)))
        self.assertEqual([row["category"] for row in rows],
                         ["training"] * 8 + ["first"] * 136 +
                         ["references"] * 44 + ["sampled_second"] * 136)
        for row in rows:
            self.assertEqual(set(row), {
                "ordinal", "category", "task_id", "arm", "messages",
                "answer", "sample", "row_sha256",
            })
            self.assertEqual(row["row_sha256"],
                             digest({name: value for name, value in row.items()
                                     if name != "row_sha256"}))
        for name in ("benchmark_eligible", "formal_evaluation_ready",
                     "net_benefit_established", "isolated",
                     "all_possible_second_rounds_validated"):
            self.assertIs(inventory[name], False)

    def test_exact_training_first_and_private_reference_projection(self):
        rows = self.module.build_inventory(*self.inputs)["rows"]
        training = training_jobs(self.inputs[2])
        first = episode_jobs(self.inputs[2], self.inputs[3])
        prompts = {}
        for actual, expected in zip(rows[:8], training):
            self.assertEqual(actual["messages"], expected["messages"])
            self.assertEqual(actual["answer"], expected["answer"])
            self.assertIsNone(actual["arm"])
            prompts[expected["task_id"]] = expected["messages"]
        for actual, expected in zip(rows[8:144], first):
            self.assertEqual(actual["messages"], expected["messages"])
            self.assertEqual(actual["arm"], expected["arm"])
            self.assertIsNone(actual["answer"])
            if expected["arm"] == "base":
                prompts[expected["task_id"]] = expected["messages"]
        for actual, expected in zip(rows[144:188], reference_jobs(*self.inputs[:2])):
            self.assertEqual(actual["task_id"], expected["id"])
            self.assertEqual(actual["answer"], expected["reference_source"])
            self.assertEqual(actual["messages"], prompts[expected["id"]])
            self.assertIsNone(actual["arm"])

    def test_fixed_second_round_and_equal_information(self):
        rows = self.module.build_inventory(*self.inputs)["rows"]
        public = {task["id"]: task for task in
                  load_frozen("prompts_v1.json", self.inputs[3])}
        for row in rows[188:]:
            task = public[row["task_id"]]
            feedback = {"kind": "visible_tests", "cases": [
                {"id": case["id"], "status": "mismatch"}
                for case in task["public_cases"]
            ]}
            expected = second_round(self.inputs[2], self.inputs[3], row["task_id"],
                                    row["arm"], task["initial_source"], feedback)
            self.assertEqual(row["messages"], expected)
            self.assertEqual(row["sample"]["kind"], "INITIAL_SOURCE_SYNTHETIC_FEEDBACK")
            self.assertEqual(row["sample"]["feedback_sha256"], digest(feedback))
            self.assertIsNone(row["answer"])
        indexed = {(row["task_id"], row["arm"]): row for row in rows[8:144]}
        for (task_id, arm), row in indexed.items():
            if arm == "lora_text":
                self.assertEqual(row["messages"], indexed[(task_id, "text")]["messages"])

    def test_frozen_byte_tampering_rejected(self):
        for index in range(4):
            changed = list(self.inputs)
            changed[index] += b"\n"
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.module.build_inventory(*changed)

    def test_minimal_rows_and_no_shared_mutable_payloads(self):
        inventory = self.module.build_inventory(*self.inputs)
        original = copy.deepcopy(inventory)
        self.assertNotIn(b"scoring_cases", encoded(inventory))
        self.assertNotIn(b"witness_prediction", encoded(inventory))
        self.assertNotIn(b"mutants", encoded(inventory))
        inventory["rows"][0]["messages"][0]["content"] = "changed"
        self.assertEqual(inventory["rows"][144:188], original["rows"][144:188])
        self.assertEqual(self.module.build_inventory(*self.inputs), original)
