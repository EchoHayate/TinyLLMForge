import ast
import json
import unittest
from unittest import mock

from development_programs import domains
from development_tasks import anchors, catalog, reference


class DevelopmentTaskTests(unittest.TestCase):
    def test_complete_finite_domains(self):
        cases = domains()
        self.assertEqual(list(cases), ["dev-001", "dev-002", "dev-003", "dev-004"])
        self.assertEqual([len(values) for values in cases.values()], [121, 40, 121, 121])
        for values in cases.values():
            self.assertEqual(len(values), len({json.dumps(value) for value in values}))

    def test_literal_anchors_are_independent_of_reference_generation(self):
        expected = {
            "dev-001": [([], 0), ([-1, -1], 1), ([1, 0, 1], 1),
                        ([-1, 0, 1], 3), ([1, -1, 0], 1)],
            "dev-002": [([0, 0, 0], 0), ([0, 1, 0], 1), ([1, 3, 1], 1),
                        ([1, 3, 3], 0)],
            "dev-003": [("", -1), ("0", 0), ("10", 2), ("+1", -1),
                        ("1+", -1), ("1111", 15)],
            "dev-004": [("", 0), ("r", 0), ("or", 1), ("cr", 0),
                        ("oor", 1), ("orrr", 3)],
        }
        self.assertEqual(anchors(), expected)
        for task_id, examples in expected.items():
            for value, answer in examples:
                self.assertEqual(reference(task_id, value), answer)

    def test_catalog_scope_partition_and_neutral_sources(self):
        tasks = catalog()
        self.assertEqual(len({task["family_id"] for task in tasks}), 4)
        for task in tasks:
            self.assertEqual(task["evidence_kind"], "fixed_development")
            self.assertEqual(task["split"], "development")
            cases = task["private"]["cases"]
            probes = task["check_inputs"]
            probe_ids = {row["id"] for row in probes}
            score_ids = set(task["private"]["score_ids"])
            self.assertEqual(len(probe_ids), 2)
            self.assertFalse(probe_ids & score_ids)
            self.assertEqual(probe_ids | score_ids, {row["id"] for row in cases})
            self.assertTrue(all(set(row) == {"id", "input"} for row in probes))
            self.assertEqual(set(task["sources"]), {"correct", "defect", "other"})
            self.assertEqual(len(set(task["sources"].values())), 3)
            for source in task["sources"].values():
                tree = ast.parse(source)
                self.assertEqual(len(tree.body), 1)
                self.assertEqual(tree.body[0].name, "solve")
                self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom))
                                     for node in ast.walk(tree)))
            for case in cases:
                self.assertIs(type(case["expected"]), int)
                self.assertEqual(case["expected"], reference(task["task_id"], case["input"]))

    def test_feedback_witness_and_information_class_are_frozen(self):
        for task, information in zip(
            catalog(), ("sufficient", "insufficient", "sufficient", "insufficient")
        ):
            private = task["private"]
            self.assertEqual(private["verification_information"], information)
            self.assertEqual(
                private["witness_id"] in {row["id"] for row in task["check_inputs"]},
                information == "sufficient",
            )
            for initial in ("correct", "defect"):
                claims = private["feedback_by_initial"][initial]
                self.assertEqual(set(claims), {"correct", "incorrect"})
                self.assertNotEqual(claims["correct"], claims["incorrect"])
                self.assertNotIn("candidate", claims["correct"])

    def test_invalid_input_domains_reject_instead_of_coercing(self):
        invalid = {
            "dev-001": ([True], [2], [0] * 5, "0", None, (0, 1)),
            "dev-002": ([0, 1, True], [2, 1, 0], [0, 4, 0], [0, 1], None, (0, 1, 0)),
            "dev-003": (True, "2", "00000", None),
            "dev-004": (True, "x", "ooooo", None),
        }
        for task_id, values in invalid.items():
            for value in values:
                with self.subTest(task=task_id, value=value), self.assertRaises(ValueError):
                    reference(task_id, value)
        with self.assertRaises(ValueError):
            reference("unknown", [])

    def test_literal_disagreement_blocks_catalog_before_execution(self):
        with mock.patch("development_tasks.reference", return_value=99):
            with self.assertRaisesRegex(ValueError, "literal/reference disagreement"):
                catalog()

    def test_returned_material_does_not_mutate_future_calls(self):
        original = catalog()
        changed = catalog()
        changed[0]["private"]["cases"][0]["input"].append(1)
        changed[0]["sources"]["correct"] = "changed"
        changed_domains = domains()
        changed_domains["dev-001"][0].append(1)
        changed_anchors = anchors()
        changed_anchors["dev-001"][0][0].append(1)
        self.assertEqual(original, catalog())
        self.assertEqual(domains()["dev-001"][0], [])
        self.assertEqual(anchors()["dev-001"][0], ([], 0))


if __name__ == "__main__":
    unittest.main()
