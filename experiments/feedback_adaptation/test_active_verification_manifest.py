import copy
import itertools
import unittest

from active_verification_manifest import (
    ROLES, build_manifest, manifest_digest, validate_manifest, validate_tasks,
)
from test_active_verification_contract import LIMITS, RUN


def sample_task(index=1, split="development", family=None):
    return {
        "evidence_kind": "synthetic", "task_id": f"{index:032x}",
        "family_id": family or f"family-{index}", "split": split, "domain": "synthetic",
        "specification": f"Synthetic task {index}; opaque strings, never execute.",
        "sources": {role: f"opaque-{index}-{role}" for role in ROLES},
        "check_inputs": [{"id": "probe-0", "input": "opaque input"}],
        "private": {
            "case_ids": ["score-0", "score-1"],
            "pass_by_source": {
                "correct": [True, True], "defect": [True, False], "other": [False, True],
            },
            "feedback_by_initial": {
                initial: {"correct": f"claim-{initial}-true", "incorrect": f"claim-{initial}-false"}
                for initial in ("correct", "defect")
            },
            "verification_information": "sufficient", "audit_note": "PRIVATE_SENTINEL",
        },
    }


def sample_manifest(all_permutations=False):
    permutations = list(map(list, itertools.permutations(ROLES))) if all_permutations else [list(ROLES)]
    return build_manifest([sample_task()], permutations, RUN, "development", LIMITS)


class ManifestTests(unittest.TestCase):
    def test_complete_grid_and_budgets(self):
        for all_permutations, expected in ((False, 6), (True, 36)):
            manifest = sample_manifest(all_permutations)
            self.assertEqual(len(manifest["cells"]), expected)
            self.assertEqual(manifest["budgets"], {
                "cells": expected, "records": expected * 5,
                "model_calls": expected * 6, "tool_calls": expected * 2,
                "executed_cases": expected * 2,
                "input_tokens": expected * 6 * 8, "output_tokens": expected * 6 * 4,
            })
            self.assertEqual(len({cell["episode_id"] for cell in manifest["cells"]}), expected)
            self.assertEqual(sum(cell["condition"] == "none" for cell in manifest["cells"]), expected // 3)

    def test_all_permutations_keep_source_and_score_alignment(self):
        manifest = sample_manifest(True)
        task = sample_task()
        for cell in manifest["cells"]:
            for letter, role in zip("ABC", cell["permutation"]):
                self.assertEqual(cell["view"]["candidates"][letter], task["sources"][role])
                self.assertEqual(cell["pass_by_letter"][letter], task["private"]["pass_by_source"][role])
            self.assertEqual(cell["view"]["initial_source"], task["sources"][cell["initial"]])
            self.assertEqual(len(cell["view"]["feedback"]), int(cell["condition"] != "none"))

    def test_family_crossing_and_exact_clones_rejected(self):
        with self.assertRaises(ValueError):
            validate_tasks([sample_task(), sample_task(2, "sealed", "family-1")])
        clone = sample_task(2, "sealed")
        original = sample_task()
        clone.update(specification=original["specification"], sources=original["sources"])
        clone["check_inputs"][0]["input"] = "changed"
        with self.assertRaises(ValueError):
            validate_tasks([original, clone])
        validate_tasks([sample_task(), sample_task(2, family="family-1")])

    def test_task_and_private_shapes_are_strict(self):
        mutations = (
            lambda task: task.update(evidence_kind="real"),
            lambda task: task.update(task_id=True),
            lambda task: task.update(split="unknown"),
            lambda task: task.update(extra="leak"),
            lambda task: task["sources"].update(other=task["sources"]["correct"]),
            lambda task: task["private"]["pass_by_source"].update(correct=[True, False]),
            lambda task: task["private"]["pass_by_source"].update(defect=[True, True]),
            lambda task: task["private"]["pass_by_source"].update(other=[1, False]),
            lambda task: task["private"].update(case_ids=["duplicate", "duplicate"]),
            lambda task: task["private"].update(verification_information="unknown"),
            lambda task: task["private"]["feedback_by_initial"]["correct"].update(incorrect="claim-correct-true"),
            lambda task: task["check_inputs"][0].update(expected="premature"),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                task = sample_task()
                mutate(task)
                validate_tasks([task])
        for tasks in ([], None, [sample_task(), sample_task()]):
            with self.assertRaises(ValueError):
                validate_tasks(tasks)

    def test_freeze_rejects_cells_scores_budgets_and_private_drift(self):
        manifest = sample_manifest()
        digest = manifest_digest(manifest)
        for mutate in (
            lambda value: value["cells"].pop(),
            lambda value: value["cells"].append(copy.deepcopy(value["cells"][0])),
            lambda value: value["cells"][0]["pass_by_letter"]["A"].__setitem__(0, False),
            lambda value: value["cells"][0]["view"].update(feedback=["changed"]),
            lambda value: value["budgets"].update(model_calls=1),
        ):
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                changed = copy.deepcopy(manifest)
                mutate(changed)
                validate_manifest(changed)
        changed = copy.deepcopy(manifest)
        changed["tasks"][0]["private"]["audit_note"] = "changed private evidence"
        self.assertNotEqual(manifest_digest(changed), digest)

    def test_isolation_and_public_only_identity(self):
        tasks = [sample_task()]
        before = copy.deepcopy(tasks)
        manifest = build_manifest(tasks, [list(ROLES)], RUN, "development", LIMITS)
        other = copy.deepcopy(tasks)
        other[0]["private"]["audit_note"] = "another private note"
        revised = build_manifest(other, [list(ROLES)], RUN, "development", LIMITS)
        self.assertEqual([cell["episode_id"] for cell in manifest["cells"]],
                         [cell["episode_id"] for cell in revised["cells"]])
        self.assertNotIn("PRIVATE_SENTINEL", repr(manifest["cells"][0]["view"]))
        manifest["cells"][0]["view"]["candidates"]["A"] = "changed"
        self.assertEqual(tasks, before)
        self.assertEqual(manifest["tasks"], before)

    def test_split_selection_and_bad_permutations(self):
        tasks = [sample_task(), sample_task(2, "sealed")]
        manifest = build_manifest(tasks, [list(ROLES)], RUN, "sealed", LIMITS)
        self.assertEqual({cell["task_id"] for cell in manifest["cells"]}, {f"{2:032x}"})
        for permutations in ([], [list(ROLES)] * 2, [["correct"] * 3], [list(ROLES[:2])], [None]):
            with self.assertRaises(ValueError):
                build_manifest(tasks, permutations, RUN, "development", LIMITS)
        with self.assertRaises(ValueError):
            build_manifest([sample_task()], [list(ROLES)], RUN, "sealed", LIMITS)


if __name__ == "__main__":
    unittest.main()
