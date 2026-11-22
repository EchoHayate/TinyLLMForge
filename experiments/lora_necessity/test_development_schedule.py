import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import test_development_final_scoring as scoring_fixtures
from prompt_rendering import PUBLIC_FIELDS, episode_jobs
from validation_contract import PINS, _tasks, digest, encoded


ORDER = [
    ("D01", "base"), ("D01", "text"), ("D02", "text"), ("D02", "base"),
    ("D03", "base"), ("D03", "text"), ("D04", "text"), ("D04", "base"),
]


def public(task):
    return {
        "id": task["id"], "entry_point": task["entry_point"],
        "initial_source": task["initial_source"], "spec": task["spec"],
        "public_cases": [
            {name: copy.deepcopy(case[name]) for name in ("id", "args", "expected")}
            for case in task["public_cases"]
        ],
    }


class DevelopmentScheduleTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name("development_schedule.py")
        self.assertTrue(path.is_file(), "missing development_schedule.py implementation")
        self.scheduler = importlib.import_module("development_schedule")
        self.original_pins = dict(PINS)
        self.addCleanup(self.assertEqual, PINS, self.original_pins)
        patch = mock.patch.dict(PINS, {})
        patch.start()
        self.addCleanup(patch.stop)
        self.tasks = copy.deepcopy(scoring_fixtures.synthetic_tasks())
        for task in self.tasks:
            task["spec"] = "Synthetic identity task"
            task["initial_witness"] = {
                "public_case_id": "P01",
                "predicted_initial": {"kind": "return", "value": 99},
            }
            if task["split"] == "supervision":
                task["reference_source"] = "def toy(value):\n    return value\n"
        self.task = next(task for task in self.tasks if task["id"] == "D01")
        self.task["scoring_cases"][0]["args"] = ["PRIVATE_SCORING_ARGUMENT"]
        self.task["scoring_cases"][0]["expected"] = {
            "kind": "return", "value": "PRIVATE_SCORING_EXPECTATION",
        }
        self.prompts = [public(task) for task in self.tasks if task["split"] != "supervision"]
        self.supervision = [
            {**public(task), "reference_source": task["reference_source"],
             "initial_witness": copy.deepcopy(task["initial_witness"])}
            for task in self.tasks if task["split"] == "supervision"
        ]
        self.run_id = "1" * 32
        self.freeze()

    def freeze(self):
        source_raw, manifest_raw = scoring_fixtures.frozen_bytes(self.tasks)
        self.raw = {
            "supervision_raw": encoded(self.supervision),
            "prompts_raw": encoded(self.prompts),
            "source_raw": source_raw,
            "manifest_raw": manifest_raw,
        }
        for key, name in (
            ("supervision_raw", "supervision_v1.json"), ("prompts_raw", "prompts_v1.json"),
            ("source_raw", "task_sources_v1.json"), ("manifest_raw", "manifest_v1.json"),
        ):
            PINS[name] = hashlib.sha256(self.raw[key]).hexdigest()

    def schedule(self, **changes):
        return self.scheduler.build_schedule(**{**self.raw, "run_id": self.run_id, **changes})

    def test_exact_schema_eight_alternating_jobs_and_sixteen_calls(self):
        result = self.schedule()
        self.assertEqual(set(result), {
            "kind", "version", "run_id", "data_sha256", "scheduled_episodes",
            "scheduled_generation_calls", "episodes",
        })
        self.assertEqual(result["kind"], "LORA_DEVELOPMENT_SCHEDULE")
        self.assertEqual(result["version"], 1)
        self.assertEqual(result["run_id"], self.run_id)
        self.assertEqual(result["data_sha256"], PINS)
        self.assertEqual(result["scheduled_episodes"], 8)
        self.assertEqual(result["scheduled_generation_calls"], 16)
        episodes = result["episodes"]
        self.assertEqual([(row["task_id"], row["arm"]) for row in episodes], ORDER)
        self.assertEqual([row["ordinal"] for row in episodes], list(range(8)))
        for row in episodes:
            self.assertEqual(set(row), {
                "ordinal", "task_id", "arm", "messages", "examples_sha256", "task_sha256",
                "messages_sha256", "first_call_id", "second_call_id", "public_attempt_id",
                "scheduled_final_cases", "final_cases",
            })

    def test_actual_frozen_loader_and_renderer_are_exercised(self):
        self.assertEqual(len(_tasks(self.raw["source_raw"], self.raw["manifest_raw"])), 44)
        jobs = episode_jobs(self.raw["supervision_raw"], self.raw["prompts_raw"])
        self.assertEqual(len(jobs), 136)
        indexed = {(job["task_id"], job["arm"]): job for job in jobs}
        for row in self.schedule()["episodes"]:
            original = indexed[(row["task_id"], row["arm"])]
            self.assertEqual(encoded(row["messages"]), encoded(original["messages"]))
            self.assertEqual(row["examples_sha256"], original["examples_sha256"])
            self.assertEqual(row["messages_sha256"], digest(original["messages"]))
            task = next(task for task in self.tasks if task["id"] == row["task_id"])
            self.assertEqual(row["task_sha256"], digest(task))

    def test_text_has_all_eight_original_examples_and_base_has_none(self):
        expected_examples = sorted(self.supervision, key=lambda task: task["id"])
        self.assertEqual(len(expected_examples), 8)
        for row in self.schedule()["episodes"]:
            content = row["messages"][0]["content"]
            current = json.loads(content.split("\n\nCurrent task:\n")[1])
            task = next(task for task in self.tasks if task["id"] == row["task_id"])
            self.assertEqual(current, public(task))
            if row["arm"] == "base":
                self.assertIsNone(row["examples_sha256"])
                self.assertNotIn("Supervised examples:", content)
            else:
                examples = json.loads(content.split("\n\nSupervised examples:\n")[1]
                                      .split("\n\nCurrent task:\n")[0])
                self.assertEqual(examples, expected_examples)
                self.assertEqual(row["examples_sha256"], digest(expected_examples))
                self.assertTrue(all(example["reference_source"] == scoring_fixtures.CANDIDATE
                                    for example in examples))

    def test_independent_call_and_attempt_digest_objects(self):
        identifiers = []
        for row in self.schedule()["episodes"]:
            identity = {"run_id": self.run_id, "task_id": row["task_id"], "arm": row["arm"]}
            for round_number, name in ((1, "first_call_id"), (2, "second_call_id")):
                expected = digest({
                    "kind": "LORA_DEVELOPMENT_GENERATION_ID", "version": 1,
                    **identity, "round": round_number,
                })[:32]
                self.assertEqual(row[name], expected)
                self.assertRegex(row[name], "^[0-9a-f]{32}$")
                identifiers.append(row[name])
            expected = digest({
                "kind": "LORA_DEVELOPMENT_PUBLIC_ATTEMPT", "version": 1,
                **identity, "round": 1,
            })[:32]
            self.assertEqual(row["public_attempt_id"], expected)
            identifiers.append(row["public_attempt_id"])
        self.assertEqual(len(identifiers), 24)
        self.assertEqual(len(set(identifiers)), 24)

    def test_different_run_changes_all_logical_ids_but_not_messages_inventory(self):
        before = self.schedule()
        after = self.schedule(run_id="a" * 32)
        for original, changed in zip(before["episodes"], after["episodes"]):
            for name in ("first_call_id", "second_call_id", "public_attempt_id"):
                self.assertNotEqual(original[name], changed[name])
            for name in ("task_id", "arm", "messages", "messages_sha256",
                         "examples_sha256", "task_sha256", "final_cases",
                         "scheduled_final_cases"):
                self.assertEqual(original[name], changed[name])
        self.assertEqual(before["data_sha256"], after["data_sha256"])

    def test_final_inventory_is_sorted_full_union_and_exact_hashes_only(self):
        self.task["scoring_cases"].append({
            "id": "A-extra", "args": [4], "expected": {"kind": "return", "value": 4},
            "reason": "extra synthetic case",
        })
        self.freeze()
        for row in self.schedule()["episodes"]:
            task = next(task for task in self.tasks if task["id"] == row["task_id"])
            expected = []
            for group, key in (("public", "public_cases"), ("scoring", "scoring_cases")):
                for case in sorted(task[key], key=lambda case: case["id"]):
                    expected.append({
                        "case_group": group, "case_id": case["id"],
                        "input_sha256": digest(case["args"]),
                        "expected_sha256": digest(case["expected"]),
                    })
            self.assertEqual(row["final_cases"], expected)
            self.assertEqual(row["scheduled_final_cases"], len(expected))
            if row["task_id"] == "D01":
                self.assertEqual([case["case_id"] for case in row["final_cases"]],
                                 ["P01", "P02", "A-extra", "H01"])

    def test_private_values_reasons_and_references_do_not_leak(self):
        result = self.schedule()
        for row in result["episodes"]:
            for value in (row["messages"], row["final_cases"]):
                raw = encoded(value)
                for marker in (b"PRIVATE_SCORING_ARGUMENT", b"PRIVATE_SCORING_EXPECTATION",
                               b"REASON_MUST_NOT_LEAK", b"REFERENCE_MUST_NOT_LEAK"):
                    self.assertNotIn(marker, raw)
            for case in row["final_cases"]:
                self.assertEqual(set(case), {
                    "case_group", "case_id", "input_sha256", "expected_sha256",
                })
        self.assertTrue(all(set(case) == {"id", "args", "expected", "reason"}
                            for case in self.task["public_cases"]))
        prompt = next(task for task in self.prompts if task["id"] == "D01")
        self.assertEqual(set(prompt), set(PUBLIC_FIELDS))
        self.assertTrue(all(set(case) == {"id", "args", "expected"}
                            for case in prompt["public_cases"]))

    def test_public_private_projection_mismatches_rejected(self):
        original = copy.deepcopy(self.task)
        mutations = {
            "entry_point": lambda task: task.update(entry_point="other"),
            "spec": lambda task: task.update(spec="different specification"),
            "initial_source": lambda task: task.update(initial_source="different source"),
            "public_args": lambda task: task["public_cases"][0].update(args=[777]),
            "public_expected": lambda task: task["public_cases"][0].update(
                expected={"kind": "return", "value": 777}),
        }
        for name, mutate in mutations.items():
            self.task.clear()
            self.task.update(copy.deepcopy(original))
            mutate(self.task)
            self.freeze()
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "public and private task views differ"
            ):
                self.schedule()

    def test_prompt_projection_cannot_include_reason(self):
        prompt = next(task for task in self.prompts if task["id"] == "D01")
        prompt["public_cases"][0]["reason"] = "not part of the public projection"
        self.freeze()
        with self.assertRaisesRegex(ValueError, "public and private task views differ"):
            self.schedule()

    def test_private_metadata_changes_hashes_not_public_messages(self):
        before = self.schedule()
        self.task["scoring_cases"][0].update(
            args=["changed private args"],
            expected={"kind": "return", "value": "changed private expectation"},
            reason="changed private reason",
        )
        self.freeze()
        after = self.schedule()
        for original, changed in zip(before["episodes"], after["episodes"]):
            self.assertEqual(original["messages"], changed["messages"])
            self.assertEqual(original["messages_sha256"], changed["messages_sha256"])
            if original["task_id"] == "D01":
                self.assertNotEqual(original["task_sha256"], changed["task_sha256"])
                self.assertNotEqual(original["final_cases"], changed["final_cases"])
            else:
                self.assertEqual(original, changed)
        self.assertNotEqual(before["data_sha256"], after["data_sha256"])

    def test_each_raw_hash_drift_is_rejected(self):
        for name, raw in self.raw.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "frozen bytes changed"):
                self.schedule(**{name: raw + b"\n"})

    def test_each_wrong_pin_is_rejected(self):
        for name in PINS:
            with self.subTest(name=name), mock.patch.dict(PINS, {name: "0" * 64}):
                with self.assertRaisesRegex(ValueError, "frozen bytes changed"):
                    self.schedule()

    def test_each_raw_input_requires_exact_bounded_bytes(self):
        for name in self.raw:
            for value in ("[]", bytearray(b"[]"), memoryview(b"[]"), b" " * 1048577):
                with self.subTest(name=name, value_type=type(value)):
                    with self.assertRaisesRegex(ValueError, "frozen file bytes"):
                        self.schedule(**{name: value})

    def test_source_metadata_mismatch_still_uses_real_frozen_loader(self):
        changed = copy.deepcopy(self.tasks)
        changed[0]["template_id"] += "_changed"
        raw = encoded({"tasks": changed})
        with mock.patch.dict(PINS, {"task_sources_v1.json": hashlib.sha256(raw).hexdigest()}):
            with self.assertRaisesRegex(ValueError, "task metadata binding"):
                self.schedule(source_raw=raw)

    def test_invalid_run_id_rejected_before_loading(self):
        for value in (None, True, 1, [], {}, "", "A" * 32, "a" * 31, "a" * 33, "g" * 32):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "run identity"):
                self.scheduler.build_schedule(b"", b"", b"", b"", run_id=value)

    def test_run_id_subclass_rejected_without_hooks(self):
        class Hostile(str):
            def __eq__(self, other):
                raise AssertionError("equality hook")

            def __str__(self):
                raise AssertionError("string hook")

        with self.assertRaisesRegex(ValueError, "run identity"):
            self.schedule(run_id=Hostile(self.run_id))

    def test_no_order_or_selection_override_keywords(self):
        for changes in ({"arm": "lora"}, {"task_id": "T01"}, {"order": ORDER[::-1]}):
            with self.subTest(changes=changes), self.assertRaises(TypeError):
                self.schedule(**changes)

    def test_missing_development_source_task_is_rejected(self):
        self.task["id"] = "X01"
        self.freeze()
        with self.assertRaisesRegex(ValueError, "missing development task"):
            self.schedule()

    def test_missing_development_prompt_is_rejected_by_renderer(self):
        self.prompts = [task for task in self.prompts if task["id"] != "D01"]
        self.freeze()
        with self.assertRaisesRegex(ValueError, "episode count"):
            self.schedule()

    def test_wrong_development_split_is_rejected(self):
        transfer = next(task for task in self.tasks if task["id"] == "T01")
        self.task["split"], transfer["split"] = transfer["split"], self.task["split"]
        self.freeze()
        with self.assertRaisesRegex(ValueError, "development split"):
            self.schedule()

    def test_missing_duplicate_and_invalid_cases_are_rejected(self):
        original = copy.deepcopy(self.task)
        for name, cases in (
            ("public_cases", []), ("scoring_cases", []),
            ("public_cases", original["public_cases"][:1]),
            ("public_cases", original["public_cases"] * 2),
            ("scoring_cases", original["scoring_cases"] * 2),
            ("scoring_cases", [{**original["scoring_cases"][0], "id": "P01"}]),
            ("scoring_cases", [{**original["scoring_cases"][0], "id": "bad case"}]),
        ):
            self.task.clear()
            self.task.update(copy.deepcopy(original))
            self.task[name] = copy.deepcopy(cases)
            self.freeze()
            with self.subTest(name=name, cases=cases), self.assertRaises(ValueError):
                self.schedule()

    def test_source_reason_required_but_not_projected(self):
        for name in ("public_cases", "scoring_cases"):
            saved = self.task[name][0].pop("reason")
            self.freeze()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unexpected fields"):
                self.schedule()
            self.task[name][0]["reason"] = saved
        self.freeze()
        self.assertEqual(self.schedule()["scheduled_episodes"], 8)

    def test_malformed_renderer_coverage_rejected(self):
        jobs = episode_jobs(self.raw["supervision_raw"], self.raw["prompts_raw"])
        selected = [job for job in jobs if job["task_id"].startswith("D")]
        invalid = [
            selected[:-1], selected + selected[:1],
            selected[:-1] + selected[:1],
            selected[:-1] + [{**selected[-1], "arm": "lora"}],
            selected[:-1] + [{**selected[-1], "task_id": "D05"}],
        ]
        for rows in invalid:
            with self.subTest(rows=[(row["task_id"], row["arm"]) for row in rows]):
                with mock.patch.object(self.scheduler, "episode_jobs", return_value=rows):
                    with self.assertRaisesRegex(ValueError, "development episode coverage"):
                        self.schedule()

    def test_collision_between_call_and_attempt_ids_is_rejected(self):
        def colliding_identity(value):
            if type(value) is dict and value.get("kind") in (
                "LORA_DEVELOPMENT_GENERATION_ID", "LORA_DEVELOPMENT_PUBLIC_ATTEMPT"
            ):
                return "0" * 64
            return digest(value)

        with mock.patch.object(self.scheduler, "digest", side_effect=colliding_identity):
            with self.assertRaisesRegex(ValueError, "duplicate logical identity"):
                self.schedule()

    def test_returned_objects_have_no_cross_arm_or_future_aliases(self):
        before = copy.deepcopy(self.raw)
        result = self.schedule()
        pristine = copy.deepcopy(result)
        result["episodes"][0]["messages"][0]["content"] = "edited"
        result["episodes"][0]["final_cases"][0]["input_sha256"] = "edited"
        result["data_sha256"].clear()
        self.assertEqual(result["episodes"][1], pristine["episodes"][1])
        self.assertEqual(self.schedule(), pristine)
        self.assertEqual(self.raw, before)
        self.assertEqual(pristine["data_sha256"], PINS)

    def test_repeated_planning_is_canonical_and_inputs_unchanged(self):
        tasks = copy.deepcopy(self.tasks)
        supervision = copy.deepcopy(self.supervision)
        prompts = copy.deepcopy(self.prompts)
        before = dict(self.raw)
        first = self.schedule()
        self.assertEqual(encoded(first), encoded(self.schedule()))
        self.assertEqual(self.tasks, tasks)
        self.assertEqual(self.supervision, supervision)
        self.assertEqual(self.prompts, prompts)
        self.assertEqual(self.raw, before)

    def test_scoring_fixture_helpers_remain_unmodified(self):
        fresh = scoring_fixtures.synthetic_tasks()
        self.assertTrue(all("spec" not in task and "initial_witness" not in task
                            for task in fresh))
        self.assertTrue(all(task["reference_source"] == "REFERENCE_MUST_NOT_LEAK"
                            for task in fresh))
        with mock.patch.dict(PINS, {"prompts_v1.json": "0" * 64}):
            with self.assertRaises(ValueError):
                self.schedule()
        self.assertEqual(self.schedule()["data_sha256"], PINS)

    def test_cold_import_and_planning_load_no_model_or_executor(self):
        script = """
import sys
assert not any(name in sys.modules for name in ('torch', 'transformers'))
import development_schedule
assert not any(name in sys.modules for name in (
    'torch', 'transformers', 'generation_core', 'observed_generation',
    'candidate_execution_controller', 'candidate_execution_worker',
    'development_final_controller', 'development_final_admission'))
print('cold schedule import: no model or execution modules')
"""
        result = subprocess.run(
            [sys.executable, "-B", "-W", "error", "-c", script],
            cwd=Path(__file__).parent, capture_output=True, text=True,
            timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "cold schedule import: no model or execution modules")
        before = set(sys.modules)
        with mock.patch("builtins.exec", side_effect=AssertionError("exec")), \
                mock.patch("builtins.eval", side_effect=AssertionError("eval")), \
                mock.patch("builtins.open", side_effect=AssertionError("file read")):
            self.assertEqual(self.schedule()["scheduled_generation_calls"], 16)
        self.assertFalse({"torch", "transformers"} & (set(sys.modules) - before))


if __name__ == "__main__":
    unittest.main()
