import copy
import hashlib
import json
import sys
import unittest
from unittest import mock

from development_final_protocol import prepare_request
from development_final_scoring import build_final_plan, score_final
from pilot_protocol import FLAGS, SPLITS, build_manifest
from validation_contract import PINS, _tasks, digest, encoded


CANDIDATE = "def toy(value):\n    return value\n"


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def synthetic_tasks():
    tasks = []
    for split, prefix in (
        ("supervision", "S"), ("development", "D"),
        ("transfer", "T"), ("retention", "R"),
    ):
        for ordinal in range(1, SPLITS[split] + 1):
            task_id = f"{prefix}{ordinal:02d}"
            tasks.append({
                "id": task_id, "split": split,
                "family": f"{'retention' if prefix == 'R' else 'family'}{ordinal % 4}",
                "template_id": f"synthetic{task_id}", "entry_point": "toy",
                "initial_source": CANDIDATE,
                "reference_source": "REFERENCE_MUST_NOT_LEAK",
                "public_cases": [
                    {"id": "P02", "args": [2],
                     "expected": {"kind": "return", "value": 2},
                     "reason": "REASON_MUST_NOT_LEAK_second"},
                    {"id": "P01", "args": [1],
                     "expected": {"kind": "return", "value": 1},
                     "reason": "REASON_MUST_NOT_LEAK_first"},
                ],
                "scoring_cases": [
                    {"id": "H01", "args": [3],
                     "expected": {"kind": "return", "value": 3},
                     "reason": "REASON_MUST_NOT_LEAK_scoring"},
                ],
            })
    return tasks


def frozen_bytes(tasks):
    metadata = [
        {**{name: task[name] for name in ("id", "split", "family", "template_id")},
         "content_sha256": digest(task)}
        for task in tasks
    ]
    source_raw = encoded({"tasks": tasks})
    manifest_raw = encoded({"cpu_manifest": build_manifest(metadata)})
    return source_raw, manifest_raw


class FinalScoringTests(unittest.TestCase):
    def setUp(self):
        self.tasks = synthetic_tasks()
        self.task = next(task for task in self.tasks if task["id"] == "D01")
        self.pin_patch = mock.patch.dict(PINS, {})
        self.pin_patch.start()
        self.addCleanup(self.pin_patch.stop)
        self.freeze()
        self.identity = {
            "run_id": "1" * 32, "task_id": "D01", "arm": "base",
            "generation_call_id": "2" * 32,
        }
        self.generation()

    def freeze(self):
        self.source_raw, self.manifest_raw = frozen_bytes(self.tasks)
        PINS.update({
            "task_sources_v1.json": sha256(self.source_raw),
            "manifest_v1.json": sha256(self.manifest_raw),
        })

    def generation(self, candidate=CANDIDATE, **changes):
        record = {
            "kind": "LORA_GREEDY_CANDIDATE_UNATTESTED", "candidate": candidate,
            "candidate_sha256": sha256(candidate.encode("utf-8")),
        }
        record.update(changes)
        self.generation_bytes(encoded(record))

    def generation_bytes(self, raw):
        self.identity.update(generation_raw=raw, expected_generation_sha256=sha256(raw))

    def plan(self, **changes):
        return build_final_plan(self.source_raw, self.manifest_raw,
                                **{**self.identity, **changes})

    def score(self, receipts=None, **changes):
        return score_final(self.source_raw, self.manifest_raw,
                           encoded([] if receipts is None else receipts),
                           **{**self.identity, **changes})

    def receipts(self):
        plan = self.plan()
        return [
            {"request_sha256": digest(request), "status": "completed",
             "outcome": copy.deepcopy(expected["expected"])}
            for request, expected in zip(plan["requests"], plan["expected_outcomes"])
        ]

    def test_real_frozen_loader_and_complete_ordered_plan(self):
        self.assertEqual(len(_tasks(self.source_raw, self.manifest_raw)), 44)
        plan = self.plan()
        self.assertEqual(set(plan), {
            "kind", "version", "run_id", "task_id", "arm", "generation_call_id",
            "generation_sha256", "candidate_sha256", "case_set_sha256", "task_sha256",
            "source_sha256", "manifest_sha256", "candidate_valid", "scheduled_cases",
            "requests", "expected_outcomes",
        })
        self.assertEqual(plan["kind"], "LORA_DEVELOPMENT_FINAL_PLAN")
        self.assertEqual(plan["version"], 1)
        self.assertIs(plan["candidate_valid"], True)
        self.assertEqual(plan["scheduled_cases"], 3)
        self.assertEqual([item["case_id"] for item in plan["requests"]],
                         ["P01", "P02", "H01"])
        self.assertEqual([item["case_group"] for item in plan["requests"]],
                         ["public", "public", "scoring"])

    def test_case_set_and_attempt_hashes_independently_bound(self):
        plan = self.plan()
        cases = []
        for group, name in (("public", "public_cases"), ("scoring", "scoring_cases")):
            for case in sorted(self.task[name], key=lambda item: item["id"]):
                cases.append({
                    "case_group": group, "case_id": case["id"],
                    "args": case["args"], "expected": case["expected"],
                })
        expected_set = digest({
            "kind": "LORA_DEVELOPMENT_FINAL_CASE_SET", "version": 1,
            "source_sha256": sha256(self.source_raw),
            "manifest_sha256": sha256(self.manifest_raw),
            "task_id": "D01", "task_sha256": digest(self.task),
            "entry_point": "toy", "cases": cases,
        })
        self.assertEqual(plan["case_set_sha256"], expected_set)
        for request, case in zip(plan["requests"], cases):
            attempt = {
                "kind": "LORA_DEVELOPMENT_FINAL_CASE_ATTEMPT", "version": 1,
                "run_id": self.identity["run_id"], "task_id": "D01", "arm": "base",
                "generation_call_id": self.identity["generation_call_id"],
                "generation_sha256": sha256(self.identity["generation_raw"]),
                "case_set_sha256": expected_set, "case_group": case["case_group"],
                "case_id": case["case_id"],
            }
            self.assertEqual(request["attempt_id"], digest(attempt)[:32])
            self.assertEqual(request["round"], 2)
        self.assertEqual(len({item["attempt_id"] for item in plan["requests"]}), 3)
        self.assertEqual(plan, self.plan())

    def test_frozen_source_and_manifest_pin_mismatches(self):
        for name in ("task_sources_v1.json", "manifest_v1.json"):
            with self.subTest(name=name), mock.patch.dict(PINS, {name: "0" * 64}):
                with self.assertRaisesRegex(ValueError, "frozen bytes changed"):
                    self.plan()

    def test_repinning_source_without_manifest_metadata_fails(self):
        self.task["scoring_cases"][0]["args"] = [7]
        self.source_raw = encoded({"tasks": self.tasks})
        PINS["task_sources_v1.json"] = sha256(self.source_raw)
        with self.assertRaisesRegex(ValueError, "task metadata binding"):
            self.plan()

    def test_real_manifest_rejects_duplicate_metadata_and_wrong_split(self):
        source = json.loads(self.source_raw)
        for field, value in (("id", "S01"), ("split", "transfer")):
            changed = copy.deepcopy(source)
            task = next(item for item in changed["tasks"] if item["id"] == "D01")
            task[field] = value
            raw = encoded(changed)
            with self.subTest(field=field), mock.patch.dict(
                PINS, {"task_sources_v1.json": sha256(raw)}
            ):
                with self.assertRaises(ValueError):
                    build_final_plan(raw, self.manifest_raw, **self.identity)

    def test_identity_types_and_domains_fail_before_frozen_loading(self):
        values = {
            "run_id": [None, True, 1, [], {}, "A" * 32, "a" * 31, "a" * 33],
            "generation_call_id": [False, "G" * 32, "b" * 31],
            "expected_generation_sha256": [1, "F" * 64, "f" * 63],
            "task_id": [None, [], 1, "S01", "T01", "R01", "D00", "D05", "d01"],
            "arm": [None, [], 1, "lora", "lora_text", "Base"],
        }
        for name, invalid_values in values.items():
            for value in invalid_values:
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        build_final_plan(b"bad", b"bad",
                                         **{**self.identity, name: value})

    def test_identity_subclasses_rejected_without_hooks(self):
        class Hostile(str):
            def __eq__(self, other):
                raise AssertionError("equality hook")

            def __str__(self):
                raise AssertionError("string hook")

        for name in ("run_id", "task_id", "arm", "generation_call_id",
                     "expected_generation_sha256"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.plan(**{name: Hostile(self.identity[name])})

    def test_all_development_tasks_and_arms(self):
        for task_id in ("D01", "D02", "D03", "D04"):
            for arm in ("base", "text"):
                with self.subTest(task_id=task_id, arm=arm):
                    plan = self.plan(task_id=task_id, arm=arm)
                    self.assertEqual(plan["task_id"], task_id)
                    self.assertEqual(plan["arm"], arm)
                    self.assertEqual(plan["scheduled_cases"], 3)

    def test_unknown_identity_and_edited_plan_rejected(self):
        plan = self.plan()
        plan["requests"].clear()
        plan["expected_outcomes"].clear()
        for changes in ({"plan": plan}, {"round": 1}, {"attempt_id": "a" * 32}):
            with self.subTest(changes=changes):
                with self.assertRaises(TypeError):
                    self.plan(**changes)
                with self.assertRaises(TypeError):
                    self.score(**changes)

    def test_generation_raw_hash_and_candidate_hash(self):
        with self.assertRaisesRegex(ValueError, "generation bytes changed"):
            self.plan(expected_generation_sha256="0" * 64)
        for candidate_hash in ("0" * 64, None, True, [], 3):
            self.generation(candidate_sha256=candidate_hash)
            with self.subTest(candidate_hash=candidate_hash), self.assertRaises(ValueError):
                self.plan()

    def test_generation_object_kind_and_candidate_schema(self):
        record = json.loads(self.identity["generation_raw"])
        invalid = [None, [], True, "candidate", 1]
        for field in ("kind", "candidate", "candidate_sha256"):
            changed = dict(record)
            del changed[field]
            invalid.append(changed)
        for kind in (None, [], {}, 1, True, "LORA_DEVELOPMENT_FINAL_REQUEST"):
            invalid.append({**record, "kind": kind})
        for candidate in (None, True, 2, [], {}):
            invalid.append({**record, "candidate": candidate})
        for item in invalid:
            self.generation_bytes(encoded(item))
            with self.subTest(record=item), self.assertRaises(ValueError):
                self.plan()

    def test_generation_strict_json_and_byte_budget(self):
        for raw in (b"", b"\xff", b"{} trailing", b'{"kind":1,"kind":2}',
                    b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}',
                    b"[" * 1100 + b"]" * 1100):
            self.generation_bytes(raw)
            with self.subTest(raw=raw[:40]), self.assertRaises(ValueError):
                self.plan()
        self.generation(extra="allowed generation evidence")
        raw = self.identity["generation_raw"]
        self.generation_bytes(raw + b" " * (1048576 - len(raw)))
        self.assertTrue(self.plan()["candidate_valid"])
        self.generation_bytes(self.identity["generation_raw"] + b" ")
        with self.assertRaisesRegex(ValueError, "JSON byte budget"):
            self.plan()
        for raw in ("{}", bytearray(b"{}"), memoryview(b"{}")):
            with self.subTest(raw_type=type(raw)), self.assertRaises(ValueError):
                self.plan(generation_raw=raw)

    def test_raw_generation_bytes_preserved_in_binding(self):
        original = self.plan()
        self.generation_bytes(self.identity["generation_raw"] + b"\n")
        changed = self.plan()
        self.assertEqual(changed["candidate_sha256"], original["candidate_sha256"])
        self.assertEqual(changed["case_set_sha256"], original["case_set_sha256"])
        self.assertNotEqual(changed["generation_sha256"], original["generation_sha256"])
        self.assertNotEqual(changed["requests"], original["requests"])

    def test_candidate_utf8_limit_and_surrogate(self):
        for character in (" ", "\u00e9"):
            prefix = 'def toy(value):\n    """'
            suffix = '"""\n    return value\n'
            available = 65536 - len((prefix + suffix).encode("utf-8"))
            padding = character * (available // len(character.encode("utf-8")))
            padding += " " * (available - len(padding.encode("utf-8")))
            candidate = prefix + padding + suffix
            self.assertEqual(len(candidate.encode("utf-8")), 65536)
            self.generation(candidate)
            self.assertTrue(self.plan()["candidate_valid"])
            self.generation(candidate + " ")
            with self.assertRaisesRegex(ValueError, "candidate byte cap"):
                self.plan()
        record = {"kind": "LORA_GREEDY_CANDIDATE_UNATTESTED",
                  "candidate": "\ud800", "candidate_sha256": "0" * 64}
        self.generation_bytes(encoded(record))
        with self.assertRaisesRegex(ValueError, "candidate encoding"):
            self.plan()

    def test_candidate_bytes_are_not_stripped_or_repaired(self):
        candidate = CANDIDATE + "\n\n"
        self.generation(candidate)
        plan = self.plan()
        self.assertEqual(plan["candidate_sha256"], sha256(candidate.encode("utf-8")))
        self.assertTrue(all(request["source"] == candidate for request in plan["requests"]))

    def test_invalid_candidates_preserve_full_denominator(self):
        for candidate in ("", "```python\n" + CANDIDATE + "```",
                          "def toy(other):\n    return other\n",
                          "def other(value):\n    return value\n",
                          "def toy(value=1/0):\n    return value\n",
                          "def toy(value: int):\n    return value\n"):
            self.generation(candidate)
            with self.subTest(candidate=candidate):
                plan = self.plan()
                self.assertIs(plan["candidate_valid"], False)
                self.assertEqual(plan["requests"], [])
                self.assertEqual(plan["scheduled_cases"], 3)
                self.assertEqual(len(plan["expected_outcomes"]), 3)
                score = self.score()
                self.assertEqual(score["status"], "invalid_source")
                self.assertEqual(score["scheduled_cases"], 3)
                self.assertEqual(score["received_cases"], 0)
                self.assertEqual(score["passed_cases"], 0)
                self.assertEqual([row["case_id"] for row in score["cases"]],
                                 ["P01", "P02", "H01"])
                self.assertTrue(all(row["request_sha256"] is None
                                    and row["status"] == "invalid_source"
                                    for row in score["cases"]))
                with self.assertRaises(ValueError):
                    self.score([{"status": "timeout", "request_sha256": "0" * 64}])

    def test_syntax_only_loops_and_raises_never_execute(self):
        before = set(sys.modules)
        for candidate in ("def toy(value):\n    while True:\n        pass\n",
                          "def toy(value):\n    raise RuntimeError('not executed')\n"):
            self.generation(candidate)
            with mock.patch("builtins.exec", side_effect=AssertionError("exec")), \
                    mock.patch("builtins.eval", side_effect=AssertionError("eval")), \
                    mock.patch("builtins.open", side_effect=AssertionError("file read")):
                self.assertTrue(self.plan()["candidate_valid"])
                self.assertEqual(self.score()["status"], "incomplete")
        self.assertFalse({"torch", "transformers"} & (set(sys.modules) - before))

    def test_case_source_schema_requires_reason_and_exact_fields(self):
        original = copy.deepcopy(self.task["scoring_cases"][0])
        invalid = [None, [], {**original, "reason": 1},
                   {**original, "args": {}}, {**original, "unexpected": 0}]
        for name in original:
            invalid.append({key: value for key, value in original.items() if key != name})
        for case in invalid:
            self.task["scoring_cases"] = [case]
            self.freeze()
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.plan()

    def test_case_groups_nonempty_and_valid_identities(self):
        public = copy.deepcopy(self.task["public_cases"])
        scoring = copy.deepcopy(self.task["scoring_cases"])
        for group in ("public_cases", "scoring_cases"):
            for items in ([], None, {}, "cases"):
                self.task["public_cases"] = copy.deepcopy(public)
                self.task["scoring_cases"] = copy.deepcopy(scoring)
                self.task[group] = items
                self.freeze()
                with self.subTest(group=group, items=items), self.assertRaises(ValueError):
                    self.plan()
        self.task["public_cases"] = public
        self.task["scoring_cases"] = scoring
        for case_id in ("", "P01", "P02", "1bad", "bad space", "H" * 33, None, True, []):
            self.task["scoring_cases"][0]["id"] = case_id
            self.freeze()
            with self.subTest(case_id=case_id), self.assertRaises(ValueError):
                self.plan()

    def test_missing_named_case_group_is_validation_error(self):
        for name in ("public_cases", "scoring_cases"):
            saved = self.task.pop(name)
            self.freeze()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "case group"):
                self.plan()
            self.task[name] = saved

    def test_duplicate_attempt_digest_rejected_without_weakening_loader(self):
        def colliding_attempt(value):
            if type(value) is dict and value.get("kind") == "LORA_DEVELOPMENT_FINAL_CASE_ATTEMPT":
                return "0" * 64
            return digest(value)

        with mock.patch("development_final_scoring.digest", side_effect=colliding_attempt):
            with self.assertRaisesRegex(ValueError, "duplicate attempt"):
                self.plan()

    def test_frozen_raw_types_and_budget_are_enforced(self):
        for name in ("source", "manifest"):
            for raw in ("{}", bytearray(b"{}"), memoryview(b"{}"), b" " * 1048577):
                source = raw if name == "source" else self.source_raw
                manifest = raw if name == "manifest" else self.manifest_raw
                with self.subTest(name=name, raw_type=type(raw)):
                    with self.assertRaisesRegex(ValueError, "frozen file bytes"):
                        build_final_plan(source, manifest, **self.identity)

    def test_frozen_source_strict_json_precedes_metadata(self):
        for raw in (b'{"tasks":[],"tasks":[]}', b'{"tasks":NaN}',
                    b'{"tasks":Infinity}', b'{"tasks":1e999}', b"\xff"):
            with self.subTest(raw=raw), mock.patch.dict(
                PINS, {"task_sources_v1.json": sha256(raw)}
            ):
                with self.assertRaises(ValueError):
                    build_final_plan(raw, self.manifest_raw, **self.identity)

    def test_valid_source_requests_still_enforce_final_payload_budget(self):
        self.task["scoring_cases"][0]["args"] = ["x" * 1000000]
        self.freeze()
        self.assertLess(len(self.source_raw), 1048576)
        self.generation(CANDIDATE + " " * (65536 - len(CANDIDATE)))
        with self.assertRaisesRegex(ValueError, "JSON byte budget"):
            self.plan()

    def test_invalid_source_does_not_bypass_expected_outcome_validation(self):
        self.generation("")
        self.task["scoring_cases"][0]["expected"] = {"kind": "raises", "type": "TypeError"}
        self.freeze()
        with self.assertRaisesRegex(ValueError, "unsupported outcome"):
            self.score()

    def test_return_mapping_order_is_canonical_but_signed_zero_is_exact(self):
        self.task["scoring_cases"][0]["expected"] = {
            "kind": "return", "value": {"first": 1, "second": 2},
        }
        self.freeze()
        receipts = self.receipts()
        receipts[-1]["outcome"]["value"] = {"second": 2, "first": 1}
        self.assertEqual(self.score(receipts)["status"], "pass")
        self.task["scoring_cases"][0]["expected"] = {"kind": "return", "value": 0.0}
        self.freeze()
        receipts = self.receipts()
        receipts[-1]["outcome"]["value"] = -0.0
        self.assertEqual(self.score(receipts)["cases"][-1]["status"], "mismatch")

    def test_case_duplicates_public_coverage_and_cross_group(self):
        public = copy.deepcopy(self.task["public_cases"])
        scoring = copy.deepcopy(self.task["scoring_cases"])
        for public_cases, scoring_cases in (
            (public[:1], scoring), (public + public[:1], scoring),
            (public, scoring * 2),
            ([{**public[0], "id": "H01"}, public[1]], scoring),
        ):
            self.task.update(public_cases=public_cases, scoring_cases=scoring_cases)
            self.freeze()
            with self.subTest(public=public_cases, scoring=scoring_cases):
                with self.assertRaises(ValueError):
                    self.plan()

    def test_permuted_cases_sorted_public_before_scoring(self):
        self.task["scoring_cases"].extend([
            {"id": case_id, "args": [4], "expected": {"kind": "return", "value": 4},
             "reason": "synthetic"} for case_id in ("z_last", "A-first")
        ])
        self.freeze()
        original = self.plan()
        self.task["public_cases"].reverse()
        self.task["scoring_cases"].reverse()
        self.freeze()
        changed = self.plan()
        expected_order = ["P01", "P02", "A-first", "H01", "z_last"]
        for plan in (original, changed):
            self.assertEqual([request["case_id"] for request in plan["requests"]],
                             expected_order)
        self.assertNotEqual(changed["case_set_sha256"], original["case_set_sha256"])
        self.assertEqual(changed["expected_outcomes"], original["expected_outcomes"])

    def test_source_args_expected_and_metadata_change_bound_digests(self):
        original_task = copy.deepcopy(self.task)
        original = self.plan()
        for change in ("args", "expected", "reason", "template_id", "reference_source"):
            self.task.clear()
            self.task.update(copy.deepcopy(original_task))
            if change in ("args", "expected", "reason"):
                self.task["scoring_cases"][0][change] = {
                    "args": [4], "expected": {"kind": "return", "value": 99},
                    "reason": "different synthetic reason",
                }[change]
            else:
                self.task[change] += "_changed"
            self.freeze()
            changed = self.plan()
            with self.subTest(change=change):
                for name in ("source_sha256", "manifest_sha256",
                             "task_sha256", "case_set_sha256"):
                    self.assertNotEqual(changed[name], original[name])
                self.assertNotEqual(changed["requests"][0]["attempt_id"],
                                    original["requests"][0]["attempt_id"])

    def test_episode_identity_changes_attempts_and_rejects_old_receipts(self):
        receipts = self.receipts()
        original = self.plan()
        for changes in ({"run_id": "a" * 32}, {"generation_call_id": "b" * 32},
                        {"task_id": "D02"}, {"arm": "text"}):
            with self.subTest(changes=changes):
                changed = self.plan(**changes)
                self.assertNotEqual(changed["requests"][0]["attempt_id"],
                                    original["requests"][0]["attempt_id"])
                with self.assertRaisesRegex(ValueError, "request binding"):
                    self.score(receipts, **changes)
        self.generation(CANDIDATE + "\n")
        with self.assertRaisesRegex(ValueError, "request binding"):
            self.score(receipts)

    def test_worker_payloads_and_scores_never_leak_expectations(self):
        secret = "SYNTHETIC_EXPECTATION_MUST_NOT_LEAK"
        self.task["scoring_cases"][0]["expected"] = {"kind": "return", "value": secret}
        self.freeze()
        plan = self.plan()
        self.assertIn(secret, encoded(plan).decode("ascii"))
        for request in plan["requests"]:
            request_hash, raw = prepare_request(request)
            self.assertEqual(request_hash, digest(request))
            self.assertEqual(set(json.loads(raw)), {"source", "entry_point", "args"})
            for forbidden in (secret, "expected", "reason", "reference_source",
                              "REFERENCE_MUST_NOT_LEAK", "REASON_MUST_NOT_LEAK"):
                self.assertNotIn(forbidden, raw.decode("ascii"))
                self.assertNotIn(forbidden, encoded(request).decode("ascii"))
        score = self.score(self.receipts())
        for forbidden in (secret, "expected", "reason", "reference_source", "outcome"):
            self.assertNotIn(forbidden, encoded(score).decode("ascii"))

    def test_complete_score_schema_binding_and_flags(self):
        plan = self.plan()
        score = self.score(self.receipts())
        binding_names = ("run_id", "task_id", "arm", "generation_call_id",
                         "generation_sha256", "candidate_sha256", "case_set_sha256",
                         "candidate_valid", "scheduled_cases")
        self.assertEqual(set(score), {
            "kind", "version", *binding_names, "plan_sha256", "status",
            "received_cases", "passed_cases", "cases", *FLAGS,
        })
        self.assertEqual(score["kind"], "LORA_DEVELOPMENT_FINAL_SCORE")
        self.assertEqual(score["version"], 1)
        self.assertEqual(score["status"], "pass")
        self.assertEqual(score["plan_sha256"], digest(plan))
        self.assertEqual(score["received_cases"], 3)
        self.assertEqual(score["passed_cases"], 3)
        for name in binding_names:
            self.assertEqual(score[name], plan[name])
        for flag in FLAGS:
            self.assertIs(score[flag], False)
        for row, request in zip(score["cases"], plan["requests"]):
            self.assertEqual(row, {
                "case_group": request["case_group"], "case_id": request["case_id"],
                "request_sha256": digest(request), "status": "pass",
            })

    def test_mixed_and_all_mismatch_outcomes_do_not_short_circuit(self):
        for count in (1, 3):
            receipts = self.receipts()
            for receipt in receipts[:count]:
                receipt["outcome"]["value"] = -1
            score = self.score(receipts)
            self.assertEqual(score["status"], "fail")
            self.assertEqual(score["received_cases"], 3)
            self.assertEqual(score["passed_cases"], 3 - count)
            self.assertEqual([row["status"] for row in score["cases"]],
                             ["mismatch"] * count + ["pass"] * (3 - count))

    def test_execution_errors_retained_and_not_mismatches(self):
        receipts = self.receipts()
        statuses = ("program_error", "timeout", "output_limit")
        for receipt, status in zip(receipts, statuses):
            receipt["status"] = status
            del receipt["outcome"]
        score = self.score(receipts)
        self.assertEqual(score["status"], "fail")
        self.assertEqual(score["passed_cases"], 0)
        self.assertEqual([row["status"] for row in score["cases"]], list(statuses))

    def test_missing_prefix_rows_preserve_full_denominator(self):
        receipts = self.receipts()
        for count in range(3):
            score = self.score(receipts[:count])
            self.assertEqual(score["status"], "incomplete")
            self.assertEqual(score["scheduled_cases"], 3)
            self.assertEqual(score["received_cases"], count)
            self.assertEqual(score["passed_cases"], count)
            self.assertEqual([row["status"] for row in score["cases"]],
                             ["pass"] * count + ["missing"] * (3 - count))
            self.assertTrue(all(row["request_sha256"] for row in score["cases"]))
        receipts[0]["outcome"]["value"] = -1
        self.assertEqual(self.score(receipts[:1])["status"], "incomplete")

    def test_receipt_reordering_duplicates_extra_and_missing_middle_fail(self):
        receipts = self.receipts()
        invalid = [
            list(reversed(receipts)), receipts[:1] * 2, receipts + receipts[:1],
            [receipts[0], receipts[2]], receipts[1:],
        ]
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.score(rows)

    def test_wrong_digest_is_binding_error_not_mismatch(self):
        for value in ("0" * 64, "", True, 1, None, []):
            receipts = self.receipts()
            receipts[0]["request_sha256"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "request binding"):
                self.score(receipts)

    def test_receipt_statuses_and_fields_are_exact(self):
        original = self.receipts()[0]
        for status in ("admission_rejected", "infrastructure_error", "invalid_source",
                       "missing", "pass", "mismatch", None, True, [], {}):
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.score([{**original, "status": status}])
        invalid = [None, [], 1, {**original, "extra": 1},
                   {**original, "status": "timeout"}]
        for name in original:
            invalid.append({key: value for key, value in original.items() if key != name})
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.score([row])

    def test_valueerror_and_canonical_type_distinctions(self):
        self.task["scoring_cases"][0]["expected"] = {"kind": "raises", "type": "ValueError"}
        self.freeze()
        self.assertEqual(self.score(self.receipts())["status"], "pass")
        receipts = self.receipts()
        receipts[-1]["outcome"] = {"kind": "return", "value": None}
        self.assertEqual(self.score(receipts)["cases"][-1]["status"], "mismatch")
        for expected in (True, 1, 1.0, "1", None, [1], {"value": 1}):
            self.task["scoring_cases"][0]["expected"] = {"kind": "return", "value": expected}
            self.freeze()
            for actual in (True, 1, 1.0, "1", None, [1], {"value": 1}):
                receipts = self.receipts()
                receipts[-1]["outcome"]["value"] = actual
                score = self.score(receipts)
                with self.subTest(expected=expected, actual=actual):
                    self.assertEqual(score["cases"][-1]["status"],
                                     "pass" if encoded(actual) == encoded(expected)
                                     else "mismatch")

    def test_malformed_expected_and_actual_outcomes_rejected(self):
        invalid = [None, [], 1, {}, {"kind": "return"},
                   {"kind": "return", "value": 1, "extra": True},
                   {"kind": "raises", "type": "TypeError"},
                   {"kind": "raises", "type": "ValueError", "value": 1},
                   {"kind": True, "type": "ValueError"}]
        original = copy.deepcopy(self.task["scoring_cases"][0]["expected"])
        receipts = self.receipts()
        for outcome in invalid:
            self.task["scoring_cases"][0]["expected"] = outcome
            self.freeze()
            with self.subTest(expected=outcome), self.assertRaises(ValueError):
                self.plan()
            self.task["scoring_cases"][0]["expected"] = copy.deepcopy(original)
            self.freeze()
            receipts[-1]["outcome"] = outcome
            with self.subTest(actual=outcome), self.assertRaises(ValueError):
                self.score(receipts)

    def test_expected_and_actual_outcome_byte_boundary(self):
        overhead = len(encoded({"kind": "return", "value": ""}))
        for size in (16384, 16385):
            outcome = {"kind": "return", "value": "x" * (size - overhead)}
            self.assertEqual(len(encoded(outcome)), size)
            receipts = self.receipts()
            receipts[-1]["outcome"] = outcome
            if size == 16384:
                self.assertEqual(self.score(receipts)["status"], "fail")
            else:
                with self.assertRaisesRegex(ValueError, "result byte budget"):
                    self.score(receipts)
            self.task["scoring_cases"][0]["expected"] = outcome
            self.freeze()
            if size == 16384:
                self.assertEqual(self.score(self.receipts())["status"], "pass")
            else:
                with self.assertRaisesRegex(ValueError, "result byte budget"):
                    self.plan()

    def test_expected_and_actual_outcome_depth_boundary(self):
        original = copy.deepcopy(self.task["scoring_cases"][0]["expected"])
        for arrays in (127, 128):
            value = 0
            for unused in range(arrays):
                value = [value]
            outcome = {"kind": "return", "value": value}
            receipts = self.receipts()
            receipts[-1]["outcome"] = outcome
            if arrays == 127:
                self.assertEqual(self.score(receipts)["status"], "fail")
            else:
                with self.assertRaisesRegex(ValueError, "result JSON depth"):
                    self.score(receipts)
            self.task["scoring_cases"][0]["expected"] = outcome
            self.freeze()
            if arrays == 127:
                self.assertEqual(self.score(self.receipts())["status"], "pass")
            else:
                with self.assertRaisesRegex(ValueError, "result JSON depth"):
                    self.plan()
            self.task["scoring_cases"][0]["expected"] = copy.deepcopy(original)
            self.freeze()

    def test_receipts_strict_json_raw_types_and_budget(self):
        for raw in (b"", b"\xff", b"{} trailing", b'{"status":1,"status":2}',
                    b'[{"outcome":NaN}]', b"[Infinity]", b"[1e999]",
                    b"[" * 1100 + b"]" * 1100, b"{}", b"null",
                    "[]", bytearray(b"[]"), memoryview(b"[]")):
            with self.subTest(raw_type=type(raw)), self.assertRaises(ValueError):
                score_final(self.source_raw, self.manifest_raw, raw, **self.identity)
        raw = b"[]" + b" " * (1048576 - 2)
        score = score_final(self.source_raw, self.manifest_raw, raw, **self.identity)
        self.assertEqual(score["status"], "incomplete")
        with self.assertRaisesRegex(ValueError, "JSON byte budget"):
            score_final(self.source_raw, self.manifest_raw, raw + b" ", **self.identity)

    def test_returned_data_and_inputs_have_no_mutable_aliases(self):
        self.task["scoring_cases"][0].update(
            args=[{"nested": [1]}], expected={"kind": "return", "value": {"nested": [1]}}
        )
        self.freeze()
        plan = self.plan()
        receipts = self.receipts()
        raw_before = (self.source_raw, self.manifest_raw, copy.deepcopy(self.identity),
                      copy.deepcopy(receipts))
        score = self.score(receipts)
        pristine_plan = copy.deepcopy(plan)
        pristine_score = copy.deepcopy(score)
        plan["requests"][-1]["args"][0]["nested"].append(2)
        plan["expected_outcomes"][-1]["expected"]["value"]["nested"].append(3)
        score["cases"].clear()
        self.assertEqual(self.plan(), pristine_plan)
        self.assertEqual(self.score(receipts), pristine_score)
        self.assertEqual((self.source_raw, self.manifest_raw, self.identity, receipts),
                         raw_before)


if __name__ == "__main__":
    unittest.main()
