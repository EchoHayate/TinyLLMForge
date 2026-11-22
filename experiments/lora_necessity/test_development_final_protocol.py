import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import candidate_execution_protocol
from validation_contract import digest, encoded


def request():
    return {
        "kind": "LORA_DEVELOPMENT_FINAL_REQUEST", "version": 1, "round": 2,
        "run_id": "a" * 32, "attempt_id": "b" * 32,
        "generation_call_id": "c" * 32, "generation_sha256": "d" * 64,
        "case_set_sha256": "e" * 64, "task_id": "D01", "arm": "base",
        "case_group": "scoring", "case_id": "C01", "entry_point": "identity",
        "source": "def identity(value):\n    return value\n", "args": [[1, 2]],
    }


class DevelopmentFinalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(
            Path(__file__).with_name("development_final_protocol.py").is_file(),
            "development final protocol not implemented",
        )
        self.module = importlib.import_module("development_final_protocol")
        self.request = request()

    def prepare(self, **changes):
        return self.module.prepare_request({**self.request, **changes})

    def test_canonical_complete_digest_and_answer_free_payload(self):
        original = copy.deepcopy(self.request)
        request_sha256, payload = self.prepare()
        canonical = json.dumps(self.request, sort_keys=True, separators=(",", ":"),
                               allow_nan=False).encode("ascii")
        self.assertIs(type(request_sha256), str)
        self.assertEqual(request_sha256, hashlib.sha256(canonical).hexdigest())
        self.assertEqual(request_sha256, digest(self.request))
        self.assertIs(type(payload), bytes)
        self.assertEqual(payload, encoded({
            name: self.request[name] for name in ("source", "entry_point", "args")
        }) + b"\n")
        self.assertEqual(set(json.loads(payload)), {"source", "entry_point", "args"})
        self.assertEqual(self.request, original)

    def test_all_variable_identity_fields_are_digest_bound(self):
        original_digest, original_payload = self.prepare()
        for name, value in (
            ("run_id", "f" * 32), ("attempt_id", "f" * 32),
            ("generation_call_id", "f" * 32), ("generation_sha256", "f" * 64),
            ("case_set_sha256", "f" * 64), ("task_id", "D04"),
            ("arm", "text"), ("case_id", "C02"),
        ):
            with self.subTest(name=name):
                changed = {**self.request, name: value}
                result = self.module.prepare_request(changed)
                self.assertEqual(result, (digest(changed), original_payload))
                self.assertNotEqual(result[0], original_digest)
        changed = {**self.request, "case_group": "public", "case_id": "P01"}
        self.assertEqual(self.module.prepare_request(changed),
                         (digest(changed), original_payload))
        self.assertNotEqual(digest(changed), original_digest)

    def test_source_entry_and_args_are_digest_and_payload_bound(self):
        original_digest, original_payload = self.prepare()
        for changes in (
            {"source": self.request["source"] + "\n"},
            {"args": [[1, 3]]},
            {"entry_point": "other",
             "source": self.request["source"].replace("identity", "other")},
        ):
            with self.subTest(changes=changes):
                changed = {**self.request, **changes}
                request_sha256, payload = self.module.prepare_request(changed)
                self.assertEqual(request_sha256, digest(changed))
                self.assertNotEqual(request_sha256, original_digest)
                self.assertNotEqual(payload, original_payload)
                self.assertEqual(json.loads(payload), {
                    name: changed[name] for name in ("source", "entry_point", "args")
                })

    def test_input_mutation_cannot_change_returned_bytes_or_digest(self):
        original = copy.deepcopy(self.request)
        result = self.module.prepare_request(self.request)
        self.request["args"][0].append({"later": [True]})
        self.request["source"] += "\n"
        self.request["generation_sha256"] = "f" * 64
        self.assertEqual(result, (
            digest(original),
            encoded({name: original[name] for name in ("source", "entry_point", "args")})
            + b"\n",
        ))
        self.assertNotEqual(result, self.module.prepare_request(self.request))

    def test_canonical_order_and_primitive_values_are_preserved(self):
        values = [None, True, False, 0, -1, 1.0, "é", {"second": [2], "first": 1}]
        forward = {**self.request, "args": values}
        reordered = dict(reversed(list(forward.items())))
        reordered["args"] = values[:-1] + [{"first": 1, "second": [2]}]
        self.assertEqual(self.module.prepare_request(forward),
                         self.module.prepare_request(reordered))
        self.assertEqual(json.loads(self.module.prepare_request(forward)[1])["args"], values)
        digests = {self.prepare(args=[value])[0] for value in (True, 1, 1.0, "1")}
        self.assertEqual(len(digests), 4)

    def test_exact_request_shape(self):
        for value in (None, [], (), "", 1, True, {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.module.prepare_request(value)
        for name in self.request:
            changed = dict(self.request)
            del changed[name]
            with self.subTest(missing=name), self.assertRaises(ValueError):
                self.module.prepare_request(changed)
        for name in ("expected", "reference_source", "status", "request_sha256"):
            with self.subTest(extra=name), self.assertRaises(ValueError):
                self.prepare(**{name: "not permitted"})

    def test_kind_version_and_round_are_exact_primitives(self):
        for name, values in (
            ("kind", (None, True, 1, [], {}, "", "LORA_FINAL_REQUEST")),
            ("version", (True, False, 1.0, "1", None, [], {}, 0, 2)),
            ("round", (True, False, 2.0, "2", None, [], {}, 0, 1, 3)),
        ):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.prepare(**{name: value})

    def test_hex_identities_require_exact_width_and_lowercase_ascii(self):
        for name, width in (
            ("run_id", 32), ("attempt_id", 32), ("generation_call_id", 32),
            ("generation_sha256", 64), ("case_set_sha256", 64),
        ):
            self.prepare(**{name: "0123456789abcdef" * (width // 16)})
            for value in (None, True, 1, [], {}, "a" * (width - 1),
                          "a" * (width + 1), "A" * width, "g" * width,
                          "a" * width + "\n", "é" * width):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.prepare(**{name: value})

    def test_only_four_development_tasks_and_two_arms(self):
        for task_id in ("D01", "D02", "D03", "D04"):
            for arm in ("base", "text"):
                with self.subTest(task_id=task_id, arm=arm):
                    changed = {**self.request, "task_id": task_id, "arm": arm}
                    self.assertEqual(self.module.prepare_request(changed)[0], digest(changed))
        for task_id in ("D00", "D05", "D1", "d01", "T01", "R01", "S01",
                        "D01\n", "", None, True, 1, [], {}):
            with self.subTest(task_id=task_id), self.assertRaises(ValueError):
                self.prepare(task_id=task_id)
        for arm in ("lora", "lora_text", "BASE", "", None, True, 1, [], {}):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                self.prepare(arm=arm)

    def test_public_and_scoring_case_domains_are_separate(self):
        for case_id in ("P01", "P02"):
            self.assertEqual(json.loads(self.prepare(
                case_group="public", case_id=case_id)[1])["args"], self.request["args"])
            with self.subTest(case_id=case_id), self.assertRaises(ValueError):
                self.prepare(case_group="scoring", case_id=case_id)
        for case_id in ("C01", "P03", "A", "a_0-Z", "A" * 32):
            self.assertEqual(self.prepare(case_id=case_id)[0],
                             digest({**self.request, "case_id": case_id}))
            with self.subTest(case_id=case_id), self.assertRaises(ValueError):
                self.prepare(case_group="public", case_id=case_id)
        for group in ("hidden", "PUBLIC", "", None, True, 1, [], {}):
            with self.subTest(group=group), self.assertRaises(ValueError):
                self.prepare(case_group=group)

    def test_case_id_lexical_and_type_limits(self):
        for case_id in ("", "A" * 33, "1A", "_A", "-A", "A.B", "A B",
                        "A\n", "é", "Aé", "Ａ", None, True, 1, [], {}):
            with self.subTest(case_id=case_id), self.assertRaises(ValueError):
                self.prepare(case_id=case_id)

    def test_args_require_exact_json_values_and_string_keys(self):
        for args in (None, {}, (), "args", True, 0, [()], [set()], [object()],
                     [b"bytes"], [bytearray()], [{1: "value"}],
                     [{"nested": {"bad": object()}}]):
            with self.subTest(kind=type(args).__name__), self.assertRaises(ValueError):
                self.prepare(args=args)

    def test_nonfinite_and_unencodable_numbers_are_value_errors(self):
        for value in (float("nan"), float("inf"), float("-inf"), 10 ** 5000):
            with self.subTest(kind=type(value).__name__), self.assertRaises(ValueError):
                self.prepare(args=[{"nested": value}])

    def test_hostile_builtin_subclasses_and_objects_have_no_hooks_called(self):
        calls = []

        def hook(*arguments, **options):
            calls.append("hook")
            raise AssertionError("hostile hook invoked")

        hostile_methods = {
            "__eq__": hook, "__ne__": hook, "__iter__": hook, "__len__": hook,
            "__getitem__": hook, "__repr__": hook, "__str__": hook,
            "__int__": hook, "__float__": hook, "items": hook, "keys": hook,
            "encode": hook, "isidentifier": hook,
        }
        hostile_values = []
        for builtin, value in ((dict, {}), (list, []), (str, "value"), (int, 1),
                               (float, 1.0)):
            hostile_type = type("Hostile" + builtin.__name__, (builtin,), hostile_methods)
            hostile_values.append(hostile_type(value))
        hostile_values.append(type("HostileObject", (), hostile_methods)())
        for ordinal, value in enumerate(hostile_values):
            requests = [value, {**self.request, "args": [value]}]
            requests.extend({**self.request, name: value} for name in self.request)
            for changed in requests:
                with self.subTest(ordinal=ordinal):
                    with mock.patch.object(self.module, "digest", side_effect=hook):
                        with self.assertRaises(ValueError):
                            self.module.prepare_request(changed)
                    self.assertEqual(calls, [])
        hostile_key_type = type("HostileKey", (str,), {
            "__eq__": hook, "__hash__": str.__hash__, "__str__": hook,
        })
        for changed in (
            {hostile_key_type("extra"): 1},
            {**self.request, "args": [{hostile_key_type("nested"): 1}]},
        ):
            with self.assertRaises(ValueError):
                self.module.prepare_request(changed)
            self.assertEqual(calls, [])

    def test_custom_metaclass_comparison_cannot_run_before_type_rejection(self):
        calls = []

        class HostileMeta(type):
            def __eq__(cls, other):
                calls.append("metaclass equality")
                raise AssertionError("metaclass hook invoked")

        class HostileObject(metaclass=HostileMeta):
            pass

        class HostileList(list, metaclass=HostileMeta):
            pass

        for value in (HostileObject(), HostileList()):
            for changed in (value, {**self.request, "args": [value]}):
                with self.assertRaises(ValueError):
                    self.module.prepare_request(changed)
                self.assertEqual(calls, [])

    def test_source_preparation_cannot_execute_body_or_dynamic_calls(self):
        for source in (
            "def identity(value):\n    while True:\n        pass\n",
            "def identity(value):\n    raise RuntimeError('not executed')\n",
            "def identity(value):\n    return __import__('os').getpid()\n",
            "def identity(value):\n    return open('not-opened')\n",
        ):
            with self.subTest(source=source):
                with mock.patch("builtins.exec", side_effect=AssertionError("exec")):
                    with mock.patch("builtins.eval", side_effect=AssertionError("eval")):
                        with mock.patch("builtins.open", side_effect=AssertionError("open")):
                            result = self.prepare(source=source)
                self.assertEqual(json.loads(result[1])["source"], source)
                self.assertEqual(len(result), 2)

    def test_source_syntax_and_entry_validation(self):
        original = self.request["source"]
        for source in (
            "", "```python\n" + original + "```", "return 0", "import os\n" + original,
            original + "identity(1)\n", original + "\ndef other():\n    pass\n",
            original.replace("identity", "other"), "@decorator\n" + original,
            original.replace("def ", "async def "), original.replace("):", ") -> int:"),
            original.replace("identity(", "identity[T]("),
            original.replace("return value", "import os\n    return value"),
            original.replace("return value", "def nested():\n        pass"),
            original.replace("return value", "class Nested:\n        pass"),
            original.replace("return value", "break"), original + "\0",
            original + "(" * 1000, None, True, 1, b"source", [],
        ):
            with self.subTest(source=repr(source)[:80]), self.assertRaises(ValueError):
                self.prepare(source=source)
        for entry in ("", "other", "bad-name", "1entry", None, True, 1, [], {}):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                self.prepare(entry_point=entry)

    def test_defaults_and_annotations_are_rejected_without_evaluation(self):
        for signature in (
            "value=0", "value=None", "value=1 / 0", "value=0, /",
            "*, value=0", "*, value=None", "*, required, value=1 / 0",
            "value: int", "value: None", "value: 1 / 0", "value: int, /",
            "*, value: int", "*values: int", "**options: int",
        ):
            with self.subTest(signature=signature), self.assertRaises(ValueError):
                self.prepare(source=f"def identity({signature}):\n    return 0\n")

    def test_required_argument_forms_are_syntax_only(self):
        for signature in ("value, /", "*, value", "*values", "**options",
                          "value, /, other, *values, required, **options"):
            source = f"def identity({signature}):\n    return 0\n"
            with self.subTest(signature=signature):
                self.assertEqual(json.loads(self.prepare(source=source)[1])["source"], source)

    def test_source_exact_utf8_byte_limit(self):
        prefix = self.request["source"] + "#"
        for character in ("x", "é"):
            remaining = 65536 - len(prefix.encode("utf-8"))
            width = len(character.encode("utf-8"))
            source = prefix + character * (remaining // width) + "x" * (remaining % width)
            with self.subTest(character=character):
                self.assertEqual(len(source.encode("utf-8")), 65536)
                self.assertEqual(json.loads(self.prepare(source=source)[1])["source"], source)
                with self.assertRaises(ValueError):
                    self.prepare(source=source + "x")

    def test_surrogate_source_and_identity_are_value_errors(self):
        for name in ("source", "entry_point", "kind", "run_id", "attempt_id",
                     "generation_call_id", "generation_sha256", "case_set_sha256",
                     "task_id", "arm", "case_group", "case_id"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.prepare(**{name: self.request[name] + "\ud800"})

    def test_payload_exact_byte_limit_includes_newline(self):
        overhead = len(self.prepare(args=[""])[1])
        value = "x" * (1048576 - overhead)
        result = self.prepare(args=[value])
        self.assertEqual(len(result[1]), 1048576)
        self.assertTrue(result[1].endswith(b"\n"))
        self.assertEqual(json.loads(result[1])["args"], [value])
        with self.assertRaises(ValueError):
            self.prepare(args=[value + "x"])
        with self.assertRaises(ValueError):
            self.prepare(args=["é" * 200000])

    def test_full_request_budget_precedes_metadata_field_comparisons(self):
        changed = {**self.request, "extra": ""}
        changed["extra"] = "x" * (1052672 - len(encoded(changed)))
        self.assertEqual(len(encoded(changed)), 1052672)
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            self.module.prepare_request(changed)
        changed["extra"] += "x"
        with self.assertRaisesRegex(ValueError, "JSON byte budget"):
            self.module.prepare_request(changed)
        with self.assertRaisesRegex(ValueError, "JSON byte budget"):
            self.prepare(generation_sha256="x" * 1052672)

    def test_both_full_and_payload_budgets_use_sealed_helper(self):
        self.assertIs(self.module._source, candidate_execution_protocol._source)
        self.assertIs(self.module._json_budget, candidate_execution_protocol._json_budget)
        with mock.patch.object(self.module, "_json_budget",
                               wraps=self.module._json_budget) as budget:
            result = self.prepare()
        self.assertEqual(budget.call_count, 2)
        self.assertEqual(budget.call_args_list[0], mock.call(self.request, 1052672))
        self.assertEqual(budget.call_args_list[1], mock.call({
            name: self.request[name] for name in ("source", "entry_point", "args")
        }, 1048575))
        self.assertEqual(result[0], digest(self.request))

    def test_cycles_depth_and_shared_expansion_raise_value_error(self):
        cycle = []
        cycle.append(cycle)
        mapping = {}
        mapping["cycle"] = mapping
        deep = []
        for ordinal in range(2000):
            deep = [deep]
        shared = ["x" * 65536]
        for ordinal in range(10):
            shared = [shared, shared]
        for ordinal, args in enumerate((cycle, [mapping], deep, shared)):
            with self.subTest(ordinal=ordinal), self.assertRaises(ValueError):
                self.prepare(args=args)

    def test_full_budget_precedes_source_and_payload_budget_follows(self):
        with mock.patch.object(self.module, "_source") as source:
            with self.assertRaises(ValueError):
                self.prepare(args=["x" * 1052672])
            source.assert_not_called()
        with mock.patch.object(self.module, "_source",
                               wraps=self.module._source) as source:
            with self.assertRaises(ValueError):
                self.prepare(args=["x" * 1048576])
            source.assert_called_once()

    def test_old_and_final_domains_are_mutually_rejected(self):
        old = {
            name: self.request[name] for name in
            ("version", "round", "attempt_id", "task_id", "arm",
             "case_id", "entry_point", "source", "args")
        }
        old.update(round=1, case_id="P01")
        expected = (digest(old), encoded({
            name: old[name] for name in ("source", "entry_point", "args")
        }) + b"\n")
        self.assertEqual(candidate_execution_protocol.prepare_request(old), expected)
        with self.assertRaises(ValueError):
            self.module.prepare_request(old)
        for final in (self.request, {**self.request, "case_group": "public", "case_id": "P01"}):
            with self.assertRaises(ValueError):
                candidate_execution_protocol.prepare_request(final)
        for changes in ({"round": 2}, {"case_id": "C01"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                candidate_execution_protocol.prepare_request({**old, **changes})
        with mock.patch.object(candidate_execution_protocol, "prepare_request",
                               side_effect=AssertionError("round-one path used")):
            self.assertEqual(self.prepare()[0], digest(self.request))
        self.assertEqual(candidate_execution_protocol.prepare_request(old), expected)

    def test_preparation_does_not_import_model_or_execution_modules(self):
        roots = {"torch", "transformers", "tokenizers", "huggingface_hub",
                 "candidate_execution_admission", "candidate_execution_controller",
                 "candidate_execution_worker"}
        before = {name for name in sys.modules if name.split(".")[0] in roots}
        self.prepare()
        self.assertEqual({name for name in sys.modules if name.split(".")[0] in roots},
                         before)


if __name__ == "__main__":
    unittest.main()
