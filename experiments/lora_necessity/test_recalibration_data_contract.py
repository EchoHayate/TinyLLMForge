import ast
import copy
import importlib
import json
from pathlib import Path
import unittest


class RecalibrationDataContractTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(
            Path(__file__).with_name("recalibration_data_contract.py").is_file(),
            "recalibration data contract not implemented",
        )
        self.module = importlib.import_module("recalibration_data_contract")
        self.root = Path(__file__).with_name("data")
        self.raw = {
            name: (self.root / "recalibration_v2_r1" / name).read_bytes()
            for name in ("manifest.json", "prompts.json", "oracles.json",
                         "structure.json", "review.md")
        }
        self.raw["supervision_v1.json"] = (
            self.root / "supervision_v1.json").read_bytes()
        self.dataset = self.module.load_dataset(self.raw)

    def test_frozen_dataset_counts_and_no_admission(self):
        self.assertEqual(self.dataset["kind"], "LORA_RECALIBRATION_DATA_STATIC_ONLY")
        self.assertEqual(self.dataset["data_version"], "recalibration_v2_r1")
        self.assertEqual(self.dataset["counts"], {
            "tasks": 12, "sources": 60, "cases": 96, "witnesses": 36,
            "future_dynamic_positions": 480,
        })
        self.assertEqual(len(self.dataset["supervision"]), 8)
        self.assertEqual([task["id"] for task in self.dataset["prompts"]],
                         list(self.module.TASK_IDS))
        for name in ("run_authorized", "benchmark_eligible",
                     "formal_evaluation_ready", "net_benefit_established", "isolated"):
            self.assertIs(self.dataset[name], False)

    def test_every_pin_rejects_changed_or_reformatted_bytes(self):
        for name in self.raw:
            with self.subTest(name=name):
                changed = dict(self.raw)
                changed[name] += b"\n"
                with self.assertRaises(ValueError):
                    self.module.load_dataset(changed)

    def test_raw_input_shape_and_types(self):
        for invalid in (None, [], {}, {**self.raw, "unexpected": b"{}"}):
            with self.subTest(kind=type(invalid).__name__):
                with self.assertRaises(ValueError):
                    self.module.load_dataset(invalid)
        for replacement in ("{}", bytearray(b"{}"), b"\xff", b"{}" * 600000):
            changed = dict(self.raw)
            changed["prompts.json"] = replacement
            with self.subTest(kind=type(replacement).__name__):
                with self.assertRaises(ValueError):
                    self.module.load_dataset(changed)

    def test_returned_objects_do_not_alias_future_loads(self):
        expected = copy.deepcopy(self.dataset)
        self.dataset["prompts"][0]["public_cases"][0]["args"].append("changed")
        self.dataset["file_sha256"].clear()
        self.assertEqual(self.module.load_dataset(self.raw), expected)

    def test_static_checker_rejects_cross_file_and_witness_changes(self):
        mutations = (
            lambda prompt, oracle, metadata: prompt.update(id="C99"),
            lambda prompt, oracle, metadata: oracle.update(entry_point="different"),
            lambda prompt, oracle, metadata: prompt.update(unexpected=True),
            lambda prompt, oracle, metadata: oracle["cases"].reverse(),
            lambda prompt, oracle, metadata: oracle["references"].reverse(),
            lambda prompt, oracle, metadata: oracle["incorrect_programs"].pop(),
            lambda prompt, oracle, metadata: metadata.update(parameters=["different"]),
            lambda prompt, oracle, metadata: oracle["incorrect_programs"][0]["witness"].update(
                case_id="H01"),
            lambda prompt, oracle, metadata: oracle["incorrect_programs"][0]["witness"].update(
                predicted=copy.deepcopy(oracle["cases"][0]["expected"])),
            lambda prompt, oracle, metadata: prompt["public_cases"][0].update(
                expected={"kind": "return", "value": [1, 0, 1, 0, 1]}),
        )
        for ordinal, mutate in enumerate(mutations):
            prompt = copy.deepcopy(self.dataset["prompts"][0])
            oracle = copy.deepcopy(self.dataset["oracles"][0])
            metadata = copy.deepcopy(self.dataset["structure"]["tasks"][0])
            mutate(prompt, oracle, metadata)
            with self.subTest(ordinal=ordinal), self.assertRaises(ValueError):
                self.module.validate_task(prompt, oracle, metadata)

    def test_static_checker_rejects_source_syntax_without_execution(self):
        for source in ("this is not python", "raise RuntimeError('must not run')",
                       "def admit_reservations(capacity, requests=[]):\n    return []\n",
                       "def admit_reservations(capacity, requests):\n    import os\n",
                       "@print('must not run')\ndef admit_reservations(capacity, requests):\n    return []\n"):
            prompt = copy.deepcopy(self.dataset["prompts"][0])
            oracle = copy.deepcopy(self.dataset["oracles"][0])
            oracle["references"][0]["source"] = source
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.module.validate_task(
                    prompt, oracle, self.dataset["structure"]["tasks"][0])

    def test_static_checker_rejects_non_json_and_nonfinite_values(self):
        for invalid in (float("nan"), float("inf"), (1, 2), {1: "value"}):
            oracle = copy.deepcopy(self.dataset["oracles"][0])
            oracle["cases"][0]["args"].append(invalid)
            with self.subTest(kind=type(invalid).__name__), self.assertRaises(ValueError):
                self.module.validate_task(
                    self.dataset["prompts"][0], oracle,
                    self.dataset["structure"]["tasks"][0])

    def test_strict_parser_retains_duplicate_key_rejection(self):
        from validation_contract import parse_json
        for raw in (b'{"id":"C01","id":"C02"}', b"NaN", b"1e999", b"\xff"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_json(raw)
        self.assertNotEqual(json.dumps(True), json.dumps(1))

    def test_raw_keys_and_values_must_be_exact_builtin_types(self):
        class Text(str):
            pass

        class Blob(bytes):
            pass

        class Mapping(dict):
            pass

        changed_key = dict(self.raw)
        changed_key[Text("prompts.json")] = changed_key.pop("prompts.json")
        changed_value = {**self.raw, "prompts.json": Blob(self.raw["prompts.json"])}
        for invalid in (changed_key, changed_value, Mapping(self.raw)):
            with self.subTest(raw_type=type(invalid).__name__,
                              key_type=type(next(reversed(invalid))).__name__):
                with self.assertRaises(ValueError):
                    self.module.load_dataset(invalid)

    def test_missing_each_file_and_equal_json_reformatting_are_rejected(self):
        for name in self.raw:
            changed = dict(self.raw)
            del changed[name]
            with self.subTest(missing=name), self.assertRaises(ValueError):
                self.module.load_dataset(changed)
            if name.endswith(".json"):
                changed = dict(self.raw)
                changed[name] = json.dumps(json.loads(self.raw[name])).encode("utf-8")
                self.assertNotEqual(changed[name], self.raw[name])
                with self.subTest(reformatted=name), self.assertRaises(ValueError):
                    self.module.load_dataset(changed)

    def test_all_snapshot_sections_are_independent_between_loads(self):
        expected = copy.deepcopy(self.dataset)
        for name in ("prompts", "oracles", "structure", "manifest", "supervision",
                     "file_sha256", "counts"):
            self.dataset[name].clear()
        self.assertEqual(self.module.load_dataset(self.raw), expected)

    def test_pins_are_checked_before_any_json_parsing(self):
        changed = dict(self.raw)
        changed["prompts.json"] = b"not json"
        with self.assertRaisesRegex(ValueError, "frozen file changed"):
            self.module.load_dataset(changed)

    def test_module_has_only_static_dependencies_and_no_execution_calls(self):
        tree = ast.parse(Path(self.module.__file__).read_text(encoding="utf-8"))
        dependencies = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                dependencies.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                dependencies.add(node.module)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, ("exec", "eval", "compile", "__import__"))
        self.assertEqual(dependencies, {
            "ast", "hashlib", "math", "pilot_protocol", "validation_contract",
        })


class RecalibrationTaskSchemaTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(
            Path(__file__).with_name("recalibration_data_contract.py").is_file(),
            "recalibration data contract not implemented",
        )
        self.module = importlib.import_module("recalibration_data_contract")
        root = Path(__file__).with_name("data") / "recalibration_v2_r1"
        self.prompts = self.module.parse_json((root / "prompts.json").read_bytes())
        self.oracles = self.module.parse_json((root / "oracles.json").read_bytes())
        self.metadata = self.module.parse_json((root / "structure.json").read_bytes())["tasks"]

    def test_all_frozen_tasks_validate_including_c10_concern(self):
        for prompt, oracle, metadata in zip(self.prompts, self.oracles, self.metadata):
            before = copy.deepcopy((prompt, oracle, metadata))
            with self.subTest(task_id=prompt["id"]):
                try:
                    result = self.module.validate_task(prompt, oracle, metadata)
                except ValueError as error:
                    self.fail(f"frozen task {prompt['id']} rejected: {error}")
                self.assertIsNone(result)
                self.assertEqual((prompt, oracle, metadata), before)
        self.assertTrue(self.metadata[9]["concern"])

    def test_metadata_descriptions_require_nonempty_text(self):
        for name in ("family", "mechanics", "initial_repair_obligation"):
            for invalid in (None, False, 1, [], {}, ""):
                metadata = copy.deepcopy(self.metadata[0])
                metadata[name] = invalid
                with self.subTest(field=name, value=invalid), self.assertRaises(ValueError):
                    self.module.validate_task(self.prompts[0], self.oracles[0], metadata)

    def test_c10_concern_is_required_nonempty_text_and_not_an_arbitrary_extra(self):
        for invalid in (None, False, 1, [], {}, ""):
            metadata = copy.deepcopy(self.metadata[9])
            metadata["concern"] = invalid
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                self.module.validate_task(self.prompts[9], self.oracles[9], metadata)
        metadata = copy.deepcopy(self.metadata[9])
        del metadata["concern"]
        with self.assertRaises(ValueError):
            self.module.validate_task(self.prompts[9], self.oracles[9], metadata)
        metadata = copy.deepcopy(self.metadata[0])
        metadata["concern"] = "Not part of this task's schema"
        with self.assertRaises(ValueError):
            self.module.validate_task(self.prompts[0], self.oracles[0], metadata)

    def test_source_byte_boundary_and_no_function_execution(self):
        source = (
            "def admit_reservations(capacity, requests):\n"
            "    raise RuntimeError('must not execute')\n"
        )
        for content in (source, source + " " * (4096 - len(source.encode("utf-8")))):
            oracle = copy.deepcopy(self.oracles[0])
            oracle["references"][0]["source"] = content
            self.assertIsNone(self.module.validate_task(
                self.prompts[0], oracle, self.metadata[0]))
        oracle["references"][0]["source"] += " "
        with self.assertRaises(ValueError):
            self.module.validate_task(self.prompts[0], oracle, self.metadata[0])
        oracle["references"][0]["source"] = source + "é" * 2048
        with self.assertRaisesRegex(ValueError, "source byte limit"):
            self.module.validate_task(self.prompts[0], oracle, self.metadata[0])

    def test_strict_builtin_json_unicode_depth_and_cycles(self):
        class Text(str):
            pass

        class Number(int):
            pass

        class Sequence(list):
            pass

        class Mapping(dict):
            pass

        nested = []
        for ordinal in range(65):
            nested = [nested]
        cycle = []
        cycle.append(cycle)
        for invalid in (Text("value"), Number(1), Sequence(), Mapping(),
                        b"bytes", {"value"}, object(), "\ud800", {"\ud800": 1},
                        {Text("key"): 1}, float("-inf"), nested, cycle):
            oracle = copy.deepcopy(self.oracles[0])
            oracle["cases"][2]["expected"] = {"kind": "return", "value": invalid}
            with self.subTest(kind=type(invalid).__name__), self.assertRaises(ValueError):
                self.module.validate_task(self.prompts[0], oracle, self.metadata[0])
        for position in range(3):
            for invalid in (None, [], True, "", Mapping()):
                arguments = [self.prompts[0], self.oracles[0], self.metadata[0]]
                arguments[position] = invalid
                with self.subTest(position=position, kind=type(invalid).__name__):
                    with self.assertRaises(ValueError):
                        self.module.validate_task(*arguments)

    def test_bool_and_int_are_distinct_in_public_binding_and_witnesses(self):
        prompt = copy.deepcopy(self.prompts[0])
        oracle = copy.deepcopy(self.oracles[0])
        oracle["cases"][0]["expected"] = {"kind": "return", "value": True}
        prompt["public_cases"][0]["expected"] = {"kind": "return", "value": True}
        witness = oracle["incorrect_programs"][0]["witness"]
        witness["case_id"] = "P01"
        witness["predicted"] = {"kind": "return", "value": 1}
        self.assertIsNone(self.module.validate_task(prompt, oracle, self.metadata[0]))
        prompt["public_cases"][0]["expected"]["value"] = 1
        with self.assertRaises(ValueError):
            self.module.validate_task(prompt, oracle, self.metadata[0])
        prompt["public_cases"][0]["expected"]["value"] = True
        witness["predicted"]["value"] = True
        with self.assertRaises(ValueError):
            self.module.validate_task(prompt, oracle, self.metadata[0])

    def test_malformed_nested_fields_and_identities_are_rejected(self):
        mutations = (
            lambda prompt, oracle, metadata: prompt.pop("spec"),
            lambda prompt, oracle, metadata: oracle.update(unexpected=True),
            lambda prompt, oracle, metadata: metadata.update(unexpected=True),
            lambda prompt, oracle, metadata: prompt.update(initial_source="different"),
            lambda prompt, oracle, metadata: prompt["public_cases"].reverse(),
            lambda prompt, oracle, metadata: oracle["cases"].pop(),
            lambda prompt, oracle, metadata: oracle["cases"].append(
                copy.deepcopy(oracle["cases"][0])),
            lambda prompt, oracle, metadata: oracle["cases"][1].update(id="P01"),
            lambda prompt, oracle, metadata: oracle["cases"][2].update(id="H07"),
            lambda prompt, oracle, metadata: oracle["cases"][2].update(
                args=copy.deepcopy(oracle["cases"][3]["args"])),
            lambda prompt, oracle, metadata: oracle["cases"][2].update(coverage=[]),
            lambda prompt, oracle, metadata: oracle["cases"][2].update(
                expected={"kind": "raises", "type": "TypeError"}),
            lambda prompt, oracle, metadata: oracle["cases"][2]["expected"].update(
                unexpected=True),
            lambda prompt, oracle, metadata: oracle["references"][1].update(
                id="reference_a"),
            lambda prompt, oracle, metadata: oracle["incorrect_programs"].reverse(),
            lambda prompt, oracle, metadata: oracle["incorrect_programs"][1].update(
                id="variant_3"),
            lambda prompt, oracle, metadata: oracle["incorrect_programs"][1]["witness"].update(
                case_id="H07"),
            lambda prompt, oracle, metadata: oracle["incorrect_programs"][1]["witness"].update(
                unexpected=True),
            lambda prompt, oracle, metadata: metadata.update(parameters=["capacity"] * 2),
        )
        for ordinal, mutate in enumerate(mutations):
            prompt = copy.deepcopy(self.prompts[0])
            oracle = copy.deepcopy(self.oracles[0])
            metadata = copy.deepcopy(self.metadata[0])
            mutate(prompt, oracle, metadata)
            with self.subTest(ordinal=ordinal), self.assertRaises(ValueError):
                self.module.validate_task(prompt, oracle, metadata)

    def test_signature_and_nested_syntax_rejections(self):
        sources = (
            "def wrong(capacity, requests):\n    pass\n",
            "def admit_reservations(capacity, /, requests):\n    pass\n",
            "def admit_reservations(capacity, *, requests):\n    pass\n",
            "def admit_reservations(capacity, *requests):\n    pass\n",
            "def admit_reservations(capacity, **requests):\n    pass\n",
            "def admit_reservations(capacity: int, requests):\n    pass\n",
            "def admit_reservations(capacity, requests) -> list:\n    pass\n",
            "async def admit_reservations(capacity, requests):\n    pass\n",
            "def admit_reservations(capacity, requests):\n    def nested():\n        pass\n",
            "def admit_reservations(capacity, requests):\n    class Nested:\n        pass\n",
            "def admit_reservations(capacity, requests):\n    from os import path\n",
            "def admit_reservations(capacity, requests):\n    pass\nprint('must not run')\n",
            "\x00",
        )
        for source in sources:
            oracle = copy.deepcopy(self.oracles[0])
            oracle["references"][0]["source"] = source
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.module.validate_task(self.prompts[0], oracle, self.metadata[0])
