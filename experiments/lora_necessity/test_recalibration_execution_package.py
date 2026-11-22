import ast
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import unittest

from pilot_protocol import FLAGS
from recalibration_data_contract import FILE_PINS
from validation_contract import digest, encoded


MODULES_BEFORE = frozenset(sys.modules)
if importlib.util.find_spec("recalibration_execution_package") is not None:
    import recalibration_execution_package as package
else:
    package = None
MODULES_AFTER = frozenset(sys.modules)

ROOT = Path(__file__).resolve().parent
SOURCE_NAMES = (
    "pilot_protocol.py", "validation_contract.py", "validation_report.py",
    "validation_real_report.py", "recalibration_data_contract.py",
    "recalibration_validation.py",
)
FALSE_FLAGS = (
    *FLAGS, "launch_ready", "run_authorized", "dispatch_authorized",
    "execution_source_manifest_complete", "execution_provenance_verified",
    "trusted_collection_accepted", "dynamic_checks_passed", "full_dynamic_validation_complete",
)
MISSING_BINDINGS = (
    "complete_execution_sources", "live_preflight", "launch_configuration", "one_time_scope",
)
PUBLIC_FIELDS = {
    "kind", "version", "data_version", "status", "stage_id", "run_ids",
    "file_sha256", "dependency_source_sha256", "backend_sha256", "worker_sha256",
    "plan_sha256", "counts", "batches", "missing_bindings", "preparation_sha256",
    *FALSE_FLAGS,
}
BATCH_FIELDS = {
    "batch_index", "start", "stop", "run_id", "plan_sha256", "bundle_sha256",
    "worker_sha256", "bundle_bytes",
}


class StringSubclass(str):
    pass


class BytesSubclass(bytes):
    pass


class DictSubclass(dict):
    pass


class ListSubclass(list):
    pass


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


class ExecutionPreparationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(package, "execution preparation not implemented")
        self.raw_files = {
            name: (ROOT / "data" / (
                name if name == "supervision_v1.json"
                else "recalibration_v2_r1/" + name)).read_bytes()
            for name in FILE_PINS
        }
        self.worker_raw = (ROOT / "validation_task_worker.py").read_bytes()
        self.backend_raw = (ROOT / "validation_remote.py").read_bytes()
        self.source_raw = {name: (ROOT / name).read_bytes() for name in SOURCE_NAMES}
        self.stage_id = "f" * 32
        self.run_ids = ["%032x" % ordinal for ordinal in range(15)]

    def arguments(self, **changes):
        return {
            "raw_files": self.raw_files, "worker_raw": self.worker_raw,
            "backend_raw": self.backend_raw, "source_raw": self.source_raw,
            "stage_id": self.stage_id, "run_ids": self.run_ids, **changes,
        }

    def prepare(self, **changes):
        return package.build_preparation(**self.arguments(**changes))

    def validate(self, prepared, **changes):
        return package.validate_preparation(prepared, **self.arguments(**changes))

    def rehash(self, prepared):
        public = prepared["public"]
        public["preparation_sha256"] = digest({
            name: value for name, value in public.items() if name != "preparation_sha256"
        })

    def test_first_preparation_is_complete_but_not_executable(self):
        prepared = self.prepare()
        self.assertEqual(prepared["public"]["status"], "PREPARATION_ONLY")
        self.assertEqual(len(prepared["private"]["bundles_hex"]), 15)
        self.assertIs(prepared["public"]["launch_ready"], False)
        self.assertEqual(prepared["public"]["counts"],
                         {"scheduled": 480, "batches": 15, "references": 192,
                          "witnesses": 36})

    def test_exact_fields_pins_digest_and_unavailable_bindings(self):
        prepared = self.prepare()
        public = prepared["public"]
        self.assertEqual(set(prepared), {"public", "private"})
        self.assertEqual(set(prepared["private"]), {"bundles_hex"})
        self.assertEqual(set(public), PUBLIC_FIELDS)
        self.assertEqual(public["kind"], "LORA_RECALIBRATION_EXECUTION_PREPARATION")
        self.assertEqual(public["version"], 1)
        self.assertEqual(public["data_version"], "recalibration_v2_r1")
        self.assertEqual(public["stage_id"], self.stage_id)
        self.assertEqual(public["run_ids"], self.run_ids)
        self.assertEqual(public["missing_bindings"], list(MISSING_BINDINGS))
        self.assertEqual(public["file_sha256"], FILE_PINS)
        self.assertEqual(public["dependency_source_sha256"], {
            name: hashlib.sha256(raw).hexdigest() for name, raw in self.source_raw.items()
        })
        self.assertEqual(public["backend_sha256"],
                         hashlib.sha256(self.backend_raw).hexdigest())
        self.assertEqual(public["worker_sha256"],
                         hashlib.sha256(self.worker_raw).hexdigest())
        self.assertEqual(set(package.SOURCE_PINS), set(SOURCE_NAMES))
        self.assertEqual(package.FALSE_FLAGS, FALSE_FLAGS)
        self.assertEqual(package.MISSING_BINDINGS, MISSING_BINDINGS)
        for flag in FALSE_FLAGS:
            self.assertIs(public[flag], False)
        unsigned = {name: value for name, value in public.items()
                    if name != "preparation_sha256"}
        self.assertEqual(public["preparation_sha256"],
                         hashlib.sha256(canonical_bytes(unsigned)).hexdigest())
        self.assertEqual(json.loads(canonical_bytes(prepared)), prepared)
        self.assertEqual(len(public["batches"]), 15)
        for ordinal, batch in enumerate(public["batches"]):
            self.assertEqual(set(batch), BATCH_FIELDS)
            self.assertEqual(batch["batch_index"], ordinal)
            self.assertEqual(batch["start"], ordinal * 32)
            self.assertEqual(batch["stop"], (ordinal + 1) * 32)
            self.assertEqual(batch["run_id"], self.run_ids[ordinal])
            self.assertEqual(batch["plan_sha256"], public["plan_sha256"])
            self.assertRegex(batch["plan_sha256"], "^[0-9a-f]{64}$")
            self.assertEqual(batch["worker_sha256"], public["worker_sha256"])

    def test_all_bundles_match_independent_raw_json_projection(self):
        prompts = json.loads(self.raw_files["prompts.json"])
        oracles = json.loads(self.raw_files["oracles.json"])
        expected_requests = []
        identities = []
        for prompt, oracle in zip(prompts, oracles):
            self.assertEqual(prompt["id"], oracle["id"])
            for program in oracle["references"] + oracle["incorrect_programs"]:
                for case in oracle["cases"]:
                    expected_requests.append({
                        "request_id": "request-%04d" % len(expected_requests),
                        "source": program["source"], "entry_point": prompt["entry_point"],
                        "args": case["args"],
                    })
                    identities.append((prompt["id"], program["id"], case["id"]))
        self.assertEqual(len(expected_requests), 480)
        prepared = self.prepare()
        observed = []
        for ordinal, raw_hex in enumerate(prepared["private"]["bundles_hex"]):
            raw = bytes.fromhex(raw_hex)
            expected = {
                "kind": "LORA_REMOTE_VALIDATION_BUNDLE", "mode": "matrix",
                "run_id": self.run_ids[ordinal],
                "worker_source": self.worker_raw.decode("utf-8"),
                "requests": expected_requests[ordinal * 32:(ordinal + 1) * 32],
            }
            self.assertEqual(raw, canonical_bytes(expected))
            self.assertEqual(raw_hex, raw.hex())
            batch = prepared["public"]["batches"][ordinal]
            self.assertEqual(batch["bundle_bytes"], len(raw))
            self.assertEqual(batch["bundle_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertLessEqual(len(raw), 131072)
            observed.extend(json.loads(raw)["requests"])
        self.assertEqual(observed, expected_requests)
        for ordinal, identity in (
                (31, ("C01", "variant_1", "H06")),
                (32, ("C01", "variant_2", "P01")),
                (39, ("C01", "variant_2", "H06")),
                (40, ("C02", "reference_a", "P01"))):
            self.assertEqual(identities[ordinal], identity)
            self.assertEqual(observed[ordinal]["request_id"], "request-%04d" % ordinal)
        for request in observed:
            self.assertEqual(set(request), {"request_id", "source", "entry_point", "args"})
            self.assertLessEqual(len(request["source"].encode("utf-8")), 4096)
            self.assertLessEqual(len(canonical_bytes(request)), 8192)

    def test_repeat_and_reversed_mappings_are_canonically_equal(self):
        prepared = self.prepare()
        repeated = self.prepare()
        reversed_inputs = self.prepare(
            raw_files=dict(reversed(list(self.raw_files.items()))),
            source_raw=dict(reversed(list(self.source_raw.items()))))
        for actual in (repeated, reversed_inputs):
            self.assertEqual(actual, prepared)
            self.assertEqual(encoded(actual), canonical_bytes(prepared))

    def test_every_frozen_input_pin_rejects_appended_bytes(self):
        for argument in ("raw_files", "source_raw"):
            for name in self.arguments()[argument]:
                changed = dict(self.arguments()[argument])
                changed[name] += b"\n"
                with self.subTest(argument=argument, name=name):
                    with self.assertRaises(ValueError):
                        self.prepare(**{argument: changed})
        for argument in ("worker_raw", "backend_raw"):
            with self.subTest(argument=argument):
                with self.assertRaises(ValueError):
                    self.prepare(**{argument: self.arguments()[argument] + b"\n"})

    def test_source_and_data_mappings_require_exact_keys_and_builtin_dict(self):
        for argument in ("source_raw", "raw_files"):
            original = self.arguments()[argument]
            first = next(iter(original))
            invalid = [None, [], tuple(original.items()), DictSubclass(original)]
            invalid.extend({name: raw for name, raw in original.items() if name != missing}
                           for missing in original)
            invalid.extend([
                {**original, "extra.py": b""},
                {**{name: raw for name, raw in original.items() if name != first},
                 first.encode("ascii"): original[first]},
                {StringSubclass(name): raw for name, raw in original.items()},
            ])
            for ordinal, value in enumerate(invalid):
                with self.subTest(argument=argument, case=ordinal):
                    with self.assertRaises(ValueError):
                        self.prepare(**{argument: value})

    def test_pinned_buffers_reject_nonbytes_subclasses_and_oversize(self):
        for argument in ("source_raw", "raw_files"):
            original = self.arguments()[argument]
            for name, raw in original.items():
                for value in (BytesSubclass(raw), bytearray(raw), raw.decode("utf-8"),
                              None, b"x" * 1048577):
                    with self.subTest(argument=argument, name=name, kind=type(value)):
                        with self.assertRaises(ValueError):
                            self.prepare(**{argument: {**original, name: value}})
        for argument, limit in (("worker_raw", 16384), ("backend_raw", 1048576)):
            raw = self.arguments()[argument]
            for value in (BytesSubclass(raw), bytearray(raw), raw.decode("utf-8"),
                          None, b"x" * (limit + 1)):
                with self.subTest(argument=argument, kind=type(value)):
                    with self.assertRaises(ValueError):
                        self.prepare(**{argument: value})

    def test_stage_requires_distinct_strict_lowercase_hex(self):
        for value in ("", "f" * 31, "f" * 33, "F" * 32, "g" * 32,
                      "f" * 32 + "\n", "\n" + "f" * 32, " " * 32,
                      None, 15, True, b"f" * 32, StringSubclass(self.stage_id),
                      *self.run_ids):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.prepare(stage_id=value)

    def test_run_ids_require_exact_list_distinct_strict_strings(self):
        invalid = [
            None, tuple(self.run_ids), ListSubclass(self.run_ids), self.run_ids[0],
            self.run_ids[:-1], self.run_ids + ["a" * 32], [self.run_ids[0]] * 15,
        ]
        invalid.extend([value] + self.run_ids[1:] for value in (
            None, 0, False, b"a" * 32, StringSubclass("a" * 32), "A" * 32,
            "g" * 32, "a" * 31, "a" * 33, "a" * 32 + "\n", " " * 32))
        for ordinal, value in enumerate(invalid):
            with self.subTest(case=ordinal):
                with self.assertRaises(ValueError):
                    self.prepare(run_ids=value)

    def test_changed_ids_produce_distinct_unexecuted_bindings(self):
        prepared = self.prepare()
        changed_stage = self.prepare(stage_id="e" * 32)
        changed_runs = list(self.run_ids)
        changed_runs[7] = "a" * 32
        changed_batch = self.prepare(run_ids=changed_runs)
        self.assertEqual(changed_stage["private"], prepared["private"])
        self.assertEqual(changed_stage["public"]["batches"], prepared["public"]["batches"])
        for changed in (changed_stage, changed_batch):
            self.assertNotEqual(changed["public"]["preparation_sha256"],
                                prepared["public"]["preparation_sha256"])
            self.assertEqual(changed["public"]["plan_sha256"],
                             prepared["public"]["plan_sha256"])
            self.assertEqual(changed["public"]["status"], "PREPARATION_ONLY")
            for flag in FALSE_FLAGS:
                self.assertIs(changed["public"][flag], False)
        for ordinal in range(15):
            before = prepared["private"]["bundles_hex"][ordinal]
            after = changed_batch["private"]["bundles_hex"][ordinal]
            if ordinal == 7:
                self.assertNotEqual(before, after)
                self.assertNotEqual(prepared["public"]["batches"][ordinal]["bundle_sha256"],
                                    changed_batch["public"]["batches"][ordinal]["bundle_sha256"])
            else:
                self.assertEqual(before, after)
                self.assertEqual(prepared["public"]["batches"][ordinal],
                                 changed_batch["public"]["batches"][ordinal])

    def test_validator_accepts_exact_reconstruction_and_returns_none(self):
        prepared = self.prepare()
        self.assertIsNone(self.validate(prepared))
        self.assertIsNone(self.validate(json.loads(canonical_bytes(prepared))))

    def test_validator_rejects_rehashed_public_forgery(self):
        prepared = self.prepare()
        for name, value in prepared["public"].items():
            if name == "preparation_sha256":
                continue
            forged = copy.deepcopy(prepared)
            forged["public"][name] = (
                True if type(value) is bool else
                value + 1 if type(value) is int else
                value + "changed" if type(value) is str else
                [] if type(value) is list else {})
            with self.subTest(name=name, rehashed=False):
                with self.assertRaises(ValueError):
                    self.validate(forged)
            self.rehash(forged)
            with self.subTest(name=name, rehashed=True):
                with self.assertRaises(ValueError):
                    self.validate(forged)
        forged = copy.deepcopy(prepared)
        forged["public"]["preparation_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.validate(forged)

    def test_validator_rejects_changed_private_hex_even_with_rehashed_binding(self):
        prepared = self.prepare()
        original = prepared["private"]["bundles_hex"]
        variants = {
            "swapped": [original[1], original[0], *original[2:]],
            "edited": [original[0][:-2] + "00", *original[1:]],
            "uppercase": [original[0].upper(), *original[1:]],
            "whitespace": [original[0] + "\n", *original[1:]],
            "nonhex": ["not hex", *original[1:]],
            "short": original[:-1],
            "extra": original + [original[0]],
            "wrong_type": [bytes.fromhex(original[0]), *original[1:]],
        }
        for name, value in variants.items():
            forged = copy.deepcopy(prepared)
            forged["private"]["bundles_hex"] = value
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.validate(forged)
        forged = copy.deepcopy(prepared)
        raw = bytes.fromhex(forged["private"]["bundles_hex"][0]) + b" "
        forged["private"]["bundles_hex"][0] = raw.hex()
        forged["public"]["batches"][0]["bundle_sha256"] = hashlib.sha256(raw).hexdigest()
        forged["public"]["batches"][0]["bundle_bytes"] = len(raw)
        self.rehash(forged)
        with self.assertRaises(ValueError):
            self.validate(forged)

    def test_validator_rejects_missing_extra_and_admission_fields(self):
        prepared = self.prepare()
        for path in ((), ("public",), ("private",), ("public", "batches", 0),
                     ("public", "counts"), ("public", "file_sha256"),
                     ("public", "dependency_source_sha256")):
            target = prepared
            for name in path:
                target = target[name]
            for field in (*target, "extra"):
                forged = copy.deepcopy(prepared)
                changed = forged
                for name in path:
                    changed = changed[name]
                if field in changed:
                    del changed[field]
                else:
                    changed[field] = None
                with self.subTest(path=path, field=field):
                    with self.assertRaises(ValueError):
                        self.validate(forged)
        for field in ("claims", "commands", "preflight", "acceptance", "runtime",
                      "runtime_profile", "complete_execution_sources", "source_manifest"):
            forged = copy.deepcopy(prepared)
            forged["public"][field] = {"approved": True}
            self.rehash(forged)
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self.validate(forged)
                with self.assertRaises(TypeError):
                    self.prepare(**{field: {"approved": True}})

    def test_validator_distinguishes_bool_int_float_and_builtin_types(self):
        prepared = self.prepare()
        substitutions = [
            (("public", "version"), True), (("public", "version"), 1.0),
            (("public", "batches", 0, "batch_index"), False),
            (("public", "run_ids"), ListSubclass(self.run_ids)),
            (("public", "stage_id"), StringSubclass(self.stage_id)),
            (("public",), DictSubclass(prepared["public"])),
            (("private", "bundles_hex"), tuple(prepared["private"]["bundles_hex"])),
        ]
        substitutions.extend((("public", flag), 0) for flag in FALSE_FLAGS)
        for path, value in substitutions:
            forged = copy.deepcopy(prepared)
            changed = forged
            for name in path[:-1]:
                changed = changed[name]
            changed[path[-1]] = value
            self.rehash(forged)
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    self.validate(forged)
        for value in (None, [], DictSubclass(prepared),
                      {StringSubclass(name): item for name, item in prepared.items()}):
            with self.subTest(kind=type(value)):
                with self.assertRaises(ValueError):
                    self.validate(value)

    def test_validator_rejects_other_ids_and_raw_inputs(self):
        prepared = self.prepare()
        changes = [
            {"stage_id": "e" * 32},
            {"run_ids": ["a" * 32] + self.run_ids[1:]},
            {"worker_raw": self.worker_raw + b"\n"},
            {"backend_raw": self.backend_raw + b"\n"},
        ]
        for argument in ("raw_files", "source_raw"):
            for name in self.arguments()[argument]:
                changed = dict(self.arguments()[argument])
                changed[name] += b"\n"
                changes.append({argument: changed})
        for ordinal, changed in enumerate(changes):
            self.rehash(prepared)
            with self.subTest(case=ordinal):
                with self.assertRaises(ValueError):
                    self.validate(prepared, **changed)
        for changed in changes[:2]:
            with self.assertRaises(ValueError):
                self.validate(self.prepare(**changed))

    def test_mutable_results_never_alias_inputs_constants_or_future_outputs(self):
        inputs_before = copy.deepcopy(self.arguments())
        constants_before = copy.deepcopy((FILE_PINS, package.SOURCE_PINS))
        prepared = self.prepare()
        pristine = copy.deepcopy(prepared)
        self.assertIsNone(self.validate(prepared))
        self.assertEqual(prepared, pristine)
        self.assertEqual(self.arguments(), inputs_before)

        def corrupt(value):
            if type(value) is dict:
                for child in list(value.values()):
                    corrupt(child)
                value["mutated"] = True
            elif type(value) is list:
                for child in list(value):
                    corrupt(child)
                value.append("mutated")

        corrupt(prepared)
        self.assertEqual(self.arguments(), inputs_before)
        self.assertEqual((FILE_PINS, package.SOURCE_PINS), constants_before)
        self.assertEqual(self.prepare(), pristine)
        self.run_ids[0] = "a" * 32
        self.source_raw.clear()
        self.raw_files.clear()
        self.assertEqual(pristine["public"]["run_ids"][0], "0" * 32)

    def test_module_is_pure_and_only_defines_the_two_public_functions(self):
        self.prepare()
        tree = ast.parse((ROOT / "recalibration_execution_package.py").read_bytes())
        public_functions = {
            node.name for node in tree.body
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
        }
        self.assertEqual(public_functions, {"build_preparation", "validate_preparation"})
        allowed_imports = {
            "hashlib", "re", "pilot_protocol", "recalibration_data_contract",
            "recalibration_validation", "validation_contract",
        }
        allowed_calls = {
            "require", "type", "len", "all", "bytes", "dict", "list",
            "fields", "_plain", "_pinned_bytes", "build_batches", "build_preparation",
            "digest", "encoded",
        }
        allowed_methods = {"sha256", "hexdigest", "fullmatch", "items", "hex"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue({alias.name for alias in node.names} <= allowed_imports)
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed_imports)
                self.assertEqual(node.level, 0)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    self.assertIn(node.func.id, allowed_calls)
                else:
                    self.assertIsInstance(node.func, ast.Attribute)
                    self.assertIn(node.func.attr, allowed_methods)
        forbidden = {
            "validation_remote", "validation_task_worker", "torch", "transformers",
            "tokenizers", "paramiko",
        }
        for modules in (MODULES_AFTER - MODULES_BEFORE, frozenset(sys.modules)):
            self.assertFalse({name.split(".")[0] for name in modules} & forbidden)
        expected_parameters = (
            "raw_files", "worker_raw", "backend_raw", "source_raw", "stage_id", "run_ids")
        for function, prefix in ((package.build_preparation, ()),
                                 (package.validate_preparation, ("preparation",))):
            signature = inspect.signature(function)
            self.assertEqual(tuple(signature.parameters), prefix + expected_parameters)
            for name in ("stage_id", "run_ids"):
                self.assertEqual(signature.parameters[name].kind,
                                 inspect.Parameter.KEYWORD_ONLY)
