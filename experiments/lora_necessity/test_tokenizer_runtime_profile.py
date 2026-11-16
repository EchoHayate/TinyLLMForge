import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from validation_contract import encoded


class RuntimeProfileTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("tokenizer_runtime_profile.py").is_file())
        self.module = importlib.import_module("tokenizer_runtime_profile")
        self.versions = {
            "python": "3.12.13", "transformers": "4.57.1",
            "tokenizers": "0.0.0+fixture", "huggingface-hub": "0.0.0+fixture",
            "jinja2": "0.0.0+fixture",
        }
        self.profile = {
            "kind": "TOKENIZER_RUNTIME_PROFILE", "version": 1,
            "run_id": "a" * 32, "revision": self.module.REVISION,
            "file_sha256": dict(self.module.FILE_PINS), "versions": self.versions,
        }

    def invoke(self, profile):
        raw = encoded(profile)
        return self.module.validate_profile(raw, hashlib.sha256(raw).hexdigest())

    def test_exact_profile_and_copy(self):
        validated = self.invoke(self.profile)
        self.assertEqual(validated, self.profile)
        validated["versions"]["python"] = "changed"
        self.assertEqual(self.profile["versions"]["python"], "3.12.13")
        raw = encoded(self.profile)
        with self.assertRaises(ValueError):
            self.module.validate_profile(raw + b"\n", hashlib.sha256(raw).hexdigest())
        duplicate = b'{"kind":1,"kind":2}'
        with self.assertRaises(ValueError):
            self.module.validate_profile(duplicate, hashlib.sha256(duplicate).hexdigest())

    def test_complete_duplicate_profile_rejected(self):
        duplicate = b'{"kind":1,' + encoded(self.profile)[1:]
        approved = hashlib.sha256(duplicate).hexdigest()
        with patch.object(self.module, "parse_json", json.loads):
            self.assertEqual(self.module.validate_profile(duplicate, approved), self.profile)
        with self.assertRaisesRegex(ValueError, "^duplicate JSON key$"):
            self.module.validate_profile(duplicate, approved)

    def test_wrong_target_fields_and_types_rejected(self):
        variants = [
            ("kind", "LORA_REAL_TOKENIZER_AUDIT"), ("version", True),
            ("run_id", "../escape"), ("revision", "b" * 40),
            ("file_sha256", {}), ("versions", []),
        ]
        for name, value in variants:
            with self.subTest(name=name), self.assertRaises(ValueError):
                profile = copy.deepcopy(self.profile)
                profile[name] = value
                self.invoke(profile)
        profile = copy.deepcopy(self.profile)
        profile["loaded"] = True
        with self.assertRaises(ValueError):
            self.invoke(profile)

    def test_incomplete_or_unpinned_versions_rejected(self):
        for name in self.versions:
            with self.subTest(missing=name), self.assertRaises(ValueError):
                profile = copy.deepcopy(self.profile)
                del profile["versions"][name]
                self.invoke(profile)
        for name, value in (("python", "3.12.12"), ("transformers", "5.9.0"),
                            ("jinja2", ""), ("tokenizers", True)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                profile = copy.deepcopy(self.profile)
                profile["versions"][name] = value
                self.invoke(profile)

    def test_metadata_inventory_normalization_and_duplicates(self):
        distributions = [
            SimpleNamespace(metadata={"Name": "huggingface_hub"}, version="fixture"),
        ]
        with patch.object(self.module.platform, "python_version", return_value="3.12.13"), \
                patch.object(self.module.metadata, "distributions", return_value=distributions):
            self.assertEqual(self.module.runtime_inventory(), {
                "python": "3.12.13", "huggingface-hub": "fixture",
            })
            distributions.append(SimpleNamespace(
                metadata={"Name": "huggingface-hub"}, version="fixture"))
            with self.assertRaises(ValueError):
                self.module.runtime_inventory()

    def test_metadata_only_result_and_inventory_drift(self):
        raw = encoded(self.profile)
        approved = hashlib.sha256(raw).hexdigest()
        with patch.object(self.module, "runtime_inventory", return_value=self.versions):
            result = self.module.inspect_runtime(raw, approved)
        self.assertEqual(result["kind"], "TOKENIZER_RUNTIME_METADATA_ONLY")
        self.assertFalse(result["tokenizer_loaded"])
        self.assertEqual(result["profile_sha256"], approved)
        for observed in ({}, {**self.versions, "extra-package": "1"}):
            with self.subTest(observed=observed), patch.object(
                    self.module, "runtime_inventory", return_value=observed), \
                    self.assertRaises(ValueError):
                self.module.inspect_runtime(raw, approved)
        with patch.object(self.module, "runtime_inventory") as inventory:
            with self.assertRaises(ValueError):
                self.module.inspect_runtime(raw, "0" * 64)
            inventory.assert_not_called()
