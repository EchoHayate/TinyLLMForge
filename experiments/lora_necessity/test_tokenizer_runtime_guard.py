import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch


class RuntimeGuardTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("tokenizer_runtime_guard")
        for context in (
            patch.object(self.module, "_USED", False),
            patch.object(sys, "meta_path", list(sys.meta_path)),
            patch.dict(self.module.os.environ),
            patch.dict(sys.modules),
        ):
            context.start()
            self.addCleanup(context.stop)
        for name in tuple(sys.modules):
            if name.split(".")[0] in self.module.COLD_STACK | {"safetensors"}:
                del sys.modules[name]
        self.guard = self.module.RuntimeGuard()
        self.addCleanup(self.guard.close)

    def test_normal_configuration_helper_import_does_not_call_weight_loader(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules):
            root = Path(temporary)
            package = root / "transformers"
            package.mkdir()
            (package / "__init__.py").write_text("")
            (package / "configuration_utils.py").write_text(
                "from .modeling_gguf_pytorch_utils import load_gguf_checkpoint\n")
            (package / "modeling_gguf_pytorch_utils.py").write_text(
                "import safetensors\n"
                "def load_gguf_checkpoint():\n"
                "    raise AssertionError('weight loading must not run')\n")
            (root / "safetensors.py").write_text("loaded = True\n")
            with patch.object(sys, "path", [temporary, *sys.path]):
                self.guard.start()
                try:
                    configuration = importlib.import_module("transformers.configuration_utils")
                except ImportError as error:
                    self.fail(f"normal configuration dependency was rejected: {error}")
                self.assertTrue(callable(configuration.load_gguf_checkpoint))
                self.assertTrue(sys.modules["safetensors"].loaded)
                self.guard.check()
                self.assertFalse(self.guard.failed)

    def test_start_check_close_leave_import_machinery_untouched(self):
        original_finders = list(sys.meta_path)
        original_discovery = importlib.util.find_spec
        self.guard.start()
        self.assertEqual(sys.meta_path, original_finders)
        self.assertIs(importlib.util.find_spec, original_discovery)
        self.guard.check()
        self.guard.close()
        self.assertEqual(sys.meta_path, original_finders)
        self.assertIs(importlib.util.find_spec, original_discovery)

    def test_optional_dependency_discovery_retains_normal_loader(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules):
            Path(temporary, "safetensors.py").write_text("loaded = True\n")
            with patch.object(sys, "path", [temporary, *sys.path]):
                self.guard.start()
                specification = importlib.util.find_spec("safetensors")
                self.assertIsNotNone(specification)
                self.assertNotIn("safetensors", sys.modules)
                try:
                    library = importlib.import_module("safetensors")
                except ImportError as error:
                    self.fail(f"ordinary optional dependency was rejected: {error}")
                self.assertTrue(library.loaded)
                self.guard.check()

    def test_missing_optional_import_does_not_poison_guard(self):
        with patch.object(sys, "path", []):
            self.guard.start()
            self.assertIsNone(importlib.util.find_spec("safetensors"))
            with self.assertRaises(ImportError):
                importlib.import_module("safetensors")
            self.assertFalse(self.guard.failed)
            self.guard.check()

    def test_imported_backend_name_does_not_prove_model_execution(self):
        self.guard.start()
        with patch.dict(sys.modules, {"torch": ModuleType("torch")}):
            try:
                self.guard.check()
            except ValueError as error:
                self.fail(f"module presence was treated as model execution: {error}")
            self.assertFalse(self.guard.failed)

    def test_environment_start_close_and_no_retry(self):
        expected_environment = {
            "USE_TORCH": "0", "USE_TF": "0", "USE_FLAX": "0",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
        self.assertEqual(self.module.ENVIRONMENT, expected_environment)
        self.guard.start()
        self.guard.check()
        for name, expected in expected_environment.items():
            self.assertEqual(self.module.os.environ[name], expected)
        self.guard.close()
        self.guard.close()
        self.assertFalse(self.guard.active)
        for name, expected in expected_environment.items():
            self.assertEqual(self.module.os.environ[name], expected)
        with self.assertRaises(ValueError):
            self.guard.check()
        with self.assertRaises(ValueError):
            self.module.RuntimeGuard().start()
        with self.assertRaises(ValueError):
            self.guard.start()

    def test_preloaded_stack_rejected_before_environment_change(self):
        before = dict(self.module.os.environ)
        for name in (
            "transformers.fixture", "tokenizers", "huggingface_hub.fixture",
            "jinja2.fixture", "torch.fixture", "tensorflow", "jax", "flax",
        ):
            with self.subTest(name=name), patch.object(self.module, "_USED", False):
                with patch.dict(sys.modules, {name: ModuleType("fixture")}):
                    with self.assertRaises(ValueError):
                        self.guard.start()
                self.assertEqual(dict(self.module.os.environ), before)
                self.assertFalse(self.guard.active)
                with self.assertRaises(ValueError):
                    self.module.RuntimeGuard().start()

    def test_every_environment_setting_is_checked_and_failure_is_sticky(self):
        for name in self.module.ENVIRONMENT:
            with self.subTest(name=name), patch.object(self.module, "_USED", False):
                guard = self.module.RuntimeGuard()
                guard.start()
                try:
                    expected = self.module.os.environ.pop(name)
                    with self.assertRaisesRegex(ValueError, "runtime environment changed"):
                        guard.check()
                    self.module.os.environ[name] = expected
                    with self.assertRaises(ValueError):
                        guard.check()
                    self.assertTrue(guard.failed)
                finally:
                    guard.close()

    def test_start_on_wrong_thread_rejected_before_environment_change(self):
        before = dict(self.module.os.environ)
        with patch.object(self.module.threading, "current_thread", return_value=object()):
            with self.assertRaises(ValueError):
                self.guard.start()
        self.assertFalse(self.module._USED)
        self.assertFalse(self.guard.active)
        self.assertEqual(dict(self.module.os.environ), before)

    def test_wrong_thread_check_is_sticky(self):
        self.guard.start()
        with patch.object(self.module.threading, "current_thread", return_value=object()):
            with self.assertRaises(ValueError):
                self.guard.check()
        with self.assertRaises(ValueError):
            self.guard.check()

    def test_close_does_not_remove_unrelated_import_hooks(self):
        self.guard.start()
        sentinel = object()
        sys.meta_path.insert(0, sentinel)
        self.guard.close()
        self.assertIs(sys.meta_path[0], sentinel)

    def test_module_import_preserves_process_state(self):
        before_environment = dict(self.module.os.environ)
        before_finders = list(sys.meta_path)
        before_modules = set(sys.modules)
        with patch.dict(sys.modules):
            del sys.modules["tokenizer_runtime_guard"]
            imported = importlib.import_module("tokenizer_runtime_guard")
            self.assertFalse(imported._USED)
            self.assertEqual(dict(imported.os.environ), before_environment)
            self.assertEqual(sys.meta_path, before_finders)
            self.assertEqual(set(sys.modules), before_modules)
