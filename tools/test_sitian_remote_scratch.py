from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "sitian_remote_scratch.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "sitian_remote_scratch_test_module",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PolicyTests(unittest.TestCase):
    def test_repo_root_accepts_only_authoritative_and_approved_remote_roots(self):
        module = load_module()
        accepted = [
            ROOT,
            Path(module.REMOTE_ROOT) / "source",
            Path(module.REMOTE_ROOT) / "red-task1",
        ]
        for path in accepted:
            with self.subTest(accepted=path):
                self.assertEqual(
                    module.ScratchConfig.default(path).repo_root,
                    path,
                )

        rejected = [
            Path(module.REMOTE_ROOT),
            Path(module.REMOTE_ROOT) / "tmp",
            Path(module.REMOTE_ROOT) / "pycache",
            Path(module.REMOTE_ROOT) / "cache",
            Path(module.REMOTE_ROOT) / "logs",
            Path(module.REMOTE_ROOT) / "receipts",
            Path(module.REMOTE_ROOT) / "env",
        ]
        for path in rejected:
            with self.subTest(rejected=path):
                with self.assertRaises(ValueError):
                    module.ScratchConfig.default(path)

    def test_fixed_configuration_values_cannot_be_overridden(self):
        module = load_module()
        overrides = [
            {"remote_host": "other-host"},
            {"remote_root": "/private/tmp/other-root"},
            {"krb5_cache": "FILE:/private/tmp/other-cache"},
            {"attempts": 1},
        ]
        for override in overrides:
            with self.subTest(override=override):
                with self.assertRaises(TypeError):
                    module.ScratchConfig(repo_root=ROOT, **override)

    def test_fixed_layout_stays_under_remote_task_root(self):
        module = load_module()
        config = module.ScratchConfig.default(ROOT)
        self.assertEqual(
            config.remote_root,
            "/data00/home/sitian/tinyllmforge-workspaces/"
            "command-timeline-20260818",
        )
        layout = module.remote_layout(config)
        self.assertEqual(
            set(layout),
            {"source", "tmp", "pycache", "cache", "logs", "receipts", "env"},
        )
        self.assertTrue(
            all(
                path.startswith(config.remote_root + "/")
                for path in layout.values()
            )
        )

    def test_explicit_paths_accept_only_clean_repository_relative_files(self):
        module = load_module()
        self.assertEqual(
            module.validate_relative_paths(
                [
                    "tools/sitian_remote_scratch.py",
                    "tools/test_sitian_remote_scratch.py",
                ]
            ),
            (
                "tools/sitian_remote_scratch.py",
                "tools/test_sitian_remote_scratch.py",
            ),
        )
        rejected = [
            "/private/tmp/output.log",
            "../TinyLLMForge-adaptive-ngram/file.py",
            ".git/config",
            "artifacts/run/output.json",
            "experiments/run/source.patch",
            "tools/__pycache__/module.pyc",
            ".superpowers/sdd/task-5-review-package.diff",
            "runner.log",
            "runner.pid",
        ]
        for path in rejected:
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    module.validate_relative_paths([path])

    def test_explicit_paths_reject_broad_or_unsafe_operands(self):
        module = load_module()
        rejected = [
            ".",
            "tools",
            "--checkpoint-action=exfiltrate",
            "tools/does-not-exist-task1.py",
        ]
        for path in rejected:
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    module.validate_relative_paths([path])

        escape = ROOT / "tools" / ".sitian-remote-scratch-escape"
        if escape.exists() or escape.is_symlink():
            escape.unlink()
        escape.symlink_to("/etc/passwd")
        try:
            with self.assertRaises(ValueError):
                module.validate_relative_paths(
                    ["tools/.sitian-remote-scratch-escape"]
                )
        finally:
            escape.unlink()

    def test_explicit_paths_reject_log_trees_and_common_archives(self):
        module = load_module()
        rejected = [
            "logs/output.txt",
            "nested/logs/output.txt",
            "bundle.tar.xz",
            "bundle.tar.bz2",
            "bundle.7z",
            "bundle.zst",
            "bundle.rar",
            "bundle.txz",
            "bundle.tbz2",
        ]
        for path in rejected:
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    module.validate_relative_paths([path])

    def test_incremental_tar_argv_terminates_options_before_paths(self):
        module = load_module()
        builder = getattr(module, "incremental_tar_argv", None)
        self.assertIsNotNone(builder)
        argv = builder(["tools/sitian_remote_scratch.py"])
        separator = argv.index("--")
        self.assertEqual(
            argv[separator + 1:],
            ("tools/sitian_remote_scratch.py",),
        )

    def test_remote_cache_environment_has_no_local_tmp_path(self):
        module = load_module()
        config = module.ScratchConfig.default(ROOT)
        environment = module.remote_cache_environment(config)
        self.assertEqual(environment["TMPDIR"], config.remote_root + "/tmp")
        self.assertEqual(
            environment["PYTHONPYCACHEPREFIX"],
            config.remote_root + "/pycache",
        )
        self.assertEqual(
            environment["XDG_CACHE_HOME"],
            config.remote_root + "/cache",
        )
        self.assertNotIn("/tmp", "\n".join(environment.values()).replace(
            config.remote_root + "/tmp",
            "",
        ))


if __name__ == "__main__":
    unittest.main()
