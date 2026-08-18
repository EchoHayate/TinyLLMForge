from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import unittest
from unittest import mock


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

    def test_explicit_paths_reject_symlink_operand_and_parent_before_sync(self):
        module = load_module()
        operand_link = ROOT / "tools" / ".sitian-in-repo-link.py"
        parent_link = ROOT / ".sitian-in-repo-parent"
        for candidate in (operand_link, parent_link):
            if candidate.exists() or candidate.is_symlink():
                candidate.unlink()
        operand_link.symlink_to("sitian_remote_scratch.py")
        parent_link.symlink_to("tools", target_is_directory=True)
        paths = [
            "tools/.sitian-in-repo-link.py",
            ".sitian-in-repo-parent/sitian_remote_scratch.py",
        ]
        try:
            for relative_path in paths:
                with self.subTest(path=relative_path):
                    with self.assertRaises(ValueError):
                        module.validate_relative_paths([relative_path])
            config = module.ScratchConfig.default(ROOT)
            with mock.patch.object(module, "_stream_with_retries") as stream:
                error = None
                try:
                    module._sync(config, [paths[0]])
                except Exception as exc:
                    error = exc
                self.assertTrue(
                    isinstance(error, ValueError) and not stream.called,
                    "symlink path must be rejected before transport",
                )
            self.assertTrue(operand_link.is_symlink())
            self.assertTrue(parent_link.is_symlink())
        finally:
            operand_link.unlink()
            parent_link.unlink()

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

    def test_explicit_paths_reject_every_shared_forbidden_class(self):
        module = load_module()
        rejected = [
            ".cache/blob.bin",
            "cache/blob.bin",
            "caches/blob.bin",
            "raw-output/chunk.txt",
            "raw_output/chunk.txt",
            "rawoutput/chunk.txt",
            "tools/._metadata.py",
            ".superpowers/sdd/taskreview-package.diff",
        ]
        created = []
        try:
            for relative_path in rejected:
                candidate = ROOT / relative_path
                candidate.parent.mkdir(parents=True, exist_ok=True)
                candidate.write_text("forbidden\n", encoding="utf-8")
                created.append(candidate)
            for relative_path in rejected:
                with self.subTest(path=relative_path):
                    with self.assertRaises(ValueError):
                        module.validate_relative_paths([relative_path])
        finally:
            for candidate in reversed(created):
                if candidate.exists() or candidate.is_symlink():
                    candidate.unlink()
            for relative_path in rejected:
                parent = (ROOT / relative_path).parent
                while parent != ROOT:
                    try:
                        parent.rmdir()
                    except OSError:
                        break
                    parent = parent.parent

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


class TransportTests(unittest.TestCase):
    def test_retry_stops_after_first_success(self):
        module = load_module()
        runner = mock.Mock(side_effect=[
            module.subprocess.CompletedProcess(
                ["ssh"], 255, "", "Connection closed"
            ),
            module.subprocess.CompletedProcess(["ssh"], 0, "ok\n", ""),
        ])
        result = module.run_with_retries(
            ["ssh"],
            attempts=5,
            runner=runner,
            sleep=lambda _: None,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(runner.call_count, 2)

    def test_initial_snapshot_uses_git_archive_and_no_local_file(self):
        module = load_module()
        config = module.ScratchConfig.default(ROOT)
        commands = module.initial_snapshot_commands(config)
        self.assertEqual(
            commands["archive"],
            ("git", "archive", "--format=tar", "HEAD"),
        )
        self.assertIn(".incoming-", commands["remote_extract"])
        self.assertIn("find source -name '._*'", commands["remote_verify"])
        self.assertNotIn("/private/tmp", json.dumps(commands))

    def test_initial_snapshot_excludes_and_verifies_every_forbidden_class(self):
        module = load_module()
        config = module.ScratchConfig.default(ROOT)
        commands = module.initial_snapshot_commands(config)
        extract = commands["remote_extract"]
        verify = commands["remote_verify"]
        for token in (
            ".git",
            "artifacts",
            "experiments",
            "__pycache__",
            ".pytest_cache",
            "cache",
            "logs",
            "*.pyc",
            "*.log",
            "*.pid",
            "raw-output",
            "raw_output",
            "*.tar",
            "*review-package.diff",
        ):
            with self.subTest(token=token):
                self.assertIn(token, extract)
                self.assertIn(token, verify)
        self.assertGreaterEqual(verify.count("| wc -l"), 8)

    def test_remote_extract_filters_forbidden_fixture_archive(self):
        module = load_module()
        stage = ROOT / ".sitian-filter-fixture"
        config = SimpleNamespace(remote_root=str(stage.parent))
        commands = module.initial_snapshot_commands(config)
        generated_stage = Path(commands["stage"])
        members = {
            "tools/allowed.py": b"allowed\n",
            ".git/config": b"git\n",
            "artifacts/run/result.json": b"artifact\n",
            "nested/artifacts/result.txt": b"nested artifact\n",
            "experiments/run.py": b"experiment\n",
            "pkg/__pycache__/module.pyc": b"cache\n",
            ".pytest_cache/state": b"cache\n",
            "cache/blob.bin": b"cache\n",
            "logs/worker.txt": b"log tree\n",
            "worker.log": b"log\n",
            "worker.pid": b"123\n",
            "raw-output/chunk.txt": b"raw\n",
            "raw_output/chunk.txt": b"raw\n",
            "source.tar": b"archive\n",
            "nested/source.tar.gz": b"archive\n",
            ".superpowers/sdd/task-review-package.diff": b"review\n",
            "._metadata": b"metadata\n",
        }
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as handle:
            for name, data in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                handle.addfile(info, io.BytesIO(data))
        try:
            extract = subprocess.run(
                ("/bin/sh", "-c", commands["remote_extract"]),
                input=archive.getvalue(),
                capture_output=True,
            )
            self.assertEqual(extract.returncode, 0, extract.stderr)
            verify = subprocess.run(
                ("/bin/sh", "-c", commands["remote_verify"]),
                capture_output=True,
                text=True,
            )
            self.assertEqual(verify.returncode, 0, verify.stderr)
            extracted = sorted(
                path.relative_to(generated_stage / "source").as_posix()
                for path in (generated_stage / "source").rglob("*")
                if path.is_file()
            )
            self.assertEqual(extracted, ["tools/allowed.py"])
        finally:
            shutil.rmtree(generated_stage, ignore_errors=True)

    def test_promotion_retry_requires_matching_transaction_receipt(self):
        module = load_module()
        root = ROOT / ".sitian-promotion-retry-fixture"
        stage = root / ".incoming-source-test-transaction"
        receipts = root / "receipts"
        head = "1" * 40
        try:
            (root / "source").mkdir(parents=True)
            receipts.mkdir()
            (receipts / "source-head.txt").write_text(
                head + "\n",
                encoding="utf-8",
            )
            (receipts / "source-files.sha256").write_text(
                "0" * 64 + "  source/old.py\n",
                encoding="utf-8",
            )
            command = module._initial_promotion_command(
                SimpleNamespace(remote_root=str(root)),
                stage=str(stage),
                head=head,
            )
            result = subprocess.run(
                ("/bin/sh", "-c", command),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(
                result.returncode,
                0,
                "same HEAD without this transaction receipt is not success",
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_incremental_sync_requires_explicit_allowed_paths(self):
        module = load_module()
        config = module.ScratchConfig.default(ROOT)
        commands = module.incremental_sync_commands(
            config,
            ["tools/sitian_remote_scratch.py"],
        )
        self.assertIn("--no-xattrs", commands["tar"])
        self.assertIn("--no-mac-metadata", commands["tar"])
        self.assertIn(
            "tools/sitian_remote_scratch.py",
            commands["tar"],
        )
        with self.assertRaises(ValueError):
            module.incremental_sync_commands(
                config,
                [".superpowers/sdd/task-5-review-package.diff"],
            )

    def test_incremental_remote_verification_uses_shared_forbidden_policy(self):
        module = load_module()
        config = SimpleNamespace(remote_root="/remote/task-root")
        command, _ = module._incremental_remote_command(
            config,
            ["tools/allowed.py"],
            nonce="policy-test",
        )
        for token in (
            ".git",
            "artifacts",
            "experiments",
            "__pycache__",
            ".pytest_cache",
            ".cache",
            "cache",
            "caches",
            "logs",
            "*.pyc",
            "*.log",
            "*.pid",
            "raw-output",
            "raw_output",
            "rawoutput",
            "*.tar",
            "*review-package.diff",
            "._*",
        ):
            with self.subTest(token=token):
                self.assertIn(token, command)

    def test_incremental_sync_recovers_commit_after_response_loss(self):
        module = load_module()
        config = module.ScratchConfig.default(ROOT)
        nonce = "100-200-300"
        expected_receipt = (
            config.remote_root + f"/receipts/sync-{nonce}.sha256"
        )
        lost_response = module.subprocess.CompletedProcess(
            ["ssh"],
            255,
            "",
            "connection lost after commit",
        )
        committed = module.subprocess.CompletedProcess(
            ["ssh"],
            0,
            expected_receipt + "\n",
            "",
        )
        with mock.patch.object(module.time, "time", return_value=100):
            with mock.patch.object(module.os, "getpid", return_value=200):
                with mock.patch.object(
                    module.time,
                    "time_ns",
                    return_value=300,
                ):
                    with mock.patch.object(
                        module,
                        "_stream_with_retries",
                        return_value=lost_response,
                    ) as stream:
                        with mock.patch.object(
                            module,
                            "_remote_command",
                            return_value=committed,
                        ) as remote:
                            error = None
                            try:
                                receipt, count = module._sync(
                                    config,
                                    ["tools/sitian_remote_scratch.py"],
                                )
                            except Exception as exc:
                                error = exc
        self.assertIsNone(error, "committed transaction must recover")
        self.assertEqual(receipt, expected_receipt)
        self.assertEqual(count, 1)
        self.assertEqual(stream.call_args[1]["attempts"], 1)
        remote.assert_called_once()

    def test_incremental_transaction_reentry_preserves_committed_result(self):
        module = load_module()
        root = ROOT / ".sitian-sync-committed-fixture"
        source_file = root / "source" / "tools" / "allowed.py"
        nonce = "committed-reentry"
        config = SimpleNamespace(remote_root=str(root))
        command, expected_receipt = module._incremental_remote_command(
            config,
            ["tools/allowed.py"],
            nonce=nonce,
        )
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as handle:
            data = b"new\n"
            info = tarfile.TarInfo("tools/allowed.py")
            info.size = len(data)
            handle.addfile(info, io.BytesIO(data))
        try:
            source_file.parent.mkdir(parents=True)
            source_file.write_text("old\n", encoding="utf-8")
            first = subprocess.run(
                ("/bin/sh", "-c", command),
                input=archive.getvalue(),
                capture_output=True,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            receipts = root / "receipts"
            committed_snapshot = {
                path.name: path.read_bytes()
                for path in receipts.iterdir()
                if path.is_file()
            }
            second = subprocess.run(
                ("/bin/sh", "-c", command),
                input=b"",
                capture_output=True,
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(
                second.stdout.decode().strip(),
                expected_receipt,
            )
            self.assertEqual(source_file.read_text(encoding="utf-8"), "new\n")
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in receipts.iterdir()
                    if path.is_file()
                },
                committed_snapshot,
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_incremental_stale_attempt_preserves_backup_and_receipts(self):
        module = load_module()
        root = ROOT / ".sitian-sync-stale-fixture"
        source_file = root / "source" / "tools" / "allowed.py"
        receipts = root / "receipts"
        nonce = "stale-overlap"
        backup = root / f".incoming-sync-{nonce}"
        path_receipt = receipts / f"sync-{nonce}.paths.txt"
        hash_receipt = receipts / f"sync-{nonce}.sha256"
        state_receipt = receipts / f"sync-{nonce}.state"
        config = SimpleNamespace(remote_root=str(root))
        command, _ = module._incremental_remote_command(
            config,
            ["tools/allowed.py"],
            nonce=nonce,
        )
        try:
            source_file.parent.mkdir(parents=True)
            source_file.write_text("verified\n", encoding="utf-8")
            receipts.mkdir()
            backup.mkdir()
            (backup / "owner-sentinel").write_text(
                "old attempt\n",
                encoding="utf-8",
            )
            path_receipt.write_text(
                "tools/allowed.py\n",
                encoding="utf-8",
            )
            hash_receipt.write_text(
                "0" * 64 + "  tools/allowed.py\n",
                encoding="utf-8",
            )
            state_receipt.write_text("started\n", encoding="utf-8")
            receipt_snapshot = {
                path.name: path.read_bytes()
                for path in receipts.iterdir()
            }
            result = subprocess.run(
                ("/bin/sh", "-c", command),
                input=b"",
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(
                source_file.read_text(encoding="utf-8"),
                "verified\n",
            )
            self.assertTrue((backup / "owner-sentinel").is_file())
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in receipts.iterdir()
                },
                receipt_snapshot,
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_stream_pipeline_closes_and_propagates_producer_failure(self):
        driver = r"""
import importlib.util
import json
from pathlib import Path
import sys

module_path = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("stream_driver_module", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.time.sleep = lambda _: None
producer = (
    sys.executable,
    "-c",
    "import os,sys\n"
    "def write_all(fd, data):\n"
    "    while data:\n"
    "        data = data[os.write(fd, data):]\n"
    "write_all(2, b'e' * 131072)\n"
    "write_all(1, b'x' * 131072)\n"
    "sys.exit(7)\n",
)
consumer = (
    sys.executable,
    "-c",
    "import sys\n"
    "data = sys.stdin.buffer.read()\n"
    "print(len(data))\n",
)
result = module._stream_with_retries(
    producer,
    consumer,
    config=module.ScratchConfig.default(repo_root),
)
print(json.dumps({
    "returncode": result.returncode,
    "stdout": result.stdout.strip(),
    "stderr_bytes": len(result.stderr),
}))
"""
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                driver,
                str(MODULE_PATH),
                str(ROOT),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.communicate()
            self.fail("stream pipeline deadlocked before producer EOF")
        self.assertEqual(stderr, "")
        self.assertEqual(process.returncode, 0)
        payload = json.loads(stdout)
        self.assertEqual(payload["returncode"], 7)
        self.assertEqual(payload["stdout"], "131072")
        self.assertEqual(payload["stderr_bytes"], 131072)

    def test_stream_pipeline_reaps_producer_when_consumer_spawn_fails(self):
        module = load_module()
        producer = mock.Mock()
        producer.stdout = mock.Mock()
        producer.stderr = mock.Mock()
        producer.poll.return_value = None
        producer.wait.side_effect = [
            subprocess.TimeoutExpired(["producer"], 1.0),
            -9,
        ]
        with mock.patch.object(
            module.subprocess,
            "Popen",
            side_effect=[producer, OSError("consumer spawn failed")],
        ):
            with self.assertRaisesRegex(OSError, "consumer spawn failed"):
                module._stream_with_retries(
                    ["producer"],
                    ["consumer"],
                    config=module.ScratchConfig.default(ROOT),
                    attempts=1,
                )
        producer.stdout.close.assert_called_once()
        producer.stderr.close.assert_called_once()
        producer.terminate.assert_called_once()
        producer.kill.assert_called_once()
        self.assertEqual(producer.wait.call_count, 2)


if __name__ == "__main__":
    unittest.main()
