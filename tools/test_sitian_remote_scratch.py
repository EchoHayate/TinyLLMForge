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
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tools import sitian_remote_transaction as transaction


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


def run_sync_helper_with_post_exchange_event(
    root, nonce, archive, event
):
    driver = r"""
import os
from pathlib import Path
import signal
import sys
from tools import sitian_remote_transaction as transaction

event = sys.argv[1]
real_inject = transaction._inject
real_renameat2 = transaction._RENAMEAT2
real_pthread_sigmask = transaction.signal.pthread_sigmask
real_consume_pending = transaction._consume_pending_transaction_signals
real_cleanup_staged_delta = transaction._cleanup_staged_delta
real_stdout = sys.stdout
remote_root = Path(sys.argv[4])
teardown_signal_sent = False
post_drain_signal_sent = False
cleanup_signal_sent = False
stdout_signal_sent = False

def inject(fault_injector, point):
    if point == "after_exchange":
        if event == "exception":
            raise transaction.InjectedFailure(point)
        if event == "sigterm":
            os.kill(os.getpid(), signal.SIGTERM)
    return real_inject(fault_injector, point)

def renameat2(*args):
    result = real_renameat2(*args)
    if result == 0 and event == "exchange_return_sigterm":
        os.kill(os.getpid(), signal.SIGTERM)
    return result

def pthread_sigmask(how, mask):
    global teardown_signal_sent
    previous = real_pthread_sigmask(how, mask)
    source = (
        remote_root
        / "source"
        / "tools"
        / "sitian_remote_scratch.py"
    )
    if (
        event == "teardown_pending_sigterm"
        and not teardown_signal_sent
        and how == signal.SIG_BLOCK
        and source.is_file()
        and source.read_bytes() == b"new\n"
    ):
        teardown_signal_sent = True
        os.kill(os.getpid(), signal.SIGTERM)
    return previous

def consume_pending():
    global post_drain_signal_sent
    real_consume_pending()
    if (
        event == "post_drain_sigterm"
        and not post_drain_signal_sent
        and (
            remote_root
            / "source"
            / "tools"
            / "sitian_remote_scratch.py"
        ).read_bytes() == b"new\n"
    ):
        post_drain_signal_sent = True
        os.kill(os.getpid(), signal.SIGTERM)

def cleanup_staged_delta(staged):
    global cleanup_signal_sent
    if (
        event == "staged_cleanup_sigterm"
        and not cleanup_signal_sent
        and (
            remote_root
            / "source"
            / "tools"
            / "sitian_remote_scratch.py"
        ).read_bytes() == b"new\n"
    ):
        cleanup_signal_sent = True
        os.kill(os.getpid(), signal.SIGTERM)
    return real_cleanup_staged_delta(staged)

class StdoutProxy:
    def write(self, data):
        global stdout_signal_sent
        if (
            event == "before_stdout_sigterm"
            and not stdout_signal_sent
            and data.endswith(".sha256\n")
        ):
            stdout_signal_sent = True
            os.kill(os.getpid(), signal.SIGTERM)
        return real_stdout.write(data)

    def flush(self):
        return real_stdout.flush()

transaction._inject = inject
transaction._RENAMEAT2 = renameat2
transaction.signal.pthread_sigmask = pthread_sigmask
transaction._consume_pending_transaction_signals = consume_pending
transaction._cleanup_staged_delta = cleanup_staged_delta
sys.stdout = StdoutProxy()
sys.argv = [sys.argv[0]] + sys.argv[2:]
sys.exit(transaction.main())
"""
    process = subprocess.run(
        (
            sys.executable,
            "-c",
            driver,
            event,
            "sync-commit",
            "--remote-root",
            str(root),
            "--nonce",
            nonce,
            "--source-head",
            "b" * 40,
            "--path",
            "tools/sitian_remote_scratch.py",
        ),
        input=archive,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    return subprocess.CompletedProcess(
        process.args,
        process.returncode,
        process.stdout.decode("utf-8", errors="replace"),
        process.stderr.decode("utf-8", errors="replace"),
    )


class PolicyTests(unittest.TestCase):
    def test_repo_root_accepts_only_authoritative_and_approved_remote_roots(self):
        module = load_module()
        accepted = [
            ROOT,
            Path(module.REMOTE_ROOT) / "source",
            Path(module.REMOTE_ROOT) / "red-task1",
            Path(module.REMOTE_ROOT) / "task2-red-c1bd1ae",
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
    @staticmethod
    def _single_file_archive(path, data):
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as handle:
            info = tarfile.TarInfo(path)
            info.size = len(data)
            handle.addfile(info, io.BytesIO(data))
        return archive.getvalue()

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
        config = SimpleNamespace(remote_root=module.REMOTE_ROOT)
        commands = module.initial_snapshot_commands(config)
        self.assertEqual(
            commands["archive"],
            ("git", "archive", "--format=tar", "HEAD"),
        )
        self.assertIn(".transactions/", commands["remote_extract"])
        self.assertIn("/generation", commands["remote_extract"])
        self.assertIn(
            "sitian_remote_transaction.py",
            commands["remote_commit"],
        )
        self.assertIn("init-commit", commands["remote_commit"])
        self.assertNotIn(
            "SITIAN_SYNC_FAIL_POINT",
            commands["remote_commit"],
        )
        self.assertNotIn("/private/tmp", json.dumps(commands))

    def test_initial_snapshot_excludes_and_verifies_every_forbidden_class(self):
        module = load_module()
        config = module.ScratchConfig.default(ROOT)
        commands = module.initial_snapshot_commands(config)
        extract = commands["remote_extract"]
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
            extracted = sorted(
                path.relative_to(generated_stage).as_posix()
                for path in generated_stage.rglob("*")
                if path.is_file()
            )
            self.assertEqual(extracted, ["tools/allowed.py"])
        finally:
            shutil.rmtree(
                generated_stage.parent,
                ignore_errors=True,
            )

    def test_remote_extract_rejects_symlinked_transaction_components(self):
        module = load_module()
        archive = self._single_file_archive("escape.txt", b"escaped\n")
        for component in (".transactions", "nonce", "generation"):
            with self.subTest(component=component):
                root = ROOT / ".sitian-init-symlink-{}".format(component)
                outside = ROOT / ".sitian-init-outside-{}".format(component)
                root.mkdir()
                outside.mkdir()
                config = SimpleNamespace(remote_root=str(root))
                commands = module.initial_snapshot_commands(config)
                generation = Path(commands["stage"])
                nonce_dir = generation.parent
                transactions = nonce_dir.parent
                if component == ".transactions":
                    transactions.symlink_to(
                        outside, target_is_directory=True
                    )
                else:
                    transactions.mkdir(mode=0o700)
                    if component == "nonce":
                        nonce_dir.symlink_to(
                            outside, target_is_directory=True
                        )
                    else:
                        nonce_dir.mkdir(mode=0o700)
                        generation.symlink_to(
                            outside, target_is_directory=True
                        )
                try:
                    result = subprocess.run(
                        ("/bin/sh", "-c", commands["remote_extract"]),
                        input=archive,
                        capture_output=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse((outside / "escape.txt").exists())
                finally:
                    shutil.rmtree(root, ignore_errors=True)
                    shutil.rmtree(outside, ignore_errors=True)

    def test_remote_extract_creates_private_transaction_directories(self):
        module = load_module()
        root = ROOT / ".sitian-init-private-modes"
        root.mkdir()
        config = SimpleNamespace(remote_root=str(root))
        commands = module.initial_snapshot_commands(config)
        generation = Path(commands["stage"])
        try:
            result = subprocess.run(
                ("/bin/sh", "-c", commands["remote_extract"]),
                input=self._single_file_archive(
                    "tools/allowed.py", b"allowed\n"
                ),
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for directory in (
                generation.parent.parent,
                generation.parent,
                generation,
            ):
                with self.subTest(directory=directory):
                    self.assertEqual(
                        directory.stat().st_mode & 0o777,
                        0o700,
                    )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_incremental_sync_requires_explicit_allowed_paths(self):
        module = load_module()
        config = module.ScratchConfig.default(ROOT)
        commands = module.incremental_sync_commands(
            config,
            ["tools/sitian_remote_scratch.py"],
            nonce="command-test",
            source_head="b" * 40,
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
                nonce="command-test",
                source_head="b" * 40,
            )

    def test_incremental_commands_invoke_sync_commit_without_shell_rollback(
        self,
    ):
        module = load_module()
        module.APPROVED_REPO_ROOTS = frozenset(
            set(module.APPROVED_REPO_ROOTS) | {ROOT.resolve()}
        )
        config = module.ScratchConfig.default(ROOT)
        commands = module.incremental_sync_commands(
            config,
            ["tools/sitian_remote_scratch.py"],
            nonce="sync-nonce",
            source_head="b" * 40,
        )

        self.assertIn("--no-xattrs", commands["tar"])
        remote = " ".join(commands["ssh"])
        self.assertIn("sitian_remote_transaction.py", remote)
        self.assertIn("sync-commit", remote)
        self.assertIn("--nonce sync-nonce", remote)
        self.assertIn("--source-head " + "b" * 40, remote)
        self.assertIn("--path tools/sitian_remote_scratch.py", remote)
        for forbidden in (
            "SITIAN_SYNC_FAIL_POINT",
            "checkpoint",
            "rollback",
            ".incoming-sync-",
            ".sync-transaction-lock",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, remote)

    def test_incremental_controller_streams_once_then_confirms_ambiguity(self):
        module = load_module()
        module.APPROVED_REPO_ROOTS = frozenset(
            set(module.APPROVED_REPO_ROOTS) | {ROOT.resolve()}
        )
        config = module.ScratchConfig.default(ROOT)
        nonce = "100-200-300"
        expected_receipt = (
            config.remote_root + "/receipts/sync-{}.sha256".format(nonce)
        )
        lost_response = module.subprocess.CompletedProcess(
            ["ssh"],
            255,
            "",
            "connection lost after exchange",
        )
        confirmed = module.subprocess.CompletedProcess(
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
                        "_resolve_local_head",
                        return_value="b" * 40,
                    ):
                        with mock.patch.object(
                            module,
                            "_stream_with_retries",
                            return_value=lost_response,
                        ) as stream:
                            with mock.patch.object(
                                module,
                                "_remote_command",
                                return_value=confirmed,
                            ) as remote:
                                receipt, count = module._sync(
                                    config,
                                    ["tools/sitian_remote_scratch.py"],
                                )

        self.assertEqual(receipt, expected_receipt)
        self.assertEqual(count, 1)
        stream.assert_called_once()
        self.assertEqual(stream.call_args[1]["attempts"], 1)
        remote.assert_called_once()
        confirm_command = remote.call_args[0][1]
        self.assertIn(" confirm ", " " + confirm_command + " ")
        self.assertIn("--nonce " + nonce, confirm_command)
        self.assertIn("--operation sync", confirm_command)
        self.assertIn("--source-head " + "b" * 40, confirm_command)
        self.assertIn(
            "--path tools/sitian_remote_scratch.py",
            confirm_command,
        )

    def test_initial_controller_preserves_unambiguous_commit_failures(self):
        module = load_module()
        config = module.ScratchConfig.default(ROOT)
        head = "b" * 40
        streamed = module.subprocess.CompletedProcess(
            ["ssh"], 0, "", ""
        )
        real_run_with_retries = module.run_with_retries
        for status in (143, 1):
            with self.subTest(status=status):
                rejected = module.subprocess.CompletedProcess(
                    ["ssh"],
                    status,
                    "",
                    "init commit failed",
                )
                attempt_limits = []
                mutation_attempts = []

                def mutation_runner(argv, **kwargs):
                    del kwargs
                    mutation_attempts.append(tuple(argv))
                    return rejected

                def run_at_retry_boundary(argv, *, attempts, **kwargs):
                    attempt_limits.append(attempts)
                    return real_run_with_retries(
                        argv,
                        attempts=attempts,
                        runner=mutation_runner,
                        sleep=lambda _: None,
                        **kwargs,
                    )

                with mock.patch.object(
                    module,
                    "_resolve_local_head",
                    return_value=head,
                ), mock.patch.object(
                    module,
                    "_stream_with_retries",
                    return_value=streamed,
                ) as stream, mock.patch.object(
                    module,
                    "run_with_retries",
                    side_effect=run_at_retry_boundary,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "initial source promotion failed",
                    ):
                        module._initialize(config)

                stream.assert_called_once()
                self.assertTrue(mutation_attempts)
                self.assertTrue(
                    all(
                        " init-commit "
                        in " " + attempt[-1] + " "
                        for attempt in mutation_attempts
                    )
                )
                self.assertTrue(
                    all(
                        " confirm " not in " " + attempt[-1] + " "
                        for attempt in mutation_attempts
                    )
                )
                self.assertEqual(
                    (attempt_limits, len(mutation_attempts)),
                    ([1], 1),
                )

    def test_initial_controller_confirms_ambiguous_matching_commit_once(self):
        module = load_module()
        config = module.ScratchConfig.default(ROOT)
        head = "b" * 40
        streamed = module.subprocess.CompletedProcess(
            ["ssh"], 0, "", ""
        )
        lost_response = module.subprocess.CompletedProcess(
            ["ssh"],
            255,
            "",
            "connection lost after init commit",
        )
        confirmed = module.subprocess.CompletedProcess(
            ["ssh"], 0, "17\n", ""
        )
        with mock.patch.object(
            module,
            "_resolve_local_head",
            return_value=head,
        ), mock.patch.object(
            module,
            "_stream_with_retries",
            return_value=streamed,
        ) as stream, mock.patch.object(
            module,
            "_remote_command",
            side_effect=(lost_response, confirmed),
        ) as remote:
            failure = None
            result = None
            try:
                result = module._initialize(config)
            except RuntimeError as exc:
                failure = exc

        stream.assert_called_once()
        self.assertEqual(remote.call_count, 2)
        mutation_command = remote.call_args_list[0][0][1]
        confirmation_command = remote.call_args_list[1][0][1]
        generation = mutation_command.split("--generation ", 1)[1].split(
            " ", 1
        )[0]
        nonce = generation.split("/")[1]
        self.assertIn(" init-commit ", " " + mutation_command + " ")
        self.assertIn(" confirm ", " " + confirmation_command + " ")
        self.assertIn(
            config.remote_root
            + "/source/tools/sitian_remote_transaction.py",
            confirmation_command,
        )
        self.assertIn("--nonce " + nonce, confirmation_command)
        self.assertIn("--operation init", confirmation_command)
        self.assertIn("--source-head " + head, confirmation_command)
        self.assertNotIn("--path", confirmation_command)
        self.assertIsNone(failure)
        self.assertEqual(result, (head, 17))

    def test_initial_controller_ambiguous_unmatched_commit_remains_failure(self):
        module = load_module()
        config = module.ScratchConfig.default(ROOT)
        head = "b" * 40
        streamed = module.subprocess.CompletedProcess(
            ["ssh"], 0, "", ""
        )
        lost_response = module.subprocess.CompletedProcess(
            ["ssh"],
            255,
            "",
            "connection lost before init commit",
        )
        not_committed = module.subprocess.CompletedProcess(
            ["ssh"], 1, "", "matching init commit not found"
        )
        with mock.patch.object(
            module,
            "_resolve_local_head",
            return_value=head,
        ), mock.patch.object(
            module,
            "_stream_with_retries",
            return_value=streamed,
        ) as stream, mock.patch.object(
            module,
            "_remote_command",
            side_effect=(lost_response, not_committed),
        ) as remote:
            failure = None
            try:
                module._initialize(config)
            except RuntimeError as exc:
                failure = exc

        stream.assert_called_once()
        self.assertEqual(remote.call_count, 2)
        mutation_command = remote.call_args_list[0][0][1]
        confirmation_command = remote.call_args_list[1][0][1]
        generation = mutation_command.split("--generation ", 1)[1].split(
            " ", 1
        )[0]
        nonce = generation.split("/")[1]
        self.assertIn(" confirm ", " " + confirmation_command + " ")
        self.assertIn("--nonce " + nonce, confirmation_command)
        self.assertIn("--operation init", confirmation_command)
        self.assertIn("--source-head " + head, confirmation_command)
        self.assertNotIn("--path", confirmation_command)
        self.assertIsInstance(failure, RuntimeError)
        self.assertEqual(
            str(failure),
            "initial source promotion failed",
        )

    def test_incremental_controller_does_not_confirm_unambiguous_rejection(
        self,
    ):
        module = load_module()
        module.APPROVED_REPO_ROOTS = frozenset(
            set(module.APPROVED_REPO_ROOTS) | {ROOT.resolve()}
        )
        config = module.ScratchConfig.default(ROOT)
        rejected = module.subprocess.CompletedProcess(
            ["ssh"],
            76,
            "",
            "incremental sync requires full init",
        )
        with mock.patch.object(
            module,
            "_resolve_local_head",
            return_value="b" * 40,
        ), mock.patch.object(
            module,
            "_stream_with_retries",
            return_value=rejected,
        ) as stream, mock.patch.object(
            module,
            "_remote_command",
        ) as remote:
            with self.assertRaisesRegex(
                RuntimeError,
                "incremental source sync failed",
            ):
                module._sync(
                    config,
                    ["tools/sitian_remote_scratch.py"],
                )

        stream.assert_called_once()
        remote.assert_not_called()

    def test_incremental_controller_non_255_helper_failure_is_precommit(self):
        module = load_module()
        with tempfile.TemporaryDirectory(
            prefix="sitian-controller-commit-boundary-"
        ) as temporary:
            root = Path(temporary)
            generation = root / ".transactions/init-nonce/generation"
            generation.joinpath("tools").mkdir(parents=True)
            generation.joinpath("tools/a.py").write_bytes(b"old\n")
            transaction.commit_initial_generation(
                root,
                ".transactions/init-nonce/generation",
                "a" * 40,
            )
            source_before = generation_snapshot = {}
            for path in sorted((root / "source").rglob("*")):
                relative = path.relative_to(root / "source").as_posix()
                if path.is_symlink():
                    generation_snapshot[relative] = (
                        "symlink",
                        os.readlink(path),
                    )
                elif path.is_dir():
                    generation_snapshot[relative] = ("directory", None)
                else:
                    generation_snapshot[relative] = (
                        "file",
                        path.read_bytes(),
                    )
            source_before = generation_snapshot
            nonce = "100-200-300"
            target = root / "receipts/target.state"
            target.write_text("outside\n", encoding="utf-8")
            receipt = root / "receipts/sync-{}.state".format(nonce)
            receipt.symlink_to(target.name)
            archive = self._single_file_archive(
                "tools/sitian_remote_scratch.py",
                b"new\n",
            )
            config = SimpleNamespace(
                repo_root=ROOT,
                remote_root=str(root),
                remote_host=module.REMOTE_HOST,
                krb5_cache=module.KRB5_CACHE,
                attempts=1,
            )

            def run_real_helper(*args, **kwargs):
                del args, kwargs
                process = subprocess.run(
                    (
                        sys.executable,
                        str(Path(transaction.__file__).resolve()),
                        "sync-commit",
                        "--remote-root",
                        str(root),
                        "--nonce",
                        nonce,
                        "--source-head",
                        "b" * 40,
                        "--path",
                        "tools/sitian_remote_scratch.py",
                    ),
                    input=archive,
                    capture_output=True,
                    env={
                        **os.environ,
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                )
                return subprocess.CompletedProcess(
                    process.args,
                    process.returncode,
                    process.stdout.decode("utf-8", errors="replace"),
                    process.stderr.decode("utf-8", errors="replace"),
                )

            with mock.patch.object(module.time, "time", return_value=100), \
                    mock.patch.object(module.os, "getpid", return_value=200), \
                    mock.patch.object(
                        module.time, "time_ns", return_value=300
                    ), mock.patch.object(
                        module,
                        "_resolve_local_head",
                        return_value="b" * 40,
                    ), mock.patch.object(
                        module,
                        "_stream_with_retries",
                        side_effect=run_real_helper,
                    ), mock.patch.object(
                        module,
                        "_remote_command",
                    ) as remote:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "incremental source sync failed",
                ):
                    module._sync(
                        config,
                        ["tools/sitian_remote_scratch.py"],
                    )

            source_after = {}
            for path in sorted((root / "source").rglob("*")):
                relative = path.relative_to(root / "source").as_posix()
                if path.is_symlink():
                    source_after[relative] = (
                        "symlink",
                        os.readlink(path),
                    )
                elif path.is_dir():
                    source_after[relative] = ("directory", None)
                else:
                    source_after[relative] = ("file", path.read_bytes())
            self.assertEqual(source_after, source_before)
            self.assertTrue(receipt.is_symlink())
            self.assertEqual(target.read_text(encoding="utf-8"), "outside\n")
            remote.assert_not_called()

    def test_incremental_controller_accepts_post_exchange_helper_success(
        self,
    ):
        module = load_module()
        for event in (
            "exception",
            "sigterm",
            "exchange_return_sigterm",
            "teardown_pending_sigterm",
            "post_drain_sigterm",
            "staged_cleanup_sigterm",
            "before_stdout_sigterm",
        ):
            with self.subTest(event=event), tempfile.TemporaryDirectory(
                prefix="sitian-controller-post-exchange-{}-".format(event)
            ) as temporary:
                root = Path(temporary)
                generation = root / ".transactions/init-nonce/generation"
                generation.joinpath("tools").mkdir(parents=True)
                generation.joinpath(
                    "tools/sitian_remote_scratch.py"
                ).write_bytes(b"old\n")
                transaction.commit_initial_generation(
                    root,
                    ".transactions/init-nonce/generation",
                    "a" * 40,
                )
                nonce = "100-200-300"
                archive = self._single_file_archive(
                    "tools/sitian_remote_scratch.py",
                    b"new\n",
                )
                config = SimpleNamespace(
                    repo_root=ROOT,
                    remote_root=str(root),
                    remote_host=module.REMOTE_HOST,
                    krb5_cache=module.KRB5_CACHE,
                    attempts=1,
                )
                helper_results = []

                def run_real_helper(*args, **kwargs):
                    del args, kwargs
                    result = run_sync_helper_with_post_exchange_event(
                        root, nonce, archive, event
                    )
                    helper_results.append(result)
                    return result

                with mock.patch.object(
                    module.time, "time", return_value=100
                ), mock.patch.object(
                    module.os, "getpid", return_value=200
                ), mock.patch.object(
                    module.time, "time_ns", return_value=300
                ), mock.patch.object(
                    module,
                    "_resolve_local_head",
                    return_value="b" * 40,
                ), mock.patch.object(
                    module,
                    "_stream_with_retries",
                    side_effect=run_real_helper,
                ), mock.patch.object(
                    module,
                    "_remote_command",
                ) as remote:
                    receipt, count = module._sync(
                        config,
                        ["tools/sitian_remote_scratch.py"],
                    )

                expected_receipt = str(
                    root / "receipts/sync-{}.sha256".format(nonce)
                )
                self.assertEqual(receipt, expected_receipt)
                self.assertEqual(count, 1)
                self.assertEqual(len(helper_results), 1)
                self.assertEqual(helper_results[0].returncode, 0)
                self.assertEqual(
                    helper_results[0].stdout, expected_receipt + "\n"
                )
                self.assertEqual(helper_results[0].stderr, "")
                self.assertEqual(
                    (
                        root
                        / "source/tools/sitian_remote_scratch.py"
                    ).read_bytes(),
                    b"new\n",
                )
                remote.assert_not_called()

    def test_production_controller_has_no_fault_injector_or_old_state_machine(
        self,
    ):
        module = load_module()
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertFalse(hasattr(module, "_incremental_remote_command"))
        self.assertFalse(hasattr(module, "_incremental_commit_status_command"))
        for forbidden in (
            "SITIAN_SYNC_FAIL_POINT",
            "testing-fault-point",
            "checkpoint()",
            "transaction_committed()",
            "rollback_commands",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

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
