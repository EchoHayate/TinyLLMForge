import ctypes
import contextlib
import errno
import hashlib
import io
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
from unittest import mock

from tools import sitian_remote_transaction as transaction


INIT_FAULT_POINTS = (
    "after_lock",
    "after_generation_ready",
    "after_embedded_receipt",
    "before_exchange",
    "after_exchange",
    "before_external_receipts",
    "after_external_receipts",
    "before_old_generation_cleanup",
)

PRE_EXCHANGE_FAULT_POINTS = INIT_FAULT_POINTS[:4]
POST_EXCHANGE_FAULT_POINTS = INIT_FAULT_POINTS[4:]
SYNC_FAULT_POINTS = INIT_FAULT_POINTS


class LockHolder:
    def __init__(self, root):
        code = "\n".join(
            (
                "import signal",
                "import sys",
                "from pathlib import Path",
                "from tools import sitian_remote_transaction as transaction",
                "signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))",
                "with transaction.locked_remote_root(Path(sys.argv[1])):",
                "    sys.stdout.write('locked\\n')",
                "    sys.stdout.flush()",
                "    sys.stdin.readline()",
            )
        )
        self.process = subprocess.Popen(
            (sys.executable, "-c", code, str(root)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

    def wait_until_locked(self):
        self.assert_running()
        line = self.process.stdout.readline()
        if line != "locked\n":
            stderr = self.process.stderr.read()
            raise AssertionError(
                "lock holder did not acquire lock: {!r} {!r}".format(
                    line, stderr
                )
            )

    def assert_running(self):
        code = self.process.poll()
        if code is not None:
            stderr = self.process.stderr.read()
            raise AssertionError(
                "lock holder exited early with {}: {}".format(code, stderr)
            )

    def terminate(self):
        self.process.terminate()

    def release_normally(self):
        self.process.stdin.write("\n")
        self.process.stdin.flush()
        self.process.stdin.close()

    def wait(self):
        code = self.process.wait(timeout=10)
        stderr = self.process.stderr.read()
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        self.process.stdout.close()
        self.process.stderr.close()
        if stderr:
            raise AssertionError("lock holder stderr: {}".format(stderr))
        return code


class TransactionPrimitiveTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(
            prefix="sitian-transaction-task1-"
        )
        self.addCleanup(self._temporary.cleanup)
        self.test_root = Path(self._temporary.name)

    def make_root(self, name="root"):
        root = self.test_root / name
        root.mkdir()
        return root

    def write_file(self, root, relative_path, content):
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def source_manifest(self, source):
        entries = []
        metadata = Path(".tinyllmforge-scratch")
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            if relative == metadata or metadata in relative.parents:
                continue
            if path.is_symlink():
                digest = hashlib.sha256(
                    b"symlink\0" + os.fsencode(os.readlink(str(path)))
                ).hexdigest()
                entries.append((relative.as_posix(), digest))
            elif path.is_file():
                digest = hashlib.sha256(
                    b"file\0" + path.read_bytes()
                ).hexdigest()
                entries.append((relative.as_posix(), digest))
        data = "".join(
            "{}  {}\n".format(digest, path) for path, digest in entries
        ).encode("utf-8")
        return entries, data

    def make_receipt(
        self,
        source,
        paths=("tools/a.py",),
        nonce="n1",
        operation="sync",
        head="abc",
    ):
        entries, manifest = self.source_manifest(source)
        hashes = {}
        for path in paths:
            hashes[path] = hashlib.sha256(
                (source / path).read_bytes()
            ).hexdigest()
        return transaction.CommitReceipt(
            operation=operation,
            nonce=nonce,
            source_head=head,
            explicit_paths=paths,
            explicit_path_sha256=hashes,
            source_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            source_file_count=len(entries),
            created_at_unix_ns=123456789,
        )

    def replace_receipt_fields(self, receipt, **overrides):
        fields = {
            "operation": receipt.operation,
            "nonce": receipt.nonce,
            "source_head": receipt.source_head,
            "explicit_paths": receipt.explicit_paths,
            "explicit_path_sha256": receipt.explicit_path_sha256,
            "source_manifest_sha256": receipt.source_manifest_sha256,
            "source_file_count": receipt.source_file_count,
            "created_at_unix_ns": receipt.created_at_unix_ns,
        }
        fields.update(overrides)
        return transaction.CommitReceipt(**fields)

    def raw_rename_exchange(
        self, left_parent_fd, left_name, right_parent_fd, right_name
    ):
        result = transaction._RENAMEAT2(
            left_parent_fd,
            os.fsencode(left_name),
            right_parent_fd,
            os.fsencode(right_name),
            transaction.RENAME_EXCHANGE,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))

    def write_generation(self, path, marker):
        path.mkdir(parents=True)
        self.write_file(path, "marker.txt", marker.encode("utf-8"))

    def read_marker(self, path):
        return (path / "marker.txt").read_text()

    def tree_snapshot(self, root):
        if not root.exists():
            return None
        snapshot = {}
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                snapshot[relative] = ("symlink", os.readlink(str(path)))
            elif path.is_dir():
                snapshot[relative] = ("directory", None)
            else:
                snapshot[relative] = ("file", path.read_bytes())
        return snapshot

    def write_detached_receipts(self, root, marker):
        receipts = root / "receipts"
        receipts.mkdir()
        values = {
            "source-head.txt": marker.encode("utf-8") + b"\n",
            "source-files.sha256": (
                b"0" * 64 + b"  old.txt\n"
            ),
            "source-transaction.txt": (
                marker.encode("utf-8") + b"\n"
            ),
        }
        for name, data in values.items():
            (receipts / name).write_bytes(data)

    def assert_initial_detached_receipts(self, root, receipt):
        receipts = root / "receipts"
        self.assertEqual(
            (receipts / "source-head.txt").read_text(),
            receipt.source_head + "\n",
        )
        self.assertEqual(
            (receipts / "source-transaction.txt").read_text(),
            receipt.nonce + "\n",
        )
        embedded_manifest = (
            root
            / "source"
            / ".tinyllmforge-scratch"
            / "source-files.sha256"
        ).read_bytes()
        self.assertEqual(
            (receipts / "source-files.sha256").read_bytes(),
            embedded_manifest,
        )

    def make_committed_root(self, paths=("tools/a.py",)):
        root = self.make_root()
        source = root / "source"
        source.mkdir()
        for index, path in enumerate(paths):
            self.write_file(
                source,
                path,
                "content-{}\n".format(index).encode("utf-8"),
            )
        self.write_file(source, "README.md", b"manifest member\n")
        receipt = self.make_receipt(source, paths=paths)
        generation_fd = transaction.open_directory_no_follow(source)
        try:
            transaction.write_embedded_receipt(generation_fd, receipt)
        finally:
            os.close(generation_fd)
        return root

    def make_initialized_root(self, name="root"):
        root = self.make_root(name)
        generation_name = ".transactions/init-nonce/generation"
        generation = root / generation_name
        self.write_file(generation, "tools/a.py", b"old-a\n")
        self.write_file(generation, "tools/b.py", b"old-b\n")
        self.write_file(generation, "README.md", b"old-readme\n")
        (generation / "current.py").symlink_to("tools/a.py")
        transaction.commit_initial_generation(
            root,
            generation_name,
            "a" * 40,
        )
        return root

    def make_delta(self, name="delta"):
        delta = self.test_root / name
        delta.mkdir()
        return delta

    def make_delta_archive(self, members):
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as handle:
            for name, data in members:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                handle.addfile(info, io.BytesIO(data))
        return archive.getvalue()

    def stage_delta(self, root, nonce, members, paths=("tools/a.py",)):
        return transaction._stage_delta_stream(
            root,
            nonce,
            paths,
            io.BytesIO(self.make_delta_archive(members)),
        )

    def helper_command(self, root, command, *arguments):
        return (
            sys.executable,
            str(Path(transaction.__file__).resolve()),
            command,
            "--remote-root",
            str(root),
            *arguments,
        )

    def helper_environment(self):
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return environment

    def run_helper_with_post_exchange_event(
        self, root, nonce, archive, event
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
    if point == "before_exchange" and event == "before_exchange_sigterm":
        os.kill(os.getpid(), signal.SIGTERM)
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
    source = remote_root / "source" / "tools" / "a.py"
    if (
        event == "teardown_pending_sigterm"
        and not teardown_signal_sent
        and how == signal.SIG_BLOCK
        and source.is_file()
        and source.read_bytes() == b"committed\n"
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
            remote_root / "source" / "tools" / "a.py"
        ).read_bytes() == b"committed\n"
    ):
        post_drain_signal_sent = True
        os.kill(os.getpid(), signal.SIGTERM)

def cleanup_staged_delta(staged):
    global cleanup_signal_sent
    if (
        event == "staged_cleanup_sigterm"
        and not cleanup_signal_sent
        and (
            remote_root / "source" / "tools" / "a.py"
        ).read_bytes() == b"committed\n"
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
        return subprocess.run(
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
                "tools/a.py",
            ),
            input=archive,
            capture_output=True,
            env=self.helper_environment(),
        )

    def fail_at(self, expected):
        def inject(actual):
            if actual == expected:
                raise transaction.InjectedFailure(actual)

        return inject

    def read_committed(self, root, paths=("tools/a.py",)):
        return transaction.read_committed_generation(
            root,
            expected_nonce="n1",
            expected_operation="sync",
            expected_head="abc",
            expected_paths=paths,
        )

    def replace_with_symlink(self, path):
        target = path.with_name(path.name + ".target")
        path.replace(target)
        path.symlink_to(target.name)

    def test_flock_releases_when_holder_exits_after_sigterm(self):
        root = self.make_root()
        first = LockHolder(root)
        first.wait_until_locked()
        first.terminate()
        self.assertEqual(first.wait(), 0)
        with transaction.locked_remote_root(root):
            pass

    def test_flock_releases_when_holder_exits_normally(self):
        root = self.make_root()
        first = LockHolder(root)
        first.wait_until_locked()
        first.release_normally()
        self.assertEqual(first.wait(), 0)
        with transaction.locked_remote_root(root):
            pass

    def test_lock_cleanup_interruption_preserves_result_and_releases_resources(
        self,
    ):
        for target in ("source.lock", ".transactions", "root"):
            with self.subTest(target=target):
                root = self.make_root("root-{}".format(target.replace(".", "")))
                before = len(os.listdir("/proc/self/fd"))
                real_close = transaction.os.close
                close_interrupted = [False]

                def fd_target(fd):
                    return os.readlink("/proc/self/fd/{}".format(fd))

                def should_interrupt(fd):
                    path = fd_target(fd)
                    if target == "source.lock":
                        return path.endswith("/.transactions/source.lock")
                    if target == ".transactions":
                        return path.endswith("/.transactions")
                    return path == str(root)

                def interrupt_target_close(fd):
                    if not close_interrupted[0] and should_interrupt(fd):
                        close_interrupted[0] = True
                        raise transaction.TransactionInterrupted(
                            signal.SIGTERM
                        )
                    return real_close(fd)

                def use_lock():
                    with transaction.locked_remote_root(root):
                        return "body-result"

                with mock.patch.object(
                    transaction.os,
                    "close",
                    side_effect=interrupt_target_close,
                ):
                    self.assertEqual(use_lock(), "body-result")

                self.assertTrue(close_interrupted[0])
                self.assertEqual(len(os.listdir("/proc/self/fd")), before)
                transactions_fd = transaction.open_directory_no_follow(
                    root / ".transactions"
                )
                competing_fd = os.open(
                    "source.lock",
                    os.O_RDWR | os.O_NOFOLLOW,
                    dir_fd=transactions_fd,
                )
                try:
                    transaction.fcntl.flock(
                        competing_fd,
                        transaction.fcntl.LOCK_EX
                        | transaction.fcntl.LOCK_NB,
                    )
                finally:
                    os.close(competing_fd)
                    os.close(transactions_fd)

    def test_directory_open_rejects_symlink(self):
        root = self.make_root()
        target = root / "real"
        target.mkdir()
        (root / "link").symlink_to(target, target_is_directory=True)
        with self.assertRaises(OSError):
            transaction.open_directory_no_follow(root / "link")

    def test_exchange_never_leaves_source_missing(self):
        root = self.make_root()
        self.write_generation(root / "source", marker="old")
        self.write_generation(
            root / ".transactions/n1/generation", marker="new"
        )
        root_fd = transaction.open_directory_no_follow(root)
        try:
            transaction.rename_exchange(
                root_fd,
                "source",
                ".transactions/n1/generation",
            )
        finally:
            os.close(root_fd)
        self.assertEqual(self.read_marker(root / "source"), "new")
        self.assertEqual(
            self.read_marker(root / ".transactions/n1/generation"),
            "old",
        )

    def test_exchange_failure_has_no_two_rename_fallback(self):
        root = self.make_root()
        self.write_generation(root / "source", marker="old")
        self.write_generation(
            root / ".transactions/n1/generation", marker="new"
        )
        root_fd = transaction.open_directory_no_follow(root)

        def unavailable(*args):
            ctypes.set_errno(errno.ENOSYS)
            return -1

        try:
            with mock.patch.object(
                transaction, "_RENAMEAT2", side_effect=unavailable
            ):
                with self.assertRaises(OSError) as raised:
                    transaction.rename_exchange(
                        root_fd,
                        "source",
                        ".transactions/n1/generation",
                    )
        finally:
            os.close(root_fd)
        self.assertEqual(raised.exception.errno, errno.ENOSYS)
        self.assertEqual(self.read_marker(root / "source"), "old")
        self.assertEqual(
            self.read_marker(root / ".transactions/n1/generation"),
            "new",
        )

    def test_exchange_rejects_symlinked_nested_parent(self):
        root = self.make_root()
        self.write_generation(root / "source", marker="old")
        outside = self.test_root / "outside-exchange"
        self.write_generation(outside / "generation", marker="outside")
        transactions = root / ".transactions"
        transactions.mkdir()
        (transactions / "n1").symlink_to(
            outside, target_is_directory=True
        )
        root_fd = transaction.open_directory_no_follow(root)
        try:
            with self.assertRaises(OSError):
                transaction.rename_exchange(
                    root_fd,
                    "source",
                    ".transactions/n1/generation",
                )
        finally:
            os.close(root_fd)
        self.assertEqual(self.read_marker(root / "source"), "old")
        self.assertEqual(self.read_marker(outside / "generation"), "outside")

    def test_exchange_partial_acquisition_closes_open_file_descriptors(self):
        root = self.make_root()
        self.write_generation(root / "source", marker="old")
        self.write_generation(
            root / ".transactions/n1/generation", marker="new"
        )
        root_fd = transaction.open_directory_no_follow(root)
        self.addCleanup(os.close, root_fd)

        real_dup = os.dup
        dup_calls = []

        def fail_second_dup(fd):
            dup_calls.append(fd)
            if len(dup_calls) == 2:
                raise OSError(errno.EMFILE, "forced dup failure")
            return real_dup(fd)

        before = len(os.listdir("/proc/self/fd"))
        with mock.patch.object(
            transaction.os, "dup", side_effect=fail_second_dup
        ):
            with self.assertRaises(OSError):
                transaction.rename_exchange(
                    root_fd,
                    "source",
                    ".transactions/n1/generation",
                )
        self.assertEqual(len(os.listdir("/proc/self/fd")), before)

        real_open = os.open

        def fail_nested_open(path, flags, *args, **kwargs):
            if path == "n1":
                raise OSError(errno.EMFILE, "forced open failure")
            return real_open(path, flags, *args, **kwargs)

        before = len(os.listdir("/proc/self/fd"))
        with mock.patch.object(
            transaction.os, "open", side_effect=fail_nested_open
        ):
            with self.assertRaises(OSError):
                transaction.rename_exchange(
                    root_fd,
                    "source",
                    ".transactions/n1/generation",
                )
        self.assertEqual(len(os.listdir("/proc/self/fd")), before)

    def test_exchange_implementation_contains_no_rename_fallback(self):
        source = inspect.getsource(
            transaction._rename_exchange_at
        ) + inspect.getsource(transaction.rename_exchange)
        self.assertIn("_RENAMEAT2", source)
        self.assertNotIn("os.rename", source)
        self.assertNotIn("os.replace", source)

    def test_first_source_promotion_uses_single_rename(self):
        root = self.make_root()
        generation = root / ".transactions/n1/generation"
        self.write_file(generation, "tools/a.py", b"new\n")
        receipt = self.make_receipt(generation)
        result = transaction.promote_generation(
            root, ".transactions/n1/generation", receipt
        )
        self.assertEqual(result.nonce, "n1")
        self.assertEqual((root / "source/tools/a.py").read_bytes(), b"new\n")
        self.assertFalse(generation.exists())
        self.read_committed(root)

    def test_first_source_promotion_rejects_swapped_generation_entry(self):
        root = self.make_root()
        nonce_root = root / ".transactions/n1"
        generation = nonce_root / "generation"
        self.write_file(generation, "tools/a.py", b"validated\n")
        receipt = self.make_receipt(generation)

        def swap_generation(point):
            if point != "before_exchange":
                return
            generation.rename(nonce_root / "validated-generation")
            self.write_file(generation, "tools/a.py", b"swapped\n")

        with self.assertRaises(transaction.TransactionError):
            transaction.promote_generation(
                root,
                ".transactions/n1/generation",
                receipt,
                fault_injector=swap_generation,
            )
        self.assertFalse((root / "source").exists())
        self.assertEqual(
            (nonce_root / "validated-generation/tools/a.py").read_bytes(),
            b"validated\n",
        )

    def test_first_source_after_exchange_failure_remains_committed(self):
        root = self.make_root()
        generation = root / ".transactions/n1/generation"
        self.write_file(generation, "tools/a.py", b"new\n")
        receipt = self.make_receipt(generation)

        def fail_after_exchange(point):
            if point == "after_exchange":
                raise RuntimeError("simulated response loss")

        with self.assertRaisesRegex(RuntimeError, "response loss"):
            transaction.promote_generation(
                root,
                ".transactions/n1/generation",
                receipt,
                fault_injector=fail_after_exchange,
            )
        committed = self.read_committed(root)
        self.assertEqual(committed.created_at_unix_ns, receipt.created_at_unix_ns)
        self.assertEqual((root / "source/tools/a.py").read_bytes(), b"new\n")
        self.assertFalse(generation.exists())

    def test_existing_source_promotion_exchanges_directories(self):
        root = self.make_root()
        self.write_file(root / "source", "old.txt", b"old\n")
        generation = root / ".transactions/n1/generation"
        self.write_file(generation, "tools/a.py", b"new\n")
        receipt = self.make_receipt(generation)
        transaction.promote_generation(
            root, ".transactions/n1/generation", receipt
        )
        self.assertEqual((root / "source/tools/a.py").read_bytes(), b"new\n")
        self.assertFalse((root / "source/old.txt").exists())
        self.read_committed(root)

    def test_existing_source_uses_held_parent_after_nonce_replacement(self):
        root = self.make_root()
        self.write_file(root / "source", "old.txt", b"old\n")
        nonce_root = root / ".transactions/n1"
        generation = nonce_root / "generation"
        self.write_file(generation, "tools/a.py", b"validated\n")
        receipt = self.make_receipt(generation)
        real_public_exchange = transaction.rename_exchange
        raced = []

        def race_exchange(*args):
            if not raced:
                raced.append(True)
                held_nonce_root = root / ".transactions/n1-held"
                nonce_root.rename(held_nonce_root)
                replacement = root / ".transactions/n1/generation"
                self.write_file(replacement, "tools/a.py", b"unexpected\n")
                alternate = self.replace_receipt_fields(
                    self.make_receipt(replacement),
                    created_at_unix_ns=receipt.created_at_unix_ns + 1,
                )
                replacement_fd = transaction.open_directory_no_follow(
                    replacement
                )
                try:
                    transaction.write_embedded_receipt(
                        replacement_fd, alternate
                    )
                finally:
                    os.close(replacement_fd)
            if len(args) == 3:
                return real_public_exchange(*args)
            return self.raw_rename_exchange(*args)

        with mock.patch.object(
            transaction,
            "_rename_exchange_at",
            side_effect=race_exchange,
            create=True,
        ), mock.patch.object(
            transaction, "rename_exchange", side_effect=race_exchange
        ):
            result = transaction.promote_generation(
                root, ".transactions/n1/generation", receipt
            )

        self.assertEqual(result.created_at_unix_ns, receipt.created_at_unix_ns)
        self.assertEqual(
            (root / "source/tools/a.py").read_bytes(), b"validated\n"
        )
        self.assertFalse((root / "source/old.txt").exists())
        self.assertEqual(
            (
                root / ".transactions/n1/generation/tools/a.py"
            ).read_bytes(),
            b"unexpected\n",
        )
        self.assertFalse(
            (root / ".transactions/n1-held/generation").exists()
        )

    def test_existing_source_after_exchange_failure_remains_committed(self):
        root = self.make_root()
        self.write_file(root / "source", "old.txt", b"old\n")
        generation = root / ".transactions/n1/generation"
        self.write_file(generation, "tools/a.py", b"new\n")
        receipt = self.make_receipt(generation)

        def fail_after_exchange(point):
            if point == "after_exchange":
                raise RuntimeError("simulated response loss")

        with self.assertRaisesRegex(RuntimeError, "response loss"):
            transaction.promote_generation(
                root,
                ".transactions/n1/generation",
                receipt,
                fault_injector=fail_after_exchange,
            )
        committed = self.read_committed(root)
        self.assertEqual(committed.created_at_unix_ns, receipt.created_at_unix_ns)
        self.assertEqual((root / "source/tools/a.py").read_bytes(), b"new\n")
        self.assertFalse((root / "source/old.txt").exists())
        self.assertFalse(generation.exists())

    def test_initial_commit_pre_exchange_faults_preserve_source_and_receipts(
        self,
    ):
        for fault_point in PRE_EXCHANGE_FAULT_POINTS:
            with self.subTest(fault_point=fault_point):
                root = self.make_root("init-pre-{}".format(fault_point))
                self.write_file(root / "source", "old.txt", b"old\n")
                self.write_detached_receipts(root, "old")
                generation = (
                    root / ".transactions/init-nonce/generation"
                )
                self.write_file(generation, "tools/new.py", b"new\n")
                source_before = self.tree_snapshot(root / "source")
                receipts_before = self.tree_snapshot(root / "receipts")

                def inject(point):
                    if point == fault_point:
                        raise RuntimeError("fault at {}".format(point))

                with self.assertRaisesRegex(
                    RuntimeError, "fault at {}".format(fault_point)
                ):
                    transaction.commit_initial_generation(
                        root,
                        ".transactions/init-nonce/generation",
                        "a" * 40,
                        fault_injector=inject,
                    )

                self.assertEqual(
                    self.tree_snapshot(root / "source"),
                    source_before,
                )
                self.assertEqual(
                    self.tree_snapshot(root / "receipts"),
                    receipts_before,
                )

    def test_initial_commit_post_exchange_faults_confirm_and_rebuild_receipts(
        self,
    ):
        for fault_point in POST_EXCHANGE_FAULT_POINTS:
            with self.subTest(fault_point=fault_point):
                root = self.make_root("init-post-{}".format(fault_point))
                self.write_file(root / "source", "old.txt", b"old\n")
                self.write_detached_receipts(root, "old")
                generation_name = (
                    ".transactions/init-nonce/generation"
                )
                generation = root / generation_name
                self.write_file(generation, "tools/new.py", b"new\n")

                def inject(point):
                    if point == fault_point:
                        raise RuntimeError("fault at {}".format(point))

                with self.assertRaisesRegex(
                    RuntimeError, "fault at {}".format(fault_point)
                ):
                    transaction.commit_initial_generation(
                        root,
                        generation_name,
                        "a" * 40,
                        fault_injector=inject,
                    )

                committed = transaction.read_committed_generation(
                    root,
                    expected_nonce="init-nonce",
                    expected_operation="init",
                    expected_head="a" * 40,
                    expected_paths=(),
                )
                self.assertEqual(
                    (root / "source/tools/new.py").read_bytes(),
                    b"new\n",
                )
                for name in (
                    "source-head.txt",
                    "source-files.sha256",
                    "source-transaction.txt",
                ):
                    try:
                        (root / "receipts" / name).unlink()
                    except FileNotFoundError:
                        pass

                confirmed = transaction.commit_initial_generation(
                    root,
                    generation_name,
                    "a" * 40,
                )
                self.assertEqual(
                    confirmed.created_at_unix_ns,
                    committed.created_at_unix_ns,
                )
                self.assert_initial_detached_receipts(root, confirmed)
                self.assertFalse(generation.exists())

    def test_initial_confirmation_remove_tree_failure_preserves_commit(self):
        root = self.make_root()
        generation_name = ".transactions/init-nonce/generation"
        generation = root / generation_name
        self.write_file(generation, "tools/new.py", b"committed\n")
        committed = transaction.commit_initial_generation(
            root,
            generation_name,
            "a" * 40,
        )
        self.write_file(generation, "residue.txt", b"old generation\n")
        for name in (
            "source-head.txt",
            "source-files.sha256",
            "source-transaction.txt",
        ):
            (root / "receipts" / name).unlink()
        source_before = self.tree_snapshot(root / "source")
        before = len(os.listdir("/dev/fd"))

        with mock.patch.object(
            transaction,
            "_remove_tree_at",
            side_effect=OSError(errno.EIO, "forced cleanup failure"),
        ):
            confirmed = transaction.commit_initial_generation(
                root,
                generation_name,
                "a" * 40,
            )

        transaction._require_exact_receipt(confirmed, committed)
        self.assert_initial_detached_receipts(root, confirmed)
        self.assertEqual(self.tree_snapshot(root / "source"), source_before)
        self.assertEqual(
            generation.joinpath("residue.txt").read_bytes(),
            b"old generation\n",
        )
        self.assertEqual(len(os.listdir("/dev/fd")), before)

    def test_initial_confirmation_nonce_cleanup_failure_preserves_commit(self):
        root = self.make_root()
        generation_name = ".transactions/init-nonce/generation"
        generation = root / generation_name
        self.write_file(generation, "tools/new.py", b"committed\n")
        committed = transaction.commit_initial_generation(
            root,
            generation_name,
            "a" * 40,
        )
        for name in (
            "source-head.txt",
            "source-files.sha256",
            "source-transaction.txt",
        ):
            (root / "receipts" / name).unlink()
        source_before = self.tree_snapshot(root / "source")
        before = len(os.listdir("/dev/fd"))

        with mock.patch.object(
            transaction,
            "_remove_empty_nonce_directory",
            side_effect=RuntimeError("forced nonce cleanup failure"),
        ):
            confirmed = transaction.commit_initial_generation(
                root,
                generation_name,
                "a" * 40,
            )

        transaction._require_exact_receipt(confirmed, committed)
        self.assert_initial_detached_receipts(root, confirmed)
        self.assertEqual(self.tree_snapshot(root / "source"), source_before)
        self.assertFalse(generation.exists())
        self.assertEqual(len(os.listdir("/dev/fd")), before)

    def test_init_commit_cli_promotes_generation_and_prints_file_count(self):
        root = self.make_root()
        generation = root / ".transactions/cli-nonce/generation"
        self.write_file(generation, "tools/a.py", b"new\n")
        with mock.patch("sys.stdout") as stdout:
            result = transaction.main(
                (
                    "init-commit",
                    "--remote-root",
                    str(root),
                    "--generation",
                    ".transactions/cli-nonce/generation",
                    "--source-head",
                    "b" * 40,
                )
            )
        self.assertEqual(result, 0)
        stdout.write.assert_called_once_with("1\n")
        transaction.read_committed_generation(
            root,
            expected_nonce="cli-nonce",
            expected_operation="init",
            expected_head="b" * 40,
            expected_paths=(),
        )

    def test_init_commit_cli_returns_conventional_signal_exit_codes(self):
        for signum, expected_status in (
            (signal.SIGINT, 130),
            (signal.SIGTERM, 143),
        ):
            with self.subTest(signum=signum):
                root = self.make_root("cli-signal-{}".format(signum))

                def interrupt_while_locked(*args, **kwargs):
                    del args, kwargs
                    with transaction.locked_remote_root(root):
                        raise transaction.TransactionInterrupted(signum)

                with mock.patch.object(
                    transaction,
                    "commit_initial_generation",
                    side_effect=interrupt_while_locked,
                ), mock.patch("sys.stderr"):
                    result = transaction.main(
                        (
                            "init-commit",
                            "--remote-root",
                            str(root),
                            "--generation",
                            ".transactions/signal-nonce/generation",
                            "--source-head",
                            "f" * 40,
                        )
                    )

                self.assertEqual(result, expected_status)
                holder = LockHolder(root)
                try:
                    holder.wait_until_locked()
                    holder.release_normally()
                    self.assertEqual(holder.wait(), 0)
                finally:
                    if holder.process.poll() is None:
                        holder.terminate()
                        holder.wait()

    def test_initial_commit_response_is_decided_under_promotion_lock(self):
        root = self.make_root()
        first_generation = (
            root / ".transactions/first-nonce/generation"
        )
        second_generation = (
            root / ".transactions/second-nonce/generation"
        )
        self.write_file(first_generation, "tools/first.py", b"first\n")
        self.write_file(second_generation, "tools/second.py", b"second\n")

        first_at_response = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        second_done = threading.Event()
        first_outcome = {}
        second_outcome = {}
        real_read = transaction.read_committed_generation

        def pause_first(point):
            if point == "before_old_generation_cleanup":
                first_at_response.set()
                self.assertTrue(release_first.wait(10))

        def first_writer():
            try:
                first_outcome["receipt"] = (
                    transaction.commit_initial_generation(
                        root,
                        ".transactions/first-nonce/generation",
                        "1" * 40,
                        fault_injector=pause_first,
                    )
                )
            except BaseException as exc:
                first_outcome["error"] = exc

        def second_writer():
            second_started.set()
            try:
                second_outcome["receipt"] = (
                    transaction.commit_initial_generation(
                        root,
                        ".transactions/second-nonce/generation",
                        "2" * 40,
                    )
                )
            except BaseException as exc:
                second_outcome["error"] = exc
            finally:
                second_done.set()

        first_thread = threading.Thread(target=first_writer)
        second_thread = threading.Thread(target=second_writer)

        def delay_unlocked_first_read(*args, **kwargs):
            if threading.current_thread() is first_thread:
                self.assertTrue(second_done.wait(10))
            return real_read(*args, **kwargs)

        with mock.patch.object(
            transaction,
            "_transaction_signal_handlers",
            side_effect=lambda: contextlib.nullcontext(),
        ), mock.patch.object(
            transaction,
            "read_committed_generation",
            side_effect=delay_unlocked_first_read,
        ):
            first_thread.start()
            self.assertTrue(first_at_response.wait(10))
            second_thread.start()
            self.assertTrue(second_started.wait(10))
            self.assertFalse(second_done.wait(0.2))
            release_first.set()
            first_thread.join(10)
            second_thread.join(10)

        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())
        self.assertNotIn("error", first_outcome)
        self.assertNotIn("error", second_outcome)
        self.assertEqual(first_outcome["receipt"].nonce, "first-nonce")
        self.assertEqual(second_outcome["receipt"].nonce, "second-nonce")

    def test_initial_commit_reuses_valid_embedded_receipt_before_exchange(self):
        root = self.make_root()
        generation_name = ".transactions/reentry-nonce/generation"
        generation = root / generation_name
        self.write_file(generation, "tools/a.py", b"new\n")
        generation_fd = transaction.open_directory_no_follow(generation)
        try:
            receipt = transaction._generation_receipt(
                generation_fd,
                operation="init",
                nonce="reentry-nonce",
                source_head="c" * 40,
                explicit_paths=(),
            )
            transaction.write_embedded_receipt(generation_fd, receipt)
        finally:
            os.close(generation_fd)

        result = transaction.commit_initial_generation(
            root,
            generation_name,
            "c" * 40,
        )

        self.assertEqual(
            result.created_at_unix_ns,
            receipt.created_at_unix_ns,
        )
        self.assert_initial_detached_receipts(root, result)
        self.assertFalse(generation.exists())

    def test_sync_clone_preserves_symlinks_and_applies_only_explicit_delta(self):
        root = self.make_initialized_root()
        delta = self.stage_delta(
            root,
            "sync-clone",
            (("tools/a.py", b"new-a\n"),),
        )

        receipt = transaction.commit_sync_generation(
            root,
            delta,
            nonce="sync-clone",
            source_head="b" * 40,
            explicit_paths=("tools/a.py",),
        )

        self.assertEqual(receipt.operation, "sync")
        self.assertEqual((root / "source/tools/a.py").read_bytes(), b"new-a\n")
        self.assertEqual((root / "source/tools/b.py").read_bytes(), b"old-b\n")
        self.assertTrue((root / "source/current.py").is_symlink())
        self.assertEqual(os.readlink(root / "source/current.py"), "tools/a.py")
        transaction.read_committed_generation(
            root,
            expected_nonce="sync-clone",
            expected_operation="sync",
            expected_head="b" * 40,
            expected_paths=("tools/a.py",),
        )

    def test_sync_remote_parent_symlink_fails_without_outside_write(self):
        root = self.make_root()
        outside = self.test_root / "outside-sync-parent"
        outside.mkdir()
        outside_file = outside / "allowed.py"
        outside_file.write_bytes(b"outside-old\n")
        generation = root / ".transactions/init-nonce/generation"
        self.write_file(generation, "tools/a.py", b"old-a\n")
        source_parent = generation / "pkg"
        source_parent.symlink_to(outside, target_is_directory=True)
        transaction.commit_initial_generation(
            root,
            ".transactions/init-nonce/generation",
            "a" * 40,
        )
        source_parent = root / "source/pkg"
        delta = self.stage_delta(
            root,
            "sync-parent-link",
            (("pkg/allowed.py", b"outside-new\n"),),
            paths=("pkg/allowed.py",),
        )

        with self.assertRaisesRegex(
            transaction.TransactionError,
            "full init",
        ):
            transaction.commit_sync_generation(
                root,
                delta,
                nonce="sync-parent-link",
                source_head="b" * 40,
                explicit_paths=("pkg/allowed.py",),
            )

        self.assertEqual(outside_file.read_bytes(), b"outside-old\n")
        self.assertTrue(source_parent.is_symlink())
        self.assertFalse(
            (root / "receipts/sync-sync-parent-link.state").exists()
        )

    def test_sync_delta_staging_rejects_nonce_parent_symlink_escape(self):
        root = self.make_root()
        outside = self.test_root / "outside-delta-stage"
        outside.mkdir()
        (root / ".transactions").symlink_to(
            outside, target_is_directory=True
        )

        with self.assertRaises((OSError, transaction.TransactionError)):
            transaction._stage_delta_stream(
                root,
                "valid-nonce",
                ("tools/a.py",),
                io.BytesIO(
                    self.make_delta_archive(
                        (("tools/a.py", b"new\n"),)
                    )
                ),
            )

        self.assertEqual(list(outside.iterdir()), [])

    def test_sync_delta_staging_rejects_valid_nonce_final_symlink(self):
        root = self.make_root()
        transactions = root / ".transactions"
        transactions.mkdir()
        outside = self.test_root / "outside-delta-final"
        outside.mkdir()

        with mock.patch.object(
            transaction,
            "_private_delta_name",
            return_value=".delta-valid-nonce-fixed",
            create=True,
        ):
            (transactions / ".delta-valid-nonce-fixed").symlink_to(
                outside, target_is_directory=True
            )
            with self.assertRaises((OSError, transaction.TransactionError)):
                transaction._stage_delta_stream(
                    root,
                    "valid-nonce",
                    ("tools/a.py",),
                    io.BytesIO(
                        self.make_delta_archive(
                            (("tools/a.py", b"new\n"),)
                        )
                    ),
                )

        self.assertTrue(
            (transactions / ".delta-valid-nonce-fixed").is_symlink()
        )
        self.assertEqual(list(outside.iterdir()), [])

    def test_sync_delta_archive_requires_exact_canonical_membership(self):
        cases = (
            ("missing", (), ("tools/a.py",)),
            (
                "unexpected",
                (
                    ("tools/a.py", b"a\n"),
                    ("tools/b.py", b"b\n"),
                ),
                ("tools/a.py",),
            ),
            (
                "duplicate",
                (
                    ("tools/a.py", b"a\n"),
                    ("tools/a.py", b"b\n"),
                ),
                ("tools/a.py",),
            ),
            (
                "dot-alias",
                (("./tools/a.py", b"a\n"),),
                ("tools/a.py",),
            ),
            (
                "double-slash-alias",
                (("tools//a.py", b"a\n"),),
                ("tools/a.py",),
            ),
        )
        for label, members, paths in cases:
            with self.subTest(label=label):
                root = self.make_root("archive-{}".format(label))
                with self.assertRaises(transaction.TransactionError):
                    transaction._stage_delta_stream(
                        root,
                        "archive-{}".format(label),
                        paths,
                        io.BytesIO(self.make_delta_archive(members)),
                    )
                transactions = root / ".transactions"
                if transactions.exists():
                    self.assertEqual(
                        [
                            path.name
                            for path in transactions.iterdir()
                            if path.name != "source.lock"
                        ],
                        [],
                    )

    def test_sync_commit_rejects_outside_delta_root(self):
        root = self.make_initialized_root()
        outside = self.make_delta("outside-authority")
        self.write_file(outside, "tools/a.py", b"outside\n")
        source_before = self.tree_snapshot(root / "source")

        with self.assertRaises(transaction.TransactionError):
            transaction.commit_sync_generation(
                root,
                outside,
                nonce="outside-authority",
                source_head="b" * 40,
                explicit_paths=("tools/a.py",),
            )

        self.assertEqual(self.tree_snapshot(root / "source"), source_before)

    def test_sync_commit_rejects_transactions_parent_swap_after_staging(self):
        root = self.make_initialized_root()
        staged = self.stage_delta(
            root,
            "parent-swap",
            (("tools/a.py", b"validated\n"),),
        )
        staged_name = Path(os.fspath(staged)).name
        transactions = root / ".transactions"
        held_transactions = root / ".transactions-held"
        transactions.rename(held_transactions)
        replacement = root / ".transactions" / staged_name
        self.write_file(replacement, "tools/a.py", b"swapped\n")
        source_before = self.tree_snapshot(root / "source")

        with self.assertRaises(transaction.TransactionError):
            transaction.commit_sync_generation(
                root,
                staged,
                nonce="parent-swap",
                source_head="b" * 40,
                explicit_paths=("tools/a.py",),
            )

        self.assertEqual(self.tree_snapshot(root / "source"), source_before)
        self.assertEqual(
            (held_transactions / staged_name / "tools/a.py").read_bytes(),
            b"validated\n",
        )
        self.assertEqual(
            (replacement / "tools/a.py").read_bytes(),
            b"swapped\n",
        )

    def test_init_and_sync_serialize_on_the_same_flock(self):
        root = self.make_initialized_root()
        delta = self.stage_delta(
            root,
            "sync-lock",
            (("tools/a.py", b"sync\n"),),
        )
        init_generation = root / ".transactions/init-two/generation"
        self.write_file(init_generation, "tools/init.py", b"init\n")
        sync_locked = threading.Event()
        release_sync = threading.Event()
        init_started = threading.Event()
        init_done = threading.Event()
        outcomes = {}

        def pause_sync(point):
            if point == "after_lock":
                sync_locked.set()
                self.assertTrue(release_sync.wait(10))

        def run_sync():
            try:
                outcomes["sync"] = transaction.commit_sync_generation(
                    root,
                    delta,
                    nonce="sync-lock",
                    source_head="b" * 40,
                    explicit_paths=("tools/a.py",),
                    fault_injector=pause_sync,
                )
            except BaseException as exc:
                outcomes["sync_error"] = exc

        def run_init():
            init_started.set()
            try:
                outcomes["init"] = transaction.commit_initial_generation(
                    root,
                    ".transactions/init-two/generation",
                    "c" * 40,
                )
            except BaseException as exc:
                outcomes["init_error"] = exc
            finally:
                init_done.set()

        with mock.patch.object(
            transaction,
            "_transaction_signal_handlers",
            side_effect=lambda *args, **kwargs: contextlib.nullcontext(),
        ):
            sync_thread = threading.Thread(target=run_sync)
            init_thread = threading.Thread(target=run_init)
            sync_thread.start()
            self.assertTrue(sync_locked.wait(10))
            init_thread.start()
            self.assertTrue(init_started.wait(10))
            self.assertFalse(init_done.wait(0.2))
            release_sync.set()
            sync_thread.join(10)
            init_thread.join(10)

        self.assertFalse(sync_thread.is_alive())
        self.assertFalse(init_thread.is_alive())
        self.assertNotIn("sync_error", outcomes)
        self.assertNotIn("init_error", outcomes)
        self.assertEqual(outcomes["sync"].nonce, "sync-lock")
        self.assertEqual(outcomes["init"].nonce, "init-two")

    def test_sync_same_nonce_reentry_uses_embedded_truth_and_repairs_receipts(
        self,
    ):
        root = self.make_initialized_root()
        delta = self.stage_delta(
            root,
            "same-nonce",
            (("tools/a.py", b"once\n"),),
        )
        first = transaction.commit_sync_generation(
            root,
            delta,
            nonce="same-nonce",
            source_head="b" * 40,
            explicit_paths=("tools/a.py",),
        )
        receipts = root / "receipts"
        for path in receipts.glob("sync-same-nonce.*"):
            path.unlink()
        second_delta = self.stage_delta(
            root,
            "same-nonce",
            (("tools/a.py", b"unexpected\n"),),
        )

        second = transaction.commit_sync_generation(
            root,
            second_delta,
            nonce="same-nonce",
            source_head="b" * 40,
            explicit_paths=("tools/a.py",),
        )

        transaction._require_exact_receipt(second, first)
        self.assertEqual((root / "source/tools/a.py").read_bytes(), b"once\n")
        for suffix in ("paths.txt", "sha256", "state"):
            self.assertTrue(
                (receipts / "sync-same-nonce.{}".format(suffix)).is_file()
            )

    def test_sync_confirmation_rejects_detached_receipt_symlink(self):
        root = self.make_initialized_root()
        delta = self.stage_delta(
            root,
            "receipt-link",
            (("tools/a.py", b"committed\n"),),
        )
        transaction.commit_sync_generation(
            root,
            delta,
            nonce="receipt-link",
            source_head="b" * 40,
            explicit_paths=("tools/a.py",),
        )
        receipt_path = root / "receipts/sync-receipt-link.state"
        target = root / "receipts/state-target"
        target.write_text("committed\n", encoding="utf-8")
        receipt_path.unlink()
        receipt_path.symlink_to(target.name)

        with self.assertRaises(transaction.TransactionError):
            transaction.confirm_committed_generation(
                root,
                nonce="receipt-link",
                operation="sync",
                source_head="b" * 40,
                explicit_paths=("tools/a.py",),
            )

        self.assertTrue(receipt_path.is_symlink())
        self.assertEqual(target.read_text(encoding="utf-8"), "committed\n")

    def test_sync_confirmation_rejects_mismatched_explicit_paths(self):
        root = self.make_initialized_root()
        delta = self.stage_delta(
            root,
            "path-mismatch",
            (("tools/a.py", b"committed\n"),),
        )
        transaction.commit_sync_generation(
            root,
            delta,
            nonce="path-mismatch",
            source_head="b" * 40,
            explicit_paths=("tools/a.py",),
        )

        with self.assertRaises(transaction.TransactionError):
            transaction.confirm_committed_generation(
                root,
                nonce="path-mismatch",
                operation="sync",
                source_head="b" * 40,
                explicit_paths=("tools/b.py",),
            )

    def test_sync_helper_preserves_success_after_exchange_event(self):
        for event in ("exception", "sigterm"):
            with self.subTest(event=event):
                root = self.make_initialized_root(
                    "post-exchange-{}".format(event)
                )
                nonce = "post-exchange-{}".format(event)
                archive = self.make_delta_archive(
                    (("tools/a.py", b"committed\n"),)
                )

                result = self.run_helper_with_post_exchange_event(
                    root, nonce, archive, event
                )

                expected = (
                    str(
                        root
                        / "receipts/sync-{}.sha256".format(nonce)
                    )
                    + "\n"
                ).encode("utf-8")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected)
                self.assertEqual(result.stderr, b"")
                self.assertEqual(
                    (root / "source/tools/a.py").read_bytes(),
                    b"committed\n",
                )

    def test_sync_helper_closes_real_signal_commit_windows(self):
        for event in (
            "exchange_return_sigterm",
            "teardown_pending_sigterm",
            "post_drain_sigterm",
            "staged_cleanup_sigterm",
            "before_stdout_sigterm",
        ):
            with self.subTest(event=event):
                root = self.make_initialized_root(
                    "signal-window-{}".format(event)
                )
                nonce = "signal-window-{}".format(event)
                archive = self.make_delta_archive(
                    (("tools/a.py", b"committed\n"),)
                )

                result = self.run_helper_with_post_exchange_event(
                    root, nonce, archive, event
                )

                expected = (
                    str(
                        root
                        / "receipts/sync-{}.sha256".format(nonce)
                    )
                    + "\n"
                ).encode("utf-8")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected)
                self.assertEqual(result.stderr, b"")
                self.assertEqual(
                    (root / "source/tools/a.py").read_bytes(),
                    b"committed\n",
                )

    def test_sync_helper_pre_exchange_sigterm_remains_failure(self):
        root = self.make_initialized_root("signal-window-pre-exchange")
        source_before = self.tree_snapshot(root / "source")
        receipts_before = self.tree_snapshot(root / "receipts")
        archive = self.make_delta_archive(
            (("tools/a.py", b"committed\n"),)
        )

        result = self.run_helper_with_post_exchange_event(
            root,
            "signal-window-pre-exchange",
            archive,
            "before_exchange_sigterm",
        )

        self.assertEqual(result.returncode, 143)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"transaction interrupted by signal 15", result.stderr)
        self.assertEqual(self.tree_snapshot(root / "source"), source_before)
        self.assertEqual(
            self.tree_snapshot(root / "receipts"), receipts_before
        )

    def test_second_cleanup_exception_does_not_leave_sync_lock_held(self):
        root = self.make_initialized_root()
        delta = self.stage_delta(
            root,
            "cleanup-signal",
            (("tools/a.py", b"committed\n"),),
        )

        def deliver_signal(point):
            if point == "before_old_generation_cleanup":
                os.kill(os.getpid(), signal.SIGTERM)

        receipt = transaction.commit_sync_generation(
            root,
            delta,
            nonce="cleanup-signal",
            source_head="b" * 40,
            explicit_paths=("tools/a.py",),
            fault_injector=deliver_signal,
        )

        self.assertEqual(receipt.nonce, "cleanup-signal")
        with transaction.locked_remote_root(root):
            pass

    def test_sync_fault_points_preserve_pre_or_post_exchange_invariants(self):
        for fault_point in SYNC_FAULT_POINTS:
            with self.subTest(fault_point=fault_point):
                root = self.make_initialized_root(
                    "sync-fault-{}".format(fault_point)
                )
                nonce = "fault-{}".format(fault_point)
                delta = self.stage_delta(
                    root,
                    nonce,
                    (("tools/a.py", b"committed\n"),),
                )
                source_before = self.tree_snapshot(root / "source")
                receipts_before = self.tree_snapshot(root / "receipts")

                if fault_point in PRE_EXCHANGE_FAULT_POINTS:
                    with self.assertRaises(transaction.InjectedFailure):
                        transaction.commit_sync_generation(
                            root,
                            delta,
                            nonce=nonce,
                            source_head="b" * 40,
                            explicit_paths=("tools/a.py",),
                            fault_injector=self.fail_at(fault_point),
                        )
                    self.assertEqual(
                        self.tree_snapshot(root / "source"),
                        source_before,
                    )
                    self.assertEqual(
                        self.tree_snapshot(root / "receipts"),
                        receipts_before,
                    )
                    continue

                receipt = transaction.commit_sync_generation(
                    root,
                    delta,
                    nonce=nonce,
                    source_head="b" * 40,
                    explicit_paths=("tools/a.py",),
                    fault_injector=self.fail_at(fault_point),
                )

                self.assertEqual(receipt.nonce, nonce)
                self.assertEqual(
                    (root / "source/tools/a.py").read_bytes(),
                    b"committed\n",
                )
                for suffix in ("paths.txt", "sha256", "state"):
                    self.assertTrue(
                        (
                            root
                            / "receipts/sync-{}.{}".format(nonce, suffix)
                        ).is_file()
                    )

    def test_sync_preexisting_new_nonce_receipt_symlink_never_commits_and_fails(
        self,
    ):
        root = self.make_initialized_root()
        source_before = self.tree_snapshot(root / "source")
        target = root / "receipts/target.state"
        target.write_text("outside\n", encoding="utf-8")
        receipt = root / "receipts/sync-topology.state"
        receipt.symlink_to(target.name)
        archive = self.make_delta_archive(
            (("tools/a.py", b"committed\n"),)
        )
        result = subprocess.run(
            self.helper_command(
                root,
                "sync-commit",
                "--nonce",
                "topology",
                "--source-head",
                "b" * 40,
                "--path",
                "tools/a.py",
            ),
            input=archive,
            capture_output=True,
            env=self.helper_environment(),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertNotEqual(result.stderr, b"")
        self.assertEqual(self.tree_snapshot(root / "source"), source_before)
        self.assertTrue(receipt.is_symlink())
        self.assertEqual(target.read_text(encoding="utf-8"), "outside\n")

    def test_sync_same_nonce_concurrent_helper_processes_both_succeed(self):
        root = self.make_initialized_root()
        nonce = "concurrent-same-nonce"
        payload = b"x" * (2 * 1024 * 1024)
        archive = self.make_delta_archive((("tools/a.py", payload),))
        command = self.helper_command(
            root,
            "sync-commit",
            "--nonce",
            nonce,
            "--source-head",
            "b" * 40,
            "--path",
            "tools/a.py",
        )
        processes = [
            subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.helper_environment(),
            )
            for _ in range(2)
        ]
        outcomes = [None, None]
        start = threading.Barrier(3)

        def communicate(index):
            start.wait()
            outcomes[index] = processes[index].communicate(
                input=archive, timeout=20
            )

        threads = [
            threading.Thread(target=communicate, args=(index,))
            for index in range(2)
        ]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join(25)

        expected = (
            str(root / "receipts/sync-{}.sha256".format(nonce)) + "\n"
        ).encode("utf-8")
        for index, process in enumerate(processes):
            with self.subTest(process=index):
                self.assertFalse(threads[index].is_alive())
                stdout, stderr = outcomes[index]
                self.assertEqual(process.returncode, 0, stderr)
                self.assertEqual(stdout, expected)
                self.assertEqual(stderr, b"")
        self.assertEqual((root / "source/tools/a.py").read_bytes(), payload)
        transactions = root / ".transactions"
        self.assertEqual(
            [
                path.name
                for path in transactions.iterdir()
                if path.name != "source.lock"
            ],
            [],
        )

    def test_sync_confirm_and_status_cli_contracts(self):
        root = self.make_initialized_root()
        archive = self.make_delta_archive(
            (("tools/a.py", b"cli-committed\n"),)
        )
        nonce = "cli-contract"
        sync = subprocess.run(
            self.helper_command(
                root,
                "sync-commit",
                "--nonce",
                nonce,
                "--source-head",
                "b" * 40,
                "--path",
                "tools/a.py",
            ),
            input=archive,
            capture_output=True,
            env=self.helper_environment(),
        )
        receipt = str(root / "receipts/sync-{}.sha256".format(nonce))
        self.assertEqual(sync.returncode, 0, sync.stderr)
        self.assertEqual(sync.stdout, (receipt + "\n").encode("utf-8"))
        self.assertEqual(sync.stderr, b"")

        confirm = subprocess.run(
            self.helper_command(
                root,
                "confirm",
                "--nonce",
                nonce,
                "--operation",
                "sync",
                "--source-head",
                "b" * 40,
                "--path",
                "tools/a.py",
            ),
            capture_output=True,
            env=self.helper_environment(),
        )
        self.assertEqual(confirm.returncode, 0, confirm.stderr)
        self.assertEqual(confirm.stdout, (receipt + "\n").encode("utf-8"))
        self.assertEqual(confirm.stderr, b"")

        committed = transaction.read_committed_generation(root)
        status = subprocess.run(
            self.helper_command(root, "status"),
            capture_output=True,
            env=self.helper_environment(),
        )
        expected_status = (
            "head={}\ncount={}\nlatest={}\n".format(
                committed.source_head,
                committed.source_file_count,
                receipt,
            )
        ).encode("utf-8")
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertEqual(status.stdout, expected_status)
        self.assertEqual(status.stderr, b"")

    def test_initial_confirmation_receipt_symlink_fails_without_exchange(self):
        root = self.make_root()
        generation_name = ".transactions/symlink-nonce/generation"
        generation = root / generation_name
        self.write_file(generation, "tools/a.py", b"committed\n")
        transaction.commit_initial_generation(
            root,
            generation_name,
            "d" * 40,
        )
        self.write_file(generation, "tools/a.py", b"unexpected\n")
        head_receipt = root / "receipts/source-head.txt"
        head_target = root / "receipts/source-head-target.txt"
        head_receipt.replace(head_target)
        head_receipt.symlink_to(head_target.name)

        with self.assertRaises(transaction.TransactionError):
            transaction.commit_initial_generation(
                root,
                generation_name,
                "d" * 40,
            )

        self.assertEqual(
            (root / "source/tools/a.py").read_bytes(),
            b"committed\n",
        )
        self.assertEqual(
            (generation / "tools/a.py").read_bytes(),
            b"unexpected\n",
        )

    def test_exact_receipt_comparison_rejects_every_changed_field(self):
        changed_fields = (
            ("explicit_path_sha256", {"tools/a.py": "0" * 64}),
            ("source_manifest_sha256", "0" * 64),
            ("source_file_count", 999),
            ("created_at_unix_ns", 987654321),
        )
        root = self.make_root()
        generation = root / ".transactions/n1/generation"
        self.write_file(generation, "tools/a.py", b"validated\n")
        receipt = self.make_receipt(generation)
        for field, value in changed_fields:
            with self.subTest(field=field):
                alternate = self.replace_receipt_fields(
                    receipt, **{field: value}
                )
                with self.assertRaises(transaction.TransactionError):
                    transaction._require_exact_receipt(alternate, receipt)

    def test_post_exchange_confirmation_error_leaves_committed_source(self):
        root = self.make_root()
        generation = root / ".transactions/n1/generation"
        self.write_file(generation, "tools/a.py", b"new\n")
        receipt = self.make_receipt(generation)
        real_read = transaction._read_generation_fd
        read_calls = []

        def fail_first_post_exchange_read(*args, **kwargs):
            read_calls.append(True)
            if len(read_calls) == 2:
                raise transaction.TransactionError(
                    "simulated confirmation failure"
                )
            return real_read(*args, **kwargs)

        with mock.patch.object(
            transaction,
            "_read_generation_fd",
            side_effect=fail_first_post_exchange_read,
        ):
            with self.assertRaisesRegex(
                transaction.TransactionError,
                "confirmation failure",
            ):
                transaction.promote_generation(
                    root, ".transactions/n1/generation", receipt
                )
        committed = self.read_committed(root)
        self.assertEqual(committed.created_at_unix_ns, receipt.created_at_unix_ns)
        self.assertEqual((root / "source/tools/a.py").read_bytes(), b"new\n")
        self.assertFalse(generation.exists())

    def test_post_commit_cleanup_failure_preserves_result_and_closes_fds(self):
        root = self.make_root()
        self.write_file(root / "source", "old.txt", b"old\n")
        generation = root / ".transactions/n1/generation"
        self.write_file(generation, "tools/a.py", b"new\n")
        receipt = self.make_receipt(generation)
        before = len(os.listdir("/proc/self/fd"))
        with mock.patch.object(
            transaction,
            "_remove_tree_at",
            side_effect=OSError(errno.EIO, "forced cleanup failure"),
        ):
            result = transaction.promote_generation(
                root, ".transactions/n1/generation", receipt
            )
        self.assertEqual(result.created_at_unix_ns, receipt.created_at_unix_ns)
        self.assertEqual(len(os.listdir("/proc/self/fd")), before)
        self.assertEqual((root / "source/tools/a.py").read_bytes(), b"new\n")
        self.assertEqual(generation.joinpath("old.txt").read_bytes(), b"old\n")

    def test_non_oserror_cleanup_failure_preserves_outcome_and_closes_fds(self):
        for with_primary_error in (False, True):
            with self.subTest(with_primary_error=with_primary_error):
                root = self.make_root(
                    "cleanup-runtime-{}".format(int(with_primary_error))
                )
                self.write_file(root / "source", "old.txt", b"old\n")
                generation = root / ".transactions/n1/generation"
                self.write_file(generation, "tools/a.py", b"new\n")
                receipt = self.make_receipt(generation)
                before = len(os.listdir("/proc/self/fd"))

                def inject(point):
                    if with_primary_error and point == "after_exchange":
                        raise ValueError("primary response failure")

                with mock.patch.object(
                    transaction,
                    "_remove_tree_at",
                    side_effect=RuntimeError("cleanup runtime failure"),
                ):
                    if with_primary_error:
                        with self.assertRaisesRegex(
                            ValueError, "primary response failure"
                        ):
                            transaction.promote_generation(
                                root,
                                ".transactions/n1/generation",
                                receipt,
                                fault_injector=inject,
                            )
                    else:
                        result = transaction.promote_generation(
                            root,
                            ".transactions/n1/generation",
                            receipt,
                        )
                        self.assertEqual(
                            result.created_at_unix_ns,
                            receipt.created_at_unix_ns,
                        )

                self.assertEqual(len(os.listdir("/proc/self/fd")), before)
                committed = self.read_committed(root)
                self.assertEqual(
                    committed.created_at_unix_ns,
                    receipt.created_at_unix_ns,
                )
                self.assertEqual(
                    generation.joinpath("old.txt").read_bytes(), b"old\n"
                )

    def test_confirmation_fstat_failure_closes_source_fd(self):
        root = self.make_committed_root()
        root_fd = transaction.open_directory_no_follow(root)
        source_info = os.stat(
            "source", dir_fd=root_fd, follow_symlinks=False
        )
        before = len(os.listdir("/proc/self/fd"))
        try:
            with mock.patch.object(
                transaction.os,
                "fstat",
                side_effect=OSError(errno.EIO, "forced fstat failure"),
            ):
                with self.assertRaisesRegex(OSError, "forced fstat failure"):
                    transaction._open_source_for_confirmation(
                        root_fd, source_info
                    )
            self.assertEqual(len(os.listdir("/proc/self/fd")), before)
        finally:
            os.close(root_fd)
        self.read_committed(root)

    def test_promotion_closes_all_open_file_descriptors(self):
        root = self.make_root()
        generation = root / ".transactions/n1/generation"
        self.write_file(generation, "tools/a.py", b"new\n")
        receipt = self.make_receipt(generation)
        before = len(os.listdir("/proc/self/fd"))
        transaction.promote_generation(
            root, ".transactions/n1/generation", receipt
        )
        after = len(os.listdir("/proc/self/fd"))
        self.assertEqual(after, before)

    def test_promote_rejects_source_directory_symlink(self):
        root = self.make_root()
        outside = self.test_root / "outside-source"
        outside.mkdir()
        (root / "source").symlink_to(outside, target_is_directory=True)
        generation = root / ".transactions/n1/generation"
        self.write_file(generation, "tools/a.py", b"new\n")
        receipt = self.make_receipt(generation)
        with self.assertRaises(transaction.TransactionError):
            transaction.promote_generation(
                root, ".transactions/n1/generation", receipt
            )
        self.assertFalse((outside / "tools/a.py").exists())
        self.assertFalse(generation.exists())

    def test_promote_rejects_generation_directory_symlink(self):
        root = self.make_root()
        outside = self.test_root / "outside-generation"
        self.write_file(outside, "tools/a.py", b"new\n")
        transaction_dir = root / ".transactions/n1"
        transaction_dir.mkdir(parents=True)
        (transaction_dir / "generation").symlink_to(
            outside, target_is_directory=True
        )
        receipt = self.make_receipt(outside)
        with self.assertRaises(transaction.TransactionError):
            transaction.promote_generation(
                root, ".transactions/n1/generation", receipt
            )
        self.assertFalse((root / "source").exists())
        self.assertEqual((outside / "tools/a.py").read_bytes(), b"new\n")

    def test_strict_receipt_accepts_exact_embedded_state(self):
        root = self.make_committed_root()
        receipt = self.read_committed(root)
        self.assertEqual(receipt.operation, "sync")
        self.assertEqual(receipt.nonce, "n1")
        self.assertEqual(receipt.source_head, "abc")
        self.assertEqual(receipt.explicit_paths, ("tools/a.py",))

    def test_explicit_paths_reject_noncanonical_spellings(self):
        variants = (
            "./tools/a.py",
            "tools//a.py",
            "tools/a.py/",
        )
        for path in variants:
            with self.subTest(path=path):
                with self.assertRaises(transaction.TransactionError):
                    transaction._validate_explicit_paths((path,))

    def test_strict_receipt_rejects_each_embedded_file_symlink(self):
        embedded_names = (
            "commit.json",
            "source-files.sha256",
            "explicit-paths.txt",
        )
        for name in embedded_names:
            with self.subTest(name=name):
                root = self.make_committed_root()
                path = root / "source/.tinyllmforge-scratch" / name
                self.replace_with_symlink(path)
                with self.assertRaises(transaction.TransactionError):
                    self.read_committed(root)
                root.rename(self.test_root / ("rejected-" + name))

    def test_strict_receipt_rejects_extra_embedded_directory_members(self):
        for kind in ("file", "appledouble", "symlink", "directory"):
            with self.subTest(kind=kind):
                root = self.make_committed_root()
                embedded = root / "source/.tinyllmforge-scratch"
                if kind == "file":
                    (embedded / "extra").write_bytes(b"extra\n")
                elif kind == "appledouble":
                    (embedded / "._metadata").write_bytes(b"metadata\n")
                elif kind == "symlink":
                    (embedded / "extra-link").symlink_to("commit.json")
                else:
                    (embedded / "extra-directory").mkdir()
                try:
                    with self.assertRaises(transaction.TransactionError):
                        self.read_committed(root)
                finally:
                    root.rename(
                        self.test_root / ("rejected-extra-" + kind)
                    )

    def test_strict_receipt_rejects_wrong_expected_identity(self):
        cases = (
            {"expected_nonce": "wrong"},
            {"expected_operation": "init"},
            {"expected_head": "wrong"},
            {"expected_paths": ["README.md"]},
        )
        for index, overrides in enumerate(cases):
            with self.subTest(overrides=overrides):
                root = self.make_committed_root()
                arguments = {
                    "expected_nonce": "n1",
                    "expected_operation": "sync",
                    "expected_head": "abc",
                    "expected_paths": ["tools/a.py"],
                }
                arguments.update(overrides)
                with self.assertRaises(transaction.TransactionError):
                    transaction.read_committed_generation(root, **arguments)
                root.rename(self.test_root / "wrong-{}".format(index))

    def test_strict_receipt_rejects_path_hash_mismatch(self):
        root = self.make_committed_root()
        (root / "source/tools/a.py").write_bytes(b"tampered\n")
        with self.assertRaises(transaction.TransactionError):
            self.read_committed(root)

    def test_strict_receipt_rejects_source_manifest_mismatch(self):
        root = self.make_committed_root()
        (root / "source/README.md").write_bytes(b"tampered manifest\n")
        with self.assertRaises(transaction.TransactionError):
            self.read_committed(root)

    def test_strict_receipt_rejects_symlink_target_mismatch(self):
        root = self.make_root()
        source = root / "source"
        source.mkdir()
        self.write_file(source, "tools/a.py", b"content\n")
        (source / "current.py").symlink_to("tools/a.py")
        receipt = self.make_receipt(source)
        source_fd = transaction.open_directory_no_follow(source)
        try:
            transaction.write_embedded_receipt(source_fd, receipt)
        finally:
            os.close(source_fd)
        (source / "current.py").unlink()
        (source / "current.py").symlink_to("README.md")
        with self.assertRaises(transaction.TransactionError):
            transaction.read_committed_generation(
                root,
                expected_nonce="n1",
                expected_operation="sync",
                expected_head="abc",
                expected_paths=["tools/a.py"],
            )

    def test_strict_receipt_rejects_symlink_replaced_by_crafted_file(self):
        root = self.make_root()
        source = root / "source"
        source.mkdir()
        self.write_file(source, "tools/a.py", b"content\n")
        link = source / "current.py"
        link.symlink_to("tools/a.py")
        receipt = self.make_receipt(source)
        source_fd = transaction.open_directory_no_follow(source)
        try:
            transaction.write_embedded_receipt(source_fd, receipt)
        finally:
            os.close(source_fd)
        link.unlink()
        link.write_bytes(b"symlink\0tools/a.py")
        with self.assertRaises(transaction.TransactionError):
            self.read_committed(root)

    def test_strict_receipt_rejects_embedded_manifest_mismatch(self):
        root = self.make_committed_root()
        manifest = (
            root / "source/.tinyllmforge-scratch/source-files.sha256"
        )
        manifest.write_bytes(manifest.read_bytes() + b"0" * 64 + b"  x\n")
        with self.assertRaises(transaction.TransactionError):
            self.read_committed(root)

    def test_strict_receipt_rejects_embedded_explicit_paths_mismatch(self):
        root = self.make_committed_root()
        paths = root / "source/.tinyllmforge-scratch/explicit-paths.txt"
        paths.write_text("README.md\n")
        with self.assertRaises(transaction.TransactionError):
            self.read_committed(root)

    def test_strict_receipt_rejects_noncanonical_commit_json(self):
        root = self.make_committed_root()
        path = root / "source/.tinyllmforge-scratch/commit.json"
        payload = json.loads(path.read_text())
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        with self.assertRaises(transaction.TransactionError):
            self.read_committed(root)

    def test_strict_receipt_rejects_wrong_embedded_identity(self):
        fields = (
            ("nonce", "wrong"),
            ("operation", "init"),
            ("source_head", "wrong"),
            ("explicit_paths", ["README.md"]),
        )
        for index, (field, value) in enumerate(fields):
            with self.subTest(field=field):
                root = self.make_committed_root()
                path = root / "source/.tinyllmforge-scratch/commit.json"
                payload = json.loads(path.read_text())
                payload[field] = value
                path.write_text(
                    json.dumps(
                        payload, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
                with self.assertRaises(transaction.TransactionError):
                    self.read_committed(root)
                root.rename(self.test_root / "embedded-{}".format(index))

    def test_strict_receipt_rejects_wrong_embedded_path_hash(self):
        root = self.make_committed_root()
        path = root / "source/.tinyllmforge-scratch/commit.json"
        payload = json.loads(path.read_text())
        payload["explicit_path_sha256"]["tools/a.py"] = "0" * 64
        path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )
        with self.assertRaises(transaction.TransactionError):
            self.read_committed(root)

    def test_strict_receipt_rejects_unknown_operation_without_expectation(self):
        root = self.make_committed_root()
        path = root / "source/.tinyllmforge-scratch/commit.json"
        payload = json.loads(path.read_text())
        payload["operation"] = "unknown"
        path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )
        with self.assertRaises(transaction.TransactionError):
            transaction.read_committed_generation(root)

    def test_strict_receipt_rejects_forbidden_member(self):
        root = self.make_committed_root()
        self.write_file(root / "source", "artifacts/result.txt", b"forbidden\n")
        with self.assertRaises(transaction.TransactionError):
            self.read_committed(root)

    def test_strict_receipt_rejects_source_directory_symlink(self):
        root = self.make_committed_root()
        source = root / "source"
        target = root / "source-target"
        source.replace(target)
        source.symlink_to(target.name, target_is_directory=True)
        with self.assertRaises(transaction.TransactionError):
            self.read_committed(root)

    def test_transaction_interrupted_carries_signal(self):
        error = transaction.TransactionInterrupted(signal.SIGTERM)
        self.assertEqual(error.signum, signal.SIGTERM)
        self.assertIn(str(int(signal.SIGTERM)), str(error))

    def test_signal_handler_only_raises_transaction_interrupted(self):
        with self.assertRaises(transaction.TransactionInterrupted) as raised:
            transaction._raise_transaction_interrupted(signal.SIGTERM, None)
        self.assertEqual(raised.exception.signum, signal.SIGTERM)

    def test_signal_handler_partial_installation_restores_installed_only(self):
        previous = {
            signal.SIGHUP: object(),
            signal.SIGINT: object(),
            signal.SIGTERM: object(),
        }
        install_attempts = []
        restored = []
        mask_calls = []

        def install_or_restore(signum, handler):
            if handler is transaction._raise_transaction_interrupted:
                install_attempts.append(signum)
                if len(install_attempts) == 2:
                    raise OSError(errno.EINTR, "forced installation failure")
                return
            restored.append((signum, handler))

        with mock.patch.object(
            transaction.signal,
            "getsignal",
            side_effect=lambda signum: previous[signum],
        ), mock.patch.object(
            transaction.signal,
            "pthread_sigmask",
            side_effect=lambda how, mask: (
                mask_calls.append((how, set(mask))) or {"caller-mask"}
            ),
        ), mock.patch.object(
            transaction.signal,
            "signal",
            side_effect=install_or_restore,
        ):
            with self.assertRaises(OSError):
                with transaction._transaction_signal_handlers():
                    self.fail("handler context should not be entered")

        self.assertEqual(
            restored,
            [(signal.SIGHUP, previous[signal.SIGHUP])],
        )
        self.assertEqual(
            mask_calls,
            [
                (
                    signal.SIG_BLOCK,
                    {signal.SIGHUP, signal.SIGINT, signal.SIGTERM},
                ),
                (signal.SIG_SETMASK, {"caller-mask"}),
            ],
        )

    def test_signal_handler_immediate_delivery_restores_installed_handler(self):
        previous = {
            signal.SIGHUP: object(),
            signal.SIGINT: object(),
            signal.SIGTERM: object(),
        }
        current = dict(previous)
        restored = []
        mask_calls = []

        def install_then_deliver(signum, handler):
            current[signum] = handler
            if handler is transaction._raise_transaction_interrupted:
                handler(signum, None)
            restored.append((signum, handler))

        with mock.patch.object(
            transaction.signal,
            "getsignal",
            side_effect=lambda signum: current[signum],
        ), mock.patch.object(
            transaction.signal,
            "pthread_sigmask",
            side_effect=lambda how, mask: (
                mask_calls.append((how, set(mask))) or {"caller-mask"}
            ),
        ), mock.patch.object(
            transaction.signal,
            "signal",
            side_effect=install_then_deliver,
        ):
            with self.assertRaises(transaction.TransactionInterrupted):
                with transaction._transaction_signal_handlers():
                    self.fail("handler context should not be entered")

        self.assertEqual(restored, [(signal.SIGHUP, previous[signal.SIGHUP])])
        self.assertEqual(current[signal.SIGHUP], previous[signal.SIGHUP])
        self.assertEqual(
            mask_calls[-1], (signal.SIG_SETMASK, {"caller-mask"})
        )

    def test_signal_handler_blocks_delivery_between_restoration_steps(self):
        managed = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
        deliveries = []

        def make_previous(signum):
            def previous_handler(delivered_signum, frame):
                del frame
                deliveries.append((signum, delivered_signum))

            return previous_handler

        previous = {signum: make_previous(signum) for signum in managed}
        current = dict(previous)
        blocked = set()
        pending = []
        restoring = []

        def change_mask(how, mask):
            old_mask = set(blocked)
            if how == signal.SIG_BLOCK:
                blocked.update(mask)
            elif how == signal.SIG_SETMASK:
                blocked.clear()
                blocked.update(mask)
                queued = list(pending)
                pending[:] = []
                for signum in queued:
                    current[signum](signum, None)
            return old_mask

        def install_or_restore(signum, handler):
            current[signum] = handler
            if handler is previous[signum]:
                restoring.append(signum)
                if len(restoring) == 1:
                    delivered = signal.SIGHUP
                    if delivered in blocked:
                        pending.append(delivered)
                    else:
                        current[delivered](delivered, None)

        with mock.patch.object(
            transaction.signal,
            "getsignal",
            side_effect=lambda signum: current[signum],
        ), mock.patch.object(
            transaction.signal,
            "pthread_sigmask",
            side_effect=change_mask,
        ), mock.patch.object(
            transaction.signal,
            "signal",
            side_effect=install_or_restore,
        ):
            with transaction._transaction_signal_handlers():
                pass

        self.assertEqual(restoring, list(reversed(managed)))
        self.assertEqual(
            deliveries, [(signal.SIGHUP, signal.SIGHUP)]
        )
        self.assertEqual(blocked, set())
        self.assertEqual(current, previous)

    def test_signal_handler_restores_mask_after_transaction_interrupted(self):
        managed = {signal.SIGHUP, signal.SIGINT, signal.SIGTERM}
        caller_mask = {signal.SIGUSR1}
        mask_calls = []

        def change_mask(how, mask):
            mask_calls.append((how, set(mask)))
            return set(caller_mask)

        with mock.patch.object(
            transaction.signal,
            "pthread_sigmask",
            side_effect=change_mask,
        ):
            with self.assertRaises(transaction.TransactionInterrupted):
                with transaction._transaction_signal_handlers():
                    transaction._raise_transaction_interrupted(
                        signal.SIGTERM, None
                    )

        self.assertEqual(
            mask_calls,
            [
                (signal.SIG_BLOCK, managed),
                (signal.SIG_SETMASK, caller_mask),
                (signal.SIG_BLOCK, managed),
                (signal.SIG_SETMASK, caller_mask),
            ],
        )

    def test_signal_restoration_failure_preserves_primary_exception(self):
        managed = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
        previous = {signum: object() for signum in managed}

        for phase in ("pre_exchange", "after_exchange"):
            with self.subTest(phase=phase):
                primary = RuntimeError("{} primary".format(phase))
                restoration_attempts = []
                mask_calls = []

                def install_or_restore(signum, handler):
                    if handler is transaction._raise_transaction_interrupted:
                        return
                    restoration_attempts.append(signum)
                    if len(restoration_attempts) == 1:
                        raise RuntimeError("forced restoration failure")

                def change_mask(how, mask):
                    mask_calls.append((how, set(mask)))
                    return {"caller-mask"}

                with mock.patch.object(
                    transaction.signal,
                    "getsignal",
                    side_effect=lambda signum: previous[signum],
                ), mock.patch.object(
                    transaction.signal,
                    "pthread_sigmask",
                    side_effect=change_mask,
                ), mock.patch.object(
                    transaction.signal,
                    "signal",
                    side_effect=install_or_restore,
                ):
                    with self.assertRaises(RuntimeError) as raised:
                        with transaction._transaction_signal_handlers():
                            raise primary

                self.assertIs(raised.exception, primary)
                self.assertEqual(
                    restoration_attempts, list(reversed(managed))
                )
                self.assertEqual(
                    mask_calls[-1],
                    (signal.SIG_SETMASK, {"caller-mask"}),
                )


if __name__ == "__main__":
    unittest.main()
