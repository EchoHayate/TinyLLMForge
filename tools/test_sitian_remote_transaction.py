import ctypes
import errno
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tools import sitian_remote_transaction as transaction


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
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
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

    def write_generation(self, path, marker):
        path.mkdir(parents=True)
        self.write_file(path, "marker.txt", marker.encode("utf-8"))

    def read_marker(self, path):
        return (path / "marker.txt").read_text()

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

    def test_exchange_implementation_contains_no_rename_fallback(self):
        source = inspect.getsource(transaction.rename_exchange)
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


if __name__ == "__main__":
    unittest.main()
