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

    def test_first_source_postcheck_rolls_back_post_identity_swap(self):
        root = self.make_root()
        nonce_root = root / ".transactions/n1"
        generation = nonce_root / "generation"
        self.write_file(generation, "tools/a.py", b"validated\n")
        receipt = self.make_receipt(generation)
        real_rename = os.rename
        raced = []

        def race_after_identity_check(src, dst, *args, **kwargs):
            if src == "generation" and dst == "source" and not raced:
                raced.append(True)
                real_rename(
                    "generation",
                    "validated-generation",
                    src_dir_fd=kwargs["src_dir_fd"],
                    dst_dir_fd=kwargs["src_dir_fd"],
                )
                replacement = nonce_root / "generation"
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
            return real_rename(src, dst, *args, **kwargs)

        with mock.patch.object(
            transaction.os, "rename", side_effect=race_after_identity_check
        ):
            with self.assertRaises(transaction.TransactionError):
                transaction.promote_generation(
                    root, ".transactions/n1/generation", receipt
                )

        self.assertFalse((root / "source").exists())
        self.assertFalse(generation.exists())
        self.assertEqual(
            (
                nonce_root / "validated-generation/tools/a.py"
            ).read_bytes(),
            b"validated\n",
        )

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

    def test_existing_source_postcheck_exchanges_back_post_identity_swap(self):
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
                generation.rename(nonce_root / "validated-generation")
                self.write_file(generation, "tools/a.py", b"unexpected\n")
                alternate = self.replace_receipt_fields(
                    self.make_receipt(generation),
                    created_at_unix_ns=receipt.created_at_unix_ns + 1,
                )
                replacement_fd = transaction.open_directory_no_follow(
                    generation
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
            with self.assertRaises(transaction.TransactionError):
                transaction.promote_generation(
                    root, ".transactions/n1/generation", receipt
                )

        self.assertEqual((root / "source/old.txt").read_bytes(), b"old\n")
        self.assertFalse((root / "source/tools/a.py").exists())
        self.assertFalse(generation.exists())
        self.assertEqual(
            (
                nonce_root / "validated-generation/tools/a.py"
            ).read_bytes(),
            b"validated\n",
        )

    def test_postcheck_requires_exact_supplied_receipt_and_rolls_back(self):
        changed_fields = (
            ("explicit_path_sha256", {"tools/a.py": "0" * 64}),
            ("source_manifest_sha256", "0" * 64),
            ("source_file_count", 999),
            ("created_at_unix_ns", 987654321),
        )
        for index, (field, value) in enumerate(changed_fields):
            with self.subTest(field=field):
                root = self.make_root("receipt-{}".format(index))
                generation = root / ".transactions/n1/generation"
                self.write_file(generation, "tools/a.py", b"validated\n")
                receipt = self.make_receipt(generation)
                alternate = self.replace_receipt_fields(
                    receipt, **{field: value}
                )
                real_read = transaction._read_generation_fd
                read_calls = []

                def return_alternate_after_promotion(*args, **kwargs):
                    read_calls.append(True)
                    if len(read_calls) == 2:
                        return alternate
                    return real_read(*args, **kwargs)

                with mock.patch.object(
                    transaction,
                    "_read_generation_fd",
                    side_effect=return_alternate_after_promotion,
                ):
                    with self.assertRaises(transaction.TransactionError):
                        transaction.promote_generation(
                            root,
                            ".transactions/n1/generation",
                            receipt,
                        )
                self.assertFalse((root / "source").exists())
                self.assertFalse(generation.exists())

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


if __name__ == "__main__":
    unittest.main()
