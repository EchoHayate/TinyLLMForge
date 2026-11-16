import errno
import hashlib
import importlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from validation_contract import encoded


class TokenizerSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name("tokenizer_snapshot.py").is_file())
        self.module = importlib.import_module("tokenizer_snapshot")
        self.contract = importlib.import_module("tokenizer_runtime_profile")
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name).resolve()
        self.source = self.directory / "source"
        self.source.mkdir()
        self.payloads = {name: name.encode("ascii") for name in self.contract.FILE_PINS}
        self.pins = {name: hashlib.sha256(data).hexdigest()
                     for name, data in self.payloads.items()}
        for name, data in self.payloads.items():
            (self.source / name).write_bytes(data)
        (self.source / "unread-weight.safetensors").mkdir()
        self.patching = patch.object(self.contract, "FILE_PINS", self.pins)
        self.patching.start()
        self.profile = {
            "kind": "TOKENIZER_RUNTIME_PROFILE", "version": 1, "run_id": "a" * 32,
            "revision": self.contract.REVISION, "file_sha256": self.pins,
            "versions": {
                "python": "3.12.13", "transformers": "4.57.1",
                "tokenizers": "fixture", "huggingface-hub": "fixture", "jinja2": "fixture",
            },
        }
        self.raw = encoded(self.profile)
        self.approved = hashlib.sha256(self.raw).hexdigest()
        self.root = self.directory / self.profile["run_id"]

    def tearDown(self):
        self.patching.stop()
        for path in self.directory.rglob("*"):
            if not path.is_symlink() and path.is_dir():
                path.chmod(0o700)
        self.temporary.cleanup()

    def create(self):
        return self.module.create_snapshot(
            self.source, self.directory, self.raw, self.approved)

    def test_exact_copy_permissions_and_verification(self):
        result = self.create()
        snapshot = self.root / "snapshot"
        self.assertEqual(set(path.name for path in snapshot.iterdir()), set(self.pins))
        self.assertEqual(self.root.stat().st_mode & 0o777, 0o700)
        self.assertEqual(snapshot.stat().st_mode & 0o777, 0o500)
        for name, data in self.payloads.items():
            self.assertEqual((snapshot / name).read_bytes(), data)
            self.assertEqual((snapshot / name).stat().st_mode & 0o777, 0o400)
        self.assertEqual(result, self.module.verify_snapshot(
            self.root, self.raw, self.approved))
        self.assertEqual(result["kind"], "TOKENIZER_FILE_SNAPSHOT_ONLY")
        self.assertFalse(result["tokenizer_loaded"])
        self.assertEqual(result["file_sha256"], self.pins)

    def test_reuse_rejected_without_overwrite(self):
        before = self.create()
        with self.assertRaises(FileExistsError):
            self.create()
        self.assertEqual(before, self.module.verify_snapshot(
            self.root, self.raw, self.approved))

    def test_source_symlink_and_hardlink_rejected(self):
        name = "config.json"
        original = self.source / name
        donor = self.directory / "donor"
        donor.write_bytes(self.payloads[name])
        original.unlink()
        original.symlink_to(donor)
        with self.assertRaises((OSError, ValueError)):
            self.create()
        self.assertFalse(self.root.exists())
        original.unlink()
        os.link(donor, original)
        with self.assertRaises(ValueError):
            self.create()
        self.assertFalse(self.root.exists())

    def test_bad_source_bytes_and_directory_alias_rejected(self):
        alias = self.directory / "alias"
        alias.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.module.create_snapshot(alias, self.directory, self.raw, self.approved)
        (self.source / "config.json").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.create()
        self.assertFalse(self.root.exists())

    def test_snapshot_tampering_and_extra_file_rejected(self):
        self.create()
        snapshot = self.root / "snapshot"
        target = snapshot / "config.json"
        target.chmod(0o600)
        target.write_bytes(b"changed")
        target.chmod(0o400)
        with self.assertRaises(ValueError):
            self.module.verify_snapshot(self.root, self.raw, self.approved)
        target.chmod(0o600)
        target.write_bytes(self.payloads["config.json"])
        target.chmod(0o400)
        snapshot.chmod(0o700)
        (snapshot / "extra.py").write_bytes(b"not executed")
        snapshot.chmod(0o500)
        with self.assertRaises(ValueError):
            self.module.verify_snapshot(self.root, self.raw, self.approved)

    def test_writable_snapshot_and_wrong_root_rejected(self):
        self.create()
        snapshot = self.root / "snapshot"
        snapshot.chmod(0o700)
        with self.assertRaises(ValueError):
            self.module.verify_snapshot(self.root, self.raw, self.approved)
        snapshot.chmod(0o500)
        wrong = self.directory / ("b" * 32)
        self.root.rename(wrong)
        with self.assertRaises(ValueError):
            self.module.verify_snapshot(wrong, self.raw, self.approved)

    def test_special_files_and_byte_ceiling_rejected(self):
        target = self.source / "config.json"
        target.unlink()
        os.mkfifo(target)
        with self.assertRaises(ValueError):
            self.create()
        target.unlink()
        target.mkdir()
        opened = []
        original_open = os.open

        def record_open(path, flags, *arguments, **keywords):
            descriptor = original_open(path, flags, *arguments, **keywords)
            if path == "config.json":
                opened.append(descriptor)
            return descriptor

        try:
            with patch.object(self.module.os, "open", side_effect=record_open), \
                    self.assertRaises((OSError, ValueError)):
                self.create()
            self.assertEqual(len(opened), 1)
            with self.assertRaises(OSError) as caught:
                os.fstat(opened[0])
            self.assertEqual(caught.exception.errno, errno.EBADF)
        finally:
            for descriptor in opened:
                try:
                    os.close(descriptor)
                except OSError as error:
                    if error.errno != errno.EBADF:
                        raise
        target.rmdir()
        target.write_bytes(self.payloads["config.json"])
        with patch.object(self.module, "MAX_SNAPSHOT_BYTES", 1), \
                self.assertRaisesRegex(ValueError, "snapshot byte ceiling"):
            self.create()
        self.assertFalse(self.root.exists())

    def test_staging_inside_source_rejected(self):
        for parent in (self.source, self.source / "child"):
            with self.subTest(parent=parent), self.assertRaises(ValueError):
                self.module.create_snapshot(
                    self.source, parent, self.raw, self.approved)
        self.assertFalse((self.source / self.profile["run_id"]).exists())
