# LoRA Tokenizer Runtime Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement exact dependency-profile checks and a five-file read-only tokenizer snapshot, without loading a tokenizer or producing a real audit receipt.

**Architecture:** One module validates an explicitly approved raw-byte profile and compares it with package metadata. A second module creates and verifies an exclusive run-bound snapshot. A subsequent loader must consume these interfaces, load its own tokenizer instance, bind observed class/mode/state and enforce session lifetime; neither module here can establish that identity.

**Tech Stack:** Python 3.12 standard library, POSIX descriptor-relative file operations, colocated unittest with temporary synthetic files and patched metadata.

## Global Constraints

- Binding spec: `docs/superpowers/specs/2026-11-12-lora-tokenizer-admission-design.md`, especially sections 3, 5 and 7.
- Preserve the current checkout and branch `experiments/feedback-active-verification`; entry HEAD is `93b4e8e0dc5174db0426a5da9e29cc360ed331a7`.
- No branch/worktree creation, commit, push, installs, network, telemetry, SSH, tokenizer/model loading, GPU use or task-source execution.
- Use apply_patch; no inline code comments or one-letter identifiers.
- All completed source/tests, frozen data, READMEs, old plans/specs/ledgers and sealed evidence remain unchanged. The preceding 69-test receipt is `/tmp/tlf-tokenizer-components-20261112-HhFKy7/final-verification.json`; do not rerun its exclusive helpers.
- Do not import or execute the old selection preflight. It imports torch and generation modules and hashes weight files; it is not the tokenizer-only route.
- Only `config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json` and `merges.txt` may enter the new snapshot. The source model directory can contain other files, but do not open, copy or hash them.
- Python is pinned to 3.12.13 and transformers to 4.57.1 for a future real run. Unknown tokenizers, huggingface-hub, jinja2 and other installed dependency versions are not guessed or taken from the different local installation.
- An authorized metadata-only discovery must precede approval of a real profile. The profile freezes the complete installed-distribution inventory, deliberately stricter than guessing an import closure. Package metadata is not code authentication.
- `benchmark_eligible`, `formal_evaluation_ready`, `net_benefit_established`, `isolated` and `all_possible_second_rounds_validated` stay false.
- These components do not transfer private prompts/answers, change 324 audit rows, implement the audit loop, enforce OS resource budgets or establish cleanup.
- Standard-library unittest/TDD replaces incompatible plugin bootstrap, telemetry and test-only ownership defaults. Do not install or invoke those plugin workflows.

## Scope Decision And Evidence

The approved parent design already selected the existing remote CPU environment
subject to fresh admission. This plan does not reopen that decision or claim a
new launch authorization.

Three possible approaches were considered:

1. Reuse selection preflight: rejected because it imports model/generation
   backends and checks weight files, contrary to this stage.
2. Assume dependency versions and use a local tokenizer installation: rejected
   because the historical receipt does not record tokenizers/hub/jinja2, and the
   inspected local transformers version differs from the intended runtime.
3. Separate metadata/profile and exact file snapshot from loading: selected.
   It is independently testable with synthetic files, and cannot accidentally
   turn an injected tokenizer or a dictionary into a trusted runtime receipt.

This is the preparation portion of runtime identity, not its complete delivery.
The loader/session remains a separately scoped implementation, with explicit
interfaces and required acceptance below. Missing real metadata prevents actual
profile admission, but does not prevent implementation of these local checks.

## File Map

| New file | Responsibility |
| --- | --- |
| `experiments/lora_necessity/tokenizer_runtime_profile.py` | Raw-byte profile approval, immutable target pins, metadata inventory comparison |
| `experiments/lora_necessity/test_tokenizer_runtime_profile.py` | Profile corruption, incomplete versions and metadata-only behavior |
| `experiments/lora_necessity/tokenizer_snapshot.py` | Exclusive five-file snapshot and subsequent integrity checks |
| `experiments/lora_necessity/test_tokenizer_snapshot.py` | Synthetic POSIX files, reuse, links, tampering and permission rejection |

Main owns only this new plan, the new
`.superpowers/sdd/lora-tokenizer-runtime-20261112-progress.md`, and a fresh evidence
directory. Do not change the previous tokenizer ledger: it is part of a sealed
receipt.

## Entry Verification

- [x] Confirm branch/HEAD and clean tracked/index diffs; preserve all untracked work.
- [x] Independently verify the preceding receipt's source/document/review/test
  hashes and its baseline manifest. Snapshot all current lora_necessity files
  plus the previously sealed documents into a new evidence manifest.
- [x] Assert the four new implementation/test paths are absent.
- [x] Run and retain the current baseline:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 \
  -m unittest discover -s experiments/lora_necessity -p 'test_*.py' -v
```

Expected at this plan's entry: 69 passing tests. If the count or source hashes
changed, investigate rather than silently rebasing the acceptance evidence.

## Task 1: Approved Profile And Metadata Check

**Files:** create `tokenizer_runtime_profile.py` and `test_tokenizer_runtime_profile.py`.

**Consumes:** existing `parse_json(raw, limit=1048576)`, `fields(value, names)`,
`require(condition, message)`, and `digest(value)`.

**Produces:**

- `validate_profile(raw: bytes, approved_sha256: str) -> dict`.
- `runtime_inventory() -> dict[str, str]`, using metadata only.
- `inspect_runtime(raw: bytes, approved_sha256: str) -> dict`.

The expected raw-byte digest must come from the trusted controlling launch scope,
not be recomputed from an untrusted incoming profile and treated as approval.
This API verifies a binding; it does not authenticate the approving person.
The approved profile is run-specific and contains the complete distribution
inventory. Its raw JSON has no actual-class or loaded-instance claim.

- [x] Write these tests first.

```python
import copy
import hashlib
import importlib
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
```

- [x] Run focused RED and retain the actual failure, without claiming that a
  missing-module failure exercises every behavioral assertion:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 \
  -m unittest discover -s experiments/lora_necessity \
  -p 'test_tokenizer_runtime_profile.py' -v
```

- [x] Implement the source below.

```python
import copy
import hashlib
from importlib import metadata
import platform
import re

from pilot_protocol import FLAGS, fields, require
from validation_contract import parse_json


REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
FILE_PINS = {
    "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
}
REQUIRED_PACKAGES = {"python", "transformers", "tokenizers", "huggingface-hub", "jinja2"}


def validate_profile(raw, approved_sha256):
    require(type(raw) is bytes and len(raw) <= 1048576, "profile bytes")
    require(type(approved_sha256) is str and
            re.fullmatch("[0-9a-f]{64}", approved_sha256) is not None, "approval digest")
    require(hashlib.sha256(raw).hexdigest() == approved_sha256, "profile approval mismatch")
    profile = parse_json(raw)
    fields(profile, ("kind", "version", "run_id", "revision", "file_sha256", "versions"))
    require(profile["kind"] == "TOKENIZER_RUNTIME_PROFILE", "profile kind")
    require(type(profile["version"]) is int and profile["version"] == 1, "profile version")
    require(type(profile["run_id"]) is str and
            re.fullmatch("[0-9a-f]{32}", profile["run_id"]) is not None, "run ID")
    require(profile["revision"] == REVISION, "revision declaration")
    require(profile["file_sha256"] == FILE_PINS, "file pins")
    versions = profile["versions"]
    require(type(versions) is dict and REQUIRED_PACKAGES <= set(versions), "version inventory")
    for name, version in versions.items():
        require(type(name) is str and re.fullmatch("[a-z0-9]+(?:-[a-z0-9]+)*", name)
                is not None, "canonical package name")
        require(type(version) is str and bool(version) and
                not any(character.isspace() for character in version), "exact version")
    require(versions["python"] == "3.12.13" and
            versions["transformers"] == "4.57.1", "pinned runtime versions")
    return profile


def runtime_inventory():
    versions = {"python": platform.python_version()}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        require(type(name) is str and bool(name), "distribution name")
        canonical = re.sub("[-_.]+", "-", name).lower()
        require(canonical not in versions, "duplicate installed distribution")
        versions[canonical] = distribution.version
    return versions


def inspect_runtime(raw, approved_sha256):
    profile = validate_profile(raw, approved_sha256)
    observed = runtime_inventory()
    require(observed == profile["versions"], "installed inventory mismatch")
    return {
        "kind": "TOKENIZER_RUNTIME_METADATA_ONLY", "run_id": profile["run_id"],
        "profile_sha256": approved_sha256, "versions": copy.deepcopy(observed),
        "tokenizer_loaded": False, "all_possible_second_rounds_validated": False,
        **{flag: False for flag in FLAGS},
    }
```

- [x] Run focused GREEN and the full suite, preserving commands, logs, exit codes
  and current source hashes. Expected counts from these snippets: five focused,
  74 total. These are planned counts, not observed results.
- [x] Independently review both complete additions, not ordinary empty Git diff.
  Verify no import-time metadata enumeration or third-party import occurs.
  Record local task acceptance only; no actual runtime profile has been approved.

## Task 2: Exclusive Five-File Snapshot

**Files:** create `tokenizer_snapshot.py` and `test_tokenizer_snapshot.py`.

**Consumes:** `validate_profile(raw, approved_sha256) -> dict` from Task 1;
`fields`, `require` and `digest` from existing pure helpers.

**Produces:**

- `create_snapshot(source, parent, raw, approved_sha256) -> dict`.
- `verify_snapshot(root, raw, approved_sha256) -> dict`.

Paths are absolute, canonical, existing directories; symlink ancestors are
rejected rather than silently resolved. The caller must pass the resolved
`TemporaryDirectory` path in local tests (macOS `/tmp` may be a symlink).
`parent` is the pre-admitted caller-owned staging parent, not a tokenizer model
directory. The new root is exactly `parent / run_id` and is never reused.

The source directory may contain weights, but only the five profile names are
opened. Reads use directory descriptors, `O_NOFOLLOW`, `O_NONBLOCK`, regular-file
checks, owner/link checks and an explicit byte ceiling. A file's exact pinned
bytes, not mtime or directory name, determine acceptance.

The internal 128 MiB snapshot ceiling is an upper bound for these local file
operations, not proof of the parent spec's 128 MiB aggregate run-root limit.
Inputs, logs and results must also fit that aggregate limit in the future wrapper.
Failure preserves the exclusive partial root and does not authorize retry,
cleanup or widening a budget.

- [x] Write these tests first.

```python
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
        with self.assertRaises((OSError, ValueError)):
            self.create()
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
```

- [x] Run focused RED with the exact command below; retain the missing-module
  failure accurately.

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.12/bin/python3.12 \
  -m unittest discover -s experiments/lora_necessity -p 'test_tokenizer_snapshot.py' -v
```

- [x] Implement the source below.

```python
from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import stat

from pilot_protocol import FLAGS, require
from tokenizer_runtime_profile import validate_profile


MAX_SNAPSHOT_BYTES = 128 * 1024 * 1024


def _directory(path):
    path = Path(path)
    require(path.is_absolute() and path == path.resolve(strict=True), "canonical directory")
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid(),
                "owned directory required")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read(directory, name, remaining, readonly=False):
    descriptor = os.open(
        name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and
                info.st_uid == os.getuid(), "owned regular single-link file")
        if readonly:
            require(stat.S_IMODE(info.st_mode) == 0o400, "read-only snapshot file")
        require(info.st_size <= remaining, "snapshot byte ceiling")
        data = stream.read(remaining + 1)
    require(len(data) <= remaining, "snapshot byte ceiling")
    return data


def _contents(directory, profile, readonly=False):
    if readonly:
        require(set(os.listdir(directory)) == set(profile["file_sha256"]), "snapshot file set")
        require(stat.S_IMODE(os.fstat(directory).st_mode) == 0o500, "read-only snapshot directory")
    contents = {}
    remaining = MAX_SNAPSHOT_BYTES
    for name, expected in sorted(profile["file_sha256"].items()):
        data = _read(directory, name, remaining, readonly)
        require(hashlib.sha256(data).hexdigest() == expected, "snapshot file digest")
        contents[name] = data
        remaining -= len(data)
    return contents


def verify_snapshot(root, raw, approved_sha256):
    profile = validate_profile(raw, approved_sha256)
    root = Path(root)
    require(root.name == profile["run_id"], "snapshot run binding")
    with ExitStack() as stack:
        root_fd = _directory(root)
        stack.callback(os.close, root_fd)
        require(stat.S_IMODE(os.fstat(root_fd).st_mode) == 0o700, "private run root")
        snapshot_fd = os.open(
            "snapshot", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        stack.callback(os.close, snapshot_fd)
        require(os.fstat(snapshot_fd).st_uid == os.getuid(), "owned snapshot")
        contents = _contents(snapshot_fd, profile, readonly=True)
    return {
        "kind": "TOKENIZER_FILE_SNAPSHOT_ONLY", "run_id": profile["run_id"],
        "profile_sha256": approved_sha256, "revision_declaration": profile["revision"],
        "file_sha256": dict(profile["file_sha256"]),
        "file_bytes": {name: len(data) for name, data in contents.items()},
        "tokenizer_loaded": False, "all_possible_second_rounds_validated": False,
        **{flag: False for flag in FLAGS},
    }


def create_snapshot(source, parent, raw, approved_sha256):
    profile = validate_profile(raw, approved_sha256)
    source = Path(source)
    parent = Path(parent)
    require(parent != source and source not in parent.parents, "staging outside source")
    with ExitStack() as stack:
        source_fd = _directory(source)
        stack.callback(os.close, source_fd)
        contents = _contents(source_fd, profile)
        parent_fd = _directory(parent)
        stack.callback(os.close, parent_fd)
        os.mkdir(profile["run_id"], mode=0o700, dir_fd=parent_fd)
        root_fd = os.open(profile["run_id"], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                          dir_fd=parent_fd)
        stack.callback(os.close, root_fd)
        os.fchmod(root_fd, 0o700)
        os.mkdir("snapshot", mode=0o700, dir_fd=root_fd)
        snapshot_fd = os.open(
            "snapshot", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        stack.callback(os.close, snapshot_fd)
        for name, data in contents.items():
            descriptor = os.open(
                name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600, dir_fd=snapshot_fd)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
                os.fchmod(stream.fileno(), 0o400)
        os.fchmod(snapshot_fd, 0o500)
    return verify_snapshot(Path(parent) / profile["run_id"], raw, approved_sha256)
```

- [x] Run focused GREEN, then the full local suite. Expected planned counts:
  eight focused and 82 total. Preserve exact source/log hashes before review.
- [x] Add focused RED/GREEN tests if review finds a concrete missing filesystem
  condition; do not silently widen ownership or replace descriptor checks with
  string-prefix path checks.
- [x] Independently review the complete snapshot additions and their Task 1
  interface. A valid snapshot is not a loaded tokenizer, a tamper-proof sandbox,
  a version-compatible model runtime or an authenticated upstream download.

## Combined Acceptance

- [x] Main freshly runs the complete suite and checks old source/data/document
  hashes; original 69-test snapshot remains sealed.
- [x] Verify code import purity: only stdlib plus the named local pure modules;
  no metadata enumeration, filesystem mutation, tokenizer import or model import
  at module import. Runtime metadata enumeration occurs only through explicit
  `runtime_inventory` / `inspect_runtime` calls.
- [x] Check all planned false flags and metadata/snapshot-only result labels.
- [x] Independently review all four additions and their exact full-file diff.
  Retain original failed reviews and correction RED/GREEN logs if any.
- [x] Main records hashes, tests, reviewer verdicts and remaining gates in the
  new runtime ledger and an exclusive closeout receipt. Do not change the old
  tokenizer-admission ledger or receipt.
- [x] Keep everything local/unpublished; do not propose merging or silently run
  dependency discovery on the remote server.

## Coverage And Remaining Interfaces

| Parent-spec requirement | Delivery boundary |
| --- | --- |
| Fixed five file hashes and target revision declaration | Task 1 |
| Missing/changed dependency versions reject before loading | Task 1 predicate and inventory; real metadata discovery/admission still absent |
| Copy only five regular files, reject links/escape/extra snapshot files, verify before and after copy | Task 2 |
| Exclusive run ID, owner/private root and read-only snapshot | Task 2; trusted parent and process lifecycle belong to wrapper |
| Reverify files after use | Task 2 supplies `verify_snapshot`; future session must call it at successful finalization |
| Actual loaded tokenizer class/fast mode/special tokens/template/vocabulary | Separate loader/session plan consuming both tasks |
| Disable backends/telemetry before import; reject torch/CUDA/generation imports | Loader/session implementation and startup probes; no import performed here |
| Exact dependency inventory recheck across session | Loader/session must call metadata checks before loading and before successful finalization |
| Only owned loaded instance produces identity-bound measurements | Loader/session must construct internally, bind `measure` results to actual observed state and invalidate after failure/close |
| Real 324-row counts, stop rules and private/public reports | Separate audit-controller plan, consuming sealed inventory/measurement and session |
| One connection/run, deadlines, aggregate memory/process/output/root limits and cleanup | Separate launch-wrapper plan and concrete scope |
| Private transfer, actual load, model generation/training and scientific benefit | Not authorized or established by this preparation |

The future loader must not accept a caller's tokenizer or identity dictionary as
proof of actual loading. It must use only the admitted snapshot with fast mode,
offline/local-files-only and remote-code disabled, and record actual class,
special-token strings/IDs and observed measurement state. Measurements remain
unattested until the complete session/controller/cleanup chain is verified.
Neither result kind above may be relabeled to `LORA_REAL_TOKENIZER_AUDIT`.

Snapshots are ordinary owner-controlled POSIX files, not immutable against their
owner. Parent directories and the trusted audit process must not be concurrently
modified by untrusted code. Candidate code is not executed in this stage. The
wrapper must enforce stronger isolation where needed; chmod and hashes are not a
substitute. This limitation is explicit rather than an untested safety claim.

## Plan Self-Review And Handoff

Before accepting this plan, check all Python blocks parse, exact pins match the
sealed parent spec, task signatures match consumers, and no actual dependency
versions were invented. The synthetic version strings and patched file hashes
occur only in tests; they are never an admissible real runtime profile.

The two tasks share only Task 1's validated-profile API and must execute
sequentially with independent reviews. Do not resume old agents, create another
worktree, or ask the user to repeat the already established execution-mode choice.
This document is an implementation plan, not implementation or execution evidence.
Final scope label: `LOCAL_RUNTIME_PREPARATION_ONLY_NOT_LOADED`.

## Execution Review Corrections

Task 1 review found that the prescribed duplicate-key sample also has an invalid
schema, so it cannot independently demonstrate duplicate rejection. Preserve
that malformed-input example and add an otherwise valid complete-profile
duplicate-key regression, with last-key-wins mutation failure evidence and
normal-parser success. This strengthens the existing rejection requirement
without changing the production contract or deleting planned demonstrations.
The corrected expected counts are six profile tests, 75 after Task 1 and
83 after Task 2; actual results must be recorded separately.

Task 2 self-review found descriptor leakage when `fdopen` rejects a directory.
The implementation now closes the raw descriptor on failed stream construction;
the original special-file test also checks `EBADF` after rejection. The retained
behavioral RED failed before that change and GREEN passed afterward. No planned
demonstration was removed. Original snippets above are the historical plan,
not a replacement for the final source or its evidence.

## Executed Local Closeout

- Both tasks are implemented and independently approved; all four additions
  were independently reviewed together, 468 lines, spec PASS / quality APPROVED.
- Main's final-source run passed 83 tests in 6.004 seconds, exit 0. Separate
  import and synthetic contract probes passed; neither is a real tokenizer run.
- All 104 entry-protected source/data/document/evidence files retain their
  hashes. Branch/HEAD and tracked/index contents remain unchanged; additions
  are local and untracked, with no commit or push.
- Evidence and exclusive closeout receipt:
  `/tmp/tlf-tokenizer-runtime-20261112-TeMBri/final-verification.json`.
- The main runtime ledger records original findings, corrections, task reviews,
  full review and remaining gates. Historical pre-fix tests are not relabeled as
  final-code evidence.
- Final boundary: `LOCAL_RUNTIME_PREPARATION_ONLY_NOT_LOADED`. Exact real
  metadata admission, owned loader/session, 324-row controller, launch-wrapper
  enforcement, real tokenizer execution and scientific evaluation remain open.
