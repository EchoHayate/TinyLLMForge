import argparse
import contextlib
import ctypes
import errno
import fcntl
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import sys
import tarfile
import time


SCHEMA_VERSION = 1
RENAME_EXCHANGE = 0x2

_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_REGULAR_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW
_REGULAR_WRITE_FLAGS = (
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
)
_EMBEDDED_DIRECTORY = ".tinyllmforge-scratch"
_COMMIT_FILE = "commit.json"
_MANIFEST_FILE = "source-files.sha256"
_EXPLICIT_PATHS_FILE = "explicit-paths.txt"
_DETACHED_HEAD_FILE = "source-head.txt"
_DETACHED_MANIFEST_FILE = "source-files.sha256"
_DETACHED_TRANSACTION_FILE = "source-transaction.txt"
_EMBEDDED_FILES = frozenset(
    (_COMMIT_FILE, _MANIFEST_FILE, _EXPLICIT_PATHS_FILE)
)
_HASH_LENGTH = 64
_FORBIDDEN_PARTS = frozenset(
    (
        ".git",
        "artifacts",
        "experiments",
        ".cache",
        "cache",
        "caches",
        "__pycache__",
        ".pytest_cache",
        "log",
        "logs",
        "raw-output",
        "raw_output",
        "rawoutput",
    )
)
_FORBIDDEN_FILE_PATTERNS = (
    "._*",
    "*.pyc",
    "*.log",
    "*.pid",
    "*.out",
    "*.7z",
    "*.bz2",
    "*.gz",
    "*.lz",
    "*.lz4",
    "*.rar",
    "*.tar",
    "*.tar.*",
    "*.tbz",
    "*.tbz2",
    "*.tgz",
    "*.txz",
    "*.xz",
    "*.zip",
    "*.zst",
    "*review-package.diff",
)
_COMMIT_KEYS = frozenset(
    (
        "schema_version",
        "operation",
        "nonce",
        "source_head",
        "explicit_paths",
        "explicit_path_sha256",
        "source_manifest_sha256",
        "source_file_count",
        "created_at_unix_ns",
    )
)

_LIBC = ctypes.CDLL(None, use_errno=True)
try:
    _RENAMEAT2 = _LIBC.renameat2
except AttributeError:
    _RENAMEAT2 = None
else:
    _RENAMEAT2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    _RENAMEAT2.restype = ctypes.c_int


class TransactionError(RuntimeError):
    pass


class _GenerationNotCommitted(TransactionError):
    pass


class TransactionInterrupted(TransactionError):
    def __init__(self, signum):
        super().__init__(
            "transaction interrupted by signal {}".format(signum)
        )
        self.signum = signum


class InjectedFailure(TransactionError):
    pass


class CommitReceipt:
    def __init__(
        self,
        *,
        operation,
        nonce,
        source_head,
        explicit_paths,
        explicit_path_sha256,
        source_manifest_sha256,
        source_file_count,
        created_at_unix_ns,
    ):
        self.operation = operation
        self.nonce = nonce
        self.source_head = source_head
        self.explicit_paths = tuple(explicit_paths)
        self.explicit_path_sha256 = dict(explicit_path_sha256)
        self.source_manifest_sha256 = source_manifest_sha256
        self.source_file_count = source_file_count
        self.created_at_unix_ns = created_at_unix_ns


class _StagedDelta:
    def __init__(
        self,
        *,
        remote_root,
        nonce,
        explicit_paths,
        root_info,
        transactions_fd,
        transactions_info,
        name,
        delta_fd,
        delta_info,
    ):
        self.remote_root = str(remote_root)
        self.nonce = nonce
        self.explicit_paths = tuple(explicit_paths)
        self.root_info = root_info
        self.transactions_fd = transactions_fd
        self.transactions_info = transactions_info
        self.name = name
        self.delta_fd = delta_fd
        self.delta_info = delta_info
        self.closed = False

    def __fspath__(self):
        return str(
            Path(self.remote_root) / ".transactions" / self.name
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                named = os.stat(
                    self.name,
                    dir_fd=self.transactions_fd,
                    follow_symlinks=False,
                )
            except (FileNotFoundError, OSError):
                named = None
            if (
                named is not None
                and stat.S_ISDIR(named.st_mode)
                and named.st_dev == self.delta_info.st_dev
                and named.st_ino == self.delta_info.st_ino
            ):
                try:
                    _remove_tree_at(self.transactions_fd, self.name)
                except BaseException:
                    pass
        finally:
            _close_fd_best_effort(self.delta_fd)
            _close_fd_best_effort(self.transactions_fd)


def _raise_transaction_interrupted(signum, frame):
    del frame
    raise TransactionInterrupted(signum)


def open_directory_no_follow(path):
    return os.open(str(path), _DIRECTORY_FLAGS)


def _split_relative_path(path):
    if not isinstance(path, str) or not path:
        raise TransactionError("path must be a non-empty string")
    if any(character in path for character in ("\0", "\n", "\r")):
        raise TransactionError("path contains a control character")
    parsed = PurePosixPath(path)
    if (
        parsed == PurePosixPath(".")
        or parsed.is_absolute()
        or ".." in parsed.parts
        or not parsed.parts
    ):
        raise TransactionError("path is not relative: {}".format(path))
    return tuple(parsed.parts)


def _open_directory_at(parent_fd, relative_path):
    parts = _split_relative_path(relative_path)
    current_fd = os.dup(parent_fd)
    try:
        for part in parts:
            next_fd = os.open(part, _DIRECTORY_FLAGS, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _open_regular_at(parent_fd, name):
    fd = os.open(name, _REGULAR_READ_FLAGS, dir_fd=parent_fd)
    try:
        mode = os.fstat(fd).st_mode
        if not stat.S_ISREG(mode):
            raise TransactionError(
                "embedded member is not a regular file: {}".format(name)
            )
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read_all(fd):
    chunks = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _read_regular_at(parent_fd, name):
    try:
        fd = _open_regular_at(parent_fd, name)
    except OSError as exc:
        raise TransactionError(
            "cannot open regular file {}: {}".format(name, exc)
        )
    try:
        return _read_all(fd)
    finally:
        os.close(fd)


def _write_regular_at(parent_fd, name, data):
    temporary = ".{}.tmp-{}".format(name, os.getpid())
    fd = None
    try:
        fd = os.open(
            temporary,
            _REGULAR_WRITE_FLAGS,
            0o600,
            dir_fd=parent_fd,
        )
        view = memoryview(data)
        written = 0
        while written < len(view):
            written += os.write(fd, view[written:])
        os.fsync(fd)
        os.close(fd)
        fd = None
        os.replace(
            temporary,
            name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
    finally:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass


def _is_forbidden(relative_path):
    parts = PurePosixPath(relative_path).parts
    return (
        any(part in _FORBIDDEN_PARTS for part in parts)
        or any(
            fnmatch.fnmatchcase(parts[-1], pattern)
            for pattern in _FORBIDDEN_FILE_PATTERNS
        )
    )


def _validate_explicit_paths(paths):
    normalized = []
    for path in paths:
        parts = _split_relative_path(path)
        normalized_path = PurePosixPath(*parts).as_posix()
        if path != normalized_path:
            raise TransactionError(
                "explicit path is not canonical: {}".format(path)
            )
        if _is_forbidden(normalized_path):
            raise TransactionError(
                "explicit path is forbidden: {}".format(normalized_path)
            )
        normalized.append(normalized_path)
    if len(set(normalized)) != len(normalized):
        raise TransactionError("explicit paths contain duplicates")
    return tuple(normalized)


def _validate_nonce(nonce):
    if not isinstance(nonce, str) or not nonce:
        raise TransactionError("transaction nonce must be non-empty")
    parts = _split_relative_path(nonce)
    if len(parts) != 1 or parts[0] != nonce:
        raise TransactionError(
            "transaction nonce must be one canonical path component"
        )
    return nonce


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_fd(fd):
    digest = hashlib.sha256()
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _manifest_file_sha256(fd):
    digest = hashlib.sha256()
    digest.update(b"file\0")
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _hash_explicit_path(source_fd, relative_path):
    parts = _split_relative_path(relative_path)
    parent_fd = os.dup(source_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, _DIRECTORY_FLAGS, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        file_fd = _open_regular_at(parent_fd, parts[-1])
        try:
            return _sha256_fd(file_fd)
        finally:
            os.close(file_fd)
    except (OSError, TransactionError) as exc:
        raise TransactionError(
            "cannot hash explicit path {}: {}".format(relative_path, exc)
        )
    finally:
        os.close(parent_fd)


def _walk_manifest(directory_fd, prefix, entries):
    for name in sorted(os.listdir(directory_fd)):
        relative_path = name if not prefix else prefix + "/" + name
        if not prefix and name == _EMBEDDED_DIRECTORY:
            continue
        if _is_forbidden(relative_path):
            raise TransactionError(
                "source contains forbidden member: {}".format(relative_path)
            )
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        mode = info.st_mode
        if stat.S_ISDIR(mode):
            child_fd = os.open(
                name, _DIRECTORY_FLAGS, dir_fd=directory_fd
            )
            try:
                _walk_manifest(child_fd, relative_path, entries)
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(mode):
            file_fd = _open_regular_at(directory_fd, name)
            try:
                entries.append(
                    (relative_path, _manifest_file_sha256(file_fd))
                )
            finally:
                os.close(file_fd)
        elif stat.S_ISLNK(mode):
            target = os.readlink(name, dir_fd=directory_fd)
            entries.append(
                (
                    relative_path,
                    _sha256_bytes(b"symlink\0" + os.fsencode(target)),
                )
            )
        else:
            raise TransactionError(
                "source contains unsupported member: {}".format(
                    relative_path
                )
            )


def _source_manifest(source_fd):
    entries = []
    _walk_manifest(source_fd, "", entries)
    data = "".join(
        "{}  {}\n".format(digest, path) for path, digest in entries
    ).encode("utf-8")
    return entries, data


def _validate_hash(value, field):
    if (
        not isinstance(value, str)
        or len(value) != _HASH_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TransactionError("{} is not a sha256 digest".format(field))


def _validate_source_head(source_head):
    if (
        not isinstance(source_head, str)
        or len(source_head) != 40
        or any(
            character not in "0123456789abcdef"
            for character in source_head
        )
    ):
        raise TransactionError("source head must be 40 lowercase hex digits")
    return source_head


def _receipt_payload(receipt):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "operation": receipt.operation,
        "nonce": receipt.nonce,
        "source_head": receipt.source_head,
        "explicit_paths": list(receipt.explicit_paths),
        "explicit_path_sha256": dict(receipt.explicit_path_sha256),
        "source_manifest_sha256": receipt.source_manifest_sha256,
        "source_file_count": receipt.source_file_count,
        "created_at_unix_ns": receipt.created_at_unix_ns,
    }
    _validate_payload(payload)
    return payload


def _validate_payload(payload):
    if not isinstance(payload, dict) or frozenset(payload) != _COMMIT_KEYS:
        raise TransactionError("commit JSON has unexpected fields")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise TransactionError("commit JSON has wrong schema version")
    for field in ("operation", "nonce", "source_head"):
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            raise TransactionError(
                "commit JSON field {} must be non-empty".format(field)
            )
    if payload["operation"] not in ("init", "sync"):
        raise TransactionError("commit JSON has unknown operation")
    paths_value = payload.get("explicit_paths")
    if not isinstance(paths_value, list) or not all(
        isinstance(path, str) for path in paths_value
    ):
        raise TransactionError("explicit_paths must be a list of strings")
    paths = _validate_explicit_paths(paths_value)
    hashes = payload.get("explicit_path_sha256")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != set(paths)
        or not all(isinstance(path, str) for path in hashes)
    ):
        raise TransactionError("explicit path hashes do not match paths")
    for path, digest in hashes.items():
        _validate_hash(digest, "explicit path hash {}".format(path))
    _validate_hash(
        payload.get("source_manifest_sha256"),
        "source_manifest_sha256",
    )
    for field in ("source_file_count", "created_at_unix_ns"):
        value = payload.get(field)
        if type(value) is not int or value < 0:
            raise TransactionError(
                "commit JSON field {} must be a non-negative integer".format(
                    field
                )
            )
    return paths


def _canonical_json(payload):
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def write_embedded_receipt(generation_fd, receipt):
    payload = _receipt_payload(receipt)
    paths = tuple(payload["explicit_paths"])
    entries, manifest = _source_manifest(generation_fd)
    if payload["source_file_count"] != len(entries):
        raise TransactionError("source file count does not match generation")
    if payload["source_manifest_sha256"] != _sha256_bytes(manifest):
        raise TransactionError("source manifest digest does not match generation")
    actual_hashes = {
        path: _hash_explicit_path(generation_fd, path) for path in paths
    }
    if payload["explicit_path_sha256"] != actual_hashes:
        raise TransactionError("explicit path hashes do not match generation")
    try:
        os.mkdir(_EMBEDDED_DIRECTORY, 0o700, dir_fd=generation_fd)
    except OSError as exc:
        raise TransactionError(
            "cannot create embedded receipt directory: {}".format(exc)
        )
    embedded_fd = os.open(
        _EMBEDDED_DIRECTORY, _DIRECTORY_FLAGS, dir_fd=generation_fd
    )
    try:
        _write_regular_at(
            embedded_fd, _COMMIT_FILE, _canonical_json(payload)
        )
        _write_regular_at(embedded_fd, _MANIFEST_FILE, manifest)
        paths_data = "".join(
            "{}\n".format(path) for path in paths
        ).encode("utf-8")
        _write_regular_at(
            embedded_fd, _EXPLICIT_PATHS_FILE, paths_data
        )
        os.fsync(embedded_fd)
    finally:
        os.close(embedded_fd)


def _generation_receipt(
    generation_fd,
    *,
    operation,
    nonce,
    source_head,
    explicit_paths
):
    paths = _validate_explicit_paths(explicit_paths)
    entries, manifest = _source_manifest(generation_fd)
    return CommitReceipt(
        operation=operation,
        nonce=nonce,
        source_head=source_head,
        explicit_paths=paths,
        explicit_path_sha256={
            path: _hash_explicit_path(generation_fd, path)
            for path in paths
        },
        source_manifest_sha256=_sha256_bytes(manifest),
        source_file_count=len(entries),
        created_at_unix_ns=time.time_ns(),
    )


def _decode_utf8(data, name):
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TransactionError("{} is not UTF-8: {}".format(name, exc))


def _read_generation_fd(
    source_fd,
    *,
    expected_nonce=None,
    expected_operation=None,
    expected_head=None,
    expected_paths=None
):
    try:
        embedded_fd = os.open(
            _EMBEDDED_DIRECTORY,
            _DIRECTORY_FLAGS,
            dir_fd=source_fd,
        )
    except OSError as exc:
        raise TransactionError(
            "cannot open embedded receipt directory: {}".format(exc)
        )
    try:
        embedded_entries = frozenset(os.listdir(embedded_fd))
        if embedded_entries != _EMBEDDED_FILES:
            raise TransactionError(
                "embedded receipt directory has unexpected members: "
                "{}".format(", ".join(sorted(embedded_entries)))
            )
        commit_data = _read_regular_at(embedded_fd, _COMMIT_FILE)
        manifest_data = _read_regular_at(embedded_fd, _MANIFEST_FILE)
        explicit_paths_data = _read_regular_at(
            embedded_fd, _EXPLICIT_PATHS_FILE
        )
    finally:
        os.close(embedded_fd)

    commit_text = _decode_utf8(commit_data, _COMMIT_FILE)
    try:
        payload = json.loads(commit_text)
    except ValueError as exc:
        raise TransactionError("commit JSON is invalid: {}".format(exc))
    paths = _validate_payload(payload)
    if commit_data != _canonical_json(payload):
        raise TransactionError("commit JSON is not canonical")

    expected_paths_data = "".join(
        "{}\n".format(path) for path in paths
    ).encode("utf-8")
    if explicit_paths_data != expected_paths_data:
        raise TransactionError("embedded explicit paths do not match commit")

    checks = (
        ("nonce", expected_nonce),
        ("operation", expected_operation),
        ("source_head", expected_head),
    )
    for field, expected in checks:
        if expected is not None and payload[field] != expected:
            raise TransactionError(
                "committed {} does not match expected value".format(field)
            )
    if expected_paths is not None:
        normalized_expected = _validate_explicit_paths(expected_paths)
        if paths != normalized_expected:
            raise TransactionError(
                "committed explicit paths do not match expected paths"
            )

    actual_hashes = {
        path: _hash_explicit_path(source_fd, path) for path in paths
    }
    if payload["explicit_path_sha256"] != actual_hashes:
        raise TransactionError("committed explicit path hash mismatch")

    entries, actual_manifest = _source_manifest(source_fd)
    if manifest_data != actual_manifest:
        raise TransactionError("embedded source manifest mismatch")
    if payload["source_manifest_sha256"] != _sha256_bytes(actual_manifest):
        raise TransactionError("source manifest digest mismatch")
    if payload["source_file_count"] != len(entries):
        raise TransactionError("source manifest file count mismatch")

    return CommitReceipt(
        operation=payload["operation"],
        nonce=payload["nonce"],
        source_head=payload["source_head"],
        explicit_paths=paths,
        explicit_path_sha256=payload["explicit_path_sha256"],
        source_manifest_sha256=payload["source_manifest_sha256"],
        source_file_count=payload["source_file_count"],
        created_at_unix_ns=payload["created_at_unix_ns"],
    )


def read_committed_generation(
    remote_root,
    *,
    expected_nonce=None,
    expected_operation=None,
    expected_head=None,
    expected_paths=None
):
    try:
        root_fd = open_directory_no_follow(Path(remote_root))
    except OSError as exc:
        raise TransactionError(
            "cannot open remote root without following symlinks: {}".format(
                exc
            )
        )
    try:
        try:
            source_fd = os.open("source", _DIRECTORY_FLAGS, dir_fd=root_fd)
        except OSError as exc:
            raise TransactionError(
                "cannot open committed source without following symlinks: "
                "{}".format(exc)
            )
        try:
            return _read_generation_fd(
                source_fd,
                expected_nonce=expected_nonce,
                expected_operation=expected_operation,
                expected_head=expected_head,
                expected_paths=expected_paths,
            )
        finally:
            os.close(source_fd)
    finally:
        os.close(root_fd)


def _close_fd_best_effort(fd):
    try:
        os.close(fd)
    except OSError:
        return
    except BaseException:
        try:
            os.close(fd)
        except BaseException:
            return


@contextlib.contextmanager
def locked_remote_root(remote_root):
    root_fd = open_directory_no_follow(Path(remote_root))
    transactions_fd = None
    lock_fd = None
    lock_acquired = False
    try:
        try:
            os.mkdir(".transactions", 0o700, dir_fd=root_fd)
        except FileExistsError:
            pass
        transactions_fd = os.open(
            ".transactions", _DIRECTORY_FLAGS, dir_fd=root_fd
        )
        lock_fd = os.open(
            "source.lock",
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
            dir_fd=transactions_fd,
        )
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise TransactionError("source lock is not a regular file")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        lock_acquired = True
        yield root_fd
    finally:
        if lock_fd is not None:
            if lock_acquired:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except BaseException:
                    pass
            _close_fd_best_effort(lock_fd)
        if transactions_fd is not None:
            _close_fd_best_effort(transactions_fd)
        _close_fd_best_effort(root_fd)


def _rename_exchange_at(
    left_parent_fd, left_name, right_parent_fd, right_name
):
    if _RENAMEAT2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable")
    result = _RENAMEAT2(
        left_parent_fd,
        os.fsencode(left_name),
        right_parent_fd,
        os.fsencode(right_name),
        RENAME_EXCHANGE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            "{} <-> {}".format(left_name, right_name),
        )


def rename_exchange(parent_fd, left, right):
    left_parts = _split_relative_path(left)
    right_parts = _split_relative_path(right)
    left_parent_fd = None
    right_parent_fd = None
    try:
        left_parent_fd = os.dup(parent_fd)
        right_parent_fd = os.dup(parent_fd)
        for part in left_parts[:-1]:
            next_fd = os.open(
                part, _DIRECTORY_FLAGS, dir_fd=left_parent_fd
            )
            os.close(left_parent_fd)
            left_parent_fd = next_fd
        for part in right_parts[:-1]:
            next_fd = os.open(
                part, _DIRECTORY_FLAGS, dir_fd=right_parent_fd
            )
            os.close(right_parent_fd)
            right_parent_fd = next_fd
        _rename_exchange_at(
            left_parent_fd,
            left_parts[-1],
            right_parent_fd,
            right_parts[-1],
        )
    finally:
        if right_parent_fd is not None:
            os.close(right_parent_fd)
        if left_parent_fd is not None:
            os.close(left_parent_fd)


def _remove_tree_contents(directory_fd):
    for name in os.listdir(directory_fd):
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            child_fd = os.open(
                name, _DIRECTORY_FLAGS, dir_fd=directory_fd
            )
            try:
                _remove_tree_contents(child_fd)
            finally:
                os.close(child_fd)
            os.rmdir(name, dir_fd=directory_fd)
        else:
            os.unlink(name, dir_fd=directory_fd)


def _remove_tree_at(parent_fd, name):
    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(info.st_mode):
        directory_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        try:
            _remove_tree_contents(directory_fd)
        finally:
            os.close(directory_fd)
        os.rmdir(name, dir_fd=parent_fd)
    else:
        os.unlink(name, dir_fd=parent_fd)


def _remove_empty_nonce_directory(
    root_fd, generation_name, expected_parent_info=None
):
    parts = _split_relative_path(generation_name)
    if len(parts) < 2:
        return
    parent_path = PurePosixPath(*parts[:-1])
    grandparent_parts = parent_path.parts[:-1]
    parent_name = parent_path.parts[-1]
    grandparent_fd = os.dup(root_fd)
    try:
        for part in grandparent_parts:
            next_fd = os.open(part, _DIRECTORY_FLAGS, dir_fd=grandparent_fd)
            os.close(grandparent_fd)
            grandparent_fd = next_fd
        try:
            if expected_parent_info is not None:
                actual_parent_info = os.stat(
                    parent_name,
                    dir_fd=grandparent_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(actual_parent_info.st_mode)
                    or actual_parent_info.st_dev
                    != expected_parent_info.st_dev
                    or actual_parent_info.st_ino
                    != expected_parent_info.st_ino
                ):
                    return
            os.rmdir(parent_name, dir_fd=grandparent_fd)
        except OSError as exc:
            if exc.errno not in (errno.ENOENT, errno.ENOTEMPTY):
                raise
    finally:
        os.close(grandparent_fd)


_TRANSACTION_SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)


def _has_committed_result(committed_result):
    return (
        committed_result is not None
        and len(committed_result) == 1
        and committed_result[0] is not None
    )


def _consume_pending_transaction_signals():
    pending = set(signal.sigpending()).intersection(_TRANSACTION_SIGNALS)
    for signum in sorted(pending):
        signal.sigwait((signum,))


@contextlib.contextmanager
def _blocked_transaction_signals(retain_if_committed=None):
    previous_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK, _TRANSACTION_SIGNALS
    )
    try:
        yield
    finally:
        if not _has_committed_result(retain_if_committed):
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


@contextlib.contextmanager
def _transaction_signal_handlers(
    committed_result=None,
    retain_committed_signals=False,
):
    installed = []
    signals = _TRANSACTION_SIGNALS
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
    body_mask_restore_attempted = False
    body_completed = False
    primary_error = None
    try:
        for signum in signals:
            previous = signal.getsignal(signum)
            installed.append((signum, previous))
            try:
                signal.signal(signum, _raise_transaction_interrupted)
            except BaseException:
                if (
                    signal.getsignal(signum)
                    is not _raise_transaction_interrupted
                ):
                    installed.pop()
                raise
        body_mask_restore_attempted = True
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        yield
        body_completed = True
    except BaseException as exc:
        primary_error = exc
        if not _has_committed_result(committed_result):
            raise
    finally:
        cleanup_error = None
        if body_mask_restore_attempted:
            try:
                signal.pthread_sigmask(signal.SIG_BLOCK, signals)
            except BaseException as exc:
                cleanup_error = exc
        if _has_committed_result(committed_result):
            try:
                _consume_pending_transaction_signals()
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        retain = (
            retain_committed_signals
            and _has_committed_result(committed_result)
        )
        if not retain:
            for signum, previous in reversed(installed):
                try:
                    signal.signal(signum, previous)
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if (
            cleanup_error is not None
            and primary_error is None
            and not body_completed
            and not _has_committed_result(committed_result)
        ):
            raise cleanup_error


def _inject(fault_injector, point):
    if fault_injector is not None:
        fault_injector(point)


def _same_file_identity(left_info, right_info):
    return (
        left_info.st_dev == right_info.st_dev
        and left_info.st_ino == right_info.st_ino
    )


def _require_exact_receipt(actual, expected):
    if _receipt_payload(actual) != _receipt_payload(expected):
        raise TransactionError(
            "committed receipt does not exactly match supplied receipt"
        )


def _open_source_for_confirmation(root_fd, generation_info):
    try:
        source_fd = os.open("source", _DIRECTORY_FLAGS, dir_fd=root_fd)
    except OSError as exc:
        raise TransactionError(
            "cannot open promoted source without following symlinks: {}".format(
                exc
            )
        )
    try:
        source_info = os.fstat(source_fd)
        if not _same_file_identity(source_info, generation_info):
            raise TransactionError(
                "promoted source does not match validated generation"
            )
    except BaseException:
        try:
            os.close(source_fd)
        except OSError:
            pass
        raise
    return source_fd


def _materialize_initial_receipts_at(root_fd, receipt):
    if receipt.operation != "init":
        raise TransactionError("detached init receipts require init operation")
    try:
        os.mkdir("receipts", 0o700, dir_fd=root_fd)
    except FileExistsError:
        pass
    receipts_fd = os.open("receipts", _DIRECTORY_FLAGS, dir_fd=root_fd)
    source_fd = None
    embedded_fd = None
    try:
        source_fd = os.open("source", _DIRECTORY_FLAGS, dir_fd=root_fd)
        embedded_fd = os.open(
            _EMBEDDED_DIRECTORY,
            _DIRECTORY_FLAGS,
            dir_fd=source_fd,
        )
        manifest = _read_regular_at(embedded_fd, _MANIFEST_FILE)
        values = (
            (
                _DETACHED_HEAD_FILE,
                (receipt.source_head + "\n").encode("utf-8"),
            ),
            (_DETACHED_MANIFEST_FILE, manifest),
            (
                _DETACHED_TRANSACTION_FILE,
                (receipt.nonce + "\n").encode("utf-8"),
            ),
        )
        for name, data in values:
            try:
                info = os.stat(
                    name, dir_fd=receipts_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISREG(info.st_mode):
                    raise TransactionError(
                        "detached receipt is not a regular file: {}".format(
                            name
                        )
                    )
            _write_regular_at(receipts_fd, name, data)
        os.fsync(receipts_fd)
    finally:
        if embedded_fd is not None:
            os.close(embedded_fd)
        if source_fd is not None:
            os.close(source_fd)
        os.close(receipts_fd)


def _validate_initial_receipt_targets_at(root_fd):
    try:
        os.mkdir("receipts", 0o700, dir_fd=root_fd)
    except FileExistsError:
        pass
    try:
        receipts_fd = os.open(
            "receipts", _DIRECTORY_FLAGS, dir_fd=root_fd
        )
    except OSError as exc:
        raise TransactionError(
            "cannot open detached receipt directory: {}".format(exc)
        )
    try:
        for name in (
            _DETACHED_HEAD_FILE,
            _DETACHED_MANIFEST_FILE,
            _DETACHED_TRANSACTION_FILE,
        ):
            try:
                info = os.stat(
                    name, dir_fd=receipts_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode):
                raise TransactionError(
                    "detached receipt is not a regular file: {}".format(name)
                )
    finally:
        os.close(receipts_fd)


def _confirm_initial_generation(
    remote_root,
    generation_name,
    nonce,
    head,
    _committed_result=None,
    _manage_signal_handlers=True,
    _retain_committed_signals=False,
):
    preserve_committed_success = _committed_result is not None
    committed_result = (
        [None] if _committed_result is None else _committed_result
    )
    signal_context = (
        _transaction_signal_handlers(
            committed_result if preserve_committed_success else None
        )
        if _manage_signal_handlers
        else contextlib.nullcontext()
    )
    with signal_context:
        with locked_remote_root(remote_root) as root_fd:
            try:
                source_fd = os.open(
                    "source", _DIRECTORY_FLAGS, dir_fd=root_fd
                )
            except OSError as exc:
                raise _GenerationNotCommitted(
                    "matching committed source is unavailable: {}".format(
                        exc
                    )
                )
            try:
                try:
                    receipt = _read_generation_fd(
                        source_fd,
                        expected_nonce=nonce,
                        expected_operation="init",
                        expected_head=head,
                        expected_paths=(),
                    )
                except TransactionError as exc:
                    raise _GenerationNotCommitted(
                        "source is not the expected committed generation: "
                        "{}".format(exc)
                    )
            finally:
                os.close(source_fd)
            _validate_initial_receipt_targets_at(root_fd)
            with _blocked_transaction_signals(
                committed_result if _retain_committed_signals else None
            ):
                committed_result[0] = receipt
            try:
                _materialize_initial_receipts_at(root_fd, receipt)
            except BaseException:
                if (
                    not preserve_committed_success
                    or not _has_committed_result(committed_result)
                ):
                    raise
            generation_parts = _split_relative_path(generation_name)
            generation_parent_fd = None
            generation_parent_info = None
            try:
                generation_parent_fd = _open_directory_at(
                    root_fd,
                    PurePosixPath(*generation_parts[:-1]).as_posix(),
                )
                generation_parent_info = os.fstat(generation_parent_fd)
                _remove_tree_at(
                    generation_parent_fd, generation_parts[-1]
                )
            except BaseException:
                pass
            finally:
                if generation_parent_fd is not None:
                    try:
                        os.close(generation_parent_fd)
                    except BaseException:
                        pass
            try:
                _remove_empty_nonce_directory(
                    root_fd,
                    generation_name,
                    expected_parent_info=generation_parent_info,
                )
            except BaseException:
                pass
            return receipt
    return committed_result[0]


def promote_generation(
    remote_root,
    generation_name,
    receipt,
    fault_injector=None,
    receipt_factory=None,
    pre_commit=None,
    post_commit=None,
    _committed_result=None,
    _manage_signal_handlers=True,
    _retain_committed_signals=False,
):
    generation_parts = _split_relative_path(generation_name)
    if (
        len(generation_parts) != 3
        or generation_parts[0] != ".transactions"
        or generation_parts[2] != "generation"
    ):
        raise TransactionError(
            "generation must be .transactions/<nonce>/generation"
        )
    if receipt is not None and generation_parts[1] != receipt.nonce:
        raise TransactionError("generation nonce does not match receipt")

    preserve_committed_success = _committed_result is not None
    committed_result = (
        [None] if _committed_result is None else _committed_result
    )
    signal_context = (
        _transaction_signal_handlers(
            committed_result if preserve_committed_success else None
        )
        if _manage_signal_handlers
        else contextlib.nullcontext()
    )
    with signal_context:
        with locked_remote_root(remote_root) as root_fd:
            _inject(fault_injector, "after_lock")
            generation_parent_fd = None
            generation_fd = None
            generation_parent_info = None
            try:
                try:
                    generation_parent_fd = _open_directory_at(
                        root_fd,
                        PurePosixPath(*generation_parts[:-1]).as_posix(),
                    )
                    generation_fd = os.open(
                        generation_parts[-1],
                        _DIRECTORY_FLAGS,
                        dir_fd=generation_parent_fd,
                    )
                except OSError as exc:
                    if generation_parent_fd is not None:
                        os.close(generation_parent_fd)
                        generation_parent_fd = None
                    raise TransactionError(
                        "cannot open generation without following symlinks: "
                        "{}".format(exc)
                    )
                generation_parent_info = os.fstat(generation_parent_fd)
                _inject(fault_injector, "after_generation_ready")
                if pre_commit is not None:
                    pre_commit(root_fd)
                embedded_receipt_exists = False
                if receipt is None:
                    if receipt_factory is None:
                        raise TransactionError(
                            "receipt or receipt factory is required"
                        )
                    try:
                        embedded_info = os.stat(
                            _EMBEDDED_DIRECTORY,
                            dir_fd=generation_fd,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        pass
                    else:
                        if not stat.S_ISDIR(embedded_info.st_mode):
                            raise TransactionError(
                                "embedded receipt is not a real directory"
                            )
                        embedded_receipt_exists = True
                    receipt = receipt_factory(generation_fd)
                if generation_parts[1] != receipt.nonce:
                    raise TransactionError(
                        "generation nonce does not match receipt"
                    )
                if not embedded_receipt_exists:
                    write_embedded_receipt(generation_fd, receipt)
                _inject(fault_injector, "after_embedded_receipt")
                _read_generation_fd(
                    generation_fd,
                    expected_nonce=receipt.nonce,
                    expected_operation=receipt.operation,
                    expected_head=receipt.source_head,
                    expected_paths=receipt.explicit_paths,
                )

                _inject(fault_injector, "before_exchange")
                source_exists = True
                try:
                    source_info = os.stat(
                        "source", dir_fd=root_fd, follow_symlinks=False
                    )
                except FileNotFoundError:
                    source_exists = False
                    prior_source_info = None
                else:
                    if not stat.S_ISDIR(source_info.st_mode):
                        raise TransactionError(
                            "source is not a real directory"
                        )
                    prior_source_fd = os.open(
                        "source", _DIRECTORY_FLAGS, dir_fd=root_fd
                    )
                    try:
                        if not _same_file_identity(
                            source_info, os.fstat(prior_source_fd)
                        ):
                            raise TransactionError(
                                "source entry changed before promotion"
                            )
                    finally:
                        os.close(prior_source_fd)

                generation_info = os.fstat(generation_fd)
                try:
                    named_info = os.stat(
                        generation_parts[-1],
                        dir_fd=generation_parent_fd,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise TransactionError(
                        "generation entry changed before promotion: "
                        "{}".format(exc)
                    )
                if (
                    not stat.S_ISDIR(named_info.st_mode)
                    or not _same_file_identity(
                        named_info, generation_info
                    )
                ):
                    raise TransactionError(
                        "generation entry changed before promotion"
                    )

                try:
                    with _blocked_transaction_signals(
                        committed_result
                        if (
                            preserve_committed_success
                            and _retain_committed_signals
                        )
                        else None
                    ):
                        if not source_exists:
                            os.rename(
                                generation_parts[-1],
                                "source",
                                src_dir_fd=generation_parent_fd,
                                dst_dir_fd=root_fd,
                            )
                        else:
                            _rename_exchange_at(
                                root_fd,
                                "source",
                                generation_parent_fd,
                                generation_parts[-1],
                            )
                        committed_result[0] = receipt
                except BaseException:
                    if (
                        not preserve_committed_success
                        or not _has_committed_result(committed_result)
                    ):
                        raise
                try:
                    _inject(fault_injector, "after_exchange")
                except BaseException:
                    if (
                        not preserve_committed_success
                        or not _has_committed_result(committed_result)
                    ):
                        raise

                try:
                    committed_fd = _open_source_for_confirmation(
                        root_fd, generation_info
                    )
                    try:
                        committed_receipt = _read_generation_fd(
                            committed_fd,
                            expected_nonce=receipt.nonce,
                            expected_operation=receipt.operation,
                            expected_head=receipt.source_head,
                            expected_paths=receipt.explicit_paths,
                        )
                        _require_exact_receipt(committed_receipt, receipt)
                    finally:
                        os.close(committed_fd)
                except BaseException:
                    if (
                        not preserve_committed_success
                        or not _has_committed_result(committed_result)
                    ):
                        raise
                try:
                    _inject(fault_injector, "before_external_receipts")
                except BaseException:
                    if (
                        not preserve_committed_success
                        or not _has_committed_result(committed_result)
                    ):
                        raise
                if post_commit is not None:
                    try:
                        post_commit(root_fd, receipt)
                    except BaseException:
                        if (
                            not preserve_committed_success
                            or not _has_committed_result(committed_result)
                        ):
                            raise
                try:
                    _inject(fault_injector, "after_external_receipts")
                except BaseException:
                    if (
                        not preserve_committed_success
                        or not _has_committed_result(committed_result)
                    ):
                        raise
                try:
                    _inject(
                        fault_injector,
                        "before_old_generation_cleanup",
                    )
                except BaseException:
                    if (
                        not preserve_committed_success
                        or not _has_committed_result(committed_result)
                    ):
                        raise
                return receipt
            finally:
                if generation_parent_fd is not None:
                    try:
                        _remove_tree_at(
                            generation_parent_fd,
                            generation_parts[-1],
                        )
                    except BaseException:
                        pass
                    if generation_fd is not None:
                        try:
                            os.close(generation_fd)
                        except BaseException:
                            pass
                    try:
                        os.close(generation_parent_fd)
                    except BaseException:
                        pass
                try:
                    _remove_empty_nonce_directory(
                        root_fd,
                        generation_name,
                        expected_parent_info=generation_parent_info,
                    )
                except BaseException:
                    pass
    return committed_result[0]


def commit_initial_generation(
    remote_root,
    generation_name,
    source_head,
    fault_injector=None,
    _committed_result=None,
    _manage_signal_handlers=True,
    _retain_committed_signals=False,
):
    generation_parts = _split_relative_path(generation_name)
    if (
        len(generation_parts) != 3
        or generation_parts[0] != ".transactions"
        or generation_parts[2] != "generation"
    ):
        raise TransactionError(
            "generation must be .transactions/<nonce>/generation"
        )
    nonce = generation_parts[1]
    head = _validate_source_head(source_head)
    preserve_committed_signals = _committed_result is not None
    committed_result = (
        [None] if _committed_result is None else _committed_result
    )
    if _manage_signal_handlers:
        signal_context = (
            _transaction_signal_handlers(committed_result)
            if preserve_committed_signals
            else _transaction_signal_handlers()
        )
    else:
        signal_context = contextlib.nullcontext()
    with signal_context:
        try:
            confirmed = _confirm_initial_generation(
                remote_root,
                generation_name,
                nonce,
                head,
                _committed_result=committed_result,
                _manage_signal_handlers=False,
                _retain_committed_signals=_retain_committed_signals,
            )
        except _GenerationNotCommitted:
            pass
        else:
            return confirmed

        def make_receipt(generation_fd):
            try:
                os.stat(
                    _EMBEDDED_DIRECTORY,
                    dir_fd=generation_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return _generation_receipt(
                    generation_fd,
                    operation="init",
                    nonce=nonce,
                    source_head=head,
                    explicit_paths=(),
                )
            return _read_generation_fd(
                generation_fd,
                expected_nonce=nonce,
                expected_operation="init",
                expected_head=head,
                expected_paths=(),
            )

        return promote_generation(
            remote_root,
            generation_name,
            None,
            fault_injector=fault_injector,
            receipt_factory=make_receipt,
            pre_commit=_validate_initial_receipt_targets_at,
            post_commit=_materialize_initial_receipts_at,
            _committed_result=committed_result,
            _manage_signal_handlers=False,
            _retain_committed_signals=_retain_committed_signals,
        )
    return committed_result[0]


def _copy_regular_file(source_fd, destination_fd, name, mode):
    temporary = ".{}.clone-{}-{}".format(name, os.getpid(), time.time_ns())
    output_fd = None
    try:
        output_fd = os.open(
            temporary,
            _REGULAR_WRITE_FLAGS,
            mode & 0o777,
            dir_fd=destination_fd,
        )
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            written = 0
            while written < len(view):
                written += os.write(output_fd, view[written:])
        os.fchmod(output_fd, mode & 0o777)
        os.fsync(output_fd)
        os.close(output_fd)
        output_fd = None
        os.replace(
            temporary,
            name,
            src_dir_fd=destination_fd,
            dst_dir_fd=destination_fd,
        )
    finally:
        if output_fd is not None:
            _close_fd_best_effort(output_fd)
        try:
            os.unlink(temporary, dir_fd=destination_fd)
        except FileNotFoundError:
            pass


def _clone_tree(source_fd, destination_fd, prefix=""):
    for name in sorted(os.listdir(source_fd)):
        if not prefix and name == _EMBEDDED_DIRECTORY:
            continue
        relative_path = name if not prefix else prefix + "/" + name
        if _is_forbidden(relative_path):
            raise TransactionError(
                "source contains forbidden member: {}".format(relative_path)
            )
        info = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            os.mkdir(name, 0o700, dir_fd=destination_fd)
            source_child_fd = os.open(
                name, _DIRECTORY_FLAGS, dir_fd=source_fd
            )
            destination_child_fd = os.open(
                name, _DIRECTORY_FLAGS, dir_fd=destination_fd
            )
            try:
                _clone_tree(
                    source_child_fd,
                    destination_child_fd,
                    relative_path,
                )
                os.fchmod(destination_child_fd, info.st_mode & 0o777)
            finally:
                os.close(destination_child_fd)
                os.close(source_child_fd)
        elif stat.S_ISREG(info.st_mode):
            file_fd = _open_regular_at(source_fd, name)
            try:
                _copy_regular_file(
                    file_fd,
                    destination_fd,
                    name,
                    info.st_mode,
                )
            finally:
                os.close(file_fd)
        elif stat.S_ISLNK(info.st_mode):
            os.symlink(
                os.readlink(name, dir_fd=source_fd),
                name,
                dir_fd=destination_fd,
            )
        else:
            raise TransactionError(
                "source contains unsupported member: {}".format(
                    relative_path
                )
            )


def _open_explicit_parent(directory_fd, relative_path, purpose):
    parts = _split_relative_path(relative_path)
    parent_fd = os.dup(directory_fd)
    try:
        for part in parts[:-1]:
            try:
                next_fd = os.open(
                    part, _DIRECTORY_FLAGS, dir_fd=parent_fd
                )
            except OSError as exc:
                raise TransactionError(
                    "incremental sync requires full init: unsafe or missing "
                    "{} parent for {}: {}".format(
                        purpose, relative_path, exc
                    )
                )
            os.close(parent_fd)
            parent_fd = next_fd
        return parent_fd, parts[-1]
    except BaseException:
        os.close(parent_fd)
        raise


def _apply_explicit_delta(generation_fd, delta_fd, explicit_paths):
    for relative_path in explicit_paths:
        delta_parent_fd, name = _open_explicit_parent(
            delta_fd, relative_path, "delta"
        )
        generation_parent_fd = None
        source_fd = None
        output_fd = None
        temporary = ".{}.sync-{}-{}".format(
            name, os.getpid(), time.time_ns()
        )
        try:
            source_fd = _open_regular_at(delta_parent_fd, name)
            generation_parent_fd, generation_name = _open_explicit_parent(
                generation_fd, relative_path, "remote"
            )
            try:
                existing = os.stat(
                    generation_name,
                    dir_fd=generation_parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                existing = None
            if existing is not None and not stat.S_ISREG(existing.st_mode):
                raise TransactionError(
                    "incremental sync requires full init: unsafe remote "
                    "final {}".format(relative_path)
                )
            source_mode = os.fstat(source_fd).st_mode
            output_fd = os.open(
                temporary,
                _REGULAR_WRITE_FLAGS,
                source_mode & 0o777,
                dir_fd=generation_parent_fd,
            )
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                view = memoryview(chunk)
                written = 0
                while written < len(view):
                    written += os.write(output_fd, view[written:])
            os.fchmod(output_fd, source_mode & 0o777)
            os.fsync(output_fd)
            os.close(output_fd)
            output_fd = None
            os.replace(
                temporary,
                generation_name,
                src_dir_fd=generation_parent_fd,
                dst_dir_fd=generation_parent_fd,
            )
        except (OSError, TransactionError) as exc:
            if isinstance(exc, TransactionError):
                raise
            raise TransactionError(
                "cannot apply explicit delta {}: {}".format(
                    relative_path, exc
                )
            )
        finally:
            if output_fd is not None:
                _close_fd_best_effort(output_fd)
            if generation_parent_fd is not None:
                try:
                    os.unlink(temporary, dir_fd=generation_parent_fd)
                except FileNotFoundError:
                    pass
                os.close(generation_parent_fd)
            if source_fd is not None:
                os.close(source_fd)
            os.close(delta_parent_fd)


def _sync_receipt_names(nonce):
    prefix = "sync-{}".format(nonce)
    return (
        prefix + ".paths.txt",
        prefix + ".sha256",
        prefix + ".state",
    )


def _sync_receipt_path(remote_root, nonce):
    return str(Path(remote_root) / "receipts" / _sync_receipt_names(nonce)[1])


def _materialize_sync_receipts_at(root_fd, receipt):
    if receipt.operation != "sync":
        raise TransactionError("detached sync receipts require sync operation")
    try:
        os.mkdir("receipts", 0o700, dir_fd=root_fd)
    except FileExistsError:
        pass
    receipts_fd = os.open("receipts", _DIRECTORY_FLAGS, dir_fd=root_fd)
    try:
        names = _sync_receipt_names(receipt.nonce)
        for name in names:
            try:
                info = os.stat(
                    name, dir_fd=receipts_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode):
                raise TransactionError(
                    "detached receipt is not a regular file: {}".format(name)
                )
        paths_data = "".join(
            "{}\n".format(path) for path in receipt.explicit_paths
        ).encode("utf-8")
        hashes_data = "".join(
            "{}  {}\n".format(
                receipt.explicit_path_sha256[path], path
            )
            for path in receipt.explicit_paths
        ).encode("utf-8")
        values = (
            (names[0], paths_data),
            (names[1], hashes_data),
            (names[2], b"committed\n"),
        )
        for name, data in values:
            _write_regular_at(receipts_fd, name, data)
        os.fsync(receipts_fd)
    finally:
        os.close(receipts_fd)


def _validate_sync_receipt_targets_at(root_fd, nonce):
    try:
        os.mkdir("receipts", 0o700, dir_fd=root_fd)
    except FileExistsError:
        pass
    try:
        receipts_fd = os.open(
            "receipts", _DIRECTORY_FLAGS, dir_fd=root_fd
        )
    except OSError as exc:
        raise TransactionError(
            "cannot open detached receipt directory: {}".format(exc)
        )
    try:
        for name in _sync_receipt_names(nonce):
            try:
                info = os.stat(
                    name, dir_fd=receipts_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode):
                raise TransactionError(
                    "detached receipt is not a regular file: {}".format(name)
                )
    finally:
        os.close(receipts_fd)


def _open_or_create_directory_at(parent_fd, name, mode=0o700):
    try:
        os.mkdir(name, mode, dir_fd=parent_fd)
    except FileExistsError:
        pass
    directory_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    os.fchmod(directory_fd, mode)
    return directory_fd


def _cleanup_nonce_at(root_fd, nonce):
    transactions_fd = None
    try:
        transactions_fd = os.open(
            ".transactions", _DIRECTORY_FLAGS, dir_fd=root_fd
        )
        _remove_tree_at(transactions_fd, nonce)
    except BaseException:
        pass
    finally:
        if transactions_fd is not None:
            _close_fd_best_effort(transactions_fd)


def _read_expected_committed_at(
    root_fd,
    *,
    nonce,
    operation,
    source_head,
    explicit_paths
):
    try:
        source_fd = os.open("source", _DIRECTORY_FLAGS, dir_fd=root_fd)
    except FileNotFoundError as exc:
        raise _GenerationNotCommitted(
            "matching committed source is unavailable: {}".format(exc)
        )
    except OSError as exc:
        raise TransactionError(
            "cannot open committed source without following symlinks: "
            "{}".format(exc)
        )
    try:
        receipt = _read_generation_fd(source_fd)
    finally:
        os.close(source_fd)
    if (
        receipt.nonce != nonce
        or receipt.operation != operation
        or receipt.source_head != source_head
        or receipt.explicit_paths != tuple(explicit_paths)
    ):
        raise _GenerationNotCommitted(
            "source is not the expected committed generation"
        )
    return receipt


def confirm_committed_generation(
    remote_root,
    *,
    nonce,
    operation,
    source_head,
    explicit_paths
):
    paths = _validate_explicit_paths(explicit_paths)
    head = _validate_source_head(source_head)
    nonce = _validate_nonce(nonce)
    if operation not in ("init", "sync"):
        raise TransactionError("confirmation operation is invalid")
    with _transaction_signal_handlers():
        with locked_remote_root(remote_root) as root_fd:
            receipt = _read_expected_committed_at(
                root_fd,
                nonce=nonce,
                operation=operation,
                source_head=head,
                explicit_paths=paths,
            )
            if operation == "init":
                _materialize_initial_receipts_at(root_fd, receipt)
            else:
                _materialize_sync_receipts_at(root_fd, receipt)
            _cleanup_nonce_at(root_fd, nonce)
            return receipt


def commit_sync_generation(
    remote_root,
    delta_root,
    *,
    nonce,
    source_head,
    explicit_paths,
    fault_injector=None,
    _committed_result=None,
    _manage_signal_handlers=True,
    _retain_committed_signals=False,
):
    paths = _validate_explicit_paths(explicit_paths)
    head = _validate_source_head(source_head)
    nonce = _validate_nonce(nonce)
    if not isinstance(delta_root, _StagedDelta):
        raise TransactionError(
            "sync delta must be created by streamed staging"
        )
    staged = delta_root
    if staged.closed:
        raise TransactionError("sync delta staging is already closed")
    if staged.nonce != nonce or staged.explicit_paths != paths:
        raise TransactionError(
            "sync delta identity does not match nonce and explicit paths"
        )

    committed_result = (
        [None] if _committed_result is None else _committed_result
    )
    signal_context = (
        _transaction_signal_handlers(committed_result)
        if _manage_signal_handlers
        else contextlib.nullcontext()
    )
    with signal_context:
        with locked_remote_root(remote_root) as root_fd:
            _inject(fault_injector, "after_lock")
            source_fd = None
            transactions_fd = None
            nonce_fd = None
            generation_fd = None
            delta_fd = None
            generation_info = None
            source_info = None
            try:
                source_fd = os.open(
                    "source", _DIRECTORY_FLAGS, dir_fd=root_fd
                )
                current = _read_generation_fd(source_fd)
                if current.nonce == nonce:
                    expected = _read_generation_fd(
                        source_fd,
                        expected_nonce=nonce,
                        expected_operation="sync",
                        expected_head=head,
                        expected_paths=paths,
                    )
                    _materialize_sync_receipts_at(root_fd, expected)
                    try:
                        _cleanup_nonce_at(root_fd, nonce)
                    except BaseException:
                        pass
                    with _blocked_transaction_signals(
                        committed_result
                        if _retain_committed_signals
                        else None
                    ):
                        committed_result[0] = expected
                    return expected

                source_info = os.fstat(source_fd)
                transactions_fd = os.open(
                    ".transactions", _DIRECTORY_FLAGS, dir_fd=root_fd
                )
                if not _same_file_identity(
                    os.fstat(root_fd), staged.root_info
                ):
                    raise TransactionError(
                        "remote root changed after delta staging"
                    )
                if not _same_file_identity(
                    os.fstat(transactions_fd),
                    staged.transactions_info,
                ):
                    raise TransactionError(
                        "transactions directory changed after delta staging"
                    )
                if not _same_file_identity(
                    os.fstat(staged.transactions_fd),
                    staged.transactions_info,
                ):
                    raise TransactionError(
                        "held transactions directory identity changed"
                    )
                if not _same_file_identity(
                    os.fstat(staged.delta_fd), staged.delta_info
                ):
                    raise TransactionError(
                        "held sync delta identity changed"
                    )
                try:
                    named_delta = os.stat(
                        staged.name,
                        dir_fd=transactions_fd,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise TransactionError(
                        "sync delta entry changed after staging: {}".format(
                            exc
                        )
                    )
                if (
                    not stat.S_ISDIR(named_delta.st_mode)
                    or not _same_file_identity(
                        named_delta, staged.delta_info
                    )
                ):
                    raise TransactionError(
                        "sync delta entry changed after staging"
                    )
                delta_fd = os.dup(staged.delta_fd)
                try:
                    os.mkdir(nonce, 0o700, dir_fd=transactions_fd)
                except FileExistsError as exc:
                    raise TransactionError(
                        "sync transaction nonce already exists"
                    ) from exc
                nonce_fd = os.open(
                    nonce, _DIRECTORY_FLAGS, dir_fd=transactions_fd
                )
                os.mkdir("generation", 0o700, dir_fd=nonce_fd)
                generation_fd = os.open(
                    "generation", _DIRECTORY_FLAGS, dir_fd=nonce_fd
                )
                _clone_tree(source_fd, generation_fd)
                _inject(fault_injector, "after_generation_ready")
                _apply_explicit_delta(generation_fd, delta_fd, paths)
                receipt = _generation_receipt(
                    generation_fd,
                    operation="sync",
                    nonce=nonce,
                    source_head=head,
                    explicit_paths=paths,
                )
                write_embedded_receipt(generation_fd, receipt)
                _inject(fault_injector, "after_embedded_receipt")
                _read_generation_fd(
                    generation_fd,
                    expected_nonce=nonce,
                    expected_operation="sync",
                    expected_head=head,
                    expected_paths=paths,
                )
                _validate_sync_receipt_targets_at(root_fd, nonce)
                _inject(fault_injector, "before_exchange")
                named_source = os.stat(
                    "source", dir_fd=root_fd, follow_symlinks=False
                )
                if (
                    not stat.S_ISDIR(named_source.st_mode)
                    or not _same_file_identity(named_source, source_info)
                ):
                    raise TransactionError(
                        "source entry changed before sync exchange"
                    )
                generation_info = os.fstat(generation_fd)
                named_generation = os.stat(
                    "generation",
                    dir_fd=nonce_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(named_generation.st_mode)
                    or not _same_file_identity(
                        named_generation, generation_info
                    )
                ):
                    raise TransactionError(
                        "generation entry changed before sync exchange"
                    )
                with _blocked_transaction_signals(
                    committed_result
                    if _retain_committed_signals
                    else None
                ):
                    _rename_exchange_at(
                        root_fd, "source", nonce_fd, "generation"
                    )
                    committed_result[0] = receipt
                try:
                    _inject(fault_injector, "after_exchange")
                except BaseException:
                    pass
                try:
                    committed = _read_expected_committed_at(
                        root_fd,
                        nonce=nonce,
                        operation="sync",
                        source_head=head,
                        explicit_paths=paths,
                    )
                    _require_exact_receipt(committed, receipt)
                except BaseException:
                    pass
                try:
                    _inject(fault_injector, "before_external_receipts")
                except BaseException:
                    pass
                try:
                    _materialize_sync_receipts_at(root_fd, receipt)
                except BaseException:
                    pass
                try:
                    _inject(fault_injector, "after_external_receipts")
                except BaseException:
                    pass
                try:
                    _inject(
                        fault_injector,
                        "before_old_generation_cleanup",
                    )
                except BaseException:
                    pass
                return receipt
            finally:
                if generation_fd is not None:
                    _close_fd_best_effort(generation_fd)
                if delta_fd is not None:
                    _close_fd_best_effort(delta_fd)
                if source_fd is not None:
                    _close_fd_best_effort(source_fd)
                if nonce_fd is not None:
                    try:
                        _remove_tree_at(nonce_fd, "generation")
                    except BaseException:
                        pass
                    _close_fd_best_effort(nonce_fd)
                if transactions_fd is not None:
                    try:
                        os.rmdir(nonce, dir_fd=transactions_fd)
                    except BaseException:
                        pass
                    _close_fd_best_effort(transactions_fd)
    return committed_result[0]


def _create_delta_parent(delta_fd, relative_path):
    parts = _split_relative_path(relative_path)
    parent_fd = os.dup(delta_fd)
    try:
        for part in parts[:-1]:
            next_fd = _open_or_create_directory_at(parent_fd, part)
            os.close(parent_fd)
            parent_fd = next_fd
        return parent_fd, parts[-1]
    except BaseException:
        os.close(parent_fd)
        raise


def _private_delta_name(nonce):
    return ".delta-{}-{}-{}".format(
        nonce, os.getpid(), time.time_ns()
    )


def _stage_delta_stream(remote_root, nonce, explicit_paths, stream):
    paths = _validate_explicit_paths(explicit_paths)
    nonce = _validate_nonce(nonce)
    delta_name = _private_delta_name(nonce)
    root_fd = open_directory_no_follow(Path(remote_root))
    transactions_fd = None
    delta_fd = None
    created_delta = False
    try:
        transactions_fd = _open_or_create_directory_at(
            root_fd, ".transactions"
        )
        try:
            os.mkdir(delta_name, 0o700, dir_fd=transactions_fd)
        except FileExistsError as exc:
            raise TransactionError(
                "sync delta private name already exists"
            ) from exc
        created_delta = True
        delta_fd = os.open(
            delta_name, _DIRECTORY_FLAGS, dir_fd=transactions_fd
        )
        expected = set(paths)
        seen = set()
        with tarfile.open(fileobj=stream, mode="r|*") as archive:
            for member in archive:
                member_path = PurePosixPath(member.name)
                normalized = member_path.as_posix()
                if (
                    member_path.is_absolute()
                    or ".." in member_path.parts
                    or member.name != normalized
                    or normalized not in expected
                    or not member.isfile()
                    or normalized in seen
                ):
                    raise TransactionError(
                        "delta archive contains unexpected member: {}".format(
                            member.name
                        )
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise TransactionError(
                        "delta archive member has no file data"
                    )
                parent_fd, name = _create_delta_parent(
                    delta_fd, normalized
                )
                output_fd = None
                try:
                    output_fd = os.open(
                        name,
                        _REGULAR_WRITE_FLAGS,
                        member.mode & 0o777,
                        dir_fd=parent_fd,
                    )
                    while True:
                        chunk = extracted.read(1024 * 1024)
                        if not chunk:
                            break
                        view = memoryview(chunk)
                        written = 0
                        while written < len(view):
                            written += os.write(
                                output_fd, view[written:]
                            )
                    os.fsync(output_fd)
                finally:
                    extracted.close()
                    if output_fd is not None:
                        os.close(output_fd)
                    os.close(parent_fd)
                seen.add(normalized)
        if seen != expected:
            raise TransactionError("delta archive is missing explicit paths")
        staged = _StagedDelta(
            remote_root=remote_root,
            nonce=nonce,
            explicit_paths=paths,
            root_info=os.fstat(root_fd),
            transactions_fd=transactions_fd,
            transactions_info=os.fstat(transactions_fd),
            name=delta_name,
            delta_fd=delta_fd,
            delta_info=os.fstat(delta_fd),
        )
        transactions_fd = None
        delta_fd = None
        return staged
    except BaseException:
        if transactions_fd is not None and created_delta:
            try:
                _remove_tree_at(transactions_fd, delta_name)
            except BaseException:
                pass
        raise
    finally:
        if delta_fd is not None:
            _close_fd_best_effort(delta_fd)
        if transactions_fd is not None:
            _close_fd_best_effort(transactions_fd)
        _close_fd_best_effort(root_fd)


def _cleanup_staged_delta(staged):
    if isinstance(staged, _StagedDelta):
        staged.close()


def _status_committed_generation(remote_root):
    with _transaction_signal_handlers():
        with locked_remote_root(remote_root) as root_fd:
            source_fd = os.open(
                "source", _DIRECTORY_FLAGS, dir_fd=root_fd
            )
            try:
                receipt = _read_generation_fd(source_fd)
            finally:
                os.close(source_fd)
            if receipt.operation == "init":
                _materialize_initial_receipts_at(root_fd, receipt)
                latest = "none"
            else:
                _materialize_sync_receipts_at(root_fd, receipt)
                latest = _sync_receipt_path(
                    remote_root, receipt.nonce
                )
            return receipt, latest


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_commit = subparsers.add_parser("init-commit")
    init_commit.add_argument("--remote-root", required=True)
    init_commit.add_argument("--generation", required=True)
    init_commit.add_argument("--source-head", required=True)
    sync_commit = subparsers.add_parser("sync-commit")
    sync_commit.add_argument("--remote-root", required=True)
    sync_commit.add_argument("--nonce", required=True)
    sync_commit.add_argument("--source-head", required=True)
    sync_commit.add_argument("--path", action="append", required=True)
    confirm = subparsers.add_parser("confirm")
    confirm.add_argument("--remote-root", required=True)
    confirm.add_argument("--nonce", required=True)
    confirm.add_argument("--operation", choices=("init", "sync"), required=True)
    confirm.add_argument("--source-head", required=True)
    confirm.add_argument("--path", action="append", default=[])
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--remote-root", required=True)
    return parser


def main(argv=None):
    retain_committed_signals = argv is None
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "init-commit":
            remote_root = Path(arguments.remote_root)
            committed_result = [None]
            with _transaction_signal_handlers(
                committed_result,
                retain_committed_signals=retain_committed_signals,
            ):
                receipt = commit_initial_generation(
                    remote_root,
                    arguments.generation,
                    arguments.source_head,
                    _committed_result=committed_result,
                    _manage_signal_handlers=False,
                    _retain_committed_signals=(
                        retain_committed_signals
                    ),
                )
                output = "{}\n".format(receipt.source_file_count)
            sys.stdout.write(output)
            sys.stdout.flush()
            return 0
        elif arguments.command == "sync-commit":
            remote_root = Path(arguments.remote_root)
            committed_result = [None]
            with _transaction_signal_handlers(
                committed_result,
                retain_committed_signals=retain_committed_signals,
            ):
                staged = None
                try:
                    receipt = confirm_committed_generation(
                        remote_root,
                        nonce=arguments.nonce,
                        operation="sync",
                        source_head=arguments.source_head,
                        explicit_paths=arguments.path,
                    )
                    with _blocked_transaction_signals(
                        committed_result
                        if retain_committed_signals
                        else None
                    ):
                        committed_result[0] = receipt
                except TransactionInterrupted:
                    raise
                except _GenerationNotCommitted:
                    staged = _stage_delta_stream(
                        remote_root,
                        arguments.nonce,
                        arguments.path,
                        sys.stdin.buffer,
                    )
                    try:
                        receipt = commit_sync_generation(
                            remote_root,
                            staged,
                            nonce=arguments.nonce,
                            source_head=arguments.source_head,
                            explicit_paths=arguments.path,
                            _committed_result=committed_result,
                            _manage_signal_handlers=False,
                            _retain_committed_signals=(
                                retain_committed_signals
                            ),
                        )
                    finally:
                        _cleanup_staged_delta(staged)
                output = _sync_receipt_path(
                    remote_root, receipt.nonce
                ) + "\n"
            sys.stdout.write(output)
            sys.stdout.flush()
            return 0
        elif arguments.command == "confirm":
            receipt = confirm_committed_generation(
                Path(arguments.remote_root),
                nonce=arguments.nonce,
                operation=arguments.operation,
                source_head=arguments.source_head,
                explicit_paths=arguments.path,
            )
            if receipt.operation == "sync":
                output = _sync_receipt_path(
                    arguments.remote_root, receipt.nonce
                ) + "\n"
            else:
                output = "{}\n".format(receipt.source_file_count)
        else:
            receipt, latest = _status_committed_generation(
                Path(arguments.remote_root)
            )
            output = (
                "head={}\ncount={}\nlatest={}\n".format(
                    receipt.source_head,
                    receipt.source_file_count,
                    latest,
                )
            )
    except TransactionInterrupted as exc:
        sys.stderr.write("{}\n".format(exc))
        return 128 + int(exc.signum)
    except (OSError, TransactionError) as exc:
        sys.stderr.write("{}\n".format(exc))
        return 1
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
