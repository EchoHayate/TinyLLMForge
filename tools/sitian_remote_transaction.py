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


class TransactionInterrupted(TransactionError):
    def __init__(self, signum):
        super().__init__(
            "transaction interrupted by signal {}".format(signum)
        )
        self.signum = signum


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
        if _is_forbidden(normalized_path):
            raise TransactionError(
                "explicit path is forbidden: {}".format(normalized_path)
            )
        normalized.append(normalized_path)
    if len(set(normalized)) != len(normalized):
        raise TransactionError("explicit paths contain duplicates")
    return tuple(normalized)


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_fd(fd):
    digest = hashlib.sha256()
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
                entries.append((relative_path, _sha256_fd(file_fd)))
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


@contextlib.contextmanager
def locked_remote_root(remote_root):
    root_fd = open_directory_no_follow(Path(remote_root))
    transactions_fd = None
    lock_fd = None
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
        yield root_fd
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if transactions_fd is not None:
            os.close(transactions_fd)
        os.close(root_fd)


def rename_exchange(parent_fd, left, right):
    if _RENAMEAT2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable")
    left_parts = _split_relative_path(left)
    right_parts = _split_relative_path(right)
    left_parent_fd = os.dup(parent_fd)
    right_parent_fd = os.dup(parent_fd)
    try:
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
        result = _RENAMEAT2(
            left_parent_fd,
            os.fsencode(left_parts[-1]),
            right_parent_fd,
            os.fsencode(right_parts[-1]),
            RENAME_EXCHANGE,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(
                error_number,
                os.strerror(error_number),
                "{} <-> {}".format(left, right),
            )
    finally:
        os.close(right_parent_fd)
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


def _remove_relative_tree(root_fd, relative_path):
    parts = _split_relative_path(relative_path)
    parent_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, _DIRECTORY_FLAGS, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        name = parts[-1]
        try:
            info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if stat.S_ISDIR(info.st_mode):
            directory_fd = os.open(
                name, _DIRECTORY_FLAGS, dir_fd=parent_fd
            )
            try:
                _remove_tree_contents(directory_fd)
            finally:
                os.close(directory_fd)
            os.rmdir(name, dir_fd=parent_fd)
        else:
            os.unlink(name, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)


def _remove_empty_nonce_directory(root_fd, generation_name):
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
            os.rmdir(parent_name, dir_fd=grandparent_fd)
        except OSError as exc:
            if exc.errno not in (errno.ENOENT, errno.ENOTEMPTY):
                raise
    finally:
        os.close(grandparent_fd)


@contextlib.contextmanager
def _transaction_signal_handlers():
    previous = {}
    signals = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
    for signum in signals:
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _raise_transaction_interrupted)
    try:
        yield
    finally:
        for signum in signals:
            signal.signal(signum, previous[signum])


def _inject(fault_injector, point):
    if fault_injector is not None:
        fault_injector(point)


def promote_generation(
    remote_root, generation_name, receipt, fault_injector=None
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
    if generation_parts[1] != receipt.nonce:
        raise TransactionError("generation nonce does not match receipt")

    with _transaction_signal_handlers():
        with locked_remote_root(remote_root) as root_fd:
            try:
                _inject(fault_injector, "after_lock")
                try:
                    generation_fd = _open_directory_at(
                        root_fd, generation_name
                    )
                except OSError as exc:
                    raise TransactionError(
                        "cannot open generation without following symlinks: "
                        "{}".format(exc)
                    )
                try:
                    _inject(fault_injector, "after_generation_ready")
                    write_embedded_receipt(generation_fd, receipt)
                    _inject(fault_injector, "after_embedded_receipt")
                    _read_generation_fd(
                        generation_fd,
                        expected_nonce=receipt.nonce,
                        expected_operation=receipt.operation,
                        expected_head=receipt.source_head,
                        expected_paths=receipt.explicit_paths,
                    )
                finally:
                    os.close(generation_fd)

                _inject(fault_injector, "before_exchange")
                try:
                    source_info = os.stat(
                        "source", dir_fd=root_fd, follow_symlinks=False
                    )
                except FileNotFoundError:
                    os.rename(
                        generation_name,
                        "source",
                        src_dir_fd=root_fd,
                        dst_dir_fd=root_fd,
                    )
                else:
                    if not stat.S_ISDIR(source_info.st_mode):
                        raise TransactionError(
                            "source is not a real directory"
                        )
                    source_fd = os.open(
                        "source", _DIRECTORY_FLAGS, dir_fd=root_fd
                    )
                    os.close(source_fd)
                    rename_exchange(root_fd, "source", generation_name)
                _inject(fault_injector, "after_exchange")
                committed_fd = os.open(
                    "source", _DIRECTORY_FLAGS, dir_fd=root_fd
                )
                try:
                    return _read_generation_fd(
                        committed_fd,
                        expected_nonce=receipt.nonce,
                        expected_operation=receipt.operation,
                        expected_head=receipt.source_head,
                        expected_paths=receipt.explicit_paths,
                    )
                finally:
                    os.close(committed_fd)
            finally:
                _remove_relative_tree(root_fd, generation_name)
                _remove_empty_nonce_directory(root_fd, generation_name)
