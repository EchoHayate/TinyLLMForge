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
    try:
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
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
