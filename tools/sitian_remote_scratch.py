from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import fnmatch
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import stat
import subprocess
import sys
import threading
import time
from typing import Callable, Mapping, Optional, Sequence


REMOTE_HOST = "sitian@10.232.195.203"
REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
KRB5_CACHE = "FILE:/Users/bytedance/krb5cc_sitian"
ATTEMPTS = 5
LOCAL_REPO_ROOT = Path("/Users/bytedance/dev/TinyLLMForge")
REMOTE_SOURCE_ROOT = Path(REMOTE_ROOT) / "source"
REMOTE_TASK1_TEST_ROOT = Path(REMOTE_ROOT) / "red-task1"
REMOTE_TASK2_TEST_ROOT = Path(REMOTE_ROOT) / "task2-red-c1bd1ae"
APPROVED_REPO_ROOTS = frozenset(
    {
        LOCAL_REPO_ROOT,
        REMOTE_SOURCE_ROOT,
        REMOTE_TASK1_TEST_ROOT,
        REMOTE_TASK2_TEST_ROOT,
    }
)
FORBIDDEN_ALWAYS_DIRS = (
    ".git",
    "artifacts",
    "experiments",
)
FORBIDDEN_CACHE_DIRS = (
    ".cache",
    "cache",
    "caches",
    "__pycache__",
    ".pytest_cache",
)
FORBIDDEN_LOG_DIRS = (
    "log",
    "logs",
)
FORBIDDEN_RAW_DIRS = (
    "raw-output",
    "raw_output",
    "rawoutput",
)
FORBIDDEN_ARCHIVE_PATTERNS = (
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
)
FORBIDDEN_PARTS = frozenset(
    FORBIDDEN_ALWAYS_DIRS
    + FORBIDDEN_CACHE_DIRS
    + FORBIDDEN_LOG_DIRS
    + FORBIDDEN_RAW_DIRS
)
FORBIDDEN_FILE_PATTERNS = (
    "._*",
    "*.pyc",
    "*.log",
    "*.pid",
    "*.out",
    *FORBIDDEN_ARCHIVE_PATTERNS,
    "*review-package.diff",
)
INITIAL_SNAPSHOT_EXCLUDES = tuple(
    pattern
    for name in sorted(FORBIDDEN_PARTS)
    for pattern in (name, f"{name}/*", f"*/{name}", f"*/{name}/*")
) + FORBIDDEN_FILE_PATTERNS


@dataclass(frozen=True)
class ScratchConfig:
    repo_root: Path
    remote_host: str = field(default=REMOTE_HOST, init=False)
    remote_root: str = field(default=REMOTE_ROOT, init=False)
    krb5_cache: str = field(default=KRB5_CACHE, init=False)
    attempts: int = field(default=ATTEMPTS, init=False)

    @classmethod
    def default(cls, repo_root: Path) -> "ScratchConfig":
        resolved = repo_root.resolve()
        if resolved not in APPROVED_REPO_ROOTS:
            raise ValueError(
                "repo root must resolve to an approved local or remote "
                "source root"
            )
        return cls(repo_root=repo_root)


def remote_layout(config: ScratchConfig) -> dict[str, str]:
    return {
        name: f"{config.remote_root}/{name}"
        for name in (
            "source",
            "tmp",
            "pycache",
            "cache",
            "logs",
            "receipts",
            "env",
        )
    }


def _is_forbidden_path(path: PurePosixPath) -> bool:
    return (
        any(part in FORBIDDEN_PARTS for part in path.parts)
        or any(
            fnmatch.fnmatchcase(path.name, pattern)
            for pattern in FORBIDDEN_FILE_PATTERNS
        )
    )


def validate_relative_paths(
    paths: Sequence[str],
    *,
    repo_root: Optional[Path] = None,
) -> tuple[str, ...]:
    if not paths:
        raise ValueError("at least one explicit path is required")
    root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else repo_root
    ).resolve()
    ScratchConfig.default(root)
    normalized = []
    for raw_path in paths:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("paths must be non-empty strings")
        if any(character in raw_path for character in ("\0", "\n", "\r")):
            raise ValueError("paths must not contain control characters")
        path = PurePosixPath(raw_path)
        if (
            path == PurePosixPath(".")
            or path.is_absolute()
            or ".." in path.parts
            or path.parts[0].startswith("-")
        ):
            raise ValueError(f"path is not repository-relative: {raw_path}")
        if _is_forbidden_path(path):
            raise ValueError(f"path is forbidden: {raw_path}")
        candidate = root
        final_mode = None
        try:
            for index, part in enumerate(path.parts):
                candidate = candidate / part
                mode = candidate.lstat().st_mode
                if stat.S_ISLNK(mode):
                    raise ValueError(f"path contains symlink: {raw_path}")
                if index < len(path.parts) - 1 and not stat.S_ISDIR(mode):
                    raise ValueError(
                        f"path is not a repository file: {raw_path}"
                    )
                final_mode = mode
        except FileNotFoundError as exc:
            raise ValueError(
                f"path is not a repository file: {raw_path}"
            ) from exc
        if final_mode is None or not stat.S_ISREG(final_mode):
            raise ValueError(f"path is not a repository file: {raw_path}")
        normalized.append(path.as_posix())
    return tuple(dict.fromkeys(normalized))


def incremental_tar_argv(
    paths: Sequence[str],
    *,
    repo_root: Optional[Path] = None,
) -> tuple[str, ...]:
    checked = validate_relative_paths(paths, repo_root=repo_root)
    return (
        "tar",
        "--no-xattrs",
        "--no-mac-metadata",
        "-cf",
        "-",
        "--",
        *checked,
    )


def ssh_argv(config: ScratchConfig) -> tuple[str, ...]:
    return (
        "ssh",
        "-o",
        "ProxyCommand=nc -x 127.0.0.1:63445 -X 5 -w 10 %h %p",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ConnectionAttempts=1",
        config.remote_host,
    )


def remote_cache_environment(
    config: ScratchConfig,
) -> dict[str, str]:
    layout = remote_layout(config)
    return {
        "TMPDIR": layout["tmp"],
        "TMP": layout["tmp"],
        "TEMP": layout["tmp"],
        "PYTHONPYCACHEPREFIX": layout["pycache"],
        "XDG_CACHE_HOME": layout["cache"],
        "PYTHONDONTWRITEBYTECODE": "0",
    }


def run_with_retries(
    argv: Sequence[str],
    *,
    attempts: int,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs,
) -> subprocess.CompletedProcess[str]:
    if attempts < 1:
        raise ValueError("attempts must be at least one")
    last = None
    for attempt in range(1, attempts + 1):
        last = runner(argv, text=True, **kwargs)
        if last.returncode == 0:
            return last
        if attempt < attempts:
            sleep(2.0)
    assert last is not None
    return last


def _shell_find_names(patterns: Sequence[str]) -> str:
    return "\\( " + " -o ".join(
        f"-name {shlex.quote(pattern)}" for pattern in patterns
    ) + " \\)"


def _forbidden_verification_checks(find_root: str = "source") -> str:
    cache_names = _shell_find_names(FORBIDDEN_CACHE_DIRS)
    log_names = _shell_find_names(FORBIDDEN_LOG_DIRS)
    raw_names = _shell_find_names(FORBIDDEN_RAW_DIRS)
    archive_names = _shell_find_names(FORBIDDEN_ARCHIVE_PATTERNS)
    return (
        f"test \"$(find {find_root} -name "
        f"{shlex.quote(FORBIDDEN_ALWAYS_DIRS[0])} | wc -l)\" -eq 0; "
        f"test \"$(find {find_root} -type d -name "
        f"{shlex.quote(FORBIDDEN_ALWAYS_DIRS[1])} | wc -l)\" "
        "-eq 0; "
        f"test \"$(find {find_root} -type d -name "
        f"{shlex.quote(FORBIDDEN_ALWAYS_DIRS[2])} | wc -l)\" "
        "-eq 0; "
        f"test \"$(find {find_root} -type d {cache_names} | wc -l)\" "
        "-eq 0; "
        f"test \"$(find {find_root} -type f -name '*.pyc' | wc -l)\" "
        "-eq 0; "
        f"test \"$(find {find_root} \\( -type d {log_names} -o "
        "-type f -name '*.log' \\) | wc -l)\" -eq 0; "
        f"test \"$(find {find_root} -type f -name '*.pid' | wc -l)\" "
        "-eq 0; "
        f"test \"$(find {find_root} \\( -type d {raw_names} -o "
        "-type f -name '*.out' "
        "\\) | wc -l)\" -eq 0; "
        f"test \"$(find {find_root} -type f {archive_names} | wc -l)\" "
        "-eq 0; "
        f"test \"$(find {find_root} -name '*review-package.diff' | wc -l)\" "
        "-eq 0; "
        f"test \"$(find {find_root} -name '._*' | wc -l)\" -eq 0"
    )


def initial_snapshot_commands(
    config: ScratchConfig,
) -> dict[str, tuple[str, ...] | str]:
    nonce = f"{os.getpid()}-{time.time_ns()}"
    generation_name = f".transactions/{nonce}/generation"
    generation = f"{config.remote_root}/{generation_name}"
    excludes = " ".join(
        f"--exclude={shlex.quote(pattern)}"
        for pattern in INITIAL_SNAPSHOT_EXCLUDES
    )
    layout = remote_layout(config)
    return {
        "archive": ("git", "archive", "--format=tar", "HEAD"),
        "stage": generation,
        "remote_extract": (
            "set -eu; "
            f"generation={shlex.quote(generation)}; "
            "mkdir -p \"$generation\"; "
            f"tar {excludes} -xf - -C \"$generation\""
        ),
        "remote_commit": (
            "set -eu; "
            f"root={shlex.quote(config.remote_root)}; "
            f"generation={shlex.quote(generation)}; "
            f"mkdir -p {shlex.quote(layout['tmp'])} "
            f"{shlex.quote(layout['pycache'])} "
            f"{shlex.quote(layout['cache'])}; "
            f"TMPDIR={shlex.quote(layout['tmp'])} "
            f"TMP={shlex.quote(layout['tmp'])} "
            f"TEMP={shlex.quote(layout['tmp'])} "
            f"PYTHONPYCACHEPREFIX={shlex.quote(layout['pycache'])} "
            f"XDG_CACHE_HOME={shlex.quote(layout['cache'])} "
            "PYTHONDONTWRITEBYTECODE=1 "
            "python3 "
            "\"$generation/tools/sitian_remote_transaction.py\" "
            "init-commit "
            "--remote-root \"$root\" "
            f"--generation {shlex.quote(generation_name)} "
            "--source-head __SOURCE_HEAD__"
        ),
    }


def incremental_sync_commands(
    config: ScratchConfig,
    paths: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    checked = validate_relative_paths(paths, repo_root=config.repo_root)
    return {
        "tar": incremental_tar_argv(
            checked,
            repo_root=config.repo_root,
        ),
        "ssh": (
            *ssh_argv(config),
            f"tar -xf - -C {shlex.quote(config.remote_root + '/source')}",
        ),
    }


def _command_environment(config: ScratchConfig) -> dict[str, str]:
    environment = os.environ.copy()
    environment["KRB5CCNAME"] = config.krb5_cache
    return environment


def _remote_command(
    config: ScratchConfig,
    command: str,
) -> subprocess.CompletedProcess[str]:
    return run_with_retries(
        (*ssh_argv(config), command),
        attempts=config.attempts,
        capture_output=True,
        env=_command_environment(config),
    )


def _decode_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _close_pipe(pipe: object | None) -> None:
    if pipe is None:
        return
    try:
        pipe.close()
    except OSError:
        pass


def _reap_process(
    process: subprocess.Popen[bytes] | None,
    *,
    timeout: float = 1.0,
) -> None:
    if process is None:
        return
    try:
        running = process.poll() is None
    except OSError:
        running = True
    if running:
        try:
            process.terminate()
        except (OSError, ProcessLookupError):
            pass
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    except (OSError, ProcessLookupError):
        return
    try:
        process.kill()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=timeout)
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
        pass


def _stream_with_retries(
    producer_argv: Sequence[str],
    consumer_argv: Sequence[str],
    *,
    config: ScratchConfig,
    attempts: Optional[int] = None,
) -> subprocess.CompletedProcess[str]:
    environment = _command_environment(config)
    attempt_limit = config.attempts if attempts is None else attempts
    if attempt_limit < 1:
        raise ValueError("attempts must be at least one")
    last = None
    for attempt in range(1, attempt_limit + 1):
        producer = None
        consumer = None
        stderr_thread = None
        stderr_thread_started = False
        producer_stderr_chunks = []
        try:
            producer = subprocess.Popen(
                producer_argv,
                cwd=config.repo_root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert producer.stdout is not None
            consumer = subprocess.Popen(
                consumer_argv,
                env=environment,
                stdin=producer.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            producer.stdout.close()

            def drain_producer_stderr() -> None:
                if producer is not None and producer.stderr is not None:
                    producer_stderr_chunks.append(producer.stderr.read())

            stderr_thread = threading.Thread(
                target=drain_producer_stderr,
                daemon=True,
            )
            stderr_thread.start()
            stderr_thread_started = True
            consumer_stdout, consumer_stderr = consumer.communicate()
            producer_returncode = producer.wait()
            stderr_thread.join()
            producer_stderr = (
                b""
                if not producer_stderr_chunks
                else producer_stderr_chunks[0]
            )
            returncode = (
                producer_returncode
                if producer_returncode != 0
                else consumer.returncode
            )
            stderr = _decode_output(producer_stderr)
            stderr += _decode_output(consumer_stderr)
            last = subprocess.CompletedProcess(
                args=tuple(consumer_argv),
                returncode=returncode,
                stdout=_decode_output(consumer_stdout),
                stderr=stderr,
            )
        finally:
            _close_pipe(None if producer is None else producer.stdout)
            _reap_process(consumer)
            _reap_process(producer)
            if stderr_thread is not None and stderr_thread_started:
                stderr_thread.join(timeout=1.0)
            _close_pipe(None if producer is None else producer.stderr)
            _close_pipe(None if consumer is None else consumer.stdout)
            _close_pipe(None if consumer is None else consumer.stderr)
        if last.returncode == 0:
            return last
        if attempt < attempt_limit:
            time.sleep(2.0)
    assert last is not None
    return last


def _incremental_remote_command(
    config: ScratchConfig,
    paths: Sequence[str],
    *,
    nonce: str,
) -> tuple[str, str]:
    root = config.remote_root
    source = f"{root}/source"
    receipts = f"{root}/receipts"
    backup = f"{root}/.incoming-sync-{nonce}"
    path_receipt = f"{receipts}/sync-{nonce}.paths.txt"
    hash_receipt = f"{receipts}/sync-{nonce}.sha256"
    state_receipt = f"{receipts}/sync-{nonce}.state"
    lock = f"{root}/.sync-transaction-lock"
    backup_commands = []
    rollback_commands = []
    verify_commands = []
    apply_commands = []
    topology_commands = []
    checked_parents = set()
    for path in paths:
        quoted_path = shlex.quote(path)
        posix_path = PurePosixPath(path)
        parent = posix_path.parent.as_posix()
        quoted_parent = shlex.quote(parent)
        parent_parts = posix_path.parts[:-1]
        for index in range(1, len(parent_parts) + 1):
            parent_path = PurePosixPath(*parent_parts[:index]).as_posix()
            if parent_path in checked_parents:
                continue
            checked_parents.add(parent_path)
            quoted_parent_path = shlex.quote(parent_path)
            topology_commands.append(
                f"if test -e \"$source\"/{quoted_parent_path} || "
                f"test -L \"$source\"/{quoted_parent_path}; then "
                f"if test -L \"$source\"/{quoted_parent_path} || "
                f"test ! -d \"$source\"/{quoted_parent_path}; then "
                "printf '%s\\n' "
                f"{shlex.quote('incremental sync requires full init: unsafe remote parent ' + parent_path)} "
                ">&2; exit 76; fi; "
                "else printf '%s\\n' "
                f"{shlex.quote('incremental sync requires full init: missing remote parent ' + parent_path)} "
                ">&2; exit 76; fi"
            )
        topology_commands.append(
            f"if test -e \"$source\"/{quoted_path} || "
            f"test -L \"$source\"/{quoted_path}; then "
            f"if test -L \"$source\"/{quoted_path} || "
            f"test ! -f \"$source\"/{quoted_path}; then "
            "printf '%s\\n' "
            f"{shlex.quote('incremental sync requires full init: unsafe remote final ' + path)} "
            ">&2; exit 76; fi; fi"
        )
        temporary_name = f".sync-{nonce}-{posix_path.name}"
        temporary_path = (
            posix_path.parent / temporary_name
        ).as_posix()
        quoted_temporary_path = shlex.quote(temporary_path)
        backup_commands.append(
            "mkdir -p \"$backup/original\"/"
            f"{quoted_parent}; "
            f"if test -e \"$source\"/{quoted_path} || "
            f"test -L \"$source\"/{quoted_path}; then "
            f"test -f \"$source\"/{quoted_path}; "
            f"test ! -L \"$source\"/{quoted_path}; "
            f"cp -p \"$source\"/{quoted_path} "
            f"\"$backup/original\"/{quoted_path}; "
            f"printf '%s\\n' {quoted_path} >> \"$backup/existing\"; "
            "fi"
        )
        rollback_commands.append(
            f"if grep -Fqx -- {quoted_path} \"$backup/intended\"; then "
            f"if grep -Fqx -- {quoted_path} \"$backup/existing\"; then "
            f"cp -p \"$backup/original\"/{quoted_path} "
            f"\"$source\"/{quoted_path} 2>/dev/null || true; "
            f"else rm -f \"$source\"/{quoted_path}; fi; fi; "
            f"rm -f \"$source\"/{quoted_temporary_path}"
        )
        verify_commands.append(
            f"test -f \"$backup/incoming\"/{quoted_path}; "
            f"test ! -L \"$backup/incoming\"/{quoted_path}"
        )
        apply_commands.append(
            f"cp -p \"$backup/incoming\"/{quoted_path} "
            f"\"$source\"/{quoted_temporary_path}; "
            f"printf '%s\\n' {quoted_path} >> \"$backup/intended\"; "
            "checkpoint before_file_rename; "
            f"mv \"$source\"/{quoted_temporary_path} "
            f"\"$source\"/{quoted_path}; "
            "checkpoint after_file_rename"
        )
    path_lines = "".join(
        f"printf '%s\\n' {shlex.quote(path)}; " for path in paths
    )
    hash_operands = " ".join(shlex.quote(path) for path in paths)
    command = (
        "set -eu; "
        f"root={shlex.quote(root)}; source={shlex.quote(source)}; "
        f"receipts={shlex.quote(receipts)}; backup={shlex.quote(backup)}; "
        f"path_receipt={shlex.quote(path_receipt)}; "
        f"hash_receipt={shlex.quote(hash_receipt)}; "
        f"state_receipt={shlex.quote(state_receipt)}; "
        f"lock={shlex.quote(lock)}; nonce={shlex.quote(nonce)}; "
        "test -d \"$source\"; test ! -L \"$source\"; "
        "mkdir -p \"$receipts\"; "
        "if test -f \"$state_receipt\"; then "
        "test \"$(cat \"$state_receipt\")\" = committed; "
        "test -f \"$path_receipt\"; test -f \"$hash_receipt\"; "
        "printf '%s\\n' \"$hash_receipt\"; exit 0; fi; "
        "if test -e \"$backup\" || test -L \"$backup\" || "
        "test -e \"$path_receipt\" || test -L \"$path_receipt\" || "
        "test -e \"$hash_receipt\" || test -L \"$hash_receipt\" || "
        "test -e \"$state_receipt\" || test -L \"$state_receipt\"; then "
        "exit 75; fi; "
        "mkdir \"$backup\" || exit 75; "
        "mkdir -p \"$backup/incoming\" \"$backup/original\"; "
        ": > \"$backup/existing\"; : > \"$backup/intended\"; "
        "committed=0; lock_owned=0; "
        "checkpoint() { "
        "if test \"${SITIAN_SYNC_FAIL_POINT-}\" = \"$1\"; then "
        "kill -HUP \"$$\"; fi; "
        "}; "
        "transaction_committed() { "
        "test -f \"$state_receipt\" && test ! -L \"$state_receipt\" && "
        "test \"$(cat \"$state_receipt\")\" = committed && "
        "test -f \"$path_receipt\" && test ! -L \"$path_receipt\" && "
        "test -f \"$hash_receipt\" && test ! -L \"$hash_receipt\"; "
        "}; "
        "release_lock() { "
        "if test \"$lock_owned\" -eq 0; then return 0; fi; "
        "released_lock=\"$backup/released-lock\"; "
        "if test -f \"$lock/owner\" && test ! -L \"$lock/owner\" && "
        "test \"$(cat \"$lock/owner\")\" = \"$nonce\"; then "
        "rm -rf \"$released_lock\"; "
        "if mv \"$lock\" \"$released_lock\"; then "
        "lock_owned=0; return 0; fi; "
        "elif test -f \"$released_lock/owner\" && "
        "test ! -L \"$released_lock/owner\" && "
        "test \"$(cat \"$released_lock/owner\")\" = \"$nonce\"; then "
        "lock_owned=0; return 0; "
        "fi; "
        "return 1; "
        "}; "
        "cleanup() { "
        "status=$?; trap - EXIT HUP INT TERM; "
        "if transaction_committed; then committed=1; fi; "
        "if test \"$committed\" -eq 0; then "
        + "; ".join(rollback_commands)
        + "; "
        "rm -f \"$path_receipt\" \"$hash_receipt\" \"$state_receipt\"; "
        "fi; "
        "release_lock || true; "
        "rm -rf \"$backup\"; "
        "exit \"$status\"; "
        "}; "
        "on_hup() { exit 129; }; "
        "on_int() { exit 130; }; "
        "on_term() { exit 143; }; "
        "enable_signal_traps() { "
        "trap on_hup HUP; trap on_int INT; trap on_term TERM; "
        "}; "
        "trap cleanup EXIT; "
        "enable_signal_traps; "
        "tar -xf - -C \"$backup/incoming\"; "
        + _forbidden_verification_checks('"$backup/incoming"')
        + "; "
        + "; ".join(verify_commands)
        + "; "
        "path_new=\"$backup/paths.txt\"; "
        "hash_new=\"$backup/files.sha256\"; "
        "state_new=\"$backup/state\"; "
        "{ "
        + path_lines
        + "} > \"$path_new\"; "
        f"(cd \"$backup/incoming\" && sha256sum -- {hash_operands}) "
        "> \"$hash_new\"; "
        "trap '' HUP INT TERM; "
        "if ! mkdir \"$lock\"; then "
        "enable_signal_traps; exit 75; fi; "
        "lock_owned=1; "
        "if ! printf '%s\\n' \"$nonce\" > \"$lock/owner\"; then "
        "rm -f \"$lock/owner\"; rmdir \"$lock\" 2>/dev/null || true; "
        "lock_owned=0; enable_signal_traps; exit 75; fi; "
        "enable_signal_traps; "
        + "; ".join(topology_commands)
        + "; "
        + "; ".join(backup_commands)
        + "; "
        + "; ".join(apply_commands)
        + "; "
        "mv \"$path_new\" \"$path_receipt\"; "
        "checkpoint after_path_receipt_publish; "
        "mv \"$hash_new\" \"$hash_receipt\"; "
        "checkpoint after_hash_receipt_publish; "
        "printf 'committed\\n' > \"$state_new\"; "
        "checkpoint before_state_receipt_publish; "
        "mv \"$state_new\" \"$state_receipt\"; "
        "checkpoint after_state_receipt_publish; committed=1; "
        "trap '' HUP INT TERM; "
        "release_lock; "
        "enable_signal_traps; "
        "rm -rf \"$backup\"; "
        "trap - EXIT HUP INT TERM; "
        "printf '%s\\n' \"$hash_receipt\""
    )
    return command, hash_receipt


def _incremental_commit_status_command(
    config: ScratchConfig,
    *,
    nonce: str,
) -> str:
    receipts = f"{config.remote_root}/receipts"
    path_receipt = f"{receipts}/sync-{nonce}.paths.txt"
    hash_receipt = f"{receipts}/sync-{nonce}.sha256"
    state_receipt = f"{receipts}/sync-{nonce}.state"
    return (
        "set -eu; "
        f"path_receipt={shlex.quote(path_receipt)}; "
        f"hash_receipt={shlex.quote(hash_receipt)}; "
        f"state_receipt={shlex.quote(state_receipt)}; "
        "attempt=0; "
        "while test \"$attempt\" -lt 100; do "
        "if test -f \"$state_receipt\"; then "
        "state=$(cat \"$state_receipt\"); "
        "if test \"$state\" = committed; then "
        "test -f \"$path_receipt\"; test -f \"$hash_receipt\"; "
        "printf '%s\\n' \"$hash_receipt\"; exit 0; fi; "
        "exit 1; fi; "
        "attempt=$((attempt + 1)); sleep 0.1; "
        "done; exit 1"
    )


def _initialize(config: ScratchConfig) -> tuple[str, int]:
    head_result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=config.repo_root,
        env=_command_environment(config),
        capture_output=True,
        text=True,
    )
    head = head_result.stdout.strip()
    if head_result.returncode != 0 or len(head) != 40:
        raise RuntimeError("unable to resolve local HEAD")
    commands = initial_snapshot_commands(config)
    stream_result = _stream_with_retries(
        commands["archive"],
        (*ssh_argv(config), commands["remote_extract"]),
        config=config,
    )
    if stream_result.returncode != 0:
        raise RuntimeError("initial source stream failed")
    remote_commit = commands["remote_commit"].replace(
        "__SOURCE_HEAD__", shlex.quote(head)
    )
    promote_result = _remote_command(
        config,
        remote_commit,
    )
    if promote_result.returncode != 0:
        raise RuntimeError("initial source promotion failed")
    count_text = promote_result.stdout.strip()
    if not count_text.isdigit():
        raise RuntimeError("invalid initial source receipt")
    return head, int(count_text)


def _sync(
    config: ScratchConfig,
    paths: Sequence[str],
) -> tuple[str, int]:
    commands = incremental_sync_commands(config, paths)
    checked = validate_relative_paths(paths, repo_root=config.repo_root)
    nonce = f"{int(time.time())}-{os.getpid()}-{time.time_ns()}"
    remote_command, expected_receipt = _incremental_remote_command(
        config,
        checked,
        nonce=nonce,
    )
    result = _stream_with_retries(
        commands["tar"],
        (*ssh_argv(config), remote_command),
        config=config,
        attempts=1,
    )
    receipt = result.stdout.strip()
    if result.returncode != 0 or receipt != expected_receipt:
        status = _remote_command(
            config,
            _incremental_commit_status_command(config, nonce=nonce),
        )
        receipt = status.stdout.strip()
        if status.returncode != 0 or receipt != expected_receipt:
            raise RuntimeError("incremental source sync failed")
    if receipt != expected_receipt:
        raise RuntimeError("invalid incremental sync receipt")
    return receipt, len(checked)


def _status(config: ScratchConfig) -> tuple[str, int, str]:
    root = shlex.quote(config.remote_root)
    command = (
        "set -eu; "
        f"root={root}; receipts=\"$root/receipts\"; "
        "head=missing; count=0; latest=none; "
        "if test -f \"$receipts/source-head.txt\"; then "
        "head=$(cat \"$receipts/source-head.txt\"); fi; "
        "if test -f \"$receipts/source-files.sha256\"; then "
        "count=$(wc -l < \"$receipts/source-files.sha256\"); fi; "
        "candidate=$(find \"$receipts\" -maxdepth 1 -type f "
        "-name 'sync-*.sha256' -printf '%f\\n' 2>/dev/null | "
        "LC_ALL=C sort | tail -n 1); "
        "if test -n \"$candidate\"; then latest=\"$receipts/$candidate\"; fi; "
        "printf 'head=%s\\ncount=%s\\nlatest=%s\\n' "
        "\"$head\" \"$count\" \"$latest\""
    )
    result = _remote_command(config, command)
    if result.returncode != 0:
        raise RuntimeError("remote scratch status failed")
    values = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {"head", "count", "latest"}:
            values[key] = value
    head = values.get("head", "")
    count = values.get("count", "")
    latest = values.get("latest", "")
    if head != "missing" and (
        len(head) != 40
        or any(character not in "0123456789abcdef" for character in head)
    ):
        raise RuntimeError("invalid remote source head")
    if not count.isdigit():
        raise RuntimeError("invalid remote source count")
    if latest != "none" and not latest.startswith(
        config.remote_root + "/receipts/sync-"
    ):
        raise RuntimeError("invalid remote sync receipt")
    return head, int(count), latest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init")
    sync_parser = subparsers.add_parser("sync")
    sync_parser.add_argument("paths", nargs="+")
    subparsers.add_parser("status")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    config = ScratchConfig.default(Path(__file__).resolve().parents[1])
    try:
        if arguments.command == "init":
            head, count = _initialize(config)
            print(f"source_head={head}")
            print(f"transferred_files={count}")
        elif arguments.command == "sync":
            receipt, count = _sync(config, arguments.paths)
            print(f"receipt={receipt}")
            print(f"transferred_files={count}")
        else:
            head, count, latest = _status(config)
            print(f"source_head={head}")
            print(f"source_file_count={count}")
            print(f"latest_sync_receipt={latest}")
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
