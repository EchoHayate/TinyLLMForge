from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
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
APPROVED_REPO_ROOTS = frozenset(
    {
        LOCAL_REPO_ROOT,
        REMOTE_SOURCE_ROOT,
        REMOTE_TASK1_TEST_ROOT,
    }
)
FORBIDDEN_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "artifacts",
    "experiments",
    "log",
    "logs",
}
FORBIDDEN_SUFFIXES = {
    ".7z",
    ".bz2",
    ".pyc",
    ".log",
    ".lz",
    ".lz4",
    ".pid",
    ".rar",
    ".tar",
    ".tbz",
    ".tbz2",
    ".tgz",
    ".gz",
    ".txz",
    ".xz",
    ".zip",
    ".zst",
}


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
        path = PurePosixPath(raw_path)
        if (
            path == PurePosixPath(".")
            or path.is_absolute()
            or ".." in path.parts
            or path.parts[0].startswith("-")
        ):
            raise ValueError(f"path is not repository-relative: {raw_path}")
        if any(part in FORBIDDEN_PARTS for part in path.parts):
            raise ValueError(f"path is forbidden: {raw_path}")
        if (
            path.suffix.lower() in FORBIDDEN_SUFFIXES
            or path.name.endswith("-review-package.diff")
        ):
            raise ValueError(f"path is forbidden: {raw_path}")
        candidate = root.joinpath(*path.parts)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"path is not a repository file: {raw_path}"
            ) from exc
        if not resolved.is_file():
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
