from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence


REMOTE_HOST = "sitian@10.232.195.203"
REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
KRB5_CACHE = "FILE:/Users/bytedance/krb5cc_sitian"
FORBIDDEN_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "artifacts",
    "experiments",
}
FORBIDDEN_SUFFIXES = {
    ".pyc",
    ".log",
    ".pid",
    ".tar",
    ".tgz",
    ".gz",
    ".zip",
}


@dataclass(frozen=True)
class ScratchConfig:
    repo_root: Path
    remote_host: str = REMOTE_HOST
    remote_root: str = REMOTE_ROOT
    krb5_cache: str = KRB5_CACHE
    attempts: int = 5

    @classmethod
    def default(cls, repo_root: Path) -> "ScratchConfig":
        resolved = repo_root.resolve()
        expected = Path("/Users/bytedance/dev/TinyLLMForge")
        if resolved != expected:
            try:
                resolved.relative_to(Path(REMOTE_ROOT))
            except ValueError as exc:
                raise ValueError(
                    "repo root must resolve to "
                    "/Users/bytedance/dev/TinyLLMForge"
                ) from exc
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


def validate_relative_paths(paths: Sequence[str]) -> tuple[str, ...]:
    if not paths:
        raise ValueError("at least one explicit path is required")
    normalized = []
    for raw_path in paths:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("paths must be non-empty strings")
        path = PurePosixPath(raw_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"path is not repository-relative: {raw_path}")
        if any(part in FORBIDDEN_PARTS for part in path.parts):
            raise ValueError(f"path is forbidden: {raw_path}")
        if (
            path.suffix in FORBIDDEN_SUFFIXES
            or path.name.endswith("-review-package.diff")
        ):
            raise ValueError(f"path is forbidden: {raw_path}")
        normalized.append(path.as_posix())
    return tuple(dict.fromkeys(normalized))


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
