#!/usr/bin/env python3
"""Remote orchestration contract for SLO-aware cohort-burst gates."""

from __future__ import annotations

import hashlib
import shlex

from tools import run_staged_inference_benchmark_remote as base


APPROVED_ROOT = base.APPROVED_ROOT
TASK_REMOTE_ROOT = APPROVED_ROOT + "/slo-cohort-burst"
REMOTE_HOST = base.REMOTE_HOST
REMOTE_PYTHON = base.REMOTE_PYTHON
MODEL_PATH = base.MODEL_PATHS["qwen3-0.6b"]
DEFAULT_KERBEROS_CACHE = base.KRB5CCNAME
MINIMUM_KERBEROS_LIFETIME_SECONDS = 1_800


def build_remote_paths(run_tag: str) -> dict[str, str]:
    tag = base.validate_run_tag(run_tag)
    paths = {
        "staging": f"{TASK_REMOTE_ROOT}/staging/{tag}",
        "primary": f"{TASK_REMOTE_ROOT}/runs/{tag}",
        "controller": (
            f"{TASK_REMOTE_ROOT}/controller-verification/{tag}"
        ),
    }
    if any(
        not path.startswith(TASK_REMOTE_ROOT + "/")
        for path in paths.values()
    ):
        raise ValueError("remote path is outside approved task root")
    return paths


def distributed_port(run_tag: str) -> int:
    tag = base.validate_run_tag(run_tag)
    digest = hashlib.sha256(tag.encode("utf-8")).digest()
    return 20_000 + int.from_bytes(digest[:4], "big") % 30_000


def build_remote_runtime_prelude(
    *,
    source: str,
    gpu_index: int,
    dist_port: int,
) -> str:
    if (
        not isinstance(source, str)
        or not source.startswith(TASK_REMOTE_ROOT + "/staging/")
        or not source.endswith("/source")
    ):
        raise ValueError("remote source path is invalid")
    if (
        isinstance(gpu_index, bool)
        or not isinstance(gpu_index, int)
        or gpu_index < 0
    ):
        raise ValueError("GPU index is invalid")
    if (
        isinstance(dist_port, bool)
        or not isinstance(dist_port, int)
        or not 20_000 <= dist_port < 50_000
    ):
        raise ValueError("distributed port is invalid")
    runtime = source.rsplit("/", 1)[0] + "/runtime"
    directories = {
        "TMPDIR": runtime + "/tmp",
        "TMP": runtime + "/tmp",
        "TEMP": runtime + "/tmp",
        "PYTHONPYCACHEPREFIX": runtime + "/pycache",
        "XDG_CACHE_HOME": runtime + "/xdg",
        "HF_HOME": runtime + "/hf-home",
        "TORCH_EXTENSIONS_DIR": runtime + "/torch-extensions",
    }
    exports = {
        **directories,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": str(gpu_index),
        "TINYVLLM_DIST_PORT": str(dist_port),
        "MASTER_PORT": str(dist_port),
        "PYTHONPATH": source,
    }
    return (
        "umask 077; mkdir -p "
        + " ".join(
            shlex.quote(path)
            for path in sorted(set(directories.values()))
        )
        + "; "
        + " ".join(
            f"export {name}={shlex.quote(value)};"
            for name, value in exports.items()
        )
        + " "
    )
