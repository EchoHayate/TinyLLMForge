#!/usr/bin/env python3
"""Remote orchestration contract for SLO-aware cohort-burst gates."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tarfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import run_staged_inference_benchmark_remote as base


APPROVED_ROOT = base.APPROVED_ROOT
TASK_REMOTE_ROOT = APPROVED_ROOT + "/slo-cohort-burst"
REMOTE_HOST = base.REMOTE_HOST
REMOTE_PYTHON = base.REMOTE_PYTHON
MODEL_PATH = base.MODEL_PATHS["qwen3-0.6b"]
DEFAULT_KERBEROS_CACHE = base.KRB5CCNAME
MINIMUM_KERBEROS_LIFETIME_SECONDS = 1_800
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ARTIFACT_ROOT = (
    REPO_ROOT / "artifacts" / "slo_cohort_burst_ceiling"
)
SOURCE_FILES = (
    "tinyvllm",
    "tools/slo_cohort_burst_ceiling.py",
    "tools/profile_slo_cohort_burst_ceiling.py",
    "tools/run_slo_cohort_burst_remote.py",
)
REQUIRED_TERMINAL_FILES = frozenset({
    "raw_rows.jsonl",
    "cost_table.json",
    "ceiling_summary.json",
    "source_manifest.json",
    "remote_verify.json",
})
COMPACT_FILES = REQUIRED_TERMINAL_FILES | frozenset({
    "runner.log",
})


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


def validate_source_commit(requested: str, *, pushed_head: str) -> str:
    if (
        not isinstance(requested, str)
        or re.fullmatch(r"[0-9a-f]{40}", requested) is None
        or not isinstance(pushed_head, str)
        or re.fullmatch(r"[0-9a-f]{40}", pushed_head) is None
    ):
        raise ValueError("source commit is invalid")
    if requested != pushed_head:
        raise ValueError("source commit does not match pushed head")
    return requested


def strict_clean_a100s(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in base.strict_clean_gpus(rows)
        if "A100" in row["name"]
    ]


def committed_source_archive(
    repo_root: Path,
    source_commit: str,
) -> bytes:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit or "") is None:
        raise ValueError("source commit is invalid")
    result = subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            "--prefix=source/",
            source_commit,
            "--",
            *SOURCE_FILES,
        ],
        cwd=Path(repo_root),
        capture_output=True,
        check=False,
    )
    base._require_success(result, "build committed source archive")
    if not isinstance(result.stdout, bytes) or not result.stdout:
        raise ValueError("committed source archive is empty")
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as bundle:
        members = bundle.getmembers()
    if not members:
        raise ValueError("committed source archive is empty")
    for member in members:
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[0] != "source"
            or member.issym()
            or member.islnk()
        ):
            raise ValueError("committed source archive is unsafe")
    return result.stdout


def _runtime_environment(
    *,
    source: str,
    primary: str,
    gpu_index: int,
    run_tag: str,
) -> str:
    prelude = build_remote_runtime_prelude(
        source=source,
        gpu_index=gpu_index,
        dist_port=distributed_port(run_tag),
    )
    return f"cd {shlex.quote(source)} && {prelude}"


def build_worker_plan(
    *,
    paths: dict[str, str],
    run_tag: str,
    source_commit: str,
    gpu: dict,
) -> dict[str, object]:
    if paths != build_remote_paths(run_tag):
        raise ValueError("remote path inventory is invalid")
    validate_source_commit(
        source_commit,
        pushed_head=source_commit,
    )
    clean = strict_clean_a100s([gpu])
    if clean != [gpu]:
        raise ValueError("selected GPU is not strict-clean A100")
    source = paths["staging"] + "/source"
    prefix = _runtime_environment(
        source=source,
        primary=paths["primary"],
        gpu_index=gpu["index"],
        run_tag=run_tag,
    )
    run_command = (
        prefix
        + shlex.quote(REMOTE_PYTHON)
        + " -m tools.profile_slo_cohort_burst_ceiling"
        + " --mode run"
        + " --model "
        + shlex.quote(MODEL_PATH)
        + " --run-tag "
        + shlex.quote(run_tag)
        + " --source-commit "
        + source_commit
        + " --output-dir "
        + shlex.quote(paths["primary"])
    )
    verify_command = (
        prefix
        + shlex.quote(REMOTE_PYTHON)
        + " -m tools.profile_slo_cohort_burst_ceiling"
        + " --mode verify"
        + " --artifact-dir "
        + shlex.quote(paths["primary"])
        + " --output "
        + shlex.quote(paths["controller"] + "/remote_verify.json")
    )
    return {
        "schema_version": "slo-cohort-burst.worker-plan.v1",
        "run_tag": run_tag,
        "source_commit": source_commit,
        "gpu": dict(gpu),
        "paths": dict(paths),
        "commands": [run_command, verify_command],
    }


def is_compact_artifact(relative: str) -> bool:
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact path is invalid")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact path is invalid")
    return path.as_posix() in COMPACT_FILES


def validate_resume_receipt(
    receipt: object,
    *,
    run_tag: str,
    source_commit: str,
    paths: dict[str, str],
) -> dict[str, object]:
    tag = base.validate_run_tag(run_tag)
    validate_source_commit(
        source_commit,
        pushed_head=source_commit,
    )
    if not isinstance(receipt, dict):
        raise ValueError("resume receipt is invalid")
    if (
        receipt.get("schema_version")
        != "slo-cohort-burst.remote-resume.v1"
        or receipt.get("status") != "COMPLETE"
        or receipt.get("run_tag") != tag
        or receipt.get("source_commit") != source_commit
        or receipt.get("remote_paths") != paths
    ):
        raise ValueError("resume source identity is invalid")
    hashes = receipt.get("artifact_sha256")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != set(REQUIRED_TERMINAL_FILES)
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
            for value in hashes.values()
        )
    ):
        raise ValueError("resume terminal hashes are invalid")
    return json.loads(json.dumps(receipt))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the SLO cohort-burst Stage-0 ceiling gate",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("ceiling",),
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-commit")
    parser.add_argument(
        "--local-artifact-root",
        default=os.fspath(LOCAL_ARTIFACT_ROOT),
    )
    args = parser.parse_args(argv)
    base.validate_run_tag(args.tag)
    return args


def main(argv=None) -> int:
    parse_args(argv)
    raise RuntimeError("remote controller is not implemented")


if __name__ == "__main__":
    raise SystemExit(main())
