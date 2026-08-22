#!/usr/bin/env python3
"""Safe remote controller for the replay-aware metadata Stage-1 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import run_staged_inference_benchmark_remote as base
from tools.replay_aware_decode_metadata_verify import (
    verify_bundle,
)


APPROVED_ROOT = base.APPROVED_ROOT
REMOTE_PYTHON = base.REMOTE_PYTHON
REMOTE_HOST = base.REMOTE_HOST
MODEL_PATH = base.MODEL_PATHS["qwen3-0.6b"]
LOCAL_ARTIFACT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "replay_aware_decode_metadata"
)
MINIMUM_REMOTE_FREE_BYTES = 20 * 1024**3
MINIMUM_KERBEROS_LIFETIME_SECONDS = 5_400
MAX_CONSECUTIVE_POLL_FAILURES = 3
COMMITTED_ARCHIVE_PATHS = (
    "tinyvllm",
    "tools",
)
REQUIRED_PRIMARY_FILES = (
    "case_rows.jsonl",
    "source_manifest.json",
    "workload_manifest.json",
    "summary.json",
    "comparison.json",
    "gate.json",
    "manifest.sha256",
    "independent-verification.json",
)

strict_clean_gpus = base.strict_clean_gpus
validate_kerberos = base.validate_kerberos


def remote_paths(run_tag: str) -> dict[str, str]:
    tag = base.validate_run_tag(run_tag)
    root = (
        APPROVED_ROOT
        + "/replay-aware-decode-metadata"
    )
    paths = {
        "staging": f"{root}/staging/{tag}",
        "primary": f"{root}/runs/{tag}",
        "controller": (
            f"{root}/controller-verification/{tag}"
        ),
    }
    for path in paths.values():
        if (
            not path.startswith(APPROVED_ROOT + "/")
            or "/tmp" in path
            or "/private/tmp" in path
            or "/data00/home/sitian/tllm/TinyLLMForge"
            in path
        ):
            raise ValueError(
                "remote path is outside the approved root"
            )
    return paths


def ensure_local_destination_absent(
    local_root: Path,
    run_tag: str,
) -> Path:
    tag = base.validate_run_tag(run_tag)
    destination = Path(local_root) / tag
    if destination.exists() or destination.is_symlink():
        raise ValueError(
            "local run tag already exists"
        )
    return destination


def validate_source_commit(
    requested: str,
    *,
    pushed_head: str,
) -> str:
    if (
        not isinstance(requested, str)
        or re.fullmatch(r"[0-9a-f]{40}", requested)
        is None
        or not isinstance(pushed_head, str)
        or re.fullmatch(r"[0-9a-f]{40}", pushed_head)
        is None
    ):
        raise ValueError("source commit is invalid")
    if requested != pushed_head:
        raise ValueError(
            "requested source commit does not match pushed head"
        )
    return requested


def validate_remote_requirements(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(
            "remote requirements are not satisfied"
        )
    python = payload.get("python")
    model = payload.get("model")
    root = payload.get("approved_root")
    if (
        not isinstance(python, dict)
        or python.get("path") != REMOTE_PYTHON
        or python.get("is_file") is not True
        or python.get("is_executable") is not True
        or not isinstance(model, dict)
        or model.get("path") != MODEL_PATH
        or model.get("is_dir") is not True
        or model.get("config_is_file") is not True
        or not isinstance(root, dict)
        or root.get("path") != APPROVED_ROOT
        or root.get("is_dir") is not True
        or isinstance(root.get("free_bytes"), bool)
        or not isinstance(root.get("free_bytes"), int)
        or root["free_bytes"] < MINIMUM_REMOTE_FREE_BYTES
    ):
        raise ValueError(
            "remote requirements are not satisfied"
        )
    return payload


def validate_terminal_download_inventory(
    inventory: list[dict],
) -> list[dict]:
    if not isinstance(inventory, list):
        raise ValueError("download is incomplete")
    names = {
        row.get("path")
        for row in inventory
        if isinstance(row, dict)
    }
    missing = set(REQUIRED_PRIMARY_FILES) - names
    if missing:
        raise ValueError(
            "download is incomplete: "
            + ", ".join(sorted(missing))
        )
    return inventory


def _run_remote_checked(
    command: str,
    *,
    context: str,
    text: bool = True,
):
    result = base._run_remote(
        command,
        text=text,
    )
    return base._require_success(result, context)


def _probe_remote_requirements() -> dict:
    return validate_remote_requirements(
        base.probe_remote_requirements(
            "qwen3-0.6b"
        )
    )


def _wait_for_clean_gpu(
    *,
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> tuple[list[dict], dict]:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
        or isinstance(poll_interval_seconds, bool)
        or not isinstance(poll_interval_seconds, int)
        or poll_interval_seconds <= 0
    ):
        raise ValueError("GPU polling policy is invalid")
    deadline = time.monotonic() + timeout_seconds
    while True:
        validate_kerberos(
            minimum_lifetime_seconds=(
                MINIMUM_KERBEROS_LIFETIME_SECONDS
            )
        )
        rows = base.query_remote_gpu_rows()
        clean = strict_clean_gpus(rows)
        if clean:
            return rows, clean[0]
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "no strict-clean GPU became available"
            )
        time.sleep(poll_interval_seconds)


def committed_archive(
    repo_root: Path,
    source_commit: str,
    *,
    command_runner=subprocess.run,
) -> bytes:
    result = command_runner(
        [
            "git",
            "archive",
            "--format=tar",
            source_commit,
            *COMMITTED_ARCHIVE_PATHS,
        ],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    base._require_success(
        result,
        "build committed source archive",
    )
    if not isinstance(result.stdout, bytes) or not result.stdout:
        raise ValueError(
            "committed source archive is empty"
        )
    return result.stdout


def _upload_source_archive(
    *,
    staging: str,
    archive: bytes,
) -> str:
    source = staging + "/source"
    script = "\n".join((
        "import pathlib,sys,tarfile",
        f"staging=pathlib.Path({staging!r})",
        "staging.parent.mkdir(parents=True,exist_ok=True)",
        "staging.mkdir(parents=False,exist_ok=False)",
        "archive=staging/'source.tar'",
        "with archive.open('xb') as handle:",
        " handle.write(sys.stdin.buffer.read())",
        "source=staging/'source'",
        "source.mkdir()",
        "with tarfile.open(archive,'r:') as bundle:",
        " members=bundle.getmembers()",
        " for member in members:",
        "  path=pathlib.PurePosixPath(member.name)",
        "  if (path.is_absolute() or '..' in path.parts",
        "      or member.issym() or member.islnk()):",
        "   raise ValueError('unsafe source archive member')",
        " bundle.extractall(source)",
    ))
    result = base._run_remote_with_input(
        f"{REMOTE_PYTHON} -c {shlex.quote(script)}",
        archive,
    )
    base._require_success(result, "source archive upload")
    return source


def _run_remote_preflight(
    *,
    source: str,
    gpu_index: int,
) -> None:
    commands = preflight_commands()
    remote = (
        "set -eu; "
        f"cd {shlex.quote(source)}; "
        f"export CUDA_VISIBLE_DEVICES={gpu_index}; "
        f"export PYTHONPATH={shlex.quote(source)}; "
        + "; ".join(commands)
    )
    _run_remote_checked(
        remote,
        context="remote source-bound preflight",
    )


def preflight_commands() -> tuple[str, ...]:
    python = shlex.quote(REMOTE_PYTHON)
    return (
        f"{python} tools/test_decode_metadata_landing.py",
        (
            f"{python} "
            "tools/test_multi_sequence_cuda_graph_gate.py"
        ),
        f"{python} tools/test_chunked_prefill.py",
        f"{python} tools/test_profile_prefix_cache.py",
    )


def _create_controller_dir(
    controller: str,
    receipt: dict,
) -> None:
    payload = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    script = "\n".join((
        "import pathlib,sys",
        f"root=pathlib.Path({controller!r})",
        "root.parent.mkdir(parents=True,exist_ok=True)",
        "root.mkdir(parents=False,exist_ok=False)",
        "path=root/'preflight.json'",
        "with path.open('xb') as handle:",
        " handle.write(sys.stdin.buffer.read())",
    ))
    result = base._run_remote_with_input(
        f"{REMOTE_PYTHON} -c {shlex.quote(script)}",
        payload,
    )
    base._require_success(
        result,
        "controller receipt upload",
    )


def _launch_worker(
    *,
    source: str,
    primary: str,
    controller: str,
    run_tag: str,
    source_commit: str,
    gpu_index: int,
) -> int:
    worker = [
        REMOTE_PYTHON,
        "tools/profile_replay_aware_decode_metadata.py",
        "--model",
        MODEL_PATH,
        "--out-dir",
        primary,
        "--source-commit",
        source_commit,
        "--run-tag",
        run_tag,
        "--repetitions",
        "5",
        "--warmup-repetitions",
        "2",
        "--prompt-lengths",
        "256,2048,8192",
        "--generated-tokens",
        "128",
        "--gpu-memory-utilization",
        "0.5",
    ]
    worker_command = " ".join(
        shlex.quote(part) for part in worker
    )
    inner = (
        "set +e; "
        f"cd {shlex.quote(source)}; "
        f"export CUDA_VISIBLE_DEVICES={gpu_index}; "
        f"export PYTHONPATH={shlex.quote(source)}; "
        f"{worker_command}; "
        "code=$?; "
        f"printf '%s\\n' \"$code\" > "
        f"{shlex.quote(controller + '/remote_exitcode')}"
    )
    launch = (
        "set -eu; "
        f"setsid sh -c {shlex.quote(inner)} "
        f"> {shlex.quote(controller + '/runner.log')} "
        "2>&1 < /dev/null & "
        "printf '%s\\n' \"$!\""
    )
    result = _run_remote_checked(
        launch,
        context="launch remote benchmark worker",
    )
    try:
        pid = int(result.stdout.strip())
    except (AttributeError, ValueError) as error:
        raise ValueError(
            "remote worker PID receipt is invalid"
        ) from error
    if pid <= 0:
        raise ValueError(
            "remote worker PID receipt is invalid"
        )
    return pid


def _poll_worker(
    *,
    controller: str,
    poll_interval_seconds: int,
) -> int:
    consecutive_failures = 0
    while True:
        script = "\n".join((
            "import json,pathlib",
            (
                "path=pathlib.Path("
                f"{(controller + '/remote_exitcode')!r})"
            ),
            "if path.is_file():",
            " print(json.dumps({'state':'finished',",
            "  'exitcode':int(path.read_text().strip())}))",
            "else:",
            " print(json.dumps({'state':'running'}))",
        ))
        result = base._run_remote(
            f"{REMOTE_PYTHON} -c {shlex.quote(script)}"
        )
        if result.returncode != 0:
            consecutive_failures += 1
            if (
                consecutive_failures
                >= MAX_CONSECUTIVE_POLL_FAILURES
            ):
                base._require_success(
                    result,
                    "remote worker polling",
                )
            time.sleep(poll_interval_seconds)
            continue
        consecutive_failures = 0
        try:
            receipt = json.loads(result.stdout)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                "remote worker poll receipt is invalid"
            ) from error
        if receipt.get("state") == "finished":
            code = receipt.get("exitcode")
            if (
                isinstance(code, bool)
                or not isinstance(code, int)
            ):
                raise ValueError(
                    "remote worker exit code is invalid"
                )
            return code
        if receipt != {"state": "running"}:
            raise ValueError(
                "remote worker poll receipt is invalid"
            )
        time.sleep(poll_interval_seconds)


def download_remote_tree_preserving_partial(
    remote_root: str,
    destination: Path,
    *,
    retries: int = 3,
) -> list[dict]:
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise ValueError(
            "download destination already exists"
        )
    partial = target.with_name(target.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise ValueError(
            "download partial destination already exists"
        )
    if (
        isinstance(retries, bool)
        or not isinstance(retries, int)
        or retries <= 0
    ):
        raise ValueError("download retry policy is invalid")
    inventory = base.fetch_remote_inventory(remote_root)
    partial.mkdir(parents=True, exist_ok=False)
    for record in inventory:
        path = partial / record["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with path.open("xb") as handle:
            for chunk in record["chunks"]:
                last_error = None
                for _attempt in range(retries):
                    try:
                        payload = base.download_chunk(
                            remote_root
                            + "/"
                            + record["path"],
                            offset=chunk["offset"],
                            length=chunk["length"],
                            expected_sha256=chunk["sha256"],
                        )
                        break
                    except (
                        RuntimeError,
                        ValueError,
                    ) as error:
                        last_error = error
                else:
                    raise RuntimeError(
                        "artifact chunk download failed: "
                        + record["path"]
                    ) from last_error
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                digest.update(payload)
        if (
            path.stat().st_size != record["size_bytes"]
            or digest.hexdigest() != record["sha256"]
        ):
            raise ValueError(
                "downloaded artifact mismatch: "
                + record["path"]
            )
    partial.rename(target)
    base.verify_downloaded_tree(target, inventory)
    return inventory


def _run_remote_gates(
    *,
    source: str,
    primary: str,
) -> None:
    commands = (
        (
            f"{REMOTE_PYTHON} "
            "tools/replay_aware_decode_metadata_gate.py "
            f"--run-dir {shlex.quote(primary)} "
            f"--repo-root {shlex.quote(source)}"
        ),
        (
            f"{REMOTE_PYTHON} "
            "tools/replay_aware_decode_metadata_verify.py "
            f"--run-dir {shlex.quote(primary)} "
            f"--repo-root {shlex.quote(source)}"
        ),
    )
    remote = (
        "set -eu; "
        f"cd {shlex.quote(source)}; "
        f"export PYTHONPATH={shlex.quote(source)}; "
        + "; ".join(commands)
    )
    _run_remote_checked(
        remote,
        context="remote producer and independent verification",
    )


def _write_remote_completion(
    *,
    controller: str,
    receipt: dict,
) -> None:
    payload = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    script = "\n".join((
        "import pathlib,sys",
        (
            "path=pathlib.Path("
            f"{(controller + '/completion.json')!r})"
        ),
        "with path.open('xb') as handle:",
        " handle.write(sys.stdin.buffer.read())",
    ))
    result = base._run_remote_with_input(
        f"{REMOTE_PYTHON} -c {shlex.quote(script)}",
        payload,
    )
    base._require_success(
        result,
        "controller completion upload",
    )


def _download_terminal_bundle(
    *,
    paths: dict[str, str],
    local_destination: Path,
) -> dict:
    local_destination.mkdir(
        parents=True,
        exist_ok=False,
    )
    primary = local_destination / "primary"
    controller = local_destination / "controller"
    primary_inventory = download_remote_tree_preserving_partial(
        paths["primary"],
        primary,
    )
    validate_terminal_download_inventory(
        primary_inventory
    )
    controller_inventory = download_remote_tree_preserving_partial(
        paths["controller"],
        controller,
    )
    local_verification = verify_bundle(
        primary,
        repo_root=REPO_ROOT,
    )
    return {
        "primary_inventory": primary_inventory,
        "controller_inventory": controller_inventory,
        "local_verification": local_verification,
    }


def run_controller(args) -> dict:
    if args.model_tier != "qwen3-0.6b":
        raise ValueError(
            "only qwen3-0.6b Stage 1 is authorized"
        )
    local_destination = ensure_local_destination_absent(
        Path(args.local_artifact_root),
        args.run_tag,
    )
    pushed_head = base.require_pushed_head(REPO_ROOT)
    source_commit = validate_source_commit(
        (
            pushed_head
            if args.source_commit is None
            else args.source_commit
        ),
        pushed_head=pushed_head,
    )
    kerberos = validate_kerberos(
        minimum_lifetime_seconds=(
            MINIMUM_KERBEROS_LIFETIME_SECONDS
        )
    )
    requirements = _probe_remote_requirements()
    paths = remote_paths(args.run_tag)
    base.require_remote_destinations_absent(paths)
    gpu_rows, selected = _wait_for_clean_gpu(
        timeout_seconds=args.max_wait_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    archive = committed_archive(
        REPO_ROOT,
        source_commit,
    )
    source = _upload_source_archive(
        staging=paths["staging"],
        archive=archive,
    )
    _run_remote_preflight(
        source=source,
        gpu_index=selected["index"],
    )
    latest_rows = base.query_remote_gpu_rows()
    latest_by_index = {
        row["index"]: row for row in latest_rows
    }
    latest = latest_by_index.get(selected["index"])
    if (
        latest is None
        or strict_clean_gpus([latest]) != [latest]
        or latest["uuid"] != selected["uuid"]
    ):
        raise RuntimeError(
            "selected GPU is no longer strict-clean"
        )
    preflight = {
        "schema_version":
            "replay-aware-decode-metadata.controller.v1",
        "status": "READY",
        "run_tag": args.run_tag,
        "source_commit": source_commit,
        "remote_host": REMOTE_HOST,
        "remote_paths": paths,
        "remote_requirements": requirements,
        "kerberos": kerberos,
        "gpu_inventory": gpu_rows,
        "selected_gpu": selected,
    }
    _create_controller_dir(paths["controller"], preflight)
    pid = _launch_worker(
        source=source,
        primary=paths["primary"],
        controller=paths["controller"],
        run_tag=args.run_tag,
        source_commit=source_commit,
        gpu_index=selected["index"],
    )
    exitcode = _poll_worker(
        controller=paths["controller"],
        poll_interval_seconds=args.poll_interval_seconds,
    )
    if exitcode != 0:
        raise RuntimeError(
            f"remote benchmark worker failed with exit code {exitcode}"
        )
    _run_remote_gates(
        source=source,
        primary=paths["primary"],
    )
    completion = {
        "schema_version":
            "replay-aware-decode-metadata.controller.v1",
        "status": "COMPLETE",
        "run_tag": args.run_tag,
        "source_commit": source_commit,
        "worker_pid": pid,
        "worker_exitcode": exitcode,
        "selected_gpu": selected,
    }
    _write_remote_completion(
        controller=paths["controller"],
        receipt=completion,
    )
    downloaded = _download_terminal_bundle(
        paths=paths,
        local_destination=local_destination,
    )
    return completion | {
        "local_destination": str(local_destination),
        "local_verification": downloaded[
            "local_verification"
        ],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    parser.add_argument(
        "--model-tier",
        choices=("qwen3-0.6b",),
        default="qwen3-0.6b",
    )
    parser.add_argument(
        "--source-commit",
        default=None,
    )
    parser.add_argument(
        "--local-artifact-root",
        default=str(LOCAL_ARTIFACT_ROOT),
    )
    parser.add_argument(
        "--max-wait-seconds",
        type=int,
        default=21_600,
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=30,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    result = run_controller(parse_args(argv))
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
