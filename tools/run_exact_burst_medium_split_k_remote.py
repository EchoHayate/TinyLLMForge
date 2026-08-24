#!/usr/bin/env python3
"""Strict-clean remote controller for the medium split-K burst gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from tools import run_exact_greedy_decode_burst_remote as legacy
from tools import run_staged_inference_benchmark_remote as base
from tools import run_zero_temperature_greedy_fast_path_remote as common
from tools.exact_burst_medium_split_k_verify import (
    verify_artifact_directory,
)


APPROVED_ROOT = base.APPROVED_ROOT
TASK_REMOTE_ROOT = (
    APPROVED_ROOT + "/exact-burst-medium-split-k"
)
REMOTE_PYTHON = base.REMOTE_PYTHON
REMOTE_HOST = base.REMOTE_HOST
MODEL_PATH = base.MODEL_PATHS["qwen3-0.6b"]
LOCAL_ARTIFACT_ROOT = (
    REPO_ROOT / "artifacts" / "exact_burst_medium_split_k"
)
MINIMUM_KERBEROS_LIFETIME_SECONDS = 5_400
COMMITTED_ARCHIVE_PATHS = ("tinyvllm", "tools")
SOURCE_PATCH_SHA256 = hashlib.sha256(b"").hexdigest()

strict_clean_gpus = base.strict_clean_gpus
validate_kerberos = base.validate_kerberos
validate_source_commit = legacy.validate_source_commit
validate_remote_requirements = legacy.validate_remote_requirements
validate_selected_gpu_still_clean = (
    legacy.validate_selected_gpu_still_clean
)
committed_archive = legacy.committed_archive
download_remote_tree_preserving_partial = (
    common.download_remote_tree_preserving_partial
)
_create_controller_dir = legacy._create_controller_dir
_poll_worker = legacy._poll_worker
_write_remote_completion = legacy._write_remote_completion


def remote_paths(run_tag: str) -> dict[str, str]:
    tag = base.validate_run_tag(run_tag)
    paths = {
        "staging": f"{TASK_REMOTE_ROOT}/staging/{tag}",
        "primary": f"{TASK_REMOTE_ROOT}/runs/{tag}",
        "controller": (
            f"{TASK_REMOTE_ROOT}/controller-verification/{tag}"
        ),
    }
    for path in paths.values():
        if not path.startswith(TASK_REMOTE_ROOT + "/"):
            raise ValueError("remote path is outside approved task root")
    return paths


def dist_port_for_run_tag(run_tag: str) -> int:
    tag = base.validate_run_tag(run_tag)
    digest = hashlib.sha256(tag.encode("utf-8")).digest()
    return 20_000 + int.from_bytes(digest[:4], "big") % 30_000


def ensure_local_destination_absent(path: Path) -> Path:
    destination = path.resolve()
    if destination.exists():
        raise ValueError(
            f"local run tag already exists: {destination}"
        )
    return destination


def _run_remote_checked(
    command: str,
    *,
    context: str,
    text: bool = True,
):
    return base._require_success(
        base._run_remote(command, text=text),
        context,
    )


def _probe_remote_requirements() -> dict:
    return validate_remote_requirements(
        base.probe_remote_requirements("qwen3-0.6b")
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


def _upload_source_archive(*, staging: str, archive: bytes) -> str:
    if not staging.startswith(TASK_REMOTE_ROOT + "/staging/"):
        raise ValueError("remote staging path is invalid")
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


def preflight_commands() -> tuple[str, ...]:
    python = shlex.quote(REMOTE_PYTHON)
    pytest_path = (
        "/data00/home/sitian/pytest-site:\"$PYTHONPATH\""
    )
    tests = (
        "tools/test_exact_greedy_decode_burst.py",
        "tools/test_model_runner_spec_verify.py",
        "tools/test_profile_exact_burst_medium_split_k.py",
        "tools/test_exact_burst_medium_split_k_gate.py",
        "tools/test_exact_burst_medium_split_k_verify.py",
        "tools/test_run_exact_burst_medium_split_k_remote.py",
    )
    return tuple(
        f"PYTHONPATH={pytest_path} "
        f"{python} -m pytest -q {shlex.quote(test)}"
        for test in tests
    )


def remote_runtime_prelude(
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


def _run_remote_preflight(
    *,
    source: str,
    gpu_index: int,
    dist_port: int,
) -> None:
    command = (
        "set -eu; "
        f"cd {shlex.quote(source)}; "
        + remote_runtime_prelude(
            source=source,
            gpu_index=gpu_index,
            dist_port=dist_port,
        )
        + "; ".join(preflight_commands())
    )
    _run_remote_checked(
        command,
        context="remote source-bound preflight",
    )


def _launch_worker(
    *,
    source: str,
    primary: str,
    controller: str,
    run_tag: str,
    source_commit: str,
    model: str,
    repetitions: int,
    warmup_repetitions: int,
    gpu_index: int,
    dist_port: int,
) -> int:
    worker = [
        REMOTE_PYTHON,
        "tools/profile_exact_burst_medium_split_k.py",
        "--model", model,
        "--out-dir", primary,
        "--source-commit", source_commit,
        "--run-tag", run_tag,
        "--repetitions", str(repetitions),
        "--warmup-repetitions", str(warmup_repetitions),
        "--context-lengths",
        "1025,1537,2049,2561,3073,3585,4090,6145",
        "--generated-tokens", "128",
        "--gpu-memory-utilization", "0.5",
    ]
    inner = (
        "set +e; "
        f"cd {shlex.quote(source)}; "
        + remote_runtime_prelude(
            source=source,
            gpu_index=gpu_index,
            dist_port=dist_port,
        )
        + " ".join(shlex.quote(part) for part in worker)
        + "; code=$?; "
        + f"printf '%s\\n' \"$code\" > "
        + shlex.quote(controller + "/remote_exitcode")
    )
    launch = (
        "set -eu; "
        f"setsid sh -c {shlex.quote(inner)} "
        f"> {shlex.quote(controller + '/runner.log')} "
        "2>&1 < /dev/null & printf '%s\\n' \"$!\""
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
        raise ValueError("remote worker PID receipt is invalid")
    return pid


def _run_remote_gates(
    *,
    source: str,
    primary: str,
    controller: str,
    gpu_index: int,
    dist_port: int,
) -> None:
    receipt = controller + "/remote_verification.json"
    commands = (
        (
            f"{REMOTE_PYTHON} "
            "tools/exact_burst_medium_split_k_gate.py "
            f"--artifact-dir {shlex.quote(primary)}"
        ),
        (
            f"{REMOTE_PYTHON} "
            "tools/exact_burst_medium_split_k_verify.py "
            f"--artifact-dir {shlex.quote(primary)} "
            f"--receipt-path {shlex.quote(receipt)}"
        ),
    )
    gate_command = (
        f"set +e; {commands[0]}; gate_code=$?; set -e; "
        "if [ \"$gate_code\" -gt 1 ]; then exit \"$gate_code\"; fi"
    )
    command = (
        "set -eu; "
        f"cd {shlex.quote(source)}; "
        + remote_runtime_prelude(
            source=source,
            gpu_index=gpu_index,
            dist_port=dist_port,
        )
        + gate_command
        + "; "
        + commands[1]
    )
    _run_remote_checked(
        command,
        context="remote producer and independent verification",
    )


def validate_terminal_download_inventory(
    inventory: list[dict],
    *,
    manifest_artifacts: dict,
) -> list[dict]:
    if (
        not isinstance(inventory, list)
        or not isinstance(manifest_artifacts, dict)
        or not manifest_artifacts
    ):
        raise ValueError("download is incomplete: manifest inventory")
    names = {
        row.get("path")
        for row in inventory
        if isinstance(row, dict)
    }
    required = set(manifest_artifacts) | {"manifest.json"}
    if required - names:
        raise ValueError("download is incomplete")
    return inventory


def validate_verification_receipt_agreement(
    remote_receipt: dict,
    local_receipt: dict,
) -> dict:
    if (
        not isinstance(remote_receipt, dict)
        or remote_receipt.get("verified") is not True
        or remote_receipt != local_receipt
    ):
        raise ValueError("verification receipt disagreement")
    return remote_receipt


def _download_terminal_bundle(
    *,
    paths: dict[str, str],
    local_destination: Path,
) -> dict:
    local_destination.mkdir(parents=True, exist_ok=False)
    primary = local_destination / "primary"
    controller = local_destination / "controller"
    primary_inventory = download_remote_tree_preserving_partial(
        paths["primary"],
        primary,
    )
    manifest = json.loads(
        (primary / "manifest.json").read_text(encoding="utf-8")
    )
    validate_terminal_download_inventory(
        primary_inventory,
        manifest_artifacts=manifest.get("artifacts"),
    )
    controller_inventory = download_remote_tree_preserving_partial(
        paths["controller"],
        controller,
    )
    remote_receipt = json.loads(
        (controller / "remote_verification.json").read_text(
            encoding="utf-8"
        )
    )
    local_receipt = verify_artifact_directory(primary)
    validate_verification_receipt_agreement(
        remote_receipt,
        local_receipt,
    )
    return {
        "primary_inventory": primary_inventory,
        "controller_inventory": controller_inventory,
        "local_verification": local_receipt,
    }


def run_controller(args) -> dict:
    local_destination = ensure_local_destination_absent(
        Path(args.local_destination)
    )
    pushed_head = base.require_pushed_head(REPO_ROOT)
    source_commit = validate_source_commit(
        args.source_commit,
        pushed_head=pushed_head,
    )
    kerberos = validate_kerberos(
        minimum_lifetime_seconds=MINIMUM_KERBEROS_LIFETIME_SECONDS
    )
    requirements = _probe_remote_requirements()
    paths = remote_paths(args.run_tag)
    dist_port = dist_port_for_run_tag(args.run_tag)
    base.require_remote_destinations_absent(paths)
    gpu_rows, selected = _wait_for_clean_gpu(
        timeout_seconds=args.gpu_wait_timeout_seconds,
        poll_interval_seconds=args.gpu_poll_interval_seconds,
    )
    archive = committed_archive(REPO_ROOT, source_commit)
    source = _upload_source_archive(
        staging=paths["staging"],
        archive=archive,
    )
    validate_kerberos(
        minimum_lifetime_seconds=MINIMUM_KERBEROS_LIFETIME_SECONDS
    )
    _run_remote_preflight(
        source=source,
        gpu_index=selected["index"],
        dist_port=dist_port,
    )
    launch_gpu = validate_selected_gpu_still_clean(
        selected,
        base.query_remote_gpu_rows(),
    )
    preflight = {
        "schema_version":
            "exact-burst-medium-split-k.controller.v1",
        "status": "READY",
        "run_tag": args.run_tag,
        "source_commit": source_commit,
        "source_patch_sha256": SOURCE_PATCH_SHA256,
        "remote_host": REMOTE_HOST,
        "remote_paths": paths,
        "remote_requirements": requirements,
        "kerberos": kerberos,
        "gpu_inventory": gpu_rows,
        "selected_gpu": selected,
        "launch_gpu": launch_gpu,
        "dist_port": dist_port,
    }
    _create_controller_dir(
        controller=paths["controller"],
        receipt=preflight,
    )
    pid = _launch_worker(
        source=source,
        primary=paths["primary"],
        controller=paths["controller"],
        run_tag=args.run_tag,
        source_commit=source_commit,
        model=args.model,
        repetitions=args.repetitions,
        warmup_repetitions=args.warmup_repetitions,
        gpu_index=selected["index"],
        dist_port=dist_port,
    )
    exitcode = _poll_worker(
        controller=paths["controller"],
        worker_pid=pid,
        poll_interval_seconds=args.gpu_poll_interval_seconds,
    )
    if exitcode != 0:
        raise RuntimeError(
            f"remote benchmark worker failed with exit code {exitcode}"
        )
    _run_remote_gates(
        source=source,
        primary=paths["primary"],
        controller=paths["controller"],
        gpu_index=selected["index"],
        dist_port=dist_port,
    )
    completion = {
        "schema_version":
            "exact-burst-medium-split-k.controller.v1",
        "status": "COMPLETE",
        "run_tag": args.run_tag,
        "source_commit": source_commit,
        "source_patch_sha256": SOURCE_PATCH_SHA256,
        "worker_pid": pid,
        "worker_exitcode": exitcode,
        "selected_gpu": selected,
        "dist_port": dist_port,
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
        "local_verification": downloaded["local_verification"],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--local-destination", required=True)
    parser.add_argument(
        "--repetitions",
        type=int,
        choices=(3, 5),
        default=5,
    )
    parser.add_argument(
        "--warmup-repetitions",
        type=int,
        choices=(1, 2),
        default=2,
    )
    parser.add_argument(
        "--gpu-wait-timeout-seconds",
        type=int,
        default=28_800,
    )
    parser.add_argument(
        "--gpu-poll-interval-seconds",
        type=int,
        default=60,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    result = run_controller(parse_args(argv))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
