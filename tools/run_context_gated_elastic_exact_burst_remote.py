#!/usr/bin/env python3
"""Remote controller for the context-gated elastic K16 ceiling probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from tools import run_exact_burst_one_phase_lease_local_journal_remote as infra
from tools import run_staged_inference_benchmark_remote as base
from tools import run_zero_temperature_greedy_fast_path_remote as common


APPROVED_ROOT = base.APPROVED_ROOT
TASK_REMOTE_ROOT = (
    APPROVED_ROOT + "/context-gated-elastic-exact-burst"
)
REMOTE_HOST = base.REMOTE_HOST
REMOTE_PYTHON = base.REMOTE_PYTHON
MODEL_PATH = base.MODEL_PATHS["qwen3-0.6b"]
REMOTE_PYTEST_SITE = "/data00/home/sitian/pytest-site"
DEFAULT_CONTROL_PATH = base.CONTROL_PATH
DEFAULT_KERBEROS_CACHE = "FILE:/Users/bytedance/krb5cc_sitian"
LOCAL_ARTIFACT_ROOT = (
    REPO_ROOT / "artifacts" / "context_gated_elastic_exact_burst"
)
MINIMUM_KERBEROS_LIFETIME_SECONDS = 5_400
GPU_MEMORY_LIMIT_MIB = 1_024
GPU_UTILIZATION_LIMIT_PERCENT = 5
PERFORMANCE_ROWS = 24
CORRECTNESS_ROWS = 32
CEILING_REPETITIONS = 3
SOURCE_PATCH_SHA256 = hashlib.sha256(b"").hexdigest()
COMMITTED_ARCHIVE_PATHS = ("tinyvllm", "tools")
TASK_TRACKED_PATHS = (
    "tinyvllm/config.py",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
    "tinyvllm/engine/llm_engine.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tools/profile_context_gated_elastic_exact_burst.py",
    "tools/test_profile_context_gated_elastic_exact_burst.py",
    "tools/context_gated_elastic_exact_burst_ceiling.py",
    "tools/test_context_gated_elastic_exact_burst_ceiling.py",
    "tools/run_context_gated_elastic_exact_burst_remote.py",
    "tools/test_run_context_gated_elastic_exact_burst_remote.py",
)
PRIMARY_FILES = (
    "workload_manifest.json",
    "source_manifest.json",
    "source.patch",
    "performance_rows.jsonl",
    "correctness_rows.jsonl",
    "profile_summary.json",
    "ceiling_source_manifest.json",
    "ceiling_summary.json",
    "ceiling_gate.json",
    "producer_receipt.json",
)

validate_kerberos = infra.validate_kerberos
validate_source_commit = infra.validate_source_commit
strict_clean_gpus = infra.strict_clean_gpus
validate_selected_gpu_still_clean = (
    infra.validate_selected_gpu_still_clean
)
committed_archive = infra.committed_archive
download_remote_tree_preserving_partial = (
    common.download_remote_tree_preserving_partial
)
_run_remote_checked = infra._run_remote_checked
_run_remote_with_input_checked = infra._run_remote_with_input_checked
_poll_worker = infra._poll_worker
_query_remote_gpu_rows = infra._query_remote_gpu_rows
_probe_remote_requirements = infra._probe_remote_requirements


def validate_remote_task_root(value: str) -> str:
    if value != APPROVED_ROOT:
        raise ValueError(
            "remote task root must be the approved mounted root"
        )
    return value


def remote_paths(run_tag: str) -> dict[str, str]:
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


def dist_port_for_run_tag(run_tag: str) -> int:
    tag = base.validate_run_tag(run_tag)
    digest = hashlib.sha256(tag.encode("utf-8")).digest()
    return 20_000 + int.from_bytes(digest[:4], "big") % 30_000


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


def _write_json(path: Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )


def _require_task_tracked_diff_clean(repo_root: Path) -> None:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            *TASK_TRACKED_PATHS,
        ],
        cwd=repo_root,
        check=False,
    )
    if result.returncode == 1:
        raise ValueError("tracked task source must be clean")
    if result.returncode != 0:
        raise RuntimeError("tracked task source check failed")


def _require_remote_destinations_absent(paths: dict[str, str]) -> None:
    base.require_remote_destinations_absent(paths)


def _wait_for_clean_gpu(
    *,
    timeout_seconds: int,
    poll_interval_seconds: int,
    local_destination: Path,
):
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
        rows = _query_remote_gpu_rows()
        receipt = {
            "observed_unix_ns": time.time_ns(),
            "gpus": rows,
        }
        _append_jsonl(
            local_destination / "gpu_inventory.jsonl",
            receipt,
        )
        _write_json(
            local_destination / "controller_state.json",
            {
                "schema": (
                    "context_gated_elastic_exact_burst_remote_v1"
                ),
                "status": "MONITORING",
                "last_gpu_inventory": receipt,
            },
        )
        clean = strict_clean_gpus(rows)
        if clean:
            return rows, clean[0]
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "no strict-clean GPU became available"
            )
        time.sleep(poll_interval_seconds)


def _write_idempotent_remote_receipt(
    path: str,
    *,
    receipt: dict,
    context: str,
) -> None:
    payload = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    script = "\n".join((
        "import pathlib,sys",
        f"path=pathlib.Path({path!r})",
        "payload=sys.stdin.buffer.read()",
        "path.parent.mkdir(parents=True,exist_ok=True)",
        "if path.exists():",
        " existing=path.read_bytes()",
        " if existing != payload:",
        "  raise ValueError('existing receipt mismatch')",
        "else:",
        " temporary=path.with_name('.'+path.name+'.pending')",
        " with temporary.open('xb') as handle:",
        "  handle.write(payload)",
        " temporary.replace(path)",
    ))
    _run_remote_with_input_checked(
        f"{REMOTE_PYTHON} -c {shlex.quote(script)}",
        payload,
        context=context,
        retry_attempts=3,
    )


def _create_controller_dir(
    *,
    controller: str,
    receipt: dict,
) -> None:
    _write_idempotent_remote_receipt(
        controller + "/preflight_receipt.json",
        receipt=receipt,
        context="controller receipt upload",
    )


def _write_remote_completion(
    *,
    controller: str,
    receipt: dict,
) -> None:
    _write_idempotent_remote_receipt(
        controller + "/completion_receipt.json",
        receipt=receipt,
        context="completion receipt upload",
    )


def _upload_source_archive(
    *,
    staging: str,
    archive: bytes,
) -> str:
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


def _run_remote_preflight(
    *,
    source: str,
    gpu_index: int,
    dist_port: int,
) -> None:
    pytest_path = (
        f"{shlex.quote(REMOTE_PYTEST_SITE)}:\"$PYTHONPATH\""
    )
    compile_files = (
        "tools/profile_context_gated_elastic_exact_burst.py",
        "tools/context_gated_elastic_exact_burst_ceiling.py",
        "tools/run_context_gated_elastic_exact_burst_remote.py",
    )
    test_files = (
        "tools/test_profile_context_gated_elastic_exact_burst.py",
        "tools/test_context_gated_elastic_exact_burst_ceiling.py",
        "tools/test_run_context_gated_elastic_exact_burst_remote.py",
    )
    command = (
        "set -eu; "
        f"cd {shlex.quote(source)}; "
        + remote_runtime_prelude(
            source=source,
            gpu_index=gpu_index,
            dist_port=dist_port,
        )
        + f"{REMOTE_PYTHON} -m py_compile "
        + " ".join(shlex.quote(path) for path in compile_files)
        + "; "
        + f"PYTHONPATH={pytest_path} "
        + f"{REMOTE_PYTHON} -m pytest -q "
        + " ".join(shlex.quote(path) for path in test_files)
    )
    _run_remote_checked(
        command,
        context="remote source-bound preflight",
        retry_attempts=3,
    )


def _launch_worker(
    *,
    source: str,
    primary: str,
    controller: str,
    run_tag: str,
    source_commit: str,
    gpu_index: int,
    dist_port: int,
) -> int:
    pid_path = controller + "/worker.pid"
    pgid_path = controller + "/worker.pgid"
    exit_path = controller + "/worker.exitcode"
    lock_path = controller + "/worker.launch.lock"
    profile_command = [
        REMOTE_PYTHON,
        "tools/profile_context_gated_elastic_exact_burst.py",
        "--model",
        MODEL_PATH,
        "--device",
        "cuda:0",
        "--output-dir",
        primary,
        "--source-commit",
        source_commit,
        "--run-tag",
        run_tag,
        "--repetitions",
        str(CEILING_REPETITIONS),
        "--warmup-repetitions",
        "1",
    ]
    classify_command = [
        REMOTE_PYTHON,
        "tools/context_gated_elastic_exact_burst_ceiling.py",
        primary,
        "--source-root",
        source,
    ]
    work = (
        " ".join(shlex.quote(part) for part in profile_command)
        + "; "
        + f": > {shlex.quote(primary + '/source.patch')}; "
        + " ".join(shlex.quote(part) for part in classify_command)
    )
    inner = (
        "set +e; umask 077; pid=$$; "
        f"printf '%s\\n' \"$pid\" > {shlex.quote(pid_path + '.pending')}; "
        f"printf '%s\\n' \"$pid\" > {shlex.quote(pgid_path + '.pending')}; "
        f"mv {shlex.quote(pid_path + '.pending')} {shlex.quote(pid_path)}; "
        f"mv {shlex.quote(pgid_path + '.pending')} {shlex.quote(pgid_path)}; "
        f"cd {shlex.quote(source)}; "
        + remote_runtime_prelude(
            source=source,
            gpu_index=gpu_index,
            dist_port=dist_port,
        )
        + "(set -e; "
        + work
        + "); code=$?; "
        + f"printf '%s\\n' \"$code\" > {shlex.quote(exit_path)}; "
        + "exit \"$code\""
    )
    launch = (
        "set -eu; "
        f"pid_path={shlex.quote(pid_path)}; "
        f"pgid_path={shlex.quote(pgid_path)}; "
        f"lock_path={shlex.quote(lock_path)}; "
        "if test -s \"$pid_path\" && test -s \"$pgid_path\"; then "
        "pid=$(cat \"$pid_path\"); "
        "test \"$pid\" = \"$(cat \"$pgid_path\")\"; "
        "printf '%s\\n' \"$pid\"; exit 0; fi; "
        "if mkdir \"$lock_path\" 2>/dev/null; then "
        f"setsid sh -c {shlex.quote(inner)} "
        f"> {shlex.quote(controller + '/worker.stdout.log')} "
        f"2> {shlex.quote(controller + '/worker.stderr.log')} "
        "< /dev/null & fi; "
        "attempt=0; "
        "while ! test -s \"$pid_path\" || ! test -s \"$pgid_path\"; do "
        "attempt=$((attempt + 1)); "
        "test \"$attempt\" -lt 10 || exit 75; sleep 1; done; "
        "rmdir \"$lock_path\" 2>/dev/null || true; "
        "pid=$(cat \"$pid_path\"); "
        "test \"$pid\" = \"$(cat \"$pgid_path\")\"; "
        "printf '%s\\n' \"$pid\""
    )
    result = _run_remote_checked(
        launch,
        context="launch remote ceiling worker",
        retry_attempts=3,
    )
    try:
        pid = int(result.stdout.strip())
    except (AttributeError, ValueError) as error:
        raise ValueError("remote worker PID receipt is invalid") from error
    if pid <= 0:
        raise ValueError("remote worker PID receipt is invalid")
    return pid


def _controller_from_primary(primary: str) -> str:
    prefix = TASK_REMOTE_ROOT + "/runs/"
    if not primary.startswith(prefix):
        raise ValueError("remote primary path is invalid")
    return (
        TASK_REMOTE_ROOT
        + "/controller-verification/"
        + primary[len(prefix):]
    )


def _run_remote_verifier(
    *,
    source: str,
    primary: str,
    gpu_index: int,
    dist_port: int,
) -> None:
    controller = _controller_from_primary(primary)
    output = controller + "/independent-verify/verification.json"
    command = (
        "set -eu; "
        f"cd {shlex.quote(source)}; "
        + remote_runtime_prelude(
            source=source,
            gpu_index=gpu_index,
            dist_port=dist_port,
        )
        + f"{REMOTE_PYTHON} "
        + "tools/context_gated_elastic_exact_burst_ceiling.py "
        + f"{shlex.quote(primary)} --verify-only "
        + f"--source-root {shlex.quote(source)} "
        + f"--output {shlex.quote(output)}"
    )
    _run_remote_checked(
        command,
        context="remote ceiling verification",
        retry_attempts=3,
    )


def _run_frozen_source_local_verifier(
    *,
    frozen_source: Path,
    primary: Path,
    output: Path,
) -> dict:
    frozen_source = Path(frozen_source).resolve()
    primary = Path(primary).resolve()
    output = Path(output).resolve()
    script = (
        frozen_source
        / "tools"
        / "context_gated_elastic_exact_burst_ceiling.py"
    )
    if not script.is_file():
        raise ValueError("frozen-source verifier is missing")
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.fspath(frozen_source)
    environment["PYTHONPYCACHEPREFIX"] = os.fspath(
        output.parent / "pycache"
    )
    result = subprocess.run(
        [
            sys.executable,
            os.fspath(script),
            os.fspath(primary),
            "--verify-only",
            "--source-root",
            os.fspath(frozen_source),
            "--output",
            os.fspath(output),
        ],
        cwd=frozen_source,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "frozen-source local verification failed: "
            + result.stderr.strip()
        )
    return json.loads(output.read_text(encoding="utf-8"))


def _download_ceiling_bundle(
    *,
    paths: dict[str, str],
    local_destination: Path,
) -> dict:
    primary = local_destination / "primary"
    controller = local_destination / "controller"
    frozen_source = local_destination / "frozen-source"
    primary_inventory = download_remote_tree_preserving_partial(
        paths["primary"],
        primary,
    )
    controller_inventory = download_remote_tree_preserving_partial(
        paths["controller"],
        controller,
    )
    source_inventory = download_remote_tree_preserving_partial(
        paths["staging"] + "/source",
        frozen_source,
    )
    missing = [
        name for name in PRIMARY_FILES
        if not (primary / name).is_file()
    ]
    if missing:
        raise ValueError(
            "download is incomplete: " + ", ".join(sorted(missing))
        )
    remote_receipt_path = (
        controller / "independent-verify" / "verification.json"
    )
    if not remote_receipt_path.is_file():
        raise ValueError("remote verification receipt is missing")
    remote_verification = json.loads(
        remote_receipt_path.read_text(encoding="utf-8")
    )
    local_verification = _run_frozen_source_local_verifier(
        frozen_source=frozen_source,
        primary=primary,
        output=(
            local_destination
            / "local-verify"
            / "verification.json"
        ),
    )
    if (
        remote_verification != local_verification
        or remote_verification.get("verified") is not True
        or remote_verification.get("performance_row_count")
        != PERFORMANCE_ROWS
        or remote_verification.get("correctness_row_count")
        != CORRECTNESS_ROWS
    ):
        raise ValueError("verification receipt disagreement")
    download_receipt = {
        "schema": "context_gated_elastic_exact_burst_remote_v1",
        "status": "DOWNLOADED_AND_VERIFIED",
        "primary_inventory": primary_inventory,
        "controller_inventory": controller_inventory,
        "source_inventory": source_inventory,
    }
    _write_json(
        local_destination / "download_receipt.json",
        download_receipt,
    )
    return {
        "remote_verification": remote_verification,
        "local_verification": local_verification,
        "download_receipt": download_receipt,
    }


def run_controller(args) -> dict:
    validate_remote_task_root(args.remote_task_root)
    if args.host != REMOTE_HOST:
        raise ValueError("remote host is not authorized")
    if args.kerberos_cache != DEFAULT_KERBEROS_CACHE:
        raise ValueError("Kerberos cache is not authorized")
    if args.control_path != DEFAULT_CONTROL_PATH:
        raise ValueError("SSH control path is not authorized")
    os.environ["TINYLLMFORGE_SSH_CONTROL_PATH"] = args.control_path
    os.environ["KRB5CCNAME"] = args.kerberos_cache

    local_destination = Path(args.local_output_dir)
    if local_destination.exists() or local_destination.is_symlink():
        raise ValueError("local run tag already exists")
    _require_task_tracked_diff_clean(REPO_ROOT)
    pushed_head = base.require_pushed_head(REPO_ROOT)
    source_commit = validate_source_commit(
        args.source_sha,
        pushed_head=pushed_head,
    )
    kerberos = validate_kerberos(
        minimum_lifetime_seconds=MINIMUM_KERBEROS_LIFETIME_SECONDS
    )
    requirements = _probe_remote_requirements()
    paths = remote_paths(args.run_tag)
    dist_port = dist_port_for_run_tag(args.run_tag)
    _require_remote_destinations_absent(paths)

    local_destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "context_gated_elastic_exact_burst_remote_v1",
        "status": "MONITORING",
        "run_tag": args.run_tag,
        "source_sha": source_commit,
        "source_patch_sha256": SOURCE_PATCH_SHA256,
        "remote_host": args.host,
        "remote_paths": paths,
        "performance_rows": PERFORMANCE_ROWS,
        "correctness_rows": CORRECTNESS_ROWS,
        "dist_port": dist_port,
        "kerberos": kerberos,
        "remote_requirements": requirements,
    }
    _write_json(
        local_destination / "controller_manifest.json",
        manifest,
    )
    gpu_rows, selected = _wait_for_clean_gpu(
        timeout_seconds=args.gpu_wait_timeout_seconds,
        poll_interval_seconds=args.gpu_poll_interval_seconds,
        local_destination=local_destination,
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
        _query_remote_gpu_rows(),
    )
    preflight = {
        **manifest,
        "status": "READY",
        "gpu_inventory": gpu_rows,
        "selected_gpu": selected,
        "launch_gpu": launch_gpu,
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
        gpu_index=selected["index"],
        dist_port=dist_port,
    )
    launch_receipt = {
        "schema": "context_gated_elastic_exact_burst_remote_v1",
        "status": "LAUNCHED",
        "run_tag": args.run_tag,
        "source_sha": source_commit,
        "worker_pid": pid,
        "worker_pgid": pid,
        "selected_gpu": selected,
        "dist_port": dist_port,
    }
    _write_json(
        local_destination / "launch_receipt.json",
        launch_receipt,
    )
    _write_json(
        local_destination / "controller_state.json",
        launch_receipt,
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
    _run_remote_verifier(
        source=source,
        primary=paths["primary"],
        gpu_index=selected["index"],
        dist_port=dist_port,
    )
    completion = {
        "schema": "context_gated_elastic_exact_burst_remote_v1",
        "status": "COMPLETE",
        "run_tag": args.run_tag,
        "source_sha": source_commit,
        "source_patch_sha256": SOURCE_PATCH_SHA256,
        "worker_pid": pid,
        "worker_pgid": pid,
        "worker_exitcode": exitcode,
        "selected_gpu": selected,
        "dist_port": dist_port,
    }
    _write_remote_completion(
        controller=paths["controller"],
        receipt=completion,
    )
    downloaded = _download_ceiling_bundle(
        paths=paths,
        local_destination=local_destination,
    )
    result = {
        **completion,
        "local_destination": os.fspath(local_destination),
        "local_verification": downloaded["local_verification"],
    }
    _write_json(local_destination / "controller_state.json", result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=REMOTE_HOST)
    parser.add_argument("--control-path", default=DEFAULT_CONTROL_PATH)
    parser.add_argument(
        "--kerberos-cache",
        default=DEFAULT_KERBEROS_CACHE,
    )
    parser.add_argument(
        "--remote-task-root",
        default=APPROVED_ROOT,
    )
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--local-output-dir", required=True)
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
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
