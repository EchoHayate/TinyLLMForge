#!/usr/bin/env python3
"""Remote controller for the one-phase lease-local journal gate."""

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
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from tools import run_exact_greedy_decode_burst_remote as legacy
from tools import run_staged_inference_benchmark_remote as base
from tools import run_zero_temperature_greedy_fast_path_remote as common
from tools.exact_burst_one_phase_lease_local_journal_verify import (
    verify_artifact_directory,
)


APPROVED_ROOT = base.APPROVED_ROOT
TASK_REMOTE_ROOT = (
    APPROVED_ROOT
    + "/exact-burst-one-phase-lease-local-journal"
)
REMOTE_HOST = base.REMOTE_HOST
REMOTE_PYTHON = base.REMOTE_PYTHON
MODEL_PATH = base.MODEL_PATHS["qwen3-0.6b"]
REMOTE_PYTEST_SITE = "/data00/home/sitian/pytest-site"
DEFAULT_CONTROL_PATH = "none"
DEFAULT_KERBEROS_CACHE = "FILE:/Users/bytedance/krb5cc_sitian"
LOCAL_ARTIFACT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "exact_burst_one_phase_lease_local_journal"
)
MINIMUM_KERBEROS_LIFETIME_SECONDS = 5_400
GPU_MEMORY_LIMIT_MIB = 1_024
GPU_UTILIZATION_LIMIT_PERCENT = 5
PERFORMANCE_ROWS = 60
CORRECTNESS_ROWS = 24
MAX_CONSECUTIVE_POLL_FAILURES = 3
REQUIREMENT_PROBE_ATTEMPTS = 3
COMMITTED_ARCHIVE_PATHS = ("tinyvllm", "tools")
SOURCE_PATCH_SHA256 = hashlib.sha256(b"").hexdigest()
TASK_TRACKED_PATHS = (
    "tinyvllm/config.py",
    "tinyvllm/engine/block_manager.py",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
    "tinyvllm/engine/llm_engine.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tools/profile_exact_burst_one_phase_lease_local_journal.py",
    "tools/test_profile_exact_burst_one_phase_lease_local_journal.py",
    "tools/exact_burst_one_phase_lease_local_journal_gate.py",
    "tools/test_exact_burst_one_phase_lease_local_journal_gate.py",
    "tools/exact_burst_one_phase_lease_local_journal_verify.py",
    "tools/test_exact_burst_one_phase_lease_local_journal_verify.py",
    "tools/run_exact_burst_one_phase_lease_local_journal_remote.py",
    "tools/test_run_exact_burst_one_phase_lease_local_journal_remote.py",
)
PRIMARY_FILES = (
    "workload_manifest.json",
    "performance_rows.jsonl",
    "correctness_rows.jsonl",
    "prepare_samples.jsonl",
    "summary.json",
    "source_manifest.json",
    "runner_receipt.json",
)

validate_kerberos = base.validate_kerberos
committed_archive = legacy.committed_archive
download_remote_tree_preserving_partial = (
    common.download_remote_tree_preserving_partial
)


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
        raise ValueError(
            "remote path is outside approved task root"
        )
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
        or not source.startswith(
            TASK_REMOTE_ROOT + "/staging/"
        )
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


def validate_source_commit(
    requested: str,
    *,
    pushed_head: str,
) -> str:
    if (
        not isinstance(requested, str)
        or re.fullmatch(r"[0-9a-f]{40}", requested) is None
        or not isinstance(pushed_head, str)
        or re.fullmatch(r"[0-9a-f]{40}", pushed_head) is None
    ):
        raise ValueError("source commit is invalid")
    if requested != pushed_head:
        raise ValueError(
            "requested source commit does not match pushed head"
        )
    return requested


def strict_clean_gpus(rows: list[dict]) -> list[dict]:
    clean = []
    seen_indices = set()
    seen_uuids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("GPU row is invalid")
        index = row.get("index")
        uuid = row.get("uuid")
        memory = row.get("memory_used_mib")
        utilization = row.get("utilization_percent")
        processes = row.get("compute_processes")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or not isinstance(uuid, str)
            or not uuid
            or isinstance(memory, bool)
            or not isinstance(memory, int)
            or memory < 0
            or isinstance(utilization, bool)
            or not isinstance(utilization, int)
            or utilization < 0
            or not isinstance(processes, list)
            or index in seen_indices
            or uuid in seen_uuids
        ):
            raise ValueError("GPU row is invalid")
        seen_indices.add(index)
        seen_uuids.add(uuid)
        if (
            memory <= GPU_MEMORY_LIMIT_MIB
            and utilization <= GPU_UTILIZATION_LIMIT_PERCENT
            and not processes
        ):
            clean.append(row)
    return clean


def validate_selected_gpu_still_clean(
    selected: dict,
    latest_rows: list[dict],
) -> dict:
    selected_index = (
        selected.get("index")
        if isinstance(selected, dict)
        else None
    )
    latest = {
        row.get("index"): row
        for row in latest_rows
        if isinstance(row, dict)
    }.get(selected_index)
    if (
        latest is None
        or latest.get("uuid") != selected.get("uuid")
        or strict_clean_gpus([latest]) != [latest]
    ):
        raise RuntimeError(
            "selected GPU is no longer strict-clean"
        )
    return latest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )


def _require_task_tracked_diff_clean(
    repo_root: Path,
) -> None:
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
    if result.returncode != 0:
        if result.returncode == 1:
            raise ValueError(
                "tracked task source must be clean"
            )
        raise RuntimeError(
            "tracked task source check failed"
        )


def _retry_idempotent(operation, *, attempts: int = 3):
    if (
        isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or attempts <= 0
    ):
        raise ValueError("remote retry policy is invalid")
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except RuntimeError:
            if attempt == attempts:
                raise
            time.sleep(attempt)
    raise AssertionError("unreachable idempotent retry")


def _run_remote_checked(
    command: str,
    *,
    context: str,
    text: bool = True,
    retry_attempts: int = 1,
):
    if (
        isinstance(retry_attempts, bool)
        or not isinstance(retry_attempts, int)
        or retry_attempts <= 0
    ):
        raise ValueError("remote retry policy is invalid")
    return _retry_idempotent(
        lambda: base._require_success(
                base._run_remote(command, text=text),
                context,
        ),
        attempts=retry_attempts,
    )


def _run_remote_with_input_checked(
    command: str,
    payload: bytes,
    *,
    context: str,
    retry_attempts: int = 1,
):
    return _retry_idempotent(
        lambda: base._require_success(
            base._run_remote_with_input(command, payload),
            context,
        ),
        attempts=retry_attempts,
    )


def _probe_remote_requirements() -> dict:
    return _retry_idempotent(
        lambda: legacy.validate_remote_requirements(
                base.probe_remote_requirements("qwen3-0.6b")
        ),
        attempts=REQUIREMENT_PROBE_ATTEMPTS,
    )


def _require_remote_destinations_absent(
    paths: dict[str, str],
) -> None:
    _retry_idempotent(
        lambda: base.require_remote_destinations_absent(paths)
    )


def _query_remote_gpu_rows() -> list[dict]:
    return _retry_idempotent(base.query_remote_gpu_rows)


def _write_idempotent_remote_receipt(
    *,
    path: str,
    receipt: dict,
    create_parent: bool,
    context: str,
) -> None:
    if (
        not isinstance(path, str)
        or not path.startswith(
            TASK_REMOTE_ROOT + "/controller-verification/"
        )
    ):
        raise ValueError("remote receipt path is invalid")
    payload = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    script_lines = [
        "import os,pathlib,sys",
        f"path=pathlib.Path({path!r})",
        "payload=sys.stdin.buffer.read()",
    ]
    if create_parent:
        script_lines.extend((
            "path.parent.parent.mkdir(parents=True,exist_ok=True)",
            "path.parent.mkdir(parents=False,exist_ok=True)",
        ))
    else:
        script_lines.extend((
            "if not path.parent.is_dir():",
            " raise ValueError('receipt parent is missing')",
        ))
    script_lines.extend((
        "pending=path.with_name(path.name+'.pending')",
        "if path.is_file():",
        " if path.read_bytes()!=payload:",
        "  raise ValueError('existing receipt mismatch')",
        "elif pending.exists():",
        " if not pending.is_file() or pending.read_bytes()!=payload:",
        "  raise ValueError('existing receipt mismatch')",
        " os.replace(pending,path)",
        "else:",
        " with pending.open('xb') as handle:",
        "  handle.write(payload)",
        "  handle.flush()",
        "  os.fsync(handle.fileno())",
        " os.replace(pending,path)",
        "if path.read_bytes()!=payload:",
        " raise ValueError('existing receipt mismatch')",
        "print('stored')",
    ))
    _run_remote_with_input_checked(
        f"{REMOTE_PYTHON} -c "
        + shlex.quote("\n".join(script_lines)),
        payload,
        context=context,
        retry_attempts=3,
    )


def _create_controller_dir(
    controller: str,
    receipt: dict,
) -> None:
    _write_idempotent_remote_receipt(
        path=controller + "/preflight.json",
        receipt=receipt,
        create_parent=True,
        context="controller receipt upload",
    )


def _write_remote_completion(
    *,
    controller: str,
    receipt: dict,
) -> None:
    _write_idempotent_remote_receipt(
        path=controller + "/completion.json",
        receipt=receipt,
        create_parent=False,
        context="controller completion upload",
    )


def _wait_for_clean_gpu(
    *,
    timeout_seconds: int,
    poll_interval_seconds: int,
    local_destination: Path | None = None,
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
        rows = _query_remote_gpu_rows()
        receipt = {
            "observed_unix_ns": time.time_ns(),
            "gpus": rows,
        }
        if local_destination is not None:
            _append_jsonl(
                local_destination / "gpu_inventory.jsonl",
                receipt,
            )
            _write_json(
                local_destination / "controller_state.json",
                {
                    "schema": (
                        "exact_burst_one_phase_lease_local_"
                        "journal_remote_v1"
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


def _upload_source_archive(
    *,
    staging: str,
    archive: bytes,
) -> str:
    if not staging.startswith(
        TASK_REMOTE_ROOT + "/staging/"
    ):
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
    base._require_success(
        result,
        "source archive upload",
    )
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
        "tinyvllm/config.py",
        "tinyvllm/engine/block_manager.py",
        "tinyvllm/engine/exact_greedy_decode_burst.py",
        "tinyvllm/engine/scheduler.py",
        (
            "tools/"
            "exact_burst_one_phase_lease_local_journal_gate.py"
        ),
        (
            "tools/"
            "exact_burst_one_phase_lease_local_journal_verify.py"
        ),
    )
    test_files = (
        "tools/test_exact_burst_one_phase_lease_local_journal_gate.py",
        "tools/test_exact_burst_one_phase_lease_local_journal_verify.py",
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
        + " ".join(
            shlex.quote(path)
            for path in compile_files
        )
        + "; "
        + f"PYTHONPATH={pytest_path} "
        + f"{REMOTE_PYTHON} -m pytest -q "
        + " ".join(
            shlex.quote(path)
            for path in test_files
        )
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
    worker = [
        REMOTE_PYTHON,
        (
            "tools/"
            "exact_burst_one_phase_lease_local_journal_gate.py"
        ),
        "--run-dir",
        primary,
        "--hardware",
        "--model",
        MODEL_PATH,
        "--source-sha",
        source_commit,
        "--run-tag",
        run_tag,
    ]
    inner = (
        "set +e; umask 077; pid=$$; "
        f"printf '%s\\n' \"$pid\" > "
        + shlex.quote(pid_path + ".pending")
        + "; "
        f"printf '%s\\n' \"$pid\" > "
        + shlex.quote(pgid_path + ".pending")
        + "; "
        f"mv {shlex.quote(pid_path + '.pending')} "
        + shlex.quote(pid_path)
        + "; "
        f"mv {shlex.quote(pgid_path + '.pending')} "
        + shlex.quote(pgid_path)
        + "; "
        f"cd {shlex.quote(source)}; "
        + remote_runtime_prelude(
            source=source,
            gpu_index=gpu_index,
            dist_port=dist_port,
        )
        + " ".join(
            shlex.quote(part)
            for part in worker
        )
        + "; code=$?; "
        + f"printf '%s\\n' \"$code\" > "
        + shlex.quote(exit_path)
        + "; exit \"$code\""
    )
    launch = (
        "set -eu; "
        f"pid_path={shlex.quote(pid_path)}; "
        f"pgid_path={shlex.quote(pgid_path)}; "
        f"lock_path={shlex.quote(lock_path)}; "
        "if test -s \"$pid_path\" && "
        "test -s \"$pgid_path\"; then "
        "pid=$(cat \"$pid_path\"); "
        "test \"$pid\" = \"$(cat \"$pgid_path\")\"; "
        "printf '%s\\n' \"$pid\"; exit 0; fi; "
        "if mkdir \"$lock_path\" 2>/dev/null; then "
        f"setsid sh -c {shlex.quote(inner)} "
        f"> {shlex.quote(controller + '/worker.stdout.log')} "
        f"2> {shlex.quote(controller + '/worker.stderr.log')} "
        "< /dev/null & fi; "
        "attempt=0; "
        "while ! test -s \"$pid_path\" || "
        "! test -s \"$pgid_path\"; do "
        "attempt=$((attempt + 1)); "
        "test \"$attempt\" -lt 10 || exit 75; "
        "sleep 1; done; "
        "rmdir \"$lock_path\" 2>/dev/null || true; "
        "pid=$(cat \"$pid_path\"); "
        "test \"$pid\" = \"$(cat \"$pgid_path\")\"; "
        "printf '%s\\n' \"$pid\""
    )
    result = _run_remote_checked(
        launch,
        context="launch remote benchmark worker",
        retry_attempts=3,
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
    worker_pid: int,
    poll_interval_seconds: int,
) -> int:
    if (
        isinstance(worker_pid, bool)
        or not isinstance(worker_pid, int)
        or worker_pid <= 0
    ):
        raise ValueError("remote worker PID is invalid")
    consecutive_failures = 0
    while True:
        script = "\n".join((
            "import json,pathlib",
            (
                "exit_path=pathlib.Path("
                f"{(controller + '/worker.exitcode')!r})"
            ),
            (
                "pid_path=pathlib.Path("
                f"{(controller + '/worker.pid')!r})"
            ),
            (
                "pgid_path=pathlib.Path("
                f"{(controller + '/worker.pgid')!r})"
            ),
            "if not pid_path.is_file() or not pgid_path.is_file():",
            " raise ValueError('worker ownership receipt is missing')",
            (
                "if int(pid_path.read_text().strip()) "
                f"!= {worker_pid}:"
            ),
            " raise ValueError('worker PID receipt mismatch')",
            (
                "if int(pgid_path.read_text().strip()) "
                f"!= {worker_pid}:"
            ),
            " raise ValueError('worker PGID receipt mismatch')",
            "if exit_path.is_file():",
            " print(json.dumps({'state':'finished',",
            "  'exitcode':int(exit_path.read_text().strip())}))",
            (
                f"elif pathlib.Path('/proc/{worker_pid}').exists():"
            ),
            " print(json.dumps({'state':'running'}))",
            "else:",
            " print(json.dumps({'state':'missing'}))",
        ))
        try:
            result = _run_remote_checked(
                f"{REMOTE_PYTHON} -c {shlex.quote(script)}",
                context="remote worker polling",
            )
        except RuntimeError:
            consecutive_failures += 1
            if (
                consecutive_failures
                >= MAX_CONSECUTIVE_POLL_FAILURES
            ):
                raise
            time.sleep(poll_interval_seconds)
            continue
        consecutive_failures = 0
        receipt = json.loads(result.stdout)
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
        if receipt == {"state": "missing"}:
            raise RuntimeError(
                "remote worker disappeared before writing exit code"
            )
        if receipt != {"state": "running"}:
            raise ValueError(
                "remote worker poll receipt is invalid"
            )
        time.sleep(poll_interval_seconds)


def _controller_from_primary(primary: str) -> str:
    prefix = TASK_REMOTE_ROOT + "/runs/"
    if not primary.startswith(prefix):
        raise ValueError(
            "remote primary path is invalid"
        )
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
    output = (
        controller
        + "/independent-verify/verification.json"
    )
    command = (
        "set -eu; "
        f"cd {shlex.quote(source)}; "
        + remote_runtime_prelude(
            source=source,
            gpu_index=gpu_index,
            dist_port=dist_port,
        )
        + f"{REMOTE_PYTHON} "
        + (
            "tools/"
            "exact_burst_one_phase_lease_local_journal_verify.py "
        )
        + f"{shlex.quote(primary)} "
        + f"--output {shlex.quote(output)}"
    )
    _run_remote_checked(
        command,
        context="remote independent verification",
        retry_attempts=3,
    )


def _download_terminal_bundle(
    *,
    paths: dict[str, str],
    local_destination: Path,
) -> dict:
    primary = local_destination / "primary"
    controller = local_destination / "controller"
    primary_inventory = (
        download_remote_tree_preserving_partial(
            paths["primary"],
            primary,
        )
    )
    controller_inventory = (
        download_remote_tree_preserving_partial(
            paths["controller"],
            controller,
        )
    )
    missing = [
        name
        for name in PRIMARY_FILES
        if not (primary / name).is_file()
    ]
    if missing:
        raise ValueError(
            "download is incomplete: "
            + ", ".join(sorted(missing))
        )
    remote_receipt_path = (
        controller
        / "independent-verify"
        / "verification.json"
    )
    if not remote_receipt_path.is_file():
        raise ValueError(
            "remote verification receipt is missing"
        )
    remote_verification = json.loads(
        remote_receipt_path.read_text(
            encoding="utf-8"
        )
    )
    local_verification = verify_artifact_directory(primary)
    if (
        remote_verification != local_verification
        or remote_verification.get("verified") is not True
        or remote_verification.get("performance_row_count")
        != PERFORMANCE_ROWS
        or remote_verification.get("correctness_row_count")
        != CORRECTNESS_ROWS
    ):
        raise ValueError(
            "verification receipt disagreement"
        )
    _write_json(
        local_destination
        / "local_verify"
        / "verification.json",
        local_verification,
    )
    download_receipt = {
        "schema": (
            "exact_burst_one_phase_lease_local_"
            "journal_remote_v1"
        ),
        "status": "DOWNLOADED_AND_VERIFIED",
        "primary_inventory": primary_inventory,
        "controller_inventory": controller_inventory,
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
        raise ValueError(
            "Kerberos cache is not authorized"
        )
    if args.control_path != DEFAULT_CONTROL_PATH:
        raise ValueError(
            "SSH control path is not authorized"
        )
    os.environ["TINYLLMFORGE_SSH_CONTROL_PATH"] = (
        args.control_path
    )
    os.environ["KRB5CCNAME"] = args.kerberos_cache

    local_destination = Path(args.local_output_dir)
    if (
        local_destination.exists()
        or local_destination.is_symlink()
    ):
        raise ValueError("local run tag already exists")
    _require_task_tracked_diff_clean(REPO_ROOT)
    pushed_head = base.require_pushed_head(REPO_ROOT)
    source_commit = validate_source_commit(
        args.source_sha,
        pushed_head=pushed_head,
    )
    kerberos = validate_kerberos(
        minimum_lifetime_seconds=(
            MINIMUM_KERBEROS_LIFETIME_SECONDS
        )
    )
    requirements = _probe_remote_requirements()
    paths = remote_paths(args.run_tag)
    dist_port = dist_port_for_run_tag(args.run_tag)
    _require_remote_destinations_absent(paths)

    local_destination.mkdir(
        parents=True,
        exist_ok=False,
    )
    manifest = {
        "schema": (
            "exact_burst_one_phase_lease_local_"
            "journal_remote_v1"
        ),
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
    archive = committed_archive(
        REPO_ROOT,
        source_commit,
    )
    source = _upload_source_archive(
        staging=paths["staging"],
        archive=archive,
    )
    validate_kerberos(
        minimum_lifetime_seconds=(
            MINIMUM_KERBEROS_LIFETIME_SECONDS
        )
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
        "schema": (
            "exact_burst_one_phase_lease_local_"
            "journal_remote_v1"
        ),
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
        poll_interval_seconds=(
            args.gpu_poll_interval_seconds
        ),
    )
    if exitcode != 0:
        raise RuntimeError(
            "remote benchmark worker failed "
            f"with exit code {exitcode}"
        )
    _run_remote_verifier(
        source=source,
        primary=paths["primary"],
        gpu_index=selected["index"],
        dist_port=dist_port,
    )
    completion = {
        "schema": (
            "exact_burst_one_phase_lease_local_"
            "journal_remote_v1"
        ),
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
    downloaded = _download_terminal_bundle(
        paths=paths,
        local_destination=local_destination,
    )
    result = {
        **completion,
        "local_destination": os.fspath(
            local_destination
        ),
        "local_verification": downloaded[
            "local_verification"
        ],
    }
    _write_json(
        local_destination / "controller_state.json",
        result,
    )
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=REMOTE_HOST)
    parser.add_argument(
        "--control-path",
        default=DEFAULT_CONTROL_PATH,
    )
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
    parser.add_argument(
        "--local-output-dir",
        required=True,
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
