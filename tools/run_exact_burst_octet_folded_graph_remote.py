#!/usr/bin/env python3
"""Run the octet-folded exact-burst ceiling on a strict-clean A100."""

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

from tools import run_phase_stitched_exact_graph_remote as transport
from tools import run_staged_inference_benchmark_remote as base
from tools import run_zero_temperature_greedy_fast_path_remote as download
from tools.exact_burst_octet_folded_graph_verify import (
    verify_artifact_directory,
)


REMOTE_HOST = "sitian@10.232.195.203"
REMOTE_PYTHON = "/data00/home/sitian/tllm/env/bin/python"
MODEL_PATH = "/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B"
KRB5_CACHE = "FILE:/Users/bytedance/krb5cc_sitian"
APPROVED_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
TASK_REMOTE_ROOT = (
    APPROVED_ROOT + "/exact-burst-octet-folded-graph"
)
LOCAL_ARTIFACT_ROOT = (
    REPO_ROOT / "artifacts" / "exact_burst_octet_folded_graph"
)
MINIMUM_KERBEROS_LIFETIME_SECONDS = 5_400
MINIMUM_REMOTE_FREE_BYTES = 20 * 1024**3
EMPTY_PATCH_SHA256 = hashlib.sha256(b"").hexdigest()

validate_kerberos = base.validate_kerberos
require_pushed_head = base.require_pushed_head
require_remote_destinations_absent = base.require_remote_destinations_absent
query_remote_gpu_rows = base.query_remote_gpu_rows
establish_ssh_control_master = transport.establish_ssh_control_master


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
        raise ValueError("remote path escaped approved task root")
    return paths


def ensure_local_destination_absent(
    local_root: Path,
    run_tag: str,
) -> Path:
    destination = Path(local_root) / base.validate_run_tag(run_tag)
    if destination.exists() or destination.is_symlink():
        raise ValueError("local run tag already exists")
    return destination


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


def strict_clean_a100s(rows: list[dict]) -> list[dict]:
    clean = []
    seen_indices = set()
    seen_uuids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("GPU row must be an object")
        index = row.get("index")
        uuid = row.get("uuid")
        name = row.get("name")
        memory = row.get("memory_used_mib")
        utilization = row.get("utilization_percent")
        processes = row.get("compute_processes")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or not isinstance(uuid, str)
            or not uuid
            or not isinstance(name, str)
            or isinstance(memory, bool)
            or not isinstance(memory, int)
            or memory < 0
            or isinstance(utilization, bool)
            or not isinstance(utilization, int)
            or utilization < 0
            or not isinstance(processes, list)
        ):
            raise ValueError("GPU row is invalid")
        if index in seen_indices or uuid in seen_uuids:
            raise ValueError("GPU inventory contains duplicate identities")
        seen_indices.add(index)
        seen_uuids.add(uuid)
        if (
            "A100" in name
            and memory <= 1024
            and utilization <= 5
            and not processes
        ):
            clean.append(row)
    return clean


def wait_for_clean_a100(
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
        rows = query_remote_gpu_rows()
        clean = strict_clean_a100s(rows)
        if clean:
            return rows, clean[0]
        if time.monotonic() >= deadline:
            raise TimeoutError("no strict-clean A100 became available")
        time.sleep(poll_interval_seconds)


def validate_selected_gpu_still_clean(selected: dict) -> dict:
    latest = {
        row["index"]: row for row in query_remote_gpu_rows()
    }.get(selected.get("index"))
    if (
        latest is None
        or latest.get("uuid") != selected.get("uuid")
        or strict_clean_a100s([latest]) != [latest]
    ):
        raise RuntimeError("selected A100 is no longer strict-clean")
    return latest


def committed_source_archive(
    repo_root: Path,
    source_commit: str,
) -> bytes:
    result = subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            "--prefix=source/",
            source_commit,
            "--",
            "tinyvllm",
            "tools",
        ],
        cwd=Path(repo_root),
        capture_output=True,
        check=False,
    )
    base._require_success(result, "build committed source archive")
    if not result.stdout:
        raise ValueError("committed source archive is empty")
    return result.stdout


def _run_remote_with_input(command: str, payload: bytes):
    return base._run_remote_with_input(command, payload)


def upload_source_archive(*, staging: str, archive: bytes) -> str:
    prefix = TASK_REMOTE_ROOT + "/staging/"
    if (
        not isinstance(staging, str)
        or not staging.startswith(prefix)
        or base.validate_run_tag(staging[len(prefix):])
        != staging[len(prefix):]
    ):
        raise ValueError("remote staging path is invalid")
    if not isinstance(archive, bytes) or not archive:
        raise ValueError("source archive is empty")
    script = "\n".join((
        "import pathlib,sys,tarfile",
        f"staging=pathlib.Path({staging!r})",
        "staging.parent.mkdir(parents=True,exist_ok=True)",
        "staging.mkdir(parents=False,exist_ok=False)",
        "archive_path=staging/'source.tar'",
        "with archive_path.open('xb') as handle:",
        " handle.write(sys.stdin.buffer.read())",
        "with tarfile.open(archive_path,'r:') as bundle:",
        " members=bundle.getmembers()",
        " if not members:",
        "  raise ValueError('empty source archive')",
        " for member in members:",
        "  path=pathlib.PurePosixPath(member.name)",
        "  if (path.is_absolute() or '..' in path.parts",
        "      or not path.parts or path.parts[0] != 'source'",
        "      or member.issym() or member.islnk()):",
        "   raise ValueError('unsafe source archive member')",
        " bundle.extractall(staging)",
    ))
    result = _run_remote_with_input(
        f"python3 -c {shlex.quote(script)}",
        archive,
    )
    base._require_success(result, "upload committed source archive")
    return staging + "/source"


def remote_runtime_prelude(
    *,
    source: str,
    run_root: str,
    gpu_index: int,
) -> str:
    if (
        not source.startswith(TASK_REMOTE_ROOT + "/")
        or not run_root.startswith(TASK_REMOTE_ROOT + "/")
        or isinstance(gpu_index, bool)
        or not isinstance(gpu_index, int)
        or gpu_index < 0
    ):
        raise ValueError("remote runtime configuration is invalid")
    runtime = run_root + "/runtime"
    paths = {
        "TMPDIR": runtime + "/tmp",
        "TMP": runtime + "/tmp",
        "TEMP": runtime + "/tmp",
        "PYTHONPYCACHEPREFIX": runtime + "/pycache",
        "XDG_CACHE_HOME": runtime + "/xdg-cache",
        "HF_HOME": runtime + "/hf-cache",
        "TORCH_EXTENSIONS_DIR": runtime + "/torch-extensions",
    }
    directories = sorted(set(paths.values()))
    exports = [
        f"export {name}={shlex.quote(value)}"
        for name, value in paths.items()
    ]
    exports.extend((
        f"export CUDA_VISIBLE_DEVICES={gpu_index}",
        f"export PYTHONPATH={shlex.quote(source)}",
        "export PYTHONNOUSERSITE=1",
        "export PYTHONDONTWRITEBYTECODE=1",
    ))
    return (
        "mkdir -p "
        + " ".join(shlex.quote(path) for path in directories)
        + " || exit $?; "
        + "; ".join(exports)
        + "; "
    )


def probe_remote_requirements() -> dict:
    script = "\n".join((
        "import json,os",
        f"python_path={REMOTE_PYTHON!r}",
        f"model_path={MODEL_PATH!r}",
        f"approved_root={APPROVED_ROOT!r}",
        "stat=os.statvfs(approved_root)",
        "payload={",
        " 'python':{",
        "  'path':python_path,",
        "  'is_file':os.path.isfile(python_path),",
        "  'is_executable':os.access(python_path,os.X_OK),",
        " },",
        " 'model':{",
        "  'path':model_path,",
        "  'is_dir':os.path.isdir(model_path),",
        "  'config_path':os.path.join(model_path,'config.json'),",
        "  'config_is_file':os.path.isfile(",
        "   os.path.join(model_path,'config.json')),",
        " },",
        " 'approved_root':{",
        "  'path':approved_root,",
        "  'is_dir':os.path.isdir(approved_root),",
        "  'free_bytes':stat.f_bavail*stat.f_frsize,",
        " },",
        "}",
        "print(json.dumps(payload,sort_keys=True,separators=(',',':')))",
    ))
    result = _run_remote("python3 -c " + shlex.quote(script))
    base._require_success(result, "remote requirement probe")
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("remote requirement receipt is invalid") from error
    python = payload.get("python") if isinstance(payload, dict) else None
    model = payload.get("model") if isinstance(payload, dict) else None
    root = (
        payload.get("approved_root")
        if isinstance(payload, dict)
        else None
    )
    if (
        not isinstance(python, dict)
        or python.get("path") != REMOTE_PYTHON
        or python.get("is_file") is not True
        or python.get("is_executable") is not True
        or not isinstance(model, dict)
        or model.get("path") != MODEL_PATH
        or model.get("is_dir") is not True
        or model.get("config_path") != MODEL_PATH + "/config.json"
        or model.get("config_is_file") is not True
        or not isinstance(root, dict)
        or root.get("path") != APPROVED_ROOT
        or root.get("is_dir") is not True
        or isinstance(root.get("free_bytes"), bool)
        or not isinstance(root.get("free_bytes"), int)
        or root["free_bytes"] < MINIMUM_REMOTE_FREE_BYTES
    ):
        raise ValueError("remote requirements are not satisfied")
    return payload


def remote_execution_commands(
    *,
    source: str,
    primary: str,
    controller: str,
    model: str,
    gpu_index: int,
    run_tag: str,
    source_commit: str,
    source_patch_sha256: str,
) -> tuple[str, str]:
    for path in (source, primary, controller, model):
        if not isinstance(path, str) or not path.startswith(
            "/data00/home/sitian/"
        ):
            raise ValueError("remote execution path is not approved")
    validate_source_commit(
        source_commit,
        pushed_head=source_commit,
    )
    if source_patch_sha256 != EMPTY_PATCH_SHA256:
        raise ValueError("ceiling source patch must be empty")
    prelude = remote_runtime_prelude(
        source=source,
        run_root=source.rsplit("/", 1)[0],
        gpu_index=gpu_index,
    )
    producer = " ".join((
        shlex.quote(REMOTE_PYTHON),
        "-m",
        "tools.profile_exact_burst_octet_folded_graph",
        "--model",
        shlex.quote(model),
        "--device",
        "cuda:0",
        "--output-dir",
        shlex.quote(primary),
        "--source-commit",
        source_commit,
        "--source-patch-sha256",
        source_patch_sha256,
        "--run-tag",
        shlex.quote(run_tag),
    ))
    verifier = " ".join((
        shlex.quote(REMOTE_PYTHON),
        "-m",
        "tools.exact_burst_octet_folded_graph_verify",
        "--run-dir",
        shlex.quote(primary),
        "--source-root",
        shlex.quote(source),
        "--output",
        shlex.quote(controller + "/remote-verification.json"),
    ))
    producer_command = (
        "set +e; "
        + f"mkdir -p {shlex.quote(controller)}; "
        + "worker_pid=$$; "
        + "worker_pgid=$(ps -o pgid= -p \"$worker_pid\" "
        + "| tr -d ' '); "
        + f"printf '%s\\n' \"$worker_pid\" > "
        + shlex.quote(controller + "/producer_worker_pid")
        + "; "
        + f"printf '%s\\n' \"$worker_pgid\" > "
        + shlex.quote(controller + "/producer_worker_pgid")
        + "; "
        + f"cd {shlex.quote(source)}; "
        + prelude
        + producer
        + f" > {shlex.quote(controller + '/producer.stdout')} "
        + f"2> {shlex.quote(controller + '/producer.stderr')}; "
        + "code=$?; "
        + f"printf '%s\\n' \"$code\" > "
        + shlex.quote(controller + "/producer_exitcode")
        + "; if [ \"$code\" -eq 0 ]; then cp "
        + shlex.quote(source.rsplit("/", 1)[0] + "/source.patch")
        + " "
        + shlex.quote(primary + "/source.patch")
        + '; fi; exit "$code"'
    )
    verifier_command = (
        "set +e; "
        + "worker_pid=$$; "
        + "worker_pgid=$(ps -o pgid= -p \"$worker_pid\" "
        + "| tr -d ' '); "
        + f"printf '%s\\n' \"$worker_pid\" > "
        + shlex.quote(controller + "/verifier_worker_pid")
        + "; "
        + f"printf '%s\\n' \"$worker_pgid\" > "
        + shlex.quote(controller + "/verifier_worker_pgid")
        + "; "
        + f"cd {shlex.quote(source)}; "
        + prelude
        + verifier
        + f" > {shlex.quote(controller + '/verifier.stdout')} "
        + f"2> {shlex.quote(controller + '/verifier.stderr')}; "
        + "code=$?; "
        + f"printf '%s\\n' \"$code\" > "
        + shlex.quote(controller + "/verifier_exitcode")
        + '; exit "$code"'
    )
    return producer_command, verifier_command


def _run_remote(command: str):
    return base._run_remote(command)


def _prepare_remote_source(
    *,
    paths: dict[str, str],
    archive: bytes,
) -> str:
    source = upload_source_archive(
        staging=paths["staging"],
        archive=archive,
    )
    result = base._run_remote(
        "set -e; "
        + f": > {shlex.quote(paths['staging'] + '/source.patch')}"
    )
    base._require_success(result, "create immutable empty source patch")
    return source


def run_remote_preflight(*, source: str, gpu_index: int) -> None:
    smoke = "; ".join((
        "from tools import profile_exact_burst_octet_folded_graph as p",
        "from tools import exact_burst_octet_folded_graph_verify as v",
        "assert p.REPETITIONS == 5",
        "assert v.REPETITIONS == 5",
    ))
    command = (
        "set -eu; "
        + f"cd {shlex.quote(source)}; "
        + remote_runtime_prelude(
            source=source,
            run_root=source.rsplit("/", 1)[0],
            gpu_index=gpu_index,
        )
        + f"test -x {shlex.quote(REMOTE_PYTHON)}; "
        + f"test -f {shlex.quote(MODEL_PATH + '/config.json')}; "
        + f"{shlex.quote(REMOTE_PYTHON)} -m compileall -q "
        + "tinyvllm tools; "
        + f"{shlex.quote(REMOTE_PYTHON)} -c {shlex.quote(smoke)}"
    )
    base._require_success(_run_remote(command), "remote source preflight")


def create_remote_controller_manifest(
    *,
    paths: dict[str, str],
    run_tag: str,
    source_commit: str,
    source: str,
    gpu_inventory: list[dict],
    selected_gpu: dict,
    kerberos_receipt: dict,
    remote_requirements: dict,
) -> None:
    payload = {
        "schema_version":
            "exact-burst-octet-folded.controller.v1",
        "run_tag": base.validate_run_tag(run_tag),
        "source_commit": validate_source_commit(
            source_commit,
            pushed_head=source_commit,
        ),
        "source_patch_sha256": EMPTY_PATCH_SHA256,
        "source": source,
        "remote_paths": paths,
        "model": MODEL_PATH,
        "python": REMOTE_PYTHON,
        "gpu_inventory": gpu_inventory,
        "selected_gpu": selected_gpu,
        "kerberos": kerberos_receipt,
        "remote_requirements": remote_requirements,
    }
    if (
        set(paths) != {"staging", "primary", "controller"}
        or not source.startswith(paths["staging"] + "/")
        or any(
            not path.startswith(TASK_REMOTE_ROOT + "/")
            for path in paths.values()
        )
    ):
        raise ValueError("controller manifest paths are invalid")
    script = "\n".join((
        "import json,pathlib,sys",
        "payload=json.load(sys.stdin)",
        f"controller=pathlib.Path({paths['controller']!r})",
        "controller.parent.mkdir(parents=True,exist_ok=True)",
        "controller.mkdir(parents=False,exist_ok=False)",
        "(controller/'controller_manifest.json').write_text(",
        " json.dumps(payload,sort_keys=True)+'\\n',encoding='utf-8')",
    ))
    result = _run_remote_with_input(
        f"python3 -c {shlex.quote(script)}",
        json.dumps(payload, allow_nan=False).encode("utf-8"),
    )
    base._require_success(result, "create remote controller manifest")


def run_remote_pipeline(
    *,
    source: str,
    paths: dict[str, str],
    model: str,
    gpu_index: int,
    run_tag: str,
    source_commit: str,
) -> dict:
    commands = remote_execution_commands(
        source=source,
        primary=paths["primary"],
        controller=paths["controller"],
        model=model,
        gpu_index=gpu_index,
        run_tag=run_tag,
        source_commit=source_commit,
        source_patch_sha256=EMPTY_PATCH_SHA256,
    )
    exitcodes = []
    for index, command in enumerate(commands):
        result = _run_remote(command)
        exitcodes.append(result.returncode)
        base._require_success(result, f"remote pipeline stage {index}")
    return {
        "producer_exitcode": exitcodes[0],
        "verifier_exitcode": exitcodes[1],
    }


def download_bundle(
    *,
    paths: dict[str, str],
    local_destination: Path,
) -> dict:
    destination = Path(local_destination)
    download.download_remote_tree_preserving_partial(
        paths["primary"],
        destination,
    )
    download.download_remote_tree_preserving_partial(
        paths["controller"],
        destination / "controller",
    )
    for name in ("producer_exitcode", "verifier_exitcode"):
        if (
            destination / "controller" / name
        ).read_text(encoding="utf-8").strip() != "0":
            raise ValueError(f"{name} is not zero")
    for stage in ("producer", "verifier"):
        for kind in ("pid", "pgid"):
            name = f"{stage}_worker_{kind}"
            path = destination / "controller" / name
            try:
                value = int(path.read_text(encoding="utf-8").strip())
            except (FileNotFoundError, ValueError) as error:
                raise ValueError(
                    f"{name} worker receipt is invalid"
                ) from error
            if value <= 0:
                raise ValueError(f"{name} worker receipt is invalid")
    remote_receipt = json.loads(
        (
            destination
            / "controller"
            / "remote-verification.json"
        ).read_text(encoding="utf-8")
    )
    try:
        controller_manifest = json.loads(
            (
                destination
                / "controller"
                / "controller_manifest.json"
            ).read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError("controller manifest is invalid") from error
    if (
        not isinstance(controller_manifest, dict)
        or controller_manifest.get("schema_version")
        != "exact-burst-octet-folded.controller.v1"
        or controller_manifest.get("run_tag")
        != remote_receipt.get("run_tag")
        or controller_manifest.get("source_commit")
        != remote_receipt.get("source_commit")
        or controller_manifest.get("source_patch_sha256")
        != remote_receipt.get("source_patch_sha256")
        or controller_manifest.get("remote_paths") != paths
    ):
        raise ValueError("controller manifest is invalid")
    local_receipt = verify_artifact_directory(
        destination,
        source_root=REPO_ROOT,
    )
    if remote_receipt != local_receipt:
        raise ValueError("remote and local verifier receipts disagree")
    return local_receipt


def download_partial_bundle(
    *,
    paths: dict[str, str],
    local_destination: Path,
) -> dict[str, dict]:
    destination = Path(local_destination)
    receipts = {}
    targets = (
        ("primary", paths["primary"], destination),
        (
            "controller",
            paths["controller"],
            destination / "controller",
        ),
    )
    for name, remote_root, local_root in targets:
        try:
            inventory = (
                download.download_remote_tree_preserving_partial(
                    remote_root,
                    local_root,
                )
            )
        except Exception as error:
            receipts[name] = {
                "downloaded": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        else:
            receipts[name] = {
                "downloaded": True,
                "file_count": len(inventory),
            }
    return receipts


def run_controller(args) -> dict:
    if args.host != REMOTE_HOST or args.model != MODEL_PATH:
        raise ValueError("remote target is not approved")
    if os.environ.get("KRB5CCNAME", KRB5_CACHE) != KRB5_CACHE:
        raise ValueError("Kerberos cache path is not approved")
    destination = ensure_local_destination_absent(
        Path(args.local_artifact_root),
        args.run_tag,
    )
    pushed_head = require_pushed_head(REPO_ROOT)
    source_commit = validate_source_commit(
        pushed_head if args.source_commit is None else args.source_commit,
        pushed_head=pushed_head,
    )
    kerberos_receipt = validate_kerberos(
        minimum_lifetime_seconds=(
            args.gpu_wait_timeout_seconds
            + MINIMUM_KERBEROS_LIFETIME_SECONDS
        )
    )
    establish_ssh_control_master()
    remote_requirements = probe_remote_requirements()
    paths = remote_paths(args.run_tag)
    require_remote_destinations_absent(paths)
    gpu_rows, selected = wait_for_clean_a100(
        timeout_seconds=args.gpu_wait_timeout_seconds,
        poll_interval_seconds=args.gpu_poll_interval_seconds,
    )
    if require_pushed_head(REPO_ROOT) != source_commit:
        raise ValueError("pushed head changed while waiting for GPU")
    archive = committed_source_archive(REPO_ROOT, source_commit)
    source = _prepare_remote_source(paths=paths, archive=archive)
    launch_kerberos_receipt = validate_kerberos(
        minimum_lifetime_seconds=MINIMUM_KERBEROS_LIFETIME_SECONDS
    )
    run_remote_preflight(
        source=source,
        gpu_index=selected["index"],
    )
    selected = validate_selected_gpu_still_clean(selected)
    create_remote_controller_manifest(
        paths=paths,
        run_tag=args.run_tag,
        source_commit=source_commit,
        source=source,
        gpu_inventory=gpu_rows,
        selected_gpu=selected,
        kerberos_receipt={
            "initial": kerberos_receipt,
            "launch": launch_kerberos_receipt,
        },
        remote_requirements=remote_requirements,
    )
    try:
        receipts = run_remote_pipeline(
            source=source,
            paths=paths,
            model=args.model,
            gpu_index=selected["index"],
            run_tag=args.run_tag,
            source_commit=source_commit,
        )
    except Exception:
        download_partial_bundle(
            paths=paths,
            local_destination=destination,
        )
        raise
    verification = download_bundle(
        paths=paths,
        local_destination=destination,
    )
    return {
        "status": "COMPLETE",
        "run_tag": args.run_tag,
        "source_commit": source_commit,
        "source_patch_sha256": EMPTY_PATCH_SHA256,
        "remote_paths": paths,
        "gpu_inventory": gpu_rows,
        "selected_gpu": selected,
        "local_destination": os.fspath(destination),
        **receipts,
        **verification,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=REMOTE_HOST)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--source-commit")
    parser.add_argument(
        "--local-artifact-root",
        default=os.fspath(LOCAL_ARTIFACT_ROOT),
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
    os.environ["KRB5CCNAME"] = KRB5_CACHE
    result = run_controller(parse_args(argv))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
